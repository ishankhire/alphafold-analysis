#!/usr/bin/env python3
"""Shared infrastructure for combined CAMEO+ECOD XGBoost searches.

The module deliberately keeps XGBoost as a lazy dependency so split, sampling,
selection, and metric tests can run in lightweight local environments.
"""

from __future__ import annotations

import csv
import gc
import hashlib
import json
import math
import os
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
GENERALIZED_DIR = CLUSTER_DIR / "generalized_model_ecod"
import sys

for _path in (CLUSTER_DIR, GENERALIZED_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cluster_regression_helpers import (  # noqa: E402
    classical_mds,
    compute_distance_matrix,
    compute_tm_score,
    extract_ca_coordinates,
    superimpose_structures,
)
from general_train_test_common import (  # noqa: E402
    build_dataset_protein_manifest,
    resolve_pair_block_path,
    resolve_pdb_file,
)


ANCHOR_LAYERS = (0, 23, 47)
CAPS = (5000, 10000, 20000, None)
CAP_LABELS = ("5k", "10k", "20k", "all")
N_CHANNELS = 128
N_SIZE_BINS = 5
RANDOM_STATE = 42
SEARCH_CAP = 10000
N_ESTIMATORS = 700
EARLY_STOPPING_ROUNDS = 30
SELECTION_TOLERANCE = 0.001
EPS = 1e-12
FIXED_XGBOOST_PARAMETERS = {
    "objective": "reg:squarederror", "tree_method": "hist", "max_bin": 256,
    "n_estimators": N_ESTIMATORS, "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
    "eval_metric": "rmse", "subsample": 0.8, "colsample_bytree": 0.8,
    "reg_lambda": 1.0, "random_state": RANDOM_STATE,
}

SHARED_CONFIGS = (
    {"config_id": "S0", "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5},
    {"config_id": "S1", "max_depth": 4, "learning_rate": 0.10, "min_child_weight": 5},
    {"config_id": "S2", "max_depth": 5, "learning_rate": 0.05, "min_child_weight": 10},
    {"config_id": "S3", "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 10},
    {"config_id": "S4", "max_depth": 6, "learning_rate": 0.10, "min_child_weight": 10},
    {"config_id": "S5", "max_depth": 7, "learning_rate": 0.05, "min_child_weight": 20},
)
PER_PROTEIN_CONFIGS = (
    {"config_id": "P0", "max_depth": 3, "learning_rate": 0.05, "min_child_weight": 5},
    {"config_id": "P1", "max_depth": 4, "learning_rate": 0.05, "min_child_weight": 5},
    {"config_id": "P2", "max_depth": 4, "learning_rate": 0.10, "min_child_weight": 5},
    {"config_id": "P3", "max_depth": 6, "learning_rate": 0.05, "min_child_weight": 10},
)


def timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{timestamp()}] {message}", flush=True)


def output_root() -> Path:
    return Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))) / "xgboost_hyperparameter_search"


def shared_search_dir() -> Path:
    return output_root() / "shared" / "search"


def shared_winner_dir() -> Path:
    return output_root() / "shared" / "winner_cap_sweep"


def per_protein_search_dir() -> Path:
    return output_root() / "per_protein" / "search"


def per_protein_winner_dir() -> Path:
    return output_root() / "per_protein" / "winner_evaluation"


def write_csv(path: Path, rows: Sequence[Mapping], fieldnames: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fieldnames is None:
        fieldnames = list(rows[0].keys()) if rows else []
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise FileNotFoundError(f"Required search artifact is missing: {path}")
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_json(path: Path, payload: Mapping | Sequence) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=True)
        handle.write("\n")


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode()).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def split_digest(assignments: Sequence[Mapping]) -> str:
    canonical = sorted((str(r["protein_id"]), str(r["split"])) for r in assignments)
    return hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def assign_size_bins(records: Sequence[dict], n_bins: int = N_SIZE_BINS) -> None:
    ordered = sorted(records, key=lambda r: (int(r["n_residues"]), str(r["protein_id"])))
    for rank, record in enumerate(ordered):
        record["size_bin"] = min(n_bins - 1, rank * n_bins // len(ordered))


def _stratified_partition(records: Sequence[dict], fractions: tuple[float, float, float], seed: int) -> dict[str, list[dict]]:
    if not records:
        raise ValueError("Cannot split an empty protein set.")
    if not np.allclose(fractions, (0.6, 0.2, 0.2)):
        raise ValueError(f"Only the locked 60/20/20 split is supported; got {fractions}.")
    ordered = sorted(records, key=lambda r: str(r["protein_id"]))
    labels = np.asarray([int(r["size_bin"]) for r in ordered], dtype=np.int64)
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size != N_SIZE_BINS or int(np.min(counts)) < 3:
        raise ValueError(f"All five size bins need at least three proteins; observed {dict(zip(unique, counts))}.")
    indices = np.arange(len(ordered), dtype=np.int64)
    first = StratifiedShuffleSplit(n_splits=1, test_size=0.4, random_state=seed)
    fit_idx, remainder_idx = next(first.split(indices, labels))
    remainder_labels = labels[remainder_idx]
    second = StratifiedShuffleSplit(n_splits=1, test_size=0.5, random_state=seed + 1)
    validation_local, test_local = next(second.split(remainder_idx, remainder_labels))
    validation_idx = remainder_idx[validation_local]
    test_idx = remainder_idx[test_local]
    result = {
        "fit": [ordered[int(i)] for i in fit_idx],
        "validation": [ordered[int(i)] for i in validation_idx],
        "test": [ordered[int(i)] for i in test_idx],
    }
    for split_records in result.values():
        split_records.sort(key=lambda r: r["protein_id"])
    return result


def build_shared_split(records: Sequence[dict], seed: int = RANDOM_STATE):
    """Return a deterministic, disjoint, size-stratified 60/20/20 split."""
    parts = _stratified_partition(records, (0.6, 0.2, 0.2), seed)
    assignments = []
    for split_name, split_records in parts.items():
        for r in split_records:
            assignments.append({
                "protein_id": r["protein_id"], "source": r.get("source", ""),
                "n_residues": int(r["n_residues"]), "n_pairs": int(r["n_pairs"]),
                "size_bin": int(r["size_bin"]), "split": split_name,
            })
    if len({r["protein_id"] for r in assignments}) != len(records):
        raise AssertionError("Shared split is not disjoint and exhaustive.")
    return parts, assignments


def pair_split_indices(n_pairs: int, protein_id: str, seed: int = RANDOM_STATE) -> dict[str, np.ndarray]:
    if n_pairs < 3:
        raise ValueError(f"Protein {protein_id} has fewer than three residue pairs.")
    order = np.random.default_rng(stable_seed(seed, "pair_split", protein_id)).permutation(n_pairs)
    n_fit = int(math.floor(0.6 * n_pairs))
    n_val = int(math.floor(0.2 * n_pairs))
    n_fit = max(1, n_fit)
    n_val = max(1, n_val)
    if n_fit + n_val >= n_pairs:
        n_fit, n_val = n_pairs - 2, 1
    return {
        "fit": np.sort(order[:n_fit]),
        "validation": np.sort(order[n_fit:n_fit + n_val]),
        "test": np.sort(order[n_fit + n_val:]),
    }


def nested_sample_indices(indices: np.ndarray, protein_id: str, cap: int | None, seed: int = RANDOM_STATE) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64)
    order = np.random.default_rng(stable_seed(seed, "nested_fit", protein_id)).permutation(indices.size)
    take = indices.size if cap is None else min(int(cap), indices.size)
    return indices[order[:take]]


def protein_equal_weights(protein_ids: Sequence[str]) -> np.ndarray:
    ids = np.asarray(protein_ids)
    weights = np.empty(ids.size, dtype=np.float64)
    unique, counts = np.unique(ids, return_counts=True)
    for protein_id, count in zip(unique, counts):
        weights[ids == protein_id] = 1.0 / float(count)
    return weights


def map_search_task(task_id: int) -> int:
    try:
        return ANCHOR_LAYERS[int(task_id)]
    except (IndexError, ValueError):
        raise ValueError(f"Search task ID must be 0..2; got {task_id}.") from None


def map_shared_winner_task(task_id: int) -> tuple[int, int | None, str]:
    if not 0 <= int(task_id) < 12:
        raise ValueError(f"Shared winner task ID must be 0..11; got {task_id}.")
    task_id = int(task_id)
    layer_idx, cap_idx = divmod(task_id, 4)
    return ANCHOR_LAYERS[layer_idx], CAPS[cap_idx], CAP_LABELS[cap_idx]


def map_winner_layer_task(task_id: int) -> int:
    return map_search_task(task_id)


def averaged_symmetric_features(pair_block: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    pb = np.asarray(pair_block)
    if pb.ndim != 3 or pb.shape[0] != pb.shape[1] or pb.shape[2] != N_CHANNELS:
        raise ValueError(f"Expected pair block shape (R,R,{N_CHANNELS}); got {pb.shape}.")
    left = np.asarray(pb[pair_i, pair_j, :], dtype=np.float64)
    right = np.asarray(pb[pair_j, pair_i, :], dtype=np.float64)
    features = 0.5 * (left + right)
    if features.dtype != np.float64 or features.ndim != 2 or features.shape[1] != N_CHANNELS:
        raise AssertionError("Feature construction must produce exactly 128 float64 channels.")
    return features


def load_pair_features(record: Mapping, layer: int, indices: np.ndarray | None = None) -> np.ndarray:
    path = resolve_pair_block_path(Path(record["protein_dir"]), str(record["protein_id"]), int(layer))
    if not path.exists():
        raise FileNotFoundError(f"Required layer {layer} pair block is missing for {record['protein_id']}: {path}")
    pb = np.load(path)
    pair_i = np.asarray(record["pair_i"])
    pair_j = np.asarray(record["pair_j"])
    if indices is not None:
        pair_i, pair_j = pair_i[indices], pair_j[indices]
    features = averaged_symmetric_features(pb, pair_i, pair_j)
    del pb
    return features


def load_combined_records(require_layers: Sequence[int] = ANCHOR_LAYERS):
    records, skipped = [], []
    manifest = build_dataset_protein_manifest()
    for entry in manifest:
        protein_id = entry["protein_id"]
        try:
            missing = [layer for layer in require_layers if not resolve_pair_block_path(entry["protein_dir"], protein_id, layer).exists()]
            if missing:
                skipped.append({"protein_id": protein_id, "source": entry["source"], "reason": f"missing pair-block layers {missing}"})
                continue
            pdb_path = resolve_pdb_file(protein_id, entry["protein_dir"], entry["pdb_root"])
            if pdb_path is None:
                skipped.append({"protein_id": protein_id, "source": entry["source"], "reason": "missing predicted PDB"})
                continue
            ca = extract_ca_coordinates(pdb_path)
            coords = np.asarray(ca["coordinates"], dtype=np.float64)
            if coords.shape[0] < 3:
                skipped.append({"protein_id": protein_id, "source": entry["source"], "reason": "fewer than 3 CA atoms"})
                continue
            shape = np.load(resolve_pair_block_path(entry["protein_dir"], protein_id, require_layers[0]), mmap_mode="r").shape
            if shape != (coords.shape[0], coords.shape[0], N_CHANNELS):
                skipped.append({"protein_id": protein_id, "source": entry["source"], "reason": f"PDB/pair-block shape mismatch: {coords.shape[0]} CA vs {shape}"})
                continue
            pair_i, pair_j = np.triu_indices(coords.shape[0], k=1)
            y = np.asarray(compute_distance_matrix(coords)[pair_i, pair_j], dtype=np.float64)
            records.append({
                **entry, "pdb_path": pdb_path, "coordinates": coords,
                "n_residues": int(coords.shape[0]), "pair_i": pair_i, "pair_j": pair_j,
                "y": y, "n_pairs": int(y.size), "target_variance": float(np.var(y, dtype=np.float64)),
            })
        except Exception as exc:
            skipped.append({"protein_id": protein_id, "source": entry["source"], "reason": f"load error: {exc}"})
    if not records:
        raise RuntimeError("No valid proteins were found in the fixed combined CAMEO+ECOD manifest.")
    assign_size_bins(records)
    return records, skipped


def select_per_protein_manifest(records: Sequence[dict], seed: int = RANDOM_STATE) -> list[dict]:
    selected = []
    for size_bin in range(N_SIZE_BINS):
        group = sorted((r for r in records if int(r["size_bin"]) == size_bin), key=lambda r: r["protein_id"])
        if len(group) < 10:
            raise ValueError(f"Size bin {size_bin} has only {len(group)} valid proteins; exactly 10 are required.")
        order = np.random.default_rng(stable_seed(seed, "manifest", size_bin)).permutation(len(group))
        selected.extend(group[int(i)] for i in order[:10])
    return sorted(selected, key=lambda r: (int(r["size_bin"]), r["protein_id"]))


def materialize_records(records: Sequence[dict], layer: int, cap: int | None = None, split_name: str | None = None):
    chunks_x, chunks_y, row_ids, offsets = [], [], [], []
    start = 0
    elapsed = 0.0
    for record in records:
        if split_name is None:
            base_indices = np.arange(record["n_pairs"], dtype=np.int64)
        else:
            base_indices = record["pair_splits"][split_name]
        indices = nested_sample_indices(base_indices, record["protein_id"], cap) if cap is not None else base_indices
        tick = time.perf_counter()
        x = load_pair_features(record, layer, indices)
        elapsed += time.perf_counter() - tick
        y = np.asarray(record["y"][indices], dtype=np.float64)
        end = start + y.size
        offsets.append((record, indices, start, end))
        chunks_x.append(x)
        chunks_y.append(y)
        row_ids.extend([record["protein_id"]] * y.size)
        start = end
    x_all = np.concatenate(chunks_x, axis=0) if chunks_x else np.empty((0, N_CHANNELS), dtype=np.float64)
    y_all = np.concatenate(chunks_y) if chunks_y else np.empty(0, dtype=np.float64)
    if x_all.dtype != np.float64 or x_all.shape[1] != N_CHANNELS:
        raise AssertionError("Materialized XGBoost features must be unstandardized float64 (n,128).")
    return {"x": x_all, "y": y_all, "protein_ids": row_ids, "offsets": offsets, "materialization_seconds": elapsed}


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2, dtype=np.float64))
    return float("nan") if denominator <= EPS else 1.0 - float(np.sum((y_true - y_pred) ** 2, dtype=np.float64)) / denominator


def metric_row(y_true, y_pred) -> dict[str, float]:
    err = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return {"r2": r2_score(y_true, y_pred), "mae": float(np.mean(np.abs(err))), "rmse": float(np.sqrt(np.mean(err * err)))}


@dataclass
class PooledStats:
    n: int = 0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0

    def update(self, y_true, y_pred) -> None:
        y = np.asarray(y_true, dtype=np.float64)
        err = np.asarray(y_pred, dtype=np.float64) - y
        self.n += int(y.size)
        self.sum_y += float(np.sum(y, dtype=np.float64))
        self.sum_y2 += float(np.sum(y * y, dtype=np.float64))
        self.sum_abs_error += float(np.sum(np.abs(err), dtype=np.float64))
        self.sum_squared_error += float(np.sum(err * err, dtype=np.float64))

    def metrics(self) -> dict[str, float]:
        if self.n == 0:
            return {"r2": float("nan"), "mae": float("nan"), "rmse": float("nan")}
        sst = self.sum_y2 - self.sum_y * self.sum_y / self.n
        return {
            "r2": float("nan") if sst <= EPS else 1.0 - self.sum_squared_error / sst,
            "mae": self.sum_abs_error / self.n,
            "rmse": math.sqrt(self.sum_squared_error / self.n),
        }


def summarize_prediction_bundle(bundle: Mapping, predictions: np.ndarray):
    rows, pooled = [], PooledStats()
    for record, indices, start, end in bundle["offsets"]:
        y = bundle["y"][start:end]
        pred = predictions[start:end]
        metrics = metric_row(y, pred)
        pooled.update(y, pred)
        rows.append({"protein_id": record["protein_id"], "source": record["source"], "n_residues": record["n_residues"], "n_pairs": int(len(indices)), **metrics})
    finite = [r for r in rows if np.isfinite(r["r2"])]
    return rows, {"mean_per_protein_r2": float(np.mean([r["r2"] for r in finite])), **{f"pooled_{k}": v for k, v in pooled.metrics().items()}}


def geometry_metrics(record: Mapping, predictions: np.ndarray) -> dict[str, float | int]:
    n = int(record["n_residues"])
    pred = np.asarray(predictions, dtype=np.float64)
    n_negative = int(np.count_nonzero(pred < 0.0))
    dist = np.zeros((n, n), dtype=np.float64)
    clipped = np.maximum(pred, 0.0)
    dist[record["pair_i"], record["pair_j"]] = clipped
    dist[record["pair_j"], record["pair_i"]] = clipped
    mds_coords, eigenvalues = classical_mds(dist, n_dims=3)
    reflected = mds_coords.copy()
    reflected[:, 2] *= -1.0
    true_coords = np.asarray(record["coordinates"], dtype=np.float64)
    aligned, rmsd = superimpose_structures(true_coords, mds_coords)
    aligned_reflected, rmsd_reflected = superimpose_structures(true_coords, reflected)
    if rmsd_reflected < rmsd:
        aligned, rmsd = aligned_reflected, rmsd_reflected
    positive = eigenvalues[eigenvalues > 0]
    positive_mass = float(np.sum(positive, dtype=np.float64))
    frac_top3 = float(np.sum(np.maximum(eigenvalues[:3], 0.0)) / positive_mass) if positive_mass > 0 else float("nan")
    negative = eigenvalues[eigenvalues < 0]
    negative_fraction = float(abs(np.sum(negative)) / positive_mass) if positive_mass > 0 else float("nan")
    mds_dist = compute_distance_matrix(aligned)
    mds_pairs = mds_dist[record["pair_i"], record["pair_j"]]
    denominator = float(np.sum(clipped * clipped, dtype=np.float64))
    stress = float(np.sqrt(np.sum((clipped - mds_pairs) ** 2) / denominator)) if denominator > 0 else float("nan")
    return {
        "tm_score": compute_tm_score(true_coords, aligned, n), "kabsch_rmsd": float(rmsd),
        "mds_stress": stress, "top3_positive_spectral_mass_fraction": frac_top3,
        "negative_eigenvalue_count": int(negative.size), "negative_eigenvalue_fraction": negative_fraction,
        "negative_prediction_count": n_negative,
    }


def require_xgboost():
    try:
        from xgboost import XGBRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "XGBoost is required for this job. Activate ~/protein_venv on Delta and install xgboost."
        ) from exc
    return XGBRegressor


def fit_xgboost(config: Mapping, train_bundle: Mapping, validation_bundle: Mapping, n_jobs: int):
    XGBRegressor = require_xgboost()
    model = XGBRegressor(
        **FIXED_XGBOOST_PARAMETERS, n_jobs=n_jobs,
        max_depth=int(config["max_depth"]), learning_rate=float(config["learning_rate"]),
        min_child_weight=float(config["min_child_weight"]),
    )
    train_weights = protein_equal_weights(train_bundle["protein_ids"])
    validation_weights = protein_equal_weights(validation_bundle["protein_ids"])
    tick = time.perf_counter()
    model.fit(
        train_bundle["x"], train_bundle["y"], sample_weight=train_weights,
        eval_set=[(validation_bundle["x"], validation_bundle["y"])],
        sample_weight_eval_set=[validation_weights], verbose=False,
    )
    fit_seconds = time.perf_counter() - tick
    return model, {
        "fit_seconds": fit_seconds,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_validation_rmse": float(getattr(model, "best_score", float("nan"))),
    }


def importance_rows(model, metadata: Mapping) -> list[dict]:
    rows = []
    booster = model.get_booster()
    for importance_type in ("gain", "weight", "cover"):
        values = booster.get_score(importance_type=importance_type)
        for channel in range(N_CHANNELS):
            rows.append({**metadata, "importance_type": importance_type, "channel": channel, "feature_name": f"avg_symmetric_ch_{channel}", "importance": float(values.get(f"f{channel}", 0.0))})
    return rows


def select_winner(layer_rows: Mapping[int, Sequence[Mapping]], configs: Sequence[Mapping]):
    """Select solely from validation mean R², then depth/runtime/ID."""
    config_map = {str(c["config_id"]): c for c in configs}
    table = []
    for config_id, config in config_map.items():
        found = []
        for layer in ANCHOR_LAYERS:
            matches = [r for r in layer_rows[layer] if str(r["config_id"]) == config_id]
            if len(matches) != 1:
                raise ValueError(f"Layer {layer} must contain exactly one row for {config_id}; found {len(matches)}.")
            found.append(matches[0])
        validation_scores = [float(r["validation_mean_per_protein_r2"]) for r in found]
        table.append({
            "config_id": config_id, "max_depth": int(config["max_depth"]),
            "mean_validation_r2_across_layers": float(np.mean(validation_scores)),
            "summed_fit_seconds": float(sum(float(r["fit_seconds"]) for r in found)),
            **{f"layer_{layer}_validation_mean_per_protein_r2": validation_scores[i] for i, layer in enumerate(ANCHOR_LAYERS)},
        })
    maximum = max(r["mean_validation_r2_across_layers"] for r in table)
    tied = [r for r in table if maximum - r["mean_validation_r2_across_layers"] <= SELECTION_TOLERANCE + 1e-15]
    selected_row = min(tied, key=lambda r: (r["max_depth"], r["summed_fit_seconds"], r["config_id"]))
    selected = dict(config_map[selected_row["config_id"]])
    provenance = {
        "selection_metric": "validation_mean_per_protein_r2_equal_layer_average",
        "test_metrics_used": False, "tie_tolerance": SELECTION_TOLERANCE,
        "tie_break_order": ["lower_max_depth", "shorter_summed_fit_seconds", "lower_config_id"],
        "selected_config": selected, "selected_score": selected_row["mean_validation_r2_across_layers"],
        "selected_at": timestamp(),
    }
    return selected, table, provenance


def load_complete_search(search_dir: Path, configs: Sequence[Mapping]):
    expected_ids = {str(c["config_id"]) for c in configs}
    layers, digests = {}, set()
    for layer in ANCHOR_LAYERS:
        layer_dir = search_dir / f"layer_{layer:02d}"
        rows = read_csv(layer_dir / "search_summary.csv")
        ids = [r.get("config_id", "") for r in rows]
        if len(rows) != len(configs) or set(ids) != expected_ids or len(ids) != len(set(ids)):
            raise RuntimeError(f"Incomplete search summary for layer {layer}: expected {sorted(expected_ids)}, observed {ids}.")
        metadata_path = layer_dir / "search_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Required search metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if int(metadata.get("layer", -1)) != layer:
            raise RuntimeError(f"Search metadata layer mismatch in {metadata_path}.")
        if metadata.get("status") != "complete":
            raise RuntimeError(f"Search task for layer {layer} is not complete: {metadata.get('status')!r}.")
        if any(row.get("split_digest") != metadata.get("split_digest") for row in rows):
            raise RuntimeError(f"Search summary rows disagree with metadata split digest in {layer_dir}.")
        digests.add(metadata.get("split_digest"))
        layers[layer] = rows
    if len(digests) != 1 or None in digests:
        raise RuntimeError(f"Search tasks used inconsistent splits: {sorted(map(str, digests))}")
    return layers, next(iter(digests))


def n_jobs_from_environment() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


def aggregate_rows(rows: Sequence[Mapping], split_name: str) -> dict:
    stats = PooledStats()
    r2s = []
    for row in rows:
        if row.get("split") != split_name:
            continue
        if "_y_true" in row:
            stats.update(row["_y_true"], row["_y_pred"])
        if np.isfinite(float(row["r2"])):
            r2s.append(float(row["r2"]))
    return {"split": split_name, "n_proteins": len(r2s), "mean_per_protein_r2": float(np.mean(r2s)), **{f"pooled_{k}": v for k, v in stats.metrics().items()}}


def write_summary(path: Path, title: str, lines: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join([title, f"Generated: {timestamp()}", *lines, ""]) + "\n")


def attach_pair_splits(records: Sequence[dict]) -> None:
    for record in records:
        record["pair_splits"] = pair_split_indices(record["n_pairs"], record["protein_id"])


def pair_split_summary(records: Sequence[dict]) -> tuple[list[dict], str]:
    rows = []
    for record in records:
        for split_name in ("fit", "validation", "test"):
            indices = np.asarray(record["pair_splits"][split_name], dtype=np.int64)
            rows.append({
                "protein_id": record["protein_id"], "split": split_name,
                "n_pairs": int(indices.size),
                "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            })
    canonical = [(r["protein_id"], r["split"], r["n_pairs"], r["indices_sha256"]) for r in rows]
    return rows, hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def evaluate_model_records(model, records: Sequence[dict], layer: int, split_name: str, geometry: bool = False, metadata: Mapping | None = None):
    """Stream full proteins through feature loading, prediction, and metrics."""
    metadata = dict(metadata or {})
    rows, geometry_rows = [], []
    pooled = PooledStats()
    times = {"materialization_seconds": 0.0, "prediction_seconds": 0.0, "metric_seconds": 0.0, "geometry_seconds": 0.0}
    for record in records:
        tick = time.perf_counter()
        x = load_pair_features(record, layer)
        times["materialization_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        pred = np.asarray(model.predict(x), dtype=np.float64)
        times["prediction_seconds"] += time.perf_counter() - tick
        del x
        tick = time.perf_counter()
        y = np.asarray(record["y"], dtype=np.float64)
        metrics = metric_row(y, pred)
        pooled.update(y, pred)
        rows.append({
            **metadata, "layer": layer, "split": split_name,
            "protein_id": record["protein_id"], "source": record["source"],
            "size_bin": record["size_bin"], "n_residues": record["n_residues"],
            "n_pairs": record["n_pairs"], **metrics,
        })
        times["metric_seconds"] += time.perf_counter() - tick
        if geometry:
            tick = time.perf_counter()
            geometry_rows.append({
                **metadata, "layer": layer, "split": split_name,
                "geometry_prediction_scope": "held_out_protein_predictions",
                "protein_id": record["protein_id"], "source": record["source"],
                "size_bin": record["size_bin"], "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"], "regression_r2": metrics["r2"],
                **geometry_metrics(record, pred),
            })
            times["geometry_seconds"] += time.perf_counter() - tick
        del pred
    finite_r2 = [float(r["r2"]) for r in rows if np.isfinite(float(r["r2"]))]
    pooled_metrics = pooled.metrics()
    aggregate = {
        **metadata, "layer": layer, "split": split_name,
        "n_proteins": len(rows), "n_pairs": pooled.n,
        "mean_per_protein_r2": float(np.mean(finite_r2)),
        "mean_per_protein_mae": float(np.mean([float(r["mae"]) for r in rows])),
        "mean_per_protein_rmse": float(np.mean([float(r["rmse"]) for r in rows])),
        **{f"pooled_{key}": value for key, value in pooled_metrics.items()},
    }
    return rows, aggregate, geometry_rows, times


def summarize_geometry(geometry_rows: Sequence[Mapping]) -> list[dict]:
    output = []
    metric_names = (
        "tm_score", "kabsch_rmsd", "mds_stress",
        "top3_positive_spectral_mass_fraction", "negative_eigenvalue_count",
        "negative_eigenvalue_fraction", "negative_prediction_count",
    )
    groups = sorted({(int(r["layer"]), str(r["split"])) for r in geometry_rows})
    for layer, split_name in groups:
        group = [r for r in geometry_rows if int(r["layer"]) == layer and str(r["split"]) == split_name]
        row = {"layer": layer, "split": split_name, "n_proteins": len(group)}
        for name in metric_names:
            values = np.asarray([float(r[name]) for r in group], dtype=np.float64)
            row[f"{name}_mean"] = float(np.nanmean(values))
            row[f"{name}_std"] = float(np.nanstd(values))
        output.append(row)
    return output


def cleanup_bundle(bundle: Mapping) -> None:
    for key in ("x", "y"):
        if key in bundle:
            del bundle[key]
    gc.collect()

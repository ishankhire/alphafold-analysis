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

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
GENERALIZED_DIR = CLUSTER_DIR / "generalized_model_ecod"
import sys

for _path in (CLUSTER_DIR, GENERALIZED_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cluster_regression_helpers import compute_distance_matrix, extract_ca_coordinates  # noqa: E402
from general_train_test_common import (  # noqa: E402
    build_dataset_protein_manifest,
    resolve_pair_block_path,
    resolve_pdb_file,
)
from regression_protocol import (  # noqa: E402
    DEVELOPMENT_SPLIT_SEED,
    FEATURE_DIM,
    FEATURE_NAMES,
    MASTER_MANIFEST_DIGEST,
    MASTER_MANIFEST_VERSION,
    OUTER_SPLIT_SEED,
    extract_directional_base256,
    load_master_manifest,
    strict_lower_triangle_indices,
)


ANCHOR_LAYERS = (0, 23, 47)
CAPS = (5000, 10000, 20000)
CAP_LABELS = ("5k", "10k", "20k")
N_CHANNELS = 128
N_SIZE_BINS = 5
RANDOM_STATE = 42
SEARCH_CAP = 10000
N_ESTIMATORS = 700
EARLY_STOPPING_ROUNDS = 30
SELECTION_TOLERANCE = 0.001
EPS = 1e-12
AVERAGED_SYMMETRIC_FEATURES = "avg_symmetric_128"
DIRECTIONAL_BASE256_FEATURES = "directional_base256"
SHARED_VARIANT = "shared_xgboost_base256"
PER_PROTEIN_VARIANT = "per_protein_xgboost_avg_symmetric_128"
SHARED_WEIGHTING_MODE = "protein_equal"
PER_PROTEIN_WEIGHTING_MODE = "uniform_rows"
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


def remove_obsolete_outputs(
    directory: Path,
    filenames: Sequence[str] = (),
    glob_patterns: Sequence[str] = (),
) -> None:
    """Remove only explicitly named legacy artifacts from a task directory."""
    candidates = [directory / filename for filename in filenames]
    for pattern in glob_patterns:
        candidates.extend(directory.glob(pattern))
    for path in sorted(set(candidates)):
        if path.is_file():
            path.unlink()
            log(f"Removed obsolete output: {path}")


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


def build_shared_split(records: Sequence[dict]):
    """Apply the frozen canonical fit/validation/test roles to loaded records."""
    manifest_entries = load_master_manifest()
    eligible_entries = {
        entry.protein_id: entry for entry in manifest_entries if entry.eligible
    }
    record_by_id = {str(record["protein_id"]): record for record in records}
    if len(record_by_id) != len(records):
        raise ValueError("Loaded shared XGBoost records contain duplicate protein IDs.")
    missing = sorted(set(eligible_entries) - set(record_by_id))
    unexpected = sorted(set(record_by_id) - set(eligible_entries))
    if missing or unexpected:
        raise RuntimeError(
            "Loaded shared XGBoost cohort differs from the frozen canonical "
            f"manifest (missing={missing}, unexpected={unexpected})."
        )

    parts = {role: [] for role in ("fit", "validation", "test")}
    for protein_id, entry in eligible_entries.items():
        record = record_by_id[protein_id]
        observed = (
            str(record.get("source", "")),
            int(record["n_residues"]),
            int(record["n_pairs"]),
        )
        expected = (entry.source, entry.n_residues, entry.n_pairs)
        if observed != expected:
            raise RuntimeError(
                f"Loaded metadata for {protein_id} differs from the frozen "
                f"manifest: observed={observed}, expected={expected}."
            )
        record["size_bin"] = int(entry.size_bin)
        parts[entry.master_role].append(record)
    for split_records in parts.values():
        split_records.sort(key=lambda record: str(record["protein_id"]))

    assignments = []
    for split_name, split_records in parts.items():
        for r in split_records:
            assignments.append({
                "manifest_version": MASTER_MANIFEST_VERSION,
                "master_manifest_digest": MASTER_MANIFEST_DIGEST,
                "protein_id": r["protein_id"], "source": r.get("source", ""),
                "n_residues": int(r["n_residues"]), "n_pairs": int(r["n_pairs"]),
                "size_bin": int(r["size_bin"]),
                "split": split_name,
                "protein_split": "test" if split_name == "test" else "development",
                "outer_split_seed": OUTER_SPLIT_SEED,
                "development_split_seed": DEVELOPMENT_SPLIT_SEED,
            })
    if (
        len({r["protein_id"] for r in assignments}) != len(records)
        or {name: len(group) for name, group in parts.items()}
        != {"fit": 330, "validation": 111, "test": 111}
    ):
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


def map_shared_winner_task(task_id: int) -> tuple[int, int, str]:
    if not 0 <= int(task_id) < 9:
        raise ValueError(f"Shared winner task ID must be 0..8; got {task_id}.")
    task_id = int(task_id)
    layer_idx, cap_idx = divmod(task_id, 3)
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


def directional_base256_features(
    pair_block: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray,
) -> np.ndarray:
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)
    if pair_i.size and not np.all(pair_i > pair_j):
        raise ValueError(
            "Canonical directional base256 features require strict-lower-triangle "
            "pair indices (i > j)."
        )
    return extract_directional_base256(pair_block, pair_i, pair_j)


def feature_dimension(feature_mode: str) -> int:
    if feature_mode == DIRECTIONAL_BASE256_FEATURES:
        return FEATURE_DIM
    if feature_mode == AVERAGED_SYMMETRIC_FEATURES:
        return N_CHANNELS
    raise ValueError(f"Unknown XGBoost feature mode: {feature_mode!r}")


def load_pair_features(
    record: Mapping,
    layer: int,
    indices: np.ndarray | None = None,
    feature_mode: str = AVERAGED_SYMMETRIC_FEATURES,
) -> np.ndarray:
    path = resolve_pair_block_path(Path(record["protein_dir"]), str(record["protein_id"]), int(layer))
    if not path.exists():
        raise FileNotFoundError(f"Required layer {layer} pair block is missing for {record['protein_id']}: {path}")
    pb = np.load(path)
    pair_i = np.asarray(record["pair_i"])
    pair_j = np.asarray(record["pair_j"])
    if indices is not None:
        pair_i, pair_j = pair_i[indices], pair_j[indices]
    if feature_mode == DIRECTIONAL_BASE256_FEATURES:
        features = directional_base256_features(pb, pair_i, pair_j)
    elif feature_mode == AVERAGED_SYMMETRIC_FEATURES:
        features = averaged_symmetric_features(pb, pair_i, pair_j)
    else:
        raise ValueError(f"Unknown XGBoost feature mode: {feature_mode!r}")
    del pb
    return features


def load_combined_records(
    require_layers: Sequence[int] = ANCHOR_LAYERS,
    pair_order: str = "upper",
):
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
            if pair_order == "lower":
                pair_i, pair_j = strict_lower_triangle_indices(coords.shape[0])
            elif pair_order == "upper":
                pair_i, pair_j = np.triu_indices(coords.shape[0], k=1)
            else:
                raise ValueError(f"Unknown pair ordering: {pair_order!r}")
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


def materialize_records(
    records: Sequence[dict],
    layer: int,
    cap: int | None = None,
    split_name: str | None = None,
    feature_mode: str = AVERAGED_SYMMETRIC_FEATURES,
):
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
        x = load_pair_features(record, layer, indices, feature_mode)
        elapsed += time.perf_counter() - tick
        y = np.asarray(record["y"][indices], dtype=np.float64)
        end = start + y.size
        offsets.append((record, indices, start, end))
        chunks_x.append(x)
        chunks_y.append(y)
        row_ids.extend([record["protein_id"]] * y.size)
        start = end
    expected_feature_dim = feature_dimension(feature_mode)
    x_all = np.concatenate(chunks_x, axis=0) if chunks_x else np.empty((0, expected_feature_dim), dtype=np.float64)
    y_all = np.concatenate(chunks_y) if chunks_y else np.empty(0, dtype=np.float64)
    if x_all.dtype != np.float64 or x_all.shape[1] != expected_feature_dim:
        raise AssertionError(
            "Materialized XGBoost features must be unstandardized float64 "
            f"with width {expected_feature_dim}."
        )
    return {
        "x": x_all, "y": y_all, "protein_ids": row_ids, "offsets": offsets,
        "feature_dim": expected_feature_dim,
        "materialization_seconds": elapsed,
    }


def r2_score(y_true, y_pred) -> float:
    y_true, y_pred = np.asarray(y_true, dtype=np.float64), np.asarray(y_pred, dtype=np.float64)
    denominator = float(np.sum((y_true - np.mean(y_true)) ** 2, dtype=np.float64))
    return float("nan") if denominator <= EPS else 1.0 - float(np.sum((y_true - y_pred) ** 2, dtype=np.float64)) / denominator


def metric_row(y_true, y_pred) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(y_pred, dtype=np.float64)
    err = pred - y
    mse = float(np.mean(err * err))
    return {
        "target_variance": float(np.var(y, dtype=np.float64)),
        "r2": r2_score(y, pred),
        "mse": mse,
        "mae": float(np.mean(np.abs(err))),
        "rmse": float(np.sqrt(mse)),
        "mean_true_distance": float(np.mean(y)),
        "mean_predicted_distance": float(np.mean(pred)),
        "mean_error_pred_minus_true": float(np.mean(err)),
    }


@dataclass
class PooledStats:
    n: int = 0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    sum_error: float = 0.0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0

    def update(self, y_true, y_pred) -> None:
        y = np.asarray(y_true, dtype=np.float64)
        err = np.asarray(y_pred, dtype=np.float64) - y
        self.n += int(y.size)
        self.sum_y += float(np.sum(y, dtype=np.float64))
        self.sum_y2 += float(np.sum(y * y, dtype=np.float64))
        self.sum_error += float(np.sum(err, dtype=np.float64))
        self.sum_abs_error += float(np.sum(np.abs(err), dtype=np.float64))
        self.sum_squared_error += float(np.sum(err * err, dtype=np.float64))

    def metrics(self) -> dict[str, float]:
        if self.n == 0:
            return {
                "r2": float("nan"), "mse": float("nan"), "mae": float("nan"),
                "rmse": float("nan"), "mean_error_pred_minus_true": float("nan"),
            }
        sst = self.sum_y2 - self.sum_y * self.sum_y / self.n
        mse = self.sum_squared_error / self.n
        return {
            "r2": float("nan") if sst <= EPS else 1.0 - self.sum_squared_error / sst,
            "mse": mse,
            "mae": self.sum_abs_error / self.n,
            "rmse": math.sqrt(mse),
            "mean_error_pred_minus_true": self.sum_error / self.n,
        }


def regression_output_row(
    record: Mapping,
    y_true: np.ndarray,
    predictions: np.ndarray,
    split_name: str,
    evaluation_scope: str,
    metadata: Mapping | None = None,
    feature_dim: int = N_CHANNELS,
) -> dict:
    metrics = metric_row(y_true, predictions)
    return {
        **dict(metadata or {}),
        "feature_dim": int(feature_dim),
        "split": "train" if split_name == "fit" else split_name,
        "evaluation_scope": evaluation_scope,
        "source": record["source"],
        "protein_id": record["protein_id"],
        "n_residues": int(record["n_residues"]),
        "size_bin": int(record["size_bin"]),
        "n_pairs": int(np.asarray(y_true).size),
        "target_variance": metrics["target_variance"],
        "per_protein_r2": metrics["r2"],
        "per_protein_mse": metrics["mse"],
        "per_protein_mae": metrics["mae"],
        "per_protein_rmse": metrics["rmse"],
        "mean_true_distance": metrics["mean_true_distance"],
        "mean_predicted_distance": metrics["mean_predicted_distance"],
        "mean_error_pred_minus_true": metrics["mean_error_pred_minus_true"],
    }


def split_summary_row(
    rows: Sequence[Mapping],
    pooled: PooledStats,
    metadata: Mapping | None = None,
    feature_dim: int = N_CHANNELS,
) -> dict:
    if not rows:
        raise ValueError("Cannot summarize an empty regression split.")

    def values(name: str) -> np.ndarray:
        return np.asarray([float(row[name]) for row in rows], dtype=np.float64)

    pooled_metrics = pooled.metrics()
    return {
        **dict(metadata or {}),
        "feature_dim": int(feature_dim),
        "split": rows[0]["split"],
        "n_proteins": len(rows),
        "n_pairs": pooled.n,
        "per_protein_r2_mean": float(np.nanmean(values("per_protein_r2"))),
        "per_protein_r2_std": float(np.nanstd(values("per_protein_r2"))),
        "per_protein_mae_mean": float(np.mean(values("per_protein_mae"))),
        "per_protein_mae_std": float(np.std(values("per_protein_mae"))),
        "per_protein_rmse_mean": float(np.mean(values("per_protein_rmse"))),
        "per_protein_rmse_std": float(np.std(values("per_protein_rmse"))),
        "pair_weighted_r2": pooled_metrics["r2"],
        "pair_weighted_mse": pooled_metrics["mse"],
        "pair_weighted_mae": pooled_metrics["mae"],
        "pair_weighted_rmse": pooled_metrics["rmse"],
        "mean_error_pred_minus_true": pooled_metrics["mean_error_pred_minus_true"],
    }


def summarize_prediction_bundle(
    bundle: Mapping,
    predictions: np.ndarray,
    split_name: str,
    evaluation_scope: str,
    metadata: Mapping | None = None,
):
    rows, pooled = [], PooledStats()
    for record, indices, start, end in bundle["offsets"]:
        y = bundle["y"][start:end]
        pred = predictions[start:end]
        pooled.update(y, pred)
        rows.append(regression_output_row(
            record, y, pred, split_name, evaluation_scope, metadata,
            int(bundle.get("feature_dim", N_CHANNELS)),
        ))
    return rows, split_summary_row(
        rows, pooled, metadata, int(bundle.get("feature_dim", N_CHANNELS))
    )


def require_xgboost():
    try:
        from xgboost import XGBRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "XGBoost is required for this job. Activate ~/protein_venv on Delta and install xgboost."
        ) from exc
    return XGBRegressor


def fit_xgboost(
    config: Mapping,
    train_bundle: Mapping,
    validation_bundle: Mapping,
    n_jobs: int,
    weighting_mode: str,
):
    XGBRegressor = require_xgboost()
    model = XGBRegressor(
        **FIXED_XGBOOST_PARAMETERS, n_jobs=n_jobs,
        max_depth=int(config["max_depth"]), learning_rate=float(config["learning_rate"]),
        min_child_weight=float(config["min_child_weight"]),
    )
    fit_kwargs = {
        "eval_set": [(validation_bundle["x"], validation_bundle["y"])],
        "verbose": False,
    }
    if weighting_mode == SHARED_WEIGHTING_MODE:
        fit_kwargs["sample_weight"] = protein_equal_weights(train_bundle["protein_ids"])
        fit_kwargs["sample_weight_eval_set"] = [
            protein_equal_weights(validation_bundle["protein_ids"])
        ]
    elif weighting_mode != PER_PROTEIN_WEIGHTING_MODE:
        raise ValueError(
            f"Unknown XGBoost weighting mode {weighting_mode!r}; expected "
            f"{SHARED_WEIGHTING_MODE!r} or {PER_PROTEIN_WEIGHTING_MODE!r}."
        )
    tick = time.perf_counter()
    model.fit(train_bundle["x"], train_bundle["y"], **fit_kwargs)
    fit_seconds = time.perf_counter() - tick
    return model, {
        "fit_seconds": fit_seconds,
        "best_iteration": int(getattr(model, "best_iteration", -1)),
        "best_validation_rmse": float(getattr(model, "best_score", float("nan"))),
        "weighting_mode": weighting_mode,
    }


def importance_rows(
    model,
    metadata: Mapping,
    feature_mode: str = AVERAGED_SYMMETRIC_FEATURES,
) -> list[dict]:
    rows = []
    booster = model.get_booster()
    if feature_mode == DIRECTIONAL_BASE256_FEATURES:
        feature_names = FEATURE_NAMES
    elif feature_mode == AVERAGED_SYMMETRIC_FEATURES:
        feature_names = tuple(
            f"avg_symmetric_ch_{channel}" for channel in range(N_CHANNELS)
        )
    else:
        raise ValueError(f"Unknown XGBoost feature mode: {feature_mode!r}")
    for importance_type in ("gain", "weight", "cover"):
        values = booster.get_score(importance_type=importance_type)
        for feature_index, feature_name in enumerate(feature_names):
            row = {
                **metadata,
                "importance_type": importance_type,
                "channel": feature_index,
                "feature_name": feature_name,
                "importance": float(values.get(f"f{feature_index}", 0.0)),
            }
            if feature_mode == DIRECTIONAL_BASE256_FEATURES:
                row["feature_index"] = feature_index
            rows.append(row)
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
        "training_metrics_used": False, "test_metrics_used": False,
        "tie_tolerance": SELECTION_TOLERANCE,
        "tie_break_order": ["lower_max_depth", "shorter_summed_fit_seconds", "lower_config_id"],
        "selected_config": selected, "selected_score": selected_row["mean_validation_r2_across_layers"],
        "selected_at": timestamp(),
    }
    return selected, table, provenance


def load_complete_search(
    search_dir: Path,
    configs: Sequence[Mapping],
    expected_weighting_mode: str | None = None,
):
    expected_ids = {str(c["config_id"]) for c in configs}
    required_columns = {
        "config_id",
        "train_mean_per_protein_r2",
        "train_pooled_r2",
        "validation_mean_per_protein_r2",
        "validation_pooled_r2",
        "test_mean_per_protein_r2",
        "test_pooled_r2",
        "fit_seconds",
        "split_digest",
    }
    layers, digests = {}, set()
    for layer in ANCHOR_LAYERS:
        layer_dir = search_dir / f"layer_{layer:02d}"
        rows = read_csv(layer_dir / "search_summary.csv")
        ids = [r.get("config_id", "") for r in rows]
        if len(rows) != len(configs) or set(ids) != expected_ids or len(ids) != len(set(ids)):
            raise RuntimeError(f"Incomplete search summary for layer {layer}: expected {sorted(expected_ids)}, observed {ids}.")
        for row in rows:
            missing_columns = sorted(required_columns - set(row))
            if missing_columns:
                raise RuntimeError(
                    f"Incomplete search summary schema for layer {layer}, "
                    f"configuration {row.get('config_id')}: missing "
                    f"{missing_columns}. Rerun the search."
                )
        metadata_path = layer_dir / "search_metadata.json"
        if not metadata_path.exists():
            raise FileNotFoundError(f"Required search metadata is missing: {metadata_path}")
        metadata = json.loads(metadata_path.read_text())
        if int(metadata.get("layer", -1)) != layer:
            raise RuntimeError(f"Search metadata layer mismatch in {metadata_path}.")
        if metadata.get("status") != "complete":
            raise RuntimeError(f"Search task for layer {layer} is not complete: {metadata.get('status')!r}.")
        if (
            expected_weighting_mode is not None
            and metadata.get("weighting_mode") != expected_weighting_mode
        ):
            raise RuntimeError(
                f"Search task for layer {layer} used weighting mode "
                f"{metadata.get('weighting_mode')!r}; expected "
                f"{expected_weighting_mode!r}. Rerun the search before winner evaluation."
            )
        if any(row.get("split_digest") != metadata.get("split_digest") for row in rows):
            raise RuntimeError(f"Search summary rows disagree with metadata split digest in {layer_dir}.")
        digests.add(metadata.get("split_digest"))
        layers[layer] = rows
    if len(digests) != 1 or None in digests:
        raise RuntimeError(f"Search tasks used inconsistent splits: {sorted(map(str, digests))}")
    return layers, next(iter(digests))


def n_jobs_from_environment() -> int:
    return int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))


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
                "protein_id": record["protein_id"],
                "split": "train" if split_name == "fit" else split_name,
                "n_pairs": int(indices.size),
                "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            })
    canonical = [(r["protein_id"], r["split"], r["n_pairs"], r["indices_sha256"]) for r in rows]
    return rows, hashlib.sha256(json.dumps(canonical, separators=(",", ":")).encode()).hexdigest()


def evaluate_model_records(
    model,
    records: Sequence[dict],
    layer: int,
    split_name: str,
    evaluation_scope: str,
    metadata: Mapping | None = None,
    feature_mode: str = AVERAGED_SYMMETRIC_FEATURES,
):
    """Stream full proteins through feature loading, prediction, and metrics."""
    metadata = dict(metadata or {})
    rows = []
    pooled = PooledStats()
    times = {
        "materialization_seconds": 0.0,
        "prediction_seconds": 0.0,
        "metric_seconds": 0.0,
    }
    for record in records:
        tick = time.perf_counter()
        x = load_pair_features(record, layer, feature_mode=feature_mode)
        times["materialization_seconds"] += time.perf_counter() - tick
        tick = time.perf_counter()
        pred = np.asarray(model.predict(x), dtype=np.float64)
        times["prediction_seconds"] += time.perf_counter() - tick
        del x
        tick = time.perf_counter()
        y = np.asarray(record["y"], dtype=np.float64)
        pooled.update(y, pred)
        rows.append(regression_output_row(
            record,
            y,
            pred,
            split_name,
            evaluation_scope,
            {**metadata, "layer": layer},
            feature_dimension(feature_mode),
        ))
        times["metric_seconds"] += time.perf_counter() - tick
        del pred
    return rows, split_summary_row(
        rows,
        pooled,
        {**metadata, "layer": layer},
        feature_dimension(feature_mode),
    ), times


def require_non_degenerate_iterations(
    config_id: str, best_iterations: Sequence[int], expected_fits: int,
) -> None:
    if len(best_iterations) != expected_fits:
        raise RuntimeError(
            f"Configuration {config_id} completed {len(best_iterations)}/{expected_fits} fits."
        )
    if all(int(iteration) == 0 for iteration in best_iterations):
        raise RuntimeError(
            f"Configuration {config_id} stopped at iteration 0 for all "
            f"{expected_fits} proteins. Refusing to save a degenerate per-protein "
            "search; verify uniform row weighting and XGBoost inputs."
        )


def cleanup_bundle(bundle: Mapping) -> None:
    for key in ("x", "y"):
        if key in bundle:
            del bundle[key]
    gc.collect()

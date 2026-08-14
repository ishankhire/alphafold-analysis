#!/usr/bin/env python3
"""Shared preparation, fitting, metric, and checkpoint helpers for gated probes."""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
GENERAL_DIR = CLUSTER_SCRIPTS_DIR / "generalized_model_ecod"
for import_dir in (CLUSTER_SCRIPTS_DIR, GENERAL_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from cluster_regression_helpers import STANDARD_AA_PAIR_LABELS  # noqa: E402
import general_train_test_common as general_common  # noqa: E402


LAYERS = (0, 10, 20, 30, 40, 47)
SEQUENCE_BIN_LABELS = ("1", "2-4", "5-9", "10-19", "20-49", "50+")
DISTANCE_BIN_LABELS = ("0-5", "5-10", "10-20", "20-40", "40+")
FEATURE_DIM = 256
FLOAT_EPS = 1e-12
MIN_TRAIN_ROWS = FEATURE_DIM + 1
MIN_TRAIN_PROTEINS = 5
MIN_EVAL_ROWS = 100
MIN_EVAL_PROTEINS = 5
CONDITION_LIMIT = 1e12
RUN_MODE_ENV = "GATED_PROBE_RUN_MODE"
VALID_RUN_MODES = {"production", "smoke"}

METRIC_FIELDNAMES = [
    "analysis_name", "model", "feature_dim", "layer", "split",
    "train_gate_aa_pair", "train_gate_sequence_bin", "evaluation_scope",
    "eval_aa_pair", "eval_sequence_bin", "eval_distance_bin", "fit_status",
    "support_flags", "n_train_pairs", "n_train_proteins", "n_pairs",
    "n_proteins", "n_proteins_with_finite_r2", "pooled_mean_true_distance",
    "pooled_median_true_distance", "pooled_target_variance", "pooled_target_std",
    "pooled_r2", "pooled_mae", "pooled_rmse", "mean_per_protein_r2",
    "mean_per_protein_mae", "mean_per_protein_rmse",
    "intercept_baseline_prediction", "baseline_pooled_r2",
    "baseline_pooled_mae", "baseline_pooled_rmse",
    "baseline_mean_per_protein_r2", "baseline_mean_per_protein_mae",
    "baseline_mean_per_protein_rmse", "pooled_mae_skill", "pooled_rmse_skill",
    "mean_per_protein_mae_skill", "mean_per_protein_rmse_skill",
]

GATE_STATUS_FIELDNAMES = [
    "analysis_name", "model", "feature_dim", "layer", "train_gate_aa_pair",
    "train_gate_sequence_bin", "fit_status", "support_flags", "n_train_pairs",
    "n_train_proteins", "design_rank", "largest_singular_value",
    "smallest_singular_value", "condition_number", "elapsed_seconds",
]

PER_PROTEIN_FIELDNAMES = [
    "analysis_name", "layer", "protein_id", "source", "n_residues", "size_bin",
    "train_gate_aa_pair", "train_gate_sequence_bin", "fit_status", "n_pairs",
    "target_variance", "per_protein_r2", "per_protein_mae", "per_protein_rmse",
    "mean_true_distance", "mean_predicted_distance",
]


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def log(message):
    peak_mb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    if sys.platform != "darwin":
        peak_mb /= 1024.0
    else:
        peak_mb /= 1024.0 * 1024.0
    print(f"[{utc_now()}] [peak_rss={peak_mb:.1f} MiB] {message}", flush=True)


def run_mode():
    value = os.environ.get(RUN_MODE_ENV, "production").strip().lower()
    if value not in VALID_RUN_MODES:
        raise ValueError(f"{RUN_MODE_ENV} must be one of {sorted(VALID_RUN_MODES)}, got {value!r}")
    return value


def configured_layers(mode=None):
    return (0,) if (mode or run_mode()) == "smoke" else LAYERS


def suite_root(mode=None):
    mode = mode or run_mode()
    csv_root = Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))).expanduser()
    return csv_root / ("gated_linear_probes_smoke" if mode == "smoke" else "gated_linear_probes")


def repository_root():
    return SCRIPT_DIR.parents[2]


def consolidated_root(mode=None):
    mode = mode or run_mode()
    if mode == "smoke":
        return suite_root(mode) / "consolidated"
    return repository_root() / "delta_files" / "gated_linear_probes"


def canonical_json_bytes(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()


def sha256_bytes(value):
    return hashlib.sha256(value).hexdigest()


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def configuration_payload(mode=None):
    mode = mode or run_mode()
    return {
        "schema_version": 1,
        "run_mode": mode,
        "layers": list(configured_layers(mode)),
        "sequence_bins": list(SEQUENCE_BIN_LABELS),
        "distance_bins": list(DISTANCE_BIN_LABELS),
        "aa_pairs": list(STANDARD_AA_PAIR_LABELS),
        "feature_mode": "concat_directional",
        "feature_dim": FEATURE_DIM,
        "precision": "float64",
        "random_state": general_common.RANDOM_STATE,
        "test_size": general_common.TEST_SIZE,
        "n_size_bins": general_common.N_SIZE_BINS,
        "min_train_rows": MIN_TRAIN_ROWS,
        "min_train_proteins": MIN_TRAIN_PROTEINS,
        "condition_limit": CONDITION_LIMIT,
    }


def configuration_digest(mode=None):
    return sha256_bytes(canonical_json_bytes(configuration_payload(mode)))


def _atomic_target(path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, temp_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    os.close(fd)
    return path, Path(temp_name)


def atomic_write_json(path, value):
    path, temp = _atomic_target(path)
    try:
        temp.write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=True) + "\n")
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_write_text(path, value):
    path, temp = _atomic_target(path)
    try:
        temp.write_text(value)
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def _csv_value(value):
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        return "nan" if not math.isfinite(value) else f"{value:.12g}"
    return value


def atomic_write_csv(path, fieldnames, rows, compress=False):
    path, temp = _atomic_target(path)
    try:
        opener = gzip.open if compress or str(path).endswith(".gz") else open
        mode = "wt" if opener is gzip.open else "w"
        with opener(temp, mode, newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            for row in rows:
                writer.writerow({key: _csv_value(row.get(key, "")) for key in fieldnames})
        os.replace(temp, path)
    finally:
        if temp.exists():
            temp.unlink()


def atomic_savez(path, **arrays):
    path, temp = _atomic_target(path)
    npz_temp = Path(str(temp) + ".npz")
    try:
        np.savez_compressed(npz_temp, **arrays)
        os.replace(npz_temp, path)
    finally:
        for candidate in (temp, npz_temp):
            if candidate.exists():
                candidate.unlink()


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def sequence_bin_codes(sequence_separation, valid_mask=None):
    values = np.asarray(sequence_separation)
    result = np.full(values.shape, -1, dtype=np.int8)
    result[values == 1] = 0
    result[(values >= 2) & (values <= 4)] = 1
    result[(values >= 5) & (values <= 9)] = 2
    result[(values >= 10) & (values <= 19)] = 3
    result[(values >= 20) & (values <= 49)] = 4
    result[values >= 50] = 5
    if valid_mask is not None:
        result[~np.asarray(valid_mask, dtype=bool)] = -1
    return result


def distance_bin_codes(distance):
    values = np.asarray(distance, dtype=np.float64)
    result = np.full(values.shape, -1, dtype=np.int8)
    finite = np.isfinite(values) & (values >= 0.0)
    result[finite & (values < 5.0)] = 0
    result[finite & (values >= 5.0) & (values < 10.0)] = 1
    result[finite & (values >= 10.0) & (values < 20.0)] = 2
    result[finite & (values >= 20.0) & (values < 40.0)] = 3
    result[finite & (values >= 40.0)] = 4
    return result


def layer_for_task(task_id, mode=None):
    layers = configured_layers(mode)
    if task_id < 0 or task_id >= len(layers):
        raise ValueError(f"Layer task {task_id} outside [0, {len(layers) - 1}]")
    return layers[task_id]


def joint_task_for_id(task_id, mode=None):
    layers = configured_layers(mode)
    task_count = len(layers) * len(SEQUENCE_BIN_LABELS)
    if task_id < 0 or task_id >= task_count:
        raise ValueError(f"Joint task {task_id} outside [0, {task_count - 1}]")
    return layers[task_id // len(SEQUENCE_BIN_LABELS)], task_id % len(SEQUENCE_BIN_LABELS)


def feature_names():
    return tuple([f"pair_ij_ch_{i}" for i in range(128)] + [f"pair_ji_ch_{i}" for i in range(128)])


def extract_base256(pb, pair_i, pair_j):
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)
    output = np.empty((pair_i.size, FEATURE_DIM), dtype=np.float64)
    output[:, :128] = pb[pair_i, pair_j, :]
    output[:, 128:] = pb[pair_j, pair_i, :]
    return output


def load_layer_feature_cache(records, layer):
    """Load each pair block once and retain its directional base256 rows as float32."""
    cache = {}
    total_rows = 0
    for index, record in enumerate(records, start=1):
        pb_path = general_common.resolve_pair_block_path(record["protein_dir"], record["protein_id"], layer)
        pb = np.load(pb_path)
        expected_shape = (record["n_residues"], record["n_residues"], 128)
        if pb.shape != expected_shape:
            raise ValueError(f"Unexpected pair-block shape for {record['protein_id']} layer {layer}: {pb.shape}")
        pair_i, pair_j = np.tril_indices(record["n_residues"], k=-1)
        features = np.empty((len(pair_i), FEATURE_DIM), dtype=np.float32)
        features[:, :128] = pb[pair_i, pair_j, :]
        features[:, 128:] = pb[pair_j, pair_i, :]
        if not np.all(np.isfinite(features)):
            raise ValueError(f"Nonfinite pair features for {record['protein_id']} layer {layer}")
        cache[record["protein_id"]] = features
        total_rows += len(features)
        del pb
        if index % 25 == 0 or index == len(records):
            log(f"Layer {layer}: cached {index}/{len(records)} proteins ({total_rows:,} pair rows)")
    return cache


def protein_equal_weights(offsets, total_rows):
    weights = np.empty(total_rows, dtype=np.float64)
    for _, start, end in offsets:
        if end > start:
            weights[start:end] = 1.0 / (end - start)
    if total_rows:
        weights *= total_rows / np.sum(weights)
    return weights


def weighted_scaler(x, offsets):
    if x.dtype != np.float64:
        raise TypeError("Scaler input must be float64")
    means = [np.mean(x[start:end], axis=0, dtype=np.float64) for _, start, end in offsets if end > start]
    seconds = [np.mean(x[start:end] ** 2, axis=0, dtype=np.float64) for _, start, end in offsets if end > start]
    if not means:
        raise ValueError("Cannot scale an empty gate")
    mean = np.mean(np.stack(means), axis=0, dtype=np.float64)
    variance = np.maximum(np.mean(np.stack(seconds), axis=0, dtype=np.float64) - mean ** 2, 0.0)
    scale = np.sqrt(variance)
    scale[scale < FLOAT_EPS] = 1.0
    return mean, scale


def gate_eligibility(n_rows, n_proteins):
    flags = []
    if n_rows < MIN_TRAIN_ROWS:
        flags.append("insufficient_train_rows")
    if n_proteins < MIN_TRAIN_PROTEINS:
        flags.append("insufficient_train_proteins")
    if n_rows < MIN_TRAIN_ROWS:
        return "insufficient_train_rows", flags
    if n_proteins < MIN_TRAIN_PROTEINS:
        return "insufficient_train_proteins", flags
    return "eligible", flags


def fit_ols(x, y, offsets):
    if x.dtype != np.float64 or y.dtype != np.float64:
        raise TypeError("OLS inputs must be float64")
    scaler_started = time.monotonic()
    mean, scale = weighted_scaler(x, offsets)
    x_standardized = np.empty_like(x)
    np.subtract(x, mean, out=x_standardized)
    np.divide(x_standardized, scale, out=x_standardized)
    weights = protein_equal_weights(offsets, len(y))
    log(f"OLS scaler/materialization completed in {time.monotonic() - scaler_started:.1f}s")
    model = LinearRegression(copy_X=False)
    solver_started = time.monotonic()
    model.fit(x_standardized, y, sample_weight=weights)
    log(f"OLS dense least-squares solver completed in {time.monotonic() - solver_started:.1f}s")
    coef_std = np.asarray(model.coef_, dtype=np.float64)
    intercept_std = float(model.intercept_)
    coef_raw = coef_std / scale
    intercept_raw = intercept_std - float(np.dot(coef_raw, mean))
    singular = np.asarray(model.singular_, dtype=np.float64)
    rank = int(model.rank_)
    largest = float(singular[0]) if singular.size else float("nan")
    smallest = float(singular[-1]) if singular.size else float("nan")
    condition = largest / smallest if singular.size and smallest > 0 else float("inf")
    flags = []
    if rank < FEATURE_DIM:
        flags.append("rank_deficient")
    if not math.isfinite(condition) or condition > CONDITION_LIMIT:
        flags.append("ill_conditioned")
    if "rank_deficient" in flags:
        status = "fitted_rank_deficient"
    elif "ill_conditioned" in flags:
        status = "fitted_ill_conditioned"
    else:
        status = "fitted"
    return {
        "model": model, "mean": mean, "scale": scale, "coef_std": coef_std,
        "intercept_std": intercept_std, "coef_raw": coef_raw,
        "intercept_raw": intercept_raw, "design_rank": rank,
        "largest_singular_value": largest, "smallest_singular_value": smallest,
        "condition_number": condition, "fit_status": status, "flags": flags,
    }


def safe_r2(y, prediction):
    y = np.asarray(y, dtype=np.float64)
    if y.size < 2:
        return float("nan")
    denominator = float(np.sum((y - np.mean(y)) ** 2, dtype=np.float64))
    if denominator <= FLOAT_EPS:
        return float("nan")
    return 1.0 - float(np.sum((y - prediction) ** 2, dtype=np.float64)) / denominator


def _skill(error, baseline_error):
    if not math.isfinite(baseline_error) or baseline_error <= FLOAT_EPS:
        return float("nan")
    return 1.0 - error / baseline_error


def cell_moments(y, prediction):
    y = np.asarray(y, dtype=np.float64)
    prediction = np.asarray(prediction, dtype=np.float64)
    error = prediction - y
    return np.asarray([
        y.size, np.sum(y), np.sum(y * y), np.sum(prediction),
        np.sum(np.abs(error)), np.sum(error * error),
    ], dtype=np.float64)


def metrics_from_moments(moment):
    n, sum_y, sum_y2, _, sum_abs, sum_sqerr = np.asarray(moment, dtype=np.float64)
    if n <= 0:
        return {key: float("nan") for key in ("r2", "mae", "rmse", "mean", "variance", "std")}
    mean = sum_y / n
    variance = max(sum_y2 / n - mean * mean, 0.0)
    denominator = sum_y2 - n * mean * mean
    r2 = float("nan") if denominator <= FLOAT_EPS else 1.0 - sum_sqerr / denominator
    return {
        "r2": float(r2), "mae": float(sum_abs / n), "rmse": float(math.sqrt(sum_sqerr / n)),
        "mean": float(mean), "variance": float(variance), "std": float(math.sqrt(variance)),
    }


def _join_flags(flags):
    return ";".join(sorted(set(flag for flag in flags if flag)))


def summarize_cell(y, prediction, baseline_prediction, protein_slices, identifiers, fit_status,
                   fit_flags, n_train_pairs, n_train_proteins):
    y = np.asarray(y, dtype=np.float64)
    baseline = np.full(y.shape, baseline_prediction, dtype=np.float64)
    prediction_valid = prediction is not None
    if prediction_valid:
        prediction = np.asarray(prediction, dtype=np.float64)
        pooled = metrics_from_moments(cell_moments(y, prediction))
    else:
        pooled = {key: float("nan") for key in ("r2", "mae", "rmse")}
        target = metrics_from_moments(cell_moments(y, np.zeros_like(y)))
        pooled.update({"mean": target["mean"], "variance": target["variance"], "std": target["std"]})
    baseline_pooled = metrics_from_moments(cell_moments(y, baseline))

    per_r2, per_mae, per_rmse = [], [], []
    base_r2, base_mae, base_rmse = [], [], []
    audit_rows = []
    for protein_code, start, end in protein_slices:
        y_p = y[start:end]
        base_p = baseline[start:end]
        pred_p = prediction[start:end] if prediction_valid else None
        baseline_metrics = metrics_from_moments(cell_moments(y_p, base_p))
        base_mae.append(baseline_metrics["mae"])
        base_rmse.append(baseline_metrics["rmse"])
        if math.isfinite(baseline_metrics["r2"]):
            base_r2.append(baseline_metrics["r2"])
        if prediction_valid:
            pred_metrics = metrics_from_moments(cell_moments(y_p, pred_p))
            per_mae.append(pred_metrics["mae"])
            per_rmse.append(pred_metrics["rmse"])
            if math.isfinite(pred_metrics["r2"]):
                per_r2.append(pred_metrics["r2"])
            audit_rows.append((protein_code, *cell_moments(y_p, pred_p)))

    n_proteins = len(protein_slices)
    flags = list(fit_flags)
    if (identifiers.get("split") == "test"
            and (y.size < MIN_EVAL_ROWS or n_proteins < MIN_EVAL_PROTEINS)):
        flags.append("sparse_test_support")
    mean_pp_mae = float(np.mean(per_mae)) if per_mae else float("nan")
    mean_pp_rmse = float(np.mean(per_rmse)) if per_rmse else float("nan")
    base_mean_pp_mae = float(np.mean(base_mae)) if base_mae else float("nan")
    base_mean_pp_rmse = float(np.mean(base_rmse)) if base_rmse else float("nan")
    if baseline_pooled["mae"] <= FLOAT_EPS:
        flags.append("zero_baseline_mae")
    if baseline_pooled["rmse"] <= FLOAT_EPS:
        flags.append("zero_baseline_rmse")
    row = {
        **identifiers, "model": "base256_ols", "feature_dim": FEATURE_DIM,
        "fit_status": fit_status, "support_flags": _join_flags(flags),
        "n_train_pairs": n_train_pairs, "n_train_proteins": n_train_proteins,
        "n_pairs": int(y.size), "n_proteins": n_proteins,
        "n_proteins_with_finite_r2": len(per_r2),
        "pooled_mean_true_distance": pooled["mean"],
        "pooled_median_true_distance": float(np.median(y)) if y.size else float("nan"),
        "pooled_target_variance": pooled["variance"], "pooled_target_std": pooled["std"],
        "pooled_r2": pooled["r2"], "pooled_mae": pooled["mae"], "pooled_rmse": pooled["rmse"],
        "mean_per_protein_r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
        "mean_per_protein_mae": mean_pp_mae, "mean_per_protein_rmse": mean_pp_rmse,
        "intercept_baseline_prediction": baseline_prediction,
        "baseline_pooled_r2": baseline_pooled["r2"],
        "baseline_pooled_mae": baseline_pooled["mae"],
        "baseline_pooled_rmse": baseline_pooled["rmse"],
        "baseline_mean_per_protein_r2": float(np.mean(base_r2)) if base_r2 else float("nan"),
        "baseline_mean_per_protein_mae": base_mean_pp_mae,
        "baseline_mean_per_protein_rmse": base_mean_pp_rmse,
        "pooled_mae_skill": _skill(pooled["mae"], baseline_pooled["mae"]),
        "pooled_rmse_skill": _skill(pooled["rmse"], baseline_pooled["rmse"]),
        "mean_per_protein_mae_skill": _skill(mean_pp_mae, base_mean_pp_mae),
        "mean_per_protein_rmse_skill": _skill(mean_pp_rmse, base_mean_pp_rmse),
    }
    audit = np.asarray(audit_rows, dtype=np.float64) if audit_rows else np.empty((0, 7), dtype=np.float64)
    return row, audit


def load_context(root=None):
    root = Path(root or suite_root())
    manifest = json.loads((root / "run_manifest.json").read_text())
    expected = configuration_digest(manifest["run_mode"])
    if manifest["configuration_digest"] != expected:
        raise ValueError("Run manifest configuration digest does not match current code/configuration")
    assignments = read_csv(root / "split_assignments.csv")
    general_manifest = {row["protein_id"]: row for row in general_common.build_dataset_protein_manifest()}
    records = []
    for row in assignments:
        protein_id = row["protein_id"]
        source_row = general_manifest[protein_id]
        metadata_path = root / "metadata" / "per_protein" / f"{protein_id}.npz"
        with np.load(metadata_path) as data:
            metadata = {key: np.asarray(data[key]) for key in data.files}
        records.append({
            "protein_id": protein_id, "source": row["source"], "split": row["split"],
            "n_residues": int(row["n_residues"]), "size_bin": int(row["size_bin"]),
            "n_pairs": int(row["n_pairs"]), "protein_dir": Path(source_row["protein_dir"]),
            "metadata": metadata,
        })
    return manifest, records


def load_intercept_baselines(root):
    baselines = {}
    for row in read_csv(Path(root) / "metadata" / "baselines" / "intercept_baselines.csv"):
        key = (row["analysis_name"], row["train_gate_aa_pair"], row["train_gate_sequence_bin"])
        baselines[key] = {
            "prediction": float(row["intercept_baseline_prediction"]),
            "n_train_pairs": int(row["n_train_pairs"]),
            "n_train_proteins": int(row["n_train_proteins"]),
        }
    return baselines


def gate_mask(record, aa_code=None, sequence_code=None):
    metadata = record["metadata"]
    mask = np.ones(int(record["n_pairs"]), dtype=bool)
    if aa_code is not None:
        mask &= metadata["aa_pair_code"] == aa_code
    if sequence_code is not None:
        mask &= metadata["sequence_bin_code"] == sequence_code
    return mask


def materialize_gate(records, layer, aa_code=None, sequence_code=None, feature_cache=None):
    selected = []
    total = 0
    for global_record_code, record in enumerate(records):
        indices = np.flatnonzero(gate_mask(record, aa_code, sequence_code))
        if indices.size:
            selected.append((global_record_code, record, indices))
            total += indices.size
    x = np.empty((total, FEATURE_DIM), dtype=np.float64)
    y = np.empty(total, dtype=np.float64)
    aa = np.empty(total, dtype=np.int16)
    sequence = np.empty(total, dtype=np.int8)
    distance = np.empty(total, dtype=np.int8)
    offsets = []
    cursor = 0
    record_by_code = {}
    for global_record_code, record, indices in selected:
        record_by_code[global_record_code] = record
        end = cursor + indices.size
        if feature_cache is None:
            pb_path = general_common.resolve_pair_block_path(record["protein_dir"], record["protein_id"], layer)
            pb = np.load(pb_path)
            if pb.shape != (record["n_residues"], record["n_residues"], 128):
                raise ValueError(f"Unexpected pair-block shape for {record['protein_id']} layer {layer}: {pb.shape}")
            pair_i, pair_j = np.tril_indices(record["n_residues"], k=-1)
            x[cursor:end] = extract_base256(pb, pair_i[indices], pair_j[indices])
            del pb
        else:
            x[cursor:end] = feature_cache[record["protein_id"]][indices]
        metadata = record["metadata"]
        y[cursor:end] = metadata["y_true"][indices]
        aa[cursor:end] = metadata["aa_pair_code"][indices]
        sequence[cursor:end] = metadata["sequence_bin_code"][indices]
        distance[cursor:end] = metadata["distance_bin_code"][indices]
        offsets.append((global_record_code, cursor, end))
        cursor = end
    return {"x": x, "y": y, "aa": aa, "sequence": sequence, "distance": distance,
            "offsets": offsets, "record_by_code": record_by_code}


def subset_materialized(materialized, row_mask):
    offsets = []
    y_pieces, pred_indices = [], []
    cursor = 0
    for protein_code, start, end in materialized["offsets"]:
        local = np.flatnonzero(row_mask[start:end]) + start
        if local.size:
            pred_indices.append(local)
            y_pieces.append(materialized["y"][local])
            offsets.append((protein_code, cursor, cursor + local.size))
            cursor += local.size
    indices = np.concatenate(pred_indices) if pred_indices else np.empty(0, dtype=np.int64)
    y = np.concatenate(y_pieces) if y_pieces else np.empty(0, dtype=np.float64)
    return indices, y, offsets


def expected_gates(analysis_name, sequence_task_code=None):
    if analysis_name == "analysis_1_sequence_gated":
        return [(None, code) for code in range(len(SEQUENCE_BIN_LABELS))]
    if analysis_name == "analysis_2_aa_pair_gated":
        return [(code, None) for code in range(len(STANDARD_AA_PAIR_LABELS))]
    if analysis_name == "analysis_4_aa_sequence_gated":
        if sequence_task_code is None:
            raise ValueError("Analysis 4 requires a sequence task code")
        return [(code, sequence_task_code) for code in range(len(STANDARD_AA_PAIR_LABELS))]
    raise ValueError(f"Unknown analysis {analysis_name}")


def gate_slug(aa_code, sequence_code):
    aa = "all" if aa_code is None else f"aa_{aa_code:03d}"
    seq = "all" if sequence_code is None else f"seq_{sequence_code}"
    return f"{aa}__{seq}"


def _identifiers(analysis_name, layer, split, aa_code, sequence_code, scope,
                 eval_aa="all", eval_sequence="all", eval_distance="all"):
    return {
        "analysis_name": analysis_name, "layer": layer, "split": split,
        "train_gate_aa_pair": "all" if aa_code is None else STANDARD_AA_PAIR_LABELS[aa_code],
        "train_gate_sequence_bin": "all" if sequence_code is None else SEQUENCE_BIN_LABELS[sequence_code],
        "evaluation_scope": scope, "eval_aa_pair": eval_aa,
        "eval_sequence_bin": eval_sequence, "eval_distance_bin": eval_distance,
    }


def evaluate_materialized(materialized, prediction, baseline_prediction, analysis_name, layer,
                          split, aa_code, sequence_code, fit_status, fit_flags,
                          n_train_pairs, n_train_proteins):
    rows, audit_arrays = [], []
    scopes = [("gate_overall", None)]
    if analysis_name == "analysis_1_sequence_gated":
        scopes.extend(("aa_pair", "aa", code) for code in range(len(STANDARD_AA_PAIR_LABELS)))
        scopes.extend(("spatial_distance", "distance", code) for code in range(len(DISTANCE_BIN_LABELS)))
    elif analysis_name == "analysis_2_aa_pair_gated":
        scopes.extend(("sequence_separation", "sequence", code) for code in range(len(SEQUENCE_BIN_LABELS)))
        scopes.extend(("spatial_distance", "distance", code) for code in range(len(DISTANCE_BIN_LABELS)))
    elif analysis_name == "analysis_4_aa_sequence_gated":
        scopes.extend(("spatial_distance", "distance", code) for code in range(len(DISTANCE_BIN_LABELS)))

    for cell_code, scope in enumerate(scopes):
        scope_name = scope[0]
        if len(scope) == 2:
            row_mask = np.ones(len(materialized["y"]), dtype=bool)
            eval_aa = eval_sequence = eval_distance = "all"
        else:
            _, field, code = scope
            row_mask = materialized[field] == code
            eval_aa = STANDARD_AA_PAIR_LABELS[code] if field == "aa" else "all"
            eval_sequence = SEQUENCE_BIN_LABELS[code] if field == "sequence" else "all"
            eval_distance = DISTANCE_BIN_LABELS[code] if field == "distance" else "all"
        indices, y, offsets = subset_materialized(materialized, row_mask)
        cell_prediction = None if prediction is None else prediction[indices]
        identifiers = _identifiers(analysis_name, layer, split, aa_code, sequence_code,
                                   scope_name, eval_aa, eval_sequence, eval_distance)
        row, audit = summarize_cell(y, cell_prediction, baseline_prediction, offsets, identifiers,
                                    fit_status, fit_flags, n_train_pairs, n_train_proteins)
        rows.append(row)
        if audit.size:
            prefix = np.full((len(audit), 1), cell_code, dtype=np.float64)
            audit_arrays.append(np.hstack((prefix, audit)))
    combined_audit = np.vstack(audit_arrays) if audit_arrays else np.empty((0, 8), dtype=np.float64)
    return rows, combined_audit


def per_protein_gate_rows(materialized, prediction, analysis_name, layer, aa_code,
                          sequence_code, fit_status):
    if prediction is None:
        return []
    rows = []
    for protein_code, start, end in materialized["offsets"]:
        record = materialized["record_by_code"][protein_code]
        y = materialized["y"][start:end]
        pred = prediction[start:end]
        moments = metrics_from_moments(cell_moments(y, pred))
        rows.append({
            "analysis_name": analysis_name, "layer": layer, "protein_id": record["protein_id"],
            "source": record["source"], "n_residues": record["n_residues"],
            "size_bin": record["size_bin"],
            "train_gate_aa_pair": "all" if aa_code is None else STANDARD_AA_PAIR_LABELS[aa_code],
            "train_gate_sequence_bin": "all" if sequence_code is None else SEQUENCE_BIN_LABELS[sequence_code],
            "fit_status": fit_status, "n_pairs": len(y), "target_variance": moments["variance"],
            "per_protein_r2": moments["r2"], "per_protein_mae": moments["mae"],
            "per_protein_rmse": moments["rmse"], "mean_true_distance": moments["mean"],
            "mean_predicted_distance": float(np.mean(pred)),
        })
    return rows


def run_analysis_task(analysis_name, task_id=None):
    mode = run_mode()
    task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if task_id is None else int(task_id)
    if analysis_name == "analysis_4_aa_sequence_gated":
        layer, sequence_task_code = joint_task_for_id(task_id, mode)
    else:
        layer, sequence_task_code = layer_for_task(task_id, mode), None
    root = suite_root(mode)
    manifest, records = load_context(root)
    intercept_baselines = load_intercept_baselines(root)
    train_records = [row for row in records if row["split"] == "train"]
    test_records = [row for row in records if row["split"] == "test"]
    gates = expected_gates(analysis_name, sequence_task_code)
    task_dir = root / analysis_name / "task_shards" / f"task_{task_id:03d}"
    gate_dir = task_dir / "gates"
    gate_dir.mkdir(parents=True, exist_ok=True)
    task_manifest_path = task_dir / "task_manifest.json"
    task_manifest = {
        "analysis_name": analysis_name, "task_id": task_id, "layer": layer,
        "sequence_task_code": sequence_task_code, "run_mode": mode,
        "configuration_digest": manifest["configuration_digest"],
        "split_digest": manifest["split_digest"],
        "expected_gate_keys": [gate_slug(*gate) for gate in gates],
        "train_protein_ids": [row["protein_id"] for row in train_records],
        "test_protein_ids": [row["protein_id"] for row in test_records],
        "completed_gate_keys": [], "started_at": utc_now(), "completed": False,
    }
    if task_manifest_path.exists():
        previous = json.loads(task_manifest_path.read_text())
        for key in (
            "analysis_name", "task_id", "layer", "sequence_task_code", "run_mode",
            "configuration_digest", "split_digest", "expected_gate_keys",
            "train_protein_ids", "test_protein_ids",
        ):
            if previous.get(key) != task_manifest[key]:
                raise ValueError(f"Existing task manifest mismatch for {key}")
        task_manifest = previous
    atomic_write_json(task_manifest_path, task_manifest)

    feature_cache = None
    for gate_number, (aa_code, sequence_code) in enumerate(gates, start=1):
        started = time.monotonic()
        slug = gate_slug(aa_code, sequence_code)
        result_path = gate_dir / f"{slug}.json"
        coefficient_path = gate_dir / f"{slug}_coefficients.npz"
        audit_path = gate_dir / f"{slug}_audit.npz"
        if slug in task_manifest["completed_gate_keys"]:
            artifacts = task_manifest.get("artifacts", {}).get(slug, {})
            if (result_path.exists() and coefficient_path.exists() and audit_path.exists()
                    and artifacts.get("result_sha256") == sha256_file(result_path)
                    and artifacts.get("coefficient_sha256") == sha256_file(coefficient_path)
                    and artifacts.get("audit_sha256") == sha256_file(audit_path)):
                log(f"{analysis_name} layer {layer}: skipping verified gate {slug}")
                continue
            raise ValueError(f"Completed gate {slug} has missing or mismatched artifacts")
        if feature_cache is None:
            log(f"{analysis_name} layer {layer}: building shared float32 feature cache")
            feature_cache = load_layer_feature_cache(records, layer)
        log(f"{analysis_name} layer {layer}: gate {gate_number}/{len(gates)} {slug}")
        train = materialize_gate(train_records, layer, aa_code, sequence_code, feature_cache)
        test = materialize_gate(test_records, layer, aa_code, sequence_code, feature_cache)
        n_train_pairs = len(train["y"])
        n_train_proteins = len(train["offsets"])
        log(f"{slug}: materialized {n_train_pairs:,} train and {len(test['y']):,} test rows")
        fit_status, fit_flags = gate_eligibility(n_train_pairs, n_train_proteins)
        fitted = None
        train_prediction = test_prediction = None
        if fit_status == "eligible":
            log(f"{slug}: fitting float64 OLS")
            fitted = fit_ols(train["x"], train["y"], train["offsets"])
            fit_status, fit_flags = fitted["fit_status"], fitted["flags"]
            train_prediction = fitted["model"].predict((train["x"] - fitted["mean"]) / fitted["scale"])
            test_prediction = (
                fitted["model"].predict((test["x"] - fitted["mean"]) / fitted["scale"])
                if len(test["y"]) else np.empty(0, dtype=np.float64)
            )
            log(f"{slug}: fit status {fit_status}")
        baseline_key = (
            analysis_name,
            "all" if aa_code is None else STANDARD_AA_PAIR_LABELS[aa_code],
            "all" if sequence_code is None else SEQUENCE_BIN_LABELS[sequence_code],
        )
        baseline_record = intercept_baselines[baseline_key]
        if (baseline_record["n_train_pairs"] != n_train_pairs
                or baseline_record["n_train_proteins"] != n_train_proteins):
            raise ValueError(f"Prepared baseline support mismatch for {baseline_key}")
        baseline_prediction = baseline_record["prediction"]
        log(f"{slug}: evaluating train/test scopes")
        train_rows, train_audit = evaluate_materialized(
            train, train_prediction, baseline_prediction, analysis_name, layer, "train",
            aa_code, sequence_code, fit_status, fit_flags, n_train_pairs, n_train_proteins,
        )
        test_rows, test_audit = evaluate_materialized(
            test, test_prediction, baseline_prediction, analysis_name, layer, "test",
            aa_code, sequence_code, fit_status, fit_flags, n_train_pairs, n_train_proteins,
        )
        per_protein = per_protein_gate_rows(
            test, test_prediction, analysis_name, layer, aa_code, sequence_code, fit_status,
        )
        status_row = {
            "analysis_name": analysis_name, "model": "base256_ols", "feature_dim": FEATURE_DIM,
            "layer": layer, "train_gate_aa_pair": "all" if aa_code is None else STANDARD_AA_PAIR_LABELS[aa_code],
            "train_gate_sequence_bin": "all" if sequence_code is None else SEQUENCE_BIN_LABELS[sequence_code],
            "fit_status": fit_status, "support_flags": _join_flags(fit_flags),
            "n_train_pairs": n_train_pairs, "n_train_proteins": n_train_proteins,
            "design_rank": fitted["design_rank"] if fitted else -1,
            "largest_singular_value": fitted["largest_singular_value"] if fitted else float("nan"),
            "smallest_singular_value": fitted["smallest_singular_value"] if fitted else float("nan"),
            "condition_number": fitted["condition_number"] if fitted else float("nan"),
            "elapsed_seconds": time.monotonic() - started,
        }
        result = {"gate_slug": slug, "metric_rows": train_rows + test_rows,
                  "per_protein_rows": per_protein, "gate_status": status_row}
        atomic_write_json(result_path, result)
        if fitted:
            atomic_savez(coefficient_path, coef_std=fitted["coef_std"], coef_raw=fitted["coef_raw"],
                         intercept_std=np.asarray(fitted["intercept_std"]),
                         intercept_raw=np.asarray(fitted["intercept_raw"]),
                         scaler_mean=fitted["mean"], scaler_scale=fitted["scale"],
                         fit_status=np.asarray(fit_status))
        else:
            atomic_savez(coefficient_path, coef_std=np.empty(0), coef_raw=np.empty(0),
                         intercept_std=np.asarray(float("nan")), intercept_raw=np.asarray(float("nan")),
                         scaler_mean=np.empty(0), scaler_scale=np.empty(0), fit_status=np.asarray(fit_status))
        atomic_savez(
            audit_path,
            train_cell_code=train_audit[:, 0].astype(np.int32),
            train_protein_code=train_audit[:, 1].astype(np.int32),
            train_moments=train_audit[:, 2:].astype(np.float64),
            test_cell_code=test_audit[:, 0].astype(np.int32),
            test_protein_code=test_audit[:, 1].astype(np.int32),
            test_moments=test_audit[:, 2:].astype(np.float64),
            moment_columns=np.asarray(["n", "sum_y", "sum_y_squared", "sum_prediction",
                                       "sum_absolute_error", "sum_squared_error"]),
        )
        task_manifest.setdefault("artifacts", {})[slug] = {
            "result_sha256": sha256_file(result_path),
            "coefficient_sha256": sha256_file(coefficient_path),
            "audit_sha256": sha256_file(audit_path),
        }
        task_manifest["completed_gate_keys"].append(slug)
        atomic_write_json(task_manifest_path, task_manifest)
        log(f"{analysis_name} layer {layer}: checkpointed {slug}")
        del train, test, fitted, train_prediction, test_prediction
    task_manifest["completed"] = True
    task_manifest["completed_at"] = utc_now()
    atomic_write_json(task_manifest_path, task_manifest)
    log(f"Completed {analysis_name} task {task_id}")


def validate_unique_rows(rows, key_fields, table_name):
    seen = set()
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key in seen:
            raise ValueError(f"Duplicate key in {table_name}: {key}")
        seen.add(key)

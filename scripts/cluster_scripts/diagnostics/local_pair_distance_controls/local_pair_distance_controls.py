#!/usr/bin/env python3
"""Compare consecutive-layer distance readouts with calibration controls.

One Delta array task handles one transition x -> x+1 for x in 10..15. The
script uses the frozen general_base256_combined split and saved layer-specific
coefficients, computes train-only moment alignment, fits one protein-equal
two-layer pooled OLS model, and evaluates four predictive variants on all
unordered pairs and on local pairs with sequence separation 4..12.

All materialized features, sufficient statistics, fits, and predictions use
float64. Predictions are raw pre-MDS values and are never clipped.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import sys
import time
import traceback
import zlib
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parents[1]
GENERALIZED_DIR = CLUSTER_SCRIPTS_DIR / "generalized_model_ecod"
for _path in (CLUSTER_SCRIPTS_DIR, GENERALIZED_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cluster_regression_helpers import extract_ca_coordinates  # noqa: E402
from general_train_test_common import (  # noqa: E402
    N_CHANNELS,
    SPLIT_ID,
    build_dataset_protein_manifest,
    feature_names_for_variant,
    make_variant,
    resolve_pair_block_path,
    resolve_pdb_file,
)


SOURCE_VARIANT = "general_base256_combined"
VARIANT_SPEC = make_variant(SOURCE_VARIANT, pair_mode="concat_directional")
FEATURE_DIM = 2 * N_CHANNELS
AUGMENTED_DIM = FEATURE_DIM + 1
START_LAYERS = tuple(range(10, 16))
PREDICTIVE_VARIANTS = (
    "fixed_raw_layer_x",
    "fixed_moment_aligned_to_x",
    "saved_own_layer",
    "pooled_two_layer",
)
SCOPES = ("all", "local_4_12")
SPLITS = ("train", "test")
AGGREGATIONS = ("pair_weighted", "protein_equal")
MIN_SEQUENCE_SEPARATION = 4
MAX_SEQUENCE_SEPARATION = 12
CHANGE_THRESHOLD = 0.5
FEATURE_CHUNK_SIZE = 100_000
FLOAT_EPS = 1e-12
QUANTILE_SAMPLE_SIZE = 200_000
QUANTILE_SEED = 20260812
OUTPUT_NAME = "local_pair_distance_controls_layers10_16"

PROBE_PERFORMANCE_FIELDS = [
    "layer_from", "layer_to", "variant", "endpoint", "layer", "split",
    "pair_scope", "n_proteins", "n_pairs", "pair_weighted_r2",
    "pair_weighted_mae", "pair_weighted_rmse", "pair_weighted_bias",
    "mean_per_protein_r2", "median_per_protein_r2",
]
DISTANCE_STATISTIC_FIELDS = [
    "layer_from", "layer_to", "variant", "endpoint", "layer", "split",
    "pair_scope", "aggregation", "n_proteins", "n_values", "mean_predicted",
    "std_predicted", "q01_predicted", "q05_predicted", "q25_predicted",
    "median_predicted", "q75_predicted", "q95_predicted", "q99_predicted",
    "minimum_predicted", "maximum_predicted", "fraction_negative_predicted",
    "mean_true", "std_true", "mean_bias", "quantile_method",
]
CHANGE_STATISTIC_FIELDS = [
    "layer_from", "layer_to", "variant", "split", "pair_scope",
    "aggregation", "n_proteins", "n_values", "mean", "std", "mean_absolute",
    "rms", "q01", "q05", "q25", "median", "q75", "q95", "q99",
    "minimum", "maximum", "fraction_decrease", "fraction_no_change",
    "fraction_increase", "threshold_angstrom", "quantile_method",
]
VARIANT_COMPARISON_FIELDS = [
    "layer_from", "layer_to", "variant", "reference_variant", "split",
    "pair_scope", "aggregation", "n_proteins", "n_values",
    "difference_in_mean_change", "change_rmse_vs_reference",
    "change_pearson_r_vs_reference", "sign_agreement",
    "threshold_class_agreement", "threshold_angstrom",
]
PER_PROTEIN_FIELDS = [
    "layer_from", "layer_to", "variant", "split", "source", "protein_id",
    "n_residues", "pair_scope", "n_pairs", "from_r2", "from_mae",
    "from_rmse", "from_bias", "to_r2", "to_mae", "to_rmse", "to_bias",
    "mean_true_distance", "mean_predicted_from", "mean_predicted_to",
    "mean_change", "std_change", "mean_absolute_change", "rms_change",
    "fraction_decrease", "fraction_no_change", "fraction_increase",
]
DRIFT_CHANNEL_FIELDS = [
    "layer_from", "layer_to", "split", "pair_scope", "aggregation",
    "feature_index", "feature_name", "mean_from", "mean_to", "mean_shift",
    "std_from", "std_to", "scale_ratio", "standardized_mean_shift",
    "coefficient_layer_x", "beta_mean_shift_contribution",
    "beta_scale_change", "from_zero_variance", "to_zero_variance",
]
DRIFT_SUMMARY_FIELDS = [
    "layer_from", "layer_to", "split", "pair_scope", "aggregation",
    "n_proteins", "n_pairs", "mean_fixed_raw_change",
    "sum_beta_mean_shift_contribution", "reconstruction_error",
    "mean_absolute_standardized_mean_shift", "maximum_absolute_standardized_mean_shift",
    "mean_absolute_log_scale_ratio", "maximum_absolute_log_scale_ratio",
    "n_from_zero_variance", "n_to_zero_variance",
]
ALIGNMENT_FIELDS = [
    "layer_from", "layer_to", "feature_index", "feature_name", "mean_from",
    "scale_from", "mean_to", "scale_to", "safe_scale_from", "safe_scale_to",
    "multiplier", "offset", "from_zero_variance", "to_zero_variance",
]
POOLED_COEFFICIENT_FIELDS = [
    "layer_from", "layer_to", "feature_index", "feature_name",
    "coefficient_standardized", "coefficient_raw_scale", "scaler_mean",
    "scaler_scale",
]
POOLED_FIT_FIELDS = [
    "layer_from", "layer_to", "n_train_proteins", "n_layer_observations",
    "rank", "normal_matrix_condition", "estimated_design_condition",
    "scale_min", "scale_max", "intercept_standardized", "intercept_raw",
    "coefficient_l2_raw", "elapsed_seconds",
]
MANIFEST_FIELDS = [
    "layer_from", "layer_to", "split", "source", "protein_id", "n_residues",
    "n_all_pairs", "n_local_pairs", "pair_block_from", "pair_block_to",
    "pdb_path", "elapsed_seconds",
]
TIMING_FIELDS = ["layer_from", "layer_to", "stage", "elapsed_seconds"]


def now_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"[{now_timestamp()}] {message}", flush=True)


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return "nan"
        return f"{value:.9g}"
    return value


def write_csv_atomic(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})
    os.replace(temporary, path)


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_text_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def requested_start_layer():
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None or raw == "":
        raise ValueError("SLURM_ARRAY_TASK_ID is required and must be in 10..15")
    layer = int(raw)
    if layer not in START_LAYERS:
        raise ValueError(f"SLURM_ARRAY_TASK_ID={layer} is outside 10..15")
    return layer


def lower_triangle_indices(n_residues):
    hi, lo = np.tril_indices(int(n_residues), k=-1)
    return np.asarray(hi, dtype=np.int64), np.asarray(lo, dtype=np.int64)


def local_pair_mask(hi, lo):
    separation = np.asarray(hi, dtype=np.int64) - np.asarray(lo, dtype=np.int64)
    return (separation >= MIN_SEQUENCE_SEPARATION) & (
        separation <= MAX_SEQUENCE_SEPARATION
    )


def directional_features(pb, hi, lo):
    """Return [pb[hi,lo], pb[lo,hi]] in the saved probe's feature order."""
    values = np.empty((len(hi), FEATURE_DIM), dtype=np.float64)
    values[:, :N_CHANNELS] = pb[hi, lo, :]
    values[:, N_CHANNELS:] = pb[lo, hi, :]
    return values


def load_saved_models(path, layers):
    expected_names = feature_names_for_variant(VARIANT_SPEC)
    requested = set(int(layer) for layer in layers)
    models = {
        layer: {
            "intercept": None,
            "coef": np.full(FEATURE_DIM, np.nan, dtype=np.float64),
            "rows": [],
        }
        for layer in requested
    }
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != SOURCE_VARIANT or int(row["split_id"]) != SPLIT_ID:
                continue
            layer = int(row["layer"])
            if layer not in requested:
                continue
            if int(row["feature_dim"]) != FEATURE_DIM:
                raise ValueError(f"Unexpected feature dimension in {path}: {row}")
            index = int(row["feature_index"])
            value = float(row["coefficient_raw_scale"])
            models[layer]["rows"].append(dict(row))
            if index == -1:
                if models[layer]["intercept"] is not None:
                    raise ValueError(f"Duplicate intercept for layer {layer}")
                models[layer]["intercept"] = value
            else:
                if index < 0 or index >= FEATURE_DIM:
                    raise ValueError(f"Invalid feature index {index} at layer {layer}")
                if row["feature_name"] != expected_names[index]:
                    raise ValueError(
                        f"Feature {index} is {row['feature_name']}, expected "
                        f"{expected_names[index]}"
                    )
                if math.isfinite(models[layer]["coef"][index]):
                    raise ValueError(f"Duplicate feature {index} at layer {layer}")
                models[layer]["coef"][index] = value
    for layer, model in models.items():
        if model["intercept"] is None or not np.all(np.isfinite(model["coef"])):
            raise ValueError(f"Incomplete saved coefficient bundle for layer {layer}")
    return models


def load_records(assignment_path):
    manifest = {row["protein_id"]: row for row in build_dataset_protein_manifest()}
    records = []
    assignment_rows = []
    seen = set()
    with Path(assignment_path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["split_id"]) != SPLIT_ID:
                continue
            protein_id = row["protein_id"]
            if protein_id in seen:
                raise ValueError(f"Duplicate split assignment for {protein_id}")
            entry = manifest.get(protein_id)
            if entry is None:
                raise ValueError(f"{protein_id} is absent from the combined manifest")
            if row["split"] not in SPLITS:
                raise ValueError(f"Unexpected split {row['split']!r}")
            records.append(
                {
                    "protein_id": protein_id,
                    "source": entry["source"],
                    "protein_dir": entry["protein_dir"],
                    "pdb_root": entry["pdb_root"],
                    "split": row["split"],
                    "n_residues": int(row["n_residues"]),
                    "n_pairs": int(row["n_pairs"]),
                }
            )
            assignment_rows.append(dict(row))
            seen.add(protein_id)
    if not records:
        raise ValueError(f"No split_id={SPLIT_ID} assignments in {assignment_path}")
    return records, assignment_rows


def resolve_inputs(record, layer_from, layer_to):
    from_path = resolve_pair_block_path(
        record["protein_dir"], record["protein_id"], layer_from
    )
    to_path = resolve_pair_block_path(
        record["protein_dir"], record["protein_id"], layer_to
    )
    pdb_path = resolve_pdb_file(
        record["protein_id"], record["protein_dir"], record["pdb_root"]
    )
    if not from_path.exists() or not to_path.exists():
        raise FileNotFoundError(f"Missing pair block: {from_path}, {to_path}")
    if pdb_path is None:
        raise FileNotFoundError(f"Missing predicted PDB for {record['protein_id']}")
    return from_path, to_path, pdb_path


def validate_pair_block(pb, record, layer):
    expected = (record["n_residues"], record["n_residues"], N_CHANNELS)
    if tuple(pb.shape) != expected:
        raise ValueError(
            f"{record['protein_id']} layer {layer} shape {pb.shape} != {expected}"
        )


def load_targets(record, pdb_path, hi, lo):
    ca_data = extract_ca_coordinates(pdb_path)
    coordinates = np.asarray(ca_data["coordinates"], dtype=np.float64)
    if coordinates.shape != (record["n_residues"], 3):
        raise ValueError(
            f"{record['protein_id']} CA coordinates {coordinates.shape} != "
            f"({record['n_residues']}, 3)"
        )
    delta = coordinates[hi] - coordinates[lo]
    return np.sqrt(np.sum(delta * delta, axis=1, dtype=np.float64))


def per_protein_layer_statistics(pb, hi, lo, target):
    """Return feature moments and augmented OLS statistics for one protein."""
    n_pairs = int(len(hi))
    feature_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
    feature_sq_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
    xx_sum = np.zeros((FEATURE_DIM, FEATURE_DIM), dtype=np.float64)
    xy_sum = np.zeros(FEATURE_DIM, dtype=np.float64)
    for start in range(0, n_pairs, FEATURE_CHUNK_SIZE):
        end = min(start + FEATURE_CHUNK_SIZE, n_pairs)
        x = directional_features(pb, hi[start:end], lo[start:end])
        y = np.asarray(target[start:end], dtype=np.float64)
        feature_sum += np.sum(x, axis=0, dtype=np.float64)
        feature_sq_sum += np.sum(x * x, axis=0, dtype=np.float64)
        xx_sum += x.T @ x
        xy_sum += x.T @ y
        del x, y
    mean = feature_sum / n_pairs
    second = feature_sq_sum / n_pairs
    gram = np.empty((AUGMENTED_DIM, AUGMENTED_DIM), dtype=np.float64)
    gram[0, 0] = 1.0
    gram[0, 1:] = mean
    gram[1:, 0] = mean
    gram[1:, 1:] = xx_sum / n_pairs
    cross = np.empty(AUGMENTED_DIM, dtype=np.float64)
    cross[0] = float(np.mean(target))
    cross[1:] = xy_sum / n_pairs
    return mean, second, gram, cross


def safe_scales(mean, second):
    raw = np.sqrt(np.maximum(np.asarray(second) - np.asarray(mean) ** 2, 0.0))
    zero = raw < FLOAT_EPS
    safe = raw.copy()
    safe[zero] = 1.0
    return raw, safe, zero


def alignment_from_moments(mean_from, second_from, mean_to, second_to):
    raw_from, safe_from, zero_from = safe_scales(mean_from, second_from)
    raw_to, safe_to, zero_to = safe_scales(mean_to, second_to)
    multiplier = safe_from / safe_to
    offset = mean_from - multiplier * mean_to
    return {
        "mean_from": np.asarray(mean_from, dtype=np.float64),
        "mean_to": np.asarray(mean_to, dtype=np.float64),
        "scale_from": raw_from,
        "scale_to": raw_to,
        "safe_scale_from": safe_from,
        "safe_scale_to": safe_to,
        "zero_from": zero_from,
        "zero_to": zero_to,
        "multiplier": multiplier,
        "offset": offset,
    }


def solve_sufficient_statistics(gram, cross):
    """Solve protein-equal OLS after moment-derived standardization."""
    gram = 0.5 * (np.asarray(gram, dtype=np.float64) + np.asarray(gram).T)
    cross = np.asarray(cross, dtype=np.float64)
    mass = float(gram[0, 0])
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Invalid regression mass {mass}")
    mean = gram[0, 1:] / mass
    second = np.diag(gram)[1:] / mass
    _, scale, _ = safe_scales(mean, second)
    transform = np.zeros((AUGMENTED_DIM, AUGMENTED_DIM), dtype=np.float64)
    transform[0, 0] = 1.0
    indices = np.arange(FEATURE_DIM)
    transform[0, 1:] = -mean / scale
    transform[indices + 1, indices + 1] = 1.0 / scale
    standardized_gram = transform.T @ gram @ transform
    standardized_gram = 0.5 * (standardized_gram + standardized_gram.T)
    standardized_cross = transform.T @ cross
    theta_std, _, rank, singular = np.linalg.lstsq(
        standardized_gram, standardized_cross, rcond=None
    )
    theta_raw = transform @ theta_std
    positive = singular[singular > 0.0]
    condition = (
        float(positive[0] / positive[-1]) if positive.size else float("inf")
    )
    return {
        "intercept_std": float(theta_std[0]),
        "coef_std": np.asarray(theta_std[1:], dtype=np.float64),
        "intercept": float(theta_raw[0]),
        "coef": np.asarray(theta_raw[1:], dtype=np.float64),
        "mean": mean,
        "scale": scale,
        "rank": int(rank),
        "normal_matrix_condition": condition,
        "estimated_design_condition": math.sqrt(condition),
    }


def predict_raw(features, model):
    return float(model["intercept"]) + np.asarray(features) @ model["coef"]


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    centered = y_true - np.mean(y_true)
    denominator = float(np.sum(centered * centered, dtype=np.float64))
    if denominator <= FLOAT_EPS:
        return float("nan")
    error = y_pred - y_true
    return 1.0 - float(np.sum(error * error, dtype=np.float64)) / denominator


def class_values(changes):
    changes = np.asarray(changes, dtype=np.float64)
    labels = np.zeros(changes.shape, dtype=np.int8)
    labels[changes < -CHANGE_THRESHOLD] = -1
    labels[changes > CHANGE_THRESHOLD] = 1
    return labels


def sign_values(changes):
    return np.sign(np.asarray(changes, dtype=np.float64)).astype(np.int8)


def stable_group_seed(parts):
    label = "|".join(str(part) for part in parts).encode("utf-8")
    return QUANTILE_SEED + int(zlib.crc32(label))


def basic_error_metrics(y_true, y_pred):
    error = np.asarray(y_pred, dtype=np.float64) - np.asarray(y_true, dtype=np.float64)
    return {
        "r2": safe_r2(y_true, y_pred),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "bias": float(np.mean(error)),
    }


class Reservoir:
    def __init__(self, capacity, seed):
        self.capacity = int(capacity)
        self.seed = int(seed)
        self.rng = np.random.default_rng(self.seed)
        self.values = np.empty(self.capacity, dtype=np.float64)
        self.priorities = np.empty(self.capacity, dtype=np.float64)
        self.size = 0
        self.seen = 0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        new_priorities = self.rng.random(values.size)
        candidate_values = np.concatenate((self.values[: self.size], values))
        candidate_priorities = np.concatenate(
            (self.priorities[: self.size], new_priorities)
        )
        keep_count = min(self.capacity, candidate_values.size)
        if keep_count < candidate_values.size:
            keep = np.argpartition(candidate_priorities, keep_count - 1)[:keep_count]
            self.values[:keep_count] = candidate_values[keep]
            self.priorities[:keep_count] = candidate_priorities[keep]
        else:
            self.values[:keep_count] = candidate_values
            self.priorities[:keep_count] = candidate_priorities
        self.size = keep_count
        self.seen += int(values.size)

    def array(self):
        return self.values[: self.size]


class DistributionAccumulator:
    def __init__(self, seed):
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0
        self.total_abs = 0.0
        self.minimum = float("inf")
        self.maximum = float("-inf")
        self.negative = 0
        self.decrease = 0
        self.no_change = 0
        self.increase = 0
        self.reservoir = Reservoir(QUANTILE_SAMPLE_SIZE, seed)

    def update(self, values, classify=False):
        values = np.asarray(values, dtype=np.float64).ravel()
        if values.size == 0:
            return
        self.n += int(values.size)
        self.total += float(np.sum(values, dtype=np.float64))
        self.total_sq += float(np.sum(values * values, dtype=np.float64))
        self.total_abs += float(np.sum(np.abs(values), dtype=np.float64))
        self.minimum = min(self.minimum, float(np.min(values)))
        self.maximum = max(self.maximum, float(np.max(values)))
        self.negative += int(np.count_nonzero(values < 0.0))
        if classify:
            labels = class_values(values)
            self.decrease += int(np.count_nonzero(labels == -1))
            self.no_change += int(np.count_nonzero(labels == 0))
            self.increase += int(np.count_nonzero(labels == 1))
        self.reservoir.update(values)

    def summary(self):
        if self.n == 0:
            return {key: float("nan") for key in (
                "mean", "std", "mean_absolute", "rms", "q01", "q05", "q25",
                "median", "q75", "q95", "q99", "minimum", "maximum"
            )}
        mean = self.total / self.n
        variance = max(self.total_sq / self.n - mean * mean, 0.0)
        q = np.quantile(
            self.reservoir.array(), [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99]
        )
        return {
            "mean": mean,
            "std": math.sqrt(variance),
            "mean_absolute": self.total_abs / self.n,
            "rms": math.sqrt(self.total_sq / self.n),
            "q01": float(q[0]), "q05": float(q[1]), "q25": float(q[2]),
            "median": float(q[3]), "q75": float(q[4]), "q95": float(q[5]),
            "q99": float(q[6]), "minimum": self.minimum, "maximum": self.maximum,
        }


class ErrorAccumulator:
    def __init__(self):
        self.n = 0
        self.sum_y = 0.0
        self.sum_y2 = 0.0
        self.sum_error = 0.0
        self.sum_abs_error = 0.0
        self.sum_sq_error = 0.0

    def update(self, y_true, y_pred):
        y_true = np.asarray(y_true, dtype=np.float64)
        error = np.asarray(y_pred, dtype=np.float64) - y_true
        self.n += int(y_true.size)
        self.sum_y += float(np.sum(y_true, dtype=np.float64))
        self.sum_y2 += float(np.sum(y_true * y_true, dtype=np.float64))
        self.sum_error += float(np.sum(error, dtype=np.float64))
        self.sum_abs_error += float(np.sum(np.abs(error), dtype=np.float64))
        self.sum_sq_error += float(np.sum(error * error, dtype=np.float64))

    def summary(self):
        mean_y = self.sum_y / max(self.n, 1)
        ss_total = self.sum_y2 - 2 * mean_y * self.sum_y + self.n * mean_y**2
        return {
            "r2": 1.0 - self.sum_sq_error / ss_total if ss_total > FLOAT_EPS else float("nan"),
            "mae": self.sum_abs_error / max(self.n, 1),
            "rmse": math.sqrt(self.sum_sq_error / max(self.n, 1)),
            "bias": self.sum_error / max(self.n, 1),
        }


class MomentAccumulator:
    def __init__(self):
        self.n = 0
        self.total = 0.0
        self.total_sq = 0.0

    def update(self, values):
        values = np.asarray(values, dtype=np.float64)
        self.n += int(values.size)
        self.total += float(np.sum(values, dtype=np.float64))
        self.total_sq += float(np.sum(values * values, dtype=np.float64))

    def summary(self):
        if self.n == 0:
            return {"mean": float("nan"), "std": float("nan")}
        mean = self.total / self.n
        return {
            "mean": mean,
            "std": math.sqrt(max(self.total_sq / self.n - mean * mean, 0.0)),
        }


class ComparisonAccumulator:
    def __init__(self):
        self.n = 0
        self.sum_control = 0.0
        self.sum_reference = 0.0
        self.sum_c2 = 0.0
        self.sum_r2 = 0.0
        self.sum_cr = 0.0
        self.sum_sq_difference = 0.0
        self.sign_agree = 0
        self.class_agree = 0

    def update(self, control, reference):
        control = np.asarray(control, dtype=np.float64)
        reference = np.asarray(reference, dtype=np.float64)
        self.n += int(control.size)
        self.sum_control += float(np.sum(control))
        self.sum_reference += float(np.sum(reference))
        self.sum_c2 += float(np.sum(control * control))
        self.sum_r2 += float(np.sum(reference * reference))
        self.sum_cr += float(np.sum(control * reference))
        self.sum_sq_difference += float(np.sum((control - reference) ** 2))
        self.sign_agree += int(np.count_nonzero(sign_values(control) == sign_values(reference)))
        self.class_agree += int(np.count_nonzero(class_values(control) == class_values(reference)))

    def summary(self):
        if self.n == 0:
            return {key: float("nan") for key in (
                "difference_in_mean_change", "rmse", "pearson", "sign_agreement",
                "class_agreement"
            )}
        mean_c = self.sum_control / self.n
        mean_r = self.sum_reference / self.n
        var_c = max(self.sum_c2 / self.n - mean_c**2, 0.0)
        var_r = max(self.sum_r2 / self.n - mean_r**2, 0.0)
        covariance = self.sum_cr / self.n - mean_c * mean_r
        denominator = math.sqrt(var_c * var_r)
        return {
            "difference_in_mean_change": mean_c - mean_r,
            "rmse": math.sqrt(self.sum_sq_difference / self.n),
            "pearson": covariance / denominator if denominator > FLOAT_EPS else float("nan"),
            "sign_agreement": self.sign_agree / self.n,
            "class_agreement": self.class_agree / self.n,
        }


class FeatureMomentAccumulator:
    def __init__(self):
        self.pair_count = 0
        self.protein_count = 0
        self.pair_sum_from = np.zeros(FEATURE_DIM, dtype=np.float64)
        self.pair_sum_to = np.zeros(FEATURE_DIM, dtype=np.float64)
        self.pair_sq_from = np.zeros(FEATURE_DIM, dtype=np.float64)
        self.pair_sq_to = np.zeros(FEATURE_DIM, dtype=np.float64)
        self.protein_mean_from = np.zeros(FEATURE_DIM, dtype=np.float64)
        self.protein_mean_to = np.zeros(FEATURE_DIM, dtype=np.float64)
        self.protein_second_from = np.zeros(FEATURE_DIM, dtype=np.float64)
        self.protein_second_to = np.zeros(FEATURE_DIM, dtype=np.float64)

    def update(self, from_features, to_features):
        from_features = np.asarray(from_features, dtype=np.float64)
        to_features = np.asarray(to_features, dtype=np.float64)
        n = int(from_features.shape[0])
        if n == 0:
            return
        from_sum = np.sum(from_features, axis=0, dtype=np.float64)
        to_sum = np.sum(to_features, axis=0, dtype=np.float64)
        from_sq = np.sum(from_features * from_features, axis=0, dtype=np.float64)
        to_sq = np.sum(to_features * to_features, axis=0, dtype=np.float64)
        self.update_sums(n, from_sum, from_sq, to_sum, to_sq)

    def update_sums(self, n, from_sum, from_sq, to_sum, to_sq):
        n = int(n)
        if n == 0:
            return
        self.pair_count += n
        self.protein_count += 1
        self.pair_sum_from += from_sum
        self.pair_sum_to += to_sum
        self.pair_sq_from += from_sq
        self.pair_sq_to += to_sq
        self.protein_mean_from += from_sum / n
        self.protein_mean_to += to_sum / n
        self.protein_second_from += from_sq / n
        self.protein_second_to += to_sq / n

    def moments(self, aggregation):
        if aggregation == "pair_weighted":
            denominator = max(self.pair_count, 1)
            return (
                self.pair_sum_from / denominator,
                self.pair_sq_from / denominator,
                self.pair_sum_to / denominator,
                self.pair_sq_to / denominator,
            )
        denominator = max(self.protein_count, 1)
        return (
            self.protein_mean_from / denominator,
            self.protein_second_from / denominator,
            self.protein_mean_to / denominator,
            self.protein_second_to / denominator,
        )


def per_protein_row(layer_from, variant, record, scope, y, pred_from, pred_to):
    from_metrics = basic_error_metrics(y, pred_from)
    to_metrics = basic_error_metrics(y, pred_to)
    change = np.asarray(pred_to) - np.asarray(pred_from)
    labels = class_values(change)
    return {
        "layer_from": layer_from, "layer_to": layer_from + 1, "variant": variant,
        "split": record["split"], "source": record["source"],
        "protein_id": record["protein_id"], "n_residues": record["n_residues"],
        "pair_scope": scope, "n_pairs": int(y.size),
        "from_r2": from_metrics["r2"], "from_mae": from_metrics["mae"],
        "from_rmse": from_metrics["rmse"], "from_bias": from_metrics["bias"],
        "to_r2": to_metrics["r2"], "to_mae": to_metrics["mae"],
        "to_rmse": to_metrics["rmse"], "to_bias": to_metrics["bias"],
        "mean_true_distance": float(np.mean(y)),
        "mean_predicted_from": float(np.mean(pred_from)),
        "mean_predicted_to": float(np.mean(pred_to)),
        "mean_change": float(np.mean(change)),
        "std_change": float(np.std(change)),
        "mean_absolute_change": float(np.mean(np.abs(change))),
        "rms_change": float(np.sqrt(np.mean(change * change))),
        "fraction_decrease": float(np.mean(labels == -1)),
        "fraction_no_change": float(np.mean(labels == 0)),
        "fraction_increase": float(np.mean(labels == 1)),
    }


def build_alignment_rows(layer_from, alignment):
    names = feature_names_for_variant(VARIANT_SPEC)
    rows = []
    for index, name in enumerate(names):
        rows.append({
            "layer_from": layer_from, "layer_to": layer_from + 1,
            "feature_index": index, "feature_name": name,
            "mean_from": alignment["mean_from"][index],
            "scale_from": alignment["scale_from"][index],
            "mean_to": alignment["mean_to"][index],
            "scale_to": alignment["scale_to"][index],
            "safe_scale_from": alignment["safe_scale_from"][index],
            "safe_scale_to": alignment["safe_scale_to"][index],
            "multiplier": alignment["multiplier"][index],
            "offset": alignment["offset"][index],
            "from_zero_variance": alignment["zero_from"][index],
            "to_zero_variance": alignment["zero_to"][index],
        })
    return rows


def build_pooled_rows(layer_from, pooled):
    names = feature_names_for_variant(VARIANT_SPEC)
    rows = [{
        "layer_from": layer_from, "layer_to": layer_from + 1,
        "feature_index": -1, "feature_name": "intercept",
        "coefficient_standardized": pooled["intercept_std"],
        "coefficient_raw_scale": pooled["intercept"],
        "scaler_mean": 0.0, "scaler_scale": 1.0,
    }]
    for index, name in enumerate(names):
        rows.append({
            "layer_from": layer_from, "layer_to": layer_from + 1,
            "feature_index": index, "feature_name": name,
            "coefficient_standardized": pooled["coef_std"][index],
            "coefficient_raw_scale": pooled["coef"][index],
            "scaler_mean": pooled["mean"][index],
            "scaler_scale": pooled["scale"][index],
        })
    return rows


def statistics_from_values(values):
    values = np.asarray(values, dtype=np.float64)
    if values.size == 0:
        return {key: float("nan") for key in (
            "mean", "std", "mean_absolute", "rms", "q01", "q05", "q25",
            "median", "q75", "q95", "q99", "minimum", "maximum"
        )}
    q = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
    return {
        "mean": float(np.mean(values)), "std": float(np.std(values)),
        "mean_absolute": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values * values))),
        "q01": float(q[0]), "q05": float(q[1]), "q25": float(q[2]),
        "median": float(q[3]), "q75": float(q[4]), "q95": float(q[5]),
        "q99": float(q[6]), "minimum": float(np.min(values)),
        "maximum": float(np.max(values)),
    }


def main():
    total_t0 = time.time()
    layer_from = requested_start_layer()
    layer_to = layer_from + 1
    csv_root = Path(os.environ.get("CSV_ROOT") or (Path.home() / "csv_files")).expanduser()
    source_dir = csv_root / SOURCE_VARIANT
    output_root = Path(
        os.environ.get("LOCAL_PAIR_DISTANCE_CONTROLS_OUTPUT_ROOT")
        or (csv_root / OUTPUT_NAME)
    ).expanduser()
    output_dir = output_root / "shards" / f"transition_{layer_from:02d}_{layer_to:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)
    metadata_path = output_dir / "run_metadata.json"
    metadata = {
        "status": "running", "layer_from": layer_from, "layer_to": layer_to,
        "source_variant": SOURCE_VARIANT, "predictive_variants": list(PREDICTIVE_VARIANTS),
        "diagnostic_variant": "distribution_drift", "feature_dtype": "float64",
        "distance_type": "pre_mds_raw_unclipped", "split_id": SPLIT_ID,
        "quantile_method": f"reservoir_sample_max_{QUANTILE_SAMPLE_SIZE}",
        "started_at": now_timestamp(),
    }
    write_json_atomic(metadata_path, metadata)

    try:
        coefficients_path = source_dir / "coefficients.csv"
        assignments_path = source_dir / "split_assignments.csv"
        models = load_saved_models(coefficients_path, (layer_from, layer_to))
        records, assignment_rows = load_records(assignments_path)
        train_records = [record for record in records if record["split"] == "train"]
        log(f"Transition {layer_from}->{layer_to}: {len(records)} proteins")
        write_csv_atomic(output_dir / "split_assignments.csv", list(assignment_rows[0]), assignment_rows)
        coefficient_rows = models[layer_from]["rows"] + models[layer_to]["rows"]
        write_csv_atomic(output_dir / "coefficients_used.csv", list(coefficient_rows[0]), coefficient_rows)

        timing_rows = []
        training_t0 = time.time()
        layer_mean_sum = [np.zeros(FEATURE_DIM), np.zeros(FEATURE_DIM)]
        layer_second_sum = [np.zeros(FEATURE_DIM), np.zeros(FEATURE_DIM)]
        pooled_gram = np.zeros((AUGMENTED_DIM, AUGMENTED_DIM), dtype=np.float64)
        pooled_cross = np.zeros(AUGMENTED_DIM, dtype=np.float64)

        for record_index, record in enumerate(train_records, start=1):
            protein_t0 = time.time()
            from_path, to_path, pdb_path = resolve_inputs(record, layer_from, layer_to)
            pb_from = np.load(str(from_path))
            pb_to = np.load(str(to_path))
            validate_pair_block(pb_from, record, layer_from)
            validate_pair_block(pb_to, record, layer_to)
            hi, lo = lower_triangle_indices(record["n_residues"])
            target = load_targets(record, pdb_path, hi, lo)
            for endpoint_index, pb in enumerate((pb_from, pb_to)):
                mean, second, gram, cross = per_protein_layer_statistics(pb, hi, lo, target)
                layer_mean_sum[endpoint_index] += mean
                layer_second_sum[endpoint_index] += second
                pooled_gram += 0.5 * gram
                pooled_cross += 0.5 * cross
            del pb_from, pb_to, hi, lo, target, mean, second, gram, cross
            gc.collect()
            if record_index % 10 == 0 or record_index == len(train_records):
                log(
                    f"Training statistics {record_index}/{len(train_records)}; "
                    f"last={time.time() - protein_t0:.1f}s; total={time.time() - training_t0:.1f}s"
                )

        denominator = float(len(train_records))
        alignment = alignment_from_moments(
            layer_mean_sum[0] / denominator, layer_second_sum[0] / denominator,
            layer_mean_sum[1] / denominator, layer_second_sum[1] / denominator,
        )
        fit_t0 = time.time()
        pooled = solve_sufficient_statistics(pooled_gram, pooled_cross)
        fit_seconds = time.time() - fit_t0
        timing_rows.append({
            "layer_from": layer_from, "layer_to": layer_to,
            "stage": "training_statistics_and_alignment", "elapsed_seconds": time.time() - training_t0,
        })
        timing_rows.append({
            "layer_from": layer_from, "layer_to": layer_to,
            "stage": "pooled_solve", "elapsed_seconds": fit_seconds,
        })
        write_csv_atomic(output_dir / "alignment_parameters.csv", ALIGNMENT_FIELDS, build_alignment_rows(layer_from, alignment))
        write_csv_atomic(output_dir / "pooled_coefficients.csv", POOLED_COEFFICIENT_FIELDS, build_pooled_rows(layer_from, pooled))
        pooled_fit_rows = [{
            "layer_from": layer_from, "layer_to": layer_to,
            "n_train_proteins": len(train_records),
            "n_layer_observations": 2 * len(train_records), "rank": pooled["rank"],
            "normal_matrix_condition": pooled["normal_matrix_condition"],
            "estimated_design_condition": pooled["estimated_design_condition"],
            "scale_min": float(np.min(pooled["scale"])),
            "scale_max": float(np.max(pooled["scale"])),
            "intercept_standardized": pooled["intercept_std"],
            "intercept_raw": pooled["intercept"],
            "coefficient_l2_raw": float(np.linalg.norm(pooled["coef"])),
            "elapsed_seconds": fit_seconds,
        }]
        write_csv_atomic(output_dir / "pooled_fit_summary.csv", POOLED_FIT_FIELDS, pooled_fit_rows)
        del pooled_gram, pooled_cross, layer_mean_sum, layer_second_sum

        error_acc = {}
        distance_acc = {}
        true_acc = {}
        change_acc = {}
        comparison_acc = {}
        feature_acc = {
            (split, scope): FeatureMomentAccumulator()
            for split in SPLITS for scope in SCOPES
        }
        per_rows = []
        manifest_rows = []
        evaluation_t0 = time.time()

        for record_index, record in enumerate(records, start=1):
            protein_t0 = time.time()
            from_path, to_path, pdb_path = resolve_inputs(record, layer_from, layer_to)
            pb_from = np.load(str(from_path))
            pb_to = np.load(str(to_path))
            validate_pair_block(pb_from, record, layer_from)
            validate_pair_block(pb_to, record, layer_to)
            hi, lo = lower_triangle_indices(record["n_residues"])
            local_mask = local_pair_mask(hi, lo)
            target_all = load_targets(record, pdb_path, hi, lo)
            n_all = int(len(hi))
            n_local = int(np.count_nonzero(local_mask))
            protein_predictions = {
                variant: {
                    "from": np.empty(n_all, dtype=np.float64),
                    "to": np.empty(n_all, dtype=np.float64),
                }
                for variant in PREDICTIVE_VARIANTS
            }
            protein_feature_sums = {
                scope: {
                    "n": 0,
                    "from_sum": np.zeros(FEATURE_DIM, dtype=np.float64),
                    "from_sq": np.zeros(FEATURE_DIM, dtype=np.float64),
                    "to_sum": np.zeros(FEATURE_DIM, dtype=np.float64),
                    "to_sq": np.zeros(FEATURE_DIM, dtype=np.float64),
                }
                for scope in SCOPES
            }

            for start in range(0, n_all, FEATURE_CHUNK_SIZE):
                end = min(start + FEATURE_CHUNK_SIZE, n_all)
                x_from = directional_features(pb_from, hi[start:end], lo[start:end])
                x_to = directional_features(pb_to, hi[start:end], lo[start:end])
                x_aligned = x_to * alignment["multiplier"] + alignment["offset"]
                protein_predictions["fixed_raw_layer_x"]["from"][start:end] = predict_raw(x_from, models[layer_from])
                protein_predictions["fixed_raw_layer_x"]["to"][start:end] = predict_raw(x_to, models[layer_from])
                protein_predictions["fixed_moment_aligned_to_x"]["from"][start:end] = predict_raw(x_from, models[layer_from])
                protein_predictions["fixed_moment_aligned_to_x"]["to"][start:end] = predict_raw(x_aligned, models[layer_from])
                protein_predictions["saved_own_layer"]["from"][start:end] = predict_raw(x_from, models[layer_from])
                protein_predictions["saved_own_layer"]["to"][start:end] = predict_raw(x_to, models[layer_to])
                protein_predictions["pooled_two_layer"]["from"][start:end] = predict_raw(x_from, pooled)
                protein_predictions["pooled_two_layer"]["to"][start:end] = predict_raw(x_to, pooled)
                all_sums = protein_feature_sums["all"]
                all_sums["n"] += int(x_from.shape[0])
                all_sums["from_sum"] += np.sum(x_from, axis=0, dtype=np.float64)
                all_sums["from_sq"] += np.sum(x_from * x_from, axis=0, dtype=np.float64)
                all_sums["to_sum"] += np.sum(x_to, axis=0, dtype=np.float64)
                all_sums["to_sq"] += np.sum(x_to * x_to, axis=0, dtype=np.float64)
                chunk_local = local_mask[start:end]
                if np.any(chunk_local):
                    local_from = x_from[chunk_local]
                    local_to = x_to[chunk_local]
                    local_sums = protein_feature_sums["local_4_12"]
                    local_sums["n"] += int(local_from.shape[0])
                    local_sums["from_sum"] += np.sum(local_from, axis=0, dtype=np.float64)
                    local_sums["from_sq"] += np.sum(local_from * local_from, axis=0, dtype=np.float64)
                    local_sums["to_sum"] += np.sum(local_to, axis=0, dtype=np.float64)
                    local_sums["to_sq"] += np.sum(local_to * local_to, axis=0, dtype=np.float64)
                    del local_from, local_to
                del x_from, x_to, x_aligned

            for scope, mask in (("all", np.ones(n_all, dtype=bool)), ("local_4_12", local_mask)):
                y = target_all[mask]
                sums = protein_feature_sums[scope]
                feature_acc[(record["split"], scope)].update_sums(
                    sums["n"], sums["from_sum"], sums["from_sq"],
                    sums["to_sum"], sums["to_sq"],
                )
                true_acc.setdefault(
                    (record["split"], scope), MomentAccumulator()
                ).update(y)
                raw_change = None
                protein_variant_changes = {}
                for variant in PREDICTIVE_VARIANTS:
                    pred_from = protein_predictions[variant]["from"][mask]
                    pred_to = protein_predictions[variant]["to"][mask]
                    change = pred_to - pred_from
                    protein_variant_changes[variant] = change
                    if variant == "fixed_raw_layer_x":
                        raw_change = change
                    per_row = per_protein_row(
                        layer_from, variant, record, scope, y, pred_from, pred_to
                    )
                    per_rows.append(per_row)
                    for endpoint, prediction in (("from", pred_from), ("to", pred_to)):
                        error_key = (record["split"], scope, variant, endpoint)
                        error_acc.setdefault(error_key, ErrorAccumulator()).update(y, prediction)
                        seed = stable_group_seed(error_key)
                        distance_acc.setdefault(error_key, DistributionAccumulator(seed)).update(prediction)
                    change_key = (record["split"], scope, variant)
                    seed = stable_group_seed(change_key)
                    change_acc.setdefault(change_key, DistributionAccumulator(seed)).update(change, classify=True)
                for variant in PREDICTIVE_VARIANTS[1:]:
                    key = (record["split"], scope, variant)
                    comparison_acc.setdefault(key, ComparisonAccumulator()).update(
                        protein_variant_changes[variant], raw_change
                    )

            manifest_rows.append({
                "layer_from": layer_from, "layer_to": layer_to,
                "split": record["split"], "source": record["source"],
                "protein_id": record["protein_id"], "n_residues": record["n_residues"],
                "n_all_pairs": n_all, "n_local_pairs": n_local,
                "pair_block_from": str(from_path), "pair_block_to": str(to_path),
                "pdb_path": str(pdb_path), "elapsed_seconds": time.time() - protein_t0,
            })
            del pb_from, pb_to, hi, lo, local_mask, target_all, protein_predictions, protein_feature_sums
            gc.collect()
            if record_index % 10 == 0 or record_index == len(records):
                write_csv_atomic(output_dir / "per_protein_metrics.csv", PER_PROTEIN_FIELDS, per_rows)
                write_csv_atomic(output_dir / "protein_manifest.csv", MANIFEST_FIELDS, manifest_rows)
                log(
                    f"Evaluation {record_index}/{len(records)}; "
                    f"last={time.time() - protein_t0:.1f}s; total={time.time() - evaluation_t0:.1f}s"
                )

        timing_rows.append({
            "layer_from": layer_from, "layer_to": layer_to,
            "stage": "evaluation", "elapsed_seconds": time.time() - evaluation_t0,
        })

        performance_rows = []
        distance_rows = []
        change_rows = []
        comparison_rows = []
        for split in SPLITS:
            for scope in SCOPES:
                scope_rows = [row for row in per_rows if row["split"] == split and row["pair_scope"] == scope]
                n_proteins = len({row["protein_id"] for row in scope_rows})
                for variant in PREDICTIVE_VARIANTS:
                    variant_rows = [row for row in scope_rows if row["variant"] == variant]
                    for endpoint in ("from", "to"):
                        key = (split, scope, variant, endpoint)
                        errors = error_acc[key].summary()
                        r2_values = np.asarray([row[f"{endpoint}_r2"] for row in variant_rows], dtype=np.float64)
                        r2_values = r2_values[np.isfinite(r2_values)]
                        performance_rows.append({
                            "layer_from": layer_from, "layer_to": layer_to,
                            "variant": variant, "endpoint": endpoint,
                            "layer": layer_from if endpoint == "from" else layer_to,
                            "split": split, "pair_scope": scope,
                            "n_proteins": n_proteins, "n_pairs": error_acc[key].n,
                            "pair_weighted_r2": errors["r2"],
                            "pair_weighted_mae": errors["mae"],
                            "pair_weighted_rmse": errors["rmse"],
                            "pair_weighted_bias": errors["bias"],
                            "mean_per_protein_r2": float(np.mean(r2_values)) if r2_values.size else float("nan"),
                            "median_per_protein_r2": float(np.median(r2_values)) if r2_values.size else float("nan"),
                        })
                        pair_pred = distance_acc[key].summary()
                        pair_true = true_acc[(split, scope)].summary()
                        distance_rows.append({
                            "layer_from": layer_from, "layer_to": layer_to,
                            "variant": variant, "endpoint": endpoint,
                            "layer": layer_from if endpoint == "from" else layer_to,
                            "split": split, "pair_scope": scope,
                            "aggregation": "pair_weighted", "n_proteins": n_proteins,
                            "n_values": distance_acc[key].n,
                            "mean_predicted": pair_pred["mean"], "std_predicted": pair_pred["std"],
                            **{f"{name}_predicted": pair_pred[name] for name in (
                                "q01", "q05", "q25", "median", "q75", "q95", "q99", "minimum", "maximum"
                            )},
                            "fraction_negative_predicted": distance_acc[key].negative / max(distance_acc[key].n, 1),
                            "mean_true": pair_true["mean"], "std_true": pair_true["std"],
                            "mean_bias": errors["bias"],
                            "quantile_method": f"reservoir_sample_max_{QUANTILE_SAMPLE_SIZE}",
                        })
                        pred_means = np.asarray([row[f"mean_predicted_{endpoint}"] for row in variant_rows])
                        true_means = np.asarray([row["mean_true_distance"] for row in variant_rows])
                        pred_stats = statistics_from_values(pred_means)
                        true_stats = statistics_from_values(true_means)
                        distance_rows.append({
                            "layer_from": layer_from, "layer_to": layer_to,
                            "variant": variant, "endpoint": endpoint,
                            "layer": layer_from if endpoint == "from" else layer_to,
                            "split": split, "pair_scope": scope,
                            "aggregation": "protein_equal", "n_proteins": n_proteins,
                            "n_values": len(variant_rows), "mean_predicted": pred_stats["mean"],
                            "std_predicted": pred_stats["std"],
                            **{f"{name}_predicted": pred_stats[name] for name in (
                                "q01", "q05", "q25", "median", "q75", "q95", "q99", "minimum", "maximum"
                            )},
                            "fraction_negative_predicted": float(np.mean(pred_means < 0.0)),
                            "mean_true": true_stats["mean"], "std_true": true_stats["std"],
                            "mean_bias": float(np.mean(pred_means - true_means)),
                            "quantile_method": "exact_per_protein_means",
                        })

                    pair_change = change_acc[(split, scope, variant)]
                    pair_stats = pair_change.summary()
                    change_rows.append({
                        "layer_from": layer_from, "layer_to": layer_to,
                        "variant": variant, "split": split, "pair_scope": scope,
                        "aggregation": "pair_weighted", "n_proteins": n_proteins,
                        "n_values": pair_change.n, **pair_stats,
                        "fraction_decrease": pair_change.decrease / max(pair_change.n, 1),
                        "fraction_no_change": pair_change.no_change / max(pair_change.n, 1),
                        "fraction_increase": pair_change.increase / max(pair_change.n, 1),
                        "threshold_angstrom": CHANGE_THRESHOLD,
                        "quantile_method": f"reservoir_sample_max_{QUANTILE_SAMPLE_SIZE}",
                    })
                    protein_changes = np.asarray([row["mean_change"] for row in variant_rows])
                    protein_stats = statistics_from_values(protein_changes)
                    protein_labels = class_values(protein_changes)
                    change_rows.append({
                        "layer_from": layer_from, "layer_to": layer_to,
                        "variant": variant, "split": split, "pair_scope": scope,
                        "aggregation": "protein_equal", "n_proteins": n_proteins,
                        "n_values": len(variant_rows), **protein_stats,
                        "fraction_decrease": float(np.mean(protein_labels == -1)),
                        "fraction_no_change": float(np.mean(protein_labels == 0)),
                        "fraction_increase": float(np.mean(protein_labels == 1)),
                        "threshold_angstrom": CHANGE_THRESHOLD,
                        "quantile_method": "exact_per_protein_means",
                    })

                raw_rows = {
                    row["protein_id"]: row for row in scope_rows
                    if row["variant"] == "fixed_raw_layer_x"
                }
                for variant in PREDICTIVE_VARIANTS[1:]:
                    pair_summary = comparison_acc[(split, scope, variant)].summary()
                    comparison_rows.append({
                        "layer_from": layer_from, "layer_to": layer_to,
                        "variant": variant, "reference_variant": "fixed_raw_layer_x",
                        "split": split, "pair_scope": scope, "aggregation": "pair_weighted",
                        "n_proteins": n_proteins,
                        "n_values": comparison_acc[(split, scope, variant)].n,
                        "difference_in_mean_change": pair_summary["difference_in_mean_change"],
                        "change_rmse_vs_reference": pair_summary["rmse"],
                        "change_pearson_r_vs_reference": pair_summary["pearson"],
                        "sign_agreement": pair_summary["sign_agreement"],
                        "threshold_class_agreement": pair_summary["class_agreement"],
                        "threshold_angstrom": CHANGE_THRESHOLD,
                    })
                    controls = []
                    references = []
                    for row in scope_rows:
                        if row["variant"] == variant:
                            controls.append(row["mean_change"])
                            references.append(raw_rows[row["protein_id"]]["mean_change"])
                    protein_comparison = ComparisonAccumulator()
                    protein_comparison.update(controls, references)
                    summary = protein_comparison.summary()
                    comparison_rows.append({
                        "layer_from": layer_from, "layer_to": layer_to,
                        "variant": variant, "reference_variant": "fixed_raw_layer_x",
                        "split": split, "pair_scope": scope, "aggregation": "protein_equal",
                        "n_proteins": n_proteins, "n_values": len(controls),
                        "difference_in_mean_change": summary["difference_in_mean_change"],
                        "change_rmse_vs_reference": summary["rmse"],
                        "change_pearson_r_vs_reference": summary["pearson"],
                        "sign_agreement": summary["sign_agreement"],
                        "threshold_class_agreement": summary["class_agreement"],
                        "threshold_angstrom": CHANGE_THRESHOLD,
                    })

        drift_channel_rows = []
        drift_summary_rows = []
        names = feature_names_for_variant(VARIANT_SPEC)
        beta = models[layer_from]["coef"]
        for split in SPLITS:
            for scope in SCOPES:
                accumulator = feature_acc[(split, scope)]
                for aggregation in AGGREGATIONS:
                    mean_from, second_from, mean_to, second_to = accumulator.moments(aggregation)
                    raw_scale_from, safe_scale_from, zero_from = safe_scales(mean_from, second_from)
                    raw_scale_to, safe_scale_to, zero_to = safe_scales(mean_to, second_to)
                    mean_shift = mean_to - mean_from
                    ratio = safe_scale_to / safe_scale_from
                    standardized_shift = mean_shift / safe_scale_from
                    contributions = beta * mean_shift
                    beta_scale = beta * (raw_scale_to - raw_scale_from)
                    for index, name in enumerate(names):
                        drift_channel_rows.append({
                            "layer_from": layer_from, "layer_to": layer_to,
                            "split": split, "pair_scope": scope, "aggregation": aggregation,
                            "feature_index": index, "feature_name": name,
                            "mean_from": mean_from[index], "mean_to": mean_to[index],
                            "mean_shift": mean_shift[index], "std_from": raw_scale_from[index],
                            "std_to": raw_scale_to[index], "scale_ratio": ratio[index],
                            "standardized_mean_shift": standardized_shift[index],
                            "coefficient_layer_x": beta[index],
                            "beta_mean_shift_contribution": contributions[index],
                            "beta_scale_change": beta_scale[index],
                            "from_zero_variance": zero_from[index],
                            "to_zero_variance": zero_to[index],
                        })
                    matching_change = next(
                        row for row in change_rows
                        if row["variant"] == "fixed_raw_layer_x"
                        and row["split"] == split and row["pair_scope"] == scope
                        and row["aggregation"] == aggregation
                    )
                    reconstructed = float(np.sum(contributions))
                    log_ratio = np.log(np.maximum(ratio, FLOAT_EPS))
                    drift_summary_rows.append({
                        "layer_from": layer_from, "layer_to": layer_to,
                        "split": split, "pair_scope": scope, "aggregation": aggregation,
                        "n_proteins": accumulator.protein_count,
                        "n_pairs": accumulator.pair_count,
                        "mean_fixed_raw_change": matching_change["mean"],
                        "sum_beta_mean_shift_contribution": reconstructed,
                        "reconstruction_error": reconstructed - matching_change["mean"],
                        "mean_absolute_standardized_mean_shift": float(np.mean(np.abs(standardized_shift))),
                        "maximum_absolute_standardized_mean_shift": float(np.max(np.abs(standardized_shift))),
                        "mean_absolute_log_scale_ratio": float(np.mean(np.abs(log_ratio))),
                        "maximum_absolute_log_scale_ratio": float(np.max(np.abs(log_ratio))),
                        "n_from_zero_variance": int(np.count_nonzero(zero_from)),
                        "n_to_zero_variance": int(np.count_nonzero(zero_to)),
                    })

        maximum_reconstruction_error = max(abs(row["reconstruction_error"]) for row in drift_summary_rows)
        if maximum_reconstruction_error > 1e-7:
            raise AssertionError(
                f"Drift contribution reconstruction error {maximum_reconstruction_error:.3g} exceeds tolerance"
            )

        write_csv_atomic(output_dir / "probe_performance.csv", PROBE_PERFORMANCE_FIELDS, performance_rows)
        write_csv_atomic(output_dir / "distance_statistics.csv", DISTANCE_STATISTIC_FIELDS, distance_rows)
        write_csv_atomic(output_dir / "change_statistics.csv", CHANGE_STATISTIC_FIELDS, change_rows)
        write_csv_atomic(output_dir / "variant_comparison.csv", VARIANT_COMPARISON_FIELDS, comparison_rows)
        write_csv_atomic(output_dir / "per_protein_metrics.csv", PER_PROTEIN_FIELDS, per_rows)
        write_csv_atomic(output_dir / "distribution_drift_channels.csv", DRIFT_CHANNEL_FIELDS, drift_channel_rows)
        write_csv_atomic(output_dir / "distribution_drift_summary.csv", DRIFT_SUMMARY_FIELDS, drift_summary_rows)
        write_csv_atomic(output_dir / "protein_manifest.csv", MANIFEST_FIELDS, manifest_rows)
        timing_rows.append({
            "layer_from": layer_from, "layer_to": layer_to,
            "stage": "total", "elapsed_seconds": time.time() - total_t0,
        })
        write_csv_atomic(output_dir / "timing.csv", TIMING_FIELDS, timing_rows)

        summary = (
            "Consecutive-layer distance-control probe\n"
            "========================================\n\n"
            f"Status: complete\nTransition: {layer_from}->{layer_to}\n"
            f"Proteins: {len(records)} ({len(train_records)} train, {len(records) - len(train_records)} test)\n"
            "Predictive variants: fixed_raw_layer_x, fixed_moment_aligned_to_x, "
            "saved_own_layer, pooled_two_layer\n"
            "Diagnostic variant: distribution_drift\n"
            "Scopes: all unordered pairs; local sequence separation 4..12\n"
            "Distances: raw pre-MDS, unclipped\n"
            "Training: protein-equal; pooled model splits each protein's weight equally across endpoints\n"
            f"Maximum drift reconstruction error: {maximum_reconstruction_error:.9g} Angstrom\n"
            f"Elapsed seconds: {time.time() - total_t0:.2f}\n"
        )
        write_text_atomic(output_dir / "summary.txt", summary)
        metadata.update({
            "status": "complete", "completed_at": now_timestamp(),
            "n_proteins": len(records), "n_train_proteins": len(train_records),
            "n_test_proteins": len(records) - len(train_records),
            "alignment_weighting": "protein_equal_all_unordered_training_pairs",
            "pooled_weighting": "protein_equal_then_half_per_endpoint",
            "maximum_drift_reconstruction_error": maximum_reconstruction_error,
            "elapsed_seconds": time.time() - total_t0,
        })
        write_json_atomic(metadata_path, metadata)
        log(f"Completed transition {layer_from}->{layer_to} in {time.time() - total_t0:.1f}s")
    except Exception as exc:
        metadata.update({
            "status": "failed", "failed_at": now_timestamp(),
            "error": f"{type(exc).__name__}: {exc}",
            "elapsed_seconds": time.time() - total_t0,
        })
        write_json_atomic(metadata_path, metadata)
        traceback.print_exc()
        raise


if __name__ == "__main__":
    main()

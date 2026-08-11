#!/usr/bin/env python3
"""Protein-bootstrap pilot for uncertainty in consecutive-layer distance changes.

One Slurm array task handles one transition x -> x+1 for x in 10..15. The
layer-x general_base256_combined distance model is refit on ten protein-level
bootstrap samples. Each refitted model is applied to both layer x and layer
x+1, so the bootstrap distribution measures weight-estimation uncertainty in
the predicted change.

The expensive all-pair training pass is performed once. Per-protein normal-
equation sufficient statistics are accumulated directly into all bootstrap
replicates, after which the ten 257-parameter solves are inexpensive. All
materialized features and regression calculations use float64.

This pilot intentionally analyzes raw pre-MDS predictions only.
"""

import csv
import gc
import gzip
import json
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
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


VARIANT = make_variant("general_base256_combined", pair_mode="concat_directional")
VARIANT_NAME = VARIANT["name"]
FEATURE_DIM = 2 * N_CHANNELS
AUGMENTED_DIM = FEATURE_DIM + 1
N_BOOTSTRAPS = 10
BOOTSTRAP_SEED = 20260811
START_LAYERS = tuple(range(10, 16))
MIN_SEQUENCE_SEPARATION = 4
MAX_SEQUENCE_SEPARATION = 12
THRESHOLDS = (0.05, 0.10, 0.20, 0.50)
FEATURE_CHUNK_SIZE = 100_000
OUTPUT_NAME = "local_pair_change_se_pilot_layers10_15"

PAIR_FIELDS = [
    "variant",
    "distance_type",
    "layer_from",
    "layer_to",
    "split",
    "source",
    "protein_id",
    "i",
    "j",
    "sequence_separation",
    "predicted_distance_from",
    "predicted_distance_to",
    "change_saved_model",
    "bootstrap_mean_change",
    "pair_change_se",
    "normal_ci95_low",
    "normal_ci95_high",
    "change_z_score",
]

DISTRIBUTION_FIELDS = [
    "n_pairs",
    "mean",
    "std",
    "naive_pair_sem",
    "mean_absolute",
    "rms",
    "minimum",
    "q01",
    "q05",
    "q25",
    "median",
    "q75",
    "q95",
    "q99",
    "maximum",
    "fraction_negative",
    "fraction_zero",
    "fraction_positive",
]

SE_DISTRIBUTION_FIELDS = [
    "n_pairs",
    "mean",
    "std",
    "minimum",
    "q01",
    "q05",
    "q25",
    "median",
    "q75",
    "q90",
    "q95",
    "q99",
    "maximum",
    "mean_ci95_half_width",
    "fraction_se_le_0_05",
    "fraction_se_le_0_10",
    "fraction_se_le_0_20",
    "fraction_se_le_0_50",
]

DISJOINT_BUCKETS = (
    ("lt_m0_50", -math.inf, -0.50, False, False),
    ("m0_50_to_m0_20", -0.50, -0.20, True, False),
    ("m0_20_to_m0_10", -0.20, -0.10, True, False),
    ("m0_10_to_m0_05", -0.10, -0.05, True, False),
    ("m0_05_to_p0_05", -0.05, 0.05, True, True),
    ("p0_05_to_p0_10", 0.05, 0.10, False, True),
    ("p0_10_to_p0_20", 0.10, 0.20, False, True),
    ("p0_20_to_p0_50", 0.20, 0.50, False, True),
    ("gt_p0_50", 0.50, math.inf, False, False),
)


def now_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_progress(message):
    print(f"[{now_timestamp()}] {message}", flush=True)


def format_csv_value(value):
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return "nan"
        return f"{value:.9g}"
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_csv_atomic(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(path.name + ".tmp")
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: format_csv_value(row.get(key, "")) for key in fieldnames}
            )
    os.replace(tmp_path, path)


def requested_start_layer():
    raw_task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw_task_id is None or raw_task_id == "":
        raise ValueError(
            "SLURM_ARRAY_TASK_ID is required and must be one of 10,11,12,13,14,15"
        )
    layer = int(raw_task_id)
    if layer not in START_LAYERS:
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID={layer} is not in allowed start layers {START_LAYERS}"
        )
    return layer


def build_bootstrap_counts(n_proteins, n_bootstraps=N_BOOTSTRAPS, seed=BOOTSTRAP_SEED):
    """Return cluster-bootstrap multiplicities; every row has n_proteins draws."""
    if n_proteins <= 0 or n_bootstraps < 2:
        raise ValueError("Need at least one protein and at least two bootstraps")
    counts = np.zeros((n_bootstraps, n_proteins), dtype=np.int16)
    for replicate in range(n_bootstraps):
        rng = np.random.default_rng(int(seed) + replicate)
        draws = rng.integers(0, n_proteins, size=n_proteins)
        counts[replicate] = np.bincount(draws, minlength=n_proteins)
    return counts


def local_pair_indices(n_residues):
    """Return canonical i<j local pairs with separation in the inclusive range."""
    i, j = np.triu_indices(int(n_residues), k=MIN_SEQUENCE_SEPARATION)
    separation = j - i
    keep = separation <= MAX_SEQUENCE_SEPARATION
    return (
        np.asarray(i[keep], dtype=np.int64),
        np.asarray(j[keep], dtype=np.int64),
        np.asarray(separation[keep], dtype=np.int16),
    )


def load_saved_coefficient_bundle(path, layer):
    """Load the exact raw-scale layer coefficient vector and retain source rows."""
    expected_names = feature_names_for_variant(VARIANT)
    coefficients = np.full((FEATURE_DIM,), np.nan, dtype=np.float64)
    intercept = None
    source_rows = []
    seen_features = set()

    with path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != VARIANT_NAME:
                continue
            if int(row["split_id"]) != SPLIT_ID or int(row["layer"]) != int(layer):
                continue
            if int(row["feature_dim"]) != FEATURE_DIM:
                raise ValueError(f"Unexpected feature dimension in {path}: {row}")
            feature_index = int(row["feature_index"])
            value = float(row["coefficient_raw_scale"])
            source_rows.append(dict(row))
            if feature_index == -1:
                if intercept is not None:
                    raise ValueError(f"Duplicate intercept for layer {layer}")
                intercept = value
            else:
                if feature_index < 0 or feature_index >= FEATURE_DIM:
                    raise ValueError(f"Invalid feature index {feature_index}")
                if feature_index in seen_features:
                    raise ValueError(f"Duplicate feature index {feature_index}")
                if row["feature_name"] != expected_names[feature_index]:
                    raise ValueError(
                        f"Feature {feature_index} is {row['feature_name']}, "
                        f"expected {expected_names[feature_index]}"
                    )
                coefficients[feature_index] = value
                seen_features.add(feature_index)

    if intercept is None or len(seen_features) != FEATURE_DIM:
        raise ValueError(
            f"Layer {layer} expected one intercept and {FEATURE_DIM} features in {path}"
        )
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"Layer {layer} contains non-finite coefficients")
    return {
        "intercept": float(intercept),
        "coef": coefficients,
        "source_rows": source_rows,
    }


def load_records(assignment_path):
    """Load the frozen general-model split and resolve its Delta input paths."""
    manifest_by_id = {
        entry["protein_id"]: entry for entry in build_dataset_protein_manifest()
    }
    records = []
    seen = set()
    with assignment_path.open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["split_id"]) != SPLIT_ID:
                continue
            protein_id = row["protein_id"]
            if protein_id in seen:
                raise ValueError(f"Duplicate split assignment for {protein_id}")
            manifest = manifest_by_id.get(protein_id)
            if manifest is None:
                raise ValueError(f"{protein_id} is absent from the combined manifest")
            split = row["split"]
            if split not in {"train", "test"}:
                raise ValueError(f"Unexpected split {split!r} for {protein_id}")
            records.append(
                {
                    "protein_id": protein_id,
                    "source": manifest["source"],
                    "protein_dir": manifest["protein_dir"],
                    "pdb_root": manifest["pdb_root"],
                    "split": split,
                    "n_residues": int(row["n_residues"]),
                    "size_bin": int(row["size_bin"]),
                    "n_pairs": int(row["n_pairs"]),
                }
            )
            seen.add(protein_id)
    if not records:
        raise ValueError(f"No split_id={SPLIT_ID} assignments found in {assignment_path}")
    return records


def validate_pair_block(pb, record, layer):
    expected = (
        int(record["n_residues"]),
        int(record["n_residues"]),
        N_CHANNELS,
    )
    if tuple(pb.shape) != expected:
        raise ValueError(
            f"{record['protein_id']} layer {layer} shape {pb.shape} != {expected}"
        )


def compute_equal_protein_sufficient_statistics(pb, coordinates):
    """Compute X'X/n and X'y/n for one protein in float64 chunks."""
    n_residues = int(pb.shape[0])
    pair_hi, pair_lo = np.tril_indices(n_residues, k=-1)
    n_pairs = int(pair_hi.size)
    if n_pairs <= 0:
        raise ValueError("A training protein must contain at least one residue pair")

    gram = np.zeros((AUGMENTED_DIM, AUGMENTED_DIM), dtype=np.float64)
    cross = np.zeros((AUGMENTED_DIM,), dtype=np.float64)
    inverse_sqrt_n = 1.0 / math.sqrt(float(n_pairs))

    for start in range(0, n_pairs, FEATURE_CHUNK_SIZE):
        end = min(start + FEATURE_CHUNK_SIZE, n_pairs)
        hi = pair_hi[start:end]
        lo = pair_lo[start:end]
        design = np.empty((end - start, AUGMENTED_DIM), dtype=np.float64)
        design[:, 0] = 1.0
        design[:, 1:1 + N_CHANNELS] = pb[hi, lo, :]
        design[:, 1 + N_CHANNELS:] = pb[lo, hi, :]
        design *= inverse_sqrt_n

        coordinate_delta = coordinates[hi] - coordinates[lo]
        target = np.sqrt(
            np.sum(coordinate_delta * coordinate_delta, axis=1, dtype=np.float64)
        )
        target *= inverse_sqrt_n
        gram += design.T @ design
        cross += design.T @ target
        del design, coordinate_delta, target

    return gram, cross


def solve_sufficient_statistics(gram, cross):
    """Solve one weighted OLS problem after standardizing from its raw moments."""
    gram = np.asarray(gram, dtype=np.float64)
    cross = np.asarray(cross, dtype=np.float64)
    gram = 0.5 * (gram + gram.T)
    mass = float(gram[0, 0])
    if not math.isfinite(mass) or mass <= 0.0:
        raise ValueError(f"Invalid regression mass {mass}")

    mean = gram[0, 1:] / mass
    second_moment = np.diag(gram)[1:] / mass
    scale = np.sqrt(np.maximum(second_moment - mean * mean, 0.0))
    scale[scale < 1e-12] = 1.0

    transform = np.zeros((AUGMENTED_DIM, AUGMENTED_DIM), dtype=np.float64)
    transform[0, 0] = 1.0
    feature_indices = np.arange(FEATURE_DIM)
    transform[0, 1:] = -mean / scale
    transform[feature_indices + 1, feature_indices + 1] = 1.0 / scale

    standardized_gram = transform.T @ gram @ transform
    standardized_gram = 0.5 * (standardized_gram + standardized_gram.T)
    standardized_cross = transform.T @ cross
    theta_standardized, _, rank, singular_values = np.linalg.lstsq(
        standardized_gram,
        standardized_cross,
        rcond=None,
    )
    theta_raw = transform @ theta_standardized
    positive_singular = singular_values[singular_values > 0.0]
    normal_condition = (
        float(positive_singular[0] / positive_singular[-1])
        if positive_singular.size
        else float("inf")
    )
    return {
        "intercept": float(theta_raw[0]),
        "coef": np.asarray(theta_raw[1:], dtype=np.float64),
        "rank": int(rank),
        "normal_matrix_condition": normal_condition,
        "estimated_design_condition": math.sqrt(normal_condition),
        "scale_min": float(np.min(scale)),
        "scale_max": float(np.max(scale)),
    }


def extract_local_change_inputs(pb_from, pb_to, record, saved_bundle):
    """Extract local feature differences and saved-model point predictions."""
    i, j, separation = local_pair_indices(record["n_residues"])
    n_local = int(i.size)
    feature_difference = np.empty((n_local, FEATURE_DIM), dtype=np.float64)

    # The fitted model used lower-triangle order: pair_ij is (j, i) for i < j.
    feature_difference[:, :N_CHANNELS] = pb_to[j, i, :] - pb_from[j, i, :]
    feature_difference[:, N_CHANNELS:] = pb_to[i, j, :] - pb_from[i, j, :]

    coef = saved_bundle["coef"]
    intercept = float(saved_bundle["intercept"])
    distance_from = np.full((n_local,), intercept, dtype=np.float64)
    distance_to = np.full((n_local,), intercept, dtype=np.float64)
    distance_from += np.asarray(pb_from[j, i, :], dtype=np.float64) @ coef[:N_CHANNELS]
    distance_from += np.asarray(pb_from[i, j, :], dtype=np.float64) @ coef[N_CHANNELS:]
    distance_to += np.asarray(pb_to[j, i, :], dtype=np.float64) @ coef[:N_CHANNELS]
    distance_to += np.asarray(pb_to[i, j, :], dtype=np.float64) @ coef[N_CHANNELS:]

    return {
        "i": i,
        "j": j,
        "sequence_separation": separation,
        "feature_difference": feature_difference,
        "distance_from": distance_from,
        "distance_to": distance_to,
    }


def pair_change_standard_error(bootstrap_changes):
    values = np.asarray(bootstrap_changes, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] < 2:
        raise ValueError("bootstrap_changes must have shape (pairs, replicates>=2)")
    return np.std(values, axis=1, ddof=1, dtype=np.float64)


def summarize_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in DISTRIBUTION_FIELDS}
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    return {
        "n_pairs": int(values.size),
        "mean": float(np.mean(values)),
        "std": std,
        "naive_pair_sem": float(std / math.sqrt(values.size)),
        "mean_absolute": float(np.mean(np.abs(values))),
        "rms": float(np.sqrt(np.mean(values * values))),
        "minimum": float(np.min(values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q95": float(quantiles[5]),
        "q99": float(quantiles[6]),
        "maximum": float(np.max(values)),
        "fraction_negative": float(np.mean(values < 0.0)),
        "fraction_zero": float(np.mean(values == 0.0)),
        "fraction_positive": float(np.mean(values > 0.0)),
    }


def summarize_se_distribution(values):
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {key: float("nan") for key in SE_DISTRIBUTION_FIELDS}
    std = float(np.std(values, ddof=1)) if values.size > 1 else 0.0
    quantiles = np.quantile(
        values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.90, 0.95, 0.99]
    )
    return {
        "n_pairs": int(values.size),
        "mean": float(np.mean(values)),
        "std": std,
        "minimum": float(np.min(values)),
        "q01": float(quantiles[0]),
        "q05": float(quantiles[1]),
        "q25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "q75": float(quantiles[4]),
        "q90": float(quantiles[5]),
        "q95": float(quantiles[6]),
        "q99": float(quantiles[7]),
        "maximum": float(np.max(values)),
        "mean_ci95_half_width": float(1.96 * np.mean(values)),
        "fraction_se_le_0_05": float(np.mean(values <= 0.05)),
        "fraction_se_le_0_10": float(np.mean(values <= 0.10)),
        "fraction_se_le_0_20": float(np.mean(values <= 0.20)),
        "fraction_se_le_0_50": float(np.mean(values <= 0.50)),
    }


def disjoint_bucket_counts(values):
    values = np.asarray(values, dtype=np.float64)
    rows = []
    total = int(values.size)
    for name, lower, upper, include_lower, include_upper in DISJOINT_BUCKETS:
        lower_mask = values >= lower if include_lower else values > lower
        upper_mask = values <= upper if include_upper else values < upper
        count = int(np.count_nonzero(lower_mask & upper_mask))
        rows.append(
            {
                "bucket": name,
                "lower_bound": lower,
                "upper_bound": upper,
                "include_lower": int(include_lower),
                "include_upper": int(include_upper),
                "count": count,
                "fraction": float(count / total) if total else float("nan"),
            }
        )
    if sum(row["count"] for row in rows) != total:
        raise AssertionError("Disjoint change buckets do not exhaust all values")
    return rows


def class_labels(values, threshold):
    values = np.asarray(values, dtype=np.float64)
    labels = np.zeros(values.shape, dtype=np.int8)
    labels[values < -float(threshold)] = -1
    labels[values > float(threshold)] = 1
    return labels


def group_masks(split_codes, separations):
    split_codes = np.asarray(split_codes, dtype=np.int8)
    separations = np.asarray(separations, dtype=np.int16)
    for split_name, split_mask in (
        ("all", np.ones(split_codes.shape, dtype=bool)),
        ("train", split_codes == 0),
        ("test", split_codes == 1),
    ):
        yield split_name, "all", split_mask
        for sequence_separation in range(
            MIN_SEQUENCE_SEPARATION, MAX_SEQUENCE_SEPARATION + 1
        ):
            yield (
                split_name,
                str(sequence_separation),
                split_mask & (separations == sequence_separation),
            )


def common_scope_fields(layer_from, split_name, sequence_separation):
    return {
        "variant": VARIANT_NAME,
        "distance_type": "pre_mds_raw",
        "aggregation": "pair_weighted",
        "layer_from": int(layer_from),
        "layer_to": int(layer_from + 1),
        "split": split_name,
        "sequence_separation": sequence_separation,
    }


def build_summary_tables(
    layer_from,
    changes,
    pair_se,
    split_codes,
    separations,
    agreement_by_threshold,
):
    change_rows = []
    se_rows = []
    bucket_rows = []
    threshold_rows = []

    for split_name, sequence_separation, mask in group_masks(split_codes, separations):
        scope = common_scope_fields(layer_from, split_name, sequence_separation)
        local_change = changes[mask]
        local_se = pair_se[mask]
        change_rows.append({**scope, **summarize_distribution(local_change)})
        se_rows.append({**scope, **summarize_se_distribution(local_se)})
        for bucket_row in disjoint_bucket_counts(local_change):
            bucket_rows.append({**scope, **bucket_row})

        for threshold in THRESHOLDS:
            labels = class_labels(local_change, threshold)
            n_pairs = int(labels.size)
            n_decrease = int(np.count_nonzero(labels == -1))
            n_no_change = int(np.count_nonzero(labels == 0))
            n_increase = int(np.count_nonzero(labels == 1))
            agreement = agreement_by_threshold[threshold][mask]
            ci_spans_zero = np.abs(local_change) <= 1.96 * local_se
            threshold_rows.append(
                {
                    **scope,
                    "threshold_angstrom": threshold,
                    "n_pairs": n_pairs,
                    "n_decrease": n_decrease,
                    "n_no_change": n_no_change,
                    "n_increase": n_increase,
                    "fraction_decrease": n_decrease / n_pairs if n_pairs else float("nan"),
                    "fraction_no_change": n_no_change / n_pairs if n_pairs else float("nan"),
                    "fraction_increase": n_increase / n_pairs if n_pairs else float("nan"),
                    "mean_saved_label_bootstrap_agreement": (
                        float(np.mean(agreement)) if agreement.size else float("nan")
                    ),
                    "fraction_pair_agreement_ge_0_80": (
                        float(np.mean(agreement >= 0.80)) if agreement.size else float("nan")
                    ),
                    "fraction_pair_agreement_ge_0_90": (
                        float(np.mean(agreement >= 0.90)) if agreement.size else float("nan")
                    ),
                    "fraction_normal_ci95_spans_zero": (
                        float(np.mean(ci_spans_zero)) if ci_spans_zero.size else float("nan")
                    ),
                }
            )
    return change_rows, se_rows, bucket_rows, threshold_rows


def write_pair_rows(path, layer_from, records, local_results):
    """Write one compressed row per local pair using an atomic rename."""
    tmp_path = path.with_name(path.name + ".tmp")
    with gzip.open(tmp_path, "wt", newline="", compresslevel=1) as handle:
        writer = csv.DictWriter(handle, fieldnames=PAIR_FIELDS)
        writer.writeheader()
        for record in records:
            result = local_results[record["protein_id"]]
            n_pairs = int(result["i"].size)
            for index in range(n_pairs):
                se = float(result["pair_se"][index])
                change = float(result["saved_change"][index])
                z_score = change / se if se > 0.0 else float("nan")
                writer.writerow(
                    {
                        "variant": VARIANT_NAME,
                        "distance_type": "pre_mds_raw",
                        "layer_from": layer_from,
                        "layer_to": layer_from + 1,
                        "split": record["split"],
                        "source": record["source"],
                        "protein_id": record["protein_id"],
                        "i": int(result["i"][index]),
                        "j": int(result["j"][index]),
                        "sequence_separation": int(result["sequence_separation"][index]),
                        "predicted_distance_from": format_csv_value(
                            result["distance_from"][index]
                        ),
                        "predicted_distance_to": format_csv_value(
                            result["distance_to"][index]
                        ),
                        "change_saved_model": format_csv_value(change),
                        "bootstrap_mean_change": format_csv_value(
                            result["bootstrap_mean"][index]
                        ),
                        "pair_change_se": format_csv_value(se),
                        "normal_ci95_low": format_csv_value(change - 1.96 * se),
                        "normal_ci95_high": format_csv_value(change + 1.96 * se),
                        "change_z_score": format_csv_value(z_score),
                    }
                )
    os.replace(tmp_path, path)


def coefficient_rows(layer_from, fitted_models, saved_bundle):
    names = feature_names_for_variant(VARIANT)
    rows = []
    models = [("recomputed_original", -1, fitted_models[0])]
    models.extend(
        ("protein_bootstrap", index, model)
        for index, model in enumerate(fitted_models[1:])
    )
    for model_type, replicate, model in models:
        rows.append(
            {
                "variant": VARIANT_NAME,
                "layer": layer_from,
                "model_type": model_type,
                "bootstrap_replicate": replicate,
                "feature_index": -1,
                "feature_name": "intercept",
                "coefficient_raw_scale": model["intercept"],
            }
        )
        for feature_index, feature_name in enumerate(names):
            rows.append(
                {
                    "variant": VARIANT_NAME,
                    "layer": layer_from,
                    "model_type": model_type,
                    "bootstrap_replicate": replicate,
                    "feature_index": feature_index,
                    "feature_name": feature_name,
                    "coefficient_raw_scale": model["coef"][feature_index],
                }
            )
    return rows


def write_summary_text(
    path,
    layer_from,
    records,
    fit_rows,
    total_local_pairs,
    elapsed_seconds,
    base_change_rmse,
    status,
):
    split_counts = {
        split: sum(record["split"] == split for record in records)
        for split in ("train", "test")
    }
    with path.open("w") as handle:
        handle.write("Local pair change SE bootstrap pilot\n")
        handle.write("=====================================\n\n")
        handle.write(f"Status:                    {status}\n")
        handle.write(f"Transition:                {layer_from} -> {layer_from + 1}\n")
        handle.write("Distance type:             raw pre-MDS\n")
        handle.write(f"Bootstrap replicates:      {N_BOOTSTRAPS}\n")
        handle.write(f"Bootstrap seed base:       {BOOTSTRAP_SEED}\n")
        handle.write(f"Proteins:                  {len(records)}\n")
        handle.write(f"Proteins by split:         {split_counts}\n")
        handle.write(f"Local pairs:               {total_local_pairs}\n")
        handle.write(
            f"Recomputed-vs-saved change RMSE: {base_change_rmse:.9g} Angstrom\n"
        )
        handle.write(f"Elapsed seconds:           {elapsed_seconds:.2f}\n\n")
        handle.write(
            "Pair-specific SE is the sample SD (ddof=1) of ten predicted changes.\n"
        )
        handle.write(
            "Each bootstrap resamples all training proteins with replacement; "
            "each sampled protein has equal total regression weight.\n"
        )
        handle.write(
            "The same refitted layer-x model is applied to both x and x+1. "
            "The intercept therefore cancels from raw changes.\n"
        )
        handle.write(
            "The naive pair SEM in change_statistics.csv is descriptive only; "
            "residue pairs within a protein are correlated.\n"
        )
        handle.write(
            "Aggregate change and SE distributions are pair-weighted; "
            "per_protein_statistics.csv provides protein-level summaries.\n"
        )
        handle.write(
            "Use train-split threshold summaries for threshold selection; "
            "leave test rows untouched for the later classifier evaluation.\n"
        )
        if fit_rows:
            ranks = [row["rank"] for row in fit_rows]
            handle.write(f"Normal-equation ranks:     {min(ranks)}..{max(ranks)}\n")


def main():
    total_t0 = time.time()
    layer_from = requested_start_layer()
    layer_to = layer_from + 1
    csv_root = Path(
        os.environ.get("CSV_ROOT") or (Path.home() / "csv_files")
    ).expanduser().resolve()
    source_dir = csv_root / VARIANT_NAME
    coefficient_path = source_dir / "coefficients.csv"
    assignment_path = source_dir / "split_assignments.csv"
    output_root = csv_root / OUTPUT_NAME
    output_dir = (
        output_root
        / "layer_shards"
        / f"transition_{layer_from:02d}_{layer_to:02d}"
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    log_progress(f"Starting transition {layer_from} -> {layer_to}")
    log_progress(f"Coefficient source: {coefficient_path}")
    log_progress(f"Split source:       {assignment_path}")
    log_progress(f"Output directory:   {output_dir}")

    saved_bundle = load_saved_coefficient_bundle(coefficient_path, layer_from)
    records = load_records(assignment_path)
    train_records = sorted(
        [record for record in records if record["split"] == "train"],
        key=lambda record: record["protein_id"],
    )
    train_index = {
        record["protein_id"]: index for index, record in enumerate(train_records)
    }
    bootstrap_counts = build_bootstrap_counts(len(train_records))

    assignment_rows = []
    for replicate in range(N_BOOTSTRAPS):
        for protein_index, record in enumerate(train_records):
            assignment_rows.append(
                {
                    "bootstrap_replicate": replicate,
                    "protein_id": record["protein_id"],
                    "draw_count": int(bootstrap_counts[replicate, protein_index]),
                }
            )
    write_csv_atomic(
        output_dir / "bootstrap_assignments.csv",
        ["bootstrap_replicate", "protein_id", "draw_count"],
        assignment_rows,
    )
    write_csv_atomic(
        output_dir / "coefficients_used.csv",
        list(saved_bundle["source_rows"][0]),
        saved_bundle["source_rows"],
    )

    fit_grams = np.zeros(
        (N_BOOTSTRAPS + 1, AUGMENTED_DIM, AUGMENTED_DIM), dtype=np.float64
    )
    fit_crosses = np.zeros((N_BOOTSTRAPS + 1, AUGMENTED_DIM), dtype=np.float64)
    local_inputs = {}
    manifest_rows = []

    write_summary_text(
        output_dir / "summary.txt",
        layer_from,
        records,
        [],
        0,
        time.time() - total_t0,
        float("nan"),
        "running: building sufficient statistics",
    )

    data_t0 = time.time()
    for record_index, record in enumerate(records, start=1):
        protein_t0 = time.time()
        protein_id = record["protein_id"]
        from_path = resolve_pair_block_path(
            record["protein_dir"], protein_id, layer_from
        )
        to_path = resolve_pair_block_path(record["protein_dir"], protein_id, layer_to)
        if not from_path.exists() or not to_path.exists():
            raise FileNotFoundError(
                f"Missing transition pair block for {protein_id}: {from_path}, {to_path}"
            )

        pb_from = np.load(str(from_path))
        pb_to = np.load(str(to_path))
        validate_pair_block(pb_from, record, layer_from)
        validate_pair_block(pb_to, record, layer_to)

        if record["split"] == "train":
            pdb_path = resolve_pdb_file(
                protein_id, record["protein_dir"], record["pdb_root"]
            )
            if pdb_path is None:
                raise FileNotFoundError(f"No predicted PDB found for {protein_id}")
            ca_data = extract_ca_coordinates(pdb_path)
            coordinates = np.asarray(ca_data["coordinates"], dtype=np.float64)
            if coordinates.shape != (int(record["n_residues"]), 3):
                raise ValueError(
                    f"{protein_id} CA coordinates {coordinates.shape} do not match "
                    f"{record['n_residues']} residues"
                )
            gram, cross = compute_equal_protein_sufficient_statistics(
                pb_from, coordinates
            )
            protein_train_index = train_index[protein_id]
            fit_grams[0] += gram
            fit_crosses[0] += cross
            for replicate in range(N_BOOTSTRAPS):
                multiplicity = int(
                    bootstrap_counts[replicate, protein_train_index]
                )
                if multiplicity:
                    fit_grams[replicate + 1] += multiplicity * gram
                    fit_crosses[replicate + 1] += multiplicity * cross
            del gram, cross, coordinates, ca_data

        local_inputs[protein_id] = extract_local_change_inputs(
            pb_from, pb_to, record, saved_bundle
        )
        n_local = int(local_inputs[protein_id]["i"].size)
        manifest_rows.append(
            {
                "variant": VARIANT_NAME,
                "layer_from": layer_from,
                "layer_to": layer_to,
                "split": record["split"],
                "source": record["source"],
                "protein_id": protein_id,
                "n_residues": record["n_residues"],
                "n_all_pairs": record["n_pairs"],
                "n_local_pairs": n_local,
                "pair_block_from": str(from_path),
                "pair_block_to": str(to_path),
                "elapsed_seconds": time.time() - protein_t0,
            }
        )
        del pb_from, pb_to
        gc.collect()

        if record_index % 10 == 0 or record_index == len(records):
            log_progress(
                f"Loaded {record_index}/{len(records)} proteins; "
                f"elapsed={time.time() - data_t0:.1f}s"
            )

    write_csv_atomic(
        output_dir / "protein_manifest.csv",
        list(manifest_rows[0]),
        manifest_rows,
    )
    log_progress(
        f"Finished data/statistics pass in {time.time() - data_t0:.1f}s; "
        "solving bootstrap models"
    )

    fitted_models = []
    fit_rows = []
    for model_index in range(N_BOOTSTRAPS + 1):
        fit_t0 = time.time()
        fitted = solve_sufficient_statistics(
            fit_grams[model_index], fit_crosses[model_index]
        )
        fitted_models.append(fitted)
        if model_index == 0:
            model_type = "recomputed_original"
            replicate = -1
            unique_proteins = len(train_records)
        else:
            model_type = "protein_bootstrap"
            replicate = model_index - 1
            unique_proteins = int(
                np.count_nonzero(bootstrap_counts[replicate])
            )
        fit_rows.append(
            {
                "variant": VARIANT_NAME,
                "layer": layer_from,
                "model_type": model_type,
                "bootstrap_replicate": replicate,
                "n_draws": len(train_records),
                "n_unique_proteins": unique_proteins,
                "rank": fitted["rank"],
                "normal_matrix_condition": fitted["normal_matrix_condition"],
                "estimated_design_condition": fitted["estimated_design_condition"],
                "scale_min": fitted["scale_min"],
                "scale_max": fitted["scale_max"],
                "intercept": fitted["intercept"],
                "coefficient_l2": float(np.linalg.norm(fitted["coef"])),
                "elapsed_seconds": time.time() - fit_t0,
            }
        )
        write_csv_atomic(
            output_dir / "bootstrap_fit_summary.csv",
            list(fit_rows[0]),
            fit_rows,
        )
        write_csv_atomic(
            output_dir / "bootstrap_coefficients.csv",
            [
                "variant",
                "layer",
                "model_type",
                "bootstrap_replicate",
                "feature_index",
                "feature_name",
                "coefficient_raw_scale",
            ],
            coefficient_rows(layer_from, fitted_models, saved_bundle),
        )
        log_progress(
            f"Solved {model_type} replicate={replicate}; rank={fitted['rank']}; "
            f"design_condition={fitted['estimated_design_condition']:.3g}"
        )

    del fit_grams, fit_crosses
    bootstrap_coefs = np.stack(
        [model["coef"] for model in fitted_models[1:]], axis=1
    )
    recomputed_coef = fitted_models[0]["coef"]

    all_changes = []
    all_pair_se = []
    all_split_codes = []
    all_separations = []
    agreement_lists = {threshold: [] for threshold in THRESHOLDS}
    per_protein_rows = []
    squared_base_difference_sum = 0.0
    base_difference_count = 0

    prediction_t0 = time.time()
    for record_index, record in enumerate(records, start=1):
        result = local_inputs[record["protein_id"]]
        feature_difference = result.pop("feature_difference")
        saved_change = result["distance_to"] - result["distance_from"]
        bootstrap_changes = feature_difference @ bootstrap_coefs
        bootstrap_mean = np.mean(bootstrap_changes, axis=1, dtype=np.float64)
        pair_se = pair_change_standard_error(bootstrap_changes)
        recomputed_change = feature_difference @ recomputed_coef

        result["saved_change"] = saved_change
        result["bootstrap_mean"] = bootstrap_mean
        result["pair_se"] = pair_se
        squared_base_difference_sum += float(
            np.sum((recomputed_change - saved_change) ** 2, dtype=np.float64)
        )
        base_difference_count += int(saved_change.size)

        all_changes.append(saved_change)
        all_pair_se.append(pair_se)
        all_split_codes.append(
            np.full(saved_change.shape, 0 if record["split"] == "train" else 1, dtype=np.int8)
        )
        all_separations.append(result["sequence_separation"])

        for threshold in THRESHOLDS:
            saved_labels = class_labels(saved_change, threshold)
            bootstrap_labels = class_labels(bootstrap_changes, threshold)
            agreement_lists[threshold].append(
                np.mean(
                    bootstrap_labels == saved_labels[:, np.newaxis],
                    axis=1,
                    dtype=np.float64,
                ).astype(np.float32)
            )

        change_summary = summarize_distribution(saved_change)
        se_summary = summarize_se_distribution(pair_se)
        per_protein_rows.append(
            {
                "variant": VARIANT_NAME,
                "distance_type": "pre_mds_raw",
                "layer_from": layer_from,
                "layer_to": layer_to,
                "split": record["split"],
                "source": record["source"],
                "protein_id": record["protein_id"],
                "n_residues": record["n_residues"],
                "n_local_pairs": int(saved_change.size),
                **{f"change_{key}": value for key, value in change_summary.items()},
                **{f"se_{key}": value for key, value in se_summary.items()},
            }
        )
        del feature_difference, bootstrap_changes, recomputed_change

        if record_index % 25 == 0 or record_index == len(records):
            log_progress(
                f"Calculated pair SE for {record_index}/{len(records)} proteins; "
                f"elapsed={time.time() - prediction_t0:.1f}s"
            )

    changes = np.concatenate(all_changes)
    pair_se = np.concatenate(all_pair_se)
    split_codes = np.concatenate(all_split_codes)
    separations = np.concatenate(all_separations)
    agreement_by_threshold = {
        threshold: np.concatenate(parts)
        for threshold, parts in agreement_lists.items()
    }
    base_change_rmse = math.sqrt(
        squared_base_difference_sum / max(base_difference_count, 1)
    )
    log_progress(
        f"Recomputed-vs-saved local-change RMSE={base_change_rmse:.6g} Angstrom"
    )
    validation_warning = base_change_rmse > 0.01
    if validation_warning:
        log_progress(
            "WARNING: recomputed sufficient-statistic fit differs from the saved "
            "model by more than 0.01 Angstrom RMSE on local changes; inspect "
            "bootstrap_fit_summary.csv before using the SE estimates"
        )

    pair_path = output_dir / "pair_specific_se.csv.gz"
    log_progress(f"Writing {changes.size:,} pair-specific rows to {pair_path.name}")
    write_pair_rows(pair_path, layer_from, records, local_inputs)

    change_rows, se_rows, bucket_rows, threshold_rows = build_summary_tables(
        layer_from,
        changes,
        pair_se,
        split_codes,
        separations,
        agreement_by_threshold,
    )
    write_csv_atomic(
        output_dir / "change_statistics.csv",
        list(change_rows[0]),
        change_rows,
    )
    write_csv_atomic(
        output_dir / "pair_se_statistics.csv",
        list(se_rows[0]),
        se_rows,
    )
    write_csv_atomic(
        output_dir / "change_buckets.csv",
        list(bucket_rows[0]),
        bucket_rows,
    )
    write_csv_atomic(
        output_dir / "threshold_stability.csv",
        list(threshold_rows[0]),
        threshold_rows,
    )
    write_csv_atomic(
        output_dir / "per_protein_statistics.csv",
        list(per_protein_rows[0]),
        per_protein_rows,
    )

    metadata = {
        "variant": VARIANT_NAME,
        "distance_type": "pre_mds_raw",
        "layer_from": layer_from,
        "layer_to": layer_to,
        "n_bootstraps": N_BOOTSTRAPS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_unit": "protein",
        "n_train_proteins": len(train_records),
        "n_test_proteins": sum(record["split"] == "test" for record in records),
        "n_local_pairs": int(changes.size),
        "sequence_separation_min": MIN_SEQUENCE_SEPARATION,
        "sequence_separation_max": MAX_SEQUENCE_SEPARATION,
        "thresholds_angstrom": list(THRESHOLDS),
        "coefficient_source": str(coefficient_path),
        "split_source": str(assignment_path),
        "feature_dtype": "float64",
        "pair_se_ddof": 1,
        "base_change_rmse_vs_saved_angstrom": base_change_rmse,
        "base_fit_validation_warning": validation_warning,
        "elapsed_seconds": time.time() - total_t0,
    }
    metadata_tmp = output_dir / "run_metadata.json.tmp"
    metadata_tmp.write_text(json.dumps(metadata, indent=2, sort_keys=True) + "\n")
    os.replace(metadata_tmp, output_dir / "run_metadata.json")

    write_summary_text(
        output_dir / "summary.txt",
        layer_from,
        records,
        fit_rows,
        int(changes.size),
        time.time() - total_t0,
        base_change_rmse,
        "complete_with_base_fit_warning" if validation_warning else "complete",
    )
    log_progress(
        f"Completed transition {layer_from}->{layer_to} in "
        f"{time.time() - total_t0:.1f}s"
    )


if __name__ == "__main__":
    main()

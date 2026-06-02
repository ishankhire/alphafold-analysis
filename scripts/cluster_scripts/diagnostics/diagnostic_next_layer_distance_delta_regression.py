#!/usr/bin/env python3
"""
Train one-protein next-layer distance-delta probes on Delta.

For each layer l=0..46, this diagnostic decodes the predicted distance at
layers l and l+1 from the saved avg-symmetric distance-probe coefficients, then
fits linear ridge probes that predict the layer-to-layer distance change from
lower-triangle pair representations and final-PDB nearest-neighbor context.
"""

import csv
import gc
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
ROOT_DIR = SCRIPT_DIR.parents[2]
if str(CLUSTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_SCRIPTS_DIR))

from cluster_regression_helpers import (  # noqa: E402
    compute_distance_matrix,
    extract_ca_coordinates,
    find_pdb_file as helper_find_pdb_file,
)


TITLE = "Diagnostic Next-Layer Distance Delta Regression"
PROTEIN_ID = "7b3a_A"
N_CHANNELS = 128
N_LAYERS = 48
DELTA_LAYERS = tuple(range(N_LAYERS - 1))
K_VALUES = (5, 10, 20)
TEST_SIZE = 0.2
RANDOM_STATE = 42
RIDGE_ALPHA = 1.0
RIDGE_SOLVER = "cholesky"
FLOAT_EPS = 1e-12

DEFAULT_COEFFICIENTS_CSV = Path(
    "/u/ikhire/general_train_test_split_csv_files/"
    "general_base256_avg_symmetric/coefficients.csv"
)
LOCAL_COEFFICIENTS_CSV = (
    ROOT_DIR / "delta_files" / "general_base256_avg_symmetric" / "coefficients.csv"
)
OUTPUT_SUBDIR = "next_layer_distance_delta_7b3a_A"

SUMMARY_CSV_NAME = "next_layer_distance_delta_summary.csv"
TARGET_SUMMARY_CSV_NAME = "next_layer_distance_delta_target_summary.csv"
SPLIT_CSV_NAME = "next_layer_distance_delta_split_assignments.csv"
PREDICTIONS_CSV_NAME = "next_layer_distance_delta_test_predictions.csv"
COEFFICIENTS_CSV_NAME = "next_layer_distance_delta_coefficients.csv"
SUMMARY_TXT_NAME = "next_layer_distance_delta_summary.txt"

VARIANT_ZERO = "zero_delta"
VARIANT_TRAIN_MEAN = "train_mean_delta"
VARIANT_CURRENT_DISTANCE = "current_distance"
VARIANT_CENTER_CURRENT = "center_pair_plus_current_distance"
VARIANT_FULL_CONTEXT = "full_context"

SUMMARY_FIELDNAMES = [
    "variant",
    "layer",
    "k_neighbors",
    "feature_dim",
    "estimator",
    "ridge_alpha",
    "n_residues",
    "n_pairs",
    "n_train_rows",
    "n_test_rows",
    "target_train_mean",
    "target_train_std",
    "target_test_mean",
    "target_test_std",
    "train_r2",
    "test_r2",
    "train_mae",
    "test_mae",
    "train_rmse",
    "test_rmse",
    "train_mean_error_pred_minus_true",
    "test_mean_error_pred_minus_true",
    "train_prediction_mean",
    "test_prediction_mean",
    "train_prediction_std",
    "test_prediction_std",
    "elapsed_seconds",
]

TARGET_SUMMARY_FIELDNAMES = [
    "layer",
    "n_pairs",
    "current_distance_mean",
    "current_distance_std",
    "next_distance_mean",
    "next_distance_std",
    "delta_mean",
    "delta_std",
    "delta_min",
    "delta_max",
    "train_delta_mean",
    "train_delta_std",
    "test_delta_mean",
    "test_delta_std",
]

SPLIT_FIELDNAMES = [
    "pair_index",
    "split",
    "residue_i_index",
    "residue_j_index",
    "residue_i_label",
    "residue_j_label",
    "seq_sep",
    "final_pdb_distance",
]

PREDICTION_FIELDNAMES = [
    "variant",
    "layer",
    "k_neighbors",
    "pair_index",
    "residue_i_index",
    "residue_j_index",
    "target_delta",
    "predicted_delta",
    "current_predicted_distance",
    "next_predicted_distance",
]

COEFFICIENT_FIELDNAMES = [
    "variant",
    "layer",
    "k_neighbors",
    "feature_dim",
    "estimator",
    "ridge_alpha",
    "feature_index",
    "feature_name",
    "coefficient_standardized",
    "coefficient_raw_scale",
]


def now_timestamp():
    """Return a readable local timestamp for progress logs."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_progress(message):
    """Emit one timestamped progress message."""
    print(f"[{now_timestamp()}] {message}", flush=True)


def format_value(value):
    """Stable CSV formatting."""
    if value is None:
        return ""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path, fieldnames, rows):
    """Write rows with stable formatting."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def initialize_append_csv(path, fieldnames):
    """Create an append-oriented CSV with only its header."""
    with path.open("w", newline="") as handle:
        csv.DictWriter(handle, fieldnames=fieldnames).writeheader()


def append_csv_rows(path, fieldnames, rows):
    """Append rows with stable formatting."""
    if not rows:
        return
    with path.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def resolve_pair_block_path(pair_blocks_root, protein_id, layer):
    """Support Delta and local protein_data pair-block layouts."""
    filename = f"{protein_id}_pair_block_{layer}.npy"
    candidates = [
        pair_blocks_root / protein_id / filename,
        pair_blocks_root / protein_id / "pair_blocks" / filename,
        pair_blocks_root / "pair_blocks" / filename,
        pair_blocks_root / filename,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_pdb_file(protein_id, pair_blocks_root, pdb_root):
    """Support Delta and local protein_data predicted-PDB layouts."""
    candidates = [
        pair_blocks_root / protein_id / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
        pdb_root / protein_id / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
        pdb_root
        / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions"
        / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
    ]
    try:
        helper_path = helper_find_pdb_file(protein_id)
    except Exception:
        helper_path = None
    if helper_path is not None:
        candidates.append(helper_path)

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def resolve_coefficients_csv():
    """Prefer the cluster coefficient path, but allow local validation."""
    raw_path = os.environ.get("DISTANCE_COEFFICIENTS_CSV")
    if raw_path:
        return Path(raw_path).resolve()
    if DEFAULT_COEFFICIENTS_CSV.exists():
        return DEFAULT_COEFFICIENTS_CSV
    return LOCAL_COEFFICIENTS_CSV


def residue_label(residue_record):
    """Render a compact PDB residue label."""
    chain_id, resname, resseq = residue_record
    return f"{chain_id}:{resname}:{resseq}"


def build_endpoint_excluded_neighbors(distance_matrix, pair_i, pair_j, max_k):
    """Build nearest-neighbor tables while excluding self and the other endpoint."""
    n_residues = int(distance_matrix.shape[0])
    if n_residues - 2 < max_k:
        raise ValueError(
            f"Need at least {max_k} context residues after endpoint exclusion; "
            f"only {n_residues - 2} available."
        )

    sorted_neighbors = []
    for center in range(n_residues):
        order = np.argsort(distance_matrix[center], kind="mergesort")
        order = order[order != center]
        sorted_neighbors.append(order.astype(np.int32, copy=False))

    n_pairs = int(pair_i.size)
    neighbors_i = np.empty((n_pairs, max_k), dtype=np.int32)
    neighbors_j = np.empty((n_pairs, max_k), dtype=np.int32)

    for row_idx in range(n_pairs):
        i = int(pair_i[row_idx])
        j = int(pair_j[row_idx])

        selected_i = []
        for candidate in sorted_neighbors[i]:
            candidate = int(candidate)
            if candidate == j:
                continue
            selected_i.append(candidate)
            if len(selected_i) == max_k:
                break

        selected_j = []
        for candidate in sorted_neighbors[j]:
            candidate = int(candidate)
            if candidate == i:
                continue
            selected_j.append(candidate)
            if len(selected_j) == max_k:
                break

        if len(selected_i) != max_k or len(selected_j) != max_k:
            raise ValueError(
                f"Could not build {max_k} endpoint-excluded neighbors for pair row {row_idx}."
            )

        neighbors_i[row_idx, :] = selected_i
        neighbors_j[row_idx, :] = selected_j

    return neighbors_i, neighbors_j


def load_protein_record(pair_blocks_root, pdb_root, protein_id, max_k):
    """Load one protein's geometry, lower-triangle rows, and neighbor tables."""
    pdb_path = resolve_pdb_file(protein_id, pair_blocks_root, pdb_root)
    if pdb_path is None:
        raise FileNotFoundError(f"No predicted PDB file found for {protein_id}")

    pb0_path = resolve_pair_block_path(pair_blocks_root, protein_id, 0)
    if not pb0_path.exists():
        raise FileNotFoundError(f"Pair block 0 not found at {pb0_path}")

    ca_data = extract_ca_coordinates(pdb_path)
    residues = ca_data["residues"]
    coordinates = np.asarray(ca_data["coordinates"], dtype=np.float64)
    n_ca = int(len(residues))
    if n_ca < 2:
        raise ValueError(f"{protein_id} has fewer than 2 CA atoms")

    pb0 = np.load(str(pb0_path), mmap_mode="r")
    pb_dim = int(pb0.shape[0])
    pb_channels = int(pb0.shape[2])
    del pb0
    if n_ca != pb_dim:
        raise ValueError(f"CA count {n_ca} != pair block dimension {pb_dim}")
    if pb_channels != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} channels, found {pb_channels}")

    pair_i, pair_j = np.tril_indices(n_ca, k=-1)
    pair_i = pair_i.astype(np.int32, copy=False)
    pair_j = pair_j.astype(np.int32, copy=False)
    final_dist = compute_distance_matrix(coordinates)
    final_pair_dist = np.asarray(final_dist[pair_i, pair_j], dtype=np.float64)
    neighbors_i, neighbors_j = build_endpoint_excluded_neighbors(
        final_dist,
        pair_i,
        pair_j,
        max_k,
    )

    return {
        "protein_id": protein_id,
        "pdb_path": pdb_path,
        "n_residues": n_ca,
        "residues": residues,
        "coordinates": coordinates,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "n_pairs": int(pair_i.size),
        "final_pair_dist": final_pair_dist,
        "neighbors_i": neighbors_i,
        "neighbors_j": neighbors_j,
    }


def load_distance_coefficients(coefficients_csv):
    """Load raw-scale avg-symmetric distance decoder coefficients."""
    intercepts = np.full((N_LAYERS,), np.nan, dtype=np.float64)
    coefficients = np.full((N_LAYERS, N_CHANNELS), np.nan, dtype=np.float64)

    with coefficients_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            try:
                layer = int(row["layer"])
                feature_index = int(row["feature_index"])
            except (KeyError, ValueError):
                continue
            if layer < 0 or layer >= N_LAYERS:
                continue

            value = float(row["coefficient_raw_scale"])
            if feature_index == -1:
                intercepts[layer] = value
            elif 0 <= feature_index < N_CHANNELS:
                coefficients[layer, feature_index] = value

    if not np.all(np.isfinite(intercepts)):
        missing = np.where(~np.isfinite(intercepts))[0].tolist()
        raise ValueError(f"Missing raw intercepts for layers: {missing}")
    if not np.all(np.isfinite(coefficients)):
        missing_layers = np.where(~np.all(np.isfinite(coefficients), axis=1))[0].tolist()
        raise ValueError(f"Missing raw coefficients for layers: {missing_layers}")

    return intercepts, coefficients


def decode_avg_symmetric_distances(pb, pair_i, pair_j, intercept, coefficients):
    """Decode pairwise distances from avg-symmetric pair reps."""
    avg_features = np.asarray(pb[pair_i, pair_j, :], dtype=np.float64)
    avg_features += np.asarray(pb[pair_j, pair_i, :], dtype=np.float64)
    avg_features *= 0.5
    return np.asarray(avg_features @ coefficients + intercept, dtype=np.float64)


def canonical_lower_features(pb, left_idx, right_idx):
    """Read lower-triangle pair reps for unordered residue pairs."""
    row_idx = np.maximum(left_idx, right_idx)
    col_idx = np.minimum(left_idx, right_idx)
    return np.asarray(pb[row_idx, col_idx, :], dtype=np.float64)


def full_context_feature_dim(k_neighbors):
    """Return feature width for center + two endpoint contexts + current distance."""
    return (1 + 2 * int(k_neighbors)) * N_CHANNELS + 1


def feature_names_for_variant(variant, k_neighbors=None):
    """Return ordered names for one fitted model variant."""
    if variant == VARIANT_CURRENT_DISTANCE:
        return ["current_predicted_distance"]

    names = [f"center_lower_ch_{idx}" for idx in range(N_CHANNELS)]
    if variant == VARIANT_CENTER_CURRENT:
        names.append("current_predicted_distance")
        return names

    if variant != VARIANT_FULL_CONTEXT:
        raise ValueError(f"Unsupported fitted variant: {variant}")

    if k_neighbors is None:
        raise ValueError("full_context requires k_neighbors")

    for rank in range(int(k_neighbors)):
        names.extend(
            f"i_neighbor_rank_{rank + 1:02d}_lower_ch_{idx}"
            for idx in range(N_CHANNELS)
        )
    for rank in range(int(k_neighbors)):
        names.extend(
            f"j_neighbor_rank_{rank + 1:02d}_lower_ch_{idx}"
            for idx in range(N_CHANNELS)
        )
    names.append("current_predicted_distance")
    return names


def build_feature_matrix(record, pb, row_indices, current_distances, variant, k_neighbors=None):
    """Materialize raw feature matrix for one variant and row subset."""
    row_indices = np.asarray(row_indices, dtype=np.int64)
    if variant == VARIANT_CURRENT_DISTANCE:
        return np.asarray(current_distances[row_indices, np.newaxis], dtype=np.float64)

    pair_i = record["pair_i"][row_indices]
    pair_j = record["pair_j"][row_indices]

    if variant == VARIANT_CENTER_CURRENT:
        x = np.empty((row_indices.size, N_CHANNELS + 1), dtype=np.float64)
        x[:, :N_CHANNELS] = canonical_lower_features(pb, pair_i, pair_j)
        x[:, N_CHANNELS] = current_distances[row_indices]
        return x

    if variant != VARIANT_FULL_CONTEXT:
        raise ValueError(f"Unsupported variant: {variant}")

    if k_neighbors is None:
        raise ValueError("full_context requires k_neighbors")
    k_neighbors = int(k_neighbors)
    x = np.empty((row_indices.size, full_context_feature_dim(k_neighbors)), dtype=np.float64)

    write_col = 0
    x[:, write_col:write_col + N_CHANNELS] = canonical_lower_features(pb, pair_i, pair_j)
    write_col += N_CHANNELS

    neighbors_i = record["neighbors_i"][row_indices, :]
    neighbors_j = record["neighbors_j"][row_indices, :]

    for rank in range(k_neighbors):
        x[:, write_col:write_col + N_CHANNELS] = canonical_lower_features(
            pb,
            pair_i,
            neighbors_i[:, rank],
        )
        write_col += N_CHANNELS

    for rank in range(k_neighbors):
        x[:, write_col:write_col + N_CHANNELS] = canonical_lower_features(
            pb,
            pair_j,
            neighbors_j[:, rank],
        )
        write_col += N_CHANNELS

    x[:, write_col] = current_distances[row_indices]
    return x


def compute_r2(y_true, y_pred):
    """Compute unweighted R-squared."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    mean_true = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot <= FLOAT_EPS:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def compute_metrics(y_true, y_pred):
    """Compute compact regression metrics."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    err = y_pred - y_true
    return {
        "r2": compute_r2(y_true, y_pred),
        "mae": float(np.mean(np.abs(err))) if y_true.size else float("nan"),
        "rmse": (
            float(math.sqrt(np.mean(err * err))) if y_true.size else float("nan")
        ),
        "mean_error_pred_minus_true": (
            float(np.mean(err)) if y_true.size else float("nan")
        ),
        "prediction_mean": float(np.mean(y_pred)) if y_pred.size else float("nan"),
        "prediction_std": float(np.std(y_pred)) if y_pred.size else float("nan"),
    }


def standardize_in_place(x, mean, scale):
    """Apply train-set standardization in place."""
    x -= mean
    x /= scale
    return x


def fit_ridge_probe(x_train_raw, y_train, x_test_raw):
    """Fit a standardized dense ridge probe and return predictions and coefficients."""
    if not np.all(np.isfinite(x_train_raw)):
        raise ValueError("Non-finite values found in train features")
    if not np.all(np.isfinite(x_test_raw)):
        raise ValueError("Non-finite values found in test features")

    mean = np.mean(x_train_raw, axis=0, dtype=np.float64)
    scale = np.std(x_train_raw, axis=0, dtype=np.float64)
    scale[scale < FLOAT_EPS] = 1.0

    standardize_in_place(x_train_raw, mean, scale)
    standardize_in_place(x_test_raw, mean, scale)

    model = Ridge(
        alpha=RIDGE_ALPHA,
        fit_intercept=True,
        copy_X=False,
        solver=RIDGE_SOLVER,
        random_state=RANDOM_STATE,
    )
    model.fit(x_train_raw, y_train)
    y_pred_train = np.asarray(model.predict(x_train_raw), dtype=np.float64)
    y_pred_test = np.asarray(model.predict(x_test_raw), dtype=np.float64)

    coef_std = np.asarray(model.coef_, dtype=np.float64)
    intercept_std = float(model.intercept_)
    coef_raw = coef_std / scale
    intercept_raw = float(intercept_std - np.sum((coef_std * mean) / scale, dtype=np.float64))

    return {
        "model": model,
        "coef_std": coef_std,
        "intercept_std": intercept_std,
        "coef_raw": coef_raw,
        "intercept_raw": intercept_raw,
        "y_pred_train": y_pred_train,
        "y_pred_test": y_pred_test,
    }


def build_summary_row(
    record,
    variant,
    layer,
    k_neighbors,
    feature_dim,
    estimator,
    y_train,
    y_test,
    y_pred_train,
    y_pred_test,
    elapsed_seconds,
):
    """Build one model/baseline summary row."""
    train_metrics = compute_metrics(y_train, y_pred_train)
    test_metrics = compute_metrics(y_test, y_pred_test)
    return {
        "variant": variant,
        "layer": int(layer),
        "k_neighbors": k_neighbors,
        "feature_dim": int(feature_dim),
        "estimator": estimator,
        "ridge_alpha": RIDGE_ALPHA if estimator == "ridge" else "",
        "n_residues": int(record["n_residues"]),
        "n_pairs": int(record["n_pairs"]),
        "n_train_rows": int(y_train.size),
        "n_test_rows": int(y_test.size),
        "target_train_mean": float(np.mean(y_train)),
        "target_train_std": float(np.std(y_train)),
        "target_test_mean": float(np.mean(y_test)),
        "target_test_std": float(np.std(y_test)),
        "train_r2": train_metrics["r2"],
        "test_r2": test_metrics["r2"],
        "train_mae": train_metrics["mae"],
        "test_mae": test_metrics["mae"],
        "train_rmse": train_metrics["rmse"],
        "test_rmse": test_metrics["rmse"],
        "train_mean_error_pred_minus_true": train_metrics["mean_error_pred_minus_true"],
        "test_mean_error_pred_minus_true": test_metrics["mean_error_pred_minus_true"],
        "train_prediction_mean": train_metrics["prediction_mean"],
        "test_prediction_mean": test_metrics["prediction_mean"],
        "train_prediction_std": train_metrics["prediction_std"],
        "test_prediction_std": test_metrics["prediction_std"],
        "elapsed_seconds": float(elapsed_seconds),
    }


def build_target_summary_row(layer, current_dist, next_dist, delta, train_idx, test_idx):
    """Build one target-distribution summary row."""
    return {
        "layer": int(layer),
        "n_pairs": int(delta.size),
        "current_distance_mean": float(np.mean(current_dist)),
        "current_distance_std": float(np.std(current_dist)),
        "next_distance_mean": float(np.mean(next_dist)),
        "next_distance_std": float(np.std(next_dist)),
        "delta_mean": float(np.mean(delta)),
        "delta_std": float(np.std(delta)),
        "delta_min": float(np.min(delta)),
        "delta_max": float(np.max(delta)),
        "train_delta_mean": float(np.mean(delta[train_idx])),
        "train_delta_std": float(np.std(delta[train_idx])),
        "test_delta_mean": float(np.mean(delta[test_idx])),
        "test_delta_std": float(np.std(delta[test_idx])),
    }


def append_prediction_rows(
    output_dir,
    record,
    variant,
    layer,
    k_neighbors,
    test_idx,
    y_test,
    y_pred_test,
    current_dist,
    next_dist,
):
    """Append per-pair test predictions for one completed variant."""
    pair_i = record["pair_i"]
    pair_j = record["pair_j"]
    rows = []
    for local_idx, pair_index in enumerate(test_idx):
        pair_index = int(pair_index)
        rows.append(
            {
                "variant": variant,
                "layer": int(layer),
                "k_neighbors": k_neighbors,
                "pair_index": pair_index,
                "residue_i_index": int(pair_i[pair_index]),
                "residue_j_index": int(pair_j[pair_index]),
                "target_delta": float(y_test[local_idx]),
                "predicted_delta": float(y_pred_test[local_idx]),
                "current_predicted_distance": float(current_dist[pair_index]),
                "next_predicted_distance": float(next_dist[pair_index]),
            }
        )
    append_csv_rows(
        output_dir / PREDICTIONS_CSV_NAME,
        PREDICTION_FIELDNAMES,
        rows,
    )


def append_coefficient_rows(
    output_dir,
    variant,
    layer,
    k_neighbors,
    fitted,
    feature_names,
):
    """Append standardized and raw-scale coefficient rows for one fitted probe."""
    rows = [
        {
            "variant": variant,
            "layer": int(layer),
            "k_neighbors": k_neighbors,
            "feature_dim": int(len(feature_names)),
            "estimator": "ridge",
            "ridge_alpha": RIDGE_ALPHA,
            "feature_index": -1,
            "feature_name": "intercept",
            "coefficient_standardized": float(fitted["intercept_std"]),
            "coefficient_raw_scale": float(fitted["intercept_raw"]),
        }
    ]
    for feature_index, feature_name in enumerate(feature_names):
        rows.append(
            {
                "variant": variant,
                "layer": int(layer),
                "k_neighbors": k_neighbors,
                "feature_dim": int(len(feature_names)),
                "estimator": "ridge",
                "ridge_alpha": RIDGE_ALPHA,
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "coefficient_standardized": float(fitted["coef_std"][feature_index]),
                "coefficient_raw_scale": float(fitted["coef_raw"][feature_index]),
            }
        )

    append_csv_rows(
        output_dir / COEFFICIENTS_CSV_NAME,
        COEFFICIENT_FIELDNAMES,
        rows,
    )


def write_split_assignments(output_dir, record, train_idx, test_idx):
    """Write the deterministic pair-level train/test split."""
    split = np.full((record["n_pairs"],), "test", dtype=object)
    split[train_idx] = "train"
    pair_i = record["pair_i"]
    pair_j = record["pair_j"]
    residues = record["residues"]
    rows = []
    for pair_index in range(record["n_pairs"]):
        i = int(pair_i[pair_index])
        j = int(pair_j[pair_index])
        rows.append(
            {
                "pair_index": int(pair_index),
                "split": split[pair_index],
                "residue_i_index": i,
                "residue_j_index": j,
                "residue_i_label": residue_label(residues[i]),
                "residue_j_label": residue_label(residues[j]),
                "seq_sep": int(abs(i - j)),
                "final_pdb_distance": float(record["final_pair_dist"][pair_index]),
            }
        )
    write_csv(output_dir / SPLIT_CSV_NAME, SPLIT_FIELDNAMES, rows)


def estimate_feature_matrix_mib(record, k_values):
    """Estimate all-row float64 feature matrix memory for logging."""
    estimates = {}
    n_pairs = int(record["n_pairs"])
    estimates[VARIANT_CURRENT_DISTANCE] = n_pairs * 1 * 8 / (1024 ** 2)
    estimates[VARIANT_CENTER_CURRENT] = n_pairs * (N_CHANNELS + 1) * 8 / (1024 ** 2)
    for k_neighbors in k_values:
        estimates[f"{VARIANT_FULL_CONTEXT}_k{k_neighbors}"] = (
            n_pairs * full_context_feature_dim(k_neighbors) * 8 / (1024 ** 2)
        )
    return estimates


def write_summary_txt(
    output_dir,
    pair_blocks_root,
    pdb_root,
    coefficients_csv,
    record,
    layer_indices,
    k_values,
    summary_rows,
    target_summary_rows,
    elapsed_seconds,
):
    """Write a compact run summary/checkpoint manifest."""
    memory_estimates = estimate_feature_matrix_mib(record, k_values)
    lines = [
        TITLE,
        "=" * len(TITLE),
        f"Last update:             {now_timestamp()}",
        f"Protein:                 {record['protein_id']}",
        f"Residues:                {record['n_residues']}",
        f"Lower-triangle pairs:    {record['n_pairs']}",
        f"Pair blocks root:        {pair_blocks_root}",
        f"PDB root:                {pdb_root}",
        f"Coefficients CSV:        {coefficients_csv}",
        f"Layer indices:           {','.join(str(layer) for layer in layer_indices)}",
        f"k values:                {','.join(str(k) for k in k_values)}",
        f"Split:                   random lower-triangle 80/20, seed={RANDOM_STATE}",
        f"Estimator:               Ridge(alpha={RIDGE_ALPHA}, solver={RIDGE_SOLVER})",
        f"Completed target rows:   {len(target_summary_rows)}",
        f"Completed summary rows:  {len(summary_rows)}",
        f"Elapsed seconds:         {elapsed_seconds:.1f}",
        "",
        "All-row float64 matrix estimates:",
    ]
    for name, mib in memory_estimates.items():
        lines.append(f"- {name}: {mib:.1f} MiB")
    (output_dir / SUMMARY_TXT_NAME).write_text("\n".join(lines) + "\n")


def checkpoint_outputs(
    output_dir,
    pair_blocks_root,
    pdb_root,
    coefficients_csv,
    record,
    layer_indices,
    k_values,
    summary_rows,
    target_summary_rows,
    t0,
):
    """Flush compact checkpoint outputs."""
    write_csv(output_dir / SUMMARY_CSV_NAME, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(
        output_dir / TARGET_SUMMARY_CSV_NAME,
        TARGET_SUMMARY_FIELDNAMES,
        target_summary_rows,
    )
    write_summary_txt(
        output_dir,
        pair_blocks_root,
        pdb_root,
        coefficients_csv,
        record,
        layer_indices,
        k_values,
        summary_rows,
        target_summary_rows,
        time.time() - t0,
    )


def run_constant_baseline(
    output_dir,
    record,
    variant,
    layer,
    constant_value,
    y_train,
    y_test,
    test_idx,
    current_dist,
    next_dist,
):
    """Evaluate and persist one constant baseline."""
    t0 = time.time()
    y_pred_train = np.full(y_train.shape, float(constant_value), dtype=np.float64)
    y_pred_test = np.full(y_test.shape, float(constant_value), dtype=np.float64)
    summary_row = build_summary_row(
        record,
        variant,
        layer,
        "",
        0,
        "constant",
        y_train,
        y_test,
        y_pred_train,
        y_pred_test,
        time.time() - t0,
    )
    append_prediction_rows(
        output_dir,
        record,
        variant,
        layer,
        "",
        test_idx,
        y_test,
        y_pred_test,
        current_dist,
        next_dist,
    )
    return summary_row


def run_fitted_variant(
    output_dir,
    record,
    pb,
    variant,
    layer,
    k_neighbors,
    train_idx,
    test_idx,
    y_train,
    y_test,
    current_dist,
    next_dist,
):
    """Fit, evaluate, and persist one ridge probe variant."""
    variant_label = variant if k_neighbors is None else f"{variant}_k{k_neighbors}"
    log_progress(f"Layer {layer:02d} {variant_label}: building features")
    t0 = time.time()
    x_train = build_feature_matrix(record, pb, train_idx, current_dist, variant, k_neighbors)
    x_test = build_feature_matrix(record, pb, test_idx, current_dist, variant, k_neighbors)
    feature_names = feature_names_for_variant(variant, k_neighbors)
    if x_train.shape[1] != len(feature_names):
        raise ValueError(
            f"Feature-name mismatch for {variant_label}: "
            f"{x_train.shape[1]} != {len(feature_names)}"
        )

    feature_elapsed = time.time() - t0
    log_progress(
        f"Layer {layer:02d} {variant_label}: fitting ridge "
        f"({x_train.shape[0]:,} train rows, {x_train.shape[1]:,} features, "
        f"feature build={feature_elapsed:.1f}s)"
    )
    fit_t0 = time.time()
    fitted = fit_ridge_probe(x_train, y_train, x_test)
    fit_elapsed = time.time() - fit_t0

    y_pred_train = fitted["y_pred_train"]
    y_pred_test = fitted["y_pred_test"]
    summary_row = build_summary_row(
        record,
        variant,
        layer,
        k_neighbors if k_neighbors is not None else "",
        x_train.shape[1],
        "ridge",
        y_train,
        y_test,
        y_pred_train,
        y_pred_test,
        time.time() - t0,
    )
    append_prediction_rows(
        output_dir,
        record,
        variant,
        layer,
        k_neighbors if k_neighbors is not None else "",
        test_idx,
        y_test,
        y_pred_test,
        current_dist,
        next_dist,
    )
    append_coefficient_rows(
        output_dir,
        variant,
        layer,
        k_neighbors if k_neighbors is not None else "",
        fitted,
        feature_names,
    )

    log_progress(
        f"Layer {layer:02d} {variant_label}: "
        f"test_R2={summary_row['test_r2']:.4f}, "
        f"test_RMSE={summary_row['test_rmse']:.4f}, fit={fit_elapsed:.1f}s"
    )

    del x_train
    del x_test
    del fitted["model"]
    del fitted
    gc.collect()
    return summary_row


def run_experiment(
    pair_blocks_root,
    pdb_root,
    coefficients_csv,
    output_dir,
    protein_id=PROTEIN_ID,
    layer_indices=DELTA_LAYERS,
    k_values=K_VALUES,
):
    """Run the checkpointed next-layer distance-delta experiment."""
    t0 = time.time()
    pair_blocks_root = Path(pair_blocks_root).resolve()
    pdb_root = Path(pdb_root).resolve()
    coefficients_csv = Path(coefficients_csv).resolve()
    output_dir = Path(output_dir).resolve()
    layer_indices = tuple(int(layer) for layer in layer_indices)
    k_values = tuple(int(k) for k in k_values)
    if not layer_indices:
        raise ValueError("At least one layer index is required.")
    if any(layer < 0 or layer >= N_LAYERS - 1 for layer in layer_indices):
        raise ValueError(f"Layer indices must be in 0..{N_LAYERS - 2}.")
    if not k_values:
        raise ValueError("At least one k value is required.")

    output_dir.mkdir(parents=True, exist_ok=True)
    initialize_append_csv(output_dir / PREDICTIONS_CSV_NAME, PREDICTION_FIELDNAMES)
    initialize_append_csv(output_dir / COEFFICIENTS_CSV_NAME, COEFFICIENT_FIELDNAMES)

    log_progress(TITLE)
    log_progress(f"Pair blocks root: {pair_blocks_root}")
    log_progress(f"PDB root:         {pdb_root}")
    log_progress(f"Coefficients CSV: {coefficients_csv}")
    log_progress(f"Output dir:       {output_dir}")

    record = load_protein_record(pair_blocks_root, pdb_root, protein_id, max(k_values))
    distance_intercepts, distance_coefficients = load_distance_coefficients(coefficients_csv)

    all_indices = np.arange(record["n_pairs"], dtype=np.int64)
    train_idx, test_idx = train_test_split(
        all_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        shuffle=True,
    )
    train_idx = np.sort(np.asarray(train_idx, dtype=np.int64))
    test_idx = np.sort(np.asarray(test_idx, dtype=np.int64))
    write_split_assignments(output_dir, record, train_idx, test_idx)

    log_progress(
        f"Loaded {record['protein_id']}: {record['n_residues']} residues, "
        f"{record['n_pairs']:,} lower-triangle pairs, "
        f"{train_idx.size:,} train, {test_idx.size:,} test"
    )
    for name, mib in estimate_feature_matrix_mib(record, k_values).items():
        log_progress(f"Memory estimate {name}: all-row float64 {mib:.1f} MiB")

    summary_rows = []
    target_summary_rows = []
    checkpoint_outputs(
        output_dir,
        pair_blocks_root,
        pdb_root,
        coefficients_csv,
        record,
        layer_indices,
        k_values,
        summary_rows,
        target_summary_rows,
        t0,
    )

    pair_i = record["pair_i"]
    pair_j = record["pair_j"]

    for layer in layer_indices:
        layer_t0 = time.time()
        log_progress(f"Layer {layer:02d}: loading pair blocks and decoding target")
        pb_path = resolve_pair_block_path(pair_blocks_root, protein_id, layer)
        next_pb_path = resolve_pair_block_path(pair_blocks_root, protein_id, layer + 1)
        pb = np.load(str(pb_path))
        next_pb = np.load(str(next_pb_path))

        current_dist = decode_avg_symmetric_distances(
            pb,
            pair_i,
            pair_j,
            distance_intercepts[layer],
            distance_coefficients[layer],
        )
        next_dist = decode_avg_symmetric_distances(
            next_pb,
            pair_i,
            pair_j,
            distance_intercepts[layer + 1],
            distance_coefficients[layer + 1],
        )
        del next_pb

        delta = np.asarray(next_dist - current_dist, dtype=np.float64)
        if not np.all(np.isfinite(delta)):
            raise ValueError(f"Non-finite target values for layer {layer}")

        target_summary_rows.append(
            build_target_summary_row(layer, current_dist, next_dist, delta, train_idx, test_idx)
        )
        y_train = np.asarray(delta[train_idx], dtype=np.float64)
        y_test = np.asarray(delta[test_idx], dtype=np.float64)

        zero_row = run_constant_baseline(
            output_dir,
            record,
            VARIANT_ZERO,
            layer,
            0.0,
            y_train,
            y_test,
            test_idx,
            current_dist,
            next_dist,
        )
        summary_rows.append(zero_row)
        checkpoint_outputs(
            output_dir,
            pair_blocks_root,
            pdb_root,
            coefficients_csv,
            record,
            layer_indices,
            k_values,
            summary_rows,
            target_summary_rows,
            t0,
        )

        train_mean_row = run_constant_baseline(
            output_dir,
            record,
            VARIANT_TRAIN_MEAN,
            layer,
            float(np.mean(y_train)),
            y_train,
            y_test,
            test_idx,
            current_dist,
            next_dist,
        )
        summary_rows.append(train_mean_row)
        checkpoint_outputs(
            output_dir,
            pair_blocks_root,
            pdb_root,
            coefficients_csv,
            record,
            layer_indices,
            k_values,
            summary_rows,
            target_summary_rows,
            t0,
        )

        for variant, k_neighbors in [
            (VARIANT_CURRENT_DISTANCE, None),
            (VARIANT_CENTER_CURRENT, None),
        ]:
            summary_rows.append(
                run_fitted_variant(
                    output_dir,
                    record,
                    pb,
                    variant,
                    layer,
                    k_neighbors,
                    train_idx,
                    test_idx,
                    y_train,
                    y_test,
                    current_dist,
                    next_dist,
                )
            )
            checkpoint_outputs(
                output_dir,
                pair_blocks_root,
                pdb_root,
                coefficients_csv,
                record,
                layer_indices,
                k_values,
                summary_rows,
                target_summary_rows,
                t0,
            )

        for k_neighbors in k_values:
            summary_rows.append(
                run_fitted_variant(
                    output_dir,
                    record,
                    pb,
                    VARIANT_FULL_CONTEXT,
                    layer,
                    k_neighbors,
                    train_idx,
                    test_idx,
                    y_train,
                    y_test,
                    current_dist,
                    next_dist,
                )
            )
            checkpoint_outputs(
                output_dir,
                pair_blocks_root,
                pdb_root,
                coefficients_csv,
                record,
                layer_indices,
                k_values,
                summary_rows,
                target_summary_rows,
                t0,
            )

        del pb
        del current_dist
        del next_dist
        del delta
        gc.collect()
        log_progress(
            f"Layer {layer:02d}: finished in {time.time() - layer_t0:.1f}s "
            f"(total {time.time() - t0:.1f}s)"
        )

    checkpoint_outputs(
        output_dir,
        pair_blocks_root,
        pdb_root,
        coefficients_csv,
        record,
        layer_indices,
        k_values,
        summary_rows,
        target_summary_rows,
        t0,
    )
    log_progress(f"Completed run in {time.time() - t0:.1f}s")


def main():
    pair_blocks_root = Path(
        os.environ.get("PAIR_BLOCKS_ROOT", ROOT_DIR.parent / "protein_data")
    )
    pdb_root = Path(os.environ.get("PDB_ROOT", ROOT_DIR.parent / "protein_data"))
    csv_root = os.environ.get("CSV_ROOT")
    if not csv_root:
        raise ValueError("CSV_ROOT must be set for output files.")
    coefficients_csv = resolve_coefficients_csv()
    run_experiment(
        pair_blocks_root=pair_blocks_root,
        pdb_root=pdb_root,
        coefficients_csv=coefficients_csv,
        output_dir=Path(csv_root) / OUTPUT_SUBDIR,
    )


if __name__ == "__main__":
    main()

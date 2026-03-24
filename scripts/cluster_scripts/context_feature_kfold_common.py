#!/usr/bin/env python3
"""
Shared helpers for 48-layer protein-level 5-fold CV context-feature regressions.
"""

import csv
import gc
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.model_selection import StratifiedKFold

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cluster_regression_helpers import (  # noqa: E402
    analyze_protein_geometry,
    compute_distance_matrix,
    discover_proteins,
    extract_ca_coordinates,
    find_pdb_file as helper_find_pdb_file,
    get_pair_block_path as helper_get_pair_block_path,
)


N_CHANNELS = 128
N_LAYERS = 48
N_FOLDS = 5
N_SIZE_BINS = 5
RANDOM_STATE = 42
FLOAT_EPS = 1e-12

AXIAL_NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

FOLD_SUMMARY_CSV_NAME = "fold_summary.csv"
PER_PROTEIN_CSV_NAME = "per_protein.csv"
COEFFICIENTS_CSV_NAME = "coefficients.csv"
FOLD_ASSIGNMENTS_CSV_NAME = "fold_assignments.csv"
GEOMETRY_METRICS_CSV_NAME = "geometry_metrics_last_fold.csv"
GEOMETRY_SUMMARY_CSV_NAME = "geometry_summary_last_fold.csv"
SUMMARY_TXT_NAME = "summary.txt"

FOLD_SUMMARY_FIELDNAMES = [
    "variant",
    "feature_dim",
    "fold",
    "layer",
    "n_train_proteins",
    "n_test_proteins",
    "n_train_rows",
    "n_test_rows",
    "train_mean_per_protein_r2",
    "test_mean_per_protein_r2",
    "train_pair_weighted_r2",
    "test_pair_weighted_r2",
    "train_pair_weighted_mae",
    "test_pair_weighted_mae",
    "train_pair_weighted_rmse",
    "test_pair_weighted_rmse",
    "train_mean_bias",
    "test_mean_bias",
    "elapsed_seconds",
]

PER_PROTEIN_FIELDNAMES = [
    "variant",
    "feature_dim",
    "fold",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
    "per_protein_r2",
    "per_protein_mse",
    "per_protein_mae",
    "per_protein_rmse",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
]

COEFFICIENT_FIELDNAMES = [
    "variant",
    "feature_dim",
    "fold",
    "layer",
    "feature_index",
    "feature_name",
    "coefficient_standardized",
    "coefficient_raw_scale",
]

FOLD_ASSIGNMENT_FIELDNAMES = [
    "fold",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
]

GEOMETRY_METRIC_KEYS = [
    "per_protein_r2",
    "tm_score",
    "rmsd",
    "stress",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "neg_eigenvalue_fraction",
    "n_negative_predictions",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
]

GEOMETRY_METRICS_FIELDNAMES = [
    "variant",
    "feature_dim",
    "fold",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "per_protein_r2",
    "rmsd",
    "tm_score",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "neg_eigenvalue_fraction",
    "n_negative_predictions",
    "stress",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
]

GEOMETRY_SUMMARY_FIELDNAMES = ["variant", "feature_dim", "fold", "layer", "split", "n_proteins"]
for _metric_key in GEOMETRY_METRIC_KEYS:
    GEOMETRY_SUMMARY_FIELDNAMES.extend([f"{_metric_key}_mean", f"{_metric_key}_std"])


def now_timestamp():
    """Return a readable local timestamp for progress logs."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_value(value):
    """Stable CSV formatting shared across outputs."""
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


def make_variant(name, use_axial=False, use_self_diag=False):
    """Construct one fixed feature specification."""
    return {
        "name": name,
        "use_axial": bool(use_axial),
        "use_self_diag": bool(use_self_diag),
    }


def format_offset_label(offset):
    """Create stable feature labels for context offsets."""
    di, dj = offset
    return f"di_{di:+d}_dj_{dj:+d}".replace("+", "p").replace("-", "m")


def feature_names_for_variant(variant):
    """Return ordered feature names for one fixed feature variant."""
    names = [f"pair_ij_ch_{idx}" for idx in range(N_CHANNELS)]
    names.extend([f"pair_ji_ch_{idx}" for idx in range(N_CHANNELS)])

    if variant["use_axial"]:
        for offset in AXIAL_NEIGHBOR_OFFSETS:
            label = format_offset_label(offset)
            names.extend([f"context_{label}_ch_{idx}" for idx in range(N_CHANNELS)])

    if variant["use_self_diag"]:
        names.extend([f"diag_ii_ch_{idx}" for idx in range(N_CHANNELS)])
        names.extend([f"diag_jj_ch_{idx}" for idx in range(N_CHANNELS)])

    return names


def feature_dim_for_variant(variant):
    """Return the dense feature width for one variant."""
    return len(feature_names_for_variant(variant))


def resolve_pair_block_path(protein_dir, protein_id, layer):
    """Support both Delta-style and local protein_data layouts."""
    candidates = [
        helper_get_pair_block_path(protein_dir, protein_id, layer),
        protein_dir / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_pdb_file(protein_id, protein_dir, pdb_root):
    """Support both Delta-style and local protein_data PDB layouts."""
    candidates = [protein_dir / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb"]

    try:
        helper_path = helper_find_pdb_file(protein_id)
    except Exception:
        helper_path = None
    if helper_path is not None:
        candidates.append(helper_path)

    candidates.extend(
        [
            pdb_root
            / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
            / "predictions"
            / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
            pdb_root / protein_id / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
        ]
    )

    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def load_protein_records(pair_blocks_root, pdb_root):
    """Load all valid proteins and attach pair targets plus geometry metadata."""
    all_proteins = discover_proteins(pair_blocks_root)
    protein_data = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        try:
            pdb_path = resolve_pdb_file(protein_id, protein_dir, pdb_root)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                continue

            pb0_path = resolve_pair_block_path(protein_dir, protein_id, 0)
            if not pb0_path.exists():
                skipped.append((protein_id, "pair block 0 not found"))
                continue

            ca_data = extract_ca_coordinates(pdb_path)
            n_ca = len(ca_data["residues"])
            if n_ca < 2:
                skipped.append((protein_id, "fewer than 2 CA atoms"))
                continue

            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = int(pb0.shape[0])
            del pb0
            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                continue

            lt_i, lt_j = np.tril_indices(n_ca, k=-1)
            if lt_i.size == 0:
                skipped.append((protein_id, "no lower-triangle residue pairs"))
                continue

            coords = np.asarray(ca_data["coordinates"], dtype=np.float64)
            dist_matrix = compute_distance_matrix(coords)
            y_abs = np.asarray(dist_matrix[lt_i, lt_j], dtype=np.float32)
            target_variance = float(np.var(y_abs, dtype=np.float64))
            if target_variance <= FLOAT_EPS:
                target_variance = FLOAT_EPS

            protein_data.append(
                {
                    "protein_id": protein_id,
                    "protein_dir": protein_dir,
                    "n_residues": int(n_ca),
                    "lt_i": lt_i,
                    "lt_j": lt_j,
                    "coordinates": coords,
                    "y_abs": y_abs,
                    "n_pairs": int(y_abs.size),
                    "target_variance": target_variance,
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            traceback.print_exc()

    return all_proteins, protein_data, skipped


def assign_size_bins(records, n_bins):
    """Assign stable rank-based size bins using residue count."""
    ordered = sorted(records, key=lambda row: (row["n_residues"], row["protein_id"]))
    total = len(ordered)
    for rank, record in enumerate(ordered):
        size_bin = int((rank * n_bins) // max(total, 1))
        if size_bin >= n_bins:
            size_bin = n_bins - 1
        record["size_bin"] = int(size_bin)


def build_strict_stratified_folds(records):
    """Build one strict reusable protein-level fold family."""
    if len(records) < N_FOLDS:
        raise ValueError(
            f"Need at least {N_FOLDS} proteins for {N_FOLDS}-fold CV; got {len(records)}."
        )

    labels = np.asarray([record["size_bin"] for record in records], dtype=np.int32)
    unique, counts = np.unique(labels, return_counts=True)
    count_map = {int(label): int(count) for label, count in zip(unique, counts)}
    if unique.size < N_SIZE_BINS or np.min(counts) < N_FOLDS:
        raise ValueError(
            "Strict size-stratified 5-fold CV is infeasible. "
            f"Need >= {N_FOLDS} proteins per size bin after assigning {N_SIZE_BINS} bins; "
            f"observed counts: {count_map}"
        )

    indices = np.arange(len(records), dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=N_FOLDS,
        shuffle=True,
        random_state=RANDOM_STATE,
    )

    folds = []
    assignment_rows = []
    for fold_index, (train_idx, test_idx) in enumerate(splitter.split(indices, labels)):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        train_records = [records[int(idx)] for idx in train_idx]
        test_records = [records[int(idx)] for idx in test_idx]
        folds.append(
            {
                "fold_index": int(fold_index),
                "train_records": train_records,
                "test_records": test_records,
            }
        )

        for split_name, split_idx in [("train", train_idx), ("test", test_idx)]:
            for idx in split_idx:
                record = records[int(idx)]
                assignment_rows.append(
                    {
                        "fold": int(fold_index),
                        "split": split_name,
                        "protein_id": record["protein_id"],
                        "n_residues": int(record["n_residues"]),
                        "size_bin": int(record["size_bin"]),
                        "n_pairs": int(record["n_pairs"]),
                        "target_variance": float(record["target_variance"]),
                    }
                )

    return folds, assignment_rows, count_map


def build_context_chunk(pb, pair_i, pair_j, offsets):
    """Build zero-padded lower-triangle context features for one protein."""
    n_pairs = pair_i.size
    chunk = np.zeros((n_pairs, len(offsets) * N_CHANNELS), dtype=np.float32)
    n_residues = int(pb.shape[0])
    write_col = 0

    for di, dj in offsets:
        src_i = pair_i + di
        src_j = pair_j + dj
        valid = (
            (src_i >= 0)
            & (src_i < n_residues)
            & (src_j >= 0)
            & (src_j < n_residues)
            & (src_i > src_j)
        )
        if np.any(valid):
            chunk[valid, write_col:write_col + N_CHANNELS] = pb[src_i[valid], src_j[valid], :]
        write_col += N_CHANNELS

    return chunk


def build_feature_chunk_from_pb(record, pb, variant):
    """Build one protein's fixed feature matrix from an already-open pair block."""
    pair_i = record["lt_i"]
    pair_j = record["lt_j"]
    base_ij = np.asarray(pb[pair_i, pair_j, :], dtype=np.float32)
    base_ji = np.asarray(pb[pair_j, pair_i, :], dtype=np.float32)
    pieces = [np.concatenate([base_ij, base_ji], axis=1).astype(np.float32, copy=False)]

    if variant["use_axial"]:
        pieces.append(build_context_chunk(pb, pair_i, pair_j, AXIAL_NEIGHBOR_OFFSETS))

    if variant["use_self_diag"]:
        diag_chunk = np.empty((int(record["n_pairs"]), 2 * N_CHANNELS), dtype=np.float32)
        diag_chunk[:, :N_CHANNELS] = pb[pair_i, pair_i, :]
        diag_chunk[:, N_CHANNELS:] = pb[pair_j, pair_j, :]
        pieces.append(diag_chunk)

    if len(pieces) == 1:
        return pieces[0]
    return np.concatenate(pieces, axis=1).astype(np.float32, copy=False)


def load_layer_feature_cache(records, layer, variant):
    """Load one layer's features once per protein and reuse them across all folds."""
    cache = {}
    total_rows = 0

    for record_index, record in enumerate(records, start=1):
        pb_path = resolve_pair_block_path(record["protein_dir"], record["protein_id"], layer)
        pb = np.load(str(pb_path), mmap_mode="r")
        chunk = build_feature_chunk_from_pb(record, pb, variant)
        del pb

        if chunk.shape[0] != int(record["n_pairs"]):
            raise ValueError(
                f"Feature row mismatch for {record['protein_id']} layer {layer}: "
                f"{chunk.shape[0]} != {record['n_pairs']}"
            )

        cache[record["protein_id"]] = chunk
        total_rows += int(chunk.shape[0])

        if record_index % 25 == 0 or record_index == len(records):
            print(
                f"[{now_timestamp()}] Layer {layer:02d}: cached {record_index}/{len(records)} proteins "
                f"({total_rows:,} rows)",
                flush=True,
            )

    return cache


def compute_scaler_from_cache(train_records, cache, variant):
    """Compute train-only protein-equal mean and std for one feature variant."""
    feature_dim = feature_dim_for_variant(variant)
    mean_acc = np.zeros((feature_dim,), dtype=np.float64)
    sq_acc = np.zeros((feature_dim,), dtype=np.float64)

    for record in train_records:
        chunk = cache[record["protein_id"]]
        mean_acc += np.asarray(np.mean(chunk, axis=0, dtype=np.float64), dtype=np.float64)
        sq_acc += np.asarray(np.mean(chunk * chunk, axis=0, dtype=np.float64), dtype=np.float64)

    mean = mean_acc / max(len(train_records), 1)
    second_moment = sq_acc / max(len(train_records), 1)
    var = np.maximum(second_moment - mean * mean, 0.0)
    scale = np.sqrt(var)
    scale[scale < FLOAT_EPS] = 1.0
    return mean, scale


def compute_r2(y_true, y_pred):
    """Compute unweighted pooled R-squared."""
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


def compute_mae(y_true, y_pred):
    """Compute MAE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true, y_pred):
    """Compute RMSE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def standardized_feature_block(raw_chunk, scaler_mean, scaler_scale):
    """Return one protein's standardized feature matrix in float64."""
    x_block = np.asarray(raw_chunk, dtype=np.float64).copy()
    x_block -= scaler_mean
    x_block /= scaler_scale
    return x_block


def normalized_block_weight(record, split_records):
    """Return the per-row training weight for one protein block."""
    if not split_records:
        return 0.0
    total_rows = float(sum(item["n_pairs"] for item in split_records))
    if total_rows <= 0.0:
        return 0.0
    return total_rows / (float(len(split_records)) * float(record["n_pairs"]))


def fit_model_from_cache(train_records, cache, scaler_mean, scaler_scale, variant):
    """Fit one standardized-space regression model via streamed normal equations."""
    feature_dim = feature_dim_for_variant(variant)
    system_dim = feature_dim + 1
    xtwx = np.zeros((system_dim, system_dim), dtype=np.float64)
    xtwy = np.zeros((system_dim,), dtype=np.float64)

    for record in train_records:
        raw_chunk = cache[record["protein_id"]]
        x_block = standardized_feature_block(raw_chunk, scaler_mean, scaler_scale)
        y_block = np.asarray(record["y_abs"], dtype=np.float64)
        row_weight = normalized_block_weight(record, train_records)
        weights = np.full((int(record["n_pairs"]),), row_weight, dtype=np.float64)

        weighted_x = x_block * weights[:, None]
        xtwx[0, 0] += float(np.sum(weights, dtype=np.float64))
        intercept_feature = np.sum(weighted_x, axis=0, dtype=np.float64)
        xtwx[0, 1:] += intercept_feature
        xtwx[1:, 0] += intercept_feature
        xtwx[1:, 1:] += x_block.T @ weighted_x
        xtwy[0] += float(np.sum(weights * y_block, dtype=np.float64))
        xtwy[1:] += x_block.T @ (weights * y_block)

    try:
        solution = np.linalg.solve(xtwx, xtwy)
    except np.linalg.LinAlgError:
        solution = np.linalg.lstsq(xtwx, xtwy, rcond=None)[0]

    return {
        "coef_std": np.asarray(solution[1:], dtype=np.float64),
        "intercept_std": float(solution[0]),
    }


def predict_from_fitted(x_block, fitted):
    """Predict from one standardized feature block."""
    return float(fitted["intercept_std"]) + (x_block @ np.asarray(fitted["coef_std"], dtype=np.float64))


def summarize_split_from_cache(records, cache, scaler_mean, scaler_scale, fitted):
    """Compute pooled and per-protein metrics for one split without stacking all rows."""
    per_rows = []
    per_r2_values = []
    negative_r2 = 0
    pair_sum_y = 0.0
    pair_sum_y2 = 0.0
    pair_sum_abs = 0.0
    pair_sum_sqerr = 0.0
    pair_total_n = 0

    for record in records:
        raw_chunk = cache[record["protein_id"]]
        x_block = standardized_feature_block(raw_chunk, scaler_mean, scaler_scale)
        y_true_p = np.asarray(record["y_abs"], dtype=np.float64)
        y_pred_p = np.asarray(predict_from_fitted(x_block, fitted), dtype=np.float64)
        mse_value = float(np.mean((y_pred_p - y_true_p) ** 2))
        r2_value = compute_r2(y_true_p, y_pred_p)
        if not math.isnan(r2_value):
            per_r2_values.append(r2_value)
            if r2_value < 0.0:
                negative_r2 += 1

        per_rows.append(
            {
                "protein_id": record["protein_id"],
                "n_residues": int(record["n_residues"]),
                "size_bin": int(record["size_bin"]),
                "n_pairs": int(record["n_pairs"]),
                "target_variance": float(record["target_variance"]),
                "per_protein_r2": float(r2_value),
                "per_protein_mse": float(mse_value),
                "per_protein_mae": compute_mae(y_true_p, y_pred_p),
                "per_protein_rmse": compute_rmse(y_true_p, y_pred_p),
                "mean_true_distance": float(np.mean(y_true_p)),
                "mean_predicted_distance": float(np.mean(y_pred_p)),
                "mean_error_pred_minus_true": float(np.mean(y_pred_p - y_true_p)),
            }
        )

        err_p = y_pred_p - y_true_p
        pair_sum_y += float(np.sum(y_true_p, dtype=np.float64))
        pair_sum_y2 += float(np.sum(y_true_p * y_true_p, dtype=np.float64))
        pair_sum_abs += float(np.sum(np.abs(err_p), dtype=np.float64))
        pair_sum_sqerr += float(np.sum(err_p * err_p, dtype=np.float64))
        pair_total_n += int(y_true_p.size)

    pair_mean = pair_sum_y / max(float(pair_total_n), 1.0)
    pair_ss_tot = (
        pair_sum_y2
        - 2.0 * pair_mean * pair_sum_y
        + float(pair_total_n) * pair_mean * pair_mean
    )
    if pair_ss_tot <= FLOAT_EPS:
        pair_r2 = float("nan")
    else:
        pair_r2 = float(1.0 - (pair_sum_sqerr / pair_ss_tot))

    pair_mae = pair_sum_abs / max(float(pair_total_n), 1.0)
    pair_rmse = float(math.sqrt(pair_sum_sqerr / max(float(pair_total_n), 1.0)))

    return {
        "pair_weighted_r2": float(pair_r2),
        "pair_weighted_mae": float(pair_mae),
        "pair_weighted_rmse": float(pair_rmse),
        "mean_per_protein_r2": (
            float(np.mean(per_r2_values)) if per_r2_values else float("nan")
        ),
        "median_per_protein_r2": (
            float(np.median(per_r2_values)) if per_r2_values else float("nan")
        ),
        "negative_r2_fraction": (
            float(negative_r2 / len(per_r2_values)) if per_r2_values else float("nan")
        ),
        "mean_bias": (
            float(np.mean([row["mean_error_pred_minus_true"] for row in per_rows]))
            if per_rows
            else float("nan")
        ),
        "per_rows": per_rows,
    }


def standardized_to_raw_coefficients(coef_std, intercept_std, scaler_mean, scaler_scale):
    """Convert standardized-space coefficients back to raw feature space."""
    coef_std = np.asarray(coef_std, dtype=np.float64)
    scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
    coef_raw = coef_std / scaler_scale
    intercept_raw = float(
        intercept_std - np.sum((coef_std * scaler_mean) / scaler_scale, dtype=np.float64)
    )
    return coef_raw, intercept_raw


def build_coefficient_rows(variant, layer, fold_index, fitted, scaler_mean, scaler_scale):
    """Emit intercept plus one row per feature coefficient."""
    coef_std = np.asarray(fitted["coef_std"], dtype=np.float64)
    intercept_std = float(fitted["intercept_std"])
    coef_raw, intercept_raw = standardized_to_raw_coefficients(
        coef_std,
        intercept_std,
        scaler_mean,
        scaler_scale,
    )
    feature_names = feature_names_for_variant(variant)
    feature_dim = int(len(feature_names))

    rows = [
        {
            "variant": variant["name"],
            "feature_dim": feature_dim,
            "fold": int(fold_index),
            "layer": int(layer),
            "feature_index": -1,
            "feature_name": "intercept",
            "coefficient_standardized": float(intercept_std),
            "coefficient_raw_scale": float(intercept_raw),
        }
    ]

    for feature_index, feature_name in enumerate(feature_names):
        rows.append(
            {
                "variant": variant["name"],
                "feature_dim": feature_dim,
                "fold": int(fold_index),
                "layer": int(layer),
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "coefficient_standardized": float(coef_std[feature_index]),
                "coefficient_raw_scale": float(coef_raw[feature_index]),
            }
        )
    return rows


def build_fold_summary_row(
    variant,
    layer,
    fold_index,
    train_records,
    test_records,
    train_metrics,
    test_metrics,
    elapsed_seconds,
):
    """Build one fold/layer summary row."""
    return {
        "variant": variant["name"],
        "feature_dim": int(feature_dim_for_variant(variant)),
        "fold": int(fold_index),
        "layer": int(layer),
        "n_train_proteins": int(len(train_records)),
        "n_test_proteins": int(len(test_records)),
        "n_train_rows": int(sum(record["n_pairs"] for record in train_records)),
        "n_test_rows": int(sum(record["n_pairs"] for record in test_records)),
        "train_mean_per_protein_r2": float(train_metrics["mean_per_protein_r2"]),
        "test_mean_per_protein_r2": float(test_metrics["mean_per_protein_r2"]),
        "train_pair_weighted_r2": float(train_metrics["pair_weighted_r2"]),
        "test_pair_weighted_r2": float(test_metrics["pair_weighted_r2"]),
        "train_pair_weighted_mae": float(train_metrics["pair_weighted_mae"]),
        "test_pair_weighted_mae": float(test_metrics["pair_weighted_mae"]),
        "train_pair_weighted_rmse": float(train_metrics["pair_weighted_rmse"]),
        "test_pair_weighted_rmse": float(test_metrics["pair_weighted_rmse"]),
        "train_mean_bias": float(train_metrics["mean_bias"]),
        "test_mean_bias": float(test_metrics["mean_bias"]),
        "elapsed_seconds": float(elapsed_seconds),
    }


def build_per_protein_output_rows(variant, layer, fold_index, split_name, metric_rows):
    """Attach fold/layer metadata to per-protein metric rows."""
    rows = []
    for row in metric_rows:
        out = dict(row)
        out["variant"] = variant["name"]
        out["feature_dim"] = int(feature_dim_for_variant(variant))
        out["fold"] = int(fold_index)
        out["layer"] = int(layer)
        out["split"] = split_name
        rows.append(out)
    return rows


def make_failed_geometry_row(variant, layer, fold_index, split_name, record, per_r2):
    """Return a placeholder geometry row when reconstruction fails."""
    return {
        "variant": variant["name"],
        "feature_dim": int(feature_dim_for_variant(variant)),
        "fold": int(fold_index),
        "layer": int(layer),
        "split": split_name,
        "protein_id": record["protein_id"],
        "n_residues": int(record["n_residues"]),
        "size_bin": int(record["size_bin"]),
        "n_pairs": int(record["n_pairs"]),
        "per_protein_r2": float(per_r2),
        "rmsd": float("nan"),
        "tm_score": float("nan"),
        "frac_top3_eigenvalues": float("nan"),
        "n_negative_eigenvalues": -1,
        "neg_eigenvalue_fraction": float("nan"),
        "n_negative_predictions": -1,
        "stress": float("nan"),
        "triangle_violation_fraction": float("nan"),
        "triangle_mean_violation": float("nan"),
        "triangle_max_violation": float("nan"),
    }


def collect_geometry_rows_for_split(
    variant,
    layer,
    fold_index,
    split_name,
    split_records,
    cache,
    scaler_mean,
    scaler_scale,
    fitted,
):
    """Run per-protein geometry analysis for one split."""
    rows = []
    for record in split_records:
        raw_chunk = cache[record["protein_id"]]
        x_block = standardized_feature_block(raw_chunk, scaler_mean, scaler_scale)
        y_true = np.asarray(record["y_abs"], dtype=np.float64)
        y_pred = np.asarray(predict_from_fitted(x_block, fitted), dtype=np.float64)
        per_r2 = compute_r2(y_true, y_pred)
        try:
            metrics = analyze_protein_geometry(
                int(record["n_residues"]),
                np.asarray(record["coordinates"], dtype=np.float64),
                record["lt_i"],
                record["lt_j"],
                y_pred,
            )
            rows.append(
                {
                    "variant": variant["name"],
                    "feature_dim": int(feature_dim_for_variant(variant)),
                    "fold": int(fold_index),
                    "layer": int(layer),
                    "split": split_name,
                    "protein_id": record["protein_id"],
                    "n_residues": int(record["n_residues"]),
                    "size_bin": int(record["size_bin"]),
                    "n_pairs": int(record["n_pairs"]),
                    "per_protein_r2": float(per_r2),
                    **metrics,
                }
            )
        except Exception:
            rows.append(
                make_failed_geometry_row(
                    variant,
                    layer,
                    fold_index,
                    split_name,
                    record,
                    per_r2,
                )
            )
    return rows


def summarize_geometry_rows(geometry_rows):
    """Aggregate geometry metrics per variant, layer, and split."""
    grouped = {}
    for row in geometry_rows:
        key = (
            row["variant"],
            int(row["feature_dim"]),
            int(row["fold"]),
            int(row["layer"]),
            row["split"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (variant_name, feature_dim, fold_index, layer, split_name), rows in sorted(grouped.items()):
        out = {
            "variant": variant_name,
            "feature_dim": int(feature_dim),
            "fold": int(fold_index),
            "layer": int(layer),
            "split": split_name,
            "n_proteins": len(rows),
        }
        for metric_key in GEOMETRY_METRIC_KEYS:
            values = [
                float(row[metric_key])
                for row in rows
                if isinstance(row[metric_key], (int, float, np.integer, np.floating))
                and not math.isnan(float(row[metric_key]))
            ]
            out[f"{metric_key}_mean"] = float(np.mean(values)) if values else float("nan")
            out[f"{metric_key}_std"] = float(np.std(values)) if values else float("nan")
        summary_rows.append(out)

    return summary_rows


def build_summary_text(
    path,
    title,
    variant,
    elapsed_seconds,
    all_proteins,
    loaded_records,
    skipped,
    size_bin_counts,
    fold_summary_rows,
    geometry_summary_rows,
):
    """Write a compact human-readable summary."""
    by_layer = {}
    for row in fold_summary_rows:
        by_layer.setdefault(int(row["layer"]), []).append(float(row["test_mean_per_protein_r2"]))

    layer_means = []
    for layer, values in sorted(by_layer.items()):
        clean_values = [value for value in values if not math.isnan(value)]
        layer_means.append(
            (
                layer,
                float(np.mean(clean_values)) if clean_values else float("nan"),
            )
        )

    best_layer = None
    best_key = None
    for layer, mean_r2 in layer_means:
        safe_score = -1e18 if math.isnan(mean_r2) else mean_r2
        key = (safe_score, -layer)
        if best_layer is None or key > best_key:
            best_layer = (layer, mean_r2)
            best_key = key

    geom_by_layer = {}
    for row in geometry_summary_rows:
        if row["split"] != "test":
            continue
        tm_key = float(row["tm_score_mean"])
        geom_by_layer.setdefault(int(row["layer"]), []).append(tm_key)

    best_tm_layer = None
    best_tm_key = None
    for layer, values in sorted(geom_by_layer.items()):
        clean_values = [value for value in values if not math.isnan(value)]
        mean_tm = float(np.mean(clean_values)) if clean_values else float("nan")
        safe_score = -1e18 if math.isnan(mean_tm) else mean_tm
        key = (safe_score, -layer)
        if best_tm_layer is None or key > best_tm_key:
            best_tm_layer = (layer, mean_tm)
            best_tm_key = key

    with path.open("w") as handle:
        handle.write(f"{title}\n")
        handle.write(f"{'=' * len(title)}\n\n")
        handle.write(f"Variant:             {variant['name']}\n")
        handle.write(f"Feature dimension:   {feature_dim_for_variant(variant)}\n")
        handle.write(f"Layers:              {N_LAYERS}\n")
        handle.write(f"Outer folds:         {N_FOLDS} (strict size-stratified)\n")
        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(loaded_records)}\n")
        handle.write(f"Size bin counts:     {size_bin_counts}\n")
        handle.write(f"Elapsed seconds:     {elapsed_seconds:.2f}\n\n")

        if best_layer is not None:
            handle.write(
                "Best layer by mean held-out per-protein R2:\n"
                f"  Layer {best_layer[0]} -> {best_layer[1]:.4f}\n\n"
            )

        if best_tm_layer is not None:
            handle.write(
                "Best last-fold test geometry layer by mean TM-score:\n"
                f"  Layer {best_tm_layer[0]} -> {best_tm_layer[1]:.4f}\n\n"
            )

        if layer_means:
            handle.write("Layer means (held-out per-protein R2):\n")
            for layer, mean_r2 in layer_means:
                handle.write(f"  Layer {layer:02d}: {mean_r2:.4f}\n")

        if skipped:
            handle.write("\nSkipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")


def flush_outputs(
    output_dir,
    title,
    variant,
    all_proteins,
    loaded_records,
    skipped,
    size_bin_counts,
    fold_summary_rows,
    per_protein_rows,
    coefficient_rows,
    fold_assignment_rows,
    geometry_rows,
    elapsed_seconds,
):
    """Persist the current partial or final outputs to disk."""
    geometry_summary_rows = summarize_geometry_rows(geometry_rows)
    write_csv(output_dir / FOLD_SUMMARY_CSV_NAME, FOLD_SUMMARY_FIELDNAMES, fold_summary_rows)
    write_csv(output_dir / PER_PROTEIN_CSV_NAME, PER_PROTEIN_FIELDNAMES, per_protein_rows)
    write_csv(output_dir / COEFFICIENTS_CSV_NAME, COEFFICIENT_FIELDNAMES, coefficient_rows)
    write_csv(output_dir / FOLD_ASSIGNMENTS_CSV_NAME, FOLD_ASSIGNMENT_FIELDNAMES, fold_assignment_rows)
    write_csv(output_dir / GEOMETRY_METRICS_CSV_NAME, GEOMETRY_METRICS_FIELDNAMES, geometry_rows)
    write_csv(output_dir / GEOMETRY_SUMMARY_CSV_NAME, GEOMETRY_SUMMARY_FIELDNAMES, geometry_summary_rows)
    build_summary_text(
        output_dir / SUMMARY_TXT_NAME,
        title=title,
        variant=variant,
        elapsed_seconds=elapsed_seconds,
        all_proteins=all_proteins,
        loaded_records=loaded_records,
        skipped=skipped,
        size_bin_counts=size_bin_counts,
        fold_summary_rows=fold_summary_rows,
        geometry_summary_rows=geometry_summary_rows,
    )


def run_variant_job(variant, title):
    """Run one full 48-layer strict k-fold regression job for a fixed variant."""
    t0 = time.time()
    pair_blocks_root_env = os.environ.get("PAIR_BLOCKS_ROOT")
    pdb_root_env = os.environ.get("PDB_ROOT")
    csv_root_env = os.environ.get("CSV_ROOT")

    if not pair_blocks_root_env or not pdb_root_env or not csv_root_env:
        raise ValueError("PAIR_BLOCKS_ROOT, PDB_ROOT, and CSV_ROOT must all be set.")

    pair_blocks_root = Path(pair_blocks_root_env).resolve()
    pdb_root = Path(pdb_root_env).resolve()
    output_dir = Path(csv_root_env).resolve() / variant["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(title)
    print("=" * len(title))
    print(f"Pair blocks root: {pair_blocks_root}")
    print(f"PDB root:         {pdb_root}")
    print(f"Output dir:       {output_dir}")

    all_proteins, records, skipped = load_protein_records(pair_blocks_root, pdb_root)
    if not records:
        print("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    assign_size_bins(records, N_SIZE_BINS)
    folds, fold_assignment_rows, size_bin_counts = build_strict_stratified_folds(records)

    print(f"Loaded proteins:  {len(records)}")
    print(f"Size bin counts:  {size_bin_counts}")
    for fold in folds:
        print(
            f"  Fold {fold['fold_index']}: "
            f"{len(fold['train_records'])} train / {len(fold['test_records'])} test proteins",
            flush=True,
        )

    fold_summary_rows = []
    per_protein_rows = []
    coefficient_rows = []
    geometry_rows = []

    flush_outputs(
        output_dir=output_dir,
        title=title,
        variant=variant,
        all_proteins=all_proteins,
        loaded_records=records,
        skipped=skipped,
        size_bin_counts=size_bin_counts,
        fold_summary_rows=fold_summary_rows,
        per_protein_rows=per_protein_rows,
        coefficient_rows=coefficient_rows,
        fold_assignment_rows=fold_assignment_rows,
        geometry_rows=geometry_rows,
        elapsed_seconds=float(time.time() - t0),
    )

    for layer in range(N_LAYERS):
        layer_t0 = time.time()
        layer_cache = load_layer_feature_cache(records, layer, variant)
        try:
            for fold in folds:
                fold_t0 = time.time()
                train_records = fold["train_records"]
                test_records = fold["test_records"]
                scaler_mean, scaler_scale = compute_scaler_from_cache(train_records, layer_cache, variant)
                fitted = fit_model_from_cache(train_records, layer_cache, scaler_mean, scaler_scale, variant)
                train_metrics = summarize_split_from_cache(
                    train_records,
                    layer_cache,
                    scaler_mean,
                    scaler_scale,
                    fitted,
                )
                test_metrics = summarize_split_from_cache(
                    test_records,
                    layer_cache,
                    scaler_mean,
                    scaler_scale,
                    fitted,
                )

                fold_summary_rows.append(
                    build_fold_summary_row(
                        variant=variant,
                        layer=layer,
                        fold_index=fold["fold_index"],
                        train_records=train_records,
                        test_records=test_records,
                        train_metrics=train_metrics,
                        test_metrics=test_metrics,
                        elapsed_seconds=float(time.time() - fold_t0),
                    )
                )
                per_protein_rows.extend(
                    build_per_protein_output_rows(
                        variant,
                        layer,
                        fold["fold_index"],
                        "train",
                        train_metrics["per_rows"],
                    )
                )
                per_protein_rows.extend(
                    build_per_protein_output_rows(
                        variant,
                        layer,
                        fold["fold_index"],
                        "test",
                        test_metrics["per_rows"],
                    )
                )
                coefficient_rows.extend(
                    build_coefficient_rows(
                        variant,
                        layer,
                        fold["fold_index"],
                        fitted,
                        scaler_mean,
                        scaler_scale,
                    )
                )

                if fold["fold_index"] == (N_FOLDS - 1):
                    geometry_rows.extend(
                        collect_geometry_rows_for_split(
                            variant,
                            layer,
                            fold["fold_index"],
                            "train",
                            train_records,
                            layer_cache,
                            scaler_mean,
                            scaler_scale,
                            fitted,
                        )
                    )
                    geometry_rows.extend(
                        collect_geometry_rows_for_split(
                            variant,
                            layer,
                            fold["fold_index"],
                            "test",
                            test_records,
                            layer_cache,
                            scaler_mean,
                            scaler_scale,
                            fitted,
                        )
                    )

                print(
                    f"[{now_timestamp()}] Layer {layer:02d} Fold {fold['fold_index']}: "
                    f"test_R2={test_metrics['mean_per_protein_r2']:.4f}, "
                    f"test_pair_R2={test_metrics['pair_weighted_r2']:.4f}, "
                    f"elapsed={time.time() - fold_t0:.1f}s",
                    flush=True,
                )

            flush_outputs(
                output_dir=output_dir,
                title=title,
                variant=variant,
                all_proteins=all_proteins,
                loaded_records=records,
                skipped=skipped,
                size_bin_counts=size_bin_counts,
                fold_summary_rows=fold_summary_rows,
                per_protein_rows=per_protein_rows,
                coefficient_rows=coefficient_rows,
                fold_assignment_rows=fold_assignment_rows,
                geometry_rows=geometry_rows,
                elapsed_seconds=float(time.time() - t0),
            )
            print(
                f"[{now_timestamp()}] Finished layer {layer:02d} "
                f"(elapsed {time.time() - layer_t0:.1f}s)",
                flush=True,
            )
        finally:
            layer_cache.clear()
            del layer_cache
            gc.collect()

    flush_outputs(
        output_dir=output_dir,
        title=title,
        variant=variant,
        all_proteins=all_proteins,
        loaded_records=records,
        skipped=skipped,
        size_bin_counts=size_bin_counts,
        fold_summary_rows=fold_summary_rows,
        per_protein_rows=per_protein_rows,
        coefficient_rows=coefficient_rows,
        fold_assignment_rows=fold_assignment_rows,
        geometry_rows=geometry_rows,
        elapsed_seconds=float(time.time() - t0),
    )
    print(
        f"[{now_timestamp()}] Completed {variant['name']} in {time.time() - t0:.1f}s",
        flush=True,
    )

#!/usr/bin/env python3
"""
diagnostic_ridge_stability_regression.py
=======================================
Layer-47 diagnostic for ridge-stability investigations using protein-equal
weighted linear probes on lower-triangle residue pairs.

The script focuses on four default outer-split probes:
1. 128d_linear_no_ridge
2. 128d_ridge_1e-8
3. 256d_linear_no_ridge
4. 256d_ridge_1e-8

It also runs:
- intercept ablations (fit_intercept / no_intercept / centered_target)
- split-family ablations for ridge(alpha=1e-8)
- ridge alpha selection by inner CV
- lasso alpha selection by inner CV

Outputs (inside CSV_ROOT/):
    diagnostic_ridge_stability_variant_summary.csv
    diagnostic_ridge_stability_per_protein.csv
    diagnostic_ridge_stability_split.csv
    diagnostic_ridge_stability_coefficients.csv
    diagnostic_ridge_stability_design_diagnostics.csv
    diagnostic_ridge_stability_symmetry_diagnostics.csv
    diagnostic_ridge_stability_comparison.csv
    diagnostic_ridge_stability_alpha_cv.csv
    diagnostic_ridge_stability_alpha_selection.csv
    diagnostic_ridge_stability_summary.txt
"""

import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import Lasso, LinearRegression, Ridge
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CLUSTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_SCRIPTS_DIR))

from diagnostic_tree_regression_common import (  # noqa: E402
    CSV_DIR,
    FLOAT_EPS,
    LAYER,
    N_CHANNELS,
    RANDOM_STATE,
    TEST_SIZE,
    load_protein_records,
    write_csv,
)


TITLE = "Diagnostic Ridge Stability Regression - Layer 47"

FEATURE_SET_CENTER = "128d"
FEATURE_SET_PAIRED = "256d"
FEATURE_DIM_LOOKUP = {
    FEATURE_SET_CENTER: N_CHANNELS,
    FEATURE_SET_PAIRED: 2 * N_CHANNELS,
}

SPLIT_SCHEME_STRATIFIED = "size_stratified_scope_split"
SPLIT_SCHEME_RANDOM = "random_protein_split"

ESTIMATOR_LINEAR = "linear"
ESTIMATOR_RIDGE = "ridge"
ESTIMATOR_LASSO = "lasso"

INTERCEPT_FIT = "fit_intercept"
INTERCEPT_NONE = "no_intercept"
INTERCEPT_CENTERED = "centered_target"

N_SIZE_BINS = 5
RIDGE_ALPHA_TINY = 1e-8
INNER_CV_FOLDS = 5
LASSO_MAX_ITER = 20_000

RIDGE_ALPHA_GRID = [
    1e-12,
    1e-10,
    1e-8,
    1e-6,
    1e-4,
    1e-3,
    1e-2,
    1e-1,
    1.0,
    10.0,
    100.0,
]
LASSO_ALPHA_GRID = [
    1e-6,
    3e-6,
    1e-5,
    3e-5,
    1e-4,
    3e-4,
    1e-3,
    3e-3,
    1e-2,
    3e-2,
]

SUMMARY_CSV_NAME = "diagnostic_ridge_stability_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_ridge_stability_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_ridge_stability_split.csv"
COEFFICIENT_CSV_NAME = "diagnostic_ridge_stability_coefficients.csv"
DESIGN_DIAGNOSTICS_CSV_NAME = "diagnostic_ridge_stability_design_diagnostics.csv"
SYMMETRY_DIAGNOSTICS_CSV_NAME = "diagnostic_ridge_stability_symmetry_diagnostics.csv"
COMPARISON_CSV_NAME = "diagnostic_ridge_stability_comparison.csv"
ALPHA_CV_CSV_NAME = "diagnostic_ridge_stability_alpha_cv.csv"
ALPHA_SELECTION_CSV_NAME = "diagnostic_ridge_stability_alpha_selection.csv"
SUMMARY_TXT_NAME = "diagnostic_ridge_stability_summary.txt"

CENTER_FEATURE_NAMES = [f"pair_ch_{idx}" for idx in range(N_CHANNELS)]
PAIRED_FEATURE_NAMES = [f"pair_ij_ch_{idx}" for idx in range(N_CHANNELS)] + [
    f"pair_ji_ch_{idx}" for idx in range(N_CHANNELS)
]

SUMMARY_FIELDNAMES = [
    "run_name",
    "analysis_family",
    "feature_set",
    "feature_dim",
    "estimator",
    "alpha",
    "intercept_mode",
    "split_scheme",
    "selected_by_cv",
    "inner_cv_metric",
    "n_train_proteins",
    "n_test_proteins",
    "n_train_rows",
    "n_test_rows",
    "train_pair_weighted_r2",
    "test_pair_weighted_r2",
    "train_protein_weighted_r2",
    "test_protein_weighted_r2",
    "train_pair_weighted_mae",
    "test_pair_weighted_mae",
    "train_pair_weighted_rmse",
    "test_pair_weighted_rmse",
    "train_mean_per_protein_r2",
    "test_mean_per_protein_r2",
    "test_median_per_protein_r2",
    "test_negative_r2_fraction",
    "train_mean_per_protein_nmse",
    "test_mean_per_protein_nmse",
    "test_median_per_protein_nmse",
    "test_max_per_protein_nmse",
    "train_mean_bias",
    "test_mean_bias",
    "intercept_standardized",
    "intercept_raw",
    "coef_l2_standardized",
    "coef_l2_raw",
    "n_nonzero_coefficients",
    "elapsed_seconds",
]

PER_PROTEIN_FIELDNAMES = [
    "run_name",
    "analysis_family",
    "feature_set",
    "estimator",
    "alpha",
    "intercept_mode",
    "split_scheme",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
    "per_protein_r2",
    "per_protein_mse",
    "per_protein_nmse",
    "per_protein_mae",
    "per_protein_rmse",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
]

SPLIT_FIELDNAMES = [
    "split_scheme",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
]

COEFFICIENT_FIELDNAMES = [
    "run_name",
    "analysis_family",
    "feature_set",
    "estimator",
    "alpha",
    "intercept_mode",
    "split_scheme",
    "feature_index",
    "feature_name",
    "coefficient_standardized",
    "coefficient_raw_scale",
]

DESIGN_DIAGNOSTICS_FIELDNAMES = [
    "split_scheme",
    "feature_set",
    "feature_dim",
    "n_train_proteins",
    "n_train_rows",
    "weight_sum",
    "weighted_column_mean_abs_max",
    "weighted_column_mean_abs_mean",
    "max_singular_value",
    "min_positive_singular_value",
    "condition_number",
    "effective_rank",
    "trace_gram",
    "fro_norm_gram",
    "near_null_rel_1e-6",
    "near_null_rel_1e-8",
    "near_null_rel_1e-10",
]

SYMMETRY_DIAGNOSTICS_FIELDNAMES = [
    "row_type",
    "split_scheme",
    "run_name",
    "feature_set",
    "estimator",
    "alpha",
    "intercept_mode",
    "channel_index",
    "pair_corr",
    "pair_cov",
    "same_sign",
    "sym_coef_standardized",
    "anti_coef_standardized",
    "sym_coef_raw",
    "anti_coef_raw",
    "sym_norm_standardized",
    "anti_norm_standardized",
    "sym_fraction_standardized",
    "sym_norm_raw",
    "anti_norm_raw",
    "sym_fraction_raw",
    "same_sign_count",
    "total_channels",
]

COMPARISON_FIELDNAMES = [
    "comparison_name",
    "feature_set",
    "split_scheme",
    "run_name_a",
    "run_name_b",
    "intercept_mode",
    "cosine_similarity_standardized",
    "cosine_similarity_raw",
    "coef_l2_standardized_a",
    "coef_l2_standardized_b",
    "coef_l2_raw_a",
    "coef_l2_raw_b",
    "coefficient_diff_l2_standardized",
    "coefficient_diff_l2_raw",
    "intercept_standardized_a",
    "intercept_standardized_b",
    "intercept_raw_a",
    "intercept_raw_b",
    "intercept_raw_diff_b_minus_a",
    "mean_abs_test_bias_delta",
    "max_abs_test_bias_delta",
]

ALPHA_CV_FIELDNAMES = [
    "row_type",
    "model_family",
    "feature_set",
    "alpha",
    "fold_family",
    "fold_index",
    "n_train_proteins",
    "n_val_proteins",
    "val_mean_per_protein_r2",
    "val_pair_weighted_r2",
    "val_pair_weighted_mae",
    "val_pair_weighted_rmse",
    "val_mean_bias",
    "n_nonzero_coefficients",
    "elapsed_seconds",
]

ALPHA_SELECTION_FIELDNAMES = [
    "model_family",
    "feature_set",
    "split_scheme",
    "selected_metric",
    "chosen_alpha",
    "inner_cv_fold_family",
    "inner_cv_mean_per_protein_r2",
    "inner_cv_std_mean_per_protein_r2",
    "inner_cv_mean_pair_weighted_r2",
    "inner_cv_mean_pair_weighted_mae",
    "outer_test_mean_per_protein_r2",
    "outer_test_pair_weighted_r2",
    "outer_test_pair_weighted_mae",
    "outer_test_pair_weighted_rmse",
    "outer_test_mean_bias",
    "outer_test_n_nonzero_coefficients",
]


def feature_names_for_set(feature_set):
    if feature_set == FEATURE_SET_CENTER:
        return CENTER_FEATURE_NAMES
    if feature_set == FEATURE_SET_PAIRED:
        return PAIRED_FEATURE_NAMES
    raise ValueError(f"Unsupported feature_set: {feature_set}")


def assign_size_bins(records, n_bins):
    """Assign stable rank-based size bins by residue count."""
    ordered = sorted(records, key=lambda row: (row["n_residues"], row["protein_id"]))
    total = len(ordered)
    for rank, record in enumerate(ordered):
        size_bin = int((rank * n_bins) // max(total, 1))
        if size_bin >= n_bins:
            size_bin = n_bins - 1
        record["size_bin"] = int(size_bin)


def choose_stratify_labels(records):
    """Only stratify when the split can actually support it."""
    if len(records) < 2:
        return None

    labels = np.asarray([record["size_bin"] for record in records], dtype=np.int32)
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        return None
    if np.min(counts) < 2:
        return None

    n_total = len(records)
    n_test = max(1, int(math.ceil(TEST_SIZE * n_total)))
    n_train = n_total - n_test
    if n_test < unique.size or n_train < unique.size:
        return None
    return labels


def split_records(records, stratified):
    """Create the outer protein split, optionally using size-bin stratification."""
    if len(records) < 2:
        raise ValueError("Need at least two proteins for train/test splitting.")

    indices = np.arange(len(records), dtype=np.int64)
    stratify_labels = choose_stratify_labels(records) if stratified else None
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )
    train_set = set(int(idx) for idx in train_idx)
    test_set = set(int(idx) for idx in test_idx)
    train_records = [record for idx, record in enumerate(records) if idx in train_set]
    test_records = [record for idx, record in enumerate(records) if idx in test_set]
    return train_records, test_records


def build_scope_spec(scope_name, all_records, stratified):
    """Build one reusable protein split scope."""
    cloned = [dict(record) for record in all_records]
    assign_size_bins(cloned, N_SIZE_BINS)
    train_records, test_records = split_records(cloned, stratified=stratified)
    return {
        "name": scope_name,
        "all_proteins": cloned,
        "train_proteins": train_records,
        "test_proteins": test_records,
    }


def choose_inner_splitter(records):
    """Choose stratified 5-fold CV when feasible, otherwise plain KFold."""
    labels = np.asarray([record["size_bin"] for record in records], dtype=np.int32)
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size >= 2 and np.min(counts) >= INNER_CV_FOLDS:
        splitter = StratifiedKFold(
            n_splits=INNER_CV_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        return "stratified_size_bins", splitter.split(np.arange(len(records)), labels)

    splitter = KFold(n_splits=INNER_CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    return "random_kfold", splitter.split(np.arange(len(records)))


def build_inner_folds(records):
    """Materialize inner CV folds as train/validation record groups."""
    if len(records) < INNER_CV_FOLDS:
        raise ValueError(
            f"Need at least {INNER_CV_FOLDS} training proteins for inner CV; "
            f"got {len(records)}"
        )

    family_name, split_iter = choose_inner_splitter(records)
    folds = []
    for fold_index, (train_idx, val_idx) in enumerate(split_iter):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        val_idx = np.asarray(val_idx, dtype=np.int64)
        folds.append(
            {
                "fold_index": int(fold_index),
                "train_records": [records[int(idx)] for idx in train_idx],
                "val_records": [records[int(idx)] for idx in val_idx],
            }
        )
    return family_name, folds


def prepare_records(records):
    """Attach target variance and mean metadata used by the summaries."""
    prepared = []
    for record in records:
        target_variance = float(np.var(record["y_abs"], dtype=np.float64))
        if target_variance <= FLOAT_EPS:
            target_variance = FLOAT_EPS

        prepared_record = dict(record)
        prepared_record["target_variance"] = target_variance
        prepared_record["target_mean"] = float(np.mean(record["y_abs"], dtype=np.float64))
        prepared.append(prepared_record)
    return prepared


def build_feature_caches(records):
    """Cache 128d and 256d layer-47 lower-triangle feature matrices."""
    center_chunk_map = {}
    paired_chunk_map = {}

    for record in records:
        pb_path = (
            record["protein_dir"] / f"{record['protein_id']}_pair_block_{LAYER}.npy"
        )
        pb = np.load(str(pb_path), mmap_mode="r")
        center_chunk = np.asarray(pb[record["lt_i"], record["lt_j"], :], dtype=np.float32)
        paired_chunk = np.concatenate(
            [
                center_chunk,
                np.asarray(pb[record["lt_j"], record["lt_i"], :], dtype=np.float32),
            ],
            axis=1,
        ).astype(np.float32, copy=False)
        del pb

        center_chunk_map[record["protein_id"]] = center_chunk
        paired_chunk_map[record["protein_id"]] = paired_chunk

        record["center_feature_mean"] = np.asarray(
            np.mean(center_chunk, axis=0, dtype=np.float64),
            dtype=np.float64,
        )
        record["center_feature_sq_mean"] = np.asarray(
            np.mean(center_chunk * center_chunk, axis=0, dtype=np.float64),
            dtype=np.float64,
        )
        record["paired_feature_mean"] = np.asarray(
            np.mean(paired_chunk, axis=0, dtype=np.float64),
            dtype=np.float64,
        )
        record["paired_feature_sq_mean"] = np.asarray(
            np.mean(paired_chunk * paired_chunk, axis=0, dtype=np.float64),
            dtype=np.float64,
        )

    return {
        FEATURE_SET_CENTER: center_chunk_map,
        FEATURE_SET_PAIRED: paired_chunk_map,
    }


def compute_scaler_from_protein_stats(train_records, feature_set):
    """Compute train-only protein-equal mean/std for the requested feature set."""
    dim = FEATURE_DIM_LOOKUP[feature_set]
    mean_acc = np.zeros(dim, dtype=np.float64)
    sq_acc = np.zeros(dim, dtype=np.float64)

    if feature_set == FEATURE_SET_CENTER:
        mean_key = "center_feature_mean"
        sq_key = "center_feature_sq_mean"
    elif feature_set == FEATURE_SET_PAIRED:
        mean_key = "paired_feature_mean"
        sq_key = "paired_feature_sq_mean"
    else:
        raise ValueError(f"Unsupported feature_set: {feature_set}")

    for record in train_records:
        mean_acc += np.asarray(record[mean_key], dtype=np.float64)
        sq_acc += np.asarray(record[sq_key], dtype=np.float64)

    mean = mean_acc / max(len(train_records), 1)
    second_moment = sq_acc / max(len(train_records), 1)
    var = np.maximum(second_moment - mean * mean, 0.0)
    scale = np.sqrt(var)
    scale[scale < FLOAT_EPS] = 1.0
    return mean, scale


def compute_split_pair_weights(records, normalize_mean=True):
    """Per-pair weights with equal total mass per protein."""
    pieces = [
        np.full(record["n_pairs"], 1.0 / record["n_pairs"], dtype=np.float64)
        for record in records
    ]
    if not pieces:
        return np.empty((0,), dtype=np.float64)
    weights = np.concatenate(pieces)
    if normalize_mean and weights.size > 0:
        weights *= weights.size / np.sum(weights)
    return weights


def materialize_split(records, chunk_map):
    """Stack one split into dense feature and target arrays."""
    x = np.vstack([chunk_map[record["protein_id"]] for record in records]).astype(
        np.float64,
        copy=False,
    )
    y = np.concatenate([record["y_abs"] for record in records]).astype(np.float64, copy=False)
    return x, y


def build_materialized_scope(scope, feature_set, chunk_maps):
    """Create standardized train/test matrices for one split scheme and feature set."""
    chunk_map = chunk_maps[feature_set]
    scaler_mean, scaler_scale = compute_scaler_from_protein_stats(
        scope["train_proteins"], feature_set
    )
    x_train, y_train = materialize_split(scope["train_proteins"], chunk_map)
    x_test, y_test = materialize_split(scope["test_proteins"], chunk_map)

    x_train = np.asarray(x_train, dtype=np.float64)
    x_test = np.asarray(x_test, dtype=np.float64)
    x_train -= scaler_mean
    x_train /= scaler_scale
    x_test -= scaler_mean
    x_test /= scaler_scale

    return {
        "x_train": x_train,
        "x_test": x_test,
        "y_train": y_train,
        "y_test": y_test,
        "train_weights": compute_split_pair_weights(scope["train_proteins"], normalize_mean=True),
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "feature_names": feature_names_for_set(feature_set),
    }


def weighted_mean(values, weights):
    values = np.asarray(values, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan")
    return float(np.sum(weights * values) / denom)


def weighted_column_mean(matrix, weights):
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return np.zeros((matrix.shape[1],), dtype=np.float64)
    return np.sum(matrix * weights[:, None], axis=0, dtype=np.float64) / denom


def weighted_corr(x, y, weights):
    """Weighted Pearson correlation."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return float("nan"), float("nan")

    mean_x = np.sum(weights * x) / denom
    mean_y = np.sum(weights * y) / denom
    x_centered = x - mean_x
    y_centered = y - mean_y
    cov = float(np.sum(weights * x_centered * y_centered) / denom)
    var_x = float(np.sum(weights * x_centered * x_centered) / denom)
    var_y = float(np.sum(weights * y_centered * y_centered) / denom)
    if var_x <= FLOAT_EPS or var_y <= FLOAT_EPS:
        return float("nan"), cov
    corr = cov / math.sqrt(var_x * var_y)
    return float(corr), cov


def compute_rmse_from_sqerr(sum_sqerr, total_n):
    return float(math.sqrt(sum_sqerr / max(float(total_n), 1.0)))


def summarize_dense_split(records, y_true, y_pred):
    """Compute pair-weighted and protein-equal metrics for one dense split."""
    per_rows = []
    per_r2_values = []
    per_nmse_values = []
    negative_r2 = 0

    pair_sum_y = 0.0
    pair_sum_y2 = 0.0
    pair_sum_abs = 0.0
    pair_sum_sqerr = 0.0
    pair_total_n = 0

    protein_weight_sum = 0.0
    protein_weighted_sum_y = 0.0
    protein_weighted_sum_y2 = 0.0
    protein_weighted_sum_abs = 0.0
    protein_weighted_sum_sqerr = 0.0

    start = 0
    for record in records:
        end = start + int(record["n_pairs"])
        y_true_p = np.asarray(y_true[start:end], dtype=np.float64)
        y_pred_p = np.asarray(y_pred[start:end], dtype=np.float64)
        start = end

        if y_true_p.size == 0:
            continue

        err_p = y_pred_p - y_true_p
        sum_y = float(np.sum(y_true_p, dtype=np.float64))
        sum_y2 = float(np.sum(y_true_p * y_true_p, dtype=np.float64))
        sum_pred = float(np.sum(y_pred_p, dtype=np.float64))
        sum_abs = float(np.sum(np.abs(err_p), dtype=np.float64))
        sum_sqerr = float(np.sum(err_p * err_p, dtype=np.float64))
        n_rows = int(y_true_p.size)

        mean_y = sum_y / float(n_rows)
        mean_pred = sum_pred / float(n_rows)
        mse_value = sum_sqerr / float(n_rows)
        mae_value = sum_abs / float(n_rows)
        rmse_value = float(math.sqrt(mse_value))
        ss_tot = sum_y2 - 2.0 * mean_y * sum_y + float(n_rows) * mean_y * mean_y
        if ss_tot <= FLOAT_EPS:
            r2_value = float("nan")
        else:
            r2_value = float(1.0 - (sum_sqerr / ss_tot))

        nmse_value = float(mse_value / max(float(record["target_variance"]), FLOAT_EPS))
        if not math.isnan(r2_value):
            per_r2_values.append(float(r2_value))
            if r2_value < 0.0:
                negative_r2 += 1
        per_nmse_values.append(nmse_value)

        per_rows.append(
            {
                "protein_id": record["protein_id"],
                "n_residues": int(record["n_residues"]),
                "size_bin": int(record["size_bin"]),
                "n_pairs": int(record["n_pairs"]),
                "target_variance": float(record["target_variance"]),
                "per_protein_r2": float(r2_value),
                "per_protein_mse": float(mse_value),
                "per_protein_nmse": float(nmse_value),
                "per_protein_mae": float(mae_value),
                "per_protein_rmse": float(rmse_value),
                "mean_true_distance": float(mean_y),
                "mean_predicted_distance": float(mean_pred),
                "mean_error_pred_minus_true": float(mean_pred - mean_y),
            }
        )

        pair_sum_y += sum_y
        pair_sum_y2 += sum_y2
        pair_sum_abs += sum_abs
        pair_sum_sqerr += sum_sqerr
        pair_total_n += n_rows

        row_weight = 1.0 / float(n_rows)
        protein_weight_sum += 1.0
        protein_weighted_sum_y += row_weight * sum_y
        protein_weighted_sum_y2 += row_weight * sum_y2
        protein_weighted_sum_abs += row_weight * sum_abs
        protein_weighted_sum_sqerr += row_weight * sum_sqerr

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
    pair_rmse = compute_rmse_from_sqerr(pair_sum_sqerr, pair_total_n)

    protein_mean_y = protein_weighted_sum_y / max(protein_weight_sum, 1.0)
    protein_ss_tot = (
        protein_weighted_sum_y2
        - 2.0 * protein_mean_y * protein_weighted_sum_y
        + protein_weight_sum * protein_mean_y * protein_mean_y
    )
    if protein_ss_tot <= FLOAT_EPS:
        protein_weighted_r2 = float("nan")
    else:
        protein_weighted_r2 = float(1.0 - (protein_weighted_sum_sqerr / protein_ss_tot))

    return {
        "pair_weighted_r2": pair_r2,
        "protein_weighted_r2": protein_weighted_r2,
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
        "mean_per_protein_nmse": (
            float(np.mean(per_nmse_values)) if per_nmse_values else float("nan")
        ),
        "median_per_protein_nmse": (
            float(np.median(per_nmse_values)) if per_nmse_values else float("nan")
        ),
        "max_per_protein_nmse": (
            float(np.max(per_nmse_values)) if per_nmse_values else float("nan")
        ),
        "mean_bias": (
            float(np.mean([row["mean_error_pred_minus_true"] for row in per_rows]))
            if per_rows
            else float("nan")
        ),
        "per_rows": per_rows,
    }


def build_split_rows(split_collections):
    rows = []
    for split_scheme, train_records, test_records in split_collections:
        for split_name, split_records in [("train", train_records), ("test", test_records)]:
            for record in split_records:
                rows.append(
                    {
                        "split_scheme": split_scheme,
                        "split": split_name,
                        "protein_id": record["protein_id"],
                        "n_residues": int(record["n_residues"]),
                        "size_bin": int(record["size_bin"]),
                        "n_pairs": int(record["n_pairs"]),
                        "target_variance": float(record["target_variance"]),
                    }
                )
    return rows


def build_estimator(estimator_name, alpha, fit_intercept):
    if estimator_name == ESTIMATOR_LINEAR:
        return LinearRegression(fit_intercept=fit_intercept)
    if estimator_name == ESTIMATOR_RIDGE:
        return Ridge(alpha=float(alpha), fit_intercept=fit_intercept)
    if estimator_name == ESTIMATOR_LASSO:
        return Lasso(alpha=float(alpha), fit_intercept=fit_intercept, max_iter=LASSO_MAX_ITER)
    raise ValueError(f"Unsupported estimator: {estimator_name}")


def fit_model(estimator_name, alpha, intercept_mode, x_train, y_train, sample_weight):
    """Fit one model in standardized feature space."""
    fit_intercept = intercept_mode == INTERCEPT_FIT
    y_fit = np.asarray(y_train, dtype=np.float64)
    prediction_offset = 0.0
    if intercept_mode == INTERCEPT_CENTERED:
        prediction_offset = weighted_mean(y_fit, sample_weight)
        y_fit = y_fit - prediction_offset
        fit_intercept = False
    elif intercept_mode == INTERCEPT_NONE:
        fit_intercept = False

    model = build_estimator(estimator_name, alpha, fit_intercept=fit_intercept)
    model.fit(x_train, y_fit, sample_weight=sample_weight)
    coef_std = np.asarray(model.coef_, dtype=np.float64)
    intercept_std = float(getattr(model, "intercept_", 0.0)) + float(prediction_offset)
    return {
        "model": model,
        "coef_std": coef_std,
        "intercept_std": intercept_std,
        "prediction_offset": float(prediction_offset),
    }


def predict_fitted(fitted, x):
    pred = np.asarray(fitted["model"].predict(x), dtype=np.float64)
    if abs(float(fitted["prediction_offset"])) > 0.0:
        pred = pred + float(fitted["prediction_offset"])
    return pred


def standardized_to_raw_coefficients(coef_std, intercept_std, scaler_mean, scaler_scale):
    coef_raw = np.asarray(coef_std, dtype=np.float64) / np.asarray(scaler_scale, dtype=np.float64)
    intercept_raw = float(
        intercept_std
        - np.sum((np.asarray(scaler_mean, dtype=np.float64) / scaler_scale) * coef_std)
    )
    return coef_raw, intercept_raw


def build_coefficient_rows(run_meta, fitted, scaler_mean, scaler_scale, feature_names):
    coef_std = np.asarray(fitted["coef_std"], dtype=np.float64)
    intercept_std = float(fitted["intercept_std"])
    coef_raw, intercept_raw = standardized_to_raw_coefficients(
        coef_std, intercept_std, scaler_mean, scaler_scale
    )

    rows = [
        {
            "run_name": run_meta["run_name"],
            "analysis_family": run_meta["analysis_family"],
            "feature_set": run_meta["feature_set"],
            "estimator": run_meta["estimator"],
            "alpha": float(run_meta["alpha"]),
            "intercept_mode": run_meta["intercept_mode"],
            "split_scheme": run_meta["split_scheme"],
            "feature_index": -1,
            "feature_name": "intercept",
            "coefficient_standardized": intercept_std,
            "coefficient_raw_scale": intercept_raw,
        }
    ]

    for feature_index, feature_name in enumerate(feature_names):
        rows.append(
            {
                "run_name": run_meta["run_name"],
                "analysis_family": run_meta["analysis_family"],
                "feature_set": run_meta["feature_set"],
                "estimator": run_meta["estimator"],
                "alpha": float(run_meta["alpha"]),
                "intercept_mode": run_meta["intercept_mode"],
                "split_scheme": run_meta["split_scheme"],
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "coefficient_standardized": float(coef_std[feature_index]),
                "coefficient_raw_scale": float(coef_raw[feature_index]),
            }
        )
    return rows


def coefficient_bundle(fitted, scaler_mean, scaler_scale):
    coef_std = np.asarray(fitted["coef_std"], dtype=np.float64)
    intercept_std = float(fitted["intercept_std"])
    coef_raw, intercept_raw = standardized_to_raw_coefficients(
        coef_std, intercept_std, scaler_mean, scaler_scale
    )
    return {
        "coef_std": coef_std,
        "intercept_std": intercept_std,
        "coef_raw": coef_raw,
        "intercept_raw": intercept_raw,
    }


def compute_effective_rank(eigenvalues):
    positive = np.asarray(eigenvalues, dtype=np.float64)
    positive = positive[positive > FLOAT_EPS]
    if positive.size == 0:
        return 0.0
    probs = positive / np.sum(positive)
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def compute_design_diagnostics(scope, feature_set, materialized):
    """Spectrum diagnostics for the weighted standardized outer-train design."""
    x_train = np.asarray(materialized["x_train"], dtype=np.float64)
    weights = np.asarray(materialized["train_weights"], dtype=np.float64)
    weighted_col_mean = weighted_column_mean(x_train, weights)
    x_centered = x_train - weighted_col_mean
    x_weighted = x_centered * np.sqrt(weights)[:, None]
    gram = x_weighted.T @ x_weighted
    eigenvalues = np.linalg.eigvalsh(gram)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    singular_values = np.sqrt(eigenvalues)
    singular_values.sort()
    singular_values = singular_values[::-1]

    max_sv = float(singular_values[0]) if singular_values.size else 0.0
    positive_sv = singular_values[singular_values > FLOAT_EPS]
    min_positive_sv = float(positive_sv[-1]) if positive_sv.size else 0.0
    if positive_sv.size and min_positive_sv > 0.0:
        condition_number = float(max_sv / min_positive_sv)
    else:
        condition_number = float("inf")

    def near_null_count(relative_tol):
        if max_sv <= 0.0:
            return int(len(singular_values))
        return int(np.sum(singular_values <= (relative_tol * max_sv)))

    return {
        "split_scheme": scope["name"],
        "feature_set": feature_set,
        "feature_dim": int(FEATURE_DIM_LOOKUP[feature_set]),
        "n_train_proteins": int(len(scope["train_proteins"])),
        "n_train_rows": int(x_train.shape[0]),
        "weight_sum": float(np.sum(weights)),
        "weighted_column_mean_abs_max": float(np.max(np.abs(weighted_col_mean))),
        "weighted_column_mean_abs_mean": float(np.mean(np.abs(weighted_col_mean))),
        "max_singular_value": max_sv,
        "min_positive_singular_value": min_positive_sv,
        "condition_number": condition_number,
        "effective_rank": compute_effective_rank(eigenvalues),
        "trace_gram": float(np.trace(gram)),
        "fro_norm_gram": float(np.linalg.norm(gram)),
        "near_null_rel_1e-6": near_null_count(1e-6),
        "near_null_rel_1e-8": near_null_count(1e-8),
        "near_null_rel_1e-10": near_null_count(1e-10),
    }


def build_symmetry_design_rows(split_scheme, materialized):
    """Per-channel weighted correlations between paired ij/ji standardized columns."""
    x_train = np.asarray(materialized["x_train"], dtype=np.float64)
    weights = np.asarray(materialized["train_weights"], dtype=np.float64)
    rows = []
    for channel_index in range(N_CHANNELS):
        corr, cov = weighted_corr(
            x_train[:, channel_index],
            x_train[:, N_CHANNELS + channel_index],
            weights,
        )
        rows.append(
            {
                "row_type": "channel_corr",
                "split_scheme": split_scheme,
                "run_name": "",
                "feature_set": FEATURE_SET_PAIRED,
                "estimator": "",
                "alpha": float("nan"),
                "intercept_mode": "",
                "channel_index": int(channel_index),
                "pair_corr": corr,
                "pair_cov": cov,
                "same_sign": "",
                "sym_coef_standardized": float("nan"),
                "anti_coef_standardized": float("nan"),
                "sym_coef_raw": float("nan"),
                "anti_coef_raw": float("nan"),
                "sym_norm_standardized": float("nan"),
                "anti_norm_standardized": float("nan"),
                "sym_fraction_standardized": float("nan"),
                "sym_norm_raw": float("nan"),
                "anti_norm_raw": float("nan"),
                "sym_fraction_raw": float("nan"),
                "same_sign_count": "",
                "total_channels": int(N_CHANNELS),
            }
        )
    return rows


def build_symmetry_coefficient_rows(run_meta, coef_std, coef_raw):
    """Symmetric / antisymmetric decomposition rows for one 256d fit."""
    if coef_std.size != 2 * N_CHANNELS:
        return []

    coef_ij_std = coef_std[:N_CHANNELS]
    coef_ji_std = coef_std[N_CHANNELS:]
    coef_ij_raw = coef_raw[:N_CHANNELS]
    coef_ji_raw = coef_raw[N_CHANNELS:]

    sym_std = 0.5 * (coef_ij_std + coef_ji_std)
    anti_std = 0.5 * (coef_ij_std - coef_ji_std)
    sym_raw = 0.5 * (coef_ij_raw + coef_ji_raw)
    anti_raw = 0.5 * (coef_ij_raw - coef_ji_raw)

    sym_norm_std = float(np.linalg.norm(sym_std))
    anti_norm_std = float(np.linalg.norm(anti_std))
    denom_std = max(sym_norm_std ** 2 + anti_norm_std ** 2, FLOAT_EPS)
    sym_fraction_std = float((sym_norm_std ** 2) / denom_std)

    sym_norm_raw = float(np.linalg.norm(sym_raw))
    anti_norm_raw = float(np.linalg.norm(anti_raw))
    denom_raw = max(sym_norm_raw ** 2 + anti_norm_raw ** 2, FLOAT_EPS)
    sym_fraction_raw = float((sym_norm_raw ** 2) / denom_raw)

    same_sign_count = int(
        np.sum(np.sign(coef_ij_std) == np.sign(coef_ji_std))
    )

    rows = [
        {
            "row_type": "coefficient_summary",
            "split_scheme": run_meta["split_scheme"],
            "run_name": run_meta["run_name"],
            "feature_set": run_meta["feature_set"],
            "estimator": run_meta["estimator"],
            "alpha": float(run_meta["alpha"]),
            "intercept_mode": run_meta["intercept_mode"],
            "channel_index": -1,
            "pair_corr": float("nan"),
            "pair_cov": float("nan"),
            "same_sign": "",
            "sym_coef_standardized": float("nan"),
            "anti_coef_standardized": float("nan"),
            "sym_coef_raw": float("nan"),
            "anti_coef_raw": float("nan"),
            "sym_norm_standardized": sym_norm_std,
            "anti_norm_standardized": anti_norm_std,
            "sym_fraction_standardized": sym_fraction_std,
            "sym_norm_raw": sym_norm_raw,
            "anti_norm_raw": anti_norm_raw,
            "sym_fraction_raw": sym_fraction_raw,
            "same_sign_count": int(same_sign_count),
            "total_channels": int(N_CHANNELS),
        }
    ]

    for channel_index in range(N_CHANNELS):
        rows.append(
            {
                "row_type": "coefficient_channel",
                "split_scheme": run_meta["split_scheme"],
                "run_name": run_meta["run_name"],
                "feature_set": run_meta["feature_set"],
                "estimator": run_meta["estimator"],
                "alpha": float(run_meta["alpha"]),
                "intercept_mode": run_meta["intercept_mode"],
                "channel_index": int(channel_index),
                "pair_corr": float("nan"),
                "pair_cov": float("nan"),
                "same_sign": bool(
                    np.sign(coef_ij_std[channel_index]) == np.sign(coef_ji_std[channel_index])
                ),
                "sym_coef_standardized": float(sym_std[channel_index]),
                "anti_coef_standardized": float(anti_std[channel_index]),
                "sym_coef_raw": float(sym_raw[channel_index]),
                "anti_coef_raw": float(anti_raw[channel_index]),
                "sym_norm_standardized": float("nan"),
                "anti_norm_standardized": float("nan"),
                "sym_fraction_standardized": float("nan"),
                "sym_norm_raw": float("nan"),
                "anti_norm_raw": float("nan"),
                "sym_fraction_raw": float("nan"),
                "same_sign_count": "",
                "total_channels": int(N_CHANNELS),
            }
        )
    return rows


def build_comparison_row(name, run_a, run_b):
    coef_std_a = run_a["coef_bundle"]["coef_std"]
    coef_std_b = run_b["coef_bundle"]["coef_std"]
    coef_raw_a = run_a["coef_bundle"]["coef_raw"]
    coef_raw_b = run_b["coef_bundle"]["coef_raw"]

    def cosine_similarity(a, b):
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        if denom <= FLOAT_EPS:
            return float("nan")
        return float(np.dot(a, b) / denom)

    per_test_a = {
        row["protein_id"]: float(row["mean_error_pred_minus_true"])
        for row in run_a["test_metrics"]["per_rows"]
    }
    per_test_b = {
        row["protein_id"]: float(row["mean_error_pred_minus_true"])
        for row in run_b["test_metrics"]["per_rows"]
    }
    deltas = [
        abs(per_test_b[protein_id] - per_test_a[protein_id])
        for protein_id in sorted(set(per_test_a) & set(per_test_b))
    ]

    return {
        "comparison_name": name,
        "feature_set": run_a["run_meta"]["feature_set"],
        "split_scheme": run_a["run_meta"]["split_scheme"],
        "run_name_a": run_a["run_meta"]["run_name"],
        "run_name_b": run_b["run_meta"]["run_name"],
        "intercept_mode": run_a["run_meta"]["intercept_mode"],
        "cosine_similarity_standardized": cosine_similarity(coef_std_a, coef_std_b),
        "cosine_similarity_raw": cosine_similarity(coef_raw_a, coef_raw_b),
        "coef_l2_standardized_a": float(np.linalg.norm(coef_std_a)),
        "coef_l2_standardized_b": float(np.linalg.norm(coef_std_b)),
        "coef_l2_raw_a": float(np.linalg.norm(coef_raw_a)),
        "coef_l2_raw_b": float(np.linalg.norm(coef_raw_b)),
        "coefficient_diff_l2_standardized": float(np.linalg.norm(coef_std_b - coef_std_a)),
        "coefficient_diff_l2_raw": float(np.linalg.norm(coef_raw_b - coef_raw_a)),
        "intercept_standardized_a": float(run_a["coef_bundle"]["intercept_std"]),
        "intercept_standardized_b": float(run_b["coef_bundle"]["intercept_std"]),
        "intercept_raw_a": float(run_a["coef_bundle"]["intercept_raw"]),
        "intercept_raw_b": float(run_b["coef_bundle"]["intercept_raw"]),
        "intercept_raw_diff_b_minus_a": float(
            run_b["coef_bundle"]["intercept_raw"] - run_a["coef_bundle"]["intercept_raw"]
        ),
        "mean_abs_test_bias_delta": (
            float(np.mean(deltas)) if deltas else float("nan")
        ),
        "max_abs_test_bias_delta": (
            float(np.max(deltas)) if deltas else float("nan")
        ),
    }


def choose_alpha_from_summary(summary_rows):
    """Pick the alpha with the best mean validation mean-per-protein R2."""
    best_row = None
    best_key = None
    for row in summary_rows:
        score = float(row["val_mean_per_protein_r2"])
        safe_score = -1e18 if math.isnan(score) else score
        key = (safe_score, -float(row["alpha"]))
        if best_row is None or key > best_key:
            best_row = row
            best_key = key
    return best_row


def evaluate_run(run_meta, scope, materialized):
    """Fit one outer-split run and return summary/per-protein/coefficient outputs."""
    t0 = time.time()
    fitted = fit_model(
        estimator_name=run_meta["estimator"],
        alpha=run_meta["alpha"],
        intercept_mode=run_meta["intercept_mode"],
        x_train=materialized["x_train"],
        y_train=materialized["y_train"],
        sample_weight=materialized["train_weights"],
    )

    y_pred_train = predict_fitted(fitted, materialized["x_train"])
    y_pred_test = predict_fitted(fitted, materialized["x_test"])
    train_metrics = summarize_dense_split(scope["train_proteins"], materialized["y_train"], y_pred_train)
    test_metrics = summarize_dense_split(scope["test_proteins"], materialized["y_test"], y_pred_test)

    coef_info = coefficient_bundle(
        fitted,
        materialized["scaler_mean"],
        materialized["scaler_scale"],
    )

    per_protein_rows = []
    for split_name, metric_group in [("train", train_metrics), ("test", test_metrics)]:
        for row in metric_group["per_rows"]:
            out = dict(row)
            out["run_name"] = run_meta["run_name"]
            out["analysis_family"] = run_meta["analysis_family"]
            out["feature_set"] = run_meta["feature_set"]
            out["estimator"] = run_meta["estimator"]
            out["alpha"] = float(run_meta["alpha"])
            out["intercept_mode"] = run_meta["intercept_mode"]
            out["split_scheme"] = run_meta["split_scheme"]
            out["split"] = split_name
            per_protein_rows.append(out)

    summary_row = {
        "run_name": run_meta["run_name"],
        "analysis_family": run_meta["analysis_family"],
        "feature_set": run_meta["feature_set"],
        "feature_dim": int(FEATURE_DIM_LOOKUP[run_meta["feature_set"]]),
        "estimator": run_meta["estimator"],
        "alpha": float(run_meta["alpha"]),
        "intercept_mode": run_meta["intercept_mode"],
        "split_scheme": run_meta["split_scheme"],
        "selected_by_cv": bool(run_meta.get("selected_by_cv", False)),
        "inner_cv_metric": run_meta.get("inner_cv_metric", ""),
        "n_train_proteins": int(len(scope["train_proteins"])),
        "n_test_proteins": int(len(scope["test_proteins"])),
        "n_train_rows": int(materialized["y_train"].size),
        "n_test_rows": int(materialized["y_test"].size),
        "train_pair_weighted_r2": float(train_metrics["pair_weighted_r2"]),
        "test_pair_weighted_r2": float(test_metrics["pair_weighted_r2"]),
        "train_protein_weighted_r2": float(train_metrics["protein_weighted_r2"]),
        "test_protein_weighted_r2": float(test_metrics["protein_weighted_r2"]),
        "train_pair_weighted_mae": float(train_metrics["pair_weighted_mae"]),
        "test_pair_weighted_mae": float(test_metrics["pair_weighted_mae"]),
        "train_pair_weighted_rmse": float(train_metrics["pair_weighted_rmse"]),
        "test_pair_weighted_rmse": float(test_metrics["pair_weighted_rmse"]),
        "train_mean_per_protein_r2": float(train_metrics["mean_per_protein_r2"]),
        "test_mean_per_protein_r2": float(test_metrics["mean_per_protein_r2"]),
        "test_median_per_protein_r2": float(test_metrics["median_per_protein_r2"]),
        "test_negative_r2_fraction": float(test_metrics["negative_r2_fraction"]),
        "train_mean_per_protein_nmse": float(train_metrics["mean_per_protein_nmse"]),
        "test_mean_per_protein_nmse": float(test_metrics["mean_per_protein_nmse"]),
        "test_median_per_protein_nmse": float(test_metrics["median_per_protein_nmse"]),
        "test_max_per_protein_nmse": float(test_metrics["max_per_protein_nmse"]),
        "train_mean_bias": float(train_metrics["mean_bias"]),
        "test_mean_bias": float(test_metrics["mean_bias"]),
        "intercept_standardized": float(coef_info["intercept_std"]),
        "intercept_raw": float(coef_info["intercept_raw"]),
        "coef_l2_standardized": float(np.linalg.norm(coef_info["coef_std"])),
        "coef_l2_raw": float(np.linalg.norm(coef_info["coef_raw"])),
        "n_nonzero_coefficients": int(np.sum(np.abs(coef_info["coef_std"]) > 1e-12)),
        "elapsed_seconds": float(time.time() - t0),
    }

    coefficient_rows = build_coefficient_rows(
        run_meta,
        fitted,
        materialized["scaler_mean"],
        materialized["scaler_scale"],
        materialized["feature_names"],
    )

    symmetry_rows = []
    if run_meta["feature_set"] == FEATURE_SET_PAIRED:
        symmetry_rows = build_symmetry_coefficient_rows(
            run_meta,
            coef_info["coef_std"],
            coef_info["coef_raw"],
        )

    return {
        "summary_row": summary_row,
        "per_protein_rows": per_protein_rows,
        "coefficient_rows": coefficient_rows,
        "symmetry_rows": symmetry_rows,
        "train_metrics": train_metrics,
        "test_metrics": test_metrics,
        "coef_bundle": coef_info,
        "run_meta": run_meta,
    }


def run_alpha_sweep(model_family, feature_set, outer_train_records, chunk_maps):
    """Inner-CV alpha sweep on the outer training proteins."""
    fold_family, folds = build_inner_folds(outer_train_records)
    alpha_grid = RIDGE_ALPHA_GRID if model_family == ESTIMATOR_RIDGE else LASSO_ALPHA_GRID

    cv_rows = []
    alpha_summary_rows = []
    for alpha in alpha_grid:
        alpha_fold_rows = []
        for fold in folds:
            t0 = time.time()
            inner_scope = {
                "name": fold_family,
                "train_proteins": [dict(record) for record in fold["train_records"]],
                "test_proteins": [dict(record) for record in fold["val_records"]],
            }
            materialized = build_materialized_scope(inner_scope, feature_set, chunk_maps)
            fitted = fit_model(
                estimator_name=model_family,
                alpha=alpha,
                intercept_mode=INTERCEPT_FIT,
                x_train=materialized["x_train"],
                y_train=materialized["y_train"],
                sample_weight=materialized["train_weights"],
            )
            y_val_pred = predict_fitted(fitted, materialized["x_test"])
            val_metrics = summarize_dense_split(
                inner_scope["test_proteins"],
                materialized["y_test"],
                y_val_pred,
            )
            row = {
                "row_type": "fold",
                "model_family": model_family,
                "feature_set": feature_set,
                "alpha": float(alpha),
                "fold_family": fold_family,
                "fold_index": int(fold["fold_index"]),
                "n_train_proteins": int(len(inner_scope["train_proteins"])),
                "n_val_proteins": int(len(inner_scope["test_proteins"])),
                "val_mean_per_protein_r2": float(val_metrics["mean_per_protein_r2"]),
                "val_pair_weighted_r2": float(val_metrics["pair_weighted_r2"]),
                "val_pair_weighted_mae": float(val_metrics["pair_weighted_mae"]),
                "val_pair_weighted_rmse": float(val_metrics["pair_weighted_rmse"]),
                "val_mean_bias": float(val_metrics["mean_bias"]),
                "n_nonzero_coefficients": int(np.sum(np.abs(fitted["coef_std"]) > 1e-12)),
                "elapsed_seconds": float(time.time() - t0),
            }
            alpha_fold_rows.append(row)
            cv_rows.append(row)

        alpha_summary = {
            "row_type": "alpha_summary",
            "model_family": model_family,
            "feature_set": feature_set,
            "alpha": float(alpha),
            "fold_family": fold_family,
            "fold_index": -1,
            "n_train_proteins": int(np.mean([row["n_train_proteins"] for row in alpha_fold_rows])),
            "n_val_proteins": int(np.mean([row["n_val_proteins"] for row in alpha_fold_rows])),
            "val_mean_per_protein_r2": float(
                np.mean([row["val_mean_per_protein_r2"] for row in alpha_fold_rows])
            ),
            "val_pair_weighted_r2": float(
                np.mean([row["val_pair_weighted_r2"] for row in alpha_fold_rows])
            ),
            "val_pair_weighted_mae": float(
                np.mean([row["val_pair_weighted_mae"] for row in alpha_fold_rows])
            ),
            "val_pair_weighted_rmse": float(
                np.mean([row["val_pair_weighted_rmse"] for row in alpha_fold_rows])
            ),
            "val_mean_bias": float(
                np.mean([row["val_mean_bias"] for row in alpha_fold_rows])
            ),
            "n_nonzero_coefficients": int(
                np.mean([row["n_nonzero_coefficients"] for row in alpha_fold_rows])
            ),
            "elapsed_seconds": float(np.sum([row["elapsed_seconds"] for row in alpha_fold_rows])),
        }
        alpha_summary_rows.append(alpha_summary)
        cv_rows.append(alpha_summary)

    chosen = choose_alpha_from_summary(alpha_summary_rows)
    return fold_family, cv_rows, chosen


def build_summary_text(
    path,
    elapsed_seconds,
    all_proteins,
    loaded_records,
    skipped,
    summary_rows,
    design_rows,
    comparison_rows,
    alpha_selection_rows,
):
    core_rows = [row for row in summary_rows if row["analysis_family"] == "core"]
    best_core = max(
        core_rows,
        key=lambda row: (
            -1e18
            if math.isnan(float(row["test_mean_per_protein_r2"]))
            else float(row["test_mean_per_protein_r2"])
        ),
    )

    with path.open("w") as handle:
        handle.write(f"{TITLE}\n")
        handle.write(f"{'=' * len(TITLE)}\n\n")
        handle.write("Model family:      protein-equal weighted linear / ridge / lasso probes\n")
        handle.write(
            "Feature policy:    layer-47 lower-triangle center-only 128d and paired 256d "
            "features in standardized train-only feature space\n"
        )
        handle.write("Target policy:     absolute CA distances\n")
        handle.write(
            "Default split:     protein-level 80/20 with residue-size stratification when feasible\n"
        )
        handle.write(
            "Special analyses:  intercept ablations, design conditioning, symmetry diagnostics, "
            "ridge/lasso alpha selection, and random-vs-stratified ridge ablations\n\n"
        )

        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(loaded_records)}\n")
        handle.write(f"Elapsed seconds:     {elapsed_seconds:.2f}\n\n")

        handle.write("Core variants:\n")
        for row in core_rows:
            handle.write(
                f"  {row['run_name']}: "
                f"test mean per-protein R2={float(row['test_mean_per_protein_r2']):.4f}, "
                f"test pair-weighted R2={float(row['test_pair_weighted_r2']):.4f}, "
                f"intercept_raw={float(row['intercept_raw']):.4f}, "
                f"coef_l2_raw={float(row['coef_l2_raw']):.4f}\n"
            )

        handle.write(
            f"\nBest core variant by test mean per-protein R2: {best_core['run_name']} "
            f"({float(best_core['test_mean_per_protein_r2']):.4f})\n"
        )

        handle.write("\nDesign diagnostics:\n")
        for row in design_rows:
            handle.write(
                f"  split={row['split_scheme']}, feature_set={row['feature_set']}: "
                f"condition_number={float(row['condition_number']):.4e}, "
                f"effective_rank={float(row['effective_rank']):.2f}, "
                f"near_null_1e-8={int(row['near_null_rel_1e-8'])}\n"
            )

        handle.write("\nLinear vs ridge comparisons:\n")
        for row in comparison_rows:
            if not row["comparison_name"].endswith("_linear_vs_ridge"):
                continue
            handle.write(
                f"  {row['comparison_name']}: "
                f"cosine_raw={float(row['cosine_similarity_raw']):.4f}, "
                f"coef_diff_l2_raw={float(row['coefficient_diff_l2_raw']):.4f}, "
                f"intercept_raw_diff={float(row['intercept_raw_diff_b_minus_a']):+.4f}\n"
            )

        handle.write("\nCV-selected models:\n")
        for row in alpha_selection_rows:
            handle.write(
                f"  {row['model_family']} / {row['feature_set']}: "
                f"alpha={float(row['chosen_alpha']):.4g}, "
                f"outer test mean per-protein R2={float(row['outer_test_mean_per_protein_r2']):.4f}, "
                f"outer test pair-weighted R2={float(row['outer_test_pair_weighted_r2']):.4f}\n"
            )

        if skipped:
            handle.write("\nSkipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")


def main():
    t0 = time.time()
    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))
    print(f"Output CSV dir: {csv_dir}")

    all_proteins, protein_data, skipped = load_protein_records()
    if not protein_data:
        print("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    prepared_records = prepare_records(protein_data)
    assign_size_bins(prepared_records, N_SIZE_BINS)
    chunk_maps = build_feature_caches(prepared_records)

    scopes = {
        SPLIT_SCHEME_STRATIFIED: build_scope_spec(
            SPLIT_SCHEME_STRATIFIED,
            prepared_records,
            stratified=True,
        ),
        SPLIT_SCHEME_RANDOM: build_scope_spec(
            SPLIT_SCHEME_RANDOM,
            prepared_records,
            stratified=False,
        ),
    }

    print(f"Loaded proteins: {len(prepared_records)}")
    for split_name, scope in scopes.items():
        print(
            f"  {split_name}: train={len(scope['train_proteins'])}, "
            f"test={len(scope['test_proteins'])}"
        )

    split_rows = build_split_rows(
        [
            (
                split_name,
                scope["train_proteins"],
                scope["test_proteins"],
            )
            for split_name, scope in scopes.items()
        ]
    )

    materialized_lookup = {}
    design_rows = []
    symmetry_rows = []
    for split_name, scope in scopes.items():
        materialized_lookup[split_name] = {}
        for feature_set in [FEATURE_SET_CENTER, FEATURE_SET_PAIRED]:
            materialized = build_materialized_scope(scope, feature_set, chunk_maps)
            materialized_lookup[split_name][feature_set] = materialized
            design_rows.append(compute_design_diagnostics(scope, feature_set, materialized))
            if feature_set == FEATURE_SET_PAIRED:
                symmetry_rows.extend(build_symmetry_design_rows(split_name, materialized))

    summary_rows = []
    per_protein_rows = []
    coefficient_rows = []
    comparison_rows = []
    alpha_cv_rows = []
    alpha_selection_rows = []
    run_results = {}

    outer_run_specs = []
    for feature_set in [FEATURE_SET_CENTER, FEATURE_SET_PAIRED]:
        label_prefix = feature_set
        outer_run_specs.extend(
            [
                {
                    "run_name": f"{label_prefix}_linear_no_ridge",
                    "analysis_family": "core",
                    "feature_set": feature_set,
                    "estimator": ESTIMATOR_LINEAR,
                    "alpha": 0.0,
                    "intercept_mode": INTERCEPT_FIT,
                    "split_scheme": SPLIT_SCHEME_STRATIFIED,
                },
                {
                    "run_name": f"{label_prefix}_ridge_1e-8",
                    "analysis_family": "core",
                    "feature_set": feature_set,
                    "estimator": ESTIMATOR_RIDGE,
                    "alpha": RIDGE_ALPHA_TINY,
                    "intercept_mode": INTERCEPT_FIT,
                    "split_scheme": SPLIT_SCHEME_STRATIFIED,
                },
                {
                    "run_name": f"{label_prefix}_linear_no_ridge_no_intercept",
                    "analysis_family": "intercept_ablation",
                    "feature_set": feature_set,
                    "estimator": ESTIMATOR_LINEAR,
                    "alpha": 0.0,
                    "intercept_mode": INTERCEPT_NONE,
                    "split_scheme": SPLIT_SCHEME_STRATIFIED,
                },
                {
                    "run_name": f"{label_prefix}_ridge_1e-8_no_intercept",
                    "analysis_family": "intercept_ablation",
                    "feature_set": feature_set,
                    "estimator": ESTIMATOR_RIDGE,
                    "alpha": RIDGE_ALPHA_TINY,
                    "intercept_mode": INTERCEPT_NONE,
                    "split_scheme": SPLIT_SCHEME_STRATIFIED,
                },
                {
                    "run_name": f"{label_prefix}_linear_no_ridge_centered_target",
                    "analysis_family": "intercept_ablation",
                    "feature_set": feature_set,
                    "estimator": ESTIMATOR_LINEAR,
                    "alpha": 0.0,
                    "intercept_mode": INTERCEPT_CENTERED,
                    "split_scheme": SPLIT_SCHEME_STRATIFIED,
                },
                {
                    "run_name": f"{label_prefix}_ridge_1e-8_centered_target",
                    "analysis_family": "intercept_ablation",
                    "feature_set": feature_set,
                    "estimator": ESTIMATOR_RIDGE,
                    "alpha": RIDGE_ALPHA_TINY,
                    "intercept_mode": INTERCEPT_CENTERED,
                    "split_scheme": SPLIT_SCHEME_STRATIFIED,
                },
            ]
        )

    for feature_set in [FEATURE_SET_CENTER, FEATURE_SET_PAIRED]:
        outer_run_specs.append(
            {
                "run_name": f"{feature_set}_ridge_1e-8_random_split",
                "analysis_family": "split_ablation",
                "feature_set": feature_set,
                "estimator": ESTIMATOR_RIDGE,
                "alpha": RIDGE_ALPHA_TINY,
                "intercept_mode": INTERCEPT_FIT,
                "split_scheme": SPLIT_SCHEME_RANDOM,
            }
        )

    for run_meta in outer_run_specs:
        print(f"\n[Run] {run_meta['run_name']}")
        scope = scopes[run_meta["split_scheme"]]
        materialized = materialized_lookup[run_meta["split_scheme"]][run_meta["feature_set"]]
        result = evaluate_run(run_meta, scope, materialized)
        run_results[run_meta["run_name"]] = result
        summary_rows.append(result["summary_row"])
        per_protein_rows.extend(result["per_protein_rows"])
        coefficient_rows.extend(result["coefficient_rows"])
        symmetry_rows.extend(result["symmetry_rows"])
        print(
            "  "
            f"test mean per-protein R2={result['summary_row']['test_mean_per_protein_r2']:.4f}, "
            f"test pair-weighted R2={result['summary_row']['test_pair_weighted_r2']:.4f}"
        )

    for feature_set in [FEATURE_SET_CENTER, FEATURE_SET_PAIRED]:
        comparison_rows.append(
            build_comparison_row(
                f"{feature_set}_fit_intercept_linear_vs_ridge",
                run_results[f"{feature_set}_linear_no_ridge"],
                run_results[f"{feature_set}_ridge_1e-8"],
            )
        )
        comparison_rows.append(
            build_comparison_row(
                f"{feature_set}_no_intercept_linear_vs_ridge",
                run_results[f"{feature_set}_linear_no_ridge_no_intercept"],
                run_results[f"{feature_set}_ridge_1e-8_no_intercept"],
            )
        )
        comparison_rows.append(
            build_comparison_row(
                f"{feature_set}_centered_target_linear_vs_ridge",
                run_results[f"{feature_set}_linear_no_ridge_centered_target"],
                run_results[f"{feature_set}_ridge_1e-8_centered_target"],
            )
        )

    default_scope = scopes[SPLIT_SCHEME_STRATIFIED]
    for feature_set in [FEATURE_SET_CENTER, FEATURE_SET_PAIRED]:
        for model_family in [ESTIMATOR_RIDGE, ESTIMATOR_LASSO]:
            print(f"\n[CV] {model_family} / {feature_set}")
            fold_family, cv_rows, chosen_alpha_row = run_alpha_sweep(
                model_family=model_family,
                feature_set=feature_set,
                outer_train_records=default_scope["train_proteins"],
                chunk_maps=chunk_maps,
            )
            alpha_cv_rows.extend(cv_rows)

            chosen_alpha = float(chosen_alpha_row["alpha"])
            run_name = f"{feature_set}_{model_family}_cv_selected"
            run_meta = {
                "run_name": run_name,
                "analysis_family": "cv_selected",
                "feature_set": feature_set,
                "estimator": model_family,
                "alpha": chosen_alpha,
                "intercept_mode": INTERCEPT_FIT,
                "split_scheme": SPLIT_SCHEME_STRATIFIED,
                "selected_by_cv": True,
                "inner_cv_metric": "mean_per_protein_r2",
            }
            result = evaluate_run(
                run_meta,
                default_scope,
                materialized_lookup[SPLIT_SCHEME_STRATIFIED][feature_set],
            )
            run_results[run_name] = result
            summary_rows.append(result["summary_row"])
            per_protein_rows.extend(result["per_protein_rows"])
            coefficient_rows.extend(result["coefficient_rows"])
            symmetry_rows.extend(result["symmetry_rows"])

            alpha_selection_rows.append(
                {
                    "model_family": model_family,
                    "feature_set": feature_set,
                    "split_scheme": SPLIT_SCHEME_STRATIFIED,
                    "selected_metric": "mean_per_protein_r2",
                    "chosen_alpha": chosen_alpha,
                    "inner_cv_fold_family": fold_family,
                    "inner_cv_mean_per_protein_r2": float(
                        chosen_alpha_row["val_mean_per_protein_r2"]
                    ),
                    "inner_cv_std_mean_per_protein_r2": float(
                        np.std(
                            [
                                row["val_mean_per_protein_r2"]
                                for row in cv_rows
                                if row["row_type"] == "fold"
                                and row["feature_set"] == feature_set
                                and row["model_family"] == model_family
                                and math.isclose(
                                    float(row["alpha"]),
                                    chosen_alpha,
                                    rel_tol=0.0,
                                    abs_tol=1e-18,
                                )
                            ]
                        )
                    ),
                    "inner_cv_mean_pair_weighted_r2": float(
                        chosen_alpha_row["val_pair_weighted_r2"]
                    ),
                    "inner_cv_mean_pair_weighted_mae": float(
                        chosen_alpha_row["val_pair_weighted_mae"]
                    ),
                    "outer_test_mean_per_protein_r2": float(
                        result["summary_row"]["test_mean_per_protein_r2"]
                    ),
                    "outer_test_pair_weighted_r2": float(
                        result["summary_row"]["test_pair_weighted_r2"]
                    ),
                    "outer_test_pair_weighted_mae": float(
                        result["summary_row"]["test_pair_weighted_mae"]
                    ),
                    "outer_test_pair_weighted_rmse": float(
                        result["summary_row"]["test_pair_weighted_rmse"]
                    ),
                    "outer_test_mean_bias": float(result["summary_row"]["test_mean_bias"]),
                    "outer_test_n_nonzero_coefficients": int(
                        result["summary_row"]["n_nonzero_coefficients"]
                    ),
                }
            )

    write_csv(csv_dir / SUMMARY_CSV_NAME, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(csv_dir / PER_PROTEIN_CSV_NAME, PER_PROTEIN_FIELDNAMES, per_protein_rows)
    write_csv(csv_dir / SPLIT_CSV_NAME, SPLIT_FIELDNAMES, split_rows)
    write_csv(csv_dir / COEFFICIENT_CSV_NAME, COEFFICIENT_FIELDNAMES, coefficient_rows)
    write_csv(
        csv_dir / DESIGN_DIAGNOSTICS_CSV_NAME,
        DESIGN_DIAGNOSTICS_FIELDNAMES,
        design_rows,
    )
    write_csv(
        csv_dir / SYMMETRY_DIAGNOSTICS_CSV_NAME,
        SYMMETRY_DIAGNOSTICS_FIELDNAMES,
        symmetry_rows,
    )
    write_csv(csv_dir / COMPARISON_CSV_NAME, COMPARISON_FIELDNAMES, comparison_rows)
    write_csv(csv_dir / ALPHA_CV_CSV_NAME, ALPHA_CV_FIELDNAMES, alpha_cv_rows)
    write_csv(
        csv_dir / ALPHA_SELECTION_CSV_NAME,
        ALPHA_SELECTION_FIELDNAMES,
        alpha_selection_rows,
    )
    build_summary_text(
        csv_dir / SUMMARY_TXT_NAME,
        elapsed_seconds=float(time.time() - t0),
        all_proteins=all_proteins,
        loaded_records=prepared_records,
        skipped=skipped,
        summary_rows=summary_rows,
        design_rows=design_rows,
        comparison_rows=comparison_rows,
        alpha_selection_rows=alpha_selection_rows,
    )

    print(f"\nSaved summary CSV -> {csv_dir / SUMMARY_CSV_NAME}")
    print(f"Saved per-protein CSV -> {csv_dir / PER_PROTEIN_CSV_NAME}")
    print(f"Saved split CSV -> {csv_dir / SPLIT_CSV_NAME}")
    print(f"Saved coefficient CSV -> {csv_dir / COEFFICIENT_CSV_NAME}")
    print(f"Saved design diagnostics CSV -> {csv_dir / DESIGN_DIAGNOSTICS_CSV_NAME}")
    print(f"Saved symmetry diagnostics CSV -> {csv_dir / SYMMETRY_DIAGNOSTICS_CSV_NAME}")
    print(f"Saved comparison CSV -> {csv_dir / COMPARISON_CSV_NAME}")
    print(f"Saved alpha CV CSV -> {csv_dir / ALPHA_CV_CSV_NAME}")
    print(f"Saved alpha selection CSV -> {csv_dir / ALPHA_SELECTION_CSV_NAME}")
    print(f"Saved text summary -> {csv_dir / SUMMARY_TXT_NAME}")


if __name__ == "__main__":
    main()

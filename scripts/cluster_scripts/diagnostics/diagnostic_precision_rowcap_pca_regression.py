#!/usr/bin/env python3
"""
diagnostic_precision_rowcap_pca_regression.py
============================================
Layer-47 diagnostic for separating row-count-driven float-precision effects
from real protein-size heterogeneity in the global linear probes.

This script reuses one deterministic protein-level split and compares:
1. 128d lower-triangle center features with protein-equal weighting across:
   - row_cap_per_train_protein in {5000, 20000, all}
   - dtype in {float32, float64}
2. 256d paired lower-triangle features with protein-equal weighting across:
   - row_cap_per_train_protein in {5000, 20000, all}
   - dtype in {float32, float64}
3. One extra 256d float64 pair-equal all-row control.
4. One 128d float64 weighted-PCA ablation over k in
   {1, 2, 5, 10, 20, 50, 100, 128}.
5. One 128d float64 fixed-bucket linear probe using residue-count buckets:
   <=255, 256-371, 372-500, 501+.

All variants:
- train on one deterministic size-stratified 80/20 protein split
- cap only the training rows used for fitting
- evaluate train and test metrics on all rows of every protein
- export per-protein R^2 and model coefficients

Outputs (inside CSV_ROOT/):
    diagnostic_precision_rowcap_pca_variant_summary.csv
    diagnostic_precision_rowcap_pca_per_protein.csv
    diagnostic_precision_rowcap_pca_split.csv
    diagnostic_precision_rowcap_pca_spectrum_summary.csv
    diagnostic_precision_rowcap_pca_singular_values.csv
    diagnostic_precision_rowcap_pca_coefficients.csv
    diagnostic_precision_rowcap_pca_summary.txt
"""

import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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
    deterministic_seed,
    load_protein_records,
    write_csv,
)


TITLE = "Diagnostic Precision / Row-Cap / PCA Regression - Layer 47"

N_SIZE_BINS = 5
ROW_CAP_OPTIONS = [5000, 20000, None]
PCA_K_VALUES = [1, 2, 5, 10, 20, 50, 100, 128]
PREDICT_BATCH_ROWS = 200_000

FEATURE_SET_128 = "128d"
FEATURE_SET_256 = "256d"
WEIGHTING_PROTEIN_EQUAL = "protein_equal"
WEIGHTING_PAIR_EQUAL = "pair_equal"

ANALYSIS_FAMILY_GLOBAL = "global_raw"
ANALYSIS_FAMILY_PCA = "pca"
ANALYSIS_FAMILY_BUCKETED = "bucketed"

BUCKET_SCHEME_MANUAL = "manual_255_371_500"

CENTER_FEATURE_NAMES = [f"pair_ch_{idx}" for idx in range(N_CHANNELS)]
PAIRED_FEATURE_NAMES = [f"pair_ij_ch_{idx}" for idx in range(N_CHANNELS)] + [
    f"pair_ji_ch_{idx}" for idx in range(N_CHANNELS)
]

MANUAL_BUCKETS = [
    {
        "label": "<=255",
        "min_residues": 0,
        "max_residues": 255,
    },
    {
        "label": "256-371",
        "min_residues": 256,
        "max_residues": 371,
    },
    {
        "label": "372-500",
        "min_residues": 372,
        "max_residues": 500,
    },
    {
        "label": "501+",
        "min_residues": 501,
        "max_residues": None,
    },
]

SUMMARY_CSV_NAME = "diagnostic_precision_rowcap_pca_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_precision_rowcap_pca_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_precision_rowcap_pca_split.csv"
SPECTRUM_SUMMARY_CSV_NAME = "diagnostic_precision_rowcap_pca_spectrum_summary.csv"
SINGULAR_VALUES_CSV_NAME = "diagnostic_precision_rowcap_pca_singular_values.csv"
COEFFICIENT_CSV_NAME = "diagnostic_precision_rowcap_pca_coefficients.csv"
SUMMARY_TXT_NAME = "diagnostic_precision_rowcap_pca_summary.txt"

SUMMARY_FIELDNAMES = [
    "variant_id",
    "analysis_family",
    "feature_set",
    "feature_dim",
    "fit_feature_dim",
    "dtype",
    "weighting_scheme",
    "row_cap_per_train_protein",
    "pca_k",
    "bucket_scheme",
    "trained_model_count",
    "n_train_proteins",
    "n_test_proteins",
    "n_train_rows_fit",
    "n_train_rows_eval",
    "n_test_rows_eval",
    "train_mean_per_protein_r2",
    "test_mean_per_protein_r2",
    "test_median_per_protein_r2",
    "test_negative_r2_count",
    "test_negative_r2_fraction",
    "elapsed_seconds",
]

PER_PROTEIN_FIELDNAMES = [
    "variant_id",
    "analysis_family",
    "feature_set",
    "feature_dim",
    "fit_feature_dim",
    "dtype",
    "weighting_scheme",
    "row_cap_per_train_protein",
    "pca_k",
    "bucket_scheme",
    "bucket_label",
    "model_bucket_label",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
    "per_protein_r2",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
]

SPLIT_FIELDNAMES = [
    "protein_id",
    "split",
    "n_residues",
    "size_bin",
    "bucket_label",
    "n_pairs",
    "target_variance",
]

SPECTRUM_SUMMARY_FIELDNAMES = [
    "feature_set",
    "weighting_scheme",
    "row_cap_per_train_protein",
    "n_train_proteins",
    "n_train_rows_fit",
    "feature_dim",
    "weighted_column_mean_abs_max",
    "weighted_column_mean_abs_mean",
    "max_singular_value",
    "min_positive_singular_value",
    "condition_number",
    "effective_rank",
    "cond_float32",
    "cond_float64",
    "cutoff_float32",
    "cutoff_float64",
    "n_cutoff_float32",
    "n_cutoff_float64",
]

SINGULAR_VALUES_FIELDNAMES = [
    "feature_set",
    "weighting_scheme",
    "row_cap_per_train_protein",
    "rank_index",
    "singular_value",
    "relative_to_max",
    "cut_by_float32",
    "cut_by_float64",
]

COEFFICIENT_FIELDNAMES = [
    "variant_id",
    "analysis_family",
    "feature_set",
    "feature_dim",
    "fit_feature_dim",
    "dtype",
    "weighting_scheme",
    "row_cap_per_train_protein",
    "pca_k",
    "bucket_scheme",
    "bucket_label",
    "model_scope",
    "feature_index",
    "feature_name",
    "coefficient_standardized",
    "intercept_standardized",
    "coefficient_raw_scale",
    "intercept_raw",
]


def feature_dim_for_set(feature_set):
    if feature_set == FEATURE_SET_128:
        return N_CHANNELS
    if feature_set == FEATURE_SET_256:
        return 2 * N_CHANNELS
    raise ValueError(f"Unknown feature set: {feature_set}")


def feature_names_for_set(feature_set):
    if feature_set == FEATURE_SET_128:
        return CENTER_FEATURE_NAMES
    if feature_set == FEATURE_SET_256:
        return PAIRED_FEATURE_NAMES
    raise ValueError(f"Unknown feature set: {feature_set}")


def row_cap_label(row_cap):
    return "all" if row_cap is None else str(int(row_cap))


def assign_size_bins(records, n_bins):
    ordered = sorted(records, key=lambda row: (row["n_residues"], row["protein_id"]))
    total = len(ordered)
    for rank, record in enumerate(ordered):
        size_bin = int((rank * n_bins) // max(total, 1))
        if size_bin >= n_bins:
            size_bin = n_bins - 1
        record["size_bin"] = int(size_bin)


def choose_stratify_labels(records):
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


def split_records(records):
    indices = np.arange(len(records), dtype=np.int64)
    stratify_labels = choose_stratify_labels(records)
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


def bucket_label_for_residues(n_residues):
    for bucket in MANUAL_BUCKETS:
        max_residues = bucket["max_residues"]
        if max_residues is None:
            if n_residues >= bucket["min_residues"]:
                return bucket["label"]
        elif bucket["min_residues"] <= n_residues <= max_residues:
            return bucket["label"]
    raise ValueError(f"No manual bucket for residue count {n_residues}")


def prepare_records(records):
    prepared = []
    for record in records:
        target_variance = float(np.var(record["y_abs"], dtype=np.float64))
        if target_variance <= FLOAT_EPS:
            target_variance = FLOAT_EPS
        prepared_record = dict(record)
        prepared_record["target_variance"] = target_variance
        prepared_record["bucket_label"] = bucket_label_for_residues(int(record["n_residues"]))
        prepared.append(prepared_record)
    return prepared


def build_split_rows(train_records, test_records):
    rows = []
    for split_name, split_records in [("train", train_records), ("test", test_records)]:
        for record in split_records:
            rows.append(
                {
                    "protein_id": record["protein_id"],
                    "split": split_name,
                    "n_residues": int(record["n_residues"]),
                    "size_bin": int(record["size_bin"]),
                    "bucket_label": record["bucket_label"],
                    "n_pairs": int(record["n_pairs"]),
                    "target_variance": float(record["target_variance"]),
                }
            )
    return rows


def build_feature_chunk_map(records, feature_set):
    chunk_map = {}
    for idx, record in enumerate(records, start=1):
        protein_id = record["protein_id"]
        pb_path = record["protein_dir"] / f"{protein_id}_pair_block_{LAYER}.npy"
        pb = np.load(str(pb_path), mmap_mode="r")
        if feature_set == FEATURE_SET_128:
            chunk = np.asarray(pb[record["lt_i"], record["lt_j"], :], dtype=np.float32)
        elif feature_set == FEATURE_SET_256:
            chunk = np.concatenate(
                [
                    pb[record["lt_i"], record["lt_j"], :],
                    pb[record["lt_j"], record["lt_i"], :],
                ],
                axis=1,
            ).astype(np.float32, copy=False)
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"Unknown feature set: {feature_set}")
        del pb
        chunk_map[protein_id] = chunk
        if idx % 25 == 0 or idx == len(records):
            print(
                f"  Loaded feature chunks for {feature_set}: {idx}/{len(records)} proteins",
                flush=True,
            )
    return chunk_map


def full_range_indices(n_rows):
    return np.arange(n_rows, dtype=np.int64)


def build_train_selection_maps(train_records):
    selection_maps = {row_cap_label(cap): {} for cap in ROW_CAP_OPTIONS}
    fit_row_counts = {row_cap_label(cap): 0 for cap in ROW_CAP_OPTIONS}

    for record in train_records:
        protein_id = record["protein_id"]
        n_rows = int(record["n_pairs"])
        all_idx = full_range_indices(n_rows)

        rng = np.random.default_rng(
            deterministic_seed(f"{protein_id}:{n_rows}:rowcap_permutation")
        )
        permutation = None

        for row_cap in ROW_CAP_OPTIONS:
            label = row_cap_label(row_cap)
            if row_cap is None or n_rows <= row_cap:
                chosen = all_idx
            else:
                if permutation is None:
                    permutation = rng.permutation(n_rows)
                chosen = np.sort(np.asarray(permutation[: int(row_cap)], dtype=np.int64))
            selection_maps[label][protein_id] = chosen
            fit_row_counts[label] += int(chosen.size)
    return selection_maps, fit_row_counts


def selected_chunk(raw_chunk, local_idx):
    if local_idx.size == 0:
        return raw_chunk[:0]
    if (
        local_idx.size == raw_chunk.shape[0]
        and int(local_idx[0]) == 0
        and int(local_idx[-1]) == raw_chunk.shape[0] - 1
    ):
        return raw_chunk
    return raw_chunk[local_idx]


def compute_scaler(train_records, chunk_map, selection_map, weighting_scheme):
    feature_dim = int(next(iter(chunk_map.values())).shape[1])
    mean_acc = np.zeros((feature_dim,), dtype=np.float64)
    sq_acc = np.zeros((feature_dim,), dtype=np.float64)
    weight_acc = 0.0

    for record in train_records:
        local_idx = selection_map[record["protein_id"]]
        if local_idx.size == 0:
            continue
        chunk = selected_chunk(chunk_map[record["protein_id"]], local_idx)
        chunk_mean = np.asarray(np.mean(chunk, axis=0), dtype=np.float64)
        chunk_sq_mean = np.asarray(np.mean(chunk * chunk, axis=0), dtype=np.float64)

        if weighting_scheme == WEIGHTING_PROTEIN_EQUAL:
            protein_weight = 1.0
        elif weighting_scheme == WEIGHTING_PAIR_EQUAL:
            protein_weight = float(local_idx.size)
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")

        mean_acc += protein_weight * chunk_mean
        sq_acc += protein_weight * chunk_sq_mean
        weight_acc += protein_weight

    if weight_acc <= 0.0:
        raise ValueError("Scaler computation received zero total weight.")

    mean = mean_acc / weight_acc
    second_moment = sq_acc / weight_acc
    var = np.maximum(second_moment - (mean * mean), 0.0)
    scale = np.sqrt(var)
    scale[scale < FLOAT_EPS] = 1.0
    return mean, scale


def compute_train_weights(train_records, selection_map, weighting_scheme):
    pieces = []
    for record in train_records:
        n_rows = int(selection_map[record["protein_id"]].size)
        if n_rows <= 0:
            continue
        if weighting_scheme == WEIGHTING_PROTEIN_EQUAL:
            piece = np.full((n_rows,), 1.0 / float(n_rows), dtype=np.float64)
        elif weighting_scheme == WEIGHTING_PAIR_EQUAL:
            piece = np.ones((n_rows,), dtype=np.float64)
        else:  # pragma: no cover - guarded by callers
            raise ValueError(f"Unknown weighting scheme: {weighting_scheme}")
        pieces.append(piece)

    if not pieces:
        return np.empty((0,), dtype=np.float64)

    weights = np.concatenate(pieces)
    weights *= weights.size / np.sum(weights)
    return weights


def build_fit_matrix(train_records, chunk_map, selection_map, scaler_mean, scaler_scale, np_dtype):
    total_rows = int(sum(selection_map[record["protein_id"]].size for record in train_records))
    feature_dim = int(next(iter(chunk_map.values())).shape[1])

    x_fit = np.empty((total_rows, feature_dim), dtype=np_dtype)
    y_fit = np.empty((total_rows,), dtype=np_dtype)

    mean_cast = scaler_mean.astype(np_dtype, copy=False)
    scale_cast = scaler_scale.astype(np_dtype, copy=False)

    start = 0
    for record in train_records:
        protein_id = record["protein_id"]
        local_idx = selection_map[protein_id]
        n_rows = int(local_idx.size)
        if n_rows == 0:
            continue
        end = start + n_rows

        raw_chunk = selected_chunk(chunk_map[protein_id], local_idx)
        x_block = raw_chunk.astype(np_dtype, copy=True)
        x_block -= mean_cast
        x_block /= scale_cast

        x_fit[start:end, :] = x_block
        y_fit[start:end] = np.asarray(record["y_abs"][local_idx], dtype=np_dtype)
        start = end

    return x_fit, y_fit


def standardized_to_raw_coefficients(coef_std, intercept_std, scaler_mean, scaler_scale):
    coef_std = np.asarray(coef_std, dtype=np.float64)
    scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
    coef_raw = coef_std / scaler_scale
    intercept_raw = float(
        float(intercept_std) - np.sum((coef_std * scaler_mean) / scaler_scale, dtype=np.float64)
    )
    return coef_raw, intercept_raw


def weighted_column_mean(matrix, weights):
    weights = np.asarray(weights, dtype=np.float64)
    denom = float(np.sum(weights))
    if denom <= 0.0:
        return np.zeros((matrix.shape[1],), dtype=np.float64)
    return np.sum(matrix * weights[:, None], axis=0, dtype=np.float64) / denom


def compute_effective_rank(eigenvalues):
    positive = np.asarray(eigenvalues, dtype=np.float64)
    positive = positive[positive > FLOAT_EPS]
    if positive.size == 0:
        return 0.0
    probs = positive / np.sum(positive)
    entropy = -np.sum(probs * np.log(probs))
    return float(np.exp(entropy))


def canonicalize_component_signs(components):
    components = np.asarray(components, dtype=np.float64).copy()
    for col_idx in range(components.shape[1]):
        column = components[:, col_idx]
        max_idx = int(np.argmax(np.abs(column)))
        if column[max_idx] < 0.0:
            components[:, col_idx] *= -1.0
    return components


def compute_spectrum_diagnostics(feature_set, weighting_scheme, row_cap_name, x_fit, sample_weight):
    x_fit = np.asarray(x_fit, dtype=np.float64)
    weights = np.asarray(sample_weight, dtype=np.float64)
    weighted_col_mean = weighted_column_mean(x_fit, weights)
    x_centered = x_fit - weighted_col_mean
    x_weighted = x_centered * np.sqrt(weights)[:, None]
    gram = x_weighted.T @ x_weighted
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = np.maximum(np.asarray(eigenvalues, dtype=np.float64), 0.0)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = np.asarray(eigenvectors[:, order], dtype=np.float64)
    components = canonicalize_component_signs(eigenvectors)

    singular_values = np.sqrt(eigenvalues)
    max_sv = float(singular_values[0]) if singular_values.size else 0.0
    positive_sv = singular_values[singular_values > FLOAT_EPS]
    min_positive_sv = float(positive_sv[-1]) if positive_sv.size else 0.0
    if positive_sv.size and min_positive_sv > 0.0:
        condition_number = float(max_sv / min_positive_sv)
    else:
        condition_number = float("inf")

    cond_float32 = float(max(x_fit.shape) * np.finfo(np.float32).eps)
    cond_float64 = float(max(x_fit.shape) * np.finfo(np.float64).eps)
    cutoff_float32 = float(cond_float32 * max_sv)
    cutoff_float64 = float(cond_float64 * max_sv)

    summary_row = {
        "feature_set": feature_set,
        "weighting_scheme": weighting_scheme,
        "row_cap_per_train_protein": row_cap_name,
        "n_train_proteins": int(0),
        "n_train_rows_fit": int(x_fit.shape[0]),
        "feature_dim": int(x_fit.shape[1]),
        "weighted_column_mean_abs_max": float(np.max(np.abs(weighted_col_mean))),
        "weighted_column_mean_abs_mean": float(np.mean(np.abs(weighted_col_mean))),
        "max_singular_value": max_sv,
        "min_positive_singular_value": min_positive_sv,
        "condition_number": condition_number,
        "effective_rank": compute_effective_rank(eigenvalues),
        "cond_float32": cond_float32,
        "cond_float64": cond_float64,
        "cutoff_float32": cutoff_float32,
        "cutoff_float64": cutoff_float64,
        "n_cutoff_float32": int(np.sum(singular_values < cutoff_float32)),
        "n_cutoff_float64": int(np.sum(singular_values < cutoff_float64)),
    }

    singular_rows = []
    for rank_index, singular_value in enumerate(singular_values, start=1):
        singular_rows.append(
            {
                "feature_set": feature_set,
                "weighting_scheme": weighting_scheme,
                "row_cap_per_train_protein": row_cap_name,
                "rank_index": int(rank_index),
                "singular_value": float(singular_value),
                "relative_to_max": float(singular_value / max_sv) if max_sv > 0.0 else 0.0,
                "cut_by_float32": bool(singular_value < cutoff_float32),
                "cut_by_float64": bool(singular_value < cutoff_float64),
            }
        )
    return summary_row, singular_rows, components, weighted_col_mean


def make_variant_meta(
    *,
    variant_id,
    analysis_family,
    feature_set,
    feature_dim,
    fit_feature_dim,
    dtype_name,
    weighting_scheme,
    row_cap_name,
    pca_k="",
    bucket_scheme="",
):
    return {
        "variant_id": variant_id,
        "analysis_family": analysis_family,
        "feature_set": feature_set,
        "feature_dim": int(feature_dim),
        "fit_feature_dim": int(fit_feature_dim),
        "dtype": dtype_name,
        "weighting_scheme": weighting_scheme,
        "row_cap_per_train_protein": row_cap_name,
        "pca_k": "" if pca_k == "" else int(pca_k),
        "bucket_scheme": bucket_scheme,
    }


def make_variant_id(analysis_family, feature_set, dtype_name, weighting_scheme, row_cap_name, pca_k="", bucket_scheme=""):
    pieces = [
        analysis_family,
        feature_set,
        dtype_name,
        weighting_scheme,
        f"cap_{row_cap_name}",
    ]
    if pca_k != "":
        pieces.append(f"pca_{int(pca_k)}")
    if bucket_scheme:
        pieces.append(bucket_scheme)
    return "_".join(pieces)


def coefficient_rows_for_model(meta, feature_names, coef_std, intercept_std, scaler_mean=None, scaler_scale=None, bucket_label="", model_scope="raw_space"):
    coef_std = np.asarray(coef_std, dtype=np.float64)
    intercept_std = float(intercept_std)
    if scaler_mean is not None and scaler_scale is not None:
        coef_raw, intercept_raw = standardized_to_raw_coefficients(
            coef_std,
            intercept_std,
            scaler_mean,
            scaler_scale,
        )
    else:
        coef_raw = np.full(coef_std.shape, np.nan, dtype=np.float64)
        intercept_raw = float("nan")

    rows = []
    for feature_index, feature_name in enumerate(feature_names):
        rows.append(
            {
                "variant_id": meta["variant_id"],
                "analysis_family": meta["analysis_family"],
                "feature_set": meta["feature_set"],
                "feature_dim": int(meta["feature_dim"]),
                "fit_feature_dim": int(meta["fit_feature_dim"]),
                "dtype": meta["dtype"],
                "weighting_scheme": meta["weighting_scheme"],
                "row_cap_per_train_protein": meta["row_cap_per_train_protein"],
                "pca_k": meta["pca_k"],
                "bucket_scheme": meta["bucket_scheme"],
                "bucket_label": bucket_label,
                "model_scope": model_scope,
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "coefficient_standardized": float(coef_std[feature_index]),
                "intercept_standardized": intercept_std,
                "coefficient_raw_scale": float(coef_raw[feature_index]),
                "intercept_raw": intercept_raw,
            }
        )
    return rows


def predict_raw_model(raw_chunk, scaler_mean, scaler_scale, np_dtype, model):
    n_rows = int(raw_chunk.shape[0])
    y_pred = np.empty((n_rows,), dtype=np.float64)
    mean_cast = scaler_mean.astype(np_dtype, copy=False)
    scale_cast = scaler_scale.astype(np_dtype, copy=False)

    for start in range(0, n_rows, PREDICT_BATCH_ROWS):
        end = min(start + PREDICT_BATCH_ROWS, n_rows)
        x_block = raw_chunk[start:end].astype(np_dtype, copy=True)
        x_block -= mean_cast
        x_block /= scale_cast
        y_pred[start:end] = np.asarray(model.predict(x_block), dtype=np.float64)
    return y_pred


def predict_pca_model(raw_chunk, scaler_mean, scaler_scale, pca_center, components_k, model):
    n_rows = int(raw_chunk.shape[0])
    y_pred = np.empty((n_rows,), dtype=np.float64)

    for start in range(0, n_rows, PREDICT_BATCH_ROWS):
        end = min(start + PREDICT_BATCH_ROWS, n_rows)
        x_block = raw_chunk[start:end].astype(np.float64, copy=True)
        x_block -= scaler_mean
        x_block /= scaler_scale
        x_block -= pca_center
        z_block = x_block @ components_k
        y_pred[start:end] = np.asarray(model.predict(z_block), dtype=np.float64)
    return y_pred


def compute_per_protein_row(meta, split_name, record, y_pred, model_bucket_label=""):
    y_true = np.asarray(record["y_abs"], dtype=np.float64)
    if y_pred.size != y_true.size:
        raise ValueError(
            f"Prediction length mismatch for {record['protein_id']}: {y_pred.size} vs {y_true.size}"
        )

    mean_true = float(np.mean(y_true)) if y_true.size else float("nan")
    if y_true.size == 0:
        r2_value = float("nan")
        mean_pred = float("nan")
        mean_error = float("nan")
    else:
        mean_pred = float(np.mean(y_pred))
        mean_error = float(np.mean(y_pred - y_true))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - mean_true) ** 2))
        if ss_tot <= FLOAT_EPS:
            r2_value = float("nan")
        else:
            r2_value = float(1.0 - (ss_res / ss_tot))

    return {
        "variant_id": meta["variant_id"],
        "analysis_family": meta["analysis_family"],
        "feature_set": meta["feature_set"],
        "feature_dim": int(meta["feature_dim"]),
        "fit_feature_dim": int(meta["fit_feature_dim"]),
        "dtype": meta["dtype"],
        "weighting_scheme": meta["weighting_scheme"],
        "row_cap_per_train_protein": meta["row_cap_per_train_protein"],
        "pca_k": meta["pca_k"],
        "bucket_scheme": meta["bucket_scheme"],
        "bucket_label": record["bucket_label"],
        "model_bucket_label": model_bucket_label,
        "split": split_name,
        "protein_id": record["protein_id"],
        "n_residues": int(record["n_residues"]),
        "size_bin": int(record["size_bin"]),
        "n_pairs": int(record["n_pairs"]),
        "target_variance": float(record["target_variance"]),
        "per_protein_r2": r2_value,
        "mean_true_distance": mean_true,
        "mean_predicted_distance": mean_pred,
        "mean_error_pred_minus_true": mean_error,
    }


def aggregate_variant_summary(meta, train_rows, test_rows, trained_model_count, n_train_rows_fit, n_train_rows_eval, n_test_rows_eval, elapsed_seconds):
    def valid_values(rows):
        values = [
            float(row["per_protein_r2"])
            for row in rows
            if not math.isnan(float(row["per_protein_r2"]))
        ]
        return values

    train_values = valid_values(train_rows)
    test_values = valid_values(test_rows)

    negative_count = int(sum(value < 0.0 for value in test_values))
    test_fraction = float(negative_count / len(test_values)) if test_values else float("nan")

    return {
        "variant_id": meta["variant_id"],
        "analysis_family": meta["analysis_family"],
        "feature_set": meta["feature_set"],
        "feature_dim": int(meta["feature_dim"]),
        "fit_feature_dim": int(meta["fit_feature_dim"]),
        "dtype": meta["dtype"],
        "weighting_scheme": meta["weighting_scheme"],
        "row_cap_per_train_protein": meta["row_cap_per_train_protein"],
        "pca_k": meta["pca_k"],
        "bucket_scheme": meta["bucket_scheme"],
        "trained_model_count": int(trained_model_count),
        "n_train_proteins": int(len(train_rows)),
        "n_test_proteins": int(len(test_rows)),
        "n_train_rows_fit": int(n_train_rows_fit),
        "n_train_rows_eval": int(n_train_rows_eval),
        "n_test_rows_eval": int(n_test_rows_eval),
        "train_mean_per_protein_r2": float(np.mean(train_values)) if train_values else float("nan"),
        "test_mean_per_protein_r2": float(np.mean(test_values)) if test_values else float("nan"),
        "test_median_per_protein_r2": float(np.median(test_values)) if test_values else float("nan"),
        "test_negative_r2_count": negative_count,
        "test_negative_r2_fraction": test_fraction,
        "elapsed_seconds": float(elapsed_seconds),
    }


def evaluate_global_raw_variant(meta, model, scaler_mean, scaler_scale, np_dtype, train_records, test_records, chunk_map):
    train_rows = []
    test_rows = []
    for split_name, split_records in [("train", train_records), ("test", test_records)]:
        target_rows = train_rows if split_name == "train" else test_rows
        for record in split_records:
            y_pred = predict_raw_model(
                raw_chunk=chunk_map[record["protein_id"]],
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                np_dtype=np_dtype,
                model=model,
            )
            target_rows.append(compute_per_protein_row(meta, split_name, record, y_pred))
    return train_rows, test_rows


def evaluate_pca_variant(meta, model, scaler_mean, scaler_scale, pca_center, components_k, train_records, test_records, chunk_map):
    train_rows = []
    test_rows = []
    for split_name, split_records in [("train", train_records), ("test", test_records)]:
        target_rows = train_rows if split_name == "train" else test_rows
        for record in split_records:
            y_pred = predict_pca_model(
                raw_chunk=chunk_map[record["protein_id"]],
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                pca_center=pca_center,
                components_k=components_k,
                model=model,
            )
            target_rows.append(compute_per_protein_row(meta, split_name, record, y_pred))
    return train_rows, test_rows


def evaluate_bucketed_variant(meta, bucket_models, train_records, test_records, chunk_map):
    train_rows = []
    test_rows = []
    for split_name, split_records in [("train", train_records), ("test", test_records)]:
        target_rows = train_rows if split_name == "train" else test_rows
        for record in split_records:
            bundle = bucket_models.get(record["bucket_label"])
            if bundle is None:
                y_pred = np.full((record["n_pairs"],), np.nan, dtype=np.float64)
                model_bucket_label = ""
            else:
                y_pred = predict_raw_model(
                    raw_chunk=chunk_map[record["protein_id"]],
                    scaler_mean=bundle["scaler_mean"],
                    scaler_scale=bundle["scaler_scale"],
                    np_dtype=bundle["np_dtype"],
                    model=bundle["model"],
                )
                model_bucket_label = bundle["bucket_label"]
            target_rows.append(
                compute_per_protein_row(
                    meta,
                    split_name,
                    record,
                    y_pred,
                    model_bucket_label=model_bucket_label,
                )
            )
    return train_rows, test_rows


def run_global_variant(
    *,
    feature_set,
    dtype_name,
    np_dtype,
    weighting_scheme,
    row_cap_name,
    train_records,
    test_records,
    chunk_map,
    selection_map,
    feature_names,
):
    t0 = time.time()
    scaler_mean, scaler_scale = compute_scaler(
        train_records=train_records,
        chunk_map=chunk_map,
        selection_map=selection_map,
        weighting_scheme=weighting_scheme,
    )
    x_fit, y_fit = build_fit_matrix(
        train_records=train_records,
        chunk_map=chunk_map,
        selection_map=selection_map,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        np_dtype=np_dtype,
    )
    sample_weight = compute_train_weights(
        train_records=train_records,
        selection_map=selection_map,
        weighting_scheme=weighting_scheme,
    )

    model = LinearRegression(copy_X=False)
    model.fit(x_fit, y_fit, sample_weight=sample_weight)

    meta = make_variant_meta(
        variant_id=make_variant_id(
            ANALYSIS_FAMILY_GLOBAL,
            feature_set,
            dtype_name,
            weighting_scheme,
            row_cap_name,
        ),
        analysis_family=ANALYSIS_FAMILY_GLOBAL,
        feature_set=feature_set,
        feature_dim=feature_dim_for_set(feature_set),
        fit_feature_dim=feature_dim_for_set(feature_set),
        dtype_name=dtype_name,
        weighting_scheme=weighting_scheme,
        row_cap_name=row_cap_name,
    )

    train_rows, test_rows = evaluate_global_raw_variant(
        meta=meta,
        model=model,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        np_dtype=np_dtype,
        train_records=train_records,
        test_records=test_records,
        chunk_map=chunk_map,
    )
    summary_row = aggregate_variant_summary(
        meta=meta,
        train_rows=train_rows,
        test_rows=test_rows,
        trained_model_count=1,
        n_train_rows_fit=x_fit.shape[0],
        n_train_rows_eval=sum(record["n_pairs"] for record in train_records),
        n_test_rows_eval=sum(record["n_pairs"] for record in test_records),
        elapsed_seconds=time.time() - t0,
    )

    coefficient_rows = coefficient_rows_for_model(
        meta=meta,
        feature_names=feature_names,
        coef_std=np.asarray(model.coef_, dtype=np.float64),
        intercept_std=float(model.intercept_),
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        bucket_label="",
        model_scope="raw_space",
    )

    return {
        "summary_row": summary_row,
        "per_protein_rows": train_rows + test_rows,
        "coefficient_rows": coefficient_rows,
        "x_fit": x_fit,
        "y_fit": y_fit,
        "sample_weight": sample_weight,
        "scaler_mean": scaler_mean,
        "scaler_scale": scaler_scale,
        "model": model,
        "meta": meta,
    }


def run_pca_variants(
    *,
    train_records,
    test_records,
    chunk_map,
    scaler_mean,
    scaler_scale,
    x_fit_float64,
    y_fit_float64,
    sample_weight,
    feature_names,
    components,
    pca_center,
):
    rows = []
    coefficient_rows = []
    score_matrix = (x_fit_float64 - pca_center) @ components

    for k in PCA_K_VALUES:
        t0 = time.time()
        components_k = components[:, :k]
        z_fit = score_matrix[:, :k]

        model = LinearRegression(copy_X=False)
        model.fit(z_fit, y_fit_float64, sample_weight=sample_weight)

        meta = make_variant_meta(
            variant_id=make_variant_id(
                ANALYSIS_FAMILY_PCA,
                FEATURE_SET_128,
                "float64",
                WEIGHTING_PROTEIN_EQUAL,
                "all",
                pca_k=k,
            ),
            analysis_family=ANALYSIS_FAMILY_PCA,
            feature_set=FEATURE_SET_128,
            feature_dim=feature_dim_for_set(FEATURE_SET_128),
            fit_feature_dim=int(k),
            dtype_name="float64",
            weighting_scheme=WEIGHTING_PROTEIN_EQUAL,
            row_cap_name="all",
            pca_k=k,
        )

        train_rows, test_rows = evaluate_pca_variant(
            meta=meta,
            model=model,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            pca_center=pca_center,
            components_k=components_k,
            train_records=train_records,
            test_records=test_records,
            chunk_map=chunk_map,
        )
        rows.append(
            aggregate_variant_summary(
                meta=meta,
                train_rows=train_rows,
                test_rows=test_rows,
                trained_model_count=1,
                n_train_rows_fit=x_fit_float64.shape[0],
                n_train_rows_eval=sum(record["n_pairs"] for record in train_records),
                n_test_rows_eval=sum(record["n_pairs"] for record in test_records),
                elapsed_seconds=time.time() - t0,
            )
        )

        coef_pc = np.asarray(model.coef_, dtype=np.float64)
        intercept_pc = float(model.intercept_)
        coefficient_rows.extend(
            coefficient_rows_for_model(
                meta=meta,
                feature_names=[f"pc_{idx}" for idx in range(int(k))],
                coef_std=coef_pc,
                intercept_std=intercept_pc,
                scaler_mean=None,
                scaler_scale=None,
                bucket_label="",
                model_scope="pc_space",
            )
        )

        coef_back = components_k @ coef_pc
        intercept_back = float(intercept_pc - np.dot(pca_center, coef_back))
        coefficient_rows.extend(
            coefficient_rows_for_model(
                meta=meta,
                feature_names=feature_names,
                coef_std=coef_back,
                intercept_std=intercept_back,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                bucket_label="",
                model_scope="back_projected_raw_space",
            )
        )
        for per_row in train_rows + test_rows:
            yield ("per_protein", per_row)

    del score_matrix
    gc.collect()

    for summary_row in rows:
        yield ("summary", summary_row)
    for coef_row in coefficient_rows:
        yield ("coefficient", coef_row)


def run_bucketed_variant(train_records, test_records, chunk_map, feature_names):
    t0 = time.time()
    bucket_models = {}
    coefficient_rows = []
    trained_model_count = 0

    meta = make_variant_meta(
        variant_id=make_variant_id(
            ANALYSIS_FAMILY_BUCKETED,
            FEATURE_SET_128,
            "float64",
            WEIGHTING_PROTEIN_EQUAL,
            "all",
            bucket_scheme=BUCKET_SCHEME_MANUAL,
        ),
        analysis_family=ANALYSIS_FAMILY_BUCKETED,
        feature_set=FEATURE_SET_128,
        feature_dim=feature_dim_for_set(FEATURE_SET_128),
        fit_feature_dim=feature_dim_for_set(FEATURE_SET_128),
        dtype_name="float64",
        weighting_scheme=WEIGHTING_PROTEIN_EQUAL,
        row_cap_name="all",
        bucket_scheme=BUCKET_SCHEME_MANUAL,
    )

    total_fit_rows = 0
    for bucket in MANUAL_BUCKETS:
        bucket_label = bucket["label"]
        bucket_train_records = [
            record for record in train_records if record["bucket_label"] == bucket_label
        ]
        if not bucket_train_records:
            continue

        selection_map = {
            record["protein_id"]: full_range_indices(record["n_pairs"])
            for record in bucket_train_records
        }
        scaler_mean, scaler_scale = compute_scaler(
            train_records=bucket_train_records,
            chunk_map=chunk_map,
            selection_map=selection_map,
            weighting_scheme=WEIGHTING_PROTEIN_EQUAL,
        )
        x_fit, y_fit = build_fit_matrix(
            train_records=bucket_train_records,
            chunk_map=chunk_map,
            selection_map=selection_map,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
            np_dtype=np.float64,
        )
        sample_weight = compute_train_weights(
            train_records=bucket_train_records,
            selection_map=selection_map,
            weighting_scheme=WEIGHTING_PROTEIN_EQUAL,
        )
        model = LinearRegression(copy_X=False)
        model.fit(x_fit, y_fit, sample_weight=sample_weight)

        bucket_models[bucket_label] = {
            "bucket_label": bucket_label,
            "model": model,
            "scaler_mean": scaler_mean,
            "scaler_scale": scaler_scale,
            "np_dtype": np.float64,
        }
        coefficient_rows.extend(
            coefficient_rows_for_model(
                meta=meta,
                feature_names=feature_names,
                coef_std=np.asarray(model.coef_, dtype=np.float64),
                intercept_std=float(model.intercept_),
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                bucket_label=bucket_label,
                model_scope="raw_space",
            )
        )
        trained_model_count += 1
        total_fit_rows += int(x_fit.shape[0])
        del x_fit, y_fit, sample_weight
        gc.collect()

    train_rows, test_rows = evaluate_bucketed_variant(
        meta=meta,
        bucket_models=bucket_models,
        train_records=train_records,
        test_records=test_records,
        chunk_map=chunk_map,
    )
    summary_row = aggregate_variant_summary(
        meta=meta,
        train_rows=train_rows,
        test_rows=test_rows,
        trained_model_count=trained_model_count,
        n_train_rows_fit=total_fit_rows,
        n_train_rows_eval=sum(record["n_pairs"] for record in train_records),
        n_test_rows_eval=sum(record["n_pairs"] for record in test_records),
        elapsed_seconds=time.time() - t0,
    )
    return summary_row, train_rows + test_rows, coefficient_rows


def write_summary_text(path, summary_rows, spectrum_rows):
    summary_lookup = {row["variant_id"]: row for row in summary_rows}

    def lookup(analysis_family, feature_set, dtype_name, weighting_scheme, row_cap_name, pca_k="", bucket_scheme=""):
        return summary_lookup.get(
            make_variant_id(
                analysis_family,
                feature_set,
                dtype_name,
                weighting_scheme,
                row_cap_name,
                pca_k=pca_k,
                bucket_scheme=bucket_scheme,
            )
        )

    with path.open("w") as handle:
        handle.write(f"{TITLE}\n")
        handle.write(f"{'=' * len(TITLE)}\n\n")

        handle.write("Headline comparisons:\n")
        for feature_set in [FEATURE_SET_128, FEATURE_SET_256]:
            base32 = lookup(
                ANALYSIS_FAMILY_GLOBAL,
                feature_set,
                "float32",
                WEIGHTING_PROTEIN_EQUAL,
                "all",
            )
            base64 = lookup(
                ANALYSIS_FAMILY_GLOBAL,
                feature_set,
                "float64",
                WEIGHTING_PROTEIN_EQUAL,
                "all",
            )
            if base32 is not None and base64 is not None:
                handle.write(
                    f"- {feature_set} protein-equal all rows: "
                    f"float32 test mean per-protein R2={float(base32['test_mean_per_protein_r2']):.4f}, "
                    f"float64 test mean per-protein R2={float(base64['test_mean_per_protein_r2']):.4f}\n"
                )

        pair_equal = lookup(
            ANALYSIS_FAMILY_GLOBAL,
            FEATURE_SET_256,
            "float64",
            WEIGHTING_PAIR_EQUAL,
            "all",
        )
        if pair_equal is not None:
            handle.write(
                f"- 256d float64 pair-equal all rows: "
                f"test mean per-protein R2={float(pair_equal['test_mean_per_protein_r2']):.4f}\n"
            )

        bucketed = lookup(
            ANALYSIS_FAMILY_BUCKETED,
            FEATURE_SET_128,
            "float64",
            WEIGHTING_PROTEIN_EQUAL,
            "all",
            bucket_scheme=BUCKET_SCHEME_MANUAL,
        )
        global_128d = lookup(
            ANALYSIS_FAMILY_GLOBAL,
            FEATURE_SET_128,
            "float64",
            WEIGHTING_PROTEIN_EQUAL,
            "all",
        )
        if bucketed is not None and global_128d is not None:
            handle.write(
                f"- 128d float64 bucketed vs global all rows: "
                f"bucketed={float(bucketed['test_mean_per_protein_r2']):.4f}, "
                f"global={float(global_128d['test_mean_per_protein_r2']):.4f}\n"
            )

        handle.write("\nPCA sweep:\n")
        for k in PCA_K_VALUES:
            row = lookup(
                ANALYSIS_FAMILY_PCA,
                FEATURE_SET_128,
                "float64",
                WEIGHTING_PROTEIN_EQUAL,
                "all",
                pca_k=k,
            )
            if row is None:
                continue
            handle.write(
                f"- k={k}: test mean per-protein R2="
                f"{float(row['test_mean_per_protein_r2']):.4f}\n"
            )

        handle.write("\nSpectrum cutoffs:\n")
        for row in spectrum_rows:
            handle.write(
                f"- {row['feature_set']} {row['weighting_scheme']} cap={row['row_cap_per_train_protein']}: "
                f"n_cutoff_float32={int(row['n_cutoff_float32'])}, "
                f"n_cutoff_float64={int(row['n_cutoff_float64'])}, "
                f"effective_rank={float(row['effective_rank']):.2f}\n"
            )


def main():
    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))

    all_proteins, protein_data, skipped = load_protein_records()
    del all_proteins
    if not protein_data:
        print("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    records = prepare_records(protein_data)
    assign_size_bins(records, N_SIZE_BINS)
    train_records, test_records = split_records(records)
    split_rows = build_split_rows(train_records, test_records)
    selection_maps, _ = build_train_selection_maps(train_records)

    summary_rows = []
    per_protein_rows = []
    coefficient_rows = []
    spectrum_summary_rows = []
    singular_value_rows = []

    for feature_set in [FEATURE_SET_128, FEATURE_SET_256]:
        print(f"\nLoading {feature_set} feature chunks...", flush=True)
        chunk_map = build_feature_chunk_map(records, feature_set)
        feature_names = feature_names_for_set(feature_set)

        for row_cap in ROW_CAP_OPTIONS:
            row_cap_name = row_cap_label(row_cap)
            selection_map = selection_maps[row_cap_name]

            for dtype_name, np_dtype in [("float32", np.float32), ("float64", np.float64)]:
                print(
                    f"Running global variant: feature_set={feature_set}, "
                    f"dtype={dtype_name}, weighting={WEIGHTING_PROTEIN_EQUAL}, "
                    f"row_cap={row_cap_name}",
                    flush=True,
                )
                result = run_global_variant(
                    feature_set=feature_set,
                    dtype_name=dtype_name,
                    np_dtype=np_dtype,
                    weighting_scheme=WEIGHTING_PROTEIN_EQUAL,
                    row_cap_name=row_cap_name,
                    train_records=train_records,
                    test_records=test_records,
                    chunk_map=chunk_map,
                    selection_map=selection_map,
                    feature_names=feature_names,
                )
                summary_rows.append(result["summary_row"])
                per_protein_rows.extend(result["per_protein_rows"])
                coefficient_rows.extend(result["coefficient_rows"])

                if dtype_name == "float64":
                    spectrum_row, full_singular_rows, components, pca_center = compute_spectrum_diagnostics(
                        feature_set=feature_set,
                        weighting_scheme=WEIGHTING_PROTEIN_EQUAL,
                        row_cap_name=row_cap_name,
                        x_fit=result["x_fit"],
                        sample_weight=result["sample_weight"],
                    )
                    spectrum_row["n_train_proteins"] = int(len(train_records))
                    spectrum_summary_rows.append(spectrum_row)
                    if row_cap_name == "all":
                        singular_value_rows.extend(full_singular_rows)

                    if feature_set == FEATURE_SET_128 and row_cap_name == "all":
                        print("Running weighted-PCA variants for 128d float64 all rows...", flush=True)
                        for row_type, row in run_pca_variants(
                            train_records=train_records,
                            test_records=test_records,
                            chunk_map=chunk_map,
                            scaler_mean=result["scaler_mean"],
                            scaler_scale=result["scaler_scale"],
                            x_fit_float64=np.asarray(result["x_fit"], dtype=np.float64),
                            y_fit_float64=np.asarray(result["y_fit"], dtype=np.float64),
                            sample_weight=np.asarray(result["sample_weight"], dtype=np.float64),
                            feature_names=feature_names,
                            components=components,
                            pca_center=pca_center,
                        ):
                            if row_type == "summary":
                                summary_rows.append(row)
                            elif row_type == "coefficient":
                                coefficient_rows.append(row)
                            else:
                                per_protein_rows.append(row)

                del result["x_fit"], result["y_fit"], result["sample_weight"]
                gc.collect()

            if feature_set == FEATURE_SET_256 and row_cap_name == "all":
                print(
                    "Running extra global variant: feature_set=256d, "
                    "dtype=float64, weighting=pair_equal, row_cap=all",
                    flush=True,
                )
                result = run_global_variant(
                    feature_set=feature_set,
                    dtype_name="float64",
                    np_dtype=np.float64,
                    weighting_scheme=WEIGHTING_PAIR_EQUAL,
                    row_cap_name=row_cap_name,
                    train_records=train_records,
                    test_records=test_records,
                    chunk_map=chunk_map,
                    selection_map=selection_map,
                    feature_names=feature_names,
                )
                summary_rows.append(result["summary_row"])
                per_protein_rows.extend(result["per_protein_rows"])
                coefficient_rows.extend(result["coefficient_rows"])
                del result["x_fit"], result["y_fit"], result["sample_weight"]
                gc.collect()

        if feature_set == FEATURE_SET_128:
            print("Running manual fixed-bucket 128d float64 variant...", flush=True)
            bucket_summary, bucket_per_rows, bucket_coef_rows = run_bucketed_variant(
                train_records=train_records,
                test_records=test_records,
                chunk_map=chunk_map,
                feature_names=feature_names,
            )
            summary_rows.append(bucket_summary)
            per_protein_rows.extend(bucket_per_rows)
            coefficient_rows.extend(bucket_coef_rows)

        del chunk_map
        gc.collect()

    if skipped:
        print(f"Skipped proteins: {len(skipped)}", flush=True)

    write_csv(csv_dir / SUMMARY_CSV_NAME, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(csv_dir / PER_PROTEIN_CSV_NAME, PER_PROTEIN_FIELDNAMES, per_protein_rows)
    write_csv(csv_dir / SPLIT_CSV_NAME, SPLIT_FIELDNAMES, split_rows)
    write_csv(
        csv_dir / SPECTRUM_SUMMARY_CSV_NAME,
        SPECTRUM_SUMMARY_FIELDNAMES,
        spectrum_summary_rows,
    )
    write_csv(
        csv_dir / SINGULAR_VALUES_CSV_NAME,
        SINGULAR_VALUES_FIELDNAMES,
        singular_value_rows,
    )
    write_csv(csv_dir / COEFFICIENT_CSV_NAME, COEFFICIENT_FIELDNAMES, coefficient_rows)
    write_summary_text(csv_dir / SUMMARY_TXT_NAME, summary_rows, spectrum_summary_rows)

    print("\nDone.")
    print(f"Wrote {len(summary_rows)} summary rows", flush=True)


if __name__ == "__main__":
    main()

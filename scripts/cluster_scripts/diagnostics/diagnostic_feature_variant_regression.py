#!/usr/bin/env python3
"""
diagnostic_feature_variant_regression.py
=======================================
Layer-47 diagnostic regression script for lower-triangle linear feature
variants with per-protein geometry analysis.

Requested variants (all with protein-equal training weights):
1. baseline_cap_1k
2. baseline_cap_2p5k
3. baseline_cap_5k
4. baseline_cap_10k
5. baseline_cap_20k
6. interaction_raw_n_cap_5k
7. baseline_cap_5k_predfloor_3p3A
8. baseline_plus_axial_cap_5k
9. baseline_plus_axial_diag_cap_5k
10. baseline_plus_seqsep_eq1_le2_cap_5k

Outputs (inside CSV_ROOT/):
    diagnostic_feature_variant_regression_variant_summary.csv
    diagnostic_feature_variant_regression_per_protein.csv
    diagnostic_feature_variant_regression_split.csv
    diagnostic_feature_variant_regression_geometry_metrics.csv
    diagnostic_feature_variant_regression_geometry_summary.csv
    diagnostic_feature_variant_regression_seq_sep_diagnostics.csv
    diagnostic_feature_variant_regression_size_bias_summary.csv
    diagnostic_feature_variant_regression_test_scatter_sample.csv
    diagnostic_feature_variant_regression_summary.txt
"""

import math
import os
import sys
import tempfile
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(CLUSTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_SCRIPTS_DIR))

from diagnostic_tree_regression_common import (  # noqa: E402
    CSV_DIR,
    FLOAT_EPS,
    LAYER,
    N_CHANNELS,
    RANDOM_STATE,
    TEST_SIZE,
    build_seq_sep_diagnostics,
    compute_mae,
    compute_distance_matrix,
    compute_pair_weights,
    compute_r2,
    compute_rmse,
    deterministic_sample_indices,
    extract_ca_coordinates,
    find_pdb_file,
    get_pair_block_path,
    load_protein_records,
    write_csv,
)


TITLE = "Diagnostic Feature Variant Regression — Layer 47"
PREDICTION_FLOOR_ANGSTROM = 3.3
N_SIZE_BINS = 5
SCATTER_SAMPLE_SIZE = 1500

SUMMARY_CSV_NAME = "diagnostic_feature_variant_regression_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_feature_variant_regression_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_feature_variant_regression_split.csv"
GEOMETRY_METRICS_CSV_NAME = "diagnostic_feature_variant_regression_geometry_metrics.csv"
GEOMETRY_SUMMARY_CSV_NAME = "diagnostic_feature_variant_regression_geometry_summary.csv"
SEQ_DIAGNOSTICS_CSV_NAME = "diagnostic_feature_variant_regression_seq_sep_diagnostics.csv"
SIZE_BIAS_CSV_NAME = "diagnostic_feature_variant_regression_size_bias_summary.csv"
SCATTER_CSV_NAME = "diagnostic_feature_variant_regression_test_scatter_sample.csv"
SUMMARY_TXT_NAME = "diagnostic_feature_variant_regression_summary.txt"

CENTER_OFFSETS = [(0, 0)]
AXIAL_OFFSETS = [(0, 0), (-1, 0), (1, 0), (0, -1), (0, 1)]
AXIAL_DIAG_OFFSETS = AXIAL_OFFSETS + [(-1, -1), (-1, 1), (1, -1), (1, 1)]

FEATURE_MODE_BASE = "base"
FEATURE_MODE_INTERACTION = "interaction"
FEATURE_MODE_AXIAL = "axial"
FEATURE_MODE_AXIAL_DIAG = "axial_diag"
FEATURE_MODE_SEQSEP = "seqsep"

VARIANTS = [
    {
        "name": "baseline_cap_1k",
        "feature_mode": FEATURE_MODE_BASE,
        "sample_cap": 1000,
        "prediction_floor": None,
    },
    {
        "name": "baseline_cap_2p5k",
        "feature_mode": FEATURE_MODE_BASE,
        "sample_cap": 2500,
        "prediction_floor": None,
    },
    {
        "name": "baseline_cap_5k",
        "feature_mode": FEATURE_MODE_BASE,
        "sample_cap": 5000,
        "prediction_floor": None,
    },
    {
        "name": "baseline_cap_10k",
        "feature_mode": FEATURE_MODE_BASE,
        "sample_cap": 10000,
        "prediction_floor": None,
    },
    {
        "name": "baseline_cap_20k",
        "feature_mode": FEATURE_MODE_BASE,
        "sample_cap": 20000,
        "prediction_floor": None,
    },
    {
        "name": "interaction_raw_n_cap_5k",
        "feature_mode": FEATURE_MODE_INTERACTION,
        "sample_cap": 5000,
        "prediction_floor": None,
    },
    {
        "name": "baseline_cap_5k_predfloor_3p3A",
        "feature_mode": FEATURE_MODE_BASE,
        "sample_cap": 5000,
        "prediction_floor": PREDICTION_FLOOR_ANGSTROM,
    },
    {
        "name": "baseline_plus_axial_cap_5k",
        "feature_mode": FEATURE_MODE_AXIAL,
        "sample_cap": 5000,
        "prediction_floor": None,
    },
    {
        "name": "baseline_plus_axial_diag_cap_5k",
        "feature_mode": FEATURE_MODE_AXIAL_DIAG,
        "sample_cap": 5000,
        "prediction_floor": None,
    },
    {
        "name": "baseline_plus_seqsep_eq1_le2_cap_5k",
        "feature_mode": FEATURE_MODE_SEQSEP,
        "sample_cap": 5000,
        "prediction_floor": None,
    },
]

SUMMARY_FIELDNAMES = [
    "variant",
    "feature_mode",
    "sample_cap_per_train_protein",
    "prediction_floor_angstrom",
    "n_train_proteins",
    "n_test_proteins",
    "n_train_rows_fit",
    "n_train_rows_eval",
    "n_test_rows_eval",
    "train_mean_per_protein_r2",
    "test_mean_per_protein_r2",
    "train_pair_weighted_r2",
    "test_pair_weighted_r2",
    "train_pair_weighted_mae",
    "test_pair_weighted_mae",
    "train_pair_weighted_rmse",
    "test_pair_weighted_rmse",
    "train_mean_error_pred_minus_true",
    "test_mean_error_pred_minus_true",
    "train_tm_score_mean",
    "test_tm_score_mean",
    "train_rmsd_mean",
    "test_rmsd_mean",
    "train_stress_mean",
    "test_stress_mean",
    "train_triangle_violation_fraction_mean",
    "test_triangle_violation_fraction_mean",
    "elapsed_seconds",
]

PER_PROTEIN_FIELDNAMES = [
    "variant",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "per_protein_r2",
    "mae",
    "rmse",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
    "median_error_pred_minus_true",
    "std_error_pred_minus_true",
    "overestimate_fraction",
    "underestimate_fraction",
    "mean_overestimate",
    "mean_underestimate",
    "mean_pred_true_ratio",
]

SPLIT_FIELDNAMES = [
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
]

GEOMETRY_METRICS_FIELDNAMES = [
    "variant",
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

GEOMETRY_SUMMARY_FIELDNAMES = [
    "variant",
    "split",
    "n_proteins",
    "n_failed_geometry",
    "per_protein_r2_mean",
    "per_protein_r2_std",
    "tm_score_mean",
    "tm_score_std",
    "rmsd_mean",
    "rmsd_std",
    "stress_mean",
    "stress_std",
    "frac_top3_eigenvalues_mean",
    "frac_top3_eigenvalues_std",
    "n_negative_eigenvalues_mean",
    "n_negative_eigenvalues_std",
    "neg_eigenvalue_fraction_mean",
    "neg_eigenvalue_fraction_std",
    "n_negative_predictions_mean",
    "n_negative_predictions_std",
    "triangle_violation_fraction_mean",
    "triangle_violation_fraction_std",
    "triangle_mean_violation_mean",
    "triangle_mean_violation_std",
    "triangle_max_violation_mean",
    "triangle_max_violation_std",
]

SEQ_DIAGNOSTICS_FIELDNAMES = [
    "variant",
    "split",
    "seq_bucket",
    "n_pairs",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
    "r2",
    "fraction_pred_lt_3A",
]

SIZE_BIAS_FIELDNAMES = [
    "variant",
    "split",
    "size_bin",
    "n_proteins",
    "mean_n_residues",
    "mean_per_protein_r2",
    "mean_error_pred_minus_true",
    "median_error_pred_minus_true",
    "mean_overestimate_fraction",
    "mean_underestimate_fraction",
    "mean_overestimate",
    "mean_underestimate",
]

SCATTER_FIELDNAMES = [
    "variant",
    "protein_id",
    "n_residues",
    "size_bin",
    "seq_sep",
    "true_distance",
    "predicted_distance",
    "error_pred_minus_true",
]


def classical_mds(distance_matrix, n_dims=3):
    """Classical (Torgerson) MDS for distance-matrix reconstruction."""
    d_sq = distance_matrix ** 2
    row_mean = d_sq.mean(axis=1, keepdims=True)
    col_mean = d_sq.mean(axis=0, keepdims=True)
    grand_mean = d_sq.mean()
    gram = -0.5 * (d_sq - row_mean - col_mean + grand_mean)

    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[order]
    eigenvectors = eigenvectors[:, order]

    top_evals = np.maximum(eigenvalues[:n_dims], 0.0)
    coords = eigenvectors[:, :n_dims] * np.sqrt(top_evals)
    return coords, eigenvalues


def superimpose_structures(fixed_coords, moving_coords):
    """Kabsch alignment for RMSD / TM-score geometry summaries."""
    fixed_center = fixed_coords.mean(axis=0)
    moving_center = moving_coords.mean(axis=0)
    fixed_centered = fixed_coords - fixed_center
    moving_centered = moving_coords - moving_center

    covariance = moving_centered.T @ fixed_centered
    u_mat, _, vt_mat = np.linalg.svd(covariance)
    det_sign = np.linalg.det(vt_mat.T @ u_mat.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(det_sign)])
    rotation = vt_mat.T @ sign_matrix @ u_mat.T

    transformed = (moving_centered @ rotation.T) + fixed_center
    diff = fixed_coords - transformed
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return transformed, rmsd


def compute_tm_score_local(fixed_coords, transformed_coords, length_target):
    """Compute TM-score after superposition."""
    d0 = 1.24 * (max(length_target, 16) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)
    distances = np.sqrt(np.sum((fixed_coords - transformed_coords) ** 2, axis=1))
    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    return float(np.sum(scores) / length_target)


def compute_triangle_violations(pred_dist):
    """Compute triangle-inequality violations using shortest two-hop paths."""
    n_residues = pred_dist.shape[0]
    min_indirect = np.full((n_residues, n_residues), np.inf)
    for jdx in range(n_residues):
        candidate = pred_dist[:, jdx:jdx + 1] + pred_dist[jdx:jdx + 1, :]
        min_indirect = np.minimum(min_indirect, candidate)
    np.fill_diagonal(min_indirect, 0.0)

    mask = pred_dist > min_indirect
    np.fill_diagonal(mask, False)
    n_violations = int(mask.sum())
    total_pairs = n_residues * (n_residues - 1)
    violation_fraction = float(n_violations / total_pairs) if total_pairs > 0 else 0.0

    if n_violations > 0:
        violations = pred_dist[mask] - min_indirect[mask]
        mean_violation = float(np.mean(violations))
        max_violation = float(np.max(violations))
    else:
        mean_violation = 0.0
        max_violation = 0.0

    return violation_fraction, mean_violation, max_violation


def analyze_protein_geometry_local(n_residues, true_coords, pair_i, pair_j, y_pred):
    """Run MDS reconstruction + geometry diagnostics for one protein."""
    pred_dist = np.zeros((n_residues, n_residues), dtype=np.float64)
    pred_dist[pair_i, pair_j] = y_pred
    pred_dist[pair_j, pair_i] = y_pred

    n_negative_preds = int(np.sum(y_pred < 0))
    pred_dist = np.maximum(pred_dist, 0.0)

    mds_coords, eigenvalues = classical_mds(pred_dist, n_dims=3)
    positive_mask = eigenvalues > 0
    positive_sum = np.sum(eigenvalues[positive_mask]) if np.any(positive_mask) else 1.0
    frac_top3 = float(np.sum(np.maximum(eigenvalues[:3], 0.0)) / positive_sum)
    n_negative_evals = int(np.sum(eigenvalues < 0))
    negative_fraction = (
        float(abs(np.sum(eigenvalues[eigenvalues < 0])) / positive_sum)
        if n_negative_evals > 0
        else 0.0
    )

    reflected_coords = mds_coords.copy()
    reflected_coords[:, 2] *= -1
    transformed_orig, rmsd_orig = superimpose_structures(true_coords, mds_coords)
    transformed_flip, rmsd_flip = superimpose_structures(true_coords, reflected_coords)
    if rmsd_flip < rmsd_orig:
        transformed, rmsd = transformed_flip, rmsd_flip
    else:
        transformed, rmsd = transformed_orig, rmsd_orig

    tm_score = compute_tm_score_local(true_coords, transformed, n_residues)
    reconstructed_dist = compute_distance_matrix(transformed)
    triu_i, triu_j = np.triu_indices(n_residues, k=1)
    d_pred = pred_dist[triu_i, triu_j]
    d_mds = reconstructed_dist[triu_i, triu_j]
    stress = float(np.sqrt(np.sum((d_pred - d_mds) ** 2) / np.sum(d_pred ** 2)))

    violation_fraction, mean_violation, max_violation = compute_triangle_violations(pred_dist)
    return {
        "rmsd": float(rmsd),
        "tm_score": float(tm_score),
        "frac_top3_eigenvalues": frac_top3,
        "n_negative_eigenvalues": n_negative_evals,
        "neg_eigenvalue_fraction": negative_fraction,
        "n_negative_predictions": n_negative_preds,
        "stress": stress,
        "triangle_violation_fraction": violation_fraction,
        "triangle_mean_violation": mean_violation,
        "triangle_max_violation": max_violation,
    }


def assign_rank_size_bins(records, n_bins):
    """Assign stable rank-based size bins across all loaded proteins."""
    ordered = sorted(records, key=lambda row: (row["n_residues"], row["protein_id"]))
    total = len(ordered)
    for rank, record in enumerate(ordered):
        size_bin = int((rank * n_bins) // max(total, 1))
        if size_bin >= n_bins:
            size_bin = n_bins - 1
        record["size_bin"] = int(size_bin)


def split_global_records(records):
    """Create a single deterministic protein-level train/test split."""
    if len(records) < 2:
        raise ValueError("Need at least two proteins to split into train/test.")

    indices = np.arange(len(records), dtype=np.int64)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_set = set(int(idx) for idx in train_idx)
    test_set = set(int(idx) for idx in test_idx)
    train_records = [record for idx, record in enumerate(records) if idx in train_set]
    test_records = [record for idx, record in enumerate(records) if idx in test_set]
    return train_records, test_records


def feature_dim_for_variant(variant):
    """Return dense feature dimension for one variant."""
    mode = variant["feature_mode"]
    if mode == FEATURE_MODE_BASE:
        return N_CHANNELS
    if mode == FEATURE_MODE_INTERACTION:
        return 2 * N_CHANNELS
    if mode == FEATURE_MODE_AXIAL:
        return len(AXIAL_OFFSETS) * N_CHANNELS
    if mode == FEATURE_MODE_AXIAL_DIAG:
        return len(AXIAL_DIAG_OFFSETS) * N_CHANNELS
    if mode == FEATURE_MODE_SEQSEP:
        return N_CHANNELS + 3
    raise ValueError(f"Unsupported feature_mode: {mode}")


def prepare_records(records):
    """Augment helper-loaded records with paths and CA coordinates."""
    prepared = []
    for record in records:
        pdb_path = find_pdb_file(record["protein_id"])
        if pdb_path is None:
            raise ValueError(f"Missing PDB path for loaded protein {record['protein_id']}")
        ca_data = extract_ca_coordinates(pdb_path)
        coordinates = np.asarray(ca_data["coordinates"], dtype=np.float32)
        if coordinates.shape[0] != int(record["n_residues"]):
            raise ValueError(
                f"CA count mismatch for {record['protein_id']}: "
                f"{coordinates.shape[0]} != {record['n_residues']}"
            )
        prepared_record = dict(record)
        prepared_record["pair_i"] = np.asarray(record["lt_i"], dtype=np.int32)
        prepared_record["pair_j"] = np.asarray(record["lt_j"], dtype=np.int32)
        prepared_record["coordinates"] = coordinates
        prepared_record["pb_path"] = get_pair_block_path(
            record["protein_dir"],
            record["protein_id"],
            LAYER,
        )
        prepared.append(prepared_record)
    return prepared


def sample_indices_for_record(record, sample_cap):
    """Return a deterministic row selection for one protein."""
    return deterministic_sample_indices(
        label=f"{record['protein_id']}:{record['n_pairs']}",
        n_rows=int(record["n_pairs"]),
        cap=sample_cap,
        offset=0,
    )


def _fill_context_features(out_chunk, pb, pair_i, pair_j, offsets):
    """Write zero-padded lower-triangle context slots into a dense feature block."""
    write_col = 0
    n_residues = pb.shape[0]
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
            out_chunk[valid, write_col:write_col + N_CHANNELS] = pb[src_i[valid], src_j[valid], :]
        write_col += N_CHANNELS


def build_feature_chunk(record, variant, local_idx):
    """Build one dense feature chunk for a selected subset of one protein."""
    pb = np.load(str(record["pb_path"]), mmap_mode="r")
    pair_i = record["pair_i"][local_idx]
    pair_j = record["pair_j"][local_idx]
    seq_sep = record["seq_sep"][local_idx]

    mode = variant["feature_mode"]
    if mode == FEATURE_MODE_BASE:
        chunk = np.asarray(pb[pair_i, pair_j, :], dtype=np.float32)
    elif mode == FEATURE_MODE_INTERACTION:
        base = np.asarray(pb[pair_i, pair_j, :], dtype=np.float32)
        chunk = np.empty((base.shape[0], 2 * N_CHANNELS), dtype=np.float32)
        chunk[:, :N_CHANNELS] = base
        chunk[:, N_CHANNELS:] = base * float(record["n_residues"])
    elif mode == FEATURE_MODE_AXIAL:
        chunk = np.zeros((local_idx.size, len(AXIAL_OFFSETS) * N_CHANNELS), dtype=np.float32)
        _fill_context_features(chunk, pb, pair_i, pair_j, AXIAL_OFFSETS)
    elif mode == FEATURE_MODE_AXIAL_DIAG:
        chunk = np.zeros((local_idx.size, len(AXIAL_DIAG_OFFSETS) * N_CHANNELS), dtype=np.float32)
        _fill_context_features(chunk, pb, pair_i, pair_j, AXIAL_DIAG_OFFSETS)
    elif mode == FEATURE_MODE_SEQSEP:
        base = np.asarray(pb[pair_i, pair_j, :], dtype=np.float32)
        chunk = np.empty((base.shape[0], N_CHANNELS + 3), dtype=np.float32)
        chunk[:, :N_CHANNELS] = base
        chunk[:, N_CHANNELS] = seq_sep
        chunk[:, N_CHANNELS + 1] = (seq_sep == 1.0).astype(np.float32, copy=False)
        chunk[:, N_CHANNELS + 2] = (seq_sep <= 2.0).astype(np.float32, copy=False)
    else:
        raise ValueError(f"Unsupported feature_mode: {mode}")

    del pb
    return chunk


def build_bundle(records, variant, sample_cap):
    """Materialize one split into a dense feature/target bundle."""
    row_counts = []
    selections = []
    total_rows = 0

    for record in records:
        local_idx = sample_indices_for_record(record, sample_cap)
        selections.append(local_idx)
        row_counts.append(int(local_idx.size))
        total_rows += int(local_idx.size)

    feature_dim = feature_dim_for_variant(variant)
    x = np.empty((total_rows, feature_dim), dtype=np.float32)
    y = np.empty((total_rows,), dtype=np.float64)
    seq_sep_rows = np.empty((total_rows,), dtype=np.int32)
    offsets = []

    start = 0
    for record, local_idx, n_rows in zip(records, selections, row_counts):
        end = start + n_rows
        x[start:end, :] = build_feature_chunk(record, variant, local_idx)
        y[start:end] = record["y_abs"][local_idx]
        seq_sep_rows[start:end] = record["seq_sep"][local_idx].astype(np.int32, copy=False)
        offsets.append((start, end))
        start = end

    return {
        "x": x,
        "y": y,
        "seq_sep": seq_sep_rows,
        "offsets": offsets,
        "row_counts": list(row_counts),
        "weights": compute_pair_weights(row_counts),
    }


def compute_standardization_from_bundle(bundle):
    """Compute weighted train-only feature moments from the fit bundle."""
    feature_dim = bundle["x"].shape[1]
    mean_acc = np.zeros((feature_dim,), dtype=np.float64)
    sq_acc = np.zeros((feature_dim,), dtype=np.float64)
    total_weight = 0.0

    for (start, end), n_rows in zip(bundle["offsets"], bundle["row_counts"]):
        block = np.asarray(bundle["x"][start:end, :], dtype=np.float64)
        block_mean = np.mean(block, axis=0, dtype=np.float64)
        block_sq_mean = np.mean(block * block, axis=0, dtype=np.float64)
        mean_acc += block_mean
        sq_acc += block_sq_mean
        total_weight += 1.0

    mean = mean_acc / max(total_weight, 1.0)
    second_moment = sq_acc / max(total_weight, 1.0)
    variance = np.maximum(second_moment - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale[scale < FLOAT_EPS] = 1.0
    return {
        "mean": mean.astype(np.float32, copy=False),
        "scale": scale.astype(np.float32, copy=False),
    }


def apply_standardization_in_place(bundle, standardization):
    """Apply a train-fitted standardization transform to a bundle."""
    bundle["x"] -= standardization["mean"]
    bundle["x"] /= standardization["scale"]


def summarize_predictions_extended(split_name, records, offsets, y_true, y_pred):
    """Summarize pooled and per-protein regression metrics plus signed-error bias."""
    per_rows = []
    per_r2 = []
    negative_r2 = 0

    for record, (start, end) in zip(records, offsets):
        y_true_p = np.asarray(y_true[start:end], dtype=np.float64)
        y_pred_p = np.asarray(y_pred[start:end], dtype=np.float64)
        error = y_pred_p - y_true_p
        over_mask = error > 0.0
        under_mask = error < 0.0

        r2_value = compute_r2(y_true_p, y_pred_p)
        if not math.isnan(r2_value):
            per_r2.append(r2_value)
            if r2_value < 0.0:
                negative_r2 += 1

        per_rows.append(
            {
                "split": split_name,
                "protein_id": record["protein_id"],
                "n_residues": int(record["n_residues"]),
                "size_bin": int(record["size_bin"]),
                "n_pairs": int(end - start),
                "per_protein_r2": r2_value,
                "mae": compute_mae(y_true_p, y_pred_p),
                "rmse": compute_rmse(y_true_p, y_pred_p),
                "mean_true_distance": float(np.mean(y_true_p)),
                "mean_predicted_distance": float(np.mean(y_pred_p)),
                "mean_error_pred_minus_true": float(np.mean(error)),
                "median_error_pred_minus_true": float(np.median(error)),
                "std_error_pred_minus_true": float(np.std(error)),
                "overestimate_fraction": float(np.mean(over_mask)),
                "underestimate_fraction": float(np.mean(under_mask)),
                "mean_overestimate": (
                    float(np.mean(error[over_mask])) if np.any(over_mask) else 0.0
                ),
                "mean_underestimate": (
                    float(np.mean(-error[under_mask])) if np.any(under_mask) else 0.0
                ),
                "mean_pred_true_ratio": float(
                    np.mean(y_pred_p / np.maximum(y_true_p, FLOAT_EPS))
                ),
            }
        )

    return {
        "mean_per_protein_r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
        "median_per_protein_r2": float(np.median(per_r2)) if per_r2 else float("nan"),
        "negative_r2_count": int(negative_r2),
        "negative_r2_fraction": (
            float(negative_r2 / len(per_r2)) if per_r2 else float("nan")
        ),
        "pair_weighted_r2": compute_r2(y_true, y_pred),
        "pair_weighted_mae": compute_mae(y_true, y_pred),
        "pair_weighted_rmse": compute_rmse(y_true, y_pred),
        "pair_weighted_mean_error_pred_minus_true": float(np.mean(y_pred - y_true)),
        "per_rows": per_rows,
    }


def make_failed_geometry_row(variant_name, split_name, record, per_protein_r2):
    """Return a placeholder geometry row when reconstruction fails."""
    return {
        "variant": variant_name,
        "split": split_name,
        "protein_id": record["protein_id"],
        "n_residues": int(record["n_residues"]),
        "size_bin": int(record["size_bin"]),
        "n_pairs": int(record["n_pairs"]),
        "per_protein_r2": per_protein_r2,
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


def collect_geometry_rows(variant_name, split_name, split_records, y_true, y_pred):
    """Run per-protein geometry analysis for one split."""
    rows = []
    start = 0
    for record in split_records:
        end = start + int(record["n_pairs"])
        per_protein_r2 = compute_r2(y_true[start:end], y_pred[start:end])
        try:
            metrics = analyze_protein_geometry(
                int(record["n_residues"]),
                record["coordinates"],
                record["pair_i"],
                record["pair_j"],
                y_pred[start:end],
            )
            rows.append(
                {
                    "variant": variant_name,
                    "split": split_name,
                    "protein_id": record["protein_id"],
                    "n_residues": int(record["n_residues"]),
                    "size_bin": int(record["size_bin"]),
                    "n_pairs": int(record["n_pairs"]),
                    "per_protein_r2": per_protein_r2,
                    **metrics,
                }
            )
        except Exception:
            traceback.print_exc()
            rows.append(make_failed_geometry_row(variant_name, split_name, record, per_protein_r2))
        start = end
    return rows


def analyze_protein_geometry(n_residues, true_coords, pair_i, pair_j, y_pred):
    """Local alias kept so the rest of the script reads naturally."""
    return analyze_protein_geometry_local(n_residues, true_coords, pair_i, pair_j, y_pred)


def summarize_geometry_rows(geometry_rows):
    """Aggregate geometry metrics per variant and split."""
    metric_keys = [
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

    grouped = {}
    for row in geometry_rows:
        key = (row["variant"], row["split"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (variant_name, split_name), rows in sorted(grouped.items()):
        out = {
            "variant": variant_name,
            "split": split_name,
            "n_proteins": int(len(rows)),
            "n_failed_geometry": int(
                sum(math.isnan(float(row["tm_score"])) for row in rows)
            ),
        }
        for metric_key in metric_keys:
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


def build_size_bias_summary(per_protein_rows):
    """Aggregate bias statistics across rank-based size bins."""
    grouped = {}
    for row in per_protein_rows:
        key = (row["variant"], row["split"], int(row["size_bin"]))
        grouped.setdefault(key, []).append(row)

    out_rows = []
    for (variant_name, split_name, size_bin), rows in sorted(grouped.items()):
        r2_values = [
            float(row["per_protein_r2"])
            for row in rows
            if not math.isnan(float(row["per_protein_r2"]))
        ]
        out_rows.append(
            {
                "variant": variant_name,
                "split": split_name,
                "size_bin": int(size_bin),
                "n_proteins": int(len(rows)),
                "mean_n_residues": float(np.mean([row["n_residues"] for row in rows])),
                "mean_per_protein_r2": (
                    float(np.mean(r2_values)) if r2_values else float("nan")
                ),
                "mean_error_pred_minus_true": float(
                    np.mean([row["mean_error_pred_minus_true"] for row in rows])
                ),
                "median_error_pred_minus_true": float(
                    np.median([row["mean_error_pred_minus_true"] for row in rows])
                ),
                "mean_overestimate_fraction": float(
                    np.mean([row["overestimate_fraction"] for row in rows])
                ),
                "mean_underestimate_fraction": float(
                    np.mean([row["underestimate_fraction"] for row in rows])
                ),
                "mean_overestimate": float(np.mean([row["mean_overestimate"] for row in rows])),
                "mean_underestimate": float(np.mean([row["mean_underestimate"] for row in rows])),
            }
        )
    return out_rows


def build_split_rows(train_records, test_records):
    """Format protein-level split assignments."""
    rows = []
    for split_name, split_records in [("train", train_records), ("test", test_records)]:
        for record in split_records:
            rows.append(
                {
                    "split": split_name,
                    "protein_id": record["protein_id"],
                    "n_residues": int(record["n_residues"]),
                    "size_bin": int(record["size_bin"]),
                    "n_pairs": int(record["n_pairs"]),
                }
            )
    return rows


def build_test_scatter_rows(variant_name, test_records, bundle, y_pred):
    """Sample a deterministic subset of test rows for plotting/debugging."""
    total_rows = int(bundle["y"].shape[0])
    sample_idx = deterministic_sample_indices(
        label=f"{variant_name}:test_scatter:{total_rows}",
        n_rows=total_rows,
        cap=SCATTER_SAMPLE_SIZE,
        offset=0,
    )

    protein_lookup = np.empty((total_rows,), dtype=np.int32)
    for record_index, (start, end) in enumerate(bundle["offsets"]):
        protein_lookup[start:end] = int(record_index)

    rows = []
    for global_idx in sample_idx:
        record = test_records[int(protein_lookup[int(global_idx)])]
        rows.append(
            {
                "variant": variant_name,
                "protein_id": record["protein_id"],
                "n_residues": int(record["n_residues"]),
                "size_bin": int(record["size_bin"]),
                "seq_sep": int(bundle["seq_sep"][int(global_idx)]),
                "true_distance": float(bundle["y"][int(global_idx)]),
                "predicted_distance": float(y_pred[int(global_idx)]),
                "error_pred_minus_true": float(y_pred[int(global_idx)] - bundle["y"][int(global_idx)]),
            }
        )
    return rows


def evaluate_variant(variant, train_records, test_records):
    """Fit one variant and return all outputs."""
    t0 = time.time()

    train_fit_bundle = build_bundle(train_records, variant, sample_cap=variant["sample_cap"])
    train_eval_bundle = build_bundle(train_records, variant, sample_cap=None)
    test_eval_bundle = build_bundle(test_records, variant, sample_cap=None)

    standardization = compute_standardization_from_bundle(train_fit_bundle)
    apply_standardization_in_place(train_fit_bundle, standardization)
    apply_standardization_in_place(train_eval_bundle, standardization)
    apply_standardization_in_place(test_eval_bundle, standardization)

    model = LinearRegression(copy_X=False)
    model.fit(train_fit_bundle["x"], train_fit_bundle["y"], sample_weight=train_fit_bundle["weights"])

    y_pred_train = np.asarray(model.predict(train_eval_bundle["x"]), dtype=np.float64)
    y_pred_test = np.asarray(model.predict(test_eval_bundle["x"]), dtype=np.float64)
    if variant["prediction_floor"] is not None:
        y_pred_train = np.maximum(y_pred_train, float(variant["prediction_floor"]))
        y_pred_test = np.maximum(y_pred_test, float(variant["prediction_floor"]))

    train_metrics = summarize_predictions_extended(
        "train",
        train_records,
        train_eval_bundle["offsets"],
        train_eval_bundle["y"],
        y_pred_train,
    )
    test_metrics = summarize_predictions_extended(
        "test",
        test_records,
        test_eval_bundle["offsets"],
        test_eval_bundle["y"],
        y_pred_test,
    )

    train_geometry_rows = collect_geometry_rows(
        variant["name"],
        "train",
        train_records,
        train_eval_bundle["y"],
        y_pred_train,
    )
    test_geometry_rows = collect_geometry_rows(
        variant["name"],
        "test",
        test_records,
        test_eval_bundle["y"],
        y_pred_test,
    )
    geometry_rows = train_geometry_rows + test_geometry_rows
    geometry_lookup = {
        (row["variant"], row["split"]): row for row in summarize_geometry_rows(geometry_rows)
    }

    seq_diag_rows = []
    for split_name, bundle, pred in [
        ("train", train_eval_bundle, y_pred_train),
        ("test", test_eval_bundle, y_pred_test),
    ]:
        rows = build_seq_sep_diagnostics(split_name, bundle["seq_sep"], bundle["y"], pred)
        for row in rows:
            row["variant"] = variant["name"]
        seq_diag_rows.extend(rows)

    per_protein_rows = []
    for metric_group in [train_metrics["per_rows"], test_metrics["per_rows"]]:
        for row in metric_group:
            out = dict(row)
            out["variant"] = variant["name"]
            per_protein_rows.append(out)

    summary_row = {
        "variant": variant["name"],
        "feature_mode": variant["feature_mode"],
        "sample_cap_per_train_protein": int(variant["sample_cap"]),
        "prediction_floor_angstrom": (
            ""
            if variant["prediction_floor"] is None
            else float(variant["prediction_floor"])
        ),
        "n_train_proteins": int(len(train_records)),
        "n_test_proteins": int(len(test_records)),
        "n_train_rows_fit": int(train_fit_bundle["x"].shape[0]),
        "n_train_rows_eval": int(train_eval_bundle["x"].shape[0]),
        "n_test_rows_eval": int(test_eval_bundle["x"].shape[0]),
        "train_mean_per_protein_r2": float(train_metrics["mean_per_protein_r2"]),
        "test_mean_per_protein_r2": float(test_metrics["mean_per_protein_r2"]),
        "train_pair_weighted_r2": float(train_metrics["pair_weighted_r2"]),
        "test_pair_weighted_r2": float(test_metrics["pair_weighted_r2"]),
        "train_pair_weighted_mae": float(train_metrics["pair_weighted_mae"]),
        "test_pair_weighted_mae": float(test_metrics["pair_weighted_mae"]),
        "train_pair_weighted_rmse": float(train_metrics["pair_weighted_rmse"]),
        "test_pair_weighted_rmse": float(test_metrics["pair_weighted_rmse"]),
        "train_mean_error_pred_minus_true": float(
            train_metrics["pair_weighted_mean_error_pred_minus_true"]
        ),
        "test_mean_error_pred_minus_true": float(
            test_metrics["pair_weighted_mean_error_pred_minus_true"]
        ),
        "train_tm_score_mean": float(
            geometry_lookup[(variant["name"], "train")]["tm_score_mean"]
        ),
        "test_tm_score_mean": float(
            geometry_lookup[(variant["name"], "test")]["tm_score_mean"]
        ),
        "train_rmsd_mean": float(
            geometry_lookup[(variant["name"], "train")]["rmsd_mean"]
        ),
        "test_rmsd_mean": float(
            geometry_lookup[(variant["name"], "test")]["rmsd_mean"]
        ),
        "train_stress_mean": float(
            geometry_lookup[(variant["name"], "train")]["stress_mean"]
        ),
        "test_stress_mean": float(
            geometry_lookup[(variant["name"], "test")]["stress_mean"]
        ),
        "train_triangle_violation_fraction_mean": float(
            geometry_lookup[(variant["name"], "train")]["triangle_violation_fraction_mean"]
        ),
        "test_triangle_violation_fraction_mean": float(
            geometry_lookup[(variant["name"], "test")]["triangle_violation_fraction_mean"]
        ),
        "elapsed_seconds": float(time.time() - t0),
    }

    scatter_rows = build_test_scatter_rows(variant["name"], test_records, test_eval_bundle, y_pred_test)

    del train_fit_bundle
    del train_eval_bundle
    del test_eval_bundle
    return {
        "summary_row": summary_row,
        "per_protein_rows": per_protein_rows,
        "geometry_rows": geometry_rows,
        "seq_diag_rows": seq_diag_rows,
        "scatter_rows": scatter_rows,
    }


def build_summary_text(
    path,
    elapsed_seconds,
    all_proteins,
    loaded_records,
    skipped,
    train_records,
    test_records,
    summary_rows,
    size_bias_rows,
):
    """Write a compact human-readable summary."""
    ranked = sorted(
        summary_rows,
        key=lambda row: float(row["test_mean_per_protein_r2"]),
        reverse=True,
    )
    summary_by_name = {row["variant"]: row for row in summary_rows}

    def best_by(field_name, reverse=True):
        rows = [row for row in summary_rows if not math.isnan(float(row[field_name]))]
        return sorted(rows, key=lambda row: float(row[field_name]), reverse=reverse)[0]

    smallest_test_rows = [
        row for row in size_bias_rows if row["split"] == "test" and int(row["size_bin"]) == 0
    ]
    if not smallest_test_rows:
        min_test_bin = min(
            int(row["size_bin"]) for row in size_bias_rows if row["split"] == "test"
        )
        smallest_test_rows = [
            row
            for row in size_bias_rows
            if row["split"] == "test" and int(row["size_bin"]) == min_test_bin
        ]
    else:
        min_test_bin = 0

    smallest_by_variant = {row["variant"]: row for row in smallest_test_rows}
    baseline_smallest_bias = float(
        smallest_by_variant["baseline_cap_5k"]["mean_error_pred_minus_true"]
    )
    floor_smallest_bias = float(
        smallest_by_variant["baseline_cap_5k_predfloor_3p3A"]["mean_error_pred_minus_true"]
    )

    best_tm = best_by("test_tm_score_mean", reverse=True)
    best_rmsd = best_by("test_rmsd_mean", reverse=False)

    with path.open("w") as handle:
        handle.write(f"{TITLE}\n")
        handle.write("=" * len(TITLE) + "\n\n")
        handle.write(f"Output CSV dir:   {CSV_DIR.resolve()}\n")
        handle.write(f"Layer:            {LAYER}\n")
        handle.write("Representation:   lower triangle only targets, layer-47 pair channels\n")
        handle.write(f"Train/test split: protein-level random {int((1.0 - TEST_SIZE) * 100)}/{int(TEST_SIZE * 100)}\n")
        handle.write("Training weights: protein-equal weighted least squares\n")
        handle.write("Geometry:         train + test MDS / TM-score / triangle analysis\n")
        handle.write(f"Elapsed seconds:  {elapsed_seconds:.1f}\n\n")

        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(loaded_records)}\n")
        handle.write(f"Proteins skipped:    {len(skipped)}\n")
        handle.write(f"Train proteins:      {len(train_records)}\n")
        handle.write(f"Test proteins:       {len(test_records)}\n\n")

        handle.write("Ranked by test mean per-protein R2:\n")
        for row in ranked:
            handle.write(
                f"  {row['variant']}: test_mean_per_protein_R2={float(row['test_mean_per_protein_r2']):.4f}, "
                f"test_pair_weighted_R2={float(row['test_pair_weighted_r2']):.4f}, "
                f"test_tm_score_mean={float(row['test_tm_score_mean']):.4f}\n"
            )

        handle.write("\nGeometry standouts:\n")
        handle.write(
            f"  Best TM-score: {best_tm['variant']} "
            f"({float(best_tm['test_tm_score_mean']):.4f})\n"
        )
        handle.write(
            f"  Best RMSD: {best_rmsd['variant']} "
            f"({float(best_rmsd['test_rmsd_mean']):.4f})\n"
        )

        handle.write(f"\nSmallest available test-bin bias check (bin {min_test_bin}):\n")
        handle.write(
            "  baseline_cap_5k smallest-bin mean error (pred - true): "
            f"{baseline_smallest_bias:+.4f}\n"
        )
        handle.write(
            "  baseline_cap_5k_predfloor_3p3A smallest-bin mean error (pred - true): "
            f"{floor_smallest_bias:+.4f}\n"
        )
        handle.write(
            "  delta from prediction floor: "
            f"{(floor_smallest_bias - baseline_smallest_bias):+.4f}\n"
        )

        handle.write("\nFeature comparisons vs baseline_cap_5k:\n")
        baseline_row = summary_by_name["baseline_cap_5k"]
        for variant_name in [
            "interaction_raw_n_cap_5k",
            "baseline_cap_5k_predfloor_3p3A",
            "baseline_plus_axial_cap_5k",
            "baseline_plus_axial_diag_cap_5k",
            "baseline_plus_seqsep_eq1_le2_cap_5k",
        ]:
            row = summary_by_name[variant_name]
            handle.write(
                f"  {variant_name}: "
                f"delta_test_mean_per_protein_R2="
                f"{(float(row['test_mean_per_protein_r2']) - float(baseline_row['test_mean_per_protein_r2'])):+.4f}, "
                f"delta_test_tm_score_mean="
                f"{(float(row['test_tm_score_mean']) - float(baseline_row['test_tm_score_mean'])):+.4f}\n"
            )


def run_sanity_checks():
    """Quick local sanity checks for shapes, sampling, and prediction flooring."""
    variant = {"feature_mode": FEATURE_MODE_AXIAL_DIAG}
    pb = np.arange(4 * 4 * N_CHANNELS, dtype=np.float32).reshape(4, 4, N_CHANNELS)
    with tempfile.TemporaryDirectory() as tmpdir:
        pb_path = Path(tmpdir) / "toy_pair_block_47.npy"
        np.save(str(pb_path), pb)
        record = {
            "protein_id": "toy",
            "n_residues": 4,
            "n_pairs": 6,
            "pair_i": np.asarray([1, 2, 2, 3, 3, 3], dtype=np.int32),
            "pair_j": np.asarray([0, 0, 1, 0, 1, 2], dtype=np.int32),
            "seq_sep": np.asarray([1, 2, 1, 3, 2, 1], dtype=np.float32),
            "y_abs": np.linspace(2.0, 7.0, num=6, dtype=np.float32),
            "pb_path": pb_path,
            "size_bin": 0,
        }
        local_idx = np.arange(record["n_pairs"], dtype=np.int64)
        chunk = build_feature_chunk(record, variant, local_idx)
        assert chunk.shape == (6, len(AXIAL_DIAG_OFFSETS) * N_CHANNELS)
        bundle = {
            "x": chunk.copy(),
            "offsets": [(0, 3), (3, 6)],
            "row_counts": [3, 3],
        }
        standardization = compute_standardization_from_bundle(bundle)
        assert standardization["mean"].shape[0] == chunk.shape[1]
        assert standardization["scale"].shape[0] == chunk.shape[1]
        weights = compute_pair_weights([2, 5])
        assert np.isclose(np.sum(weights[:2]), np.sum(weights[2:]))
        pred = np.asarray([1.0, 4.0, 2.5], dtype=np.float64)
        floored = np.maximum(pred, PREDICTION_FLOOR_ANGSTROM)
        assert np.all(floored >= PREDICTION_FLOOR_ANGSTROM)


def main():
    t0 = time.time()
    CSV_DIR.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))
    print(f"Loading proteins from pair blocks / PDBs for layer {LAYER}...")

    all_proteins, base_records, skipped = load_protein_records()
    records = prepare_records(base_records)
    assign_rank_size_bins(records, n_bins=N_SIZE_BINS)
    train_records, test_records = split_global_records(records)

    print(
        f"Loaded {len(records)} proteins; train={len(train_records)}, test={len(test_records)}, "
        f"skipped={len(skipped)}"
    )

    summary_rows = []
    per_protein_rows = []
    geometry_rows = []
    seq_diag_rows = []
    scatter_rows = []

    for variant_index, variant in enumerate(VARIANTS, start=1):
        print(
            f"[{variant_index}/{len(VARIANTS)}] Running {variant['name']} "
            f"(mode={variant['feature_mode']}, cap={variant['sample_cap']})"
        )
        result = evaluate_variant(variant, train_records, test_records)
        summary_rows.append(result["summary_row"])
        per_protein_rows.extend(result["per_protein_rows"])
        geometry_rows.extend(result["geometry_rows"])
        seq_diag_rows.extend(result["seq_diag_rows"])
        scatter_rows.extend(result["scatter_rows"])

    geometry_summary_rows = summarize_geometry_rows(geometry_rows)
    size_bias_rows = build_size_bias_summary(per_protein_rows)
    split_rows = build_split_rows(train_records, test_records)

    write_csv(CSV_DIR / SUMMARY_CSV_NAME, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(CSV_DIR / PER_PROTEIN_CSV_NAME, PER_PROTEIN_FIELDNAMES, per_protein_rows)
    write_csv(CSV_DIR / SPLIT_CSV_NAME, SPLIT_FIELDNAMES, split_rows)
    write_csv(
        CSV_DIR / GEOMETRY_METRICS_CSV_NAME,
        GEOMETRY_METRICS_FIELDNAMES,
        geometry_rows,
    )
    write_csv(
        CSV_DIR / GEOMETRY_SUMMARY_CSV_NAME,
        GEOMETRY_SUMMARY_FIELDNAMES,
        geometry_summary_rows,
    )
    write_csv(
        CSV_DIR / SEQ_DIAGNOSTICS_CSV_NAME,
        SEQ_DIAGNOSTICS_FIELDNAMES,
        seq_diag_rows,
    )
    write_csv(CSV_DIR / SIZE_BIAS_CSV_NAME, SIZE_BIAS_FIELDNAMES, size_bias_rows)
    write_csv(CSV_DIR / SCATTER_CSV_NAME, SCATTER_FIELDNAMES, scatter_rows)

    build_summary_text(
        CSV_DIR / SUMMARY_TXT_NAME,
        elapsed_seconds=time.time() - t0,
        all_proteins=all_proteins,
        loaded_records=records,
        skipped=skipped,
        train_records=train_records,
        test_records=test_records,
        summary_rows=summary_rows,
        size_bias_rows=size_bias_rows,
    )

    print(f"Wrote outputs to {CSV_DIR.resolve()}")


if __name__ == "__main__":
    main()

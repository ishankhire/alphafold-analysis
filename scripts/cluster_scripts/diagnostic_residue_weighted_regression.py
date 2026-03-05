#!/usr/bin/env python3
"""
diagnostic_residue_weighted_regression.py
=========================================
Diagnostic script for testing protein-size effects in layer-47 distance
regression across all valid proteins.

Key design choices:
1. Use each unordered residue pair exactly once via lower-triangle indices.
2. Build a paired feature per residue pair by concatenating both directional
   pair representations: [pb[i, j, :], pb[j, i, :]].
3. Compare pooled, protein-equal, normalized-target, interaction, filtered,
   direct-length-plus-sequence-separation, and bucketed linear probes.

Outputs (inside CSV_ROOT/):
    diagnostic_residue_weighted_v2_variant_summary.csv
    diagnostic_residue_weighted_v2_per_protein.csv
    diagnostic_residue_weighted_v2_split.csv
    diagnostic_residue_weighted_v2_bias_by_bucket.csv
    diagnostic_residue_weighted_v2_summary.txt
"""

import csv
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


# ===========================================================================
# Configuration
# ===========================================================================

N_CHANNELS = 128
BASE_FEATURE_DIM = 2 * N_CHANNELS
LAYER = 47
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SIZE_BINS = 5
FLOAT_EPS = 1e-12
RATIO_EPS = 1e-8

SIZE_BUCKETS = [
    {
        "label": "0-150",
        "min_residues": 0,
        "max_residues": 150,
        "scope_name": "bucketed_0_150",
    },
    {
        "label": "150-300",
        "min_residues": 150,
        "max_residues": 300,
        "scope_name": "bucketed_150_300",
    },
    {
        "label": "300-500",
        "min_residues": 300,
        "max_residues": 500,
        "scope_name": "bucketed_300_500",
    },
    {
        "label": "500+",
        "min_residues": 500,
        "max_residues": None,
        "scope_name": "bucketed_500_plus",
    },
]


# ===========================================================================
# Directories (set via environment variables exported by sbatch files)
# ===========================================================================

PAIR_BLOCKS_ROOT = Path(os.environ["PAIR_BLOCKS_ROOT"])
PDB_ROOT = Path(os.environ["PDB_ROOT"])
CSV_DIR = Path(os.environ["CSV_ROOT"])


SUMMARY_CSV_NAME = "diagnostic_residue_weighted_v2_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_residue_weighted_v2_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_residue_weighted_v2_split.csv"
BIAS_BUCKET_CSV_NAME = "diagnostic_residue_weighted_v2_bias_by_bucket.csv"
SUMMARY_TXT_NAME = "diagnostic_residue_weighted_v2_summary.txt"


VARIANTS = [
    {
        "name": "baseline_pair_weighted_abs",
        "scope_name": "global_shared",
        "target_mode": "absolute",
        "feature_mode": "base",
        "length_transform": "none",
        "uses_protein_equal_weighting": False,
    },
    {
        "name": "baseline_protein_equal_abs",
        "scope_name": "global_shared",
        "target_mode": "absolute",
        "feature_mode": "base",
        "length_transform": "none",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "baseline_protein_equal_norm_max",
        "scope_name": "global_shared",
        "target_mode": "normalized_max",
        "feature_mode": "base",
        "length_transform": "none",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "interaction_n_abs",
        "scope_name": "global_shared",
        "target_mode": "absolute",
        "feature_mode": "interaction",
        "length_transform": "n",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "interaction_sqrt_n_abs",
        "scope_name": "global_shared",
        "target_mode": "absolute",
        "feature_mode": "interaction",
        "length_transform": "sqrt_n",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "interaction_log_n_abs",
        "scope_name": "global_shared",
        "target_mode": "absolute",
        "feature_mode": "interaction",
        "length_transform": "log_n",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "interaction_n_sq_abs",
        "scope_name": "global_shared",
        "target_mode": "absolute",
        "feature_mode": "interaction",
        "length_transform": "n_sq",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "baseline_pair_weighted_abs_lt_500",
        "scope_name": "lt_500_only",
        "target_mode": "absolute",
        "feature_mode": "base",
        "length_transform": "none",
        "uses_protein_equal_weighting": False,
    },
    {
        "name": "baseline_protein_equal_abs_plus_n_feature",
        "scope_name": "global_shared",
        "target_mode": "absolute",
        "feature_mode": "base_plus_n",
        "length_transform": "n",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "bucketed_base_abs_fixed_bins",
        "scope_name": "bucketed_fixed_bins",
        "target_mode": "absolute",
        "feature_mode": "base",
        "length_transform": "none",
        "uses_protein_equal_weighting": True,
        "is_bucketed": True,
    },
]


# ===========================================================================
# Helper functions — data loading
# ===========================================================================

def find_pdb_file(protein_id):
    """Locate the predicted PDB file for a protein."""
    pdb_path = (
        PDB_ROOT
        / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions"
        / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    )
    return pdb_path if pdb_path.exists() else None


def get_pair_block_path(protein_dir, protein_id, layer):
    """Return the path to a pair block .npy file for a given layer."""
    return protein_dir / f"{protein_id}_pair_block_{layer}.npy"


def discover_proteins(pair_blocks_root):
    """Find all protein subdirectories under the pair blocks root."""
    proteins = []
    for entry in sorted(pair_blocks_root.iterdir()):
        if entry.is_dir():
            proteins.append((entry.name, entry))
    return proteins


# ===========================================================================
# Helper functions — coordinate extraction & distance computation
# ===========================================================================

def extract_ca_coordinates(pdb_path):
    """Extract CA atom coordinates from a PDB file across all chains."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", str(pdb_path))

    residues = []
    coordinates = []

    model = structure[0]
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] == "W":
                continue
            if "CA" not in residue:
                continue
            residues.append((chain.get_id(), residue.get_resname(), residue.get_id()[1]))
            coordinates.append(residue["CA"].get_coord())

    return {
        "residues": residues,
        "coordinates": (
            np.array(coordinates, dtype=np.float32)
            if coordinates
            else np.empty((0, 3), dtype=np.float32)
        ),
    }


def compute_distance_matrix(coords):
    """Vectorized pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def build_feature_chunk(protein_record, layer):
    """Load one protein/layer pair block and extract 256-dim paired features."""
    pb_path = get_pair_block_path(
        protein_record["protein_dir"], protein_record["protein_id"], layer
    )
    pb = np.load(str(pb_path))
    chunk = np.concatenate(
        [
            pb[protein_record["lt_i"], protein_record["lt_j"], :],
            pb[protein_record["lt_j"], protein_record["lt_i"], :],
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    del pb
    return chunk


# ===========================================================================
# Helper functions — scope, buckets, and splitting
# ===========================================================================

def get_bucket_info(n_residues):
    """Return fixed residue bucket metadata."""
    for bucket in SIZE_BUCKETS:
        max_residues = bucket["max_residues"]
        if max_residues is None:
            if n_residues >= bucket["min_residues"]:
                return bucket
        elif bucket["min_residues"] <= n_residues < max_residues:
            return bucket
    raise ValueError(f"No bucket defined for residue count: {n_residues}")


def assign_size_bins(protein_data, n_bins):
    """Assign rank-based size bins by residue count."""
    sorted_records = sorted(
        protein_data,
        key=lambda record: (record["n_residues"], record["protein_id"]),
    )
    total = len(sorted_records)
    for rank, record in enumerate(sorted_records):
        record["size_bin"] = int((rank * n_bins) // max(total, 1))
        if record["size_bin"] >= n_bins:
            record["size_bin"] = n_bins - 1


def choose_stratify_labels(proteins):
    """Use stratification only when it is feasible for the requested split."""
    if len(proteins) < 2:
        return None

    labels = np.array([d["size_bin"] for d in proteins], dtype=int)
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        return None
    if np.min(counts) < 2:
        return None

    n_total = len(proteins)
    n_test = max(1, int(math.ceil(TEST_SIZE * n_total)))
    n_train = n_total - n_test
    if n_test < unique.size or n_train < unique.size:
        return None
    return labels


def split_scope_records(scope_records):
    """Split one scope into train/test sets with safe optional stratification."""
    if len(scope_records) < 2:
        raise ValueError("Need at least two proteins to form a train/test split")

    indices = np.arange(len(scope_records))
    stratify_labels = choose_stratify_labels(scope_records)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    train_set = set(int(idx) for idx in train_idx)
    test_set = set(int(idx) for idx in test_idx)
    train_records = [record for idx, record in enumerate(scope_records) if idx in train_set]
    test_records = [record for idx, record in enumerate(scope_records) if idx in test_set]
    return train_records, test_records


def build_scope_spec(scope_name, protein_records, bucket_label=None):
    """Create one reusable split scope from a list of proteins."""
    if not protein_records:
        return None

    scope_records = [dict(record) for record in protein_records]
    assign_size_bins(scope_records, N_SIZE_BINS)
    train_records, test_records = split_scope_records(scope_records)

    split_rows = []
    for split_name, split_group in [("train", train_records), ("test", test_records)]:
        for record in split_group:
            split_rows.append(
                {
                    "variant_scope": scope_name,
                    "protein_id": record["protein_id"],
                    "split": split_name,
                    "size_bin": record["size_bin"],
                    "bucket_label": record["bucket_label"],
                    "n_residues": record["n_residues"],
                    "n_pairs": record["n_pairs"],
                }
            )

    return {
        "name": scope_name,
        "bucket_label": bucket_label,
        "all_proteins": scope_records,
        "train_proteins": train_records,
        "test_proteins": test_records,
        "split_rows": split_rows,
    }


# ===========================================================================
# Helper functions — scalar transforms and scaling
# ===========================================================================

def transform_length_value(n_residues, transform_name):
    """Apply the requested scalar transform to residue count."""
    if transform_name == "none":
        return 0.0
    if transform_name == "n":
        return float(n_residues)
    if transform_name == "sqrt_n":
        return float(math.sqrt(n_residues))
    if transform_name == "log_n":
        return float(math.log(n_residues))
    if transform_name == "n_sq":
        return float(n_residues ** 2)
    raise ValueError(f"Unsupported length transform: {transform_name}")


def build_zscore_lookup(train_proteins, all_proteins, transform_name):
    """Compute a train-fitted z-score lookup for a residue-count transform."""
    if transform_name == "none":
        return {record["protein_id"]: 0.0 for record in all_proteins}

    train_values = np.array(
        [transform_length_value(record["n_residues"], transform_name) for record in train_proteins],
        dtype=np.float64,
    )
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    if std < FLOAT_EPS:
        std = 1.0

    lookup = {}
    for record in all_proteins:
        value = transform_length_value(record["n_residues"], transform_name)
        lookup[record["protein_id"]] = float((value - mean) / std)
    return lookup


def compute_split_pair_weights(split_proteins, normalize_mean=False):
    """Return per-pair weights, constant within each protein."""
    pieces = [
        np.full(record["n_pairs"], 1.0 / record["n_pairs"], dtype=np.float64)
        for record in split_proteins
    ]
    if not pieces:
        return np.empty((0,), dtype=np.float64)

    out = np.concatenate(pieces)
    if normalize_mean and out.size > 0:
        out *= out.size / np.sum(out)
    return out


def feature_dim_for_mode(feature_mode):
    """Return the full dense feature width for one feature mode."""
    if feature_mode == "base":
        return BASE_FEATURE_DIM
    if feature_mode == "interaction":
        return BASE_FEATURE_DIM * 2
    if feature_mode == "base_plus_n":
        return BASE_FEATURE_DIM + 3
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def compute_scaler_from_protein_stats(train_proteins, feature_mode, transform_lookup, pair_weighted):
    """Compute train-only mean/std using cached per-protein feature moments."""
    dim = feature_dim_for_mode(feature_mode)
    mean_acc = np.zeros(dim, dtype=np.float64)
    sq_acc = np.zeros(dim, dtype=np.float64)
    total_weight = 0.0

    for record in train_proteins:
        g_val = transform_lookup[record["protein_id"]]
        base_mean = record["base_feature_mean"]
        base_sq_mean = record["base_feature_sq_mean"]

        if feature_mode == "base":
            feature_mean = base_mean
            feature_sq_mean = base_sq_mean
        elif feature_mode == "interaction":
            feature_mean = np.concatenate([base_mean, base_mean * g_val])
            feature_sq_mean = np.concatenate([base_sq_mean, base_sq_mean * (g_val ** 2)])
        else:
            feature_mean = np.concatenate(
                [
                    base_mean,
                    np.array(
                        [
                            g_val,
                            record["seq_sep_mean"],
                            record["log_seq_sep_mean"],
                        ],
                        dtype=np.float64,
                    ),
                ]
            )
            feature_sq_mean = np.concatenate(
                [
                    base_sq_mean,
                    np.array(
                        [
                            g_val ** 2,
                            record["seq_sep_sq_mean"],
                            record["log_seq_sep_sq_mean"],
                        ],
                        dtype=np.float64,
                    ),
                ]
            )

        weight = float(record["n_pairs"]) if pair_weighted else 1.0
        mean_acc += weight * feature_mean
        sq_acc += weight * feature_sq_mean
        total_weight += weight

    mean = mean_acc / max(total_weight, 1.0)
    second_moment = sq_acc / max(total_weight, 1.0)
    var = np.maximum(second_moment - mean * mean, 0.0)
    scale = np.sqrt(var)
    scale[scale < FLOAT_EPS] = 1.0
    return mean.astype(np.float32), scale.astype(np.float32)


def apply_scaler_in_place(matrix, mean, scale):
    """Standardize a matrix in place."""
    matrix -= mean
    matrix /= scale
    return matrix


def repeat_protein_scalar(split_proteins, lookup):
    """Broadcast a per-protein scalar across all pair rows in a split."""
    pieces = [
        np.full(record["n_pairs"], lookup[record["protein_id"]], dtype=np.float32)
        for record in split_proteins
    ]
    if not pieces:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(pieces)


def build_split_sequence_features(split_proteins):
    """Concatenate per-pair sequence-separation features for one split."""
    seq_sep = []
    log_seq_sep = []
    for record in split_proteins:
        seq_sep.append(record["seq_sep"])
        log_seq_sep.append(record["log_seq_sep"])

    if not seq_sep:
        empty = np.empty((0,), dtype=np.float32)
        return empty, empty

    return np.concatenate(seq_sep), np.concatenate(log_seq_sep)


def build_variant_matrices(
    feature_mode,
    x_train_base,
    x_test_base,
    g_train_rows,
    g_test_rows,
    seq_sep_train_rows=None,
    seq_sep_test_rows=None,
    log_seq_sep_train_rows=None,
    log_seq_sep_test_rows=None,
):
    """Construct train/test matrices for one feature mode."""
    if feature_mode == "base":
        return x_train_base.copy(), x_test_base.copy()

    if feature_mode == "interaction":
        n_train = x_train_base.shape[0]
        n_test = x_test_base.shape[0]
        x_train = np.empty((n_train, BASE_FEATURE_DIM * 2), dtype=np.float32)
        x_test = np.empty((n_test, BASE_FEATURE_DIM * 2), dtype=np.float32)
        x_train[:, :BASE_FEATURE_DIM] = x_train_base
        x_test[:, :BASE_FEATURE_DIM] = x_test_base
        np.multiply(x_train_base, g_train_rows[:, None], out=x_train[:, BASE_FEATURE_DIM:])
        np.multiply(x_test_base, g_test_rows[:, None], out=x_test[:, BASE_FEATURE_DIM:])
        return x_train, x_test

    if feature_mode == "base_plus_n":
        n_train = x_train_base.shape[0]
        n_test = x_test_base.shape[0]
        if (
            seq_sep_train_rows is None
            or seq_sep_test_rows is None
            or log_seq_sep_train_rows is None
            or log_seq_sep_test_rows is None
        ):
            raise ValueError("base_plus_n mode requires sequence-separation feature rows")
        x_train = np.empty((n_train, BASE_FEATURE_DIM + 3), dtype=np.float32)
        x_test = np.empty((n_test, BASE_FEATURE_DIM + 3), dtype=np.float32)
        x_train[:, :BASE_FEATURE_DIM] = x_train_base
        x_test[:, :BASE_FEATURE_DIM] = x_test_base
        x_train[:, BASE_FEATURE_DIM] = g_train_rows
        x_train[:, BASE_FEATURE_DIM + 1] = seq_sep_train_rows
        x_train[:, BASE_FEATURE_DIM + 2] = log_seq_sep_train_rows
        x_test[:, BASE_FEATURE_DIM] = g_test_rows
        x_test[:, BASE_FEATURE_DIM + 1] = seq_sep_test_rows
        x_test[:, BASE_FEATURE_DIM + 2] = log_seq_sep_test_rows
        return x_train, x_test

    raise ValueError(f"Unsupported feature mode: {feature_mode}")


# ===========================================================================
# Helper functions — metrics and bookkeeping
# ===========================================================================

def compute_r2(y_true, y_pred, sample_weight=None):
    """Compute R-squared, optionally weighted."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")

    if sample_weight is None:
        mean_true = float(np.mean(y_true))
        ss_res = float(np.sum((y_true - y_pred) ** 2))
        ss_tot = float(np.sum((y_true - mean_true) ** 2))
    else:
        sample_weight = np.asarray(sample_weight, dtype=np.float64)
        if sample_weight.size != y_true.size:
            raise ValueError("sample_weight must align with y_true")
        weight_sum = float(np.sum(sample_weight))
        if weight_sum <= 0.0:
            return float("nan")
        mean_true = float(np.sum(sample_weight * y_true) / weight_sum)
        ss_res = float(np.sum(sample_weight * (y_true - y_pred) ** 2))
        ss_tot = float(np.sum(sample_weight * (y_true - mean_true) ** 2))

    if ss_tot <= FLOAT_EPS:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def compute_mae(y_true, y_pred, sample_weight=None):
    """Compute MAE, optionally weighted."""
    residual = np.abs(np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64))
    if residual.size == 0:
        return float("nan")
    if sample_weight is None:
        return float(np.mean(residual))

    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    weight_sum = float(np.sum(sample_weight))
    if weight_sum <= 0.0:
        return float("nan")
    return float(np.sum(sample_weight * residual) / weight_sum)


def compute_rmse(y_true, y_pred, sample_weight=None):
    """Compute RMSE, optionally weighted."""
    sq = (np.asarray(y_true, dtype=np.float64) - np.asarray(y_pred, dtype=np.float64)) ** 2
    if sq.size == 0:
        return float("nan")
    if sample_weight is None:
        return float(math.sqrt(np.mean(sq)))

    sample_weight = np.asarray(sample_weight, dtype=np.float64)
    weight_sum = float(np.sum(sample_weight))
    if weight_sum <= 0.0:
        return float("nan")
    return float(math.sqrt(np.sum(sample_weight * sq) / weight_sum))


def build_offsets(split_proteins):
    """Build contiguous row offsets for one split."""
    offsets = []
    start = 0
    for record in split_proteins:
        end = start + record["n_pairs"]
        offsets.append((start, end))
        start = end
    return offsets


def summarize_predictions(y_true, y_pred, split_proteins):
    """Compute pooled, protein-weighted, and per-protein metrics."""
    protein_equal_weights = compute_split_pair_weights(split_proteins, normalize_mean=False)
    offsets = build_offsets(split_proteins)

    per_rows = []
    per_r2 = []
    negative_r2 = 0

    for record, (start, end) in zip(split_proteins, offsets):
        y_true_p = y_true[start:end]
        y_pred_p = y_pred[start:end]

        per_r2_val = compute_r2(y_true_p, y_pred_p)
        per_mae_val = compute_mae(y_true_p, y_pred_p)
        per_rmse_val = compute_rmse(y_true_p, y_pred_p)

        residual_true_minus_pred = y_true_p - y_pred_p
        error_pred_minus_true = y_pred_p - y_true_p
        ratio = y_pred_p / np.maximum(y_true_p, RATIO_EPS)

        if not math.isnan(per_r2_val):
            per_r2.append(per_r2_val)
            if per_r2_val < 0.0:
                negative_r2 += 1

        per_rows.append(
            {
                "protein_id": record["protein_id"],
                "size_bin": record["size_bin"],
                "bucket_label": record["bucket_label"],
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
                "max_distance": record["max_distance"],
                "per_protein_r2": per_r2_val,
                "per_protein_mae": per_mae_val,
                "per_protein_rmse": per_rmse_val,
                "mean_true_distance": float(np.mean(y_true_p)),
                "std_true_distance": float(np.std(y_true_p)),
                "mean_predicted_distance": float(np.mean(y_pred_p)),
                "std_predicted_distance": float(np.std(y_pred_p)),
                "mean_residual": float(np.mean(residual_true_minus_pred)),
                "std_residual": float(np.std(residual_true_minus_pred)),
                "mean_error_pred_minus_true": float(np.mean(error_pred_minus_true)),
                "std_error_pred_minus_true": float(np.std(error_pred_minus_true)),
                "variance_error_pred_minus_true": float(np.var(error_pred_minus_true)),
                "mean_pred_true_ratio": float(np.mean(ratio)),
            }
        )

    return {
        "pair_weighted_r2": compute_r2(y_true, y_pred),
        "protein_weighted_r2": compute_r2(y_true, y_pred, sample_weight=protein_equal_weights),
        "pair_weighted_mae": compute_mae(y_true, y_pred),
        "pair_weighted_rmse": compute_rmse(y_true, y_pred),
        "mean_per_protein_r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
        "median_per_protein_r2": float(np.median(per_r2)) if per_r2 else float("nan"),
        "negative_r2_count": int(negative_r2),
        "negative_r2_fraction": (
            float(negative_r2 / len(per_r2)) if per_r2 else float("nan")
        ),
        "per_rows": per_rows,
    }


def make_per_protein_output_rows(variant_name, data_scope, target_mode, split_name, metrics_bundle, oracle_bundle):
    """Build output rows for one variant/split."""
    rows = []
    oracle_rows = oracle_bundle["per_rows"] if oracle_bundle is not None else None

    for idx, base_row in enumerate(metrics_bundle["per_rows"]):
        row = {
            "variant": variant_name,
            "data_scope": data_scope,
            "split": split_name,
            "protein_id": base_row["protein_id"],
            "size_bin": base_row["size_bin"],
            "bucket_label": base_row["bucket_label"],
            "n_residues": base_row["n_residues"],
            "n_pairs": base_row["n_pairs"],
            "max_distance": base_row["max_distance"],
            "target_mode": target_mode,
            "per_protein_r2": base_row["per_protein_r2"],
            "per_protein_mae": base_row["per_protein_mae"],
            "per_protein_rmse": base_row["per_protein_rmse"],
            "mean_true_distance": base_row["mean_true_distance"],
            "std_true_distance": base_row["std_true_distance"],
            "mean_predicted_distance": base_row["mean_predicted_distance"],
            "std_predicted_distance": base_row["std_predicted_distance"],
            "mean_residual": base_row["mean_residual"],
            "std_residual": base_row["std_residual"],
            "mean_error_pred_minus_true": base_row["mean_error_pred_minus_true"],
            "std_error_pred_minus_true": base_row["std_error_pred_minus_true"],
            "variance_error_pred_minus_true": base_row["variance_error_pred_minus_true"],
            "mean_pred_true_ratio": base_row["mean_pred_true_ratio"],
            "per_protein_r2_oracle_abs": float("nan"),
            "per_protein_mae_oracle_abs": float("nan"),
            "per_protein_rmse_oracle_abs": float("nan"),
            "mean_true_distance_oracle_abs": float("nan"),
            "std_true_distance_oracle_abs": float("nan"),
            "mean_predicted_distance_oracle_abs": float("nan"),
            "std_predicted_distance_oracle_abs": float("nan"),
            "mean_residual_oracle_abs": float("nan"),
            "std_residual_oracle_abs": float("nan"),
            "mean_error_pred_minus_true_oracle_abs": float("nan"),
            "std_error_pred_minus_true_oracle_abs": float("nan"),
            "variance_error_pred_minus_true_oracle_abs": float("nan"),
            "mean_pred_true_ratio_oracle_abs": float("nan"),
        }

        if oracle_rows is not None:
            oracle_row = oracle_rows[idx]
            row.update(
                {
                    "per_protein_r2_oracle_abs": oracle_row["per_protein_r2"],
                    "per_protein_mae_oracle_abs": oracle_row["per_protein_mae"],
                    "per_protein_rmse_oracle_abs": oracle_row["per_protein_rmse"],
                    "mean_true_distance_oracle_abs": oracle_row["mean_true_distance"],
                    "std_true_distance_oracle_abs": oracle_row["std_true_distance"],
                    "mean_predicted_distance_oracle_abs": oracle_row["mean_predicted_distance"],
                    "std_predicted_distance_oracle_abs": oracle_row["std_predicted_distance"],
                    "mean_residual_oracle_abs": oracle_row["mean_residual"],
                    "std_residual_oracle_abs": oracle_row["std_residual"],
                    "mean_error_pred_minus_true_oracle_abs": oracle_row["mean_error_pred_minus_true"],
                    "std_error_pred_minus_true_oracle_abs": oracle_row["std_error_pred_minus_true"],
                    "variance_error_pred_minus_true_oracle_abs": oracle_row["variance_error_pred_minus_true"],
                    "mean_pred_true_ratio_oracle_abs": oracle_row["mean_pred_true_ratio"],
                }
            )

        rows.append(row)

    return rows


def feature_mode_label(feature_mode):
    """Human-readable feature mode label."""
    if feature_mode == "base":
        return "base256"
    if feature_mode == "interaction":
        return "interaction512"
    if feature_mode == "base_plus_n":
        return "base259_plus_n_sep"
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def best_variant_name(summary_rows, prefix, data_scope=None):
    """Return the best variant in a family by test mean per-protein R2."""
    candidates = [row for row in summary_rows if row["variant"].startswith(prefix)]
    if data_scope is not None:
        candidates = [row for row in candidates if row["data_scope"] == data_scope]
    if not candidates:
        return None

    best = max(
        candidates,
        key=lambda row: (
            -math.inf
            if math.isnan(float(row["test_mean_per_protein_r2"]))
            else float(row["test_mean_per_protein_r2"])
        ),
    )
    return best["variant"]


def aggregate_bias_by_bucket(per_protein_rows):
    """Aggregate per-protein bias metrics by variant, split, and fixed size bucket."""
    grouped = defaultdict(list)
    for row in per_protein_rows:
        grouped[(row["variant"], row["split"], row["bucket_label"])].append(row)

    bucket_order = {bucket["label"]: idx for idx, bucket in enumerate(SIZE_BUCKETS)}
    out_rows = []

    for (variant, split_name, bucket_label), rows in sorted(
        grouped.items(),
        key=lambda item: (
            item[0][0],
            item[0][1],
            bucket_order.get(item[0][2], 999),
        ),
    ):
        bucket_info = next(bucket for bucket in SIZE_BUCKETS if bucket["label"] == bucket_label)
        r2_values = [float(row["per_protein_r2"]) for row in rows if not math.isnan(float(row["per_protein_r2"]))]
        neg_count = sum(1 for value in r2_values if value < 0.0)
        error_values = np.array(
            [float(row["mean_error_pred_minus_true"]) for row in rows],
            dtype=np.float64,
        )
        ratio_values = np.array(
            [float(row["mean_pred_true_ratio"]) for row in rows],
            dtype=np.float64,
        )
        var_values = np.array(
            [float(row["variance_error_pred_minus_true"]) for row in rows],
            dtype=np.float64,
        )

        out_rows.append(
            {
                "variant": variant,
                "split": split_name,
                "bucket_label": bucket_label,
                "bucket_min_residues": bucket_info["min_residues"],
                "bucket_max_residues": (
                    "" if bucket_info["max_residues"] is None else bucket_info["max_residues"]
                ),
                "n_proteins": len(rows),
                "n_pairs": sum(int(row["n_pairs"]) for row in rows),
                "mean_per_protein_r2": (
                    float(np.mean(r2_values)) if r2_values else float("nan")
                ),
                "median_per_protein_r2": (
                    float(np.median(r2_values)) if r2_values else float("nan")
                ),
                "negative_r2_fraction": (
                    float(neg_count / len(r2_values)) if r2_values else float("nan")
                ),
                "mean_error_pred_minus_true": float(np.mean(error_values)),
                "median_error_pred_minus_true": float(np.median(error_values)),
                "mean_abs_error_pred_minus_true": float(np.mean(np.abs(error_values))),
                "mean_pred_true_ratio": float(np.mean(ratio_values)),
                "median_pred_true_ratio": float(np.median(ratio_values)),
                "mean_error_variance": float(np.mean(var_values)),
            }
        )

    return out_rows


# ===========================================================================
# Helper functions — scope materialization and model execution
# ===========================================================================

def materialize_scope(scope, chunk_map):
    """Build dense train/test matrices and targets for one split scope."""
    x_train_base = np.vstack(
        [chunk_map[record["protein_id"]] for record in scope["train_proteins"]]
    ).astype(np.float32, copy=False)
    x_test_base = np.vstack(
        [chunk_map[record["protein_id"]] for record in scope["test_proteins"]]
    ).astype(np.float32, copy=False)

    y_train_abs = np.concatenate(
        [record["y_abs"] for record in scope["train_proteins"]]
    ).astype(np.float64, copy=False)
    y_test_abs = np.concatenate(
        [record["y_abs"] for record in scope["test_proteins"]]
    ).astype(np.float64, copy=False)
    y_train_norm = np.concatenate(
        [record["y_norm"] for record in scope["train_proteins"]]
    ).astype(np.float64, copy=False)
    y_test_norm = np.concatenate(
        [record["y_norm"] for record in scope["test_proteins"]]
    ).astype(np.float64, copy=False)

    return {
        "x_train_base": x_train_base,
        "x_test_base": x_test_base,
        "y_train_abs": y_train_abs,
        "y_test_abs": y_test_abs,
        "y_train_norm": y_train_norm,
        "y_test_norm": y_test_norm,
        "train_weights_model": compute_split_pair_weights(
            scope["train_proteins"], normalize_mean=True
        ),
    }


def run_single_scope_variant(variant, scope, chunk_map):
    """Fit and evaluate one standard single-model variant on one scope."""
    scope_data = materialize_scope(scope, chunk_map)

    feature_mode = variant["feature_mode"]
    target_mode = variant["target_mode"]
    length_transform = variant["length_transform"]
    use_protein_equal_weighting = variant["uses_protein_equal_weighting"]

    transform_lookup = build_zscore_lookup(
        scope["train_proteins"], scope["all_proteins"], length_transform
    )
    g_train_rows = repeat_protein_scalar(scope["train_proteins"], transform_lookup)
    g_test_rows = repeat_protein_scalar(scope["test_proteins"], transform_lookup)
    seq_sep_train_rows = None
    seq_sep_test_rows = None
    log_seq_sep_train_rows = None
    log_seq_sep_test_rows = None
    if feature_mode == "base_plus_n":
        seq_sep_train_rows, log_seq_sep_train_rows = build_split_sequence_features(
            scope["train_proteins"]
        )
        seq_sep_test_rows, log_seq_sep_test_rows = build_split_sequence_features(
            scope["test_proteins"]
        )

    pair_weighted_scaler = not use_protein_equal_weighting
    scaler_mean, scaler_scale = compute_scaler_from_protein_stats(
        scope["train_proteins"],
        feature_mode=feature_mode,
        transform_lookup=transform_lookup,
        pair_weighted=pair_weighted_scaler,
    )

    x_train, x_test = build_variant_matrices(
        feature_mode=feature_mode,
        x_train_base=scope_data["x_train_base"],
        x_test_base=scope_data["x_test_base"],
        g_train_rows=g_train_rows,
        g_test_rows=g_test_rows,
        seq_sep_train_rows=seq_sep_train_rows,
        seq_sep_test_rows=seq_sep_test_rows,
        log_seq_sep_train_rows=log_seq_sep_train_rows,
        log_seq_sep_test_rows=log_seq_sep_test_rows,
    )
    apply_scaler_in_place(x_train, scaler_mean, scaler_scale)
    apply_scaler_in_place(x_test, scaler_mean, scaler_scale)

    if target_mode == "absolute":
        y_train = scope_data["y_train_abs"]
        y_test = scope_data["y_test_abs"]
    elif target_mode == "normalized_max":
        y_train = scope_data["y_train_norm"]
        y_test = scope_data["y_test_norm"]
    else:
        raise ValueError(f"Unsupported target mode: {target_mode}")

    model = LinearRegression(copy_X=False)
    if use_protein_equal_weighting:
        model.fit(x_train, y_train, sample_weight=scope_data["train_weights_model"])
    else:
        model.fit(x_train, y_train)

    y_pred_train = np.asarray(model.predict(x_train), dtype=np.float64)
    y_pred_test = np.asarray(model.predict(x_test), dtype=np.float64)

    train_metrics = summarize_predictions(y_train, y_pred_train, scope["train_proteins"])
    test_metrics = summarize_predictions(y_test, y_pred_test, scope["test_proteins"])

    oracle_train = None
    oracle_test = None
    if target_mode == "normalized_max":
        train_offsets = build_offsets(scope["train_proteins"])
        test_offsets = build_offsets(scope["test_proteins"])
        oracle_pred_train = np.concatenate(
            [
                y_pred_train[start:end] * record["max_distance"]
                for record, (start, end) in zip(scope["train_proteins"], train_offsets)
            ]
        )
        oracle_pred_test = np.concatenate(
            [
                y_pred_test[start:end] * record["max_distance"]
                for record, (start, end) in zip(scope["test_proteins"], test_offsets)
            ]
        )
        oracle_train = summarize_predictions(
            scope_data["y_train_abs"], oracle_pred_train, scope["train_proteins"]
        )
        oracle_test = summarize_predictions(
            scope_data["y_test_abs"], oracle_pred_test, scope["test_proteins"]
        )

    summary_row = {
        "variant": variant["name"],
        "data_scope": scope["name"],
        "target_mode": target_mode,
        "feature_mode": feature_mode_label(feature_mode),
        "length_transform": length_transform,
        "uses_protein_equal_weighting": use_protein_equal_weighting,
        "n_train_proteins": len(scope["train_proteins"]),
        "n_test_proteins": len(scope["test_proteins"]),
        "n_train_pairs": int(y_train.size),
        "n_test_pairs": int(y_test.size),
        "train_r2_pair_weighted": train_metrics["pair_weighted_r2"],
        "test_r2_pair_weighted": test_metrics["pair_weighted_r2"],
        "train_r2_protein_weighted": train_metrics["protein_weighted_r2"],
        "test_r2_protein_weighted": test_metrics["protein_weighted_r2"],
        "train_mae_pair_weighted": train_metrics["pair_weighted_mae"],
        "test_mae_pair_weighted": test_metrics["pair_weighted_mae"],
        "train_rmse_pair_weighted": train_metrics["pair_weighted_rmse"],
        "test_rmse_pair_weighted": test_metrics["pair_weighted_rmse"],
        "train_mean_per_protein_r2": train_metrics["mean_per_protein_r2"],
        "test_mean_per_protein_r2": test_metrics["mean_per_protein_r2"],
        "test_median_per_protein_r2": test_metrics["median_per_protein_r2"],
        "test_negative_r2_count": test_metrics["negative_r2_count"],
        "test_negative_r2_fraction": test_metrics["negative_r2_fraction"],
        "train_r2_oracle_rescaled_abs": float("nan"),
        "test_r2_oracle_rescaled_abs": float("nan"),
        "train_mae_oracle_rescaled_abs": float("nan"),
        "test_mae_oracle_rescaled_abs": float("nan"),
        "train_rmse_oracle_rescaled_abs": float("nan"),
        "test_rmse_oracle_rescaled_abs": float("nan"),
    }

    if oracle_train is not None and oracle_test is not None:
        summary_row.update(
            {
                "train_r2_oracle_rescaled_abs": oracle_train["pair_weighted_r2"],
                "test_r2_oracle_rescaled_abs": oracle_test["pair_weighted_r2"],
                "train_mae_oracle_rescaled_abs": oracle_train["pair_weighted_mae"],
                "test_mae_oracle_rescaled_abs": oracle_test["pair_weighted_mae"],
                "train_rmse_oracle_rescaled_abs": oracle_train["pair_weighted_rmse"],
                "test_rmse_oracle_rescaled_abs": oracle_test["pair_weighted_rmse"],
            }
        )

    per_rows = []
    per_rows.extend(
        make_per_protein_output_rows(
            variant["name"],
            scope["name"],
            target_mode,
            "train",
            train_metrics,
            oracle_train,
        )
    )
    test_rows = make_per_protein_output_rows(
        variant["name"],
        scope["name"],
        target_mode,
        "test",
        test_metrics,
        oracle_test,
    )
    per_rows.extend(test_rows)

    del x_train, x_test, y_pred_train, y_pred_test, g_train_rows, g_test_rows
    del seq_sep_train_rows, seq_sep_test_rows, log_seq_sep_train_rows, log_seq_sep_test_rows
    del scope_data

    return summary_row, per_rows, test_rows


def run_bucketed_variant(variant, bucket_scopes, chunk_map):
    """Fit one separate model per fixed size bucket and aggregate the outputs."""
    train_true_parts = []
    train_pred_parts = []
    test_true_parts = []
    test_pred_parts = []
    train_proteins_all = []
    test_proteins_all = []

    for scope in bucket_scopes:
        print(
            f"    Bucket {scope['bucket_label']}: "
            f"{len(scope['train_proteins'])} train / {len(scope['test_proteins'])} test proteins"
        )

        scope_data = materialize_scope(scope, chunk_map)
        transform_lookup = build_zscore_lookup(
            scope["train_proteins"], scope["all_proteins"], "none"
        )
        g_train_rows = repeat_protein_scalar(scope["train_proteins"], transform_lookup)
        g_test_rows = repeat_protein_scalar(scope["test_proteins"], transform_lookup)

        scaler_mean, scaler_scale = compute_scaler_from_protein_stats(
            scope["train_proteins"],
            feature_mode="base",
            transform_lookup=transform_lookup,
            pair_weighted=False,
        )

        x_train, x_test = build_variant_matrices(
            feature_mode="base",
            x_train_base=scope_data["x_train_base"],
            x_test_base=scope_data["x_test_base"],
            g_train_rows=g_train_rows,
            g_test_rows=g_test_rows,
        )
        apply_scaler_in_place(x_train, scaler_mean, scaler_scale)
        apply_scaler_in_place(x_test, scaler_mean, scaler_scale)

        model = LinearRegression(copy_X=False)
        model.fit(
            x_train,
            scope_data["y_train_abs"],
            sample_weight=scope_data["train_weights_model"],
        )

        train_true_parts.append(scope_data["y_train_abs"])
        test_true_parts.append(scope_data["y_test_abs"])
        train_pred_parts.append(np.asarray(model.predict(x_train), dtype=np.float64))
        test_pred_parts.append(np.asarray(model.predict(x_test), dtype=np.float64))
        train_proteins_all.extend(scope["train_proteins"])
        test_proteins_all.extend(scope["test_proteins"])

        del x_train, x_test, g_train_rows, g_test_rows
        del scope_data

    y_train_true = np.concatenate(train_true_parts)
    y_train_pred = np.concatenate(train_pred_parts)
    y_test_true = np.concatenate(test_true_parts)
    y_test_pred = np.concatenate(test_pred_parts)

    train_metrics = summarize_predictions(y_train_true, y_train_pred, train_proteins_all)
    test_metrics = summarize_predictions(y_test_true, y_test_pred, test_proteins_all)

    summary_row = {
        "variant": variant["name"],
        "data_scope": "bucketed_fixed_bins",
        "target_mode": "absolute",
        "feature_mode": feature_mode_label("base"),
        "length_transform": "none",
        "uses_protein_equal_weighting": True,
        "n_train_proteins": len(train_proteins_all),
        "n_test_proteins": len(test_proteins_all),
        "n_train_pairs": int(y_train_true.size),
        "n_test_pairs": int(y_test_true.size),
        "train_r2_pair_weighted": train_metrics["pair_weighted_r2"],
        "test_r2_pair_weighted": test_metrics["pair_weighted_r2"],
        "train_r2_protein_weighted": train_metrics["protein_weighted_r2"],
        "test_r2_protein_weighted": test_metrics["protein_weighted_r2"],
        "train_mae_pair_weighted": train_metrics["pair_weighted_mae"],
        "test_mae_pair_weighted": test_metrics["pair_weighted_mae"],
        "train_rmse_pair_weighted": train_metrics["pair_weighted_rmse"],
        "test_rmse_pair_weighted": test_metrics["pair_weighted_rmse"],
        "train_mean_per_protein_r2": train_metrics["mean_per_protein_r2"],
        "test_mean_per_protein_r2": test_metrics["mean_per_protein_r2"],
        "test_median_per_protein_r2": test_metrics["median_per_protein_r2"],
        "test_negative_r2_count": test_metrics["negative_r2_count"],
        "test_negative_r2_fraction": test_metrics["negative_r2_fraction"],
        "train_r2_oracle_rescaled_abs": float("nan"),
        "test_r2_oracle_rescaled_abs": float("nan"),
        "train_mae_oracle_rescaled_abs": float("nan"),
        "test_mae_oracle_rescaled_abs": float("nan"),
        "train_rmse_oracle_rescaled_abs": float("nan"),
        "test_rmse_oracle_rescaled_abs": float("nan"),
    }

    per_rows = []
    per_rows.extend(
        make_per_protein_output_rows(
            variant["name"],
            "bucketed_fixed_bins",
            "absolute",
            "train",
            train_metrics,
            None,
        )
    )
    test_rows = make_per_protein_output_rows(
        variant["name"],
        "bucketed_fixed_bins",
        "absolute",
        "test",
        test_metrics,
        None,
    )
    per_rows.extend(test_rows)

    return summary_row, per_rows, test_rows


# ===========================================================================
# Helper functions — CSV formatting
# ===========================================================================

def format_value(value):
    """Format values for CSV output."""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_csv(path, fieldnames, rows):
    """Write rows to CSV with stable float formatting."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()

    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 72}")
    print("DIAGNOSTIC RESIDUE-WEIGHTED REGRESSION — layer 47, paired 256-dim")
    print(f"{'=' * 72}")
    print(f"Pair blocks root:       {PAIR_BLOCKS_ROOT}")
    print(f"PDB root:               {PDB_ROOT}")
    print(f"Output CSV dir:         {csv_dir}")
    print(f"Layer:                  {LAYER}")
    print("Pair representation:    lower triangle rows, concatenate pb[i,j,:] + pb[j,i,:]")
    print(f"Train/test split:       80/20 protein-level, seed={RANDOM_STATE}")
    print(f"Variants:               {len(VARIANTS)}")

    # ------------------------------------------------------------------
    # Phase 1: Discover and validate proteins
    # ------------------------------------------------------------------
    all_proteins = discover_proteins(PAIR_BLOCKS_ROOT)
    print(f"\nDiscovered {len(all_proteins)} protein directories total.")

    protein_data = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        try:
            pdb_path = find_pdb_file(protein_id)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                print(f"[SKIP] {protein_id}: no predicted PDB file found")
                continue

            pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
            if not pb0_path.exists():
                skipped.append((protein_id, "pair block 0 not found"))
                print(f"[SKIP] {protein_id}: pair block 0 not found at {pb0_path}")
                continue

            ca_data = extract_ca_coordinates(pdb_path)
            n_ca = len(ca_data["residues"])
            if n_ca < 2:
                skipped.append((protein_id, "fewer than 2 CA atoms"))
                print(f"[SKIP] {protein_id}: fewer than 2 CA atoms")
                continue

            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = int(pb0.shape[0])
            del pb0

            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                print(f"[SKIP] {protein_id}: CA count {n_ca} != pair block dim {pb_dim}")
                continue

            lt_i, lt_j = np.tril_indices(n_ca, k=-1)
            if lt_i.size == 0:
                skipped.append((protein_id, "no lower-triangle residue pairs"))
                print(f"[SKIP] {protein_id}: no lower-triangle residue pairs")
                continue

            seq_sep = (lt_i - lt_j).astype(np.float32, copy=False)
            log_seq_sep = np.log1p(seq_sep).astype(np.float32, copy=False)

            dist_matrix = compute_distance_matrix(ca_data["coordinates"])
            y_abs = np.asarray(dist_matrix[lt_i, lt_j], dtype=np.float32)
            max_distance = float(np.max(y_abs))
            if max_distance <= 0.0:
                skipped.append((protein_id, "max CA distance <= 0"))
                print(f"[SKIP] {protein_id}: max CA distance <= 0")
                continue

            bucket_info = get_bucket_info(n_ca)
            protein_data.append(
                {
                    "protein_id": protein_id,
                    "protein_dir": protein_dir,
                    "n_residues": n_ca,
                    "lt_i": lt_i,
                    "lt_j": lt_j,
                    "seq_sep": seq_sep,
                    "log_seq_sep": log_seq_sep,
                    "seq_sep_mean": float(np.mean(seq_sep)),
                    "seq_sep_sq_mean": float(np.mean(seq_sep * seq_sep)),
                    "log_seq_sep_mean": float(np.mean(log_seq_sep)),
                    "log_seq_sep_sq_mean": float(np.mean(log_seq_sep * log_seq_sep)),
                    "y_abs": y_abs,
                    "y_norm": y_abs / max_distance,
                    "n_pairs": int(y_abs.size),
                    "max_distance": max_distance,
                    "mean_distance": float(np.mean(y_abs)),
                    "std_distance": float(np.std(y_abs)),
                    "bucket_label": bucket_info["label"],
                }
            )
            print(f"[OK]   {protein_id}: {n_ca} residues, {y_abs.size:,} lower-triangle pairs")

        except Exception as exc:
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {exc}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(protein_data)} proteins (skipped {len(skipped)}).")

    # ------------------------------------------------------------------
    # Phase 2: Load layer 47 paired features and cache per-protein stats
    # ------------------------------------------------------------------
    print(f"\nLoading layer {LAYER} paired pair representations...")
    t_load = time.time()
    chunk_map = {}
    for record in protein_data:
        chunk = build_feature_chunk(record, LAYER)
        record["base_feature_mean"] = np.asarray(np.mean(chunk, axis=0), dtype=np.float64)
        record["base_feature_sq_mean"] = np.asarray(np.mean(chunk * chunk, axis=0), dtype=np.float64)
        chunk_map[record["protein_id"]] = chunk
    print(f"Loaded all layer-{LAYER} chunks in {time.time() - t_load:.1f}s")

    # ------------------------------------------------------------------
    # Phase 3: Build split scopes
    # ------------------------------------------------------------------
    global_scope = build_scope_spec("global_shared", protein_data)
    lt_500_records = [record for record in protein_data if record["n_residues"] < 500]
    lt_500_scope = build_scope_spec("lt_500_only", lt_500_records)
    bucket_scopes = []
    for bucket in SIZE_BUCKETS:
        bucket_records = [
            record for record in protein_data if record["bucket_label"] == bucket["label"]
        ]
        scope = build_scope_spec(
            bucket["scope_name"],
            bucket_records,
            bucket_label=bucket["label"],
        )
        if scope is not None:
            bucket_scopes.append(scope)

    scopes = {
        "global_shared": global_scope,
        "lt_500_only": lt_500_scope,
    }

    print(
        f"\nGlobal split: {len(global_scope['train_proteins'])} train / "
        f"{len(global_scope['test_proteins'])} test proteins"
    )
    if lt_500_scope is not None:
        print(
            f"<500 split:  {len(lt_500_scope['train_proteins'])} train / "
            f"{len(lt_500_scope['test_proteins'])} test proteins"
        )
    for scope in bucket_scopes:
        print(
            f"Bucket {scope['bucket_label']}: {len(scope['train_proteins'])} train / "
            f"{len(scope['test_proteins'])} test proteins"
        )

    # ------------------------------------------------------------------
    # Phase 4: Persist split membership
    # ------------------------------------------------------------------
    split_rows = []
    split_rows.extend(global_scope["split_rows"])
    if lt_500_scope is not None:
        split_rows.extend(lt_500_scope["split_rows"])
    for scope in bucket_scopes:
        split_rows.extend(scope["split_rows"])

    split_fields = [
        "variant_scope",
        "protein_id",
        "split",
        "size_bin",
        "bucket_label",
        "n_residues",
        "n_pairs",
    ]
    write_csv(csv_dir / SPLIT_CSV_NAME, split_fields, split_rows)
    print(f"Saved split CSV -> {csv_dir / SPLIT_CSV_NAME}")

    # ------------------------------------------------------------------
    # Phase 5: Run all variants
    # ------------------------------------------------------------------
    summary_rows = []
    per_protein_rows = []
    variant_rankings = {}

    for variant in VARIANTS:
        print(f"\n{'-' * 72}")
        print(f"Running variant: {variant['name']}")
        print(
            f"  scope={variant['scope_name']}, target_mode={variant['target_mode']}, "
            f"feature_mode={variant['feature_mode']}, "
            f"length_transform={variant['length_transform']}, "
            f"protein_equal_weighting={variant['uses_protein_equal_weighting']}"
        )
        t_variant = time.time()

        if variant.get("is_bucketed"):
            summary_row, variant_rows, test_rows = run_bucketed_variant(
                variant, bucket_scopes, chunk_map
            )
        else:
            scope = scopes.get(variant["scope_name"])
            if scope is None:
                raise ValueError(f"Scope not found for variant: {variant['name']}")
            summary_row, variant_rows, test_rows = run_single_scope_variant(
                variant, scope, chunk_map
            )

        summary_rows.append(summary_row)
        per_protein_rows.extend(variant_rows)

        rankable = [
            row for row in test_rows if not math.isnan(float(row["per_protein_r2"]))
        ]
        variant_rankings[variant["name"]] = sorted(
            rankable,
            key=lambda row: float(row["per_protein_r2"]),
        )

        print(
            f"  Test pooled R2={summary_row['test_r2_pair_weighted']:.4f}, "
            f"test protein-weighted R2={summary_row['test_r2_protein_weighted']:.4f}, "
            f"test mean per-protein R2={summary_row['test_mean_per_protein_r2']:.4f}"
        )
        print(f"  Completed in {time.time() - t_variant:.1f}s")

    # ------------------------------------------------------------------
    # Phase 6: Persist outputs
    # ------------------------------------------------------------------
    summary_fields = [
        "variant",
        "data_scope",
        "target_mode",
        "feature_mode",
        "length_transform",
        "uses_protein_equal_weighting",
        "n_train_proteins",
        "n_test_proteins",
        "n_train_pairs",
        "n_test_pairs",
        "train_r2_pair_weighted",
        "test_r2_pair_weighted",
        "train_r2_protein_weighted",
        "test_r2_protein_weighted",
        "train_mae_pair_weighted",
        "test_mae_pair_weighted",
        "train_rmse_pair_weighted",
        "test_rmse_pair_weighted",
        "train_mean_per_protein_r2",
        "test_mean_per_protein_r2",
        "test_median_per_protein_r2",
        "test_negative_r2_count",
        "test_negative_r2_fraction",
        "train_r2_oracle_rescaled_abs",
        "test_r2_oracle_rescaled_abs",
        "train_mae_oracle_rescaled_abs",
        "test_mae_oracle_rescaled_abs",
        "train_rmse_oracle_rescaled_abs",
        "test_rmse_oracle_rescaled_abs",
    ]
    write_csv(csv_dir / SUMMARY_CSV_NAME, summary_fields, summary_rows)
    print(f"\nSaved summary CSV -> {csv_dir / SUMMARY_CSV_NAME}")

    per_protein_fields = [
        "variant",
        "data_scope",
        "split",
        "protein_id",
        "size_bin",
        "bucket_label",
        "n_residues",
        "n_pairs",
        "max_distance",
        "target_mode",
        "per_protein_r2",
        "per_protein_mae",
        "per_protein_rmse",
        "mean_true_distance",
        "std_true_distance",
        "mean_predicted_distance",
        "std_predicted_distance",
        "mean_residual",
        "std_residual",
        "mean_error_pred_minus_true",
        "std_error_pred_minus_true",
        "variance_error_pred_minus_true",
        "mean_pred_true_ratio",
        "per_protein_r2_oracle_abs",
        "per_protein_mae_oracle_abs",
        "per_protein_rmse_oracle_abs",
        "mean_true_distance_oracle_abs",
        "std_true_distance_oracle_abs",
        "mean_predicted_distance_oracle_abs",
        "std_predicted_distance_oracle_abs",
        "mean_residual_oracle_abs",
        "std_residual_oracle_abs",
        "mean_error_pred_minus_true_oracle_abs",
        "std_error_pred_minus_true_oracle_abs",
        "variance_error_pred_minus_true_oracle_abs",
        "mean_pred_true_ratio_oracle_abs",
    ]
    write_csv(csv_dir / PER_PROTEIN_CSV_NAME, per_protein_fields, per_protein_rows)
    print(f"Saved per-protein CSV -> {csv_dir / PER_PROTEIN_CSV_NAME}")

    bias_bucket_rows = aggregate_bias_by_bucket(per_protein_rows)
    bias_bucket_fields = [
        "variant",
        "split",
        "bucket_label",
        "bucket_min_residues",
        "bucket_max_residues",
        "n_proteins",
        "n_pairs",
        "mean_per_protein_r2",
        "median_per_protein_r2",
        "negative_r2_fraction",
        "mean_error_pred_minus_true",
        "median_error_pred_minus_true",
        "mean_abs_error_pred_minus_true",
        "mean_pred_true_ratio",
        "median_pred_true_ratio",
        "mean_error_variance",
    ]
    write_csv(csv_dir / BIAS_BUCKET_CSV_NAME, bias_bucket_fields, bias_bucket_rows)
    print(f"Saved bias-by-bucket CSV -> {csv_dir / BIAS_BUCKET_CSV_NAME}")

    # ------------------------------------------------------------------
    # Phase 7: Human-readable summary
    # ------------------------------------------------------------------
    summary_path = csv_dir / SUMMARY_TXT_NAME
    interaction_best = best_variant_name(summary_rows, "interaction_", data_scope="global_shared")
    global_rows = [row for row in summary_rows if row["data_scope"] == "global_shared"]
    best_global = max(
        global_rows,
        key=lambda row: (
            -math.inf
            if math.isnan(float(row["test_mean_per_protein_r2"]))
            else float(row["test_mean_per_protein_r2"])
        ),
    )["variant"]
    best_overall = max(
        summary_rows,
        key=lambda row: (
            -math.inf
            if math.isnan(float(row["test_mean_per_protein_r2"]))
            else float(row["test_mean_per_protein_r2"])
        ),
    )["variant"]

    baseline_pair = next(
        row for row in summary_rows if row["variant"] == "baseline_pair_weighted_abs"
    )
    baseline_protein = next(
        row for row in summary_rows if row["variant"] == "baseline_protein_equal_abs"
    )
    baseline_norm = next(
        row for row in summary_rows if row["variant"] == "baseline_protein_equal_norm_max"
    )

    test_bias_lookup = {
        (row["variant"], row["bucket_label"]): row
        for row in bias_bucket_rows
        if row["split"] == "test"
    }

    with summary_path.open("w") as handle:
        handle.write("Diagnostic Residue-Weighted Regression v2 — Layer 47\n")
        handle.write("=" * 60 + "\n\n")
        handle.write(f"Pair blocks root:   {PAIR_BLOCKS_ROOT}\n")
        handle.write(f"PDB root:           {PDB_ROOT}\n")
        handle.write(f"Output CSV dir:     {csv_dir}\n")
        handle.write(f"Layer:              {LAYER}\n")
        handle.write(
            "Representation:     one lower-triangle row per unordered pair, "
            "features=[pb[i,j,:], pb[j,i,:]]\n"
        )
        handle.write(f"Train/test split:   protein-level 80/20, seed={RANDOM_STATE}\n")
        handle.write(f"Rank size bins:     {N_SIZE_BINS}\n")
        handle.write("Fixed buckets:      0-150, 150-300, 300-500, 500+\n")
        handle.write("Large proteins:     retained globally; separate <500 variant included\n\n")

        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(protein_data)}\n")
        handle.write(f"Proteins skipped:    {len(skipped)}\n")
        handle.write(
            f"Global train/test:   {len(global_scope['train_proteins'])} / "
            f"{len(global_scope['test_proteins'])}\n"
        )
        if lt_500_scope is not None:
            handle.write(
                f"<500 train/test:     {len(lt_500_scope['train_proteins'])} / "
                f"{len(lt_500_scope['test_proteins'])}\n"
            )
        handle.write("\n")

        train_res = [record["n_residues"] for record in global_scope["train_proteins"]]
        test_res = [record["n_residues"] for record in global_scope["test_proteins"]]
        handle.write("Global residue count summary:\n")
        handle.write(
            f"  Train: min={min(train_res)}, median={int(np.median(train_res))}, "
            f"mean={np.mean(train_res):.1f}, max={max(train_res)}\n"
        )
        handle.write(
            f"  Test:  min={min(test_res)}, median={int(np.median(test_res))}, "
            f"mean={np.mean(test_res):.1f}, max={max(test_res)}\n\n"
        )

        handle.write("Variant comparison:\n")
        handle.write(
            "  "
            f"{'Variant':<36} {'Scope':<18} {'TestPairR2':>10} "
            f"{'TestProtR2':>10} {'MeanPP_R2':>10} {'NegFrac':>10}\n"
        )
        for row in summary_rows:
            handle.write(
                "  "
                f"{row['variant']:<36} "
                f"{row['data_scope']:<18} "
                f"{row['test_r2_pair_weighted']:>10.4f} "
                f"{row['test_r2_protein_weighted']:>10.4f} "
                f"{row['test_mean_per_protein_r2']:>10.4f} "
                f"{row['test_negative_r2_fraction']:>10.4f}\n"
            )
        handle.write("\n")

        handle.write("Interpretation snapshot:\n")
        handle.write(
            "  Protein-equal weighting delta (global mean per-protein test R2): "
            f"{baseline_protein['test_mean_per_protein_r2'] - baseline_pair['test_mean_per_protein_r2']:+.4f}\n"
        )
        handle.write(
            "  Normalized-target global test mean per-protein R2 (normalized space): "
            f"{baseline_norm['test_mean_per_protein_r2']:.4f}\n"
        )
        handle.write(
            "  Normalized-target global test pair-weighted oracle absolute R2: "
            f"{baseline_norm['test_r2_oracle_rescaled_abs']:.4f}\n"
        )
        handle.write(f"  Best interaction variant (global): {interaction_best}\n")
        handle.write(f"  Best global variant: {best_global}\n")
        handle.write(
            f"  Best overall variant by mean per-protein test R2: {best_overall} "
            "(scopes are not directly comparable)\n\n"
        )

        handle.write("Bias by size bucket (test split):\n")
        for variant_name in [row["variant"] for row in summary_rows]:
            handle.write(f"  {variant_name}\n")
            handle.write(
                "    "
                f"{'Bucket':<8} {'NProt':>5} {'MeanR2':>9} {'MeanErr':>10} "
                f"{'Pred/True':>10} {'ErrVar':>10}\n"
            )
            for bucket in SIZE_BUCKETS:
                row = test_bias_lookup.get((variant_name, bucket["label"]))
                if row is None:
                    continue
                handle.write(
                    "    "
                    f"{bucket['label']:<8} "
                    f"{int(row['n_proteins']):>5} "
                    f"{float(row['mean_per_protein_r2']):>9.4f} "
                    f"{float(row['mean_error_pred_minus_true']):>10.4f} "
                    f"{float(row['mean_pred_true_ratio']):>10.4f} "
                    f"{float(row['mean_error_variance']):>10.4f}\n"
                )

            small_row = test_bias_lookup.get((variant_name, "0-150"))
            large_row = test_bias_lookup.get((variant_name, "500+"))
            if small_row is not None and large_row is not None:
                if (
                    float(small_row["mean_error_pred_minus_true"]) > 0.0
                    and float(small_row["mean_pred_true_ratio"]) > 1.0
                    and float(large_row["mean_error_pred_minus_true"])
                    < float(small_row["mean_error_pred_minus_true"])
                ):
                    handle.write(
                        "    Small proteins are overpredicted relative to the 500+ bucket.\n"
                    )
            handle.write("\n")

        for variant_name in [row["variant"] for row in summary_rows]:
            ranked = variant_rankings.get(variant_name, [])
            handle.write(f"Top 10 worst test proteins for {variant_name}:\n")
            for row in ranked[:10]:
                handle.write(
                    f"  {row['protein_id']}: R2={float(row['per_protein_r2']):.4f}, "
                    f"n_res={row['n_residues']}, bucket={row['bucket_label']}, "
                    f"mean(pred-true)={float(row['mean_error_pred_minus_true']):+.4f}\n"
                )
            handle.write(f"Top 10 best test proteins for {variant_name}:\n")
            for row in reversed(ranked[-10:]):
                handle.write(
                    f"  {row['protein_id']}: R2={float(row['per_protein_r2']):.4f}, "
                    f"n_res={row['n_residues']}, bucket={row['bucket_label']}, "
                    f"mean(pred-true)={float(row['mean_error_pred_minus_true']):+.4f}\n"
                )
            handle.write("\n")

        if skipped:
            handle.write("Skipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")

    print(f"Saved summary text -> {summary_path}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 72}")
    print("DIAGNOSTIC RUN COMPLETE")
    print(f"Elapsed time: {elapsed:.1f} seconds ({elapsed / 60.0:.1f} minutes)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

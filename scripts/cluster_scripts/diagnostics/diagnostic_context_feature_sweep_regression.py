#!/usr/bin/env python3
"""
diagnostic_context_feature_sweep_regression.py
==============================================
Layer-47 diagnostic sweep for context-augmentation experiments that compare a
shared family of linear probes under one protein-aware evaluation setup.

Requested linear variants:
1. baseline128
2. baseline128_plus_axial
3. baseline128_plus_axial_diag
4. baseline128_plus_aa_onehot_sorted
5. baseline128_plus_diagonal
6. baseline256
7. baseline256_plus_axial
8. baseline256_plus_axial_plus_aa_onehot_sorted
9. baseline256_plus_axial_plus_aa_onehot_sorted_plus_seqsep_raw
10. baseline256_plus_diagonal
11. baseline256_plus_n_residues
12. baseline256_materialized_float64

Requested ridge experiment:
13. baseline256_ridge_cv

Outputs (inside CSV_ROOT/):
    diagnostic_context_feature_sweep_variant_summary.csv
    diagnostic_context_feature_sweep_leaderboard.csv
    diagnostic_context_feature_sweep_per_protein.csv
    diagnostic_context_feature_sweep_split.csv
    diagnostic_context_feature_sweep_coefficients.csv
    diagnostic_context_feature_sweep_alpha_cv.csv
    diagnostic_context_feature_sweep_alpha_selection.csv
    diagnostic_context_feature_sweep_summary.txt
"""

import csv
import math
import os
import time
import traceback
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


TITLE = "Diagnostic Context Feature Sweep Regression - Layer 47"

N_CHANNELS = 128
LAYER = 47
TEST_SIZE = 0.2
RANDOM_STATE = 42
FLOAT_EPS = 1e-12
N_SIZE_BINS = 5
INNER_CV_FOLDS = 5

AXIAL_NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
DIAGONAL_NEIGHBOR_OFFSETS = [(-1, -1), (-1, 1), (1, -1), (1, 1)]

STANDARD_AAS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLU",
    "GLN",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(STANDARD_AAS)}

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

DTYPE_MODE_STD = "materialize_float32_fit_float64"
DTYPE_MODE_FLOAT64 = "materialize_float64_fit_float64"
INTERCEPT_MODE = "fit_intercept"

SUMMARY_CSV_NAME = "diagnostic_context_feature_sweep_variant_summary.csv"
LEADERBOARD_CSV_NAME = "diagnostic_context_feature_sweep_leaderboard.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_context_feature_sweep_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_context_feature_sweep_split.csv"
COEFFICIENT_CSV_NAME = "diagnostic_context_feature_sweep_coefficients.csv"
ALPHA_CV_CSV_NAME = "diagnostic_context_feature_sweep_alpha_cv.csv"
ALPHA_SELECTION_CSV_NAME = "diagnostic_context_feature_sweep_alpha_selection.csv"
SUMMARY_TXT_NAME = "diagnostic_context_feature_sweep_summary.txt"

SUMMARY_FIELDNAMES = [
    "variant",
    "model_family",
    "feature_dim",
    "dtype_mode",
    "alpha",
    "selected_by_cv",
    "intercept_mode",
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
    "intercept_standardized",
    "intercept_raw",
    "coef_l2_standardized",
    "coef_l2_raw",
    "n_nonzero_coefficients",
    "elapsed_seconds",
]

LEADERBOARD_FIELDNAMES = [
    "leaderboard_rank",
    "variant",
    "model_family",
    "dtype_mode",
    "feature_dim",
    "alpha",
    "test_mean_per_protein_r2",
    "test_pair_weighted_r2",
    "test_pair_weighted_mae",
    "test_pair_weighted_rmse",
    "test_mean_bias",
    "coef_l2_raw",
    "n_nonzero_coefficients",
]

PER_PROTEIN_FIELDNAMES = [
    "variant",
    "model_family",
    "dtype_mode",
    "alpha",
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

SPLIT_FIELDNAMES = [
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
]

COEFFICIENT_FIELDNAMES = [
    "variant",
    "model_family",
    "feature_dim",
    "dtype_mode",
    "alpha",
    "intercept_mode",
    "feature_index",
    "feature_name",
    "coefficient_standardized",
    "coefficient_raw_scale",
]

ALPHA_CV_FIELDNAMES = [
    "variant",
    "model_family",
    "alpha",
    "fold_family",
    "fold_index",
    "n_train_proteins",
    "n_val_proteins",
    "val_mean_per_protein_r2",
    "val_pair_weighted_r2",
    "val_pair_weighted_mae",
    "val_pair_weighted_rmse",
    "elapsed_seconds",
]

ALPHA_SELECTION_FIELDNAMES = [
    "variant",
    "model_family",
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


def discover_proteins(pair_blocks_root):
    """Find protein directories under the configured pair-block root."""
    proteins = []
    for entry in sorted(pair_blocks_root.iterdir()):
        if entry.is_dir():
            proteins.append((entry.name, entry))
    return proteins


def get_pair_block_path(protein_dir, protein_id, layer):
    """Support both Delta-style and local protein_data layouts."""
    candidates = [
        protein_dir / f"{protein_id}_pair_block_{layer}.npy",
        protein_dir / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def find_pdb_file(protein_id, protein_dir, pdb_root):
    """Support both Delta-style and local protein_data PDB layouts."""
    candidates = [
        protein_dir / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
        pdb_root
        / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions"
        / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
        pdb_root / protein_id / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def extract_ca_coordinates(pdb_path):
    """Extract CA coordinates and residue metadata from the predicted PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", str(pdb_path))
    residues = []
    coordinates = []
    for chain in structure[0]:
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
            np.asarray(coordinates, dtype=np.float32)
            if coordinates
            else np.empty((0, 3), dtype=np.float32)
        ),
    }


def compute_distance_matrix(coords):
    """Vectorized pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def build_sorted_aa_pair_indices(residues, pair_i, pair_j):
    """Encode alphabetically sorted residue identities for each pair."""
    residue_names = [record[1] for record in residues]
    n_pairs = len(pair_i)
    aa_first = np.full(n_pairs, -1, dtype=np.int16)
    aa_second = np.full(n_pairs, -1, dtype=np.int16)

    for idx in range(n_pairs):
        aa_i = residue_names[pair_i[idx]]
        aa_j = residue_names[pair_j[idx]]
        if aa_i <= aa_j:
            aa_left = aa_i
            aa_right = aa_j
        else:
            aa_left = aa_j
            aa_right = aa_i

        left_idx = AA_TO_INDEX.get(aa_left)
        right_idx = AA_TO_INDEX.get(aa_right)
        if left_idx is not None:
            aa_first[idx] = left_idx
        if right_idx is not None:
            aa_second[idx] = right_idx

    return aa_first, aa_second


def load_protein_records(pair_blocks_root, pdb_root, protein_ids=None):
    """Load all valid proteins and attach pair targets plus residue identities."""
    requested = set(protein_ids) if protein_ids is not None else None
    all_proteins = discover_proteins(pair_blocks_root)
    protein_data = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        if requested is not None and protein_id not in requested:
            continue
        try:
            pdb_path = find_pdb_file(protein_id, protein_dir, pdb_root)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                continue

            pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
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

            seq_sep = (lt_i - lt_j).astype(np.float32, copy=False)
            dist_matrix = compute_distance_matrix(ca_data["coordinates"])
            y_abs = np.asarray(dist_matrix[lt_i, lt_j], dtype=np.float32)
            aa_first, aa_second = build_sorted_aa_pair_indices(ca_data["residues"], lt_i, lt_j)
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
                    "seq_sep": seq_sep,
                    "y_abs": y_abs,
                    "n_pairs": int(y_abs.size),
                    "aa_first_idx": aa_first,
                    "aa_second_idx": aa_second,
                    "target_variance": target_variance,
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            traceback.print_exc()

    return all_proteins, protein_data, skipped


def assign_size_bins(records, n_bins):
    """Assign stable rank-based bins using residue count."""
    ordered = sorted(records, key=lambda row: (row["n_residues"], row["protein_id"]))
    total = len(ordered)
    for rank, record in enumerate(ordered):
        size_bin = int((rank * n_bins) // max(total, 1))
        if size_bin >= n_bins:
            size_bin = n_bins - 1
        record["size_bin"] = int(size_bin)


def choose_stratify_labels(records):
    """Only stratify when the requested split is feasible."""
    if len(records) < 2:
        return None

    labels = np.asarray([record["size_bin"] for record in records], dtype=np.int32)
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2 or np.min(counts) < 2:
        return None

    n_total = len(records)
    n_test = max(1, int(math.ceil(TEST_SIZE * n_total)))
    n_train = n_total - n_test
    if n_test < unique.size or n_train < unique.size:
        return None
    return labels


def split_records(records):
    """Create one deterministic size-stratified outer split when feasible."""
    if len(records) < 2:
        raise ValueError("Need at least two proteins for train/test splitting.")

    indices = np.arange(len(records), dtype=np.int64)
    labels = choose_stratify_labels(records)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=labels,
    )
    train_set = set(int(idx) for idx in train_idx)
    test_set = set(int(idx) for idx in test_idx)
    train_records = [record for idx, record in enumerate(records) if idx in train_set]
    test_records = [record for idx, record in enumerate(records) if idx in test_set]
    return train_records, test_records


def choose_inner_splitter(records):
    """Use stratified folds when the training pool supports them."""
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
    """Materialize inner train/validation folds for ridge alpha selection."""
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


def format_offset_label(offset):
    """Create stable feature labels for context offsets."""
    di, dj = offset
    return f"di_{di:+d}_dj_{dj:+d}".replace("+", "p").replace("-", "m")


def make_variant(
    name,
    base_mode,
    model_family="linear",
    use_axial=False,
    use_diag_neighbors=False,
    use_aa_onehot=False,
    use_self_diag=False,
    use_seqsep_raw=False,
    use_n_residues=False,
    dtype_mode=DTYPE_MODE_STD,
):
    """Construct one fixed in-code feature spec."""
    return {
        "name": name,
        "base_mode": base_mode,
        "model_family": model_family,
        "use_axial": bool(use_axial),
        "use_diag_neighbors": bool(use_diag_neighbors),
        "use_aa_onehot": bool(use_aa_onehot),
        "use_self_diag": bool(use_self_diag),
        "use_seqsep_raw": bool(use_seqsep_raw),
        "use_n_residues": bool(use_n_residues),
        "dtype_mode": dtype_mode,
    }


LINEAR_VARIANTS = [
    make_variant("baseline128", "128"),
    make_variant("baseline128_plus_axial", "128", use_axial=True),
    make_variant("baseline128_plus_axial_diag", "128", use_axial=True, use_diag_neighbors=True),
    make_variant("baseline128_plus_aa_onehot_sorted", "128", use_aa_onehot=True),
    make_variant("baseline128_plus_diagonal", "128", use_self_diag=True),
    make_variant("baseline256", "256"),
    make_variant("baseline256_plus_axial", "256", use_axial=True),
    make_variant("baseline256_plus_axial_plus_aa_onehot_sorted", "256", use_axial=True, use_aa_onehot=True),
    make_variant(
        "baseline256_plus_axial_plus_aa_onehot_sorted_plus_seqsep_raw",
        "256",
        use_axial=True,
        use_aa_onehot=True,
        use_seqsep_raw=True,
    ),
    make_variant("baseline256_plus_diagonal", "256", use_self_diag=True),
    make_variant("baseline256_plus_n_residues", "256", use_n_residues=True),
    make_variant("baseline256_materialized_float64", "256", dtype_mode=DTYPE_MODE_FLOAT64),
]

RIDGE_VARIANT = make_variant("baseline256_ridge_cv", "256", model_family="ridge_cv")


def variant_context_offsets(variant):
    """Return the ordered context offsets appended for one variant."""
    offsets = []
    if variant["use_axial"]:
        offsets.extend(AXIAL_NEIGHBOR_OFFSETS)
    if variant["use_diag_neighbors"]:
        offsets.extend(DIAGONAL_NEIGHBOR_OFFSETS)
    return offsets


def feature_names_for_variant(variant):
    """Return ordered feature names for one variant."""
    names = []
    if variant["base_mode"] == "128":
        names.extend([f"pair_ij_ch_{idx}" for idx in range(N_CHANNELS)])
    elif variant["base_mode"] == "256":
        names.extend([f"pair_ij_ch_{idx}" for idx in range(N_CHANNELS)])
        names.extend([f"pair_ji_ch_{idx}" for idx in range(N_CHANNELS)])
    else:  # pragma: no cover - fixed variant table
        raise ValueError(f"Unsupported base_mode: {variant['base_mode']}")

    for offset in variant_context_offsets(variant):
        label = format_offset_label(offset)
        names.extend([f"context_{label}_ch_{idx}" for idx in range(N_CHANNELS)])

    if variant["use_self_diag"]:
        names.extend([f"diag_ii_ch_{idx}" for idx in range(N_CHANNELS)])
        names.extend([f"diag_jj_ch_{idx}" for idx in range(N_CHANNELS)])

    if variant["use_aa_onehot"]:
        names.extend([f"aa_sorted_first_{aa}" for aa in STANDARD_AAS])
        names.extend([f"aa_sorted_second_{aa}" for aa in STANDARD_AAS])

    if variant["use_seqsep_raw"]:
        names.append("seq_sep_raw")

    if variant["use_n_residues"]:
        names.append("n_residues")

    return names


def feature_dim_for_variant(variant):
    """Return the dense feature width for one variant."""
    return len(feature_names_for_variant(variant))


def chunk_dtype_for_variant(variant):
    """Return raw materialization dtype for one variant."""
    if variant["dtype_mode"] == DTYPE_MODE_FLOAT64:
        return np.float64
    return np.float32


def build_context_chunk(pb, pair_i, pair_j, offsets, chunk_dtype):
    """Build zero-padded lower-triangle context features for one protein."""
    n_pairs = pair_i.size
    chunk = np.zeros((n_pairs, len(offsets) * N_CHANNELS), dtype=chunk_dtype)
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


def build_aa_onehot_chunk(record, chunk_dtype):
    """Build the sorted residue-identity one-hot block for one protein."""
    n_pairs = int(record["n_pairs"])
    chunk = np.zeros((n_pairs, 2 * len(STANDARD_AAS)), dtype=chunk_dtype)
    aa_first = record["aa_first_idx"]
    aa_second = record["aa_second_idx"]

    mask_first = aa_first >= 0
    mask_second = aa_second >= 0
    if np.any(mask_first):
        chunk[mask_first, aa_first[mask_first]] = 1.0
    if np.any(mask_second):
        chunk[mask_second, len(STANDARD_AAS) + aa_second[mask_second]] = 1.0
    return chunk


def build_protein_feature_chunk(record, variant):
    """Load one protein pair block and build the requested dense feature rows."""
    pb_path = get_pair_block_path(record["protein_dir"], record["protein_id"], LAYER)
    pb = np.load(str(pb_path), mmap_mode="r")
    pair_i = record["lt_i"]
    pair_j = record["lt_j"]
    chunk_dtype = chunk_dtype_for_variant(variant)
    pieces = []

    if variant["base_mode"] == "128":
        pieces.append(np.asarray(pb[pair_i, pair_j, :], dtype=chunk_dtype))
    elif variant["base_mode"] == "256":
        base_ij = np.asarray(pb[pair_i, pair_j, :], dtype=chunk_dtype)
        base_ji = np.asarray(pb[pair_j, pair_i, :], dtype=chunk_dtype)
        pieces.append(np.concatenate([base_ij, base_ji], axis=1).astype(chunk_dtype, copy=False))
    else:  # pragma: no cover - fixed variant table
        raise ValueError(f"Unsupported base_mode: {variant['base_mode']}")

    offsets = variant_context_offsets(variant)
    if offsets:
        pieces.append(build_context_chunk(pb, pair_i, pair_j, offsets, chunk_dtype))

    if variant["use_self_diag"]:
        diag_chunk = np.empty((int(record["n_pairs"]), 2 * N_CHANNELS), dtype=chunk_dtype)
        diag_chunk[:, :N_CHANNELS] = pb[pair_i, pair_i, :]
        diag_chunk[:, N_CHANNELS:] = pb[pair_j, pair_j, :]
        pieces.append(diag_chunk)

    if variant["use_aa_onehot"]:
        pieces.append(build_aa_onehot_chunk(record, chunk_dtype))

    if variant["use_seqsep_raw"]:
        pieces.append(record["seq_sep"].astype(chunk_dtype, copy=False).reshape(-1, 1))

    if variant["use_n_residues"]:
        pieces.append(
            np.full((int(record["n_pairs"]), 1), float(record["n_residues"]), dtype=chunk_dtype)
        )

    del pb
    if len(pieces) == 1:
        return pieces[0]
    return np.concatenate(pieces, axis=1).astype(chunk_dtype, copy=False)


def compute_scaler_from_variant(train_records, variant):
    """Compute train-only protein-equal mean and std for one feature variant."""
    feature_dim = feature_dim_for_variant(variant)
    mean_acc = np.zeros((feature_dim,), dtype=np.float64)
    sq_acc = np.zeros((feature_dim,), dtype=np.float64)

    for record in train_records:
        chunk = build_protein_feature_chunk(record, variant)
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


def standardized_feature_block(record, variant, scaler_mean, scaler_scale):
    """Return one protein's standardized feature matrix in float64."""
    raw_chunk = build_protein_feature_chunk(record, variant)
    x_block = raw_chunk.astype(np.float64, copy=True)
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


def fit_model(model_family, train_records, variant, scaler_mean, scaler_scale, alpha=float("nan")):
    """Fit one standardized-space regression model via streamed normal equations."""
    feature_dim = feature_dim_for_variant(variant)
    system_dim = feature_dim + 1
    xtwx = np.zeros((system_dim, system_dim), dtype=np.float64)
    xtwy = np.zeros((system_dim,), dtype=np.float64)

    for record in train_records:
        x_block = standardized_feature_block(record, variant, scaler_mean, scaler_scale)
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

    if model_family == "ridge_cv":
        xtwx[1:, 1:] += float(alpha) * np.eye(feature_dim, dtype=np.float64)
    elif model_family != "linear":  # pragma: no cover - fixed run table
        raise ValueError(f"Unsupported model_family: {model_family}")

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


def summarize_split(records, variant, scaler_mean, scaler_scale, fitted):
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
        x_block = standardized_feature_block(record, variant, scaler_mean, scaler_scale)
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


def build_coefficient_rows(variant, fitted, scaler_mean, scaler_scale):
    """Emit intercept plus one row per feature coefficient."""
    coef_std = np.asarray(fitted["coef_std"], dtype=np.float64)
    intercept_std = float(fitted["intercept_std"])
    coef_raw, intercept_raw = standardized_to_raw_coefficients(
        coef_std, intercept_std, scaler_mean, scaler_scale
    )
    feature_names = feature_names_for_variant(variant)

    rows = [
        {
            "variant": variant["name"],
            "model_family": variant["model_family"],
            "feature_dim": int(len(feature_names)),
            "dtype_mode": variant["dtype_mode"],
            "alpha": float(variant.get("alpha", float("nan"))),
            "intercept_mode": INTERCEPT_MODE,
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
                "model_family": variant["model_family"],
                "feature_dim": int(len(feature_names)),
                "dtype_mode": variant["dtype_mode"],
                "alpha": float(variant.get("alpha", float("nan"))),
                "intercept_mode": INTERCEPT_MODE,
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "coefficient_standardized": float(coef_std[feature_index]),
                "coefficient_raw_scale": float(coef_raw[feature_index]),
            }
        )
    return rows


def evaluate_variant(variant, train_records, test_records):
    """Fit one final outer-split variant and collect summary outputs."""
    t0 = time.time()
    scaler_mean, scaler_scale = compute_scaler_from_variant(train_records, variant)
    fitted = fit_model(
        model_family=variant["model_family"],
        train_records=train_records,
        variant=variant,
        scaler_mean=scaler_mean,
        scaler_scale=scaler_scale,
        alpha=variant.get("alpha", float("nan")),
    )
    train_metrics = summarize_split(train_records, variant, scaler_mean, scaler_scale, fitted)
    test_metrics = summarize_split(test_records, variant, scaler_mean, scaler_scale, fitted)

    coef_raw, intercept_raw = standardized_to_raw_coefficients(
        fitted["coef_std"],
        fitted["intercept_std"],
        scaler_mean,
        scaler_scale,
    )

    per_protein_rows = []
    for split_name, metric_group in [("train", train_metrics), ("test", test_metrics)]:
        for row in metric_group["per_rows"]:
            out = dict(row)
            out["variant"] = variant["name"]
            out["model_family"] = variant["model_family"]
            out["dtype_mode"] = variant["dtype_mode"]
            out["alpha"] = float(variant.get("alpha", float("nan")))
            out["split"] = split_name
            per_protein_rows.append(out)

    summary_row = {
        "variant": variant["name"],
        "model_family": variant["model_family"],
        "feature_dim": int(feature_dim_for_variant(variant)),
        "dtype_mode": variant["dtype_mode"],
        "alpha": float(variant.get("alpha", float("nan"))),
        "selected_by_cv": bool(variant.get("selected_by_cv", False)),
        "intercept_mode": INTERCEPT_MODE,
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
        "intercept_standardized": float(fitted["intercept_std"]),
        "intercept_raw": float(intercept_raw),
        "coef_l2_standardized": float(np.linalg.norm(fitted["coef_std"])),
        "coef_l2_raw": float(np.linalg.norm(coef_raw)),
        "n_nonzero_coefficients": int(np.sum(np.abs(fitted["coef_std"]) > 1e-12)),
        "elapsed_seconds": float(time.time() - t0),
    }

    return {
        "summary_row": summary_row,
        "per_protein_rows": per_protein_rows,
        "coefficient_rows": build_coefficient_rows(variant, fitted, scaler_mean, scaler_scale),
    }


def choose_alpha_from_summary(summary_rows):
    """Pick the alpha with the best validation mean per-protein R2."""
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


def run_ridge_alpha_sweep(variant, outer_train_records):
    """Run inner-CV ridge alpha selection for the baseline256 feature set."""
    fold_family, folds = build_inner_folds(outer_train_records)
    cv_rows = []
    alpha_summary_rows = []

    for alpha in RIDGE_ALPHA_GRID:
        fold_rows = []
        for fold in folds:
            t0 = time.time()
            scaler_mean, scaler_scale = compute_scaler_from_variant(fold["train_records"], variant)
            fitted = fit_model(
                model_family="ridge_cv",
                train_records=fold["train_records"],
                variant=variant,
                scaler_mean=scaler_mean,
                scaler_scale=scaler_scale,
                alpha=alpha,
            )
            val_metrics = summarize_split(
                fold["val_records"],
                variant,
                scaler_mean,
                scaler_scale,
                fitted,
            )

            row = {
                "variant": variant["name"],
                "model_family": "ridge_cv",
                "alpha": float(alpha),
                "fold_family": fold_family,
                "fold_index": int(fold["fold_index"]),
                "n_train_proteins": int(len(fold["train_records"])),
                "n_val_proteins": int(len(fold["val_records"])),
                "val_mean_per_protein_r2": float(val_metrics["mean_per_protein_r2"]),
                "val_pair_weighted_r2": float(val_metrics["pair_weighted_r2"]),
                "val_pair_weighted_mae": float(val_metrics["pair_weighted_mae"]),
                "val_pair_weighted_rmse": float(val_metrics["pair_weighted_rmse"]),
                "elapsed_seconds": float(time.time() - t0),
            }
            fold_rows.append(row)
            cv_rows.append(row)

        summary_row = {
            "variant": variant["name"],
            "model_family": "ridge_cv",
            "alpha": float(alpha),
            "fold_family": fold_family,
            "fold_index": -1,
            "n_train_proteins": int(np.mean([row["n_train_proteins"] for row in fold_rows])),
            "n_val_proteins": int(np.mean([row["n_val_proteins"] for row in fold_rows])),
            "val_mean_per_protein_r2": float(
                np.mean([row["val_mean_per_protein_r2"] for row in fold_rows])
            ),
            "val_pair_weighted_r2": float(
                np.mean([row["val_pair_weighted_r2"] for row in fold_rows])
            ),
            "val_pair_weighted_mae": float(
                np.mean([row["val_pair_weighted_mae"] for row in fold_rows])
            ),
            "val_pair_weighted_rmse": float(
                np.mean([row["val_pair_weighted_rmse"] for row in fold_rows])
            ),
            "elapsed_seconds": float(np.sum([row["elapsed_seconds"] for row in fold_rows])),
        }
        alpha_summary_rows.append(summary_row)
        cv_rows.append(summary_row)

    chosen_alpha_row = choose_alpha_from_summary(alpha_summary_rows)
    return fold_family, cv_rows, chosen_alpha_row


def build_split_rows(train_records, test_records):
    """Emit one split-membership row per loaded protein."""
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
                    "target_variance": float(record["target_variance"]),
                }
            )
    return rows


def build_leaderboard_rows(summary_rows):
    """Rank final fitted variants by held-out mean per-protein R2."""
    ordered = sorted(
        summary_rows,
        key=lambda row: (
            -1e18 if math.isnan(float(row["test_mean_per_protein_r2"])) else float(row["test_mean_per_protein_r2"]),
            -1e18 if math.isnan(float(row["test_pair_weighted_r2"])) else float(row["test_pair_weighted_r2"]),
        ),
        reverse=True,
    )
    rows = []
    for rank, row in enumerate(ordered, start=1):
        rows.append(
            {
                "leaderboard_rank": int(rank),
                "variant": row["variant"],
                "model_family": row["model_family"],
                "dtype_mode": row["dtype_mode"],
                "feature_dim": int(row["feature_dim"]),
                "alpha": float(row["alpha"]),
                "test_mean_per_protein_r2": float(row["test_mean_per_protein_r2"]),
                "test_pair_weighted_r2": float(row["test_pair_weighted_r2"]),
                "test_pair_weighted_mae": float(row["test_pair_weighted_mae"]),
                "test_pair_weighted_rmse": float(row["test_pair_weighted_rmse"]),
                "test_mean_bias": float(row["test_mean_bias"]),
                "coef_l2_raw": float(row["coef_l2_raw"]),
                "n_nonzero_coefficients": int(row["n_nonzero_coefficients"]),
            }
        )
    return rows


def build_summary_text(
    path,
    elapsed_seconds,
    all_proteins,
    loaded_records,
    skipped,
    leaderboard_rows,
    alpha_selection_rows,
):
    """Write a compact human-readable summary."""
    with path.open("w") as handle:
        handle.write(f"{TITLE}\n")
        handle.write(f"{'=' * len(TITLE)}\n\n")
        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(loaded_records)}\n")
        handle.write(f"Elapsed seconds:     {elapsed_seconds:.2f}\n\n")

        handle.write("Leaderboard:\n")
        for row in leaderboard_rows[:10]:
            handle.write(
                f"  #{int(row['leaderboard_rank'])} {row['variant']}: "
                f"test mean per-protein R2={float(row['test_mean_per_protein_r2']):.4f}, "
                f"test pair-weighted R2={float(row['test_pair_weighted_r2']):.4f}, "
                f"feature_dim={int(row['feature_dim'])}, "
                f"dtype_mode={row['dtype_mode']}\n"
            )

        if alpha_selection_rows:
            handle.write("\nRidge alpha selection:\n")
            for row in alpha_selection_rows:
                handle.write(
                    f"  {row['variant']}: "
                    f"alpha={float(row['chosen_alpha']):.4g}, "
                    f"outer test mean per-protein R2={float(row['outer_test_mean_per_protein_r2']):.4f}, "
                    f"outer test pair-weighted R2={float(row['outer_test_pair_weighted_r2']):.4f}\n"
                )

        if skipped:
            handle.write("\nSkipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")


def flush_outputs(
    csv_dir,
    summary_rows,
    per_protein_rows,
    split_rows,
    coefficient_rows,
    alpha_cv_rows,
    alpha_selection_rows,
    all_proteins,
    loaded_records,
    skipped,
    elapsed_seconds,
):
    """Persist the current partial or final outputs to disk."""
    leaderboard_rows = build_leaderboard_rows(summary_rows)
    write_csv(csv_dir / SUMMARY_CSV_NAME, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(csv_dir / LEADERBOARD_CSV_NAME, LEADERBOARD_FIELDNAMES, leaderboard_rows)
    write_csv(csv_dir / PER_PROTEIN_CSV_NAME, PER_PROTEIN_FIELDNAMES, per_protein_rows)
    write_csv(csv_dir / SPLIT_CSV_NAME, SPLIT_FIELDNAMES, split_rows)
    write_csv(csv_dir / COEFFICIENT_CSV_NAME, COEFFICIENT_FIELDNAMES, coefficient_rows)
    write_csv(csv_dir / ALPHA_CV_CSV_NAME, ALPHA_CV_FIELDNAMES, alpha_cv_rows)
    write_csv(csv_dir / ALPHA_SELECTION_CSV_NAME, ALPHA_SELECTION_FIELDNAMES, alpha_selection_rows)
    build_summary_text(
        csv_dir / SUMMARY_TXT_NAME,
        elapsed_seconds=elapsed_seconds,
        all_proteins=all_proteins,
        loaded_records=loaded_records,
        skipped=skipped,
        leaderboard_rows=leaderboard_rows,
        alpha_selection_rows=alpha_selection_rows,
    )


def main():
    t0 = time.time()
    pair_blocks_root_env = os.environ.get("PAIR_BLOCKS_ROOT")
    pdb_root_env = os.environ.get("PDB_ROOT")
    csv_root_env = os.environ.get("CSV_ROOT")

    if not pair_blocks_root_env or not pdb_root_env or not csv_root_env:
        raise ValueError("PAIR_BLOCKS_ROOT, PDB_ROOT, and CSV_ROOT must all be set.")

    pair_blocks_root = Path(pair_blocks_root_env).resolve()
    pdb_root = Path(pdb_root_env).resolve()
    csv_dir = Path(csv_root_env).resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))
    print(f"Pair blocks root: {pair_blocks_root}")
    print(f"PDB root:         {pdb_root}")
    print(f"Output CSV dir:   {csv_dir}")

    all_proteins, records, skipped = load_protein_records(pair_blocks_root, pdb_root)
    if not records:
        print("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    assign_size_bins(records, N_SIZE_BINS)
    train_records, test_records = split_records(records)
    split_rows = build_split_rows(train_records, test_records)

    print(f"Loaded proteins: {len(records)}")
    print(f"Train proteins:  {len(train_records)}")
    print(f"Test proteins:   {len(test_records)}")

    summary_rows = []
    per_protein_rows = []
    coefficient_rows = []
    alpha_cv_rows = []
    alpha_selection_rows = []

    flush_outputs(
        csv_dir=csv_dir,
        summary_rows=summary_rows,
        per_protein_rows=per_protein_rows,
        split_rows=split_rows,
        coefficient_rows=coefficient_rows,
        alpha_cv_rows=alpha_cv_rows,
        alpha_selection_rows=alpha_selection_rows,
        all_proteins=all_proteins,
        loaded_records=records,
        skipped=skipped,
        elapsed_seconds=float(time.time() - t0),
    )

    for variant in LINEAR_VARIANTS:
        print(f"\n[Run] {variant['name']}", flush=True)
        result = evaluate_variant(variant, train_records, test_records)
        summary_rows.append(result["summary_row"])
        per_protein_rows.extend(result["per_protein_rows"])
        coefficient_rows.extend(result["coefficient_rows"])
        flush_outputs(
            csv_dir=csv_dir,
            summary_rows=summary_rows,
            per_protein_rows=per_protein_rows,
            split_rows=split_rows,
            coefficient_rows=coefficient_rows,
            alpha_cv_rows=alpha_cv_rows,
            alpha_selection_rows=alpha_selection_rows,
            all_proteins=all_proteins,
            loaded_records=records,
            skipped=skipped,
            elapsed_seconds=float(time.time() - t0),
        )
        print(
            "  "
            f"test mean per-protein R2={result['summary_row']['test_mean_per_protein_r2']:.4f}, "
            f"test pair-weighted R2={result['summary_row']['test_pair_weighted_r2']:.4f}"
        )

    print(f"\n[CV] {RIDGE_VARIANT['name']}", flush=True)
    fold_family, cv_rows, chosen_alpha_row = run_ridge_alpha_sweep(RIDGE_VARIANT, train_records)
    alpha_cv_rows.extend(cv_rows)

    ridge_variant = dict(RIDGE_VARIANT)
    ridge_variant["alpha"] = float(chosen_alpha_row["alpha"])
    ridge_variant["selected_by_cv"] = True
    ridge_result = evaluate_variant(ridge_variant, train_records, test_records)
    summary_rows.append(ridge_result["summary_row"])
    per_protein_rows.extend(ridge_result["per_protein_rows"])
    coefficient_rows.extend(ridge_result["coefficient_rows"])
    alpha_selection_rows.append(
        {
            "variant": ridge_variant["name"],
            "model_family": ridge_variant["model_family"],
            "selected_metric": "mean_per_protein_r2",
            "chosen_alpha": float(chosen_alpha_row["alpha"]),
            "inner_cv_fold_family": fold_family,
            "inner_cv_mean_per_protein_r2": float(chosen_alpha_row["val_mean_per_protein_r2"]),
            "inner_cv_std_mean_per_protein_r2": float(
                np.std(
                    [
                        row["val_mean_per_protein_r2"]
                        for row in cv_rows
                        if row["fold_index"] >= 0
                        and math.isclose(
                            float(row["alpha"]),
                            float(chosen_alpha_row["alpha"]),
                            rel_tol=0.0,
                            abs_tol=1e-18,
                        )
                    ]
                )
            ),
            "inner_cv_mean_pair_weighted_r2": float(chosen_alpha_row["val_pair_weighted_r2"]),
            "inner_cv_mean_pair_weighted_mae": float(chosen_alpha_row["val_pair_weighted_mae"]),
            "outer_test_mean_per_protein_r2": float(
                ridge_result["summary_row"]["test_mean_per_protein_r2"]
            ),
            "outer_test_pair_weighted_r2": float(
                ridge_result["summary_row"]["test_pair_weighted_r2"]
            ),
            "outer_test_pair_weighted_mae": float(
                ridge_result["summary_row"]["test_pair_weighted_mae"]
            ),
            "outer_test_pair_weighted_rmse": float(
                ridge_result["summary_row"]["test_pair_weighted_rmse"]
            ),
            "outer_test_mean_bias": float(ridge_result["summary_row"]["test_mean_bias"]),
            "outer_test_n_nonzero_coefficients": int(
                ridge_result["summary_row"]["n_nonzero_coefficients"]
            ),
        }
    )
    flush_outputs(
        csv_dir=csv_dir,
        summary_rows=summary_rows,
        per_protein_rows=per_protein_rows,
        split_rows=split_rows,
        coefficient_rows=coefficient_rows,
        alpha_cv_rows=alpha_cv_rows,
        alpha_selection_rows=alpha_selection_rows,
        all_proteins=all_proteins,
        loaded_records=records,
        skipped=skipped,
        elapsed_seconds=float(time.time() - t0),
    )
    print(
        "  "
        f"chosen alpha={ridge_variant['alpha']:.4g}, "
        f"test mean per-protein R2={ridge_result['summary_row']['test_mean_per_protein_r2']:.4f}, "
        f"test pair-weighted R2={ridge_result['summary_row']['test_pair_weighted_r2']:.4f}"
    )

    print(f"\nSaved summary CSV -> {csv_dir / SUMMARY_CSV_NAME}")
    print(f"Saved leaderboard CSV -> {csv_dir / LEADERBOARD_CSV_NAME}")
    print(f"Saved per-protein CSV -> {csv_dir / PER_PROTEIN_CSV_NAME}")
    print(f"Saved split CSV -> {csv_dir / SPLIT_CSV_NAME}")
    print(f"Saved coefficient CSV -> {csv_dir / COEFFICIENT_CSV_NAME}")
    print(f"Saved alpha CV CSV -> {csv_dir / ALPHA_CV_CSV_NAME}")
    print(f"Saved alpha selection CSV -> {csv_dir / ALPHA_SELECTION_CSV_NAME}")
    print(f"Saved text summary -> {csv_dir / SUMMARY_TXT_NAME}")


if __name__ == "__main__":
    main()

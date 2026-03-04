#!/usr/bin/env python3
"""
diagnostic_residue_weighted_regression.py
=========================================
Diagnostic script for testing protein-size effects in layer-47 distance
regression across all valid proteins.

Key design choices:
1. Use each unordered residue pair exactly once via lower-triangle indices.
2. Use only one directional pair representation per pair: pb[i, j, :] for i > j.
3. Compare unweighted, protein-equal weighted, normalized-target, interaction,
   and strict scaling-only linear probes on one deterministic size-stratified
   80/20 protein split.

Outputs (inside CSV_ROOT/):
    diagnostic_residue_weighted_variant_summary.csv
    diagnostic_residue_weighted_per_protein.csv
    diagnostic_residue_weighted_split.csv
    diagnostic_residue_weighted_summary.txt
"""

import csv
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

from Bio.PDB import PDBParser


# ===========================================================================
# Configuration
# ===========================================================================

N_CHANNELS = 128
FEATURE_DIM = N_CHANNELS
LAYER = 47
TEST_SIZE = 0.2
RANDOM_STATE = 42
N_SIZE_BINS = 5
FLOAT_EPS = 1e-12


# ===========================================================================
# Directories (set via environment variables exported by sbatch files)
# ===========================================================================

PAIR_BLOCKS_ROOT = Path(os.environ["PAIR_BLOCKS_ROOT"])
PDB_ROOT = Path(os.environ["PDB_ROOT"])
CSV_DIR = Path(os.environ["CSV_ROOT"])


SUMMARY_CSV_NAME = "diagnostic_residue_weighted_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_residue_weighted_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_residue_weighted_split.csv"
SUMMARY_TXT_NAME = "diagnostic_residue_weighted_summary.txt"


VARIANTS = [
    {
        "name": "baseline_pair_weighted_abs",
        "target_mode": "absolute",
        "feature_mode": "base",
        "length_transform": "none",
        "uses_protein_equal_weighting": False,
    },
    {
        "name": "baseline_protein_equal_abs",
        "target_mode": "absolute",
        "feature_mode": "base",
        "length_transform": "none",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "baseline_protein_equal_norm_max",
        "target_mode": "normalized_max",
        "feature_mode": "base",
        "length_transform": "none",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "interaction_n_abs",
        "target_mode": "absolute",
        "feature_mode": "interaction",
        "length_transform": "n",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "interaction_sqrt_n_abs",
        "target_mode": "absolute",
        "feature_mode": "interaction",
        "length_transform": "sqrt_n",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "interaction_log_n_abs",
        "target_mode": "absolute",
        "feature_mode": "interaction",
        "length_transform": "log_n",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "interaction_n_sq_abs",
        "target_mode": "absolute",
        "feature_mode": "interaction",
        "length_transform": "n_sq",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "scaled_only_n_abs",
        "target_mode": "absolute",
        "feature_mode": "scaled_only",
        "length_transform": "n",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "scaled_only_sqrt_n_abs",
        "target_mode": "absolute",
        "feature_mode": "scaled_only",
        "length_transform": "sqrt_n",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "scaled_only_log_n_abs",
        "target_mode": "absolute",
        "feature_mode": "scaled_only",
        "length_transform": "log_n",
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "scaled_only_n_sq_abs",
        "target_mode": "absolute",
        "feature_mode": "scaled_only",
        "length_transform": "n_sq",
        "uses_protein_equal_weighting": True,
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
        if not entry.is_dir():
            continue
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
            if "CA" in residue:
                ca = residue["CA"]
                residues.append((
                    chain.get_id(),
                    residue.get_resname(),
                    residue.get_id()[1],
                ))
                coordinates.append(ca.get_coord())

    return {
        "residues": residues,
        "coordinates": np.array(coordinates, dtype=np.float32)
        if coordinates
        else np.empty((0, 3), dtype=np.float32),
    }


def compute_distance_matrix(coords):
    """Vectorized pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def build_feature_chunk(protein_record, layer):
    """Load one protein/layer pair block and extract lower-triangle features."""
    pb_path = get_pair_block_path(
        protein_record["protein_dir"], protein_record["protein_id"], layer
    )
    pb = np.load(str(pb_path))
    chunk = np.asarray(pb[protein_record["lt_i"], protein_record["lt_j"], :], dtype=np.float32)
    del pb
    return chunk


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
        return {d["protein_id"]: 0.0 for d in all_proteins}

    train_values = np.array(
        [transform_length_value(d["n_residues"], transform_name) for d in train_proteins],
        dtype=np.float64,
    )
    mean = float(np.mean(train_values))
    std = float(np.std(train_values))
    if std < FLOAT_EPS:
        std = 1.0

    lookup = {}
    for d in all_proteins:
        raw_value = transform_length_value(d["n_residues"], transform_name)
        lookup[d["protein_id"]] = float((raw_value - mean) / std)
    return lookup


def compute_split_pair_weights(split_proteins, normalize_mean=False):
    """Return per-pair weights, constant within each protein."""
    weights = []
    for d in split_proteins:
        weights.append(np.full(d["n_pairs"], 1.0 / d["n_pairs"], dtype=np.float64))
    if not weights:
        return np.empty((0,), dtype=np.float64)
    out = np.concatenate(weights)
    if normalize_mean and out.size > 0:
        out *= out.size / np.sum(out)
    return out


def compute_scaler_from_protein_stats(train_proteins, feature_mode, transform_lookup, pair_weighted):
    """Compute train-only mean/std using cached per-protein feature moments."""
    if feature_mode == "base":
        dim = FEATURE_DIM
    elif feature_mode == "interaction":
        dim = FEATURE_DIM * 2
    elif feature_mode == "scaled_only":
        dim = FEATURE_DIM
    else:
        raise ValueError(f"Unsupported feature mode: {feature_mode}")

    mean_acc = np.zeros(dim, dtype=np.float64)
    sq_acc = np.zeros(dim, dtype=np.float64)
    total_weight = 0.0

    for d in train_proteins:
        pid = d["protein_id"]
        base_mean = d["base_feature_mean"]
        base_sq_mean = d["base_feature_sq_mean"]
        g_val = transform_lookup[pid]

        if feature_mode == "base":
            feature_mean = base_mean
            feature_sq_mean = base_sq_mean
        elif feature_mode == "interaction":
            feature_mean = np.concatenate([base_mean, base_mean * g_val])
            feature_sq_mean = np.concatenate([base_sq_mean, base_sq_mean * (g_val ** 2)])
        else:
            feature_mean = base_mean * g_val
            feature_sq_mean = base_sq_mean * (g_val ** 2)

        weight = float(d["n_pairs"]) if pair_weighted else 1.0
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


def build_offsets(split_proteins):
    """Build contiguous row offsets for one split."""
    offsets = []
    start = 0
    for d in split_proteins:
        end = start + d["n_pairs"]
        offsets.append((start, end))
        start = end
    return offsets


def repeat_protein_scalar(split_proteins, lookup):
    """Broadcast a per-protein scalar across all of its pairs."""
    pieces = []
    for d in split_proteins:
        pieces.append(np.full(d["n_pairs"], lookup[d["protein_id"]], dtype=np.float32))
    if not pieces:
        return np.empty((0,), dtype=np.float32)
    return np.concatenate(pieces)


def build_variant_matrices(feature_mode, x_train_base, x_test_base, g_train_rows, g_test_rows):
    """Construct train/test matrices for one feature mode."""
    if feature_mode == "base":
        return x_train_base.copy(), x_test_base.copy()

    if feature_mode == "interaction":
        n_train = x_train_base.shape[0]
        n_test = x_test_base.shape[0]
        x_train = np.empty((n_train, FEATURE_DIM * 2), dtype=np.float32)
        x_test = np.empty((n_test, FEATURE_DIM * 2), dtype=np.float32)
        x_train[:, :FEATURE_DIM] = x_train_base
        x_test[:, :FEATURE_DIM] = x_test_base
        np.multiply(x_train_base, g_train_rows[:, None], out=x_train[:, FEATURE_DIM:])
        np.multiply(x_test_base, g_test_rows[:, None], out=x_test[:, FEATURE_DIM:])
        return x_train, x_test

    if feature_mode == "scaled_only":
        x_train = np.empty_like(x_train_base)
        x_test = np.empty_like(x_test_base)
        np.multiply(x_train_base, g_train_rows[:, None], out=x_train)
        np.multiply(x_test_base, g_test_rows[:, None], out=x_test)
        return x_train, x_test

    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def summarize_predictions(y_true, y_pred, split_proteins):
    """Compute pair-weighted, protein-weighted, and per-protein metrics."""
    protein_equal_weights = compute_split_pair_weights(split_proteins, normalize_mean=False)
    offsets = build_offsets(split_proteins)

    per_rows = []
    per_r2 = []
    negative_r2 = 0

    for d, (start, end) in zip(split_proteins, offsets):
        y_true_p = y_true[start:end]
        y_pred_p = y_pred[start:end]
        per_r2_val = compute_r2(y_true_p, y_pred_p)
        per_mae_val = compute_mae(y_true_p, y_pred_p)
        per_rmse_val = compute_rmse(y_true_p, y_pred_p)
        residual = y_true_p - y_pred_p

        if not math.isnan(per_r2_val):
            per_r2.append(per_r2_val)
            if per_r2_val < 0:
                negative_r2 += 1

        row = {
            "protein_id": d["protein_id"],
            "size_bin": d["size_bin"],
            "n_residues": d["n_residues"],
            "n_pairs": d["n_pairs"],
            "max_distance": d["max_distance"],
            "per_protein_r2": per_r2_val,
            "per_protein_mae": per_mae_val,
            "per_protein_rmse": per_rmse_val,
            "mean_true_distance": float(np.mean(y_true_p)),
            "std_true_distance": float(np.std(y_true_p)),
            "mean_predicted_distance": float(np.mean(y_pred_p)),
            "std_predicted_distance": float(np.std(y_pred_p)),
            "mean_residual": float(np.mean(residual)),
            "std_residual": float(np.std(residual)),
        }
        per_rows.append(row)

    return {
        "pair_weighted_r2": compute_r2(y_true, y_pred),
        "protein_weighted_r2": compute_r2(y_true, y_pred, sample_weight=protein_equal_weights),
        "pair_weighted_mae": compute_mae(y_true, y_pred),
        "pair_weighted_rmse": compute_rmse(y_true, y_pred),
        "mean_per_protein_r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
        "median_per_protein_r2": float(np.median(per_r2)) if per_r2 else float("nan"),
        "negative_r2_count": int(negative_r2),
        "negative_r2_fraction": (float(negative_r2 / len(per_r2)) if per_r2 else float("nan")),
        "per_rows": per_rows,
    }


def best_variant_name(summary_rows, prefix):
    """Return the best variant within a family by test mean per-protein R2."""
    candidates = [row for row in summary_rows if row["variant"].startswith(prefix)]
    if not candidates:
        return None
    best = max(
        candidates,
        key=lambda row: (-math.inf if math.isnan(row["test_mean_per_protein_r2"]) else row["test_mean_per_protein_r2"]),
    )
    return best["variant"]


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()

    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 72}")
    print("DIAGNOSTIC RESIDUE-WEIGHTED REGRESSION — layer 47, all valid proteins")
    print(f"{'=' * 72}")
    print(f"Pair blocks root:       {PAIR_BLOCKS_ROOT}")
    print(f"PDB root:               {PDB_ROOT}")
    print(f"Output CSV dir:         {csv_dir}")
    print(f"Layer:                  {LAYER}")
    print("Pair representation:    lower triangle only, single direction (128-dim)")
    print(f"Train/test split:       80/20 protein-level, size-stratified, seed={RANDOM_STATE}")
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
            if n_ca == 0:
                skipped.append((protein_id, "no CA atoms found in PDB"))
                print(f"[SKIP] {protein_id}: no CA atoms found in PDB")
                continue
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

            dist_matrix = compute_distance_matrix(ca_data["coordinates"])
            y_abs = np.asarray(dist_matrix[lt_i, lt_j], dtype=np.float32)
            max_distance = float(np.max(y_abs))
            if max_distance <= 0.0:
                skipped.append((protein_id, "max CA distance <= 0"))
                print(f"[SKIP] {protein_id}: max CA distance <= 0")
                continue

            protein_data.append({
                "protein_id": protein_id,
                "protein_dir": protein_dir,
                "n_residues": n_ca,
                "lt_i": lt_i,
                "lt_j": lt_j,
                "y_abs": y_abs,
                "y_norm": y_abs / max_distance,
                "n_pairs": int(y_abs.size),
                "max_distance": max_distance,
                "mean_distance": float(np.mean(y_abs)),
                "std_distance": float(np.std(y_abs)),
            })
            print(f"[OK]   {protein_id}: {n_ca} residues, {y_abs.size:,} lower-triangle pairs")

        except Exception as exc:
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {exc}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(protein_data)} proteins (skipped {len(skipped)}).")

    assign_size_bins(protein_data, N_SIZE_BINS)
    size_counts = {bin_idx: 0 for bin_idx in range(N_SIZE_BINS)}
    for d in protein_data:
        size_counts[d["size_bin"]] += 1

    # ------------------------------------------------------------------
    # Phase 2: Deterministic size-stratified train/test split
    # ------------------------------------------------------------------
    protein_indices = np.arange(len(protein_data))
    stratify_labels = np.array([d["size_bin"] for d in protein_data], dtype=int)

    train_indices, test_indices = train_test_split(
        protein_indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )

    train_index_set = set(int(i) for i in train_indices)
    test_index_set = set(int(i) for i in test_indices)
    train_proteins = [d for idx, d in enumerate(protein_data) if idx in train_index_set]
    test_proteins = [d for idx, d in enumerate(protein_data) if idx in test_index_set]

    print(f"\nTrain proteins: {len(train_proteins)}")
    print(f"Test proteins:  {len(test_proteins)}")

    for bin_idx in range(N_SIZE_BINS):
        train_bin = sum(1 for d in train_proteins if d["size_bin"] == bin_idx)
        test_bin = sum(1 for d in test_proteins if d["size_bin"] == bin_idx)
        print(
            f"  Size bin {bin_idx}: total={size_counts[bin_idx]}, "
            f"train={train_bin}, test={test_bin}"
        )

    # ------------------------------------------------------------------
    # Phase 3: Load layer 47 features and cache per-protein feature stats
    # ------------------------------------------------------------------
    print(f"\nLoading layer {LAYER} lower-triangle pair representations...")
    t_load = time.time()
    chunk_map = {}
    for d in protein_data:
        chunk = build_feature_chunk(d, LAYER)
        d["base_feature_mean"] = np.asarray(np.mean(chunk, axis=0), dtype=np.float64)
        d["base_feature_sq_mean"] = np.asarray(np.mean(chunk * chunk, axis=0), dtype=np.float64)
        chunk_map[d["protein_id"]] = chunk
    print(f"Loaded all layer-{LAYER} chunks in {time.time() - t_load:.1f}s")

    x_train_base = np.vstack([chunk_map[d["protein_id"]] for d in train_proteins]).astype(np.float32, copy=False)
    x_test_base = np.vstack([chunk_map[d["protein_id"]] for d in test_proteins]).astype(np.float32, copy=False)
    del chunk_map

    y_train_abs = np.concatenate([d["y_abs"] for d in train_proteins]).astype(np.float64, copy=False)
    y_test_abs = np.concatenate([d["y_abs"] for d in test_proteins]).astype(np.float64, copy=False)
    y_train_norm = np.concatenate([d["y_norm"] for d in train_proteins]).astype(np.float64, copy=False)
    y_test_norm = np.concatenate([d["y_norm"] for d in test_proteins]).astype(np.float64, copy=False)

    train_weights_model = compute_split_pair_weights(train_proteins, normalize_mean=True)

    print(f"Base train matrix: {x_train_base.shape}")
    print(f"Base test matrix:  {x_test_base.shape}")
    print(f"Train pairs:       {y_train_abs.size:,}")
    print(f"Test pairs:        {y_test_abs.size:,}")

    # ------------------------------------------------------------------
    # Phase 4: Persist split membership for reproducibility
    # ------------------------------------------------------------------
    split_rows = []
    for split_name, split_proteins in [("train", train_proteins), ("test", test_proteins)]:
        for d in split_proteins:
            split_rows.append({
                "protein_id": d["protein_id"],
                "split": split_name,
                "size_bin": d["size_bin"],
                "n_residues": d["n_residues"],
                "n_pairs": d["n_pairs"],
            })

    write_csv(
        csv_dir / SPLIT_CSV_NAME,
        ["protein_id", "split", "size_bin", "n_residues", "n_pairs"],
        split_rows,
    )
    print(f"Saved split CSV -> {csv_dir / SPLIT_CSV_NAME}")

    # ------------------------------------------------------------------
    # Phase 5: Run all variants
    # ------------------------------------------------------------------
    summary_rows = []
    per_protein_rows = []
    variant_rankings = {}

    for variant in VARIANTS:
        variant_name = variant["name"]
        target_mode = variant["target_mode"]
        feature_mode = variant["feature_mode"]
        length_transform = variant["length_transform"]
        use_protein_equal_weighting = variant["uses_protein_equal_weighting"]

        print(f"\n{'-' * 72}")
        print(f"Running variant: {variant_name}")
        print(
            f"  target_mode={target_mode}, feature_mode={feature_mode}, "
            f"length_transform={length_transform}, "
            f"protein_equal_weighting={use_protein_equal_weighting}"
        )
        t_variant = time.time()

        transform_lookup = build_zscore_lookup(train_proteins, protein_data, length_transform)
        g_train_rows = repeat_protein_scalar(train_proteins, transform_lookup)
        g_test_rows = repeat_protein_scalar(test_proteins, transform_lookup)

        pair_weighted_scaler = not use_protein_equal_weighting
        scaler_mean, scaler_scale = compute_scaler_from_protein_stats(
            train_proteins,
            feature_mode=feature_mode,
            transform_lookup=transform_lookup,
            pair_weighted=pair_weighted_scaler,
        )

        x_train, x_test = build_variant_matrices(
            feature_mode=feature_mode,
            x_train_base=x_train_base,
            x_test_base=x_test_base,
            g_train_rows=g_train_rows,
            g_test_rows=g_test_rows,
        )
        apply_scaler_in_place(x_train, scaler_mean, scaler_scale)
        apply_scaler_in_place(x_test, scaler_mean, scaler_scale)

        if target_mode == "absolute":
            y_train = y_train_abs
            y_test = y_test_abs
        elif target_mode == "normalized_max":
            y_train = y_train_norm
            y_test = y_test_norm
        else:
            raise ValueError(f"Unsupported target mode: {target_mode}")

        model = LinearRegression(copy_X=False)
        if use_protein_equal_weighting:
            model.fit(x_train, y_train, sample_weight=train_weights_model)
        else:
            model.fit(x_train, y_train)

        y_pred_train = np.asarray(model.predict(x_train), dtype=np.float64)
        y_pred_test = np.asarray(model.predict(x_test), dtype=np.float64)

        train_metrics = summarize_predictions(y_train, y_pred_train, train_proteins)
        test_metrics = summarize_predictions(y_test, y_pred_test, test_proteins)

        oracle_train = None
        oracle_test = None
        if target_mode == "normalized_max":
            oracle_pred_train = np.concatenate([
                y_pred_train[start:end] * d["max_distance"]
                for d, (start, end) in zip(train_proteins, build_offsets(train_proteins))
            ])
            oracle_pred_test = np.concatenate([
                y_pred_test[start:end] * d["max_distance"]
                for d, (start, end) in zip(test_proteins, build_offsets(test_proteins))
            ])
            oracle_train = summarize_predictions(y_train_abs, oracle_pred_train, train_proteins)
            oracle_test = summarize_predictions(y_test_abs, oracle_pred_test, test_proteins)

        for split_name, split_proteins, metrics_bundle in [
            ("train", train_proteins, train_metrics),
            ("test", test_proteins, test_metrics),
        ]:
            oracle_rows = None
            if target_mode == "normalized_max":
                oracle_rows = oracle_train["per_rows"] if split_name == "train" else oracle_test["per_rows"]

            split_variant_rows = []
            for idx, base_row in enumerate(metrics_bundle["per_rows"]):
                out_row = {
                    "variant": variant_name,
                    "split": split_name,
                    "protein_id": base_row["protein_id"],
                    "size_bin": base_row["size_bin"],
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
                    "per_protein_r2_oracle_abs": float("nan"),
                    "per_protein_mae_oracle_abs": float("nan"),
                    "per_protein_rmse_oracle_abs": float("nan"),
                    "mean_true_distance_oracle_abs": float("nan"),
                    "std_true_distance_oracle_abs": float("nan"),
                    "mean_predicted_distance_oracle_abs": float("nan"),
                    "std_predicted_distance_oracle_abs": float("nan"),
                    "mean_residual_oracle_abs": float("nan"),
                    "std_residual_oracle_abs": float("nan"),
                }
                if oracle_rows is not None:
                    oracle_row = oracle_rows[idx]
                    out_row.update({
                        "per_protein_r2_oracle_abs": oracle_row["per_protein_r2"],
                        "per_protein_mae_oracle_abs": oracle_row["per_protein_mae"],
                        "per_protein_rmse_oracle_abs": oracle_row["per_protein_rmse"],
                        "mean_true_distance_oracle_abs": oracle_row["mean_true_distance"],
                        "std_true_distance_oracle_abs": oracle_row["std_true_distance"],
                        "mean_predicted_distance_oracle_abs": oracle_row["mean_predicted_distance"],
                        "std_predicted_distance_oracle_abs": oracle_row["std_predicted_distance"],
                        "mean_residual_oracle_abs": oracle_row["mean_residual"],
                        "std_residual_oracle_abs": oracle_row["std_residual"],
                    })
                split_variant_rows.append(out_row)

            if split_name == "test":
                rankable = [
                    row for row in split_variant_rows if not math.isnan(float(row["per_protein_r2"]))
                ]
                variant_rankings[variant_name] = sorted(
                    rankable, key=lambda row: float(row["per_protein_r2"])
                )

            per_protein_rows.extend(split_variant_rows)

        summary_row = {
            "variant": variant_name,
            "target_mode": target_mode,
            "feature_mode": (
                "base128" if feature_mode == "base"
                else "interaction256" if feature_mode == "interaction"
                else "scaled_only128"
            ),
            "length_transform": length_transform,
            "uses_protein_equal_weighting": use_protein_equal_weighting,
            "n_train_proteins": len(train_proteins),
            "n_test_proteins": len(test_proteins),
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
            summary_row.update({
                "train_r2_oracle_rescaled_abs": oracle_train["pair_weighted_r2"],
                "test_r2_oracle_rescaled_abs": oracle_test["pair_weighted_r2"],
                "train_mae_oracle_rescaled_abs": oracle_train["pair_weighted_mae"],
                "test_mae_oracle_rescaled_abs": oracle_test["pair_weighted_mae"],
                "train_rmse_oracle_rescaled_abs": oracle_train["pair_weighted_rmse"],
                "test_rmse_oracle_rescaled_abs": oracle_test["pair_weighted_rmse"],
            })

        summary_rows.append(summary_row)

        print(
            f"  Test pooled R2={summary_row['test_r2_pair_weighted']:.4f}, "
            f"test protein-weighted R2={summary_row['test_r2_protein_weighted']:.4f}, "
            f"test mean per-protein R2={summary_row['test_mean_per_protein_r2']:.4f}"
        )
        print(f"  Completed in {time.time() - t_variant:.1f}s")

        del x_train, x_test, y_pred_train, y_pred_test
        del g_train_rows, g_test_rows

    # ------------------------------------------------------------------
    # Phase 6: Persist outputs
    # ------------------------------------------------------------------
    summary_fields = [
        "variant",
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
        "split",
        "protein_id",
        "size_bin",
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
        "per_protein_r2_oracle_abs",
        "per_protein_mae_oracle_abs",
        "per_protein_rmse_oracle_abs",
        "mean_true_distance_oracle_abs",
        "std_true_distance_oracle_abs",
        "mean_predicted_distance_oracle_abs",
        "std_predicted_distance_oracle_abs",
        "mean_residual_oracle_abs",
        "std_residual_oracle_abs",
    ]
    write_csv(csv_dir / PER_PROTEIN_CSV_NAME, per_protein_fields, per_protein_rows)
    print(f"Saved per-protein CSV -> {csv_dir / PER_PROTEIN_CSV_NAME}")

    # ------------------------------------------------------------------
    # Phase 7: Human-readable summary
    # ------------------------------------------------------------------
    summary_path = csv_dir / SUMMARY_TXT_NAME
    interaction_best = best_variant_name(summary_rows, "interaction_")
    scaled_only_best = best_variant_name(summary_rows, "scaled_only_")
    best_overall = max(
        summary_rows,
        key=lambda row: (-math.inf if math.isnan(row["test_mean_per_protein_r2"]) else row["test_mean_per_protein_r2"]),
    )["variant"]

    baseline_pair = next(row for row in summary_rows if row["variant"] == "baseline_pair_weighted_abs")
    baseline_protein = next(row for row in summary_rows if row["variant"] == "baseline_protein_equal_abs")
    baseline_norm = next(row for row in summary_rows if row["variant"] == "baseline_protein_equal_norm_max")

    with summary_path.open("w") as handle:
        handle.write("Diagnostic Residue-Weighted Regression — Layer 47\n")
        handle.write("=" * 56 + "\n\n")
        handle.write(f"Pair blocks root:   {PAIR_BLOCKS_ROOT}\n")
        handle.write(f"PDB root:           {PDB_ROOT}\n")
        handle.write(f"Output CSV dir:     {csv_dir}\n")
        handle.write(f"Layer:              {LAYER}\n")
        handle.write("Representation:     lower triangle only, single direction (pb[i,j,:] for i>j)\n")
        handle.write(f"Train/test split:   protein-level 80/20, size-stratified, seed={RANDOM_STATE}\n")
        handle.write(f"Size bins:          {N_SIZE_BINS} rank-based bins\n")
        handle.write("Large proteins:     retained (no >500 residue exclusion)\n\n")

        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(protein_data)}\n")
        handle.write(f"Proteins skipped:    {len(skipped)}\n")
        handle.write(f"Train proteins:      {len(train_proteins)}\n")
        handle.write(f"Test proteins:       {len(test_proteins)}\n")
        handle.write(f"Train pairs:         {y_train_abs.size:,}\n")
        handle.write(f"Test pairs:          {y_test_abs.size:,}\n\n")

        handle.write("Residue count summary:\n")
        train_res = [d["n_residues"] for d in train_proteins]
        test_res = [d["n_residues"] for d in test_proteins]
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
            f"{'Variant':<32} {'TestPairR2':>10} {'TestProtR2':>10} "
            f"{'MeanPP_R2':>10} {'MedPP_R2':>10} {'NegFrac':>10}\n"
        )
        for row in summary_rows:
            handle.write(
                "  "
                f"{row['variant']:<32} "
                f"{row['test_r2_pair_weighted']:>10.4f} "
                f"{row['test_r2_protein_weighted']:>10.4f} "
                f"{row['test_mean_per_protein_r2']:>10.4f} "
                f"{row['test_median_per_protein_r2']:>10.4f} "
                f"{row['test_negative_r2_fraction']:>10.4f}\n"
            )
        handle.write("\n")

        handle.write("Interpretation snapshot:\n")
        handle.write(
            "  Protein-equal weighting delta (mean per-protein test R2): "
            f"{baseline_protein['test_mean_per_protein_r2'] - baseline_pair['test_mean_per_protein_r2']:+.4f}\n"
        )
        handle.write(
            "  Normalized-target test mean per-protein R2 (normalized space): "
            f"{baseline_norm['test_mean_per_protein_r2']:.4f}\n"
        )
        handle.write(
            "  Normalized-target test pair-weighted oracle absolute R2: "
            f"{baseline_norm['test_r2_oracle_rescaled_abs']:.4f}\n"
        )
        handle.write(f"  Best interaction variant: {interaction_best}\n")
        handle.write(f"  Best scaling-only variant: {scaled_only_best}\n")
        handle.write(f"  Best overall variant by mean per-protein test R2: {best_overall}\n\n")

        for variant_name in [row["variant"] for row in summary_rows]:
            ranked = variant_rankings.get(variant_name, [])
            handle.write(f"Top 10 worst test proteins for {variant_name}:\n")
            for row in ranked[:10]:
                handle.write(
                    f"  {row['protein_id']}: R2={float(row['per_protein_r2']):.4f}, "
                    f"n_res={row['n_residues']}, size_bin={row['size_bin']}\n"
                )
            handle.write(f"Top 10 best test proteins for {variant_name}:\n")
            for row in reversed(ranked[-10:]):
                handle.write(
                    f"  {row['protein_id']}: R2={float(row['per_protein_r2']):.4f}, "
                    f"n_res={row['n_residues']}, size_bin={row['size_bin']}\n"
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

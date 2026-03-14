#!/usr/bin/env python3
"""
diagnostic_amino_acid_encodings_regression.py
=============================================
Layer-47 diagnostic regression script focused on feature augmentations and
size-aware linear-tree variants.

Requested variants (all with protein-equal training weights):
1. baseline_base256_lr
2. baseline_plus_aa_onehot296_lr
3. baseline_plus_diag512_lr
4. baseline_plus_size_linear_tree
5. baseline_plus_diag512_size_linear_tree

Outputs (inside CSV_ROOT/):
    diagnostic_amino_acid_encodings_variant_summary.csv
    diagnostic_amino_acid_encodings_per_protein.csv
    diagnostic_amino_acid_encodings_split.csv
    diagnostic_amino_acid_encodings_summary.txt
"""

import csv
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_X_y

try:
    from lineartree import LinearTreeRegressor
except Exception as exc:  # pragma: no cover - handled explicitly in main
    LinearTreeRegressor = None
    LINEARTREE_IMPORT_ERROR = exc
else:
    LINEARTREE_IMPORT_ERROR = None


# ===========================================================================
# Compatibility helpers
# ===========================================================================

def _apply_lineartree_sklearn_compat_patch():
    """Patch lineartree for sklearn versions where BaseEstimator._validate_data is absent."""
    if LinearTreeRegressor is None:
        return
    if hasattr(LinearTreeRegressor, "_validate_data"):
        return

    def _compat_validate_data(self, X, y=None, **kwargs):
        params = dict(kwargs)
        params.pop("reset", None)
        params.pop("validate_separately", None)

        if y is None:
            params.pop("multi_output", None)
            params.pop("y_numeric", None)
            try:
                return check_array(X, **params)
            except TypeError:
                if "force_all_finite" in params and "ensure_all_finite" not in params:
                    params["ensure_all_finite"] = params.pop("force_all_finite")
                return check_array(X, **params)

        try:
            return check_X_y(X, y, **params)
        except TypeError:
            if "force_all_finite" in params and "ensure_all_finite" not in params:
                params["ensure_all_finite"] = params.pop("force_all_finite")
            return check_X_y(X, y, **params)

    setattr(LinearTreeRegressor, "_validate_data", _compat_validate_data)


# ===========================================================================
# Configuration
# ===========================================================================

N_CHANNELS = 128
BASE_FEATURE_DIM = 2 * N_CHANNELS  # 256
AA_FEATURE_DIM = 40
DIAG_FEATURE_DIM = 4 * N_CHANNELS  # 512
LAYER = 47
TEST_SIZE = 0.2
RANDOM_STATE = 42
FLOAT_EPS = 1e-12

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

SUMMARY_CSV_NAME = "diagnostic_amino_acid_encodings_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_amino_acid_encodings_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_amino_acid_encodings_split.csv"
SUMMARY_TXT_NAME = "diagnostic_amino_acid_encodings_summary.txt"

VARIANTS = [
    {
        "name": "baseline_base256_lr",
        "model_type": "linear_regression",
        "feature_mode": "base",
        "feature_dim": BASE_FEATURE_DIM,
    },
    {
        "name": "baseline_plus_aa_onehot296_lr",
        "model_type": "linear_regression",
        "feature_mode": "base_plus_aa",
        "feature_dim": BASE_FEATURE_DIM + AA_FEATURE_DIM,
    },
    {
        "name": "baseline_plus_diag512_lr",
        "model_type": "linear_regression",
        "feature_mode": "base_plus_diag",
        "feature_dim": DIAG_FEATURE_DIM,
    },
    {
        "name": "baseline_plus_size_linear_tree",
        "model_type": "linear_tree",
        "feature_mode": "base_plus_size",
        "feature_dim": BASE_FEATURE_DIM + 1,
        "split_feature_index": BASE_FEATURE_DIM,
        "max_depth": 2,
    },
    {
        "name": "baseline_plus_diag512_size_linear_tree",
        "model_type": "linear_tree",
        "feature_mode": "base_plus_diag_plus_size",
        "feature_dim": DIAG_FEATURE_DIM + 1,
        "split_feature_index": DIAG_FEATURE_DIM,
        "max_depth": 2,
    },
]


# ===========================================================================
# Directories (set via environment variables exported by sbatch files)
# ===========================================================================

PAIR_BLOCKS_ROOT = Path(os.environ["PAIR_BLOCKS_ROOT"])
PDB_ROOT = Path(os.environ["PDB_ROOT"])
CSV_DIR = Path(os.environ["CSV_ROOT"])


# ===========================================================================
# Data loading helpers
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
    """Return path to pair block .npy file."""
    return protein_dir / f"{protein_id}_pair_block_{layer}.npy"


def discover_proteins(pair_blocks_root):
    """Find all protein directories under pair blocks root."""
    proteins = []
    for entry in sorted(pair_blocks_root.iterdir()):
        if entry.is_dir():
            proteins.append((entry.name, entry))
    return proteins


def extract_ca_coordinates(pdb_path):
    """Extract CA atom coordinates and residue metadata across all chains."""
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


def build_sorted_aa_pair_indices(residues, pair_i, pair_j):
    """Build per-pair AA indices after alphabetical sorting of residue names."""
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


# ===========================================================================
# Split and weighting helpers
# ===========================================================================

def split_global_records(records):
    """Create a single deterministic protein-level train/test split."""
    if len(records) < 2:
        raise ValueError("Need at least two proteins to split into train/test.")

    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_set = set(int(i) for i in train_idx)
    test_set = set(int(i) for i in test_idx)
    train_records = [record for idx, record in enumerate(records) if idx in train_set]
    test_records = [record for idx, record in enumerate(records) if idx in test_set]
    return train_records, test_records


def compute_split_pair_weights(split_records, normalize_mean=True):
    """Return per-pair sample weights so each protein contributes equally."""
    pieces = [
        np.full(record["n_pairs"], 1.0 / record["n_pairs"], dtype=np.float64)
        for record in split_records
    ]
    if not pieces:
        return np.empty((0,), dtype=np.float64)

    weights = np.concatenate(pieces)
    if normalize_mean and weights.size > 0:
        weights *= weights.size / np.sum(weights)
    return weights


def build_offsets(split_records):
    """Build row offsets for one split."""
    offsets = []
    start = 0
    for record in split_records:
        end = start + record["n_pairs"]
        offsets.append((start, end))
        start = end
    return offsets


# ===========================================================================
# Feature construction helpers
# ===========================================================================

def build_protein_feature_chunk(record, feature_mode):
    """Load one protein pair block and build feature rows for one variant mode."""
    pb_path = get_pair_block_path(record["protein_dir"], record["protein_id"], LAYER)
    pb = np.load(str(pb_path), mmap_mode="r")

    pair_i = record["pair_i"]
    pair_j = record["pair_j"]
    n_pairs = record["n_pairs"]

    if feature_mode == "base":
        chunk = np.empty((n_pairs, BASE_FEATURE_DIM), dtype=np.float32)
        chunk[:, :N_CHANNELS] = pb[pair_i, pair_j, :]
        chunk[:, N_CHANNELS:BASE_FEATURE_DIM] = pb[pair_j, pair_i, :]
        del pb
        return chunk

    if feature_mode == "base_plus_aa":
        chunk = np.zeros((n_pairs, BASE_FEATURE_DIM + AA_FEATURE_DIM), dtype=np.float32)
        chunk[:, :N_CHANNELS] = pb[pair_i, pair_j, :]
        chunk[:, N_CHANNELS:BASE_FEATURE_DIM] = pb[pair_j, pair_i, :]

        aa_first = record["aa_first_idx"]
        aa_second = record["aa_second_idx"]
        mask_first = aa_first >= 0
        mask_second = aa_second >= 0

        chunk[mask_first, BASE_FEATURE_DIM + aa_first[mask_first]] = 1.0
        chunk[mask_second, BASE_FEATURE_DIM + 20 + aa_second[mask_second]] = 1.0
        del pb
        return chunk

    if feature_mode == "base_plus_diag":
        chunk = np.empty((n_pairs, DIAG_FEATURE_DIM), dtype=np.float32)
        chunk[:, :N_CHANNELS] = pb[pair_i, pair_j, :]
        chunk[:, N_CHANNELS:BASE_FEATURE_DIM] = pb[pair_j, pair_i, :]
        chunk[:, BASE_FEATURE_DIM : BASE_FEATURE_DIM + N_CHANNELS] = pb[pair_i, pair_i, :]
        chunk[:, BASE_FEATURE_DIM + N_CHANNELS : DIAG_FEATURE_DIM] = pb[pair_j, pair_j, :]
        del pb
        return chunk

    if feature_mode == "base_plus_size":
        chunk = np.empty((n_pairs, BASE_FEATURE_DIM + 1), dtype=np.float32)
        chunk[:, :N_CHANNELS] = pb[pair_i, pair_j, :]
        chunk[:, N_CHANNELS:BASE_FEATURE_DIM] = pb[pair_j, pair_i, :]
        chunk[:, BASE_FEATURE_DIM] = float(record["n_residues"])
        del pb
        return chunk

    if feature_mode == "base_plus_diag_plus_size":
        chunk = np.empty((n_pairs, DIAG_FEATURE_DIM + 1), dtype=np.float32)
        chunk[:, :N_CHANNELS] = pb[pair_i, pair_j, :]
        chunk[:, N_CHANNELS:BASE_FEATURE_DIM] = pb[pair_j, pair_i, :]
        chunk[:, BASE_FEATURE_DIM : BASE_FEATURE_DIM + N_CHANNELS] = pb[pair_i, pair_i, :]
        chunk[:, BASE_FEATURE_DIM + N_CHANNELS : DIAG_FEATURE_DIM] = pb[pair_j, pair_j, :]
        chunk[:, DIAG_FEATURE_DIM] = float(record["n_residues"])
        del pb
        return chunk

    del pb
    raise ValueError(f"Unsupported feature mode: {feature_mode}")


def build_split_matrix(split_records, feature_mode):
    """Build dense matrix and target vector for one split and one feature mode."""
    n_rows = int(sum(record["n_pairs"] for record in split_records))
    feature_dim = next(v["feature_dim"] for v in VARIANTS if v["feature_mode"] == feature_mode)
    x = np.empty((n_rows, feature_dim), dtype=np.float32)
    y = np.empty((n_rows,), dtype=np.float64)

    offset = 0
    for record in split_records:
        chunk = build_protein_feature_chunk(record, feature_mode)
        end = offset + record["n_pairs"]
        x[offset:end, :] = chunk
        y[offset:end] = record["y_abs"]
        offset = end
        del chunk

    return x, y


def fit_and_apply_standardization(x_train, x_test, n_scale_features):
    """Standardize leading feature columns using train-only moments."""
    if n_scale_features <= 0:
        return

    mean = np.mean(x_train[:, :n_scale_features], axis=0, dtype=np.float64)
    scale = np.std(x_train[:, :n_scale_features], axis=0, dtype=np.float64)
    scale[scale < FLOAT_EPS] = 1.0

    x_train[:, :n_scale_features] -= mean
    x_train[:, :n_scale_features] /= scale
    x_test[:, :n_scale_features] -= mean
    x_test[:, :n_scale_features] /= scale


# ===========================================================================
# Metrics helpers
# ===========================================================================

def compute_r2(y_true, y_pred):
    """Compute vanilla R-squared."""
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


def summarize_predictions(y_true, y_pred, split_records):
    """Compute per-protein R2 rows and split-level summaries."""
    offsets = build_offsets(split_records)
    per_rows = []
    per_r2 = []
    negative_r2_count = 0

    for record, (start, end) in zip(split_records, offsets):
        y_true_p = y_true[start:end]
        y_pred_p = y_pred[start:end]
        per_r2_val = compute_r2(y_true_p, y_pred_p)

        if not math.isnan(per_r2_val):
            per_r2.append(per_r2_val)
            if per_r2_val < 0.0:
                negative_r2_count += 1

        error = y_pred_p - y_true_p
        per_rows.append(
            {
                "protein_id": record["protein_id"],
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
                "per_protein_r2": per_r2_val,
                "mean_true_distance": float(np.mean(y_true_p)),
                "std_true_distance": float(np.std(y_true_p)),
                "mean_predicted_distance": float(np.mean(y_pred_p)),
                "std_predicted_distance": float(np.std(y_pred_p)),
                "mean_error_pred_minus_true": float(np.mean(error)),
            }
        )

    return {
        "mean_per_protein_r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
        "median_per_protein_r2": float(np.median(per_r2)) if per_r2 else float("nan"),
        "negative_r2_count": int(negative_r2_count),
        "negative_r2_fraction": (
            float(negative_r2_count / len(per_r2)) if per_r2 else float("nan")
        ),
        "per_rows": per_rows,
    }


# ===========================================================================
# Variant execution
# ===========================================================================

def run_variant(variant, train_records, test_records):
    """Fit one variant and return summary + per-protein rows."""
    t0 = time.time()
    feature_mode = variant["feature_mode"]
    model_type = variant["model_type"]

    x_train, y_train = build_split_matrix(train_records, feature_mode)
    x_test, y_test = build_split_matrix(test_records, feature_mode)

    if model_type == "linear_regression":
        n_scale_features = x_train.shape[1]
    elif model_type == "linear_tree":
        n_scale_features = int(variant["split_feature_index"])
    else:
        raise ValueError(f"Unsupported model_type: {model_type}")

    fit_and_apply_standardization(x_train, x_test, n_scale_features=n_scale_features)
    train_weights = compute_split_pair_weights(train_records, normalize_mean=True)

    trained_leaf_count = float("nan")

    if model_type == "linear_regression":
        model = LinearRegression(copy_X=False)
        model.fit(x_train, y_train, sample_weight=train_weights)
    else:
        if LinearTreeRegressor is None:
            raise RuntimeError(
                "lineartree is not available but linear-tree variants were requested. "
                f"Import error: {LINEARTREE_IMPORT_ERROR}"
            )

        split_idx = int(variant["split_feature_index"])
        model = LinearTreeRegressor(
            base_estimator=LinearRegression(copy_X=False),
            max_depth=int(variant["max_depth"]),
            split_features=[split_idx],
            linear_features=list(range(split_idx)),
        )
        model.fit(x_train, y_train, sample_weight=train_weights)
        trained_leaf_count = float(len(getattr(model, "_leaves", {})))

    y_pred_train = np.asarray(model.predict(x_train), dtype=np.float64)
    y_pred_test = np.asarray(model.predict(x_test), dtype=np.float64)

    train_metrics = summarize_predictions(y_train, y_pred_train, train_records)
    test_metrics = summarize_predictions(y_test, y_pred_test, test_records)

    summary_row = {
        "variant": variant["name"],
        "model_type": model_type,
        "feature_mode": feature_mode,
        "feature_dim": variant["feature_dim"],
        "uses_protein_equal_weighting": True,
        "max_depth": int(variant.get("max_depth", 0)),
        "split_feature_index": int(variant.get("split_feature_index", -1)),
        "trained_leaf_count": trained_leaf_count,
        "n_train_proteins": len(train_records),
        "n_test_proteins": len(test_records),
        "n_train_pairs": int(y_train.size),
        "n_test_pairs": int(y_test.size),
        "train_mean_per_protein_r2": train_metrics["mean_per_protein_r2"],
        "test_mean_per_protein_r2": test_metrics["mean_per_protein_r2"],
        "test_median_per_protein_r2": test_metrics["median_per_protein_r2"],
        "test_negative_r2_count": test_metrics["negative_r2_count"],
        "test_negative_r2_fraction": test_metrics["negative_r2_fraction"],
        "elapsed_seconds": float(time.time() - t0),
    }

    per_rows = []
    for row in train_metrics["per_rows"]:
        out = dict(row)
        out["variant"] = variant["name"]
        out["model_type"] = model_type
        out["feature_mode"] = feature_mode
        out["split"] = "train"
        per_rows.append(out)

    test_rows = []
    for row in test_metrics["per_rows"]:
        out = dict(row)
        out["variant"] = variant["name"]
        out["model_type"] = model_type
        out["feature_mode"] = feature_mode
        out["split"] = "test"
        per_rows.append(out)
        test_rows.append(out)

    del x_train, x_test, y_train, y_test, y_pred_train, y_pred_test

    return summary_row, per_rows, test_rows


# ===========================================================================
# CSV helpers
# ===========================================================================

def format_value(value):
    """Stable CSV float formatting."""
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
    """Write CSV with stable field ordering."""
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

    if any(variant["model_type"] == "linear_tree" for variant in VARIANTS):
        if LinearTreeRegressor is None:
            print("ERROR: lineartree package is required for linear-tree variants.")
            print(f"Import error: {LINEARTREE_IMPORT_ERROR}")
            print("Install in environment: pip install --upgrade linear-tree")
            sys.exit(1)
        _apply_lineartree_sklearn_compat_patch()

    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 72}")
    print("DIAGNOSTIC AMINO ACID ENCODINGS REGRESSION — layer 47")
    print(f"{'=' * 72}")
    print(f"Pair blocks root:     {PAIR_BLOCKS_ROOT}")
    print(f"PDB root:             {PDB_ROOT}")
    print(f"Output CSV dir:       {csv_dir}")
    print(f"Layer:                {LAYER}")
    print(f"Train/test split:     80/20 protein-level, seed={RANDOM_STATE}")
    print("Training weights:     protein-equal (per-pair weight = 1 / n_pairs)")
    print(f"Variants:             {len(VARIANTS)}")

    # ------------------------------------------------------------------
    # Phase 1: Discover and validate proteins
    # ------------------------------------------------------------------
    all_proteins = discover_proteins(PAIR_BLOCKS_ROOT)
    print(f"\nDiscovered {len(all_proteins)} protein directories.")

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

            pair_i, pair_j = np.tril_indices(n_ca, k=-1)
            if pair_i.size == 0:
                skipped.append((protein_id, "no lower-triangle residue pairs"))
                print(f"[SKIP] {protein_id}: no lower-triangle residue pairs")
                continue

            dist_matrix = compute_distance_matrix(ca_data["coordinates"])
            y_abs = np.asarray(dist_matrix[pair_i, pair_j], dtype=np.float32)

            aa_first_idx, aa_second_idx = build_sorted_aa_pair_indices(
                ca_data["residues"], pair_i, pair_j
            )

            protein_data.append(
                {
                    "protein_id": protein_id,
                    "protein_dir": protein_dir,
                    "n_residues": n_ca,
                    "pair_i": pair_i,
                    "pair_j": pair_j,
                    "n_pairs": int(y_abs.size),
                    "y_abs": y_abs,
                    "aa_first_idx": aa_first_idx,
                    "aa_second_idx": aa_second_idx,
                }
            )
            print(
                f"[OK]   {protein_id}: {n_ca} residues, {y_abs.size:,} lower-triangle pairs"
            )

        except Exception as exc:
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {exc}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(protein_data)} proteins (skipped {len(skipped)}).")

    # ------------------------------------------------------------------
    # Phase 2: Build a single global train/test split
    # ------------------------------------------------------------------
    train_records, test_records = split_global_records(protein_data)
    print(
        f"Global split: {len(train_records)} train / {len(test_records)} test proteins"
    )

    split_rows = []
    for record in train_records:
        split_rows.append(
            {
                "scope": "global_shared",
                "protein_id": record["protein_id"],
                "split": "train",
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
            }
        )
    for record in test_records:
        split_rows.append(
            {
                "scope": "global_shared",
                "protein_id": record["protein_id"],
                "split": "test",
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
            }
        )

    split_fields = ["scope", "protein_id", "split", "n_residues", "n_pairs"]
    write_csv(csv_dir / SPLIT_CSV_NAME, split_fields, split_rows)
    print(f"Saved split CSV -> {csv_dir / SPLIT_CSV_NAME}")

    # ------------------------------------------------------------------
    # Phase 3: Run requested variants
    # ------------------------------------------------------------------
    summary_rows = []
    per_protein_rows = []
    ranking_lookup = {}

    for variant in VARIANTS:
        print(f"\n{'-' * 72}")
        print(
            f"Running variant: {variant['name']} "
            f"(type={variant['model_type']}, feature_mode={variant['feature_mode']})"
        )
        summary_row, variant_per_rows, variant_test_rows = run_variant(
            variant, train_records, test_records
        )
        summary_rows.append(summary_row)
        per_protein_rows.extend(variant_per_rows)

        rankable = [
            row for row in variant_test_rows if not math.isnan(float(row["per_protein_r2"]))
        ]
        ranking_lookup[variant["name"]] = sorted(
            rankable, key=lambda row: float(row["per_protein_r2"])
        )
        print(
            f"  test mean per-protein R2={summary_row['test_mean_per_protein_r2']:.4f}, "
            f"test median={summary_row['test_median_per_protein_r2']:.4f}, "
            f"neg_frac={summary_row['test_negative_r2_fraction']:.4f}, "
            f"elapsed={summary_row['elapsed_seconds']:.1f}s"
        )

    # ------------------------------------------------------------------
    # Phase 4: Persist outputs
    # ------------------------------------------------------------------
    summary_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "feature_dim",
        "uses_protein_equal_weighting",
        "max_depth",
        "split_feature_index",
        "trained_leaf_count",
        "n_train_proteins",
        "n_test_proteins",
        "n_train_pairs",
        "n_test_pairs",
        "train_mean_per_protein_r2",
        "test_mean_per_protein_r2",
        "test_median_per_protein_r2",
        "test_negative_r2_count",
        "test_negative_r2_fraction",
        "elapsed_seconds",
    ]
    write_csv(csv_dir / SUMMARY_CSV_NAME, summary_fields, summary_rows)
    print(f"\nSaved summary CSV -> {csv_dir / SUMMARY_CSV_NAME}")

    per_protein_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "split",
        "protein_id",
        "n_residues",
        "n_pairs",
        "per_protein_r2",
        "mean_true_distance",
        "std_true_distance",
        "mean_predicted_distance",
        "std_predicted_distance",
        "mean_error_pred_minus_true",
    ]
    write_csv(csv_dir / PER_PROTEIN_CSV_NAME, per_protein_fields, per_protein_rows)
    print(f"Saved per-protein CSV -> {csv_dir / PER_PROTEIN_CSV_NAME}")

    # ------------------------------------------------------------------
    # Phase 5: Human-readable summary
    # ------------------------------------------------------------------
    summary_path = csv_dir / SUMMARY_TXT_NAME
    best_variant = max(
        summary_rows,
        key=lambda row: (
            -math.inf
            if math.isnan(float(row["test_mean_per_protein_r2"]))
            else float(row["test_mean_per_protein_r2"])
        ),
    )

    with summary_path.open("w") as handle:
        handle.write("Diagnostic Amino Acid Encodings Regression — Layer 47\n")
        handle.write("=" * 64 + "\n\n")
        handle.write(f"Pair blocks root:   {PAIR_BLOCKS_ROOT}\n")
        handle.write(f"PDB root:           {PDB_ROOT}\n")
        handle.write(f"Output CSV dir:     {csv_dir}\n")
        handle.write(f"Layer:              {LAYER}\n")
        handle.write(
            "Representation:     one lower-triangle row per unordered pair\n"
        )
        handle.write(
            "Training weighting: protein-equal (per-pair weight = 1 / n_pairs)\n"
        )
        handle.write(f"Train/test split:   protein-level 80/20, seed={RANDOM_STATE}\n\n")

        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(protein_data)}\n")
        handle.write(f"Proteins skipped:    {len(skipped)}\n")
        handle.write(f"Train proteins:      {len(train_records)}\n")
        handle.write(f"Test proteins:       {len(test_records)}\n\n")

        handle.write("Variant comparison:\n")
        handle.write(
            "  "
            f"{'Variant':<45} {'FeatDim':>7} {'TestMeanPP_R2':>14} "
            f"{'TestMedian':>11} {'NegFrac':>10}\n"
        )
        for row in summary_rows:
            handle.write(
                "  "
                f"{row['variant']:<45} "
                f"{int(row['feature_dim']):>7d} "
                f"{float(row['test_mean_per_protein_r2']):>14.4f} "
                f"{float(row['test_median_per_protein_r2']):>11.4f} "
                f"{float(row['test_negative_r2_fraction']):>10.4f}\n"
            )
        handle.write("\n")

        handle.write(
            "Best variant by test mean per-protein R2: "
            f"{best_variant['variant']} ({float(best_variant['test_mean_per_protein_r2']):.4f})\n\n"
        )

        for variant in VARIANTS:
            variant_name = variant["name"]
            ranked = ranking_lookup.get(variant_name, [])
            handle.write(f"Top 10 worst test proteins for {variant_name}:\n")
            for row in ranked[:10]:
                handle.write(
                    f"  {row['protein_id']}: R2={float(row['per_protein_r2']):.4f}, "
                    f"n_res={int(row['n_residues'])}, "
                    f"mean(pred-true)={float(row['mean_error_pred_minus_true']):+.4f}\n"
                )
            handle.write(f"Top 10 best test proteins for {variant_name}:\n")
            for row in reversed(ranked[-10:]):
                handle.write(
                    f"  {row['protein_id']}: R2={float(row['per_protein_r2']):.4f}, "
                    f"n_res={int(row['n_residues'])}, "
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

#!/usr/bin/env python3
"""
train_test_separated_multi_protein_regression.py
================================================
Protein-level train/test split variant of cluster_multi_protein_regression.py.

Key difference from cluster_multi_protein_regression.py:
    Instead of pooling all residue pairs and splitting randomly (which leaks
    information because pairs from the same protein appear in both train and
    test), this script splits at the PROTEIN level:
        - ~80% of proteins go into training
        - ~20% of proteins go into testing
    The model never sees any pairs from test proteins during training.

Cluster layout (same as cluster_multi_protein_regression.py):
  - PDB files are under:   ROOT_DIR/Structures/my_outputs_align_{id}_representations_load_from_saved_layer_47/predictions/
  - Pair blocks are under: ROOT_DIR/{protein_id}/{protein_id}_pair_block_{layer}.npy
  - Output CSVs go to:     ROOT_DIR/csv_files/

Dependencies (all pip-installable):
    pip install numpy scipy matplotlib scikit-learn biopython

Outputs (inside ROOT_DIR/csv_files/):
    protein_split_regression_r2.csv              — per-layer R2 (train/test), MAE, RMSE, per-protein R2
    protein_split_regression_coefficients.csv    — model coefficients per layer
    protein_split_predicted_distances.csv        — actual vs predicted distances (last layer)
    protein_split_regression_summary.txt         — human-readable summary of key statistics
"""

import csv
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from Bio.PDB import PDBParser


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48          # AlphaFold pair representation layers
N_CHANNELS = 128       # channels per pair representation direction
FEATURE_DIM = 2 * N_CHANNELS  # 256 (concatenated [i,j] and [j,i])
TEST_SIZE = 0.2        # fraction of PROTEINS held out for testing
RANDOM_STATE = 42

# ===========================================================================
# Hardcoded directories
# ===========================================================================

# Auto-detect scratch directory from Slurm environment, with fallback.
# FLAG: Update this path if your root directory is not on scratch.
ROOT_DIR = Path(os.environ.get("SCRATCH", "/path/to/your/root/directory"))
CSV_DIR  = ROOT_DIR / "csv_files"

# Subdirectory inside ROOT_DIR that contains PDB structures.
# FLAG: If the Structures directory is renamed or moved, update here.
STRUCTURES_DIR = ROOT_DIR / "Structures"


# ===========================================================================
# Helper functions — data loading
# ===========================================================================

# FLAG: This function defines the PDB filename pattern.
# PDB path: ROOT_DIR/Structures/my_outputs_align_{protein_id}_representations_load_from_saved_layer_47/predictions/{protein_id}_model_1_ptm_unrelaxed.pdb
# If the naming convention changes, update the pattern here.
def find_pdb_file(protein_id):
    """Locate the predicted PDB file for a protein.

    Expected path:
        STRUCTURES_DIR/my_outputs_align_{protein_id}_representations_load_from_saved_layer_47/
            predictions/{protein_id}_model_1_ptm_unrelaxed.pdb

    Returns the Path object, or None if not found.
    """
    pdb_path = (
        STRUCTURES_DIR
        / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions"
        / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    )
    return pdb_path if pdb_path.exists() else None


# FLAG: This function defines the pair block path pattern.
# Pair block path: ROOT_DIR/{protein_id}/{protein_id}_pair_block_{layer}.npy
# If the naming convention or directory structure changes, update here.
def get_pair_block_path(protein_dir, protein_id, layer):
    """Return the path to a pair block .npy file for a given layer.

    Expected: <protein_dir>/<protein_id>_pair_block_<layer>.npy
    (pair blocks live directly in the protein directory, no pair_blocks/ subdirectory)
    """
    return protein_dir / f"{protein_id}_pair_block_{layer}.npy"


# FLAG: This function discovers protein directories.
# It lists subdirectories of ROOT_DIR, excluding 'Structures' and 'csv_files'.
# If the directory layout changes (e.g. nested folders, manifest file), update here.
def discover_proteins(root_dir):
    """Find all protein subdirectories under the given root.

    Returns a sorted list of (protein_id, protein_dir) tuples.
    Skips 'Structures', 'csv_files', and any non-directory entries.
    """
    skip_names = {"Structures", "csv_files"}
    proteins = []
    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in skip_names:
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
            het_flag = residue.get_id()[0]
            if het_flag == "W":
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
        "coordinates": np.array(coordinates) if coordinates else np.empty((0, 3)),
    }


def compute_distance_matrix(coords):
    """Vectorized pairwise Euclidean distance matrix.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 3)

    Returns
    -------
    np.ndarray, shape (n, n) -- symmetric, zero diagonal.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


# ===========================================================================
# Main
# ===========================================================================

def main():
    root_dir = ROOT_DIR.resolve()
    n_layers = N_LAYERS

    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"Root directory:     {root_dir}")
    print(f"Structures dir:     {STRUCTURES_DIR}")
    print(f"Output CSV dir:     {csv_dir}")
    print(f"Layers:             {n_layers}")
    print(f"Split strategy:     PROTEIN-LEVEL (test_size={TEST_SIZE})")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------
    all_proteins = discover_proteins(root_dir)
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
            if n_ca == 0:
                skipped.append((protein_id, "no CA atoms found in PDB"))
                print(f"[SKIP] {protein_id}: no CA atoms found in PDB")
                continue

            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = pb0.shape[0]
            del pb0

            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                print(f"[SKIP] {protein_id}: CA count {n_ca} != pair block dim {pb_dim}")
                continue

            dist_matrix = compute_distance_matrix(ca_data["coordinates"])

            struct_i, struct_j = np.triu_indices(n_ca, k=1)
            y_protein = dist_matrix[struct_i, struct_j]

            protein_data.append({
                "protein_id": protein_id,
                "protein_dir": protein_dir,
                "n_residues": n_ca,
                "pb_i": struct_i,
                "pb_j": struct_j,
                "y": y_protein,
            })
            print(f"[OK]   {protein_id}: {n_ca} residues, {len(y_protein):,} pairs")

        except Exception as e:
            skipped.append((protein_id, f"unexpected error -- {e}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {e}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    n_proteins = len(protein_data)

    # ------------------------------------------------------------------
    # Phase 2: Protein-level train/test split
    # ------------------------------------------------------------------
    protein_indices = np.arange(n_proteins)

    # Determine number of test proteins (at least 1)
    n_test = max(1, int(round(n_proteins * TEST_SIZE)))
    n_train = n_proteins - n_test

    # If we only have 1 protein, we can't do a protein-level split
    if n_proteins < 2:
        print("\nERROR: Need at least 2 proteins for protein-level train/test split. Exiting.")
        sys.exit(1)

    train_idx, test_idx = train_test_split(
        protein_indices, test_size=n_test, random_state=RANDOM_STATE
    )

    train_proteins = [protein_data[i] for i in sorted(train_idx)]
    test_proteins = [protein_data[i] for i in sorted(test_idx)]

    train_ids = [d["protein_id"] for d in train_proteins]
    test_ids = [d["protein_id"] for d in test_proteins]

    # Pool targets per split
    y_train = np.concatenate([d["y"] for d in train_proteins])
    y_test = np.concatenate([d["y"] for d in test_proteins])

    n_train_pairs = len(y_train)
    n_test_pairs = len(y_test)
    n_total_pairs = n_train_pairs + n_test_pairs

    # Compute per-protein offsets within their respective split arrays
    train_offsets = []
    offset = 0
    for d in train_proteins:
        n = len(d["y"])
        train_offsets.append((d["protein_id"], offset, offset + n))
        offset += n

    test_offsets = []
    offset = 0
    for d in test_proteins:
        n = len(d["y"])
        test_offsets.append((d["protein_id"], offset, offset + n))
        offset += n

    print(f"\n{'='*60}")
    print(f"PROTEIN-LEVEL SPLIT")
    print(f"{'='*60}")
    print(f"  Train proteins ({n_train}): {', '.join(train_ids)}")
    print(f"  Test proteins  ({n_test}):  {', '.join(test_ids)}")
    print(f"  Train pairs: {n_train_pairs:,}")
    print(f"  Test pairs:  {n_test_pairs:,}")
    print(f"  Total pairs: {n_total_pairs:,}")
    print(f"  Actual test fraction: {n_test_pairs / n_total_pairs:.3f} "
          f"(proteins: {n_test}/{n_proteins} = {n_test/n_proteins:.3f})")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Phase 3: Layer-wise regression
    # ------------------------------------------------------------------
    results = []
    all_coefs = []
    last_layer_preds_test = None
    last_layer_preds_train = None

    all_protein_ids = train_ids + test_ids

    for layer in range(n_layers):
        try:
            # Build training feature matrix
            X_train_chunks = []
            for d in train_proteins:
                pb_path = get_pair_block_path(d["protein_dir"], d["protein_id"], layer)
                pb = np.load(str(pb_path))
                X_chunk = np.concatenate([
                    pb[d["pb_i"], d["pb_j"], :],
                    pb[d["pb_j"], d["pb_i"], :],
                ], axis=1)
                X_train_chunks.append(X_chunk)
                del pb

            X_train_raw = np.vstack(X_train_chunks)
            del X_train_chunks

            # Build test feature matrix
            X_test_chunks = []
            for d in test_proteins:
                pb_path = get_pair_block_path(d["protein_dir"], d["protein_id"], layer)
                pb = np.load(str(pb_path))
                X_chunk = np.concatenate([
                    pb[d["pb_i"], d["pb_j"], :],
                    pb[d["pb_j"], d["pb_i"], :],
                ], axis=1)
                X_test_chunks.append(X_chunk)
                del pb

            X_test_raw = np.vstack(X_test_chunks)
            del X_test_chunks

            # Scale features
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_train_raw)
            X_test = scaler.transform(X_test_raw)

            # Fit model
            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_train = model.predict(X_train)
            y_pred_test = model.predict(X_test)

            r2_train = r2_score(y_train, y_pred_train)
            r2_test = r2_score(y_test, y_pred_test)
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = math.sqrt(np.mean((y_test - y_pred_test) ** 2))

            # Per-protein R2 (test proteins only — the meaningful metric)
            per_protein_r2 = {}
            for pid, start, end in test_offsets:
                if end - start > 1:
                    per_protein_r2[pid] = r2_score(
                        y_test[start:end], y_pred_test[start:end]
                    )

            # Per-protein R2 on training set (for diagnostics)
            for pid, start, end in train_offsets:
                if end - start > 1:
                    per_protein_r2[f"{pid}_train"] = r2_score(
                        y_train[start:end], y_pred_train[start:end]
                    )

            results.append({
                "r2_test": r2_test, "r2_train": r2_train,
                "mae_test": mae, "rmse_test": rmse,
                "per_protein_r2": per_protein_r2,
            })
            print(f"Layer {layer:2d}  R2_test={r2_test:.4f}  R2_train={r2_train:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

            coef_row = {"layer": layer, "intercept": float(model.intercept_)}
            coef_row.update({f"coef_{i}": float(c) for i, c in enumerate(model.coef_)})
            all_coefs.append(coef_row)

            if layer == n_layers - 1:
                last_layer_preds_train = y_pred_train
                last_layer_preds_test = y_pred_test

            del X_train_raw, X_test_raw, X_train, X_test

        except Exception as e:
            print(f"Layer {layer:2d}  ERROR: {e}")
            traceback.print_exc()
            results.append({"r2_test": float("nan"), "r2_train": float("nan"),
                            "mae_test": float("nan"), "rmse_test": float("nan"),
                            "per_protein_r2": {}})
            all_coefs.append({"layer": layer, "intercept": 0.0})

    # ------------------------------------------------------------------
    # Phase 4: Save CSV outputs
    # ------------------------------------------------------------------

    # CSV 1: Layer metrics
    r2_csv = csv_dir / "protein_split_regression_r2.csv"
    # Columns for per-protein R2: test proteins (main), plus train proteins (diagnostic)
    per_prot_test_cols = [f"r2_test_{pid}" for pid in test_ids]
    per_prot_train_cols = [f"r2_train_{pid}" for pid in train_ids]
    with open(r2_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            ["layer", "r2_test", "r2_train", "mae_test", "rmse_test",
             "n_train_proteins", "n_test_proteins",
             "n_train_pairs", "n_test_pairs"]
            + per_prot_test_cols + per_prot_train_cols
        )
        for layer, res in enumerate(results):
            row = [layer, f"{res['r2_test']:.6f}", f"{res['r2_train']:.6f}",
                   f"{res['mae_test']:.4f}", f"{res['rmse_test']:.4f}",
                   n_train, n_test, n_train_pairs, n_test_pairs]
            for pid in test_ids:
                val = res["per_protein_r2"].get(pid, float("nan"))
                row.append(f"{val:.6f}")
            for pid in train_ids:
                val = res["per_protein_r2"].get(f"{pid}_train", float("nan"))
                row.append(f"{val:.6f}")
            writer.writerow(row)
    print(f"\nSaved metrics CSV -> {r2_csv}")

    # CSV 2: Coefficients
    coef_csv = csv_dir / "protein_split_regression_coefficients.csv"
    fieldnames = ["layer", "intercept"] + [f"coef_{i}" for i in range(FEATURE_DIM)]
    with open(coef_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_coefs)
    print(f"Saved coefficients CSV -> {coef_csv}")

    # CSV 3: Predicted distances (last layer)
    if last_layer_preds_test is not None and last_layer_preds_train is not None:
        pred_csv = csv_dir / "protein_split_predicted_distances.csv"
        with open(pred_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "protein_id", "struct_i", "struct_j",
                "actual_distance_A", "predicted_distance_A", "split",
            ])
            # Training proteins
            for pid, start, end in train_offsets:
                d = next(x for x in train_proteins if x["protein_id"] == pid)
                for k in range(end - start):
                    writer.writerow([
                        pid,
                        int(d["pb_i"][k]),
                        int(d["pb_j"][k]),
                        f"{y_train[start + k]:.4f}",
                        f"{last_layer_preds_train[start + k]:.4f}",
                        "train",
                    ])
            # Test proteins
            for pid, start, end in test_offsets:
                d = next(x for x in test_proteins if x["protein_id"] == pid)
                for k in range(end - start):
                    writer.writerow([
                        pid,
                        int(d["pb_i"][k]),
                        int(d["pb_j"][k]),
                        f"{y_test[start + k]:.4f}",
                        f"{last_layer_preds_test[start + k]:.4f}",
                        "test",
                    ])
        print(f"Saved predicted distances CSV -> {pred_csv}")
    else:
        print("WARNING: Last layer failed; predicted distances CSV not saved.")

    # Summary text file
    summary_path = csv_dir / "protein_split_regression_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Protein-Level Split Linear Regression Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Split strategy:  PROTEIN-LEVEL\n")
        f.write(f"Random state:    {RANDOM_STATE}\n")
        f.write(f"Feature dim:     {FEATURE_DIM}\n")
        f.write(f"Layers:          {n_layers}\n\n")

        f.write(f"Train proteins ({n_train}):\n")
        for pid, start, end in train_offsets:
            d = next(x for x in train_proteins if x["protein_id"] == pid)
            f.write(f"  {pid}: {d['n_residues']} residues, {end - start:,} pairs\n")
        f.write(f"\nTest proteins ({n_test}):\n")
        for pid, start, end in test_offsets:
            d = next(x for x in test_proteins if x["protein_id"] == pid)
            f.write(f"  {pid}: {d['n_residues']} residues, {end - start:,} pairs\n")

        f.write(f"\nTrain pairs:     {n_train_pairs:,}\n")
        f.write(f"Test pairs:      {n_test_pairs:,}\n")
        f.write(f"Total pairs:     {n_total_pairs:,}\n")
        f.write(f"Actual test fraction: {n_test_pairs / n_total_pairs:.3f} "
                f"(proteins: {n_test}/{n_proteins} = {n_test/n_proteins:.3f})\n\n")

        valid = [(i, r["r2_test"]) for i, r in enumerate(results) if not math.isnan(r["r2_test"])]
        if valid:
            best_layer, best_r2 = max(valid, key=lambda x: x[1])
            f.write(f"Best layer:      {best_layer} (R2_test = {best_r2:.6f})\n")
            last = results[-1]
            f.write(f"Last layer (47): R2_test={last['r2_test']:.6f}  R2_train={last['r2_train']:.6f}  "
                    f"MAE={last['mae_test']:.4f}  RMSE={last['rmse_test']:.4f}\n\n")
            f.write("Per-protein R2 (test set) at last layer:\n")
            for pid in test_ids:
                val = last["per_protein_r2"].get(pid, float("nan"))
                f.write(f"  {pid}: {val:.6f}\n")
            f.write("\nPer-protein R2 (train set) at last layer:\n")
            for pid in train_ids:
                val = last["per_protein_r2"].get(f"{pid}_train", float("nan"))
                f.write(f"  {pid}: {val:.6f}\n")

        f.write(f"\nLayer-by-layer R2 (test):\n")
        for i, res in enumerate(results):
            f.write(f"  Layer {i:2d}: {res['r2_test']:.6f}\n")

    print(f"Saved summary    -> {summary_path}")


if __name__ == "__main__":
    main()

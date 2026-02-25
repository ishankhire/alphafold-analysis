#!/usr/bin/env python3
"""
cluster_multi_protein_regression.py
====================================
ICE cluster variant of multi_protein_regression.py. Same regression logic
but adapted for the cluster's $SCRATCH directory layout:
  - PDB files are under:   ROOT_DIR/Structures/my_outputs_align_{id}_representations_load_from_saved_layer_47/predictions/
  - Pair blocks are under: ROOT_DIR/{protein_id}/{protein_id}_pair_block_{layer}.npy
  - Output CSVs go to:     ROOT_DIR/csv_files/

For the local version (project repo layout), see multi_protein_regression.py.

All helper functions are defined inline (no lib/ imports).

Dependencies (all pip-installable):
    pip install numpy scipy matplotlib scikit-learn biopython

Outputs (inside ROOT_DIR/csv_files/):
    layer_regression_r2_predicted.csv          — per-layer R2 (train/test), MAE, RMSE, per-protein R2
    layer_regression_coefficients_predicted.csv — model coefficients per layer
    predicted_distances_last_layer_predicted.csv — actual vs predicted distances (last layer)
    regression_summary.txt                     — human-readable summary of key statistics
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
TEST_SIZE = 0.2
RANDOM_STATE = 42

# ===========================================================================
# Hardcoded directories
# ===========================================================================

# Auto-detect scratch directory from Slurm environment, with fallback.
# ⚠️ FLAG: Update this path if your root directory is not on scratch.
ROOT_DIR = Path(os.environ.get("SCRATCH", "/path/to/your/root/directory"))
CSV_DIR  = ROOT_DIR / "csv_files"

# Subdirectory inside ROOT_DIR that contains PDB structures.
# ⚠️ FLAG: If the Structures directory is renamed or moved, update here.
STRUCTURES_DIR = ROOT_DIR / "Structures"


# ===========================================================================
# Helper functions — data loading
# ===========================================================================

# ⚠️ FLAG: This function defines the PDB filename pattern.
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

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------
    # FLAG: discover_proteins scans ROOT_DIR for subdirectories.
    # It skips 'Structures' and 'csv_files'.
    # If the directory layout changes, update discover_proteins().
    all_proteins = discover_proteins(root_dir)
    print(f"\nDiscovered {len(all_proteins)} protein directories.")

    protein_data = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        try:
            # FLAG: find_pdb_file looks under STRUCTURES_DIR with pattern:
            # Structures/my_outputs_align_{protein_id}_representations_load_from_saved_layer_47/predictions/{protein_id}_model_1_ptm_unrelaxed.pdb
            # If the naming convention changes, update find_pdb_file().
            pdb_path = find_pdb_file(protein_id)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                print(f"[SKIP] {protein_id}: no predicted PDB file found")
                continue

            # FLAG: get_pair_block_path looks for pair blocks directly in protein_dir:
            # {protein_dir}/{protein_id}_pair_block_{layer}.npy
            # If the naming convention changes, update get_pair_block_path().
            pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
            if not pb0_path.exists():
                skipped.append((protein_id, "pair block 0 not found"))
                print(f"[SKIP] {protein_id}: pair block 0 not found at {pb0_path}")
                continue

            # FLAG: chain ID is parsed from protein_id (e.g. "7b3a_A" -> "A").
            # If the naming convention changes, update this line.
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
    # Phase 2: Pool targets and split
    # ------------------------------------------------------------------
    y_all = np.concatenate([d["y"] for d in protein_data])
    n_total_pairs = len(y_all)

    protein_offsets = []
    offset = 0
    for d in protein_data:
        n = len(d["y"])
        protein_offsets.append((d["protein_id"], offset, offset + n))
        offset += n

    print(f"\nTotal pairs across {n_proteins} protein(s): {n_total_pairs:,}")

    all_idx = np.arange(n_total_pairs)
    idx_train, idx_test = train_test_split(all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_train = y_all[idx_train]
    y_test = y_all[idx_test]

    split_labels = np.empty(n_total_pairs, dtype=object)
    split_labels[idx_train] = "train"
    split_labels[idx_test] = "test"

    # Precompute per-protein test positions (indices into y_test array)
    protein_test_positions = {}
    for pid, start, end in protein_offsets:
        protein_test_positions[pid] = np.where((idx_test >= start) & (idx_test < end))[0]
    protein_ids_sorted = [pid for pid, _, _ in protein_offsets]

    # ------------------------------------------------------------------
    # Phase 3: Layer-wise regression
    # ------------------------------------------------------------------
    results = []
    all_coefs = []
    last_layer_preds = None

    print()
    for layer in range(n_layers):
        try:
            X_chunks = []
            for d in protein_data:
                # FLAG: get_pair_block_path looks for pair blocks directly in protein_dir:
                # {protein_dir}/{protein_id}_pair_block_{layer}.npy
                # If the naming convention changes, update get_pair_block_path().
                pb_path = get_pair_block_path(d["protein_dir"], d["protein_id"], layer)
                pb = np.load(str(pb_path))
                X_chunk = np.concatenate([
                    pb[d["pb_i"], d["pb_j"], :],
                    pb[d["pb_j"], d["pb_i"], :],
                ], axis=1)
                X_chunks.append(X_chunk)
                del pb

            X_all = np.vstack(X_chunks)
            del X_chunks

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_all[idx_train])
            X_test = scaler.transform(X_all[idx_test])

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            r2 = r2_score(y_test, y_pred_test)
            y_pred_train = model.predict(X_train)
            r2_train = r2_score(y_train, y_pred_train)
            mae = mean_absolute_error(y_test, y_pred_test)
            rmse = math.sqrt(np.mean((y_test - y_pred_test) ** 2))

            per_protein_r2 = {}
            for pid, positions in protein_test_positions.items():
                if len(positions) > 1:
                    per_protein_r2[pid] = r2_score(y_test[positions], y_pred_test[positions])

            results.append({
                "r2_test": r2, "r2_train": r2_train,
                "mae_test": mae, "rmse_test": rmse,
                "per_protein_r2": per_protein_r2,
            })
            print(f"Layer {layer:2d}  R2_test={r2:.4f}  R2_train={r2_train:.4f}  MAE={mae:.2f}  RMSE={rmse:.2f}")

            coef_row = {"layer": layer, "intercept": float(model.intercept_)}
            coef_row.update({f"coef_{i}": float(c) for i, c in enumerate(model.coef_)})
            all_coefs.append(coef_row)

            if layer == n_layers - 1:
                X_all_std = scaler.transform(X_all)
                last_layer_preds = model.predict(X_all_std)

            del X_all, X_train, X_test

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
    r2_csv = csv_dir / "layer_regression_r2_predicted.csv"
    per_prot_cols = [f"r2_test_{pid}" for pid in protein_ids_sorted]
    with open(r2_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "r2_test", "r2_train", "mae_test", "rmse_test",
                          "n_proteins", "n_total_pairs"] + per_prot_cols)
        for layer, res in enumerate(results):
            row = [layer, f"{res['r2_test']:.6f}", f"{res['r2_train']:.6f}",
                   f"{res['mae_test']:.4f}", f"{res['rmse_test']:.4f}",
                   n_proteins, n_total_pairs]
            for pid in protein_ids_sorted:
                val = res["per_protein_r2"].get(pid, float("nan"))
                row.append(f"{val:.6f}")
            writer.writerow(row)
    print(f"\nSaved metrics CSV -> {r2_csv}")

    # CSV 2: Coefficients
    coef_csv = csv_dir / "layer_regression_coefficients_predicted.csv"
    fieldnames = ["layer", "intercept"] + [f"coef_{i}" for i in range(FEATURE_DIM)]
    with open(coef_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_coefs)
    print(f"Saved coefficients CSV -> {coef_csv}")

    # CSV 3: Predicted distances (last layer)
    if last_layer_preds is not None:
        pred_csv = csv_dir / "predicted_distances_last_layer_predicted.csv"
        with open(pred_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "protein_id", "struct_i", "struct_j",
                "actual_distance_A", "predicted_distance_A", "split",
            ])
            for pid, start, end in protein_offsets:
                d = next(x for x in protein_data if x["protein_id"] == pid)
                for k in range(end - start):
                    g = start + k
                    writer.writerow([
                        pid,
                        int(d["pb_i"][k]),
                        int(d["pb_j"][k]),
                        f"{y_all[g]:.4f}",
                        f"{last_layer_preds[g]:.4f}",
                        split_labels[g],
                    ])
        print(f"Saved predicted distances CSV -> {pred_csv}")
    else:
        print("WARNING: Last layer failed; predicted distances CSV not saved.")

    # Summary text file
    summary_path = csv_dir / "regression_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Pooled Multi-Protein Linear Regression Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Proteins:        {n_proteins}\n")
        for pid, start, end in protein_offsets:
            d = next(x for x in protein_data if x["protein_id"] == pid)
            f.write(f"  {pid}: {d['n_residues']} residues, {end - start:,} pairs\n")
        f.write(f"\nTotal pairs:     {n_total_pairs:,}\n")
        f.write(f"Train pairs:     {len(idx_train):,}\n")
        f.write(f"Test pairs:      {len(idx_test):,}\n")
        f.write(f"Test fraction:   {TEST_SIZE}\n")
        f.write(f"Random state:    {RANDOM_STATE}\n")
        f.write(f"Feature dim:     {FEATURE_DIM}\n")
        f.write(f"Layers:          {n_layers}\n\n")

        valid = [(i, r["r2_test"]) for i, r in enumerate(results) if not math.isnan(r["r2_test"])]
        if valid:
            best_layer, best_r2 = max(valid, key=lambda x: x[1])
            f.write(f"Best layer:      {best_layer} (R2_test = {best_r2:.6f})\n")
            last = results[-1]
            f.write(f"Last layer (47): R2_test={last['r2_test']:.6f}  R2_train={last['r2_train']:.6f}  "
                    f"MAE={last['mae_test']:.4f}  RMSE={last['rmse_test']:.4f}\n\n")
            f.write("Per-protein R2 (test) at last layer:\n")
            for pid in protein_ids_sorted:
                val = last["per_protein_r2"].get(pid, float("nan"))
                f.write(f"  {pid}: {val:.6f}\n")

        f.write(f"\nLayer-by-layer R2 (test):\n")
        for i, res in enumerate(results):
            f.write(f"  Layer {i:2d}: {res['r2_test']:.6f}\n")

    print(f"Saved summary    -> {summary_path}")


if __name__ == "__main__":
    main()

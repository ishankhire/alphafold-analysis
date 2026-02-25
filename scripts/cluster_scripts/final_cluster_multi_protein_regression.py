#!/usr/bin/env python3
"""
final_cluster_multi_protein_regression.py
=========================================
K-fold cross-validated protein-level linear regression across AlphaFold pair
representation layers.

Splits at the PROTEIN level using 5-fold CV so the model never sees any pairs
from test proteins during training.  For each fold and each layer, fits a
StandardScaler + LinearRegression on training proteins and evaluates on the
held-out fold.  Aggregates mean +/- std metrics across folds.

Cluster layout (configured via environment variables set in sbatch files):
  - Pair blocks:  PAIR_BLOCKS_ROOT/{protein_id}/{protein_id}_pair_block_{layer}.npy
  - PDB files:    PDB_ROOT/my_outputs_align_{id}_representations_load_from_saved_layer_47/predictions/{id}_model_1_ptm_unrelaxed.pdb
  - Output CSVs:  CSV_ROOT/

Dependencies (all pip-installable):
    pip install numpy scipy matplotlib scikit-learn biopython

Outputs (inside CSV_ROOT/):
    kfold_regression_r2.csv              — per-(fold, layer) R2, MAE, RMSE
    kfold_regression_r2_summary.csv      — per-layer mean +/- std across folds
    kfold_regression_coefficients.csv    — model coefficients per (fold, layer)
    kfold_predicted_distances.csv        — actual vs predicted distances (last fold, last layer)
    kfold_regression_summary.txt         — human-readable summary
"""

import csv
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

from Bio.PDB import PDBParser


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48          # AlphaFold pair representation layers
N_CHANNELS = 128       # channels per pair representation direction
FEATURE_DIM = 2 * N_CHANNELS  # 256 (concatenated [i,j] and [j,i])
N_FOLDS = 5            # number of cross-validation folds (protein-level)
RANDOM_STATE = 42

# ===========================================================================
# Directories (set via environment variables exported by sbatch files)
# ===========================================================================

PAIR_BLOCKS_ROOT = Path(os.environ["PAIR_BLOCKS_ROOT"])
PDB_ROOT = Path(os.environ["PDB_ROOT"])
CSV_DIR = Path(os.environ["CSV_ROOT"])


# ===========================================================================
# Helper functions — data loading
# ===========================================================================

def find_pdb_file(protein_id):
    """Locate the predicted PDB file for a protein.

    Expected path:
        PDB_ROOT/my_outputs_align_{protein_id}_representations_load_from_saved_layer_47/
            predictions/{protein_id}_model_1_ptm_unrelaxed.pdb

    Returns the Path object, or None if not found.
    """
    pdb_path = (
        PDB_ROOT
        / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions"
        / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    )
    return pdb_path if pdb_path.exists() else None


def get_pair_block_path(protein_dir, protein_id, layer):
    """Return the path to a pair block .npy file for a given layer.

    Expected: <protein_dir>/<protein_id>_pair_block_<layer>.npy
    (pair blocks live directly in the protein directory, no pair_blocks/ subdirectory)
    """
    return protein_dir / f"{protein_id}_pair_block_{layer}.npy"


def discover_proteins(pair_blocks_root):
    """Find all protein subdirectories under the pair blocks root.

    Returns a sorted list of (protein_id, protein_dir) tuples.
    Skips non-directory entries.
    """
    proteins = []
    for entry in sorted(pair_blocks_root.iterdir()):
        if not entry.is_dir():
            continue
        proteins.append((entry.name, entry))
    return proteins


# ===========================================================================
# Helper functions — coordinate extraction & distance computation
# ===========================================================================

def extract_ca_coordinates(pdb_path, chain_id):
    """Extract CA atom coordinates from a PDB file.

    Extracts ALL residues with a CA atom (no standard-residue filtering).

    Parameters
    ----------
    pdb_path : str or Path
        Path to the PDB file.
    chain_id : str
        Chain identifier to extract (e.g. 'A', 'B').

    Returns
    -------
    dict with keys:
        'residues'    : list of (chain_id, residue_name, residue_number) tuples
        'coordinates' : np.ndarray of shape (n_residues, 3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", str(pdb_path))

    residues = []
    coordinates = []

    model = structure[0]
    for chain in model:
        if chain.get_id() != chain_id:
            continue
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
# Helper — build feature matrix for a list of proteins at a given layer
# ===========================================================================

def build_feature_matrix(proteins, layer):
    """Load pair blocks and build the concatenated feature matrix for one layer.

    Parameters
    ----------
    proteins : list of dict
        Each dict has keys: protein_dir, protein_id, pb_i, pb_j.
    layer : int
        Layer index (0..47).

    Returns
    -------
    np.ndarray of shape (n_total_pairs, FEATURE_DIM)
    """
    chunks = []
    for d in proteins:
        pb_path = get_pair_block_path(d["protein_dir"], d["protein_id"], layer)
        pb = np.load(str(pb_path))
        chunk = np.concatenate([
            pb[d["pb_i"], d["pb_j"], :],
            pb[d["pb_j"], d["pb_i"], :],
        ], axis=1)
        chunks.append(chunk)
        del pb
    X = np.vstack(chunks)
    del chunks
    return X


# ===========================================================================
# Main
# ===========================================================================

def main():
    n_layers = N_LAYERS

    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"Pair blocks root:   {PAIR_BLOCKS_ROOT}")
    print(f"PDB root:           {PDB_ROOT}")
    print(f"Output CSV dir:     {csv_dir}")
    print(f"Layers:             {n_layers}")
    print(f"Split strategy:     {N_FOLDS}-FOLD CROSS-VALIDATION (protein-level)")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
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

            # FLAG: chain ID is parsed from protein_id (e.g. "7b3a_A" -> "A").
            chain_id = protein_id.split("_")[-1]

            ca_data = extract_ca_coordinates(pdb_path, chain_id=chain_id)
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

    if n_proteins < N_FOLDS:
        print(f"\nERROR: Need at least {N_FOLDS} proteins for {N_FOLDS}-fold CV. "
              f"Only {n_proteins} loaded. Exiting.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2: Set up K-fold cross-validation (protein-level)
    # ------------------------------------------------------------------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    protein_indices = np.arange(n_proteins)
    folds = list(kf.split(protein_indices))

    # Summarize folds
    print(f"\n{'='*60}")
    print(f"{N_FOLDS}-FOLD PROTEIN-LEVEL CROSS-VALIDATION")
    print(f"{'='*60}")
    total_pairs = sum(len(d["y"]) for d in protein_data)
    print(f"  Total proteins: {n_proteins}")
    print(f"  Total pairs:    {total_pairs:,}")
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_ids = [protein_data[i]["protein_id"] for i in train_idx]
        test_ids = [protein_data[i]["protein_id"] for i in test_idx]
        n_train_pairs = sum(len(protein_data[i]["y"]) for i in train_idx)
        n_test_pairs = sum(len(protein_data[i]["y"]) for i in test_idx)
        print(f"  Fold {fold_idx}: {len(train_idx)} train / {len(test_idx)} test proteins, "
              f"{n_train_pairs:,} train / {n_test_pairs:,} test pairs")
    print(f"{'='*60}\n")

    # ------------------------------------------------------------------
    # Phase 3: K-fold cross-validated layer-wise regression
    # ------------------------------------------------------------------
    # Store per-(fold, layer) results
    # results_grid[fold_idx][layer] = {r2_test, r2_train, mae_test, rmse_test, per_protein_r2}
    results_grid = []
    coefs_grid = []

    # For predicted distances CSV: save last fold, last layer
    last_fold_preds = None

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_proteins = [protein_data[i] for i in train_idx]
        test_proteins = [protein_data[i] for i in test_idx]

        y_train = np.concatenate([d["y"] for d in train_proteins])
        y_test = np.concatenate([d["y"] for d in test_proteins])

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

        fold_results = []
        fold_coefs = []

        print(f"--- Fold {fold_idx} ({len(train_idx)} train / {len(test_idx)} test proteins) ---")

        for layer in range(n_layers):
            try:
                X_train_raw = build_feature_matrix(train_proteins, layer)
                X_test_raw = build_feature_matrix(test_proteins, layer)

                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train_raw)
                X_test = scaler.transform(X_test_raw)

                model = LinearRegression()
                model.fit(X_train, y_train)
                y_pred_train = model.predict(X_train)
                y_pred_test = model.predict(X_test)

                r2_train = r2_score(y_train, y_pred_train)
                r2_test = r2_score(y_test, y_pred_test)
                mae = mean_absolute_error(y_test, y_pred_test)
                rmse = math.sqrt(np.mean((y_test - y_pred_test) ** 2))

                # Per-protein R2 (test proteins)
                per_protein_r2 = {}
                for pid, start, end in test_offsets:
                    if end - start > 1:
                        per_protein_r2[pid] = r2_score(
                            y_test[start:end], y_pred_test[start:end]
                        )

                fold_results.append({
                    "r2_test": r2_test, "r2_train": r2_train,
                    "mae_test": mae, "rmse_test": rmse,
                    "per_protein_r2": per_protein_r2,
                })
                print(f"  Fold {fold_idx} Layer {layer:2d}  "
                      f"R2_test={r2_test:.4f}  R2_train={r2_train:.4f}  "
                      f"MAE={mae:.2f}  RMSE={rmse:.2f}")

                coef_row = {"fold": fold_idx, "layer": layer,
                            "intercept": float(model.intercept_)}
                coef_row.update({f"coef_{i}": float(c)
                                 for i, c in enumerate(model.coef_)})
                fold_coefs.append(coef_row)

                # Save predictions for last fold, last layer
                if fold_idx == N_FOLDS - 1 and layer == n_layers - 1:
                    last_fold_preds = {
                        "train_proteins": train_proteins,
                        "test_proteins": test_proteins,
                        "train_offsets": train_offsets,
                        "test_offsets": test_offsets,
                        "y_train": y_train,
                        "y_test": y_test,
                        "y_pred_train": y_pred_train,
                        "y_pred_test": y_pred_test,
                    }

                del X_train_raw, X_test_raw, X_train, X_test

            except Exception as e:
                print(f"  Fold {fold_idx} Layer {layer:2d}  ERROR: {e}")
                traceback.print_exc()
                fold_results.append({
                    "r2_test": float("nan"), "r2_train": float("nan"),
                    "mae_test": float("nan"), "rmse_test": float("nan"),
                    "per_protein_r2": {},
                })
                fold_coefs.append({"fold": fold_idx, "layer": layer,
                                   "intercept": 0.0})

        results_grid.append(fold_results)
        coefs_grid.append(fold_coefs)

    # ------------------------------------------------------------------
    # Phase 4: Aggregate metrics across folds
    # ------------------------------------------------------------------
    layer_summaries = []
    for layer in range(n_layers):
        r2_tests = [results_grid[f][layer]["r2_test"] for f in range(N_FOLDS)]
        r2_trains = [results_grid[f][layer]["r2_train"] for f in range(N_FOLDS)]
        maes = [results_grid[f][layer]["mae_test"] for f in range(N_FOLDS)]
        rmses = [results_grid[f][layer]["rmse_test"] for f in range(N_FOLDS)]

        # Filter out NaN values for aggregation
        r2_tests_valid = [v for v in r2_tests if not math.isnan(v)]
        r2_trains_valid = [v for v in r2_trains if not math.isnan(v)]
        maes_valid = [v for v in maes if not math.isnan(v)]
        rmses_valid = [v for v in rmses if not math.isnan(v)]

        layer_summaries.append({
            "r2_test_mean": np.mean(r2_tests_valid) if r2_tests_valid else float("nan"),
            "r2_test_std": np.std(r2_tests_valid) if r2_tests_valid else float("nan"),
            "r2_train_mean": np.mean(r2_trains_valid) if r2_trains_valid else float("nan"),
            "r2_train_std": np.std(r2_trains_valid) if r2_trains_valid else float("nan"),
            "mae_mean": np.mean(maes_valid) if maes_valid else float("nan"),
            "mae_std": np.std(maes_valid) if maes_valid else float("nan"),
            "rmse_mean": np.mean(rmses_valid) if rmses_valid else float("nan"),
            "rmse_std": np.std(rmses_valid) if rmses_valid else float("nan"),
        })

    # ------------------------------------------------------------------
    # Phase 5: Save CSV outputs
    # ------------------------------------------------------------------

    # CSV 1: Per-(fold, layer) raw metrics
    r2_csv = csv_dir / "kfold_regression_r2.csv"
    with open(r2_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "fold", "layer", "r2_test", "r2_train", "mae_test", "rmse_test",
            "n_train_proteins", "n_test_proteins", "n_train_pairs", "n_test_pairs",
        ])
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            n_train_pairs = sum(len(protein_data[i]["y"]) for i in train_idx)
            n_test_pairs = sum(len(protein_data[i]["y"]) for i in test_idx)
            for layer in range(n_layers):
                res = results_grid[fold_idx][layer]
                writer.writerow([
                    fold_idx, layer,
                    f"{res['r2_test']:.6f}", f"{res['r2_train']:.6f}",
                    f"{res['mae_test']:.4f}", f"{res['rmse_test']:.4f}",
                    len(train_idx), len(test_idx),
                    n_train_pairs, n_test_pairs,
                ])
    print(f"\nSaved per-fold metrics CSV -> {r2_csv}")

    # CSV 2: Aggregated summary per layer
    summary_csv = csv_dir / "kfold_regression_r2_summary.csv"
    with open(summary_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "layer",
            "r2_test_mean", "r2_test_std",
            "r2_train_mean", "r2_train_std",
            "mae_mean", "mae_std",
            "rmse_mean", "rmse_std",
        ])
        for layer, s in enumerate(layer_summaries):
            writer.writerow([
                layer,
                f"{s['r2_test_mean']:.6f}", f"{s['r2_test_std']:.6f}",
                f"{s['r2_train_mean']:.6f}", f"{s['r2_train_std']:.6f}",
                f"{s['mae_mean']:.4f}", f"{s['mae_std']:.4f}",
                f"{s['rmse_mean']:.4f}", f"{s['rmse_std']:.4f}",
            ])
    print(f"Saved summary CSV      -> {summary_csv}")

    # CSV 3: Coefficients per (fold, layer)
    coef_csv = csv_dir / "kfold_regression_coefficients.csv"
    fieldnames = ["fold", "layer", "intercept"] + [f"coef_{i}" for i in range(FEATURE_DIM)]
    with open(coef_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for fold_coefs in coefs_grid:
            writer.writerows(fold_coefs)
    print(f"Saved coefficients CSV -> {coef_csv}")

    # CSV 4: Predicted distances (last fold, last layer only)
    if last_fold_preds is not None:
        pred_csv = csv_dir / "kfold_predicted_distances.csv"
        lp = last_fold_preds
        with open(pred_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([
                "protein_id", "struct_i", "struct_j",
                "actual_distance_A", "predicted_distance_A", "split",
            ])
            # Training proteins
            for pid, start, end in lp["train_offsets"]:
                d = next(x for x in lp["train_proteins"] if x["protein_id"] == pid)
                for k in range(end - start):
                    writer.writerow([
                        pid,
                        int(d["pb_i"][k]),
                        int(d["pb_j"][k]),
                        f"{lp['y_train'][start + k]:.4f}",
                        f"{lp['y_pred_train'][start + k]:.4f}",
                        "train",
                    ])
            # Test proteins
            for pid, start, end in lp["test_offsets"]:
                d = next(x for x in lp["test_proteins"] if x["protein_id"] == pid)
                for k in range(end - start):
                    writer.writerow([
                        pid,
                        int(d["pb_i"][k]),
                        int(d["pb_j"][k]),
                        f"{lp['y_test'][start + k]:.4f}",
                        f"{lp['y_pred_test'][start + k]:.4f}",
                        "test",
                    ])
        print(f"Saved predicted distances CSV -> {pred_csv}")
    else:
        print("WARNING: Last fold/layer failed; predicted distances CSV not saved.")

    # Summary text file
    summary_path = csv_dir / "kfold_regression_summary.txt"
    with open(summary_path, "w") as f:
        f.write(f"{N_FOLDS}-Fold Cross-Validated Linear Regression Summary\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Split strategy:  {N_FOLDS}-FOLD PROTEIN-LEVEL CV\n")
        f.write(f"Random state:    {RANDOM_STATE}\n")
        f.write(f"Feature dim:     {FEATURE_DIM}\n")
        f.write(f"Layers:          {n_layers}\n")
        f.write(f"Total proteins:  {n_proteins}\n")
        f.write(f"Total pairs:     {total_pairs:,}\n\n")

        f.write("Fold breakdown:\n")
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            test_ids = [protein_data[i]["protein_id"] for i in test_idx]
            n_tp = sum(len(protein_data[i]["y"]) for i in train_idx)
            n_ep = sum(len(protein_data[i]["y"]) for i in test_idx)
            f.write(f"  Fold {fold_idx}: {len(train_idx)} train / {len(test_idx)} test proteins, "
                    f"{n_tp:,} / {n_ep:,} pairs\n")
            f.write(f"         Test proteins: {', '.join(test_ids)}\n")

        # Best layer by mean R2
        valid = [(i, s["r2_test_mean"]) for i, s in enumerate(layer_summaries)
                 if not math.isnan(s["r2_test_mean"])]
        if valid:
            best_layer, best_r2 = max(valid, key=lambda x: x[1])
            best_std = layer_summaries[best_layer]["r2_test_std"]
            f.write(f"\nBest layer:      {best_layer} "
                    f"(R2_test = {best_r2:.6f} +/- {best_std:.6f})\n")

            last = layer_summaries[-1]
            f.write(f"Last layer (47): R2_test = {last['r2_test_mean']:.6f} "
                    f"+/- {last['r2_test_std']:.6f}\n")
            f.write(f"                 R2_train = {last['r2_train_mean']:.6f} "
                    f"+/- {last['r2_train_std']:.6f}\n")
            f.write(f"                 MAE = {last['mae_mean']:.4f} "
                    f"+/- {last['mae_std']:.4f}\n")
            f.write(f"                 RMSE = {last['rmse_mean']:.4f} "
                    f"+/- {last['rmse_std']:.4f}\n")

        f.write(f"\nLayer-by-layer R2_test (mean +/- std):\n")
        for i, s in enumerate(layer_summaries):
            f.write(f"  Layer {i:2d}: {s['r2_test_mean']:.6f} +/- {s['r2_test_std']:.6f}\n")

    print(f"Saved summary    -> {summary_path}")
    print(f"\nDone. {N_FOLDS}-fold CV complete across {n_layers} layers.")


if __name__ == "__main__":
    main()

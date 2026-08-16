#!/usr/bin/env python3
"""
per_protein_regression.py
==========================
Variant that trains a SEPARATE linear regression per protein
per layer (instead of pooling all proteins into one model).

Cluster layout (configured via environment variables set in sbatch files):
  - Pair blocks:  PAIR_BLOCKS_ROOT/{protein_id}/{protein_id}_pair_block_{layer}.npy
  - PDB files:    PDB_ROOT/my_outputs_align_{id}_representations_load_from_saved_layer_47/predictions/{id}_model_1_ptm_unrelaxed.pdb
  - Output CSVs:  CSV_ROOT/

All helper functions are defined inline (no lib/ imports).

Dependencies (all pip-installable):
    pip install numpy scipy matplotlib scikit-learn biopython

Outputs (inside CSV_ROOT/):
    per_protein_regression_metrics.csv      — per-(protein, layer) R2, MAE, RMSE
    per_protein_coefficients.csv            — model coefficients per (protein, layer)
    per_protein_mds_tm_scores.csv           — per-(protein, layer) MDS RMSD + TM-score
    per_protein_predicted_distances.csv     — actual vs predicted (layer 47 only)
    per_protein_regression_summary.txt      — human-readable summary
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

def extract_ca_coordinates(pdb_path):
    """Extract CA atom coordinates from a PDB file.

    Extracts ALL residues with a CA atom across all chains.
    Chain filtering is unnecessary because each PDB file is already
    specific to a single protein.

    Parameters
    ----------
    pdb_path : str or Path
        Path to the PDB file.

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
# Classical MDS + TM-score helpers
# ===========================================================================

def classical_mds(distance_matrix, n_dims=3):
    """Classical (Torgerson) MDS: recover coordinates from pairwise distances."""
    D_sq = distance_matrix ** 2

    row_mean = D_sq.mean(axis=1, keepdims=True)
    col_mean = D_sq.mean(axis=0, keepdims=True)
    grand_mean = D_sq.mean()
    B = -0.5 * (D_sq - row_mean - col_mean + grand_mean)

    eigenvalues, eigenvectors = np.linalg.eigh(B)
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    top_evals = np.maximum(eigenvalues[:n_dims], 0.0)
    coords = eigenvectors[:, :n_dims] * np.sqrt(top_evals)
    return coords


def superimpose_structures(fixed_coords, moving_coords):
    """Kabsch algorithm for optimal rigid superimposition."""
    fixed_center = fixed_coords.mean(axis=0)
    moving_center = moving_coords.mean(axis=0)
    fixed_centered = fixed_coords - fixed_center
    moving_centered = moving_coords - moving_center

    covariance = moving_centered.T @ fixed_centered
    U, _, Vt = np.linalg.svd(covariance)
    det_sign = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(det_sign)])
    rotation = Vt.T @ sign_matrix @ U.T

    transformed = (moving_centered @ rotation.T) + fixed_center
    diff = fixed_coords - transformed
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))
    return transformed, float(rmsd)


def compute_tm_score(fixed_coords, transformed_coords, length_target):
    """Compute TM-score between two superimposed structures."""
    d0 = 1.24 * (max(length_target, 16) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)

    distances = np.sqrt(np.sum((fixed_coords - transformed_coords) ** 2, axis=1))
    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    return float(np.sum(scores) / length_target)


def analyze_protein_geometry(n_residues, true_coords, pb_i, pb_j, y_pred):
    """Run MDS reconstruction + rigid superimposition for one protein/layer."""
    pred_dist = np.zeros((n_residues, n_residues))
    pred_dist[pb_i, pb_j] = y_pred
    pred_dist[pb_j, pb_i] = y_pred
    pred_dist = np.maximum(pred_dist, 0.0)

    mds_coords = classical_mds(pred_dist, n_dims=3)
    mds_reflected = mds_coords.copy()
    mds_reflected[:, 2] *= -1

    transformed_orig, rmsd_orig = superimpose_structures(true_coords, mds_coords)
    transformed_flip, rmsd_flip = superimpose_structures(true_coords, mds_reflected)

    if rmsd_flip < rmsd_orig:
        transformed, rmsd = transformed_flip, rmsd_flip
    else:
        transformed, rmsd = transformed_orig, rmsd_orig

    tm_score = compute_tm_score(true_coords, transformed, n_residues)
    return {"tm_score": tm_score, "rmsd": rmsd}


# ===========================================================================
# Per-protein regression
# ===========================================================================

def process_single_protein(protein_id, protein_dir, n_layers):
    """Train per-layer linear regression for a single protein.

    Parameters
    ----------
    protein_id : str
        e.g. "7b3a_A"
    protein_dir : Path
        Directory containing pair block .npy files for this protein.
    n_layers : int
        Number of layers to process (48).

    Returns
    -------
    dict with keys:
        'protein_id'             : str
        'n_residues'             : int
        'n_pairs'                : int
        'metrics'                : list of 48 dicts (layer, r2_train, r2_test, mae_test, rmse_test)
        'coefficients'           : list of 48 dicts (layer, intercept, coef_0..coef_255)
        'mds_metrics'            : list of 48 dicts (layer, tm_score, rmsd)
        'last_layer_predictions' : dict with struct_i, struct_j, actual, predicted, split arrays

    Raises
    ------
    ValueError
        If PDB not found, no CA atoms, dimension mismatch, or too few pairs.
    """
    # --- Locate PDB ---
    pdb_path = find_pdb_file(protein_id)
    if pdb_path is None:
        raise ValueError("no predicted PDB file found")

    # --- Check all pair blocks exist ---
    pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
    if not pb0_path.exists():
        raise ValueError(f"pair block 0 not found at {pb0_path}")

    # --- Extract CA coordinates ---
    ca_data = extract_ca_coordinates(pdb_path)
    n_ca = len(ca_data["residues"])
    if n_ca == 0:
        raise ValueError("no CA atoms found in PDB")

    # --- Check pair block dimension matches ---
    pb0 = np.load(str(pb0_path), mmap_mode="r")
    pb_dim = pb0.shape[0]
    del pb0

    if n_ca != pb_dim:
        raise ValueError(f"CA count {n_ca} != pair block dim {pb_dim}")

    # --- Compute distance matrix and targets ---
    dist_matrix = compute_distance_matrix(ca_data["coordinates"])
    struct_i, struct_j = np.triu_indices(n_ca, k=1)
    y = dist_matrix[struct_i, struct_j]
    n_pairs = len(y)
    del dist_matrix

    if n_pairs < FEATURE_DIM:
        raise ValueError(
            f"underdetermined: {n_pairs} pairs < {FEATURE_DIM} features "
            f"({n_ca} residues)"
        )

    # --- Train/test split (per-protein, fixed seed) ---
    all_idx = np.arange(n_pairs)
    idx_train, idx_test = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_train = y[idx_train]
    y_test = y[idx_test]

    split_labels = np.empty(n_pairs, dtype=object)
    split_labels[idx_train] = "train"
    split_labels[idx_test] = "test"

    # --- Layer-wise regression ---
    metrics = []
    coefficients = []
    mds_metrics = []
    last_layer_preds = None

    for layer in range(n_layers):
        pb_path = get_pair_block_path(protein_dir, protein_id, layer)
        pb = np.load(str(pb_path))

        X = np.concatenate([
            pb[struct_i, struct_j, :],
            pb[struct_j, struct_i, :],
        ], axis=1)
        del pb

        scaler = StandardScaler()
        X_train = scaler.fit_transform(X[idx_train])
        X_test = scaler.transform(X[idx_test])

        model = LinearRegression()
        model.fit(X_train, y_train)

        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_test = math.sqrt(np.mean((y_test - y_pred_test) ** 2))

        metrics.append({
            "layer": layer,
            "r2_train": r2_train,
            "r2_test": r2_test,
            "mae_test": mae_test,
            "rmse_test": rmse_test,
        })

        coef_row = {"layer": layer, "intercept": float(model.intercept_)}
        coef_row.update({
            f"coef_{i}": float(c) for i, c in enumerate(model.coef_)
        })
        coefficients.append(coef_row)

        X_all_std = scaler.transform(X)
        y_pred_all = model.predict(X_all_std)

        try:
            geom = analyze_protein_geometry(
                n_ca,
                ca_data["coordinates"],
                struct_i,
                struct_j,
                y_pred_all,
            )
            tm_score = geom["tm_score"]
            rmsd = geom["rmsd"]
        except Exception:
            tm_score = float("nan")
            rmsd = float("nan")

        mds_metrics.append({
            "layer": layer,
            "tm_score": tm_score,
            "rmsd": rmsd,
        })

        if layer == n_layers - 1:
            last_layer_preds = y_pred_all

        del X, X_train, X_test, X_all_std

    return {
        "protein_id": protein_id,
        "n_residues": n_ca,
        "n_pairs": n_pairs,
        "metrics": metrics,
        "coefficients": coefficients,
        "mds_metrics": mds_metrics,
        "last_layer_predictions": {
            "struct_i": struct_i,
            "struct_j": struct_j,
            "actual": y,
            "predicted": last_layer_preds,
            "split": split_labels,
        },
    }


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
    print(f"Mode:               PER-PROTEIN (independent model per protein per layer)")

    # ------------------------------------------------------------------
    # Phase 1: Discover proteins
    # ------------------------------------------------------------------
    all_proteins = discover_proteins(PAIR_BLOCKS_ROOT)
    n_total = len(all_proteins)
    print(f"\nDiscovered {n_total} protein directories.")

    # ------------------------------------------------------------------
    # Phase 2: Open streaming CSV for predicted distances
    # ------------------------------------------------------------------
    pred_path = csv_dir / "per_protein_predicted_distances.csv"
    f_pred = open(pred_path, "w", newline="")
    w_pred = csv.writer(f_pred)
    w_pred.writerow([
        "protein_id", "struct_i", "struct_j",
        "actual_distance_A", "predicted_distance_A", "split",
    ])

    # Accumulate metrics and coefficients in memory (small: ~24K rows)
    all_metrics = []
    all_coefs = []
    all_mds = []
    succeeded = []
    skipped = []

    # ------------------------------------------------------------------
    # Phase 3: Process each protein
    # ------------------------------------------------------------------
    for idx, (protein_id, protein_dir) in enumerate(all_proteins):
        t0 = time.time()
        start_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(t0))
        print(
            f"\n[{idx + 1}/{n_total}] Processing {protein_id} ... "
            f"(start {start_ts})",
            flush=True,
        )

        try:
            result = process_single_protein(protein_id, protein_dir, n_layers)
        except ValueError as e:
            skipped.append((protein_id, str(e)))
            end_t = time.time()
            end_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_t))
            print(f"  [SKIP] {e} (end {end_ts}, {end_t - t0:.1f}s)")
            continue
        except Exception as e:
            skipped.append((protein_id, f"unexpected error -- {e}"))
            end_t = time.time()
            end_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(end_t))
            print(f"  [SKIP] unexpected error -- {e} (end {end_ts}, {end_t - t0:.1f}s)")
            traceback.print_exc()
            continue

        # Accumulate metrics (add protein_id to each row)
        for m in result["metrics"]:
            all_metrics.append({
                "protein_id": protein_id,
                "n_residues": result["n_residues"],
                "n_pairs": result["n_pairs"],
                **m,
            })

        # Accumulate coefficients (add protein_id to each row)
        for c in result["coefficients"]:
            c["protein_id"] = protein_id
            all_coefs.append(c)

        # Accumulate MDS/TM metrics (add protein_id to each row)
        for mds in result["mds_metrics"]:
            all_mds.append({
                "protein_id": protein_id,
                "n_residues": result["n_residues"],
                **mds,
            })

        # Stream predicted distances to disk immediately
        preds = result["last_layer_predictions"]
        if preds["predicted"] is not None:
            for k in range(result["n_pairs"]):
                w_pred.writerow([
                    protein_id,
                    int(preds["struct_i"][k]),
                    int(preds["struct_j"][k]),
                    f"{preds['actual'][k]:.4f}",
                    f"{preds['predicted'][k]:.4f}",
                    preds["split"][k],
                ])
            f_pred.flush()

        elapsed = time.time() - t0
        end_ts = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        best_layer = max(result["metrics"], key=lambda m: m["r2_test"])
        succeeded.append({
            "protein_id": protein_id,
            "n_residues": result["n_residues"],
            "n_pairs": result["n_pairs"],
            "best_layer": best_layer["layer"],
            "best_r2": best_layer["r2_test"],
            "r2_layer47": result["metrics"][-1]["r2_test"],
        })
        print(f"  [OK] {result['n_residues']} residues, {result['n_pairs']:,} pairs, "
              f"best R2={best_layer['r2_test']:.4f} (layer {best_layer['layer']}), "
              f"end {end_ts}, {elapsed:.1f}s")

    f_pred.close()
    print(f"\nSaved predictions CSV  -> {pred_path}")

    # ------------------------------------------------------------------
    # Phase 4: Write metrics and coefficients CSVs
    # ------------------------------------------------------------------

    # CSV 1: Metrics
    metrics_path = csv_dir / "per_protein_regression_metrics.csv"
    with open(metrics_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "protein_id", "layer", "n_residues", "n_pairs",
            "r2_train", "r2_test", "mae_test", "rmse_test",
        ])
        for m in all_metrics:
            writer.writerow([
                m["protein_id"], m["layer"], m["n_residues"], m["n_pairs"],
                f"{m['r2_train']:.6f}", f"{m['r2_test']:.6f}",
                f"{m['mae_test']:.4f}", f"{m['rmse_test']:.4f}",
            ])
    print(f"Saved metrics CSV      -> {metrics_path}")

    # CSV 2: MDS/TM metrics
    mds_path = csv_dir / "per_protein_mds_tm_scores.csv"
    with open(mds_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "protein_id", "layer", "n_residues", "tm_score", "rmsd",
        ])
        for m in all_mds:
            writer.writerow([
                m["protein_id"], m["layer"], m["n_residues"],
                f"{m['tm_score']:.6f}", f"{m['rmsd']:.6f}",
            ])
    print(f"Saved MDS/TM CSV       -> {mds_path}")

    # CSV 3: Coefficients
    coef_path = csv_dir / "per_protein_coefficients.csv"
    coef_fieldnames = (
        ["protein_id", "layer", "intercept"]
        + [f"coef_{i}" for i in range(FEATURE_DIM)]
    )
    with open(coef_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=coef_fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_coefs)
    print(f"Saved coefficients CSV -> {coef_path}")

    # ------------------------------------------------------------------
    # Phase 5: Summary
    # ------------------------------------------------------------------
    summary_path = csv_dir / "per_protein_regression_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Per-Protein Linear Regression Summary\n")
        f.write("=" * 50 + "\n\n")
        f.write(f"Total discovered:  {n_total}\n")
        f.write(f"Succeeded:         {len(succeeded)}\n")
        f.write(f"Skipped:           {len(skipped)}\n")
        f.write(f"Test fraction:     {TEST_SIZE}\n")
        f.write(f"Random state:      {RANDOM_STATE}\n")
        f.write(f"Feature dim:       {FEATURE_DIM}\n")
        f.write(f"Layers:            {n_layers}\n\n")

        if succeeded:
            r2_47_values = [s["r2_layer47"] for s in succeeded]
            f.write("Layer 47 R2 (test) across proteins:\n")
            f.write(f"  Min:    {min(r2_47_values):.6f}\n")
            f.write(f"  Median: {np.median(r2_47_values):.6f}\n")
            f.write(f"  Mean:   {np.mean(r2_47_values):.6f}\n")
            f.write(f"  Max:    {max(r2_47_values):.6f}\n\n")

            f.write("Per-protein summary (sorted by layer-47 R2):\n")
            for s in sorted(succeeded, key=lambda x: x["r2_layer47"], reverse=True):
                f.write(f"  {s['protein_id']}: {s['n_residues']} res, "
                        f"{s['n_pairs']:,} pairs, "
                        f"best layer={s['best_layer']} (R2={s['best_r2']:.4f}), "
                        f"layer47 R2={s['r2_layer47']:.6f}\n")

        if skipped:
            f.write(f"\nSkipped proteins ({len(skipped)}):\n")
            for pid, reason in skipped:
                f.write(f"  {pid}: {reason}\n")

    print(f"Saved summary          -> {summary_path}")

    # Console summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Succeeded: {len(succeeded)} / {n_total}")
    print(f"Skipped:   {len(skipped)}")
    if succeeded:
        r2_47 = [s["r2_layer47"] for s in succeeded]
        print(f"Layer 47 R2: min={min(r2_47):.4f}, median={np.median(r2_47):.4f}, "
              f"mean={np.mean(r2_47):.4f}, max={max(r2_47):.4f}")


if __name__ == "__main__":
    main()

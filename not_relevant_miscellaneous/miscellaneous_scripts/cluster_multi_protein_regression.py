#!/usr/bin/env python3
"""
cluster_multi_protein_regression.py
===================================
Full-cluster, 5-fold protein-level regression with MDS reconstruction and
triangle inequality analysis across all discovered proteins.

This is the production, all-protein version of the `test2` workflow:
  1. Protein-level 5-fold CV (no pair leakage across train/test proteins)
  2. Layer-wise `StandardScaler + LinearRegression`
  3. Classical MDS reconstruction of predicted distance matrices
  4. Geometric diagnostics (TM-score, RMSD, stress, eigenvalue metrics,
     triangle inequality violations) on both train and test proteins

The main regression loop is ordered as `layer -> fold` so each protein's pair
block for a given layer is loaded once and reused across all folds.

Cluster layout (configured via environment variables set in sbatch files):
  - Pair blocks:  PAIR_BLOCKS_ROOT/{protein_id}/{protein_id}_pair_block_{layer}.npy
  - PDB files:    PDB_ROOT/my_outputs_align_{id}_representations_load_from_saved_layer_47/predictions/{id}_model_1_ptm_unrelaxed.pdb
  - Output CSVs:  CSV_ROOT/

Outputs (inside CSV_ROOT/):
    kfold_regression_r2.csv              — per-(fold, layer) R2, MAE, RMSE
    kfold_regression_r2_summary.csv      — per-layer mean +/- std across folds
    kfold_regression_coefficients.csv    — model coefficients per (fold, layer)
    kfold_mds_metrics.csv                — per-protein geometry metrics (train + test)
    kfold_mds_summary.csv                — per-layer geometry summary across all rows
    kfold_predicted_distances.csv        — actual vs predicted distances (last fold, last layer)
    kfold_regression_summary.txt         — human-readable summary
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
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from Bio.PDB import PDBParser


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48
N_CHANNELS = 128
FEATURE_DIM = 2 * N_CHANNELS  # 256
N_FOLDS = 5
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
    """Vectorized pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


# ===========================================================================
# Helper — build per-protein feature chunks and stack from cache
# ===========================================================================

def build_feature_chunk(protein_record, layer):
    """Load one protein/layer pair block and extract its dense feature chunk."""
    pb_path = get_pair_block_path(
        protein_record["protein_dir"], protein_record["protein_id"], layer
    )
    pb = np.load(str(pb_path))
    chunk = np.concatenate([
        pb[protein_record["pb_i"], protein_record["pb_j"], :],
        pb[protein_record["pb_j"], protein_record["pb_i"], :],
    ], axis=1)
    del pb
    return chunk


def stack_cached_chunks(proteins, chunk_map):
    """Stack cached per-protein chunks in the supplied protein order."""
    return np.vstack([chunk_map[d["protein_id"]] for d in proteins])


# ===========================================================================
# Classical MDS
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

    return coords, eigenvalues


# ===========================================================================
# Superimposition and TM-score (inline for cluster portability)
# ===========================================================================

def superimpose_structures(fixed_coords, moving_coords):
    """Kabsch algorithm for optimal rotation + translation."""
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

    return transformed, rmsd


def compute_tm_score(fixed_coords, transformed_coords, length_target):
    """Compute TM-score between two superimposed structures."""
    d0 = 1.24 * (max(length_target, 16) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)

    distances = np.sqrt(np.sum((fixed_coords - transformed_coords) ** 2, axis=1))
    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    return float(np.sum(scores) / length_target)


# ===========================================================================
# Triangle inequality analysis
# ===========================================================================

def compute_triangle_violations(pred_dist):
    """Compute triangle inequality violations using min-indirect-path approach."""
    n_residues = pred_dist.shape[0]
    min_indirect = np.full((n_residues, n_residues), np.inf)
    for j in range(n_residues):
        candidate = pred_dist[:, j:j + 1] + pred_dist[j:j + 1, :]
        min_indirect = np.minimum(min_indirect, candidate)
    np.fill_diagonal(min_indirect, 0.0)

    mask = pred_dist > min_indirect
    np.fill_diagonal(mask, False)
    n_violations = mask.sum()
    total_pairs = n_residues * (n_residues - 1)

    violation_fraction = float(n_violations / total_pairs) if total_pairs > 0 else 0.0

    if n_violations > 0:
        violations = pred_dist[mask] - min_indirect[mask]
        mean_violation = float(violations.mean())
        max_violation = float(violations.max())
    else:
        mean_violation = 0.0
        max_violation = 0.0

    return violation_fraction, mean_violation, max_violation


# ===========================================================================
# MDS + geometric analysis for one protein
# ===========================================================================

def analyze_protein_geometry(n_residues, true_coords, pb_i, pb_j, y_pred):
    """Run MDS reconstruction + triangle inequality analysis on one protein."""
    pred_dist = np.zeros((n_residues, n_residues))
    pred_dist[pb_i, pb_j] = y_pred
    pred_dist[pb_j, pb_i] = y_pred

    n_negative_preds = int(np.sum(y_pred < 0))
    pred_dist = np.maximum(pred_dist, 0.0)

    mds_coords, eigenvalues = classical_mds(pred_dist, n_dims=3)

    pos_mask = eigenvalues > 0
    sum_pos = np.sum(eigenvalues[pos_mask]) if np.any(pos_mask) else 1.0
    frac_top3 = float(np.sum(np.maximum(eigenvalues[:3], 0.0)) / sum_pos)
    n_neg_evals = int(np.sum(eigenvalues < 0))
    neg_frac = (
        float(abs(np.sum(eigenvalues[eigenvalues < 0])) / sum_pos)
        if n_neg_evals > 0 else 0.0
    )

    mds_reflected = mds_coords.copy()
    mds_reflected[:, 2] *= -1

    transformed_orig, rmsd_orig = superimpose_structures(true_coords, mds_coords)
    transformed_flip, rmsd_flip = superimpose_structures(true_coords, mds_reflected)

    if rmsd_flip < rmsd_orig:
        transformed, rmsd = transformed_flip, rmsd_flip
    else:
        transformed, rmsd = transformed_orig, rmsd_orig

    tm_score = compute_tm_score(true_coords, transformed, n_residues)

    mds_dist = compute_distance_matrix(transformed)
    triu_i, triu_j = np.triu_indices(n_residues, k=1)
    d_pred = pred_dist[triu_i, triu_j]
    d_mds = mds_dist[triu_i, triu_j]
    stress = float(np.sqrt(np.sum((d_pred - d_mds) ** 2) / np.sum(d_pred ** 2)))

    viol_frac, viol_mean, viol_max = compute_triangle_violations(pred_dist)

    return {
        "rmsd": rmsd,
        "tm_score": tm_score,
        "frac_top3_eigenvalues": frac_top3,
        "n_negative_eigenvalues": n_neg_evals,
        "neg_eigenvalue_fraction": neg_frac,
        "n_negative_predictions": n_negative_preds,
        "stress": stress,
        "triangle_violation_fraction": viol_frac,
        "triangle_mean_violation": viol_mean,
        "triangle_max_violation": viol_max,
    }


def make_failed_geometry_row(fold_idx, layer, protein_id, split, n_residues, per_protein_r2):
    """Return a placeholder per-protein row when geometry analysis fails."""
    return {
        "fold": fold_idx,
        "layer": layer,
        "protein_id": protein_id,
        "split": split,
        "n_residues": n_residues,
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


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()
    n_layers = N_LAYERS

    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print("FULL CLUSTER RUN — all proteins, 5-fold CV")
    print("with MDS reconstruction + triangle inequality analysis")
    print(f"{'=' * 60}")
    print(f"Pair blocks root:   {PAIR_BLOCKS_ROOT}")
    print(f"PDB root:           {PDB_ROOT}")
    print(f"Output CSV dir:     {csv_dir}")
    print(f"Layers:             {n_layers}")
    print(f"Split strategy:     {N_FOLDS}-FOLD CROSS-VALIDATION (protein-level)")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------
    all_proteins = discover_proteins(PAIR_BLOCKS_ROOT)
    print(f"\nDiscovered {len(all_proteins)} protein directories total.")
    print(f"Using all {len(all_proteins)} discovered proteins.")

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
                "coordinates": ca_data["coordinates"],
                "pb_i": struct_i,
                "pb_j": struct_j,
                "y": y_protein,
            })
            print(f"[OK]   {protein_id}: {n_ca} residues, {len(y_protein):,} pairs")

        except Exception as exc:
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {exc}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    n_proteins = len(protein_data)
    if n_proteins < N_FOLDS:
        print(
            f"\nERROR: Need at least {N_FOLDS} proteins for {N_FOLDS}-fold CV. "
            f"Only {n_proteins} loaded. Exiting."
        )
        sys.exit(1)

    # ------------------------------------------------------------------
    # Phase 2: Set up K-fold cross-validation (protein-level)
    # ------------------------------------------------------------------
    kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    protein_indices = np.arange(n_proteins)
    folds = list(kf.split(protein_indices))

    print(f"\n{'=' * 60}")
    print(f"{N_FOLDS}-FOLD PROTEIN-LEVEL CROSS-VALIDATION")
    print(f"{'=' * 60}")
    total_pairs = sum(len(d["y"]) for d in protein_data)
    print(f"  Total proteins: {n_proteins}")
    print(f"  Total pairs:    {total_pairs:,}")

    fold_data = []
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        n_train_pairs = sum(len(protein_data[i]["y"]) for i in train_idx)
        n_test_pairs = sum(len(protein_data[i]["y"]) for i in test_idx)
        print(
            f"  Fold {fold_idx}: {len(train_idx)} train / {len(test_idx)} test proteins, "
            f"{n_train_pairs:,} train / {n_test_pairs:,} test pairs"
        )

        train_proteins = [protein_data[i] for i in train_idx]
        test_proteins = [protein_data[i] for i in test_idx]

        y_train = np.concatenate([d["y"] for d in train_proteins])
        y_test = np.concatenate([d["y"] for d in test_proteins])

        train_offsets = []
        offset = 0
        for d in train_proteins:
            n_pairs = len(d["y"])
            train_offsets.append((d["protein_id"], offset, offset + n_pairs))
            offset += n_pairs

        test_offsets = []
        offset = 0
        for d in test_proteins:
            n_pairs = len(d["y"])
            test_offsets.append((d["protein_id"], offset, offset + n_pairs))
            offset += n_pairs

        fold_data.append({
            "train_proteins": train_proteins,
            "test_proteins": test_proteins,
            "y_train": y_train,
            "y_test": y_test,
            "train_offsets": train_offsets,
            "test_offsets": test_offsets,
        })
    print(f"{'=' * 60}\n")

    # ------------------------------------------------------------------
    # Phase 3: K-fold cross-validated layer-wise regression + MDS
    # ------------------------------------------------------------------
    results_grid = [[None for _ in range(n_layers)] for _ in range(N_FOLDS)]
    coefs_grid = [[None for _ in range(n_layers)] for _ in range(N_FOLDS)]
    mds_rows = []
    last_fold_preds = None

    for layer in range(n_layers):
        print(f"--- Layer {layer:2d} (cached across all folds) ---")
        chunk_map = {
            d["protein_id"]: build_feature_chunk(d, layer)
            for d in protein_data
        }

        for fold_idx, fold_info in enumerate(fold_data):
            train_proteins = fold_info["train_proteins"]
            test_proteins = fold_info["test_proteins"]
            y_train = fold_info["y_train"]
            y_test = fold_info["y_test"]
            train_offsets = fold_info["train_offsets"]
            test_offsets = fold_info["test_offsets"]

            try:
                X_train_raw = stack_cached_chunks(train_proteins, chunk_map)
                X_test_raw = stack_cached_chunks(test_proteins, chunk_map)

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

                results_grid[fold_idx][layer] = {
                    "r2_test": r2_test,
                    "r2_train": r2_train,
                    "mae_test": mae,
                    "rmse_test": rmse,
                }

                coef_row = {
                    "fold": fold_idx,
                    "layer": layer,
                    "intercept": float(model.intercept_),
                }
                coef_row.update({
                    f"coef_{i}": float(c)
                    for i, c in enumerate(model.coef_)
                })
                coefs_grid[fold_idx][layer] = coef_row

                print(
                    f"  Fold {fold_idx} Layer {layer:2d}  "
                    f"R2_test={r2_test:.4f}  R2_train={r2_train:.4f}  "
                    f"MAE={mae:.2f}  RMSE={rmse:.2f}",
                    end="",
                )

                tm_train_vals = []
                for d, (pid, start, end) in zip(train_proteins, train_offsets):
                    per_protein_r2 = (
                        float(r2_score(y_train[start:end], y_pred_train[start:end]))
                        if end - start > 1 else float("nan")
                    )
                    try:
                        metrics = analyze_protein_geometry(
                            d["n_residues"],
                            d["coordinates"],
                            d["pb_i"],
                            d["pb_j"],
                            y_pred_train[start:end],
                        )
                        mds_rows.append({
                            "fold": fold_idx,
                            "layer": layer,
                            "protein_id": pid,
                            "split": "train",
                            "n_residues": d["n_residues"],
                            "per_protein_r2": per_protein_r2,
                            **metrics,
                        })
                        tm_train_vals.append(metrics["tm_score"])
                    except Exception as exc:
                        print(f"\n    Train MDS failed for {pid}: {exc}")
                        mds_rows.append(make_failed_geometry_row(
                            fold_idx,
                            layer,
                            pid,
                            "train",
                            d["n_residues"],
                            per_protein_r2,
                        ))

                tm_test_vals = []
                for d, (pid, start, end) in zip(test_proteins, test_offsets):
                    per_protein_r2 = (
                        float(r2_score(y_test[start:end], y_pred_test[start:end]))
                        if end - start > 1 else float("nan")
                    )
                    try:
                        metrics = analyze_protein_geometry(
                            d["n_residues"],
                            d["coordinates"],
                            d["pb_i"],
                            d["pb_j"],
                            y_pred_test[start:end],
                        )
                        mds_rows.append({
                            "fold": fold_idx,
                            "layer": layer,
                            "protein_id": pid,
                            "split": "test",
                            "n_residues": d["n_residues"],
                            "per_protein_r2": per_protein_r2,
                            **metrics,
                        })
                        tm_test_vals.append(metrics["tm_score"])
                    except Exception as exc:
                        print(f"\n    Test MDS failed for {pid}: {exc}")
                        mds_rows.append(make_failed_geometry_row(
                            fold_idx,
                            layer,
                            pid,
                            "test",
                            d["n_residues"],
                            per_protein_r2,
                        ))

                tm_train_str = f"{np.mean(tm_train_vals):.4f}" if tm_train_vals else "nan"
                tm_test_str = f"{np.mean(tm_test_vals):.4f}" if tm_test_vals else "nan"
                print(f"  TM_train={tm_train_str}  TM_test={tm_test_str}")

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

            except Exception as exc:
                print(f"  Fold {fold_idx} Layer {layer:2d}  ERROR: {exc}")
                traceback.print_exc()
                results_grid[fold_idx][layer] = {
                    "r2_test": float("nan"),
                    "r2_train": float("nan"),
                    "mae_test": float("nan"),
                    "rmse_test": float("nan"),
                }
                coefs_grid[fold_idx][layer] = {
                    "fold": fold_idx,
                    "layer": layer,
                    "intercept": 0.0,
                }

        del chunk_map

    # ------------------------------------------------------------------
    # Phase 4: Aggregate metrics across folds
    # ------------------------------------------------------------------
    layer_summaries = []
    for layer in range(n_layers):
        r2_tests = [
            results_grid[f][layer]["r2_test"]
            for f in range(N_FOLDS)
            if not math.isnan(results_grid[f][layer]["r2_test"])
        ]
        r2_trains = [
            results_grid[f][layer]["r2_train"]
            for f in range(N_FOLDS)
            if not math.isnan(results_grid[f][layer]["r2_train"])
        ]
        maes = [
            results_grid[f][layer]["mae_test"]
            for f in range(N_FOLDS)
            if not math.isnan(results_grid[f][layer]["mae_test"])
        ]
        rmses = [
            results_grid[f][layer]["rmse_test"]
            for f in range(N_FOLDS)
            if not math.isnan(results_grid[f][layer]["rmse_test"])
        ]

        layer_summaries.append({
            "r2_test_mean": np.mean(r2_tests) if r2_tests else float("nan"),
            "r2_test_std": np.std(r2_tests) if r2_tests else float("nan"),
            "r2_train_mean": np.mean(r2_trains) if r2_trains else float("nan"),
            "r2_train_std": np.std(r2_trains) if r2_trains else float("nan"),
            "mae_mean": np.mean(maes) if maes else float("nan"),
            "mae_std": np.std(maes) if maes else float("nan"),
            "rmse_mean": np.mean(rmses) if rmses else float("nan"),
            "rmse_std": np.std(rmses) if rmses else float("nan"),
        })

    # ------------------------------------------------------------------
    # Phase 5: Save CSV outputs
    # ------------------------------------------------------------------
    r2_csv = csv_dir / "kfold_regression_r2.csv"
    with open(r2_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
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

    summary_csv = csv_dir / "kfold_regression_r2_summary.csv"
    with open(summary_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "layer",
            "r2_test_mean", "r2_test_std",
            "r2_train_mean", "r2_train_std",
            "mae_mean", "mae_std",
            "rmse_mean", "rmse_std",
        ])
        for layer, summary in enumerate(layer_summaries):
            writer.writerow([
                layer,
                f"{summary['r2_test_mean']:.6f}", f"{summary['r2_test_std']:.6f}",
                f"{summary['r2_train_mean']:.6f}", f"{summary['r2_train_std']:.6f}",
                f"{summary['mae_mean']:.4f}", f"{summary['mae_std']:.4f}",
                f"{summary['rmse_mean']:.4f}", f"{summary['rmse_std']:.4f}",
            ])
    print(f"Saved summary CSV      -> {summary_csv}")

    coef_csv = csv_dir / "kfold_regression_coefficients.csv"
    fieldnames = ["fold", "layer", "intercept"] + [f"coef_{i}" for i in range(FEATURE_DIM)]
    with open(coef_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        for fold_coefs in coefs_grid:
            writer.writerows(fold_coefs)
    print(f"Saved coefficients CSV -> {coef_csv}")

    mds_csv = csv_dir / "kfold_mds_metrics.csv"
    mds_fields = [
        "fold", "layer", "protein_id", "split", "n_residues", "per_protein_r2",
        "rmsd", "tm_score",
        "frac_top3_eigenvalues", "n_negative_eigenvalues", "neg_eigenvalue_fraction",
        "n_negative_predictions", "stress",
        "triangle_violation_fraction", "triangle_mean_violation", "triangle_max_violation",
    ]
    with open(mds_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=mds_fields)
        writer.writeheader()
        for row in mds_rows:
            row_fmt = {
                key: (f"{value:.6f}" if isinstance(value, float) else value)
                for key, value in row.items()
            }
            writer.writerow(row_fmt)
    print(f"Saved MDS metrics CSV -> {mds_csv}")

    mds_summary_csv = csv_dir / "kfold_mds_summary.csv"
    mds_metric_keys = [
        "rmsd", "tm_score", "frac_top3_eigenvalues", "neg_eigenvalue_fraction",
        "stress", "triangle_violation_fraction", "triangle_mean_violation",
        "triangle_max_violation",
    ]
    with open(mds_summary_csv, "w", newline="") as handle:
        header = ["layer"]
        for key in mds_metric_keys:
            header.extend([f"{key}_mean", f"{key}_std"])
        writer = csv.writer(handle)
        writer.writerow(header)

        for layer in range(n_layers):
            layer_rows = [row for row in mds_rows if row["layer"] == layer]
            row_out = [layer]
            for key in mds_metric_keys:
                vals = [
                    row[key]
                    for row in layer_rows
                    if isinstance(row[key], (int, float)) and not math.isnan(row[key])
                ]
                if vals:
                    row_out.extend([f"{np.mean(vals):.6f}", f"{np.std(vals):.6f}"])
                else:
                    row_out.extend(["nan", "nan"])
            writer.writerow(row_out)
    print(f"Saved MDS summary CSV -> {mds_summary_csv}")

    if last_fold_preds is not None:
        pred_csv = csv_dir / "kfold_predicted_distances.csv"
        last_preds = last_fold_preds
        with open(pred_csv, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow([
                "protein_id", "struct_i", "struct_j",
                "actual_distance_A", "predicted_distance_A", "split",
            ])
            for pid, start, end in last_preds["train_offsets"]:
                protein_record = next(
                    x for x in last_preds["train_proteins"] if x["protein_id"] == pid
                )
                for offset in range(end - start):
                    writer.writerow([
                        pid,
                        int(protein_record["pb_i"][offset]),
                        int(protein_record["pb_j"][offset]),
                        f"{last_preds['y_train'][start + offset]:.4f}",
                        f"{last_preds['y_pred_train'][start + offset]:.4f}",
                        "train",
                    ])
            for pid, start, end in last_preds["test_offsets"]:
                protein_record = next(
                    x for x in last_preds["test_proteins"] if x["protein_id"] == pid
                )
                for offset in range(end - start):
                    writer.writerow([
                        pid,
                        int(protein_record["pb_i"][offset]),
                        int(protein_record["pb_j"][offset]),
                        f"{last_preds['y_test'][start + offset]:.4f}",
                        f"{last_preds['y_pred_test'][start + offset]:.4f}",
                        "test",
                    ])
        print(f"Saved predicted distances CSV -> {pred_csv}")
    else:
        print("WARNING: Last fold/layer failed; predicted distances CSV not saved.")

    summary_path = csv_dir / "kfold_regression_summary.txt"
    with open(summary_path, "w") as handle:
        handle.write(f"{N_FOLDS}-Fold Cross-Validated Linear Regression\n")
        handle.write("with MDS Reconstruction + Triangle Inequality Analysis\n")
        handle.write("(All discovered proteins)\n")
        handle.write("=" * 55 + "\n\n")
        handle.write(f"Split strategy:  {N_FOLDS}-FOLD PROTEIN-LEVEL CV\n")
        handle.write(f"Random state:    {RANDOM_STATE}\n")
        handle.write(f"Feature dim:     {FEATURE_DIM}\n")
        handle.write(f"Layers:          {n_layers}\n")
        handle.write(f"Total proteins:  {n_proteins}\n")
        handle.write(f"Total pairs:     {total_pairs:,}\n")
        handle.write(f"Skipped:         {len(skipped)}\n\n")

        handle.write("Fold breakdown:\n")
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            test_ids = [protein_data[i]["protein_id"] for i in test_idx]
            n_train_pairs = sum(len(protein_data[i]["y"]) for i in train_idx)
            n_test_pairs = sum(len(protein_data[i]["y"]) for i in test_idx)
            handle.write(
                f"  Fold {fold_idx}: {len(train_idx)} train / {len(test_idx)} test proteins, "
                f"{n_train_pairs:,} / {n_test_pairs:,} pairs\n"
            )
            handle.write(f"         Test proteins: {', '.join(test_ids)}\n")

        valid = [
            (layer, summary["r2_test_mean"])
            for layer, summary in enumerate(layer_summaries)
            if not math.isnan(summary["r2_test_mean"])
        ]
        if valid:
            best_layer, best_r2 = max(valid, key=lambda x: x[1])
            best_std = layer_summaries[best_layer]["r2_test_std"]
            handle.write(
                f"\nBest layer:      {best_layer} "
                f"(R2_test = {best_r2:.6f} +/- {best_std:.6f})\n"
            )

            last = layer_summaries[-1]
            handle.write(
                f"Last layer (47): R2_test = {last['r2_test_mean']:.6f} "
                f"+/- {last['r2_test_std']:.6f}\n"
            )
            handle.write(
                f"                 R2_train = {last['r2_train_mean']:.6f} "
                f"+/- {last['r2_train_std']:.6f}\n"
            )
            handle.write(
                f"                 MAE = {last['mae_mean']:.4f} "
                f"+/- {last['mae_std']:.4f}\n"
            )
            handle.write(
                f"                 RMSE = {last['rmse_mean']:.4f} "
                f"+/- {last['rmse_std']:.4f}\n"
            )

        handle.write("\nLayer 47 MDS results:\n")
        for row in mds_rows:
            if row["layer"] == n_layers - 1:
                handle.write(
                    f"  {row['protein_id']} ({row['split']}): TM={row['tm_score']:.4f}  "
                    f"RMSD={row['rmsd']:.2f}A  "
                    f"Top3EV={row['frac_top3_eigenvalues']:.4f}  "
                    f"TriViol={row['triangle_violation_fraction']:.6f}\n"
                )

        handle.write(
            "\nNote: kfold_mds_metrics.csv includes both train and test proteins, "
            "with per-protein R2 in the per_protein_r2 column.\n"
        )
        handle.write(
            "Note: kfold_predicted_distances.csv is retained for the last fold "
            "and last layer only.\n"
        )

        handle.write("\nLayer-by-layer R2_test (mean +/- std):\n")
        for layer, summary in enumerate(layer_summaries):
            handle.write(
                f"  Layer {layer:2d}: "
                f"{summary['r2_test_mean']:.6f} +/- {summary['r2_test_std']:.6f}\n"
            )

        if skipped:
            handle.write("\nSkipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")

    print(f"Saved summary    -> {summary_path}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print("FULL CLUSTER RUN COMPLETE")
    print(f"Elapsed time: {elapsed:.1f} seconds ({elapsed / 60.0:.1f} minutes)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
kfold_mds_regression.py
=======================
K-fold cross-validated linear regression with MDS reconstruction and
triangle inequality analysis. For each AlphaFold layer, trains a pooled
linear regression to predict CA-CA distances from pair-block features,
then tests geometric coherence via:
  1. Classical MDS reconstruction (TM-score, RMSD, stress)
  2. Eigenvalue diagnostics (top-3 fraction, negative eigenvalue fraction)
  3. Triangle inequality violations (min-indirect-path approach)

Local version: uses lib/ imports, runs on all locally available proteins
with 3-fold CV.

Outputs (inside proteins/multi_protein_results/kfold_mds/):
    csv_files/kfold_regression_r2.csv
    csv_files/kfold_regression_r2_summary.csv
    csv_files/kfold_mds_metrics.csv
    csv_files/kfold_mds_summary.csv
    visualizations/mds_tmscore_vs_layer.png
    visualizations/mds_rmsd_vs_layer.png
    visualizations/triangle_violation_vs_layer.png
    visualizations/r2_vs_tmscore.png
"""

import csv
import math
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score, mean_absolute_error

# lib/ imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from lib.data_loading import discover_proteins, find_pdb_file, get_pair_block_path
from lib.structure import (
    extract_ca_coordinates,
    compute_distance_matrix,
    superimpose_structures,
    compute_tm_score,
)


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48
N_CHANNELS = 128
FEATURE_DIM = 2 * N_CHANNELS  # 256
N_FOLDS = 3
RANDOM_STATE = 42

PROTEINS_DIR = PROJECT_ROOT / "proteins"
OUTPUT_DIR = PROTEINS_DIR / "multi_protein_results" / "kfold_mds"


# ===========================================================================
# Classical MDS (from mds_reconstruction.py)
# ===========================================================================

def classical_mds(distance_matrix, n_dims=3):
    """Classical (Torgerson) MDS: recover coordinates from pairwise distances.

    Returns (coords, eigenvalues) where eigenvalues are sorted descending.
    """
    n = distance_matrix.shape[0]
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
# Triangle inequality analysis
# ===========================================================================

def compute_triangle_violations(pred_dist):
    """Compute triangle inequality violations using min-indirect-path approach.

    For each pair (i,k), finds the minimum indirect path min_j(d(i,j) + d(j,k)).
    A violation occurs when d(i,k) > min_indirect(i,k).

    Returns (violation_fraction, mean_violation, max_violation).
    """
    R = pred_dist.shape[0]
    min_indirect = np.full((R, R), np.inf)
    for j in range(R):
        candidate = pred_dist[:, j:j+1] + pred_dist[j:j+1, :]
        min_indirect = np.minimum(min_indirect, candidate)
    np.fill_diagonal(min_indirect, 0.0)

    mask = pred_dist > min_indirect
    # Exclude diagonal from count
    np.fill_diagonal(mask, False)
    n_violations = mask.sum()
    total_pairs = R * (R - 1)

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
    """Run MDS reconstruction + triangle inequality analysis on one protein's
    predicted distances.

    Returns a dict of metrics, or a dict of NaNs if analysis fails.
    """
    R = n_residues

    # Reconstruct symmetric predicted distance matrix
    pred_dist = np.zeros((R, R))
    pred_dist[pb_i, pb_j] = y_pred
    pred_dist[pb_j, pb_i] = y_pred

    n_negative_preds = int(np.sum(y_pred < 0))
    pred_dist = np.maximum(pred_dist, 0.0)

    # Classical MDS
    mds_coords, eigenvalues = classical_mds(pred_dist, n_dims=3)

    # Eigenvalue diagnostics
    pos_mask = eigenvalues > 0
    sum_pos = np.sum(eigenvalues[pos_mask]) if np.any(pos_mask) else 1.0
    frac_top3 = float(np.sum(np.maximum(eigenvalues[:3], 0.0)) / sum_pos)
    n_neg_evals = int(np.sum(eigenvalues < 0))
    neg_frac = float(abs(np.sum(eigenvalues[eigenvalues < 0])) / sum_pos) if n_neg_evals > 0 else 0.0

    # Kabsch alignment with reflection handling
    mds_reflected = mds_coords.copy()
    mds_reflected[:, 2] *= -1

    transformed_orig, rmsd_orig = superimpose_structures(true_coords, mds_coords)
    transformed_flip, rmsd_flip = superimpose_structures(true_coords, mds_reflected)

    if rmsd_flip < rmsd_orig:
        transformed, rmsd = transformed_flip, rmsd_flip
    else:
        transformed, rmsd = transformed_orig, rmsd_orig

    tm = compute_tm_score(true_coords, transformed, R)

    # Stress
    mds_dist = compute_distance_matrix(transformed)
    triu_i, triu_j = np.triu_indices(R, k=1)
    d_pred = pred_dist[triu_i, triu_j]
    d_mds = mds_dist[triu_i, triu_j]
    stress = float(np.sqrt(np.sum((d_pred - d_mds) ** 2) / np.sum(d_pred ** 2)))

    # Triangle inequality violations
    viol_frac, viol_mean, viol_max = compute_triangle_violations(pred_dist)

    return {
        "rmsd": rmsd,
        "tm_score": tm,
        "frac_top3_eigenvalues": frac_top3,
        "n_negative_eigenvalues": n_neg_evals,
        "neg_eigenvalue_fraction": neg_frac,
        "n_negative_predictions": n_negative_preds,
        "stress": stress,
        "triangle_violation_fraction": viol_frac,
        "triangle_mean_violation": viol_mean,
        "triangle_max_violation": viol_max,
    }


# ===========================================================================
# Feature matrix construction
# ===========================================================================

def build_feature_matrix(proteins, layer):
    """Load pair blocks and build concatenated [i,j]+[j,i] feature matrix."""
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
    t_start = time.time()

    proteins_dir = PROTEINS_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    csv_dir = output_dir / "csv_files"
    vis_dir = output_dir / "visualizations"
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'='*60}")
    print(f"K-fold MDS Regression (local)")
    print(f"{'='*60}")
    print(f"Proteins directory: {proteins_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Layers:             {N_LAYERS}")
    print(f"Folds:              {N_FOLDS}")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------
    all_proteins = discover_proteins(proteins_dir)
    print(f"\nDiscovered {len(all_proteins)} protein directories.")

    protein_data = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        try:
            pdb_path = find_pdb_file(protein_dir, protein_id)
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
    # Phase 3: K-fold cross-validated layer-wise regression + MDS
    # ------------------------------------------------------------------
    results_grid = []   # [fold][layer] -> regression metrics dict
    mds_rows = []       # flat list of per-(fold, layer, protein) MDS metrics

    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_proteins = [protein_data[i] for i in train_idx]
        test_proteins = [protein_data[i] for i in test_idx]

        y_train = np.concatenate([d["y"] for d in train_proteins])
        y_test = np.concatenate([d["y"] for d in test_proteins])

        test_offsets = []
        offset = 0
        for d in test_proteins:
            n = len(d["y"])
            test_offsets.append((d["protein_id"], offset, offset + n))
            offset += n

        fold_results = []

        print(f"--- Fold {fold_idx} ({len(train_idx)} train / {len(test_idx)} test proteins) ---")

        for layer in range(N_LAYERS):
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

                fold_results.append({
                    "r2_test": r2_test, "r2_train": r2_train,
                    "mae_test": mae, "rmse_test": rmse,
                })

                print(f"  Fold {fold_idx} Layer {layer:2d}  "
                      f"R2_test={r2_test:.4f}  R2_train={r2_train:.4f}  "
                      f"MAE={mae:.2f}  RMSE={rmse:.2f}", end="")

                # --- MDS + triangle inequality on test proteins ---
                tm_vals = []
                for d, (pid, start, end) in zip(test_proteins, test_offsets):
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
                            "n_residues": d["n_residues"],
                            **metrics,
                        })
                        tm_vals.append(metrics["tm_score"])
                    except Exception as e:
                        print(f"\n    MDS failed for {pid}: {e}")
                        mds_rows.append({
                            "fold": fold_idx, "layer": layer,
                            "protein_id": pid, "n_residues": d["n_residues"],
                            "rmsd": float("nan"), "tm_score": float("nan"),
                            "frac_top3_eigenvalues": float("nan"),
                            "n_negative_eigenvalues": -1,
                            "neg_eigenvalue_fraction": float("nan"),
                            "n_negative_predictions": -1,
                            "stress": float("nan"),
                            "triangle_violation_fraction": float("nan"),
                            "triangle_mean_violation": float("nan"),
                            "triangle_max_violation": float("nan"),
                        })

                if tm_vals:
                    print(f"  TM={np.mean(tm_vals):.4f}")
                else:
                    print()

                del X_train_raw, X_test_raw, X_train, X_test

            except Exception as e:
                print(f"  Fold {fold_idx} Layer {layer:2d}  ERROR: {e}")
                traceback.print_exc()
                fold_results.append({
                    "r2_test": float("nan"), "r2_train": float("nan"),
                    "mae_test": float("nan"), "rmse_test": float("nan"),
                })

        results_grid.append(fold_results)

    # ------------------------------------------------------------------
    # Phase 4: Aggregate regression metrics across folds
    # ------------------------------------------------------------------
    layer_summaries = []
    for layer in range(N_LAYERS):
        r2_tests = [results_grid[f][layer]["r2_test"] for f in range(N_FOLDS)
                     if not math.isnan(results_grid[f][layer]["r2_test"])]
        r2_trains = [results_grid[f][layer]["r2_train"] for f in range(N_FOLDS)
                      if not math.isnan(results_grid[f][layer]["r2_train"])]
        maes = [results_grid[f][layer]["mae_test"] for f in range(N_FOLDS)
                if not math.isnan(results_grid[f][layer]["mae_test"])]
        rmses = [results_grid[f][layer]["rmse_test"] for f in range(N_FOLDS)
                 if not math.isnan(results_grid[f][layer]["rmse_test"])]

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

    # CSV 1: Per-fold regression metrics
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
            for layer in range(N_LAYERS):
                res = results_grid[fold_idx][layer]
                writer.writerow([
                    fold_idx, layer,
                    f"{res['r2_test']:.6f}", f"{res['r2_train']:.6f}",
                    f"{res['mae_test']:.4f}", f"{res['rmse_test']:.4f}",
                    len(train_idx), len(test_idx),
                    n_train_pairs, n_test_pairs,
                ])
    print(f"\nSaved per-fold regression CSV -> {r2_csv}")

    # CSV 2: Regression summary
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
    print(f"Saved regression summary CSV -> {summary_csv}")

    # CSV 3: MDS metrics (per fold, layer, protein)
    mds_csv = csv_dir / "kfold_mds_metrics.csv"
    mds_fields = [
        "fold", "layer", "protein_id", "n_residues",
        "rmsd", "tm_score",
        "frac_top3_eigenvalues", "n_negative_eigenvalues", "neg_eigenvalue_fraction",
        "n_negative_predictions", "stress",
        "triangle_violation_fraction", "triangle_mean_violation", "triangle_max_violation",
    ]
    with open(mds_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=mds_fields)
        writer.writeheader()
        for row in mds_rows:
            row_fmt = {k: (f"{v:.6f}" if isinstance(v, float) else v)
                       for k, v in row.items()}
            writer.writerow(row_fmt)
    print(f"Saved MDS metrics CSV -> {mds_csv}")

    # CSV 4: MDS summary (per layer, aggregated across all proteins)
    mds_summary_csv = csv_dir / "kfold_mds_summary.csv"
    mds_metric_keys = [
        "rmsd", "tm_score", "frac_top3_eigenvalues", "neg_eigenvalue_fraction",
        "stress", "triangle_violation_fraction", "triangle_mean_violation",
        "triangle_max_violation",
    ]
    with open(mds_summary_csv, "w", newline="") as f:
        header = ["layer"]
        for key in mds_metric_keys:
            header.extend([f"{key}_mean", f"{key}_std"])
        writer = csv.writer(f)
        writer.writerow(header)

        for layer in range(N_LAYERS):
            layer_rows = [r for r in mds_rows if r["layer"] == layer]
            row_out = [layer]
            for key in mds_metric_keys:
                vals = [r[key] for r in layer_rows
                        if isinstance(r[key], (int, float)) and not math.isnan(r[key])]
                if vals:
                    row_out.extend([f"{np.mean(vals):.6f}", f"{np.std(vals):.6f}"])
                else:
                    row_out.extend(["nan", "nan"])
            writer.writerow(row_out)
    print(f"Saved MDS summary CSV -> {mds_summary_csv}")

    # ------------------------------------------------------------------
    # Phase 6: Visualizations
    # ------------------------------------------------------------------
    protein_ids = [d["protein_id"] for d in protein_data]
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Build per-protein arrays from mds_rows.
    # Each protein appears as test in exactly one fold, so one entry per layer.
    def get_protein_layer_values(metric_key):
        """Return {pid: [value_layer0, ..., value_layer47]}."""
        result = {pid: [None] * N_LAYERS for pid in protein_ids}
        for row in mds_rows:
            pid = row["protein_id"]
            if pid in result:
                result[pid][row["layer"]] = row[metric_key]
        return result

    tm_by_protein = get_protein_layer_values("tm_score")
    rmsd_by_protein = get_protein_layer_values("rmsd")
    viol_by_protein = get_protein_layer_values("triangle_violation_fraction")

    layers = list(range(N_LAYERS))

    # --- Plot 1: TM-score vs layer ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pid in enumerate(protein_ids):
        vals = tm_by_protein[pid]
        if any(v is not None for v in vals):
            plot_vals = [v if v is not None else float("nan") for v in vals]
            ax.plot(layers, plot_vals, marker=".", markersize=4,
                    linewidth=1.2, color=COLORS[i % len(COLORS)], label=pid, alpha=0.8)
    # Mean across proteins
    mean_tm = []
    for layer in layers:
        vals = [tm_by_protein[pid][layer] for pid in protein_ids
                if tm_by_protein[pid][layer] is not None
                and not math.isnan(tm_by_protein[pid][layer])]
        mean_tm.append(np.mean(vals) if vals else float("nan"))
    ax.plot(layers, mean_tm, "k-", linewidth=2.5, label="mean", alpha=0.9)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6,
               label="TM=0.5 (fold threshold)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("TM-score")
    ax.set_title("MDS Reconstruction: TM-score vs AlphaFold Layer (K-fold CV)")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "mds_tmscore_vs_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved TM-score plot -> {vis_dir / 'mds_tmscore_vs_layer.png'}")

    # --- Plot 2: RMSD vs layer ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pid in enumerate(protein_ids):
        vals = rmsd_by_protein[pid]
        if any(v is not None for v in vals):
            plot_vals = [v if v is not None else float("nan") for v in vals]
            ax.plot(layers, plot_vals, marker=".", markersize=4,
                    linewidth=1.2, color=COLORS[i % len(COLORS)], label=pid, alpha=0.8)
    mean_rmsd = []
    for layer in layers:
        vals = [rmsd_by_protein[pid][layer] for pid in protein_ids
                if rmsd_by_protein[pid][layer] is not None
                and not math.isnan(rmsd_by_protein[pid][layer])]
        mean_rmsd.append(np.mean(vals) if vals else float("nan"))
    ax.plot(layers, mean_rmsd, "k-", linewidth=2.5, label="mean", alpha=0.9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("RMSD (A)")
    ax.set_title("MDS Reconstruction: RMSD vs AlphaFold Layer (K-fold CV)")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "mds_rmsd_vs_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved RMSD plot -> {vis_dir / 'mds_rmsd_vs_layer.png'}")

    # --- Plot 3: Triangle violation fraction vs layer ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pid in enumerate(protein_ids):
        vals = viol_by_protein[pid]
        if any(v is not None for v in vals):
            plot_vals = [v if v is not None else float("nan") for v in vals]
            ax.plot(layers, plot_vals, marker=".", markersize=4,
                    linewidth=1.2, color=COLORS[i % len(COLORS)], label=pid, alpha=0.8)
    mean_viol = []
    for layer in layers:
        vals = [viol_by_protein[pid][layer] for pid in protein_ids
                if viol_by_protein[pid][layer] is not None
                and not math.isnan(viol_by_protein[pid][layer])]
        mean_viol.append(np.mean(vals) if vals else float("nan"))
    ax.plot(layers, mean_viol, "k-", linewidth=2.5, label="mean", alpha=0.9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Violation Fraction")
    ax.set_title("Triangle Inequality Violations vs AlphaFold Layer (K-fold CV)")
    ax.set_ylim(bottom=0)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "triangle_violation_vs_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved triangle violation plot -> {vis_dir / 'triangle_violation_vs_layer.png'}")

    # --- Plot 4: R2 vs TM-score scatter ---
    fig, ax = plt.subplots(figsize=(8, 7))
    # Build R2 by layer (from regression summaries)
    r2_by_layer = [s["r2_test_mean"] for s in layer_summaries]
    for i, pid in enumerate(protein_ids):
        tm_vals_pid = []
        r2_vals_pid = []
        for layer in range(N_LAYERS):
            tm_val = tm_by_protein[pid][layer]
            if tm_val is not None and not math.isnan(tm_val):
                tm_vals_pid.append(tm_val)
                r2_vals_pid.append(r2_by_layer[layer])
        if tm_vals_pid:
            ax.scatter(r2_vals_pid, tm_vals_pid, s=15, alpha=0.6,
                       color=COLORS[i % len(COLORS)], label=pid)
    ax.set_xlabel("Test R2 (mean across folds)")
    ax.set_ylabel("TM-score")
    ax.set_title("Regression R2 vs MDS Reconstruction TM-score")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "r2_vs_tmscore.png", dpi=150)
    plt.close(fig)
    print(f"Saved R2 vs TM-score plot -> {vis_dir / 'r2_vs_tmscore.png'}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    elapsed = time.time() - t_start
    print(f"\n{'='*60}")
    print(f"SUMMARY")
    print(f"{'='*60}")
    print(f"Included proteins ({n_proteins}):")
    for d in protein_data:
        print(f"  {d['protein_id']}: {d['n_residues']} residues")
    if skipped:
        print(f"\nSkipped proteins ({len(skipped)}):")
        for pid, reason in skipped:
            print(f"  {pid}: {reason}")

    # Last-layer summary
    print(f"\nLayer {N_LAYERS - 1} MDS results:")
    for pid in protein_ids:
        row = next((r for r in mds_rows
                     if r["layer"] == N_LAYERS - 1 and r["protein_id"] == pid), None)
        if row:
            print(f"  {pid}: TM={row['tm_score']:.4f}  RMSD={row['rmsd']:.2f}A  "
                  f"Top3EV={row['frac_top3_eigenvalues']:.4f}  "
                  f"TriViol={row['triangle_violation_fraction']:.6f}")

    print(f"\nElapsed time: {elapsed:.1f} seconds ({elapsed/60:.1f} minutes)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()

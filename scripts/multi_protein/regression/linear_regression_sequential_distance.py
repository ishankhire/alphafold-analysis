#!/usr/bin/env python3
"""
linear_regression_sequential_distance.py
=========================================
Trains linear regression models to predict sequential distance (j - i) between
residue pairs from AlphaFold pair representations across 48 layers.

For each layer, two separate models are trained:
  - Upper model: uses pb[i,j,:] (i→j direction) as the cross-pair feature
  - Lower model: uses pb[j,i,:] (j→i direction) as the cross-pair feature

Both models share the same diagonal features (pb[i,i,:] and pb[j,j,:]) and
the same train/test split. Comparing R² and RMSE across layers reveals whether
the asymmetric pair representation encodes directional sequence position information.

All pairs are from the upper triangle (i < j), so j - i is always positive.

Feature vector (384-dim):
    [cross_pair (128)] + [pb[i,i,:] (128)] + [pb[j,j,:] (128)]

Outputs (inside proteins/multi_protein_results/linear_regression_sequential_distance/):
    csv_files/layer_metrics.csv
    visualizations/layer_r2.png
    visualizations/scatter_layer47.png
"""

import csv
import sys
import os
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from lib.data_loading import discover_proteins, find_pdb_file, get_pair_block_path
from lib.structure import extract_ca_coordinates


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48
N_CHANNELS = 128
FEATURE_DIM = 3 * N_CHANNELS  # 384: one cross-pair direction (128) + two diagonal (256)
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_PROTEINS = 4

# ===========================================================================
# Hardcoded directories
# ===========================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROTEINS_DIR = PROJECT_ROOT / "proteins"
OUTPUT_DIR = PROTEINS_DIR / "multi_protein_results" / "linear_regression_sequential_distance"


# ===========================================================================
# Helper functions
# ===========================================================================

def build_sequential_distance_targets(n_residues):
    """Build (j - i) targets and upper-triangle indices for a protein.

    Since we restrict to i < j, j - i is always strictly positive.

    Parameters
    ----------
    n_residues : int

    Returns
    -------
    targets : np.ndarray of float, shape (n_pairs,)
        Sequential distance j - i for each pair.
    struct_i : np.ndarray of int, shape (n_pairs,)
    struct_j : np.ndarray of int, shape (n_pairs,)
    """
    struct_i, struct_j = np.triu_indices(n_residues, k=1)
    targets = (struct_j - struct_i).astype(float)
    return targets, struct_i, struct_j


# ===========================================================================
# Main
# ===========================================================================

def main():
    proteins_dir = PROTEINS_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()

    csv_dir = output_dir / "csv_files"
    vis_dir = output_dir / "visualizations"
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Proteins directory: {proteins_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Layers:             {N_LAYERS}")
    print(f"Feature dim:        {FEATURE_DIM} (128 cross-pair + 256 diagonal)")
    print(f"Max proteins:       {MAX_PROTEINS}")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------

    all_proteins = discover_proteins(proteins_dir)[:MAX_PROTEINS]
    print(f"\nUsing {len(all_proteins)} protein(s) (capped at {MAX_PROTEINS}).")

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

            targets, struct_i, struct_j = build_sequential_distance_targets(n_ca)

            protein_data.append({
                "protein_id": protein_id,
                "protein_dir": protein_dir,
                "n_residues": n_ca,
                "pb_i": struct_i,
                "pb_j": struct_j,
                "labels": targets,
            })
            print(f"[OK]   {protein_id}: {n_ca} residues, {len(targets):,} pairs, "
                  f"j-i range [{int(targets.min())}, {int(targets.max())}]")

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

    y_all = np.concatenate([d["labels"] for d in protein_data])
    n_total_pairs = len(y_all)

    print(f"\nTotal pairs across {n_proteins} protein(s): {n_total_pairs:,}")
    print(f"j-i range: [{int(y_all.min())}, {int(y_all.max())}]  mean={y_all.mean():.1f}  std={y_all.std():.1f}")

    all_idx = np.arange(n_total_pairs)
    idx_train, idx_test = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )

    y_train = y_all[idx_train]
    y_test = y_all[idx_test]

    print(f"Train: {len(idx_train):,}  Test: {len(idx_test):,}")

    # ------------------------------------------------------------------
    # Phase 3: Layer-wise regression
    # ------------------------------------------------------------------

    results = []
    last_layer_preds = None  # save predictions at layer 47 for scatterplot

    print()
    for layer in range(N_LAYERS):
        try:
            X_upper_chunks = []
            X_lower_chunks = []

            for d in protein_data:
                pb_path = get_pair_block_path(d["protein_dir"], d["protein_id"], layer)
                pb = np.load(str(pb_path))

                pi = d["pb_i"]
                pj = d["pb_j"]

                # Upper: i→j cross-pair direction + diagonals
                X_upper_chunk = np.concatenate([
                    pb[pi, pj, :],   # (n_pairs, 128): i→j
                    pb[pi, pi, :],   # (n_pairs, 128): i diagonal
                    pb[pj, pj, :],   # (n_pairs, 128): j diagonal
                ], axis=1)

                # Lower: j→i cross-pair direction + same diagonals
                X_lower_chunk = np.concatenate([
                    pb[pj, pi, :],   # (n_pairs, 128): j→i
                    pb[pi, pi, :],   # (n_pairs, 128): i diagonal
                    pb[pj, pj, :],   # (n_pairs, 128): j diagonal
                ], axis=1)

                X_upper_chunks.append(X_upper_chunk)
                X_lower_chunks.append(X_lower_chunk)
                del pb

            X_upper_all = np.vstack(X_upper_chunks)
            X_lower_all = np.vstack(X_lower_chunks)
            del X_upper_chunks, X_lower_chunks

            # Fit separate scalers for upper and lower
            scaler_upper = StandardScaler()
            X_upper_train = scaler_upper.fit_transform(X_upper_all[idx_train])
            X_upper_test = scaler_upper.transform(X_upper_all[idx_test])

            scaler_lower = StandardScaler()
            X_lower_train = scaler_lower.fit_transform(X_lower_all[idx_train])
            X_lower_test = scaler_lower.transform(X_lower_all[idx_test])

            del X_upper_all, X_lower_all

            model_upper = LinearRegression()
            model_upper.fit(X_upper_train, y_train)
            y_pred_upper = model_upper.predict(X_upper_test)

            model_lower = LinearRegression()
            model_lower.fit(X_lower_train, y_train)
            y_pred_lower = model_lower.predict(X_lower_test)

            r2_upper = r2_score(y_test, y_pred_upper)
            r2_lower = r2_score(y_test, y_pred_lower)
            rmse_upper = float(np.sqrt(np.mean((y_pred_upper - y_test) ** 2)))
            rmse_lower = float(np.sqrt(np.mean((y_pred_lower - y_test) ** 2)))

            results.append({
                "layer": layer,
                "r2_upper": r2_upper,
                "r2_lower": r2_lower,
                "rmse_upper": rmse_upper,
                "rmse_lower": rmse_lower,
            })
            print(f"Layer {layer:2d}  r2_upper={r2_upper:.4f}  r2_lower={r2_lower:.4f}"
                  f"  rmse_upper={rmse_upper:.2f}  rmse_lower={rmse_lower:.2f}")

            if layer == N_LAYERS - 1:
                last_layer_preds = {
                    "y_test": y_test,
                    "y_pred_upper": y_pred_upper,
                    "y_pred_lower": y_pred_lower,
                }

            del X_upper_train, X_upper_test, X_lower_train, X_lower_test

        except Exception as e:
            print(f"Layer {layer:2d}  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "layer": layer,
                "r2_upper": float("nan"),
                "r2_lower": float("nan"),
                "rmse_upper": float("nan"),
                "rmse_lower": float("nan"),
            })

    # ------------------------------------------------------------------
    # Phase 4: Save CSV
    # ------------------------------------------------------------------

    metrics_csv = csv_dir / "layer_metrics.csv"
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "r2_upper", "r2_lower", "rmse_upper", "rmse_lower"])
        for r in results:
            writer.writerow([
                r["layer"],
                f"{r['r2_upper']:.6f}", f"{r['r2_lower']:.6f}",
                f"{r['rmse_upper']:.4f}", f"{r['rmse_lower']:.4f}",
            ])
    print(f"\nSaved metrics CSV -> {metrics_csv}")

    # ------------------------------------------------------------------
    # Phase 5: R² line plot
    # ------------------------------------------------------------------

    layers = [r["layer"] for r in results]
    r2_uppers = [r["r2_upper"] for r in results]
    r2_lowers = [r["r2_lower"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(layers, r2_uppers, marker="o", markersize=3, linewidth=1.5, label="Upper (i→j)")
    ax.plot(layers, r2_lowers, marker="s", markersize=3, linewidth=1.5, label="Lower (j→i)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R²")
    ax.set_title(
        f"Sequential Distance (j-i) Regression — Upper vs Lower Triangle\n"
        f"({n_proteins} proteins, {n_total_pairs:,} pairs)"
    )
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    r2_plot_path = vis_dir / "layer_r2.png"
    fig.savefig(r2_plot_path, dpi=150)
    plt.close(fig)
    print(f"Saved R² plot -> {r2_plot_path}")

    # ------------------------------------------------------------------
    # Phase 6: Scatterplot — predicted vs actual at layer 47
    # ------------------------------------------------------------------

    if last_layer_preds is not None:
        y_t = last_layer_preds["y_test"]
        y_u = last_layer_preds["y_pred_upper"]
        y_l = last_layer_preds["y_pred_lower"]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        diag_min = min(y_t.min(), y_u.min(), y_l.min())
        diag_max = max(y_t.max(), y_u.max(), y_l.max())

        for ax, y_pred, label, color in [
            (ax1, y_u, "Upper (i→j)", "steelblue"),
            (ax2, y_l, "Lower (j→i)", "darkorange"),
        ]:
            ax.scatter(y_t, y_pred, alpha=0.05, s=2, color=color, rasterized=True)
            ax.plot([diag_min, diag_max], [diag_min, diag_max], "k--", linewidth=1, label="Perfect")
            rmse = float(np.sqrt(np.mean((y_pred - y_t) ** 2)))
            r2 = r2_score(y_t, y_pred)
            ax.set_xlabel("Actual j - i (residues)")
            ax.set_ylabel("Predicted j - i (residues)")
            ax.set_title(f"Layer 47 — {label}\nR²={r2:.4f}  RMSE={rmse:.1f} residues")
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.3)

        fig.suptitle("Predicted vs Actual Sequential Distance at Layer 47", fontsize=13)
        fig.tight_layout()
        scatter_path = vis_dir / "scatter_layer47.png"
        fig.savefig(scatter_path, dpi=150)
        plt.close(fig)
        print(f"Saved scatter plot -> {scatter_path}")

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Feature vector: {FEATURE_DIM}-dim (128 cross-pair + 256 diagonal)")
    print(f"Target std (baseline RMSE if predicting mean): {y_all.std():.1f} residues")
    print(f"Proteins ({n_proteins}):")
    for d in protein_data:
        n_pairs = len(d["labels"])
        print(f"  {d['protein_id']}: {d['n_residues']} residues, {n_pairs:,} pairs")
    if skipped:
        print(f"\nSkipped ({len(skipped)}):")
        for pid, reason in skipped:
            print(f"  {pid}: {reason}")

    if any(not np.isnan(r["r2_upper"]) for r in results):
        best_upper_layer = int(np.nanargmax([r["r2_upper"] for r in results]))
        best_lower_layer = int(np.nanargmax([r["r2_lower"] for r in results]))
        print(f"\nBest upper R²: layer {best_upper_layer} = {results[best_upper_layer]['r2_upper']:.4f}"
              f"  RMSE={results[best_upper_layer]['rmse_upper']:.2f}")
        print(f"Best lower R²: layer {best_lower_layer} = {results[best_lower_layer]['r2_lower']:.4f}"
              f"  RMSE={results[best_lower_layer]['rmse_lower']:.2f}")
        print(f"Layer 47: r2_upper={results[-1]['r2_upper']:.4f}  rmse_upper={results[-1]['rmse_upper']:.2f}"
              f"  r2_lower={results[-1]['r2_lower']:.4f}  rmse_lower={results[-1]['rmse_lower']:.2f}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
multi_protein_regression.py
============================
Local regression script for the project repository. Runs on your machine
against the standard project directory layout:
    proteins/<id>/predicted_pdb/  — AlphaFold predicted PDB files
    proteins/<id>/pair_blocks/    — pair representation .npy files

What it does:
  1. Extracts CA coordinates from AlphaFold predicted PDB files
  2. Computes pairwise CA-CA distance matrices
  3. Trains a pooled linear regression (pair-block features -> CA-CA distance)
     independently for each of 48 AlphaFold layers

For the ICE cluster variant, see cluster_multi_protein_regression.py.

Dependencies (all pip-installable):
    pip install numpy scipy matplotlib scikit-learn biopython

Outputs:
    Per-protein (inside each protein directory):
        csv_files/predicted_residue_coordinates.csv
        csv_files/predicted_residue_distances.csv

    Multi-protein summary (inside proteins/multi_protein_results/):
        csv_files/layer_regression_r2_predicted.csv
        csv_files/layer_regression_coefficients_predicted.csv
        csv_files/predicted_distances_last_layer_predicted.csv
        visualizations/layer_regression_r2_predicted.png
        visualizations/pred_vs_actual_all_predicted.png
        visualizations/pred_vs_actual_per_protein_predicted.png
"""

import csv
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


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

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROTEINS_DIR = PROJECT_ROOT / "proteins"
OUTPUT_DIR   = PROTEINS_DIR / "multi_protein_results"

sys.path.insert(0, str(PROJECT_ROOT))
from lib.output_writers import ensure_output_dirs, save_coordinates_csv, save_distance_csv
from lib.protein_dataset import load_local_protein_data
from lib.regression_common import (
    build_layer_feature_matrix,
    concatenate_targets,
    compute_protein_offsets,
    split_indices,
)


# ===========================================================================
# Main
# ===========================================================================

def main():
    proteins_dir = PROTEINS_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    n_layers = N_LAYERS

    csv_dir, vis_dir = ensure_output_dirs(output_dir)

    print(f"Proteins directory: {proteins_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Layers:             {n_layers}")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------
    protein_data, skipped = load_local_protein_data(proteins_dir, include_distances=True)
    print(f"\nDiscovered {len(protein_data) + len(skipped)} protein directories.")
    for d in protein_data:
        print(f"[OK]   {d['protein_id']}: {d['n_residues']} residues, {len(d['y']):,} pairs")
        prot_csv_dir = d["protein_dir"] / "csv_files"
        prot_csv_dir.mkdir(parents=True, exist_ok=True)
        residue_data = {"residues": d["residues"], "coordinates": d["coordinates"]}
        save_coordinates_csv(prot_csv_dir / "predicted_residue_coordinates.csv", residue_data)
        save_distance_csv(
            prot_csv_dir / "predicted_residue_distances.csv",
            d["distance_matrix"],
            residue_data,
        )
    for protein_id, reason in skipped:
        print(f"[SKIP] {protein_id}: {reason}")

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    n_proteins = len(protein_data)

    # ------------------------------------------------------------------
    # Phase 2: Pool targets and split
    # ------------------------------------------------------------------
    y_all = concatenate_targets(protein_data)
    n_total_pairs = len(y_all)

    protein_offsets = compute_protein_offsets(protein_data)

    print(f"\nTotal pairs across {n_proteins} protein(s): {n_total_pairs:,}")

    idx_train, idx_test = split_indices(
        n_samples=n_total_pairs,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    y_train = y_all[idx_train]
    y_test = y_all[idx_test]

    split_labels = np.empty(n_total_pairs, dtype=object)
    split_labels[idx_train] = "train"
    split_labels[idx_test] = "test"

    # ------------------------------------------------------------------
    # Phase 3: Layer-wise regression
    # ------------------------------------------------------------------
    results = []
    all_coefs = []
    last_layer_preds = None

    print()
    for layer in range(n_layers):
        try:
            X_all = build_layer_feature_matrix(protein_data, layer, mode="base")

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_all[idx_train])
            X_test = scaler.transform(X_all[idx_test])

            model = LinearRegression()
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)
            r2 = r2_score(y_test, y_pred_test)
            results.append(r2)
            print(f"Layer {layer:2d}  R2 = {r2:.4f}")

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
            results.append(float("nan"))
            all_coefs.append({"layer": layer, "intercept": 0.0})

    # ------------------------------------------------------------------
    # Phase 4: Save CSV outputs
    # ------------------------------------------------------------------

    # CSV 1: Layer R2
    r2_csv = csv_dir / "layer_regression_r2_predicted.csv"
    with open(r2_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "r2", "n_proteins", "n_total_pairs"])
        for layer, r2 in enumerate(results):
            writer.writerow([layer, f"{r2:.6f}", n_proteins, n_total_pairs])
    print(f"\nSaved R2 CSV -> {r2_csv}")

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

    # ------------------------------------------------------------------
    # Visualizations
    # ------------------------------------------------------------------

    # Plot 1: R² vs layer
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_layers), results, marker="o", markersize=3, linewidth=1.5)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test R²")
    ax.set_title(f"Layer-wise Regression R² ({n_proteins} proteins, {n_total_pairs:,} pairs)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "layer_regression_r2_predicted.png", dpi=150)
    plt.close(fig)
    print(f"Saved R² plot -> {vis_dir / 'layer_regression_r2_predicted.png'}")

    # Plot 2: Predicted vs actual — all proteins combined (last layer)
    if last_layer_preds is not None:
        COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig, ax = plt.subplots(figsize=(8, 7))
        for idx_p, (pid, start, end) in enumerate(protein_offsets):
            y_act = y_all[start:end]
            y_pred = last_layer_preds[start:end]
            ax.scatter(
                y_act, y_pred,
                s=2, alpha=0.3,
                color=COLORS[idx_p % len(COLORS)],
                label=f"{pid} ({end - start:,} pairs)",
            )

        lim_max = y_all.max() * 1.05
        ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, label="y = x")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_xlabel("Actual CA-CA Distance (Å)")
        ax.set_ylabel("Predicted CA-CA Distance (Å)")
        ax.set_title(f"Layer {n_layers - 1}: Predicted vs Actual (all proteins)")
        ax.legend(markerscale=4, fontsize=9)
        fig.tight_layout()
        fig.savefig(vis_dir / "pred_vs_actual_all_predicted.png", dpi=150)
        plt.close(fig)
        print(f"Saved combined plot -> {vis_dir / 'pred_vs_actual_all_predicted.png'}")

        # Plot 3: Per-protein grid (last layer)
        n_cols = 2
        n_rows = math.ceil(n_proteins / n_cols)
        fig, axes = plt.subplots(
            n_rows, n_cols,
            figsize=(7 * n_cols, 6 * n_rows),
            squeeze=False,
        )

        for idx_p, (pid, start, end) in enumerate(protein_offsets):
            row, col = divmod(idx_p, n_cols)
            ax = axes[row][col]
            y_act = y_all[start:end]
            y_pred = last_layer_preds[start:end]
            r2_p = r2_score(y_act, y_pred)
            ax.scatter(
                y_act, y_pred,
                s=2, alpha=0.4,
                color=COLORS[idx_p % len(COLORS)],
            )
            lim_max_p = max(y_act.max(), y_pred.max()) * 1.05
            ax.plot([0, lim_max_p], [0, lim_max_p], "k--", linewidth=1)
            ax.set_xlim(0, lim_max_p)
            ax.set_ylim(0, lim_max_p)
            ax.set_xlabel("Actual CA-CA Distance (Å)")
            ax.set_ylabel("Predicted CA-CA Distance (Å)")
            ax.set_title(f"{pid}  (R² = {r2_p:.4f})")
            ax.grid(True, alpha=0.3)

        # Hide unused panels
        for idx_p in range(n_proteins, n_rows * n_cols):
            row, col = divmod(idx_p, n_cols)
            axes[row][col].set_visible(False)

        fig.suptitle(
            f"Layer {n_layers - 1}: Predicted vs Actual Distance (per protein)",
            fontsize=14,
        )
        fig.tight_layout()
        fig.savefig(vis_dir / "pred_vs_actual_per_protein_predicted.png", dpi=150)
        plt.close(fig)
        print(f"Saved per-protein plot -> {vis_dir / 'pred_vs_actual_per_protein_predicted.png'}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Included proteins ({n_proteins}):")
    for pid, start, end in protein_offsets:
        n_res = next(d["n_residues"] for d in protein_data if d["protein_id"] == pid)
        print(f"  {pid}: {n_res} residues, {end - start:,} pairs")
    if skipped:
        print(f"\nSkipped proteins ({len(skipped)}):")
        for pid, reason in skipped:
            print(f"  {pid}: {reason}")
    valid_results = [r for r in results if not math.isnan(r)]
    if valid_results:
        best_layer = int(np.nanargmax(results))
        print(f"\nBest layer: {best_layer}  R² = {results[best_layer]:.4f}")
        print(f"Layer {n_layers - 1} R² = {results[-1]:.4f}")


if __name__ == "__main__":
    main()

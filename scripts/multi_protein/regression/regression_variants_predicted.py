#!/usr/bin/env python3
"""
regression_variants_predicted.py
================================
Compares Linear, Ridge, and Lasso regression as probes for predicting
CA-CA distance from AlphaFold pair-block embeddings across all 48 layers.

Uses predicted PDB files (not CIF structures) for ground-truth distances.
Standalone, cluster-ready — no lib/ imports.

For each of the 48 layers:
  - Feature: concatenated pair-block vectors [pair_block[i,j,:], pair_block[j,i,:]] → 256-dim
  - Target: CA-CA distance in Angstroms for that residue pair
  - Split: fixed 80/20 train/test across the pooled dataset (random_state=42)
  - Standardization: StandardScaler fit on training rows only
  - Models: LinearRegression, Ridge(alpha=1.0), Lasso(alpha=1.0)

Outputs (inside output_dir/regression_variants/):
    csv_files/layer_r2_variants.csv
    visualizations/layer_r2_comparison.png
    visualizations/pred_vs_actual_comparison_layer47.png

Usage:
    python regression_variants_predicted.py
    python regression_variants_predicted.py --proteins_dir /data/proteins --output_dir /data/results
"""

import argparse
import csv
import math
import os
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for cluster
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

# BioPython — needed for PDB parsing
from Bio.PDB import PDBParser


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48          # AlphaFold pair representation layers
N_CHANNELS = 128       # channels per pair representation direction
FEATURE_DIM = 2 * N_CHANNELS  # 256 (concatenated [i,j] and [j,i])
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Model definitions: name → constructor
MODEL_DEFS = {
    "linear": lambda: LinearRegression(),
    "ridge":  lambda: Ridge(alpha=1.0),
    "lasso":  lambda: Lasso(alpha=1.0, max_iter=10000),
}
MODEL_NAMES = list(MODEL_DEFS.keys())
MODEL_LABELS = {
    "linear": "Linear Regression",
    "ridge":  "Ridge (α=1.0)",
    "lasso":  "Lasso (α=1.0)",
}
MODEL_COLORS = {
    "linear": "steelblue",
    "ridge":  "darkorange",
    "lasso":  "forestgreen",
}


# ===========================================================================
# Helper functions — data loading
# ===========================================================================

def find_pdb_file(protein_dir, protein_id):
    """Locate the predicted PDB file for a protein.

    Expected path: <protein_dir>/predicted_pdb/<protein_id>_model_1_ptm_unrelaxed.pdb
    """
    pdb_path = protein_dir / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    return pdb_path if pdb_path.exists() else None


def get_pair_block_path(protein_dir, protein_id, layer):
    """Return the path to a pair block .npy file for a given layer."""
    return protein_dir / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy"


def discover_proteins(proteins_dir):
    """Find all protein subdirectories under the given root.

    Returns a sorted list of (protein_id, protein_dir) tuples.
    """
    proteins = []
    for entry in sorted(proteins_dir.iterdir()):
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
            if het_flag == "W":  # water
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
# Main script
# ===========================================================================

def parse_args():
    script_dir = Path(__file__).parent
    default_proteins = script_dir.parent.parent.parent / "proteins"
    default_output = default_proteins / "multi_protein_results" / "regression_variants"

    parser = argparse.ArgumentParser(
        description="Compare Linear, Ridge, and Lasso regression probes across 48 layers."
    )
    parser.add_argument(
        "--proteins_dir", type=Path, default=default_proteins,
        help="Root directory containing protein subdirectories (default: %(default)s)",
    )
    parser.add_argument(
        "--output_dir", type=Path, default=default_output,
        help="Directory for output CSVs and plots (default: %(default)s)",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    proteins_dir = args.proteins_dir.resolve()
    output_dir = args.output_dir.resolve()

    csv_dir = output_dir / "csv_files"
    vis_dir = output_dir / "visualizations"
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Proteins directory: {proteins_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Layers:             {N_LAYERS}")
    print(f"Models:             {', '.join(MODEL_LABELS.values())}")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------
    all_proteins = discover_proteins(proteins_dir)
    print(f"\nDiscovered {len(all_proteins)} protein directories.")

    protein_data = []   # successfully loaded proteins
    skipped = []        # (protein_id, reason)

    for protein_id, protein_dir in all_proteins:
        try:
            # --- Find PDB file ---
            pdb_path = find_pdb_file(protein_dir, protein_id)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                print(f"[SKIP] {protein_id}: no predicted PDB file found")
                continue

            # --- Check pair blocks exist ---
            pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
            if not pb0_path.exists():
                skipped.append((protein_id, "pair block 0 not found"))
                print(f"[SKIP] {protein_id}: pair block 0 not found at {pb0_path}")
                continue

            # --- Parse chain ID from protein_id (e.g. "7b3a_A" → "A") ---
            # --- Extract CA coordinates ---
            ca_data = extract_ca_coordinates(pdb_path)
            n_ca = len(ca_data["residues"])
            if n_ca == 0:
                skipped.append((protein_id, "no CA atoms found in PDB"))
                print(f"[SKIP] {protein_id}: no CA atoms found in PDB")
                continue

            # --- Verify dimension match with pair blocks ---
            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = pb0.shape[0]
            del pb0

            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                print(f"[SKIP] {protein_id}: CA count {n_ca} != pair block dim {pb_dim}")
                continue

            # --- Compute distance matrix ---
            dist_matrix = compute_distance_matrix(ca_data["coordinates"])

            # --- Build upper-triangle pair indices ---
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
            skipped.append((protein_id, f"unexpected error — {e}"))
            print(f"[SKIP] {protein_id}: unexpected error — {e}")
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

    # Fixed 80/20 train/test split
    all_idx = np.arange(n_total_pairs)
    idx_train, idx_test = train_test_split(all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)
    y_train = y_all[idx_train]
    y_test = y_all[idx_test]

    # ------------------------------------------------------------------
    # Phase 3: Layer-wise regression with all model variants
    # ------------------------------------------------------------------
    results = {name: [] for name in MODEL_NAMES}
    last_layer_preds = {}  # model_name → predictions array (filled at layer 47)

    print()
    for layer in range(N_LAYERS):
        try:
            # Build features for this layer across all proteins
            X_chunks = []
            for d in protein_data:
                pb_path = get_pair_block_path(d["protein_dir"], d["protein_id"], layer)
                pb = np.load(str(pb_path))
                X_chunk = np.concatenate([
                    pb[d["pb_i"], d["pb_j"], :],   # (n_pairs, 128)
                    pb[d["pb_j"], d["pb_i"], :],   # (n_pairs, 128)
                ], axis=1)                          # (n_pairs, 256)
                X_chunks.append(X_chunk)
                del pb

            X_all = np.vstack(X_chunks)
            del X_chunks

            # Standardize (fit on train only)
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_all[idx_train])
            X_test = scaler.transform(X_all[idx_test])

            # Train all models on the SAME standardized data
            r2_strs = []
            for name in MODEL_NAMES:
                model = MODEL_DEFS[name]()
                model.fit(X_train, y_train)
                r2 = r2_score(y_test, model.predict(X_test))
                results[name].append(r2)
                r2_strs.append(f"{name}: {r2:.4f}")

                # Last layer: predict on ALL pairs for plots
                if layer == N_LAYERS - 1:
                    X_all_std = scaler.transform(X_all)
                    last_layer_preds[name] = model.predict(X_all_std)

            print(f"Layer {layer:2d}  {'  '.join(r2_strs)}")

            del X_all, X_train, X_test

        except Exception as e:
            print(f"Layer {layer:2d}  ERROR: {e}")
            traceback.print_exc()
            for name in MODEL_NAMES:
                results[name].append(float("nan"))

    # ------------------------------------------------------------------
    # Phase 4: Save CSV — layer R² for all variants
    # ------------------------------------------------------------------
    r2_csv = csv_dir / "layer_r2_variants.csv"
    with open(r2_csv, "w", newline="") as f:
        writer = csv.writer(f)
        header = ["layer"] + [f"{name}_r2" for name in MODEL_NAMES] + ["n_proteins", "n_total_pairs"]
        writer.writerow(header)
        for layer in range(N_LAYERS):
            row = [layer]
            for name in MODEL_NAMES:
                row.append(f"{results[name][layer]:.6f}")
            row.extend([n_proteins, n_total_pairs])
            writer.writerow(row)
    print(f"\nSaved R² CSV → {r2_csv}")

    # ------------------------------------------------------------------
    # Visualization 1: R² vs layer — all models overlaid
    # ------------------------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    for name in MODEL_NAMES:
        ax.plot(range(N_LAYERS), results[name],
                marker="o", markersize=3, linewidth=1.5,
                color=MODEL_COLORS[name], label=MODEL_LABELS[name])
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test R²")
    ax.set_title(f"Regression Variants: R² vs Layer ({n_proteins} proteins, {n_total_pairs:,} pairs)")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "layer_r2_comparison.png", dpi=150)
    plt.close(fig)
    print(f"Saved R² comparison plot → {vis_dir / 'layer_r2_comparison.png'}")

    # ------------------------------------------------------------------
    # Visualization 2: Predicted vs actual — 1x3 grid (last layer)
    # ------------------------------------------------------------------
    if last_layer_preds:
        n_models = len(MODEL_NAMES)
        fig, axes = plt.subplots(1, n_models, figsize=(7 * n_models, 6), squeeze=False)

        for idx, name in enumerate(MODEL_NAMES):
            ax = axes[0][idx]
            y_pred = last_layer_preds[name]

            # Plot test-set points only
            ax.scatter(y_test, y_pred[idx_test],
                       s=2, alpha=0.3, color=MODEL_COLORS[name], rasterized=True)

            lim_max = y_all.max() * 1.05
            ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1)
            ax.set_xlim(0, lim_max)
            ax.set_ylim(0, lim_max)
            ax.set_xlabel("Actual CA-CA Distance (Å)")
            ax.set_ylabel("Predicted CA-CA Distance (Å)")

            r2_test = r2_score(y_test, y_pred[idx_test])
            ax.set_title(f"{MODEL_LABELS[name]}\nR² = {r2_test:.4f}")
            ax.set_aspect("equal")
            ax.grid(True, alpha=0.3)

        fig.suptitle("Layer 47: Predicted vs Actual Distance — Model Comparison", fontsize=14)
        fig.tight_layout()
        fig.savefig(vis_dir / "pred_vs_actual_comparison_layer47.png", dpi=150)
        plt.close(fig)
        print(f"Saved pred vs actual comparison → {vis_dir / 'pred_vs_actual_comparison_layer47.png'}")

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

    print(f"\nBest layer per model:")
    for name in MODEL_NAMES:
        valid = [r for r in results[name] if not math.isnan(r)]
        if valid:
            best_layer = int(np.nanargmax(results[name]))
            best_r2 = results[name][best_layer]
            print(f"  {MODEL_LABELS[name]:30s}  layer {best_layer}  R² = {best_r2:.4f}")

    print(f"\nLayer 47 R² per model:")
    for name in MODEL_NAMES:
        r2_47 = results[name][-1]
        print(f"  {MODEL_LABELS[name]:30s}  R² = {r2_47:.4f}")


if __name__ == "__main__":
    main()

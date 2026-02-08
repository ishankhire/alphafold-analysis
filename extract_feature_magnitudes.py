#!/usr/bin/env python3
"""
Extract absolute feature magnitudes for pairwise representations.

For all layers, computes the L2 norm of the feature vector for each
residue pair (i, j) where i != j (excluding diagonal only).
Generates visualizations for layers 10, 20, 30, and 40.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Settings
base_dir = os.path.dirname(os.path.abspath(__file__))
protein = "7b3a_A"
all_layers = list(range(48))  # All layers 0-47
visualize_layers = [10, 20, 30, 40]  # Only generate heatmaps for these

# Create output directory
output_dir = "feature_magnitudes"
os.makedirs(output_dir, exist_ok=True)

for layer_idx in all_layers:
    filepath = os.path.join(base_dir, protein, f"{protein}_pair_block_{layer_idx}.npy")

    if not os.path.exists(filepath):
        print(f"Skipping layer {layer_idx}: file not found")
        continue

    print(f"\nProcessing layer {layer_idx}...")
    data = np.load(filepath)

    r, r2, C = data.shape
    assert r == r2, "Expected square spatial dimensions"

    if layer_idx == all_layers[0]:
        print(f"Shape: ({r}, {r}, {C}) - {r} residues, {C} channels")

    # Compute magnitude matrix for all pairs (excluding diagonal)
    # Use vectorized computation for speed
    mag_matrix = np.linalg.norm(data, axis=2)  # (r, r) matrix of L2 norms

    # Set diagonal to NaN (exclude i == j)
    np.fill_diagonal(mag_matrix, np.nan)

    # Extract all off-diagonal magnitudes for statistics
    mask = ~np.isnan(mag_matrix)
    magnitudes = mag_matrix[mask]
    num_pairs = len(magnitudes)

    print(f"  Extracted {num_pairs} pairs (all i != j)")
    print(f"  Expected: r*(r-1) = {r * (r - 1)}")

    # Statistics
    print(f"  Min: {magnitudes.min():.4f}, Max: {magnitudes.max():.4f}, "
          f"Mean: {magnitudes.mean():.4f}, Std: {magnitudes.std():.4f}")

    # Save matrix
    output_matrix = os.path.join(output_dir, f"layer{layer_idx}_magnitudes.npy")
    np.save(output_matrix, mag_matrix)

    # Save CSV
    output_csv = os.path.join(output_dir, f"layer{layer_idx}_magnitudes.csv")
    with open(output_csv, "w") as f:
        f.write("residue_i,residue_j,magnitude\n")
        for i in range(r):
            for j in range(r):
                if i != j:
                    f.write(f"{i},{j},{mag_matrix[i, j]:.6f}\n")

    # Generate visualization only for specified layers
    if layer_idx in visualize_layers:
        fig, ax = plt.subplots(figsize=(10, 8))

        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='white')  # Diagonal shown as white

        im = ax.imshow(mag_matrix, cmap=cmap, aspect='equal', origin='upper')

        cbar = plt.colorbar(im, ax=ax, shrink=0.8)
        cbar.set_label('Feature Magnitude (L2 norm)', fontsize=11)

        ax.set_xlabel('Residue j', fontsize=11)
        ax.set_ylabel('Residue i', fontsize=11)
        ax.set_title(f'Pairwise Feature Magnitudes - Layer {layer_idx}\n(All pairs, excluding diagonal)', fontsize=12)

        tick_interval = max(1, r // 10)
        tick_positions = list(range(0, r, tick_interval))
        if r - 1 not in tick_positions:
            tick_positions.append(r - 1)
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)

        plt.tight_layout()

        vis_dir = os.path.join("visualizations", "feature_magnitudes")
        os.makedirs(vis_dir, exist_ok=True)
        output_fig = os.path.join(vis_dir, f"layer{layer_idx}_heatmap.png")
        plt.savefig(output_fig, dpi=150)
        plt.close()

        print(f"  Saved heatmap: {output_fig}")

print(f"\nAll outputs saved to {output_dir}/")

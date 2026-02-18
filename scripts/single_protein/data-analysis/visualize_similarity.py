# -*- coding: utf-8 -*-
"""
visualize_similarity.py

Visualize the subspace similarity matrix between layers.
This shows which layers have similar PCA subspaces.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from pca_subspace_main import analyze

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTEIN_DIR = os.path.join(ROOT_DIR, "proteins", "7b3a")
VIS_DIR = os.path.join(PROTEIN_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

# Settings
base_dir = os.path.join(PROTEIN_DIR, "pair_blocks")
protein = "7b3a_A"
layers = 48  # All layers 0-47
k = 32
metric = "affinity"  # Options: affinity, maxcorr, mincorr, chordal, projection, procrustes, martin, grassmann

# Run analysis to get similarity matrix
print(f"Computing subspace similarity (metric: {metric})...")
layer_names, sim_mat, evr_mat, _ = analyze(
    base_dir=base_dir,
    protein=protein,
    layers=layers,
    k=k,
    spatial_mode="flatten",
    subsample=None,
    seed=0,
    metric=metric,
    normalize_similarity=True
)

# Get actual layer indices for labeling
layer_indices = list(range(48))  # All layers 0-47
layer_labels = [f"L{i}" for i in layer_indices]  # Shorter labels for 48 layers

print(f"\nSimilarity matrix shape: {sim_mat.shape}")
print(f"Similarity matrix:\n{sim_mat}")

# Create heatmap - larger figure for 48 layers
fig, ax = plt.subplots(figsize=(14, 12))

im = ax.imshow(sim_mat, cmap='RdYlBu_r', aspect='equal', vmin=0, vmax=1)

# Add colorbar
cbar = plt.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label(f'Subspace Similarity ({metric})', fontsize=11)

# Set ticks and labels - show every 5th layer for readability
tick_positions = list(range(0, len(layer_labels), 5)) + [len(layer_labels) - 1]
tick_labels = [layer_labels[i] for i in tick_positions]
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.set_xticklabels(tick_labels, rotation=45, ha='right', fontsize=9)
ax.set_yticklabels(tick_labels, fontsize=9)

# Skip text in cells for 48x48 matrix (too dense to read)

ax.set_title(f'PCA Subspace Similarity Between All {layers} Layers\n(k={k}, metric={metric})', fontsize=12)
ax.set_xlabel('Layer')
ax.set_ylabel('Layer')

plt.tight_layout()
plt.savefig(os.path.join(VIS_DIR, 'subspace_similarity_heatmap.png'), dpi=150)
plt.show()

print(f"\nSaved: {os.path.join(VIS_DIR, 'subspace_similarity_heatmap.png')}")

# Also print explained variance summary
print(f"\nExplained variance ratio (cumulative for top {k} PCs):")
for i, (label, evr) in enumerate(zip(layer_labels, evr_mat)):
    cumulative = evr.sum()
    print(f"  {label}: {cumulative:.2%} (top 5: {evr[:5].round(3)})")

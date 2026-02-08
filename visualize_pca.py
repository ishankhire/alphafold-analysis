# -*- coding: utf-8 -*-
"""
visualize_pca.py

Visualize top-k PCA components as heatmaps for all layers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

# Settings
input_dir = "pca_projections"
output_dir = "visualizations/pc_maps"
protein = "7b3a_A"
layer_indices = list(range(48))  # All layers 0-47
num_pcs = 4  # Top 4 principal components

# Create output directory
os.makedirs(output_dir, exist_ok=True)

for layer_idx in layer_indices:
    filepath = os.path.join(input_dir, f"{protein}_pair_block_{layer_idx}_pca32.npy")
    
    if not os.path.exists(filepath):
        print(f"Skipping layer {layer_idx}: file not found")
        continue
    
    # Load the PCA projection
    data = np.load(filepath)
    
    # Verify shape
    print(f"\nLayer {layer_idx}: {filepath}")
    print(f"  Shape: {data.shape}")
    r1, r2, k = data.shape
    
    # Create heatmaps for top 4 PCs
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i in range(num_pcs):
        pc = data[:, :, i].copy()  # (r x r)
        
        # Mask the diagonal (set to NaN so it appears black)
        np.fill_diagonal(pc, np.nan)
        
        # Set up colormap with black for NaN values
        cmap = plt.cm.viridis.copy()
        cmap.set_bad(color='black')
        
        im = axes[i].imshow(pc, cmap=cmap, aspect='equal')
        axes[i].set_title(f'PC {i+1}')
        axes[i].set_xlabel('Residue j')
        axes[i].set_ylabel('Residue i')
        plt.colorbar(im, ax=axes[i], shrink=0.8)
    
    plt.suptitle(f'Layer {layer_idx} - Top 4 PCA Components', fontsize=14)
    plt.tight_layout()
    
    output_path = os.path.join(output_dir, f"layer_{layer_idx}_pca_heatmaps.png")
    plt.savefig(output_path, dpi=150)
    plt.close()
    
    print(f"  Saved: {output_path}")

print(f"\nAll heatmaps saved to {output_dir}/")

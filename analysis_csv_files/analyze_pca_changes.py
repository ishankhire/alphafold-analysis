#!/usr/bin/env python3
"""
Analyze how principal component activations change between successive layers.

For each layer transition, computes the change in PC magnitude:
    ||PC_L+1[i,j]| - |PC_L[i,j]||

This measures how much the absolute activation value changes, regardless of sign.
Compares short-range (≤8 Å) vs long-range (>8 Å) residue pairs.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

try:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    print("Biopython is required. Install with: pip install biopython")
    exit(1)

# Settings
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
protein = "7b3a_A"
cif_file = os.path.join(base_dir, "7b3a.cif")
pca_dir = os.path.join(base_dir, "pca_projections")
num_layers = 48
num_pcs = 4  # Top 4 principal components
spatial_threshold = 8.0  # Angstroms


# =============================================================================
# Step 1: Load structure and build spatial distance masks
# =============================================================================
print("=" * 60)
print("STEP 1: Building spatial distance matrix and pair masks")
print("=" * 60)

# Parse structure to get CA coordinates
print(f"Parsing {cif_file}...")
parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("protein", cif_file)

ca_coords = []
for model in structure:
    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            if "CA" in residue:
                ca_coords.append(residue["CA"].get_coord())

ca_coords = np.array(ca_coords)
n_residues = len(ca_coords)
print(f"Found {n_residues} residues with CA atoms")

# Compute pairwise CA-CA distance matrix
print("Computing pairwise CA-CA distances...")
dist_matrix = np.zeros((n_residues, n_residues))
for i in range(n_residues):
    for j in range(n_residues):
        dist_matrix[i, j] = np.linalg.norm(ca_coords[i] - ca_coords[j])

# Create masks for upper triangle only (i < j) to avoid double counting
valid_pairs_mask = np.triu(np.ones((n_residues, n_residues), dtype=bool), k=1)
n_valid_pairs = np.sum(valid_pairs_mask)
print(f"Total valid pairs (i < j): {n_valid_pairs}")

# Create near and far masks
near_mask = valid_pairs_mask & (dist_matrix <= spatial_threshold)
far_mask = valid_pairs_mask & (dist_matrix > spatial_threshold)

n_near = np.sum(near_mask)
n_far = np.sum(far_mask)
print(f"Near pairs (≤ {spatial_threshold} Å): {n_near}")
print(f"Far pairs (> {spatial_threshold} Å): {n_far}")

assert n_near + n_far == n_valid_pairs, "ERROR: near + far != total valid pairs"
print("✓ Sanity check passed: near_count + far_count = total valid pairs")


# =============================================================================
# Step 2: Compute PC magnitude changes between layers
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2: Computing PC magnitude changes")
print("=" * 60)
print(f"Formula: ||PC_L+1[i,j]| - |PC_L[i,j]||")
print(f"Analyzing top {num_pcs} principal components...")

# Storage for results
pca_results = {
    'layer_index': [],
}
for pc in range(1, num_pcs + 1):
    pca_results[f'pc{pc}_near_mean'] = []
    pca_results[f'pc{pc}_far_mean'] = []
    pca_results[f'pc{pc}_diff'] = []  # near - far
    # Percentage change storage
    pca_results[f'pc{pc}_near_pct'] = []
    pca_results[f'pc{pc}_far_pct'] = []

# For percentage change, we need a reasonable minimum threshold
# Values near zero will give artificially high percentages
# We'll compute threshold dynamically based on the data
pct_min_threshold = None  # Will be set based on first layer's data

for layer_idx in range(1, num_layers):
    layer_prev = layer_idx - 1
    layer_curr = layer_idx

    filepath_prev = os.path.join(pca_dir, f"{protein}_pair_block_{layer_prev}_pca32.npy")
    filepath_curr = os.path.join(pca_dir, f"{protein}_pair_block_{layer_curr}_pca32.npy")

    if not os.path.exists(filepath_prev) or not os.path.exists(filepath_curr):
        print(f"Skipping layer {layer_prev} -> {layer_curr}: PCA files not found")
        continue

    # Load PCA projections: shape (r, r, 32)
    pca_prev = np.load(filepath_prev).astype(np.float64)
    pca_curr = np.load(filepath_curr).astype(np.float64)

    r_prev = pca_prev.shape[0]
    r_curr = pca_curr.shape[0]
    r = min(r_prev, r_curr, n_residues)

    # Adjust spatial masks
    if r < n_residues:
        local_near_mask = near_mask[:r, :r]
        local_far_mask = far_mask[:r, :r]
    else:
        local_near_mask = near_mask
        local_far_mask = far_mask

    pca_results['layer_index'].append(layer_idx)

    # Analyze each of the top 4 PCs
    for pc_idx in range(num_pcs):
        pc_num = pc_idx + 1

        # Get this PC's values for both layers
        pc_prev = pca_prev[:r, :r, pc_idx]
        pc_curr = pca_curr[:r, :r, pc_idx]

        # Compute ||PC_L+1| - |PC_L|| (absolute difference of absolute values)
        abs_prev = np.abs(pc_prev)
        abs_curr = np.abs(pc_curr)
        magnitude_change = np.abs(abs_curr - abs_prev)

        # Set percentage threshold based on data scale (use 5th percentile of non-zero values)
        # This avoids dividing by near-zero values which gives artificially high percentages
        if pct_min_threshold is None:
            all_abs_vals = abs_prev[abs_prev > 0].flatten()
            if len(all_abs_vals) > 0:
                pct_min_threshold = np.percentile(all_abs_vals, 5)
                print(f"  Setting percentage threshold to {pct_min_threshold:.4f} (5th percentile)")
            else:
                pct_min_threshold = 0.1  # Fallback

        # Get values for near and far pairs (upper triangle only, i < j)
        near_changes = []
        far_changes = []
        near_pct = []
        far_pct = []

        for i in range(r):
            for j in range(i + 1, r):
                if local_near_mask[i, j]:
                    near_changes.append(magnitude_change[i, j])
                    # Only compute percentage if baseline is above threshold
                    if abs_prev[i, j] > pct_min_threshold:
                        pct = (magnitude_change[i, j] / abs_prev[i, j]) * 100
                        near_pct.append(pct)
                elif local_far_mask[i, j]:
                    far_changes.append(magnitude_change[i, j])
                    if abs_prev[i, j] > pct_min_threshold:
                        pct = (magnitude_change[i, j] / abs_prev[i, j]) * 100
                        far_pct.append(pct)

        near_mean = np.mean(near_changes) if near_changes else np.nan
        far_mean = np.mean(far_changes) if far_changes else np.nan
        diff = near_mean - far_mean if not (np.isnan(near_mean) or np.isnan(far_mean)) else np.nan

        # Use median for percentage to be robust to outliers
        near_pct_median = np.median(near_pct) if near_pct else np.nan
        far_pct_median = np.median(far_pct) if far_pct else np.nan

        pca_results[f'pc{pc_num}_near_mean'].append(near_mean)
        pca_results[f'pc{pc_num}_far_mean'].append(far_mean)
        pca_results[f'pc{pc_num}_diff'].append(diff)
        pca_results[f'pc{pc_num}_near_pct'].append(near_pct_median)
        pca_results[f'pc{pc_num}_far_pct'].append(far_pct_median)

    print(f"Layer {layer_prev:2d} -> {layer_curr:2d}: "
          f"PC1 diff={pca_results['pc1_diff'][-1]:+.4f}, "
          f"PC2 diff={pca_results['pc2_diff'][-1]:+.4f}, "
          f"PC3 diff={pca_results['pc3_diff'][-1]:+.4f}, "
          f"PC4 diff={pca_results['pc4_diff'][-1]:+.4f}")


# =============================================================================
# Step 3: Save results and create plots
# =============================================================================
print("\n" + "=" * 60)
print("STEP 3: Generating outputs")
print("=" * 60)

# Convert to arrays
pca_layer_indices = np.array(pca_results['layer_index'])

# Save CSV
pca_csv_path = "layer_pca_magnitude_change.csv"
with open(pca_csv_path, 'w') as f:
    header = "layer_index"
    for pc in range(1, num_pcs + 1):
        header += f",pc{pc}_near_mean,pc{pc}_far_mean,pc{pc}_diff"
    f.write(header + "\n")

    for i in range(len(pca_results['layer_index'])):
        row = f"{pca_results['layer_index'][i]}"
        for pc in range(1, num_pcs + 1):
            row += f",{pca_results[f'pc{pc}_near_mean'][i]:.6f}"
            row += f",{pca_results[f'pc{pc}_far_mean'][i]:.6f}"
            row += f",{pca_results[f'pc{pc}_diff'][i]:.6f}"
        f.write(row + "\n")
print(f"Saved: {pca_csv_path}")

# Plot 1: 2x2 grid showing near vs far for each PC
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for pc_idx in range(num_pcs):
    pc_num = pc_idx + 1
    ax = axes[pc_idx]

    near_vals = np.array(pca_results[f'pc{pc_num}_near_mean'])
    far_vals = np.array(pca_results[f'pc{pc_num}_far_mean'])

    ax.plot(pca_layer_indices, near_vals, 'o-', label=f'Near (≤ {spatial_threshold} Å)',
            color='tab:blue', markersize=4, linewidth=1.5)
    ax.plot(pca_layer_indices, far_vals, 's-', label=f'Far (> {spatial_threshold} Å)',
            color='tab:orange', markersize=4, linewidth=1.5)

    ax.set_xlabel('Layer Index', fontsize=11)
    ax.set_ylabel('Mean ||PC_L+1| - |PC_L||', fontsize=11)
    ax.set_title(f'PC{pc_num}: Magnitude Change Between Layers', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, num_layers, 5))

plt.suptitle('PC Magnitude Changes: Short-range vs Long-range Pairs', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('pca_magnitude_change_by_pc.png', dpi=150, bbox_inches='tight')
print("Saved: pca_magnitude_change_by_pc.png")

# Plot 2: Difference (near - far) for each PC
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for pc_idx in range(num_pcs):
    pc_num = pc_idx + 1
    ax = axes[pc_idx]

    diff_vals = np.array(pca_results[f'pc{pc_num}_diff'])
    colors = ['tab:blue' if d > 0 else 'tab:orange' for d in diff_vals]

    ax.bar(pca_layer_indices, diff_vals, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
    ax.axhline(y=0, color='black', linestyle='-', linewidth=1)

    ax.set_xlabel('Layer Index', fontsize=11)
    ax.set_ylabel('Near − Far', fontsize=11)
    ax.set_title(f'PC{pc_num}: Difference (Near minus Far)\n(Positive = Near changes MORE)', fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    ax.set_xticks(range(0, num_layers, 5))

plt.suptitle('PC Magnitude Change Difference: Which Pairs Update More?', fontsize=13, y=1.02)
plt.tight_layout()
plt.savefig('pca_magnitude_change_diff.png', dpi=150, bbox_inches='tight')
print("Saved: pca_magnitude_change_diff.png")

# Plot 3: All PCs on same axes for comparison
pc_colors = ['tab:blue', 'tab:green', 'tab:red', 'tab:purple']
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: Near values for all PCs
ax1 = axes[0]
for pc_idx in range(num_pcs):
    pc_num = pc_idx + 1
    near_vals = np.array(pca_results[f'pc{pc_num}_near_mean'])
    ax1.plot(pca_layer_indices, near_vals, 'o-', label=f'PC{pc_num}',
             color=pc_colors[pc_idx], markersize=4, linewidth=1.5)

ax1.set_xlabel('Layer Index', fontsize=11)
ax1.set_ylabel('Mean ||PC_L+1| - |PC_L||', fontsize=11)
ax1.set_title(f'Short-range Pairs (≤ {spatial_threshold} Å): All PCs', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, num_layers, 5))

# Right: Far values for all PCs
ax2 = axes[1]
for pc_idx in range(num_pcs):
    pc_num = pc_idx + 1
    far_vals = np.array(pca_results[f'pc{pc_num}_far_mean'])
    ax2.plot(pca_layer_indices, far_vals, 's-', label=f'PC{pc_num}',
             color=pc_colors[pc_idx], markersize=4, linewidth=1.5)

ax2.set_xlabel('Layer Index', fontsize=11)
ax2.set_ylabel('Mean ||PC_L+1| - |PC_L||', fontsize=11)
ax2.set_title(f'Long-range Pairs (> {spatial_threshold} Å): All PCs', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)
ax2.set_xticks(range(0, num_layers, 5))

plt.tight_layout()
plt.savefig('pca_magnitude_change_all_pcs.png', dpi=150)
print("Saved: pca_magnitude_change_all_pcs.png")

# Plot 4: Percentage change by PC (2x2 grid)
fig, axes = plt.subplots(2, 2, figsize=(14, 10))
axes = axes.flatten()

for pc_idx in range(num_pcs):
    pc_num = pc_idx + 1
    ax = axes[pc_idx]

    near_pct_vals = np.array(pca_results[f'pc{pc_num}_near_pct'])
    far_pct_vals = np.array(pca_results[f'pc{pc_num}_far_pct'])

    ax.plot(pca_layer_indices, near_pct_vals, 'o-', label=f'Near (≤ {spatial_threshold} Å)',
            color='tab:blue', markersize=4, linewidth=1.5)
    ax.plot(pca_layer_indices, far_pct_vals, 's-', label=f'Far (> {spatial_threshold} Å)',
            color='tab:orange', markersize=4, linewidth=1.5)

    ax.set_xlabel('Layer Index', fontsize=11)
    ax.set_ylabel('Median % Change', fontsize=11)
    ax.set_title(f'PC{pc_num}: Percentage Change Between Layers', fontsize=12)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, num_layers, 5))

plt.suptitle('PC Percentage Change (Median): ||PC_L+1| - |PC_L|| / |PC_L| × 100\n(Excluding pairs with baseline < 5th percentile)', fontsize=12, y=1.02)
plt.tight_layout()
plt.savefig('pca_percentage_change_by_pc.png', dpi=150, bbox_inches='tight')
print("Saved: pca_percentage_change_by_pc.png")


# =============================================================================
# Step 4: Summary
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)

for pc_num in range(1, num_pcs + 1):
    near_vals = np.array(pca_results[f'pc{pc_num}_near_mean'])
    far_vals = np.array(pca_results[f'pc{pc_num}_far_mean'])
    diff_vals = np.array(pca_results[f'pc{pc_num}_diff'])
    near_pct_vals = np.array(pca_results[f'pc{pc_num}_near_pct'])
    far_pct_vals = np.array(pca_results[f'pc{pc_num}_far_pct'])

    n_near_greater = np.sum(diff_vals > 0)
    n_far_greater = np.sum(diff_vals < 0)

    print(f"\nPC{pc_num}:")
    print(f"  Layers where near changes MORE: {n_near_greater} / {len(diff_vals)}")
    print(f"  Layers where far changes MORE:  {n_far_greater} / {len(diff_vals)}")
    print(f"  Avg near abs change: {np.nanmean(near_vals):.4f} ± {np.nanstd(near_vals):.4f}")
    print(f"  Avg far abs change:  {np.nanmean(far_vals):.4f} ± {np.nanstd(far_vals):.4f}")
    print(f"  Median near % change: {np.nanmedian(near_pct_vals):.2f}%")
    print(f"  Median far % change:  {np.nanmedian(far_pct_vals):.2f}%")

# Show plots at the end
plt.show()

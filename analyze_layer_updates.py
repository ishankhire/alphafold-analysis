#!/usr/bin/env python3
"""
Analyze cosine change of pairwise feature vectors across layers.

Compares how much pair representations change in direction between consecutive
layers for spatially-near (≤8 Å) vs spatially-far (>8 Å) residue pairs.

Uses within-layer z-score normalization for comparability across layers.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("visualizations", exist_ok=True)

try:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    print("Biopython is required. Install with: pip install biopython")
    exit(1)

# Settings
base_dir = os.path.dirname(os.path.abspath(__file__))
protein = "7b3a_A"
cif_file = os.path.join(base_dir, "7b3a.cif")
num_layers = 48
spatial_threshold = 8.0  # Angstroms
eps = 1e-10  # Small epsilon to avoid divide-by-zero


# =============================================================================
# Step 1: Build pair masks for "near" and "far"
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
# valid_pairs_mask[i, j] = True if i < j (upper triangle)
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

# Sanity check: near + far should equal total valid pairs
assert n_near + n_far == n_valid_pairs, "ERROR: near + far != total valid pairs"
print("✓ Sanity check passed: near_count + far_count = total valid pairs")


# =============================================================================
# Step 2 & 3: Compute cosine change per layer transition and z-score normalize
# =============================================================================
print("\n" + "=" * 60)
print("STEP 2-3: Computing cosine changes")
print("=" * 60)

# Storage for results
results = {
    'layer_index': [],
    'near_count': [],
    'far_count': [],
    'mean_z_cosine_change_near': [],
    'mean_z_cosine_change_far': [],
    'median_z_cosine_change_near': [],
    'median_z_cosine_change_far': [],
    'mean_difference_near_minus_far': [],
    'raw_mean_cosine_change_near': [],
    'raw_mean_cosine_change_far': [],
    'layer_mean_cosine_change': [],
    'layer_std_cosine_change': [],
}

for layer_idx in range(1, num_layers):
    layer_prev = layer_idx - 1
    layer_curr = layer_idx

    filepath_prev = os.path.join(base_dir, protein, f"{protein}_pair_block_{layer_prev}.npy")
    filepath_curr = os.path.join(base_dir, protein, f"{protein}_pair_block_{layer_curr}.npy")

    if not os.path.exists(filepath_prev) or not os.path.exists(filepath_curr):
        print(f"Skipping layer {layer_prev} -> {layer_curr}: files not found")
        continue

    # Load layer data
    data_prev = np.load(filepath_prev).astype(np.float64)
    data_curr = np.load(filepath_curr).astype(np.float64)

    r_prev, _, C = data_prev.shape
    r_curr = data_curr.shape[0]

    # Alignment: the entity sequence (280 residues) has a 4-residue N-terminal
    # prefix (GHMA) not in the resolved structure. CIF residue i corresponds to
    # pair block index i + PAIR_OFFSET.
    PAIR_OFFSET = 4
    r = n_residues  # iterate over CIF residues

    # Masks are already n_residues × n_residues, no adjustment needed
    local_valid_mask = valid_pairs_mask
    local_near_mask = near_mask
    local_far_mask = far_mask

    # Compute cosine similarity and cosine change for each valid pair
    cosine_change_matrix = np.full((r, r), np.nan)

    for i in range(r):
        for j in range(i + 1, r):  # Only upper triangle (i < j)
            vec_prev = data_prev[i + PAIR_OFFSET, j + PAIR_OFFSET, :]
            vec_curr = data_curr[i + PAIR_OFFSET, j + PAIR_OFFSET, :]

            # Compute norms
            norm_prev = np.linalg.norm(vec_prev)
            norm_curr = np.linalg.norm(vec_curr)

            # Compute cosine similarity with epsilon guard
            if norm_prev > eps and norm_curr > eps:
                cos_sim = np.dot(vec_prev, vec_curr) / (norm_prev * norm_curr)
                # Clamp to [-1, 1] for numerical stability
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
            else:
                cos_sim = 0.0  # If either vector is zero, treat as orthogonal

            # Convert to cosine change
            cosine_change = 1.0 - cos_sim
            cosine_change_matrix[i, j] = cosine_change

    # Sanity check: cosine similarity should be in [-1, 1], so cosine_change in [0, 2]
    valid_changes = cosine_change_matrix[local_valid_mask[:r, :r]]
    assert np.all((valid_changes >= 0) & (valid_changes <= 2)), \
        f"ERROR: cosine_change out of range [0, 2] at layer {layer_idx}"

    # Step 3: Z-score normalization within this layer transition
    layer_mean = np.mean(valid_changes)
    layer_std = np.std(valid_changes)

    if layer_std < eps:
        layer_std = 1.0  # Avoid division by zero if all changes are identical

    z_scored_matrix = (cosine_change_matrix - layer_mean) / layer_std

    # Sanity check: z-scored values should have ~0 mean and ~1 std
    z_valid = z_scored_matrix[local_valid_mask[:r, :r]]
    z_mean = np.mean(z_valid)
    z_std = np.std(z_valid)
    assert abs(z_mean) < 0.01, f"ERROR: z-score mean not ~0 at layer {layer_idx}: {z_mean}"
    assert abs(z_std - 1.0) < 0.01, f"ERROR: z-score std not ~1 at layer {layer_idx}: {z_std}"

    # Step 4: Aggregate near vs far statistics
    near_z_values = z_scored_matrix[local_near_mask[:r, :r]]
    far_z_values = z_scored_matrix[local_far_mask[:r, :r]]

    near_raw_values = cosine_change_matrix[local_near_mask[:r, :r]]
    far_raw_values = cosine_change_matrix[local_far_mask[:r, :r]]

    local_near_count = len(near_z_values)
    local_far_count = len(far_z_values)

    # Handle empty sets
    if local_near_count > 0:
        mean_z_near = np.mean(near_z_values)
        median_z_near = np.median(near_z_values)
        raw_mean_near = np.mean(near_raw_values)
    else:
        mean_z_near = np.nan
        median_z_near = np.nan
        raw_mean_near = np.nan

    if local_far_count > 0:
        mean_z_far = np.mean(far_z_values)
        median_z_far = np.median(far_z_values)
        raw_mean_far = np.mean(far_raw_values)
    else:
        mean_z_far = np.nan
        median_z_far = np.nan
        raw_mean_far = np.nan

    # Difference score
    if not np.isnan(mean_z_near) and not np.isnan(mean_z_far):
        diff = mean_z_near - mean_z_far
    else:
        diff = np.nan

    # Store results
    results['layer_index'].append(layer_idx)
    results['near_count'].append(local_near_count)
    results['far_count'].append(local_far_count)
    results['mean_z_cosine_change_near'].append(mean_z_near)
    results['mean_z_cosine_change_far'].append(mean_z_far)
    results['median_z_cosine_change_near'].append(median_z_near)
    results['median_z_cosine_change_far'].append(median_z_far)
    results['mean_difference_near_minus_far'].append(diff)
    results['raw_mean_cosine_change_near'].append(raw_mean_near)
    results['raw_mean_cosine_change_far'].append(raw_mean_far)
    results['layer_mean_cosine_change'].append(layer_mean)
    results['layer_std_cosine_change'].append(layer_std)

    print(f"Layer {layer_prev:2d} -> {layer_curr:2d}: "
          f"z_near = {mean_z_near:+.4f}, z_far = {mean_z_far:+.4f}, "
          f"diff = {diff:+.4f}")

print("✓ Sanity check passed: all z-scores have ~0 mean and ~1 std within each layer")
print("✓ Sanity check passed: all cosine similarities in [-1, 1]")


# =============================================================================
# Step 5: Outputs - Table and Plots
# =============================================================================
print("\n" + "=" * 60)
print("STEP 5: Generating outputs")
print("=" * 60)

# Convert to arrays
layer_indices = np.array(results['layer_index'])
z_near = np.array(results['mean_z_cosine_change_near'])
z_far = np.array(results['mean_z_cosine_change_far'])
z_diff = np.array(results['mean_difference_near_minus_far'])
raw_near = np.array(results['raw_mean_cosine_change_near'])
raw_far = np.array(results['raw_mean_cosine_change_far'])

# Save CSV table
csv_path = "layer_cosine_change_analysis.csv"
with open(csv_path, 'w') as f:
    f.write("layer_index,near_count,far_count,"
            "raw_mean_cosine_change_near,raw_mean_cosine_change_far,raw_difference_near_minus_far,"
            "mean_z_cosine_change_near,mean_z_cosine_change_far,"
            "layer_mean_cosine_change,layer_std_cosine_change\n")
    for i in range(len(results['layer_index'])):
        raw_diff_i = results['raw_mean_cosine_change_near'][i] - results['raw_mean_cosine_change_far'][i]
        f.write(f"{results['layer_index'][i]},"
                f"{results['near_count'][i]},"
                f"{results['far_count'][i]},"
                f"{results['raw_mean_cosine_change_near'][i]:.6f},"
                f"{results['raw_mean_cosine_change_far'][i]:.6f},"
                f"{raw_diff_i:.6f},"
                f"{results['mean_z_cosine_change_near'][i]:.6f},"
                f"{results['mean_z_cosine_change_far'][i]:.6f},"
                f"{results['layer_mean_cosine_change'][i]:.6f},"
                f"{results['layer_std_cosine_change'][i]:.6f}\n")
print(f"Saved: {csv_path}")

# Compute raw difference for plotting
raw_diff = raw_near - raw_far

# Create plots - focus on RAW values which are more interpretable for near vs far comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Raw cosine change for near vs far (MAIN PLOT)
ax1 = axes[0, 0]
ax1.plot(layer_indices, raw_near, 'o-', label=f'Near (≤ {spatial_threshold} Å)',
         color='tab:blue', markersize=5, linewidth=1.5)
ax1.plot(layer_indices, raw_far, 's-', label=f'Far (> {spatial_threshold} Å)',
         color='tab:orange', markersize=5, linewidth=1.5)
ax1.set_xlabel('Layer Index', fontsize=11)
ax1.set_ylabel('Mean Cosine Change (1 - cos_sim)', fontsize=11)
ax1.set_title('Cosine Change by Spatial Distance\n(Higher = More Direction Change)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, num_layers, 5))

# Plot 2: Raw difference (near - far) - MOST INTERPRETABLE
ax2 = axes[0, 1]
colors = ['tab:blue' if d > 0 else 'tab:orange' for d in raw_diff]
ax2.bar(layer_indices, raw_diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Layer Index', fontsize=11)
ax2.set_ylabel('Near − Far (Raw Cosine Change)', fontsize=11)
ax2.set_title('Difference: Near minus Far\n(Positive = Near pairs change MORE)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(range(0, num_layers, 5))

# Plot 3: Ratio of near/far change (another interpretable metric)
ax3 = axes[1, 0]
ratio = raw_near / np.where(raw_far > eps, raw_far, eps)
ax3.plot(layer_indices, ratio, 'o-', color='tab:purple', markersize=5, linewidth=1.5)
ax3.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Equal change')
ax3.set_xlabel('Layer Index', fontsize=11)
ax3.set_ylabel('Near / Far Ratio', fontsize=11)
ax3.set_title('Ratio of Cosine Change: Near / Far\n(>1 = Near changes more, <1 = Far changes more)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, num_layers, 5))

# Plot 4: Layer-wise mean and std of cosine change (overall activity)
ax4 = axes[1, 1]
layer_means = np.array(results['layer_mean_cosine_change'])
layer_stds = np.array(results['layer_std_cosine_change'])
ax4.errorbar(layer_indices, layer_means, yerr=layer_stds, fmt='o-',
             color='tab:green', markersize=4, linewidth=1.5, capsize=3, alpha=0.8)
ax4.set_xlabel('Layer Index', fontsize=11)
ax4.set_ylabel('Cosine Change (Mean ± Std)', fontsize=11)
ax4.set_title('Overall Cosine Change per Layer\n(How much does the layer update representations?)', fontsize=12)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(0, num_layers, 5))

plt.tight_layout()
plt.savefig('visualizations/layer_cosine_change_analysis.png', dpi=150)
plt.show()
print("Saved: visualizations/layer_cosine_change_analysis.png")


# =============================================================================
# Step 6: Final Summary (using RAW values, not z-scored)
# =============================================================================
print("\n" + "=" * 60)
print("SUMMARY (Raw Cosine Change)")
print("=" * 60)

# Count layers where near > far vs far > near (using raw values)
n_near_greater = np.sum(raw_diff > 0)
n_far_greater = np.sum(raw_diff < 0)

print(f"Layers where near pairs change MORE than far: {n_near_greater} / {len(raw_diff)}")
print(f"Layers where far pairs change MORE than near: {n_far_greater} / {len(raw_diff)}")
print(f"\nAverage raw cosine change (near): {np.nanmean(raw_near):.4f} ± {np.nanstd(raw_near):.4f}")
print(f"Average raw cosine change (far):  {np.nanmean(raw_far):.4f} ± {np.nanstd(raw_far):.4f}")
print(f"Average difference (near - far):  {np.nanmean(raw_diff):.4f} ± {np.nanstd(raw_diff):.4f}")

# Identify layers with largest near vs far differences (using raw values)
sorted_indices = np.argsort(raw_diff)[::-1]
print(f"\nTop 5 layers with MOST short-range (near) updates:")
for idx in sorted_indices[:5]:
    print(f"  Layer {layer_indices[idx]}: near={raw_near[idx]:.4f}, far={raw_far[idx]:.4f}, diff={raw_diff[idx]:+.4f}")

print(f"\nTop 5 layers with MOST long-range (far) updates:")
for idx in sorted_indices[-5:]:
    print(f"  Layer {layer_indices[idx]}: near={raw_near[idx]:.4f}, far={raw_far[idx]:.4f}, diff={raw_diff[idx]:+.4f}")

# Note about z-scoring
print("\n" + "-" * 60)
print("NOTE: Z-scored values are included in the CSV but plots use raw values.")
print("Z-scoring artifacts occur because far pairs dominate the distribution,")
print("causing far pairs to always center around z=0 by construction.")
print("Raw cosine change is more interpretable for near vs far comparison.")


# =============================================================================
# SEQUENTIAL DISTANCE ANALYSIS
# Repeat the analysis using sequential distance (|i-j| <= 5) instead of spatial
# =============================================================================
print("\n" + "=" * 60)
print("SEQUENTIAL DISTANCE ANALYSIS")
print("=" * 60)

seq_threshold = 5  # Sequential distance threshold

# Create sequential distance matrix
seq_dist_matrix = np.zeros((n_residues, n_residues))
for i in range(n_residues):
    for j in range(n_residues):
        seq_dist_matrix[i, j] = abs(i - j)

# Create sequential near and far masks
seq_near_mask = valid_pairs_mask & (seq_dist_matrix <= seq_threshold)
seq_far_mask = valid_pairs_mask & (seq_dist_matrix > seq_threshold)

n_seq_near = np.sum(seq_near_mask)
n_seq_far = np.sum(seq_far_mask)
print(f"Sequential near pairs (|i-j| ≤ {seq_threshold}): {n_seq_near}")
print(f"Sequential far pairs (|i-j| > {seq_threshold}): {n_seq_far}")

assert n_seq_near + n_seq_far == n_valid_pairs, "ERROR: seq near + far != total valid pairs"
print("✓ Sanity check passed: seq_near + seq_far = total valid pairs")

# Storage for sequential results
seq_results = {
    'layer_index': [],
    'near_count': [],
    'far_count': [],
    'raw_mean_cosine_change_near': [],
    'raw_mean_cosine_change_far': [],
    'layer_mean_cosine_change': [],
    'layer_std_cosine_change': [],
}

print("\nComputing cosine changes with sequential distance masks...")
for layer_idx in range(1, num_layers):
    layer_prev = layer_idx - 1
    layer_curr = layer_idx

    filepath_prev = os.path.join(base_dir, protein, f"{protein}_pair_block_{layer_prev}.npy")
    filepath_curr = os.path.join(base_dir, protein, f"{protein}_pair_block_{layer_curr}.npy")

    if not os.path.exists(filepath_prev) or not os.path.exists(filepath_curr):
        continue

    # Load layer data
    data_prev = np.load(filepath_prev).astype(np.float64)
    data_curr = np.load(filepath_curr).astype(np.float64)

    r_prev, _, C = data_prev.shape
    r_curr = data_curr.shape[0]
    PAIR_OFFSET = 4
    r = n_residues  # iterate over CIF residues

    # Masks are already n_residues × n_residues, no adjustment needed
    local_valid_mask = valid_pairs_mask
    local_seq_near_mask = seq_near_mask
    local_seq_far_mask = seq_far_mask

    # Compute cosine change matrix
    cosine_change_matrix = np.full((r, r), np.nan)

    for i in range(r):
        for j in range(i + 1, r):
            vec_prev = data_prev[i + PAIR_OFFSET, j + PAIR_OFFSET, :]
            vec_curr = data_curr[i + PAIR_OFFSET, j + PAIR_OFFSET, :]

            norm_prev = np.linalg.norm(vec_prev)
            norm_curr = np.linalg.norm(vec_curr)

            if norm_prev > eps and norm_curr > eps:
                cos_sim = np.dot(vec_prev, vec_curr) / (norm_prev * norm_curr)
                cos_sim = np.clip(cos_sim, -1.0, 1.0)
            else:
                cos_sim = 0.0

            cosine_change_matrix[i, j] = 1.0 - cos_sim

    # Get values for sequential masks
    valid_changes = cosine_change_matrix[local_valid_mask[:r, :r]]
    layer_mean = np.mean(valid_changes)
    layer_std = np.std(valid_changes)

    seq_near_values = cosine_change_matrix[local_seq_near_mask[:r, :r]]
    seq_far_values = cosine_change_matrix[local_seq_far_mask[:r, :r]]

    local_near_count = len(seq_near_values)
    local_far_count = len(seq_far_values)

    raw_mean_near = np.mean(seq_near_values) if local_near_count > 0 else np.nan
    raw_mean_far = np.mean(seq_far_values) if local_far_count > 0 else np.nan

    seq_results['layer_index'].append(layer_idx)
    seq_results['near_count'].append(local_near_count)
    seq_results['far_count'].append(local_far_count)
    seq_results['raw_mean_cosine_change_near'].append(raw_mean_near)
    seq_results['raw_mean_cosine_change_far'].append(raw_mean_far)
    seq_results['layer_mean_cosine_change'].append(layer_mean)
    seq_results['layer_std_cosine_change'].append(layer_std)

    print(f"Layer {layer_prev:2d} -> {layer_curr:2d}: "
          f"seq_near = {raw_mean_near:.4f}, seq_far = {raw_mean_far:.4f}, "
          f"diff = {raw_mean_near - raw_mean_far:+.4f}")

# Convert to arrays for plotting
seq_layer_indices = np.array(seq_results['layer_index'])
seq_raw_near = np.array(seq_results['raw_mean_cosine_change_near'])
seq_raw_far = np.array(seq_results['raw_mean_cosine_change_far'])
seq_raw_diff = seq_raw_near - seq_raw_far

# Save CSV for sequential analysis
seq_csv_path = "layer_cosine_change_sequential.csv"
with open(seq_csv_path, 'w') as f:
    f.write("layer_index,near_count,far_count,"
            "raw_mean_cosine_change_near,raw_mean_cosine_change_far,raw_difference_near_minus_far,"
            "layer_mean_cosine_change,layer_std_cosine_change\n")
    for i in range(len(seq_results['layer_index'])):
        diff_i = seq_results['raw_mean_cosine_change_near'][i] - seq_results['raw_mean_cosine_change_far'][i]
        f.write(f"{seq_results['layer_index'][i]},"
                f"{seq_results['near_count'][i]},"
                f"{seq_results['far_count'][i]},"
                f"{seq_results['raw_mean_cosine_change_near'][i]:.6f},"
                f"{seq_results['raw_mean_cosine_change_far'][i]:.6f},"
                f"{diff_i:.6f},"
                f"{seq_results['layer_mean_cosine_change'][i]:.6f},"
                f"{seq_results['layer_std_cosine_change'][i]:.6f}\n")
print(f"Saved: {seq_csv_path}")

# Create plots for sequential distance analysis
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Plot 1: Raw cosine change for sequential near vs far
ax1 = axes[0, 0]
ax1.plot(seq_layer_indices, seq_raw_near, 'o-', label=f'Near (|i-j| ≤ {seq_threshold})',
         color='tab:blue', markersize=5, linewidth=1.5)
ax1.plot(seq_layer_indices, seq_raw_far, 's-', label=f'Far (|i-j| > {seq_threshold})',
         color='tab:orange', markersize=5, linewidth=1.5)
ax1.set_xlabel('Layer Index', fontsize=11)
ax1.set_ylabel('Mean Cosine Change (1 - cos_sim)', fontsize=11)
ax1.set_title('Cosine Change by Sequential Distance\n(Higher = More Direction Change)', fontsize=12)
ax1.legend(fontsize=10)
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, num_layers, 5))

# Plot 2: Raw difference (near - far)
ax2 = axes[0, 1]
colors = ['tab:blue' if d > 0 else 'tab:orange' for d in seq_raw_diff]
ax2.bar(seq_layer_indices, seq_raw_diff, color=colors, alpha=0.7, edgecolor='black', linewidth=0.5)
ax2.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax2.set_xlabel('Layer Index', fontsize=11)
ax2.set_ylabel('Near − Far (Raw Cosine Change)', fontsize=11)
ax2.set_title('Difference: Sequential Near minus Far\n(Positive = Near pairs change MORE)', fontsize=12)
ax2.grid(True, alpha=0.3, axis='y')
ax2.set_xticks(range(0, num_layers, 5))

# Plot 3: Ratio of near/far change
ax3 = axes[1, 0]
seq_ratio = seq_raw_near / np.where(seq_raw_far > eps, seq_raw_far, eps)
ax3.plot(seq_layer_indices, seq_ratio, 'o-', color='tab:purple', markersize=5, linewidth=1.5)
ax3.axhline(y=1, color='black', linestyle='--', linewidth=1, label='Equal change')
ax3.set_xlabel('Layer Index', fontsize=11)
ax3.set_ylabel('Near / Far Ratio', fontsize=11)
ax3.set_title('Ratio of Cosine Change: Sequential Near / Far\n(>1 = Near changes more)', fontsize=12)
ax3.legend(fontsize=10)
ax3.grid(True, alpha=0.3)
ax3.set_xticks(range(0, num_layers, 5))

# Plot 4: Comparison of spatial vs sequential difference
ax4 = axes[1, 1]
ax4.plot(layer_indices, raw_diff, 'o-', label=f'Spatial (≤{spatial_threshold} Å)',
         color='tab:green', markersize=5, linewidth=1.5)
ax4.plot(seq_layer_indices, seq_raw_diff, 's-', label=f'Sequential (|i-j| ≤{seq_threshold})',
         color='tab:red', markersize=5, linewidth=1.5)
ax4.axhline(y=0, color='black', linestyle='-', linewidth=1)
ax4.set_xlabel('Layer Index', fontsize=11)
ax4.set_ylabel('Near − Far (Raw Cosine Change)', fontsize=11)
ax4.set_title('Comparison: Spatial vs Sequential Distance\n(Near minus Far Difference)', fontsize=12)
ax4.legend(fontsize=10)
ax4.grid(True, alpha=0.3)
ax4.set_xticks(range(0, num_layers, 5))

plt.tight_layout()
plt.savefig('visualizations/layer_cosine_change_sequential.png', dpi=150)
print("Saved: visualizations/layer_cosine_change_sequential.png")

# Sequential Summary
print("\n" + "=" * 60)
print("SEQUENTIAL DISTANCE SUMMARY")
print("=" * 60)

n_seq_near_greater = np.sum(seq_raw_diff > 0)
n_seq_far_greater = np.sum(seq_raw_diff < 0)

print(f"Layers where sequential-near pairs change MORE: {n_seq_near_greater} / {len(seq_raw_diff)}")
print(f"Layers where sequential-far pairs change MORE:  {n_seq_far_greater} / {len(seq_raw_diff)}")
print(f"\nAverage raw cosine change (seq near): {np.nanmean(seq_raw_near):.4f} ± {np.nanstd(seq_raw_near):.4f}")
print(f"Average raw cosine change (seq far):  {np.nanmean(seq_raw_far):.4f} ± {np.nanstd(seq_raw_far):.4f}")
print(f"Average difference (near - far):      {np.nanmean(seq_raw_diff):.4f} ± {np.nanstd(seq_raw_diff):.4f}")

# Correlation between spatial and sequential results
corr = np.corrcoef(raw_diff, seq_raw_diff)[0, 1]
print(f"\nCorrelation between spatial and sequential (near-far) differences: {corr:.4f}")

# Show plots at the end
plt.show()

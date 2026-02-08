#!/usr/bin/env python3
"""
Analyze the correlation between sequential distance and spatial distance.
Also run the magnitude analysis at 14 Angstroms threshold.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats

try:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    print("Biopython is required. Install with: pip install biopython")
    exit(1)

# Settings
base_dir = os.path.dirname(os.path.abspath(__file__))
os.makedirs("visualizations", exist_ok=True)
protein = "7b3a_A"
cif_file = os.path.join(base_dir, "7b3a.cif")
all_layers = list(range(48))

# --- Load structure and compute distances ---
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
n = len(ca_coords)
print(f"Found {n} residues")

# Compute both distance types for all pairs
print("\nComputing pairwise distances...")
sequential_distances = []
spatial_distances = []

for i in range(n):
    for j in range(i + 1, n):  # Upper triangle only to avoid duplicates
        seq_dist = abs(i - j)
        spatial_dist = np.linalg.norm(ca_coords[i] - ca_coords[j])
        sequential_distances.append(seq_dist)
        spatial_distances.append(spatial_dist)

sequential_distances = np.array(sequential_distances)
spatial_distances = np.array(spatial_distances)

# --- Correlation Analysis ---
print("\n" + "=" * 60)
print("CORRELATION ANALYSIS: Sequential vs Spatial Distance")
print("=" * 60)

pearson_r, pearson_p = stats.pearsonr(sequential_distances, spatial_distances)
spearman_r, spearman_p = stats.spearmanr(sequential_distances, spatial_distances)

print(f"Pearson correlation:  r = {pearson_r:.4f} (p = {pearson_p:.2e})")
print(f"Spearman correlation: r = {spearman_r:.4f} (p = {spearman_p:.2e})")

# Check overlap between definitions
seq_threshold = 5
spatial_thresholds = [8, 14]

print(f"\n--- Overlap Analysis ---")
print(f"Sequential short-range: |i-j| <= {seq_threshold}")

for spatial_thresh in spatial_thresholds:
    seq_short = sequential_distances <= seq_threshold
    spatial_short = spatial_distances <= spatial_thresh

    both_short = np.sum(seq_short & spatial_short)
    only_seq_short = np.sum(seq_short & ~spatial_short)
    only_spatial_short = np.sum(~seq_short & spatial_short)
    both_long = np.sum(~seq_short & ~spatial_short)

    total = len(sequential_distances)

    print(f"\nSpatial threshold: {spatial_thresh} Å")
    print(f"  Both short-range:        {both_short:6d} ({100*both_short/total:5.1f}%)")
    print(f"  Only sequential short:   {only_seq_short:6d} ({100*only_seq_short/total:5.1f}%)")
    print(f"  Only spatial short:      {only_spatial_short:6d} ({100*only_spatial_short/total:5.1f}%)")
    print(f"  Both long-range:         {both_long:6d} ({100*both_long/total:5.1f}%)")

    # Jaccard similarity for short-range definitions
    jaccard = both_short / (both_short + only_seq_short + only_spatial_short) if (both_short + only_seq_short + only_spatial_short) > 0 else 0
    print(f"  Jaccard similarity (short-range): {jaccard:.3f}")

# --- Visualization: Scatter/Density Plot ---
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Left: 2D histogram (density)
ax1 = axes[0]
h = ax1.hist2d(sequential_distances, spatial_distances, bins=50, cmap='viridis',
               norm=plt.matplotlib.colors.LogNorm())
plt.colorbar(h[3], ax=ax1, label='Count (log scale)')
ax1.set_xlabel('Sequential Distance |i - j|', fontsize=11)
ax1.set_ylabel('Spatial Distance (Å)', fontsize=11)
ax1.set_title(f'Sequential vs Spatial Distance\nPearson r = {pearson_r:.3f}, Spearman r = {spearman_r:.3f}', fontsize=12)

# Add threshold lines
ax1.axhline(y=8, color='red', linestyle='--', alpha=0.7, label='8 Å')
ax1.axhline(y=14, color='orange', linestyle='--', alpha=0.7, label='14 Å')
ax1.axvline(x=5, color='cyan', linestyle='--', alpha=0.7, label='Seq dist = 5')
ax1.legend(loc='upper right', fontsize=9)

# Right: Mean spatial distance per sequential distance
ax2 = axes[1]
unique_seq = np.unique(sequential_distances)
mean_spatial = [spatial_distances[sequential_distances == s].mean() for s in unique_seq]
std_spatial = [spatial_distances[sequential_distances == s].std() for s in unique_seq]

ax2.errorbar(unique_seq, mean_spatial, yerr=std_spatial, fmt='o-', markersize=3,
             capsize=2, alpha=0.7, color='tab:blue')
ax2.axhline(y=8, color='red', linestyle='--', alpha=0.7, label='8 Å threshold')
ax2.axhline(y=14, color='orange', linestyle='--', alpha=0.7, label='14 Å threshold')
ax2.set_xlabel('Sequential Distance |i - j|', fontsize=11)
ax2.set_ylabel('Mean Spatial Distance (Å)', fontsize=11)
ax2.set_title('Mean Spatial Distance vs Sequential Distance\n(error bars = 1 std)', fontsize=12)
ax2.legend(fontsize=10)
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('visualizations/distance_correlation_analysis.png', dpi=150)
plt.show()
print("\nSaved: visualizations/distance_correlation_analysis.png")

# --- Now run magnitude analysis at 14 Å ---
print("\n" + "=" * 60)
print("MAGNITUDE ANALYSIS AT 14 Å THRESHOLD")
print("=" * 60)

# Build full distance matrix
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = np.linalg.norm(ca_coords[i] - ca_coords[j])

distance_threshold = 14.0

short_range_means = []
long_range_means = []
valid_layers = []

# Alignment: the entity sequence (280 residues) has a 4-residue N-terminal
# prefix (GHMA) not in the resolved structure. CIF residue i corresponds to
# pair block index i + PAIR_OFFSET.
PAIR_OFFSET = 4

for layer_idx in all_layers:
    filepath = os.path.join(base_dir, protein, f"{protein}_pair_block_{layer_idx}.npy")

    if not os.path.exists(filepath):
        continue

    data = np.load(filepath)
    mag_matrix = np.linalg.norm(data, axis=2)

    short_mags = []
    long_mags = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            if dist_matrix[i, j] <= distance_threshold:
                short_mags.append(mag_matrix[i + PAIR_OFFSET, j + PAIR_OFFSET])
            else:
                long_mags.append(mag_matrix[i + PAIR_OFFSET, j + PAIR_OFFSET])

    short_range_means.append(np.mean(short_mags) if short_mags else 0)
    long_range_means.append(np.mean(long_mags) if long_mags else 0)
    valid_layers.append(layer_idx)

    print(f"Layer {layer_idx:2d}: Short-range mean = {short_range_means[-1]:.2f}, Long-range mean = {long_range_means[-1]:.2f}")

valid_layers = np.array(valid_layers)
short_range_means = np.array(short_range_means)
long_range_means = np.array(long_range_means)

# Plot for 14 Å
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(valid_layers, short_range_means, 'o-',
        label=f'Short-range (CA-CA ≤ {distance_threshold} Å)',
        color='tab:blue', markersize=4, linewidth=1.5)
ax.plot(valid_layers, long_range_means, 's-',
        label=f'Long-range (CA-CA > {distance_threshold} Å)',
        color='tab:orange', markersize=4, linewidth=1.5)

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Average Feature Magnitude (L2 norm)', fontsize=12)
ax.set_title(f'Average Pairwise Feature Magnitude by Spatial Distance (14 Å threshold)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 48, 5))
ax.set_xlim(-1, 48)

plt.tight_layout()
plt.savefig('visualizations/spatial_range_magnitude_14A.png', dpi=150)
plt.show()

print(f"\nSaved: visualizations/spatial_range_magnitude_14A.png")

# Save data
with open('spatial_range_magnitude_14A.csv', 'w') as f:
    f.write("layer,short_range_mean,long_range_mean\n")
    for layer, short, long in zip(valid_layers, short_range_means, long_range_means):
        f.write(f"{layer},{short:.6f},{long:.6f}\n")
print("Saved: spatial_range_magnitude_14A.csv")

# --- Sequential distance distribution for spatially close pairs (8 Å) ---
print("\n" + "=" * 60)
print("SEQUENTIAL DISTANCE DISTRIBUTION FOR SPATIALLY CLOSE PAIRS")
print("=" * 60)

spatial_threshold_analysis = 8.0

# Filter pairs that are within 8 Å spatially
spatially_close_mask = spatial_distances <= spatial_threshold_analysis
seq_distances_of_close_pairs = sequential_distances[spatially_close_mask]

total_close_pairs = len(seq_distances_of_close_pairs)
print(f"Total pairs within {spatial_threshold_analysis} Å: {total_close_pairs}")

# Count how many pairs have each sequential distance
max_seq_dist = int(sequential_distances.max())
seq_dist_counts = np.zeros(max_seq_dist + 1)
for seq_d in seq_distances_of_close_pairs:
    seq_dist_counts[int(seq_d)] += 1

# Calculate percentages (skip 0 since i != j)
seq_dist_range = np.arange(1, max_seq_dist + 1)
seq_dist_percentages = (seq_dist_counts[1:] / total_close_pairs) * 100

# Print summary
print(f"\nSequential distance distribution for pairs within {spatial_threshold_analysis} Å:")
cumulative = 0
for i, pct in enumerate(seq_dist_percentages[:20], start=1):  # Print first 20
    cumulative += pct
    if pct > 0:
        print(f"  |i-j| = {i:2d}: {pct:5.2f}% (cumulative: {cumulative:5.1f}%)")

# Find where 90%, 95%, 99% of pairs are covered
cumsum = np.cumsum(seq_dist_percentages)
for threshold_pct in [50, 75, 90, 95, 99]:
    idx = np.searchsorted(cumsum, threshold_pct)
    if idx < len(seq_dist_range):
        print(f"\n{threshold_pct}% of spatially close pairs have sequential distance ≤ {seq_dist_range[idx]}")

# Plot bar chart
fig, ax = plt.subplots(figsize=(14, 6))

# Only plot up to where there are meaningful counts (or max 50 for readability)
plot_range = min(50, max_seq_dist)
bars = ax.bar(seq_dist_range[:plot_range], seq_dist_percentages[:plot_range],
              color='tab:blue', edgecolor='navy', alpha=0.7)

ax.set_xlabel('Sequential Distance |i - j|', fontsize=12)
ax.set_ylabel('Percentage of Spatially Close Pairs (%)', fontsize=12)
ax.set_title(f'Sequential Distance Distribution for Residue Pairs Within {spatial_threshold_analysis} Å\n'
             f'(Total: {total_close_pairs} pairs)', fontsize=13)
ax.grid(True, alpha=0.3, axis='y')

# Add cumulative line on secondary axis
ax2 = ax.twinx()
ax2.plot(seq_dist_range[:plot_range], cumsum[:plot_range], 'r-', linewidth=2, label='Cumulative %')
ax2.set_ylabel('Cumulative Percentage (%)', fontsize=12, color='red')
ax2.tick_params(axis='y', labelcolor='red')
ax2.axhline(y=90, color='red', linestyle='--', alpha=0.5, linewidth=1)
ax2.axhline(y=95, color='red', linestyle=':', alpha=0.5, linewidth=1)
ax2.set_ylim(0, 105)

# Add text annotation for key percentiles
idx_90 = np.searchsorted(cumsum, 90)
if idx_90 < plot_range:
    ax.axvline(x=seq_dist_range[idx_90], color='green', linestyle='--', alpha=0.7)
    ax.text(seq_dist_range[idx_90] + 0.5, max(seq_dist_percentages[:plot_range]) * 0.9,
            f'90% at |i-j|={seq_dist_range[idx_90]}', fontsize=10, color='green')

plt.tight_layout()
plt.savefig('visualizations/spatial_close_pairs_seq_distribution.png', dpi=150)
plt.show()

print(f"\nSaved: visualizations/spatial_close_pairs_seq_distribution.png")

# Save data
with open('spatial_close_pairs_seq_distribution.csv', 'w') as f:
    f.write("sequential_distance,count,percentage,cumulative_percentage\n")
    for i, (cnt, pct, cum) in enumerate(zip(seq_dist_counts[1:], seq_dist_percentages, cumsum), start=1):
        f.write(f"{i},{int(cnt)},{pct:.4f},{cum:.4f}\n")
print("Saved: spatial_close_pairs_seq_distribution.csv")

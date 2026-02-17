#!/usr/bin/env python3
"""
Compare feature magnitude analysis at different spatial thresholds (8 Å and 14 Å).
Generates side-by-side plots for easy comparison.
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
all_layers = list(range(48))
thresholds = [8.0, 14.0]  # Angstroms

# --- Load structure ---
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

# Compute distance matrix
print("Computing CA-CA distance matrix...")
dist_matrix = np.zeros((n, n))
for i in range(n):
    for j in range(n):
        dist_matrix[i, j] = np.linalg.norm(ca_coords[i] - ca_coords[j])

# --- Analyze at each threshold ---
results = {}

for threshold in thresholds:
    print(f"\n--- Analyzing at {threshold} Å threshold ---")

    short_range_means = []
    long_range_means = []
    valid_layers = []

    for layer_idx in all_layers:
        filepath = os.path.join(base_dir, protein, f"{protein}_pair_block_{layer_idx}.npy")

        if not os.path.exists(filepath):
            continue

        data = np.load(filepath)
        r = data.shape[0]
        num = min(r, n)

        mag_matrix = np.linalg.norm(data, axis=2)

        short_mags = []
        long_mags = []

        for i in range(num):
            for j in range(num):
                if i == j:
                    continue
                if dist_matrix[i, j] <= threshold:
                    short_mags.append(mag_matrix[i, j])
                else:
                    long_mags.append(mag_matrix[i, j])

        short_range_means.append(np.mean(short_mags) if short_mags else 0)
        long_range_means.append(np.mean(long_mags) if long_mags else 0)
        valid_layers.append(layer_idx)

    results[threshold] = {
        'layers': np.array(valid_layers),
        'short': np.array(short_range_means),
        'long': np.array(long_range_means)
    }

    print(f"  Short-range pairs: {len(short_mags)}, Long-range pairs: {len(long_mags)}")

# --- Create comparison plots ---

# Plot 1: Side by side
fig, axes = plt.subplots(1, 2, figsize=(14, 5), sharey=True)

for idx, threshold in enumerate(thresholds):
    ax = axes[idx]
    data = results[threshold]

    ax.plot(data['layers'], data['short'], 'o-',
            label=f'Short-range (≤ {threshold} Å)',
            color='tab:blue', markersize=4, linewidth=1.5)
    ax.plot(data['layers'], data['long'], 's-',
            label=f'Long-range (> {threshold} Å)',
            color='tab:orange', markersize=4, linewidth=1.5)

    ax.set_xlabel('Layer', fontsize=12)
    ax.set_ylabel('Average Feature Magnitude (L2 norm)', fontsize=12)
    ax.set_title(f'Spatial Threshold: {threshold} Å', fontsize=13)
    ax.legend(fontsize=10)
    ax.grid(True, alpha=0.3)
    ax.set_xticks(range(0, 48, 5))
    ax.set_xlim(-1, 48)

plt.suptitle('Pairwise Feature Magnitude by Spatial Distance', fontsize=14, y=1.02)
plt.tight_layout()
vis_dir = os.path.join(base_dir, "visualizations")
os.makedirs(vis_dir, exist_ok=True)
plt.savefig(os.path.join(vis_dir, 'spatial_threshold_comparison_sidebyside.png'), dpi=150, bbox_inches='tight')
plt.show()
print(f"\nSaved: {os.path.join(vis_dir, 'spatial_threshold_comparison_sidebyside.png')}")

# Plot 2: All on same axes for direct comparison
fig, ax = plt.subplots(figsize=(12, 6))

colors = {'short': ['tab:blue', 'tab:cyan'], 'long': ['tab:orange', 'tab:red']}
markers = ['o', 's']

for idx, threshold in enumerate(thresholds):
    data = results[threshold]

    ax.plot(data['layers'], data['short'], f'{markers[idx]}-',
            label=f'Short-range (≤ {threshold} Å)',
            color=colors['short'][idx], markersize=4, linewidth=1.5, alpha=0.8)
    ax.plot(data['layers'], data['long'], f'{markers[idx]}--',
            label=f'Long-range (> {threshold} Å)',
            color=colors['long'][idx], markersize=4, linewidth=1.5, alpha=0.8)

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Average Feature Magnitude (L2 norm)', fontsize=12)
ax.set_title('Pairwise Feature Magnitude: 8 Å vs 14 Å Thresholds', fontsize=14)
ax.legend(fontsize=10, ncol=2)
ax.grid(True, alpha=0.3)
ax.set_xticks(range(0, 48, 5))
ax.set_xlim(-1, 48)

plt.tight_layout()
plt.savefig(os.path.join(vis_dir, 'spatial_threshold_comparison_overlay.png'), dpi=150)
plt.show()
print(f"Saved: {os.path.join(vis_dir, 'spatial_threshold_comparison_overlay.png')}")

# --- Save data ---
for threshold in thresholds:
    data = results[threshold]
    filename = f'spatial_range_magnitude_{int(threshold)}A.csv'
    with open(filename, 'w') as f:
        f.write("layer,short_range_mean,long_range_mean\n")
        for layer, short, long_val in zip(data['layers'], data['short'], data['long']):
            f.write(f"{layer},{short:.6f},{long_val:.6f}\n")
    print(f"Saved: {filename}")

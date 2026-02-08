#!/usr/bin/env python3
"""
Analyze feature magnitudes by residue range.

Short-range: |i - j| <= 5
Long-range: |i - j| > 5

Computes average feature magnitude for each category across all layers
and plots them.
"""

import os
import numpy as np
import matplotlib.pyplot as plt

os.makedirs("visualizations", exist_ok=True)

# Settings
base_dir = os.path.dirname(os.path.abspath(__file__))
protein = "7b3a_A"
all_layers = list(range(48))
range_threshold = 5  # Short-range if |i - j| <= 5

# Storage for results
short_range_means = []
long_range_means = []
valid_layers = []

for layer_idx in all_layers:
    filepath = os.path.join(base_dir, protein, f"{protein}_pair_block_{layer_idx}.npy")

    if not os.path.exists(filepath):
        print(f"Skipping layer {layer_idx}: file not found")
        continue

    data = np.load(filepath)
    r, r2, C = data.shape

    # Compute magnitude matrix
    mag_matrix = np.linalg.norm(data, axis=2)  # (r, r)

    # Separate short-range and long-range magnitudes
    short_range_mags = []
    long_range_mags = []

    for i in range(r):
        for j in range(r):
            if i == j:
                continue  # Skip diagonal

            distance = abs(i - j)
            mag = mag_matrix[i, j]

            if distance <= range_threshold:
                short_range_mags.append(mag)
            else:
                long_range_mags.append(mag)

    # Compute averages
    short_mean = np.mean(short_range_mags) if short_range_mags else 0
    long_mean = np.mean(long_range_mags) if long_range_mags else 0

    short_range_means.append(short_mean)
    long_range_means.append(long_mean)
    valid_layers.append(layer_idx)

    print(f"Layer {layer_idx:2d}: Short-range mean = {short_mean:.2f}, Long-range mean = {long_mean:.2f}")

# Convert to arrays
valid_layers = np.array(valid_layers)
short_range_means = np.array(short_range_means)
long_range_means = np.array(long_range_means)

# Plot
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(valid_layers, short_range_means, 'o-', label=f'Short-range (|i-j| ≤ {range_threshold})',
        color='tab:blue', markersize=4, linewidth=1.5)
ax.plot(valid_layers, long_range_means, 's-', label=f'Long-range (|i-j| > {range_threshold})',
        color='tab:orange', markersize=4, linewidth=1.5)

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Average Feature Magnitude (L2 norm)', fontsize=12)
ax.set_title('Average Pairwise Feature Magnitude by Residue Range', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

# Set x-ticks at intervals
ax.set_xticks(range(0, 48, 5))
ax.set_xlim(-1, 48)

plt.tight_layout()

output_fig = "visualizations/range_magnitude_by_layer.png"
plt.savefig(output_fig, dpi=150)
plt.show()

print(f"\nSaved plot to: {output_fig}")

# Also save the data
output_csv = "range_magnitude_by_layer.csv"
with open(output_csv, "w") as f:
    f.write("layer,short_range_mean,long_range_mean\n")
    for layer, short, long in zip(valid_layers, short_range_means, long_range_means):
        f.write(f"{layer},{short:.6f},{long:.6f}\n")

print(f"Saved data to: {output_csv}")

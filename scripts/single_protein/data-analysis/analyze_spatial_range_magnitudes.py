#!/usr/bin/env python3
"""
Analyze feature magnitudes by spatial (3D) distance.

Short-range: CA-CA distance <= 8 Angstroms
Long-range: CA-CA distance > 8 Angstroms

Uses the 7b3a.cif structure file to compute pairwise CA distances.
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
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTEIN_DIR = os.path.join(ROOT_DIR, "proteins", "7b3a")
VIS_DIR = os.path.join(PROTEIN_DIR, "visualizations")
CSV_DIR = os.path.join(PROTEIN_DIR, "csv_files")
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs(CSV_DIR, exist_ok=True)

protein = "7b3a_A"
cif_file = os.path.join(PROTEIN_DIR, "7b3a.cif")
all_layers = list(range(48))
distance_threshold = 8.0  # Angstroms

# --- Step 1: Extract CA coordinates from CIF file ---
print(f"Parsing {cif_file}...")
parser = MMCIFParser(QUIET=True)
structure = parser.get_structure("protein", cif_file)

ca_coords = []
residue_info = []

for model in structure:
    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            if "CA" in residue:
                ca = residue["CA"]
                ca_coords.append(ca.get_coord())
                residue_info.append((chain.get_id(), residue.get_resname(), residue.get_id()[1]))

ca_coords = np.array(ca_coords)
num_residues_structure = len(ca_coords)
print(f"Found {num_residues_structure} residues with CA atoms in structure")

# --- Step 2: Compute pairwise CA-CA distance matrix ---
print("Computing pairwise CA-CA distances...")
dist_matrix = np.zeros((num_residues_structure, num_residues_structure))
for i in range(num_residues_structure):
    for j in range(num_residues_structure):
        dist_matrix[i, j] = np.linalg.norm(ca_coords[i] - ca_coords[j])

# Print distance statistics
off_diag = dist_matrix[~np.eye(num_residues_structure, dtype=bool)]
print(f"CA-CA distance stats: Min={off_diag.min():.2f} Å, Max={off_diag.max():.2f} Å, Mean={off_diag.mean():.2f} Å")

# Count contacts
num_short = np.sum((dist_matrix <= distance_threshold) & (dist_matrix > 0)) // 2
num_long = np.sum(dist_matrix > distance_threshold) // 2
total_pairs = num_residues_structure * (num_residues_structure - 1) // 2
print(f"Short-range pairs (≤{distance_threshold} Å): {num_short} ({100*num_short/total_pairs:.1f}%)")
print(f"Long-range pairs (>{distance_threshold} Å): {num_long} ({100*num_long/total_pairs:.1f}%)")

# --- Step 3: Analyze feature magnitudes by spatial distance ---
print("\nAnalyzing feature magnitudes by spatial distance...")

short_range_means = []
long_range_means = []
valid_layers = []

for layer_idx in all_layers:
    filepath = os.path.join(PROTEIN_DIR, "pair_blocks", f"{protein}_pair_block_{layer_idx}.npy")

    if not os.path.exists(filepath):
        print(f"Skipping layer {layer_idx}: file not found")
        continue

    data = np.load(filepath)
    r, r2, C = data.shape

    # Check if dimensions match
    if layer_idx == all_layers[0]:
        print(f"Pairwise representation shape: ({r}, {r}, {C})")
        if r != num_residues_structure:
            print(f"WARNING: Mismatch! Structure has {num_residues_structure} residues, "
                  f"representation has {r} positions.")
            print("Will use min of both for analysis.")

    # Alignment: the entity sequence (280 residues) has a 4-residue N-terminal
    # prefix (GHMA) not in the resolved structure. CIF residue i corresponds to
    # pair block index i + PAIR_OFFSET.
    PAIR_OFFSET = 4
    n = num_residues_structure  # iterate over CIF residues

    # Compute magnitude matrix
    mag_matrix = np.linalg.norm(data, axis=2)  # (r, r)

    # Separate by spatial distance
    short_range_mags = []
    long_range_mags = []

    for i in range(n):
        for j in range(n):
            if i == j:
                continue

            ca_dist = dist_matrix[i, j]
            mag = mag_matrix[i + PAIR_OFFSET, j + PAIR_OFFSET]

            if ca_dist <= distance_threshold:
                short_range_mags.append(mag)
            else:
                long_range_mags.append(mag)

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

# --- Step 4: Plot ---
fig, ax = plt.subplots(figsize=(12, 6))

ax.plot(valid_layers, short_range_means, 'o-',
        label=f'Short-range (CA-CA ≤ {distance_threshold} Å)',
        color='tab:blue', markersize=4, linewidth=1.5)
ax.plot(valid_layers, long_range_means, 's-',
        label=f'Long-range (CA-CA > {distance_threshold} Å)',
        color='tab:orange', markersize=4, linewidth=1.5)

ax.set_xlabel('Layer', fontsize=12)
ax.set_ylabel('Average Feature Magnitude (L2 norm)', fontsize=12)
ax.set_title('Average Pairwise Feature Magnitude by Spatial Distance (3D)', fontsize=14)
ax.legend(fontsize=11)
ax.grid(True, alpha=0.3)

ax.set_xticks(range(0, 48, 5))
ax.set_xlim(-1, 48)

plt.tight_layout()

output_fig = os.path.join(VIS_DIR, "spatial_range_magnitude_by_layer.png")
plt.savefig(output_fig, dpi=150)
plt.show()

print(f"\nSaved plot to: {output_fig}")

# Save data
output_csv = os.path.join(CSV_DIR, "spatial_range_magnitude_by_layer.csv")
with open(output_csv, "w") as f:
    f.write("layer,short_range_mean,long_range_mean\n")
    for layer, short, long in zip(valid_layers, short_range_means, long_range_means):
        f.write(f"{layer},{short:.6f},{long:.6f}\n")

print(f"Saved data to: {output_csv}")

# Also save the CA distance matrix for reference
ca_matrix_path = os.path.join(CSV_DIR, "ca_distance_matrix.npy")
np.save(ca_matrix_path, dist_matrix)
print(f"Saved CA distance matrix to: {ca_matrix_path}")

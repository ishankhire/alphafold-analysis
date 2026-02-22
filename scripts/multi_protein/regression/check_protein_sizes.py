#!/usr/bin/env python3
"""
check_protein_sizes.py
======================
Quick utility to survey protein sizes on the cluster and estimate memory
requirements for multi_protein_regression.py.

Uses the same directory layout as cluster_multi_protein_regression.py:
  - Protein dirs:  ROOT_DIR/{protein_id}/
  - Pair blocks:   ROOT_DIR/{protein_id}/{protein_id}_pair_block_{layer}.npy
  - Skips:         'Structures', 'csv_files'

Run on the cluster before submitting the regression job:
    python check_protein_sizes.py

Set $SCRATCH if it is not already configured.
"""

import os
import sys
from pathlib import Path

import numpy as np


# ---------------------------------------------------------------------------
# Configuration (same as cluster_multi_protein_regression.py)
# ---------------------------------------------------------------------------

ROOT_DIR = Path(os.environ.get("SCRATCH", "/path/to/your/root/directory"))
SKIP_NAMES = {"Structures", "csv_files"}

FEATURE_DIM = 256       # 128 channels × 2 directions
BYTES_PER_FLOAT = 4     # float32
PEAK_MULTIPLIER = 3.5   # empirical: peak ≈ 3–4× X_all size (vstack + split + scaler)
N_CORES = 4             # cores per sbatch Option 2


# ---------------------------------------------------------------------------
# Discover proteins (same logic as cluster_multi_protein_regression.py)
# ---------------------------------------------------------------------------

def discover_proteins(root_dir):
    """Find all protein subdirectories, skipping Structures and csv_files."""
    proteins = []
    for entry in sorted(root_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in SKIP_NAMES:
            continue
        proteins.append((entry.name, entry))
    return proteins


def get_pair_block_path(protein_dir, protein_id, layer):
    """Return path to a pair block .npy file (same convention as cluster script)."""
    return protein_dir / f"{protein_id}_pair_block_{layer}.npy"


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    root_dir = ROOT_DIR.resolve()
    print(f"Root directory: {root_dir}\n")

    if not root_dir.exists():
        print("ERROR: Root directory does not exist. Set $SCRATCH or edit ROOT_DIR.")
        sys.exit(1)

    all_proteins = discover_proteins(root_dir)
    print(f"Discovered {len(all_proteins)} protein directories.\n")

    sizes = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
        if not pb0_path.exists():
            skipped.append((protein_id, "pair_block_0 not found"))
            continue
        pb0 = np.load(str(pb0_path), mmap_mode="r")
        sizes.append(pb0.shape[0])
        del pb0

    if not sizes:
        print("ERROR: No proteins with pair blocks found.")
        sys.exit(1)

    sizes = np.array(sizes)
    n = len(sizes)

    # --- Statistics ---
    print(f"Proteins with pair blocks: {n}")
    if skipped:
        print(f"Skipped (no pair_block_0):  {len(skipped)}")
    print()
    print(f"Residue counts:")
    print(f"  Min:    {sizes.min()}")
    print(f"  Median: {int(np.median(sizes))}")
    print(f"  Mean:   {sizes.mean():.0f}")
    print(f"  Max:    {sizes.max()}")
    print()

    # --- Memory estimates ---
    pairs_per_protein = sizes * (sizes - 1) // 2
    total_pairs = pairs_per_protein.sum()
    x_all_bytes = total_pairs * FEATURE_DIM * BYTES_PER_FLOAT
    x_all_gb = x_all_bytes / 1e9
    peak_gb = x_all_gb * PEAK_MULTIPLIER

    print(f"Total upper-triangle pairs: {total_pairs:,}")
    print(f"X_all size (per layer):     {x_all_gb:.1f} GB")
    print(f"Estimated peak memory:      {peak_gb:.0f} GB")
    print()

    # --- Recommendation ---
    # Round up to nearest 8 GB for clean sbatch values
    recommended_total = int(np.ceil(peak_gb / 8) * 8)
    mem_per_cpu = recommended_total // N_CORES

    print(f"Recommended sbatch (Option 2: {N_CORES} cores):")
    print(f"  #SBATCH -N1 --ntasks-per-node={N_CORES}")
    print(f"  #SBATCH --mem-per-cpu={mem_per_cpu}G    # total: {mem_per_cpu * N_CORES} GB")
    print(f"  #SBATCH -t8:00:00")


if __name__ == "__main__":
    main()

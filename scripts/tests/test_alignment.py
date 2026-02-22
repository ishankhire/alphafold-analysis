#!/usr/bin/env python3
"""
scripts/tests/test_alignment.py
================================
Standalone alignment sanity checks. Run from the project root:

    python scripts/tests/test_alignment.py

All 5 tests must print PASS.
"""

import sys
import os
from pathlib import Path

# scripts/tests/ is 2 levels deep; climb to project root and add to sys.path.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
from lib.alignment import build_alignment, validate_alignment
from lib.structure import extract_residue_coordinates, compute_distance_matrix, load_distance_matrix

# ---------------------------------------------------------------------------
# Paths (relative to project root)
# ---------------------------------------------------------------------------
CIF_PATH   = str(PROJECT_ROOT / "proteins" / "7b3a_A" / "7b3a.cif")
FIRST_PB   = str(PROJECT_ROOT / "proteins" / "7b3a_A" / "pair_blocks" / "7b3a_A_pair_block_0.npy")
DIST_CSV   = str(PROJECT_ROOT / "proteins" / "7b3a_A" / "csv_files" / "residue_distances.csv")

# ---------------------------------------------------------------------------
# Test 1 — build_alignment produces correct pb_indices for 7B3A
# ---------------------------------------------------------------------------
alignment = build_alignment(CIF_PATH, "A", "7b3a")
assert alignment.n_entity == 280, \
    f"Expected n_entity=280, got {alignment.n_entity}"
assert alignment.n_modeled == 276, \
    f"Expected n_modeled=276, got {alignment.n_modeled}"
assert list(alignment.pb_indices) == list(range(4, 280)), \
    f"Expected [4..279], got {alignment.pb_indices[:5]}..."
print("PASS: 7B3A pb_indices == [4..279]")

# ---------------------------------------------------------------------------
# Test 2 — validate_alignment passes on 7B3A
# ---------------------------------------------------------------------------
ca_result = extract_residue_coordinates(CIF_PATH, "A")
n_ca = len(ca_result['residues'])
validate_alignment(alignment, FIRST_PB, n_ca)   # must not raise
print(f"PASS: validate_alignment OK ({n_ca} CA atoms, n_entity={alignment.n_entity})")

# ---------------------------------------------------------------------------
# Test 3 — pair index spot-check
# ---------------------------------------------------------------------------
assert alignment.pb_indices[0] == 4, \
    f"Expected pb_indices[0]=4, got {alignment.pb_indices[0]}"
assert alignment.pb_indices[10] == 14, \
    f"Expected pb_indices[10]=14, got {alignment.pb_indices[10]}"
pb = np.load(FIRST_PB)
vec = pb[alignment.pb_indices[0], alignment.pb_indices[10], :]
assert vec.shape == (128,), \
    f"Expected shape (128,), got {vec.shape}"
print("PASS: pair block fancy indexing with pb_indices works")

# ---------------------------------------------------------------------------
# Test 4 — distance matrix shape and symmetry
# ---------------------------------------------------------------------------
coords = ca_result['coordinates']
dm = compute_distance_matrix(coords)
assert dm.shape == (276, 276), \
    f"Expected shape (276, 276), got {dm.shape}"
assert np.allclose(dm, dm.T), \
    "Distance matrix is not symmetric"
assert np.all(np.diag(dm) == 0), \
    "Distance matrix diagonal is not zero"
print("PASS: distance matrix shape and symmetry")

# ---------------------------------------------------------------------------
# Test 5 — load_distance_matrix matches compute_distance_matrix
# ---------------------------------------------------------------------------
dm_csv = load_distance_matrix(DIST_CSV)
assert np.allclose(dm, dm_csv, atol=1e-3), \
    "load_distance_matrix does not match computed result"
print("PASS: load_distance_matrix matches computed result")

print("\nAll tests passed.")

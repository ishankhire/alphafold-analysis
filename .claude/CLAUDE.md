# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Analysis of AlphaFold pair representations across 48 layers for protein **7B3A chain A** (280 residues, 128 feature channels per pair).
 The research investigates how pairwise residue features evolve through network depth, comparing spatially-near vs spatially-far residue pairs.

## Dependencies

numpy, scipy, matplotlib, biopython. No package manager config exists — install manually:
```
pip install numpy scipy matplotlib biopython
```

## Running Scripts

All scripts are standalone with hardcoded paths rooted at `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein`. Run directly:
```
python <script_name>.py
```
No build system, test suite, or linter is configured.

## Data Layout

- `7b3a.cif` — mmCIF structure file (ground truth 3D coordinates)
- `residue_coordinates.csv` — CA atom coordinates; columns `chain, residue, number, x, y, z` (276 rows, resolved structure only)
- `residue_distances.csv` — 276×276 CA-CA distance matrix in Angstroms; first row is residue labels (`A_SER0`, `A_MET1`, …), subsequent rows are comma-separated distances
- `7b3a_A/7b3a_A_pair_block_{0..47}.npy` — raw pair representations, shape `(280, 280, 128)`
- `pca_projections/` — PCA-reduced pair blocks, shape `(280, 280, 32)`
- `feature_magnitudes/` — precomputed L2 norm matrices per layer
- `analysis_csv_files/` — analysis output CSVs and scripts for PCA change and threshold comparison

## Data Pipeline

```
7b3a.cif ──► extract_coordinates.py ──► residue_coordinates.csv, residue_distances.csv
                                                │
7b3a_A/*.npy ──────────────────────────────────┤
   │                                            ▼
   ├─► extract_feature_magnitudes.py ──► feature_magnitudes/ (L2 norm matrices + heatmaps)
   ├─► analyze_range_magnitudes.py ──► magnitude by sequential distance (|i-j| ≤ 5)
   ├─► analyze_spatial_range_magnitudes.py ──► magnitude by spatial distance (CA-CA ≤ 8Å)
   ├─► analyze_distance_correlation.py ──► sequential vs spatial distance correlation
   ├─► analyze_layer_updates.py ──► cosine change near vs far (spatial + sequential)
   └─► pca_subspace_main.py ──► pca_projections/ + subspace similarity matrix
        ├─► visualize_pca.py ──► PC component heatmaps
        └─► visualize_similarity.py ──► layer similarity heatmap
```

## Architecture Patterns

**Common structure across all analysis scripts:**
1. Parse `7b3a.cif` with Biopython to get CA coordinates and compute pairwise distance matrix
2. Load pair block `.npy` files per layer
3. Define near/far masks — spatial (CA-CA ≤ 8Å) or sequential (|i-j| ≤ 5)
4. Compute per-pair metrics (L2 norm, cosine change, PCA projection)
5. Aggregate near vs far, plot trends across layers, save CSV + PNG

**Diagonal exclusion:** All analyses exclude self-pairs (i=j). The cosine change analysis (`analyze_layer_updates.py`) uses upper-triangle masks (`np.triu` with `k=1`) counting each pair once. The magnitude analyses use `if i == j: continue` or `np.fill_diagonal(mag_matrix, np.nan)`, iterating both (i,j) and (j,i) — symmetric so means are unaffected.

**Pair representation shape:** Always `(R, R, C)` where R=280 residues and C=128 channels (or 32 after PCA).

**Pair block alignment offset:** The AlphaFold entity sequence has 280 residues, but the resolved CIF structure only contains 276 (residues 0–275). The entity sequence has a 4-residue N-terminal prefix (`GHMA`) not present in the deposited structure. Therefore **CIF residue `i` corresponds to pair block index `i + 4`**. Any script that cross-references pair block data with CIF-derived spatial distances must apply `PAIR_OFFSET = 4` when indexing into the pair block. Scripts that operate solely on pair blocks (e.g., `extract_feature_magnitudes.py`, `analyze_range_magnitudes.py`, PCA scripts) are unaffected.

## Visualizations

All generated plots and figures must be saved to the `visualizations/` directory (or a subdirectory within it, e.g. `visualizations/feature_magnitudes/`, `visualizations/pc_maps/`). When creating new scripts that produce visualizations, always use `os.makedirs("visualizations", exist_ok=True)` and save outputs under `visualizations/`.

## Change Log

When making important or substantive changes to the codebase, record them in `branch-analysis.md`. This file serves as a scratchpad for reasoning about changes — what was changed, why, and any key decisions. Keep it focused: skip minor or trivial edits, but document anything that affects analysis logic, data flow, or architecture.

## Git Workflow

After every significant change, run:
```
git add -A
git commit -m "<descriptive message>"
git push
```
Commit frequently with clear messages. Push after each commit to keep the remote up to date.

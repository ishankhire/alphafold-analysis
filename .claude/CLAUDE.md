# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Analysis of AlphaFold pair representations across 48 layers for multiple proteins (7B3A, 6TF4, 6WMK, 6WQC). Investigates how pairwise residue features evolve through network depth, comparing spatially-near vs spatially-far residue pairs.

## Dependencies

numpy, scipy, matplotlib, biopython, scikit-learn. No package manager config exists — install manually:
```
pip install numpy scipy matplotlib biopython scikit-learn
```

## Project Structure

```
protein/                              ← project root
├── lib/                              ← shared library (importable by all scripts)
│   ├── __init__.py
│   ├── alignment.py                  ← AlignmentResult, build_alignment, validate_alignment
│   └── structure.py                  ← extract_residue_coordinates, compute_distance_matrix,
│                                        load_distance_matrix, detect_chain
├── proteins/
│   ├── 7b3a_A/                       ← all data for protein 7B3A chain A
│   │   ├── 7b3a.cif                  ← mmCIF structure file
│   │   ├── pair_blocks/              ← raw AlphaFold pair representations (7b3a_A_pair_block_{0..47}.npy)
│   │   ├── pca_projections/          ← PCA-reduced pair blocks (shape 280×280×32)
│   │   ├── feature_magnitudes/       ← precomputed L2 norm matrices per layer
│   │   ├── csv_files/                ← all analysis output CSVs + residue_coordinates/distances
│   │   └── visualizations/           ← all output plots (with pc_maps/ subdirectory)
│   ├── 6tf4_A/                       ← same layout as 7b3a_A/
│   ├── 6wmk_A/
│   ├── 6wqc_A/
│   └── multi_protein_results/        ← created at runtime by multi-protein scripts
│       ├── csv_files/
│       └── visualizations/
├── scripts/
│   ├── extract_coordinates.py        ← CIF → coordinates + distance matrix CSV (imports from lib/)
│   ├── single_protein/               ← single-protein analyses (7B3A)
│   │   ├── regression/               ← distance-prediction regression probes
│   │   └── data-analysis/            ← magnitude, PCA, cosine-change, correlation
│   ├── multi_protein/                ← multi-protein analyses (planned)
│   │   └── regression/
│   │       └── probe_layer_regression_multi.py
│   └── tests/
│       └── test_alignment.py         ← alignment sanity checks (run standalone)
├── old_analysis/                     ← archived legacy scripts and CSVs
│   ├── analysis_csv_files/
│   └── misc/
├── CLAUDE.md
└── branch-analysis.md
```

When adding a new protein, create `proteins/<pdb_id>_A/` with the same subdirectory layout as `proteins/7b3a_A/`.

## Running Scripts

All scripts use absolute path computation (`ROOT_DIR = os.path.dirname(os.path.dirname(...))`) so they resolve data correctly regardless of working directory. Run from the project root:
```
python scripts/extract_coordinates.py
python scripts/single_protein/data-analysis/analyze_range_magnitudes.py
python scripts/single_protein/regression/probe_layer_regression.py
python scripts/tests/test_alignment.py
```
No build system or linter is configured. `scripts/tests/test_alignment.py` is a standalone test script (no pytest).

## Data Layout (for protein 7b3a_A)

- `proteins/7b3a_A/7b3a.cif` — mmCIF structure file (ground truth 3D coordinates)
- `proteins/7b3a_A/csv_files/residue_coordinates.csv` — CA atom coordinates; columns `chain, residue, number, x, y, z` (276 rows, resolved structure only)
- `proteins/7b3a_A/csv_files/residue_distances.csv` — 276×276 CA-CA distance matrix in Angstroms; first row is residue labels (`A_SER0`, `A_MET1`, …)
- `proteins/7b3a_A/pair_blocks/7b3a_A_pair_block_{0..47}.npy` — raw pair representations, shape `(280, 280, 128)`
- `proteins/7b3a_A/pca_projections/` — PCA-reduced pair blocks, shape `(280, 280, 32)`
- `proteins/7b3a_A/feature_magnitudes/` — precomputed L2 norm matrices per layer
- `proteins/7b3a_A/csv_files/` — all analysis output CSVs (regression R², coefficients, magnitudes, etc.)
- `proteins/7b3a_A/visualizations/` — all output plots

## Data Pipeline

```
proteins/<pdb_id>_A/<pdb_id>.cif ──► scripts/extract_coordinates.py  (uses lib/structure)
                                         ──► csv_files/residue_coordinates.csv
                                         ──► csv_files/residue_distances.csv
                                                  │
proteins/<pdb_id>_A/pair_blocks/*.npy ────────────┤
   │                                              ▼
   ├─► single_protein/data-analysis/extract_feature_magnitudes.py ──► feature_magnitudes/ + visualizations/
   ├─► single_protein/data-analysis/analyze_range_magnitudes.py ──► magnitude by sequential distance (|i-j| ≤ 5)
   ├─► single_protein/data-analysis/analyze_spatial_range_magnitudes.py ──► magnitude by spatial distance (CA-CA ≤ 8Å)
   ├─► single_protein/data-analysis/analyze_distance_correlation.py ──► sequential vs spatial distance correlation
   ├─► single_protein/data-analysis/analyze_layer_updates.py ──► cosine change near vs far (spatial + sequential)
   └─► single_protein/data-analysis/pca_subspace_main.py ──► pca_projections/ + subspace similarity matrix
        ├─► single_protein/data-analysis/visualize_pca.py ──► PC component heatmaps
        └─► single_protein/data-analysis/visualize_similarity.py ──► layer similarity heatmap

   single_protein/regression/linear_regression_distance.py ──► layer-47 full regression
   single_protein/regression/probe_layer_regression.py ──► layer-wise R² + coefficient evolution
   single_protein/regression/probe_layer_regression_standardized.py ──► standardized version
   single_protein/regression/probe_pca_regression.py ──► PCA dimensionality probe
   single_protein/regression/regression_variants.py ──► top-k selection + upper/lower triangle

   multi_protein/regression/probe_layer_regression_multi.py ──► pooled R² across all proteins
                                         ──► multi_protein_results/csv_files/layer_regression_r2_multi.csv
                                         ──► multi_protein_results/visualizations/layer_regression_r2_multi.png
```

## Architecture Patterns

**Common structure across all analysis scripts:**
1. Compute `ROOT_DIR` and `PROTEIN_DIR` from `__file__` (two or three `os.path.dirname` calls up to project root, then `proteins/7b3a_A`)
2. Parse `proteins/7b3a_A/7b3a.cif` with Biopython to get CA coordinates and compute pairwise distance matrix
3. Load pair block `.npy` files from `proteins/7b3a_A/pair_blocks/` per layer
4. Define near/far masks — spatial (CA-CA ≤ 8Å) or sequential (|i-j| ≤ 5)
5. Compute per-pair metrics (L2 norm, cosine change, PCA projection)
6. Aggregate near vs far, plot trends across layers, save CSV + PNG

**Path convention in scripts:**
```python
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # for scripts/single_protein/**
PROTEIN_DIR = os.path.join(ROOT_DIR, "proteins", "7b3a_A")
```

**lib/ imports:** Scripts that use `lib/` must add the project root to `sys.path`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.alignment import build_alignment, validate_alignment
from lib.structure import extract_residue_coordinates, compute_distance_matrix, detect_chain
```
`scripts/extract_coordinates.py` (one level deep) uses `Path(__file__).parent.parent` as the project root.

**Residue alignment (general):** Use `lib/alignment.build_alignment(cif_path, chain_id, protein_id)` to get an `AlignmentResult` with explicit `pb_indices` mapping each resolved structure residue to its pair-block index. This replaces the old `PAIR_OFFSET = 4` hardcode and works for any protein. Always call `validate_alignment` before using `pb_indices`.

**Diagonal exclusion:** All analyses exclude self-pairs (i=j). The cosine change analysis (`analyze_layer_updates.py`) uses upper-triangle masks (`np.triu` with `k=1`) counting each pair once. The magnitude analyses use `if i == j: continue` or `np.fill_diagonal(mag_matrix, np.nan)`, iterating both (i,j) and (j,i) — symmetric so means are unaffected.

**Pair representation shape:** Always `(R, R, C)` where R=entity sequence length (e.g. 280 for 7B3A) and C=128 channels (or 32 after PCA).

**Pair block alignment offset (7B3A):** The AlphaFold entity sequence has 280 residues, but the resolved CIF structure only contains 276 (residues 0–275). The entity sequence has a 4-residue N-terminal prefix (`GHMA`) not present in the deposited structure. `build_alignment` returns `pb_indices = [4, 5, ..., 279]` for 7B3A, which is numerically identical to the old `PAIR_OFFSET = 4` pattern.

**pca_subspace_main.py** takes `--base_dir` (the pair_blocks directory directly, e.g. `proteins/7b3a_A/pair_blocks`) and `--protein` (filename prefix, e.g. `7b3a_A`). The `base_dir` arg no longer needs a protein subdirectory appended — `load_layer_paths` uses it as the root directly.

## Visualizations

All generated plots must be saved to `proteins/<pdb_id>_A/visualizations/` (or a subdirectory within it, e.g. `visualizations/feature_magnitudes/`, `visualizations/pc_maps/`). Multi-protein results go to `proteins/multi_protein_results/visualizations/`. In scripts, use:
```python
VIS_DIR = os.path.join(PROTEIN_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)
```

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

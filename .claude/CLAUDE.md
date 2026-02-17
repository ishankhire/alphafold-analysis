# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Analysis of AlphaFold pair representations across 48 layers for multiple proteins (currently 7B3A chain A). Investigates how pairwise residue features evolve through network depth, comparing spatially-near vs spatially-far residue pairs.

## Dependencies

numpy, scipy, matplotlib, biopython, scikit-learn. No package manager config exists — install manually:
```
pip install numpy scipy matplotlib biopython scikit-learn
```

## Project Structure

```
protein/                              ← project root
├── proteins/
│   └── 7b3a/                         ← all data for protein 7B3A
│       ├── 7b3a.cif                   ← mmCIF structure file
│       ├── pair_blocks/               ← raw AlphaFold pair representations (7b3a_A_pair_block_{0..47}.npy)
│       ├── pca_projections/           ← PCA-reduced pair blocks (shape 280×280×32)
│       ├── feature_magnitudes/        ← precomputed L2 norm matrices per layer
│       ├── csv_files/                 ← all analysis output CSVs + residue_coordinates/distances
│       └── visualizations/            ← all output plots (with pc_maps/ subdirectory)
├── scripts/
│   ├── extract_coordinates.py         ← CIF → coordinates + distance matrix CSV
│   ├── linear-regression/             ← distance-prediction regression probes
│   │   ├── linear_regression_distance.py
│   │   ├── probe_layer_regression.py
│   │   ├── probe_layer_regression_standardized.py
│   │   ├── probe_pca_regression.py
│   │   └── regression_variants.py
│   └── data-analysis/                 ← magnitude, PCA, cosine-change, correlation
│       ├── analyze_range_magnitudes.py
│       ├── analyze_spatial_range_magnitudes.py
│       ├── analyze_layer_updates.py
│       ├── analyze_distance_correlation.py
│       ├── extract_feature_magnitudes.py
│       ├── pca_subspace_main.py
│       ├── visualize_pca.py
│       └── visualize_similarity.py
├── old_analysis/                       ← archived legacy scripts and CSVs
│   ├── analysis_csv_files/
│   └── misc/
├── CLAUDE.md
└── branch-analysis.md
```

When adding a new protein, create `proteins/<pdb_id>/` with the same subdirectory layout as `proteins/7b3a/`.

## Running Scripts

All scripts use absolute path computation (`ROOT_DIR = os.path.dirname(os.path.dirname(...))`) so they resolve data correctly regardless of working directory. Run from the project root:
```
python scripts/extract_coordinates.py
python scripts/data-analysis/analyze_range_magnitudes.py
python scripts/linear-regression/probe_layer_regression.py
```
No build system, test suite, or linter is configured.

## Data Layout (for protein 7b3a)

- `proteins/7b3a/7b3a.cif` — mmCIF structure file (ground truth 3D coordinates)
- `proteins/7b3a/csv_files/residue_coordinates.csv` — CA atom coordinates; columns `chain, residue, number, x, y, z` (276 rows, resolved structure only)
- `proteins/7b3a/csv_files/residue_distances.csv` — 276×276 CA-CA distance matrix in Angstroms; first row is residue labels (`A_SER0`, `A_MET1`, …)
- `proteins/7b3a/pair_blocks/7b3a_A_pair_block_{0..47}.npy` — raw pair representations, shape `(280, 280, 128)`
- `proteins/7b3a/pca_projections/` — PCA-reduced pair blocks, shape `(280, 280, 32)`
- `proteins/7b3a/feature_magnitudes/` — precomputed L2 norm matrices per layer
- `proteins/7b3a/csv_files/` — all analysis output CSVs (regression R², coefficients, magnitudes, etc.)
- `proteins/7b3a/visualizations/` — all output plots

## Data Pipeline

```
proteins/7b3a/7b3a.cif ──► scripts/extract_coordinates.py
                                ──► csv_files/residue_coordinates.csv
                                ──► csv_files/residue_distances.csv
                                         │
proteins/7b3a/pair_blocks/*.npy ─────────┤
   │                                     ▼
   ├─► data-analysis/extract_feature_magnitudes.py ──► feature_magnitudes/ + visualizations/
   ├─► data-analysis/analyze_range_magnitudes.py ──► magnitude by sequential distance (|i-j| ≤ 5)
   ├─► data-analysis/analyze_spatial_range_magnitudes.py ──► magnitude by spatial distance (CA-CA ≤ 8Å)
   ├─► data-analysis/analyze_distance_correlation.py ──► sequential vs spatial distance correlation
   ├─► data-analysis/analyze_layer_updates.py ──► cosine change near vs far (spatial + sequential)
   └─► data-analysis/pca_subspace_main.py ──► pca_projections/ + subspace similarity matrix
        ├─► data-analysis/visualize_pca.py ──► PC component heatmaps
        └─► data-analysis/visualize_similarity.py ──► layer similarity heatmap

   linear-regression/linear_regression_distance.py ──► layer-47 full regression
   linear-regression/probe_layer_regression.py ──► layer-wise R² + coefficient evolution
   linear-regression/probe_layer_regression_standardized.py ──► standardized version
   linear-regression/probe_pca_regression.py ──► PCA dimensionality probe
   linear-regression/regression_variants.py ──► top-k selection + upper/lower triangle
```

## Architecture Patterns

**Common structure across all analysis scripts:**
1. Compute `ROOT_DIR` and `PROTEIN_DIR` from `__file__` (two or three `os.path.dirname` calls up to project root, then `proteins/7b3a`)
2. Parse `proteins/7b3a/7b3a.cif` with Biopython to get CA coordinates and compute pairwise distance matrix
3. Load pair block `.npy` files from `proteins/7b3a/pair_blocks/` per layer
4. Define near/far masks — spatial (CA-CA ≤ 8Å) or sequential (|i-j| ≤ 5)
5. Compute per-pair metrics (L2 norm, cosine change, PCA projection)
6. Aggregate near vs far, plot trends across layers, save CSV + PNG

**Path convention in scripts:**
```python
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # for scripts/data-analysis/ and scripts/linear-regression/
PROTEIN_DIR = os.path.join(ROOT_DIR, "proteins", "7b3a")
```

**Diagonal exclusion:** All analyses exclude self-pairs (i=j). The cosine change analysis (`analyze_layer_updates.py`) uses upper-triangle masks (`np.triu` with `k=1`) counting each pair once. The magnitude analyses use `if i == j: continue` or `np.fill_diagonal(mag_matrix, np.nan)`, iterating both (i,j) and (j,i) — symmetric so means are unaffected.

**Pair representation shape:** Always `(R, R, C)` where R=280 residues and C=128 channels (or 32 after PCA).

**Pair block alignment offset:** The AlphaFold entity sequence has 280 residues, but the resolved CIF structure only contains 276 (residues 0–275). The entity sequence has a 4-residue N-terminal prefix (`GHMA`) not present in the deposited structure. Therefore **CIF residue `i` corresponds to pair block index `i + 4`**. Any script that cross-references pair block data with CIF-derived spatial distances must apply `PAIR_OFFSET = 4` when indexing into the pair block. Scripts that operate solely on pair blocks (e.g., `extract_feature_magnitudes.py`, `analyze_range_magnitudes.py`, PCA scripts) are unaffected.

**pca_subspace_main.py** takes `--base_dir` (the pair_blocks directory directly, e.g. `proteins/7b3a/pair_blocks`) and `--protein` (filename prefix, e.g. `7b3a_A`). The `base_dir` arg no longer needs a protein subdirectory appended — `load_layer_paths` uses it as the root directly.

## Visualizations

All generated plots must be saved to `proteins/7b3a/visualizations/` (or a subdirectory within it, e.g. `visualizations/feature_magnitudes/`, `visualizations/pc_maps/`). In scripts, use:
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

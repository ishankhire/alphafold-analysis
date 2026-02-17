# Branch Analysis

## Session: 2026-02-07 — Git setup

### Changes made

**`.gitignore` added**
Excludes large data and generated files from version control:
- `*.npy`, `*.cif` — binary data files
- `7b3a_A/`, `pca_projections/`, `feature_magnitudes/` — large data directories
- `.venv/`, `.DS_Store`, `__pycache__/`, `*.pyc` — environment/OS artifacts
- Generated CSVs: `residue_coordinates.csv`, `residue_distances.csv`, `linear_regression_results.csv`, `layer_cosine_change_analysis.csv`, `spatial_range_magnitude_14A.csv`, `spatial_close_pairs_seq_distribution.csv`
- Note: PNG visualizations are intentionally **included** (not ignored) so they appear on GitHub.

**Absolute paths replaced in 8 Python scripts**
All hardcoded absolute filepaths replaced with dynamic equivalents:
- Root-level scripts: `os.path.dirname(os.path.abspath(__file__))`
- `analysis_csv_files/` scripts: `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (one extra level up)

Affected files:
- `analyze_distance_correlation.py`
- `analyze_spatial_range_magnitudes.py`
- `extract_feature_magnitudes.py`
- `linear_regression_distance.py`
- `analyze_layer_updates.py`
- `analyze_range_magnitudes.py`
- `analysis_csv_files/compare_spatial_thresholds.py`
- `analysis_csv_files/analyze_pca_changes.py`

## Session: 2026-02-08 — Visualization output refactor

### What changed
All 10 scripts that generate plots now save to the `visualizations/` directory instead of scattered locations (project root, `feature_magnitudes/`, `pc_maps/`, `analysis_csv_files/`).

### Why
Visualizations were saved in inconsistent locations across the project — some in the root, some in data directories, some relative to the script's CWD. Centralizing them in `visualizations/` makes it easy to find all generated figures and keeps data directories clean.

### Scripts modified (20 total savefig calls across 10 files)
| Script | Output redirect |
|--------|----------------|
| `analyze_distance_correlation.py` | 3 PNGs → `visualizations/` |
| `visualize_similarity.py` | 1 PNG → `visualizations/` |
| `extract_feature_magnitudes.py` | heatmap PNGs → `visualizations/feature_magnitudes/` (data .npy/.csv stays in `feature_magnitudes/`) |
| `analyze_spatial_range_magnitudes.py` | 1 PNG → `visualizations/` |
| `analyze_range_magnitudes.py` | 1 PNG → `visualizations/` |
| `visualize_pca.py` | 48 PNGs → `visualizations/pc_maps/` |
| `analyze_layer_updates.py` | 2 PNGs → `visualizations/` |
| `linear_regression_distance.py` | 1 PNG → `visualizations/` |
| `analysis_csv_files/compare_spatial_thresholds.py` | 2 PNGs → `visualizations/` (uses `base_dir` for absolute path) |
| `analysis_csv_files/analyze_pca_changes.py` | 4 PNGs → `visualizations/` (uses `base_dir` for absolute path) |

### CLAUDE.md updates
Added three new sections:
1. **Visualizations** — all plots save to `visualizations/`
2. **Change Log** — record substantive changes in `branch-analysis.md`
3. **Git Workflow** — `git add -A && commit && push` after significant changes

## Session: 2026-02-08 — Distance encoding probes

### What changed
Added two new analysis scripts that probe how spatial distance information is encoded in AlphaFold pair representations.

### New scripts
- `probe_layer_regression.py` — trains a linear regression (pair features → CA-CA distance) independently for each of the 48 layers. Uses the same fixed train/test split (random_state=42) across all layers for fair comparison.
- `probe_pca_regression.py` — runs PCA on layer-47 pair features, then trains linear regression on top-k PCs for k ∈ {1, 2, 3, 5, 10, 20, 32, 64, 128, 256}. PCA is fit on training data only to avoid leakage.

### New outputs
- `layer_regression_r2.csv` + `visualizations/layer_regression_r2.png`
- `pca_regression_r2.csv` + `visualizations/pca_regression_r2.png`

### Key findings
- **Layer-wise probe**: Distance information emerges primarily in layers 8–18 (R² jumps from 0.71 to 0.96). Gradual refinement continues through later layers. Peak R² at layer 46 (0.9897); slight dip at layer 47 (0.9881).
- **PCA probe**: 20 PCs achieve R²=0.97 — distance is encoded in a ~20-dimensional subspace of the 256-dim feature space. Note: PCA maximizes feature variance, not distance-predictiveness; the fact that the high-variance subspace also predicts distance well is itself a meaningful finding.

### Code review of linear_regression_distance.py
Verified correct: PAIR_OFFSET=4 properly applied, proper train/test split before fitting, concatenated 256-dim feature vector matches reference. Train R²=0.9882 ≈ Test R²=0.9881 confirms no classical overfitting.

## Session: 2026-02-12 — Coefficient evolution across layers

### What changed
Extended `probe_layer_regression.py` to track how regression coefficients evolve across all 48 layers. The script now stores all 256 coefficients per layer (128 upper + 128 lower), computes derived metrics (top-K overlap, Jaccard similarity, rank evolution), saves two CSVs, and produces four new visualizations.

### New outputs
- `regression_coefficients_by_layer.csv` — all 256 coefficients at each of 48 layers (12,288 rows)
- `regression_coefficient_overlap.csv` — per-layer overlap with final layer's top-50 + Jaccard similarity
- `visualizations/regression_coef_heatmap.png` — top 30 feature |coefficients| as a heatmap
- `visualizations/regression_coef_overlap.png` — overlap fraction and Jaccard similarity plots
- `visualizations/regression_coef_rank_evolution.png` — rank trajectories of final layer's top 10 features
- `visualizations/regression_coef_top_channels.png` — top 5 feature magnitude traces + R² vs total importance

### Key findings
- **The top features are NOT stable across layers.** Overlap between any earlier layer's top-50 and the final layer's top-50 stays flat at ~20-30% (vs. ~10% chance baseline). This means different features encode distance at different layers — the representation undergoes substantial reorganization.
- **Coefficient magnitudes decrease as R² increases.** Early layers (0-4) have large individual coefficients (~0.3-0.4) but low R². Later layers have small coefficients (~0.05-0.07) but high R². This suggests distance information becomes more distributed across features in later layers.
- **Consecutive-layer Jaccard similarity is ~0.3-0.7**, meaning 30-70% of the top-50 set changes from one layer to the next. There is no stabilization — even between layers 46 and 47, significant reshuffling occurs.
- **Rank evolution of the final layer's top-10 shows wild oscillation** across earlier layers. Features like upper[9] and lower[64] are NOT consistently important — they bounce between rank 0 and rank 200+.
- **Interpretation**: AlphaFold does not maintain dedicated "distance channels." Instead, distance information is progressively constructed through changing combinations of features, becoming more diffusely distributed in later layers.

## Session: 2026-02-13 — Regression variant analyses

### What changed
Added `regression_variants.py` with two variant analyses on the layer-47 linear regression probe.

### New outputs
- `regression_topk_r2.csv` + `visualizations/regression_topk_r2.png`
- `visualizations/regression_upper_lower_scatter.png`

### Analysis 1: Top-k feature selection
Takes the k features with the largest |coefficient| from the full 256-dim regression on layer 47, retrains using only those features.

| k | R² |
|---|-----|
| 10 | 0.8696 |
| 20 | 0.9274 |
| 32 | 0.9502 |
| 64 | 0.9663 |
| 128 | 0.9867 |
| 256 | 0.9881 |

Just 32 features capture R²=0.95 — most of the distance signal is concentrated in a small subset of channels. Diminishing returns beyond 64 features.

### Analysis 1b: Random-k feature baseline
Same k values, but channels selected uniformly at random (20 trials each). Tests whether top-k channels are genuinely special.

| k | Top-k R² | Random R² (mean±std) | Gap |
|---|----------|----------------------|-----|
| 10 | 0.8696 | 0.4785 ± 0.137 | +0.39 |
| 20 | 0.9274 | 0.7062 ± 0.112 | +0.22 |
| 32 | 0.9502 | 0.8366 ± 0.058 | +0.11 |
| 64 | 0.9663 | 0.9437 ± 0.016 | +0.02 |
| 128 | 0.9867 | 0.9750 ± 0.005 | +0.01 |
| 256 | 0.9881 | 0.9881 | 0.00 |

The top-k channels are clearly special at low k (10 random channels only get R²≈0.48 vs 0.87 for top-k). But the gap shrinks fast — by k=64, random channels achieve R²=0.94. This suggests distance information is broadly distributed across many channels, but a small core of ~10-20 channels carries disproportionate signal. The top-k selection identifies these high-signal channels effectively.

### Analysis 2: Upper vs lower feature triangle
Trains separate regressions using only pair_block[i,j,:] (upper, 128-dim) vs pair_block[j,i,:] (lower, 128-dim).

| Features | Test R² |
|----------|---------|
| Upper (i,j) | 0.9798 |
| Lower (j,i) | 0.9801 |
| Full (concat) | 0.9881 |

Upper and lower triangles encode distance nearly identically (R²≈0.98 each). Concatenating both adds only ~0.8 percentage points, suggesting the two directions carry largely redundant distance information — the pair representation is near-symmetric with respect to spatial distance.

---

## Session: 2026-02-17 — Codebase reorganization for multi-protein support

### Motivation
The root directory had grown to 14 scripts + raw data + outputs in a flat layout that doesn't scale. A second protein is planned, so the structure was reorganized now rather than after the fact.

### What moved

**Data → `proteins/7b3a/`**
- `7b3a.cif` → `proteins/7b3a/7b3a.cif`
- `7b3a_A/` → `proteins/7b3a/pair_blocks/` (directory renamed; file names unchanged)
- `pca_projections/` → `proteins/7b3a/pca_projections/`
- `feature_magnitudes/` → `proteins/7b3a/feature_magnitudes/`
- `csv_files/` → `proteins/7b3a/csv_files/`
- `visualizations/` → `proteins/7b3a/visualizations/`

**Scripts → `scripts/`**
- `scripts/extract_coordinates.py` — standalone extraction (no subfolder)
- `scripts/linear-regression/` — all 5 regression probe scripts
- `scripts/data-analysis/` — all 8 magnitude/PCA/cosine-change/correlation scripts

**Archived → `old_analysis/`**
- `analysis_csv_files/` → `old_analysis/analysis_csv_files/`
- `misc/` → `old_analysis/misc/`

### Path updates
All 14 scripts updated to use `ROOT_DIR` / `PROTEIN_DIR` computed from `__file__` (no more reliance on working directory). Common pattern for scripts two levels deep:
```python
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
PROTEIN_DIR = os.path.join(ROOT_DIR, "proteins", "7b3a")
```
Output paths updated: `csv_files/`, `visualizations/`, `pair_blocks/`, `feature_magnitudes/`, `pca_projections/` now all reference `PROTEIN_DIR`.

`pca_subspace_main.py`: `load_layer_paths` updated to use `base_dir` directly as the pair_blocks directory (no longer appends `protein` as a subdirectory). Pass `--base_dir proteins/7b3a/pair_blocks` when calling via CLI.

### No analysis logic changed
This is a pure structural reorganization. All script outputs (CSVs, PNGs) are identical — only the paths they read from and write to have changed.

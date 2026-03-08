# Branch Analysis

## Session: 2026-03-08 — Add dedicated linear-tree diagnostic variants with rule/coef exports

### What changed
Added two new Delta cluster files:
- `scripts/cluster_scripts/diagnostic_linear_tree_regression.py`
- `scripts/cluster_scripts/diagnostic_linear_tree_regression.sbatch`

Updated:
- `branch-analysis.md`

### Why
You requested a new, separate cluster diagnostic (without modifying the existing
amino-acid diagnostic) that runs five linear-tree variants focused on triangle
usage and sequential-distance features, while exposing tree split rules and
leaf regression coefficients for interpretability.

### Key design decisions
- Created a new script rather than modifying existing diagnostics.
- All five variants are linear-tree models with:
  - `max_depth=2` (at most 4 leaves)
  - split feature restricted to raw `n_residues`
  - `n_residues` included as a regression feature in every leaf model
- Preserved protein-equal training weights (`1 / n_pairs`) and one shared
  protein-level 80/20 split (`seed=42`).
- Added sequential-distance feature variants:
  - raw `seq_sep = |i-j|`
  - cumulative indicators `I[sep==1]`, `I[sep<=2]`, `I[sep<=4]`
- Applied train-only z-scoring to all non-`n_residues` features; kept
  `n_residues` unscaled so tree thresholds remain interpretable in residue
  units.
- Added two tree-interpretability outputs:
  - `diagnostic_linear_tree_rules.csv` (node-level split rules)
  - `diagnostic_linear_tree_leaf_coefficients_raw.csv` (leaf-level raw
    coefficients and intercepts, long-form by feature)
- Added a test-only scatter sample export:
  - `diagnostic_linear_tree_test_scatter_sample.csv`
  - up to 100,000 rows per variant
  - protein-equal sampling to avoid large-protein dominance
- Added a dedicated Delta sbatch file targeting the new script and output
  directory (`$HOME/diagnostic_linear_tree_csv_files`), keeping old sbatch
  wiring untouched.

## Session: 2026-03-07 — Patch lineartree/sklearn compatibility in amino-acid diagnostic

### What changed
Updated:
- `scripts/cluster_scripts/diagnostic_amino_acid_encodings_regression.py`

### Why
Delta runs failed in linear-tree variants with:
`AttributeError: 'LinearTreeRegressor' object has no attribute '_validate_data'`.
This indicates a compatibility mismatch between the installed `lineartree`
version and newer `scikit-learn` behavior in the cluster environment.

### Key design decisions
- Added a small runtime compatibility shim that injects `_validate_data` into
  `LinearTreeRegressor` only when missing.
- The shim delegates to sklearn validation utilities (`check_X_y` /
  `check_array`) and handles keyword differences (`force_all_finite` vs
  `ensure_all_finite`) across sklearn versions.
- Patch is applied only when tree variants are requested and only if the method
  is absent, so native behavior is preserved when environments are already
  compatible.

## Session: 2026-03-05 — Add amino-acid/diagonal/linear-tree layer-47 diagnostic on Delta

### What changed
Added two new Delta cluster files:
- `scripts/cluster_scripts/diagnostic_amino_acid_encodings_regression.py`
- `scripts/cluster_scripts/diagnostic_amino_acid_encodings_regression.sbatch`

### Why
The residue-weighted diagnostics showed that simple global linear probes can
underperform compared to size-bucketed models. This new diagnostic isolates
five feature/model variants at layer 47, with protein-equal training weights,
to test whether amino-acid identity, diagonal pair features, and learned
size-based piecewise linear models improve per-protein `R^2` without manually
hard-coding size bins.

### Key design decisions
- One deterministic protein-level 80/20 split (`seed=42`) shared by all variants
- One unordered residue pair per row via lower-triangle indexing (`i > j`)
- All variants train with protein-equal sample weighting (`1 / n_pairs` per row)
- Variant set fixed to:
  - baseline 256-dim linear regression
  - baseline + 40-dim amino-acid one-hot augmentation (296 dims total)
  - baseline + diagonal self-pair features (`pb[i,i,:]`, `pb[j,j,:]`, 512 dims)
  - baseline + linear tree split on raw `n_residues`
  - diagonal baseline + linear tree split on raw `n_residues`
- Linear-tree variants enforce a strict small-leaf regime with `max_depth=2`
  (at most 4 leaves) and split only on the appended `n_residues` column
- Outputs are written to a dedicated Delta CSV directory:
  `$HOME/diagnostic_amino_acid_encodings_csv_files`

## Session: 2026-03-04 — Extend residue-weighted diagnostic to paired-feature v2

### What changed
Updated the existing cluster diagnostic:
- `scripts/cluster_scripts/diagnostic_residue_weighted_regression.py`

### Why
The first diagnostic still showed a strong small-vs-large protein calibration
gap, and it was using a simplified 128-dim one-direction feature that was not
directly comparable to the production 256-dim paired regression. This revision
restores the paired feature layout and adds more targeted diagnostics for
protein-length effects.

### Key design decisions
- Restored the production-style paired feature per unordered residue pair:
  `[pb[i,j,:], pb[j,i,:]]` (256 dims), while still using one lower-triangle row
  per unordered pair
- Added `v2` output filenames to avoid overwriting earlier diagnostic outputs
- Added three new diagnostic variants:
  - a `<500` residue pooled baseline
  - a protein-equal baseline with `n_residues`, `|i-j|`, and `log(|i-j|+1)`
    appended as scalar features
  - a bucketed fixed-bin model family (`0-150`, `150-300`, `300-500`, `500+`)
- Expanded split bookkeeping so the script now writes split membership for:
  - the shared global split
  - the `<500` split
  - each bucketed sub-split
- Added explicit calibration metrics to the per-protein outputs:
  - `pred - true` mean/std/variance
  - mean `pred / true` ratio
- Added a new bucket-level bias CSV and summary text section to show whether
  specific size buckets are systematically over- or under-predicted
- Kept the script structure flat and procedural: no new CLI, no new model
  classes, and no separate entrypoint

## Session: 2026-03-04 — Add residue-weighted size-diagnostic cluster regression

### What changed
Added two new Delta cluster files:
- `scripts/cluster_scripts/diagnostic_residue_weighted_regression.py`
- `scripts/cluster_scripts/diagnostic_residue_weighted_regression.sbatch`

### Why
The current pooled multi-protein regression is dominated by large proteins
because pair count scales quadratically with residue count. This diagnostic
tests whether equalizing protein contribution, normalizing distances within each
protein, and conditioning the linear probe on protein size improves
generalization across the full protein set.

### Key design decisions
- All valid proteins are retained, including proteins with more than 500 residues
- Each unordered residue pair is used exactly once via lower-triangle indexing
- Only one directional pair representation is used per pair (`pb[i, j, :]` for `i > j`),
  reducing the base feature width from 256 to 128
- The script uses one deterministic size-stratified 80/20 protein split
- Weighted variants give each training protein equal total loss mass via
  per-pair weights of `1 / n_pairs`
- Size-conditioned variants use `[X, X * g(n)]` interaction models for
  `g(n) in {n, sqrt(n), log(n), n^2}`
- The normalized-distance variant reports both native normalized metrics and
  oracle-rescaled absolute-distance diagnostics
- Outputs are written to a dedicated CSV directory to avoid overwriting
  existing regression runs

## Session: 2026-03-03 — Diagnostic regression script for R² investigation

### What changed
Added two new files for the Delta cluster:
- `scripts/cluster_scripts/diagnostic_regression.py`
- `scripts/cluster_scripts/diagnostic_delta_job.sbatch`

### Why
The full 184-protein run shows pooled R² ~0.53 and mean per-protein R² near 0.
To diagnose this, the new script strips away k-fold CV, MDS, and multi-layer
iteration. It runs only layer 47, excludes proteins >500 residues, randomly
samples 100 proteins 3 times with 80/20 splits, and collects per-protein
diagnostics (distance distribution stats, feature variance, residual analysis).

### Key design decisions
- 3 random samples (seeds 42, 123, 456) to test cross-sample stability
- Per-protein metrics include mean/std/max true distance and mean feature
  variance to correlate with R² failures
- Outlier consistency analysis identifies proteins that fail across multiple
  samples (not just due to random split)
- Outputs go to `$HOME/diagnostic_csv_files` to avoid overwriting production CSVs

## Session: 2026-03-03 — Document core multi-protein regression failure mode

### What changed
Added a new analysis note:
- `regression_problems.md`

### Why
The 184-protein cluster regression results show a large drop in pooled `R^2`
relative to the earlier 10/25-protein runs. The new note captures the main
diagnosis: the current setup uses one global linear model and evaluates it on
all concatenated residue pairs, which implicitly weights proteins by pair count
and hides substantial protein-level failures.

### Key conclusion recorded
- The drop is primarily a conceptual/modeling issue, not an obvious regression
  code bug.
- Small benchmark subsets were optimistic because they were fixed sorted
  subsets, not representative random samples.
- Per-protein performance is much worse than the pooled metric suggests,
  especially for shorter proteins.

## Session: 2026-03-01 — Add neighbor-context pair regression probe

### What changed
Added a new local regression script:
- `scripts/multi_protein/regression/neighbor_context_pair_regression.py`

The script compares three pooled pair-level linear probes across all 48 layers on
a fixed 4-protein subset (`6tf4_A`, `6wmk_A`, `6wqc_A`, `6xqj_A`):
- Baseline: `[pb[i,j,:], pb[j,i,:]]` (256 dims)
- Neighbor-context: baseline + `(i-1,j)`, `(i+1,j)`, `(i,j-1)`, `(i,j+1)` (768 dims)
- Fixed-random control: baseline + four fixed random off-diagonal cells (768 dims)

### Edge handling decision
Edge pairs are retained. If an orthogonal neighbor would fall outside the pair
matrix bounds, that neighbor block is zero-padded instead of dropping the pair.
This keeps the full upper triangle while preserving a fixed feature width.

### Why
This implements the first idea in `llm_generated_research_ideas.md` with a
dimension-matched control, so we can test whether local pair context improves
distance regression beyond what would be expected from simply adding parameters.

### Output path decision
To avoid overwriting existing pooled regression outputs, the new script writes to:
- `proteins/multi_protein_results/neighbor_context_pair_features/csv_files/`
- `proteins/multi_protein_results/neighbor_context_pair_features/visualizations/`

### Additional outputs
The script also exports:
- per-layer pooled R^2 comparisons for all three probes
- deterministic random control cell coordinates
- a unified pooled coefficient CSV with one coefficient vector per layer per model
- layer-47 predicted vs actual distances for all residue pairs
- comparison plots for R^2-by-layer and layer-47 predictions

## Session: 2026-02-25 — Add multi-protein regression with diagonal pair features

### What changed
Added a new local regression script:
- `scripts/multi_protein/regression/multi_protein_regression_with_diagonal.py`

It is based on `multi_protein_regression.py` but changes input features per pair
from 256-dim to 512-dim by appending diagonal self-representations:
- Baseline: `[pb[i,j,:], pb[j,i,:]]` (256)
- New: `[pb[i,j,:], pb[j,i,:], pb[i,i,:], pb[j,j,:]]` (512)

### Why
This enables a direct test of whether residue self-representations (diagonal entries)
improve CA-CA distance regression performance in the pooled multi-protein setting.

### Output path decision
To avoid overwriting baseline pooled outputs, the new script writes to:
- `proteins/multi_protein_results/regression_with_diagonal/csv_files/`
- `proteins/multi_protein_results/regression_with_diagonal/visualizations/`

Per-protein CSV writes remain unchanged (same behavior as baseline script).

## Session: 2026-02-25 — Remove chain_id filtering from all CA extraction functions

### Problem
All `extract_ca_coordinates` functions (and `lib/structure.py`'s `extract_pdb_ca_coordinates`)
were filtering residues by chain ID parsed from the protein_id string (e.g. `"7b3a_A" -> "A"`).
This caused silent failures for proteins whose PDB file stores residues under a chain letter that
doesn't match the protein_id suffix (e.g. a PDB named `7ab3_E_...pdb` but containing chain `A`).

### Fix
Removed `chain_id` parameter and the `if chain.get_id() != chain_id: continue` filter from all
`extract_ca_coordinates` / `extract_pdb_ca_coordinates` functions. The functions now iterate over
all chains in the structure. This is safe because each PDB file is already protein-specific (one
file per protein), so all chains in the file belong to the protein of interest.

### Files changed
- `scripts/cluster_scripts/final_cluster_multi_protein_regression.py`
- `scripts/cluster_scripts/cluster_per_protein_regression.py`
- `scripts/cluster_scripts/test_cluster_multi_protein_regression.py`
- `scripts/multi_protein/regression/regression_variants_predicted.py`
- `scripts/multi_protein/regression/aa_augmented_regression.py`
- `scripts/multi_protein/regression/cluster_multi_protein_regression.py`
- `scripts/multi_protein/regression/mds_reconstruction.py`
- `scripts/multi_protein/regression/multi_protein_regression.py`
- `scripts/multi_protein/regression/train_test_separated_multi_protein_regression.py`
- `scripts/multi_protein/regression/logistic_regression_pair_type.py`
- `scripts/multi_protein/regression/logistic_regression_pair_type_with_diagonal.py`
- `lib/structure.py` (`extract_pdb_ca_coordinates`)

---

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

## Session: 2026-02-18 — lib/ shared library + multi-protein alignment

### Motivation
Preparing for multi-protein regression (`probe_layer_regression_multi.py`). The old `PAIR_OFFSET = 4` hardcode only worked for 7B3A. A general alignment mechanism is needed that reads the mmCIF `_pdbx_poly_seq_scheme` table to map resolved structure residues to pair-block indices for any protein.

### New files

**`lib/__init__.py`** — empty; makes `lib/` importable as a Python package.

**`lib/alignment.py`**
- `AlignmentResult` dataclass: holds `protein_id`, `chain_id`, `pb_indices` (shape `n_modeled`, dtype int32), `n_entity`, `n_modeled`.
- `build_alignment(cif_path, chain_id, protein_id)`: reads `_pdbx_poly_seq_scheme` from the CIF, filters to the requested `asym_id`, converts 1-based `seq_id` to 0-based pair-block indices, excludes unmodeled residues (`auth_mon_id == '?'`). For 7B3A returns `pb_indices = [4..279]` — identical to the old offset=4 pattern.
- `validate_alignment(alignment, first_pb_path, n_ca_atoms)`: raises `AssertionError` if the pair-block dimension, index bounds, monotonicity, or CA count don't match.

**`lib/structure.py`**
Functions migrated verbatim from `scripts/extract_coordinates.py` so multi-protein scripts can import them:
- `extract_residue_coordinates(cif_file, chain_id)` — unchanged logic
- `compute_distance_matrix(coords)` — **vectorized** replacement: `np.linalg.norm(diff, axis=2)` via broadcasting; results are numerically identical to the old triangle loop
- `load_distance_matrix(csv_path)` — `np.genfromtxt` with `skip_header=1`
- `detect_chain(protein_dir)` — unchanged logic

**`scripts/tests/test_alignment.py`** — standalone (no pytest), 5 tests, all pass:
1. `build_alignment` returns `pb_indices == [4..279]`, `n_entity=280`, `n_modeled=276`
2. `validate_alignment` passes without raising
3. Spot-check: `pb_indices[0]==4`, `pb_indices[10]==14`; fancy indexing into pair block returns shape `(128,)`
4. `compute_distance_matrix` returns shape `(276, 276)`, symmetric, zero diagonal
5. `load_distance_matrix` matches computed result within `atol=1e-3`

### Modified files

**`scripts/extract_coordinates.py`** — removed inline definitions of `extract_residue_coordinates`, `compute_distance_matrix`, `detect_chain`. Replaced with `from lib.structure import ...`. `process_protein()` and `main()` unchanged; external behaviour identical. Adds `sys.path.insert(0, project_root)` so `lib/` is importable regardless of cwd.

### Key decisions
- Protein dirs confirmed to follow `proteins/<pdb_id>_A/` convention (not `proteins/<pdb_id>/`). Test paths updated accordingly.
- `compute_distance_matrix` in `lib/structure` uses vectorized broadcasting; the loop-based version is removed from `extract_coordinates.py`. The two are numerically identical.
- `validate_alignment` is designed to hard-fail (raise) rather than warn silently — a wrong alignment is worse than a skipped protein.

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

---

## Session: 2026-02-22 — SLURM resource sizing for 500-protein regression

### Motivation
Scaling `cluster_multi_protein_regression.py` from 4 proteins to ~500. The existing `regression_job.sbatch` requested only 8 GB total memory (1 node, 1 task, 8 GB/cpu), which would OOM for 500 proteins.

### Analysis
The regression script pools ALL upper-triangle residue pairs across all proteins into a single feature matrix `X_all` per layer. For 500 proteins with average sequence length R=200, this produces ~10 million pairs and an `X_all` of ~10.2 GB (10M × 256 features × 4 bytes). With numpy fancy-indexing copies for train/test split and StandardScaler, peak memory reaches ~30–40 GB. Memory scales as R² per protein since pair count = R(R-1)/2.

### Changes

**`regression_job.sbatch`** — Updated resource requests:
- Cores: 1 → 4 (`-N1 --ntasks-per-node=4`) for numpy BLAS threading
- Memory: 8 GB → 64 GB (`--mem-per-cpu=16G`, 4 × 16 = 64 GB total)
- Walltime: 2 hours → 8 hours
- Changed `srun python` to `python` (srun with ntasks=4 would launch 4 copies)
- CPU hours: 4 × 8 = 32, well under the 512 limit

**`check_protein_sizes.py`** — New utility script that surveys protein sizes on the cluster before submitting the regression job. Uses the same `discover_proteins()` / `get_pair_block_path()` directory conventions as the cluster regression script. Memory-maps layer-0 pair blocks to read shapes without loading data. Prints residue length statistics, total pairs, estimated peak memory, and a ready-to-paste sbatch recommendation.

## Session: 2026-02-22 — Per-protein regression script

### Motivation
The existing `cluster_multi_protein_regression.py` trains one pooled model per layer across all ~500 proteins. A new script is needed that trains a **separate regression model per protein per layer** to answer a different question: "How well does each protein's own pair representation predict its own distances?"

### New files

**`per_protein_regression.py`** — Trains 48 independent linear regressions per protein (one per layer), with per-protein 80/20 train/test splits. Same cluster directory layout and helper functions as `cluster_multi_protein_regression.py`.

Key differences from the pooled script:
- Processes one protein at a time (peak memory ~128 MB vs. 30-40 GB for pooled)
- Per-protein train/test split (not global)
- Streams predicted distances CSV to disk after each protein for crash resilience
- Skips proteins with fewer pairs than features (underdetermined regression)
- Produces 3 output CSVs: metrics, coefficients, and last-layer predicted distances

**`per_protein_regression_job.sbatch`** — SLURM job script. Only 4 GB total memory (vs. 64 GB for pooled), 18-hour walltime for ~500 proteins.

## Session: 2026-02-22 — Logistic regression amino acid pair type classifier

### Motivation
The existing regression scripts probe whether pair representations encode **spatial distance**. This new script asks a different question: do pair representations encode **amino acid identity**? Specifically, can we predict the unordered amino acid pair type (e.g., ALA-GLY) from the 256-dim pair embedding?

### New files

**`scripts/multi_protein/regression/logistic_regression_pair_type.py`** — Pooled multinomial logistic regression classifier. One model per layer, trained on all 4 proteins combined (~86,688 pairs, 210 classes). Uses stratified train/test split to handle rare pair types (e.g., TRP-TRP with only 14 samples).

Key design decisions:
- Labels: alphabetically-sorted unordered pairs (e.g., "ALA-GLY"), so (GLU, LYS) and (LYS, GLU) map to the same class
- Features: same 256-dim concatenation `[pb[i,j,:], pb[j,i,:]]` as the distance regression
- Model: `LogisticRegression(C=1e12, solver='lbfgs', max_iter=2000)` — effectively unregularized
- Primary metric: balanced accuracy (accounts for class imbalance)
- StandardScaler fit on training set only

### Outputs (in `proteins/multi_protein_results/logistic_regression/`)
- `csv_files/layer_balanced_accuracy.csv` — balanced + raw accuracy for all 48 layers
- `csv_files/per_class_f1_layer47.csv` — per-class F1/precision/recall for layer 47
- `csv_files/confusion_matrix_layer47.csv` — full 210×210 confusion matrix
- `csv_files/top_confused_pairs_layer47.csv` — top 20 most confused pair types
- `visualizations/layer_balanced_accuracy.png` — accuracy vs layer curve
- `visualizations/confusion_matrix_layer47.png` — row-normalized heatmap
- `visualizations/top_confused_pairs_layer47.png` — horizontal bar chart of top confusions

### Key findings
- **Peak balanced accuracy: 89.8% at layer 15** (raw accuracy 91.9%), far above the 0.48% random baseline for 210 classes
- **Layer 0 already achieves 67% balanced accuracy** — amino acid identity information is present from the start
- **Accuracy declines monotonically after layer 15**, dropping to 9.5% balanced accuracy at layer 47 — the representation progressively loses amino acid identity information in later layers
- **Most confused pairs are biochemically similar**: ILE-LEU ↔ LEU-LEU (hydrophobic aliphatics), GLU-GLU ↔ ASP-GLU (acidic residues), LEU-PHE ↔ LEU-TYR (aromatic substitution)
- **Contrasts with distance regression**: distance R² increases monotonically and peaks at later layers, while amino acid identity peaks early and declines — the network trades identity information for structural/spatial information through its depth

## Session: 2026-02-22 — Protein-level train/test split regression

### Motivation
The existing `cluster_multi_protein_regression.py` pools all residue pairs across proteins and splits them randomly into train/test. This creates data leakage: pairs from the same protein appear in both train and test sets, so the model can exploit protein-specific patterns (e.g., overall fold shape, size) rather than learning generalizable pair-representation-to-distance mappings.

### New files

**`train_test_separated_multi_protein_regression.py`** — Same regression logic as `cluster_multi_protein_regression.py`, but splits at the **protein level** instead of the pair level:
- ~80% of proteins go into training, ~20% into testing
- The model never sees any pairs from test proteins during training
- Uses `sklearn.train_test_split` on protein indices with `random_state=42`
- Requires at least 2 proteins (exits with error if only 1)
- Reports per-protein R² for both test proteins (main metric) and train proteins (diagnostic)

**`protein_split_regression_job.sbatch`** — SLURM job script. Same resource requests as `regression_job.sbatch` (64 GB total, 4 cores, 8h walltime) since memory usage is similar.

### Output files (in ROOT_DIR/csv_files/)
- `protein_split_regression_r2.csv` — per-layer R² with per-protein breakdown
- `protein_split_regression_coefficients.csv` — model coefficients per layer
- `protein_split_predicted_distances.csv` — actual vs predicted (last layer)
- `protein_split_regression_summary.txt` — human-readable summary

### Key difference from pooled script
The pooled script's test R² measures interpolation within known proteins. The protein-level split measures **generalization to unseen proteins** — a much harder and more meaningful test of whether pair representations encode distance in a protein-agnostic way.

## Session: 2026-02-22 — AA-augmented distance regression

### Motivation
The logistic regression classifier showed that amino acid identity is present in early layers but fades by later layers. The distance regression showed R² increases through the layers. A natural question: does **explicitly providing** amino acid identity to the distance regression model improve predictions? If so, the pair representation does not fully capture AA-relevant distance information.

### New files

**`scripts/multi_protein/regression/aa_augmented_regression.py`** — Runs two linear regression models side-by-side per layer:
1. **Baseline** (256-dim): pair-block features only `[pb[i,j,:], pb[j,i,:]]`
2. **Augmented** (296-dim): pair-block features + 40-dim one-hot amino acid encoding

The amino acid encoding works as follows: for each pair (i, j), sort the two 3-letter amino acid names alphabetically, then one-hot encode each into 20 dims (one per standard amino acid). Concatenating gives 40 dims. This ensures ALA-GLY == GLY-ALA automatically.

Non-standard amino acids get all-zero encoding (graceful degradation, no crash).

### Outputs (in `proteins/multi_protein_results/aa_augmented_regression/`)
- `csv_files/layer_regression_r2_comparison.csv` — baseline vs augmented R² per layer
- `csv_files/layer_regression_coefficients_augmented.csv` — augmented model coefficients (296 features)
- `visualizations/layer_r2_comparison.png` — two-line R² vs layer plot
- `visualizations/pred_vs_actual_augmented.png` — scatter plot (layer 47, augmented model)

### Key findings
- **AA features provide a small but consistent improvement** — delta ranges from +0.008 at layer 0 down to +0.0002 at layer 47
- **The improvement shrinks monotonically with layer depth** — early layers gain the most from AA identity (delta ~0.004–0.008), while later layers gain almost nothing (delta < 0.001 after layer 40)
- **Interpretation**: The pair representation progressively encodes the distance-relevant aspects of amino acid identity through its depth. By the final layers, explicitly providing AA identity adds essentially nothing — the pair block already contains all AA-related distance information
- **This is consistent with the logistic regression findings**: AA identity is explicitly readable in early layers, then gets "compiled" into structural features. The distance regression doesn't need the AA hint because the pair block has already internalized it

## Session: 2026-02-22 — MDS reconstruction from predicted distances

### Motivation
R² measures marginal prediction accuracy (averaging over individual pairs) but not **geometric coherence** — whether the predicted distances correspond to a realizable 3D structure. This script uses classical MDS to reconstruct 3D coordinates from regression-predicted distances, then compares to actual PDB structures via TM-score and RMSD.

### New files

**`scripts/multi_protein/regression/mds_reconstruction.py`** — For each of 48 layers:
1. Trains pooled linear regression (same 80/20 split, RANDOM_STATE=42)
2. Predicts ALL pairwise distances per protein (train + test pairs, needed for complete distance matrix)
3. Classical MDS (Torgerson) to embed predicted distances into 3D coordinates
4. Kabsch alignment onto true PDB coordinates via `lib/structure.superimpose_structures`
5. Computes TM-score, RMSD, eigenvalue diagnostics, stress

Classical MDS implementation: eigendecomposition of double-centered Gram matrix, top 3 eigenvalues → 3D coordinates. Uses numpy only (no sklearn MDS). Negative eigenvalues clamped to zero for coordinate extraction; their magnitude is tracked as a diagnostic of non-Euclidean distance predictions.

### Outputs (in `proteins/multi_protein_results/mds_reconstruction/`)
- `csv_files/mds_reconstruction_metrics.csv` — per (layer, protein): R², RMSD, TM-score, top-3 eigenvalue fraction, negative eigenvalue count, stress
- `csv_files/mds_eigenvalue_spectrum.csv` — full eigenvalue spectrum for layers 0, 11, 23, 35, 47
- `csv_files/mds_reconstructed_coords_layer47.csv` — per-residue MDS vs PDB coordinates (last layer)
- `visualizations/mds_tmscores_vs_layer.png` — TM-score vs layer per protein + mean
- `visualizations/mds_rmsd_vs_layer.png` — RMSD vs layer
- `visualizations/mds_eigenvalue_fraction_vs_layer.png` — top-3 eigenvalue variance fraction vs layer
- `visualizations/mds_reconstruction_overlay_layer47.png` — 2×2 3D scatter overlay (PDB vs MDS)
- `visualizations/mds_r2_vs_tmscore.png` — R² vs TM-score scatter

### Key findings
- **TM-scores are surprisingly non-monotonic** — they oscillate significantly across layers even as R² increases steadily. This confirms R² and geometric coherence measure fundamentally different things.
- **7b3a_A achieves the highest TM-scores** (~0.95 at layers 43-46) but crashes to 0.33 at layer 47. The larger proteins show better MDS reconstruction overall.
- **Best mean TM-score is ~0.59 at layers 42-46**, then drops at layer 47 despite R²=0.98.
- **Top-3 eigenvalue fraction reaches only ~0.70 by layer 47** — 30% of variance is in non-3D dimensions, meaning predicted distances are significantly non-Euclidean even at high R².
- **Small proteins (6tf4_A, 6wmk_A, 67-69 residues) have consistently lower TM-scores** (<0.6) than large proteins (6wqc_A, 7b3a_A, 280-294 residues), likely because fewer pairs constrain the MDS embedding less.
- **The R² vs TM-score scatter reveals a protein-dependent relationship**: high R² is necessary but not sufficient for good TM-score. For 7b3a_A, TM-scores reach 0.95 at R²>0.97; for 6wmk_A, TM-scores barely reach 0.25 at the same R² values.

## Session: 2026-02-24 — Delta NCSA cluster sbatch files

### Motivation
Migrating from the ICE cluster to the Delta NCSA cluster. Delta has different Slurm conventions: mandatory `--account` flag, different partition names (`cpu`, `gpuA100x4`, etc.), `module reset` + `module load python` instead of `module load anaconda3`, filesystem constraint annotations (`--constraint="scratch"`), and recommended use of `srun` for launching programs.

### New files
- **`scripts/multi_protein/regression/delta_regression_job.sbatch`** — Delta version of `regression_job.sbatch` (pooled multi-protein regression)
- **`scripts/multi_protein/regression/delta_per_protein_regression_job.sbatch`** — Delta version of `per_protein_regression_job.sbatch` (per-protein regression)

### Key differences from ICE sbatch files
| Aspect | ICE | Delta |
|--------|-----|-------|
| Account | not specified | `--account=<YOUR_ACCOUNT>` (required) |
| Partition | not specified | `--partition=cpu` |
| Module load | `module load anaconda3` | `module reset` + `module load python` |
| Filesystem | no annotation | `--constraint="scratch"` |
| Output files | single `-o` | separate `-o` (stdout) and `-e` (stderr) |
| Launch | `python script.py` | `srun python3 script.py` |
| Task layout | `-N1 --ntasks-per-node=4` | `--nodes=1 --ntasks-per-node=1 --cpus-per-task=4` |

Existing ICE sbatch files left untouched.

## Session: 2026-02-25 — Logistic regression with diagonal features

### Motivation
The existing `logistic_regression_pair_type.py` uses only cross-pair features `[pb[i,j,:], pb[j,i,:]]` (256-dim) to predict amino acid pair types. Adding diagonal self-representation features `[pb[i,i,:], pb[j,j,:]]` gives the model direct information about each residue's own identity, which should improve pair-type classification.

### New files

**`scripts/multi_protein/regression/logistic_regression_pair_type_with_diagonal.py`** — Same structure as `logistic_regression_pair_type.py`, with two key changes:

1. **512-dim features**: For each pair (i, j), concatenates `[pb[i,j,:], pb[j,i,:], pb[i,i,:], pb[j,j,:]]` — cross-pair (256) + diagonal self-representations (256).

2. **C=10 regularization**: Light regularization (vs C=1e12 in original) to help L-BFGS converge faster with doubled feature dimension, with negligible accuracy impact.

### Outputs (in `proteins/multi_protein_results/logistic_regression_with_diagonal/`)
Same output structure as the original logistic regression script: layer balanced accuracy CSV, per-class F1, confusion matrix, top confused pairs, and corresponding visualizations.

## Session: 2026-02-25 — Extract duplicated functions into lib/

### Motivation
Five utility functions were copy-pasted across 6-7 multi-protein regression scripts (each 10-35 lines), totaling ~500+ lines of duplicated code. Extracted them into shared lib/ modules so scripts import instead of redefining.

### New files
**`lib/data_loading.py`** — Three shared data-loading utilities:
- `discover_proteins(proteins_dir)` — scan for protein subdirectories, skip `multi_protein_results`
- `find_pdb_file(protein_dir, protein_id)` — locate predicted PDB file
- `get_pair_block_path(protein_dir, protein_id, layer)` — construct pair block .npy path

### Modified files

**`lib/structure.py`** — Added `extract_ca_coordinates(pdb_path)` using the `het_flag == "W"` filtering approach (preferred over the existing `extract_pdb_ca_coordinates` which uses `is_aa()`). Both functions are kept; the existing `extract_pdb_ca_coordinates` is still used by `structure_comparison.py`.

**6 scripts refactored** (inline functions replaced with lib/ imports):
- `scripts/multi_protein/regression/multi_protein_regression.py`
- `scripts/multi_protein/regression/logistic_regression_pair_type.py`
- `scripts/multi_protein/regression/logistic_regression_pair_type_with_diagonal.py`
- `scripts/multi_protein/regression/aa_augmented_regression.py`
- `scripts/multi_protein/regression/regression_variants_predicted.py`
- `scripts/multi_protein/regression/mds_reconstruction.py`

### Not modified (by design)
- Cluster scripts (`cluster_multi_protein_regression.py`, `train_test_separated_multi_protein_regression.py`) — must remain standalone for cluster execution
- `check_protein_sizes.py` — cluster utility with different directory layout (no `pair_blocks/` subdirectory)
- Functions appearing in only 2 files (`build_pair_labels`, `extract_top_confused_pairs`) — not worth extracting

### Verification
- All 6 refactored scripts pass import checks
- `multi_protein_regression.py` run end-to-end: produces identical R² values (peak 0.9842 at layer 45)
- `mds_reconstruction.py` run end-to-end: produces identical TM-scores and RMSD values

## Session: 2026-02-26 — Test Delta k-fold R² visualizations

### Motivation
Create quick sanity-check plots from the Delta test-run k-fold regression outputs so fold-level behavior and aggregate performance can be inspected without rerunning cluster jobs.

### New files
- **`scripts/multi_protein/regression/plot_test_delta_kfold_r2.py`**
  - Reads:
    - `delta_files/test_linear_regression_csv_files/test_kfold_regression_r2.csv`
    - `delta_files/test_linear_regression_csv_files/test_kfold_regression_r2_summary.csv`
  - Writes to:
    - `proteins/multi_protein_results/test delta files/visualizations/test_kfold_r2_per_fold_and_mean.png`
    - `proteins/multi_protein_results/test delta files/visualizations/test_kfold_r2_mean_std.png`

### Notes
- Output folder name intentionally includes spaces to match request: `test delta files`.
- First plot shows each fold's R²-vs-layer curve with the cross-fold mean overlaid.
- Second plot shows mean R² with ±1 std shading.

## Session: 2026-03-02 — Local benchmark harness for k-fold MDS runtime attribution

### Motivation
The `test2` cluster run is slower than expected, and the next decision depends on separating three effects:
- baseline runtime of the current k-fold MDS regression workflow
- savings from reordering loops to cache one layer of features at a time
- additional savings from replacing stacked full-matrix OLS with exact streaming OLS

To make that comparison explicit without changing the production analysis entrypoints, a standalone local benchmark harness was added.

### New files
- **`scripts/multi_protein/regression/benchmark_kfold_mds_runtime.py`**
  - Runs three modes: `baseline`, `layer_cache`, and `layer_cache_streaming`
  - Reuses the same local protein loading and k-fold split logic across all modes
  - Keeps the MDS + triangle geometry logic unchanged on purpose, so the runtime comparison isolates regression/data-path differences
  - Writes timing breakdowns, mode summaries, and validation CSVs under `proteins/multi_protein_results/kfold_mds_runtime_benchmark/`

### Key design choices
- `baseline` preserves the current fold-first, stacked `StandardScaler + LinearRegression` path.
- `layer_cache` changes only loop order and per-layer feature reuse, so its delta versus `baseline` isolates cache/IO savings.
- `layer_cache_streaming` keeps the same one-layer cache, then replaces stacked OLS with exact chunked normal-equation accumulation, so its delta versus `layer_cache` isolates streaming-solver savings.
- Validation compares regression metrics, coefficients, per-protein predictions, and geometry outputs across modes with strict tolerances for `baseline` vs `layer_cache` and relaxed floating-point tolerances when `layer_cache_streaming` is involved.

### Execution note
This benchmark must be run with the repo venv:
- `./.venv/bin/python scripts/multi_protein/regression/benchmark_kfold_mds_runtime.py ...`

The local system `python3` does not have the required packages.

## Session: 2026-03-02 — `test2` cluster layer-cache refactor with full train/test geometry

### Motivation
The `test2` cluster regression run was spending too much time repeatedly loading the same pair-block data across folds. At the same time, the per-protein geometry export only covered test proteins, but the desired output was to show every protein in every fold/layer, including train proteins.

### Modified files
- **`scripts/cluster_scripts/test2_multi_protein_regression.py`**
  - Reworked the main regression loop from `fold -> layer` to `layer -> fold`
  - Added a one-layer feature cache so each protein's `(layer, feature chunk)` is loaded once and reused across all folds
  - Kept the regression math unchanged: still `StandardScaler + LinearRegression` on stacked train/test matrices
  - Expanded `test2_kfold_mds_metrics.csv` rows to include both train and test proteins
  - Added `split` and `per_protein_r2` to the per-protein MDS metrics rows
  - Runs full MDS / triangle analysis on train and test predictions alike
  - Updated logging and text summary so train/test geometry is explicit

### Notes
- Aggregate regression CSV outputs are intentionally unchanged.
- `test2_kfold_mds_summary.csv` now summarizes all train + test geometry rows because the goal is to show all proteins.
- Running geometry on both splits increases the geometry workload by about 5x per layer, so total runtime is expected to increase materially even though the layer-cache change reduces repeated I/O.

## Session: 2026-03-02 — Symmetric vs antisymmetric pair regression probe

### What changed
Added a new local regression script:
- `scripts/multi_protein/regression/symmetric_antisymmetric_pair_regression.py`

The script compares four pooled pair-level linear probes across all 48 layers on
the same fixed 4-protein subset used by the neighbor-context analysis
(`6tf4_A`, `6wmk_A`, `6wqc_A`, `6xqj_A`):
- Baseline: `[pb[i,j,:], pb[j,i,:]]` (256 dims)
- Symmetric-only: `S = 0.5 * (pb[i,j,:] + pb[j,i,:])` (128 dims)
- Antisymmetric-only: `A = 0.5 * (pb[i,j,:] - pb[j,i,:])` (128 dims)
- Combined decomposition: `[S, A]` (256 dims)

### Validation logic
The script checks the decomposition explicitly for every loaded layer/protein
batch:
- `upper == S + A`
- `lower == S - A`

It also verifies that the feature shapes match the expected 128-dim and 256-dim
layouts before fitting each probe.

### Why
This implements the second idea in `llm_generated_research_ideas.md` in a form
that cleanly separates symmetric and directional signal. Because `[S, A]` is an
invertible linear reparameterization of `[pb[i,j], pb[j,i]]`, the combined probe
should match the baseline up to numerical precision. That built-in equivalence
acts as a sanity check on the feature construction and regression pipeline.

### Output path decision
To avoid overwriting existing pooled regression outputs, the new script writes to:
- `proteins/multi_protein_results/symmetric_antisymmetric_pair_features/csv_files/`
- `proteins/multi_protein_results/symmetric_antisymmetric_pair_features/visualizations/`

### Additional outputs
The script exports:
- per-layer pooled R^2 comparisons for all four probes
- a unified pooled coefficient CSV with one coefficient vector per layer per model
- layer-47 predicted vs actual distances for all residue pairs
- comparison plots for R^2-by-layer and layer-47 predictions

## Session: 2026-03-02 — Test2 Delta post-run geometry analysis

### Motivation
The updated `test2` Delta run now includes train and test geometry rows for every
protein, so the follow-up question is no longer whether the run completed, but
whether the refactor preserved the old results and what the richer per-protein
geometry table says about TM-score, protein size, and train-vs-test behavior.

### New files
- **`scripts/multi_protein/regression/analyze_test2_delta_geometry.py`**
  - Reads:
    - `delta_files/test2_linear_reg_data/`
    - `delta_files/updated_test2_linear_reg_data/`
  - Writes:
    - `proteins/multi_protein_results/test2_updated_delta_analysis/`
    - `analysis_summaries/test2_updated_delta_validation_and_geometry_analysis.md`

### Analysis decisions
- Keep parity checking lightweight: verify the unchanged regression CSVs and that
  the old MDS table matches the updated `split=test` subset, then summarize that
  briefly in the Markdown report instead of creating a standalone parity artifact.
- Plot two versions of per-protein R^2 trajectories:
  - train-only mean across each protein's 4 train rows
  - 5-row mean across each protein's 4 train + 1 test rows
- Add explicit train-minus-test gap plots so averaging does not hide overfitting.
- Use pooled train+test rows for descriptive correlations, but also compute
  within-`(fold, layer, split)` correlations to avoid over-interpreting repeated
  measurements as independent samples.
- Use `TM-score ~ per_protein_r2 + n_residues` as the primary linear model, with
  a sensitivity model that adds `layer` and `split` only as confounder controls.

## Session: 2026-03-02 — Production cluster script upgraded to full `test2` geometry workflow

### What changed
Rewrote the production cluster entrypoint:
- `scripts/cluster_scripts/cluster_multi_protein_regression.py`

The script now matches the `test2` regression + geometry workflow, but runs on
all discovered proteins instead of truncating to a 25-protein subset.

### Key implementation changes
- Reordered the main regression loop from `fold -> layer` to `layer -> fold`
  so each protein's feature chunk for a given layer is loaded once and reused
  across all folds.
- Added inline classical MDS reconstruction, Kabsch superposition, TM-score,
  eigenvalue diagnostics, stress, and triangle-inequality analysis.
- Expanded the per-protein geometry export to include both train and test
  proteins for every fold/layer, with explicit `split` and `per_protein_r2`.
- Kept the regression math unchanged: `StandardScaler + LinearRegression`
  with the same 5-fold protein-level CV setup.

### Output changes
- Added:
  - `kfold_mds_metrics.csv`
  - `kfold_mds_summary.csv`
- Retained:
  - `kfold_predicted_distances.csv` as a backward-compatible extra artifact,
    still limited to the last fold and last layer
- Existing aggregate regression outputs keep the same filenames:
  - `kfold_regression_r2.csv`
  - `kfold_regression_r2_summary.csv`
  - `kfold_regression_coefficients.csv`
  - `kfold_regression_summary.txt`

### Notes
- `kfold_mds_summary.csv` aggregates across all geometry rows (train + test),
  mirroring the updated `test2` behavior.
- Runtime is expected to increase versus the old production script because it
  now runs full geometry analysis on both splits, although the per-layer cache
  reduces repeated pair-block loading.

## Session: 2026-03-05 — Delta per-protein visualization suite (no-CLI)

### What changed
Added a new fixed-configuration analysis script:
- `scripts/multi_protein/regression/analyze_delta_per_protein_visualizations.py`

The script reads Delta per-protein outputs from:
- `delta_files/per_protein_files/per_protein_regression_metrics.csv`
- `delta_files/per_protein_files/per_protein_mds_tm_scores.csv`
- `delta_files/per_protein_files/per_protein_coefficients.csv`

and writes visualizations/summaries under:
- `delta_files/per_protein_files/visualizations/`

### Implemented analyses
- Numeric train-vs-test R^2 gap summaries:
  - `r2_train_vs_test_gap_summary.txt`
  - `r2_train_vs_test_gap_by_layer.csv`
- Test R^2 across layers with line conventions:
  - thin per-protein lines
  - thick black global mean
  - size-bucket means for `small (<100)`, `medium (100-249)`, `large (250-499)`, `very_large (>=500)`
- TM-score across layers with the same line conventions.
- Protein x protein (184 x 184, size-sorted) heatmaps at layers `0,5,10,20,30,40,47` for:
  - cosine similarity
  - Euclidean distance
  - Manhattan distance
  with no more than two heatmaps per image page and fixed per-metric color scales across the selected layers.
- Mean within-protein 48 x 48 layer-relation heatmaps (averaged across proteins) for:
  - cosine similarity
  - Euclidean distance
  - Manhattan distance
- Mean within-protein 48 x 48 cosine similarity by size bucket (2x2 panel).
- Layer-47 TM-score vs test R^2 scatter + correlation CSV:
  - `layer47_tm_vs_r2_scatter.png`
  - `layer47_tm_vs_r2_correlation.csv`

### Validation safeguards added
- Required-column checks for all three input CSVs.
- Strict `(protein_id, layer)` key consistency checks across metrics/TM/coefficients.
- Strict coverage check: `184 proteins x 48 layers`.
- Symmetry/diagonal checks for all pairwise matrices:
  - cosine diagonal approximately 1
  - Euclidean/Manhattan diagonals 0
- Output contract check ensuring all required files are generated, including exactly 4 paged images per pair-grid metric.

### Documentation updates
Added a new **Script Configuration Policy** section to:
- `AGENTS.md`
- `.claude/CLAUDE.md`

Policy summary:
- default to fixed script configuration
- avoid CLI args unless necessary
- if needed, keep CLI minimal and justify in docstring/comments

### Design decisions
- Coefficient vectors for all similarity/distance analyses use `coef_0..coef_255` only (intercept excluded).
- Pair-grid heatmap ordering is by protein size ascending, tie-broken by protein ID.
- Spearman correlation is computed via rank-correlation implementation without adding extra runtime dependencies.

## Session: 2026-03-08 — Linear-tree leaf feature policy correction

### What changed
- Updated `scripts/cluster_scripts/diagnostic_linear_tree_regression.py` so `n_residues` is no longer used as a leaf regression feature.
- Linear-tree fit now uses:
  - `split_features=[split_idx]` (unchanged, still splits on `n_residues`)
  - `linear_features=list(range(split_idx))` (changed, excludes `n_residues` from leaf regressors)
- Updated the module header policy text to reflect this behavior.

### Why
- Restore intended comparison behavior with prior diagnostics where `n_residues` controls partitioning but does not act as a direct regressor inside each leaf.
- Reduce risk of leaf models over-relying on protein-size signals for pair-level prediction.

### Safety check added
- Added an explicit runtime check that the configured split feature resolves to `"n_residues"` before training.

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

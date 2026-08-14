# Gated Linear Distance Probe Suite Plan

## 1. Goal and settled decisions

Build one reproducible analysis suite for asking how linearly decodable true
C-alpha pairwise distance is within sequence, amino-acid, and joint gates. Use
the combined TM-score-filtered CAMEO + ECOD pool, the six requested trunk
layers `(0, 10, 20, 30, 40, 47)`, and a single frozen protein-level split.

The suite has four analyses:

1. Fit linear probes within sequence-separation gates. Evaluate each fitted
   probe overall within its training gate, by amino-acid pair, and separately by
   true spatial-distance bin.
2. Fit linear probes within each of the 210 unordered standard amino-acid pair
   types. Evaluate each fitted probe overall within its training gate, by
   sequence-separation bin, and separately by true spatial-distance bin.
3. Report true-distance distributions once, independent of layer, for every
   category and accepted category cross-product.
4. Fit linear probes within amino-acid-pair x sequence-separation gates.
   Evaluate each fitted probe overall within its training gate and by true
   spatial-distance bin.

Settled modeling and reporting choices:

- Use the same base256 representation as `generalized_model_ecod`: concatenate
  the directional vectors as `z[i,j,:] || z[j,i,:]` for 256 features.
- Use scikit-learn ordinary least squares (`LinearRegression`), not ridge.
- Convert every materialized feature matrix, regression target, scaler, and fit
  to `float64`. Pair-block extraction may use a temporary `float32` cache, as
  in the existing combined model, but OLS must never be fit in `float32`.
- Use all available valid proteins. Do not retain the 100-training-protein cap
  from `diagnostic_sequence_separation_gated_probes.py`.
- Freeze one deterministic, size-stratified 80/20 train/test protein split with
  `RANDOM_STATE=42`, matching `generalized_model_ecod`. There is no validation
  split because there is no hyperparameter selection.
- Give every contributing training protein equal total sample weight within a
  gate, matching the current gated diagnostic and combined-model convention.
- Treat amino-acid pairs as unordered canonical labels such as `ALA-GLY`, while
  preserving directional representation features.
- Exclude noncanonical/unmapped residues from amino-acid-dependent gates and
  report every exclusion explicitly.
- Do not save pair-level predictions or reader-facing per-protein subgroup
  tables. Save the requested aggregate subgroup metrics, exact true-distance
  medians, a manageable test-only per-protein gate table, and compact task-level
  train/test audit arrays sufficient to verify or reconstruct the macro
  calculations.
- Do not bootstrap confidence intervals.
- Use SLURM arrays for every layer-dependent analysis and save progress after
  every completed gate.

## 2. Statistical interpretation

Do not collapse R-squared, MAE, and RMSE into a composite score. They answer
different questions:

- R-squared measures how much within-category target variation is explained.
  Low R-squared for sequence separation 1 is meaningful even if the absolute
  errors are tiny.
- MAE and RMSE measure absolute distance error in Angstroms, but must be read
  relative to how well a constant category mean already performs.
- True-distance mean, median, variance, and standard deviation state how hard
  and how dispersed the category is before considering a probe.

Every fitted gate and evaluation subgroup must therefore report the probe, an
intercept-only baseline, and baseline-relative skill:

```text
mae_skill  = 1 - probe_mae  / intercept_baseline_mae
rmse_skill = 1 - probe_rmse / intercept_baseline_rmse
```

Positive skill means that the representation probe improves over the training-
derived gate mean; zero means no improvement; negative values mean it is worse.
If a baseline denominator is zero, save the corresponding skill as `nan` and
an explicit reason/status.

Compute pooled skill from the pooled probe and baseline errors. Compute macro
skill as `1 - mean_per_protein_probe_error /
mean_per_protein_baseline_error`; do not average per-protein error ratios, which
would give unstable cells with nearly zero baseline error excessive influence.
The aggregate macro field follows the ratio-of-macro-errors definition above.

For every metric family, retain both weightings rather than choosing one:

- `pooled_*`: every residue pair has equal evaluation weight across the split.
- `mean_per_protein_*`: every protein with a defined metric has equal evaluation
  weight.

For R-squared, report `n_proteins_with_finite_r2`. A protein/category R-squared
is defined only when it has at least two pairs and target variance greater than
`FLOAT_EPS = 1e-12`. MAE and RMSE remain defined for a single pair.

## 3. Category definitions

### 3.1 Sequence-separation gates

Use the existing half-closed integer gates:

| Label | Included sequence separations |
|---|---|
| `1` | exactly 1 |
| `2-4` | 2 through 4 |
| `5-9` | 5 through 9 |
| `10-19` | 10 through 19 |
| `20-49` | 20 through 49 |
| `50+` | 50 or greater |

Sequence separation is `abs(i - j)` in the pair-block/PDB-aligned residue
index. Validate that analyzed proteins are single-chain or that chain identity
is aligned. If an interchain pair is encountered, exclude it from sequence-
separation gates and record `interchain_sequence_separation_undefined` rather
than treating concatenated array indices as biological sequence separation.

### 3.2 Spatial-distance evaluation bins

Use true C-alpha distance only after fitting/prediction, with half-open bins:

| Label | Included true distance (Angstroms) |
|---|---|
| `0-5` | `[0, 5)` |
| `5-10` | `[5, 10)` |
| `10-20` | `[10, 20)` |
| `20-40` | `[20, 40)` |
| `40+` | `[40, infinity)` |

True spatial distance must never select training rows for a fitted model. It is
an evaluation/reporting subgroup only.

### 3.3 Amino-acid pair gates

Use the 20 standard three-letter amino-acid names from
`cluster_regression_helpers.STANDARD_AAS`. Sort the two identities
alphabetically and enumerate combinations with replacement, yielding exactly
210 labels. Centralize this enumeration and pair-code lookup in the shared
module; do not copy separate label lists into the runners.

## 4. Data, target, and frozen split preparation

Create a one-time preparation/statistics entrypoint:

```text
scripts/cluster_scripts/gated_linear_probes/prepare_gated_linear_probe_suite.py
scripts/cluster_scripts/gated_linear_probes/prepare_gated_linear_probe_suite.sbatch
```

It should reuse the combined-dataset path resolution and stable helpers from
`general_train_test_common.py` and `cluster_regression_helpers.py` rather than
creating a third dataset manifest.

Preparation responsibilities:

1. Load the 554 manifest proteins and resolve source-specific CAMEO/ECOD PDB
   and pair-block paths.
2. Verify PDB C-alpha count equals pair-block dimension, residue indexing is
   aligned, all six requested layer files exist, and true distances are finite.
3. Assign the same five rank-based protein-size bins as
   `generalized_model_ecod`.
4. Reuse an existing compatible `general_base256_combined/split_assignments.csv`
   if its protein set, size bins, random state, and path checks match. Otherwise
   reproduce the deterministic 80/20 split once. Save and hash the result.
5. For each valid protein, atomically save a compressed metadata cache containing
   `y_true`, raw sequence separation, sequence-bin code, AA-pair code,
   distance-bin code, and canonical-validity mask. Keep pair order identical to
   `np.tril_indices(R, k=-1)`.
6. Produce Analysis 3 from these caches before any layer-dependent job begins.
7. Produce intercept-only baselines for every Analysis 1, 2, and 4 training
   gate. The constant is the equal-protein-weighted mean of the training-
   protein gate means, matching the OLS sample-weight policy.
8. Fit the raw-sequence-separation baseline for Analysis 1 once per sequence
   gate. This baseline is layer-independent and must not be refit by each layer
   task. Use raw sequence separation as its only feature and the same train/test
   proteins and protein-equal sample weights.

The suite split is fixed before excluding individual noncanonical AA pairs. A
protein does not move between train and test because a particular gate is
missing or under-supported.

## 5. Shared implementation module

Create:

```text
scripts/cluster_scripts/gated_linear_probes/gated_linear_probe_common.py
```

This module should own:

- layer, sequence-bin, spatial-bin, and AA-pair constants;
- manifest and frozen-split loading plus digest validation;
- prepared per-protein metadata loading;
- direct `np.load(...)` pair-block extraction and preallocated directional
  base256 feature writes;
- gate masks and deterministic gate ordering;
- train-only protein-equal scaler computation;
- float64 materialization and protein-equal sample weights;
- intercept-only baseline lookup;
- OLS eligibility checks, fitting, rank/conditioning diagnostics, coefficient
  conversion, and prediction;
- pooled additive moments, exact true-distance median calculation, per-protein
  subgroup metrics, and macro aggregation;
- atomic CSV, compressed NPZ, JSON, and text output helpers;
- SLURM task-to-layer/gate mapping;
- timestamped progress logging and checkpoint manifests.

Do not import one thin analysis runner from another. All reusable logic belongs
in this common module.

## 6. OLS eligibility and numerical diagnostics

OLS is appropriate for dense, well-supported gates and matches the existing
combined model. It is not automatically reliable for every Analysis 4 gate.
Apply this policy uniformly:

1. A gate is eligible to fit only when it has at least 257 training pairs (more
   rows than the 256 representation features) and at least five contributing
   training proteins.
2. Ineligible gates are not fit. Emit their expected rows with status
   `insufficient_train_rows` or `insufficient_train_proteins` and preserve all
   support and true-distance fields.
3. For eligible gates, fit `sklearn.linear_model.LinearRegression` on the
   standardized float64 matrix with protein-equal row weights.
4. Save `design_rank`, largest and smallest singular values, and
   `condition_number`. Use the singular values exposed by the dense fit where
   available; otherwise compute the same weighted-design diagnostic explicitly.
5. If `design_rank < 256`, keep the predictions but mark the gate
   `fitted_rank_deficient`; do not present it as a primary well-supported result.
6. If the condition number is nonfinite or greater than `1e12`, mark the gate
   `fitted_ill_conditioned`. Retain metrics for diagnosis but flag them in every
   consolidated table.
7. Mark an evaluation cell `sparse_test_support` when it has fewer than 100
   pairs or fewer than five contributing test proteins. Retain its metrics and
   counts rather than silently removing it.
8. A cell can carry multiple semicolon-separated flags while preserving one
   primary fit status.

This policy keeps OLS as requested while preventing minimum-norm solutions for
underdetermined gates from looking equivalent to stable fits.

## 7. Analysis 1: sequence-separation-gated probes

Create:

```text
scripts/cluster_scripts/gated_linear_probes/sequence_gated_linear_probes.py
scripts/cluster_scripts/gated_linear_probes/sequence_gated_linear_probes.sbatch
```

One SLURM array task handles one requested layer. Within a task:

1. Load all valid train and test proteins for the selected layer.
2. For each of the six sequence gates, fit one base256 OLS probe.
3. Evaluate train and test predictions overall within the fitted sequence gate.
4. Evaluate train and test predictions by unordered AA pair, producing
   `layer x sequence gate x AA pair` cells.
5. Separately evaluate train and test predictions by spatial-distance bin, producing
   `layer x sequence gate x spatial-distance` cells.
6. Compare every cell with the precomputed intercept-only baseline prediction.
7. Write outputs after every sequence gate so a timeout loses at most one gate.

Do not compute the full AA-pair x spatial-distance cross-product in Analysis 1.
Do not refit the raw-sequence baseline in layer tasks; use the one-time baseline
outputs from preparation.

## 8. Analysis 2: amino-acid-pair-gated probes

Create:

```text
scripts/cluster_scripts/gated_linear_probes/aa_pair_gated_linear_probes.py
scripts/cluster_scripts/gated_linear_probes/aa_pair_gated_linear_probes.sbatch
```

One SLURM array task handles one requested layer. Within a task:

1. Load all valid train and test proteins for the selected layer.
2. Iterate through the 210 canonical AA-pair gates in fixed order.
3. Fit one base256 OLS probe for every eligible gate.
4. Evaluate train and test predictions overall within the fitted AA gate.
5. Evaluate train and test predictions by sequence-separation bin, producing
   `layer x AA pair x sequence separation` cells.
6. Separately evaluate train and test predictions by spatial-distance bin, producing
   `layer x AA pair x spatial distance` cells.
7. Compare every cell with its precomputed AA-gate intercept baseline.
8. Checkpoint after every AA gate.

Do not add a raw-sequence-separation baseline to Analysis 2. Do not compute the
full sequence-separation x spatial-distance cross-product in this analysis.

Save Analysis 2 coefficients in a normal `coefficients.csv`; its roughly
323,820 coefficient/intercept rows are manageable and easier to inspect than
NPZ shards.

## 9. Analysis 3: true-distance category statistics

Analysis 3 runs once during preparation and does not depend on representation
layer. Compute exact statistics for splits `all`, `train`, and `test` at these
scopes:

1. overall;
2. sequence separation;
3. AA pair;
4. spatial distance;
5. AA pair x sequence separation;
6. sequence separation x spatial distance;
7. AA pair x spatial distance;
8. AA pair x sequence separation x spatial distance.

For each nonempty category cell, compute pair-pooled `n_pairs`, mean, exact
median, and population variance (`ddof=0`). Also compute the number of proteins,
standard deviation, the mean per-protein true distance, median per-protein true
distance, and mean per-protein target variance. These are the quantities needed
to interpret the probe errors; omit minimum, maximum, quartiles, and IQR. Save
noncanonical/interchain exclusions separately rather than folding them into an
`unknown` AA or sequence category.

## 10. Analysis 4: AA-pair x sequence-separation-gated probes

Create:

```text
scripts/cluster_scripts/gated_linear_probes/aa_sequence_gated_linear_probes.py
scripts/cluster_scripts/gated_linear_probes/aa_sequence_gated_linear_probes.sbatch
```

Use 36 array tasks: six requested layers x six sequence gates. Each task maps
deterministically to one `(layer, sequence gate)` and iterates through all 210
AA pairs. Within a task:

1. Extract only pairs in the task's sequence gate.
2. Fit one base256 OLS probe per eligible AA-pair x sequence gate.
3. Evaluate train and test predictions overall within the fitted joint gate.
4. Evaluate train and test predictions by spatial-distance bin, producing the
   full requested `layer x AA pair x sequence separation x spatial distance`
   cells.
5. Compare each cell with its precomputed joint-gate intercept baseline.
6. Checkpoint after every AA pair.

Do not add a raw-sequence-separation baseline. Save the approximately 1.94
million expected coefficient values as compressed NPZ shards plus a CSV index.

## 11. Metric definitions and common schemas

### 11.1 Aggregate metric fields

Reader-facing gate-level and subgroup-level rows should contain only the fields
needed to interpret or filter the scientific results. Identifiers that do not
apply use the literal `all`, not an empty string:

```text
analysis_name, model, feature_dim, layer, split,
train_gate_aa_pair, train_gate_sequence_bin,
evaluation_scope, eval_aa_pair, eval_sequence_bin, eval_distance_bin,
fit_status, support_flags,
n_train_pairs, n_train_proteins, n_pairs, n_proteins,
n_proteins_with_finite_r2,
pooled_mean_true_distance, pooled_median_true_distance,
pooled_target_variance, pooled_target_std,
pooled_r2, pooled_mae, pooled_rmse,
mean_per_protein_r2, mean_per_protein_mae, mean_per_protein_rmse,
intercept_baseline_prediction,
baseline_pooled_r2, baseline_pooled_mae, baseline_pooled_rmse,
baseline_mean_per_protein_r2, baseline_mean_per_protein_mae,
baseline_mean_per_protein_rmse,
pooled_mae_skill, pooled_rmse_skill,
mean_per_protein_mae_skill, mean_per_protein_rmse_skill
```

`evaluation_scope` is one of `gate_overall`, `aa_pair`,
`sequence_separation`, or `spatial_distance`. Pooled R-squared always uses the
mean of the current evaluation cell in its denominator; it is never an average
of lower-level R-squared values. Bias, median per-protein performance, singular
values, and elapsed time are not repeated in every scientific subgroup row.
Baseline R-squared is retained because a constant learned from the training
gate can expose train/test or gate/subgroup mean shifts.

### 11.2 Gate-status fields

Keep numerical and operational details once per fitted training gate in
`gate_status.csv`:

```text
analysis_name, model, feature_dim, layer,
train_gate_aa_pair, train_gate_sequence_bin,
fit_status, support_flags, n_train_pairs, n_train_proteins,
design_rank, largest_singular_value, smallest_singular_value,
condition_number, elapsed_seconds
```

This avoids duplicating the same OLS rank and conditioning information across
thousands of subgroup rows. For example, the layer-20 `ALA-GLY` probe in
Analysis 2 has one design rank and one condition number; those values do not
change when its predictions are summarized by six sequence bins or five
spatial-distance bins. Each subgroup row retains the gate key and `fit_status`,
so it can be joined to the single detailed gate-status row when needed.

### 11.3 Compact per-protein and audit artifacts

Retain a reader-facing `test_per_protein_gate_metrics.csv.gz`, one row per test
protein x primary fitted gate, with only:

```text
analysis_name, layer, protein_id, source, n_residues, size_bin,
train_gate_aa_pair, train_gate_sequence_bin, fit_status, n_pairs,
target_variance, per_protein_r2, per_protein_mae, per_protein_rmse,
mean_true_distance, mean_predicted_distance
```

Do not publish train per-protein tables or per-protein subgroup CSVs. During a
task, compute all macro subgroup metrics from per-protein values and retain a
compact compressed NPZ audit shard for both train and test containing integer
cell codes plus `n`, `sum_y`, `sum_y_squared`, `sum_prediction`,
`sum_absolute_error`, and `sum_squared_error` per protein-cell. These arrays are
enough to reproduce the reported train/test macro and pooled values without
retaining pair predictions or a very large human-facing table. Keep audit
shards under the cluster task directory; do not copy them into the consolidated
repository results by default.

### 11.4 Coefficient fields

Analysis 1 may use a normal CSV with one row per fitted gate and feature:

```text
analysis_name, layer, train_gate_sequence_bin, feature_index, feature_name,
coefficient_standardized, coefficient_raw_scale
```

Include an intercept row (`feature_index=-1`) with standardized and raw-scale
intercepts so predictions can be replayed directly from raw pair features.

Analysis 2 also uses a normal `coefficients.csv` with the same gate identifiers
plus AA pair. Its expected size is manageable and CSV is easier to inspect.

Analysis 4 uses one NPZ per task containing ordered gate labels, standardized
and raw-scale coefficient matrices, standardized and raw-scale intercepts,
scaler means/scales, and fit statuses. The accompanying coefficient index CSV
maps each gate to its NPZ row and records dimensions and checksums. Rank and
singular-value summaries remain in `gate_status.csv` rather than being repeated
inside reader-facing coefficient tables.

### 11.5 Recoverability contract

The streamlined reader-facing tables must not prevent later detailed analysis:

- Per-protein train/test audit arrays reconstruct pooled, mean per-protein, and
  median per-protein R-squared, MAE, RMSE, bias, true mean, predicted mean, and
  variance for every reported subgroup.
- Prepared metadata caches reconstruct true-distance minimum, maximum,
  quartiles, IQR, or alternative category binning without loading pair
  representations.
- Raw-scale coefficients/intercepts, split assignments, gate definitions, and
  the original pair blocks reproduce pair-level predictions without refitting
  a probe. Analysis 4's NPZ must retain the equivalent raw-scale information.
- Pair-level prediction/error quantiles are not immediately available because
  pair predictions are not saved; recovering them requires a deterministic
  coefficient replay against the pair blocks.

Thus additive metrics and target-distribution statistics are exactly
recoverable from compact artifacts. New pair-level error analyses require
replay, but not retraining.

## 12. Exact output organization

Cluster task outputs go under:

```text
$CSV_ROOT/gated_linear_probes/
```

Consolidated repository outputs go under:

```text
delta_files/gated_linear_probes/
```

The root files are:

- `run_manifest.json`: versions, layers, bins, feature mode, precision, OLS
  policy, eligibility thresholds, split digest, input roots, and timestamps.
- `split_assignments.csv`: one row per valid protein with source, residue/pair
  counts, size bin, and train/test split.
- `skipped_proteins.csv`: one row per input exclusion with stage and reason.
- `excluded_pairs_summary.csv`: one row per protein/split/exclusion reason with
  counts for noncanonical, interchain, or otherwise invalid gated pairs.
- `$CSV_ROOT/gated_linear_probes/metadata/per_protein/<protein_id>.npz`:
  prepared pair metadata cache used by cluster tasks. Keep this working cache on
  the cluster; do not copy it into `delta_files/gated_linear_probes/`.

### Analysis 1 directory

```text
analysis_1_sequence_gated/
```

- `gate_summary.csv`: one row per layer x sequence gate x split; retains train
  and test metrics for overfitting checks.
- `subgroup_by_amino_acid_pair.csv`: one row per layer x sequence gate x AA
  pair x split.
- `subgroup_by_spatial_distance.csv`: one row per layer x sequence gate x
  distance bin x split.
- `test_per_protein_gate_metrics.csv.gz`: one row per test protein x layer x
  sequence gate; no subgroup expansion.
- `coefficients.csv`: one row per layer x sequence gate x intercept/feature.
- `raw_sequence_baseline_summary.csv`: one row per sequence gate x split; no
  layer duplication.
- `gate_status.csv` and `summary.md`.
- `task_shards/`: layer-task CSV/JSON checkpoints, compact train/test audit
  NPZs, and task summaries; these remain cluster-side unless needed for
  debugging.

### Analysis 2 directory

```text
analysis_2_aa_pair_gated/
```

- `gate_summary.csv`: one row per layer x AA gate x split.
- `subgroup_by_sequence_separation.csv`: one row per layer x AA gate x sequence
  bin x split.
- `subgroup_by_spatial_distance.csv`: one row per layer x AA gate x distance
  bin x split.
- `test_per_protein_gate_metrics.csv.gz`: one row per test protein x layer x AA
  gate.
- `coefficients.csv`: one row per fitted layer x AA gate x intercept/feature.
- `gate_status.csv` and `summary.md`.
- `task_shards/`: layer-task checkpoints, compact train/test audit NPZs, and task
  summaries retained cluster-side.

### Analysis 3 directory

```text
analysis_3_true_distance_statistics/
```

- `true_distance_category_stats.csv`: one row per split x category scope x
  category cell.
- `category_support.csv`: expected/nonempty/sparse category counts by scope and
  split.
- `summary.md`: readable overall, sequence-bin, spatial-bin, and coverage
  tables, with links/names for the larger AA cross-product CSVs.

Per-protein values are computed transiently to form the macro fields but are
not published as another large category table. They can be recomputed later
from the retained cluster metadata caches if a protein-level question arises.

### Analysis 4 directory

```text
analysis_4_aa_sequence_gated/
```

- `gate_summary.csv`: one row per layer x AA gate x sequence gate x split.
- `subgroup_by_spatial_distance.csv`: one row per layer x AA gate x sequence
  gate x distance bin x split.
- `test_per_protein_gate_metrics.csv.gz`: one row per test protein x layer x AA
  gate x sequence gate.
- `coefficient_index.csv` and `coefficient_shards/*.npz`.
- `gate_status.csv` and `summary.md`.
- `task_shards/`: layer x sequence-gate checkpoints, compact train/test audit
  NPZs, and summaries retained cluster-side.

The consolidated CSVs are the reader-facing authoritative results. Cluster
metadata and train/test audit NPZs are reproducibility artifacts, not additional
tables to inspect routinely. Do not add a visualization suite until coverage
and numerical diagnostics show which cells are scientifically interpretable.

## 13. Aggregation entrypoint

Create:

```text
scripts/cluster_scripts/gated_linear_probes/aggregate_gated_linear_probe_suite.py
scripts/cluster_scripts/gated_linear_probes/aggregate_gated_linear_probe_suite.sbatch
```

Aggregation must:

1. Verify the complete expected task set and reject mixed split/configuration
   digests.
2. Concatenate only completed atomic gate shards in deterministic order.
3. Verify no duplicate identifier key exists in any final table.
4. Recompute train/test pooled and macro metrics from the compact audit NPZ
   arrays and compare them with task metrics using `1e-8` absolute and `1e-9`
   relative tolerances before publishing.
5. Never average already-pooled values or average R-squared across lower-level
   bins.
6. Verify all expected gates have a row, including skipped/ineligible gates.
7. Copy or materialize the consolidated tree under
   `delta_files/gated_linear_probes/`.
8. Write concise Markdown summaries emphasizing held-out test results, OLS
   coverage, rank-deficient/ill-conditioned counts, and baseline-relative
   skill. Training subgroup rows remain available for overfitting checks, while
   the Markdown report remains test-focused.

## 14. SLURM layout and resources

All new sbatch files must include:

```text
#SBATCH --account=bekh-delta-cpu
#SBATCH --partition=cpu
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu
```

They must activate `~/protein_venv`, set BLAS thread variables from
`SLURM_CPUS_PER_TASK`, use `%A_%a` logs for arrays, and launch with:

```bash
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/<entrypoint>.py"
```

Initial resource requests, to be refined from `sacct` elapsed time and `MaxRSS`
after one representative task from each family:

| Job | Array mapping | CPU | Memory | Walltime | Concurrency |
|---|---|---:|---:|---:|---:|
| preparation + Analysis 3 | single job | 4 | 32 GB | 8 h | 1 |
| Analysis 1 | `0-5` -> requested layer | 8 | 192 GB | 24 h | 6 |
| Analysis 2 | `0-5` -> requested layer | 8 | 128 GB | 24 h | 6 |
| Analysis 4 | `0-35` -> layer x sequence gate | 8 | 96 GB | 24 h | `%6` |
| aggregation | single dependency job | 4 | 32 GB | 8 h | 1 |

Analysis 1 retains the largest request because the broad `50+` gate may require
the largest dense float64 materialization. Analysis 2 and 4 materialize one AA
gate at a time and should require less memory. Reduce requests after observed
usage; do not raise them preemptively without evidence.

Every runner must use direct `np.load(...)`, direct fancy-index extraction into
preallocated arrays, and bounded per-gate materialization. Avoid
`mmap_mode="r"` for hot fancy-index paths and avoid repeated `np.concatenate`.

Add a submission helper:

```text
scripts/cluster_scripts/gated_linear_probes/submit_gated_linear_probe_suite.sh
```

It should submit preparation first; submit the three probe arrays with
`afterok` on preparation; and submit aggregation with `afterok` on all three
arrays. Echo all job IDs and the expected output root. It must not hide failed
dependencies or automatically resubmit failed tasks.

## 15. Checkpointing, logs, and reruns

- Write a task manifest before computation with task ID, layer/gate mapping,
  split/configuration hashes, and expected gate keys.
- After each gate, atomically rewrite/append deterministic shard files and
  update `completed_gate_keys` in the task manifest.
- On rerun, skip a completed gate only if its recorded split/configuration
  hashes and file checksums match. Otherwise fail clearly instead of mixing
  outputs.
- Log timestamps for metadata loading, each 25 proteins loaded, feature-cache
  construction, scaler computation, each OLS fit, each evaluation scope, each
  checkpoint, and total task time.
- Log current and peak resident memory when available.
- Keep all completed gates if a later gate times out or fails.

## 16. Tests and validation

Add focused tests under:

```text
scripts/cluster_scripts/gated_linear_probes/tests/
```

Required unit tests:

1. Enumerating standard unordered AA pairs produces exactly 210 stable labels.
2. Sequence and spatial boundary values map to the intended half-open bins.
3. Noncanonical and interchain pairs receive explicit exclusion codes.
4. Array task IDs map exactly to layers and Analysis 4 joint tasks.
5. Protein split assignments are disjoint, deterministic, size-stratified, and
   identical across all four analyses.
6. The base256 extractor equals explicit `z[i,j] || z[j,i]` concatenation.
7. Float64 OLS on a synthetic full-rank gate recovers known predictions.
8. A gate with fewer than 257 rows is not fit and receives the expected status.
9. Rank-deficient and ill-conditioned synthetic gates receive the correct
   diagnostic flags.
10. Protein-equal weights give every contributing protein equal total mass.
11. Pooled metrics reconstructed from moments match direct calculations.
12. Macro metrics reconstructed from compact per-protein audit arrays match
    direct calculations.
13. Constant/near-constant targets yield undefined R-squared but finite
    MAE/RMSE without crashing.
14. Intercept baseline skill calculations handle zero denominators explicitly.
15. Analysis 1's raw sequence baseline is layer-independent and appears once.
16. Analysis 1 and 2 do not accidentally emit the unrequested full four-way
    cross-product; Analysis 4 does.
17. Analysis 2 coefficient CSV rows are complete and unique; Analysis 4
    coefficient NPZ rows round-trip through their CSV index and checksums.
18. Aggregation rejects missing tasks, duplicate keys, and mixed split hashes.

Before full submission, run a local/small synthetic smoke test for all four
analyses and a Delta smoke test on one layer and a small deterministic protein
subset. The smoke subset is for validation only and must use a distinct output
root so it cannot be aggregated with the final run.

After creating or modifying every sbatch file, scan every sbatch entrypoint and
verify that the referenced Python filename exists. Run `bash -n` on shell and
sbatch files, compile Python files, and run the focused test suite with
`./.venv/bin/python`.

## 17. Implementation order

1. Implement shared constants, frozen-split preparation, metadata caches, and
   Analysis 3; validate category counts and split compatibility.
2. Implement common OLS/scaling/baseline/metric/checkpoint helpers and their
   synthetic tests.
3. Implement Analysis 1 and compare its all-protein OLS outputs against the
   existing gated diagnostic on a controlled subset.
4. Implement Analysis 2, its normal coefficient CSV, and sparse-gate coverage
   reporting.
5. Implement Analysis 4 using the 36-task layer x sequence-gate mapping.
6. Implement deterministic aggregation and Markdown summaries.
7. Add sbatch files and dependency submission helper, then run local and Delta
   smoke tests.
8. Submit one representative final task from each family, inspect runtime and
   `MaxRSS`, adjust resources downward where appropriate, and then submit the
   full arrays.
9. Aggregate only after all expected tasks complete and review test-set support,
   rank, conditioning, and baseline-skill coverage before interpreting layer
   trends.

## 18. Interpretation guardrails

- A low R-squared and low MAE can both be correct when target variance is low;
  show target variance and baseline-relative skill beside both.
- Spatial-distance subgroup results describe conditional error by the true
  target range; they are not evidence for a deployable distance-gated model.
- Do not compare raw R-squared values across cells without their target
  variance, pair count, protein count, and numerical-status fields.
- Do not treat rank-deficient or ill-conditioned Analysis 4 gates as stable
  coefficient estimates. Their predictions may be diagnostic, but the status
  must travel with every result.
- Pooled metrics can be dominated by large proteins; macro metrics can be
  noisy for sparse protein/category cells. Report both and their support.
- The target is distance in the saved predicted PDB, so results measure what is
  linearly accessible about OpenFold's eventual predicted geometry, not an
  independent experimental structure unless the target source changes.

# Canonical Outputs for Distance-Regression Probes

## Scope and design principle

This document defines the output contract for the paper's shared OLS,
per-protein OLS, gated OLS, shared XGBoost, and future distance-regression
variants. It complements `regression_methodology.md`: that document specifies
how models are fitted, whereas this document specifies what every completed run
must save.

The current additive output-schema version is **2.0**, paired with fivefold OOF
protocol 2.0. Existing schema-1.0 80/20 files are immutable historical outputs;
their helpers retain the v1 identifier and remain operational.

The output design has three goals:

1. make every model directly comparable at the protein, layer, and dataset-split
   levels;
2. retain enough information for publication tables and numerical auditing;
   and
3. avoid storing pair-level predictions or repeatedly saving statistics that
   can be reconstructed exactly from smaller authoritative files.

Outputs are divided into three tiers:

- **Core:** compact, authoritative outputs produced for every regression.
- **Derived:** small summaries that are convenient but reproducible from core
  outputs.
- **Supplementary:** compressed detail retained for plausible later analyses
  but not intended as a primary paper table.

## Recommended result-family layout

Each final regression variant should use the following layout. Search-only or
diagnostic runs may live elsewhere, but they must not be confused with these
final-model outputs.

```text
regression_5fold_v1/
  metadata/
    regression_outer_folds_v1.csv
    per_protein_pair_folds_v1.csv
    fold_digests.json
  <result_family>/
    run_manifest.json
    per_protein.csv
    pooled_metrics.csv
    fold_metrics.csv
    fold_summary.csv
    training_metrics.csv
    stable_fit_metrics.csv
    model_status.csv
    runtime.csv
    geometry_metrics.csv
    geometry_summary.csv
    model_stability.csv
    subgroup_metrics/
      by_sequence_separation.csv
      by_spatial_distance.csv
    model_artifacts/
      replay_index.csv
      coefficients or coefficient_shards/
      boosters/
      feature_importance.csv
    layer_shards/
```

The four family directories are `shared_ols_base256/`,
`per_protein_ols_base256/`, `sequence_gated_ols_base256/`, and
`shared_xgboost_base256_fixed_s4/`. Only applicable artifacts are required.
The run manifest identifies schema version 2.0, CV protocol version 2.0, and
the shared cohort and fold digests rather than copying split definitions.

## Regression metric definitions

For an evaluated set of residue pairs with true distances $y_k$, predictions
$\hat y_k$, and $n$ evaluated pairs, define:

\[
R^2 = 1 - \frac{\sum_k (y_k-\hat y_k)^2}
{\sum_k (y_k-\bar y)^2},
\]

\[
\mathrm{MAE} = \frac{1}{n}\sum_k |y_k-\hat y_k|,
\qquad
\mathrm{RMSE} = \sqrt{\frac{1}{n}\sum_k (y_k-\hat y_k)^2},
\]

and

\[
\mathrm{bias} = \frac{1}{n}\sum_k(\hat y_k-y_k).
\]

MAE, RMSE, and bias are reported in angstroms. $R^2$ is undefined when the
target variance is numerically zero and must then be recorded as `NaN`, with the
row retained. Negative $R^2$ is valid and must not be clipped.

The core regression metrics are $R^2$, MAE, RMSE, and bias:

- $R^2$ measures variance explained but can be unstable in narrow categories
  with very little target variation;
- MAE is the most directly interpretable typical distance error;
- RMSE gives additional weight to large errors; and
- bias detects systematic over- or underprediction.

MSE is omitted because it is exactly RMSE squared. Pearson and Spearman
correlations are not core metrics because they do not measure calibration and
would add little beyond $R^2$, error metrics, and the category analyses.

## Core `per_protein.csv`

Every regression produces one authoritative OOF `per_protein.csv` with exactly
one row per analysis, layer, and eligible protein: 26,496 rows for a complete
family. Shared families evaluate all pairs withheld by that protein's outer
fold. Per-protein OLS evaluates the complete concatenated five-pair-fold OOF
vector. Training rows are not included here.

### Identity and scope columns

| Column | Meaning |
|---|---|
| `analysis_name` | Stable unique variant name |
| `model_family` | `shared_ols`, `per_protein_ols`, `gated_ols`, or `shared_xgboost` |
| `feature_dim` | Number of model input features |
| `layer` | Pair-representation layer |
| `protein_id` | Canonical protein identifier |
| `source` | Input-layout source, `cameo` or `ecod` |
| `cv_protocol_version` | Fivefold protocol version, `2.0` |
| `evaluation_design` | `protein_outer_5fold` or `within_protein_pair_5fold` |
| `outer_fold` | Protein fold 0 through 4; reporting-only for per-protein OLS |
| `evaluation_scope` | `oof_all_pairs` |
| `oof_prediction_scope` | Held-out proteins or held-out pairs, explicitly named |
| `prediction_status` | `complete` or an explicit failure/incomplete status |

### Support and coverage columns

| Column | Meaning |
|---|---|
| `n_residues` | Number of aligned C-alpha residues |
| `n_protein_pairs` | $R(R-1)/2$, the complete unordered pair count |
| `n_expected_pairs` | Number expected for the stated evaluation scope |
| `n_evaluated_pairs` | Number with finite true and predicted distances |
| `pair_coverage` | `n_evaluated_pairs / n_expected_pairs` |
| `target_variance` | Population variance of true distances in the evaluated rows |

Every successful v2 row expects the protein's complete unordered-pair
population. Sequence-gated OLS is complete only when every expected gate
prediction is present; it has no fallback.

### Metric columns

| Column | Meaning |
|---|---|
| `r2` | Per-protein $R^2$ over the evaluated pairs |
| `mae` | Per-protein MAE in angstroms |
| `rmse` | Per-protein RMSE in angstroms |
| `bias_pred_minus_true` | Mean signed prediction error in angstroms |
| `mean_true_distance` | Mean target distance for the evaluated rows |
| `mean_predicted_distance` | Mean prediction for the evaluated rows |

The two means and bias are partly redundant, but they are inexpensive and make
calibration and integrity checks easy. Values should be written with enough
precision to avoid visibly changing recomputed summaries.

### Mean-per-protein results

Mean-per-protein metrics are derived by grouping complete OOF rows by analysis
and layer and taking the arithmetic mean across the 552 proteins. $R^2$
averages exclude undefined values and report the finite count. Fold-specific
means are derived by adding `outer_fold` to that grouping. Mean-per-protein
columns are not repeated in `pooled_metrics.csv`.

## Core `pooled_metrics.csv`

Pooled metrics are computed after concatenating the evaluated pair rows across
proteins. They are pair-weighted and therefore give longer proteins more
influence than a mean-per-protein statistic. Both views are scientifically
useful and should be available for the paper.

`pooled_metrics.csv` has one authoritative complete-cohort OOF row per analysis
and layer. It is calculated by merging OOF sufficient statistics across all
five folds, never by averaging fold metrics.

| Column group | Columns |
|---|---|
| Identity | `analysis_name`, `model_family`, `feature_dim`, `layer`, `cv_protocol_version`, `evaluation_design`, `evaluation_scope`, `oof_prediction_scope` |
| Support | `n_proteins`, `n_proteins_with_finite_r2`, `n_expected_pairs`, `n_evaluated_pairs`, `pair_coverage` |
| Model metrics | `pooled_r2`, `pooled_mae`, `pooled_rmse`, `pooled_bias_pred_minus_true` |
| Global baseline | `global_baseline_type`, `global_baseline_pooled_r2`, `global_baseline_pooled_mae`, `global_baseline_pooled_rmse`, `global_baseline_pooled_bias_pred_minus_true` |
| Conditional baseline | `conditional_baseline_type`, `conditional_baseline_pooled_r2`, `conditional_baseline_pooled_mae`, `conditional_baseline_pooled_rmse`, `conditional_baseline_pooled_bias_pred_minus_true` |

This file is authoritative for pooled calculations. Pooled $R^2$ must not be
calculated by averaging per-protein $R^2$, and pooled RMSE must not be
calculated by averaging per-protein RMSE.

`fold_metrics.csv` contains five otherwise equivalent OOF rows per layer, one
per outer fold. For per-protein OLS this is only a matched grouping by protein's
outer-fold label. `fold_summary.csv` records the arithmetic mean and standard
deviation across the five fold metrics. Those values are descriptive and are
not confidence intervals.

## Global and conditional baselines

Every fold model is evaluated against a global intercept-only baseline fitted
on exactly that model's training observations. Its held-out predictions are
concatenated in the same order as model OOF predictions before authoritative
baseline metrics are calculated. Its type is `global_fold_training_mean`.

Models whose fitting scope supplies useful non-representation information also
receive a conditional baseline:

- a gated model predicts the protein-equal weighted fold-training mean for the
  pair's sequence-separation gate; and
- per-protein OLS predicts that protein's four-pair-fold training mean.

The conditional baseline measures improvement over the model's routing or
protein identity alone. It is not featureless and must not be described as the
same control as the global baseline. Shared OLS and shared XGBoost have no
distinct conditional baseline, so their conditional fields are `NaN` with
`conditional_baseline_type=not_applicable`.

Both baseline families use the same OOF evaluation rows as the fitted model and
never access the contributing held-out targets. Baseline metrics belong only in
compact `pooled_metrics.csv`; they are not repeated in every per-protein and
subgroup table.

MAE skill and RMSE skill are omitted because they are exactly
`1 - model_error / baseline_error` and can be calculated when needed. Baseline
$R^2$ may be negative on held-out data because the baseline mean is learned
from training data rather than from the evaluated targets.

## Training diagnostics and stable-fit sensitivity

`training_metrics.csv` contains compact fold-level training R², MAE, RMSE,
bias, support, and fit diagnostics. OLS rows cover all training pairs; XGBoost
rows cover only sampled fitting rows. Training values are diagnostic and never
enter OOF outputs. Training geometry and training subgroup tables are omitted.

Primary OOF outputs retain finite predictions from OLS fits that are flagged
by the current rank/conditioning policy and mark their status explicitly.
`stable_fit_metrics.csv` provides the corresponding OOF sensitivity summaries
restricted to rows for which every contributing OLS fit passed that policy.

## OOF subgroup accuracy

Every final regression produces pooled complete-cohort OOF accuracy in two
fixed category systems. These evaluate the same OOF predictions used for the
whole protein; they do not fit new category-specific models.

### Sequence separation

`subgroup_metrics/by_sequence_separation.csv` uses the fixed bins:

- `1`
- `2-4`
- `5-9`
- `10-19`
- `20-49`
- `50+`

### True spatial distance

`subgroup_metrics/by_spatial_distance.csv` uses true C-alpha distance bins:

- `0-5` for [0, 5) angstroms;
- `5-10` for [5, 10) angstroms;
- `10-20` for [10, 20) angstroms;
- `20-40` for [20, 40) angstroms; and
- `40+` for distances at least 40 angstroms.

True distance is used only to stratify evaluation, never to route training or
prediction.

### Common subgroup schema

Both files share this schema:

| Column group | Columns |
|---|---|
| Identity | `analysis_name`, `model_family`, `feature_dim`, `layer`, `cv_protocol_version`, `evaluation_design`, `evaluation_scope`, `oof_prediction_scope` |
| Category | `category_scope`, `category_label` |
| Support | `n_proteins`, `n_expected_pairs`, `n_evaluated_pairs`, `pair_coverage`, `support_status` |
| Metrics | `pooled_r2`, `pooled_mae`, `pooled_rmse`, `pooled_bias_pred_minus_true` |

These files combine each pair's single held-out prediction across all 552
proteins. Fold-specific subgroup rows are not required.

Subgroup $R^2$ can become extremely negative when a category has almost no
target variance, especially sequence separation 1. MAE, RMSE, bias, pair count,
and protein count should therefore be emphasized when interpreting category
accuracy. Cells with fewer than 100 pairs or fewer than five proteins are
retained but marked in `support_status`.

## Gated-model recombination and coverage

The main gated `per_protein.csv` evaluates the final piecewise predictor after
all gates have been recombined. It must not contain one main row per gate.
Gate-level protein results may remain in explicitly named diagnostic artifacts,
but neither per-gate nor per-protein-by-sequence-bin expansions are required
canonical outputs.

Every sequence-gated fold-layer requires all six valid fitted gates and exactly
one prediction for every expected pair. A missing or invalid gate makes the
fold-layer incomplete; there is no fallback and no partial primary metric.

## Per-protein geometry outputs

Every final regression produces `geometry_metrics.csv`, one row per analysis,
layer, and eligible protein. Geometry is calculated once from the complete
symmetric OOF predicted distance matrix. For per-protein OLS every matrix entry
comes from the model that withheld that pair; training predictions are never
used. The scope is `oof_all_pairs`.

Geometry requires 100% prediction coverage. If even one pair is missing, the
protein-layer row remains in the CSV with
`geometry_status=incomplete_pair_coverage`, an exclusion reason, and `NaN`
geometry metrics. It is excluded from `geometry_summary.csv`. This is preferable
to silently imputing distances or deleting the row.

### Distance-matrix construction and classical MDS

Predictions fill a symmetric $R \times R$ matrix with a zero diagonal. The
number of negative predictions is recorded, after which negative distances are
clipped to zero for geometry calculations.

Classical multidimensional scaling forms the double-centered Gram matrix

\[
B=-\frac{1}{2}J\hat D^{\circ 2}J,
\qquad J=I-\frac{1}{R}\mathbf{1}\mathbf{1}^{\mathsf T},
\]

eigendecomposes $B$, and constructs three-dimensional coordinates from the
three largest nonnegative eigenvalues. Both the original and reflected
three-dimensional solutions are aligned to the predicted-PDB C-alpha
coordinates using the Kabsch algorithm; the lower-RMSD alignment supplies RMSD.
TM-score is optimized independently with the standard `TMscore` executable,
using fixed residue-array correspondence, reference-length protein
normalization, and both MDS reflections. The highest score is retained.
The executable searches temporary C-alpha traces and returns a rigid
transformation, which is evaluated on the original `float64` coordinates with
the existing TM-score formula. Kabsch is also retained as a candidate, so the
score cannot decrease for the same input coordinates. The standard search is
heuristic and does not guarantee a global maximum.

The executable is required for new geometry calculations; missing or failed
backends abort scoring without reverting to the historical metric. Fewer than
three residues are unsupported by the executable. This does not affect the
current frozen cohort, whose minimum length is 30. Historical outputs retain
their original TM-style scores after Kabsch alignment until explicitly rescored.
Installation and saved-distance rescoring are documented in the
[TM-score runbook](../../scripts/cluster_scripts/tm_scores/README.md).

The saved-distance rescoring workflow also retains one reconstructed C-alpha
PDB per protein, probe, and layer, using the winning TM-score reflection and
superposition in the original reference PDB frame. Residue identities follow
the frozen export metadata. These supplementary PDBs use standard 0.001 Å
coordinate precision; reported scores are evaluated before that rounding.
The runbook defines their layout, checksum index, and completion checks.

The MDS geometry metrics are:

- `rmsd`: aligned coordinate RMSD in angstroms;
- `tm_score`: optimized reconstruction TM-score, independent of RMSD alignment;
- `frac_top3_eigenvalues`: fraction of positive eigenvalue mass in the largest
  three nonnegative eigenvalues;
- `n_negative_eigenvalues`: number of negative Gram-matrix eigenvalues;
- `negative_eigenvalue_fraction`: absolute negative eigenvalue mass divided by
  positive eigenvalue mass;
- `stress`: normalized distance discrepancy
  \(\sqrt{\sum(\hat d-d_{MDS})^2/\sum \hat d^2}\); and
- `n_negative_predictions`: number of raw predicted pair distances below zero.

### Exact triangle violations

For each of the $\binom{R}{3}$ unordered residue triples, let $a$ be the
longest predicted side and $b,c$ the other two. The violation magnitude is
$\max(a-b-c,0)$, with a numerical tolerance of $10^{-12}$ angstrom.

Save:

- `n_triangles`;
- `triangle_violation_fraction`;
- `triangle_mean_violation`, defined as the sum of positive violation
  magnitudes divided by all triangles, so nonviolating triangles contribute
  zero;
- `triangle_max_violation`;
- `triangle_violation_0_5_fraction`;
- `triangle_violation_5_10_fraction`; and
- `triangle_violation_10_plus_fraction`.

The three severity fractions use all triangles as the denominator and sum to
`triangle_violation_fraction`. `triangle_no_violation_fraction` is omitted
because it is exactly `1 - triangle_violation_fraction`. Pair- or triangle-level
samples are not core outputs; the aggregate statistics above preserve the paper
result without producing hundreds of megabytes of intermediate CSVs.

### `geometry_metrics.csv` schema

In addition to the metrics above, each row contains:

- the standard analysis, layer, protein, source, outer-fold, CV-protocol, and
  OOF-scope identifiers;
- `n_residues`, `n_protein_pairs`, and `pair_coverage`;
- `evaluation_scope=oof_all_pairs`;
- `geometry_status`; and
- `geometry_exclusion_reason`.

### Derived `geometry_summary.csv`

`geometry_metrics.csv` is authoritative. `geometry_summary.csv` is a compact
long-format convenience table with one row per analysis, layer, and geometry
metric across the complete OOF cohort. It records:

- `n_proteins_attempted`, `n_proteins_included`, and `n_proteins_excluded`;
- `mean`, `std`, `median`, `q25`, and `q75` across included proteins; and
- the metric name and units.

Uncertainty intervals are not part of the current output contract. They can be
added later by common publication-analysis code if the paper requires them.

## Coefficients and model artifacts

### Linear models

All fitted linear probes retain coefficients, intercepts, and scaler state
sufficient for exact prediction replay. Standardized coefficients are the
primary values for comparing feature use; raw-scale coefficients are useful for
replay and interpretation in original feature units.

For shared OLS and sequence-gated OLS,
`model_artifacts/coefficients.csv` may contain:

- analysis, layer, protein ID if applicable, and gate identifiers;
- `feature_index` and `feature_name`;
- `coefficient_standardized` and `coefficient_raw_scale`; and
- standardized and raw intercept rows with `feature_index=-1`.

Per-protein OLS must not expand 132,480 models into repeated CSV rows. It uses
compressed coefficient/scaler shards. Every linear artifact retains
standardized and raw coefficients and intercepts plus feature means and scales.
`model_stability.csv` stores compact standardized-coefficient summaries across
folds rather than duplicating the raw model arrays.

### XGBoost

XGBoost has no linear coefficients. Save one native JSON booster per outer
fold and layer, the fixed S4/5k/700 configuration, selection-source path and
digest, and XGBoost version. `feature_importance.csv` stores fold-summarized
gain and split-count importance; it is descriptive and not directly comparable
to OLS coefficients.

### Required replay index

`model_artifacts/replay_index.csv` has one row per retained artifact, not per
pair prediction. Its required identity fields are `model_family`, `layer`,
nullable `outer_fold`, nullable `protein_id`, nullable `pair_fold`, nullable
`sequence_gate`, `artifact_path`, `artifact_sha256`, configuration and scaler
digests, and `status`. Pair responsibility is recovered by joining this index
to the deterministic outer- or pair-fold manifest.

Expected complete row counts are 240 shared-OLS models, 1,440 sequence-gated
models, 132,480 per-protein OLS models, and 240 XGBoost boosters. This is the
stable interface for a separate triangle-subgroup replay project; v2 does not
save raw pair predictions.

## `model_status.csv`

One row represents one fitted fold model: outer-fold/layer for shared models,
protein/pair-fold/layer for per-protein OLS, or
outer-fold/layer/sequence-gate for gated OLS. Common fields are:

- analysis and fitted-unit identifiers;
- `fit_status` and `support_flags`;
- training protein and pair counts; and
- feature dimension.

OLS rows save design rank and conditioning diagnostics when the fitted solver
already exposes them. For the current dense scikit-learn OLS implementation,
save its reported rank and singular-value extrema and derive the condition
number from those values. Do not run an additional SVD or other matrix
factorization solely to populate diagnostics. If a future solver does not
expose them, retain the fit-status row, set the unavailable fields to `NaN`,
and record that they were not available. XGBoost rows save the selected
configuration ID, final boosting rounds, training row cap, and random seed.
Failed units remain present with explicit status fields.

## Brief runtime output

`runtime.csv` contains one row per executed fold-layer task for shared models
and one row per layer task for per-protein OLS. It should stay small and
include:

- `analysis_name`, `layer`, and completion status;
- numbers of proteins and fitted/evaluated pairs;
- total `elapsed_seconds`; and
- optional peak resident memory if it is already available from the process.

Task, stage-level, and per-protein timing remain in SLURM stdout and are not
required final outputs unless performance itself is being studied.

## Layer shards and consolidation

Array tasks may checkpoint layer-wise or task-wise shards after each completed
layer. Every shard uses the final canonical schema. After consolidation, the
top-level files are authoritative and must be checked for:

- expected layer coverage;
- unique primary keys;
- matching cohort, outer-fold, pair-fold, and configuration digests;
- expected protein and pair counts;
- finite metrics where the status indicates success; and
- exact agreement between consolidated row counts and shard row counts.

Large coefficient NPZ shards may remain the authoritative storage and should
not be duplicated into one monolithic CSV.

## Publication reporting derived from these outputs

The core files support the following paper results without rerunning a model:

- complete-cohort OOF pooled $R^2$, MAE, RMSE, and bias by layer;
- complete-cohort OOF mean-per-protein metrics by layer;
- five-fold means and standard deviations as descriptive summaries;
- paired protein-level differences between model variants;
- OOF accuracy trajectories by sequence separation and spatial distance;
- MDS reconstruction quality and distance-matrix validity by layer; and
- coefficient trajectories and numerical-stability audits.

All 48 layers are saved and reported as trajectories; the output contract does
not designate or save a primary layer. If a later paper table highlights a
single layer, that reporting choice is made downstream and does not replace the
complete layer-wise outputs. Report at least pooled and mean-per-protein OOF
performance, fold mean and standard deviation, protein/pair counts, and
coverage. Do not report confidence intervals from this project.

## Outputs intentionally not required

The following are omitted from final regression families unless a specific
analysis needs them:

- raw prediction rows for every residue pair;
- pair- or triangle-level violation samples;
- duplicate true-distance category statistics already owned by
  `analysis_3_true_distance_statistics`;
- MSE, because RMSE is saved;
- MAE/RMSE skill columns, because model and baseline errors are saved;
- mean-per-protein summaries duplicated inside pooled output;
- baseline columns repeated in every subgroup or protein row;
- per-protein sequence-separation metric expansions;
- per-protein amino-acid-pair metric expansions;
- amino-acid-gated, joint-gated, or amino-acid subgroup migrations from v1;
- per-protein runtime files; and
- bootstrap confidence intervals or uncertainty bands.

These exclusions are storage and consistency choices, not prohibitions on later
derived analyses.

## Optional packed predicted-distance export

The separate [predicted-distance exporter](../../scripts/cluster_scripts/predicted_distances/README.md)
replays the four retained canonical model families without refitting. It saves
one strict-lower-triangle OOF float32 vector per protein, layer, and probe under
`/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/` on
Delta. Existing model inputs remain under `$HOME/csv_files/regression_5fold_v1/`.
The local copy, if downloaded, belongs in
`delta_files/regression_5fold_v1/predicted_distances_5fold_v1/`.

The exporter owns its prediction index, validation, aligned residue metadata,
runtime, and checkpoint/completion manifests. It does not duplicate model
artifacts or produce an aggregated prediction array. Replay parity is checked
against the authoritative producer metrics before casting; stored float32
rounding error and metric drift are recorded separately. Core geometry and
metric tables retain their original authority and precision. See the export
runbook for file schemas, restart behavior, and submission order.

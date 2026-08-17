# Canonical Outputs for Distance-Regression Probes

## Scope and design principle

This document defines the output contract for the paper's shared OLS,
per-protein OLS, gated OLS, shared XGBoost, and future distance-regression
variants. It complements `regression_methodology.md`: that document specifies
how models are fitted, whereas this document specifies what every completed run
must save.

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
<result_family>/
  run_manifest.json
  per_protein.csv
  pooled_metrics.csv
  model_status.csv
  runtime.csv
  geometry_metrics.csv
  geometry_summary.csv
  subgroup_metrics/
    by_sequence_separation.csv
    by_amino_acid_pair.csv
    by_spatial_distance.csv
  model_artifacts/
    coefficient_index.csv
    coefficients.csv or coefficient_shards/*.npz
    booster_layer_XX.json
    feature_importance.csv
  layer_shards/
```

Only applicable artifacts are required. Linear models have coefficients but no
booster file; XGBoost has booster files but no linear coefficients. The run
manifest must identify the output-schema version and the digest of the canonical
protein split rather than copying independent split definitions into every
result family.

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

Every regression produces one authoritative long-format `per_protein.csv`
covering every eligible protein, not only test proteins. One row represents one
analysis, layer, protein, and evaluation scope.

### Identity and scope columns

| Column | Meaning |
|---|---|
| `analysis_name` | Stable unique variant name |
| `model_family` | `shared_ols`, `per_protein_ols`, `gated_ols`, or `shared_xgboost` |
| `feature_dim` | Number of model input features |
| `layer` | Pair-representation layer |
| `protein_id` | Canonical protein identifier |
| `source` | Input-layout source, `cameo` or `ecod` |
| `protein_split` | `development` or locked `test` in the final-model output |
| `evaluation_scope` | Pair population evaluated in this row |
| `prediction_status` | `complete`, `partial_coverage`, or an explicit failure status |

Shared OLS, gated OLS, and final shared XGBoost normally use
`evaluation_scope=all_pairs`. Joint AA-pair-by-sequence gated models may use
that scope with coverage below one. Per-protein OLS uses three rows per
protein-layer combination: `train_pairs`, `heldout_pairs`, and `all_pairs`.

### Support and coverage columns

| Column | Meaning |
|---|---|
| `n_residues` | Number of aligned C-alpha residues |
| `n_protein_pairs` | $R(R-1)/2$, the complete unordered pair count |
| `n_expected_pairs` | Number expected for the stated evaluation scope |
| `n_evaluated_pairs` | Number with finite true and predicted distances |
| `pair_coverage` | `n_evaluated_pairs / n_expected_pairs` |
| `target_variance` | Population variance of true distances in the evaluated rows |

The distinction between expected and evaluated pairs prevents an 80/20
within-protein split from looking like a coverage failure. For example, a
held-out-pair row can have 100% coverage even though its expected population is
only 20% of all protein pairs.

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

Mean-per-protein metrics are derived by grouping `per_protein.csv` by analysis,
layer, protein split, and evaluation scope and taking the arithmetic mean across
proteins. $R^2$ averages must exclude undefined values and report the number
of proteins with finite $R^2$. MAE and RMSE normally include every successfully
evaluated protein.

Therefore, mean-per-protein columns do not need to be repeated in the core
pooled output. For historical k-fold outputs, a cross-validated mean must use
only held-out/test rows. Averaging training and test rows together would count
proteins multiple times and mix evaluation scopes.

In the current `delta_files/baseline256_kfold/per_protein.csv`, each of the 184
proteins has exactly one test row per layer across the five folds. Consequently,
filtering to `split=test` and averaging `per_protein_r2`, `per_protein_mae`, or
`per_protein_rmse` by layer directly gives the cross-validated
mean-per-protein result.

## Core `pooled_metrics.csv`

Pooled metrics are computed after concatenating the evaluated pair rows across
proteins. They are pair-weighted and therefore give longer proteins more
influence than a mean-per-protein statistic. Both views are scientifically
useful and should be available for the paper.

`pooled_metrics.csv` has one row per analysis, layer, protein split, and
evaluation scope with these columns:

| Column group | Columns |
|---|---|
| Identity | `analysis_name`, `model_family`, `feature_dim`, `layer`, `protein_split`, `evaluation_scope` |
| Support | `n_proteins`, `n_proteins_with_finite_r2`, `n_expected_pairs`, `n_evaluated_pairs`, `pair_coverage` |
| Model metrics | `pooled_r2`, `pooled_mae`, `pooled_rmse`, `pooled_bias_pred_minus_true` |
| Global baseline | `global_baseline_type`, `global_baseline_pooled_r2`, `global_baseline_pooled_mae`, `global_baseline_pooled_rmse`, `global_baseline_pooled_bias_pred_minus_true` |
| Conditional baseline | `conditional_baseline_type`, `conditional_baseline_pooled_r2`, `conditional_baseline_pooled_mae`, `conditional_baseline_pooled_rmse`, `conditional_baseline_pooled_bias_pred_minus_true` |

This file is authoritative for pooled calculations. Pooled $R^2$ must not be
calculated by averaging per-protein $R^2$, and pooled RMSE must not be
calculated by averaging per-protein RMSE.

Final shared models should report `development` and locked `test` rows. The
paper's main generalization claim uses the locked test row. Search-stage fit and
validation metrics remain in the XGBoost search directory and are not mixed
into the final-model output.

## Global and conditional baselines

Every model is evaluated against a common global intercept-only baseline. It
predicts the protein-equal weighted mean distance estimated from the 80%
development proteins, without using locked-test targets. This gives all model
families the same featureless reference and catches gross metric or split
errors. Its type is recorded as `global_development_mean`.

Models whose fitting scope supplies useful non-representation information also
receive a conditional baseline:

- a gated model predicts the protein-equal weighted development mean for the
  pair's sequence-separation, amino-acid-pair, or joint gate; and
- per-protein OLS predicts that protein's 80% training-pair mean.

The conditional baseline measures improvement over the model's routing or
protein identity alone. It is not featureless and must not be described as the
same control as the global baseline. Shared OLS and shared XGBoost have no
distinct conditional baseline, so their conditional fields are `NaN` with
`conditional_baseline_type=not_applicable`.

Both baseline families are estimated only from their defined training
population and use the same evaluation rows as the fitted model. Neither uses
locked-test targets. The per-protein conditional baseline also excludes that
protein's held-out pair targets. Development rows are in-sample descriptions,
just as they are for the corresponding fitted models. Baseline metrics belong
only in compact `pooled_metrics.csv`; they do not need to be repeated in every
per-protein and subgroup table.

MAE skill and RMSE skill are omitted because they are exactly
`1 - model_error / baseline_error` and can be calculated when needed. Baseline
$R^2$ may be negative on held-out data because the baseline mean is learned
from training data rather than from the evaluated targets.

## Test-set subgroup accuracy

Every final regression produces pooled test-set accuracy in three fixed
category systems. These evaluate the same final piecewise prediction used for
the whole protein; they do not fit new category-specific models.

### Sequence separation

`subgroup_metrics/by_sequence_separation.csv` uses the fixed bins:

- `1`
- `2-4`
- `5-9`
- `10-19`
- `20-49`
- `50+`

### Amino-acid pair

`subgroup_metrics/by_amino_acid_pair.csv` uses the 210 unordered pairs of the
20 standard amino acids. Labels are alphabetically canonicalized, such as
`ALA-ARG`, so residue order does not create duplicate categories.

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

All three files share this schema:

| Column group | Columns |
|---|---|
| Identity | `analysis_name`, `model_family`, `feature_dim`, `layer`, `protein_split`, `evaluation_scope` |
| Category | `category_scope`, `category_label` |
| Support | `n_proteins`, `n_expected_pairs`, `n_evaluated_pairs`, `pair_coverage`, `support_status` |
| Metrics | `pooled_r2`, `pooled_mae`, `pooled_rmse`, `pooled_bias_pred_minus_true` |

These files contain locked-test-protein results only. For per-protein OLS, the
canonical subgroup scope is held-out pairs from proteins whose master role is
test. For shared models it is all pairs from locked test proteins.

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

For sequence-only and amino-acid-only gating, every valid standard-residue pair
is expected to receive a prediction. For joint AA-pair-by-sequence gating,
sparse unfitted gates may leave pairs without predictions. The main regression
metrics may be calculated on the predicted subset, but every row must report
`n_expected_pairs`, `n_evaluated_pairs`, `pair_coverage`, and
`prediction_status=partial_coverage` when appropriate. Missing pairs may never
be treated as zero error.

## Per-protein geometry outputs

Every final regression produces `geometry_metrics.csv`, one row per analysis,
layer, and eligible protein. Geometry is calculated from the complete symmetric
predicted distance matrix, using all unordered residue pairs. For per-protein
OLS, this necessarily combines its training-pair and held-out-pair predictions;
the output must be labeled `evaluation_scope=all_pairs`.

Geometry requires 100% prediction coverage. If even one pair is missing, as can
happen for the joint gated probe, the protein-layer row remains in the CSV with
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
coordinates using the Kabsch algorithm; the lower-RMSD alignment is retained.

The MDS geometry metrics are:

- `rmsd`: aligned coordinate RMSD in angstroms;
- `tm_score`: TM-score of the aligned MDS reconstruction;
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

- the standard analysis, layer, protein, source, and protein-split identifiers;
- `n_residues`, `n_protein_pairs`, and `pair_coverage`;
- `evaluation_scope=all_pairs`;
- `geometry_status`; and
- `geometry_exclusion_reason`.

### Derived `geometry_summary.csv`

`geometry_metrics.csv` is authoritative. `geometry_summary.csv` is a compact
long-format convenience table with one row per analysis, layer, protein split,
and geometry metric. It records:

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

For small families, such as one shared model per layer or six sequence gates
per layer, `model_artifacts/coefficients.csv` may contain:

- analysis, layer, protein ID if applicable, and gate identifiers;
- `feature_index` and `feature_name`;
- `coefficient_standardized` and `coefficient_raw_scale`; and
- standardized and raw intercept rows with `feature_index=-1`.

Large families must not be expanded into enormous repeated CSV rows.
Per-protein OLS, amino-acid-gated OLS, and joint-gated OLS should use compressed
NPZ shards plus `coefficient_index.csv`. Each shard stores dense coefficient,
intercept, feature-mean, and feature-scale arrays; the index records the model
identity, shard path, array row, fit status, feature dimension, and file digest.
Feature order is fixed in the run manifest.

### XGBoost

XGBoost has no linear coefficients. Save one native JSON booster per final
layer, the complete selected configuration, the validation-derived boosting
round count, and the XGBoost version. A compact `feature_importance.csv` may
store gain and split-count importance, but it is a descriptive model-specific
artifact and must not be treated as directly comparable to OLS coefficients.

## `model_status.csv`

One row represents one fitted model unit: a layer for a shared model, a
protein-layer for per-protein OLS, or a layer-gate for gated OLS. Common fields
are:

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

`runtime.csv` contains one row per analysis and layer. It should stay small and
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
- matching split and configuration digests;
- expected protein and pair counts;
- finite metrics where the status indicates success; and
- exact agreement between consolidated row counts and shard row counts.

Large coefficient NPZ shards may remain the authoritative storage and should
not be duplicated into one monolithic CSV.

## Publication reporting derived from these outputs

The core files support the following paper results without rerunning a model:

- pooled test $R^2$, MAE, RMSE, and bias by layer;
- mean-per-protein test metrics by layer;
- paired protein-level differences between model variants;
- accuracy trajectories by sequence separation, amino-acid pair, and spatial
  distance;
- MDS reconstruction quality and distance-matrix validity by layer; and
- coefficient trajectories and numerical-stability audits.

All 48 layers are saved and reported as trajectories; the output contract does
not designate or save a primary layer. If a later paper table highlights a
single layer, that reporting choice is made downstream and does not replace the
complete layer-wise outputs. Report at least pooled and mean-per-protein test
performance, test protein/pair counts, and coverage.

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
- per-protein runtime files; and
- bootstrap confidence intervals or uncertainty bands.

These exclusions are storage and consistency choices, not prohibitions on later
derived analyses.

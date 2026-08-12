# Local Biological-Context Distance-Update Probe Plan

## Purpose

This diagnostic asks whether a small, biologically interpretable description of
a residue pair and its local triangle geometry predicts how the linearly decoded
distance changes at the next OpenFold trunk layer. The intended result is not
only a predictive score. It is a set of inspectable rules such as:

> A pair containing glycine is likely to shrink when a nearby residue creates a
> strong upper-bound triangle violation.

The first run is deliberately narrow and should finish each array task in
roughly 30--60 minutes. It uses five transitions, seven fixed feature bundles,
and fixed model settings apart from a compact training-only logistic
regularization search.

## Locked initial scope

- Analyze transitions 10->11, 15->16, 20->21, 30->31, and 40->41.
- Use the frozen split-0 combined CAMEO+ECOD assignment from
  `general_base256_combined`: currently 441 training proteins and 111 held-out
  test proteins. CAMEO and ECOD remain one analysis population.
- Use every canonical pair `i < j` with inclusive sequence separation 4--12.
- Use only sequence information and distances decoded at layer `x` as model
  inputs. Layer `x+1` is used only to construct the target.
- Do not use final-PDB coordinates, final contacts, DSSP, or solvent exposure
  as predictor features. Pair-representation channels are excluded from every
  bundle except the explicitly named `biology_local_triangle_plus_center`
  ablation. The PDB residue records may be used to recover amino-acid identity
  after validating that their order and count match the pair block.
- Fit continuous distance-change regression on all eligible pairs.
- Fit binary direction classification only on meaningful updates:
  `shrink` for `delta_d < -0.5 A` and `expand` for `delta_d > +0.5 A`.
  Pairs with `abs(delta_d) <= 0.5 A` do not enter classifier fitting or its
  headline evaluation.
- Keep all feature matrices and numerical fitting in `float64`.
- Give every training protein equal total sample weight. For classification,
  additionally balance the two classes after protein weighting.

## Distance target and layer-specific readouts

For transition `x -> x+1`, decode the two endpoints with distinct saved native
linear distance probes:

```text
d_x(i,j)     = saved layer-x probe applied to the layer-x pair representation
d_x+1(i,j)   = saved layer-(x+1) probe applied to the layer-(x+1) representation
delta_d(i,j) = d_x+1(i,j) - d_x(i,j)
```

For example, 10->11 must use the saved layer-10 coefficients and intercept for
`d_10`, and the separately fitted layer-11 coefficients and intercept for
`d_11`. It must not apply one shared probe to both endpoint representations.
The same rule applies to 15->16, 20->21, 30->31, and 40->41. The source is the
split-0 `general_base256_combined` 256-feature bidirectional linear probe
family.

The regression target and current-distance control use raw pre-MDS decoded
values without clipping. Local triangle calculations use the complete decoded
layer-`x` distance matrix after clipping negative distances to zero. The raw
negative count and clipping policy are recorded in metadata.

## Local sequence context

For target pair `(i,j)`, define the unique local third-residue set

```text
K(i,j) = ({i-5,...,i+5} union {j-5,...,j+5}) minus {i,j}
```

after removing out-of-range positions and deduplicating overlaps. A residue is
therefore local when it lies within five sequence positions of either endpoint;
it need not be spatially close. This same set supplies the neighborhood biology
and triangle-pressure features.

The local representation is intentionally pooled and symmetric. The first run
does not create exact-offset features such as `i_minus_3_is_PRO`, and a residue
in overlapping endpoint windows is counted once.

## Exact feature ledger

All categorical definitions and amino-acid property tables must be fixed in the
code, copied into run metadata, and listed feature-by-feature in
`feature_schema.csv`.

### 1. Basic controls: 5 features

1. Raw decoded current distance `d_x(i,j)`.
2. Exact sequence separation `j-i`.
3. Protein residue count.
4. Normalized pair midpoint `(i+j)/(2*(R-1))`.
5. Normalized distance of the pair to its nearest sequence terminus,
   `min(i, R-1-j)/(R-1)`.

### 2. Endpoint amino-acid identity: 20 features

Use one count for each standard amino acid. A heterogeneous pair activates two
columns with value one; a homotypic pair gives its amino acid a value of two.
This unordered encoding directly represents "the pair contains amino acid X"
without the 210-dimensional full pair-type table.

Common aliases such as MSE must be normalized by an explicit mapping. Any other
nonstandard residue is all-zero in this block and is counted in the manifest;
the run should fail if this is not rare enough to make that policy auditable.

### 3. Endpoint physicochemical summaries: 12 features

For each of four fixed residue properties, save the endpoint sum and absolute
difference, giving eight features:

- Kyte--Doolittle hydrophobicity;
- side-chain volume from one fixed recorded table;
- polarity from one fixed recorded table;
- nominal side-chain charge from one fixed recorded table.

Also save four endpoint class counts, each in `0,1,2`:

- hydrophobic;
- aromatic;
- small;
- polar.

These grouped properties are intentionally somewhat redundant with identity.
They let shallow trees discover chemistry-level rules that can generalize
across amino acids.

### 4. Explicit pair chemistry: 10 features

Use ten binary indicators:

1. same amino acid;
2. Cys--Cys;
3. Gly--Pro;
4. both hydrophobic;
5. both aromatic;
6. opposite charge;
7. both positively charged;
8. both negatively charged;
9. both small;
10. both bulky.

The full set of 210 unordered amino-acid pair types is deferred to a later
ablation.

### 5. Local sequence biology: 26 features

Using the unique `K(i,j)` window, save:

- 20 local amino-acid counts;
- four local class counts: hydrophobic, aromatic, charged, and small;
- the number of valid unique local third residues;
- the number of those residues lying strictly between `i` and `j`.

The window size features allow linear and tree models to distinguish genuine
composition effects from protein-boundary or window-overlap effects. Counts
rather than exact offsets make rules such as "at least one nearby proline"
directly readable.

### 6. Compact local triangle pressure: 6 features

For each `k` in `K(i,j)`, compute from clipped layer-`x` decoded distances:

```text
shrink_pressure_k = max(d_ij - d_ik - d_kj, 0)
grow_pressure_k   = max(abs(d_ik - d_kj) - d_ij, 0)
```

Aggregate only:

1. maximum shrink pressure;
2. mean shrink pressure over every valid local `k`, including zeros;
3. fraction of valid local `k` with positive shrink pressure;
4. maximum grow pressure;
5. mean grow pressure over every valid local `k`, including zeros;
6. fraction of valid local `k` with positive grow pressure.

Together with the local-third-residue count in family 5, these distinguish one
severe inconsistency from many modest inconsistencies. Sums, multiple
quantiles, global/all-residue triangles, distance-profile features, and
separate statistics for every mediator amino acid are excluded from the first
run.

### 7. Strongest-mediator identity extension: 40 features

As one final ablation, add:

- a 20-dimensional one-hot identity for the `k` with maximum positive shrink
  pressure;
- a 20-dimensional one-hot identity for the `k` with maximum positive grow
  pressure.

The corresponding block is all zero when no positive pressure exists. Ties use
the smallest sequence index and the tie policy is recorded. This tests whether
mediator identity adds information beyond the magnitude of the geometric
inconsistency without constructing per-amino-acid pressure summaries.

### 8. Center pair-representation extension: 128 features

As a separate final ablation, append the symmetrized layer-`x` center pair
representation:

```text
center_ch_c = 0.5 * (pair_block_x[i,j,c] + pair_block_x[j,i,c])
```

This is the only predictor family permitted to contain pair-representation
channels. It is not combined with the strongest-mediator identity extension in
the initial run. Layer `x+1` representations remain target-only.

### Feature totals

| Feature family | Width | Cumulative full width |
|---|---:|---:|
| Basic controls | 5 | 5 |
| Endpoint amino-acid identity | 20 | 25 |
| Endpoint physicochemical summaries | 12 | 37 |
| Explicit pair chemistry | 10 | 47 |
| Local sequence biology | 26 | 73 |
| Compact local triangle pressure | 6 | 79 |
| Optional strongest-mediator identity | 40 | 119 |
| Optional symmetrized center representation | 128 | 207 when added to the 79-feature model |

## Feature bundles

Fit the following seven nested or targeted bundles. The local-triangle bundle
uses the local-third-residue count from the local-biology family but none of
its other biological composition columns.

| Bundle | Included information | Width |
|---|---|---:|
| `controls` | Basic controls | 5 |
| `endpoint_biology` | Controls + endpoint identity + endpoint chemistry + pair chemistry | 47 |
| `local_biology` | Endpoint biology + all local sequence biology | 73 |
| `local_triangle` | Controls + local-third-residue count + six triangle-pressure features | 12 |
| `biology_local_triangle` | All families except mediator identity | 79 |
| `biology_local_triangle_mediator` | Full model + strongest shrink/grow mediator identities | 119 |
| `biology_local_triangle_plus_center` | 79-feature biology/triangle model + layer-`x` symmetrized center representation | 207 |

The primary scientific comparisons are:

- `local_biology - endpoint_biology`: value of nearby residue composition;
- `local_triangle - controls`: value of local geometric inconsistency;
- `biology_local_triangle - local_triangle`: biological information beyond
  geometry;
- `biology_local_triangle_mediator - biology_local_triangle`: value of the
  strongest mediator's amino-acid identity.
- `biology_local_triangle_plus_center - biology_local_triangle`: additional
  pair-representation signal beyond the explicit biological and local-geometric
  features.

## Models and fixed settings

Every feature bundle and transition receives both regression and binary
classification models. Scaling parameters are learned only from training
proteins and give proteins equal influence. Trees consume the unstandardized
features; linear models consume standardized features.

### Interpretable linear models

- Continuous target: L2-regularized Ridge regression using the existing
  relative-alpha convention (`0.001` times normalized training weight mass).
- Binary target: L2-regularized logistic regression with `lbfgs` and a
  convergence limit of 200 iterations. Select `C` from the fixed grid
  `{0.001, 0.01, 0.1, 1.0}` using a deterministic source/size-stratified 80/20
  split of the 441 training proteins. Use mean per-protein balanced accuracy on
  meaningful-change validation pairs as the selection metric, break exact ties
  toward smaller `C`, then refit the selected setting on all 441 training
  proteins. The held-out 111 proteins never participate in selection. Reuse
  one saved validation assignment across transitions and bundles.

Save all grid scores, the selected `C`, validation assignments, intercepts,
standardized coefficients, raw-unit coefficients when well defined,
convergence status, and iteration counts. Ignore nonconverged candidates when
selecting and fail the shard if no candidate converges.

### Primary shallow decision trees

Fit a `DecisionTreeRegressor` and `DecisionTreeClassifier` with:

- maximum depth 4;
- minimum leaf size `max(1,000 rows, 0.2% of eligible training rows)`;
- fixed random seed;
- squared-error regression and log-loss classification criteria;
- the same protein-equal/class-balanced sample weights as the linear models.

A depth-4 tree has at most 16 leaves and remains small enough to inspect in
full. To check the most obvious underfitting concern without tuning against the
test set, also fit a depth-6 sensitivity tree only for
`biology_local_triangle`. It is reported as a sensitivity analysis, never used
to select or replace the depth-4 tree. A large held-out gain from depth 6 would
motivate a later validation-based depth study.

For every fitted tree, save:

- a plain-text rule listing;
- Graphviz `.dot` structure;
- a node table with node/parent IDs, depth, branch direction, split feature,
  threshold, impurity, prediction, and weighted/unweighted training support;
- a leaf/rule table with the complete path and train/test pair support, unique
  protein support, target mean or class prevalence, prediction, and held-out
  accuracy/error;
- a serialized model and its exact feature ordering.

These artifacts, rather than feature importance alone, are the main
interpretability interface.

### Gradient-boosted nonlinear benchmark

Use scikit-learn histogram gradient boosting for both targets with fixed
settings:

- 500 boosting iterations;
- learning rate 0.08;
- at most 15 leaves per component tree;
- minimum 200 rows per leaf;
- L2 regularization 1.0;
- fixed random seed;
- `early_stopping=False` passed explicitly to both estimators;
- no histogram-gradient-boosting hyperparameter search.

This is a stronger nonlinear benchmark, not a formal upper bound. Its purpose
is to reveal whether simple nonlinear combinations contain substantially more
signal than the linear or depth-4 models. Save the serialized estimator and
training/test metrics. For inexpensive interpretation, measure held-out
permutation importance by shuffling one feature family at a time on a fixed
protein-balanced sample of at most 25,000 test pairs, with three repeats. Do
not run feature-by-feature SHAP in the pilot.

## Triangle-pressure collinearity diagnostic

For every transition and split, compute the Pearson correlation between raw
decoded current distance `d_x(i,j)` and each of the six compact triangle-
pressure features. Save pair-pooled correlation, mean finite per-protein
correlation, and the number of proteins with a defined correlation in
`triangle_distance_correlations.csv`.

Whenever a pressure feature appears in a reported shallow-tree rule, include
the corresponding correlation beside the rule in `tree_rule_metrics.csv` and
`summary.txt`. Label absolute correlations below 0.3 as weak, 0.3--0.7 as
moderate, and at least 0.7 as high. A high-correlation pressure rule must be
flagged as ambiguous with current distance rather than described as cleanly
independent geometric evidence.

## Evaluation

### Binary direction classification

Headline evaluation uses only `abs(delta_d) > 0.5 A` and reports pair-pooled
and mean-per-protein:

- accuracy and balanced accuracy;
- shrink precision, recall, and F1;
- AUROC and average precision using the continuous decision score;
- class counts, protein counts, and coverage relative to all local pairs.

Also apply the frozen classifier to all nonzero changes for a diagnostic only.
Report accuracy and prediction-score behavior in signed true-change bins:

```text
(-inf,-4), [-4,-2), [-2,-1), [-1,-0.5), [-0.5,-0.25), [-0.25,0),
(0,0.25], (0.25,0.5], (0.5,1], (1,2], (2,4], (4,inf)
```

The four central bins remain explicitly labeled `excluded_from_training`; they
must not contribute to the headline score. Also report cumulative direction
accuracy at `abs(delta_d) > 0.5`, `>1`, `>2`, and `>4 A`. Each bin records
pair support, unique-protein support, accuracy, balanced accuracy, shrink
precision/recall, and mean score.

### Continuous distance-change regression

Evaluate all pairs using pair-pooled and mean-per-protein R2, MAE, RMSE,
Pearson correlation, bias, and sign accuracy. Reuse the signed bins above for
MAE, RMSE, bias, and sign accuracy. Bin-specific R2 is saved but is secondary
because narrow target ranges make it unstable.

### Uncertainty and model comparison

Use 200 deterministic protein-bootstrap replicates over held-out proteins
without refitting models. Within a transition, every bundle and estimator must
use the same protein draw for a given replicate ID. Generate the assignments
from the ordered test-protein list and a fixed transition-specific seed, save
each protein's multiplicity in `bootstrap_assignments.csv`, and make aggregation
fail if bundle shards disagree about an assignment.

Save replicate-level headline metrics in `bootstrap_metrics.csv`. Construct
bundle-to-bundle gains by subtracting matched replicate metrics, so confidence
intervals are paired rather than based on independent draws. Use the same
assignments for shallow-tree rule/leaf effects. This reflects protein-level
evaluation uncertainty rather than full refit uncertainty.

The held-out test split is used only for final reporting. Fixed model settings
and the depth-6 sensitivity check are declared in advance; the pilot does not
select settings by test performance.

## Parallel execution and runtime budget

Use a 35-task Slurm array: five transitions crossed with seven feature bundles.
Each task builds one bundle once, then fits its regression and classification
linear, shallow-tree, and histogram-boosting models. The depth-6 sensitivity
models run inside the five `biology_local_triangle` tasks. A dependent
aggregation job validates and combines all completed shards.

The initial sbatch target should be one hour per array task, with the same Delta
CPU partition and scratch-node convention as the existing local update pilot.
Start with at most 18 concurrent tasks, 16 CPUs, and 64 GB per task; lower the
concurrency if allocation pressure warrants it. The 35 tasks may therefore run
in two scheduler waves, so total elapsed array time can exceed one hour even
when each task meets the target. Log feature construction and each fit
separately. Save classification results before regression/remaining fits so a
timeout leaves useful partial diagnostics, but aggregation requires all
declared models to be complete.

Apart from the declared four-value training-only logistic `C` search, no
cross-validated hyperparameter search, SHAP calculation, all-layer sweep,
all-residue triangle scan, or pair-level prediction export belongs in this
pilot. If 500-round histogram boosting threatens the one-hour budget, change
its fixed iteration count only in a new versioned configuration rather than
silently sampling the test set or dropping models.

## Intended implementation and outputs

Implement the portable Delta bundle under:

```text
scripts/cluster_scripts/diagnostics/local_pair_biological_context_probe/
```

Write cluster results under:

```text
$HOME/csv_files/local_pair_biological_context_probe/
```

and copy completed aggregate results into:

```text
delta_files/local_pair_biological_context_probe/
```

Array shards use:

```text
shards/transition_XX_YY/<feature_bundle>/
```

Principal aggregate files and row granularities are:

- `model_summary.csv`: one row per transition, feature bundle, target,
  estimator, and split; headline metrics, row/protein counts, convergence, and
  runtime.
- `feature_ablation_summary.csv`: one row per transition, target, estimator,
  and declared bundle contrast; held-out metric difference and protein-
  bootstrap interval.
- `logistic_regularization_search.csv`: one row per transition, bundle, and
  candidate `C`, with validation metrics, convergence, selection status, and
  fit timing.
- `logistic_validation_assignments.csv`: one row per training protein with its
  deterministic logistic fit/validation role and stratification fields.
- `classification_magnitude_metrics.csv`: one row per transition, bundle,
  estimator, split, and signed or cumulative magnitude bin.
- `regression_magnitude_metrics.csv`: one row per transition, bundle,
  estimator, split, and signed or cumulative magnitude bin.
- `per_protein_metrics.csv`: one row per protein, transition, bundle, target,
  estimator, and split.
- `linear_coefficients.csv`: one row per transition, bundle, target, output
  class where relevant, and feature.
- `tree_nodes.csv`: one row per transition, bundle, target, tree depth variant,
  and tree node.
- `tree_rule_metrics.csv`: one row per transition, bundle, target, tree depth
  variant, split, and terminal rule.
- `tree_structures/<transition>/<bundle>/<target>_<depth>.txt`: complete
  human-readable decision paths.
- `tree_structures/<transition>/<bundle>/<target>_<depth>.dot`: Graphviz tree.
- `feature_family_importance.csv`: one row per transition, bundle, target,
  estimator, feature family, and permutation repeat/summary.
- `triangle_distance_correlations.csv`: one row per transition, split, and
  triangle-pressure feature, with pair-pooled and mean per-protein correlation
  plus the interpretive correlation label.
- `bootstrap_assignments.csv`: one row per transition, replicate, and held-out
  protein with its draw multiplicity.
- `bootstrap_metrics.csv`: one row per transition, replicate, bundle, target,
  estimator, metric, and aggregation scope.
- `feature_schema.csv`: one row per feature and bundle membership, with family,
  units, biological definition, transformation, and source layer.
- `feature_transforms.csv`: one row per transition, bundle, and feature with
  training-only mean, scale, missing policy, and observed range.
- `models/<transition>/<bundle>/<target>_<estimator>.joblib`: serialized fitted
  estimators and feature order.
- `change_statistics.csv`: one row per transition and split with target
  moments, quantiles, direction counts, and threshold coverage.
- `protein_manifest.csv`: one row per protein and transition with source,
  split, residue/pair counts, PDB and pair-block paths, mapping validity,
  negative decoded-distance counts, status, and elapsed time.
- `skipped_proteins.csv`: one row per failed protein/transition and reason.
- `split_assignments.csv` and `coefficients_used.csv`: exact frozen split and
  endpoint-specific distance-readout provenance.
- `timing.csv`: one row per shard and processing/fitting/evaluation stage.
- `run_metadata.json`: complete feature, threshold, model, seed, dependency,
  path, and completion configuration.
- `summary.txt`: human-readable held-out rankings, ablation gains, magnitude
  trends, strongest stable rules, limitations, and total wall time.

Pair-level feature matrices and predictions are intentionally not persistent
outputs. The per-protein, magnitude-bin, rule, and bootstrap tables preserve
the diagnostic conclusions without a very large export.

## Acceptance checks

- Every transition loads two distinct endpoint coefficient bundles and records
  their layer numbers and hashes.
- Recomputed target samples agree exactly with direct application of the two
  saved probes.
- `K(i,j)` contains only valid unique indices within five positions of at least
  one endpoint and never contains `i` or `j`.
- Triangle pressure boundary cases match the upper and reverse triangle
  inequalities, and all pressure values are nonnegative.
- Feature schemas have widths 5, 47, 73, 12, 79, 119, and 207 for the seven
  bundles, and center channels occur only in the 207-feature schema.
- Training-only scaling and protein-level splitting are verified by tests.
- Logistic validation assignments are deterministic and test proteins never
  enter `C` selection; selection uses only converged candidates and applies the
  declared tie break.
- The classifier never fits on `abs(delta_d) <= 0.5 A`; central-bin reports
  state that those rows were excluded.
- Both histogram-gradient-boosting estimators explicitly have
  `early_stopping=False` and complete all 500 declared iterations.
- All bundle shards within a transition contain identical bootstrap
  assignments, and aggregate gain intervals subtract matched replicate IDs.
- Triangle-distance correlations match direct calculations and every reported
  pressure-based rule includes its correlation and ambiguity label.
- Every saved tree text/DOT/node/rule representation reconstructs the same
  predictions as its serialized model on test fixtures.
- Aggregate tables reconcile exactly with all 35 complete shards and with the
  frozen protein assignment.

## Interpretation boundary

Success would show that local sequence biology and/or local metric
inconsistency predicts the direction or magnitude of the next linearly decoded
distance update on held-out proteins. It would not show that OpenFold performs
physical folding in time, that a fitted rule is causal, or that final structure
necessarily follows the same rule. Those claims require later interventions or
independent structural validation.

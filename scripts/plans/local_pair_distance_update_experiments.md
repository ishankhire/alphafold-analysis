# Local Pair Distance-Update Experiments

## Research question

These experiments ask whether OpenFold's pair representation at one layer
contains information about how its implied residue-pair distance will change at
the next layer. The focus is on local residue pairs: two residues are included
when they are separated by 4–12 positions in the sequence, inclusive. For
example, residues 1 and 5 are included, while immediately adjacent residues
and pairs separated by more than 12 positions are not.

The main experiment tests two related forms of prediction:

1. Can a linear classifier predict whether the distance will decrease by more
   than 0.5 Å, remain within ±0.5 Å, or increase by more than 0.5 Å?
2. Can a linear regression probe predict the actual signed change in distance,
   in Ångströms?

The feature comparison is intended to reveal whether the pair's own
representation is sufficient, and whether the neighboring pair
representations that complete local triangles provide additional predictive
information.

## Defining a predicted distance update

The experiment uses the existing `general_base256_combined` linear distance
probe family. Each probe maps the two directional pair representations for a
residue pair to a predicted Cα distance. For a transition from layer `x` to
layer `x+1`, each representation is decoded with its own saved layer-specific
probe:

```text
distance at x     = layer-x probe applied to the layer-x pair representation
distance at x + 1 = layer-(x+1) probe applied to the layer-(x+1) representation
change            = distance at x + 1 minus distance at x
```

This own-layer definition preserves each layer's fitted distance calibration.
The target therefore includes both changes in the pair representation and
changes in the fitted readout. Because the probes have separate intercepts,
the intercept terms do not generally cancel when the distances are
subtracted.

All changes are raw, pre-MDS predictions. They are not clipped at zero and are
not projected into a three-dimensional structure. Consequently, this is an
experiment about the distance information linearly readable from the pair
representations, not necessarily about a geometrically realizable distance
matrix.

## Dataset and evaluation split

The scripts use the fixed combined CAMEO and ECOD dataset associated with the
saved distance probe. The usable frozen split contains 552 proteins:

- 441 training proteins, contributing 812,097 local residue pairs;
- 111 held-out test proteins, contributing 207,900 local residue pairs.

Every local pair is used. Proteins, rather than residue pairs, define the split,
so no pair from a test protein can leak into training. CAMEO and ECOD are
treated as one analysis population; their source labels are retained only to
resolve their different pair-block locations and to support auditing.

Metrics are reported in two complementary ways. Pair-pooled metrics describe
performance across all residue pairs and therefore give more influence to
larger proteins. Mean per-protein metrics first score each protein separately
and then average the scores, preventing a small number of large proteins from
dominating the conclusion.

## Main classification and regression pilot

The implementation lives in
`scripts/cluster_scripts/diagnostics/local_pair_update_probe_pilot/`. It tests
three widely separated transitions—10→11, 20→21, and 30→31—to provide an
early-, middle-, and later-trunk snapshot without running all consecutive
layers.

### Information given to the probes

Only layer-`x` information is used as input. Layer `x+1` is used solely to
construct the target, so the predictor cannot directly observe the update it
is being asked to predict.

Except for the scalar baseline, every pair representation is symmetrized by
averaging its two directions. This makes an edge representation invariant to
whether it is read as `(a,b)` or `(b,a)`.

Five feature sets are compared:

1. **Current distance and sequence separation.** This two-number baseline asks
   how predictable the update is from the current decoded distance and how far
   apart the residues are in sequence.
2. **Center pair only.** The input is the 128-channel symmetrized representation
   of the target pair `(i,j)`. This asks how much update information is already
   stored in the pair itself.
3. **Radius-1 triangle context.** In addition to the center pair, the probe
   sees complete two-edge paths from `i` to `j` through residues one position
   away from either endpoint: `i-1`, `i+1`, `j-1`, and `j+1`.
4. **Radius-2 triangle context.** This contains radius 1 and adds the analogous
   paths through residues two positions from either endpoint.
5. **Radius-3 triangle context.** This contains radii 1 and 2 and adds paths
   through residues three positions from either endpoint.

For each possible intermediate residue `k`, both triangle legs—`(i,k)` and
`(k,j)`—are retained in separate, consistently ordered feature slots. They are
not averaged together, so the probe can learn different roles for the two
legs. A validity indicator accompanies each slot. When a neighbor would fall
outside the protein or coincide with `i` or `j`, its edge features are filled
with zeros and its validity indicator is set to false. If two endpoint-relative
slots happen to name the same intermediate residue, both slots are retained;
this preserves their meanings relative to the two endpoints.

The resulting feature widths are 2 for the scalar baseline, 128 for the center
pair, and 1,156, 2,184, and 3,212 for triangle radii 1, 2, and 3.

### Classification target and model

The signed change is converted into three classes:

- **decrease:** change is less than −0.5 Å;
- **no change:** change is between −0.5 and +0.5 Å, including both boundaries;
- **increase:** change is greater than +0.5 Å.

Each feature set is fitted with an L2-regularized linear Ridge classifier. The
model is intentionally simple: the aim is to measure linearly accessible
information rather than maximize predictive performance with a nonlinear
model. Training weights first give every protein equal total influence and
then balance the three classes according to their protein-weighted frequency.
This prevents both long proteins and the most common update direction from
overwhelming the fit.

The classification evaluation reports overall accuracy, balanced accuracy,
and macro-F1, as well as precision, recall, F1, and prevalence for decrease,
no-change, and increase separately. Confusion matrices show which update types
are mistaken for one another. The primary comparison between input variants
uses held-out macro-F1 and balanced accuracy rather than raw accuracy, because
the three classes can be highly imbalanced and their prevalence changes across
layers.

### Continuous-change regression model

The same five feature sets are also fitted with L2-regularized Ridge regression
to predict the signed change directly. Positive predictions mean an increase
in decoded distance and negative predictions mean a decrease.

Evaluation includes R², mean absolute error, root mean squared error, Pearson
correlation, and average signed bias. R² is the central measure of how much
variation in the update is explained, while the error measures retain the
physical Ångström scale. Both pair-pooled and mean per-protein versions are
saved.

### Fitting details and parallel execution

Feature scaling is computed from training proteins only. The mean and variance
give every training protein equal influence, and all matrices and fitting
calculations use `float64`. Both probes use the iterative LSQR Ridge solver
with a fixed relative regularization strength of `0.001`, tolerance `0.001`,
and a maximum of 30 iterations. The regularization is fixed in advance; this
rough pilot does not tune it on the test set or run a hyperparameter search.

The Delta job is a 15-task array: three transitions crossed with five feature
sets. A task constructs its feature matrix once, fits both the classifier and
regressor, and saves classifier outputs before beginning regression so partial
work survives a later failure. Each task records feature-building, scaling,
fitting, evaluation, and total runtime. A dependent aggregation job runs only
after the complete array succeeds and refuses to merge missing or incomplete
shards.

## Main pilot outputs

Task-specific files are first written beneath
`shards/transition_XX_YY/<feature set>/`. The aggregator then creates one set of
combined files in `delta_files/local_pair_update_probe_pilot/`.

- **`classification_summary.csv`** is the main classifier comparison. It gives
  train and test performance for every transition and feature set, with both
  pair-pooled and protein-equal scores, class counts, and an indication of
  whether the iterative solver reached its limit.
- **`class_metrics.csv`** breaks classification performance down by decrease,
  no-change, and increase. It is the place to inspect whether a strong overall
  score hides failure on a less common update direction.
- **`confusion_matrix.csv`** records how often each true update class is
  predicted as each possible class, in both raw counts and within-class
  proportions.
- **`regression_summary.csv`** is the main continuous-target comparison. It
  reports explained variance, errors in Ångströms, correlation, and bias for
  the train and held-out protein sets.
- **`per_protein_metrics.csv`** shows classification or regression performance
  for each individual protein. It supports checks for consistency across
  proteins, sources, and protein sizes rather than relying only on pooled
  results.
- **`change_statistics.csv`** describes the target itself for each transition:
  its mean, spread, quantiles, absolute and RMS magnitude, and the proportions
  assigned to decrease, no-change, and increase at the 0.5 Å threshold.
- **`coefficients.csv`** saves the learned standardized linear weights. These
  can later be used to inspect which center channels, triangle legs, offsets,
  or validity indicators contribute to each class or to the continuous
  prediction.
- **`feature_schema.csv`** is the decoding guide for those coefficients. It
  maps each feature position back to the center pair, triangle shell,
  endpoint-relative intermediate residue, triangle leg, channel, or validity
  indicator.
- **`coefficients_used.csv`** preserves the exact pre-existing distance-probe
  coefficients used to define each target, while **`split_assignments.csv`**
  preserves the exact protein split.
- **`protein_manifest.csv`** records which proteins and pair-block files were
  processed, their split/source, residue and local-pair counts, and processing
  status. **`skipped_proteins.csv`** records any failures explicitly.
- **`timing.csv`** separates feature construction, standardization, fitting,
  evaluation, and total runtime for performance diagnosis.
- **`run_metadata.json`** records the complete run configuration and completion
  state. **`summary.txt`** gives a compact human-readable ranking of feature
  sets and their gains over the scalar and center-only baselines.

Full pair-level predictions are deliberately not saved by this pilot. The
summary and per-protein outputs preserve the main scientific comparisons while
avoiding a much larger, repetitive output dataset.

## Preliminary result from the completed pilot

The completed run shows a consistent ordering across the three transitions:
the scalar baseline performs worst, the center representation adds substantial
predictive information, and triangle context adds a further improvement. When
test scores are averaged equally across transitions, radius 3 obtains the best
pair-pooled macro-F1 (0.578) and balanced accuracy (0.626). Its mean held-out
regression R² is 0.603, compared with 0.516 for the center-only representation
and 0.127 for current distance plus sequence separation. Mean per-protein R²
follows the same ordering.

The gain from radius 1 to radius 3 is smaller than the gain from the scalar
baseline to the center pair, but it is consistent in the regression results
and usually in classification. This is preliminary evidence that nearby
triangle edges contain update information beyond the target pair itself.

All radius-1, radius-2, and radius-3 fits reached the configured 30-iteration
LSQR limit, whereas the smaller baselines did not. Their coefficients and
metrics are finite, but the context-model ranking should be treated as a pilot
result until convergence is checked with a higher iteration allowance or a
tighter solver study. This caveat matters especially when interpreting small
differences between the three radii.

## Consecutive-layer distance-control probe

The fixed-readout definition above is sensitive to shifts and rescaling of the
pair-representation channels between layers. The
`local_pair_distance_controls` Delta bundle therefore compares four predictive
definitions over transitions 10→11 through 15→16:

1. `fixed_raw_layer_x` is the original target, applying the saved layer-`x`
   probe to both raw representations.
2. `fixed_moment_aligned_to_x` maps layer `x+1` into layer-`x` coordinates
   using per-channel means and standard deviations estimated from training
   proteins only, then applies the saved layer-`x` probe.
3. `saved_own_layer` uses the saved probe associated with each endpoint.
4. `pooled_two_layer` fits one unregularized distance model jointly to both
   endpoint layers. Each training protein has equal total weight, divided
   equally between its two layer observations.

The separate `distribution_drift` diagnostic reports channelwise mean shifts,
scale ratios, standardized shifts, and their coefficient-weighted
contributions. It is not presented as a fifth predictive model and therefore
does not receive artificial R² values. As an internal identity check, the sum
of the layer-`x` coefficient times each channel's mean shift must reproduce the
mean raw fixed-readout change.

Every predictive variant is evaluated against structural Cα distances at both
endpoints, on both all unordered residue pairs and local pairs with sequence
separation 4–12. Outputs include held-out and training R²/error metrics,
predicted-distance distributions, change distributions, per-protein results,
pair-weighted versus protein-equal summaries, control agreement, fit and
alignment provenance, and held-out layer-trend plots. Raw pair-level records
are intentionally not saved. Pair-level means and error metrics are exact;
pair-level quantiles use bounded deterministic reservoir samples.

These controls answer different questions. Moment alignment removes marginal
per-channel location and scale drift but not channel rotations or nonlinear
changes. Own-layer probes preserve each layer's calibration but combine
representation and readout changes. The pooled model supplies one shared
coordinate system learned from both layers, at the cost of using information
from both endpoints when defining that coordinate system.

## Supporting distance-probe uncertainty pilot

`scripts/cluster_scripts/diagnostics/local_pair_change_se_pilot.py` addresses a
different question: whether an apparent layer-to-layer distance change is
stable to uncertainty in the weights of the original distance probe. It is not
part of the training or labeling procedure for the fixed-threshold main pilot.

For each transition from 10→11 through 15→16, the script resamples the 441
training proteins with replacement ten times. For every bootstrap sample, it
separately refits the full 256-feature directional distance model at both
endpoint layers using the structural Cα distances as targets. Each protein has
equal total influence regardless of its number of residue pairs. The same
resampled protein draw is used for both fits, and each refitted model is
applied only to its own layer, so every local residue pair receives ten paired
bootstrap estimates of its own-layer-probe distance change.

The per-pair standard error is the sample standard deviation of those ten
bootstrap changes. This is a protein-level cluster bootstrap: it captures how
the fitted distance readout changes when the training proteins change, while
respecting the strong dependence among residue pairs from the same protein.
The implementation avoids repeatedly materializing the full training problem
by accumulating each protein's normal-equation sufficient statistics once and
combining them according to its bootstrap multiplicity.

The SE pilot saves:

- the raw change distribution and disjoint change-size buckets, overall and by
  sequence separation;
- the distribution of pair-specific standard errors and approximate 95%
  normal intervals;
- class prevalence and the stability of decrease/no-change/increase labels at
  thresholds of 0.05, 0.1, 0.2, and 0.5 Å;
- one compressed record per local pair containing its two decoded distances,
  saved-model change, bootstrap-average change, estimated SE, interval, and
  change-to-SE ratio;
- per-protein summaries, the bootstrap draw assignments, refitted
  coefficients, fit rank/conditioning diagnostics, exact source coefficients,
  manifests, metadata, and a human-readable summary.

Because this was a small ten-bootstrap pilot, its SE estimates and nominal 95%
intervals are diagnostic rather than publication-quality uncertainty
estimates. It nevertheless checks whether the fixed 0.5 Å classification rule
is large relative to ordinary variation in the fitted distance weights.

## Interpretation boundaries

- These experiments demonstrate predictive information that is linearly
  accessible from layer-`x` representations. They do not prove that the model
  causally uses a particular triangle edge to produce the next update.
- The target is a change under an existing linear distance readout, not a
  directly observed physical movement and not a comparison to experimental
  distance change between layers.
- The main test split protects against protein-level leakage, but residue-pair
  observations within a protein remain correlated. Pair-pooled significance
  calculations must not treat all one million pairs as independent.
- Pre-MDS outputs may include negative distances and triangle-inequality
  violations. That is intentional here because the experiment targets changes
  in the raw readable signal before geometric projection.
- Differences between nearby context radii should be confirmed after resolving
  the current iteration-cap warning and, eventually, across more transitions
  or protein-family-aware splits.

# Canonical Methodology for Distance-Regression Probes

## Scope and status

This document defines the common methodology for the paper's residue-pair
distance regressions. It is a normative experimental protocol rather than a
description of every historical script. Existing implementations are assessed
against it in `regression_protocol_audit.md`.

The current canonical protocol version is **2.0**. It uses fivefold
out-of-fold (OOF) evaluation. Protocol 1.0 and its fixed 80/20 implementations
remain reproducible historical analyses; they are not the primary paper
performance estimate.

The protocol currently covers:

- one linear model shared across proteins;
- one linear model fitted separately within each protein;
- shared linear models gated by sequence separation; and
- one XGBoost model shared across proteins.

The common output files, metric definitions, geometry calculations, and
storage policy are specified in `regression_outputs.md`. This document fixes
what is fitted, which observations are used, and what kind of generalization
each analysis measures.

## Scientific interpretation

At each OpenFold Evoformer layer, a probe predicts the final C-alpha distance
between two residues from the frozen pair representation at that layer. Probe
performance measures how accessible the distance is to a specified readout. A
successful probe does not establish that OpenFold itself implements the fitted
regression, that the fitted feature directions are causal, or that the decoded
distances form an exact Euclidean structure.

The target is derived from OpenFold's final predicted PDB. The primary probe
therefore measures how strongly an intermediate representation anticipates the
model's own final structure. It is not, by itself, a measure of agreement with
an experimental structure.

The model families answer distinct questions:

| Model family | Scientific question | Generalization scope |
|---|---|---|
| Shared linear | Is one linear distance rule transferable across proteins? | Unseen protein IDs |
| Per-protein linear | Can a protein-specific linear rule predict pairs withheld from fitting within the same protein? | Unseen pairs within a seen protein |
| Gated shared linear | Does the transferable linear rule depend on a predefined pair category? | Unseen protein IDs |
| Shared XGBoost | Is transferable distance information accessible through nonlinear thresholds and feature interactions? | Unseen protein IDs |

The per-protein model is intentionally not a held-out-protein model. Its scores
must not be interpreted as directly equivalent to shared-model OOF scores.

Per-protein XGBoost pair splits are currently on hold and are not part of this
canonical protocol pending a separate methodological decision.

## Dataset and fixed analysis cohort

The default dataset is the single combined pool defined in `datasets.md`:

- 154 CAMEO proteins with TM-score greater than 0.7;
- 400 ECOD proteins with TM-score greater than 0.7; and
- 554 unique protein IDs before technical exclusions.

CAMEO and ECOD are not separate analysis groups. The source label is retained
for file resolution and auditing, but it does not determine model fitting or
the primary summaries.

One eligible-protein manifest must be frozen before any paper regression is
fitted. A protein is eligible only if:

1. its final predicted PDB can be resolved and parsed;
2. it contains at least two aligned C-alpha residues;
3. its PDB C-alpha count matches both residue dimensions of the pair tensor;
4. its reference-layer pair tensor is present, shape compatible, and has 128
   channels; and
5. its regression targets and reference-layer extracted features are finite.

It is not necessary to open and prevalidate all 48 pair tensors before fixing
the cohort. Each analysis must instead use the same frozen protein membership
at every requested layer and validate the loaded tensor and extracted features
at runtime. A protein may not be silently dropped at one layer and reappear at
another. If a frozen-cohort protein fails at a later layer, the layer is
incomplete until the input is repaired or the resulting coverage difference is
explicitly qualified. The manifest must record every initially included
protein, every initial exclusion, and a concise exclusion reason. The current
principal 48-layer analyses contain the same 552 usable proteins at every
layer, but the saved manifest, rather than this number in prose, is
authoritative.

The intended generalization claim is to unseen protein IDs. The protocol does
not group homologous proteins or structural families across splits. Therefore,
results should not be described as generalization to unseen sequence or fold
families.

## Canonical fivefold protein split

Every shared model uses the checked-in `regression_outer_folds_v1.csv`
manifest. It contains all 552 eligible proteins exactly once. The 111 proteins
in the historical locked-test set are retained unchanged as fold 0. The other
441 proteins are deterministically shuffled with seed 44 and stratified over
the same five rank-based protein-length bins into folds 1 through 4. The
manifest records source, residue and pair counts, length bin, outer fold, and
historical role for every protein. Its semantic digest is shared by every model
family.

For outer fold `f` in `0,1,2,3,4`, shared OLS, sequence-gated OLS, and
shared XGBoost fit on all proteins not assigned to `f` and predict all
unordered pairs for every protein assigned to `f`. Each eligible protein is
therefore evaluated exactly once per layer by each shared family.
Fold-specific scalers, global baselines, and sequence-gate baselines use only
the four training folds. No model is subsequently refitted on all proteins.

Fold 0 preserves historical membership for continuity, but it is not a newly
untouched test set. Fivefold OOF metrics combine all five held-out folds and
are the primary paper estimate. The historical `fit`, `validation`, and `test`
roles remain provenance only and do not change protocol-2.0 fitting.

## Regression observations and targets

For a protein with pair representation

\[
Z^{(\ell)} \in \mathbb{R}^{R \times R \times 128},
\]

one regression observation is constructed for each unordered residue pair.
The canonical implementation enumerates the strict lower triangle using
`np.tril_indices(R, k=-1)`. Thus every enumerated pair satisfies \(i>j\), every
unordered pair appears exactly once, and self-pairs are excluded.

Let \(c_i\in\mathbb{R}^3\) be the final predicted C-alpha coordinate for
residue \(i\). The target is

\[
y_{ij}=\lVert c_i-c_j\rVert_2.
\]

Targets are absolute distances in Angstroms. They are not transformed,
clipped, or standardized for fitting. Pair ordering, target ordering, and fold
assignments must remain identical across layers.

Unless a model-specific computational cap is declared, all unordered pairs are
used. Spatial distance may be used to form evaluation subgroups, but it must
not determine training membership, sampling, or gates in the primary distance
regressions.

## Canonical representation features

The base feature for layer \(\ell\) concatenates the two directional central
pair entries:

\[
x_{ij}^{(\ell)} =
\left[Z_{ij}^{(\ell)} \;\|\; Z_{ji}^{(\ell)}\right]
\in \mathbb{R}^{256}, \qquad i>j.
\]

The order is fixed: the strict-lower-triangle direction comes first and its
reverse comes second. Concatenation retains directional information that would
be discarded by averaging. Predictions are scalar distances and may be placed
symmetrically into both \((i,j)\) and \((j,i)\) entries of a reconstructed
distance matrix.

All primary model families use these same 256 features. Symmetric averaging,
diagonal residue features, axial neighbors, raw sequence separation, protein
length, amino-acid one-hot vectors, and other context variables define separate
feature-ablation analyses and are not part of the base representation.

Gating variables route a pair to a model; they are not appended to the 256
features unless a separately named feature-augmentation experiment explicitly
does so.

## Numerical precision and preprocessing

Pair tensors may be stored on disk or held in a temporary cache as `float32`.
Before feature moments, scaling, loss construction, or dense least-squares
fitting are computed, materialized feature blocks and targets must be converted
to `float64`. In particular, products used to calculate second moments must be
formed after conversion rather than squared in `float32` and accumulated in
`float64`.

For every OLS model:

1. estimate the feature mean and scale from that model's fitting observations
   only;
2. calculate those moments using the same protein-equal weighting used by the
   regression loss;
3. replace a scale below `1e-12` with 1;
4. reuse the fitted scaler without alteration for held-out rows; and
5. retain both standardized-space and raw-feature-space coefficients and
   intercepts.

XGBoost receives unstandardized `float64` feature and target arrays. Tree
standardization is unnecessary. Its histogram implementation may quantize
values internally, so the implementation must record the XGBoost version,
`tree_method`, and relevant numerical parameters rather than claiming that all
internal tree calculations remain 64-bit.

## Shared training objective and protein weighting

For shared models, proteins receive equal total fitting weight. If fitting
protein \(p\) contributes \(n_p\) selected pairs, each of its rows receives
weight proportional to \(1/n_p\). The resulting squared-error objective is

\[
\mathcal{L} =
\frac{1}{P}\sum_{p=1}^{P}
\frac{1}{n_p}\sum_{(i,j)\in p}
\left(y_{pij}-\hat y_{pij}\right)^2.
\]

This is a protein-equal mean-squared-error objective. It prevents a protein
from dominating training merely because its number of residue pairs grows
quadratically with length. It is not identical to directly maximizing mean
per-protein R-squared, which also normalizes each protein by target variance.

Evaluation summaries may later include both protein-equal and pair-pooled
views, but evaluation weights never change a fitted model after its held-out
fold has been opened.

## Shared linear model

At each layer, fit one ordinary least-squares model with an intercept:

\[
\hat y_{ij}=\beta_0 + x_{ij}^{\mathsf T}\beta.
\]

For each outer fold, the model is fitted on all pairs from the other four folds
using the protein-equal objective and that fold's train-derived scaler. No
ridge penalty, feature selection, pair cap, or target transformation is used.
Five models are fitted at every one of the 48 layers.

When the fitted solver exposes them directly, the implementation saves design
rank and singular-value extrema and derives the condition number. It must not
run an additional SVD or other matrix factorization solely to populate these
diagnostics. A rank-deficient or severely ill-conditioned solution may be
retained diagnostically but must be flagged rather than presented as an
unqualified stable OLS result. A future solver that does not expose these
quantities records them as unavailable rather than recomputing them at
substantial cost.

Linear OLS fits use a dense, rank-revealing `float64` least-squares solve,
currently scikit-learn `LinearRegression` backed by SciPy `lstsq`; primary fits
must not explicitly form normal equations. Loading mode, temporary `float32`
caches, chunk boundaries, and copy behavior are implementation details provided
they preserve the selected rows, weights, fitted scaler, and final `float64`
objective. Predictions are evaluated without clipping or other post-processing.

## Per-protein linear models

For each eligible protein, deterministically shuffle its canonical unordered
pair indices using the frozen protein-ID-derived seed and divide them as evenly
as possible into five pair folds. The checked-in compact
`per_protein_pair_folds_v1.csv` records the algorithm, seed, five counts, and an
assignment digest for exact regeneration; it does not expand to one row per
pair. The split is reused at every layer.

For each protein-layer combination:

1. estimate a scaler from the four training pair folds only;
2. fit one OLS model with an intercept on those training pairs;
3. predict only the held-out pair fold; and
4. repeat for all five folds and concatenate the five disjoint prediction
   vectors in canonical pair order.

The complete OOF vector supplies one authoritative regression result and one
geometry reconstruction per protein-layer. No pair receives more than one
prediction, fold-level predictions are never averaged, and training
predictions never enter OOF metrics or geometry. Compact fold-level training
metrics remain diagnostic. The protein's outer-fold label is reporting
metadata only and does not affect any within-protein fit.

OLS support must be auditable for every protein. Solver-exposed rank and
conditioning diagnostics are retained when available, without an additional
factorization. Fits with fewer training rows than model parameters, deficient
reported rank, or extreme reported condition number must be flagged and must
not be silently pooled with stable fits.

## Gated shared linear models

A gated probe assigns each pair to a predefined category and fits an independent
shared OLS model within each category. Gate definitions use sequence or residue
metadata only; the target distance never determines which model is trained.
Each gate uses the canonical 256 directional features, four outer training
folds, a gate- and fold-specific train-only scaler, and protein-equal weighting
among proteins that contribute to that gate.

### Sequence-separation gates

For intrachain pairs, sequence separation is \(|i-j|\). The six fixed gates
are:

| Gate | Sequence separation |
|---|---|
| `1` | exactly 1 |
| `2-4` | 2 through 4 |
| `5-9` | 5 through 9 |
| `10-19` | 10 through 19 |
| `20-49` | 20 through 49 |
| `50+` | 50 or greater |

Sequence separation is undefined across chains. Such pairs are explicitly
flagged rather than assigned a separation from concatenated array indices.

### Historical amino-acid-pair gates

Protocol 1.0 also evaluated amino-acid-pair and joint gates. The 20 standard
amino acids are paired as unordered combinations with
replacement, producing 210 gates. For example, `ALA-GLY` and `GLY-ALA` route
to the same gate even though the underlying directional pair features remain
ordered. Pairs containing a noncanonical or unmapped residue are explicitly
excluded from amino-acid-dependent analyses and counted.

### Historical joint gates

The amino-acid-pair by sequence-separation model contains up to
\(210\times6=1260\) gates per layer. A gate is eligible only when its training
data contain at least 257 pair rows, at least five contributing proteins, and
finite features and targets. Fitted gates retain solver-exposed rank and
condition diagnostics without an additional diagnostic factorization.

### Unresolved sparse-gate coverage policy

Some joint gates have too little training support to fit 256 OLS coefficients.
The current results contain 510 such failed layer-gate fits out of 7,560
requested fits. A test pair routed to one of these gates consequently has no
joint-model prediction. This is what a fallback policy would address.

Three defensible policies are possible:

1. **No fallback:** leave the pair unpredicted, report coverage, and restrict
   claims to supported gates. Whole-protein metrics are then not directly
   comparable with models evaluated on every pair.
2. **Hierarchical fallback:** use the joint model when supported, otherwise the
   corresponding sequence-separation model, and finally the global shared
   model if needed. This yields complete predictions but evaluates a composite
   model rather than a pure joint-gate bank.
3. **Matched exclusion:** omit unsupported pairs from every comparator when
   calculating a particular comparison. This preserves a pure joint model but
   changes the evaluation population.

No option is part of protocol 2.0. Historical joint
gate-level results remain valid with their support counts, but recombined
whole-protein joint-model results must be labeled incomplete and include exact
test-pair coverage. The sequence-only and amino-acid-only models do not
currently exhibit this fitting failure in the saved six-layer run.

## Shared XGBoost model

At each layer, fit an XGBoost regressor with squared-error objective on the same
canonical 256 directional features. Use protein-equal sample weights. Do not
append protein size, sequence separation, or other context features to the
primary model.

Training is computationally capped at a deterministic maximum of 5,000 pairs
per fitting protein. A protein with fewer pairs contributes all of them. The
sample is fixed by protein ID and pair indices, reused across layers and
the fixed configuration, and drawn without using target distance. Held-out-fold
evaluation uses all available unordered pairs.

The production configuration is frozen to S4, 5,000 sampled training rows per
protein, and exactly 700 boosting rounds. One target- and layer-independent
sample is generated per protein and reused whenever that protein belongs to an
outer training set. Fold `f` uses model seed `42 + f`, no early stopping, and
no `eval_set`. The held-out fold is not loaded for evaluation until its booster
checkpoint exists.

S4/5k/700 was selected through preliminary work on this same cohort. The source
decision table is
`delta_files/xgboost_hyperparameter_search/shared/winner_cap_sweep/cross_layer_selection_table.csv`;
its path and digest must be retained in provenance. The preliminary artifacts
also exposed the historical test proteins.

## Layer comparisons and selection

The same cohort, pair definition, feature definition, weighting, and model
procedure are applied independently at layers 0 through 47. A model family is
not a complete layer-trajectory analysis if it covers only a selected subset.
Sparse anchor-layer experiments may be retained as exploratory analyses.

OOF performance may be plotted across layers. The layer with the highest OOF
score is a descriptive maximum, not a validation-selected confirmatory optimum.
All 48 layers remain the reported trajectory.

## Reproducibility and implementation requirements

Every paper regression must record or enforce:

- the eligible-protein manifest and exclusion reasons;
- the frozen outer-fold and compact pair-fold manifests and their digests;
- deterministic pair ordering and any pair-sampling or pair-split digest;
- layers, feature names, feature order, target definition, and precision mode;
- scaler provenance proving that only fitting observations were used;
- model class, full configuration, random seeds, and library versions;
- per-protein row counts and effective training weights;
- finite-value, tensor-shape, residue-alignment, and split-disjointness checks;
- solver-exposed OLS rank, singular values, derived condition number, and
  explicit failure status, without an additional diagnostic factorization;
- XGBoost selection provenance and fixed final boosting rounds;
- explicit prediction coverage for gated models;
- timestamped per-layer timing and checkpointed outputs; and
- a stable machine-readable configuration digest.

The protein manifest and split logic should have one canonical implementation.
Scripts should import it rather than copying protein lists or independently
reconstructing supposedly identical splits.

## Admission of future analyses

A new regression may enter the primary comparison only after documenting:

1. its scientific question;
2. whether it is shared across proteins or fitted within proteins;
3. its exact input features and any routing variables;
4. its training observations, weighting, loss, and sampling;
5. its tunable hyperparameters and validation-only selection rule;
6. its final refitting procedure;
7. its OOF evaluation population and prediction coverage;
8. its layer coverage; and
9. every intentional deviation from this protocol.

Analyses with different targets, such as salt-bridge classification, require
their own task-specific protocol. Analyses that replay frozen distance
predictions, such as secondary-structure or MDS evaluations, inherit the
regression provenance but are not themselves additional regression fits.

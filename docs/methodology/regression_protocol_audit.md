# Regression Protocol Audit and Methodology Gaps

## Purpose and audit boundary

This document compares the active regression implementations and saved result
families with `regression_methodology.md`. It is a code-and-artifact audit as of
2026-08-27, not a claim that historical results are invalid. A result can remain
useful as exploratory evidence while requiring a rerun before entering the
fully controlled paper comparison.

The summary table assesses the active implementation. Its disposition column
and the family-specific text separately identify whether canonical production
artifacts have actually been generated and audited.

Output schemas and output-specific implementation gaps are assessed separately
in `regression_output_audit.md`.

Status labels:

- **Meets:** implements the protocol requirement.
- **Partial:** implements the main idea but differs in a material detail.
- **Gap:** does not implement the requirement.
- **Not applicable:** the requirement does not apply to that analysis.
- **Pending decision:** the protocol itself is not yet settled.

Protocol 2.0 fivefold OOF is now canonical. All existing 80/20 runners and
saved result families described below are reclassified as historical
protocol-1.0 analyses. Stage 1 has frozen and validated the shared outer- and
pair-fold metadata, but no v2 model runner or production result exists yet.

## Protocol-2.0 implementation status

| Family | Frozen folds | V2 runner | Production artifacts | Disposition |
|---|---|---|---|---|
| Shared linear OLS | Meets | Pending Stage 3 | Not generated | Historical 80/20 output remains available; not the primary estimate |
| Per-protein linear OLS | Meets | Pending Stage 4 | Not generated | Historical single 80/20 pair split remains available; not the primary estimate |
| Sequence-gated OLS | Meets | Pending Stage 5 | Not generated | Historical 80/20 output remains available; not the primary estimate |
| Shared XGBoost | Meets | Pending Stage 6 | Not generated | Historical fixed-model output remains available; preliminary same-cohort selection and test exposure remain disclosed |

The checked-in v2 metadata contains 552 eligible proteins, preserves the 111
historical test proteins as fold 0, assigns all other proteins once across
folds 1–4 with seed 44, and records exact fivefold pair-assignment digests for
every protein. Stage 1 does not establish model-fitting compliance.

## Protocol checklist for the historical implementation audit

The short identifiers below are used throughout the audit.

| ID | Protocol requirement |
|---|---|
| P1 | Frozen combined CAMEO+ECOD cohort with stable membership across analyzed layers; runtime failures are explicit |
| P2 | Canonical master split with locked test IDs |
| P3 | One strict-lower-triangle row per unordered pair and final-PDB C-alpha distance target |
| P4 | Directional central features concatenated as 256 channels |
| P5 | `float64` before feature statistics and model fitting |
| P6 | Preprocessing estimated from fitting observations only |
| P7 | Protein-equal objective for shared models |
| P8 | OLS for linear probes, with rank and conditioning diagnostics |
| P9 | Train and out-of-sample evaluation kept separate |
| P10 | Complete 48-layer trajectory for primary comparisons |
| P11 | No test calculation or inspection during model selection |
| P12 | Saved manifests, split/configuration digests, versions, and explicit failures |
| P13 | Complete or explicitly qualified prediction coverage for gated models |

## Historical protocol-1.0 summary by principal model family

| Analysis | P1 cohort | P2 split | P3 pairs/target | P4 base256 | P5 precision | P7 weighting | P8 model | P9 evaluation | P10 layers | P11 test lock | Overall disposition |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Shared linear OLS | Meets | Partial | Meets | Meets | Partial | Meets | Partial | Meets | Meets | Meets | Historical v1 80/20 baseline |
| Per-protein linear OLS | Meets | Meets | Meets | Meets | Meets | Not applicable | Meets | Meets | Meets | Not applicable | Historical v1 single-pair-split analysis |
| Sequence-gated OLS | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Historical v1 80/20 gated analysis |
| AA-pair-gated OLS | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Historical v1 analysis; excluded from the primary v2 family set |
| AA-pair x sequence-gated OLS | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Historical v1 analysis; excluded from the primary v2 family set |
| Shared XGBoost | Meets | Meets | Meets | Meets | Meets | Meets | Not applicable | Meets | Meets | Meets | Historical v1 fixed-model analysis with disclosed test exposure |

## Shared linear OLS

Primary implementation:

- `scripts/cluster_scripts/generalized_model_ecod/general_train_test_common.py`
- `scripts/cluster_scripts/generalized_model_ecod/general_base256_train_test_layers_00_23.py`
- `scripts/cluster_scripts/generalized_model_ecod/general_base256_train_test_layers_24_47.py`
- saved family: `delta_files/general_base256_combined/`

### What already aligns

- **P3/P4:** Uses the strict lower triangle, excludes the diagonal, predicts
  final-PDB C-alpha distance, and concatenates both 128-channel directions.
- **P6:** Fits one scaler from training proteins and reuses it for test
  proteins.
- **P7:** Gives every training protein equal total row weight.
- **P8 — Partial:** Fits scikit-learn `LinearRegression` OLS with an intercept
  and saves standardized and raw-scale coefficients.
- **P9:** Saves separate training and held-out-protein metrics.
- **P10:** The paired launchers cover layers 0 through 47.
- The saved outer split contains 441 training and 111 test proteins. Its
  membership exactly matches the saved gated-suite outer split.
- **P1:** The completed saved trajectory contains the same 552 proteins at
  every one of the 48 layers.

### Gaps and required changes

- **P2 — Partial:** The current 80/20 outer split is suitable as the canonical
  development/test division, but it has no saved `fit`/`validation` subdivision.
  Add that subdivision to the master manifest without changing the outer test
  membership.
- **P5 — Partial:** Final matrices and targets are fitted in `float64`, but the
  shared feature cache and stored targets are `float32`. More importantly,
  scaler second moments use `chunk * chunk` while `chunk` is still `float32`.
  Cast before multiplication to meet the numerical protocol exactly.
- **P8 — Partial:** The fit does not save the design rank and singular-value
  extrema already exposed by scikit-learn or the condition number derived from
  them. No additional SVD is required.
- **P12 — Partial:** Split assignments and substantial diagnostics are saved,
  but the current workflow lacks a complete configuration/version manifest and
  a split digest enforced by every entrypoint.
- The dataset protein IDs are embedded in the Python helper as well as listed
  in `datasets.md`. Replace duplicated sources of truth with one importable
  canonical manifest.

### Checklist

- [ ] Preserve the existing 441/111 outer membership in the master split.
- [ ] Add deterministic `fit` and `validation` roles within the 441 development proteins.
- [ ] Form scaler moments after casting feature blocks to `float64`.
- [ ] Save solver-exposed rank and singular-value diagnostics plus the derived
  condition number per layer; do not run an additional SVD.
- [ ] Save and validate manifest, split, configuration, and version digests.
- [ ] Retain runtime validation and fail explicitly rather than silently
  changing protein membership at an individual layer.

## Per-protein linear OLS

Primary implementation:

- `scripts/cluster_scripts/per_protein_scripts/per_protein_base256_common.py`
- `scripts/cluster_scripts/per_protein_scripts/per_protein_base256_array.py`
- saved family: `delta_files/per_protein_base256_combined/`

### What already aligns

- **P1/P2:** Loads and validates the frozen 552-protein manifest, attaches the
  canonical master role and development/test label, and retains explicit
  protein-layer failures rather than silently changing membership.
- **P3/P4/P5/P6:** Uses all strict-lower-triangle pairs, directional base256
  `float64` features and targets, and a scaler estimated from training pairs
  only.
- Uses the canonical protein-ID-derived 80/20 pair split at every layer and
  records per-protein counts, seed, and split digest.
- **P8:** Fits intercept-enabled dense OLS and saves solver-exposed rank,
  singular extrema, condition number, and explicit underdetermined,
  rank-deficient, or ill-conditioned flags. Flagged fitted units are included
  in aggregates under the manifest-recorded `include_with_flags` policy.
- **P9:** Saves distinct long-format `train_pairs`, `heldout_pairs`, and clearly
  mixed-scope `all_pairs` rows. Geometry is explicitly all-pairs.
- **P10/P12:** The array and legacy batch entrypoints cover all 48 layers;
  canonical layer manifests retain split/configuration digests, versions, code
  revision, failure status, and replayable model provenance.
- Direct shared-versus-per-protein analysis now filters to locked test protein
  IDs.

### Remaining production action

The active implementation is protocol-complete, but the current copied
`delta_files/per_protein_base256_combined/` artifacts predate this migration.
They remain historical evidence until the 48-layer Delta rerun and canonical
consolidation finish.

### Checklist

- [x] Add the master protein role to per-protein outputs.
- [x] Keep separate train-pair and held-out-pair metrics, and label combined
  all-pair fields as mixed-scope descriptive metrics.
- [x] Define and save OLS eligibility/conditioning status before aggregation.
- [x] Save pair-split counts and a deterministic split digest.
- [x] Restrict direct shared-versus-per-protein summaries to master test protein IDs.
- [x] Label geometry reconstruction as using all training and held-out predictions.
- [ ] Complete the Delta production rerun and audit the consolidated artifacts.

## Gated shared linear OLS suite

Primary implementation:

- `scripts/cluster_scripts/gated_linear_probes/prepare_gated_linear_probe_suite.py`
- `scripts/cluster_scripts/gated_linear_probes/gated_linear_probe_common.py`
- `scripts/cluster_scripts/gated_linear_probes/sequence_gated_linear_probes.py`
- `scripts/cluster_scripts/gated_linear_probes/aa_pair_gated_linear_probes.py`
- `scripts/cluster_scripts/gated_linear_probes/aa_sequence_gated_linear_probes.py`
- `scripts/cluster_scripts/gated_linear_probes/consolidate_joint_gated_layer.py`
- `scripts/cluster_scripts/gated_linear_probes/aggregate_gated_linear_probe_suite.py`
- historical saved family: `delta_files/consolidated/`

### Historical v1 implementation

- **P1/P2:** Preparation imports and verifies the frozen 554-entry canonical
  manifest and split digest, uses all 552 eligible proteins in production, and
  preserves the 330 `fit`, 111 `validation`, and 111 locked `test` roles. A
  later-layer input failure leaves the task/layer incomplete rather than
  changing cohort membership. Smoke mode selects a deterministic subset of
  those frozen roles without creating a new split.
- **P3/P4:** Pair indices and directional base256 extraction come from
  `regression_protocol.py`, giving one strict-lower-triangle target row and
  ordered `[Z_ij | Z_ji]` features per unordered pair.
- **P5/P6:** The temporary feature cache is explicitly labeled `float32`, but
  every gate matrix, target, scaler moment, weight, loss, and OLS fit is
  materialized in `float64`. Gate-specific scalers use only the combined
  development fitting population and are replayed unchanged for locked test
  proteins.
- **P7:** `protein_equal_weights` and `ProteinEqualFeatureMoments` implement
  equal total weight per contributing development protein within every gate.
- **P8:** The fixed eligibility rules remain at 257 training rows and five
  contributing proteins. Fitted scikit-learn OLS models retain solver-exposed
  rank, singular extrema, derived condition number, and stability flags without
  an additional diagnostic factorization.
- **P9/P11:** Fixed models are fitted on `fit + validation`; recombined
  development and locked-test results are evaluated separately. No gate,
  feature, layer, or tuning choice uses test outcomes.
- **P10:** Production task mappings and launchers cover layers 0 through 47;
  smoke mode covers layer 0 only.
- **P12:** Preparation, task, layer-shard, and final run manifests enforce the
  canonical split and family-configuration digests, record versions and feature
  order, checkpoint completed gates/layers, and retain explicit failed units.
- **P13:** The joint model uses the selected no-fallback policy. Unsupported
  gates remain unpredicted, and recombined rows report exact partial coverage.

### Historical evidence and remaining run action

The existing six-layer saved family predates the canonical migration. Its
conditioning diagnostics remain useful historical evidence:

- all 36 sequence-gate fits have rank 256, with condition numbers from 32.9 to
  101.8;
- all 1,260 amino-acid-pair fits have rank 256, with condition numbers from
  46.7 to 166.9; and
- 7,050 joint AA-pair-by-sequence fits have rank 256, but their condition
  numbers range from 34.0 to 7,236 (median 86.1; 95th percentile 187.8).

Thus the diagnostics mostly confirm stable full-rank fits for the first two
families while identifying a small, weakly supported tail in the joint family.
They are quality-control metadata rather than primary scientific outcomes.

The migrated v1 implementation has no remaining v1 gated-methodology code gap,
but it is not a protocol-2.0 runner. The historical joint counts above must not
be assumed to describe any future v2 run.

### Checklist

- [x] Replace the six-layer constant with a complete 0-47 production trajectory.
- [x] Reuse the frozen 552-protein cohort and canonical master split.
- [x] Fit final fixed OLS gates on the combined 80% development proteins.
- [x] Use no fallback for sparse joint gates and report exact partial coverage.
- [x] Keep exact support and failure counts for every requested gate.
- [ ] Complete the Delta production rerun and audit the resulting 48-layer artifacts.

## Shared XGBoost

Primary implementation:

- `scripts/cluster_scripts/xgboost_search/xgboost_search_common.py`
- `scripts/cluster_scripts/xgboost_search/shared_xgboost_hyperparameter_search.py`
- `scripts/cluster_scripts/xgboost_search/shared_xgboost_winner_cap_sweep.py`
- `scripts/cluster_scripts/xgboost_search/shared_xgboost_final_common.py`
- `scripts/cluster_scripts/xgboost_search/shared_xgboost_final_layer.py`
- `scripts/cluster_scripts/xgboost_search/aggregate_shared_xgboost_final.py`
- historical selection family: `delta_files/xgboost_hyperparameter_search/shared/`
- pending final family: `delta_files/shared_xgboost_base256_combined/`

### Historical v1 implementation status

- **P1/P2:** The runner enforces the frozen 552-protein cohort, the canonical
  330/111/111 master roles, and explicit layer failure rather than silent
  membership changes.
- **P3/P4/P5:** Every observation is a strict-lower-triangle C-alpha-distance
  row with ordered directional base256 features, and all model inputs, targets,
  weights, predictions, baselines, and metric accumulation are `float64`.
- **P7:** Each of the 441 development proteins has total fitting weight one
  after its deterministic protein-ID-derived 5k cap.
- **P9/P10:** A fresh fixed S4 model is fit on `fit + validation` for exactly
  700 rounds at each of all 48 independently required layers. Development and
  locked-test all-pair outputs remain distinct.
- **P11:** Current search and cap-sensitivity code loads only `fit` and
  `validation` roles and rejects search schemas containing test fields. The
  final runner validates frozen selection, fits the booster, and only then
  opens test records. The older copied artifacts did expose the canonical test
  IDs; this historical limitation is explicit in the frozen provenance and
  prevents a claim that the test set was never inspected.
- **P12:** Configuration, master-manifest, split-assignment, sample-policy,
  code, input, artifact, replay, and version information is saved and checked;
  failures are written explicitly and incomplete shards cannot consolidate.

Local compilation, focused tests, and launcher syntax checks cover the active
implementation. The 48-layer Delta production rerun is still pending, so the
final artifact family is not yet complete in storage.

### Checklist

- [x] Reuse the canonical master `fit`, `validation`, and `test` protein roles.
- [x] Use ordered directional base256 features for the final model.
- [x] Freeze S4/5k/700 from validation-only decision tables.
- [x] Prevent search and cap-sensitivity code from loading or reporting test data.
- [x] Refit each final layer model on the combined 80% development proteins.
- [x] Evaluate test records only after selection validation and final fitting.
- [x] Require the final selected model at all 48 layers in code and aggregation.
- [ ] Complete the Delta production rerun and audit the final stored family.

## Other regression and diagnostic analyses

These analyses are useful but should not be presented as members of the
canonical four-family comparison without being redesigned.

### Per-protein XGBoost search

Files under `scripts/cluster_scripts/xgboost_search/per_protein_*` fit XGBoost
within 50 selected proteins at layers 0, 23, and 47 using 60/20/20 pair splits
and averaged 128-channel features. The paper's stated XGBoost family is shared
only. Treat these results as exploratory and do not mix them with the canonical
per-protein OLS trajectory.

### Context-feature sweep

`diagnostic_context_feature_sweep_regression.py` is a layer-47 feature and
precision diagnostic. It uses its own 80/20 split, inner cross-validation for a
ridge variant, and feature sets containing averaged or concatenated pair
features plus optional axial, diagonal, amino-acid, sequence, and protein-size
context. It should be cited as an ablation that motivated the canonical base256
and `float64` decisions, not as a protocol-compliant primary model.

### Diagnostic decision tree and XGBoost regressions

`diagnostic_decision_tree_regression.py`,
`diagnostic_xgboost_regression.py`, and
`diagnostic_tree_regression_common.py` are layer-47 diagnostics. They use a
separate outer/inner split, a 5,000-pair cap, 128 directional channels plus six
context features, `float32` feature matrices, and legacy CAMEO-style input
roots. They do not satisfy the combined-dataset, base256, full-trajectory protocol.
Their role is exploratory evidence about feature routing and nonlinear
structure.

### Earlier sequence-separation diagnostic

`diagnostic_sequence_separation_gated_probes.py` uses base256 OLS and the same
outer split logic but only six layers and a 100-training-protein subset. It is
superseded by the full gated suite and should not supply primary paper numbers.

### K-fold result family

`delta_files/baseline256_kfold/` is a historical stability analysis. The
canonical protocol now also uses five folds, but this older artifact is not a
protocol-2.0 result: it covers 184 proteins and predates the frozen 552-protein
outer folds, canonical v2 precision/weighting rules, complete OOF outputs, and
v2 provenance. It may support a coefficient-stability statement only when
those differences are disclosed.

### Downstream analyses that are not new distance regressions

The following inherit predictions or coefficients from a distance model and
should preserve that model's split and provenance, but they are not additional
distance-regression families:

- exact triangle-violation and other geometry diagnostics;
- MDS reconstruction and TM-score analysis;
- distance-error geometry nulls;
- secondary-structure replay from frozen general or gated coefficients; and
- true-distance category statistics.

Salt-bridge prediction and other classification targets require separate
classification methodology because their labels, sampling, loss, and metrics
differ from continuous distance regression.

## Historical remediation list (superseded by the staged v2 refactor)

### Priority 0: required for a controlled model comparison

- [ ] Reuse one canonical frozen 552-protein manifest and make any runtime
  layer failure explicit.
- [ ] Preserve the existing 441/111 shared/gated outer split and add the
  deterministic fit/validation subdivision inside the 441 development proteins.
- [ ] Preserve separate per-protein training and held-out-pair reporting;
  clearly label any combined all-pair score as mixed-scope.
- [x] Rebuild shared XGBoost with the canonical split, base256 features,
  validation-only search, and 80% final refit; historical test exposure remains
  disclosed and a never-inspected-test claim is unavailable.

### Priority 1: required for complete trajectories

- [x] Extend all three gated OLS implementations from six layers to all 48
  (production rerun pending).
- [x] Extend selected shared XGBoost from three anchor layers to all 48 in code
  (production rerun pending).
- [x] Resolve sparse joint-gate handling with explicit no-fallback partial coverage.
- [ ] Make shared-linear scaler moments fully `float64`.

### Priority 2: provenance and presentation hardening

- [ ] Centralize dataset and split construction in one importable module.
- [ ] Add configuration/version manifests and digests to every principal family.
- [ ] Label exploratory diagnostics and historical result families clearly.
- [ ] Implement the common reporting and geometry contract defined in
  `regression_outputs.md`.

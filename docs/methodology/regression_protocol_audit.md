# Regression Protocol Audit and Methodology Gaps

## Purpose and audit boundary

This document compares the active regression implementations and saved result
families with `regression_methodology.md`. It is a code-and-artifact audit as of
2026-08-17, not a claim that historical results are invalid. A result can remain
useful as exploratory evidence while requiring a rerun before entering the
fully controlled paper comparison.

Output schemas and output-specific implementation gaps are assessed separately
in `regression_output_audit.md`.

Status labels:

- **Meets:** implements the protocol requirement.
- **Partial:** implements the main idea but differs in a material detail.
- **Gap:** does not implement the requirement.
- **Not applicable:** the requirement does not apply to that analysis.
- **Pending decision:** the protocol itself is not yet settled.

## Protocol checklist

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

## Summary by principal model family

| Analysis | P1 cohort | P2 split | P3 pairs/target | P4 base256 | P5 precision | P7 weighting | P8 model | P9 evaluation | P10 layers | P11 test lock | Overall disposition |
|---|---|---|---|---|---|---|---|---|---|---|---|
| Shared linear OLS | Meets | Partial | Meets | Meets | Partial | Meets | Partial | Meets | Meets | Meets | Closest current primary baseline; targeted rerun needed |
| Per-protein linear OLS | Meets | Partial | Meets | Meets | Meets | Not applicable | Partial | Meets | Meets | Not applicable | Pair split and separate train/test metrics are present; combined metrics are permissible supplementary descriptions |
| Sequence-gated OLS | Meets | Partial | Meets | Meets | Meets | Meets | Meets | Meets | Gap | Meets | Strong six-layer exploratory result; extend to 48 layers and master split |
| AA-pair-gated OLS | Meets | Partial | Meets | Meets | Meets | Meets | Meets | Meets | Gap | Meets | Strong six-layer exploratory result; extend to 48 layers and master split |
| AA-pair x sequence-gated OLS | Meets | Partial | Meets | Meets | Meets | Meets | Meets | Partial | Gap | Meets | Sparse-gate coverage policy unresolved |
| Shared XGBoost | Meets | Gap | Meets | Gap | Meets | Meets | Not applicable | Partial | Gap | Gap | Requires a new canonical implementation and rerun |

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

- **P3/P4/P5:** Uses all strict-lower-triangle pairs, directional base256
  features, and materializes the protein-layer feature matrix and target in
  `float64`.
- **P6:** Calculates the scaler from each protein's training pairs only.
- Uses a deterministic protein-ID-derived 80/20 pair split that is reused at
  every layer.
- Fits OLS with an intercept-equivalent centered-target implementation.
- Saves train-pair and held-out-pair R-squared as `train_r2` and `test_r2`.
- **P10:** Array and batch entrypoints cover all 48 layers.
- Saves rank, singular-value, and condition-number diagnostics.
- **P1:** The completed saved trajectory contains the same 552 proteins at
  every one of the 48 layers.

### Gaps and required changes

- **P2 — Partial:** Models are appropriately fitted within proteins rather than
  across master training proteins, but outputs do not identify the master
  protein role. Direct comparisons therefore cannot automatically restrict the
  summary to the shared model's locked test proteins.
- **P8 — Partial:** The implementation diagnoses underdetermined or unstable
  OLS fits but still fits and aggregates them without an explicit eligibility
  or primary-analysis exclusion status.
- **P9:** The implementation saves separate `train_r2` and `test_r2` fields as
  required. Its `per_protein_r2`, MSE, MAE, and RMSE fields combine all pairs;
  these are permissible supplementary mixed-scope descriptions but must remain
  clearly labeled. Geometry rows are likewise based on predictions over all
  pairs.
- **P12 — Partial:** The deterministic pair split can be reproduced, but the
  outputs do not save train/test pair counts, the pair-split seed/digest, or an
  explicit fit-status field for every protein-layer combination.

### Checklist

- [ ] Add the master protein role to per-protein outputs.
- [ ] Keep separate train-pair and held-out-pair metrics, and label combined
  all-pair fields as mixed-scope descriptive metrics.
- [ ] Define and save OLS eligibility/conditioning status before aggregation.
- [ ] Save pair-split counts and a deterministic split digest.
- [ ] Restrict direct shared-versus-per-protein summaries to master test protein IDs.
- [ ] Decide separately how geometry reconstruction should treat training-pair predictions.

## Gated shared linear OLS suite

Primary implementation:

- `scripts/cluster_scripts/gated_linear_probes/prepare_gated_linear_probe_suite.py`
- `scripts/cluster_scripts/gated_linear_probes/gated_linear_probe_common.py`
- `scripts/cluster_scripts/gated_linear_probes/sequence_gated_linear_probes.py`
- `scripts/cluster_scripts/gated_linear_probes/aa_pair_gated_linear_probes.py`
- `scripts/cluster_scripts/gated_linear_probes/aa_sequence_gated_linear_probes.py`
- saved family: `delta_files/consolidated/`

### What already aligns

- The saved run contains 552 analyzed proteins and uses the exact same 441/111
  outer protein membership as `general_base256_combined`.
- **P1:** Preparation freezes this cohort across every requested gated layer
  and records exclusions explicitly.
- **P3/P4:** Uses strict-lower-triangle targets and directional base256
  features.
- **P5/P6:** Materializes gate matrices in `float64`; scaling and fitting are
  train-only and `float64`.
- **P7:** Gives each contributing training protein equal total weight within a
  gate.
- **P8:** Uses OLS and saves gate eligibility, rank, singular values, condition
  number, and explicit failure flags.
- **P9:** Evaluates training and held-out proteins separately.
- **P11/P12:** Gate definitions are fixed, manifests and configuration/split
  digests are saved, and failures and exclusions are explicit.
- The saved six-layer run fitted all 36 sequence gates and all 1,260
  amino-acid-only layer-gates.

The saved conditioning diagnostics provide a useful numerical audit:

- all 36 sequence-gate fits have rank 256, with condition numbers from 32.9 to
  101.8;
- all 1,260 amino-acid-pair fits have rank 256, with condition numbers from
  46.7 to 166.9; and
- 7,050 joint AA-pair-by-sequence fits have rank 256, but their condition
  numbers range from 34.0 to 7,236 (median 86.1; 95th percentile 187.8).

Thus the diagnostics mostly confirm stable full-rank fits for the first two
families while identifying a small, weakly supported tail in the joint family.
They are quality-control metadata rather than primary scientific outcomes.

### Gaps common to all gated families

- **P2 — Partial:** The outer 80/20 membership matches the general model, but
  the new master `fit`/`validation` roles do not yet exist.
- **P10 — Gap:** Each family covers six representative layers rather than the
  complete trajectory.
- The preparation manifest labels numerical precision as `float64` while also
  using a temporary `float32` feature cache. This is acceptable under the new
  protocol because materialized gate matrices are converted before statistics
  and fitting, but the manifest should describe both storage and fitting
  precision explicitly.

### Joint-gate-specific gap

- **P13 — Pending decision/Partial:** The saved joint run requested 7,560
  layer-gate fits. It fitted 7,050 and marked 510
  `insufficient_train_rows`. Unsupported test pairs have no joint-gate
  prediction. Current recombined metrics therefore cover about 99.9% rather
  than exactly the same pair population as the global and sequence-gated
  models.
- Until a fallback or matched-exclusion policy is chosen, joint whole-protein
  scores must retain their coverage qualification.

### Checklist

- [ ] Replace the six-layer constant with a complete 0-47 production trajectory.
- [ ] Reuse the frozen 552-protein cohort and master protein split.
- [ ] Fit final fixed OLS gates on the combined 80% development proteins.
- [ ] Choose no fallback, hierarchical fallback, or matched exclusion for sparse joint gates.
- [ ] Enforce and report 100% coverage if hierarchical fallback is selected.
- [ ] Keep exact support and failure counts under every policy.

## Shared XGBoost

Primary implementation:

- `scripts/cluster_scripts/xgboost_search/xgboost_search_common.py`
- `scripts/cluster_scripts/xgboost_search/shared_xgboost_hyperparameter_search.py`
- `scripts/cluster_scripts/xgboost_search/shared_xgboost_winner_cap_sweep.py`
- saved family: `delta_files/xgboost_hyperparameter_search/shared/`

### What already aligns

- Uses the combined 552-protein usable cohort for the three anchor layers.
- **P1:** The saved search keeps this cohort fixed across all three analyzed
  layers.
- **P3/P5:** Uses one row per unordered pair, final-PDB C-alpha targets, and
  unstandardized `float64` feature/target matrices.
- **P7:** Uses protein-equal sample weights for shared fitting and validation.
- Uses deterministic, nested per-protein sampling and includes the accepted
  10,000-row cap.
- Hyperparameter winner selection is based on validation mean per-protein
  R-squared and uses a recorded deterministic tie-break procedure.
- Saves split digests, configurations, run metadata, and skipped-protein
  information.

### Gaps and required changes

- **P2 — Gap:** It constructs a separate 60/20/20 split. Its digest and protein
  membership differ from the shared/gated linear outer split, so the test set
  is not the locked canonical test set.
- **P4 — Gap:** It averages the two directions into 128 features. The protocol
  requires their ordered concatenation into 256 features.
- **P9 — Partial:** Search fits distinguish fit, validation, and test proteins,
  but the selected model remains an early-stopped 60%-fit model. It is not
  freshly refitted on the combined 80% development proteins.
- **P10 — Gap:** Only anchor layers 0, 23, and 47 are implemented and saved.
- **P11 — Gap:** Although test columns are not used by the selection function,
  `shared_xgboost_hyperparameter_search.py` calculates and writes test metrics
  for every candidate configuration. The locked test set is therefore exposed
  during search.
- **P12 — Partial:** The search records substantial configuration and split
  provenance, but its metadata does not record Python, NumPy, or XGBoost
  versions.
- The current selected-model sweep varies caps across 5k, 10k, and 20k. The
  canonical paper model should identify 10k as its fixed cap; other caps remain
  sensitivity analyses.

### Checklist

- [ ] Reuse the canonical master `fit`, `validation`, and `test` protein roles.
- [ ] Switch the primary feature matrix from averaged 128 to concatenated 256.
- [ ] Freeze one configuration using validation results only.
- [ ] Stop computing candidate test metrics during search.
- [ ] Save validation-derived boosting rounds for every layer.
- [ ] Refit each final layer model on the combined 80% development proteins.
- [ ] Evaluate the locked test proteins only after selection artifacts are frozen.
- [ ] Run and save the final selected model at all 48 layers with the fixed 10k cap.

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
current protocol does not use k-fold cross-validation for primary performance
estimation. It may support a coefficient-stability statement if its older
cohort, features, and precision are disclosed.

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

## Prioritized remediation order

### Priority 0: required for a controlled model comparison

- [ ] Reuse one canonical frozen 552-protein manifest and make any runtime
  layer failure explicit.
- [ ] Preserve the existing 441/111 shared/gated outer split and add the
  deterministic fit/validation subdivision inside the 441 development proteins.
- [ ] Preserve separate per-protein training and held-out-pair reporting;
  clearly label any combined all-pair score as mixed-scope.
- [ ] Rebuild shared XGBoost with the canonical split, base256 features,
  validation-only search, 80% final refit, and untouched test evaluation.

### Priority 1: required for complete trajectories

- [ ] Extend all three gated OLS families from six layers to all 48.
- [ ] Extend selected shared XGBoost from three anchor layers to all 48.
- [ ] Resolve sparse joint-gate prediction coverage.
- [ ] Make shared-linear scaler moments fully `float64`.

### Priority 2: provenance and presentation hardening

- [ ] Centralize dataset and split construction in one importable module.
- [ ] Add configuration/version manifests and digests to every principal family.
- [ ] Label exploratory diagnostics and historical result families clearly.
- [ ] Implement the common reporting and geometry contract defined in
  `regression_outputs.md`.

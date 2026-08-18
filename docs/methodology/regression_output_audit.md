# Regression Output Audit and Gaps

## Purpose and audit boundary

This document compares the active regression implementations and saved result
families with `regression_outputs.md`. It is an artifact-and-code audit as of
2026-08-18. Historical files remain useful, but they do not automatically meet
the final paper output contract.

The summary table assesses the active implementation. Its disposition column
and the family-specific text separately identify whether canonical production
artifacts have actually been generated and audited.

The audit assesses whether a result family saves the information needed for
consistent comparison. It does not require historical filenames to be renamed
before their results can be inspected.

Status labels are:

- **Meets:** the active implementation provides the required output; the
  disposition states separately when a production rerun is pending.
- **Partial:** the main information exists but its scope, schema, or storage
  differs materially.
- **Gap:** the active implementation does not provide the output.
- **Not applicable:** the requirement does not apply to that model family.

## Output protocol checklist

| ID | Output requirement |
|---|---|
| O1 | Versioned run manifest, canonical split digest, and explicit failures |
| O2 | All-protein, long-format `per_protein.csv` with common metrics and coverage |
| O3 | Compact authoritative `pooled_metrics.csv` without duplicated mean-per-protein fields |
| O4 | Locked-test pooled metrics by sequence separation, AA pair, and spatial distance |
| O5 | Gated whole-protein recombination without required per-gate or per-protein-bin expansions |
| O6 | Compact global and model-conditional baseline metrics; no redundant skill columns |
| O7 | Per-protein geometry and compact derived geometry summary |
| O8 | Geometry requires complete pair coverage; exclusions remain explicit |
| O9 | Replayable coefficients or boosters stored without avoidable CSV expansion |
| O10 | Per-fitted-unit status and OLS/XGBoost diagnostics |
| O11 | Brief per-layer timing |
| O12 | Validated layer shards and consolidated canonical outputs |

## Summary by principal model family

| Analysis | O1 provenance | O2 per protein | O3 pooled | O4 subgroups | O5 gated recombination | O7/O8 geometry | O9 artifacts | O10 status | O11 timing | Overall disposition |
|---|---|---|---|---|---|---|---|---|---|---|
| Shared linear OLS | Partial | Partial | Partial | Gap | Not applicable | Meets | Meets | Partial | Partial | Strongest current output family; schema normalization and subgroup output remain |
| Per-protein linear OLS | Meets | Meets | Meets | Meets | Not applicable | Meets | Meets | Meets | Meets | Canonical output implementation complete; 48-layer production rerun pending |
| Sequence-gated OLS | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Canonical output implementation complete; 48-layer production rerun pending |
| AA-pair-gated OLS | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Canonical output implementation complete; 48-layer production rerun pending |
| AA-pair x sequence-gated OLS | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Canonical no-fallback outputs complete in code; 48-layer production rerun pending |
| Shared XGBoost | Partial | Gap | Gap | Gap | Not applicable | Gap | Partial | Partial | Partial | Current saved artifacts are search outputs, not a canonical final refitted model family |

## Shared linear OLS

Primary implementation and saved family:

- `scripts/cluster_scripts/generalized_model_ecod/general_train_test_common.py`
- `delta_files/general_base256_combined/`

### What already aligns

- `per_protein.csv` covers all 552 proteins and all 48 layers, with separate
  train/test protein labels.
- It includes pair count, target variance, $R^2$, MSE, MAE, RMSE, true and
  predicted means, and bias. This is a strong basis for the common schema.
- `split_summary.csv` contains separate train/test pooled $R^2$, MAE, and
  RMSE plus mean-per-protein $R^2$.
- `geometry_metrics.csv` has exactly one row per protein-layer combination and
  contains MDS, eigenvalue, stress, negative-prediction, and exact triangle
  statistics.
- `geometry_summary.csv` supplies small layer/split summaries.
- `coefficients.csv` is compact and contains standardized and raw-scale
  coefficients for every layer.
- Layer-batch files are saved and consolidate to the expected 48 layers.

### Gaps and changes

- **O1 — Partial:** The result family has split assignments and summaries but
  lacks one final schema-versioned run manifest with library versions and a
  configuration digest.
- **O2 — Partial:** Rename metric fields to the common schema, add
  `n_expected_pairs`, `n_evaluated_pairs`, `pair_coverage`, and
  `prediction_status`, and omit redundant MSE in the next canonical run.
- **O3 — Partial:** `split_summary.csv` is close to `pooled_metrics.csv`, but it
  duplicates mean-per-protein $R^2$, lacks baseline metrics and explicit
  coverage, and uses train/test-specific wide column names rather than one row
  per split.
- **O4 — Gap:** Category accuracy exists only in separate diagnostics and not
  as canonical all-layer outputs with the fixed category definitions.
- **O10 — Partial:** Shared OLS does not save the rank and singular values
  already exposed by its fitted solver, a derived condition number, or a
  model-status row per layer. No additional SVD is required.
- **O11 — Partial:** Total elapsed seconds are present in the split summary and
  detailed stdout, but there is no small common per-layer runtime table.
- The exact triangle values are already reconciled into `geometry_metrics.csv`.
  The separate `exact_triangle_metrics/` copy is therefore redundant for a new
  canonical run.
- The six large pair/triangle sample CSVs under
  `triangle_violation_samples_test/` total roughly 256 MB and are not required
  by the canonical output contract.

### Checklist

- [ ] Emit common-schema `per_protein.csv` and `pooled_metrics.csv`.
- [ ] Add all three locked-test subgroup files for all 48 layers.
- [ ] Add `model_status.csv`, `runtime.csv`, and a schema-versioned manifest.
- [ ] Keep the existing geometry metrics while adding coverage/status fields.
- [ ] Stop producing duplicate exact-triangle and sampled-triangle files in the
  final result family unless a specific diagnostic requests them.

## Per-protein linear OLS

Primary implementation and saved family:

- `scripts/cluster_scripts/per_protein_scripts/per_protein_base256_common.py`
- `delta_files/per_protein_base256_combined/`

### What already aligns

- **O1/O12:** Layer shards and final aggregation validate the schema, frozen
  split, pair-split/configuration digests, versions, code revision, hashes,
  primary keys, expected row counts, and explicit failures before publishing a
  schema-versioned run manifest.
- **O2:** `per_protein.csv` uses the common long schema with three pair scopes,
  master development/test labels, status, coverage, and the four core metrics.
- **O3/O6:** `pooled_metrics.csv` has development/test rows for every scope plus
  the global development-training mean and per-protein training-mean baseline.
- **O4:** All three canonical subgroup files evaluate held-out pairs from
  locked-test proteins only.
- **O7/O8:** Canonical all-pair geometry rows and compact long-format summaries
  retain explicit coverage and failure status.
- **O9:** One compressed coefficient/scaler NPZ per layer plus a hashed compact
  index replaces the approximately 647 MB repeated coefficient CSV.
- **O10/O11:** One model-status row per protein-layer holds OLS support and
  diagnostics, while runtime is reduced to one row per layer.
- The canonical workflow does not emit redundant MSE, summary text,
  per-protein timings, or duplicate exact-triangle artifacts.

### Remaining production action

The active output implementation is complete. The currently copied result
family remains the historical pre-migration output and is not rewritten in
place; a new 48-layer Delta run and artifact audit are still required.

### Checklist

- [x] Emit the common long-format pair-scope table in canonical layer shards.
- [x] Add pooled metrics for train, held-out, and all-pair scopes.
- [x] Add the three locked-test subgroup outputs, using held-out pairs.
- [x] Replace the repeated coefficient CSV with indexed NPZ shards.
- [x] Add explicit prediction/fit status, coverage, and master protein role.
- [x] Consolidate timing to one row per layer.
- [ ] Complete the Delta production rerun and audit the consolidated artifacts.

## Gated shared linear OLS suite

Primary implementation and saved families:

- `scripts/cluster_scripts/gated_linear_probes/gated_linear_probe_common.py`
- `scripts/cluster_scripts/gated_linear_probes/consolidate_joint_gated_layer.py`
- `scripts/cluster_scripts/gated_linear_probes/aggregate_gated_linear_probe_suite.py`
- intended production root: `$CSV_ROOT/gated_linear_probes/consolidated/`
- historical saved root: `delta_files/consolidated/`

### Active canonical implementation

- **O1:** Each final family writes a schema-versioned `run_manifest.json` with
  protocol, frozen-split, prepared-cohort, family-configuration, feature-order,
  version, and code-revision provenance. Preparation and task manifests retain
  explicit input and fit failures.
- **O2/O5:** Each layer recombines every routed gate prediction into one
  all-pairs vector per protein. `per_protein.csv` contains one canonical
  whole-protein row per protein-layer with expected/evaluated counts, coverage,
  and prediction status; it does not expand proteins by gate or sequence bin.
- **O3/O6:** `pooled_metrics.csv` contains separate development and locked-test
  rows for the recombined predictor. Global and gate-conditional development
  baselines are evaluated on exactly the model's finite rows. MSE, skill ratios,
  and mean-per-protein duplicates are not emitted.
- **O4:** All three families evaluate the same locked-test piecewise predictions
  in the canonical sequence-separation, amino-acid-pair, and spatial-distance
  subgroup schemas.
- **O7/O8:** `geometry_metrics.csv` retains every protein-layer attempt, and
  `geometry_summary.csv` summarizes complete rows only. Any missing joint-gate
  prediction yields `geometry_status=incomplete_pair_coverage`, an explicit
  reason, and `NaN` geometry metrics; no imputation or fallback is used.
- **O9:** All families use compressed coefficient NPZ shards plus
  `model_artifacts/coefficient_index.csv`. Shards retain standardized and raw
  coefficients and intercepts plus feature means/scales for exact replay.
- **O10:** `model_status.csv` has one row per requested layer-gate, including
  unsupported gates, support counts, fit flags, and solver-exposed OLS
  diagnostics without an additional SVD.
- **O11:** `runtime.csv` has one compact row per analysis-layer; detailed gate
  and stage timing remains in timestamped stdout.
- **O12:** Sequence and AA workers checkpoint canonical layer shards directly.
  Six joint task fragments are recombined by a separate per-layer array. Final
  aggregation verifies schemas, hashes, configuration/split digests, layer
  coverage, primary-key uniqueness, and exact row counts before publishing the
  top-level tables.
- The canonical path does not emit legacy gate summaries, large per-protein by
  gate files, `summary.md`, bootstrap tables, detailed timing CSVs, redundant
  MSE/skill fields, or duplicate triangle samples.

### Historical evidence and remaining run action

The current `delta_files/consolidated/` family remains the historical six-layer
output described by the earlier audit. It is not rewritten in place and does
not demonstrate that the new 48-layer production run has completed. There is
no remaining gated-output implementation gap, but the Delta run and subsequent
artifact audit are still required before the canonical result family is
complete in storage.

### Checklist

- [x] Produce all-protein recombined `per_protein.csv` for each gated family.
- [x] Produce native whole-model `pooled_metrics.csv` rather than relying on a
  visualization postprocessor.
- [x] Replace the current wide gate-summary metric block with the lean common
  metric schema.
- [x] Produce all three test subgroup files for every gated family.
- [x] Add full-coverage-only geometry and explicit incomplete-coverage rows.
- [x] Move gated coefficients to indexed compressed shards for 48 layers.
- [x] Add compact per-layer runtime output.
- [ ] Complete the Delta production rerun and audit the consolidated artifacts.

## Shared XGBoost

Primary implementation and saved family:

- `scripts/cluster_scripts/xgboost_search/`
- `delta_files/xgboost_hyperparameter_search/shared/`

### What already aligns

- Search artifacts save configurations, split assignments and digests, skipped
  proteins, validation results, best iterations, and detailed timing.
- The selected-model implementation can save a native `booster.json`, channel
  importance, per-protein regression rows, split summaries, timing, and run
  metadata.
- Per-protein and pooled metric helpers already use a schema close to the shared
  linear output.

### Gaps and changes

- The copied result family currently contains search outputs only. No saved
  canonical selected-model `per_protein.csv`, pooled output, geometry output,
  or booster is present.
- The existing selected-model code describes an early-stopped 60%-fit model and
  explicitly records that it was not refitted on the combined 80%. It is not
  the final model required by `regression_methodology.md`.
- **O2/O3:** The eventual 80%-refitted model needs all-protein per-protein rows
  and development/test pooled metrics using the common schema.
- **O4:** All three subgroup outputs are absent.
- **O7/O8:** Geometry is explicitly marked as not computed.
- **O9:** Native booster saving is correct. The final run should save one JSON
  booster per layer; gain/split importance is optional supplementary output.
- **O10:** Final boosting rounds and selected configuration belong in
  `model_status.csv`, separately from the exploratory search table.
- **O11:** Existing detailed timings can be reduced to the common per-layer
  runtime schema for final outputs while the logs retain fine detail.

### Checklist

- [ ] Complete the canonical 80%-refit implementation first.
- [ ] Save final booster, manifest, status, and common per-protein/pooled files
  for all 48 layers.
- [ ] Add test subgroup and full per-protein geometry outputs.
- [ ] Keep search results separate from final-model paper outputs.

## Baselines and true-distance statistics

The final contract includes two controls in `pooled_metrics.csv`: a common
global development-mean predictor and, where applicable, a conditional
predictor based on gate identity or a protein's training-pair mean. The latter
is not featureless and must be labeled separately. The migrated gated
implementation writes both controls only in its compact pooled output; the
historical six-layer saved family still predates this layout.

Retain baseline $R^2$, MAE, RMSE, and bias only in `pooled_metrics.csv`. Skill
ratios are derived later. True-distance means, medians, variances, and category
support belong in `analysis_3_true_distance_statistics` and should not be
copied into every model family.

## Publication-level analyses not emitted by model scripts

The following should be produced once by common downstream analysis code:

- mean-per-protein layer trajectories from `per_protein.csv`;
- paired per-protein differences between model variants on identical test IDs;
- final layer-wise geometry and regression tables and plot-data CSVs; and
- baseline skill ratios if a figure or table actually uses them.

Bootstrap intervals and uncertainty bands are deferred and are not part of the
current implementation contract. This keeps filtering and averaging logic out
of individual regression scripts while preserving the core inputs needed for
later analyses.

## Prioritized remediation order

### Priority 0: common paper-comparison outputs

- [ ] Implement one common metric/schema writer used by all final regressions.
- [ ] Emit authoritative `per_protein.csv` and `pooled_metrics.csv` everywhere.
- [ ] Emit the three locked-test subgroup tables everywhere.
- [x] Add gated whole-protein recombination and exact coverage fields.

### Priority 1: geometry and replayability

- [ ] Run the common geometry function for every final model.
- [x] Exclude incomplete joint-gated matrices explicitly rather than imputing.
- [ ] Standardize coefficient/model storage and migrate large families to NPZ
  shards plus indexes.

### Priority 2: compactness and provenance

- [ ] Add schema-versioned manifests, model-status tables, and brief runtimes.
- [ ] Remove duplicated mean-per-protein, skill, MSE, true-statistic, and
  exact-triangle copies from future canonical runs while retaining the seven
  canonical triangle statistics in `geometry_metrics.csv`.
- [ ] Validate shards before declaring consolidated outputs complete.

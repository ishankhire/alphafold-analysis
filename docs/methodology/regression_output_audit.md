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
| Shared linear OLS | Meets | Meets | Meets | Meets | Not applicable | Meets | Meets | Partial | Meets | Canonical coefficient-replay output implementation complete; unavailable historical fit diagnostics remain explicit and the 48-layer production replay is pending |
| Per-protein linear OLS | Meets | Meets | Meets | Meets | Not applicable | Meets | Meets | Meets | Meets | Canonical output implementation complete; 48-layer production rerun pending |
| Sequence-gated OLS | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Canonical output implementation complete; 48-layer production rerun pending |
| AA-pair-gated OLS | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Canonical output implementation complete; 48-layer production rerun pending |
| AA-pair x sequence-gated OLS | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Meets | Canonical no-fallback outputs complete in code; 48-layer production rerun pending |
| Shared XGBoost | Meets | Meets | Meets | Meets | Not applicable | Meets | Meets | Meets | Meets | Canonical output implementation complete; 48-layer production rerun pending |

## Shared linear OLS

Primary implementation and saved family:

- `scripts/cluster_scripts/generalized_model_ecod/general_train_test_common.py`
- `scripts/cluster_scripts/generalized_model_ecod/general_base256_coefficient_replay_common.py`
- `scripts/cluster_scripts/generalized_model_ecod/aggregate_general_base256_coefficient_replay.py`
- `delta_files/general_base256_combined/`
- pending canonical Delta family: `$CSV_ROOT/general_base256_combined_canonical/`

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

### Active canonical replay implementation

- **O1/O12:** A non-fitting 48-layer array replays the saved raw-scale
  coefficients into checksum-bearing canonical shards. The strict aggregator
  validates source hashes, frozen outer-split membership, schemas, primary
  keys, row counts, coverage, pooled reconstruction, deterministic coefficient
  replay, and agreement with the historical metrics before publishing a
  versioned manifest.
- **O2/O3/O6:** Every layer emits all-protein common-schema rows and separate
  development/test pooled metrics with the common protein-equal development
  mean baseline. Redundant MSE and mean-per-protein summary fields are omitted.
- **O4:** The same replayed predictions produce all fixed sequence-separation,
  amino-acid-pair, and spatial-distance subgroup rows for locked-test proteins.
- **O7/O8:** Geometry is recomputed from complete replay predictions and saved
  in authoritative per-protein plus compact long-summary form. Any geometry or
  coverage failure prevents layer completion.
- **O9/O11:** The small shared coefficient family is normalized under
  `model_artifacts/coefficients.csv`, and runtime is one row per layer.
- **O10 — Partial by provenance:** Each layer has an explicit
  `historical_fit_replayed` model-status row, but rank and singular-value
  diagnostics cannot be recovered from the six-decimal coefficient artifact.
  They remain `NaN` with an unavailable-historical-fit flag; the replay does
  not perform the extra SVD prohibited solely for output backfilling.
- The run manifest discloses the historical P2, P5, and P8 deviations and the
  six-decimal coefficient serialization rather than claiming full methodology
  compliance. Exact-triangle duplicates and sampled-triangle files are not
  regenerated.

### Remaining production action

The output migration is complete in code, but `$CSV_ROOT/general_base256_combined_canonical/`
has not yet completed and been copied back under `delta_files/`. Run a one-layer
pilot, right-size from MaxRSS, complete all 48 shards, and audit the consolidated
family before treating it as the stored canonical result.

### Checklist

- [x] Emit common-schema `per_protein.csv` and `pooled_metrics.csv`.
- [x] Add all three locked-test subgroup files for all 48 layers.
- [x] Add `model_status.csv`, `runtime.csv`, and a schema-versioned manifest.
- [x] Recompute canonical geometry with explicit coverage/status fields.
- [x] Stop producing duplicate exact-triangle and sampled-triangle files in the
  final result family unless a specific diagnostic requests them.
- [ ] Complete the Delta production replay and audit the stored final family.

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
- historical selection outputs: `delta_files/xgboost_hyperparameter_search/shared/`
- pending canonical output: `delta_files/shared_xgboost_base256_combined/`

### Active implementation status

- **O1/O12:** The final runner writes checksum-bearing canonical layer shards;
  the non-fitting aggregator rejects missing, failed, duplicate,
  digest-inconsistent, schema-invalid, or incomplete shards before publishing
  a complete schema-versioned run manifest.
- **O2/O3/O6:** Every layer produces all-protein canonical per-protein rows and
  separate development/test pooled model and global-baseline metrics.
- **O4:** Each layer emits all six sequence, 210 amino-acid-pair, and five
  true-distance locked-test subgroup rows, retaining sparse support cells.
- **O7/O8:** Complete all-pair predictions feed per-protein geometry and the 28
  layer/split/metric summary rows.
- **O9:** Each layer immediately checkpoints a native JSON booster, emits all
  256 gain/split-count rows including zeroes, and saves a deterministic replay
  batch verified again during consolidation.
- **O10/O11:** Each layer emits one fixed-S4/5k/700 status row and one compact
  runtime/MaxRSS row.

The code and local tests now meet the output contract. No final production
artifacts have yet been copied under `delta_files/shared_xgboost_base256_combined/`;
the 48-layer Delta run and independent artifact audit remain pending.

### Checklist

- [x] Complete the canonical 80%-refit implementation.
- [x] Implement final boosters, manifests, status, and common outputs for all 48 layers.
- [x] Add test subgroup and full per-protein geometry outputs.
- [x] Keep historical search/sensitivity outputs separate from the final family.
- [ ] Complete the Delta production rerun and audit the stored final family.

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

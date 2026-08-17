# Regression Output Audit and Gaps

## Purpose and audit boundary

This document compares the active regression implementations and saved result
families with `regression_outputs.md`. It is an artifact-and-code audit as of
2026-08-17. Historical files remain useful, but they do not automatically meet
the final paper output contract.

The audit assesses whether a result family saves the information needed for
consistent comparison. It does not require historical filenames to be renamed
before their results can be inspected.

Status labels are:

- **Meets:** the saved family already provides the required output.
- **Partial:** the main information exists but its scope, schema, or storage
  differs materially.
- **Gap:** the output is absent.
- **Not applicable:** the requirement does not apply to that model family.

## Output protocol checklist

| ID | Output requirement |
|---|---|
| O1 | Versioned run manifest, canonical split digest, and explicit failures |
| O2 | All-protein, long-format `per_protein.csv` with common metrics and coverage |
| O3 | Compact authoritative `pooled_metrics.csv` without duplicated mean-per-protein fields |
| O4 | Locked-test pooled metrics by sequence separation, AA pair, and spatial distance |
| O5 | Gated whole-protein recombination plus compressed per-protein sequence-bin detail |
| O6 | Compact training-derived baseline metrics; no redundant skill columns |
| O7 | Per-protein geometry and compact derived geometry summary |
| O8 | Geometry requires complete pair coverage; exclusions remain explicit |
| O9 | Replayable coefficients or boosters stored without avoidable CSV expansion |
| O10 | Per-fitted-unit status and OLS/XGBoost diagnostics |
| O11 | Brief layer/task timing |
| O12 | Validated layer shards and consolidated canonical outputs |

## Summary by principal model family

| Analysis | O1 provenance | O2 per protein | O3 pooled | O4 subgroups | O5 gated detail | O7/O8 geometry | O9 artifacts | O10 status | O11 timing | Overall disposition |
|---|---|---|---|---|---|---|---|---|---|---|
| Shared linear OLS | Partial | Partial | Partial | Gap | Not applicable | Meets | Meets | Partial | Partial | Strongest current output family; schema normalization and subgroup output remain |
| Per-protein linear OLS | Partial | Partial | Gap | Gap | Not applicable | Meets | Partial | Partial | Partial | Core metrics and geometry exist, but pooled/subgroup outputs and compact coefficient storage are missing |
| Sequence-gated OLS | Meets | Gap | Partial | Partial | Partial | Gap | Meets | Meets | Partial | Detailed gate output exists, but canonical whole-protein and geometry outputs do not |
| AA-pair-gated OLS | Meets | Gap | Partial | Partial | Partial | Gap | Partial | Meets | Partial | Same recombination gaps; coefficient CSV will become large at 48 layers |
| AA-pair x sequence-gated OLS | Meets | Gap | Partial | Partial | Partial | Gap | Meets | Meets | Partial | Coverage is tracked, but whole-protein and explicit geometry-exclusion outputs are missing |
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
- **O10 — Partial:** Shared OLS does not save rank, singular values, condition
  number, or a model-status row per layer.
- **O11 — Partial:** Total elapsed seconds are present in the split summary and
  detailed stdout, but there is no small common runtime table with major stage
  timings.
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

- The saved `per_protein.csv` covers all 552 proteins and all 48 layers.
- It includes all-pair regression metrics plus rank, singular-value, condition,
  and timing diagnostics.
- The active implementation now defines `train_r2` and `test_r2` fields, uses a
  deterministic within-protein 80/20 pair split, and predicts all pairs for
  geometry.
- `geometry_metrics.csv` and `geometry_summary.csv` cover every protein and
  layer with the same strong geometry schema as the shared model.
- Coefficients exist for every protein-layer fit.

### Gaps and changes

- **O1 — Partial:** There is no final run manifest tying the pair-split rule,
  master protein roles, configuration, and library versions to a schema
  version.
- **O2 — Partial:** The currently copied `per_protein.csv` predates the active
  train/test-pair fields and contains only combined all-pair metrics. The next
  run should save long-format `train_pairs`, `heldout_pairs`, and `all_pairs`
  rows, with master development/test protein roles and coverage.
- **O3 — Gap:** There is no authoritative pooled regression CSV. Text summaries
  and geometry summaries do not replace pooled $R^2$, MAE, RMSE, bias, and
  baseline metrics for each evaluation scope.
- **O4 — Gap:** No canonical test-protein subgroup files exist.
- **O9 — Partial:** `coefficients.csv` is about 647 MB because protein and model
  metadata are repeated for every coefficient. This is the clearest current
  example of avoidable output bloat. Use NPZ coefficient shards plus a compact
  index instead.
- **O10 — Partial:** Numerical diagnostics are embedded in `per_protein.csv`;
  add explicit fit/support status so unstable or failed units remain auditable.
- **O11 — Partial:** Per-protein elapsed seconds exist, but the desired final
  output is a small layer/task timing table rather than thousands of timing
  values mixed into scientific metrics.
- The separate exact-triangle tree duplicates metrics that have already been
  merged into the canonical geometry table.

### Checklist

- [ ] Rerun or regenerate the common long-format pair-scope table.
- [ ] Add pooled metrics for train, held-out, and all-pair scopes.
- [ ] Add the three locked-test subgroup outputs, using held-out pairs.
- [ ] Replace the 647 MB coefficient CSV with indexed NPZ shards.
- [ ] Add explicit prediction/fit status, coverage, and master protein role.
- [ ] Consolidate timing to one row per layer/task.

## Gated shared linear OLS suite

Primary implementation and saved families:

- `scripts/cluster_scripts/gated_linear_probes/gated_linear_probe_common.py`
- `delta_files/consolidated/analysis_1_sequence_gated/`
- `delta_files/consolidated/analysis_2_aa_pair_gated/`
- `delta_files/consolidated/analysis_4_aa_sequence_gated/`

### What already aligns

- **O1/O10:** The suite has a run manifest, split/configuration digests,
  `gate_status.csv`, support counts, explicit failures, OLS rank/singular-value
  diagnostics, and elapsed time per fitted gate.
- `gate_summary.csv` saves pooled $R^2$, MAE, and RMSE separately for train
  and held-out proteins at each routing gate.
- Existing subgroup files evaluate held-out predictions by relevant sequence,
  amino-acid, and spatial-distance categories.
- Compressed `test_per_protein_gate_metrics.csv.gz` retains detailed per-gate
  protein results.
- Sequence-gated coefficients are compact. The joint model already uses the
  preferred compressed NPZ-shard-plus-index design.
- The joint analysis retains 510 `insufficient_train_rows` statuses instead of
  silently omitting those requested gates.

### Common gaps and changes

- **O2/O5 — Gap:** No result family has a canonical main `per_protein.csv` that
  recombines the piecewise predictions into one whole-protein row. The existing
  compressed table is gate-level, test-only detail.
- **O3 — Partial:** Whole-test pooled metrics are reconstructed later into
  `gated_probe_visualizations/whole_protein_comparison_metrics.csv`; they should
  instead be native canonical outputs for both development and test proteins.
- `gate_summary.csv` duplicates target mean, median, variance, standard
  deviation, mean-per-protein metrics, baseline metrics, and four skill
  columns. Most of these can be removed from the final pooled schema.
- **O4 — Partial:** Category metrics exist, but schemas and category coverage
  differ by gated family. The final piecewise prediction from every model must
  be evaluated in all three common category systems, using test proteins only.
- **O5 — Partial:** The existing compressed per-gate tables can become very
  large: the six-layer test-only joint table is about 28 MB compressed. The
  canonical supplementary expansion should be by the six sequence bins, not by
  all 1,260 joint routing gates, and should include all proteins.
- **O6 — Partial:** The current baseline is correctly training-derived, but it
  is repeated across many output scopes along with derivable skill scores.
  Retain baseline metrics only in compact whole-model pooled output.
- **O7 — Gap:** None of the gated result families saves MDS or triangle geometry
  outputs.
- **O11 — Partial:** Per-gate elapsed time is available, but one compact
  per-layer/task runtime output is still needed.

### Sequence-gated-specific notes

- All 36 saved gate fits succeeded.
- `gate_summary.csv` is small, and the coefficient CSV is appropriately
  readable.
- The present `subgroup_by_amino_acid_pair.csv` is about 6 MB for six layers;
  an all-48-layer common subgroup file remains reasonable and does not need a
  per-protein AA expansion.

### Amino-acid-gated-specific notes

- All 1,260 saved gate fits succeeded.
- The six-layer coefficient CSV is already about 28 MB. At 48 layers it would
  grow by roughly eightfold, so indexed compressed arrays are preferable for
  the final run.
- Sequence-separation and spatial-distance subgroup outputs exist, but the
  common marginal AA-pair output should also evaluate the recombined model in
  the unified schema.

### Joint-gated-specific notes

- **O8 — Gap:** The saved regression summaries qualify pair coverage, but there
  is no per-protein whole-model coverage row and no geometry-exclusion table.
- The current `delta_files/consolidated/excluded_pairs_summary.csv` contains
  only its header. It records preparation-time pair exclusions and does not
  account for prediction-time pairs routed to an unfitted sparse joint gate.
- Whole-protein regression metrics may use the covered subset as requested,
  with the exact coverage reported.
- Geometry must not use the covered subset as if it were a complete distance
  matrix. Every incomplete protein-layer combination should remain in
  `geometry_metrics.csv` with `geometry_status=incomplete_pair_coverage` and
  `NaN` geometry values.
- The current coefficient NPZ shards and `coefficient_index.csv` are the model
  to reuse for other large coefficient families.

### Checklist

- [ ] Produce all-protein recombined `per_protein.csv` for each gated family.
- [ ] Produce native whole-model `pooled_metrics.csv` rather than relying on a
  visualization postprocessor.
- [ ] Replace the current wide gate-summary metric block with the lean common
  metric schema.
- [ ] Produce all three test subgroup files for every gated family.
- [ ] Save all-protein `per_protein_by_sequence_separation.csv.gz`.
- [ ] Add full-coverage-only geometry and explicit incomplete-coverage rows.
- [ ] Move AA-gated coefficients to indexed compressed shards for 48 layers.
- [ ] Add compact per-layer runtime output.

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

The gated-suite intercept baseline predicts a training-derived mean distance;
it is not a random predictor. It is scientifically useful as a minimal sanity
check, but the current gate summaries repeat more baseline-derived columns than
the paper needs.

The final contract retains baseline $R^2$, MAE, RMSE, and bias only in
`pooled_metrics.csv`. Skill ratios are derived later. True-distance means,
medians, variances, and category support belong in
`analysis_3_true_distance_statistics` and should not be copied into every model
family.

## Publication-level analyses not emitted by model scripts

The following should be produced once by common downstream analysis code:

- mean-per-protein layer trajectories from `per_protein.csv`;
- protein-bootstrap 95% confidence intervals;
- paired per-protein differences between model variants on identical test IDs;
- geometry distribution summaries and confidence intervals;
- final tables and plot-data CSVs; and
- baseline skill ratios if a figure or table actually uses them.

This avoids slightly different confidence-interval, filtering, and averaging
logic being embedded in every regression script.

## Prioritized remediation order

### Priority 0: common paper-comparison outputs

- [ ] Implement one common metric/schema writer used by all final regressions.
- [ ] Emit authoritative `per_protein.csv` and `pooled_metrics.csv` everywhere.
- [ ] Emit the three locked-test subgroup tables everywhere.
- [ ] Add gated whole-protein recombination and exact coverage fields.

### Priority 1: geometry and replayability

- [ ] Run the common geometry function for every final model.
- [ ] Exclude incomplete joint-gated matrices explicitly rather than imputing.
- [ ] Standardize coefficient/model storage and migrate large families to NPZ
  shards plus indexes.

### Priority 2: compactness and provenance

- [ ] Add schema-versioned manifests, model-status tables, and brief runtimes.
- [ ] Remove duplicated mean-per-protein, skill, MSE, true-statistic, and exact
  triangle outputs from future canonical runs.
- [ ] Validate shards before declaring consolidated outputs complete.

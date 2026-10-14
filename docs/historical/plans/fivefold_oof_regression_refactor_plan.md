> Historical reference; not current execution instructions. See [current sources and status](../../project_status.md).

# Fivefold OOF Regression Refactor Plan

## Summary

Make fivefold out-of-fold evaluation the primary paper protocol for:

- Shared base256 OLS.
- Per-protein base256 OLS.
- Sequence-separation-gated shared OLS.
- Shared base256 XGBoost with fixed S4/5k/700.

Preserve existing 80/20 implementations and outputs as historical. Add the workflows under `scripts/cluster_scripts/cross_validation/`, with results in `$CSV_ROOT/regression_5fold_v1/` on Delta and `delta_files/regression_5fold_v1/` locally.

Codex may implement, test, document, commit, and push one stage at a time. Codex will not deploy or submit Delta jobs.

No new resource pilots will be required. Memory and walltime will be estimated from completed comparable workflows. This is an explicit project-level override of the usual `AGENTS.md` exact-workflow pilot rule and must be recorded in the resource documentation.

## Decision Rationale for Implementation Handoff

- Build a new CV workflow tree rather than rewrite the historical runners. This preserves reproducibility of the existing 80/20 results while allowing minimal, backward-compatible extraction of tested helpers.
- Shared OLS, sequence-gated OLS, and XGBoost answer an unseen-protein question, so their held-out unit is the whole protein. Per-protein OLS answers a within-protein held-out-pair question, so its held-out unit is the residue pair; its outer-fold label is reporting metadata only.
- Complete OOF predictions are concatenated before authoritative metrics and geometry are calculated. Do not average fold-level R² values or construct five separate MDS results for one protein-layer, and do not calculate geometry from training predictions.
- Fixed XGBoost is a deliberate compute/interpretation tradeoff. The existing equal-anchor validation table selected S4 with mean per-protein R² `0.820963`, versus `0.819661` for runner-up S1, a difference of about `0.00130`. This supports treating nearby hyperparameter choices as similar, but it does not remove selection bias.
- Consequently, do not add a new hyperparameter-selection script or inner CV loop in this project. The XGBoost production workflow consists of the fixed fold/layer runner, aggregation, and launch/replay machinery. A fully selection-independent XGBoost claim would require a separately approved nested-CV analysis or new external holdout.
- The source of the frozen XGBoost decision is `delta_files/xgboost_hyperparameter_search/shared/winner_cap_sweep/cross_layer_selection_table.csv`; preserve that path and its digest in provenance rather than copying its values into a new selection artifact.

## Locked Experimental Protocol

### Outer protein folds

- Retain the frozen 552 eligible proteins and two technical exclusions.
- Add `regression_outer_folds_v1.csv`:
  - Existing 111 test proteins become fold 0.
  - Assign the remaining 441 proteins to folds 1–4 using deterministic shuffled stratification over the existing five length bins.
  - Use seed 44 and record a semantic digest.
- Every eligible protein belongs to exactly one outer fold.
- Shared OLS, sequence-gated OLS, and XGBoost train on four folds and predict every pair in the fifth.
- Claims remain limited to unseen protein IDs, not unseen sequence or structural families.

### Per-protein folds

- Outer folds do not affect per-protein model fitting; they only provide matched five-group summaries.
- Deterministically shuffle each protein’s canonical unordered pairs using a protein-ID-derived seed.
- Divide the pairs as evenly as possible into five pair folds.
- Fit five 80/20 OLS models per protein-layer.
- Each model predicts only its held-out pair fold.
- Concatenate the five disjoint held-out vectors into one complete OOF prediction vector.
- Calculate one authoritative OOF result and one OOF geometry reconstruction per protein-layer.
- Never average multiple predictions for a pair.

### Model rules

- Preserve canonical directional base256 float64 features and absolute Cα-distance targets.
- Fit scalers and baselines separately using only each fold’s training observations.
- Shared models retain protein-equal fitting weights.
- Sequence-gated OLS fits the six existing sequence gates with gate-specific scalers and protein-equal weighting. Any invalid gate makes that fold-layer incomplete; there is no fallback.
- XGBoost:
  - Freeze S4, 5,000 sampled training rows per protein, and 700 rounds.
  - Reuse one target- and layer-independent 5k sample per protein across outer folds.
  - Use deterministic model seed `42 + outer_fold`.
  - Do not use early stopping or `eval_set`.
  - Record that the specification was selected through preliminary work on the same cohort, so the fixed fivefold estimate is not nested or fully selection-independent.
  - Preserve the historical-test-exposure disclosure.
- Do not fit additional all-data models after CV.
- Report all 48 layers. Any observed best layer is descriptive.

## Outputs and Replay Interface

Use additive output schema v2 without changing historical files. Shared metadata lives under:

```text
regression_5fold_v1/
  metadata/
    regression_outer_folds_v1.csv
    per_protein_pair_folds_v1.csv
    fold_digests.json
```

Metadata granularity is compact and deterministic:

- `regression_outer_folds_v1.csv` has one row per eligible protein, with at least `protein_id`, `source`, `n_residues`, `length_bin`, `outer_fold`, and the historical split role used to verify fold 0.
- `per_protein_pair_folds_v1.csv` has one row per eligible protein, not one row per pair. It records `protein_id`, `n_residues`, `n_pairs`, the pair-fold algorithm/version and seed, five fold counts, and an assignment digest. Pair assignments must be regenerated exactly from canonical pair indices.
- `fold_digests.json` records the protocol version, construction parameters, cohort digest, outer-fold digest, pair-fold manifest digest, and per-fold protein counts.

Family roots are:

```text
shared_ols_base256/
per_protein_ols_base256/
sequence_gated_ols_base256/
shared_xgboost_base256_fixed_s4/
```

Each applicable family contains:

```text
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

### Output semantics

- Add `cv_protocol_version`, `evaluation_design`, `outer_fold`, and explicit OOF scope fields.
- `per_protein.csv`: one authoritative OOF row per protein-layer—26,496 rows per family.
- `pooled_metrics.csv`: one complete-cohort OOF row per layer, calculated by combining OOF sufficient statistics rather than averaging fold metrics.
- `fold_metrics.csv`: five outer-fold OOF rows per layer. Per-protein OLS uses each protein’s outer-fold label only for this matched grouping.
- `fold_summary.csv`: mean and standard deviation across five fold metrics. These are descriptive, not confidence intervals.
- `training_metrics.csv`: compact fold-level training R², MAE, RMSE, bias, support, and fit diagnostics:
  - OLS summaries cover all training pairs.
  - XGBoost summaries cover sampled fitting rows.
  - No training geometry or training subgroup outputs.
- Preserve train-derived global baselines for every model.
- Preserve sequence-gate and per-protein conditional baselines.
- `stable_fit_metrics.csv` contains OOF sensitivity summaries restricted to rows whose contributing OLS fits all pass the current rank/conditioning policy.
- Primary complete-cohort output retains finite predictions from flagged OLS fits but marks them explicitly.
- Produce sequence-separation and true-spatial-distance subgroup tables only.
- Keep CAMEO and ECOD combined; source labels remain for provenance and path resolution only.
- Do not create a cluster-side cross-family comparison table.
- Do not save raw pair predictions.

### Geometry

Compute geometry once per protein-layer from complete OOF predictions:

- Classical MDS.
- Aligned RMSD and TM-score.
- Stress and spectral metrics.
- Negative prediction counts.
- Exact triangle-violation metrics.

For per-protein OLS, every matrix entry comes from the model that withheld that pair. Sequence-gated geometry requires complete gate coverage.

### Model retention

Retain every fold model for exact downstream replay:

- Shared OLS: 240 models.
- Sequence-gated OLS: 1,440 gate models.
- Per-protein OLS: 132,480 compressed coefficient/scaler models.
- XGBoost: 240 boosters.

`replay_index.csv` indexes retained model artifacts rather than individual pair predictions. It has one row per artifact with at least `model_family`, `layer`, nullable `outer_fold`, nullable `protein_id`, nullable `pair_fold`, nullable `sequence_gate`, `artifact_path`, `artifact_sha256`, configuration/scaler digests, and `status`. Pair-level responsibility is recovered by joining this compact index to the deterministic outer- or pair-fold manifest. Its expected row counts equal the model-retention counts above. This is a required interface for the separately planned triangle-subgroup refactor. Flag that dependency but do not plan or modify triangle-subgroup scripts here.

Save compact standardized-coefficient fold stability for OLS and fold-summarized feature importance for XGBoost.

## Arrays and Estimated Resources

No production pilot is required. Record that estimates use completed same-family jobs and conservative scaling.

| Workflow | Array | Concurrency | Memory | Walltime | Basis |
|---|---:|---:|---:|---:|---|
| Shared OLS | `0-239` = fold × layer | `%4` | `120g` | `00:30:00` | Existing shared OLS peaked near 98 GiB; single-layer reference completed well within 30 minutes. |
| Per-protein OLS | `0-47` = layer | `%12` | `12g` | `02:00:00` | Existing one-split layer used under 8 GiB and up to ~14 minutes; five pair-fold fits increase compute but reuse loaded features and geometry. |
| Sequence-gated OLS | `0-239` = fold × layer | `%8` | `120g` | `03:00:00` | Existing sequence-gated layer peaked near 93.4 GiB and 1:37:20. |
| Fixed XGBoost | `0-239` = fold × layer | `%3` | `12g` | `00:30:00` | Completed final base256 layers peaked near 6.4 GiB and 16:06 while evaluating all 552 proteins; each new task evaluates only one outer fold. |

All launchers must retain:

- CPU partition and account conventions.
- `scratch` constraint.
- Thread-count environment variables.
- `#SBATCH --mail-type=BEGIN,END,FAIL`.
- `#SBATCH --mail-user=ikhire3@gatech.edu`.
- `SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}`.
- `srun python3 -u "$SCRIPT_DIR/<entrypoint>.py"`.
- Atomic, restartable layer/fold shards.
- Stage and layer timestamps.

Update the shared memory and walltime guides with these estimates and explicitly label them as comparator-derived rather than newly measured.

## Seven Codex Stages

### Stage 1 — Protocol and frozen folds

- Update the canonical methodology and output contract to fivefold OOF.
- Increment protocol and output-schema versions.
- Reclassify the 80/20 and historical k-fold outputs in the audits.
- Extend the protocol helper with outer-fold and pair-fold construction and validation.
- Generate and check in the outer-fold and compact pair-fold manifests.
- Document the no-pilot resource-estimation override.
- Test manifest counts, stratification, disjointness, determinism, and digests.
- Commit and push.

### Stage 2 — Common CV engine

- Create `scripts/cluster_scripts/cross_validation/`.
- Implement shared:
  - Fold loading and task mapping.
  - Deterministic sampling.
  - Train-only scalers and baselines.
  - Protein-equal weighting.
  - OOF sufficient-statistic accumulators.
  - Geometry construction.
  - Replay indexing.
  - Artifact checksums.
  - Atomic checkpoint and resume validation.
  - Additive v2 output builders.
- Keep existing helper changes minimal and backward-compatible.
- Add synthetic protein-level and pair-level CV integration tests.
- Commit and push.

### Stage 3 — Shared OLS

- Implement the 240-task fold/layer runner and aggregator.
- Fit on all pairs from four protein folds.
- Stream the held-out fold through regression, subgroup, and geometry outputs.
- Save five coefficient/scaler models per layer.
- Produce compact cross-fold coefficient stability.
- Add the 120 GiB, 30-minute launcher.
- Commit and push after unit, integration, resume, and corruption tests pass.

### Stage 4 — Per-protein OLS

- Implement the 48-task layer array.
- Load each protein-layer once, fit five pair-fold models, and assemble its complete OOF vector.
- Compute authoritative OOF regression and geometry once per protein-layer.
- Save all fold coefficient/scaler models in compressed layer shards.
- Require all five contributing models to pass for a stable-only row.
- Add the 12 GiB, two-hour launcher.
- Commit and push after exact-once pair coverage tests pass.

### Stage 5 — Sequence-gated OLS

- Implement the 240-task fold/layer runner and aggregator.
- Fit and recombine the six sequence gates.
- Fail the shard on any invalid gate or missing expected prediction.
- Save gate/fold coefficients and stability summaries.
- Add the 120 GiB, three-hour launcher.
- Commit and push after gate isolation, scaling, weighting, and complete-coverage tests pass.

### Stage 6 — Fixed XGBoost

- Implement the 240-task fold/layer runner and aggregator.
- Add no hyperparameter-search workflow.
- Create a checksummed fixed configuration derived from the existing S4/5k/700 artifact.
- Enforce exact rounds, no early stopping, no evaluation set, deterministic sampling, and test loading only after the booster checkpoint exists.
- Save every booster, replay index, and fold feature-importance output.
- Local `.venv` lacks XGBoost, so use mocked estimator tests and retain an explicit cluster dependency error.
- Add the 12 GiB, 30-minute launcher.
- Commit and push.

### Stage 7 — Release hardening

- Run the complete local regression test suite.
- Validate every task mapping and expected shard count.
- Confirm all aggregators reject duplicate, missing, incomplete, or checksum-mismatched shards.
- Update memory/walltime guides with comparator-derived estimates and the explicit no-pilot decision.
- Finalize deployment and submission instructions without submitting jobs.
- Document the triangle-subgroup replay interface without modifying that workflow.
- Update audits and `branch-analysis.md`.
- Commit and push the production-ready implementation.

## Acceptance Criteria

- Outer folds are disjoint, exhaustive, deterministic, length-stratified, and share one digest across all model families.
- Fold 0 contains exactly the current 111 test proteins.
- Every shared-model protein is evaluated out of fold exactly once per layer.
- Every per-protein unordered pair is predicted out of fold exactly once per layer.
- No scaler, baseline, pair sample, or fit accesses evaluation targets.
- Protein-equal weights give every contributing training protein equal total mass.
- OOF pooled metrics match direct concatenation on synthetic data.
- Per-protein geometry uses one held-out prediction per pair and has 100% coverage.
- Sequence-gated output has 100% coverage or the fold-layer is incomplete.
- XGBoost fits exactly 700 rounds without early stopping or `eval_set`.
- All expected models, replay indexes, checksums, status rows, and geometry rows are present.
- A production family is complete only with:
  - 48 layers.
  - 552 OOF proteins per layer.
  - Complete expected pair coverage.
  - All fold artifacts.
  - Complete regression, fold, training, subgroup, geometry, stability, status, and runtime outputs.
- Existing 80/20 runners and tests remain operational.
- Fold-0 numerical parity with historical outputs is not an acceptance requirement.

## Paper Boundaries

- Fivefold OOF replaces 80/20 as the primary estimate.
- Report overall OOF pooled and mean-per-protein metrics plus fold mean and SD.
- Do not report confidence intervals in this project.
- Report the complete layer trajectory; maxima remain descriptive.
- Treat XGBoost as a fixed central nonlinear benchmark and disclose preliminary same-cohort selection.
- Training summaries are diagnostic and never mixed into paper OOF metrics.
- Exclude all-data refits, raw pair-prediction files, amino-acid gates/subgroups, secondary-structure migration, source-separated summaries, and cluster-side cross-family tables.
- The triangle-subgroup refactor remains a separate project that will consume these replayable OOF artifacts.

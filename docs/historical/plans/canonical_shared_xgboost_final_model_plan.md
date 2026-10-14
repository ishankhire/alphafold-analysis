> Historical reference; not current execution instructions. See [current sources and status](../../project_status.md).

# Canonical Shared XGBoost Final-Model Plan

## 1. Goal and scope

Implement the final paper shared-XGBoost distance probe as a separate,
restartable 48-layer workflow. The workflow must satisfy
`docs/methodology/regression_methodology.md` and
`docs/methodology/regression_outputs.md`, close the shared-XGBoost findings in
both regression audits, and leave the anchor-layer search and cap-sensitivity
outputs clearly separated from the final result family.

The final fixed model specification is:

| Setting | Value |
|---|---:|
| configuration ID | `S4` |
| feature definition | ordered `[Z[i,j,:] \| Z[j,i,:]]` for `i > j` |
| feature dimension | 256 |
| objective | `reg:squarederror` |
| tree method | `hist` |
| maximum depth | 6 |
| learning rate | 0.1 |
| minimum child weight | 10 |
| subsample | 0.8 |
| column subsample | 0.8 |
| L2 regularization | 1.0 |
| maximum histogram bins | 256 |
| final boosting rounds | 700 |
| row cap | 5,000 pairs per development protein |
| random seed | 42 |
| fitting weight | equal total weight per development protein |

The final workflow fits one model independently at each layer `0..47`. It is
the shared nonlinear counterpart to the final shared OLS trajectory; it is not
the exploratory per-protein XGBoost analysis.

## 2. Methodology decisions that must be reconciled first

The current normative methodology still names a 10,000-row cap and describes
validation-derived per-layer boosting counts. The completed validation sweep
instead supports the user-selected 5,000-row cap, and the requested final
configuration fixes 700 rounds globally. Before implementation, amend the
XGBoost section of `regression_methodology.md` so code and protocol agree:

1. Replace the accepted 10,000-pair cap with 5,000 pairs per fitting protein.
2. Record that the cap was selected by the equal-anchor-layer average of
   validation mean per-protein R-squared, with validation MAE and RMSE as
   consistency checks. Locked-test metrics must not be selection inputs.
3. Record `S4` as the common configuration at every layer.
4. Record 700 as the fixed final boosting-round count at every layer. The
   anchor runs reached zero-indexed iteration 699, so this is a fixed validated
   compute budget, not a claim that early stopping found a performance plateau.
5. State that the final 80% refit has no early stopping and receives no
   validation or test evaluation set.

Update the two audits only after their corresponding implementation and tests
exist. Keep the active-code status separate from the production-artifact
status: a complete implementation can meet the protocol while the 48-layer
Delta production run remains pending.

### Historical locked-test limitation

The currently copied anchor search and cap-sweep artifacts already contain
candidate/cap metrics for the canonical test IDs. Code changes cannot make
those IDs historically untouched again. The implementation should close the
P11 code gap by preventing all future search and sensitivity entrypoints from
loading or reporting test data, and the final workflow must never consult the
old test columns. The audit must nevertheless disclose the prior exposure.

If the paper requires the literal claim that the final test set was never
previously calculated or inspected, a new untouched external holdout is
required. Reassigning the canonical split after seeing its results would not
repair the issue and would break comparability with the other shared models.

## 3. Files to add or change

Add the final workflow beside the search code, while keeping its output root
independent:

```text
scripts/cluster_scripts/xgboost_search/
  shared_xgboost_final_common.py
  shared_xgboost_final_layer.py
  shared_xgboost_final_layer.sbatch
  aggregate_shared_xgboost_final.py
  aggregate_shared_xgboost_final.sbatch
  shared_xgboost_final_config.json
  tests/
    test_shared_xgboost_final.py
```

Change these existing files only where the listed responsibility belongs
there:

- `xgboost_search_common.py`: retain search utilities, but expose reusable
  deterministic capped-index and base256-loading helpers without making the
  final runner depend on exploratory output schemas.
- `shared_xgboost_hyperparameter_search.py`: remove candidate test loading,
  prediction, metrics, timing columns, and metadata.
- `shared_xgboost_winner_cap_sweep.py`: make the sensitivity stage
  fit/validation-only, rename its user-facing descriptions accordingly, and
  prohibit test evaluation.
- `regression_output_helpers.py`: reuse its existing schemas and builders. Add
  only genuinely model-agnostic serialization or validation support needed by
  the XGBoost runner.
- `regression_methodology.md`, `regression_protocol_audit.md`, and
  `regression_output_audit.md`: reconcile the frozen specification and update
  XGBoost status after implementation.
- `docs/plans/suggested_sbatch_memory.md`: replace the preliminary XGBoost
  estimate with observed final-pilot MaxRSS when it is available.

Do not change, revive, or publish the per-protein XGBoost workflow. It remains
outside the canonical protocol.

## 4. Frozen selection artifact

Check in `shared_xgboost_final_config.json` as the single machine-readable
source for final choices. It must contain:

- analysis name and model family;
- `S4` and every fixed XGBoost parameter, including 700 rounds;
- 5,000-row cap and sampling seed;
- canonical feature mode, feature dimension, pair ordering, target, and
  precision;
- protein-equal weighting mode;
- anchor layers and validation-only selection metric;
- the canonical master-manifest/split digest;
- hashes of the validation-only selection and cap-sweep tables used to freeze
  the decision;
- `training_metrics_used=false` and `test_metrics_used=false`;
- `source_runs_exposed_test=true` and
  `historical_test_exposure_disclosed=true`, so selection provenance does not
  erase the history of the exploratory artifacts;
- `round_selection_status=validation_boundary_at_700`, because all three 5k
  anchor fits reached iteration 699 rather than early-stopping below the cap;
- selection timestamp and a configuration digest.

The final runner must validate this artifact before loading pair matrices. It
must reject a missing/changed digest, a non-S4 configuration, any cap other
than 5,000, any round count other than 700, or provenance that reports test use.
It must not derive configuration choices at runtime or read
`search_summary.csv`.

Because the current selection tables contain test columns, first regenerate a
development-only selection table from the validation fields, with a schema
that has no `test_*` fields. Archive the existing test-exposing outputs as
historical/non-canonical rather than rewriting their provenance.

## 5. Canonical data and sampling

Use the existing common helpers rather than rebuilding dataset or metric logic:

- `regression_protocol.load_master_manifest` and
  `partition_manifest_by_role` for the frozen 330 `fit`, 111 `validation`, and
  111 `test` roles;
- `strict_lower_triangle_indices` for exactly one row per unordered pair;
- `extract_directional_base256` for ordered 256-channel float64 features;
- `protein_equal_weights` and `protein_equal_mean` for fitting weights and the
  global baseline;
- `sequence_separation_bin_codes` and `spatial_distance_bin_codes` for fixed
  evaluation categories;
- `general_train_test_common.build_dataset_protein_manifest`,
  `resolve_pair_block_path`, and `resolve_pdb_file` for source-aware input
  discovery;
- `cluster_regression_helpers.extract_ca_coordinates`,
  `compute_distance_matrix`, and `build_unordered_aa_pair_codes` for targets
  and subgroup metadata; and
- `regression_output_helpers` for per-protein, pooled, subgroup, geometry,
  status, runtime, manifest, atomic-write, and hashing behavior.

For every eligible protein, enumerate the strict lower triangle once. Select
training rows with a stable protein-ID-derived permutation of those canonical
pair positions, taking `min(5000, n_pairs)` rows. The selection must:

- be independent of layer, pair features, and target distance;
- be reused for all 48 final models;
- contain all pairs for proteins with fewer than 5,000 pairs;
- have an index SHA-256 digest recorded per development protein; and
- give every development protein total sample weight one, so each selected row
  receives weight `1 / n_selected_rows_for_that_protein`.

The final fitting population is the union of the canonical `fit` and
`validation` proteins: 441 development proteins. The locked 111 test proteins
must never be sampled for fitting. A failure to resolve or validate any frozen
eligible protein at a layer makes that layer incomplete; it must not silently
change cohort membership.

Materialize pair tensors with plain `np.load(...)` and preallocated writes.
Temporary pair tensors may remain float32, but every XGBoost feature matrix,
target vector, sample weight, metric accumulation, and baseline calculation
must be float64 before use.

## 6. One-layer final fitting and evaluation

`shared_xgboost_final_layer.py` maps `SLURM_ARRAY_TASK_ID` directly to one
layer in `0..47`. Its phases are:

1. Validate the frozen configuration, master manifest, split digest, input
   paths, and the complete layer-specific frozen cohort.
2. Build or validate the per-development-protein 5k sample manifest and its
   cross-protein digest.
3. Materialize the sampled 441-protein development matrix in float64.
4. Fit a fresh `XGBRegressor` for exactly 700 rounds with protein-equal sample
   weights. Do not set `early_stopping_rounds`, `eval_set`, or
   `sample_weight_eval_set` in the final fit.
5. Immediately save the native booster and a fit-stage checkpoint before
   beginning full-protein evaluation.
6. Stream all unordered pairs for all 552 eligible proteins through the model,
   one protein at a time. Do not concatenate the complete development/test
   evaluation matrices in memory.
7. For each protein, use the same prediction vector to update the canonical
   per-protein row, split-pooled accumulators, locked-test subgroup
   accumulators, and all-pairs geometry row.
8. Atomically write the complete canonical layer shard and a checksum-bearing
   shard manifest only after every eligible protein has been evaluated.

Predictions are raw scalar XGBoost outputs. Do not clip or otherwise
post-process them for regression metrics. `build_geometry_row` records negative
predictions and performs only the geometry-specific zero clipping required by
the output contract.

### Development and test scope

`per_protein.csv` has one `evaluation_scope=all_pairs` row per eligible protein
and layer. Its `protein_split` is `development` for all 441 proteins used in
the final refit and `test` for the 111 locked proteins. Development metrics are
descriptive and partly in-sample; paper generalization claims use test rows.

`pooled_metrics.csv` has exactly two rows per layer:

- `protein_split=development`, `evaluation_scope=all_pairs`;
- `protein_split=test`, `evaluation_scope=all_pairs`.

The global baseline is the protein-equal mean target distance computed from all
pairs of the 441 development proteins, without test targets. Evaluate that
same frozen scalar on the exact finite rows used by the model in both pooled
rows. Shared XGBoost has no conditional baseline, so use
`conditional_baseline_type=not_applicable` and `NaN` conditional metrics.

### Locked-test subgroups

Only test proteins contribute to subgroup files. Each layer shard emits:

- 6 sequence-separation rows;
- 210 unordered standard-amino-acid-pair rows; and
- 5 true-spatial-distance rows.

All use `protein_split=test` and `evaluation_scope=all_pairs`. Retain sparse
cells and mark support status through `SubgroupAccumulator`; do not fit subgroup
models. Spatial distance is an evaluation label only.

### Geometry

Produce one `geometry_metrics.csv` row per eligible protein and layer from the
complete symmetric predicted distance matrix. Geometry must include the MDS,
TM-score, eigenvalue, stress, negative-prediction, and seven exact-triangle
fields already implemented by the common helpers. Produce the compact
long-format `geometry_summary.csv` for development and test separately.

Do not save pair predictions or pair/triangle samples in the final result
family.

## 7. Exact final output contract

Use this cluster root:

```text
$CSV_ROOT/shared_xgboost_base256_combined/
```

Copy the audited completed family to:

```text
delta_files/shared_xgboost_base256_combined/
```

The final tree is:

```text
shared_xgboost_base256_combined/
  run_manifest.json
  per_protein.csv
  pooled_metrics.csv
  model_status.csv
  runtime.csv
  geometry_metrics.csv
  geometry_summary.csv
  metadata/
    selection_provenance.json
    split_assignments.csv
    training_sample_manifest.csv
    excluded_proteins.csv
  subgroup_metrics/
    by_sequence_separation.csv
    by_amino_acid_pair.csv
    by_spatial_distance.csv
  model_artifacts/
    booster_layer_00.json
    ...
    booster_layer_47.json
    feature_importance.csv
  layer_shards/
    layer_00/
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
        booster_layer_00.json
        feature_importance.csv
      shard_manifest.json
    ...
    layer_47/
```

Working fit-stage checkpoints and detailed logs remain cluster-side outside
the consolidated family. They are not copied into `delta_files/`.

### Row granularity and principal columns

- `per_protein.csv`: 48 x 552 = 26,496 rows, keyed by analysis, layer,
  protein ID, and `all_pairs` scope. Use the exact `PER_PROTEIN_COLUMNS`
  schema; omit MSE.
- `pooled_metrics.csv`: 48 x 2 = 96 rows, keyed by layer and
  development/test split. Include model and global-baseline R-squared, MAE,
  RMSE, bias, support, and coverage; omit duplicated mean-per-protein fields.
- `model_status.csv`: 48 rows, one shared fitted unit per layer. Record fit
  status, 441 fitting proteins, sampled fitting-row count, `S4`, 700 rounds,
  cap 5,000, and seed 42; OLS diagnostic fields are `NaN`.
- `runtime.csv`: 48 rows, one per layer, with fitted/evaluated support, total
  elapsed seconds, completion status, and peak RSS when available.
- `geometry_metrics.csv`: 26,496 rows, one per protein-layer, with complete
  pair coverage and geometry status.
- `geometry_summary.csv`: 48 x 2 x 14 = 1,344 rows, one per layer, split,
  and canonical geometry metric.
- sequence subgroup file: 48 x 6 = 288 rows.
- amino-acid subgroup file: 48 x 210 = 10,080 rows.
- spatial subgroup file: 48 x 5 = 240 rows.
- `feature_importance.csv`: 48 x 256 = 12,288 rows. Store gain and split
  count for every feature, including explicit zeroes. Do not call this directly
  comparable to OLS coefficients.
- `training_sample_manifest.csv`: 441 rows with protein identity, canonical
  role/source, total and selected pair counts, sampling seed, index digest,
  row weight, and total protein weight.
- `split_assignments.csv`: the frozen 552 eligible protein assignments and
  canonical split digest. Do not create a new split.
- `excluded_proteins.csv`: retain the two frozen manifest exclusions and any
  explicit preparation/runtime failure status. A runtime failure prevents a
  layer from being marked complete.

`run_manifest.json` must include output/protocol/manifest versions, all 48
requested and completed layers, split and sample digests, the complete XGBoost
configuration and its digest, feature order, target and precision definition,
selection provenance, Python/NumPy/XGBoost versions, XGBoost booster config,
code revision, input-root identifiers, timestamps, and final run status.

The metadata schemas are fixed as follows:

```text
split_assignments.csv:
manifest_version, master_manifest_digest, protein_id, source, n_residues,
n_pairs, size_bin, master_role, protein_split, outer_split_seed,
development_split_seed, split_assignment_digest

training_sample_manifest.csv:
analysis_name, protein_id, source, master_role, protein_split,
n_protein_pairs, n_selected_pairs, sampling_seed, sampling_policy,
indices_sha256, row_weight, total_protein_weight, sample_policy_digest

excluded_proteins.csv:
protein_id, source, eligibility_status, stage, layer, reason
```

The manifest must distinguish `master_manifest_digest`, the semantic digest of
all frozen cohort fields, from `split_assignment_digest`, the digest of the 552
eligible protein roles, and `sample_policy_digest`, the digest of all 441
development sampling rows. Never use these names interchangeably.

## 8. Booster and importance artifacts

Save each final booster with XGBoost's native JSON format immediately after the
fit. The shard manifest hashes it, and the consolidator copies it to the final
`model_artifacts/booster_layer_XX.json` path without modifying it.

Build feature importance from the final booster only:

- map `f0..f255` to `regression_protocol.FEATURE_NAMES`;
- save total gain and split count (`weight`) as separate columns;
- write zero for an unused feature;
- reject an out-of-range or unexpected feature key; and
- treat importance as supplementary descriptive output.

As an integrity test, reload every consolidated booster and verify that it
reproduces a small deterministic development prediction batch to numerical
tolerance. Keep the replay batch/checksum in the shard manifest, not as a
reader-facing pair-prediction file.

## 9. Restartability and failure behavior

Use atomic writes throughout. Each layer has these durable states:

1. input/config validation complete;
2. booster fitted and saved;
3. all-protein evaluation complete;
4. canonical layer shard checksummed and complete.

If a job times out after fitting, a retry may reuse the booster only when its
configuration, code, split, training-sample, layer-input, and XGBoost-version
digests all match. Otherwise it must refit. Detailed stage timestamps and
elapsed times go to stdout; only the compact one-row layer runtime enters the
final family.

Do not publish partially written canonical CSVs. On an exception, atomically
write a cluster-side failure JSON containing layer, phase, protein if
applicable, exception type/message, timestamp, and configuration/split digest,
then exit nonzero. The consolidator refuses to declare a family complete if a
layer is missing, failed, duplicated, or digest-inconsistent.

## 10. Consolidation and validation

`aggregate_shared_xgboost_final.py` is non-fitting and must not import or load
test pair features. It validates and concatenates already computed canonical
shards.

For every layer, verify:

- `shard_manifest.json` status and file hashes;
- exact output schemas and primary-key uniqueness;
- identical master-split, configuration, sample-policy, protocol, and feature
  digests;
- one booster and one status/runtime row;
- 552 per-protein and geometry rows;
- two pooled rows;
- expected geometry-summary and subgroup row counts;
- 100% pair coverage and finite core metrics for successful rows;
- `S4`, 5,000 cap, 700 rounds, and seed 42 in status/provenance;
- exact protein membership at every layer; and
- no test rows or fields in selection provenance artifacts.

Across all layers, enforce the exact consolidated row counts listed above.
Recompute selected pooled metrics from the mergeable sufficient statistics or
layer rows and compare with the saved values. Verify that global-baseline
support exactly equals model support and that conditional-baseline fields are
not applicable.

Only after all validations pass should the aggregator atomically publish the
top-level CSVs, 48 boosters, feature importance, metadata, and a
`run_status=complete` manifest. Preserve the validated layer shards for audit
and restartability.

## 11. Delta launchers and resource sizing

The layer launcher must follow the repository's required form:

```bash
#SBATCH --array=0-47%3
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64g
#SBATCH --time=04:00:00
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu

SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/shared_xgboost_final_layer.py"
```

Retain the current Delta account/CPU partition and module conventions only
after checking the live NCSA Delta documentation at implementation time. Use
the existing 64 GiB base256 XGBoost request for the initial development-only
smoke/pilot. The 5k final refit has 441 fitting proteins, so record pilot MaxRSS
and lower or raise memory only from observed use with reasonable headroom.
Delta may derive charges from requested memory, so do not preserve 64 GiB by
default after measurements show a smaller safe request.

The first cluster check must be development-only and write to a separate smoke
root; it must not load locked-test pair blocks. The first test-evaluating layer
is a retained canonical production task, not a disposable pilot. Limit initial
array concurrency to three to reduce simultaneous filesystem pressure, then
adjust only if observed I/O and memory support it.

The aggregation launcher requests one task, 8 GiB, and one hour initially,
matching the scale of existing canonical aggregators. Submit it with an
`afterok` dependency on the layer array, while retaining its independent
schema/checksum validation.

Both launchers must write timestamped stage logs under a workflow-specific
cluster log directory and pass `bash -n` before submission.

## 12. Test plan

Run all local tests with `./.venv`.

### Unit tests

1. Assert that final configuration loading accepts only the exact S4/5k/700
   specification and validates its digest.
2. Assert that the final estimator has no early stopping or evaluation set and
   receives exactly 700 estimators.
3. Assert frozen 330/111/111 roles and a 441-protein final fitting union.
4. Assert deterministic, layer-independent, target-independent capped indices,
   including all-pair behavior below 5k and exact index digests.
5. Assert equal total sample-weight mass per development protein.
6. Assert strict-lower-triangle ordering, directional feature order, 256
   columns, and float64 features/targets.
7. Assert that search and cap-sensitivity schemas contain no test columns and
   that their code paths never request test records.
8. Assert that the final phase cannot access test records until frozen
   selection/configuration validation and booster fitting have completed.
9. Compare canonical per-protein, pooled, subgroup, baseline, geometry, status,
   and runtime rows with small hand-computed synthetic examples.
10. Assert feature-importance mapping, including unused features and malformed
    booster keys.
11. Assert atomic shard restart behavior and rejection of mismatched booster,
    split, sample, code, or configuration digests.
12. Assert aggregator schemas, row counts, unique keys, hashes, layer coverage,
    protein membership, and failure on any incomplete shard.

### Integration checks

1. Compile all changed Python files with `./.venv/bin/python -m py_compile`.
2. Run the focused XGBoost and common-helper unit suites with
   `./.venv/bin/python -m unittest`.
3. Run a tiny synthetic end-to-end fit when local XGBoost is available;
   otherwise require an explicit dependency error and run it in the Delta
   environment before submission.
4. Validate both sbatch files with `bash -n`.
5. Run a development-only layer-0 Delta smoke task and inspect elapsed time,
   MaxRSS, feature dtype/shape, fitting support, booster replay, and all
   non-test output schemas.
6. Run the retained 48-layer production array once the smoke task passes.
7. Run consolidation, then independently audit the final tree against the
   expected row counts, hashes, finite-value rules, and booster replay checks.

## 13. Audit-closure checklist

The implementation is complete only when these shared-XGBoost findings are
closed in code:

- [ ] P1: frozen combined cohort is enforced and layer failures are explicit.
- [ ] P2: canonical master roles and locked test IDs are reused.
- [ ] P3: strict-lower-triangle C-alpha-distance rows are used.
- [ ] P4: ordered directional base256 features are used.
- [ ] P5: model inputs, targets, weights, and metrics are float64.
- [ ] P7: development proteins receive equal total fitting weight.
- [ ] P9: final 80% refit and development/test evaluation scopes are distinct.
- [ ] P10: all 48 layer models are implemented and required for completion.
- [ ] P11: search/cap code cannot calculate test metrics; final evaluation is
      downstream of frozen selection, with prior historical exposure disclosed.
- [ ] P12: versions, split/sample/configuration digests, code revision, and
      explicit failures are saved.
- [ ] O1: schema-versioned manifest and provenance are complete.
- [ ] O2: canonical all-protein `per_protein.csv` is emitted.
- [ ] O3/O6: compact pooled model/global-baseline metrics are emitted.
- [ ] O4: all three locked-test subgroup files are emitted.
- [ ] O7/O8: complete-coverage per-protein geometry and summary are emitted.
- [ ] O9: one replayable native booster per layer is emitted.
- [ ] O10: one final-model status row per layer is emitted.
- [ ] O11: one compact runtime row per layer is emitted.
- [ ] O12: all layer shards are hashed, validated, and consolidated.

After code and local tests pass, update both audits to mark the active
implementation accurately and state `48-layer production rerun pending`. After
the Delta outputs are copied and independently audited, update only the
artifact disposition to complete. Do not mark historical search/cap outputs as
canonical final models.

## 14. Completion criteria

The work is complete when:

1. protocol documentation and the frozen S4/5k/700 artifact agree;
2. no selection or sensitivity code loads or reports test predictions;
3. every final layer refits on the same 441 development proteins with the same
   deterministic 5k policy and exact 700-round S4 configuration;
4. all 552 proteins receive all-pair predictions at every layer;
5. all core, subgroup, geometry, booster, status, runtime, metadata, and shard
   outputs meet their schemas and exact row counts;
6. all tests, syntax checks, smoke checks, and consolidation validations pass;
7. the final family exists under
   `delta_files/shared_xgboost_base256_combined/`; and
8. both audits distinguish implementation completion, production completion,
   and the disclosed historical test exposure.

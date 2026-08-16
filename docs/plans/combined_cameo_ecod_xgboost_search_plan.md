# Combined CAMEO+ECOD XGBoost Hyperparameter Search Plan

## Goal

Build two independent XGBoost pipelines—a shared model and per-protein models—
with separate search and selected-model evaluation arrays. Use the fixed
TM-score-filtered combined CAMEO+ECOD dataset, select hyperparameters using
validation results only, and generate detailed regression outputs only for
freshly trained models using the selected configuration.

This document specifies the intended final behavior and output schema. It
supersedes the original implementation where the requirements below differ.

## Locked experiment design

- Use anchor layers 0, 23, and 47.
- Construct each unordered-pair feature as
  `0.5 * (pair_block[i,j,:] + pair_block[j,i,:])`.
- Materialize exactly 128 unstandardized feature channels in `float64`.
- Use absolute Cα pairwise distances as `float64` regression targets.
- Do not add scalar context, diagonal, axial, or sequence-separation features.
- Use `reg:squarederror`, histogram trees, 700 maximum boosting rounds, and
  validation RMSE with 30-round early stopping.
- Keep the early-stopped model trained on the 60% training split. Do not refit
  on training plus validation.
- Select configurations solely by mean per-protein validation R². Training and
  test metrics are reporting outputs and are never selection inputs.
- Search shared configurations only at 10k sampled training pairs per training
  protein.
- After shared-config selection, train fresh models at 5k, 10k, and 20k
  sampled training pairs per training protein. Do not run an all-pairs cap.
- Compute per-protein regression rows only for selected-config evaluations,
  never for search candidates.
- Do not compute or save geometry, MDS, TM-score, RMSD, spectral, stress,
  negative-distance, or triangle-violation statistics in either pipeline.
- Use the reference combined-regression output style in
  `delta_files/general_base256_combined/` unless a schema below explicitly
  differs.

## Shared infrastructure

Use `scripts/cluster_scripts/xgboost_search/` for the common helper, four Python
entrypoints, four matching sbatch files, and focused synthetic tests.

The helper module will:

1. Reuse the fixed 554-protein manifest and CAMEO/ECOD-specific Delta paths
   from the generalized combined-dataset code.
2. Require valid PDB coordinates and all three anchor-layer pair blocks so
   every array task operates on the same protein pool.
3. Assign five rank-based residue-count bins across the merged dataset.
4. Provide deterministic protein-level and pair-level splitting utilities.
5. Provide deterministic nested 5k, 10k, and 20k pair samples.
6. Support two explicit XGBoost row-weighting modes:
   - Shared models use protein-equal weights for both training loss and the
     validation RMSE used by early stopping. If a protein supplies `m` rows,
     each of its rows receives weight proportional to `1/m`.
   - Per-protein models use ordinary uniform rows. Their calls omit
     `sample_weight` and `sample_weight_eval_set`; they must not normalize the
     total Hessian mass of a protein to approximately one.
7. Compute per-protein and pooled R², MSE, MAE, RMSE, target variance, mean true
   distance, mean predicted distance, and mean prediction bias.
8. Provide streaming pooled statistics for full-protein evaluation.
9. Map XGBoost gain, weight, and cover importance back to channels 0–127.
10. Implement deterministic cross-layer winner selection without reading
    training or test columns.
11. Fail before fitting when search artifacts are missing, incomplete, or use
    inconsistent splits.
12. Fail clearly if a complete per-protein search configuration remains
    degenerate at boosting iteration zero for every protein.
13. Import XGBoost lazily and emit a clear dependency error when it is absent.

## XGBoost settings

All configurations use:

```text
objective = reg:squarederror
tree_method = hist
max_bin = 256
n_estimators = 700
early_stopping_rounds = 30
eval_metric = rmse
subsample = 0.8
colsample_bytree = 0.8
reg_lambda = 1.0
random_state = 42
n_jobs = SLURM_CPUS_PER_TASK
```

Shared grid:

| ID | Max depth | Learning rate | Min child weight |
|---|---:|---:|---:|
| S0 | 4 | 0.05 | 5 |
| S1 | 4 | 0.10 | 5 |
| S2 | 5 | 0.05 | 10 |
| S3 | 6 | 0.05 | 10 |
| S4 | 6 | 0.10 | 10 |
| S5 | 7 | 0.05 | 20 |

Per-protein grid:

| ID | Max depth | Learning rate | Min child weight |
|---|---:|---:|---:|
| P0 | 3 | 0.05 | 5 |
| P1 | 4 | 0.05 | 5 |
| P2 | 4 | 0.10 | 5 |
| P3 | 6 | 0.05 | 10 |

## Shared-model search

Map array tasks 0–2 to layers 0, 23, and 47.

For each task:

1. Create one deterministic size-stratified 60% training / 20% validation /
   20% test protein split.
2. Materialize one 10k-pairs-per-training-protein matrix and one uncapped
   validation matrix, then reuse both across all six candidates.
3. Fit every candidate with protein-equal training and validation weights.
4. Predict the actual sampled training rows, every full validation protein,
   and every full test protein. Stream full proteins where practical.
5. Checkpoint the aggregate search CSV after every completed configuration.
6. Delete each candidate booster after recording its metrics. Do not save
   candidate per-protein rows, channel importance, or boosters.
7. Keep one stable `summary.txt`. Do not create timestamped
   `summary_YYYYMMDD_HHMMSS.txt` checkpoint files.

### Exact shared-search outputs

Each `shared/search/layer_XX/` directory contains:

- `search_summary.csv`: one row per completed configuration; six rows when
  complete. Columns are:
  - Identity/configuration: `config_id`, `max_depth`, `learning_rate`,
    `min_child_weight`, `layer`.
  - Training reporting: `train_mean_per_protein_r2`, `train_pooled_r2`.
    These use only the sampled 10k-or-fewer rows actually supplied to fitting.
  - Validation reporting: `validation_mean_per_protein_r2`,
    `validation_pooled_r2`.
  - Test reporting: `test_mean_per_protein_r2`, `test_pooled_r2`.
  - Early stopping: `best_iteration`, `best_validation_rmse`.
  - Counts: `n_train_rows`, `n_train_proteins`, `n_validation_rows`,
    `n_validation_proteins`, `n_test_rows`, `n_test_proteins`.
  - Timing: `feature_materialization_seconds`, `fit_seconds`,
    `train_prediction_seconds`, `validation_prediction_seconds`,
    `test_materialization_seconds`, `test_prediction_seconds`,
    `metric_seconds`, `total_elapsed_seconds`.
  - Reproducibility: `split_digest`.
- `split_assignments.csv`: one row per usable protein with `protein_id`,
  `source`, `n_residues`, `n_pairs`, `size_bin`, and `split` (`train`,
  `validation`, or `test`).
- `skipped_proteins.csv`: `protein_id`, `source`, and `reason`.
- `configurations.json`: the six shared candidate definitions.
- `search_metadata.json`: layer/task identity, feature definition, weighting
  mode, 10k cap, fixed XGBoost settings, split digest, timestamps, counts, and
  completion status.
- `summary.txt`: final completed-configuration count, split digest, selected
  reporting notes, and total layer-task elapsed time.

There is no per-protein metrics CSV and no per-protein timing CSV in a search
directory.

## Shared selected-configuration cap evaluation

Map nine array tasks to the Cartesian product of layers `{0,23,47}` and
training caps `{5k,10k,20k}`. The task order is layer-major, with caps ordered
5k, 10k, then 20k. There is no all-pairs task.

Every task loads all three completed shared-search summaries and independently
selects the same configuration:

1. Average each configuration's mean per-protein validation R² equally across
   the three layers.
2. Find the maximum average score.
3. Treat configurations within 0.001 R² of that maximum as tied.
4. Break ties by lower depth, then lower summed search fit time, then lower
   configuration ID.
5. Never inspect training or test columns.

Each task trains a fresh early-stopped booster for its assigned layer and cap.
The sampled training sets remain nested: 5k is contained in 10k, which is
contained in 20k.

### Regression evaluation scopes

- A training protein's regression row uses only the cap-specific sampled rows
  that were actually supplied to `model.fit`. Its `evaluation_scope` is
  `sampled_training_rows`.
- Validation and test protein regression rows use every unordered pair in the
  protein. Their `evaluation_scope` is `full_protein_pairs`.

### Exact shared selected-model outputs

The shared selected-model root contains:

- `selected_config.json`: canonical validation-only selection provenance,
  written by task 0.
- `cross_layer_selection_table.csv`: one row per shared configuration with its
  three layer validation scores, equal-layer mean validation score, depth, and
  summed search fit time.

Each `shared/winner_cap_sweep/layer_XX/cap_{5k|10k|20k}/` directory contains:

- `per_protein.csv`: exactly one regression row per dataset protein. Columns:
  - Model identity: `variant`, `feature_dim`, `config_id`, `cap`, `layer`.
  - Membership/scope: `split` (`train`, `validation`, or `test`) and
    `evaluation_scope` (`sampled_training_rows` or `full_protein_pairs`).
  - Protein identity: `source`, `protein_id`, `n_residues`, `size_bin`,
    `n_pairs`.
  - Regression metrics: `target_variance`, `per_protein_r2`,
    `per_protein_mse`, `per_protein_mae`, `per_protein_rmse`,
    `mean_true_distance`, `mean_predicted_distance`, and
    `mean_error_pred_minus_true`.
- `split_summary.csv`: three rows—train, validation, and test—for this
  layer/cap. Columns:
  - `variant`, `feature_dim`, `config_id`, `cap`, `layer`, `split`.
  - `n_proteins`, `n_pairs`.
  - `per_protein_r2_mean`, `per_protein_r2_std`,
    `per_protein_mae_mean`, `per_protein_mae_std`,
    `per_protein_rmse_mean`, `per_protein_rmse_std`.
  - `pair_weighted_r2`, `pair_weighted_mse`, `pair_weighted_mae`,
    `pair_weighted_rmse`, and `mean_error_pred_minus_true`.
- `channel_importance.csv`: 384 rows—128 channels for each of gain, weight,
  and cover—with `layer`, `cap`, `config_id`, `importance_type`, `channel`,
  `feature_name`, and `importance`.
- `booster.json`: the fresh early-stopped model for this layer/cap.
- `timing.csv`: one task-level row with model identity; row/protein counts;
  training and validation materialization; fit time; train, validation, and
  test prediction/materialization/metric times; best iteration; best
  validation RMSE; and total elapsed time.
- `selection_provenance.json`: validation-only selection rule, tolerance,
  tie-break order, selected configuration, and selected score.
- `cross_layer_selection_table.csv`: task-local copy of the canonical
  selection table.
- `split_assignments.csv`, `skipped_proteins.csv`, and `run_metadata.json`:
  reproducibility and completion metadata, including the explicit weighting
  and evaluation scopes.
- `summary.txt`: selected configuration, cap, layer, best iteration, best
  validation RMSE, split digest, completed per-protein regression count, and
  total elapsed time.

The task does not write `geometry_metrics.csv`, `geometry_summary.csv`, or an
exact-triangle output. It does not perform complete-matrix prediction solely
for structural reconstruction.

Across the nine task directories, `split_summary.csv` provides aggregate
train/validation/test comparisons for all three caps and all three layers. Only
the selected configuration receives these cap-sweep fits and detailed rows;
search candidates do not.

## Per-protein search

Map array tasks 0–2 to layers 0, 23, and 47.

For each task:

1. From proteins valid at all anchor layers, deterministically choose exactly
   10 proteins from each of the five size bins.
2. Save the fixed 50-protein manifest with source, size, pair count, and bin.
3. Create one deterministic 60% training / 20% validation / 20% test pair
   split per protein and persist split counts and index digests.
4. Reuse those pair splits across layers and configurations.
5. Use every training pair; do not apply a pair cap.
6. Fit all four candidates independently for every protein, using ordinary
   unweighted rows and validation pairs only for early stopping.
7. Predict and aggregate training, validation, and test R² for each candidate.
8. Checkpoint only after all 50 proteins complete for a configuration and
   refuse to complete an incomplete configuration.
9. Do not save candidate per-protein regression rows, per-protein timing rows,
   boosters, or importance.
10. Keep one stable `summary.txt`; do not create timestamped summary files.

### Exact per-protein-search outputs

Each `per_protein/search/layer_XX/` directory contains:

- `search_summary.csv`: one row per completed configuration; four rows when
  complete. Columns are:
  - Identity/configuration: `config_id`, `max_depth`, `learning_rate`,
    `min_child_weight`, `layer`, `n_proteins`.
  - Training reporting: `train_mean_per_protein_r2`, `train_pooled_r2`.
  - Validation reporting: `validation_mean_per_protein_r2`,
    `validation_pooled_r2`.
  - Test reporting: `test_mean_per_protein_r2`, `test_pooled_r2`.
  - Early stopping: `best_iteration_mean`, `best_iteration_median`,
    `best_iteration_min`, `best_iteration_max`,
    `best_validation_rmse_mean`.
  - Aggregate configuration timing required for comparison and tie-breaking:
    `materialization_seconds`, `fit_seconds`, `train_prediction_seconds`,
    `validation_prediction_seconds`, `test_prediction_seconds`,
    `metric_seconds`, `total_elapsed_seconds`.
  - Reproducibility: `split_digest`.
- `protein_manifest.csv`: 50 rows with `protein_id`, `source`, `n_residues`,
  `n_pairs`, and `size_bin`.
- `pair_split_summary.csv`: 150 rows, one per protein and pair split, with
  `protein_id`, `split` (`train`, `validation`, or `test`), `n_pairs`, and
  `indices_sha256`.
- `skipped_proteins.csv`: `protein_id`, `source`, and `reason`.
- `configurations.json`: the four per-protein candidate definitions.
- `search_metadata.json`: layer/task identity, feature definition,
  `uniform_rows` weighting mode, pair-split digest, fixed XGBoost settings,
  timestamps, counts, and completion status.
- `summary.txt`: completed-configuration count, split digest, total number of
  completed fits, iteration-zero diagnostic status, and total layer-task
  elapsed time.

There is no `per_protein_timing.csv`. The task total is kept in `summary.txt`,
while configuration-level aggregate time remains in `search_summary.csv`
because `fit_seconds` is required by the winner tie-break.

## Per-protein selected-configuration evaluation

Map array tasks 0–2 to layers 0, 23, and 47. Load all three completed search
summaries and apply the same validation-only cross-layer selection and tie-break
policy used by the shared pipeline.

For each of the 50 proteins at the assigned layer:

1. Train a fresh selected-configuration model on the original 60% training
   pairs, using uniform rows and early stopping on the original 20% validation
   pairs.
2. Predict and save training, validation, and held-out test pair metrics.
3. Save gain/weight/cover importance per protein and equal-protein aggregate
   channel summaries.
4. Do not save booster JSON files.

### Exact per-protein selected-model outputs

The per-protein selected-model root contains:

- `selected_config.json`: canonical validation-only selection provenance,
  written by task 0.

Each `per_protein/winner_evaluation/layer_XX/` directory contains:

- `per_protein.csv`: 150 rows—training, validation, and test for each of the 50
  proteins. Columns follow the shared selected-model regression schema, without
  `cap`: `variant`, `feature_dim`, `config_id`, `layer`, `split`,
  `evaluation_scope` (`pair_split_rows`), `source`, `protein_id`, `n_residues`,
  `size_bin`, `n_pairs`, `target_variance`, `per_protein_r2`,
  `per_protein_mse`, `per_protein_mae`, `per_protein_rmse`,
  `mean_true_distance`, `mean_predicted_distance`, and
  `mean_error_pred_minus_true`.
- `split_summary.csv`: three rows—train, validation, and test—with the same
  aggregate regression columns as the shared `split_summary.csv`, excluding
  `cap`.
- `per_protein_channel_importance.csv`: 50 × 128 × 3 rows with model/protein
  identity, importance type, channel, feature name, and importance.
- `aggregate_channel_importance.csv`: 128 × 3 rows containing equal-protein
  mean and standard deviation for each channel and importance type.
- `selection_provenance.json` and `cross_layer_selection_table.csv`: task-local
  selected-configuration evidence.
- `pair_split_summary.csv`, `skipped_proteins.csv`, and `run_metadata.json`:
  reproducibility and completion metadata, including uniform-row weighting and
  regression scopes.
- `summary.txt`: selected configuration, layer, completed fit/regression
  counts, split digest, best-iteration summary, and total layer-task elapsed
  time.

There is no per-protein timing CSV. Only the selected configuration receives
per-protein regression and importance outputs. The task does not write
`geometry_metrics.csv`, `geometry_summary.csv`, or an exact-triangle output.

## Output layout

```text
$HOME/csv_files/xgboost_hyperparameter_search/
  shared/
    search/
      layer_00/
      layer_23/
      layer_47/
    winner_cap_sweep/
      selected_config.json
      cross_layer_selection_table.csv
      layer_00/{cap_5k,cap_10k,cap_20k}/
      layer_23/{cap_5k,cap_10k,cap_20k}/
      layer_47/{cap_5k,cap_10k,cap_20k}/
  per_protein/
    search/
      layer_00/
      layer_23/
      layer_47/
    winner_evaluation/
      selected_config.json
      layer_00/
      layer_23/
      layer_47/
  logs/
```

## Documentation and Delta launchers

Update `AGENTS.md` during implementation to require:

1. Every new or modified sbatch file includes:
   - `#SBATCH --mail-type=BEGIN,END,FAIL`
   - `#SBATCH --mail-user=ikhire3@gatech.edu`
2. Plans clearly enumerate exact output paths, filenames, row granularity,
   split/evaluation scope, and principal columns.
3. Regression work follows the output organization and column style in
   `delta_files/general_base256_combined/` unless the user explicitly requests
   another schema.

Apply the notification directives to the four XGBoost launchers in this change.
Do not retroactively edit unrelated existing sbatch files.

All four launchers use the Delta CPU account and partition, scratch constraint,
`module load python`, `~/protein_venv`, eight CPU threads, timestamped array
logs, and the repository's `SCRIPT_DIR`/`srun` convention.

| Launcher | Array | CPUs | Memory | Time |
|---|---|---:|---:|---:|
| Shared search | `0-2%3` | 8 | 128 GB | 10 h |
| Shared winner/cap sweep | `0-8%3` | 8 | 128 GB | 10 h |
| Per-protein search | `0-2%3` | 8 | 32 GB | 10 h |
| Per-protein winner | `0-2%3` | 8 | 32 GB | 10 h |

Submit both searches first. Submit each selected-model array only after all
three tasks of its corresponding search finish successfully.

## Verification

Add or update local synthetic tests covering:

- Exact float64 `(n_pairs,128)` averaged-symmetric feature construction.
- Deterministic, disjoint, exhaustive, size-stratified shared splits.
- Deterministic, disjoint, exhaustive per-protein pair splits.
- Equal total sample-weight mass for shared-model proteins.
- Uniform/unweighted rows for per-protein fits, including omission of
  `sample_weight` and validation sample weights.
- Deterministic nested 5k/10k/20k samples.
- Search mappings for three layers, shared winner mappings for nine layer/cap
  tasks, and per-protein winner mappings for three layers.
- Validation-only winner selection and exact tie-break order.
- Training R² aggregation from actual sampled/shared rows and pair-split
  per-protein rows.
- Streaming pooled R² parity with direct concatenation.
- Early failure for missing, incomplete, or inconsistent search artifacts.
- Failure when an entire completed per-protein candidate remains at boosting
  iteration zero for all 50 proteins.
- Absence of search per-protein metric/timing outputs and timestamped summary
  files.

Finally:

1. Compile all XGBoost Python files with `./.venv/bin/python -m py_compile`.
2. Run the focused helper test suite in `./.venv`.
3. Validate every XGBoost sbatch file with `bash -n`.
4. Audit each sbatch filename against its Python entrypoint, requested
   resources, array range, and email notification directives.
5. Verify the explicit dependency error if local `./.venv` lacks XGBoost.
6. Confirm each output writer uses the exact filenames, split labels, scopes,
   and column sets specified above.
7. Record the significant implementation briefly in `branch-analysis.md`.
8. Commit and push the completed implementation.

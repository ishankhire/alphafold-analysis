# Combined CAMEO+ECOD XGBoost Hyperparameter Search Plan

## Goal

Build two independent XGBoost search pipelines—a shared model and per-protein
models—each with a search-array entrypoint and a winner-evaluation entrypoint.
Use the fixed TM-score-filtered combined CAMEO+ECOD dataset, select
hyperparameters using validation results only, and reserve geometry analysis for
the selected models.

## Locked experiment design

- Use anchor layers 0, 23, and 47.
- Construct each unordered-pair feature as
  `0.5 * (pair_block[i,j,:] + pair_block[j,i,:])`.
- Materialize exactly 128 unstandardized feature channels in `float64`.
- Use absolute Cα pairwise distances as `float64` regression targets.
- Do not add scalar context, diagonal, axial, or sequence-separation features.
- Use `reg:squarederror`, histogram trees, 700 maximum boosting rounds, and
  validation RMSE with 30-round early stopping.
- Keep the early-stopped model trained on the 60% fit split. Do not refit on
  fit plus validation.
- Select configurations solely by mean per-protein validation R². Test metrics
  are reporting outputs and never selection inputs.

## Shared infrastructure

Create `scripts/cluster_scripts/xgboost_search/` with a common helper module,
four Python entrypoints, four matching sbatch files, and focused synthetic
tests.

The helper module should:

1. Reuse the fixed 554-protein manifest and CAMEO/ECOD-specific Delta paths
   from the generalized combined-dataset code.
2. Require valid PDB coordinates and all three anchor-layer pair blocks so
   every array task operates on the same protein pool.
3. Assign five rank-based residue-count bins across the merged dataset.
4. Provide deterministic protein-level and pair-level splitting utilities.
5. Provide deterministic nested 5k, 10k, and 20k pair samples.
6. Give each shared-model protein equal total training and validation loss
   mass by assigning row weights proportional to `1 / rows_from_protein`.
7. Compute per-protein and pooled R², MAE, and RMSE, including streaming pooled
   statistics for full test-protein evaluation.
8. Map XGBoost gain, weight, and cover importance back to channels 0–127.
9. Implement deterministic cross-layer winner selection without reading test
   columns.
10. Implement MDS/TM-score geometry diagnostics without exact triangle checks.
11. Fail before fitting when search artifacts are missing, incomplete, or use
    inconsistent splits.
12. Import XGBoost lazily and emit a clear dependency error when it is absent.

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

1. Create one deterministic size-stratified 60% fit / 20% validation / 20%
   test protein split.
2. Save split assignments, skipped-protein records, split digest, fixed model
   settings, and grid metadata.
3. Materialize one 10k-pairs-per-fit-protein matrix and one uncapped validation
   matrix, then reuse both across all six candidates.
4. Fit every candidate with protein-equal fit and validation weights.
5. Predict all validation pairs and stream every full test protein through
   feature construction and prediction.
6. Record validation and test mean per-protein and pooled R², best iteration,
   best validation RMSE, row/protein counts, and detailed timings.
7. Checkpoint the search CSV after every completed configuration.
8. Delete each candidate booster after recording its metrics. Do not save
   candidate geometry or importance.

## Shared winner and cap sweep

Map 12 array tasks to the Cartesian product of layers `{0,23,47}` and fit caps
`{5k,10k,20k,all}`.

Every task should load all three completed shared-search summaries and select
the same winner as follows:

1. Average each configuration's mean per-protein validation R² equally across
   the three layers.
2. Find the maximum average score.
3. Treat configurations within 0.001 R² of that maximum as tied.
4. Break ties by lower depth, then lower summed search fit time, then lower
   configuration ID.
5. Never inspect test columns.

Then train a fresh early-stopped booster for the task's layer and cap. The cap
changes the training data; do not reuse a search booster.

Save:

- Per-protein and aggregate validation/test R², MAE, and RMSE.
- Complete materialization, fit, prediction, metric, geometry, and total timing.
- Booster JSON plus gain/weight/cover channel importance.
- Winner selection provenance and the cross-layer selection table.
- Per-protein and aggregate validation/test geometry metrics.

For geometry, predict complete symmetric distance matrices, count negative raw
predictions, clamp negatives to zero only for geometry, and compute classical
MDS, reflection-aware Kabsch RMSD, TM-score, MDS stress, top-three positive
spectral mass fraction, and negative-eigenvalue diagnostics. Do not compute
triangle-violation metrics.

Task 0 writes the canonical shared `selected_config.json` and canonical
cross-layer selection table.

## Per-protein search

Map array tasks 0–2 to layers 0, 23, and 47.

For each task:

1. From proteins valid at all anchor layers, deterministically choose exactly
   10 proteins from each of the five size bins.
2. Save the fixed 50-protein manifest with source, size, pair count, and bin.
3. Create one deterministic 60% fit / 20% validation / 20% test pair split per
   protein and persist split counts and index digests.
4. Reuse those pair splits across layers and configurations.
5. Use every fit pair; do not apply a pair cap.
6. Fit all four candidates independently for every protein, using the validation
   pairs only for early stopping.
7. Report mean and pooled validation/test R² for each candidate, best-iteration
   summaries, and aggregate timings.
8. Save per-protein/configuration timing rows and checkpoint after each complete
   configuration.
9. Refuse to complete unless all 50 proteins finished for every configuration.
10. Do not save candidate boosters, importance, or geometry.

## Per-protein winner evaluation

Map array tasks 0–2 to layers 0, 23, and 47. Load all three completed search
summaries and apply the same validation-only cross-layer selection and tie-break
policy used by the shared pipeline.

For each of the 50 proteins at the assigned layer:

1. Train a fresh selected-configuration model on the original 60% fit pairs,
   with early stopping on the original 20% validation pairs.
2. Save fit, validation, and held-out test R², MAE, and RMSE.
3. Save fit and prediction timings plus best-iteration information.
4. Save gain/weight/cover importance per protein and equal-protein aggregate
   channel summaries.
5. Do not save booster JSON files.
6. Predict all pairs in that protein to create a complete matrix for geometry.

Label geometry explicitly as `all_pairs_model_predictions`: regression test R²
is held out at the pair level, while geometry is not a wholly held-out-protein
result. Save the same triangle-free MDS/TM metrics as the shared winner. Task 0
writes the canonical per-protein `selected_config.json`.

## Output layout

```text
$HOME/csv_files/xgboost_hyperparameter_search/
  shared/
    search/
    winner_cap_sweep/
  per_protein/
    search/
    winner_evaluation/
  logs/
```

## Delta launchers

Create four launchers using the Delta CPU account and partition, scratch
constraint, `module load python`, `~/protein_venv`, eight CPU threads,
timestamped array logs, and the repository's `SCRIPT_DIR`/`srun` convention.

| Launcher | Array | CPUs | Memory | Time |
|---|---|---:|---:|---:|
| Shared search | `0-2%3` | 8 | 128 GB | 10 h |
| Shared winner/cap sweep | `0-11%3` | 8 | 128 GB | 10 h |
| Per-protein search | `0-2%3` | 8 | 32 GB | 10 h |
| Per-protein winner | `0-2%3` | 8 | 32 GB | 10 h |

Submit both searches first. Submit each winner array only after all three tasks
of its corresponding search finish successfully.

## Verification

Add local synthetic tests covering:

- Exact float64 `(n_pairs,128)` averaged-symmetric feature construction.
- Deterministic, disjoint, exhaustive, size-stratified shared splits.
- Deterministic, disjoint, exhaustive per-protein pair splits.
- Equal total sample-weight mass per protein.
- Deterministic nested 5k/10k/20k samples.
- All search and winner array-task mappings.
- Validation-only winner selection and exact tie-break order.
- Streaming pooled R² parity with direct concatenation.
- Early failure for missing, incomplete, or inconsistent search artifacts.

Finally:

1. Compile all new Python files with `./.venv/bin/python -m py_compile`.
2. Run the focused helper test suite in `./.venv`.
3. Validate every sbatch file with `bash -n`.
4. Audit each sbatch filename against its Python entrypoint and requested
   resources.
5. Verify the explicit dependency error because local `./.venv` lacks XGBoost.
6. Record the significant implementation briefly in `branch-analysis.md`.
7. Commit and push the completed change.

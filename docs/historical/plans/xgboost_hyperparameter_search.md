> Historical reference; not current execution instructions. See [current sources and status](../../project_status.md).

A previous agent produced the plan below to accomplish the user's task. Implement the plan in a fresh context. Treat the plan as the source of user intent, re-read files as needed, and carry the work through implementation and verification.

# Combined CAMEO+ECOD XGBoost Hyperparameter Searches

## Summary

Implement two independent XGBoost search pipelines—shared and per-protein—each with a search-array entrypoint and a winner-evaluation entrypoint.

- Features: 128 float64 averaged-symmetric pair channels, with no scalar context features and no standardization.
- Anchor layers: 0, 23, and 47.
- Shared split: size-stratified 60% fit / 20% validation / 20% test at the protein level.
- Per-protein split: deterministic 60% fit / 20% validation / 20% test at the pair level.
- Search at 10k fit pairs per shared-model protein only.
- Select configurations strictly by validation mean per-protein R²; test R² is reported but never used for selection.
- After selection, train fresh shared models—not reuse the 10k fitted booster—at 5k, 10k, 20k, and all fit pairs for every anchor layer.
- Keep the early-stopped 60%-fit models for final evaluation; do not refit on fit+validation.
- Compute MDS/TM-score only for winner evaluations.

## Implementation

### Shared infrastructure

Add a dedicated `scripts/cluster_scripts/xgboost_search/` package containing a common helper module, four Python entrypoints, four matching sbatch launchers, and focused tests.

The common module will:

- Reuse the fixed TM-score-filtered combined CAMEO+ECOD manifest and source-specific Delta path resolution from the current generalized code.
- Load Cα coordinates and absolute pairwise-distance targets, materializing regression targets as float64.
- Build each unordered-pair feature as:
  `0.5 * (pair_block[i,j,:] + pair_block[j,i,:])`, with arithmetic and output in float64.
- Validate the feature width is exactly 128 and that no scalar, diagonal, axial, or sequence-separation features are present.
- Assign five residue-count size bins across the merged dataset.
- Create one deterministic shared 60/20/20 protein split, reused across layers and configurations.
- Create deterministic nested fit-pair samples so each protein’s 5k sample is contained in its 10k sample, which is contained in its 20k sample.
- Compute R², MAE, RMSE, pooled streaming statistics, equal-protein sample weights, XGBoost importance mappings, timing records, and deterministic winner selection.
- Give every shared-model fit protein equal total training loss mass: if a protein supplies `m` sampled rows, each row receives weight proportional to `1/m`. Apply the same equal-protein rule to validation weights used for early stopping.
- Use full, uncapped validation and test pairs. XGBoost prediction traverses hundreds of trees rather than multiplying linear coefficients, so this inference is material but still substantially cheaper than histogram construction and boosting. Validation is costlier than ordinary inference because it is evaluated after every boosting round.
- Stream test proteins through prediction instead of retaining one enormous test matrix where possible.
- Fail clearly if XGBoost is unavailable, required layer files are missing, splits differ across tasks, or expected search outputs are incomplete.

### Hyperparameter grids

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
n_jobs = allocated SLURM CPUs
```

Shared grid:

| ID | Depth | Learning rate | Min child weight |
|---|---:|---:|---:|
| S0 | 4 | 0.05 | 5 |
| S1 | 4 | 0.10 | 5 |
| S2 | 5 | 0.05 | 10 |
| S3 | 6 | 0.05 | 10 |
| S4 | 6 | 0.10 | 10 |
| S5 | 7 | 0.05 | 20 |

Per-protein grid:

| ID | Depth | Learning rate | Min child weight |
|---|---:|---:|---:|
| P0 | 3 | 0.05 | 5 |
| P1 | 4 | 0.05 | 5 |
| P2 | 4 | 0.10 | 5 |
| P3 | 6 | 0.05 | 10 |

### Shared search array

`shared_xgboost_hyperparameter_search.py` will map array tasks 0–2 to layers 0, 23, and 47.

For each layer it will:

- Materialize one 10k-per-fit-protein matrix and full validation data, then reuse them for all six configurations.
- Fit each candidate with protein-equal training and validation weights.
- Use validation RMSE only for early stopping.
- Predict all validation and test pairs.
- Save, for every configuration:
  - Validation mean per-protein and pooled pair-weighted R².
  - Test mean per-protein and pooled pair-weighted R².
  - Best iteration and best validation RMSE.
  - Fit-row and protein counts.
  - Feature-materialization, `model.fit`, validation prediction, test prediction, metric, and total elapsed seconds.
- Flush/checkpoint the layer search CSV after every completed configuration.
- Save layer-specific split assignments, skipped-protein records, configuration metadata, and a timestamped text summary.
- Delete candidate boosters after their metrics are recorded; candidate models and geometry will not be saved.

### Shared winner and cap-sweep array

`shared_xgboost_winner_cap_sweep.py` will map 12 tasks to the Cartesian product of layers `{0,23,47}` and caps `{5k,10k,20k}`.

Every task will read all three shared-search summaries and independently make the same deterministic selection:

1. Average each configuration’s validation mean per-protein R² equally across layers 0, 23, and 47.
2. Find the highest average.
3. Treat configurations within 0.001 R² of that maximum as tied.
4. Select lower depth, then shorter summed fit time, then lower configuration ID.
5. Never inspect test columns during selection.

Each task will train a fresh booster using the selected hyperparameters and its assigned layer/cap. The cap changes the training data, not just evaluation; each cap therefore gets a new early-stopped model and may produce a different best iteration.

For full validation and test proteins, save:

- Aggregate mean per-protein and pooled R², MAE, and RMSE.
- Individual protein regression rows.
- Full timing breakdown, allowing direct 5k/10k/20k comparisons.
- Booster JSON and gain/weight/cover channel importance.
- Winner configuration and selection provenance.

For both validation and test proteins, construct full model-predicted symmetric distance matrices, clamp negative distances to zero for geometry only, and save:

- TM-score and Kabsch-aligned RMSD, checking both MDS reflections.
- MDS stress.
- Fraction of positive spectral mass in the top three eigenvalues.
- Negative eigenvalue count/fraction and negative prediction count.
- Per-protein and aggregate geometry rows.

Do not compute exact triangle-violation metrics. Array task 0 will additionally write the canonical shared `selected_config.json` and cross-layer selection table; all other tasks independently verify that they derived the same configuration.

### Per-protein search array

`per_protein_xgboost_hyperparameter_search.py` will map tasks 0–2 to layers 0, 23, and 47.

Before sampling, it will require proteins to have valid PDB data and all three anchor-layer pair blocks. It will then:

- Assign five size bins and select exactly 10 proteins per bin using a deterministic seed, treating CAMEO and ECOD as one dataset.
- Save the fixed 50-protein manifest, including source, size, pair count, and size bin.
- Generate one deterministic 60/20/20 pair split per protein and reuse it across layers/configurations.
- Use all fit pairs without a cap.
- Fit four independent candidate boosters per protein per layer, with validation-only early stopping and no model refit.
- Give every pair equal weight within its protein; give every protein equal weight when aggregating the configuration-selection metric.
- Save one candidate summary row per configuration containing:
  - Mean validation R² over the 50 proteins.
  - Pooled validation R² over their validation pairs.
  - Mean test R² over the 50 proteins.
  - Pooled test R² over their test pairs.
  - Best-iteration summaries and complete timing aggregates.
- Save a separate timing CSV with materialization, fit, prediction, and total seconds for every protein/configuration fit.
- Checkpoint after every completed configuration and fail rather than silently selecting from an incomplete 50-protein set.
- Save no candidate boosters, feature importance, or geometry.

### Per-protein winner array

`per_protein_xgboost_winner_evaluation.py` will use three tasks for layers 0, 23, and 47.

Each task will read all three per-protein search summaries and select one global configuration using the same validation-only mean, 0.001 tolerance, lower-depth, and runtime tie-breaking policy. It will then rerun that configuration for the 50 proteins at its assigned layer using the original 60/20/20 pair splits, without a cap or 80% refit.

It will save:

- Individual fit/validation/test R², MAE, and RMSE rows.
- Mean and pooled aggregate metrics.
- Per-protein fit and prediction timing.
- Per-protein gain/weight/cover importance and equal-protein aggregate channel-importance summaries.
- No booster JSON files.

For geometry, the selected model will predict all pairs in its own protein so MDS receives a complete matrix. Outputs will explicitly label this as `all_pairs_model_predictions`: regression test R² remains held out at the pair level, but per-protein geometry is not a wholly held-out-protein result. Save the same MDS/TM metrics as the shared winner, without triangle analysis. Task 0 will write the canonical per-protein `selected_config.json`.

### Output layout and SLURM

Use:

```text
$HOME/csv_files/xgboost_hyperparameter_search/
  shared/
    search/
    winner_cap_sweep/
  per_protein/
    search/
    winner_evaluation/
```

Provide four Delta-ready sbatch files:

- Shared search: `--array=0-2%3`, 8 CPUs, 128 GB, 10 hours.
- Shared winner/cap sweep: `--array=0-11%3`, 8 CPUs, 128 GB, 10 hours.
- Per-protein search: `--array=0-2%3`, 8 CPUs, 32 GB, 10 hours.
- Per-protein winner evaluation: `--array=0-2%3`, 8 CPUs, 32 GB, 10 hours.

All launchers will use the Delta CPU account/partition, scratch constraint, `module load python`, `~/protein_venv`, thread-count environment variables, timestamped array logs, and:

```bash
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/<script>.py"
```

The two searches are submitted first. Their corresponding winner arrays are submitted only after their three search tasks finish successfully; winner scripts fail fast if any anchor-layer result is absent. No production six-layer regression code is included in this change.

## Validation and Acceptance

Add focused synthetic tests that do not require cluster data:

- Averaged-symmetric feature construction produces float64 arrays of shape `(n_pairs,128)` and contains no scalar columns.
- Shared 60/20/20 splits are deterministic, disjoint, size-stratified, and exhaustive.
- Per-protein pair splits are deterministic, disjoint, exhaustive, and reused across layers.
- Protein-equal weights assign identical total loss mass to every protein.
- 5k/10k/20k samples are nested and deterministic.
- Task IDs map correctly to three search layers, twelve layer-cap combinations, and three winner layers.
- Winner selection ignores test columns and applies the 0.001/lower-depth/runtime tie-break exactly.
- Streaming pooled R² matches direct concatenated R² on synthetic data.
- Missing or inconsistent search artifacts cause winner jobs to fail before fitting.

Run local syntax compilation and helper tests in `./.venv`. Since local `./.venv` currently lacks XGBoost, full fitting validation will be limited to a cluster-environment import check and an initial Delta task; the scripts will provide an explicit dependency error if `~/protein_venv` lacks XGBoost.

Audit every new sbatch filename against its Python entrypoint, record the significant addition briefly in `branch-analysis.md`, then follow the repository-required add/commit/push workflow.

## Locked Assumptions

- “Shared linear model” in the latest clarification means the shared XGBoost model.
- Search is performed only at 10k. The selected hyperparameter configuration—not the already-fitted 10k booster—is reused to train fresh 5k/10k/20k models.
- Shared proteins have equal training and validation loss weight.
- Validation mean per-protein R² is the sole selection score; test scores are output-only.
- Candidate outputs contain aggregate mean and pooled R²; detailed per-protein rows are retained for winners.
- The 60%-fit early-stopped model is the final evaluated model; validation rows are not folded back into training.
- Shared cap sweeps and MDS/TM-score cover all three anchor layers and all four caps.
- Per-protein hyperparameter search has no pair cap and no cap sweep.
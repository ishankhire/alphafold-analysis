# Delta cluster guide

Use the live [NCSA Delta documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/)
for current partitions, accounts, modules, limits, and filesystem availability.
This guide defines this repository's deployment and submission conventions.
[Current sources and status](../project_status.md) identifies the active results;
[datasets.md](../../datasets.md) defines the input pool and protein input paths.

## Agent access and permissions

Agents may connect to Delta, inspect files and manifests, read logs, and monitor
jobs without asking for permission. Use the configured `ssh delta` alias and an
authenticated Kerberos/Duo session. Report authentication failures rather than
assuming remote work is incomplete.

Agents may make cluster changes needed for the user's task without asking,
including submitting or cancelling jobs, deploying or editing files, and
creating directories. Always ask the user's permission before deleting any
files on the cluster. Clearly report the exact cluster changes made in chat,
including affected paths and job IDs where applicable; documentation updates
do not substitute for this chat report. Keep login-node inspection lightweight;
analysis belongs in a Slurm job.

## Repository deployment layout on Delta

Source launchers are colocated with workflows under `scripts/cluster_scripts/`,
including `cross_validation/`, `generalized_model_ecod/`, `per_protein_scripts/`,
`gated_linear_probes/`, `diagnostics/`, and `xgboost_search/`.

Deploy cluster Python entrypoints, launchers, shared helpers, and configuration
files for active canonical workflows flat into
`/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`. Repository source
subdirectories are for organization; do not reproduce them in the deployment.
Retain `regression_protocol_data/` for the frozen master split manifest.

```text
$HOME/
  openfold_interpretability_canonical/
    <entrypoints, launchers, helpers, configuration, workflow READMEs>
    regression_protocol_data/regression_master_split_v1.csv
  csv_files/
    regression_5fold_v1/             # legacy models and compact outputs
/work/hdd/bfna/ikhire/
  regression_5fold_v1/               # final, analysis-ready results and artifacts
    predicted_distances_5fold_v1/    # packed OOF distance exports
    tm_score_optimized_v1/          # optimized scoring outputs
  diagnostic_experiments/
    <experiment_name>/              # exploratory and diagnostic outputs
```

## CSV and result storage

The default Delta CSV output root is `/work/hdd/bfna/ikhire/`:

- `regression_5fold_v1/` holds final, analysis-ready regression results and their
  associated model artifacts, exports, and provenance. Derived analysis tables
  belong under `regression_5fold_v1/analysis/<analysis_name>/`.
- `diagnostic_experiments/<experiment_name>/` holds exploratory runs and
  diagnostic experiments, with a separate directory per experiment.

New workflows must select the appropriate output directory explicitly.
For workflows that append `regression_5fold_v1/` to `CSV_ROOT`, use
`CSV_ROOT=/work/hdd/bfna/ikhire` for new final-result outputs. Diagnostics should
resolve their output directory under `diagnostic_experiments/`.

`$HOME/csv_files/` is the legacy root. Existing retained model inputs and compact
outputs may still live there; this convention does not migrate them. Older
launchers may still hard-code or default to that root, so verify their effective
paths before running. If a workflow uses `CSV_ROOT` for both inputs and outputs,
preserve the actual input paths and configure outputs separately when needed.
Pair tensors and original PDBs remain at the locations in `datasets.md`.

## Submission directory

Submit from the flat deployment directory so `SLURM_SUBMIT_DIR` resolves
colocated entrypoints and helpers:

```bash
cd /work/hdd/bfna/ikhire/openfold_interpretability_canonical
sbatch <launcher>.sbatch
```

Copy each workflow README under a distinct name, such as
`predicted_distance_export_README.md`, so flat deployment does not overwrite
another workflow's instructions.

## Launcher requirements

- When creating, moving, or renaming a cluster Python entrypoint, verify every
  sbatch reference under `scripts/cluster_scripts/`.
- Request memory and walltime according to the [memory](suggested_sbatch_memory.md)
  and [walltime](suggested_sbatch_walltime.md) guides, with reasonable headroom.
- Save outputs after every completed layer or similarly meaningful unit so
  partial results survive timeouts.
- Log timestamps and elapsed times for every major stage and layer.
- Every new or modified sbatch file must include
  `#SBATCH --mail-type=BEGIN,END,FAIL` and
  `#SBATCH --mail-user=ikhire3@gatech.edu`.
- Resolve Python entrypoints with `SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}` and
  `srun python3 -u "$SCRIPT_DIR/<script>.py"`.

## Fivefold OOF regression deployment

Before running protocol-2.0 arrays, place these frozen metadata files under
`$CSV_ROOT/regression_5fold_v1/metadata/`:

- `regression_outer_folds_v1.csv`
- `per_protein_pair_folds_v1.csv`
- `fold_digests.json`

The master split stays under `regression_protocol_data/` beside the deployed
`regression_protocol.py`. The checked-in workers declare `scratch&work` filesystem
dependencies for `/work/hdd` inputs; verify current Delta requirements before
submission. See the [fivefold release runbook](../../scripts/cluster_scripts/cross_validation/README.md)
for the exact file inventory, task maps, aggregation, and replay commands.

Related runbooks: [distance export](../../scripts/cluster_scripts/predicted_distances/README.md),
[TM-score rescoring](../../scripts/cluster_scripts/tm_scores/README.md), and
[publication subgroup replay](../../scripts/cluster_scripts/publication_figures/README.md).

## Experiment READMEs and submission

Every new cluster experiment needs a short colocated `README.md` containing:

- Files to deploy, including shared helpers and required executable/configuration files.
- Exact input and output roots, including the Slurm log directory.
- Commands to submit the full arrays, their task mapping/concurrency, and any
  preparation, aggregation, or finalization dependencies.

When the user's task calls for submission, submit the full arrays with appropriate
concurrency caps. A separate one-task resource pilot is not required by default.
Use existing measurements or comparator-based estimates from the
[memory](suggested_sbatch_memory.md) and
[walltime](suggested_sbatch_walltime.md) guides; record successful
production measurements for subsequent runs. Independent arrays may be submitted
together. Dependent stages must wait for their prerequisites or use Slurm
`afterok` dependencies. Create required log directories only after authorization.

## Arrays and monitoring

`#SBATCH --array=0-47%4` creates 48 tasks, with at most four running at once.
`SLURM_ARRAY_TASK_ID` selects the task; log names should use `%A` for the array
job ID and `%a` for the task ID. Concurrency caps limit simultaneous memory,
CPU, and filesystem load; full-array submission does not mean unlimited concurrency.

Useful read-only commands, also covered in the live
[Delta running-jobs guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html):

```bash
squeue -u "$USER" -o "%.18i %.12P %.100j %.2t %.10M %R"
scontrol show job JOB_ID
sacct -j JOB_ID --format=JobID%24,JobName%35,State,Elapsed,Timelimit,MaxRSS
tail -n 40 /path/from/StdOut
```

`scontrol show job` reports stdout/stderr paths and pending reasons. Inspect
array tasks and relevant steps when reading `sacct`; process-reported peak RSS
is not interchangeable with allocation accounting.

## Layerwise ablation CSV source

The canonical source used by `copy_layerwise_ablation_csvs.sh` is
`/work/hdd/bfna/thayes/FINAL_RESULTS_CSVS_TRI_END_FIX_PATCHED_FROM_OLD`.
The local copies are under `delta_files/layerwise_ablation_csvs/` when present.
Set `LAYERWISE_ABLATION_SOURCE_DIR` only when intentionally using another source.

## Canonical migration handoff

The new flat runtime is deployed at
`/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`;
`deployment_manifest.json` records Git revision, source mapping, hashes, and
Linux binary provenance. The HOME deployment remains preserved. Migration job
`23142906` has not yet been confirmed complete; use its logs and final
collection/audit manifests before switching readers to the migrated families.
See the [execution handoff](../plans/delta_canonical_results_migration_plan.md#execution-handoff--october-23-2026).

Mutating entrypoints refuse a collection with `collection_manifest.json`
before creating locks or tee logs. Reruns require explicit fresh roots.
The coefficient, gain/split importance, and TM-score-comparison derived bundles
remain repository-only. Their local analysis entrypoints accept `--input-root`
and `--output-root`; no local MDS exports or publication figures were migrated.
After successful cutover, finish the storage-layout/status documentation and
synchronize updated runtime READMEs with a new deployment inventory, preserving
prior deployed files. Do not modify the running job's deployed inventory.

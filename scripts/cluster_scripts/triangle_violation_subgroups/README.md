# Fivefold triangle subgroup replay

Deploy the non-test Python and SBATCH files flat with the fivefold runtime,
`canonical_storage.py`, the shared regression helpers, the frozen master split,
and Linux TMscore binary. The shared deployment is
`/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`; the deployment
builder records all files in `deployment_manifest.json`.

Read producers from `$CSV_ROOT/regression_5fold_v1/` (Delta default
`CSV_ROOT=/work/hdd/bfna/ikhire`). The complete producer directory is
`regression_5fold_v1/triangle_violation_subgroups_5fold_v1/`, with the relative
alias `analysis/triangle_violation_subgroups_5fold_v1`. The published collection
is protected; reruns require a fresh `TRIANGLE_OUTPUT_ROOT`.

```bash
cd /work/hdd/bfna/ikhire/openfold_interpretability_canonical
export TRIANGLE_OUTPUT_ROOT=/work/hdd/bfna/ikhire/diagnostic_experiments/triangle_rerun_v2
prepare_id=$(sbatch --parsable prepare_triangle_violation_reference.sbatch)
replay_id=$(sbatch --parsable --dependency="afterok:$prepare_id" triangle_violation_subgroup_replay.sbatch)
sbatch --dependency="afterok:$replay_id" aggregate_triangle_violation_subgroups.sbatch
```

The replay array covers layers 0–47 with its checked-in concurrency cap. Slurm
logs stay in the submission directory. Output schemas and scientific evaluation
are defined by `docs/methodology/`; the migration preserves all historical
checkpoints, geometry reproduction tables, and distinct protein-equal and
triangle-pooled summaries. Do not aggregate the published historical bundle
again to update its recorded producer hashes.

# Local Biological-Context Distance-Update Probe

This portable Delta bundle implements the experiment in
`scripts/plans/local_pair_biological_context_update_probe_plan.md`. It compares
seven compact biological/geometric feature bundles for transitions 10->11,
15->16, 20->21, 30->31, and 40->41. Every endpoint distance is decoded with
that endpoint layer's saved pooled `general_base256_combined` split-0 probe.

The runner fits continuous-change Ridge, shallow-tree, and histogram-gradient-
boosting regressors on all local pairs. It also fits logistic, shallow-tree, and
histogram-gradient-boosting shrink-versus-expand classifiers after excluding
`abs(delta_d) <= 0.5 A`. Logistic `C` is chosen from
`{0.001, 0.01, 0.1, 1.0}` using a fixed training-protein validation split and
then refitted on all training proteins. Histogram boosting always runs exactly
100 iterations with `early_stopping=False`.

## Delta inputs

- `$HOME/csv_files/general_base256_combined/coefficients.csv`
- `$HOME/csv_files/general_base256_combined/split_assignments.csv`
- CAMEO and ECOD pair blocks and predicted PDBs at the paths in `datasets.md`

The four input roots can be overridden with `CAMEO_PAIR_BLOCKS_ROOT`,
`ECOD_PAIR_BLOCKS_ROOT`, `CAMEO_PDB_ROOT`, and `ECOD_PDB_ROOT`. Output defaults
to `$HOME/csv_files/local_pair_biological_context_probe/` and can be overridden
with `LOCAL_BIO_CONTEXT_OUTPUT_ROOT`.

## Submit

Copy this entire directory to Delta, enter that copied directory, and run:

```bash
ARRAY_JOB_ID=$(sbatch --parsable local_pair_biological_context_probe.sbatch)
sbatch --dependency="afterok:${ARRAY_JOB_ID}" aggregate_local_pair_biological_context_probe.sbatch
```

The `0-34%18` array maps transitions first and feature bundles second. Up to 18
tasks run concurrently, so the 35 tasks normally require two scheduler waves.
The per-task target is 30--60 minutes; it is not a total array elapsed-time
guarantee.

Each shard is written to `shards/transition_XX_YY/<bundle>/`. The aggregator
validates all 35 shards, requires identical validation/bootstrap assignments
where appropriate, creates paired bootstrap feature-ablation intervals, and
copies model/tree artifacts into the aggregate output root. Pair-level matrices
and predictions are not saved.

## Local validation

From the repository root:

```bash
./.venv/bin/python -m unittest discover \
  -s scripts/cluster_scripts/diagnostics/local_pair_biological_context_probe/tests \
  -p 'test_*.py'
```

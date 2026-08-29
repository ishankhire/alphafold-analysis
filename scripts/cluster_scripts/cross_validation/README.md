# Fivefold OOF regression release runbook

This directory contains the production-ready protocol-2.0 implementation for
the four fivefold out-of-fold regression families. Stage 7 release hardening
validated the task mappings, exact shard inventories, aggregation failure
policy, launchers, and replay contract. No files have been deployed and no
Delta jobs have been submitted; protocol-2.0 production result families do not
yet exist.

The scientific protocol and output contract remain authoritative in
`docs/methodology/regression_methodology.md` and
`docs/methodology/regression_outputs.md`. This file only describes deployment,
execution, aggregation, and replay.

## Delta prerequisites

Recheck the live NCSA Delta
[running-jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html),
[data-management](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html),
and [software](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/software.html)
documentation before deployment. As checked on 2026-08-28, the launchers use
the CPU partition, explicitly request both the locked `scratch` dependency and
the `work` dependency required for `/work/hdd` inputs, reset/load modules, and
run through Slurm rather than on a login node.

Use the repository's flat deployment root and output convention:

```text
$HOME/openfold_interpretability_canonical/
$HOME/csv_files/regression_5fold_v1/
```

The account in the checked-in launchers is `bekh-delta-cpu`. Confirm that it is
still returned by `accounts` before submission. Confirm available HOME quota
with `quota`, because the locked project convention writes compact shards and
final outputs under `$HOME/csv_files` even though pair tensors and PDB inputs
are read from `/work/hdd`.

## Flat deployment inventory

Copy the following to `$HOME/openfold_interpretability_canonical/` without
preserving the repository's `cross_validation/` subdirectory:

- every non-test `.py`, `.sbatch`, and `.json` file in this directory;
- `scripts/cluster_scripts/cluster_regression_helpers.py`;
- `scripts/cluster_scripts/regression_output_helpers.py`;
- `scripts/cluster_scripts/regression_protocol.py`; and
- `scripts/cluster_scripts/regression_protocol_data/regression_master_split_v1.csv`
  under the sole retained subdirectory
  `regression_protocol_data/regression_master_split_v1.csv`.

The frozen XGBoost selection table is not a runtime input. Its repository path
and SHA-256 digest are embedded in and validated by the portable fixed
configuration.

Copy these three checked-in metadata files to
`$HOME/csv_files/regression_5fold_v1/metadata/`:

```text
regression_outer_folds_v1.csv
per_protein_pair_folds_v1.csv
fold_digests.json
```

They come from `delta_files/regression_5fold_v1/metadata/`. When `CSV_ROOT` is
set, every worker and aggregator deliberately loads them from the output root,
not from the local repository copy.

## Preflight

From the flat deployment directory:

```bash
cd "$HOME/openfold_interpretability_canonical"
module reset
module load python
source ~/protein_venv/bin/activate
export CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}
python3 -c 'from cross_validation_common import load_cv_folds; b=load_cv_folds(); print(len(b.eligible_protein_ids), [len(b.protein_ids_for_outer_fold(f)) for f in range(5)])'
python3 -c 'import xgboost; print(xgboost.__version__)'
bash -n ./*.sbatch
```

The fold check must print `552 [111, 111, 110, 110, 110]`. A missing XGBoost
dependency must be repaired in `~/protein_venv` before submitting the XGBoost
array; the local `.venv` intentionally tests that workflow with a mock.

## Task maps and locked worker requests

| Family | Launcher | Task mapping | Expected shards | Concurrency | Memory | Walltime |
|---|---|---|---:|---:|---:|---:|
| Shared OLS | `shared_ols_base256_fold_layer.sbatch` | `task_id = outer_fold * 48 + layer` | 240 | 4 | `120g` | `00:30:00` |
| Per-protein OLS | `per_protein_ols_base256_layer.sbatch` | `task_id = layer` | 48 | 12 | `12g` | `02:00:00` |
| Sequence-gated OLS | `sequence_gated_ols_base256_fold_layer.sbatch` | `task_id = outer_fold * 48 + layer` | 240 | 8 | `120g` | `03:00:00` |
| Fixed shared XGBoost | `shared_xgboost_base256_fixed_s4_fold_layer.sbatch` | `task_id = outer_fold * 48 + layer` | 240 | 3 | `12g` | `00:30:00` |

These requests are the plan's comparator-derived estimates, not measurements
of the fivefold workflows. The explicit no-pilot decision is recorded in the
shared memory and walltime guides. Do not silently change any request before
submission.

## Submission and restart

The four families are independent and may be submitted separately:

```bash
sbatch shared_ols_base256_fold_layer.sbatch
sbatch per_protein_ols_base256_layer.sbatch
sbatch sequence_gated_ols_base256_fold_layer.sbatch
sbatch shared_xgboost_base256_fixed_s4_fold_layer.sbatch
```

Record every returned job ID. Verify completion with allocation-level records,
for example:

```bash
sacct -X -j JOB_ID --format=JobID%24,JobName%35,State,Elapsed,ExitCode
```

Do not aggregate a family until every expected array task is `COMPLETED`.
Rerunning the same array is the supported restart path: a task skips only when
its invariants, artifact names, sizes, and SHA-256 hashes match. A missing or
corrupt checkpoint is not accepted as complete.

## Aggregation and completion audit

Aggregation is production computation and must run inside an approved CPU
compute allocation, not on a login node. The locked plan does not assign new
aggregation resource estimates or aggregation sbatch launchers, so obtain an
appropriately sized allocation under current Delta policy and then run the
applicable commands from the flat deployment directory:

```bash
python3 -u aggregate_shared_ols_base256.py
python3 -u aggregate_per_protein_ols_base256.py
python3 -u aggregate_sequence_gated_ols_base256.py
python3 -u aggregate_shared_xgboost_base256_fixed_s4.py
```

Each production aggregator requires all 48 layers and the exact physical shard
inventory listed above. It rejects missing or unexpected directories,
duplicate task identities, incomplete checkpoints, schema/count/coverage
errors, and size or SHA-256 mismatches. Do not place backup shard directories
inside `layer_shards/`.

A family is complete only after its aggregator publishes a
`run_manifest.json` with `run_status=complete`. Audit the manifest's task IDs,
digests, output counts, and replay count before copying results locally:

| Family | Complete `per_protein.csv` rows | Replay-index rows |
|---|---:|---:|
| Shared OLS | 26,496 | 240 |
| Per-protein OLS | 26,496 | 132,480 |
| Sequence-gated OLS | 26,496 | 1,440 |
| Fixed shared XGBoost | 26,496 | 240 |

The absence of a top-level complete manifest means the family is incomplete,
even if some worker shards exist.

## Triangle-subgroup replay interface

The stable downstream interface is each family's
`model_artifacts/replay_index.csv` plus the frozen metadata manifests. Resolve
every `artifact_path` relative to that family root, require `status=complete`,
and verify the recorded artifact SHA-256, configuration digest, and scaler
digest before opening pair tensors. Raw pair predictions are intentionally not
saved.

Responsibility resolution is exact:

- shared OLS and XGBoost use `(layer, outer_fold)`; a protein is replayed with
  the model whose outer fold equals its entry in
  `regression_outer_folds_v1.csv`;
- sequence-gated OLS uses `(layer, outer_fold, sequence_gate)`, with every pair
  routed through the six canonical sequence-separation gates and no fallback;
  and
- per-protein OLS uses `(layer, protein_id, pair_fold)`. Regenerate the five
  pair folds from `per_protein_pair_folds_v1.csv`, then locate the matching
  responsibility inside the referenced compressed coefficient shard rather
  than assuming that replay-index row order is the NPZ array order.

All replay uses strict-lower-triangle unordered pairs and ordered directional
base256 float64 features. For shared families, every protein receives exactly
one held-out-fold model. For per-protein OLS, every pair receives exactly the
one model that withheld its regenerated pair fold. Complete OOF predictions
must be assembled before triangle or other geometry calculations.

The existing triangle-violation subgroup workflow remains unchanged and uses
historical protocol-1.0 artifacts. Moving it to this protocol-2.0 replay
interface, including any change from its current 111-protein evaluation scope,
is a separate project.

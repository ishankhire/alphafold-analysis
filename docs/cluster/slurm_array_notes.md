# Delta Slurm Arrays

Use the live [NCSA Delta user documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/)
for current partitions, accounts, module setup, filesystem constraints, and
storage behavior. Use upstream Slurm/SchedMD documentation for generic Slurm
syntax when the Delta guide does not show an example.

## Repository deployment layout on Delta

The canonical Delta deployment uses a flat script directory and a separate
output directory:

```text
$HOME/
├── openfold_interpretability_canonical/
│   ├── <all cluster Python entrypoints and shared Python helpers>
│   ├── <all cluster sbatch launchers and colocated configuration files>
│   └── regression_protocol_data/
│       └── regression_master_split_v1.csv
└── csv_files/
    └── <analysis result families>
```

The subdirectories under the repository's `scripts/cluster_scripts/` tree are
for source organization only. When deploying to Delta, copy their cluster-run
files directly into `$HOME/openfold_interpretability_canonical/`. The sole
required deployment subdirectory is `regression_protocol_data/`, because
`regression_protocol.py` resolves the frozen split manifest relative to itself.

Submit jobs from the flat script directory so both older relative-path
launchers and launchers based on `SLURM_SUBMIT_DIR` resolve the same files:

```bash
cd "$HOME/openfold_interpretability_canonical"
sbatch <launcher>.sbatch
```

Cluster launchers should set `CSV_ROOT=$HOME/csv_files` (or an explicitly
documented analysis-specific child path). Python entrypoints should use that
environment variable for cluster outputs rather than infer a repository tree
from `__file__`.

## Fivefold OOF regression deployment

The protocol-2.0 regression arrays additionally require the three frozen fold
metadata files under
`$CSV_ROOT/regression_5fold_v1/metadata/` before any task starts. The master
manifest remains beside `regression_protocol.py` under
`regression_protocol_data/`. See the
[fivefold OOF release runbook](../../scripts/cluster_scripts/cross_validation/README.md)
for the exact flat deployment inventory, task maps, submission order,
aggregation checks, and replay contract.

These workers read pair tensors and PDBs from `/work/hdd`. In addition to the
project's locked `scratch` constraint, their launchers declare Delta's current
`work` filesystem dependency as `#SBATCH --constraint="scratch&work"`.

## Layerwise ablation CSV source

The current canonical source for the layerwise ablation CSVs on Delta is
`/work/hdd/bfna/thayes/FINAL_RESULTS_CSVS_TRI_END_FIX_PATCHED_FROM_OLD`. The
`copy_layerwise_ablation_csvs.sh` utility copies the expected files from that
directory by default; set `LAYERWISE_ABLATION_SOURCE_DIR` only when intentionally
using another source.

## Arrays

`#SBATCH --array=0-47%12` creates 48 tasks with task IDs `0` through `47`, with
at most 12 tasks running at the same time. Each task gets its own
`SLURM_ARRAY_TASK_ID`, which can be mapped to a layer, shard, parameter index, or
input file.

Use `%A` and `%a` in log filenames for array job ID and task ID:

```bash
#SBATCH -o slurm-%A_%a.out
#SBATCH -e slurm-%A_%a.err
```

Cancel the whole array with `scancel <array_job_id>`, or one task with
`scancel <array_job_id>_<task_id>`.

Concurrency caps trade wall-clock time against shared-resource pressure. More
concurrent tasks can finish sooner, but can also increase CPU allocation and
stress shared filesystems such as `/work/hdd`.

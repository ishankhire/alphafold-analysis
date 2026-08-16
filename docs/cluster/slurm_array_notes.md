# Delta Slurm Arrays

Use the live [NCSA Delta user documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/)
for current partitions, accounts, module setup, filesystem constraints, and
storage behavior. Use upstream Slurm/SchedMD documentation for generic Slurm
syntax when the Delta guide does not show an example.

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

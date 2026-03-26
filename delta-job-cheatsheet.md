# Delta Job Cheat Sheet

Short reference for the Delta cluster, focused on checking running jobs, seeing full job names, and inspecting output/log files.

Based on:
- `proteins/llm_reference/delta_slurm_guide.pdf`
- `proteins/llm_reference/slurm_guidelines.pdf`

## See My Jobs

Show all of your jobs:

```bash
squeue -u $USER
```

Show only running jobs:

```bash
squeue -u $USER -t R
```

Show only pending jobs and the reason they are waiting:

```bash
squeue -u $USER -t PD -o "%.18i %.100j %.2t %.10M %R"
```

## Show Full Job Names

The default `squeue` view often truncates long job names. This version gives the job-name column much more room:

```bash
squeue -u $USER -o "%.18i %.12P %.100j %.8u %.2t %.10M %.6D %R"
```

If names are still getting cut off, increase `100` to something larger like `140`.

## Detailed Info For One Job

Show detailed information for a specific job:

```bash
scontrol show job JOBID
```

Good for checking:
- stdout/stderr file paths
- assigned node(s)
- requested resources
- current state and pending reason

## Find The Output File For A Running Job

Show where Slurm is writing stdout/stderr for one job:

```bash
scontrol show job JOBID | egrep 'StdOut=|StdErr='
```

## Look At The Current Output Of Every Running Job

Print the stdout/stderr path for every running job:

```bash
for j in $(squeue -h -u $USER -t R -o "%i"); do
  echo "===== Job $j ====="
  scontrol show job "$j" | egrep 'StdOut=|StdErr='
done
```

Print the last 20 lines of each running job's stdout file:

```bash
for j in $(squeue -h -u $USER -t R -o "%i"); do
  f=$(scontrol show job "$j" | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p')
  echo "===== Job $j : $f ====="
  if [ -f "$f" ]; then
    tail -n 20 "$f"
  else
    echo "stdout file not created yet"
  fi
  echo
done
```

## Follow One Job Live

Once you know the stdout path:

```bash
tail -f path/to/slurm-output-file.out
```

## See Which Node A Running Job Is On

```bash
squeue -j JOBID
```

Then SSH to that node if you want to inspect it directly:

```bash
ssh NODE_NAME
```

Useful once connected:

```bash
top -u $USER
```

GPU jobs:

```bash
nvidia-smi
```

## Completed Jobs

For jobs that have already finished, `sacct` is often more useful than `squeue`:

```bash
sacct -j JOBID -X -o JobID,JobName,State,Elapsed,ExitCode
```

Show recent jobs for your account:

```bash
sacct -u $USER -S today -X -o JobID,JobName,State,Elapsed,ExitCode
```

## Cancel A Job

```bash
scancel JOBID
```

## Practical Defaults I’d Reuse A Lot

Wide job table:

```bash
squeue -u $USER -o "%.18i %.12P %.100j %.2t %.10M %.6D %R"
```

Deep dive on one job:

```bash
scontrol show job JOBID
```

Check output for all running jobs:

```bash
for j in $(squeue -h -u $USER -t R -o "%i"); do
  f=$(scontrol show job "$j" | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p')
  echo "===== Job $j : $f ====="
  [ -f "$f" ] && tail -n 20 "$f"
  echo
done
```

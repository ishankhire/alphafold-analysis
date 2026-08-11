# Local Pair-Update Probe Pilot

This portable Delta bundle fits classification and continuous-change Ridge
probes for transitions 10→11, 20→21, and 30→31. It uses raw pre-MDS distance
changes and all residue pairs with inclusive sequence separation 4–12.

## Inputs

The runner expects:

- `$HOME/csv_files/general_base256_combined/coefficients.csv`
- `$HOME/csv_files/general_base256_combined/split_assignments.csv`
- CAMEO pair blocks under `/work/hdd/bekh/thayes/openfold_reps`
- ECOD pair blocks under
  `/work/hdd/bfna/thayes/ecod/experiments/representation_saves`

The pair-block roots can be overridden with `CAMEO_PAIR_BLOCKS_ROOT` and
`ECOD_PAIR_BLOCKS_ROOT`. Outputs default to
`$HOME/csv_files/local_pair_update_probe_pilot/` and can be overridden with
`LOCAL_PAIR_UPDATE_OUTPUT_ROOT`.

## Submit on Delta

Copy this entire directory to
`$HOME/openfold_interpretability_files/local_pair_update_probe_pilot/`, then:

```bash
cd "$HOME/openfold_interpretability_files/local_pair_update_probe_pilot"
ARRAY_JOB_ID=$(sbatch --parsable local_pair_update_probe_pilot.sbatch)
sbatch --dependency="afterok:${ARRAY_JOB_ID}" aggregate_local_pair_update_probe_pilot.sbatch
```

The array has 15 tasks. Tasks 0–4 run the five variants for 10→11, tasks 5–9
for 20→21, and tasks 10–14 for 30→31. Every task writes an isolated shard and
saves classifier results before beginning regression. The aggregator refuses
to run unless all 15 shards report `status: complete`.

## Outputs

Shards are written beneath:

```text
$HOME/csv_files/local_pair_update_probe_pilot/shards/transition_XX_YY/<variant>/
```

After aggregation, the output root contains classification and regression
summaries, per-class and per-protein metrics, confusion matrices, probe
coefficients, change statistics, feature schemas, manifests, timings, metadata,
and a human-readable `summary.txt`. Pair-level predictions are intentionally
not saved.

## Local tests

From the repository root:

```bash
./.venv/bin/python -m unittest discover \
  -s proteins/multi_protein_results/diagnostics/local_pair_update_probe_pilot/tests \
  -p 'test_*.py'
```

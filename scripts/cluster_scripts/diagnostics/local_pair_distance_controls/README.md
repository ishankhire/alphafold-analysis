# Consecutive-Layer Distance Controls

This Delta bundle compares four ways to decode distance changes over transitions
10→11 through 15→16 and separately measures pair-representation distribution
drift. Predictions are raw pre-MDS values from the frozen
`general_base256_combined` train/test split.

## Inputs

- `$HOME/csv_files/general_base256_combined/coefficients.csv`
- `$HOME/csv_files/general_base256_combined/split_assignments.csv`
- The combined CAMEO and ECOD pair-block/PDB roots defined in
  `general_train_test_common.py`

## Run on Delta

Copy this directory beside `cluster_regression_helpers.py`'s parent-compatible
repository layout, then submit from this directory:

```bash
ARRAY_JOB_ID=$(sbatch --parsable local_pair_distance_controls.sbatch)
sbatch --dependency="afterok:${ARRAY_JOB_ID}" aggregate_local_pair_distance_controls.sbatch
```

Each transition writes an isolated shard under
`$HOME/csv_files/local_pair_distance_controls_layers10_16/shards/`. The
aggregator requires all six complete shards, produces root-level CSVs and a
summary, and saves five held-out-test figures under `visualizations/`.

Pair-level rows are not saved. Pair-weighted means and error metrics are exact;
pair-level quantiles use deterministic bounded reservoir samples of at most
200,000 values per group. Protein-equal distribution rows use exact per-protein
means.

## Local validation

From the repository root:

```bash
./.venv/bin/python -m unittest discover \
  -s scripts/cluster_scripts/diagnostics/local_pair_distance_controls/tests \
  -p 'test_*.py'
```

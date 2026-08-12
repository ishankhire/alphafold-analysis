# Expanded Local Pair-Representation Update Probe

This portable Delta bundle extends `local_pair_update_probe_pilot` while keeping
the model linear. It fits protein-equal Ridge regressions for transitions 0→1,
10→11, 20→21, 30→31, 40→41, and 46→47 using the symmetrized center pair
representation and nested triangle radii 1, 2, 3, 5, 7, and 9.

The target is signed raw pre-MDS Δd. Each endpoint is decoded with its own saved
split-0 `general_base256_combined` distance probe. All residue pairs with
inclusive sequence separation 4–12 are used. Classification is intentionally
omitted; `regression_magnitude_metrics.csv` retains sign accuracy and reports
performance in signed and cumulative `|Δd|` buckets.

## Features and fitting

For every potential intermediate residue `k`, the input contains separate
symmetrized `(i,k)` and `(k,j)` 128-channel edges plus a validity flag. Candidate
slots are retained when two endpoint-relative definitions identify the same
valid residue. If `k` is outside the protein or equals endpoint `i` or `j`, both
edges and the validity flag are zero.

| Variant | Width |
|---|---:|
| `center` | 128 |
| `triangle_r1` | 1,156 |
| `triangle_r2` | 2,184 |
| `triangle_r3` | 3,212 |
| `triangle_r5` | 5,268 |
| `triangle_r7` | 7,324 |
| `triangle_r9` | 9,380 |

All feature matrices and targets use `float64`. Scaling and fitting weights give
each training protein equal total influence. Ridge uses relative alpha 0.001 and
a weighted, centered SciPy LSQR `LinearOperator` with tolerance 0.001 and a
100-iteration safety cap. The operator avoids materializing either `X'X` or the
extra `sqrt(W)X` copy that sklearn creates for weighted dense Ridge.

## Submit on Delta

Copy this directory to Delta, enter it, and submit both arrays followed by the
aggregator:

```bash
SMALL_JOB_ID=$(sbatch --parsable local_pair_representation_update_small.sbatch)
LARGE_JOB_ID=$(sbatch --parsable local_pair_representation_update_large.sbatch)
sbatch --dependency="afterok:${SMALL_JOB_ID}:${LARGE_JOB_ID}" \
  aggregate_local_pair_representation_update_probe.sbatch
```

The small array has 30 tasks for center through radius 5 and requests 96 GB.
The large array has 12 tasks for radii 7 and 9 and requests 128 GB. A 20-minute
runtime is an optimization target per task, not a guarantee. The submitted wall
limits are deliberately conservative (45 and 90 minutes) because radius 9 alone
holds about 71 GiB of train/test feature data and repeatedly scans it during
LSQR. Every shard records stage time, maximum RSS, LSQR stopping code, iteration
count, and whether it reached the cap.

Before launching the full large array, a radius-9 benchmark for 0→1 can be run
by overriding its array selection:

```bash
sbatch --array=1 local_pair_representation_update_large.sbatch
```

Inputs default to `$HOME/csv_files/general_base256_combined/` plus the CAMEO and
ECOD pair-block/PDB paths documented in `datasets.md`. The four data roots and
`LOCAL_PAIR_REP_UPDATE_OUTPUT_ROOT` can be overridden with environment variables.

## Outputs

Shards are written beneath
`$HOME/csv_files/local_pair_representation_update_probe/shards/transition_XX_YY/<variant>/`.
The aggregator requires all 42 complete shards and writes:

- `model_summary.csv`: one train/test row per transition and variant, with
  pair-pooled and mean-per-protein R², MAE, RMSE, Pearson correlation, bias,
  sign accuracy, support, and solver diagnostics.
- `per_protein_metrics.csv`: one protein/transition/variant/split row with the
  regression metrics.
- `regression_magnitude_metrics.csv`: signed Δd bins, exact-zero audit, and
  cumulative `|Δd| > 0.5, 1, 2, 4 Å` metrics.
- `aa_pair_target_statistics.csv`: one transition/split/category row for all 210
  unordered standard-amino-acid pairs, including support, target moments,
  quantiles, direction fractions, and mean Δd relative to all pairs.
- `aa_pair_regression_metrics.csv`: pair-weighted regression metrics for every
  transition, variant, split, and amino-acid category.
- `aa_pair_target_uncertainty.csv`: held-out protein-bootstrap intervals for
  category mean Δd and its contrast with the all-pair mean.
- `aa_pair_bootstrap_summary.csv`: held-out intervals for category R² and
  category-minus-overall R², valid replicate counts, rank intervals, and
  top-quartile rank probability.
- `bootstrap_assignments.csv` and `bootstrap_metrics.csv`: the 200 deterministic
  held-out protein draws and replicate-level headline R² values.
- `variant_gain_summary.csv`: paired bootstrap gains between consecutive feature
  variants per transition and equally across transitions.
- `coefficients.csv`, `feature_schema.csv`, `feature_transforms.csv`,
  `change_statistics.csv`, `coefficients_used.csv`, `split_assignments.csv`,
  `protein_manifest.csv`, `skipped_proteins.csv`, `timing.csv`,
  `run_metadata.json`, and `summary.txt` for interpretation and auditing.

Pair-level predictions are not saved. Amino-acid category differences should be
interpreted using category support, target variance, bootstrap intervals, and
rank stability together; the tables do not make binary significance claims.

## Local validation

From the repository root:

```bash
./.venv/bin/python -m unittest discover \
  -s scripts/cluster_scripts/diagnostics/local_pair_representation_update_probe/tests \
  -p 'test_*.py'
```

# All-layer shared-OLS probe transfer

Diagnostic replay of all 240 frozen protocol-2.0 `shared_ols_base256` probes
across all 48 Evoformer layers. It fits nothing; the canonical families and the
ten-layer [`cross_layer_transfer`](../cross_layer_transfer/README.md) run are
unchanged. Follow the [cluster guide](../../../docs/cluster/README.md) for
authorization and deletion rules.

## Design

- Cells: the full 48 × 48 source and target block, 2,304 cells, in
  `target_restandardized` mode only. `raw` is not run.
- Cohort: `ALL_LAYER_COHORT` must be stated. `cameo` is the 154 CAMEO proteins of
  the frozen cohort, and `full` is all 552. The ECOD pair tensors and final
  structure PDBs left `toDelete` on 2026-11-06, so current results are
  `cameo`. They use the same frozen probes and stay out-of-fold, but they are a
  different cohort: re-run as `full` once ECOD returns, and do not compare pooled
  numbers across cohorts.
- Each held-out protein in outer fold `f` is evaluated with the fold-`f` probe
  from the source layer, applied to its target-layer `[Z_ij | Z_ji]` features.
- `target_restandardized`: the source standardized coefficients and intercept
  with the target layer's same-fold, train-only feature mean and scale, taken
  from the frozen target-layer artifact.
- The regression target is the Cα distance in OpenFold's final predicted
  structure. Signed error is probe distance minus final structure distance, so
  it measures how far a probe reads from that structure.
- Error is captured per cell, with no pair-level storage, along three axes:
  sequence separation by final structure distance, protein, and position along
  the chain. Geometry is not computed.
- `7stz_C` stays in every primary metric. Because it dominates pooled squared
  error, `_excl_7stz_C` columns report each sensitivity separately.

Signed error uses 150 bins: 0.5 Å steps to ±20 Å, 2 Å to ±60 Å, 10 Å to ±200 Å,
and one open bin in each tail. Sequence separation uses 13 fine bins and final
structure distance 17, and both nest exactly in the canonical subgroup bins.

Completion gates, all required for `analysis_status=complete`. The tolerance is
`1e-8·max(1,|x|)`:

- each worker checks its native cell against the producer's `per_protein.csv`
  before writing a shard;
- all 48 native cells reproduce the producer's per-protein metrics for the
  cohort;
- the 100 cells shared with the ten-layer run reproduce its per-protein metrics
  and every per-protein statistic in all 12 categories, from its `layer_shards/`;
- `full` also requires parity with the producer and ten-layer `pooled`, `fold`,
  and subgroup tables. Those pool all 552 proteins, so `cameo` cannot use them;
- every histogram sums to its pair count, joint histograms agree with protein
  histograms and strata counts, exact error sums agree across categories,
  strata, and chain positions, and each exact stratum sum lies within the bounds
  its bin counts imply;
- all shards share the aggregator's code revision.

A worker refuses a shard or checkpoint written by another revision. The
aggregator marks the run `in_progress` until the gates finish, and figures refuse
any status but `complete`.

## Deploy

Launchers run from the submit directory and expect a flat layout. Copy these
into `$HOME/openfold_interpretability_canonical/`:

- from this directory: every `.py` and `.sbatch` file, plus this README as
  `cross_layer_transfer_all_layers_README.md`;
- from `../cross_layer_transfer/`: `cross_layer_transfer_common.py`,
  `cross_layer_transfer_replay.py`, and `aggregate_cross_layer_transfer.py`;
- from `../cross_validation/`: `cross_validation_common.py`,
  `shared_ols_base256_common.py`, `per_protein_ols_base256_common.py`, and
  `sequence_gated_ols_base256_common.py`;
- from `../triangle_violation_subgroups/`:
  `triangle_violation_subgroup_common.py` and
  `triangle_violation_subgroup_replay.py`;
- from `..`: `canonical_storage.py`, `cluster_regression_helpers.py`,
  `regression_output_helpers.py`, `regression_protocol.py`, and
  `regression_protocol_data/regression_master_split_v1.csv`.

Add a `deployment_manifest.json` holding `git_commit`, so `code_revision` is
recorded in the run manifest.

## Inputs and outputs

- Producer: `$CSV_ROOT/regression_5fold_v1/shared_ols_base256/` and
  `$CSV_ROOT/regression_5fold_v1/metadata/`. `CSV_ROOT` defaults to
  `/work/hdd/bfna/ikhire`.
- Ten-layer run for the overlap gate: `$CROSS_LAYER_V1_ROOT`, default
  `/work/hdd/bfna/ikhire/diagnostic_experiments/cross_layer_probe_transfer_v1`.
- Pair tensors and PDBs: the CAMEO paths in `datasets.md`. `full` also needs
  `$CROSS_LAYER_ECOD_EXPERIMENTS_ROOT`, set to wherever the ECOD
  `representation_saves/` and `openfold_reps_outputs_sm/` now live. It has no
  default, and the path in `datasets.md` is stale.
- Output root: `$ALL_LAYER_TRANSFER_OUTPUT_ROOT`, default
  `~/diagnostic_experiments/cross_layer_probe_transfer_all_layers_<cohort>_v1/`
  for `cameo`, and `~/diagnostic_experiments/cross_layer_probe_transfer_all_layers_v1/`
  for `full`.
- Slurm logs: `slurm-<array>_<task>.{out,err}` and `slurm-<job>.{out,err}`, in
  the deployment directory.

The worker writes, per target layer `TT`, under `layer_shards/`:

- `target_layer_TT.npz`: one compressed shard. Per-protein sufficient statistics
  and signed-error histograms; fold-level joint histograms and exact error sums
  by sequence separation and final structure distance; fold-level chain-position
  sums; and the same arrays for `7stz_C` alone.
- `artifact_provenance_target_layer_TT.json`
- `runtime_target_layer_TT.csv`

The aggregator writes one row per source and target cell unless noted.
`per_protein.csv`, `error_strata.csv`, and `parity_checks.csv` change with the cohort:

| File | Rows | One row per | Principal columns |
|---|---:|---|---|
| `per_protein.csv` | 354,816 (`cameo`), 1,271,808 (`full`) | cell, protein | `r2`, `mae`, `rmse`, `bias_pred_minus_true` |
| `pooled_metrics.csv` | 2,304 | cell | pooled metrics, global baseline |
| `fold_metrics.csv` | 11,520 | cell, fold | fold metrics |
| `fold_summary.csv` | 27,648 | cell, metric | fold mean and standard deviation |
| `subgroup_metrics/by_sequence_separation.csv` | 13,824 | cell, canonical bin | pooled metrics, `support_status` |
| `subgroup_metrics/by_spatial_distance.csv` | 11,520 | cell, canonical bin | pooled metrics, `support_status` |
| `transfer_matrix.csv` | 2,304 | cell | `pooled_r2`, native deltas, `layer_offset`, `top_protein_id`, `pooled_r2_excl_7stz_C` |
| `protein_anomalies.csv` | 11,520 | cell, top-5 protein | `squared_error_share`, `mae_robust_z` |
| `error_quantiles.csv` | 32,256 | cell, category, weighting | `signed_error_q01`–`q99`, `signed_error_iqr`, `share_within_5A` |
| `error_strata.csv` | 359,424 (`cameo`), up to 509,184 (`full`) | cell, occupied fine stratum | `mean_signed_error`, `sd_signed_error`, `share_of_squared_error`, `support_status` |
| `error_position.csv` | 46,080 | cell, chain bin | `mae`, `mean_signed_error`, `rmse` |
| `error_histograms_by_cell.npz` | array | cell, error bin | pair-weighted, protein-equal, and excluding-`7stz_C` counts |
| `source_probe_status.csv` | 240 | source layer, fold | fit status, artifact checksum |
| `parity_checks.csv` | varies | check, layer, metric | `status`, `max_abs_diff` |
| `runtime.csv` | 48 | target layer | `elapsed_seconds`, `peak_rss_mb` |
| `run_manifest.json` | one object | run | `analysis_status`, digests, row counts |

`error_quantiles.csv` has 14 rows per cell: the 12 canonical categories
pair-weighted, then whole-cell rows weighted protein-equal and excluding
`7stz_C`. In `per_protein.csv` and the other canonical tables, `layer` equals
`target_layer`, and `pooled_bias_pred_minus_true` keeps its canonical name.

The figure stage writes PNG and PDF files to `figures/`, with one ridgeline page
per source layer in `figures/per_probe/`. Canonical column names keep the repository's
`true`, but figures and new columns say final structure distance.

## Submit

Announce each job first. Launchers use `~/protein_venv` (see the
[cluster guide](../../../docs/cluster/README.md#python-environment)), need
`ALL_LAYER_COHORT` set, and leave `--mail-user` to the submit command, so export
`CLUSTER_NOTICE_EMAIL` in your shell profile.

```bash
cd "$HOME/openfold_interpretability_canonical"
MAIL=--mail-user="${CLUSTER_NOTICE_EMAIL:?Set CLUSTER_NOTICE_EMAIL}"
export ALL_LAYER_COHORT=cameo
REPLAY=$(sbatch --parsable "$MAIL" all_layer_transfer_replay.sbatch)   # array 0-47%16, task i is target layer i
AGG=$(sbatch --parsable "$MAIL" --dependency=afterok:$REPLAY aggregate_all_layer_transfer.sbatch)
sbatch --parsable "$MAIL" --dependency=afterok:$AGG make_all_layer_transfer_figures.sbatch
```

Workers checkpoint every 100 proteins (`ALL_LAYER_CHECKPOINT_PROTEINS`), and on
any failed protein, and resume when resubmitted. A task exits immediately if its
valid final shard already exists, so a one-task pilot of `--array=47` is reused,
provided the deployed code revision is unchanged.

### Re-running on the full cohort

Once the ECOD inputs are back, nothing below needs a code change. Confirm the
`representation_saves/` and `openfold_reps_outputs_sm/` folder names, then repeat
the commands above with:

```bash
export ALL_LAYER_COHORT=full
export CROSS_LAYER_ECOD_EXPERIMENTS_ROOT=/path/holding/both/folders
```

The launcher's 4 CPUs and 1-hour limit should still suit `full`: each target
layer reads about 17 GB of tensors, against 7.6 GB for `cameo`, which took under
4 minutes per layer. The `full` run writes
to its own output root, leaving the `cameo` results in place, and its
`complete` status also requires the pooled, fold, and subgroup gates.

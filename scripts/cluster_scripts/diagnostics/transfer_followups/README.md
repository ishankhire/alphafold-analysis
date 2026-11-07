# Transfer follow-up diagnostics

Experiments 3 to 8 of the
[all-layer transfer and triangle violations report](../../../../docs/historical/analysis/all_layer_transfer_and_triangle_report_2026-11-06/REPORT.md).
They read the CAMEO all-layer transfer run, the native error strata, and stored
geometry and predictions. No probe is refitted. Follow the
[cluster guide](../../../../docs/cluster/README.md) for authorization and deletion
rules.

## Design

| Script | Experiment | What it does |
|---|---|---|
| `analyze_transfer_breakdown.py` | 3 | Reads the transfer tables and probe weights: segmentation, retained R², weight rotation, calibration, paired proteins |
| `analyze_ablation.py` | 4 | Converts ablation curves to TM lost per block, and compares them with probe series and structure module TM |
| `gain_position_replay.py` | 5, 6 | Replays all source probes on one target layer in `target_restandardized` mode and stores prediction moments and chain position sums |
| `analyze_gain_position.py` | 5, 6 | Splits transfer loss into gain and direction, and standardizes chain ends to the mid-chain distance mix |
| `triangle_strata_replay.py` | 7 | Scores observed predictions and six error models on the same triangles, for OLS and XGBoost |
| `analyze_triangle_strata.py` | 7 | Compares each model with the observed violation curve and with earlier block series |
| `analyze_violation_events.py` | 8 | Finds blocks where proteins move together, tests them against earlier series, and regresses violations on the margin |
| `export_report_data.py` | Report | Collects the rows behind the report's figures |

- Target: the Cα distance in OpenFold's final predicted structure. Signed error is
  probe minus final structure.
- Proteins: the 154 CAMEO proteins, with `7stz_C` excluded where a table says so.
  ECOD pair tensors and final structures are unavailable, and the two replays select
  CAMEO directly, so a full-cohort run needs that selection changed.
- Gates: `gain_position_replay.py` reproduces the all-layer transfer shard, and
  `triangle_strata_replay.py` reproduces the stored strata and violation fractions,
  before either writes. Proteins above 1,000,000 triples use a fixed sample.
- FWD at L is read as ablating blocks 0 to L, and REV at L as ablating blocks L to 47.
  `structure_module_tm_scores.csv` is read as the structure module run on the saved
  layer L representation. Both readings are inferred; no document states them.

## Deploy

Copy these flat into `$HOME/openfold_interpretability_canonical/`:

- from this directory: every `.py` and `.sbatch` file, plus this README as
  `transfer_followups_README.md`;
- from `../cross_layer_transfer_all_layers/`: every `.py` file;
- from `../native_error_strata/`: `strata_bins.py`;
- from `../../cross_layer_transfer/`: `cross_layer_transfer_common.py`,
  `cross_layer_transfer_replay.py`, and `aggregate_cross_layer_transfer.py`;
- from `../../cross_validation/`: `cross_validation_common.py`,
  `shared_ols_base256_common.py`, `per_protein_ols_base256_common.py`, and
  `sequence_gated_ols_base256_common.py`;
- from `../../triangle_violation_subgroups/`:
  `triangle_violation_subgroup_common.py` and
  `triangle_violation_subgroup_replay.py`;
- from `../..`: `canonical_storage.py`, `cluster_regression_helpers.py`,
  `regression_output_helpers.py`, `regression_protocol.py`, and
  `regression_protocol_data/regression_master_split_v1.csv`.

## Inputs and outputs

Read from `$CSV_ROOT/regression_5fold_v1/`: `metadata/`, the `geometry_metrics.csv`,
`geometry_summary.csv`, and `per_protein.csv` of each family, the `shared_ols_base256`
coefficient artifacts, and `predicted_distances_5fold_v1/`.

| Environment variable | Default | Holds |
|---|---|---|
| `ALL_LAYER_TRANSFER_OUTPUT_ROOT` | `~/diagnostic_experiments/cross_layer_probe_transfer_all_layers_cameo_v1` | All-layer transfer tables and shards |
| `NATIVE_OLS_STRATA_ROOT` | `~/diagnostic_experiments/native_error_strata_v1` | `data/protein_strata_cameo.npz` |
| `NATIVE_XGBOOST_STRATA_ROOT` | `~/diagnostic_experiments/native_error_strata_xgboost_v1` | `data/protein_strata_cameo.npz` |
| `LAYERWISE_ABLATION_CSV_DIR` | The launcher's source in `toDelete`; see below | Ablation CSVs and `structure_module_tm_scores.csv` |
| `GAIN_POSITION_OUTPUT_ROOT` | `~/diagnostic_experiments/gain_and_chain_position_cameo_v1` | Written: gain replay shards and the gain table |
| `TRIANGLE_STRATA_OUTPUT_ROOT` | `~/diagnostic_experiments/triangle_strata_cameo_v1` | Written: triangle shards, tables, and chart rows |
| `TRANSFER_FOLLOWUPS_REPORT_DIR` | `~/diagnostic_experiments/transfer_followups_cameo_v1` | Written: text reports |

The ablation source sits in a to-delete tree. Copy it with
[`copy_layerwise_ablation_csvs.sh`](../../copy_layerwise_ablation_csvs.sh) and point
`LAYERWISE_ABLATION_CSV_DIR` at the copy.

| Output | Rows | Row | Principal columns |
|---|---:|---|---|
| `layer_shards/target_layer_TT.npz` (gain root) | 48 files | target layer | `moments`, `targets`, `pos_dist`, `pos_seq`, `out_*` |
| `layer_shards/layer_TT.npz` (triangle root) | 48 files | layer | `counts`, `slack_viol`, `position_viol`, `edges` |
| `gain_direction_by_cell_excl_7stz_C.csv` | 2,304 | source, target | `r2`, `r2_recal`, `loss`, `gain_part`, `direction_part`, `amp` |
| `triangle_models_by_layer.csv` | 672 | probe, layer, model | `violation_fraction` |
| `triangle_margin_by_layer.csv` | 96 | probe, layer | edge correlations, `margin_mean`, `margin_sd`, shuffled margins |
| `report_chart_data.json` | 1 | report | rows for the report's figures |
| `*_report.txt` | 5 files | analysis | the printed report of each analysis |

Slurm logs: `slurm-<job>.{out,err}`, in the deployment directory. `.npz` shards stay
out of the repository; the tables and text reports are in
[`delta_files/diagnostic_experiments/transfer_followups_cameo_v1/`](../../../../delta_files/diagnostic_experiments/transfer_followups_cameo_v1/).

## Run on Delta

Launchers run from the submit directory and leave `--mail-user` to the submit command.
A resubmission skips any layer whose shard already exists.

```bash
cd "$HOME/openfold_interpretability_canonical"
MAIL=--mail-user="${CLUSTER_NOTICE_EMAIL:?Set CLUSTER_NOTICE_EMAIL}"
gain=$(sbatch --parsable "$MAIL" gain_position_replay.sbatch)
triangle=$(sbatch --parsable "$MAIL" triangle_strata_replay.sbatch)
sbatch "$MAIL" --dependency="afterok:$gain:$triangle" transfer_followups_report.sbatch
```

`transfer_followups_report.sbatch` runs the gain analysis before the triangle analyses,
because they read the gain table it writes.

## Tests

`../tests/test_transfer_followups.py` checks the triangle sampling, pair indexing,
violation scoring, within-stratum shuffling, grouped sums, and launcher rules. The
analysis scripts have no unit tests. Job `23624510` ran the moved code from a flat
deployment: all five reports matched their first outputs line for line, and layer 47
of both replays matched the stored shards array for array.

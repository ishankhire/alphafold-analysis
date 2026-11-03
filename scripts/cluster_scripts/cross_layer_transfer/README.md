# Cross-layer shared-OLS probe transfer

Diagnostic replay of the frozen protocol-2.0 `shared_ols_base256` probes across
Evoformer layers. It fits nothing; the canonical families are unchanged. Follow
the [cluster guide](../../../docs/cluster/README.md) for authorization and
deletion rules.

## Design

- Layers: mid plateau `22–26` and final `43–47`, as a full 10×10 source→target
  block. This gives 25 mid→end, 25 end→mid, 40 within-group, and 10 diagonal
  cells.
- Each held-out protein in outer fold `f` is evaluated with the fold-`f` probe
  from the source layer, applied to its target-layer `[Z_ij | Z_ji]` features.
- `raw` mode: the frozen source probe exactly as fitted.
- `target_restandardized` mode: the source standardized coefficients and
  intercept, with the target layer's same-fold, train-only feature mean and
  scale taken from the frozen target-layer artifact.
- Statistics computed: per-protein, pooled, fold, fold-summary, and subgroup
  regression metrics, plus per-protein subgroup MAE and the global baseline.
  Geometry is not computed.
- Gate: the diagonal `raw` cells must reproduce the producer's `per_protein`,
  `pooled`, `fold`, and subgroup metrics within `1e-8·max(1,|x|)`.
  `run_manifest.json` reports `analysis_status=complete` only when every row of
  `parity_checks.csv` passes.

## Deploy

Copy these files flat into
`/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`:

- `cross_layer_transfer_common.py`
- `cross_layer_transfer_replay.py` and `cross_layer_transfer_replay.sbatch`
- `aggregate_cross_layer_transfer.py` and `aggregate_cross_layer_transfer.sbatch`
- this README, as `cross_layer_transfer_README.md`

The workflow also reuses these already-deployed helpers:

- `canonical_storage.py`
- `cluster_regression_helpers.py`
- `regression_output_helpers.py`
- `regression_protocol.py` (with `regression_protocol_data/`)
- `cross_validation_common.py`
- `shared_ols_base256_common.py`
- `per_protein_ols_base256_common.py`
- `sequence_gated_ols_base256_common.py`
- `triangle_violation_subgroup_common.py`
- `triangle_violation_subgroup_replay.py`

## Inputs and outputs

- Producer:
  `$CSV_ROOT/regression_5fold_v1/shared_ols_base256/` (`run_manifest.json`,
  `model_artifacts/`, compact tables) and `$CSV_ROOT/regression_5fold_v1/metadata/`.
  `CSV_ROOT` defaults to `/work/hdd/bfna/ikhire`.
- Pair tensors and PDBs: the paths given in `datasets.md`.
- Output root: `$CROSS_LAYER_TRANSFER_OUTPUT_ROOT`, default
  `/work/hdd/bfna/ikhire/diagnostic_experiments/cross_layer_probe_transfer_v1/`.
- Slurm logs: `slurm-<array>_<task>.{out,err}` and `slurm-<job>.{out,err}`, in
  the deployment directory.

The worker writes, per target layer `TT`:

- `layer_shards/cross_layer_primitives_target_layer_TT.csv`: 133,032 rows of
  additive sufficient statistics.
- `artifact_provenance_target_layer_TT.json`
- `runtime_target_layer_TT.csv`

The aggregator writes:

| File | Rows |
|---|---|
| `per_protein.csv` | 110,400 |
| `per_protein_subgroup_mae.csv` | 1,214,400 |
| `pooled_metrics.csv` | 200 |
| `fold_metrics.csv` | 1,000 |
| `fold_summary.csv` | 2,400 |
| `subgroup_metrics/by_sequence_separation.csv` | 1,200 |
| `subgroup_metrics/by_spatial_distance.csv` | 1,000 |
| `transfer_matrix.csv` | 200 |
| `source_probe_status.csv` | 50 |
| `parity_checks.csv` | — |
| `runtime.csv` | 10 |
| `run_manifest.json` | — |

Every metric row adds these columns: `source_layer`, `target_layer`,
`transfer_mode`, `source_group`, `target_group`, `transfer_block`, and
`is_same_layer`. In these rows, `layer` equals `target_layer`.

## Submit

```bash
cd /work/hdd/bfna/ikhire/openfold_interpretability_canonical
export CSV_ROOT=/work/hdd/bfna/ikhire
export CROSS_LAYER_TRANSFER_OUTPUT_ROOT=/work/hdd/bfna/ikhire/diagnostic_experiments/cross_layer_probe_transfer_v1
REPLAY=$(sbatch --parsable cross_layer_transfer_replay.sbatch)   # array 0-9%10, task i -> layer i of (22..26, 43..47)
sbatch --parsable --dependency=afterok:$REPLAY aggregate_cross_layer_transfer.sbatch
```

Workers checkpoint every 25 proteins (`CROSS_LAYER_CHECKPOINT_PROTEINS`) and
resume when resubmitted. A task exits immediately if its valid final shard
already exists.

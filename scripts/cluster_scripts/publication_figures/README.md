# Publication subgroup-MAE replay

This independent protocol-2.0 replay supplies per-protein sequence-separation
and true-spatial-distance MAE for the September 1 publication figures. It does
not refit a probe and does not modify the four canonical regression families.

Deploy these files flat to `$HOME/openfold_interpretability_canonical/` along
with the non-test files required by the fivefold release and triangle replay:

- all `.py` and `.sbatch` files in this directory;
- `cluster_regression_helpers.py`, `regression_output_helpers.py`, and
  `regression_protocol.py`;
- the non-test Python helpers in `cross_validation/` used by the four replay
  families; and
- `triangle_violation_subgroup_common.py` and
  `triangle_violation_subgroup_replay.py`.

The three frozen fold metadata files must already be under
`$CSV_ROOT/regression_5fold_v1/metadata/`, and all four producer manifests and
replay artifacts must be complete.

Submit from the flat deployment directory:

```bash
sbatch replay_per_protein_subgroup_mae.sbatch
# after all 48 tasks are COMPLETED
sbatch aggregate_per_protein_subgroup_mae.sbatch
```

The aggregator writes beneath
`$CSV_ROOT/analysis/publication_subgroup_mae_5fold_v1/`. It requires all 48
layer shards, validates exactly 1,165,824 per-protein subgroup rows, and
requires additive replay primitives to reproduce the canonical pair-pooled
MAE/count rows before publishing `run_manifest.json` with
`analysis_status=complete`.

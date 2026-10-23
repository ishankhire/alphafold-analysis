# Publication subgroup-MAE replay

The published collection is protected by `collection_manifest.json`. Reading
retained artifacts uses relative indexes and historical checksums. New runs
require explicit fresh output roots; existing checkpoint identities remain
unchanged. `canonical_storage.py` defaults repository runs to `delta_files`
and flat Delta deployments to `/work/hdd/bfna/ikhire`; `CSV_ROOT` is the
parent of `regression_5fold_v1`. Include `canonical_storage.py` with the
shared helpers. `canonical_deployment.py` builds the runtime allowlist and
`deployment_manifest.json` (Git revision, mapping, hashes, binary provenance).
Slurm stdout/stderr stay in the deployment directory, so rejected launches
cannot write inside protected results.

This independent protocol-2.0 replay supplies per-protein sequence-separation
and true-spatial-distance MAE for the publication figures. It does
not refit a probe and does not modify the four canonical regression families.

Follow the [cluster guide](../../../docs/cluster/README.md); cluster changes follow its task authorization and deletion rules. The current publication inputs are identified in
[current sources and status](../../../docs/project_status.md).

Deploy these files flat to `/work/hdd/bfna/ikhire/openfold_interpretability_canonical/` along
with the non-test files required by the fivefold release and triangle replay:

- all `.py` and `.sbatch` files in this directory;
- this README as `publication_subgroup_mae_README.md`;
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
cd "/work/hdd/bfna/ikhire/openfold_interpretability_canonical"
export PUBLICATION_SUBGROUP_OUTPUT_ROOT=/work/hdd/bfna/ikhire/diagnostic_experiments/publication_subgroup_rerun_v2
sbatch replay_per_protein_subgroup_mae.sbatch
# after all 48 tasks are COMPLETED
sbatch aggregate_per_protein_subgroup_mae.sbatch
```

The aggregator writes beneath
`$CSV_ROOT/regression_5fold_v1/analysis/publication_subgroup_mae_5fold_v1/`. The current local copy
is `delta_files/regression_5fold_v1/analysis/publication_subgroup_mae_5fold_v1/`.
It requires all 48 layer shards, validates exactly 1,165,824 per-protein subgroup rows, and
requires additive replay primitives to reproduce the canonical pair-pooled
MAE/count rows before publishing `run_manifest.json` with
`analysis_status=complete`.

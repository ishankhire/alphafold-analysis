# Suggested SBATCH memory

These are per-job or per-array-task resource references and existing requests.
Observed MaxRSS values come from Delta accounting, including
the 2026 export in `delta_files/delta_usage_export_2026.tar.gz` and subsequent
`sacct` measurements; they are converted to GiB. Because Delta can derive SU
charges from requested memory, use the smallest request that retains reasonable
headroom and recheck MaxRSS after the first canonical run.

## Sizing rules

- Normally request only 1.5–2x the observed peak MaxRSS for the workflow.
  Requests above 2x require a concrete, documented reason.
- When the exact workflow is unmeasured, estimate from the closest completed
  jobs, accounting for cohort size, feature width and dtype, dense matrix
  materialization, and per-task concurrency. Leave reasonable headroom.
- Submit the authorized full array without a separate resource pilot by default;
  use successful task measurements to refine requests for subsequent runs.
- Preserve the regression pipeline while right-sizing; resource changes must
  not alter experimental behavior.

## Fivefold OOF resource references

### Saved-distance TM-score rescoring estimates

The [TM-score runbook](../../scripts/cluster_scripts/tm_scores/README.md) uses
completed packed arrays and does not load pair tensors or model artifacts.
Submit the authorized full array using these estimates; record successful
allocation measurements for subsequent runs. In score-only array `22908533`,
37 successful tasks reached a maximum Python-step MaxRSS of 1,022,276 KiB
(0.975 GiB, layer 11); the original 1 GiB allocation had little headroom.
The PDB-enabled worker adds residue formatting and a per-layer PDB index.
Finalization also retains the 105,984-row score/PDB indexes and replacement/
summary rows. PDB-enabled worker allocation measurements remain unrecorded;
successful consolidation measurements are below.

| Stage | Initial `--mem` | Basis |
|---|---:|---|
| One-layer four-probe rescoring and PDB export | `2g` | About 2x the observed 0.975 GiB score-only peak, with additional PDB/index working memory explaining the slight excess over 2x; one protein at a time and one BLAS thread. |
| Consolidation only | `3g` | Successful job `22930940` reached Python-step MaxRSS 1,634,616 KiB (1.56 GiB), giving 1.92x headroom for the full score/PDB indexes, source tables, and summaries. |
| Optional full-file audit | `3g` | Same full-ledger memory requirements as consolidation; the former `2g` allocation was measured only through half the layers. |

Record successful PDB-enabled step MaxRSS for both stages and replace the estimates.

The one-off dependency-gated TM-score deployment/validation job uses `1g`,
matching the scoring worker's metadata-loading estimate. It validates the
completed export, deployment checksums, source geometry coverage, and native
backend before the scoring array starts; it does not reconstruct a layer.

### Packed predicted-distance export estimates

The [export runbook](../../scripts/cluster_scripts/predicted_distances/README.md)
uses full-array submission. Successful full-stage allocation measurements
remain unrecorded here; finalization's partial-run lower bound is noted below.

| Stage | Initial `--mem` | Basis |
|---|---:|---|
| Export preparation | `3g` | Approximately 1.68x a 1.79 GiB partial-run observation; successful allocation MaxRSS is not recorded here. |
| One-layer four-family prediction export | `12g` | Closest completed replay peaked at 6.15 GiB (1.95x headroom). Keeps one protein feature working set and compact model state; maximum 794 residues implies about 308 MiB float32 tensor and 615 MiB float64 base256 features before temporaries. |
| Export finalization | `3g` | Job `22907683` timed out after four validated layers with a Python-step MaxRSS of 2,078,640 KiB (1.98 GiB). This is a partial-run lower bound, not a successful stage measurement; `3g` provides 1.51x that observed peak while retaining the streaming pipeline. |

Record allocation MaxRSS after the first successful task of each stage and
adjust requests before repeat runs.

### Regression worker observations

Saved family `runtime.csv` files under `delta_files/regression_5fold_v1/`
record the following successful process peaks. These are process RSS, not
Slurm allocation MaxRSS. Requests below are the unchanged launcher values;
use allocation accounting before revising them.

| Workflow | Tasks | Maximum process RSS (GiB) | Existing `--mem` |
|---|---:|---:|---:|
| Shared OLS, outer-fold/layer | 240 | 92.69 | `120g` |
| Per-protein OLS, five pair folds/layer | 48 | 3.68 | `12g` |
| Sequence-gated OLS, outer-fold/layer | 240 | 78.50 | `120g` |
| Fixed shared XGBoost, outer-fold/layer | 240 | 6.15 | `12g` |

### Fivefold aggregation estimates

The four production aggregators are serial validation and consolidation jobs.
They do not reopen pair tensors or materialize the worker fitting matrices.
Their initial requests are:

| Fivefold aggregation workflow | Comparator or working-set basis | Estimated `--mem` | Status |
|---|---|---:|---|
| Shared OLS aggregation | Measured gated aggregation peaked at 2.86 GiB; this job validates 240 shards | `8g` | Comparator-derived, not measured |
| Per-protein OLS aggregation | One stability shard is validated at a time; final stability output is streamed | `8g` | Working-set estimate, not measured |
| Sequence-gated OLS aggregation | Measured gated aggregation plus validation and copying of 1,440 compact coefficient artifacts | `8g` | Comparator-derived, not measured |
| Fixed shared XGBoost aggregation | Compact metrics plus validation and copying of 240 booster artifacts | `8g` | Comparator-derived, not measured |

Record allocation MaxRSS from successful aggregations to refine these estimates
for subsequent runs.

### Fivefold OOF triangle-subgroup replay estimates

The fivefold triangle replay has complete local outputs and process timings/RSS
in its `runtime.csv`. Allocation MaxRSS is not recorded here; the requests below
remain the existing estimates, pending allocation accounting.

| Stage | Initial `--mem` | Basis |
|---|---:|---|
| Reference preparation | `12g` | Bounded one-protein reference arrays and chunked digests; allocation MaxRSS not recorded here. |
| One-layer four-family OOF replay | `12g` | Historical 111-protein replay peaked at 6.15 GiB; the new implementation retains one protein working set plus compact model state. |
| Streaming aggregation | `16g` | One 56,856-row layer plus compact summaries and producer geometry indexes; allocation MaxRSS not recorded here. |

### Publication per-protein subgroup-MAE replay estimates

The independent `publication_subgroup_mae_5fold_v1` workflow reuses the same
four model-artifact loaders and one-protein pair working set as the fivefold
triangle replay, but it does not materialize triangle geometry.

| Stage | Initial `--mem` | Basis |
|---|---:|---|
| One-layer four-family subgroup replay | `12g` | Same bounded model/pair working set as the fivefold triangle replay; allocation MaxRSS not recorded here. |
| Streaming aggregation and pooled reconciliation | `8g` | One 24,288-row shard plus 2,112 compact summary groups; allocation MaxRSS not recorded here. |

## Other workflow references

| Workflow | Best comparable observed MaxRSS | Suggested `--mem` | Basis |
|---|---:|---:|---|
| Shared OLS, `general_base256_combined` | 97.96 GiB | `120g` | Completed base256 jobs peaked at 97.96 GiB for layers 0–23 and 94.63 GiB for layers 24–47. |
| Gated preparation | 13.39 GiB | `20g` | Measured preparation peak with the full combined cohort. |
| Sequence-gated OLS | 93.42 GiB | `120g` | All six measured layer tasks peaked between 93.17 and 93.42 GiB. Full 48-layer coverage changes task count, not per-task matrix size. |
| AA-pair-gated OLS | 28.48 GiB | `40g` | Maximum across the six completed layer tasks. |
| AA-pair × sequence-gated OLS | 31.87 GiB | `48g` | Maximum across the 36 completed layer/sequence tasks. |
| Joint-gate per-layer consolidation | Not yet measured | `16g` initially | This recombines six completed prediction fragments for one layer. Start conservatively, then right-size from its first MaxRSS. |
| Gated final aggregation | 2.86 GiB | `8g` | Maximum across the completed and retried aggregation jobs. |
| Per-protein OLS, `per_protein_base256_common.py` | 7.98 GiB | `12g` | The completed base256 layer jobs reached the current 8 GiB request, leaving effectively no headroom. |
| Triangle-violation subgroup replay, `triangle_violation_subgroup_replay.py` | 6.15 GiB | `8g` | Maximum across 21 successful array tasks (layers 0–20) from Delta job `21445976`; the tasks had consistent memory use, so 8 GiB provides sufficient workflow-specific headroom. |

The canonical paper XGBoost model is the shared 256-feature model. If the
exploratory per-protein XGBoost workflow is retained and also changed from 128
to 256 features, its historical 4.58 GiB peak supports an initial `12g`
request.

These values assume the current dense float64 OLS implementations. A future
streaming or bounded-memory solver should be benchmarked separately rather
than inheriting these requests.

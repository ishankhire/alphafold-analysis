# Suggested SBATCH memory

These are per-job or per-array-task recommendations for the likely final
implementations. Observed MaxRSS values come from Delta accounting, including
the 2026 export in `delta_files/delta_usage_export_2026.tar.gz` and subsequent
`sacct` measurements; they are converted to GiB. Because Delta can derive SU
charges from requested memory, use the smallest request that retains reasonable
headroom and recheck MaxRSS after the first canonical run.

## Fivefold OOF comparator-derived estimates

The fivefold OOF refactor has an explicit project-level no-pilot decision. No
new resource pilot is required before production submission. The requests below
are planning estimates derived from completed same-family jobs plus conservative
scaling, not measurements of the new workflows. This is the authorized override
of the usual exact-workflow pilot rule in `AGENTS.md`; it does not authorize a
deployment or submission during implementation.

| Fivefold workflow | Comparator | Estimated `--mem` | Status |
|---|---:|---:|---|
| Shared OLS, one outer-fold/layer task | Historical shared OLS peak near 98 GiB | `120g` | Comparator-derived, not measured |
| Per-protein OLS, five pair folds per layer | Historical one-split layer below 8 GiB | `12g` | Comparator-derived; loaded features are reused |
| Sequence-gated OLS, one outer-fold/layer task | Historical sequence-gated peak 93.42 GiB | `120g` | Comparator-derived, not measured |
| Fixed shared XGBoost S4/5k/700, one outer-fold/layer task | Completed final base256 peak near 6.4 GiB while evaluating all 552 proteins | `12g` | Comparator-derived; each v2 task evaluates one fold |

Stage 7 confirmed that the four checked-in launchers request exactly `120g`,
`12g`, `120g`, and `12g`, respectively. It also confirmed their planned
concurrency caps and current Delta `scratch&work` filesystem constraints. No
new MaxRSS observation was produced, so every value in this table remains an
estimate and the explicit no-pilot decision remains in force.

After production completes, measured MaxRSS may replace these estimates, but a
pre-production pilot is not a prerequisite for this project.

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

Record MaxRSS from each first successful production aggregation and replace
these estimates before future broad reruns.

### Fivefold OOF triangle-subgroup replay estimates

The versioned `triangle_violation_subgroups_5fold_v1` replay is implemented but
has not yet been measured on Delta. Its initial requests follow the approved
implementation plan and must be replaced with allocation MaxRSS after the first
representative run of each stage.

| Stage | Initial `--mem` | Basis |
|---|---:|---|
| Reference preparation | `12g` | Bounded one-protein reference arrays and chunked digests; no 552-protein measurement yet. |
| One-layer four-family OOF replay | `12g` | Historical 111-protein replay peaked at 6.15 GiB; the new implementation retains one protein working set plus compact model state. |
| Streaming aggregation | `16g` | One 56,856-row layer plus compact summaries and producer geometry indexes; no exact measurement yet. |

### Publication per-protein subgroup-MAE replay estimates

The independent `publication_subgroup_mae_5fold_v1` workflow reuses the same
four model-artifact loaders and one-protein pair working set as the fivefold
triangle replay, but it does not materialize triangle geometry.

| Stage | Initial `--mem` | Basis |
|---|---:|---|
| One-layer four-family subgroup replay | `12g` | Same bounded model/pair working set as the unmeasured fivefold triangle replay; retain its initial request until a representative MaxRSS is recorded. |
| Streaming aggregation and pooled reconciliation | `8g` | One 24,288-row shard plus 2,112 compact summary groups; no production measurement yet. |

| Workflow | Best comparable observed MaxRSS | Suggested `--mem` | Basis |
|---|---:|---:|---|
| Shared OLS, `general_base256_combined` | 97.96 GiB | `120g` | Completed base256 jobs peaked at 97.96 GiB for layers 0–23 and 94.63 GiB for layers 24–47. |
| Final shared XGBoost, directional base256 | 27.43 GiB at 128 features | `64g` initially | The measured search used averaged 128-channel features. Straight doubling gives a conservative planning estimate of about 55 GiB for the final 256 features; measure one canonical layer before lowering this. |
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

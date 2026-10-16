# Suggested SBATCH walltime

These are per-job or per-array-task walltime references for cluster workflows.
Observed elapsed times come from successful Delta allocation records in
`delta_files/delta_usage_export_2026.tar.gz` and subsequent `sacct`
measurements. Slurm schedules against requested walltime rather than expected
runtime, so substantially oversized requests can prevent otherwise short jobs
from fitting into backfill windows.

## Sizing rule

- Use the maximum elapsed time from representative successful tasks, not the
  mean, and normally request 1.5–2x that maximum.
- Round up enough to cover environment startup, output finalization, and modest
  filesystem variability. Very short jobs should generally retain at least a
  10–15 minute request.
- Match the reference workflow's code path, cohort, feature width, stage, and
  task granularity. Exclude failed, cancelled, cached, restart-skipped, smoke,
  and no-op runs unless sizing that exact path.
- Treat a timeout as a lower bound rather than a usable runtime measurement.
- When no exact reference exists, estimate from the closest completed workflow
  and account explicitly for additional proteins, layers, features, folds,
  models, or sequential stages. Submit the authorized full array, then use
  successful allocation times to refine the estimate for subsequent runs.
- For heterogeneous pipelines, size preparation, layer, and consolidation jobs
  separately rather than applying the slowest stage's limit to every stage.

## Fivefold OOF resource references

### Saved-distance TM-score rescoring estimates

The [TM-score runbook](../../scripts/cluster_scripts/tm_scores/README.md) submits
the authorized full array using these estimates, without a separate pilot.

| Stage | Initial `--time` | Basis |
|---|---:|---|
| One-layer four-probe rescoring and PDB export | `02:00:00` | Retained provisionally for the new 2,208 atomic PDB writes per layer. Score-only array `22908533` completed 37 tasks in 9m23s–13m15s; PDB-enabled filesystem latency remains unmeasured. |
| Consolidation only | `00:30:00` | Checks compact manifests/tables and writes consolidated scores, PDB indexes, and geometry summaries. The full audit consumed only about 4m20s of CPU in its first three hours; allow roughly 10–15 minutes for complete table processing and small-file I/O, with 2x headroom. No individual arrays/PDBs are reopened. |
| Optional full-file audit | `09:00:00` | Job `22923530` verified 24/48 layers without errors before user-requested cancellation at 3h03m. Extrapolate about six hours plus 1.5x headroom; serial shared-filesystem reads dominate. The former 4.5-hour request was insufficient. |

Replace with 1.5–2x representative successful allocation times, excluding
checkpoint-skipped and local smoke runs.
The retained two-hour PDB-worker limit is deliberately conservative for the
first durable PDB export; it is not a 1.5–2x estimate from the score-only run.

The one-off dependency-gated deployment/validation job uses `00:10:00` for
environment startup, archive/checksum checks, loading the worker's source
metadata, and a small native-backend check. This is an unmeasured estimate at
the usual minimum scheduling limit, not a scoring resource pilot.

### Packed predicted-distance export estimates

The [export runbook](../../scripts/cluster_scripts/predicted_distances/README.md)
uses full-array submission. Successful full-stage allocation timings remain
unrecorded here; finalization's revised whole-run estimate is noted below.

| Stage | Initial `--time` | Basis |
|---|---:|---|
| Export preparation | `00:15:00` | Streaming hashes of approximately 1.35 GiB of unique models, compact producer tables, and 552 PDBs; no fitting or geometry. |
| One-layer four-family prediction export | `02:00:00` | Matches the closest 552-protein subgroup-replay estimate; removes geometry, adds pair-input hashes, 2,208 per-protein NPY writes, and durable per-protein progress updates. |
| Export finalization | `04:30:00` | The original 15-minute estimate was disproved by timeout `22907683`; only four layers were verified. Their validation timestamps imply roughly 2.5–3.6 minutes per layer, giving an estimated 2–3 hours for 48 layers plus producer checks. The new provisional request adds about 1.5x headroom to that whole-run estimate; the timed-out allocation is only a lower bound, not a successful runtime reference. |

Replace these estimates with 1.5–2x representative successful elapsed times
before repeat runs; exclude failed, smoke, skipped, and no-op tasks.

### Regression worker observations

Successful process elapsed times come from family `runtime.csv` files under
`delta_files/regression_5fold_v1/`. They are not allocation walltimes and may
omit startup/finalization. Requests below are the unchanged launcher values;
use allocation accounting before revising them.

| Workflow | Tasks | Maximum process elapsed | Existing `--time` |
|---|---:|---:|---:|
| Shared OLS, outer-fold/layer | 240 | `00:22:28` | `00:30:00` |
| Per-protein OLS, five pair folds/layer | 48 | `01:39:34` | `02:00:00` |
| Sequence-gated OLS, outer-fold/layer | 240 | `00:12:57` | `03:00:00` |
| Fixed shared XGBoost, outer-fold/layer | 240 | `00:13:50` | `00:30:00` |

### Fivefold aggregation estimates

| Fivefold aggregation workflow | Comparator or workload basis | Estimated `--time` | Status |
|---|---|---:|---|
| Shared OLS aggregation | Measured gated aggregation completed in `00:04:09`; allow for strict validation of 240 shards | `00:30:00` | Comparator-derived, not measured |
| Per-protein OLS aggregation | Validate and stream 6,782,976 stability rows and build a 132,480-row replay index | `01:00:00` | Workload estimate, not measured |
| Sequence-gated OLS aggregation | Validate 240 shards and copy/check 1,440 coefficient artifacts | `00:30:00` | Comparator-derived, not measured |
| Fixed shared XGBoost aggregation | Validate 240 shards and copy/check 240 native boosters | `00:30:00` | Comparator-derived, not measured |

Record successful allocation-level aggregation times to refine these estimates
for subsequent runs.

### Fivefold OOF triangle-subgroup replay estimates

The fivefold triangle replay has complete local outputs and process timings
in its `runtime.csv`. Allocation walltimes are not recorded here; the limits
below remain existing estimates, pending allocation accounting.

| Stage | Initial `--time` | Basis |
|---|---:|---|
| Reference preparation | `06:00:00` | Exact reference enumeration for 552 proteins with per-protein checkpoints. |
| One-layer four-family OOF replay | `06:00:00` | Approximately five times the historical 111-protein cohort, with all four OOF families replayed sequentially per protein. |
| Streaming aggregation | `02:00:00` | Layer-bounded validation and derivation of 2,729,088 consolidated rows; allocation timing not recorded here. |

### Publication per-protein subgroup-MAE replay estimates

| Stage | Initial `--time` | Basis |
|---|---:|---|
| One-layer four-family subgroup replay | `02:00:00` | Replays all 552 proteins but omits the triangle enumeration performed by the closest fivefold replay comparator. |
| Streaming aggregation and pooled reconciliation | `00:30:00` | Streams 1,165,824 rows and validates 2,112 additive comparisons; allocation timing not recorded here. |

## Measured references

The recommendations below retain practical headroom while avoiding the much
larger historical limits. A row with only one completed allocation is an
initial reference and should be confirmed on its next representative run.

| Workflow or SBATCH job name | Representative completed tasks | Maximum elapsed | Suggested `--time` | Basis |
|---|---:|---:|---:|---|
| Triangle-violation subgroup replay, `TriSubReplay` | 21 | `00:40:50` | `01:20:00` | Layers 0–20 from job `21445976`; 20 tasks finished in about 11–15 minutes and layer 6 set the maximum. |
| Context-feature sweep, `DiagContextSweep` | 1 | `08:09:11` | `12:00:00` | One full successful allocation; retain the existing limit until another representative run is measured. |
| Sequence-separation gated diagnostic, `DiagSeqSepGated` | 6 | `00:04:51` | `00:15:00` | Maximum across six successful layer tasks. |
| Gated preparation, `GatedPrep` | 1 | `00:12:29` | `00:30:00` | One successful full-cohort preparation allocation. |
| Sequence-gated OLS, `GatedSeq` | 6 | `01:37:20` | `03:00:00` | Maximum across six successful layer tasks. |
| AA-pair-gated OLS, `GatedAA` | 6 | `00:11:26` | `00:30:00` | Maximum across six successful layer tasks. |
| AA-pair × sequence-gated OLS, `GatedAASeq` | 36 | `00:06:46` | `00:20:00` | Maximum across 36 successful layer/sequence tasks. |
| Gated aggregation, `GatedAgg` | 1 | `00:04:09` | `00:15:00` | One successful complete aggregation allocation. |
| Shared exact-triangle analysis, `Gen256ExactTri` | 48 | `00:04:48` | `00:15:00` | Maximum across the complete 48-layer array. |
| Shared train/test layers 0–23, `GenTTComb256L0023` | 1 | `09:44:45` | `18:00:00` | One successful 24-layer sequential allocation. |
| Shared train/test layers 24–47, `GenTTComb256L2447` | 1 | `08:56:51` | `18:00:00` | One successful 24-layer sequential allocation; aligned with the slower first half. |
| Shared train/test layer 47, `GenTTComb256L47` | 2 | `00:10:38` | `00:30:00` | Maximum across two successful single-layer allocations. |
| Per-protein OLS array, `PerProtB256Array` | 48 | `00:10:41` | `00:30:00` | Maximum across the complete 48-layer array. |
| Per-protein exact-triangle analysis, `PerProt256ExactTri` | 48 | `00:06:01` | `00:20:00` | Maximum across the complete 48-layer array. |
| Per-protein OLS ten-layer sequential shards, `PerProtB256L0009` through `PerProtB256L4047` | 5 | `01:40:38` | `03:00:00` | Maximum across the five successful historical layer-range allocations. |
| Per-protein OLS layer 47, `PerProtB256L47` | 1 | `00:13:19` | `00:30:00` | One successful single-layer allocation. |
| Salt-bridge layer diagnostic, `SaltBridgeLayers` | 48 | `00:02:38` | `00:10:00` | Maximum across the complete 48-layer array, rounded up for startup variability. |
| Sequence-gated secondary-structure layer stage, `SeqGateSecLayers` | 6 | `00:01:12` | `00:15:00` | Maximum across six successful layer tasks; use only for the layer stage. |
| Per-protein XGBoost search, `XGBPerProtSearch` | 6 | `00:12:02` | `00:30:00` | Maximum across six successful search tasks. |
| Shared XGBoost search, `XGBSharedSearch` | 6 | `00:25:03` | `01:00:00` | Maximum across six successful search tasks. |

## Collecting new references

For a completed job or array, obtain allocation-level runtimes with:

```bash
sacct -X -j JOB_ID \
  --format=JobID%24,JobName%35,State,Elapsed,ElapsedRaw,Timelimit
```

Use step-level records when a batch allocation contains multiple meaningful
steps, and preserve MaxRSS separately in
`docs/cluster/suggested_sbatch_memory.md`:

```bash
sacct -j JOB_ID \
  --format=JobID%24,JobName%35,State,Elapsed,Timelimit,MaxRSS
```

The accounting export currently ends on 2026-08-18. Measurements from newer
jobs must therefore be collected from live `sacct` output and recorded here.

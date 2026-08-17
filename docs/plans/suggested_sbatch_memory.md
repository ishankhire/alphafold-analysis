# Suggested SBATCH memory

These are per-job or per-array-task recommendations for the likely final
implementations. Observed MaxRSS values come from the 2026 Delta accounting
export in `delta_files/delta_usage_export_2026.tar.gz`; they are converted to
GiB. Because Delta can derive SU charges from requested memory, use the smallest
request that retains reasonable headroom and recheck MaxRSS after the first
canonical run.

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

The canonical paper XGBoost model is the shared 256-feature model. If the
exploratory per-protein XGBoost workflow is retained and also changed from 128
to 256 features, its historical 4.58 GiB peak supports an initial `12g`
request.

These values assume the current dense float64 OLS implementations. A future
streaming or bounded-memory solver should be benchmarked separately rather
than inheriting these requests.

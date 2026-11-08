# Null models for triangle violations (CAMEO)

Are the probes' triangle-violation curves explained by per-pair error size alone? Each
null adds errors to the final OpenFold Cα–Cα distances (`d_true`). The errors match the
probe's residuals bin by bin but are rearranged among pairs. The nulls are scored with
the paper's all-triangles metrics (violation fraction, normalized mean violation) and
compared with the observed OOF predictions.

## Design

- **Proteins:** the 154 CAMEO proteins of the 552 cohort. ECOD inputs are unavailable.
- **Probes:** `shared_ols_base256`, `per_protein_ols_base256`, `sequence_gated_ols_base256`
  and `shared_xgboost_base256_fixed_s4`, at layers 0–47.
- **Residuals:** `r = d_pred − d_true`, raw.
- **Bins on `d_true`**, half-open:
  - B10 = [0,4) [4,6) [6,8) [8,10) [10,14) [14,20) [20,30) [30,40) [40,55) [55,∞)
  - B5 = [0,5) [5,10) [10,20) [20,40) [40,∞)
  - B20 = B10 with each bin split at its midpoint; the last bin is split at 70.
  - B10×S6 = B10 crossed with the six canonical sequence-separation bins.
- **Null models:**

  | Model | Errors added to `d_true` | Runs |
  |---|---|---|
  | Null A | Gaussian with this protein's bin mean and SD. A bin with fewer than 30 pairs uses the cohort's protein-equal mean of those, from proteins with ≥30 pairs | B10 |
  | Null B | This protein's residuals, permuted within each bin | B10, B10×S6; B5 and B20 for shared OLS |
  | Null C | Bin mean plus per-residue offsets `u_i + u_j` kept, with `u = Σ(centred residuals touching i)/(n−2)`; only the remainder is permuted within bin | B10, B10×S6 |

- **Draws:** 10 per null, seeded by `(20261107, layer, probe, protein, model, binning, draw)`.
- **Triangles:**
  - **Observed:** scored on every triple (exact), and also on the protein's triangle sample.
  - **Triangle sample:** every triple if C(R,3) ≤ 1M, else 1M distinct triples, fixed per protein across layers, probes and nulls.
  - **Nulls:** scored on the sample, so comparing a null with sampled observed cancels triangle sampling.
- **Metric:** `null_triangles_common.score_set` follows the all-triangles reduction in `compute_metric_rows` operation by operation: clip at 0, sort the sides, use the 1e-12 Å tolerance, and normalize by the reference RMS side.
- **Gates:** each layer must pass three gates before anything is written.
  1. Bit-identical to `compute_metric_rows` on every protein whose sample is every triple.
  2. Exact observed matches the paper's per-protein values (`paper_triangle_parity_cameo.csv`, extracted from `per_protein_triangle_metrics.csv`), within float32-cache tolerance:
     - violation-fraction gap ≤ 5e-5;
     - relative gap in the normalized-violation sum ≤ 1e-4;
     - negative-prediction count within 2.

     The paper replayed float64 predictions; the cache is float32.
  3. Null B keeps every bin's residual histogram on every draw.

## Deploy

Copy flat into `/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`:

- from this directory:
  - `null_triangles_common.py`, `null_triangles_replay.py`, `aggregate_null_triangles.py`;
  - both `.sbatch` files;
  - `paper_triangle_parity_cameo.csv`;
  - this README, as `null_triangles_README.md`.
- These shared helpers are already deployed; check that their checksums match the repository:
  - `cluster_regression_helpers.py`, `regression_protocol.py`, `regression_output_helpers.py` and `canonical_storage.py` from `../..`;
  - `triangle_violation_subgroup_common.py` from `../../triangle_violation_subgroups/`;
  - `regression_protocol_data/`.

## Inputs and outputs

- **Inputs, from `$CSV_ROOT/regression_5fold_v1/`:**
  - `metadata/regression_outer_folds_v1.csv`;
  - `predicted_distances_5fold_v1/layers/layer_LL/` (`prediction_index.csv` checksums and `<id>/<probe>.npy`);
  - CAMEO final PDBs via `resolve_combined_pdb_path`.
- **Output root:** `NULL_TRIANGLES_OUTPUT_ROOT`, default `/work/hdd/bfna/ikhire/diagnostic_experiments/null_triangles_cameo_v1/`.
- **Slurm logs:** `<root>/logs/`, which must exist before submission.

| Output | Row | Principal columns |
|---|---|---|
| `layer_shards/null_metrics_layer_LL.csv.gz` | probe × protein × model × binning × triangle set × draw (draw −1 = observed) | `n_triangles, n_violations, violation_sum_angstrom, relative_violation_sum, violation_fraction, mean_relative_violation_all, n_violations_with_clipped_side, n_negative_values` |
| `layer_shards/pair_diagnostics_layer_LL.csv.gz` | probe × protein × binning (B10, B10xS6) × occupied cell | `n_pairs, mean_r, sd_r, rmse, skew, excess_kurtosis, null_a_fallback` |
| `layer_shards/calibration_layer_LL.csv` | probe × protein | `alpha, beta` of `d_pred = α + β·d_true`, `mean_r, rmse, n_negative_predictions` |
| `validation/layer_LL.json` | layer | gate results, Null A fallback counts, timings; marks the layer complete |
| `tables/null_per_protein.csv` | probe × layer × protein × model × binning × triangle set | draw mean and draw SD of both metrics, `clipped_side_fraction` (observed) |
| `tables/error_diagnostics_by_bin.csv` | probe × layer × B10 bin | protein-equal means over proteins with ≥10 pairs: bias, sd, rmse, bias², sd², mse, skew, kurtosis |
| `tables/crosstab_b10_seqsep.csv` | probe × layer × B10×S6 cell | the same statistics; empty if the cohort has < 100 pairs in the cell |
| `tables/calibration_per_protein.csv`, `tables/null_a_fallback_counts.csv`, `tables/run_manifest.json` | — | — |

## Submit

```bash
cd /work/hdd/bfna/ikhire/openfold_interpretability_canonical
mkdir -p /work/hdd/bfna/ikhire/diagnostic_experiments/null_triangles_cameo_v1/logs
array=$(sbatch --parsable null_triangles_replay.sbatch)          # tasks 0-47 = layers, 16 at once
sbatch --dependency=afterok:${array%%;*} aggregate_null_triangles.sbatch
```

Add `--mail-user="$CLUSTER_NOTICE_EMAIL"` to both commands when that variable is set.
A layer whose `validation/layer_LL.json` exists is skipped, so rerunning the array only
redoes missing layers. Bootstraps and figures are made locally by
`scripts/multi_protein/null_triangles/analyze_null_triangles.py` from `tables/`.

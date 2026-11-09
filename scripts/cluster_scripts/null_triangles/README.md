# Null models for triangle violations (552-protein cohort)

Are the probes' triangle-violation curves explained by per-pair error size alone? Each
null adds errors to the final OpenFold Cα–Cα distances (`d_true`). The errors match the
probe's residuals bin by bin but are rearranged among pairs. Observed OOF predictions and
nulls are scored on **every triangle** with the paper's all-triangles metrics (violation
fraction, normalized mean violation).

This supersedes the CAMEO-only diagnostic `null_triangles_cameo_v1`. Its code is at commit
`4741ee5`; its outputs are in `delta_files/historical_outputs/null_triangles_cameo_v1/`.

## Design

- **Proteins:** the frozen 552-protein cohort, 154 CAMEO + 398 ECOD, treated as one dataset.
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
  | Null A | Gaussian with this protein's bin mean and SD. A bin with fewer than 30 pairs uses the cohort's protein-equal mean of those, from proteins with ≥30 pairs | B10, all probes |
  | Null B | This protein's residuals, permuted within each bin | B10, all probes; B5, B20, B10×S6 for shared OLS |

- **Draws:** 3 per null, seeded by `(20261109, layer, probe, protein, model, binning, draw)`.
- **Triangles:** every triple, for observed and every null. All 37 vectors of a protein are scored together in one pass:
  - 4 observed;
  - 24 for Nulls A and B on B10;
  - 9 for the shared-OLS sensitivity runs.

  Each `iter_triangle_indices` batch is generated once.
- **Metric:** `null_triangles_common.score_streaming` follows the all-triangles reduction in `compute_metric_rows` operation by operation: clip at 0, sort the sides, use the 1e-12 Å tolerance, and normalize by the reference RMS side.
- **Gates:** each layer must pass these before anything is written.
  1. Bit-identical to `compute_metric_rows` on every protein with ≤ 1M triangles.
  2. Observed matches the paper's per-protein values for all 552 × 4 (`paper_triangle_parity_5fold_v1.csv.gz`), within float32-cache tolerance:
     - violation-fraction gap ≤ 5e-5;
     - relative gap in the normalized-violation sum ≤ 1e-4;
     - negative-prediction count within 2.

     This also checks that the restored ECOD PDBs are correct.
  3. Null B keeps every bin's residual histogram on every draw.
  4. The cohort is 154 CAMEO + 398 ECOD; each PDB's Cα count equals `n_residues`; prediction checksums match `prediction_index.csv`.

**Parity table.** `paper_triangle_parity_5fold_v1.csv.gz` holds the `all_triangles` rows of `regression_5fold_v1/analysis/triangle_violation_subgroups_5fold_v1/per_protein_triangle_metrics.csv` (source sha256 `badc45f0…24c7f`). It was extracted with:

```python
rows = [r for r in csv.DictReader(open(src)) if r["category_system"] == "all_triangles"]
rows.sort(key=lambda r: (int(r["layer"]), r["protein_id"], r["analysis_name"]))
# columns: source, analysis_name, layer, protein_id, n_residues, n_triangles, n_violations,
#          n_negative_predictions, violation_sum_angstrom, relative_violation_sum
```

**Cost.** About 85 CPU-hours: 2.06×10⁹ triangles × 37 vectors per layer, 48 layers. At 16 CPUs a layer should take about 7–12 minutes.

## Deploy

Copy flat into `/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`:

- from this directory:
  - `null_triangles_common.py`, `null_triangles_replay.py`, `aggregate_null_triangles.py`;
  - both `.sbatch` files;
  - `paper_triangle_parity_5fold_v1.csv.gz`;
  - this README, as `null_triangles_README.md`.
- `cluster_regression_helpers.py` from `..`, since `ECOD_PDB_ROOT` now points to the restored `/work/hdd/bfna/ikhire/thayes/ecod/experiments/openfold_reps_outputs_sm`.
- These shared helpers are already deployed; check that their checksums match the repository:
  - `regression_protocol.py`, `regression_output_helpers.py` and `canonical_storage.py` from `..`;
  - `triangle_violation_subgroup_common.py` from `../triangle_violation_subgroups/`;
  - `regression_protocol_data/`.

## Inputs and outputs

- **Inputs, from `$CSV_ROOT/regression_5fold_v1/`:**
  - `metadata/regression_outer_folds_v1.csv`;
  - `predicted_distances_5fold_v1/layers/layer_LL/` (`prediction_index.csv` checksums and `<id>/<probe>.npy`);
  - final PDBs via `resolve_combined_pdb_path`. CAMEO PDBs come from bekh; ECOD PDBs come from the restored copy.
- **Output root:** `NULL_TRIANGLES_OUTPUT_ROOT`, default `/work/hdd/bfna/ikhire/regression_5fold_v1/analysis/null_triangles_5fold_v1/`.
- **Slurm logs:** `<root>/logs/`, which must exist before submission.

| Output | Row | Principal columns |
|---|---|---|
| `layer_shards/null_metrics_layer_LL.csv.gz` | probe × protein × model × binning × draw (draw −1 = observed); 20,424 rows per layer | `source, n_residues, n_triangles, n_violations, violation_sum_angstrom, relative_violation_sum, violation_fraction, mean_relative_violation_all, n_violations_with_clipped_side, n_negative_values` |
| `layer_shards/pair_diagnostics_layer_LL.csv.gz` | probe × protein × binning (B10, B10xS6) × occupied cell | `n_pairs, mean_r, sd_r, rmse, skew, excess_kurtosis, null_a_fallback` |
| `layer_shards/calibration_layer_LL.csv` | probe × protein | `alpha, beta` of `d_pred = α + β·d_true`, `mean_r, rmse, n_negative_predictions` |
| `validation/layer_LL.json` | layer | gate results, Null A fallback counts, timings, inputs; marks the layer complete |
| `tables/null_per_protein.csv.gz` | probe × layer × protein × model × binning | draw mean and draw SD of both metrics, `n_draws`, `clipped_side_fraction` (observed) |
| `tables/error_diagnostics_by_bin.csv` | probe × layer × B10 bin | protein-equal means over proteins with ≥10 pairs: bias, sd, rmse, bias², sd², mse, skew, kurtosis |
| `tables/crosstab_b10_seqsep.csv` | probe × layer × B10×S6 cell | the same statistics; empty if the cohort has < 100 pairs in the cell |
| `tables/calibration_per_protein.csv`, `tables/null_a_fallback_counts.csv`, `tables/run_manifest.json` | — | — |

## Submit

```bash
cd /work/hdd/bfna/ikhire/openfold_interpretability_canonical
mkdir -p /work/hdd/bfna/ikhire/regression_5fold_v1/analysis/null_triangles_5fold_v1/logs
# Pilot one layer, check its validation JSON, time and memory, then run the rest.
sbatch --mail-user="${CLUSTER_NOTICE_EMAIL:?}" --array=47 null_triangles_replay.sbatch
array=$(sbatch --parsable --mail-user="$CLUSTER_NOTICE_EMAIL" --array=0-46%16 null_triangles_replay.sbatch)
sbatch --mail-user="$CLUSTER_NOTICE_EMAIL" --dependency=afterok:${array%%;*} aggregate_null_triangles.sbatch
```

A layer whose `validation/layer_LL.json` exists is skipped, so rerunning the array only
redoes missing layers.

Copy `tables/` to `delta_files/regression_5fold_v1/analysis/null_triangles_5fold_v1/tables/`. Bootstraps, text and figures are then made locally by `scripts/multi_protein/null_triangles/analyze_null_triangles.py`, under `figures/{main,supplement,diagnostics}/`.

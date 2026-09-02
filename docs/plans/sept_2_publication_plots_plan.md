# September 2 Fivefold Publication Figure Bundle

## Summary

- Create `scripts/multi_protein/generate_sept_2_publication_plots.py` and write
  the complete bundle to `delta_files/publication_plots/sept_2_plots/`.
- Read analytical and provenance inputs exclusively from
  `delta_files/regression_5fold_v1/`; never modify that input tree.
- Use the four protocol-2.0 models, 552 proteins, and layers 0--47. Exclude the
  AA-gated model through an explicit whitelist.
- Use protein-equal means with deterministic pointwise percentile 95%
  intervals from 2,000 paired-protein bootstrap replicates and seed
  `20260902`.
- Apply BH separately to each triangle metric across all four models, 13
  plotted triangle trajectories, both directions, and layers 5--42: 3,952
  hypotheses per metric.

## Inputs and statistical methods

- Read R2 from each model's `per_protein.csv`, TM-score from
  `geometry_metrics.csv`, per-protein subgroup MAE from
  `analysis/publication_subgroup_mae_5fold_v1/per_protein_subgroup_mae.csv`,
  and triangle values from
  `analysis/triangle_violation_subgroups_5fold_v1/per_protein_triangle_metrics.csv`.
- Plot four spatial bins (`0-5`, `5-10`, `10-40`, `40+`) and six sequence bins
  (`1`, `2-4`, `5-9`, `10-19`, `20-49`, `50+`). Derive each protein's `10-40`
  MAE exactly by summing its `10-20` and `20-40` absolute-error totals and pair
  counts before division. Average finite per-protein MAEs equally and retain
  unsupported rows and support counts.
- Use only `all_triangles`, `rms_size`, `area`, `relative_slack`, and broad
  `angle_shape`. Both triangle fraction and normalized violation are computed
  per protein before equal-protein aggregation.
- Reproduce the historical five-layer peak/trough candidate and paired-test
  method. Use the larger of the left/right one-sided p-values and mark hollow
  candidates and filled BH-supported extrema.
- Min-max normalize the 48 protein-equal R2 values within model for the
  decodability-consistency phase plots. Jointly bootstrap matched R2 and
  triangle observations and show horizontal/vertical intervals at layers
  0, 10, 20, 30, 40, and 47.

## Figures and source data

- Write 9 main PDF/PNG pairs: R2, TM-score, combined plus four standalone
  spatial MAE figures, normalized all-triangle peak/trough overlay, and
  normalized all-triangle decodability-consistency.
- Write 17 supplemental PDF/PNG pairs: combined plus six standalone sequence
  MAE figures, fraction-only all-triangle peak/trough and phase figures, four
  annotated triangle-stratification figures, and four 2x2 model-separated
  individual-protein trajectory figures.
- Put raw values, bootstrap summaries, all 7,904 peak/trough tests, candidates,
  trajectory summaries, and decodability-consistency points in `source_data/`.
- Write a source-hashed `figure_manifest.csv`, `generation_summary.json`, and
  README. Do not create SD or SE figure variants.

## Validation and acceptance

- Validate the exact model, protein, layer, key, coverage, support, and raw-row
  counts specified in the approved implementation request.
- Confirm all 2,112 pooled reconciliation rows are complete, but never use
  pooled MAE for a plotted value.
- Enforce that every resolved analytical input is below
  `delta_files/regression_5fold_v1/` and fail on historical, pooled-only,
  incomplete, or out-of-root inputs.
- Add synthetic tests for aggregation, missing support, bootstrap
  determinism, joint phase intervals, BH families, extrema candidates, domain
  clipping, and filenames.
- Generate and inspect the bundle, verify all 28 PDF/PNG pairs and hashes,
  update `branch-analysis.md`, and commit/push only scoped changes while
  preserving unrelated working-tree changes.

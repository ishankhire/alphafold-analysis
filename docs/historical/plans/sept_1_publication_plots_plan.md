> Historical reference; not current execution instructions. See [current sources and status](../../project_status.md).

# September 1 Publication Figure Bundle

## Summary

- Create a self-contained publication bundle at
  `delta_files/publication_plots/sept_1_plots/`, with `main_figures/`,
  `supplemental_figures/`, `source_data/`, a figure manifest, generation
  metadata, and README.
- Use the complete protocol-2.0 fivefold OOF results: four models, 552
  proteins, and layers 0--47. Exclude the AA-gated model everywhere.
- Use the historical 111-protein triangle results only as placeholders. Mark
  this in CSV/manifest provenance without adding placeholder text to figures.

## Missing per-protein subgroup data

### Interim pooled-MAE generation mode (user-approved September 1, 2026)

- Until the per-protein subgroup replay is copied back from Delta, the local
  generator may use the canonical protocol-2.0 pair-pooled subgroup tables for
  the same four models, 552 proteins, and 48 layers.
- This is an explicit, temporary fallback rather than a silent substitution.
  Source tables, the figure manifest, bundle README, and generation metadata
  label the aggregation as `pair_pooled_mae` and flag the temporary status.
- Temporary MAE plots use the y-axis label `Pooled MAE (Å)` and do not show
  bootstrap, SD, or individual-protein variants because those quantities
  cannot be recovered from a single pooled estimate. The combined and
  standalone spatial/sequence PDF/PNG pairs are still generated.
- When the complete per-protein replay CSV becomes available, rerunning the
  same generator automatically restores the planned protein-equal curves and
  their uncertainty/heterogeneity variants.

- Add an independent Delta replay workflow under
  `scripts/cluster_scripts/publication_figures/`. Reuse the saved OOF
  coefficient and booster artifacts without refitting.
- Write
  `$CSV_ROOT/analysis/publication_subgroup_mae_5fold_v1/per_protein_subgroup_mae.csv`
  and copy it locally to the matching `delta_files/analysis/` path.
- Emit exactly 1,165,824 rows: one row per model, layer, protein, and each of
  six sequence-separation or five spatial-distance bins. Retain empty bins as
  explicit unsupported rows.
- Aggregate per-protein MAE with equal protein weight. Reconcile additive
  error/count primitives against the existing pair-pooled subgroup tables as
  a validation only; never use pooled MAE for publication curves.
- Provide restartable layer shards, provenance, runtimes, strict aggregation,
  tests, and Delta launchers. Start with 8 CPUs, 12 GiB, 2 hours, and a
  `0-47%12` array for replay; use 8 GiB and 30 minutes for aggregation.

## Figures and source data

- Implement
  `scripts/multi_protein/generate_sept_1_publication_plots.py` as a fresh local
  generator.
- Main PDF/PNG pairs: fivefold mean per-protein R2; reconstruction TM-score;
  combined and five standalone spatial-distance MAE figures; a historical
  normalized-mean all-triangles peak/trough overlay; and a historical
  normalized-mean all-triangles decodability-consistency figure. The
  triangle-inequality violation-fraction versions of the latter two figures
  are retained as supplemental PDF/PNG pairs.
- Supplemental PDF/PNG pairs: combined and six standalone sequence-separation
  MAE figures; verbatim historical RMS-size, area, relative-slack, and
  angle-shape annotated trajectories; and two uncertainty/heterogeneity
  variants for the top-level R2, TM-score, combined spatial, and combined
  sequence figures.
- Primary newly rendered means use a deterministic paired-protein percentile
  bootstrap with 10,000 replicates and seed 20260901. Supplemental variants
  show sample mean +/- 2 SD and thin, transparent individual-protein
  trajectories with bold model means.
- Put raw values and plotted summaries in `source_data/`. Summary rows retain
  mean, SD, SE, pointwise bootstrap bounds, 2-SD bounds, counts, aggregation,
  scope, and historical-placeholder status.
- Write a source-hashed `figure_manifest.csv`, `generation_summary.json`, and
  README. Preserve model scopes and flag historical triangle provenance in
  metadata without adding visual placeholder labels.

## Tests and acceptance criteria

- Test canonical bin boundaries, empty bins, protein-equal aggregation,
  deterministic bootstrap behavior, 2-SD calculations, and filenames.
- Validate four models, 552 common proteins, 48 layers, complete OOF coverage,
  unique keys, and no AA-gated data.
- Reconcile replay primitives to the existing pooled subgroup MAEs.
- Require every requested figure to have nonempty PDF/PNG files and declared
  source CSVs; require copied historical assets to retain their hashes.
- If the per-protein subgroup replay output exists, fail clearly when it is
  incomplete rather than falling back. If it is absent, use only the explicit
  user-approved pooled-MAE placeholder mode documented above.
- Run relevant tests with `./.venv`, generate the complete bundle once the
  replay CSV is available, visually inspect representative figures, run
  integrity checks, update `branch-analysis.md`, and commit/push only the
  scoped changes.

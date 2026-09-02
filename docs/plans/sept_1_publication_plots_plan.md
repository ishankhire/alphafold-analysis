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
  all-triangles peak/trough overlay; and a verbatim copy of the historical
  all-triangles decodability-consistency figure.
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
- Fail clearly if the per-protein subgroup replay output is missing or
  incomplete. Never silently substitute pooled MAE.
- Run relevant tests with `./.venv`, generate the complete bundle once the
  replay CSV is available, visually inspect representative figures, run
  integrity checks, update `branch-analysis.md`, and commit/push only the
  scoped changes.

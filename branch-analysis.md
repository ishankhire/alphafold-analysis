# Branch Analysis

## Session: 2026-08-25 — Implement triangle-violation subgroup replay

Added reference preparation, checkpointed 48-layer four-probe replay, strict
aggregation, artifact/Figure-5 validation, protein-length analyses, synthetic
geometry tests, and local PDF generation for exact RMS, angle, slack, and area
triangle subgroups. The workflow only evaluates all edges, retains individual
protein rows, and performs no bootstrap resampling. Its Delta replay resolves
the per-protein artifacts from their producer-native
`$CSV_ROOT/per_protein_base256_combined` directory.

## Session: 2026-08-24 — Expand decodability-versus-error phase plots

Added reproducible Figure-8-style phase trajectories for all canonical exact-
triangle metrics, conditional violation severity, and MAE in all five true
spatial-distance bins. Kept the 14 PNG/PDF figure pairs in a dedicated paper-
figure subdirectory and saved their exact points, transition directions,
source-digested manifest, evaluation caveats, and concise interpretation. The
transition diagnostic remains tabular only; it is not overlaid on the plots.

## Session: 2026-08-24 — Triangle-violation subgroup analysis plan

Added a four-model, 48-layer frozen-replay plan for exact triangle violations
stratified by reference RMS size, angle, area, and relative slack. Defined
scale-normalized severity metrics, exact output schemas, and protein-length
composition, stratification, adjustment, and weighting checks.

## Session: 2026-08-18 — Interpret generated main paper figures

Added a concise results narrative for generated main Figures 1--9 and 11--18,
covering the five probe variants, evaluation-scope caveats, figure-specific
findings, and the combined scientific interpretation.

## Session: 2026-08-18 — Generate paper figure catalog items 1–9 and 11–18

Added a reproducible local workflow that derives the catalog tables and writes
PNG/PDF main-paper figures for items 1–9 and 11–18 from the five canonical
probe families. Consolidated triangle catalog items 5 and 6 into one six-panel
figure covering total violation fraction, conditional and maximum magnitude,
and the 0–5, 5–10, and 10+ Å violation fractions. Added source-digested figure
provenance, paired-protein uncertainty summaries, AA-pair timing/gain tables,
and ECOD category, architecture, and protein-length analyses.

## Session: 2026-08-18 — Paper figure catalog plan

Added a content-first plan for 20 separately numbered main figures and 30
supplemental figures covering model comparisons, geometry consistency,
sequence separation, all 210 amino-acid pairs, ECOD categories, protein-length
effects, and secondary-structure emergence. The plan distinguishes ready,
local-derived, running, and new-computation figures and defines their intended
analysis outputs.

## Session: 2026-08-18 — Canonical shared-OLS coefficient replay

Added a non-fitting 48-layer Delta workflow that replays the historical
`general_base256_combined` raw coefficients into canonical regression,
subgroup, geometry, provenance, status, runtime, and model-artifact outputs.
The workflow uses validated layer shards and strict aggregation while retaining
explicit historical split, precision, and unavailable-conditioning caveats.

## Session: 2026-08-13 — Secondary-structure emergence diagnostic

Added a staged DSSP/calibration, six-layer replay array, and consolidation
workflow for held-out helix and beta-sheet emergence, stability, and stratified
distance accuracy using saved `general_base256_combined` coefficients.

## Session: 2026-08-13 — DSSP final-PDB smoke test

Added a one-protein Delta DSSP smoke test that exports raw DSSP, residue and
secondary-structure segment tables, plus coordinate-based CYS SG--SG candidates.
Its temporary PDB preparation now removes OpenFold metadata records that
mkdssp rejects, and the shared annotation pipeline applies the same fix.

## Session: 2026-08-13 — Matched-R² geometry null

Added a checkpointed Delta diagnostic that perturbs final-PDB distances to each
probe's saved held-out R², reruns MDS, and compares observed geometry with an IID
matched-error null without refitting probes.

Restricted it to layers 0/10/20/30/40/47 and made the launcher a concurrency-safe
six-task Slurm array with one checkpointed layer per task.

Made this diagnostic and the DSSP smoke test safe when copied into a flat Delta
working directory instead of retaining their repository-relative nesting.

## Session: 2026-08-07 — ECOD category probe analysis

Added a local ECOD category analysis for shared/per-protein base256 trajectories,
coefficient cosine clustering, permutation testing, and length-adjusted effects.

## Session: 2026-08-07 — CAMEO-50 pair-interaction annotation prototype

Added Delta DSSP setup/job scripts and a restart-safe structural annotation
pipeline for salt bridges, DSSP backbone/beta interactions, and hydrophobic cores.

## Session: 2026-07-27 — Add ECOD input copy utility

Added a Delta-ready script to copy the first 100 ECOD `.a3m` files and layer-47
predicted PDBs into `$HOME/csv_files`, plus a matching sbatch launcher that
activates `~/protein_venv`.

## Session: 2026-07-03 — Rename generalized exact triangle replay

Renamed the generalized base256 exact-triangle replay script/sbatch from
`sample_triangle_violations` to `exact_triangle_violations`.

## Session: 2026-06-19 — Reconcile per-protein geometry CSVs to exact triangle metrics

Added `combine_per_protein_exact_triangle_metrics.py` (merges the 48 per-protein
exact-triangle shards into `exact_triangle_metrics.csv` + `layer_batches/`) and
`reconcile_per_protein_base256_combined_triangle_metrics.py` (per-protein analogue
of the general reconcile; no split, summary grouped by layer). Rewrote
`per_protein_base256_combined/{geometry_metrics,geometry_summary}.csv` to the 7
exact triangle fields. Updated `analyze_delta_per_protein_base256_combined.py`
(new schema + severity-bin triangle plot + corrected report). Verified
`per_protein_base256_common.py` already emits the 7-field schema for future runs
and the shard aggregator is header-agnostic.

## Session: 2026-06-19 — Reconcile combined geometry CSVs to exact triangle metrics

Added `reconcile_general_base256_combined_triangle_metrics.py`: rewrites
`general_base256_combined/geometry_metrics.csv` and `geometry_summary.csv`,
dropping the old min-indirect-path triangle columns and substituting the 7 exact
per-triangle fields from `exact_triangle_metrics.csv` (summary mean/std, ddof=0).
Non-triangle columns preserved exactly. Updated
`analyze_delta_general_base256_combined.py` for the new schema: enriched triangle
plot with severity bins (0-5/5-10/10+ A) and corrected report description.

## Session: 2026-06-19 — Per-protein exact triangle replay

Added `per_protein_base256_exact_triangle_violations.py` (+ sbatch) under
`per_protein_scripts/`: per-protein port of the generalized exact-triangle replay.
Reads each protein's own coefficients from the per-protein layer-window shards
(48-task layer array, one shard per layer), reconstructs pred_dist, writes
triangle-only metrics per protein to `exact_triangle_metrics/`.

## Session: 2026-06-19 — Add project orientation summary

- Added `analysis_summaries/project_orientation_summary.md`: high-level map of
  the project, the two focus directories (`general_base256_combined`,
  `per_protein_base256_combined`), the exact-triangle thread, and repo
  conventions.

## Session: 2026-06-18 — Add exact triangle local analysis

- Added a local exact-triangle helper to merge layer shards, validate the new
  triangle bins, and plot layer-wise violation proportions/magnitudes.

## Session: 2026-06-18 — Add Slurm array reference note

- Added a concise local note for when to use Delta refs vs upstream Slurm docs
  and how Slurm array task IDs/concurrency behave.

## Session: 2026-06-18 — Exact triangle metric replay

- Reworked the generalized base256 triangle replay into a 48-task layer array
  that writes exact per-protein triangle metric shards for all 552 proteins.

## Session: 2026-06-16 — Sample held-out triangle violations

- Added a Delta replay job that samples 2,000 held-out/test triangles per
  protein for layers 0,10,20,30,40,47 using saved `general_base256_combined`
  raw coefficients.

## Session: 2026-06-16 — Analyze per-protein combined base256 outputs

- Added per-protein combined base256 plots and comparison against
  `general_base256_combined`.

## Session: 2026-06-16 — Analyze generalized combined base256 outputs

- Added a local analyzer for `general_base256_combined` plots, coefficient
  cosine heatmap, geometry trends, and baseline comparison.

## Session: 2026-06-16 — Split per-protein base256 layers

Added Delta per-protein base256 shards for layers 0-9, 10-19, 20-29, 30-39,
and 40-47.

## Session: 2026-06-12 — Add per-protein combined base256 regression

Added a Delta layer-47 per-protein base256 regression workflow for the combined
CAMEO+ECOD manifest, including per-model coefficients and geometry metrics.
Fixed its sbatch launcher to resolve the Python entrypoint from
`SLURM_SUBMIT_DIR` instead of Slurm's spool copy.

## Session: 2026-06-12 — Document combined CAMEO and ECOD dataset

Added `datasets.md` with TM-score > 0.7 CAMEO/ECOD counts, Delta input paths,
and protein IDs; updated AGENTS.md to treat them as one combined analysis pool.
Set the default Delta CSV output directory to `$HOME/csv_files`.
Added a combined CAMEO+ECOD generalized train/test layer-47 base256 cluster job.
Added combined base256 wrappers for layer ranges 0-23 and 24-47.

## Session: 2026-06-03 — Add next-layer delta analysis plots

Added a local analyzer for the 7b3a_A next-layer distance-delta outputs that
plots held-out R2, train/test gaps, and context gain over the center-pair probe.

## Session: 2026-06-03 — Add baseline256 layer-47 copy_X k-fold diagnostic

Added a Delta-ready baseline256 layer-47 diagnostic that compares
`LinearRegression(copy_X=False)` against `copy_X=True` under the shared strict
size-stratified 5-fold pipeline.
Kept the diagnostic sbatch launcher aligned with the existing diagnostics
convention: submit from the script directory.

## Session: 2026-06-02 — Add next-layer distance-delta diagnostic

Added a Delta-ready 7b3a_A ridge probe for layer-to-layer decoded distance
changes with k=5/10/20 final-PDB context features.
Fixed two stale miscellaneous ICE sbatch Python targets found during sbatch
reference verification.

## Session: 2026-05-02 — Add Variant A MDS publication diagnostics

Extended publication summary stats with avg-symmetric stress, negative
prediction, and MDS eigenvalue diagnostics.
Moved the generated MDS geometry writeup into a separate publication markdown.

## Session: 2026-05-02 — Add Chimera publication grid

Added a fixed-config Chimera montage composer for the 10 rendered proteins,
writing SM/L10/L15/L22/L47 PNG and PDF grids.

## Session: 2026-05-02 — Add publication summary stats

Added `delta_files/publication_plots/summarize_publication_statistics.py` and
generated compact R2/TM-score summary tables/report for the two figure variants.

## Session: 2026-05-02 — Add original/full 7b3a PyMOL renderer

Added a fixed-view 7b3a renderer for the green original full model and selected
red reconstructed CA layers without B-factor coloring.
Scoped CA-trace settings to reconstructed layers and restored moderate ray
shadows so the full original uses normal secondary-structure cartoon.
Slightly darkened the fixed green/red PyMOL colors and disabled auto-zoom during
the renderer.
Enabled mild depth fading for ray renders so rear cartoon/trace elements fade
toward the white background.
Added per-render clipping reset and forced orthoscopic view to prevent accidental
PyMOL scroll/slab settings from cutting structures out of PNGs.
Reduced the clipping buffer to preserve visible depth fading while still
resetting bad scroll/slab state.
Switched fading from PyMOL fog to deterministic camera-depth residue colors so
the fade survives clipping resets.

## Session: 2026-04-30 — Add PyMOL CA B-factor renderer

Added `chimera_files/render_ca_bfactor_layers.py` to batch-render saved-view
CA traces colored by B-factor bins for the first three reconstructed proteins.
Updated it to find the repo root reliably from PyMOL sessions.
Temporarily narrowed rendering to `7b3a_A` reference plus layer 47 for PyMOL
debugging.
Changed test rendering to create the 7b3a view from the reference and ray-trace
CA tube/stick output to debug blank PyMOL PNGs.
Switched the 7b3a test renderer back to recalling the saved `7b3a_A_view`.
Removed stick overlay artifacts and disabled ray tracing for cleaner PyMOL PNGs.
Changed all-layer 7b3a batch rendering to ray PNGs with shadows disabled.

## Session: 2026-04-26 — Add fold-4 coefficient replay MDS export for first five proteins

### What changed
Updated:
- `branch-analysis.md`

Added:
- `scripts/cluster_scripts/diagnostics/diagnostic_general_base256_avg_symmetric_montage10_mds_replay.py`
- `scripts/cluster_scripts/diagnostics/diagnostic_general_base256_avg_symmetric_montage10_mds_replay.sbatch`

### Why
Adds a Delta-ready replay job that reuses saved baseline256 fold-4
coefficients to reconstruct all-layer MDS coordinates and ChimeraX-friendly
PDB exports for the first five proteins without rerunning regression.

## Session: 2026-04-29 — Retarget replay exporter to avg-symmetric montage proteins

### What changed
Updated:
- `branch-analysis.md`
- `scripts/cluster_scripts/diagnostics/diagnostic_general_base256_avg_symmetric_montage10_mds_replay.py`
- `scripts/cluster_scripts/diagnostics/diagnostic_general_base256_avg_symmetric_montage10_mds_replay.sbatch`

### Why
Repoints the replay job to `general_base256_avg_symmetric` coefficients and a
fixed 10-protein montage set, with avg-symmetric feature replay and renamed
output paths that reflect the new model source.

## Session: 2026-04-29 — Add constrained MDS replay variants

Updated the avg-symmetric montage replay script to also export seq1-clipped and
global-floor-plus-seq1-clipped MDS PDBs, metrics, and coordinate rows.

## Session: 2026-04-26 — Add MDS coordinate PDB exporter

### What changed
Updated:
- `branch-analysis.md`

Added:
- `scripts/multi_protein/export_mds_coords_to_pdb.py`

### Why
Adds a small local utility that converts reconstructed MDS coordinate CSVs into
CA-trace PDBs for PyMOL overlays.

## Session: 2026-04-14 — Add baseline256 segment coefficient evolution analysis

### What changed
Updated:
- `branch-analysis.md`

Added:
- `scripts/multi_protein/regression/analyze_delta_baseline256_kfold_segment_coefficients.py`

### Why
Adds a fold-4 post-processing script that reuses the saved K=3 layer
segmentation to summarize coefficient channels by segment and write
interpretation-focused CSVs plus plots.

## Session: 2026-03-26 — Add generalized avg-symmetric axial train/test shard for layers 00-04

### What changed
Updated:
- `branch-analysis.md`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_train_test_common.py`

Added:
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_avg_symmetric_plus_axial_tt_layer_00_04.py`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_avg_symmetric_plus_axial_tt_layer_00_04.sbatch`

### Why
Adds a memory-conscious generalized train/test variant that uses the symmetric
average center pair plus four symmetric-average axial neighbors, restricted to
layers 0-4 with the usual per-layer checkpointing and logging.

## Session: 2026-03-26 — Simplify cluster guidance to Delta only

### What changed
Updated:
- `AGENTS.md`
- `.claude/CLAUDE.md`
- `branch-analysis.md`

### Why
Removed obsolete ICE guidance and added cluster-script rules for per-layer output checkpointing and detailed stdout time logging.

## Session: 2026-03-26 — Add dedicated generalized-model train/test regressions for diagonal and average-symmetric variants

### What changed
Updated:
- `branch-analysis.md`

Added one shared helper:
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_train_test_common.py`

Added two train/test Python launchers:
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_plus_diagonal_train_test.py`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_avg_symmetric_train_test.py`

Added two matching Delta sbatch files:
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_plus_diagonal_train_test.sbatch`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_avg_symmetric_train_test.sbatch`

### Why
You wanted a permanent generalized-model train/test workflow that stays separate
from the strict K-fold suite, preserves the same regression/geometry outputs,
and supports new feature variants without having to retrofit the existing
K-fold helper into a dual-mode script.

### Key design decisions
- Copied the generalized-model K-fold helper structure into a standalone
  train/test helper instead of importing the K-fold runner, so future split
  changes cannot accidentally alter the K-fold behavior.
- Switched the outer split policy to one deterministic protein-level
  `StratifiedShuffleSplit` with:
  - `test_size=0.2`
  - `random_state=42`
  - strict failure when all five size bins cannot be represented in both train
    and test
- Renamed fold-oriented outputs to split-oriented outputs and replaced `fold`
  metadata with `split_id=0` while keeping the same coefficient, per-protein,
  and geometry statistics.
- Kept checkpoint flushing after every completed layer and timestamped progress
  messages throughout, so partial outputs survive timeouts or cluster failures.
- Added exactly the two requested feature variants:
  - directional `(i,j)+(j,i)` plus long diagonals `(i,i)+(j,j)` -> 512 features
  - memory-conscious symmetric average of `(i,j)` and `(j,i)` -> 128 features
- Used Delta sbatch settings tailored to the expected memory/runtime footprint:
  - diagonal variant: `8` CPUs, `128g`, `12:00:00`
  - avg-symmetric variant: `4` CPUs, `64g`, `10:00:00`
  - shared output root: `$HOME/general_train_test_split_csv_files`
- Follow-up performance tuning for the train/test helper:
  - switched the hot layer-cache path from `np.load(..., mmap_mode="r")` back
    to plain `np.load(...)` to match the older fast cluster scripts and avoid
    repeated random-access page faults during advanced indexing
  - rewrote diagonal/symmetric feature assembly to fill one preallocated output
    array directly instead of building intermediate concatenation pieces
  - added layer-level timing breakdowns for regression, geometry, and
    checkpoint writing so slow future runs are easier to diagnose from logs

## Session: 2026-03-25 — Add local aggregate analysis for baseline256 generalized-model outputs

### What changed
Updated:
- `branch-analysis.md`

Added one local analysis script:
- `scripts/multi_protein/regression/analyze_delta_baseline256_aggregate.py`

Generated aggregate-analysis outputs under:
- `delta_files/baseline256/analysis/`

### Why
The merged `delta_files/baseline256/` CSVs now cover all 48 layers of the
generalized baseline256 run, and you wanted a fixed local post-processing pass
that turns those aggregates into reusable summary statistics, cross-fold
coefficient stability measures, a Markdown report, and a small set of plots
without adding any new CLI surface.

### Key design decisions
- Kept the workflow repo-relative and non-interactive:
  - no CLI arguments
  - fixed input directory `delta_files/baseline256/`
  - fixed output directory `delta_files/baseline256/analysis/`
- Used the merged aggregate CSVs directly rather than recomputing any regression
  fits, with only lightweight validation for required columns, full layer/fold
  coverage, coefficient-vector completeness, and duplicate plotting keys.
- Defined protein-level R² trajectories as the mean of each protein's five
  fold-specific `per_protein_r2` values at a given layer, after combining train
  and test membership into one row per `(fold, layer, protein)` because each
  protein appears exactly once per fold.
- Restricted every cosine analysis to raw-scale coefficients with the intercept
  removed:
  - same-layer pairwise cosine similarity across the five folds
  - 48x48 layer-by-layer cosine matrix for fold 4
- Wrote both human-readable and machine-readable outputs:
  - Markdown report
  - summary statistics CSV
  - same-layer fold-cosine CSV
  - last-fold layer-cosine matrix CSV
  - three PNG plots covering layer trajectories, fold-4 layer similarity, and
    layer-47 size trends

## Session: 2026-03-25 — Add temporary single-split layer shards under temp_files

### What changed
Updated:
- `branch-analysis.md`

Added one temporary shared helper:
- `scripts/cluster_scripts/temp_files/context_feature_train_test_common.py`

Added ten temporary baseline256 train/test shard launchers plus ten matching
sbatch files under `scripts/cluster_scripts/temp_files/`.

Added ten temporary baseline256-plus-axial train/test shard launchers plus ten
matching sbatch files under `scripts/cluster_scripts/temp_files/`.

### Why
The strict 5-fold generalized-model jobs were still too slow and too variable to
be practical on the current Delta resource budget. You wanted an isolated
temporary fallback that keeps the same generalized feature construction and
geometry/statistics flow, but drops the 5-fold outer loop in favor of one
deterministic train/test protein split and still shards the work by 5-layer
windows.

### Key design decisions
- Kept the temporary path entirely inside `scripts/cluster_scripts/temp_files/`
  so the main generalized-model helper and launchers remain untouched.
- Reused the stable generalized-model numeric and feature-cache helpers from
  `context_feature_kfold_common.py` instead of copying the full regression
  stack again.
- Borrowed the deterministic size-stratified-when-feasible 80/20 split policy
  from `diagnostic_context_feature_sweep_regression.py`.
- Wrote temporary outputs under
  `CSV_ROOT/temp_context_feature_train_test/<variant_name>/` so they cannot
  collide with the main k-fold artifacts while still staying under the same
  configured root.
- Used single-split CSV schemas that mirror the generalized outputs but drop the
  `fold` column:
  - `split_summary`
  - `per_protein`
  - `coefficients`
  - `split_assignments`
  - `geometry_metrics`
  - `geometry_summary`
- Kept geometry on both train and test proteins for every requested layer,
  since there is no longer a “last fold” concept.
- Sized the temporary shard jobs to the current cluster preference:
  - baseline256 shards: `8` CPUs, `64g`, `03:00:00`
  - baseline256_plus_axial shards: `8` CPUs, `128g`, `03:00:00`

## Session: 2026-03-25 — Retune all split-layer generalized-model sbatch jobs to 8 CPUs / 3 hours

### What changed
Updated:
- `branch-analysis.md`
- all baseline256 split-layer sbatch files in `scripts/cluster_scripts/generalized_model/`
- all baseline256-plus-axial split-layer sbatch files in
  `scripts/cluster_scripts/generalized_model/splitting_layers/`

### Why
Early Delta runs showed large wall-clock variability across layer shards, with
some split jobs spending far longer than expected inside the first fold. You
wanted every layer-split sbatch launcher moved to a consistent `8` CPUs and
`03:00:00` walltime to give the runs more headroom while keeping the split-job
shape unchanged.

### Key design decisions
- Updated only the split-layer sbatch launchers; the full-range generalized
  sbatch files were left alone.
- Standardized both split families on the same CPU count (`8`) and the same
  walltime (`03:00:00`) so comparisons across baseline and axial shards stay
  simpler.
- Left each split family's memory request unchanged:
  - baseline256 split jobs stay at `64g`
  - baseline256-plus-axial split jobs stay at `128g`

## Session: 2026-03-25 — Add layer-split baseline256-plus-axial launchers under splitting_layers

### What changed
Updated:
- `branch-analysis.md`

Added ten baseline256-plus-axial Python launchers under:
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_00_04.py`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_05_09.py`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_10_14.py`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_15_19.py`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_20_24.py`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_25_29.py`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_30_34.py`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_35_39.py`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_40_44.py`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_45_47.py`

Added ten matching Delta sbatch files under:
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_00_04.sbatch`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_05_09.sbatch`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_10_14.sbatch`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_15_19.sbatch`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_20_24.sbatch`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_25_29.sbatch`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_30_34.sbatch`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_35_39.sbatch`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_40_44.sbatch`
- `scripts/cluster_scripts/generalized_model/splitting_layers/context_feature_kfold_baseline256_plus_axial_layers_45_47.sbatch`

### Why
The full `baseline256_plus_axial` generalized-model run is even heavier than the
plain baseline split, so you wanted the same layer-window sharding pattern for
the axial variant while keeping the new files isolated inside
`generalized_model/splitting_layers/`.

### Key design decisions
- Reused the shared layer-window and filename-suffix support already added to
  `context_feature_kfold_common.py`, so the new axial shards keep the exact same
  regression logic, fold construction, geometry path, and output schemas as the
  existing full-range launcher.
- Left the original full-range `context_feature_kfold_baseline256_plus_axial.py`
  and `.sbatch` files untouched as a fallback.
- Matched the baseline split output convention:
  - write to `CSV_ROOT/baseline256_plus_axial/`
  - keep `variant=baseline256_plus_axial`
  - encode layer windows only in filenames such as `fold_summary_0-4.csv`
  - `coefficients_25-29.csv`, and `summary_45-47.txt`, which keeps later CSV
    merging simple while avoiding extra output-directory branching.
- Placed the new files in `generalized_model/splitting_layers/`, so each shard
  wrapper adds the parent `generalized_model` directory to `sys.path` before
  importing `context_feature_kfold_common`.
- Sized each axial shard job more generously than the plain baseline shards:
  `8` CPUs, `128g`, and `02:00:00`.

## Session: 2026-03-25 — Split generalized baseline256 Delta job into layer-window launchers

### What changed
Updated:
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_common.py`
- `branch-analysis.md`

Added ten baseline-only Python launchers:
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_00_04.py`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_05_09.py`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_10_14.py`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_15_19.py`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_20_24.py`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_25_29.py`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_30_34.py`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_35_39.py`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_40_44.py`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_45_47.py`

Added ten matching Delta sbatch files:
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_00_04.sbatch`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_05_09.sbatch`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_10_14.sbatch`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_15_19.sbatch`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_20_24.sbatch`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_25_29.sbatch`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_30_34.sbatch`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_35_39.sbatch`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_40_44.sbatch`
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_baseline256_layers_45_47.sbatch`

### Why
The full 48-layer `baseline256` generalized-model job is too slow for one Delta
submission window. You wanted that same regression procedure split into smaller
layer-wise jobs so each run can finish in about two hours, produce mergeable CSV
artifacts, and preserve the current output conventions closely enough for later
aggregation on your local machine.

### Key design decisions
- Kept the existing unsplit `context_feature_kfold_baseline256.py` launcher as a
  fallback and added layer-window launchers alongside it, so the full job path
  remains available.
- Added optional `layer_indices` and `output_filename_suffix` plumbing to the
  shared generalized runner rather than copying the regression logic into ten
  near-duplicate scripts.
- Kept split outputs under the existing `CSV_ROOT/<variant_name>/` directory and
  encoded the layer window in filenames only, e.g. `fold_summary_0-4.csv`,
  `coefficients_25-29.csv`, and `summary_45-47.txt`, which keeps later CSV
  merging simple while avoiding extra output-directory branching.
- Preserved the `variant` field as `baseline256` in every split artifact so the
  layer shards can be concatenated directly without post-hoc renaming.
- Fixed the generalized-model helper import path by adding the parent
  `scripts/cluster_scripts` directory to `sys.path`, because
  `cluster_regression_helpers.py` lives one level above `generalized_model`.
- Sized each new Delta shard job for the requested lighter footprint:
  `4` CPUs, `64g`, and `02:00:00`.

## Session: 2026-03-24 — Refactor generalized context-feature k-fold runner toward stacked sklearn fitting

### What changed
Updated:
- `scripts/cluster_scripts/generalized_model/context_feature_kfold_common.py`
- `branch-analysis.md`

### Why
The generalized context-feature k-fold jobs were timing out because the shared
runner still used a streamed normal-equations fit path with repeated float64
materialization and repeated per-protein standardization passes. You wanted the
runtime shape moved closer to the older `cluster_multi_protein_regression.py`
workflow while keeping the newer semantics: strict size-stratified folds,
protein-equal weighting, float64 fitting, current output files, and last-fold
geometry on every layer.

### Key design decisions
- Kept the existing `layer -> fold` execution order and shared generalized CSV
  schemas so all four fixed-variant entrypoints continue to behave the same from
  the outside.
- Switched the per-layer feature-cache path from `mmap_mode="r"` gathers to
  plain `np.load(...)`, building one dense float32 feature chunk per protein and
  freeing the raw pair block immediately after extraction.
- Replaced streamed normal-equations fitting with stacked `LinearRegression`
  over one float64 train matrix per fold, using per-row `sample_weight` values
  that give equal total mass to each protein.
- Kept the float64 boundary convention from the precision diagnostics:
  - cached raw feature chunks stay float32
  - scaler statistics are accumulated in float64
  - train/test matrices, targets, weights, and predictions are materialized in
    float64 only at fit / eval time
- Removed repeated per-protein re-standardization during train metrics, test
  metrics, and last-fold geometry by reusing stacked split predictions and
  stable row offsets instead.
- Preserved last-fold-only geometry across all 48 layers and kept per-layer
  checkpoint writes, since the outputs are useful even when long cluster jobs
  stop early.
- Expanded timestamped progress logging so cluster `slurm` output now clearly
  marks cache-build start/end, per-fold fit/eval start, and per-layer / total
  elapsed time.

## Session: 2026-03-24 — Add small Delta experiment for 256d center plus lower+upper axial context

### What changed
Added one new small-experiment Delta regression script:
- `scripts/cluster_scripts/small_experiments/context_feature_sweep_256_plus_axial_both_triangles_regression.py`

Added one matching Delta sbatch launcher:
- `scripts/cluster_scripts/small_experiments/context_feature_sweep_256_plus_axial_both_triangles_regression.sbatch`

Updated:
- `branch-analysis.md`

### Why
You wanted a focused follow-up to the `baseline256_plus_axial` context-sweep
variant that keeps the same 256d center pair features but augments them with
axial neighbors from both triangles instead of only the lower triangle.

### Key design decisions
- Kept the experiment in `small_experiments/` as a single-purpose script rather
  than expanding the larger diagnostic sweep again.
- Reused the existing context-sweep diagnostic for:
  - protein discovery
  - PDB loading
  - deterministic train/test splitting
  - streamed float64 fitting
  - metric / coefficient export
- Overrode only the feature-construction path so each lower-triangle training
  row now concatenates:
  - `pair(i, j)`
  - `pair(j, i)`
  - 4 axial lower-triangle neighbors around `(i, j)`
  - 4 axial upper-triangle neighbors around the mirrored `(j, i)`
- Used explicit `lower_context_*` and `upper_context_*` coefficient names so
  the exported weights make it obvious which triangle each axial block came
  from.
- Left the Delta memory request at `192g` because this small experiment only
  fits one streamed variant, and the larger parent diagnostic already ran with
  that allocation after its OOM-driven increase.

## Session: 2026-03-24 — Add main cluster scripts for strict 5-fold context-feature runs

### What changed
Added one shared main-module helper:
- `scripts/cluster_scripts/context_feature_kfold_common.py`

Added four standalone main cluster scripts:
- `scripts/cluster_scripts/context_feature_kfold_baseline256.py`
- `scripts/cluster_scripts/context_feature_kfold_baseline256_plus_axial.py`
- `scripts/cluster_scripts/context_feature_kfold_baseline256_plus_diagonal.py`
- `scripts/cluster_scripts/context_feature_kfold_baseline256_plus_diagonal_plus_axial.py`

Added four matching Delta sbatch launchers:
- `scripts/cluster_scripts/context_feature_kfold_baseline256.sbatch`
- `scripts/cluster_scripts/context_feature_kfold_baseline256_plus_axial.sbatch`
- `scripts/cluster_scripts/context_feature_kfold_baseline256_plus_diagonal.sbatch`
- `scripts/cluster_scripts/context_feature_kfold_baseline256_plus_diagonal_plus_axial.sbatch`

### Why
You wanted the previously planned context-feature ablations promoted out of the
diagnostic folder into main cluster scripts, with one fixed variant per file,
strict size-stratified outer 5-fold CV, all 48 layers, and detailed last-fold
geometry exports for both train and test proteins.

### Key design decisions
- Kept the new jobs in `scripts/cluster_scripts/` instead of `diagnostics/` so
  they read like production cluster entrypoints rather than one-off probes.
- Split the work into four thin entrypoints over one shared helper module so:
  - the fold logic, feature construction, and CSV schemas stay identical
  - each sbatch file points at exactly one fixed feature variant
- Matched the requested feature definitions to the reference context sweep:
  - `baseline256` uses `[pb[i,j,:], pb[j,i,:]]`
  - `+axial` appends four zero-padded lower-triangle axial context blocks
  - `+diagonal` appends the self-diagonal blocks `pb[i,i,:]` and `pb[j,j,:]`
- Used a `layer -> fold` execution order with a per-layer protein cache so each
  protein's feature block is loaded once per layer and then reused across all
  five folds.
- Preserved the protein-equal weighted linear objective from the reference sweep
  while switching from one held-out split to strict outer 5-fold CV.
- Made fold construction fail fast when strict 5-fold size stratification is
  infeasible, reporting the observed size-bin counts directly in the error.
- Restricted geometry reconstruction to the final outer fold while still
  evaluating all 48 layers and both train/test splits, keeping the requested
  detail without the much larger all-fold geometry cost.
- Wrote each variant's outputs under `CSV_ROOT/<variant_name>/` so the four jobs
  can share one root output directory without filename collisions.

## Session: 2026-03-23 — Add unified context-feature sweep diagnostic for held-out R² comparisons

### What changed
Added one new Delta/local-compatible diagnostic:
- `scripts/cluster_scripts/diagnostics/diagnostic_context_feature_sweep_regression.py`

Added one matching Delta sbatch launcher:
- `scripts/cluster_scripts/diagnostics/diagnostic_context_feature_sweep_regression.sbatch`

Updated:
- `branch-analysis.md`

### Why
You wanted one apples-to-apples sweep over context and feature-augmentation
variants after the float-precision issue was resolved, with a shared held-out
evaluation protocol and coefficient export for every final fitted model.

### Key design decisions
- Kept the sweep in one standalone diagnostic so all requested variants share:
  - one deterministic size-stratified protein split
  - one protein-equal weighting policy
  - one train-only standardization path
  - one primary leaderboard metric: test mean per-protein `R^2`
- Implemented the requested linear variants in one fixed in-code table instead
  of spreading them across several older diagnostics.
- Exported coefficients for every final fitted model, including:
  - all linear variants
  - the final CV-selected ridge baseline
- Added a compact ranked leaderboard CSV and text summary on top of the raw
  per-protein and coefficient tables, so the best variants are easy to inspect.
- Treated the float64 question as an in-memory materialization ablation for the
  `baseline256` feature family, since the stored pair-block `.npy` files are
  already `float32` on disk.
- Made pair-block and PDB path resolution work for both:
  - Delta cluster layout via the documented environment variables
  - the local sibling `../protein_data/` layout used for smoke testing
- After the first all-row implementation hit a Delta OOM before writing CSVs,
  replaced full `n_rows x d` matrix materialization with streamed weighted
  normal-equation fitting:
  - process one protein block at a time
  - keep only the current standardized block plus a `d x d` system in memory
  - preserve the same linear / ridge objectives and coefficient outputs
- Added incremental output flushing after each finished variant so partial CSVs
  still appear on disk even if a later variant or cluster job fails.
- Increased the sbatch memory request from `128g` to `192g` to add headroom for
  the largest per-protein feature blocks during the streamed run.

## Session: 2026-03-23 — Add repo-wide float64 reminder for dense regression diagnostics

### What changed
Updated local agent docs:
- `AGENTS.md`
- `.claude/CLAUDE.md`

Updated:
- `branch-analysis.md`

### Why
Recent precision diagnostics showed that large dense least-squares fits on
pair-block features can behave pathologically in `float32`, while `float64`
materially improves stability and downstream `R^2`.

### Key design decisions
- Added the reminder in both local instruction files so the precision lesson is
  easy to find regardless of which agent is editing the repo.
- Scoped the note to the operations where it matters most here:
  - dense linear regression
  - least-squares
  - PCA / SVD
  - related matrix-based diagnostics on pair-block features
- Kept `float32` available as an explicit opt-in for:
  - intentional precision-comparison experiments
  - memory/performance tradeoff studies

## Session: 2026-03-23 — Move local regression inputs to sibling protein_data root

### What changed
Updated local agent docs:
- `AGENTS.md`
- `.claude/CLAUDE.md`

Updated local multi-protein regression scripts:
- `scripts/multi_protein/regression/aa_augmented_regression.py`
- `scripts/multi_protein/regression/benchmark_kfold_mds_runtime.py`
- `scripts/multi_protein/regression/kfold_mds_regression.py`
- `scripts/multi_protein/regression/linear_regression_sequential_distance.py`
- `scripts/multi_protein/regression/logistic_regression_pair_type.py`
- `scripts/multi_protein/regression/logistic_regression_pair_type_with_diagonal.py`
- `scripts/multi_protein/regression/mds_reconstruction.py`
- `scripts/multi_protein/regression/multi_protein_regression.py`
- `scripts/multi_protein/regression/multi_protein_regression_with_diagonal.py`
- `scripts/multi_protein/regression/neighbor_context_pair_regression.py`
- `scripts/multi_protein/regression/regression_variants_predicted.py`
- `scripts/multi_protein/regression/symmetric_antisymmetric_pair_regression.py`

Updated:
- `branch-analysis.md`

### Why
The per-protein pair representations and predicted PDB inputs were moved out of
the repo workspace from `protein/proteins/` into the sibling directory
`../protein_data/`, but the local regression scripts were still defaulting to
the old input root.

### Key design decisions
- Split local input and output roots in every affected local regression script:
  - inputs now come from `PROJECT_ROOT.parent / "protein_data"`
  - aggregated outputs stay in `PROJECT_ROOT / "proteins" / "multi_protein_results"`
- Left `lib.data_loading` unchanged because its per-protein subdirectory
  expectations still match the new layout inside each `../protein_data/<id>/`
  directory.
- Preserved per-protein CSV writes relative to each discovered `protein_dir`, so
  those outputs continue to land beside the moved protein data.
- Updated the repo guidance docs so future agents know that local per-protein
  inputs live outside the workspace and are reached from the repo root via
  `../protein_data/`.

## Session: 2026-03-23 — Add local analysis script for precision / row-cap / PCA diagnostic

### What changed
Added one new local analysis script:
- `scripts/multi_protein/regression/analyze_delta_precision_rowcap_pca.py`

Updated:
- `branch-analysis.md`

### Why
You wanted a reproducible post-hoc analysis pass over the new
`diagnostic_precision_rowcap_pca` outputs that would summarize the main
conclusions, quantify residual size effects, and measure coefficient similarity
across the manual size buckets.

### Key design decisions
- Kept the work in a standalone local analysis script instead of extending the
  cluster diagnostic, so the original run outputs stay reproducible and the
  interpretability layer can evolve independently.
- Focused the report on the user-facing questions that motivated the diagnostic:
  - how much of the old failure was float precision
  - whether size effects remain after switching to float64
  - whether the manual bucket models learn meaningfully different coefficient
    vectors
- Exported both machine-readable tables and a markdown report:
  - key-variant overview
  - row-cap and PCA summaries
  - per-bucket test `R^2` aggregates
  - size-vs-`R^2` correlation summaries
  - bucketed coefficient cosine-similarity tables
  - top shifted channels between bucket models
- Compared bucket coefficients in both standardized and raw-scale spaces, while
  treating raw-scale comparisons as the main interpretive view because the
  bucketed models use different scalers.
- Added a small, focused visualization set rather than a large figure dump:
  - row-cap precision sweep
  - PCA sweep
  - test `R^2` by size bucket for key variants
  - bucketed-coefficient cosine heatmap
  - test size-vs-`R^2` scatter panels

## Session: 2026-03-22 — Add layer-47 precision / row-cap / PCA / bucketed diagnostic

### What changed
Added two new Delta diagnostic files:
- `scripts/cluster_scripts/diagnostics/diagnostic_precision_rowcap_pca_regression.py`
- `scripts/cluster_scripts/diagnostics/diagnostic_precision_rowcap_pca_regression.sbatch`

Updated:
- `branch-analysis.md`

### Why
You wanted one combined follow-up diagnostic that can separate row-count-driven
float32 instability from genuine protein-size heterogeneity, while also
exporting the coefficient vectors needed for later similarity analyses.

### Key design decisions
- Kept the work in one standalone layer-47 diagnostic so the comparison uses:
  - one deterministic protein-level size-stratified 80/20 split
  - one shared output family
  - one consistent per-protein evaluation policy across all variants
- Focused the global exact-control sweep on the requested matrix:
  - `128d` lower-triangle center features
  - `256d` paired lower-triangle features
  - `row_cap_per_train_protein in {5000, 20000, all}`
  - `float32` and `float64` for protein-equal fits
  - one extra `256d` float64 pair-equal all-row control
- Computed the singular-spectrum diagnostics from the weighted standardized
  `d x d` Gram matrix instead of a massive full-row SVD, so the script can
  report effective rank and float32 / float64 cutoff counts cheaply once the
  design matrix is materialized.
- Added a weighted train-row PCA ablation only for the `128d` float64
  protein-equal all-row baseline:
  - fit over `k in {1, 2, 5, 10, 20, 50, 100, 128}`
  - export both PC-space coefficients and back-projected raw-space
    coefficients for direct later similarity tests
- Added one manual fixed-bucket `128d` float64 protein-equal variant using the
  integer bucket boundaries implied by the earlier tree thresholds:
  - `<=255`
  - `256-371`
  - `372-500`
  - `501+`
- Exported coefficients for every fitted model, including:
  - every global raw linear fit
  - every PCA fit in PC space
  - every PCA back-projected raw-space coefficient vector
  - every bucket-specific linear model

## Session: 2026-03-22 — Add float-precision vs protein-size takeaway summary note

### What changed
Added one new analysis summary:
- `analysis_summaries/float_precision_size_takeaways.md`

Updated:
- `branch-analysis.md`

### Why
You wanted a durable repo note that captures the main conclusions from the
float32 vs float64 debugging discussion, especially the overlap between solver
precision issues and protein-size / dataset-heterogeneity effects.

### Key design decisions
- Kept the note focused on the most decision-relevant conclusions:
  - what is actually proven about float32-backed code paths
  - what remains a likely interpretation rather than a proof
  - which next analyses would best separate competing explanations
- Explicitly distinguished pooled pair-level scripts from later
  protein-level diagnostics, since that difference matters for interpreting
  the older 25-protein versus 184-protein results.
- Recorded both sides of the current picture instead of forcing a single-cause
  story:
  - float32 least-squares appears to become much less reliable as row count
    grows
  - size-aware weighting / bucketing still improves results, which points to a
    real heterogeneity problem beyond precision alone

## Session: 2026-03-17 — Add fast 128d float32 vs float64 linear-regression check

### What changed
Added one new standalone diagnostic script:
- `scripts/cluster_scripts/diagnostics/diagnostic_float_precision_128d_regression.py`
- `scripts/cluster_scripts/diagnostics/diagnostic_float_precision_128d_regression.sbatch`

Updated:
- `branch-analysis.md`

### Why
You wanted a very small follow-up script to isolate whether solver precision
alone changes the layer-47 128d linear-regression result, without the larger
ridge-stability diagnostic machinery.

### Key design decisions
- Kept the script narrowly scoped to one feature family:
  - lower-triangle 128d center features `pb[i,j,:]`
- Preserved the same core training setup for both runs:
  - one protein-level 80/20 split with size-bin stratification when feasible
  - protein-equal training weights
  - train-only protein-equal feature standardization
  - plain `LinearRegression`
- Ran the fit twice with identical data and split:
  - once with `float32` feature / target arrays
  - once with `float64` feature / target arrays
- Limited output to exactly what was needed for the check:
  - test mean per-protein `R^2` printed to stdout
  - a tiny CSV with one row per dtype

## Session: 2026-03-17 — Add ridge-stability diagnostic and Delta sbatch for exact 128d/256d controls

### What changed
Added two new Delta diagnostic files:
- `scripts/cluster_scripts/diagnostics/diagnostic_ridge_stability_regression.py`
- `scripts/cluster_scripts/diagnostics/diagnostic_ridge_stability_regression.sbatch`

Updated:
- `branch-analysis.md`

### Why
You wanted a dedicated follow-up to the protein-fairness exact-control work to
explain why a tiny ridge penalty (`alpha=1e-8`) can move the 256d paired-feature
baseline so far from the no-ridge solution, while also testing whether ridge or
lasso tuning can improve `R^2`.

### Key design decisions
- Kept the new work in a standalone diagnostic instead of modifying
  `diagnostic_protein_fairness_regression.py`, so the earlier fairness run
  remains reproducible and the ridge-stability analysis has its own output
  family.
- Preserved the residue-exact baseline setup as the default outer evaluation:
  - lower-triangle unordered residue pairs only
  - protein-level 80/20 split with size-bin stratification when feasible
  - protein-equal training weights
  - train-only protein-equal feature standardization
  - absolute-distance targets
- Added both feature families side by side:
  - `128d`: center-only `pb[i,j,:]`
  - `256d`: paired `[pb[i,j,:], pb[j,i,:]]`
- Ran the main instability comparison with matched linear and ridge fits for
  both feature sets, then extended that with two intercept ablations:
  - `fit_intercept=True`
  - `fit_intercept=False`
  - centered-target / no-intercept fits
- Added weighted design-spectrum diagnostics from the standardized train design
  to expose singular-value structure, effective rank, and near-null directions
  for each feature set and split family.
- Added explicit symmetry diagnostics for the 256d design:
  - weighted train correlations between matched `ij` / `ji` channels
  - symmetric / antisymmetric coefficient decompositions for every 256d fit
- Added ridge-vs-linear comparison rows that summarize coefficient cosine
  similarity, intercept shifts, coefficient-difference norms, and per-protein
  bias deltas on the test split.
- Added a small performance-tuning section using inner 5-fold protein-level CV
  on the outer-train proteins:
  - ridge alpha sweep across `1e-12` to `100`
  - lasso alpha sweep across `1e-6` to `3e-2`
  - alpha chosen by validation mean per-protein `R^2`
- Added a limited split-family ablation for `ridge(alpha=1e-8)` only, comparing
  the default stratified outer split against a plain random protein split.
- Sized the new Delta sbatch above the fairness diagnostic (`8` CPUs, `128g`,
  `08:00:00`) because this script now performs multiple outer fits plus nested
  ridge and lasso CV sweeps.

## Session: 2026-03-16 — Add fixed-threshold all-layer piecewise linear context regression for Delta

### What changed
Added two new Delta cluster files:
- `scripts/cluster_scripts/piecewise_linear_regression.py`
- `scripts/cluster_scripts/piecewise_linear_regression.sbatch`

Updated:
- `branch-analysis.md`

### Why
You wanted a standalone all-layer follow-up to the layer-47 context linear-tree
diagnostic that keeps the same data-loading, context-feature, split, weighting,
and geometry-analysis pipeline, but replaces learned tree splits with fixed
leaf regions derived from the selected layer-47 thresholds.

### Key design decisions
- Kept the work in a new standalone cluster script instead of modifying the
  existing context linear-tree diagnostic, so the original tree run remains
  reproducible and the fixed-threshold comparison has its own output family.
- Reused the same two context feature variants from the reference diagnostic:
  - lower-triangle center only
  - lower-triangle center plus axial context
- Preserved the reference train/test setup:
  - one deterministic protein-level 80/20 split with `seed=42`
  - protein-equal pair weights with each protein summing to `1.0`
  - z-scoring of all non-`n_residues` features using train-only moments
- Replaced the learned depth-2 tree with four fixed residue-count leaf models
  per variant and per layer:
  - `LL`, `LR`, `RL`, `RR`
  - routing uses only raw `n_residues`
  - each leaf fits its own ordinary `LinearRegression`
- Hardcoded the selected layer-47 thresholds into an exported rules CSV so the
  exact routing policy is explicit and easy to analyze later.
- Extended the diagnostic from layer 47 only to all 48 AlphaFold layers while
  keeping per-protein geometry exports and adding region-mass diagnostics for
  both train and test.
- Added explicit start/end timestamps and elapsed-time prints after every layer
  so cluster logs make layer runtimes easy to monitor during long runs.

## Session: 2026-03-26 — Add explicit resume-from-layer support for avg-symmetric train/test

### What changed
Updated:
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_train_test_common.py`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_avg_symmetric_train_test.py`
- `branch-analysis.md`

### Why
The original avg-symmetric train/test job saved layers incrementally before timing
out. You already have canonical outputs through layer 33 and needed the rerun to
continue from layer 34 while preserving and extending the existing CSV and
summary files.

### Key design decisions
- Implemented the resume logic in the shared train/test helper rather than in
  the wrapper so the existing checkpoint/write flow stays in one place.
- Kept the resume path narrow and explicit:
  - wrapper now requests `resume_from_layer=34`
  - helper validates that canonical outputs exist and cover exactly layers `0-33`
  - helper refuses to resume if saved files already contain layer `34+` or if the
    saved deterministic split assignments no longer match the rebuilt split
- Reused the canonical output filenames rather than writing temp suffix files,
  and skipped the initial empty-output flush when resuming so earlier results are
  not touched before validation passes.
- Preserved cumulative elapsed time by parsing the prior `summary.txt` elapsed
  value and adding the new resumed-run time during later checkpoints.

## Session: 2026-03-26 — Resume diagonal train/test from layer 16

### What changed
Updated:
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_plus_diagonal_train_test.py`
- `branch-analysis.md`

### Why
The diagonal train/test run completed through layer 15 before timing out. You
needed the rerun to preserve the canonical outputs already written for layers
`0-15` and continue from layer 16 through layer 47.

### Key design decisions
- Reused the existing shared helper resume path instead of adding diagonal-
  specific merge logic.
- Kept the wrapper change narrow and temporary:
  - requests all 48 layers
  - hard-codes `resume_from_layer=16`
- Left the Delta sbatch resource request unchanged at 12 hours, 8 CPUs, and
  128 GB because that already matches the requested runtime envelope.
- The prior hot-path speed fix already applies to this diagonal run because it
  uses the same shared helper:
  - plain `np.load(...)` in the layer cache path
  - direct preallocated feature writes instead of repeated concatenation

## Session: 2026-04-01 — Verify resume neutrality and remove resume-from-layer support

### What changed
Updated:
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_train_test_common.py`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_avg_symmetric_train_test.py`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_plus_diagonal_train_test.py`
- `branch-analysis.md`

### Why
The helper already supported arbitrary layer subsets through `layer_indices`, so
the separate `resume_from_layer` path was unnecessary extra orchestration. Before
removing it, we needed to verify that prior resumed runs were numerically
consistent with direct subset runs for the same resumed layers.

### Key design decisions
- Verified resume neutrality before removal using a controlled synthetic setup
  for both full-run variants:
  - `general_base256_avg_symmetric`
  - `general_base256_plus_diagonal`
- Compared resumed layers against direct subset runs for:
  - `split_summary.csv` ignoring only `elapsed_seconds`
  - `per_protein.csv`
  - `coefficients.csv`
  - `geometry_metrics.csv`
  - `geometry_summary.csv`
- The resumed-layer CSV rows matched exactly in the synthetic comparison, which
  supports treating previous resumed outputs as valid.
- Removed the resume-specific helper orchestration entirely:
  - no CSV reload/validation helpers
  - no prior-summary elapsed parsing
  - no `resume_from_layer` argument on `run_variant_job(...)`
- Kept `layer_indices` and `output_filename_suffix` as the long-term mechanism
  for partial-layer train/test runs, including the existing axial layer-split
  wrappers.
- Simplified the two full-run wrappers back to plain all-layer calls without
  explicit resume constants.
Adds a fold-4 post-processing script that reuses the saved K=3 layer
segmentation to summarize 256 coefficient channels by segment and write
interpretation-focused CSVs plus plots.

## Session: 2026-03-27 — Narrow generalized geometry resume coercion

### What changed
Updated:
- `branch-analysis.md`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_train_test_common.py`
- `scripts/multi_protein/regression/backfill_general_base256_plus_diagonal_geometry_summary.py`
- `scripts/multi_protein/regression/backfill_general_base256_avg_symmetric_geometry_summary.py`

### Why
Shrinks the earlier resume fix to one geometry-only coercion path instead of
typing every resumed CSV table.

## Session: 2026-03-27 — Backfill general avg-symmetric geometry summary

### What changed
Updated:
- `branch-analysis.md`

Added:
- `scripts/multi_protein/regression/backfill_general_base256_avg_symmetric_geometry_summary.py`

Generated:
- `delta_files/general_base256_avg_symmetric/geometry_summary.csv`

### Why
Repairs the already-saved `general_base256_avg_symmetric/geometry_summary.csv`
after confirming the same resume-string aggregation bug left the early layers as
`NaN`.

## Session: 2026-03-27 — Add fold-4 layer segmentation analysis for baseline256_kfold

### What changed
Updated:
- `branch-analysis.md`

Added:
- `scripts/multi_protein/regression/analyze_delta_baseline256_kfold_segmentation.py`

### Why
Adds a fixed local post-processing script that finds contiguous layer
segments from the fold-4 coefficient vectors, writes boundary/evidence CSVs,
and overlays the K=3 boundaries on the fold-4 cosine heatmap.

## Session: 2026-03-27 — Fix resumed generalized geometry summary backfill

### What changed
Updated:
- `branch-analysis.md`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_train_test_common.py`

Added:
- `scripts/multi_protein/regression/backfill_general_base256_plus_diagonal_geometry_summary.py`

Generated:
- `delta_files/general_base256_plus_diagonal/geometry_summary.csv`

### Why
Retypes resumed CSV rows before aggregation so resumed train/test jobs no longer
emit `NaN` geometry summaries, and backfills the already-broken
`general_base256_plus_diagonal/geometry_summary.csv` from the saved per-protein
geometry metrics.

## Session: 2026-03-27 — Add one combined base256 variant analysis workflow

### What changed
Updated:
- `branch-analysis.md`

Added:
- `scripts/multi_protein/regression/analyze_delta_base256_variants.py`

### Why
Adds one fixed local post-processing script that analyzes `baseline256_kfold`
plus the three `general_base256*` train/test variants, writes per-variant
analysis outputs, and produces shared comparison plots under
`delta_files/comparing_variants/`.

## Session: 2026-04-29 — Add Chimera image strip builder

Added `chimera_files/make_chimera_image_strip.py` for composing Chimera PNG
exports into a labeled horizontal strip.

Follow-up:
- switched the comparison outputs to held-out/test-only aggregates so the
  train/test variant curves line up with each variant `summary.txt`
- added `delta_files/publication_plots/create_publication_plots.py` to
  render paper-oriented held-out R2/TM-score panels in
  `delta_files/publication_plots/`.

## Session: 2026-06-08 — Ignore oversized local CSV outputs

- Added exact `.gitignore` entries for tracked CSV outputs >=25 MB and removed them from the git index.

## Session: 2026-06-16 — Aggregate per-protein base256 combined shards

- Added a streaming merge script for `delta_files/per_protein_base256_combined`
  layer shards and generated canonical combined CSVs plus `summary.txt`.

## Session: 2026-03-27 — Aggregate generalized avg-symmetric axial train/test shards

### What changed
Updated:
- `branch-analysis.md`

Added:
- `scripts/multi_protein/regression/aggregate_general_avg_symmetric_plus_axial_shards.py`

Generated:
- `delta_files/general_base256_avg_symmetric_plus_axial/coefficients.csv`
- `delta_files/general_base256_avg_symmetric_plus_axial/geometry_metrics.csv`
- `delta_files/general_base256_avg_symmetric_plus_axial/geometry_summary.csv`
- `delta_files/general_base256_avg_symmetric_plus_axial/per_protein.csv`
- `delta_files/general_base256_avg_symmetric_plus_axial/split_assignments.csv`
- `delta_files/general_base256_avg_symmetric_plus_axial/split_summary.csv`
- `delta_files/general_base256_avg_symmetric_plus_axial/summary.txt`

### Why
Adds a checked local merge step so the three layer shards produce the same
single-file aggregate statistics layout as the other generalized train/test
variants.

## Session: 2026-03-26 — Add remaining generalized avg-symmetric axial train/test shards

### What changed
Updated:
- `branch-analysis.md`

Added:
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_avg_symmetric_plus_axial_tt_layer_05_25.py`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_avg_symmetric_plus_axial_tt_layer_05_25.sbatch`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_avg_symmetric_plus_axial_tt_layer_26_47.py`
- `scripts/cluster_scripts/generalized_model/mar25_train_test_split_regressions/general_base256_avg_symmetric_plus_axial_tt_layer_26_47.sbatch`

### Why
## Session: 2026-06-16 — Plot sampled triangle violation diagnostics

Added a reproducible plotting script for `general_base256_combined` sampled
triangle violations, including thresholded summaries and sample-vs-aggregate
comparison artifacts under the variant analysis directory.

## Session: 2026-06-18 — Use exact triangle violation metrics

Updated geometry helpers and CSV schemas to report exact unordered-triangle
violation fractions plus severity-bin proportions.

## Session: 2026-06-19 — Publication R²/TM layer grid

Added `delta_files/publication_plots/june19_plots/make_r2_tm_layer_grid.py`
(2x3 grid: general vs per-protein base256 R²/TM-score across layers, mean +
individual-protein background, y∈[0,1]). Outputs `r2_tm_layer_grid.{png,pdf}`.

## Session: 2026-06-19 — Triangle-violation 2x4 grid

Added `delta_files/publication_plots/june19_plots/make_triangle_violation_grid.py`
(2x4: left 2x2 general held-out, right 2x2 per-protein; exact triangle violation
total + 0–5/5–10/10+ Å bins across layers, mean + individual background,
per-metric shared y-scales). Outputs `triangle_violation_grid.{png,pdf}`.

## Session: 2026-07-03 — June 19 combined geometry CSV

Added one CSV combining generalized held-out and per-protein layer summaries used
by the June 19 publication plots.

## Session: 2026-08-03 — Correct XGBoost search evaluation

Fixed per-protein XGBoost row weighting, added train R², reduced the shared cap
sweep to 5k/10k/20k, and made winner outputs regression-only.

## Session: 2026-07-04 — Per-protein held-out pair R²

Updated per-protein base256 runs to fit train pairs only, save train/test R²,
add a Slurm layer array, and plot held-out pair scores.

## Session: 2026-08-02 — Combined XGBoost searches

Added shared and per-protein XGBoost search/winner arrays for anchor layers,
with deterministic splits, cap sweeps, validation-only selection, and MDS/TM outputs.

## Session: 2026-08-07 — Layer-47 salt-bridge classification probe

Added a two-hour CAMEO-50 probe with matched controls, protein-level splits,
regularized logistic feature ablations, and matched/natural-pool evaluation.
Revised the salt-bridge probe to ignore failed DSSP/SASA annotations entirely.
Generalized the probe to one configurable layer and made its sbatch run all 48
zero-indexed layers (0–47) as independent array tasks with isolated outputs.

## Session: 2026-08-07 — Salt-bridge layer trajectories

Added reproducible 48-layer metric/hyperparameter plots, aggregate CSVs, and a
report recommending a compact cross-probe regularization grid.

## Session: 2026-08-08 — Layer-47 coefficient clustering

Added reproducible k-means diagnostics and metadata-confound checks for the
per-protein layer-47 standardized coefficient profiles.

## Session: 2026-08-13 — Sequence-separation-gated distance probes

Added a six-layer array comparing sequence-separation-gated base256 probes with
a sequence-separation-only baseline using 100 training proteins.

## Session: 2026-08-14 — Gated-probe aggregate metrics

Added whole-protein and whole-layer summaries that correctly recombine the six
sequence-separation gates from additive error and target moments, plus readable
layer-by-separation metric matrices.

## Session: 2026-08-14 — Gated linear distance probe suite

Added reproducible sequence-, AA-pair-, and joint-gated float64 OLS probes,
true-distance statistics, restartable Delta arrays, strict aggregation, and a
separate deterministic smoke workflow.

## Session: 2026-08-14 — Remove redundant gated-probe rank SVD

- Reused scikit-learn's dense OLS rank/singular diagnostics instead of running
  a second full matrix-rank SVD, and tightened dense materialization, subgroup
  slicing, timing logs, and resume-manifest validation without changing outputs.
- Relaxed audit replay tolerance to `1e-9` for harmless summation-order drift.
- Made audit replay scale-aware at nine significant digits for extreme negative
  R-squared values.
- Kept consolidated production outputs under `$CSV_ROOT` for flat Delta runs.
- Made the aggregator derive its destination directly from its working root.

## Session: 2026-08-14 — Gated-probe visualization analysis

Added local trajectory, coefficient-cosine, and whole-protein comparison plots
for consolidated A1/A2/A4 results and the general base256 probe.

## Session: 2026-08-14 — Sequence-gated secondary-structure replay

Added a sparse six-layer replay of the sequence-gated coefficients for only
helix offsets and beta-sheet controls, reusing frozen DSSP annotations so the
parallel Delta workflow targets a sub-25-minute elapsed runtime.

## Session: 2026-08-15 — Recover XGBoost history

Restored the final corrected XGBoost search/diagnostic code and the saved
layer-0/23/47 search artifacts and logs from the archive branch.

## Session: 2026-08-15 — First-draft publication trajectories

Added reproducible R², TM-score, short-range MAE, MDS top-3 fraction, and
helix AUROC/recall figures with companion plot-data CSVs.

## Session: 2026-08-16 — Initial repository cleanup

Removed explicitly retired local/cluster code and generated clutter, archived
the MDS reference renders, and replaced stale cluster PDFs with live Delta docs.

## Session: 2026-08-16 — Remove legacy multi-protein outputs

Removed the retired result tree and obsolete local analysis scripts, while
keeping the MDS-to-PDB exporter and routing active local outputs to `delta_files/`.

## Session: 2026-08-16 — Archive miscellaneous workflows

Archived old generalized-model scripts and supporting results, moved plans into
`docs/`, and removed redundant test outputs and empty legacy directories.

## Session: 2026-08-16 — Regression methodology standard

Added a canonical paper protocol for shared, per-protein, gated, and XGBoost
distance probes plus a code-and-results audit of current methodology gaps.

## Session: 2026-08-17 — Refine regression protocol

Replaced mandatory 48-layer cohort prevalidation with stable runtime-checked
membership and allowed clearly labeled all-pair per-protein summaries.

## Session: 2026-08-17 — Regression output standard

Defined compact common regression, subgroup, geometry, model-artifact, and
runtime outputs and audited the current principal result families.

## Session: 2026-08-17 — Simplify regression output contract

Dropped per-protein sequence-bin and uncertainty outputs, added global plus
conditional baselines, and limited timing and OLS diagnostics to cheap fields.

## Session: 2026-08-17 — Add shared regression contract helpers

Added a frozen canonical protein split plus tested shared protocol, metric,
coverage, subgroup, geometry-summary, schema, and atomic-output helpers without
changing existing regression runners.

## Session: 2026-08-17 — Canonical gated OLS suite

Migrated all three gated families to the frozen development/test protocol,
recombined canonical outputs, replayable shards, complete-coverage geometry,
and restartable 48-layer Delta arrays with explicit joint partial coverage.

## Session: 2026-08-17 — Agent regression and SU guidance

Routed regression work through the canonical methodology and added streaming,
MaxRSS-based sizing, and Delta memory-billing guidance.

## Session: 2026-08-18 — SBATCH memory guidance

Added MaxRSS-based memory recommendations for the final regression workflows.

## Session: 2026-08-18 — Close gated regression audit gaps

Marked the migrated gated OLS code as canonical while retaining the pending
Delta rerun explicitly, and right-sized its six launch stages from the recorded
MaxRSS guidance.

## Session: 2026-08-18 — Align shared XGBoost search inputs

Moved the shared XGBoost anchor-layer search and cap sweep to the frozen master
protein split and canonical directional base256 features, leaving per-protein
XGBoost behavior unchanged and explicitly on hold. Right-sized both shared
XGBoost launchers to the recommended initial 64 GiB request.

## Session: 2026-08-18 — Canonical per-protein base256 OLS

Migrated per-protein OLS to the frozen cohort and deterministic 80/20 pair
split, canonical train/held-out/all-pair outputs, pooled baselines, locked-test
subgroups, explicit fit status, compact coefficient shards, and validated
48-layer consolidation. The production Delta rerun remains pending.

## Session: 2026-08-18 — Plan final shared XGBoost trajectory

Specified the canonical S4/5k/700 shared-XGBoost 80%-refit workflow, including
validation-only selection provenance, 48 restartable layer shards, all common
regression/subgroup/geometry outputs, native boosters, strict consolidation,
and explicit disclosure of the historical locked-test exposure.

## Session: 2026-08-18 — Implement final shared XGBoost trajectory

Added the frozen validation-only S4/5k/700 decision artifact, development-only
search and cap-sensitivity paths, restartable 48-layer final fits, canonical
streamed metrics and geometry, native-booster replay checks, strict shard
consolidation, Delta launchers, and focused tests. The production Delta rerun
and artifact copy remain pending.

## Session: 2026-08-18 — Colocate per-protein cluster aggregation

Moved the canonical per-protein base256 shard aggregator into its cluster
workflow, added a Delta launcher, and clarified that all cluster entrypoints
must live under `scripts/cluster_scripts/` rather than `scripts/multi_protein/`.

## Session: 2026-08-18 — Document flat Delta deployment

Documented the canonical flat `$HOME/openfold_interpretability_canonical`
script deployment, its sole `regression_protocol_data/` subdirectory, and the
separate `$HOME/csv_files` output root.

## Session: 2026-08-18 — Secondary-structure model comparison

Implemented the restartable DSSP-based 48-layer comparison of sequence-gated
OLS, shared OLS, per-protein OLS, and final shared XGBoost, with canonical
artifact/split validation, sparse shared-pair replay, matched held-out
sensitivity, strict consolidation, trajectories, plots, Delta launchers, and
focused regression tests. Production execution and result transfer remain
pending until all four source result families are complete.

## Session: 2026-08-19 — Add repository README

Added a concise repository guide linking the shared regression methodology,
main result summary, analysis outputs, paper figures, and experiment plans, and
excluded generated coefficient artifacts, compressed model shards, and outputs
larger than 20 MB from Git.

## Session: 2026-08-25 — Tighten SBATCH memory sizing

Recorded the triangle-violation subgroup replay's 6.15 GiB peak, reduced its
request from 64 GiB to 8 GiB, and made the shared memory guide mandatory for
all new or modified SBATCH allocations with a default 1.5–2x MaxRSS target.

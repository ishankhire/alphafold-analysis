# Current sources and status

As of October 15, 2026. Local manifests and compact result tables were inspected;
remote progress is identified separately. This page owns current source and
status information. The [project summary](../project-summary.md) owns scientific
interpretation; [methodology](methodology/regression_methodology.md) and the
[output contract](methodology/regression_outputs.md) own experimental definitions.
Use the [cluster guide](cluster/README.md) and workflow READMEs for execution.

## Primary results

The canonical comparison is protocol 2.0, fivefold OOF, four probe families,
552 frozen eligible proteins, and layers 0–47. The [input pool](../datasets.md)
contains 554 proteins; the [frozen metadata](../delta_files/regression_5fold_v1/metadata/)
defines the eligible cohort and folds.

All four local family manifests report complete runs. Each family's
`per_protein.csv` and `geometry_metrics.csv` contains 26,496 unique protein/layer
rows, with all 552 cohort proteins at every layer and complete row statuses.
These are manifest and table-coverage checks, not a fresh artifact-replay or
independent scientific audit.

| Family | Local results and manifest | Evaluation scope |
|---|---|---|
| Shared linear | [shared_ols_base256](../delta_files/regression_5fold_v1/shared_ols_base256/) | Unseen protein IDs |
| Per-protein linear | [per_protein_ols_base256](../delta_files/regression_5fold_v1/per_protein_ols_base256/) | Unseen pairs within seen proteins |
| Sequence-gated linear | [sequence_gated_ols_base256](../delta_files/regression_5fold_v1/sequence_gated_ols_base256/) | Unseen protein IDs |
| Shared fixed XGBoost | [shared_xgboost_base256_fixed_s4](../delta_files/regression_5fold_v1/shared_xgboost_base256_fixed_s4/) | Unseen protein IDs; same-cohort, non-nested configuration selection |

## Where to read the latest data and plots

Paths below are relative to the repository unless they name a Delta root.

| Source | Location and use |
|---|---|
| Protein inputs | [datasets.md](../datasets.md): membership and CAMEO/ECOD pair-tensor/PDB layouts. |
| Model outputs and retained models | `delta_files/regression_5fold_v1/<family>/`; existing Delta inputs: `$HOME/csv_files/regression_5fold_v1/<family>/`. New final-result outputs default to `/work/hdd/bfna/ikhire/regression_5fold_v1/`. Use each `run_manifest.json` and `model_artifacts/replay_index.csv`. |
| Exploratory and diagnostic experiments | New Delta outputs: `/work/hdd/bfna/ikhire/diagnostic_experiments/<experiment_name>/`. See the [storage conventions](cluster/README.md#csv-and-result-storage). |
| Fivefold triangle subgroups | [producer](../delta_files/regression_5fold_v1/triangle_violation_subgroups_5fold_v1/) has a complete 48-layer, 552-protein manifest; [publication analysis tables](../delta_files/regression_5fold_v1/analysis/triangle_violation_subgroups_5fold_v1/) supply the plots. |
| Per-protein subgroup MAE | [per_protein_subgroup_mae.csv](../delta_files/regression_5fold_v1/analysis/publication_subgroup_mae_5fold_v1/per_protein_subgroup_mae.csv) supplies protein-equal sequence/spatial curves; do not substitute pooled MAE. |
| Ablation CSVs | Committed in full (layerwise, `global/`, `pair_layer_similarity_outputs/`; checksums match Delta) under `delta_files/layerwise_ablation_csvs/`. Delta copy: `/work/hdd/bfna/ikhire/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD`; the original `toDelete` source was purged. The [ablation plot bundle](../delta_files/publication_plots/sept_2_plots/layerwise_ablation_trajectories/README.md) records selected trajectories. |
| Packed OOF distances | Delta: `/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/`; see the [export runbook](../scripts/cluster_scripts/predicted_distances/README.md). |
| Optimized TM-score replacements and reconstructed PDBs | Delta: `/work/hdd/bfna/ikhire/regression_5fold_v1/tm_score_optimized_v1/`; local geometry tables, score ledger, and compact provenance: `delta_files/regression_5fold_v1/tm_score_optimized_v1/`. Reconstructed PDBs remain on Delta. All 48 tasks of `22923529` and consolidation `22930940` succeeded. The root manifest is complete with 105,984 score/PDB index rows and `validation.mode=consolidation_only`. |
| Native error strata | Native `shared_ols_base256` signed error by fine sequence separation and final structure distance at all 48 layers, for all 552 proteins and excluding `7stz_C`, plus a CAMEO per-protein view ([runbook](../scripts/cluster_scripts/diagnostics/native_error_strata/README.md)). Job `23613749` confirmed these distances are the probes' training targets. Compact tables: `delta_files/diagnostic_experiments/native_error_strata_v1/`; overview figures: `delta_files/diagnostic_plots/native_error_strata/`. `native_strata.npz`, per-layer figures, and built pages remain under `~/diagnostic_experiments/native_error_strata_v1/`. Pages: [cohort](../delta_files/diagnostic_plots/native_error_strata/native_error_strata_page.html), [per protein](../delta_files/diagnostic_plots/native_error_strata/protein_error_strata_page.html). |
| Native XGBoost error strata | The same strata for native `shared_xgboost_base256_fixed_s4` on the 154 CAMEO proteins, with a per-protein view and a comparison against OLS on the same pairs. ECOD is left out while its final structures are unavailable, so values do not match the 551-protein OLS view. Job `23618658` reconciled 7,392 stored rows and 81,312 published per-protein subgroup rows. Compact tables: `delta_files/diagnostic_experiments/native_error_strata_xgboost_v1/`; overview figures: `delta_files/diagnostic_plots/native_error_strata_xgboost/`. Per-protein arrays, per-layer figures, and built pages remain under `~/diagnostic_experiments/native_error_strata_xgboost_v1/`. Pages: [cohort](../delta_files/diagnostic_plots/native_error_strata_xgboost/native_error_strata_page.html), [per protein](../delta_files/diagnostic_plots/native_error_strata_xgboost/protein_error_strata_page.html). |
| All-layer shared-OLS transfer | CAMEO-only diagnostic replay of all 240 frozen probes across 48 × 48 layers in `target_restandardized` mode, with signed-error distributions ([runbook](../scripts/cluster_scripts/diagnostics/cross_layer_transfer_all_layers/README.md)). Pilot `23615941`, array `23616061`, and aggregation `23616062` succeeded; `analysis_status=complete` with 792/792 checks. Pooled CAMEO numbers are not comparable to full-cohort tables. Compact tables: `delta_files/diagnostic_experiments/cross_layer_probe_transfer_all_layers_cameo_v1/`; figures: `delta_files/diagnostic_plots/cross_layer_transfer_all_layers_cameo/`. Shards, `per_protein.csv`, and `error_strata.csv` remain under `~/diagnostic_experiments/cross_layer_probe_transfer_all_layers_cameo_v1/`. [Results page](../delta_files/diagnostic_plots/cross_layer_transfer_all_layers_cameo/results_page.html). |
| Transfer follow-ups | Mechanistic breakdown, ablation and structure module comparison, gain versus direction, chain position × distance, and triangle violations against the error strata, on CAMEO except where stored 552-protein tables apply. [Report](../delta_files/analysis/all_layer_transfer_and_triangle_report_v1/REPORT.md); [tables and text reports](../delta_files/diagnostic_experiments/transfer_followups_cameo_v1/); [workflow](../scripts/cluster_scripts/diagnostics/transfer_followups/README.md). Re-run on the full cohort when ECOD returns. |
| Cross-layer shared-OLS transfer | Complete diagnostic replay of frozen probes from layers 22–26 and 43–47 over the full 10×10 block, in `raw` and `target_restandardized` modes ([runbook](../scripts/cluster_scripts/cross_layer_transfer/README.md)). Array `23533030` and aggregation `23533031` succeeded; `analysis_status=complete` with 431/431 parity checks passing. Compact tables: `delta_files/diagnostic_experiments/cross_layer_probe_transfer_v1/`; heatmaps: `delta_files/diagnostic_plots/cross_layer_transfer/`. `per_protein_subgroup_mae.csv` and `layer_shards/` remain on Delta under `/work/hdd/bfna/ikhire/diagnostic_experiments/cross_layer_probe_transfer_v1/`. |
| Optimized versus saved TM-score comparison | [Local model/layer summaries and trajectories](../delta_files/regression_5fold_v1/analysis/tm_score_optimization_comparison_v1/), derived from a checksum-verified local copy of the complete score ledger. All-layer mean gains are +0.012–0.023 across the four probes; final-layer gains are +0.0026–0.0079. These quantify alignment changes for the same predictions. |

The designated publication bundle is
[September 2](../delta_files/publication_plots/sept_2_plots/README.md), generated by
[generate_sept_2_publication_plots.py](../scripts/multi_protein/generate_sept_2_publication_plots.py).
Its `source_data/`, `figure_manifest.csv`, and `generation_summary.json` record
inputs, plotted values, and output provenance. Generation metadata records the
expected four models, 552 proteins, and 48 layers. Figures use protein-equal
means and paired-protein bootstrap intervals.

**The canonical geometry tables and both reconstruction TM-score figures now
use optimized TM-scores.** Only TM-score cells and their summary statistics were
replaced; all non-TM geometry values, including triangle violations, were
preserved. The [refresh checks and threshold table](../delta_files/regression_5fold_v1/analysis/tm_score_optimization_comparison_v1/)
record 240 passing column comparisons and input/output hashes. The active main
draft and supplement use the refreshed figures and scoring description.

## Canonical migration: data validated; access work stopped

Implementation `d18b486` is pushed and the active flat runtime is deployed at
`/work/hdd/bfna/ikhire/openfold_interpretability_canonical/` with a Git/source
inventory and retained Linux TMscore binary. Migration job `23142906` timed
out at 1h00m01s. All 13,312 copied files and all 40,341 pre-publication checks
passed, and the copied components reached the final root. Recovery `23162599`
passed all 40,343 revalidation checks, then was cancelled at the user's request
after 2h54m12s during permission normalization. The user accepted data
validation and declined further permission/access work. Migrated results can
be read by their owner; uniform allocation-group access remains unconfirmed.
Completion statuses remain `publishing` and `verifying_access`, and the stop
is recorded at `_migration_v1/cancellation_23162599.json` under the shared
results root. Do not resume unless the user requests it. No files were deleted.
The previous runtime/inventory is retained under
`deployment_history/recovery_de3827c_20261024/` in the shared deployment.
Local checks passed 143 tests with four documented expected failures. See the
[execution handoff](plans/delta_canonical_results_migration_plan.md#execution-handoff--october-23-2026).

## Pending work and evidence limits

| Workstream | Current state | Next step |
|---|---|---|
| Packed-distance export | Verified October 15: finalizer `22908531` completed in 2h09m25s, deployment check `22908532` succeeded, and the root manifest reports complete preparation/export with all 48 layers. | Reuse the finalized arrays for scoring and PDB export; no inference or fitting rerun is needed. |
| Optimized TM-score and reconstructed PDBs | All 48 tasks of `22923529` succeeded. Audit `22923530` verified layers 0–23 without errors before user-requested cancellation at 3h03m. Consolidation `22930940` completed in 1m43s, preserving producer identity and explicitly recording skipped bulk revalidation. Verified all 16 consolidated output hashes, 105,984 rows in each score/PDB index, four 26,496-row geometry tables, and 48 layer-manifest hashes. All 24 local workflow tests pass. | Compact outputs downloaded and canonical TM-scores refreshed locally. Retain `validation.mode=consolidation_only` when describing integrity checks; individual artifact files were not exhaustively reread. Future consolidation requests are 3 GiB / 10 minutes, based on the successful run. |
| Publication refresh | Complete: canonical TM-scores and summaries replaced; September 2 main and individual-protein TM-score figures regenerated with the existing formatting and bootstrap settings. Active manuscript thresholds are now layers 9 (>0.5) and 18 (>0.7) across all four probes. | Other geometry and ablation figures retain their existing values. Validation remains consolidation-only; no new physical-artifact audit was performed. |
| Adversarial review | [Saved August 30 review](../delta_files/analysis/cross_validation_adversarial_review_v1/review_findings.md) predates the production families. Its deferred production checks have not been established by the coverage checks above. | Retain the review as dated evidence; do not describe all production results as independently audited. |
| ECOD inputs | `/work/hdd/bfna/toDelete/thayes/` was purged by 2026-11-07. ECOD pair representations and alignments are lost; the surviving data, including 303/398 checksum-identical predicted PDBs and all ablation tables, is in `/work/hdd/bfna/ikhire/thayes/`. See the [Delta data inventory](cluster/delta_data_inventory.md). The 552-protein native strata table and training-target check cannot be rebuilt as specified. | Follow the inventory's to-do list; check the remaining recovery leads before rerunning anything. |

The geometry helper's defensive distance-matrix checks remain an unresolved
finding: [four tests](../scripts/cluster_scripts/cross_validation/tests/test_geometry_independent_oracles.py)
are still marked expected failures. Fixing scientific code is outside this
documentation cleanup.

The [pair-interaction](plans/pair_interaction_classification_probe_plan.md) and
[representation triangle-dominance](plans/pair_representation_triangle_dominance_plan.md)
plans describe other experiments; their current priority and production execution
are unconfirmed. Existing MDS visualization exports under
`delta_files/shared_ols_mds_pdb_exports/` are separate from the primary paper tables.

## Historical material

Completed or superseded plans and dated methodology/storage audits are under
[docs/historical](historical/). They are reference material, not current task
instructions. Archived preprint figures are under
[delta_files/historical_outputs](../delta_files/historical_outputs/).
Older triangle outputs, the XGBoost selection table, and saved adversarial-review
artifacts remain at established paths for existing scripts and provenance.

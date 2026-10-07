# Delta file dependencies and storage cleanup audit

Audit date: October 7, 2026. This is a source audit, not a live Delta disk scan.
No research files were deleted, moved, compressed, or changed.

## Recommendation for the PI

Keep one complete original copy of the baseline pair tensors for all 48 layers,
the final OpenFold PDBs, the experimental reference structures, sequence/MSA
inputs for the 554-protein dataset, and the regression models/results. Look for
3 TB first in repeated ablation representation saves, older duplicate or
full-MSA representation stores, and unused attention dumps. The current
552-protein baseline pair dataset is approximately 832 GB at float32, and the
bundled inference code saves only the first MSA row, approximately 5.8 GB over
all 48 layers. Those small MSA saves are useful for replaying structures and
are not a promising cleanup target. Hundreds of thousands of PDBs can occupy
substantial space, but are unlikely to provide 3 TB by themselves.

There is no unconditional deletion list: static code cannot establish which
files exist, their size, whether another researcher uses them, or whether a
particular copy is the only surviving original. The tables below distinguish
retained inputs from candidates with explicit conditions. They do not authorize
or perform deletion.

## Scope, evidence, and deliverables

The scan covers 196 Python, shell, Slurm, and notebook files in this checkout
(including tests and five result-directory scripts), plus 751 code/notebook files
inside `vizfold-main.zip`. The archive includes the inference implementation and
launchers named in Tyler Hayes's December 2025 path document. Its contents are
a local snapshot, not proof of the version deployed on Delta. The relevant
saving and loading code was inspected directly, alongside the dataset manifests,
regression methodology/output contracts, publication generators, and active
supplemental draft. No archive scripts were executed.

- [path_decisions.csv](path_decisions.csv): curated path families, usage, retention
  recommendations, and cleanup conditions; one row per path family.
- [protected_input_paths.csv.gz](protected_input_paths.csv.gz): 27,146 expanded
  expected input paths, one row per protein/layer pair tensor or final PDB for
  554 proteins. Alternative locations are recorded in separate columns. These
  are expected/candidate locations, not remotely verified files.
- [cohort.csv](cohort.csv): all 554 protein IDs, source, eligibility, exclusion
  reason, residue count, and current fivefold membership.
- [code_path_references.csv](code_path_references.csv): 11,379 source lines with
  path construction, filename, root, or filesystem references, including
  historical/archived code and comments. A reference does not prove active use.
- [absolute_path_references.csv](absolute_path_references.csv): 2,784 literal
  absolute path/template occurrences from code, repository documentation, and
  the PDF. Shell braces and variable names are preserved as written. Wrapped
  PDF paths can appear as fragments; reconstructed paths are in this report.
- [scan_coverage.csv](scan_coverage.csv): every scanned code file with its scope,
  line count, decoded-text SHA-256, and reference-line count. Notebook line
  numbers refer to concatenated code-cell sources, not output/image JSON.
- [audit_summary.json](audit_summary.json): cohort counts and size calculations.
- [inventory_delta_storage.py](../../../scripts/cluster_scripts/diagnostics/inventory_delta_storage.py):
  read-only Delta measurement tool. Its output is described below.

Dynamic paths, environment overrides, command-line inputs, and other users'
workflows prevent a static scan from listing every existing file. The compressed
input list expands the canonical cohort paths; `files.csv` from a Delta scan is
the actual file-by-file inventory.

## Files needed for the current analysis and its reproduction

`PROT` means a protein in [cohort.csv](cohort.csv); `LAYER` is 0 through 47.
The 554-protein pool is 154 CAMEO + 400 ECOD. The frozen fivefold cohort is
154 CAMEO + 398 ECOD = 552. Retain the full 554-protein pool conservatively:
older workflows and the original ablation tables use all 554 IDs.

| Exact Delta path or pattern | Usage and retention |
|---|---|
| `/work/hdd/bekh/thayes/openfold_reps/{PROT}/{PROT}_pair_block_{LAYER}.npy` | CAMEO baseline inputs for every layer. Keep all 48 for the cohort. |
| `/work/hdd/bfna/thayes/ecod/experiments/representation_saves/{PROT}_save_layers_0_47/{PROT}_pair_block_{LAYER}.npy` | ECOD baseline inputs for every layer. Keep all 48 for the cohort. |
| The above protein directories followed by `/pair_blocks/{PROT}_pair_block_{LAYER}.npy` | Supported alternative pair location. Determine which location is actually used before deduplicating. |
| `/work/hdd/bekh/thayes/openfold_reps_outputs_sm/my_outputs_align_{PROT}_representations_load_from_saved_layer_47/predictions/{PROT}_model_1_ptm_unrelaxed.pdb` | CAMEO probe target, geometry reference, secondary-structure and interaction input. Keep full-atom original. |
| `/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm/my_outputs_align_{PROT}_representations_load_from_saved_layer_47/predictions/{PROT}_model_1_ptm_unrelaxed.pdb` | ECOD equivalent. Keep full-atom original. |
| Baseline representation protein directory + `/predicted_pdb/{PROT}_model_1_ptm_unrelaxed.pdb` | Alternative PDB resolved before the layer-47 path by the shared helper. Preserve the actual input and verify duplicate identity. |
| Either `openfold_reps_outputs_sm` root + `/{PROT}/predicted_pdb/{PROT}_model_1_ptm_unrelaxed.pdb` | Third supported PDB location. |
| `/work/hdd/bekh/thayes/openfold_reps/{PROT}/{PROT}_msa_block_{LAYER}.npy` | Not read by current regression scripts; read by archived CAMEO structure-from-representation inference. Keep for inexpensive replay if query-row saves. |
| `/work/hdd/bfna/thayes/ecod/experiments/representation_saves/{PROT}_save_layers_0_47/{PROT}_msa_block_{LAYER}.npy` | Same ECOD dependency and retention. |
| `/projects/bekh/thayes/cameo_dset/rcsb_pdb/{PROT}.pdb` | Experimental CAMEO truth for ablation TM-score/LDDT recomputation. Preserve the exact chain-matched version. |
| `/work/hdd/bfna/thayes/ecod/domain_pdbs/` | Experimental ECOD domain structures. Keep cohort originals; do not assume all other domains are unused by the lab. |
| `/projects/bekh/thayes/cameo_dset/fasta_dir/{PROT}.fasta` | Input for CAMEO inference and ablations. |
| `/work/hdd/bekh/thayes/ecod_dataset/fasta_files/{PROT}.fasta` | Input for ECOD inference and ablations. |
| `/projects/bekh/thayes/alignments/{PROT}/` | Precomputed CAMEO alignments/template-search outputs used by inference, not by probe fitting. Keep cohort inputs for reruns. |
| `/work/hdd/bfna/thayes/ecod/alignments/{PROT}/` | Precomputed ECOD alignments/template-search outputs. Keep at least the selected 400 ECOD IDs. |
| `/work/hdd/bfna/thayes/FINAL_RESULTS_CSVS_TRI_END_FIX_PATCHED_FROM_OLD/` | **Current source** of layerwise CSVs in `copy_layerwise_ablation_csvs.sh`. Keep all 15 copied files. |
| `/work/hdd/bfna/thayes/FINAL_RESULT_CSVS/` | Older PDF/archived analysis source, including global results, representation-similarity outputs, and metadata used by repair scripts. Small; preserve. |
| `/work/hdd/bfna/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD/` | Archived correction/metadata repair source; distinct singular `RESULT` spelling. Preserve until provenance is reconciled. |
| `/work/hdd/bfna/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_STANDARDIZED/` | Archived standardized correction source. Preserve compact tables. |
| `$HOME/openfold_interpretability_canonical/` | Canonical scripts/helpers/configuration and `regression_protocol_data/regression_master_split_v1.csv`. |
| `$HOME/csv_files/regression_5fold_v1/metadata/` | `regression_outer_folds_v1.csv`, `per_protein_pair_folds_v1.csv`, `fold_digests.json`: frozen reproducibility inputs. |
| `$HOME/csv_files/regression_5fold_v1/{shared_ols_base256,per_protein_ols_base256,sequence_gated_ols_base256,shared_xgboost_base256_fixed_s4}/` | Current four model families: keep core CSVs, manifests, replay indices, coefficients/scalers, boosters, and validated task shards. |
| `$HOME/csv_files/regression_5fold_v1/triangle_violation_subgroups_5fold_v1/` | Producer triangle replay, thresholds, cohort/provenance, geometry reproduction checks, and layer checkpoints. |
| `$HOME/csv_files/regression_5fold_v1/analysis/triangle_violation_subgroups_5fold_v1/` | Current publication-ready triangle subgroup tables. |
| `$HOME/csv_files/regression_5fold_v1/analysis/publication_subgroup_mae_5fold_v1/` | Current per-protein sequence/spatial subgroup MAE tables. |
| `$HOME/csv_files/regression_5fold_v1/analysis/shared_ols_mds_pdb_exports/` | Default MDS export output; can be overridden by `MDS_PDB_OUTPUT_ROOT`. Keep selected structures/provenance. |
| `$HOME/csv_files/layerwise_ablation_csvs/` | Copies of corrected ablation inputs. Six are directly plotted in the active paper workflow. |
| `$HOME/csv_files/xgboost_hyperparameter_search/shared/winner_cap_sweep/cross_layer_selection_table.csv` | Selection evidence for the fixed shared-XGBoost configuration. Preserve with search tables/configs. |

The shared helper's candidate order matters: it first tries a PDB colocated with
the representation, then the layer-47 predictions path, then the alternative
PDB-root path. Two same-named PDBs are not necessarily interchangeable. The
model targets, atom order, residue sequence, and replay provenance must match.

**All 48 pair layers are used.** Four fivefold model families, model replay,
triangle subgroup replay, and publication subgroup replay read the original
pair tensors. The archive's cosine/correlation/CKA analysis also uses them.
Keeping only layer 47, only figure milestone layers, or only held-out proteins
would remove required training/replay inputs. Both directional entries are
used in the 256-feature probe; retaining only one triangle or a symmetrized
average is not a compatible replacement. Float16 conversion would be lossy;
lossless NPZ/HDF5 storage would require reader changes or exact restoration to
`.npy` before rerunning. Neither is a drop-in cleanup.

The two excluded ECOD IDs are `000196893` and `001681437`, because layer-0 pair
dimensions disagree with PDB residue counts. They are outside the current
fivefold analyses but still present in historical dataset/ablation provenance.
Together their 48-layer float32 pair tensors are only roughly 5.4 GB. Deleting
them cannot materially address a 3 TB target.

### Exact layerwise CSV filenames to retain

Within the current corrected source and `$HOME/csv_files/layerwise_ablation_csvs/`:

```text
ablate_msa_col_layerwise_fwd.csv
ablate_msa_col_layerwise_rev.csv
ablate_msa_row_layerwise_fwd.csv
ablate_msa_row_layerwise_rev.csv
ablate_msa_transition_layerwise_fwd.csv
ablate_msa_transition_layerwise_rev.csv
ablate_pair_transition_layerwise_fwd.csv
ablate_pair_transition_layerwise_rev.csv
ablate_tmu_layerwise_fwd.csv
ablate_tmu_layerwise_rev.csv
ablate_tri_start_ablate_tri_end_ablate_tmu_layerwise_fwd.csv
ablate_tri_start_ablate_tri_end_ablate_tmu_layerwise_rev.csv
ablate_tri_start_ablate_tri_end_layerwise_fwd.csv
ablate_tri_start_ablate_tri_end_layerwise_rev.csv
structure_module_tm_scores.csv
```

The current plotter directly uses forward/reverse Pair Transition, MSA
Transition, and joint Triangle Attention + TMU: six trajectories. The other
compact tables retain other experiments and should stay. Preserve all global
CSVs and master tables under the `global/` children of the original/corrected
result roots as well; the old document names 16 global tables. The literal
source inventory includes their paths.

### Additional analysis paths still referenced by code

The current paper uses local `delta_files/regression_5fold_v1/`,
`delta_files/publication_plots/sept_2_plots/`,
`delta_files/layerwise_ablation_csvs/`, and
`delta_files/shared_ols_mds_pdb_exports/`. These local plotting outputs are not
automatically present on Delta. Some historical/local scripts have different
roots. Preserve them cheaply if still available, rather than guessing that an
old-looking name is redundant:

```text
$HOME/csv_files/general_base256_combined/
$HOME/csv_files/general_base256_combined_canonical/
$HOME/csv_files/general_base256_combined_canonical_work/
$HOME/csv_files/per_protein_base256_combined/
$HOME/csv_files/shared_xgboost_base256_combined/
$HOME/csv_files/gated_linear_probes/
$HOME/csv_files/gated_linear_probes_smoke/
$HOME/csv_files/gated_linear_probes/consolidated/
$HOME/csv_files/diagnostic_sequence_separation_gated_probes/
$HOME/csv_files/diagnostic_secondary_structure_emergence/
$HOME/csv_files/diagnostic_secondary_structure_sequence_gated/
$HOME/csv_files/distance_error_geometry_null/
$HOME/csv_files/dssp_smoke_test/
$HOME/csv_files/pair_interaction_classification/cameo50_v1/
$HOME/csv_files/xgboost_hyperparameter_search/
$HOME/diagnostic_context_feature_sweep_csv_files/
```

The gated consolidated families are `analysis_1_sequence_gated/`,
`analysis_2_aa_pair_gated/`, `analysis_3_true_distance_statistics/`, and
`analysis_4_aa_sequence_gated/` under the suite's consolidated root (see
`gated_linear_probe_common.py` and `prepare_gated_linear_probe_suite.py`).
Some local diagnostics explicitly reference older roots such as
`delta_files/baseline256/`, `diagnostic_precision_rowcap_pca_csv_files/`,
`diagnostic_sampling_weighting_kfold_csv_files/`,
`diagnostic_lin_tree_context_reg_csv_files/`, `diagnostic_*tree*_csv_files/`,
`mds_reconstruction/`, and `pair_representation_triangle_dominance/`.
Their presence and use on Delta are unverified; full source references are in
the CSV inventory. The current PyMOL renderer also uses the ten original
full-atom PDBs under
`delta_files/shared_ols_mds_pdb_exports/aug-16-archive/reconstructed_pdb_files/general_base256_avg_symmetric/*/original_full/`.
Those remain active visualization inputs despite the archive directory name.

## Cleanup candidates, ranked by likely impact

Every candidate requires checking actual sizes and experiment ownership. No
savings below are measured Delta totals.

| Priority | Exact path family | What can be removed, and conditions |
|---|---|---|
| 1 | `/work/hdd/bfna/cameo/openfold_outputs_layerwise_ablation/representation_saves/{PROT}/` | **Ablation-specific pair/MSA saves.** The archived `...LAYER_UPDATE_SAVE_PAIR_REP_ABLATE.sh` saves layers 0–47 for each ablation boundary by default. Current probes use unablated baseline roots; ablation plots use CSVs. If this representation-intervention study is finished/abandoned and no archive analysis still needs these tensors, this is a strong cleanup candidate. Keep ablation PDBs and score/provenance tables. |
| 2 | `/projects/bekh/thayes/final_cameo_outputs/{representation_full_saves,representation_saves,representation_pool_saves}/` | Older representation stores. `representation_full_saves` is a particularly useful place to check for full-MSA tensors. Verify shape/dtype, cohort, seed/recycle/model version, and exact identity before treating any store as a duplicate. Older evaluation/AE code references these roots. |
| 2 | `/work/nvme/bekh/thayes/cameo_reps/`, including `subset_10/` | Older representation copies/correlation inputs. Remove redundant copies only after identifying a surviving original and confirming the archived analysis is no longer needed. |
| 2 | `/projects/bekh/thayes/final_cameo_outputs_ae/{representation_ae10_saves,representation_ae10_saves_clean}/` | Autoencoder study saves/cleaned duplicates, unused by current four probe families. Confirm that the separate AE study is finished and checkpoint/provenance requirements are preserved. |
| 3 | `/work/nvme/bekh/thayes/cameo_triplets/attn_full/7ebt_B/` and other `attn_full/{PROT}/` children | Full triangle attention maps, unused by this repository's current probes. Archive's attention/triplet study does use them. Retain needed summaries/examples or remove after that study's owner agrees it is finished. Could be large because full triangle maps have cubic residue scaling. |
| 3 | `/work/nvme/bekh/thayes/cameo_triplets/attn_random/` and abandoned triplet-run outputs | Same study-specific conditions. Sparse/random summaries can be much smaller than full maps; measure rather than assuming terabytes. |
| 4 | `/work/hdd/bfna/thayes/ecod/alignments/{PROT}/` for IDs outside the selected 400 | PDF says 25,000+ precomputed alignments. Most are outside this project cohort, so there is scope to retire the unselected collection **if no other lab project needs it**. Alignment count alone does not establish size: 24,600 extra directories need to average about 122 MB to yield 3 TB. Keep selected alignments/templates/FASTA and exact-generation provenance. |
| 5 | Original pair roots, IDs outside the 554-protein pool; extra noncanonical layer/run directories | Current fivefold workflows do not need these extras. Some legacy diagnostics discover all directories rather than using only the frozen manifest. Review those and other users before pruning. Long proteins disproportionately dominate pair storage through R-squared scaling. |
| 5 | `*_output_dict.pkl`, other embedded-array pickles, extra `.npy/.npz/.h5` stores under experiment output directories | Inspect before cleanup: serialized model outputs can hold MSA/pair tensors or other large arrays. Standard OpenFold saves these when `--save_outputs` is enabled. Representative ablation launchers inspected here do **not** enable that flag, so their existence/size must be verified, not presumed. Keep original PDBs, seeds/configuration, and required metrics; do not delete model/replay NPZs by extension. |
| 6 | Old ablation structures under the roots listed below | Compress/archive verified PDBs and prune superseded runs only after scores, corrected experiment identities, and provenance are preserved. Current plots only read CSVs, but PDBs are needed to recompute TM-score/LDDT, investigate artifacts, or perform new structural analyses. Not a likely standalone 3 TB source. |
| 7 | `my_outputs_align_{PROT}_representations_load_from_saved_layer_{0..46}/` under both `openfold_reps_outputs_sm` roots | Intermediate structure-module predictions are unused as current regression targets; older per-layer structure analyses use them. Conditional archive/removal after those analyses are complete. **Keep layer 47.** Just one PDB per layer/protein is around a few GB, not TB. |
| 8 | Old pairwise predictions, duplicated distance/feature caches, obsolete completed shards, temporary files, obsolete environments/logs | Measure individually. Keep files referenced by replay/checkpoint manifests. The old gated workflow needs `*_predictions.npz` and `task_predictions.npz` for resume/aggregation; they are not immediately disposable merely because the v2 output contract omits raw pair rows. Many objects named `feature_cache` in the code live only in RAM. |

The archived ablation saver constructs representation paths as:

```text
${OUT_BASE}/representation_saves/${PROT}/${PROT}_${ABLATE_KEY}_${MODE_TAG}_layers_${RANGE_START}_to_${RANGE_END}_${RUN_TAG}_save_layers_${SAVE_LAYER_TAG}/
```

Its default `OUT_BASE` is
`/work/hdd/bfna/cameo/openfold_outputs_layerwise_ablation`, with
`SAVE_LAYER_TAG=0_to_47`. A run's root can be overridden; look for the same
`representation_saves/` pattern in other ablation output roots too.
For illustration, one protein with R=300 has 2.212 GB of float32 pair data
across 48 layers. Repeating those 48 saves for 48 ablation boundaries is
106.17 GB **per protein, per component/direction**. Thirty such proteins would
exceed 3 TB. This proves capacity for large redundancy, not that those files
actually exist or are scientifically redundant.

### Ablation result roots: preserve provenance before pruning

```text
/work/hdd/bekh/openfold_outputs_layerwise_ablation/
/work/hdd/bfna/openfold_outputs_layerwise_ablation/
/work/hdd/bfna/cameo/openfold_outputs_layerwise_ablation/
/work/nvme/bekh/openfold_outputs_layerwise_ablation/
/work/hdd/bfna/openfold_outputs_layerwise_ablation_TRI_END_FIX/
/work/hdd/bekh/openfold_outputs_global_ablations/
/work/hdd/bfna/openfold_outputs_global_ablations_TRI_END_FIX/
/work/hdd/bfna/openfold_outputs_fixed_ablation/
/work/hdd/bfna/thayes/ecod/experiments/{PROT}/
/work/hdd/bfna/thayes/ecod/experiments_TRI_END_FIX/{PROT}/
```

The corrected experiment roots are visible in the bundled archive and may
produce the patched CSVs. Preserve corrected runs needed to verify the paper.
The parent `/work/hdd/bfna/thayes/ecod/experiments/` also contains the **required**
`representation_saves/` and `openfold_reps_outputs_sm/` directories. Deleting
that parent would delete active baseline inputs. Original and corrected runs
are different interventions/results, not automatically duplicate files.

Keep the small per-protein and aggregate score tables, including
`tm_layerwise_matrix.csv`, `/work/hdd/bfna/cameo/{layerwise_tm_csvs,global_tm_csvs}/`,
the `_TRI_END_FIX` counterparts, ECOD standardized score directories,
`/u/thayes/vizfold/slurm_scripts_ecod/SM_SCORES/tm_scores_by_layer_and_protein.csv`,
`/projects/bekh/thayes/final_cameo_outputs/cameo_openfold_tm_scores.csv`, and
`/u/thayes/vizfold/slurm_scripts_ecod/selected_proteins/{all_categories_selected.csv,all_selected_proteins.txt}`.
These preserve filtering, lengths, categories, and result provenance at low cost.

### Files outside this project's deletion scope

Do not remove `/work/hdd/data/alphafold2/database/` or its template/database
children: these are shared inference resources. The PDF's reference to the old
`/ime/hdd/rhaas/SUP-5301/database` is historical context, not a cleanup command.

The archive references `/work/hdd/bekh/thayes/boltz1_reps/`,
`/work/hdd/bekh/thayes/boltz2_reps/`,
`/projects/bekh/thayes/boltz1_representations_clean/`, and
`/projects/bekh/thayes/boltz_representations_clean/`. They are not inputs to this
repository's current OpenFold probes, but they belong to cross-model studies.
Do not classify them as unused simply from the present paper's scope.

## How much space do the different objects occupy?

All quantities here use decimal GB/TB (1 TB = 10^12 bytes); measurements are
explicitly labeled local. Float32 estimates exclude small NPY headers and
filesystem metadata. The 552-cohort manifest has mean length 213.3, median 180,
range 30–794, sum of lengths 117,749, and sum of squared lengths 33,858,983.

| Object | Size or calculation | Implication |
|---|---|---|
| Required float32 baseline pair tensors | `48 * 128 * 4 * sum(R^2)` = **832.118 GB** (0.757 TiB) | Keep one original complete copy. A single cohort copy cannot supply 3 TB. |
| Saved query-row float32 MSA tensors | `48 * 256 * 4 * sum(R)` = **5.788 GB** | Bundled saver uses `m_save = m_np[0:1, ...]`. Cheap to retain for structure-module replay. |
| Older full float32 MSA tensors with 512 sequences | `48 * 512 * 256 * 4 * sum(R)` = **2.963 TB** | Potential 3 TB target if this older full-depth format actually exists and is no longer required. |
| Older full float32 MSA tensors with 1,024 sequences | Same formula with 1,024 = **5.927 TB** | Depth can vary by protein; actual headers and disk measurements are authoritative. |
| 500,000 ablation PDBs at 100/200/500 KB each | **50/100/250 GB** | Substantial, but reaching 3 TB requires 6 MB per PDB on average. |
| Ten local original full-atom PDBs | 42,444–384,993 bytes each; total **1.635 MB** | Extrapolating their mean to 500,000 gives **81.75 GB**, illustrative only. Actual ablation structures may differ. |
| Locally available `delta_files/` regular files | Approximately **5.21 GB** including ignored/untracked results | Local copy size, not the remote result tree. Compact results are worth preserving. |
| Baseline tensors' local sample | All ten layer-0 pair tensors inspected are **float32**, shape `(R,R,128)` | Supports the float32 estimate; remote headers still need verification. |

The distinction between **alignment files** (A3M/STO/HHR search outputs) and
**MSA representation tensors** (per-layer learned arrays) is essential. The
25,000+ count in the PDF refers to precomputed alignments, not 25,000 sets of
48-layer learned tensors. It cannot be multiplied by the full-MSA formula.

Full attention tensors can grow cubically in residues and also scale with
heads/layers/recycles. There is no current Delta measurement or verified map
shape for the mentioned attention directory, so no TB savings are assigned.

Lossless compression of PDBs is a reasonable archive option after verifying
a copy, but current readers expect plain `.pdb`. To rerun scoring or render
figures, restore original files or deliberately add compressed-input support.
Do not delete the full-atom final reference merely because a CA-only MDS
reference exists: interaction/DSSP diagnostics and original-structure rendering
need information absent from a CA-only trace.

## Read-only measurement on Delta

Deploy the standalone script with the other canonical analysis entrypoints;
it needs only the Python standard library. It does not import model code,
load full tensors, read pickle contents, or remove input files. Run in a
suitable Delta session with the relevant filesystems accessible:

```bash
cd "$HOME/openfold_interpretability_canonical"
~/protein_venv/bin/python inventory_delta_storage.py \
  --cohort-manifest "$HOME/openfold_interpretability_canonical/regression_protocol_data/regression_master_split_v1.csv" \
  --output-dir "$HOME/csv_files/storage_audit_$(date +%Y%m%d_%H%M%S)"
```

If only one allocation needs space, scan that allocation's relevant roots
first. Example for the highest-priority bfna ablation candidate:

```bash
~/protein_venv/bin/python inventory_delta_storage.py \
  --root /work/hdd/bfna/cameo/openfold_outputs_layerwise_ablation \
  --output-dir "$HOME/csv_files/storage_audit_ablation_$(date +%Y%m%d_%H%M%S)"
```

Outputs, all incrementally streamed where practical:

| File | Row granularity and principal fields |
|---|---|
| `files.csv` | One regular file: scan root, absolute path, category, cohort-ID scope, logical bytes, allocated bytes, unique allocated bytes, modification timestamp, hard-link count. |
| `summary.csv` | One root/category/cohort-ID group, sorted by unique allocated bytes: file count, logical/allocated bytes and decimal TB. |
| `roots.csv` | One scan root: complete/partial/unavailable, file/error counts, elapsed seconds. |
| `npy_header_samples.csv` | Up to three samples per root/category/cohort group: path, array shape, dtype, Fortran order. |
| `errors.csv` | One unreadable/missing root/file or header error. Missing roots are not treated as zero-sized complete scans. |

Root/category sums exclude directory/symlink metadata. The scanner does not
follow symlinks, prunes overlapping roots/output directories, and counts
hard-linked file blocks once across the scan. Hard links still outside the
scanned roots can prevent deletion from reclaiming those blocks. Cohort scope
means only that an ID appears in a path, not that the file is required or
removable. Header sampling is bounded and does not establish that an entire
store has one shape/dtype. Add `--root` entries for paths from the source
inventory outside the defaults, such as old Boltz stores or custom `CSV_ROOT`
locations, if the owners want them measured.

Use **allocated bytes on the allocation/filesystem that is actually full**
when assembling the 3 TB cleanup. `bekh` versus `bfna`, `/projects` versus
`/work/hdd` versus `/work/nvme`, and HOME are separate paths/quotas; freeing one
must not be assumed to relieve the other. NCSA's current documentation
describes the filesystems and quota tools:
[Delta system architecture](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/architecture.html),
[Delta data management](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html).

Before removing a selected candidate, verify no running job needs it, preserve
compact results/configuration/seed provenance, and validate a surviving copy
if it is being deduplicated. A different seed/recycle count, corrected triangle
ablation, pooled/symmetrized feature, or CA-only reference is not an exact copy.
No remote measurement, backup verification, or cross-project usage confirmation
was available during this audit, so the report cannot guarantee 3 TB of
reclaimable data until `summary.csv` and candidate provenance are checked.

## Principal source anchors

- `datasets.md`; `cluster_regression_helpers.py:63–136`: baseline layouts and
  supported pair/PDB alternatives.
- `regression_protocol_data/regression_master_split_v1.csv`; frozen fivefold
  metadata: cohort, technical exclusions, and actual length-based estimates.
- `scripts/cluster_scripts/copy_layerwise_ablation_csvs.sh`: corrected source
  and all 15 filenames; `plot_layerwise_ablation_tm_scores.py`: six current plots.
- `generate_sept_2_publication_plots.py:45–55`: current figure inputs.
- `docs/methodology/regression_outputs.md`; fivefold aggregators/checkpoint
  code: preserve models and authoritative shard/replay dependencies.
- `vizfold-main.zip!/vizfold-main/openfold/model/evoformer.py:1726–1762`:
  representation saver stores only MSA row zero and the full pair tensor.
- `vizfold-main.zip!/vizfold-main/openfold/model/model.py:600–618`: inference
  loads both saved MSA and pair tensors for structure-module replay.
- Archive's CAMEO/ECOD `inference_representation_load_from_layer.sh`:
  FASTA, alignments, templates, and both saved representation paths.
- Archive's `inference_general_per_layer_ablate_flags_LAYER_UPDATE_SAVE_PAIR_REP_ABLATE.sh:45–49,113–114,154–157`:
  all-48-layer saving repeated across ablation boundaries.
- Archive's `MAKE_FINAL_CORR_COS_REP_DEC.py:32–47`: additional pair-tensor use
  and small similarity result files under the older final-results tree.
- [Upstream OpenFold inference](https://github.com/aqlaboratory/openfold/blob/main/run_pretrained_openfold.py):
  `--save_outputs` writes serialized model outputs. Local archived fork also
  implements that option, but it does not prove the flag was used on Delta.

## Validation performed locally

The read-only scanner passed synthetic checks for file totals, hard-link block
accounting, overlapping-root pruning, output-directory exclusion, directory
symlink exclusion, NPY header parsing, cohort classification, and partial/missing
root reporting. It scanned 626 local protein-data files with zero errors and
without loading tensor payloads. The expanded keep list was checked for 554
unique proteins, 48 distinct pair layers and one final PDB per protein, and
552 fivefold members. Source inventories were validated as parseable CSVs.

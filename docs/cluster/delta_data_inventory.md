# Delta data inventory

Status of the Delta data this project depends on, after the November 2026
purge of thayes's folder. It covers ECOD and the CAMEO copy; the full CAMEO
catalogue and the other cluster data are in the
[cluster data preservation plan](../plans/cluster_data_preservation_plan.md).

[Important Delta Paths 1.docx.pdf](<../../Important Delta Paths 1.docx.pdf>)
is a version of the documentation for the files on the cluster. Some of it may
be outdated, so where it conflicts with this repository's documentation, defer
to the repository documentation.

## Summary

- thayes's ECOD data was moved to `/work/hdd/bfna/toDelete/thayes/` and
  purged between November 3 and 7, 2026. The folder no longer exists.
- What survived was copied to **`/work/hdd/bfna/ikhire/thayes/`**.
- The PI's copy of the ECOD tree is on PACE Phoenix at
  `/storage/cedar/cedar0/cedarp-gkrishnan46-0/ecod/`. On Nov 8 its predicted
  PDBs (all 400 proteins × 48 layers) were copied back to Delta.
- **Safe on Delta:** all ablation result tables, all probe predictions, and
  predicted structures for every ECOD protein.
- **Not on Delta, but on PACE:** ECOD pair representations, alignments,
  `domain_pdbs/`, and the raw ablation runs that were lost. These have not been
  copied, and the contents of the PACE subfolders have not been checked.
- CAMEO has a verified second copy under `/work/hdd/bfna/ikhire/thayes/cameo/`.
- Every ablation figure in the paper can still be regenerated from saved
  tables.

## Timeline

| Date | Event |
|---|---|
| Nov 3 | Directory check: thayes's data had moved from `/work/hdd/bfna/thayes/` to `/work/hdd/bfna/toDelete/thayes/` and was still intact. |
| Nov 3–6 | Most ECOD folders disappeared, including `representation_saves/` and `openfold_reps_outputs_sm/`. Cause unknown. |
| Nov 6, 15:52 | Only 5 top-level folders remained. `ecod/experiments/` was losing ~100–190 entries a minute. The rescue copy started. |
| Nov 7 | The source folder is gone entirely. Copying finished. |
| Nov 7–8 | Job `23649985` copied CAMEO data and triangle attention maps to `ikhire/thayes/cameo/` (777 GiB, verified). See [CAMEO](#cameo). |
| Nov 8 | The PI's ECOD copy was found on PACE Phoenix. `experiments/openfold_reps_outputs_sm/` was copied back to Delta by Globus. |

No process from gprash or thayes was deleting files. gprash was only copying
`cameo` and `cameo_data`. The purge was probably done by an admin or the
system, but that isn't confirmed.

## Rescued or restored: `/work/hdd/bfna/ikhire/thayes/`

Everything here came from the Nov 6–7 rescue copy, except the first row, which
came from PACE on Nov 8.

| Path | What it is | Status |
|---|---|---|
| `ecod/experiments/openfold_reps_outputs_sm/my_outputs_align_<ID>_representations_load_from_saved_layer_<LAYER>/predictions/<ID>_model_1_ptm_unrelaxed.pdb` | Per-layer predicted structures, restored from PACE. Layer 47 is the probes' distance target. This is the path `datasets.md` uses. | ✅ All 400 TM > 0.7 proteins × layers 0–47 (19,200 folders, one non-empty PDB each). Layer 47 matches byte for byte the 305 complete rescued PDBs below and 100 older local copies. |
| `FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD/` | All ablation TM-score tables (post-tri-end-fix): 14 layerwise FWD/REV files, structure-module scores, `MASTER_all_experiments_wide.csv` | ✅ Complete (65 files, 47 MB) |
| `FINAL_RESULT_CSVS_…/global/` | Global ablation tables: 13 conditions + baseline, 3-run mean per protein | ✅ Complete; also committed in git (see below) |
| `ecod/experiments/<ID>/` | Per-protein raw runs: layerwise and global ablations, `zero_m`/`zero_z`, baseline, representation-save run (PDBs + `timings.json`), `<ID>_tm_scores.csv` | ⚠️ Partial, see below |
| `ecod/experiments/<ID>/my_outputs_align_<ID>_representations_layers_save_0_47/predictions/<ID>_model_1_ptm_unrelaxed.pdb` | Unablated final predicted structure, the probes' distance target | ⚠️ 303 / 398 cohort proteins. SHA-256 identical to the originals the probes used. Superseded by the complete PACE copy above. |
| `ecod/pdbs/ecod.latest.pdb.tar.gz` | ECOD experimental (reference) structures | ✅ Complete (537 MB) |
| `ecod/domain_pdbs_mapping.csv` | ECOD domain-to-PDB mapping | ✅ |
| `ecod_reps/logs/` | Baseline representation-save Slurm logs | ✅ (814 files) |
| `ecod/logs/` | Ablation Slurm logs (`ablate_layers-*.out/err`) | ⚠️ 96,178 of 125,942 files |
| `cameo/logs_TRI_END_FIX/` | CAMEO tri-end-fix ablation Slurm logs | ✅ (7,002 files, 13 GB) |
| `cameo_data/pairwise_dist_mats/` | CAMEO pairwise distance matrices (`<PROT>_pairwise_dist.npy`) | ✅ (184 files) |

**`ecod/experiments` completeness:**
- **TM > 0.7 cohort (400 proteins):** 313 folders saved, 87 lost (50 before
  the copy started, 37 during it). List of the 50:
  `/work/hdd/bfna/ikhire/thayes_copy_logs/priority_missing_from_source.txt`.
  **305 folders are complete:** all 716 runs, each with its PDB (672
  layerwise, 36 global, 3 `zero_m`, 3 `zero_z`, 1 baseline, 1
  representation-save run). A further 8 are partial. Per-protein counts are in
  `thayes_copy_logs/inventory_priority_400.csv`. Two of the 305 are outside the
  552-protein cohort, which leaves 303.
- **Other ECOD entries (~25,900):** about 63% have files. The rest are empty
  folders whose contents were deleted before they were copied.

Copy scripts and logs are in `/work/hdd/bfna/ikhire/` (`copy_*.sh`,
`thayes_copy_logs/`).

## Missing from Delta, on PACE

The PI's copy at `/storage/cedar/cedar0/cedarp-gkrishnan46-0/ecod/` has the
same top level as the Nov 3 Delta listing
([`remote_context.txt`](../../miscellaneous/required_paths_2026-11-03/remote_context.txt)),
plus `alignments/` and `logs_TRI_END_FIX/`. Apart from
`experiments/openfold_reps_outputs_sm/`, nothing below the top level has been
checked or copied.

| Path (under `ecod/`; originally `/work/hdd/bfna/thayes/ecod/`) | What it is | Impact while missing |
|---|---|---|
| `experiments/representation_saves/` | ECOD pair (and MSA) representations, all 48 layers; about 450 GB for the cohort (estimated) | Cannot refit probes or run new representation analyses on ECOD. CAMEO pair blocks are safe. Copy with [`pair_blocks.batch`](../../miscellaneous/ecod_pace_transfer/README.md). |
| `alignments/` | ECOD MSAs and template hits | Needed only to rerun ECOD inference. It was already missing from Delta on Nov 3. |
| `domain_pdbs/` | ECOD domain reference PDBs. On Delta these were symlinks into bekh. | The rescued tarball in `ecod/pdbs/` covers this. Check that the PACE copy holds real files, not broken links. |
| `experiments_TRI_END_FIX/` | Corrected ECOD ablation run outputs | Score tables survive. The raw structures exist only on PACE. |
| `experiments/<ID>/` for the 87 cohort proteins lost in the purge | Raw ablation runs | Score tables survive. |
| `{layerwise,global}_tm_csvs_TRI_END_FIX_STANDARDIZED/` | Per-source ECOD score tables | Superseded by the combined tables in `FINAL_RESULT_CSVS_…`. |
| `logs_TRI_END_FIX/` | ECOD tri-end-fix ablation logs | Low impact. |

## Lost (not found anywhere)

| Original path (under `/work/hdd/bfna/thayes/`, later `toDelete/thayes/`) | What it was | Impact |
|---|---|---|
| `FINAL_RESULT_CSVS/`, `FINAL_RESULT_CSVS_TRI_END_FIX_STANDARDIZED/` | Older and standardized versions of the result tables. They sat outside `ecod/`, so they are not in the PACE copy. | Low impact. The patched version used by the paper survives. |

## Safe, outside the purge

| Path | What it is |
|---|---|
| `/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/` | Out-of-fold probe predictions for every residue pair: 552 proteins × 48 layers × 4 probes (all 105,984 arrays complete). Distances can be analysed without pair reps. |
| `/work/hdd/bfna/ikhire/regression_5fold_v1/` and `$HOME/csv_files/` | All regression results, per-protein metrics, geometry and triangle tables |
| `/work/hdd/bekh/thayes/ecod_dataset/fasta_files/` | ECOD FASTA sequences (all 400 cohort IDs present) |
| `delta_files/layerwise_ablation_csvs/` (git, `main`) | Full copy of `FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD/`: layerwise and global ablation tables, master tables, pre-patch backups, `pair_layer_similarity_outputs/` (Fig. 6/S13). The older `sept-2-archive` branch has the 15 layerwise CSVs only. |
| `/sw/external/alphafold2/data_hyun_official/` | Shared AlphaFold2 databases on Delta (UniRef90, MGnify, BFD, UniRef30, PDB70), if MSAs must be regenerated |

## Paper reproducibility

- **Layerwise ablation figures (3, 4, 5, S7–S11, S21):** ✅ All 7 components ×
  FWD/REV × 48 layers × 554 proteins, with no gaps, in git and on Delta.
- **Global ablation figures (2, S2–S6):** ✅ All conditions × 554 proteins ×
  3 runs, in git and on Delta. Per-run scores aren't stored, only the 3-run mean.
  Five triangle files have an empty `length` column; join on protein ID.
- **Probe and triangle figures:** ✅ The saved predictions and result tables
  are intact.

## To do

1. ✅ Done Nov 7: `FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD/` (all 65
   files, checksums verified) is committed as
   `delta_files/layerwise_ablation_csvs/`.
2. ✅ Done Nov 8: the ECOD PDB path in `datasets.md` points to the copy
   restored from PACE.
3. Repoint code that uses the old ECOD roots. The restored PDBs keep the
   original layout, so only the root changes, from
   `/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm` to
   `/work/hdd/bfna/ikhire/thayes/ecod/experiments/openfold_reps_outputs_sm`.
   The old root is hard-coded in `cluster_regression_helpers.py`
   (`ECOD_PDB_ROOT`, `ECOD_PAIR_BLOCKS_ROOT`),
   `generalized_model_ecod/general_train_test_common.py`, and the secondary
   structure, DSSP and `distance_error_geometry_null` diagnostics and their
   sbatch files. The `pdb_path` column in
   `predicted_distances_5fold_v1/metadata/proteins.csv` is also stale; the
   checksums still match.
4. Copy ECOD pair blocks from PACE with
   [`pair_blocks.batch`](../../miscellaneous/ecod_pace_transfer/README.md)
   into `/work/hdd/bfna/ikhire/thayes/ecod/experiments/representation_saves/`
   when they are needed. First check that the PACE folder holds all 400
   proteins × 48 layers, and keep the space under the bfna quota in mind (see
   the [preservation plan](../plans/cluster_data_preservation_plan.md)). The
   Granite tape lead (`/granite/nsf/access/bekh`) is needed only if PACE turns
   out to be incomplete.
5. Analyses that need final structures can use all 552 cohort proteins again.
   ECOD analyses that need pair blocks still cannot run.
6. Check `domain_pdbs/` on PACE for broken symlinks before copying it.

## CAMEO

The full CAMEO catalogue, sizes and purge-risk evidence are in the
[cluster data preservation plan](../plans/cluster_data_preservation_plan.md).

**Copied Nov 7–8 (job `23649985`), verified.** CAMEO MSAs, reference
structures, predicted and ablation PDBs, 48-layer pair/MSA blocks, and triangle
attention maps are now also under **`/work/hdd/bfna/ikhire/thayes/cameo/`**:
777.3 GiB, 293,280 files. Each of the 383 copy tasks has matching source and
destination file counts and bytes, and a dry-run rsync found nothing pending.
The originals are untouched.

| Copy path (under `thayes/cameo/`) | Source |
|---|---|
| `projects_bekh/alignments/` | `/projects/bekh/thayes/alignments` (MSAs) |
| `projects_bekh/cameo_dset/` | `/projects/bekh/thayes/cameo_dset` (reference PDBs, FASTA) |
| `work_hdd_bekh/openfold_reps/` | `/work/hdd/bekh/thayes/openfold_reps` (pair + MSA blocks, 48 layers) |
| `work_hdd_bekh/openfold_reps_outputs_sm/` | Per-layer predicted PDBs |
| `work_hdd_bekh/openfold_reps_outputs/`, `work_nvme_bekh/cameo_reps_outputs/` | Representation-save run outputs |
| `work_hdd_bfna/openfold_outputs_{layerwise,global,fixed}_ablation/` | Ablation PDBs |
| `work_nvme_bekh/openfold_outputs_layerwise_ablation/` | 10-protein layerwise ablation PDBs |
| `work_hdd_bfna/gprash/cameo_pdbs/` | gprash per-layer PDBs |
| `work_hdd_bekh/{cameo_triplets,test_attn,temp_store}/`, `work_nvme_bekh/cameo_triplets/`, `work_hdd_bfna/attention_map_dir_test_triplets/` | Triangle attention maps |

Not copied: `attention_map_dir_test_triplets/plots/` (unreadable to ikhire),
the nvme duplicate of the pair blocks, and the older CAMEO experiments. Logs and
`verify.tsv` are in `/work/hdd/bfna/ikhire/cameo_copy/logs/23649985/`.

Quick checks on Nov 7: all 154 CAMEO predicted PDBs
(`/work/hdd/bekh/thayes/openfold_reps_outputs_sm/`), layer-47 pair blocks
(`/work/hdd/bekh/thayes/openfold_reps/`) and alignments
(`/projects/bekh/thayes/alignments/`) are present.

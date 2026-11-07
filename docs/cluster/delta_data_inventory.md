# Delta data inventory

Status of the Delta data this project depends on, after the November 2026
purge of thayes's folder. ECOD is covered here; CAMEO and other cluster data
are still to be added.

[Important Delta Paths 1.docx.pdf](<../../Important Delta Paths 1.docx.pdf>)
is a version of the documentation for the files on the cluster. Some of it may
be outdated, so where it conflicts with this repository's documentation, defer
to the repository documentation.

## Summary

- thayes's ECOD data was moved to `/work/hdd/bfna/toDelete/thayes/` and
  purged between November 3 and 7, 2026. The folder no longer exists.
- What survived was copied to **`/work/hdd/bfna/ikhire/thayes/`**.
- **Lost:** ECOD pair representations, ECOD alignments, and some per-protein
  ablation runs. **Safe:** all ablation result tables, all probe predictions,
  and 303 of 398 ECOD predicted structures.
- Every ablation figure in the paper can still be regenerated from saved
  tables.

## Timeline

| Date | Event |
|---|---|
| Nov 3 | Directory check: thayes's data had moved from `/work/hdd/bfna/thayes/` to `/work/hdd/bfna/toDelete/thayes/` and was still intact. |
| Nov 3–6 | Most ECOD folders disappeared, including `representation_saves/` and `openfold_reps_outputs_sm/`. Cause unknown. |
| Nov 6, 15:52 | Only 5 top-level folders remained. `ecod/experiments/` was losing ~100–190 entries a minute. The rescue copy started. |
| Nov 7 | The source folder is gone entirely. Copying finished. |

No process from gprash or thayes was deleting files. gprash was only copying
`cameo` and `cameo_data`. The purge was probably done by an admin or the
system, but that isn't confirmed.

## Rescued: `/work/hdd/bfna/ikhire/thayes/`

| Path | What it is | Status |
|---|---|---|
| `FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD/` | All ablation TM-score tables (post-tri-end-fix): 14 layerwise FWD/REV files, structure-module scores, `MASTER_all_experiments_wide.csv` | ✅ Complete (65 files, 47 MB) |
| `FINAL_RESULT_CSVS_…/global/` | Global ablation tables: 13 conditions + baseline, 3-run mean per protein | ✅ Complete; also committed in git (see below) |
| `ecod/experiments/<ID>/` | Per-protein raw runs: layerwise and global ablations, `zero_m`/`zero_z`, baseline, representation-save run (PDBs + `timings.json`), `<ID>_tm_scores.csv` | ⚠️ Partial, see below |
| `ecod/experiments/<ID>/my_outputs_align_<ID>_representations_layers_save_0_47/predictions/<ID>_model_1_ptm_unrelaxed.pdb` | Unablated final predicted structure, the probes' distance target | ⚠️ 303 / 398 cohort proteins. SHA-256 identical to the originals the probes used |
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

## Lost (not found anywhere)

| Original path (all under `/work/hdd/bfna/thayes/`, later `toDelete/thayes/`) | What it was | Impact |
|---|---|---|
| `ecod/experiments/representation_saves/` | ECOD pair (and MSA) representations, all 48 layers | Cannot refit probes or run new representation analyses on ECOD. CAMEO pair blocks are safe. |
| `ecod/experiments/openfold_reps_outputs_sm/` | ECOD layer-47 predicted structures | Mostly replaced by the identical rescued PDBs above (303 / 398). |
| `ecod/alignments/` | ECOD MSAs and template hits | Rerunning ECOD inference needs new MSAs (about 6–12 CPU-hours per protein). Was already missing on Nov 3. |
| `ecod/experiments_TRI_END_FIX/` | Corrected ECOD ablation run outputs | Score tables survive; raw structures don't. |
| `ecod/domain_pdbs/` | Symlinks to ECOD domain reference PDBs | The rescued tarball covers this. |
| `ecod/{layerwise,global}_tm_csvs_TRI_END_FIX_STANDARDIZED/` | Per-source ECOD score tables | Superseded by the combined tables in `FINAL_RESULT_CSVS_…`. |
| `FINAL_RESULT_CSVS/`, `FINAL_RESULT_CSVS_TRI_END_FIX_STANDARDIZED/` | Older and standardized versions of the result tables | Low impact. The patched version used by the paper survives. |
| `ecod/experiments/<ID>/` for 87 cohort proteins | Raw ablation runs and predicted PDBs | 95 cohort proteins have no predicted PDB at all. |

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
2. Update the ECOD paths in `datasets.md`. The pair-block path is gone; the
   PDB path should point to the rescued `representations_layers_save_0_47`
   files.
3. Repoint code that uses the old ECOD PDB root:
   `scripts/cluster_scripts/diagnostics/distance_error_geometry_null/` (and
   any helpers using `ECOD_PDB_ROOT`). The `pdb_path` column in
   `predicted_distances_5fold_v1/metadata/proteins.csv` is stale; the
   checksums still match.
4. Ask thayes/gprash about ECOD `representation_saves/` and `alignments/`. The
   remaining leads are `/storage/cedar/cedar0/cedarp-gkrishnan46-0/ecod`
   (another cluster) and `/granite/nsf/access/bekh` (NCSA Granite tape,
   reachable through Globus). Neither has been checked yet.
5. Treat the 95 ECOD proteins without PDBs as lost unless recovered. Analyses
   that need reference structures run on 457 proteins (154 CAMEO + 303 ECOD).

## CAMEO

The full CAMEO catalogue, sizes, purge-risk evidence and the pending copy to
`/work/hdd/bfna/ikhire/thayes/cameo/` are in the
[cluster data preservation plan](../plans/cluster_data_preservation_plan.md).
Quick checks on Nov 7: all 154 CAMEO predicted PDBs
(`/work/hdd/bekh/thayes/openfold_reps_outputs_sm/`), layer-47 pair blocks
(`/work/hdd/bekh/thayes/openfold_reps/`) and alignments
(`/projects/bekh/thayes/alignments/`) are present.

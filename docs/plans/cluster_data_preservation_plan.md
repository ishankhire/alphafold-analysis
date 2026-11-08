# Cluster data preservation plan

Catalogue of project data still on Delta, and the plan to copy it into
`/work/hdd/bfna/ikhire/` before further cleanup. Surveyed November 7, 2026.
What has already been lost or rescued is in the
[Delta data inventory](../cluster/delta_data_inventory.md); the November 3
directory checks are in
[`miscellaneous/required_paths_2026-11-03/`](../../miscellaneous/required_paths_2026-11-03/README.md).

## Summary

- **Status:** tiers 1–3 **copied and verified** on Nov 7–8 by job
  `23649985`: 777.3 GiB and 293,280 files, with 0 mismatches. See the
  [Delta data inventory](../cluster/delta_data_inventory.md#cameo).
- **Scope:** CAMEO inputs/outputs and triangle attention maps, now under
  `/work/hdd/bfna/ikhire/thayes/cameo/`.
- **Space:** `/work/hdd/bfna` was at 2.87 TB before the copy and is now at
  3.63 TB, against a 4.88 TB soft / 5.37 TB hard quota. About 1.25 TB is free
  under the soft quota.
- **Not all critical data is in a `thayes/` folder.** thayes also owns
  top-level bfna folders, and gprash owns the CAMEO ablation PDB folders.

## Why the data is at risk

- **Purge lists:** `/work/hdd/bfna/bfna.sixmonth` (5.8M files) and
  `bfna.oneyear` (0.86M files), root-owned, dated October 12, 2026, list files
  untouched for six months or a year. `subfolders_remove.oneyear` (gprash,
  October 24) lists their folders. thayes's ECOD tree was on the six-month list
  and has since been purged, along with `/work/hdd/bfna/cameo/` and
  `/work/hdd/bfna/protein-cor/`.
- **Still present but on the six-month list:**
  `openfold_outputs_layerwise_ablation`, `openfold_outputs_fixed_ablation`,
  `cameo_dset`, `attention_map_dir_test_triplets`, `tylers_home_dir`, and
  `gprash/openfold_outputs_layerwise_ablation`.
- **bekh is over quota:** `/projects/bekh` is at 948 GB of 500 GB, and
  `/work/hdd/bekh` is at 1.04 TB of 1.00 TB. The CAMEO pair blocks, MSAs and
  PDBs all live in bekh.
- **Unknown timestamp basis:** it isn't known whether the lists are built from
  access time or modification time. Copies made with `rsync -a`/`-t` keep the
  2024–2025 modification times. Tell whoever runs the cleanup (likely gprash)
  that `ikhire/` is in active use.

## Catalogue

Sizes are disk usage from read-only job `23646162`. Its output is in
`/work/hdd/bfna/ikhire/storage_audit/du_entries.tsv`, with per-entry KiB and
inode counts.

### Tier 1: MSAs, reference structures, PDBs (copy first, ~30 GiB)

| Path | Contents | Size |
|---|---|---|
| `/projects/bekh/thayes/alignments` | CAMEO MSAs (`bfd_uniclust_hits.a3m`, `mgnify_hits.sto`, `uniref90_hits.sto`, `hhsearch_output.hhr`); all 154 cohort proteins present, plus 43 other entries | 4.3 GiB |
| `/projects/bekh/thayes/cameo_dset` | Experimental PDBs (`rcsb_pdb/`, `pdb_dir/`), FASTA files, annotations, lDDT scores | 15.8 GiB |
| `/work/hdd/bekh/thayes/openfold_reps_outputs_sm` | Per-layer predicted PDBs, 185 × 48 = 8,880 runs; layer 47 is the probes' distance target | 1.5 GiB |
| `/work/hdd/bfna/openfold_outputs_layerwise_ablation` (gprash) | CAMEO layerwise ablation PDBs (paper Figs. 3–5) | pending |
| `/work/hdd/bfna/openfold_outputs_global_ablations` (gprash) | CAMEO global ablation PDBs, 3 runs | 3.0 GiB |
| `/work/hdd/bfna/openfold_outputs_fixed_ablation` (thayes) | "keep layers X–Y" ablation PDBs | 0.2 GiB |
| `/work/nvme/bekh/openfold_outputs_layerwise_ablation` | Layerwise ablation PDBs for 10 proteins | ~1 GiB |
| `/work/hdd/bfna/gprash/cameo_pdbs` | gprash's per-layer CAMEO PDBs | 1.4 GiB |
| `openfold_reps_outputs`, `/work/nvme/bekh/thayes/cameo_reps_outputs` | Representation-save run outputs | <0.1 GiB |

### Tier 2: pair representations (386 GiB)

| Path | Contents | Size |
|---|---|---|
| `/work/hdd/bekh/thayes/openfold_reps` | CAMEO pair blocks `(R,R,128)` and MSA blocks, 48 layers, 184 proteins | 385.8 GiB |
| `/work/nvme/bekh/thayes/cameo_reps` | Pair blocks only; same proteins plus `subset_10/`. Spot-check: same byte sizes as hdd. **Skip as a duplicate** after a sample checksum comparison | 387.0 GiB |

### Tier 3: triangle attention maps (347 GiB)

| Path | Contents | Size |
|---|---|---|
| `/work/hdd/bekh/thayes/cameo_triplets` | Triangle attention outputs for 7b3a, 7dkk, 7dq9, 7ebt, 7tbs (+ masked TMU run) | 264.7 GiB |
| `/work/nvme/bekh/thayes/cameo_triplets` | `attn_full/`, `attn_random/` (includes full 7ebt maps) | 56.3 GiB |
| `/work/hdd/bekh/thayes/temp_store` | Tarballs for 7dkk, 7pq7, 7stt, 7tkv | 22.0 GiB |
| `/work/hdd/bfna/attention_map_dir_test_triplets` | MSA attention top-N CSVs and plots; `plots/` is unreadable to ikhire | 3.4 GiB |
| `/work/hdd/bekh/thayes/test_attn` | Demo triangle attention files (7dkk, 7tkv) | 1.0 GiB |

### Older CAMEO experiments (not in the copy; decide later, ~776 GiB)

The November 3 decision was that older representation and autoencoder studies
are not planned. These predate the paper's runs.

| Path | Contents | Size |
|---|---|---|
| `/projects/bekh/thayes/final_cameo_outputs` | 2025 ablation sweeps (~470 PDBs per protein), ~5,000 `*_update` runs, per-layer PDBs. A few runs per protein have `*_output_dict.pkl` (final-layer `pair`, `msa`, `single`, logits; up to 265 MB). Per-protein folders range from 67 MB to 15.8 GiB; the bulk is not yet characterized | 660.5 GiB |
| `/projects/bekh/thayes/output_dirs_cameo` | Sparsification/masking experiments, precomputed-alignment runs | 44.8 GiB |
| `/projects/bekh/thayes/cameo_ae_experiments` | Autoencoder checkpoints | 52.5 GiB |
| `/projects/bekh/thayes/cameo_reps` | Channel-wise analysis pickles | 14.2 GiB |
| `cameo_mlp_sequence_label`, `final_cameo_outputs_ae`, `final_cameo_plots`, `cameo_yaml` | Small analysis outputs | ~4 GiB |

### Other large data, not CAMEO (not in the copy)

| Path | Contents | Size |
|---|---|---|
| `/work/hdd/bfna/gprash/svd_shared_flatten_results` | gprash SVD analysis | 778.7 GiB |
| `/work/hdd/bekh/thayes/boltz2_reps` | Boltz-2 representations | 385.6 GiB |
| `/work/hdd/bfna/gprash/ablations_reps` | Representations from ablation runs | 342.0 GiB |
| `/projects/bekh/thayes/openproteinset/pdb_mmcif.zip` | PDB mmCIF archive (re-downloadable) | 51.8 GiB |
| `/projects/bekh/thayes/final_scope_outputs`, `scope_alignments` | SCOPE outputs and MSAs | 48.7 + 31.1 GiB |
| `/work/hdd/bfna/gprash/results_pca_subspace`, `repeated_pairs*` | gprash analyses | ~44 GiB |
| `/projects/bfna/vizfold_code` | thayes's VizFold code checkout | 22.5 GiB |
| `/u/ikhire/csv_files` | Legacy regression results (home quota, not purged) | 57.1 GiB |

### Not readable by ikhire

`/work/hdd/bfna/cameo_dset`, `/work/hdd/bfna/tylers_home_dir/thayes`,
`/work/hdd/bekh/thayes/ecod_dataset/data`,
`/projects/bfna/gprash/FINAL_RESULT_CSVS`, `/work/hdd/bfna/gprash/results_pca_grassmann`.
Only the owners can copy these; ask thayes and gprash.

## Copy job

- **Script:** [`scripts/cluster_scripts/data_rescue/copy_cameo.sbatch`](../../scripts/cluster_scripts/data_rescue/copy_cameo.sbatch),
  deployed to `/work/hdd/bfna/ikhire/cameo_copy/copy_cameo.sbatch`.
- **Destination:** `/work/hdd/bfna/ikhire/thayes/cameo/{work_hdd_bekh,work_nvme_bekh,projects_bekh,work_hdd_bfna}/<source name>`,
  next to the existing `logs_TRI_END_FIX/`.
- **Method:** one Slurm job on `bekh-delta-cpu` (16 CPUs, 32G, 24 h) running 8
  parallel `rsync -rlpt --no-owner --no-group` streams, tier by tier. Large
  folders are split into one task per child folder. Copy only; nothing at the
  source is changed. Re-running resumes.
- **Group handling:** files must not keep `delta_bekh`, because bekh's quota
  is full and group-charged writes could fail. New files inherit `delta_bfna`
  from the setgid destination.
- **Logs:** `/work/hdd/bfna/ikhire/cameo_copy/logs/<jobid>/`
  - `copy_results.tsv`: time, rsync exit code, source, seconds, destination
  - `rsync_errors.txt`
  - `verify.tsv`: source, pending transfers from a dry-run rsync, source and
    destination file counts, source and destination bytes
- **Success:** every `verify.tsv` row has `pending=0` and equal file counts.
- **Expected time:** a few hours; not yet measured on Delta. No symlinks exist
  in the sources.

Submit:

```bash
cd /work/hdd/bfna/ikhire/cameo_copy
sbatch --mail-user="${CLUSTER_NOTICE_EMAIL:?Set CLUSTER_NOTICE_EMAIL}" copy_cameo.sbatch
```

## Open items

1. ✅ Copy submitted (job `23649985`), verified, and recorded in the Delta
   data inventory. It took 5 h to copy (the layerwise ablation PDB folder,
   with hundreds of thousands of small files, dominated) and 6.8 h to verify.
2. Finish sizing the layerwise ablation folders (top-level and gprash copies);
   job `23646162` was still running at the time of writing.
3. Compare checksums on a sample of hdd vs nvme pair blocks before treating
   the nvme copy as redundant.
4. Decide on the older CAMEO experiments, after characterizing what fills
   `final_cameo_outputs`.
5. Ask thayes and gprash about the unreadable folders and the purge schedule.

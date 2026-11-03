# Important Delta paths and directory access

Update: the copy helper now uses the accessible singular-spelling source
`/work/hdd/bfna/toDelete/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD/`.
Spelling-issue descriptions below record the pre-fix audit. The user wants to
retain support for possible OpenFold/ablation reruns and attention/triplet work,
but does not plan older representation/autoencoder studies. See the current
[short summary](README.md) for retention decisions; raw access checks are unchanged.

Checked November 3, 2026, as `ikhire` through `ssh delta`.
The final check was simply `cd -- "$directory_path"` for each directory.

**88 directories checked: 60 accessible, 23 missing at the tested path, and
5 permission denied.** These are directory-access results, not a recursive file
inventory or a guarantee that every file inside is readable or complete.

- [Full keep list](important_directories.csv): 76 entries with purpose, original
  path, access result, and alternate path where found.
- [All unavailable directories](unavailable_directories.csv): all 28 failures
  with the actual shell error.
- [Raw directory checks](directory_checks.tsv).
- [Supporting directory listings and job evidence](remote_context.txt).

## Most urgent: preserve the relocated data

`/work/hdd/bfna/thayes/` no longer exists at its original location.
Important subdirectories are accessible under
**`/work/hdd/bfna/toDelete/thayes/`**. The name is a reason to flag this tree
for preservation; it does not establish whether or when anyone will delete it.

These current locations are accessible:

```text
/work/hdd/bfna/toDelete/thayes/ecod/experiments/representation_saves/
/work/hdd/bfna/toDelete/thayes/ecod/experiments/openfold_reps_outputs_sm/
/work/hdd/bfna/toDelete/thayes/ecod/experiments/
/work/hdd/bfna/toDelete/thayes/ecod/experiments_TRI_END_FIX/
/work/hdd/bfna/toDelete/thayes/ecod/domain_pdbs/
/work/hdd/bfna/toDelete/thayes/ecod/layerwise_tm_csvs_TRI_END_FIX_STANDARDIZED/
/work/hdd/bfna/toDelete/thayes/ecod/global_tm_csvs_TRI_END_FIX_STANDARDIZED/
/work/hdd/bfna/toDelete/thayes/FINAL_RESULT_CSVS/
/work/hdd/bfna/toDelete/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD/
/work/hdd/bfna/toDelete/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_STANDARDIZED/
```

The original locations are obtained by replacing `/bfna/toDelete/thayes/`
with `/bfna/thayes/`; each original location above failed its directory check.
Finding a corresponding directory does not verify byte-for-byte identity or
complete contents.

Current code still points to the original ECOD representation/PDB locations.
Cross-layer array `23482267` failed on missing ECOD inputs; dependent aggregation
`23482269` was cancelled. A representative traceback is in
`remote_context.txt`. No input paths or scientific code were changed here.

## Main directories to keep

Every directory in this table passed the final `cd` check.
Keep the original full-atom predicted PDBs, not just reconstructed C-alpha PDBs.

| Directory | Why keep it |
|---|---|
| `/work/hdd/bekh/thayes/openfold_reps/` | CAMEO baseline pair representations across all 48 layers; saved MSA blocks also support inference replay. |
| `/work/hdd/bekh/thayes/openfold_reps_outputs_sm/` | Original CAMEO predicted structures; layer 47 supplies the regression targets/references. |
| `/work/hdd/bfna/ikhire/regression_5fold_v1/` | Entire current canonical results collection, fitted models, frozen metadata, exports and provenance. |
| `/work/hdd/bfna/ikhire/openfold_interpretability_canonical/` | Active flat code/Slurm deployment, configuration, Linux TMscore executable and deployment manifest. |
| `/work/hdd/bfna/ikhire/diagnostic_experiments/` | Diagnostic experiments and their logs/results. |
| `/u/ikhire/csv_files/` | Legacy results, model-selection evidence, copied ablation CSVs and original pre-migration model outputs. |
| `/u/ikhire/openfold_interpretability_canonical/` | Preserved earlier runtime. |
| `/u/ikhire/protein_venv/` | Python runtime used by launchers. |
| `/u/ikhire/dssp_env/` | Secondary-structure annotation runtime. |
| `/projects/bekh/thayes/cameo_dset/rcsb_pdb/` | Experimental CAMEO reference structures for original ablation scoring. |
| `/projects/bekh/thayes/cameo_dset/fasta_dir/` | CAMEO sequences. |
| `/projects/bekh/thayes/alignments/` | CAMEO precomputed alignment/template-search inputs. |
| `/work/hdd/bekh/thayes/ecod_dataset/fasta_files/` | ECOD sequences. |
| `/work/hdd/bfna/openfold_outputs_layerwise_ablation/` | Accessible raw layerwise ablation outputs. |
| `/work/hdd/bfna/openfold_outputs_global_ablations/` | Accessible raw global ablation outputs. |
| `/work/hdd/bfna/openfold_outputs_fixed_ablation/` | Fixed-ablation outputs and provenance. |
| `/work/hdd/data/alphafold2/database/` | Shared inference/search/template database, including `pdb_mmcif/mmcif_files/`. |

The full CSV also includes older representation/AE stores, attention studies,
migration records and the relocated ablation tables. Those study-specific
outputs are kept for their respective analyses; they are not all inputs to the
four current regression probes. The PDF's attention example
`/work/nvme/bekh/thayes/cameo_triplets/attn_full/7ebt_B/` is accessible.

### Important files within those directories

The frozen analysis has 552 eligible proteins; preserve the original
554-protein input pool as well for historical ablation reproducibility.
All pair layers 0–47 matter, rather than only the final layer.

Pair/PDB filename patterns:

```text
/work/hdd/bekh/thayes/openfold_reps/{PROT}/{PROT}_pair_block_{LAYER}.npy
/work/hdd/bfna/toDelete/thayes/ecod/experiments/representation_saves/{PROT}_save_layers_0_47/{PROT}_pair_block_{LAYER}.npy
/work/hdd/bekh/thayes/openfold_reps/{PROT}/{PROT}_msa_block_{LAYER}.npy
/work/hdd/bfna/toDelete/thayes/ecod/experiments/representation_saves/{PROT}_save_layers_0_47/{PROT}_msa_block_{LAYER}.npy
/work/hdd/bekh/thayes/openfold_reps_outputs_sm/my_outputs_align_{PROT}_representations_load_from_saved_layer_47/predictions/{PROT}_model_1_ptm_unrelaxed.pdb
/work/hdd/bfna/toDelete/thayes/ecod/experiments/openfold_reps_outputs_sm/my_outputs_align_{PROT}_representations_load_from_saved_layer_47/predictions/{PROT}_model_1_ptm_unrelaxed.pdb
```

Supported loader alternatives include `pair_blocks/` inside representation
directories and colocated `predicted_pdb/` files. Preserve whichever exact
files were used. These patterns were derived from the code; individual files
were not exhaustively checked.

Under `/work/hdd/bfna/ikhire/regression_5fold_v1/`, keep:

```text
metadata/regression_outer_folds_v1.csv
metadata/per_protein_pair_folds_v1.csv
metadata/fold_digests.json
shared_ols_base256/
per_protein_ols_base256/
sequence_gated_ols_base256/
shared_xgboost_base256_fixed_s4/
triangle_violation_subgroups_5fold_v1/
analysis/publication_subgroup_mae_5fold_v1/
predicted_distances_5fold_v1/
tm_score_optimized_v1/
_migration_v1/
collection_manifest.json
```

Inside each model family retain the core CSVs and `run_manifest.json`,
`subgroup_metrics/`, `model_artifacts/replay_index.csv`, the indexed
coefficients/scalers or boosters, and validated `layer_shards/`. Core files
include `per_protein.csv`, `pooled_metrics.csv`, `fold_metrics.csv`,
`fold_summary.csv`, `training_metrics.csv`, `stable_fit_metrics.csv`,
`model_status.csv`, `runtime.csv`, `geometry_metrics.csv`,
`geometry_summary.csv`, and `model_stability.csv` where applicable.
Do not retain only the plots and discard the fitted models.

The packed-distance bundle needs its arrays, `prediction_index.csv`,
`metadata/proteins.csv`, `metadata/residue_index.csv`, validation tables and
manifests. The optimized TM-score bundle needs its reconstructed PDBs,
`pdb_index.csv`, `tm_scores.csv`, geometry replacements and manifests.

Under the active runtime retain `regression_protocol_data/regression_master_split_v1.csv`,
`TMscore`, `deployment_manifest.json`, deployed scripts/configs, and
`deployment_history/recovery_de3827c_20261024/`. Preserve migration records
including `_migration_v1/cancellation_23162599.json` in the results root and
the separate `/work/hdd/bfna/ikhire/.regression_5fold_v1_migration_v1/` staging
tree. Existing migration/access completion statuses remain unchanged.

Retain all compact ablation CSVs under the three accessible relocated
`FINAL_RESULT_CSVS*` roots, including their `global/` children, master tables
and repair provenance; copied layerwise inputs remain under
`/u/ikhire/csv_files/layerwise_ablation_csvs/`. The fixed-XGBoost selection
evidence is under `/u/ikhire/csv_files/xgboost_hyperparameter_search/`.

## Missing at the tested paths

These are all 23 directories that returned “No such file or directory”:

```text
/work/hdd/bekh/openfold_outputs_global_ablations
/work/hdd/bekh/openfold_outputs_layerwise_ablation
/work/hdd/bfna/cameo/global_tm_csvs
/work/hdd/bfna/cameo/global_tm_csvs_TRI_END_FIX
/work/hdd/bfna/cameo/layerwise_tm_csvs
/work/hdd/bfna/cameo/layerwise_tm_csvs_TRI_END_FIX
/work/hdd/bfna/cameo/openfold_outputs_layerwise_ablation
/work/hdd/bfna/openfold_outputs_global_ablations_TRI_END_FIX
/work/hdd/bfna/openfold_outputs_layerwise_ablation_TRI_END_FIX
/work/hdd/bfna/thayes/FINAL_RESULTS_CSVS_TRI_END_FIX_PATCHED_FROM_OLD
/work/hdd/bfna/thayes/FINAL_RESULT_CSVS
/work/hdd/bfna/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD
/work/hdd/bfna/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_STANDARDIZED
/work/hdd/bfna/thayes/ecod/alignments
/work/hdd/bfna/thayes/ecod/domain_pdbs
/work/hdd/bfna/thayes/ecod/experiments
/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm
/work/hdd/bfna/thayes/ecod/experiments/representation_saves
/work/hdd/bfna/thayes/ecod/experiments_TRI_END_FIX
/work/hdd/bfna/thayes/ecod/global_tm_csvs_TRI_END_FIX_STANDARDIZED
/work/hdd/bfna/thayes/ecod/layerwise_tm_csvs_TRI_END_FIX_STANDARDIZED
/work/hdd/bfna/toDelete/thayes/FINAL_RESULTS_CSVS_TRI_END_FIX_PATCHED_FROM_OLD
/work/hdd/bfna/toDelete/thayes/ecod/alignments
```

Most old `/bfna/thayes/` entries have an accessible counterpart under
`/bfna/toDelete/thayes/`, as listed above. Remaining priorities:

- **ECOD alignments:** missing at both tested roots. They are needed for
  original inference/ablation reproduction, although regression fitting reads
  the already-saved pair tensors and predicted PDBs directly. Ask about other
  copies before concluding they were deleted.
- **Old ablation roots and CAMEO score folders:** missing at the exact locations
  in the PDF/archive. Other ablation directories and combined score tables
  survive; this check did not establish whether their contents are equivalent.
- **Spelling mismatch:** `copy_layerwise_ablation_csvs.sh` names
  `FINAL_RESULTS_CSVS_TRI_END_FIX_PATCHED_FROM_OLD` (plural RESULTS).
  The accessible relocated directory is
  `FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD` (singular RESULT).
  This could be a stale path/spelling problem; do not infer a separate lost
  dataset or silently substitute one for the other.

## Permission denied

These five paths returned “Permission denied”:

```text
/u/thayes/vizfold
/work/hdd/bekh/thayes/ecod_dataset/data
/work/hdd/bekh/thayes/ecod_dataset/data/ecod/domain_data
/work/hdd/bfna/tylers_home_dir/thayes
/work/hdd/bfna/tylers_home_dir/thayes/vizfold
```

The accessible ECOD `domain_pdbs/` directory contains symlinks named
`{PROT}.pdbnum.pdb` pointing into
`/work/hdd/bekh/thayes/ecod_dataset/data/ecod/domain_data/`. For example:

```text
/work/hdd/bfna/toDelete/thayes/ecod/domain_pdbs/000000022.pdbnum.pdb
  -> /work/hdd/bekh/thayes/ecod_dataset/data/ecod/domain_data/00000/000000022/000000022.pdbnum.pdb
```

Preserve both the links and the underlying data. Directory access to
`domain_pdbs/` does not imply access to the targets; permissions currently
block verification. Likewise, Tyler's original/saved VizFold checkout is
unverified below the blocked parent. Its code, selected-protein lists, settings
and pretrained weights (normally
`openfold/resources/params/params_model_1_ptm.npz`) matter for regenerating the
OpenFold inputs. The local `vizfold-main.zip` contains a code snapshot, but
its existence does not establish that pretrained weights are backed up.

The PDF explicitly retires `/ime/hdd/rhaas/SUP-5301/database` in favor of
`/work/hdd/data/alphafold2/database`; the retired path is not a recovery target.

## Scope and cluster changes

Sources reviewed were the provided eight-page PDF, `datasets.md`,
`docs/cluster/README.md`, current status/output documentation, shared path
resolvers, canonical migration/storage code, launchers, plotting input paths
and relevant inference/ablation scripts in `vizfold-main.zip`. Source scanning
covered 227 checkout code files and 751 archived code/notebook files, plus
documentation. Historical references were used as leads, not proof of current use.
This is a list of important known paths, not every directory on Delta.

Before the request to simplify, a bulk collector was started. Both audit jobs
**23484057 and 23484189 were cancelled**. No full per-file audit was completed,
and none of its partial counts are used here.

The only remote additions were these three prototype collector files under
`/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`:

```text
audit_delta_required_paths.py
audit_delta_required_paths.sbatch
required_path_audit_README.md
```

The audit input and incomplete outputs remain under
`/work/hdd/bfna/ikhire/diagnostic_experiments/required_path_audit_2026-11-03/`
(`requests.csv.gz`, `results/`, `results_metadata/`). Its logs remain at
`required_path_audit-{23484057,23484189}.{out,err}` in the deployment.
These are audit artifacts, not additional scientific inputs to preserve.

No cluster files were deleted, moved, or repaired; research data, permissions,
production launchers and migration statuses were not changed.

NCSA's [storage documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html)
states that WORK-HDD has no snapshots. For unresolved missing WORK paths,
confirm relocation/other copies with the owner or NCSA before assuming recovery
is available.

# Missing Delta paths to preserve or locate

Based on the November 3, 2026 directory checks. “Missing” means `cd` returned
“No such file or directory” at that path; it does not prove deletion. Individual
file contents were not audited. The inventory has moved out of `docs/`.

## Missing, without a verified accessible replacement

ECOD alignments (needed for regenerating OpenFold inputs and original ablations):

```text
/work/hdd/bfna/thayes/ecod/alignments/
```

The alternate `/work/hdd/bfna/toDelete/thayes/ecod/alignments/` also failed.

Raw ablation outputs (needed to revisit structures or recompute scores):

```text
/work/hdd/bekh/openfold_outputs_layerwise_ablation/
/work/hdd/bekh/openfold_outputs_global_ablations/
/work/hdd/bfna/cameo/openfold_outputs_layerwise_ablation/
/work/hdd/bfna/openfold_outputs_layerwise_ablation_TRI_END_FIX/
/work/hdd/bfna/openfold_outputs_global_ablations_TRI_END_FIX/
```

The two `/work/hdd/bekh/openfold_outputs_*` roots are referenced in the current
checkout only by `scripts/cluster_scripts/diagnostics/inventory_delta_storage.py`.
The archived `vizfold-main.zip` contains actual consumers/producers: layerwise
`slurm_scripts_cameo/compute_tm_scores_LAYER_UPDATE_subset_MIN.sh` and
`inference_general_per_layer_ablate_flags_LAYER_UPDATE_MIN_SET.sh`, and global
`slurm_scripts_cameo/compute_tm_scores_GLOBAL_UPDATE.py` and
`compute_tm_scores_GLOBAL_UPDATE_BASH.sh`. They remain recovery candidates for
the possible future ablation work; current regression/plotting code does not
directly reference these two roots.

Per-source ablation score tables (preserve for provenance):

```text
/work/hdd/bfna/cameo/layerwise_tm_csvs/
/work/hdd/bfna/cameo/global_tm_csvs/
/work/hdd/bfna/cameo/layerwise_tm_csvs_TRI_END_FIX/
/work/hdd/bfna/cameo/global_tm_csvs_TRI_END_FIX/
```

Other ablation roots, combined tables and local copied inputs survive; their
existence does not establish that all missing material is duplicated there.
These ten paths are recovery/preservation leads, not ten established losses of
inputs required by the current regression fitting or publication plotting.

## Original paths missing, corresponding directories found elsewhere

For every path below, replace `/work/hdd/bfna/thayes/` with
**`/work/hdd/bfna/toDelete/thayes/`** to obtain the accessible counterpart:

```text
/work/hdd/bfna/thayes/ecod/experiments/representation_saves/
/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm/
/work/hdd/bfna/thayes/ecod/experiments/
/work/hdd/bfna/thayes/ecod/experiments_TRI_END_FIX/
/work/hdd/bfna/thayes/ecod/domain_pdbs/
/work/hdd/bfna/thayes/ecod/layerwise_tm_csvs_TRI_END_FIX_STANDARDIZED/
/work/hdd/bfna/thayes/ecod/global_tm_csvs_TRI_END_FIX_STANDARDIZED/
/work/hdd/bfna/thayes/FINAL_RESULT_CSVS/
/work/hdd/bfna/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD/
/work/hdd/bfna/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_STANDARDIZED/
```

Preserve the accessible tree under `toDelete`. The first two original paths
are still referenced by current regression/replay code. Corresponding directory
names do not establish complete or byte-identical contents.

The copy helper's spelling/path issue is fixed: its default source is now
`/work/hdd/bfna/toDelete/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD/`.
The original directory-check files retain the earlier failed plural path as
historical evidence; it is not an additional recovery target.

## Permission blocked, rather than confirmed missing

```text
/work/hdd/bekh/thayes/ecod_dataset/data/
/u/thayes/vizfold/
/work/hdd/bfna/tylers_home_dir/thayes/
```

The first includes the underlying ECOD experimental-PDB symlink targets.
The other two may hold original inference code, selected-protein lists and
pretrained weights. Child paths also failed; avoid counting these as separate
deletions. The current canonical regression results and runtime are accessible.

## Confirmed future-work retention choices

User choices recorded November 3: possible OpenFold/ablation reruns and
attention/triplet work should remain supported. Older representation and
autoencoder experiments are not planned.

| Future work | Retention decision |
|---|---|
| Regenerate OpenFold representations or run new ablations — might | **Keep/recover:** ECOD alignments above, original inference code and weights, sequences/templates, original PDBs, all baseline pair/MSA layers and raw ablation outputs. |
| Attention/triplet analyses — might | **Keep:** `/work/nvme/bekh/thayes/cameo_triplets/attn_full/7ebt_B/` and `/work/hdd/bekh/thayes/cameo_triplets/`. Both directory checks passed. |
| Older representation or autoencoder experiments — no | **Not required for the user's planned work:** `/projects/bekh/thayes/final_cameo_outputs/`, `/projects/bekh/thayes/final_cameo_outputs_ae/`, `/work/nvme/bekh/thayes/cameo_reps/`, `/work/hdd/bekh/thayes/openfold_reps_outputs/`. This does not designate their contents as safe to delete or affect the baseline inputs above. |

No cleanup or deletion is authorized by this list.

[Complete keep list](important_directories.csv) ·
[All failed directory checks](unavailable_directories.csv) ·
[Detailed report and audit provenance](DETAILED_REPORT.md)

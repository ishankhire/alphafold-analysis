# Per-channel classical MDS of the pair representation (CAMEO)

How Euclidean is the pair representation itself, with no probe? Each of the 128 channels is
treated as a dissimilarity matrix and put through the classical MDS the probe families use.
This is a descriptive diagnostic: a channel is one axis of an arbitrary basis, so means over
channels are not a basis-invariant property of the representation. The permuted null, the
correlation columns and the reference rows are what make the numbers interpretable.

## Design

- **Proteins:** the 154 CAMEO proteins of the frozen 552-protein cohort. **Layers:** 0–47.
- **Channel matrix**, for pair block `Z` of shape `(R, R, 128)` and channel `c`:
  1. `S = (Z[:,:,c] + Z[:,:,c]ᵀ) / 2`; the diagonal is ignored.
  2. For each sign `s ∈ {+1, −1}`: `D = s·S − min(s·S)` over off-diagonal entries, diagonal 0.
  3. Classical MDS in 3 dimensions, exactly `cluster_regression_helpers.classical_mds`.
  4. The sign with the lower stress is kept (`best_sign`); the other sign's stress is saved.
- **Metrics**, the `geometry_metrics.csv` formulas:
  - `stress = sqrt(Σ(D − D_embedded)² / ΣD²)` over unordered pairs;
  - `frac_top3_eigenvalues`, `frac_top1_eigenvalue`: top non-negative eigenvalues over the
    positive eigenvalue mass;
  - `n_negative_eigenvalues`, `negative_eigenvalue_fraction = |Σλ⁻| / Σλ⁺`.
- **Null:** the channel's upper-triangle values in a random order (one permutation, seed
  `[20261118, layer, protein_index, channel]`, `protein_index` from the sorted 552 cohort),
  then steps 2–4. Saved in the `perm_*` columns.
- **Interpretation columns**, on the upper triangle of `S`:
  - `spearman_sequence_separation`: Spearman with `|i − j|`;
  - `spearman_true_distance`: Spearman with the final-structure Cα distance;
  - `embedding_true_distance_pearson`: Pearson between the kept embedding's distances and the
    final-structure distances;
  - `antisymmetric_fraction = ΣA² / (ΣA² + Σ(S − mean S)²)` with `A = (Z − Zᵀ)/2`;
  - `offdiag_mean`, `offdiag_std` of `S`.
- **Constant channels** get `status = constant` and blank metrics.
- **Reference rows** (per protein, made by the aggregation job): the final-structure
  distances and `|i − j|` through the same MDS, unshifted (`raw`) and min-shifted
  (`minshift`). The `minshift` row of the true distances is the stress the shift rule alone
  costs a channel that is exactly affine in distance.
- **Probe numbers are not recomputed.** Compare with the CAMEO rows of
  `regression_5fold_v1/<family>/geometry_metrics.csv` (clipped OOF distances, no shift).
- **Gates:** each layer must pass these before anything is written. A failure writes only
  `validation/layer_LL_FAILED.json`.
  1. The cohort is the 154 CAMEO proteins; each PDB's Cα count equals `n_residues`.
  2. Every pair block is finite and `(R, R, 128)`.
  3. On channel 0 of every protein the batched MDS reproduces `helpers.classical_mds` and the
     canonical formulas within 1e-8.
  4. Every non-constant channel has a finite stress ≥ 0, real and permuted.

## Deploy

Copy flat into `/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`:

- from this directory: `per_channel_mds_common.py`, `per_channel_mds.py`,
  `aggregate_per_channel_mds.py`, both `.sbatch` files, and this README as
  `per_channel_mds_README.md`.
- These are already deployed; check that their checksums match the repository:
  - `null_triangles_common.py` and `null_triangles_replay.py` from `../../null_triangles/`;
  - `cluster_regression_helpers.py`, `regression_protocol.py`, `regression_output_helpers.py`
    and `canonical_storage.py` from `../..`;
  - `triangle_violation_subgroup_common.py` from `../../triangle_violation_subgroups/`;
  - `regression_protocol_data/`.

## Inputs and outputs

- **Inputs:**
  - `$CSV_ROOT/regression_5fold_v1/metadata/regression_outer_folds_v1.csv`;
  - CAMEO pair blocks via `resolve_combined_pair_block_path`
    (`/work/hdd/bekh/thayes/openfold_reps/<id>/<id>_pair_block_<layer>.npy`);
  - final PDBs via `resolve_combined_pdb_path`.
- **Output root:** `PER_CHANNEL_MDS_OUTPUT_ROOT`, default
  `/work/hdd/bfna/ikhire/diagnostic_experiments/per_channel_mds_cameo_v1/`.
- **Slurm logs:** `<default root>/logs/`, which must exist before submission.

| Output | Row | Principal columns |
|---|---|---|
| `layer_shards/channel_metrics_layer_LL.csv.gz` | protein × channel; 19,712 rows per layer | `status`, `best_sign`, `stress`, `stress_other_sign`, `frac_top1_eigenvalue`, `frac_top3_eigenvalues`, `n_negative_eigenvalues`, `negative_eigenvalue_fraction`, `perm_best_sign`, `perm_stress`, `perm_frac_top3_eigenvalues`, `perm_n_negative_eigenvalues`, `perm_negative_eigenvalue_fraction`, `spearman_sequence_separation`, `spearman_true_distance`, `embedding_true_distance_pearson`, `antisymmetric_fraction`, `offdiag_mean`, `offdiag_std` |
| `validation/layer_LL.json` | layer | gate results, settings, timings, peak RSS, inputs; marks the layer complete |
| `tables/channel_metrics.csv.gz` | all shards concatenated; 946,176 rows | as the shard |
| `tables/protein_layer_summary.csv` | protein × layer; 7,392 rows | `n_channels_ok`, `best_channel`, and `mean`, `var`, `min`, `p10`, `median`, `max` over channels of `stress`, `negative_eigenvalue_fraction`, `frac_top3_eigenvalues` and their `perm_` versions |
| `tables/reference_metrics.csv` | protein × reference × shift; 616 rows | the geometry metrics and `embedding_true_distance_pearson` |
| `tables/runtime.csv` | layer | `elapsed_seconds`, `protein_seconds_sum`, `load_seconds_sum`, `peak_rss_mib`, `workers` |
| `tables/run_manifest.json` | — | status, settings, checks, row counts, `code_revision` |

## Submit

```bash
cd /work/hdd/bfna/ikhire/openfold_interpretability_canonical
root=/work/hdd/bfna/ikhire/diagnostic_experiments/per_channel_mds_cameo_v1
mkdir -p $root/logs
export PER_CHANNEL_MDS_CODE_REVISION=<git commit of the deployed files>

array=$(sbatch --parsable --mail-user="${CLUSTER_NOTICE_EMAIL:?}" per_channel_mds.sbatch)
sbatch --mail-user="$CLUSTER_NOTICE_EMAIL" --dependency=afterok:${array%%;*} aggregate_per_channel_mds.sbatch
```

The array is `0-47%16`, one layer per task, 16 CPUs, parallel over proteins within a task
(largest first, channels in chunks of 8). A layer whose `validation/layer_LL.json` exists is
skipped, so rerunning the array only redoes missing layers.

Copy `tables/` from the Delta root to the local
`delta_files/diagnostic_experiments/per_channel_mds_cameo_v1/tables/`. Cohort statistics,
text and figures are made locally by
`scripts/multi_protein/per_channel_mds/analyze_per_channel_mds.py`.

## Local smoke test

```bash
export PER_CHANNEL_MDS_OUTPUT_ROOT=<scratch dir>
export PER_CHANNEL_MDS_LOCAL_DATA_ROOT=$(cd ../protein_data && pwd)
export NULL_TRIANGLES_PROTEINS=6tf4_A,6wmk_A,6wqc_A,6zpp_A,7ab3_E,7ar0_B,7atr_A,7b3a_A
PER_CHANNEL_MDS_LAYER=47 ./.venv/bin/python scripts/cluster_scripts/diagnostics/per_channel_mds/per_channel_mds.py
PER_CHANNEL_MDS_AGGREGATE_LAYERS=47 ./.venv/bin/python scripts/cluster_scripts/diagnostics/per_channel_mds/aggregate_per_channel_mds.py
```

Unit tests: `./.venv/bin/python -m pytest scripts/multi_protein/tests/test_per_channel_mds.py`.

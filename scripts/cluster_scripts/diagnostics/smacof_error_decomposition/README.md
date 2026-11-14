# SMACOF error decomposition (552-protein cohort)

How much of a probe's distance error is geometrically inconsistent, and how much is a
consistent error about a real 3D shape? Classical MDS does not minimise stress, so each
decoded distance matrix is fitted here with metric SMACOF, and the error against OpenFold's
final structure is split in two. The triangle experiment's two matched-noise controls go
through the same pipeline. This is a diagnostic experiment; the classical-MDS `stress` in
the four families' `geometry_metrics.csv` is not changed.

## Design

- **Proteins:** the frozen 552-protein cohort, 154 CAMEO + 398 ECOD, treated as one dataset.
- **Probes:** `shared_ols_base256`, `per_protein_ols_base256`, `sequence_gated_ols_base256`
  and `shared_xgboost_base256_fixed_s4`, at layers 0–47. Scope: OOF, all pairs.
- **Matrices:** per probe and protein, 7 per layer:
  - `observed`: the cached OOF decoded distances;
  - `control_gaussian`: final-structure distances plus Gaussian noise matched per B10 bin, 3 draws;
  - `control_shuffled`: final-structure distances plus the probe's own errors shuffled within B10 bin, 3 draws.

  The controls are the triangle experiment's Null A and Null B on B10. They are rebuilt by
  `null_triangles_replay.build_vectors` from the same seeds
  `(20261109, 1, layer, probe, protein, model, B10, draw)`, predictions and structures, so
  each control matrix is the one the triangle tables scored.
- **Clipping:** negatives are clipped to zero (`np.maximum`), as in `analyze_protein_geometry`.
  Nothing is rescaled or normalised.
- **SMACOF:** metric, 3 dimensions, raw unweighted stress on distances in Å, one start from
  that matrix's own classical-MDS coordinates. `smacof_error_decomposition_common.smacof_metric`
  is a plain Guttman-transform loop. A run has converged when one iteration lowers raw stress by
  at most `SMACOF_REL_TOL` times its value (default 1e-10), with at most `SMACOF_MAX_ITER`
  iterations (default 20,000). The iteration count and a converged flag are saved, and so are
  the terms at the moment each of the tolerances 1e-6, 1e-8 and 1e-10 was first met.
- **Decomposition**, means over unordered pairs, with `d̂` the clipped decoded distances, `p`
  the SMACOF distances and `d` the final-structure distances:

  | Column | Definition |
  |---|---|
  | `total_sq` | mean (d̂ − d)² |
  | `smacof_inconsistent_sq` | mean (d̂ − p)² |
  | `smacof_consistent_sq` | mean (p − d)² |
  | `smacof_cross` | 2 · mean (d̂ − p)(p − d) |

  `total_sq = inconsistent + consistent + cross` exactly. The same split with `p` from the
  classical-MDS start is saved in the `classical_*` columns.
- **Gates:** each layer must pass these before anything is written. A failure writes only
  `validation/layer_LL_FAILED.json`.
  1. The identity holds on every row to 1e-9 relative, for both splits.
  2. SMACOF raw stress is no higher than the classical-MDS raw stress it started from.
  3. Every rebuilt vector has the `n_negative_values` the triangle run saved; on proteins with
     ≤ 1M triangles it also has the same `n_violations`. The shuffled control keeps every bin's
     error histogram.
  4. The classical start reproduces the canonical `geometry_metrics.csv` values within the
     float32-cache tolerance: normalized stress within 1e-4, RMSD within 0.01 Å,
     negative-prediction count within 2 (`classical_geometry_parity_5fold_v1.csv.gz`).
  5. The cohort is 154 CAMEO + 398 ECOD; each PDB's Cα count equals `n_residues`; prediction
     checksums match `prediction_index.csv`.
- **Not gated, saved:** `smacof_converged`, and `smacof_scale_ratio = Σ d̂·p / Σ p²`, which is 1
  at a stress minimum.

**Parity table.** `classical_geometry_parity_5fold_v1.csv.gz` holds `analysis_name, layer,
protein_id, stress, rmsd, n_negative_predictions` from the four local
`delta_files/regression_5fold_v1/<family>/geometry_metrics.csv` files (105,984 rows), sorted by
layer, protein and family.

## Deploy

Copy flat into `/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`:

- from this directory:
  - `smacof_error_decomposition_common.py`, `smacof_error_decomposition.py`,
    `aggregate_smacof_error_decomposition.py`;
  - both `.sbatch` files;
  - `classical_geometry_parity_5fold_v1.csv.gz`;
  - this README, as `smacof_error_decomposition_README.md`.
- These are already deployed; check that their checksums match the repository:
  - `null_triangles_common.py` and `null_triangles_replay.py` from `../../null_triangles/`;
  - `cluster_regression_helpers.py`, `regression_protocol.py`, `regression_output_helpers.py`
    and `canonical_storage.py` from `../..`;
  - `triangle_violation_subgroup_common.py` from `../../triangle_violation_subgroups/`;
  - `regression_protocol_data/` and the `TMscore` executable.

## Inputs and outputs

- **Inputs:**
  - `$CSV_ROOT/regression_5fold_v1/metadata/regression_outer_folds_v1.csv`;
  - `$CSV_ROOT/regression_5fold_v1/predicted_distances_5fold_v1/layers/layer_LL/`;
  - final PDBs via `resolve_combined_pdb_path`;
  - `$CSV_ROOT/main_experiments/null_triangles_5fold_v1/layer_shards/null_metrics_layer_LL.csv.gz` (gate 3).
- **Output root:** `SMACOF_DECOMPOSITION_OUTPUT_ROOT`, default
  `/work/hdd/bfna/ikhire/diagnostic_experiments/smacof_error_decomposition_5fold_v1/`.
- **Slurm logs:** `<default root>/logs/`, which must exist before submission.

| Output | Row | Principal columns |
|---|---|---|
| `layer_shards/decomposition_layer_LL.csv.gz` | probe × protein × `matrix_type` × draw (draw −1 = observed); 15,456 rows per layer | the four terms above, `smacof_identity_residual`, `smacof_raw_stress`, `classical_raw_stress`, `smacof_normalized_stress`, `classical_normalized_stress`, `smacof_scale_ratio`, `smacof_iterations`, `smacof_converged`, `smacof_final_relative_decrease`, `smacof_{iterations,inconsistent_sq,consistent_sq}_at_{1e-6,1e-8,1e-10}`, `classical_{inconsistent_sq,consistent_sq,cross}`, `smacof_rmsd`, `classical_rmsd`, `smacof_tm_score` (observed only), `n_negative_values` |
| `layer_shards/bin_sums_layer_LL.csv.gz` | probe × protein × `matrix_type` × `bin_system` × occupied bin; control draws averaged | `n_pairs`, `n_draws`, `sum_total_sq`, `sum_smacof_inconsistent_sq`, `sum_smacof_consistent_sq`, `sum_classical_inconsistent_sq`, `sum_classical_consistent_sq` (cross sum = total − inconsistent − consistent) |
| `layer_shards/smacof_coordinates_layer_LL.npz` | observed matrices; key `<protein_id>/<probe>` | float32 `(R, 3)` SMACOF coordinates, Kabsch-aligned to the final structure |
| `validation/layer_LL.json` | layer | gate results, convergence counts, snapshot comparison, timings, inputs; marks the layer complete |
| `tables/decomposition_per_draw.csv.gz` | all shards concatenated | as the shard |
| `tables/decomposition_per_protein.csv.gz` | probe × layer × protein × `matrix_type` | draw means and draw SDs of the terms, `n_draws`, `n_unconverged`, RMSD, TM-score |
| `tables/bin_sums_by_sequence_separation.csv.gz`, `tables/bin_sums_by_true_distance.csv.gz` | per-protein bin sums | as the shard |
| `tables/bin_terms_cohort.csv` | probe × layer × `matrix_type` × bin | protein-equal means of per-bin mean squared terms over proteins with ≥ 10 pairs |
| `tables/convergence_summary.csv`, `tables/run_manifest.json` | — | — |

The bin systems are `sequence_separation` (1, 2-4, 5-9, 10-19, 20-49, 50+) and
`true_distance_b10` (the triangle experiment's B10 bins on final-structure distance).

## Submit

```bash
cd /work/hdd/bfna/ikhire/openfold_interpretability_canonical
root=/work/hdd/bfna/ikhire/diagnostic_experiments/smacof_error_decomposition_5fold_v1
mkdir -p $root/logs
export SMACOF_DECOMPOSITION_CODE_REVISION=<git commit of the deployed files>

# 1. Gate rehearsal: zero SMACOF iterations, no TM-score. Checks gates 3-5 in minutes.
SMACOF_DECOMPOSITION_OUTPUT_ROOT=$root/gate_check SMACOF_MAX_ITER=0 SMACOF_DECOMPOSITION_SKIP_TM_SCORE=1 \
  sbatch --mail-user="${CLUSTER_NOTICE_EMAIL:?}" --array=0,47 --time=00:30:00 smacof_error_decomposition.sbatch

# 2. Pilot: two full layers at the tightest tolerance. Read validation/layer_LL.json for the
#    snapshot comparison, convergence counts, time and memory, then choose SMACOF_REL_TOL.
SMACOF_DECOMPOSITION_OUTPUT_ROOT=$root/pilot \
  sbatch --mail-user="$CLUSTER_NOTICE_EMAIL" --array=0,47 smacof_error_decomposition.sbatch

# 3. Full run at the chosen tolerance, then aggregation.
array=$(SMACOF_REL_TOL=<chosen> sbatch --parsable --mail-user="$CLUSTER_NOTICE_EMAIL" smacof_error_decomposition.sbatch)
sbatch --mail-user="$CLUSTER_NOTICE_EMAIL" --dependency=afterok:${array%%;*} aggregate_smacof_error_decomposition.sbatch
```

The array is `0-47%16`, one layer per task, parallel over proteins within a task. A layer
whose `validation/layer_LL.json` exists is skipped, so rerunning the array only redoes
missing layers. Aggregation refuses to mix layers run with different SMACOF settings.

Copy `tables/` from the Delta root to the local
`delta_files/diagnostic_experiments/smacof_error_decomposition_5fold_v1/tables/`. Bootstraps,
text and figures are then made locally by
`scripts/multi_protein/smacof_error_decomposition/analyze_smacof_error_decomposition.py`.

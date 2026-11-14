# SMACOF Error Decomposition Plan (`smacof_error_decomposition_5fold_v1`)

## 1. Goal and fixed scope

Classical MDS does not minimise stress, so its residual overstates how far a probe's decoded
distance matrix is from a real 3D structure. This diagnostic adds a metric SMACOF
reconstruction and uses it to split each probe's distance error into an **inconsistent**
part (decoded distances against their best 3D fit) and a **consistent** part (best 3D fit
against OpenFold's final structure). The triangle experiment's two matched-noise controls go
through the identical pipeline.

Scope: OOF all pairs, the four probe families, layers 0–47, the frozen 552-protein cohort.
The classical-MDS `stress` in `geometry_metrics.csv` is not changed. The runbook, including
the full output schema, is
[`scripts/cluster_scripts/diagnostics/smacof_error_decomposition/README.md`](../../scripts/cluster_scripts/diagnostics/smacof_error_decomposition/README.md).

## 2. Method

- `d̂`: decoded distances after clipping negatives to zero (as the geometry code does). No
  rescaling or normalisation.
- SMACOF: metric, 3 dimensions, raw unweighted stress on distances in Å, one start from that
  matrix's classical-MDS coordinates, own NumPy Guttman-transform loop. The iteration count
  and a converged flag are saved.
- `p`: distances in the SMACOF reconstruction. `d`: distances in the final structure.
- Means over unordered pairs, per protein, layer and probe:

  ```text
  total²        = mean (d̂ − d)²
  inconsistent² = mean (d̂ − p)²
  consistent²   = mean (p − d)²
  cross         = 2 · mean (d̂ − p)(p − d)
  total² = inconsistent² + consistent² + cross   (exact)
  ```

- Controls: Gaussian-matched noise and shuffled errors on the triangle experiment's B10 bins,
  3 draws each, rebuilt from the triangle run's seeds so the two analyses are paired. Each
  control matrix is final-structure distances plus control noise, clipped the same way, with
  its own classical-MDS start. Outputs call them `control_gaussian` and `control_shuffled`.

## 3. Frozen decisions

| Topic | Decision |
|---|---|
| SMACOF code | Own NumPy; scikit-learn only as a unit-test cross-check |
| Stopping rule | Relative decrease in raw stress per iteration; value chosen from the pilot (section 6) |
| Controls | B10 only, all 3 triangle draws, all 4 probes: 24 control + 4 observed matrices per protein-layer |
| TM-score | Observed matrices only; coordinate RMSD for every matrix |
| Per-bin sums | Sequence separation (6 bins) and B10 true distance (10 bins), two marginal tables; controls averaged over draws |
| Unconverged runs | Kept and flagged; included in cohort curves; sensitivity reported |
| Failed check | The layer aborts and writes only `validation/layer_LL_FAILED.json` |
| Extras | Classical-MDS split in `classical_*` columns; observed SMACOF coordinates saved per layer |
| Plots | Plot 3 on twin y-axes; Plot 4 one panel with four probes and layers 0/10/20/30/40/47 labelled; cross share as a ratio of cohort means |

## 4. Checks

Gates per layer: the four-term identity on every row; SMACOF raw stress no higher than its
classical-MDS start; every rebuilt control vector matches the triangle run's saved
negative-value count (and violation count on proteins with ≤ 1M triangles); the classical
start reproduces the canonical stress, RMSD and negative-prediction count; cohort, Cα-count
and prediction-checksum checks. Saved but not gated: the converged flag and
`Σ d̂·p / Σ p²`, which is 1 at a stress minimum.

Unit tests cover: true distances give all zeros; true distances plus random noise are almost
all inconsistent (share ≈ 1 − 6/R); a perturbed valid structure is all consistent; the input
is not rescaled; agreement with scikit-learn; and a full layer run on a fake export that
first runs the real triangle replay.

## 5. Outputs

- Delta: `/work/hdd/bfna/ikhire/diagnostic_experiments/smacof_error_decomposition_5fold_v1/`
  with `layer_shards/`, `validation/`, `tables/` and `logs/`.
- Local: `delta_files/diagnostic_experiments/smacof_error_decomposition_5fold_v1/` with
  `tables/`, `summary_table.csv`, `cross_share_by_layer.csv`, `RESULTS.md`,
  `sanity_checks.md` and `figures/{main,diagnostics}/`.
- Principal table: `tables/decomposition_per_protein.csv.gz`, one row per probe × layer ×
  protein × matrix type, with the four squared terms (draw means for controls).

Cohort curves average each squared term equally across proteins and show the square root in
Å, with the paired-protein bootstrap (2,000 replicates, seed 20260902).

## 6. Pilot (2026-11-13) and the stopping tolerance

Layers 0 and 47 were run on the full cohort at three tolerances. Every gate passed and
every run converged.

| Tolerance | Job | Layer 0 walltime (32 CPUs) | Layer 0 max iterations | Rows moving > 1e-4 Å by 1e-13 (layer 0, consistent part) | Largest change in a cohort curve |
|---|---|---|---|---|---|
| 1e-10 | `23922169` | 3m16s | 6,807 | 1,166 of 15,456 | 2.3e-4 Å |
| 1e-12 | `23922775` | 5m52s | 16,473 | 10 of 15,456 | 4.9e-5 Å |
| 1e-13 | `23923418` | 6m29s | 17,790 | reference | reference |

SMACOF has plateaus: on a minority of noisy early-layer matrices the stress keeps falling
slowly long after a loose tolerance is met, and the consistent part moves with it (by up to
0.37 Å for one row between 1e-8 and 1e-13). The inconsistent part and all cohort curves are
far less sensitive. Layer 47 has no row moving by more than 1e-4 Å after 1e-12. 1e-13 is
near the float64 noise floor of the stress, so nothing tighter can be checked.

**Decision (2026-11-13):** the full run uses 1e-10 with a 200,000-iteration cap. The cohort
curves and plots are the deliverable and are unaffected at that tolerance; per-protein
consistent values at early layers carry the row-level sensitivity in the table above.

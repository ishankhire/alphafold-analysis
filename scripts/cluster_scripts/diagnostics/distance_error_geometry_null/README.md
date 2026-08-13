# Matched-R² distance-error geometry null

This diagnostic asks whether the geometry recovered from a layer's probe is
better than would be expected from its scalar distance accuracy alone.

For each held-out protein and layer, the script starts with all CA–CA distances
from the final OpenFold-predicted PDB. It adds IID Gaussian errors, carries over
the probe's saved mean bias, clips negative distances to zero, and numerically
rescales the errors until the perturbed distances have the saved probe R². It
then runs classical MDS and compares the aligned reconstruction with the source
PDB using the same fixed-correspondence, Kabsch-aligned TM-score calculation as
the existing regression pipeline.

This is feasible from the saved results. `per_protein.csv` contains the needed
layer/protein R², MSE/RMSE, target variance, and mean error, while
`geometry_metrics.csv` contains the observed MDS geometry. The probes do not
need to be fitted or replayed.

## What the null tests

The null destroys relationships among pair errors while exactly matching each
protein/layer's R². Therefore:

- Observed TM-score above the matched null, together with observed stress below
  the null, is evidence that probe errors have additional geometrically coherent
  structure that is not explained by R² alone.
- Observed and null trajectories that overlap support the simpler explanation
  that improving marginal distance accuracy drives the apparent geometric
  improvement.
- Observed geometry below the null means the probe's structured errors are less
  compatible with a 3D structure than independent errors of the same size.

The empirical `null_tail_p_favorable_*` columns report the fraction of null
replicates at least as favorable as the observation, with a plus-one correction.
With the default 20 replicates their smallest possible value is 1/21, so focus
on effect sizes and the cross-protein/layer pattern rather than treating these
as high-resolution p-values.

## Important limitation

The existing CSVs do not save pair-level residuals. They cannot reproduce the
probe residual distribution's tails, distance dependence, residue-sharing
correlations, or other covariance. This script is explicitly an IID Gaussian
matched-R² null. A stronger residual-permutation null can be obtained without
refitting by replaying `coefficients.csv` on the pair blocks and retaining the
resulting residual vectors. A true out-of-sample residual model would require a
probe run that saves residuals or sufficient residual-bin/covariance summaries.

## Run on Delta

Submit from this directory so `SLURM_SUBMIT_DIR` resolves the Python entrypoint:

```bash
sbatch diagnostic_distance_error_geometry_null.sbatch
```

The launcher is a six-task Slurm array (`--array=0-5`). Each task processes one
layer using this mapping:

| Array task | Layer |
|---:|---:|
| 0 | 0 |
| 1 | 10 |
| 2 | 20 |
| 3 | 30 |
| 4 | 40 |
| 5 | 47 |

All tasks write atomic, layer-specific checkpoints into the same output
directory. After writing its shard, each task rebuilds the consolidated CSVs
from every complete shard it can see. This brief consolidation is protected by
an advisory filesystem lock, while the expensive MDS work remains parallel.
Consequently, the last task to finish produces the complete six-layer root
CSVs. Logs use `slurm-<job>_<task>.out` and `.err`.

The launcher reads:

- `$HOME/csv_files/general_base256_combined/per_protein.csv`
- `$HOME/csv_files/general_base256_combined/geometry_metrics.csv`
- the combined CAMEO and ECOD final predicted-PDB roots documented in
  `datasets.md`

It writes `$HOME/csv_files/distance_error_geometry_null/`. The defaults are the
held-out `test` split, layers 0, 10, 20, 30, 40, and 47 only, 20 replicates per
protein/layer, and random seed `20260813`. For a two-layer smoke test with two
replicates, submit only array tasks 0 and 5:

```bash
NULL_N_REPLICATES=2 \
  sbatch --array=0,5 diagnostic_distance_error_geometry_null.sbatch
```

Do not reuse an output directory with different settings. The script validates
`run_manifest.csv` and stops instead of mixing incompatible shards.

## Outputs

All root CSVs use the combined held-out protein pool; CAMEO/ECOD is only a PDB
path-resolution detail.

- `null_replicates.csv`: one row per layer × held-out protein × replicate. It
  includes saved and achieved R²/MSE/bias checks plus TM-score, RMSD, stress,
  top-three eigenvalue fraction, negative eigenvalue count, and negative
  eigenvalue fraction.
- `per_protein_comparison.csv`: one row per layer × held-out protein, with each
  observed metric, null mean/SD, observed-minus-null effect, and favorable-tail
  probability.
- `layer_summary.csv`: one row per layer. Observed held-out means are compared
  with null layer means formed separately for each replicate.
- `layer_shards/`: the same results checkpointed independently by array task
  after every completed layer, so partial work survives a timeout. A rerun with
  the same manifest skips complete layers and rebuilds the consolidated CSVs.
- `pdb_exports/`: reference CA traces and replicate-0 MDS reconstructions at
  layers 0, 10, 20, 30, 40, and 47 for up to ten proteins selected across the
  held-out length range. B-factors hold per-residue reconstruction deviations.
- `pdb_exports.csv`: the representative protein selection.
- `run_manifest.csv`, `skipped_proteins.csv`, and `summary.txt`: configuration,
  missing-PDB diagnostics, progress, limitations, and interpretation guidance.

The PDB exports are CA-only visualization traces, not chemically complete PDB
models. The reported TM-score is the repository's current approximation, not an
external TM-align execution.

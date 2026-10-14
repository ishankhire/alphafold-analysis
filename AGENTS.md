# AGENTS.md

Guidance for coding agents working in this repository.

## Project overview

This repository analyzes AlphaFold/OpenFold pair representations across 48
layers, primarily with distance-regression probes derived from predicted PDB
structures.

## Python environment

- Use the local `./.venv` for scripts, tests, and import/parity checks.
- Cluster launchers use `~/protein_venv`; treat that as the Delta runtime
  convention.
- Core dependencies are NumPy, SciPy, Matplotlib, Biopython, and scikit-learn.

## Repository map

- `datasets.md` — dataset membership, filtering, and local/Delta input layouts.
- `docs/project_status.md` — current data sources, results, plots, and pending work.
- `docs/methodology/` — canonical regression protocol and output contract.
- `docs/cluster/` — Delta operating notes and links to live documentation.
- `docs/plans/` — experiment specifications and resource references.
- `docs/historical/` — completed/superseded plans and dated audits; not current instructions.
- `scripts/cluster_scripts/` — all cluster-ready Python and SLURM entrypoints.
- `scripts/multi_protein/` — local-only analysis and post-processing.
- `delta_files/` — copied cluster results and current canonical result families.
- `../protein_data/<id>_A/` — local per-protein inputs when available.

## Regression methodology

Before designing or modifying any regression workflow, consult the relevant
documents in `docs/methodology/`:

- `regression_methodology.md` is the normative experimental protocol, including
  cohort, split, feature, precision, weighting, fitting, and evaluation rules.
- `regression_outputs.md` is the normative output and storage contract.

Do not duplicate these rules in script-specific documentation. Historical
scripts and results remain useful, but they are not automatically compliant
with the current methodology or output contract.

## Data and outputs

- The input pool in `datasets.md` combines 154 CAMEO plus 400 ECOD proteins
  with TM-score > 0.7. Canonical regressions use the frozen 552-protein
  eligible cohort; its manifests are authoritative.
- Treat CAMEO and ECOD as one analysis dataset. Use the source label only to
  resolve their different input layouts unless an experiment explicitly asks
  for a source comparison.
- Current outputs belong under `delta_files/`; do not recreate the retired
  `proteins/multi_protein_results/` tree.
- Clearly superseded outputs may be archived under `delta_files/historical_outputs/`.
  Keep outputs referenced by existing analysis scripts at their established paths.

## Delta cluster workflows

Launchers are colocated with their workflows under directories including
`generalized_model_ecod/`, `per_protein_scripts/`, `gated_linear_probes/`,
`diagnostics/`, and `xgboost_search/`.

On Delta, deploy all analysis scripts to and run them from
`$HOME/openfold_interpretability_canonical`.

Agents may inspect cluster files and monitor jobs without asking. Obtain the
user's permission before submitting or cancelling jobs, deploying files, or
making any other cluster changes. Authorization to edit this checkout does not
authorize cluster changes.

Every script intended to run on the cluster, including aggregation and
post-processing entrypoints, must live under `scripts/cluster_scripts/` and
never under `scripts/multi_protein/`. Keep `scripts/multi_protein/` for scripts
that are explicitly intended to run locally.

Before writing or modifying cluster scripts, consult:

- `docs/cluster/README.md`; and
- the live [NCSA Delta documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/)
  for current partitions, limits, accounting, modules, and filesystem behavior.

When moving, renaming, or creating a cluster Python entrypoint, verify every
sbatch reference under `scripts/cluster_scripts/`. Submit from the flat Delta
deployment directory so `SLURM_SUBMIT_DIR` resolves colocated scripts and helpers.

Every new cluster experiment must have a short colocated `README.md` naming
deployment files, input/output roots, exact submission commands, and dependent
aggregation/finalization steps. After submission is authorized, submit the
full arrays with appropriate concurrency caps; no separate one-task resource
pilot is required by default. Preserve dependencies between stages.

### Memory and SU efficiency

Do not use or request excessive memory. Before adding or changing **any** SBATCH
memory request, consult `docs/plans/suggested_sbatch_memory.md` and use its
workflow-specific measurement when one exists. Normally target only 1.5–2x the
observed peak MaxRSS; requests above 2x require a concrete, documented reason.
When the exact workflow has no measurement, estimate its peak from the closest
completed jobs, accounting for cohort size, feature width and dtype, dense
matrix materialization, and per-task concurrency. Start from that estimate with
reasonable headroom and submit the authorized full array. Use successful task
measurements to update `docs/plans/suggested_sbatch_memory.md` for subsequent
runs. Preserve the regression pipeline while right-sizing, since Delta charges
can be memory-derived.

### Walltime and scheduling efficiency

Before adding or changing **any** SBATCH walltime request, consult
`docs/plans/suggested_sbatch_walltime.md` and use its workflow-specific
measurement when one exists. Normally target 1.5–2x the maximum elapsed time
of representative successful tasks. Do not size from failed, cancelled,
cached, restart-skipped, smoke, or no-op runs. When the exact workflow is
unmeasured, estimate from the closest completed workflow and account for
differences in cohort size, feature width, layers, folds, models, and sequential
stages. Submit the authorized full array and record successful elapsed times
for subsequent runs. Avoid excessive limits because Slurm schedules against
requested walltime and shorter accurate jobs can backfill sooner.

### Launcher requirements

- Request walltime and memory from the shared MaxRSS and elapsed-time guides,
  leaving reasonable headroom without substantially over-reserving.
- Save outputs after every completed layer or similarly meaningful unit so
  partial results survive timeouts.
- Log timestamps and elapsed times for each major stage and layer.
- Every new or modified sbatch file must include:
  `#SBATCH --mail-type=BEGIN,END,FAIL` and
  `#SBATCH --mail-user=ikhire3@gatech.edu`.
- Resolve Python entrypoints with
  `SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}` and
  `srun python3 -u "$SCRIPT_DIR/<script>.py"`.

## Implementation conventions

- Pair representations have shape `(R, R, 128)`.
- Reuse `scripts/cluster_scripts/cluster_regression_helpers.py` for dataset
  discovery, pair-block/PDB resolution, coordinate extraction, distance
  matrices, AA-pair indexing, and geometry utilities. Diagnostics must add the
  parent cluster-script directory to `sys.path` before importing it.
- For hot pair-block extraction, plain `np.load(...)` with preallocated writes
  has been faster than `mmap_mode="r"` plus repeated concatenation.
- Default to fixed repo-relative configuration. Add CLI arguments only for
  genuine environment, reproducibility, or input/output variability.

## Planning and documentation

- Every implementation plan must name exact output directories and filenames,
  row granularity, split/evaluation scope, and principal columns or fields.
- Record important codebase changes very briefly in `branch-analysis.md`; omit
  minor edits.

## Overleaf

Before editing anything inside the overleaf folder, please check the AGENTS.md located inside overleaf/

## Git workflow

After every significant change:

```bash
git add <changed-files>
git commit -m "<descriptive message>"
git push
```

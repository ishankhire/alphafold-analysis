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
- `docs/cluster/` — deployment, launcher rules, resource guides, and Delta documentation.
- `docs/plans/` — experiment specifications.
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

## Cluster work

Before writing or modifying cluster scripts, read the required deployment,
launcher, and submission rules in [the cluster guide](docs/cluster/README.md)
and consult the live [Delta documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/).
Before adding or changing SBATCH memory or walltime, consult the corresponding
[memory](docs/cluster/suggested_sbatch_memory.md) or
[walltime](docs/cluster/suggested_sbatch_walltime.md) guide.

Cluster entrypoints, including aggregation and post-processing, belong under
`scripts/cluster_scripts/`; `scripts/multi_protein/` is for local-only scripts.
Agents may inspect cluster files and monitor jobs without asking. Obtain the
user's permission before submitting or cancelling jobs, deploying files, or
making any other cluster changes. Local edits do not authorize remote changes.

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

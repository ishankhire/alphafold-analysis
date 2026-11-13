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
- On Delta, the default CSV output root is `/work/hdd/bfna/ikhire/`.
  The published `regression_5fold_v1/` collection is write-protected
  (`collection_manifest.json`), so save new final, analysis-ready results under
  `main_experiments/<experiment_name>/` and exploratory runs under
  `diagnostic_experiments/<experiment_name>/`. `$HOME/csv_files/` is
  a legacy location; preserve existing input paths unless explicitly migrated.
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
Agents may inspect cluster files, monitor jobs, and make cluster changes needed
for the user's task without asking, including submitting or cancelling jobs,
deploying or editing files, and creating directories. Always ask the user's
permission before deleting any files on the cluster. Clearly report the exact
cluster changes made in chat, including affected paths and job IDs where
applicable; documentation updates do not substitute for this chat report.

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

## Temporary reminder: repoint ECOD paths (delete this section when done)

The ECOD data under `/work/hdd/bfna/thayes/ecod/` was purged in November 2026.
Scripts still hard-code that root: `cluster_regression_helpers.py`
(`ECOD_PDB_ROOT`, `ECOD_PAIR_BLOCKS_ROOT`),
`generalized_model_ecod/general_train_test_common.py`, and several diagnostics
and sbatch files (`grep -rn "bfna/thayes/ecod" scripts`). They should point to
`/work/hdd/bfna/ikhire/thayes/ecod/`, which has the same layout. Remind the user
about this when ECOD paths come up, but don't change the scripts unless asked.
For PDBs, use `experiments/openfold_reps_outputs_sm/` (complete), never the
partial rescued `experiments/<ID>/..._representations_layers_save_0_47/` copy.
The pair blocks and alignments are not on Delta yet; the user will copy them
from PACE. See `datasets.md` and inventory to-do 3–4 in
`docs/cluster/delta_data_inventory.md`. Once the scripts are repointed,
delete this whole section.

## Publication figures

- All publication graphs must use large, readable axis labels, tick labels,
  titles, legends, annotations, and colorbar labels at the final manuscript size.
- Match the typography in `scripts/multi_protein/generate_sept_2_publication_plots.py`
  (main-figure font scale 1.75); expand the layout to prevent overlap or clipping.
- Keep numeric axes uncluttered: aim for 4–5 major y-axis ticks unless the data
  require otherwise. Preserve meaningful categorical ticks on heatmaps.
- Regenerate both PDF and PNG outputs after changing figure styling and visually
  inspect the results.

## Planning and documentation

- Whenever you make a plan, please interrogate me and ask me lots of questions! I'd rather there be too many than too few questions.
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

# AGENTS.md

This file provides guidance to coding agents when working with code in this repository.

## Project Overview

Analysis of AlphaFold pair representations across 48 layers for multiple proteins. Uses PDB-predicted structures for distance-based regression probes. Investigates how pairwise residue features evolve through network depth.

**Dependencies:** numpy, scipy, matplotlib, biopython, scikit-learn (no package manager — `pip install` manually).

## Python Environment

- A local project virtual environment exists at `./.venv`.
- Use it for local Python scripts, tests, and import/parity checks that need `scikit-learn`, `biopython`, or other analysis dependencies:
  ```bash
  source .venv/bin/activate
  ```
- Many cluster sbatch files currently activate `~/protein_venv`. Treat that as the cluster/runtime convention, but for local validation in this repo prefer `./.venv` over the system Python.

## Numerical Precision

- For dense linear regression, least-squares, PCA, SVD, and related matrix-based diagnostics on pair-block features, default to converting materialized feature matrices and regression targets to `float64` before scaling and fitting.
- Treat `float32` as opt-in for this repo: use it only when the experiment is explicitly about float precision or when memory/performance tradeoffs are the point.
- Reason: recent large all-row diagnostics in this repo showed that `float32` dense least-squares can become numerically pathological, while `float64` materially improves stability and performance.

## Key Directories

- `../protein_data/<id>_A/` — local per-protein input data: `pair_blocks/`, `predicted_pdb/`, `csv_files/`, `visualizations/`
- `datasets.md` — combined CAMEO+ECOD dataset paths, counts, and TM-score filtered protein IDs
- `delta_files/` — saved Delta outputs and the current canonical result families
- `docs/cluster/` — concise local Delta operating notes and links to live documentation
- `not_relevant_miscellaneous/mds_structures/` — archived MDS/Chimera reference renders
- `scripts/cluster_scripts/` — all cluster-ready Python scripts and SLURM sbatch files
- `scripts/multi_protein/` — local aggregation, post-processing, and analysis scripts

## Data Essentials

- The default analysis set is the combined TM-score > 0.7 CAMEO + ECOD protein pool documented in `datasets.md`.
- Treat CAMEO and ECOD proteins as one analysis dataset. The source label is only for resolving different Delta input file layouts.
- Current filtered set size: 154 CAMEO proteins + 400 ECOD proteins = 554 unique proteins.
- **Local data root:** when working from the `protein` workspace, local per-protein inputs, when present, live in the sibling directory `../protein_data/`
- **Outputs:** current copied cluster results and local diagnostic outputs live under `delta_files/`. Do not recreate the retired `proteins/multi_protein_results/` tree.

## Cluster Setup

Cluster scripts are currently written for the **Delta** SLURM cluster. The main Delta sbatch files in `scripts/cluster_scripts/` are `delta_regression_job.sbatch` and `delta_per_protein_regression_job.sbatch`. They use `module load python`, `srun`, and `--account=bekh-delta-cpu`.

Before writing or modifying sbatch files or cluster scripts, consult:
- `docs/cluster/slurm_array_notes.md` for the repository's array conventions;
- the live [NCSA Delta user documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/) for current partitions, limits, modules, and filesystem behavior.

**IMPORTANT:** When moving, renaming, or creating cluster Python scripts, always verify that every sbatch file in `scripts/cluster_scripts/` references the correct Python script filename. A mismatch will cause silent job failures on the cluster.

### Cluster repository and documentation conventions

- Cluster-ready Python scripts and sbatch launchers are normally run from the repository checkout, with submission from the relevant script directory so `SLURM_SUBMIT_DIR` resolves the entrypoint. Flat copies under a Delta working directory are supported only when a workflow explicitly requires them; copied scripts must retain their documented environment variables and any colocated/shared helper modules.
- Keep script-specific usage notes concise and centralized in `datasets.md`, this file, `branch-analysis.md`, or a nearby diagnostics index. Do not create a separate README for every cluster script unless the workflow has substantial, standalone operational instructions that cannot reasonably live in shared documentation.

### Cluster script guidelines

- Request walltime and memory conservatively, leaving reasonable headroom without reserving substantially more than the job is expected to use. Scale requests to the number of proteins and layers processed by each individual job or array task. For reference, the `192g` request in `general_base256_train_test_layers_24_47.sbatch` supports a large combined-dataset run over roughly 500 proteins and many layers. Use observed runtime and peak memory from comparable jobs to refine future requests.
- Save outputs after every completed layer so partial results survive timeouts or mid-run failures.
- Add detailed timestamped time logging to stdout for monitoring, including per-layer timing and major stage timing where relevant.
- Every new or modified sbatch file must include both notification directives:
  `#SBATCH --mail-type=BEGIN,END,FAIL` and
  `#SBATCH --mail-user=ikhire3@gatech.edu`.
- For sbatch files that need to call a Python script by path, follow the local convention:
  `SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}` and then `srun python3 -u "$SCRIPT_DIR/<script>.py"`.

### Cluster data paths

Use `datasets.md` as the source of truth for dataset membership, TM-score filtering, and exact Delta pair-block/PDB input layouts.

Many older CAMEO-only cluster scripts still read these environment variables from their sbatch files:

| Env var | Delta value |
|---------|-------------|
| `PAIR_BLOCKS_ROOT` | `/work/hdd/bekh/thayes/openfold_reps` |
| `PDB_ROOT` | `/work/hdd/bekh/thayes/openfold_reps_outputs_sm` |
| `CSV_ROOT` | `$HOME/csv_files` |

Those roots cover the CAMEO layout only. New or modified combined-dataset analyses must resolve CAMEO and ECOD input paths separately, then merge the proteins before sampling, splitting, fitting, and reporting. Do not treat CAMEO and ECOD as separate analysis groups unless an experiment explicitly asks for that comparison.

## Architecture Patterns

**Pair representation shape:** Always `(R, R, C)` where R = entity sequence length and C = 128 channels.

**PDB-based regression scripts** use inline PDB parsing (no lib/ imports for coordinate extraction). They extract CA coordinates directly from predicted PDB files.

**Visualizations:** Save current aggregate plots beside their result family under
`delta_files/<analysis>/analysis/` or under `delta_files/publication_plots/`.

**Regression output schema:** Unless a task explicitly requests another layout,
follow the file organization and column style in
`delta_files/general_base256_combined/`. In particular, prefer separate
`per_protein.csv` and aggregate summary CSVs with explicit split labels rather
than combining unrelated row granularities.

**Path resolution:** All scripts compute paths via `ROOT_DIR = os.path.dirname(...)` so they work regardless of working directory.

**Shared cluster helper:** A reusable helper module now lives at `scripts/cluster_scripts/cluster_regression_helpers.py`.

- Use this helper for future cluster scripts when you need stable shared utilities such as:
  - `find_pdb_file`
  - `get_pair_block_path`
  - `discover_proteins`
  - `extract_ca_coordinates`
  - `compute_distance_matrix`
  - `build_sorted_aa_pair_indices`
  - geometry helpers such as `classical_mds`, `compute_tm_score`, and `analyze_protein_geometry`
- Do not feel obligated to retrofit older scripts just to use this helper; it is intended mainly to reduce duplication in new cluster work.
- For scripts under `scripts/cluster_scripts/diagnostics/`, add the parent `scripts/cluster_scripts` directory to `sys.path` before importing `cluster_regression_helpers`.
- For hot per-layer pair-block extraction, prefer plain `np.load(...)` plus direct preallocated writes over `mmap_mode="r"` and repeated `np.concatenate(...)`; large fancy-index slices have been materially faster this way on cluster runs.

## Script Configuration Policy

- Default to fixed script configuration (repo-relative paths/constants in code).
- Do not add CLI arguments unless they are necessary for multi-environment use, reproducibility control, or unavoidable input/output variability.

## Planning Requirements

- Every implementation plan must specify the exact intended outputs, including
  output directories, filenames, row granularity, split/evaluation scope, and
  principal columns or fields.

## Change Log

When making important changes to the codebase, record them in `branch-analysis.md`. This file serves as a scratchpad for reasoning about changes. Keep it focused: skip minor edits. It is very important to keep your descriptions super brief. I only want things that meaningfully alter the codebase, and even then, no need to be thorough with descriptions.

## Git Workflow

After every significant change, run:
```
git add -A
git commit -m "<descriptive message>"
git push
```
Commit frequently with clear messages. Push after each commit to keep the remote up to date.

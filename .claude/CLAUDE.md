# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

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

- `lib/` — shared library: alignment utilities (`build_alignment`, `validate_alignment`) and structure utilities (`extract_residue_coordinates`, `compute_distance_matrix`, etc.)
- `../protein_data/<id>_A/` — local per-protein input data: `pair_blocks/`, `predicted_pdb/`, `csv_files/`, `visualizations/`
- `proteins/multi_protein_results/` — aggregated multi-protein outputs created inside the repo at runtime
- `proteins/llm_reference/` — **reference PDFs for cluster usage** (see Cluster section below)
- `scripts/cluster_scripts/` — all cluster-ready Python scripts and SLURM sbatch files
- `scripts/multi_protein/` — local/dev analysis scripts and regression variants

## Data Essentials

- **Local data root:** when working from the `protein` workspace, the moved per-protein inputs live in the sibling directory `../protein_data/`
- **Pair blocks:** `../protein_data/<id>/pair_blocks/<id>_pair_block_{0..47}.npy` — shape `(R, R, 128)` where R = entity sequence length, 128 channels
- **PDB structures:** `../protein_data/<id>/predicted_pdb/` — AlphaFold predicted structures; CA coordinates extracted inline by regression scripts
- **Outputs:** per-protein CSVs and plots still live beside each protein under `../protein_data/<id>/`, while aggregated multi-protein outputs stay under `proteins/multi_protein_results/`

## Cluster Setup

Scripts run on two SLURM clusters: **Delta** and **ICE**. Each has its own sbatch files in `scripts/cluster_scripts/`:

| Cluster | Sbatch files | Notes |
|---------|-------------|-------|
| ICE | `regression_job.sbatch`, `per_protein_regression_job.sbatch`, `protein_split_regression_job.sbatch` | Uses `module load anaconda3`, `$SCRATCH` for data root |
| Delta | `delta_regression_job.sbatch`, `delta_per_protein_regression_job.sbatch` | Uses `module load python`, `srun`, `--account=bekh-delta-cpu` |

**Reference documentation** lives in `proteins/llm_reference/`:
- `delta_slurm_guide.pdf` — Delta cluster SLURM specifics
- `slurm_guidelines.pdf` — general SLURM guidelines
- `storage_on_ice.pdf` — ICE storage and `$SCRATCH` layout

Always consult these PDFs when writing or modifying sbatch files or cluster scripts.

**IMPORTANT:** When moving, renaming, or creating cluster Python scripts, always verify that every sbatch file in `scripts/cluster_scripts/` references the correct Python script filename. A mismatch will cause silent job failures on the cluster.

### Cluster data paths (environment variables)

Cluster Python scripts read data paths from three environment variables exported by their sbatch files:

| Env var | Delta value | ICE value |
|---------|------------|-----------|
| `PAIR_BLOCKS_ROOT` | `/work/hdd/bekh/thayes/openfold_reps` | `$SCRATCH` |
| `PDB_ROOT` | `/work/hdd/bekh/thayes/openfold_reps_outputs_sm` | `$SCRATCH/Structures` |
| `CSV_ROOT` | `/work/hdd/bekh/thayes/linear_regression_csv_files` | `$SCRATCH/csv_files` |

**Delta file layout:**
- Pair block: `PAIR_BLOCKS_ROOT/{protein_id}/{protein_id}_pair_block_{layer}.npy`
- PDB file: `PDB_ROOT/my_outputs_align_{protein_id}_representations_load_from_saved_layer_47/predictions/{protein_id}_model_1_ptm_unrelaxed.pdb`
- CSV output: `CSV_ROOT/`

When writing new cluster regression scripts for Delta, use these env vars rather than hardcoding paths.

## Architecture Patterns

**lib/ imports:** Scripts that use `lib/` must add the project root to `sys.path`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.alignment import build_alignment, validate_alignment
from lib.structure import extract_residue_coordinates, compute_distance_matrix, detect_chain
```

**Pair representation shape:** Always `(R, R, C)` where R = entity sequence length and C = 128 channels.

**PDB-based regression scripts** use inline PDB parsing (no lib/ imports for coordinate extraction). They extract CA coordinates directly from predicted PDB files.

**Visualizations:** All plots saved to `proteins/<id>/visualizations/` or `proteins/multi_protein_results/visualizations/`.

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

## Script Configuration Policy

- Default to fixed script configuration (repo-relative paths/constants in code).
- Do not add CLI arguments unless they are necessary for multi-environment use, reproducibility control, or unavoidable input/output variability.
- If CLI arguments are necessary, keep them minimal and justify them in the script docstring or code comments.

## Change Log

When making important or substantive changes to the codebase, record them in `branch-analysis.md`. This file serves as a scratchpad for reasoning about changes — what was changed, why, and any key decisions. Keep it focused: skip minor or trivial edits, but document anything that affects analysis logic, data flow, or architecture.

## Git Workflow

After every significant change, run:
```
git add -A
git commit -m "<descriptive message>"
git push
```
Commit frequently with clear messages. Push after each commit to keep the remote up to date.

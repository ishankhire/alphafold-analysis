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
- `docs/methodology/` — canonical regression protocol, output contract, and
  implementation audits.
- `docs/cluster/` — Delta operating notes and links to live documentation.
- `docs/plans/` — experiment and implementation plans.
- `scripts/cluster_scripts/` — cluster-ready Python and SLURM entrypoints.
- `scripts/multi_protein/` — local aggregation and post-processing.
- `delta_files/` — copied cluster results and current canonical result families.
- `../protein_data/<id>_A/` — local per-protein inputs when available.

## Regression methodology

Before designing or modifying any regression workflow, consult the relevant
documents in `docs/methodology/`:

- `regression_methodology.md` is the normative experimental protocol, including
  cohort, split, feature, precision, weighting, fitting, and evaluation rules.
- `regression_outputs.md` is the normative output and storage contract.
- `regression_protocol_audit.md` records which implementations meet the
  methodology and where reruns or decisions remain.
- `regression_output_audit.md` records output coverage and remaining gaps.

Do not duplicate these rules in script-specific documentation. Historical
scripts and results remain useful, but they are not automatically compliant
with the current methodology or output contract.

## Data and outputs

- The default analysis cohort is the combined TM-score > 0.7 CAMEO + ECOD pool
  in `datasets.md`: 154 CAMEO plus 400 ECOD proteins, 554 unique proteins.
- Treat CAMEO and ECOD as one analysis dataset. Use the source label only to
  resolve their different input layouts unless an experiment explicitly asks
  for a source comparison.
- Current outputs belong under `delta_files/`; do not recreate the retired
  `proteins/multi_protein_results/` tree.

## Delta cluster workflows

Launchers are colocated with their workflows under directories including
`generalized_model_ecod/`, `per_protein_scripts/`, `gated_linear_probes/`,
`diagnostics/`, and `xgboost_search/`.

Before writing or modifying cluster scripts, consult:

- `docs/cluster/slurm_array_notes.md`;
- `docs/cluster/delta-job-cheatsheet.md`; and
- the live [NCSA Delta documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/)
  for current partitions, limits, accounting, modules, and filesystem behavior.

When moving, renaming, or creating a cluster Python entrypoint, verify every
sbatch reference under `scripts/cluster_scripts/`. Submit from the relevant
script directory so `SLURM_SUBMIT_DIR` resolves colocated scripts and helpers.

### Memory and SU efficiency

Delta CPU charges are based on reserved resources, using the larger of the CPU
or memory-derived charge, rather than actual CPU utilization. Large memory
requests can therefore dominate SU usage even for jobs using only a few cores.

A recent gated sequence-probe run retained a layer-wide feature cache plus raw,
standardized, sample-weighted, and train/test copies of dense matrices. MaxRSS
was about 93 GiB per task, and a 192 GB request made six tasks cost about 375
SUs despite only about 6.4 hours of measured CPU work. A bounded-memory design
targeting roughly 16 GB would cost about 31 SUs at the same runtime.

For future large regressions:

- Prefer bounded-memory streaming or chunked algorithms when materialized
  matrices or concurrent copies create a large working set. For OLS, use a
  stable approach such as chunked TSQR, preserve the canonical objective, and
  validate numerical parity.
- Fuse passes across categories when practical, process one protein or bounded
  chunk at a time, and retain compact statistics rather than pair-level arrays.
- Before launching a full array, run a representative high-memory task, inspect
  completed-job MaxRSS and elapsed time, and estimate its Delta charge. Size
  memory from representative completed runs with reasonable headroom; failed or
  partial runs may not have reached their eventual peak.
- Do not assume a larger memory request improves runtime. It is a reservation
  and limit; excessive headroom raises charges and may increase queue time.
- Keep a dense materialized design only when methodologically necessary, and
  document and right-size its memory requirement.

### Launcher requirements

- Request walltime and memory from observed comparable runs, leaving reasonable
  headroom without substantially over-reserving.
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
  has been faster than `mmap_mode="r"` plus repeated concatenation. This does not
  override the bounded-memory requirement for large aggregate workflows.
- Default to fixed repo-relative configuration. Add CLI arguments only for
  genuine environment, reproducibility, or input/output variability.

## Planning and documentation

- Every implementation plan must name exact output directories and filenames,
  row granularity, split/evaluation scope, and principal columns or fields.
- Record important codebase changes very briefly in `branch-analysis.md`; omit
  minor edits.

## Git workflow

After every significant change:

```bash
git add -A
git commit -m "<descriptive message>"
git push
```

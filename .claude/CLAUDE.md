# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

Analysis of AlphaFold pair representations across 48 layers for multiple proteins (7B3A, 6TF4, 6WMK, 6WQC). Uses PDB-predicted structures for distance-based regression probes. Investigates how pairwise residue features evolve through network depth.

## Dependencies

numpy, scipy, matplotlib, biopython, scikit-learn. No package manager config exists — install manually:
```
pip install numpy scipy matplotlib biopython scikit-learn
```

## Project Structure

```
protein/                              ← project root
├── lib/                              ← shared library (importable by all scripts)
│   ├── __init__.py
│   ├── alignment.py                  ← AlignmentResult, build_alignment, validate_alignment
│   └── structure.py                  ← extract_residue_coordinates, compute_distance_matrix,
│                                        load_distance_matrix, detect_chain, extract_pdb_ca_coordinates,
│                                        superimpose_structures, compute_tm_score
├── proteins/
│   ├── 7b3a_A/                       ← all data for protein 7B3A chain A
│   │   ├── pair_blocks/              ← raw AlphaFold pair representations (7b3a_A_pair_block_{0..47}.npy)
│   │   ├── predicted_pdb/            ← AlphaFold predicted PDB structure
│   │   ├── csv_files/                ← analysis output CSVs + distances
│   │   └── visualizations/           ← output plots
│   ├── 6tf4_A/                       ← same layout
│   ├── 6wmk_A/
│   ├── 6wqc_A/
│   ├── multi_protein_results/        ← created at runtime by multi-protein scripts
│   │   ├── csv_files/
│   │   └── visualizations/
│   └── claude_reference/             ← reference PDFs (SLURM guidelines, storage docs)
├── scripts/
│   ├── multi_protein/
│   │   ├── structure_comparison.py   ← CIF vs PDB comparison (RMSD, TM-score)
│   │   └── regression/
│   │       ├── final_multi_protein_regression.py    ← PDB-based, cluster-ready
│   │       ├── new_multi_protein_regression.py      ← PDB-based multi-protein regression
│   │       ├── probe_layer_regression_predicted.py  ← layer-wise R² with predicted distances
│   │       ├── regression_variants_predicted.py     ← top-k + upper/lower triangle variants
│   │       └── regression_job.sbatch                ← SLURM job submission script
│   └── tests/
│       └── test_alignment.py         ← alignment sanity checks (run standalone)
├── .claude/CLAUDE.md
├── branch-analysis.md
├── cluster_multi_protein_regression.md
└── multi-protein-regression-plan.md
```

When adding a new protein, create `proteins/<pdb_id>_A/` with the same subdirectory layout.

## Running Scripts

All scripts use absolute path computation (`ROOT_DIR = os.path.dirname(...)`) so they resolve data correctly regardless of working directory. Run from the project root:
```
python scripts/multi_protein/regression/probe_layer_regression_predicted.py
python scripts/multi_protein/structure_comparison.py
python scripts/tests/test_alignment.py
```
No build system or linter is configured. `scripts/tests/test_alignment.py` is a standalone test script (no pytest).

## Data Layout (per protein)

- `proteins/<id>/pair_blocks/<id>_pair_block_{0..47}.npy` — raw pair representations, shape `(R, R, 128)`
- `proteins/<id>/predicted_pdb/` — AlphaFold predicted PDB structure files
- `proteins/<id>/csv_files/predicted_residue_coordinates.csv` — PDB-extracted CA coordinates
- `proteins/<id>/csv_files/predicted_residue_distances.csv` — PDB-based CA-CA distance matrix
- `proteins/<id>/csv_files/structure_comparison.csv` — RMSD, TM-score, per-residue deviation
- `proteins/<id>/csv_files/` — all analysis output CSVs
- `proteins/<id>/visualizations/` — all output plots

## Data Pipeline

```
proteins/<id>/predicted_pdb/*.pdb ──► PDB CA coordinate extraction (inline in regression scripts)
                                         ──► predicted_residue_coordinates.csv
                                         ──► predicted_residue_distances.csv

proteins/<id>/pair_blocks/*.npy ──┐
predicted_residue_distances.csv ──┤
                                  ▼
   multi_protein/regression/probe_layer_regression_predicted.py ──► per-layer R² (PDB-based)
   multi_protein/regression/final_multi_protein_regression.py ──► cluster-ready pooled regression
   multi_protein/regression/new_multi_protein_regression.py ──► multi-protein regression
   multi_protein/regression/regression_variants_predicted.py ──► top-k + triangle variants

   multi_protein/structure_comparison.py ──► CIF vs PDB RMSD, TM-score, per-residue deviation
```

## Architecture Patterns

**lib/ imports:** Scripts that use `lib/` must add the project root to `sys.path`:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from lib.alignment import build_alignment, validate_alignment
from lib.structure import extract_residue_coordinates, compute_distance_matrix, detect_chain
```

**Residue alignment:** Use `lib/alignment.build_alignment(cif_path, chain_id, protein_id)` to get an `AlignmentResult` with explicit `pb_indices` mapping each resolved structure residue to its pair-block index. Always call `validate_alignment` before using `pb_indices`.

**Pair representation shape:** Always `(R, R, C)` where R=entity sequence length (e.g. 280 for 7B3A) and C=128 channels.

**PDB-based regression scripts** use inline PDB parsing (no lib/ imports for coordinate extraction). They extract CA coordinates directly from predicted PDB files.

## Visualizations

All generated plots must be saved to `proteins/<pdb_id>_A/visualizations/` (or a subdirectory within it). Multi-protein results go to `proteins/multi_protein_results/visualizations/`. In scripts, use:
```python
VIS_DIR = os.path.join(PROTEIN_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)
```

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

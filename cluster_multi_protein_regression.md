# Multi-Protein Regression on Cluster

## Goal

Train a linear regression probe at each of the 48 AlphaFold pair-representation layers to predict CA-CA distances from pair-block features. This tests how much structural (distance) information is encoded at each layer. The script pools residue pairs across all proteins into a single train/test split, producing one R² curve over layers.

## Scripts

- **Cluster:** `scripts/multi_protein/regression/cluster_multi_protein_regression.py` — adapted for ICE `$SCRATCH` layout
- **Local:** `scripts/multi_protein/regression/multi_protein_regression.py` — runs against the project repo layout

Both are fully standalone (no `lib/` imports) and skip proteins on failure.

## How It Works

1. **Discover proteins**: scans `ROOT_DIR` (cluster) or `PROTEINS_DIR` (local) for subdirectories.
2. **Parse chain ID**: extracted from `protein_id` — the part after the last underscore (e.g. `7b3a_A` → chain `A`).
3. **Load predicted PDB**: extracts all CA atoms on the target chain (no standard-residue filtering).
4. **Verify dimension match**: CA atom count must exactly equal pair block dimension (no alignment needed for predicted PDBs).
5. **Compute distance matrix**: vectorized pairwise Euclidean distances between CA atoms.
6. **Pool upper-triangle pairs** across all proteins into a single dataset.
7. **80/20 train/test split** (random_state=42), fixed across all layers.
8. **Per-layer regression**: for each layer 0–47, load pair blocks, concatenate `[pb[i,j,:], pb[j,i,:]]` → 256-dim feature vector, standardize (fit on train only), fit `LinearRegression`, record test R².
9. **Save outputs**: R² CSV, coefficient CSV, last-layer predicted-vs-actual CSV. Local version also saves visualizations.

## Usage

```bash
# Local (from project root)
python scripts/multi_protein/regression/multi_protein_regression.py

# Cluster (submitted via SLURM)
sbatch regression_job.sbatch
```

## Configuration Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `N_LAYERS` | 48 | AlphaFold pair representation layers |
| `N_CHANNELS` | 128 | Channels per direction in pair block |
| `FEATURE_DIM` | 256 | Concatenated feature dim (128 × 2) |
| `TEST_SIZE` | 0.2 | Train/test split ratio |
| `RANDOM_STATE` | 42 | Reproducible split seed |

## Directory Layouts

**Local** (`multi_protein_regression.py`):
```
proteins/<protein_id>/
├── predicted_pdb/<protein_id>_model_1_ptm_unrelaxed.pdb
├── pair_blocks/<protein_id>_pair_block_{0..47}.npy
└── csv_files/   (created by script)
```

**Cluster** (`cluster_multi_protein_regression.py`):
```
$SCRATCH/
├── Structures/my_outputs_align_{id}_..._layer_47/predictions/*.pdb
├── <protein_id>/<protein_id>_pair_block_{0..47}.npy
└── csv_files/   (created by script)
```

## Outputs

**Local version** saves:
- Per-protein CSVs (`predicted_residue_coordinates.csv`, `predicted_residue_distances.csv`)
- `csv_files/layer_regression_r2_predicted.csv` — layer, R², n_proteins, n_total_pairs
- `csv_files/layer_regression_coefficients_predicted.csv` — 256 coefficients + intercept per layer
- `csv_files/predicted_distances_last_layer_predicted.csv` — actual vs predicted with train/test label
- `visualizations/layer_regression_r2_predicted.png` — R² vs layer curve
- `visualizations/pred_vs_actual_all_predicted.png` — scatter, all proteins combined
- `visualizations/pred_vs_actual_per_protein_predicted.png` — scatter grid, one panel per protein

**Cluster version** saves:
- `csv_files/layer_regression_r2_predicted.csv` — per-layer R² with per-protein breakdown
- `csv_files/layer_regression_coefficients_predicted.csv` — coefficients per layer
- `csv_files/predicted_distances_last_layer_predicted.csv` — actual vs predicted
- `csv_files/regression_summary.txt` — human-readable summary

## Results (4 proteins, local run)

- Proteins: 7b3a_A (280 res), 6tf4_A (67 res), 6wmk_A (69 res), 6wqc_A (294 res)
- Total pairs: 86,688
- Peak R²: 0.984 at layer 45
- Layer 47 R²: 0.982

## Flagged Code (needs updating when scaling)

Three functions in each script are marked with FLAG comments — these encode assumptions about directory layout and file naming:

1. **`find_pdb_file()`** — PDB filename pattern (`<protein_id>_model_1_ptm_unrelaxed.pdb`)
2. **`get_pair_block_path()`** — pair block path pattern (`pair_blocks/<protein_id>_pair_block_<layer>.npy`)
3. **`discover_proteins()`** — assumes flat directory listing under `proteins_dir`

## Key Design Decisions

- **No alignment needed**: predicted PDBs have the same residue count as pair blocks (unlike CIF files which may have unresolved residues).
- **No standard-residue filtering**: includes all residues with a CA atom (skips water only).
- **Pooled regression**: all proteins' pairs are combined into one dataset, not trained per-protein.
- **StandardScaler fit on train only**: avoids data leakage.
- **Memory approach**: all protein data loaded into RAM at once. Fine for hundreds of proteins; may need batching at thousands.

## Dependencies

```
pip install numpy scipy matplotlib scikit-learn biopython
```

# Multi-Protein Regression on Cluster

## Goal

Train a linear regression probe at each of the 48 AlphaFold pair-representation layers to predict CA-CA distances from pair-block features. This tests how much structural (distance) information is encoded at each layer. The script pools residue pairs across all proteins into a single train/test split, producing one R² curve over layers.

## Script

`scripts/multi_protein/regression/probe_layer_regression_predicted.py`

- **Fully standalone** — no imports from `lib/` or any other project file. All helpers are inline.
- **Cluster-ready** — headless matplotlib (`Agg` backend), argparse CLI, only pip-installable dependencies.
- **Robust** — skips individual proteins on failure (missing files, dimension mismatch) without crashing.

## How It Works

1. **Discover proteins**: scans `--proteins_dir` for subdirectories. Each subdirectory name is the `protein_id` (e.g. `7b3a_A`).
2. **Parse chain ID**: extracted from `protein_id` — the part after the last underscore (e.g. `7b3a_A` → chain `A`).
3. **Load predicted PDB**: from `<protein_dir>/predicted_pdb/<protein_id>_model_1_ptm_unrelaxed.pdb`. Extracts all CA atoms on the target chain (no standard-residue filtering).
4. **Verify dimension match**: CA atom count must exactly equal pair block dimension (no alignment needed for predicted PDBs).
5. **Compute distance matrix**: vectorized pairwise Euclidean distances between CA atoms.
6. **Pool upper-triangle pairs** across all proteins into a single dataset.
7. **80/20 train/test split** (random_state=42), fixed across all layers.
8. **Per-layer regression**: for each layer 0–47, load pair blocks, concatenate `[pb[i,j,:], pb[j,i,:]]` → 256-dim feature vector, standardize (fit on train only), fit `LinearRegression`, record test R².
9. **Save outputs**: R² CSV, coefficient CSV, last-layer predicted-vs-actual CSV, three plots.

## CLI Usage

```bash
# Default paths (run from project root)
python scripts/multi_protein/regression/probe_layer_regression_predicted.py

# Custom paths (cluster)
python probe_layer_regression_predicted.py \
    --proteins_dir /data/proteins \
    --output_dir /data/results
```

## Configuration Constants

| Constant | Value | Meaning |
|----------|-------|---------|
| `N_LAYERS` | 48 | AlphaFold pair representation layers |
| `N_CHANNELS` | 128 | Channels per direction in pair block |
| `FEATURE_DIM` | 256 | Concatenated feature dim (128 × 2) |
| `TEST_SIZE` | 0.2 | Train/test split ratio |
| `RANDOM_STATE` | 42 | Reproducible split seed |

## Expected Directory Layout Per Protein

```
proteins/<protein_id>/
├── predicted_pdb/<protein_id>_model_1_ptm_unrelaxed.pdb
├── pair_blocks/<protein_id>_pair_block_{0..47}.npy
└── csv_files/   (created by script)
    ├── predicted_residue_coordinates.csv
    └── predicted_residue_distances.csv
```

## Outputs

**Per-protein** (saved inside each protein's `csv_files/`):
- `predicted_residue_coordinates.csv` — chain, residue name, number, x, y, z
- `predicted_residue_distances.csv` — N×N distance matrix with residue labels

**Multi-protein summary** (saved to `--output_dir`):
- `csv_files/layer_regression_r2_predicted.csv` — layer, R², n_proteins, n_total_pairs
- `csv_files/layer_regression_coefficients_predicted.csv` — 256 regression coefficients + intercept per layer
- `csv_files/predicted_distances_last_layer_predicted.csv` — per-pair actual vs predicted distance with train/test label
- `visualizations/layer_regression_r2_predicted.png` — R² vs layer curve
- `visualizations/pred_vs_actual_all_predicted.png` — scatter, all proteins combined
- `visualizations/pred_vs_actual_per_protein_predicted.png` — scatter grid, one panel per protein

## Results (4 proteins, local run)

- Proteins: 7b3a_A (280 res), 6tf4_A (67 res), 6wmk_A (69 res), 6wqc_A (294 res)
- Total pairs: 86,688
- Peak R²: 0.984 at layer 45
- Layer 47 R²: 0.982

## Flagged Code (needs updating when scaling)

Three functions are marked with `⚠️ FLAG` comments — these encode assumptions about directory layout and file naming that may change on the cluster:

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

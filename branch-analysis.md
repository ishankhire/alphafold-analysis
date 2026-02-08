# Branch Analysis

## Session: 2026-02-07 — Git setup

### Changes made

**`.gitignore` added**
Excludes large data and generated files from version control:
- `*.npy`, `*.cif` — binary data files
- `7b3a_A/`, `pca_projections/`, `feature_magnitudes/` — large data directories
- `.venv/`, `.DS_Store`, `__pycache__/`, `*.pyc` — environment/OS artifacts
- Generated CSVs: `residue_coordinates.csv`, `residue_distances.csv`, `linear_regression_results.csv`, `layer_cosine_change_analysis.csv`, `spatial_range_magnitude_14A.csv`, `spatial_close_pairs_seq_distribution.csv`
- Note: PNG visualizations are intentionally **included** (not ignored) so they appear on GitHub.

**Absolute paths replaced in 8 Python scripts**
All hardcoded absolute filepaths replaced with dynamic equivalents:
- Root-level scripts: `os.path.dirname(os.path.abspath(__file__))`
- `analysis_csv_files/` scripts: `os.path.dirname(os.path.dirname(os.path.abspath(__file__)))` (one extra level up)

Affected files:
- `analyze_distance_correlation.py`
- `analyze_spatial_range_magnitudes.py`
- `extract_feature_magnitudes.py`
- `linear_regression_distance.py`
- `analyze_layer_updates.py`
- `analyze_range_magnitudes.py`
- `analysis_csv_files/compare_spatial_thresholds.py`
- `analysis_csv_files/analyze_pca_changes.py`

## Session: 2026-02-08 — Visualization output refactor

### What changed
All 10 scripts that generate plots now save to the `visualizations/` directory instead of scattered locations (project root, `feature_magnitudes/`, `pc_maps/`, `analysis_csv_files/`).

### Why
Visualizations were saved in inconsistent locations across the project — some in the root, some in data directories, some relative to the script's CWD. Centralizing them in `visualizations/` makes it easy to find all generated figures and keeps data directories clean.

### Scripts modified (20 total savefig calls across 10 files)
| Script | Output redirect |
|--------|----------------|
| `analyze_distance_correlation.py` | 3 PNGs → `visualizations/` |
| `visualize_similarity.py` | 1 PNG → `visualizations/` |
| `extract_feature_magnitudes.py` | heatmap PNGs → `visualizations/feature_magnitudes/` (data .npy/.csv stays in `feature_magnitudes/`) |
| `analyze_spatial_range_magnitudes.py` | 1 PNG → `visualizations/` |
| `analyze_range_magnitudes.py` | 1 PNG → `visualizations/` |
| `visualize_pca.py` | 48 PNGs → `visualizations/pc_maps/` |
| `analyze_layer_updates.py` | 2 PNGs → `visualizations/` |
| `linear_regression_distance.py` | 1 PNG → `visualizations/` |
| `analysis_csv_files/compare_spatial_thresholds.py` | 2 PNGs → `visualizations/` (uses `base_dir` for absolute path) |
| `analysis_csv_files/analyze_pca_changes.py` | 4 PNGs → `visualizations/` (uses `base_dir` for absolute path) |

### CLAUDE.md updates
Added three new sections:
1. **Visualizations** — all plots save to `visualizations/`
2. **Change Log** — record substantive changes in `branch-analysis.md`
3. **Git Workflow** — `git add -A && commit && push` after significant changes

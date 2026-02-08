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
All hardcoded `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein` paths replaced with dynamic equivalents:
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

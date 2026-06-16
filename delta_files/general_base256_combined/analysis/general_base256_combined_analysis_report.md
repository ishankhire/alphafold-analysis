# general_base256_combined analysis

Inputs:
- `delta_files/general_base256_combined/per_protein.csv`
- `delta_files/general_base256_combined/geometry_metrics.csv`
- `delta_files/general_base256_combined/coefficients.csv`

Loaded proteins: 552
Train proteins: 441
Held-out proteins: 111

Held-out layer highlights:
- Best mean R^2: layer 44 (0.9406)
- Best mean TM-score: layer 45 (0.8693)
- Lowest mean triangle violation proportion: layer 18 (0.7968)
- Highest mean top-3 eigenvalue fraction: layer 45 (0.6462)

Individual triangle violation data:
The current saved artifacts do not contain one row per triangle/triple. They only contain aggregate per-protein fields: `triangle_violation_fraction`, `triangle_mean_violation`, and `triangle_max_violation`.

The geometry helper computes a min-indirect-path summary for each directed residue pair, then saves only the fraction, mean violation magnitude, and max violation magnitude. It does not save each `(i, j, k)` triple, the intermediate residue, or each individual violation amount. Recovering those details would require a replay/export job from predicted distance matrices.

Baseline comparison caveat:
The baseline plot uses saved held-out summaries. `general_base256_combined` has 552 loaded proteins with 111 held-out proteins; `baseline256_kfold` has 184 proteins, R^2 pooled across five test folds, and TM-score from the saved fold-4 geometry file.

Generated artifacts:
- `delta_files/general_base256_combined/analysis/general_base256_combined_r2_tmscore_vs_layer.png`
- `delta_files/general_base256_combined/analysis/general_base256_combined_r2_tmscore_vs_layer_mean_only.png`
- `delta_files/general_base256_combined/analysis/general_base256_combined_layer_cosine_matrix.csv`
- `delta_files/general_base256_combined/analysis/general_base256_combined_layer_cosine_heatmap.png`
- `delta_files/general_base256_combined/analysis/general_base256_combined_geometry_layer_means.csv`
- `delta_files/general_base256_combined/analysis/general_base256_combined_triangle_violations_vs_layer.png`
- `delta_files/general_base256_combined/analysis/general_base256_combined_top3_eigenvalue_fraction_vs_layer.png`
- `delta_files/general_base256_combined/analysis/general_base256_combined_negative_eigenvalues_vs_layer.png`
- `delta_files/general_base256_combined/analysis/general_base256_combined_baseline256_comparison_layer_means.csv`
- `delta_files/general_base256_combined/analysis/general_base256_combined_vs_baseline256_kfold_r2_tm_score.png`

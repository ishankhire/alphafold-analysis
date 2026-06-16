# per_protein_base256_combined analysis

Inputs:
- `delta_files/per_protein_base256_combined/per_protein.csv`
- `delta_files/per_protein_base256_combined/geometry_metrics.csv`

Loaded proteins: 552
CAMEO proteins: 154
ECOD proteins: 398

Layer highlights:
- Best mean R^2: layer 45 (0.9876)
- Best mean TM-score: layer 46 (0.9423)
- Lowest mean triangle violation proportion: layer 44 (0.6754)
- Highest mean top-3 eigenvalue fraction: layer 45 (0.7655)

Individual triangle violation data:
The current saved artifacts do not contain one row per triangle/triple. They only contain aggregate per-protein fields: `triangle_violation_fraction`, `triangle_mean_violation`, and `triangle_max_violation`.

The geometry helper computes a min-indirect-path summary for each directed residue pair, then saves only the fraction, mean violation magnitude, and max violation magnitude. It does not save each `(i, j, k)` triple, the intermediate residue, or each individual violation amount. Recovering those details would require a replay/export job from predicted distance matrices.

Comparison caveat:
The comparison plot uses `per_protein_base256_combined` all-protein means and `general_base256_combined` held-out/test means. The per-protein run fits one model per protein using all unordered residue pairs, while the generalized run fits a shared train/test model.

Generated artifacts:
- `delta_files/per_protein_base256_combined/analysis/per_protein_base256_combined_r2_tmscore_vs_layer.png`
- `delta_files/per_protein_base256_combined/analysis/per_protein_base256_combined_r2_tmscore_vs_layer_mean_only.png`
- `delta_files/per_protein_base256_combined/analysis/per_protein_base256_combined_geometry_layer_means.csv`
- `delta_files/per_protein_base256_combined/analysis/per_protein_base256_combined_triangle_violations_vs_layer.png`
- `delta_files/per_protein_base256_combined/analysis/per_protein_base256_combined_top3_eigenvalue_fraction_vs_layer.png`
- `delta_files/per_protein_base256_combined/analysis/per_protein_base256_combined_negative_eigenvalues_vs_layer.png`
- `delta_files/per_protein_base256_combined/analysis/per_protein_base256_combined_general_base256_combined_comparison_layer_means.csv`
- `delta_files/per_protein_base256_combined/analysis/per_protein_base256_combined_vs_general_base256_combined_r2_tm_score.png`

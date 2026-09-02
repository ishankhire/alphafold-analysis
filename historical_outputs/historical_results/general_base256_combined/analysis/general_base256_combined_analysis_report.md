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
- Lowest mean triangle violation fraction: layer 1 (0.0778)
- Highest mean severe (>=10 A) violation fraction: layer 10 (0.0032)
- Highest mean top-3 eigenvalue fraction: layer 45 (0.6462)

Triangle inequality metrics (exact):
The triangle columns in `geometry_metrics.csv` / `geometry_summary.csv` are the exact per-triangle metrics sourced from `exact_triangle_metrics/exact_triangle_metrics.csv`, replacing the earlier min-indirect-path triangle fields. For every protein and layer, all C(n_residues, 3) unordered residue triples are enumerated from the reconstructed predicted distance matrix. A triple is a violation when `(longest side) - (sum of the two shorter sides)` exceeds 1e-12 A; that difference is the violation magnitude. Fields:
- `triangle_violation_fraction`: fraction of triples that violate the inequality.
- `triangle_no_violation_fraction`: 1 - `triangle_violation_fraction`.
- `triangle_violation_0_5_fraction` / `_5_10_fraction` / `_10_plus_fraction`: violating triples binned by magnitude in Angstrom; the three bins sum to `triangle_violation_fraction`.
- `triangle_mean_violation`: sum of violation magnitudes divided by all triples (not just violating ones).
- `triangle_max_violation`: largest single violation magnitude.

These are per-protein aggregates; individual `(i, j, k)` triples are not stored.

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

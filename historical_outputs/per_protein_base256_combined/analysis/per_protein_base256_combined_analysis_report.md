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
- Lowest mean triangle violation fraction: layer 44 (0.0415)
- Highest mean severe (>=10 A) violation fraction: layer 6 (0.0031)
- Highest mean top-3 eigenvalue fraction: layer 45 (0.7655)

Triangle inequality metrics (exact):
The triangle columns in `geometry_metrics.csv` / `geometry_summary.csv` are the exact per-triangle metrics sourced from `exact_triangle_metrics/exact_triangle_metrics.csv`, replacing the earlier min-indirect-path triangle fields. For every protein and layer, all C(n_residues, 3) unordered residue triples are enumerated from the reconstructed predicted distance matrix. A triple is a violation when `(longest side) - (sum of the two shorter sides)` exceeds 1e-12 A; that difference is the violation magnitude. Fields:
- `triangle_violation_fraction`: fraction of triples that violate the inequality.
- `triangle_no_violation_fraction`: 1 - `triangle_violation_fraction`.
- `triangle_violation_0_5_fraction` / `_5_10_fraction` / `_10_plus_fraction`: violating triples binned by magnitude in Angstrom; the three bins sum to `triangle_violation_fraction`.
- `triangle_mean_violation`: sum of violation magnitudes divided by all triples (not just violating ones).
- `triangle_max_violation`: largest single violation magnitude.

These are per-protein aggregates; individual `(i, j, k)` triples are not stored.

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

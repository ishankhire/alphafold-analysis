# Exact Triangle Metrics Analysis

This note documents the local exact-triangle analysis for `general_base256_combined`.

Inputs:
- Layer shards are stored in `delta_files/general_base256_combined/exact_triangle_metrics/layer_batches/`.
- The combined per-protein file is `delta_files/general_base256_combined/exact_triangle_metrics/exact_triangle_metrics.csv`.

Generated analysis artifacts:
- `delta_files/general_base256_combined/analysis/general_base256_combined_exact_triangle_layer_means.csv`
- `delta_files/general_base256_combined/analysis/general_base256_combined_exact_triangle_violation_proportions.png`
- `delta_files/general_base256_combined/analysis/general_base256_combined_exact_triangle_mean_violation.png`

Triangle fields:
- `triangle_no_violation_fraction`: fraction of unordered residue triples with no triangle inequality violation.
- `triangle_violation_0_5_fraction`: fraction of unordered residue triples with a positive violation below 5 Angstrom.
- `triangle_violation_5_10_fraction`: fraction of unordered residue triples with a positive violation from 5 Angstrom up to, but not including, 10 Angstrom.
- `triangle_violation_10_plus_fraction`: fraction of unordered residue triples with a positive violation of at least 10 Angstrom.
- `triangle_violation_fraction`: sum of the three positive violation bins.
- `triangle_mean_violation`: per-protein violation magnitude sum divided by all unordered residue triples.

Layer summaries:
- The plots and layer summary CSV use an unweighted mean across all 552 proteins in each layer.
- Train and held-out proteins are not separated in these exact-triangle plots.

Compatibility note:
- Future `geometry_metrics.csv` files should include the same triangle fields so downstream analyses can use the same plotting and validation logic.

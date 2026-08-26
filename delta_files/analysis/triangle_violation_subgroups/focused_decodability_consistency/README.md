# Focused normalized decodability–consistency grids

These five 2×4 phase grids combine the layout and fixed reference-geometry categories from `focused_main_figures` with the decodability-versus-consistency trajectory style of main Figure 8. Columns are probe models; rows are triangle violation fraction and normalized mean triangle violation.

## Normalization and metrics

The x-axis starts from the protein-equal mean test-set `all_pairs` R² at each layer, then min–max normalizes those 48 means separately within each model. A value of 0 is that model's lowest layerwise mean R² and 1 is its highest. This makes relative layerwise progression comparable but removes absolute R² differences between models.

`mean_relative_violation_all` is not min–max normalized. For each triangle, the positive triangle-inequality excess is divided by that reference triangle's RMS side length and then averaged over all triangles; nonviolating triangles contribute zero. Both y-metrics are protein-equal means over 111 proteins using the canonical `all_edges` triangle scope.

## Figures

| Figure | Categories |
|---|---|
| 01 | All triangles |
| 02 | RMS size: small, medium, large |
| 03 | Area: low, medium, high |
| 04 | Relative slack: low, medium, high |
| 05 | Largest-angle shape: clearly acute, near right, clearly obtuse |

Paths connect layers 0–47 in order. Labels mark layers 0, 10, 20, 30, 40, and 47; circles and stars emphasize layers 0 and 47. Category membership is fixed from final-PDB reference geometry. Axes are shared across model columns within each row.

## Evaluation caveats

The per-protein linear model's `all_pairs` R² mixes its fitted and held-out pair subsets because triangle consistency requires a complete decoded distance matrix. The other three probes evaluate unseen test proteins. These plots are descriptive; no confidence bands, bootstrap resampling, or inferential tests are included.

Exact coordinates are in `decodability_consistency_points.csv`. Regenerate with:

```bash
.venv/bin/python scripts/multi_protein/generate_normalized_decodability_consistency.py
```

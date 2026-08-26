# Focused triangle-violation main figures

This five-figure set retains only the triangle-inequality violation fraction and the normalized mean triangle violation (`mean_relative_violation_all`). The latter is averaged over all triangles, so nonviolating triangles contribute zero.

## Layout

`figure_01_all_triangles_key_metrics` overlays the four models in one panel per metric. Figures 02–05 use columns for models, rows for metrics, and colored lines for the three categories. Axes are shared across model columns within each metric row, allowing direct category and model comparisons without rescaling.

| Figure | Categories |
|---|---|
| 01 | All triangles |
| 02 | RMS size: small, medium, large |
| 03 | Area: low, medium, high |
| 04 | Relative slack: low, medium, high |
| 05 | Largest-angle shape: clearly acute, near right, clearly obtuse |

All values are protein-equal means over the same 111 proteins, use the canonical `all_edges` evaluation scope, and cover Evoformer layers 0–47. Category membership is fixed from final-PDB reference geometry.

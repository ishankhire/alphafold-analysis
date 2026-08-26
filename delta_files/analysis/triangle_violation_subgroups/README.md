# Triangle-violation subgroup trajectories

## Scope and interpretation

This directory visualizes all 7 reference category systems, all 21 category labels, all 13 canonical violation metrics, all 4 frozen probe families, and all 48 Evoformer layers in `triangle_metric_summary.csv`. Values are protein-equal means over 111 analysis proteins; every trajectory uses the canonical `all_edges` scope. The category PDFs are the direct subgroup analogue of `main_figure_05_triangle_violation_diagnostics.pdf`.

There are 18 comprehensive diagnostic figures plus 5 focused main figures, each saved as both PDF and PNG. The `absolute` figures reproduce the Figure-5-style violation fraction, absolute burden/severity, and 0–5/5–10/10+ Å panels. The `relative` figures add all seven scale-normalized metrics. In subgroup matrices, rows are models, columns are metrics, and colored lines are reference categories.

## Overall layer trajectories

| Model | Layer 0 | Layer 47 | Change | Minimum (layer) |
|---|---:|---:|---:|---:|
| Shared linear | 0.0782 | 0.0902 | +0.0121 | 0.0778 (L1) |
| Per-protein linear | 0.0741 | 0.0433 | -0.0308 | 0.0396 (L44) |
| XGBoost | 0.0580 | 0.0400 | -0.0180 | 0.0395 (L46) |
| Sequence gated | 0.0613 | 0.0646 | +0.0033 | 0.0587 (L1) |

At layer 47, XGBoost and the per-protein linear probe have the lowest mean violation fractions, whereas the shared linear and sequence-gated probes end higher. This is descriptive geometric consistency, not a stand-alone accuracy ranking: the per-protein model also has a mixed within-protein train/held-out generalization scope, unlike the three unseen-protein models.

## Layer-47 subgroup ordering

Across models, the strongest and most stable separation is geometric. Clearly obtuse, low-slack, and low-area reference triangles carry much larger violation fractions than clearly acute, high-slack, and high-area triangles. The size-by-angle figures show that angle/slack-like degeneracy is more discriminating than RMS size alone.

| Model | Category system | Lowest layer-47 category | Highest layer-47 category |
|---|---|---:|---:|
| Shared linear | Reference RMS-size tertiles | Medium (0.0737) | Small (0.1111) |
| Shared linear | Reference largest-angle classes | Clearly Acute (0.0097) | Clearly Obtuse (0.1481) |
| Shared linear | Exact reference largest-angle classes | Acute (0.0115) | Obtuse (0.1415) |
| Shared linear | Reference RMS size by largest angle | Large · Clearly Acute (0.0020) | Small · Clearly Obtuse (0.1767) |
| Shared linear | Reference relative-slack tertiles | High (0.0053) | Low (0.2341) |
| Shared linear | Reference area tertiles | High (0.0143) | Low (0.2064) |
| Per-protein linear | Reference RMS-size tertiles | Large (0.0327) | Small (0.0571) |
| Per-protein linear | Reference largest-angle classes | Clearly Acute (0.0014) | Clearly Obtuse (0.0734) |
| Per-protein linear | Exact reference largest-angle classes | Acute (0.0018) | Obtuse (0.0696) |
| Per-protein linear | Reference RMS size by largest angle | Large · Clearly Acute (0.0002) | Small · Clearly Obtuse (0.0987) |
| Per-protein linear | Reference relative-slack tertiles | High (0.0009) | Low (0.1233) |
| Per-protein linear | Reference area tertiles | High (0.0014) | Low (0.1163) |
| XGBoost | Reference RMS-size tertiles | Medium (0.0320) | Large (0.0476) |
| XGBoost | Reference largest-angle classes | Clearly Acute (0.0005) | Clearly Obtuse (0.0684) |
| XGBoost | Exact reference largest-angle classes | Acute (0.0006) | Obtuse (0.0647) |
| XGBoost | Reference RMS size by largest angle | Medium · Clearly Acute (0.0001) | Large · Clearly Obtuse (0.0758) |
| XGBoost | Reference relative-slack tertiles | High (0.0004) | Low (0.1167) |
| XGBoost | Reference area tertiles | High (0.0080) | Low (0.0945) |
| Sequence gated | Reference RMS-size tertiles | Medium (0.0547) | Small (0.0708) |
| Sequence gated | Reference largest-angle classes | Clearly Acute (0.0028) | Clearly Obtuse (0.1097) |
| Sequence gated | Exact reference largest-angle classes | Acute (0.0032) | Obtuse (0.1040) |
| Sequence gated | Reference RMS size by largest angle | Large · Clearly Acute (0.0006) | Small · Clearly Obtuse (0.1204) |
| Sequence gated | Reference relative-slack tertiles | High (0.0018) | Low (0.1786) |
| Sequence gated | Reference area tertiles | High (0.0111) | Low (0.1445) |

## Protein length

| Model | Layer-47 slope per 1 SD log length | Spearman ρ |
|---|---:|---:|
| Shared linear | -0.0150 | -0.648 |
| Per-protein linear | +0.0029 | +0.106 |
| XGBoost | +0.0012 | -0.132 |
| Sequence gated | +0.0005 | -0.022 |

The shared linear probe is the only model with a large layer-47 monotonic length association: longer proteins have lower violation fractions. The other three slopes are near zero, although their length-quintile trajectories are not perfectly monotone. Use the length-bin figures for composition-sensitive checks and the slope figures for the continuous association.

## Files and caveats

- `category_trajectories/` contains the complete category-system figure suite.
- `focused_main_figures/` contains the five reduced figures using only violation fraction and normalized mean triangle violation.
- `length_diagnostics/` contains length-quintile trajectories and continuous length-effect slopes.
- `trajectory_changes.csv` records layer-0/layer-47 values and trajectory extrema for every model/category/metric.
- `paired_model_contrasts_layer47.csv` and `protein_length_effects_layer47.csv` retain compact final-layer sensitivity tables.
- `figure_manifest.csv` and `generation_summary.json` record figure provenance and source SHA-256 digests.
- Absolute and relative band fractions use all triangles as the denominator. Conditional means and p99 values condition on violations.
- Categories are fixed from final-PDB reference geometry within each protein; they are not inferred from decoded distances.
- The exact-90° `right_within_tolerance` category has no finite support at the workflow's 10⁻¹² relative equality tolerance, so its legend entry is retained but its trajectories are blank.
- No bootstrap resampling or inferential significance testing is performed. Bands are omitted to keep the dense all-category plots readable.

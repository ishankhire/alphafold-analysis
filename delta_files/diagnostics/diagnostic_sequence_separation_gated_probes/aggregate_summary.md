# Sequence-separation-gated probe aggregate summary

Generated: 2026-08-14T12:50:22+05:30

## Held-out metrics

| Model | Layer | Mean protein R² | Median protein R² | Pooled R² | Mean protein MAE (Å) | Pooled MAE (Å) | Pooled RMSE (Å) |
|---|---:|---:|---:|---:|---:|---:|---:|
| base256_gated_by_sequence_separation | 0 | 0.5202 | 0.5717 | 0.5692 | 4.780 | 6.122 | 9.034 |
| base256_gated_by_sequence_separation | 10 | 0.6354 | 0.6696 | 0.6360 | 4.181 | 5.591 | 8.304 |
| base256_gated_by_sequence_separation | 20 | 0.8595 | 0.8788 | 0.8871 | 2.468 | 2.961 | 4.625 |
| base256_gated_by_sequence_separation | 30 | 0.8615 | 0.8849 | 0.8873 | 2.414 | 2.893 | 4.620 |
| base256_gated_by_sequence_separation | 40 | 0.9244 | 0.9342 | 0.9263 | 1.790 | 2.221 | 3.736 |
| base256_gated_by_sequence_separation | 47 | 0.9574 | 0.9611 | 0.9623 | 1.382 | 1.714 | 2.671 |
| sequence_separation_only | 0 | 0.1078 | 0.1378 | 0.2921 | 6.952 | 8.577 | 11.580 |
| sequence_separation_only | 10 | 0.1078 | 0.1378 | 0.2921 | 6.952 | 8.577 | 11.580 |
| sequence_separation_only | 20 | 0.1078 | 0.1378 | 0.2921 | 6.952 | 8.577 | 11.580 |
| sequence_separation_only | 30 | 0.1078 | 0.1378 | 0.2921 | 6.952 | 8.577 | 11.580 |
| sequence_separation_only | 40 | 0.1078 | 0.1378 | 0.2921 | 6.952 | 8.577 | 11.580 |
| sequence_separation_only | 47 | 0.1078 | 0.1378 | 0.2921 | 6.952 | 8.577 | 11.580 |

## Calculation

The six sequence-separation bins are recombined within each protein by summing pair counts, squared errors, absolute errors, and target moments. Whole-protein R² is then computed from the combined residual and total sums of squares. It is not an average of bin-level R² values.

`mean_per_protein_*` gives every protein equal weight. `pooled_*` gives every residue pair equal weight across the split.

The source per-bin CSV values are rounded to six decimal places, so the reconstructed metrics can differ slightly from metrics computed from raw predictions.

Source files: 6 `per_protein_layer_XX.csv` files.

# ECOD category analysis

Uses 398 annotated ECOD proteins. The category file contains 400 entries; `000196893` (beta) and `001681437` (alpha_beta) are absent from both probe outputs. Shared-probe results pool saved train and test rows.

## Best category-layer means

- per_protein r2: alpha_plus_beta at layer 45 (0.9905)
- per_protein tm_score: alpha_beta at layer 45 (0.9737)
- shared r2: alpha_beta at layer 44 (0.9539)
- shared tm_score: alpha_beta at layer 45 (0.9230)

## Coefficients

The largest observed within-minus-between cosine gap occurs at layer 0. The global across-layer permutation p-value is 0.0004998.

## Length-adjusted category effects

- shared r2 (layer 44): partial R² 0.0080, category-effect p = 0.3668.
- shared tm_score (layer 45): partial R² 0.0757, category-effect p = 8.558e-07.
- per_protein r2 (layer 45): partial R² 0.0199, category-effect p = 0.0477.
- per_protein tm_score (layer 46): partial R² 0.0584, category-effect p = 2.904e-05.

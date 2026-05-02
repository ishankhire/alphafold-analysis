# Publication Summary Statistics

Sources: `delta_files/general_base256_avg_symmetric/` and `delta_files/general_base256_avg_symmetric_plus_axial/`.
All statistics are unweighted across proteins unless otherwise noted. `all` combines train and test proteins.

## Layer 47
| Variant | Metric | Split | n | Mean | Median | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Variant A | R^2 | test | 37 | 0.904 | 0.923 | 0.054 | 0.698 | 0.960 |
| Variant A | R^2 | train | 147 | 0.906 | 0.919 | 0.063 | 0.357 | 0.970 |
| Variant A | R^2 | all | 184 | 0.906 | 0.921 | 0.061 | 0.357 | 0.970 |
| Variant A | TM-score | test | 37 | 0.749 | 0.814 | 0.190 | 0.202 | 0.931 |
| Variant A | TM-score | train | 147 | 0.727 | 0.782 | 0.181 | 0.135 | 0.929 |
| Variant A | TM-score | all | 184 | 0.731 | 0.787 | 0.183 | 0.135 | 0.931 |
| Variant B | R^2 | test | 37 | 0.937 | 0.950 | 0.049 | 0.689 | 0.972 |
| Variant B | R^2 | train | 147 | 0.939 | 0.948 | 0.038 | 0.650 | 0.977 |
| Variant B | R^2 | all | 184 | 0.939 | 0.949 | 0.040 | 0.650 | 0.977 |
| Variant B | TM-score | test | 37 | 0.794 | 0.870 | 0.187 | 0.228 | 0.956 |
| Variant B | TM-score | train | 147 | 0.776 | 0.842 | 0.174 | 0.168 | 0.952 |
| Variant B | TM-score | all | 184 | 0.780 | 0.845 | 0.176 | 0.168 | 0.956 |

## TM-score Threshold
| Variant | Split | Threshold | Crossed | First Layer | Mean TM There |
| --- | --- | --- | --- | --- | --- |
| Variant A | test | 0.600 | yes | 24 | 0.606 |
| Variant A | train | 0.600 | yes | 35 | 0.604 |
| Variant A | all | 0.600 | yes | 35 | 0.613 |
| Variant B | test | 0.600 | yes | 18 | 0.618 |
| Variant B | train | 0.600 | yes | 22 | 0.601 |
| Variant B | all | 0.600 | yes | 20 | 0.604 |

## Negative Held-out R^2
| Variant | Test n | Ever Negative n | Ever Negative Fraction | Zero-Negative Layers | Zero Through Layer 47 From |
| --- | --- | --- | --- | --- | --- |
| Variant A | 37 | 7 | 0.189 | 27;29;30;31;32;33;34;35;36;37;38;39;40;41;42;43;44;45;46;47 | 29 |
| Variant B | 37 | 8 | 0.216 | 23;24;25;26;27;28;29;30;31;32;33;34;35;36;37;38;39;40;41;42;43;44;45;46;47 | 23 |

Per-layer negative R^2 counts are in `publication_negative_r2_summary.csv`.

## Additional Useful Numbers
- Variant A held-out R^2 peaks at layer 47 with mean 0.904; layer 47 mean is 0.904.
- Variant A held-out TM-score peaks at layer 45 with mean 0.776; layer 47 mean is 0.749.
- Variant B held-out R^2 peaks at layer 47 with mean 0.937; layer 47 mean is 0.937.
- Variant B held-out TM-score peaks at layer 45 with mean 0.813; layer 47 mean is 0.794.
- At layer 47 on held-out proteins, Variant B minus Variant A mean R^2 delta is 0.032; 36/37 proteins improved (0.973).
- At layer 47 on held-out proteins, Variant B minus Variant A mean TM-score delta is 0.045; 37/37 proteins improved (1.000).
- Variant A layer-47 held-out R^2/TM-score Pearson correlation: 0.497 across 37 proteins.
- Variant B layer-47 held-out R^2/TM-score Pearson correlation: 0.599 across 37 proteins.

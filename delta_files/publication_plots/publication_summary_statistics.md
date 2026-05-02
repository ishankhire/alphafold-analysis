# Publication Summary Statistics

Sources: `delta_files/general_base256_avg_symmetric/` and `delta_files/general_base256_avg_symmetric_plus_axial/`.
All statistics are unweighted across proteins unless otherwise noted. `all` combines train and test proteins.
Variant A MDS diagnostics come from `delta_files/general_base256_avg_symmetric/geometry_metrics.csv`.

## Layer 47
| Variant | Metric | Split | n | Mean | Median | Std | Min | Max |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Variant A | R^2 | test | 37 | 0.904 | 0.923 | 0.054 | 0.698 | 0.960 |
| Variant A | R^2 | train | 147 | 0.906 | 0.919 | 0.063 | 0.357 | 0.970 |
| Variant A | R^2 | all | 184 | 0.906 | 0.921 | 0.061 | 0.357 | 0.970 |
| Variant A | TM-score | test | 37 | 0.749 | 0.814 | 0.190 | 0.202 | 0.931 |
| Variant A | TM-score | train | 147 | 0.727 | 0.782 | 0.181 | 0.135 | 0.929 |
| Variant A | TM-score | all | 184 | 0.731 | 0.787 | 0.183 | 0.135 | 0.931 |
| Variant A | Kruskal stress-1 | test | 37 | 0.120 | 0.110 | 0.034 | 0.080 | 0.265 |
| Variant A | Kruskal stress-1 | train | 147 | 0.119 | 0.111 | 0.025 | 0.081 | 0.254 |
| Variant A | Kruskal stress-1 | all | 184 | 0.119 | 0.111 | 0.027 | 0.080 | 0.265 |
| Variant A | Negative predicted distances | test | 37 | 157.568 | 150.000 | 106.359 | 8.000 | 405.000 |
| Variant A | Negative predicted distances | train | 147 | 169.946 | 144.000 | 114.244 | 7.000 | 516.000 |
| Variant A | Negative predicted distances | all | 184 | 167.457 | 144.500 | 112.530 | 7.000 | 516.000 |
| Variant A | Negative predicted distance fraction | test | 37 | 0.005 | 0.005 | 0.003 | 0.001 | 0.014 |
| Variant A | Negative predicted distance fraction | train | 147 | 0.006 | 0.005 | 0.004 | 0.002 | 0.021 |
| Variant A | Negative predicted distance fraction | all | 184 | 0.006 | 0.005 | 0.003 | 0.001 | 0.021 |
| Variant A | Top-3 eigenvalue fraction | test | 37 | 0.593 | 0.584 | 0.051 | 0.479 | 0.770 |
| Variant A | Top-3 eigenvalue fraction | train | 147 | 0.599 | 0.589 | 0.051 | 0.517 | 0.809 |
| Variant A | Top-3 eigenvalue fraction | all | 184 | 0.598 | 0.588 | 0.051 | 0.479 | 0.809 |
| Variant A | Negative eigenvalue count fraction | test | 37 | 0.480 | 0.481 | 0.011 | 0.449 | 0.496 |
| Variant A | Negative eigenvalue count fraction | train | 147 | 0.485 | 0.487 | 0.011 | 0.441 | 0.502 |
| Variant A | Negative eigenvalue count fraction | all | 184 | 0.484 | 0.486 | 0.011 | 0.441 | 0.502 |
| Variant A | Negative eigenvalue mass fraction | test | 37 | 0.402 | 0.415 | 0.062 | 0.208 | 0.487 |
| Variant A | Negative eigenvalue mass fraction | train | 147 | 0.400 | 0.404 | 0.053 | 0.232 | 0.510 |
| Variant A | Negative eigenvalue mass fraction | all | 184 | 0.400 | 0.407 | 0.054 | 0.208 | 0.510 |
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

## Variant A MDS Geometry
`Kruskal stress-1` is `sqrt(sum((predicted_distance - MDS_distance)^2) / sum(predicted_distance^2))`, so lower values mean the 3D MDS embedding better preserves the regressed distance matrix. `Top-3 eigenvalue fraction` is the share of positive classical-MDS eigenvalue mass captured by the three retained dimensions. The negative eigenvalue count fraction is the share of centered-distance eigenvalues below zero; the negative eigenvalue mass fraction is the saved `neg_eigenvalue_fraction` field.

| Split | Stress Mean | Top-3 Eig Frac Mean | Neg Eig Count Frac Mean | Neg Eig Mass Frac Mean | Neg Pred Pair Frac | Proteins With Any Neg Pred |
| --- | --- | --- | --- | --- | --- | --- |
| test | 0.120 | 0.593 | 0.480 | 0.402 | 0.0037 | 1.000 |
| train | 0.119 | 0.599 | 0.485 | 0.400 | 0.0037 | 1.000 |
| all | 0.119 | 0.598 | 0.484 | 0.400 | 0.0037 | 1.000 |

Full per-layer rows for these diagnostics are in `publication_extra_paper_stats.csv`.

## Variant A Kruskal Stress By Layer
| Layer | Test Mean | Train Mean | All Mean |
| --- | --- | --- | --- |
| 0 | 0.306 | 0.302 | 0.303 |
| 1 | 0.301 | 0.297 | 0.298 |
| 2 | 0.299 | 0.294 | 0.295 |
| 3 | 0.298 | 0.295 | 0.296 |
| 4 | 0.301 | 0.298 | 0.299 |
| 5 | 0.301 | 0.297 | 0.297 |
| 6 | 0.305 | 0.299 | 0.300 |
| 7 | 0.299 | 0.296 | 0.297 |
| 8 | 0.286 | 0.282 | 0.283 |
| 9 | 0.280 | 0.275 | 0.276 |
| 10 | 0.265 | 0.264 | 0.264 |
| 11 | 0.257 | 0.256 | 0.256 |
| 12 | 0.243 | 0.243 | 0.243 |
| 13 | 0.202 | 0.207 | 0.206 |
| 14 | 0.203 | 0.206 | 0.206 |
| 15 | 0.187 | 0.192 | 0.191 |
| 16 | 0.170 | 0.175 | 0.174 |
| 17 | 0.161 | 0.164 | 0.163 |
| 18 | 0.152 | 0.156 | 0.155 |
| 19 | 0.153 | 0.157 | 0.156 |
| 20 | 0.146 | 0.149 | 0.149 |
| 21 | 0.146 | 0.149 | 0.148 |
| 22 | 0.143 | 0.146 | 0.146 |
| 23 | 0.143 | 0.147 | 0.146 |
| 24 | 0.144 | 0.147 | 0.146 |
| 25 | 0.143 | 0.146 | 0.145 |
| 26 | 0.143 | 0.147 | 0.146 |
| 27 | 0.141 | 0.146 | 0.145 |
| 28 | 0.144 | 0.148 | 0.147 |
| 29 | 0.146 | 0.150 | 0.149 |
| 30 | 0.150 | 0.153 | 0.153 |
| 31 | 0.153 | 0.155 | 0.155 |
| 32 | 0.146 | 0.150 | 0.150 |
| 33 | 0.145 | 0.150 | 0.149 |
| 34 | 0.144 | 0.148 | 0.147 |
| 35 | 0.133 | 0.135 | 0.135 |
| 36 | 0.136 | 0.137 | 0.137 |
| 37 | 0.131 | 0.134 | 0.134 |
| 38 | 0.129 | 0.132 | 0.131 |
| 39 | 0.125 | 0.129 | 0.128 |
| 40 | 0.129 | 0.131 | 0.131 |
| 41 | 0.128 | 0.131 | 0.130 |
| 42 | 0.122 | 0.124 | 0.124 |
| 43 | 0.124 | 0.123 | 0.123 |
| 44 | 0.111 | 0.111 | 0.111 |
| 45 | 0.114 | 0.113 | 0.113 |
| 46 | 0.120 | 0.118 | 0.118 |
| 47 | 0.120 | 0.119 | 0.119 |

## Additional Useful Numbers
- Variant A held-out R^2 peaks at layer 47 with mean 0.904; layer 47 mean is 0.904.
- Variant A held-out TM-score peaks at layer 45 with mean 0.776; layer 47 mean is 0.749.
- Variant B held-out R^2 peaks at layer 47 with mean 0.937; layer 47 mean is 0.937.
- Variant B held-out TM-score peaks at layer 45 with mean 0.813; layer 47 mean is 0.794.
- At layer 47 on held-out proteins, Variant B minus Variant A mean R^2 delta is 0.032; 36/37 proteins improved (0.973).
- At layer 47 on held-out proteins, Variant B minus Variant A mean TM-score delta is 0.045; 37/37 proteins improved (1.000).
- Variant A layer-47 held-out R^2/TM-score Pearson correlation: 0.497 across 37 proteins.
- Variant B layer-47 held-out R^2/TM-score Pearson correlation: 0.599 across 37 proteins.
- Variant A held-out Kruskal stress-1 is lowest at layer 44 with mean 0.111; layer 47 mean is 0.120.
- Variant A layer-47 held-out MDS diagnostics: top-3 eigenvalue fraction mean 0.593, negative eigenvalue count fraction mean 0.480, negative predicted pair fraction 0.0037, and proteins with any negative prediction 1.000.

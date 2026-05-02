# Publication Geometry Diagnostics

Source: `delta_files/general_base256_avg_symmetric/geometry_metrics.csv`.
These diagnostics are for Variant A: averaged pair representations.
All statistics are unweighted across proteins unless otherwise noted. `all` combines train and test proteins.

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

## Additional Geometry Numbers
- Variant A held-out Kruskal stress-1 is lowest at layer 44 with mean 0.111; layer 47 mean is 0.120.
- Variant A layer-47 held-out MDS diagnostics: top-3 eigenvalue fraction mean 0.593, negative eigenvalue count fraction mean 0.480, negative predicted pair fraction 0.0037, and proteins with any negative prediction 1.000.

# Sanity checks

- Slim reducer vs `compute_metric_rows`, bit-identical: 9600 comparisons, 0 failures.
- Exact observed vs the paper's stored per-protein values: 29568 comparisons, worst violation-fraction gap 1.75e-05, worst normalized-sum relative gap 1.27e-07, worst negative-count gap 0 (float32 cache vs float64 replay).
- Null B kept each bin's residual histogram in all 739200 draws (0 failures).

## Null A vs Null B, XGBoost, layer 47
- Normalized mean violation: A − B = 0.0003807 [0.0003038, 0.0004742] (differ).
- Violation fraction: A − B = 0.005805 [0.005335, 0.006347] (differ).

Per-bin residual shape at layer 47 (protein-equal means):

| cell_label | protein_mean_skew | protein_mean_excess_kurtosis | n_proteins_contributing |
|---|---|---|---|
| [0,4) | 4.648 | 34.87 | 154 |
| [4,6) | 3.215 | 19.54 | 154 |
| [6,8) | 3.787 | 26.22 | 154 |
| [8,10) | 2.656 | 30.62 | 154 |
| [10,14) | 1.257 | 15.01 | 154 |
| [14,20) | 0.9881 | 10.09 | 154 |
| [20,30) | 0.8192 | 7.865 | 154 |
| [30,40) | 0.5188 | 5.622 | 153 |
| [40,55) | 0.5668 | 2.254 | 142 |
| [55,inf) | 0.03086 | 0.5533 | 108 |

## Monte Carlo noise (K draws per protein)

| Model | Binning | Metric | Mean per-protein draw SD | Cohort-mean MC SE | Median bootstrap CI half-width |
|---|---|---|---|---|---|
| null_a | B10 | Normalized mean violation | 0.000526 | 2.51e-05 | 0.00279 |
| null_a | B10 | Violation fraction | 0.00184 | 6.69e-05 | 0.00848 |
| null_b | B10 | Normalized mean violation | 0.000161 | 6.74e-06 | 0.00262 |
| null_b | B10 | Violation fraction | 0.000649 | 2.24e-05 | 0.00817 |
| null_c | B10 | Normalized mean violation | 0.000199 | 8.96e-06 | 0.0024 |
| null_c | B10 | Violation fraction | 0.000732 | 2.58e-05 | 0.00771 |
| null_b | B10xS6 | Normalized mean violation | 0.000155 | 6.38e-06 | 0.0025 |
| null_b | B10xS6 | Violation fraction | 0.000636 | 2.18e-05 | 0.00814 |
| null_c | B10xS6 | Normalized mean violation | 0.000184 | 8.01e-06 | 0.00226 |
| null_c | B10xS6 | Violation fraction | 0.000702 | 2.44e-05 | 0.00764 |

## Robustness: observed − Null B (B10) without 7stz_C

| Probe | Metric | Layer | All proteins | Without 7stz_C |
|---|---|---|---|---|
| Shared OLS | Normalized mean violation | 0 | -0.01748 [-0.02057, -0.01517] | -0.01628 [-0.01813, -0.0148] |
| Shared OLS | Normalized mean violation | 10 | -0.02008 [-0.02423, -0.01648] | -0.01887 [-0.02227, -0.01571] |
| Shared OLS | Normalized mean violation | 21 | -0.01467 [-0.01786, -0.01223] | -0.01347 [-0.01526, -0.01184] |
| Shared OLS | Normalized mean violation | 32 | -0.0123 [-0.01533, -0.0101] | -0.01116 [-0.01267, -0.00978] |
| Shared OLS | Normalized mean violation | 40 | -0.008213 [-0.01108, -0.006357] | -0.007097 [-0.008096, -0.006211] |
| Shared OLS | Normalized mean violation | 47 | -0.005597 [-0.008068, -0.004085] | -0.004575 [-0.005226, -0.00397] |
| Shared OLS | Violation fraction | 0 | -0.05193 [-0.06151, -0.04543] | -0.04818 [-0.05185, -0.04484] |
| Shared OLS | Violation fraction | 10 | -0.05621 [-0.0662, -0.04909] | -0.05233 [-0.05675, -0.04805] |
| Shared OLS | Violation fraction | 21 | -0.05516 [-0.06482, -0.04828] | -0.05159 [-0.05617, -0.04741] |
| Shared OLS | Violation fraction | 32 | -0.04368 [-0.05304, -0.03707] | -0.04013 [-0.04414, -0.03647] |
| Shared OLS | Violation fraction | 40 | -0.0343 [-0.04324, -0.02817] | -0.03088 [-0.03462, -0.02761] |
| Shared OLS | Violation fraction | 47 | -0.02432 [-0.03198, -0.01904] | -0.0213 [-0.02438, -0.01843] |
| Per-protein OLS | Normalized mean violation | 0 | -0.01722 [-0.01907, -0.01561] | -0.01679 [-0.01839, -0.01527] |
| Per-protein OLS | Normalized mean violation | 10 | -0.01395 [-0.01583, -0.0124] | -0.0134 [-0.01476, -0.01202] |
| Per-protein OLS | Normalized mean violation | 21 | -0.005875 [-0.00694, -0.005029] | -0.00559 [-0.006394, -0.004838] |
| Per-protein OLS | Normalized mean violation | 32 | -0.004917 [-0.005848, -0.004152] | -0.004672 [-0.005376, -0.003983] |
| Per-protein OLS | Normalized mean violation | 40 | -0.002563 [-0.003042, -0.002107] | -0.002466 [-0.00289, -0.002046] |
| Per-protein OLS | Normalized mean violation | 47 | -0.001403 [-0.001657, -0.001161] | -0.00136 [-0.001607, -0.001116] |
| Per-protein OLS | Violation fraction | 0 | -0.05273 [-0.05763, -0.04837] | -0.05193 [-0.05643, -0.04767] |
| Per-protein OLS | Violation fraction | 10 | -0.04467 [-0.04903, -0.04062] | -0.04372 [-0.04765, -0.03992] |
| Per-protein OLS | Violation fraction | 21 | -0.02753 [-0.03048, -0.02491] | -0.02698 [-0.02957, -0.02436] |
| Per-protein OLS | Violation fraction | 32 | -0.02118 [-0.02395, -0.01862] | -0.0207 [-0.02324, -0.01819] |
| Per-protein OLS | Violation fraction | 40 | -0.01292 [-0.01494, -0.01099] | -0.0126 [-0.01449, -0.01072] |
| Per-protein OLS | Violation fraction | 47 | -0.008215 [-0.009621, -0.006855] | -0.008015 [-0.009423, -0.00662] |
| Seq-gated OLS | Normalized mean violation | 0 | -0.01674 [-0.01979, -0.01466] | -0.01552 [-0.01688, -0.01439] |
| Seq-gated OLS | Normalized mean violation | 10 | -0.01764 [-0.02105, -0.01511] | -0.0164 [-0.01838, -0.01459] |
| Seq-gated OLS | Normalized mean violation | 21 | -0.01344 [-0.01655, -0.01112] | -0.01225 [-0.01393, -0.01073] |
| Seq-gated OLS | Normalized mean violation | 32 | -0.01115 [-0.01398, -0.009112] | -0.01003 [-0.01128, -0.008877] |
| Seq-gated OLS | Normalized mean violation | 40 | -0.007425 [-0.01017, -0.005638] | -0.006344 [-0.00728, -0.005484] |
| Seq-gated OLS | Normalized mean violation | 47 | -0.00447 [-0.006755, -0.003051] | -0.003487 [-0.004061, -0.00295] |
| Seq-gated OLS | Violation fraction | 0 | -0.06302 [-0.07294, -0.0563] | -0.05925 [-0.06329, -0.05559] |
| Seq-gated OLS | Violation fraction | 10 | -0.0663 [-0.07601, -0.05907] | -0.06247 [-0.06723, -0.05796] |
| Seq-gated OLS | Violation fraction | 21 | -0.05832 [-0.06792, -0.05124] | -0.05482 [-0.05984, -0.05015] |
| Seq-gated OLS | Violation fraction | 32 | -0.04926 [-0.05839, -0.04241] | -0.0458 [-0.05017, -0.04176] |
| Seq-gated OLS | Violation fraction | 40 | -0.03629 [-0.04509, -0.03005] | -0.03296 [-0.03693, -0.02948] |
| Seq-gated OLS | Violation fraction | 47 | -0.02434 [-0.03164, -0.01907] | -0.02144 [-0.02459, -0.01851] |
| XGBoost | Normalized mean violation | 0 | -0.01788 [-0.02121, -0.01561] | -0.01663 [-0.01831, -0.01516] |
| XGBoost | Normalized mean violation | 10 | -0.0191 [-0.02269, -0.01643] | -0.01784 [-0.01997, -0.01595] |
| XGBoost | Normalized mean violation | 21 | -0.01284 [-0.01635, -0.01009] | -0.01173 [-0.0141, -0.009572] |
| XGBoost | Normalized mean violation | 32 | -0.01103 [-0.01405, -0.008588] | -0.01002 [-0.01193, -0.008245] |
| XGBoost | Normalized mean violation | 40 | -0.005718 [-0.008262, -0.003919] | -0.004754 [-0.005841, -0.00382] |
| XGBoost | Normalized mean violation | 47 | -0.002663 [-0.004835, -0.001389] | -0.001724 [-0.0022, -0.001302] |
| XGBoost | Violation fraction | 0 | -0.06466 [-0.0747, -0.05803] | -0.06089 [-0.06526, -0.05689] |
| XGBoost | Violation fraction | 10 | -0.06942 [-0.07983, -0.06212] | -0.06569 [-0.07077, -0.06076] |
| XGBoost | Violation fraction | 21 | -0.04873 [-0.05799, -0.04178] | -0.0455 [-0.0507, -0.04055] |
| XGBoost | Violation fraction | 32 | -0.04157 [-0.05039, -0.03441] | -0.03839 [-0.04357, -0.03375] |
| XGBoost | Violation fraction | 40 | -0.02605 [-0.03411, -0.02032] | -0.02302 [-0.02688, -0.01966] |
| XGBoost | Violation fraction | 47 | -0.01316 [-0.02039, -0.008544] | -0.01027 [-0.01266, -0.008113] |

## Null A fallback (proteins per layer using cohort bin statistics)

| probe | bin_label | mean | max |
|---|---|---|---|
| shared_ols_base256 | [0,4) | 0 | 0 |
| shared_ols_base256 | [4,6) | 0 | 0 |
| shared_ols_base256 | [6,8) | 0 | 0 |
| shared_ols_base256 | [8,10) | 0 | 0 |
| shared_ols_base256 | [10,14) | 0 | 0 |
| shared_ols_base256 | [14,20) | 0 | 0 |
| shared_ols_base256 | [20,30) | 1 | 1 |
| shared_ols_base256 | [30,40) | 1 | 1 |
| shared_ols_base256 | [40,55) | 7 | 7 |
| shared_ols_base256 | [55,inf) | 10 | 10 |
| per_protein_ols_base256 | [0,4) | 0 | 0 |
| per_protein_ols_base256 | [4,6) | 0 | 0 |
| per_protein_ols_base256 | [6,8) | 0 | 0 |
| per_protein_ols_base256 | [8,10) | 0 | 0 |
| per_protein_ols_base256 | [10,14) | 0 | 0 |
| per_protein_ols_base256 | [14,20) | 0 | 0 |
| per_protein_ols_base256 | [20,30) | 1 | 1 |
| per_protein_ols_base256 | [30,40) | 1 | 1 |
| per_protein_ols_base256 | [40,55) | 7 | 7 |
| per_protein_ols_base256 | [55,inf) | 10 | 10 |
| sequence_gated_ols_base256 | [0,4) | 0 | 0 |
| sequence_gated_ols_base256 | [4,6) | 0 | 0 |
| sequence_gated_ols_base256 | [6,8) | 0 | 0 |
| sequence_gated_ols_base256 | [8,10) | 0 | 0 |
| sequence_gated_ols_base256 | [10,14) | 0 | 0 |
| sequence_gated_ols_base256 | [14,20) | 0 | 0 |
| sequence_gated_ols_base256 | [20,30) | 1 | 1 |
| sequence_gated_ols_base256 | [30,40) | 1 | 1 |
| sequence_gated_ols_base256 | [40,55) | 7 | 7 |
| sequence_gated_ols_base256 | [55,inf) | 10 | 10 |
| shared_xgboost_base256_fixed_s4 | [0,4) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [4,6) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [6,8) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [8,10) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [10,14) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [14,20) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [20,30) | 1 | 1 |
| shared_xgboost_base256_fixed_s4 | [30,40) | 1 | 1 |
| shared_xgboost_base256_fixed_s4 | [40,55) | 7 | 7 |
| shared_xgboost_base256_fixed_s4 | [55,inf) | 10 | 10 |

# Sanity checks

- Streaming reducer vs `compute_metric_rows`, bit-identical (proteins with ≤ 1,000,000 triangles): 53184 comparisons, 0 failures.
- Observed vs the paper's stored per-protein values: 105984 comparisons, worst violation-fraction gap 2.4e-05, worst normalized-sum relative gap 3.33e-07, worst negative-count gap 0 (float32 cache vs float64 replay).
- Null B kept each bin's residual histogram in all 556416 draws (0 failures).
- Triangles scored per layer per vector: 2,056,326,431.

## Null A vs Null B, XGBoost, layer 47
- Normalized mean violation: A − B = 0.0005177 [0.000478, 0.0005591] (differ).
- Violation fraction: A − B = 0.005287 [0.005037, 0.005554] (differ).

Per-bin residual shape at layer 47 (protein-equal means):

| cell_label | protein_mean_skew | protein_mean_excess_kurtosis | n_proteins_contributing |
|---|---|---|---|
| [0,4) | 4.29 | 30.29 | 552 |
| [4,6) | 3.08 | 17.73 | 552 |
| [6,8) | 3.765 | 25.94 | 552 |
| [8,10) | 2.724 | 31.02 | 552 |
| [10,14) | 1.017 | 13.09 | 552 |
| [14,20) | 0.6658 | 6.525 | 552 |
| [20,30) | 0.5493 | 4.473 | 552 |
| [30,40) | 0.1132 | 3.991 | 535 |
| [40,55) | 0.1827 | 2.491 | 448 |
| [55,inf) | 0.2026 | 0.5765 | 217 |

## Monte Carlo noise (K draws per protein)

| Model | Binning | Metric | Mean per-protein draw SD | Cohort-mean MC SE | Median bootstrap CI half-width |
|---|---|---|---|---|---|
| null_a | B10 | Normalized mean violation | 0.000493 | 2.21e-05 | 0.000946 |
| null_a | B10 | Violation fraction | 0.00188 | 6.75e-05 | 0.00332 |
| null_b | B10 | Normalized mean violation | 0.000136 | 6.06e-06 | 0.000888 |
| null_b | B10 | Violation fraction | 0.000596 | 2.17e-05 | 0.00318 |
| null_b | B5 | Normalized mean violation | 0.000297 | 1.33e-05 | 0.00139 |
| null_b | B5 | Violation fraction | 0.000977 | 3.54e-05 | 0.00386 |
| null_b | B20 | Normalized mean violation | 0.000166 | 7.55e-06 | 0.00109 |
| null_b | B20 | Violation fraction | 0.000549 | 2.04e-05 | 0.00322 |
| null_b | B10xS6 | Normalized mean violation | 0.000187 | 8.19e-06 | 0.00113 |
| null_b | B10xS6 | Violation fraction | 0.000654 | 2.39e-05 | 0.00335 |

## Robustness: observed − Null B (B10) without 7stz_C

| Probe | Metric | Layer | All proteins | Without 7stz_C |
|---|---|---|---|---|
| Shared OLS | Normalized mean violation | 0 | -0.01624 [-0.01725, -0.0154] | -0.01591 [-0.01655, -0.01527] |
| Shared OLS | Normalized mean violation | 10 | -0.01535 [-0.01686, -0.01405] | -0.01501 [-0.01625, -0.01382] |
| Shared OLS | Normalized mean violation | 21 | -0.01024 [-0.0113, -0.009356] | -0.009898 [-0.01061, -0.009192] |
| Shared OLS | Normalized mean violation | 32 | -0.00834 [-0.00932, -0.007556] | -0.008014 [-0.00864, -0.007405] |
| Shared OLS | Normalized mean violation | 40 | -0.005438 [-0.006328, -0.00476] | -0.005123 [-0.005615, -0.004665] |
| Shared OLS | Normalized mean violation | 47 | -0.003432 [-0.004186, -0.002926] | -0.003143 [-0.003453, -0.002864] |
| Shared OLS | Violation fraction | 0 | -0.04619 [-0.04908, -0.044] | -0.04513 [-0.04663, -0.04354] |
| Shared OLS | Violation fraction | 10 | -0.0422 [-0.04545, -0.03963] | -0.0411 [-0.04299, -0.03915] |
| Shared OLS | Violation fraction | 21 | -0.03718 [-0.04038, -0.03453] | -0.03616 [-0.03818, -0.03405] |
| Shared OLS | Violation fraction | 32 | -0.02652 [-0.0295, -0.02412] | -0.0255 [-0.02734, -0.02363] |
| Shared OLS | Violation fraction | 40 | -0.02023 [-0.02303, -0.01806] | -0.01926 [-0.02091, -0.01765] |
| Shared OLS | Violation fraction | 47 | -0.01348 [-0.01588, -0.0117] | -0.01262 [-0.01387, -0.01143] |
| Per-protein OLS | Normalized mean violation | 0 | -0.01235 [-0.01305, -0.01172] | -0.01222 [-0.01287, -0.01158] |
| Per-protein OLS | Normalized mean violation | 10 | -0.008627 [-0.009285, -0.008013] | -0.008463 [-0.009029, -0.007923] |
| Per-protein OLS | Normalized mean violation | 21 | -0.003265 [-0.003611, -0.002948] | -0.003182 [-0.00348, -0.002905] |
| Per-protein OLS | Normalized mean violation | 32 | -0.002504 [-0.002813, -0.002216] | -0.002432 [-0.002699, -0.002182] |
| Per-protein OLS | Normalized mean violation | 40 | -0.001165 [-0.001343, -0.001008] | -0.001136 [-0.001302, -0.000984] |
| Per-protein OLS | Normalized mean violation | 47 | -0.0006227 [-0.0007118, -0.0005413] | -0.0006094 [-0.000698, -0.0005307] |
| Per-protein OLS | Violation fraction | 0 | -0.03918 [-0.041, -0.0375] | -0.03893 [-0.04063, -0.03716] |
| Per-protein OLS | Violation fraction | 10 | -0.03006 [-0.0318, -0.02849] | -0.02977 [-0.03138, -0.02819] |
| Per-protein OLS | Violation fraction | 21 | -0.01743 [-0.01851, -0.01634] | -0.01726 [-0.01829, -0.01624] |
| Per-protein OLS | Violation fraction | 32 | -0.01169 [-0.01272, -0.01072] | -0.01154 [-0.01252, -0.01058] |
| Per-protein OLS | Violation fraction | 40 | -0.006128 [-0.006907, -0.005419] | -0.006026 [-0.006788, -0.005309] |
| Per-protein OLS | Violation fraction | 47 | -0.00344 [-0.004002, -0.002943] | -0.003378 [-0.003922, -0.002877] |
| Seq-gated OLS | Normalized mean violation | 0 | -0.01484 [-0.01586, -0.01406] | -0.0145 [-0.01508, -0.01393] |
| Seq-gated OLS | Normalized mean violation | 10 | -0.01316 [-0.01431, -0.01222] | -0.01281 [-0.01357, -0.01208] |
| Seq-gated OLS | Normalized mean violation | 21 | -0.00891 [-0.009917, -0.008084] | -0.008572 [-0.009221, -0.00793] |
| Seq-gated OLS | Normalized mean violation | 32 | -0.00708 [-0.008027, -0.006349] | -0.006761 [-0.007319, -0.006215] |
| Seq-gated OLS | Normalized mean violation | 40 | -0.004355 [-0.005203, -0.003759] | -0.004049 [-0.004454, -0.003646] |
| Seq-gated OLS | Normalized mean violation | 47 | -0.002303 [-0.003, -0.001859] | -0.002027 [-0.002259, -0.00181] |
| Seq-gated OLS | Violation fraction | 0 | -0.05336 [-0.05638, -0.05101] | -0.0523 [-0.05399, -0.05053] |
| Seq-gated OLS | Violation fraction | 10 | -0.04972 [-0.05302, -0.04695] | -0.04863 [-0.05067, -0.04655] |
| Seq-gated OLS | Violation fraction | 21 | -0.03933 [-0.04245, -0.03664] | -0.03832 [-0.0405, -0.03607] |
| Seq-gated OLS | Violation fraction | 32 | -0.0308 [-0.0337, -0.02836] | -0.02981 [-0.03181, -0.02774] |
| Seq-gated OLS | Violation fraction | 40 | -0.02151 [-0.0244, -0.01935] | -0.02055 [-0.02225, -0.01889] |
| Seq-gated OLS | Violation fraction | 47 | -0.01314 [-0.01544, -0.01143] | -0.01232 [-0.0135, -0.01116] |
| XGBoost | Normalized mean violation | 0 | -0.01424 [-0.01535, -0.01334] | -0.01389 [-0.0146, -0.01321] |
| XGBoost | Normalized mean violation | 10 | -0.01206 [-0.01323, -0.01103] | -0.0117 [-0.01255, -0.0109] |
| XGBoost | Normalized mean violation | 21 | -0.006886 [-0.007998, -0.005945] | -0.006568 [-0.007422, -0.005783] |
| XGBoost | Normalized mean violation | 32 | -0.00564 [-0.006626, -0.004842] | -0.005348 [-0.006059, -0.004665] |
| XGBoost | Normalized mean violation | 40 | -0.002575 [-0.003331, -0.002026] | -0.002301 [-0.002718, -0.001918] |
| XGBoost | Normalized mean violation | 47 | -0.001008 [-0.001642, -0.0006407] | -0.000744 [-0.0009048, -0.0006054] |
| XGBoost | Violation fraction | 0 | -0.05133 [-0.0545, -0.04867] | -0.05026 [-0.0522, -0.04834] |
| XGBoost | Violation fraction | 10 | -0.04643 [-0.04988, -0.04339] | -0.04535 [-0.04771, -0.04304] |
| XGBoost | Violation fraction | 21 | -0.02939 [-0.03253, -0.02669] | -0.02846 [-0.03071, -0.02626] |
| XGBoost | Violation fraction | 32 | -0.02286 [-0.0258, -0.02038] | -0.02195 [-0.024, -0.01991] |
| XGBoost | Violation fraction | 40 | -0.01232 [-0.0148, -0.01043] | -0.01145 [-0.01289, -0.01001] |
| XGBoost | Violation fraction | 47 | -0.005066 [-0.007162, -0.003703] | -0.004249 [-0.005048, -0.003496] |

## Null A fallback (proteins per layer using cohort bin statistics)

| probe | bin_label | mean | max |
|---|---|---|---|
| shared_ols_base256 | [0,4) | 0 | 0 |
| shared_ols_base256 | [4,6) | 0 | 0 |
| shared_ols_base256 | [6,8) | 0 | 0 |
| shared_ols_base256 | [8,10) | 0 | 0 |
| shared_ols_base256 | [10,14) | 0 | 0 |
| shared_ols_base256 | [14,20) | 0 | 0 |
| shared_ols_base256 | [20,30) | 2 | 2 |
| shared_ols_base256 | [30,40) | 22 | 22 |
| shared_ols_base256 | [40,55) | 58 | 58 |
| shared_ols_base256 | [55,inf) | 58 | 58 |
| per_protein_ols_base256 | [0,4) | 0 | 0 |
| per_protein_ols_base256 | [4,6) | 0 | 0 |
| per_protein_ols_base256 | [6,8) | 0 | 0 |
| per_protein_ols_base256 | [8,10) | 0 | 0 |
| per_protein_ols_base256 | [10,14) | 0 | 0 |
| per_protein_ols_base256 | [14,20) | 0 | 0 |
| per_protein_ols_base256 | [20,30) | 2 | 2 |
| per_protein_ols_base256 | [30,40) | 22 | 22 |
| per_protein_ols_base256 | [40,55) | 58 | 58 |
| per_protein_ols_base256 | [55,inf) | 58 | 58 |
| sequence_gated_ols_base256 | [0,4) | 0 | 0 |
| sequence_gated_ols_base256 | [4,6) | 0 | 0 |
| sequence_gated_ols_base256 | [6,8) | 0 | 0 |
| sequence_gated_ols_base256 | [8,10) | 0 | 0 |
| sequence_gated_ols_base256 | [10,14) | 0 | 0 |
| sequence_gated_ols_base256 | [14,20) | 0 | 0 |
| sequence_gated_ols_base256 | [20,30) | 2 | 2 |
| sequence_gated_ols_base256 | [30,40) | 22 | 22 |
| sequence_gated_ols_base256 | [40,55) | 58 | 58 |
| sequence_gated_ols_base256 | [55,inf) | 58 | 58 |
| shared_xgboost_base256_fixed_s4 | [0,4) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [4,6) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [6,8) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [8,10) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [10,14) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [14,20) | 0 | 0 |
| shared_xgboost_base256_fixed_s4 | [20,30) | 2 | 2 |
| shared_xgboost_base256_fixed_s4 | [30,40) | 22 | 22 |
| shared_xgboost_base256_fixed_s4 | [40,55) | 58 | 58 |
| shared_xgboost_base256_fixed_s4 | [55,inf) | 58 | 58 |

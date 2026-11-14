# SMACOF error decomposition: results

552 proteins, 4 probes, layers 0–47, OOF all pairs. Each value is the square root of a protein-equal mean of a squared term, in Å, with a 95% paired-protein bootstrap interval (2000 replicates, seed 20260902). Control values average the 3 draws within each protein first. The cross share is mean cross / mean total², bootstrapped as a ratio.

SMACOF: metric, 3D, raw unweighted stress in Å, one start from classical MDS, relative-decrease tolerance 1e-10, at most 200000 iterations.

## Shared linear

| Layer | Total | Inconsistent | Consistent | Cross share | Inconsistent, Gaussian control | Inconsistent, shuffled control | Consistent, Gaussian control | Consistent, shuffled control |
|---|---|---|---|---|---|---|---|---|
| 0 | 7.82 | 4.32 [4.22, 4.42] | 7.28 [6.57, 8.27] | -0.173 [-0.205, -0.139] | 5.70 [5.51, 5.95] | 5.70 [5.51, 5.94] | 5.49 [4.82, 6.34] | 5.48 [4.82, 6.33] |
| 10 | 7.13 | 3.83 [3.74, 3.92] | 6.51 [5.69, 7.66] | -0.122 [-0.148, -0.094] | 5.12 [4.92, 5.38] | 5.12 [4.93, 5.37] | 5.02 [4.29, 5.98] | 5.02 [4.29, 5.97] |
| 21 | 4.73 | 2.24 [2.21, 2.28] | 4.29 [3.51, 5.39] | -0.049 [-0.069, -0.034] | 3.61 [3.37, 3.96] | 3.60 [3.37, 3.93] | 2.99 [2.24, 3.97] | 2.99 [2.25, 3.96] |
| 32 | 4.67 | 2.32 [2.29, 2.36] | 4.17 [3.36, 5.31] | -0.045 [-0.064, -0.032] | 3.49 [3.27, 3.83] | 3.49 [3.27, 3.80] | 3.03 [2.26, 4.06] | 3.03 [2.26, 4.05] |
| 40 | 3.85 | 2.05 [2.01, 2.08] | 3.33 [2.54, 4.42] | -0.031 [-0.045, -0.021] | 2.95 [2.72, 3.32] | 2.95 [2.72, 3.30] | 2.42 [1.68, 3.39] | 2.42 [1.68, 3.38] |
| 47 | 2.93 | 1.76 [1.74, 1.80] | 2.39 [1.73, 3.27] | -0.025 [-0.037, -0.018] | 2.44 [2.20, 2.82] | 2.44 [2.20, 2.81] | 1.59 [1.01, 2.32] | 1.59 [1.01, 2.31] |

- Inconsistent², observed minus the Gaussian-matched noise control: below 0 at L0–L47; above 0 at none (95% interval of the paired difference excludes 0).
- Inconsistent², observed minus the shuffled errors control: below 0 at L0–L47; above 0 at none (95% interval of the paired difference excludes 0).
- Consistent², observed minus the Gaussian-matched noise control: below 0 at none; above 0 at L0–L47 (95% interval of the paired difference excludes 0).
- Consistent², observed minus the shuffled errors control: below 0 at none; above 0 at L0–L47 (95% interval of the paired difference excludes 0).

## Per-protein linear

| Layer | Total | Inconsistent | Consistent | Cross share | Inconsistent, Gaussian control | Inconsistent, shuffled control | Consistent, Gaussian control | Consistent, shuffled control |
|---|---|---|---|---|---|---|---|---|
| 0 | 5.65 | 4.05 [3.94, 4.16] | 4.66 [4.43, 4.89] | -0.196 [-0.215, -0.177] | 5.08 [4.88, 5.29] | 5.07 [4.88, 5.29] | 2.62 [2.51, 2.73] | 2.63 [2.52, 2.74] |
| 10 | 4.79 | 3.56 [3.43, 3.69] | 3.64 [3.38, 3.93] | -0.128 [-0.144, -0.112] | 4.34 [4.11, 4.62] | 4.35 [4.11, 4.63] | 2.13 [2.01, 2.26] | 2.13 [2.00, 2.26] |
| 21 | 2.50 | 1.79 [1.69, 1.91] | 1.86 [1.68, 2.06] | -0.071 [-0.091, -0.051] | 2.41 [2.22, 2.62] | 2.41 [2.23, 2.62] | 0.68 [0.63, 0.74] | 0.69 [0.63, 0.74] |
| 32 | 2.46 | 1.84 [1.73, 1.95] | 1.75 [1.59, 1.93] | -0.060 [-0.080, -0.043] | 2.36 [2.19, 2.55] | 2.36 [2.19, 2.55] | 0.70 [0.65, 0.76] | 0.71 [0.65, 0.76] |
| 40 | 1.82 | 1.43 [1.35, 1.52] | 1.16 [1.06, 1.26] | -0.028 [-0.035, -0.023] | 1.76 [1.64, 1.89] | 1.76 [1.64, 1.89] | 0.47 [0.43, 0.50] | 0.47 [0.44, 0.50] |
| 47 | 1.44 | 1.19 [1.12, 1.27] | 0.82 [0.75, 0.90] | -0.017 [-0.021, -0.013] | 1.40 [1.31, 1.50] | 1.40 [1.31, 1.50] | 0.32 [0.30, 0.34] | 0.32 [0.30, 0.34] |

- Inconsistent², observed minus the Gaussian-matched noise control: below 0 at L0–L47; above 0 at none (95% interval of the paired difference excludes 0).
- Inconsistent², observed minus the shuffled errors control: below 0 at L0–L47; above 0 at none (95% interval of the paired difference excludes 0).
- Consistent², observed minus the Gaussian-matched noise control: below 0 at none; above 0 at L0–L47 (95% interval of the paired difference excludes 0).
- Consistent², observed minus the shuffled errors control: below 0 at none; above 0 at L0–L47 (95% interval of the paired difference excludes 0).

## XGBoost

| Layer | Total | Inconsistent | Consistent | Cross share | Inconsistent, Gaussian control | Inconsistent, shuffled control | Consistent, Gaussian control | Consistent, shuffled control |
|---|---|---|---|---|---|---|---|---|
| 0 | 7.30 | 3.96 [3.86, 4.07] | 6.80 [6.09, 7.78] | -0.161 [-0.193, -0.128] | 5.42 [5.20, 5.68] | 5.42 [5.21, 5.68] | 4.98 [4.32, 5.85] | 4.98 [4.33, 5.86] |
| 10 | 6.29 | 3.24 [3.14, 3.34] | 5.78 [5.00, 6.92] | -0.111 [-0.140, -0.083] | 4.64 [4.40, 4.95] | 4.64 [4.40, 4.93] | 4.25 [3.53, 5.20] | 4.26 [3.54, 5.20] |
| 21 | 3.85 | 1.83 [1.77, 1.89] | 3.47 [2.76, 4.43] | -0.041 [-0.058, -0.028] | 3.06 [2.76, 3.45] | 3.05 [2.77, 3.43] | 2.29 [1.60, 3.14] | 2.29 [1.61, 3.13] |
| 32 | 3.72 | 1.84 [1.78, 1.90] | 3.31 [2.56, 4.32] | -0.038 [-0.056, -0.026] | 2.91 [2.64, 3.28] | 2.90 [2.64, 3.25] | 2.27 [1.53, 3.19] | 2.27 [1.53, 3.18] |
| 40 | 2.76 | 1.40 [1.35, 1.46] | 2.41 [1.66, 3.37] | -0.021 [-0.031, -0.015] | 2.17 [1.87, 2.57] | 2.16 [1.88, 2.56] | 1.68 [0.98, 2.50] | 1.68 [0.98, 2.49] |
| 47 | 2.09 | 1.15 [1.09, 1.20] | 1.77 [0.98, 2.69] | -0.017 [-0.027, -0.012] | 1.68 [1.34, 2.16] | 1.67 [1.35, 2.14] | 1.20 [0.54, 1.92] | 1.20 [0.54, 1.92] |

- Inconsistent², observed minus the Gaussian-matched noise control: below 0 at L0–L47; above 0 at none (95% interval of the paired difference excludes 0).
- Inconsistent², observed minus the shuffled errors control: below 0 at L0–L47; above 0 at none (95% interval of the paired difference excludes 0).
- Consistent², observed minus the Gaussian-matched noise control: below 0 at none; above 0 at L0–L47 (95% interval of the paired difference excludes 0).
- Consistent², observed minus the shuffled errors control: below 0 at none; above 0 at L0–L47 (95% interval of the paired difference excludes 0).

## Sequence-gated

| Layer | Total | Inconsistent | Consistent | Cross share | Inconsistent, Gaussian control | Inconsistent, shuffled control | Consistent, Gaussian control | Consistent, shuffled control |
|---|---|---|---|---|---|---|---|---|
| 0 | 7.55 | 4.02 [3.92, 4.11] | 7.12 [6.40, 8.11] | -0.172 [-0.206, -0.136] | 5.50 [5.30, 5.76] | 5.50 [5.30, 5.74] | 5.28 [4.60, 6.16] | 5.28 [4.60, 6.14] |
| 10 | 6.84 | 3.54 [3.44, 3.63] | 6.33 [5.52, 7.48] | -0.124 [-0.153, -0.094] | 4.91 [4.70, 5.18] | 4.90 [4.70, 5.16] | 4.81 [4.08, 5.78] | 4.80 [4.07, 5.76] |
| 21 | 4.50 | 2.04 [1.99, 2.08] | 4.13 [3.36, 5.20] | -0.049 [-0.070, -0.034] | 3.40 [3.15, 3.77] | 3.39 [3.15, 3.74] | 2.88 [2.14, 3.84] | 2.87 [2.14, 3.83] |
| 32 | 4.36 | 2.07 [2.03, 2.12] | 3.96 [3.17, 5.07] | -0.051 [-0.073, -0.034] | 3.25 [3.01, 3.60] | 3.24 [3.01, 3.58] | 2.85 [2.09, 3.84] | 2.85 [2.09, 3.83] |
| 40 | 3.53 | 1.76 [1.72, 1.80] | 3.12 [2.33, 4.19] | -0.029 [-0.044, -0.020] | 2.67 [2.42, 3.05] | 2.66 [2.41, 3.03] | 2.26 [1.51, 3.22] | 2.25 [1.51, 3.20] |
| 47 | 2.66 | 1.54 [1.50, 1.57] | 2.21 [1.56, 3.05] | -0.023 [-0.035, -0.016] | 2.20 [1.94, 2.60] | 2.20 [1.94, 2.58] | 1.47 [0.90, 2.16] | 1.47 [0.90, 2.15] |

- Inconsistent², observed minus the Gaussian-matched noise control: below 0 at L0–L47; above 0 at none (95% interval of the paired difference excludes 0).
- Inconsistent², observed minus the shuffled errors control: below 0 at L0–L47; above 0 at none (95% interval of the paired difference excludes 0).
- Consistent², observed minus the Gaussian-matched noise control: below 0 at none; above 0 at L0–L47 (95% interval of the paired difference excludes 0).
- Consistent², observed minus the shuffled errors control: below 0 at none; above 0 at L0–L47 (95% interval of the paired difference excludes 0).

## Sensitivity to 7stz_C

Cohort curves average squared terms, so one protein with very large errors has a lot of weight. `7stz_C` is that protein (it is also the outlier in the triangle analysis). The table gives each observed curve with all 552 proteins and without it, and its share of the cohort sum of consistent².

| Probe | Layer | Inconsistent, all | Inconsistent, without | Consistent, all | Consistent, without | Share of consistent² |
|---|---|---|---|---|---|---|
| Shared linear | 0 | 4.32 [4.22, 4.42] | 4.31 [4.21, 4.41] | 7.28 [6.57, 8.27] | 6.84 [6.49, 7.18] | 12% |
| Shared linear | 10 | 3.83 [3.74, 3.92] | 3.82 [3.73, 3.91] | 6.51 [5.69, 7.66] | 5.99 [5.61, 6.34] | 16% |
| Shared linear | 21 | 2.24 [2.21, 2.28] | 2.24 [2.20, 2.28] | 4.29 [3.51, 5.39] | 3.72 [3.43, 4.00] | 25% |
| Shared linear | 32 | 2.32 [2.29, 2.36] | 2.31 [2.28, 2.35] | 4.17 [3.36, 5.31] | 3.57 [3.29, 3.86] | 27% |
| Shared linear | 40 | 2.05 [2.01, 2.08] | 2.04 [2.01, 2.07] | 3.33 [2.54, 4.42] | 2.72 [2.48, 2.97] | 33% |
| Shared linear | 47 | 1.76 [1.74, 1.80] | 1.76 [1.73, 1.79] | 2.39 [1.73, 3.27] | 1.86 [1.69, 2.03] | 40% |
| Per-protein linear | 0 | 4.05 [3.94, 4.16] | 4.03 [3.92, 4.13] | 4.66 [4.43, 4.89] | 4.59 [4.39, 4.80] | 3% |
| Per-protein linear | 10 | 3.56 [3.43, 3.69] | 3.52 [3.40, 3.63] | 3.64 [3.38, 3.93] | 3.53 [3.33, 3.72] | 6% |
| Per-protein linear | 21 | 1.79 [1.69, 1.91] | 1.75 [1.67, 1.83] | 1.86 [1.68, 2.06] | 1.78 [1.66, 1.92] | 8% |
| Per-protein linear | 32 | 1.84 [1.73, 1.95] | 1.80 [1.72, 1.88] | 1.75 [1.59, 1.93] | 1.68 [1.56, 1.80] | 8% |
| Per-protein linear | 40 | 1.43 [1.35, 1.52] | 1.41 [1.34, 1.47] | 1.16 [1.06, 1.26] | 1.12 [1.05, 1.21] | 6% |
| Per-protein linear | 47 | 1.19 [1.12, 1.27] | 1.17 [1.12, 1.22] | 0.82 [0.75, 0.90] | 0.79 [0.74, 0.85] | 6% |
| XGBoost | 0 | 3.96 [3.86, 4.07] | 3.96 [3.85, 4.06] | 6.80 [6.09, 7.78] | 6.36 [6.01, 6.69] | 13% |
| XGBoost | 10 | 3.24 [3.14, 3.34] | 3.23 [3.13, 3.33] | 5.78 [5.00, 6.92] | 5.26 [4.91, 5.59] | 17% |
| XGBoost | 21 | 1.83 [1.77, 1.89] | 1.82 [1.76, 1.88] | 3.47 [2.76, 4.43] | 2.97 [2.68, 3.25] | 27% |
| XGBoost | 32 | 1.84 [1.78, 1.90] | 1.83 [1.77, 1.89] | 3.31 [2.56, 4.32] | 2.77 [2.49, 3.06] | 30% |
| XGBoost | 40 | 1.40 [1.35, 1.46] | 1.39 [1.34, 1.44] | 2.41 [1.66, 3.37] | 1.84 [1.61, 2.08] | 42% |
| XGBoost | 47 | 1.15 [1.09, 1.20] | 1.13 [1.08, 1.18] | 1.77 [0.98, 2.69] | 1.09 [0.94, 1.23] | 62% |
| Sequence-gated | 0 | 4.02 [3.92, 4.11] | 4.01 [3.91, 4.11] | 7.12 [6.40, 8.11] | 6.68 [6.32, 7.02] | 12% |
| Sequence-gated | 10 | 3.54 [3.44, 3.63] | 3.53 [3.44, 3.62] | 6.33 [5.52, 7.48] | 5.81 [5.44, 6.15] | 16% |
| Sequence-gated | 21 | 2.04 [1.99, 2.08] | 2.03 [1.99, 2.07] | 4.13 [3.36, 5.20] | 3.57 [3.27, 3.85] | 25% |
| Sequence-gated | 32 | 2.07 [2.03, 2.12] | 2.06 [2.02, 2.11] | 3.96 [3.17, 5.07] | 3.38 [3.09, 3.66] | 27% |
| Sequence-gated | 40 | 1.76 [1.72, 1.80] | 1.75 [1.71, 1.78] | 3.12 [2.33, 4.19] | 2.51 [2.26, 2.76] | 35% |
| Sequence-gated | 47 | 1.54 [1.50, 1.57] | 1.53 [1.50, 1.56] | 2.21 [1.56, 3.05] | 1.70 [1.53, 1.87] | 41% |

Figures without it are in `figures/sensitivity/`. Across probes and layers its share of the cohort consistent² ranges up to 64%.

## Unconverged SMACOF runs

None: all 741888 runs met the tolerance before the iteration cap.

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

## Mean per-protein RMS and shares

Here the square root is taken per protein first and the results are averaged equally over proteins, so one protein with very large errors has little weight. Shares are each protein's inconsistent², consistent² and cross term divided by its own total², then averaged; they add to exactly 100%.

| Probe | Layer | Inconsistent (Å) | Consistent (Å) | Inconsistent, shuffled control (Å) | Consistent, shuffled control (Å) | Inconsistent² share | Consistent² share | Cross share |
|---|---|---|---|---|---|---|---|---|
| Shared linear | 0 | 4.17 [4.09, 4.27] | 6.16 [5.86, 6.50] | 5.48 [5.35, 5.61] | 4.18 [3.90, 4.48] | 43% | 75% | -17% |
| Shared linear | 10 | 3.69 [3.61, 3.78] | 5.21 [4.90, 5.56] | 4.88 [4.75, 5.02] | 3.60 [3.33, 3.90] | 44% | 66% | -11% |
| Shared linear | 21 | 2.21 [2.17, 2.24] | 3.18 [2.96, 3.44] | 3.33 [3.22, 3.45] | 1.72 [1.54, 1.94] | 42% | 61% | -3% |
| Shared linear | 32 | 2.29 [2.26, 2.32] | 3.02 [2.80, 3.28] | 3.25 [3.15, 3.36] | 1.77 [1.58, 1.99] | 47% | 56% | -3% |
| Shared linear | 40 | 2.02 [1.99, 2.05] | 2.25 [2.07, 2.48] | 2.70 [2.61, 2.80] | 1.33 [1.19, 1.52] | 55% | 47% | -2% |
| Shared linear | 47 | 1.74 [1.72, 1.76] | 1.59 [1.46, 1.76] | 2.20 [2.12, 2.29] | 0.84 [0.74, 0.97] | 62% | 39% | -2% |
| Per-protein linear | 0 | 3.87 [3.77, 3.97] | 4.19 [4.03, 4.36] | 4.77 [4.63, 4.92] | 2.38 [2.29, 2.47] | 56% | 60% | -16% |
| Per-protein linear | 10 | 3.31 [3.20, 3.42] | 3.11 [2.95, 3.27] | 3.95 [3.80, 4.10] | 1.84 [1.75, 1.93] | 61% | 49% | -10% |
| Per-protein linear | 21 | 1.61 [1.54, 1.67] | 1.55 [1.47, 1.64] | 2.11 [2.02, 2.21] | 0.57 [0.54, 0.60] | 55% | 48% | -3% |
| Per-protein linear | 32 | 1.64 [1.58, 1.71] | 1.43 [1.35, 1.52] | 2.06 [1.97, 2.16] | 0.58 [0.55, 0.61] | 61% | 42% | -3% |
| Per-protein linear | 40 | 1.27 [1.22, 1.33] | 0.95 [0.89, 1.01] | 1.53 [1.46, 1.60] | 0.39 [0.37, 0.41] | 68% | 34% | -1% |
| Per-protein linear | 47 | 1.07 [1.03, 1.11] | 0.68 [0.64, 0.72] | 1.23 [1.18, 1.29] | 0.28 [0.26, 0.29] | 74% | 27% | -1% |
| XGBoost | 0 | 3.79 [3.69, 3.88] | 5.64 [5.34, 5.96] | 5.11 [4.96, 5.26] | 3.70 [3.42, 3.98] | 42% | 73% | -15% |
| XGBoost | 10 | 3.04 [2.95, 3.13] | 4.47 [4.19, 4.79] | 4.25 [4.09, 4.41] | 2.89 [2.65, 3.17] | 42% | 67% | -9% |
| XGBoost | 21 | 1.73 [1.68, 1.78] | 2.36 [2.16, 2.58] | 2.61 [2.48, 2.75] | 1.16 [1.01, 1.34] | 45% | 57% | -2% |
| XGBoost | 32 | 1.73 [1.69, 1.79] | 2.14 [1.95, 2.36] | 2.47 [2.35, 2.60] | 1.11 [0.96, 1.29] | 52% | 50% | -2% |
| XGBoost | 40 | 1.31 [1.27, 1.35] | 1.33 [1.18, 1.52] | 1.74 [1.64, 1.85] | 0.69 [0.59, 0.84] | 63% | 38% | -1% |
| XGBoost | 47 | 1.04 [1.01, 1.09] | 0.80 [0.69, 0.95] | 1.26 [1.18, 1.36] | 0.43 [0.35, 0.54] | 74% | 27% | -0% |
| Sequence-gated | 0 | 3.87 [3.78, 3.96] | 5.96 [5.65, 6.30] | 5.24 [5.10, 5.38] | 3.95 [3.67, 4.24] | 41% | 76% | -17% |
| Sequence-gated | 10 | 3.38 [3.29, 3.46] | 5.02 [4.72, 5.36] | 4.61 [4.48, 4.76] | 3.39 [3.13, 3.68] | 42% | 69% | -11% |
| Sequence-gated | 21 | 1.98 [1.95, 2.02] | 3.00 [2.78, 3.25] | 3.08 [2.97, 3.21] | 1.60 [1.42, 1.82] | 40% | 63% | -3% |
| Sequence-gated | 32 | 2.02 [1.99, 2.06] | 2.79 [2.57, 3.04] | 2.95 [2.85, 3.07] | 1.58 [1.40, 1.80] | 45% | 58% | -3% |
| Sequence-gated | 40 | 1.72 [1.69, 1.75] | 2.01 [1.83, 2.23] | 2.36 [2.26, 2.47] | 1.15 [1.01, 1.33] | 54% | 48% | -1% |
| Sequence-gated | 47 | 1.49 [1.46, 1.52] | 1.40 [1.27, 1.56] | 1.92 [1.83, 2.02] | 0.72 [0.63, 0.84] | 62% | 39% | -1% |

## Reconstruction quality: SMACOF against classical MDS, and the controls

Protein-equal means. TM-score was computed for observed matrices only; coordinate RMSD is available for every matrix. The classical-MDS TM-score is the canonical optimized value.

| Probe | Layer | TM-score, classical MDS | TM-score, SMACOF | Gain | SMACOF RMSD, observed (Å) | SMACOF RMSD, shuffled control (Å) |
|---|---|---|---|---|---|---|
| Shared linear | 0 | 0.462 | 0.540 | 0.078 [0.073, 0.084] | 8.75 [8.37, 9.17] | 4.83 [4.54, 5.16] |
| Shared linear | 10 | 0.529 | 0.616 | 0.087 [0.081, 0.092] | 7.27 [6.88, 7.68] | 4.21 [3.91, 4.54] |
| Shared linear | 21 | 0.707 | 0.770 | 0.063 [0.060, 0.067] | 4.08 [3.81, 4.37] | 1.93 [1.74, 2.15] |
| Shared linear | 32 | 0.736 | 0.793 | 0.057 [0.054, 0.061] | 3.84 [3.58, 4.14] | 1.93 [1.74, 2.14] |
| Shared linear | 40 | 0.810 | 0.860 | 0.050 [0.046, 0.054] | 2.79 [2.57, 3.04] | 1.51 [1.36, 1.70] |
| Shared linear | 47 | 0.850 | 0.903 | 0.053 [0.049, 0.057] | 2.03 [1.88, 2.22] | 1.01 [0.90, 1.15] |
| Per-protein linear | 0 | 0.508 | 0.615 | 0.107 [0.102, 0.112] | 6.82 [6.48, 7.16] | 2.94 [2.79, 3.09] |
| Per-protein linear | 10 | 0.596 | 0.719 | 0.124 [0.119, 0.129] | 5.18 [4.87, 5.50] | 2.37 [2.22, 2.52] |
| Per-protein linear | 21 | 0.819 | 0.878 | 0.059 [0.055, 0.063] | 2.48 [2.31, 2.68] | 0.78 [0.74, 0.83] |
| Per-protein linear | 32 | 0.838 | 0.894 | 0.056 [0.052, 0.060] | 2.28 [2.12, 2.46] | 0.80 [0.75, 0.85] |
| Per-protein linear | 40 | 0.902 | 0.944 | 0.042 [0.039, 0.046] | 1.45 [1.34, 1.56] | 0.56 [0.52, 0.59] |
| Per-protein linear | 47 | 0.934 | 0.964 | 0.030 [0.027, 0.033] | 1.05 [0.98, 1.13] | 0.41 [0.39, 0.43] |
| XGBoost | 0 | 0.500 | 0.594 | 0.094 [0.089, 0.099] | 8.14 [7.75, 8.56] | 4.25 [3.96, 4.55] |
| XGBoost | 10 | 0.582 | 0.688 | 0.106 [0.101, 0.111] | 6.42 [6.03, 6.83] | 3.35 [3.07, 3.64] |
| XGBoost | 21 | 0.787 | 0.855 | 0.067 [0.063, 0.072] | 3.22 [2.96, 3.49] | 1.30 [1.14, 1.49] |
| XGBoost | 32 | 0.807 | 0.877 | 0.071 [0.066, 0.075] | 2.95 [2.70, 3.22] | 1.25 [1.10, 1.44] |
| XGBoost | 40 | 0.888 | 0.939 | 0.052 [0.047, 0.056] | 1.79 [1.60, 2.01] | 0.81 [0.69, 0.97] |
| XGBoost | 47 | 0.930 | 0.967 | 0.037 [0.032, 0.042] | 1.07 [0.93, 1.24] | 0.52 [0.44, 0.65] |
| Sequence-gated | 0 | 0.471 | 0.558 | 0.087 [0.082, 0.093] | 8.57 [8.16, 8.99] | 4.52 [4.23, 4.84] |
| Sequence-gated | 10 | 0.534 | 0.633 | 0.100 [0.095, 0.105] | 7.12 [6.72, 7.54] | 3.93 [3.64, 4.25] |
| Sequence-gated | 21 | 0.727 | 0.795 | 0.068 [0.065, 0.072] | 3.89 [3.62, 4.19] | 1.77 [1.58, 1.98] |
| Sequence-gated | 32 | 0.750 | 0.821 | 0.071 [0.067, 0.075] | 3.64 [3.37, 3.94] | 1.73 [1.54, 1.94] |
| Sequence-gated | 40 | 0.830 | 0.888 | 0.058 [0.055, 0.062] | 2.53 [2.32, 2.78] | 1.29 [1.14, 1.47] |
| Sequence-gated | 47 | 0.870 | 0.925 | 0.055 [0.052, 0.060] | 1.81 [1.65, 1.99] | 0.85 [0.75, 0.99] |

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

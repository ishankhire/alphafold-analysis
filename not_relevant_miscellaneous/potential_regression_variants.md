# Potential Regression Variants (Single-Model Alternatives to Size Bins)

## Goal
Build a **single generalized model** that approaches the strong performance of size-bucketed regressions, while preserving interpretability and adding physically sensible constraints.

## Current signal from v2 diagnostics
From `delta_files/diagnostic_residue_weighted_csv_files/diagnostic_residue_weighted_v2_variant_summary.csv`:

- `bucketed_base_abs_fixed_bins` is currently strongest:
  - test mean per-protein R^2: **0.8737**
- Global single-model variants are much lower:
  - `baseline_protein_equal_abs`: **0.0520**
  - `interaction_log_n_abs` (best interaction): **0.0474**
  - `baseline_protein_equal_abs_plus_n_feature`: **0.0580**

Interpretation: one global linear map + simple size interaction is too restrictive; mapping appears to change by protein size regime.

---

## Variants already tried (for reference)
- `baseline_pair_weighted_abs`
- `baseline_protein_equal_abs`
- `baseline_protein_equal_norm_max`
- `interaction_n_abs`
- `interaction_sqrt_n_abs`
- `interaction_log_n_abs`
- `interaction_n_sq_abs`
- `baseline_pair_weighted_abs_lt_500`
- `baseline_protein_equal_abs_plus_n_feature`
- `bucketed_base_abs_fixed_bins`

---

## New variants to try

## A. User-proposed variants

### 1) Linear Tree (piecewise linear model) on protein size
- **Source:** User
- **Core idea:** Automatically learn size boundaries (tree), fit linear model per leaf.
- **Recommended setup:**
  - Split features for tree: `log(n_residues)` and optionally coarse `seq_sep` stats.
  - Leaf model: ridge linear regression on pair features.
  - Enforce minimum leaf proteins (avoid tiny leaves).
- **Why it may work:** approximates the strong bucket behavior without hand-picked bins.
- **Risks:** boundary instability, discontinuities at split points, overfitting in small leaves.

### 2) XGBoost regressor
- **Source:** User
- **Core idea:** non-linear trees capture interactions and regime changes automatically.
- **Recommended setup:**
  - Features: base pair representation + `log(n_residues)` + sequence-separation features.
  - Keep protein-level split and protein-equal evaluation.
- **Why it may work:** handles nonlinear and threshold behavior better than global linear.
- **Risks:** less interpretable, harder to enforce strict physics rules directly.

---

## B. Assistant-proposed variants

### 3) Varying-coefficient ridge with splines on `log(n_residues)`
- **Core idea:** one model, but coefficients change smoothly with size.
- **Form:**  
  `y_hat = x^T beta0 + sum_k phi_k(log n) * x^T beta_k`
- **Implementation:**
  - Build 3-5 spline basis terms `phi_k`.
  - Interact each with base pair features.
  - Fit with `RidgeCV`.
- **Why it may work:** smooth single-model approximation to piecewise bucket effects.
- **Risks:** more features, requires regularization tuning.

### 4) Piecewise-hinge varying-coefficient ridge (no spline lib)
- **Core idea:** use hinge bases over size (`max(0, log n - c)`).
- **Why:** easier to implement/debug than splines; still single model with flexible size adaptation.

### 5) GAM-style model (reduced pair representation)
- **Core idea:** PCA on pair features + smooth terms on `log(n)` and `log(seq_sep)`.
- **Why:** interpretability of smooth trends.
- **Risks:** dimensionality reduction may discard useful pair signal.

### 6) Two-stage residualization with size calibration
- **Core idea:** train baseline model, then fit residual correction as function of size and seq-sep.
- **Why:** directly targets systematic overprediction for small proteins.

---

## C. Constraint-aware prediction strategies

### 7) Physics prior + residual model (recommended)
- **Core idea:**  
  `distance = prior(seq_sep) + correction_model(...)`
- **Suggested prior examples:**
  - if `seq_sep == 1`: fixed 3.8 Å
  - if `seq_sep == 2`: learned constant or narrow range prior
- **Post-process:** `pred = clip(pred, lower_bound, upper_bound_by_sep)`
- **Why:** enforces hard physical logic outside the learned correction model.

### 8) Bounded output parameterization
- **Core idea:** predict unconstrained `z`, map to range:
  - `pred = L_s + (U_s - L_s) * sigmoid(z)` where bounds depend on `seq_sep` bucket.
- **Why:** hard bounds by construction.

### 9) Separate heads by sequence-separation class
- **Core idea:** train separate models for:
  - `seq_sep = 1`
  - `seq_sep = 2`
  - `seq_sep = 3-4`
  - `seq_sep >= 5`
- **Why:** lets short-range geometry be handled differently and more strictly.

---

## Sequence-separation indicators (clarification)
These are extra regression features added per residue pair, where `s = |i-j|`:
- `I[s=1]`
- `I[s=2]`
- `I[s=3]`
- `I[s<=4]`
- optional one-hot buckets for larger `s`

They help the model learn short-range-specific behavior, but **alone do not enforce hard bounds**.  
Use them together with clipping/rule-based constraints if hard cutoffs are required.

---

## Prioritized run order (recommended)
1. **Linear Tree + ridge leaves** (closest to current bucket success, still interpretable).
2. **Spline/hinge varying-coefficient ridge** (single smooth global model).
3. **XGBoost** (strong nonlinear baseline).
4. Add **physics prior + residual + hard clipping** to the best of (1)-(3).

---

## Evaluation protocol (keep fixed across variants)
- Protein-level split (same seed and split logic as v2).
- Report:
  - test mean per-protein R^2
  - test protein-weighted R^2
  - per-size-bucket mean per-protein R^2
  - negative R^2 fraction
  - mean(pred-true) by bucket
- Add constraint diagnostics:
  - `% predictions < 2.0 Å` before clipping
  - `% violations of seq-sep-specific bounds` before clipping

---

## Decision criteria for “better than current”
- Primary target: improve global single-model test mean per-protein R^2 well above current ~0.05 range.
- Secondary target: reduce small-protein overprediction bias.
- Maintain interpretability and stable behavior across buckets.

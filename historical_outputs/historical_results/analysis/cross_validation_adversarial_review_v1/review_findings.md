# Cross-Validation Adversarial Correctness Review

Review version: `cross_validation_adversarial_review_v1`.

## Summary

- Focused cross-validation suite: 68 tests; 4 expected failures representing classified findings; 0 unexpected failures/errors.
- Independent geometry-oracle comparisons: 17 rows; 0 discrepancies.
- Broader release baseline: passed (107 tests, 1 skipped).
- No production regression or analysis implementation was changed by this review.

## Findings

### Moderate — ADV-GEOM-VALIDATION-001: low-level complete-distance helpers lack defensive validation

`classical_mds` accepts asymmetric matrices and nonzero diagonals, while `compute_triangle_violations` accepts asymmetric and nonfinite matrices. The current protocol-2.0 workflow constructs symmetric zero-diagonal matrices and checks prediction coverage before calling these helpers, so the reviewed production path is protected. Direct or future reuse can nevertheless yield plausible invalid results.

Minimized tests:

- `test_geometry_independent_oracles.DefensiveDistanceMatrixValidationFindings.test_classical_mds_rejects_asymmetric_complete_distance_matrix`
- `test_geometry_independent_oracles.DefensiveDistanceMatrixValidationFindings.test_classical_mds_rejects_nonzero_diagonal`
- `test_geometry_independent_oracles.DefensiveDistanceMatrixValidationFindings.test_triangle_helper_rejects_asymmetric_matrix`
- `test_geometry_independent_oracles.DefensiveDistanceMatrixValidationFindings.test_triangle_helper_rejects_nonfinite_matrix`

No fix was made. Addressing this finding requires an explicit edit to `scripts/cluster_scripts/cluster_regression_helpers.py` and regression tests that cease being marked as expected failures.

### Major semantic decision — ADV-TM-SEMANTICS-001: the saved value is a Kabsch-aligned TM-style score

The implemented value applies the standard length-dependent TM kernel after an RMSD-optimal Kabsch alignment. It does not optimize the superposition under the TM objective, and no trusted TM-score/US-align executable was available in this local review. The result should not be claimed equivalent to canonical TM-score without that external comparison. The literal formula is independently confirmed by `test_tm_score_is_literal_post_kabsch_tm_style_formula`.

### Moderate semantic decision — ADV-EIGEN-TOLERANCE-001: negative-eigenvalue count has literal-sign semantics

Exact Euclidean examples can contain tiny negative eigenvalues from roundoff, and the literal count can change under algebraically equivalent centering or residue permutation. Negative eigenvalue mass remained negligible and is the robust diagnostic. Decide separately whether the count should retain literal-sign semantics or use a scale-aware tolerance; no output-contract or implementation change was made.

### Minor semantic decision — ADV-REFLECTION-001: reconstruction quality is reflection-invariant

The reviewed geometry intentionally compares both the original and reflected MDS solutions and retains the lower RMSD. This measures reflection-invariant reconstruction quality and does not establish recovered chirality. Whether to add `reflection_selected` remains an output-contract decision.

## Confirmed high-risk behavior

The passing adversarial cases independently confirmed held-out-target isolation, pair-fold responsibility, joint row-permutation invariance, directional feature order, conversion before float64 products, direct-concatenation pooled metrics, distinct pair/protein denominators, stable-only filtering, explicit incomplete coverage, exact 3D/4D/non-Euclidean geometry behavior, triangle enumeration and severity denominators, negative-prediction clipping, and cross-implementation triangle primitives.

## Unavailable or deferred checks

- shared_ols_base256: No complete protocol-2.0 family exists at /Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/regression_5fold_v1/shared_ols_base256
- per_protein_ols_base256: No complete protocol-2.0 family exists at /Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/regression_5fold_v1/per_protein_ols_base256
- sequence_gated_ols_base256: No complete protocol-2.0 family exists at /Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/regression_5fold_v1/sequence_gated_ols_base256
- shared_xgboost_base256_fixed_s4: No complete protocol-2.0 family exists at /Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/regression_5fold_v1/shared_xgboost_base256_fixed_s4
- Real XGBoost 700-round native JSON save/load parity remains a Delta or matched-environment check.
- No canonical-TM equivalence claim was made because no trusted executable comparison was run.

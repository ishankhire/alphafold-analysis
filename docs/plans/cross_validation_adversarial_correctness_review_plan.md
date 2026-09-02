# Cross-Validation Adversarial Correctness Review Plan

## 1. Purpose and scope

Review the protocol-2.0 implementations under
`scripts/cluster_scripts/cross_validation/` for failure modes that can produce
plausible, favorable, and internally consistent results while still being
wrong. The review is organized around dataflow boundaries, independent
numerical oracles, perturbation tests, and artifact replay rather than a
line-by-line reading of the implementation.

The review covers all four canonical families:

- shared directional-base256 OLS;
- within-protein pair-fivefold directional-base256 OLS;
- shared six-gate sequence-separation OLS; and
- shared fixed-S4 directional-base256 XGBoost.

The scientific protocol, output definitions, and existing implementation audit
remain authoritative in:

- `docs/methodology/regression_methodology.md`;
- `docs/methodology/regression_outputs.md`;
- `docs/methodology/regression_protocol_audit.md`; and
- `docs/methodology/regression_output_audit.md`.

This project reviews implementation correctness and the semantics of reported
geometry quantities. It does not redesign the cohort, introduce homology-grouped
folds, select new hyperparameters, or revise the paper's scientific claims.
Those assumptions should be reviewed separately. Lightweight distribution
checks are included only when they can expose silent dropping or a routing
error.

No production implementation is changed during the review phase. A finding
must first be reproduced, minimized, classified, and documented. Any fix is a
separate, explicitly scoped stage with a regression test that fails before the
fix and passes after it.

## 2. Existing coverage that will not be duplicated

Stage 7 of `fivefold_oof_regression_refactor_plan.md` already established:

- exhaustive `5 x 48 = 240` fold/layer and `48` layer-only task mappings;
- exact expected physical-shard inventories;
- rejection of missing, incomplete, duplicate-identity, unexpected, or
  checksum-corrupted shards;
- checkpoint invariant and artifact hash validation;
- duplicate task-ID rejection in final manifests;
- launcher and flat-deployment checks; and
- a passing active local regression suite, recorded as 107 tests with one
  declared skip.

Earlier stages already test synthetic exact-once OOF coverage, train-only
scaling and baselines, protein-equal weighting, gate isolation and
recombination, XGBoost fit controls, replay, resume, and synthetic aggregation.

These tests will be rerun once as a baseline, but the review will not spend time
reimplementing their shard-corruption matrix or manually recounting task IDs.
New tests must add an independent oracle, an adversarial mutation, a public
end-to-end boundary, or real-artifact evidence.

## 3. Threat model and review questions

| Failure mode | Primary review question | New evidence |
|---|---|---|
| Correct functions connected incorrectly | Does each fit, prediction, metric, and geometry row use exactly the intended training or held-out observations? | Target poisoning, joint/independent permutation, responsibility replay |
| Intermediate dtype drift | Are casts performed before squaring, products, moments, and matrix multiplication? | High-dynamic-range float32/float64 oracle cases |
| Shape or broadcasting errors | Do wrong but broadcastable shapes fail instead of producing plausible values? | Adversarial `(n,1)`, transposed, shortened, and asymmetric inputs |
| Denominator or aggregation drift | Are R2, RMSE, triangle means, severity fractions, and geometry summaries reduced over the documented population? | Hand-computed scalar cases and direct sufficient-statistic reconstruction |
| Enumeration or boundary errors | Is every pair or triangle included exactly once, with no diagonal, duplicate permutation, or endpoint loss? | Literal `itertools.combinations` oracle and residue-permutation invariance |
| Preprocessing or selection leakage | Can held-out targets change a fitted scaler, baseline, coefficient artifact, or booster? | Held-out-target poisoning and artifact digest comparison |
| Silent dropping | Does every invalid or missing value cause an explicit failed/incomplete status or task failure? | Injected NaNs, missing pairs, missing gates, and failed geometry cases |
| Wrong artifact or stale cache | Can an artifact with the wrong responsibility or configuration be accepted? | One public replay spot check; rely otherwise on Stage 7 coverage |
| Tests duplicate implementation | Are expected values constructed without the helper under test? | Closed-form cases, literal loops, independent formulas, and differential implementations |
| Semantic drift between utilities | Do the canonical geometry helper and triangle-subgroup utility agree on shared primitives? | Cross-implementation differential tests |
| Symptom masking | Are clipping, exclusions, and broad exception handling visible in outputs? | Negative-prediction, exclusion-count, and failure-row assertions |

## 4. Independence rules for new tests

The new review tests must obey the following rules.

1. An expected value may not be calculated with the production function being
   tested or with a wrapper that calls it.
2. Triangle expectations use a literal scalar loop over
   `itertools.combinations(range(R), 3)`.
3. Exact-MDS expectations preferentially use coordinate constructions whose
   answers are known from geometry: exact 3D reconstruction, rigid transforms,
   reflection, scale changes, and deliberately higher-dimensional points.
4. Expected regression metrics use explicit residual sums, absolute-error
   sums, and target moments rather than the production accumulator.
5. Metamorphic tests state which quantities must be invariant and which must
   change before the test is written.
6. Randomized cases use checked-in deterministic seeds and are paired with
   small hand-constructed cases so a failure remains interpretable.
7. Tests target public workflow boundaries where practical. A private helper
   test is insufficient when the suspected failure can arise from ordering or
   wiring outside that helper.
8. Tests may inspect explicit failure rows, but they must never treat a raised
   error, omitted row, or `NaN` cleanup as successful completion.

## 5. Exact test locations

Add the following local test modules:

```text
scripts/cluster_scripts/cross_validation/tests/
  test_adversarial_cv_wiring.py
  test_geometry_independent_oracles.py
```

`test_adversarial_cv_wiring.py` owns dataflow, poisoning, dtype, shape,
aggregation, coverage, and silent-dropping cases. Family-specific tests may be
added to the existing family modules only when the test must enter through a
family runner or artifact format.

`test_geometry_independent_oracles.py` owns strict-lower-triangle assembly,
classical MDS, rigid alignment, reflection, TM-score semantics, eigenvalue and
stress diagnostics, exact triangle enumeration, tolerance boundaries, severity
bands, clipping, and cross-implementation triangle comparisons.

Use the repository `./.venv` and the existing `unittest` style. Do not add a
new test dependency. The focused cross-validation command is:

```bash
./.venv/bin/python -m unittest discover -s scripts/cluster_scripts/cross_validation/tests -p 'test_*.py'
```

Individual modules may run while developing. Acceptance requires this complete
cross-validation suite plus the broader active regression-suite command used
for the release baseline; record both exact commands and their discovered test
counts rather than relying on the historical count in prose.

## 6. Stage A — Dataflow and leakage perturbations

### A1. Held-out-target poisoning

For each shared family, construct a deterministic synthetic outer fold with a
strong known feature-to-target relationship. Fit the model, then replace only
the held-out proteins' targets with visibly different finite values.

Require:

- identical fitted scaler and baseline digests;
- identical OLS coefficient/intercept artifacts or identical mocked/real
  booster bytes when determinism permits byte comparison;
- identical training metrics;
- changed held-out targets and evaluation metrics; and
- unchanged held-out protein responsibility.

For per-protein OLS, apply the mutation to the held-out rows of one specified
pair-fold fit only. The affected fold's fitted artifact must remain unchanged.
Do not mutate a pair and then compare the complete five-model family, because
that pair correctly belongs to the training set of the other four models.

### A2. Training-target poisoning

Change one or more fitting targets while leaving held-out targets unchanged.
Require the corresponding scaler to remain unchanged, the training baseline
and fitted model artifact to change, and the held-out evaluation population to
remain identical.

### A3. Pair-order metamorphisms

Create three versions of the same deterministic case:

1. canonical pair ordering;
2. one shared permutation applied to features, targets, pair identities, and
   pair-fold labels; and
3. the same feature permutation applied without permuting targets.

Require identical fits and metrics between cases 1 and 2 after undoing the row
permutation. Require case 3 to produce different predictions and materially
worse held-out metrics on the constructed exact-signal dataset.

### A4. Directional feature order

Populate `Z_ij` and `Z_ji` with distinguishable deterministic channel values.
Verify the exact `[Z_ij | Z_ji]` order through extraction, fitting, artifact
replay, and held-out prediction. A deliberately swapped block must be detected
by the independent expected prediction.

### A5. Dtype-before-operation cases

Use float32 inputs with magnitudes selected so float32 squaring or dot products
produce a measurably different result from conversion-before-operation.
Independently calculate float64 feature moments, target moments, squared
distances, OOF sufficient statistics, and MDS inputs.

Require agreement with the convert-first float64 oracle. Merely asserting that
the final array has dtype float64 is insufficient.

### A6. Shape and broadcasting rejection

Require explicit rejection of:

- target and prediction arrays shaped `(n, 1)`;
- a transposed feature matrix;
- feature width 255 or 257;
- one missing or one additional prediction;
- mismatched pair-index lengths;
- coordinates not shaped `(R, 3)`;
- an asymmetric or nonzero-diagonal matrix at any public API that accepts a
  complete distance matrix; and
- nonfinite features, targets, coordinates, predictions, or artifact values.

Where a low-level helper intentionally accepts a narrower contract, test the
public workflow wrapper that is responsible for validating it.

### A7. Silent dropping and incomplete coverage

Inject one missing prediction, one nonfinite prediction, one unsupported gate,
and one synthetic geometry exception. The acceptable outcomes are:

- an explicit task failure that prevents a complete checkpoint; or
- a retained row with an explicit incomplete/failed status and exclusion
  reason where the output contract calls for such a row.

Never accept a complete family with fewer than 552 protein rows per layer, a
geometry summary that hides its attempted/included/excluded counts, or a
primary metric calculated on a silently reduced pair set.

## 7. Stage B — Independent regression and aggregation oracles

### B1. Direct OOF metrics

For small deterministic proteins, concatenate literal held-out target and
prediction arrays and calculate:

- R2 from direct SSE and target SST;
- MAE from the absolute-error sum;
- RMSE from the total squared-error sum and total row count; and
- bias from the signed-error sum and total row count.

Compare these values with the merged accumulator, serialized/restored
accumulator, fold rows, and consolidated pooled row. Explicitly demonstrate
that averaging fold R2 or fold RMSE gives a different answer for the chosen
case.

### B2. Protein-level versus pair-level summaries

Use proteins with deliberately unequal pair counts and errors. Verify that:

- `pooled_metrics.csv` semantics are pair-pooled;
- per-protein means give every retained protein one unit of weight; and
- geometry summaries give every complete protein row one unit of weight rather
  than weighting by pairs or triangles.

This is a correctness test of denominators, not a broad distribution analysis.

### B3. Stable-fit scope

Construct stable and flagged OLS fits with finite predictions. Require the
primary output to retain both according to the current policy, while
`stable_fit_metrics.csv` contains only proteins whose required contributing
fits are all stable. Verify all support counts directly.

### B4. Failure-row summaries

Mix complete, incomplete-coverage, and geometry-failed protein rows. Verify
that `geometry_summary.csv` includes only complete values in metric summaries
while preserving exact attempted, included, and excluded protein counts.

## 8. Stage C — Independent MDS and alignment tests

### C1. Exact three-dimensional Euclidean reconstruction

Generate deterministic nondegenerate 3D coordinates and calculate their full
distance matrix with a literal coordinate norm. Require, within scale-aware
tolerances:

- aligned RMSD approximately zero;
- TM-score approximately one;
- stress approximately zero;
- top-three positive eigenvalue fraction approximately one;
- negative eigenvalue mass approximately zero; and
- zero triangle violations.

Run the case after casting the source distances to float32 to confirm that the
public geometry boundary converts before squaring and double centering.

### C2. Known rigid transform

Apply a known proper rotation and translation to coordinates. Test
`superimpose_structures` against the known inverse relationship and direct
coordinate residuals. This test must not construct its expected transformation
by calling another repository Kabsch helper.

### C3. Reflection behavior

Use a chiral, non-coplanar coordinate set and its mirror image. Verify that a
proper rotation alone cannot superimpose the mirror exactly, while the
documented reflected-MDS branch can. Confirm that the reported value is
reflection-invariant reconstruction quality and does not imply recovery of
chirality.

Record as a review decision whether the final artifacts need an explicit
`reflection_selected` field. Do not add it without a separate output-contract
decision.

### C4. Higher-dimensional Euclidean distances

Generate points in four dimensions and use their exact distances. Require:

- zero triangle violations;
- negligible negative eigenvalue mass;
- a top-three positive eigenvalue fraction below one; and
- nonzero three-dimensional stress.

This distinguishes metric validity, Euclidean embeddability, and 3D
embeddability.

### C5. Triangle-valid but non-Euclidean distances

Use a hand-validated metric distance matrix whose double-centered Gram matrix
is not positive semidefinite. Require zero triangle violations, detectable
negative eigenvalue mass, and nonzero MDS stress.

### C6. Residue-permutation invariance

Apply the same residue permutation to coordinates and both axes of the
distance matrix. Require unchanged triangle metrics, spectral fractions,
stress, aligned RMSD, and TM-score within numerical tolerance.

### C7. Scale behavior

Multiply coordinates and predicted distances by a positive constant. Require:

- RMSD and absolute triangle violation magnitudes to scale by that constant;
- total triangle-violation fraction, stress, and eigenvalue-mass fractions to
  remain unchanged for cases chosen away from the fixed violation tolerance;
- absolute severity-band membership to change exactly when scaled magnitudes
  cross the fixed 5- or 10-Angstrom boundaries; and
- exact reconstructions to retain TM-score one.

### C8. Degenerate and zero-scale behavior

Exercise two residues, three collinear residues, repeated points, an all-zero
distance matrix, and nearly zero distances. Require a documented successful or
explicit failed status with no divide-by-zero value entering a complete row.

### C9. Negative eigenvalue tolerance audit

The current helper counts every eigenvalue strictly below zero. Quantify the
negative eigenvalues produced by exact Euclidean cases across several scales
and residue counts. Compare raw counts with negative eigenvalue mass and with a
scale-aware tolerance diagnostic.

This stage produces a finding, not an automatic implementation change. Decide
separately whether `n_negative_eigenvalues` should retain literal-sign semantics
or adopt a documented tolerance.

### C10. TM-score semantic audit

Independently verify the documented `d0` formula and the score calculated from
the final aligned coordinate residuals. Determine whether the intended metric
is:

- a TM-style score after RMSD-optimal Kabsch alignment; or
- a canonical TM-score whose superposition is optimized under the TM
  objective.

When a trusted TM-score or US-align executable is available, compare a small
fixed set of coordinate pairs. Record the executable version, command, and
whether reflection was permitted. If no trusted executable is available, do
not make an equivalence claim beyond the literal implemented formula.

## 9. Stage D — Exact triangle-violation tests

### D1. Literal enumeration oracle

For residue counts 3 through 12, compare the production result with a literal
`itertools.combinations` implementation. Require exactly `math.comb(R, 3)`
unordered triples and no diagonal or repeated-index triple.

### D2. Hand-constructed triangle cases

Test individual triangles with:

- a valid 3-4-5 triangle;
- an equilateral triangle;
- a degenerate equality case;
- a known positive violation;
- a violation just below and just above `1e-12` Angstrom;
- violations just below 5, exactly 5, just below 10, and exactly 10 Angstrom;
  and
- one or more negative raw predicted sides passed through the public clipping
  path.

Verify violation count, unconditional mean, maximum, no-violation fraction,
and each severity-band count and fraction exactly.

### D3. Denominator identities

For mixed violating and nonviolating triples, require:

- `triangle_mean_violation = violation_sum / n_triangles`;
- severity fractions to use `n_triangles` as their denominator;
- the three severity fractions to sum to
  `triangle_violation_fraction`; and
- no-violation fraction to equal one minus violation fraction.

Also calculate the conditional mean over violated triangles in the test oracle
to demonstrate that it is intentionally different from the canonical
unconditional mean.

### D4. Randomized differential cases

Generate deterministic random symmetric nonnegative matrices for small residue
counts. Compare:

1. the literal brute-force oracle;
2. `cluster_regression_helpers.compute_triangle_violations`; and
3. the compatible all-triangle primitives from
   `triangle_violation_subgroup_common.py`.

Reconcile clipping and field-name differences explicitly. Any disagreement in
tolerance, severity boundaries, or denominator semantics is a review finding
even when each implementation is internally self-consistent.

### D5. Permutation and scale metamorphisms

Require triangle statistics to be invariant to residue relabeling. Under a
positive distance scale factor and cases chosen away from the fixed tolerance,
require total violation counts and fractions to remain fixed and positive
violation sums, means, and maxima to scale linearly. Verify absolute
severity-band membership against the scaled magnitudes rather than treating
those fixed-Angstrom bands as scale invariant.

### D6. Chunk-boundary differential test

The cross-validation exact helper enumerates triangles by leading residue,
while the triangle-subgroup utility uses batched enumeration. Run the subgroup
utility with several batch sizes, including boundaries that leave a one-item
final batch, and compare every all-triangle primitive with the literal oracle
and the unbatched cross-validation helper.

## 10. Stage E — Real artifact replay

This stage begins only after a protocol-2.0 production family has a complete
`run_manifest.json` and has been copied under
`delta_files/regression_5fold_v1/<family>/`. At present, only shared metadata
exists, so real-artifact replay is pending rather than silently replaced with
historical output.

Select layers `0`, `23`, and `47`. For each family and outer fold, select the
lexicographically first protein with complete artifacts. This produces at most
60 family-layer-fold-protein replay cases. For per-protein OLS, replay all five
pair-fold models for each selected protein and assemble its complete OOF
vector.

For every selected case, independently:

1. validate the run, fold, configuration, artifact, and scaler digests;
2. reconstruct canonical strict-lower-triangle directional features;
3. resolve the exact held-out model responsibility;
4. replay predictions without the production output-row builders;
5. calculate regression metrics from literal sums;
6. reconstruct the symmetric distance matrix;
7. calculate the independent geometry and triangle oracle metrics; and
8. compare with `per_protein.csv` and `geometry_metrics.csv`.

Fold-0 historical parity is retained as an additional anchor only where the
historical and protocol-2.0 fit, weighting, precision, and evaluation semantics
are identical. A documented protocol difference is not treated as a failure;
unexplained disagreement is.

Raw pair predictions are held only in memory and are not saved.

## 11. Stage F — Stability and actual XGBoost boundary

### F1. OLS stability

From each completed family, verify exact agreement among `model_status.csv`,
`training_metrics.csv`, replay artifacts, per-protein stability flags, and
`stable_fit_metrics.csv`. Report, by family and layer:

- requested and completed fits;
- stable, rank-deficient, underdetermined, and ill-conditioned counts;
- minimum and maximum design rank;
- condition-number range; and
- complete versus stable-only metric differences.

This is a status and sensitivity audit, not coefficient interpretation.

### F2. Real XGBoost save/load/replay

Because local protocol-2.0 tests use a mocked estimator, execute at least one
real fit in the Delta `~/protein_venv` environment or another explicitly
matched environment. Verify:

- the installed XGBoost version;
- exact 700-round completion;
- absence of early stopping and an evaluation set;
- native JSON save/load;
- prediction equality before and after reload;
- feature-importance key validation; and
- unchanged booster state under held-out-target poisoning.

This environment check is not replaced by a mocked unit test.

## 12. Review artifacts and exact output contract

Unit tests write only to temporary test directories. The review driver is a
local-only script:

```text
scripts/multi_protein/regression/review_cross_validation_adversarial.py
```

It reads completed copied-back protocol-2.0 outputs and writes:

```text
delta_files/analysis/cross_validation_adversarial_review_v1/
  review_manifest.json
  automated_test_results.json
  perturbation_checks.csv
  geometry_oracle_checks.csv
  artifact_replay_checks.csv
  stability_checks.csv
  review_findings.md
```

No pair-level or triangle-level prediction file is saved.

### `review_manifest.json`

One JSON object records:

- review version and timestamp;
- repository commit and dirty-worktree disclosure;
- Python, NumPy, SciPy, scikit-learn, Biopython, and XGBoost versions where
  applicable;
- cohort, fold, configuration, and reviewed run-manifest digests;
- test commands and overall status;
- selected real replay cases; and
- explicit unavailable or deferred checks.

### `automated_test_results.json`

One JSON record per test contains `test_id`, `test_module`, `status`, elapsed
time, and concise failure or skip reason. It is a review execution record, not
a replacement for test-runner output.

### `perturbation_checks.csv`

One row per family, layer or synthetic case, model responsibility, and
perturbation. Principal columns are:

- `test_id`, `model_family`, `layer`, `outer_fold`, `protein_id`, `pair_fold`,
  and `sequence_gate`;
- `perturbation`, `expected_invariant`, `artifact_digest_before`,
  `artifact_digest_after`;
- `metric_before`, `metric_after`, `observed_behavior`, `status`, and
  `details`.

### `geometry_oracle_checks.csv`

One row per oracle case and metric. Principal columns are:

- `case_id`, `case_type`, `n_residues`, `input_dtype`, `metric`;
- `expected_value`, `observed_value`, `absolute_error`, `relative_error`,
  `tolerance`, and `status`; and
- `oracle_method` and `details`.

### `artifact_replay_checks.csv`

One row per selected family-layer-fold-protein metric. Principal columns are:

- model responsibility and artifact path/hash fields;
- `metric_scope`, `metric`, `saved_value`, `recomputed_value`, `difference`,
  `tolerance`, and `status`; and
- fold, configuration, scaler, and pair-assignment digests.

### `stability_checks.csv`

One row per family, layer, and status scope. Principal columns are:

- requested/completed/stable/flagged fit counts;
- design-rank and condition-number extrema;
- complete and stable-only metric values and their difference; and
- status and details.

### `review_findings.md`

The concise human-readable report groups findings as:

- **critical:** a central reported value uses the wrong observations,
  responsibility, enumeration, or denominator;
- **major:** a result remains numerically reproducible but its named metric or
  stated semantics are materially inaccurate;
- **moderate:** an edge case, status, or numerical policy can mislead a subset
  of results;
- **minor:** clarity, diagnostics, or defensive validation improvements; and
- **confirmed:** high-risk behavior independently validated.

Every finding cites a minimized reproducer and its automated test ID. The
report distinguishes confirmed defects, semantic decisions, and unavailable
real-artifact checks.

## 13. Acceptance criteria

The review is complete only when:

- the complete active regression suite and the two new adversarial modules
  pass in `./.venv`;
- every new expected value is independent of the production helper under test;
- held-out-target poisoning cannot change fitted preprocessing or model state;
- joint row permutations preserve results and mismatched permutations are
  detected;
- wrong shapes, nonfinite inputs, missing pairs, and unsupported gates cannot
  enter complete outputs;
- pooled R2, MAE, RMSE, and bias match direct concatenation without fold-metric
  averaging;
- exact 3D Euclidean distances reconstruct with negligible stress and aligned
  error;
- reflection, higher-dimensional, non-Euclidean, scale, permutation, and
  degenerate MDS cases behave according to their predeclared invariants;
- literal triangle enumeration agrees on counts, tolerance, unconditional
  mean, maximum, and severity bands;
- the cross-validation and triangle-subgroup implementations agree on shared
  all-triangle primitives after explicitly matching clipping and names;
- geometry summaries preserve attempted, included, and excluded counts;
- real XGBoost native save/load/replay is verified in a matching environment;
- real protocol-2.0 artifact replay is completed when production artifacts
  exist, or is explicitly marked unavailable; and
- every discrepancy is classified before any production code is changed.

## 14. Implementation and change-control stages

1. Add the two independent adversarial test modules without modifying
   production behavior. Run focused and complete tests.
2. Implement the local review driver and exact review-artifact contract.
3. Run synthetic perturbation and geometry reviews; document findings.
4. Run real XGBoost and protocol-2.0 artifact replay when the required
   environment and outputs exist.
5. For each confirmed defect, approve a separate fix, add a failing regression
   test, make the smallest root-cause change, rerun the complete suite, and
   determine whether any stored result family requires regeneration.

After each significant implementation or confirmed-fix stage, update
`branch-analysis.md`, commit only the scoped files, and push.

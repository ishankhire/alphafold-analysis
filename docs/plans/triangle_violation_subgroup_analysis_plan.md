# Triangle-Violation Subgroup Analysis Plan

## 1. Goal and fixed scope

Replay four frozen distance probes across Evoformer layers `0..47` and measure
which kinds of final-PDB C-alpha triangles produce inconsistent decoded
distances:

1. shared global OLS (`general_base256_combined`);
2. per-protein OLS (`per_protein_base256_combined`);
3. shared XGBoost (`shared_xgboost_base256_combined`); and
4. shared sequence-separation-gated OLS (`analysis_1_sequence_gated`).

The primary questions are whether violation frequency and scale-normalized
severity follow different trajectories for RMS-small, RMS-medium, and RMS-large
triangles and for clearly acute, near-right, and clearly obtuse triangles.
Area is a secondary mixed size/shape analysis. Relative slack is a separate
explanatory analysis of geometric vulnerability.

This workflow replays saved models; it fits, tunes, and selects nothing. Use the
same 111 canonical locked-test proteins for every model. Shared OLS,
sequence-gated OLS, and XGBoost evaluate unseen proteins. Per-protein OLS fits
inside each test protein, so its complete-matrix result mixes fitting and
held-out edges and remains explicitly descriptive.

Reference labels come only from OpenFold's final predicted PDB and are frozen
per residue triple across all models and layers. Outcomes come from decoded
distances. The target is therefore the model's own final structure, not an
experimental structure. CAMEO and ECOD remain one dataset.

Cluster output:

```text
$HOME/csv_files/triangle_violation_subgroups/
```

Audited local copy:

```text
delta_files/triangle_violation_subgroups/
```

## 2. Artifact replay contract

Validate producer manifests, feature order, completed layers, model status,
checksums, and the canonical split digest before loading pair tensors.

- **Shared OLS:** read the canonical replay result, requiring one raw intercept
  and 256 ordered raw coefficients per layer. Preserve the disclosure that the
  historical coefficients were serialized to six decimal places.
- **Per-protein OLS:** resolve exactly one indexed coefficient-shard row per
  layer and test protein. Reconstruct
  `regression_protocol.within_protein_pair_split` and verify its counts, seed,
  and digest against `metadata/pair_splits.csv`.
- **XGBoost:** read only the final `booster_layer_00.json` through
  `booster_layer_47.json`; never use search or cap-sweep boosters.
- **Sequence-gated OLS:** require the six fitted canonical gates per layer,
  resolve indexed raw coefficients, and use the frozen sequence-separation
  routing without adding a feature or fallback.

For every unordered pair use the canonical strict-lower-triangle feature

\[
x_{ij}=[Z_{ij}\mid Z_{ji}],\qquad i>j,
\]

materialized as float64. Record negative predictions, fill a symmetric matrix,
and clip negative distances to zero only for triangle calculations, matching
Figure 5.

### 2.1 Future protocol-2.0 OOF replay interface

Stage 7 of the fivefold refactor documents the interface that a separately
approved triangle-subgroup refactor will consume. It does not change this
workflow's implemented historical-artifact inputs, scripts, 111-protein scope,
or saved results.

Protocol-2.0 families live under
`$CSV_ROOT/regression_5fold_v1/<family>/`. A future consumer must read
`run_manifest.json` and `model_artifacts/replay_index.csv`, resolve every
`artifact_path` relative to the family root, require complete status, and
verify artifact SHA-256 plus configuration and scaler digests before replay.
Responsibility keys are:

- `(layer, outer_fold)` for shared OLS and fixed shared XGBoost;
- `(layer, outer_fold, sequence_gate)` for sequence-gated OLS; and
- `(layer, protein_id, pair_fold)` for per-protein OLS.

Shared-family proteins select the model for their frozen outer fold. Sequence
gates use the canonical six-gate routing without fallback. Per-protein pair
folds are regenerated from the compact checked-in manifest, and the matching
identity is located inside the referenced compressed shard rather than inferred
from CSV row order. Raw pair predictions are absent, so the consumer replays
the canonical directional base256 features and assembles exactly one held-out
prediction per protein pair before computing triangles. Exact deployment and
consumer checks are in
`scripts/cluster_scripts/cross_validation/README.md`.

## 3. Frozen reference categories

For each unordered residue triple, calculate final-PDB side lengths and sort
them as \(a\le b\le c\).

### 3.1 RMS size: primary

\[
L_{\mathrm{RMS}}=\sqrt{(a^2+b^2+c^2)/3}.
\]

Calculate exact within-protein 1/3 and 2/3 quantiles and freeze:

- `small`: below the 1/3 quantile;
- `medium`: from the 1/3 quantile up to the 2/3 quantile;
- `large`: at or above the 2/3 quantile.

Use half-open intervals and record actual counts because ties can make the bins
slightly unequal.

### 3.2 Angle shape: primary

Use the largest final-PDB angle \(\theta_{\max}\):

- `clearly_acute`: \(\theta_{\max}<88^\circ\);
- `near_right`: \(88^\circ\le\theta_{\max}\le92^\circ\);
- `clearly_obtuse`: \(\theta_{\max}>92^\circ\).

Also derive an exact-90-degree acute/obtuse sensitivity using the sign of
\(c^2-a^2-b^2\) and a recorded relative numerical tolerance.

### 3.3 Area: sensitivity

\[
A=\tfrac12\lVert(C_j-C_i)\times(C_k-C_i)\rVert.
\]

Use exact within-protein area tertiles. Interpret area as a mixture of scale and
shape rather than as an independent size variable.

### 3.4 Relative slack: explanatory

\[
s_{\mathrm{rel}}=(a+b-c)/c.
\]

This lies in `[0,1]` for a valid triangle, up to tolerance. Low values identify
nearly degenerate triangles requiring little asymmetric edge error to violate
the inequality. Use exact within-protein low, medium, and high slack tertiles.
Analyze slack separately rather than crossing it with RMS and angle initially.

### 3.5 Required category systems

- `all_triangles`: one category;
- `rms_size`: three categories;
- `angle_shape`: three categories;
- `angle_shape_exact90`: acute, right within the recorded relative numerical
  tolerance, and obtuse sensitivity categories;
- `rms_by_angle`: nine categories;
- `relative_slack`: three categories;
- `area`: three categories.

Every system must partition its intended exact triangle population. Do not
sample final category metrics or save triangle-level rows.

## 4. Violation outcomes

For clipped predicted sides \(\hat a\le\hat b\le\hat c\), use the established
\(10^{-12}\) Angstrom tolerance and define

\[
\delta=\max(\hat c-\hat a-\hat b,0),\qquad
r=\delta/L_{\mathrm{RMS,true}}.
\]

For each model-layer-protein-scope-category row save these primitives:

- `n_triangles`, `n_violations`, absolute violation sum and maximum;
- absolute-band counts `0-5`, `5-10`, and `10+` Angstrom;
- relative violation sum and maximum;
- relative-band counts `(0,0.05)`, `[0.05,0.10)`, and `[0.10,infinity)`;
- conditional relative-violation p50, p90, and p99; and
- explicit zero-support and zero-violation statuses.

Derive:

- violation fraction;
- mean absolute violation over all triangles and conditional on violation;
- mean relative violation over all triangles (`normalized burden`) and
  conditional on violation;
- absolute and relative severity-band fractions over all category triangles;
- maximum and conditional p99 relative violation.

Validate that each all-triangle mean equals violation fraction times conditional
mean. Retain maximum for audit, but emphasize p99 because maximum is sensitive
to the number of available triangles.

## 5. Evaluation scopes

### 5.1 `all_edges`

Primary Figure-5-compatible exact result over all triangles in each test
protein. Per-protein OLS rows carry
`generalization_scope=mixed_within_protein_train_and_heldout`.

This is the workflow's only evaluation scope.

## 6. Protein-size dependence

Protein-equal aggregation stops a large protein's \(\binom{R}{3}\) triangles
from dominating the primary mean, but normalization does not remove category
composition, absolute-scale, or extreme-value differences associated with
protein length. Predeclare four checks.

1. **Composition:** by canonical five-bin protein length, report angle-category
   fractions; absolute RMS/area tertile boundaries; median and p90 RMS, area,
   and slack; and category support.
2. **Stratified trajectories:** calculate protein-equal outcomes separately in
   the five canonical length bins and retain the individual protein rows and
   protein counts. Do not bootstrap.
3. **Continuous association:** for every model-layer-scope-category outcome,
   regress the protein-level value on standardized `log(n_residues)` without
   triangle weights. Save the slope, intercept, Spearman correlation, and
   protein count. Do not bootstrap.
4. **Weighting/tail sensitivity:** compare protein-equal and triangle-pooled
   summaries to quantify naive long-protein domination. Compare maximum with
   conditional p99 and their length associations.

Within-protein tertiles balance category prevalence per protein but do not make
`large` represent the same absolute RMS scale in short and long proteins; the
composition and continuous analyses explicitly quantify this limitation.

## 7. Exact output contract

```text
triangle_violation_subgroups/
  run_manifest.json
  per_protein_triangle_metrics.csv
  triangle_metric_summary.csv
  length_bin_summary.csv
  protein_length_effects.csv
  paired_model_contrasts.csv
  runtime.csv
  metadata/
    artifact_provenance.json
    split_assignments.csv
    protein_reference_thresholds.csv
    preparation_failures.csv
  layer_shards/
    per_protein_triangle_metrics_layer_00.csv
    ...
    per_protein_triangle_metrics_layer_47.csv
  visualizations/
    rms_size_trajectories.pdf
    angle_shape_trajectories.pdf
    rms_by_angle_trajectories.pdf
    relative_slack_trajectories.pdf
    area_sensitivity_trajectories.pdf
    protein_length_sensitivity.pdf
```

`per_protein_triangle_metrics.csv` has one row per model, layer, test protein,
scope, category system, and category label. Principal columns:

```text
analysis_name,model_family,layer,protein_id,source,protein_split,
evaluation_scope,generalization_scope,category_system,category_label,
n_residues,protein_length_bin,n_triangles,n_violations,
n_negative_predictions,negative_prediction_fraction,
violation_sum_angstrom,violation_max_angstrom,
n_abs_0_5,n_abs_5_10,n_abs_10_plus,
relative_violation_sum,relative_violation_max,
n_rel_0_0p05,n_rel_0p05_0p10,n_rel_0p10_plus,
relative_violation_p50_conditional,relative_violation_p90_conditional,
relative_violation_p99_conditional,violation_fraction,
mean_violation_all,mean_violation_conditional,
mean_relative_violation_all,mean_relative_violation_conditional,support_status
```

`triangle_metric_summary.csv` has one row per model, layer, scope, category, and
derived metric, with protein-equal mean, standard deviation, median, quartiles,
finite and total protein counts, and triangle-pooled sensitivity value.
Resampling is not performed. Individual protein analyses remain in
`per_protein_triangle_metrics.csv`.

`protein_reference_thresholds.csv` has one row per protein with residue and
triangle counts, RMS/area/slack tertile boundaries, angle counts, reference
distribution summaries, length bin, and status. `length_bin_summary.csv` has
one row per model-layer-scope-category-metric-length-bin.
`protein_length_effects.csv` has one row per
model-layer-scope-category-metric. `paired_model_contrasts.csv` uses identical
protein-category rows for paired model differences.

## 8. Implementation and Delta execution

Add:

```text
scripts/cluster_scripts/triangle_violation_subgroups/
  prepare_triangle_violation_reference.py
  prepare_triangle_violation_reference.sbatch
  triangle_violation_subgroup_replay.py
  triangle_violation_subgroup_replay.sbatch
  aggregate_triangle_violation_subgroups.py
  aggregate_triangle_violation_subgroups.sbatch
  tests/test_triangle_violation_subgroups.py
scripts/multi_protein/
  analyze_triangle_violation_subgroups.py
```

Preparation performs chunked exact triangle enumeration once per protein and
writes only compact thresholds and summaries. Use node-local `/tmp` for
transient arrays. Replay is a 48-task CPU array with one layer per task. Load a
protein's pair tensor once, generate all four predictions, stream triangle
batches, and checkpoint after small protein batches. Atomically finalize a
layer only after all 111 proteins complete for `all_edges`.

Pilot from the comparable exact-triangle launcher request (`8` CPUs, `64G`,
`06:00:00`), then right-size memory and walltime from observed MaxRSS and elapsed
time. Pair tensors remain on `/work/hdd`; HOME receives only scripts, compact
checkpoints, and outputs. Recheck the live NCSA Delta
[running-jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html)
and [data-management](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html)
documentation before implementation.

Every sbatch file includes BEGIN/END/FAIL mail settings and resolves its
entrypoint with:

```bash
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/<entrypoint>.py"
```

Deploy and submit from `$HOME/openfold_interpretability_canonical`.

## 9. Acceptance criteria and reporting

- Complete 48 layers, four models, 111 test proteins, the `all_edges` scope, and all
  category systems with explicit failures and support.
- Verify artifact checksums, split digest, pair order, and model coverage.
- Require finite positive RMS, nonnegative area, and slack in `[0,1]` within
  tolerance; require exact category partitions and RMS-by-angle margins.
- Require both absolute and relative band counts to sum to `n_violations` and
  primitives to reproduce every derived metric.
- Reproduce each canonical model's unstratified `all_edges` Figure-5 metrics
  within declared tolerances.
- Aggregate primary results from protein rows, never triangles; retain zero
  cells; save no pair or triangle rows.
- Test synthetic acute, near-right, obtuse, degenerate, violating, negative,
  threshold-boundary, and partition cases.
- Log timestamps and elapsed time for preparation, each model/protein
  checkpoint, layer completion, and aggregation.

Report overall reproduction first; then RMS size, angle, RMS-by-angle,
frequency-versus-conditional-severity decomposition, normalized severity bands
and p99, slack, area, and protein-length sensitivity.

## 10. Pre-implementation decisions

No question currently blocks implementation. Recommended frozen defaults are
the 88--92 degree near-right band, relative severity bands 0--5%, 5--10%, and
10%+, RMS as primary size, area as sensitivity, and relative-slack tertiles as
a separate explanation. Change them only before production replay, record the
decision, and never tune them from model outcomes. Bootstrap resampling is not
part of this workflow; descriptive summaries and associations retain their
underlying per-protein analyses instead.

> Historical reference; not current execution instructions. See [current sources and status](../../project_status.md).

# Fivefold OOF Triangle-Violation Subgroup Refactor Plan

## 1. Decision and scientific scope

Refactor `scripts/cluster_scripts/triangle_violation_subgroups/` from the
historical protocol-1.0, 111-protein, mixed-edge replay to protocol 2.0 using
the frozen outputs under `regression_5fold_v1/`.

Replay these four final distance-regression families across Evoformer layers
`0..47`:

1. shared OLS (`shared_ols_base256`);
2. per-protein OLS (`per_protein_ols_base256`);
3. fixed shared XGBoost (`shared_xgboost_base256_fixed_s4`); and
4. shared sequence-separation-gated OLS (`sequence_gated_ols_base256`).

The primary analysis uses only out-of-fold (OOF) predictions. Shared families
evaluate proteins held out by their outer fold. Per-protein OLS evaluates every
pair using the model that withheld that pair's fold, then computes triangles
from the single completed cross-fitted distance matrix. It does not apply each
pair-fold model to all pairs, average predictions across models, or average five
mostly in-sample triangle analyses.

Add one per-protein-only diagnostic that partitions triangles by whether their
three OOF edge predictions came from `1_model`, `2_models`, or `3_models`.
This directly measures whether cross-fit stitching is associated with triangle
inconsistency. It is a separate category system and is not crossed with RMS,
angle, area, or slack in this refactor.

The main scientific questions remain whether violation frequency and
scale-normalized severity follow different layer trajectories for RMS-small,
RMS-medium, and RMS-large triangles and for clearly acute, near-right, and
clearly obtuse triangles. Area remains a mixed size/shape sensitivity, and
relative slack remains an explanatory analysis of geometric vulnerability.

This workflow replays saved models and fits, tunes, or selects nothing.
Reference labels come only from each final predicted PDB and remain fixed across
models and layers. Outcomes therefore measure consistency with OpenFold's own
final structure, not an experimental structure. Treat CAMEO and ECOD as one
analysis dataset; retain `source` only for input resolution and provenance.

## 2. Versioned inputs, cohort, and outputs

### 2.1 Input roots

On Delta, read:

```text
$CSV_ROOT/regression_5fold_v1/
  metadata/
    regression_outer_folds_v1.csv
    per_protein_pair_folds_v1.csv
    fold_digests.json
  shared_ols_base256/
  per_protein_ols_base256/
  sequence_gated_ols_base256/
  shared_xgboost_base256_fixed_s4/
```

For each family, require a complete `run_manifest.json` and
`model_artifacts/replay_index.csv`. Resolve every `artifact_path` relative to
its family root. Verify artifact SHA-256, configuration digest, scaler digest,
CV protocol version, feature dimension/order, cohort digest, outer-fold digest,
pair-fold digest where applicable, and complete model status before opening
pair tensors.

All four families must cover the same 552-protein eligible cohort at all 48
layers. A missing family, layer, protein, gate, pair fold, or artifact makes the
affected layer incomplete; there is no fallback to historical artifacts or a
different model.

### 2.2 Output roots

Write cluster results to:

```text
$CSV_ROOT/analysis/triangle_violation_subgroups_5fold_v1/
```

Copy the audited results and write local-only figures to:

```text
delta_files/analysis/triangle_violation_subgroups_5fold_v1/
```

Do not overwrite the historical protocol-1.0 raw results in
`delta_files/triangle_violation_subgroups/` or their derived analyses in
`delta_files/analysis/triangle_violation_subgroups/`.

### 2.3 Fixed identity and scope values

Use:

```text
analysis_name=triangle_violation_subgroups_5fold_v1
output_schema_version=2.0
cv_protocol_version=2.0
evaluation_scope=oof_all_pairs
```

Family-specific values are:

| Analysis | `model_family` | `evaluation_design` | `oof_prediction_scope` | `generalization_scope` |
|---|---|---|---|---|
| `shared_ols_base256` | `shared_ols` | `protein_outer_5fold` | `heldout_proteins_all_pairs` | `unseen_protein_ids` |
| `per_protein_ols_base256` | `per_protein_ols` | `within_protein_pair_5fold` | `heldout_pair_folds_all_pairs` | `unseen_pairs_within_seen_protein` |
| `sequence_gated_ols_base256` | `gated_ols` | `protein_outer_5fold` | `heldout_proteins_all_pairs` | `unseen_protein_ids` |
| `shared_xgboost_base256_fixed_s4` | `shared_xgboost` | `protein_outer_5fold` | `heldout_proteins_all_pairs` | `unseen_protein_ids` |

Every per-protein result retains `outer_fold`. It selects the shared-family
model and is reporting-only for per-protein OLS. A triangle row has no
`pair_fold`: per-protein triangles can contain edges from several pair folds.

## 3. Exact OOF replay contract

### 3.1 Common feature and matrix construction

Reuse `scripts/cluster_scripts/cluster_regression_helpers.py` and the canonical
cross-validation helpers for discovery, PDB resolution, C-alpha coordinates,
strict-lower-triangle pair order, fold regeneration, sequence gates, and
digest validation.

For every unordered pair, use the directional float64 base256 feature

\[
x_{ij}=[Z_{ij}\mid Z_{ji}],\qquad i>j.
\]

Load each `(R,R,128)` pair tensor once per protein-layer with plain
`np.load(...)`; extract into preallocated arrays. Preserve canonical pair order
and require finite features and predictions. Fill a symmetric `R x R` matrix
with a zero diagonal. Record raw negative-prediction counts, then clip negative
distances to zero only for triangle calculations.

### 3.2 Shared OLS and fixed XGBoost

Join each protein to `regression_outer_folds_v1.csv`. For layer `l` and protein
outer fold `f`, load exactly the `(layer=l, outer_fold=f)` artifact and predict
all unordered pairs. The model was trained on the other four protein folds, so
the complete protein distance matrix is OOF.

### 3.3 Sequence-gated shared OLS

For layer `l` and outer fold `f`, require all six
`(layer=l, outer_fold=f, sequence_gate=g)` artifacts. Route each pair through
the canonical sequence-separation gate without fallback, and require every pair
to receive exactly one prediction.

### 3.4 Per-protein OLS cross-fitted matrix

For each protein, regenerate the canonical five pair folds using the algorithm,
seed, counts, and assignment digest in `per_protein_pair_folds_v1.csv`. Reuse
the same assignment at all layers.

For each layer and pair fold `f`:

1. locate `(layer, protein_id, pair_fold=f)` in `replay_index.csv`;
2. locate that identity inside the referenced compressed layer shard rather
   than inferring NPZ order from the CSV;
3. validate the artifact and scaler state; and
4. apply it only to pairs assigned to fold `f`.

Preallocate one prediction vector in canonical pair order and one assignment
counter. Require every pair to be written exactly once, no pair to be written
twice, and 100% finite coverage before forming the symmetric matrix. Training
predictions never enter the primary analysis.

The resulting geometry estimates the consistency of the cross-fitted
within-protein predictor. It does not estimate a single model fitted on all
pairs, and it must not be described as generalization to wholly unseen
triangles because the evaluation unit frozen by the regression protocol is the
pair.

## 4. Frozen triangle category systems

Enumerate every unordered residue triple exactly once. For each protein,
derive all geometry-defined categories from its final-PDB C-alpha coordinates
once and freeze their thresholds and digests across models and layers.

For sorted true sides `a <= b <= c`, define:

\[
L_{\mathrm{RMS}}=\sqrt{(a^2+b^2+c^2)/3},
\]

\[
A=\tfrac12\lVert(C_j-C_i)\times(C_k-C_i)\rVert,
\]

\[
s_{\mathrm{rel}}=(a+b-c)/c.
\]

Freeze these category systems:

- `all_triangles`: `all`;
- `rms_size`: `small`, `medium`, `large`, using exact within-protein one-third
  and two-third quantiles with half-open intervals;
- `angle_shape`: `clearly_acute` below 88 degrees, `near_right` from 88 through
  92 degrees inclusive, and `clearly_obtuse` above 92 degrees;
- `angle_shape_exact90`: `acute`, `right_within_tolerance`, `obtuse`, using the
  sign of `c^2-a^2-b^2` and the existing scale-aware `1e-12` tolerance;
- `rms_by_angle`: the nine RMS-size by broad-angle combinations;
- `relative_slack`: `low`, `medium`, `high` within-protein tertiles; and
- `area`: `low`, `medium`, `high` within-protein tertiles.

These seven systems contain 25 category labels in total. Each system must
partition the exact all-triangle population. Record actual tertile counts
because ties can make bins unequal. Save no triangle-level rows.

## 5. Per-protein edge-model-count diagnostic

### 5.1 Definition

The source model for a per-protein OOF edge is its withheld `pair_fold`. For a
triangle with pair-fold labels `f_1`, `f_2`, and `f_3`, define

\[
n_{\mathrm{source\ models}}=|\{f_1,f_2,f_3\}|.
\]

Add the per-protein-only category system:

```text
category_system=edge_model_count
category_label=1_model | 2_models | 3_models
```

`1_model` means that one pair-fold model withheld all three edges, so all sides
come from one coherent OOF probe. `2_models` and `3_models` quantify stitched
triangles. Under approximately uniform independent pair-fold labels, expected
fractions are about 4%, 48%, and 48%; observed exact counts are authoritative.

Construct a symmetric integer pair-fold matrix once per protein. During each
triangle batch, read the three fold labels and calculate the number of distinct
labels without retaining triangle rows. The assignments and category membership
are layer-invariant.

### 5.2 Interpretation and comparisons

Report protein-equal layer trajectories for all standard violation outcomes in
the three labels. Predeclare these descriptive paired contrasts at each layer:

- `2_models - 1_model`;
- `3_models - 1_model`; and
- `3_models - 2_models`.

Do not attach a causal interpretation: model-count categories differ greatly
in support, and small prediction changes can cross the zero-violation
threshold. Emphasize violation fraction and mean relative violation over all
triangles. Retain conditional p99 and maximum as tail diagnostics, with the
same support caveats as the main analysis.

Because pair-fold assignments are random but fixed, save one layer-invariant
reference-balance table by protein and model-count label. It records support,
triangle fraction, RMS and slack summaries, and broad-angle fractions so a
systematic reference-geometry imbalance is visible. This diagnostic is not
crossed with the seven main category systems in production.

## 6. Violation outcomes and compact primitives

For clipped predicted sides `a_hat <= b_hat <= c_hat`, use the canonical
`1e-12` Angstrom tolerance and define

\[
\delta=\max(\hat c-\hat a-\hat b,0),\qquad
r=\delta/L_{\mathrm{RMS,true}}.
\]

For every analysis-layer-protein-category row, save:

- `n_triangles`, `n_violations`, absolute violation sum and maximum;
- absolute-band counts `(0,5)`, `[5,10)`, and `[10,infinity)` Angstrom;
- relative violation sum and maximum;
- relative-band counts `(0,0.05)`, `[0.05,0.10)`, and `[0.10,infinity)`;
- conditional relative-violation p50, p90, and p99; and
- explicit zero-support and zero-violation statuses.

Derive the same 13 reported metrics as the historical workflow:

1. `violation_fraction`;
2. `mean_violation_all`;
3. `mean_violation_conditional`;
4. `mean_relative_violation_all`;
5. `mean_relative_violation_conditional`;
6. `abs_0_5_fraction_all`;
7. `abs_5_10_fraction_all`;
8. `abs_10_plus_fraction_all`;
9. `rel_0_0p05_fraction_all`;
10. `rel_0p05_0p10_fraction_all`;
11. `rel_0p10_plus_fraction_all`;
12. `relative_violation_max`; and
13. `relative_violation_p99_conditional`.

Validate that every all-triangle mean equals violation fraction times its
conditional mean within floating-point tolerance. Absolute and relative band
counts must each sum to `n_violations`.

## 7. Aggregation and statistical unit

Proteins, not triangles or folds, are the independent analysis units. For each
analysis, layer, category system, category label, and metric, compute the
arithmetic mean, population standard deviation, median, quartiles, and finite
protein count directly across the 552 per-protein rows. This protein-equal mean
is primary.

Do not average the five outer-fold summaries. Fold sizes are `111, 111, 110,
110, 110`; outer fold remains provenance and may support a descriptive
sensitivity later. For per-protein OLS, outer fold is only a matched protein
group and does not select its pair-fold models.

Also calculate triangle-pooled values from additive primitives as a
length-weighted sensitivity. Do not substitute them for protein-equal results.
Conditional p99 is not reconstructible by pooling per-protein quantiles and
must remain unavailable in the triangle-pooled column.

Retain the existing protein-length checks for the seven geometry-defined
systems: five-bin stratified trajectories, continuous association with
standardized `log(n_residues)`, and protein-equal versus triangle-pooled
sensitivity. The edge-model-count diagnostic uses its dedicated reference
balance and contrast outputs instead of being added to every length analysis.

Paired comparisons among the four probe families use identical protein rows
and only the seven category systems common to all families. Do not include
`edge_model_count` in cross-family contrasts.

No bootstrap intervals, hypothesis tests, or uncertainty bands are part of
this replay. Any later peak/trough testing must be regenerated from the new
552-protein rows with its hypothesis family predeclared; historical
111-protein p-values are not reusable.

## 8. Exact output contract

```text
triangle_violation_subgroups_5fold_v1/
  run_manifest.json
  per_protein_triangle_metrics.csv
  triangle_metric_summary.csv
  edge_model_count_contrasts.csv
  edge_model_count_reference_balance.csv
  length_bin_summary.csv
  protein_length_effects.csv
  paired_model_contrasts.csv
  geometry_reproduction_summary.csv
  geometry_reproduction_failures.csv
  runtime.csv
  metadata/
    cohort.csv
    protein_reference_thresholds.csv
    preparation_failures.csv
    artifact_provenance.json
  layer_shards/
    per_protein_triangle_metrics_layer_00.csv
    ...
    per_protein_triangle_metrics_layer_47.csv
    runtime_layer_00.csv
    ...
    runtime_layer_47.csv
    artifact_provenance_layer_00.json
    ...
    artifact_provenance_layer_47.json
  visualizations/
    all_triangles_key_metrics.pdf
    rms_size_key_metrics.pdf
    angle_shape_key_metrics.pdf
    rms_by_angle_key_metrics.pdf
    relative_slack_key_metrics.pdf
    area_sensitivity_key_metrics.pdf
    edge_model_count_key_metrics.pdf
    edge_model_count_reference_balance.pdf
    protein_length_sensitivity.pdf
```

The local analyzer writes a PNG alongside every PDF with the same stem.

### 8.1 Per-protein metric rows

`per_protein_triangle_metrics.csv` has one row per family, layer, protein,
category system, and category label. Principal columns are:

```text
analysis_name,model_family,feature_dim,layer,protein_id,source,
cv_protocol_version,evaluation_design,outer_fold,evaluation_scope,
oof_prediction_scope,generalization_scope,category_system,category_label,
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

The seven common systems contribute
`4 * 48 * 552 * 25 = 2,649,600` rows. `edge_model_count` contributes
`48 * 552 * 3 = 79,488` per-protein OLS rows. A complete run therefore has
exactly `2,729,088` consolidated rows and `56,856` rows in every layer shard.

### 8.2 Summary and contrast rows

`triangle_metric_summary.csv` has one row per family, layer, category, and
derived metric, with protein-equal descriptives, finite/total protein counts,
and the triangle-pooled sensitivity and status. With 13 metrics, its complete
row count is `(4 * 25 + 3) * 48 * 13 = 64,272`.

`edge_model_count_contrasts.csv` has one row per layer, metric, and ordered
predeclared contrast. Principal columns are:

```text
layer,metric,category_label_a,category_label_b,n_paired_proteins,
mean_difference_a_minus_b,std_difference,median_difference,
q25_difference,q75_difference,contrast_status
```

`edge_model_count_reference_balance.csv` has one row per protein and model-count
label, with:

```text
protein_id,source,n_residues,protein_length_bin,category_label,
n_triangles,triangle_fraction,reference_rms_mean,reference_rms_median,
reference_slack_mean,reference_slack_median,clearly_acute_fraction,
near_right_fraction,clearly_obtuse_fraction,support_status
```

`length_bin_summary.csv` and `protein_length_effects.csv` retain their existing
row granularities for the seven main systems. `paired_model_contrasts.csv` has
one row per layer, shared category, metric, and family pair.

### 8.3 Provenance and reproduction outputs

`cohort.csv` contains one row per eligible protein with source, residue and pair
counts, length bin, outer fold, historical role, and the cohort/outer/pair-fold
digests. `protein_reference_thresholds.csv` contains one row per protein with
triangle counts, RMS/area/slack boundaries, category counts, reference
distribution summaries, and reference digest.

`artifact_provenance.json` records every producer root and manifest digest plus
the metadata digests shared across families. Layer provenance records the exact
replay-index rows and artifact hashes used for that layer without copying model
arrays.

For the `all_triangles` category, compare replayed results with each producer's
authoritative `geometry_metrics.csv`. `geometry_reproduction_summary.csv` has
one row per family, layer, and comparable metric with the compared count and
maximum absolute difference. `geometry_reproduction_failures.csv` contains
only mismatched family-layer-protein-metric rows and must be header-only for a
successful run.

## 9. Implementation stages and files

Refactor these existing cluster files in place:

```text
scripts/cluster_scripts/triangle_violation_subgroups/
  triangle_violation_subgroup_common.py
  prepare_triangle_violation_reference.py
  prepare_triangle_violation_reference.sbatch
  triangle_violation_subgroup_replay.py
  triangle_violation_subgroup_replay.sbatch
  aggregate_triangle_violation_subgroups.py
  aggregate_triangle_violation_subgroups.sbatch
  tests/test_triangle_violation_subgroups.py
```

Add a version-specific local-only analyzer so the historical analyzer remains
reproducible:

```text
scripts/multi_protein/analyze_triangle_violation_subgroups_5fold_v1.py
```

### Stage 1: common contracts and replay loaders

- Replace historical model identities and `all_edges` constants with the fixed
  protocol-2.0 identities and scopes.
- Reuse `cross_validation_common.load_cv_folds` and producer-native artifact
  loaders where practical instead of duplicating split or NPZ validation.
- Add strict replay-index key resolution and artifact digest checks for all
  four families.
- Add batched edge-model-count codes and compact accumulators without retaining
  pair or triangle prediction rows.

### Stage 2: reference preparation

- Validate the 552-protein cohort and final-PDB alignment.
- Enumerate exact reference triangles once per protein using node-local
  temporary memmaps.
- Save thresholds, reference digests, exact category support, fold provenance,
  and the layer-invariant edge-model-count balance table.
- Checkpoint after every completed protein so a timeout does not discard the
  whole cohort.

### Stage 3: 48-layer OOF replay

- Map `SLURM_ARRAY_TASK_ID` directly to one layer.
- Load one protein-layer pair tensor once and construct predictions for the
  four families.
- Assemble the per-protein OOF vector before any triangle calculation.
- Stream exact triangle batches through the seven common systems and the
  per-protein-only model-count system.
- Checkpoint atomically after a small protein batch, with restart validation of
  complete family-protein row groups.
- Finalize a layer only after all 552 proteins and all expected category rows
  are complete.

### Stage 4: streaming aggregation and local analysis

- Validate each layer independently and stream consolidated CSV output instead
  of loading 2.7 million rows at once.
- Derive summaries, contrasts, length analyses, and geometry reproduction
  checks from per-protein rows and additive primitives.
- Run the version-specific local analyzer only after copying the complete
  audited raw results into the versioned local directory.
- Generate model-count trajectories separately from the shared four-family
  category figures.

## 10. Delta execution and initial resource estimates

Deploy all cluster scripts to and run them from:

```text
$HOME/openfold_interpretability_canonical
```

Submit from `scripts/cluster_scripts/triangle_violation_subgroups/` so
`SLURM_SUBMIT_DIR` resolves colocated entrypoints and helpers. Every launcher
must retain:

```bash
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/<entrypoint>.py"
```

The historical replay measured 6.15 GiB MaxRSS and at most `00:40:50` over 111
proteins. The refactor processes about five times as many proteins per layer,
uses one prediction per pair for per-protein OLS, and adds compact fold-model
state. Use these initial estimates only for the first representative task:

| Stage | CPUs | Initial memory | Initial walltime | Basis |
|---|---:|---:|---:|---|
| Reference preparation | 8 | `12g` | `06:00:00` | Bounded one-protein working set; no exact 552-protein measurement |
| One-layer replay array task | 8 | `12g` | `06:00:00` | About 1.95x measured memory and 1.75x linearly scaled historical maximum |
| Streaming aggregation | 4 | `16g` | `02:00:00` | Layer-wise bounded aggregation of 2,729,088 final rows; unmeasured |

Run one representative replay task before broad submission, collect allocation
MaxRSS and elapsed time, and update both
`docs/cluster/suggested_sbatch_memory.md` and
`docs/cluster/suggested_sbatch_walltime.md`. Normally retain 1.5--2x measured
headroom. Measure preparation and aggregation separately because their working
sets differ. Do not inherit the historical `64g` preparation or `32g`
aggregation requests without measurement.

Use node-local temporary storage for transient triangle arrays. Keep pair
tensors on the project work filesystem and write only scripts, checkpoints,
provenance, and compact outputs under HOME-backed `$CSV_ROOT`. Recheck the live
NCSA Delta running-jobs, program-environment, accounting, and data-management
documentation immediately before implementing or changing launchers.

## 11. Tests and acceptance criteria

### 11.1 Unit and synthetic tests

- Preserve exact batched triangle enumeration and acute/right/obtuse,
  degenerate, tertile-boundary, negative-prediction, severity-band, and
  tolerance tests.
- Add a synthetic pair-fold matrix covering exactly the `1_model`, `2_models`,
  and `3_models` cases and prove that their counts partition all triangles.
- Verify that one-model triangles use three edges predicted by the same model,
  while mixed labels are calculated from source identities rather than
  predicted values.
- Verify per-protein OOF assembly gives every pair exactly one prediction and
  rejects missing or duplicate pair coverage.
- Verify replay-index resolution is key-based inside per-protein coefficient
  shards and rejects order assumptions, wrong scalers, wrong digests, or
  incomplete status.
- Verify shared outer-fold and sequence-gate routing on synthetic manifests.
- Verify means, conditional means, bands, quantiles, pooled primitives, and
  zero-support statuses independently.

Run tests and import/parity checks with `./.venv`.

### 11.2 Production acceptance

A production run is complete only if it:

- covers 48 layers, four families, and the same 552 proteins;
- has 100% pair coverage for every family-protein-layer matrix;
- has exactly `56,856` metric rows per layer and `2,729,088` consolidated rows;
- verifies all producer, artifact, configuration, scaler, cohort, outer-fold,
  and pair-fold digests;
- partitions every geometry-defined system and the per-protein model-count
  system exactly into the all-triangle count;
- reproduces all comparable producer `geometry_metrics.csv` all-triangle values
  within declared floating-point tolerances, with a header-only reproduction
  failure file;
- retains zero-support cells and explicit failure reasons rather than silently
  dropping proteins;
- saves no pair-level or triangle-level prediction rows;
- logs timestamps and elapsed time for preparation, each protein/model stage,
  checkpoints, layer completion, and aggregation; and
- records first-run MaxRSS and elapsed time in the shared resource guides before
  broad submission.

## 12. Reporting order and non-goals

Report results in this order:

1. complete-cohort OOF all-triangle reproduction;
2. RMS size and broad angle shape;
3. RMS-by-angle interaction;
4. frequency versus conditional-severity decomposition;
5. normalized severity bands and conditional p99;
6. relative slack and area;
7. protein-length sensitivity; and
8. the per-protein `1_model`/`2_models`/`3_models` stitching diagnostic.

State explicitly that per-protein triangle edges are individually OOF but a
mixed-model triangle is not a wholly held-out triangle under a triangle-level
split. The one-model category is the coherent all-three-edges-held-out subset;
the two- and three-model categories diagnose the cross-fitted matrix used by
the canonical protocol.

This refactor does not add all-pairs per-fold predictions, a full-data
per-protein refit, triangle-level cross-validation, new category cutoffs,
bootstrap intervals, or new peak/trough significance tests. Those require
separate predeclared analyses if later desired.

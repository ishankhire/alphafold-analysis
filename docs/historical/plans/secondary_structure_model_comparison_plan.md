> Historical reference; not current execution instructions. See [current sources and status](../../project_status.md).

# Secondary-Structure Model Comparison Plan

## 1. Goal

Build one primary, restartable result family that compares when alpha-helix and
beta-sheet geometry is accessible to four already-fitted distance probes:

1. the shared sequence-separation-gated OLS probe;
2. the shared global OLS probe (`general_base256_combined`);
3. the per-protein OLS probe (`per_protein_base256_combined`); and
4. the canonical shared XGBoost probe (`shared_xgboost_base256_combined`).

Evaluate every model at every layer `0..47` on the same canonical locked-test
proteins, the same DSSP annotations, the same helix calibration, and the same
beta-sheet controls. The result should answer whether the scientific
conclusions are stable across linear routing, global linear, protein-specific
linear, and shared nonlinear readouts.

The canonical cluster output root is:

```text
$HOME/csv_files/secondary_structure_model_comparison/
```

After transfer, the corresponding local result family will be:

```text
delta_files/secondary_structure_model_comparison/
```

This is a primary downstream result family, not an exploratory diagnostic
directory.

## 2. Fixed scope and non-goals

This workflow replays frozen models. It does not fit, tune, or select any new
distance probe, classifier, gate, threshold, feature set, or layer.

The sequence-gated OLS model still routes each queried intrachain pair through
its existing sequence-separation gate. That routing is part of the frozen
model. The comparison must not add new sequence features, new sequence bins,
or new predictions categorized by sequence separation.

The workflow will:

- run DSSP on the final OpenFold-predicted PDBs;
- use DSSP `H` as alpha helix and `E` as beta strand;
- calculate the existing helix offset-triple score and absolute-closeness
  requirement;
- evaluate DSSP bridge partners against matched-random and register-shifted
  controls;
- retain parallel and antiparallel sheet results separately;
- report continuous metrics, binary visibility, trajectory stability, AUROC,
  within-bridge margins, and paired-correct fractions; and
- cover all 48 layers for all four model families.

It will not:

- compare with experimental structures;
- detect helices or sheets without DSSP labels;
- train a secondary-structure classifier;
- run the exploratory per-protein XGBoost workflow;
- produce accuracy tables by amino-acid pair, true-distance bin, or sequence
  separation;
- decode every unordered pair solely to reproduce general regression metrics;
- bootstrap confidence intervals;
- choose a preferred layer from locked-test performance; or
- change any source model's training, sampling, scaling, or split.

The target structure is the model's own final predicted PDB. Conclusions must
therefore be phrased as anticipation of OpenFold's final predicted structure,
not agreement with an experimental structure.

## 3. Source-of-truth rule for model artifacts

Artifact paths, schemas, array layouts, feature order, and replay formulas must
be derived from the current producers under `scripts/cluster_scripts/`. Do not
infer any of these contracts by inspecting currently copied files under
`delta_files/`; those files may be historical or incomplete while production
runs are in progress.

The implementation must validate the following producer-defined contracts.

### 3.1 Shared global OLS

Producer sources:

- `generalized_model_ecod/general_train_test_common.py`;
- `generalized_model_ecod/general_base256_train_test_layers_00_23.py`; and
- `generalized_model_ecod/general_base256_train_test_layers_24_47.py`.

Expected result root:

```text
$CSV_ROOT/general_base256_combined/
```

Read:

```text
coefficients_0-23.csv
coefficients_24-47.csv
split_assignments_0-23.csv
split_assignments_24-47.csv
per_protein_0-23.csv
per_protein_24-47.csv
```

For each layer require exactly one intercept row (`feature_index=-1`) and 256
ordered feature rows (`feature_index=0..255`) for
`variant=general_base256_combined`. Replay directly with
`coefficient_raw_scale`:

\[
\hat d=b_{\mathrm{raw}}+x^\mathsf{T}\beta_{\mathrm{raw}}.
\]

The two split-assignment files must give identical protein membership and
metadata where their layer batches overlap conceptually. Their test IDs and
residue counts must match the canonical locked-test manifest.

### 3.2 Shared sequence-gated OLS

Producer sources:

- `gated_linear_probes/gated_linear_probe_common.py`; and
- `gated_linear_probes/aggregate_gated_linear_probe_suite.py`.

Expected result root:

```text
$CSV_ROOT/gated_linear_probes/consolidated/analysis_1_sequence_gated/
```

Read:

```text
run_manifest.json
model_status.csv
model_artifacts/coefficient_index.csv
model_artifacts/coefficient_shards/*.npz
```

For every layer require exactly six fitted index rows, one for each canonical
gate `1`, `2-4`, `5-9`, `10-19`, `20-49`, and `50+`, with
`gate_aa_pair=all`. Resolve each index row's `shard_path` and `array_row`, check
the recorded SHA-256, and load `coefficient_raw_scale` and
`intercept_raw_scale`. The gate is chosen only from same-chain sequence
separation using the canonical routing helper.

### 3.3 Per-protein OLS

Producer sources:

- `per_protein_scripts/per_protein_base256_common.py`;
- `per_protein_scripts/per_protein_base256_array.py`; and
- `per_protein_scripts/aggregate_per_protein_base256_combined_shards.py`.

Expected result root:

```text
$CSV_ROOT/per_protein_base256_combined/
```

Read:

```text
run_manifest.json
model_status.csv
metadata/pair_splits.csv
model_artifacts/coefficient_index.csv
model_artifacts/coefficient_shards/layer_XX.npz
```

For every test protein and layer, locate exactly one index row, validate its
NPZ checksum and `array_row`, and load the protein's
`coefficient_raw_scale` and `intercept_raw_scale`. Preserve fit-status flags in
the comparison metadata; do not silently substitute another model for a
failed or unstable fit.

Reconstruct the canonical within-protein pair split with
`regression_protocol.within_protein_pair_split`. Validate its seed, counts, and
digest against `metadata/pair_splits.csv` before using it for the held-out
sensitivity analysis.

### 3.4 Shared XGBoost

Producer sources:

- `xgboost_search/shared_xgboost_final_common.py`;
- `xgboost_search/shared_xgboost_final_layer.py`; and
- `xgboost_search/aggregate_shared_xgboost_final.py`.

Expected result root:

```text
$CSV_ROOT/shared_xgboost_base256_combined/
```

Read:

```text
run_manifest.json
model_status.csv
metadata/split_assignments.csv
model_artifacts/booster_layer_00.json
...
model_artifacts/booster_layer_47.json
```

Load the native booster for the requested layer and predict the selected
float64 base256 rows without clipping. Validate that the manifest identifies
the canonical directional 256-feature model, all 48 completed layers, the
frozen master split, and the expected final-model configuration. Do not use
search-stage or cap-sweep models.

## 4. Common cohort, features, and evaluation scopes

Import the eligible cohort and split from `regression_protocol.py`. Require the
canonical master-manifest digest and use the 111 `master_role=test` proteins as
the evaluation population for every model. Preparation may process all 552
eligible proteins because development proteins are required for calibration,
but decoded layer results are test-only.

For every selected unordered pair use the canonical strict-lower-triangle
direction and ordered feature vector

\[
x_{ij}=[Z_{ij}\mid Z_{ji}],\qquad i>j,
\]

with all 256 values materialized as float64 before linear algebra or XGBoost
prediction. A layer task must load a protein's pair tensor once, extract the
union of required structure pairs once, and apply all four models to the same
feature rows.

Every result row carries `evaluation_scope`. Define two scopes.

### 4.1 `all_selected_structure_pairs`

This is the primary descriptive scope. It includes every valid helix offset,
bridge partner, and control pair selected by the frozen DSSP annotations.

For shared OLS, sequence-gated OLS, and shared XGBoost, these are all
out-of-protein test observations. For per-protein OLS, this scope mixes the
80% fitting pairs with the 20% held-out pairs. Label the per-protein result
explicitly as `mixed_within_protein_train_and_heldout`; never call it fully
out-of-sample.

### 4.2 `per_protein_heldout_matched`

This is the secondary matched sensitivity scope. Apply the per-protein OLS
held-out-pair mask to all four model families so their scores use exactly the
same smaller pair population.

- A helix/strand residue is eligible only when all three pairs
  `(i,i+2)`, `(i,i+3)`, and `(i,i+4)` are in that protein's held-out set.
- A beta contrast observation is eligible only when both the true bridge
  partner and that individual random/shifted control pair are held out.
- A bridge's paired mean margin for a contrast uses only its eligible controls
  and requires at least one such control.
- Segment summaries use only eligible complete triples and must retain
  `n_scored_residues`. Rows with zero eligible residues remain represented in
  support accounting but have undefined scores and visibility.

The same held-out mask is layer-independent. Record the much smaller support
at every aggregation level. The purpose of this scope is sensitivity analysis,
not replacement of the better-supported primary curves.

## 5. Preparation stage: DSSP and frozen calibration

Implement one preparation stage shared by all layers and model families.

### 5.1 DSSP annotations

Resolve source-aware final predicted PDBs and reference pair blocks with the
canonical cluster helpers. Run the configured `mkdssp` executable for all 552
eligible proteins. Map DSSP records back to the exact ordered PDB C-alpha
residues used by the pair representation.

Use the eight DSSP states as recorded metadata, but use only:

- `H` for the main alpha-helix class; and
- `E` for the main beta-strand class and eligible bridge endpoints.

Do not merge `G`, `I`, `B`, `T`, `S`, or coil into either main class.

Cache, per protein:

- float64 C-alpha coordinates;
- chain IDs, residue numbers, insertion codes, and residue names;
- DSSP codes;
- contiguous same-chain H- and E-segment IDs and boundaries;
- deduplicated DSSP bridge pairs and orientations;
- deterministic matched-random and register-shifted controls; and
- the strict-lower-triangle positions of every required pair, including its
  per-protein train/held-out membership.

### 5.2 DSSP and indexing validation

For every protein save DSSP exit status, PDB/pair/DSSP residue counts, counts
of all DSSP codes, H/E fractions, segment counts and length distributions,
bridge counts by orientation, chain transitions, numbering gaps, and
same-chain numbering-contiguous adjacent C-alpha distances.

Flag, without silently dropping:

- DSSP execution failure;
- PDB, pair-tensor, or DSSP residue-count mismatch;
- zero H and zero E residues;
- coil fraction above 0.9;
- missing or malformed bridge mappings; and
- an adjacent C-alpha distance outside `[3.6, 4.0]` Angstrom for a same-chain,
  numbering-contiguous pair.

Aggregate H/E fractions are advisory because the combined CAMEO+ECOD cohort is
not a random protein sample. Residue-count mismatches are fatal for that
protein. Unexpected cohort loss or split mismatch makes preparation fail.

### 5.3 Helix calibration

For each residue `i` labeled H or E, retain it only if `i+2`, `i+3`, and `i+4`
exist in the same chain. From the true C-alpha distance matrix define

\[
t_i=(d_{i,i+2},d_{i,i+3},d_{i,i+4}).
\]

Using development proteins only (`master_role` in `fit, validation`):

1. calculate one combined H/E offset mean `m` and standard deviation `s`;
2. replace any scale below `1e-12` with one;
3. calculate raw centroids `mu_H_raw` and `mu_E_raw`;
4. transform the centroids to `mu_H_z` and `mu_E_z`; and
5. freeze this single calibration for all models and all layers.

For any decoded triple `t`, calculate

\[
z=(t-m)/s,
\]

\[
S_H(t)=\lVert z-\mu_{E,z}\rVert_2-\lVert z-\mu_{H,z}\rVert_2,
\]

and raw absolute closeness

\[
C_H(t)=\lVert t-\mu_{H,raw}\rVert_2.
\]

Positive `S_H` means relatively closer to the helix centroid. It does not by
itself mean the decoded geometry is close to a real helix.

Freeze two development-only thresholds:

- `helix_score_threshold_95_specificity`: the empirical 95th percentile of
  true E scores; and
- `helix_closeness_threshold_95_sensitivity`: the empirical 95th percentile of
  true H raw centroid distances.

A decoded residue `looks_helical` only when

```text
score > helix_score_threshold_95_specificity
and
raw_closeness <= helix_closeness_threshold_95_sensitivity
```

Validate the score on true matrices for both development and locked-test
proteins. Preparation fails unless the locked-test H-versus-E AUROC is finite
and at least `0.95`; the error must report the observed AUROC.

### 5.4 Helix segments

Define a helix segment as a contiguous same-chain DSSP-H run. The main segment
analysis includes original segments of length at least six and trims the first
and last H residue. Never pad offsets across a chain end.

For each model, layer, scope, protein, and eligible segment calculate:

- number of scored residues;
- mean helix score;
- mean raw distance to the helix centroid;
- mean decoded-versus-true triple RMSE;
- fraction of scored residues satisfying both residue gates; and
- binary segment visibility.

A segment is visible when it has at least one scored residue, its mean score is
above the fixed score threshold, and its mean raw closeness is no greater than
the fixed closeness threshold. Always report support so sparsely scored
held-out-sensitivity segments are not mistaken for fully observed segments.

### 5.5 Beta-sheet annotations and controls

An eligible main bridge must:

- have both endpoints labeled DSSP `E`;
- be within one chain;
- have a deduplicated DSSP bridge partner mapping; and
- have orientation `parallel` or `antiparallel`.

Create controls once during preparation and reuse them for every layer and
model.

**Matched random:** For each bridge, sample up to 20 non-bridge pairs from the
same protein and chain with exactly the same absolute sequence separation.
Sampling is deterministic from a fixed seed plus protein and bridge identity.

**Register shift 1:** Use valid `(i,j-1)` and `(i,j+1)` pairs that remain in
the same DSSP-E segment as `j`. Exclude self-pairs and known bridges.

**Register shift 2:** Apply the same rule to `(i,j-2)` and `(i,j+2)`.

Calculate true-distance ceilings on development and test proteins, separately
for parallel and antiparallel bridges:

- partner and contrast means;
- mean within-bridge margin;
- partner-versus-contrast AUROC using score `-distance`; and
- paired-correct fraction.

Freeze the sheet visibility threshold as the empirical fifth percentile of
development matched-random true distances, giving 95% specificity against
that control. A bridge is visible when its decoded partner distance is no
greater than this threshold.

## 6. One-layer prediction stage

Map `SLURM_ARRAY_TASK_ID` directly to layer `0..47`. A task must not begin until
preparation is complete and all four layer-specific model artifacts pass
validation.

For each of the 111 locked-test proteins:

1. Load its annotation cache and layer pair tensor.
2. Construct the sorted, deduplicated union of all helix offset, bridge
   partner, matched-random, shift-1, and shift-2 pairs.
3. Convert every pair to canonical strict-lower-triangle order and retain its
   canonical pair-vector position.
4. Extract one float64 directional base256 feature matrix.
5. Predict that same matrix with the shared OLS model.
6. Route each row through the frozen sequence gate and predict it with the
   corresponding gated OLS vector.
7. Predict it with that protein's per-protein OLS vector.
8. Predict it with the layer's final shared XGBoost booster.
9. Build primary and held-out-matched helix, segment, and beta rows from the
   same predictions and frozen calibration.
10. Release the tensor and feature matrix before loading the next protein.

Do not create a dense decoded distance matrix when a sparse lookup keyed by
selected residue pair is sufficient. Do not calculate unrelated all-pair
regression accuracy. Count negative predictions over the selected pair union
and label that denominator `n_decoded_structure_pairs`.

If one protein or model fails, write an explicit failure row. A layer shard is
not complete if any expected test protein lacks a prediction from any of the
four models. Do not publish partial cross-model summaries with different
protein membership.

## 7. Metrics

### 7.1 Helix metrics

For each model, layer, and evaluation scope report pooled locked-test:

- H and E residue counts;
- mean H score and mean E score;
- H-versus-E AUROC with H positive;
- mean raw H-centroid distance among true H residues;
- mean decoded and true distances at offsets 2, 3, and 4 among true H
  residues;
- pooled triple RMSE among true H residues;
- fraction of true H residues passing the relative gate, absolute gate, and
  both gates;
- number and fraction of eligible helix segments visible; and
- segment support distribution.

AUROC is the probability that a randomly selected H residue receives a higher
helix score than a randomly selected E residue. It is not threshold accuracy.

### 7.2 Beta metrics

For each model, layer, evaluation scope, and orientation in
`all`, `parallel`, and `antiparallel`, report for matched-random, shift-1, and
shift-2 contrasts:

- number of bridges;
- number of individual eligible contrast observations;
- mean decoded partner distance;
- mean decoded contrast distance;
- mean within-bridge margin, defined as
  `mean_control_distance - partner_distance`;
- partner-versus-control AUROC using `-decoded_distance`; and
- paired-correct fraction.

Define a bridge as paired-correct for one contrast when it has at least one
eligible control and

```text
decoded_partner_distance < mean_decoded_control_distance
```

Thus the paired-correct fraction is the fraction of eligible bridge cases with
a positive within-bridge margin. This differs from pooled AUROC, which compares
a randomly drawn partner observation with a randomly drawn control
observation across the full pool.

Also report the number and fraction of bridges passing the fixed sheet
visibility threshold.

### 7.3 Trajectory metrics

Because every layer is present, calculate exact 48-layer trajectories rather
than six-layer approximations.

For every helix segment and bridge record:

- first visible layer;
- whether it is ever visible;
- whether it remains visible at all later layers after first crossing;
- whether it crosses and later reverts;
- number of visibility transitions;
- longest consecutive visible run;
- final-layer visibility;
- nondecreasing helix-score indicator;
- nonincreasing helix-closeness or partner-distance indicator; and
- area under the 48-layer visibility curve, defined as the fraction of layers
  visible.

Do not interpret monotonicity as a requirement for a correct representation;
it is a descriptive stability measure.

### 7.4 Cross-model comparison

`layer_summary.csv` is the authoritative model-by-layer table.
`model_pairwise_differences.csv` is a derived convenience table. For every
unordered model pair, layer, evaluation scope, metric, and orientation where
applicable, save:

```text
value_model_a
value_model_b
difference_b_minus_a
support_model_a
support_model_b
```

No significance tests or bootstrap intervals are part of this run. Never rank
or select layers from these locked-test differences.

## 8. Exact output contract

The completed family is:

```text
secondary_structure_model_comparison/
  run_manifest.json
  calibration.json
  validation.csv
  layer_summary.csv
  model_pairwise_differences.csv
  per_protein.csv
  helix_residue_scores.csv.gz
  helix_segment_trajectories.csv.gz
  bridge_pair_trajectories.csv.gz
  runtime.csv
  failures.csv
  preparation_complete.json
  consolidation_complete.json
  metadata/
    split_assignments.csv
    input_artifacts.csv
    dssp_validation.csv
    dssp_failures.csv
  annotation_cache/
    <protein_id>.npz
  layer_shards/
    layer_00/
      shard_manifest.json
      complete.json
      layer_summary.csv
      per_protein.csv
      helix_residue_scores.csv.gz
      helix_segment_rows.csv.gz
      bridge_pair_rows.csv.gz
      runtime.csv
      failures.csv
    ...
    layer_47/
  visualizations/
    helix_strand_auroc_by_model.png
    helix_scores_by_model.png
    helix_offsets_by_model.png
    helix_segment_visibility_by_model.png
    sheet_partner_vs_random_auroc_by_model.png
    sheet_register_auroc_by_model.png
    sheet_register_margin_by_model.png
    sheet_register_paired_correct_by_model.png
    sheet_visibility_by_model.png
    heldout_sensitivity_support.png
```

### 8.1 `run_manifest.json`

One JSON object for the complete comparison. Principal fields:

- analysis name and schema version;
- status and completed layers;
- canonical master-manifest digest;
- model-family names and exact input roots;
- SHA-256 hashes of every model manifest, coefficient index/shard, coefficient
  CSV, booster, split table, calibration, and preparation manifest used;
- feature definition and order;
- evaluation protein IDs and scopes;
- DSSP executable and version;
- calibration population and formulas;
- random seed and matched-random policy;
- helix and sheet thresholds;
- implementation revision and Python/library versions; and
- output file row counts and hashes.

### 8.2 `metadata/input_artifacts.csv`

One row per consumed artifact. Principal columns:

```text
model_family, layer, protein_id, artifact_role, artifact_path,
array_row, fit_status, sha256, validation_status, details
```

Shared artifacts leave `protein_id` blank. This table proves which source
models were replayed without copying the models into the result family.

### 8.3 `validation.csv`

One row per validation check. Principal columns:

```text
category, metric, model_family, layer, split, orientation,
value, n, threshold, status, details
```

Include DSSP aggregate checks, indexing checks, split parity, artifact
coverage, coefficient/booster load checks, true-matrix helix AUROC, threshold
values, beta true-distance ceilings, per-protein split-digest parity, and
consolidated row-count/hash checks.

### 8.4 `layer_summary.csv`

One row per
`model_family x layer x evaluation_scope` on locked-test proteins. Primary key:

```text
model_family, layer, evaluation_scope
```

Principal columns:

```text
analysis_name, model_family, layer, evaluation_scope, split,
n_proteins, n_decoded_structure_pairs, n_negative_predictions,
negative_prediction_fraction,
n_helix_residues, n_strand_residues,
mean_helix_score, mean_strand_score, helix_strand_auroc,
mean_helix_centroid_distance, helix_triple_rmse,
mean_helix_offset_{2,3,4}_decoded,
mean_helix_offset_{2,3,4}_true,
helix_relative_gate_fraction, helix_absolute_gate_fraction,
helix_both_gates_fraction,
n_helix_segments_eligible, n_helix_segments_visible,
helix_segment_visible_fraction
```

For each orientation prefix `sheet_{all,parallel,antiparallel}_`, append:

```text
n_bridges, n_visible_bridges, visible_bridge_fraction,
mean_partner_distance,
matched_random_n_controls, matched_random_mean_distance,
matched_random_margin, matched_random_auroc,
matched_random_paired_correct_fraction,
shift_1_n_controls, shift_1_mean_distance, shift_1_margin,
shift_1_auroc, shift_1_paired_correct_fraction,
shift_2_n_controls, shift_2_mean_distance, shift_2_margin,
shift_2_auroc, shift_2_paired_correct_fraction
```

### 8.5 `per_protein.csv`

One row per
`model_family x layer x evaluation_scope x locked-test protein`.
Principal columns:

```text
analysis_name, model_family, layer, evaluation_scope, split,
protein_id, source, n_residues, pair_scope_provenance,
n_decoded_structure_pairs, n_negative_predictions,
n_helix_residues, n_strand_residues,
mean_helix_score, mean_strand_score, helix_strand_auroc,
mean_helix_centroid_distance, helix_triple_rmse,
n_helix_segments_eligible, n_helix_segments_visible,
n_bridges, n_parallel_bridges, n_antiparallel_bridges,
prediction_status
```

Keep support-zero rows with `NaN` metrics rather than deleting proteins.

### 8.6 `helix_residue_scores.csv.gz`

One row per
`model_family x layer x evaluation_scope x valid DSSP-H/E residue`.
Principal columns:

```text
model_family, layer, evaluation_scope, protein_id, source,
residue_index, chain, pdb_resseq, insertion_code, resname,
dssp_code, segment_id, segment_length,
offset_2_decoded, offset_3_decoded, offset_4_decoded,
offset_2_true, offset_3_true, offset_4_true,
score, distance_to_helix_centroid,
distance_to_helix_centroid_z, distance_to_strand_centroid_z,
raw_triple_rmse, passes_relative_gate, passes_absolute_gate,
looks_helical, all_three_pairs_heldout
```

### 8.7 `helix_segment_trajectories.csv.gz`

One row per
`model_family x evaluation_scope x protein x segment x layer`.
Principal columns include segment identity and boundaries, original length,
scored-residue support, layer metrics and visibility, followed by the complete
trajectory fields from section 7.3. A segment must have one row at every layer
even when its held-out-matched support is zero.

### 8.8 `bridge_pair_trajectories.csv.gz`

One row per
`model_family x evaluation_scope x protein x bridge x layer`.
Principal columns:

```text
model_family, layer, evaluation_scope, protein_id, bridge_id,
i, j, orientation, sequence_separation,
decoded_partner_distance, true_partner_distance,
n_matched_random, mean_decoded_random_distance,
mean_true_random_distance, matched_random_margin,
matched_random_paired_correct,
n_shift_1, mean_decoded_shift_1_distance,
mean_true_shift_1_distance, shift_1_margin, shift_1_paired_correct,
n_shift_2, mean_decoded_shift_2_distance,
mean_true_shift_2_distance, shift_2_margin, shift_2_paired_correct,
partner_pair_heldout, visible
```

Append the bridge trajectory fields from section 7.3 during consolidation.

### 8.9 `runtime.csv` and `failures.csv`

`runtime.csv` has one row per stage and, for the layer stage, per layer. It
records start/end timestamps, elapsed seconds, proteins processed, selected
pairs predicted, model prediction seconds, peak RSS, and completion status.

`failures.csv` retains every preparation, artifact, protein, model, layer, or
consolidation failure with explicit scope. Principal columns:

```text
stage, model_family, layer, protein_id, source, failure_type,
reason, traceback_summary, fatal
```

## 9. Plots

All plots use layers `0..47` on the x-axis and distinguish the four model
families consistently. Primary plots use `all_selected_structure_pairs`.
Where useful, the held-out-matched sensitivity appears as a lighter/dashed
curve or a separate panel with its support shown.

Plot continuous metrics and support; do not show uncertainty bands because no
bootstrap is part of this workflow. Facet beta register results by parallel
and antiparallel orientation where a combined panel would hide a disagreement.

## 10. Cluster workflow

Add these cluster-ready files in a dedicated primary-workflow directory under
`scripts/cluster_scripts/secondary_structure/`:

```text
scripts/cluster_scripts/secondary_structure/
  secondary_structure_model_comparison.py
  secondary_structure_model_comparison.sbatch
  submit_secondary_structure_model_comparison.sh
  tests/
    test_secondary_structure_model_comparison.py
```

The Python entrypoint supports exactly three stages through
`SECONDARY_STRUCTURE_STAGE`:

- `prepare`;
- `layer`; and
- `consolidate`.

The submit helper creates this dependency chain:

```text
prepare array=0
  -> layer array=0-47 with a conservative concurrency cap
    -> consolidate array=0
```

Use a 48-task layer array rather than 192 model-layer tasks so each pair tensor
is read once per protein-layer and all four models operate on identical rows.
Make concurrency configurable by
`SECONDARY_STRUCTURE_ARRAY_CONCURRENCY`, with a conservative default of six to
limit simultaneous `/work/hdd` reads.

During implementation, start from a conservative one-hour, two-CPU, 12-GiB
layer request, then run a representative layer smoke test and right-size memory
from MaxRSS with reasonable headroom before production. The sparse selected-pair
design should not require the memory used by a full all-pair decoder. If the
smoke run disproves the estimate, update the request and document the observed
runtime/MaxRSS rather than silently over-reserving.

The launcher must follow repository conventions:

- deploy and submit from `$HOME/openfold_interpretability_canonical`;
- activate `~/protein_venv`;
- export `CSV_ROOT=${CSV_ROOT:-$HOME/csv_files}`;
- set the four input roots from `CSV_ROOT` with explicit environment-variable
  overrides;
- set source-aware pair/PDB roots;
- set `DSSP_EXECUTABLE` explicitly;
- use timestamped stdout/stderr and per-layer logs;
- include `--mail-type=BEGIN,END,FAIL` and
  `--mail-user=ikhire3@gatech.edu`;
- use `SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}`; and
- launch with
  `srun python3 -u "$SCRIPT_DIR/secondary_structure_model_comparison.py"`.

Preparation, every layer, and consolidation must write completion markers
atomically. Reruns skip only shards whose manifest, hashes, configuration, and
completion marker all match the current run.

## 11. Validation and failure policy

Preparation is successful only when:

- all four model families have the exact required 48-layer artifacts;
- every artifact hash and index row validates;
- all model families agree with the canonical locked-test protein IDs and
  residue counts;
- the sequence-gated family has six fitted gates per layer;
- every test protein has one usable per-protein OLS model per layer;
- all 48 final XGBoost boosters load successfully;
- all expected PDBs and reference pair blocks resolve;
- DSSP mappings are complete for the retained cohort;
- per-protein pair split digests reproduce exactly;
- true-matrix locked-test helix AUROC is at least 0.95; and
- calibration and control supports are nonzero.

A completed layer requires all 111 test proteins and all four model families.
It must reject nonfinite selected features or predictions, missing gates,
missing protein coefficients, and dimensional disagreement. Negative finite
predictions are retained and counted rather than rejected or clipped.

Consolidation is successful only when:

- every layer `0..47` has a valid complete shard;
- primary keys are unique;
- every model has identical expected protein and layer coverage;
- detail-row supports reconstruct the summary counts;
- all recorded hashes match;
- every trajectory has the expected 48 layer positions, including explicit
  support-zero rows where required; and
- pairwise comparison rows reconstruct exactly from `layer_summary.csv`.

No partial output directory may be labeled complete.

## 12. Tests

Use `./.venv` for all local tests.

### Unit tests

1. Load shared OLS coefficients correctly across the two producer-defined
   batch CSVs and reject missing/duplicate feature rows.
2. Resolve sequence-gated coefficient index rows into the correct NPZ array
   rows and route every separation boundary correctly.
3. Resolve per-protein coefficient rows by protein and layer and reject a
   mismatched checksum, array row, fit status, or protein ID.
4. Load a native XGBoost booster through an isolated loader; when XGBoost is
   unavailable, emit the producer-consistent dependency error.
5. Preserve the ordered `[Z_ij|Z_ji]` directional feature convention for all
   model families.
6. Deduplicate the union of helix offsets, partners, and controls while keeping
   stable lookup mappings.
7. Parse DSSP bridge partners and orientations and prevent segments from
   crossing chain boundaries.
8. Generate deterministic matched-random controls with exact sequence
   separation.
9. Fit calibration only from development true matrices and reproduce the
   score, closeness, thresholds, and true-matrix AUROC formulas.
10. Require both relative and absolute gates for `looks_helical`.
11. Reproduce the canonical per-protein split and digest, map sparse selected
    pairs to strict-lower-triangle positions, and enforce all-three-held-out
    helix eligibility.
12. Require partner and individual control pairs to be held out for the beta
    sensitivity scope.
13. Distinguish pooled AUROC from the within-bridge paired-correct fraction.
14. Detect first crossing, reversion, transition count, longest run,
    monotonicity, and visibility area over a synthetic 48-layer trajectory.
15. Preserve support-zero protein, segment, and bridge rows with undefined
    metrics.

### Integration tests

Create a small synthetic fixture with:

- development and test proteins;
- directional pair tensors;
- two shared OLS vectors;
- six sequence-gated vectors;
- per-protein indexed NPZ coefficients;
- a tiny native XGBoost booster or a controlled predictor stub;
- frozen pair splits; and
- DSSP-like H/E/bridge annotations.

Verify that one layer task loads each pair tensor once, sends identical feature
rows to all models, writes both evaluation scopes, and reconstructs expected
helix and beta metrics exactly. Then synthesize all 48 layer shards and verify
consolidation, trajectory fields, pairwise differences, hashes, and output
primary keys.

### Static and launcher checks

Run:

```bash
./.venv/bin/python -m py_compile \
  scripts/cluster_scripts/secondary_structure/secondary_structure_model_comparison.py
./.venv/bin/python -m unittest \
  scripts.cluster_scripts.secondary_structure.tests.test_secondary_structure_model_comparison
bash -n scripts/cluster_scripts/secondary_structure/secondary_structure_model_comparison.sbatch
bash -n scripts/cluster_scripts/secondary_structure/submit_secondary_structure_model_comparison.sh
```

Also retain focused regression tests for the existing DSSP parser, helix gate,
bridge controls, and sequence-gated decoder when shared helpers are refactored.

## 13. Implementation sequence

1. Extract or reuse task-specific DSSP, calibration, helix-row, bridge-row,
   and trajectory helpers without changing their scientific definitions.
2. Implement producer-contract artifact loaders with checksum and split-parity
   validation for all four model families.
3. Implement preparation and the canonical annotation/calibration cache.
4. Implement sparse union-pair extraction and the four-model prediction path.
5. Implement the primary and held-out-matched evaluation scopes.
6. Implement atomic layer shards, schemas, failure rows, and runtime rows.
7. Implement strict consolidation, trajectory annotations, pairwise
   differences, and plots.
8. Add unit, integration, compilation, and shell-syntax tests.
9. Run a representative layer smoke test, inspect elapsed time and MaxRSS, and
   right-size the sbatch request.
10. Run the full dependency chain only after all four source result families
    are complete.
11. Transfer the completed family to
    `delta_files/secondary_structure_model_comparison/` and perform a
    read-only artifact audit before interpreting the results.

## 14. Interpretation safeguards

- The shared models test transfer to unseen protein IDs; the per-protein model
  tests prediction within a seen protein. Keep those claims distinct.
- The per-protein primary scope is mixed in-sample/out-of-sample. Use the
  matched held-out sensitivity scope when discussing readout generalization.
- H/E AUROC measures ranking, not threshold accuracy or geometric closeness.
- A positive helix score is insufficient without the absolute-closeness gate.
- Matched-random beta AUROC is a coarse adjacency test, not proof of correct
  beta register.
- Shift-1 and shift-2 comparisons address register; report both pooled AUROC
  and paired-correct fraction because they answer different questions.
- Use the true-distance register ceilings to distinguish model/readout failure
  from an intrinsically nonseparable control definition.
- Raw decoded distances, margins, and thresholds are interpreted within each
  layer's frozen probe. Layer trajectories are descriptive and do not imply
  that the probe itself is part of OpenFold.
- The best locked-test layer is descriptive only and must not be presented as
  validation-selected.

# Predicted-distance export plan

## Agreed scope

Replay the four canonical protocol-2.0 probes at all 48 layers for the frozen
552-protein cohort, and save **one out-of-fold (OOF) prediction for every
unordered residue pair**. The user confirmed canonical OOF scope and packed
`float32` storage on 2026-10-13. This is a plan; the exporter has not been
implemented or submitted.

The experimental rules remain in
[regression_methodology.md](../methodology/regression_methodology.md), and the
producer artifact contract remains in
[regression_outputs.md](../methodology/regression_outputs.md). This export is
an optional derived artifact: the producers currently retain models rather
than raw predictions. It does not change their fitting or evaluation rules.

Every pair receives one prediction per family and layer. Shared probes use
the model that withheld that protein's outer fold. Per-protein OLS stitches
together five disjoint held-out pair-fold predictions. Self-pairs are omitted;
when constructing a matrix, set the diagonal to zero and mirror every stored
value into both directions. Predictions remain in angstroms, including any
negative values; clipping belongs to downstream geometry calculations.

"All proteins" means all 552 frozen eligible proteins, rather than the 554
original candidates. The saved metadata is authoritative. Missing inputs or
models must produce explicit failures and an incomplete layer, never silent
protein exclusion, imputation, or a fallback to historical models.

## Where the saved models live

The local root is `delta_files/regression_5fold_v1/`. On Delta the matching
root is `$CSV_ROOT/regression_5fold_v1/`, normally
`$HOME/csv_files/regression_5fold_v1/`. In the following table all artifact
paths are relative to the named family directory. `f` is outer fold 0–4 and
`LL` is zero-padded layer 00–47.

| Probe / family directory | Authoritative prediction artifact | Inventory |
|---|---|---|
| Shared OLS: `shared_ols_base256/` | `model_artifacts/coefficients/fold_<f>_layer_<LL>.npz` | 240 files/models |
| Per-protein OLS: `per_protein_ols_base256/` | `model_artifacts/coefficient_shards/layer_<LL>.npz` | 48 files containing 132,480 models |
| Sequence-gated OLS: `sequence_gated_ols_base256/` | `model_artifacts/coefficients/gate_<slug>_fold_<f>_layer_<LL>.npz` | 1,440 files/models |
| Shared XGBoost: `shared_xgboost_base256_fixed_s4/` | `model_artifacts/boosters/fold_<f>_layer_<LL>.json` | 240 native boosters |

Gate slugs are `1`, `2_to_4`, `5_to_9`, `10_to_19`, `20_to_49`, and
`50_plus`. Resolve filenames through each family's
`model_artifacts/replay_index.csv`, rather than reconstructing them from this
table. Every index row names an artifact, responsibility, SHA-256 checksum,
configuration digest, scaler digest where applicable, and completion status.
Paths resolve relative to the family root. Its `run_manifest.json` records
protocol, feature order, fold digests, configuration, and software versions.

The OLS NPZ files contain `coefficient_standardized`,
`intercept_standardized`, `feature_mean`, and `feature_scale`, plus raw-scale
coefficients/intercepts and fit status. Per-protein shards have 2,760 model
rows per layer; identify rows by `(protein_id, pair_fold)`, not physical row
position. Shared and gated artifacts each contain one 256-feature model.

Use standardized-space replay, matching the existing production replay:

```python
prediction = intercept_standardized + (
    (features - feature_mean) / feature_scale
) @ coefficient_standardized
```

The raw-scale form is algebraically equivalent, but floating-point rounding
can differ. Retain it for inspection rather than switching replay arithmetic.
XGBoost has no linear coefficients: load the native JSON booster and predict
with the saved fixed S4 / 5,000-row-cap / 700-round model. The producer records
XGBoost 3.2.0; record the replay version and validate agreement if it differs.

The routing manifests live under the same root's `metadata/`:

- `regression_outer_folds_v1.csv` selects shared-model responsibility;
- `per_protein_pair_folds_v1.csv` records deterministic pair-fold regeneration;
- `fold_digests.json` pins the cohort and both fold definitions.

Local inventory check on 2026-10-13 found all four manifests reporting
`run_status=complete`, the expected replay-index row counts, and all 1,968
unique model files present with matching SHA-256 checksums. This checks
artifact integrity, not prediction parity against the original pair tensors.
The older protocol/output audits and adversarial-review availability records
still contain pre-production statements; do not use those statements to infer
that the current files are absent.

## Storage choice and size

Use uncompressed NumPy `.npy` arrays with explicit little-endian `float32`
(`'<f4'`), saving each unordered pair once in canonical
`np.tril_indices(R, k=-1)` order. Compute OLS replay with `float64` features,
saved scalers, and coefficients. Preserve XGBoost's native inference result,
then cast only for output. Do not use `float16` or round distances to decimal
places.

The frozen manifest contains 16,870,617 pairs per layer. The exact prediction
payload, excluding small NPY headers and metadata, is:

| Layout, all 48 layers | Bytes | Decimal GB | GiB |
|---|---:|---:|---:|
| Packed float32, one probe | 3,239,158,464 | 3.24 | 3.02 |
| **Packed float32, all four probes** | **12,956,633,856** | **12.96** | **12.07** |
| Packed float64, all four probes | 25,913,267,712 | 25.91 | 24.13 |

Full symmetric matrices would approximately double the payload and repeat
every off-diagonal value. Pair-index arrays, targets, and fold assignments
must not be repeated alongside every prediction array; recover them from the
shared index and deterministic manifests.

Plain `.npy` is not compressed. Its advantages here are a predictable modest
total size and memory-mapped access to any protein slice. Compressed `.npz`
would save an uncertain amount, require decompressing an array for ordinary
slice access, and complicate partial writes. Do not keep both formats. If
13 GB proves too large, benchmark lossless compression on a representative
completed layer before changing the format; no compression ratio is assumed.

Float32 is a storage approximation. Save measured cast error and metric drift.
For calculations sensitive to the geometry protocol's `1e-12 Å` tolerance,
replay the retained models with the existing inference precision instead of
treating float32 exports as exact substitutes for canonical geometry results.

## Exact output layout and schemas

Create a sibling of the four model-family folders:

```text
delta_files/regression_5fold_v1/predicted_distances_5fold_v1/
  README.md
  run_manifest.json
  metadata/
    protein_offsets.csv
    residue_index.csv
  prediction_index.csv
  validation.csv
  runtime.csv
  layers/
    layer_00/
      shared_ols_base256.npy
      per_protein_ols_base256.npy
      sequence_gated_ols_base256.npy
      shared_xgboost_base256_fixed_s4.npy
      layer_manifest.json
      validation.csv
      runtime.json
    ... layer_47/
  .checkpoints/
    layer_<LL>/
      <four partially written NPY arrays>
      progress.json
      validation.csv
```

On Delta, use the same relative layout beneath
`$CSV_ROOT/regression_5fold_v1/`. Each of the 192 final prediction arrays has
shape `(16870617,)`, concatenating proteins in lexicographically sorted
canonical protein-ID order. Preserve IDs as strings, including leading zeros.
Each array element is one raw OOF pair-distance prediction. Each layer is
about 270 MB across its four arrays.

The metadata and index contracts are:

| File | Row granularity / expected count | Principal fields |
|---|---|---|
| `metadata/protein_offsets.csv` | One protein; 552 rows | `protein_id`, `source`, `outer_fold`, `n_residues`, `n_pairs`, `offset_start`, `offset_stop` (exclusive), `pair_assignment_digest` |
| `metadata/residue_index.csv` | One aligned C-alpha residue | `protein_id`, `residue_index` (zero based), `chain_id`, `pdb_resseq`, `insertion_code`, `residue_name`, `pdb_path`, `pdb_sha256` |
| `prediction_index.csv` | One family/layer array; 192 rows | `analysis_name`, `model_family`, `layer`, `cv_protocol_version`, `evaluation_design`, `evaluation_scope`, `oof_prediction_scope`, `array_path`, `dtype`, `n_values`, `file_bytes`, `array_sha256`, `layer_manifest_path`, `status` |
| `validation.csv` | One family/layer/protein; 105,984 rows | Family identity and scope, `layer`, `protein_id`, `outer_fold`, `n_expected_pairs`, `n_predicted_pairs`, `pair_coverage`, `prediction_status`, `contributing_fit_status`, raw negative count, replay R²/MAE/RMSE/bias, their differences from producer metrics, `max_abs_cast_error`, `mean_abs_cast_error`, stored metric drift, `validation_status`, `failure_reason` |
| `runtime.csv` | One layer; 48 successful rows | `layer`, start/end UTC timestamps, `elapsed_seconds`, completed protein/array counts, output bytes, peak RSS if available, `status` |

Family-specific scope labels must come from validated producer manifests,
including the gated producer's explicit recombined-gate scope. Top-level CSVs
are consolidated indexes/validation; retain small layer shards for restart and
audit. Do not duplicate the large NPY arrays during consolidation.

The run manifest pins export schema `predicted_distances_5fold_v1`, CV
protocol `2.0`, `evaluation_scope=oof_all_pairs`, dtype, angstrom units,
strict-lower-triangle ordering, raw/unclipped handling, protein ordering,
metadata checksums, producer paths and manifest/index checksums, configuration
and fold digests, requested/completed layers, code revision, software versions,
and final status. Layer manifests additionally pin every model file used,
its checksum, validation summary, and final array checksums. The existing
producer replay indexes remain the authority for individual model routing.

## Proposed implementation

Add cluster entrypoints and helpers under
`scripts/cluster_scripts/predicted_distances/`:

- `predicted_distance_export_common.py`: paths, schemas, layout/checkpoint
  validation, and a reader for a protein's packed vector or reconstructed matrix;
- `prepare_predicted_distance_export.py` and its `.sbatch`: validate producers,
  freeze offsets/residue alignment, initialize the run manifest and README;
- `export_predicted_distances_layer.py` and its `.sbatch`: one task per layer,
  `SLURM_ARRAY_TASK_ID=0..47`, replaying all four families;
- `aggregate_predicted_distance_export.py` and its `.sbatch`: strict final
  coverage/integrity validation and compact index consolidation;
- `tests/test_predicted_distance_export.py`: synthetic routing, round-trip,
  coverage, quantization, and restart/corruption checks.

Reuse `load_layer_artifacts`, `load_protein`, and `model_predictions` from
`scripts/cluster_scripts/triangle_violation_subgroups/triangle_violation_subgroup_replay.py`.
They already implement the exact four-family OOF routing. Reuse canonical
artifact loaders, frozen folds, pair-indexing, PDB resolution, and feature
extraction. The export must not invoke triangle enumeration or MDS. Avoid a
large helper refactor unless needed for a concrete export requirement.

Preparation validates full replay-index coverage and hashes each unique model
file once. Build residue mapping from the same aligned C-alpha order used by
the producers; pin PDB checksums. Preserve one combined cohort, using source
labels only to resolve CAMEO/ECOD input layouts. Workers recheck each artifact
they open and verify that inputs/producer identities agree with preparation.

For each layer, load the needed models once. For each protein, load the pair
tensor once with plain `np.load`, extract the canonical directional base256
features as float64, validate shape/finiteness, and replay all four probes.
Shared OLS and XGBoost select `(outer_fold, layer)`; gated OLS selects that fold
and routes to six saved sequence gates; per-protein OLS uses `(protein_id,
pair_fold, layer)` and verifies the regenerated pair-assignment digest.

Compute compact validation before casting, then write the four float32 vectors
into their common offset slice using `numpy.lib.format.open_memmap`. Keep only
one protein's tensor/features/predictions and compact layer model state in
memory. The 794-residue maximum gives approximately 308 MiB for a float32 pair
tensor and 615 MiB for the extracted float64 features, before temporary arrays
and model/runtime overhead; never materialize the whole cohort's features.

Flush checkpoint arrays and atomically update progress/validation after each
completed protein. Progress records the four segment checksums; on restart,
accept a completed protein only after validating all four segments and pinned
input/configuration identities. An interrupted protein is recomputed. Reject
resume against incompatible manifests. Partial files stay in `.checkpoints/`.

Once all 552 proteins pass coverage and validation, close/flush the arrays,
write array checksums and a complete layer manifest, and atomically rename the
layer directory into `layers/layer_<LL>/` on the same filesystem. A worker
failure must produce an explicit diagnostic and a nonzero exit, preserving
prior checkpoints. Consolidation requires exactly 48 completed layers, 192
arrays, and 105,984 successful validation rows, with no duplicate keys or
missing pairs. Mark the top-level run complete only after these checks.

## Validation and reader behavior

1. Test small synthetic proteins: packed order, offsets, leading-zero IDs,
   symmetry/zero diagonal, correct outer-fold and gate selection, and exactly
   one held-out pair-fold write per pair. Retain negative predictions.
2. Compare uncast replay R², MAE, RMSE, bias, and negative counts with each
   producer's authoritative `per_protein.csv` and applicable geometry count.
   Use the existing replay's numerical tolerance policy where applicable;
   document any extension rather than hiding unexplained differences. Finite
   predictions from explicitly flagged OLS fits remain included with flags.
3. Require stored values to equal the direct float32 cast of replay values,
   remain finite, and cover every expected pair. Record actual rounding error
   and metric changes per protein; float32 export does not replace producer
   metric tables. Do not assert bitwise cross-version model parity.
4. Test interrupted writes, incompatible resume, a changed artifact checksum,
   corrupt completed segments, missing gates/folds, duplicate index rows, and
   absent arrays. Never infer completion from file existence alone.
5. Run one representative full-cohort layer on Delta. Validate it, record
   successful MaxRSS/elapsed time in the shared sizing guides, then run the
   remaining layers. Check final pooled metrics using merged sufficient
   statistics, rather than averaging per-protein metrics.

The reader resolves the array from `prediction_index.csv`, verifies completed
status, and returns `np.load(path, mmap_mode='r')[offset_start:offset_stop]`.
For an unordered pair `(i,j)`, swap to `i > j` and index its protein vector at
`i*(i-1)//2 + j`; no separate pair-index array is needed. For a full matrix,
fill both lower/upper triangles from that vector and set the diagonal to zero.
Include these examples in the generated README.

## Delta resources and deployment

Consult the shared [memory](suggested_sbatch_memory.md) and
[walltime](suggested_sbatch_walltime.md) guides again at implementation time.
The closest bounded four-family subgroup replay uses initial estimates of
`12g` and `02:00:00` with 8 CPUs. Its historical comparator peaked at 6.15 GiB;
12 GiB is about 1.95x that observed peak. Use those as provisional worker
requests, with an initial array concurrency cap of 4 to limit input I/O.
This workflow omits geometry but adds persistent NPY writes; 2 hours is a
comparator estimate, not a measured requirement. Right-size from the first
successful full layer before broad submission. Preparation and consolidation
use streaming hashes/metadata, with provisional `2g` / `00:15:00` each;
verify the actual working set and update both guides when adding launchers.

Deploy entrypoints and every imported helper to the flat
`$HOME/openfold_interpretability_canonical/` directory; retain the frozen master
manifest's required `regression_protocol_data/` subdirectory. Run via
`~/protein_venv`, submit from the flat deployment directory, verify every sbatch
Python reference under `scripts/cluster_scripts/`, and log stage/layer UTC
timestamps and elapsed times. Every new launcher includes mail settings
`BEGIN,END,FAIL` / `ikhire3@gatech.edu`, resolves
`SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}`, and invokes
`srun python3 -u "$SCRIPT_DIR/<entrypoint>.py"`.

Use the CPU partition, verify the allocation with `accounts`, and declare
filesystem dependencies for the actual inputs/outputs. The current repo
launchers use `scratch&work`. Live references checked for this plan:
[running jobs](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html),
[data management](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html),
and [software](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/software.html).

The requested sibling layout defaults to `$HOME/csv_files` like the existing
families. Delta's live filesystem guide says HOME is not intended for job I/O
and lists a 100 GB quota; check actual available quota for the additional
13 GB before exporting. Support a single export-root override for an allocated
`/work/hdd` or `/projects` location if needed, recording the physical root in
the manifest. Copy the completed export into the documented local sibling
directory. This plan does not deploy or submit jobs, and does not commit the
generated 13 GB arrays to Git; add an explicit generated-output ignore rule
when implementing the exporter.

After implementation, link the optional export from the normative output
document and cross-validation runbook without duplicating scientific rules.
Record the implementation briefly in `branch-analysis.md`, run the meaningful
local tests with `./.venv`, and follow the repository commit/push workflow.

> Historical reference; not current execution instructions. See [current sources and status](../../project_status.md).

# Predicted-distance export plan

## Agreed scope

Replay the four canonical protocol-2.0 probes at all 48 layers for the frozen
552-protein cohort, and save **one out-of-fold (OOF) prediction for every
unordered residue pair**. The user confirmed canonical OOF scope and packed
`float32` storage on 2026-10-13. The exporter is now implemented under
`scripts/cluster_scripts/predicted_distances/`; deployment and production
submission remain separate. See its README for commands and prerequisites.

The user subsequently requested one NPY file per protein/layer/probe. Use that
layout throughout: no array concatenates proteins, and no aggregated NPY is
produced. Coefficients and boosters remain in the existing model-family
directories and are never copied into this export.

After preparation job `22876444` failed because the `bekh` HDD quota was
exceeded, the user selected `/work/hdd/bfna/ikhire` for all generated export
data on Delta. The exact export root is
`/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/`,
including prediction files, checkpoints, indexes, validation, and manifests.

The experimental rules remain in
[regression_methodology.md](../../methodology/regression_methodology.md), and the
producer artifact contract remains in
[regression_outputs.md](../../methodology/regression_outputs.md). This export is
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

The existing models' local root is `delta_files/regression_5fold_v1/`. On Delta
their input root remains `$CSV_ROOT/regression_5fold_v1/`, normally
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
total size and direct access to an individual protein's predictions. Compressed
`.npz` would save an uncertain amount and require decompression when reading.
Do not keep both formats. If
13 GB proves too large, benchmark lossless compression on a representative
completed layer before changing the format; no compression ratio is assumed.

There will be 105,984 NPY files (`552 * 48 * 4`). The numeric payload remains
12.96 GB; individual NPY headers and filesystem allocation add some overhead.
The tradeoff is more files to transfer and index, with simpler per-protein
loading. Check available file quota as well as byte quota before submission.

Float32 is a storage approximation. Save measured cast error and metric drift.
For calculations sensitive to the geometry protocol's `1e-12 Å` tolerance,
replay the retained models with the existing inference precision instead of
treating float32 exports as exact substitutes for canonical geometry results.

## Exact output layout and schemas

Store the Delta export under
`/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/`.
The corresponding local copy, if downloaded, remains a sibling of the four
model-family folders:

```text
delta_files/regression_5fold_v1/predicted_distances_5fold_v1/
  README.md
  run_manifest.json
  metadata/
    proteins.csv
    residue_index.csv
  prediction_index.csv
  validation.csv
  pooled_validation.csv
  runtime.csv
  layers/
    layer_00/
      000000022/
        shared_ols_base256.npy
        per_protein_ols_base256.npy
        sequence_gated_ols_base256.npy
        shared_xgboost_base256_fixed_s4.npy
      ... <other protein IDs>/
      layer_manifest.json
      prediction_index.csv
      validation.csv
      pooled_validation.csv
      runtime.json
    ... layer_47/
  .checkpoints/
    layer_<LL>/
      progress.json
      prediction_index.csv
      validation.csv
      <protein_id>/
        <probe>.npy.tmp   # only while writing this individual file
```

On Delta, use this layout beneath the export root
`/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/`.
Resolve generated paths independently of the existing models' `CSV_ROOT`.
Each final prediction array has shape
`(R*(R-1)//2,)` for its protein and contains only that protein's raw OOF pair
distances, in strict-lower-triangle order. Preserve IDs as strings, including
leading zeros. Each layer is about 270 MB across its 2,208 arrays.

`layers/` holds the prediction files and small completion/validation metadata;
it never holds coefficients. `.checkpoints/` holds progress metadata and an
individual temporary prediction file while it is being written. A successful
write moves that file into its final location in `layers/`, rather than copying
it. Already completed predictions remain there across restarts. Layer/run
status stays incomplete until every expected prediction has passed validation.

The metadata and index contracts are:

| File | Row granularity / expected count | Principal fields |
|---|---|---|
| `metadata/proteins.csv` | One protein; 552 rows | `protein_id`, `source`, `outer_fold`, `n_residues`, `n_pairs`, `pair_assignment_digest`, `pdb_path`, `pdb_sha256` |
| `metadata/residue_index.csv` | One aligned C-alpha residue | `protein_id`, `residue_index` (zero based), `chain_id`, `pdb_resseq`, `insertion_code`, `residue_name`, `pdb_path`, `pdb_sha256` |
| `prediction_index.csv` | One protein/family/layer array; 105,984 rows | `protein_id`, `analysis_name`, `model_family`, `layer`, `cv_protocol_version`, `evaluation_design`, `evaluation_scope`, `oof_prediction_scope`, `array_path`, `dtype`, `n_values`, `file_bytes`, `array_sha256`, `layer_manifest_path`, `status` |
| `validation.csv` | One family/layer/protein; 105,984 rows | Family identity and scope, `layer`, `protein_id`, `outer_fold`, `n_expected_pairs`, `n_predicted_pairs`, `pair_coverage`, `prediction_status`, `contributing_fit_status`, raw negative count, replay R²/MAE/RMSE/bias, their differences from producer metrics, `max_abs_cast_error`, `mean_abs_cast_error`, stored metric drift, `validation_status`, `failure_reason` |
| `runtime.csv` | One layer; 48 successful rows | `layer`, start/end UTC timestamps, `elapsed_seconds`, completed protein/array counts, output bytes, peak RSS if available, `status` |
| `pooled_validation.csv` | One family/layer; 192 rows | `analysis_name`, `layer`, `n_pairs`, replay/stored pooled metrics, differences from producer metrics, stored-minus-replay drift, `validation_status` |

Family-specific scope labels must come from validated producer manifests,
including the gated producer's explicit recombined-gate scope. Top-level CSVs
are consolidated indexes/validation; retain small layer shards for restart and
audit. Do not duplicate the large NPY arrays during consolidation.
Consolidation combines small CSV indexes and validation summaries only; it
does not combine, average, or rewrite any prediction arrays.

The run manifest pins export schema `predicted_distances_5fold_v1`, CV
protocol `2.0`, `evaluation_scope=oof_all_pairs`, dtype, angstrom units,
strict-lower-triangle ordering, raw/unclipped handling, protein ordering,
metadata checksums, producer paths and manifest/index checksums, configuration
and fold digests, requested/completed layers, code revision, software versions,
and final status. Layer manifests additionally pin every model file used,
its checksum, validation summary, and final array checksums. The existing
producer replay indexes remain the authority for individual model routing.

## Implementation

Cluster entrypoints and helpers live under
`scripts/cluster_scripts/predicted_distances/`:

- `predicted_distance_export_common.py`: paths, schemas, layout/checkpoint
  validation, and a reader for a protein's packed vector or reconstructed matrix;
- `prepare_predicted_distance_export.py` and its `.sbatch`: validate producers,
  freeze protein metadata/residue alignment, initialize the run manifest and README;
- `export_predicted_distances_layer.py` and its `.sbatch`: one task per layer,
  `SLURM_ARRAY_TASK_ID=0..47`, replaying all four families;
- `finalize_predicted_distance_export.py` and its `.sbatch`: strict final
  coverage/integrity validation and compact CSV index consolidation; no NPY
  aggregation;
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
to four separate files using `np.save(..., allow_pickle=False)`. Each filename
is scoped to its protein, layer, and probe. Keep only
one protein's tensor/features/predictions and compact layer model state in
memory. The 794-residue maximum gives approximately 308 MiB for a float32 pair
tensor and 615 MiB for the extracted float64 features, before temporary arrays
and model/runtime overhead; never materialize the whole cohort's features.

Write each NPY to a temporary file beneath `.checkpoints/layer_<LL>/`,
flush/fsync and validate it, then atomically move it to
`layers/layer_<LL>/<protein_id>/<probe>.npy` on the same filesystem. Use an
open file handle with `np.save` so it does not append `.npy` to the temporary
filename. No full-layer partial arrays or second copy of completed arrays are
needed.

Atomically update the authoritative `progress.json` after each completed
protein, including its four file paths/checksums and index/validation records.
The checkpoint CSVs are derived from this progress snapshot and can be rebuilt
if interrupted while being written. On restart, accept a
completed protein only after validating all four files and pinned
input/configuration identities. If a crash occurred after a file was moved but
before the checkpoint was committed, recompute that uncommitted protein and
replace its files atomically. Reject resume against incompatible manifests.
Files from incomplete layers are not advertised as a complete export.

Once all 552 proteins pass coverage and validation, finalize the small layer
index/validation/runtime files and write a complete layer manifest last.
Remove only temporary files/progress superseded by validated final metadata.
A worker failure produces an explicit diagnostic and a nonzero exit,
preserving completed files and prior checkpoints. Finalization requires
exactly 48 completed layers, 105,984 arrays, and 105,984 successful validation
rows, with no duplicate keys or missing pairs. Mark the top-level run complete
only after these checks.

## Validation and reader behavior

1. Test small synthetic proteins: packed order, per-protein paths, leading-zero IDs,
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
   corrupt completed files, missing gates/folds, duplicate index rows, and
   absent arrays. Never infer completion from file existence alone.
5. Submit the full 48-layer array after preparation succeeds, following the
   user's explicit decision to omit a separate one-layer pilot. Record successful
   MaxRSS/elapsed time in the shared sizing guides for future sizing. Check
   final pooled metrics using merged sufficient
   statistics, rather than averaging per-protein metrics.

The reader resolves the protein/layer/probe array from `prediction_index.csv`,
verifies completed status, and returns `np.load(path, allow_pickle=False)`.
For example:

```python
prediction = np.load(
    export_root / "layers/layer_00/000000022/shared_ols_base256.npy",
    allow_pickle=False,
)
```

For an unordered pair `(i,j)`, swap to `i > j` and index its protein vector at
`i*(i-1)//2 + j`; no separate pair-index array is needed. For a full matrix,
fill both lower/upper triangles from that vector and set the diagonal to zero.
Include these examples in the generated README.

## Delta resources and deployment

Consult the shared [memory](../../plans/suggested_sbatch_memory.md) and
[walltime](../../plans/suggested_sbatch_walltime.md) guides again at implementation time.
The closest bounded four-family subgroup replay uses initial estimates of
`12g` and `02:00:00` with 8 CPUs. Its historical comparator peaked at 6.15 GiB;
12 GiB is about 1.95x that observed peak. Use those as provisional worker
requests, with an initial array concurrency cap of 4 to limit input I/O.
This workflow omits geometry but adds persistent NPY writes; 2 hours is a
comparator estimate, not a measured requirement. The user chose full-array
submission without a separate pilot; right-size from successful allocations
before future runs. Preparation uses `3g` / `00:15:00`, allowing about 1.68x
the 1.79 GiB peak seen before its quota failure. Finalization uses provisional
`2g` / `00:15:00` with streaming hashes/metadata.
Account for the extra file creation, stat, and hashing work when measuring
elapsed time; do not assume the same timing as a 192-file layout.

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

Keep the model-input root and export-output root separate. `CSV_ROOT` still
selects the existing model inputs, normally `$HOME/csv_files`; do not change
it to relocate generated predictions. Every new exporter entrypoint instead
uses `PREDICTED_DISTANCE_OUTPUT_ROOT`, defaulting to
`/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/`.
Set that value explicitly in the export launchers and record the resolved
physical root in the manifest. All export metadata and temporary files use
this same root; none are generated under `$HOME/csv_files`. The existing model
artifacts are read in place without moving or duplicating them.

Check write access and byte/file quota at the chosen `/work/hdd` location
before exporting. If downloading results, copy the completed export into the
documented local sibling directory. Deployment and preparation submission were
authorized following the quota failure. Generated 13 GB arrays are excluded
from Git by the explicit local export directory ignore rule.

The normative output document and cross-validation runbook now link to the
export runbook. Implementation validation covers synthetic full-stage runs,
failure/resume/corruption cases, and real saved-OLS replay parity on an available
local protein. Native XGBoost prediction parity remains a required per-layer
Delta check; allocation resource sizing will use successful array measurements.

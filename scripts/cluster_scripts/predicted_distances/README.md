# Packed OOF predicted-distance export

Replay the four frozen protocol-2.0 probes without fitting models. Scientific
rules remain in [regression_methodology.md](../../../docs/methodology/regression_methodology.md).
The [output contract](../../../docs/methodology/regression_outputs.md) defines
its relationship to the canonical producer outputs.

## Paths and files

Existing model inputs are read from
`$CSV_ROOT/regression_5fold_v1/`, normally
`$HOME/csv_files/regression_5fold_v1/`. All export files, including temporary
files and checkpoints, default to:

```text
/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/
  run_manifest.json
  metadata/proteins.csv
  metadata/residue_index.csv
  layers/layer_<LL>/<protein_id>/<probe>.npy
  layers/layer_<LL>/layer_manifest.json
  layers/layer_<LL>/prediction_index.csv
  layers/layer_<LL>/validation.csv
  layers/layer_<LL>/pooled_validation.csv
  layers/layer_<LL>/runtime.json
  .checkpoints/layer_<LL>/progress.json
  prediction_index.csv
  validation.csv
  pooled_validation.csv
  runtime.csv
```

The four probe filenames are `shared_ols_base256.npy`,
`per_protein_ols_base256.npy`, `sequence_gated_ols_base256.npy`, and
`shared_xgboost_base256_fixed_s4.npy`. Each file is a little-endian float32
vector of shape `(R*(R-1)//2,)`, in `np.tril_indices(R, k=-1)` order, with
one raw OOF distance per unordered pair in angstroms. Negative predictions are
retained. No diagonal, upper triangle, coefficients, targets, or aggregated
prediction array is saved. The 552 proteins, 48 layers, and four probes produce
105,984 arrays with a 12.96 GB numeric payload plus headers/metadata.

`prediction_index.csv` has one row per array, with family/protein/layer identity,
scope, relative path, dtype, pair count, bytes, SHA-256, and completion status.
`validation.csv` has one row per array recording full pair coverage, contributing
fit statuses, replay parity with producer metrics, raw/stored negative counts,
float32 cast errors, and metric drift. `pooled_validation.csv` has four rows per
layer (192 final rows); it merges sufficient statistics across proteins and
checks the producer's pooled metrics. `runtime.csv` has 48 rows.

Preparation records PDB checksums and aligned residue identities, including
chain, PDB residue number, insertion code, and zero-based array index. It also
pins producer/configuration/fold digests, code hashes, and software versions.
Workers verify every opened model artifact and require complete OOF coverage.
An invalid gate/fold/model or metric mismatch fails the layer without fallback.

## Deployment and submission

Follow the [cluster guide](../../../docs/cluster/README.md); deployment and job
submission require the user's permission. Current execution status lives in
[current sources and status](../../../docs/project_status.md).

Deploy files flat to `$HOME/openfold_interpretability_canonical/`:

- All four `.py` and three `.sbatch` files in this directory.
- Copy this README as `predicted_distance_export_README.md` in the flat directory.
- All non-test `.py` files under `scripts/cluster_scripts/cross_validation/`.
- `triangle_violation_subgroup_common.py` and
  `triangle_violation_subgroup_replay.py` from `triangle_violation_subgroups/`.
- `cluster_regression_helpers.py`, `regression_protocol.py`, and
  `regression_output_helpers.py` from the parent cluster directory.
- The canonical `regression_protocol_data/regression_master_split_v1.csv`
  under its sole retained subdirectory.

The existing four producer families, three frozen metadata files, and all
per-protein/geometry/pooled CSVs must already exist under the model input root.
`~/protein_venv` needs the producer dependencies and XGBoost (producer version:
3.2.0). Preparation and workers must use the same environment/code. A changed
environment, producer, PDB, or checkpointed pair tensor is rejected on resume.

Check `accounts`, filesystem byte/file quota, and write access first. Create
the Slurm log directory before submission because Slurm opens stdout before
the script starts:

```bash
mkdir -p /work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/logs
cd "$HOME/openfold_interpretability_canonical"
sbatch prepare_predicted_distance_export.sbatch
```

After preparation succeeds, submit all 48 layers, with concurrency 4:

```bash
sbatch export_predicted_distances_layer.sbatch
```

No separate one-layer pilot is required. Record successful allocation MaxRSS
and elapsed time in the shared guides and right-size the provisional 12 GiB /
2 hour worker requests for
future runs.

After all 48 layers succeed:

```bash
sbatch finalize_predicted_distance_export.sbatch
```

Preparation requests 3 GiB / 15 minutes, allowing 1.68x the approximately
1.79 GiB peak observed before the original quota failure. Finalization initially
requests 2 GiB / 15 minutes; it streams one layer's metadata at a time.
The CPU partition, filesystem dependencies,
and module setup follow the live
[Delta running-jobs guide](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html).

`CSV_ROOT` controls existing model inputs. `PREDICTED_DISTANCE_OUTPUT_ROOT`
independently overrides the export root if needed. The launchers' Slurm log
paths explicitly target the agreed `/work/hdd` root; overriding the export
root also requires `sbatch -o ... -e ...` pointing at an existing log directory.

## Restart and validation

A file is written to a temporary file in `.checkpoints/`, flushed, verified,
and atomically moved into `layers/`. After all four files for a protein pass,
the worker atomically commits their paths, checksums, validation, sufficient
statistics, and pair-input checksum to `progress.json`. Completed files serve
as checkpoints; they are not duplicated. If interrupted before the progress
commit, that protein is recomputed. Resume rechecks committed files and inputs.

Layer manifests are marked complete only after every protein passes and pooled
replay metrics match the originals (`rtol=atol=1e-8`). Finalization requires
exactly 48 layers and all 105,984 prediction files, checks every checksum, and
combines only CSV metadata. It never merges NPY arrays. Failure details remain
in progress metadata and Slurm logs. Successful checkpoint metadata is removed
after validating the complete layer; final layer manifests retain provenance.

## Reading a protein

Simple NumPy loading for a known completed layer:

```python
from pathlib import Path
import numpy as np

root = Path("/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1")
v = np.load(root / "layers/layer_00/000000022/shared_ols_base256.npy",
            allow_pickle=False)
# For residues i > j, the packed index is i*(i-1)//2 + j.
```

The helper checks completion, shape, dtype, and checksum before returning a
vector or a full symmetric matrix:

```python
from predicted_distance_export_common import load_predicted_distances

v = load_predicted_distances(root, 0, "000000022", "shared_ols_base256")
D = load_predicted_distances(root, 0, "000000022", "shared_ols_base256",
                            as_matrix=True)
```

`D` has a zero diagonal and mirrors every lower-triangle prediction; it does
not clip negative values. Float32 exports approximate inference precision.
Replay the retained models for geometry calculations sensitive to `1e-12 Å`.

Run local tests with:

```bash
./.venv/bin/python -m pytest scripts/cluster_scripts/predicted_distances/tests -q
```

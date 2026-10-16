# Reconstruction TM-score optimization

Scientific scope and geometry definitions live in
[regression_outputs.md](../../../docs/methodology/regression_outputs.md).
The existing regression geometry CSV schemas are unchanged. Fresh geometry
calculations search both MDS reflections with the official TMscore executable;
minimum Kabsch RMSD is still calculated independently. Completed regression
checkpoints and saved scores are not automatically rewritten.

## Install the executable

Build on each target platform. The local validation used upstream commit
`746ad8a2a4dea9f317075cc482df0ecb5440bdea` from
[pylelab/USalign](https://github.com/pylelab/USalign). From the repository root:

```bash
git clone https://github.com/pylelab/USalign.git .cache/tmscore-source
git -C .cache/tmscore-source checkout 746ad8a2a4dea9f317075cc482df0ecb5440bdea
g++ -O3 -o scripts/cluster_scripts/TMscore .cache/tmscore-source/TMscore.cpp
```

The compiled binary is ignored by Git. On Delta, compile the same source into
`$HOME/openfold_interpretability_canonical/TMscore`. A Linux build is required;
do not deploy the local macOS binary. `TMSCORE_BINARY=/absolute/path/TMscore`
overrides the colocated binary; PATH is searched last. Missing or failed
executables abort scoring without substituting a historical score.

Temporary CA traces use sequential residue identities in one chain, preserving
array correspondence across real chain boundaries and insertion codes. The
standard search uses reference-length protein normalization and proper rigid
rotations. Its transformation is evaluated on original float64 coordinates;
the Kabsch candidate supplies a score floor for the same coordinates. Search
inputs normally have PDB's 0.001 Å precision (extreme coordinates use scientific
notation); the stored score is not the four-decimal printed value. This is a
heuristic optimization, not a guarantee of the global maximum. Inputs with fewer
than three residues are unsupported by the executable; this does not affect the
frozen 552-protein cohort, whose minimum length is 30.

## Rescore the packed export

See the [cluster guide](../../../docs/cluster/README.md) for access and submission
permissions. The commands below require authorization to change cluster state.

Run only after the predicted-distance exporter is finalized. Keep its deployment
frozen until it finishes: its restart identity includes helper-code hashes.
The rescoring reader checks the export's saved checksums without requiring its
historical helper hashes to equal the new implementation.

Deploy `recompute_tm_scores.py`, `consolidate_tm_scores.py`, both `.sbatch`
launchers, and the changed shared helpers flat
to `$HOME/openfold_interpretability_canonical`. Retain the predicted-distance
exporter's existing Python dependency inventory, its deployed README, and frozen
protocol data. Copy this README as `tm_scores_README.md`.

Default inputs:

- Packed arrays: `/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1/`.
- Source geometry CSVs: `$CSV_ROOT/regression_5fold_v1/`, with
  `CSV_ROOT=$HOME/csv_files` on Delta.
- References: original PDB paths recorded in `metadata/proteins.csv`, with
  residue identities and insertion codes checked against `metadata/residue_index.csv`.

Outputs go to `/work/hdd/bfna/ikhire/regression_5fold_v1/tm_score_optimized_v1/`.
For local copies use `delta_files/regression_5fold_v1/tm_score_optimized_v1/`.
The inputs are read-only; model artifacts and pair tensors are never loaded.
Each layer processes all four probes and 552 proteins, reconstructing only MDS
and TM-scores, and saving every reconstructed C-alpha trace in the winning
TM-score alignment and the original reference PDB coordinate frame. Existing
non-TM geometry metrics are copied rather than recomputed. Float32 distance
storage can cause small differences from the original full-precision
reconstruction. Saved PDBs use 0.001 Å precision; the ledger retains scores
evaluated before PDB rounding. The ledger separates alignment gain from the
difference against the old saved score.

```bash
mkdir -p /work/hdd/bfna/ikhire/regression_5fold_v1/tm_score_optimized_v1/logs
cd "$HOME/openfold_interpretability_canonical"
sbatch --array=0-47%4 recompute_tm_scores.sbatch
```

Submit all 48 layers with concurrency four; no separate resource pilot is
required. Record successful allocation MaxRSS and elapsed time in the shared
resource guides to refine the 2 GiB / 2 hour request for future runs. The prior
score-only run reached 0.975 GiB allocation-step MaxRSS, leaving little headroom
under its original 1 GiB request; PDB export and indexing add a small working set.

After all layers complete, consolidate their compact tables:

```bash
sbatch consolidate_tm_scores.sbatch
```

Consolidation requests 3 GiB / 10 minutes. Production job `22930940` completed
in 1m43s with Python-step MaxRSS 1,634,616 KiB (1.56 GiB); the request retains
1.92x memory headroom and the guide's minimum scheduling allowance. It verifies all 48
completed layer manifests, score/PDB index CSV checksums, identities, coverage,
and prediction hashes recorded in the export manifests. It does not reopen
individual prediction arrays, reconstructed PDBs, or reference PDBs. Original
scoring workers already validated the inputs and recorded PDB checksums at
write time. The root and per-probe manifests explicitly record
`validation.mode=consolidation_only` and the skipped bulk revalidation.

The optional full-file audit remains available:

```bash
sbatch --array=0 --mem=3g --time=09:00:00 --export=ALL,TMSCORE_FINALIZE=1 recompute_tm_scores.sbatch
```

Audit job `22923530` verified layers 0–23 without validation errors before
user-requested cancellation after 3h03m. It spent only about 4m20s of CPU time
in nearly three hours; serial shared-filesystem access dominates. Its original
4.5-hour estimate was insufficient. The optional audit's 9-hour request is
based on an approximately six-hour whole-run extrapolation with 1.5x headroom.
It does not checkpoint verified layers. Run only one audit/consolidation task
at a time; both use the same exclusive finalization lock.

For a local run, override roots explicitly:

```bash
./.venv/bin/python scripts/cluster_scripts/tm_scores/recompute_tm_scores.py \
  --layer 0 \
  --predictions-root delta_files/regression_5fold_v1/predicted_distances_5fold_v1 \
  --regression-root delta_files/regression_5fold_v1 \
  --output-root delta_files/regression_5fold_v1/tm_score_optimized_v1
```

After scoring all 48 layers, run `consolidate_tm_scores.py` with the same root
arguments (without `--layer`). The scoring worker's `--finalize` mode retains
the optional full-file audit. Environment
overrides are `PREDICTED_DISTANCE_OUTPUT_ROOT`, `CSV_ROOT`, and
`TMSCORE_OUTPUT_ROOT`. Changing the log destination requires matching `sbatch`
`-o`/`-e` overrides pointing to an existing directory.

## Output contract and restart

`layers/layer_<LL>/tm_scores.csv` and consolidated `tm_scores.csv` contain one
row per probe/protein/layer. Principal fields are the existing OOF identity,
`n_residues`, `tm_score`, `tm_score_kabsch`, `tm_score_saved`,
`delta_tm_score_saved`, `delta_tm_score_alignment`, `array_sha256`,
`tm_score_method`, and `status`. Protein IDs remain strings.

All 105,984 reconstructed C-alpha PDBs are saved under
`layers/layer_<LL>/pdb_files/<analysis_name>/<protein_id>_reconstructed_ca.pdb`.
The files preserve residue names, chain IDs, PDB numbers, and insertion codes
from `metadata/residue_index.csv`. Chain boundaries have `TER` records;
occupancy is 1 and B factors are 0. They contain the winning reflection and
superposition, including a Kabsch winner when it scores best. No all-atom
coordinates or additional original/reflected candidate files are inferred.

`layers/layer_<LL>/pdb_index.csv` and consolidated `pdb_index.csv` contain one
row per reconstruction: `protein_id`, `analysis_name`, `layer`, `n_residues`,
`pdb_path` (relative to the output root), `pdb_sha256`, `array_sha256`,
`tm_score`, and `status`. The consolidated index has 105,984 rows.

Per-protein JSON checkpoints under `layers/layer_<LL>/progress/` retain four
completed scores, prediction hashes, and PDB index records. Checkpoints are
committed only after all four PDB writes succeed. Resume validates predictions,
references, and saved PDB checksums before skipping native scoring. Missing or
corrupted reconstructed PDBs regenerate the affected protein's four scores/PDBs;
completion is invalidated before any repair writes. Changed inputs, code, or
executable identity require a new or explicitly cleared output root. The layer
manifest checksums both CSVs and records the PDB count. Preparation waits for
concurrent starts; duplicate workers for the same layer remain rejected.

Finalization writes each `<probe>/geometry_metrics.csv` and
`<probe>/geometry_summary.csv` using the existing column schemas, plus a small
scoring provenance manifest per probe. Root `run_manifest.json` records input
hashes, executable identity, scoring/PDB policy, completion, `n_pdbs`, output
hashes, and checksums of all 48 layer manifests;
`runtime.csv` records each layer's elapsed time and process peak memory.

`consolidation_runtime.json` records consolidation elapsed time, peak RSS,
score/PDB index row counts, and validation mode. Root `consolidation` records
the consumer script's SHA-256 separately from the unchanged scoring
`identity`/`identity_digest`. Deploying this additional consumer does not
invalidate completed workers or require changing their frozen code hashes.

These are replacement geometry tables, not complete regression result families.
They are ready for the later publication refresh; source model manifests and
canonical inputs are preserved. This change does not regenerate figures or edit
the paper.

## Clean PDB-enabled rerun of the October 14 array

Read-only inspection on October 15 found 37 completed layers in `22908533`.
Layers 16, 17, 18, 23, 33, 38, 39, 40, 43, 45, and 47 all failed with
`Another prepare task is running`; finalizer `22908534` is pending with
`DependencyNeverSatisfied`. The waiting preparation lock fixes this race.
The user chose to discard these partial scoring outputs and rerun at the same
path. No prediction export or model fitting is needed.

**These commands change Delta state and require separate user authorization.**
Before executing, confirm no scoring workers are active and the export root
manifest is complete. From the local repository, deploy only these changed
files; retain the existing Linux `TMscore` binary and exporter dependencies:

```bash
scp scripts/cluster_scripts/tm_scores/recompute_tm_scores.py \
  scripts/cluster_scripts/tm_scores/recompute_tm_scores.sbatch \
  scripts/cluster_scripts/tm_scores/consolidate_tm_scores.py \
  scripts/cluster_scripts/tm_scores/consolidate_tm_scores.sbatch \
  scripts/cluster_scripts/cluster_regression_helpers.py \
  delta:openfold_interpretability_canonical/
scp scripts/cluster_scripts/tm_scores/README.md \
  delta:openfold_interpretability_canonical/tm_scores_README.md
```

On Delta, cancel the blocked finalizer if it is still queued, remove only the
disposable scoring tree, recreate its logs, and submit the full array and a
success-dependent finalizer:

```bash
scancel 22908534
rm -rf -- /work/hdd/bfna/ikhire/regression_5fold_v1/tm_score_optimized_v1
mkdir -p /work/hdd/bfna/ikhire/regression_5fold_v1/tm_score_optimized_v1/logs
cd "$HOME/openfold_interpretability_canonical"
unset TMSCORE_FINALIZE
tm_array_id=$(sbatch --parsable --array=0-47%4 recompute_tm_scores.sbatch)
tm_array_id=${tm_array_id%%;*}
sbatch --dependency="afterok:$tm_array_id" consolidate_tm_scores.sbatch
```

Confirm the root manifest is complete with `n_scores == n_pdbs == 105984` and
48 layer-manifest hashes. Consolidation checks recorded artifact identities;
only the optional audit checks physical PDB checksums again. Do not reuse the
old finalizer's failed dependency. This clean-rerun procedure is unnecessary
when merely replacing an audit with consolidation: preserve every layer/PDB
and submit only `consolidate_tm_scores.sbatch` after stopping the audit.

```bash
./.venv/bin/python -m pytest scripts/cluster_scripts/tm_scores/tests -q
```

Native tests need the installed executable. Local PDB smoke results are retained
under `delta_files/analysis/tm_score_optimization_validation_v1/`.

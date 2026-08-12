# CAMEO-50 Pair-Interaction Annotation Prototype

## Purpose and scope

Build a reusable structural-annotation cache for the first 50 TM-score-filtered
CAMEO proteins listed in `datasets.md`. The prototype labels every unordered
residue pair in the saved unrelaxed OpenFold structures for:

- salt bridges;
- DSSP backbone hydrogen bonds and beta bridges;
- hydrophobic contacts and buried hydrophobic-core contacts.

This stage produces labels and structural covariates only. Pair-representation
extraction, matched-control construction, protein splits, and classifier fitting
remain separate downstream steps. Results describe interactions in OpenFold's
predicted structures, not independent experimental biology.

## Runtime workflow

The implementation lives in `scripts/cluster_scripts/diagnostics/`:

1. `setup_dssp_delta.sbatch` installs DSSP 4.6.1 into
   `$HOME/dssp_env`. Delta RH9 no longer provides Anaconda modules, so the job
   bootstraps the standalone Micromamba executable under `$HOME/.local/bin` and
   uses conda-forge without changing the user's shell configuration.
2. `diagnostic_pair_interaction_annotation.sbatch` activates the existing
   `~/protein_venv`, checks the Python dependencies, DSSP executable, and input
   roots, then launches `diagnostic_pair_interaction_annotation.py`.
3. The Python job processes one protein at a time and checkpoints completed
   gzip/DSSP artifacts. A matching configuration hash allows a restart to skip
   completed proteins; a different hash causes a hard failure rather than
   mixing annotation definitions.

The Delta inputs are the CAMEO layer-0 pair blocks under
`/work/hdd/bekh/thayes/openfold_reps` and unrelaxed PDBs under
`/work/hdd/bekh/thayes/openfold_reps_outputs_sm`. Layer 0 is loaded only to
validate that structural residue order and pair-block indexing agree.

## Residue and protein eligibility

For each requested protein, parse model 0 and retain standard amino-acid
residues containing C-alpha in PDB order. Exactly one polymer chain is
required. The layer-0 pair block must have shape `(R, R, 128)`, where `R` is
the retained residue count.

Missing inputs, parse errors, multiple polymer chains, and dimension mismatches
exclude the protein. Backbone and side-chain completeness are recorded per
residue. Coordinates and distance calculations use `float64`.

This strict mapping is preferable to guessing around missing residues because
the cached pair indices must remain directly compatible with pair-block indices.

## Interaction definitions

All geometric distances use heavy atoms only. Every label is nullable and has
an explicit validity field so unavailable evidence does not silently become a
negative.

### Salt bridge

Candidate pairs contain ASP/GLU and LYS/ARG. The eligible atoms are ASP
`OD1/OD2`, GLU `OE1/OE2`, LYS `NZ`, and ARG `NH1/NH2`; histidine and ARG `NE`
are excluded. A candidate is positive at a minimum eligible N--O distance of
at most 4.0 A. A candidate missing its required charged atoms is invalid;
noncandidate residue pairs are valid negatives.

### DSSP backbone interactions

Run `mkdssp --output-format=dssp` once per protein and retain its classic DSSP
output for auditing. If a source PDB lacks `CRYST1`, prepend a large dummy
`P 1` record to a temporary copy without modifying the source file.

A directed backbone hydrogen bond is present when a DSSP donor entry targets
the other residue with energy below -0.5 kcal/mol. The unordered label is
positive if either direction qualifies. A beta bridge is present when either
residue's `BP1` or `BP2` index points to the other. Separate hydrogen-bond and
beta-bridge labels are retained along with their union.

DSSP secondary-structure codes are collapsed as `H/G/I` to helix, `E/B` to
beta, and `T/S/P/-` to coil. Absolute ASA is normalized using Biopython's
Sander maximum-accessibility scale and capped at 1.0.

A per-protein DSSP failure does not discard geometric or salt-bridge results.
Instead, DSSP mapping, secondary structure, SASA, backbone interaction, and
hydrophobic-core fields are invalidated and the failure is recorded. There is
no alternate SASA method, avoiding mixed annotation methods across proteins.

### Hydrophobic contact and core contact

The hydrophobic set is ALA, VAL, ILE, LEU, MET, PHE, TRP, and TYR. A pair is a
hydrophobic contact when both residues are in this set and their minimum
side-chain-heavy-atom distance is at most 4.5 A. A hydrophobic-core contact
additionally requires both relative SASAs to be at most 0.20.

A hydrophobic candidate missing side-chain atoms is invalid. A core candidate
also becomes invalid when either relative SASA is unavailable. Nonhydrophobic
pairs are valid negatives.

## Computation and checkpointing decisions

C-alpha distances are computed directly from the retained `float64` coordinates,
while minimum atom distances are reduced from per-residue heavy-atom arrays.
Pair rows are streamed directly to gzip in lexicographic `i < j` order instead
of materializing a protein-wide table.
Each successful protein must contain exactly `R(R-1)/2` pair rows.

Temporary per-residue, per-pair, and DSSP files are renamed only after the
protein passes row-count validation. Aggregate counts are rebuilt from all
completed per-pair checkpoints on every run, ensuring restart results reconcile
with the cached annotations rather than only the proteins processed in that
invocation.

## Stable output interface

Write all prototype outputs beneath:

```text
$HOME/csv_files/pair_interaction_classification/cameo50_v1/
```

The files are:

- `annotation_config.json`: annotation definitions, thresholds, atom and
  residue sets, input roots, DSSP version, cohort, version, and configuration
  hash.
- `protein_manifest.csv`: one row per requested protein, including input
  paths, PDB hash, residue count, pair-block shape, expected pair count,
  status, configuration hash, and failure reason.
- `annotations/per_residue/<protein_id>.csv.gz`: one row per pair-block residue
  index with PDB identity, completeness, DSSP mapping/code/class, absolute and
  relative SASA, phi/psi, and DSSP failure state.
- `annotations/per_protein/<protein_id>.csv.gz`: one row per unordered pair.
  Principal fields include pair/residue identifiers, sequence separation,
  C-alpha and minimum atom distances, DSSP and SASA covariates, directed
  hydrogen-bond evidence and energies, each interaction label, each validity
  field, and an exclusion reason.
- `annotations/dssp_raw/<protein_id>.dssp`: exact classic DSSP output used for
  parsing.
- `annotations/annotation_counts.csv`: one row per protein, interaction type,
  and count category (`positive`, `valid_negative`, or `invalid`).
- `annotations/exclusions.csv`: protein-level input or DSSP failures with the
  affected annotation scope.
- `summary.txt`: requested/successful/excluded protein counts, total residues
  and pairs, interaction counts, DSSP/configuration provenance, failures, and
  runtime.

The per-pair interface deliberately preserves both ordinary structural
covariates and labels. Downstream matching can therefore control for amino-acid
identity, distance, sequence separation, secondary structure, and exposure
without recomputing annotations.

## Validation and acceptance

Local tests cover cutoff boundaries, ARG `NE` exclusion, missing-atom validity,
noncandidate negatives, hydrophobic-core validity propagation, and DSSP
three-state collapsing. The script and sbatch files receive syntax checks
locally using `./.venv`.

The Delta acceptance run should additionally verify:

- DSSP 4.6.1 is reported by `$HOME/dssp_env/bin/mkdssp`;
- the manifest has exactly 50 requested proteins;
- every successful pair file has unique, ordered pair IDs and exactly
  `R(R-1)/2` rows with finite C-alpha distances;
- interaction implications and nullable validity fields are internally
  consistent;
- `annotation_counts.csv` exactly reconciles with the per-pair files;
- DSSP failures preserve geometric and salt-bridge annotations while marking
  the DSSP-dependent fields invalid.

Once this prototype passes acceptance, the next stage may consume these cached
labels for matched negatives and protein-level classification splits. It should
not modify the annotation definitions in place; a definition change requires a
new annotation version and output root.

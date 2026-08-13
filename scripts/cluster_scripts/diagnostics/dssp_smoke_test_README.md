# DSSP final-PDB smoke test

This is a minimal one-structure check for the `mkdssp` installation and the
final OpenFold-predicted PDB layout. It defaults to CAMEO protein `6tf4_A`.

First install DSSP once on Delta, if needed:

```bash
sbatch setup_dssp_delta.sbatch
```

Then submit the smoke test from this directory:

```bash
sbatch dssp_smoke_test_delta.sbatch
```

Set `DSSP_SAMPLE_PROTEIN` in the launcher to inspect another structure. Numeric
IDs use the ECOD final-PDB root; other IDs use CAMEO.

The Python script is safe to run from a flat Delta working directory; it no
longer assumes a fixed number of parent directories above the script. In that
layout the launcher-provided PDB and output environment variables are used.

The job writes `$HOME/csv_files/dssp_smoke_test/<protein-id>/`:

- `summary.json`: DSSP version, mapped-residue count, and compact structural counts.
- `prepared_input.pdb`: structural-only temporary input passed to `mkdssp`,
  retained to make parser failures reproducible.
- `raw.dssp`: direct `mkdssp` output for manual inspection.
- `residues.csv`: one row per polymer residue with DSSP code, simplified class,
  solvent-accessible surface area, and phi/psi.
- `secondary_structure_segments.csv`: contiguous helix, beta-strand, and
  isolated beta-bridge ranges. DSSP reports beta strands/bridges, rather than a
  globally named beta-sheet object.
- `disulfide_candidates.csv`: CYS SG--SG pairs at most 2.3 Å apart. These are
  coordinate-based candidates, not confirmed chemical disulfide bonds.
- `mkdssp.stderr.txt`: retained error details if DSSP cannot process the PDB.

The script adds a large P1 `CRYST1` record when the prediction lacks one and
keeps only coordinate/model records in the temporary DSSP input. This removes
OpenFold's nonstandard `REMARK` and `PARENT` metadata without modifying the
original PDB.

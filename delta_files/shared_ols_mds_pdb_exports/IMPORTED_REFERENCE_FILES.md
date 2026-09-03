# Imported reference files

These files were copied from the `aug-16-archive` branch because two parts remain
useful to the current shared-OLS MDS analysis:

- `aug-16-archive/mds_structures/*.py` preserves the old protein-rendering and
  figure-composition code as a visual-workflow reference. These scripts were
  written for an older reconstruction pipeline, contain old paths and analysis
  choices, and are **not canonical code**. Do not run them unchanged or treat
  their inputs as current. Always use the most recent PDB exports and defer to
  newer analysis instructions.
- `aug-16-archive/reconstructed_pdb_files/general_base256_avg_symmetric/*/original_full/`
  contains only the original full-atom OpenFold predictions for the ten-protein
  visualization cohort. Use these PDBs as the final predicted OpenFold
  structures when comparing the current reconstructed structures with the
  original model outputs.

No historical layer reconstructions, reference-CA files, rendered images, or
saved visualization sessions were imported.

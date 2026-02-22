#!/usr/bin/env python3
"""
extract_coordinates.py
======================
Extract 3D coordinates of amino acid residues from an mmCIF structure file.

Uses the alpha-carbon (CA) atom as the representative position for each residue.
Outputs two CSVs per protein:
  - residue_coordinates.csv   — one row per residue, columns: chain, name, number, x, y, z
  - residue_distances.csv     — NxN matrix of pairwise Euclidean (CA-CA) distances in Angstroms

---------------------------------------------------------------------------
BIOLOGY BACKGROUND
---------------------------------------------------------------------------
Proteins are chains of amino acids linked by peptide bonds. Each amino acid
has a backbone consisting of three atoms in sequence:

    N — CA — C(=O) — [next N — CA — C ...]

  • N  = nitrogen (amino group end)
  • CA = alpha-carbon (the central carbon bonded to the side chain)
  • C  = carbonyl carbon (part of the peptide bond to the next residue)

The alpha-carbon (CA) is the standard "representative" atom for a residue
because:
  1. Every standard amino acid has exactly one CA.
  2. Its position captures the backbone geometry well.
  3. It lies at the branch point between the backbone and the side chain (R group),
     making it a natural centre-of-mass proxy for the residue.

Pairwise CA-CA distances are the most common way to define whether two
residues are spatially "close" in 3D space (e.g., ≤ 8 Å is a typical
threshold for "in contact").

Structure files (CIF / PDB) encode the experimentally determined (or
computationally predicted) 3D positions of every atom in the protein.

mmCIF (macromolecular Crystallographic Information File):
  The modern, machine-readable format used by the Protein Data Bank (PDB).
  Supersedes the older fixed-column PDB format. Contains atom coordinates,
  metadata, symmetry information, and quality metrics.

NMR structures deposit multiple "models" (conformational snapshots). X-ray
and cryo-EM structures typically have a single model. We always use model 0
(the first/representative model) for consistency.

---------------------------------------------------------------------------
LIBRARIES USED
---------------------------------------------------------------------------

sys
  Part of the Python standard library. Used here only for sys.exit(), which
  terminates the process with a non-zero exit code when a required dependency
  is missing.

pathlib.Path
  Standard-library object-oriented path handling. Cleaner than os.path for
  constructing and querying filesystem paths; supports / operator for joining.

Biopython (Bio.PDB)
  A mature bioinformatics library for parsing, manipulating, and analyzing
  biological sequence and structure data.

  Bio.PDB.MMCIFParser
    Reads an mmCIF file into Biopython's SMCRA hierarchy:
      Structure → Model → Chain → Residue → Atom
    QUIET=True suppresses non-critical parse warnings (common for legacy CIF
    quirks that don't affect coordinate extraction).

numpy (np)
  Foundational numerical computing library.
  - np.array()          : converts a Python list to an efficient ndarray.
  - np.zeros()          : allocates a zero-filled matrix.
  - np.linalg.norm()    : computes the L2 (Euclidean) norm of a vector,
                          i.e., the straight-line distance between two points
                          in 3D space: sqrt((x2-x1)^2 + (y2-y1)^2 + (z2-z1)^2)
  - np.savetxt()        : writes an ndarray to a text/CSV file with optional
                          header, delimiter, and format string.
"""

import sys
from pathlib import Path

# Add project root to sys.path so lib/ is importable regardless of cwd.
_PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(_PROJECT_ROOT))

try:
    from lib.structure import extract_residue_coordinates, compute_distance_matrix, detect_chain
except ImportError as e:
    print(f"Could not import from lib.structure: {e}")
    sys.exit(1)

import numpy as np


# =============================================================================
# FUNCTION: process_protein
# =============================================================================
def process_protein(protein_dir: Path) -> None:
    """
    Orchestrate extraction and distance computation for a single protein
    directory, then write results to CSV files.

    Expected protein_dir layout (before running this script):
        protein_dir/
        ├── <pdb_id>.cif          ← structure file (required)
        └── pair_blocks/           ← used to infer the chain letter

    After running, this function creates:
        protein_dir/
        └── csv_files/
            ├── residue_coordinates.csv   ← CA positions
            └── residue_distances.csv     ← NxN distance matrix

    Parameters
    ----------
    protein_dir : Path
        Root directory for one protein (e.g., proteins/7b3a/).
    """

    # --- Locate the CIF file ---
    # Glob for both lowercase and uppercase extensions for robustness.
    cif_files = list(protein_dir.glob("*.cif")) + list(protein_dir.glob("*.CIF"))
    if not cif_files:
        print(f"  ERROR: no .cif file found in {protein_dir}, skipping.")
        return
    # Take the first match (there should only be one CIF per protein directory).
    cif_file = cif_files[0]

    # Infer the chain letter from pair_block filenames.
    chain = detect_chain(protein_dir)
    print(f"  Chain: {chain if chain else 'all (not detected)'}")

    # Ensure the csv_files/ output directory exists.
    # mkdir(parents=True) creates any missing parent directories.
    # exist_ok=True prevents an error if the directory already exists.
    csv_dir = protein_dir / "csv_files"
    csv_dir.mkdir(parents=True, exist_ok=True)

    # --- Parse the CIF and extract CA coordinates ---
    print(f"  Parsing {cif_file.name} (model 0)...")
    data = extract_residue_coordinates(str(cif_file), chain_id=chain)
    residues = data["residues"]      # list of (chain_id, res_name, res_num)
    coords = data["coordinates"]     # np.ndarray of shape (n_residues, 3)
    print(f"  Found {len(residues)} amino acid residues")

    # --- Compute the pairwise CA-CA distance matrix ---
    dist_matrix = compute_distance_matrix(coords)

    # --- Save distance matrix as CSV ---
    output_file = csv_dir / "residue_distances.csv"

    # Build the header row: one label per residue in the format
    #   <chain>_<3-letter-name><residue-number>  e.g. "A_SER1"
    # This header acts as both row and column label for the square matrix.
    header = ",".join([f"{r[0]}_{r[1]}{r[2]}" for r in residues])

    # np.savetxt writes the 2D array to a text file.
    #   delimiter="," → CSV format
    #   header=header  → first line is the residue-label row
    #   fmt="%.3f"     → three decimal places (sub-Angstrom precision is ample)
    #   comments=""    → suppress numpy's default '#' comment character on the header
    np.savetxt(output_file, dist_matrix, delimiter=",",
               header=header, fmt="%.3f", comments="")
    print(f"  Distance matrix saved to: {output_file}")

    # --- Save coordinates as CSV ---
    coord_file = csv_dir / "residue_coordinates.csv"
    with open(coord_file, "w") as f:
        # Write the column header.
        f.write("chain,residue,number,x,y,z\n")
        # zip() pairs each residue metadata tuple with its coordinate vector,
        # iterating both lists in lockstep.
        for (chain, name, num), (x, y, z) in zip(residues, coords):
            # :.3f formats each coordinate to 3 decimal places (0.001 Å = 0.1 pm precision).
            f.write(f"{chain},{name},{num},{x:.3f},{y:.3f},{z:.3f}\n")
    print(f"  Coordinates saved to: {coord_file}")

    # --- Print summary statistics for the distance matrix ---
    # Exclude zero values (the diagonal, where i==j) when computing min/mean
    # to avoid the self-distance of 0 pulling those statistics down.
    print(f"  Stats — min: {dist_matrix[dist_matrix > 0].min():.2f} Å, "
          f"max: {dist_matrix.max():.2f} Å, "
          f"mean: {dist_matrix[dist_matrix > 0].mean():.2f} Å")


# =============================================================================
# FUNCTION: main
# =============================================================================
def main():
    """
    Entry point. Discovers all protein subdirectories under proteins/ and
    calls process_protein() on each one.

    Directory layout expected:
        protein/              ← project root (parent of this script's parent)
        └── proteins/
            ├── 7b3a/         ← one directory per protein
            └── 6tf4/         ← another protein, if present

    Path computation:
        __file__            → .../protein/scripts/extract_coordinates.py
        Path(__file__).parent        → .../protein/scripts/
        Path(__file__).parent.parent → .../protein/          (project root)
        .../protein/proteins         → the proteins/ directory
    """
    # Compute the path to the proteins/ directory relative to this script.
    proteins_root = Path(__file__).parent.parent / "proteins"

    # List all immediate subdirectories inside proteins/, sorted alphabetically.
    # iterdir() yields all entries (files and dirs); we keep only directories.
    protein_dirs = sorted([p for p in proteins_root.iterdir() if p.is_dir()])

    if not protein_dirs:
        print(f"No protein directories found under {proteins_root}")
        sys.exit(1)

    print(f"Found {len(protein_dirs)} protein(s): {[p.name for p in protein_dirs]}\n")

    # Process each protein directory in turn.
    for protein_dir in protein_dirs:
        print(f"[{protein_dir.name}]")
        process_protein(protein_dir)
        print()   # blank line between proteins for readability


# Standard Python idiom: only run main() when the script is executed directly,
# not when it is imported as a module by another script.
if __name__ == "__main__":
    main()

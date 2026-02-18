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

# --- Biopython dependency check ---
# Biopython is not part of the standard library; provide a helpful error
# message rather than a cryptic ImportError if it's absent.
try:
    from Bio.PDB import MMCIFParser
    # MMCIFParser: parses the mmCIF format used by the Protein Data Bank.
    # It returns a Biopython Structure object organised as:
    #   Structure
    #   └── Model(s)   — index 0 is the first (representative) model
    #       └── Chain(s)  — identified by a letter, e.g. 'A', 'B'
    #           └── Residue(s)  — amino acids, water, ligands, etc.
    #               └── Atom(s) — N, CA, C, O, side-chain atoms, etc.

    from Bio.PDB.Polypeptide import is_aa
    # is_aa(residue, standard=True):
    #   Checks whether a Residue object is a standard amino acid.
    #   Returns False for water molecules (HOH), metal ions, small-molecule
    #   ligands, and non-standard/modified residues.
    #   We use standard=True so that only the canonical 20 amino acids pass.

except ImportError:
    print("Biopython is required. Install with: pip install biopython")
    sys.exit(1)

import numpy as np


# =============================================================================
# FUNCTION: extract_residue_coordinates
# =============================================================================
def extract_residue_coordinates(cif_file: str, chain_id: str = None) -> dict:
    """
    Parse an mmCIF structure file and extract the 3D position of the
    alpha-carbon (CA) for every standard amino acid residue.

    Parameters
    ----------
    cif_file : str
        Absolute or relative path to the .cif structure file.
    chain_id : str or None
        If given, only residues belonging to this chain are returned
        (e.g., 'A'). If None, residues from ALL chains are returned.

    Returns
    -------
    dict with two keys:
      'residues'    : list of (chain_id, residue_name, residue_number) tuples
                      e.g., ('A', 'SER', 1)
      'coordinates' : numpy array of shape (n_residues, 3)
                      Each row is [x, y, z] in Angstroms for the CA atom.

    Biology note
    ------------
    We use CA-only coordinates rather than all-atom representations because:
      • They are simpler (one point per residue vs ~10 per residue).
      • They capture backbone topology reliably.
      • CA-CA distances correlate well with whether residues interact.
      • AlphaFold's pair representations are indexed per residue, not per atom,
        so a per-residue coordinate is the natural counterpart.
    """

    # Instantiate the mmCIF parser.
    # QUIET=True suppresses verbose warnings about mmCIF quirks that are
    # harmless for coordinate extraction (e.g., missing SEQRES records).
    parser = MMCIFParser(QUIET=True)

    # get_structure(id, filename):
    #   Parses the file and returns a Biopython Structure object.
    #   The first argument ("protein") is an arbitrary identifier string
    #   stored internally; it does not affect parsing.
    structure = parser.get_structure("protein", cif_file)

    # Initialise collectors for residue metadata and 3D positions.
    residues = []      # list of (chain_id, residue_name, residue_number)
    coordinates = []   # list of [x, y, z] Angstrom vectors

    # --- Model selection ---
    # structure[0] accesses the first model (index 0).
    # X-ray / cryo-EM files have exactly one model; NMR deposits many.
    # Using model 0 universally gives a single consistent conformation.
    model = structure[0]

    # --- Iterate over every chain in the model ---
    for chain in model:
        # chain.get_id() returns the single-letter chain identifier, e.g. 'A'.
        # Skip chains that don't match the requested chain_id (if specified).
        if chain_id is not None and chain.get_id() != chain_id:
            continue

        # --- Iterate over every residue in the chain ---
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue

            # --- Extract the alpha-carbon atom ---
            # In Biopython, residue["CA"] retrieves the Atom object named "CA".
            # Some residues (rare edge cases or truncated structures) may be
            # missing their CA; the "if 'CA' in residue" guard skips those.
            if "CA" in residue:
                ca = residue["CA"]
                # get_coord() returns a numpy array [x, y, z] in Angstroms.
                # Angstrom (Å) = 1×10⁻¹⁰ metres. Typical bond lengths are
                # ~1.5 Å; typical protein diameters are 30–100 Å.
                coord = ca.get_coord()

                # get_resname() returns the 3-letter residue code, e.g. 'SER'.
                res_name = residue.get_resname()

                # get_id() returns a tuple: (hetfield, sequence_number, icode).
                # Index [1] is the integer residue sequence number from the CIF.
                # For 7B3A chain A this ranges from 1 to 276 (resolved region).
                res_num = residue.get_id()[1]

                # Store the metadata tuple and the coordinate vector.
                residues.append((chain.get_id(), res_name, res_num))
                coordinates.append(coord)

    return {
        "residues": residues,
        # np.array() converts the Python list of [x,y,z] vectors into a
        # 2D numpy array of shape (n_residues, 3) — each row is one residue.
        "coordinates": np.array(coordinates)
    }


# =============================================================================
# FUNCTION: compute_distance_matrix
# =============================================================================
def compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """
    Compute the symmetric pairwise Euclidean distance matrix between all
    residue CA positions.

    Parameters
    ----------
    coordinates : np.ndarray, shape (n, 3)
        CA atom positions in Angstroms (one row per residue).

    Returns
    -------
    dist_matrix : np.ndarray, shape (n, n)
        dist_matrix[i, j] = Euclidean distance between residue i and j (Å).
        The diagonal is 0 (distance of a residue from itself).
        The matrix is symmetric: dist_matrix[i,j] == dist_matrix[j,i].

    Biology note
    ------------
    CA-CA distances encode spatial proximity in the folded protein:
      •  ≤ 8 Å  → residues are typically "in contact" (side chains can interact).
      • 3.8 Å   → the distance between sequential CA atoms along the backbone
                   (characteristic of an extended strand or helix).
      • Pairs with small spatial distance but large sequence separation (|i-j|)
        are formed by long-range contacts that define tertiary structure.

    Algorithm note
    --------------
    We iterate only the upper triangle (j > i) because the matrix is symmetric,
    then fill both [i,j] and [j,i] in one step — halving the work compared to
    a naive double loop. For 276 residues this is 37,950 pair evaluations.
    """
    n = len(coordinates)

    # Allocate an n×n zero matrix. The diagonal remains 0 (self-distance).
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):  # j > i → upper triangle only
            dist = np.linalg.norm(coordinates[i] - coordinates[j])

            # Fill both the upper and lower triangle symmetrically.
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


# =============================================================================
# FUNCTION: detect_chain
# =============================================================================
def detect_chain(protein_dir: Path) -> str:
    """
    Infer the relevant chain letter from the naming convention of pair block
    files found in the protein's pair_blocks/ subdirectory.

    Pair block filenames follow the pattern:
        <pdb_id>_<chain>_pair_block_<n>.npy
    e.g.:
        7b3a_A_pair_block_0.npy

    Parameters
    ----------
    protein_dir : Path
        Root directory for a single protein (e.g., proteins/7b3a/).

    Returns
    -------
    str or None
        Single uppercase chain letter, e.g. 'A'.
        Returns None if pair_blocks/ doesn't exist or no .npy files are found.

    Why this matters
    ----------------
    CIF files often contain multiple chains (biological assembly subunits,
    crystal packing partners). AlphaFold pair representations are computed for
    a single chain, so we must restrict coordinate extraction to that chain to
    keep indices aligned between the pair blocks and the distance matrix.
    """
    pb_dir = protein_dir / "pair_blocks"
    if not pb_dir.exists():
        return None

    # Glob all .npy files in the pair_blocks directory.
    npy_files = list(pb_dir.glob("*.npy"))
    if not npy_files:
        return None

    # Take the first file (arbitrary — all files share the same protein/chain).
    # .stem strips the '.npy' extension, giving e.g. '7b3a_A_pair_block_0'.
    # .split("_") splits on underscores → ['7b3a', 'A', 'pair', 'block', '0'].
    parts = npy_files[0].stem.split("_")

    # parts[1] is the chain letter when the PDB ID contains no underscores.
    # We validate that it is a single alphabetic character (chain IDs are
    # always a single letter in standard PDB convention).
    if len(parts) >= 2 and len(parts[1]) == 1 and parts[1].isalpha():
        return parts[1].upper()

    return None


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

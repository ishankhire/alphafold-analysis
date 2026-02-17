#!/usr/bin/env python3
"""
Extract 3D coordinates of amino acid residues from a mmCIF file.
Uses CA (alpha carbon) atoms as the representative position for each residue.
"""

import sys
from pathlib import Path

try:
    from Bio.PDB import MMCIFParser
    from Bio.PDB.Polypeptide import is_aa
except ImportError:
    print("Biopython is required. Install with: pip install biopython")
    sys.exit(1)

import numpy as np


def extract_residue_coordinates(cif_file: str) -> dict:
    """
    Extract CA coordinates for each residue in the structure.

    Returns a dict with:
        - 'residues': list of (chain_id, residue_name, residue_number)
        - 'coordinates': numpy array of shape (n_residues, 3)
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_file)

    residues = []
    coordinates = []

    for model in structure:
        for chain in model:
            for residue in chain:
                # Skip non-amino acid residues (water, ligands, etc.)
                if not is_aa(residue, standard=True):
                    continue

                # Get CA atom coordinates
                if "CA" in residue:
                    ca = residue["CA"]
                    coord = ca.get_coord()

                    res_name = residue.get_resname()
                    res_num = residue.get_id()[1]
                    chain_id = chain.get_id()

                    residues.append((chain_id, res_name, res_num))
                    coordinates.append(coord)

    return {
        "residues": residues,
        "coordinates": np.array(coordinates)
    }


def compute_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """Compute pairwise distance matrix between all residues."""
    n = len(coordinates)
    dist_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(i + 1, n):
            dist = np.linalg.norm(coordinates[i] - coordinates[j])
            dist_matrix[i, j] = dist
            dist_matrix[j, i] = dist

    return dist_matrix


def main():
    PROTEIN_DIR = Path(__file__).parent.parent / "proteins" / "7b3a"
    cif_file = PROTEIN_DIR / "7b3a.cif"
    csv_dir = PROTEIN_DIR / "csv_files"
    csv_dir.mkdir(parents=True, exist_ok=True)

    if not cif_file.exists():
        print(f"Error: {cif_file} not found")
        sys.exit(1)

    print(f"Parsing {cif_file}...")
    data = extract_residue_coordinates(str(cif_file))

    residues = data["residues"]
    coords = data["coordinates"]

    print(f"\nFound {len(residues)} amino acid residues\n")

    # Print coordinates for each residue
    print("Residue Coordinates (CA atoms):")
    print("-" * 60)
    print(f"{'Chain':<6} {'Residue':<8} {'#':<6} {'X':>10} {'Y':>10} {'Z':>10}")
    print("-" * 60)

    for (chain, name, num), (x, y, z) in zip(residues, coords):
        print(f"{chain:<6} {name:<8} {num:<6} {x:>10.3f} {y:>10.3f} {z:>10.3f}")

    # Compute and save distance matrix
    print("\n" + "=" * 60)
    print("Computing pairwise distance matrix...")
    dist_matrix = compute_distance_matrix(coords)

    # Save to CSV
    output_file = csv_dir / "residue_distances.csv"

    # Create header with residue labels
    header = ",".join([f"{r[0]}_{r[1]}{r[2]}" for r in residues])
    np.savetxt(output_file, dist_matrix, delimiter=",",
               header=header, fmt="%.3f", comments="")

    print(f"Distance matrix saved to: {output_file}")

    # Also save raw coordinates
    coord_file = csv_dir / "residue_coordinates.csv"
    with open(coord_file, "w") as f:
        f.write("chain,residue,number,x,y,z\n")
        for (chain, name, num), (x, y, z) in zip(residues, coords):
            f.write(f"{chain},{name},{num},{x:.3f},{y:.3f},{z:.3f}\n")

    print(f"Coordinates saved to: {coord_file}")

    # Print some statistics
    print("\n" + "=" * 60)
    print("Distance Statistics:")
    print(f"  Min distance between residues: {dist_matrix[dist_matrix > 0].min():.2f} Å")
    print(f"  Max distance between residues: {dist_matrix.max():.2f} Å")
    print(f"  Mean distance: {dist_matrix[dist_matrix > 0].mean():.2f} Å")


if __name__ == "__main__":
    main()

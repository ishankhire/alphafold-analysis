"""
lib/structure.py
================
Shared utilities for extracting CA coordinates from mmCIF files and computing
pairwise distance matrices. Migrated verbatim from scripts/extract_coordinates.py
so that multi-protein scripts can import them without duplicating code.
"""

from pathlib import Path
import numpy as np
from Bio.PDB import MMCIFParser, PDBParser
from Bio.PDB.Polypeptide import is_aa


def extract_residue_coordinates(cif_file: str, chain_id: str = None) -> dict:
    """Parse an mmCIF structure file and extract the 3D position of the alpha-carbon
    (CA) for every standard amino acid residue.

    Parameters
    ----------
    cif_file : str
        Absolute or relative path to the .cif structure file.
    chain_id : str or None
        If given, only residues belonging to this chain are returned (e.g. 'A').
        If None, residues from ALL chains are returned.

    Returns
    -------
    dict with two keys:
      'residues'    : list of (chain_id, residue_name, residue_number) tuples
      'coordinates' : numpy array of shape (n_residues, 3)
    """
    parser = MMCIFParser(QUIET=True)
    structure = parser.get_structure("protein", cif_file)

    residues = []
    coordinates = []

    model = structure[0]

    # Known modified amino acids that have CA atoms and should be included
    MODIFIED_AA = {"MSE"}  # selenomethionine

    for chain in model:
        if chain_id is not None and chain.get_id() != chain_id:
            continue

        for residue in chain:
            if not is_aa(residue, standard=True):
                if residue.get_resname() not in MODIFIED_AA:
                    continue

            if "CA" in residue:
                ca = residue["CA"]
                coord = ca.get_coord()
                res_name = residue.get_resname()
                res_num = residue.get_id()[1]

                residues.append((chain.get_id(), res_name, res_num))
                coordinates.append(coord)

    return {
        "residues": residues,
        "coordinates": np.array(coordinates),
    }


def extract_ca_coordinates(pdb_path):
    """Extract CA atom coordinates from a PDB file across all chains.

    Skips water molecules (het_flag == "W") and extracts all residues
    with a CA atom. Chain filtering is unnecessary because each PDB file
    is already specific to a single protein.

    Parameters
    ----------
    pdb_path : str or Path
        Path to the .pdb structure file.

    Returns
    -------
    dict with two keys:
      'residues'    : list of (chain_id, residue_name, residue_number) tuples
      'coordinates' : numpy array of shape (n_residues, 3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", str(pdb_path))

    residues = []
    coordinates = []

    model = structure[0]
    for chain in model:
        for residue in chain:
            het_flag = residue.get_id()[0]
            if het_flag == "W":
                continue
            if "CA" in residue:
                ca = residue["CA"]
                residues.append((
                    chain.get_id(),
                    residue.get_resname(),
                    residue.get_id()[1],
                ))
                coordinates.append(ca.get_coord())

    return {
        "residues": residues,
        "coordinates": np.array(coordinates) if coordinates else np.empty((0, 3)),
    }


def extract_pdb_ca_coordinates(pdb_file: str) -> dict:
    """Parse a PDB file and extract the 3D position of the alpha-carbon (CA)
    for every standard amino acid residue across all chains.

    Chain filtering is unnecessary because each PDB file is already
    specific to a single protein.

    Parameters
    ----------
    pdb_file : str
        Path to the .pdb structure file.

    Returns
    -------
    dict with two keys:
      'residues'    : list of (chain_id, residue_name, residue_number) tuples
      'coordinates' : numpy array of shape (n_residues, 3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", pdb_file)

    residues = []
    coordinates = []

    model = structure[0]
    for chain in model:
        for residue in chain:
            if not is_aa(residue, standard=True):
                continue
            if "CA" in residue:
                ca = residue["CA"]
                residues.append((chain.get_id(), residue.get_resname(), residue.get_id()[1]))
                coordinates.append(ca.get_coord())

    return {
        "residues": residues,
        "coordinates": np.array(coordinates),
    }


def superimpose_structures(fixed_coords: np.ndarray, moving_coords: np.ndarray):
    """Superimpose moving_coords onto fixed_coords using the Kabsch algorithm.

    Parameters
    ----------
    fixed_coords : np.ndarray, shape (n, 3)
        Reference coordinates (ground truth).
    moving_coords : np.ndarray, shape (n, 3)
        Coordinates to transform (predicted). Must have same length.

    Returns
    -------
    transformed : np.ndarray, shape (n, 3)
        moving_coords after optimal rotation + translation.
    rmsd : float
        Root mean square deviation after superposition.
    """
    fixed_center = fixed_coords.mean(axis=0)
    moving_center = moving_coords.mean(axis=0)
    fixed_centered = fixed_coords - fixed_center
    moving_centered = moving_coords - moving_center

    H = moving_centered.T @ fixed_centered
    U, S, Vt = np.linalg.svd(H)

    d = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(d)])
    R = Vt.T @ sign_matrix @ U.T

    transformed = (moving_centered @ R.T) + fixed_center

    diff = fixed_coords - transformed
    rmsd = np.sqrt(np.mean(np.sum(diff**2, axis=1)))

    return transformed, rmsd


def compute_tm_score(fixed_coords: np.ndarray, transformed_coords: np.ndarray, L_target: int) -> float:
    """Compute TM-score between two superimposed structures.

    TM = (1/L_target) * sum [ 1 / (1 + (d_i / d_0)^2) ]
    where d_0 = 1.24 * (L_target - 15)^(1/3) - 1.8

    Parameters
    ----------
    fixed_coords, transformed_coords : np.ndarray, shape (n, 3)
        Aligned coordinate arrays (after superposition).
    L_target : int
        Length of the target protein (number of resolved residues).

    Returns
    -------
    float : TM-score in [0, 1].
    """
    d0 = 1.24 * (max(L_target, 16) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)

    distances = np.sqrt(np.sum((fixed_coords - transformed_coords) ** 2, axis=1))
    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    return float(np.sum(scores) / L_target)


def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Vectorized pairwise Euclidean distance matrix.

    Parameters
    ----------
    coords : np.ndarray, shape (n, 3)
        CA atom positions in Angstroms.

    Returns
    -------
    np.ndarray, shape (n, n)
        Symmetric distance matrix; diagonal is 0.
    """
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n, n, 3)
    return np.linalg.norm(diff, axis=2)                          # (n, n)


def load_distance_matrix(csv_path) -> np.ndarray:
    """Load a pre-computed residue_distances.csv file.

    Parameters
    ----------
    csv_path : str or Path
        Path to a CSV whose first row is a residue-label header and remaining
        rows form the square distance matrix.

    Returns
    -------
    np.ndarray, shape (n, n)
    """
    return np.genfromtxt(csv_path, delimiter=',', skip_header=1)


def detect_chain(protein_dir: Path) -> str:
    """Infer the relevant chain letter from pair-block filenames.

    Pair block filenames follow the pattern: <pdb_id>_<chain>_pair_block_<n>.npy

    Parameters
    ----------
    protein_dir : Path
        Root directory for a single protein (e.g. proteins/7b3a/).

    Returns
    -------
    str or None
        Single uppercase chain letter, e.g. 'A'. Returns None if the
        pair_blocks/ subdirectory doesn't exist or contains no .npy files.
    """
    pb_dir = protein_dir / "pair_blocks"
    if not pb_dir.exists():
        return None

    npy_files = list(pb_dir.glob("*.npy"))
    if not npy_files:
        return None

    parts = npy_files[0].stem.split("_")
    if len(parts) >= 2 and len(parts[1]) == 1 and parts[1].isalpha():
        return parts[1].upper()

    return None

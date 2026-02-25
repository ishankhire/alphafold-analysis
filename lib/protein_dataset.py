"""
Shared helpers for local multi-protein PDB/pair-block datasets.
"""

from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

from .pair_features import upper_triangle_indices, pair_targets_from_distance_matrix
from .classification_common import build_pair_type_labels


DEFAULT_SKIP_NAMES = {
    "multi_protein_results",
    "reference_for_llms",
    "random_files",
}


def discover_local_proteins(proteins_dir, skip_names=None):
    """Return sorted (protein_id, protein_dir) tuples under proteins_dir."""
    if skip_names is None:
        skip_names = DEFAULT_SKIP_NAMES

    proteins = []
    for entry in sorted(Path(proteins_dir).iterdir()):
        if not entry.is_dir():
            continue
        if entry.name in skip_names:
            continue
        proteins.append((entry.name, entry))
    return proteins


def find_local_pdb_file(protein_dir, protein_id):
    """Return predicted PDB path if present, else None."""
    pdb_path = Path(protein_dir) / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    return pdb_path if pdb_path.exists() else None


def get_local_pair_block_path(protein_dir, protein_id, layer):
    """Return local pair block path for a layer."""
    return Path(protein_dir) / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy"


def extract_ca_coordinates_from_pdb(pdb_path):
    """Extract CA coordinates across all chains from a predicted PDB."""
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
                residues.append((chain.get_id(), residue.get_resname(), residue.get_id()[1]))
                coordinates.append(ca.get_coord())

    return {
        "residues": residues,
        "coordinates": np.array(coordinates) if coordinates else np.empty((0, 3)),
    }


def compute_pairwise_distances(coords):
    """Compute a vectorized pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def _validate_pair_block_dim(protein_dir, protein_id, n_ca):
    """Validate layer-0 pair-block dimension matches n_ca."""
    pb0_path = get_local_pair_block_path(protein_dir, protein_id, 0)
    if not pb0_path.exists():
        return False, "pair block 0 not found", None

    pb0 = np.load(str(pb0_path), mmap_mode="r")
    pb_dim = int(pb0.shape[0])
    del pb0

    if n_ca != pb_dim:
        return False, f"CA count {n_ca} != pair block dim {pb_dim}", None
    return True, None, pb0_path


def load_local_protein_data(
    proteins_dir,
    include_distances=False,
    include_pair_labels=False,
    skip_names=None,
):
    """Load validated local proteins for pooled multi-protein analyses.

    Returns (protein_data, skipped) where:
      protein_data is a list of dicts with shared fields:
        protein_id, protein_dir, residues, coordinates, n_residues, pb_i, pb_j
      include_distances=True adds:
        distance_matrix, y
      include_pair_labels=True adds:
        labels
      skipped is a list of (protein_id, reason) tuples.
    """
    all_proteins = discover_local_proteins(proteins_dir, skip_names=skip_names)
    protein_data = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        try:
            pdb_path = find_local_pdb_file(protein_dir, protein_id)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                continue

            ca_data = extract_ca_coordinates_from_pdb(pdb_path)
            residues = ca_data["residues"]
            coordinates = ca_data["coordinates"]
            n_ca = len(residues)
            if n_ca == 0:
                skipped.append((protein_id, "no CA atoms found in PDB"))
                continue

            is_valid, reason, _ = _validate_pair_block_dim(protein_dir, protein_id, n_ca)
            if not is_valid:
                skipped.append((protein_id, reason))
                continue

            pb_i, pb_j = upper_triangle_indices(n_ca)
            row = {
                "protein_id": protein_id,
                "protein_dir": protein_dir,
                "residues": residues,
                "coordinates": coordinates,
                "n_residues": n_ca,
                "pb_i": pb_i,
                "pb_j": pb_j,
            }

            if include_distances:
                dist_matrix = compute_pairwise_distances(coordinates)
                row["distance_matrix"] = dist_matrix
                row["y"] = pair_targets_from_distance_matrix(dist_matrix, pb_i, pb_j)

            if include_pair_labels:
                row["labels"] = build_pair_type_labels(ca_data, pb_i, pb_j)

            protein_data.append(row)
        except Exception as exc:
            skipped.append((protein_id, f"unexpected error -- {exc}"))

    return protein_data, skipped

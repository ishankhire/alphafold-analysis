"""
lib/data_loading.py
===================
Shared utilities for discovering protein directories and locating
data files (PDB structures, pair block arrays).
"""

from pathlib import Path


def discover_proteins(proteins_dir):
    """Find all protein subdirectories under the given root.

    Returns a sorted list of (protein_id, protein_dir) tuples.
    Skips 'multi_protein_results' and any non-directory entries.
    """
    proteins = []
    for entry in sorted(proteins_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "multi_protein_results" or entry.name == "llm_reference":
            continue
        proteins.append((entry.name, entry))
    return proteins


def find_pdb_file(protein_dir, protein_id):
    """Locate the predicted PDB file for a protein.

    Expected path: <protein_dir>/predicted_pdb/<protein_id>_model_1_ptm_unrelaxed.pdb

    Returns the Path object, or None if not found.
    """
    pdb_path = protein_dir / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    return pdb_path if pdb_path.exists() else None


def get_pair_block_path(protein_dir, protein_id, layer):
    """Return the path to a pair block .npy file for a given layer.

    Expected: <protein_dir>/pair_blocks/<protein_id>_pair_block_<layer>.npy
    """
    return protein_dir / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy"

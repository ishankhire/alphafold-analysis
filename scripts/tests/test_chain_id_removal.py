from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

from lib.protein_dataset import DEFAULT_SKIP_NAMES, discover_local_proteins, extract_ca_coordinates_from_pdb


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTEINS_DIR = PROJECT_ROOT / "proteins"


def extract_ca_old_chain_filtered(pdb_path, chain_id):
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", str(pdb_path))
    residues = []
    coordinates = []
    for chain in structure[0]:
        if chain.get_id() != chain_id:
            continue
        for residue in chain:
            if residue.get_id()[0] == "W":
                continue
            if "CA" in residue:
                residues.append((chain.get_id(), residue.get_resname(), residue.get_id()[1]))
                coordinates.append(residue["CA"].get_coord())
    return {
        "residues": residues,
        "coordinates": np.array(coordinates) if coordinates else np.empty((0, 3)),
    }


def test_chain_id_removal_matches_existing_local_proteins():
    proteins = discover_local_proteins(PROTEINS_DIR, skip_names=DEFAULT_SKIP_NAMES)
    assert proteins, "no proteins discovered"

    for protein_id, protein_dir in proteins:
        chain_id = protein_id.split("_")[-1]
        pdb_path = protein_dir / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
        assert pdb_path.exists(), f"missing pdb for {protein_id}: {pdb_path}"

        old = extract_ca_old_chain_filtered(pdb_path, chain_id)
        new = extract_ca_coordinates_from_pdb(pdb_path)

        assert len(old["residues"]) > 0, f"{protein_id}: old chain-filtered method returned zero residues"
        assert len(old["residues"]) == len(new["residues"]), f"{protein_id}: residue count mismatch"
        assert old["residues"] == new["residues"], f"{protein_id}: residue identity mismatch"
        assert np.allclose(old["coordinates"], new["coordinates"]), f"{protein_id}: coordinate mismatch"

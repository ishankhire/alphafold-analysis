from pathlib import Path

import numpy as np

from lib.protein_dataset import (
    compute_pairwise_distances,
    discover_local_proteins,
    find_local_pdb_file,
    get_local_pair_block_path,
    load_local_protein_data,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROTEINS_DIR = PROJECT_ROOT / "proteins"


def test_compute_pairwise_distances_shape():
    coords = np.array([[0, 0, 0], [3, 0, 0], [0, 4, 0]], dtype=float)
    dm = compute_pairwise_distances(coords)
    assert dm.shape == (3, 3)
    assert np.allclose(dm, dm.T)
    assert np.allclose(np.diag(dm), 0.0)


def test_local_path_helpers():
    proteins = discover_local_proteins(PROTEINS_DIR)
    assert proteins
    protein_id, protein_dir = proteins[0]
    assert find_local_pdb_file(protein_dir, protein_id) is not None
    assert get_local_pair_block_path(protein_dir, protein_id, 0).exists()


def test_load_local_protein_data_distance_and_labels():
    protein_data, skipped = load_local_protein_data(
        PROTEINS_DIR,
        include_distances=True,
        include_pair_labels=True,
    )
    assert protein_data
    for d in protein_data:
        assert len(d["pb_i"]) == len(d["pb_j"])
        assert len(d["y"]) == len(d["pb_i"])
        assert len(d["labels"]) == len(d["pb_i"])
    assert isinstance(skipped, list)

from pathlib import Path

import numpy as np

from lib.regression_common import (
    build_layer_feature_matrix,
    compute_protein_offsets,
    compute_protein_offsets_for_key,
    concatenate_labels,
    concatenate_targets,
    encode_labels,
    split_indices,
)


def test_concatenate_and_offsets():
    protein_data = [
        {"protein_id": "p1", "y": np.array([1.0, 2.0]), "labels": np.array(["A", "B"])},
        {"protein_id": "p2", "y": np.array([3.0]), "labels": np.array(["A"])},
    ]
    assert concatenate_targets(protein_data).tolist() == [1.0, 2.0, 3.0]
    assert concatenate_labels(protein_data).tolist() == ["A", "B", "A"]
    assert compute_protein_offsets(protein_data) == [("p1", 0, 2), ("p2", 2, 3)]
    assert compute_protein_offsets_for_key(protein_data, "labels") == [("p1", 0, 2), ("p2", 2, 3)]


def test_encode_labels_and_split_indices():
    y, enc = encode_labels(np.array(["x", "y", "x", "z"]))
    assert set(enc.classes_.tolist()) == {"x", "y", "z"}
    idx_train, idx_test = split_indices(10, test_size=0.2, random_state=42)
    assert len(idx_train) == 8
    assert len(idx_test) == 2


def test_build_layer_feature_matrix(tmp_path: Path):
    protein_dir = tmp_path / "p1"
    pair_dir = protein_dir / "pair_blocks"
    pair_dir.mkdir(parents=True)
    protein_id = "p1"
    pb = np.arange(3 * 3 * 2, dtype=np.float32).reshape(3, 3, 2)
    np.save(pair_dir / f"{protein_id}_pair_block_0.npy", pb)

    protein_data = [
        {
            "protein_id": protein_id,
            "protein_dir": protein_dir,
            "pb_i": np.array([0, 1]),
            "pb_j": np.array([1, 2]),
        }
    ]
    X = build_layer_feature_matrix(protein_data, layer=0, mode="base")
    assert X.shape == (2, 4)

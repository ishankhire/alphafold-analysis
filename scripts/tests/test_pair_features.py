import numpy as np
import pytest

from lib.pair_features import build_pair_features, pair_targets_from_distance_matrix, upper_triangle_indices


def test_upper_triangle_indices_size():
    i_idx, j_idx = upper_triangle_indices(5)
    assert len(i_idx) == 10
    assert len(j_idx) == 10


def test_build_pair_features_base_and_diagonal():
    pb = np.arange(4 * 4 * 2, dtype=np.float32).reshape(4, 4, 2)
    i_idx = np.array([0, 1])
    j_idx = np.array([2, 3])

    base = build_pair_features(pb, i_idx, j_idx, mode="base")
    diag = build_pair_features(pb, i_idx, j_idx, mode="diagonal")
    assert base.shape == (2, 4)
    assert diag.shape == (2, 8)
    assert np.array_equal(base[:, :2], pb[i_idx, j_idx, :])
    assert np.array_equal(base[:, 2:], pb[j_idx, i_idx, :])


def test_build_pair_features_invalid_mode():
    pb = np.zeros((2, 2, 2), dtype=np.float32)
    with pytest.raises(ValueError):
        build_pair_features(pb, np.array([0]), np.array([1]), mode="bad")


def test_pair_targets_from_distance_matrix():
    dm = np.array([[0.0, 1.0], [1.0, 0.0]])
    y = pair_targets_from_distance_matrix(dm, np.array([0]), np.array([1]))
    assert y.shape == (1,)
    assert y[0] == 1.0

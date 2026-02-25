"""
Feature construction helpers for pair-block analyses.
"""

import numpy as np


def upper_triangle_indices(n_residues):
    """Return upper-triangle pair indices (excluding diagonal)."""
    return np.triu_indices(n_residues, k=1)


def pair_targets_from_distance_matrix(distance_matrix, i_idx, j_idx):
    """Select target values for pair indices from a distance matrix."""
    return distance_matrix[i_idx, j_idx]


def build_pair_features(pair_block, i_idx, j_idx, mode="base"):
    """Construct per-pair features from a pair-block tensor.

    mode="base": [pb[i,j,:], pb[j,i,:]] -> 256 dims
    mode="diagonal": base + [pb[i,i,:], pb[j,j,:]] -> 512 dims
    """
    if mode == "base":
        return np.concatenate(
            [
                pair_block[i_idx, j_idx, :],
                pair_block[j_idx, i_idx, :],
            ],
            axis=1,
        )
    if mode == "diagonal":
        return np.concatenate(
            [
                pair_block[i_idx, j_idx, :],
                pair_block[j_idx, i_idx, :],
                pair_block[i_idx, i_idx, :],
                pair_block[j_idx, j_idx, :],
            ],
            axis=1,
        )
    raise ValueError(f"Unsupported feature mode: {mode}")

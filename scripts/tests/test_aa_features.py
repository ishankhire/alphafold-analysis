import numpy as np

from lib.aa_features import AA_FEATURE_DIM, N_AA, build_aa_onehot_features


def test_build_aa_onehot_features_symmetry():
    residue_data = {"residues": [("A", "GLY", 1), ("A", "ALA", 2), ("A", "SER", 3)]}
    struct_i = np.array([0, 1])
    struct_j = np.array([1, 0])

    feats = build_aa_onehot_features(residue_data, struct_i, struct_j)
    assert feats.shape == (2, AA_FEATURE_DIM)
    assert np.array_equal(feats[0], feats[1])


def test_build_aa_onehot_features_nonstandard_zeroed():
    residue_data = {"residues": [("A", "MSE", 1), ("A", "ALA", 2)]}
    feats = build_aa_onehot_features(residue_data, np.array([0]), np.array([1]))
    assert feats.shape == (1, AA_FEATURE_DIM)
    assert np.sum(feats[0, :N_AA]) == 1.0 or np.sum(feats[0, :N_AA]) == 0.0

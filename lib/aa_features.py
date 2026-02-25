"""
Shared amino-acid feature helpers.
"""

import numpy as np


STANDARD_AAS = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]
AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AAS)}
N_AA = len(STANDARD_AAS)
AA_FEATURE_DIM = 2 * N_AA


def build_aa_onehot_features(residue_data, struct_i, struct_j):
    """Build 40-dim one-hot AA features for residue pairs."""
    res_names = [r[1] for r in residue_data["residues"]]
    n_pairs = len(struct_i)
    aa_features = np.zeros((n_pairs, AA_FEATURE_DIM), dtype=np.float32)

    for k in range(n_pairs):
        aa1 = res_names[struct_i[k]]
        aa2 = res_names[struct_j[k]]
        sorted_pair = sorted([aa1, aa2])

        idx1 = AA_TO_INDEX.get(sorted_pair[0])
        idx2 = AA_TO_INDEX.get(sorted_pair[1])
        if idx1 is not None:
            aa_features[k, idx1] = 1.0
        if idx2 is not None:
            aa_features[k, N_AA + idx2] = 1.0

    return aa_features

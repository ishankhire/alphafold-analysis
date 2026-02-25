"""
Shared utilities for pooled multi-protein regression pipelines.
"""

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from .pair_features import build_pair_features
from .protein_dataset import get_local_pair_block_path


def concatenate_targets(protein_data):
    """Concatenate per-protein y arrays into one pooled target array."""
    return np.concatenate([d["y"] for d in protein_data])


def compute_protein_offsets(protein_data):
    """Return [(protein_id, start, end), ...] offsets in pooled arrays."""
    if not protein_data:
        return []
    key = "y" if "y" in protein_data[0] else "labels"
    return compute_protein_offsets_for_key(protein_data, key)


def compute_protein_offsets_for_key(protein_data, key):
    """Return [(protein_id, start, end), ...] offsets for a per-protein key."""
    offsets = []
    offset = 0
    for d in protein_data:
        n = len(d[key])
        offsets.append((d["protein_id"], offset, offset + n))
        offset += n
    return offsets


def split_indices(n_samples, test_size, random_state, stratify=None):
    """Return train/test index arrays for pooled data."""
    all_idx = np.arange(n_samples)
    idx_train, idx_test = train_test_split(
        all_idx,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )
    return idx_train, idx_test


def concatenate_labels(protein_data):
    """Concatenate per-protein pair labels into a pooled string array."""
    return np.concatenate([d["labels"] for d in protein_data])


def encode_labels(labels):
    """Encode string labels to integer class IDs and return (y, encoder)."""
    enc = LabelEncoder()
    return enc.fit_transform(labels), enc


def build_layer_feature_matrix(protein_data, layer, mode="base"):
    """Load pair blocks for one layer and build the pooled feature matrix."""
    chunks = []
    for d in protein_data:
        pb_path = get_local_pair_block_path(d["protein_dir"], d["protein_id"], layer)
        pb = np.load(str(pb_path))
        chunk = build_pair_features(pb, d["pb_i"], d["pb_j"], mode=mode)
        chunks.append(chunk)
        del pb
    return np.vstack(chunks)

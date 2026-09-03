from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_layerwise_ablation_change_scatter import (  # noqa: E402
    EXPECTED_PROTEINS,
    compute_layer_changes,
    expected_output_filenames,
)


def test_layer_changes_average_per_protein_tm_differences() -> None:
    protein_offsets = np.arange(EXPECTED_PROTEINS, dtype=float)[:, None]
    layer_values = np.arange(48, dtype=float)[None, :] ** 2
    matrix = protein_offsets + layer_values
    triangle_means = np.arange(48, dtype=float) ** 3

    triangle_change, mean_tm_change = compute_layer_changes(matrix, triangle_means)

    assert triangle_change.shape == (47,)
    assert mean_tm_change.shape == (47,)
    assert np.array_equal(triangle_change, np.diff(triangle_means))
    assert np.array_equal(mean_tm_change, np.diff(layer_values[0]))


def test_layer_changes_reject_incomplete_tm_matrix() -> None:
    with pytest.raises(ValueError, match="TM-score matrix has shape"):
        compute_layer_changes(np.zeros((EXPECTED_PROTEINS, 47)), np.zeros(48))


def test_expected_inventory_has_six_png_pdf_pairs() -> None:
    names = expected_output_filenames()
    assert len(names) == 12
    assert sum(name.endswith(".png") for name in names) == 6
    assert sum(name.endswith(".pdf") for name in names) == 6

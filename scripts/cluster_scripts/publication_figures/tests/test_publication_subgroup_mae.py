from __future__ import annotations

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
CLUSTER_DIR = SCRIPT_DIR.parent
TRIANGLE_DIR = CLUSTER_DIR / "triangle_violation_subgroups"
for path in (SCRIPT_DIR, CLUSTER_DIR, TRIANGLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from publication_subgroup_mae_common import (  # noqa: E402
    PER_PROTEIN_COLUMNS, subgroup_rows, validate_complete_layer,
)
from regression_protocol import (  # noqa: E402
    sequence_separation_bin_codes, spatial_distance_bin_codes,
)
from triangle_violation_subgroup_common import FAMILY_SPECS  # noqa: E402


def test_canonical_half_open_bins() -> None:
    pair_i = np.asarray([1, 4, 9, 19, 49, 50])
    pair_j = np.zeros(6, dtype=int)
    assert sequence_separation_bin_codes(pair_i, pair_j).tolist() == [0, 1, 2, 3, 4, 5]
    assert spatial_distance_bin_codes(np.asarray([0, 4.999, 5, 9.999, 10, 19.999, 20, 39.999, 40])).tolist() == [0, 0, 1, 1, 2, 2, 3, 3, 4]
    chains = ("A", "A", "B", "B", "B")
    assert sequence_separation_bin_codes(np.asarray([2]), np.asarray([0]), chain_ids=chains).tolist() == [-1]


def test_subgroup_rows_preserve_empty_categories_and_additive_mae() -> None:
    spec = FAMILY_SPECS[0]
    rows = subgroup_rows(
        spec=spec, layer=3, protein_id="toy", source="ecod", outer_fold=1,
        n_residues=4, absolute_error=np.asarray([1.0, 2.0, 4.0]),
        sequence_codes=np.asarray([0, 0, 1]), spatial_codes=np.asarray([2, 2, 4]),
    )
    assert len(rows) == 11
    assert all(set(row) == set(PER_PROTEIN_COLUMNS) for row in rows)
    sequence_one = next(row for row in rows if row["category_scope"] == "sequence_separation" and row["category_label"] == "1")
    assert sequence_one["n_pairs"] == 2
    assert sequence_one["sum_absolute_error"] == 3.0
    assert sequence_one["mae"] == 1.5
    empty = next(row for row in rows if row["category_scope"] == "spatial_distance" and row["category_label"] == "0-5")
    assert empty["support_status"] == "empty_category"
    assert empty["n_pairs"] == 0 and math.isnan(empty["mae"])


def test_layer_validation_requires_four_models_and_unique_complete_keys() -> None:
    rows = []
    for spec in FAMILY_SPECS:
        rows.extend(subgroup_rows(
            spec=spec, layer=0, protein_id="toy", source="ecod", outer_fold=0,
            n_residues=3, absolute_error=np.asarray([1.0, 2.0, 3.0]),
            sequence_codes=np.asarray([0, 1, 2]), spatial_codes=np.asarray([0, 1, 2]),
        ))
    validate_complete_layer(rows, 0, expected_proteins=1)
    with pytest.raises(ValueError, match="expected 44"):
        validate_complete_layer(rows[:-1], 0, expected_proteins=1)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_complete_layer(rows[:-1] + [rows[0]], 0, expected_proteins=1)

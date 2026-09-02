from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_sept_1_publication_plots import (  # noqa: E402
    MODEL_ORDER, bin_slug, paired_bootstrap_summary, validate_subgroup_values,
)


def synthetic_values() -> pd.DataFrame:
    rows = []
    for analysis_index, analysis in enumerate(MODEL_ORDER):
        for protein_index, protein_id in enumerate(("p1", "p2", "p3")):
            for layer in range(48):
                rows.append({
                    "analysis_name": analysis, "model": analysis,
                    "protein_id": protein_id, "layer": layer,
                    "value": analysis_index + protein_index + layer / 100.0,
                })
    return pd.DataFrame(rows)


def test_paired_bootstrap_is_deterministic_and_two_sd_uses_sample_sd() -> None:
    values = synthetic_values()
    first = paired_bootstrap_summary(values, replicates=300, seed=17)
    second = paired_bootstrap_summary(values, replicates=300, seed=17)
    pd.testing.assert_frame_equal(first, second)
    row = first[(first["analysis_name"].eq(MODEL_ORDER[0])) & first["layer"].eq(0)].iloc[0]
    assert row["mean"] == 1.0
    assert row["sd"] == 1.0
    assert row["se"] == pytest.approx(1 / np.sqrt(3))
    assert row["two_sd_low"] == -1.0 and row["two_sd_high"] == 3.0
    assert row["bootstrap_ci_low"] <= row["mean"] <= row["bootstrap_ci_high"]


def test_bin_slug_is_stable_for_publication_filenames() -> None:
    assert bin_slug("0-5") == "0_5"
    assert bin_slug("40+") == "40_plus"
    assert bin_slug("50+") == "50_plus"


def test_subgroup_validator_rejects_incomplete_data() -> None:
    with pytest.raises(ValueError, match="missing columns"):
        validate_subgroup_values(pd.DataFrame({"analysis_name": []}), expected_proteins=1)

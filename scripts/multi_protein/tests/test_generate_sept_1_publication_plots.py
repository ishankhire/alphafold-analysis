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
    MODEL_ORDER, SEQUENCE_BINS, SPATIAL_BINS, bin_slug, paired_bootstrap_summary,
    pooled_subgroup_summary, validate_pooled_subgroup_values, validate_subgroup_values,
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


def synthetic_pooled_subgroups() -> pd.DataFrame:
    rows = []
    for analysis in MODEL_ORDER:
        for layer in range(48):
            for scope, labels in (
                ("spatial_distance", SPATIAL_BINS),
                ("sequence_separation", SEQUENCE_BINS),
            ):
                for category_label in labels:
                    rows.append({
                        "analysis_name": analysis,
                        "model_family": "synthetic",
                        "layer": layer,
                        "evaluation_design": "protein_outer_5fold",
                        "evaluation_scope": "oof_all_pairs",
                        "oof_prediction_scope": "heldout_proteins_all_pairs",
                        "category_scope": scope,
                        "category_label": category_label,
                        "n_proteins": 552,
                        "n_expected_pairs": 100,
                        "n_evaluated_pairs": 100,
                        "pair_coverage": 1.0,
                        "support_status": "sufficient",
                        "pooled_mae": 1.0 + layer / 100.0,
                    })
    return pd.DataFrame(rows)


def test_pooled_subgroup_placeholder_is_explicit_and_has_no_fake_uncertainty() -> None:
    values = synthetic_pooled_subgroups()
    validate_pooled_subgroup_values(values)
    summary = pooled_subgroup_summary(values)
    assert set(summary["aggregation"]) == {"pair_pooled_mae"}
    assert summary["temporary_pooled_mae_placeholder"].all()
    assert summary["n_proteins_total"].eq(552).all()
    assert summary["n_proteins_finite"].eq(552).all()
    assert np.allclose(summary["mean"], summary["pooled_mae"])
    assert summary[[
        "sd", "se", "bootstrap_ci_low", "bootstrap_ci_high", "two_sd_low", "two_sd_high",
    ]].isna().all().all()


def test_pooled_subgroup_validator_rejects_incomplete_pair_coverage() -> None:
    values = synthetic_pooled_subgroups()
    values.loc[0, "n_evaluated_pairs"] = 99
    with pytest.raises(ValueError, match="complete pair coverage"):
        validate_pooled_subgroup_values(values)


def test_pooled_subgroup_validator_allows_empty_bin_support_for_some_proteins() -> None:
    values = synthetic_pooled_subgroups()
    values.loc[values["category_label"].eq("40+"), "n_proteins"] = 479
    validate_pooled_subgroup_values(values)

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from generate_sept_2_publication_plots import (  # noqa: E402
    MODEL_ORDER,
    REGRESSION_ROOT,
    SEQUENCE_BINS,
    SPATIAL_BINS,
    benjamini_hochberg,
    bin_slug,
    bootstrap_counts,
    bootstrap_matrix_summary,
    bootstrap_summary,
    clip_interval,
    expected_figure_ids,
    peak_trough_test_at_layer,
    phase_axis_summary,
    require_input_path,
)


def synthetic_values() -> pd.DataFrame:
    rows = []
    for analysis_index, analysis in enumerate(MODEL_ORDER):
        for protein_index, protein_id in enumerate(("p1", "p2", "p3")):
            for layer in range(48):
                rows.append(
                    {
                        "analysis_name": analysis,
                        "model": analysis,
                        "protein_id": protein_id,
                        "layer": layer,
                        "value": analysis_index + protein_index + layer / 100.0,
                        "evaluation_design": "synthetic_5fold",
                        "evaluation_scope": "oof_all_pairs",
                        "oof_prediction_scope": "synthetic_oof",
                    }
                )
    return pd.DataFrame(rows)


def test_bootstrap_is_deterministic_and_protein_equal() -> None:
    values = synthetic_values()
    first_counts = bootstrap_counts(n_proteins=3, replicates=500, seed=17)
    second_counts = bootstrap_counts(n_proteins=3, replicates=500, seed=17)
    assert np.array_equal(first_counts, second_counts)
    first = bootstrap_summary(values, ("p1", "p2", "p3"), first_counts)
    second = bootstrap_summary(values, ("p1", "p2", "p3"), second_counts)
    pd.testing.assert_frame_equal(first, second)
    row = first[
        first["analysis_name"].eq(MODEL_ORDER[0]) & first["layer"].eq(0)
    ].iloc[0]
    assert row["mean"] == pytest.approx(1.0)
    assert row["n_proteins_finite"] == 3
    assert row["bootstrap_ci_low"] <= row["mean"] <= row["bootstrap_ci_high"]


def test_missing_bin_values_average_only_finite_proteins() -> None:
    matrix = np.array([[1.0, np.nan], [3.0, 4.0], [5.0, np.nan]])
    counts = bootstrap_counts(n_proteins=3, replicates=500, seed=4)
    result = bootstrap_matrix_summary(matrix, counts)
    assert result["mean"].tolist() == pytest.approx([3.0, 4.0])
    assert result["n_proteins_finite"].tolist() == [3, 1]


def test_joint_phase_intervals_use_fixed_point_estimate_range() -> None:
    matrix = np.array([[0.0, 2.0, 4.0], [2.0, 4.0, 6.0], [4.0, 6.0, 8.0]])
    counts = bootstrap_counts(n_proteins=3, replicates=500, seed=11)
    result = phase_axis_summary(matrix, counts)
    assert result["mean"].tolist() == pytest.approx([2.0, 4.0, 6.0])
    assert result["normalized_mean"].tolist() == pytest.approx([0.0, 0.5, 1.0])
    expected_low = (result["bootstrap_ci_low"] - 2.0) / 4.0
    assert result["normalized_bootstrap_ci_low"].tolist() == pytest.approx(
        expected_low.tolist()
    )


def test_bh_adjustment_is_one_family_and_monotone_by_rank() -> None:
    p_values = np.array([0.01, 0.04, 0.03, 0.20])
    adjusted = benjamini_hochberg(p_values)
    assert adjusted.tolist() == pytest.approx([0.04, 0.0533333333, 0.0533333333, 0.20])
    ranked = adjusted[np.argsort(p_values)]
    assert np.all(np.diff(ranked) >= 0)


def test_peak_candidate_requires_strict_window_and_slope_shape() -> None:
    trajectory = np.array([0, 1, 2, 3, 4, 8, 4, 3, 2, 1, 0], dtype=float)
    matrix = np.vstack([trajectory, trajectory + 0.1, trajectory - 0.1])
    peak = peak_trough_test_at_layer(matrix, layer=5, direction="peak", window=5)
    trough = peak_trough_test_at_layer(matrix, layer=5, direction="trough", window=5)
    assert peak["shape_qualified_candidate"] is True
    assert peak["combined_p_value"] <= 0.05
    assert trough["shape_qualified_candidate"] is False


def test_metric_domain_clipping_is_display_only() -> None:
    low = np.array([-0.2, 0.2])
    high = np.array([0.8, 1.2])
    clipped_low, clipped_high = clip_interval(low, high, "unit")
    assert clipped_low.tolist() == [0.0, 0.2]
    assert clipped_high.tolist() == [0.8, 1.0]
    assert low.tolist() == [-0.2, 0.2]
    unbounded_low, unbounded_high = clip_interval(low, high, "unbounded")
    assert np.array_equal(unbounded_low, low)
    assert np.array_equal(unbounded_high, high)


def test_figure_names_cover_29_pairs_and_all_canonical_bins() -> None:
    figure_ids = expected_figure_ids()
    assert len(figure_ids) == 29
    assert len(set(figure_ids)) == 29
    assert "supp_figure_top3_eigenvalue_fraction" in figure_ids
    assert "supp_figure_mds_stress" in figure_ids
    for category in SPATIAL_BINS:
        assert f"main_figure_15_spatial_distance_mae_{bin_slug(category)}" in figure_ids
    for category in SEQUENCE_BINS:
        assert f"supp_figure_sequence_separation_mae_{bin_slug(category)}" in figure_ids


def test_input_guard_rejects_out_of_root_paths() -> None:
    with pytest.raises(ValueError, match="outside regression_5fold_v1"):
        require_input_path(Path(__file__))
    assert require_input_path(REGRESSION_ROOT / "metadata/regression_outer_folds_v1.csv").is_file()

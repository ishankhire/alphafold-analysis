from __future__ import annotations

import sys
import unittest
from pathlib import Path

import numpy as np
import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from analyze_triangle_metric_peak_troughs import (  # noqa: E402
    ELIGIBLE_LAYERS,
    EXPECTED_PROTEINS,
    analyze_metric_wide,
    benjamini_hochberg,
    match_event_rows,
    one_sample_greater,
)


def synthetic_wide(means: np.ndarray) -> pd.DataFrame:
    protein_axis = np.linspace(-1.0, 1.0, EXPECTED_PROTEINS)
    perturbation = np.outer(protein_axis, 0.002 * np.sin(np.arange(48) + 1.0))
    return pd.DataFrame(means[np.newaxis, :] + perturbation, columns=range(48))


def analyze_synthetic(means: np.ndarray) -> pd.DataFrame:
    return pd.DataFrame(
        analyze_metric_wide(
            synthetic_wide(means),
            analysis_name="general_base256_combined",
            model_family="shared_ols",
            category_system="all_triangles",
            category_label="all",
            metric="mean_relative_violation_all",
        )
    )


class BenjaminiHochbergTests(unittest.TestCase):
    def test_known_adjustment_and_stable_bounds(self) -> None:
        adjusted = benjamini_hochberg([0.01, 0.04, 0.03, 0.002])
        np.testing.assert_allclose(adjusted, [0.02, 0.04, 0.04, 0.008])
        self.assertTrue(np.logical_and(adjusted >= 0.0, adjusted <= 1.0).all())

    def test_nonfinite_values_are_retained_as_missing(self) -> None:
        adjusted = benjamini_hochberg([0.01, np.nan, 0.02])
        np.testing.assert_allclose(adjusted[[0, 2]], [0.02, 0.02])
        self.assertTrue(np.isnan(adjusted[1]))


class OneSampleGreaterTests(unittest.TestCase):
    def test_zero_variance_statuses_are_explicit(self) -> None:
        positive = one_sample_greater(np.ones(5))
        zero = one_sample_greater(np.zeros(5))
        negative = one_sample_greater(-np.ones(5))
        self.assertEqual(positive["status"], "zero_variance_positive")
        self.assertEqual(positive["p"], 0.0)
        self.assertEqual(zero["status"], "zero_variance_zero")
        self.assertEqual(zero["p"], 1.0)
        self.assertEqual(negative["status"], "zero_variance_negative")
        self.assertEqual(negative["p"], 1.0)

    def test_nonfinite_status_is_explicit(self) -> None:
        result = one_sample_greater([1.0, np.nan, 2.0])
        self.assertEqual(result["status"], "nonfinite_input")
        self.assertTrue(np.isnan(result["p"]))


class CandidateDefinitionTests(unittest.TestCase):
    def test_clear_peak_and_sign_reversal(self) -> None:
        means = np.ones(48)
        means[5:16] = [1, 2, 3, 4, 5, 6, 5, 4, 3, 2, 1]
        result = analyze_synthetic(means)
        peak = result[(result["layer"].eq(10)) & result["direction"].eq("peak")].iloc[0]
        trough = result[(result["layer"].eq(10)) & result["direction"].eq("trough")].iloc[0]

        self.assertTrue(peak["strict_window_extremum"])
        self.assertTrue(peak["slope_shape_pass"])
        self.assertTrue(peak["shape_qualified_candidate"])
        self.assertFalse(trough["shape_qualified_candidate"])
        self.assertAlmostEqual(peak["left_contrast_mean"], -trough["left_contrast_mean"])
        self.assertAlmostEqual(peak["right_contrast_mean"], -trough["right_contrast_mean"])
        self.assertEqual(
            peak["combined_p_value"],
            max(peak["left_p_value"], peak["right_p_value"]),
        )
        self.assertEqual(set(result["layer"]), set(ELIGIBLE_LAYERS))

    def test_plateau_is_not_a_strict_candidate(self) -> None:
        means = np.ones(48)
        means[5:16] = [1, 2, 3, 4, 5, 6, 6, 4, 3, 2, 1]
        result = analyze_synthetic(means)
        peak = result[(result["layer"].eq(10)) & result["direction"].eq("peak")].iloc[0]
        self.assertFalse(peak["strict_window_extremum"])
        self.assertFalse(peak["shape_qualified_candidate"])

    def test_single_jump_with_wrong_left_trend_fails_shape(self) -> None:
        means = np.ones(48)
        means[5:16] = [10, 8, 6, 4, 2, 11, 10, 9, 8, 7, 6]
        result = analyze_synthetic(means)
        peak = result[(result["layer"].eq(10)) & result["direction"].eq("peak")].iloc[0]
        self.assertTrue(peak["strict_window_extremum"])
        self.assertLess(peak["left_slope"], 0.0)
        self.assertFalse(peak["slope_shape_pass"])
        self.assertFalse(peak["shape_qualified_candidate"])


class AlignmentTests(unittest.TestCase):
    @staticmethod
    def events(layers: list[int]) -> pd.DataFrame:
        return pd.DataFrame(
            {
                "layer": layers,
                "raw_prominence": np.linspace(0.1, 0.2, len(layers)),
                "standardized_prominence": np.linspace(0.3, 0.4, len(layers)),
                "bh_q_value": np.full(len(layers), 0.01),
            }
        )

    def test_exact_nearby_and_unmatched_events(self) -> None:
        records, summary = match_event_rows(
            self.events([10, 30, 40]),
            self.events([10, 32]),
            {"comparison_id": "test"},
        )
        statuses = [record["match_status"] for record in records]
        self.assertEqual(statuses.count("exact_match"), 1)
        self.assertEqual(statuses.count("within_2_match"), 1)
        self.assertEqual(statuses.count("only_a"), 1)
        self.assertEqual(summary["n_exact_matches"], 1)
        self.assertEqual(summary["n_within_2_matches_including_exact"], 2)
        self.assertEqual(summary["n_unmatched_a"], 1)
        self.assertEqual(summary["n_unmatched_b"], 0)


if __name__ == "__main__":
    unittest.main()

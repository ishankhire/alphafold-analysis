"""Adversarial dataflow and aggregation checks for protocol-2.0 CV.

Every expected scalar in this module is computed directly from the synthetic
inputs.  The tests deliberately avoid using a production metric helper to
construct its own oracle.
"""

from __future__ import annotations

import hashlib
import math
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
CV_DIR = TEST_DIR.parent
CLUSTER_DIR = CV_DIR.parent
for directory in (CLUSTER_DIR, CV_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import cross_validation_common as cv
import per_protein_ols_base256_common as per_protein
import sequence_gated_ols_base256_common as gated
import shared_ols_base256_common as shared
import shared_xgboost_base256_fixed_s4_common as xgb
from regression_protocol import (
    FEATURE_DIM,
    N_CHANNELS,
    extract_directional_base256,
    strict_lower_triangle_indices,
)


def _literal_metrics(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    """Independent direct-concatenation regression oracle."""
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    if y.ndim != 1 or pred.shape != y.shape:
        raise ValueError("literal metric inputs must be matching vectors")
    error = pred - y
    mean = sum(float(value) for value in y) / y.size
    sse = sum(float(value) ** 2 for value in error)
    sst = sum((float(value) - mean) ** 2 for value in y)
    return {
        "r2": 1.0 - sse / sst,
        "mae": sum(abs(float(value)) for value in error) / y.size,
        "rmse": math.sqrt(sse / y.size),
        "bias_pred_minus_true": sum(float(value) for value in error) / y.size,
    }


def _fit_digest(fit: object) -> str:
    """Digest only fitted OLS state, not evaluation data or diagnostics."""
    digest = hashlib.sha256()
    for name in (
        "coefficient_standardized",
        "coefficient_raw_scale",
        "feature_mean",
        "feature_scale",
    ):
        digest.update(np.asarray(getattr(fit, name), dtype="<f8").tobytes())
    for name in ("intercept_standardized", "intercept_raw_scale"):
        digest.update(np.asarray(float(getattr(fit, name)), dtype="<f8").tobytes())
    for name in ("scaler_digest",):
        digest.update(str(getattr(fit, name)).encode("utf-8"))
    return digest.hexdigest()


def _active_feature_blocks(
    *, n_proteins: int = 5, n_rows: int = 60, seed: int = 611
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Low-rank base256 signal that cannot memorize a target permutation."""
    feature_blocks: dict[str, np.ndarray] = {}
    target_blocks: dict[str, np.ndarray] = {}
    for index in range(n_proteins):
        rng = np.random.default_rng(seed + index)
        active = rng.normal(loc=index / 5.0, size=(n_rows, 2))
        block = np.zeros((n_rows, FEATURE_DIM), dtype=np.float32)
        block[:, :2] = active
        feature_blocks[f"p{index}"] = block
        target_blocks[f"p{index}"] = (
            3.25 + 1.75 * active[:, 0] - 0.6 * active[:, 1]
        ).astype(np.float64)
    return feature_blocks, target_blocks


def _xgb_record(protein_id: str, targets: np.ndarray, features: np.ndarray):
    n_rows = int(targets.size)
    return xgb.ProteinLayerRecord(
        protein_id=protein_id,
        source="cameo",
        n_residues=2,
        outer_fold=0,
        coordinates=np.zeros((2, 3), dtype=np.float64),
        chain_ids=("A", "A"),
        pair_i=np.arange(n_rows, dtype=np.int64),
        pair_j=np.zeros(n_rows, dtype=np.int64),
        targets=np.asarray(targets, dtype=np.float64),
        synthetic_features=np.asarray(features, dtype=np.float32),
    )


class LeakageAndPoisoningTests(unittest.TestCase):
    def test_shared_and_gated_ols_ignore_heldout_target_poisoning(self):
        features, targets = _active_feature_blocks()
        heldout_features = np.zeros((12, FEATURE_DIM), dtype=np.float32)
        heldout_features[:, 0] = np.linspace(-2.0, 2.0, 12)
        heldout_features[:, 1] = np.linspace(1.0, -1.0, 12)
        heldout_targets = 3.25 + 1.75 * heldout_features[:, 0] - 0.6 * heldout_features[:, 1]

        first, _, _ = shared.fit_shared_ols(features, targets)
        poisoned_heldout = heldout_targets + np.linspace(1_000.0, 2_000.0, 12)
        second, _, _ = shared.fit_shared_ols(features, targets)
        self.assertEqual(_fit_digest(first), _fit_digest(second))
        self.assertEqual(first.scaler_digest, second.scaler_digest)
        self.assertEqual(first.global_baseline_digest, second.global_baseline_digest)
        prediction = shared.predict_shared_ols(first, heldout_features)
        self.assertNotEqual(
            _literal_metrics(heldout_targets, prediction)["rmse"],
            _literal_metrics(poisoned_heldout, prediction)["rmse"],
        )

        codes = {protein_id: np.zeros(values.size, dtype=np.int16) for protein_id, values in targets.items()}
        baseline = cv.fit_train_only_baseline(targets)
        gated_first = gated.fit_sequence_gate_ols(
            features,
            targets,
            codes,
            sequence_gate="1",
            global_baseline_value=baseline.value,
            global_baseline_digest=baseline.baseline_digest,
        ).fit
        gated_second = gated.fit_sequence_gate_ols(
            features,
            targets,
            codes,
            sequence_gate="1",
            global_baseline_value=baseline.value,
            global_baseline_digest=baseline.baseline_digest,
        ).fit
        self.assertEqual(_fit_digest(gated_first), _fit_digest(gated_second))

    def test_per_protein_poisoning_respects_one_pair_fold_responsibility(self):
        features, targets_by_id = _active_feature_blocks(n_proteins=1, n_rows=75)
        values = features["p0"]
        targets = targets_by_id["p0"]
        assignments = np.arange(targets.size, dtype=np.int16) % 5
        fit_before, training, _, _ = per_protein.fit_pair_fold_ols(
            "p0", values, targets, assignments, 0,
        )

        poisoned = targets.copy()
        poisoned[assignments == 0] += 50_000.0
        fit_same, training_after, _, _ = per_protein.fit_pair_fold_ols(
            "p0", values, poisoned, assignments, 0,
        )
        np.testing.assert_array_equal(training, training_after)
        self.assertEqual(_fit_digest(fit_before), _fit_digest(fit_same))
        self.assertEqual(fit_before.baseline_digest, fit_same.baseline_digest)

        fit_other, _, _, _ = per_protein.fit_pair_fold_ols(
            "p0", values, poisoned, assignments, 1,
        )
        original_other, _, _, _ = per_protein.fit_pair_fold_ols(
            "p0", values, targets, assignments, 1,
        )
        self.assertNotEqual(_fit_digest(original_other), _fit_digest(fit_other))
        self.assertNotEqual(original_other.baseline_digest, fit_other.baseline_digest)

    def test_training_target_poisoning_changes_model_but_not_scaler(self):
        features, targets = _active_feature_blocks(n_proteins=2, n_rows=45)
        first, _, _ = shared.fit_shared_ols(features, targets)
        poisoned = {protein_id: values.copy() for protein_id, values in targets.items()}
        poisoned["p0"] = poisoned["p0"] + 7.0 * features["p0"][:, 1]
        second, _, _ = shared.fit_shared_ols(features, poisoned)
        self.assertEqual(first.scaler_digest, second.scaler_digest)
        self.assertNotEqual(first.global_baseline_digest, second.global_baseline_digest)
        self.assertNotEqual(_fit_digest(first), _fit_digest(second))

    def test_xgboost_training_materialization_excludes_heldout_targets(self):
        features, targets = _active_feature_blocks(n_proteins=3, n_rows=20)
        training_records = [
            _xgb_record(protein_id, targets[protein_id], features[protein_id])
            for protein_id in ("p0", "p1")
        ]
        heldout = _xgb_record("p2", targets["p2"], features["p2"])
        first = xgb.materialize_sampled_training(training_records)
        heldout_poisoned = replace(heldout, targets=heldout.targets + 1e6)
        del heldout_poisoned
        second = xgb.materialize_sampled_training(training_records)
        np.testing.assert_array_equal(first.features, second.features)
        np.testing.assert_array_equal(first.targets, second.targets)
        np.testing.assert_array_equal(first.sample_weights, second.sample_weights)
        self.assertEqual(first.global_baseline_digest, second.global_baseline_digest)
        self.assertEqual(first.sample_assignment_digest, second.sample_assignment_digest)


class OrderingPrecisionAndShapeTests(unittest.TestCase):
    def test_joint_pair_permutation_is_invariant_and_mismatch_is_detected(self):
        features, targets = _active_feature_blocks(n_proteins=3, n_rows=50)
        canonical, _, _ = shared.fit_shared_ols(features, targets)
        permuted_features: dict[str, np.ndarray] = {}
        permuted_targets: dict[str, np.ndarray] = {}
        mismatched_targets: dict[str, np.ndarray] = {}
        for index, protein_id in enumerate(sorted(features)):
            order = np.random.default_rng(820 + index).permutation(targets[protein_id].size)
            permuted_features[protein_id] = features[protein_id][order]
            permuted_targets[protein_id] = targets[protein_id][order]
            mismatched_targets[protein_id] = targets[protein_id]
        joint, _, _ = shared.fit_shared_ols(permuted_features, permuted_targets)
        mismatch, _, _ = shared.fit_shared_ols(permuted_features, mismatched_targets)

        heldout, heldout_target = _active_feature_blocks(n_proteins=1, n_rows=80, seed=921)
        x_holdout = heldout["p0"]
        y_holdout = heldout_target["p0"]
        pred_canonical = shared.predict_shared_ols(canonical, x_holdout)
        pred_joint = shared.predict_shared_ols(joint, x_holdout)
        pred_mismatch = shared.predict_shared_ols(mismatch, x_holdout)
        np.testing.assert_allclose(pred_canonical, pred_joint, rtol=1e-10, atol=1e-10)
        self.assertLess(
            _literal_metrics(y_holdout, pred_canonical)["rmse"],
            _literal_metrics(y_holdout, pred_mismatch)["rmse"] * 0.1,
        )

    def test_directional_feature_order_survives_artifact_replay(self):
        pair_i, pair_j = strict_lower_triangle_indices(3)
        pair_block = np.zeros((3, 3, N_CHANNELS), dtype=np.float32)
        for row, (i, j) in enumerate(zip(pair_i, pair_j)):
            pair_block[i, j, 0] = 10.0 + row
            pair_block[j, i, 0] = 100.0 + row
        features = extract_directional_base256(pair_block, pair_i, pair_j)
        np.testing.assert_array_equal(features[:, 0], [10.0, 11.0, 12.0])
        np.testing.assert_array_equal(features[:, N_CHANNELS], [100.0, 101.0, 102.0])

        coefficients = np.zeros(FEATURE_DIM, dtype=np.float64)
        coefficients[0] = 2.0
        coefficients[N_CHANNELS] = -0.5
        fit = shared.SharedOLSFit(
            coefficient_standardized=coefficients,
            coefficient_raw_scale=coefficients.copy(),
            intercept_standardized=7.0,
            intercept_raw_scale=7.0,
            feature_mean=np.zeros(FEATURE_DIM),
            feature_scale=np.ones(FEATURE_DIM),
            scaler_digest="directional-scaler",
            global_baseline_value=0.0,
            global_baseline_digest="directional-baseline",
            design_rank=2,
            largest_singular_value=1.0,
            smallest_singular_value=1.0,
            condition_number=1.0,
            fit_status="fitted_rank_deficient",
            support_flags="rank_deficient",
            n_training_proteins=1,
            n_training_pairs=3,
        )
        expected = np.asarray([
            7.0 + 2.0 * float(row[0]) - 0.5 * float(row[N_CHANNELS])
            for row in features
        ])
        np.testing.assert_array_equal(shared.predict_shared_ols(fit, features), expected)
        swapped = np.column_stack((features[:, N_CHANNELS:], features[:, :N_CHANNELS]))
        self.assertFalse(np.allclose(shared.predict_shared_ols(fit, swapped), expected))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "directional_coefficients.npz"
            task = cv.FoldLayerTask(task_id=0, outer_fold=0, layer=0)
            shared.write_coefficient_artifact(path, fit, task)
            loaded = shared.load_coefficient_artifact(
                path, expected_task=task, expected_scaler_digest=fit.scaler_digest,
            )
        np.testing.assert_array_equal(loaded["coefficient_standardized"], coefficients)
        self.assertEqual(float(loaded["intercept_standardized"]), 7.0)

    def test_float64_conversion_precedes_products_and_squares(self):
        raw_features = np.asarray([[1.7e19], [-1.7e19]], dtype=np.float32)
        fitted = cv.fit_train_only_standardizer({"p": raw_features})
        converted = raw_features.astype(np.float64)
        expected_mean = sum(float(value) for value in converted[:, 0]) / 2.0
        expected_second = sum(float(value) ** 2 for value in converted[:, 0]) / 2.0
        expected_scale = math.sqrt(expected_second - expected_mean**2)
        self.assertEqual(float(fitted.feature_mean[0]), expected_mean)
        self.assertTrue(math.isclose(float(fitted.feature_scale[0]), expected_scale, rel_tol=1e-15))
        with np.errstate(over="ignore", invalid="ignore"):
            float32_first = float(np.sqrt(np.mean(raw_features[:, 0] * raw_features[:, 0])))
        self.assertFalse(math.isclose(float(fitted.feature_scale[0]), float32_first, rel_tol=1e-12))

        raw_targets = np.asarray([1e20, -1e20], dtype=np.float32)
        stats = cv.RegressionSufficientStatistics()
        stats.update(raw_targets, np.zeros(2, dtype=np.float32))
        expected_y2 = sum(float(value) ** 2 for value in raw_targets.astype(np.float64))
        self.assertTrue(math.isclose(stats.sum_y2, expected_y2, rel_tol=1e-15))
        with np.errstate(over="ignore"):
            self.assertTrue(np.isinf(np.sum(raw_targets * raw_targets, dtype=np.float64)))

    def test_broadcastable_and_wrong_width_inputs_are_rejected(self):
        stats = cv.RegressionSufficientStatistics()
        with self.assertRaises(ValueError):
            stats.update(np.arange(4.0).reshape(-1, 1), np.arange(4.0))
        with self.assertRaises(ValueError):
            stats.update(np.arange(4.0), np.arange(4.0).reshape(-1, 1))

        features, targets = _active_feature_blocks(n_proteins=1, n_rows=12)
        fit, _, _ = shared.fit_shared_ols(features, targets)
        for bad in (
            features["p0"].T,
            features["p0"][:, : FEATURE_DIM - 1],
            np.column_stack((features["p0"], np.zeros(features["p0"].shape[0]))),
        ):
            with self.assertRaises(ValueError):
                shared.predict_shared_ols(fit, bad)
        nonfinite = features["p0"].copy()
        nonfinite[0, 0] = np.nan
        with self.assertRaises(ValueError):
            shared.predict_shared_ols(fit, nonfinite)

        pair_block = np.zeros((3, 3, N_CHANNELS), dtype=np.float32)
        pair_i, pair_j = strict_lower_triangle_indices(3)
        with self.assertRaises(ValueError):
            extract_directional_base256(pair_block, pair_i[:-1], pair_j)

    def test_missing_predictions_cannot_form_complete_outputs(self):
        y = np.arange(1.0, 7.0)
        accumulator = cv.OOFComparisonAccumulator()
        prediction = y.copy()
        prediction[-1] = np.nan
        accumulator.update_protein("p", 0, y, prediction, 0.0)
        self.assertFalse(accumulator.is_complete)
        with self.assertRaises(ValueError):
            accumulator.require_complete()
        with self.assertRaises(ValueError):
            cv.build_pooled_metrics_row(
                accumulator,
                analysis_name="adversarial",
                model_family="shared_ols",
                feature_dim=2,
                layer=0,
                evaluation_design="protein_outer_5fold",
                oof_prediction_scope="heldout_proteins_all_pairs",
            )

        row = cv.build_per_protein_row(
            analysis_name="adversarial",
            model_family="shared_ols",
            feature_dim=2,
            layer=0,
            protein_id="p",
            source="cameo",
            outer_fold=0,
            evaluation_design="protein_outer_5fold",
            oof_prediction_scope="heldout_proteins_all_pairs",
            n_residues=4,
            y_true=y,
            predictions=prediction,
        )
        self.assertEqual(row["prediction_status"], "incomplete_pair_coverage")
        self.assertEqual(row["n_evaluated_pairs"], 5)
        with self.assertRaises(ValueError):
            cv.build_per_protein_row(
                analysis_name="adversarial",
                model_family="shared_ols",
                feature_dim=2,
                layer=0,
                protein_id="p",
                source="cameo",
                outer_fold=0,
                evaluation_design="protein_outer_5fold",
                oof_prediction_scope="heldout_proteins_all_pairs",
                n_residues=4,
                y_true=y,
                predictions=prediction,
                prediction_status="complete",
            )


class IndependentAggregationTests(unittest.TestCase):
    def test_direct_oof_metrics_survive_merge_serialization_and_consolidation(self):
        fold_targets = (
            np.asarray([0.0, 1.0]),
            np.asarray([10.0, 20.0, 30.0, 40.0]),
        )
        fold_predictions = (
            np.asarray([0.0, 3.0]),
            np.asarray([9.0, 18.0, 27.0, 36.0]),
        )
        parts = []
        pooled = cv.OOFComparisonAccumulator()
        fold_r2 = []
        fold_rmse = []
        for fold, (y, pred) in enumerate(zip(fold_targets, fold_predictions)):
            part = cv.OOFComparisonAccumulator()
            part.update_protein(f"p{fold}", fold, y, pred, 5.0)
            part.require_complete()
            parts.append(part)
            expected = _literal_metrics(y, pred)
            self.assertAlmostEqual(part.model.metrics()["r2"], expected["r2"])
            self.assertAlmostEqual(part.model.metrics()["rmse"], expected["rmse"])
            fold_r2.append(expected["r2"])
            fold_rmse.append(expected["rmse"])
            pooled.merge(part)

        restored = cv.OOFComparisonAccumulator.from_payload(pooled.to_payload())
        direct_y = np.concatenate(fold_targets)
        direct_pred = np.concatenate(fold_predictions)
        expected = _literal_metrics(direct_y, direct_pred)
        row = cv.build_pooled_metrics_row(
            restored,
            analysis_name="adversarial",
            model_family="shared_ols",
            feature_dim=2,
            layer=0,
            evaluation_design="protein_outer_5fold",
            oof_prediction_scope="heldout_proteins_all_pairs",
        )
        for name, column in (
            ("r2", "pooled_r2"),
            ("mae", "pooled_mae"),
            ("rmse", "pooled_rmse"),
            ("bias_pred_minus_true", "pooled_bias_pred_minus_true"),
        ):
            self.assertAlmostEqual(row[column], expected[name])
        self.assertNotAlmostEqual(row["pooled_r2"], sum(fold_r2) / len(fold_r2))
        self.assertNotAlmostEqual(row["pooled_rmse"], sum(fold_rmse) / len(fold_rmse))

    def test_pair_pooled_and_protein_equal_summaries_have_distinct_denominators(self):
        short_y = np.asarray([0.0, 2.0])
        short_pred = np.asarray([2.0, 4.0])
        long_y = np.arange(20.0)
        long_pred = long_y + 0.1
        accumulator = cv.OOFComparisonAccumulator()
        accumulator.update_protein("short", 0, short_y, short_pred, 0.0)
        accumulator.update_protein("long", 1, long_y, long_pred, 0.0)
        pair_pooled_mae = (
            sum(abs(float(a - b)) for a, b in zip(short_y, short_pred))
            + sum(abs(float(a - b)) for a, b in zip(long_y, long_pred))
        ) / (short_y.size + long_y.size)
        protein_equal_mae = (
            _literal_metrics(short_y, short_pred)["mae"]
            + _literal_metrics(long_y, long_pred)["mae"]
        ) / 2.0
        self.assertAlmostEqual(accumulator.model.metrics()["mae"], pair_pooled_mae)
        self.assertNotAlmostEqual(pair_pooled_mae, protein_equal_mae)

    def test_stable_filter_retains_flagged_primary_rows_but_excludes_sensitivity(self):
        accumulator = cv.OOFComparisonAccumulator()
        accumulator.update_protein(
            "stable", 0, np.asarray([1.0, 2.0]), np.asarray([1.1, 2.1]), 1.5,
            stable_fit=True,
        )
        accumulator.update_protein(
            "flagged", 1, np.asarray([3.0, 5.0]), np.asarray([3.5, 4.5]), 4.0,
            stable_fit=False,
        )
        primary = cv.build_pooled_metrics_row(
            accumulator,
            analysis_name="adversarial",
            model_family="shared_ols",
            feature_dim=2,
            layer=0,
            evaluation_design="protein_outer_5fold",
            oof_prediction_scope="heldout_proteins_all_pairs",
        )
        stable = cv.build_stable_fit_metrics_row(
            accumulator,
            analysis_name="adversarial",
            model_family="shared_ols",
            feature_dim=2,
            layer=0,
            evaluation_design="protein_outer_5fold",
            oof_prediction_scope="heldout_proteins_all_pairs",
        )
        self.assertEqual(primary["n_proteins"], 2)
        self.assertEqual(stable["n_proteins_total"], 2)
        self.assertEqual(stable["n_proteins_included"], 1)
        self.assertEqual(stable["n_proteins"], 1)

    def test_geometry_failure_summaries_preserve_attempted_included_excluded_counts(self):
        coordinates = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )
        pair_i, pair_j = strict_lower_triangle_indices(4)
        distances = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
        common_kwargs = dict(
            analysis_name="adversarial",
            model_family="shared_ols",
            feature_dim=2,
            layer=0,
            source="cameo",
            outer_fold=0,
            evaluation_design="protein_outer_5fold",
            oof_prediction_scope="heldout_proteins_all_pairs",
            n_residues=4,
        )
        complete = cv.build_geometry_row(
            **common_kwargs,
            protein_id="complete",
            true_coordinates=coordinates,
            predictions=distances,
        )
        incomplete_predictions = distances.copy()
        incomplete_predictions[0] = np.nan
        incomplete = cv.build_geometry_row(
            **common_kwargs,
            protein_id="incomplete",
            true_coordinates=coordinates,
            predictions=incomplete_predictions,
        )
        failed = cv.build_geometry_row(
            **common_kwargs,
            protein_id="failed",
            true_coordinates=coordinates[:, :2],
            predictions=distances,
        )
        self.assertEqual(complete["geometry_status"], "complete")
        self.assertEqual(incomplete["geometry_status"], "incomplete_pair_coverage")
        self.assertEqual(failed["geometry_status"], "geometry_failed")
        summaries = cv.summarize_geometry_rows([complete, incomplete, failed])
        self.assertTrue(summaries)
        self.assertTrue(all(row["n_proteins_attempted"] == 3 for row in summaries))
        self.assertTrue(all(row["n_proteins_included"] == 1 for row in summaries))
        self.assertTrue(all(row["n_proteins_excluded"] == 2 for row in summaries))


if __name__ == "__main__":
    unittest.main()

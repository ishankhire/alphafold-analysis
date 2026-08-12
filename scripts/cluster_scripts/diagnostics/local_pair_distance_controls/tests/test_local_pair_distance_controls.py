import csv
import importlib.util
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np


BUNDLE = Path(__file__).parents[1]


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


controls = load_module("local_pair_distance_controls", "local_pair_distance_controls.py")
aggregate = load_module(
    "aggregate_local_pair_distance_controls",
    "aggregate_local_pair_distance_controls.py",
)


class DistanceControlTests(unittest.TestCase):
    def test_directional_feature_order_matches_saved_probe(self):
        pb = np.empty((5, 5, controls.N_CHANNELS), dtype=np.float64)
        channels = np.arange(controls.N_CHANNELS, dtype=np.float64)
        for i in range(5):
            for j in range(5):
                pb[i, j] = 1000 * i + 10 * j + channels
        hi = np.asarray([3, 4])
        lo = np.asarray([0, 2])
        actual = controls.directional_features(pb, hi, lo)
        np.testing.assert_allclose(actual[:, : controls.N_CHANNELS], pb[hi, lo])
        np.testing.assert_allclose(actual[:, controls.N_CHANNELS :], pb[lo, hi])

    def test_local_pair_mask_is_inclusive_four_through_twelve(self):
        hi = np.asarray([4, 12, 13, 3, 20])
        lo = np.asarray([0, 0, 0, 0, 7])
        np.testing.assert_array_equal(
            controls.local_pair_mask(hi, lo), [True, True, False, False, False]
        )

    def test_saved_coefficient_loading_requires_both_complete_layers(self):
        fields = [
            "variant", "feature_dim", "split_id", "layer", "feature_index",
            "feature_name", "coefficient_standardized", "coefficient_raw_scale",
        ]
        names = controls.feature_names_for_variant(controls.VARIANT_SPEC)
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "coefficients.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for layer in (10, 11):
                    writer.writerow({
                        "variant": controls.SOURCE_VARIANT, "feature_dim": controls.FEATURE_DIM,
                        "split_id": controls.SPLIT_ID, "layer": layer, "feature_index": -1,
                        "feature_name": "intercept", "coefficient_standardized": 0,
                        "coefficient_raw_scale": layer,
                    })
                    for index, name in enumerate(names):
                        writer.writerow({
                            "variant": controls.SOURCE_VARIANT,
                            "feature_dim": controls.FEATURE_DIM,
                            "split_id": controls.SPLIT_ID, "layer": layer,
                            "feature_index": index, "feature_name": name,
                            "coefficient_standardized": index,
                            "coefficient_raw_scale": index + layer,
                        })
            models = controls.load_saved_models(path, (10, 11))
            self.assertEqual(models[10]["intercept"], 10)
            self.assertEqual(models[11]["intercept"], 11)
            self.assertEqual(models[11]["coef"][-1], controls.FEATURE_DIM - 1 + 11)

    def test_affine_drift_is_removed_by_moment_alignment(self):
        rng = np.random.default_rng(7)
        x_from = rng.normal(size=(3000, controls.FEATURE_DIM))
        multiplier = rng.uniform(0.5, 2.0, size=controls.FEATURE_DIM)
        shift = rng.normal(size=controls.FEATURE_DIM)
        x_to = x_from * multiplier + shift
        alignment = controls.alignment_from_moments(
            x_from.mean(axis=0), np.mean(x_from * x_from, axis=0),
            x_to.mean(axis=0), np.mean(x_to * x_to, axis=0),
        )
        x_aligned = x_to * alignment["multiplier"] + alignment["offset"]
        np.testing.assert_allclose(x_aligned, x_from, rtol=1e-10, atol=1e-10)
        np.testing.assert_allclose(x_aligned.mean(axis=0), x_from.mean(axis=0), atol=1e-11)
        np.testing.assert_allclose(x_aligned.std(axis=0), x_from.std(axis=0), atol=1e-11)

        beta = rng.normal(size=controls.FEATURE_DIM)
        raw_change = (x_to - x_from) @ beta
        aligned_change = (x_aligned - x_from) @ beta
        self.assertGreater(float(np.sqrt(np.mean(raw_change**2))), 1.0)
        self.assertLess(float(np.max(np.abs(aligned_change))), 1e-8)
        self.assertAlmostEqual(
            float(np.mean(raw_change)),
            float(np.sum(beta * (x_to.mean(axis=0) - x_from.mean(axis=0)))),
            places=10,
        )

    def test_shared_intercepts_cancel_but_own_layer_intercepts_do_not(self):
        x = np.zeros((4, controls.FEATURE_DIM))
        shared = {"intercept": 3.0, "coef": np.zeros(controls.FEATURE_DIM)}
        own_to = {"intercept": 7.5, "coef": np.zeros(controls.FEATURE_DIM)}
        raw_change = controls.predict_raw(x, shared) - controls.predict_raw(x, shared)
        own_change = controls.predict_raw(x, own_to) - controls.predict_raw(x, shared)
        np.testing.assert_allclose(raw_change, 0.0)
        np.testing.assert_allclose(own_change, 4.5)

    def test_pooled_sufficient_statistics_match_explicit_weighted_lstsq(self):
        rng = np.random.default_rng(19)
        protein_sizes = (70, 130)
        explicit_x = []
        explicit_y = []
        gram = np.zeros((controls.AUGMENTED_DIM, controls.AUGMENTED_DIM))
        cross = np.zeros(controls.AUGMENTED_DIM)
        true_beta = rng.normal(size=controls.FEATURE_DIM)
        for n_pairs in protein_sizes:
            for _endpoint in range(2):
                x = rng.normal(size=(n_pairs, controls.FEATURE_DIM))
                y = 2.3 + x @ true_beta + rng.normal(scale=0.01, size=n_pairs)
                augmented = np.column_stack((np.ones(n_pairs), x))
                gram += 0.5 * (augmented.T @ augmented) / n_pairs
                cross += 0.5 * (augmented.T @ y) / n_pairs
                row_scale = math.sqrt(0.5 / n_pairs)
                explicit_x.append(augmented * row_scale)
                explicit_y.append(y * row_scale)
        expected, _, _, _ = np.linalg.lstsq(
            np.vstack(explicit_x), np.concatenate(explicit_y), rcond=None
        )
        fitted = controls.solve_sufficient_statistics(gram, cross)
        actual = np.concatenate(([fitted["intercept"]], fitted["coef"]))
        np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)

    def test_streaming_error_and_distribution_statistics(self):
        y = np.asarray([1.0, 2.0, 3.0, 4.0])
        pred = np.asarray([1.5, 1.0, 2.5, 5.0])
        errors = controls.ErrorAccumulator()
        errors.update(y[:2], pred[:2])
        errors.update(y[2:], pred[2:])
        summary = errors.summary()
        self.assertAlmostEqual(summary["r2"], controls.safe_r2(y, pred))
        self.assertAlmostEqual(summary["mae"], float(np.mean(np.abs(pred - y))))
        distribution = controls.DistributionAccumulator(seed=3)
        distribution.update(pred[:2], classify=True)
        distribution.update(pred[2:], classify=True)
        stats = distribution.summary()
        self.assertAlmostEqual(stats["mean"], float(np.mean(pred)))
        self.assertAlmostEqual(stats["std"], float(np.std(pred)))

    def test_comparison_aggregates_pairwise_metrics(self):
        reference = np.asarray([-1.0, -0.2, 0.2, 1.0])
        control_values = np.asarray([-0.8, -0.1, 0.1, 1.2])
        accumulator = controls.ComparisonAccumulator()
        accumulator.update(control_values[:2], reference[:2])
        accumulator.update(control_values[2:], reference[2:])
        summary = accumulator.summary()
        self.assertAlmostEqual(
            summary["difference_in_mean_change"],
            float(np.mean(control_values) - np.mean(reference)),
        )
        self.assertAlmostEqual(
            summary["rmse"], float(np.sqrt(np.mean((control_values - reference) ** 2)))
        )
        self.assertEqual(summary["sign_agreement"], 1.0)

    def test_aggregator_rejects_missing_incomplete_and_schema_mismatched_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "missing metadata"):
                aggregate.validate_shards(root, required_files=())
            for layer_from, layer_to in aggregate.TRANSITIONS:
                directory = aggregate.shard_path(root, layer_from, layer_to)
                directory.mkdir(parents=True)
                (directory / "run_metadata.json").write_text(json.dumps({
                    "status": "complete", "layer_from": layer_from,
                    "layer_to": layer_to,
                    "predictive_variants": list(aggregate.PREDICTIVE_VARIANTS),
                }))
            validated = aggregate.validate_shards(root, required_files=())
            self.assertEqual(len(validated), 6)
            first = validated[0][2] / "example.csv"
            second = validated[1][2] / "example.csv"
            first.write_text("a,b\n1,2\n")
            second.write_text("a,c\n1,2\n")
            for _a, _b, directory, _metadata in validated[2:]:
                (directory / "example.csv").write_text("a,b\n1,2\n")
            with self.assertRaisesRegex(ValueError, "Schema mismatch"):
                aggregate.merge_csv(validated, "example.csv")


if __name__ == "__main__":
    unittest.main()

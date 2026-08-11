import csv
import importlib.util
import json
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


pilot = load_module("local_pair_update_probe_pilot", "local_pair_update_probe_pilot.py")
aggregate = load_module(
    "aggregate_local_pair_update_probe_pilot",
    "aggregate_local_pair_update_probe_pilot.py",
)


def toy_pair_block(n_residues=10):
    pb = np.empty((n_residues, n_residues, pilot.N_CHANNELS), dtype=np.float64)
    channels = np.arange(pilot.N_CHANNELS, dtype=np.float64)
    for a in range(n_residues):
        for b in range(n_residues):
            pb[a, b] = 1000.0 * a + 10.0 * b + channels
    return pb


class LocalPairUpdateProbeTests(unittest.TestCase):
    def test_task_mapping_is_transition_major_and_complete(self):
        specs = [pilot.task_spec(index) for index in range(15)]
        self.assertEqual(specs[0], (10, 11, "distance_seqsep"))
        self.assertEqual(specs[4], (10, 11, "triangle_r3"))
        self.assertEqual(specs[5], (20, 21, "distance_seqsep"))
        self.assertEqual(specs[-1], (30, 31, "triangle_r3"))
        self.assertEqual(len(set(specs)), 15)
        with self.assertRaises(ValueError):
            pilot.task_spec(15)

    def test_local_pairs_are_inclusive_and_have_expected_count(self):
        i, j, separation = pilot.local_pair_indices(16)
        self.assertTrue(np.all(i < j))
        self.assertEqual(separation.min(), 4)
        self.assertEqual(separation.max(), 12)
        self.assertIn((0, 4), set(zip(i.tolist(), j.tolist())))
        self.assertEqual(i.size, sum(16 - distance for distance in range(4, 13)))
        self.assertEqual(i.size, pilot.local_pair_count(16))

    def test_threshold_boundaries_are_no_change(self):
        values = np.array([-0.5001, -0.5, 0.0, 0.5, 0.5001])
        np.testing.assert_array_equal(pilot.class_labels(values), [-1, 0, 0, 0, 1])

    def test_feature_dimensions_and_schema_indices(self):
        expected = {
            "distance_seqsep": 2,
            "center": 128,
            "triangle_r1": 1156,
            "triangle_r2": 2184,
            "triangle_r3": 3212,
        }
        for variant, dimension in expected.items():
            schema = pilot.feature_schema(variant)
            self.assertEqual(pilot.feature_dim(variant), dimension)
            self.assertEqual(
                [row["feature_index"] for row in schema], list(range(dimension))
            )
            self.assertTrue(all(row["feature_dim"] == dimension for row in schema))

    def test_symmetric_edge_and_center_are_endpoint_invariant(self):
        pb = toy_pair_block()
        np.testing.assert_allclose(
            pilot.symmetric_edge(pb, np.array([2]), np.array([7])),
            pilot.symmetric_edge(pb, np.array([7]), np.array([2])),
        )
        sep = np.array([5], dtype=np.int16)
        forward = pilot.build_feature_chunk(
            pb, np.array([2]), np.array([7]), sep, "center"
        )
        reverse = pilot.build_feature_chunk(
            pb, np.array([7]), np.array([2]), sep, "center"
        )
        np.testing.assert_allclose(forward, reverse)

    def test_triangle_slots_include_both_legs_masks_boundaries_and_overlap(self):
        pb = toy_pair_block()
        i = np.array([0, 1], dtype=np.int64)
        j = np.array([4, 5], dtype=np.int64)
        sep = j - i
        x = pilot.build_feature_chunk(pb, i, j, sep, "triangle_r2")

        first_slot = pilot.N_CHANNELS
        first_mask = first_slot + 2 * pilot.N_CHANNELS
        self.assertEqual(x[0, first_mask], 0.0)
        self.assertEqual(x[1, first_mask], 1.0)
        np.testing.assert_allclose(
            x[1, first_slot:first_slot + pilot.N_CHANNELS],
            pilot.symmetric_edge(pb, np.array([1]), np.array([0]))[0],
        )
        np.testing.assert_allclose(
            x[1, first_slot + pilot.N_CHANNELS:first_mask],
            pilot.symmetric_edge(pb, np.array([0]), np.array([5]))[0],
        )

        slot_width = 2 * pilot.N_CHANNELS + 1
        shell2_start = pilot.N_CHANNELS + 4 * slot_width
        i_plus_slot = shell2_start + slot_width
        j_minus_slot = shell2_start + 2 * slot_width
        np.testing.assert_allclose(
            x[1, i_plus_slot:i_plus_slot + 2 * pilot.N_CHANNELS],
            x[1, j_minus_slot:j_minus_slot + 2 * pilot.N_CHANNELS],
        )
        self.assertEqual(x[1, i_plus_slot + 2 * pilot.N_CHANNELS], 1.0)
        self.assertEqual(x[1, j_minus_slot + 2 * pilot.N_CHANNELS], 1.0)

    def test_saved_distance_change_order_and_intercept_cancellation(self):
        rng = np.random.default_rng(7)
        pb_from = rng.normal(size=(9, 9, pilot.N_CHANNELS))
        pb_to = rng.normal(size=(9, 9, pilot.N_CHANNELS))
        i = np.array([0, 2, 3])
        j = np.array([4, 7, 8])
        coefficients = rng.normal(size=pilot.DISTANCE_FEATURE_DIM)
        intercept = 17.0
        distance_from, change = pilot.distance_change_and_current(
            pb_from, pb_to, i, j, intercept, coefficients
        )
        direct_from = (
            intercept
            + pb_from[j, i] @ coefficients[:pilot.N_CHANNELS]
            + pb_from[i, j] @ coefficients[pilot.N_CHANNELS:]
        )
        direct_to = (
            intercept
            + pb_to[j, i] @ coefficients[:pilot.N_CHANNELS]
            + pb_to[i, j] @ coefficients[pilot.N_CHANNELS:]
        )
        np.testing.assert_allclose(distance_from, direct_from)
        np.testing.assert_allclose(
            change, direct_to - direct_from, rtol=1e-12, atol=1e-12
        )

    def test_protein_equal_scaler_and_weights_ignore_protein_size(self):
        records = [
            ({"protein_id": "a"}, 0, 2),
            ({"protein_id": "b"}, 2, 6),
        ]
        x = np.array([[0.0], [2.0], [8.0], [8.0], [8.0], [8.0]])
        mean, scale = pilot.protein_equal_scaler(x, records)
        np.testing.assert_allclose(mean, [4.5])
        weights = pilot.protein_equal_weights(records)
        np.testing.assert_allclose(weights[:2].sum(), weights[2:].sum())
        self.assertTrue(np.isfinite(scale).all() and np.all(scale > 0))

    def test_small_synthetic_ridge_models_produce_finite_outputs(self):
        rng = np.random.default_rng(11)
        x = rng.normal(size=(300, 2))
        target = x[:, 0] - 0.5 * x[:, 1] + rng.normal(scale=0.05, size=300)
        labels = pilot.class_labels(target, threshold=0.25)
        weights = np.ones(300)
        alpha = pilot.RELATIVE_RIDGE_ALPHA * weights.sum()

        classifier = pilot.RidgeClassifier(
            alpha=alpha,
            solver="lsqr",
            tol=pilot.LSQR_TOLERANCE,
            max_iter=pilot.LSQR_MAX_ITER,
        ).fit(x, labels, sample_weight=weights)
        regressor = pilot.Ridge(
            alpha=alpha,
            solver="lsqr",
            tol=pilot.LSQR_TOLERANCE,
            max_iter=pilot.LSQR_MAX_ITER,
        ).fit(x, target, sample_weight=weights)
        self.assertTrue(np.isfinite(classifier.coef_).all())
        self.assertTrue(np.isfinite(regressor.coef_).all())
        self.assertGreater(pilot.balanced_accuracy(labels, classifier.predict(x)), 0.5)
        self.assertGreater(pilot.safe_r2(target, regressor.predict(x)), 0.9)

        records = [
            ({"protein_id": "p1", "source": "cameo", "n_residues": 20}, 0, 150),
            ({"protein_id": "p2", "source": "ecod", "n_residues": 20}, 150, 300),
        ]
        offsets = {"train": records, "test": records}
        labels_by_split = {"train": labels, "test": labels}
        class_predictions = {
            "train": classifier.predict(x),
            "test": classifier.predict(x),
        }
        class_summary, per_class_probe, class_metrics, matrix = pilot.classification_outputs(
            10, 11, "distance_seqsep", classifier, labels_by_split,
            class_predictions, offsets, alpha,
        )
        self.assertEqual(len(class_summary), 2)
        self.assertEqual(len(per_class_probe), 4)
        self.assertEqual(len(class_metrics), 6)
        self.assertEqual(len(matrix), 18)

        targets_by_split = {"train": target, "test": target}
        regression_predictions = {
            "train": regressor.predict(x),
            "test": regressor.predict(x),
        }
        regression_summary, per_regression_probe = pilot.regression_outputs(
            10, 11, "distance_seqsep", regressor, targets_by_split,
            regression_predictions, offsets, alpha,
        )
        self.assertEqual(len(regression_summary), 2)
        self.assertEqual(len(per_regression_probe), 4)
        self.assertTrue(np.isfinite(float(regression_summary[1]["pair_r2"])))

        class_coefficients = pilot.coefficient_rows(
            10, 11, "distance_seqsep", classifier, "classification", alpha
        )
        regression_coefficients = pilot.coefficient_rows(
            10, 11, "distance_seqsep", regressor, "regression", alpha
        )
        self.assertEqual(len(class_coefficients), 3 * (x.shape[1] + 1))
        self.assertEqual(len(regression_coefficients), x.shape[1] + 1)
        self.assertTrue(
            all(set(row).issubset(pilot.COEFFICIENT_FIELDS) for row in class_coefficients)
        )

    def test_aggregator_requires_all_15_complete_shards(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "missing metadata"):
                aggregate.validate_shards(root, required_tables=())

            for layer_from, layer_to, variant in aggregate.expected_specs():
                directory = aggregate.shard_path(root, layer_from, layer_to, variant)
                directory.mkdir(parents=True)
                (directory / "run_metadata.json").write_text(
                    json.dumps(
                        {
                            "status": "complete",
                            "layer_from": layer_from,
                            "layer_to": layer_to,
                            "variant": variant,
                        }
                    )
                )
            validated = aggregate.validate_shards(root, required_tables=())
            self.assertEqual(len(validated), 15)

    def test_csv_writer_emits_requested_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "table.csv"
            pilot.write_csv_atomic(path, ["a", "b"], [{"a": 1, "b": 2.5}])
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            self.assertEqual(rows, [{"a": "1", "b": "2.5"}])


if __name__ == "__main__":
    unittest.main()

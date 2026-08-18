import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from xgboost_search_common import (  # noqa: E402
    ANCHOR_LAYERS, PER_PROTEIN_WEIGHTING_MODE, SHARED_WEIGHTING_MODE,
    PooledStats, averaged_symmetric_features, build_shared_split,
    directional_base256_features, fit_xgboost, load_complete_search,
    load_master_manifest, map_search_task,
    map_shared_winner_task, map_winner_layer_task, nested_sample_indices,
    pair_split_indices, protein_equal_weights, r2_score,
    require_non_degenerate_iterations, select_winner,
    summarize_prediction_bundle,
)


class FeatureTests(unittest.TestCase):
    def test_averaged_symmetric_features_are_exactly_128_float64_channels(self):
        rng = np.random.default_rng(4)
        pair_block = rng.normal(size=(7, 7, 128)).astype(np.float32)
        pair_i, pair_j = np.triu_indices(7, k=1)
        actual = averaged_symmetric_features(pair_block, pair_i, pair_j)
        expected = 0.5 * (
            pair_block[pair_i, pair_j].astype(np.float64)
            + pair_block[pair_j, pair_i].astype(np.float64)
        )
        self.assertEqual(actual.shape, (21, 128))
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_allclose(actual, expected)

    def test_directional_base256_features_preserve_canonical_order(self):
        pair_block = np.arange(5 * 5 * 128, dtype=np.float32).reshape(5, 5, 128)
        pair_i, pair_j = np.tril_indices(5, k=-1)
        actual = directional_base256_features(pair_block, pair_i, pair_j)
        expected = np.concatenate(
            [pair_block[pair_i, pair_j], pair_block[pair_j, pair_i]], axis=1
        ).astype(np.float64)
        self.assertEqual(actual.shape, (10, 256))
        self.assertEqual(actual.dtype, np.float64)
        np.testing.assert_array_equal(actual, expected)
        with self.assertRaisesRegex(ValueError, "strict-lower-triangle"):
            directional_base256_features(pair_block, pair_j, pair_i)


class SplitAndSamplingTests(unittest.TestCase):
    @staticmethod
    def records():
        return [
            {
                "protein_id": entry.protein_id,
                "source": entry.source,
                "n_residues": entry.n_residues,
                "n_pairs": entry.n_pairs,
                "size_bin": entry.size_bin,
            }
            for entry in load_master_manifest()
            if entry.eligible
        ]

    def test_shared_split_is_deterministic_disjoint_stratified_exhaustive(self):
        split_a, assignments_a = build_shared_split(self.records())
        split_b, assignments_b = build_shared_split(self.records())
        self.assertEqual(assignments_a, assignments_b)
        ids = [{r["protein_id"] for r in split_a[name]} for name in ("fit", "validation", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(len(set.union(*ids)), 552)
        self.assertEqual(
            [len(split_a[name]) for name in ("fit", "validation", "test")],
            [330, 111, 111],
        )
        self.assertEqual({row["split"] for row in assignments_a}, {
            "fit", "validation", "test",
        })

    def test_shared_split_rejects_incomplete_cohort(self):
        with self.assertRaisesRegex(RuntimeError, "frozen canonical manifest"):
            build_shared_split(self.records()[:-1])

    def test_pair_split_is_deterministic_disjoint_exhaustive_and_layer_independent(self):
        split_a = pair_split_indices(103, "protein-A")
        split_b = pair_split_indices(103, "protein-A")
        for layer in ANCHOR_LAYERS:
            del layer
            for name in split_a:
                np.testing.assert_array_equal(split_a[name], split_b[name])
        sets = [set(split_a[name]) for name in ("fit", "validation", "test")]
        self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2])
        self.assertEqual(set.union(*sets), set(range(103)))

    def test_protein_equal_weights_give_equal_total_mass(self):
        ids = ["a"] * 2 + ["b"] * 5 + ["c"] * 11
        weights = protein_equal_weights(ids)
        totals = [weights[np.asarray(ids) == protein].sum() for protein in ("a", "b", "c")]
        np.testing.assert_allclose(totals, [1.0, 1.0, 1.0])

    def test_nested_samples_are_deterministic_and_nested(self):
        indices = np.arange(30000)
        sample_5k = nested_sample_indices(indices, "p", 5000)
        sample_10k = nested_sample_indices(indices, "p", 10000)
        sample_20k = nested_sample_indices(indices, "p", 20000)
        np.testing.assert_array_equal(sample_10k, nested_sample_indices(indices, "p", 10000))
        self.assertTrue(set(sample_5k).issubset(sample_10k))
        self.assertTrue(set(sample_10k).issubset(sample_20k))

    def test_task_mappings(self):
        self.assertEqual([map_search_task(i) for i in range(3)], [0, 23, 47])
        self.assertEqual([map_winner_layer_task(i) for i in range(3)], [0, 23, 47])
        self.assertEqual(
            [map_shared_winner_task(i) for i in range(9)],
            [(layer, cap, label) for layer in (0, 23, 47) for cap, label in zip((5000, 10000, 20000), ("5k", "10k", "20k"))],
        )
        with self.assertRaises(ValueError):
            map_shared_winner_task(9)


class WeightingTests(unittest.TestCase):
    class FakeRegressor:
        instances = []

        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.fit_kwargs = None
            self.best_iteration = 7
            self.best_score = 1.25
            self.__class__.instances.append(self)

        def fit(self, x, y, **kwargs):
            self.fit_kwargs = kwargs
            return self

    @staticmethod
    def bundle(ids):
        n = len(ids)
        return {
            "x": np.zeros((n, 128), dtype=np.float64),
            "y": np.arange(n, dtype=np.float64),
            "protein_ids": ids,
        }

    def setUp(self):
        self.FakeRegressor.instances.clear()

    def test_per_protein_fit_omits_all_sample_weights(self):
        train = self.bundle(["p"] * 12)
        validation = self.bundle(["p"] * 4)
        with patch(
            "xgboost_search_common.require_xgboost",
            return_value=self.FakeRegressor,
        ):
            _, info = fit_xgboost(
                {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 5},
                train,
                validation,
                2,
                PER_PROTEIN_WEIGHTING_MODE,
            )
        kwargs = self.FakeRegressor.instances[-1].fit_kwargs
        self.assertNotIn("sample_weight", kwargs)
        self.assertNotIn("sample_weight_eval_set", kwargs)
        self.assertEqual(info["weighting_mode"], "uniform_rows")

    def test_shared_fit_uses_equal_protein_mass_for_train_and_validation(self):
        train = self.bundle(["a"] * 2 + ["b"] * 5)
        validation = self.bundle(["c"] * 3 + ["d"] * 6)
        with patch(
            "xgboost_search_common.require_xgboost",
            return_value=self.FakeRegressor,
        ):
            fit_xgboost(
                {"max_depth": 3, "learning_rate": 0.1, "min_child_weight": 5},
                train,
                validation,
                2,
                SHARED_WEIGHTING_MODE,
            )
        kwargs = self.FakeRegressor.instances[-1].fit_kwargs
        train_weights = kwargs["sample_weight"]
        validation_weights = kwargs["sample_weight_eval_set"][0]
        np.testing.assert_allclose(
            [train_weights[:2].sum(), train_weights[2:].sum()], [1.0, 1.0]
        )
        np.testing.assert_allclose(
            [validation_weights[:3].sum(), validation_weights[3:].sum()],
            [1.0, 1.0],
        )

    def test_all_iteration_zero_configuration_fails(self):
        with self.assertRaisesRegex(RuntimeError, "iteration 0"):
            require_non_degenerate_iterations("P0", [0] * 50, 50)
        require_non_degenerate_iterations("P0", [0] * 49 + [1], 50)


class SelectionAndMetricTests(unittest.TestCase):
    def test_selection_ignores_test_and_applies_ties_in_order(self):
        configs = [
            {"config_id": "A", "max_depth": 6, "learning_rate": 0.1, "min_child_weight": 1},
            {"config_id": "B", "max_depth": 4, "learning_rate": 0.1, "min_child_weight": 1},
            {"config_id": "C", "max_depth": 4, "learning_rate": 0.1, "min_child_weight": 1},
        ]
        rows = {}
        for layer in ANCHOR_LAYERS:
            rows[layer] = [
                {"config_id": "A", "validation_mean_per_protein_r2": 0.8000, "fit_seconds": 1, "test_mean_per_protein_r2": 999},
                {"config_id": "B", "validation_mean_per_protein_r2": 0.7995, "fit_seconds": 5, "train_mean_per_protein_r2": -9999, "test_mean_per_protein_r2": -999},
                {"config_id": "C", "validation_mean_per_protein_r2": 0.7995, "fit_seconds": 8, "test_mean_per_protein_r2": 9999},
            ]
        winner, _, provenance = select_winner(rows, configs)
        self.assertEqual(winner["config_id"], "B")  # lower depth, then shorter runtime
        self.assertFalse(provenance["training_metrics_used"])
        self.assertFalse(provenance["test_metrics_used"])
        for layer in ANCHOR_LAYERS:
            rows[layer][2]["fit_seconds"] = 5
        winner, _, _ = select_winner(rows, configs)
        self.assertEqual(winner["config_id"], "B")  # final tie: lower ID

    def test_streaming_pooled_r2_matches_direct_concatenation(self):
        y_parts = [np.array([1.0, 2.0]), np.array([4.0, 8.0, 16.0])]
        p_parts = [np.array([1.2, 1.8]), np.array([3.0, 9.0, 14.0])]
        stats = PooledStats()
        for y, pred in zip(y_parts, p_parts):
            stats.update(y, pred)
        direct = r2_score(np.concatenate(y_parts), np.concatenate(p_parts))
        self.assertAlmostEqual(stats.metrics()["r2"], direct, places=14)

    def test_regression_rows_and_summary_use_reference_style_schema(self):
        records = [
            {"protein_id": "a", "source": "synthetic", "n_residues": 3,
             "size_bin": 0},
            {"protein_id": "b", "source": "synthetic", "n_residues": 4,
             "size_bin": 1},
        ]
        y = np.array([1.0, 2.0, 4.0, 8.0, 16.0])
        pred = np.array([1.1, 1.9, 3.0, 9.0, 15.0])
        bundle = {
            "y": y,
            "offsets": [
                (records[0], np.array([0, 1]), 0, 2),
                (records[1], np.array([0, 1, 2]), 2, 5),
            ],
        }
        rows, summary = summarize_prediction_bundle(
            bundle,
            pred,
            "fit",
            "sampled_training_rows",
            {"variant": "test", "config_id": "S0", "layer": 0},
        )
        self.assertEqual([row["split"] for row in rows], ["train", "train"])
        self.assertTrue(all(row["evaluation_scope"] == "sampled_training_rows" for row in rows))
        for column in (
            "target_variance", "per_protein_r2", "per_protein_mse",
            "per_protein_mae", "per_protein_rmse", "mean_true_distance",
            "mean_predicted_distance", "mean_error_pred_minus_true",
        ):
            self.assertIn(column, rows[0])
        self.assertEqual(summary["split"], "train")
        self.assertAlmostEqual(summary["pair_weighted_r2"], r2_score(y, pred))


class ArtifactValidationTests(unittest.TestCase):
    CONFIGS = ({"config_id": "X", "max_depth": 1},)

    @staticmethod
    def write_layer(root: Path, layer: int, digest: str, include_row: bool = True):
        layer_dir = root / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True)
        with (layer_dir / "search_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=[
                "config_id", "train_mean_per_protein_r2", "train_pooled_r2",
                "validation_mean_per_protein_r2", "validation_pooled_r2",
                "fit_seconds", "split_digest",
            ])
            writer.writeheader()
            if include_row:
                writer.writerow({
                    "config_id": "X", "train_mean_per_protein_r2": 0.3,
                    "train_pooled_r2": 0.3,
                    "validation_mean_per_protein_r2": 0.2,
                    "validation_pooled_r2": 0.2,
                    "fit_seconds": 1,
                    "split_digest": digest,
                })
        (layer_dir / "search_metadata.json").write_text(json.dumps({"layer": layer, "split_digest": digest, "status": "complete"}))

    def test_missing_incomplete_and_inconsistent_artifacts_fail(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with self.assertRaises(FileNotFoundError):
                load_complete_search(root, self.CONFIGS)
            for layer in ANCHOR_LAYERS:
                self.write_layer(root, layer, "same", include_row=layer != 23)
            with self.assertRaises(RuntimeError):
                load_complete_search(root, self.CONFIGS)
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for layer in ANCHOR_LAYERS:
                self.write_layer(root, layer, f"digest-{layer}")
            with self.assertRaises(RuntimeError):
                load_complete_search(root, self.CONFIGS)

    def test_winner_rejects_search_with_old_weighting_mode(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for layer in ANCHOR_LAYERS:
                self.write_layer(root, layer, "same")
            with self.assertRaisesRegex(RuntimeError, "weighting mode"):
                load_complete_search(
                    root, self.CONFIGS, PER_PROTEIN_WEIGHTING_MODE
                )

    def test_search_loader_rejects_any_locked_test_column(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            for layer in ANCHOR_LAYERS:
                self.write_layer(root, layer, "same")
                path = root / f"layer_{layer:02d}" / "search_summary.csv"
                with path.open() as handle:
                    rows = list(csv.DictReader(handle))
                with path.open("w", newline="") as handle:
                    fields = list(rows[0]) + ["test_pooled_r2"]
                    writer = csv.DictWriter(handle, fieldnames=fields)
                    writer.writeheader()
                    writer.writerow({**rows[0], "test_pooled_r2": 0.1})
            with self.assertRaisesRegex(RuntimeError, "exposes locked-test fields"):
                load_complete_search(root, self.CONFIGS)


if __name__ == "__main__":
    unittest.main()

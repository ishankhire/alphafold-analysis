import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

from xgboost_search_common import (  # noqa: E402
    ANCHOR_LAYERS, PooledStats, averaged_symmetric_features,
    build_shared_split, load_complete_search, map_search_task,
    map_shared_winner_task, map_winner_layer_task, nested_sample_indices,
    pair_split_indices, protein_equal_weights, r2_score, select_winner,
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


class SplitAndSamplingTests(unittest.TestCase):
    @staticmethod
    def records():
        return [
            {"protein_id": f"p{size_bin}_{i:02d}", "source": "synthetic", "n_residues": 20 + i,
             "n_pairs": 100 + i, "size_bin": size_bin}
            for size_bin in range(5) for i in range(10)
        ]

    def test_shared_split_is_deterministic_disjoint_stratified_exhaustive(self):
        split_a, assignments_a = build_shared_split(self.records())
        split_b, assignments_b = build_shared_split(self.records())
        self.assertEqual(assignments_a, assignments_b)
        ids = [{r["protein_id"] for r in split_a[name]} for name in ("fit", "validation", "test")]
        self.assertFalse(ids[0] & ids[1] or ids[0] & ids[2] or ids[1] & ids[2])
        self.assertEqual(len(set.union(*ids)), 50)
        for split_name, expected_per_bin in (("fit", 6), ("validation", 2), ("test", 2)):
            counts = [sum(r["size_bin"] == b for r in split_a[split_name]) for b in range(5)]
            self.assertEqual(counts, [expected_per_bin] * 5)

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
            [map_shared_winner_task(i) for i in range(12)],
            [(layer, cap, label) for layer in (0, 23, 47) for cap, label in zip((5000, 10000, 20000, None), ("5k", "10k", "20k", "all"))],
        )


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
                {"config_id": "B", "validation_mean_per_protein_r2": 0.7995, "fit_seconds": 5, "test_mean_per_protein_r2": -999},
                {"config_id": "C", "validation_mean_per_protein_r2": 0.7995, "fit_seconds": 8, "test_mean_per_protein_r2": 9999},
            ]
        winner, _, provenance = select_winner(rows, configs)
        self.assertEqual(winner["config_id"], "B")  # lower depth, then shorter runtime
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


class ArtifactValidationTests(unittest.TestCase):
    CONFIGS = ({"config_id": "X", "max_depth": 1},)

    @staticmethod
    def write_layer(root: Path, layer: int, digest: str, include_row: bool = True):
        layer_dir = root / f"layer_{layer:02d}"
        layer_dir.mkdir(parents=True)
        with (layer_dir / "search_summary.csv").open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=["config_id", "validation_mean_per_protein_r2", "fit_seconds", "split_digest"])
            writer.writeheader()
            if include_row:
                writer.writerow({"config_id": "X", "validation_mean_per_protein_r2": 0.2, "fit_seconds": 1, "split_digest": digest})
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


if __name__ == "__main__":
    unittest.main()

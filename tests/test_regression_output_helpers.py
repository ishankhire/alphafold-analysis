import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = ROOT / "scripts" / "cluster_scripts"
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import regression_output_helpers as outputs
import regression_protocol as protocol


class RegressionMetricTests(unittest.TestCase):
    def test_hand_calculated_metrics_and_negative_r2(self):
        stats = outputs.RegressionStats()
        stats.update(np.asarray([1.0, 2.0, 3.0]), np.asarray([2.0, 2.0, 2.0]))
        metrics = stats.metrics()
        self.assertAlmostEqual(metrics["r2"], 0.0)
        self.assertAlmostEqual(metrics["mae"], 2.0 / 3.0)
        self.assertAlmostEqual(metrics["rmse"], math.sqrt(2.0 / 3.0))
        self.assertAlmostEqual(metrics["bias_pred_minus_true"], 0.0)

        poor = outputs.RegressionStats()
        poor.update(np.asarray([0.0, 1.0]), np.asarray([10.0, 10.0]))
        self.assertLess(poor.metrics()["r2"], 0.0)
        constant = outputs.RegressionStats()
        constant.update(np.asarray([2.0, 2.0]), np.asarray([2.0, 3.0]))
        self.assertTrue(math.isnan(constant.metrics()["r2"]))

    def test_merge_matches_single_pass(self):
        left = outputs.RegressionStats()
        left.update(np.asarray([1.0, 2.0]), np.asarray([1.5, 2.5]))
        right = outputs.RegressionStats()
        right.update(np.asarray([3.0]), np.asarray([2.0]))
        left.merge(right)
        direct = outputs.RegressionStats()
        direct.update(np.asarray([1.0, 2.0, 3.0]), np.asarray([1.5, 2.5, 2.0]))
        self.assertEqual(left.n, direct.n)
        for key in direct.metrics():
            self.assertAlmostEqual(left.metrics()[key], direct.metrics()[key])

    def test_partial_per_protein_row_has_canonical_schema(self):
        row = outputs.build_per_protein_row(
            analysis_name="probe", model_family="gated_ols", feature_dim=256, layer=4,
            protein_id="p1", source="cameo", protein_split="test",
            evaluation_scope="all_pairs", n_residues=3,
            y_true=np.asarray([1.0, 2.0, 3.0]), predictions=np.asarray([1.0, np.nan, 4.0]),
        )
        self.assertEqual(tuple(row), outputs.PER_PROTEIN_COLUMNS)
        self.assertEqual(row["prediction_status"], "partial_coverage")
        self.assertEqual(row["n_expected_pairs"], 3)
        self.assertEqual(row["n_evaluated_pairs"], 2)
        self.assertAlmostEqual(row["pair_coverage"], 2.0 / 3.0)

    def test_pooled_model_and_baselines_use_identical_rows(self):
        accumulator = outputs.PooledComparisonAccumulator()
        accumulator.update_protein(
            "p1", np.asarray([1.0, 2.0, 3.0]), np.asarray([1.0, np.nan, 4.0]), 2.0,
            conditional_baseline_predictions=1.5,
        )
        row = outputs.build_pooled_metrics_row(
            accumulator, analysis_name="probe", model_family="gated_ols", feature_dim=256,
            layer=0, protein_split="test", evaluation_scope="all_pairs",
            conditional_baseline_type="gate_development_mean",
        )
        self.assertEqual(tuple(row), outputs.POOLED_METRICS_COLUMNS)
        self.assertEqual(row["n_expected_pairs"], 3)
        self.assertEqual(row["n_evaluated_pairs"], 2)
        self.assertEqual(accumulator.global_baseline.n, 2)
        self.assertEqual(accumulator.conditional_baseline.n, 2)

    def test_nonfinite_targets_are_rejected(self):
        accumulator = outputs.PooledRegressionAccumulator()
        with self.assertRaises(ValueError):
            accumulator.update_protein("p", np.asarray([1.0, np.nan]), np.asarray([1.0, 2.0]))


class SubgroupAndGeometryTests(unittest.TestCase):
    def test_subgroups_emit_all_categories_and_support_status(self):
        accumulator = outputs.SubgroupAccumulator("toy", ("large", "small", "empty"))
        for index in range(5):
            y = np.arange(21, dtype=np.float64) + index
            pred = y + 0.5
            codes = np.asarray([0] * 20 + [1], dtype=np.int16)
            accumulator.update_protein(f"p{index}", y, pred, codes)
        rows = accumulator.rows(
            analysis_name="probe", model_family="shared_ols", feature_dim=256, layer=0,
            protein_split="test", evaluation_scope="all_pairs",
        )
        self.assertEqual([row["category_label"] for row in rows], ["large", "small", "empty"])
        self.assertEqual(rows[0]["support_status"], "sufficient")
        self.assertEqual(rows[1]["support_status"], "low_pair_support")
        self.assertEqual(rows[2]["support_status"], "empty")
        self.assertTrue(math.isnan(rows[2]["pair_coverage"]))

    def test_geometry_complete_and_incomplete_rows(self):
        coordinates = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
        pair_i, pair_j = protocol.strict_lower_triangle_indices(3)
        distances = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
        complete = outputs.build_geometry_row(
            analysis_name="probe", model_family="shared_ols", feature_dim=256, layer=0,
            protein_id="p1", source="cameo", protein_split="test", n_residues=3,
            true_coordinates=coordinates, pair_i=pair_i, pair_j=pair_j, predictions=distances,
        )
        self.assertEqual(tuple(complete), outputs.GEOMETRY_METRICS_COLUMNS)
        self.assertEqual(complete["geometry_status"], "complete")
        self.assertEqual(complete["n_triangles"], 1)
        self.assertNotIn("triangle_no_violation_fraction", complete)

        incomplete_predictions = distances.copy()
        incomplete_predictions[0] = np.nan
        incomplete = outputs.build_geometry_row(
            analysis_name="probe", model_family="shared_ols", feature_dim=256, layer=0,
            protein_id="p2", source="ecod", protein_split="test", n_residues=3,
            true_coordinates=coordinates, pair_i=pair_i, pair_j=pair_j,
            predictions=incomplete_predictions,
        )
        self.assertEqual(incomplete["geometry_status"], "incomplete_pair_coverage")
        self.assertTrue(math.isnan(incomplete["rmsd"]))

        summaries = outputs.summarize_geometry_rows([complete, incomplete])
        self.assertEqual(len(summaries), len(outputs.GEOMETRY_METRIC_NAMES))
        self.assertEqual(tuple(summaries[0]), outputs.GEOMETRY_SUMMARY_COLUMNS)
        self.assertEqual(summaries[0]["n_proteins_attempted"], 2)
        self.assertEqual(summaries[0]["n_proteins_included"], 1)


class SchemaAndWriterTests(unittest.TestCase):
    def test_historical_and_fivefold_schema_versions_are_distinct(self):
        self.assertEqual(outputs.OUTPUT_SCHEMA_VERSION, "1.0")
        self.assertEqual(outputs.FIVEFOLD_OUTPUT_SCHEMA_VERSION, "2.0")

    def test_status_runtime_and_manifest_schemas(self):
        status = outputs.build_model_status_row(
            analysis_name="probe", model_family="shared_ols", feature_dim=256,
            layer=0, fit_status="complete", n_training_proteins=330, n_training_pairs=1000,
        )
        runtime = outputs.build_runtime_row(
            analysis_name="probe", model_family="shared_ols", layer=0,
            completion_status="complete", n_fitted_proteins=330, n_evaluated_proteins=552,
            n_fitted_pairs=1000, n_evaluated_pairs=2000, elapsed_seconds=12.5,
        )
        self.assertEqual(tuple(status), outputs.MODEL_STATUS_COLUMNS)
        self.assertEqual(tuple(runtime), outputs.RUNTIME_COLUMNS)
        manifest = outputs.build_run_manifest(
            analysis_name="probe", model_family="shared_ols", feature_dim=256,
            requested_layers=[1, 0, 1], completed_layers=[0], configuration={"alpha": 1},
            created_at="2026-01-01T00:00:00+00:00",
        )
        self.assertEqual(manifest["requested_layers"], [0, 1])
        self.assertEqual(manifest["master_split_digest"], protocol.MASTER_MANIFEST_DIGEST)
        self.assertEqual(manifest["configuration_digest"], outputs.semantic_digest({"alpha": 1}))

    def test_atomic_writers_round_trip_and_reject_bad_schema(self):
        row = outputs.build_runtime_row(
            analysis_name="probe", model_family="shared_ols", layer=0,
            completion_status="complete", n_fitted_proteins=1, n_evaluated_proteins=1,
            n_fitted_pairs=3, n_evaluated_pairs=3, elapsed_seconds=1.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            csv_path = root / "runtime.csv"
            outputs.atomic_write_csv(csv_path, outputs.RUNTIME_COLUMNS, [row])
            with csv_path.open(newline="") as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(loaded[0]["peak_rss_mb"], "NaN")

            json_path = root / "manifest.json"
            outputs.atomic_write_json(json_path, {"ok": True})
            self.assertEqual(json.loads(json_path.read_text()), {"ok": True})
            with self.assertRaises(ValueError):
                outputs.atomic_write_json(json_path, {"bad": float("nan")})

            npz_path = root / "arrays.npz"
            outputs.atomic_savez_compressed(npz_path, values=np.arange(4))
            with np.load(npz_path) as archive:
                np.testing.assert_array_equal(archive["values"], np.arange(4))

            with self.assertRaises(ValueError):
                outputs.atomic_write_csv(root / "bad.csv", outputs.RUNTIME_COLUMNS, [{"extra": 1}])


if __name__ == "__main__":
    unittest.main()

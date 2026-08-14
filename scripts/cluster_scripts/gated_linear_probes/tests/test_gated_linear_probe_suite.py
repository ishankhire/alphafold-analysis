import math
import importlib.util
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SUITE_DIR = Path(__file__).parents[1]
CLUSTER_DIR = SUITE_DIR.parent
GENERAL_DIR = CLUSTER_DIR / "generalized_model_ecod"
for path in (SUITE_DIR, CLUSTER_DIR, GENERAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cluster_regression_helpers as helpers
import aggregate_gated_linear_probe_suite as aggregate
import gated_linear_probe_common as common
import general_train_test_common as general_common

DIAGNOSTIC_PATH = CLUSTER_DIR / "diagnostics" / "diagnostic_sequence_separation_gated_probes.py"
DIAGNOSTIC_SPEC = importlib.util.spec_from_file_location("existing_sequence_gated", DIAGNOSTIC_PATH)
existing_sequence_gated = importlib.util.module_from_spec(DIAGNOSTIC_SPEC)
DIAGNOSTIC_SPEC.loader.exec_module(existing_sequence_gated)


class CategoryAndMappingTests(unittest.TestCase):
    def test_standard_aa_pairs_are_stable_and_complete(self):
        self.assertEqual(len(helpers.STANDARD_AA_PAIR_LABELS), 210)
        self.assertEqual(len(set(helpers.STANDARD_AA_PAIR_LABELS)), 210)
        self.assertEqual(helpers.STANDARD_AA_PAIR_LABELS[0], "ALA-ALA")
        self.assertEqual(helpers.STANDARD_AA_PAIR_LABELS[-1], "VAL-VAL")
        residues = [("A", "GLY", 1), ("A", "ALA", 2), ("A", "MSE", 3)]
        codes = helpers.build_unordered_aa_pair_codes(residues, [0, 0], [1, 2])
        self.assertEqual(codes[0], helpers.AA_PAIR_TO_CODE["ALA-GLY"])
        self.assertEqual(codes[1], -1)

    def test_bin_boundaries_and_explicit_invalid_codes(self):
        sequence = common.sequence_bin_codes(
            [0, 1, 2, 4, 5, 9, 10, 19, 20, 49, 50],
            [True, True, True, True, True, True, True, True, True, True, False],
        )
        np.testing.assert_array_equal(sequence, [-1, 0, 1, 1, 2, 2, 3, 3, 4, 4, -1])
        distance = common.distance_bin_codes([0, 4.999, 5, 9.999, 10, 19.999, 20, 39.999, 40])
        np.testing.assert_array_equal(distance, [0, 0, 1, 1, 2, 2, 3, 3, 4])

    def test_array_task_mappings(self):
        self.assertEqual([common.layer_for_task(i, "production") for i in range(6)], list(common.LAYERS))
        self.assertEqual(common.layer_for_task(0, "smoke"), 0)
        self.assertEqual(common.joint_task_for_id(0, "production"), (0, 0))
        self.assertEqual(common.joint_task_for_id(35, "production"), (47, 5))
        self.assertEqual(common.joint_task_for_id(5, "smoke"), (0, 5))
        with self.assertRaises(ValueError):
            common.joint_task_for_id(6, "smoke")

    def test_split_is_deterministic_disjoint_and_stratified(self):
        def make_records():
            records = [{"protein_id": f"p{i:03d}", "n_residues": i + 20,
                        "n_pairs": 10, "target_variance": 1.0} for i in range(50)]
            general_common.assign_size_bins(records, 5)
            return records
        first = general_common.build_strict_stratified_train_test_split(make_records())
        second = general_common.build_strict_stratified_train_test_split(make_records())
        first_rows, second_rows = first[2], second[2]
        self.assertEqual(first_rows, second_rows)
        train = {row["protein_id"] for row in first_rows if row["split"] == "train"}
        test = {row["protein_id"] for row in first_rows if row["split"] == "test"}
        self.assertFalse(train & test)
        self.assertEqual(len(train | test), 50)
        self.assertEqual({row["size_bin"] for row in first_rows if row["split"] == "test"}, set(range(5)))


class NumericalTests(unittest.TestCase):
    def test_directional_extractor_is_float64_and_ordered(self):
        pb = np.arange(4 * 4 * 128, dtype=np.float32).reshape(4, 4, 128)
        extracted = common.extract_base256(pb, [2, 3], [0, 1])
        self.assertEqual(extracted.dtype, np.float64)
        np.testing.assert_array_equal(extracted[0, :128], pb[2, 0])
        np.testing.assert_array_equal(extracted[0, 128:], pb[0, 2])

    def test_protein_equal_weights_assign_equal_mass(self):
        offsets = [(0, 0, 2), (1, 2, 7), (2, 7, 8)]
        weights = common.protein_equal_weights(offsets, 8)
        masses = [weights[start:end].sum() for _, start, end in offsets]
        np.testing.assert_allclose(masses, [8 / 3] * 3)

    def test_scaling_and_weights_match_existing_sequence_gated_diagnostic(self):
        rng = np.random.default_rng(8)
        chunks = [
            ({"protein_id": "a"}, rng.normal(size=(3, 4)).astype(np.float32), np.arange(3.0)),
            ({"protein_id": "b"}, rng.normal(size=(7, 4)).astype(np.float32), np.arange(7.0)),
        ]
        old_mean, old_scale, _ = existing_sequence_gated.compute_protein_equal_scaler(chunks)
        x = np.vstack([chunk[1] for chunk in chunks]).astype(np.float64)
        offsets = [(0, 0, 3), (1, 3, 10)]
        new_mean, new_scale = common.weighted_scaler(x, offsets)
        np.testing.assert_allclose(new_mean, old_mean)
        np.testing.assert_allclose(new_scale, old_scale)
        _, _, old_weights, _ = existing_sequence_gated.materialize_chunks(chunks, old_mean, old_scale)
        np.testing.assert_allclose(common.protein_equal_weights(offsets, 10), old_weights)

    def test_float64_ols_recovers_full_rank_predictions(self):
        rng = np.random.default_rng(12)
        x = rng.normal(size=(320, common.FEATURE_DIM)).astype(np.float64)
        beta = rng.normal(size=common.FEATURE_DIM)
        y = (x @ beta + 0.7).astype(np.float64)
        offsets = [(i, i * 64, (i + 1) * 64) for i in range(5)]
        with mock.patch.object(
            common.np.linalg,
            "matrix_rank",
            side_effect=AssertionError("fit_ols must reuse sklearn's reported rank"),
        ):
            fitted = common.fit_ols(x, y, offsets)
        prediction = fitted["model"].predict((x - fitted["mean"]) / fitted["scale"])
        self.assertEqual(fitted["coef_raw"].dtype, np.float64)
        self.assertLess(np.max(np.abs(prediction - y)), 1e-8)
        self.assertEqual(fitted["fit_status"], "fitted")

    def test_subset_materialized_preserves_protein_slices(self):
        materialized = {
            "y": np.arange(9, dtype=np.float64),
            "offsets": [(2, 0, 3), (5, 3, 7), (8, 7, 9)],
        }
        indices, y, offsets = common.subset_materialized(
            materialized,
            np.asarray([False, True, True, False, True, False, True, True, False]),
        )
        np.testing.assert_array_equal(indices, [1, 2, 4, 6, 7])
        np.testing.assert_array_equal(y, [1.0, 2.0, 4.0, 6.0, 7.0])
        self.assertEqual(offsets, [(2, 0, 2), (5, 2, 4), (8, 4, 5)])

    def test_eligibility_precedence_and_rank_flags(self):
        status, flags = common.gate_eligibility(10, 1)
        self.assertEqual(status, "insufficient_train_rows")
        self.assertEqual(flags, ["insufficient_train_rows", "insufficient_train_proteins"])
        self.assertEqual(common.gate_eligibility(300, 2)[0], "insufficient_train_proteins")
        rng = np.random.default_rng(4)
        x = rng.normal(size=(300, common.FEATURE_DIM)).astype(np.float64)
        x[:, -1] = x[:, 0]
        y = rng.normal(size=300).astype(np.float64)
        fitted = common.fit_ols(x, y, [(i, i * 60, (i + 1) * 60) for i in range(5)])
        self.assertEqual(fitted["fit_status"], "fitted_rank_deficient")
        self.assertIn("rank_deficient", fitted["flags"])
        self.assertIn("ill_conditioned", fitted["flags"])

    def test_additive_moments_and_constant_targets(self):
        y = np.asarray([1.0, 2.0, 4.0, 7.0])
        pred = np.asarray([1.5, 1.0, 5.0, 6.0])
        direct = common.metrics_from_moments(common.cell_moments(y, pred))
        self.assertAlmostEqual(direct["mae"], np.mean(np.abs(pred - y)))
        self.assertAlmostEqual(direct["rmse"], np.sqrt(np.mean((pred - y) ** 2)))
        pieces = common.cell_moments(y[:2], pred[:2]) + common.cell_moments(y[2:], pred[2:])
        replay = common.metrics_from_moments(pieces)
        for key in ("r2", "mae", "rmse", "mean", "variance"):
            self.assertAlmostEqual(direct[key], replay[key])
        constant = common.metrics_from_moments(common.cell_moments(np.ones(5), np.ones(5)))
        self.assertTrue(math.isnan(constant["r2"]))
        self.assertEqual(constant["mae"], 0.0)

    def test_zero_baseline_skill_is_flagged(self):
        identifiers = {
            "analysis_name": "test", "layer": 0, "split": "test",
            "train_gate_aa_pair": "all", "train_gate_sequence_bin": "all",
            "evaluation_scope": "gate_overall", "eval_aa_pair": "all",
            "eval_sequence_bin": "all", "eval_distance_bin": "all",
        }
        row, _ = common.summarize_cell(
            np.ones(5), np.ones(5), 1.0, [(0, 0, 5)], identifiers,
            "fitted", [], 300, 5,
        )
        self.assertTrue(math.isnan(row["pooled_mae_skill"]))
        self.assertIn("zero_baseline_mae", row["support_flags"])
        self.assertIn("sparse_test_support", row["support_flags"])


class SchemaAndArtifactTests(unittest.TestCase):
    @staticmethod
    def materialized_fixture():
        return {
            "y": np.asarray([2.0, 3.0, 4.0, 5.0, 6.0]),
            "aa": np.asarray([0, 1, 2, 3, 4]),
            "sequence": np.asarray([0, 1, 2, 3, 4]),
            "distance": np.asarray([0, 1, 2, 3, 4]),
            "offsets": [(0, 0, 5)],
        }

    def test_analysis_scopes_do_not_emit_unrequested_cross_products(self):
        fixture = self.materialized_fixture()
        prediction = fixture["y"] + 0.1
        counts = {}
        for analysis in ("analysis_1_sequence_gated", "analysis_2_aa_pair_gated",
                         "analysis_4_aa_sequence_gated"):
            rows, audit = common.evaluate_materialized(
                fixture, prediction, 4.0, analysis, 0, "test", None, None,
                "fitted", [], 300, 5,
            )
            counts[analysis] = len(rows)
            self.assertTrue(all(
                not (row["eval_aa_pair"] != "all" and row["eval_sequence_bin"] != "all"
                     and row["eval_distance_bin"] != "all") for row in rows
            ))
            self.assertEqual(audit.shape[1], 8)
        self.assertEqual(counts["analysis_1_sequence_gated"], 216)
        self.assertEqual(counts["analysis_2_aa_pair_gated"], 12)
        self.assertEqual(counts["analysis_4_aa_sequence_gated"], 6)

    def test_atomic_npz_round_trip_and_duplicate_rejection(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coefficient.npz"
            common.atomic_savez(path, coef=np.arange(10), status=np.asarray("fitted"))
            with np.load(path) as data:
                np.testing.assert_array_equal(data["coef"], np.arange(10))
                self.assertEqual(str(data["status"]), "fitted")
        with self.assertRaises(ValueError):
            common.validate_unique_rows([{"a": 1}, {"a": 1}], ["a"], "table")

    def test_audit_replay_tolerates_only_small_summation_order_drift(self):
        aggregate._assert_close(-103.48088684898289, -103.480886848873, "pooled_r2")
        with self.assertRaises(ValueError):
            aggregate._assert_close(0.25, 0.25000001, "pooled_r2")


if __name__ == "__main__":
    unittest.main()

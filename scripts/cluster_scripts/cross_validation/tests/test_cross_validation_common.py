import csv
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
CV_DIR = TEST_DIR.parent
CLUSTER_DIR = CV_DIR.parent
for directory in (CLUSTER_DIR, CV_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import cross_validation_common as common
import regression_protocol as protocol


def fit_weighted_linear(prepared):
    design = np.column_stack([
        np.ones(prepared.standardized_features.shape[0], dtype=np.float64),
        prepared.standardized_features,
    ])
    sqrt_weight = np.sqrt(prepared.sample_weights)
    fitted, *_ = np.linalg.lstsq(
        design * sqrt_weight[:, None], prepared.targets * sqrt_weight, rcond=None,
    )

    def predict(features):
        standardized = prepared.standardizer.transform(features)
        return fitted[0] + standardized @ fitted[1:]

    return predict


class FoldAndSamplingTests(unittest.TestCase):
    def test_loads_frozen_folds_and_partitions_every_protein(self):
        bundle = common.load_cv_folds()
        self.assertEqual(len(bundle.eligible_protein_ids), 552)
        heldout = [set(bundle.protein_ids_for_outer_fold(fold)) for fold in range(5)]
        self.assertEqual([len(ids) for ids in heldout], [111, 111, 110, 110, 110])
        self.assertEqual(set.union(*heldout), set(bundle.eligible_protein_ids))
        self.assertTrue(all(
            heldout[left].isdisjoint(heldout[right])
            for left in range(5) for right in range(left + 1, 5)
        ))
        for fold in range(5):
            self.assertEqual(
                len(bundle.training_protein_ids(fold)) + len(heldout[fold]), 552,
            )

    def test_fold_major_task_mapping_round_trips(self):
        self.assertEqual(common.map_fold_layer_task(0), common.FoldLayerTask(0, 0, 0))
        self.assertEqual(common.map_fold_layer_task(47), common.FoldLayerTask(47, 0, 47))
        self.assertEqual(common.map_fold_layer_task(48), common.FoldLayerTask(48, 1, 0))
        self.assertEqual(common.map_fold_layer_task(239), common.FoldLayerTask(239, 4, 47))
        for fold in range(5):
            for layer in range(48):
                task = common.fold_layer_task_id(fold, layer)
                self.assertEqual(common.map_fold_layer_task(task).outer_fold, fold)
                self.assertEqual(common.map_fold_layer_task(task).layer, layer)
        with self.assertRaises(ValueError):
            common.map_fold_layer_task(240)

    def test_deterministic_sampling_is_sorted_bounded_and_layer_independent(self):
        first = common.deterministic_sample_indices(100, "protein-A", 17)
        second = common.deterministic_sample_indices(100, "protein-A", 17)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, 17)
        self.assertTrue(np.all(first[:-1] < first[1:]))
        self.assertEqual(np.unique(first).size, first.size)
        np.testing.assert_array_equal(
            common.deterministic_sample_indices(5, "protein-A", 17), np.arange(5),
        )
        self.assertNotEqual(
            common.sampling_assignment_digest(100, "protein-A", 17),
            common.sampling_assignment_digest(100, "protein-B", 17),
        )


class TrainingAndAccumulatorTests(unittest.TestCase):
    def test_train_only_scaler_baseline_and_protein_equal_weights(self):
        features = {
            "short": np.asarray([[1.0, 2.0], [3.0, 4.0]], dtype=np.float32),
            "long": np.asarray([[10.0, 20.0], [12.0, 24.0], [14.0, 28.0]], dtype=np.float32),
        }
        targets = {
            "short": np.asarray([1.0, 3.0], dtype=np.float32),
            "long": np.asarray([10.0, 12.0, 14.0], dtype=np.float32),
        }
        prepared = common.prepare_training_fold(features, targets)
        expected_mean = (
            np.mean(features["short"].astype(np.float64), axis=0)
            + np.mean(features["long"].astype(np.float64), axis=0)
        ) / 2.0
        np.testing.assert_allclose(prepared.standardizer.feature_mean, expected_mean)
        self.assertAlmostEqual(prepared.global_baseline.value, (2.0 + 12.0) / 2.0)
        self.assertEqual(prepared.standardized_features.dtype, np.float64)
        self.assertEqual(prepared.targets.dtype, np.float64)
        for start, stop in prepared.row_slices.values():
            self.assertAlmostEqual(float(np.sum(prepared.sample_weights[start:stop])), 1.0)

        heldout_features = np.full((4, 2), 1e9, dtype=np.float64)
        heldout_targets = np.full(4, -1e9, dtype=np.float64)
        del heldout_features, heldout_targets
        repeated = common.prepare_training_fold(features, targets)
        self.assertEqual(
            prepared.standardizer.scaler_digest, repeated.standardizer.scaler_digest,
        )
        self.assertEqual(
            prepared.global_baseline.baseline_digest,
            repeated.global_baseline.baseline_digest,
        )

    def test_conditional_baselines_are_gate_and_protein_equal(self):
        targets = {
            "a": np.asarray([1.0, 3.0, 50.0]),
            "b": np.asarray([9.0, 20.0, 40.0]),
        }
        codes = {
            "a": np.asarray([0, 0, 1], dtype=np.int16),
            "b": np.asarray([0, 1, 1], dtype=np.int16),
        }
        baselines = common.fit_conditional_train_only_baselines(
            targets, codes, ("near", "far", "empty"),
        )
        self.assertAlmostEqual(baselines["near"].value, (2.0 + 9.0) / 2.0)
        self.assertAlmostEqual(baselines["far"].value, (50.0 + 30.0) / 2.0)
        self.assertTrue(math.isnan(baselines["empty"].value))

    def test_sufficient_statistics_merge_and_serialization_match_direct(self):
        y = np.asarray([1.0, 2.0, 3.0, 5.0])
        pred = np.asarray([1.5, 2.5, 2.0, 4.0])
        left = common.RegressionSufficientStatistics()
        left.update(y[:2], pred[:2])
        right = common.RegressionSufficientStatistics()
        right.update(y[2:], pred[2:])
        left.merge(right)
        direct = common.RegressionSufficientStatistics()
        direct.update(y, pred)
        self.assertEqual(left.to_payload(), direct.to_payload())
        restored = common.RegressionSufficientStatistics.from_payload(left.to_payload())
        self.assertEqual(restored.to_payload(), direct.to_payload())


class SyntheticCVIntegrationTests(unittest.TestCase):
    def test_protein_level_cv_merges_to_direct_oof_metrics(self):
        rng = np.random.default_rng(123)
        features = {}
        targets = {}
        folds = {}
        for index in range(10):
            protein_id = f"p{index:02d}"
            block = rng.normal(loc=index / 3.0, scale=1.0, size=(6, 2))
            features[protein_id] = block
            targets[protein_id] = 2.5 + 1.75 * block[:, 0] - 0.4 * block[:, 1]
            folds[protein_id] = index % 5

        pooled = common.OOFComparisonAccumulator()
        fold_rows = []
        direct_y = []
        direct_pred = []
        for fold in range(5):
            training_ids = [protein_id for protein_id in features if folds[protein_id] != fold]
            heldout_ids = [protein_id for protein_id in features if folds[protein_id] == fold]
            prepared = common.prepare_training_fold(
                {protein_id: features[protein_id] for protein_id in training_ids},
                {protein_id: targets[protein_id] for protein_id in training_ids},
            )
            predict = fit_weighted_linear(prepared)
            fold_accumulator = common.OOFComparisonAccumulator()
            for protein_id in heldout_ids:
                prediction = predict(features[protein_id])
                fold_accumulator.update_protein(
                    protein_id, fold, targets[protein_id], prediction,
                    prepared.global_baseline.value,
                )
                direct_y.append(targets[protein_id])
                direct_pred.append(prediction)
                row = common.build_per_protein_row(
                    analysis_name="synthetic_shared", model_family="shared_ols",
                    feature_dim=2, layer=0, protein_id=protein_id, source="cameo",
                    outer_fold=fold, evaluation_design="protein_outer_5fold",
                    oof_prediction_scope="heldout_proteins_all_pairs", n_residues=4,
                    y_true=targets[protein_id], predictions=prediction,
                )
                self.assertEqual(tuple(row), common.PER_PROTEIN_COLUMNS)
                self.assertEqual(row["prediction_status"], "complete")
            fold_rows.append(common.build_fold_metrics_row(
                fold_accumulator, outer_fold=fold, analysis_name="synthetic_shared",
                model_family="shared_ols", feature_dim=2, layer=0,
                evaluation_design="protein_outer_5fold",
                oof_prediction_scope="heldout_proteins_all_pairs",
            ))
            pooled.merge(fold_accumulator)

        pooled_row = common.build_pooled_metrics_row(
            pooled, analysis_name="synthetic_shared", model_family="shared_ols",
            feature_dim=2, layer=0, evaluation_design="protein_outer_5fold",
            oof_prediction_scope="heldout_proteins_all_pairs",
        )
        direct = common.RegressionSufficientStatistics()
        direct.update(np.concatenate(direct_y), np.concatenate(direct_pred))
        self.assertAlmostEqual(pooled_row["pooled_r2"], direct.metrics()["r2"])
        self.assertAlmostEqual(pooled_row["pooled_mae"], direct.metrics()["mae"])
        self.assertAlmostEqual(pooled_row["pooled_rmse"], direct.metrics()["rmse"])
        self.assertEqual(pooled_row["n_proteins"], 10)
        self.assertTrue(pooled.is_complete)
        summaries = common.build_fold_summary_rows(fold_rows)
        self.assertEqual(len(summaries), 12)
        self.assertTrue(all(row["n_folds"] in {0, 5} for row in summaries))

        payload = pooled.to_payload()
        restored = common.OOFComparisonAccumulator.from_payload(payload)
        self.assertEqual(restored.to_payload(), payload)

    def test_pair_level_cv_has_exact_once_coverage_and_one_geometry(self):
        coordinates = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0],
            [0.0, 0.0, 1.0], [1.0, 1.0, 0.5], [0.5, 1.5, 1.0],
        ])
        pair_i, pair_j = protocol.strict_lower_triangle_indices(6)
        targets = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
        features = np.column_stack([targets, np.abs(pair_i - pair_j).astype(np.float64)])
        assignments = protocol.within_protein_pair_folds(6, "synthetic-pair")
        assembler = common.OOFPairAssembler(targets, with_conditional_baseline=True)
        for fold in range(5):
            train = assignments != fold
            heldout = np.flatnonzero(assignments == fold)
            prepared = common.prepare_training_fold(
                {"synthetic-pair": features[train]},
                {"synthetic-pair": targets[train]},
            )
            predict = fit_weighted_linear(prepared)
            assembler.add_fold(
                fold, heldout, predict(features[heldout]), prepared.global_baseline.value,
                conditional_baseline_predictions=prepared.global_baseline.value,
            )
        assembled = assembler.finalize()
        np.testing.assert_array_equal(assembled["pair_folds"], assignments)
        self.assertTrue(assembler.assignment_complete)
        self.assertTrue(assembler.prediction_complete)

        accumulator = common.OOFComparisonAccumulator()
        accumulator.update_protein(
            "synthetic-pair", 3, assembled["y_true"], assembled["predictions"],
            assembled["global_baseline_predictions"],
            conditional_baseline_predictions=assembled["conditional_baseline_predictions"],
        )
        row = common.build_pooled_metrics_row(
            accumulator, analysis_name="synthetic_per_protein",
            model_family="per_protein_ols", feature_dim=2, layer=0,
            evaluation_design="within_protein_pair_5fold",
            oof_prediction_scope="heldout_pairs_complete_oof",
            conditional_baseline_type="per_protein_pair_fold_training_mean",
        )
        self.assertEqual(row["n_evaluated_pairs"], targets.size)
        geometry = common.build_geometry_row(
            analysis_name="synthetic_per_protein", model_family="per_protein_ols",
            feature_dim=2, layer=0, protein_id="synthetic-pair", source="cameo",
            outer_fold=3, evaluation_design="within_protein_pair_5fold",
            oof_prediction_scope="heldout_pairs_complete_oof", n_residues=6,
            true_coordinates=coordinates, predictions=assembled["predictions"],
        )
        self.assertEqual(tuple(geometry), common.GEOMETRY_METRICS_COLUMNS)
        self.assertEqual(geometry["geometry_status"], "complete")
        self.assertEqual(geometry["n_triangles"], math.comb(6, 3))

        with self.assertRaises(ValueError):
            assembler.add_fold(0, np.asarray([0]), np.asarray([1.0]), 1.0,
                               conditional_baseline_predictions=1.0)


class OutputReplayAndCheckpointTests(unittest.TestCase):
    def test_v2_writer_and_schema_are_additive(self):
        row = common.build_runtime_row(
            analysis_name="probe", model_family="shared_ols", task_id=0, outer_fold=0,
            layer=0, completion_status="complete", n_fitted_proteins=4,
            n_evaluated_proteins=1, n_fitted_pairs=20, n_evaluated_pairs=5,
            elapsed_seconds=1.25,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "runtime.csv"
            common.write_v2_csv(path, common.RUNTIME_FILENAME, [row])
            with path.open(newline="") as handle:
                loaded = list(csv.DictReader(handle))
            self.assertEqual(tuple(loaded[0]), common.RUNTIME_COLUMNS)
            self.assertEqual(common.OUTPUT_SCHEMA_VERSION, "2.0")
            with self.assertRaises(ValueError):
                common.write_v2_csv(path, common.RUNTIME_FILENAME, [{"bad": 1}])

        importance = common.build_feature_importance_rows(
            np.arange(10, dtype=np.float64).reshape(5, 2),
            np.ones((5, 2), dtype=np.float64),
            analysis_name="probe", model_family="shared_xgboost", layer=0,
            feature_names=("f0", "f1"),
        )
        self.assertEqual(len(importance), 2)
        self.assertEqual(tuple(importance[0]), common.FEATURE_IMPORTANCE_COLUMNS)
        self.assertEqual(importance[0]["n_outer_folds"], 5)

    def test_replay_index_checks_artifact_hash_and_unique_responsibility(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "model_artifacts" / "coefficients" / "fold_0_layer_00.npz"
            common.write_v2_npz(artifact, coefficients=np.arange(3, dtype=np.float64))
            row = common.build_replay_index_row(
                family_root=root,
                artifact_path="model_artifacts/coefficients/fold_0_layer_00.npz",
                analysis_name="probe", model_family="shared_ols", layer=0,
                outer_fold=0, artifact_type="ols_coefficients",
                configuration_digest="config", scaler_digest="scaler",
            )
            common.validate_replay_index([row], family_root=root, expected_count=1)
            with self.assertRaisesRegex(ValueError, "duplicated"):
                common.validate_replay_index([row, row], family_root=root)
            artifact.write_bytes(b"corrupt")
            with self.assertRaisesRegex(ValueError, "checksum"):
                common.validate_replay_index([row], family_root=root)

    def test_checkpoint_resume_rejects_invariant_and_artifact_corruption(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            artifact = root / "layer_rows.json"
            artifact.write_text(json.dumps({"complete": True}) + "\n")
            manifest_path = root / "checkpoint.json"
            invariants = {"analysis": "probe", "outer_fold": 0, "layer": 0}
            manifest = common.commit_checkpoint(
                manifest_path, invariants=invariants,
                artifacts={"rows": "layer_rows.json"},
                completed_at="2026-01-01T00:00:00+00:00",
            )
            self.assertEqual(manifest["status"], "complete")
            validated = common.validate_resume_checkpoint(
                manifest_path, expected_invariants=invariants,
                expected_artifact_names={"rows"},
            )
            self.assertEqual(validated, manifest)
            with self.assertRaisesRegex(ValueError, "invariants"):
                common.validate_resume_checkpoint(
                    manifest_path,
                    expected_invariants={**invariants, "layer": 1},
                )
            artifact.write_text("changed\n")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                common.validate_resume_checkpoint(
                    manifest_path, expected_invariants=invariants,
                )


if __name__ == "__main__":
    unittest.main()

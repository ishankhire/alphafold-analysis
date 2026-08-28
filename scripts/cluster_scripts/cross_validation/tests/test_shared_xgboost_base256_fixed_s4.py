import csv
import json
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace
from unittest import mock

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
CV_DIR = TEST_DIR.parent
CLUSTER_DIR = CV_DIR.parent
for directory in (CLUSTER_DIR, CV_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import aggregate_shared_xgboost_base256_fixed_s4 as aggregate
import cross_validation_common as cv
import shared_xgboost_base256_fixed_s4_common as xgb
from regression_output_helpers import sha256_file
from regression_protocol import FEATURE_DIM, strict_lower_triangle_indices


class SyntheticFoldBundle:
    def __init__(self, records):
        entries = {
            record.protein_id: SimpleNamespace(
                protein_id=record.protein_id,
                source=record.source,
                n_residues=record.n_residues,
                n_pairs=record.targets.size,
                outer_fold=record.outer_fold,
            )
            for record in records
        }
        self.outer_by_protein = MappingProxyType(entries)
        self.eligible_protein_ids = tuple(sorted(entries))
        self.fold_digests = MappingProxyType({
            "cohort_digest": "synthetic-cohort",
            "outer_fold_digest": "synthetic-outer-folds",
            "pair_fold_manifest_digest": "synthetic-pair-folds",
        })

    def protein_ids_for_outer_fold(self, outer_fold):
        return tuple(
            protein_id for protein_id in self.eligible_protein_ids
            if self.outer_by_protein[protein_id].outer_fold == int(outer_fold)
        )

    def training_protein_ids(self, heldout_fold):
        heldout = set(self.protein_ids_for_outer_fold(heldout_fold))
        return tuple(
            protein_id for protein_id in self.eligible_protein_ids
            if protein_id not in heldout
        )


def synthetic_records():
    records = []
    pair_i, pair_j = strict_lower_triangle_indices(5)
    for fold in range(5):
        rng = np.random.default_rng(1200 + fold)
        coordinates = rng.normal(size=(5, 3))
        targets = np.linalg.norm(
            coordinates[pair_i] - coordinates[pair_j], axis=1,
        )
        features = rng.normal(scale=0.1, size=(targets.size, FEATURE_DIM))
        features[:, 0] = targets
        records.append(xgb.ProteinLayerRecord(
            protein_id=f"p{fold}",
            source="cameo" if fold % 2 == 0 else "ecod",
            n_residues=5, outer_fold=fold, coordinates=coordinates,
            chain_ids=tuple("A" for _ in range(5)),
            pair_i=pair_i.copy(), pair_j=pair_j.copy(), targets=targets,
            synthetic_features=features,
        ))
    return records


class FakeEstimator:
    def __init__(self, **kwargs):
        self.init_kwargs = kwargs
        self.fit_kwargs = None
        self.rounds = int(kwargs.get("n_estimators", xgb.BOOSTING_ROUNDS))
        self.fold = int(kwargs.get("random_state", xgb.BASE_RANDOM_SEED)) - xgb.BASE_RANDOM_SEED

    def fit(self, features, targets, **kwargs):
        self.fit_kwargs = dict(kwargs)
        self.fit_shape = features.shape
        return self

    def predict(self, features):
        return np.asarray(features[:, 0], dtype=np.float64)

    def get_booster(self):
        return self

    def num_boosted_rounds(self):
        return self.rounds

    def get_score(self, importance_type):
        if importance_type == "gain":
            return {"f0": float(self.fold + 1)}
        if importance_type == "weight":
            return {"f0": float((self.fold + 1) * 10)}
        raise AssertionError(importance_type)

    def save_model(self, path):
        Path(path).write_text(json.dumps({
            "learner": {"rounds": self.rounds, "fold": self.fold},
        }))

    def load_model(self, path):
        payload = json.loads(Path(path).read_text())["learner"]
        self.rounds = int(payload["rounds"])
        self.fold = int(payload["fold"])


def fake_estimator_builder(outer_fold, n_jobs):
    return FakeEstimator(
        **xgb.CONFIGURATION["xgboost_parameters"],
        random_state=xgb.BASE_RANDOM_SEED + int(outer_fold),
        n_jobs=int(n_jobs),
    )


class FrozenConfigurationAndSamplingTests(unittest.TestCase):
    def test_checked_in_configuration_and_selection_checksum_are_fixed(self):
        config = xgb.load_fixed_configuration()
        self.assertEqual(config["configuration_id"], "S4")
        self.assertEqual(config["training_row_cap_per_protein"], 5000)
        self.assertEqual(config["final_boosting_rounds"], 700)
        self.assertFalse(config["nested_selection"])
        self.assertTrue(config["source_runs_exposed_historical_test"])
        source = xgb.REPO_ROOT / config["selection_source_path"]
        self.assertEqual(sha256_file(source), config["selection_source_sha256"])

    def test_configuration_changes_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "config.json"
            config = json.loads(xgb.CONFIG_PATH.read_text())
            config["training_row_cap_per_protein"] = 10000
            config["configuration_digest"] = cv.semantic_digest(
                xgb.configuration_payload(config)
            )
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "training_row_cap"):
                xgb.load_fixed_configuration(path)

    def test_sampling_is_target_and_layer_independent_with_equal_protein_mass(self):
        first = cv.deterministic_sample_indices(8000, "protein-x", 5000)
        second = cv.deterministic_sample_indices(8000, "protein-x", 5000)
        np.testing.assert_array_equal(first, second)
        self.assertEqual(first.size, 5000)
        records = synthetic_records()
        training = xgb.materialize_sampled_training(records)
        self.assertEqual(training.features.dtype, np.float64)
        self.assertEqual(training.targets.dtype, np.float64)
        cursor = 0
        for record in sorted(records, key=lambda item: item.protein_id):
            stop = cursor + record.targets.size
            self.assertAlmostEqual(float(np.sum(training.sample_weights[cursor:stop])), 1.0)
            cursor = stop


class MockedEstimatorTests(unittest.TestCase):
    def test_exact_rounds_seed_and_fit_arguments(self):
        with mock.patch.object(
            xgb, "require_xgboost", return_value=(mock.Mock(__version__="test"), FakeEstimator),
        ):
            model = xgb.build_fixed_estimator(3, 8)
        self.assertEqual(model.init_kwargs["n_estimators"], 700)
        self.assertEqual(model.init_kwargs["random_state"], 45)
        self.assertNotIn("early_stopping_rounds", model.init_kwargs)
        training = xgb.materialize_sampled_training(synthetic_records()[:2])
        xgb.fit_fixed_estimator(model, training)
        self.assertEqual(set(model.fit_kwargs), {"sample_weight"})
        self.assertNotIn("eval_set", model.fit_kwargs)

    def test_wrong_round_count_is_rejected(self):
        model = FakeEstimator(n_estimators=699)
        training = xgb.materialize_sampled_training(synthetic_records()[:2])
        with self.assertRaisesRegex(RuntimeError, "699 rounds"):
            xgb.fit_fixed_estimator(model, training)

    def test_missing_xgboost_has_explicit_cluster_dependency_error(self):
        with mock.patch.dict(sys.modules, {"xgboost": None}):
            with self.assertRaisesRegex(RuntimeError, "activate ~/protein_venv on Delta"):
                xgb.require_xgboost()

    def test_importance_includes_zero_use_features_and_rejects_bad_keys(self):
        model = FakeEstimator(n_estimators=700, random_state=43)
        gain, split = xgb.feature_importance_arrays(model)
        self.assertEqual(gain.shape, (FEATURE_DIM,))
        self.assertEqual(gain[0], 2.0)
        self.assertEqual(split[0], 20.0)
        self.assertTrue(np.all(gain[1:] == 0.0))
        with mock.patch.object(model, "get_score", return_value={"bad": 1.0}):
            with self.assertRaisesRegex(ValueError, "Unexpected booster"):
                xgb.feature_importance_arrays(model)

    def test_launcher_has_locked_stage_six_resources_and_interface(self):
        launcher = (CV_DIR / "shared_xgboost_base256_fixed_s4_fold_layer.sbatch").read_text()
        required = (
            "#SBATCH --array=0-239%3", "#SBATCH --mem=12g",
            "#SBATCH --time=00:30:00", "#SBATCH --constraint=\"scratch&work\"",
            "#SBATCH --mail-type=BEGIN,END,FAIL",
            "#SBATCH --mail-user=ikhire3@gatech.edu",
            "SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}",
            'srun python3 -u "$SCRIPT_DIR/shared_xgboost_base256_fixed_s4_fold_layer.py"',
        )
        self.assertTrue(all(value in launcher for value in required))


class FivefoldIntegrationResumeCorruptionTests(unittest.TestCase):
    def test_partial_task_resumes_the_checksummed_booster_before_heldout_loading(self):
        records = synthetic_records()
        bundle = SyntheticFoldBundle(records)
        task = cv.map_fold_layer_task(0)
        events = []

        class TrackingEstimator(FakeEstimator):
            def fit(self, features, targets, **kwargs):
                events.append("fit")
                return super().fit(features, targets, **kwargs)

            def load_model(self, path):
                events.append("load")
                return super().load_model(path)

        def builder(outer_fold, n_jobs):
            return TrackingEstimator(
                **xgb.CONFIGURATION["xgboost_parameters"],
                random_state=xgb.BASE_RANDOM_SEED + int(outer_fold),
                n_jobs=int(n_jobs),
            )

        kwargs = {
            "task": task,
            "training_records": [record for record in records if record.outer_fold != 0],
            "heldout_protein_ids": bundle.protein_ids_for_outer_fold(0),
            "fold_digests": bundle.fold_digests,
            "input_digest": "synthetic-input-v1",
            "sample_manifest_digest_value": xgb.sampling_manifest_digest(bundle),
            "estimator_builder": builder,
            "versions": {"python": "test", "numpy": np.__version__, "xgboost": "mock-1"},
        }
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / xgb.FAMILY_DIR_NAME
            with self.assertRaisesRegex(RuntimeError, "stop after booster"):
                xgb.execute_fold_layer_task(
                    **kwargs, root=root,
                    heldout_loader=lambda: (_ for _ in ()).throw(
                        RuntimeError("stop after booster")
                    ),
                )
            shard = xgb.task_shard_path(root, task)
            self.assertTrue((shard / xgb.FIT_CHECKPOINT_FILENAME).is_file())
            self.assertFalse((shard / xgb.TASK_CHECKPOINT_FILENAME).exists())
            xgb.execute_fold_layer_task(
                **kwargs, root=root,
                heldout_loader=lambda: [record for record in records if record.outer_fold == 0],
            )
        self.assertEqual(events, ["fit", "load"])

    def test_booster_precedes_heldout_loading_and_family_aggregates(self):
        records = synthetic_records()
        bundle = SyntheticFoldBundle(records)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / xgb.FAMILY_DIR_NAME
            events = []
            for fold in range(5):
                task = cv.map_fold_layer_task(cv.fold_layer_task_id(fold, 0))
                training = [record for record in records if record.outer_fold != fold]
                heldout = [record for record in records if record.outer_fold == fold]

                def heldout_loader(task=task, heldout=heldout):
                    shard = xgb.task_shard_path(root, task)
                    self.assertTrue((shard / xgb.TASK_BOOSTER_PATH).is_file())
                    self.assertTrue((shard / xgb.FIT_CHECKPOINT_FILENAME).is_file())
                    events.append((task.outer_fold, "heldout_loaded"))
                    return heldout

                xgb.execute_fold_layer_task(
                    task=task, training_records=training,
                    heldout_protein_ids=bundle.protein_ids_for_outer_fold(fold),
                    heldout_loader=heldout_loader,
                    fold_digests=bundle.fold_digests,
                    input_digest="synthetic-input-v1",
                    sample_manifest_digest_value=xgb.sampling_manifest_digest(bundle),
                    root=root, estimator_builder=fake_estimator_builder,
                    versions={"python": "test", "numpy": np.__version__, "xgboost": "mock-1"},
                    n_jobs=2,
                )
            self.assertEqual(events, [(fold, "heldout_loaded") for fold in range(5)])

            first = cv.map_fold_layer_task(0)
            xgb.execute_fold_layer_task(
                task=first,
                training_records=[record for record in records if record.outer_fold != 0],
                heldout_protein_ids=bundle.protein_ids_for_outer_fold(0),
                heldout_loader=lambda: self.fail("resume opened held-out records"),
                fold_digests=bundle.fold_digests,
                input_digest="synthetic-input-v1",
                sample_manifest_digest_value=xgb.sampling_manifest_digest(bundle),
                root=root, estimator_builder=fake_estimator_builder,
                versions={"xgboost": "mock-1"},
            )

            with self.assertRaisesRegex(ValueError, "all 48 layers"):
                aggregate.aggregate_shared_xgboost(
                    root=root, fold_bundle=bundle, layers=(0,),
                )
            aggregate.aggregate_shared_xgboost(
                root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
            )
            with (root / cv.PER_PROTEIN_FILENAME).open(newline="") as handle:
                per_rows = list(csv.DictReader(handle))
            with (root / cv.REPLAY_INDEX_FILENAME).open(newline="") as handle:
                replay_rows = list(csv.DictReader(handle))
            with (root / cv.FEATURE_IMPORTANCE_FILENAME).open(newline="") as handle:
                importance_rows = list(csv.DictReader(handle))
            self.assertEqual(len(per_rows), 5)
            self.assertEqual({row["outer_fold"] for row in per_rows}, set("01234"))
            self.assertEqual(len(replay_rows), 5)
            self.assertTrue(all(row["artifact_type"] == "xgboost_native_json_booster" for row in replay_rows))
            self.assertEqual(len(importance_rows), FEATURE_DIM)
            feature_zero = importance_rows[0]
            self.assertEqual(feature_zero["n_outer_folds"], "5")
            self.assertAlmostEqual(float(feature_zero["gain_mean"]), 3.0)
            manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(manifest["completed_task_ids"], [0, 48, 96, 144, 192])
            self.assertEqual(manifest["expected_replay_artifacts"], 5)
            self.assertFalse(manifest["selection_is_nested"])
            self.assertTrue(manifest["historical_test_exposure_disclosed"])
            self.assertFalse(manifest["raw_pair_predictions_saved"])

            runtime_path = xgb.task_shard_path(root, first) / "runtime.csv"
            runtime_path.write_text(runtime_path.read_text() + "corruption\n")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                xgb.execute_fold_layer_task(
                    task=first,
                    training_records=[record for record in records if record.outer_fold != 0],
                    heldout_protein_ids=bundle.protein_ids_for_outer_fold(0),
                    heldout_loader=lambda: self.fail("corrupt resume opened held-out records"),
                    fold_digests=bundle.fold_digests,
                    input_digest="synthetic-input-v1",
                    sample_manifest_digest_value=xgb.sampling_manifest_digest(bundle),
                    root=root, estimator_builder=fake_estimator_builder,
                    versions={"xgboost": "mock-1"},
                )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                aggregate.validate_task_shard(root, first, bundle)


if __name__ == "__main__":
    unittest.main()

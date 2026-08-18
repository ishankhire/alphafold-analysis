import hashlib
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


PACKAGE_DIR = Path(__file__).resolve().parents[1]
if str(PACKAGE_DIR) not in sys.path:
    sys.path.insert(0, str(PACKAGE_DIR))

import shared_xgboost_final_common as common
import shared_xgboost_final_layer as layer_runner
import aggregate_shared_xgboost_final as aggregator
from regression_protocol import FEATURE_DIM, load_master_manifest


class FrozenConfigurationTests(unittest.TestCase):
    def test_exact_checked_in_configuration_is_valid(self):
        config = common.load_final_configuration()
        self.assertEqual(config["configuration_id"], "S4")
        self.assertEqual(config["training_row_cap_per_development_protein"], 5000)
        self.assertEqual(config["xgboost_parameters"]["n_estimators"], 700)
        self.assertFalse(config["test_metrics_used"])
        self.assertEqual(config["split_assignment_digest"], common.split_assignment_rows()[1])

    def test_changed_digest_and_specification_are_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            config = json.loads(common.CONFIG_PATH.read_text())
            for key in ("validation_selection_table", "validation_cap_sweep_table"):
                shutil.copy2(common.CONFIG_PATH.parent / config[key], root / config[key])
            config["training_row_cap_per_development_protein"] = 10000
            path = root / "config.json"
            path.write_text(json.dumps(config))
            with self.assertRaisesRegex(ValueError, "digest"):
                common.load_final_configuration(path)

    def test_selection_tables_have_no_test_fields(self):
        config = common.load_final_configuration()
        for key in ("validation_selection_table", "validation_cap_sweep_table"):
            header = (common.CONFIG_PATH.parent / config[key]).read_text().splitlines()[0].split(",")
            self.assertFalse(any(column.startswith("test_") for column in header))

    def test_search_and_cap_entrypoints_are_development_only(self):
        for filename in (
            "shared_xgboost_hyperparameter_search.py",
            "shared_xgboost_winner_cap_sweep.py",
        ):
            source = (common.SCRIPT_DIR / filename).read_text()
            self.assertIn('master_roles=("fit", "validation")', source)
            self.assertNotIn('split["test"]', source)
            self.assertNotIn("evaluate_model_records", source)


class SamplingAndSplitTests(unittest.TestCase):
    def test_frozen_roles_and_final_development_union(self):
        entries = [entry for entry in load_master_manifest() if entry.eligible]
        counts = {role: sum(entry.master_role == role for entry in entries) for role in ("fit", "validation", "test")}
        self.assertEqual(counts, {"fit": 330, "validation": 111, "test": 111})
        rows, digest, indices = common.build_training_sample_manifest()
        self.assertEqual(len(rows), 441)
        self.assertEqual({row["sample_policy_digest"] for row in rows}, {digest})
        self.assertEqual(set(indices), {row["protein_id"] for row in rows})
        for row in rows:
            selected = indices[row["protein_id"]]
            self.assertEqual(len(selected), min(5000, row["n_protein_pairs"]))
            self.assertEqual(hashlib.sha256(selected.tobytes()).hexdigest(), row["indices_sha256"])
            self.assertAlmostEqual(row["row_weight"] * row["n_selected_pairs"], 1.0)

    def test_capped_indices_are_layer_and_target_independent_and_keep_small_proteins(self):
        first = common.capped_pair_indices(8000, "protein-x", 5000, 42)
        second = common.capped_pair_indices(8000, "protein-x", 5000, 42)
        np.testing.assert_array_equal(first, second)
        small = common.capped_pair_indices(17, "protein-y", 5000, 42)
        self.assertEqual(set(small), set(range(17)))


class EstimatorTests(unittest.TestCase):
    class FakeEstimator:
        def __init__(self, **kwargs):
            self.init_kwargs = kwargs
            self.fit_kwargs = None

        def fit(self, x, y, **kwargs):
            self.fit_kwargs = kwargs

    def test_final_fit_is_exactly_700_rounds_without_eval_or_early_stopping(self):
        config = common.load_final_configuration()
        with mock.patch.object(common, "require_xgboost", return_value=(mock.Mock(), self.FakeEstimator)):
            model = common.build_final_estimator(config, 8)
        self.assertEqual(model.init_kwargs["n_estimators"], 700)
        self.assertNotIn("early_stopping_rounds", model.init_kwargs)
        bundle = {
            "x": np.zeros((4, FEATURE_DIM), dtype=np.float64),
            "y": np.arange(4, dtype=np.float64),
            "weights": np.ones(4, dtype=np.float64),
        }
        common.fit_final_estimator(model, bundle)
        self.assertEqual(set(model.fit_kwargs), {"sample_weight"})
        self.assertNotIn("eval_set", model.fit_kwargs)

    def test_missing_xgboost_has_explicit_dependency_error(self):
        with mock.patch.dict(sys.modules, {"xgboost": None}):
            with self.assertRaisesRegex(RuntimeError, "XGBoost is required"):
                common.require_xgboost()

    def test_importance_maps_zeroes_and_rejects_malformed_keys(self):
        booster = mock.Mock()
        booster.get_score.side_effect = [{"f1": 2.5}, {"f1": 3.0}]
        model = mock.Mock()
        model.get_booster.return_value = booster
        rows = common.feature_importance_rows(model, 7)
        self.assertEqual(len(rows), FEATURE_DIM)
        self.assertEqual(rows[0]["gain"], 0.0)
        self.assertEqual(rows[1]["gain"], 2.5)
        booster.get_score.side_effect = [{"bad": 1.0}, {}]
        with self.assertRaisesRegex(ValueError, "Unexpected booster feature key"):
            common.feature_importance_rows(model, 7)


class PhaseOrderingTests(unittest.TestCase):
    class FakeModel:
        def predict(self, features):
            return np.zeros(features.shape[0], dtype=np.float64)

    def test_test_records_cannot_load_before_final_booster_exists(self):
        events = []
        development = [{
            "protein_id": f"p{index}", "source": "cameo", "master_role": "fit",
            "y": np.ones(1), "protein_dir": Path("."),
        } for index in range(441)]

        def load_records(*args, **kwargs):
            roles = tuple(kwargs["master_roles"])
            if roles == ("test",):
                self.assertIn("fit", events)
                raise RuntimeError("stop-after-ordering-check")
            events.append("development")
            return development, []

        bundle_record = development[0]
        bundle = {
            "x": np.zeros((1, FEATURE_DIM), dtype=np.float64),
            "y": np.ones(1, dtype=np.float64),
            "weights": np.ones(1, dtype=np.float64),
            "offsets": [(bundle_record, np.array([0]), 0, 1)],
        }

        def fit_or_resume(*args, **kwargs):
            events.append("fit")
            return self.FakeModel(), False

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(layer_runner, "final_output_root", return_value=Path(directory)),
                mock.patch.object(layer_runner, "working_output_root", return_value=Path(directory) / "work"),
                mock.patch.object(layer_runner, "load_final_configuration", return_value={"configuration_digest": "cfg", "split_assignment_digest": "split"}),
                mock.patch.object(layer_runner, "split_assignment_rows", return_value=([], "split")),
                mock.patch.object(layer_runner, "build_training_sample_manifest", return_value=([{}] * 441, "sample", {row["protein_id"]: np.array([0]) for row in development})),
                mock.patch.object(layer_runner, "load_combined_records", side_effect=load_records),
                mock.patch.object(layer_runner, "build_shared_split", return_value=({"fit": development, "validation": []}, [])),
                mock.patch.object(layer_runner, "validate_frozen_layer_input_paths", return_value="input"),
                mock.patch.object(layer_runner, "protein_equal_mean", return_value=1.0),
                mock.patch.object(layer_runner, "materialize_development_matrix", return_value=bundle),
                mock.patch.object(layer_runner, "version_payload", return_value={"xgboost": "test"}),
                mock.patch.object(layer_runner, "_checkpoint_invariants", return_value={}),
                mock.patch.object(layer_runner, "_fit_or_resume", side_effect=fit_or_resume),
            ):
                with self.assertRaisesRegex(RuntimeError, "ordering-check"):
                    layer_runner.run(0)
        self.assertEqual(events, ["development", "fit"])

    def test_fit_checkpoint_reuse_requires_all_invariants(self):
        events = []

        class CheckpointModel:
            def get_booster(self):
                return self

            def save_model(self, path):
                Path(path).write_text("booster")

            def load_model(self, path):
                events.append(("load", Path(path).name))

        with tempfile.TemporaryDirectory() as directory:
            with (
                mock.patch.object(layer_runner, "build_final_estimator", side_effect=lambda *args: CheckpointModel()),
                mock.patch.object(layer_runner, "fit_final_estimator", side_effect=lambda *args: events.append(("fit", ""))),
            ):
                root = Path(directory)
                _, reused = layer_runner._fit_or_resume(root, 0, {}, {}, {"identity": "a"}, 1)
                self.assertFalse(reused)
                _, reused = layer_runner._fit_or_resume(root, 0, {}, {}, {"identity": "a"}, 1)
                self.assertTrue(reused)
                _, reused = layer_runner._fit_or_resume(root, 0, {}, {}, {"identity": "b"}, 1)
                self.assertFalse(reused)
        self.assertEqual([event[0] for event in events], ["fit", "load", "fit"])


class AggregatorValidationTests(unittest.TestCase):
    def test_missing_or_schema_invalid_shard_is_rejected(self):
        config = common.load_final_configuration()
        with tempfile.TemporaryDirectory() as directory:
            with self.assertRaisesRegex(ValueError, "Missing shard manifest"):
                aggregator.validate_layer_shard(
                    Path(directory), 0, config, "split", "sample",
                )
            bad = Path(directory) / "bad.csv"
            bad.write_text("wrong\nvalue\n")
            with self.assertRaisesRegex(ValueError, "Schema mismatch"):
                common.read_csv_strict(bad, ("expected",))


class CanonicalEvaluationTests(unittest.TestCase):
    class MeanModel:
        def predict(self, features):
            return np.asarray(features[:, 0] + 3.0, dtype=np.float64)

    def test_synthetic_all_pair_outputs_match_canonical_schemas(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            records = []
            rng = np.random.default_rng(9)
            for index, role in enumerate(("fit", "test")):
                protein_id = f"p{index}"
                protein_dir = root / protein_id
                protein_dir.mkdir()
                pair_block = rng.normal(size=(4, 4, 128)).astype(np.float32)
                np.save(protein_dir / f"{protein_id}_pair_block_0.npy", pair_block)
                pair_i, pair_j = np.tril_indices(4, k=-1)
                coordinates = rng.normal(size=(4, 3)).astype(np.float64)
                y = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
                records.append({
                    "protein_id": protein_id, "source": "cameo", "protein_dir": protein_dir,
                    "master_role": role, "n_residues": 4, "pair_i": pair_i,
                    "pair_j": pair_j, "coordinates": coordinates, "y": y,
                    "residues": [("A", "ALA", residue + 1) for residue in range(4)],
                })
            tables = common.evaluate_records(
                self.MeanModel(), records, 0, float(np.mean(records[0]["y"])),
            )
            self.assertEqual(len(tables["per_protein.csv"]), 2)
            self.assertEqual(len(tables["pooled_metrics.csv"]), 2)
            self.assertEqual(len(tables["geometry_metrics.csv"]), 2)
            self.assertEqual(len(tables["geometry_summary.csv"]), 28)
            self.assertEqual(len(tables["subgroup_metrics/by_sequence_separation.csv"]), 6)
            self.assertEqual(len(tables["subgroup_metrics/by_amino_acid_pair.csv"]), 210)
            self.assertEqual(len(tables["subgroup_metrics/by_spatial_distance.csv"]), 5)
            pooled = tables["pooled_metrics.csv"]
            self.assertTrue(all(row["conditional_baseline_type"] == "not_applicable" for row in pooled))
            self.assertTrue(all(np.isnan(row["conditional_baseline_pooled_r2"]) for row in pooled))

    def test_expected_production_row_counts(self):
        per_layer = common.expected_layer_counts()
        self.assertEqual(per_layer["per_protein.csv"] * 48, 26496)
        self.assertEqual(per_layer["pooled_metrics.csv"] * 48, 96)
        self.assertEqual(per_layer["geometry_summary.csv"] * 48, 1344)
        self.assertEqual(per_layer["model_artifacts/feature_importance.csv"] * 48, 12288)


if __name__ == "__main__":
    unittest.main()

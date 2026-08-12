import importlib.util
import csv
import json
import sys
import tempfile
import unittest
from unittest import mock
from pathlib import Path

import numpy as np
import joblib
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.tree import DecisionTreeClassifier


BUNDLE_DIR = Path(__file__).parents[1]
sys.path.insert(0, str(BUNDLE_DIR))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BUNDLE_DIR / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module("local_pair_biological_context_common", "local_pair_biological_context_common.py")
runner = load_module("local_pair_biological_context_probe", "local_pair_biological_context_probe.py")
aggregate = load_module("aggregate_local_pair_biological_context_probe", "aggregate_local_pair_biological_context_probe.py")


def toy_pair_block(n_residues):
    pb = np.empty((n_residues, n_residues, common.N_CHANNELS), dtype=np.float64)
    channels = np.arange(common.N_CHANNELS, dtype=np.float64)
    for i in range(n_residues):
        for j in range(n_residues):
            pb[i, j] = 100 * i + 10 * j + channels
    return pb


class CommonDefinitionTests(unittest.TestCase):
    def test_task_mapping_has_five_transitions_and_seven_bundles(self):
        specs = [common.task_spec(index) for index in range(35)]
        self.assertEqual(specs[0], (10, 11, "controls"))
        self.assertEqual(specs[6], (10, 11, "biology_local_triangle_plus_center"))
        self.assertEqual(specs[7], (15, 16, "controls"))
        self.assertEqual(specs[-1], (40, 41, "biology_local_triangle_plus_center"))
        self.assertEqual(len(set(specs)), 35)
        with self.assertRaises(ValueError):
            common.task_spec(35)

    def test_feature_schemas_have_declared_widths_and_center_is_isolated(self):
        expected = {
            "controls": 5, "endpoint_biology": 47, "local_biology": 73,
            "local_triangle": 12, "biology_local_triangle": 79,
            "biology_local_triangle_mediator": 119,
            "biology_local_triangle_plus_center": 207,
        }
        for bundle, width in expected.items():
            schema = common.feature_schema(bundle)
            self.assertEqual(len(schema), width)
            self.assertEqual([row["feature_index"] for row in schema], list(range(width)))
            center = [row for row in schema if row["feature_family"] == "center_representation"]
            self.assertEqual(len(center), 128 if bundle.endswith("plus_center") else 0)

    def test_local_context_is_unique_bounded_and_excludes_endpoints(self):
        np.testing.assert_array_equal(
            common.local_context_indices(0, 4, 12),
            np.array([1, 2, 3, 5, 6, 7, 8, 9]),
        )
        values = common.local_context_indices(3, 7, 12)
        self.assertEqual(values.tolist(), sorted(set(values.tolist())))
        self.assertNotIn(3, values)
        self.assertNotIn(7, values)
        self.assertTrue(np.all((values >= 0) & (values < 12)))

    def test_amino_acid_aliases_and_property_tables_are_locked(self):
        self.assertEqual(common.normalize_aa("MSE"), "MET")
        self.assertEqual(common.normalize_aa("HID"), "HIS")
        self.assertIsNone(common.normalize_aa("UNK"))
        self.assertEqual(common.HYDROPATHY["ILE"], 4.5)
        self.assertEqual(common.HYDROPATHY["ARG"], -4.5)
        self.assertEqual(common.GRANTHAM_VOLUME["TRP"], 170.0)
        self.assertEqual(common.GRANTHAM_POLARITY["ASP"], 13.0)
        self.assertEqual(common.NOMINAL_CHARGE["HIS"], 0.0)
        self.assertIn("TYR", common.POLAR)

    def test_local_pair_boundaries_are_inclusive(self):
        i, j, separation = common.local_pair_indices(16)
        self.assertEqual(separation.min(), 4)
        self.assertEqual(separation.max(), 12)
        self.assertEqual(i.size, common.local_pair_count(16))
        self.assertTrue(np.all(j - i == separation))

    def test_binary_threshold_excludes_both_boundaries(self):
        labels = common.class_labels(np.array([-0.5001, -0.5, 0.0, 0.5, 0.5001]))
        np.testing.assert_array_equal(labels, [1, -1, -1, -1, 0])

    def test_deterministic_validation_is_source_size_stratified(self):
        records = []
        for source in ("cameo", "ecod"):
            for size_bin in (0, 1):
                for index in range(10):
                    protein_id = f"p_{source}_{size_bin}_{index}"
                    records.append({"protein_id": protein_id, "source": source, "size_bin": size_bin, "split": "train"})
        records.append({"protein_id": "held_out", "source": "cameo", "size_bin": 0, "split": "test"})
        first, rows_first = common.logistic_validation_assignments(records)
        second, rows_second = common.logistic_validation_assignments(records)
        self.assertEqual(first, second)
        self.assertEqual(rows_first, rows_second)
        self.assertNotIn("held_out", first)
        for source in ("cameo", "ecod"):
            for size_bin in (0, 1):
                group = [row for row in rows_first if row["source"] == source and row["size_bin"] == size_bin]
                self.assertEqual(sum(row["role"] == "validation" for row in group), 2)

    def test_bootstrap_draws_are_deterministic_and_sum_to_test_size(self):
        offsets = [({"protein_id": f"p{index}"}, index, index + 1) for index in range(4)]
        counts_a, rows_a = common.bootstrap_assignments(offsets, 10, 11)
        counts_b, rows_b = common.bootstrap_assignments(offsets, 10, 11)
        np.testing.assert_array_equal(counts_a, counts_b)
        self.assertEqual(rows_a, rows_b)
        self.assertTrue(np.all(counts_a.sum(axis=1) == 4))
        self.assertEqual(counts_a.shape, (200, 4))


class FeatureConstructionTests(unittest.TestCase):
    def test_feature_construction_counts_endpoint_and_local_identity(self):
        n = 10
        pb = toy_pair_block(n)
        raw = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(np.float64)
        i, j, separation = np.array([1]), np.array([5]), np.array([4])
        residues = ["ALA", "GLY", "PRO", "CYS", "VAL", "GLY", "LYS", "ASP", "PHE", "SER"]
        families = common.build_all_feature_families(pb, raw, i, j, separation, residues)
        self.assertEqual(families["endpoint_aa"][0, common.AA_TO_INDEX["GLY"]], 2.0)
        self.assertEqual(families["pair_chemistry"][0, 0], 1.0)
        self.assertEqual(families["local_biology"][0, 24], 8.0)
        self.assertEqual(families["local_biology"][0, 25], 3.0)
        for bundle, width in common.BUNDLE_WIDTHS.items():
            self.assertEqual(common.assemble_bundle(families, bundle).shape, (1, width))

    def test_triangle_pressure_matches_upper_and_reverse_inequalities(self):
        n = 10
        pb = np.zeros((n, n, common.N_CHANNELS), dtype=np.float64)
        raw = np.full((n, n), 10.0)
        np.fill_diagonal(raw, 0.0)
        # Pair (1,5), mediator 2: 10 > 2+3 gives shrink pressure 5.
        raw[1, 2] = raw[2, 1] = 2.0
        raw[2, 5] = raw[5, 2] = 3.0
        # Pair (2,6), mediator 3: |9-2| > 1 gives grow pressure 6.
        raw[2, 6] = raw[6, 2] = 1.0
        raw[2, 3] = raw[3, 2] = 9.0
        raw[3, 6] = raw[6, 3] = 2.0
        raw[1, 6] = raw[6, 1] = 3.0
        i, j, sep = np.array([1, 2]), np.array([5, 6]), np.array([4, 4])
        residues = ["ALA"] * n
        families = common.build_all_feature_families(pb, raw, i, j, sep, residues)
        self.assertEqual(families["triangle_pressure"][0, 0], 5.0)
        self.assertGreater(families["triangle_pressure"][0, 1], 0.0)
        self.assertEqual(families["triangle_pressure"][1, 3], 6.0)
        self.assertTrue(np.all(families["triangle_pressure"] >= 0.0))

    def test_directional_distance_decoder_uses_both_saved_halves(self):
        rng = np.random.default_rng(8)
        pb = rng.normal(size=(7, 7, common.N_CHANNELS))
        coefficients = rng.normal(size=common.DISTANCE_FEATURE_DIM)
        i, j = np.array([1, 2]), np.array([5, 6])
        decoded = common.decode_selected_distances(pb, i, j, 3.0, coefficients)
        expected = (
            3.0 + pb[j, i] @ coefficients[:common.N_CHANNELS]
            + pb[i, j] @ coefficients[common.N_CHANNELS:]
        )
        np.testing.assert_allclose(decoded, expected)
        matrix = common.decode_distance_matrix(pb, 3.0, coefficients)
        np.testing.assert_allclose(matrix[i, j], expected)
        np.testing.assert_allclose(matrix, matrix.T)

    def test_center_representation_is_symmetric_average(self):
        n = 9
        pb = toy_pair_block(n)
        raw = np.ones((n, n), dtype=np.float64)
        i, j, sep = np.array([1]), np.array([5]), np.array([4])
        families = common.build_all_feature_families(pb, raw, i, j, sep, ["ALA"] * n)
        expected = 0.5 * (pb[1, 5] + pb[5, 1])
        np.testing.assert_allclose(families["center_representation"][0], expected)

    def test_correlation_labels_and_constant_handling(self):
        self.assertEqual(common.correlation_label(0.29), "weak")
        self.assertEqual(common.correlation_label(-0.3), "moderate")
        self.assertEqual(common.correlation_label(0.7), "high")
        self.assertEqual(common.correlation_label(float("nan")), "undefined")
        self.assertTrue(np.isnan(common.safe_pearson(np.ones(3), np.arange(3))))


class ModelAndAggregationTests(unittest.TestCase):
    def test_miniature_end_to_end_controls_shard(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_root = root / "csv"
            source_root = csv_root / common.SOURCE_VARIANT
            source_root.mkdir(parents=True)
            pair_root = root / "pairs"
            structure_root = root / "pdbs"
            split_rows = []
            protein_ids = [f"p{index}_A" for index in range(12)]
            for index, protein_id in enumerate(protein_ids):
                split_rows.append({
                    "split_id": 0, "split": "train" if index < 10 else "test",
                    "protein_id": protein_id, "n_residues": 16, "size_bin": 0,
                    "n_pairs": 120, "target_variance": 1.0,
                })
                protein_dir = pair_root / protein_id
                protein_dir.mkdir(parents=True)
                base = np.zeros((16, 16, common.N_CHANNELS), dtype=np.float32)
                updated = np.zeros_like(base)
                for i in range(16):
                    for j in range(16):
                        distance = 5.0 + abs(i - j)
                        delta = -1.0 if (i + j) % 2 == 0 else 1.0
                        base[i, j, 0] = distance
                        updated[i, j, 0] = distance + delta
                np.save(protein_dir / f"{protein_id}_pair_block_10.npy", base)
                np.save(protein_dir / f"{protein_id}_pair_block_11.npy", updated)
                pdb_dir = (
                    structure_root
                    / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
                    / "predictions"
                )
                pdb_dir.mkdir(parents=True)
                lines = []
                for residue_index in range(1, 17):
                    lines.append(
                        f"ATOM  {residue_index:5d}  CA  ALA A{residue_index:4d}    "
                        f"{float(residue_index):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 90.00           C  \n"
                    )
                (pdb_dir / f"{protein_id}_model_1_ptm_unrelaxed.pdb").write_text("".join(lines))
            common.write_csv_atomic(source_root / "split_assignments.csv", split_rows)
            coefficient_rows = []
            for layer in (10, 11):
                coefficient_rows.append({
                    "variant": common.SOURCE_VARIANT, "feature_dim": 256,
                    "split_id": 0, "layer": layer, "feature_index": -1,
                    "feature_name": "intercept", "coefficient_standardized": 0.0,
                    "coefficient_raw_scale": 0.0,
                })
                for feature_index in range(256):
                    feature_name = (
                        f"pair_ij_ch_{feature_index}" if feature_index < 128
                        else f"pair_ji_ch_{feature_index - 128}"
                    )
                    coefficient_rows.append({
                        "variant": common.SOURCE_VARIANT, "feature_dim": 256,
                        "split_id": 0, "layer": layer, "feature_index": feature_index,
                        "feature_name": feature_name,
                        "coefficient_standardized": 1.0 if feature_index == 0 else 0.0,
                        "coefficient_raw_scale": 1.0 if feature_index == 0 else 0.0,
                    })
            common.write_csv_atomic(source_root / "coefficients.csv", coefficient_rows)
            output_root = root / "output"
            environment = {
                "SLURM_ARRAY_TASK_ID": "0", "CSV_ROOT": str(csv_root),
                "CAMEO_PAIR_BLOCKS_ROOT": str(pair_root),
                "CAMEO_PDB_ROOT": str(structure_root),
                "LOCAL_BIO_CONTEXT_OUTPUT_ROOT": str(output_root),
            }
            with mock.patch.dict("os.environ", environment, clear=False):
                runner.main()
            shard = output_root / "shards/transition_10_11/controls"
            metadata = json.loads((shard / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["hist_gradient_boosting_max_iter"], 100)
            self.assertIs(metadata["hist_gradient_boosting_early_stopping"], False)
            classifier = joblib.load(shard / "models/classification_hist_gradient_boosting.joblib")["model"]
            regressor = joblib.load(shard / "models/regression_hist_gradient_boosting.joblib")["model"]
            self.assertEqual(classifier.n_iter_, 100)
            self.assertEqual(regressor.n_iter_, 100)
            self.assertTrue((shard / "tree_structures/classification_decision_tree_depth4.txt").exists())
            with (shard / "logistic_regularization_search.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 4)

    def test_hist_gradient_boosting_runs_exactly_100_without_early_stopping(self):
        rng = np.random.default_rng(4)
        x = rng.normal(size=(400, 4))
        y_class = (x[:, 0] > 0).astype(int)
        y_reg = x[:, 0] - x[:, 1]
        classifier = HistGradientBoostingClassifier(
            max_iter=common.HIST_GB_MAX_ITER, max_leaf_nodes=5,
            min_samples_leaf=10, early_stopping=False, random_state=1,
        ).fit(x, y_class)
        regressor = HistGradientBoostingRegressor(
            max_iter=common.HIST_GB_MAX_ITER, max_leaf_nodes=5,
            min_samples_leaf=10, early_stopping=False, random_state=1,
        ).fit(x, y_reg)
        self.assertIs(classifier.early_stopping, False)
        self.assertIs(regressor.early_stopping, False)
        self.assertEqual(classifier.n_iter_, 100)
        self.assertEqual(regressor.n_iter_, 100)

    def test_tree_artifact_tables_match_model_leaves(self):
        rng = np.random.default_rng(5)
        x = rng.normal(size=(120, 5))
        y = (x[:, 0] > 0).astype(int)
        model = DecisionTreeClassifier(max_depth=2, random_state=1).fit(x, y)
        offsets = [({"protein_id": "p1", "source": "cameo", "n_residues": 20}, 0, 60),
                   ({"protein_id": "p2", "source": "ecod", "n_residues": 20}, 60, 120)]
        with tempfile.TemporaryDirectory() as temporary:
            nodes, rules = runner.tree_artifacts(
                10, 11, "controls", "classification", "decision_tree_depth4",
                model, {"train": x, "test": x}, {"train": y, "test": y},
                {"train": model.predict(x), "test": model.predict(x)},
                {"train": model.predict_proba(x)[:, 1], "test": model.predict_proba(x)[:, 1]},
                {"train": offsets, "test": offsets}, {}, Path(temporary),
            )
            self.assertEqual(len(nodes), model.tree_.node_count)
            test_rules = [row for row in rules if row["split"] == "test"]
            self.assertEqual(sum(row["n_pairs"] for row in test_rules), x.shape[0])
            self.assertTrue((Path(temporary) / "tree_structures/classification_decision_tree_depth4.txt").exists())
            self.assertTrue((Path(temporary) / "tree_structures/classification_decision_tree_depth4.dot").exists())

    def test_paired_ablation_subtracts_matching_replicates(self):
        model_rows = []
        for bundle, score in (("local_triangle", 0.7), ("controls", 0.5)):
            model_rows.append({
                "layer_from": "10", "layer_to": "11", "bundle": bundle,
                "target": "classification", "estimator": "logistic", "split": "test",
                "mean_per_protein_balanced_accuracy": str(score),
            })
        bootstrap_rows = []
        for replicate, first, second in ((0, 0.8, 0.6), (1, 0.7, 0.4)):
            for bundle, value in (("local_triangle", first), ("controls", second)):
                bootstrap_rows.append({
                    "layer_from": "10", "layer_to": "11", "bundle": bundle,
                    "target": "classification", "estimator": "logistic",
                    "bootstrap_replicate": str(replicate), "metric": "balanced_accuracy",
                    "aggregation_scope": "mean_per_protein", "value": str(value),
                })
        rows = aggregate.feature_ablation_rows(model_rows, bootstrap_rows)
        row = next(item for item in rows if item["layer_from"] == 10 and item["contrast"] == "local_triangle_minus_controls" and item["estimator"] == "logistic")
        self.assertAlmostEqual(row["point_gain"], 0.2)
        self.assertAlmostEqual(row["bootstrap_mean_gain"], 0.25)
        self.assertEqual(row["n_paired_replicates"], 2)

    def test_aggregator_expects_35_specs(self):
        self.assertEqual(len(aggregate.expected_specs()), 35)
        self.assertEqual(len(set(aggregate.expected_specs())), 35)


if __name__ == "__main__":
    unittest.main()

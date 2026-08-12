import csv
import importlib.util
import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from sklearn.linear_model import Ridge


BUNDLE = Path(__file__).parents[1]
sys.path.insert(0, str(BUNDLE))


def load_module(name, filename):
    spec = importlib.util.spec_from_file_location(name, BUNDLE / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


common = load_module(
    "local_pair_representation_update_common", "local_pair_representation_update_common.py"
)
probe = load_module(
    "local_pair_representation_update_probe", "local_pair_representation_update_probe.py"
)
aggregate = load_module(
    "aggregate_local_pair_representation_update_probe",
    "aggregate_local_pair_representation_update_probe.py",
)


def toy_pair_block(n_residues=14):
    channels = np.arange(common.N_CHANNELS, dtype=np.float64)
    first = np.arange(n_residues, dtype=np.float64)[:, None, None]
    second = np.arange(n_residues, dtype=np.float64)[None, :, None]
    return first * 1000.0 + second * 10.0 + channels[None, None, :]


def write_toy_pdb(path, n_residues):
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = []
    for index in range(n_residues):
        aa = common.STANDARD_AAS[index % len(common.STANDARD_AAS)]
        lines.append(
            f"ATOM  {index + 1:5d}  CA  {aa:>3s} A{index + 1:4d}    "
            f"{float(index):8.3f}{0.0:8.3f}{0.0:8.3f}  1.00 20.00           C\n"
        )
    path.write_text("".join(lines) + "END\n")


def write_distance_coefficients(path, layers):
    fields = (
        "variant", "feature_dim", "split_id", "layer", "feature_index",
        "feature_name", "coefficient_raw_scale",
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for layer in layers:
            writer.writerow({
                "variant": common.SOURCE_VARIANT, "feature_dim": 256,
                "split_id": 0, "layer": layer, "feature_index": -1,
                "feature_name": "intercept", "coefficient_raw_scale": float(layer),
            })
            for index in range(256):
                writer.writerow({
                    "variant": common.SOURCE_VARIANT, "feature_dim": 256,
                    "split_id": 0, "layer": layer, "feature_index": index,
                    "feature_name": (
                        f"pair_ij_ch_{index}" if index < 128
                        else f"pair_ji_ch_{index - 128}"
                    ),
                    "coefficient_raw_scale": (layer + 1) * (index + 1) * 1e-6,
                })


class LocalPairRepresentationUpdateTests(unittest.TestCase):
    def test_task_grids_are_complete_and_transition_major(self):
        small = [common.task_spec(index, "small") for index in range(30)]
        large = [common.task_spec(index, "large") for index in range(12)]
        self.assertEqual(small[0], (0, 1, "center"))
        self.assertEqual(small[4], (0, 1, "triangle_r5"))
        self.assertEqual(small[5], (10, 11, "center"))
        self.assertEqual(small[-1], (46, 47, "triangle_r5"))
        self.assertEqual(large[0], (0, 1, "triangle_r7"))
        self.assertEqual(large[1], (0, 1, "triangle_r9"))
        self.assertEqual(large[-1], (46, 47, "triangle_r9"))
        self.assertEqual(len(set(small + large)), 42)

    def test_feature_dimensions_and_schema(self):
        expected = {
            "center": 128, "triangle_r1": 1156, "triangle_r2": 2184,
            "triangle_r3": 3212, "triangle_r5": 5268,
            "triangle_r7": 7324, "triangle_r9": 9380,
        }
        for variant, width in expected.items():
            schema = common.feature_schema(variant)
            self.assertEqual(common.feature_dim(variant), width)
            self.assertEqual(len(schema), width)
            self.assertEqual([row["feature_index"] for row in schema], list(range(width)))
            self.assertTrue(all(row["feature_dim"] == width for row in schema))

    def test_local_pairs_are_sequence_separation_4_through_12(self):
        pair_i, pair_j, separation = common.local_pair_indices(16)
        self.assertEqual(separation.min(), 4)
        self.assertEqual(separation.max(), 12)
        self.assertEqual(pair_i.size, sum(16 - value for value in range(4, 13)))
        self.assertTrue(np.all(pair_i < pair_j))

    def test_endpoint_collision_and_boundaries_are_zero_with_invalid_mask(self):
        pb = toy_pair_block(14)
        pair_i = np.array([0], dtype=np.int64)
        pair_j = np.array([5], dtype=np.int64)
        x = common.build_feature_chunk(pb, pair_i, pair_j, "triangle_r5")
        slot_width = 2 * common.N_CHANNELS + 1
        shell5 = common.N_CHANNELS + 4 * 4 * slot_width
        for slot_index in (0, 1, 2):
            start = shell5 + slot_index * slot_width
            self.assertTrue(np.all(x[0, start:start + 2 * common.N_CHANNELS] == 0.0))
            self.assertEqual(x[0, start + 2 * common.N_CHANNELS], 0.0)
        valid_start = shell5 + 3 * slot_width
        self.assertEqual(x[0, valid_start + 2 * common.N_CHANNELS], 1.0)

    def test_duplicate_valid_intermediate_slots_are_retained(self):
        pb = toy_pair_block(14)
        pair_i = np.array([2], dtype=np.int64)
        pair_j = np.array([6], dtype=np.int64)
        x = common.build_feature_chunk(pb, pair_i, pair_j, "triangle_r2")
        slot_width = 2 * common.N_CHANNELS + 1
        shell2 = common.N_CHANNELS + 4 * slot_width
        i_plus = shell2 + slot_width
        j_minus = shell2 + 2 * slot_width
        np.testing.assert_allclose(
            x[:, i_plus:i_plus + 2 * common.N_CHANNELS],
            x[:, j_minus:j_minus + 2 * common.N_CHANNELS],
        )
        self.assertEqual(x[0, i_plus + 2 * common.N_CHANNELS], 1.0)
        self.assertEqual(x[0, j_minus + 2 * common.N_CHANNELS], 1.0)

    def test_unordered_amino_acid_categories_cover_210_pairs(self):
        self.assertEqual(len(common.AA_PAIRS), 210)
        names = ["ALA", "GLY", "LEU", "MSE", None, "GLU", "GLN"]
        pair_i = np.array([0, 2, 1, 3, 0, 5])
        pair_j = np.array([1, 1, 2, 2, 4, 6])
        first = common.aa_pair_indices(names, pair_i, pair_j)
        reverse = common.aa_pair_indices(names, pair_j, pair_i)
        np.testing.assert_array_equal(first, reverse)
        self.assertEqual(first[4], -1)
        self.assertEqual(common.normalize_aa("MSE"), "MET")
        all_first, all_second = np.triu_indices(len(common.STANDARD_AAS), k=0)
        all_categories = common.aa_pair_indices(
            common.STANDARD_AAS, all_first, all_second
        )
        np.testing.assert_array_equal(all_categories, np.arange(210))
        np.testing.assert_array_equal(
            all_categories,
            common.aa_pair_indices(common.STANDARD_AAS, all_second, all_first),
        )

    def test_weighted_linear_operator_ridge_matches_sklearn(self):
        rng = np.random.default_rng(11)
        raw = rng.normal(size=(120, 8))
        target = raw @ rng.normal(size=8) + rng.normal(scale=0.1, size=120)
        offsets = [({"protein_id": "a"}, 0, 30), ({"protein_id": "b"}, 30, 120)]
        mean, scale = common.protein_equal_scaler(raw, offsets)
        x = raw.copy()
        common.standardize_in_place(x, mean, scale)
        weights = common.protein_equal_weights(offsets)
        fit = probe.fit_weighted_ridge_lsqr(
            x, target, weights, tolerance=1e-11, max_iter=1000
        )
        expected = Ridge(
            alpha=common.RELATIVE_RIDGE_ALPHA * weights.sum(), solver="cholesky"
        ).fit(x, target, sample_weight=weights)
        np.testing.assert_allclose(fit["coefficient"], expected.coef_, rtol=1e-7, atol=1e-8)
        self.assertAlmostEqual(fit["intercept"], expected.intercept_, places=8)
        np.testing.assert_allclose(
            fit["intercept"] + x @ fit["coefficient"], expected.predict(x),
            rtol=1e-7, atol=1e-8,
        )

    def test_magnitude_boundaries_and_category_r2(self):
        target = np.array([-4.0, -2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 1.0, 2.0, 4.0, 5.0])
        prediction = target.copy()
        offsets = [({"protein_id": "p", "source": "cameo", "n_residues": 20}, 0, target.size)]
        rows = probe.magnitude_metrics(0, 1, "center", "test", target, prediction, offsets)
        lookup = {row["bin_name"]: row for row in rows}
        self.assertEqual(int(lookup["abs_gt_0_5"]["n_pairs"]), 7)
        self.assertEqual(int(lookup["abs_gt_4_0"]["n_pairs"]), 1)
        self.assertEqual(int(lookup["exact_zero"]["n_pairs"]), 1)
        categories = np.array([0] * 6 + [1] * 6, dtype=np.int16)
        category_rows = probe.aa_regression_metrics(
            0, 1, "center", "test", target, prediction, categories, offsets
        )
        self.assertEqual(len(category_rows), 210)
        self.assertAlmostEqual(category_rows[0]["r2"], 1.0)
        self.assertAlmostEqual(category_rows[1]["r2"], 1.0)

    def test_bootstrap_outputs_are_deterministic_and_include_all_categories(self):
        target = np.array([-2.0, -1.0, 1.0, 2.0, -3.0, -1.0, 1.0, 3.0])
        prediction = 0.9 * target
        categories = np.array([0, 0, 1, 1, 0, 0, 1, 1], dtype=np.int16)
        offsets = [
            ({"protein_id": "a"}, 0, 4), ({"protein_id": "b"}, 4, 8),
        ]
        counts, rows = common.bootstrap_assignments(offsets, 0, 1)
        counts_again, rows_again = common.bootstrap_assignments(offsets, 0, 1)
        np.testing.assert_array_equal(counts, counts_again)
        self.assertEqual(rows, rows_again)
        headline, aa_rows, target_rows = probe.bootstrap_outputs(
            0, 1, "center", target, prediction, categories, offsets, counts
        )
        self.assertEqual(len(headline), 2 * common.BOOTSTRAP_REPLICATES)
        self.assertEqual(len(aa_rows), 210)
        self.assertEqual(len(target_rows), 210)
        self.assertEqual(int(aa_rows[0]["n_valid_bootstrap_replicates"]), 200)

    def test_aggregator_expects_all_42_shards(self):
        self.assertEqual(len(aggregate.expected_specs()), 42)
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            with self.assertRaisesRegex(RuntimeError, "missing metadata"):
                aggregate.validate_shards(root, required_files=())
            for layer_from, layer_to, variant in aggregate.expected_specs():
                directory = aggregate.shard_path(root, layer_from, layer_to, variant)
                directory.mkdir(parents=True)
                (directory / "run_metadata.json").write_text(json.dumps({
                    "status": "complete", "layer_from": layer_from,
                    "layer_to": layer_to, "variant": variant,
                    "feature_dim": common.feature_dim(variant),
                    "distance_probe_layers": [layer_from, layer_to],
                    "lsqr_max_iter": common.LSQR_MAX_ITER,
                }))
            with mock.patch.object(aggregate, "validate_shared_rows"):
                validated = aggregate.validate_shards(root, required_files=())
            self.assertEqual(len(validated), 42)

    def test_aggregator_uses_paired_replicates_for_radius_gains(self):
        model_rows, bootstrap_rows = [], []
        for layer_from, layer_to in common.TRANSITIONS:
            for variant_index, variant in enumerate(common.VARIANTS):
                model_rows.append({
                    "layer_from": str(layer_from), "layer_to": str(layer_to),
                    "variant": variant, "split": "test",
                    "pair_r2": str(variant_index),
                    "mean_per_protein_r2": str(variant_index + 0.25),
                })
                for replicate in range(common.BOOTSTRAP_REPLICATES):
                    for metric, offset in (("pair_r2", 0.0), ("mean_per_protein_r2", 0.25)):
                        bootstrap_rows.append({
                            "layer_from": str(layer_from), "layer_to": str(layer_to),
                            "variant": variant, "bootstrap_replicate": str(replicate),
                            "metric": metric,
                            "value": str(variant_index + offset + replicate * 1e-4),
                        })
        rows = aggregate.variant_gain_rows(model_rows, bootstrap_rows)
        aggregate_rows = [row for row in rows if row["scope"] == "equal_transition_mean"]
        self.assertEqual(len(aggregate_rows), 12)
        for row in rows:
            self.assertAlmostEqual(float(row["point_gain"]), 1.0)
            self.assertAlmostEqual(float(row["bootstrap_mean_gain"]), 1.0)
            self.assertEqual(int(row["n_paired_replicates"]), 200)

    def test_synthetic_shard_writes_declared_outputs(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            csv_root = root / "csv"
            pair_root = root / "pairs"
            pdb_root = root / "pdb"
            source_root = csv_root / common.SOURCE_VARIANT
            source_root.mkdir(parents=True)
            proteins = [f"toy{index}_A" for index in range(6)]
            with (source_root / "split_assignments.csv").open("w", newline="") as handle:
                writer = csv.DictWriter(
                    handle, fieldnames=("split_id", "protein_id", "split", "n_residues")
                )
                writer.writeheader()
                for index, protein_id in enumerate(proteins):
                    writer.writerow({
                        "split_id": 0, "protein_id": protein_id,
                        "split": "train" if index < 4 else "test", "n_residues": 14,
                    })
            write_distance_coefficients(source_root / "coefficients.csv", (0, 1))
            for index, protein_id in enumerate(proteins):
                directory = pair_root / protein_id
                directory.mkdir(parents=True)
                pb = toy_pair_block(14) + index * 0.5
                np.save(directory / f"{protein_id}_pair_block_0.npy", pb)
                np.save(directory / f"{protein_id}_pair_block_1.npy", pb * 1.01 + 0.25)
                write_toy_pdb(
                    pdb_root
                    / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
                    / "predictions" / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
                    14,
                )
            n_pairs = common.local_pair_count(14)
            output_root = root / "output"
            environment = {
                "SLURM_ARRAY_TASK_ID": "0", "VARIANT_GROUP": "all",
                "CSV_ROOT": str(csv_root), "CAMEO_PAIR_BLOCKS_ROOT": str(pair_root),
                "CAMEO_PDB_ROOT": str(pdb_root),
                "LOCAL_PAIR_REP_UPDATE_OUTPUT_ROOT": str(output_root),
            }
            with mock.patch.dict(os.environ, environment, clear=False), mock.patch.object(
                probe, "EXPECTED_COUNTS", (4, 2, 4 * n_pairs, 2 * n_pairs)
            ):
                probe.main()
            shard = output_root / "shards" / "transition_00_01" / "center"
            for filename in aggregate.REQUIRED_FILES:
                self.assertTrue((shard / filename).exists(), filename)
            metadata = json.loads((shard / "run_metadata.json").read_text())
            self.assertEqual(metadata["status"], "complete")
            self.assertEqual(metadata["feature_dim"], 128)
            with (shard / "aa_pair_target_statistics.csv").open(newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 420)

    def test_csv_writer_preserves_explicit_empty_schema(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "empty.csv"
            common.write_csv_atomic(path, [], ("a", "b"))
            self.assertEqual(path.read_text(), "a,b\n")


if __name__ == "__main__":
    unittest.main()

import csv
import json
import sys
import unittest
from collections import Counter
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CLUSTER_DIR = ROOT / "scripts" / "cluster_scripts"
if str(CLUSTER_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_DIR))

import regression_protocol as protocol


class MasterManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.entries = protocol.load_master_manifest()

    def test_frozen_counts_and_digest(self):
        self.assertEqual(len(self.entries), 554)
        self.assertEqual(Counter(entry.source for entry in self.entries), {"cameo": 154, "ecod": 400})
        self.assertEqual(
            Counter(entry.master_role for entry in self.entries),
            {"fit": 330, "validation": 111, "test": 111, "excluded": 2},
        )
        self.assertEqual(protocol.manifest_digest(self.entries), protocol.MASTER_MANIFEST_DIGEST)
        self.assertEqual(
            {entry.protein_id for entry in self.entries if not entry.eligible},
            {"000196893", "001681437"},
        )

    def test_size_bin_balance(self):
        counts = Counter(
            (entry.master_role, entry.size_bin) for entry in self.entries if entry.eligible
        )
        self.assertEqual([counts[("fit", size_bin)] for size_bin in range(5)], [66] * 5)
        self.assertEqual(
            [counts[("validation", size_bin)] for size_bin in range(5)],
            [22, 22, 23, 22, 22],
        )
        self.assertEqual(
            [counts[("test", size_bin)] for size_bin in range(5)],
            [23, 22, 22, 22, 22],
        )

    def test_outer_membership_matches_historical_shared_ols(self):
        historical_path = ROOT / "delta_files" / "general_base256_combined" / "split_assignments.csv"
        if not historical_path.exists():
            self.skipTest("Historical shared-OLS split artifact is not present locally")
        with historical_path.open(newline="") as handle:
            historical = {row["protein_id"]: row["split"] for row in csv.DictReader(handle)}
        canonical = {
            entry.protein_id: "test" if entry.master_role == "test" else "train"
            for entry in self.entries
            if entry.eligible
        }
        self.assertEqual(canonical, historical)

    def test_role_builder_reproduces_frozen_subdivision(self):
        eligible = [entry for entry in self.entries if entry.eligible]
        records = [
            {
                "protein_id": entry.protein_id,
                "n_residues": entry.n_residues,
                "size_bin": entry.size_bin,
            }
            for entry in eligible
        ]
        locked = {entry.protein_id for entry in eligible if entry.master_role == "test"}
        roles = protocol.build_master_roles(records, locked)
        self.assertEqual(roles, {entry.protein_id: entry.master_role for entry in eligible})


class FivefoldManifestTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.master = protocol.load_master_manifest()
        cls.outer = protocol.load_outer_fold_manifest(master_entries=cls.master)
        cls.pair_folds = protocol.load_pair_fold_manifest(master_entries=cls.master)

    def test_protocol_versions_are_separate_and_incremented(self):
        self.assertEqual(protocol.HISTORICAL_PROTOCOL_VERSION, "1.0")
        self.assertEqual(protocol.PROTOCOL_VERSION, "2.0")

    def test_outer_fold_counts_stratification_and_frozen_fold_zero(self):
        counts = Counter(entry.outer_fold for entry in self.outer)
        self.assertEqual(counts, {0: 111, 1: 111, 2: 110, 3: 110, 4: 110})
        historical_test = {
            entry.protein_id
            for entry in self.master
            if entry.eligible and entry.master_role == "test"
        }
        self.assertEqual(
            {entry.protein_id for entry in self.outer if entry.outer_fold == 0},
            historical_test,
        )
        by_fold_bin = Counter((entry.outer_fold, entry.length_bin) for entry in self.outer)
        self.assertEqual(
            [[by_fold_bin[(fold, length_bin)] for length_bin in range(5)] for fold in range(5)],
            [
                [23, 22, 22, 22, 22],
                [22, 22, 23, 22, 22],
                [22, 22, 22, 22, 22],
                [22, 22, 22, 22, 22],
                [22, 22, 22, 22, 22],
            ],
        )

    def test_outer_fold_determinism_and_digest(self):
        rebuilt = protocol.build_outer_fold_manifest(self.master)
        self.assertEqual(rebuilt, self.outer)
        self.assertEqual(
            protocol.outer_fold_manifest_digest(self.outer),
            protocol.OUTER_FOLD_MANIFEST_DIGEST,
        )

    def test_pair_fold_counts_disjointness_and_determinism(self):
        self.assertEqual(len(self.pair_folds), 552)
        for row in self.pair_folds:
            self.assertEqual(sum(row.fold_counts), row.n_pairs)
            self.assertLessEqual(max(row.fold_counts) - min(row.fold_counts), 1)

        representative = max(self.pair_folds, key=lambda row: row.n_pairs)
        first = protocol.within_protein_pair_folds(
            representative.n_residues, representative.protein_id
        )
        second = protocol.within_protein_pair_folds(
            representative.n_residues, representative.protein_id
        )
        np.testing.assert_array_equal(first, second)
        heldout_sets = [set(np.flatnonzero(first == fold)) for fold in range(5)]
        self.assertEqual(sum(len(indices) for indices in heldout_sets), representative.n_pairs)
        self.assertEqual(set.union(*heldout_sets), set(range(representative.n_pairs)))
        self.assertTrue(
            all(
                heldout_sets[left].isdisjoint(heldout_sets[right])
                for left in range(5)
                for right in range(left + 1, 5)
            )
        )

    def test_pair_fold_manifest_and_metadata_digests(self):
        self.assertEqual(
            protocol.pair_fold_manifest_digest(self.pair_folds),
            protocol.PAIR_FOLD_MANIFEST_DIGEST,
        )
        payload = protocol.load_fold_digests(
            master_entries=self.master,
            outer_entries=self.outer,
            pair_entries=self.pair_folds,
        )
        self.assertEqual(payload["cohort_digest"], protocol.ELIGIBLE_COHORT_DIGEST)
        self.assertEqual(payload["outer_fold_digest"], protocol.OUTER_FOLD_MANIFEST_DIGEST)
        self.assertEqual(
            payload["pair_fold_manifest_digest"],
            protocol.PAIR_FOLD_MANIFEST_DIGEST,
        )
        with protocol.FOLD_DIGESTS_PATH.open() as handle:
            self.assertEqual(json.load(handle), payload)


class PairAndFeatureProtocolTests(unittest.TestCase):
    def test_within_protein_pair_split_is_pinned(self):
        train, heldout = protocol.within_protein_pair_split(10, "protein-A")
        self.assertEqual(train.tolist(), [0, 1, 2, 3, 4, 5, 7, 8])
        self.assertEqual(heldout.tolist(), [6, 9])
        self.assertEqual(
            protocol.within_protein_pair_split_digest(10, "protein-A"),
            "27adff8472c25d54ff8522e8b7942e28df3a44ed64d25112465102d8f41c5030",
        )

    def test_directional_feature_order_and_precision(self):
        pair_block = np.empty((3, 3, protocol.N_CHANNELS), dtype=np.float32)
        channels = np.arange(protocol.N_CHANNELS, dtype=np.float32)
        for i in range(3):
            for j in range(3):
                pair_block[i, j] = i * 10000 + j * 100 + channels
        pair_i, pair_j = protocol.strict_lower_triangle_indices(3)
        features = protocol.extract_directional_base256(pair_block, pair_i, pair_j)
        self.assertEqual(features.dtype, np.float64)
        self.assertEqual(features.shape, (3, 256))
        np.testing.assert_array_equal(features[0, :128], pair_block[1, 0])
        np.testing.assert_array_equal(features[0, 128:], pair_block[0, 1])
        self.assertEqual(protocol.FEATURE_NAMES[0], "pair_ij_ch_000")
        self.assertEqual(protocol.FEATURE_NAMES[128], "pair_ji_ch_000")

    def test_category_boundaries_and_interchain_flag(self):
        self.assertEqual(len(protocol.AMINO_ACID_PAIR_LABELS), 210)
        pair_i = np.asarray([1, 4, 9, 19, 49, 50])
        pair_j = np.zeros(6, dtype=np.int64)
        np.testing.assert_array_equal(
            protocol.sequence_separation_bin_codes(pair_i, pair_j),
            np.arange(6, dtype=np.int16),
        )
        chains = np.asarray(["A"] * 51)
        chains[50] = "B"
        self.assertEqual(
            protocol.sequence_separation_bin_codes(pair_i, pair_j, chain_ids=chains)[-1], -1
        )
        distances = np.asarray([0.0, 4.999, 5.0, 9.999, 10.0, 19.999, 20.0, 39.999, 40.0])
        np.testing.assert_array_equal(
            protocol.spatial_distance_bin_codes(distances),
            np.asarray([0, 0, 1, 1, 2, 2, 3, 3, 4], dtype=np.int16),
        )

    def test_protein_equal_weights_mean_and_moments(self):
        weights = protocol.protein_equal_weights(["a", "a", "b", "b", "b"])
        self.assertAlmostEqual(float(np.sum(weights[:2])), 1.0)
        self.assertAlmostEqual(float(np.sum(weights[2:])), 1.0)
        self.assertAlmostEqual(
            protocol.protein_equal_mean([np.asarray([1.0, 3.0]), np.asarray([10.0])]), 6.0
        )

        first = np.asarray([[1.0, 2.0], [3.0, 4.0]])
        second = np.asarray([[10.0, 20.0]])
        moments = protocol.ProteinEqualFeatureMoments()
        moments.update_protein(first)
        other = protocol.ProteinEqualFeatureMoments()
        other.update_protein(second)
        moments.merge(other)
        mean, scale = moments.finalize()
        expected_mean = (np.mean(first, axis=0) + second[0]) / 2.0
        expected_second = (np.mean(first * first, axis=0) + second[0] ** 2) / 2.0
        np.testing.assert_allclose(mean, expected_mean)
        np.testing.assert_allclose(scale, np.sqrt(expected_second - expected_mean ** 2))

    def test_nonfinite_features_are_rejected(self):
        pair_block = np.zeros((2, 2, 128), dtype=np.float32)
        pair_block[1, 0, 0] = np.nan
        with self.assertRaises(ValueError):
            protocol.extract_directional_base256(pair_block, np.asarray([1]), np.asarray([0]))


if __name__ == "__main__":
    unittest.main()

import csv
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

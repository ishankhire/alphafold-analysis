import csv
import importlib.util
import tempfile
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "diagnostic_secondary_structure_sequence_gated.py"
SPEC = importlib.util.spec_from_file_location("secondary_structure_sequence_gated", MODULE_PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


class SecondaryStructureSequenceGatedTests(unittest.TestCase):
    def test_sequence_gate_boundaries_and_interchain_rejection(self):
        pair_i = np.asarray([1, 4, 9, 19, 49, 50, 7])
        pair_j = np.zeros_like(pair_i)
        chains = np.asarray(["A"] * 51)
        chains[7] = "B"
        codes = diagnostic.sequence_gate_codes(pair_i, pair_j, chains)
        self.assertEqual(codes.tolist(), [0, 1, 2, 3, 4, 5, -1])

    def test_sparse_decoder_selects_gate_and_preserves_direction(self):
        cache = {
            "chains": np.asarray(["A"] * 8),
            "dssp_codes": np.asarray(list("H-------")),
            "bridge_id": np.empty(0, dtype=np.int32),
            "bridge_i": np.empty(0, dtype=np.int32),
            "bridge_j": np.empty(0, dtype=np.int32),
            "bridge_main_eligible": np.empty(0, dtype=np.int8),
            "control_bridge_id": np.empty(0, dtype=np.int32),
            "control_i": np.empty(0, dtype=np.int32),
            "control_j": np.empty(0, dtype=np.int32),
        }
        pair_block = np.zeros((8, 8, diagnostic.N_CHANNELS), dtype=np.float32)
        pair_block[2, 0, 0] = 2.0
        pair_block[0, 2, 0] = 5.0
        intercepts = np.arange(6, dtype=np.float64) + 10.0
        coefficients = np.zeros((6, diagnostic.FEATURE_DIM), dtype=np.float64)
        coefficients[1, 0] = 3.0
        coefficients[1, diagnostic.N_CHANNELS] = 7.0
        decoded, predicted = diagnostic.decode_needed_pairs(
            pair_block, cache, intercepts, coefficients
        )
        self.assertEqual(decoded[2, 0], 10.0 + 1.0 + 2.0 * 3.0 + 5.0 * 7.0)
        self.assertEqual(decoded[0, 2], decoded[2, 0])
        self.assertEqual(len(predicted), 3)
        self.assertTrue(np.isnan(decoded[7, 0]))

    def test_needed_pairs_include_offsets_bridges_and_controls_once(self):
        cache = {
            "chains": np.asarray(["A"] * 8),
            "dssp_codes": np.asarray(list("H-------")),
            "bridge_id": np.asarray([4], dtype=np.int32),
            "bridge_i": np.asarray([1], dtype=np.int32),
            "bridge_j": np.asarray([7], dtype=np.int32),
            "bridge_main_eligible": np.asarray([1], dtype=np.int8),
            "control_bridge_id": np.asarray([4, 4], dtype=np.int32),
            "control_i": np.asarray([7, 0], dtype=np.int32),
            "control_j": np.asarray([1, 6], dtype=np.int32),
        }
        pair_i, pair_j = diagnostic.needed_secondary_structure_pairs(cache)
        pairs = set(zip(pair_i.tolist(), pair_j.tolist()))
        self.assertEqual(pairs, {(2, 0), (3, 0), (4, 0), (7, 1), (6, 0)})

    def test_loads_all_six_raw_scale_coefficient_vectors(self):
        fields = [
            "analysis_name", "layer", "train_gate_aa_pair",
            "train_gate_sequence_bin", "feature_index", "coefficient_raw_scale",
        ]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coefficients.csv"
            with path.open("w", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                for gate_index, label in enumerate(diagnostic.SEQUENCE_GATE_LABELS):
                    for feature_index in range(-1, diagnostic.FEATURE_DIM):
                        writer.writerow(
                            {
                                "analysis_name": "analysis_1_sequence_gated", "layer": 10,
                                "train_gate_aa_pair": "all", "train_gate_sequence_bin": label,
                                "feature_index": feature_index,
                                "coefficient_raw_scale": gate_index * 1000 + feature_index,
                            }
                        )
            intercepts, coefficients = diagnostic.load_gated_coefficients(Path(directory), 10)
        np.testing.assert_array_equal(intercepts, np.arange(6) * 1000 - 1)
        self.assertEqual(coefficients.shape, (6, diagnostic.FEATURE_DIM))
        self.assertEqual(coefficients[5, 255], 5255)


if __name__ == "__main__":
    unittest.main()

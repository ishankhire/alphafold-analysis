import importlib.util
import unittest
from pathlib import Path

import numpy as np


MODULE_PATH = Path(__file__).parents[1] / "diagnostic_secondary_structure_emergence.py"
SPEC = importlib.util.spec_from_file_location("secondary_structure_emergence", MODULE_PATH)
diagnostic = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(diagnostic)


def dssp_line(index, resseq, chain, code, label_1=" ", bp_1=0, label_2=" ", bp_2=0):
    chars = [" "] * 140
    chars[0:5] = f"{index:5d}"
    chars[5:10] = f"{resseq:5d}"
    chars[10] = " "
    chars[11] = chain
    chars[13] = "A"
    chars[16] = code
    chars[23] = label_1
    chars[24] = label_2
    chars[25:29] = f"{bp_1:4d}"
    chars[29:33] = f"{bp_2:4d}"
    return "".join(chars)


class SecondaryStructureEmergenceTests(unittest.TestCase):
    def test_parse_dssp_bridge_orientation_and_partners(self):
        raw = "\n".join(
            [
                "  #  RESIDUE AA STRUCTURE BP1 BP2",
                dssp_line(1, 10, "A", "E", "a", 8, "B", 9),
            ]
        )
        rows = diagnostic.parse_dssp_bridge_rows(raw)
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["dssp_index"], 1)
        self.assertEqual(
            rows[0]["partners"],
            [(8, "parallel", "a"), (9, "antiparallel", "B")],
        )

    def test_directional_raw_coefficient_replay(self):
        pair_block = np.zeros((3, 3, diagnostic.N_CHANNELS), dtype=np.float32)
        pair_block[2, 0, 0] = 2.0
        pair_block[0, 2, 0] = 5.0
        coefficients = np.zeros(diagnostic.FEATURE_DIM, dtype=np.float64)
        coefficients[0] = 3.0
        coefficients[diagnostic.N_CHANNELS] = 7.0
        matrix, pair_i, pair_j, predicted = diagnostic.decode_distances(
            pair_block, 11.0, coefficients
        )
        location = np.flatnonzero((pair_i == 2) & (pair_j == 0))[0]
        expected = 11.0 + 2.0 * 3.0 + 5.0 * 7.0
        self.assertEqual(predicted[location], expected)
        self.assertEqual(matrix[2, 0], expected)
        self.assertEqual(matrix[0, 2], expected)
        np.testing.assert_array_equal(np.diag(matrix), 0.0)

    def test_segments_do_not_cross_chain_boundaries(self):
        codes = np.asarray(list("HHHH"), dtype="U1")
        chains = np.asarray(["A", "A", "B", "B"])
        segment_ids, segments = diagnostic.build_segments(codes, chains, "H")
        self.assertEqual(segment_ids.tolist(), [0, 0, 1, 1])
        self.assertEqual([row["length"] for row in segments], [2, 2])

    def test_helix_gate_requires_relative_and_absolute_conditions(self):
        calibration = {
            "zscore_mean": [0.0, 0.0, 0.0],
            "zscore_scale": [1.0, 1.0, 1.0],
            "mu_helix_z": [0.0, 0.0, 0.0],
            "mu_strand_z": [3.0, 3.0, 3.0],
        }
        triples = np.asarray([[0.1, 0.1, 0.1], [-10.0, -10.0, -10.0]])
        scores, closeness, _ = diagnostic.score_triples(triples, calibration)
        self.assertGreater(scores[0], 0)
        self.assertLess(closeness[0], 1)
        self.assertGreater(scores[1], 0)
        self.assertGreater(closeness[1], 1)

    def test_matched_random_controls_are_deterministic_and_exactly_matched(self):
        residues = [
            {
                "chain": "A", "pdb_resseq": index + 1, "insertion_code": "",
                "hetflag": " ", "resname": "ALA", "coordinate": np.zeros(3),
            }
            for index in range(40)
        ]
        codes = np.asarray(["E"] * 40)
        segment_ids = np.zeros(40, dtype=np.int32)
        bridges = [
            {"bridge_id": 0, "i": 2, "j": 22, "orientation": "parallel", "main_eligible": 1}
        ]
        first = diagnostic.choose_bridge_controls(
            "protein", residues, codes, segment_ids, bridges
        )
        second = diagnostic.choose_bridge_controls(
            "protein", residues, codes, segment_ids, bridges
        )
        self.assertEqual(first, second)
        random_controls = [row for row in first if row["type"] == "matched_random"]
        self.assertLessEqual(len(random_controls), diagnostic.N_MATCHED_RANDOM)
        self.assertTrue(all(abs(row["i"] - row["j"]) == 20 for row in random_controls))
        self.assertTrue(
            all(tuple(sorted((row["i"], row["j"]))) != (2, 22) for row in random_controls)
        )

    def test_trajectory_flags_capture_reversion(self):
        rows = [
            {
                "split": "test", "protein_id": "p", "segment_id": "0",
                "layer": layer, "mean_score": score,
                "mean_distance_to_helix_centroid": distance, "visible": visible,
            }
            for layer, score, distance, visible in [
                (0, -1.0, 3.0, 0), (10, 1.0, 1.0, 1), (20, 0.5, 2.0, 0)
            ]
        ]
        diagnostic.add_helix_trajectory_fields(rows)
        self.assertTrue(all(row["first_visible_layer"] == 10 for row in rows))
        self.assertTrue(all(row["crossed_then_reverted"] == 1 for row in rows))
        self.assertTrue(all(row["score_monotone_nondecreasing"] == 0 for row in rows))

    def test_grouped_accuracy_uses_exact_pair_moments(self):
        true = np.asarray([1.0, 2.0, 10.0, 12.0])
        predicted = np.asarray([2.0, 0.0, 11.0, 15.0])
        groups = np.asarray([0, 0, 1, 1], dtype=np.int8)
        distance_bins = np.asarray([0, 0, 1, 1], dtype=np.int8)
        rows = diagnostic.grouped_accuracy_rows(
            0, "test", groups, ["first", "second"], distance_bins,
            true, predicted, include_distance_bins=False,
        )
        first = next(row for row in rows if row["group"] == "first")
        self.assertEqual(first["n_pairs"], 2)
        self.assertAlmostEqual(first["mae"], 1.5)
        self.assertAlmostEqual(first["rmse"], np.sqrt(2.5))
        self.assertAlmostEqual(first["bias"], -0.5)
        self.assertAlmostEqual(first["r2"], -9.0)

    def test_helix_rows_trim_segment_endpoints(self):
        n = 12
        codes = np.asarray(["H"] * 8 + ["-"] * 4)
        chains = np.asarray(["A"] * n)
        segment_ids, segments = diagnostic.build_segments(codes, chains, "H")
        cache = {
            "dssp_codes": codes, "chains": chains,
            "h_segment_ids": segment_ids,
            "h_segment_start": np.asarray([segments[0]["start"]]),
            "h_segment_end": np.asarray([segments[0]["end"]]),
            "h_segment_length": np.asarray([segments[0]["length"]]),
            "pdb_resseq": np.arange(1, n + 1),
            "insertion_codes": np.asarray([""] * n),
            "resnames": np.asarray(["ALA"] * n),
        }
        coordinates = np.arange(n, dtype=np.float64)[:, None] * np.asarray([[3.8, 0.0, 0.0]])
        distances = diagnostic.distance_matrix(coordinates)
        calibration = {
            "zscore_mean": [0.0, 0.0, 0.0], "zscore_scale": [1.0, 1.0, 1.0],
            "mu_helix_z": [0.0, 0.0, 0.0], "mu_strand_z": [20.0, 20.0, 20.0],
            "mu_helix_raw": [0.0, 0.0, 0.0],
            "helix_score_threshold_95_specificity": 0.0,
            "helix_closeness_threshold_95_sensitivity": 100.0,
        }
        assignment = {"split": "test", "protein_id": "p", "source": "CAMEO"}
        rows, segment_rows = diagnostic.build_helix_rows(
            0, assignment, cache, distances, distances, calibration
        )
        indices = {row["residue_index"] for row in rows if row["dssp_code"] == "H"}
        self.assertNotIn(0, indices)
        self.assertNotIn(7, indices)
        self.assertEqual(len(segment_rows), 1)
        self.assertEqual(segment_rows[0]["original_segment_length"], 8)


if __name__ == "__main__":
    unittest.main()

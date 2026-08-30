"""Independent geometry and exact-triangle oracles for protocol-2.0 review."""

from __future__ import annotations

import itertools
import math
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
CV_DIR = TEST_DIR.parent
CLUSTER_DIR = CV_DIR.parent
for directory in (CLUSTER_DIR, CV_DIR, CLUSTER_DIR / "triangle_violation_subgroups"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import cluster_regression_helpers as geometry
import cross_validation_common as cv
import triangle_violation_subgroup_common as subgroup
from regression_protocol import strict_lower_triangle_indices


TOLERANCE = geometry.TRIANGLE_VIOLATION_TOLERANCE


def _literal_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    result = np.empty((points.shape[0], points.shape[0]), dtype=np.float64)
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            result[i, j] = math.sqrt(
                sum((float(points[i, axis]) - float(points[j, axis])) ** 2 for axis in range(points.shape[1]))
            )
    return result


def _literal_gram_eigenvalues(distance_matrix: np.ndarray) -> np.ndarray:
    distances = np.asarray(distance_matrix, dtype=np.float64)
    n = distances.shape[0]
    centering = np.eye(n, dtype=np.float64) - np.ones((n, n), dtype=np.float64) / n
    gram = -0.5 * centering @ np.square(distances) @ centering
    return np.linalg.eigvalsh(gram)[::-1]


def _literal_triangle_metrics(distance_matrix: np.ndarray) -> dict[str, float | int]:
    distances = np.asarray(distance_matrix, dtype=np.float64)
    deltas: list[float] = []
    for i, j, k in itertools.combinations(range(distances.shape[0]), 3):
        sides = [float(distances[i, j]), float(distances[i, k]), float(distances[j, k])]
        longest = max(sides)
        raw = longest - (sum(sides) - longest)
        deltas.append(raw if raw > TOLERANCE else 0.0)
    positives = [value for value in deltas if value > 0.0]
    n_triangles = len(deltas)
    n_0_5 = sum(value < 5.0 for value in positives)
    n_5_10 = sum(5.0 <= value < 10.0 for value in positives)
    n_10_plus = sum(value >= 10.0 for value in positives)
    return {
        "n_triangles": n_triangles,
        "n_violations": len(positives),
        "triangle_violation_fraction": len(positives) / n_triangles if n_triangles else 0.0,
        "triangle_mean_violation": sum(positives) / n_triangles if n_triangles else 0.0,
        "triangle_max_violation": max(positives, default=0.0),
        "triangle_no_violation_fraction": (n_triangles - len(positives)) / n_triangles if n_triangles else 0.0,
        "triangle_violation_0_5_fraction": n_0_5 / n_triangles if n_triangles else 0.0,
        "triangle_violation_5_10_fraction": n_5_10 / n_triangles if n_triangles else 0.0,
        "triangle_violation_10_plus_fraction": n_10_plus / n_triangles if n_triangles else 0.0,
    }


def _predictions(distance_matrix: np.ndarray) -> np.ndarray:
    pair_i, pair_j = strict_lower_triangle_indices(distance_matrix.shape[0])
    return np.asarray(distance_matrix[pair_i, pair_j], dtype=np.float64)


def _geometry_row(
    coordinates: np.ndarray,
    predicted_distances: np.ndarray,
    *,
    protein_id: str = "oracle",
) -> dict[str, object]:
    return cv.build_geometry_row(
        analysis_name="adversarial_geometry",
        model_family="shared_ols",
        feature_dim=2,
        layer=0,
        protein_id=protein_id,
        source="cameo",
        outer_fold=0,
        evaluation_design="protein_outer_5fold",
        oof_prediction_scope="heldout_proteins_all_pairs",
        n_residues=int(coordinates.shape[0]),
        true_coordinates=coordinates,
        predictions=_predictions(predicted_distances),
    )


def _nondegenerate_coordinates() -> np.ndarray:
    return np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.7, 0.2, -0.1],
            [0.3, 2.1, 0.4],
            [-0.2, 0.5, 2.4],
            [1.2, 1.6, 1.1],
            [2.0, -0.4, 1.5],
        ],
        dtype=np.float64,
    )


def _subgroup_identity(n_residues: int) -> dict[str, object]:
    return {
        "analysis_name": "oracle",
        "model_family": "shared_ols",
        "layer": 0,
        "protein_id": "p",
        "source": "cameo",
        "protein_split": "test",
        "generalization_scope": "unseen_proteins",
        "n_residues": n_residues,
        "protein_length_bin": "length_bin_1",
    }


class MDSAlignmentAndTMTests(unittest.TestCase):
    def test_exact_three_dimensional_reconstruction_including_float32_boundary(self):
        coordinates = _nondegenerate_coordinates()
        distances = _literal_distance_matrix(coordinates)
        eigenvalues = _literal_gram_eigenvalues(distances)
        positive_mass = sum(max(float(value), 0.0) for value in eigenvalues)
        expected_top3 = sum(max(float(value), 0.0) for value in eigenvalues[:3]) / positive_mass
        expected_negative_mass = sum(abs(float(value)) for value in eigenvalues if value < 0.0) / positive_mass

        for values, tolerance in ((distances, 1e-10), (distances.astype(np.float32), 2e-6)):
            row = _geometry_row(coordinates, values)
            self.assertEqual(row["geometry_status"], "complete")
            self.assertLess(float(row["rmsd"]), tolerance)
            self.assertGreater(float(row["tm_score"]), 1.0 - tolerance)
            self.assertLess(float(row["stress"]), tolerance)
            self.assertAlmostEqual(float(row["frac_top3_eigenvalues"]), expected_top3, delta=tolerance)
            self.assertLess(float(row["negative_eigenvalue_fraction"]), max(1e-12, expected_negative_mass + tolerance))
            self.assertEqual(float(row["triangle_violation_fraction"]), 0.0)

    def test_known_rigid_transform_is_removed_by_proper_kabsch_alignment(self):
        fixed = _nondegenerate_coordinates()
        angle = 0.73
        rotation = np.asarray(
            [
                [math.cos(angle), -math.sin(angle), 0.0],
                [math.sin(angle), math.cos(angle), 0.0],
                [0.0, 0.0, 1.0],
            ]
        )
        translation = np.asarray([4.0, -2.5, 7.2])
        moving = fixed @ rotation.T + translation
        transformed, rmsd = geometry.superimpose_structures(fixed, moving)
        self.assertLess(float(rmsd), 1e-12)
        np.testing.assert_allclose(transformed, fixed, rtol=0.0, atol=1e-12)
        self.assertAlmostEqual(float(np.linalg.det(rotation)), 1.0, places=12)

    def test_reflection_branch_is_explicitly_reflection_invariant(self):
        fixed = _nondegenerate_coordinates()
        mirrored = fixed.copy()
        mirrored[:, 0] *= -1.0
        _, proper_only_rmsd = geometry.superimpose_structures(fixed, mirrored)
        self.assertGreater(float(proper_only_rmsd), 0.1)

        distances = _literal_distance_matrix(fixed)
        row = _geometry_row(fixed, distances)
        self.assertEqual(row["geometry_status"], "complete")
        self.assertLess(float(row["rmsd"]), 1e-10)
        self.assertGreater(float(row["tm_score"]), 1.0 - 1e-12)

    def test_four_dimensional_distances_are_metric_but_not_exactly_3d(self):
        points = np.asarray(
            [
                [0.0, 0.0, 0.0, 0.0],
                [1.0, 0.0, 0.0, 0.0],
                [0.0, 1.0, 0.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
                [1.0, 1.0, 1.0, 1.0],
            ],
            dtype=np.float64,
        )
        distances = _literal_distance_matrix(points)
        row = _geometry_row(points[:, :3], distances)
        self.assertEqual(row["geometry_status"], "complete")
        self.assertEqual(float(row["triangle_violation_fraction"]), 0.0)
        self.assertLess(float(row["frac_top3_eigenvalues"]), 0.999)
        self.assertLess(float(row["negative_eigenvalue_fraction"]), 1e-12)
        self.assertGreater(float(row["stress"]), 1e-3)

    def test_triangle_valid_cycle_metric_has_negative_gram_mass(self):
        distances = np.asarray(
            [
                [0.0, 1.0, 2.0, 1.0],
                [1.0, 0.0, 1.0, 2.0],
                [2.0, 1.0, 0.0, 1.0],
                [1.0, 2.0, 1.0, 0.0],
            ]
        )
        oracle = _literal_triangle_metrics(distances)
        eigenvalues = _literal_gram_eigenvalues(distances)
        self.assertEqual(oracle["n_violations"], 0)
        self.assertTrue(any(value < -1e-12 for value in eigenvalues))
        row = _geometry_row(_nondegenerate_coordinates()[:4], distances)
        self.assertEqual(row["geometry_status"], "complete")
        self.assertEqual(float(row["triangle_violation_fraction"]), 0.0)
        self.assertGreater(float(row["negative_eigenvalue_fraction"]), 0.0)
        self.assertGreater(float(row["stress"]), 0.0)

    def test_residue_permutation_preserves_all_geometry_metrics(self):
        coordinates = _nondegenerate_coordinates()
        distances = _literal_distance_matrix(coordinates)
        original = _geometry_row(coordinates, distances, protein_id="original")
        order = np.asarray([4, 1, 5, 0, 3, 2])
        permuted = _geometry_row(
            coordinates[order], distances[np.ix_(order, order)], protein_id="permuted",
        )
        invariant_metrics = tuple(
            metric for metric in cv.GEOMETRY_METRIC_NAMES
            if metric != "n_negative_eigenvalues"
        )
        for metric in invariant_metrics:
            self.assertAlmostEqual(float(original[metric]), float(permuted[metric]), delta=1e-10)

    def test_scale_metamorphism_and_absolute_severity_band_crossing(self):
        coordinates = _nondegenerate_coordinates()[:5]
        distances = _literal_distance_matrix(coordinates)
        distances[0, 1] = distances[1, 0] = distances[0, 2] + distances[2, 1] + 2.0
        factor = 3.0
        original = _geometry_row(coordinates, distances, protein_id="scale-one")
        scaled = _geometry_row(coordinates * factor, distances * factor, protein_id="scale-three")
        self.assertEqual(original["geometry_status"], "complete")
        self.assertEqual(scaled["geometry_status"], "complete")
        for metric in (
            "triangle_violation_fraction",
            "stress",
            "frac_top3_eigenvalues",
            "negative_eigenvalue_fraction",
        ):
            self.assertAlmostEqual(float(original[metric]), float(scaled[metric]), delta=1e-10)
        for metric in ("rmsd", "triangle_mean_violation", "triangle_max_violation"):
            self.assertAlmostEqual(float(scaled[metric]), factor * float(original[metric]), delta=1e-9)
        self.assertGreater(float(original["triangle_violation_0_5_fraction"]), 0.0)
        self.assertGreater(float(scaled["triangle_violation_5_10_fraction"]), 0.0)

    def test_degenerate_inputs_are_complete_or_explicitly_failed(self):
        cases = (
            np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
            np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0]]),
            np.zeros((3, 3), dtype=np.float64),
            np.asarray([[0.0, 0.0, 0.0], [1e-100, 0.0, 0.0], [0.0, 1e-100, 0.0]]),
        )
        for index, coordinates in enumerate(cases):
            with np.errstate(divide="ignore", invalid="ignore"):
                row = _geometry_row(
                    coordinates, _literal_distance_matrix(coordinates), protein_id=f"degenerate-{index}",
                )
            self.assertIn(row["geometry_status"], {"complete", "geometry_failed"})
            if row["geometry_status"] == "complete":
                self.assertTrue(all(np.isfinite(float(row[name])) for name in cv.GEOMETRY_METRIC_NAMES))
            else:
                self.assertTrue(str(row["geometry_exclusion_reason"]))

    def test_negative_eigenvalue_literal_sign_count_and_mass_are_auditable(self):
        coordinates = _nondegenerate_coordinates()
        for factor in (1e-4, 1.0, 1e4):
            distances = _literal_distance_matrix(coordinates * factor)
            eigenvalues = _literal_gram_eigenvalues(distances)
            positive_mass = sum(float(value) for value in eigenvalues if value > 0.0)
            negative_mass = sum(abs(float(value)) for value in eigenvalues if value < 0.0)
            row = _geometry_row(coordinates * factor, distances)
            _, implementation_eigenvalues = geometry.classical_mds(distances)
            self.assertEqual(
                int(row["n_negative_eigenvalues"]),
                int(np.count_nonzero(implementation_eigenvalues < 0.0)),
            )
            self.assertLess(negative_mass / positive_mass, 1e-12)
            self.assertLess(float(row["negative_eigenvalue_fraction"]), 1e-12)

    def test_tm_score_is_literal_post_kabsch_tm_style_formula(self):
        fixed = _nondegenerate_coordinates()
        moving = fixed.copy()
        moving[0] += np.asarray([0.4, -0.2, 0.3])
        transformed, _ = geometry.superimpose_structures(fixed, moving)
        length = fixed.shape[0]
        d0 = max(1.24 * (max(length, 16) - 15) ** (1.0 / 3.0) - 1.8, 0.5)
        residuals = [
            math.sqrt(sum((float(fixed[i, axis]) - float(transformed[i, axis])) ** 2 for axis in range(3)))
            for i in range(length)
        ]
        expected = sum(1.0 / (1.0 + (distance / d0) ** 2) for distance in residuals) / length
        self.assertAlmostEqual(geometry.compute_tm_score(fixed, transformed, length), expected, places=15)


class ExactTriangleTests(unittest.TestCase):
    def test_literal_enumeration_matches_every_count_from_three_through_twelve(self):
        for n_residues in range(3, 13):
            triples = list(itertools.combinations(range(n_residues), 3))
            self.assertEqual(len(triples), math.comb(n_residues, 3))
            self.assertEqual(len(set(triples)), len(triples))
            self.assertTrue(all(len(set(triple)) == 3 for triple in triples))

    def test_hand_constructed_tolerance_and_severity_boundaries(self):
        deltas = (
            0.0,
            TOLERANCE / 2.0,
            TOLERANCE * 2.0,
            5.0 - 1e-9,
            5.0,
            10.0 - 1e-9,
            10.0,
        )
        for delta in deltas:
            distances = np.asarray(
                [[0.0, 1.0, 2.0 + delta], [1.0, 0.0, 1.0], [2.0 + delta, 1.0, 0.0]],
                dtype=np.float64,
            )
            expected = _literal_triangle_metrics(distances)
            observed = geometry.compute_triangle_violations(distances)
            for name in geometry.TRIANGLE_VIOLATION_KEYS:
                self.assertAlmostEqual(float(observed[name]), float(expected[name]), places=14)

        valid = (
            np.asarray([[0.0, 3.0, 5.0], [3.0, 0.0, 4.0], [5.0, 4.0, 0.0]]),
            np.asarray([[0.0, 2.0, 2.0], [2.0, 0.0, 2.0], [2.0, 2.0, 0.0]]),
        )
        self.assertTrue(all(geometry.compute_triangle_violations(case)["triangle_violation_fraction"] == 0.0 for case in valid))

    def test_denominator_identities_use_all_triangles(self):
        distances = np.asarray(
            [
                [0.0, 1.0, 8.0, 3.0],
                [1.0, 0.0, 1.0, 3.0],
                [8.0, 1.0, 0.0, 3.0],
                [3.0, 3.0, 3.0, 0.0],
            ],
            dtype=np.float64,
        )
        oracle = _literal_triangle_metrics(distances)
        observed = geometry.compute_triangle_violations(distances)
        for name in geometry.TRIANGLE_VIOLATION_KEYS:
            self.assertAlmostEqual(float(observed[name]), float(oracle[name]), places=14)
        total_fraction = sum(
            float(observed[name])
            for name in (
                "triangle_violation_0_5_fraction",
                "triangle_violation_5_10_fraction",
                "triangle_violation_10_plus_fraction",
            )
        )
        self.assertAlmostEqual(total_fraction, float(observed["triangle_violation_fraction"]))
        self.assertAlmostEqual(
            float(observed["triangle_no_violation_fraction"]),
            1.0 - float(observed["triangle_violation_fraction"]),
        )
        conditional = float(oracle["triangle_mean_violation"]) * int(oracle["n_triangles"]) / int(oracle["n_violations"])
        self.assertNotAlmostEqual(conditional, float(observed["triangle_mean_violation"]))

    def test_negative_predictions_are_counted_then_clipped_at_public_boundary(self):
        coordinates = np.asarray(
            [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64,
        )
        predictions = np.asarray([-2.0, 4.0, 1.0])
        row = cv.build_geometry_row(
            analysis_name="adversarial_geometry",
            model_family="shared_ols",
            feature_dim=2,
            layer=0,
            protein_id="negative",
            source="cameo",
            outer_fold=0,
            evaluation_design="protein_outer_5fold",
            oof_prediction_scope="heldout_proteins_all_pairs",
            n_residues=3,
            true_coordinates=coordinates,
            predictions=predictions,
        )
        self.assertEqual(row["geometry_status"], "complete")
        self.assertEqual(int(row["n_negative_predictions"]), 1)
        clipped = np.zeros((3, 3), dtype=np.float64)
        pair_i, pair_j = strict_lower_triangle_indices(3)
        clipped[pair_i, pair_j] = np.maximum(predictions, 0.0)
        clipped[pair_j, pair_i] = np.maximum(predictions, 0.0)
        oracle = _literal_triangle_metrics(clipped)
        self.assertAlmostEqual(float(row["triangle_violation_fraction"]), float(oracle["triangle_violation_fraction"]))
        self.assertAlmostEqual(float(row["triangle_mean_violation"]), float(oracle["triangle_mean_violation"]))

    def test_randomized_literal_and_cross_validation_helpers_agree(self):
        for n_residues in range(3, 10):
            rng = np.random.default_rng(4_000 + n_residues)
            lower = rng.uniform(0.0, 15.0, size=(n_residues, n_residues))
            distances = np.tril(lower, k=-1)
            distances += distances.T
            oracle = _literal_triangle_metrics(distances)
            observed = geometry.compute_triangle_violations(distances)
            for name in geometry.TRIANGLE_VIOLATION_KEYS:
                self.assertAlmostEqual(float(observed[name]), float(oracle[name]), places=13)

    def test_permutation_and_scale_metamorphisms_match_literal_oracle(self):
        rng = np.random.default_rng(5_010)
        raw = rng.uniform(1.0, 8.0, size=(7, 7))
        distances = np.tril(raw, k=-1)
        distances += distances.T
        order = np.asarray([6, 2, 5, 0, 3, 1, 4])
        permuted = distances[np.ix_(order, order)]
        scaled = distances * 2.25
        original = geometry.compute_triangle_violations(distances)
        relabeled = geometry.compute_triangle_violations(permuted)
        enlarged = geometry.compute_triangle_violations(scaled)
        for name in geometry.TRIANGLE_VIOLATION_KEYS:
            self.assertAlmostEqual(float(original[name]), float(relabeled[name]), places=13)
        self.assertAlmostEqual(
            float(enlarged["triangle_violation_fraction"]),
            float(original["triangle_violation_fraction"]),
        )
        self.assertAlmostEqual(
            float(enlarged["triangle_mean_violation"]),
            2.25 * float(original["triangle_mean_violation"]),
            places=12,
        )
        self.assertAlmostEqual(
            float(enlarged["triangle_max_violation"]),
            2.25 * float(original["triangle_max_violation"]),
            places=12,
        )

    def test_subgroup_batches_and_cross_validation_primitives_agree(self):
        coordinates = _nondegenerate_coordinates()
        predicted = np.asarray(
            [
                [0.0, 2.0, 9.0, 3.0, 4.0, 6.0],
                [2.0, 0.0, 1.0, 8.0, 3.0, 2.0],
                [9.0, 1.0, 0.0, 2.0, 7.0, 3.0],
                [3.0, 8.0, 2.0, 0.0, 1.0, 9.0],
                [4.0, 3.0, 7.0, 1.0, 0.0, 2.0],
                [6.0, 2.0, 3.0, 9.0, 2.0, 0.0],
            ],
            dtype=np.float64,
        )
        literal = _literal_triangle_metrics(predicted)
        exact = geometry.compute_triangle_violations(predicted)
        self.assertEqual(int(literal["n_triangles"]), math.comb(6, 3))
        for name in geometry.TRIANGLE_VIOLATION_KEYS:
            self.assertAlmostEqual(float(exact[name]), float(literal[name]), places=13)

        all_rows = []
        for batch_size in (1, 4, 7, 19, 100):
            with tempfile.TemporaryDirectory() as directory:
                cache, _ = subgroup.build_reference_cache(coordinates, directory, batch_size=3)
                try:
                    rows = subgroup.compute_metric_rows(
                        predicted, cache, _subgroup_identity(6), batch_size=batch_size,
                    )
                finally:
                    cache.cleanup()
            all_row = next(row for row in rows if row["category_system"] == "all_triangles")
            all_rows.append(all_row)
            self.assertEqual(int(all_row["n_triangles"]), int(literal["n_triangles"]))
            self.assertEqual(int(all_row["n_violations"]), int(literal["n_violations"]))
            self.assertAlmostEqual(float(all_row["violation_fraction"]), float(literal["triangle_violation_fraction"]))
            self.assertAlmostEqual(float(all_row["mean_violation_all"]), float(literal["triangle_mean_violation"]))
            self.assertAlmostEqual(float(all_row["violation_max_angstrom"]), float(literal["triangle_max_violation"]))
            self.assertAlmostEqual(int(all_row["n_abs_0_5"]) / int(all_row["n_triangles"]), float(literal["triangle_violation_0_5_fraction"]))
            self.assertAlmostEqual(int(all_row["n_abs_5_10"]) / int(all_row["n_triangles"]), float(literal["triangle_violation_5_10_fraction"]))
            self.assertAlmostEqual(int(all_row["n_abs_10_plus"]) / int(all_row["n_triangles"]), float(literal["triangle_violation_10_plus_fraction"]))
        primitives = (
            "n_triangles",
            "n_violations",
            "violation_sum_angstrom",
            "violation_max_angstrom",
            "n_abs_0_5",
            "n_abs_5_10",
            "n_abs_10_plus",
        )
        self.assertTrue(all(tuple(row[name] for name in primitives) == tuple(all_rows[0][name] for name in primitives) for row in all_rows[1:]))


class DefensiveDistanceMatrixValidationFindings(unittest.TestCase):
    @unittest.expectedFailure
    def test_classical_mds_rejects_asymmetric_complete_distance_matrix(self):
        asymmetric = np.asarray([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            geometry.classical_mds(asymmetric)

    @unittest.expectedFailure
    def test_classical_mds_rejects_nonzero_diagonal(self):
        nonzero_diagonal = np.asarray([[1.0, 1.0, 2.0], [1.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            geometry.classical_mds(nonzero_diagonal)

    @unittest.expectedFailure
    def test_triangle_helper_rejects_asymmetric_matrix(self):
        asymmetric = np.asarray([[0.0, 1.0, 2.0], [3.0, 0.0, 1.0], [2.0, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            geometry.compute_triangle_violations(asymmetric)

    @unittest.expectedFailure
    def test_triangle_helper_rejects_nonfinite_matrix(self):
        nonfinite = np.asarray([[0.0, 1.0, np.nan], [1.0, 0.0, 1.0], [np.nan, 1.0, 0.0]])
        with self.assertRaises(ValueError):
            geometry.compute_triangle_violations(nonfinite)


if __name__ == "__main__":
    unittest.main()

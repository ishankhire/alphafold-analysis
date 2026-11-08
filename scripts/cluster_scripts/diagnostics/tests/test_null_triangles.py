"""Checks for the null-triangle experiment: metric parity, triangle sets, and nulls."""
from __future__ import annotations

import math
import sys
import tempfile
from pathlib import Path

import numpy as np

DIAGNOSTICS = Path(__file__).resolve().parents[1]
SCRIPT_DIR = DIAGNOSTICS / "null_triangles"
CLUSTER_DIR = DIAGNOSTICS.parent
for path in (SCRIPT_DIR, CLUSTER_DIR, CLUSTER_DIR / "triangle_violation_subgroups"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import null_triangles_common as common  # noqa: E402
import null_triangles_replay as replay  # noqa: E402
from triangle_violation_subgroup_common import (  # noqa: E402
    iter_triangle_indices, triangle_geometry,
)

LOCAL_PDB = CLUSTER_DIR.parents[2] / "protein_data/7b3a_A/predicted_pdb/7b3a_A_model_1_ptm_unrelaxed.pdb"


def random_coordinates(n: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n, 3))
    steps *= 3.8 / np.linalg.norm(steps, axis=1, keepdims=True)
    return np.cumsum(steps, axis=0).astype(np.float32).astype(np.float64)


def packed_true(coordinates: np.ndarray) -> np.ndarray:
    i, j = np.tril_indices(coordinates.shape[0], k=-1)
    return np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)[i, j]


def perturbed(d_true: np.ndarray, seed: int, scale: float = 3.0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    return (d_true + rng.normal(-1.0, scale, d_true.size)).astype(np.float32)


def test_unranking_follows_iter_triangle_indices():
    for n in (3, 4, 9, 23):
        expected = np.concatenate([np.stack(t, axis=1) for t in iter_triangle_indices(n, 7)])
        got = np.stack(common.unrank_triangles(np.arange(expected.shape[0]), n), axis=1)
        np.testing.assert_array_equal(got, expected)


def test_reference_rms_is_bit_identical_to_triangle_geometry():
    coordinates = random_coordinates(40, 1)
    d_true = packed_true(coordinates)
    for tri_i, tri_j, tri_k in iter_triangle_indices(40, 1000):
        expected = triangle_geometry(coordinates, tri_i, tri_j, tri_k)["rms"]
        got = common.reference_rms(d_true, common.side_positions(tri_i, tri_j, tri_k))
        assert np.array_equal(got, expected)


def test_slim_reducer_matches_compute_metric_rows_bitwise():
    coordinates = random_coordinates(60, 2)
    d_true = packed_true(coordinates)
    with tempfile.TemporaryDirectory() as directory:
        for seed in range(3):
            prediction = perturbed(d_true, seed)
            assert np.any(prediction < 0)
            expected = replay.paper_all_triangles(prediction, coordinates, Path(directory))
            exact = common.score_exact_streaming({"p": prediction}, d_true, 60)["p"]
            listed = common.score_set(prediction, common.full_triangle_set(d_true, 60))
            for got in (exact, listed):
                for key in replay.PARITY_KEYS:
                    assert got[key] == expected[key], (key, got[key], expected[key])


def test_slim_reducer_on_real_structure():
    if not LOCAL_PDB.is_file():
        return
    sys.path.insert(0, str(CLUSTER_DIR))
    import cluster_regression_helpers as helpers
    coordinates = np.asarray(helpers.extract_ca_coordinates(LOCAL_PDB)["coordinates"], dtype=np.float64)[:120]
    d_true = packed_true(coordinates)
    prediction = perturbed(d_true, 7, 4.0)
    with tempfile.TemporaryDirectory() as directory:
        expected = replay.paper_all_triangles(prediction, coordinates, Path(directory))
    got = common.score_set(prediction, common.full_triangle_set(d_true, coordinates.shape[0]))
    for key in replay.PARITY_KEYS:
        assert got[key] == expected[key]


def test_sampled_set_is_distinct_and_fixed():
    d_true = packed_true(random_coordinates(200, 3))       # C(200, 3) > 1M
    first = common.sampled_triangle_set(d_true, 200, 5)
    second = common.sampled_triangle_set(d_true, 200, 5)
    assert not first.exact and first.n == common.N_SAMPLE_TRIANGLES
    np.testing.assert_array_equal(first.positions, second.positions)
    assert np.unique(first.positions, axis=0).shape[0] == first.n
    small = common.sampled_triangle_set(packed_true(random_coordinates(50, 4)), 50, 5)
    assert small.exact and small.n == math.comb(50, 3)


def test_bins_are_half_open():
    codes = common.distance_codes(np.array([0.0, 3.999, 4.0, 54.9, 55.0, 1e4]), common.B10_EDGES)
    assert codes.tolist() == [0, 0, 1, 8, 9, 9]
    assert len(common.B20_EDGES) == 21 and len(common.BIN_LABELS["B10xS6"]) == 60


def test_null_b_keeps_bin_histograms_and_null_c_reconstructs():
    rng = np.random.default_rng(0)
    n = 50
    i, j = np.tril_indices(n, k=-1)
    d_true = rng.uniform(3, 40, i.size)
    residual = rng.normal(size=i.size) + 0.3 * (i % 5)
    codes = common.distance_codes(d_true, common.B10_EDGES)
    moments = common.bin_moments(residual, codes, 10)
    values, same = common.make_null("null_b", rng, d_true, residual, codes, i, j, n,
                                    moments["mean"], moments["sd"])
    assert same and not np.allclose(values, d_true + residual)
    for b in range(10):
        np.testing.assert_allclose(np.sort((values - d_true)[codes == b]), np.sort(residual[codes == b]),
                                   rtol=0, atol=1e-12)
    # Null C with the identity permutation gives back the observed residuals.
    mu = moments["mean"]
    centred = residual - mu[codes]
    u = common.residue_offsets(centred, i, j, n)
    rebuilt = mu[codes] + u[i] + u[j] + (centred - u[i] - u[j])
    np.testing.assert_allclose(rebuilt, residual, atol=1e-12)


def test_null_a_fallback_below_threshold():
    n_pairs = np.array([5.0, 29.0, 30.0, 400.0])
    mu, sd = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0, 1.0])
    cohort_mu, cohort_sd = np.full(4, 9.0), np.full(4, 8.0)
    a_mu, a_sd, used = replay.null_a_parameters(n_pairs, mu, sd, cohort_mu, cohort_sd)
    assert used.tolist() == [True, True, False, False]
    assert a_mu.tolist() == [9.0, 9.0, 3.0, 4.0] and a_sd.tolist() == [8.0, 8.0, 1.0, 1.0]

"""Checks for the null-triangle experiment: metric parity, streaming scoring, cohort, and nulls."""
from __future__ import annotations

import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parents[1]
CLUSTER_DIR = SCRIPT_DIR.parent
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


def test_reference_rms_is_bit_identical_to_triangle_geometry():
    coordinates = random_coordinates(40, 1)
    d_true = packed_true(coordinates)
    for tri_i, tri_j, tri_k in iter_triangle_indices(40, 1000):
        expected = triangle_geometry(coordinates, tri_i, tri_j, tri_k)["rms"]
        got = common.reference_rms(d_true, common.side_positions(tri_i, tri_j, tri_k))
        assert np.array_equal(got, expected)


def test_streaming_many_vectors_matches_compute_metric_rows_bitwise():
    coordinates = random_coordinates(60, 2)
    d_true = packed_true(coordinates)
    vectors = {("p", seed): perturbed(d_true, seed) for seed in range(3)}
    vectors[("null", 0)] = d_true + np.random.default_rng(9).normal(0, 2, d_true.size)
    scored = common.score_streaming(vectors, d_true, 60, track_clipped={("p", 0)})
    with tempfile.TemporaryDirectory() as directory:
        for key, values in vectors.items():
            expected = replay.paper_all_triangles(values, coordinates, Path(directory), "ecod")
            for metric in replay.PARITY_KEYS:
                assert scored[key][metric] == expected[metric], (key, metric)
    # Scoring alone gives the same numbers as scoring alongside other vectors.
    alone = common.score_streaming({"x": vectors[("p", 1)]}, d_true, 60)["x"]
    assert alone == {**scored[("p", 1)]}
    assert scored[("p", 0)]["n_violations_with_clipped_side"] > 0
    assert scored[("p", 1)]["n_violations_with_clipped_side"] == 0      # not tracked


def test_streaming_on_real_structure():
    if not LOCAL_PDB.is_file():
        return
    import cluster_regression_helpers as helpers
    coordinates = np.asarray(helpers.extract_ca_coordinates(LOCAL_PDB)["coordinates"], dtype=np.float64)[:120]
    d_true = packed_true(coordinates)
    prediction = perturbed(d_true, 7, 4.0)
    with tempfile.TemporaryDirectory() as directory:
        expected = replay.paper_all_triangles(prediction, coordinates, Path(directory), "cameo")
    got = common.score_streaming({"p": prediction}, d_true, coordinates.shape[0])["p"]
    for key in replay.PARITY_KEYS:
        assert got[key] == expected[key]


def test_run_set_per_probe():
    for probe in common.PROBES:
        runs = common.null_runs(probe)
        assert {m for m, _ in runs} == {"null_a", "null_b"}
        expected = 5 if probe == common.SENSITIVITY_PROBE else 2
        assert len(runs) == expected
    assert set(common.null_runs(common.SENSITIVITY_PROBE)) == {
        ("null_a", "B10"), ("null_b", "B10"), ("null_b", "B5"), ("null_b", "B20"), ("null_b", "B10xS6")}
    assert common.N_DRAWS == 3 and common.MODELS == ("observed", "null_a", "null_b")


def test_cohort_check_counts_sources():
    def rows(n_cameo, n_ecod):
        return ([{"protein_id": f"c{k}", "source": "cameo", "n_residues": "10", "n_pairs": "45"} for k in range(n_cameo)]
                + [{"protein_id": f"e{k}", "source": "ecod", "n_residues": "10", "n_pairs": "45"} for k in range(n_ecod)])
    replay.check_cohort(rows(154, 398))
    for bad in (rows(154, 397), rows(153, 398)):
        try:
            replay.check_cohort(bad)
        except ValueError:
            continue
        raise AssertionError("cohort check accepted a wrong cohort")
    wrong_pairs = rows(154, 398)
    wrong_pairs[0]["n_pairs"] = "44"
    try:
        replay.check_cohort(wrong_pairs)
    except ValueError:
        return
    raise AssertionError("cohort check accepted a wrong pair count")


def test_bins_are_half_open():
    codes = common.distance_codes(np.array([0.0, 3.999, 4.0, 54.9, 55.0, 1e4]), common.B10_EDGES)
    assert codes.tolist() == [0, 0, 1, 8, 9, 9]
    assert len(common.B20_EDGES) == 21 and len(common.BIN_LABELS["B10xS6"]) == 60


def test_null_b_keeps_bin_histograms_on_every_binning():
    rng = np.random.default_rng(0)
    n = 50
    i, j = np.tril_indices(n, k=-1)
    d_true = rng.uniform(3, 70, i.size)
    residual = rng.normal(size=i.size) + 0.3 * (i % 5)
    seq = np.minimum(np.abs(i - j), 60)
    seq_codes = np.searchsorted(np.array([1, 2, 5, 10, 20, 50]), seq, side="right") - 1
    all_codes = common.all_bin_codes(d_true, seq_codes)
    for binning in ("B5", "B10", "B20", "B10xS6"):
        codes = all_codes[binning]
        n_bins = len(common.BIN_LABELS[binning])
        moments = common.bin_moments(residual, codes, n_bins)
        values, same = common.make_null("null_b", rng, d_true, residual, codes, moments["mean"], moments["sd"])
        assert same and not np.allclose(values, d_true + residual)
        for b in range(n_bins):
            np.testing.assert_allclose(np.sort((values - d_true)[codes == b]), np.sort(residual[codes == b]),
                                       rtol=0, atol=1e-12)


def test_null_a_matches_bin_moments_in_expectation():
    rng = np.random.default_rng(1)
    d_true = rng.uniform(3, 40, 200_000)
    codes = common.distance_codes(d_true, common.B10_EDGES)
    mu, sd = np.linspace(-1, 1, 10), np.linspace(0.5, 3, 10)
    values, same = common.make_null("null_a", rng, d_true, np.zeros_like(d_true), codes, mu, sd)
    assert same is None
    moments = common.bin_moments(values - d_true, codes, 10)
    occupied = moments["n"] > 1000
    np.testing.assert_allclose(moments["mean"][occupied], mu[occupied], atol=0.05)
    np.testing.assert_allclose(moments["sd"][occupied], sd[occupied], rtol=0.05)


def test_null_draws_are_reproducible_and_distinct():
    a = common.null_draw_rng(3, common.PROBES[0], 7, "null_b", "B10", 0).random(5)
    b = common.null_draw_rng(3, common.PROBES[0], 7, "null_b", "B10", 0).random(5)
    c = common.null_draw_rng(3, common.PROBES[0], 7, "null_b", "B10", 1).random(5)
    assert np.array_equal(a, b) and not np.array_equal(a, c)


def test_parity_table_covers_cohort():
    import csv
    import gzip
    with gzip.open(replay.PARITY_CSV, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 552 * 48 * len(common.PROBES)
    assert {r["analysis_name"] for r in rows} == set(common.PROBES)
    assert sum(r["source"] == "ecod" for r in rows) == 398 * 48 * len(common.PROBES)


def test_null_a_fallback_below_threshold():
    n_pairs = np.array([5.0, 29.0, 30.0, 400.0])
    mu, sd = np.array([1.0, 2.0, 3.0, 4.0]), np.array([1.0, 1.0, 1.0, 1.0])
    cohort_mu, cohort_sd = np.full(4, 9.0), np.full(4, 8.0)
    a_mu, a_sd, used = replay.null_a_parameters(n_pairs, mu, sd, cohort_mu, cohort_sd)
    assert used.tolist() == [True, True, False, False]
    assert a_mu.tolist() == [9.0, 9.0, 3.0, 4.0] and a_sd.tolist() == [8.0, 8.0, 1.0, 1.0]

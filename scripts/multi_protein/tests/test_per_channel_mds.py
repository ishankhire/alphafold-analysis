"""Checks for the per-channel MDS maths: parity with the canonical classical MDS and the channel pipeline."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
from scipy.stats import spearmanr

PROJECT_ROOT = Path(__file__).resolve().parents[3]
CLUSTER = PROJECT_ROOT / "scripts/cluster_scripts"
sys.path.insert(0, str(CLUSTER))
sys.path.insert(0, str(CLUSTER / "diagnostics/per_channel_mds"))

import cluster_regression_helpers as helpers  # noqa: E402
import per_channel_mds_common as common  # noqa: E402


def structure(n: int = 40, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    coordinates = np.cumsum(rng.normal(size=(n, 3)) * 2.0, axis=0)
    return coordinates, helpers.compute_distance_matrix(coordinates)


def block_from(planes: list[np.ndarray]) -> np.ndarray:
    return np.stack(planes, axis=2).astype(np.float32)


def test_true_distances_embed_exactly_and_min_shift_does_not():
    _, distances = structure()
    rows = {(r["reference"], r["shift"]): r for r in common.reference_rows(distances)}
    raw = rows[("true_distance", "raw")]
    assert raw["stress"] < 1e-10 and raw["negative_eigenvalue_fraction"] < 1e-10
    assert np.isclose(raw["frac_top3_eigenvalues"], 1.0)
    assert np.isclose(raw["embedding_true_distance_pearson"], 1.0)
    assert rows[("true_distance", "minshift")]["stress"] > 1e-3          # the shift itself costs stress
    line = rows[("sequence_separation", "raw")]
    assert line["stress"] < 1e-10 and np.isclose(line["frac_top1_eigenvalue"], 1.0)


def test_batched_mds_matches_the_canonical_helper():
    rng = np.random.default_rng(1)
    n = 30
    upper = np.triu_indices(n, k=1)
    values = rng.random((5, upper[0].size)) * 10.0
    matrices = common.to_matrices(values, upper, n)
    metrics, _ = common.mds_metrics(matrices, upper)
    for k in range(5):
        reference = common.single_matrix_metrics(matrices[k], helpers.classical_mds)
        for name in common.GEOMETRY:
            assert np.isclose(metrics[name][k], reference[name], rtol=0, atol=1e-10), name


def test_channel_pipeline_picks_the_sign_and_reports_correlations():
    _, distances = structure(seed=2)
    n = distances.shape[0]
    rng = np.random.default_rng(3)
    noise = rng.normal(size=(n, n))
    separation = np.abs(np.subtract.outer(np.arange(n), np.arange(n))).astype(float)
    directional = np.triu(np.ones((n, n)), 1) - np.tril(np.ones((n, n)), -1)       # purely antisymmetric
    planes = [3.0 * distances + 7.0,            # distance-like
              -2.0 * distances + 1.0,           # closeness-like: the sign must flip
              separation,                       # sequence separation
              noise,                            # asymmetric noise
              np.full((n, n), 4.0),             # constant
              distances + 5.0 * directional]    # distance plus an antisymmetric part
    metrics = common.channel_metrics(block_from(planes), distances, layer=5, protein_index=11, chunk=4)
    assert list(metrics["status"]) == ["ok", "ok", "ok", "ok", "constant", "ok"]
    assert metrics["best_sign"][0] == 1 and metrics["best_sign"][1] == -1
    assert np.isclose(metrics["stress"][0], metrics["stress"][1], atol=1e-5)       # same matrix after orienting
    assert metrics["stress"][0] < metrics["stress_other_sign"][0]
    assert metrics["stress"][0] < metrics["perm_stress"][0]
    assert np.isclose(metrics["spearman_true_distance"][0], 1.0)
    assert np.isclose(metrics["spearman_true_distance"][1], -1.0)
    assert np.isclose(metrics["spearman_sequence_separation"][2], 1.0)
    upper = np.triu_indices(n, k=1)
    symmetric_noise = ((noise + noise.T) / 2.0).astype(np.float32).astype(np.float64)[upper]
    expected = spearmanr(symmetric_noise, distances[upper]).statistic
    assert np.isclose(metrics["spearman_true_distance"][3], expected, atol=1e-6)
    assert np.isnan(metrics["stress"][4]) and np.isnan(metrics["perm_stress"][4])
    assert metrics["antisymmetric_fraction"][0] < 1e-12 and metrics["antisymmetric_fraction"][5] > 0.1
    assert np.isclose(metrics["stress"][5], metrics["stress"][0], atol=1e-5)       # the antisymmetric part is dropped
    assert metrics["embedding_true_distance_pearson"][0] > 0.99


def test_chunking_does_not_change_the_result_and_parity_matrix_matches():
    rng = np.random.default_rng(4)
    _, distances = structure(n=25, seed=5)
    block = rng.normal(size=(25, 25, 7)).astype(np.float32)
    a = common.channel_metrics(block, distances, layer=0, protein_index=3, chunk=2)
    b = common.channel_metrics(block, distances, layer=0, protein_index=3, chunk=7)
    for name in common.CHANNEL_COLUMNS:
        if name == "status":
            assert list(a[name]) == list(b[name])
        else:
            assert np.allclose(a[name], b[name], rtol=0, atol=1e-12, equal_nan=True), name
    for channel in range(7):
        matrix = common.channel_distance_matrix(block, channel, int(a["best_sign"][channel]))
        reference = common.single_matrix_metrics(matrix, helpers.classical_mds)
        for name in common.GEOMETRY:
            assert np.isclose(a[name][channel], reference[name], rtol=0, atol=1e-9), name


def test_permutation_keeps_each_channels_values_and_depends_on_the_seed():
    rng = np.random.default_rng(6)
    values = rng.normal(size=(3, 200))
    permuted = common.permute_rows(values, layer=1, protein_index=2, first_channel=10)
    assert np.array_equal(np.sort(permuted, axis=1), np.sort(values, axis=1))
    assert not np.array_equal(permuted, values)
    assert np.array_equal(permuted, common.permute_rows(values, 1, 2, 10))
    assert not np.array_equal(permuted, common.permute_rows(values, 2, 2, 10))
    # channel k of a chunk starting at 10 equals channel 0 of a chunk starting at 10 + k
    assert np.array_equal(permuted[2], common.permute_rows(values[2:], 1, 2, 12)[0])

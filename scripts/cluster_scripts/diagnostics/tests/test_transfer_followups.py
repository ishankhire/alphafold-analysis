"""Unit checks for the transfer follow-up replays and launchers."""
from __future__ import annotations

import math
import subprocess
import sys
from pathlib import Path

import numpy as np

DIAGNOSTICS = Path(__file__).resolve().parents[1]
SCRIPT_DIR = DIAGNOSTICS / "transfer_followups"
CLUSTER_DIR = DIAGNOSTICS.parent
for path in (SCRIPT_DIR, CLUSTER_DIR, CLUSTER_DIR / "cross_layer_transfer", CLUSTER_DIR / "cross_validation",
             CLUSTER_DIR / "triangle_violation_subgroups", DIAGNOSTICS / "cross_layer_transfer_all_layers",
             DIAGNOSTICS / "native_error_strata"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import gain_position_replay as gain  # noqa: E402
import triangle_strata_replay as triangle  # noqa: E402


def test_grouped_sums_match_bincount():
    rng = np.random.default_rng(0)
    keys = rng.integers(0, 9, 500)
    values = rng.normal(size=(500, 3))
    occupied, (sums,) = gain.grouped_sums(keys, [values])
    assert occupied.tolist() == sorted(set(keys.tolist()))
    for column in range(3):
        expected = np.bincount(keys, weights=values[:, column], minlength=9)
        np.testing.assert_allclose(sums[:, column], expected[occupied])


def test_pair_position_follows_strict_lower_order():
    high, low = np.tril_indices(12, k=-1)
    np.testing.assert_array_equal(triangle.pair_position(high, low), np.arange(high.size))


def test_small_proteins_use_every_triple():
    triples, exact = triangle.sample_triangles(9, 0)
    assert exact
    assert triples.shape == (math.comb(9, 3), 3)
    assert len({tuple(t) for t in triples.tolist()}) == math.comb(9, 3)
    assert np.all(triples[:, 0] < triples[:, 1]) and np.all(triples[:, 1] < triples[:, 2])


def test_large_proteins_use_a_fixed_sample(monkeypatch):
    monkeypatch.setattr(triangle, "N_TRIANGLES", 1000)
    first, exact = triangle.sample_triangles(60, 3)
    second, _ = triangle.sample_triangles(60, 3)
    assert not exact
    assert first.shape == (1000, 3)
    np.testing.assert_array_equal(first, second)
    assert np.all(first[:, 0] < first[:, 1]) and np.all(first[:, 1] < first[:, 2])


def test_score_finds_violations_and_clips_negative_distances():
    sides = np.array([[0, 1, 2], [3, 4, 5]])
    violating, excess = triangle.score(np.array([3.0, 4.0, 5.0, 1.0, -2.0, 4.0]), sides)
    assert violating.tolist() == [False, True]
    np.testing.assert_allclose(excess, [3.0])


def test_within_cell_permutation_stays_in_its_cell():
    rng = np.random.default_rng(1)
    cell = rng.integers(0, 6, 400)
    permutation = triangle.within_cell_permutation(rng, cell)
    assert sorted(permutation.tolist()) == list(range(400))
    np.testing.assert_array_equal(cell[permutation], cell)
    assert np.any(permutation != np.arange(400))


def test_launchers_follow_the_cluster_rules():
    launchers = sorted(SCRIPT_DIR.glob("*.sbatch"))
    assert len(launchers) == 3
    for launcher in launchers:
        subprocess.run(["bash", "-n", str(launcher)], check=True)
        content = launcher.read_text()
        assert "--mail-type=BEGIN,END,FAIL" in content
        assert "--mail-user" not in content
        assert '--constraint="scratch&work"' in content
        assert "/u/" not in content

import importlib.util
from pathlib import Path

import numpy as np


MODULE = (
    Path(__file__).parents[1]
    / "distance_error_geometry_null"
    / "diagnostic_distance_error_geometry_null.py"
)
spec = importlib.util.spec_from_file_location("distance_error_geometry_null", MODULE)
diagnostic = importlib.util.module_from_spec(spec)
spec.loader.exec_module(diagnostic)


def test_matched_r2_noise_is_exact_nonnegative_and_reproducible():
    true_dist = np.linspace(1.0, 30.0, 300)
    target_r2 = 0.42
    first, first_metrics = diagnostic.perturb_distances_matched_r2(
        true_dist,
        target_r2=target_r2,
        requested_bias=-0.5,
        seed=1234,
    )
    second, second_metrics = diagnostic.perturb_distances_matched_r2(
        true_dist,
        target_r2=target_r2,
        requested_bias=-0.5,
        seed=1234,
    )

    assert np.array_equal(first, second)
    assert first_metrics == second_metrics
    assert np.all(first >= 0.0)
    assert abs(first_metrics["achieved_r2"] - target_r2) < 1e-10


def test_reconstruction_of_euclidean_distances_is_nearly_exact():
    rng = np.random.default_rng(7)
    true_coords = rng.normal(size=(12, 3))
    true_dist = diagnostic.compute_distance_matrix(true_coords)
    geometry = diagnostic.reconstruct_geometry(true_coords, true_dist)

    assert geometry["tm_score"] > 0.999999
    assert geometry["rmsd"] < 1e-6
    assert geometry["stress"] < 1e-12
    assert geometry["frac_top3_eigenvalues"] > 0.999999


def test_empirical_tail_uses_metric_direction():
    null_values = [0.1, 0.2, 0.3]
    assert diagnostic.empirical_favorable_tail(null_values, 0.25, True) == 0.5
    assert diagnostic.empirical_favorable_tail(null_values, 0.25, False) == 0.75

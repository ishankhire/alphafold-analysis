import importlib.util
import math
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[1] / "local_pair_change_se_pilot.py"
SPEC = importlib.util.spec_from_file_location("local_change_se", MODULE)
pilot = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(pilot)


def test_local_pair_indices_are_canonical_and_inclusive():
    i, j, separation = pilot.local_pair_indices(16)
    assert np.all(i < j)
    assert separation.min() == 4
    assert separation.max() == 12
    assert i.size == sum(16 - offset for offset in range(4, 13))
    assert (0, 4) in set(zip(i.tolist(), j.tolist()))


def test_bootstrap_counts_are_deterministic_full_size_cluster_draws():
    first = pilot.build_bootstrap_counts(7, n_bootstraps=10, seed=19)
    second = pilot.build_bootstrap_counts(7, n_bootstraps=10, seed=19)
    np.testing.assert_array_equal(first, second)
    np.testing.assert_array_equal(first.sum(axis=1), np.full(10, 7))
    assert first.shape == (10, 7)


def test_sufficient_statistic_solver_matches_weighted_raw_lstsq():
    rng = np.random.default_rng(4)
    design = rng.normal(size=(300, pilot.FEATURE_DIM))
    true_coef = rng.normal(size=pilot.FEATURE_DIM)
    target = 1.7 + design @ true_coef + rng.normal(scale=0.01, size=300)
    augmented = np.column_stack([np.ones(design.shape[0]), design])
    gram = augmented.T @ augmented
    cross = augmented.T @ target
    expected, _, _, _ = np.linalg.lstsq(augmented, target, rcond=None)
    fitted = pilot.solve_sufficient_statistics(gram, cross)
    actual = np.concatenate([[fitted["intercept"]], fitted["coef"]])
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_sufficient_statistics_reproduce_equal_protein_bootstrap_weights():
    rng = np.random.default_rng(17)
    protein_designs = [
        rng.normal(size=(400, pilot.FEATURE_DIM)),
        rng.normal(size=(650, pilot.FEATURE_DIM)),
    ]
    true_coef = rng.normal(size=pilot.FEATURE_DIM)
    protein_targets = [
        -0.4 + design @ true_coef + rng.normal(scale=0.02, size=design.shape[0])
        for design in protein_designs
    ]
    multiplicities = [2, 1]
    gram = np.zeros((pilot.AUGMENTED_DIM, pilot.AUGMENTED_DIM))
    cross = np.zeros(pilot.AUGMENTED_DIM)
    explicit_design = []
    explicit_target = []
    for design, target, count in zip(
        protein_designs, protein_targets, multiplicities
    ):
        augmented = np.column_stack([np.ones(design.shape[0]), design])
        gram += count * (augmented.T @ augmented) / design.shape[0]
        cross += count * (augmented.T @ target) / design.shape[0]
        row_scale = math.sqrt(count / design.shape[0])
        explicit_design.append(augmented * row_scale)
        explicit_target.append(target * row_scale)
    expected, _, _, _ = np.linalg.lstsq(
        np.vstack(explicit_design), np.concatenate(explicit_target), rcond=None
    )
    fitted = pilot.solve_sufficient_statistics(gram, cross)
    actual = np.concatenate([[fitted["intercept"]], fitted["coef"]])
    np.testing.assert_allclose(actual, expected, rtol=1e-8, atol=1e-8)


def test_pair_change_se_uses_sample_standard_deviation():
    changes = np.asarray([[1.0, 2.0, 3.0], [-1.0, -1.0, -1.0]])
    actual = pilot.pair_change_standard_error(changes)
    np.testing.assert_allclose(actual, [1.0, 0.0])


def test_local_predictions_use_distinct_saved_endpoint_probes():
    rng = np.random.default_rng(23)
    pb_from = rng.normal(size=(9, 9, pilot.N_CHANNELS))
    pb_to = rng.normal(size=(9, 9, pilot.N_CHANNELS))
    bundle_from = {
        "intercept": 4.0,
        "coef": rng.normal(size=pilot.FEATURE_DIM),
    }
    bundle_to = {
        "intercept": -2.5,
        "coef": rng.normal(size=pilot.FEATURE_DIM),
    }
    result = pilot.extract_local_change_inputs(
        pb_from,
        pb_to,
        {"n_residues": 9},
        bundle_from,
        bundle_to,
    )

    expected_from = (
        bundle_from["intercept"] + result["feature_from"] @ bundle_from["coef"]
    )
    expected_to = bundle_to["intercept"] + result["feature_to"] @ bundle_to["coef"]
    np.testing.assert_allclose(result["distance_from"], expected_from)
    np.testing.assert_allclose(result["distance_to"], expected_to)


def test_bootstrap_coefficient_rows_include_both_layers():
    model = {
        "intercept": 1.0,
        "coef": np.zeros(pilot.FEATURE_DIM, dtype=np.float64),
    }
    rows = pilot.coefficient_rows({10: [model], 11: [model]})
    assert {row["layer"] for row in rows} == {10, 11}
    assert len(rows) == 2 * pilot.AUGMENTED_DIM


def test_disjoint_buckets_exhaust_values_and_respect_boundaries():
    values = np.asarray([-0.6, -0.5, -0.2, -0.1, -0.05, 0.05, 0.1, 0.2, 0.5, 0.6])
    rows = pilot.disjoint_bucket_counts(values)
    assert sum(row["count"] for row in rows) == values.size
    counts = {row["bucket"]: row["count"] for row in rows}
    assert counts["lt_m0_50"] == 1
    assert counts["m0_05_to_p0_05"] == 2
    assert counts["gt_p0_50"] == 1


def test_threshold_boundaries_are_no_change():
    labels = pilot.class_labels(np.asarray([-0.11, -0.1, 0.0, 0.1, 0.11]), 0.1)
    np.testing.assert_array_equal(labels, [-1, 0, 0, 0, 1])

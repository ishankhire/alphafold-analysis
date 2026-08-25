from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np


DIAGNOSTICS_DIR = Path(__file__).resolve().parents[1]
if str(DIAGNOSTICS_DIR) not in sys.path:
    sys.path.insert(0, str(DIAGNOSTICS_DIR))

from aggregate_triangle_violation_subgroups import (  # noqa: E402
    descriptive,
    length_effects,
    pooled_value,
)
from triangle_violation_subgroup_common import (  # noqa: E402
    CATEGORY_LABELS,
    DERIVED_METRICS,
    EVALUATION_SCOPE,
    LENGTH_EFFECT_COLUMNS,
    PER_PROTEIN_COLUMNS,
    SUMMARY_COLUMNS,
    TRIANGLE_TOLERANCE_ANGSTROM,
    build_reference_cache,
    category_codes,
    compute_metric_rows,
    iter_triangle_indices,
    tertile_codes,
    triangle_count,
    triangle_geometry,
)


def _identity(n_residues: int) -> dict[str, object]:
    return {
        "analysis_name": "synthetic",
        "model_family": "shared_ols",
        "layer": 0,
        "protein_id": "toy",
        "source": "ecod",
        "protein_split": "test",
        "generalization_scope": "unseen_proteins",
        "n_residues": n_residues,
        "protein_length_bin": "length_bin_1",
    }


def _geometry_for_three(points: np.ndarray) -> dict[str, np.ndarray]:
    return triangle_geometry(
        points,
        np.asarray([2]),
        np.asarray([1]),
        np.asarray([0]),
    )


def test_triangle_enumeration_is_exact_and_batched() -> None:
    triples = []
    for i, j, k in iter_triangle_indices(8, batch_size=3):
        triples.extend(zip(i.tolist(), j.tolist(), k.tolist()))
    assert len(triples) == triangle_count(8) == 56
    assert len(set(triples)) == 56
    assert all(k < j < i for i, j, k in triples)


def test_acute_near_right_obtuse_and_degenerate_geometry() -> None:
    equilateral = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.5, math.sqrt(3) / 2, 0.0]])
    right = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    obtuse = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [-0.5, 0.1, 0.0]])
    degenerate = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
    thresholds = {
        "rms_tertile_1": 0.5, "rms_tertile_2": 2.0,
        "area_tertile_1": 0.1, "area_tertile_2": 0.6,
        "slack_tertile_1": 0.1, "slack_tertile_2": 0.5,
    }
    assert category_codes(_geometry_for_three(equilateral), thresholds)["angle_shape"][0] == 0
    right_codes = category_codes(_geometry_for_three(right), thresholds)
    assert right_codes["angle_shape"][0] == 1
    assert right_codes["angle_shape_exact90"][0] == 1
    assert category_codes(_geometry_for_three(obtuse), thresholds)["angle_shape"][0] == 2
    degenerate_geometry = _geometry_for_three(degenerate)
    assert degenerate_geometry["area"][0] == 0.0
    assert degenerate_geometry["slack"][0] == 0.0


def test_half_open_tertile_boundaries() -> None:
    values = np.asarray([0.9, 1.0, 1.9, 2.0, 2.1])
    assert tertile_codes(values, 1.0, 2.0).tolist() == [0, 1, 1, 2, 2]


def test_angle_and_exact90_boundaries() -> None:
    geometry = {
        "rms": np.ones(4), "area": np.ones(4), "slack": np.full(4, 0.5),
        "angle": np.asarray([87.999, 88.0, 92.0, 92.001]),
        "angle_sign": np.asarray([-2e-12, -1e-12, 1e-12, 2e-12]),
        "angle_sign_tolerance": np.full(4, 1e-12),
    }
    thresholds = {
        "rms_tertile_1": 0.5, "rms_tertile_2": 2.0,
        "area_tertile_1": 0.5, "area_tertile_2": 2.0,
        "slack_tertile_1": 0.2, "slack_tertile_2": 0.8,
    }
    codes = category_codes(geometry, thresholds)
    assert codes["angle_shape"].tolist() == [0, 1, 1, 2]
    assert codes["angle_shape_exact90"].tolist() == [0, 1, 1, 2]


def test_reference_cache_partitions_every_category_system(tmp_path: Path) -> None:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.1, 0.0],
            [0.0, 0.3, 1.3], [1.2, 1.0, 0.7], [0.4, 1.7, 1.4],
        ],
        dtype=np.float64,
    )
    cache, stats = build_reference_cache(coordinates, tmp_path)
    try:
        assert cache.n_triangles == triangle_count(6) == 20
        for system, labels in CATEGORY_LABELS.items():
            assert set(np.unique(cache.codes[system])).issubset(set(range(len(labels))))
            assert sum(np.count_nonzero(cache.codes[system] == code) for code in range(len(labels))) == 20
        assert sum(stats[key] for key in ("n_rms_small", "n_rms_medium", "n_rms_large")) == 20
        assert sum(stats[key] for key in ("n_clearly_acute", "n_near_right", "n_clearly_obtuse")) == 20
        assert sum(stats[key] for key in ("n_exact90_acute", "n_exact90_right_within_tolerance", "n_exact90_obtuse")) == 20
    finally:
        cache.cleanup()


def test_violation_primitives_bands_negative_clipping_and_partitions(tmp_path: Path) -> None:
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 2.0]],
        dtype=np.float64,
    )
    cache, _ = build_reference_cache(coordinates, tmp_path)
    predicted = np.full((4, 4), 2.0, dtype=np.float64)
    np.fill_diagonal(predicted, 0.0)
    predicted[2, 0] = predicted[0, 2] = 10.0
    predicted[3, 1] = predicted[1, 3] = -1.0
    try:
        rows = compute_metric_rows(predicted, cache, _identity(4), batch_size=1)
    finally:
        cache.cleanup()
    assert all(set(row) == set(PER_PROTEIN_COLUMNS) for row in rows)
    assert all(row["evaluation_scope"] == EVALUATION_SCOPE for row in rows)
    all_row = next(row for row in rows if row["category_system"] == "all_triangles")
    assert all_row["n_triangles"] == 4
    assert all_row["n_violations"] == 2
    assert all_row["n_negative_predictions"] == 1
    assert all_row["n_abs_5_10"] == 2
    assert all_row["n_abs_0_5"] == all_row["n_abs_10_plus"] == 0
    assert math.isclose(
        all_row["mean_violation_all"],
        all_row["violation_fraction"] * all_row["mean_violation_conditional"],
    )
    for system, labels in CATEGORY_LABELS.items():
        system_rows = [row for row in rows if row["category_system"] == system]
        assert {row["category_label"] for row in system_rows} == set(labels)
        assert sum(row["n_triangles"] for row in system_rows) == 4


def test_violation_tolerance_boundary(tmp_path: Path) -> None:
    coordinates = np.asarray([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 1.0, 0.0]])
    cache, _ = build_reference_cache(coordinates, tmp_path)
    predicted = np.asarray(
        [
            [0.0, 1.0, 2.0 + TRIANGLE_TOLERANCE_ANGSTROM / 2.0],
            [1.0, 0.0, 1.0],
            [2.0 + TRIANGLE_TOLERANCE_ANGSTROM / 2.0, 1.0, 0.0],
        ]
    )
    try:
        rows = compute_metric_rows(predicted, cache, _identity(3))
    finally:
        cache.cleanup()
    all_row = next(row for row in rows if row["category_system"] == "all_triangles")
    assert all_row["n_violations"] == 0
    assert all_row["support_status"] == "zero_violations"


def test_deterministic_descriptives_pooled_values_and_no_bootstrap_schema() -> None:
    assert not any("bootstrap" in column or "confidence" in column for column in SUMMARY_COLUMNS)
    assert not any("bootstrap" in column or "confidence" in column for column in LENGTH_EFFECT_COLUMNS)
    stats = descriptive([1.0, 2.0, 3.0, float("nan")])
    assert stats["n_proteins_total"] == 4
    assert stats["n_proteins_finite"] == 3
    assert stats["protein_equal_mean"] == 2.0
    rows = [
        {
            "n_triangles": 10, "n_violations": 2, "violation_sum_angstrom": 5.0,
            "relative_violation_sum": 0.5, "n_abs_0_5": 1, "n_abs_5_10": 1,
            "n_abs_10_plus": 0, "n_rel_0_0p05": 1, "n_rel_0p05_0p10": 1,
            "n_rel_0p10_plus": 0, "relative_violation_max": 0.4,
        },
        {
            "n_triangles": 30, "n_violations": 3, "violation_sum_angstrom": 9.0,
            "relative_violation_sum": 0.9, "n_abs_0_5": 2, "n_abs_5_10": 1,
            "n_abs_10_plus": 0, "n_rel_0_0p05": 2, "n_rel_0p05_0p10": 1,
            "n_rel_0p10_plus": 0, "relative_violation_max": 0.3,
        },
    ]
    assert pooled_value(rows, "violation_fraction") == (0.125, "complete")
    assert pooled_value(rows, "mean_violation_all") == (0.35, "complete")
    assert pooled_value(rows, "relative_violation_p99_conditional")[1].startswith("not_reconstructible")


def test_length_effects_retain_proteins_without_bootstrap() -> None:
    base = {
        "analysis_name": "synthetic", "model_family": "shared_ols", "layer": 0,
        "evaluation_scope": EVALUATION_SCOPE, "category_system": "all_triangles",
        "category_label": "all", "n_triangles": 10, "n_violations": 1,
        "violation_sum_angstrom": 1.0, "relative_violation_sum": 0.1,
        "n_abs_0_5": 1, "n_abs_5_10": 0, "n_abs_10_plus": 0,
        "n_rel_0_0p05": 1, "n_rel_0p05_0p10": 0, "n_rel_0p10_plus": 0,
        "relative_violation_max": 0.1, "relative_violation_p99_conditional": 0.1,
        "violation_fraction": 0.1, "mean_violation_all": 0.1,
        "mean_violation_conditional": 1.0, "mean_relative_violation_all": 0.01,
        "mean_relative_violation_conditional": 0.1,
    }
    rows = [{**base, "protein_id": str(index), "n_residues": length} for index, length in enumerate((50, 80, 130))]
    effects = length_effects(rows)
    assert len(effects) == len(DERIVED_METRICS)
    assert all(set(row) == set(LENGTH_EFFECT_COLUMNS) for row in effects)
    assert all(row["n_proteins"] == 3 for row in effects)

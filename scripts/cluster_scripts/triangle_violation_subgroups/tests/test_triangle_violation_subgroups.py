from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
CLUSTER_DIR = SCRIPT_DIR.parent
CV_DIR = CLUSTER_DIR / "cross_validation"
for path in (SCRIPT_DIR, CLUSTER_DIR, CV_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_triangle_violation_subgroups import (  # noqa: E402
    descriptive,
    length_effects,
    pooled_value,
    validate_complete_layer,
)
from regression_protocol import (  # noqa: E402
    SEQUENCE_BIN_LABELS,
    pair_fold_assignment_digest,
    sequence_separation_bin_codes,
    within_protein_pair_folds,
)
from triangle_violation_subgroup_replay import model_predictions  # noqa: E402
from triangle_violation_subgroup_common import (  # noqa: E402
    CATEGORY_LABELS,
    CV_PROTOCOL_VERSION,
    DERIVED_METRICS,
    EDGE_MODEL_COUNT_LABELS,
    EVALUATION_SCOPE,
    FAMILY_SPECS,
    LENGTH_EFFECT_COLUMNS,
    PER_PROTEIN_COLUMNS,
    REPLAY_INDEX_COLUMNS,
    SUMMARY_COLUMNS,
    TRIANGLE_TOLERANCE_ANGSTROM,
    assemble_oof_predictions,
    build_reference_cache,
    category_codes,
    compute_metric_rows,
    edge_model_count_codes,
    edge_model_count_reference_balance,
    index_replay_rows,
    iter_triangle_indices,
    load_and_validate_replay_index,
    sha256_file,
    symmetric_pair_matrix,
    tertile_codes,
    triangle_count,
    triangle_geometry,
    FamilySpec,
)


def _identity(n_residues: int, *, family: str = "shared_ols") -> dict[str, object]:
    return {
        "analysis_name": "synthetic",
        "model_family": family,
        "feature_dim": 256,
        "layer": 0,
        "protein_id": "toy",
        "source": "ecod",
        "cv_protocol_version": CV_PROTOCOL_VERSION,
        "evaluation_design": "protein_outer_5fold",
        "outer_fold": 2,
        "oof_prediction_scope": "heldout_proteins_all_pairs",
        "generalization_scope": "unseen_protein_ids",
        "n_residues": n_residues,
        "protein_length_bin": "length_bin_1",
    }


def _geometry_for_three(points: np.ndarray) -> dict[str, np.ndarray]:
    return triangle_geometry(points, np.asarray([2]), np.asarray([1]), np.asarray([0]))


def _coordinates() -> np.ndarray:
    return np.asarray([
        [0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.2, 1.1, 0.0],
        [0.0, 0.3, 1.3], [1.2, 1.0, 0.7], [0.4, 1.7, 1.4],
    ], dtype=np.float64)


def test_triangle_enumeration_is_exact_and_batched() -> None:
    triples = []
    for i, j, k in iter_triangle_indices(8, batch_size=3):
        triples.extend(zip(i.tolist(), j.tolist(), k.tolist()))
    assert len(triples) == triangle_count(8) == 56
    assert len(set(triples)) == 56
    assert all(k < j < i for i, j, k in triples)


def test_acute_near_right_obtuse_and_degenerate_geometry() -> None:
    equilateral = np.asarray([[0, 0, 0], [1, 0, 0], [0.5, math.sqrt(3) / 2, 0]])
    right = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]])
    obtuse = np.asarray([[0, 0, 0], [1, 0, 0], [-0.5, 0.1, 0]])
    degenerate = np.asarray([[0, 0, 0], [1, 0, 0], [2, 0, 0]])
    thresholds = {
        "rms_tertile_1": 0.5, "rms_tertile_2": 2.0,
        "area_tertile_1": 0.1, "area_tertile_2": 0.6,
        "slack_tertile_1": 0.1, "slack_tertile_2": 0.5,
    }
    assert category_codes(_geometry_for_three(equilateral), thresholds)["angle_shape"][0] == 0
    right_codes = category_codes(_geometry_for_three(right), thresholds)
    assert right_codes["angle_shape"][0] == right_codes["angle_shape_exact90"][0] == 1
    assert category_codes(_geometry_for_three(obtuse), thresholds)["angle_shape"][0] == 2
    assert _geometry_for_three(degenerate)["area"][0] == 0.0
    assert _geometry_for_three(degenerate)["slack"][0] == 0.0


def test_half_open_tertile_and_angle_boundaries() -> None:
    assert tertile_codes(np.asarray([0.9, 1.0, 1.9, 2.0, 2.1]), 1.0, 2.0).tolist() == [0, 1, 1, 2, 2]
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


def test_reference_cache_partitions_and_digest_is_batch_invariant(tmp_path: Path) -> None:
    first, first_stats = build_reference_cache(_coordinates(), tmp_path / "first", batch_size=1)
    second, second_stats = build_reference_cache(_coordinates(), tmp_path / "second", batch_size=100)
    try:
        assert first.n_triangles == second.n_triangles == 20
        assert first_stats["reference_digest"] == second_stats["reference_digest"]
        for system, labels in CATEGORY_LABELS.items():
            first_codes = np.asarray(first.codes[system]).copy()
            second_codes = np.asarray(second.codes[system]).copy()
            assert np.array_equal(first_codes, second_codes)
            assert sum(np.count_nonzero(first_codes == code) for code in range(len(labels))) == 20
        assert sum(first_stats[key] for key in ("n_rms_small", "n_rms_medium", "n_rms_large")) == 20
        assert sum(first_stats[key] for key in ("n_clearly_acute", "n_near_right", "n_clearly_obtuse")) == 20
    finally:
        first.cleanup()
        second.cleanup()


def test_edge_model_count_codes_cover_one_two_and_three_models() -> None:
    pair_i, pair_j = np.tril_indices(4, k=-1)
    # (2,1,0) uses one model; (3,1,0)/(3,2,0) use two; (3,2,1) uses three.
    assignments_by_edge = {(1, 0): 0, (2, 0): 0, (2, 1): 0, (3, 0): 0, (3, 1): 1, (3, 2): 2}
    assignments = np.asarray([assignments_by_edge[(int(i), int(j))] for i, j in zip(pair_i, pair_j)])
    matrix = symmetric_pair_matrix(4, pair_i, pair_j, assignments, diagonal=-1)
    observed = []
    for tri_i, tri_j, tri_k in iter_triangle_indices(4, batch_size=1):
        observed.extend(edge_model_count_codes(matrix, tri_i, tri_j, tri_k).tolist())
    assert observed == [0, 1, 1, 2]
    assert np.bincount(observed, minlength=3).tolist() == [1, 2, 1]


def test_edge_model_reference_balance_partitions_without_crossing_systems(tmp_path: Path) -> None:
    coordinates = _coordinates()[:4]
    cache, _ = build_reference_cache(coordinates, tmp_path)
    pair_i, pair_j = np.tril_indices(4, k=-1)
    assignments = np.asarray([0, 0, 0, 0, 1, 2], dtype=np.int16)
    matrix = symmetric_pair_matrix(4, pair_i, pair_j, assignments, diagonal=-1)
    try:
        rows = edge_model_count_reference_balance(cache, matrix, {
            "protein_id": "toy", "source": "ecod", "n_residues": 4,
            "protein_length_bin": "length_bin_1",
        })
    finally:
        cache.cleanup()
    assert [row["category_label"] for row in rows] == list(EDGE_MODEL_COUNT_LABELS)
    assert sum(row["n_triangles"] for row in rows) == 4
    assert math.isclose(sum(row["triangle_fraction"] for row in rows), 1.0)


def test_oof_assembly_writes_each_pair_once_and_uses_source_identity() -> None:
    features = np.arange(20, dtype=np.float64).reshape(10, 2)
    assignments = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4], dtype=np.int16)
    models = {fold: (np.asarray([1.0, 0.0]), float(100 * fold)) for fold in range(5)}
    predictions = assemble_oof_predictions(features, assignments, models)
    expected = features[:, 0] + 100 * assignments
    assert np.array_equal(predictions, expected)
    with pytest.raises(ValueError, match="Exactly five"):
        assemble_oof_predictions(features, assignments, {key: value for key, value in models.items() if key != 4})
    with pytest.raises(ValueError, match="empty pair fold"):
        assemble_oof_predictions(features, np.zeros(10, dtype=np.int16), models)


def test_shared_pair_fold_xgboost_and_sequence_gate_routing() -> None:
    class FakeBooster:
        def inplace_predict(self, features: np.ndarray) -> np.ndarray:
            return np.full(features.shape[0], 7.0)

    n_residues = 4
    pair_i, pair_j = np.tril_indices(n_residues, k=-1)
    features = np.zeros((pair_i.size, 256), dtype=np.float64)
    assignments = within_protein_pair_folds(n_residues, "toy")
    fold_counts = tuple(int(value) for value in np.bincount(assignments, minlength=5))
    outer = SimpleNamespace(protein_id="toy", n_residues=n_residues, outer_fold=2)
    fold_bundle = SimpleNamespace(pair_by_protein={
        "toy": SimpleNamespace(
            assignment_digest=pair_fold_assignment_digest(n_residues, "toy"),
            fold_counts=fold_counts,
        )
    })
    zero = np.zeros(256, dtype=np.float64)
    one = np.ones(256, dtype=np.float64)
    shared = {
        fold: {
            "coefficient_standardized": zero, "intercept_standardized": float(fold),
            "feature_mean": zero, "feature_scale": one,
        }
        for fold in range(5)
    }
    gated = {
        (2, gate): {
            "coefficient_standardized": zero, "intercept_standardized": float(code + 10),
            "feature_mean": zero, "feature_scale": one,
        }
        for code, gate in enumerate(SEQUENCE_BIN_LABELS)
    }
    shard = {
        "coefficient_standardized": np.zeros((5, 256), dtype=np.float64),
        "intercept_standardized": np.arange(5, dtype=np.float64),
        "feature_mean": np.zeros((5, 256), dtype=np.float64),
        "feature_scale": np.ones((5, 256), dtype=np.float64),
    }
    artifacts = {
        "shared_ols_base256": shared,
        "per_protein_ols_base256": (shard, {("toy", fold): fold for fold in range(5)}),
        "shared_xgboost_base256_fixed_s4": {2: FakeBooster()},
        "sequence_gated_ols_base256": gated,
    }
    routed, observed_assignments = model_predictions(
        outer, ("A",) * n_residues, pair_i, pair_j, features, artifacts, fold_bundle,
    )
    values = {spec.analysis_name: prediction for spec, prediction in routed}
    assert np.array_equal(observed_assignments, assignments)
    assert np.all(values["shared_ols_base256"] == 2.0)
    assert np.array_equal(values["per_protein_ols_base256"], assignments.astype(np.float64))
    assert np.all(values["shared_xgboost_base256_fixed_s4"] == 7.0)
    sequence_codes = sequence_separation_bin_codes(pair_i, pair_j, chain_ids=("A",) * n_residues)
    assert np.array_equal(values["sequence_gated_ols_base256"], sequence_codes + 10.0)


def test_violation_primitives_bands_negative_clipping_and_edge_partition(tmp_path: Path) -> None:
    coordinates = np.asarray([[0, 0, 0], [2, 0, 0], [0, 2, 0], [0, 0, 2]], dtype=np.float64)
    cache, _ = build_reference_cache(coordinates, tmp_path)
    predicted = np.full((4, 4), 2.0, dtype=np.float64)
    np.fill_diagonal(predicted, 0.0)
    predicted[2, 0] = predicted[0, 2] = 10.0
    predicted[3, 1] = predicted[1, 3] = -1.0
    pair_i, pair_j = np.tril_indices(4, k=-1)
    fold_matrix = symmetric_pair_matrix(
        4, pair_i, pair_j, np.asarray([0, 0, 0, 0, 1, 2]), diagonal=-1,
    )
    try:
        rows = compute_metric_rows(
            predicted, cache, _identity(4, family="per_protein_ols"),
            pair_fold_matrix=fold_matrix, batch_size=1,
        )
    finally:
        cache.cleanup()
    assert len(rows) == 28
    assert all(set(row) == set(PER_PROTEIN_COLUMNS) for row in rows)
    all_row = next(row for row in rows if row["category_system"] == "all_triangles")
    assert all_row["n_triangles"] == 4
    assert all_row["n_violations"] == 2
    assert all_row["n_negative_predictions"] == 1
    assert all_row["n_abs_5_10"] == 2
    edge_rows = [row for row in rows if row["category_system"] == "edge_model_count"]
    assert sum(row["n_triangles"] for row in edge_rows) == 4


def test_violation_tolerance_and_zero_violation_status(tmp_path: Path) -> None:
    coordinates = np.asarray([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=np.float64)
    cache, _ = build_reference_cache(coordinates, tmp_path)
    predicted = np.asarray([
        [0.0, 1.0, 2.0 + TRIANGLE_TOLERANCE_ANGSTROM / 2.0],
        [1.0, 0.0, 1.0], [2.0 + TRIANGLE_TOLERANCE_ANGSTROM / 2.0, 1.0, 0.0],
    ])
    try:
        rows = compute_metric_rows(predicted, cache, _identity(3))
    finally:
        cache.cleanup()
    all_row = next(row for row in rows if row["category_system"] == "all_triangles")
    assert all_row["n_violations"] == 0
    assert all_row["support_status"] == "zero_violations"
    assert math.isnan(all_row["mean_violation_conditional"])


def test_layer_validator_accepts_complete_groups_and_rejects_partial_layer(tmp_path: Path) -> None:
    coordinates = _coordinates()[:4]
    cache, _ = build_reference_cache(coordinates, tmp_path)
    distances = np.linalg.norm(coordinates[:, None, :] - coordinates[None, :, :], axis=2)
    pair_i, pair_j = np.tril_indices(4, k=-1)
    fold_matrix = symmetric_pair_matrix(
        4, pair_i, pair_j, np.asarray([0, 0, 0, 0, 1, 2]), diagonal=-1,
    )
    rows = []
    try:
        for spec in FAMILY_SPECS:
            identity = {
                **_identity(4, family=spec.model_family),
                "analysis_name": spec.analysis_name,
                "evaluation_design": spec.evaluation_design,
                "oof_prediction_scope": spec.oof_prediction_scope,
                "generalization_scope": spec.generalization_scope,
            }
            rows.extend(compute_metric_rows(
                distances, cache, identity,
                pair_fold_matrix=fold_matrix if spec.uses_pair_folds else None,
            ))
    finally:
        cache.cleanup()
    assert len(rows) == 103
    validate_complete_layer(rows, 0, expected_proteins=1)
    with pytest.raises(ValueError, match="expected 103"):
        validate_complete_layer(rows[:-1], 0, expected_proteins=1)


def _write_replay_index(path: Path, row: dict[str, str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=REPLAY_INDEX_COLUMNS)
        writer.writeheader()
        writer.writerow(row)


def test_replay_index_is_keyed_and_rejects_digest_status_or_configuration(tmp_path: Path) -> None:
    artifact = tmp_path / "model_artifacts/model.npz"
    artifact.parent.mkdir(parents=True)
    artifact.write_bytes(b"synthetic model")
    spec = FamilySpec(
        "synthetic_family", "shared_ols", "protein_outer_5fold",
        "heldout_proteins_all_pairs", "unseen_protein_ids", "synthetic_family",
    )
    manifest = {"configuration_digest": "config", "expected_replay_artifacts": 1}
    row = dict.fromkeys(REPLAY_INDEX_COLUMNS, "")
    row.update({
        "replay_index_version": "regression_5fold_replay_index_v1",
        "analysis_name": spec.analysis_name, "model_family": spec.model_family,
        "layer": "0", "outer_fold": "2", "artifact_type": "ols",
        "artifact_path": "model_artifacts/model.npz", "artifact_sha256": sha256_file(artifact),
        "configuration_digest": "config", "scaler_digest": "scaler", "status": "complete",
    })
    index_path = tmp_path / "model_artifacts/replay_index.csv"
    _write_replay_index(index_path, row)
    rows = load_and_validate_replay_index(tmp_path, spec, manifest, verify_coverage=False)
    assert index_replay_rows(rows, ("layer", "outer_fold"))[("0", "2")]["scaler_digest"] == "scaler"
    with pytest.raises(ValueError, match="responsibilities are incomplete"):
        load_and_validate_replay_index(tmp_path, spec, manifest)
    for column, wrong, match in (
        ("artifact_sha256", "wrong", "digest mismatch"),
        ("status", "incomplete", "identity/status"),
        ("configuration_digest", "wrong", "identity/status"),
    ):
        changed = {**row, column: wrong}
        _write_replay_index(index_path, changed)
        with pytest.raises(ValueError, match=match):
            load_and_validate_replay_index(tmp_path, spec, manifest, verify_coverage=False)


def test_deterministic_descriptives_and_pooled_primitives() -> None:
    assert not any("bootstrap" in column or "confidence" in column for column in SUMMARY_COLUMNS)
    assert not any("bootstrap" in column or "confidence" in column for column in LENGTH_EFFECT_COLUMNS)
    stats = descriptive([1.0, 2.0, 3.0, float("nan")])
    assert stats["n_proteins_total"] == 4
    assert stats["n_proteins_finite"] == 3
    assert stats["protein_equal_mean"] == 2.0
    rows = [
        {"n_triangles": 10, "n_violations": 2, "violation_sum_angstrom": 5.0,
         "relative_violation_sum": 0.5, "n_abs_0_5": 1, "n_abs_5_10": 1,
         "n_abs_10_plus": 0, "n_rel_0_0p05": 1, "n_rel_0p05_0p10": 1,
         "n_rel_0p10_plus": 0, "relative_violation_max": 0.4},
        {"n_triangles": 30, "n_violations": 3, "violation_sum_angstrom": 9.0,
         "relative_violation_sum": 0.9, "n_abs_0_5": 2, "n_abs_5_10": 1,
         "n_abs_10_plus": 0, "n_rel_0_0p05": 2, "n_rel_0p05_0p10": 1,
         "n_rel_0p10_plus": 0, "relative_violation_max": 0.3},
    ]
    assert pooled_value(rows, "violation_fraction") == (0.125, "complete")
    assert pooled_value(rows, "mean_violation_all") == (0.35, "complete")
    assert pooled_value(rows, "relative_violation_p99_conditional")[1].startswith("not_reconstructible")


def test_length_effects_use_proteins_and_exclude_edge_diagnostic() -> None:
    base = {
        "analysis_name": "synthetic", "model_family": "shared_ols", "feature_dim": 256,
        "layer": 0, "cv_protocol_version": CV_PROTOCOL_VERSION,
        "evaluation_design": "protein_outer_5fold", "evaluation_scope": EVALUATION_SCOPE,
        "oof_prediction_scope": "heldout_proteins_all_pairs",
        "category_system": "all_triangles", "category_label": "all",
        "n_triangles": 10, "n_violations": 1, "violation_sum_angstrom": 1.0,
        "relative_violation_sum": 0.1, "n_abs_0_5": 1, "n_abs_5_10": 0,
        "n_abs_10_plus": 0, "n_rel_0_0p05": 1, "n_rel_0p05_0p10": 0,
        "n_rel_0p10_plus": 0, "relative_violation_max": 0.1,
        "relative_violation_p99_conditional": 0.1, "violation_fraction": 0.1,
        "mean_violation_all": 0.1, "mean_violation_conditional": 1.0,
        "mean_relative_violation_all": 0.01, "mean_relative_violation_conditional": 0.1,
    }
    rows = [{**base, "protein_id": str(index), "n_residues": length} for index, length in enumerate((50, 80, 130))]
    rows.append({**rows[0], "category_system": "edge_model_count", "category_label": "1_model"})
    effects = length_effects(rows)
    assert len(effects) == len(DERIVED_METRICS)
    assert all(set(row) == set(LENGTH_EFFECT_COLUMNS) for row in effects)
    assert all(row["n_proteins"] == 3 for row in effects)

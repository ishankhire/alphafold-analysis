#!/usr/bin/env python3
"""Protocol-2.0 helpers for exact OOF triangle-violation subgroup replay.

This module contains contracts, geometry reducers, and strict replay helpers.
It never fits or selects a model and never saves pair- or triangle-level rows.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np


ANALYSIS_NAME = "triangle_violation_subgroups_5fold_v1"
OUTPUT_SCHEMA_VERSION = "2.0"
CV_PROTOCOL_VERSION = "2.0"
EVALUATION_SCOPE = "oof_all_pairs"
FEATURE_DIM = 256
LAYERS = tuple(range(48))
N_OUTER_FOLDS = 5
N_PAIR_FOLDS = 5
EXPECTED_PROTEINS = 552
SEQUENCE_GATE_LABELS = ("1", "2-4", "5-9", "10-19", "20-49", "50+")
TRIANGLE_TOLERANCE_ANGSTROM = 1e-12
ANGLE_SIGN_RELATIVE_TOLERANCE = 1e-12
ANGLE_ACUTE_MAX_DEGREES = 88.0
ANGLE_OBTUSE_MIN_DEGREES = 92.0
DEFAULT_TRIANGLE_BATCH_SIZE = 250_000
REPLAY_INDEX_COLUMNS = (
    "replay_index_version", "analysis_name", "model_family", "layer", "outer_fold",
    "protein_id", "pair_fold", "sequence_gate", "artifact_type", "artifact_path",
    "artifact_sha256", "configuration_digest", "scaler_digest", "status",
)

CATEGORY_LABELS = {
    "all_triangles": ("all",),
    "rms_size": ("small", "medium", "large"),
    "angle_shape": ("clearly_acute", "near_right", "clearly_obtuse"),
    "angle_shape_exact90": ("acute", "right_within_tolerance", "obtuse"),
    "rms_by_angle": tuple(
        f"{size}__{angle}"
        for size in ("small", "medium", "large")
        for angle in ("clearly_acute", "near_right", "clearly_obtuse")
    ),
    "relative_slack": ("low", "medium", "high"),
    "area": ("low", "medium", "high"),
}
EDGE_MODEL_COUNT_LABELS = ("1_model", "2_models", "3_models")
ALL_CATEGORY_LABELS = {**CATEGORY_LABELS, "edge_model_count": EDGE_MODEL_COUNT_LABELS}


@dataclass(frozen=True)
class FamilySpec:
    analysis_name: str
    model_family: str
    evaluation_design: str
    oof_prediction_scope: str
    generalization_scope: str
    family_dir: str
    uses_pair_folds: bool = False
    uses_sequence_gates: bool = False


FAMILY_SPECS = (
    FamilySpec(
        "shared_ols_base256", "shared_ols", "protein_outer_5fold",
        "heldout_proteins_all_pairs", "unseen_protein_ids", "shared_ols_base256",
    ),
    FamilySpec(
        "per_protein_ols_base256", "per_protein_ols", "within_protein_pair_5fold",
        "heldout_pair_folds_all_pairs", "unseen_pairs_within_seen_protein",
        "per_protein_ols_base256", uses_pair_folds=True,
    ),
    FamilySpec(
        "shared_xgboost_base256_fixed_s4", "shared_xgboost", "protein_outer_5fold",
        "heldout_proteins_all_pairs", "unseen_protein_ids",
        "shared_xgboost_base256_fixed_s4",
    ),
    FamilySpec(
        "sequence_gated_ols_base256", "gated_ols", "protein_outer_5fold",
        "heldout_proteins_all_pairs", "unseen_protein_ids",
        "sequence_gated_ols_base256", uses_sequence_gates=True,
    ),
)
MODEL_IDENTITIES = tuple(
    (spec.analysis_name, spec.model_family, spec.generalization_scope)
    for spec in FAMILY_SPECS
)
FAMILY_BY_NAME = {spec.analysis_name: spec for spec in FAMILY_SPECS}

PER_PROTEIN_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_id", "source",
    "cv_protocol_version", "evaluation_design", "outer_fold", "evaluation_scope",
    "oof_prediction_scope", "generalization_scope", "category_system", "category_label",
    "n_residues", "protein_length_bin", "n_triangles", "n_violations",
    "n_negative_predictions", "negative_prediction_fraction",
    "violation_sum_angstrom", "violation_max_angstrom", "n_abs_0_5", "n_abs_5_10",
    "n_abs_10_plus", "relative_violation_sum", "relative_violation_max",
    "n_rel_0_0p05", "n_rel_0p05_0p10", "n_rel_0p10_plus",
    "relative_violation_p50_conditional", "relative_violation_p90_conditional",
    "relative_violation_p99_conditional", "violation_fraction", "mean_violation_all",
    "mean_violation_conditional", "mean_relative_violation_all",
    "mean_relative_violation_conditional", "support_status",
)
SUMMARY_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "cv_protocol_version",
    "evaluation_design", "evaluation_scope", "oof_prediction_scope",
    "category_system", "category_label", "metric", "protein_equal_mean", "protein_std",
    "protein_median", "protein_q25", "protein_q75", "n_proteins_total",
    "n_proteins_finite", "triangle_pooled_value", "triangle_pooled_status",
)
LENGTH_BIN_SUMMARY_COLUMNS = SUMMARY_COLUMNS + ("protein_length_bin",)
LENGTH_EFFECT_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "cv_protocol_version",
    "evaluation_design", "evaluation_scope", "oof_prediction_scope", "category_system",
    "category_label", "metric", "n_proteins", "log_length_mean", "log_length_std",
    "standardized_log_length_slope", "regression_intercept", "spearman_rho",
    "association_status",
)
PAIRED_CONTRAST_COLUMNS = (
    "layer", "evaluation_scope", "category_system", "category_label", "metric",
    "analysis_name_a", "analysis_name_b", "n_paired_proteins",
    "mean_difference_a_minus_b", "std_difference", "median_difference",
    "q25_difference", "q75_difference", "contrast_status",
)
EDGE_CONTRAST_COLUMNS = (
    "layer", "metric", "category_label_a", "category_label_b", "n_paired_proteins",
    "mean_difference_a_minus_b", "std_difference", "median_difference",
    "q25_difference", "q75_difference", "contrast_status",
)
EDGE_REFERENCE_BALANCE_COLUMNS = (
    "protein_id", "source", "n_residues", "protein_length_bin", "category_label",
    "n_triangles", "triangle_fraction", "reference_rms_mean", "reference_rms_median",
    "reference_slack_mean", "reference_slack_median", "clearly_acute_fraction",
    "near_right_fraction", "clearly_obtuse_fraction", "support_status",
)
THRESHOLD_COLUMNS = (
    "protein_id", "source", "n_residues", "protein_length_bin", "n_triangles",
    "rms_tertile_1", "rms_tertile_2", "area_tertile_1", "area_tertile_2",
    "slack_tertile_1", "slack_tertile_2", "n_rms_small", "n_rms_medium",
    "n_rms_large", "n_clearly_acute", "n_near_right", "n_clearly_obtuse",
    "n_exact90_acute", "n_exact90_right_within_tolerance", "n_exact90_obtuse",
    *(f"n_rms_by_angle_{label}" for label in CATEGORY_LABELS["rms_by_angle"]),
    "n_area_low", "n_area_medium", "n_area_high", "n_slack_low", "n_slack_medium",
    "n_slack_high", "rms_median", "rms_p90", "area_median", "area_p90",
    "slack_median", "slack_p90", "angle_sign_relative_tolerance",
    "reference_digest", "status",
)
COHORT_COLUMNS = (
    "protein_id", "source", "n_residues", "n_pairs", "protein_length_bin",
    "outer_fold", "historical_master_role", "historical_protein_split",
    "cohort_digest", "outer_fold_digest", "pair_fold_manifest_digest",
    "pair_fold_assignment_digest",
)
PREPARATION_FAILURE_COLUMNS = ("protein_id", "source", "stage", "reason")
RUNTIME_COLUMNS = (
    "stage", "layer", "protein_id", "started_at", "finished_at", "elapsed_seconds",
    "peak_rss_mb", "status", "detail",
)
GEOMETRY_REPRODUCTION_SUMMARY_COLUMNS = (
    "analysis_name", "model_family", "layer", "metric", "n_compared",
    "maximum_absolute_difference", "tolerance", "status",
)
GEOMETRY_REPRODUCTION_FAILURE_COLUMNS = (
    "analysis_name", "model_family", "layer", "protein_id", "metric",
    "producer_value", "replay_value", "absolute_difference", "tolerance", "reason",
)
DERIVED_METRICS = (
    "violation_fraction", "mean_violation_all", "mean_violation_conditional",
    "mean_relative_violation_all", "mean_relative_violation_conditional",
    "abs_0_5_fraction_all", "abs_5_10_fraction_all", "abs_10_plus_fraction_all",
    "rel_0_0p05_fraction_all", "rel_0p05_0p10_fraction_all",
    "rel_0p10_plus_fraction_all", "relative_violation_max",
    "relative_violation_p99_conditional",
)


def triangle_count(n_residues: int) -> int:
    return math.comb(int(n_residues), 3) if int(n_residues) >= 3 else 0


def iter_triangle_indices(
    n_residues: int, batch_size: int = DEFAULT_TRIANGLE_BATCH_SIZE,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield every unordered triple once, in deterministic ``k < j < i`` order."""
    if batch_size < 1:
        raise ValueError("batch_size must be positive")
    for i_value in range(2, int(n_residues)):
        j_all, k_all = np.tril_indices(i_value, k=-1)
        for start in range(0, j_all.size, batch_size):
            stop = min(start + batch_size, j_all.size)
            yield (
                np.full(stop - start, i_value, dtype=np.int64),
                j_all[start:stop].astype(np.int64, copy=False),
                k_all[start:stop].astype(np.int64, copy=False),
            )


def triangle_geometry(
    coordinates: np.ndarray, tri_i: np.ndarray, tri_j: np.ndarray, tri_k: np.ndarray,
) -> dict[str, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    ci, cj, ck = coordinates[tri_i], coordinates[tri_j], coordinates[tri_k]
    sides = np.stack((
        np.linalg.norm(ci - cj, axis=1), np.linalg.norm(ci - ck, axis=1),
        np.linalg.norm(cj - ck, axis=1),
    ), axis=1)
    sides.sort(axis=1)
    a, b, c = sides.T
    if np.any(a <= 0.0):
        raise ValueError("Reference triangles must have three positive side lengths")
    rms = np.sqrt((a * a + b * b + c * c) / 3.0)
    area = 0.5 * np.linalg.norm(np.cross(cj - ci, ck - ci), axis=1)
    slack = np.clip((a + b - c) / c, 0.0, 1.0)
    if not all(np.all(np.isfinite(v)) for v in (rms, area, slack)):
        raise ValueError("Reference triangle geometry is nonfinite")
    cosine = np.clip((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    sign_value = c * c - a * a - b * b
    sign_scale = np.maximum.reduce((c * c, a * a + b * b, np.ones_like(c)))
    return {
        "rms": rms, "area": area, "slack": slack, "angle": angle,
        "angle_sign": sign_value,
        "angle_sign_tolerance": ANGLE_SIGN_RELATIVE_TOLERANCE * sign_scale,
    }


def tertile_codes(values: np.ndarray, first: float, second: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values < first, 0, np.where(values < second, 1, 2)).astype(np.uint8)


def category_codes(
    geometry: Mapping[str, np.ndarray], thresholds: Mapping[str, float],
) -> dict[str, np.ndarray]:
    rms = tertile_codes(geometry["rms"], thresholds["rms_tertile_1"], thresholds["rms_tertile_2"])
    angle_values = np.asarray(geometry["angle"], dtype=np.float64)
    angle = np.where(
        angle_values < ANGLE_ACUTE_MAX_DEGREES, 0,
        np.where(angle_values <= ANGLE_OBTUSE_MIN_DEGREES, 1, 2),
    ).astype(np.uint8)
    sign = np.asarray(geometry["angle_sign"], dtype=np.float64)
    tolerance = np.asarray(geometry["angle_sign_tolerance"], dtype=np.float64)
    exact = np.where(sign < -tolerance, 0, np.where(sign > tolerance, 2, 1)).astype(np.uint8)
    slack = tertile_codes(
        geometry["slack"], thresholds["slack_tertile_1"], thresholds["slack_tertile_2"],
    )
    area = tertile_codes(
        geometry["area"], thresholds["area_tertile_1"], thresholds["area_tertile_2"],
    )
    return {
        "all_triangles": np.zeros(rms.size, dtype=np.uint8), "rms_size": rms,
        "angle_shape": angle, "angle_shape_exact90": exact,
        "rms_by_angle": (3 * rms + angle).astype(np.uint8),
        "relative_slack": slack, "area": area,
    }


def edge_model_count_codes(
    pair_fold_matrix: np.ndarray, tri_i: np.ndarray, tri_j: np.ndarray, tri_k: np.ndarray,
) -> np.ndarray:
    """Return 0/1/2 for triangles whose edges use one/two/three source models."""
    matrix = np.asarray(pair_fold_matrix)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1]:
        raise ValueError("Pair-fold source matrix must be square")
    edge_a = matrix[tri_i, tri_j]
    edge_b = matrix[tri_i, tri_k]
    edge_c = matrix[tri_j, tri_k]
    if np.any(edge_a < 0) or np.any(edge_b < 0) or np.any(edge_c < 0):
        raise ValueError("Every triangle edge must have a source-model identity")
    count = 1 + (edge_b != edge_a).astype(np.uint8)
    count += ((edge_c != edge_a) & (edge_c != edge_b)).astype(np.uint8)
    return count - 1


def symmetric_pair_matrix(
    n_residues: int, pair_i: np.ndarray, pair_j: np.ndarray, values: np.ndarray,
    *, diagonal: float | int = 0,
) -> np.ndarray:
    values = np.asarray(values)
    if values.shape != np.asarray(pair_i).shape or values.shape != np.asarray(pair_j).shape:
        raise ValueError("Pair values and indices must have identical shapes")
    matrix = np.empty((int(n_residues), int(n_residues)), dtype=values.dtype)
    matrix.fill(diagonal)
    matrix[pair_i, pair_j] = values
    matrix[pair_j, pair_i] = values
    return matrix


def assemble_oof_predictions(
    features: np.ndarray, assignments: np.ndarray,
    models: Mapping[int, tuple[object, ...]],
) -> np.ndarray:
    """Apply each pair-fold model only to its held-out rows and enforce exact coverage."""
    x = np.asarray(features, dtype=np.float64)
    folds = np.asarray(assignments, dtype=np.int16)
    if x.ndim != 2 or x.shape[0] != folds.size or not np.all(np.isfinite(x)):
        raise ValueError("Invalid OOF assembly inputs")
    if set(models) != set(range(N_PAIR_FOLDS)):
        raise ValueError("Exactly five pair-fold models are required")
    predictions = np.full(folds.size, np.nan, dtype=np.float64)
    writes = np.zeros(folds.size, dtype=np.uint8)
    for fold in range(N_PAIR_FOLDS):
        selected = np.flatnonzero(folds == fold)
        model = models[fold]
        if len(model) == 2:
            coefficient, intercept = model
            transformed = x[selected]
        elif len(model) == 4:
            coefficient, intercept, feature_mean, feature_scale = model
            mean = np.asarray(feature_mean, dtype=np.float64)
            scale = np.asarray(feature_scale, dtype=np.float64)
            if mean.shape != (x.shape[1],) or scale.shape != mean.shape or np.any(scale <= 0.0):
                raise ValueError(f"Invalid scaler state for pair fold {fold}")
            transformed = (x[selected] - mean) / scale
        else:
            raise ValueError("OOF linear models must contain raw or standardized replay state")
        coefficient = np.asarray(coefficient, dtype=np.float64)
        if selected.size == 0 or coefficient.shape != (x.shape[1],):
            raise ValueError(f"Invalid or empty pair fold {fold}")
        if np.any(writes[selected]):
            raise ValueError("Duplicate OOF pair coverage")
        predictions[selected] = transformed @ coefficient + float(intercept)
        writes[selected] += 1
    if not np.all(writes == 1) or not np.all(np.isfinite(predictions)):
        raise ValueError("Missing, duplicate, or nonfinite OOF pair coverage")
    return predictions


@dataclass
class ReferenceCache:
    directory: Path
    n_triangles: int
    rms: np.memmap
    slack: np.memmap
    angle: np.memmap
    codes: Mapping[str, np.memmap]

    def cleanup(self) -> None:
        for value in (self.rms, self.slack, self.angle, *self.codes.values()):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


def _open_memmap(path: Path, dtype: str, size: int) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(size,))


def _array_digest(arrays: Iterable[np.ndarray], payload: Mapping[str, object]) -> str:
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode())
    for array in arrays:
        values = np.asarray(array)
        digest.update(str(values.dtype).encode())
        digest.update(np.asarray(values.shape, dtype=np.int64).tobytes())
        flat = values.reshape(-1)
        for start in range(0, flat.size, 1_000_000):
            digest.update(np.ascontiguousarray(flat[start:start + 1_000_000]).tobytes())
    return digest.hexdigest()


def build_reference_cache(
    coordinates: np.ndarray, directory: str | Path, *,
    thresholds: Mapping[str, float] | None = None,
    batch_size: int = DEFAULT_TRIANGLE_BATCH_SIZE,
) -> tuple[ReferenceCache, dict[str, object]]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    n_triangles = triangle_count(coordinates.shape[0])
    if coordinates.ndim != 2 or coordinates.shape[1] != 3 or n_triangles == 0:
        raise ValueError("At least three finite C-alpha coordinates are required")
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    raw = {
        name: _open_memmap(directory / f"{name}.npy", "float64", n_triangles)
        for name in ("rms", "area", "slack", "angle", "angle_sign", "angle_sign_tolerance")
    }
    cursor = 0
    for tri_i, tri_j, tri_k in iter_triangle_indices(coordinates.shape[0], batch_size):
        geometry = triangle_geometry(coordinates, tri_i, tri_j, tri_k)
        stop = cursor + tri_i.size
        for name, values in geometry.items():
            raw[name][cursor:stop] = values
        cursor = stop
    if cursor != n_triangles:
        raise RuntimeError("Reference triangle enumeration is incomplete")
    calculated = {
        f"{name}_tertile_{index}": float(np.quantile(raw[name], index / 3.0))
        for name in ("rms", "area", "slack") for index in (1, 2)
    }
    if thresholds is not None:
        for key, observed in calculated.items():
            if not math.isclose(observed, float(thresholds[key]), rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Frozen reference threshold mismatch for {key}")
    selected = calculated if thresholds is None else {key: float(thresholds[key]) for key in calculated}
    codes = {
        system: _open_memmap(directory / f"code_{system}.npy", "uint8", n_triangles)
        for system in CATEGORY_LABELS
    }
    for system, values in category_codes(raw, selected).items():
        codes[system][:] = values
        codes[system].flush()
    counts: dict[str, int] = {}
    count_keys = {
        "rms_size": ("n_rms_small", "n_rms_medium", "n_rms_large"),
        "angle_shape": ("n_clearly_acute", "n_near_right", "n_clearly_obtuse"),
        "angle_shape_exact90": (
            "n_exact90_acute", "n_exact90_right_within_tolerance", "n_exact90_obtuse",
        ),
        "rms_by_angle": tuple(
            f"n_rms_by_angle_{label}" for label in CATEGORY_LABELS["rms_by_angle"]
        ),
        "area": ("n_area_low", "n_area_medium", "n_area_high"),
        "relative_slack": ("n_slack_low", "n_slack_medium", "n_slack_high"),
    }
    for system, keys in count_keys.items():
        for code, key in enumerate(keys):
            counts[key] = int(np.count_nonzero(codes[system] == code))
    stats: dict[str, object] = {
        "n_triangles": n_triangles, **selected, **counts,
        **{
            f"{name}_{quantile}": float(np.quantile(raw[name], probability))
            for name in ("rms", "area", "slack")
            for quantile, probability in (("median", 0.5), ("p90", 0.9))
        },
        "angle_sign_relative_tolerance": ANGLE_SIGN_RELATIVE_TOLERANCE,
    }
    digest_payload = {key: stats[key] for key in sorted(stats)}
    stats["reference_digest"] = _array_digest(
        (raw["rms"], raw["area"], raw["slack"], raw["angle"], *(codes[s] for s in CATEGORY_LABELS)),
        digest_payload,
    )
    for name in ("area", "angle_sign", "angle_sign_tolerance"):
        mmap = getattr(raw[name], "_mmap", None)
        if mmap is not None:
            mmap.close()
        (directory / f"{name}.npy").unlink(missing_ok=True)
    return ReferenceCache(directory, n_triangles, raw["rms"], raw["slack"], raw["angle"], codes), stats


class _PrimitiveAccumulator:
    def __init__(self, n_categories: int) -> None:
        self.n = np.zeros(n_categories, dtype=np.int64)
        self.nv = np.zeros(n_categories, dtype=np.int64)
        self.abs_sum = np.zeros(n_categories, dtype=np.float64)
        self.abs_max = np.zeros(n_categories, dtype=np.float64)
        self.rel_sum = np.zeros(n_categories, dtype=np.float64)
        self.rel_max = np.zeros(n_categories, dtype=np.float64)
        self.abs_bands = np.zeros((n_categories, 3), dtype=np.int64)
        self.rel_bands = np.zeros((n_categories, 3), dtype=np.int64)

    def add(self, codes: np.ndarray, delta: np.ndarray, relative: np.ndarray) -> None:
        for code in range(self.n.size):
            selected = codes == code
            self.n[code] += int(np.count_nonzero(selected))
            violation = selected & (delta > 0.0)
            d, r = delta[violation], relative[violation]
            self.nv[code] += d.size
            self.abs_sum[code] += float(np.sum(d, dtype=np.float64))
            self.rel_sum[code] += float(np.sum(r, dtype=np.float64))
            if d.size:
                self.abs_max[code] = max(self.abs_max[code], float(np.max(d)))
                self.rel_max[code] = max(self.rel_max[code], float(np.max(r)))
                self.abs_bands[code] += (
                    np.count_nonzero(d < 5.0), np.count_nonzero((d >= 5.0) & (d < 10.0)),
                    np.count_nonzero(d >= 10.0),
                )
                self.rel_bands[code] += (
                    np.count_nonzero(r < 0.05), np.count_nonzero((r >= 0.05) & (r < 0.10)),
                    np.count_nonzero(r >= 0.10),
                )


def _safe_divide(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def compute_metric_rows(
    predicted_distances: np.ndarray, reference_cache: ReferenceCache,
    identity: Mapping[str, object], *, pair_fold_matrix: np.ndarray | None = None,
    batch_size: int = DEFAULT_TRIANGLE_BATCH_SIZE,
) -> list[dict[str, object]]:
    """Reduce one OOF distance matrix to compact category primitives."""
    normalized_identity = dict(identity)
    # Retain compatibility with the independent geometry oracle while keeping
    # every emitted row on the v2 schema. Production callers supply these
    # explicitly; the defaults are only for synthetic shared-OLS parity tests.
    normalized_identity.pop("protein_split", None)
    normalized_identity.setdefault("feature_dim", FEATURE_DIM)
    normalized_identity.setdefault("cv_protocol_version", CV_PROTOCOL_VERSION)
    normalized_identity.setdefault("evaluation_design", "protein_outer_5fold")
    normalized_identity.setdefault("outer_fold", 0)
    normalized_identity.setdefault("oof_prediction_scope", "heldout_proteins_all_pairs")
    predicted = np.asarray(predicted_distances, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[0] != predicted.shape[1] or not np.all(np.isfinite(predicted)):
        raise ValueError("Predicted distances must be a finite square matrix")
    lower = np.tril_indices(predicted.shape[0], k=-1)
    n_edges = lower[0].size
    n_negative = int(np.count_nonzero(predicted[lower] < 0.0))
    clipped = np.maximum(predicted, 0.0)
    labels = CATEGORY_LABELS if pair_fold_matrix is None else ALL_CATEGORY_LABELS
    accumulators = {system: _PrimitiveAccumulator(len(values)) for system, values in labels.items()}
    descriptor, rel_name = tempfile.mkstemp(
        prefix="triangle_relative_", suffix=".npy", dir=reference_cache.directory,
    )
    os.close(descriptor)
    rel_path = Path(rel_name)
    rel_values = np.lib.format.open_memmap(
        rel_path, mode="w+", dtype="float64", shape=(reference_cache.n_triangles,),
    )
    cursor = 0
    try:
        for tri_i, tri_j, tri_k in iter_triangle_indices(predicted.shape[0], batch_size):
            sides = np.stack((
                clipped[tri_i, tri_j], clipped[tri_i, tri_k], clipped[tri_j, tri_k],
            ), axis=1)
            sides.sort(axis=1)
            raw_delta = sides[:, 2] - sides[:, 0] - sides[:, 1]
            delta = np.where(raw_delta > TRIANGLE_TOLERANCE_ANGSTROM, raw_delta, 0.0)
            stop = cursor + tri_i.size
            relative = delta / reference_cache.rms[cursor:stop]
            rel_values[cursor:stop] = relative
            for system in CATEGORY_LABELS:
                accumulators[system].add(reference_cache.codes[system][cursor:stop], delta, relative)
            if pair_fold_matrix is not None:
                code = edge_model_count_codes(pair_fold_matrix, tri_i, tri_j, tri_k)
                accumulators["edge_model_count"].add(code, delta, relative)
            cursor = stop
        if cursor != reference_cache.n_triangles:
            raise RuntimeError("Prediction and reference enumeration differ")
        rel_values.flush()
        rows: list[dict[str, object]] = []
        for system, system_labels in labels.items():
            accumulator = accumulators[system]
            if system == "edge_model_count":
                code_chunks = []
                for tri_i, tri_j, tri_k in iter_triangle_indices(predicted.shape[0], batch_size):
                    code_chunks.append(edge_model_count_codes(pair_fold_matrix, tri_i, tri_j, tri_k))
                codes = np.concatenate(code_chunks)
            else:
                codes = np.asarray(reference_cache.codes[system])
            for code, label in enumerate(system_labels):
                n, nv = int(accumulator.n[code]), int(accumulator.nv[code])
                conditional = np.asarray(rel_values)[(codes == code) & (np.asarray(rel_values) > 0.0)]
                p50, p90, p99 = (
                    np.quantile(conditional, (0.5, 0.9, 0.99)) if conditional.size
                    else (float("nan"),) * 3
                )
                row = {
                    **normalized_identity, "evaluation_scope": EVALUATION_SCOPE,
                    "category_system": system, "category_label": label,
                    "n_triangles": n, "n_violations": nv,
                    "n_negative_predictions": n_negative,
                    "negative_prediction_fraction": _safe_divide(n_negative, n_edges),
                    "violation_sum_angstrom": float(accumulator.abs_sum[code]),
                    "violation_max_angstrom": float(accumulator.abs_max[code]),
                    "n_abs_0_5": int(accumulator.abs_bands[code, 0]),
                    "n_abs_5_10": int(accumulator.abs_bands[code, 1]),
                    "n_abs_10_plus": int(accumulator.abs_bands[code, 2]),
                    "relative_violation_sum": float(accumulator.rel_sum[code]),
                    "relative_violation_max": float(accumulator.rel_max[code]),
                    "n_rel_0_0p05": int(accumulator.rel_bands[code, 0]),
                    "n_rel_0p05_0p10": int(accumulator.rel_bands[code, 1]),
                    "n_rel_0p10_plus": int(accumulator.rel_bands[code, 2]),
                    "relative_violation_p50_conditional": float(p50),
                    "relative_violation_p90_conditional": float(p90),
                    "relative_violation_p99_conditional": float(p99),
                    "violation_fraction": _safe_divide(nv, n),
                    "mean_violation_all": _safe_divide(accumulator.abs_sum[code], n),
                    "mean_violation_conditional": _safe_divide(accumulator.abs_sum[code], nv),
                    "mean_relative_violation_all": _safe_divide(accumulator.rel_sum[code], n),
                    "mean_relative_violation_conditional": _safe_divide(accumulator.rel_sum[code], nv),
                    "support_status": "zero_support" if n == 0 else (
                        "zero_violations" if nv == 0 else "complete"
                    ),
                }
                if set(row) != set(PER_PROTEIN_COLUMNS):
                    raise ValueError(
                        f"Metric schema mismatch: missing={set(PER_PROTEIN_COLUMNS)-set(row)}, "
                        f"extra={set(row)-set(PER_PROTEIN_COLUMNS)}"
                    )
                if sum(row[k] for k in ("n_abs_0_5", "n_abs_5_10", "n_abs_10_plus")) != nv:
                    raise RuntimeError("Absolute bands do not partition violations")
                if sum(row[k] for k in ("n_rel_0_0p05", "n_rel_0p05_0p10", "n_rel_0p10_plus")) != nv:
                    raise RuntimeError("Relative bands do not partition violations")
                if n and nv:
                    if not math.isclose(
                        row["mean_violation_all"],
                        row["violation_fraction"] * row["mean_violation_conditional"],
                        rel_tol=1e-12, abs_tol=1e-12,
                    ) or not math.isclose(
                        row["mean_relative_violation_all"],
                        row["violation_fraction"] * row["mean_relative_violation_conditional"],
                        rel_tol=1e-12, abs_tol=1e-12,
                    ):
                        raise RuntimeError("All/conditional mean identity failed")
                rows.append(row)
        return rows
    finally:
        mmap = getattr(rel_values, "_mmap", None)
        if mmap is not None:
            mmap.close()
        rel_path.unlink(missing_ok=True)


def edge_model_count_reference_balance(
    reference_cache: ReferenceCache, pair_fold_matrix: np.ndarray,
    identity: Mapping[str, object], *, batch_size: int = DEFAULT_TRIANGLE_BATCH_SIZE,
) -> list[dict[str, object]]:
    codes = []
    for tri_i, tri_j, tri_k in iter_triangle_indices(pair_fold_matrix.shape[0], batch_size):
        codes.append(edge_model_count_codes(pair_fold_matrix, tri_i, tri_j, tri_k))
    all_codes = np.concatenate(codes)
    if all_codes.size != reference_cache.n_triangles:
        raise RuntimeError("Edge-model-count enumeration is incomplete")
    rows = []
    broad_angle = np.asarray(reference_cache.codes["angle_shape"])
    for code, label in enumerate(EDGE_MODEL_COUNT_LABELS):
        selected = all_codes == code
        n = int(np.count_nonzero(selected))
        row = {
            **identity, "category_label": label, "n_triangles": n,
            "triangle_fraction": _safe_divide(n, reference_cache.n_triangles),
            "reference_rms_mean": float(np.mean(reference_cache.rms[selected])) if n else float("nan"),
            "reference_rms_median": float(np.median(reference_cache.rms[selected])) if n else float("nan"),
            "reference_slack_mean": float(np.mean(reference_cache.slack[selected])) if n else float("nan"),
            "reference_slack_median": float(np.median(reference_cache.slack[selected])) if n else float("nan"),
            "clearly_acute_fraction": _safe_divide(np.count_nonzero(selected & (broad_angle == 0)), n),
            "near_right_fraction": _safe_divide(np.count_nonzero(selected & (broad_angle == 1)), n),
            "clearly_obtuse_fraction": _safe_divide(np.count_nonzero(selected & (broad_angle == 2)), n),
            "support_status": "complete" if n else "zero_support",
        }
        if set(row) != set(EDGE_REFERENCE_BALANCE_COLUMNS):
            raise ValueError("Edge reference-balance schema mismatch")
        rows.append(row)
    if sum(row["n_triangles"] for row in rows) != reference_cache.n_triangles:
        raise RuntimeError("Edge-model-count categories do not partition triangles")
    return rows


def metric_value(row: Mapping[str, object], metric: str) -> float:
    if metric in row:
        return float(row[metric])
    column = {
        "abs_0_5_fraction_all": "n_abs_0_5", "abs_5_10_fraction_all": "n_abs_5_10",
        "abs_10_plus_fraction_all": "n_abs_10_plus",
        "rel_0_0p05_fraction_all": "n_rel_0_0p05",
        "rel_0p05_0p10_fraction_all": "n_rel_0p05_0p10",
        "rel_0p10_plus_fraction_all": "n_rel_0p10_plus",
    }.get(metric)
    if column is None:
        raise KeyError(metric)
    return _safe_divide(float(row[column]), int(row["n_triangles"]))


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def load_json(path: str | Path) -> dict[str, object]:
    with Path(path).open() as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected JSON object in {path}")
    return value


def read_csv_strict(path: str | Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"CSV schema mismatch in {path}: {reader.fieldnames}")
        return list(reader)


def validate_family_manifest(
    family_root: str | Path, spec: FamilySpec, fold_digests: Mapping[str, object],
) -> dict[str, object]:
    root = Path(family_root)
    manifest = load_json(root / "run_manifest.json")
    expected = {
        "analysis_name": spec.analysis_name, "model_family": spec.model_family,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "cv_protocol_version": CV_PROTOCOL_VERSION, "feature_dim": FEATURE_DIM,
        "evaluation_scope": EVALUATION_SCOPE, "evaluation_design": spec.evaluation_design,
    }
    for key, value in expected.items():
        if manifest.get(key) != value:
            raise ValueError(f"Producer {spec.analysis_name} has wrong {key}")
    expected_producer_scope = (
        "heldout_proteins_all_pairs_recombined_sequence_gates"
        if spec.uses_sequence_gates else spec.oof_prediction_scope
    )
    if manifest.get("oof_prediction_scope") != expected_producer_scope:
        raise ValueError(f"Producer {spec.analysis_name} has wrong OOF prediction scope")
    if manifest.get("run_status") != "complete" or manifest.get("requested_layers") != list(LAYERS):
        raise ValueError(f"Producer {spec.analysis_name} is incomplete")
    completed = manifest.get("completed_layers")
    if completed is not None and completed != list(LAYERS):
        raise ValueError(f"Producer {spec.analysis_name} layer coverage is incomplete")
    for key in ("cohort_digest", "outer_fold_digest", "pair_fold_manifest_digest"):
        if manifest.get(key) != fold_digests[key]:
            raise ValueError(f"Producer {spec.analysis_name} has wrong {key}")
    feature_names = manifest.get("feature_names")
    expected_names = [f"pair_ij_ch_{i:03d}" for i in range(128)] + [
        f"pair_ji_ch_{i:03d}" for i in range(128)
    ]
    if feature_names != expected_names:
        raise ValueError(f"Producer {spec.analysis_name} feature order mismatch")
    configuration = manifest.get("configuration")
    if not isinstance(configuration, dict) or hashlib.sha256(
        json.dumps(
            configuration, sort_keys=True, separators=(",", ":"), allow_nan=False,
        ).encode("utf-8")
    ).hexdigest() != manifest.get("configuration_digest"):
        raise ValueError(f"Producer {spec.analysis_name} configuration digest mismatch")
    fixed_path = manifest.get("fixed_configuration_path")
    fixed_digest = manifest.get("fixed_configuration_sha256")
    if fixed_path is not None:
        path = root / str(fixed_path)
        if not path.is_file() or sha256_file(path) != fixed_digest:
            raise ValueError(f"Producer {spec.analysis_name} fixed configuration changed")
    return manifest


def load_and_validate_replay_index(
    family_root: str | Path, spec: FamilySpec, manifest: Mapping[str, object], *,
    verify_artifacts: bool = True, verify_coverage: bool = True,
) -> list[dict[str, str]]:
    root = Path(family_root)
    rows = read_csv_strict(root / "model_artifacts/replay_index.csv", REPLAY_INDEX_COLUMNS)
    expected_count = int(manifest.get("expected_replay_artifacts", -1))
    if expected_count >= 0 and len(rows) != expected_count:
        raise ValueError(f"Replay index count mismatch for {spec.analysis_name}")
    artifact_hashes: dict[Path, str] = {}
    responsibilities: set[tuple[str, ...]] = set()
    for row in rows:
        if (
            row["analysis_name"] != spec.analysis_name or row["model_family"] != spec.model_family
            or row["status"] != "complete"
            or row["configuration_digest"] != manifest["configuration_digest"]
        ):
            raise ValueError(f"Invalid replay-index identity/status for {spec.analysis_name}")
        responsibility = tuple(row[k] for k in (
            "layer", "outer_fold", "protein_id", "pair_fold", "sequence_gate",
        ))
        if responsibility in responsibilities:
            raise ValueError(f"Duplicate replay responsibility: {responsibility}")
        responsibilities.add(responsibility)
        relative = Path(row["artifact_path"])
        if relative.is_absolute() or ".." in relative.parts:
            raise ValueError("Replay artifact_path must be family-relative")
        if verify_artifacts:
            artifact = root / relative
            if artifact not in artifact_hashes:
                if not artifact.is_file():
                    raise FileNotFoundError(artifact)
                artifact_hashes[artifact] = sha256_file(artifact)
            if artifact_hashes[artifact] != row["artifact_sha256"]:
                raise ValueError(f"Replay artifact digest mismatch: {artifact}")
    if not verify_coverage:
        return rows
    if spec.uses_pair_folds:
        protein_ids = {row["protein_id"] for row in rows}
        expected = {
            (str(layer), protein_id, str(pair_fold))
            for layer in LAYERS for protein_id in protein_ids for pair_fold in range(N_PAIR_FOLDS)
        }
        observed = {(row["layer"], row["protein_id"], row["pair_fold"]) for row in rows}
        if (
            len(protein_ids) != EXPECTED_PROTEINS or observed != expected
            or any(row["outer_fold"] or row["sequence_gate"] for row in rows)
        ):
            raise ValueError("Per-protein replay responsibilities are incomplete")
    elif spec.uses_sequence_gates:
        expected = {
            (str(layer), str(fold), gate)
            for layer in LAYERS for fold in range(N_OUTER_FOLDS) for gate in SEQUENCE_GATE_LABELS
        }
        observed = {(row["layer"], row["outer_fold"], row["sequence_gate"]) for row in rows}
        if observed != expected or any(row["protein_id"] or row["pair_fold"] for row in rows):
            raise ValueError("Sequence-gated replay responsibilities are incomplete")
    else:
        expected = {(str(layer), str(fold)) for layer in LAYERS for fold in range(N_OUTER_FOLDS)}
        observed = {(row["layer"], row["outer_fold"]) for row in rows}
        if observed != expected or any(
            row["protein_id"] or row["pair_fold"] or row["sequence_gate"] for row in rows
        ):
            raise ValueError("Shared-family replay responsibilities are incomplete")
    return rows


def index_replay_rows(
    rows: Sequence[Mapping[str, str]], keys: Sequence[str],
) -> dict[tuple[str, ...], Mapping[str, str]]:
    index: dict[tuple[str, ...], Mapping[str, str]] = {}
    for row in rows:
        key = tuple(str(row[column]) for column in keys)
        if key in index:
            raise ValueError(f"Duplicate replay key {key}")
        index[key] = row
    return index


__all__ = [name for name in globals() if name.isupper()] + [
    "FamilySpec", "ReferenceCache", "assemble_oof_predictions", "build_reference_cache",
    "category_codes", "compute_metric_rows", "edge_model_count_codes",
    "edge_model_count_reference_balance", "index_replay_rows", "iter_triangle_indices",
    "load_and_validate_replay_index", "load_json", "metric_value", "read_csv_strict",
    "sha256_file", "symmetric_pair_matrix", "tertile_codes", "triangle_count",
    "triangle_geometry", "validate_family_manifest",
]

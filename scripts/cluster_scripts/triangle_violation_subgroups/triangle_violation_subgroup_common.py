#!/usr/bin/env python3
"""Shared exact-geometry helpers for triangle-violation subgroup replay.

The module intentionally contains no fitting code.  It builds frozen reference
categories from final-PDB C-alpha coordinates and reduces decoded distance
matrices to compact per-protein primitives without retaining triangle rows.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, Mapping, Sequence

import numpy as np


ANALYSIS_NAME = "triangle_violation_subgroups"
OUTPUT_SCHEMA_VERSION = "1.0"
EVALUATION_SCOPE = "all_edges"
TRIANGLE_TOLERANCE_ANGSTROM = 1e-12
ANGLE_SIGN_RELATIVE_TOLERANCE = 1e-12
ANGLE_ACUTE_MAX_DEGREES = 88.0
ANGLE_OBTUSE_MIN_DEGREES = 92.0
DEFAULT_TRIANGLE_BATCH_SIZE = 250_000

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

MODEL_IDENTITIES = (
    ("general_base256_combined", "shared_ols", "unseen_proteins"),
    (
        "per_protein_base256_combined",
        "per_protein_ols",
        "mixed_within_protein_train_and_heldout",
    ),
    ("shared_xgboost_base256_combined", "shared_xgboost", "unseen_proteins"),
    ("analysis_1_sequence_gated", "gated_ols", "unseen_proteins"),
)

PER_PROTEIN_COLUMNS = (
    "analysis_name", "model_family", "layer", "protein_id", "source",
    "protein_split", "evaluation_scope", "generalization_scope",
    "category_system", "category_label", "n_residues", "protein_length_bin",
    "n_triangles", "n_violations", "n_negative_predictions",
    "negative_prediction_fraction", "violation_sum_angstrom",
    "violation_max_angstrom", "n_abs_0_5", "n_abs_5_10", "n_abs_10_plus",
    "relative_violation_sum", "relative_violation_max", "n_rel_0_0p05",
    "n_rel_0p05_0p10", "n_rel_0p10_plus",
    "relative_violation_p50_conditional", "relative_violation_p90_conditional",
    "relative_violation_p99_conditional", "violation_fraction",
    "mean_violation_all", "mean_violation_conditional",
    "mean_relative_violation_all", "mean_relative_violation_conditional",
    "support_status",
)

SUMMARY_COLUMNS = (
    "analysis_name", "model_family", "layer", "evaluation_scope",
    "category_system", "category_label", "metric", "protein_equal_mean",
    "protein_std", "protein_median", "protein_q25", "protein_q75",
    "n_proteins_total", "n_proteins_finite", "triangle_pooled_value",
    "triangle_pooled_status",
)

LENGTH_BIN_SUMMARY_COLUMNS = SUMMARY_COLUMNS + ("protein_length_bin",)

LENGTH_EFFECT_COLUMNS = (
    "analysis_name", "model_family", "layer", "evaluation_scope",
    "category_system", "category_label", "metric", "n_proteins",
    "log_length_mean", "log_length_std", "standardized_log_length_slope",
    "regression_intercept", "spearman_rho", "association_status",
)

PAIRED_CONTRAST_COLUMNS = (
    "layer", "evaluation_scope", "category_system", "category_label", "metric",
    "analysis_name_a", "analysis_name_b", "n_paired_proteins",
    "mean_difference_a_minus_b", "std_difference", "median_difference",
    "q25_difference", "q75_difference", "contrast_status",
)

THRESHOLD_COLUMNS = (
    "protein_id", "source", "protein_split", "n_residues",
    "protein_length_bin", "n_triangles", "rms_tertile_1", "rms_tertile_2",
    "area_tertile_1", "area_tertile_2", "slack_tertile_1", "slack_tertile_2",
    "n_rms_small", "n_rms_medium", "n_rms_large", "n_clearly_acute",
    "n_near_right", "n_clearly_obtuse", "n_exact90_acute",
    "n_exact90_right_within_tolerance", "n_exact90_obtuse", "n_area_low",
    "n_area_medium", "n_area_high", "n_slack_low", "n_slack_medium",
    "n_slack_high", "rms_median", "rms_p90", "area_median", "area_p90",
    "slack_median", "slack_p90", "angle_sign_relative_tolerance",
    "reference_digest", "status",
)

SPLIT_ASSIGNMENT_COLUMNS = (
    "master_manifest_digest", "protein_id", "source", "n_residues",
    "protein_length_bin", "master_role", "protein_split",
)

PREPARATION_FAILURE_COLUMNS = ("protein_id", "source", "stage", "reason")

RUNTIME_COLUMNS = (
    "stage", "layer", "protein_id", "started_at", "finished_at",
    "elapsed_seconds", "peak_rss_mb", "status", "detail",
)

DERIVED_METRICS = (
    "violation_fraction", "mean_violation_all", "mean_violation_conditional",
    "mean_relative_violation_all", "mean_relative_violation_conditional",
    "abs_0_5_fraction_all", "abs_5_10_fraction_all",
    "abs_10_plus_fraction_all", "rel_0_0p05_fraction_all",
    "rel_0p05_0p10_fraction_all", "rel_0p10_plus_fraction_all",
    "relative_violation_max", "relative_violation_p99_conditional",
)


def triangle_count(n_residues: int) -> int:
    return math.comb(int(n_residues), 3) if int(n_residues) >= 3 else 0


def iter_triangle_indices(
    n_residues: int,
    batch_size: int = DEFAULT_TRIANGLE_BATCH_SIZE,
) -> Iterator[tuple[np.ndarray, np.ndarray, np.ndarray]]:
    """Yield each unordered triple once in deterministic ``k < j < i`` order."""
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
    coordinates: np.ndarray,
    tri_i: np.ndarray,
    tri_j: np.ndarray,
    tri_k: np.ndarray,
) -> dict[str, np.ndarray]:
    coordinates = np.asarray(coordinates, dtype=np.float64)
    ci, cj, ck = coordinates[tri_i], coordinates[tri_j], coordinates[tri_k]
    sides = np.stack(
        (
            np.linalg.norm(ci - cj, axis=1),
            np.linalg.norm(ci - ck, axis=1),
            np.linalg.norm(cj - ck, axis=1),
        ),
        axis=1,
    )
    sides.sort(axis=1)
    a, b, c = sides[:, 0], sides[:, 1], sides[:, 2]
    if np.any(a <= 0.0):
        raise ValueError("Reference triangles must have three strictly positive side lengths")
    rms = np.sqrt((a * a + b * b + c * c) / 3.0)
    area = 0.5 * np.linalg.norm(np.cross(cj - ci, ck - ci), axis=1)
    if np.any(rms <= 0.0) or not np.all(np.isfinite(rms)):
        raise ValueError("Reference triangles must have finite positive RMS size")
    if np.any(area < 0.0) or not np.all(np.isfinite(area)):
        raise ValueError("Reference triangle areas must be finite and nonnegative")
    slack = (a + b - c) / c
    slack_tolerance = 64.0 * np.finfo(np.float64).eps
    if np.any(slack < -slack_tolerance) or np.any(slack > 1.0 + slack_tolerance):
        raise ValueError("Reference relative slack falls outside [0,1] tolerance")
    slack = np.clip(slack, 0.0, 1.0)
    cosine = np.clip((a * a + b * b - c * c) / (2.0 * a * b), -1.0, 1.0)
    angle = np.degrees(np.arccos(cosine))
    sign_value = c * c - a * a - b * b
    sign_scale = np.maximum.reduce((c * c, a * a + b * b, np.ones_like(c)))
    sign_tolerance = ANGLE_SIGN_RELATIVE_TOLERANCE * sign_scale
    return {
        "rms": rms,
        "area": area,
        "slack": slack,
        "angle": angle,
        "angle_sign": sign_value,
        "angle_sign_tolerance": sign_tolerance,
    }


def tertile_codes(values: np.ndarray, first: float, second: float) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    return np.where(values < first, 0, np.where(values < second, 1, 2)).astype(np.uint8)


def category_codes(
    geometry: Mapping[str, np.ndarray],
    thresholds: Mapping[str, float],
) -> dict[str, np.ndarray]:
    rms_code = tertile_codes(
        geometry["rms"], float(thresholds["rms_tertile_1"]),
        float(thresholds["rms_tertile_2"]),
    )
    angle = np.asarray(geometry["angle"], dtype=np.float64)
    angle_code = np.where(
        angle < ANGLE_ACUTE_MAX_DEGREES,
        0,
        np.where(angle <= ANGLE_OBTUSE_MIN_DEGREES, 1, 2),
    ).astype(np.uint8)
    sign = np.asarray(geometry["angle_sign"], dtype=np.float64)
    sign_tol = np.asarray(geometry["angle_sign_tolerance"], dtype=np.float64)
    exact_code = np.where(sign < -sign_tol, 0, np.where(sign > sign_tol, 2, 1)).astype(np.uint8)
    slack_code = tertile_codes(
        geometry["slack"], float(thresholds["slack_tertile_1"]),
        float(thresholds["slack_tertile_2"]),
    )
    area_code = tertile_codes(
        geometry["area"], float(thresholds["area_tertile_1"]),
        float(thresholds["area_tertile_2"]),
    )
    return {
        "all_triangles": np.zeros(rms_code.size, dtype=np.uint8),
        "rms_size": rms_code,
        "angle_shape": angle_code,
        "angle_shape_exact90": exact_code,
        "rms_by_angle": (3 * rms_code + angle_code).astype(np.uint8),
        "relative_slack": slack_code,
        "area": area_code,
    }


@dataclass
class ReferenceCache:
    directory: Path
    n_triangles: int
    rms: np.memmap
    codes: Mapping[str, np.memmap]

    def cleanup(self) -> None:
        for value in (self.rms, *self.codes.values()):
            mmap = getattr(value, "_mmap", None)
            if mmap is not None:
                mmap.close()


def _open_memmap(path: Path, dtype: str, size: int) -> np.memmap:
    return np.lib.format.open_memmap(path, mode="w+", dtype=dtype, shape=(size,))


def build_reference_cache(
    coordinates: np.ndarray,
    directory: str | Path,
    *,
    thresholds: Mapping[str, float] | None = None,
    batch_size: int = DEFAULT_TRIANGLE_BATCH_SIZE,
) -> tuple[ReferenceCache, dict[str, object]]:
    """Build transient exact reference arrays and return compact statistics."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    n_triangles = triangle_count(coordinates.shape[0])
    if n_triangles == 0:
        raise ValueError("At least three C-alpha residues are required")
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
        raise RuntimeError(f"Enumerated {cursor} triangles, expected {n_triangles}")
    for values in raw.values():
        values.flush()

    calculated = {
        "rms_tertile_1": float(np.quantile(raw["rms"], 1.0 / 3.0)),
        "rms_tertile_2": float(np.quantile(raw["rms"], 2.0 / 3.0)),
        "area_tertile_1": float(np.quantile(raw["area"], 1.0 / 3.0)),
        "area_tertile_2": float(np.quantile(raw["area"], 2.0 / 3.0)),
        "slack_tertile_1": float(np.quantile(raw["slack"], 1.0 / 3.0)),
        "slack_tertile_2": float(np.quantile(raw["slack"], 2.0 / 3.0)),
    }
    if thresholds is not None:
        for key, observed in calculated.items():
            expected = float(thresholds[key])
            if not math.isclose(observed, expected, rel_tol=1e-12, abs_tol=1e-12):
                raise ValueError(f"Frozen reference threshold mismatch for {key}: {observed} != {expected}")
    selected = calculated if thresholds is None else {key: float(thresholds[key]) for key in calculated}

    codes = {
        system: _open_memmap(directory / f"code_{system}.npy", "uint8", n_triangles)
        for system in CATEGORY_LABELS
    }
    source_geometry = {name: raw[name] for name in raw}
    all_codes = category_codes(source_geometry, selected)
    for system, values in all_codes.items():
        codes[system][:] = values
        codes[system].flush()

    counts: dict[str, int] = {}
    prefixes = {
        "rms_size": "rms", "angle_shape": "", "angle_shape_exact90": "exact90",
        "relative_slack": "slack", "area": "area",
    }
    for system, prefix in prefixes.items():
        for code, label in enumerate(CATEGORY_LABELS[system]):
            key_label = label
            if system == "angle_shape_exact90" and label == "right_within_tolerance":
                key_label = "right_within_tolerance"
            key = f"n_{prefix + '_' if prefix else ''}{key_label}"
            counts[key] = int(np.count_nonzero(codes[system] == code))

    stats: dict[str, object] = {
        "n_triangles": n_triangles,
        **selected,
        **counts,
        "rms_median": float(np.quantile(raw["rms"], 0.5)),
        "rms_p90": float(np.quantile(raw["rms"], 0.9)),
        "area_median": float(np.quantile(raw["area"], 0.5)),
        "area_p90": float(np.quantile(raw["area"], 0.9)),
        "slack_median": float(np.quantile(raw["slack"], 0.5)),
        "slack_p90": float(np.quantile(raw["slack"], 0.9)),
        "angle_sign_relative_tolerance": ANGLE_SIGN_RELATIVE_TOLERANCE,
    }
    digest_payload = {
        key: stats[key]
        for key in (
            "n_triangles", "rms_tertile_1", "rms_tertile_2", "area_tertile_1",
            "area_tertile_2", "slack_tertile_1", "slack_tertile_2",
            "angle_sign_relative_tolerance",
        )
    }
    stats["reference_digest"] = hashlib.sha256(
        json.dumps(digest_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    for name, values in raw.items():
        if name != "rms":
            mmap = getattr(values, "_mmap", None)
            if mmap is not None:
                mmap.close()
            try:
                (directory / f"{name}.npy").unlink()
            except FileNotFoundError:
                pass
    return ReferenceCache(directory, n_triangles, raw["rms"], codes), stats


class _PrimitiveAccumulator:
    def __init__(self, n_categories: int) -> None:
        self.n = np.zeros(n_categories, dtype=np.int64)
        self.n_violations = np.zeros(n_categories, dtype=np.int64)
        self.abs_sum = np.zeros(n_categories, dtype=np.float64)
        self.abs_max = np.zeros(n_categories, dtype=np.float64)
        self.rel_sum = np.zeros(n_categories, dtype=np.float64)
        self.rel_max = np.zeros(n_categories, dtype=np.float64)
        self.abs_bands = np.zeros((n_categories, 3), dtype=np.int64)
        self.rel_bands = np.zeros((n_categories, 3), dtype=np.int64)

    def add(self, codes: np.ndarray, delta: np.ndarray, relative: np.ndarray) -> None:
        for code in range(self.n.size):
            mask = codes == code
            self.n[code] += int(np.count_nonzero(mask))
            if not np.any(mask):
                continue
            d = delta[mask]
            r = relative[mask]
            violation = d > 0.0
            dv, rv = d[violation], r[violation]
            self.n_violations[code] += dv.size
            self.abs_sum[code] += float(np.sum(dv, dtype=np.float64))
            self.rel_sum[code] += float(np.sum(rv, dtype=np.float64))
            if dv.size:
                self.abs_max[code] = max(self.abs_max[code], float(np.max(dv)))
                self.rel_max[code] = max(self.rel_max[code], float(np.max(rv)))
                self.abs_bands[code] += (
                    np.count_nonzero(dv < 5.0),
                    np.count_nonzero((dv >= 5.0) & (dv < 10.0)),
                    np.count_nonzero(dv >= 10.0),
                )
                self.rel_bands[code] += (
                    np.count_nonzero(rv < 0.05),
                    np.count_nonzero((rv >= 0.05) & (rv < 0.10)),
                    np.count_nonzero(rv >= 0.10),
                )


def _safe_divide(numerator: float, denominator: int) -> float:
    return float(numerator / denominator) if denominator else float("nan")


def compute_metric_rows(
    predicted_distances: np.ndarray,
    reference_cache: ReferenceCache,
    identity: Mapping[str, object],
    *,
    batch_size: int = DEFAULT_TRIANGLE_BATCH_SIZE,
) -> list[dict[str, object]]:
    """Reduce one decoded distance matrix to every frozen subgroup row."""
    predicted = np.asarray(predicted_distances, dtype=np.float64)
    if predicted.ndim != 2 or predicted.shape[0] != predicted.shape[1]:
        raise ValueError("Predicted distances must be a square matrix")
    if not np.all(np.isfinite(predicted)):
        raise ValueError("Predicted distances contain nonfinite values")
    lower = np.tril_indices(predicted.shape[0], k=-1)
    n_edges = lower[0].size
    n_negative = int(np.count_nonzero(predicted[lower] < 0.0))
    clipped = np.maximum(predicted, 0.0)
    accumulators = {
        system: _PrimitiveAccumulator(len(labels))
        for system, labels in CATEGORY_LABELS.items()
    }
    descriptor, rel_path_text = tempfile.mkstemp(
        prefix="triangle_relative_", suffix=".npy", dir=reference_cache.directory,
    )
    os.close(descriptor)
    rel_path = Path(rel_path_text)
    rel_values = np.lib.format.open_memmap(
        rel_path, mode="w+", dtype="float64", shape=(reference_cache.n_triangles,),
    )
    cursor = 0
    try:
        for tri_i, tri_j, tri_k in iter_triangle_indices(predicted.shape[0], batch_size):
            sides = np.stack(
                (clipped[tri_i, tri_j], clipped[tri_i, tri_k], clipped[tri_j, tri_k]),
                axis=1,
            )
            sides.sort(axis=1)
            raw_delta = sides[:, 2] - sides[:, 0] - sides[:, 1]
            delta = np.where(raw_delta > TRIANGLE_TOLERANCE_ANGSTROM, raw_delta, 0.0)
            stop = cursor + tri_i.size
            relative = delta / reference_cache.rms[cursor:stop]
            rel_values[cursor:stop] = relative
            for system, accumulator in accumulators.items():
                accumulator.add(reference_cache.codes[system][cursor:stop], delta, relative)
            cursor = stop
        if cursor != reference_cache.n_triangles:
            raise RuntimeError("Prediction and reference triangle enumeration differ")
        rel_values.flush()

        rows: list[dict[str, object]] = []
        for system, labels in CATEGORY_LABELS.items():
            accumulator = accumulators[system]
            codes = reference_cache.codes[system]
            for code, label in enumerate(labels):
                n = int(accumulator.n[code])
                nv = int(accumulator.n_violations[code])
                if n == 0:
                    support_status = "zero_support"
                elif nv == 0:
                    support_status = "zero_violations"
                else:
                    support_status = "complete"
                conditional = np.asarray(rel_values)[(codes == code) & (np.asarray(rel_values) > 0.0)]
                if conditional.size:
                    p50, p90, p99 = np.quantile(conditional, (0.5, 0.9, 0.99))
                else:
                    p50 = p90 = p99 = float("nan")
                row = {
                    **identity,
                    "evaluation_scope": EVALUATION_SCOPE,
                    "category_system": system,
                    "category_label": label,
                    "n_triangles": n,
                    "n_violations": nv,
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
                    "support_status": support_status,
                }
                missing = set(PER_PROTEIN_COLUMNS) - set(row)
                extra = set(row) - set(PER_PROTEIN_COLUMNS)
                if missing or extra:
                    raise ValueError(f"Metric identity/schema mismatch: missing={missing}, extra={extra}")
                if sum(row[name] for name in ("n_abs_0_5", "n_abs_5_10", "n_abs_10_plus")) != nv:
                    raise RuntimeError("Absolute severity bands do not partition violations")
                if sum(row[name] for name in ("n_rel_0_0p05", "n_rel_0p05_0p10", "n_rel_0p10_plus")) != nv:
                    raise RuntimeError("Relative severity bands do not partition violations")
                if n and nv:
                    expected = row["violation_fraction"] * row["mean_violation_conditional"]
                    if not math.isclose(row["mean_violation_all"], expected, rel_tol=1e-12, abs_tol=1e-12):
                        raise RuntimeError("Absolute primitive identity failed")
                    expected_rel = row["violation_fraction"] * row["mean_relative_violation_conditional"]
                    if not math.isclose(row["mean_relative_violation_all"], expected_rel, rel_tol=1e-12, abs_tol=1e-12):
                        raise RuntimeError("Relative primitive identity failed")
                rows.append(row)
        return rows
    finally:
        mmap = getattr(rel_values, "_mmap", None)
        if mmap is not None:
            mmap.close()
        try:
            rel_path.unlink()
        except FileNotFoundError:
            pass


def metric_value(row: Mapping[str, object], metric: str) -> float:
    if metric in row:
        return float(row[metric])
    n = int(row["n_triangles"])
    numerator_columns = {
        "abs_0_5_fraction_all": "n_abs_0_5",
        "abs_5_10_fraction_all": "n_abs_5_10",
        "abs_10_plus_fraction_all": "n_abs_10_plus",
        "rel_0_0p05_fraction_all": "n_rel_0_0p05",
        "rel_0p05_0p10_fraction_all": "n_rel_0p05_0p10",
        "rel_0p10_plus_fraction_all": "n_rel_0p10_plus",
    }
    if metric in numerator_columns:
        return _safe_divide(float(row[numerator_columns[metric]]), n)
    raise KeyError(metric)


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


def validate_producer_manifest(
    root: str | Path,
    *,
    analysis_name: str,
    model_family: str,
    master_digest: str,
    layers: Sequence[int] = tuple(range(48)),
) -> dict[str, object]:
    root = Path(root)
    manifest = load_json(root / "run_manifest.json")
    expected_layers = list(map(int, layers))
    manifest_digest = manifest.get("master_manifest_digest", manifest.get("master_split_digest"))
    if manifest.get("analysis_name") != analysis_name or manifest.get("model_family") != model_family:
        raise ValueError(f"Producer identity mismatch in {root}")
    if manifest.get("run_status") != "complete":
        raise ValueError(f"Producer run is not complete: {root}")
    if manifest_digest != master_digest:
        raise ValueError(f"Producer master split digest mismatch in {root}")
    if manifest.get("requested_layers") != expected_layers or manifest.get("completed_layers") != expected_layers:
        raise ValueError(f"Producer layer coverage mismatch in {root}")
    feature_names = manifest.get("feature_names")
    if not isinstance(feature_names, list) or len(feature_names) != 256:
        raise ValueError(f"Producer feature order is absent or malformed in {root}")
    expected_names = [f"pair_ij_ch_{index:03d}" for index in range(128)] + [
        f"pair_ji_ch_{index:03d}" for index in range(128)
    ]
    if feature_names != expected_names:
        raise ValueError(f"Producer feature order mismatch in {root}")
    return manifest


def resolve_result_root(path: str | Path) -> Path:
    """Resolve a family root, tolerating one redundant copied directory level."""
    root = Path(path).expanduser().resolve()
    if (root / "run_manifest.json").is_file():
        return root
    matches = sorted(root.glob("*/run_manifest.json")) if root.is_dir() else []
    if len(matches) == 1:
        return matches[0].parent
    raise FileNotFoundError(f"Could not uniquely resolve result root beneath {root}")


__all__ = [name for name in globals() if name.isupper()] + [
    "ReferenceCache", "build_reference_cache", "category_codes",
    "compute_metric_rows", "iter_triangle_indices", "load_json", "metric_value",
    "resolve_result_root", "sha256_file", "tertile_codes", "triangle_count",
    "triangle_geometry", "validate_producer_manifest",
]

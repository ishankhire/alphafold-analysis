#!/usr/bin/env python3
"""Canonical output schemas and evaluation helpers for regression probes.

The helpers in this module consume predictions. They do not fit models, choose
hyperparameters, or calculate numerical-rank diagnostics.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import tempfile
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Mapping, Sequence

import numpy as np

from cluster_regression_helpers import analyze_protein_geometry, TMScoreError
from regression_protocol import (
    FEATURE_NAMES,
    FLOAT_EPS,
    LAYERS,
    MASTER_MANIFEST_DIGEST,
    MASTER_MANIFEST_VERSION,
    HISTORICAL_PROTOCOL_VERSION as PROTOCOL_VERSION,
    canonical_json_bytes,
)


# The existing helpers and runners remain the historical additive-v1 contract.
# Fivefold v2 builders live separately under cross_validation/ so importing an
# existing runner cannot silently change its output schema.
OUTPUT_SCHEMA_VERSION = "1.0"
FIVEFOLD_OUTPUT_SCHEMA_VERSION = "2.0"

RUN_MANIFEST_FILENAME = "run_manifest.json"
PER_PROTEIN_FILENAME = "per_protein.csv"
POOLED_METRICS_FILENAME = "pooled_metrics.csv"
MODEL_STATUS_FILENAME = "model_status.csv"
RUNTIME_FILENAME = "runtime.csv"
GEOMETRY_METRICS_FILENAME = "geometry_metrics.csv"
GEOMETRY_SUMMARY_FILENAME = "geometry_summary.csv"
SUBGROUP_SEQUENCE_FILENAME = "subgroup_metrics/by_sequence_separation.csv"
SUBGROUP_AMINO_ACID_FILENAME = "subgroup_metrics/by_amino_acid_pair.csv"
SUBGROUP_SPATIAL_FILENAME = "subgroup_metrics/by_spatial_distance.csv"

PER_PROTEIN_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_id", "source",
    "protein_split", "evaluation_scope", "prediction_status", "n_residues",
    "n_protein_pairs", "n_expected_pairs", "n_evaluated_pairs", "pair_coverage",
    "target_variance", "r2", "mae", "rmse", "bias_pred_minus_true",
    "mean_true_distance", "mean_predicted_distance",
)

POOLED_METRICS_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_split",
    "evaluation_scope", "n_proteins", "n_proteins_with_finite_r2",
    "n_expected_pairs", "n_evaluated_pairs", "pair_coverage", "pooled_r2",
    "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true",
    "global_baseline_type", "global_baseline_pooled_r2", "global_baseline_pooled_mae",
    "global_baseline_pooled_rmse", "global_baseline_pooled_bias_pred_minus_true",
    "conditional_baseline_type", "conditional_baseline_pooled_r2",
    "conditional_baseline_pooled_mae", "conditional_baseline_pooled_rmse",
    "conditional_baseline_pooled_bias_pred_minus_true",
)

SUBGROUP_METRICS_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_split",
    "evaluation_scope", "category_scope", "category_label", "n_proteins",
    "n_expected_pairs", "n_evaluated_pairs", "pair_coverage", "support_status",
    "pooled_r2", "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true",
)

GEOMETRY_METRIC_NAMES = (
    "rmsd", "tm_score", "frac_top3_eigenvalues", "n_negative_eigenvalues",
    "negative_eigenvalue_fraction", "stress", "n_negative_predictions", "n_triangles",
    "triangle_violation_fraction", "triangle_mean_violation", "triangle_max_violation",
    "triangle_violation_0_5_fraction", "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
)
GEOMETRY_METRIC_UNITS = MappingProxyType({
    "rmsd": "angstrom", "tm_score": "fraction", "frac_top3_eigenvalues": "fraction",
    "n_negative_eigenvalues": "count", "negative_eigenvalue_fraction": "fraction",
    "stress": "fraction", "n_negative_predictions": "count", "n_triangles": "count",
    "triangle_violation_fraction": "fraction", "triangle_mean_violation": "angstrom",
    "triangle_max_violation": "angstrom", "triangle_violation_0_5_fraction": "fraction",
    "triangle_violation_5_10_fraction": "fraction",
    "triangle_violation_10_plus_fraction": "fraction",
})
GEOMETRY_METRICS_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_id", "source",
    "protein_split", "evaluation_scope", "n_residues", "n_protein_pairs",
    "pair_coverage", "geometry_status", "geometry_exclusion_reason",
    *GEOMETRY_METRIC_NAMES,
)
GEOMETRY_SUMMARY_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_split",
    "evaluation_scope", "geometry_metric", "units", "n_proteins_attempted",
    "n_proteins_included", "n_proteins_excluded", "mean", "std", "median", "q25", "q75",
)

MODEL_STATUS_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_id",
    "gate_aa_pair", "gate_sequence_bin", "fit_status", "support_flags",
    "n_training_proteins", "n_training_pairs", "design_rank", "largest_singular_value",
    "smallest_singular_value", "condition_number", "selected_configuration_id",
    "final_boosting_rounds", "training_row_cap", "random_seed",
)
RUNTIME_COLUMNS = (
    "analysis_name", "model_family", "layer", "completion_status", "n_fitted_proteins",
    "n_evaluated_proteins", "n_fitted_pairs", "n_evaluated_pairs", "elapsed_seconds",
    "peak_rss_mb",
)
COEFFICIENT_COLUMNS = (
    "analysis_name", "model_family", "layer", "protein_id", "gate_aa_pair",
    "gate_sequence_bin", "feature_index", "feature_name", "coefficient_standardized",
    "coefficient_raw_scale",
)
COEFFICIENT_INDEX_COLUMNS = (
    "analysis_name", "model_family", "layer", "protein_id", "gate_aa_pair",
    "gate_sequence_bin", "shard_path", "array_row", "fit_status", "feature_dim",
    "file_sha256",
)
FEATURE_IMPORTANCE_COLUMNS = (
    "analysis_name", "model_family", "layer", "feature_index", "feature_name",
    "gain", "split_count",
)

CSV_SCHEMAS = MappingProxyType({
    PER_PROTEIN_FILENAME: PER_PROTEIN_COLUMNS,
    POOLED_METRICS_FILENAME: POOLED_METRICS_COLUMNS,
    MODEL_STATUS_FILENAME: MODEL_STATUS_COLUMNS,
    RUNTIME_FILENAME: RUNTIME_COLUMNS,
    GEOMETRY_METRICS_FILENAME: GEOMETRY_METRICS_COLUMNS,
    GEOMETRY_SUMMARY_FILENAME: GEOMETRY_SUMMARY_COLUMNS,
    SUBGROUP_SEQUENCE_FILENAME: SUBGROUP_METRICS_COLUMNS,
    SUBGROUP_AMINO_ACID_FILENAME: SUBGROUP_METRICS_COLUMNS,
    SUBGROUP_SPATIAL_FILENAME: SUBGROUP_METRICS_COLUMNS,
    "model_artifacts/coefficients.csv": COEFFICIENT_COLUMNS,
    "model_artifacts/coefficient_index.csv": COEFFICIENT_INDEX_COLUMNS,
    "model_artifacts/feature_importance.csv": FEATURE_IMPORTANCE_COLUMNS,
})


def semantic_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _as_float_vector(values: np.ndarray | Sequence[float], name: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{name} must be a one-dimensional vector")
    return array


@dataclass
class RegressionStats:
    """Mergeable sufficient statistics for exact pair-pooled regression metrics."""

    n: int = 0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    sum_prediction: float = 0.0
    sum_error: float = 0.0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0

    def update(self, y_true: np.ndarray, predictions: np.ndarray) -> None:
        y = _as_float_vector(y_true, "y_true")
        pred = _as_float_vector(predictions, "predictions")
        if y.shape != pred.shape:
            raise ValueError("Targets and predictions must have identical shape")
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
            raise ValueError("RegressionStats accepts only finite evaluated rows")
        if y.size == 0:
            return
        error = pred - y
        self.n += int(y.size)
        self.sum_y += float(np.sum(y, dtype=np.float64))
        self.sum_y2 += float(np.sum(y * y, dtype=np.float64))
        self.sum_prediction += float(np.sum(pred, dtype=np.float64))
        self.sum_error += float(np.sum(error, dtype=np.float64))
        self.sum_abs_error += float(np.sum(np.abs(error), dtype=np.float64))
        self.sum_squared_error += float(np.sum(error * error, dtype=np.float64))

    def merge(self, other: "RegressionStats") -> "RegressionStats":
        for name in (
            "n", "sum_y", "sum_y2", "sum_prediction", "sum_error",
            "sum_abs_error", "sum_squared_error",
        ):
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    def metrics(self) -> dict[str, float]:
        if self.n == 0:
            return {
                "target_variance": float("nan"), "r2": float("nan"),
                "mae": float("nan"), "rmse": float("nan"),
                "bias_pred_minus_true": float("nan"), "mean_true_distance": float("nan"),
                "mean_predicted_distance": float("nan"),
            }
        mean_y = self.sum_y / self.n
        variance = max(0.0, self.sum_y2 / self.n - mean_y * mean_y)
        total_sum_squares = self.sum_y2 - self.sum_y * self.sum_y / self.n
        mse = self.sum_squared_error / self.n
        return {
            "target_variance": variance,
            "r2": float("nan") if total_sum_squares <= FLOAT_EPS else 1.0 - self.sum_squared_error / total_sum_squares,
            "mae": self.sum_abs_error / self.n,
            "rmse": math.sqrt(mse),
            "bias_pred_minus_true": self.sum_error / self.n,
            "mean_true_distance": mean_y,
            "mean_predicted_distance": self.sum_prediction / self.n,
        }


@dataclass
class PooledRegressionAccumulator:
    """Track pooled metrics plus protein and pair coverage."""

    stats: RegressionStats = field(default_factory=RegressionStats)
    n_expected_pairs: int = 0
    expected_protein_ids: set[str] = field(default_factory=set)
    evaluated_protein_ids: set[str] = field(default_factory=set)
    per_protein_stats: dict[str, RegressionStats] = field(default_factory=dict)

    def update_protein(
        self,
        protein_id: str,
        y_true: np.ndarray,
        predictions: np.ndarray,
        *,
        expected_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        y = _as_float_vector(y_true, "y_true")
        pred = _as_float_vector(predictions, "predictions")
        if y.shape != pred.shape:
            raise ValueError("Targets and predictions must have identical shape")
        if not np.all(np.isfinite(y)):
            raise ValueError("Canonical regression targets must be finite")
        if expected_mask is None:
            expected = np.ones(y.size, dtype=bool)
        else:
            expected = np.asarray(expected_mask, dtype=bool)
            if expected.shape != y.shape:
                raise ValueError("Expected-pair mask must match target shape")
        n_expected = int(np.count_nonzero(expected))
        self.n_expected_pairs += n_expected
        if n_expected:
            self.expected_protein_ids.add(str(protein_id))
        evaluated = expected & np.isfinite(pred)
        if np.any(evaluated):
            protein_key = str(protein_id)
            self.evaluated_protein_ids.add(protein_key)
            self.stats.update(y[evaluated], pred[evaluated])
            local = RegressionStats()
            local.update(y[evaluated], pred[evaluated])
            self.per_protein_stats.setdefault(protein_key, RegressionStats()).merge(local)
        return evaluated

    def merge(self, other: "PooledRegressionAccumulator") -> "PooledRegressionAccumulator":
        self.stats.merge(other.stats)
        self.n_expected_pairs += other.n_expected_pairs
        self.expected_protein_ids.update(other.expected_protein_ids)
        self.evaluated_protein_ids.update(other.evaluated_protein_ids)
        for protein_id, stats in other.per_protein_stats.items():
            self.per_protein_stats.setdefault(protein_id, RegressionStats()).merge(stats)
        return self

    @property
    def n_evaluated_pairs(self) -> int:
        return self.stats.n

    @property
    def pair_coverage(self) -> float:
        return (
            float("nan") if self.n_expected_pairs == 0
            else self.n_evaluated_pairs / self.n_expected_pairs
        )

    @property
    def n_proteins_with_finite_r2(self) -> int:
        return int(sum(
            np.isfinite(stats.metrics()["r2"])
            for stats in self.per_protein_stats.values()
        ))


@dataclass
class PooledComparisonAccumulator:
    """Evaluate model and baselines over exactly the model's finite prediction rows."""

    model: PooledRegressionAccumulator = field(default_factory=PooledRegressionAccumulator)
    global_baseline: RegressionStats = field(default_factory=RegressionStats)
    conditional_baseline: RegressionStats | None = None

    def update_protein(
        self,
        protein_id: str,
        y_true: np.ndarray,
        predictions: np.ndarray,
        global_baseline_predictions: np.ndarray | float,
        *,
        conditional_baseline_predictions: np.ndarray | float | None = None,
        expected_mask: np.ndarray | None = None,
    ) -> np.ndarray:
        y = _as_float_vector(y_true, "y_true")
        evaluated = self.model.update_protein(
            protein_id, y, predictions, expected_mask=expected_mask,
        )
        global_pred = np.broadcast_to(
            np.asarray(global_baseline_predictions, dtype=np.float64), y.shape,
        )
        if np.any(evaluated):
            if not np.all(np.isfinite(global_pred[evaluated])):
                raise ValueError("Global baseline is nonfinite on evaluated model rows")
            self.global_baseline.update(y[evaluated], global_pred[evaluated])
        if conditional_baseline_predictions is not None:
            if self.conditional_baseline is None:
                self.conditional_baseline = RegressionStats()
            conditional_pred = np.broadcast_to(
                np.asarray(conditional_baseline_predictions, dtype=np.float64), y.shape,
            )
            if np.any(evaluated):
                if not np.all(np.isfinite(conditional_pred[evaluated])):
                    raise ValueError("Conditional baseline is nonfinite on evaluated model rows")
                self.conditional_baseline.update(y[evaluated], conditional_pred[evaluated])
        elif self.conditional_baseline is not None:
            raise ValueError("Conditional baseline must be supplied consistently")
        return evaluated

    def merge(self, other: "PooledComparisonAccumulator") -> "PooledComparisonAccumulator":
        self.model.merge(other.model)
        self.global_baseline.merge(other.global_baseline)
        if (self.conditional_baseline is None) != (other.conditional_baseline is None):
            raise ValueError("Cannot merge applicable and non-applicable conditional baselines")
        if self.conditional_baseline is not None:
            self.conditional_baseline.merge(other.conditional_baseline)
        return self


def build_per_protein_row(
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    protein_id: str,
    source: str,
    protein_split: str,
    evaluation_scope: str,
    n_residues: int,
    y_true: np.ndarray,
    predictions: np.ndarray,
    prediction_status: str | None = None,
) -> dict[str, object]:
    y = _as_float_vector(y_true, "y_true")
    pred = _as_float_vector(predictions, "predictions")
    if y.shape != pred.shape or not np.all(np.isfinite(y)):
        raise ValueError("Per-protein targets must be finite and match predictions")
    evaluated = np.isfinite(pred)
    stats = RegressionStats()
    stats.update(y[evaluated], pred[evaluated])
    metrics = stats.metrics()
    n_expected = int(y.size)
    n_evaluated = int(np.count_nonzero(evaluated))
    coverage = float("nan") if n_expected == 0 else n_evaluated / n_expected
    automatic_status = (
        "complete" if n_evaluated == n_expected
        else "no_predictions" if n_evaluated == 0
        else "partial_coverage"
    )
    status = prediction_status or automatic_status
    if status == "complete" and n_evaluated != n_expected:
        raise ValueError("prediction_status=complete requires complete finite coverage")
    return {
        "analysis_name": analysis_name, "model_family": model_family,
        "feature_dim": int(feature_dim), "layer": int(layer), "protein_id": str(protein_id),
        "source": source, "protein_split": protein_split, "evaluation_scope": evaluation_scope,
        "prediction_status": status, "n_residues": int(n_residues),
        "n_protein_pairs": int(n_residues) * (int(n_residues) - 1) // 2,
        "n_expected_pairs": n_expected, "n_evaluated_pairs": n_evaluated,
        "pair_coverage": coverage, "target_variance": metrics["target_variance"],
        "r2": metrics["r2"], "mae": metrics["mae"], "rmse": metrics["rmse"],
        "bias_pred_minus_true": metrics["bias_pred_minus_true"],
        "mean_true_distance": metrics["mean_true_distance"],
        "mean_predicted_distance": metrics["mean_predicted_distance"],
    }


def _prefixed_metrics(metrics: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}r2": metrics["r2"], f"{prefix}mae": metrics["mae"],
        f"{prefix}rmse": metrics["rmse"],
        f"{prefix}bias_pred_minus_true": metrics["bias_pred_minus_true"],
    }


def build_pooled_metrics_row(
    accumulator: PooledComparisonAccumulator,
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    protein_split: str,
    evaluation_scope: str,
    global_baseline_type: str = "global_development_mean",
    conditional_baseline_type: str = "not_applicable",
) -> dict[str, object]:
    model_metrics = accumulator.model.stats.metrics()
    global_metrics = accumulator.global_baseline.metrics()
    if accumulator.global_baseline.n != accumulator.model.n_evaluated_pairs:
        raise ValueError("Global baseline and model were not evaluated over identical rows")
    if accumulator.conditional_baseline is None:
        if conditional_baseline_type != "not_applicable":
            raise ValueError("A conditional baseline type was supplied without predictions")
        conditional_metrics = RegressionStats().metrics()
    else:
        if conditional_baseline_type == "not_applicable":
            raise ValueError("Conditional predictions require an explicit baseline type")
        if accumulator.conditional_baseline.n != accumulator.model.n_evaluated_pairs:
            raise ValueError("Conditional baseline and model were not evaluated over identical rows")
        conditional_metrics = accumulator.conditional_baseline.metrics()
    return {
        "analysis_name": analysis_name, "model_family": model_family,
        "feature_dim": int(feature_dim), "layer": int(layer), "protein_split": protein_split,
        "evaluation_scope": evaluation_scope,
        "n_proteins": len(accumulator.model.evaluated_protein_ids),
        "n_proteins_with_finite_r2": accumulator.model.n_proteins_with_finite_r2,
        "n_expected_pairs": accumulator.model.n_expected_pairs,
        "n_evaluated_pairs": accumulator.model.n_evaluated_pairs,
        "pair_coverage": accumulator.model.pair_coverage,
        **_prefixed_metrics(model_metrics, "pooled_"),
        "global_baseline_type": global_baseline_type,
        **_prefixed_metrics(global_metrics, "global_baseline_pooled_"),
        "conditional_baseline_type": conditional_baseline_type,
        **_prefixed_metrics(conditional_metrics, "conditional_baseline_pooled_"),
    }


@dataclass
class SubgroupAccumulator:
    category_scope: str
    category_labels: tuple[str, ...]
    cells: list[PooledRegressionAccumulator] = field(init=False)

    def __post_init__(self) -> None:
        if not self.category_labels or len(set(self.category_labels)) != len(self.category_labels):
            raise ValueError("Subgroup labels must be nonempty and unique")
        self.cells = [PooledRegressionAccumulator() for _ in self.category_labels]

    def update_protein(
        self,
        protein_id: str,
        y_true: np.ndarray,
        predictions: np.ndarray,
        category_codes: np.ndarray,
    ) -> None:
        y = _as_float_vector(y_true, "y_true")
        pred = _as_float_vector(predictions, "predictions")
        codes = np.asarray(category_codes)
        if y.shape != pred.shape or y.shape != codes.shape:
            raise ValueError("Targets, predictions, and subgroup codes must match")
        if not np.issubdtype(codes.dtype, np.integer):
            raise ValueError("Subgroup codes must use an integer dtype")
        if not np.all(np.isfinite(y)):
            raise ValueError("Subgroup targets must be finite")
        invalid = (codes < -1) | (codes >= len(self.category_labels))
        if np.any(invalid):
            raise ValueError("Subgroup codes must be -1 or a valid category index")
        for code, cell in enumerate(self.cells):
            mask = codes == code
            if np.any(mask):
                cell.update_protein(protein_id, y[mask], pred[mask])

    @staticmethod
    def _support_status(n_pairs: int, n_proteins: int) -> str:
        if n_pairs == 0:
            return "empty"
        low_pairs = n_pairs < 100
        low_proteins = n_proteins < 5
        if low_pairs and low_proteins:
            return "low_pair_and_protein_support"
        if low_pairs:
            return "low_pair_support"
        if low_proteins:
            return "low_protein_support"
        return "sufficient"

    def rows(
        self,
        *,
        analysis_name: str,
        model_family: str,
        feature_dim: int,
        layer: int,
        protein_split: str,
        evaluation_scope: str,
    ) -> list[dict[str, object]]:
        rows = []
        for label, cell in zip(self.category_labels, self.cells):
            metrics = cell.stats.metrics()
            n_proteins = len(cell.evaluated_protein_ids)
            rows.append({
                "analysis_name": analysis_name, "model_family": model_family,
                "feature_dim": int(feature_dim), "layer": int(layer),
                "protein_split": protein_split, "evaluation_scope": evaluation_scope,
                "category_scope": self.category_scope, "category_label": label,
                "n_proteins": n_proteins, "n_expected_pairs": cell.n_expected_pairs,
                "n_evaluated_pairs": cell.n_evaluated_pairs, "pair_coverage": cell.pair_coverage,
                "support_status": self._support_status(cell.n_evaluated_pairs, n_proteins),
                **_prefixed_metrics(metrics, "pooled_"),
            })
        return rows


def _empty_geometry_metrics() -> dict[str, float]:
    return {metric: float("nan") for metric in GEOMETRY_METRIC_NAMES}


def build_geometry_row(
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    protein_id: str,
    source: str,
    protein_split: str,
    n_residues: int,
    true_coordinates: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)
    pred = _as_float_vector(predictions, "predictions")
    if int(n_residues) < 2:
        raise ValueError("Geometry requires at least two residues")
    expected_pairs = int(n_residues) * (int(n_residues) - 1) // 2
    if pair_i.shape != pair_j.shape or pair_i.shape != pred.shape:
        raise ValueError("Geometry pair indices and predictions must have identical shape")
    if pred.size != expected_pairs:
        raise ValueError(f"Geometry requires all {expected_pairs} unordered pair slots")
    if (
        pair_i.ndim != 1 or np.any(pair_i < 0) or np.any(pair_j < 0)
        or np.any(pair_i >= n_residues) or np.any(pair_j >= n_residues)
        or np.any(pair_i == pair_j)
    ):
        raise ValueError("Geometry pair indices must identify valid off-diagonal residue pairs")
    unordered_codes = (
        np.minimum(pair_i, pair_j) * int(n_residues) + np.maximum(pair_i, pair_j)
    )
    if np.unique(unordered_codes).size != expected_pairs:
        raise ValueError("Geometry pair indices must cover every unordered pair exactly once")
    n_evaluated = int(np.count_nonzero(np.isfinite(pred)))
    coverage = n_evaluated / expected_pairs
    identity = {
        "analysis_name": analysis_name, "model_family": model_family,
        "feature_dim": int(feature_dim), "layer": int(layer), "protein_id": str(protein_id),
        "source": source, "protein_split": protein_split, "evaluation_scope": "all_pairs",
        "n_residues": int(n_residues), "n_protein_pairs": expected_pairs,
        "pair_coverage": coverage,
    }
    if n_evaluated != expected_pairs:
        return {
            **identity, "geometry_status": "incomplete_pair_coverage",
            "geometry_exclusion_reason": f"{n_evaluated}/{expected_pairs} finite pair predictions",
            **_empty_geometry_metrics(),
        }
    try:
        raw = analyze_protein_geometry(
            int(n_residues), np.asarray(true_coordinates, dtype=np.float64), pair_i, pair_j, pred,
        )
        metrics = {
            "rmsd": float(raw["rmsd"]), "tm_score": float(raw["tm_score"]),
            "frac_top3_eigenvalues": float(raw["frac_top3_eigenvalues"]),
            "n_negative_eigenvalues": int(raw["n_negative_eigenvalues"]),
            "negative_eigenvalue_fraction": float(raw["neg_eigenvalue_fraction"]),
            "stress": float(raw["stress"]),
            "n_negative_predictions": int(raw["n_negative_predictions"]),
            "n_triangles": math.comb(int(n_residues), 3),
            "triangle_violation_fraction": float(raw["triangle_violation_fraction"]),
            "triangle_mean_violation": float(raw["triangle_mean_violation"]),
            "triangle_max_violation": float(raw["triangle_max_violation"]),
            "triangle_violation_0_5_fraction": float(raw["triangle_violation_0_5_fraction"]),
            "triangle_violation_5_10_fraction": float(raw["triangle_violation_5_10_fraction"]),
            "triangle_violation_10_plus_fraction": float(raw["triangle_violation_10_plus_fraction"]),
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Geometry calculation returned nonfinite metrics")
        return {
            **identity, "geometry_status": "complete", "geometry_exclusion_reason": "", **metrics,
        }
    except TMScoreError:
        raise
    except Exception as exc:
        return {
            **identity, "geometry_status": "geometry_failed",
            "geometry_exclusion_reason": f"{type(exc).__name__}: {exc}",
            **_empty_geometry_metrics(),
        }


def summarize_geometry_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Build canonical long-format summaries from authoritative geometry rows."""
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    group_fields = (
        "analysis_name", "model_family", "feature_dim", "layer", "protein_split", "evaluation_scope",
    )
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key in sorted(grouped, key=lambda value: tuple(map(str, value))):
        group = grouped[key]
        included = [row for row in group if row["geometry_status"] == "complete"]
        identity = dict(zip(group_fields, key))
        for metric in GEOMETRY_METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in included], dtype=np.float64)
            values = values[np.isfinite(values)]
            if values.size:
                statistics = {
                    "mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
                    "median": float(np.median(values)), "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
            else:
                statistics = {name: float("nan") for name in ("mean", "std", "median", "q25", "q75")}
            summaries.append({
                **identity, "geometry_metric": metric, "units": GEOMETRY_METRIC_UNITS[metric],
                "n_proteins_attempted": len(group), "n_proteins_included": len(included),
                "n_proteins_excluded": len(group) - len(included), **statistics,
            })
    return summaries


def build_model_status_row(
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    fit_status: str,
    support_flags: str = "",
    protein_id: str = "",
    gate_aa_pair: str = "",
    gate_sequence_bin: str = "",
    n_training_proteins: int = 0,
    n_training_pairs: int = 0,
    design_rank: float = float("nan"),
    largest_singular_value: float = float("nan"),
    smallest_singular_value: float = float("nan"),
    condition_number: float = float("nan"),
    selected_configuration_id: str = "",
    final_boosting_rounds: float = float("nan"),
    training_row_cap: float = float("nan"),
    random_seed: float = float("nan"),
) -> dict[str, object]:
    return {
        "analysis_name": analysis_name, "model_family": model_family,
        "feature_dim": int(feature_dim), "layer": int(layer), "protein_id": protein_id,
        "gate_aa_pair": gate_aa_pair, "gate_sequence_bin": gate_sequence_bin,
        "fit_status": fit_status, "support_flags": support_flags,
        "n_training_proteins": int(n_training_proteins),
        "n_training_pairs": int(n_training_pairs), "design_rank": design_rank,
        "largest_singular_value": largest_singular_value,
        "smallest_singular_value": smallest_singular_value,
        "condition_number": condition_number,
        "selected_configuration_id": selected_configuration_id,
        "final_boosting_rounds": final_boosting_rounds,
        "training_row_cap": training_row_cap, "random_seed": random_seed,
    }


def build_runtime_row(
    *,
    analysis_name: str,
    model_family: str,
    layer: int,
    completion_status: str,
    n_fitted_proteins: int,
    n_evaluated_proteins: int,
    n_fitted_pairs: int,
    n_evaluated_pairs: int,
    elapsed_seconds: float,
    peak_rss_mb: float = float("nan"),
) -> dict[str, object]:
    if elapsed_seconds < 0:
        raise ValueError("Elapsed time cannot be negative")
    return {
        "analysis_name": analysis_name, "model_family": model_family, "layer": int(layer),
        "completion_status": completion_status, "n_fitted_proteins": int(n_fitted_proteins),
        "n_evaluated_proteins": int(n_evaluated_proteins), "n_fitted_pairs": int(n_fitted_pairs),
        "n_evaluated_pairs": int(n_evaluated_pairs), "elapsed_seconds": float(elapsed_seconds),
        "peak_rss_mb": peak_rss_mb,
    }


def build_run_manifest(
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    requested_layers: Sequence[int],
    completed_layers: Sequence[int],
    split_digest: str = MASTER_MANIFEST_DIGEST,
    configuration: Mapping[str, object] | None = None,
    versions: Mapping[str, str] | None = None,
    code_revision: str = "",
    run_status: str = "complete",
    created_at: str | None = None,
) -> dict[str, object]:
    config = dict(configuration or {})
    requested = sorted({int(layer) for layer in requested_layers})
    completed = sorted({int(layer) for layer in completed_layers})
    if not requested or any(layer not in LAYERS for layer in requested):
        raise ValueError("Requested layers must be a nonempty subset of canonical layers 0-47")
    if not set(completed).issubset(requested):
        raise ValueError("Completed layers must be a subset of requested layers")
    if feature_dim < 1 or feature_dim > len(FEATURE_NAMES):
        raise ValueError(f"feature_dim must be between 1 and {len(FEATURE_NAMES)}")
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "master_manifest_version": MASTER_MANIFEST_VERSION,
        "master_split_digest": split_digest,
        "analysis_name": analysis_name,
        "model_family": model_family,
        "feature_dim": int(feature_dim),
        "feature_names": list(FEATURE_NAMES[: int(feature_dim)]),
        "requested_layers": requested,
        "completed_layers": completed,
        "configuration": config,
        "configuration_digest": semantic_digest(config),
        "versions": dict(versions or {}),
        "code_revision": code_revision,
        "run_status": run_status,
        "created_at": created_at or datetime.now(timezone.utc).isoformat(),
    }


def _csv_value(value: object) -> object:
    if value is None:
        return "NaN"
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        if not math.isfinite(value):
            return "NaN"
        return format(value, ".17g")
    return value


def atomic_write_csv(
    path: str | Path,
    fieldnames: Sequence[str],
    rows: Sequence[Mapping[str, object]],
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    expected = set(fieldnames)
    for index, row in enumerate(rows):
        missing = expected - set(row)
        extra = set(row) - expected
        if missing or extra:
            raise ValueError(
                f"CSV row {index} schema mismatch; missing={sorted(missing)}, extra={sorted(extra)}"
            )
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            for row in rows:
                writer.writerow({field: _csv_value(row[field]) for field in fieldnames})
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_json(path: str | Path, payload: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    encoded = json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n"
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write(encoded)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_savez_compressed(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".npz", dir=destination.parent,
    )
    os.close(descriptor)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


__all__ = [
    "COEFFICIENT_COLUMNS", "COEFFICIENT_INDEX_COLUMNS", "CSV_SCHEMAS",
    "FEATURE_IMPORTANCE_COLUMNS", "GEOMETRY_METRIC_NAMES", "GEOMETRY_METRIC_UNITS",
    "GEOMETRY_METRICS_COLUMNS", "GEOMETRY_SUMMARY_COLUMNS", "MODEL_STATUS_COLUMNS",
    "FIVEFOLD_OUTPUT_SCHEMA_VERSION", "OUTPUT_SCHEMA_VERSION", "PER_PROTEIN_COLUMNS",
    "POOLED_METRICS_COLUMNS",
    "PooledComparisonAccumulator", "PooledRegressionAccumulator", "RUNTIME_COLUMNS",
    "RegressionStats", "SUBGROUP_METRICS_COLUMNS", "SubgroupAccumulator",
    "atomic_savez_compressed", "atomic_write_csv", "atomic_write_json",
    "build_geometry_row", "build_model_status_row", "build_per_protein_row",
    "build_pooled_metrics_row", "build_run_manifest", "build_runtime_row",
    "semantic_digest", "sha256_file", "summarize_geometry_rows",
]

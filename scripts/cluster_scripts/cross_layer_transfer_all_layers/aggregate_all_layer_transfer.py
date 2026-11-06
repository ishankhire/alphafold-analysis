#!/usr/bin/env python3
"""Consolidate all-layer transfer shards into metric and error-distribution tables."""

from __future__ import annotations

import csv
import math
import os
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Iterator, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, CLUSTER_DIR, CLUSTER_DIR / "cross_layer_transfer",
             CLUSTER_DIR / "cross_validation", CLUSTER_DIR / "triangle_violation_subgroups"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_cross_layer_transfer import (  # noqa: E402
    PARITY_COLUMNS, PARITY_TOLERANCE, RUNTIME_OUTPUT_COLUMNS, SOURCE_STATUS_COLUMNS, _compare,
    stats_from_vector, stream_write_csv,
)
from all_layer_transfer_common import (  # noqa: E402
    CANONICAL_DISTANCE_GROUPS, CANONICAL_SEQUENCE_GROUPS, CATEGORY_CELLS, CELL_INDEX, CV_PROTOCOL_VERSION,
    DEFAULT_V1_ROOT, DISTANCE_LABELS, ERROR_EDGES, EVALUATION_DESIGN, EVALUATION_SCOPE, FEATURE_DIM,
    LAYERS, MODEL_FAMILY, N_ERROR_BINS, N_OUTER_FOLDS, N_POSITION_BINS, OOF_PREDICTION_SCOPE,
    OUTLIER_PROTEIN, OUTPUT_ENV, PRODUCER_SPEC, SEQUENCE_LABELS, STAT_FIELDS, TRANSFER_COLUMNS,
    TRANSFER_MODE, V1_LAYERS, V1_ROOT_ENV, TargetAccumulator, analysis_label, analysis_name, cohort_name,
    default_output_root, histogram_quantiles, load_probe_bank, select_cohort, strata_support_status,
    transfer_identity, validate_final_shard,
)
from canonical_storage import assert_writable, csv_root, deployment_revision  # noqa: E402
from cross_validation_common import (  # noqa: E402
    FOLD_METRICS_COLUMNS, FOLD_SUMMARY_COLUMNS, GLOBAL_BASELINE_TYPE, PER_PROTEIN_COLUMNS,
    POOLED_METRICS_COLUMNS, SUBGROUP_METRICS_COLUMNS, OOFComparisonAccumulator, OOFSubgroupAccumulator,
    RegressionSufficientStatistics, build_fold_metrics_row, build_fold_summary_rows,
    build_pooled_metrics_row, load_cv_folds,
)
from regression_output_helpers import atomic_savez_compressed, atomic_write_csv, atomic_write_json, sha256_file  # noqa: E402
from regression_protocol import FLOAT_EPS  # noqa: E402


SUBGROUP_SCOPES = ("sequence_separation", "spatial_distance")
WHOLE = CELL_INDEX[("all_pairs", "all")]
SUM_Y, SUM_Y2, SUM_PREDICTION, SUM_ERROR, SUM_ABS, SUM_SQUARED = range(1, 7)
QUANTILES = (.01, .05, .25, .5, .75, .95, .99)
QUANTILE_COLUMNS = tuple(f"signed_error_q{int(q * 100):02d}" for q in QUANTILES)
WITHIN_ANGSTROMS = (1, 2, 5, 10)
TOP_PROTEINS_PER_CELL = 5
EXCLUDED = OUTLIER_PROTEIN

PER_PROTEIN_OUTPUT_COLUMNS = (*PER_PROTEIN_COLUMNS, *TRANSFER_COLUMNS)
POOLED_OUTPUT_COLUMNS = (*POOLED_METRICS_COLUMNS, *TRANSFER_COLUMNS)
FOLD_OUTPUT_COLUMNS = (*FOLD_METRICS_COLUMNS, *TRANSFER_COLUMNS)
FOLD_SUMMARY_OUTPUT_COLUMNS = (*FOLD_SUMMARY_COLUMNS, *TRANSFER_COLUMNS)
SUBGROUP_OUTPUT_COLUMNS = (*SUBGROUP_METRICS_COLUMNS, *TRANSFER_COLUMNS)
TRANSFER_MATRIX_COLUMNS = (
    *TRANSFER_COLUMNS, "analysis_name", "n_proteins", "n_evaluated_pairs", "pooled_r2",
    "mean_protein_r2", "median_protein_r2", "n_finite_r2", "pooled_mae", "mean_protein_mae",
    "median_protein_mae", "pooled_rmse", "pooled_bias_pred_minus_true",
    *(f"{kind}_{metric}" for metric in ("r2", "mae") for kind in (
        "native_target_pooled", "delta_pooled_vs_native_target",
        "native_source_pooled", "delta_pooled_vs_native_source",
    )),
    "top_protein_id", "top_protein_squared_error_share", f"squared_error_share_{EXCLUDED}",
    f"pooled_r2_excl_{EXCLUDED}", f"pooled_mae_excl_{EXCLUDED}", f"pooled_rmse_excl_{EXCLUDED}",
)
ERROR_QUANTILE_COLUMNS = (
    *TRANSFER_COLUMNS, "category_scope", "category_label", "weighting", "n_pairs",
    *QUANTILE_COLUMNS, "signed_error_iqr", *(f"share_within_{a}A" for a in WITHIN_ANGSTROMS),
    "share_in_open_tail_bins", "cell_error_min", "cell_error_max",
)
ERROR_STRATA_COLUMNS = (
    *TRANSFER_COLUMNS, "sequence_separation_bin", "final_structure_distance_bin",
    "canonical_sequence_bin", "canonical_distance_bin", "n_pairs", "n_proteins", "support_status",
    "sum_signed_error", "sum_abs_error", "sum_squared_error", "mean_signed_error", "sd_signed_error",
    "mae", "rmse", "share_of_squared_error", f"n_pairs_excl_{EXCLUDED}",
    f"sum_signed_error_excl_{EXCLUDED}", f"sum_abs_error_excl_{EXCLUDED}",
    f"sum_squared_error_excl_{EXCLUDED}", f"mean_signed_error_excl_{EXCLUDED}",
    f"share_of_squared_error_excl_{EXCLUDED}",
)
ERROR_POSITION_COLUMNS = (
    *TRANSFER_COLUMNS, "position_bin", "position_start", "position_end", "n_residue_pairs",
    "mean_signed_error", "mae", "rmse", f"n_residue_pairs_excl_{EXCLUDED}",
    f"mean_signed_error_excl_{EXCLUDED}", f"mae_excl_{EXCLUDED}", f"rmse_excl_{EXCLUDED}",
)
PROTEIN_ANOMALY_COLUMNS = (
    *TRANSFER_COLUMNS, "rank", "protein_id", "source", "outer_fold", "n_residues", "n_pairs",
    "squared_error_share", "r2", "mae", "rmse", "mean_signed_error", "mae_robust_z",
)
PER_PROTEIN_METRICS = ("r2", "mae", "rmse", "bias_pred_minus_true", "mean_true_distance",
                       "mean_predicted_distance", "target_variance", "n_evaluated_pairs")
POOLED_CHECK_METRICS = (
    "n_proteins", "n_evaluated_pairs", "pooled_r2", "pooled_mae", "pooled_rmse",
    "pooled_bias_pred_minus_true", "global_baseline_pooled_r2", "global_baseline_pooled_mae",
    "global_baseline_pooled_rmse", "global_baseline_pooled_bias_pred_minus_true",
)
SUBGROUP_CHECK_METRICS = ("n_proteins", "n_evaluated_pairs", "pooled_r2", "pooled_mae",
                          "pooled_rmse", "pooled_bias_pred_minus_true")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def producer_root() -> Path:
    return csv_root() / "regression_5fold_v1"


def output_root(cohort: str) -> Path:
    value = os.environ.get(OUTPUT_ENV)
    return Path(value).expanduser().resolve() if value else default_output_root(cohort)


def v1_root() -> Path:
    value = os.environ.get(V1_ROOT_ENV)
    return Path(value).expanduser().resolve() if value else DEFAULT_V1_ROOT


def shard_path(root: Path, target_layer: int) -> Path:
    return root / "layer_shards" / f"target_layer_{target_layer:02d}.npz"


class Primitives:
    """Per-protein statistics indexed by (target, source, protein, category, statistic)."""

    def __init__(self, protein_ids: Sequence[str], metadata: Mapping[str, Mapping[str, object]],
                 layers: Sequence[int], cohort: str):
        self.layers = tuple(layers)
        self.cohort = cohort
        self.analysis_name, self.analysis_label = analysis_name(cohort), analysis_label(cohort)
        self.protein_ids = tuple(protein_ids)
        self.metadata = metadata
        self.folds = np.asarray([metadata[p]["outer_fold"] for p in self.protein_ids], dtype=np.int64)
        self.n_pairs = np.asarray([
            metadata[p]["n_residues"] * (metadata[p]["n_residues"] - 1) // 2 for p in self.protein_ids
        ], dtype=np.float64)
        n_layers, n_proteins = len(self.layers), len(self.protein_ids)
        self.probe = np.full((n_layers, n_layers, n_proteins, len(CATEGORY_CELLS), len(STAT_FIELDS)), np.nan)
        self.baseline = np.full((n_layers, n_proteins, len(STAT_FIELDS)), np.nan)
        self.error_min = np.full((n_layers, n_layers), np.inf)
        self.error_max = np.full((n_layers, n_layers), -np.inf)

    def add_shard(self, target_position: int, accumulator: TargetAccumulator) -> None:
        arrays = accumulator.arrays
        self.probe[target_position] = arrays["probe"]
        self.baseline[target_position] = arrays["baseline"]
        self.error_min[target_position] = arrays["error_min"]
        self.error_max[target_position] = arrays["error_max"]

    def require_complete(self) -> None:
        if np.isnan(self.probe).any() or np.isnan(self.baseline).any():
            raise ValueError("Primitive arrays are incomplete")
        if not np.array_equal(self.probe[..., WHOLE, 0], np.broadcast_to(self.n_pairs, self.probe.shape[:3])):
            raise ValueError("Whole-protein pair counts disagree with frozen n_residues")
        if not np.array_equal(self.baseline[..., 0], np.broadcast_to(self.n_pairs, self.baseline.shape[:2])):
            raise ValueError("Baseline pair counts disagree with frozen n_residues")


def shard_consistency_checks(
    target_layer: int, accumulator: TargetAccumulator, primitives: Primitives,
) -> list[dict[str, object]]:
    """Histograms and strata must agree with each other and with the exact statistics."""
    a = accumulator.arrays
    checks: list[dict[str, object]] = []

    def exact(check: str, ok: bool, compared: int) -> None:
        checks.append({"check": check, "layer": target_layer, "metric": "counts", "n_compared": compared,
                       "max_abs_diff": 0.0, "tolerance": "exact", "status": "pass" if ok else "fail",
                       "detail": ""})

    joint = a["hist_joint"].astype(np.int64)
    by_source_error = joint.sum(axis=(3, 4))                       # (sources, folds, error)
    protein_hist = a["hist_protein"].astype(np.int64)
    fold_ok = all(
        np.array_equal(by_source_error[:, fold], protein_hist[:, primitives.folds == fold].sum(axis=1))
        for fold in range(N_OUTER_FOLDS)
    )
    exact("joint_histogram_matches_protein_histograms", fold_ok, int(by_source_error.size))
    exact("joint_histogram_counts_match_strata_n",
          bool(np.all(joint.sum(axis=2) == a["strata_n"])), int(a["strata_n"].size))

    stratified = a["strata_stats"].sum(axis=1)                     # (sources, sequence, distance, 3)
    totals = a["strata_n"].sum(axis=0)
    category_n, category_pairs = [], []
    for scope, label in CATEGORY_CELLS:
        if scope == "all_pairs":
            sums, count = stratified.sum(axis=(1, 2)), totals.sum()
        elif scope == "sequence_separation":
            group = list(CANONICAL_SEQUENCE_GROUPS[label])
            sums, count = stratified[:, group].sum(axis=(1, 2)), totals[group].sum()
        else:
            group = list(CANONICAL_DISTANCE_GROUPS[label])
            sums, count = stratified[:, :, group].sum(axis=(1, 2)), totals[:, group].sum()
        expected = a["probe"][:, :, CELL_INDEX[(scope, label)], SUM_ERROR:].sum(axis=1)
        category_n.append(int(count) == int(a["probe"][0, :, CELL_INDEX[(scope, label)], 0].sum()))
        category_pairs.extend(zip(expected.ravel(), sums.ravel()))
    exact("strata_counts_match_categories", all(category_n), len(category_n))
    checks.append(_relative("strata_error_sums_match_categories", target_layer, category_pairs))

    whole = a["probe"][:, :, WHOLE, SUM_ERROR:].sum(axis=1)
    checks.append(_relative("position_error_sums_are_twice_totals", target_layer,
                            list(zip((2.0 * whole).ravel(), a["position_stats"].sum(axis=(1, 2)).ravel()))))
    exact("position_counts_are_twice_pairs", int(a["position_n"].sum()) == 2 * int(primitives.n_pairs.sum()), 1)

    # Each exact stratum sum must lie between the sums of its counts' bin edges.
    edges = np.tile(ERROR_EDGES, (joint.shape[0], 1))
    edges[:, 0] = np.minimum(a["error_min"], edges[:, 1])
    edges[:, -1] = np.maximum(a["error_max"], edges[:, -2])
    cells = joint.sum(axis=1).astype(np.float64)                   # (sources, error, sequence, distance)
    lower = np.einsum("se,seqd->sqd", edges[:, :-1], cells)
    upper = np.einsum("se,seqd->sqd", edges[:, 1:], cells)
    exact_sums = stratified[..., 0]
    slack = 1e-9 * (1.0 + np.abs(exact_sums))
    violation = np.maximum(lower - exact_sums, exact_sums - upper)
    checks.append({
        "check": "histogram_bounds_contain_exact_sums", "layer": target_layer, "metric": "sum_signed_error",
        "n_compared": int(exact_sums.size), "max_abs_diff": float(max(violation.max(), 0.0)),
        "tolerance": "within bin edges", "status": "pass" if np.all(violation <= slack) else "fail", "detail": "",
    })

    if OUTLIER_PROTEIN in primitives.protein_ids:
        position = primitives.protein_ids.index(OUTLIER_PROTEIN)
        exact("outlier_arrays_match_protein",
              int(a["outlier_strata_n"].sum()) == int(primitives.n_pairs[position])
              and np.array_equal(a["outlier_hist_joint"].astype(np.int64).sum(axis=(2, 3)), protein_hist[:, position]), 1)
    return checks


def _relative(check: str, layer: object, pairs: Sequence[tuple[float, float]]) -> dict[str, object]:
    row = _compare(check, layer, "sums", [(float(a), float(b)) for a, b in pairs])
    return row


def load_primitives(root: Path, fold_bundle, layers: Sequence[int], cohort: str):
    metadata = {
        protein: {"source": str(entry.source), "outer_fold": int(entry.outer_fold),
                  "n_residues": int(entry.n_residues)}
        for protein, entry in fold_bundle.outer_by_protein.items()
    }
    protein_ids = select_cohort(fold_bundle, cohort)
    primitives = Primitives(protein_ids, metadata, layers, cohort)
    n_layers, runtime_rows, consistency, revisions = len(layers), [], [], set()
    histograms = {
        "pair": np.zeros((n_layers, n_layers, N_ERROR_BINS), dtype=np.int64),
        "protein_equal": np.zeros((n_layers, n_layers, N_ERROR_BINS)),
        "pair_excl_outlier": np.zeros((n_layers, n_layers, N_ERROR_BINS), dtype=np.int64),
    }
    for position, target_layer in enumerate(layers):
        path = shard_path(root, target_layer)
        if not path.is_file():
            raise FileNotFoundError(f"Missing final target-layer shard: {path}")
        accumulator = TargetAccumulator.from_file(path, target_layer, protein_ids, n_layers)
        validate_final_shard(accumulator)
        revisions.add(accumulator.code_revision)
        primitives.add_shard(position, accumulator)
        consistency.extend(shard_consistency_checks(target_layer, accumulator, primitives))
        counts = accumulator.arrays["hist_protein"].astype(np.int64)
        histograms["pair"][position] = counts.sum(axis=1)
        histograms["protein_equal"][position] = (counts / primitives.n_pairs[:, None]).sum(axis=1)
        kept = np.asarray([p != OUTLIER_PROTEIN for p in protein_ids])
        histograms["pair_excl_outlier"][position] = counts[:, kept].sum(axis=1)
        runtime_rows.append(_runtime_summary(root, target_layer, primitives, position))
        del accumulator, counts
        log(f"validated target-layer shard {target_layer}")
    primitives.require_complete()
    consistency.append({
        "check": "shards_share_this_code_revision", "layer": "", "metric": "code_revision",
        "n_compared": n_layers, "max_abs_diff": 0.0, "tolerance": "exact",
        "status": "pass" if revisions == {deployment_revision(SCRIPT_DIR)} else "fail",
        "detail": ",".join(sorted(revisions)),
    })
    return primitives, runtime_rows, consistency, histograms


def _runtime_summary(root: Path, target_layer: int, primitives: Primitives, position: int) -> dict[str, object]:
    path = root / "layer_shards" / f"runtime_target_layer_{target_layer:02d}.csv"
    elapsed, rss, status = float("nan"), float("nan"), "unknown"
    if path.is_file():
        with path.open(newline="") as handle:
            completes = [row for row in csv.DictReader(handle) if row["stage"] == "target_layer_complete"]
        if completes:
            elapsed, rss = float(completes[-1]["elapsed_seconds"]), float(completes[-1]["peak_rss_mb"])
            status = completes[-1]["status"]
    return {
        "analysis_name": primitives.analysis_name, "target_layer": target_layer, "completion_status": status,
        "n_proteins": len(primitives.protein_ids), "n_pairs": int(np.sum(primitives.baseline[position, :, 0])),
        "elapsed_seconds": elapsed, "peak_rss_mb": rss,
    }


def iter_cells(layers: Sequence[int]) -> Iterator[tuple[int, int, int, int]]:
    for t, target_layer in enumerate(layers):
        for s, source_layer in enumerate(layers):
            yield s, source_layer, t, target_layer


def per_protein_rows(primitives: Primitives, retained: list[dict[str, object]]) -> Iterator[dict[str, object]]:
    """Stream the per-protein table, keeping only rows the parity checks compare."""
    v1 = set(V1_LAYERS)
    for s, source_layer, t, target_layer in iter_cells(primitives.layers):
        transfer = transfer_identity(source_layer, target_layer)
        keep = source_layer == target_layer or (source_layer in v1 and target_layer in v1)
        for p, protein in enumerate(primitives.protein_ids):
            meta = primitives.metadata[protein]
            stats = stats_from_vector(primitives.probe[t, s, p, WHOLE])
            n_pairs = int(primitives.n_pairs[p])
            row = {
                "analysis_name": primitives.analysis_label, "model_family": MODEL_FAMILY, "feature_dim": FEATURE_DIM,
                "layer": target_layer, "protein_id": protein, "source": meta["source"],
                "cv_protocol_version": CV_PROTOCOL_VERSION, "evaluation_design": EVALUATION_DESIGN,
                "outer_fold": meta["outer_fold"], "evaluation_scope": EVALUATION_SCOPE,
                "oof_prediction_scope": OOF_PREDICTION_SCOPE, "prediction_status": "complete",
                "n_residues": meta["n_residues"], "n_protein_pairs": n_pairs, "n_expected_pairs": n_pairs,
                "n_evaluated_pairs": stats.n, "pair_coverage": stats.n / n_pairs, **stats.metrics(), **transfer,
            }
            if keep:
                retained.append(row)
            yield row


def comparison_accumulator(primitives: Primitives, s: int, t: int) -> OOFComparisonAccumulator:
    proteins = []
    model_total, baseline_total = RegressionSufficientStatistics(), RegressionSufficientStatistics()
    for p, protein in enumerate(primitives.protein_ids):
        model = stats_from_vector(primitives.probe[t, s, p, WHOLE])
        baseline = stats_from_vector(primitives.baseline[t, p])
        model_total.merge(model)
        baseline_total.merge(baseline)
        proteins.append({
            "protein_id": protein, "outer_fold": primitives.metadata[protein]["outer_fold"],
            "n_expected_pairs": model.n, "n_evaluated_pairs": model.n, "stable_fit": True,
            "model": model.to_payload(), "global_baseline": baseline.to_payload(),
            "conditional_baseline": None,
        })
    return OOFComparisonAccumulator.from_payload({
        "model": model_total.to_payload(), "global_baseline": baseline_total.to_payload(),
        "conditional_baseline": None, "proteins": proteins,
    })


def subgroup_accumulator(primitives: Primitives, s: int, t: int, scope: str, labels: Sequence[str]) -> OOFSubgroupAccumulator:
    cells = []
    for label in labels:
        block = primitives.probe[t, s, :, CELL_INDEX[(scope, label)]]
        total = RegressionSufficientStatistics()
        supported = []
        for p, protein in enumerate(primitives.protein_ids):
            if block[p, 0] > 0:
                total.merge(stats_from_vector(block[p]))
                supported.append(protein)
        cells.append({"stats": total.to_payload(), "n_expected_pairs": total.n, "protein_ids": supported})
    return OOFSubgroupAccumulator.from_payload({
        "category_scope": scope, "category_labels": list(labels),
        "seen_proteins": list(primitives.protein_ids), "cells": cells,
    })


def build_metric_tables(primitives: Primitives, fold_bundle) -> dict[str, list[dict[str, object]]]:
    tables: dict[str, list[dict[str, object]]] = {
        "pooled": [], "fold": [], "fold_summary": [], "sequence_separation": [], "spatial_distance": [],
    }
    labels = {scope: tuple(label for s, label in CATEGORY_CELLS if s == scope) for scope in SUBGROUP_SCOPES}
    cohort = set(primitives.protein_ids)
    fold_proteins = {fold: tuple(p for p in fold_bundle.protein_ids_for_outer_fold(fold) if p in cohort)
                     for fold in range(N_OUTER_FOLDS)}
    for s, source_layer, t, target_layer in iter_cells(primitives.layers):
        transfer = transfer_identity(source_layer, target_layer)
        identity = {
            "analysis_name": primitives.analysis_label, "model_family": MODEL_FAMILY, "feature_dim": FEATURE_DIM,
            "layer": target_layer, "evaluation_design": EVALUATION_DESIGN,
            "oof_prediction_scope": OOF_PREDICTION_SCOPE,
        }
        accumulator = comparison_accumulator(primitives, s, t)
        pooled = build_pooled_metrics_row(accumulator, global_baseline_type=GLOBAL_BASELINE_TYPE, **identity)
        tables["pooled"].append({**pooled, **transfer})
        fold_rows = [
            build_fold_metrics_row(accumulator.subset(fold_proteins[fold]), outer_fold=fold,
                                   global_baseline_type=GLOBAL_BASELINE_TYPE, **identity)
            for fold in range(N_OUTER_FOLDS)
        ]
        tables["fold"].extend({**row, **transfer} for row in fold_rows)
        tables["fold_summary"].extend({**row, **transfer} for row in build_fold_summary_rows(fold_rows))
        for scope in SUBGROUP_SCOPES:
            rows = subgroup_accumulator(primitives, s, t, scope, labels[scope]).rows(**identity)
            tables[scope].extend({**row, **transfer} for row in rows)
    return tables


def _pooled_from_sums(n: np.ndarray, sum_y: np.ndarray, sum_y2: np.ndarray, sse: np.ndarray, sae: np.ndarray):
    total_squares = sum_y2 - sum_y * sum_y / n
    return 1.0 - sse / total_squares, sae / n, np.sqrt(sse / n)


def protein_metric_arrays(primitives: Primitives) -> dict[str, np.ndarray]:
    """Per-protein whole-cell metrics of shape (target, source, protein)."""
    whole = primitives.probe[..., WHOLE, :]
    n = whole[..., 0]
    total_squares = whole[..., SUM_Y2] - whole[..., SUM_Y] * whole[..., SUM_Y] / n
    with np.errstate(divide="ignore", invalid="ignore"):
        r2 = np.where(total_squares <= FLOAT_EPS, np.nan, 1.0 - whole[..., SUM_SQUARED] / total_squares)
    return {"r2": r2, "mae": whole[..., SUM_ABS] / n, "rmse": np.sqrt(whole[..., SUM_SQUARED] / n),
            "bias": whole[..., SUM_ERROR] / n, "squared_error": whole[..., SUM_SQUARED]}


def transfer_matrix_rows(pooled: Sequence[Mapping[str, object]], primitives: Primitives) -> list[dict[str, object]]:
    layers = primitives.layers
    by_cell = {(int(r["source_layer"]), int(r["target_layer"])): r for r in pooled}
    metrics = protein_metric_arrays(primitives)
    whole = primitives.probe[..., WHOLE, :]
    kept = np.asarray([p != OUTLIER_PROTEIN for p in primitives.protein_ids])
    has_outlier = not kept.all()
    outlier = int(np.flatnonzero(~kept)[0]) if has_outlier else -1
    rows = []
    for s, source_layer, t, target_layer in iter_cells(layers):
        cell = by_cell[(source_layer, target_layer)]
        native_target, native_source = by_cell[(target_layer, target_layer)], by_cell[(source_layer, source_layer)]
        r2 = metrics["r2"][t, s]
        finite_r2 = r2[np.isfinite(r2)]
        mae = metrics["mae"][t, s]
        squared = metrics["squared_error"][t, s]
        top = int(np.argmax(squared))
        kept_sums = whole[t, s, kept].sum(axis=0)
        r2_excl, mae_excl, rmse_excl = _pooled_from_sums(
            kept_sums[0], kept_sums[SUM_Y], kept_sums[SUM_Y2], kept_sums[SUM_SQUARED], kept_sums[SUM_ABS])
        row = {
            **transfer_identity(source_layer, target_layer), "analysis_name": primitives.analysis_label,
            "n_proteins": cell["n_proteins"], "n_evaluated_pairs": cell["n_evaluated_pairs"],
            "pooled_r2": cell["pooled_r2"],
            "mean_protein_r2": float(np.mean(finite_r2)) if finite_r2.size else float("nan"),
            "median_protein_r2": float(np.median(finite_r2)) if finite_r2.size else float("nan"),
            "n_finite_r2": int(finite_r2.size), "pooled_mae": cell["pooled_mae"],
            "mean_protein_mae": float(np.mean(mae)), "median_protein_mae": float(np.median(mae)),
            "pooled_rmse": cell["pooled_rmse"],
            "pooled_bias_pred_minus_true": cell["pooled_bias_pred_minus_true"],
            "top_protein_id": primitives.protein_ids[top],
            "top_protein_squared_error_share": float(squared[top] / squared.sum()),
            f"squared_error_share_{EXCLUDED}": float(squared[outlier] / squared.sum()) if has_outlier else 0.0,
            f"pooled_r2_excl_{EXCLUDED}": float(r2_excl), f"pooled_mae_excl_{EXCLUDED}": float(mae_excl),
            f"pooled_rmse_excl_{EXCLUDED}": float(rmse_excl),
        }
        for metric in ("r2", "mae"):
            value = float(cell[f"pooled_{metric}"])
            row[f"native_target_pooled_{metric}"] = float(native_target[f"pooled_{metric}"])
            row[f"delta_pooled_vs_native_target_{metric}"] = value - float(native_target[f"pooled_{metric}"])
            row[f"native_source_pooled_{metric}"] = float(native_source[f"pooled_{metric}"])
            row[f"delta_pooled_vs_native_source_{metric}"] = value - float(native_source[f"pooled_{metric}"])
        rows.append(row)
    return rows


def protein_anomaly_rows(primitives: Primitives) -> list[dict[str, object]]:
    """The proteins contributing most squared error in each cell."""
    metrics = protein_metric_arrays(primitives)
    rows = []
    for s, source_layer, t, target_layer in iter_cells(primitives.layers):
        squared = metrics["squared_error"][t, s]
        mae = metrics["mae"][t, s]
        median = float(np.median(mae))
        spread = 1.4826 * float(np.median(np.abs(mae - median)))
        for rank, p in enumerate(np.argsort(-squared)[:TOP_PROTEINS_PER_CELL], start=1):
            protein = primitives.protein_ids[int(p)]
            meta = primitives.metadata[protein]
            rows.append({
                **transfer_identity(source_layer, target_layer), "rank": rank, "protein_id": protein,
                "source": meta["source"], "outer_fold": meta["outer_fold"], "n_residues": meta["n_residues"],
                "n_pairs": int(primitives.n_pairs[p]),
                "squared_error_share": float(squared[p] / squared.sum()), "r2": float(metrics["r2"][t, s, p]),
                "mae": float(mae[p]), "rmse": float(metrics["rmse"][t, s, p]),
                "mean_signed_error": float(metrics["bias"][t, s, p]),
                "mae_robust_z": (float(mae[p]) - median) / spread if spread > 0 else float("nan"),
            })
    return rows


def _quantile_row(identity: Mapping[str, object], scope: str, label: str, weighting: str,
                  counts: np.ndarray, low: float, high: float, n_pairs: int) -> dict[str, object]:
    q = histogram_quantiles(counts, QUANTILES, low, high)
    total = float(counts.sum())
    share = {}
    for angstroms in WITHIN_ANGSTROMS:
        inside = (ERROR_EDGES[:-1] >= -angstroms) & (ERROR_EDGES[1:] <= angstroms)
        share[f"share_within_{angstroms}A"] = float(counts[inside].sum() / total) if total else float("nan")
    return {
        **identity, "category_scope": scope, "category_label": label, "weighting": weighting,
        "n_pairs": int(n_pairs),
        **dict(zip(QUANTILE_COLUMNS, map(float, q))), "signed_error_iqr": float(q[4] - q[2]), **share,
        "share_in_open_tail_bins": float((counts[0] + counts[-1]) / total) if total else float("nan"),
        "cell_error_min": low, "cell_error_max": high,
    }


def error_quantile_rows(root: Path, primitives: Primitives) -> Iterator[dict[str, object]]:
    kept = np.asarray([p != OUTLIER_PROTEIN for p in primitives.protein_ids])
    for t, target_layer in enumerate(primitives.layers):
        with np.load(shard_path(root, target_layer)) as shard:
            joint = shard["hist_joint"].astype(np.int64).sum(axis=1)     # (sources, error, sequence, distance)
            protein_hist = shard["hist_protein"].astype(np.int64)
        for s, source_layer in enumerate(primitives.layers):
            identity = transfer_identity(source_layer, target_layer)
            low, high = float(primitives.error_min[t, s]), float(primitives.error_max[t, s])
            cell = joint[s]
            for scope, label in CATEGORY_CELLS:
                if scope == "all_pairs":
                    counts = cell.sum(axis=(1, 2))
                elif scope == "sequence_separation":
                    counts = cell[:, list(CANONICAL_SEQUENCE_GROUPS[label])].sum(axis=(1, 2))
                else:
                    counts = cell[:, :, list(CANONICAL_DISTANCE_GROUPS[label])].sum(axis=(1, 2))
                yield _quantile_row(identity, scope, label, "pair", counts, low, high, int(counts.sum()))
            yield _quantile_row(identity, "all_pairs", "all", "protein_equal",
                                (protein_hist[s] / primitives.n_pairs[:, None]).sum(axis=0), low, high,
                                int(primitives.n_pairs.sum()))
            without = protein_hist[s, kept].sum(axis=0)
            yield _quantile_row(identity, "all_pairs", "all", f"pair_excl_{EXCLUDED}", without, low, high,
                                int(without.sum()))


def _canonical_label(groups: Mapping[str, Sequence[int]], index: int) -> str:
    return next(label for label, members in groups.items() if index in members)


def error_strata_rows(root: Path, primitives: Primitives) -> Iterator[dict[str, object]]:
    sequence_canonical = [_canonical_label(CANONICAL_SEQUENCE_GROUPS, q) for q in range(len(SEQUENCE_LABELS))]
    distance_canonical = [_canonical_label(CANONICAL_DISTANCE_GROUPS, d) for d in range(len(DISTANCE_LABELS))]
    for target_layer in primitives.layers:
        with np.load(shard_path(root, target_layer)) as shard:
            stats = shard["strata_stats"].sum(axis=1)                    # (sources, sequence, distance, 3)
            counts, proteins = shard["strata_n"].sum(axis=0), shard["strata_proteins"].sum(axis=0)
            outlier_stats, outlier_n = shard["outlier_strata_stats"], shard["outlier_strata_n"]
        status = strata_support_status(counts, proteins)
        occupied = np.argwhere(counts > 0)
        kept_stats, kept_n = stats - outlier_stats, counts - outlier_n
        total_squared = stats[..., 2].sum(axis=(1, 2))
        kept_total_squared = kept_stats[..., 2].sum(axis=(1, 2))
        for s, source_layer in enumerate(primitives.layers):
            identity = transfer_identity(source_layer, target_layer)
            for q, d in occupied:
                n, sum_error, sum_abs, sum_squared = int(counts[q, d]), *map(float, stats[s, q, d])
                mean = sum_error / n
                n_kept = int(kept_n[q, d])
                error_kept, abs_kept, squared_kept = map(float, kept_stats[s, q, d])
                yield {
                    **identity, "sequence_separation_bin": SEQUENCE_LABELS[q],
                    "final_structure_distance_bin": DISTANCE_LABELS[d],
                    "canonical_sequence_bin": sequence_canonical[q], "canonical_distance_bin": distance_canonical[d],
                    "n_pairs": n, "n_proteins": int(proteins[q, d]), "support_status": str(status[q, d]),
                    "sum_signed_error": sum_error, "sum_abs_error": sum_abs, "sum_squared_error": sum_squared,
                    "mean_signed_error": mean, "sd_signed_error": math.sqrt(max(0.0, sum_squared / n - mean * mean)),
                    "mae": sum_abs / n, "rmse": math.sqrt(sum_squared / n),
                    "share_of_squared_error": sum_squared / float(total_squared[s]),
                    f"n_pairs_excl_{EXCLUDED}": n_kept, f"sum_signed_error_excl_{EXCLUDED}": error_kept,
                    f"sum_abs_error_excl_{EXCLUDED}": abs_kept, f"sum_squared_error_excl_{EXCLUDED}": squared_kept,
                    f"mean_signed_error_excl_{EXCLUDED}": error_kept / n_kept if n_kept else float("nan"),
                    f"share_of_squared_error_excl_{EXCLUDED}": squared_kept / float(kept_total_squared[s]),
                }


def error_position_rows(root: Path, primitives: Primitives) -> Iterator[dict[str, object]]:
    for target_layer in primitives.layers:
        with np.load(shard_path(root, target_layer)) as shard:
            stats = shard["position_stats"].sum(axis=1)                  # (sources, position, 3)
            counts = shard["position_n"].sum(axis=0)
            outlier_stats, outlier_n = shard["outlier_position_stats"], shard["outlier_position_n"]
        kept_stats, kept_n = stats - outlier_stats, counts - outlier_n
        for s, source_layer in enumerate(primitives.layers):
            identity = transfer_identity(source_layer, target_layer)
            for b in range(N_POSITION_BINS):
                n, n_kept = int(counts[b]), int(kept_n[b])
                yield {
                    **identity, "position_bin": b, "position_start": b / N_POSITION_BINS,
                    "position_end": (b + 1) / N_POSITION_BINS, "n_residue_pairs": n,
                    "mean_signed_error": float(stats[s, b, 0]) / n, "mae": float(stats[s, b, 1]) / n,
                    "rmse": math.sqrt(float(stats[s, b, 2]) / n), f"n_residue_pairs_excl_{EXCLUDED}": n_kept,
                    f"mean_signed_error_excl_{EXCLUDED}": float(kept_stats[s, b, 0]) / n_kept,
                    f"mae_excl_{EXCLUDED}": float(kept_stats[s, b, 1]) / n_kept,
                    f"rmse_excl_{EXCLUDED}": math.sqrt(float(kept_stats[s, b, 2]) / n_kept),
                }


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def producer_parity_checks(
    producer_family: Path, per_protein: Sequence[Mapping[str, object]],
    tables: Mapping[str, Sequence[Mapping[str, object]]], layers: Sequence[int],
    protein_ids: Sequence[str], full_cohort: bool,
) -> list[dict[str, object]]:
    """Every native cell must reproduce the frozen producer's published metrics.

    The producer pools all 552 proteins, so pooled, fold, and subgroup comparisons
    apply only to the full cohort. Per-protein metrics apply to any cohort.
    """
    checks: list[dict[str, object]] = []
    wanted, cohort = set(layers), set(protein_ids)

    def native(rows: Iterable[Mapping[str, object]]):
        return [r for r in rows if int(r["source_layer"]) == int(r["target_layer"])]

    producer_pp = {(int(r["layer"]), r["protein_id"]): r for r in _read_csv(producer_family / "per_protein.csv")
                   if int(r["layer"]) in wanted and r["protein_id"] in cohort}
    ours_pp = {(int(r["layer"]), str(r["protein_id"])): r for r in native(per_protein)}
    for layer in layers:
        keys = sorted(k for k in ours_pp if k[0] == layer)
        same_keys = keys == sorted(k for k in producer_pp if k[0] == layer)
        for metric in PER_PROTEIN_METRICS:
            row = _compare("native_per_protein_vs_producer", layer, metric,
                           [(producer_pp[k][metric], ours_pp[k][metric]) for k in keys if k in producer_pp],
                           "" if same_keys else "protein sets differ")
            if not same_keys:
                row["status"] = "fail"
            checks.append(row)
    if not full_cohort:
        return checks

    producer_pooled = {int(r["layer"]): r for r in _read_csv(producer_family / "pooled_metrics.csv")}
    ours_pooled = {int(r["layer"]): r for r in native(tables["pooled"])}
    producer_fold = {(int(r["layer"]), int(r["outer_fold"])): r for r in _read_csv(producer_family / "fold_metrics.csv")}
    ours_fold = {(int(r["layer"]), int(r["outer_fold"])): r for r in native(tables["fold"])}
    for layer in layers:
        for metric in POOLED_CHECK_METRICS:
            checks.append(_compare("native_pooled_vs_producer", layer, metric,
                                   [(producer_pooled[layer][metric], ours_pooled[layer][metric])]))
            checks.append(_compare("native_fold_vs_producer", layer, metric, [
                (producer_fold[(layer, f)][metric], ours_fold[(layer, f)][metric]) for f in range(N_OUTER_FOLDS)
            ]))

    for scope, filename in (("sequence_separation", "by_sequence_separation.csv"),
                            ("spatial_distance", "by_spatial_distance.csv")):
        producer_sub = {(int(r["layer"]), r["category_label"]): r
                        for r in _read_csv(producer_family / "subgroup_metrics" / filename)}
        ours_sub = {(int(r["layer"]), str(r["category_label"])): r for r in native(tables[scope])}
        labels = [label for s, label in CATEGORY_CELLS if s == scope]
        for layer in layers:
            for metric in SUBGROUP_CHECK_METRICS:
                checks.append(_compare(f"native_subgroup_{scope}_vs_producer", layer, metric, [
                    (producer_sub[(layer, label)][metric], ours_sub[(layer, label)][metric]) for label in labels
                ]))
            same = all(producer_sub[(layer, label)]["support_status"] == ours_sub[(layer, label)]["support_status"]
                       for label in labels)
            checks.append({
                "check": f"native_subgroup_{scope}_vs_producer", "layer": layer, "metric": "support_status",
                "n_compared": len(labels), "max_abs_diff": 0.0, "tolerance": "exact",
                "status": "pass" if same else "fail", "detail": "",
            })
    return checks


def _strict(row: dict[str, object]) -> dict[str, object]:
    """A comparison whose key sets differ fails even if the shared keys agree."""
    if row["detail"]:
        row["status"] = "fail"
    return row


def v1_primitive_checks(root_v1: Path, primitives: Primitives) -> list[dict[str, object]]:
    """Every per-protein statistic, in every category, must match the ten-layer shards.

    These additive statistics apply to any cohort, so they stand in for the pooled
    comparisons a partial cohort cannot make.
    """
    layers = [layer for layer in V1_LAYERS if layer in primitives.layers]
    position = {layer: primitives.layers.index(layer) for layer in layers}
    proteins = {protein: p for p, protein in enumerate(primitives.protein_ids)}
    pairs: dict[str, dict[str, list[tuple[str, float]]]] = {
        "probe": {field: [] for field in STAT_FIELDS}, "baseline": {field: [] for field in STAT_FIELDS},
    }
    for target in layers:
        for row in _read_csv(root_v1 / "layer_shards" / f"cross_layer_primitives_target_layer_{target:02d}.csv"):
            p = proteins.get(row["protein_id"])
            if p is None:
                continue
            if row["predictor"] == "global_baseline":
                kind, vector = "baseline", primitives.baseline[position[target], p]
            elif row["transfer_mode"] == TRANSFER_MODE and int(row["source_layer"]) in position:
                cell = CELL_INDEX[(row["category_scope"], row["category_label"])]
                kind = "probe"
                vector = primitives.probe[position[target], position[int(row["source_layer"])], p, cell]
            else:
                continue
            for index, field in enumerate(STAT_FIELDS):
                pairs[kind][field].append((row[field], float(vector[index])))
    expected = {"probe": len(layers) ** 2 * len(proteins) * len(CATEGORY_CELLS), "baseline": len(layers) * len(proteins)}
    checks = []
    for kind, fields in pairs.items():
        for field, compared in fields.items():
            detail = "" if len(compared) == expected[kind] else f"{len(compared)} rows, expected {expected[kind]}"
            checks.append(_strict(_compare(f"v1_overlap_primitives_{kind}", "", field, compared, detail)))
    return checks


def v1_overlap_checks(
    root_v1: Path, per_protein: Sequence[Mapping[str, object]],
    tables: Mapping[str, Sequence[Mapping[str, object]]], primitives: Primitives, full_cohort: bool,
) -> list[dict[str, object]]:
    """The cells measured by the ten-layer run must come out the same."""
    checks: list[dict[str, object]] = []
    v1_cells = {(s, t) for s in V1_LAYERS for t in V1_LAYERS}

    def cell(row: Mapping[str, object]) -> tuple[int, int]:
        return int(row["source_layer"]), int(row["target_layer"])

    def load(relative: str) -> list[dict[str, str]]:
        return [r for r in _read_csv(root_v1 / relative) if r["transfer_mode"] == TRANSFER_MODE]

    cohort = set(primitives.protein_ids)
    theirs = {(*cell(r), r["protein_id"]): r for r in load("per_protein.csv") if r["protein_id"] in cohort}
    ours = {(*cell(r), str(r["protein_id"])): r for r in per_protein if cell(r) in v1_cells}
    same_keys = set(theirs) == set(ours)
    for metric in PER_PROTEIN_METRICS:
        checks.append(_strict(_compare(
            "v1_overlap_per_protein", "", metric,
            [(theirs[k][metric], ours[k][metric]) for k in sorted(theirs) if k in ours],
            "" if same_keys else "cell or protein sets differ")))
    checks.extend(v1_primitive_checks(root_v1, primitives))
    if not full_cohort:
        return checks
    for name, relative, key in (("pooled", "pooled_metrics.csv", ()), ("fold", "fold_metrics.csv", ("outer_fold",))):
        theirs_table = {(*cell(r), *(str(r[k]) for k in key)): r for r in load(relative)}
        ours_table = {(*cell(r), *(str(r[k]) for k in key)): r for r in tables[name] if cell(r) in v1_cells}
        for metric in POOLED_CHECK_METRICS:
            checks.append(_strict(_compare(f"v1_overlap_{name}", "", metric, [
                (theirs_table[k][metric], ours_table[k][metric]) for k in sorted(theirs_table) if k in ours_table
            ], "" if set(theirs_table) == set(ours_table) else "cell sets differ")))
    for scope, filename in (("sequence_separation", "by_sequence_separation.csv"),
                            ("spatial_distance", "by_spatial_distance.csv")):
        theirs_table = {(*cell(r), r["category_label"]): r for r in load(f"subgroup_metrics/{filename}")}
        ours_table = {(*cell(r), str(r["category_label"])): r for r in tables[scope] if cell(r) in v1_cells}
        for metric in SUBGROUP_CHECK_METRICS:
            checks.append(_strict(_compare(f"v1_overlap_subgroup_{scope}", "", metric, [
                (theirs_table[k][metric], ours_table[k][metric]) for k in sorted(theirs_table) if k in ours_table
            ], "" if set(theirs_table) == set(ours_table) else "cell sets differ")))
    return checks


def baseline_consistent(primitives: Primitives) -> bool:
    reference = primitives.baseline[0]
    return all(np.allclose(primitives.baseline[t], reference, rtol=1e-12, atol=0.0)
               for t in range(1, len(primitives.layers)))


def main() -> None:
    started = time.perf_counter()
    cohort = cohort_name()
    full_cohort = cohort == "full"
    root = output_root(cohort)
    assert_writable(root)
    layers = LAYERS
    # A crash below must not leave an earlier run's "complete" standing.
    atomic_write_json(root / "run_manifest.json", {
        "analysis_name": analysis_name(cohort), "analysis_status": "in_progress", "cohort": cohort,
        "created_at": utc_now(),
    })
    fold_bundle = load_cv_folds(metadata_dir=producer_root() / "metadata")
    _, provenance, status_rows = load_probe_bank(producer_root(), fold_bundle, layers)
    primitives, runtime_rows, consistency, histograms = load_primitives(root, fold_bundle, layers, cohort)
    protein_ids = primitives.protein_ids
    log("validated all target-layer shards")

    retained: list[dict[str, object]] = []
    n_per_protein = stream_write_csv(root / "per_protein.csv", PER_PROTEIN_OUTPUT_COLUMNS,
                                     per_protein_rows(primitives, retained))
    log("wrote per-protein table")
    tables = build_metric_tables(primitives, fold_bundle)
    atomic_write_csv(root / "pooled_metrics.csv", POOLED_OUTPUT_COLUMNS, tables["pooled"])
    atomic_write_csv(root / "fold_metrics.csv", FOLD_OUTPUT_COLUMNS, tables["fold"])
    atomic_write_csv(root / "fold_summary.csv", FOLD_SUMMARY_OUTPUT_COLUMNS, tables["fold_summary"])
    atomic_write_csv(root / "subgroup_metrics/by_sequence_separation.csv", SUBGROUP_OUTPUT_COLUMNS,
                     tables["sequence_separation"])
    atomic_write_csv(root / "subgroup_metrics/by_spatial_distance.csv", SUBGROUP_OUTPUT_COLUMNS,
                     tables["spatial_distance"])
    matrix = transfer_matrix_rows(tables["pooled"], primitives)
    atomic_write_csv(root / "transfer_matrix.csv", TRANSFER_MATRIX_COLUMNS, matrix)
    anomalies = protein_anomaly_rows(primitives)
    atomic_write_csv(root / "protein_anomalies.csv", PROTEIN_ANOMALY_COLUMNS, anomalies)
    atomic_write_csv(root / "source_probe_status.csv", SOURCE_STATUS_COLUMNS, status_rows)
    atomic_write_csv(root / "runtime.csv", RUNTIME_OUTPUT_COLUMNS, runtime_rows)
    log("wrote metric tables")

    n_quantiles = stream_write_csv(root / "error_quantiles.csv", ERROR_QUANTILE_COLUMNS,
                                   error_quantile_rows(root, primitives))
    n_strata = stream_write_csv(root / "error_strata.csv", ERROR_STRATA_COLUMNS, error_strata_rows(root, primitives))
    n_position = stream_write_csv(root / "error_position.csv", ERROR_POSITION_COLUMNS,
                                  error_position_rows(root, primitives))
    atomic_savez_compressed(
        root / "error_histograms_by_cell.npz", pair=histograms["pair"], protein_equal=histograms["protein_equal"],
        pair_excl_outlier=histograms["pair_excl_outlier"], error_edges=ERROR_EDGES,
        layers=np.asarray(layers), error_min=primitives.error_min, error_max=primitives.error_max,
    )
    log("wrote error tables")

    checks = list(consistency)
    checks.extend(producer_parity_checks(producer_root() / PRODUCER_SPEC.family_dir, retained, tables, layers,
                                         protein_ids, full_cohort))
    checks.extend(v1_overlap_checks(v1_root(), retained, tables, primitives, full_cohort))
    checks.append({
        "check": "global_baseline_identical_across_target_shards", "layer": "", "metric": "all_primitives",
        "n_compared": int(primitives.baseline.size), "max_abs_diff": 0.0, "tolerance": "1e-12 relative",
        "status": "pass" if baseline_consistent(primitives) else "fail", "detail": "",
    })
    atomic_write_csv(root / "parity_checks.csv", PARITY_COLUMNS, checks)
    failed = [c for c in checks if c["status"] != "pass"]
    status = "complete" if not failed else "failed_parity"
    tail_share = float((histograms["pair"][..., 0].sum() + histograms["pair"][..., -1].sum()) / histograms["pair"].sum())
    atomic_write_json(root / "run_manifest.json", {
        "analysis_name": analysis_name(cohort), "analysis_status": status, "created_at": utc_now(),
        "cohort": cohort, "n_proteins": len(protein_ids),
        "cohort_note": "" if full_cohort else (
            "Partial cohort: the ECOD inputs were unavailable. Re-run with ALL_LAYER_COHORT=full when they return. "
            "Pooled metrics here are not comparable to full-cohort tables."
        ),
        "gates_not_applicable": [] if full_cohort else [
            "pooled, fold, and subgroup parity with the producer and ten-layer run, which pool all 552 proteins",
        ],
        "description": (
            "Frozen protocol-2.0 shared-OLS probes replayed across all 48 Evoformer layers without "
            "refitting; fold-f probes evaluate fold-f held-out proteins at the target layer. "
            "Signed error is probe distance minus final structure distance."
        ),
        "producer_analysis": PRODUCER_SPEC.analysis_name, "producer": provenance,
        "metadata_digests": dict(fold_bundle.fold_digests), "layers": list(layers),
        "transfer_mode": TRANSFER_MODE, "analysis_label": analysis_label(cohort),
        "mode_definition": (
            "source intercept + ((x - target_mean)/target_scale) @ source_coef_std; target scaler is "
            "the same-fold train-only scaler from the target-layer artifact"
        ),
        "regression_target": "OpenFold final predicted structure C-alpha distance",
        "layer_column_definition": "layer equals target_layer (the evaluated representation)",
        "evaluation_scope": EVALUATION_SCOPE, "oof_prediction_scope": OOF_PREDICTION_SCOPE,
        "cv_protocol_version": CV_PROTOCOL_VERSION, "feature_dim": FEATURE_DIM,
        "row_counts": {
            "per_protein.csv": n_per_protein, "pooled_metrics.csv": len(tables["pooled"]),
            "fold_metrics.csv": len(tables["fold"]), "fold_summary.csv": len(tables["fold_summary"]),
            "subgroup_metrics/by_sequence_separation.csv": len(tables["sequence_separation"]),
            "subgroup_metrics/by_spatial_distance.csv": len(tables["spatial_distance"]),
            "transfer_matrix.csv": len(matrix), "protein_anomalies.csv": len(anomalies),
            "error_quantiles.csv": n_quantiles, "error_strata.csv": n_strata, "error_position.csv": n_position,
            "source_probe_status.csv": len(status_rows), "parity_checks.csv": len(checks),
            "runtime.csv": len(runtime_rows),
        },
        "error_bins": {"n_bins": N_ERROR_BINS, "edges": [float(e) if np.isfinite(e) else str(e) for e in ERROR_EDGES],
                       "share_of_predictions_in_open_tail_bins": tail_share,
                       "error_min": float(primitives.error_min.min()), "error_max": float(primitives.error_max.max())},
        "sequence_separation_bins": list(SEQUENCE_LABELS), "final_structure_distance_bins": list(DISTANCE_LABELS),
        "excluded_sensitivity_protein": OUTLIER_PROTEIN, "geometry_metrics": "not_computed_by_request",
        "parity": {"n_checks": len(checks), "n_failed": len(failed), "tolerance": PARITY_TOLERANCE,
                   "failed": [f"{c['check']}:{c['layer']}:{c['metric']}" for c in failed[:50]]},
        "code_revision": deployment_revision(SCRIPT_DIR), "aggregator_sha256": sha256_file(Path(__file__)),
        "aggregation_elapsed_seconds": time.perf_counter() - started,
    })
    log(f"aggregation {status}; {len(failed)} failed parity checks")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

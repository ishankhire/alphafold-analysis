#!/usr/bin/env python3
"""Consolidate cross-layer transfer primitives into canonical metric tables."""

from __future__ import annotations

import csv
import math
import os
import resource
import sys
import tempfile
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
CV_DIR = CLUSTER_DIR / "cross_validation"
TRIANGLE_DIR = CLUSTER_DIR / "triangle_violation_subgroups"
for path in (SCRIPT_DIR, CLUSTER_DIR, CV_DIR, TRIANGLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from canonical_storage import (  # noqa: E402
    SHARED_PARENT, assert_writable, csv_root, deployment_revision, normalize_shared_tree,
)
from cross_layer_transfer_common import (  # noqa: E402
    ANALYSIS_BY_MODE, ANALYSIS_NAME, CATEGORY_LABELS, CV_PROTOCOL_VERSION,
    DEFAULT_OUTPUT_SUBDIR, END_LAYERS, EVALUATION_DESIGN, EVALUATION_SCOPE,
    EXPECTED_PROTEINS, FEATURE_DIM, MID_LAYERS, MODEL_FAMILY, MODES, N_OUTER_FOLDS,
    OOF_PREDICTION_SCOPE, OUTPUT_ENV, PRIMITIVE_COLUMNS, PRODUCER_SPEC, STAT_FIELDS,
    TRANSFER_COLUMNS, TRANSFER_LAYERS, convert_primitive_row, load_transfer_artifacts,
    transfer_identity, validate_target_shard,
)
from cross_validation_common import (  # noqa: E402
    FOLD_METRICS_COLUMNS, FOLD_SUMMARY_COLUMNS, GLOBAL_BASELINE_TYPE, PER_PROTEIN_COLUMNS,
    POOLED_METRICS_COLUMNS, SUBGROUP_METRICS_COLUMNS, OOFComparisonAccumulator,
    OOFSubgroupAccumulator, RegressionSufficientStatistics, build_fold_metrics_row,
    build_fold_summary_rows, build_pooled_metrics_row, load_cv_folds,
)
from regression_output_helpers import _csv_value, atomic_write_csv, atomic_write_json, sha256_file  # noqa: E402


CELL_INDEX = {
    (scope, label): index
    for index, (scope, label) in enumerate(
        (scope, label) for scope, labels in CATEGORY_LABELS.items() for label in labels
    )
}
SUBGROUP_SCOPES = ("sequence_separation", "spatial_distance")
PARITY_TOLERANCE = 1e-8
GENERALIZATION_SCOPE = "unseen_protein_ids"

PER_PROTEIN_OUTPUT_COLUMNS = (*PER_PROTEIN_COLUMNS, *TRANSFER_COLUMNS)
POOLED_OUTPUT_COLUMNS = (*POOLED_METRICS_COLUMNS, *TRANSFER_COLUMNS)
FOLD_OUTPUT_COLUMNS = (*FOLD_METRICS_COLUMNS, *TRANSFER_COLUMNS)
FOLD_SUMMARY_OUTPUT_COLUMNS = (*FOLD_SUMMARY_COLUMNS, *TRANSFER_COLUMNS)
SUBGROUP_OUTPUT_COLUMNS = (*SUBGROUP_METRICS_COLUMNS, *TRANSFER_COLUMNS)
PER_PROTEIN_SUBGROUP_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_id", "source",
    "cv_protocol_version", "evaluation_design", "outer_fold", "evaluation_scope",
    "oof_prediction_scope", "generalization_scope", "category_scope", "category_label",
    "n_residues", "n_pairs", "sum_absolute_error", "mae", "support_status",
    *TRANSFER_COLUMNS,
)
MATRIX_METRICS = ("r2", "mae")
TRANSFER_MATRIX_COLUMNS = (
    *TRANSFER_COLUMNS, "analysis_name", "n_proteins", "n_evaluated_pairs",
    "pooled_r2", "mean_protein_r2", "median_protein_r2", "n_finite_r2",
    "pooled_mae", "mean_protein_mae", "median_protein_mae", "pooled_rmse",
    "pooled_bias_pred_minus_true",
    *(f"{kind}_{metric}" for metric in MATRIX_METRICS for kind in (
        "native_target_pooled", "delta_pooled_vs_native_target",
        "native_source_pooled", "delta_pooled_vs_native_source",
    )),
)
SOURCE_STATUS_COLUMNS = (
    "source_layer", "outer_fold", "fit_status", "support_flags", "condition_number",
    "artifact_path", "artifact_sha256", "scaler_digest", "global_baseline_value",
)
PARITY_COLUMNS = (
    "check", "layer", "metric", "n_compared", "max_abs_diff", "tolerance", "status", "detail",
)
RUNTIME_OUTPUT_COLUMNS = (
    "analysis_name", "target_layer", "completion_status", "n_proteins", "n_pairs",
    "elapsed_seconds", "peak_rss_mb",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def producer_root() -> Path:
    return csv_root() / "regression_5fold_v1"


def output_root() -> Path:
    value = os.environ.get(OUTPUT_ENV)
    if value:
        return Path(value).expanduser().resolve()
    return csv_root() / DEFAULT_OUTPUT_SUBDIR


def stream_write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> int:
    """Atomically write a large CSV without materializing every row."""
    assert_writable(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    count = 0
    expected = set(columns)
    try:
        with os.fdopen(descriptor, "w", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(columns)
            for row in rows:
                if set(row) != expected:
                    raise ValueError(f"Row schema mismatch for {path.name}: {sorted(set(row) ^ expected)}")
                writer.writerow([_csv_value(row[column]) for column in columns])
                count += 1
            handle.flush()
            os.fsync(handle.fileno())
        if path.resolve().is_relative_to(SHARED_PARENT):
            os.chmod(temporary, 0o640)
        os.replace(temporary, path)
    except Exception:
        Path(temporary).unlink(missing_ok=True)
        raise
    return count


def stats_from_vector(vector: np.ndarray) -> RegressionSufficientStatistics:
    return RegressionSufficientStatistics.from_payload({
        "n": int(vector[0]),
        **{field: float(vector[index]) for index, field in enumerate(STAT_FIELDS) if index},
    })


class TransferPrimitives:
    """Dense primitive arrays indexed by target, source, mode, protein, cell, stat."""

    def __init__(self, protein_ids: Sequence[str], metadata: Mapping[str, object]):
        self.protein_ids = tuple(protein_ids)
        self.protein_index = {protein: index for index, protein in enumerate(self.protein_ids)}
        self.metadata = metadata
        shape = (len(TRANSFER_LAYERS), len(TRANSFER_LAYERS), len(MODES), len(self.protein_ids),
                 len(CELL_INDEX), len(STAT_FIELDS))
        self.probe = np.full(shape, np.nan, dtype=np.float64)
        self.baseline = np.full((len(TRANSFER_LAYERS), len(self.protein_ids), len(STAT_FIELDS)),
                                np.nan, dtype=np.float64)

    def add_shard(self, target_layer: int, rows: Sequence[Mapping[str, object]]) -> None:
        t = TRANSFER_LAYERS.index(target_layer)
        for row in rows:
            protein = str(row["protein_id"])
            p = self.protein_index[protein]
            meta = self.metadata[protein]
            if (row["source"], int(row["outer_fold"]), int(row["n_residues"])) != (
                meta["source"], meta["outer_fold"], meta["n_residues"],
            ):
                raise ValueError(f"Shard identity for {protein} disagrees with frozen folds")
            values = [float(row[field]) for field in STAT_FIELDS]
            if row["predictor"] == "global_baseline":
                self.baseline[t, p] = values
            else:
                s = TRANSFER_LAYERS.index(int(row["source_layer"]))
                m = MODES.index(str(row["transfer_mode"]))
                c = CELL_INDEX[(str(row["category_scope"]), str(row["category_label"]))]
                self.probe[t, s, m, p, c] = values

    def require_complete(self) -> None:
        if np.isnan(self.probe).any() or np.isnan(self.baseline).any():
            raise ValueError("Primitive arrays are incomplete")
        all_cell = CELL_INDEX[("all_pairs", "all")]
        expected_pairs = np.asarray([
            self.metadata[p]["n_residues"] * (self.metadata[p]["n_residues"] - 1) // 2
            for p in self.protein_ids
        ], dtype=np.float64)
        if not np.array_equal(self.probe[..., all_cell, 0], np.broadcast_to(expected_pairs, self.probe.shape[:4])):
            raise ValueError("Whole-protein pair counts disagree with frozen n_residues")
        if not np.array_equal(self.baseline[..., 0], np.broadcast_to(expected_pairs, self.baseline.shape[:2])):
            raise ValueError("Baseline pair counts disagree with frozen n_residues")


def load_primitives(root: Path, fold_bundle) -> tuple[TransferPrimitives, list[dict[str, object]]]:
    metadata = {
        protein: {"source": str(entry.source), "outer_fold": int(entry.outer_fold),
                  "n_residues": int(entry.n_residues)}
        for protein, entry in fold_bundle.outer_by_protein.items()
    }
    primitives = TransferPrimitives(fold_bundle.eligible_protein_ids, metadata)
    runtime_rows = []
    for target_layer in TRANSFER_LAYERS:
        path = root / "layer_shards" / f"cross_layer_primitives_target_layer_{target_layer:02d}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing final target-layer shard: {path}")
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != PRIMITIVE_COLUMNS:
                raise ValueError(f"Shard schema mismatch: {path}")
            rows = [convert_primitive_row(row) for row in reader]
        proteins = validate_target_shard(rows, target_layer, expected_proteins=len(fold_bundle.eligible_protein_ids))
        if proteins != set(fold_bundle.eligible_protein_ids):
            raise ValueError(f"Target layer {target_layer} protein set differs from the frozen cohort")
        primitives.add_shard(target_layer, rows)
        del rows
        runtime_rows.append(_runtime_summary(root, target_layer, primitives))
        log(f"loaded target-layer shard {target_layer}")
    primitives.require_complete()
    return primitives, runtime_rows


def _runtime_summary(root: Path, target_layer: int, primitives: TransferPrimitives) -> dict[str, object]:
    path = root / "layer_shards" / f"runtime_target_layer_{target_layer:02d}.csv"
    elapsed, rss, status = float("nan"), float("nan"), "unknown"
    if path.is_file():
        with path.open(newline="") as handle:
            completes = [row for row in csv.DictReader(handle) if row["stage"] == "target_layer_complete"]
        if completes:
            elapsed = float(completes[-1]["elapsed_seconds"])
            rss = float(completes[-1]["peak_rss_mb"])
            status = completes[-1]["status"]
    t = TRANSFER_LAYERS.index(target_layer)
    return {
        "analysis_name": ANALYSIS_NAME, "target_layer": target_layer,
        "completion_status": status, "n_proteins": len(primitives.protein_ids),
        "n_pairs": int(np.sum(primitives.baseline[t, :, 0])),
        "elapsed_seconds": elapsed, "peak_rss_mb": rss,
    }


def iter_cells():
    for target_layer in TRANSFER_LAYERS:
        for source_layer in TRANSFER_LAYERS:
            for mode in MODES:
                yield source_layer, target_layer, mode


def _indices(source_layer: int, target_layer: int, mode: str) -> tuple[int, int, int]:
    return TRANSFER_LAYERS.index(target_layer), TRANSFER_LAYERS.index(source_layer), MODES.index(mode)


def per_protein_rows(primitives: TransferPrimitives):
    all_cell = CELL_INDEX[("all_pairs", "all")]
    for source_layer, target_layer, mode in iter_cells():
        t, s, m = _indices(source_layer, target_layer, mode)
        transfer = transfer_identity(source_layer, target_layer, mode)
        for p, protein in enumerate(primitives.protein_ids):
            meta = primitives.metadata[protein]
            stats = stats_from_vector(primitives.probe[t, s, m, p, all_cell])
            n_pairs = meta["n_residues"] * (meta["n_residues"] - 1) // 2
            yield {
                "analysis_name": ANALYSIS_BY_MODE[mode], "model_family": MODEL_FAMILY,
                "feature_dim": FEATURE_DIM, "layer": target_layer, "protein_id": protein,
                "source": meta["source"], "cv_protocol_version": CV_PROTOCOL_VERSION,
                "evaluation_design": EVALUATION_DESIGN, "outer_fold": meta["outer_fold"],
                "evaluation_scope": EVALUATION_SCOPE, "oof_prediction_scope": OOF_PREDICTION_SCOPE,
                "prediction_status": "complete", "n_residues": meta["n_residues"],
                "n_protein_pairs": n_pairs, "n_expected_pairs": n_pairs,
                "n_evaluated_pairs": stats.n, "pair_coverage": stats.n / n_pairs,
                **stats.metrics(), **transfer,
            }


def per_protein_subgroup_rows(primitives: TransferPrimitives):
    for source_layer, target_layer, mode in iter_cells():
        t, s, m = _indices(source_layer, target_layer, mode)
        transfer = transfer_identity(source_layer, target_layer, mode)
        for p, protein in enumerate(primitives.protein_ids):
            meta = primitives.metadata[protein]
            for scope in SUBGROUP_SCOPES:
                for label in CATEGORY_LABELS[scope]:
                    vector = primitives.probe[t, s, m, p, CELL_INDEX[(scope, label)]]
                    n_pairs = int(vector[0])
                    sae = float(vector[STAT_FIELDS.index("sum_abs_error")])
                    yield {
                        "analysis_name": ANALYSIS_BY_MODE[mode], "model_family": MODEL_FAMILY,
                        "feature_dim": FEATURE_DIM, "layer": target_layer, "protein_id": protein,
                        "source": meta["source"], "cv_protocol_version": CV_PROTOCOL_VERSION,
                        "evaluation_design": EVALUATION_DESIGN, "outer_fold": meta["outer_fold"],
                        "evaluation_scope": EVALUATION_SCOPE,
                        "oof_prediction_scope": OOF_PREDICTION_SCOPE,
                        "generalization_scope": GENERALIZATION_SCOPE,
                        "category_scope": scope, "category_label": label,
                        "n_residues": meta["n_residues"], "n_pairs": n_pairs,
                        "sum_absolute_error": sae,
                        "mae": sae / n_pairs if n_pairs else float("nan"),
                        "support_status": "complete" if n_pairs else "empty_category",
                        **transfer,
                    }


def comparison_accumulator(primitives: TransferPrimitives, source_layer: int, target_layer: int, mode: str) -> OOFComparisonAccumulator:
    t, s, m = _indices(source_layer, target_layer, mode)
    all_cell = CELL_INDEX[("all_pairs", "all")]
    proteins = []
    for p, protein in enumerate(primitives.protein_ids):
        model = stats_from_vector(primitives.probe[t, s, m, p, all_cell])
        proteins.append({
            "protein_id": protein, "outer_fold": primitives.metadata[protein]["outer_fold"],
            "n_expected_pairs": model.n, "n_evaluated_pairs": model.n, "stable_fit": True,
            "model": model.to_payload(),
            "global_baseline": stats_from_vector(primitives.baseline[t, p]).to_payload(),
            "conditional_baseline": None,
        })
    model_total, baseline_total = RegressionSufficientStatistics(), RegressionSufficientStatistics()
    for row in proteins:
        model_total.merge(RegressionSufficientStatistics.from_payload(row["model"]))
        baseline_total.merge(RegressionSufficientStatistics.from_payload(row["global_baseline"]))
    return OOFComparisonAccumulator.from_payload({
        "model": model_total.to_payload(), "global_baseline": baseline_total.to_payload(),
        "conditional_baseline": None, "proteins": proteins,
    })


def subgroup_accumulator(primitives: TransferPrimitives, source_layer: int, target_layer: int, mode: str, scope: str) -> OOFSubgroupAccumulator:
    t, s, m = _indices(source_layer, target_layer, mode)
    cells = []
    for label in CATEGORY_LABELS[scope]:
        block = primitives.probe[t, s, m, :, CELL_INDEX[(scope, label)]]
        total = RegressionSufficientStatistics()
        supported = []
        for p, protein in enumerate(primitives.protein_ids):
            if block[p, 0] > 0:
                total.merge(stats_from_vector(block[p]))
                supported.append(protein)
        cells.append({"stats": total.to_payload(), "n_expected_pairs": total.n,
                      "protein_ids": supported})
    return OOFSubgroupAccumulator.from_payload({
        "category_scope": scope, "category_labels": list(CATEGORY_LABELS[scope]),
        "seen_proteins": list(primitives.protein_ids), "cells": cells,
    })


def metric_identity(source_layer: int, target_layer: int, mode: str) -> dict[str, object]:
    return {
        "analysis_name": ANALYSIS_BY_MODE[mode], "model_family": MODEL_FAMILY,
        "feature_dim": FEATURE_DIM, "layer": target_layer,
        "evaluation_design": EVALUATION_DESIGN, "oof_prediction_scope": OOF_PREDICTION_SCOPE,
    }


def build_metric_tables(primitives: TransferPrimitives, fold_bundle) -> dict[str, list[dict[str, object]]]:
    tables: dict[str, list[dict[str, object]]] = {
        "pooled": [], "fold": [], "fold_summary": [], "sequence_separation": [],
        "spatial_distance": [],
    }
    fold_proteins = {fold: fold_bundle.protein_ids_for_outer_fold(fold) for fold in range(N_OUTER_FOLDS)}
    for source_layer, target_layer, mode in iter_cells():
        transfer = transfer_identity(source_layer, target_layer, mode)
        identity = metric_identity(source_layer, target_layer, mode)
        accumulator = comparison_accumulator(primitives, source_layer, target_layer, mode)
        pooled = build_pooled_metrics_row(accumulator, global_baseline_type=GLOBAL_BASELINE_TYPE, **identity)
        tables["pooled"].append({**pooled, **transfer})
        fold_rows = [
            build_fold_metrics_row(
                accumulator.subset(fold_proteins[fold]), outer_fold=fold,
                global_baseline_type=GLOBAL_BASELINE_TYPE, **identity,
            )
            for fold in range(N_OUTER_FOLDS)
        ]
        tables["fold"].extend({**row, **transfer} for row in fold_rows)
        tables["fold_summary"].extend({**row, **transfer} for row in build_fold_summary_rows(fold_rows))
        for scope in SUBGROUP_SCOPES:
            rows = subgroup_accumulator(primitives, source_layer, target_layer, mode, scope).rows(**identity)
            tables[scope].extend({**row, **transfer} for row in rows)
    return tables


def transfer_matrix_rows(pooled: Sequence[Mapping[str, object]], per_protein: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    by_cell = {(int(r["source_layer"]), int(r["target_layer"]), str(r["transfer_mode"])): r for r in pooled}
    protein_values: dict[tuple[int, int, str], dict[str, list[float]]] = {}
    for row in per_protein:
        key = (int(row["source_layer"]), int(row["target_layer"]), str(row["transfer_mode"]))
        bucket = protein_values.setdefault(key, {"r2": [], "mae": []})
        bucket["r2"].append(float(row["r2"]))
        bucket["mae"].append(float(row["mae"]))
    rows = []
    for source_layer, target_layer, mode in iter_cells():
        cell = by_cell[(source_layer, target_layer, mode)]
        native_target = by_cell[(target_layer, target_layer, "raw")]
        native_source = by_cell[(source_layer, source_layer, "raw")]
        r2 = np.asarray(protein_values[(source_layer, target_layer, mode)]["r2"])
        mae = np.asarray(protein_values[(source_layer, target_layer, mode)]["mae"])
        finite_r2 = r2[np.isfinite(r2)]
        row = {
            **transfer_identity(source_layer, target_layer, mode),
            "analysis_name": ANALYSIS_BY_MODE[mode], "n_proteins": cell["n_proteins"],
            "n_evaluated_pairs": cell["n_evaluated_pairs"],
            "pooled_r2": cell["pooled_r2"],
            "mean_protein_r2": float(np.mean(finite_r2)) if finite_r2.size else float("nan"),
            "median_protein_r2": float(np.median(finite_r2)) if finite_r2.size else float("nan"),
            "n_finite_r2": int(finite_r2.size),
            "pooled_mae": cell["pooled_mae"], "mean_protein_mae": float(np.mean(mae)),
            "median_protein_mae": float(np.median(mae)), "pooled_rmse": cell["pooled_rmse"],
            "pooled_bias_pred_minus_true": cell["pooled_bias_pred_minus_true"],
        }
        for metric in MATRIX_METRICS:
            value = float(cell[f"pooled_{metric}"])
            row[f"native_target_pooled_{metric}"] = float(native_target[f"pooled_{metric}"])
            row[f"delta_pooled_vs_native_target_{metric}"] = value - float(native_target[f"pooled_{metric}"])
            row[f"native_source_pooled_{metric}"] = float(native_source[f"pooled_{metric}"])
            row[f"delta_pooled_vs_native_source_{metric}"] = value - float(native_source[f"pooled_{metric}"])
        rows.append(row)
    return rows


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _compare(check: str, layer: object, metric: str, pairs: Sequence[tuple[float, float]], detail: str = "") -> dict[str, object]:
    worst, failed = 0.0, False
    for expected, observed in pairs:
        expected, observed = float(expected), float(observed)
        if math.isnan(expected) or math.isnan(observed):
            failed |= math.isnan(expected) != math.isnan(observed)
            continue
        diff = abs(expected - observed)
        worst = max(worst, diff)
        failed |= diff > PARITY_TOLERANCE * max(1.0, abs(expected))
    if not pairs:
        failed = True
        detail = detail or "nothing compared"
    return {
        "check": check, "layer": layer, "metric": metric, "n_compared": len(pairs),
        "max_abs_diff": worst, "tolerance": f"{PARITY_TOLERANCE:g}*max(1,|expected|)",
        "status": "fail" if failed else "pass", "detail": detail,
    }


def parity_checks(
    producer_family: Path,
    per_protein: Sequence[Mapping[str, object]],
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    primitives: TransferPrimitives,
    baseline_consistent: bool,
) -> list[dict[str, object]]:
    checks: list[dict[str, object]] = []
    layers = set(TRANSFER_LAYERS)

    def diagonal(rows: Iterable[Mapping[str, object]]):
        return [r for r in rows if r["transfer_mode"] == "raw" and int(r["source_layer"]) == int(r["target_layer"])]

    producer_pp = {
        (int(r["layer"]), r["protein_id"]): r
        for r in _read_csv(producer_family / "per_protein.csv") if int(r["layer"]) in layers
    }
    ours_pp = {(int(r["layer"]), str(r["protein_id"])): r for r in diagonal(per_protein)}
    for layer in TRANSFER_LAYERS:
        keys = sorted(k for k in ours_pp if k[0] == layer)
        expected_keys = sorted(k for k in producer_pp if k[0] == layer)
        detail = "" if keys == expected_keys else "protein sets differ"
        for metric in ("r2", "mae", "rmse", "bias_pred_minus_true", "mean_true_distance",
                       "mean_predicted_distance", "target_variance", "n_evaluated_pairs"):
            pairs = [(producer_pp[k][metric], ours_pp[k][metric]) for k in keys if k in producer_pp]
            row = _compare("diagonal_per_protein_vs_producer", layer, metric, pairs, detail)
            if detail:
                row["status"] = "fail"
            checks.append(row)

    value_metrics = (
        "n_proteins", "n_evaluated_pairs", "pooled_r2", "pooled_mae", "pooled_rmse",
        "pooled_bias_pred_minus_true", "global_baseline_pooled_r2", "global_baseline_pooled_mae",
        "global_baseline_pooled_rmse", "global_baseline_pooled_bias_pred_minus_true",
    )
    producer_pooled = {int(r["layer"]): r for r in _read_csv(producer_family / "pooled_metrics.csv")}
    ours_pooled = {int(r["layer"]): r for r in diagonal(tables["pooled"])}
    for layer in TRANSFER_LAYERS:
        for metric in value_metrics:
            checks.append(_compare("diagonal_pooled_vs_producer", layer, metric,
                                   [(producer_pooled[layer][metric], ours_pooled[layer][metric])]))

    producer_fold = {(int(r["layer"]), int(r["outer_fold"])): r for r in _read_csv(producer_family / "fold_metrics.csv")}
    ours_fold = {(int(r["layer"]), int(r["outer_fold"])): r for r in diagonal(tables["fold"])}
    for layer in TRANSFER_LAYERS:
        for metric in value_metrics:
            pairs = [(producer_fold[(layer, f)][metric], ours_fold[(layer, f)][metric]) for f in range(N_OUTER_FOLDS)]
            checks.append(_compare("diagonal_fold_vs_producer", layer, metric, pairs))

    for scope, filename in (("sequence_separation", "by_sequence_separation.csv"),
                            ("spatial_distance", "by_spatial_distance.csv")):
        producer_sub = {(int(r["layer"]), r["category_label"]): r
                        for r in _read_csv(producer_family / "subgroup_metrics" / filename)}
        ours_sub = {(int(r["layer"]), str(r["category_label"])): r for r in diagonal(tables[scope])}
        for layer in TRANSFER_LAYERS:
            for metric in ("n_proteins", "n_evaluated_pairs", "pooled_r2", "pooled_mae",
                           "pooled_rmse", "pooled_bias_pred_minus_true"):
                pairs = [(producer_sub[(layer, label)][metric], ours_sub[(layer, label)][metric])
                         for label in CATEGORY_LABELS[scope]]
                checks.append(_compare(f"diagonal_subgroup_{scope}_vs_producer", layer, metric, pairs))
            status_match = all(
                producer_sub[(layer, label)]["support_status"] == ours_sub[(layer, label)]["support_status"]
                for label in CATEGORY_LABELS[scope]
            )
            checks.append({
                "check": f"diagonal_subgroup_{scope}_vs_producer", "layer": layer,
                "metric": "support_status", "n_compared": len(CATEGORY_LABELS[scope]),
                "max_abs_diff": 0.0, "tolerance": "exact",
                "status": "pass" if status_match else "fail", "detail": "",
            })

    raw_index, restd_index = MODES.index("raw"), MODES.index("target_restandardized")
    for t, layer in enumerate(TRANSFER_LAYERS):
        same = np.array_equal(primitives.probe[t, t, raw_index], primitives.probe[t, t, restd_index])
        checks.append({
            "check": "diagonal_restandardized_equals_raw", "layer": layer, "metric": "all_primitives",
            "n_compared": int(primitives.probe[t, t, raw_index].size), "max_abs_diff": float(
                np.max(np.abs(primitives.probe[t, t, raw_index] - primitives.probe[t, t, restd_index]))
            ), "tolerance": "exact", "status": "pass" if same else "fail", "detail": "",
        })
    checks.append({
        "check": "global_baseline_identical_across_target_shards", "layer": "", "metric": "all_primitives",
        "n_compared": int(primitives.baseline.size), "max_abs_diff": 0.0, "tolerance": "1e-12 relative",
        "status": "pass" if baseline_consistent else "fail", "detail": "",
    })
    return checks


def baseline_consistent(primitives: TransferPrimitives) -> bool:
    reference = primitives.baseline[0]
    return all(
        np.allclose(primitives.baseline[t], reference, rtol=1e-12, atol=0.0)
        for t in range(1, len(TRANSFER_LAYERS))
    )


def main() -> None:
    started = time.perf_counter()
    root = output_root()
    assert_writable(root)
    fold_bundle = load_cv_folds(metadata_dir=producer_root() / "metadata")
    if len(fold_bundle.eligible_protein_ids) != EXPECTED_PROTEINS:
        raise ValueError("Frozen fivefold cohort size changed")
    _, provenance, status_rows = load_transfer_artifacts(producer_root(), fold_bundle)
    primitives, runtime_rows = load_primitives(root, fold_bundle)
    consistent = baseline_consistent(primitives)
    log("validated all target-layer shards")

    per_protein = list(per_protein_rows(primitives))
    stream_write_csv(root / "per_protein.csv", PER_PROTEIN_OUTPUT_COLUMNS, per_protein)
    n_subgroup = stream_write_csv(
        root / "per_protein_subgroup_mae.csv", PER_PROTEIN_SUBGROUP_COLUMNS,
        per_protein_subgroup_rows(primitives),
    )
    log("wrote per-protein tables")
    tables = build_metric_tables(primitives, fold_bundle)
    atomic_write_csv(root / "pooled_metrics.csv", POOLED_OUTPUT_COLUMNS, tables["pooled"])
    atomic_write_csv(root / "fold_metrics.csv", FOLD_OUTPUT_COLUMNS, tables["fold"])
    atomic_write_csv(root / "fold_summary.csv", FOLD_SUMMARY_OUTPUT_COLUMNS, tables["fold_summary"])
    atomic_write_csv(root / "subgroup_metrics/by_sequence_separation.csv", SUBGROUP_OUTPUT_COLUMNS,
                     tables["sequence_separation"])
    atomic_write_csv(root / "subgroup_metrics/by_spatial_distance.csv", SUBGROUP_OUTPUT_COLUMNS,
                     tables["spatial_distance"])
    matrix = transfer_matrix_rows(tables["pooled"], per_protein)
    atomic_write_csv(root / "transfer_matrix.csv", TRANSFER_MATRIX_COLUMNS, matrix)
    atomic_write_csv(root / "source_probe_status.csv", SOURCE_STATUS_COLUMNS, status_rows)
    atomic_write_csv(root / "runtime.csv", RUNTIME_OUTPUT_COLUMNS, runtime_rows)
    log("wrote metric tables")

    producer_family = producer_root() / PRODUCER_SPEC.family_dir
    checks = parity_checks(producer_family, per_protein, tables, primitives, consistent)
    atomic_write_csv(root / "parity_checks.csv", PARITY_COLUMNS, checks)
    failed = [c for c in checks if c["status"] != "pass"]
    status = "complete" if not failed else "failed_parity"
    counts = {
        "per_protein.csv": len(per_protein), "per_protein_subgroup_mae.csv": n_subgroup,
        "pooled_metrics.csv": len(tables["pooled"]), "fold_metrics.csv": len(tables["fold"]),
        "fold_summary.csv": len(tables["fold_summary"]),
        "subgroup_metrics/by_sequence_separation.csv": len(tables["sequence_separation"]),
        "subgroup_metrics/by_spatial_distance.csv": len(tables["spatial_distance"]),
        "transfer_matrix.csv": len(matrix), "source_probe_status.csv": len(status_rows),
        "parity_checks.csv": len(checks), "runtime.csv": len(runtime_rows),
    }
    atomic_write_json(root / "run_manifest.json", {
        "analysis_name": ANALYSIS_NAME, "analysis_status": status, "created_at": utc_now(),
        "description": (
            "Frozen protocol-2.0 shared-OLS probes replayed across Evoformer layers without "
            "refitting; fold-f probes evaluate fold-f held-out proteins at the target layer."
        ),
        "producer_analysis": PRODUCER_SPEC.analysis_name, "producer": provenance,
        "metadata_digests": dict(fold_bundle.fold_digests),
        "mid_layers": list(MID_LAYERS), "end_layers": list(END_LAYERS),
        "transfer_layers": list(TRANSFER_LAYERS), "transfer_modes": list(MODES),
        "analysis_name_by_mode": dict(ANALYSIS_BY_MODE),
        "mode_definitions": {
            "raw": "source intercept + ((x - source_mean)/source_scale) @ source_coef_std",
            "target_restandardized": (
                "source intercept + ((x - target_mean)/target_scale) @ source_coef_std; target "
                "scaler is the same-fold train-only scaler from the target-layer artifact"
            ),
        },
        "layer_column_definition": "layer equals target_layer (the evaluated representation)",
        "evaluation_scope": EVALUATION_SCOPE, "oof_prediction_scope": OOF_PREDICTION_SCOPE,
        "cv_protocol_version": CV_PROTOCOL_VERSION, "feature_dim": FEATURE_DIM,
        "expected_proteins": EXPECTED_PROTEINS, "row_counts": counts,
        "geometry_metrics": "not_computed_by_request",
        "stable_fit_metrics": "omitted; all source fits are listed in source_probe_status.csv",
        "parity": {"n_checks": len(checks), "n_failed": len(failed),
                   "failed": [f"{c['check']}:{c['layer']}:{c['metric']}" for c in failed[:50]]},
        "code_revision": deployment_revision(SCRIPT_DIR),
        "aggregator_sha256": sha256_file(Path(__file__)),
        "aggregation_elapsed_seconds": time.perf_counter() - started,
    })
    normalize_shared_tree(root)
    log(f"aggregation {status}; {len(failed)} failed parity checks")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

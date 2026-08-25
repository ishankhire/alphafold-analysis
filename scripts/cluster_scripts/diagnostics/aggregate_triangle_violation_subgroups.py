#!/usr/bin/env python3
"""Validate and aggregate exact triangle-violation subgroup layer shards."""

from __future__ import annotations

import csv
import itertools
import math
import os
import resource
import sys
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np
from scipy.stats import spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, CLUSTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from regression_output_helpers import atomic_write_csv, atomic_write_json  # noqa: E402
from triangle_violation_subgroup_common import (  # noqa: E402
    ANALYSIS_NAME,
    CATEGORY_LABELS,
    DERIVED_METRICS,
    EVALUATION_SCOPE,
    LENGTH_BIN_SUMMARY_COLUMNS,
    LENGTH_EFFECT_COLUMNS,
    MODEL_IDENTITIES,
    PAIRED_CONTRAST_COLUMNS,
    PER_PROTEIN_COLUMNS,
    RUNTIME_COLUMNS,
    SUMMARY_COLUMNS,
    THRESHOLD_COLUMNS,
    load_json,
    metric_value,
    sha256_file,
)


FIGURE5_COLUMNS = (
    "analysis_name", "layer", "protein_id", "metric", "producer_value",
    "replay_value", "absolute_difference", "tolerance", "status",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def output_root() -> Path:
    csv_root = Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))).expanduser()
    return csv_root / ANALYSIS_NAME


def read_csv(path: Path, expected: Sequence[str] | None = None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if expected is not None and tuple(reader.fieldnames or ()) != tuple(expected):
            raise ValueError(f"Schema mismatch in {path}: {reader.fieldnames}")
        return list(reader)


def convert_metric_row(source: Mapping[str, str]) -> dict[str, object]:
    row: dict[str, object] = dict(source)
    integer_columns = {
        "layer", "n_residues", "n_triangles", "n_violations",
        "n_negative_predictions", "n_abs_0_5", "n_abs_5_10", "n_abs_10_plus",
        "n_rel_0_0p05", "n_rel_0p05_0p10", "n_rel_0p10_plus",
    }
    identity_columns = {
        "analysis_name", "model_family", "protein_id", "source", "protein_split",
        "evaluation_scope", "generalization_scope", "category_system",
        "category_label", "protein_length_bin", "support_status",
    }
    for column in PER_PROTEIN_COLUMNS:
        if column in integer_columns:
            row[column] = int(source[column])
        elif column not in identity_columns:
            row[column] = float(source[column])
    return row


def validate_complete_rows(rows: Sequence[Mapping[str, object]]) -> None:
    expected_analyses = {name for name, _, _ in MODEL_IDENTITIES}
    expected_per_layer = 111 * len(MODEL_IDENTITIES) * sum(map(len, CATEGORY_LABELS.values()))
    if any(row["evaluation_scope"] != EVALUATION_SCOPE for row in rows):
        raise ValueError("Only all_edges is permitted")
    if {str(row["analysis_name"]) for row in rows} != expected_analyses:
        raise ValueError("Aggregated model coverage differs from the four frozen probes")
    for layer in range(48):
        layer_rows = [row for row in rows if int(row["layer"]) == layer]
        if len(layer_rows) != expected_per_layer:
            raise ValueError(f"Layer {layer} has {len(layer_rows)} rows, expected {expected_per_layer}")

    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(row["analysis_name"], row["layer"], row["protein_id"])].append(row)
    expected_groups = 48 * 4 * 111
    if len(grouped) != expected_groups:
        raise ValueError(f"Observed {len(grouped)} model-layer-protein groups, expected {expected_groups}")
    for key, group in grouped.items():
        by_system: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in group:
            by_system[str(row["category_system"])].append(row)
        if set(by_system) != set(CATEGORY_LABELS):
            raise ValueError(f"Category system coverage mismatch for {key}")
        all_rows = by_system["all_triangles"]
        if len(all_rows) != 1:
            raise ValueError(f"All-triangle row is not unique for {key}")
        n_all = int(all_rows[0]["n_triangles"])
        for system, labels in CATEGORY_LABELS.items():
            system_rows = by_system[system]
            if {row["category_label"] for row in system_rows} != set(labels):
                raise ValueError(f"Category labels mismatch for {key}/{system}")
            if sum(int(row["n_triangles"]) for row in system_rows) != n_all:
                raise ValueError(f"Category counts do not partition triangles for {key}/{system}")
            for row in system_rows:
                nv = int(row["n_violations"])
                if sum(int(row[name]) for name in ("n_abs_0_5", "n_abs_5_10", "n_abs_10_plus")) != nv:
                    raise ValueError("Absolute bands do not sum to n_violations")
                if sum(int(row[name]) for name in ("n_rel_0_0p05", "n_rel_0p05_0p10", "n_rel_0p10_plus")) != nv:
                    raise ValueError("Relative bands do not sum to n_violations")


def descriptive(values: Iterable[float]) -> dict[str, object]:
    all_values = np.asarray(list(values), dtype=np.float64)
    finite = all_values[np.isfinite(all_values)]
    if finite.size == 0:
        return {
            "protein_equal_mean": float("nan"), "protein_std": float("nan"),
            "protein_median": float("nan"), "protein_q25": float("nan"),
            "protein_q75": float("nan"), "n_proteins_total": all_values.size,
            "n_proteins_finite": 0,
        }
    q25, median, q75 = np.quantile(finite, (0.25, 0.5, 0.75))
    return {
        "protein_equal_mean": float(np.mean(finite)),
        "protein_std": float(np.std(finite, ddof=0)),
        "protein_median": float(median), "protein_q25": float(q25),
        "protein_q75": float(q75), "n_proteins_total": all_values.size,
        "n_proteins_finite": finite.size,
    }


def pooled_value(rows: Sequence[Mapping[str, object]], metric: str) -> tuple[float, str]:
    n = sum(int(row["n_triangles"]) for row in rows)
    nv = sum(int(row["n_violations"]) for row in rows)
    formulas = {
        "violation_fraction": (sum(int(row["n_violations"]) for row in rows), n),
        "mean_violation_all": (sum(float(row["violation_sum_angstrom"]) for row in rows), n),
        "mean_violation_conditional": (
            sum(float(row["violation_sum_angstrom"]) for row in rows), nv,
        ),
        "mean_relative_violation_all": (
            sum(float(row["relative_violation_sum"]) for row in rows), n,
        ),
        "mean_relative_violation_conditional": (
            sum(float(row["relative_violation_sum"]) for row in rows), nv,
        ),
    }
    band_columns = {
        "abs_0_5_fraction_all": "n_abs_0_5", "abs_5_10_fraction_all": "n_abs_5_10",
        "abs_10_plus_fraction_all": "n_abs_10_plus",
        "rel_0_0p05_fraction_all": "n_rel_0_0p05",
        "rel_0p05_0p10_fraction_all": "n_rel_0p05_0p10",
        "rel_0p10_plus_fraction_all": "n_rel_0p10_plus",
    }
    if metric in formulas:
        numerator, denominator = formulas[metric]
        return (float(numerator / denominator), "complete") if denominator else (float("nan"), "zero_denominator")
    if metric in band_columns:
        return (
            (sum(int(row[band_columns[metric]]) for row in rows) / n, "complete")
            if n else (float("nan"), "zero_denominator")
        )
    if metric == "relative_violation_max":
        values = [float(row[metric]) for row in rows if int(row["n_violations"]) > 0]
        return (max(values), "complete") if values else (float("nan"), "zero_violations")
    if metric == "relative_violation_p99_conditional":
        return float("nan"), "not_reconstructible_without_triangle_rows"
    raise KeyError(metric)


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = (
        "analysis_name", "model_family", "layer", "evaluation_scope",
        "category_system", "category_label",
    )
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    summaries = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        identity = dict(zip(keys, key))
        for metric in DERIVED_METRICS:
            pooled, pooled_status = pooled_value(group, metric)
            summaries.append({
                **identity, "metric": metric,
                **descriptive(metric_value(row, metric) for row in group),
                "triangle_pooled_value": pooled,
                "triangle_pooled_status": pooled_status,
            })
    return summaries


def summarize_length_bins(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[str(row["protein_length_bin"])].append(row)
    output = []
    for length_bin, group in sorted(grouped.items()):
        output.extend({**row, "protein_length_bin": length_bin} for row in summarize_rows(group))
    return output


def add_reference_length_composition(
    output: list[dict[str, object]], threshold_rows: Sequence[Mapping[str, str]],
) -> None:
    numeric_metrics = (
        "rms_tertile_1", "rms_tertile_2", "area_tertile_1", "area_tertile_2",
        "slack_tertile_1", "slack_tertile_2", "rms_median", "rms_p90",
        "area_median", "area_p90", "slack_median", "slack_p90",
    )
    for length_bin in sorted({row["protein_length_bin"] for row in threshold_rows}):
        group = [row for row in threshold_rows if row["protein_length_bin"] == length_bin]
        for metric in numeric_metrics:
            output.append({
                "analysis_name": "reference_geometry", "model_family": "reference_geometry",
                "layer": -1, "evaluation_scope": EVALUATION_SCOPE,
                "category_system": "reference_distribution", "category_label": "all",
                "metric": metric, **descriptive(float(row[metric]) for row in group),
                "triangle_pooled_value": float("nan"),
                "triangle_pooled_status": "not_applicable",
                "protein_length_bin": length_bin,
            })
        angle_columns = {
            "clearly_acute": "n_clearly_acute", "near_right": "n_near_right",
            "clearly_obtuse": "n_clearly_obtuse",
        }
        for label, column in angle_columns.items():
            output.append({
                "analysis_name": "reference_geometry", "model_family": "reference_geometry",
                "layer": -1, "evaluation_scope": EVALUATION_SCOPE,
                "category_system": "angle_shape", "category_label": label,
                "metric": "category_fraction",
                **descriptive(float(row[column]) / float(row["n_triangles"]) for row in group),
                "triangle_pooled_value": (
                    sum(int(row[column]) for row in group)
                    / sum(int(row["n_triangles"]) for row in group)
                ),
                "triangle_pooled_status": "complete",
                "protein_length_bin": length_bin,
            })


def length_effects(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    keys = (
        "analysis_name", "model_family", "layer", "evaluation_scope",
        "category_system", "category_label",
    )
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in keys)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        for metric in DERIVED_METRICS:
            pairs = [
                (math.log(int(row["n_residues"])), metric_value(row, metric))
                for row in group
            ]
            pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
            if len(pairs) >= 3:
                x = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
                y = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
                x_mean, x_std = float(np.mean(x)), float(np.std(x, ddof=0))
                if x_std > 0.0:
                    z = (x - x_mean) / x_std
                    slope, intercept = np.polyfit(z, y, 1)
                    if float(np.ptp(y)) <= 1e-15:
                        rho = float("nan")
                        status = "constant_outcome"
                    else:
                        rho = float(spearmanr(x, y).statistic)
                        status = "complete" if math.isfinite(rho) else "undefined_spearman"
                else:
                    slope = intercept = rho = float("nan")
                    status = "constant_length"
            else:
                x_mean = x_std = slope = intercept = rho = float("nan")
                status = "insufficient_finite_proteins"
            output.append({
                **dict(zip(keys, key)), "metric": metric, "n_proteins": len(pairs),
                "log_length_mean": x_mean, "log_length_std": x_std,
                "standardized_log_length_slope": float(slope),
                "regression_intercept": float(intercept), "spearman_rho": rho,
                "association_status": status,
            })
    return output


def paired_contrasts(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    row_map = {
        (
            row["analysis_name"], row["layer"], row["protein_id"],
            row["evaluation_scope"], row["category_system"], row["category_label"],
        ): row
        for row in rows
    }
    analyses = [name for name, _, _ in MODEL_IDENTITIES]
    output = []
    protein_ids = sorted({str(row["protein_id"]) for row in rows})
    for layer in range(48):
        for system, labels in CATEGORY_LABELS.items():
            for label in labels:
                for analysis_a, analysis_b in itertools.combinations(analyses, 2):
                    for metric in DERIVED_METRICS:
                        differences = []
                        for protein_id in protein_ids:
                            common = (layer, protein_id, EVALUATION_SCOPE, system, label)
                            a = row_map[(analysis_a, *common)]
                            b = row_map[(analysis_b, *common)]
                            av, bv = metric_value(a, metric), metric_value(b, metric)
                            if math.isfinite(av) and math.isfinite(bv):
                                differences.append(av - bv)
                        stats = descriptive(differences)
                        output.append({
                            "layer": layer, "evaluation_scope": EVALUATION_SCOPE,
                            "category_system": system, "category_label": label,
                            "metric": metric, "analysis_name_a": analysis_a,
                            "analysis_name_b": analysis_b,
                            "n_paired_proteins": stats["n_proteins_finite"],
                            "mean_difference_a_minus_b": stats["protein_equal_mean"],
                            "std_difference": stats["protein_std"],
                            "median_difference": stats["protein_median"],
                            "q25_difference": stats["protein_q25"],
                            "q75_difference": stats["protein_q75"],
                            "contrast_status": "complete" if differences else "no_finite_pairs",
                        })
    return output


def figure5_reproduction(
    rows: Sequence[Mapping[str, object]], provenance: Mapping[str, object],
) -> list[dict[str, object]]:
    producer_roots = {
        "general_base256_combined": Path(provenance["shared_ols"]["root"]),
        "per_protein_base256_combined": Path(provenance["per_protein_ols"]["root"]),
        "shared_xgboost_base256_combined": Path(provenance["shared_xgboost"]["root"]),
        "analysis_1_sequence_gated": Path(provenance["sequence_gated_ols"]["root"]),
    }
    replay_by_key = {
        (row["analysis_name"], int(row["layer"]), row["protein_id"]): row
        for row in rows if row["category_system"] == "all_triangles"
    }
    metric_map = {
        "n_negative_predictions": ("n_negative_predictions", 0.0),
        "n_triangles": ("n_triangles", 0.0),
        "triangle_violation_fraction": ("violation_fraction", 1e-12),
        "triangle_mean_violation": ("mean_violation_all", 1e-10),
        "triangle_max_violation": ("violation_max_angstrom", 1e-10),
        "triangle_violation_0_5_fraction": ("n_abs_0_5", 1e-12),
        "triangle_violation_5_10_fraction": ("n_abs_5_10", 1e-12),
        "triangle_violation_10_plus_fraction": ("n_abs_10_plus", 1e-12),
    }
    output = []
    for analysis_name, root in producer_roots.items():
        producer_rows = read_csv(root / "geometry_metrics.csv")
        for producer in producer_rows:
            if producer["protein_split"] != "test" or producer["evaluation_scope"] != "all_pairs":
                continue
            key = (analysis_name, int(producer["layer"]), producer["protein_id"])
            replay = replay_by_key[key]
            for producer_metric, (replay_column, tolerance) in metric_map.items():
                producer_value = float(producer[producer_metric])
                if replay_column in {"n_abs_0_5", "n_abs_5_10", "n_abs_10_plus"}:
                    replay_value = float(replay[replay_column]) / int(replay["n_triangles"])
                else:
                    replay_value = float(replay[replay_column])
                difference = abs(producer_value - replay_value)
                output.append({
                    "analysis_name": analysis_name, "layer": key[1],
                    "protein_id": key[2], "metric": producer_metric,
                    "producer_value": producer_value, "replay_value": replay_value,
                    "absolute_difference": difference, "tolerance": tolerance,
                    "status": "pass" if difference <= tolerance else "fail",
                })
    failures = [row for row in output if row["status"] != "pass"]
    if failures:
        example = failures[0]
        raise ValueError(f"Figure-5 reproduction failed ({len(failures)} rows); first={example}")
    return output


def main() -> None:
    started = time.perf_counter()
    root = output_root()
    shard_rows = []
    provenance_by_layer = []
    runtime_rows = []
    for layer in range(48):
        shard = root / f"layer_shards/per_protein_triangle_metrics_layer_{layer:02d}.csv"
        shard_rows.extend(convert_metric_row(row) for row in read_csv(shard, PER_PROTEIN_COLUMNS))
        provenance_by_layer.append(
            load_json(root / f"layer_shards/artifact_provenance_layer_{layer:02d}.json")
        )
        runtime_rows.extend(
            read_csv(root / f"layer_shards/runtime_layer_{layer:02d}.csv", RUNTIME_COLUMNS)
        )
    validate_complete_rows(shard_rows)
    log(f"validated {len(shard_rows)} per-protein category rows")

    thresholds = read_csv(root / "metadata/protein_reference_thresholds.csv", THRESHOLD_COLUMNS)
    summaries = summarize_rows(shard_rows)
    length_summaries = summarize_length_bins(shard_rows)
    add_reference_length_composition(length_summaries, thresholds)
    effects = length_effects(shard_rows)
    contrasts = paired_contrasts(shard_rows)
    reproduction = figure5_reproduction(shard_rows, provenance_by_layer[0])

    atomic_write_csv(root / "per_protein_triangle_metrics.csv", PER_PROTEIN_COLUMNS, shard_rows)
    atomic_write_csv(root / "triangle_metric_summary.csv", SUMMARY_COLUMNS, summaries)
    atomic_write_csv(root / "length_bin_summary.csv", LENGTH_BIN_SUMMARY_COLUMNS, length_summaries)
    atomic_write_csv(root / "protein_length_effects.csv", LENGTH_EFFECT_COLUMNS, effects)
    atomic_write_csv(root / "paired_model_contrasts.csv", PAIRED_CONTRAST_COLUMNS, contrasts)
    atomic_write_csv(root / "runtime.csv", RUNTIME_COLUMNS, runtime_rows)
    atomic_write_csv(root / "metadata/figure5_reproduction.csv", FIGURE5_COLUMNS, reproduction)
    artifact_provenance = {
        "master_manifest_digest": provenance_by_layer[0]["master_manifest_digest"],
        "layers": provenance_by_layer,
        "validation": {
            "producer_manifests": "complete_48_layer_feature_order_and_split_digest",
            "coefficient_shards": "sha256_verified_from_index",
            "figure5_reproduction": "pass",
        },
    }
    atomic_write_json(root / "metadata/artifact_provenance.json", artifact_provenance)

    manifest = load_json(root / "run_manifest.json")
    output_paths = (
        "per_protein_triangle_metrics.csv", "triangle_metric_summary.csv",
        "length_bin_summary.csv", "protein_length_effects.csv",
        "paired_model_contrasts.csv", "runtime.csv",
        "metadata/artifact_provenance.json", "metadata/figure5_reproduction.csv",
        "metadata/split_assignments.csv", "metadata/protein_reference_thresholds.csv",
        "metadata/preparation_failures.csv",
    ) + tuple(
        relative
        for layer in range(48)
        for relative in (
            f"layer_shards/per_protein_triangle_metrics_layer_{layer:02d}.csv",
            f"layer_shards/runtime_layer_{layer:02d}.csv",
            f"layer_shards/artifact_provenance_layer_{layer:02d}.json",
        )
    )
    manifest.update({
        "run_status": "complete", "completed_layers": list(range(48)),
        "finalized_at": utc_now(), "bootstrap_resampling": False,
        "aggregation": {
            "protein_rows_retained_separately": True,
            "protein_equal_summaries": True,
            "triangle_pooled_sensitivity": True,
            "conditional_p99_triangle_pooled_status": "not_reconstructible_without_triangle_rows",
            "figure5_reproduction_status": "pass",
        },
        "aggregation_elapsed_seconds": time.perf_counter() - started,
        "aggregation_peak_rss_mb": peak_rss_mb(),
        "artifact_sha256": {relative: sha256_file(root / relative) for relative in output_paths},
    })
    atomic_write_json(root / "run_manifest.json", manifest)
    log(
        f"aggregation complete summaries={len(summaries)} effects={len(effects)} "
        f"contrasts={len(contrasts)} elapsed={time.perf_counter() - started:.1f}s"
    )


if __name__ == "__main__":
    main()

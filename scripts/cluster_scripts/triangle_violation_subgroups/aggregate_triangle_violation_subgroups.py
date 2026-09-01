#!/usr/bin/env python3
"""Validate and stream-aggregate fivefold OOF triangle subgroup shards."""

from __future__ import annotations

import csv
import itertools
import math
import os
import resource
import sys
import tempfile
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
    CV_PROTOCOL_VERSION,
    DERIVED_METRICS,
    EDGE_CONTRAST_COLUMNS,
    EDGE_MODEL_COUNT_LABELS,
    EVALUATION_SCOPE,
    EXPECTED_PROTEINS,
    FAMILY_BY_NAME,
    FAMILY_SPECS,
    GEOMETRY_REPRODUCTION_FAILURE_COLUMNS,
    GEOMETRY_REPRODUCTION_SUMMARY_COLUMNS,
    LENGTH_BIN_SUMMARY_COLUMNS,
    LENGTH_EFFECT_COLUMNS,
    LAYERS,
    PAIRED_CONTRAST_COLUMNS,
    PER_PROTEIN_COLUMNS,
    RUNTIME_COLUMNS,
    SUMMARY_COLUMNS,
    load_json,
    metric_value,
    read_csv_strict,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def csv_root() -> Path:
    return Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))).expanduser()


def producer_root() -> Path:
    return csv_root() / "regression_5fold_v1"


def output_root() -> Path:
    return csv_root() / "analysis" / ANALYSIS_NAME


INTEGER_COLUMNS = {
    "feature_dim", "layer", "outer_fold", "n_residues", "n_triangles", "n_violations",
    "n_negative_predictions", "n_abs_0_5", "n_abs_5_10", "n_abs_10_plus",
    "n_rel_0_0p05", "n_rel_0p05_0p10", "n_rel_0p10_plus",
}
IDENTITY_COLUMNS = {
    "analysis_name", "model_family", "protein_id", "source", "cv_protocol_version",
    "evaluation_design", "evaluation_scope", "oof_prediction_scope",
    "generalization_scope", "category_system", "category_label", "protein_length_bin",
    "support_status",
}


def convert_metric_row(source: Mapping[str, str]) -> dict[str, object]:
    row: dict[str, object] = {}
    for column in PER_PROTEIN_COLUMNS:
        if column in INTEGER_COLUMNS:
            row[column] = int(source[column])
        elif column in IDENTITY_COLUMNS:
            row[column] = source[column]
        else:
            row[column] = float(source[column])
    return row


def validate_complete_layer(
    rows: Sequence[Mapping[str, object]], layer: int, *,
    expected_proteins: int = EXPECTED_PROTEINS,
) -> None:
    expected_rows = int(expected_proteins) * 103
    if len(rows) != expected_rows:
        raise ValueError(f"Layer {layer} has {len(rows)} rows, expected {expected_rows}")
    if any(
        int(row["layer"]) != layer
        or row["evaluation_scope"] != EVALUATION_SCOPE
        or row["cv_protocol_version"] != CV_PROTOCOL_VERSION
        for row in rows
    ):
        raise ValueError(f"Layer {layer} has wrong identity/scope values")
    grouped: dict[tuple[str, str], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[(str(row["analysis_name"]), str(row["protein_id"]))].append(row)
    if len(grouped) != 4 * expected_proteins:
        raise ValueError(f"Layer {layer} family/protein coverage is incomplete")
    proteins_by_family: dict[str, set[str]] = defaultdict(set)
    for (analysis_name, protein_id), group in grouped.items():
        if analysis_name not in FAMILY_BY_NAME:
            raise ValueError(f"Unknown analysis {analysis_name}")
        proteins_by_family[analysis_name].add(protein_id)
        expected_systems = set(CATEGORY_LABELS)
        if analysis_name == "per_protein_ols_base256":
            expected_systems.add("edge_model_count")
        by_system: dict[str, list[Mapping[str, object]]] = defaultdict(list)
        for row in group:
            by_system[str(row["category_system"])].append(row)
        if set(by_system) != expected_systems:
            raise ValueError(f"Category-system coverage mismatch for {analysis_name}/{protein_id}")
        all_rows = by_system["all_triangles"]
        if len(all_rows) != 1:
            raise ValueError("All-triangle row is not unique")
        n_all = int(all_rows[0]["n_triangles"])
        for system, system_rows in by_system.items():
            labels = EDGE_MODEL_COUNT_LABELS if system == "edge_model_count" else CATEGORY_LABELS[system]
            if {row["category_label"] for row in system_rows} != set(labels):
                raise ValueError(f"Category-label coverage mismatch for {analysis_name}/{protein_id}/{system}")
            if sum(int(row["n_triangles"]) for row in system_rows) != n_all:
                raise ValueError(f"Categories do not partition triangles for {analysis_name}/{protein_id}/{system}")
            for row in system_rows:
                nv = int(row["n_violations"])
                if sum(int(row[name]) for name in ("n_abs_0_5", "n_abs_5_10", "n_abs_10_plus")) != nv:
                    raise ValueError("Absolute severity bands do not sum to n_violations")
                if sum(int(row[name]) for name in ("n_rel_0_0p05", "n_rel_0p05_0p10", "n_rel_0p10_plus")) != nv:
                    raise ValueError("Relative severity bands do not sum to n_violations")
                if int(row["n_triangles"]) and nv:
                    if not math.isclose(
                        float(row["mean_violation_all"]),
                        float(row["violation_fraction"]) * float(row["mean_violation_conditional"]),
                        rel_tol=1e-12, abs_tol=1e-12,
                    ):
                        raise ValueError("Absolute all/conditional mean identity failed")
                    if not math.isclose(
                        float(row["mean_relative_violation_all"]),
                        float(row["violation_fraction"]) * float(row["mean_relative_violation_conditional"]),
                        rel_tol=1e-12, abs_tol=1e-12,
                    ):
                        raise ValueError("Relative all/conditional mean identity failed")
    protein_sets = list(proteins_by_family.values())
    if any(len(values) != expected_proteins or values != protein_sets[0] for values in protein_sets):
        raise ValueError("The four families do not contain identical protein IDs")


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
        "protein_equal_mean": float(np.mean(finite)), "protein_std": float(np.std(finite)),
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
        "mean_violation_conditional": (sum(float(row["violation_sum_angstrom"]) for row in rows), nv),
        "mean_relative_violation_all": (sum(float(row["relative_violation_sum"]) for row in rows), n),
        "mean_relative_violation_conditional": (sum(float(row["relative_violation_sum"]) for row in rows), nv),
    }
    bands = {
        "abs_0_5_fraction_all": "n_abs_0_5", "abs_5_10_fraction_all": "n_abs_5_10",
        "abs_10_plus_fraction_all": "n_abs_10_plus",
        "rel_0_0p05_fraction_all": "n_rel_0_0p05",
        "rel_0p05_0p10_fraction_all": "n_rel_0p05_0p10",
        "rel_0p10_plus_fraction_all": "n_rel_0p10_plus",
    }
    if metric in formulas:
        numerator, denominator = formulas[metric]
        return (float(numerator / denominator), "complete") if denominator else (float("nan"), "zero_denominator")
    if metric in bands:
        return (
            (sum(int(row[bands[metric]]) for row in rows) / n, "complete")
            if n else (float("nan"), "zero_denominator")
        )
    if metric == "relative_violation_max":
        values = [float(row[metric]) for row in rows if int(row["n_violations"]) > 0]
        return (max(values), "complete") if values else (float("nan"), "zero_violations")
    if metric == "relative_violation_p99_conditional":
        return float("nan"), "not_reconstructible_without_triangle_rows"
    raise KeyError(metric)


SUMMARY_KEYS = (
    "analysis_name", "model_family", "feature_dim", "layer", "cv_protocol_version",
    "evaluation_design", "evaluation_scope", "oof_prediction_scope",
    "category_system", "category_label",
)


def summarize_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        grouped[tuple(row[key] for key in SUMMARY_KEYS)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        for metric in DERIVED_METRICS:
            pooled, status = pooled_value(group, metric)
            output.append({
                **dict(zip(SUMMARY_KEYS, key)), "metric": metric,
                **descriptive(metric_value(row, metric) for row in group),
                "triangle_pooled_value": pooled, "triangle_pooled_status": status,
            })
    return output


def summarize_length_bins(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    output = []
    by_bin: dict[str, list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row["category_system"] != "edge_model_count":
            by_bin[str(row["protein_length_bin"])].append(row)
    for length_bin, group in sorted(by_bin.items()):
        output.extend({**row, "protein_length_bin": length_bin} for row in summarize_rows(group))
    return output


def length_effects(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = defaultdict(list)
    for row in rows:
        if row["category_system"] != "edge_model_count":
            grouped[tuple(row[key] for key in SUMMARY_KEYS)].append(row)
    output = []
    for key, group in sorted(grouped.items(), key=lambda item: tuple(map(str, item[0]))):
        for metric in DERIVED_METRICS:
            pairs = [
                (math.log(int(row["n_residues"])), metric_value(row, metric)) for row in group
            ]
            pairs = [(x, y) for x, y in pairs if math.isfinite(x) and math.isfinite(y)]
            if len(pairs) >= 3:
                x = np.asarray([value[0] for value in pairs])
                y = np.asarray([value[1] for value in pairs])
                x_mean, x_std = float(np.mean(x)), float(np.std(x))
                if x_std > 0.0:
                    z = (x - x_mean) / x_std
                    slope, intercept = np.polyfit(z, y, 1)
                    if np.ptp(y) <= 1e-15:
                        rho, status = float("nan"), "constant_outcome"
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
                **dict(zip(SUMMARY_KEYS, key)), "metric": metric, "n_proteins": len(pairs),
                "log_length_mean": x_mean, "log_length_std": x_std,
                "standardized_log_length_slope": float(slope),
                "regression_intercept": float(intercept), "spearman_rho": rho,
                "association_status": status,
            })
    return output


def _contrast_statistics(differences: Sequence[float]) -> dict[str, object]:
    stats = descriptive(differences)
    return {
        "n_paired_proteins": stats["n_proteins_finite"],
        "mean_difference_a_minus_b": stats["protein_equal_mean"],
        "std_difference": stats["protein_std"],
        "median_difference": stats["protein_median"],
        "q25_difference": stats["protein_q25"], "q75_difference": stats["protein_q75"],
        "contrast_status": "complete" if differences else "no_finite_pairs",
    }


def paired_contrasts(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    common = [row for row in rows if row["category_system"] != "edge_model_count"]
    row_map = {
        (row["analysis_name"], row["protein_id"], row["category_system"], row["category_label"]): row
        for row in common
    }
    layer = int(common[0]["layer"])
    proteins = sorted({str(row["protein_id"]) for row in common})
    output = []
    for system, labels in CATEGORY_LABELS.items():
        for label in labels:
            for spec_a, spec_b in itertools.combinations(FAMILY_SPECS, 2):
                for metric in DERIVED_METRICS:
                    differences = []
                    for protein_id in proteins:
                        a = row_map[(spec_a.analysis_name, protein_id, system, label)]
                        b = row_map[(spec_b.analysis_name, protein_id, system, label)]
                        av, bv = metric_value(a, metric), metric_value(b, metric)
                        if math.isfinite(av) and math.isfinite(bv):
                            differences.append(av - bv)
                    output.append({
                        "layer": layer, "evaluation_scope": EVALUATION_SCOPE,
                        "category_system": system, "category_label": label, "metric": metric,
                        "analysis_name_a": spec_a.analysis_name,
                        "analysis_name_b": spec_b.analysis_name,
                        **_contrast_statistics(differences),
                    })
    return output


def edge_model_count_contrasts(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    edge = [row for row in rows if row["category_system"] == "edge_model_count"]
    row_map = {(row["protein_id"], row["category_label"]): row for row in edge}
    proteins = sorted({str(row["protein_id"]) for row in edge})
    layer = int(edge[0]["layer"])
    contrasts = (("2_models", "1_model"), ("3_models", "1_model"), ("3_models", "2_models"))
    output = []
    for label_a, label_b in contrasts:
        for metric in DERIVED_METRICS:
            differences = []
            for protein_id in proteins:
                av = metric_value(row_map[(protein_id, label_a)], metric)
                bv = metric_value(row_map[(protein_id, label_b)], metric)
                if math.isfinite(av) and math.isfinite(bv):
                    differences.append(av - bv)
            output.append({
                "layer": layer, "metric": metric, "category_label_a": label_a,
                "category_label_b": label_b, **_contrast_statistics(differences),
            })
    return output


GEOMETRY_METRICS = {
    "n_negative_predictions": ("n_negative_predictions", 0.0, False),
    "n_triangles": ("n_triangles", 0.0, False),
    "triangle_violation_fraction": ("violation_fraction", 1e-12, False),
    "triangle_mean_violation": ("mean_violation_all", 1e-10, False),
    "triangle_max_violation": ("violation_max_angstrom", 1e-10, False),
    "triangle_violation_0_5_fraction": ("n_abs_0_5", 1e-12, True),
    "triangle_violation_5_10_fraction": ("n_abs_5_10", 1e-12, True),
    "triangle_violation_10_plus_fraction": ("n_abs_10_plus", 1e-12, True),
}


def load_producer_geometry() -> dict[str, dict[tuple[int, str], dict[str, str]]]:
    output = {}
    for spec in FAMILY_SPECS:
        path = producer_root() / spec.family_dir / "geometry_metrics.csv"
        with path.open(newline="") as handle:
            rows = list(csv.DictReader(handle))
        selected = {
            (int(row["layer"]), row["protein_id"]): row
            for row in rows
            if row["evaluation_scope"] == EVALUATION_SCOPE and row["geometry_status"] == "complete"
        }
        if len(selected) != len(LAYERS) * EXPECTED_PROTEINS:
            raise ValueError(f"Producer geometry coverage is incomplete for {spec.analysis_name}")
        output[spec.analysis_name] = selected
    return output


def reproduce_geometry(
    rows: Sequence[Mapping[str, object]], producer: Mapping[str, Mapping[tuple[int, str], Mapping[str, str]]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    all_rows = [row for row in rows if row["category_system"] == "all_triangles"]
    layer = int(all_rows[0]["layer"])
    summary = []
    failures = []
    for spec in FAMILY_SPECS:
        family_rows = [row for row in all_rows if row["analysis_name"] == spec.analysis_name]
        for metric, (replay_column, tolerance, is_band) in GEOMETRY_METRICS.items():
            differences = []
            for replay in family_rows:
                source = producer[spec.analysis_name][(layer, str(replay["protein_id"]))]
                producer_value = float(source[metric])
                replay_value = float(replay[replay_column])
                if is_band:
                    replay_value /= int(replay["n_triangles"])
                difference = abs(producer_value - replay_value)
                differences.append(difference)
                if not math.isfinite(difference) or difference > tolerance:
                    failures.append({
                        "analysis_name": spec.analysis_name, "model_family": spec.model_family,
                        "layer": layer, "protein_id": replay["protein_id"], "metric": metric,
                        "producer_value": producer_value, "replay_value": replay_value,
                        "absolute_difference": difference, "tolerance": tolerance,
                        "reason": "difference_exceeds_tolerance",
                    })
            maximum = max(differences) if differences else float("nan")
            summary.append({
                "analysis_name": spec.analysis_name, "model_family": spec.model_family,
                "layer": layer, "metric": metric, "n_compared": len(differences),
                "maximum_absolute_difference": maximum, "tolerance": tolerance,
                "status": "pass" if differences and maximum <= tolerance else "fail",
            })
    return summary, failures


class AtomicCSVStream:
    def __init__(self, path: Path, columns: Sequence[str]) -> None:
        self.path, self.columns = path, tuple(columns)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = tempfile.NamedTemporaryFile(
            mode="w", newline="", encoding="utf-8", dir=path.parent,
            prefix=f".{path.name}.", suffix=".tmp", delete=False,
        )
        self.handle = handle
        self.temp_path = Path(handle.name)
        self.writer = csv.DictWriter(handle, fieldnames=self.columns, extrasaction="raise")
        self.writer.writeheader()
        self.n_rows = 0

    def write(self, rows: Iterable[Mapping[str, object]]) -> None:
        for row in rows:
            self.writer.writerow({column: row[column] for column in self.columns})
            self.n_rows += 1

    def close(self) -> None:
        self.handle.flush()
        os.fsync(self.handle.fileno())
        self.handle.close()
        os.replace(self.temp_path, self.path)

    def abort(self) -> None:
        self.handle.close()
        self.temp_path.unlink(missing_ok=True)


def main() -> None:
    started = time.perf_counter()
    started_wall = utc_now()
    root = output_root()
    preparation = load_json(root / "run_manifest.json")
    if preparation.get("preparation_status") != "complete":
        raise ValueError("Reference preparation is incomplete")
    producer_geometry = load_producer_geometry()
    streams = {
        "metrics": AtomicCSVStream(root / "per_protein_triangle_metrics.csv", PER_PROTEIN_COLUMNS),
        "summary": AtomicCSVStream(root / "triangle_metric_summary.csv", SUMMARY_COLUMNS),
        "length_bins": AtomicCSVStream(root / "length_bin_summary.csv", LENGTH_BIN_SUMMARY_COLUMNS),
        "length_effects": AtomicCSVStream(root / "protein_length_effects.csv", LENGTH_EFFECT_COLUMNS),
        "paired": AtomicCSVStream(root / "paired_model_contrasts.csv", PAIRED_CONTRAST_COLUMNS),
        "edge": AtomicCSVStream(root / "edge_model_count_contrasts.csv", EDGE_CONTRAST_COLUMNS),
        "reproduction_summary": AtomicCSVStream(
            root / "geometry_reproduction_summary.csv", GEOMETRY_REPRODUCTION_SUMMARY_COLUMNS,
        ),
        "reproduction_failures": AtomicCSVStream(
            root / "geometry_reproduction_failures.csv", GEOMETRY_REPRODUCTION_FAILURE_COLUMNS,
        ),
        "runtime": AtomicCSVStream(root / "runtime.csv", RUNTIME_COLUMNS),
    }
    provenance_layers = []
    try:
        streams["runtime"].write(({
            "stage": "reference_preparation", "layer": -1, "protein_id": "",
            "started_at": preparation.get("started_at", ""),
            "finished_at": preparation.get("completed_at", preparation.get("created_at", "")),
            "elapsed_seconds": preparation.get("elapsed_seconds", float("nan")),
            "peak_rss_mb": preparation.get("peak_rss_mb", float("nan")),
            "status": "complete", "detail": "",
        },))
        for layer in LAYERS:
            shard_path = root / f"layer_shards/per_protein_triangle_metrics_layer_{layer:02d}.csv"
            rows = [convert_metric_row(row) for row in read_csv_strict(shard_path, PER_PROTEIN_COLUMNS)]
            validate_complete_layer(rows, layer)
            streams["metrics"].write(rows)
            streams["summary"].write(summarize_rows(rows))
            streams["length_bins"].write(summarize_length_bins(rows))
            streams["length_effects"].write(length_effects(rows))
            streams["paired"].write(paired_contrasts(rows))
            streams["edge"].write(edge_model_count_contrasts(rows))
            reproduction_summary, reproduction_failures = reproduce_geometry(rows, producer_geometry)
            streams["reproduction_summary"].write(reproduction_summary)
            streams["reproduction_failures"].write(reproduction_failures)
            runtime_path = root / f"layer_shards/runtime_layer_{layer:02d}.csv"
            streams["runtime"].write(read_csv_strict(runtime_path, RUNTIME_COLUMNS))
            provenance_layers.append(load_json(
                root / f"layer_shards/artifact_provenance_layer_{layer:02d}.json"
            ))
            log(f"aggregated layer={layer} rows={len(rows)} failures={len(reproduction_failures)}")
        if streams["metrics"].n_rows != 2_729_088:
            raise RuntimeError(
                f"Consolidated row count is {streams['metrics'].n_rows}, expected 2,729,088"
            )
        if streams["summary"].n_rows != 64_272:
            raise RuntimeError(
                f"Summary row count is {streams['summary'].n_rows}, expected 64,272"
            )
        if streams["reproduction_failures"].n_rows:
            raise RuntimeError(
                f"Geometry reproduction has {streams['reproduction_failures'].n_rows} failures"
            )
        streams["runtime"].write(({
            "stage": "aggregation_complete", "layer": -1, "protein_id": "",
            "started_at": started_wall, "finished_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb(),
            "status": "complete", "detail": "",
        },))
        for stream in streams.values():
            stream.close()
    except Exception:
        for stream in streams.values():
            if not stream.handle.closed:
                stream.abort()
        raise

    if len(provenance_layers) != len(LAYERS) or any(
        int(record.get("layer", -1)) != layer
        for layer, record in enumerate(provenance_layers)
    ):
        raise RuntimeError("Layer artifact provenance is incomplete or misordered")
    output_files = (
        "per_protein_triangle_metrics.csv", "triangle_metric_summary.csv",
        "edge_model_count_contrasts.csv", "edge_model_count_reference_balance.csv",
        "length_bin_summary.csv", "protein_length_effects.csv", "paired_model_contrasts.csv",
        "geometry_reproduction_summary.csv", "geometry_reproduction_failures.csv",
        "runtime.csv",
    )
    manifest = {
        **preparation, "analysis_status": "complete", "aggregation_completed_at": utc_now(),
        "evaluation_scope": EVALUATION_SCOPE, "n_layers": len(LAYERS),
        "n_model_families": len(FAMILY_SPECS), "n_proteins": EXPECTED_PROTEINS,
        "n_per_protein_triangle_metric_rows": streams["metrics"].n_rows,
        "n_triangle_metric_summary_rows": streams["summary"].n_rows,
        "n_geometry_reproduction_failures": streams["reproduction_failures"].n_rows,
        "aggregation_elapsed_seconds": time.perf_counter() - started,
        "aggregation_peak_rss_mb": peak_rss_mb(),
        "files": {
            **dict(preparation.get("files", {})),
            **{name: sha256_file(root / name) for name in output_files},
        },
    }
    atomic_write_json(root / "run_manifest.json", manifest)
    log(f"aggregation complete elapsed={manifest['aggregation_elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()

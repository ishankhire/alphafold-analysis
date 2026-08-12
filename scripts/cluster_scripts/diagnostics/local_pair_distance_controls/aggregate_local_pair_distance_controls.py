#!/usr/bin/env python3
"""Validate, aggregate, summarize, and plot all six distance-control shards."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


OUTPUT_NAME = "local_pair_distance_controls_layers10_16"
TRANSITIONS = tuple((layer, layer + 1) for layer in range(10, 16))
PREDICTIVE_VARIANTS = (
    "fixed_raw_layer_x",
    "fixed_moment_aligned_to_x",
    "saved_own_layer",
    "pooled_two_layer",
)
MERGED_TABLES = (
    "probe_performance.csv",
    "distance_statistics.csv",
    "change_statistics.csv",
    "variant_comparison.csv",
    "per_protein_metrics.csv",
    "distribution_drift_channels.csv",
    "distribution_drift_summary.csv",
    "alignment_parameters.csv",
    "pooled_coefficients.csv",
    "pooled_fit_summary.csv",
    "coefficients_used.csv",
    "split_assignments.csv",
    "protein_manifest.csv",
    "timing.csv",
)
REQUIRED_FILES = MERGED_TABLES + ("summary.txt",)
COLORS = {
    "fixed_raw_layer_x": "#111111",
    "fixed_moment_aligned_to_x": "#0072B2",
    "saved_own_layer": "#D55E00",
    "pooled_two_layer": "#009E73",
}
LABELS = {
    "fixed_raw_layer_x": "Fixed raw layer-x",
    "fixed_moment_aligned_to_x": "Moment-aligned to x",
    "saved_own_layer": "Own-layer saved probes",
    "pooled_two_layer": "Pooled two-layer probe",
}
SCOPES = ("all", "local_4_12")
AGGREGATIONS = ("pair_weighted", "protein_equal")


def format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.9g}"
    return value


def write_csv_atomic(path, fieldnames, rows):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})
    os.replace(temporary, path)


def write_json_atomic(path, payload):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_text_atomic(path, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def shard_path(output_root, layer_from, layer_to):
    return Path(output_root) / "shards" / f"transition_{layer_from:02d}_{layer_to:02d}"


def validate_shards(output_root, required_files=REQUIRED_FILES):
    validated = []
    problems = []
    for layer_from, layer_to in TRANSITIONS:
        directory = shard_path(output_root, layer_from, layer_to)
        metadata_path = directory / "run_metadata.json"
        if not metadata_path.exists():
            problems.append(f"missing metadata: {metadata_path}")
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"invalid metadata {metadata_path}: {exc}")
            continue
        if metadata.get("status") != "complete":
            problems.append(f"incomplete shard {directory}: status={metadata.get('status')!r}")
        if int(metadata.get("layer_from", -1)) != layer_from or int(metadata.get("layer_to", -1)) != layer_to:
            problems.append(f"metadata identity mismatch: {metadata_path}")
        if tuple(metadata.get("predictive_variants", ())) != PREDICTIVE_VARIANTS:
            problems.append(f"predictive variant mismatch: {metadata_path}")
        for filename in required_files:
            if not (directory / filename).exists():
                problems.append(f"missing required file: {directory / filename}")
        validated.append((layer_from, layer_to, directory, metadata))
    if problems:
        raise RuntimeError("Cannot aggregate distance controls:\n  - " + "\n  - ".join(problems))
    if len(validated) != len(TRANSITIONS):
        raise RuntimeError(f"Expected {len(TRANSITIONS)} complete shards, found {len(validated)}")
    return validated


def merge_csv(validated, filename):
    fieldnames = None
    rows = []
    for _layer_from, _layer_to, directory, _metadata in validated:
        with (directory / filename).open(newline="") as handle:
            reader = csv.DictReader(handle)
            current = list(reader.fieldnames or [])
            if fieldnames is None:
                fieldnames = current
            elif current != fieldnames:
                raise ValueError(f"Schema mismatch in {directory / filename}")
            rows.extend(reader)
    return fieldnames or [], rows


def deduplicate(rows, key_fields):
    unique = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in unique and unique[key] != row:
            raise ValueError(f"Conflicting duplicate for key={key}")
        unique[key] = row
    return list(unique.values())


def finite(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def matching(rows, **conditions):
    return [row for row in rows if all(row.get(key) == str(value) for key, value in conditions.items())]


def transition_x(row):
    return int(row["layer_to"])


def style_axis(axis, title, ylabel):
    axis.set_title(title)
    axis.set_xlabel("Transition endpoint layer")
    axis.set_ylabel(ylabel)
    axis.set_xticks(range(11, 17))
    axis.grid(alpha=0.25)


def plot_predicted_distance(rows, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for column, scope in enumerate(SCOPES):
        for row_index, aggregation in enumerate(AGGREGATIONS):
            axis = axes[row_index, column]
            for variant in PREDICTIVE_VARIANTS:
                selected = matching(
                    rows, variant=variant, endpoint="to", split="test",
                    pair_scope=scope, aggregation=aggregation,
                )
                selected.sort(key=transition_x)
                axis.plot(
                    [transition_x(row) for row in selected],
                    [finite(row, "mean_predicted") for row in selected],
                    marker="o", linewidth=2, color=COLORS[variant], label=LABELS[variant],
                )
            style_axis(axis, f"{scope}; {aggregation}", "Mean predicted distance (Å)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Held-out predicted distance at layer x+1")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_change(rows, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for column, scope in enumerate(SCOPES):
        for row_index, aggregation in enumerate(AGGREGATIONS):
            axis = axes[row_index, column]
            for variant in PREDICTIVE_VARIANTS:
                selected = matching(
                    rows, variant=variant, split="test", pair_scope=scope,
                    aggregation=aggregation,
                )
                selected.sort(key=transition_x)
                axis.plot(
                    [transition_x(row) for row in selected],
                    [finite(row, "mean") for row in selected], marker="o",
                    linewidth=2, color=COLORS[variant], label=LABELS[variant],
                )
            axis.axhline(0.0, color="#777777", linewidth=1)
            style_axis(axis, f"{scope}; {aggregation}", "Mean predicted change (Å)")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Held-out consecutive-layer predicted-distance change")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_r2(rows, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    for column, scope in enumerate(SCOPES):
        for row_index, metric in enumerate(("pair_weighted_r2", "mean_per_protein_r2")):
            axis = axes[row_index, column]
            for variant in PREDICTIVE_VARIANTS:
                selected = matching(
                    rows, variant=variant, endpoint="to", split="test", pair_scope=scope,
                )
                selected.sort(key=transition_x)
                axis.plot(
                    [transition_x(row) for row in selected],
                    [finite(row, metric) for row in selected], marker="o", linewidth=2,
                    color=COLORS[variant], label=LABELS[variant],
                )
            label = "Pair-weighted R²" if metric == "pair_weighted_r2" else "Mean per-protein R²"
            style_axis(axis, f"{scope}; {label}", label)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Held-out layer-x+1 distance accuracy")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_drift(rows, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True)
    metrics = (
        ("mean_absolute_standardized_mean_shift", "Mean |standardized mean shift|"),
        ("mean_absolute_log_scale_ratio", "Mean |log scale ratio|"),
    )
    for column, scope in enumerate(SCOPES):
        for row_index, (metric, label) in enumerate(metrics):
            axis = axes[row_index, column]
            for aggregation, linestyle in (("pair_weighted", "-"), ("protein_equal", "--")):
                selected = matching(
                    rows, split="test", pair_scope=scope, aggregation=aggregation,
                )
                selected.sort(key=transition_x)
                axis.plot(
                    [transition_x(row) for row in selected],
                    [finite(row, metric) for row in selected], marker="o",
                    linestyle=linestyle, linewidth=2, label=aggregation,
                )
            style_axis(axis, f"{scope}; {label}", label)
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Held-out pair-representation distribution drift")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def plot_agreement(rows, output_path):
    figure, axes = plt.subplots(2, 2, figsize=(13, 8), sharex=True, sharey=True)
    controls = PREDICTIVE_VARIANTS[1:]
    for column, scope in enumerate(SCOPES):
        for row_index, aggregation in enumerate(AGGREGATIONS):
            axis = axes[row_index, column]
            for variant in controls:
                selected = matching(
                    rows, variant=variant, split="test", pair_scope=scope,
                    aggregation=aggregation,
                )
                selected.sort(key=transition_x)
                axis.plot(
                    [transition_x(row) for row in selected],
                    [finite(row, "threshold_class_agreement") for row in selected],
                    marker="o", linewidth=2, color=COLORS[variant], label=LABELS[variant],
                )
            axis.set_ylim(0.0, 1.02)
            style_axis(axis, f"{scope}; {aggregation}", "0.5 Å class agreement")
    axes[0, 0].legend(fontsize=8)
    figure.suptitle("Control agreement with raw fixed-readout changes")
    figure.tight_layout()
    figure.savefig(output_path, dpi=180)
    plt.close(figure)


def summary_text(change_rows, performance_rows, drift_rows, metadata_rows):
    lines = [
        "Consecutive-layer distance-control probe aggregate",
        "===================================================",
        "",
        "Status: complete",
        f"Completed transitions: {len(metadata_rows)}/6 (10->11 through 15->16)",
        "Distance type: raw pre-MDS, unclipped",
        "Scopes: all pairs and local sequence separation 4..12",
        "",
        "Held-out local pair-weighted mean changes:",
    ]
    for layer_from, layer_to in TRANSITIONS:
        parts = []
        for variant in PREDICTIVE_VARIANTS:
            selected = matching(
                change_rows, layer_from=layer_from, layer_to=layer_to,
                variant=variant, split="test", pair_scope="local_4_12",
                aggregation="pair_weighted",
            )
            parts.append(f"{variant}={finite(selected[0], 'mean'):+.4f} Å")
        lines.append(f"  {layer_from}->{layer_to}: " + "; ".join(parts))
    lines.extend(["", "Held-out all-pair layer-x+1 pair-weighted R2:"])
    for layer_from, layer_to in TRANSITIONS:
        parts = []
        for variant in PREDICTIVE_VARIANTS:
            selected = matching(
                performance_rows, layer_from=layer_from, layer_to=layer_to,
                variant=variant, endpoint="to", split="test", pair_scope="all",
            )
            parts.append(f"{variant}={finite(selected[0], 'pair_weighted_r2'):.4f}")
        lines.append(f"  {layer_from}->{layer_to}: " + "; ".join(parts))
    maximum_error = max(abs(finite(row, "reconstruction_error")) for row in drift_rows)
    lines.extend([
        "", f"Maximum drift reconstruction error: {maximum_error:.9g} Å",
        "Pair-level quantiles use bounded reservoir samples; means and error metrics are exact.",
    ])
    return "\n".join(lines) + "\n"


def main():
    csv_root = Path(os.environ.get("CSV_ROOT") or (Path.home() / "csv_files")).expanduser()
    output_root = Path(
        os.environ.get("LOCAL_PAIR_DISTANCE_CONTROLS_OUTPUT_ROOT")
        or (csv_root / OUTPUT_NAME)
    ).expanduser()
    validated = validate_shards(output_root)
    merged = {}
    for filename in MERGED_TABLES:
        fieldnames, rows = merge_csv(validated, filename)
        if filename == "split_assignments.csv":
            rows = deduplicate(rows, ("split_id", "protein_id"))
        elif filename == "coefficients_used.csv":
            rows = deduplicate(rows, ("variant", "split_id", "layer", "feature_index"))
        write_csv_atomic(output_root / filename, fieldnames, rows)
        merged[filename] = rows

    visualizations = output_root / "visualizations"
    visualizations.mkdir(parents=True, exist_ok=True)
    plot_predicted_distance(merged["distance_statistics.csv"], visualizations / "test_predicted_distance_by_transition.png")
    plot_change(merged["change_statistics.csv"], visualizations / "test_change_by_transition.png")
    plot_r2(merged["probe_performance.csv"], visualizations / "test_r2_by_transition.png")
    plot_drift(merged["distribution_drift_summary.csv"], visualizations / "distribution_drift_by_transition.png")
    plot_agreement(merged["variant_comparison.csv"], visualizations / "raw_vs_control_change_agreement.png")

    metadata_rows = [metadata for _a, _b, _directory, metadata in validated]
    aggregate_metadata = {
        "status": "complete", "output_name": OUTPUT_NAME,
        "transitions": [f"{a}->{b}" for a, b in TRANSITIONS],
        "predictive_variants": list(PREDICTIVE_VARIANTS),
        "diagnostic_variant": "distribution_drift",
        "n_complete_shards": len(validated),
        "shard_metadata": metadata_rows,
    }
    write_json_atomic(output_root / "run_metadata.json", aggregate_metadata)
    write_text_atomic(
        output_root / "summary.txt",
        summary_text(
            merged["change_statistics.csv"], merged["probe_performance.csv"],
            merged["distribution_drift_summary.csv"], metadata_rows,
        ),
    )


if __name__ == "__main__":
    main()

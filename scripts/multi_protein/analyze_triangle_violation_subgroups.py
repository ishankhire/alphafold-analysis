#!/usr/bin/env python3
"""Create local publication-ready plots from copied triangle subgroup outputs."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import matplotlib.pyplot as plt
import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = REPO_ROOT / "delta_files" / "triangle_violation_subgroups"
MODEL_LABELS = {
    "general_base256_combined": "Shared OLS",
    "per_protein_base256_combined": "Per-protein OLS",
    "shared_xgboost_base256_combined": "Shared XGBoost",
    "analysis_1_sequence_gated": "Sequence-gated OLS",
}
MODEL_COLORS = {
    name: color for name, color in zip(MODEL_LABELS, ("#0072B2", "#D55E00", "#009E73", "#CC79A7"))
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_json(path: Path, payload: object) -> None:
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent,
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            json.dump(payload, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def plot_category_trajectories(
    rows: Sequence[Mapping[str, str]],
    category_system: str,
    destination: Path,
    title: str,
) -> None:
    metrics = (
        ("violation_fraction", "Violation fraction"),
        ("mean_relative_violation_all", "Normalized burden"),
        ("mean_relative_violation_conditional", "Conditional relative severity"),
        ("relative_violation_p99_conditional", "Conditional relative p99"),
    )
    selected = [row for row in rows if row["category_system"] == category_system]
    labels = sorted({row["category_label"] for row in selected})
    linestyles = ["-", "--", ":", "-."]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharex=True, constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes.flat, metrics):
        for analysis_name in MODEL_LABELS:
            for label_index, category_label in enumerate(labels):
                values = sorted(
                    (
                        (int(row["layer"]), float(row["protein_equal_mean"]))
                        for row in selected
                        if row["analysis_name"] == analysis_name
                        and row["category_label"] == category_label
                        and row["metric"] == metric
                    ),
                    key=lambda pair: pair[0],
                )
                if not values:
                    continue
                x, y = zip(*values)
                axis.plot(
                    x, y, color=MODEL_COLORS[analysis_name],
                    linestyle=linestyles[label_index % len(linestyles)], linewidth=1.5,
                    label=f"{MODEL_LABELS[analysis_name]} · {category_label}",
                )
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    for axis in axes[-1]:
        axis.set_xlabel("Evoformer layer")
    handles, legend_labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc="outside lower center", ncol=min(4, max(1, len(labels))))
    fig.suptitle(title)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, bbox_inches="tight")
    plt.close(fig)


def plot_length_sensitivity(rows: Sequence[Mapping[str, str]], destination: Path) -> None:
    metrics = (
        ("violation_fraction", "Violation-fraction slope"),
        ("mean_relative_violation_all", "Normalized-burden slope"),
        ("relative_violation_p99_conditional", "Conditional-p99 slope"),
    )
    fig, axes = plt.subplots(3, 1, figsize=(10, 9), sharex=True, constrained_layout=True)
    for axis, (metric, ylabel) in zip(axes, metrics):
        for analysis_name in MODEL_LABELS:
            values = sorted(
                (
                    (int(row["layer"]), float(row["standardized_log_length_slope"]))
                    for row in rows
                    if row["analysis_name"] == analysis_name
                    and row["category_system"] == "all_triangles"
                    and row["category_label"] == "all"
                    and row["metric"] == metric
                    and row["association_status"] == "complete"
                ),
                key=lambda pair: pair[0],
            )
            if values:
                x, y = zip(*values)
                axis.plot(x, y, color=MODEL_COLORS[analysis_name], label=MODEL_LABELS[analysis_name])
        axis.axhline(0.0, color="black", linewidth=0.7)
        axis.set_ylabel(ylabel)
        axis.grid(alpha=0.2)
    axes[-1].set_xlabel("Evoformer layer")
    axes[0].legend(ncol=2)
    fig.suptitle("Protein-length sensitivity (slope per 1 SD log length)")
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination)
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = args.input_root.resolve()
    summaries = read_csv(root / "triangle_metric_summary.csv")
    effects = read_csv(root / "protein_length_effects.csv")
    visualizations = root / "visualizations"
    plots = (
        ("rms_size", "rms_size_trajectories.pdf", "RMS-size triangle subgroups"),
        ("angle_shape", "angle_shape_trajectories.pdf", "Largest-angle triangle subgroups"),
        ("rms_by_angle", "rms_by_angle_trajectories.pdf", "RMS-by-angle triangle subgroups"),
        ("relative_slack", "relative_slack_trajectories.pdf", "Relative-slack triangle subgroups"),
        ("area", "area_sensitivity_trajectories.pdf", "Area-tertile sensitivity"),
    )
    for system, filename, title in plots:
        plot_category_trajectories(summaries, system, visualizations / filename, title)
    plot_length_sensitivity(effects, visualizations / "protein_length_sensitivity.pdf")
    manifest_path = root / "run_manifest.json"
    with manifest_path.open() as handle:
        manifest = json.load(handle)
    if manifest.get("run_status") != "complete":
        raise ValueError("Refusing to finalize visualizations for an incomplete aggregation")
    artifact_sha256 = dict(manifest.get("artifact_sha256", {}))
    for path in sorted(visualizations.glob("*.pdf")):
        artifact_sha256[str(path.relative_to(root))] = sha256_file(path)
    manifest["artifact_sha256"] = artifact_sha256
    manifest["visualizations_finalized_at"] = datetime.now(timezone.utc).isoformat()
    atomic_write_json(manifest_path, manifest)
    print(f"Wrote six PDFs beneath {visualizations}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create the versioned local figure suite for fivefold OOF triangle replay."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


ANALYSIS_NAME = "triangle_violation_subgroups_5fold_v1"
DEFAULT_ROOT = REPO_ROOT / "delta_files" / "analysis" / ANALYSIS_NAME
LAYERS = tuple(range(48))
KEY_METRICS = (
    ("violation_fraction", "Violation frequency", "Fraction of triangles"),
    ("mean_relative_violation_all", "Scale-normalized burden", "Mean relative violation"),
)
MODEL_SPECS = {
    "shared_ols_base256": ("Shared OLS", "#4C78A8", "-"),
    "per_protein_ols_base256": ("Per-protein OLS", "#7F7F7F", (0, (5, 2))),
    "shared_xgboost_base256_fixed_s4": ("Shared XGBoost", "#E45756", "-"),
    "sequence_gated_ols_base256": ("Sequence-gated OLS", "#F58518", "-"),
}
CATEGORY_LABELS = {
    "all_triangles": ("all",),
    "rms_size": ("small", "medium", "large"),
    "angle_shape": ("clearly_acute", "near_right", "clearly_obtuse"),
    "rms_by_angle": tuple(
        f"{size}__{angle}"
        for size in ("small", "medium", "large")
        for angle in ("clearly_acute", "near_right", "clearly_obtuse")
    ),
    "relative_slack": ("low", "medium", "high"),
    "area": ("low", "medium", "high"),
}
FIGURES = (
    ("all_triangles", "all_triangles_key_metrics", "All reference triangles"),
    ("rms_size", "rms_size_key_metrics", "Reference RMS-size tertiles"),
    ("angle_shape", "angle_shape_key_metrics", "Reference broad-angle classes"),
    ("rms_by_angle", "rms_by_angle_key_metrics", "RMS size by broad angle"),
    ("relative_slack", "relative_slack_key_metrics", "Reference relative-slack tertiles"),
    ("area", "area_sensitivity_key_metrics", "Reference area tertiles"),
)


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 9, "axes.titlesize": 10,
        "axes.labelsize": 9, "legend.fontsize": 8, "axes.spines.top": False,
        "axes.spines.right": False, "axes.grid": True, "grid.alpha": 0.22,
        "savefig.dpi": 220, "savefig.bbox": "tight", "pdf.fonttype": 42,
    })


def display_label(value: str) -> str:
    return value.replace("__", " · ").replace("_", " ").title()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def require_columns(frame: pd.DataFrame, columns: Sequence[str], path: Path) -> None:
    missing = set(columns) - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}")


def validate_inputs(root: Path, summary: pd.DataFrame, length: pd.DataFrame) -> None:
    with (root / "run_manifest.json").open() as handle:
        manifest = json.load(handle)
    if manifest.get("analysis_name") != ANALYSIS_NAME or manifest.get("analysis_status") != "complete":
        raise ValueError("Refusing to plot an incomplete or wrong-version analysis")
    required = (
        "analysis_name", "model_family", "layer", "evaluation_scope", "category_system",
        "category_label", "metric", "protein_equal_mean", "n_proteins_finite",
    )
    require_columns(summary, required, root / "triangle_metric_summary.csv")
    require_columns(length, (*required, "protein_length_bin"), root / "length_bin_summary.csv")
    if set(summary["analysis_name"]) != set(MODEL_SPECS):
        raise ValueError("Summary model coverage differs from the four frozen families")
    if set(summary["layer"].astype(int)) != set(LAYERS):
        raise ValueError("Summary does not cover every layer")
    if set(summary["evaluation_scope"]) != {"oof_all_pairs"}:
        raise ValueError("Summary is not the OOF all-pairs analysis")
    failures = root / "geometry_reproduction_failures.csv"
    with failures.open(newline="") as handle:
        if any(True for _ in csv.DictReader(handle)):
            raise ValueError("Geometry reproduction failures must be header-only before plotting")


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> list[dict[str, object]]:
    records = []
    for extension in ("pdf", "png"):
        path = directory / f"{stem}.{extension}"
        figure.savefig(
            path,
            metadata={"Creator": Path(__file__).name, "CreationDate": None, "ModDate": None}
            if extension == "pdf" else None,
        )
        records.append({
            "figure": stem, "format": extension, "path": path.relative_to(directory).as_posix(),
            "sha256": sha256_file(path), "size_bytes": path.stat().st_size,
        })
    plt.close(figure)
    return records


def plot_category(
    summary: pd.DataFrame, system: str, title: str, output: Path, stem: str,
) -> list[dict[str, object]]:
    labels = CATEGORY_LABELS[system]
    figure, axes = plt.subplots(
        len(labels), len(KEY_METRICS), figsize=(12.0, max(3.3, 2.25 * len(labels))),
        sharex=True, squeeze=False,
    )
    subset = summary.loc[summary["category_system"].eq(system)]
    for row_index, label in enumerate(labels):
        for column_index, (metric, metric_title, ylabel) in enumerate(KEY_METRICS):
            axis = axes[row_index, column_index]
            selected = subset.loc[
                subset["category_label"].eq(label) & subset["metric"].eq(metric)
            ]
            for analysis, (model_label, color, linestyle) in MODEL_SPECS.items():
                points = selected.loc[selected["analysis_name"].eq(analysis)].sort_values("layer")
                axis.plot(
                    points["layer"], points["protein_equal_mean"], label=model_label,
                    color=color, linestyle=linestyle, linewidth=1.8,
                )
            axis.set_title(f"{display_label(label)} — {metric_title}")
            axis.set_ylabel(ylabel)
            axis.set_xticks((0, 10, 20, 30, 40, 47))
            if row_index == len(labels) - 1:
                axis.set_xlabel("Evoformer layer")
    handles = [
        Line2D([0], [0], color=color, linestyle=linestyle, lw=2, label=label)
        for label, color, linestyle in MODEL_SPECS.values()
    ]
    figure.legend(handles=handles, loc="upper center", ncol=4, frameon=False)
    figure.suptitle(title, y=1.01)
    figure.tight_layout(rect=(0, 0, 1, 0.97))
    return save_figure(figure, output, stem)


def plot_edge_trajectories(summary: pd.DataFrame, output: Path) -> list[dict[str, object]]:
    subset = summary.loc[summary["category_system"].eq("edge_model_count")]
    colors = {"1_model": "#4C78A8", "2_models": "#F58518", "3_models": "#E45756"}
    figure, axes = plt.subplots(1, 2, figsize=(11.5, 3.8), sharex=True)
    for axis, (metric, title, ylabel) in zip(axes, KEY_METRICS):
        for label in ("1_model", "2_models", "3_models"):
            points = subset.loc[
                subset["category_label"].eq(label) & subset["metric"].eq(metric)
            ].sort_values("layer")
            axis.plot(points["layer"], points["protein_equal_mean"], color=colors[label], lw=2,
                      label=display_label(label))
        axis.set_title(title)
        axis.set_ylabel(ylabel)
        axis.set_xlabel("Evoformer layer")
        axis.set_xticks((0, 10, 20, 30, 40, 47))
    axes[0].legend(frameon=False)
    figure.suptitle("Per-protein OLS cross-fit stitching diagnostic")
    figure.tight_layout()
    return save_figure(figure, output, "edge_model_count_key_metrics")


def plot_edge_balance(balance: pd.DataFrame, output: Path) -> list[dict[str, object]]:
    labels = ("1_model", "2_models", "3_models")
    grouped = balance.groupby("category_label", sort=False)
    figure, axes = plt.subplots(1, 3, figsize=(12.5, 3.8))
    axes[0].bar(labels, [grouped.get_group(label)["triangle_fraction"].mean() for label in labels])
    axes[0].set_title("Triangle support")
    axes[0].set_ylabel("Mean within-protein fraction")
    axes[1].bar(labels, [grouped.get_group(label)["reference_rms_mean"].mean() for label in labels])
    axes[1].set_title("Reference RMS size")
    axes[1].set_ylabel("Protein-equal mean (Å)")
    angle_columns = (
        ("clearly_acute_fraction", "Acute"), ("near_right_fraction", "Near-right"),
        ("clearly_obtuse_fraction", "Obtuse"),
    )
    bottom = [0.0] * len(labels)
    for column, name in angle_columns:
        values = [grouped.get_group(label)[column].mean() for label in labels]
        axes[2].bar(labels, values, bottom=bottom, label=name)
        bottom = [a + b for a, b in zip(bottom, values)]
    axes[2].set_title("Broad-angle composition")
    axes[2].set_ylabel("Mean within-protein fraction")
    axes[2].legend(frameon=False)
    for axis in axes:
        axis.tick_params(axis="x", rotation=20)
    figure.suptitle("Layer-invariant edge-model-count reference balance")
    figure.tight_layout()
    return save_figure(figure, output, "edge_model_count_reference_balance")


def plot_length_sensitivity(length: pd.DataFrame, output: Path) -> list[dict[str, object]]:
    selected = length.loc[
        length["category_system"].eq("all_triangles")
        & length["category_label"].eq("all")
        & length["metric"].isin([metric[0] for metric in KEY_METRICS])
    ]
    bins = [f"length_bin_{index}" for index in range(1, 6)]
    palette = plt.get_cmap("viridis")
    figure, axes = plt.subplots(2, 2, figsize=(12, 7), sharex=True)
    for axis, (analysis, (model_label, _, _)) in zip(axes.flat, MODEL_SPECS.items()):
        model = selected.loc[selected["analysis_name"].eq(analysis)]
        for index, length_bin in enumerate(bins):
            points = model.loc[
                model["protein_length_bin"].eq(length_bin)
                & model["metric"].eq("violation_fraction")
            ].sort_values("layer")
            axis.plot(points["layer"], points["protein_equal_mean"], color=palette(index / 4),
                      lw=1.6, label=f"Quintile {index + 1}")
        axis.set_title(model_label)
        axis.set_ylabel("Violation fraction")
        axis.set_xlabel("Evoformer layer")
        axis.set_xticks((0, 10, 20, 30, 40, 47))
    axes[0, 0].legend(ncol=2, frameon=False)
    figure.suptitle("Protein-length sensitivity: all triangles")
    figure.tight_layout()
    return save_figure(figure, output, "protein_length_sensitivity")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--output-root", type=Path, default=None)
    arguments = parser.parse_args()
    root = arguments.input_root.resolve()
    output = (arguments.output_root or root / "visualizations").resolve()
    output.mkdir(parents=True, exist_ok=True)
    summary = pd.read_csv(root / "triangle_metric_summary.csv")
    length = pd.read_csv(root / "length_bin_summary.csv")
    balance = pd.read_csv(root / "edge_model_count_reference_balance.csv")
    validate_inputs(root, summary, length)
    configure_style()
    records = []
    for system, stem, title in FIGURES:
        records.extend(plot_category(summary, system, title, output, stem))
    records.extend(plot_edge_trajectories(summary, output))
    records.extend(plot_edge_balance(balance, output))
    records.extend(plot_length_sensitivity(length, output))
    manifest = {
        "analysis_name": ANALYSIS_NAME, "created_at": datetime.now(timezone.utc).isoformat(),
        "input_root": str(root), "output_root": str(output), "n_files": len(records),
        "files": records,
    }
    with (output / "figure_manifest.json").open("w") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    print(f"Wrote {len(records)} figure files to {output}")


if __name__ == "__main__":
    main()

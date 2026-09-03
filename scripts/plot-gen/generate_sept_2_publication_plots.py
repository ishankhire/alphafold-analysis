#!/usr/bin/env python3
"""Render the five probe figures used by the NeurIPS workshop submission."""

from __future__ import annotations

import hashlib
import os
from pathlib import Path
from typing import Collection, Sequence

PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

SOURCE_DIR = PROJECT_ROOT / "results/publication_plots/source_data"
OUTPUT_DIR = PROJECT_ROOT / "results/publication_plots/main_figures"
MANIFEST_PATH = PROJECT_ROOT / "results/publication_plots/figure_manifest.csv"

LAYERS = tuple(range(48))
MILESTONE_LAYERS = (0, 10, 20, 30, 40, 47)
SPATIAL_BINS = ("0-5", "5-10", "10-40", "40+")
MODEL_SPECS = {
    "shared_ols_base256": {"label": "Shared linear", "color": "#4C78A8", "linestyle": "-"},
    "per_protein_ols_base256": {"label": "Per-protein linear", "color": "#7F7F7F", "linestyle": (0, (5, 2))},
    "shared_xgboost_base256_fixed_s4": {"label": "XGBoost", "color": "#E45756", "linestyle": "-"},
    "sequence_gated_ols_base256": {"label": "Sequence gated", "color": "#F58518", "linestyle": "-"},
}
MODEL_ORDER = tuple(MODEL_SPECS)
FIGURE_SOURCES = {
    "main_figure_01_mean_per_protein_r2": ("r2_plot_summary.csv",),
    "main_figure_03_tm_score": ("tm_score_plot_summary.csv",),
    "main_figure_15_spatial_distance_mae_combined": ("spatial_distance_mae_plot_summary.csv",),
    "main_figure_all_triangles_peak_troughs_overlay": ("triangle_plot_summary.csv", "triangle_candidate_extrema.csv"),
    "main_figure_all_triangles_decodability_consistency": ("decodability_consistency_points.csv",),
}


def configure_style() -> None:
    scale = 1.5
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10 * scale,
            "axes.titlesize": 12 * scale,
            "axes.labelsize": 11 * scale,
            "xtick.labelsize": 9 * scale,
            "ytick.labelsize": 9 * scale,
            "legend.fontsize": 8.5 * scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def require_columns(frame: pd.DataFrame, columns: Collection[str], name: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{name} is missing columns: {missing}")


def load_layer_summary(name: str, *, categories: Sequence[str] | None = None) -> pd.DataFrame:
    frame = pd.read_csv(SOURCE_DIR / name)
    require_columns(
        frame,
        {"analysis_name", "layer", "mean", "bootstrap_ci_low", "bootstrap_ci_high"},
        name,
    )
    if set(frame["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError(f"{name} does not contain exactly the four submitted probes")
    if set(frame["layer"].astype(int)) != set(LAYERS):
        raise ValueError(f"{name} does not contain all 48 Evoformer layers")
    expected_rows = len(MODEL_ORDER) * len(LAYERS)
    if categories is not None:
        require_columns(frame, {"category_label"}, name)
        if set(frame["category_label"]) != set(categories):
            raise ValueError(f"{name} has an unexpected category inventory")
        expected_rows *= len(categories)
    if len(frame) != expected_rows:
        raise ValueError(f"{name} has {len(frame)} rows; expected {expected_rows}")
    return frame


def clip_interval(low: pd.Series | np.ndarray, high: pd.Series | np.ndarray, domain: str) -> tuple[np.ndarray, np.ndarray]:
    low_values = np.asarray(low, dtype=float)
    high_values = np.asarray(high, dtype=float)
    if domain == "unit":
        return np.clip(low_values, 0, 1), np.clip(high_values, 0, 1)
    if domain == "nonnegative":
        return np.maximum(low_values, 0), np.maximum(high_values, 0)
    if domain == "unbounded":
        return low_values, high_values
    raise ValueError(f"Unknown domain: {domain}")


def plot_model_lines(axis: plt.Axes, summary: pd.DataFrame, domain: str) -> None:
    for analysis in MODEL_ORDER:
        data = summary[summary["analysis_name"].eq(analysis)].sort_values("layer")
        spec = MODEL_SPECS[analysis]
        axis.plot(data["layer"], data["mean"], color=spec["color"], linestyle=spec["linestyle"], linewidth=2, label=spec["label"])
        low, high = clip_interval(data["bootstrap_ci_low"], data["bootstrap_ci_high"], domain)
        axis.fill_between(data["layer"], low, high, color=spec["color"], alpha=0.13, linewidth=0)


def style_layer_axis(axis: plt.Axes, ylabel: str, domain: str) -> None:
    axis.set(xlabel="Evoformer layer", ylabel=ylabel, xlim=(0, 47))
    axis.set_xticks(MILESTONE_LAYERS)
    if domain == "unit":
        axis.set_ylim(0, min(1, axis.get_ylim()[1]))
    elif domain == "nonnegative":
        axis.set_ylim(bottom=0)
    else:
        axis.axhline(0, color="#666666", linewidth=0.7, alpha=0.45, zorder=0)


def save_pdf(figure: plt.Figure, stem: str) -> Path:
    path = OUTPUT_DIR / f"{stem}.pdf"
    figure.savefig(path)
    plt.close(figure)
    return path


def line_figure(summary: pd.DataFrame, ylabel: str, stem: str, domain: str) -> Path:
    figure, axis = plt.subplots(figsize=(8.4, 5.3))
    plot_model_lines(axis, summary, domain)
    style_layer_axis(axis, ylabel, domain)
    axis.legend(frameon=True)
    figure.tight_layout()
    return save_pdf(figure, stem)


def spatial_figure(summary: pd.DataFrame) -> Path:
    figure, axes = plt.subplots(1, 4, figsize=(12.5, 3.9), sharex=True)
    limits = {"0-5": (0, 5), "5-10": (0, 5), "10-40": (0, 5), "40+": (0, 15)}
    for axis, category in zip(axes, SPATIAL_BINS):
        plot_model_lines(axis, summary[summary["category_label"].eq(category)], "nonnegative")
        axis.set(xlim=(0, 47), ylim=limits[category], title=category)
        axis.set_xticks(MILESTONE_LAYERS)
        axis.set_yticks(tuple(range(6)) if category != "40+" else (0, 5, 10, 15))
    handles, labels = axes[0].get_legend_handles_labels()
    ylabel = figure.supylabel("Mean per-protein\nMAE (Å)", x=0.045)
    figure.supxlabel("Evoformer layer", y=0.14)
    figure.legend(handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01), ncol=4, frameon=False)
    figure.tight_layout(rect=(0.025, 0.22, 1, 0.98))
    bounds = axes[0].get_position()
    ylabel.set_y((bounds.y0 + bounds.y1) / 2)
    return save_pdf(figure, "main_figure_15_spatial_distance_mae_combined")


def triangle_figure(summary: pd.DataFrame, candidates: pd.DataFrame) -> Path:
    require_columns(candidates, {"analysis_name", "layer", "center_mean", "direction", "supported_extremum"}, "triangle_candidate_extrema.csv")
    figure, axis = plt.subplots(figsize=(8.6, 5.6))
    plot_model_lines(axis, summary, "nonnegative")
    for analysis in MODEL_ORDER:
        color = MODEL_SPECS[analysis]["color"]
        for row in candidates[candidates["analysis_name"].eq(analysis)].itertuples():
            supported = str(row.supported_extremum).lower() == "true"
            axis.scatter(row.layer, row.center_mean, marker="^" if row.direction == "peak" else "v", s=40, facecolor=color if supported else "white", edgecolor=color, linewidth=1.1, zorder=6)
    style_layer_axis(axis, "Mean relative violation per triangle", "nonnegative")
    axis.set_title("Normalized mean triangle violation")
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([], [], marker="^", linestyle="none", markerfacecolor="white", markeredgecolor="black"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="black", markeredgecolor="black"),
        ]
    )
    labels.extend(("Candidate peak/trough", "BH-supported"))
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.15, 1, 1))
    return save_pdf(figure, "main_figure_all_triangles_peak_troughs_overlay")


def phase_figure(points: pd.DataFrame) -> Path:
    require_columns(
        points,
        {"analysis_name", "layer", "normalized_r2", "consistency_value", "normalized_r2_bootstrap_ci_low", "normalized_r2_bootstrap_ci_high", "consistency_bootstrap_ci_low", "consistency_bootstrap_ci_high", "milestone"},
        "decodability_consistency_points.csv",
    )
    if len(points) != len(MODEL_ORDER) * len(LAYERS):
        raise ValueError("Decodability-consistency source data is incomplete")
    figure, axis = plt.subplots(figsize=(8.7, 5.9))
    label_shifts = (-8, 8, -7, 7)
    for model_index, analysis in enumerate(MODEL_ORDER):
        spec = MODEL_SPECS[analysis]
        data = points[points["analysis_name"].eq(analysis)].sort_values("layer")
        axis.plot(data["normalized_r2"], data["consistency_value"], color=spec["color"], linestyle=spec["linestyle"], linewidth=2, label=spec["label"])
        axis.scatter(data["normalized_r2"], data["consistency_value"], color=spec["color"], s=10)
        milestones = data[data["milestone"].astype(str).str.lower().eq("true")]
        x = milestones["normalized_r2"].to_numpy(dtype=float)
        y = milestones["consistency_value"].to_numpy(dtype=float)
        x_low, x_high = clip_interval(milestones["normalized_r2_bootstrap_ci_low"], milestones["normalized_r2_bootstrap_ci_high"], "unit")
        y_low, y_high = clip_interval(milestones["consistency_bootstrap_ci_low"], milestones["consistency_bootstrap_ci_high"], "nonnegative")
        axis.errorbar(x, y, xerr=np.vstack((np.maximum(0, x - x_low), np.maximum(0, x_high - x))), yerr=np.vstack((np.maximum(0, y - y_low), np.maximum(0, y_high - y))), fmt="none", ecolor=spec["color"], elinewidth=0.8, alpha=0.42, capsize=1.5)
        axis.scatter(x, y, color=spec["color"], s=31, edgecolors="white", linewidths=0.65, zorder=4)
        for row in milestones.itertuples():
            axis.annotate(str(row.layer), (row.normalized_r2, row.consistency_value), xytext=(4, label_shifts[model_index]), textcoords="offset points", fontsize=9.3, color=spec["color"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65})
    axis.set(title="Normalized mean triangle violation", xlabel="Normalized mean per-protein all-pairs $R^2$", ylabel="Mean relative violation per triangle", xlim=(0, 1))
    axis.set_xticks((0, 0.25, 0.5, 0.75, 1))
    axis.set_ylim(bottom=0)
    axis.legend(frameon=True)
    figure.tight_layout()
    return save_pdf(figure, "main_figure_all_triangles_decodability_consistency")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    r2 = load_layer_summary("r2_plot_summary.csv")
    tm_score = load_layer_summary("tm_score_plot_summary.csv")
    spatial = load_layer_summary("spatial_distance_mae_plot_summary.csv", categories=SPATIAL_BINS)
    triangle = load_layer_summary("triangle_plot_summary.csv")
    candidates = pd.read_csv(SOURCE_DIR / "triangle_candidate_extrema.csv")
    phase_points = pd.read_csv(SOURCE_DIR / "decodability_consistency_points.csv")
    outputs = {
        "main_figure_01_mean_per_protein_r2": line_figure(r2, "Mean per-protein $R^2$", "main_figure_01_mean_per_protein_r2", "unbounded"),
        "main_figure_03_tm_score": line_figure(tm_score, "Mean reconstruction TM-score", "main_figure_03_tm_score", "unit"),
        "main_figure_15_spatial_distance_mae_combined": spatial_figure(spatial),
        "main_figure_all_triangles_peak_troughs_overlay": triangle_figure(triangle, candidates),
        "main_figure_all_triangles_decodability_consistency": phase_figure(phase_points),
    }
    actual = {path.stem for path in OUTPUT_DIR.glob("*.pdf")}
    if actual != set(outputs):
        raise RuntimeError(f"Unexpected PDF inventory: {sorted(actual)}")
    rows = []
    for figure_id, path in outputs.items():
        sources = [SOURCE_DIR / name for name in FIGURE_SOURCES[figure_id]]
        rows.append(
            {
                "figure_id": figure_id,
                "pdf_path": path.relative_to(PROJECT_ROOT).as_posix(),
                "source_csvs": ";".join(source.relative_to(PROJECT_ROOT).as_posix() for source in sources),
                "pdf_sha256": sha256_file(path),
                "source_sha256": ";".join(sha256_file(source) for source in sources),
            }
        )
    pd.DataFrame(rows).to_csv(MANIFEST_PATH, index=False)
    print(f"Generated {len(outputs)} submission figures in {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

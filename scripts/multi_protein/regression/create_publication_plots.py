#!/usr/bin/env python3
"""Create publication-focused plots from saved Delta regression outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

EARLY_PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.makedirs(EARLY_PROJECT_ROOT / ".mplconfig", exist_ok=True)
os.makedirs(EARLY_PROJECT_ROOT / ".cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(EARLY_PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(EARLY_PROJECT_ROOT / ".cache"))

import matplotlib
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = EARLY_PROJECT_ROOT
DELTA_DIR = PROJECT_ROOT / "delta_files"
OUT_DIR = DELTA_DIR / "publication_plots"
EXPECTED_LAYERS = tuple(range(48))

ORANGE = "#f58518"
RED = "#e45756"
TRAJECTORY_GREY = "#b8b8b8"
GRID_GREY = "#dddddd"

VARIANT_A = {
    "folder": "general_base256_avg_symmetric",
    "label": "Variant A: Averaged (i, j) pair representations",
    "legend_label": "Variant A: Averaged (i, j) pair\nrepresentations",
    "short_label": "Variant A",
    "color": ORANGE,
}
VARIANT_B = {
    "folder": "general_base256_avg_symmetric_plus_axial",
    "label": "Variant B: Averaged (i, j) pair representations + neighboring pair representations",
    "legend_label": "Variant B: Averaged (i, j) pair representations\n+ neighboring pair representations",
    "short_label": "Variant B",
    "color": RED,
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 13,
            "axes.titlesize": 18,
            "axes.labelsize": 16,
            "xtick.labelsize": 13,
            "ytick.labelsize": 13,
            "legend.fontsize": 12,
            "figure.titlesize": 20,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.1,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def geometry_path(folder: str) -> Path:
    return DELTA_DIR / folder / "geometry_metrics.csv"


def load_test_geometry(folder: str) -> pd.DataFrame:
    path = geometry_path(folder)
    df = pd.read_csv(path)
    required = {
        "layer",
        "split",
        "protein_id",
        "tm_score",
        "triangle_violation_fraction",
        "triangle_mean_violation",
    }
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise ValueError(f"{path} has no test rows")

    layers = tuple(sorted(int(value) for value in test["layer"].unique()))
    if layers != EXPECTED_LAYERS:
        raise ValueError(f"{path} expected layers 0-47 for test rows, found {layers}")

    counts = test.groupby(["protein_id", "layer"], sort=True).size()
    if not (counts == 1).all():
        raise ValueError(f"{path} test rows are not unique per (protein_id, layer)")

    return test.sort_values(["protein_id", "layer"]).reset_index(drop=True)


def layer_mean(df: pd.DataFrame, column: str) -> pd.Series:
    return df.groupby("layer", sort=True)[column].mean()


def add_common_axis_style(ax: plt.Axes) -> None:
    ax.set_xlim(0, 47)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 47])
    ax.grid(True, color=GRID_GREY, linewidth=0.8, alpha=0.7)
    ax.tick_params(axis="both", width=1.0, length=5)


def plot_variant_tm_score(
    ax: plt.Axes,
    variant_a: pd.DataFrame,
    variant_b: pd.DataFrame,
    *,
    title: str = "Held-out TM-score by representation variant",
    legend_fontsize: int = 12,
) -> None:
    for spec, df in [(VARIANT_A, variant_a), (VARIANT_B, variant_b)]:
        mean = layer_mean(df, "tm_score")
        ax.plot(
            mean.index,
            mean.values,
            color=spec["color"],
            linewidth=3.2,
            label=spec["legend_label"],
        )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean held-out TM-score")
    ax.set_ylim(0.28, 0.84)
    add_common_axis_style(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=legend_fontsize)


def plot_tm_trajectories(
    ax: plt.Axes,
    variant_a: pd.DataFrame,
    *,
    title: str = "Held-out protein TM-score trajectories",
    legend_fontsize: int = 12,
) -> None:
    for _, group in variant_a.groupby("protein_id", sort=True):
        ax.plot(
            group["layer"],
            group["tm_score"],
            color=TRAJECTORY_GREY,
            linewidth=0.8,
            alpha=0.45,
            zorder=1,
        )

    mean = layer_mean(variant_a, "tm_score")
    ax.plot(
        mean.index,
        mean.values,
        color=ORANGE,
        linewidth=3.2,
        label="Held-out mean",
        zorder=3,
    )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel("TM-score")
    ax.set_ylim(0.0, 0.98)
    add_common_axis_style(ax)
    handles = [
        Line2D([0], [0], color=TRAJECTORY_GREY, linewidth=1.2, alpha=0.6, label="Held-out proteins"),
        Line2D([0], [0], color=ORANGE, linewidth=3.2, label="Held-out mean"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=legend_fontsize)


def plot_triangle_metric(
    ax: plt.Axes,
    variant_a: pd.DataFrame,
    column: str,
    *,
    title: str,
    ylabel: str,
    ylim: tuple[float, float],
    xlabel: bool,
    show_legend: bool,
    legend_fontsize: int = 12,
) -> None:
    for _, group in variant_a.groupby("protein_id", sort=True):
        ax.plot(
            group["layer"],
            group[column],
            color=TRAJECTORY_GREY,
            linewidth=0.8,
            alpha=0.45,
            zorder=1,
        )

    mean = layer_mean(variant_a, column)
    ax.plot(
        mean.index,
        mean.values,
        color=ORANGE,
        linewidth=3.2,
        label="Held-out mean",
        zorder=3,
    )

    ax.set_title(title, pad=10)
    ax.set_ylabel(ylabel)
    if xlabel:
        ax.set_xlabel("Layer")
    ax.set_ylim(*ylim)
    add_common_axis_style(ax)
    if show_legend:
        handles = [
            Line2D([0], [0], color=TRAJECTORY_GREY, linewidth=1.2, alpha=0.6, label="Held-out proteins"),
            Line2D([0], [0], color=ORANGE, linewidth=3.2, label="Held-out mean"),
        ]
        ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=legend_fontsize)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    for suffix in ("png", "pdf"):
        fig.savefig(OUT_DIR / f"{stem}.{suffix}")
    plt.close(fig)


def create_standalone_figures(variant_a: pd.DataFrame, variant_b: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    plot_variant_tm_score(ax, variant_a, variant_b)
    save_figure(fig, "01_variant_tm_score_comparison")

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    plot_tm_trajectories(ax, variant_a)
    save_figure(fig, "02_avg_symmetric_tm_score_trajectories")

    fig, axes = plt.subplots(2, 1, figsize=(9.2, 8.8), sharex=True, constrained_layout=True)
    plot_triangle_metric(
        axes[0],
        variant_a,
        "triangle_violation_fraction",
        title="Held-out triangle inequality violation fraction",
        ylabel="Violation fraction",
        ylim=(0.2, 1.0),
        xlabel=False,
        show_legend=True,
    )
    plot_triangle_metric(
        axes[1],
        variant_a,
        "triangle_mean_violation",
        title="Held-out triangle inequality violation magnitude",
        ylabel="Mean violation magnitude",
        ylim=(2.0, 11.2),
        xlabel=True,
        show_legend=False,
    )
    save_figure(fig, "03_avg_symmetric_triangle_violations")


def create_combined_figure(variant_a: pd.DataFrame, variant_b: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12.5), constrained_layout=True)

    plot_variant_tm_score(
        axes[0, 0],
        variant_a,
        variant_b,
        title="Mean held-out TM-score",
        legend_fontsize=10,
    )
    plot_tm_trajectories(
        axes[0, 1],
        variant_a,
        title="Held-out TM-score trajectories",
        legend_fontsize=10,
    )
    plot_triangle_metric(
        axes[1, 0],
        variant_a,
        "triangle_violation_fraction",
        title="Triangle inequality violation fraction",
        ylabel="Violation fraction",
        ylim=(0.2, 1.0),
        xlabel=True,
        show_legend=True,
        legend_fontsize=10,
    )
    plot_triangle_metric(
        axes[1, 1],
        variant_a,
        "triangle_mean_violation",
        title="Triangle inequality violation magnitude",
        ylabel="Mean violation magnitude",
        ylim=(2.0, 11.2),
        xlabel=True,
        show_legend=False,
        legend_fontsize=10,
    )

    save_figure(fig, "04_combined_publication_panel")


def write_manifest(output_stems: Iterable[str]) -> None:
    rows = []
    for stem in output_stems:
        rows.append({"figure": stem, "png": f"{stem}.png", "pdf": f"{stem}.pdf"})
    pd.DataFrame(rows).to_csv(OUT_DIR / "publication_plot_manifest.csv", index=False)


def main() -> None:
    configure_style()
    variant_a = load_test_geometry(VARIANT_A["folder"])
    variant_b = load_test_geometry(VARIANT_B["folder"])
    create_standalone_figures(variant_a, variant_b)
    create_combined_figure(variant_a, variant_b)
    write_manifest(
        [
            "01_variant_tm_score_comparison",
            "02_avg_symmetric_tm_score_trajectories",
            "03_avg_symmetric_triangle_violations",
            "04_combined_publication_panel",
        ]
    )


if __name__ == "__main__":
    main()

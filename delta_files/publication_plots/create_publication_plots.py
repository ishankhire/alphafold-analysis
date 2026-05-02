#!/usr/bin/env python3
"""Create publication-focused plots from saved Delta regression outputs."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Iterable

EARLY_PROJECT_ROOT = Path(__file__).resolve().parents[2]
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
OUT_DIR = Path(__file__).resolve().parent
EXPECTED_LAYERS = tuple(range(48))

ORANGE = "#f58518"
RED = "#e45756"
TRAJECTORY_GREY = "#b8b8b8"
GRID_GREY = "#dddddd"

VARIANT_A = {
    "folder": "general_base256_avg_symmetric",
    "legend_label": "Variant A: Averaged (i, j) pair\nrepresentations",
    "color": ORANGE,
}
VARIANT_B = {
    "folder": "general_base256_avg_symmetric_plus_axial",
    "legend_label": "Variant B: Averaged (i, j) pair representations\n+ neighboring pair representations",
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
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def load_test_rows(folder: str, filename: str, value_column: str) -> pd.DataFrame:
    path = DELTA_DIR / folder / filename
    df = pd.read_csv(path)
    required = {"layer", "split", "protein_id", value_column}
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


def add_panel_label(ax: plt.Axes, label: str) -> None:
    ax.text(
        -0.08,
        1.04,
        label,
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=16,
        fontweight="bold",
    )


def plot_variant_comparison(
    ax: plt.Axes,
    variant_a: pd.DataFrame,
    variant_b: pd.DataFrame,
    *,
    column: str,
    title: str,
    ylabel: str,
    legend_fontsize: int = 12,
) -> None:
    for spec, df in [(VARIANT_A, variant_a), (VARIANT_B, variant_b)]:
        mean = layer_mean(df, column)
        ax.plot(
            mean.index,
            mean.values,
            color=spec["color"],
            linewidth=3.2,
            label=spec["legend_label"],
        )

    ax.set_title(title, pad=10)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    add_common_axis_style(ax)
    ax.legend(loc="upper left", frameon=False, fontsize=legend_fontsize)


def plot_orange_trajectories(
    ax: plt.Axes,
    variant_a: pd.DataFrame,
    *,
    column: str,
    title: str,
    ylabel: str,
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
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.set_ylim(0.0, 1.0)
    add_common_axis_style(ax)
    handles = [
        Line2D([0], [0], color=TRAJECTORY_GREY, linewidth=1.2, alpha=0.6, label="Held-out proteins"),
        Line2D([0], [0], color=ORANGE, linewidth=3.2, label="Held-out mean"),
    ]
    ax.legend(handles=handles, loc="lower right", frameon=False, fontsize=legend_fontsize)


def save_figure(fig: plt.Figure, stem: str) -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT_DIR / f"{stem}.png")
    plt.close(fig)


def create_standalone_figures(
    r2_variant_a: pd.DataFrame,
    r2_variant_b: pd.DataFrame,
    tm_variant_a: pd.DataFrame,
    tm_variant_b: pd.DataFrame,
) -> None:
    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    plot_variant_comparison(
        ax,
        r2_variant_a,
        r2_variant_b,
        column="per_protein_r2",
        title="Held-out R$^2$ by representation variant",
        ylabel="Mean held-out R$^2$",
    )
    save_figure(fig, "variant_r2_comparison")

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    plot_orange_trajectories(
        ax,
        r2_variant_a,
        column="per_protein_r2",
        title="Held-out protein R$^2$ trajectories",
        ylabel="R$^2$",
    )
    save_figure(fig, "avg_symmetric_r2_trajectories")

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    plot_variant_comparison(
        ax,
        tm_variant_a,
        tm_variant_b,
        column="tm_score",
        title="Held-out TM-score by representation variant",
        ylabel="Mean held-out TM-score",
    )
    save_figure(fig, "01_variant_tm_score_comparison")

    fig, ax = plt.subplots(figsize=(9.2, 5.6), constrained_layout=True)
    plot_orange_trajectories(
        ax,
        tm_variant_a,
        column="tm_score",
        title="Held-out protein TM-score trajectories",
        ylabel="TM-score",
    )
    save_figure(fig, "02_avg_symmetric_tm_score_trajectories")


def create_combined_figure(
    r2_variant_a: pd.DataFrame,
    r2_variant_b: pd.DataFrame,
    tm_variant_a: pd.DataFrame,
    tm_variant_b: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(17.5, 12.5), constrained_layout=True)

    plot_variant_comparison(
        axes[0, 0],
        r2_variant_a,
        r2_variant_b,
        column="per_protein_r2",
        title="Mean held-out R$^2$",
        ylabel="Mean held-out R$^2$",
        legend_fontsize=10,
    )
    plot_orange_trajectories(
        axes[0, 1],
        r2_variant_a,
        column="per_protein_r2",
        title="Held-out R$^2$ trajectories",
        ylabel="R$^2$",
        legend_fontsize=10,
    )
    plot_variant_comparison(
        axes[1, 0],
        tm_variant_a,
        tm_variant_b,
        column="tm_score",
        title="Mean held-out TM-score",
        ylabel="Mean held-out TM-score",
        legend_fontsize=10,
    )
    plot_orange_trajectories(
        axes[1, 1],
        tm_variant_a,
        column="tm_score",
        title="Held-out TM-score trajectories",
        ylabel="TM-score",
        legend_fontsize=10,
    )

    for ax, label in zip(axes.flat, ["(a)", "(b)", "(c)", "(d)"]):
        add_panel_label(ax, label)

    save_figure(fig, "04_combined_publication_panel")


def write_manifest(output_stems: Iterable[str]) -> None:
    rows = [{"figure": stem, "png": f"{stem}.png"} for stem in output_stems]
    pd.DataFrame(rows).to_csv(OUT_DIR / "publication_plot_manifest.csv", index=False)


def main() -> None:
    configure_style()

    r2_variant_a = load_test_rows(VARIANT_A["folder"], "per_protein.csv", "per_protein_r2")
    r2_variant_b = load_test_rows(VARIANT_B["folder"], "per_protein.csv", "per_protein_r2")
    tm_variant_a = load_test_rows(VARIANT_A["folder"], "geometry_metrics.csv", "tm_score")
    tm_variant_b = load_test_rows(VARIANT_B["folder"], "geometry_metrics.csv", "tm_score")

    create_standalone_figures(r2_variant_a, r2_variant_b, tm_variant_a, tm_variant_b)
    create_combined_figure(r2_variant_a, r2_variant_b, tm_variant_a, tm_variant_b)
    write_manifest(
        [
            "variant_r2_comparison",
            "avg_symmetric_r2_trajectories",
            "01_variant_tm_score_comparison",
            "02_avg_symmetric_tm_score_trajectories",
            "04_combined_publication_panel",
        ]
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local analysis of the SMACOF error decomposition (552 proteins, 4 probes, 48 layers).

Reads the aggregated tables under
``delta_files/diagnostic_experiments/smacof_error_decomposition_5fold_v1/tables`` and writes
``summary_table.csv``, ``cross_share_by_layer.csv``, ``RESULTS.md``, ``sanity_checks.md``
and the figures. Every cohort curve averages a squared term equally across proteins (control
draws are already averaged within protein) and shows its square root in angstroms. Bands are
the paper's paired-protein bootstrap on the mean of squares, with the square root applied to
the interval ends. The cross share is the ratio of cohort means, bootstrapped as a ratio.
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/multi_protein"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from generate_sept_2_publication_plots import (  # noqa: E402
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, MILESTONE_LAYERS, bootstrap_counts, bootstrap_matrix_summary,
    configure_style,
)

SMOKE = "SMACOF_DECOMPOSITION_ANALYSIS_ROOT" in os.environ          # local smoke tests only
ROOT = Path(os.environ.get(
    "SMACOF_DECOMPOSITION_ANALYSIS_ROOT",
    PROJECT_ROOT / "delta_files/diagnostic_experiments/smacof_error_decomposition_5fold_v1"))
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
REGRESSION = PROJECT_ROOT / "delta_files/regression_5fold_v1"
TRIANGLE_TABLE = REGRESSION / "analysis/null_triangles_5fold_v1/tables/null_per_protein.csv.gz"
LAYERS = np.arange(48)
TABLE_LAYERS = (0, 10, 21, 32, 40, 47)
EXPECTED_PROTEINS = 552
OUTLIER = "7stz_C"        # one protein whose squared errors dominate cohort means of squares
# Paper order and names: shared linear, per-protein linear, XGBoost, sequence-gated.
PROBES = ("shared_ols_base256", "per_protein_ols_base256", "shared_xgboost_base256_fixed_s4",
          "sequence_gated_ols_base256")
PROBE_LABELS = {"shared_ols_base256": "Shared linear", "per_protein_ols_base256": "Per-protein linear",
                "sequence_gated_ols_base256": "Sequence-gated", "shared_xgboost_base256_fixed_s4": "XGBoost"}
PROBE_COLORS = {"shared_ols_base256": "#2a6fb0", "per_protein_ols_base256": "#d9822b",
                "sequence_gated_ols_base256": "#2a9d8f", "shared_xgboost_base256_fixed_s4": "#9b4dca"}
PROBE_STYLES = {"shared_ols_base256": "-", "per_protein_ols_base256": "--",
                "sequence_gated_ols_base256": "-.", "shared_xgboost_base256_fixed_s4": ":"}
MATRIX_TYPES = ("observed", "control_gaussian", "control_shuffled")
CONTROLS = MATRIX_TYPES[1:]
SERIES_COLORS = {"observed": "#222222", "control_gaussian": "#d9822b", "control_shuffled": "#2a6fb0"}
SERIES_LABELS = {"observed": "Observed", "control_gaussian": "Control: Gaussian-matched noise",
                 "control_shuffled": "Control: shuffled errors"}
PART_STYLES = {"inconsistent": "-", "consistent": "--"}
PART_COLORS = {"inconsistent": "#2a6fb0", "consistent": "#d9822b"}
VIOLATION_COLOR = "#b5403a"
RMS_QUANTITIES = ("total_sq", "smacof_inconsistent_sq", "smacof_consistent_sq",
                  "classical_inconsistent_sq", "classical_consistent_sq")

# Wide multi-panel canvases need a larger scale than the 8.6 in single-panel figures (1.75),
# so text reads at the same size once each figure is shrunk to the page.
FONT_SCALE = 3.0
SINGLE_PANEL_FONT_SCALE = 1.75
LINE_WIDTH = 3.0
Y_TICKS = 5


def wide_style() -> None:
    configure_style(font_scale=FONT_SCALE)
    plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.color": "#e6e6e6", "grid.linewidth": 1.0, "grid.alpha": 1.0, "axes.axisbelow": True,
                         "legend.frameon": False, "figure.titlesize": 12 * FONT_SCALE,
                         "lines.linewidth": LINE_WIDTH, "legend.handlelength": 2.6,
                         "axes.linewidth": 1.6, "xtick.major.size": 8, "ytick.major.size": 8,
                         "xtick.major.width": 1.6, "ytick.major.width": 1.6})


wide_style()


# ---------------------------------------------------------------- statistics

def rms_summary(squares: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Protein-equal mean of a squared term, shown as its square root, with the bootstrap band."""
    s = bootstrap_matrix_summary(squares, counts)
    root = lambda v: np.sqrt(np.maximum(v, 0.0))  # noqa: E731
    return {"mean_sq": s["mean"], "sq_ci_low": s["bootstrap_ci_low"], "sq_ci_high": s["bootstrap_ci_high"],
            "mean": root(s["mean"]), "bootstrap_ci_low": root(s["bootstrap_ci_low"]),
            "bootstrap_ci_high": root(s["bootstrap_ci_high"])}


def ratio_summary(numerator: np.ndarray, denominator: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Ratio of protein-equal means with a paired-protein bootstrap of the ratio."""
    numerator, denominator = np.asarray(numerator, float), np.asarray(denominator, float)
    if not (np.all(np.isfinite(numerator)) and np.all(np.isfinite(denominator))):
        raise ValueError("The ratio bootstrap needs finite values for every protein")
    boot = (counts @ numerator) / (counts @ denominator)
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    return {"mean": numerator.mean(axis=0) / denominator.mean(axis=0),
            "bootstrap_ci_low": low, "bootstrap_ci_high": high}


# ---------------------------------------------------------------- data

def load() -> dict:
    per = pd.read_csv(TABLES / "decomposition_per_protein.csv.gz", dtype={"protein_id": str})
    cohort = sorted(per["protein_id"].unique())
    if len(cohort) != EXPECTED_PROTEINS and not SMOKE:
        raise ValueError(f"Expected {EXPECTED_PROTEINS} proteins, found {len(cohort)}")
    if sorted(per["layer"].unique()) != LAYERS.tolist():
        raise ValueError("The per-protein table does not cover layers 0-47")
    triangles = pd.read_csv(TRIANGLE_TABLE, dtype={"protein_id": str})
    triangles = triangles[(triangles.model == "observed") & triangles.protein_id.isin(cohort)]
    geometry, accuracy = [], []
    for probe in PROBES:
        frame = pd.read_csv(REGRESSION / "tm_score_optimized_v1" / probe / "geometry_metrics.csv",
                            dtype={"protein_id": str}, usecols=["layer", "protein_id", "tm_score", "rmsd"])
        geometry.append(frame.assign(probe=probe))
        frame = pd.read_csv(REGRESSION / probe / "per_protein.csv", dtype={"protein_id": str},
                            usecols=["layer", "protein_id", "rmse"])
        accuracy.append(frame.assign(probe=probe))
    return {
        "per": per, "cohort": cohort, "triangles": triangles,
        "classical_geometry": pd.concat(geometry), "accuracy": pd.concat(accuracy),
        "bins": pd.read_csv(TABLES / "bin_terms_cohort.csv"),
        "convergence": pd.read_csv(TABLES / "convergence_summary.csv"),
        "manifest": json.loads((TABLES / "run_manifest.json").read_text()),
    }


def pivot(frame: pd.DataFrame, cohort: list[str], column: str) -> np.ndarray:
    if frame.empty:
        raise ValueError(f"No rows for {column}")
    return frame.pivot(index="protein_id", columns="layer", values=column).reindex(
        index=cohort, columns=LAYERS).to_numpy(float)


def matrix(data: dict, probe: str, matrix_type: str, column: str) -> np.ndarray:
    """Proteins x layers for one series."""
    per = data["per"]
    return pivot(per[(per.probe == probe) & (per.matrix_type == matrix_type)], data["cohort"], column)


# ---------------------------------------------------------------- figure helpers

def five_y_ticks(ax) -> None:
    """Exactly Y_TICKS round-valued y ticks spanning the data, with the limits snapped to them."""
    low = 0.0 if ax.get_ylim()[0] == 0 else ax.dataLim.y0
    high = ax.dataLim.y1
    if abs(high) < 1e-9 * (high - low):
        high = 0.0
    headroom = 0.01 * (high - low)
    magnitude = 10.0 ** np.floor(np.log10((high - low) / (Y_TICKS - 1)))
    for step in magnitude * np.array([1, 1.5, 2, 2.5, 3, 4, 5, 10]):
        start = np.floor((low - headroom) / step + 1e-9) * step if low != 0 else 0.0
        if start + (Y_TICKS - 1) * step >= high + (headroom if high != 0 else 0) - 1e-9 * step:
            break
    ticks = start + step * np.arange(Y_TICKS)
    ax.set_yticks(ticks)
    ax.set_ylim(ticks[0], ticks[-1])


def save(fig, folder: str, name: str, layer_axis: bool = True) -> None:
    if layer_axis:
        for ax in fig.axes:
            five_y_ticks(ax)
            ax.set_xlim(0, 47)
            ax.set_xticks(range(0, 48, 10))
    directory = FIGURES / folder
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def band(ax, s, color, label=None, style="-", alpha=0.18, lw=LINE_WIDTH):
    ax.plot(LAYERS, s["mean"], style, color=color, lw=lw, label=label)
    ax.fill_between(LAYERS, s["bootstrap_ci_low"], s["bootstrap_ci_high"], color=color, alpha=alpha, lw=0)


def probe_grid(figsize=(19, 12.5), sharey=True):
    fig, axes = plt.subplots(2, 2, figsize=figsize, sharex=True, sharey=sharey, layout="constrained")
    return fig, axes.ravel()


# ---------------------------------------------------------------- main figures

def without_outlier(data: dict) -> dict:
    """The same tables with the outlier protein left out of the cohort."""
    return {**data, "cohort": [pid for pid in data["cohort"] if pid != OUTLIER]}


def figure_parts_by_layer(data, counts, folder="main", name="inconsistent_consistent_by_layer") -> None:
    """Plot 1. Inconsistent (solid) and consistent (dashed) error by layer; controls overlaid.

    The two controls nearly coincide, so the Gaussian one is drawn last as a thinner line.
    """
    widths = {"control_shuffled": LINE_WIDTH + 1.5, "observed": LINE_WIDTH, "control_gaussian": LINE_WIDTH - 1.4}
    fig, axes = probe_grid()
    for ax, probe in zip(axes, PROBES):
        for matrix_type, width in widths.items():
            for part, style in PART_STYLES.items():
                s = rms_summary(matrix(data, probe, matrix_type, f"smacof_{part}_sq"), counts)
                band(ax, s, SERIES_COLORS[matrix_type], style=style, lw=width,
                     alpha=0.0 if matrix_type == "control_gaussian" else 0.12)
        ax.set_title(PROBE_LABELS[probe])
        ax.set_ylim(bottom=0)
    for ax in axes[2:]:
        ax.set_xlabel("Layer")
    for ax in axes[::2]:
        ax.set_ylabel("RMS error (Å)")
    handles = [Line2D([], [], color=SERIES_COLORS[m], lw=widths[m], label=SERIES_LABELS[m]) for m in MATRIX_TYPES]
    handles += [Line2D([], [], color="#666666", lw=LINE_WIDTH, ls=style, label=f"{part.capitalize()} part")
                for part, style in PART_STYLES.items()]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    save(fig, folder, name)


def cross_share(data, probe, matrix_type, counts):
    return ratio_summary(matrix(data, probe, matrix_type, "smacof_cross"),
                         matrix(data, probe, matrix_type, "total_sq"), counts)


def figure_cross_share(data, counts) -> None:
    """Plot 2. Cross term as a share of total squared error (ratio of cohort means), observed."""
    fig, ax = plt.subplots(figsize=(14, 7.5), layout="constrained")
    for probe in PROBES:
        band(ax, cross_share(data, probe, "observed", counts), PROBE_COLORS[probe], PROBE_LABELS[probe],
             PROBE_STYLES[probe])
    ax.axhline(0, color="#888888", lw=1.6)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Cross term / total²")
    fig.legend(*ax.get_legend_handles_labels(), loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    save(fig, "main", "cross_share_by_layer")


def figure_inconsistent_vs_triangles(data, counts) -> None:
    """Plot 3. Inconsistent error (left axis) on the normalized triangle-violation curve (right axis)."""
    fig, axes = probe_grid(sharey=False)
    for index, (ax, probe) in enumerate(zip(axes, PROBES)):
        band(ax, rms_summary(matrix(data, probe, "observed", "smacof_inconsistent_sq"), counts),
             PART_COLORS["inconsistent"])
        ax.set_title(PROBE_LABELS[probe])
        ax.set_ylim(bottom=0)
        ax.tick_params(axis="y", colors=PART_COLORS["inconsistent"])
        twin = ax.twinx()
        violation = pivot(data["triangles"][data["triangles"].probe == probe], data["cohort"],
                          "mean_relative_violation_all")
        band(twin, bootstrap_matrix_summary(violation, counts), VIOLATION_COLOR, style="--")
        twin.set_ylim(bottom=0)
        twin.grid(False)
        twin.spines["right"].set_visible(True)
        twin.tick_params(axis="y", colors=VIOLATION_COLOR)
        if index % 2 == 0:
            ax.set_ylabel("Inconsistent error (Å)", color=PART_COLORS["inconsistent"])
        else:
            twin.set_ylabel("Normalized mean\nviolation", color=VIOLATION_COLOR)
        if index >= 2:
            ax.set_xlabel("Layer")
    handles = [Line2D([], [], color=PART_COLORS["inconsistent"], lw=LINE_WIDTH, label="Inconsistent error (left axis)"),
               Line2D([], [], color=VIOLATION_COLOR, lw=LINE_WIDTH, ls="--",
                      label="Normalized mean triangle violation (right axis)")]
    fig.legend(handles=handles, loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.01))
    save(fig, "main", "inconsistent_vs_triangle_violation")


def figure_phase(data, counts, folder="main", name="consistent_vs_inconsistent_phase") -> None:
    """Plot 4. Consistent (x) against inconsistent (y) error, layers labelled, as in Figure 2b."""
    with plt.rc_context():
        configure_style(font_scale=SINGLE_PANEL_FONT_SCALE)
        fig, ax = plt.subplots(figsize=(8.7, 6.4))
        label_shifts = (-8, 8, -7, 7)
        milestones = list(MILESTONE_LAYERS)
        for index, probe in enumerate(PROBES):
            x = rms_summary(matrix(data, probe, "observed", "smacof_consistent_sq"), counts)
            y = rms_summary(matrix(data, probe, "observed", "smacof_inconsistent_sq"), counts)
            color = PROBE_COLORS[probe]
            ax.plot(x["mean"], y["mean"], color=color, ls=PROBE_STYLES[probe], lw=2.0, label=PROBE_LABELS[probe], zorder=2)
            ax.scatter(x["mean"], y["mean"], color=color, s=10, zorder=3)
            xm, ym = x["mean"][milestones], y["mean"][milestones]
            ax.errorbar(xm, ym,
                        xerr=np.vstack([xm - x["bootstrap_ci_low"][milestones], x["bootstrap_ci_high"][milestones] - xm]),
                        yerr=np.vstack([ym - y["bootstrap_ci_low"][milestones], y["bootstrap_ci_high"][milestones] - ym]),
                        fmt="none", ecolor=color, elinewidth=0.8, alpha=0.42, capsize=1.5, zorder=1)
            ax.scatter(xm, ym, color=color, s=31, edgecolors="white", linewidths=0.65, zorder=4)
            for layer, px, py in zip(milestones, xm, ym):
                ax.annotate(str(layer), (px, py), xytext=(4, label_shifts[index]), textcoords="offset points",
                            fontsize=6.2 * SINGLE_PANEL_FONT_SCALE, color=color,
                            bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.1})
        ax.set(xlabel="Consistent error (Å)", ylabel="Inconsistent error (Å)")
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)
        ax.locator_params(axis="both", nbins=5)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        save(fig, folder, name, layer_axis=False)


# ---------------------------------------------------------------- diagnostic figures

def figure_cross_share_controls(data, counts) -> None:
    fig, axes = probe_grid()
    for ax, probe in zip(axes, PROBES):
        for matrix_type in MATRIX_TYPES:
            band(ax, cross_share(data, probe, matrix_type, counts), SERIES_COLORS[matrix_type], SERIES_LABELS[matrix_type])
        ax.axhline(0, color="#888888", lw=1.6)
        ax.set_title(PROBE_LABELS[probe])
    for ax in axes[2:]:
        ax.set_xlabel("Layer")
    for ax in axes[::2]:
        ax.set_ylabel("Cross term / total²")
    fig.legend(*axes[0].get_legend_handles_labels(), loc="upper center", ncol=1, bbox_to_anchor=(0.5, -0.01))
    save(fig, "diagnostics", "cross_share_with_controls")


def figure_smacof_vs_classical_split(data, counts) -> None:
    """The same split with the classical-MDS reconstruction (dashed) against SMACOF (solid), observed."""
    fig, axes = probe_grid()
    for ax, probe in zip(axes, PROBES):
        for part, color in PART_COLORS.items():
            band(ax, rms_summary(matrix(data, probe, "observed", f"smacof_{part}_sq"), counts), color, style="-")
            band(ax, rms_summary(matrix(data, probe, "observed", f"classical_{part}_sq"), counts), color, style=":")
        ax.set_title(PROBE_LABELS[probe])
        ax.set_ylim(bottom=0)
    for ax in axes[2:]:
        ax.set_xlabel("Layer")
    for ax in axes[::2]:
        ax.set_ylabel("RMS error (Å)")
    handles = [Line2D([], [], color=color, lw=LINE_WIDTH, label=f"{part.capitalize()} part")
               for part, color in PART_COLORS.items()]
    handles += [Line2D([], [], color="#666666", lw=LINE_WIDTH, ls="-", label="SMACOF"),
                Line2D([], [], color="#666666", lw=LINE_WIDTH, ls=":", label="Classical MDS")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    save(fig, "diagnostics", "smacof_vs_classical_split")


def figure_reconstruction_quality(data, counts) -> None:
    """TM-score and coordinate RMSD of the SMACOF (solid) and classical-MDS (dotted) reconstructions."""
    fig, axes = plt.subplots(1, 2, figsize=(19, 7.5), sharex=True, layout="constrained")
    classical = data["classical_geometry"]
    for probe in PROBES:
        color = PROBE_COLORS[probe]
        rows = classical[classical.probe == probe]
        band(axes[0], bootstrap_matrix_summary(matrix(data, probe, "observed", "smacof_tm_score"), counts), color,
             PROBE_LABELS[probe])
        band(axes[0], bootstrap_matrix_summary(pivot(rows, data["cohort"], "tm_score"), counts), color, style=":")
        band(axes[1], bootstrap_matrix_summary(matrix(data, probe, "observed", "smacof_rmsd"), counts), color)
        band(axes[1], bootstrap_matrix_summary(matrix(data, probe, "observed", "classical_rmsd"), counts), color, style=":")
    axes[0].set_ylabel("TM-score")
    axes[1].set_ylabel("Coordinate RMSD (Å)")
    for ax in axes:
        ax.set_xlabel("Layer")
        ax.set_ylim(bottom=0)
    handles, labels = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color="#666666", lw=LINE_WIDTH, ls="-", label="SMACOF"),
                Line2D([], [], color="#666666", lw=LINE_WIDTH, ls=":", label="Classical MDS")]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.01))
    save(fig, "diagnostics", "smacof_vs_classical_reconstruction_quality")


def figure_convergence(data) -> None:
    """Median (solid) and 95th-percentile (dotted) SMACOF iterations by layer."""
    fig, axes = probe_grid()
    convergence = data["convergence"]
    for ax, probe in zip(axes, PROBES):
        for matrix_type in MATRIX_TYPES:
            rows = convergence[(convergence.probe == probe) & (convergence.matrix_type == matrix_type)].sort_values("layer")
            ax.plot(rows.layer, rows.iterations_median, "-", color=SERIES_COLORS[matrix_type], lw=LINE_WIDTH,
                    label=SERIES_LABELS[matrix_type])
            ax.plot(rows.layer, rows.iterations_p95, ":", color=SERIES_COLORS[matrix_type], lw=LINE_WIDTH)
        ax.set_title(PROBE_LABELS[probe])
        ax.set_ylim(bottom=0)
    for ax in axes[2:]:
        ax.set_xlabel("Layer")
    for ax in axes[::2]:
        ax.set_ylabel("SMACOF iterations")
    handles, _ = axes[0].get_legend_handles_labels()
    handles += [Line2D([], [], color="#666666", lw=LINE_WIDTH, ls="-", label="Median"),
                Line2D([], [], color="#666666", lw=LINE_WIDTH, ls=":", label="95th percentile")]
    fig.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.01))
    save(fig, "diagnostics", "smacof_iterations_by_layer")


def figure_bins(data, system: str, name: str, legend_title: str) -> None:
    """Observed inconsistent (top) and consistent (bottom) error by bin, protein-equal, per probe."""
    bins = data["bins"]
    bins = bins[(bins.bin_system == system) & (bins.matrix_type == "observed")]
    labels = bins.drop_duplicates("bin").sort_values("bin")["bin_label"].tolist()
    colors = plt.get_cmap("viridis")(np.linspace(0.0, 0.92, len(labels)))
    fig, axes = plt.subplots(2, len(PROBES), figsize=(24, 11), sharex=True, sharey="row", layout="constrained")
    for column, probe in enumerate(PROBES):
        for row, part in enumerate(("inconsistent", "consistent")):
            ax = axes[row, column]
            for label, color in zip(labels, colors):
                cell = bins[(bins.probe == probe) & (bins.bin_label == label)].set_index("layer").reindex(LAYERS)
                ax.plot(LAYERS, np.sqrt(cell[f"protein_mean_smacof_{part}_sq"].clip(lower=0)), color=color,
                        lw=LINE_WIDTH, label=label.replace("inf", "∞"))
            ax.set_ylim(bottom=0)
            if row == 0:
                ax.set_title(PROBE_LABELS[probe])
            else:
                ax.set_xlabel("Layer")
            if column == 0:
                ax.set_ylabel(f"{part.capitalize()}\nerror (Å)")
    fig.legend(*axes[0, 0].get_legend_handles_labels(), loc="upper center", ncol=min(len(labels), 5),
               bbox_to_anchor=(0.5, -0.01), title=legend_title)
    save(fig, "diagnostics", name)


# ---------------------------------------------------------------- tables and text

def summary_tables(data, counts) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows, shares = [], []
    for probe in PROBES:
        for matrix_type in MATRIX_TYPES:
            for quantity in RMS_QUANTITIES:
                s = rms_summary(matrix(data, probe, matrix_type, quantity), counts)
                for layer in LAYERS:
                    rows.append({"probe": probe, "matrix_type": matrix_type, "quantity": quantity, "layer": layer,
                                 "protein_mean_sq": s["mean_sq"][layer], "sq_ci_low": s["sq_ci_low"][layer],
                                 "sq_ci_high": s["sq_ci_high"][layer], "rms_angstrom": s["mean"][layer],
                                 "rms_ci_low": s["bootstrap_ci_low"][layer], "rms_ci_high": s["bootstrap_ci_high"][layer]})
            for quantity in ("smacof_cross", "classical_cross"):
                s = bootstrap_matrix_summary(matrix(data, probe, matrix_type, quantity), counts)
                for layer in LAYERS:
                    rows.append({"probe": probe, "matrix_type": matrix_type, "quantity": quantity, "layer": layer,
                                 "protein_mean_sq": s["mean"][layer], "sq_ci_low": s["bootstrap_ci_low"][layer],
                                 "sq_ci_high": s["bootstrap_ci_high"][layer], "rms_angstrom": np.nan,
                                 "rms_ci_low": np.nan, "rms_ci_high": np.nan})
            for split in ("smacof", "classical"):
                s = ratio_summary(matrix(data, probe, matrix_type, f"{split}_cross"),
                                  matrix(data, probe, matrix_type, "total_sq"), counts)
                for layer in LAYERS:
                    shares.append({"probe": probe, "matrix_type": matrix_type, "split": split, "layer": layer,
                                   "cross_share": s["mean"][layer], "ci_low": s["bootstrap_ci_low"][layer],
                                   "ci_high": s["bootstrap_ci_high"][layer]})
    return pd.DataFrame(rows), pd.DataFrame(shares)


def markdown_table(frame: pd.DataFrame) -> str:
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(str(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def contiguous_ranges(layers: np.ndarray) -> str:
    if layers.size == 0:
        return "none"
    parts, start = [], layers[0]
    for prev, cur in zip(layers[:-1], layers[1:]):
        if cur != prev + 1:
            parts.append(f"L{start}" if start == prev else f"L{start}–L{prev}")
            start = cur
    parts.append(f"L{start}" if start == layers[-1] else f"L{start}–L{layers[-1]}")
    return ", ".join(parts)


def interval(s: dict, layer: int, digits: int = 2) -> str:
    return (f"{s['mean'][layer]:.{digits}f} [{s['bootstrap_ci_low'][layer]:.{digits}f}, "
            f"{s['bootstrap_ci_high'][layer]:.{digits}f}]")


def results_markdown(data, counts) -> str:
    smacof = data["manifest"]["smacof"]
    lines = [
        "# SMACOF error decomposition: results", "",
        f"{len(data['cohort'])} proteins, 4 probes, layers 0–47, OOF all pairs. Each value is the square root of a "
        "protein-equal mean of a squared term, in Å, with a 95% paired-protein bootstrap interval "
        f"({BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}). Control values average the 3 draws within "
        "each protein first. The cross share is mean cross / mean total², bootstrapped as a ratio.", "",
        f"SMACOF: metric, 3D, raw unweighted stress in Å, one start from classical MDS, relative-decrease "
        f"tolerance {smacof['rel_tol']:g}, at most {smacof['max_iter']} iterations.", "",
    ]
    for probe in PROBES:
        parts = {(m, p): rms_summary(matrix(data, probe, m, f"smacof_{p}_sq"), counts)
                 for m in MATRIX_TYPES for p in PART_STYLES}
        total = rms_summary(matrix(data, probe, "observed", "total_sq"), counts)
        share = cross_share(data, probe, "observed", counts)
        table = pd.DataFrame([{
            "Layer": layer, "Total": f"{total['mean'][layer]:.2f}",
            "Inconsistent": interval(parts[("observed", "inconsistent")], layer),
            "Consistent": interval(parts[("observed", "consistent")], layer),
            "Cross share": interval(share, layer, 3),
            "Inconsistent, Gaussian control": interval(parts[("control_gaussian", "inconsistent")], layer),
            "Inconsistent, shuffled control": interval(parts[("control_shuffled", "inconsistent")], layer),
            "Consistent, Gaussian control": interval(parts[("control_gaussian", "consistent")], layer),
            "Consistent, shuffled control": interval(parts[("control_shuffled", "consistent")], layer),
        } for layer in TABLE_LAYERS])
        lines += [f"## {PROBE_LABELS[probe]}", "", markdown_table(table), ""]
        for part in PART_STYLES:
            for control in CONTROLS:
                gap = bootstrap_matrix_summary(matrix(data, probe, "observed", f"smacof_{part}_sq")
                                               - matrix(data, probe, control, f"smacof_{part}_sq"), counts)
                below = LAYERS[gap["bootstrap_ci_high"] < 0]
                above = LAYERS[gap["bootstrap_ci_low"] > 0]
                name = SERIES_LABELS[control].split(": ")[1]
                lines.append(f"- {part.capitalize()}², observed minus the {name} control: below 0 at "
                             f"{contiguous_ranges(below)}; above 0 at {contiguous_ranges(above)} "
                             "(95% interval of the paired difference excludes 0).")
        lines.append("")

    if OUTLIER in data["cohort"]:
        trimmed = without_outlier(data)
        trimmed_counts = bootstrap_counts(n_proteins=len(trimmed["cohort"]))
        lines += [f"## Sensitivity to {OUTLIER}", "",
                  f"Cohort curves average squared terms, so one protein with very large errors has a lot of weight. "
                  f"`{OUTLIER}` is that protein (it is also the outlier in the triangle analysis). The table gives each "
                  f"observed curve with all {len(data['cohort'])} proteins and without it, and its share of the cohort "
                  "sum of consistent².", ""]
        rows = []
        for probe in PROBES:
            consistent = matrix(data, probe, "observed", "smacof_consistent_sq")
            share = consistent[data["cohort"].index(OUTLIER)] / consistent.sum(axis=0)
            parts = {(label, part): rms_summary(matrix(source, probe, "observed", f"smacof_{part}_sq"), source_counts)
                     for label, source, source_counts in (("all", data, counts), ("without", trimmed, trimmed_counts))
                     for part in PART_STYLES}
            for layer in TABLE_LAYERS:
                rows.append({"Probe": PROBE_LABELS[probe], "Layer": layer,
                             "Inconsistent, all": interval(parts[("all", "inconsistent")], layer),
                             "Inconsistent, without": interval(parts[("without", "inconsistent")], layer),
                             "Consistent, all": interval(parts[("all", "consistent")], layer),
                             "Consistent, without": interval(parts[("without", "consistent")], layer),
                             "Share of consistent²": f"{share[layer]:.0%}"})
        lines += [markdown_table(pd.DataFrame(rows)), "",
                  f"Figures without it are in `figures/sensitivity/`. Across probes and layers its share of the cohort "
                  f"consistent² ranges up to "
                  f"{max(float((matrix(data, p, 'observed', 'smacof_consistent_sq')[data['cohort'].index(OUTLIER)] / matrix(data, p, 'observed', 'smacof_consistent_sq').sum(axis=0)).max()) for p in PROBES):.0%}.", ""]

    lines += ["## Unconverged SMACOF runs", ""]
    per = data["per"]
    unconverged = per[per.n_unconverged > 0]
    total_runs = int(per.n_draws.sum())
    if unconverged.empty:
        lines.append(f"None: all {total_runs} runs met the tolerance before the iteration cap.")
    else:
        lines.append(f"{int(per.n_unconverged.sum())} of {total_runs} runs hit the iteration cap "
                     f"({unconverged.shape[0]} protein × layer × probe × matrix-type cells). They stay in every curve above.")
    if not unconverged.empty:
        by_type = unconverged.groupby("matrix_type").n_unconverged.sum().to_dict()
        lines.append(f"By matrix type: {by_type}. Layers affected: "
                     f"{contiguous_ranges(np.array(sorted(unconverged.layer.unique())))}.")
        worst, emptied = 0.0, 0
        for probe in PROBES:
            for matrix_type in MATRIX_TYPES:
                flagged = matrix(data, probe, matrix_type, "n_unconverged") > 0
                supported = ~flagged.all(axis=0)          # layers that keep at least one protein
                emptied += int((~supported).sum())
                for part in PART_STYLES:
                    values = matrix(data, probe, matrix_type, f"smacof_{part}_sq")
                    full = np.sqrt(values.mean(axis=0))
                    kept = np.sqrt(np.nanmean(np.where(flagged, np.nan, values)[:, supported], axis=0))
                    if supported.any():
                        worst = max(worst, float(np.max(np.abs(full[supported] - kept))))
        lines.append(f"Dropping the affected proteins at the affected layers changes no cohort curve "
                     f"(inconsistent or consistent, any probe or matrix type) by more than {worst:.4f} Å."
                     + (f" {emptied} probe × matrix-type × layer curve points have an unconverged run in every "
                        "protein and cannot be compared this way." if emptied else ""))
    lines.append("")
    return "\n".join(lines)


def sanity_markdown(data, counts) -> str:
    manifest, per = data["manifest"], data["per"]
    draws = pd.read_csv(TABLES / "decomposition_per_draw.csv.gz", dtype={"protein_id": str})
    checks = manifest["checks"]
    converged = draws[draws.smacof_converged == 1]
    lines = [
        "# SMACOF error decomposition: sanity checks", "",
        "## Gates (all layers passed before anything was written)", "",
        f"- Identity and stress: {checks['identity_and_stress_rows']} rows checked, {checks['failures']} failures. "
        f"Largest |total² − (inconsistent² + consistent² + cross)|: SMACOF "
        f"{draws.smacof_identity_residual.abs().max():.2e} Å², classical {draws.classical_identity_residual.abs().max():.2e} Å².",
        f"- SMACOF raw stress ≤ classical raw stress on every row; largest ratio "
        f"{(draws.smacof_raw_stress / draws.classical_raw_stress).max():.6f}.",
        f"- Control pairing with the triangle run: {checks['triangle_pairing_vectors']} vectors matched on "
        f"negative-value count (and violation count on small proteins); {checks['histogram_checks']} shuffled-control "
        "histogram checks.",
        f"- Classical-MDS start vs canonical geometry: worst gaps {checks['classical_parity_worst']}.", "",
        "## Stress minimum", "",
        f"- Scale ratio Σd̂·p / Σp² over converged runs: min {converged.smacof_scale_ratio.min():.8f}, "
        f"max {converged.smacof_scale_ratio.max():.8f} (1 at a stress minimum). Classical MDS: "
        f"min {draws.classical_scale_ratio.min():.4f}, max {draws.classical_scale_ratio.max():.4f}.",
        f"- Converged runs: {len(converged)} of {len(draws)}. Iterations: median {draws.smacof_iterations.median():.0f}, "
        f"95th percentile {draws.smacof_iterations.quantile(0.95):.0f}, max {draws.smacof_iterations.max()}.", "",
        "## Tolerance", "",
    ]
    lines.append(f"Runs stopped at a relative stress decrease of {manifest['smacof']['rel_tol']:g} per iteration. "
                 "Change in each row's RMS term had the run stopped when a looser tolerance was first met:")
    lines += ["", "| Stop at | Part | Rows changed by > 1e-4 Å | > 1e-2 Å | Largest change (Å) | "
              "Largest change in a cohort curve (Å) |", "|---|---|---|---|---|---|"]
    prefix = "smacof_inconsistent_sq_at_"
    for label in [c[len(prefix):] for c in draws.columns if c.startswith(prefix)]:
        for part in ("inconsistent_sq", "consistent_sq"):
            at = draws[f"smacof_{part}_at_{label}"]
            if at.isna().all() or float(label) <= manifest["smacof"]["rel_tol"]:
                continue                                    # not looser than the tolerance the run used
            change = (np.sqrt(at.clip(lower=0)) - np.sqrt(draws[f"smacof_{part}"].clip(lower=0))).abs()
            keys = ["probe", "matrix_type", "layer"]
            cohort = (np.sqrt(draws.assign(at=at).groupby(keys)["at"].mean())
                      - np.sqrt(draws.groupby(keys)[f"smacof_{part}"].mean())).abs().max()
            lines.append(f"| {label} | {part.replace('_sq', '')} | {int((change > 1e-4).sum())} | "
                         f"{int((change > 1e-2).sum())} | {change.max():.3f} | {cohort:.5f} |")
    lines += ["", "## Agreement with the canonical accuracy tables", ""]
    observed = per[per.matrix_type == "observed"].merge(data["accuracy"], on=["probe", "layer", "protein_id"])
    unclipped = observed[observed.n_negative_values == 0]
    gap = (np.sqrt(unclipped.total_sq) - unclipped.rmse).abs()
    lines.append(f"- √total² against the canonical per-protein RMSE, rows with no negative prediction "
                 f"({len(unclipped)} of {len(observed)}): largest gap {gap.max():.2e} Å (the cache is float32).")
    lines += ["", "## Monte Carlo spread of the controls (3 draws)", "",
              "| Control | Term | Mean per-protein draw SD (Å²) | Median bootstrap half-width of the cohort mean (Å²) |",
              "|---|---|---|---|"]
    for control in CONTROLS:
        rows = per[per.matrix_type == control]
        for term in ("smacof_inconsistent_sq", "smacof_consistent_sq"):
            widths = []
            for probe in PROBES:
                s = bootstrap_matrix_summary(matrix(data, probe, control, term), counts)
                widths.append(np.median((s["bootstrap_ci_high"] - s["bootstrap_ci_low"]) / 2))
            lines.append(f"| {SERIES_LABELS[control]} | {term} | {rows[f'{term}_draw_sd'].mean():.4g} | "
                         f"{np.median(widths):.4g} |")
    lines.append("")
    return "\n".join(lines)


def main() -> None:
    data = load()
    counts = bootstrap_counts(n_proteins=len(data["cohort"]))
    summary, shares = summary_tables(data, counts)
    summary.to_csv(ROOT / "summary_table.csv", index=False)
    shares.to_csv(ROOT / "cross_share_by_layer.csv", index=False)
    figure_parts_by_layer(data, counts)
    figure_cross_share(data, counts)
    figure_inconsistent_vs_triangles(data, counts)
    figure_phase(data, counts)
    if OUTLIER in data["cohort"]:
        trimmed = without_outlier(data)
        trimmed_counts = bootstrap_counts(n_proteins=len(trimmed["cohort"]))
        figure_parts_by_layer(trimmed, trimmed_counts, "sensitivity", f"inconsistent_consistent_by_layer_without_{OUTLIER}")
        figure_phase(trimmed, trimmed_counts, "sensitivity", f"consistent_vs_inconsistent_phase_without_{OUTLIER}")
    figure_cross_share_controls(data, counts)
    figure_smacof_vs_classical_split(data, counts)
    figure_reconstruction_quality(data, counts)
    figure_convergence(data)
    figure_bins(data, "sequence_separation", "error_by_sequence_separation", "Sequence separation")
    figure_bins(data, "true_distance_b10", "error_by_true_distance", "Final-structure distance (Å)")
    (ROOT / "RESULTS.md").write_text(results_markdown(data, counts))
    (ROOT / "sanity_checks.md").write_text(sanity_markdown(data, counts))
    print(f"wrote tables, text and figures under {ROOT}")


if __name__ == "__main__":
    main()

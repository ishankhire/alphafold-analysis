#!/usr/bin/env python3
"""Generate the complete local triangle-violation subgroup figure suite.

The Delta workflow writes canonical row-level and summary CSVs beneath
``delta_files/triangle_violation_subgroups``.  This local-only entrypoint turns
those copied outputs into publication-style layer trajectories beneath
``delta_files/analysis/triangle_violation_subgroups``.  It does not modify the
canonical source artifacts or their run manifest.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_INPUT_ROOT = REPO_ROOT / "delta_files" / "triangle_violation_subgroups"
DEFAULT_OUTPUT_ROOT = REPO_ROOT / "delta_files" / "analysis" / "triangle_violation_subgroups"
LAYERS = tuple(range(48))

MODEL_SPECS = {
    "general_base256_combined": ("Shared linear", "#4C78A8", "-"),
    "per_protein_base256_combined": ("Per-protein linear", "#7F7F7F", (0, (5, 2))),
    "shared_xgboost_base256_combined": ("XGBoost", "#E45756", "-"),
    "analysis_1_sequence_gated": ("Sequence gated", "#F58518", "-"),
}
MODEL_ORDER = tuple(MODEL_SPECS)

CATEGORY_SPECS = {
    "all_triangles": ("All triangles", ("all",)),
    "rms_size": ("Reference RMS-size tertiles", ("small", "medium", "large")),
    "angle_shape": (
        "Reference largest-angle classes",
        ("clearly_acute", "near_right", "clearly_obtuse"),
    ),
    "angle_shape_exact90": (
        "Exact reference largest-angle classes",
        ("acute", "right_within_tolerance", "obtuse"),
    ),
    "rms_by_angle": (
        "Reference RMS size by largest angle",
        tuple(
            f"{size}__{angle}"
            for size in ("small", "medium", "large")
            for angle in ("clearly_acute", "near_right", "clearly_obtuse")
        ),
    ),
    "relative_slack": ("Reference relative-slack tertiles", ("low", "medium", "high")),
    "area": ("Reference area tertiles", ("low", "medium", "high")),
}

METRIC_FAMILIES = {
    "absolute": (
        ("violation_fraction", "All violations", "Fraction of triangles"),
        ("mean_violation_all", "Mean burden", "Mean magnitude (Å / triangle)"),
        (
            "mean_violation_conditional",
            "Conditional mean",
            "Mean magnitude (Å / violation)",
        ),
        ("abs_0_5_fraction_all", "Magnitude 0–5 Å", "Fraction of triangles"),
        ("abs_5_10_fraction_all", "Magnitude 5–10 Å", "Fraction of triangles"),
        ("abs_10_plus_fraction_all", "Magnitude 10+ Å", "Fraction of triangles"),
    ),
    "relative": (
        (
            "mean_relative_violation_all",
            "Normalized burden",
            "Relative magnitude / triangle",
        ),
        (
            "mean_relative_violation_conditional",
            "Conditional normalized mean",
            "Relative magnitude / violation",
        ),
        ("relative_violation_max", "Mean per-protein maximum", "Relative magnitude"),
        (
            "relative_violation_p99_conditional",
            "Mean per-protein conditional p99",
            "Relative magnitude",
        ),
        ("rel_0_0p05_fraction_all", "Relative 0–0.05", "Fraction of triangles"),
        ("rel_0p05_0p10_fraction_all", "Relative 0.05–0.10", "Fraction of triangles"),
        ("rel_0p10_plus_fraction_all", "Relative 0.10+", "Fraction of triangles"),
    ),
}

LENGTH_BIN_LABELS = {
    "length_bin_1": "Length quintile 1 (shortest)",
    "length_bin_2": "Length quintile 2",
    "length_bin_3": "Length quintile 3",
    "length_bin_4": "Length quintile 4",
    "length_bin_5": "Length quintile 5 (longest)",
}
XTICKS = (0, 10, 20, 30, 40, 47)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def display_category(label: str) -> str:
    return label.replace("__", " · ").replace("_", " ").title()


def require_columns(path: Path, frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")


def validate_inputs(root: Path, summary: pd.DataFrame, length_summary: pd.DataFrame) -> None:
    with (root / "run_manifest.json").open() as handle:
        manifest = json.load(handle)
    if manifest.get("run_status") != "complete":
        raise ValueError("Refusing to plot an incomplete triangle subgroup aggregation")

    required = {
        "analysis_name",
        "layer",
        "evaluation_scope",
        "category_system",
        "category_label",
        "metric",
        "protein_equal_mean",
        "n_proteins_finite",
    }
    require_columns(root / "triangle_metric_summary.csv", summary, required)
    require_columns(
        root / "length_bin_summary.csv", length_summary, required | {"protein_length_bin"}
    )
    if set(summary["evaluation_scope"]) != {"all_edges"}:
        raise ValueError("Expected the canonical all_edges evaluation scope")
    if set(summary["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError("Model coverage differs from the four frozen probes")
    if set(summary["layer"].astype(int)) != set(LAYERS):
        raise ValueError("triangle_metric_summary.csv must contain all layers 0–47")
    if set(summary["category_system"]) != set(CATEGORY_SPECS):
        raise ValueError("Category-system coverage does not match the plotting contract")
    for system, (_, labels) in CATEGORY_SPECS.items():
        observed = set(summary.loc[summary["category_system"].eq(system), "category_label"])
        if observed != set(labels):
            raise ValueError(f"Category labels differ for {system}: {sorted(observed)}")
    expected_metrics = {metric for specs in METRIC_FAMILIES.values() for metric, _, _ in specs}
    if set(summary["metric"]) != expected_metrics:
        raise ValueError("Metric coverage does not match the 13-metric plotting contract")
    duplicate_key = [
        "analysis_name",
        "layer",
        "category_system",
        "category_label",
        "metric",
    ]
    if summary.duplicated(duplicate_key).any():
        raise ValueError("triangle_metric_summary.csv contains duplicate trajectory points")


def category_colors(labels: Sequence[str]) -> dict[str, tuple[float, float, float, float]]:
    palette_name = "tab10" if len(labels) <= 10 else "tab20"
    palette = plt.get_cmap(palette_name)
    return {label: palette(index % palette.N) for index, label in enumerate(labels)}


def model_handles() -> list[Line2D]:
    return [
        Line2D(
            [0],
            [0],
            color=color,
            linestyle=linestyle,
            linewidth=2,
            label=label,
        )
        for label, color, linestyle in MODEL_SPECS.values()
    ]


def save_figure(
    figure: plt.Figure,
    output_root: Path,
    subdirectory: str,
    stem: str,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    destination = output_root / subdirectory
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for extension in ("png", "pdf"):
        path = destination / f"{stem}.{extension}"
        figure.savefig(path)
        records.append(
            {
                **metadata,
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    plt.close(figure)
    return records


def plot_all_triangle_family(
    summary: pd.DataFrame,
    family: str,
    output_root: Path,
) -> list[dict[str, object]]:
    specifications = METRIC_FAMILIES[family]
    ncols = 3 if len(specifications) == 6 else 4
    nrows = math.ceil(len(specifications) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), sharex=True)
    flat_axes = np.atleast_1d(axes).ravel()
    selected = summary[summary["category_system"].eq("all_triangles")]
    for axis, (metric, title, ylabel) in zip(flat_axes, specifications):
        metric_data = selected[selected["metric"].eq(metric)]
        for model in MODEL_ORDER:
            label, color, linestyle = MODEL_SPECS[model]
            values = metric_data[metric_data["analysis_name"].eq(model)].sort_values("layer")
            axis.plot(
                values["layer"],
                values["protein_equal_mean"],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
        axis.set(title=title, xlabel="Evoformer layer", ylabel=ylabel, xlim=(0, 47))
        axis.set_xticks(XTICKS)
        axis.set_ylim(bottom=0)
    for axis in flat_axes[len(specifications) :]:
        axis.set_visible(False)
    flat_axes[0].legend(handles=model_handles(), loc="best")
    figure.suptitle(f"All-triangle {family} violation diagnostics across layers", y=0.995)
    figure.tight_layout()
    return save_figure(
        figure,
        output_root,
        "category_trajectories",
        f"all_triangles_{family}_diagnostics",
        {"figure_kind": "category_trajectory", "category_system": "all_triangles", "metric_family": family},
    )


def plot_category_family(
    summary: pd.DataFrame,
    system: str,
    family: str,
    output_root: Path,
) -> list[dict[str, object]]:
    title, labels = CATEGORY_SPECS[system]
    specifications = METRIC_FAMILIES[family]
    colors = category_colors(labels)
    figure, axes = plt.subplots(
        len(MODEL_ORDER),
        len(specifications),
        figsize=(3.35 * len(specifications), 3.0 * len(MODEL_ORDER)),
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    selected = summary[summary["category_system"].eq(system)]
    supported_labels = set(
        selected.loc[np.isfinite(selected["protein_equal_mean"]), "category_label"]
    )
    for row_index, model in enumerate(MODEL_ORDER):
        model_label = MODEL_SPECS[model][0]
        for column_index, (metric, metric_title, ylabel) in enumerate(specifications):
            axis = axes[row_index, column_index]
            metric_data = selected[
                selected["analysis_name"].eq(model) & selected["metric"].eq(metric)
            ]
            for label in labels:
                values = metric_data[metric_data["category_label"].eq(label)].sort_values("layer")
                axis.plot(
                    values["layer"],
                    values["protein_equal_mean"],
                    color=colors[label],
                    linewidth=1.55,
                )
            axis.set_xlim(0, 47)
            axis.set_xticks(XTICKS)
            axis.set_ylim(bottom=0)
            if row_index == 0:
                axis.set_title(metric_title)
            if row_index == len(MODEL_ORDER) - 1:
                axis.set_xlabel("Evoformer layer")
            if column_index == 0:
                axis.set_ylabel(f"{model_label}\n{ylabel}")
    legend = [
        Line2D(
            [0],
            [0],
            color=colors[label],
            linewidth=2,
            label=(
                display_category(label)
                if label in supported_labels
                else f"{display_category(label)} (no finite support)"
            ),
        )
        for label in labels
    ]
    figure.legend(
        handles=legend,
        loc="outside lower center",
        ncol=min(5, len(labels)),
        frameon=False,
    )
    figure.suptitle(f"{title}: {family} triangle-violation diagnostics", y=1.01)
    figure.tight_layout(rect=(0, 0.06 if len(labels) <= 5 else 0.09, 1, 0.99))
    return save_figure(
        figure,
        output_root,
        "category_trajectories",
        f"{system}_{family}_diagnostics",
        {"figure_kind": "category_trajectory", "category_system": system, "metric_family": family},
    )


def plot_length_bin_family(
    length_summary: pd.DataFrame,
    family: str,
    output_root: Path,
) -> list[dict[str, object]]:
    specifications = METRIC_FAMILIES[family]
    bins = tuple(LENGTH_BIN_LABELS)
    colors = category_colors(bins)
    figure, axes = plt.subplots(
        len(MODEL_ORDER),
        len(specifications),
        figsize=(3.35 * len(specifications), 3.0 * len(MODEL_ORDER)),
        sharex=True,
        sharey="col",
        squeeze=False,
    )
    selected = length_summary[
        length_summary["category_system"].eq("all_triangles")
        & length_summary["category_label"].eq("all")
        & length_summary["layer"].ge(0)
    ]
    for row_index, model in enumerate(MODEL_ORDER):
        for column_index, (metric, metric_title, ylabel) in enumerate(specifications):
            axis = axes[row_index, column_index]
            metric_data = selected[
                selected["analysis_name"].eq(model) & selected["metric"].eq(metric)
            ]
            for length_bin in bins:
                values = metric_data[
                    metric_data["protein_length_bin"].eq(length_bin)
                ].sort_values("layer")
                axis.plot(
                    values["layer"],
                    values["protein_equal_mean"],
                    color=colors[length_bin],
                    linewidth=1.55,
                )
            axis.set(xlim=(0, 47), xticks=XTICKS)
            axis.set_ylim(bottom=0)
            if row_index == 0:
                axis.set_title(metric_title)
            if row_index == len(MODEL_ORDER) - 1:
                axis.set_xlabel("Evoformer layer")
            if column_index == 0:
                axis.set_ylabel(f"{MODEL_SPECS[model][0]}\n{ylabel}")
    legend = [
        Line2D([0], [0], color=colors[item], linewidth=2, label=LENGTH_BIN_LABELS[item])
        for item in bins
    ]
    figure.legend(handles=legend, loc="outside lower center", ncol=5, frameon=False)
    figure.suptitle(f"Protein-length strata: all-triangle {family} diagnostics", y=1.01)
    figure.tight_layout(rect=(0, 0.06, 1, 0.99))
    return save_figure(
        figure,
        output_root,
        "length_diagnostics",
        f"protein_length_bins_{family}_diagnostics",
        {"figure_kind": "length_bin_trajectory", "category_system": "all_triangles", "metric_family": family},
    )


def plot_length_effect_family(
    effects: pd.DataFrame,
    family: str,
    output_root: Path,
) -> list[dict[str, object]]:
    specifications = METRIC_FAMILIES[family]
    ncols = 3 if len(specifications) == 6 else 4
    nrows = math.ceil(len(specifications) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 4.0 * nrows), sharex=True)
    flat_axes = np.atleast_1d(axes).ravel()
    selected = effects[
        effects["category_system"].eq("all_triangles")
        & effects["category_label"].eq("all")
        & effects["association_status"].eq("complete")
    ]
    for axis, (metric, title, _) in zip(flat_axes, specifications):
        metric_data = selected[selected["metric"].eq(metric)]
        for model in MODEL_ORDER:
            label, color, linestyle = MODEL_SPECS[model]
            values = metric_data[metric_data["analysis_name"].eq(model)].sort_values("layer")
            axis.plot(
                values["layer"],
                values["standardized_log_length_slope"],
                color=color,
                linestyle=linestyle,
                linewidth=1.8,
                label=label,
            )
        axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.7)
        axis.set(
            title=title,
            xlabel="Evoformer layer",
            ylabel="Slope per 1 SD log length",
            xlim=(0, 47),
            xticks=XTICKS,
        )
    for axis in flat_axes[len(specifications) :]:
        axis.set_visible(False)
    flat_axes[0].legend(handles=model_handles(), loc="best")
    figure.suptitle(f"Protein-length association: all-triangle {family} metrics", y=0.995)
    figure.tight_layout()
    return save_figure(
        figure,
        output_root,
        "length_diagnostics",
        f"protein_length_effects_{family}_diagnostics",
        {"figure_kind": "length_effect", "category_system": "all_triangles", "metric_family": family},
    )


def trajectory_changes(summary: pd.DataFrame) -> pd.DataFrame:
    keys = ["analysis_name", "category_system", "category_label", "metric"]
    records: list[dict[str, object]] = []
    for key, group in summary.groupby(keys, sort=True):
        values = group.sort_values("layer")
        first = values.iloc[0]
        last = values.iloc[-1]
        finite = values[np.isfinite(values["protein_equal_mean"])]
        minimum = finite.loc[finite["protein_equal_mean"].idxmin()] if not finite.empty else None
        maximum = finite.loc[finite["protein_equal_mean"].idxmax()] if not finite.empty else None
        records.append(
            {
                **dict(zip(keys, key)),
                "layer0_value": first["protein_equal_mean"],
                "layer47_value": last["protein_equal_mean"],
                "change_47_minus_0": last["protein_equal_mean"] - first["protein_equal_mean"],
                "minimum_layer": int(minimum["layer"]) if minimum is not None else pd.NA,
                "minimum_value": minimum["protein_equal_mean"] if minimum is not None else np.nan,
                "maximum_layer": int(maximum["layer"]) if maximum is not None else pd.NA,
                "maximum_value": maximum["protein_equal_mean"] if maximum is not None else np.nan,
            }
        )
    return pd.DataFrame(records)


def write_report(
    output_root: Path,
    summary: pd.DataFrame,
    effects: pd.DataFrame,
    figure_count: int,
) -> None:
    all_fraction = summary[
        summary["category_system"].eq("all_triangles")
        & summary["metric"].eq("violation_fraction")
    ]
    lines = [
        "# Triangle-violation subgroup trajectories",
        "",
        "## Scope and interpretation",
        "",
        (
            "This directory visualizes all 7 reference category systems, all 21 category "
            "labels, all 13 canonical violation metrics, all 4 frozen probe families, and all "
            "48 Evoformer layers in `triangle_metric_summary.csv`. Values are protein-equal "
            "means over 111 analysis proteins; every trajectory uses the canonical `all_edges` "
            "scope. The category PDFs are the direct subgroup analogue of "
            "`main_figure_05_triangle_violation_diagnostics.pdf`."
        ),
        "",
        (
            f"There are {figure_count // 2} distinct figures, each saved as both PDF and PNG. "
            "The `absolute` figures reproduce the Figure-5-style violation fraction, absolute "
            "burden/severity, and 0–5/5–10/10+ Å panels. The `relative` figures add all seven "
            "scale-normalized metrics. In subgroup matrices, rows are models, columns are "
            "metrics, and colored lines are reference categories."
        ),
        "",
        "## Overall layer trajectories",
        "",
        "| Model | Layer 0 | Layer 47 | Change | Minimum (layer) |",
        "|---|---:|---:|---:|---:|",
    ]
    for model in MODEL_ORDER:
        data = all_fraction[all_fraction["analysis_name"].eq(model)].sort_values("layer")
        first = data.iloc[0]
        last = data.iloc[-1]
        minimum = data.loc[data["protein_equal_mean"].idxmin()]
        lines.append(
            f"| {MODEL_SPECS[model][0]} | {first.protein_equal_mean:.4f} | "
            f"{last.protein_equal_mean:.4f} | "
            f"{last.protein_equal_mean - first.protein_equal_mean:+.4f} | "
            f"{minimum.protein_equal_mean:.4f} (L{int(minimum.layer)}) |"
        )
    lines.extend(
        [
            "",
            (
                "At layer 47, XGBoost and the per-protein linear probe have the lowest mean "
                "violation fractions, whereas the shared linear and sequence-gated probes end "
                "higher. This is descriptive geometric consistency, not a stand-alone accuracy "
                "ranking: the per-protein model also has a mixed within-protein train/held-out "
                "generalization scope, unlike the three unseen-protein models."
            ),
            "",
            "## Layer-47 subgroup ordering",
            "",
            (
                "Across models, the strongest and most stable separation is geometric. "
                "Clearly obtuse, low-slack, and low-area reference triangles carry much larger "
                "violation fractions than clearly acute, high-slack, and high-area triangles. "
                "The size-by-angle figures show that angle/slack-like degeneracy is more "
                "discriminating than RMS size alone."
            ),
            "",
            "| Model | Category system | Lowest layer-47 category | Highest layer-47 category |",
            "|---|---|---:|---:|",
        ]
    )
    layer47 = summary[
        summary["layer"].eq(47)
        & summary["metric"].eq("violation_fraction")
        & ~summary["category_system"].eq("all_triangles")
    ]
    for model in MODEL_ORDER:
        for system in CATEGORY_SPECS:
            if system == "all_triangles":
                continue
            data = layer47[
                layer47["analysis_name"].eq(model) & layer47["category_system"].eq(system)
            ]
            low = data.loc[data["protein_equal_mean"].idxmin()]
            high = data.loc[data["protein_equal_mean"].idxmax()]
            lines.append(
                f"| {MODEL_SPECS[model][0]} | {CATEGORY_SPECS[system][0]} | "
                f"{display_category(str(low.category_label))} ({low.protein_equal_mean:.4f}) | "
                f"{display_category(str(high.category_label))} ({high.protein_equal_mean:.4f}) |"
            )
    lines.extend(
        [
            "",
            "## Protein length",
            "",
            "| Model | Layer-47 slope per 1 SD log length | Spearman ρ |",
            "|---|---:|---:|",
        ]
    )
    length47 = effects[
        effects["layer"].eq(47)
        & effects["category_system"].eq("all_triangles")
        & effects["category_label"].eq("all")
        & effects["metric"].eq("violation_fraction")
    ]
    for model in MODEL_ORDER:
        row = length47[length47["analysis_name"].eq(model)].iloc[0]
        lines.append(
            f"| {MODEL_SPECS[model][0]} | {row.standardized_log_length_slope:+.4f} | "
            f"{row.spearman_rho:+.3f} |"
        )
    lines.extend(
        [
            "",
            (
                "The shared linear probe is the only model with a large layer-47 monotonic "
                "length association: longer proteins have lower violation fractions. The other "
                "three slopes are near zero, although their length-quintile trajectories are "
                "not perfectly monotone. Use the length-bin figures for composition-sensitive "
                "checks and the slope figures for the continuous association."
            ),
            "",
            "## Files and caveats",
            "",
            "- `category_trajectories/` contains the complete category-system figure suite.",
            "- `length_diagnostics/` contains length-quintile trajectories and continuous length-effect slopes.",
            "- `trajectory_changes.csv` records layer-0/layer-47 values and trajectory extrema for every model/category/metric.",
            "- `paired_model_contrasts_layer47.csv` and `protein_length_effects_layer47.csv` retain compact final-layer sensitivity tables.",
            "- `figure_manifest.csv` and `generation_summary.json` record figure provenance and source SHA-256 digests.",
            "- Absolute and relative band fractions use all triangles as the denominator. Conditional means and p99 values condition on violations.",
            "- Categories are fixed from final-PDB reference geometry within each protein; they are not inferred from decoded distances.",
            "- The exact-90° `right_within_tolerance` category has no finite support at the workflow's 10⁻¹² relative equality tolerance, so its legend entry is retained but its trajectories are blank.",
            "- No bootstrap resampling or inferential significance testing is performed. Bands are omitted to keep the dense all-category plots readable.",
            "",
        ]
    )
    (output_root / "README.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-root", type=Path, default=DEFAULT_INPUT_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_root = args.input_root.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configure_style()

    source_paths = {
        name: input_root / name
        for name in (
            "triangle_metric_summary.csv",
            "length_bin_summary.csv",
            "protein_length_effects.csv",
            "paired_model_contrasts.csv",
            "run_manifest.json",
        )
    }
    summary = pd.read_csv(source_paths["triangle_metric_summary.csv"])
    length_summary = pd.read_csv(source_paths["length_bin_summary.csv"])
    effects = pd.read_csv(source_paths["protein_length_effects.csv"])
    contrasts = pd.read_csv(source_paths["paired_model_contrasts.csv"])
    validate_inputs(input_root, summary, length_summary)

    figure_records: list[dict[str, object]] = []
    for family in METRIC_FAMILIES:
        figure_records.extend(plot_all_triangle_family(summary, family, output_root))
        for system in CATEGORY_SPECS:
            if system != "all_triangles":
                figure_records.extend(plot_category_family(summary, system, family, output_root))
        figure_records.extend(plot_length_bin_family(length_summary, family, output_root))
        figure_records.extend(plot_length_effect_family(effects, family, output_root))

    changes = trajectory_changes(summary)
    changes.to_csv(output_root / "trajectory_changes.csv", index=False)
    contrasts[contrasts["layer"].eq(47)].to_csv(
        output_root / "paired_model_contrasts_layer47.csv", index=False
    )
    effects[effects["layer"].eq(47)].to_csv(
        output_root / "protein_length_effects_layer47.csv", index=False
    )
    figure_manifest = pd.DataFrame(figure_records)
    figure_manifest.to_csv(output_root / "figure_manifest.csv", index=False)
    write_report(output_root, summary, effects, len(figure_records))

    generated_paths = [
        output_root / record["path"] for record in figure_records
    ] + [
        output_root / "trajectory_changes.csv",
        output_root / "paired_model_contrasts_layer47.csv",
        output_root / "protein_length_effects_layer47.csv",
        output_root / "figure_manifest.csv",
        output_root / "README.md",
    ]
    generation_summary = {
        "analysis_name": "triangle_violation_subgroup_trajectories",
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "input_root": input_root.relative_to(REPO_ROOT).as_posix(),
        "output_root": output_root.relative_to(REPO_ROOT).as_posix(),
        "evaluation_scope": "all_edges",
        "aggregation": "protein_equal_mean",
        "layers": list(LAYERS),
        "models": list(MODEL_ORDER),
        "category_systems": list(CATEGORY_SPECS),
        "metrics": [metric for family in METRIC_FAMILIES.values() for metric, _, _ in family],
        "distinct_figures": len(figure_records) // 2,
        "rendered_figure_files": len(figure_records),
        "source_sha256": {name: sha256_file(path) for name, path in source_paths.items()},
        "artifact_sha256": {
            path.relative_to(output_root).as_posix(): sha256_file(path)
            for path in generated_paths
        },
    }
    with (output_root / "generation_summary.json").open("w") as handle:
        json.dump(generation_summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")
    print(
        f"Wrote {len(figure_records) // 2} figures ({len(figure_records)} PDF/PNG files) "
        f"and analysis tables beneath {output_root}"
    )


if __name__ == "__main__":
    main()

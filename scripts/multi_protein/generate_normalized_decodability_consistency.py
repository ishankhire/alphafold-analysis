#!/usr/bin/env python3
"""Generate focused normalized-R2 decodability-consistency phase grids.

The local-only workflow combines the all-pairs, protein-equal R2 convention
from main Figure 8 with the two metrics and five category systems used by the
focused triangle-violation figures. It reads copied canonical artifacts and
writes only derived tables, documentation, provenance, and figures.
"""

from __future__ import annotations

import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence

import generate_paper_figure_catalog as catalog

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
INPUT_SUMMARY = (
    REPO_ROOT / "delta_files" / "triangle_violation_subgroups" / "triangle_metric_summary.csv"
)
OUTPUT_ROOT = (
    REPO_ROOT
    / "delta_files"
    / "analysis"
    / "triangle_violation_subgroups"
    / "focused_decodability_consistency"
)
LAYERS = tuple(range(48))
SELECTED_LAYERS = (0, 10, 20, 30, 40, 47)
EXPECTED_PROTEINS = 111

MODEL_SPECS = {
    "general_base256_combined": {
        "label": "Shared linear",
        "catalog_model": "Shared linear",
    },
    "per_protein_base256_combined": {
        "label": "Per-protein linear",
        "catalog_model": "Per-protein linear",
    },
    "shared_xgboost_base256_combined": {
        "label": "XGBoost",
        "catalog_model": "XGBoost",
    },
    "analysis_1_sequence_gated": {
        "label": "Sequence gated",
        "catalog_model": "Sequence gated",
    },
}
MODEL_ORDER = tuple(MODEL_SPECS)

CATEGORY_SPECS = {
    "all_triangles": {
        "figure_number": 1,
        "title": "All triangles",
        "labels": ("all",),
    },
    "rms_size": {
        "figure_number": 2,
        "title": "Reference RMS-size tertiles",
        "labels": ("small", "medium", "large"),
    },
    "area": {
        "figure_number": 3,
        "title": "Reference area tertiles",
        "labels": ("low", "medium", "high"),
    },
    "relative_slack": {
        "figure_number": 4,
        "title": "Reference relative-slack tertiles",
        "labels": ("low", "medium", "high"),
    },
    "angle_shape": {
        "figure_number": 5,
        "title": "Reference largest-angle classes",
        "labels": ("clearly_acute", "near_right", "clearly_obtuse"),
    },
}

METRIC_SPECS = {
    "violation_fraction": {
        "title": "Triangle-inequality violation fraction",
        "ylabel": "Fraction of triangles",
    },
    "mean_relative_violation_all": {
        "title": "Normalized mean triangle violation",
        "ylabel": "Mean relative violation per triangle",
    },
}

POINT_COLUMNS = (
    "analysis_name",
    "model_family",
    "model",
    "layer",
    "raw_mean_per_protein_r2",
    "model_r2_min",
    "model_r2_max",
    "normalized_r2",
    "r2_metric",
    "r2_aggregation",
    "r2_protein_split",
    "r2_evaluation_scope",
    "r2_n_proteins",
    "category_system",
    "category_label",
    "consistency_metric",
    "consistency_value",
    "consistency_aggregation",
    "consistency_evaluation_scope",
    "consistency_n_proteins",
)


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_digest(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(path.resolve() for path in paths):
        digest.update(relative(path).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def require_columns(path: Path, frame: pd.DataFrame, columns: Iterable[str]) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{relative(path)} is missing columns: {missing}")


def display_category(label: str) -> str:
    if label == "all":
        return "All triangles"
    return label.replace("__", " · ").replace("_", " ").title()


def category_colors(labels: Sequence[str]) -> dict[str, object]:
    palette = plt.get_cmap("tab10")
    return {label: palette(index) for index, label in enumerate(labels)}


def normalize_r2_by_model(frame: pd.DataFrame) -> pd.DataFrame:
    """Min-max normalize the 48 aggregated R2 values within each model."""
    required = {"analysis_name", "layer", "raw_mean_per_protein_r2"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"R2 summary is missing columns: {sorted(missing)}")
    duplicate_key = ["analysis_name", "layer"]
    if frame.duplicated(duplicate_key).any():
        raise ValueError("R2 summary contains duplicate model-layer rows")

    normalized_groups: list[pd.DataFrame] = []
    for analysis_name, group in frame.groupby("analysis_name", sort=False):
        values = pd.to_numeric(group["raw_mean_per_protein_r2"], errors="coerce")
        if not np.isfinite(values.to_numpy(dtype=float)).all():
            raise ValueError(f"{analysis_name} contains nonfinite aggregated R2 values")
        minimum = float(values.min())
        maximum = float(values.max())
        span = maximum - minimum
        if not span > 0.0:
            raise ValueError(f"{analysis_name} has zero aggregated R2 span")
        result = group.copy()
        result["model_r2_min"] = minimum
        result["model_r2_max"] = maximum
        result["normalized_r2"] = (values - minimum) / span
        normalized_groups.append(result)

    result = pd.concat(normalized_groups, ignore_index=True)
    return result.sort_values(["analysis_name", "layer"], kind="stable").reset_index(drop=True)


def load_decodability() -> tuple[pd.DataFrame, list[Path]]:
    """Load and aggregate Figure-8-style all-pairs R2 for the four replayed probes."""
    rows: list[dict[str, object]] = []
    paths: list[Path] = []
    common_test_ids: set[str] | None = None

    for analysis_name, spec in MODEL_SPECS.items():
        model = str(spec["catalog_model"])
        path = Path(catalog.MODEL_SPECS[model]["root"]) / "per_protein.csv"
        frame = pd.read_csv(path, dtype={"protein_id": str})
        paths.append(path)
        require_columns(
            path,
            frame,
            {
                "protein_id",
                "layer",
                "protein_split",
                "evaluation_scope",
                "r2",
            },
        )
        selected = frame[
            frame["protein_split"].eq("test")
            & frame["evaluation_scope"].eq("all_pairs")
        ][["protein_id", "layer", "r2"]].copy()
        selected["protein_id"] = selected["protein_id"].str.zfill(9)
        if selected.duplicated(["protein_id", "layer"]).any():
            raise ValueError(f"{model} has duplicate all-pairs test protein-layer R2 rows")
        if set(selected["layer"].astype(int)) != set(LAYERS):
            raise ValueError(f"{model} does not contain all layers 0-47")
        test_ids = set(selected["protein_id"])
        if len(test_ids) != EXPECTED_PROTEINS:
            raise ValueError(
                f"{model} has {len(test_ids)} all-pairs test proteins; expected {EXPECTED_PROTEINS}"
            )
        if common_test_ids is None:
            common_test_ids = test_ids
        elif test_ids != common_test_ids:
            raise ValueError(f"{model} does not use the common locked test-protein set")

        for layer, group in selected.groupby("layer", sort=True):
            values = pd.to_numeric(group["r2"], errors="coerce")
            if len(group) != EXPECTED_PROTEINS or not np.isfinite(values).all():
                raise ValueError(f"{model} layer {layer} lacks 111 finite all-pairs R2 values")
            rows.append(
                {
                    "analysis_name": analysis_name,
                    "model": spec["label"],
                    "layer": int(layer),
                    "raw_mean_per_protein_r2": float(values.mean()),
                    "r2_metric": "mean_per_protein_r2",
                    "r2_aggregation": "protein_equal_mean",
                    "r2_protein_split": "test",
                    "r2_evaluation_scope": "all_pairs",
                    "r2_n_proteins": int(len(group)),
                }
            )

    result = normalize_r2_by_model(pd.DataFrame(rows))
    expected_rows = len(MODEL_ORDER) * len(LAYERS)
    if len(result) != expected_rows:
        raise ValueError(f"Expected {expected_rows} normalized R2 rows; found {len(result)}")
    for analysis_name in MODEL_ORDER:
        values = result.loc[result["analysis_name"].eq(analysis_name), "normalized_r2"]
        if not np.isclose(values.min(), 0.0) or not np.isclose(values.max(), 1.0):
            raise ValueError(f"{analysis_name} normalized R2 does not span 0-1")
    return result, paths


def select_consistency_summary(path: Path = INPUT_SUMMARY) -> pd.DataFrame:
    summary = pd.read_csv(path)
    require_columns(
        path,
        summary,
        {
            "analysis_name",
            "model_family",
            "layer",
            "evaluation_scope",
            "category_system",
            "category_label",
            "metric",
            "protein_equal_mean",
            "n_proteins_total",
            "n_proteins_finite",
        },
    )
    selected = summary[
        summary["analysis_name"].isin(MODEL_ORDER)
        & summary["category_system"].isin(CATEGORY_SPECS)
        & summary["metric"].isin(METRIC_SPECS)
    ].copy()
    duplicate_key = [
        "analysis_name",
        "layer",
        "category_system",
        "category_label",
        "metric",
    ]
    if selected.duplicated(duplicate_key).any():
        raise ValueError("Selected consistency summary contains duplicate trajectory points")
    if set(selected["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError("Consistency summary does not contain exactly the four requested models")
    if set(selected["layer"].astype(int)) != set(LAYERS):
        raise ValueError("Consistency summary does not contain all layers 0-47")
    if set(selected["evaluation_scope"]) != {"all_edges"}:
        raise ValueError("Consistency summary must use only the canonical all_edges scope")
    if set(selected["category_system"]) != set(CATEGORY_SPECS):
        raise ValueError("Consistency summary does not contain all five focused category systems")
    if set(selected["metric"]) != set(METRIC_SPECS):
        raise ValueError("Consistency summary does not contain both focused metrics")
    for system, spec in CATEGORY_SPECS.items():
        observed = set(
            selected.loc[selected["category_system"].eq(system), "category_label"]
        )
        if observed != set(spec["labels"]):
            raise ValueError(f"Unexpected category labels for {system}: {sorted(observed)}")
    expected_rows = (
        len(MODEL_ORDER)
        * len(LAYERS)
        * len(METRIC_SPECS)
        * sum(len(spec["labels"]) for spec in CATEGORY_SPECS.values())
    )
    if len(selected) != expected_rows:
        raise ValueError(f"Expected {expected_rows} consistency rows; found {len(selected)}")
    if not np.isfinite(pd.to_numeric(selected["protein_equal_mean"], errors="coerce")).all():
        raise ValueError("Consistency summary contains nonfinite plotted values")
    for count_column in ("n_proteins_total", "n_proteins_finite"):
        counts = set(pd.to_numeric(selected[count_column], errors="raise").astype(int))
        if counts != {EXPECTED_PROTEINS}:
            raise ValueError(f"{count_column} differs from the required 111 proteins: {counts}")
    return selected


def build_points(decodability: pd.DataFrame, consistency: pd.DataFrame) -> pd.DataFrame:
    selected = consistency.rename(
        columns={
            "metric": "consistency_metric",
            "protein_equal_mean": "consistency_value",
            "evaluation_scope": "consistency_evaluation_scope",
            "n_proteins_finite": "consistency_n_proteins",
        }
    )
    selected["consistency_aggregation"] = "protein_equal_mean"
    points = selected.merge(
        decodability,
        on=["analysis_name", "layer"],
        how="left",
        validate="many_to_one",
    )
    if points["normalized_r2"].isna().any():
        raise ValueError("Some consistency rows lack a normalized R2 match")
    points = points.loc[:, POINT_COLUMNS].sort_values(
        ["category_system", "consistency_metric", "analysis_name", "category_label", "layer"],
        kind="stable",
    )
    if len(points) != 4_992:
        raise ValueError(f"Expected 4,992 plotted points; found {len(points)}")
    numeric = points[["raw_mean_per_protein_r2", "normalized_r2", "consistency_value"]]
    if not np.isfinite(numeric.to_numpy(dtype=float)).all():
        raise ValueError("Plotted point table contains nonfinite coordinates")
    return points.reset_index(drop=True)


def annotate_layers(
    axis: plt.Axes,
    data: pd.DataFrame,
    color: object,
    category_index: int,
) -> None:
    offsets = ((3, 3), (3, -8), (3, 3), (3, -8), (3, 3), (3, -8))
    category_shift = (category_index - 1) * 2
    selected = data[data["layer"].isin(SELECTED_LAYERS)]
    for selected_index, row in enumerate(selected.itertuples(index=False)):
        dx, dy = offsets[selected_index]
        horizontal_alignment = "left"
        if row.layer == 47:
            dx = -3
            horizontal_alignment = "right"
        axis.annotate(
            str(row.layer),
            (row.normalized_r2, row.consistency_value),
            xytext=(dx, dy + category_shift),
            textcoords="offset points",
            fontsize=5.2,
            fontweight="bold" if row.layer in (0, 47) else "normal",
            color=color,
            horizontalalignment=horizontal_alignment,
            clip_on=False,
        )


def annotate_overlay_layers(
    axis: plt.Axes,
    data: pd.DataFrame,
    color: object,
    model_index: int,
) -> None:
    """Label decade layers while separating nearby overlaid model labels."""
    offsets = ((4, 3), (4, -7), (4, 3), (4, -7), (4, 3), (-4, -7))
    model_shift = (-5, 5, -5, 5)[model_index]
    selected = data[data["layer"].isin(SELECTED_LAYERS)]
    for selected_index, row in enumerate(selected.itertuples(index=False)):
        dx, dy = offsets[selected_index]
        horizontal_alignment = "right" if row.layer == 47 else "left"
        axis.annotate(
            str(row.layer),
            (row.normalized_r2, row.consistency_value),
            xytext=(dx, dy + model_shift),
            textcoords="offset points",
            fontsize=6.2,
            fontweight="semibold" if row.layer in (0, 47) else "normal",
            color=color,
            horizontalalignment=horizontal_alignment,
            bbox={
                "facecolor": "white",
                "edgecolor": "none",
                "alpha": 0.68,
                "pad": 0.15,
            },
            clip_on=False,
            zorder=6,
        )


def plot_all_triangles_overlay(points: pd.DataFrame) -> plt.Figure:
    """Plot both all-triangle metrics with the four probe trajectories overlaid."""
    figure, axes = plt.subplots(
        1,
        len(METRIC_SPECS),
        figsize=(13.2, 5.2),
        sharex=True,
        squeeze=False,
    )
    axes = axes[0]
    system_points = points[
        points["category_system"].eq("all_triangles")
        & points["category_label"].eq("all")
    ]

    for axis, (metric, metric_spec) in zip(axes, METRIC_SPECS.items()):
        metric_points = system_points[
            system_points["consistency_metric"].eq(metric)
        ]
        for model_index, analysis_name in enumerate(MODEL_ORDER):
            model_spec = MODEL_SPECS[analysis_name]
            catalog_spec = catalog.MODEL_SPECS[str(model_spec["catalog_model"])]
            data = metric_points[
                metric_points["analysis_name"].eq(analysis_name)
            ].sort_values("layer")
            color = catalog_spec["color"]
            axis.plot(
                data["normalized_r2"],
                data["consistency_value"],
                color=color,
                linestyle=catalog_spec["linestyle"],
                linewidth=2.05,
                alpha=0.96,
                zorder=2,
            )
            axis.scatter(
                data["normalized_r2"],
                data["consistency_value"],
                color=color,
                s=12,
                edgecolors="white",
                linewidths=0.25,
                alpha=0.9,
                zorder=3,
            )
            milestones = data[data["layer"].isin(SELECTED_LAYERS[1:-1])]
            axis.scatter(
                milestones["normalized_r2"],
                milestones["consistency_value"],
                color=color,
                s=34,
                edgecolors="white",
                linewidths=0.75,
                alpha=1.0,
                zorder=4,
            )
            start = data[data["layer"].eq(0)]
            end = data[data["layer"].eq(47)]
            axis.scatter(
                start["normalized_r2"],
                start["consistency_value"],
                color=color,
                marker="o",
                s=58,
                edgecolors="white",
                linewidths=0.9,
                clip_on=False,
                zorder=5,
            )
            axis.scatter(
                end["normalized_r2"],
                end["consistency_value"],
                color=color,
                marker="*",
                s=120,
                edgecolors="white",
                linewidths=0.7,
                clip_on=False,
                zorder=6,
            )
            annotate_overlay_layers(axis, data, color, model_index)

        axis.set_title(str(metric_spec["title"]), pad=10, fontweight="semibold")
        axis.set_xlabel("Normalized mean per-protein all-pairs $R^2$")
        axis.set_ylabel(str(metric_spec["ylabel"]))
        axis.set_xlim(0.0, 1.0)
        axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
        axis.set_ylim(bottom=0.0)
        axis.margins(y=0.08)

    model_handles = []
    for analysis_name in MODEL_ORDER:
        model_spec = MODEL_SPECS[analysis_name]
        catalog_spec = catalog.MODEL_SPECS[str(model_spec["catalog_model"])]
        model_handles.append(
            Line2D(
                [0],
                [0],
                color=catalog_spec["color"],
                linestyle=catalog_spec["linestyle"],
                linewidth=2.4,
                label=str(model_spec["label"]),
            )
        )
    endpoint_handles = [
        Line2D(
            [0],
            [0],
            color="#4A4A4A",
            marker="o",
            linestyle="none",
            markersize=6,
            label="Layer 0",
        ),
        Line2D(
            [0],
            [0],
            color="#4A4A4A",
            marker="*",
            linestyle="none",
            markersize=9,
            label="Layer 47",
        ),
    ]
    figure.legend(
        handles=model_handles + endpoint_handles,
        loc="outside lower center",
        ncol=len(model_handles) + len(endpoint_handles),
        frameon=False,
        handlelength=2.8,
        columnspacing=1.7,
    )
    figure.suptitle(
        "All triangles: normalized decodability versus triangle consistency",
        y=0.99,
        fontweight="semibold",
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.94), w_pad=3.0)
    return figure


def plot_grid(points: pd.DataFrame, system: str) -> plt.Figure:
    if system == "all_triangles":
        return plot_all_triangles_overlay(points)

    category_spec = CATEGORY_SPECS[system]
    labels = tuple(category_spec["labels"])
    colors = category_colors(labels)
    figure, axes = plt.subplots(
        len(METRIC_SPECS),
        len(MODEL_ORDER),
        figsize=(16.4, 8.0),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    system_points = points[points["category_system"].eq(system)]
    for row_index, (metric, metric_spec) in enumerate(METRIC_SPECS.items()):
        metric_points = system_points[system_points["consistency_metric"].eq(metric)]
        for column_index, analysis_name in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            model_points = metric_points[metric_points["analysis_name"].eq(analysis_name)]
            for category_index, label in enumerate(labels):
                data = model_points[
                    model_points["category_label"].eq(label)
                ].sort_values("layer")
                color = colors[label]
                axis.plot(
                    data["normalized_r2"],
                    data["consistency_value"],
                    color=color,
                    linewidth=1.55,
                    alpha=0.9,
                    zorder=2,
                )
                axis.scatter(
                    data["normalized_r2"],
                    data["consistency_value"],
                    color=color,
                    s=8,
                    edgecolors="none",
                    alpha=0.85,
                    zorder=3,
                )
                start = data[data["layer"].eq(0)]
                end = data[data["layer"].eq(47)]
                axis.scatter(
                    start["normalized_r2"],
                    start["consistency_value"],
                    color=color,
                    marker="o",
                    s=34,
                    edgecolors="white",
                    linewidths=0.7,
                    clip_on=False,
                    zorder=4,
                )
                axis.scatter(
                    end["normalized_r2"],
                    end["consistency_value"],
                    color=color,
                    marker="*",
                    s=72,
                    edgecolors="white",
                    linewidths=0.6,
                    clip_on=False,
                    zorder=5,
                )
                annotate_layers(axis, data, color, category_index)

            axis.set_xlim(0.0, 1.0)
            axis.set_xticks((0.0, 0.25, 0.5, 0.75, 1.0))
            axis.set_ylim(bottom=0.0)
            if row_index == 0:
                axis.set_title(str(MODEL_SPECS[analysis_name]["label"]))
            if row_index == len(METRIC_SPECS) - 1:
                axis.set_xlabel("Normalized mean per-protein all-pairs $R^2$")
            if column_index == 0:
                axis.set_ylabel(f"{metric_spec['title']}\n({metric_spec['ylabel']})")

    category_handles = [
        Line2D([0], [0], color=colors[label], linewidth=2.1, label=display_category(label))
        for label in labels
    ]
    endpoint_handles = [
        Line2D(
            [0],
            [0],
            color="black",
            marker="o",
            linestyle="none",
            markersize=5,
            label="Layer 0",
        ),
        Line2D(
            [0],
            [0],
            color="black",
            marker="*",
            linestyle="none",
            markersize=8,
            label="Layer 47",
        ),
    ]
    figure.legend(
        handles=category_handles + endpoint_handles,
        loc="outside lower center",
        ncol=len(category_handles) + len(endpoint_handles),
        frameon=False,
    )
    figure.suptitle(
        f"{category_spec['title']}: normalized decodability versus triangle consistency",
        y=1.005,
    )
    figure.tight_layout(rect=(0, 0.07, 1, 0.99))
    return figure


def save_figures(points: pd.DataFrame, generated_at: str, source_digest: str) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for system, spec in CATEGORY_SPECS.items():
        stem = (
            f"figure_{int(spec['figure_number']):02d}_{system}_decodability_consistency"
        )
        figure = plot_grid(points, system)
        for extension in ("png", "pdf"):
            path = OUTPUT_ROOT / f"{stem}.{extension}"
            figure.savefig(path)
            rows.append(
                {
                    "figure_number": int(spec["figure_number"]),
                    "category_system": system,
                    "figure_title": spec["title"],
                    "metrics": ";".join(METRIC_SPECS),
                    "format": extension,
                    "path": relative(path),
                    "sha256": sha256_file(path),
                    "source_digest": source_digest,
                    "generated_at": generated_at,
                }
            )
        plt.close(figure)
    return rows


def write_readme() -> Path:
    path = OUTPUT_ROOT / "README.md"
    lines = [
        "# Focused normalized decodability–consistency grids",
        "",
        (
            "These five phase figures combine the fixed reference-geometry categories from "
            "`focused_main_figures` with the decodability-versus-consistency trajectory style "
            "of main Figure 8. Figure 01 overlays the four probe models in two side-by-side "
            "metric panels. Figures 02–05 retain separate model columns and metric rows so "
            "their three category trajectories remain legible."
        ),
        "",
        "## Normalization and metrics",
        "",
        (
            "The x-axis starts from the protein-equal mean test-set `all_pairs` R² at each "
            "layer, then min–max normalizes those 48 means separately within each model. A "
            "value of 0 is that model's lowest layerwise mean R² and 1 is its highest. This "
            "makes relative layerwise progression comparable but removes absolute R² "
            "differences between models."
        ),
        "",
        (
            "`mean_relative_violation_all` is not min–max normalized. For each triangle, the "
            "positive triangle-inequality excess is divided by that reference triangle's RMS "
            "side length and then averaged over all triangles; nonviolating triangles "
            "contribute zero. Both y-metrics are protein-equal means over 111 proteins using "
            "the canonical `all_edges` triangle scope."
        ),
        "",
        "## Figures",
        "",
        "| Figure | Categories |",
        "|---|---|",
        "| 01 | All triangles |",
        "| 02 | RMS size: small, medium, large |",
        "| 03 | Area: low, medium, high |",
        "| 04 | Relative slack: low, medium, high |",
        "| 05 | Largest-angle shape: clearly acute, near right, clearly obtuse |",
        "",
        (
            "Paths connect layers 0–47 in order. Labels mark layers 0, 10, 20, 30, 40, and "
            "47; circles and stars emphasize layers 0 and 47. Figure 01 colors paths by "
            "model, while Figures 02–05 color paths by category. Category membership is "
            "fixed from final-PDB reference geometry. Axes are shared across model columns "
            "within each row of Figures 02–05."
        ),
        "",
        "## Evaluation caveats",
        "",
        (
            "The per-protein linear model's `all_pairs` R² mixes its fitted and held-out pair "
            "subsets because triangle consistency requires a complete decoded distance "
            "matrix. The other three probes evaluate unseen test proteins. These plots are "
            "descriptive; no confidence bands, bootstrap resampling, or inferential tests are "
            "included."
        ),
        "",
        "Exact coordinates are in `decodability_consistency_points.csv`. Regenerate with:",
        "",
        "```bash",
        ".venv/bin/python scripts/multi_protein/generate_normalized_decodability_consistency.py",
        "```",
        "",
    ]
    path.write_text("\n".join(lines))
    return path


def main() -> None:
    catalog.configure_style()
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 10.5,
            "axes.labelsize": 9,
            "figure.titlesize": 15,
            "savefig.dpi": 240,
        }
    )
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    generated_at = datetime.now(timezone.utc).isoformat()

    decodability, r2_paths = load_decodability()
    consistency = select_consistency_summary(INPUT_SUMMARY)
    points = build_points(decodability, consistency)
    points_path = OUTPUT_ROOT / "decodability_consistency_points.csv"
    points.to_csv(points_path, index=False)

    source_paths = [INPUT_SUMMARY, *r2_paths]
    source_digest = combined_digest(source_paths)
    figure_rows = save_figures(points, generated_at, source_digest)
    readme_path = write_readme()

    manifest_path = OUTPUT_ROOT / "figure_manifest.csv"
    pd.DataFrame(figure_rows).to_csv(manifest_path, index=False)
    artifact_paths = [
        points_path,
        readme_path,
        manifest_path,
        *(REPO_ROOT / row["path"] for row in figure_rows),
    ]
    generation_summary = {
        "analysis_name": "focused_normalized_decodability_consistency",
        "generated_at": generated_at,
        "source_files": [relative(path) for path in source_paths],
        "source_sha256": {relative(path): sha256_file(path) for path in source_paths},
        "source_digest": source_digest,
        "output_root": relative(OUTPUT_ROOT),
        "models": [MODEL_SPECS[name]["label"] for name in MODEL_ORDER],
        "layers": list(LAYERS),
        "category_systems": list(CATEGORY_SPECS),
        "metrics": list(METRIC_SPECS),
        "normalization": "per-model min-max over 48 protein-equal layer means",
        "r2_protein_split": "test",
        "r2_evaluation_scope": "all_pairs",
        "consistency_evaluation_scope": "all_edges",
        "aggregation": "protein_equal_mean",
        "n_proteins": EXPECTED_PROTEINS,
        "point_rows": int(len(points)),
        "distinct_figures": len(CATEGORY_SPECS),
        "rendered_figure_files": len(figure_rows),
        "artifact_sha256": {
            path.relative_to(OUTPUT_ROOT).as_posix(): sha256_file(path)
            for path in artifact_paths
        },
    }
    summary_path = OUTPUT_ROOT / "generation_summary.json"
    with summary_path.open("w") as handle:
        json.dump(generation_summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    print(
        f"Wrote {len(points):,} points and {len(CATEGORY_SPECS)} figures "
        f"({len(figure_rows)} PNG/PDF files) beneath {OUTPUT_ROOT}"
    )


if __name__ == "__main__":
    main()

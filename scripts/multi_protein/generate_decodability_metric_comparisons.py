#!/usr/bin/env python3
"""Plot layer trajectories in R2-versus-error/consistency space.

This local-only companion to ``generate_paper_figure_catalog.py`` expands main
Figure 8 to every canonical exact-triangle metric and every true spatial-
distance MAE bin. Outputs are intentionally isolated under a supplementary
paper-figure subdirectory rather than added to the main figure catalog.
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import generate_paper_figure_catalog as catalog

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


OUTPUT_DIR = catalog.PAPER_DIR / "decodability_metric_comparisons"
LAYERS = catalog.LAYERS
SELECTED_LAYERS = {0, 10, 20, 30, 40, 47}
CONFLICT_COLOR = "#B22222"

TRIANGLE_METRICS = {
    "triangle_violation_fraction": {
        "title": "Any triangle violation",
        "ylabel": "Mean violation fraction",
        "slug": "violation_fraction",
    },
    "triangle_mean_violation": {
        "title": "Mean violation across all triangles",
        "ylabel": "Mean violation (Å)",
        "slug": "mean_violation",
    },
    "conditional_triangle_violation_magnitude": {
        "title": "Mean magnitude conditional on violation",
        "ylabel": "Conditional mean magnitude (Å)",
        "slug": "conditional_mean_violation",
    },
    "triangle_max_violation": {
        "title": "Maximum triangle violation",
        "ylabel": "Mean per-protein maximum (Å)",
        "slug": "max_violation",
    },
    "triangle_violation_0_5_fraction": {
        "title": "Violation magnitude 0–5 Å",
        "ylabel": "Mean fraction of all triangles",
        "slug": "violation_0_5_fraction",
    },
    "triangle_violation_5_10_fraction": {
        "title": "Violation magnitude 5–10 Å",
        "ylabel": "Mean fraction of all triangles",
        "slug": "violation_5_10_fraction",
    },
    "triangle_violation_10_plus_fraction": {
        "title": "Violation magnitude 10+ Å",
        "ylabel": "Mean fraction of all triangles",
        "slug": "violation_10_plus_fraction",
    },
}

SPATIAL_METRICS = {
    distance_bin: {
        "title": f"True distance {distance_bin} Å",
        "ylabel": "Pooled MAE (Å)",
        "slug": distance_bin.replace("+", "_plus").replace("-", "_"),
    }
    for distance_bin in catalog.DISTANCE_BINS
}


def require_complete_model_layers(frame: pd.DataFrame, context: str) -> None:
    """Require one finite point for every model-layer combination."""
    expected = pd.MultiIndex.from_product(
        [catalog.MODEL_ORDER, LAYERS], names=["model", "layer"]
    )
    observed = pd.MultiIndex.from_frame(frame[["model", "layer"]].drop_duplicates())
    missing = expected.difference(observed)
    if len(missing):
        raise ValueError(f"{context} is missing model-layer points: {list(missing[:8])}")
    if not np.isfinite(frame[["r2_value", "comparison_value"]].to_numpy(dtype=float)).all():
        raise ValueError(f"{context} contains non-finite plotted values")


def derive_triangle_points(
    per_protein: dict[str, pd.DataFrame],
    geometry: dict[str, pd.DataFrame],
) -> pd.DataFrame:
    """Pair protein-equal all-pairs R2 with protein-equal geometry metrics."""
    rows: list[dict[str, object]] = []
    for model in catalog.MODEL_ORDER:
        accuracy = per_protein[model][
            per_protein[model]["protein_split"].eq("test")
            & per_protein[model]["evaluation_scope"].eq("all_pairs")
        ][["protein_id", "layer", "r2"]]
        metric_columns = ["protein_id", "layer", *TRIANGLE_METRICS]
        merged = accuracy.merge(
            geometry[model][metric_columns],
            on=["protein_id", "layer"],
            validate="one_to_one",
        )
        for metric in TRIANGLE_METRICS:
            for layer, group in merged.groupby("layer", sort=True):
                values = group[["r2", metric]].apply(pd.to_numeric, errors="coerce")
                finite = values[np.isfinite(values).all(axis=1)]
                rows.append(
                    {
                        "comparison_family": "triangle_metric",
                        "model": model,
                        "layer": int(layer),
                        "r2_metric": "mean_per_protein_r2",
                        "r2_value": float(finite["r2"].mean()),
                        "comparison_metric": metric,
                        "comparison_category": "all_triangles",
                        "comparison_value": float(finite[metric].mean()),
                        "r2_aggregation": "mean_per_protein",
                        "comparison_aggregation": "mean_per_protein",
                        "protein_split": "test",
                        "r2_evaluation_scope": "all_pairs",
                        "comparison_evaluation_scope": "all_pairs",
                        "n_proteins": int(len(finite)),
                    }
                )
    result = pd.DataFrame(rows)
    for metric in TRIANGLE_METRICS:
        require_complete_model_layers(
            result[result["comparison_metric"].eq(metric)], f"triangle metric {metric}"
        )
    return result


def derive_spatial_points(
    per_protein: dict[str, pd.DataFrame], spatial: pd.DataFrame
) -> pd.DataFrame:
    """Pair primary-scope mean R2 with primary-scope pooled spatial-bin MAE."""
    rows: list[dict[str, object]] = []
    for model in catalog.MODEL_ORDER:
        scope = str(catalog.MODEL_SPECS[model]["main_scope"])
        accuracy = per_protein[model][
            per_protein[model]["protein_split"].eq("test")
            & per_protein[model]["evaluation_scope"].eq(scope)
        ]
        r2_by_layer = (
            accuracy.groupby("layer", as_index=False)["r2"]
            .mean()
            .rename(columns={"r2": "r2_value"})
        )
        model_spatial = spatial[
            spatial["model"].eq(model)
            & spatial["protein_split"].eq("test")
            & spatial["evaluation_scope"].eq(scope)
        ].copy()
        merged = model_spatial.merge(r2_by_layer, on="layer", validate="many_to_one")
        for row in merged.itertuples():
            rows.append(
                {
                    "comparison_family": "spatial_distance_mae",
                    "model": model,
                    "layer": int(row.layer),
                    "r2_metric": "mean_per_protein_r2",
                    "r2_value": float(row.r2_value),
                    "comparison_metric": "pooled_mae",
                    "comparison_category": str(row.category_label),
                    "comparison_value": float(row.pooled_mae),
                    "r2_aggregation": "mean_per_protein",
                    "comparison_aggregation": "pooled_pairs",
                    "protein_split": "test",
                    "r2_evaluation_scope": scope,
                    "comparison_evaluation_scope": scope,
                    "n_proteins": int(row.n_proteins),
                }
            )
    result = pd.DataFrame(rows)
    if set(result["comparison_category"]) != set(catalog.DISTANCE_BINS):
        raise ValueError("Spatial comparisons do not contain the five canonical bins")
    for distance_bin in catalog.DISTANCE_BINS:
        require_complete_model_layers(
            result[result["comparison_category"].eq(distance_bin)],
            f"spatial-distance bin {distance_bin}",
        )
    return result


def add_transition_directions(points: pd.DataFrame) -> pd.DataFrame:
    """Annotate whether each incoming layer transition improves R2 but worsens y."""
    result = points.sort_values(
        ["comparison_family", "comparison_metric", "comparison_category", "model", "layer"]
    ).copy()
    group_columns = [
        "comparison_family",
        "comparison_metric",
        "comparison_category",
        "model",
    ]
    result["r2_change_from_previous_layer"] = result.groupby(group_columns)[
        "r2_value"
    ].diff()
    result["comparison_change_from_previous_layer"] = result.groupby(group_columns)[
        "comparison_value"
    ].diff()
    result["r2_up_comparison_worse"] = (
        result["r2_change_from_previous_layer"].gt(0)
        & result["comparison_change_from_previous_layer"].gt(0)
    )
    return result


def model_handles() -> list[Line2D]:
    handles = [
        Line2D(
            [0],
            [0],
            color=catalog.MODEL_SPECS[model]["color"],
            linestyle=catalog.MODEL_SPECS[model]["linestyle"],
            linewidth=2.2,
            label=model,
        )
        for model in catalog.MODEL_ORDER
    ]
    handles.append(
        Line2D(
            [0],
            [0],
            marker="o",
            linestyle="none",
            markerfacecolor="none",
            markeredgecolor=CONFLICT_COLOR,
            markeredgewidth=1.3,
            markersize=7,
            label="$R^2$ rises, plotted error rises",
        )
    )
    return handles


def plot_phase_axis(
    axis: plt.Axes,
    points: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    annotate: bool,
) -> None:
    cmap = plt.get_cmap("viridis")
    for model in catalog.MODEL_ORDER:
        data = points[points["model"].eq(model)].sort_values("layer")
        axis.plot(
            data["r2_value"],
            data["comparison_value"],
            color=catalog.MODEL_SPECS[model]["color"],
            linestyle=catalog.MODEL_SPECS[model]["linestyle"],
            linewidth=1.2,
            alpha=0.72,
        )
        axis.scatter(
            data["r2_value"],
            data["comparison_value"],
            c=data["layer"],
            cmap=cmap,
            vmin=0,
            vmax=47,
            s=20,
            edgecolors="none",
            zorder=3,
        )
        conflict = data[data["r2_up_comparison_worse"]]
        axis.scatter(
            conflict["r2_value"],
            conflict["comparison_value"],
            facecolors="none",
            edgecolors=CONFLICT_COLOR,
            linewidths=1.05,
            s=48,
            zorder=4,
        )
        if annotate:
            for row in data[data["layer"].isin(SELECTED_LAYERS)].itertuples():
                axis.annotate(
                    str(row.layer),
                    (row.r2_value, row.comparison_value),
                    xytext=(3, 3),
                    textcoords="offset points",
                    fontsize=6.5,
                    color=catalog.MODEL_SPECS[model]["color"],
                )
    axis.set(
        title=title,
        xlabel="Mean per-protein $R^2$",
        ylabel=ylabel,
    )
    axis.set_ylim(bottom=0)


def save_figure(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    png = OUTPUT_DIR / f"{stem}.png"
    pdf = OUTPUT_DIR / f"{stem}.pdf"
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def plot_individual(
    points: pd.DataFrame,
    *,
    title: str,
    ylabel: str,
    stem: str,
) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(8.2, 6.6))
    plot_phase_axis(axis, points, title=title, ylabel=ylabel, annotate=True)
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 47), cmap="viridis"),
        ax=axis,
        pad=0.02,
    )
    colorbar.set_label("Evoformer layer")
    axis.legend(handles=model_handles(), loc="best", fontsize=8)
    figure.tight_layout()
    return save_figure(figure, stem)


def plot_overview(
    points: pd.DataFrame,
    specifications: dict[str, dict[str, str]],
    *,
    selector_column: str,
    stem: str,
    suptitle: str,
) -> tuple[Path, Path]:
    ncols = 4 if len(specifications) > 6 else 3
    nrows = int(np.ceil(len(specifications) / ncols))
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 4.2 * nrows), squeeze=False
    )
    flat_axes = axes.ravel()
    for axis, (selector, spec) in zip(flat_axes, specifications.items()):
        subset = points[points[selector_column].eq(selector)]
        plot_phase_axis(
            axis,
            subset,
            title=spec["title"],
            ylabel=spec["ylabel"],
            annotate=False,
        )
    unused_axes = list(flat_axes[len(specifications) :])
    for axis in unused_axes:
        axis.axis("off")
    if unused_axes:
        unused_axes[0].legend(
            handles=model_handles(), loc="center", frameon=False, fontsize=9
        )
    else:
        flat_axes[0].legend(handles=model_handles(), loc="best", fontsize=7.5)
    colorbar_axis = figure.add_axes([0.965, 0.20, 0.012, 0.62])
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 47), cmap="viridis"),
        cax=colorbar_axis,
    )
    colorbar.set_label("Evoformer layer")
    figure.suptitle(suptitle, y=0.995)
    figure.subplots_adjust(left=0.06, right=0.94, bottom=0.08, top=0.92, wspace=0.31, hspace=0.34)
    return save_figure(figure, stem)


def main() -> None:
    catalog.configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_protein, per_protein_paths = catalog.load_per_protein_models()
    geometry, geometry_paths = catalog.load_geometry_models()
    spatial, spatial_paths = catalog.load_subgroups("by_spatial_distance.csv")

    triangle_points = derive_triangle_points(per_protein, geometry)
    spatial_points = derive_spatial_points(per_protein, spatial)
    points = add_transition_directions(
        pd.concat([triangle_points, spatial_points], ignore_index=True)
    )
    points.to_csv(OUTPUT_DIR / "comparison_points.csv", index=False)
    points[points["layer"].gt(0)].to_csv(
        OUTPUT_DIR / "layer_transition_directions.csv", index=False
    )

    output_rows: list[dict[str, object]] = []
    generated_at = datetime.now(timezone.utc).isoformat()
    triangle_sources = per_protein_paths + geometry_paths
    spatial_sources = per_protein_paths + spatial_paths

    for metric, spec in TRIANGLE_METRICS.items():
        subset = points[
            points["comparison_family"].eq("triangle_metric")
            & points["comparison_metric"].eq(metric)
        ]
        stem = f"r2_vs_triangle_{spec['slug']}"
        png, pdf = plot_individual(
            subset, title=spec["title"], ylabel=spec["ylabel"], stem=stem
        )
        output_rows.append(
            {
                "figure": stem,
                "comparison_family": "triangle_metric",
                "comparison_metric": metric,
                "png_file": catalog.relative(png),
                "pdf_file": catalog.relative(pdf),
                "source_files": ";".join(catalog.relative(path) for path in triangle_sources),
                "source_digest": catalog.combined_digest(triangle_sources),
                "generated_at": generated_at,
            }
        )

    for distance_bin, spec in SPATIAL_METRICS.items():
        subset = points[
            points["comparison_family"].eq("spatial_distance_mae")
            & points["comparison_category"].eq(distance_bin)
        ]
        stem = f"r2_vs_spatial_mae_{spec['slug']}a"
        png, pdf = plot_individual(
            subset, title=spec["title"], ylabel=spec["ylabel"], stem=stem
        )
        output_rows.append(
            {
                "figure": stem,
                "comparison_family": "spatial_distance_mae",
                "comparison_metric": f"pooled_mae_{distance_bin}A",
                "png_file": catalog.relative(png),
                "pdf_file": catalog.relative(pdf),
                "source_files": ";".join(catalog.relative(path) for path in spatial_sources),
                "source_digest": catalog.combined_digest(spatial_sources),
                "generated_at": generated_at,
            }
        )

    triangle_png, triangle_pdf = plot_overview(
        points[points["comparison_family"].eq("triangle_metric")],
        TRIANGLE_METRICS,
        selector_column="comparison_metric",
        stem="overview_r2_vs_triangle_metrics",
        suptitle="Decodability versus exact triangle-inequality metrics",
    )
    spatial_png, spatial_pdf = plot_overview(
        points[points["comparison_family"].eq("spatial_distance_mae")],
        SPATIAL_METRICS,
        selector_column="comparison_category",
        stem="overview_r2_vs_spatial_distance_mae",
        suptitle="Decodability versus MAE by true spatial distance",
    )
    for stem, family, metric, png, pdf, sources in (
        (
            "overview_r2_vs_triangle_metrics",
            "triangle_metric",
            "seven_panel_overview",
            triangle_png,
            triangle_pdf,
            triangle_sources,
        ),
        (
            "overview_r2_vs_spatial_distance_mae",
            "spatial_distance_mae",
            "five_panel_overview",
            spatial_png,
            spatial_pdf,
            spatial_sources,
        ),
    ):
        output_rows.append(
            {
                "figure": stem,
                "comparison_family": family,
                "comparison_metric": metric,
                "png_file": catalog.relative(png),
                "pdf_file": catalog.relative(pdf),
                "source_files": ";".join(catalog.relative(path) for path in sources),
                "source_digest": catalog.combined_digest(sources),
                "generated_at": generated_at,
            }
        )

    manifest = pd.DataFrame(output_rows)
    for column in ("png_file", "pdf_file"):
        for filename in manifest[column]:
            path = catalog.PROJECT_ROOT / filename
            if not path.exists() or path.stat().st_size == 0:
                raise ValueError(f"Missing or empty figure output: {filename}")
    manifest.to_csv(OUTPUT_DIR / "figure_manifest.csv", index=False)

    transitions = points[points["layer"].gt(0)]
    conflict_summary = (
        transitions.groupby(
            ["comparison_family", "comparison_metric", "comparison_category"]
        )["r2_up_comparison_worse"]
        .agg(["sum", "count"])
        .reset_index()
    )
    summary = {
        "generated_at": generated_at,
        "individual_figures": len(TRIANGLE_METRICS) + len(SPATIAL_METRICS),
        "overview_figures": 2,
        "models": list(catalog.MODEL_ORDER),
        "layers": list(LAYERS),
        "comparison_points_file": catalog.relative(OUTPUT_DIR / "comparison_points.csv"),
        "transition_file": catalog.relative(OUTPUT_DIR / "layer_transition_directions.csv"),
        "manifest": catalog.relative(OUTPUT_DIR / "figure_manifest.csv"),
        "r2_up_metric_worse_transitions": int(
            transitions["r2_up_comparison_worse"].sum()
        ),
        "transition_counts_by_comparison": conflict_summary.to_dict(orient="records"),
    }
    (OUTPUT_DIR / "generation_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n"
    )
    print(
        f"Generated {len(manifest)} figures in {catalog.relative(OUTPUT_DIR)} "
        f"from {len(points)} model-layer comparison points."
    )


if __name__ == "__main__":
    main()

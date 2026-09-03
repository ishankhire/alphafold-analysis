#!/usr/bin/env python3
"""Plot corrected layerwise-ablation trajectories and probe-metric overlays."""

from __future__ import annotations

import os
import shutil
from pathlib import Path
from typing import Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


INPUT_DIR = PROJECT_ROOT / "delta_files/layerwise_ablation_csvs"
REGRESSION_ROOT = PROJECT_ROOT / "delta_files/regression_5fold_v1"
COHORT_PATH = REGRESSION_ROOT / "metadata/regression_outer_folds_v1.csv"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "delta_files/publication_plots/sept_2_plots/layerwise_ablation_trajectories"
)
OVERLEAF_OUTPUT_DIR = PROJECT_ROOT / "overleaf/layerwise_ablation_trends"
SOURCE_DATA_DIR = PROJECT_ROOT / "delta_files/publication_plots/sept_2_plots/source_data"
R2_SUMMARY_PATH = SOURCE_DATA_DIR / "r2_plot_summary.csv"
TRIANGLE_SUMMARY_PATH = SOURCE_DATA_DIR / "triangle_plot_summary.csv"

LAYERS = tuple(range(48))
LAYER_COLUMNS = tuple(f"L{layer}" for layer in LAYERS)
EXPECTED_SOURCE_PROTEINS = 554
EXPECTED_PROTEINS = 552
EXPECTED_EXCLUDED_KEYS = frozenset(
    {("ecod", "000196893"), ("ecod", "001681437")}
)

ABLATIONS = (
    {
        "key": "pair_transition_mlp",
        "label": "Pair transition MLPs",
        "source_stem": "ablate_pair_transition_layerwise",
    },
    {
        "key": "msa_mlp",
        "label": "MSA MLPs",
        "source_stem": "ablate_msa_transition_layerwise",
    },
    {
        "key": "tmu_triangle_attention",
        "label": "TMU + triangle attention",
        "source_stem": "ablate_tri_start_ablate_tri_end_ablate_tmu_layerwise",
    },
)
DIRECTIONS = {"fwd": "FWD", "rev": "REV"}

MODEL_SPECS = {
    "shared_ols_base256": {
        "label": "Shared linear",
        "color": "#4C78A8",
        "linestyle": "-",
    },
    "per_protein_ols_base256": {
        "label": "Per-protein linear",
        "color": "#7F7F7F",
        "linestyle": (0, (5, 2)),
    },
    "shared_xgboost_base256_fixed_s4": {
        "label": "XGBoost",
        "color": "#E45756",
        "linestyle": "-",
    },
    "sequence_gated_ols_base256": {
        "label": "Sequence gated",
        "color": "#F58518",
        "linestyle": "-",
    },
}
MODEL_ORDER = tuple(MODEL_SPECS)

PROBE_METRICS = {
    "r2": {
        "stem": "r2",
        "ylabel": "Mean per-protein $R^2$",
        "title": "mean per-protein $R^2$",
    },
    "mean_relative_violation_all": {
        "stem": "normalized_triangle_violation",
        "ylabel": "Mean normalized triangle violation",
        "title": "mean normalized triangle violation",
    },
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_protein_ids(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.zfill(9)
    if normalized.isna().any() or normalized.eq("").any():
        raise ValueError("Protein IDs contain missing or empty values")
    return normalized


def load_cohort(path: Path = COHORT_PATH) -> pd.MultiIndex:
    if not path.is_file():
        raise FileNotFoundError(f"Missing frozen cohort manifest: {path}")
    frame = pd.read_csv(path, dtype={"protein_id": "string"})
    required = {"source", "protein_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    frame["source"] = frame["source"].astype("string").str.strip().str.lower()
    frame["protein_id"] = normalize_protein_ids(frame["protein_id"])
    if len(frame) != EXPECTED_PROTEINS:
        raise ValueError(
            f"{path.name} has {len(frame)} rows; expected {EXPECTED_PROTEINS}"
        )
    if frame.duplicated(["source", "protein_id"]).any():
        raise ValueError(f"{path.name} contains duplicate source/protein keys")
    if frame["protein_id"].nunique() != EXPECTED_PROTEINS:
        raise ValueError(f"{path.name} does not contain 552 unique protein IDs")
    return pd.MultiIndex.from_frame(
        frame.sort_values(["source", "protein_id"], kind="stable")[["source", "protein_id"]]
    )


def load_tm_score_matrix(
    path: Path, cohort_keys: pd.MultiIndex
) -> tuple[pd.MultiIndex, np.ndarray]:
    """Load, audit, and filter one 554-row ablation table to the frozen cohort."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required ablation input: {path}")

    frame = pd.read_csv(path, dtype={"protein": "string"})
    required = {"dataset", "protein", "experiment", *LAYER_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if len(frame) != EXPECTED_SOURCE_PROTEINS:
        raise ValueError(
            f"{path.name} has {len(frame)} proteins; expected {EXPECTED_SOURCE_PROTEINS}"
        )

    frame["source"] = frame["dataset"].astype("string").str.strip().str.lower()
    frame["protein_id"] = normalize_protein_ids(frame["protein"])
    if frame.duplicated(["source", "protein_id"]).any():
        raise ValueError(f"{path.name} contains duplicate source/protein keys")
    if frame["protein_id"].nunique() != EXPECTED_SOURCE_PROTEINS:
        raise ValueError(f"{path.name} does not contain 554 unique protein IDs")

    source_keys = pd.MultiIndex.from_frame(frame[["source", "protein_id"]])
    source_key_set = set(source_keys.tolist())
    cohort_key_set = set(cohort_keys.tolist())
    missing_cohort = cohort_key_set - source_key_set
    exclusions = source_key_set - cohort_key_set
    if missing_cohort:
        raise ValueError(
            f"{path.name} is missing frozen-cohort proteins: {sorted(missing_cohort)}"
        )
    if exclusions != EXPECTED_EXCLUDED_KEYS:
        raise ValueError(
            f"{path.name} has unexpected cohort exclusions: {sorted(exclusions)}; "
            f"expected {sorted(EXPECTED_EXCLUDED_KEYS)}"
        )

    indexed = frame.set_index(["source", "protein_id"])
    filtered = indexed.reindex(cohort_keys)
    matrix = filtered.loc[:, LAYER_COLUMNS].to_numpy(dtype=float)
    if matrix.shape != (EXPECTED_PROTEINS, len(LAYERS)):
        raise ValueError(f"{path.name} has an unexpected filtered TM-score shape")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path.name} contains nonfinite TM-scores")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError(f"{path.name} contains TM-scores outside [0, 1]")
    return cohort_keys, matrix


def validate_probe_summary(
    frame: pd.DataFrame,
    *,
    analyses: Sequence[str] = MODEL_ORDER,
    layers: Sequence[int] = LAYERS,
) -> dict[str, np.ndarray]:
    """Validate a complete 552-protein publication summary and return its means."""
    required = {
        "analysis_name",
        "layer",
        "mean",
        "n_proteins_total",
        "n_proteins_finite",
        "aggregation",
    }
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Probe summary is missing columns: {missing}")

    values = frame.copy()
    values["layer"] = pd.to_numeric(values["layer"], errors="raise").astype(int)
    expected_analyses = set(analyses)
    expected_layers = set(layers)
    if set(values["analysis_name"]) != expected_analyses:
        raise ValueError("Probe summary does not contain all four models")
    if set(values["layer"]) != expected_layers:
        raise ValueError("Probe summary does not contain all 48 layers")
    expected_rows = len(analyses) * len(layers)
    if len(values) != expected_rows:
        raise ValueError(
            f"Probe summary has {len(values)} rows; expected {expected_rows}"
        )
    if values.duplicated(["analysis_name", "layer"]).any():
        raise ValueError("Probe summary contains duplicate model/layer keys")
    if not values["n_proteins_total"].eq(EXPECTED_PROTEINS).all():
        raise ValueError("Probe summary does not report 552 total proteins")
    if not values["n_proteins_finite"].eq(EXPECTED_PROTEINS).all():
        raise ValueError("Probe summary does not report 552 finite proteins")
    if not values["aggregation"].eq("protein_equal_mean").all():
        raise ValueError("Probe summary is not protein-equal")
    numeric = pd.to_numeric(values["mean"], errors="coerce")
    if not np.isfinite(numeric).all():
        raise ValueError("Probe summary contains nonfinite means")
    values["mean"] = numeric

    result: dict[str, np.ndarray] = {}
    for analysis in analyses:
        mean = (
            values[values["analysis_name"].eq(analysis)]
            .set_index("layer")["mean"]
            .reindex(layers)
        )
        if mean.isna().any() or len(mean) != len(layers):
            raise ValueError("Probe summary has an incomplete mean trajectory")
        result[analysis] = mean.to_numpy(dtype=float)
    return result


def load_probe_means(
    cohort_keys: pd.MultiIndex,
) -> dict[str, dict[str, np.ndarray]]:
    if len(cohort_keys) != EXPECTED_PROTEINS:
        raise ValueError("Probe summaries require the frozen 552-protein cohort")
    for path in (R2_SUMMARY_PATH, TRIANGLE_SUMMARY_PATH):
        if not path.is_file():
            raise FileNotFoundError(f"Missing publication summary input: {path}")

    r2_means = validate_probe_summary(pd.read_csv(R2_SUMMARY_PATH))
    triangle = pd.read_csv(TRIANGLE_SUMMARY_PATH)
    required = {"category_system", "category_label", "metric"}
    missing = sorted(required - set(triangle.columns))
    if missing:
        raise ValueError(f"Triangle summary is missing columns: {missing}")
    triangle = triangle[
        triangle["category_system"].eq("all_triangles")
        & triangle["category_label"].eq("all")
        & triangle["metric"].eq("mean_relative_violation_all")
    ].copy()
    triangle_means = validate_probe_summary(triangle)
    return {"r2": r2_means, "mean_relative_violation_all": triangle_means}


def draw_ablation_trajectory(
    axis: plt.Axes,
    matrix: np.ndarray,
    *,
    trajectory_alpha: float = 0.12,
    mean_label: str | None = "Mean ablation TM-score",
) -> None:
    layers = np.asarray(LAYERS)
    segments = np.stack(
        (
            np.broadcast_to(layers, matrix.shape),
            matrix,
        ),
        axis=-1,
    )
    axis.add_collection(
        LineCollection(
            segments,
            colors="#8A8A8A",
            linewidths=0.55,
            alpha=trajectory_alpha,
            rasterized=True,
        )
    )
    axis.plot(
        layers,
        matrix.mean(axis=0),
        color="black",
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        markeredgewidth=0,
        zorder=5,
        label=mean_label,
    )


def style_ablation_axis(axis: plt.Axes, *, label: str, direction: str) -> None:
    axis.set(
        title=f"{label}: {DIRECTIONS[direction]} layerwise ablation",
        xlabel="Ablation boundary layer",
        ylabel="TM-score",
        xlim=(0, 47),
        ylim=(0, 1),
    )
    axis.set_xticks((0, 10, 20, 30, 40, 47))


def plot_trajectory(*, matrix: np.ndarray, stem: str) -> tuple[Path, Path, Path]:
    figure, axis = plt.subplots(figsize=(7.0, 5.3))
    draw_ablation_trajectory(
        axis,
        matrix,
        trajectory_alpha=0.10,
        mean_label=None,
    )
    axis.set(
        xlabel="Layer",
        ylabel="TM-score",
        xlim=(0, 47),
        ylim=(0, 1),
    )
    axis.set_xticks((0, 10, 20, 30, 40, 47))
    axis.xaxis.label.set_size(14)
    axis.yaxis.label.set_size(14)
    axis.tick_params(axis="both", labelsize=12)
    figure.tight_layout()

    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    overleaf_path = OVERLEAF_OUTPUT_DIR / f"{stem}.pdf"
    figure.savefig(png_path)
    figure.savefig(pdf_path)
    shutil.copyfile(pdf_path, overleaf_path)
    plt.close(figure)
    return png_path, pdf_path, overleaf_path


def padded_limits(trajectories: Iterable[np.ndarray]) -> tuple[float, float]:
    values = np.concatenate([np.asarray(item, dtype=float) for item in trajectories])
    lower = float(values.min())
    upper = float(values.max())
    span = upper - lower
    padding = 0.06 * span if span > 0 else max(abs(lower) * 0.06, 0.01)
    return lower - padding, upper + padding


def plot_overlay(
    *,
    matrix: np.ndarray,
    label: str,
    direction: str,
    stem: str,
    metric: str,
    means: Mapping[str, np.ndarray],
) -> Path:
    spec = PROBE_METRICS[metric]
    figure, axis = plt.subplots(figsize=(9.2, 5.7))
    draw_ablation_trajectory(axis, matrix)
    style_ablation_axis(axis, label=label, direction=direction)
    axis.set_title(
        f"{label}: {DIRECTIONS[direction]} ablation and {spec['title']}"
    )

    probe_axis = axis.twinx()
    probe_axis.grid(False)
    probe_axis.spines["right"].set_visible(True)
    for analysis in MODEL_ORDER:
        model = MODEL_SPECS[analysis]
        probe_axis.plot(
            LAYERS,
            means[analysis],
            color=model["color"],
            linestyle=model["linestyle"],
            linewidth=2.0,
            label=model["label"],
            zorder=6,
        )
    probe_axis.set_ylabel(str(spec["ylabel"]))
    probe_axis.set_ylim(*padded_limits(means.values()))

    handles_left, labels_left = axis.get_legend_handles_labels()
    handles_right, labels_right = probe_axis.get_legend_handles_labels()
    axis.legend(
        handles_left + handles_right,
        labels_left + labels_right,
        loc="best",
        frameon=True,
    )
    axis.text(
        0.01,
        0.02,
        f"{matrix.shape[0]} proteins",
        transform=axis.transAxes,
        ha="left",
        va="bottom",
        color="#555555",
        fontsize=9,
    )
    figure.tight_layout()

    path = OVERLEAF_OUTPUT_DIR / f"{stem}_with_{spec['stem']}.pdf"
    figure.savefig(path)
    plt.close(figure)
    return path


def expected_overleaf_filenames() -> set[str]:
    names: set[str] = set()
    for ablation in ABLATIONS:
        for direction in DIRECTIONS:
            base = f"{ablation['key']}_{direction}_tm_score_trajectory"
            names.add(f"{base}.pdf")
            for metric in PROBE_METRICS.values():
                names.add(f"{ablation['key']}_{direction}_tm_score_with_{metric['stem']}.pdf")
    return names


def write_readme() -> None:
    excluded = ", ".join(key[1] for key in sorted(EXPECTED_EXCLUDED_KEYS))
    text = f"""# Selected layerwise-ablation TM-score trajectories

This folder contains FWD and REV layerwise ablations for Pair transition MLPs,
MSA MLPs (`msa_transition`), and the combined TMU + triangle-attention
(`tri_start`, `tri_end`, and `tmu`) intervention. Each figure is supplied as
both PNG and PDF.

The ablation inputs contain 554 proteins. These plots use the frozen 552-protein
protocol-2.0 regression cohort and exclude ECOD proteins {excluded}. Every graph
plots all 552 retained proteins as thin, semi-transparent gray lines. The thicker
black line is their protein-equal mean, with a black dot at every layer; no
uncertainty band is shown. The x-axis, labeled `Layer`, is the boundary layer
recorded by each FWD or REV source CSV.

PDF-only copies and twin-axis overlays with the four protocol-2.0 probe families
are stored in `overleaf/layerwise_ablation_trends/`. The overlay probe curves use
the validated protein-equal means across the same 552 proteins in the September 2
`r2_plot_summary.csv` and `triangle_plot_summary.csv` source tables. The triangle
curve is the all-triangle `mean_relative_violation_all` metric.

Figures are regenerated by
`scripts/multi_protein/plot_layerwise_ablation_tm_scores.py`.
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    OVERLEAF_OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort_keys = load_cohort()
    probe_means = load_probe_means(cohort_keys)
    generated_delta: list[Path] = []
    generated_overleaf: list[Path] = []
    for ablation in ABLATIONS:
        for direction in DIRECTIONS:
            filename = f"{ablation['source_stem']}_{direction}.csv"
            keys, matrix = load_tm_score_matrix(INPUT_DIR / filename, cohort_keys)
            if not keys.equals(cohort_keys):
                raise ValueError(f"{filename} does not match the frozen protein cohort")

            base_stem = f"{ablation['key']}_{direction}_tm_score_trajectory"
            png, pdf, overleaf_pdf = plot_trajectory(
                matrix=matrix,
                stem=base_stem,
            )
            generated_delta.extend((png, pdf))
            generated_overleaf.append(overleaf_pdf)
            overlay_stem = f"{ablation['key']}_{direction}_tm_score"
            for metric, means in probe_means.items():
                generated_overleaf.append(
                    plot_overlay(
                        matrix=matrix,
                        label=str(ablation["label"]),
                        direction=direction,
                        stem=overlay_stem,
                        metric=metric,
                        means=means,
                    )
                )

    write_readme()
    if len(generated_delta) != 12 or not all(path.is_file() for path in generated_delta):
        raise RuntimeError("The six Delta PNG/PDF figure pairs were not generated")
    generated_names = {path.name for path in generated_overleaf}
    expected_names = expected_overleaf_filenames()
    if generated_names != expected_names or len(generated_overleaf) != 18:
        raise RuntimeError("The expected 18 Overleaf PDFs were not generated")
    on_disk_names = {path.name for path in OVERLEAF_OUTPUT_DIR.glob("*.pdf")}
    if on_disk_names != expected_names:
        raise RuntimeError("The Overleaf output directory has an unexpected PDF inventory")
    print(
        f"Generated 6 Delta PNG/PDF pairs and 18 Overleaf PDFs from "
        f"{EXPECTED_PROTEINS} proteins"
    )


if __name__ == "__main__":
    main()

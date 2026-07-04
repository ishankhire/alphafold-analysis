#!/usr/bin/env python3
"""Create local analysis plots for the combined per-protein base256 run."""

from __future__ import annotations

import math
import os
from pathlib import Path

EARLY_PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.makedirs(EARLY_PROJECT_ROOT / ".mplconfig", exist_ok=True)
os.makedirs(EARLY_PROJECT_ROOT / ".cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(EARLY_PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(EARLY_PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


PROJECT_ROOT = EARLY_PROJECT_ROOT
DELTA_DIR = PROJECT_ROOT / "delta_files"
PER_PROTEIN_DIR = DELTA_DIR / "per_protein_base256_combined"
GENERAL_DIR = DELTA_DIR / "general_base256_combined"
OUT_DIR = PER_PROTEIN_DIR / "analysis"

PREFIX = "per_protein_base256_combined"
EXPECTED_LAYERS = tuple(range(48))
FEATURE_DIM = 256
R2_COLUMN = "test_r2"
R2_LABEL = "Held-out pair R$^2$"

ORANGE = "#f58518"
BLUE = "#4c78a8"
RED = "#e45756"
GREEN = "#54a24b"
GREY = "#b8b8b8"
GRID_GREY = "#dddddd"

# Exact per-triangle metric columns (authoritative; sourced from
# exact_triangle_metrics.csv, replacing the old min-indirect-path triangle fields).
TRIANGLE_COLUMNS = [
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]
TRIANGLE_BIN_PANELS = [
    ("triangle_violation_0_5_fraction", "0–5 Å", GREEN),
    ("triangle_violation_5_10_fraction", "5–10 Å", ORANGE),
    ("triangle_violation_10_plus_fraction", "≥10 Å", RED),
]


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 12,
            "axes.titlesize": 16,
            "axes.labelsize": 14,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 18,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def require_columns(path: Path, df: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def require_layers(path: Path, df: pd.DataFrame) -> None:
    layers = tuple(sorted(int(value) for value in df["layer"].dropna().unique()))
    if layers != EXPECTED_LAYERS:
        raise ValueError(f"{path} expected layers 0-47, found {layers}")


def require_no_duplicates(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    n_duplicates = int(df.duplicated(columns).sum())
    if n_duplicates:
        raise ValueError(f"{path} has {n_duplicates} duplicate rows on {columns}")


def require_nonempty(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Expected nonempty output file: {path}")


def load_per_protein_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    per_protein_path = PER_PROTEIN_DIR / "per_protein.csv"
    geometry_path = PER_PROTEIN_DIR / "geometry_metrics.csv"

    per_protein = pd.read_csv(per_protein_path)
    geometry = pd.read_csv(geometry_path)

    require_columns(
        per_protein_path,
        per_protein,
        {
            "layer",
            "source",
            "protein_id",
            "n_residues",
            "feature_dim",
            "per_protein_r2",
            "train_r2",
            "test_r2",
        },
    )
    require_columns(
        geometry_path,
        geometry,
        {
            "layer",
            "source",
            "protein_id",
            "n_residues",
            "feature_dim",
            "per_protein_r2",
            "train_r2",
            "test_r2",
            "tm_score",
            "frac_top3_eigenvalues",
            "n_negative_eigenvalues",
            "neg_eigenvalue_fraction",
            *TRIANGLE_COLUMNS,
        },
    )

    for path, df in [(per_protein_path, per_protein), (geometry_path, geometry)]:
        require_layers(path, df)
        feature_dims = tuple(sorted(int(value) for value in df["feature_dim"].dropna().unique()))
        if feature_dims != (FEATURE_DIM,):
            raise ValueError(f"{path} expected feature_dim 256, found {feature_dims}")
        require_no_duplicates(path, df, ["layer", "protein_id"])

    per_ids = set(per_protein["protein_id"])
    geom_ids = set(geometry["protein_id"])
    if per_ids != geom_ids:
        raise ValueError("per_protein.csv and geometry_metrics.csv have different protein sets")

    expected_rows = len(per_ids) * len(EXPECTED_LAYERS)
    if per_protein.shape[0] != expected_rows or geometry.shape[0] != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows in both inputs, found "
            f"{per_protein.shape[0]} and {geometry.shape[0]}"
        )

    return per_protein, geometry


def load_general_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    per_protein_path = GENERAL_DIR / "per_protein.csv"
    geometry_path = GENERAL_DIR / "geometry_metrics.csv"

    per_protein = pd.read_csv(per_protein_path)
    geometry = pd.read_csv(geometry_path)

    require_columns(
        per_protein_path,
        per_protein,
        {"split_id", "layer", "split", "protein_id", "n_residues", "per_protein_r2"},
    )
    require_columns(
        geometry_path,
        geometry,
        {"split_id", "layer", "split", "protein_id", "n_residues", "tm_score"},
    )
    require_layers(per_protein_path, per_protein)
    require_layers(geometry_path, geometry)
    require_no_duplicates(per_protein_path, per_protein, ["split_id", "layer", "split", "protein_id"])
    require_no_duplicates(geometry_path, geometry, ["split_id", "layer", "split", "protein_id"])

    return per_protein, geometry


def layer_mean(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    return df.groupby("layer", sort=True)[value_columns].mean().reset_index()


def general_test_layer_mean(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise ValueError("general_base256_combined input has no test rows")
    return test.groupby("layer", sort=True)[value_columns].mean().reset_index()


def add_common_axis_style(ax: plt.Axes) -> None:
    ax.set_xlim(0, 47)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 47])
    ax.grid(True, color=GRID_GREY, linewidth=0.8, alpha=0.7)
    ax.tick_params(axis="both", width=1.0, length=5)


def padded_limits(values: list[np.ndarray], include_zero: bool = False) -> tuple[float, float]:
    arrays = [np.asarray(value, dtype=np.float64).ravel() for value in values if np.asarray(value).size]
    if not arrays:
        return (0.0, 1.0)
    flat = np.concatenate(arrays)
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return (0.0, 1.0)
    ymin = float(np.min(flat))
    ymax = float(np.max(flat))
    if include_zero:
        ymin = min(ymin, 0.0)
        ymax = max(ymax, 0.0)
    if math.isclose(ymin, ymax):
        pad = 0.05 if math.isclose(ymin, 0.0) else abs(ymin) * 0.05
        return (ymin - pad, ymax + pad)
    pad = (ymax - ymin) * 0.06
    return (ymin - pad, ymax + pad)


def plot_r2_tm_with_trajectories(per_protein: pd.DataFrame, geometry: pd.DataFrame) -> Path:
    out_path = OUT_DIR / f"{PREFIX}_r2_tmscore_vs_layer.png"
    r2_mean = layer_mean(per_protein, [R2_COLUMN])
    tm_mean = layer_mean(geometry, ["tm_score"])

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 9.2), sharex=True, constrained_layout=True)

    for _, group in per_protein.sort_values(["protein_id", "layer"]).groupby("protein_id", sort=True):
        axes[0].plot(group["layer"], group[R2_COLUMN], color=GREY, linewidth=0.65, alpha=0.35)
    axes[0].plot(
        r2_mean["layer"],
        r2_mean[R2_COLUMN],
        color=ORANGE,
        linewidth=2.8,
        label="Held-out pair mean",
    )
    axes[0].set_title("per_protein_base256_combined held-out pair R$^2$ across layers")
    axes[0].set_ylabel(R2_LABEL)
    axes[0].set_ylim(padded_limits([per_protein[R2_COLUMN].to_numpy()], include_zero=True))
    add_common_axis_style(axes[0])
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=GREY, linewidth=1.0, alpha=0.55, label="Individual proteins"),
            Line2D([0], [0], color=ORANGE, linewidth=2.8, label="Held-out pair mean"),
        ],
        loc="lower right",
        frameon=False,
    )

    for _, group in geometry.sort_values(["protein_id", "layer"]).groupby("protein_id", sort=True):
        axes[1].plot(group["layer"], group["tm_score"], color=GREY, linewidth=0.65, alpha=0.35)
    axes[1].plot(
        tm_mean["layer"],
        tm_mean["tm_score"],
        color=ORANGE,
        linewidth=2.8,
        label="Per-protein model mean",
    )
    axes[1].set_title("per_protein_base256_combined TM-score across layers")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("TM-score")
    axes[1].set_ylim(0.0, 1.0)
    add_common_axis_style(axes[1])
    axes[1].legend(loc="lower right", frameon=False)

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_r2_tm_mean_only(per_protein: pd.DataFrame, geometry: pd.DataFrame) -> Path:
    out_path = OUT_DIR / f"{PREFIX}_r2_tmscore_vs_layer_mean_only.png"
    r2_mean = layer_mean(per_protein, [R2_COLUMN])
    tm_mean = layer_mean(geometry, ["tm_score"])

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.5), sharex=True, constrained_layout=True)
    for ax, mean_df, column, title, ylabel in [
        (
            axes[0],
            r2_mean,
            R2_COLUMN,
            "per_protein_base256_combined mean held-out pair R$^2$ across layers",
            f"Mean {R2_LABEL}",
        ),
        (
            axes[1],
            tm_mean,
            "tm_score",
            "per_protein_base256_combined mean TM-score across layers",
            "Mean TM-score",
        ),
    ]:
        ax.plot(mean_df["layer"], mean_df[column], color=ORANGE, linewidth=2.8, label="Per-protein model mean")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        add_common_axis_style(ax)
        ax.legend(loc="lower right", frameon=False)
    axes[1].set_xlabel("Layer")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_geometry_layer_means(geometry: pd.DataFrame) -> Path:
    df = geometry.copy()
    df["negative_eigenvalue_count_fraction"] = df["n_negative_eigenvalues"] / df["n_residues"]
    value_columns = [
        "per_protein_r2",
        "train_r2",
        "test_r2",
        "tm_score",
        "frac_top3_eigenvalues",
        "n_negative_eigenvalues",
        "negative_eigenvalue_count_fraction",
        "neg_eigenvalue_fraction",
        *TRIANGLE_COLUMNS,
    ]
    mean_df = layer_mean(df, value_columns)
    out_path = OUT_DIR / f"{PREFIX}_geometry_layer_means.csv"
    mean_df.to_csv(out_path, index=False, float_format="%.6f")
    return out_path


def plot_triangle_violations(geometry: pd.DataFrame) -> Path:
    out_path = OUT_DIR / f"{PREFIX}_triangle_violations_vs_layer.png"
    bin_cols = [column for column, _, _ in TRIANGLE_BIN_PANELS]
    mean_df = layer_mean(
        geometry,
        ["triangle_violation_fraction", "triangle_mean_violation", "triangle_max_violation", *bin_cols],
    )
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 9.0), sharex=True, constrained_layout=True)

    for ax, column, title, ylabel in [
        (axes[0, 0], "triangle_violation_fraction", "Any triangle inequality violation", "Mean violating-triangle fraction"),
        (axes[0, 1], "triangle_mean_violation", "Mean violation magnitude (over all triangles)", "Mean magnitude (Å)"),
        (axes[1, 1], "triangle_max_violation", "Max violation magnitude", "Mean per-protein max (Å)"),
    ]:
        ax.plot(mean_df["layer"], mean_df[column], color=ORANGE, linewidth=2.8,
                marker="o", markersize=3.2, label="Per-protein model mean")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(padded_limits([mean_df[column].to_numpy()], include_zero=True))
        add_common_axis_style(ax)
        ax.legend(loc="best", frameon=False)

    ax_bins = axes[1, 0]
    for column, bin_label, color in TRIANGLE_BIN_PANELS:
        ax_bins.plot(mean_df["layer"], mean_df[column], color=color, linewidth=2.2,
                     marker="o", markersize=3.2, label=bin_label)
    ax_bins.set_title("Violation severity bins")
    ax_bins.set_ylabel("Mean fraction of triangles")
    ax_bins.set_ylim(padded_limits([mean_df[column].to_numpy() for column in bin_cols], include_zero=True))
    add_common_axis_style(ax_bins)
    ax_bins.legend(loc="best", frameon=False, title="Violation size")

    axes[1, 0].set_xlabel("Layer")
    axes[1, 1].set_xlabel("Layer")
    fig.suptitle("per_protein_base256_combined exact triangle inequality violations across layers")
    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_top3_eigenvalue_fraction(geometry: pd.DataFrame) -> Path:
    out_path = OUT_DIR / f"{PREFIX}_top3_eigenvalue_fraction_vs_layer.png"
    mean_df = layer_mean(geometry, ["frac_top3_eigenvalues"])
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)

    ax.plot(
        mean_df["layer"],
        mean_df["frac_top3_eigenvalues"],
        color=ORANGE,
        linewidth=2.8,
        label="Per-protein model mean",
    )
    ax.set_title("Top-3 eigenvalue fraction across layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean top-3 eigenvalue fraction")
    ax.set_ylim(0.0, 1.0)
    add_common_axis_style(ax)
    ax.legend(loc="best", frameon=False)

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_negative_eigenvalues(geometry: pd.DataFrame) -> Path:
    out_path = OUT_DIR / f"{PREFIX}_negative_eigenvalues_vs_layer.png"
    df = geometry.copy()
    df["negative_eigenvalue_count_fraction"] = df["n_negative_eigenvalues"] / df["n_residues"]
    mean_df = layer_mean(
        df,
        ["n_negative_eigenvalues", "negative_eigenvalue_count_fraction", "neg_eigenvalue_fraction"],
    )
    fig, axes = plt.subplots(3, 1, figsize=(9.5, 11.0), sharex=True, constrained_layout=True)

    for ax, column, title, ylabel, fixed_ylim in [
        (
            axes[0],
            "n_negative_eigenvalues",
            "Negative eigenvalue count across layers",
            "Mean count",
            None,
        ),
        (
            axes[1],
            "negative_eigenvalue_count_fraction",
            "Negative eigenvalue count fraction across layers",
            "Mean count fraction",
            (0.0, 1.0),
        ),
        (
            axes[2],
            "neg_eigenvalue_fraction",
            "Negative eigenvalue mass fraction across layers",
            "Mean mass fraction",
            None,
        ),
    ]:
        ax.plot(mean_df["layer"], mean_df[column], color=ORANGE, linewidth=2.8, label="Per-protein model mean")
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
        else:
            ax.set_ylim(padded_limits([mean_df[column].to_numpy()], include_zero=True))
        add_common_axis_style(ax)
        ax.legend(loc="best", frameon=False)
    axes[2].set_xlabel("Layer")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_general_comparison(per_protein: pd.DataFrame, geometry: pd.DataFrame) -> tuple[Path, Path]:
    general_per_protein, general_geometry = load_general_inputs()

    per_r2 = layer_mean(per_protein, [R2_COLUMN]).rename(
        columns={R2_COLUMN: "per_protein_base256_combined_test_mean_r2"}
    )
    per_tm = layer_mean(geometry, ["tm_score"]).rename(
        columns={"tm_score": "per_protein_base256_combined_mean_tm_score"}
    )
    general_r2 = general_test_layer_mean(general_per_protein, ["per_protein_r2"]).rename(
        columns={"per_protein_r2": "general_base256_combined_test_mean_r2"}
    )
    general_tm = general_test_layer_mean(general_geometry, ["tm_score"]).rename(
        columns={"tm_score": "general_base256_combined_test_mean_tm_score"}
    )

    merged = per_r2.merge(per_tm, on="layer").merge(general_r2, on="layer").merge(general_tm, on="layer")
    merged["per_protein_test_r2_minus_general_test"] = (
        merged["per_protein_base256_combined_test_mean_r2"] - merged["general_base256_combined_test_mean_r2"]
    )
    merged["per_protein_tm_minus_general_test"] = (
        merged["per_protein_base256_combined_mean_tm_score"]
        - merged["general_base256_combined_test_mean_tm_score"]
    )

    csv_path = OUT_DIR / f"{PREFIX}_general_base256_combined_comparison_layer_means.csv"
    merged.to_csv(csv_path, index=False, float_format="%.6f")

    plot_path = OUT_DIR / f"{PREFIX}_vs_general_base256_combined_r2_tm_score.png"
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.5), sharex=True, constrained_layout=True)

    axes[0].plot(
        merged["layer"],
        merged["per_protein_base256_combined_test_mean_r2"],
        color=ORANGE,
        linewidth=2.8,
        label="per_protein_base256_combined held-out pair mean",
    )
    axes[0].plot(
        merged["layer"],
        merged["general_base256_combined_test_mean_r2"],
        color=BLUE,
        linewidth=2.4,
        label="general_base256_combined held-out mean",
    )
    axes[0].set_title("Held-out R$^2$: per-protein base256 vs general base256")
    axes[0].set_ylabel("Mean held-out R$^2$")
    axes[0].set_ylim(0.0, 1.0)
    add_common_axis_style(axes[0])
    axes[0].legend(loc="lower right", frameon=False)

    axes[1].plot(
        merged["layer"],
        merged["per_protein_base256_combined_mean_tm_score"],
        color=ORANGE,
        linewidth=2.8,
        label="per_protein_base256_combined mean",
    )
    axes[1].plot(
        merged["layer"],
        merged["general_base256_combined_test_mean_tm_score"],
        color=BLUE,
        linewidth=2.4,
        label="general_base256_combined held-out mean",
    )
    axes[1].set_title("TM-score: per-protein base256 vs general base256")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean TM-score")
    axes[1].set_ylim(0.0, 1.0)
    add_common_axis_style(axes[1])
    axes[1].legend(loc="lower right", frameon=False)

    fig.savefig(plot_path)
    plt.close(fig)
    return csv_path, plot_path


def best_layer(df: pd.DataFrame, column: str, higher_is_better: bool) -> tuple[int, float]:
    series = df.set_index("layer")[column].astype(np.float64)
    layer = int(series.idxmax() if higher_is_better else series.idxmin())
    return layer, float(series.loc[layer])


def write_report(per_protein: pd.DataFrame, geometry: pd.DataFrame, artifact_paths: list[Path]) -> Path:
    r2_mean = layer_mean(per_protein, [R2_COLUMN])
    tm_mean = layer_mean(geometry, ["tm_score"])
    triangle_mean = layer_mean(
        geometry,
        ["triangle_violation_fraction", "triangle_mean_violation", "triangle_violation_10_plus_fraction"],
    )
    top3_mean = layer_mean(geometry, ["frac_top3_eigenvalues"])

    best_r2_layer, best_r2_value = best_layer(r2_mean, R2_COLUMN, True)
    best_tm_layer, best_tm_value = best_layer(tm_mean, "tm_score", True)
    best_triangle_frac_layer, best_triangle_frac_value = best_layer(
        triangle_mean,
        "triangle_violation_fraction",
        False,
    )
    worst_triangle_10plus_layer, worst_triangle_10plus_value = best_layer(
        triangle_mean,
        "triangle_violation_10_plus_fraction",
        True,
    )
    best_top3_layer, best_top3_value = best_layer(top3_mean, "frac_top3_eigenvalues", True)

    source_counts = (
        per_protein[per_protein["layer"] == 0]
        .groupby("source", sort=True)["protein_id"]
        .nunique()
        .to_dict()
    )
    artifact_lines = "\n".join(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in artifact_paths)

    report = f"""# per_protein_base256_combined analysis

Inputs:
- `delta_files/per_protein_base256_combined/per_protein.csv`
- `delta_files/per_protein_base256_combined/geometry_metrics.csv`

Loaded proteins: {int(per_protein["protein_id"].nunique())}
CAMEO proteins: {int(source_counts.get("cameo", 0))}
ECOD proteins: {int(source_counts.get("ecod", 0))}

Layer highlights:
- Best mean held-out pair R^2: layer {best_r2_layer} ({best_r2_value:.4f})
- Best mean TM-score: layer {best_tm_layer} ({best_tm_value:.4f})
- Lowest mean triangle violation fraction: layer {best_triangle_frac_layer} ({best_triangle_frac_value:.4f})
- Highest mean severe (>=10 A) violation fraction: layer {worst_triangle_10plus_layer} ({worst_triangle_10plus_value:.4f})
- Highest mean top-3 eigenvalue fraction: layer {best_top3_layer} ({best_top3_value:.4f})

Triangle inequality metrics (exact):
The triangle columns in `geometry_metrics.csv` / `geometry_summary.csv` are the exact per-triangle metrics sourced from `exact_triangle_metrics/exact_triangle_metrics.csv`, replacing the earlier min-indirect-path triangle fields. For every protein and layer, all C(n_residues, 3) unordered residue triples are enumerated from the reconstructed predicted distance matrix. A triple is a violation when `(longest side) - (sum of the two shorter sides)` exceeds 1e-12 A; that difference is the violation magnitude. Fields:
- `triangle_violation_fraction`: fraction of triples that violate the inequality.
- `triangle_no_violation_fraction`: 1 - `triangle_violation_fraction`.
- `triangle_violation_0_5_fraction` / `_5_10_fraction` / `_10_plus_fraction`: violating triples binned by magnitude in Angstrom; the three bins sum to `triangle_violation_fraction`.
- `triangle_mean_violation`: sum of violation magnitudes divided by all triples (not just violating ones).
- `triangle_max_violation`: largest single violation magnitude.

These are per-protein aggregates; individual `(i, j, k)` triples are not stored.

Comparison caveat:
The comparison plot uses `per_protein_base256_combined` held-out residue-pair R^2 means and `general_base256_combined` held-out/test protein means. The per-protein run fits one model per protein on deterministic 80/20 train pairs, predicts all pairs for geometry, and reports `per_protein_r2` as all-pair R^2 from that train-fit model.

Generated artifacts:
{artifact_lines}
"""

    out_path = OUT_DIR / f"{PREFIX}_analysis_report.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


def assert_outputs(paths: list[Path]) -> None:
    for path in paths:
        require_nonempty(path)

    geometry_means = pd.read_csv(OUT_DIR / f"{PREFIX}_geometry_layer_means.csv")
    if geometry_means.shape[0] != 48:
        raise ValueError("geometry layer means must contain 48 rows")
    if tuple(geometry_means["layer"]) != EXPECTED_LAYERS:
        raise ValueError("geometry layer means must contain layers 0-47")

    comparison = pd.read_csv(OUT_DIR / f"{PREFIX}_general_base256_combined_comparison_layer_means.csv")
    if comparison.shape[0] != 48:
        raise ValueError("general comparison means must contain 48 rows")
    if tuple(comparison["layer"]) != EXPECTED_LAYERS:
        raise ValueError("general comparison means must contain layers 0-47")


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_protein, geometry = load_per_protein_inputs()
    output_paths = [
        plot_r2_tm_with_trajectories(per_protein, geometry),
        plot_r2_tm_mean_only(per_protein, geometry),
        save_geometry_layer_means(geometry),
        plot_triangle_violations(geometry),
        plot_top3_eigenvalue_fraction(geometry),
        plot_negative_eigenvalues(geometry),
    ]

    comparison_csv, comparison_plot = save_general_comparison(per_protein, geometry)
    output_paths.extend([comparison_csv, comparison_plot])
    output_paths.append(write_report(per_protein, geometry, output_paths))

    assert_outputs(output_paths)
    print("Wrote analysis artifacts:")
    for path in output_paths:
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

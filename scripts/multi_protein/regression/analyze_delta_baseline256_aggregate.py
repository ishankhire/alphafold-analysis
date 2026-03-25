#!/usr/bin/env python3
"""
Analyze aggregated baseline256 generalized-model outputs.

Inputs:
  - delta_files/baseline256/
    - per_protein.csv
    - coefficients.csv
    - fold_summary.csv
    - geometry_metrics_last_fold.csv
    - geometry_summary_last_fold.csv

Outputs:
  - delta_files/baseline256/analysis/
"""

from __future__ import annotations

import itertools
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


PROJECT_ROOT = EARLY_PROJECT_ROOT
INPUT_DIR = PROJECT_ROOT / "delta_files" / "baseline256"
OUTPUT_DIR = INPUT_DIR / "analysis"

PER_PROTEIN_CSV = INPUT_DIR / "per_protein.csv"
COEFFICIENTS_CSV = INPUT_DIR / "coefficients.csv"
FOLD_SUMMARY_CSV = INPUT_DIR / "fold_summary.csv"
GEOMETRY_METRICS_CSV = INPUT_DIR / "geometry_metrics_last_fold.csv"
GEOMETRY_SUMMARY_CSV = INPUT_DIR / "geometry_summary_last_fold.csv"

REPORT_MD = OUTPUT_DIR / "baseline256_analysis_report.md"
R2_TM_PLOT = OUTPUT_DIR / "baseline256_r2_tmscore_vs_layer.png"
LAST_FOLD_HEATMAP = OUTPUT_DIR / "baseline256_last_fold_layer_cosine_heatmap.png"
LAYER47_SIZE_PLOT = OUTPUT_DIR / "baseline256_layer47_size_trends.png"
SAME_LAYER_COSINE_CSV = OUTPUT_DIR / "baseline256_same_layer_fold_cosine_similarity.csv"
LAST_FOLD_LAYER_COSINE_CSV = OUTPUT_DIR / "baseline256_last_fold_layer_cosine_matrix.csv"
SUMMARY_STATS_CSV = OUTPUT_DIR / "baseline256_summary_statistics.csv"

EXPECTED_LAYERS = list(range(48))
EXPECTED_FOLDS = list(range(5))
LAST_FOLD = 4
FOLD_PAIRS = list(itertools.combinations(EXPECTED_FOLDS, 2))
FOLD_PAIR_COLUMNS = [f"fold_{fold_a}_vs_{fold_b}" for fold_a, fold_b in FOLD_PAIRS]
FOLD_COLORS = {
    0: "#1f77b4",
    1: "#ff7f0e",
    2: "#2ca02c",
    3: "#d62728",
    4: "#9467bd",
}


def ensure_required_columns(path: Path, df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")


def ensure_exact_values(path: Path, df: pd.DataFrame, column: str, expected: list[int]) -> None:
    found = sorted(int(value) for value in df[column].dropna().unique())
    if found != expected:
        raise ValueError(f"{path.name} expected {column} values {expected}, found {found}")


def ensure_single_value(path: Path, df: pd.DataFrame, column: str, expected: int) -> None:
    found = sorted(int(value) for value in df[column].dropna().unique())
    if found != [expected]:
        raise ValueError(f"{path.name} expected only {column}={expected}, found {found}")


def ensure_no_duplicates(path: Path, df: pd.DataFrame, columns: list[str]) -> None:
    duplicates = int(df.duplicated(columns).sum())
    if duplicates:
        raise ValueError(f"{path.name} has {duplicates} duplicate rows on {columns}")


def ensure_complete_group_count(
    path: Path,
    df: pd.DataFrame,
    group_columns: list[str],
    expected_count: int,
    label: str,
) -> None:
    counts = df.groupby(group_columns, sort=True).size()
    bad = counts[counts != expected_count]
    if not bad.empty:
        sample = ", ".join(
            f"{tuple(index)}={int(count)}" for index, count in list(bad.items())[:5]
        )
        raise ValueError(
            f"{path.name} expected {expected_count} rows per {label}; sample mismatches: {sample}"
        )


def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    norm_a = float(np.linalg.norm(vec_a))
    norm_b = float(np.linalg.norm(vec_b))
    if norm_a <= 0.0 or norm_b <= 0.0:
        return float("nan")
    value = float(np.dot(vec_a, vec_b) / (norm_a * norm_b))
    return float(np.clip(value, -1.0, 1.0))


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = vectors / safe_norms
    matrix = normalized @ normalized.T
    return np.clip(matrix, -1.0, 1.0)


def summarize_distribution(values: pd.Series) -> dict[str, float | int]:
    series = values.astype(np.float64).dropna()
    if series.empty:
        raise ValueError("Cannot summarize an empty distribution")
    q1 = float(series.quantile(0.25))
    q3 = float(series.quantile(0.75))
    return {
        "count": int(series.shape[0]),
        "mean": float(series.mean()),
        "variance": float(series.var(ddof=0)),
        "std": float(series.std(ddof=0)),
        "min": float(series.min()),
        "max": float(series.max()),
        "median": float(series.median()),
        "q1": q1,
        "q3": q3,
        "iqr": q3 - q1,
        "p05": float(series.quantile(0.05)),
        "p95": float(series.quantile(0.95)),
    }


def format_float(value: float) -> str:
    if pd.isna(value) or math.isinf(value):
        return "nan"
    return f"{value:.6f}"


def pearson_corr(xs: pd.Series, ys: pd.Series) -> float:
    if xs.shape[0] != ys.shape[0] or xs.shape[0] < 2:
        return float("nan")
    x = xs.astype(np.float64).to_numpy()
    y = ys.astype(np.float64).to_numpy()
    if float(np.std(x)) <= 0.0 or float(np.std(y)) <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(xs: pd.Series, ys: pd.Series) -> float:
    if xs.shape[0] != ys.shape[0] or xs.shape[0] < 2:
        return float("nan")
    return float(xs.astype(np.float64).corr(ys.astype(np.float64), method="spearman"))


def fit_line(xs: pd.Series, ys: pd.Series) -> tuple[np.ndarray, np.ndarray]:
    x = xs.astype(np.float64).to_numpy()
    y = ys.astype(np.float64).to_numpy()
    if x.shape[0] < 2 or float(np.std(x)) <= 0.0:
        return np.asarray([]), np.asarray([])
    slope, intercept = np.polyfit(x, y, deg=1)
    x_line = np.linspace(float(x.min()), float(x.max()), num=100)
    y_line = slope * x_line + intercept
    return x_line, y_line

def write_report(
    source_files: list[Path],
    r2_stats: dict[str, float | int],
    r2_layer_stats: dict[str, float | int],
    tm_stats: dict[str, float | int],
    tm_layer_stats: dict[str, float | int],
    r2_best_layer: tuple[int, float],
    r2_worst_layer: tuple[int, float],
    tm_best_layer: tuple[int, float],
    tm_worst_layer: tuple[int, float],
    cosine_df: pd.DataFrame,
    protein_count: int,
    layer_count: int,
    fold_count: int,
) -> None:
    most_stable = cosine_df.nlargest(3, "mean_pairwise_cosine")[
        ["layer", "mean_pairwise_cosine"]
    ]
    least_stable = cosine_df.nsmallest(3, "mean_pairwise_cosine")[
        ["layer", "mean_pairwise_cosine"]
    ]

    lines: list[str] = []
    lines.append("# Baseline256 Aggregate Analysis Report")
    lines.append("")
    lines.append("## Dataset overview")
    lines.append("")
    lines.append("Source files used:")
    for path in source_files:
        lines.append(f"- `{path.relative_to(PROJECT_ROOT)}`")
    lines.append(f"- proteins: {protein_count}")
    lines.append(f"- layers: {layer_count}")
    lines.append(f"- folds: {fold_count}")
    lines.append("- TM-score is available only for fold 4.")
    lines.append("")
    lines.append("## R² summary statistics")
    lines.append("")
    lines.append(
        "Protein-level trajectories are built by averaging each protein's "
        "`per_protein_r2` across its five fold values at each layer."
    )
    lines.append("")
    lines.append("| Distribution | count | mean | variance | std | min | max | median | Q1 | Q3 | IQR | p05 | p95 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        "| Fold-averaged protein×layer R² | "
        f"{int(r2_stats['count'])} | {format_float(r2_stats['mean'])} | {format_float(r2_stats['variance'])} | "
        f"{format_float(r2_stats['std'])} | {format_float(r2_stats['min'])} | {format_float(r2_stats['max'])} | "
        f"{format_float(r2_stats['median'])} | {format_float(r2_stats['q1'])} | {format_float(r2_stats['q3'])} | "
        f"{format_float(r2_stats['iqr'])} | {format_float(r2_stats['p05'])} | {format_float(r2_stats['p95'])} |"
    )
    lines.append(
        "| Per-layer mean R² across 48 layers | "
        f"{int(r2_layer_stats['count'])} | {format_float(r2_layer_stats['mean'])} | {format_float(r2_layer_stats['variance'])} | "
        f"{format_float(r2_layer_stats['std'])} | {format_float(r2_layer_stats['min'])} | {format_float(r2_layer_stats['max'])} | "
        f"{format_float(r2_layer_stats['median'])} | {format_float(r2_layer_stats['q1'])} | {format_float(r2_layer_stats['q3'])} | "
        f"{format_float(r2_layer_stats['iqr'])} | {format_float(r2_layer_stats['p05'])} | {format_float(r2_layer_stats['p95'])} |"
    )
    lines.append("")
    lines.append(
        f"Best layer by mean R²: layer {r2_best_layer[0]} ({format_float(r2_best_layer[1])})."
    )
    lines.append(
        f"Worst layer by mean R²: layer {r2_worst_layer[0]} ({format_float(r2_worst_layer[1])})."
    )
    lines.append("")
    lines.append("## TM-score summary statistics")
    lines.append("")
    lines.append("TM-score statistics use fold-4 values from `geometry_metrics_last_fold.csv`.")
    lines.append("")
    lines.append("| Distribution | count | mean | variance | std | min | max | median | Q1 | Q3 | IQR | p05 | p95 |")
    lines.append("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    lines.append(
        "| Fold-4 protein×layer TM-score | "
        f"{int(tm_stats['count'])} | {format_float(tm_stats['mean'])} | {format_float(tm_stats['variance'])} | "
        f"{format_float(tm_stats['std'])} | {format_float(tm_stats['min'])} | {format_float(tm_stats['max'])} | "
        f"{format_float(tm_stats['median'])} | {format_float(tm_stats['q1'])} | {format_float(tm_stats['q3'])} | "
        f"{format_float(tm_stats['iqr'])} | {format_float(tm_stats['p05'])} | {format_float(tm_stats['p95'])} |"
    )
    lines.append(
        "| Per-layer mean TM-score across 48 layers | "
        f"{int(tm_layer_stats['count'])} | {format_float(tm_layer_stats['mean'])} | {format_float(tm_layer_stats['variance'])} | "
        f"{format_float(tm_layer_stats['std'])} | {format_float(tm_layer_stats['min'])} | {format_float(tm_layer_stats['max'])} | "
        f"{format_float(tm_layer_stats['median'])} | {format_float(tm_layer_stats['q1'])} | {format_float(tm_layer_stats['q3'])} | "
        f"{format_float(tm_layer_stats['iqr'])} | {format_float(tm_layer_stats['p05'])} | {format_float(tm_layer_stats['p95'])} |"
    )
    lines.append("")
    lines.append(
        f"Best layer by mean TM-score: layer {tm_best_layer[0]} ({format_float(tm_best_layer[1])})."
    )
    lines.append(
        f"Worst layer by mean TM-score: layer {tm_worst_layer[0]} ({format_float(tm_worst_layer[1])})."
    )
    lines.append("")
    lines.append("## Same-layer fold cosine similarity")
    lines.append("")
    lines.append(
        "Cosine similarity uses raw-scale feature coefficients with the intercept removed."
    )
    lines.append("")
    lines.append(
        f"Most cross-fold stable layers: "
        + ", ".join(
            f"layer {int(row.layer)} ({format_float(row.mean_pairwise_cosine)})"
            for row in most_stable.itertuples(index=False)
        )
        + "."
    )
    lines.append(
        f"Least cross-fold stable layers: "
        + ", ".join(
            f"layer {int(row.layer)} ({format_float(row.mean_pairwise_cosine)})"
            for row in least_stable.itertuples(index=False)
        )
        + "."
    )
    lines.append("")
    lines.append(
        "| layer | "
        + " | ".join(FOLD_PAIR_COLUMNS)
        + " | mean_pairwise_cosine | min_pairwise_cosine | max_pairwise_cosine |"
    )
    lines.append(
        "| ---: | "
        + " | ".join(["---:"] * len(FOLD_PAIR_COLUMNS))
        + " | ---: | ---: | ---: |"
    )
    for row in cosine_df.itertuples(index=False):
        pair_values = " | ".join(format_float(float(getattr(row, column))) for column in FOLD_PAIR_COLUMNS)
        lines.append(
            f"| {int(row.layer)} | {pair_values} | "
            f"{format_float(float(row.mean_pairwise_cosine))} | "
            f"{format_float(float(row.min_pairwise_cosine))} | "
            f"{format_float(float(row.max_pairwise_cosine))} |"
        )
    REPORT_MD.write_text("\n".join(lines) + "\n")


def plot_r2_and_tm(
    r2_protein_layer: pd.DataFrame,
    fold_mean_r2: pd.DataFrame,
    tm_protein_layer: pd.DataFrame,
    tm_layer_mean: pd.Series,
) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    r2_ax = axes[0]
    for _, group in r2_protein_layer.groupby("protein_id", sort=True):
        r2_ax.plot(group["layer"], group["per_protein_r2"], color="#d0d0d0", linewidth=0.6, alpha=0.7)
    for fold in EXPECTED_FOLDS:
        fold_group = fold_mean_r2[fold_mean_r2["fold"] == fold]
        r2_ax.plot(
            fold_group["layer"],
            fold_group["per_protein_r2"],
            color=FOLD_COLORS[fold],
            linewidth=1.7,
            label=f"Fold {fold} mean",
        )
    overall_mean = (
        fold_mean_r2.groupby("layer", sort=True)["per_protein_r2"].mean().reset_index()
    )
    r2_ax.plot(
        overall_mean["layer"],
        overall_mean["per_protein_r2"],
        color="black",
        linewidth=2.4,
        label="Mean of fold means",
    )
    r2_ax.set_ylabel("Per-protein R²")
    r2_ax.set_title("Baseline256 aggregate R² across layers")
    r2_ax.grid(True, alpha=0.25)
    r2_ax.legend(ncol=3, fontsize=9)

    tm_ax = axes[1]
    for _, group in tm_protein_layer.groupby("protein_id", sort=True):
        tm_ax.plot(group["layer"], group["tm_score"], color="#d0d0d0", linewidth=0.6, alpha=0.7)
    tm_ax.plot(
        tm_layer_mean.index,
        tm_layer_mean.values,
        color="#b85c38",
        linewidth=1.8,
        label="Fold 4 mean",
    )
    tm_ax.plot(
        tm_layer_mean.index,
        tm_layer_mean.values,
        color="black",
        linewidth=2.4,
        label="Overall mean (fold 4 only)",
    )
    tm_ax.set_xlabel("Layer")
    tm_ax.set_ylabel("TM-score")
    tm_ax.set_title("Baseline256 fold-4 TM-score across layers")
    tm_ax.grid(True, alpha=0.25)
    tm_ax.legend(fontsize=9)

    axes[1].set_xticks(EXPECTED_LAYERS[::2])
    fig.savefig(R2_TM_PLOT, dpi=200)
    plt.close(fig)


def plot_last_fold_heatmap(layer_cosine_matrix: np.ndarray) -> None:
    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(layer_cosine_matrix, origin="lower", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("Fold-4 layer coefficient cosine similarity")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ticks = np.arange(0, 48, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(int(tick)) for tick in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([str(int(tick)) for tick in ticks], fontsize=7)
    colorbar = fig.colorbar(image, ax=ax)
    colorbar.set_label("Cosine similarity")
    fig.savefig(LAST_FOLD_HEATMAP, dpi=200)
    plt.close(fig)


def add_trend_panel(
    ax: plt.Axes,
    df: pd.DataFrame,
    y_column: str,
    color: str,
    title: str,
    y_label: str,
) -> None:
    ax.scatter(df["n_residues"], df[y_column], color=color, alpha=0.7, edgecolor="none")
    x_line, y_line = fit_line(df["n_residues"], df[y_column])
    if x_line.size:
        ax.plot(x_line, y_line, color="black", linewidth=2.0)
    pearson = pearson_corr(df["n_residues"], df[y_column])
    spearman = spearman_corr(df["n_residues"], df[y_column])
    ax.text(
        0.03,
        0.97,
        "Positive correlation => smaller proteins tend to do worse\n"
        f"Pearson r = {format_float(pearson)}\n"
        f"Spearman rho = {format_float(spearman)}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=9,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.9, "edgecolor": "#999999"},
    )
    ax.set_title(title)
    ax.set_xlabel("n_residues")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)


def plot_layer47_size_trends(
    r2_layer47: pd.DataFrame,
    tm_layer47: pd.DataFrame,
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), constrained_layout=True)
    add_trend_panel(
        axes[0],
        r2_layer47,
        "per_protein_r2",
        "#2b7a78",
        "Layer-47 fold-averaged R² vs size",
        "Per-protein R²",
    )
    add_trend_panel(
        axes[1],
        tm_layer47,
        "tm_score",
        "#c06c2b",
        "Layer-47 fold-4 TM-score vs size",
        "TM-score",
    )
    fig.savefig(LAYER47_SIZE_PLOT, dpi=200)
    plt.close(fig)


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_protein = pd.read_csv(PER_PROTEIN_CSV)
    coefficients = pd.read_csv(COEFFICIENTS_CSV)
    fold_summary = pd.read_csv(FOLD_SUMMARY_CSV)
    geometry_metrics = pd.read_csv(GEOMETRY_METRICS_CSV)
    geometry_summary = pd.read_csv(GEOMETRY_SUMMARY_CSV)
    return per_protein, coefficients, fold_summary, geometry_metrics, geometry_summary


def validate_inputs(
    per_protein: pd.DataFrame,
    coefficients: pd.DataFrame,
    fold_summary: pd.DataFrame,
    geometry_metrics: pd.DataFrame,
    geometry_summary: pd.DataFrame,
) -> None:
    ensure_required_columns(
        PER_PROTEIN_CSV,
        per_protein,
        {
            "fold",
            "layer",
            "split",
            "protein_id",
            "n_residues",
            "per_protein_r2",
        },
    )
    ensure_required_columns(
        COEFFICIENTS_CSV,
        coefficients,
        {
            "fold",
            "layer",
            "feature_index",
            "feature_name",
            "coefficient_raw_scale",
        },
    )
    ensure_required_columns(
        FOLD_SUMMARY_CSV,
        fold_summary,
        {
            "fold",
            "layer",
            "test_mean_per_protein_r2",
        },
    )
    ensure_required_columns(
        GEOMETRY_METRICS_CSV,
        geometry_metrics,
        {
            "fold",
            "layer",
            "split",
            "protein_id",
            "n_residues",
            "tm_score",
        },
    )
    ensure_required_columns(
        GEOMETRY_SUMMARY_CSV,
        geometry_summary,
        {
            "fold",
            "layer",
            "split",
            "tm_score_mean",
        },
    )

    ensure_exact_values(PER_PROTEIN_CSV, per_protein, "layer", EXPECTED_LAYERS)
    ensure_exact_values(COEFFICIENTS_CSV, coefficients, "layer", EXPECTED_LAYERS)
    ensure_exact_values(FOLD_SUMMARY_CSV, fold_summary, "layer", EXPECTED_LAYERS)
    ensure_exact_values(GEOMETRY_METRICS_CSV, geometry_metrics, "layer", EXPECTED_LAYERS)
    ensure_exact_values(GEOMETRY_SUMMARY_CSV, geometry_summary, "layer", EXPECTED_LAYERS)

    ensure_exact_values(PER_PROTEIN_CSV, per_protein, "fold", EXPECTED_FOLDS)
    ensure_exact_values(COEFFICIENTS_CSV, coefficients, "fold", EXPECTED_FOLDS)
    ensure_exact_values(FOLD_SUMMARY_CSV, fold_summary, "fold", EXPECTED_FOLDS)
    ensure_single_value(GEOMETRY_METRICS_CSV, geometry_metrics, "fold", LAST_FOLD)
    ensure_single_value(GEOMETRY_SUMMARY_CSV, geometry_summary, "fold", LAST_FOLD)

    ensure_no_duplicates(PER_PROTEIN_CSV, per_protein, ["fold", "layer", "protein_id", "split"])
    ensure_no_duplicates(GEOMETRY_METRICS_CSV, geometry_metrics, ["fold", "layer", "protein_id", "split"])
    ensure_no_duplicates(FOLD_SUMMARY_CSV, fold_summary, ["fold", "layer"])
    ensure_no_duplicates(GEOMETRY_SUMMARY_CSV, geometry_summary, ["fold", "layer", "split"])

    ensure_complete_group_count(
        COEFFICIENTS_CSV,
        coefficients,
        ["fold", "layer"],
        257,
        "(fold, layer)",
    )
    non_intercept = coefficients[coefficients["feature_index"] != -1]
    ensure_complete_group_count(
        COEFFICIENTS_CSV,
        non_intercept,
        ["fold", "layer"],
        256,
        "non-intercept (fold, layer)",
    )
    intercept_counts = coefficients[coefficients["feature_index"] == -1].groupby(["fold", "layer"]).size()
    if not (intercept_counts == 1).all():
        raise ValueError("coefficients.csv must contain exactly one intercept row per (fold, layer)")
    unique_feature_counts = non_intercept.groupby(["fold", "layer"])["feature_index"].nunique()
    if not (unique_feature_counts == 256).all():
        raise ValueError("coefficients.csv must contain one complete 256-value feature vector per (fold, layer)")


def build_r2_tables(per_protein: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.DataFrame]:
    r2_data = per_protein[
        ["fold", "layer", "protein_id", "n_residues", "per_protein_r2"]
    ].copy()
    per_fold_counts = r2_data.groupby(["protein_id", "layer"], sort=True).size()
    if not (per_fold_counts == len(EXPECTED_FOLDS)).all():
        raise ValueError("Expected exactly five R² rows per (protein_id, layer) after combining train/test")

    protein_layer = (
        r2_data.groupby(["protein_id", "layer"], sort=True)
        .agg(n_residues=("n_residues", "first"), per_protein_r2=("per_protein_r2", "mean"))
        .reset_index()
    )
    layer_mean = protein_layer.groupby("layer", sort=True)["per_protein_r2"].mean()
    fold_mean = (
        r2_data.groupby(["fold", "layer"], sort=True)["per_protein_r2"].mean().reset_index()
    )
    layer47 = protein_layer[protein_layer["layer"] == 47].copy()
    return protein_layer, fold_mean, layer_mean, layer47


def build_tm_tables(geometry_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    tm_data = geometry_metrics[
        ["fold", "layer", "protein_id", "n_residues", "tm_score"]
    ].copy()
    protein_layer_counts = tm_data.groupby(["protein_id", "layer"], sort=True).size()
    if not (protein_layer_counts == 1).all():
        raise ValueError("Expected exactly one fold-4 TM-score row per (protein_id, layer)")

    protein_layer = (
        tm_data.groupby(["protein_id", "layer"], sort=True)
        .agg(n_residues=("n_residues", "first"), tm_score=("tm_score", "first"))
        .reset_index()
    )
    layer_mean = protein_layer.groupby("layer", sort=True)["tm_score"].mean()
    layer47 = protein_layer[protein_layer["layer"] == 47].copy()
    return protein_layer, layer_mean, layer47


def build_same_layer_cosines(coefficients: pd.DataFrame) -> tuple[pd.DataFrame, np.ndarray]:
    raw_coefficients = coefficients[coefficients["feature_index"] != -1].copy()
    raw_coefficients = raw_coefficients.sort_values(["fold", "layer", "feature_index"])

    vectors: dict[tuple[int, int], np.ndarray] = {}
    for (fold, layer), group in raw_coefficients.groupby(["fold", "layer"], sort=True):
        vectors[(int(fold), int(layer))] = group["coefficient_raw_scale"].astype(np.float64).to_numpy()

    same_layer_rows: list[dict[str, float]] = []
    for layer in EXPECTED_LAYERS:
        row: dict[str, float] = {"layer": float(layer)}
        pair_values: list[float] = []
        for fold_a, fold_b in FOLD_PAIRS:
            cosine = cosine_similarity(vectors[(fold_a, layer)], vectors[(fold_b, layer)])
            row[f"fold_{fold_a}_vs_{fold_b}"] = cosine
            pair_values.append(cosine)
        row["mean_pairwise_cosine"] = float(np.mean(pair_values))
        row["min_pairwise_cosine"] = float(np.min(pair_values))
        row["max_pairwise_cosine"] = float(np.max(pair_values))
        same_layer_rows.append(row)

    same_layer_df = pd.DataFrame(same_layer_rows)
    ordered_columns = ["layer", *FOLD_PAIR_COLUMNS, "mean_pairwise_cosine", "min_pairwise_cosine", "max_pairwise_cosine"]
    same_layer_df = same_layer_df[ordered_columns]

    fold4_vectors = np.vstack([vectors[(LAST_FOLD, layer)] for layer in EXPECTED_LAYERS])
    layer_matrix = cosine_similarity_matrix(fold4_vectors)
    return same_layer_df, layer_matrix


def build_summary_stats_rows(
    r2_stats: dict[str, float | int],
    r2_layer_stats: dict[str, float | int],
    tm_stats: dict[str, float | int],
    tm_layer_stats: dict[str, float | int],
    r2_best_layer: tuple[int, float],
    r2_worst_layer: tuple[int, float],
    tm_best_layer: tuple[int, float],
    tm_worst_layer: tuple[int, float],
) -> pd.DataFrame:
    rows = [
        {
            "metric": "r2",
            "distribution": "fold_averaged_protein_layer",
            **r2_stats,
            "best_layer": r2_best_layer[0],
            "best_layer_mean": r2_best_layer[1],
            "worst_layer": r2_worst_layer[0],
            "worst_layer_mean": r2_worst_layer[1],
            "notes": "Per-protein R² averaged across five folds at each layer.",
        },
        {
            "metric": "r2",
            "distribution": "layer_mean",
            **r2_layer_stats,
            "best_layer": r2_best_layer[0],
            "best_layer_mean": r2_best_layer[1],
            "worst_layer": r2_worst_layer[0],
            "worst_layer_mean": r2_worst_layer[1],
            "notes": "Distribution over 48 layer means after fold-averaging per protein.",
        },
        {
            "metric": "tm_score",
            "distribution": "fold4_protein_layer",
            **tm_stats,
            "best_layer": tm_best_layer[0],
            "best_layer_mean": tm_best_layer[1],
            "worst_layer": tm_worst_layer[0],
            "worst_layer_mean": tm_worst_layer[1],
            "notes": "Fold-4 TM-score values from geometry_metrics_last_fold.csv.",
        },
        {
            "metric": "tm_score",
            "distribution": "layer_mean",
            **tm_layer_stats,
            "best_layer": tm_best_layer[0],
            "best_layer_mean": tm_best_layer[1],
            "worst_layer": tm_worst_layer[0],
            "worst_layer_mean": tm_worst_layer[1],
            "notes": "Distribution over 48 fold-4 TM-score layer means.",
        },
    ]
    return pd.DataFrame(rows)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    per_protein, coefficients, fold_summary, geometry_metrics, geometry_summary = load_inputs()
    validate_inputs(per_protein, coefficients, fold_summary, geometry_metrics, geometry_summary)

    r2_protein_layer, fold_mean_r2, r2_layer_mean, r2_layer47 = build_r2_tables(per_protein)
    tm_protein_layer, tm_layer_mean, tm_layer47 = build_tm_tables(geometry_metrics)
    same_layer_cosine_df, last_fold_layer_cosine = build_same_layer_cosines(coefficients)

    r2_stats = summarize_distribution(r2_protein_layer["per_protein_r2"])
    r2_layer_stats = summarize_distribution(r2_layer_mean)
    tm_stats = summarize_distribution(tm_protein_layer["tm_score"])
    tm_layer_stats = summarize_distribution(tm_layer_mean)

    r2_best_layer = (
        int(r2_layer_mean.idxmax()),
        float(r2_layer_mean.loc[r2_layer_mean.idxmax()]),
    )
    r2_worst_layer = (
        int(r2_layer_mean.idxmin()),
        float(r2_layer_mean.loc[r2_layer_mean.idxmin()]),
    )
    tm_best_layer = (
        int(tm_layer_mean.idxmax()),
        float(tm_layer_mean.loc[tm_layer_mean.idxmax()]),
    )
    tm_worst_layer = (
        int(tm_layer_mean.idxmin()),
        float(tm_layer_mean.loc[tm_layer_mean.idxmin()]),
    )

    summary_stats_df = build_summary_stats_rows(
        r2_stats,
        r2_layer_stats,
        tm_stats,
        tm_layer_stats,
        r2_best_layer,
        r2_worst_layer,
        tm_best_layer,
        tm_worst_layer,
    )
    summary_stats_df.to_csv(SUMMARY_STATS_CSV, index=False, float_format="%.6f")

    cosine_output = same_layer_cosine_df.copy()
    cosine_output["layer"] = cosine_output["layer"].astype(int)
    cosine_output.to_csv(SAME_LAYER_COSINE_CSV, index=False, float_format="%.6f")

    matrix_df = pd.DataFrame(
        last_fold_layer_cosine,
        columns=[f"layer_{layer}" for layer in EXPECTED_LAYERS],
    )
    matrix_df.insert(0, "layer", EXPECTED_LAYERS)
    matrix_df.to_csv(LAST_FOLD_LAYER_COSINE_CSV, index=False, float_format="%.6f")

    plot_r2_and_tm(r2_protein_layer, fold_mean_r2, tm_protein_layer, tm_layer_mean)
    plot_last_fold_heatmap(last_fold_layer_cosine)
    plot_layer47_size_trends(r2_layer47, tm_layer47)

    write_report(
        source_files=[
            PER_PROTEIN_CSV,
            COEFFICIENTS_CSV,
            FOLD_SUMMARY_CSV,
            GEOMETRY_METRICS_CSV,
            GEOMETRY_SUMMARY_CSV,
        ],
        r2_stats=r2_stats,
        r2_layer_stats=r2_layer_stats,
        tm_stats=tm_stats,
        tm_layer_stats=tm_layer_stats,
        r2_best_layer=r2_best_layer,
        r2_worst_layer=r2_worst_layer,
        tm_best_layer=tm_best_layer,
        tm_worst_layer=tm_worst_layer,
        cosine_df=cosine_output,
        protein_count=int(r2_protein_layer["protein_id"].nunique()),
        layer_count=int(r2_protein_layer["layer"].nunique()),
        fold_count=len(EXPECTED_FOLDS),
    )


if __name__ == "__main__":
    main()

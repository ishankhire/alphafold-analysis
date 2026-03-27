#!/usr/bin/env python3
"""
Analyze baseline256/general_base256 Delta outputs with one fixed-path script.

Per-variant outputs:
  - delta_files/<variant>/analysis/

Cross-variant outputs:
  - delta_files/comparing_variants/
"""

from __future__ import annotations

import itertools
import math
import os
from dataclasses import dataclass
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
DELTA_DIR = PROJECT_ROOT / "delta_files"
COMPARE_DIR = DELTA_DIR / "comparing_variants"

EXPECTED_LAYERS = list(range(48))
EXPECTED_FOLDS = list(range(5))
LAST_FOLD = 4
FOLD_PAIRS = list(itertools.combinations(EXPECTED_FOLDS, 2))
FOLD_PAIR_COLUMNS = [f"fold_{fold_a}_vs_{fold_b}" for fold_a, fold_b in FOLD_PAIRS]

FOLD_COLORS = {
    0: "#4c78a8",
    1: "#f58518",
    2: "#54a24b",
    3: "#e45756",
    4: "#9d755d",
}
SPLIT_COLORS = {
    "train": "#4c78a8",
    "test": "#e45756",
}
VARIANT_COLORS = {
    "baseline256_kfold": "#4c78a8",
    "general_base256_avg_symmetric": "#f58518",
    "general_base256_plus_diagonal": "#54a24b",
    "general_base256_avg_symmetric_plus_axial": "#e45756",
}


@dataclass(frozen=True)
class VariantSpec:
    folder: str
    label: str
    feature_dim: int
    replicate_col: str
    per_protein_replicates: tuple[int, ...]
    coefficient_replicates: tuple[int, ...]
    geometry_replicates: tuple[int, ...]
    geometry_metrics_name: str
    baseline: bool = False

    @property
    def input_dir(self) -> Path:
        return DELTA_DIR / self.folder

    @property
    def analysis_dir(self) -> Path:
        return self.input_dir / "analysis"

    @property
    def prefix(self) -> str:
        return self.folder


@dataclass
class ProcessedVariant:
    spec: VariantSpec
    r2_protein_layer: pd.DataFrame
    r2_replicate_mean: pd.DataFrame
    r2_split_mean: pd.DataFrame
    r2_layer_mean: pd.Series
    geometry_protein_layer: pd.DataFrame
    geometry_split_mean: pd.DataFrame
    geometry_layer_mean: pd.DataFrame
    layer_cosine_matrix: np.ndarray
    same_layer_fold_cosine: pd.DataFrame | None
    summary_stats: pd.DataFrame


VARIANT_SPECS = [
    VariantSpec(
        folder="baseline256_kfold",
        label="baseline256_kfold",
        feature_dim=256,
        replicate_col="fold",
        per_protein_replicates=(0, 1, 2, 3, 4),
        coefficient_replicates=(0, 1, 2, 3, 4),
        geometry_replicates=(4,),
        geometry_metrics_name="geometry_metrics_last_fold.csv",
        baseline=True,
    ),
    VariantSpec(
        folder="general_base256_avg_symmetric",
        label="general_base256_avg_symmetric",
        feature_dim=128,
        replicate_col="split_id",
        per_protein_replicates=(0,),
        coefficient_replicates=(0,),
        geometry_replicates=(0,),
        geometry_metrics_name="geometry_metrics.csv",
    ),
    VariantSpec(
        folder="general_base256_plus_diagonal",
        label="general_base256_plus_diagonal",
        feature_dim=512,
        replicate_col="split_id",
        per_protein_replicates=(0,),
        coefficient_replicates=(0,),
        geometry_replicates=(0,),
        geometry_metrics_name="geometry_metrics.csv",
    ),
    VariantSpec(
        folder="general_base256_avg_symmetric_plus_axial",
        label="general_base256_avg_symmetric_plus_axial",
        feature_dim=640,
        replicate_col="split_id",
        per_protein_replicates=(0,),
        coefficient_replicates=(0,),
        geometry_replicates=(0,),
        geometry_metrics_name="geometry_metrics.csv",
    ),
]


def ensure_required_columns(path: Path, df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")


def ensure_exact_values(path: Path, df: pd.DataFrame, column: str, expected: tuple[int, ...]) -> None:
    found = tuple(sorted(int(value) for value in df[column].dropna().unique()))
    if found != expected:
        raise ValueError(f"{path.name} expected {column} values {list(expected)}, found {list(found)}")


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


def metric_direction(metric: str) -> str:
    return "higher_is_better" if metric in {"r2", "tm_score"} else "lower_is_better"


def best_worst_layers(layer_mean: pd.Series, higher_is_better: bool) -> tuple[tuple[int, float], tuple[int, float]]:
    if higher_is_better:
        best_idx = int(layer_mean.idxmax())
        worst_idx = int(layer_mean.idxmin())
    else:
        best_idx = int(layer_mean.idxmin())
        worst_idx = int(layer_mean.idxmax())
    return (best_idx, float(layer_mean.loc[best_idx])), (worst_idx, float(layer_mean.loc[worst_idx]))


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


def padded_limits(values: list[np.ndarray]) -> tuple[float, float]:
    flat = np.concatenate([arr.astype(np.float64).ravel() for arr in values if arr.size])
    flat = flat[np.isfinite(flat)]
    if flat.size == 0:
        return (0.0, 1.0)
    ymin = float(flat.min())
    ymax = float(flat.max())
    if math.isclose(ymin, ymax):
        pad = 0.05 if math.isclose(ymin, 0.0) else abs(ymin) * 0.05
        return (ymin - pad, ymax + pad)
    pad = (ymax - ymin) * 0.05
    return (ymin - pad, ymax + pad)


def load_inputs(spec: VariantSpec) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_protein = pd.read_csv(spec.input_dir / "per_protein.csv")
    coefficients = pd.read_csv(spec.input_dir / "coefficients.csv")
    geometry_metrics = pd.read_csv(spec.input_dir / spec.geometry_metrics_name)
    return per_protein, coefficients, geometry_metrics


def validate_inputs(
    spec: VariantSpec,
    per_protein: pd.DataFrame,
    coefficients: pd.DataFrame,
    geometry_metrics: pd.DataFrame,
) -> None:
    per_protein_path = spec.input_dir / "per_protein.csv"
    coefficients_path = spec.input_dir / "coefficients.csv"
    geometry_path = spec.input_dir / spec.geometry_metrics_name

    ensure_required_columns(
        per_protein_path,
        per_protein,
        {spec.replicate_col, "layer", "split", "protein_id", "n_residues", "per_protein_r2"},
    )
    ensure_required_columns(
        coefficients_path,
        coefficients,
        {spec.replicate_col, "layer", "feature_index", "feature_name", "coefficient_raw_scale"},
    )
    ensure_required_columns(
        geometry_path,
        geometry_metrics,
        {
            spec.replicate_col,
            "layer",
            "split",
            "protein_id",
            "n_residues",
            "tm_score",
            "triangle_violation_fraction",
            "triangle_mean_violation",
        },
    )

    ensure_exact_values(per_protein_path, per_protein, "layer", tuple(EXPECTED_LAYERS))
    ensure_exact_values(coefficients_path, coefficients, "layer", tuple(EXPECTED_LAYERS))
    ensure_exact_values(geometry_path, geometry_metrics, "layer", tuple(EXPECTED_LAYERS))

    ensure_exact_values(per_protein_path, per_protein, spec.replicate_col, spec.per_protein_replicates)
    ensure_exact_values(coefficients_path, coefficients, spec.replicate_col, spec.coefficient_replicates)
    ensure_exact_values(geometry_path, geometry_metrics, spec.replicate_col, spec.geometry_replicates)

    ensure_no_duplicates(
        per_protein_path,
        per_protein,
        [spec.replicate_col, "layer", "split", "protein_id"],
    )
    ensure_no_duplicates(
        coefficients_path,
        coefficients,
        [spec.replicate_col, "layer", "feature_index"],
    )
    ensure_no_duplicates(
        geometry_path,
        geometry_metrics,
        [spec.replicate_col, "layer", "split", "protein_id"],
    )

    ensure_complete_group_count(
        coefficients_path,
        coefficients,
        [spec.replicate_col, "layer"],
        spec.feature_dim + 1,
        f"({spec.replicate_col}, layer)",
    )
    non_intercept = coefficients[coefficients["feature_index"] != -1]
    ensure_complete_group_count(
        coefficients_path,
        non_intercept,
        [spec.replicate_col, "layer"],
        spec.feature_dim,
        f"non-intercept ({spec.replicate_col}, layer)",
    )

    intercept_counts = coefficients[coefficients["feature_index"] == -1].groupby(
        [spec.replicate_col, "layer"], sort=True
    ).size()
    if not (intercept_counts == 1).all():
        raise ValueError(f"{coefficients_path.name} must contain one intercept row per layer")


def build_r2_tables(spec: VariantSpec, per_protein: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series]:
    r2_split_mean = (
        per_protein.groupby(["split", "layer"], sort=True)["per_protein_r2"].mean().reset_index()
    )
    r2_replicate_mean = (
        per_protein.groupby([spec.replicate_col, "layer"], sort=True)["per_protein_r2"].mean().reset_index()
    )

    if spec.baseline:
        counts = per_protein.groupby(["protein_id", "layer"], sort=True).size()
        if not (counts == len(spec.per_protein_replicates)).all():
            raise ValueError("Baseline per_protein.csv must have five rows per (protein_id, layer)")
        r2_protein_layer = (
            per_protein.groupby(["protein_id", "layer"], sort=True)
            .agg(n_residues=("n_residues", "first"), per_protein_r2=("per_protein_r2", "mean"))
            .reset_index()
        )
    else:
        counts = per_protein.groupby(["protein_id", "layer"], sort=True).size()
        if not (counts == 1).all():
            raise ValueError(f"{spec.folder} per_protein.csv must have one row per (protein_id, layer)")
        r2_protein_layer = per_protein[["protein_id", "layer", "n_residues", "per_protein_r2"]].copy()

    r2_layer_mean = r2_protein_layer.groupby("layer", sort=True)["per_protein_r2"].mean()
    return r2_protein_layer, r2_replicate_mean, r2_split_mean, r2_layer_mean


def build_geometry_tables(geometry_metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    value_columns = ["tm_score", "triangle_violation_fraction", "triangle_mean_violation"]
    counts = geometry_metrics.groupby(["protein_id", "layer"], sort=True).size()
    if not (counts == 1).all():
        raise ValueError("geometry metrics must have one row per (protein_id, layer)")

    geometry_protein_layer = geometry_metrics[["protein_id", "layer", "n_residues", *value_columns]].copy()
    geometry_split_mean = (
        geometry_metrics.groupby(["split", "layer"], sort=True)[value_columns].mean().reset_index()
    )
    geometry_layer_mean = (
        geometry_protein_layer.groupby("layer", sort=True)[value_columns].mean().reset_index()
    )
    return geometry_protein_layer, geometry_split_mean, geometry_layer_mean


def build_layer_cosine_matrix(spec: VariantSpec, coefficients: pd.DataFrame) -> np.ndarray:
    replicate_value = spec.geometry_replicates[0]
    filtered = coefficients[
        (coefficients[spec.replicate_col] == replicate_value) & (coefficients["feature_index"] != -1)
    ].copy()
    filtered = filtered.sort_values(["layer", "feature_index"])
    vectors = []
    for layer, group in filtered.groupby("layer", sort=True):
        if int(layer) not in EXPECTED_LAYERS:
            raise ValueError(f"Unexpected layer {layer} in coefficients")
        vector = group["coefficient_raw_scale"].astype(np.float64).to_numpy()
        if vector.shape[0] != spec.feature_dim:
            raise ValueError(f"{spec.folder} layer {layer} expected {spec.feature_dim} coefficients")
        vectors.append(vector)
    matrix = cosine_similarity_matrix(np.vstack(vectors))
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError(f"{spec.folder} layer cosine matrix is not symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-6):
        raise ValueError(f"{spec.folder} layer cosine matrix diagonal is not ~1")
    return matrix


def build_same_layer_fold_cosines(coefficients: pd.DataFrame) -> pd.DataFrame:
    filtered = coefficients[coefficients["feature_index"] != -1].copy()
    filtered = filtered.sort_values(["fold", "layer", "feature_index"])
    vectors: dict[tuple[int, int], np.ndarray] = {}
    for (fold, layer), group in filtered.groupby(["fold", "layer"], sort=True):
        vectors[(int(fold), int(layer))] = group["coefficient_raw_scale"].astype(np.float64).to_numpy()

    rows = []
    for layer in EXPECTED_LAYERS:
        row: dict[str, float] = {"layer": float(layer)}
        pair_values = []
        for fold_a, fold_b in FOLD_PAIRS:
            value = cosine_similarity(vectors[(fold_a, layer)], vectors[(fold_b, layer)])
            row[f"fold_{fold_a}_vs_{fold_b}"] = value
            pair_values.append(value)
        row["mean_pairwise_cosine"] = float(np.mean(pair_values))
        row["min_pairwise_cosine"] = float(np.min(pair_values))
        row["max_pairwise_cosine"] = float(np.max(pair_values))
        rows.append(row)
    return pd.DataFrame(rows)[
        ["layer", *FOLD_PAIR_COLUMNS, "mean_pairwise_cosine", "min_pairwise_cosine", "max_pairwise_cosine"]
    ]


def build_summary_stats(spec: VariantSpec, r2_protein_layer: pd.DataFrame, geometry_protein_layer: pd.DataFrame) -> pd.DataFrame:
    metric_specs = [
        ("r2", "per_protein_r2", r2_protein_layer, spec.baseline, "Fold-averaged per-protein R^2 across all folds." if spec.baseline else "Split-0 per-protein R^2 combining train and test proteins."),
        ("tm_score", "tm_score", geometry_protein_layer, True, "Fold-4 TM-score values." if spec.baseline else "Split-0 TM-score values combining train and test proteins."),
        ("triangle_violation_fraction", "triangle_violation_fraction", geometry_protein_layer, False, "Fold-4 triangle violation fraction values." if spec.baseline else "Split-0 triangle violation fraction values."),
        ("triangle_mean_violation", "triangle_mean_violation", geometry_protein_layer, False, "Fold-4 triangle mean violation values." if spec.baseline else "Split-0 triangle mean violation values."),
    ]

    rows = []
    for metric_name, column, df, higher_is_better, note in metric_specs:
        protein_stats = summarize_distribution(df[column])
        layer_mean = df.groupby("layer", sort=True)[column].mean()
        layer_stats = summarize_distribution(layer_mean)
        best_layer, worst_layer = best_worst_layers(layer_mean, higher_is_better)
        rows.append(
            {
                "variant": spec.folder,
                "feature_dim": spec.feature_dim,
                "metric": metric_name,
                "distribution": "protein_layer",
                **protein_stats,
                "best_layer": best_layer[0],
                "best_layer_mean": best_layer[1],
                "worst_layer": worst_layer[0],
                "worst_layer_mean": worst_layer[1],
                "optimization_direction": metric_direction(metric_name),
                "notes": note,
            }
        )
        rows.append(
            {
                "variant": spec.folder,
                "feature_dim": spec.feature_dim,
                "metric": metric_name,
                "distribution": "layer_mean",
                **layer_stats,
                "best_layer": best_layer[0],
                "best_layer_mean": best_layer[1],
                "worst_layer": worst_layer[0],
                "worst_layer_mean": worst_layer[1],
                "optimization_direction": metric_direction(metric_name),
                "notes": f"Distribution over 48 layer means. {note}",
            }
        )
    return pd.DataFrame(rows)


def add_trend_panel(ax: plt.Axes, df: pd.DataFrame, y_column: str, color: str, title: str, y_label: str) -> None:
    ax.scatter(df["n_residues"], df[y_column], color=color, alpha=0.7, edgecolor="none")
    x_line, y_line = fit_line(df["n_residues"], df[y_column])
    if x_line.size:
        ax.plot(x_line, y_line, color="black", linewidth=2.0)
    pearson = pearson_corr(df["n_residues"], df[y_column])
    spearman = spearman_corr(df["n_residues"], df[y_column])
    ax.text(
        0.03,
        0.97,
        f"Pearson r = {pearson:.3f}\nSpearman rho = {spearman:.3f}",
        transform=ax.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "#bbbbbb"},
    )
    ax.set_title(title)
    ax.set_xlabel("Number of residues")
    ax.set_ylabel(y_label)
    ax.grid(True, alpha=0.25)


def plot_variant_r2_tm(processed: ProcessedVariant) -> None:
    spec = processed.spec
    out_path = spec.analysis_dir / f"{spec.prefix}_r2_tmscore_vs_layer.png"
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    r2_ax = axes[0]
    for _, group in processed.r2_protein_layer.groupby("protein_id", sort=True):
        r2_ax.plot(group["layer"], group["per_protein_r2"], color="#d0d0d0", linewidth=0.6, alpha=0.7)
    if spec.baseline:
        for fold in EXPECTED_FOLDS:
            fold_group = processed.r2_replicate_mean[processed.r2_replicate_mean["fold"] == fold]
            r2_ax.plot(
                fold_group["layer"],
                fold_group["per_protein_r2"],
                color=FOLD_COLORS[fold],
                linewidth=1.6,
                label=f"Fold {fold} mean",
            )
    else:
        for split in ("train", "test"):
            split_group = processed.r2_split_mean[processed.r2_split_mean["split"] == split]
            r2_ax.plot(
                split_group["layer"],
                split_group["per_protein_r2"],
                color=SPLIT_COLORS[split],
                linewidth=2.0,
                label=f"{split.title()} mean",
            )
    r2_ax.plot(
        processed.r2_layer_mean.index,
        processed.r2_layer_mean.values,
        color="black",
        linewidth=2.6,
        label="All-proteins mean",
    )
    r2_ax.set_title(f"{spec.label} R^2 across layers")
    r2_ax.set_ylabel("Per-protein R^2")
    r2_ax.grid(True, alpha=0.25)
    r2_ax.legend(fontsize=9, ncol=3 if spec.baseline else 2)

    tm_ax = axes[1]
    for _, group in processed.geometry_protein_layer.groupby("protein_id", sort=True):
        tm_ax.plot(group["layer"], group["tm_score"], color="#d0d0d0", linewidth=0.6, alpha=0.7)
    for split in ("train", "test"):
        split_group = processed.geometry_split_mean[processed.geometry_split_mean["split"] == split]
        tm_ax.plot(
            split_group["layer"],
            split_group["tm_score"],
            color=SPLIT_COLORS[split],
            linewidth=1.8,
            label=f"{split.title()} mean",
        )
    tm_ax.plot(
        processed.geometry_layer_mean["layer"],
        processed.geometry_layer_mean["tm_score"],
        color="black",
        linewidth=2.6,
        label="All-proteins mean",
    )
    title_suffix = " (fold 4 geometry)" if spec.baseline else ""
    tm_ax.set_title(f"{spec.label} TM-score across layers{title_suffix}")
    tm_ax.set_xlabel("Layer")
    tm_ax.set_ylabel("TM-score")
    tm_ax.grid(True, alpha=0.25)
    tm_ax.legend(fontsize=9, ncol=2)
    tm_ax.set_xticks(EXPECTED_LAYERS[::2])

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_variant_layer47_size_trends(processed: ProcessedVariant) -> None:
    spec = processed.spec
    out_path = spec.analysis_dir / f"{spec.prefix}_layer47_size_trends.png"

    r2_layer47 = processed.r2_protein_layer[processed.r2_protein_layer["layer"] == 47].copy()
    tm_layer47 = processed.geometry_protein_layer[processed.geometry_protein_layer["layer"] == 47].copy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.2), constrained_layout=True)
    add_trend_panel(
        axes[0],
        r2_layer47,
        "per_protein_r2",
        "#4c78a8",
        f"{spec.label} layer 47 R^2 vs protein size",
        "Per-protein R^2",
    )
    add_trend_panel(
        axes[1],
        tm_layer47,
        "tm_score",
        "#e45756",
        f"{spec.label} layer 47 TM-score vs protein size",
        "TM-score",
    )
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_layer_cosine_outputs(processed: ProcessedVariant) -> None:
    spec = processed.spec
    matrix_path = spec.analysis_dir / f"{spec.prefix}_layer_cosine_matrix.csv"
    heatmap_path = spec.analysis_dir / f"{spec.prefix}_layer_cosine_heatmap.png"

    matrix_df = pd.DataFrame(
        processed.layer_cosine_matrix,
        columns=[f"layer_{layer}" for layer in EXPECTED_LAYERS],
    )
    matrix_df.insert(0, "layer", EXPECTED_LAYERS)
    matrix_df.to_csv(matrix_path, index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(10, 8), constrained_layout=True)
    image = ax.imshow(
        processed.layer_cosine_matrix,
        origin="lower",
        cmap="coolwarm",
        vmin=-1.0,
        vmax=1.0,
    )
    replicate_note = "Fold 4" if spec.baseline else "Split 0"
    ax.set_title(f"{spec.label} layer coefficient cosine similarity ({replicate_note})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ticks = np.arange(0, 48, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(int(value)) for value in ticks], rotation=90, fontsize=7)
    ax.set_yticklabels([str(int(value)) for value in ticks], fontsize=7)
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Cosine similarity")
    fig.savefig(heatmap_path, dpi=200)
    plt.close(fig)


def plot_trajectory_grid(
    processed_variants: list[ProcessedVariant],
    metric_column: str,
    ylabel: str,
    out_path: Path,
    title: str,
    footer: str,
) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(14, 10), sharex=True, sharey=True, constrained_layout=True)
    axes_flat = axes.ravel()
    limits = padded_limits([variant.r2_protein_layer[metric_column].to_numpy() if metric_column == "per_protein_r2" else variant.geometry_protein_layer[metric_column].to_numpy() for variant in processed_variants])

    for ax, processed in zip(axes_flat, processed_variants):
        source_df = processed.r2_protein_layer if metric_column == "per_protein_r2" else processed.geometry_protein_layer
        mean_series = processed.r2_layer_mean if metric_column == "per_protein_r2" else processed.geometry_layer_mean.set_index("layer")[metric_column]
        for _, group in source_df.groupby("protein_id", sort=True):
            ax.plot(group["layer"], group[metric_column], color="#d0d0d0", linewidth=0.55, alpha=0.55)
        ax.plot(
            mean_series.index,
            mean_series.values,
            color=VARIANT_COLORS[processed.spec.folder],
            linewidth=2.4,
        )
        ax.set_title(processed.spec.label)
        ax.set_xlim(0, 47)
        ax.set_ylim(*limits)
        ax.set_xticks([0, 10, 20, 30, 40, 47])
        ax.grid(True, alpha=0.2)

    for ax in axes[1]:
        ax.set_xlabel("Layer")
    for ax in axes[:, 0]:
        ax.set_ylabel(ylabel)

    fig.suptitle(title, fontsize=14)
    fig.text(0.5, 0.01, footer, ha="center", fontsize=9)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_mean_curves(processed_variants: list[ProcessedVariant], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    for processed in processed_variants:
        color = VARIANT_COLORS[processed.spec.folder]
        axes[0].plot(
            processed.r2_layer_mean.index,
            processed.r2_layer_mean.values,
            color=color,
            linewidth=2.2,
            label=processed.spec.label,
        )
        axes[1].plot(
            processed.geometry_layer_mean["layer"],
            processed.geometry_layer_mean["tm_score"],
            color=color,
            linewidth=2.2,
            label=processed.spec.label,
        )

    axes[0].set_title("Average R^2 across layers by variant")
    axes[0].set_ylabel("Mean per-protein R^2")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Average TM-score across layers by variant")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean TM-score")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(EXPECTED_LAYERS[::2])

    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def plot_triangle_curves(processed_variants: list[ProcessedVariant], out_path: Path) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 10), sharex=True, constrained_layout=True)

    for processed in processed_variants:
        color = VARIANT_COLORS[processed.spec.folder]
        axes[0].plot(
            processed.geometry_layer_mean["layer"],
            processed.geometry_layer_mean["triangle_violation_fraction"],
            color=color,
            linewidth=2.2,
            label=processed.spec.label,
        )
        axes[1].plot(
            processed.geometry_layer_mean["layer"],
            processed.geometry_layer_mean["triangle_mean_violation"],
            color=color,
            linewidth=2.2,
            label=processed.spec.label,
        )

    axes[0].set_title("Triangle inequality violation fraction across layers")
    axes[0].set_ylabel("Mean violation fraction")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(fontsize=9)

    axes[1].set_title("Triangle inequality mean violation magnitude across layers")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean violation magnitude")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(fontsize=9)
    axes[1].set_xticks(EXPECTED_LAYERS[::2])

    fig.text(0.5, 0.01, "Baseline geometry uses fold 4; train/test variants use split 0.", ha="center", fontsize=9)
    fig.savefig(out_path, dpi=200)
    plt.close(fig)


def save_variant_outputs(processed: ProcessedVariant) -> None:
    spec = processed.spec
    spec.analysis_dir.mkdir(parents=True, exist_ok=True)

    summary_path = spec.analysis_dir / f"{spec.prefix}_summary_statistics.csv"
    processed.summary_stats.to_csv(summary_path, index=False, float_format="%.6f")

    plot_variant_r2_tm(processed)
    plot_variant_layer47_size_trends(processed)
    save_layer_cosine_outputs(processed)

    if processed.same_layer_fold_cosine is not None:
        same_layer_path = spec.analysis_dir / f"{spec.prefix}_same_layer_fold_cosine_similarity.csv"
        processed.same_layer_fold_cosine.to_csv(same_layer_path, index=False, float_format="%.6f")


def save_comparison_outputs(processed_variants: list[ProcessedVariant]) -> None:
    COMPARE_DIR.mkdir(parents=True, exist_ok=True)

    plot_trajectory_grid(
        processed_variants,
        metric_column="per_protein_r2",
        ylabel="Per-protein R^2",
        out_path=COMPARE_DIR / "protein_trajectory_grid_r2.png",
        title="Per-protein R^2 trajectories by variant",
        footer="Baseline R^2 is averaged across its five folds.",
    )
    plot_trajectory_grid(
        processed_variants,
        metric_column="tm_score",
        ylabel="TM-score",
        out_path=COMPARE_DIR / "protein_trajectory_grid_tm_score.png",
        title="Per-protein TM-score trajectories by variant",
        footer="Baseline TM-score uses fold 4 geometry; train/test variants use split 0 geometry.",
    )
    plot_mean_curves(processed_variants, COMPARE_DIR / "variant_mean_r2_tm_across_layers.png")
    plot_triangle_curves(processed_variants, COMPARE_DIR / "variant_triangle_violation_across_layers.png")

    rows = []
    for processed in processed_variants:
        geometry_means = processed.geometry_layer_mean.set_index("layer")
        for layer in EXPECTED_LAYERS:
            rows.append(
                {
                    "variant": processed.spec.folder,
                    "label": processed.spec.label,
                    "feature_dim": processed.spec.feature_dim,
                    "layer": layer,
                    "mean_r2": float(processed.r2_layer_mean.loc[layer]),
                    "mean_tm_score": float(geometry_means.loc[layer, "tm_score"]),
                    "mean_triangle_violation_fraction": float(
                        geometry_means.loc[layer, "triangle_violation_fraction"]
                    ),
                    "mean_triangle_mean_violation": float(
                        geometry_means.loc[layer, "triangle_mean_violation"]
                    ),
                    "r2_source": "fold_averaged_all_proteins" if processed.spec.baseline else "split0_all_proteins",
                    "geometry_source": "fold4_all_proteins" if processed.spec.baseline else "split0_all_proteins",
                }
            )
    pd.DataFrame(rows).to_csv(COMPARE_DIR / "variant_layer_means.csv", index=False, float_format="%.6f")


def process_variant(spec: VariantSpec) -> ProcessedVariant:
    per_protein, coefficients, geometry_metrics = load_inputs(spec)
    validate_inputs(spec, per_protein, coefficients, geometry_metrics)

    r2_protein_layer, r2_replicate_mean, r2_split_mean, r2_layer_mean = build_r2_tables(spec, per_protein)
    geometry_protein_layer, geometry_split_mean, geometry_layer_mean = build_geometry_tables(geometry_metrics)
    layer_cosine_matrix = build_layer_cosine_matrix(spec, coefficients)
    same_layer_fold_cosine = build_same_layer_fold_cosines(coefficients) if spec.baseline else None
    summary_stats = build_summary_stats(spec, r2_protein_layer, geometry_protein_layer)

    return ProcessedVariant(
        spec=spec,
        r2_protein_layer=r2_protein_layer,
        r2_replicate_mean=r2_replicate_mean,
        r2_split_mean=r2_split_mean,
        r2_layer_mean=r2_layer_mean,
        geometry_protein_layer=geometry_protein_layer,
        geometry_split_mean=geometry_split_mean,
        geometry_layer_mean=geometry_layer_mean,
        layer_cosine_matrix=layer_cosine_matrix,
        same_layer_fold_cosine=same_layer_fold_cosine,
        summary_stats=summary_stats,
    )


def main() -> None:
    processed_variants = [process_variant(spec) for spec in VARIANT_SPECS]
    for processed in processed_variants:
        save_variant_outputs(processed)
    save_comparison_outputs(processed_variants)


if __name__ == "__main__":
    main()

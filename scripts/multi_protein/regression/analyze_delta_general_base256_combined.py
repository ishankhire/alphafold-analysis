#!/usr/bin/env python3
"""Create local analysis plots for the combined CAMEO+ECOD base256 run."""

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
COMBINED_DIR = DELTA_DIR / "general_base256_combined"
BASELINE_DIR = DELTA_DIR / "baseline256_kfold"
OUT_DIR = COMBINED_DIR / "analysis"

PREFIX = "general_base256_combined"
EXPECTED_LAYERS = tuple(range(48))
FEATURE_DIM = 256
SPLIT_ID = 0

ORANGE = "#f58518"
BLUE = "#4c78a8"
RED = "#e45756"
GREEN = "#54a24b"
PURPLE = "#7f6edb"
GREY = "#b8b8b8"
GRID_GREY = "#dddddd"


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


def require_split_id(path: Path, df: pd.DataFrame) -> None:
    split_ids = tuple(sorted(int(value) for value in df["split_id"].dropna().unique()))
    if split_ids != (SPLIT_ID,):
        raise ValueError(f"{path} expected split_id 0, found {split_ids}")


def require_nonempty(path: Path) -> None:
    if not path.exists() or path.stat().st_size == 0:
        raise ValueError(f"Expected nonempty output file: {path}")


def load_combined_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    per_protein_path = COMBINED_DIR / "per_protein.csv"
    geometry_path = COMBINED_DIR / "geometry_metrics.csv"
    coefficients_path = COMBINED_DIR / "coefficients.csv"

    per_protein = pd.read_csv(per_protein_path)
    geometry = pd.read_csv(geometry_path)
    coefficients = pd.read_csv(coefficients_path)

    require_columns(
        per_protein_path,
        per_protein,
        {"split_id", "layer", "split", "protein_id", "n_residues", "per_protein_r2"},
    )
    require_columns(
        geometry_path,
        geometry,
        {
            "split_id",
            "layer",
            "split",
            "protein_id",
            "n_residues",
            "tm_score",
            "frac_top3_eigenvalues",
            "n_negative_eigenvalues",
            "neg_eigenvalue_fraction",
            "triangle_violation_fraction",
            "triangle_mean_violation",
            "triangle_max_violation",
        },
    )
    require_columns(
        coefficients_path,
        coefficients,
        {"split_id", "layer", "feature_index", "feature_name", "coefficient_raw_scale"},
    )

    for path, df in [
        (per_protein_path, per_protein),
        (geometry_path, geometry),
        (coefficients_path, coefficients),
    ]:
        require_layers(path, df)
        require_split_id(path, df)

    expected_splits = ("test", "train")
    for path, df in [(per_protein_path, per_protein), (geometry_path, geometry)]:
        splits = tuple(sorted(str(value) for value in df["split"].dropna().unique()))
        if splits != expected_splits:
            raise ValueError(f"{path} expected train/test splits, found {splits}")

    require_no_duplicates(per_protein_path, per_protein, ["split_id", "layer", "split", "protein_id"])
    require_no_duplicates(geometry_path, geometry, ["split_id", "layer", "split", "protein_id"])
    require_no_duplicates(coefficients_path, coefficients, ["split_id", "layer", "feature_index"])

    non_intercept = coefficients[coefficients["feature_index"] != -1]
    coefficient_counts = non_intercept.groupby(["split_id", "layer"], sort=True).size()
    if not (coefficient_counts == FEATURE_DIM).all():
        raise ValueError("coefficients.csv must contain 256 non-intercept coefficients per layer")

    intercept_counts = coefficients[coefficients["feature_index"] == -1].groupby(
        ["split_id", "layer"], sort=True
    ).size()
    if not (intercept_counts == 1).all():
        raise ValueError("coefficients.csv must contain one intercept per layer")

    return per_protein, geometry, coefficients


def load_baseline_inputs() -> tuple[pd.DataFrame, pd.DataFrame]:
    per_protein_path = BASELINE_DIR / "per_protein.csv"
    geometry_path = BASELINE_DIR / "geometry_metrics_last_fold.csv"

    per_protein = pd.read_csv(per_protein_path)
    geometry = pd.read_csv(geometry_path)

    require_columns(
        per_protein_path,
        per_protein,
        {"fold", "layer", "split", "protein_id", "n_residues", "per_protein_r2"},
    )
    require_columns(
        geometry_path,
        geometry,
        {"fold", "layer", "split", "protein_id", "n_residues", "tm_score"},
    )
    require_layers(per_protein_path, per_protein)
    require_layers(geometry_path, geometry)

    folds = tuple(sorted(int(value) for value in per_protein["fold"].dropna().unique()))
    if folds != (0, 1, 2, 3, 4):
        raise ValueError(f"{per_protein_path} expected folds 0-4, found {folds}")

    geometry_folds = tuple(sorted(int(value) for value in geometry["fold"].dropna().unique()))
    if geometry_folds != (4,):
        raise ValueError(f"{geometry_path} expected fold 4 only, found {geometry_folds}")

    return per_protein, geometry


def split_layer_mean(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    return df.groupby(["split", "layer"], sort=True)[value_columns].mean().reset_index()


def test_layer_mean(df: pd.DataFrame, value_columns: list[str]) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise ValueError("Expected at least one test row")
    return test.groupby("layer", sort=True)[value_columns].mean().reset_index()


def test_rows(df: pd.DataFrame) -> pd.DataFrame:
    test = df[df["split"] == "test"].copy()
    if test.empty:
        raise ValueError("Expected at least one test row")
    counts = test.groupby(["protein_id", "layer"], sort=True).size()
    if not (counts == 1).all():
        raise ValueError("Test rows must be unique per (protein_id, layer)")
    return test.sort_values(["protein_id", "layer"]).reset_index(drop=True)


def add_common_axis_style(ax: plt.Axes) -> None:
    ax.set_xlim(0, 47)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 47])
    ax.grid(True, color=GRID_GREY, linewidth=0.8, alpha=0.7)
    ax.tick_params(axis="both", width=1.0, length=5)


def padded_limits(values: list[np.ndarray], include_zero: bool = False) -> tuple[float, float]:
    finite = [
        np.asarray(array, dtype=np.float64).ravel()
        for array in values
        if np.asarray(array).size
    ]
    if not finite:
        return (0.0, 1.0)
    flat = np.concatenate(finite)
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
    r2_test = test_rows(per_protein)
    tm_test = test_rows(geometry)
    r2_split_mean = split_layer_mean(per_protein, ["per_protein_r2"])
    tm_split_mean = split_layer_mean(geometry, ["tm_score"])

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 9.2), sharex=True, constrained_layout=True)

    for _, group in r2_test.groupby("protein_id", sort=True):
        axes[0].plot(group["layer"], group["per_protein_r2"], color=GREY, linewidth=0.65, alpha=0.45)
    for split, color, label, linewidth in [
        ("train", BLUE, "Train mean", 2.0),
        ("test", ORANGE, "Held-out mean", 2.8),
    ]:
        group = r2_split_mean[r2_split_mean["split"] == split]
        axes[0].plot(group["layer"], group["per_protein_r2"], color=color, linewidth=linewidth, label=label)
    axes[0].set_title("general_base256_combined R$^2$ across layers")
    axes[0].set_ylabel("Per-protein R$^2$")
    axes[0].set_ylim(padded_limits([r2_test["per_protein_r2"].to_numpy()], include_zero=True))
    add_common_axis_style(axes[0])
    axes[0].legend(
        handles=[
            Line2D([0], [0], color=GREY, linewidth=1.0, alpha=0.6, label="Held-out proteins"),
            Line2D([0], [0], color=BLUE, linewidth=2.0, label="Train mean"),
            Line2D([0], [0], color=ORANGE, linewidth=2.8, label="Held-out mean"),
        ],
        loc="lower right",
        frameon=False,
    )

    for _, group in tm_test.groupby("protein_id", sort=True):
        axes[1].plot(group["layer"], group["tm_score"], color=GREY, linewidth=0.65, alpha=0.45)
    for split, color, label, linewidth in [
        ("train", BLUE, "Train mean", 2.0),
        ("test", ORANGE, "Held-out mean", 2.8),
    ]:
        group = tm_split_mean[tm_split_mean["split"] == split]
        axes[1].plot(group["layer"], group["tm_score"], color=color, linewidth=linewidth, label=label)
    axes[1].set_title("general_base256_combined TM-score across layers")
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
    r2_split_mean = split_layer_mean(per_protein, ["per_protein_r2"])
    tm_split_mean = split_layer_mean(geometry, ["tm_score"])

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.5), sharex=True, constrained_layout=True)
    for ax, mean_df, column, title, ylabel in [
        (
            axes[0],
            r2_split_mean,
            "per_protein_r2",
            "general_base256_combined mean R$^2$ across layers",
            "Mean per-protein R$^2$",
        ),
        (
            axes[1],
            tm_split_mean,
            "tm_score",
            "general_base256_combined mean TM-score across layers",
            "Mean TM-score",
        ),
    ]:
        for split, color, label, linewidth in [
            ("train", BLUE, "Train mean", 2.0),
            ("test", ORANGE, "Held-out mean", 2.8),
        ]:
            group = mean_df[mean_df["split"] == split]
            ax.plot(group["layer"], group[column], color=color, linewidth=linewidth, label=label)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0.0, 1.0)
        add_common_axis_style(ax)
        ax.legend(loc="lower right", frameon=False)
    axes[1].set_xlabel("Layer")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("Cannot compute cosine similarity for a zero coefficient vector")
    normalized = vectors / norms
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def build_layer_cosine_matrix(coefficients: pd.DataFrame) -> np.ndarray:
    filtered = coefficients[
        (coefficients["split_id"] == SPLIT_ID) & (coefficients["feature_index"] != -1)
    ].copy()
    filtered = filtered.sort_values(["layer", "feature_index"])

    vectors = []
    for layer, group in filtered.groupby("layer", sort=True):
        if int(layer) not in EXPECTED_LAYERS:
            raise ValueError(f"Unexpected layer {layer} in coefficients")
        vector = group["coefficient_raw_scale"].astype(np.float64).to_numpy()
        if vector.shape != (FEATURE_DIM,):
            raise ValueError(f"Layer {layer} expected {FEATURE_DIM} coefficients, found {vector.shape[0]}")
        vectors.append(vector)

    matrix = cosine_similarity_matrix(np.vstack(vectors))
    if matrix.shape != (48, 48):
        raise ValueError(f"Expected 48x48 cosine matrix, found {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=1e-8):
        raise ValueError("Layer cosine matrix is not symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-6):
        raise ValueError("Layer cosine matrix diagonal is not approximately 1")
    return matrix


def save_layer_cosine_outputs(coefficients: pd.DataFrame) -> tuple[Path, Path]:
    matrix = build_layer_cosine_matrix(coefficients)
    matrix_path = OUT_DIR / f"{PREFIX}_layer_cosine_matrix.csv"
    heatmap_path = OUT_DIR / f"{PREFIX}_layer_cosine_heatmap.png"

    matrix_df = pd.DataFrame(matrix, columns=[f"layer_{layer}" for layer in EXPECTED_LAYERS])
    matrix_df.insert(0, "layer", EXPECTED_LAYERS)
    matrix_df.to_csv(matrix_path, index=False, float_format="%.6f")

    fig, ax = plt.subplots(figsize=(10.5, 8.5), constrained_layout=True)
    image = ax.imshow(matrix, origin="lower", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    ax.set_title("general_base256_combined layer coefficient cosine similarity (Split 0)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    ticks = np.arange(0, 48, 2)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    ax.set_xticklabels([str(int(value)) for value in ticks], rotation=90)
    ax.set_yticklabels([str(int(value)) for value in ticks])
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("Cosine similarity")
    fig.savefig(heatmap_path)
    plt.close(fig)

    return matrix_path, heatmap_path


def add_train_test_lines(ax: plt.Axes, mean_df: pd.DataFrame, column: str, *, train_label: str = "Train mean") -> None:
    for split, color, label, linewidth in [
        ("train", BLUE, train_label, 1.9),
        ("test", ORANGE, "Held-out mean", 2.7),
    ]:
        group = mean_df[mean_df["split"] == split]
        ax.plot(group["layer"], group[column], color=color, linewidth=linewidth, label=label)


def save_geometry_layer_means(geometry: pd.DataFrame) -> Path:
    df = geometry.copy()
    df["negative_eigenvalue_count_fraction"] = df["n_negative_eigenvalues"] / df["n_residues"]
    value_columns = [
        "tm_score",
        "frac_top3_eigenvalues",
        "n_negative_eigenvalues",
        "negative_eigenvalue_count_fraction",
        "neg_eigenvalue_fraction",
        "triangle_violation_fraction",
        "triangle_mean_violation",
        "triangle_max_violation",
    ]
    mean_df = split_layer_mean(df, value_columns)
    out_path = OUT_DIR / f"{PREFIX}_geometry_layer_means.csv"
    mean_df.to_csv(out_path, index=False, float_format="%.6f")
    return out_path


def plot_triangle_violations(geometry: pd.DataFrame) -> Path:
    out_path = OUT_DIR / f"{PREFIX}_triangle_violations_vs_layer.png"
    mean_df = split_layer_mean(geometry, ["triangle_violation_fraction", "triangle_mean_violation"])
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.5), sharex=True, constrained_layout=True)

    add_train_test_lines(axes[0], mean_df, "triangle_violation_fraction")
    axes[0].set_title("Triangle inequality violation proportion across layers")
    axes[0].set_ylabel("Mean violation proportion")
    axes[0].set_ylim(0.0, 1.0)
    add_common_axis_style(axes[0])
    axes[0].legend(loc="best", frameon=False)

    add_train_test_lines(axes[1], mean_df, "triangle_mean_violation")
    axes[1].set_title("Triangle inequality violation magnitude across layers")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean violation magnitude")
    add_common_axis_style(axes[1])
    axes[1].legend(loc="best", frameon=False)

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def plot_top3_eigenvalue_fraction(geometry: pd.DataFrame) -> Path:
    out_path = OUT_DIR / f"{PREFIX}_top3_eigenvalue_fraction_vs_layer.png"
    mean_df = split_layer_mean(geometry, ["frac_top3_eigenvalues"])
    fig, ax = plt.subplots(figsize=(9.5, 5.5), constrained_layout=True)

    add_train_test_lines(ax, mean_df, "frac_top3_eigenvalues")
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
    mean_df = split_layer_mean(
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
        add_train_test_lines(ax, mean_df, column)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        if fixed_ylim is not None:
            ax.set_ylim(*fixed_ylim)
        else:
            values = [mean_df[column].to_numpy()]
            ax.set_ylim(padded_limits(values, include_zero=True))
        add_common_axis_style(ax)
        ax.legend(loc="best", frameon=False)
    axes[2].set_xlabel("Layer")

    fig.savefig(out_path)
    plt.close(fig)
    return out_path


def save_baseline_comparison(per_protein: pd.DataFrame, geometry: pd.DataFrame) -> tuple[Path, Path]:
    baseline_per_protein, baseline_geometry = load_baseline_inputs()

    combined_r2 = test_layer_mean(per_protein, ["per_protein_r2"]).rename(
        columns={"per_protein_r2": "general_base256_combined_test_mean_r2"}
    )
    combined_tm = test_layer_mean(geometry, ["tm_score"]).rename(
        columns={"tm_score": "general_base256_combined_test_mean_tm_score"}
    )
    baseline_r2 = test_layer_mean(baseline_per_protein, ["per_protein_r2"]).rename(
        columns={"per_protein_r2": "baseline256_kfold_test_mean_r2"}
    )
    baseline_tm = test_layer_mean(baseline_geometry, ["tm_score"]).rename(
        columns={"tm_score": "baseline256_kfold_fold4_test_mean_tm_score"}
    )

    merged = combined_r2.merge(combined_tm, on="layer").merge(baseline_r2, on="layer").merge(baseline_tm, on="layer")
    merged["general_base256_combined_r2_minus_baseline"] = (
        merged["general_base256_combined_test_mean_r2"] - merged["baseline256_kfold_test_mean_r2"]
    )
    merged["general_base256_combined_tm_minus_baseline"] = (
        merged["general_base256_combined_test_mean_tm_score"]
        - merged["baseline256_kfold_fold4_test_mean_tm_score"]
    )

    csv_path = OUT_DIR / f"{PREFIX}_baseline256_comparison_layer_means.csv"
    merged.to_csv(csv_path, index=False, float_format="%.6f")

    plot_path = OUT_DIR / f"{PREFIX}_vs_baseline256_kfold_r2_tm_score.png"
    fig, axes = plt.subplots(2, 1, figsize=(9.5, 8.5), sharex=True, constrained_layout=True)
    axes[0].plot(
        merged["layer"],
        merged["general_base256_combined_test_mean_r2"],
        color=ORANGE,
        linewidth=2.8,
        label="general_base256_combined test mean",
    )
    axes[0].plot(
        merged["layer"],
        merged["baseline256_kfold_test_mean_r2"],
        color=BLUE,
        linewidth=2.4,
        label="baseline256_kfold test mean",
    )
    axes[0].set_title("Held-out R$^2$: combined base256 vs baseline256_kfold")
    axes[0].set_ylabel("Mean per-protein R$^2$")
    axes[0].set_ylim(0.0, 1.0)
    add_common_axis_style(axes[0])
    axes[0].legend(loc="lower right", frameon=False)

    axes[1].plot(
        merged["layer"],
        merged["general_base256_combined_test_mean_tm_score"],
        color=ORANGE,
        linewidth=2.8,
        label="general_base256_combined test mean",
    )
    axes[1].plot(
        merged["layer"],
        merged["baseline256_kfold_fold4_test_mean_tm_score"],
        color=BLUE,
        linewidth=2.4,
        label="baseline256_kfold fold-4 test mean",
    )
    axes[1].set_title("Held-out TM-score: combined base256 vs baseline256_kfold")
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


def write_report(
    per_protein: pd.DataFrame,
    geometry: pd.DataFrame,
    artifact_paths: list[Path],
) -> Path:
    r2_test_mean = test_layer_mean(per_protein, ["per_protein_r2"])
    tm_test_mean = test_layer_mean(geometry, ["tm_score"])
    triangle_test_mean = test_layer_mean(
        geometry,
        ["triangle_violation_fraction", "triangle_mean_violation"],
    )
    top3_test_mean = test_layer_mean(geometry, ["frac_top3_eigenvalues"])

    best_r2_layer, best_r2_value = best_layer(r2_test_mean, "per_protein_r2", True)
    best_tm_layer, best_tm_value = best_layer(tm_test_mean, "tm_score", True)
    best_triangle_frac_layer, best_triangle_frac_value = best_layer(
        triangle_test_mean,
        "triangle_violation_fraction",
        False,
    )
    best_top3_layer, best_top3_value = best_layer(top3_test_mean, "frac_top3_eigenvalues", True)

    n_loaded = int(per_protein["protein_id"].nunique())
    split_counts = (
        per_protein[per_protein["layer"] == 0]
        .groupby("split", sort=True)["protein_id"]
        .nunique()
        .to_dict()
    )

    artifact_lines = "\n".join(f"- `{path.relative_to(PROJECT_ROOT)}`" for path in artifact_paths)
    report = f"""# general_base256_combined analysis

Inputs:
- `delta_files/general_base256_combined/per_protein.csv`
- `delta_files/general_base256_combined/geometry_metrics.csv`
- `delta_files/general_base256_combined/coefficients.csv`

Loaded proteins: {n_loaded}
Train proteins: {int(split_counts.get("train", 0))}
Held-out proteins: {int(split_counts.get("test", 0))}

Held-out layer highlights:
- Best mean R^2: layer {best_r2_layer} ({best_r2_value:.4f})
- Best mean TM-score: layer {best_tm_layer} ({best_tm_value:.4f})
- Lowest mean triangle violation proportion: layer {best_triangle_frac_layer} ({best_triangle_frac_value:.4f})
- Highest mean top-3 eigenvalue fraction: layer {best_top3_layer} ({best_top3_value:.4f})

Individual triangle violation data:
The current saved artifacts do not contain one row per triangle/triple. They only contain aggregate per-protein fields: `triangle_violation_fraction`, `triangle_mean_violation`, and `triangle_max_violation`.

The geometry helper computes a min-indirect-path summary for each directed residue pair, then saves only the fraction, mean violation magnitude, and max violation magnitude. It does not save each `(i, j, k)` triple, the intermediate residue, or each individual violation amount. Recovering those details would require a replay/export job from predicted distance matrices.

Baseline comparison caveat:
The baseline plot uses saved held-out summaries. `general_base256_combined` has 552 loaded proteins with 111 held-out proteins; `baseline256_kfold` has 184 proteins, R^2 pooled across five test folds, and TM-score from the saved fold-4 geometry file.

Generated artifacts:
{artifact_lines}
"""

    out_path = OUT_DIR / f"{PREFIX}_analysis_report.md"
    out_path.write_text(report, encoding="utf-8")
    return out_path


def assert_outputs(paths: list[Path]) -> None:
    for path in paths:
        require_nonempty(path)

    matrix_path = OUT_DIR / f"{PREFIX}_layer_cosine_matrix.csv"
    matrix = pd.read_csv(matrix_path).drop(columns=["layer"]).to_numpy(dtype=np.float64)
    if matrix.shape != (48, 48):
        raise ValueError(f"{matrix_path} expected 48x48 matrix, found {matrix.shape}")
    if not np.allclose(matrix, matrix.T, atol=1e-6):
        raise ValueError(f"{matrix_path} is not symmetric")
    if not np.allclose(np.diag(matrix), 1.0, atol=1e-6):
        raise ValueError(f"{matrix_path} diagonal is not 1")


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    per_protein, geometry, coefficients = load_combined_inputs()

    output_paths = [
        plot_r2_tm_with_trajectories(per_protein, geometry),
        plot_r2_tm_mean_only(per_protein, geometry),
        *save_layer_cosine_outputs(coefficients),
        save_geometry_layer_means(geometry),
        plot_triangle_violations(geometry),
        plot_top3_eigenvalue_fraction(geometry),
        plot_negative_eigenvalues(geometry),
    ]

    baseline_csv, baseline_plot = save_baseline_comparison(per_protein, geometry)
    output_paths.extend([baseline_csv, baseline_plot])
    output_paths.append(write_report(per_protein, geometry, output_paths))

    assert_outputs(output_paths)
    print("Wrote analysis artifacts:")
    for path in output_paths:
        print(f"  {path.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze updated Delta test2 k-fold geometry outputs.

This script:
1. Performs a lightweight validation that the updated Delta regression outputs
   preserved the old run's regression and test-split MDS results.
2. Builds per-protein layer aggregates from the expanded train+test MDS table.
3. Generates plots and compact CSV summaries under a dedicated output folder.
4. Writes a Markdown summary in analysis_summaries/.
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from datetime import datetime
from pathlib import Path

EARLY_PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.makedirs(EARLY_PROJECT_ROOT / ".mplconfig", exist_ok=True)
os.makedirs(EARLY_PROJECT_ROOT / ".cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(EARLY_PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(EARLY_PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
from scipy import stats
from sklearn.linear_model import LinearRegression

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = EARLY_PROJECT_ROOT
OLD_DIR = PROJECT_ROOT / "delta_files" / "test2_linear_reg_data"
UPDATED_DIR = PROJECT_ROOT / "delta_files" / "updated_test2_linear_reg_data"

OUTPUT_ROOT = (
    PROJECT_ROOT / "proteins" / "multi_protein_results" / "test2_updated_delta_analysis"
)
CSV_DIR = OUTPUT_ROOT / "csv_files"
VIS_DIR = OUTPUT_ROOT / "visualizations"
SUMMARY_PATH = (
    PROJECT_ROOT
    / "analysis_summaries"
    / "test2_updated_delta_validation_and_geometry_analysis.md"
)

NUMERIC_MDS_FIELDS = [
    "fold",
    "layer",
    "n_residues",
    "per_protein_r2",
    "rmsd",
    "tm_score",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "neg_eigenvalue_fraction",
    "n_negative_predictions",
    "stress",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
]

AGG_METRICS = [
    "n_residues",
    "per_protein_r2",
    "tm_score",
    "frac_top3_eigenvalues",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
]

SHARED_OLD_MDS_FIELDS = [
    "fold",
    "layer",
    "protein_id",
    "n_residues",
    "rmsd",
    "tm_score",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "neg_eigenvalue_fraction",
    "n_negative_predictions",
    "stress",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
]

PLOT_PROTEIN_COLORS = plt.cm.turbo(np.linspace(0.08, 0.92, 25))
SPLIT_COLORS = {"train": "#1f77b4", "test": "#d95f02"}


def ensure_dirs():
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    VIS_DIR.mkdir(parents=True, exist_ok=True)
    SUMMARY_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_csv_rows(path: Path):
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def convert_value(key: str, value: str):
    if key in NUMERIC_MDS_FIELDS:
        try:
            return float(value)
        except ValueError:
            return float("nan")
    return value


def convert_rows(rows):
    converted = []
    for row in rows:
        converted.append({key: convert_value(key, value) for key, value in row.items()})
    return converted


def exact_text_equal(path_a: Path, path_b: Path):
    return path_a.read_text() == path_b.read_text()


def validate_outputs():
    exact_files = [
        "test2_kfold_regression_r2.csv",
        "test2_kfold_regression_r2_summary.csv",
        "test2_kfold_regression_coefficients.csv",
    ]
    exact_results = {
        name: exact_text_equal(OLD_DIR / name, UPDATED_DIR / name) for name in exact_files
    }

    old_rows = load_csv_rows(OLD_DIR / "test2_kfold_mds_metrics.csv")
    updated_rows = load_csv_rows(UPDATED_DIR / "test2_kfold_mds_metrics.csv")
    updated_test = [row for row in updated_rows if row["split"] == "test"]

    old_map = {
        (row["fold"], row["layer"], row["protein_id"]): {
            key: row[key] for key in SHARED_OLD_MDS_FIELDS
        }
        for row in old_rows
    }
    updated_map = {
        (row["fold"], row["layer"], row["protein_id"]): {
            key: row[key] for key in SHARED_OLD_MDS_FIELDS
        }
        for row in updated_test
    }
    mds_test_match = old_map == updated_map

    return {
        "exact_results": exact_results,
        "old_mds_rows": len(old_rows),
        "updated_mds_rows": len(updated_rows),
        "updated_test_rows": len(updated_test),
        "mds_test_match": mds_test_match,
    }


def safe_corr(x, y):
    x_arr = np.asarray(x, dtype=float)
    y_arr = np.asarray(y, dtype=float)
    mask = ~(np.isnan(x_arr) | np.isnan(y_arr))
    x_arr = x_arr[mask]
    y_arr = y_arr[mask]
    n = int(x_arr.size)
    if n < 3:
        return {"n": n, "pearson_r": math.nan, "pearson_p": math.nan,
                "spearman_r": math.nan, "spearman_p": math.nan}
    if np.allclose(x_arr, x_arr[0]) or np.allclose(y_arr, y_arr[0]):
        return {"n": n, "pearson_r": math.nan, "pearson_p": math.nan,
                "spearman_r": math.nan, "spearman_p": math.nan}

    pearson_r, pearson_p = stats.pearsonr(x_arr, y_arr)
    spearman_r, spearman_p = stats.spearmanr(x_arr, y_arr)
    return {
        "n": n,
        "pearson_r": float(pearson_r),
        "pearson_p": float(pearson_p),
        "spearman_r": float(spearman_r),
        "spearman_p": float(spearman_p),
    }


def format_float(value, digits=4):
    if value is None:
        return "nan"
    try:
        value = float(value)
    except (TypeError, ValueError):
        return str(value)
    if math.isnan(value):
        return "nan"
    return f"{value:.{digits}f}"


def write_csv(path: Path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            out = {}
            for key in fieldnames:
                value = row.get(key)
                if isinstance(value, float):
                    out[key] = "nan" if math.isnan(value) else f"{value:.6f}"
                else:
                    out[key] = value
            writer.writerow(out)


def aggregate_means(rows, group_keys, metrics):
    grouped = {}
    for row in rows:
        key = tuple(row[k] for k in group_keys)
        if key not in grouped:
            grouped[key] = {k: row[k] for k in group_keys}
            for metric in metrics:
                grouped[key][metric] = []
        for metric in metrics:
            value = row[metric]
            if not math.isnan(value):
                grouped[key][metric].append(value)

    output = []
    for key in sorted(grouped):
        record = {}
        bucket = grouped[key]
        for group_key in group_keys:
            record[group_key] = bucket[group_key]
        for metric in metrics:
            values = bucket[metric]
            record[metric] = float(np.mean(values)) if values else math.nan
        output.append(record)
    return output


def rows_to_lookup(rows, keys):
    lookup = {}
    for row in rows:
        lookup[tuple(row[k] for k in keys)] = row
    return lookup


def build_gap_rows(train_means, test_rows):
    train_lookup = rows_to_lookup(train_means, ["protein_id", "layer"])
    test_lookup = rows_to_lookup(test_rows, ["protein_id", "layer"])
    output = []
    for key in sorted(train_lookup):
        if key not in test_lookup:
            continue
        train_row = train_lookup[key]
        test_row = test_lookup[key]
        output.append({
            "protein_id": key[0],
            "layer": key[1],
            "n_residues": test_row["n_residues"],
            "train_mean_r2": train_row["per_protein_r2"],
            "test_r2": test_row["per_protein_r2"],
            "train_minus_test_r2": train_row["per_protein_r2"] - test_row["per_protein_r2"],
            "train_mean_tm_score": train_row["tm_score"],
            "test_tm_score": test_row["tm_score"],
            "train_minus_test_tm_score": train_row["tm_score"] - test_row["tm_score"],
            "train_mean_triangle_violation_fraction": train_row["triangle_violation_fraction"],
            "test_triangle_violation_fraction": test_row["triangle_violation_fraction"],
            "train_minus_test_triangle_violation_fraction": (
                train_row["triangle_violation_fraction"]
                - test_row["triangle_violation_fraction"]
            ),
        })
    return output


def get_layers(rows):
    return sorted({int(row["layer"]) for row in rows})


def get_proteins(rows):
    return sorted({row["protein_id"] for row in rows})


def build_matrix(rows, metric):
    layers = get_layers(rows)
    proteins = get_proteins(rows)
    lookup = rows_to_lookup(rows, ["protein_id", "layer"])
    matrix = np.full((len(proteins), len(layers)), np.nan)
    for p_idx, protein_id in enumerate(proteins):
        for l_idx, layer in enumerate(layers):
            row = lookup.get((protein_id, layer))
            if row is not None:
                matrix[p_idx, l_idx] = row[metric]
    return proteins, layers, matrix


def plot_protein_lines(rows, metric, title, ylabel, out_path, zero_line=False):
    proteins, layers, matrix = build_matrix(rows, metric)
    fig, ax = plt.subplots(figsize=(11, 6))
    for idx, protein_id in enumerate(proteins):
        ax.plot(
            layers,
            matrix[idx],
            color=PLOT_PROTEIN_COLORS[idx % len(PLOT_PROTEIN_COLORS)],
            linewidth=1.0,
            alpha=0.6,
        )
    mean_vals = np.nanmean(matrix, axis=0)
    ax.plot(layers, mean_vals, color="black", linewidth=3.0, label="Protein mean")
    if zero_line:
        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def layer47_rows(rows):
    return [row for row in rows if int(row["layer"]) == 47]


def label_top_outliers(ax, xs, ys, labels, n_labels=4):
    diffs = np.abs(np.asarray(ys) - np.asarray(xs))
    order = np.argsort(diffs)[::-1][: min(n_labels, len(diffs))]
    for idx in order:
        ax.annotate(
            labels[idx],
            (xs[idx], ys[idx]),
            fontsize=7,
            xytext=(4, 4),
            textcoords="offset points",
        )


def plot_layer47_train_vs_test_scatter(gap_rows, out_path):
    subset = [row for row in gap_rows if int(row["layer"]) == 47]
    xs = np.array([row["test_r2"] for row in subset], dtype=float)
    ys = np.array([row["train_mean_r2"] for row in subset], dtype=float)
    labels = [row["protein_id"] for row in subset]
    corr = safe_corr(xs, ys)

    fig, ax = plt.subplots(figsize=(6.5, 6.0))
    ax.scatter(xs, ys, s=55, alpha=0.8, color="#2a9d8f", edgecolors="black", linewidths=0.5)
    low = min(np.nanmin(xs), np.nanmin(ys))
    high = max(np.nanmax(xs), np.nanmax(ys))
    ax.plot([low, high], [low, high], color="gray", linestyle="--", linewidth=1.2)
    label_top_outliers(ax, xs, ys, labels)
    ax.set_title(
        "Layer 47: Train-Mean vs Test Per-Protein R^2\n"
        f"Pearson={format_float(corr['pearson_r'], 3)}  Spearman={format_float(corr['spearman_r'], 3)}"
    )
    ax.set_xlabel("Test per-protein R^2")
    ax.set_ylabel("Train-mean per-protein R^2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_fold_panel_scatter(rows, x_key, y_key, out_path, title, x_label, y_label):
    subset = layer47_rows(rows)
    folds = sorted({int(row["fold"]) for row in subset})
    x_vals = np.array([row[x_key] for row in subset], dtype=float)
    y_vals = np.array([row[y_key] for row in subset], dtype=float)
    x_pad = 0.05 * (np.nanmax(x_vals) - np.nanmin(x_vals) + 1e-9)
    y_pad = 0.05 * (np.nanmax(y_vals) - np.nanmin(y_vals) + 1e-9)

    fig, axes = plt.subplots(1, len(folds), figsize=(4.2 * len(folds), 4.2), sharex=True, sharey=True)
    if len(folds) == 1:
        axes = [axes]

    for ax, fold in zip(axes, folds):
        fold_rows = [row for row in subset if int(row["fold"]) == fold]
        for split, marker in [("train", "o"), ("test", "^")]:
            split_rows = [row for row in fold_rows if row["split"] == split]
            xs = np.array([row[x_key] for row in split_rows], dtype=float)
            ys = np.array([row[y_key] for row in split_rows], dtype=float)
            if xs.size == 0:
                continue
            ax.scatter(
                xs,
                ys,
                s=30 if split == "train" else 55,
                alpha=0.75,
                marker=marker,
                color=SPLIT_COLORS[split],
                edgecolors="black",
                linewidths=0.4,
                label=split.capitalize(),
            )
        corr = safe_corr(
            [row[x_key] for row in fold_rows],
            [row[y_key] for row in fold_rows],
        )
        ax.set_title(
            f"Fold {fold}\n"
            f"Pearson={format_float(corr['pearson_r'], 3)}  "
            f"Spearman={format_float(corr['spearman_r'], 3)}"
        )
        ax.grid(True, alpha=0.3)
        ax.set_xlim(np.nanmin(x_vals) - x_pad, np.nanmax(x_vals) + x_pad)
        ax.set_ylim(np.nanmin(y_vals) - y_pad, np.nanmax(y_vals) + y_pad)

    axes[0].set_ylabel(y_label)
    for ax in axes:
        ax.set_xlabel(x_label)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    fig.suptitle(title, y=1.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180, bbox_inches="tight")
    plt.close(fig)


def compute_within_slice_correlations(rows, x_key, y_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[(int(row["fold"]), int(row["layer"]), row["split"])].append(row)

    output = []
    for (fold, layer, split), bucket in sorted(grouped.items()):
        corr = safe_corr(
            [row[x_key] for row in bucket],
            [row[y_key] for row in bucket],
        )
        output.append({
            "fold": fold,
            "layer": layer,
            "split": split,
            "n_points": corr["n"],
            "pearson_r": corr["pearson_r"],
            "pearson_p": corr["pearson_p"],
            "spearman_r": corr["spearman_r"],
            "spearman_p": corr["spearman_p"],
        })
    return output


def summarize_corr_rows(rows):
    summary = {}
    for key in ["pearson_r", "spearman_r"]:
        vals = np.array([row[key] for row in rows], dtype=float)
        vals = vals[~np.isnan(vals)]
        if vals.size == 0:
            summary[f"{key}_mean"] = math.nan
            summary[f"{key}_median"] = math.nan
            summary[f"{key}_std"] = math.nan
            summary[f"{key}_fraction_positive"] = math.nan
            summary[f"{key}_count"] = 0
            continue
        summary[f"{key}_mean"] = float(np.mean(vals))
        summary[f"{key}_median"] = float(np.median(vals))
        summary[f"{key}_std"] = float(np.std(vals))
        summary[f"{key}_fraction_positive"] = float(np.mean(vals > 0))
        summary[f"{key}_count"] = int(vals.size)
    return summary


def plot_correlation_distribution(rows, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.5))
    for ax, metric_key, label in [
        (axes[0], "pearson_r", "Pearson r"),
        (axes[1], "spearman_r", "Spearman r"),
    ]:
        for split in ["train", "test"]:
            vals = np.array(
                [row[metric_key] for row in rows if row["split"] == split],
                dtype=float,
            )
            vals = vals[~np.isnan(vals)]
            if vals.size == 0:
                continue
            ax.hist(vals, bins=14, alpha=0.5, color=SPLIT_COLORS[split], label=split.capitalize())
        ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_xlabel(label)
        ax.set_ylabel("Count")
        ax.grid(True, alpha=0.3)
    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_correlation_by_layer(rows, out_path, title):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.8), sharex=True)
    for ax, metric_key, label in [
        (axes[0], "pearson_r", "Pearson r"),
        (axes[1], "spearman_r", "Spearman r"),
    ]:
        for split in ["train", "test"]:
            split_rows = [row for row in rows if row["split"] == split]
            for fold in sorted({row["fold"] for row in split_rows}):
                fold_rows = [row for row in split_rows if row["fold"] == fold]
                fold_rows.sort(key=lambda row: row["layer"])
                ax.plot(
                    [row["layer"] for row in fold_rows],
                    [row[metric_key] for row in fold_rows],
                    color=SPLIT_COLORS[split],
                    linewidth=1.0,
                    alpha=0.2,
                )

            layer_means = []
            layers = sorted({row["layer"] for row in split_rows})
            for layer in layers:
                vals = np.array(
                    [row[metric_key] for row in split_rows if row["layer"] == layer],
                    dtype=float,
                )
                vals = vals[~np.isnan(vals)]
                layer_means.append(float(np.mean(vals)) if vals.size else math.nan)
            ax.plot(
                layers,
                layer_means,
                color=SPLIT_COLORS[split],
                linewidth=3.0,
                label=f"{split.capitalize()} mean",
            )

        ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
        ax.set_title(label)
        ax.set_xlabel("Layer")
        ax.grid(True, alpha=0.3)
    axes[0].set_ylabel("Correlation")
    axes[0].legend(loc="best")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def compute_pooled_corr(rows, x_key, y_key):
    corr = safe_corr([row[x_key] for row in rows], [row[y_key] for row in rows])
    return corr


def compute_test_only_size_r2_by_layer(test_rows):
    output = []
    for layer in get_layers(test_rows):
        layer_rows = [row for row in test_rows if int(row["layer"]) == layer]
        corr = safe_corr(
            [row["n_residues"] for row in layer_rows],
            [row["per_protein_r2"] for row in layer_rows],
        )
        output.append({
            "layer": layer,
            "n_points": corr["n"],
            "pearson_r": corr["pearson_r"],
            "pearson_p": corr["pearson_p"],
            "spearman_r": corr["spearman_r"],
            "spearman_p": corr["spearman_p"],
        })
    return output


def plot_test_only_size_r2_by_layer(rows, out_path):
    layers = [row["layer"] for row in rows]
    pearson = [row["pearson_r"] for row in rows]
    spearman = [row["spearman_r"] for row in rows]
    fig, ax = plt.subplots(figsize=(10.5, 5.2))
    ax.plot(layers, pearson, linewidth=2.2, label="Pearson")
    ax.plot(layers, spearman, linewidth=2.2, label="Spearman")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title("Test Only: Correlation of Protein Size with Per-Protein R^2")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Correlation")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_test_only_layer47_size_r2(test_rows, out_path):
    subset = layer47_rows(test_rows)
    xs = np.array([row["n_residues"] for row in subset], dtype=float)
    ys = np.array([row["per_protein_r2"] for row in subset], dtype=float)
    corr = safe_corr(xs, ys)
    fig, ax = plt.subplots(figsize=(6.5, 5.6))
    ax.scatter(xs, ys, s=50, alpha=0.8, color="#4c78a8", edgecolors="black", linewidths=0.5)
    ax.set_title(
        "Layer 47 Test Only: Protein Size vs Per-Protein R^2\n"
        f"Pearson={format_float(corr['pearson_r'], 3)}  Spearman={format_float(corr['spearman_r'], 3)}"
    )
    ax.set_xlabel("Number of residues")
    ax.set_ylabel("Per-protein R^2")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def design_matrix(rows, include_r2, include_size, include_split, include_layer):
    cols = []
    names = []
    if include_r2:
        cols.append(np.array([row["per_protein_r2"] for row in rows], dtype=float))
        names.append("per_protein_r2")
    if include_size:
        cols.append(np.array([row["n_residues"] for row in rows], dtype=float))
        names.append("n_residues")
    if include_split:
        cols.append(np.array([1.0 if row["split"] == "train" else 0.0 for row in rows], dtype=float))
        names.append("split_train")
    if include_layer:
        layers = sorted({int(row["layer"]) for row in rows})
        base = layers[0]
        for layer in layers[1:]:
            cols.append(np.array([1.0 if int(row["layer"]) == layer else 0.0 for row in rows], dtype=float))
            names.append(f"layer_{layer}_vs_{base}")

    X = np.column_stack(cols) if cols else np.empty((len(rows), 0), dtype=float)
    return X, names


def fit_linear_model(rows, include_r2, include_size, include_split, include_layer):
    y = np.array([row["tm_score"] for row in rows], dtype=float)
    X, feature_names = design_matrix(rows, include_r2, include_size, include_split, include_layer)
    mask = ~(np.isnan(y))
    if X.size:
        mask &= ~np.isnan(X).any(axis=1)
    y = y[mask]
    X = X[mask]

    if y.size < 3:
        raise ValueError("Not enough valid rows to fit linear model")

    model = LinearRegression()
    model.fit(X, y)
    y_pred = model.predict(X)
    model_r2 = float(1.0 - np.sum((y - y_pred) ** 2) / np.sum((y - np.mean(y)) ** 2))

    coef_map = {name: math.nan for name in feature_names}
    std_coef_map = {name: math.nan for name in feature_names}
    if feature_names:
        for name, coef in zip(feature_names, model.coef_):
            coef_map[name] = float(coef)
        scales = np.std(X, axis=0)
        for idx, name in enumerate(feature_names):
            std_coef_map[name] = float(model.coef_[idx] * scales[idx])

    layer_std_coefs = [
        abs(std_coef_map[name])
        for name in feature_names
        if name.startswith("layer_") and not math.isnan(std_coef_map[name])
    ]
    return {
        "n_rows": int(y.size),
        "feature_names": feature_names,
        "model_r2": model_r2,
        "coef_map": coef_map,
        "std_coef_map": std_coef_map,
        "layer_coef_mean_abs_std": (
            float(np.mean(layer_std_coefs)) if layer_std_coefs else math.nan
        ),
        "layer_coef_max_abs_std": (
            float(np.max(layer_std_coefs)) if layer_std_coefs else math.nan
        ),
    }


def build_model_summary_rows(rows, scope, include_layer_for_adjusted):
    configs = [
        ("r2_only", True, False, False, False),
        ("size_only", False, True, False, False),
        ("r2_plus_size", True, True, False, False),
        (
            "adjusted",
            True,
            True,
            True,
            include_layer_for_adjusted,
        ),
    ]

    output = []
    for model_name, include_r2, include_size, include_split, include_layer in configs:
        result = fit_linear_model(
            rows,
            include_r2=include_r2,
            include_size=include_size,
            include_split=include_split,
            include_layer=include_layer,
        )
        output.append({
            "scope": scope,
            "model_name": model_name,
            "n_rows": result["n_rows"],
            "model_r2": result["model_r2"],
            "coef_per_protein_r2": result["coef_map"].get("per_protein_r2", math.nan),
            "coef_n_residues": result["coef_map"].get("n_residues", math.nan),
            "coef_split_train": result["coef_map"].get("split_train", math.nan),
            "std_coef_per_protein_r2": result["std_coef_map"].get("per_protein_r2", math.nan),
            "std_coef_n_residues": result["std_coef_map"].get("n_residues", math.nan),
            "std_coef_split_train": result["std_coef_map"].get("split_train", math.nan),
            "layer_coef_mean_abs_std": result["layer_coef_mean_abs_std"],
            "layer_coef_max_abs_std": result["layer_coef_max_abs_std"],
        })
    return output


def residualize_by_layer(rows, x_key, y_key):
    grouped = defaultdict(list)
    for row in rows:
        grouped[int(row["layer"])].append(row)

    x_resid = []
    y_resid = []
    for layer in sorted(grouped):
        bucket = grouped[layer]
        x_vals = np.array([row[x_key] for row in bucket], dtype=float)
        y_vals = np.array([row[y_key] for row in bucket], dtype=float)
        x_mean = np.nanmean(x_vals)
        y_mean = np.nanmean(y_vals)
        x_resid.extend(list(x_vals - x_mean))
        y_resid.extend(list(y_vals - y_mean))
    return safe_corr(x_resid, y_resid)


def top_gap_rows(gap_rows, key, n=5):
    valid = [row for row in gap_rows if not math.isnan(row[key])]
    return sorted(valid, key=lambda row: abs(row[key]), reverse=True)[:n]


def summarize_gap_by_layer(gap_rows, key):
    layers = get_layers(gap_rows)
    values = []
    for layer in layers:
        bucket = [row[key] for row in gap_rows if int(row["layer"]) == layer]
        arr = np.array(bucket, dtype=float)
        arr = arr[~np.isnan(arr)]
        values.append(float(np.mean(arr)) if arr.size else math.nan)
    return layers, values


def plot_gap_distribution(gap_rows, out_path):
    subset = [row["train_minus_test_r2"] for row in gap_rows if int(row["layer"]) == 47]
    arr = np.array(subset, dtype=float)
    arr = arr[~np.isnan(arr)]
    fig, ax = plt.subplots(figsize=(6.5, 4.8))
    ax.hist(arr, bins=10, color="#6c5ce7", alpha=0.75)
    ax.axvline(0.0, color="gray", linestyle="--", linewidth=1.0)
    ax.set_title("Layer 47: Distribution of Train-Mean Minus Test R^2")
    ax.set_xlabel("Train-mean R^2 - Test R^2")
    ax.set_ylabel("Count")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def run_analysis():
    ensure_dirs()

    validation = validate_outputs()

    updated_mds = convert_rows(load_csv_rows(UPDATED_DIR / "test2_kfold_mds_metrics.csv"))
    updated_train = [row for row in updated_mds if row["split"] == "train"]
    updated_test = [row for row in updated_mds if row["split"] == "test"]

    per_protein_train_mean = aggregate_means(
        updated_train, ["protein_id", "layer"], AGG_METRICS
    )
    per_protein_all_mean = aggregate_means(
        updated_mds, ["protein_id", "layer"], AGG_METRICS
    )
    per_protein_test = aggregate_means(
        updated_test, ["protein_id", "layer"], AGG_METRICS
    )
    gap_rows = build_gap_rows(per_protein_train_mean, per_protein_test)

    data_checks = {
        "unique_proteins": len(get_proteins(updated_mds)),
        "train_rows": len(updated_train),
        "test_rows": len(updated_test),
        "train_rows_per_protein": len(updated_train) / max(len(get_proteins(updated_mds)), 1),
        "test_rows_per_protein": len(updated_test) / max(len(get_proteins(updated_mds)), 1),
        "aggregated_rows_per_layer": len([row for row in per_protein_all_mean if int(row["layer"]) == 0]),
    }

    tm_r2_pooled = compute_pooled_corr(updated_mds, "per_protein_r2", "tm_score")
    tm_size_pooled = compute_pooled_corr(updated_mds, "n_residues", "tm_score")
    triangle_tm_pooled = compute_pooled_corr(
        updated_mds, "triangle_violation_fraction", "tm_score"
    )
    triangle_r2_pooled = compute_pooled_corr(
        updated_mds, "triangle_violation_fraction", "per_protein_r2"
    )
    residual_tm_r2 = residualize_by_layer(updated_mds, "per_protein_r2", "tm_score")

    tm_r2_slice = compute_within_slice_correlations(updated_mds, "per_protein_r2", "tm_score")
    tm_size_slice = compute_within_slice_correlations(updated_mds, "n_residues", "tm_score")
    triangle_tm_slice = compute_within_slice_correlations(
        updated_mds, "triangle_violation_fraction", "tm_score"
    )
    triangle_r2_slice = compute_within_slice_correlations(
        updated_mds, "triangle_violation_fraction", "per_protein_r2"
    )
    test_size_r2_by_layer = compute_test_only_size_r2_by_layer(updated_test)

    layer47_model_rows = build_model_summary_rows(
        layer47_rows(updated_mds), "layer47", include_layer_for_adjusted=False
    )
    all_rows_model_rows = build_model_summary_rows(
        updated_mds, "all_rows", include_layer_for_adjusted=True
    )

    write_csv(
        CSV_DIR / "per_slice_tm_r2_correlations.csv",
        ["fold", "layer", "split", "n_points", "pearson_r", "pearson_p", "spearman_r", "spearman_p"],
        tm_r2_slice,
    )
    write_csv(
        CSV_DIR / "per_slice_tm_size_correlations.csv",
        ["fold", "layer", "split", "n_points", "pearson_r", "pearson_p", "spearman_r", "spearman_p"],
        tm_size_slice,
    )
    write_csv(
        CSV_DIR / "test_only_size_r2_correlations_by_layer.csv",
        ["layer", "n_points", "pearson_r", "pearson_p", "spearman_r", "spearman_p"],
        test_size_r2_by_layer,
    )
    write_csv(
        CSV_DIR / "layer47_joint_model_summary.csv",
        [
            "scope", "model_name", "n_rows", "model_r2",
            "coef_per_protein_r2", "coef_n_residues", "coef_split_train",
            "std_coef_per_protein_r2", "std_coef_n_residues", "std_coef_split_train",
            "layer_coef_mean_abs_std", "layer_coef_max_abs_std",
        ],
        layer47_model_rows,
    )
    write_csv(
        CSV_DIR / "all_rows_joint_model_summary.csv",
        [
            "scope", "model_name", "n_rows", "model_r2",
            "coef_per_protein_r2", "coef_n_residues", "coef_split_train",
            "std_coef_per_protein_r2", "std_coef_n_residues", "std_coef_split_train",
            "layer_coef_mean_abs_std", "layer_coef_max_abs_std",
        ],
        all_rows_model_rows,
    )
    write_csv(
        CSV_DIR / "train_test_r2_gap_summary.csv",
        [
            "protein_id", "layer", "n_residues",
            "train_mean_r2", "test_r2", "train_minus_test_r2",
            "train_mean_tm_score", "test_tm_score", "train_minus_test_tm_score",
            "train_mean_triangle_violation_fraction", "test_triangle_violation_fraction",
            "train_minus_test_triangle_violation_fraction",
        ],
        gap_rows,
    )
    triangle_summary_rows = [
        {
            "relation": "triangle_violation_fraction_vs_tm_score_pooled",
            "scope": "pooled",
            **triangle_tm_pooled,
        },
        {
            "relation": "triangle_violation_fraction_vs_per_protein_r2_pooled",
            "scope": "pooled",
            **triangle_r2_pooled,
        },
    ]
    triangle_summary_rows.extend(
        [{"relation": "triangle_violation_fraction_vs_tm_score_slice", "scope": "within_slice", **summarize_corr_rows(triangle_tm_slice)}]
    )
    triangle_summary_rows.extend(
        [{"relation": "triangle_violation_fraction_vs_per_protein_r2_slice", "scope": "within_slice", **summarize_corr_rows(triangle_r2_slice)}]
    )
    write_csv(
        CSV_DIR / "triangle_metric_correlation_summary.csv",
        sorted({key for row in triangle_summary_rows for key in row.keys()}),
        triangle_summary_rows,
    )

    plot_protein_lines(
        per_protein_train_mean,
        "per_protein_r2",
        "Per-Protein R^2 Across Layers (Train Only)\nMean over each protein's 4 train rows",
        "Per-protein R^2",
        VIS_DIR / "per_protein_train_mean_r2_across_layers.png",
    )
    plot_protein_lines(
        per_protein_all_mean,
        "per_protein_r2",
        "Per-Protein R^2 Across Layers (5-Row Mean)\nMean over each protein's 4 train + 1 test rows",
        "Per-protein R^2",
        VIS_DIR / "per_protein_five_row_mean_r2_across_layers.png",
    )
    plot_protein_lines(
        gap_rows,
        "train_minus_test_r2",
        "Train-Mean Minus Test Per-Protein R^2 Across Layers",
        "Train-mean R^2 - Test R^2",
        VIS_DIR / "per_protein_train_minus_test_r2_across_layers.png",
        zero_line=True,
    )
    plot_layer47_train_vs_test_scatter(
        gap_rows, VIS_DIR / "layer47_train_vs_test_r2_scatter.png"
    )
    plot_gap_distribution(
        gap_rows, VIS_DIR / "layer47_train_minus_test_r2_distribution.png"
    )

    plot_fold_panel_scatter(
        updated_mds,
        "per_protein_r2",
        "tm_score",
        VIS_DIR / "layer47_tm_vs_r2_by_fold.png",
        "Layer 47: TM-score vs Per-Protein R^2 by Fold",
        "Per-protein R^2",
        "TM-score",
    )
    plot_correlation_distribution(
        tm_r2_slice,
        VIS_DIR / "per_slice_tm_vs_r2_correlation_distribution.png",
        "Within-Slice Correlation Distribution: TM-score vs Per-Protein R^2",
    )
    plot_correlation_by_layer(
        tm_r2_slice,
        VIS_DIR / "per_slice_tm_vs_r2_correlation_by_layer.png",
        "Within-Slice Correlation by Layer: TM-score vs Per-Protein R^2",
    )

    plot_fold_panel_scatter(
        updated_mds,
        "n_residues",
        "tm_score",
        VIS_DIR / "layer47_tm_vs_size_by_fold.png",
        "Layer 47: TM-score vs Protein Size by Fold",
        "Number of residues",
        "TM-score",
    )
    plot_correlation_distribution(
        tm_size_slice,
        VIS_DIR / "per_slice_tm_vs_size_correlation_distribution.png",
        "Within-Slice Correlation Distribution: TM-score vs Protein Size",
    )

    plot_protein_lines(
        per_protein_all_mean,
        "frac_top3_eigenvalues",
        "Per-Protein Mean Top-3 Eigenvalue Fraction Across Layers\nMean over each protein's 5 rows per layer",
        "Top-3 eigenvalue fraction",
        VIS_DIR / "per_protein_mean_top3_eigenvalue_fraction_across_layers.png",
    )
    plot_protein_lines(
        per_protein_all_mean,
        "triangle_violation_fraction",
        "Per-Protein Mean Triangle Violation Fraction Across Layers\nMean over each protein's 5 rows per layer",
        "Triangle violation fraction",
        VIS_DIR / "per_protein_mean_triangle_violation_fraction_across_layers.png",
    )
    plot_protein_lines(
        per_protein_all_mean,
        "triangle_mean_violation",
        "Per-Protein Mean Triangle Mean Violation Across Layers\nMean over each protein's 5 rows per layer",
        "Triangle mean violation",
        VIS_DIR / "per_protein_mean_triangle_mean_violation_across_layers.png",
    )
    plot_protein_lines(
        per_protein_all_mean,
        "triangle_max_violation",
        "Per-Protein Mean Triangle Max Violation Across Layers\nMean over each protein's 5 rows per layer",
        "Triangle max violation",
        VIS_DIR / "per_protein_mean_triangle_max_violation_across_layers.png",
    )
    plot_protein_lines(
        gap_rows,
        "train_minus_test_tm_score",
        "Train-Mean Minus Test TM-score Across Layers",
        "Train-mean TM-score - Test TM-score",
        VIS_DIR / "per_protein_train_minus_test_tm_score_across_layers.png",
        zero_line=True,
    )
    plot_protein_lines(
        gap_rows,
        "train_minus_test_triangle_violation_fraction",
        "Train-Mean Minus Test Triangle Violation Fraction Across Layers",
        "Train-mean triangle violation fraction - Test",
        VIS_DIR / "per_protein_train_minus_test_triangle_violation_fraction_across_layers.png",
        zero_line=True,
    )

    plot_test_only_layer47_size_r2(
        updated_test, VIS_DIR / "test_only_size_vs_r2_layer47.png"
    )
    plot_test_only_size_r2_by_layer(
        test_size_r2_by_layer,
        VIS_DIR / "test_only_size_vs_r2_correlation_by_layer.png",
    )

    summary_lines = []
    summary_lines.append("# Test2 Updated Delta Validation and Geometry Analysis")
    summary_lines.append("")
    summary_lines.append(
        f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
    )
    summary_lines.append("")
    summary_lines.append("## Inputs")
    summary_lines.append(f"- Updated data: `{UPDATED_DIR}`")
    summary_lines.append(f"- Old reference data: `{OLD_DIR}`")
    summary_lines.append(f"- Output root: `{OUTPUT_ROOT}`")
    summary_lines.append("")
    summary_lines.append("## Validation Note")
    summary_lines.append(
        "- Regression outputs are unchanged: "
        + ", ".join(
            f"`{name}`={'match' if ok else 'DIFF'}"
            for name, ok in validation["exact_results"].items()
        )
    )
    summary_lines.append(
        f"- Old MDS rows: {validation['old_mds_rows']}; updated rows: {validation['updated_mds_rows']}; "
        f"updated test rows: {validation['updated_test_rows']}."
    )
    summary_lines.append(
        "- Old `test2_kfold_mds_metrics.csv` matches the updated `split=test` subset: "
        f"{'yes' if validation['mds_test_match'] else 'no'}."
    )
    summary_lines.append(
        "- Updated `test2_kfold_mds_summary.csv` differs by design because it includes both train and test rows."
    )
    summary_lines.append("")
    summary_lines.append("## Data Integrity")
    summary_lines.append(f"- Unique proteins in updated MDS metrics: {data_checks['unique_proteins']}")
    summary_lines.append(
        f"- Train rows: {data_checks['train_rows']} ({format_float(data_checks['train_rows_per_protein'], 1)} per protein)"
    )
    summary_lines.append(
        f"- Test rows: {data_checks['test_rows']} ({format_float(data_checks['test_rows_per_protein'], 1)} per protein)"
    )
    summary_lines.append(
        f"- Aggregated rows per layer after per-protein averaging: {data_checks['aggregated_rows_per_layer']}"
    )
    summary_lines.append("")
    summary_lines.append("## R^2 Across Layers")
    layer47_gap = [row["train_minus_test_r2"] for row in gap_rows if int(row["layer"]) == 47]
    layer47_gap_arr = np.array(layer47_gap, dtype=float)
    layer47_gap_arr = layer47_gap_arr[~np.isnan(layer47_gap_arr)]
    summary_lines.append(
        "- Train-only and 5-row-mean protein trajectories were plotted separately so the averaged view does not hide split effects."
    )
    summary_lines.append(
        "- Layer 47 train-test gap: "
        f"mean={format_float(np.mean(layer47_gap_arr), 4)}, "
        f"median={format_float(np.median(layer47_gap_arr), 4)}, "
        f"std={format_float(np.std(layer47_gap_arr), 4)}."
    )
    top_gap = top_gap_rows([row for row in gap_rows if int(row["layer"]) == 47], "train_minus_test_r2", n=3)
    if top_gap:
        summary_lines.append(
            "- Largest absolute layer-47 train-test R^2 gaps: "
            + ", ".join(
                f"{row['protein_id']} ({format_float(row['train_minus_test_r2'], 4)})"
                for row in top_gap
            )
            + "."
        )
    summary_lines.append("")
    summary_lines.append("## TM-score vs R^2")
    summary_lines.append(
        "- Pooled train+test correlation is descriptive only because the rows are repeated measurements across folds and layers."
    )
    tm_r2_slice_summary = summarize_corr_rows(tm_r2_slice)
    summary_lines.append(
        "- Pooled correlation: "
        f"Pearson={format_float(tm_r2_pooled['pearson_r'], 4)}, "
        f"Spearman={format_float(tm_r2_pooled['spearman_r'], 4)} "
        f"(n={tm_r2_pooled['n']})."
    )
    summary_lines.append(
        "- Within-slice correlation (safer): "
        f"Pearson mean={format_float(tm_r2_slice_summary['pearson_r_mean'], 4)}, "
        f"median={format_float(tm_r2_slice_summary['pearson_r_median'], 4)}, "
        f"positive fraction={format_float(tm_r2_slice_summary['pearson_r_fraction_positive'], 3)}; "
        f"Spearman mean={format_float(tm_r2_slice_summary['spearman_r_mean'], 4)}."
    )
    summary_lines.append(
        "- Residualized by layer (subtracting per-layer means): "
        f"Pearson={format_float(residual_tm_r2['pearson_r'], 4)}, "
        f"Spearman={format_float(residual_tm_r2['spearman_r'], 4)}."
    )
    summary_lines.append("")
    summary_lines.append("## TM-score vs Protein Size")
    tm_size_slice_summary = summarize_corr_rows(tm_size_slice)
    summary_lines.append(
        "- Pooled correlation: "
        f"Pearson={format_float(tm_size_pooled['pearson_r'], 4)}, "
        f"Spearman={format_float(tm_size_pooled['spearman_r'], 4)} "
        f"(n={tm_size_pooled['n']})."
    )
    summary_lines.append(
        "- Within-slice correlation: "
        f"Pearson mean={format_float(tm_size_slice_summary['pearson_r_mean'], 4)}, "
        f"Spearman mean={format_float(tm_size_slice_summary['spearman_r_mean'], 4)}."
    )
    summary_lines.append("")
    summary_lines.append("## Joint TM-score Models")
    model_lookup = {
        (row["scope"], row["model_name"]): row
        for row in (layer47_model_rows + all_rows_model_rows)
    }
    summary_lines.append(
        "- Layer 47 model R^2: "
        f"`R2-only`={format_float(model_lookup[('layer47', 'r2_only')]['model_r2'], 4)}, "
        f"`size-only`={format_float(model_lookup[('layer47', 'size_only')]['model_r2'], 4)}, "
        f"`R2+size`={format_float(model_lookup[('layer47', 'r2_plus_size')]['model_r2'], 4)}, "
        f"`adjusted`={format_float(model_lookup[('layer47', 'adjusted')]['model_r2'], 4)}."
    )
    summary_lines.append(
        "- All-row model R^2: "
        f"`R2-only`={format_float(model_lookup[('all_rows', 'r2_only')]['model_r2'], 4)}, "
        f"`size-only`={format_float(model_lookup[('all_rows', 'size_only')]['model_r2'], 4)}, "
        f"`R2+size`={format_float(model_lookup[('all_rows', 'r2_plus_size')]['model_r2'], 4)}, "
        f"`adjusted`={format_float(model_lookup[('all_rows', 'adjusted')]['model_r2'], 4)}."
    )
    summary_lines.append(
        "- The adjusted all-row model includes layer and split controls only as a confounder check, not as scientific predictors."
    )
    summary_lines.append("")
    summary_lines.append("## Geometry Trends")
    triangle_tm_slice_summary = summarize_corr_rows(triangle_tm_slice)
    triangle_r2_slice_summary = summarize_corr_rows(triangle_r2_slice)
    summary_lines.append(
        "- Top-3 eigenvalue fraction, triangle violation fraction, and triangle violation severity were plotted as per-protein 5-row means across layers."
    )
    summary_lines.append(
        "- Triangle violation fraction vs TM-score: "
        f"pooled Pearson={format_float(triangle_tm_pooled['pearson_r'], 4)}, "
        f"within-slice Pearson mean={format_float(triangle_tm_slice_summary['pearson_r_mean'], 4)}."
    )
    summary_lines.append(
        "- Triangle violation fraction vs per-protein R^2: "
        f"pooled Pearson={format_float(triangle_r2_pooled['pearson_r'], 4)}, "
        f"within-slice Pearson mean={format_float(triangle_r2_slice_summary['pearson_r_mean'], 4)}."
    )
    summary_lines.append("")
    summary_lines.append("## Test-Only Size vs Held-Out R^2")
    layer47_test_corr = safe_corr(
        [row["n_residues"] for row in layer47_rows(updated_test)],
        [row["per_protein_r2"] for row in layer47_rows(updated_test)],
    )
    mean_test_size_r2 = summarize_corr_rows(test_size_r2_by_layer)
    summary_lines.append(
        "- Layer 47 test-only correlation: "
        f"Pearson={format_float(layer47_test_corr['pearson_r'], 4)}, "
        f"Spearman={format_float(layer47_test_corr['spearman_r'], 4)}."
    )
    summary_lines.append(
        "- Across-layer test-only correlation means: "
        f"Pearson={format_float(mean_test_size_r2['pearson_r_mean'], 4)}, "
        f"Spearman={format_float(mean_test_size_r2['spearman_r_mean'], 4)}."
    )
    summary_lines.append("")
    summary_lines.append("## Key Outputs")
    for rel_path in [
        VIS_DIR / "per_protein_train_mean_r2_across_layers.png",
        VIS_DIR / "per_protein_five_row_mean_r2_across_layers.png",
        VIS_DIR / "per_protein_train_minus_test_r2_across_layers.png",
        VIS_DIR / "layer47_tm_vs_r2_by_fold.png",
        VIS_DIR / "layer47_tm_vs_size_by_fold.png",
        VIS_DIR / "per_protein_mean_top3_eigenvalue_fraction_across_layers.png",
        VIS_DIR / "test_only_size_vs_r2_layer47.png",
    ]:
        summary_lines.append(f"- `{rel_path}`")
    summary_lines.append("")
    summary_lines.append("## Suggested Follow-Up")
    summary_lines.append(
        "- If the residualized TM-score vs R^2 correlation stays sizable, test whether specific proteins drive the effect by repeating the analysis leave-one-protein-out."
    )
    summary_lines.append(
        "- If size effects look nonlinear in the scatter plots, add a follow-up sensitivity model with `log(n_residues)`."
    )

    SUMMARY_PATH.write_text("\n".join(summary_lines) + "\n")

    return {
        "validation": validation,
        "summary_path": SUMMARY_PATH,
        "output_root": OUTPUT_ROOT,
    }


def main():
    result = run_analysis()
    print(f"Saved analysis outputs under: {result['output_root']}")
    print(f"Saved summary: {result['summary_path']}")
    print(
        "Validation status: "
        + ", ".join(
            f"{name}={'match' if ok else 'DIFF'}"
            for name, ok in result["validation"]["exact_results"].items()
        )
        + f"; mds_test_match={'yes' if result['validation']['mds_test_match'] else 'no'}"
    )


if __name__ == "__main__":
    main()

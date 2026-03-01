#!/usr/bin/env python3
"""
Create basic R^2 visualizations from test Delta k-fold regression CSV files.

Inputs:
  - delta_files/test_linear_regression_csv_files/test_kfold_regression_r2.csv
  - delta_files/test_linear_regression_csv_files/test_kfold_regression_r2_summary.csv

Outputs:
  - proteins/multi_protein_results/test delta files/visualizations/test_kfold_r2_per_fold_and_mean.png
  - proteins/multi_protein_results/test delta files/visualizations/test_kfold_r2_mean_std.png
"""

import csv
from collections import defaultdict
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / "delta_files" / "test_linear_regression_csv_files"
OUTPUT_DIR = PROJECT_ROOT / "proteins" / "multi_protein_results" / "test delta files"
VIS_DIR = OUTPUT_DIR / "visualizations"


def load_fold_r2(csv_path):
    fold_points = defaultdict(list)
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"fold", "layer", "r2_test"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            fold = int(row["fold"])
            layer = int(row["layer"])
            r2_test = float(row["r2_test"])
            fold_points[fold].append((layer, r2_test))

    if not fold_points:
        raise ValueError(f"No rows found in {csv_path}")

    fold_data = {}
    for fold, points in fold_points.items():
        points.sort(key=lambda x: x[0])
        layers = np.array([p[0] for p in points], dtype=int)
        r2_values = np.array([p[1] for p in points], dtype=float)
        fold_data[fold] = {"layers": layers, "r2": r2_values}

    return fold_data


def load_r2_summary(csv_path):
    layers = []
    means = []
    stds = []
    with csv_path.open("r", newline="") as f:
        reader = csv.DictReader(f)
        required = {"layer", "r2_test_mean", "r2_test_std"}
        missing = required.difference(set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"Missing required columns in {csv_path}: {sorted(missing)}")

        for row in reader:
            layers.append(int(row["layer"]))
            means.append(float(row["r2_test_mean"]))
            stds.append(float(row["r2_test_std"]))

    if not layers:
        raise ValueError(f"No rows found in {csv_path}")

    order = np.argsort(np.array(layers))
    layers = np.array(layers, dtype=int)[order]
    means = np.array(means, dtype=float)[order]
    stds = np.array(stds, dtype=float)[order]
    return layers, means, stds


def plot_per_fold_and_mean(fold_data, mean_layers, mean_r2, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.5))

    for fold in sorted(fold_data):
        ax.plot(
            fold_data[fold]["layers"],
            fold_data[fold]["r2"],
            linewidth=1.8,
            alpha=0.85,
            label=f"Fold {fold}",
        )

    ax.plot(
        mean_layers,
        mean_r2,
        color="black",
        linewidth=2.8,
        label="Mean across folds",
    )
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Test Delta 3-Fold CV: R^2 Across Layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R^2 (test)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_mean_with_std(layers, mean_r2, std_r2, out_path):
    fig, ax = plt.subplots(figsize=(10, 5.2))
    lower = mean_r2 - std_r2
    upper = mean_r2 + std_r2

    ax.fill_between(layers, lower, upper, alpha=0.2, label="Mean ± std")
    ax.plot(layers, mean_r2, linewidth=2.5, label="Mean R^2 (test)")
    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_title("Test Delta 3-Fold CV: Mean R^2 Across Layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("R^2 (test)")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    r2_csv = INPUT_DIR / "test_kfold_regression_r2.csv"
    summary_csv = INPUT_DIR / "test_kfold_regression_r2_summary.csv"

    if not r2_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {r2_csv}")
    if not summary_csv.exists():
        raise FileNotFoundError(f"Missing input CSV: {summary_csv}")

    VIS_DIR.mkdir(parents=True, exist_ok=True)

    fold_data = load_fold_r2(r2_csv)
    layers, mean_r2, std_r2 = load_r2_summary(summary_csv)

    out_per_fold = VIS_DIR / "test_kfold_r2_per_fold_and_mean.png"
    out_mean_std = VIS_DIR / "test_kfold_r2_mean_std.png"

    plot_per_fold_and_mean(fold_data, layers, mean_r2, out_per_fold)
    plot_mean_with_std(layers, mean_r2, std_r2, out_mean_std)

    print(f"Saved: {out_per_fold}")
    print(f"Saved: {out_mean_std}")


if __name__ == "__main__":
    main()

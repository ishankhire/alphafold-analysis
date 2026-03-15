#!/usr/bin/env python3
"""
Create diagnostic plots from Delta sampling/weighting k-fold regression outputs.

Inputs:
  - delta_files/diagnostic_sampling_weighting_kfold_csv_files/
    - diagnostic_sampling_weighting_kfold_variant_summary.csv
    - diagnostic_sampling_weighting_kfold_fold_summary.csv
    - diagnostic_sampling_weighting_kfold_per_protein.csv
    - diagnostic_sampling_weighting_kfold_train_sampling_audit.csv

Outputs:
  - delta_files/diagnostic_sampling_weighting_kfold_csv_files/visualizations/
    - variant_summary_mean_std.png
    - per_protein_r2_violin.png
    - per_protein_r2_histograms.png
    - per_protein_r2_qq_normal.png
    - size_bin_r2_profile.png
    - train_sampling_audit.png
"""

import csv
import math
from collections import defaultdict
from pathlib import Path
from statistics import NormalDist

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = Path(__file__).resolve().parents[3]
INPUT_DIR = PROJECT_ROOT / "delta_files" / "diagnostic_sampling_weighting_kfold_csv_files"
VIS_DIR = INPUT_DIR / "visualizations"

VARIANT_ORDER = [
    "all_pairs_protein_equal_stratified_kfold",
    "all_pairs_protein_equal_random_kfold",
    "cap5k_protein_equal_stratified_kfold",
    "cap5k_protein_equal_random_kfold",
    "cap5k_pair_weighted_random_kfold",
]

VARIANT_LABELS = {
    "all_pairs_protein_equal_stratified_kfold": "All Pairs\nProtein-Eq\nStratified",
    "all_pairs_protein_equal_random_kfold": "All Pairs\nProtein-Eq\nRandom",
    "cap5k_protein_equal_stratified_kfold": "Cap 5k\nProtein-Eq\nStratified",
    "cap5k_protein_equal_random_kfold": "Cap 5k\nProtein-Eq\nRandom",
    "cap5k_pair_weighted_random_kfold": "Cap 5k\nPair-Wtd\nRandom",
}

VARIANT_COLORS = {
    "all_pairs_protein_equal_stratified_kfold": "#c0392b",
    "all_pairs_protein_equal_random_kfold": "#e67e22",
    "cap5k_protein_equal_stratified_kfold": "#1f77b4",
    "cap5k_protein_equal_random_kfold": "#2ca02c",
    "cap5k_pair_weighted_random_kfold": "#16a085",
}


def load_csv_rows(path):
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def as_float(row, key):
    return float(row[key])


def load_inputs():
    return {
        "variant_summary": load_csv_rows(INPUT_DIR / "diagnostic_sampling_weighting_kfold_variant_summary.csv"),
        "fold_summary": load_csv_rows(INPUT_DIR / "diagnostic_sampling_weighting_kfold_fold_summary.csv"),
        "per_protein": load_csv_rows(INPUT_DIR / "diagnostic_sampling_weighting_kfold_per_protein.csv"),
        "train_audit": load_csv_rows(INPUT_DIR / "diagnostic_sampling_weighting_kfold_train_sampling_audit.csv"),
    }


def variant_positions():
    return np.arange(len(VARIANT_ORDER), dtype=float)


def plot_variant_summary(summary_rows, out_path):
    rows_by_variant = {row["variant"]: row for row in summary_rows}
    x = variant_positions()
    labels = [VARIANT_LABELS[name] for name in VARIANT_ORDER]
    colors = [VARIANT_COLORS[name] for name in VARIANT_ORDER]

    mean_pp = [as_float(rows_by_variant[name], "mean_test_mean_per_protein_r2") for name in VARIANT_ORDER]
    std_pp = [as_float(rows_by_variant[name], "std_test_mean_per_protein_r2") for name in VARIANT_ORDER]
    mean_pair = [as_float(rows_by_variant[name], "mean_test_pair_weighted_r2") for name in VARIANT_ORDER]
    std_pair = [as_float(rows_by_variant[name], "std_test_pair_weighted_r2") for name in VARIANT_ORDER]

    fig, axes = plt.subplots(2, 1, figsize=(10.5, 8.0), sharex=True)

    axes[0].bar(x, mean_pp, yerr=std_pp, color=colors, alpha=0.9, capsize=6)
    axes[0].axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[0].set_ylabel("Mean test per-protein $R^2$")
    axes[0].set_title("Fold-Mean Performance With Fold-to-Fold Spread")
    axes[0].grid(True, axis="y", alpha=0.25)

    axes[1].bar(x, mean_pair, yerr=std_pair, color=colors, alpha=0.9, capsize=6)
    axes[1].axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    axes[1].set_ylabel("Mean test pair-weighted $R^2$")
    axes[1].grid(True, axis="y", alpha=0.25)

    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_protein_violin(per_rows, out_path):
    values_by_variant = defaultdict(list)
    for row in per_rows:
        values_by_variant[row["variant"]].append(as_float(row, "per_protein_r2"))

    data = [np.asarray(values_by_variant[name], dtype=float) for name in VARIANT_ORDER]
    x = variant_positions()
    rng = np.random.default_rng(0)

    fig, ax = plt.subplots(figsize=(11.0, 6.5))
    violin = ax.violinplot(
        data,
        positions=x,
        widths=0.8,
        showmeans=False,
        showmedians=True,
        showextrema=False,
    )
    for body, variant in zip(violin["bodies"], VARIANT_ORDER):
        body.set_facecolor(VARIANT_COLORS[variant])
        body.set_edgecolor("black")
        body.set_alpha(0.35)
    violin["cmedians"].set_color("black")
    violin["cmedians"].set_linewidth(1.6)

    for xpos, variant, values in zip(x, VARIANT_ORDER, data):
        jitter = rng.uniform(-0.12, 0.12, size=values.size)
        ax.scatter(
            np.full(values.size, xpos) + jitter,
            values,
            s=10,
            alpha=0.22,
            color=VARIANT_COLORS[variant],
            edgecolors="none",
        )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_ylabel("Per-protein test $R^2$")
    ax.set_title("Per-Protein $R^2$ Distribution by Variant")
    ax.set_xticks(x)
    ax.set_xticklabels([VARIANT_LABELS[name] for name in VARIANT_ORDER])
    ax.grid(True, axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_per_protein_histograms(per_rows, out_path):
    values_by_variant = defaultdict(list)
    for row in per_rows:
        values_by_variant[row["variant"]].append(as_float(row, "per_protein_r2"))

    all_values = np.concatenate(
        [np.asarray(values_by_variant[name], dtype=float) for name in VARIANT_ORDER]
    )
    bins = np.linspace(float(np.min(all_values)), float(np.max(all_values)), 34)

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for axis_index, variant in enumerate(VARIANT_ORDER):
        ax = axes[axis_index]
        values = np.asarray(values_by_variant[variant], dtype=float)
        mean_value = float(np.mean(values))
        median_value = float(np.median(values))
        std_value = float(np.std(values))

        ax.hist(values, bins=bins, color=VARIANT_COLORS[variant], alpha=0.85, edgecolor="white")
        ax.axvline(mean_value, color="black", linewidth=1.6, label="mean")
        ax.axvline(median_value, color="black", linestyle="--", linewidth=1.4, label="median")
        ax.axvline(0.0, color="gray", linestyle=":", linewidth=1.0)
        ax.set_title(f"{VARIANT_LABELS[variant].replace(chr(10), ' ')}\nmean={mean_value:.3f}, std={std_value:.3f}")
        ax.grid(True, axis="y", alpha=0.2)

    axes[-1].axis("off")
    axes[0].legend(loc="upper left")
    fig.suptitle("Histogram of Per-Protein Test $R^2$ Values", y=0.98)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def normal_quantiles(n_points):
    dist = NormalDist()
    probs = [(idx + 0.5) / n_points for idx in range(n_points)]
    return np.asarray([dist.inv_cdf(prob) for prob in probs], dtype=float)


def plot_per_protein_qq(per_rows, out_path):
    values_by_variant = defaultdict(list)
    for row in per_rows:
        values_by_variant[row["variant"]].append(as_float(row, "per_protein_r2"))

    fig, axes = plt.subplots(3, 2, figsize=(12.5, 10.0), sharex=True, sharey=True)
    axes = axes.ravel()
    for axis_index, variant in enumerate(VARIANT_ORDER):
        ax = axes[axis_index]
        values = np.sort(np.asarray(values_by_variant[variant], dtype=float))
        mean_value = float(np.mean(values))
        std_value = float(np.std(values))
        if std_value < 1e-12:
            standardized = np.zeros_like(values)
        else:
            standardized = (values - mean_value) / std_value
        theoretical = normal_quantiles(len(values))

        low = min(float(np.min(theoretical)), float(np.min(standardized)))
        high = max(float(np.max(theoretical)), float(np.max(standardized)))
        ax.scatter(theoretical, standardized, s=12, alpha=0.75, color=VARIANT_COLORS[variant])
        ax.plot([low, high], [low, high], color="black", linewidth=1.2, linestyle="--")
        ax.set_title(VARIANT_LABELS[variant].replace("\n", " "))
        ax.grid(True, alpha=0.2)

    axes[-1].axis("off")
    fig.suptitle("Q-Q Plot Against a Matched Gaussian", y=0.98)
    fig.text(0.5, 0.02, "Theoretical normal quantile", ha="center")
    fig.text(0.02, 0.5, "Empirical standardized $R^2$", va="center", rotation="vertical")
    fig.tight_layout(rect=[0.04, 0.04, 1.0, 0.96])
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_size_bin_profile(per_rows, out_path):
    grouped = defaultdict(list)
    for row in per_rows:
        grouped[(row["variant"], int(row["size_bin"]))].append(as_float(row, "per_protein_r2"))

    fig, ax = plt.subplots(figsize=(10.5, 6.0))
    x = np.arange(5, dtype=int)
    for variant in VARIANT_ORDER:
        means = []
        stds = []
        for size_bin in x:
            values = np.asarray(grouped[(variant, int(size_bin))], dtype=float)
            means.append(float(np.mean(values)))
            stds.append(float(np.std(values)))
        ax.errorbar(
            x,
            means,
            yerr=stds,
            marker="o",
            linewidth=2.0,
            capsize=4,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant].replace("\n", " "),
        )

    ax.axhline(0.0, color="gray", linestyle="--", linewidth=1.0, alpha=0.8)
    ax.set_xlabel("Rank-based size bin")
    ax.set_ylabel("Per-protein test $R^2$ (mean ± std)")
    ax.set_title("Performance Profile Across Protein Size Bins")
    ax.set_xticks(x)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_train_sampling_audit(audit_rows, out_path):
    focus_variants = [
        "all_pairs_protein_equal_random_kfold",
        "cap5k_protein_equal_random_kfold",
        "cap5k_pair_weighted_random_kfold",
    ]

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.5))
    for variant in focus_variants:
        available = np.asarray(
            [int(row["n_pairs_available"]) for row in audit_rows if row["variant"] == variant],
            dtype=float,
        )
        used = np.asarray(
            [int(row["n_pairs_used"]) for row in audit_rows if row["variant"] == variant],
            dtype=float,
        )
        total_weight = np.asarray(
            [float(row["protein_total_train_weight"]) for row in audit_rows if row["variant"] == variant],
            dtype=float,
        )

        axes[0].scatter(
            available,
            used,
            s=14,
            alpha=0.35,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant].replace("\n", " "),
        )
        axes[1].scatter(
            available,
            total_weight,
            s=14,
            alpha=0.35,
            color=VARIANT_COLORS[variant],
            label=VARIANT_LABELS[variant].replace("\n", " "),
        )

    max_pairs = max(int(row["n_pairs_available"]) for row in audit_rows)
    ref_x = np.asarray([1.0, float(max_pairs)], dtype=float)
    axes[0].plot(ref_x, ref_x, color="black", linestyle="--", linewidth=1.2, label="all pairs")
    axes[0].axhline(5000.0, color="gray", linestyle=":", linewidth=1.2, label="5k cap")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Train pairs available per protein")
    axes[0].set_ylabel("Train pairs used per protein")
    axes[0].set_title("Sampling Audit")
    axes[0].grid(True, which="both", alpha=0.2)

    axes[1].set_xscale("log")
    axes[1].set_yscale("log")
    axes[1].set_xlabel("Train pairs available per protein")
    axes[1].set_ylabel("Total train weight per protein")
    axes[1].set_title("Weighting Audit")
    axes[1].grid(True, which="both", alpha=0.2)

    axes[1].legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def main():
    if not INPUT_DIR.exists():
        raise FileNotFoundError(f"Missing input directory: {INPUT_DIR}")

    VIS_DIR.mkdir(parents=True, exist_ok=True)
    loaded = load_inputs()

    outputs = [
        VIS_DIR / "variant_summary_mean_std.png",
        VIS_DIR / "per_protein_r2_violin.png",
        VIS_DIR / "per_protein_r2_histograms.png",
        VIS_DIR / "per_protein_r2_qq_normal.png",
        VIS_DIR / "size_bin_r2_profile.png",
        VIS_DIR / "train_sampling_audit.png",
    ]

    plot_variant_summary(loaded["variant_summary"], outputs[0])
    plot_per_protein_violin(loaded["per_protein"], outputs[1])
    plot_per_protein_histograms(loaded["per_protein"], outputs[2])
    plot_per_protein_qq(loaded["per_protein"], outputs[3])
    plot_size_bin_profile(loaded["per_protein"], outputs[4])
    plot_train_sampling_audit(loaded["train_audit"], outputs[5])

    for path in outputs:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot held-out sampled triangle inequality violations for Delta outputs."""

from __future__ import annotations

import csv
import os
import re
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".mplconfig"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANT = "general_base256_combined"
THRESHOLD_ANGSTROM = 2.0
SAMPLES_DIR = (
    ROOT_DIR
    / "delta_files"
    / VARIANT
    / "triangle_violation_samples_test"
)
ANALYSIS_DIR = ROOT_DIR / "delta_files" / VARIANT / "analysis"
GEOMETRY_SUMMARY = ROOT_DIR / "delta_files" / VARIANT / "geometry_summary.csv"

VIOLATION_COLUMNS = ("violation_ab", "violation_ac", "violation_bc")
LAYER_RE = re.compile(r"triangle_samples_layer_(\d+)\.csv$")


def layer_from_sample_path(path: Path) -> int:
    match = LAYER_RE.search(path.name)
    if not match:
        raise ValueError(f"Could not parse layer from {path}")
    return int(match.group(1))


def read_raw_sample_layer(path: Path) -> dict[str, object]:
    positive_chunks: list[np.ndarray] = []
    above_threshold_chunks: list[np.ndarray] = []
    n_valid_rows = 0
    n_skipped_rows = 0

    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if any(row.get(col) in (None, "") for col in VIOLATION_COLUMNS):
                n_skipped_rows += 1
                continue
            values = np.array([float(row[col]) for col in VIOLATION_COLUMNS], dtype=np.float64)
            positives = values[values > 0.0]
            above_threshold = values[values > THRESHOLD_ANGSTROM]
            if positives.size:
                positive_chunks.append(positives)
            if above_threshold.size:
                above_threshold_chunks.append(above_threshold)
            n_valid_rows += 1

    positive = (
        np.concatenate(positive_chunks)
        if positive_chunks
        else np.array([], dtype=np.float64)
    )
    above_threshold = (
        np.concatenate(above_threshold_chunks)
        if above_threshold_chunks
        else np.array([], dtype=np.float64)
    )
    n_inequalities = n_valid_rows * len(VIOLATION_COLUMNS)

    return {
        "layer": layer_from_sample_path(path),
        "n_valid_rows": n_valid_rows,
        "n_skipped_rows": n_skipped_rows,
        "positive": positive,
        "above_threshold": above_threshold,
        "n_inequalities": n_inequalities,
        "positive_fraction": positive.size / n_inequalities if n_inequalities else np.nan,
        "positive_mean": float(positive.mean()) if positive.size else 0.0,
        "threshold_fraction": (
            above_threshold.size / n_inequalities if n_inequalities else np.nan
        ),
        "threshold_mean": (
            float(above_threshold.mean()) if above_threshold.size else 0.0
        ),
    }


def read_layer_sample_summary() -> dict[int, dict[str, str]]:
    with (SAMPLES_DIR / "triangle_sample_summary.csv").open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {
            int(row["layer"]): row
            for row in reader
            if row["scope"] == "layer" and row["split"] == "test"
        }
    return rows


def read_aggregate_summary() -> dict[int, dict[str, str]]:
    with GEOMETRY_SUMMARY.open(newline="") as handle:
        reader = csv.DictReader(handle)
        rows = {
            int(row["layer"]): row
            for row in reader
            if row["variant"] == VARIANT and row["split"] == "test"
        }
    return rows


def plot_positive_violation_distributions(raw_layers: list[dict[str, object]]) -> Path:
    fig, ax = plt.subplots(figsize=(10.5, 6.2))
    colors = plt.get_cmap("viridis")(np.linspace(0.08, 0.92, len(raw_layers)))

    all_positive = np.concatenate(
        [layer["positive"] for layer in raw_layers if len(layer["positive"]) > 0]
    )
    max_violation = float(np.max(all_positive)) if all_positive.size else 1.0
    bins = np.linspace(0.0, max_violation, 72)

    for layer, color in zip(raw_layers, colors):
        values = layer["positive"]
        ax.hist(
            values,
            bins=bins,
            density=True,
            histtype="step",
            linewidth=2.0,
            color=color,
            label=f"L{layer['layer']:02d} (n={len(values):,})",
        )

    ax.axvline(
        THRESHOLD_ANGSTROM,
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"{THRESHOLD_ANGSTROM:g} Å threshold",
    )
    ax.set_title("Positive sampled triangle inequality violation distributions")
    ax.set_xlabel("Violation magnitude (Å)")
    ax.set_ylabel("Density over positive inequality violations")
    ax.grid(alpha=0.25)
    ax.legend(frameon=False, ncol=2)
    fig.tight_layout()

    out_path = ANALYSIS_DIR / f"{VARIANT}_sample_triangle_violation_distributions.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def plot_threshold_summary(
    raw_layers: list[dict[str, object]], excluded_raw_layers: list[int]
) -> Path:
    layers = np.array([layer["layer"] for layer in raw_layers], dtype=int)
    threshold_fraction = np.array(
        [layer["threshold_fraction"] for layer in raw_layers], dtype=np.float64
    )
    threshold_mean = np.array(
        [layer["threshold_mean"] for layer in raw_layers], dtype=np.float64
    )

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.2), sharex=True)
    axes[0].plot(layers, threshold_fraction, marker="o", linewidth=2.2, color="#2f6f9f")
    axes[0].set_ylabel(f"Fraction > {THRESHOLD_ANGSTROM:g} Å")
    axes[0].set_title("Sampled triangle inequality violations after thresholding")
    axes[0].grid(alpha=0.25)

    axes[1].plot(layers, threshold_mean, marker="o", linewidth=2.2, color="#c66a1d")
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel(f"Mean violation among > {THRESHOLD_ANGSTROM:g} Å (Å)")
    axes[1].set_xticks(layers)
    axes[1].grid(alpha=0.25)

    if excluded_raw_layers:
        excluded_text = ", ".join(f"L{layer:02d}" for layer in excluded_raw_layers)
        fig.text(
            0.01,
            0.01,
            f"Excluded incomplete raw sample file(s): {excluded_text}.",
            fontsize=9,
            color="#555555",
        )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    out_path = ANALYSIS_DIR / f"{VARIANT}_sample_triangle_violation_threshold2_vs_layer.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def write_threshold_csv(raw_layers: list[dict[str, object]]) -> Path:
    out_path = ANALYSIS_DIR / f"{VARIANT}_sample_triangle_violation_threshold2_summary.csv"
    fieldnames = [
        "layer",
        "n_valid_sample_rows",
        "n_skipped_sample_rows",
        "n_inequalities",
        "n_positive_violations",
        "positive_fraction",
        "positive_mean",
        "n_violations_gt_2a",
        "violation_fraction_gt_2a",
        "mean_violation_gt_2a",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_layers:
            writer.writerow(
                {
                    "layer": row["layer"],
                    "n_valid_sample_rows": row["n_valid_rows"],
                    "n_skipped_sample_rows": row["n_skipped_rows"],
                    "n_inequalities": row["n_inequalities"],
                    "n_positive_violations": len(row["positive"]),
                    "positive_fraction": f"{row['positive_fraction']:.10g}",
                    "positive_mean": f"{row['positive_mean']:.10g}",
                    "n_violations_gt_2a": len(row["above_threshold"]),
                    "violation_fraction_gt_2a": f"{row['threshold_fraction']:.10g}",
                    "mean_violation_gt_2a": f"{row['threshold_mean']:.10g}",
                }
            )
    return out_path


def build_sample_vs_aggregate_rows() -> list[dict[str, object]]:
    sample_summary = read_layer_sample_summary()
    aggregate_summary = read_aggregate_summary()
    rows: list[dict[str, object]] = []

    for layer in sorted(set(sample_summary) & set(aggregate_summary)):
        sample = sample_summary[layer]
        aggregate = aggregate_summary[layer]
        rows.append(
            {
                "layer": layer,
                "sample_inequality_violation_fraction": float(
                    sample["inequality_violation_fraction"]
                ),
                "aggregate_triangle_violation_fraction": float(
                    aggregate["triangle_violation_fraction_mean"]
                ),
                "sample_mean_positive_violation": float(
                    sample["mean_positive_violation"]
                ),
                "aggregate_triangle_mean_violation": float(
                    aggregate["triangle_mean_violation_mean"]
                ),
            }
        )
    return rows


def write_sample_vs_aggregate_csv(rows: list[dict[str, object]]) -> Path:
    out_path = ANALYSIS_DIR / f"{VARIANT}_sample_vs_aggregate_triangle_summary.csv"
    fieldnames = [
        "layer",
        "sample_inequality_violation_fraction",
        "aggregate_triangle_violation_fraction",
        "sample_mean_positive_violation",
        "aggregate_triangle_mean_violation",
    ]
    with out_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
    return out_path


def plot_sample_vs_aggregate(rows: list[dict[str, object]]) -> Path:
    layers = np.array([row["layer"] for row in rows], dtype=int)

    fig, axes = plt.subplots(2, 1, figsize=(9.5, 7.6), sharex=True)
    axes[0].plot(
        layers,
        [row["sample_inequality_violation_fraction"] for row in rows],
        marker="o",
        linewidth=2.2,
        label="Sample random-triangle inequality fraction",
        color="#2f6f9f",
    )
    axes[0].plot(
        layers,
        [row["aggregate_triangle_violation_fraction"] for row in rows],
        marker="s",
        linewidth=2.2,
        label="Aggregate min-indirect directed-pair fraction",
        color="#8c4a2f",
    )
    axes[0].set_ylabel("Violation fraction")
    axes[0].set_title("Sampled vs aggregate triangle violation statistics")
    axes[0].grid(alpha=0.25)
    axes[0].legend(frameon=False)

    axes[1].plot(
        layers,
        [row["sample_mean_positive_violation"] for row in rows],
        marker="o",
        linewidth=2.2,
        label="Sample mean positive inequality violation",
        color="#2f6f9f",
    )
    axes[1].plot(
        layers,
        [row["aggregate_triangle_mean_violation"] for row in rows],
        marker="s",
        linewidth=2.2,
        label="Aggregate mean min-indirect violation",
        color="#8c4a2f",
    )
    axes[1].set_xlabel("Layer")
    axes[1].set_ylabel("Mean violation (Å)")
    axes[1].set_xticks(layers)
    axes[1].grid(alpha=0.25)
    axes[1].legend(frameon=False)

    fig.text(
        0.01,
        0.01,
        "Caveat: sample and aggregate series use different estimators; compare trends and scale, not exact equality.",
        fontsize=9,
        color="#555555",
    )
    fig.tight_layout(rect=(0, 0.035, 1, 1))

    out_path = ANALYSIS_DIR / f"{VARIANT}_sample_vs_aggregate_triangle_violations.png"
    fig.savefig(out_path, dpi=220)
    plt.close(fig)
    return out_path


def main() -> None:
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    (ROOT_DIR / ".mplconfig").mkdir(parents=True, exist_ok=True)
    sample_summary = read_layer_sample_summary()

    sample_paths = sorted(
        SAMPLES_DIR.glob("triangle_samples_layer_*.csv"), key=layer_from_sample_path
    )
    if not sample_paths:
        raise FileNotFoundError(f"No raw sample CSVs found under {SAMPLES_DIR}")

    raw_candidates = [read_raw_sample_layer(path) for path in sample_paths]
    raw_layers: list[dict[str, object]] = []
    excluded_raw_layers: list[int] = []
    for raw_layer in raw_candidates:
        layer = int(raw_layer["layer"])
        expected_rows = int(sample_summary.get(layer, {}).get("n_triangles_sampled", 0))
        if expected_rows and int(raw_layer["n_valid_rows"]) != expected_rows:
            excluded_raw_layers.append(layer)
            continue
        raw_layers.append(raw_layer)

    if not raw_layers:
        raise RuntimeError("No complete raw sample CSVs were available for plotting")

    comparison_rows = build_sample_vs_aggregate_rows()

    outputs = [
        plot_positive_violation_distributions(raw_layers),
        plot_threshold_summary(raw_layers, excluded_raw_layers),
        write_threshold_csv(raw_layers),
        write_sample_vs_aggregate_csv(comparison_rows),
        plot_sample_vs_aggregate(comparison_rows),
    ]

    print("Generated triangle violation sample analysis artifacts:")
    for path in outputs:
        print(f"- {path.relative_to(ROOT_DIR)}")


if __name__ == "__main__":
    main()

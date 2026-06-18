#!/usr/bin/env python3
"""Merge and plot exact triangle metrics for general_base256_combined."""

from __future__ import annotations

import filecmp
import os
import re
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.makedirs(PROJECT_ROOT / ".mplconfig", exist_ok=True)
os.makedirs(PROJECT_ROOT / ".cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


VARIANT = "general_base256_combined"
EXPECTED_LAYERS = tuple(range(48))
EXPECTED_PROTEINS_PER_LAYER = 552
SPLIT_ID = 0

EXACT_DIR = PROJECT_ROOT / "delta_files" / VARIANT / "exact_triangle_metrics"
LAYER_BATCH_DIR = EXACT_DIR / "layer_batches"
ANALYSIS_DIR = PROJECT_ROOT / "delta_files" / VARIANT / "analysis"

COMBINED_CSV = EXACT_DIR / "exact_triangle_metrics.csv"
LAYER_MEANS_CSV = ANALYSIS_DIR / f"{VARIANT}_exact_triangle_layer_means.csv"
PROPORTION_PNG = ANALYSIS_DIR / f"{VARIANT}_exact_triangle_violation_proportions.png"
MEAN_VIOLATION_PNG = ANALYSIS_DIR / f"{VARIANT}_exact_triangle_mean_violation.png"
ANALYSIS_MD = ANALYSIS_DIR / f"{VARIANT}_exact_triangle_metrics_analysis.md"

LAYER_FILE_RE = re.compile(r"exact_triangle_metrics_layer_(\d+)\.csv$")
VALIDATION_TOLERANCE = 5e-5

REQUIRED_COLUMNS = {
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "n_triangles",
    "n_negative_predictions",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
}

PROPORTION_COLUMNS = [
    "triangle_violation_fraction",
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]

CATEGORY_COLUMNS = [
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]

VIOLATION_BIN_COLUMNS = [
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]

BLUE = "#4c78a8"
ORANGE = "#f58518"
GREEN = "#54a24b"
RED = "#e45756"
PURPLE = "#7f6edb"
GRID_GREY = "#dddddd"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "figure.titlesize": 17,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def parse_layer(path: Path) -> int:
    match = LAYER_FILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Could not parse layer from {path}")
    return int(match.group(1))


def map_layer_files(paths: list[Path]) -> dict[int, Path]:
    mapped: dict[int, Path] = {}
    for path in paths:
        layer = parse_layer(path)
        if layer in mapped:
            raise ValueError(f"Duplicate layer {layer} files: {mapped[layer]} and {path}")
        mapped[layer] = path
    return mapped


def find_layer_shards() -> list[Path]:
    top_level = map_layer_files(sorted(EXACT_DIR.glob("exact_triangle_metrics_layer_*.csv")))
    batched = map_layer_files(sorted(LAYER_BATCH_DIR.glob("exact_triangle_metrics_layer_*.csv")))
    expected = set(EXPECTED_LAYERS)

    if set(top_level) == expected:
        return [top_level[layer] for layer in EXPECTED_LAYERS]
    if set(batched) == expected:
        return [batched[layer] for layer in EXPECTED_LAYERS]

    combined = dict(batched)
    for layer, path in top_level.items():
        if layer in combined:
            raise ValueError(
                f"Layer {layer} exists in both {EXACT_DIR} and {LAYER_BATCH_DIR}; "
                "remove one copy or rerun after cleaning duplicates."
            )
        combined[layer] = path

    if set(combined) == expected:
        return [combined[layer] for layer in EXPECTED_LAYERS]

    if COMBINED_CSV.exists():
        return []

    missing = sorted(expected.difference(combined))
    extra = sorted(set(combined).difference(expected))
    raise FileNotFoundError(
        "Could not find a complete set of exact triangle layer shards; "
        f"missing={missing}, extra={extra}"
    )


def read_exact_metrics(shard_paths: list[Path]) -> pd.DataFrame:
    if shard_paths:
        frames = []
        for path in shard_paths:
            frame = pd.read_csv(path)
            file_layer = parse_layer(path)
            row_layers = tuple(sorted(int(value) for value in frame["layer"].unique()))
            if row_layers != (file_layer,):
                raise ValueError(f"{path} contains layer values {row_layers}, expected {file_layer}")
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(COMBINED_CSV)

    validate_exact_metrics(df)
    return df.sort_values(["layer", "split", "protein_id"]).reset_index(drop=True)


def validate_exact_metrics(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Exact triangle metrics are missing required columns: {missing}")

    layers = tuple(sorted(int(value) for value in df["layer"].dropna().unique()))
    if layers != EXPECTED_LAYERS:
        raise ValueError(f"Expected layers 0-47, found {layers}")

    split_ids = tuple(sorted(int(value) for value in df["split_id"].dropna().unique()))
    if split_ids != (SPLIT_ID,):
        raise ValueError(f"Expected split_id {SPLIT_ID}, found {split_ids}")

    variants = tuple(sorted(str(value) for value in df["variant"].dropna().unique()))
    if variants != (VARIANT,):
        raise ValueError(f"Expected variant {VARIANT!r}, found {variants}")

    row_counts = df.groupby("layer", sort=True).size()
    bad_row_counts = row_counts[row_counts != EXPECTED_PROTEINS_PER_LAYER]
    if not bad_row_counts.empty:
        raise ValueError(f"Expected 552 rows per layer, found {bad_row_counts.to_dict()}")

    protein_counts = df.groupby("layer", sort=True)["protein_id"].nunique()
    bad_protein_counts = protein_counts[protein_counts != EXPECTED_PROTEINS_PER_LAYER]
    if not bad_protein_counts.empty:
        raise ValueError(f"Expected 552 unique proteins per layer, found {bad_protein_counts.to_dict()}")

    n_duplicates = int(df.duplicated(["split_id", "layer", "split", "protein_id"]).sum())
    if n_duplicates:
        raise ValueError(f"Found {n_duplicates} duplicate protein-layer rows")

    expected_triangles = (
        df["n_residues"].astype(np.int64)
        * (df["n_residues"].astype(np.int64) - 1)
        * (df["n_residues"].astype(np.int64) - 2)
        // 6
    )
    if not np.array_equal(expected_triangles.to_numpy(), df["n_triangles"].astype(np.int64).to_numpy()):
        raise ValueError("n_triangles does not match n_residues choose 3 for every row")

    for column in PROPORTION_COLUMNS:
        values = df[column].astype(np.float64)
        if ((values < -VALIDATION_TOLERANCE) | (values > 1.0 + VALIDATION_TOLERANCE)).any():
            raise ValueError(f"{column} contains values outside [0, 1]")

    category_sum = df[CATEGORY_COLUMNS].astype(np.float64).sum(axis=1)
    category_error = float(np.max(np.abs(category_sum - 1.0)))
    if category_error > VALIDATION_TOLERANCE:
        raise ValueError(f"Triangle category proportions do not sum to 1; max error={category_error}")

    violation_bin_sum = df[VIOLATION_BIN_COLUMNS].astype(np.float64).sum(axis=1)
    violation_error = float(
        np.max(np.abs(df["triangle_violation_fraction"].astype(np.float64) - violation_bin_sum))
    )
    if violation_error > VALIDATION_TOLERANCE:
        raise ValueError(
            "Triangle violation bin proportions do not match total violation fraction; "
            f"max error={violation_error}"
        )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp_path, index=False, float_format="%.6f")
    tmp_path.replace(path)


def move_top_level_layer_artifacts(shard_paths: list[Path]) -> None:
    LAYER_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    for path in shard_paths:
        if path.parent != EXACT_DIR:
            continue
        target = LAYER_BATCH_DIR / path.name
        if target.exists():
            if filecmp.cmp(path, target, shallow=False):
                path.unlink()
            else:
                raise FileExistsError(f"Refusing to overwrite non-identical {target}")
        else:
            path.replace(target)

    summary_path = EXACT_DIR / "summary.txt"
    if summary_path.exists():
        target = LAYER_BATCH_DIR / summary_path.name
        if target.exists():
            if filecmp.cmp(summary_path, target, shallow=False):
                summary_path.unlink()
            else:
                raise FileExistsError(f"Refusing to overwrite non-identical {target}")
        else:
            summary_path.replace(target)


def normalize_line_endings(path: Path) -> None:
    data = path.read_bytes()
    normalized = data.replace(b"\r\n", b"\n")
    if normalized != data:
        path.write_bytes(normalized)


def normalize_layer_batch_artifacts() -> None:
    for path in sorted(LAYER_BATCH_DIR.glob("exact_triangle_metrics_layer_*.csv")):
        normalize_line_endings(path)

    summary_path = LAYER_BATCH_DIR / "summary.txt"
    if summary_path.exists():
        normalize_line_endings(summary_path)


def build_layer_means(df: pd.DataFrame) -> pd.DataFrame:
    grouped = df.groupby("layer", sort=True)
    mean_columns = [
        "triangle_violation_fraction",
        "triangle_no_violation_fraction",
        "triangle_violation_0_5_fraction",
        "triangle_violation_5_10_fraction",
        "triangle_violation_10_plus_fraction",
        "triangle_mean_violation",
        "triangle_max_violation",
        "n_negative_predictions",
    ]
    means = grouped[mean_columns].mean().reset_index()
    means.insert(1, "n_proteins", grouped["protein_id"].nunique().to_numpy(dtype=np.int64))
    means.insert(2, "n_rows", grouped.size().to_numpy(dtype=np.int64))
    means.insert(3, "n_triangles_total", grouped["n_triangles"].sum().to_numpy(dtype=np.int64))
    return means


def add_common_axis_style(ax: plt.Axes) -> None:
    ax.set_xlim(0, 47)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 47])
    ax.grid(True, color=GRID_GREY, linewidth=0.8, alpha=0.7)
    ax.tick_params(axis="both", width=1.0, length=5)


def set_zero_based_ylim(ax: plt.Axes, values: pd.Series) -> None:
    ymax = float(np.nanmax(values.to_numpy(dtype=np.float64)))
    if not np.isfinite(ymax) or ymax <= 0.0:
        ymax = 1.0
    ax.set_ylim(0.0, ymax * 1.12)


def plot_violation_proportions(layer_means: pd.DataFrame) -> None:
    panels = [
        ("triangle_violation_fraction", "Any violation", BLUE),
        ("triangle_violation_0_5_fraction", "Violation 0-5 A", GREEN),
        ("triangle_violation_5_10_fraction", "Violation 5-10 A", ORANGE),
        ("triangle_violation_10_plus_fraction", "Violation >=10 A", RED),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.2), sharex=True, constrained_layout=True)
    for ax, (column, title, color) in zip(axes.ravel(), panels):
        ax.plot(
            layer_means["layer"],
            layer_means[column],
            color=color,
            linewidth=2.3,
            marker="o",
            markersize=3.8,
        )
        ax.set_title(title)
        ax.set_ylabel("Mean proportion")
        set_zero_based_ylim(ax, layer_means[column])
        add_common_axis_style(ax)

    axes[1, 0].set_xlabel("Layer")
    axes[1, 1].set_xlabel("Layer")
    fig.suptitle("Exact triangle inequality violation proportions across layers")
    fig.savefig(PROPORTION_PNG)
    plt.close(fig)


def plot_mean_violation(layer_means: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 5.6), constrained_layout=True)
    ax.plot(
        layer_means["layer"],
        layer_means["triangle_mean_violation"],
        color=PURPLE,
        linewidth=2.6,
        marker="o",
        markersize=4.0,
    )
    ax.set_title("Mean exact triangle inequality violation across layers")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Mean violation over all triangles (A)")
    set_zero_based_ylim(ax, layer_means["triangle_mean_violation"])
    add_common_axis_style(ax)
    fig.savefig(MEAN_VIOLATION_PNG)
    plt.close(fig)


def write_analysis_note() -> None:
    content = f"""# Exact Triangle Metrics Analysis

This note documents the local exact-triangle analysis for `{VARIANT}`.

Inputs:
- Layer shards are stored in `delta_files/{VARIANT}/exact_triangle_metrics/layer_batches/`.
- The combined per-protein file is `delta_files/{VARIANT}/exact_triangle_metrics/exact_triangle_metrics.csv`.

Generated analysis artifacts:
- `delta_files/{VARIANT}/analysis/{LAYER_MEANS_CSV.name}`
- `delta_files/{VARIANT}/analysis/{PROPORTION_PNG.name}`
- `delta_files/{VARIANT}/analysis/{MEAN_VIOLATION_PNG.name}`

Triangle fields:
- `triangle_no_violation_fraction`: fraction of unordered residue triples with no triangle inequality violation.
- `triangle_violation_0_5_fraction`: fraction of unordered residue triples with a positive violation below 5 Angstrom.
- `triangle_violation_5_10_fraction`: fraction of unordered residue triples with a positive violation from 5 Angstrom up to, but not including, 10 Angstrom.
- `triangle_violation_10_plus_fraction`: fraction of unordered residue triples with a positive violation of at least 10 Angstrom.
- `triangle_violation_fraction`: sum of the three positive violation bins.
- `triangle_mean_violation`: per-protein violation magnitude sum divided by all unordered residue triples.

Layer summaries:
- The plots and layer summary CSV use an unweighted mean across all 552 proteins in each layer.
- Train and held-out proteins are not separated in these exact-triangle plots.

Compatibility note:
- Future `geometry_metrics.csv` files should include the same triangle fields so downstream analyses can use the same plotting and validation logic.
"""
    ANALYSIS_MD.write_text(content, encoding="utf-8")


def require_nonempty(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Expected nonempty output file: {path}")


def main() -> None:
    configure_style()
    ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    EXACT_DIR.mkdir(parents=True, exist_ok=True)

    shard_paths = find_layer_shards()
    metrics = read_exact_metrics(shard_paths)
    write_csv(metrics, COMBINED_CSV)
    move_top_level_layer_artifacts(shard_paths)
    normalize_layer_batch_artifacts()

    layer_means = build_layer_means(metrics)
    write_csv(layer_means, LAYER_MEANS_CSV)
    plot_violation_proportions(layer_means)
    plot_mean_violation(layer_means)
    write_analysis_note()

    require_nonempty([COMBINED_CSV, LAYER_MEANS_CSV, PROPORTION_PNG, MEAN_VIOLATION_PNG, ANALYSIS_MD])

    print("Wrote exact triangle analysis artifacts:")
    for path in [COMBINED_CSV, LAYER_MEANS_CSV, PROPORTION_PNG, MEAN_VIOLATION_PNG, ANALYSIS_MD]:
        print(f"  {path.relative_to(PROJECT_ROOT)}")
    print(f"Moved layer shards to: {LAYER_BATCH_DIR.relative_to(PROJECT_ROOT)}")


if __name__ == "__main__":
    main()

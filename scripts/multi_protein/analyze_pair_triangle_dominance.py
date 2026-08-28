#!/usr/bin/env python3
"""Create local trajectories, channel heatmaps, and channel rankings."""

from __future__ import annotations

import csv
import json
import os
import tempfile
from collections import defaultdict
from pathlib import Path

os.environ.setdefault(
    "MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "pair-triangle-matplotlib")
)
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import TwoSlopeNorm


REPOSITORY_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = REPOSITORY_ROOT / "delta_files" / "pair_representation_triangle_dominance"
OUTPUT_DIR = (
    REPOSITORY_ROOT
    / "delta_files"
    / "analysis"
    / "pair_representation_triangle_dominance"
)
LAYERS = tuple(range(48))
VARIANTS = (
    "directional_raw",
    "directional_abs",
    "abs_then_sym",
    "sym_then_abs",
)
TRAJECTORY_METRICS = (
    ("positive_normalized_violation", "Positive normalized violation"),
    ("positive_violation", "Positive raw-scale violation"),
    ("signed_normalized_margin", "Signed normalized margin"),
    ("signed_margin", "Signed raw-scale margin"),
)
CHANNEL_METRICS = tuple(metric for metric, _ in TRAJECTORY_METRICS) + (
    "positive_channel_fraction",
    "zero_scale_fraction",
)
RANKED_COLUMNS = (
    "representation_variant",
    "metric",
    "rank",
    "channel",
    "peak_absolute_protein_equal_mean",
    "peak_signed_protein_equal_mean",
    "peak_layer",
    "mean_absolute_protein_equal_mean",
    "n_finite_layers",
    "minimum_finite_proteins",
)


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def atomic_write_csv(path: Path, columns: tuple[str, ...], rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def plot_trajectories(layer_rows: list[dict[str, str]]) -> list[str]:
    output_names: list[str] = []
    lookup = {
        (int(row["layer"]), row["representation_variant"], row["metric"]): row
        for row in layer_rows
    }
    specs = list(TRAJECTORY_METRICS) + [
        ("positive_channel_fraction", "Fraction of channels with positive margin")
    ]
    for metric, label in specs:
        figure, axis = plt.subplots(figsize=(9.5, 5.2))
        for variant in VARIANTS:
            values = np.full(len(LAYERS), np.nan)
            finite_counts = []
            for layer in LAYERS:
                row = lookup.get((layer, variant, metric))
                if row is not None:
                    values[layer] = float(row["protein_equal_mean"])
                    finite_counts.append(int(row["n_proteins_finite"]))
            coverage = (
                f"n={min(finite_counts)}–{max(finite_counts)}"
                if finite_counts
                else "n=0"
            )
            axis.plot(LAYERS, values, marker="o", markersize=2.5, linewidth=1.5,
                      label=f"{variant} ({coverage})")
        axis.set_xlabel("Pair-representation layer")
        axis.set_ylabel(label)
        axis.set_title(f"Pair-representation triangle dominance: {label.lower()}")
        axis.grid(alpha=0.25)
        axis.legend(fontsize=8)
        figure.tight_layout()
        suffix = {
            "positive_violation": "positive_violation_raw_scale",
            "signed_margin": "signed_margin_raw_scale",
        }.get(metric, metric)
        filename = f"trajectory_{suffix}.png"
        figure.savefig(OUTPUT_DIR / filename, dpi=200)
        plt.close(figure)
        output_names.append(filename)
    return output_names


def plot_channel_heatmaps(channel_rows: list[dict[str, str]]) -> list[str]:
    output_names: list[str] = []
    lookup = {
        (
            int(row["layer"]),
            row["representation_variant"],
            int(row["channel"]),
            row["metric"],
        ): float(row["protein_equal_mean"])
        for row in channel_rows
    }
    for variant in VARIANTS:
        for metric in CHANNEL_METRICS:
            matrix = np.full((128, len(LAYERS)), np.nan)
            for channel in range(128):
                for layer in LAYERS:
                    matrix[channel, layer] = lookup.get(
                        (layer, variant, channel, metric), np.nan
                    )
            finite = matrix[np.isfinite(matrix)]
            figure, axis = plt.subplots(figsize=(11, 6))
            kwargs: dict[str, object] = {"aspect": "auto", "origin": "lower"}
            if finite.size and np.min(finite) < 0 < np.max(finite):
                kwargs["cmap"] = "coolwarm"
                kwargs["norm"] = TwoSlopeNorm(
                    vmin=float(np.min(finite)), vcenter=0.0, vmax=float(np.max(finite))
                )
            else:
                kwargs["cmap"] = "viridis"
            image = axis.imshow(np.ma.masked_invalid(matrix), **kwargs)
            axis.set_xlabel("Pair-representation layer")
            axis.set_ylabel("Pair channel")
            axis.set_title(f"{variant}: {metric}")
            figure.colorbar(image, ax=axis, label="Protein-equal mean")
            figure.tight_layout()
            filename = f"channel_heatmap_{variant}_{metric}.png"
            figure.savefig(OUTPUT_DIR / filename, dpi=180)
            plt.close(figure)
            output_names.append(filename)
    return output_names


def rank_channels(channel_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, int], list[dict[str, str]]] = defaultdict(list)
    for row in channel_rows:
        grouped[
            (row["representation_variant"], row["metric"], int(row["channel"]))
        ].append(row)
    output: list[dict[str, object]] = []
    for variant in VARIANTS:
        for metric in CHANNEL_METRICS:
            records = []
            for channel in range(128):
                rows = sorted(grouped[(variant, metric, channel)], key=lambda row: int(row["layer"]))
                values = np.asarray(
                    [float(row["protein_equal_mean"]) for row in rows], dtype=np.float64
                )
                finite_mask = np.isfinite(values)
                if not np.any(finite_mask):
                    peak_absolute = peak_signed = mean_absolute = float("nan")
                    peak_layer = -1
                else:
                    finite_indices = np.flatnonzero(finite_mask)
                    local = int(np.argmax(np.abs(values[finite_mask])))
                    peak_index = int(finite_indices[local])
                    peak_signed = float(values[peak_index])
                    peak_absolute = abs(peak_signed)
                    mean_absolute = float(np.mean(np.abs(values[finite_mask])))
                    peak_layer = int(rows[peak_index]["layer"])
                records.append(
                    {
                        "representation_variant": variant,
                        "metric": metric,
                        "channel": channel,
                        "peak_absolute_protein_equal_mean": peak_absolute,
                        "peak_signed_protein_equal_mean": peak_signed,
                        "peak_layer": peak_layer,
                        "mean_absolute_protein_equal_mean": mean_absolute,
                        "n_finite_layers": int(np.sum(finite_mask)),
                        "minimum_finite_proteins": min(
                            (int(row["n_proteins_finite"]) for row in rows), default=0
                        ),
                    }
                )
            records.sort(
                key=lambda row: (
                    -float(row["peak_absolute_protein_equal_mean"])
                    if np.isfinite(row["peak_absolute_protein_equal_mean"])
                    else float("inf"),
                    int(row["channel"]),
                )
            )
            for rank, row in enumerate(records, start=1):
                output.append({**row, "rank": rank})
    return output


def main() -> None:
    layer_path = INPUT_DIR / "layer_summary.csv"
    channel_path = INPUT_DIR / "channel_summary.csv"
    if not layer_path.is_file() or not channel_path.is_file():
        raise FileNotFoundError(
            f"Copy consolidated results into {INPUT_DIR} before local analysis"
        )
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    layer_rows = read_rows(layer_path)
    channel_rows = read_rows(channel_path)
    expected_layer_rows = 48 * 4 * 5
    expected_channel_rows = 48 * 4 * 128 * 6
    if len(layer_rows) != expected_layer_rows:
        raise ValueError(f"Expected {expected_layer_rows} layer summaries; observed {len(layer_rows)}")
    if len(channel_rows) != expected_channel_rows:
        raise ValueError(
            f"Expected {expected_channel_rows} channel summaries; observed {len(channel_rows)}"
        )
    trajectory_files = plot_trajectories(layer_rows)
    heatmap_files = plot_channel_heatmaps(channel_rows)
    ranked = rank_channels(channel_rows)
    atomic_write_csv(OUTPUT_DIR / "ranked_channels.csv", RANKED_COLUMNS, ranked)
    manifest = {
        "input_directory": str(INPUT_DIR),
        "output_directory": str(OUTPUT_DIR),
        "trajectory_files": trajectory_files,
        "channel_heatmap_files": heatmap_files,
        "ranked_channel_rows": len(ranked),
        "missing_layers_are_not_interpolated": True,
    }
    temporary = OUTPUT_DIR / ".analysis_manifest.json.tmp"
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(OUTPUT_DIR / "analysis_manifest.json")
    print(f"Wrote pair-triangle analysis to {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
Analyze fold-4 baseline256 coefficient evolution across the saved K=3 layer segments.

Inputs:
  - delta_files/baseline256_kfold/coefficients.csv
  - delta_files/baseline256_kfold/analysis/baseline256_fold4_layer_segmentation_segments.csv

Outputs:
  - delta_files/baseline256_kfold/analysis/
    - baseline256_fold4_segment_mean_coefficients.csv
    - baseline256_fold4_segment_top_shifted_coefficients.csv
    - baseline256_fold4_segment_mean_coefficient_magnitudes.png
    - baseline256_fold4_segment_signed_coefficient_heatmap.png
    - baseline256_fold4_segment_top_shifted_coefficients.png
"""

from __future__ import annotations

import os
import re
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
INPUT_DIR = PROJECT_ROOT / "delta_files" / "baseline256_kfold"
OUTPUT_DIR = INPUT_DIR / "analysis"

COEFFICIENTS_CSV = INPUT_DIR / "coefficients.csv"
SEGMENTS_CSV = OUTPUT_DIR / "baseline256_fold4_layer_segmentation_segments.csv"

SEGMENT_MEAN_CSV = OUTPUT_DIR / "baseline256_fold4_segment_mean_coefficients.csv"
TOP_SHIFT_CSV = OUTPUT_DIR / "baseline256_fold4_segment_top_shifted_coefficients.csv"
MAGNITUDE_PNG = OUTPUT_DIR / "baseline256_fold4_segment_mean_coefficient_magnitudes.png"
HEATMAP_PNG = OUTPUT_DIR / "baseline256_fold4_segment_signed_coefficient_heatmap.png"
TOP_SHIFT_PNG = OUTPUT_DIR / "baseline256_fold4_segment_top_shifted_coefficients.png"

EXPECTED_LAYERS = list(range(48))
EXPECTED_FOLDS = list(range(5))
LAST_FOLD = 4
FEATURE_DIM = 256
CHANNELS_PER_DIRECTION = 128
PLOT_SEGMENT_K = 3
TOP_SHIFT_COUNT = 20
FEATURE_NAME_PATTERN = re.compile(r"^pair_(ij|ji)_ch_(\d+)$")

SEGMENT_COLORS = {
    1: "#7d2b2b",
    2: "#1f5a91",
    3: "#2d7d5f",
}


@dataclass(frozen=True)
class Segment:
    segment_id: int
    start_layer: int
    end_layer: int

    @property
    def n_layers(self) -> int:
        return self.end_layer - self.start_layer + 1

    @property
    def label(self) -> str:
        return f"Segment {self.segment_id} ({self.start_layer}-{self.end_layer})"


def ensure_required_columns(path: Path, df: pd.DataFrame, required: set[str]) -> None:
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")


def ensure_exact_values(path: Path, df: pd.DataFrame, column: str, expected: list[int]) -> None:
    found = sorted(int(value) for value in df[column].dropna().unique())
    if found != expected:
        raise ValueError(f"{path.name} expected {column} values {expected}, found {found}")


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


def parse_feature_metadata(feature_index: int, feature_name: str) -> tuple[str, int]:
    match = FEATURE_NAME_PATTERN.fullmatch(feature_name)
    if match is None:
        raise ValueError(f"Unexpected feature_name format: {feature_name}")
    direction = match.group(1)
    channel_id = int(match.group(2))
    if not 0 <= channel_id < CHANNELS_PER_DIRECTION:
        raise ValueError(f"Unexpected channel id {channel_id} in {feature_name}")
    expected_direction = "ij" if feature_index < CHANNELS_PER_DIRECTION else "ji"
    if direction != expected_direction:
        raise ValueError(
            f"Feature direction mismatch for index {feature_index}: "
            f"expected {expected_direction}, found {direction}"
        )
    return direction, channel_id


def validate_coefficients(coefficients: pd.DataFrame) -> None:
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
    ensure_exact_values(COEFFICIENTS_CSV, coefficients, "fold", EXPECTED_FOLDS)
    ensure_exact_values(COEFFICIENTS_CSV, coefficients, "layer", EXPECTED_LAYERS)
    ensure_no_duplicates(COEFFICIENTS_CSV, coefficients, ["fold", "layer", "feature_index"])
    ensure_complete_group_count(
        COEFFICIENTS_CSV,
        coefficients,
        ["fold", "layer"],
        FEATURE_DIM + 1,
        "(fold, layer)",
    )

    non_intercept = coefficients[coefficients["feature_index"] != -1].copy()
    ensure_complete_group_count(
        COEFFICIENTS_CSV,
        non_intercept,
        ["fold", "layer"],
        FEATURE_DIM,
        "non-intercept (fold, layer)",
    )

    unique_feature_counts = non_intercept.groupby(["fold", "layer"])["feature_index"].nunique()
    if not (unique_feature_counts == FEATURE_DIM).all():
        raise ValueError("coefficients.csv must contain one complete 256-value feature vector per (fold, layer)")

    intercept_counts = coefficients[coefficients["feature_index"] == -1].groupby(["fold", "layer"]).size()
    if not (intercept_counts == 1).all():
        raise ValueError("coefficients.csv must contain exactly one intercept row per (fold, layer)")


def load_fold4_coefficients() -> pd.DataFrame:
    coefficients = pd.read_csv(COEFFICIENTS_CSV)
    validate_coefficients(coefficients)

    fold4 = coefficients[
        (coefficients["fold"] == LAST_FOLD) & (coefficients["feature_index"] != -1)
    ].copy()
    fold4 = fold4.sort_values(["layer", "feature_index"]).reset_index(drop=True)

    layer_counts = fold4.groupby("layer", sort=True).size()
    if not (layer_counts == FEATURE_DIM).all():
        raise ValueError("Fold 4 must contain exactly 256 non-intercept coefficients per layer")

    feature_counts = fold4.groupby("feature_index", sort=True).size()
    if not (feature_counts == len(EXPECTED_LAYERS)).all():
        raise ValueError("Fold 4 must contain every feature index at every layer")

    return fold4


def load_k3_segments() -> tuple[Segment, ...]:
    segments_df = pd.read_csv(SEGMENTS_CSV)
    ensure_required_columns(
        SEGMENTS_CSV,
        segments_df,
        {"k", "segment_id", "start_layer", "end_layer", "n_layers"},
    )

    k3 = segments_df[segments_df["k"] == PLOT_SEGMENT_K].copy()
    if k3.shape[0] != PLOT_SEGMENT_K:
        raise ValueError(f"Expected exactly {PLOT_SEGMENT_K} rows for k={PLOT_SEGMENT_K}, found {k3.shape[0]}")

    k3["segment_id"] = k3["segment_id"].astype(int)
    k3["start_layer"] = k3["start_layer"].astype(int)
    k3["end_layer"] = k3["end_layer"].astype(int)
    k3["n_layers"] = k3["n_layers"].astype(int)

    k3 = k3.sort_values(["segment_id", "start_layer", "end_layer"]).reset_index(drop=True)
    expected_segment_ids = list(range(1, PLOT_SEGMENT_K + 1))
    found_segment_ids = k3["segment_id"].tolist()
    if found_segment_ids != expected_segment_ids:
        raise ValueError(
            f"Expected segment ids {expected_segment_ids} for k={PLOT_SEGMENT_K}, found {found_segment_ids}"
        )

    segments = tuple(
        Segment(
            segment_id=int(row.segment_id),
            start_layer=int(row.start_layer),
            end_layer=int(row.end_layer),
        )
        for row in k3.itertuples(index=False)
    )

    if segments[0].start_layer != EXPECTED_LAYERS[0]:
        raise ValueError(f"First segment must start at layer {EXPECTED_LAYERS[0]}")
    if segments[-1].end_layer != EXPECTED_LAYERS[-1]:
        raise ValueError(f"Last segment must end at layer {EXPECTED_LAYERS[-1]}")

    previous_end = None
    for segment, expected_id in zip(segments, expected_segment_ids):
        if segment.segment_id != expected_id:
            raise ValueError(f"Expected segment id {expected_id}, found {segment.segment_id}")
        if segment.n_layers <= 0:
            raise ValueError(f"Segment {segment.segment_id} must contain at least one layer")
        if previous_end is not None and segment.start_layer != previous_end + 1:
            raise ValueError(
                f"Segments must be contiguous: expected start {previous_end + 1}, found {segment.start_layer}"
            )
        previous_end = segment.end_layer

    covered_layers = [layer for segment in segments for layer in range(segment.start_layer, segment.end_layer + 1)]
    if covered_layers != EXPECTED_LAYERS:
        raise ValueError("K=3 segments must cover layers 0..47 exactly once")

    return segments


def build_feature_metadata(fold4: pd.DataFrame) -> pd.DataFrame:
    metadata = fold4[["feature_index", "feature_name"]].drop_duplicates().copy()
    metadata["feature_index"] = metadata["feature_index"].astype(int)
    metadata = metadata.sort_values("feature_index").reset_index(drop=True)
    if metadata.shape[0] != FEATURE_DIM:
        raise ValueError(f"Expected {FEATURE_DIM} unique non-intercept features, found {metadata.shape[0]}")

    directions: list[str] = []
    channel_ids: list[int] = []
    for row in metadata.itertuples(index=False):
        direction, channel_id = parse_feature_metadata(int(row.feature_index), str(row.feature_name))
        directions.append(direction)
        channel_ids.append(channel_id)
    metadata["direction"] = directions
    metadata["channel_id"] = channel_ids
    return metadata


def build_segment_summary(fold4: pd.DataFrame, segments: tuple[Segment, ...]) -> pd.DataFrame:
    summary = build_feature_metadata(fold4)
    summary = summary.sort_values("feature_index").reset_index(drop=True)

    segment_means: list[np.ndarray] = []
    segment_abs_means: list[np.ndarray] = []

    for segment in segments:
        segment_slice = fold4[
            (fold4["layer"] >= segment.start_layer) & (fold4["layer"] <= segment.end_layer)
        ].copy()
        grouped = (
            segment_slice.groupby(["feature_index", "feature_name"], sort=True)["coefficient_raw_scale"]
            .mean()
            .reset_index()
            .sort_values("feature_index")
            .reset_index(drop=True)
        )
        if grouped.shape[0] != FEATURE_DIM:
            raise ValueError(
                f"Segment {segment.segment_id} expected {FEATURE_DIM} feature means, found {grouped.shape[0]}"
            )

        means = grouped["coefficient_raw_scale"].astype(np.float64).to_numpy()
        if means.shape != (FEATURE_DIM,):
            raise ValueError(
                f"Segment {segment.segment_id} mean vector has shape {means.shape}, expected ({FEATURE_DIM},)"
            )

        segment_means.append(means)
        segment_abs_means.append(np.abs(means))
        summary[f"segment_{segment.segment_id}_mean"] = means
        summary[f"segment_{segment.segment_id}_abs_mean"] = np.abs(means)

    mean_matrix = np.vstack(segment_means)
    abs_matrix = np.vstack(segment_abs_means)
    summary["signed_range"] = np.max(mean_matrix, axis=0) - np.min(mean_matrix, axis=0)
    summary["abs_range"] = np.max(abs_matrix, axis=0) - np.min(abs_matrix, axis=0)

    return summary


def build_top_shift_table(summary: pd.DataFrame) -> pd.DataFrame:
    ranked = summary.copy()
    ranked["signed_range_rank"] = (
        ranked["signed_range"].rank(method="min", ascending=False).astype(int)
    )
    ranked["abs_range_rank"] = (
        ranked["abs_range"].rank(method="min", ascending=False).astype(int)
    )
    ranked = ranked.sort_values(
        ["abs_range_rank", "signed_range_rank", "feature_index"],
        ascending=[True, True, True],
    ).reset_index(drop=True)
    return ranked


def plot_segment_magnitudes(summary: pd.DataFrame, segments: tuple[Segment, ...]) -> None:
    fig, ax = plt.subplots(figsize=(15, 6), constrained_layout=True)

    x_values = summary["feature_index"].astype(np.int64).to_numpy()
    for segment in segments:
        ax.plot(
            x_values,
            summary[f"segment_{segment.segment_id}_abs_mean"].astype(np.float64).to_numpy(),
            color=SEGMENT_COLORS[segment.segment_id],
            linewidth=1.6,
            label=segment.label,
        )

    ax.axvline(127.5, color="#555555", linestyle="--", linewidth=1.1)
    ax.text(63.5, ax.get_ylim()[1] * 0.98, "pair_ij", ha="center", va="top", fontsize=10)
    ax.text(191.5, ax.get_ylim()[1] * 0.98, "pair_ji", ha="center", va="top", fontsize=10)
    ax.set_xlim(-0.5, FEATURE_DIM - 0.5)
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Absolute segment-mean coefficient")
    ax.set_title("Fold-4 baseline256 coefficient magnitudes by K=3 layer segment")
    ax.set_xticks(np.arange(0, FEATURE_DIM, 16))
    ax.grid(True, alpha=0.22)
    ax.legend(loc="upper right", fontsize=9)

    fig.savefig(MAGNITUDE_PNG, dpi=200)
    plt.close(fig)


def plot_signed_heatmap(summary: pd.DataFrame, segments: tuple[Segment, ...]) -> None:
    matrix = np.vstack(
        [
            summary[f"segment_{segment.segment_id}_mean"].astype(np.float64).to_numpy()
            for segment in segments
        ]
    )
    vmax = float(np.max(np.abs(matrix)))
    if vmax <= 0.0:
        vmax = 1.0

    fig, ax = plt.subplots(figsize=(15, 4.8), constrained_layout=True)
    image = ax.imshow(matrix, aspect="auto", cmap="coolwarm", vmin=-vmax, vmax=vmax)
    ax.axvline(127.5, color="black", linestyle="--", linewidth=1.0)
    ax.set_title("Fold-4 baseline256 signed segment-mean coefficients")
    ax.set_xlabel("Coefficient index")
    ax.set_ylabel("Layer segment")
    ax.set_xticks(np.arange(0, FEATURE_DIM, 16))
    ax.set_yticks(np.arange(len(segments)))
    ax.set_yticklabels([segment.label for segment in segments])
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Segment-mean coefficient")

    fig.savefig(HEATMAP_PNG, dpi=200)
    plt.close(fig)


def plot_top_shifted_coefficients(summary: pd.DataFrame, segments: tuple[Segment, ...]) -> None:
    top = summary.sort_values(
        ["abs_range", "signed_range", "feature_index"],
        ascending=[False, False, True],
    ).head(TOP_SHIFT_COUNT)
    top = top.sort_values("abs_range", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(16, 7), constrained_layout=True)
    x_positions = np.arange(top.shape[0], dtype=np.float64)
    width = 0.24

    for offset, segment in zip([-width, 0.0, width], segments):
        ax.bar(
            x_positions + offset,
            top[f"segment_{segment.segment_id}_mean"].astype(np.float64).to_numpy(),
            width=width,
            color=SEGMENT_COLORS[segment.segment_id],
            label=segment.label,
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0)
    ax.set_xticks(x_positions)
    ax.set_xticklabels(top["feature_name"].tolist(), rotation=90, fontsize=9)
    ax.set_ylabel("Segment-mean coefficient")
    ax.set_title(f"Top {TOP_SHIFT_COUNT} coefficient shifts by segment-mean magnitude change")
    ax.grid(True, axis="y", alpha=0.22)
    ax.legend(fontsize=9)

    fig.savefig(TOP_SHIFT_PNG, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    fold4 = load_fold4_coefficients()
    segments = load_k3_segments()
    summary = build_segment_summary(fold4, segments)
    top_shift = build_top_shift_table(summary)

    summary.to_csv(SEGMENT_MEAN_CSV, index=False, float_format="%.6f")
    top_shift.to_csv(TOP_SHIFT_CSV, index=False, float_format="%.6f")

    plot_segment_magnitudes(summary, segments)
    plot_signed_heatmap(summary, segments)
    plot_top_shifted_coefficients(summary, segments)

    print(f"Saved segment mean coefficients CSV -> {SEGMENT_MEAN_CSV}")
    print(f"Saved top-shift coefficients CSV -> {TOP_SHIFT_CSV}")
    print(f"Saved coefficient magnitude plot -> {MAGNITUDE_PNG}")
    print(f"Saved signed heatmap -> {HEATMAP_PNG}")
    print(f"Saved top-shift plot -> {TOP_SHIFT_PNG}")


if __name__ == "__main__":
    main()

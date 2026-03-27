#!/usr/bin/env python3
"""
Segment fold-4 baseline256 coefficient vectors into contiguous layer regimes.

Inputs:
  - delta_files/baseline256_kfold/coefficients.csv

Outputs:
  - delta_files/baseline256_kfold/analysis/
    - baseline256_fold4_layer_segmentation_model_selection.csv
    - baseline256_fold4_layer_segmentation_segments.csv
    - baseline256_fold4_layer_segmentation_permutation_test.csv
    - baseline256_fold4_layer_segmentation_overview.png
"""

from __future__ import annotations

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
INPUT_DIR = PROJECT_ROOT / "delta_files" / "baseline256_kfold"
OUTPUT_DIR = INPUT_DIR / "analysis"

COEFFICIENTS_CSV = INPUT_DIR / "coefficients.csv"
MODEL_SELECTION_CSV = OUTPUT_DIR / "baseline256_fold4_layer_segmentation_model_selection.csv"
SEGMENTS_CSV = OUTPUT_DIR / "baseline256_fold4_layer_segmentation_segments.csv"
PERMUTATION_CSV = OUTPUT_DIR / "baseline256_fold4_layer_segmentation_permutation_test.csv"
OVERVIEW_PNG = OUTPUT_DIR / "baseline256_fold4_layer_segmentation_overview.png"

EXPECTED_LAYERS = list(range(48))
EXPECTED_FOLDS = list(range(5))
LAST_FOLD = 4
FEATURE_DIM = 256
MAX_K = 6
PLOT_K = 3
N_PERMUTATIONS = 1000
RNG_SEED = 0


@dataclass(frozen=True)
class Segment:
    start: int
    end: int
    within_sse: float
    mean_internal_cosine: float
    pair_count: int


@dataclass(frozen=True)
class SegmentationResult:
    k: int
    total_within_sse: float
    segments: tuple[Segment, ...]


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


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = vectors / safe_norms
    matrix = normalized @ normalized.T
    return np.clip(matrix, -1.0, 1.0)


def format_p_value(value: float) -> str:
    if np.isnan(value):
        return "nan"
    if value < 0.001:
        return "p<0.001"
    return f"p={value:.3f}"


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


def load_fold4_vectors() -> np.ndarray:
    coefficients = pd.read_csv(COEFFICIENTS_CSV)
    validate_coefficients(coefficients)

    fold4 = coefficients[
        (coefficients["fold"] == LAST_FOLD) & (coefficients["feature_index"] != -1)
    ].copy()
    fold4 = fold4.sort_values(["layer", "feature_index"])
    layer_counts = fold4.groupby("layer", sort=True).size()
    if not (layer_counts == FEATURE_DIM).all():
        raise ValueError("Fold 4 must contain exactly 256 non-intercept coefficients per layer")

    vectors = []
    for layer in EXPECTED_LAYERS:
        layer_values = fold4[fold4["layer"] == layer]["coefficient_raw_scale"].astype(np.float64).to_numpy()
        if layer_values.shape != (FEATURE_DIM,):
            raise ValueError(f"Layer {layer} has shape {layer_values.shape}, expected ({FEATURE_DIM},)")
        vectors.append(layer_values)
    return np.vstack(vectors)


def normalize_rows(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= 0.0):
        raise ValueError("All layer vectors must have positive norm for cosine-based segmentation")
    return vectors / norms


def build_segment_costs(vectors: np.ndarray) -> np.ndarray:
    n_layers = vectors.shape[0]
    prefix_sum = np.zeros((n_layers + 1, vectors.shape[1]), dtype=np.float64)
    prefix_sq = np.zeros(n_layers + 1, dtype=np.float64)
    prefix_sum[1:] = np.cumsum(vectors, axis=0, dtype=np.float64)
    prefix_sq[1:] = np.cumsum(np.sum(vectors * vectors, axis=1, dtype=np.float64), dtype=np.float64)

    costs = np.full((n_layers + 1, n_layers + 1), np.inf, dtype=np.float64)
    for start in range(n_layers):
        for end in range(start + 1, n_layers + 1):
            length = end - start
            summed = prefix_sum[end] - prefix_sum[start]
            sum_sq = prefix_sq[end] - prefix_sq[start]
            costs[start, end] = float(sum_sq - np.dot(summed, summed) / float(length))
    return costs


def segment_pair_count(start: int, end: int) -> int:
    length = end - start
    if length < 2:
        return 0
    return int(length * (length - 1) // 2)


def segment_mean_internal_cosine(cosine_matrix: np.ndarray, start: int, end: int) -> float:
    if end - start < 2:
        return float("nan")
    block = cosine_matrix[start:end, start:end]
    rows, cols = np.triu_indices(end - start, k=1)
    values = block[rows, cols]
    return float(np.mean(values))


def solve_segmentations(costs: np.ndarray, cosine_matrix: np.ndarray, max_k: int) -> dict[int, SegmentationResult]:
    n_layers = costs.shape[0] - 1
    dp = np.full((max_k + 1, n_layers + 1), np.inf, dtype=np.float64)
    back = np.full((max_k + 1, n_layers + 1), -1, dtype=np.int64)
    dp[0, 0] = 0.0

    for k in range(1, max_k + 1):
        for end in range(k, n_layers + 1):
            best_cost = np.inf
            best_start = -1
            for start in range(k - 1, end):
                candidate = dp[k - 1, start] + costs[start, end]
                if candidate < best_cost:
                    best_cost = candidate
                    best_start = start
            dp[k, end] = best_cost
            back[k, end] = best_start

    results: dict[int, SegmentationResult] = {}
    for k in range(1, max_k + 1):
        ranges: list[tuple[int, int]] = []
        end = n_layers
        current_k = k
        while current_k > 0:
            start = int(back[current_k, end])
            if start < 0:
                raise RuntimeError(f"Failed to reconstruct segmentation for k={k}")
            ranges.append((start, end))
            end = start
            current_k -= 1
        ranges.reverse()
        segments = tuple(
            Segment(
                start=start,
                end=end - 1,
                within_sse=float(costs[start, end]),
                mean_internal_cosine=segment_mean_internal_cosine(cosine_matrix, start, end),
                pair_count=segment_pair_count(start, end),
            )
            for start, end in ranges
        )
        results[k] = SegmentationResult(
            k=k,
            total_within_sse=float(dp[k, n_layers]),
            segments=segments,
        )
    return results


def overall_mean_internal_cosine(segments: tuple[Segment, ...], cosine_matrix: np.ndarray) -> float:
    all_values: list[float] = []
    for segment in segments:
        if segment.pair_count == 0:
            continue
        block = cosine_matrix[segment.start : segment.end + 1, segment.start : segment.end + 1]
        rows, cols = np.triu_indices(segment.end - segment.start + 1, k=1)
        all_values.extend(block[rows, cols].tolist())
    if not all_values:
        return float("nan")
    return float(np.mean(all_values))


def mean_boundary_adjacent_cosine(segments: tuple[Segment, ...], cosine_matrix: np.ndarray) -> float:
    boundaries: list[float] = []
    for segment in segments[:-1]:
        left = segment.end
        right = left + 1
        boundaries.append(float(cosine_matrix[left, right]))
    if not boundaries:
        return float("nan")
    return float(np.mean(boundaries))


def boundaries_string(segments: tuple[Segment, ...]) -> str:
    if len(segments) <= 1:
        return ""
    return "|".join(str(segment.start) for segment in segments[1:])


def segments_string(segments: tuple[Segment, ...]) -> str:
    return "|".join(f"{segment.start}-{segment.end}" for segment in segments)


def run_permutation_test(vectors: np.ndarray, base_loss: float, max_k: int) -> dict[int, dict[str, float | np.ndarray]]:
    rng = np.random.default_rng(RNG_SEED)
    gain_samples = {k: np.zeros(N_PERMUTATIONS, dtype=np.float64) for k in range(2, max_k + 1)}

    for perm_idx in range(N_PERMUTATIONS):
        permuted = vectors[rng.permutation(vectors.shape[0])]
        costs = build_segment_costs(permuted)
        cosine_matrix = cosine_similarity_matrix(permuted)
        results = solve_segmentations(costs, cosine_matrix, max_k=max_k)
        for k in range(2, max_k + 1):
            gain_samples[k][perm_idx] = base_loss - results[k].total_within_sse

    stats: dict[int, dict[str, float | np.ndarray]] = {}
    for k, gains in gain_samples.items():
        stats[k] = {
            "shuffle_mean_gain": float(np.mean(gains)),
            "shuffle_std_gain": float(np.std(gains, ddof=0)),
            "samples": gains,
        }
    return stats


def build_model_selection_rows(
    results: dict[int, SegmentationResult],
    cosine_matrix: np.ndarray,
    permutation_stats: dict[int, dict[str, float | np.ndarray]],
) -> pd.DataFrame:
    rows: list[dict[str, float | int | str]] = []
    base_loss = results[1].total_within_sse

    for k in range(1, MAX_K + 1):
        result = results[k]
        gain_vs_k1 = base_loss - result.total_within_sse
        gain_vs_prev = float("nan")
        pct_reduction_vs_prev = float("nan")
        if k > 1:
            prev_loss = results[k - 1].total_within_sse
            gain_vs_prev = prev_loss - result.total_within_sse
            pct_reduction_vs_prev = gain_vs_prev / prev_loss if prev_loss > 0.0 else float("nan")

        row: dict[str, float | int | str] = {
            "k": k,
            "boundaries": boundaries_string(result.segments),
            "segments": segments_string(result.segments),
            "total_within_sse": result.total_within_sse,
            "gain_vs_k1": gain_vs_k1,
            "variance_explained_vs_k1": gain_vs_k1 / base_loss if base_loss > 0.0 else float("nan"),
            "gain_vs_prev_k": gain_vs_prev,
            "pct_loss_reduction_vs_prev_k": pct_reduction_vs_prev,
            "mean_within_segment_cosine": overall_mean_internal_cosine(result.segments, cosine_matrix),
            "mean_boundary_adjacent_cosine": mean_boundary_adjacent_cosine(result.segments, cosine_matrix),
            "permutation_mean_gain": float("nan"),
            "permutation_std_gain": float("nan"),
            "permutation_z_score": float("nan"),
            "permutation_p_value": float("nan"),
            "n_permutations": 0,
        }
        if k > 1:
            shuffle_mean = float(permutation_stats[k]["shuffle_mean_gain"])
            shuffle_std = float(permutation_stats[k]["shuffle_std_gain"])
            shuffle_samples = permutation_stats[k]["samples"]
            if not isinstance(shuffle_samples, np.ndarray):
                raise TypeError("Permutation samples must be a numpy array")
            p_value = float((1.0 + np.sum(shuffle_samples >= gain_vs_k1)) / (N_PERMUTATIONS + 1.0))
            z_score = (gain_vs_k1 - shuffle_mean) / shuffle_std if shuffle_std > 0.0 else float("nan")
            row["permutation_mean_gain"] = shuffle_mean
            row["permutation_std_gain"] = shuffle_std
            row["permutation_z_score"] = z_score
            row["permutation_p_value"] = p_value
            row["n_permutations"] = N_PERMUTATIONS
        rows.append(row)

    return pd.DataFrame(rows)


def build_segment_rows(results: dict[int, SegmentationResult]) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for k in range(1, MAX_K + 1):
        for idx, segment in enumerate(results[k].segments, start=1):
            rows.append(
                {
                    "k": k,
                    "segment_id": idx,
                    "start_layer": segment.start,
                    "end_layer": segment.end,
                    "n_layers": segment.end - segment.start + 1,
                    "segment_within_sse": segment.within_sse,
                    "segment_mean_internal_cosine": segment.mean_internal_cosine,
                }
            )
    return pd.DataFrame(rows)


def build_permutation_rows(model_selection_df: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "k",
        "gain_vs_k1",
        "permutation_mean_gain",
        "permutation_std_gain",
        "permutation_z_score",
        "permutation_p_value",
        "n_permutations",
    ]
    return model_selection_df[model_selection_df["k"] > 1][columns].copy()


def plot_overview(
    cosine_matrix: np.ndarray,
    model_selection_df: pd.DataFrame,
    k3_segments: tuple[Segment, ...],
) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(15, 6), constrained_layout=True)

    heatmap_ax = axes[0]
    image = heatmap_ax.imshow(cosine_matrix, origin="lower", cmap="coolwarm", vmin=-1.0, vmax=1.0)
    heatmap_ax.set_title(f"Fold-4 layer cosine similarity with K={PLOT_K} boundaries")
    heatmap_ax.set_xlabel("Layer")
    heatmap_ax.set_ylabel("Layer")
    ticks = np.arange(0, len(EXPECTED_LAYERS), 2)
    heatmap_ax.set_xticks(ticks)
    heatmap_ax.set_yticks(ticks)
    heatmap_ax.set_xticklabels([str(int(tick)) for tick in ticks], rotation=90, fontsize=7)
    heatmap_ax.set_yticklabels([str(int(tick)) for tick in ticks], fontsize=7)
    for segment in k3_segments[1:]:
        boundary = segment.start - 0.5
        heatmap_ax.axvline(boundary, color="white", linestyle="--", linewidth=1.3)
        heatmap_ax.axhline(boundary, color="white", linestyle="--", linewidth=1.3)
    colorbar = fig.colorbar(image, ax=heatmap_ax, fraction=0.046, pad=0.04)
    colorbar.set_label("Cosine similarity")

    gain_ax = axes[1]
    ks = model_selection_df["k"].to_numpy(dtype=np.int64)
    variance_explained = model_selection_df["variance_explained_vs_k1"].to_numpy(dtype=np.float64)
    gain_ax.plot(ks, variance_explained, marker="o", color="#1f77b4", linewidth=2.0)
    gain_ax.set_xlabel("Number of segments (K)")
    gain_ax.set_ylabel("Variance explained vs K=1")
    gain_ax.set_title("Ordered segmentation gain and permutation evidence")
    gain_ax.set_xticks(ks)
    gain_ax.grid(True, alpha=0.25)

    for row in model_selection_df.itertuples(index=False):
        if int(row.k) <= 1:
            continue
        gain_ax.annotate(
            format_p_value(float(row.permutation_p_value)),
            (int(row.k), float(row.variance_explained_vs_k1)),
            textcoords="offset points",
            xytext=(0, 10),
            ha="center",
            fontsize=8,
        )

    fig.savefig(OVERVIEW_PNG, dpi=200)
    plt.close(fig)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    raw_vectors = load_fold4_vectors().astype(np.float64, copy=False)
    normalized_vectors = normalize_rows(raw_vectors)
    cosine_matrix = cosine_similarity_matrix(raw_vectors)
    costs = build_segment_costs(normalized_vectors)
    results = solve_segmentations(costs, cosine_matrix, max_k=MAX_K)
    permutation_stats = run_permutation_test(
        normalized_vectors,
        results[1].total_within_sse,
        max_k=MAX_K,
    )

    model_selection_df = build_model_selection_rows(results, cosine_matrix, permutation_stats)
    segments_df = build_segment_rows(results)
    permutation_df = build_permutation_rows(model_selection_df)

    model_selection_df.to_csv(MODEL_SELECTION_CSV, index=False, float_format="%.6f")
    segments_df.to_csv(SEGMENTS_CSV, index=False, float_format="%.6f")
    permutation_df.to_csv(PERMUTATION_CSV, index=False, float_format="%.6f")
    plot_overview(cosine_matrix, model_selection_df, results[PLOT_K].segments)

    print(f"Saved model-selection CSV -> {MODEL_SELECTION_CSV}")
    print(f"Saved segments CSV -> {SEGMENTS_CSV}")
    print(f"Saved permutation CSV -> {PERMUTATION_CSV}")
    print(f"Saved overview plot -> {OVERVIEW_PNG}")


if __name__ == "__main__":
    main()

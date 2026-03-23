#!/usr/bin/env python3
"""
Analyze Delta precision / row-cap / PCA diagnostic outputs.

Inputs:
  - delta_files/diagnostic_precision_rowcap_pca_csv_files/
    - diagnostic_precision_rowcap_pca_variant_summary.csv
    - diagnostic_precision_rowcap_pca_per_protein.csv
    - diagnostic_precision_rowcap_pca_coefficients.csv
    - diagnostic_precision_rowcap_pca_spectrum_summary.csv
    - diagnostic_precision_rowcap_pca_singular_values.csv

Outputs:
  - delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/
"""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from pathlib import Path

EARLY_PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.makedirs(EARLY_PROJECT_ROOT / ".mplconfig", exist_ok=True)
os.makedirs(EARLY_PROJECT_ROOT / ".cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(EARLY_PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(EARLY_PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt


PROJECT_ROOT = EARLY_PROJECT_ROOT
INPUT_DIR = PROJECT_ROOT / "delta_files" / "diagnostic_precision_rowcap_pca_csv_files"
OUTPUT_DIR = INPUT_DIR / "analysis"

SUMMARY_CSV = INPUT_DIR / "diagnostic_precision_rowcap_pca_variant_summary.csv"
PER_PROTEIN_CSV = INPUT_DIR / "diagnostic_precision_rowcap_pca_per_protein.csv"
COEFFICIENT_CSV = INPUT_DIR / "diagnostic_precision_rowcap_pca_coefficients.csv"
SPECTRUM_SUMMARY_CSV = INPUT_DIR / "diagnostic_precision_rowcap_pca_spectrum_summary.csv"
SINGULAR_VALUES_CSV = INPUT_DIR / "diagnostic_precision_rowcap_pca_singular_values.csv"

REPORT_MD = OUTPUT_DIR / "diagnostic_precision_rowcap_pca_analysis_report.md"
OVERVIEW_CSV = OUTPUT_DIR / "key_variant_overview.csv"
ROWCAP_CSV = OUTPUT_DIR / "rowcap_sweep_summary.csv"
PCA_CSV = OUTPUT_DIR / "pca_sweep_summary.csv"
BUCKET_R2_CSV = OUTPUT_DIR / "test_bucket_r2_summary.csv"
SIZE_CORR_CSV = OUTPUT_DIR / "test_size_r2_correlation.csv"
BUCKET_COSINE_CSV = OUTPUT_DIR / "bucketed_128d_coefficient_cosine_similarity.csv"
BUCKET_GLOBAL_COSINE_CSV = OUTPUT_DIR / "bucketed_128d_vs_global_cosine_similarity.csv"
TOP_SHIFT_CSV = OUTPUT_DIR / "bucketed_128d_top_shifted_channels.csv"
SPECTRUM_INTERPRET_CSV = OUTPUT_DIR / "spectrum_cutoff_interpretation.csv"

ROWCAP_PLOT = OUTPUT_DIR / "rowcap_precision_test_r2.png"
PCA_PLOT = OUTPUT_DIR / "pca_sweep_test_r2.png"
BUCKET_R2_PLOT = OUTPUT_DIR / "test_bucket_r2_key_variants.png"
BUCKET_COSINE_PLOT = OUTPUT_DIR / "bucketed_128d_raw_cosine_heatmap.png"
SIZE_SCATTER_PLOT = OUTPUT_DIR / "test_size_vs_r2_key_variants.png"

BUCKET_ORDER = ["<=255", "256-371", "372-500", "501+"]
ROWCAP_ORDER = ["5000", "20000", "all"]
PCA_K_ORDER = [1, 2, 5, 10, 20, 50, 100, 128]

KEY_VARIANTS = [
    "global_raw_128d_float32_protein_equal_cap_all",
    "global_raw_128d_float64_protein_equal_cap_all",
    "bucketed_128d_float64_protein_equal_cap_all_manual_255_371_500",
    "global_raw_256d_float32_protein_equal_cap_all",
    "global_raw_256d_float64_protein_equal_cap_all",
    "global_raw_256d_float64_pair_equal_cap_all",
]

VARIANT_LABELS = {
    "global_raw_128d_float32_protein_equal_cap_5000": "128d f32 cap=5k",
    "global_raw_128d_float64_protein_equal_cap_5000": "128d f64 cap=5k",
    "global_raw_128d_float32_protein_equal_cap_20000": "128d f32 cap=20k",
    "global_raw_128d_float64_protein_equal_cap_20000": "128d f64 cap=20k",
    "global_raw_128d_float32_protein_equal_cap_all": "128d f32 cap=all",
    "global_raw_128d_float64_protein_equal_cap_all": "128d f64 cap=all",
    "bucketed_128d_float64_protein_equal_cap_all_manual_255_371_500": "128d f64 bucketed",
    "global_raw_256d_float32_protein_equal_cap_5000": "256d f32 cap=5k",
    "global_raw_256d_float64_protein_equal_cap_5000": "256d f64 cap=5k",
    "global_raw_256d_float32_protein_equal_cap_20000": "256d f32 cap=20k",
    "global_raw_256d_float64_protein_equal_cap_20000": "256d f64 cap=20k",
    "global_raw_256d_float32_protein_equal_cap_all": "256d f32 cap=all",
    "global_raw_256d_float64_protein_equal_cap_all": "256d f64 cap=all",
    "global_raw_256d_float64_pair_equal_cap_all": "256d f64 pair-equal",
}

VARIANT_COLORS = {
    "global_raw_128d_float32_protein_equal_cap_all": "#9c2f2f",
    "global_raw_128d_float64_protein_equal_cap_all": "#1f5a91",
    "bucketed_128d_float64_protein_equal_cap_all_manual_255_371_500": "#2a7f62",
    "global_raw_256d_float32_protein_equal_cap_all": "#b24b2a",
    "global_raw_256d_float64_protein_equal_cap_all": "#5b2c83",
    "global_raw_256d_float64_pair_equal_cap_all": "#7b8a2e",
}


def load_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def ensure_required_columns(path: Path, rows: list[dict[str, str]], required: set[str]) -> None:
    columns = set(rows[0].keys())
    missing = sorted(required - columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")


def as_float(value: str) -> float:
    if value == "" or value.lower() == "nan":
        return float("nan")
    return float(value)


def as_int(value: str) -> int:
    return int(float(value))


def format_value(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: format_value(row.get(name, "")) for name in fieldnames})


def safe_mean(values: list[float]) -> float:
    return float(np.mean(values)) if values else float("nan")


def safe_median(values: list[float]) -> float:
    return float(np.median(values)) if values else float("nan")


def safe_fraction(values: list[bool]) -> float:
    return float(np.mean(np.asarray(values, dtype=np.float64))) if values else float("nan")


def average_ranks(values: list[float]) -> list[float]:
    indexed = sorted(enumerate(values), key=lambda item: item[1])
    ranks = [0.0] * len(values)
    pos = 1
    idx = 0
    while idx < len(indexed):
        start = idx
        value = indexed[idx][1]
        while idx < len(indexed) and indexed[idx][1] == value:
            idx += 1
        avg_rank = (pos + (pos + (idx - start) - 1)) / 2.0
        for original_index, _ in indexed[start:idx]:
            ranks[original_index] = avg_rank
        pos += idx - start
    return ranks


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    x = np.asarray(xs, dtype=np.float64)
    y = np.asarray(ys, dtype=np.float64)
    x_std = float(np.std(x))
    y_std = float(np.std(y))
    if x_std <= 0.0 or y_std <= 0.0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def spearman_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or len(xs) < 2:
        return float("nan")
    return pearson_corr(average_ranks(xs), average_ranks(ys))


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


def index_summary_rows(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, str]]:
    return {row["variant_id"]: row for row in summary_rows}


def build_rowcap_summary(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for feature_set in ["128d", "256d"]:
        for dtype in ["float32", "float64"]:
            for row_cap in ROWCAP_ORDER:
                variant_id = f"global_raw_{feature_set}_{dtype}_protein_equal_cap_{row_cap}"
                row = next((item for item in summary_rows if item["variant_id"] == variant_id), None)
                if row is None:
                    continue
                rows.append(
                    {
                        "variant_id": variant_id,
                        "variant_label": VARIANT_LABELS.get(variant_id, variant_id),
                        "feature_set": feature_set,
                        "dtype": dtype,
                        "row_cap_per_train_protein": row_cap,
                        "test_mean_per_protein_r2": as_float(row["test_mean_per_protein_r2"]),
                        "test_median_per_protein_r2": as_float(row["test_median_per_protein_r2"]),
                        "test_negative_r2_fraction": as_float(row["test_negative_r2_fraction"]),
                        "elapsed_seconds": as_float(row["elapsed_seconds"]),
                    }
                )
    return rows


def build_pca_summary(summary_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    rows = []
    for k in PCA_K_ORDER:
        variant_id = f"pca_128d_float64_protein_equal_cap_all_pca_{k}"
        row = next((item for item in summary_rows if item["variant_id"] == variant_id), None)
        if row is None:
            continue
        rows.append(
            {
                "variant_id": variant_id,
                "pca_k": k,
                "test_mean_per_protein_r2": as_float(row["test_mean_per_protein_r2"]),
                "test_median_per_protein_r2": as_float(row["test_median_per_protein_r2"]),
                "test_negative_r2_fraction": as_float(row["test_negative_r2_fraction"]),
                "elapsed_seconds": as_float(row["elapsed_seconds"]),
            }
        )
    return rows


def build_bucket_r2_summary(per_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in per_rows:
        grouped[(row["variant_id"], row["split"], row["bucket_label"])].append(row)

    out_rows = []
    for variant_id, split_name, bucket_label in sorted(
        grouped.keys(),
        key=lambda item: (
            KEY_VARIANTS.index(item[0]) if item[0] in KEY_VARIANTS else 999,
            item[1],
            BUCKET_ORDER.index(item[2]) if item[2] in BUCKET_ORDER else 999,
        ),
    ):
        rows = grouped[(variant_id, split_name, bucket_label)]
        values = [as_float(row["per_protein_r2"]) for row in rows]
        out_rows.append(
            {
                "variant_id": variant_id,
                "variant_label": VARIANT_LABELS.get(variant_id, variant_id),
                "split": split_name,
                "bucket_label": bucket_label,
                "n_proteins": len(rows),
                "mean_per_protein_r2": safe_mean(values),
                "median_per_protein_r2": safe_median(values),
                "negative_r2_fraction": safe_fraction([value < 0.0 for value in values]),
                "mean_n_residues": safe_mean([as_float(row["n_residues"]) for row in rows]),
            }
        )
    return out_rows


def build_size_correlation_summary(per_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[tuple[str, str], list[dict[str, str]]] = defaultdict(list)
    for row in per_rows:
        grouped[(row["variant_id"], row["split"])].append(row)

    out_rows = []
    for variant_id, split_name in sorted(grouped.keys()):
        rows = grouped[(variant_id, split_name)]
        xs = [as_float(row["n_residues"]) for row in rows]
        ys = [as_float(row["per_protein_r2"]) for row in rows]
        bucket_means = {}
        for bucket in BUCKET_ORDER:
            bucket_values = [as_float(row["per_protein_r2"]) for row in rows if row["bucket_label"] == bucket]
            bucket_means[bucket] = safe_mean(bucket_values)
        out_rows.append(
            {
                "variant_id": variant_id,
                "variant_label": VARIANT_LABELS.get(variant_id, variant_id),
                "split": split_name,
                "pearson_size_vs_r2": pearson_corr(xs, ys),
                "spearman_size_vs_r2": spearman_corr(xs, ys),
                "mean_r2_bucket_le_255": bucket_means["<=255"],
                "mean_r2_bucket_256_371": bucket_means["256-371"],
                "mean_r2_bucket_372_500": bucket_means["372-500"],
                "mean_r2_bucket_501_plus": bucket_means["501+"],
                "small_vs_large_gap": (
                    bucket_means["<=255"] - bucket_means["501+"]
                    if not math.isnan(bucket_means["<=255"]) and not math.isnan(bucket_means["501+"])
                    else float("nan")
                ),
            }
        )
    return out_rows


def extract_bucket_coefficient_vectors(coef_rows: list[dict[str, str]]) -> dict[str, dict[str, np.ndarray]]:
    variant_id = "bucketed_128d_float64_protein_equal_cap_all_manual_255_371_500"
    variant_rows = [
        row
        for row in coef_rows
        if row["variant_id"] == variant_id and row["model_scope"] == "raw_space"
    ]

    vectors: dict[str, dict[str, np.ndarray]] = {}
    for coefficient_name, value_key in [
        ("standardized", "coefficient_standardized"),
        ("raw_scale", "coefficient_raw_scale"),
    ]:
        bucket_vectors: dict[str, np.ndarray] = {}
        for bucket_label in BUCKET_ORDER:
            bucket_rows = sorted(
                [row for row in variant_rows if row["bucket_label"] == bucket_label],
                key=lambda row: as_int(row["feature_index"]),
            )
            if not bucket_rows:
                continue
            bucket_vectors[bucket_label] = np.asarray(
                [as_float(row[value_key]) for row in bucket_rows],
                dtype=np.float64,
            )
        vectors[coefficient_name] = bucket_vectors
    return vectors


def build_bucket_cosine_rows(vectors: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    rows = []
    for coefficient_space, by_bucket in vectors.items():
        for left_bucket in BUCKET_ORDER:
            if left_bucket not in by_bucket:
                continue
            for right_bucket in BUCKET_ORDER:
                if right_bucket not in by_bucket:
                    continue
                rows.append(
                    {
                        "coefficient_space": coefficient_space,
                        "left_bucket_label": left_bucket,
                        "right_bucket_label": right_bucket,
                        "cosine_similarity": cosine_similarity(
                            by_bucket[left_bucket],
                            by_bucket[right_bucket],
                        ),
                        "left_norm": float(np.linalg.norm(by_bucket[left_bucket])),
                        "right_norm": float(np.linalg.norm(by_bucket[right_bucket])),
                    }
                )
    return rows


def build_bucket_vs_global_rows(coef_rows: list[dict[str, str]], vectors: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    global_variant_id = "global_raw_128d_float64_protein_equal_cap_all"
    global_rows = [
        row
        for row in coef_rows
        if row["variant_id"] == global_variant_id and row["model_scope"] == "raw_space"
    ]

    standardized_global = np.asarray(
        [as_float(row["coefficient_standardized"]) for row in sorted(global_rows, key=lambda row: as_int(row["feature_index"]))],
        dtype=np.float64,
    )
    raw_global = np.asarray(
        [as_float(row["coefficient_raw_scale"]) for row in sorted(global_rows, key=lambda row: as_int(row["feature_index"]))],
        dtype=np.float64,
    )

    rows = []
    for coefficient_space, global_vector in [
        ("standardized", standardized_global),
        ("raw_scale", raw_global),
    ]:
        for bucket_label, bucket_vector in vectors[coefficient_space].items():
            rows.append(
                {
                    "coefficient_space": coefficient_space,
                    "bucket_label": bucket_label,
                    "cosine_vs_global_128d_float64": cosine_similarity(bucket_vector, global_vector),
                    "bucket_norm": float(np.linalg.norm(bucket_vector)),
                    "global_norm": float(np.linalg.norm(global_vector)),
                }
            )
    return rows


def build_top_shift_rows(coef_rows: list[dict[str, str]], vectors: dict[str, dict[str, np.ndarray]]) -> list[dict[str, object]]:
    variant_id = "bucketed_128d_float64_protein_equal_cap_all_manual_255_371_500"
    variant_rows = [
        row
        for row in coef_rows
        if row["variant_id"] == variant_id and row["model_scope"] == "raw_space"
    ]
    feature_names = {
        as_int(row["feature_index"]): row["feature_name"]
        for row in variant_rows
        if row["bucket_label"] == BUCKET_ORDER[0]
    }

    rows = []
    for coefficient_space, by_bucket in vectors.items():
        for idx, left_bucket in enumerate(BUCKET_ORDER):
            if left_bucket not in by_bucket:
                continue
            for right_bucket in BUCKET_ORDER[idx + 1 :]:
                if right_bucket not in by_bucket:
                    continue
                diff = np.abs(by_bucket[left_bucket] - by_bucket[right_bucket])
                order = np.argsort(diff)[::-1][:20]
                for rank, feature_index in enumerate(order, start=1):
                    rows.append(
                        {
                            "coefficient_space": coefficient_space,
                            "left_bucket_label": left_bucket,
                            "right_bucket_label": right_bucket,
                            "rank": rank,
                            "feature_index": int(feature_index),
                            "feature_name": feature_names[int(feature_index)],
                            "abs_difference": float(diff[feature_index]),
                            "left_value": float(by_bucket[left_bucket][feature_index]),
                            "right_value": float(by_bucket[right_bucket][feature_index]),
                        }
                    )
    return rows


def build_spectrum_interpretation_rows(spectrum_rows: list[dict[str, str]], singular_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped_singular: dict[tuple[str, str, str], list[dict[str, str]]] = defaultdict(list)
    for row in singular_rows:
        grouped_singular[(row["feature_set"], row["weighting_scheme"], row["row_cap_per_train_protein"])].append(row)

    out_rows = []
    for row in spectrum_rows:
        key = (row["feature_set"], row["weighting_scheme"], row["row_cap_per_train_protein"])
        singular_subset = grouped_singular.get(key, [])
        kept_count = sum(1 for item in singular_subset if item["cut_by_float32"].lower() in {"false", "0"})
        out_rows.append(
            {
                "feature_set": row["feature_set"],
                "weighting_scheme": row["weighting_scheme"],
                "row_cap_per_train_protein": row["row_cap_per_train_protein"],
                "feature_dim": as_int(row["feature_dim"]),
                "cond_float32": as_float(row["cond_float32"]),
                "cutoff_float32": as_float(row["cutoff_float32"]),
                "n_cutoff_float32": as_int(row["n_cutoff_float32"]),
                "n_cutoff_float64": as_int(row["n_cutoff_float64"]),
                "n_directions_kept_by_float32": kept_count if singular_subset else (as_int(row["feature_dim"]) - as_int(row["n_cutoff_float32"])),
                "fraction_cut_by_float32": (
                    as_float(row["n_cutoff_float32"]) / max(as_float(row["feature_dim"]), 1.0)
                ),
                "max_singular_value": as_float(row["max_singular_value"]),
                "min_positive_singular_value": as_float(row["min_positive_singular_value"]),
                "effective_rank": as_float(row["effective_rank"]),
            }
        )
    return out_rows


def plot_rowcap_precision(rowcap_rows: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.5), sharey=True)
    for ax, feature_set in zip(axes, ["128d", "256d"]):
        for dtype, color in [("float32", "#a33a2f"), ("float64", "#1f5a91")]:
            subset = [
                row for row in rowcap_rows
                if row["feature_set"] == feature_set and row["dtype"] == dtype
            ]
            subset.sort(key=lambda row: ROWCAP_ORDER.index(str(row["row_cap_per_train_protein"])))
            xs = np.arange(len(subset))
            ys = [float(row["test_mean_per_protein_r2"]) for row in subset]
            ax.plot(xs, ys, marker="o", linewidth=2.0, color=color, label=dtype)
        ax.set_xticks(np.arange(len(ROWCAP_ORDER)))
        ax.set_xticklabels(ROWCAP_ORDER)
        ax.set_title(f"{feature_set} Test Mean Per-Protein R^2")
        ax.set_xlabel("Train Row Cap Per Protein")
        ax.grid(True, alpha=0.25)
    axes[0].set_ylabel("Test Mean Per-Protein R^2")
    axes[1].legend(frameon=False, title="dtype")
    fig.tight_layout()
    fig.savefig(ROWCAP_PLOT, dpi=180)
    plt.close(fig)


def plot_pca_sweep(pca_rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(6.8, 4.4))
    xs = [int(row["pca_k"]) for row in pca_rows]
    ys = [float(row["test_mean_per_protein_r2"]) for row in pca_rows]
    ax.plot(xs, ys, marker="o", linewidth=2.0, color="#5b2c83")
    ax.set_xscale("log")
    ax.set_xticks(PCA_K_ORDER)
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.set_xlabel("Top-k Principal Components")
    ax.set_ylabel("Test Mean Per-Protein R^2")
    ax.set_title("128d Float64 PCA Sweep")
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(PCA_PLOT, dpi=180)
    plt.close(fig)


def plot_bucket_r2(bucket_rows: list[dict[str, object]]) -> None:
    selected = [
        row for row in bucket_rows
        if row["split"] == "test" and row["variant_id"] in KEY_VARIANTS
    ]
    fig, ax = plt.subplots(figsize=(11.5, 5.0))
    x = np.arange(len(BUCKET_ORDER))
    width = 0.12
    offsets = np.linspace(-2.5 * width, 2.5 * width, num=len(KEY_VARIANTS))

    for offset, variant_id in zip(offsets, KEY_VARIANTS):
        subset = {row["bucket_label"]: row for row in selected if row["variant_id"] == variant_id}
        ys = [float(subset[bucket]["mean_per_protein_r2"]) for bucket in BUCKET_ORDER]
        ax.bar(
            x + offset,
            ys,
            width=width,
            label=VARIANT_LABELS.get(variant_id, variant_id),
            color=VARIANT_COLORS.get(variant_id),
        )

    ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.6)
    ax.set_xticks(x)
    ax.set_xticklabels(BUCKET_ORDER)
    ax.set_ylabel("Mean Test Per-Protein R^2")
    ax.set_title("Test R^2 by Protein-Size Bucket")
    ax.grid(True, axis="y", alpha=0.25)
    ax.legend(frameon=False, fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(BUCKET_R2_PLOT, dpi=180)
    plt.close(fig)


def plot_bucket_cosine_heatmap(cosine_rows: list[dict[str, object]]) -> None:
    selected = [
        row for row in cosine_rows
        if row["coefficient_space"] == "raw_scale"
    ]
    matrix = np.zeros((len(BUCKET_ORDER), len(BUCKET_ORDER)), dtype=np.float64)
    for row in selected:
        i = BUCKET_ORDER.index(str(row["left_bucket_label"]))
        j = BUCKET_ORDER.index(str(row["right_bucket_label"]))
        matrix[i, j] = float(row["cosine_similarity"])

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(matrix, cmap="viridis", vmin=0.55, vmax=1.0)
    ax.set_xticks(np.arange(len(BUCKET_ORDER)))
    ax.set_yticks(np.arange(len(BUCKET_ORDER)))
    ax.set_xticklabels(BUCKET_ORDER, rotation=30, ha="right")
    ax.set_yticklabels(BUCKET_ORDER)
    ax.set_title("Bucketed 128d Raw-Scale Coefficient Cosine Similarity")

    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(j, i, f"{matrix[i, j]:.3f}", ha="center", va="center", color="white", fontsize=9)

    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(BUCKET_COSINE_PLOT, dpi=180)
    plt.close(fig)


def plot_size_vs_r2(per_rows: list[dict[str, str]]) -> None:
    selected = [row for row in per_rows if row["split"] == "test" and row["variant_id"] in KEY_VARIANTS]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 7.8), sharex=True, sharey=True)
    axes = axes.ravel()

    for ax, variant_id in zip(axes, KEY_VARIANTS):
        subset = [row for row in selected if row["variant_id"] == variant_id]
        xs = [as_float(row["n_residues"]) for row in subset]
        ys = [as_float(row["per_protein_r2"]) for row in subset]
        ax.scatter(xs, ys, s=28, alpha=0.8, color=VARIANT_COLORS.get(variant_id, "#333333"))
        ax.axhline(0.0, color="#444444", linewidth=1.0, alpha=0.5)
        corr = pearson_corr(xs, ys)
        ax.set_title(f"{VARIANT_LABELS.get(variant_id, variant_id)}\nPearson={corr:.3f}")
        ax.grid(True, alpha=0.2)

    for ax in axes[3:]:
        ax.set_xlabel("n_residues")
    for ax in axes[::3]:
        ax.set_ylabel("Test Per-Protein R^2")

    fig.tight_layout()
    fig.savefig(SIZE_SCATTER_PLOT, dpi=180)
    plt.close(fig)


def write_report(
    summary_lookup: dict[str, dict[str, str]],
    bucket_rows: list[dict[str, object]],
    size_corr_rows: list[dict[str, object]],
    bucket_cos_rows: list[dict[str, object]],
    bucket_global_rows: list[dict[str, object]],
    top_shift_rows: list[dict[str, object]],
    spectrum_rows: list[dict[str, object]],
) -> None:
    bucket_lookup = {
        (row["variant_id"], row["split"], row["bucket_label"]): row
        for row in bucket_rows
    }
    size_corr_lookup = {
        (row["variant_id"], row["split"]): row
        for row in size_corr_rows
    }

    global_128_f32 = summary_lookup["global_raw_128d_float32_protein_equal_cap_all"]
    global_128_f64 = summary_lookup["global_raw_128d_float64_protein_equal_cap_all"]
    bucketed_128 = summary_lookup["bucketed_128d_float64_protein_equal_cap_all_manual_255_371_500"]
    global_256_f32 = summary_lookup["global_raw_256d_float32_protein_equal_cap_all"]
    global_256_f64 = summary_lookup["global_raw_256d_float64_protein_equal_cap_all"]
    pair_256_f64 = summary_lookup["global_raw_256d_float64_pair_equal_cap_all"]

    def bucket_mean(variant_id: str, bucket_label: str) -> float:
        return float(bucket_lookup[(variant_id, "test", bucket_label)]["mean_per_protein_r2"])

    def corr_value(variant_id: str) -> float:
        return float(size_corr_lookup[(variant_id, "test")]["pearson_size_vs_r2"])

    raw_cos_lookup = {
        (row["left_bucket_label"], row["right_bucket_label"]): row
        for row in bucket_cos_rows
        if row["coefficient_space"] == "raw_scale"
    }
    raw_global_lookup = {
        row["bucket_label"]: row
        for row in bucket_global_rows
        if row["coefficient_space"] == "raw_scale"
    }
    strongest_shift_rows = [
        row
        for row in top_shift_rows
        if row["coefficient_space"] == "raw_scale"
        and row["left_bucket_label"] == "<=255"
        and row["right_bucket_label"] == "501+"
    ][:5]

    with REPORT_MD.open("w") as handle:
        handle.write("# Diagnostic Precision / Row-Cap / PCA Analysis\n\n")
        handle.write("## Main Conclusions\n")
        handle.write(
            f"- Float precision is clearly a major part of the old failure mode: the all-row protein-equal global fits jump from "
            f"`128d float32 = {as_float(global_128_f32['test_mean_per_protein_r2']):.4f}` to "
            f"`128d float64 = {as_float(global_128_f64['test_mean_per_protein_r2']):.4f}`, and from "
            f"`256d float32 = {as_float(global_256_f32['test_mean_per_protein_r2']):.4f}` to "
            f"`256d float64 = {as_float(global_256_f64['test_mean_per_protein_r2']):.4f}`.\n"
        )
        handle.write(
            "- Row-capping explains why the older small / capped analyses could look much healthier in float32: "
            "for both 128d and 256d, the float32 variants are strong at `5k` and `20k` rows per protein but collapse only when all rows are used.\n"
        )
        handle.write(
            f"- Float precision is not the whole story. Even after switching to float64, test `R^2` still rises with protein size. "
            f"For the global all-row float64 models, Pearson correlation between `n_residues` and test per-protein `R^2` is "
            f"`{corr_value('global_raw_128d_float64_protein_equal_cap_all'):.3f}` for 128d and "
            f"`{corr_value('global_raw_256d_float64_protein_equal_cap_all'):.3f}` for 256d.\n"
        )
        handle.write(
            f"- Manual size bucketing still matters after fixing precision. The `128d` bucketed float64 model improves overall test mean per-protein `R^2` from "
            f"`{as_float(global_128_f64['test_mean_per_protein_r2']):.4f}` to "
            f"`{as_float(bucketed_128['test_mean_per_protein_r2']):.4f}`.\n"
        )
        handle.write(
            f"- The bucketed model mostly helps the smallest proteins: in the `<=255` bucket, test mean per-protein `R^2` goes from "
            f"`{bucket_mean('global_raw_128d_float64_protein_equal_cap_all', '<=255'):.4f}` to "
            f"`{bucket_mean('bucketed_128d_float64_protein_equal_cap_all_manual_255_371_500', '<=255'):.4f}`. "
            f"It reduces the size-dependence correlation from `{corr_value('global_raw_128d_float64_protein_equal_cap_all'):.3f}` to "
            f"`{corr_value('bucketed_128d_float64_protein_equal_cap_all_manual_255_371_500'):.3f}`.\n"
        )
        handle.write(
            f"- Training weights still matter in float64. The `256d` float64 pair-equal model reaches "
            f"`{as_float(pair_256_f64['test_mean_per_protein_r2']):.4f}`, below the protein-equal float64 result "
            f"`{as_float(global_256_f64['test_mean_per_protein_r2']):.4f}`. The pair-equal model especially gives back performance on the smallest proteins.\n"
        )
        handle.write(
            "- The PCA sweep suggests the 128d decoder has substantial low-rank structure, but not an extremely tiny intrinsic dimension. "
            "The top `20` PCs already recover a lot, but the curve keeps improving through `50`, `100`, and `128`.\n\n"
        )

        handle.write("## Size-Bucket Test R^2\n")
        handle.write("| Variant | <=255 | 256-371 | 372-500 | 501+ |\n")
        handle.write("| --- | ---: | ---: | ---: | ---: |\n")
        for variant_id in [
            "global_raw_128d_float32_protein_equal_cap_all",
            "global_raw_128d_float64_protein_equal_cap_all",
            "bucketed_128d_float64_protein_equal_cap_all_manual_255_371_500",
            "global_raw_256d_float32_protein_equal_cap_all",
            "global_raw_256d_float64_protein_equal_cap_all",
            "global_raw_256d_float64_pair_equal_cap_all",
        ]:
            handle.write(
                f"| {VARIANT_LABELS.get(variant_id, variant_id)} "
                f"| {bucket_mean(variant_id, '<=255'):.4f} "
                f"| {bucket_mean(variant_id, '256-371'):.4f} "
                f"| {bucket_mean(variant_id, '372-500'):.4f} "
                f"| {bucket_mean(variant_id, '501+'):.4f} |\n"
            )
        handle.write("\n")

        handle.write("## Bucketed-Coefficient Similarity\n")
        handle.write(
            f"- Raw-scale cosine similarity between the smallest and largest bucket decoders is "
            f"`{float(raw_cos_lookup[('<=255', '501+')]['cosine_similarity']):.4f}`.\n"
        )
        handle.write(
            f"- The most similar pair is `256-371` vs `372-500` at "
            f"`{float(raw_cos_lookup[('256-371', '372-500')]['cosine_similarity']):.4f}`.\n"
        )
        handle.write(
            f"- Relative to the global `128d` float64 decoder, cosine similarity is "
            f"`{float(raw_global_lookup['<=255']['cosine_vs_global_128d_float64']):.4f}` for `<=255` and "
            f"`{float(raw_global_lookup['501+']['cosine_vs_global_128d_float64']):.4f}` for `501+`.\n\n"
        )

        handle.write("Top raw-scale coefficient shifts between `<=255` and `501+`:\n")
        for row in strongest_shift_rows:
            handle.write(
                f"- `{row['feature_name']}`: abs diff `{float(row['abs_difference']):.4f}` "
                f"(`<=255`={float(row['left_value']):.4f}, `501+`={float(row['right_value']):.4f})\n"
            )
        handle.write("\n")

        handle.write("## Spectrum Interpretation\n")
        for feature_set in ["128d", "256d"]:
            all_row = next(
                row
                for row in spectrum_rows
                if row["feature_set"] == feature_set
                and row["weighting_scheme"] == "protein_equal"
                and row["row_cap_per_train_protein"] == "all"
            )
            handle.write(
                f"- {feature_set} protein-equal all rows: float32 cutoff removes "
                f"`{int(all_row['n_cutoff_float32'])}` of `{int(all_row['feature_dim'])}` singular directions, "
                f"leaving `{int(all_row['n_directions_kept_by_float32'])}` directions above the heuristic threshold.\n"
            )

        handle.write("\n## Key Output Files\n")
        for path in [
            ROWCAP_PLOT,
            PCA_PLOT,
            BUCKET_R2_PLOT,
            BUCKET_COSINE_PLOT,
            SIZE_SCATTER_PLOT,
            OVERVIEW_CSV,
            BUCKET_R2_CSV,
            SIZE_CORR_CSV,
            BUCKET_COSINE_CSV,
            TOP_SHIFT_CSV,
            SPECTRUM_INTERPRET_CSV,
        ]:
            handle.write(f"- `{path}`\n")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    summary_rows = load_csv_rows(SUMMARY_CSV)
    per_rows = load_csv_rows(PER_PROTEIN_CSV)
    coef_rows = load_csv_rows(COEFFICIENT_CSV)
    spectrum_rows = load_csv_rows(SPECTRUM_SUMMARY_CSV)
    singular_rows = load_csv_rows(SINGULAR_VALUES_CSV)

    ensure_required_columns(
        SUMMARY_CSV,
        summary_rows,
        {
            "variant_id",
            "feature_set",
            "dtype",
            "row_cap_per_train_protein",
            "test_mean_per_protein_r2",
            "test_negative_r2_fraction",
        },
    )
    ensure_required_columns(
        PER_PROTEIN_CSV,
        per_rows,
        {"variant_id", "split", "bucket_label", "n_residues", "per_protein_r2"},
    )
    ensure_required_columns(
        COEFFICIENT_CSV,
        coef_rows,
        {
            "variant_id",
            "bucket_label",
            "model_scope",
            "feature_index",
            "feature_name",
            "coefficient_standardized",
            "coefficient_raw_scale",
        },
    )
    ensure_required_columns(
        SPECTRUM_SUMMARY_CSV,
        spectrum_rows,
        {
            "feature_set",
            "weighting_scheme",
            "row_cap_per_train_protein",
            "feature_dim",
            "n_cutoff_float32",
            "effective_rank",
        },
    )
    ensure_required_columns(
        SINGULAR_VALUES_CSV,
        singular_rows,
        {
            "feature_set",
            "weighting_scheme",
            "row_cap_per_train_protein",
            "rank_index",
            "cut_by_float32",
        },
    )

    summary_lookup = index_summary_rows(summary_rows)

    overview_rows = []
    for variant_id in KEY_VARIANTS:
        row = summary_lookup[variant_id]
        overview_rows.append(
            {
                "variant_id": variant_id,
                "variant_label": VARIANT_LABELS.get(variant_id, variant_id),
                "feature_set": row["feature_set"],
                "dtype": row["dtype"],
                "weighting_scheme": row["weighting_scheme"],
                "row_cap_per_train_protein": row["row_cap_per_train_protein"],
                "bucket_scheme": row["bucket_scheme"],
                "test_mean_per_protein_r2": as_float(row["test_mean_per_protein_r2"]),
                "test_median_per_protein_r2": as_float(row["test_median_per_protein_r2"]),
                "test_negative_r2_fraction": as_float(row["test_negative_r2_fraction"]),
                "elapsed_seconds": as_float(row["elapsed_seconds"]),
            }
        )

    rowcap_rows = build_rowcap_summary(summary_rows)
    pca_rows = build_pca_summary(summary_rows)
    bucket_r2_rows = build_bucket_r2_summary(per_rows)
    size_corr_rows = build_size_correlation_summary(per_rows)
    bucket_vectors = extract_bucket_coefficient_vectors(coef_rows)
    bucket_cos_rows = build_bucket_cosine_rows(bucket_vectors)
    bucket_global_rows = build_bucket_vs_global_rows(coef_rows, bucket_vectors)
    top_shift_rows = build_top_shift_rows(coef_rows, bucket_vectors)
    spectrum_interpret_rows = build_spectrum_interpretation_rows(spectrum_rows, singular_rows)

    write_csv(
        OVERVIEW_CSV,
        [
            "variant_id",
            "variant_label",
            "feature_set",
            "dtype",
            "weighting_scheme",
            "row_cap_per_train_protein",
            "bucket_scheme",
            "test_mean_per_protein_r2",
            "test_median_per_protein_r2",
            "test_negative_r2_fraction",
            "elapsed_seconds",
        ],
        overview_rows,
    )
    write_csv(
        ROWCAP_CSV,
        [
            "variant_id",
            "variant_label",
            "feature_set",
            "dtype",
            "row_cap_per_train_protein",
            "test_mean_per_protein_r2",
            "test_median_per_protein_r2",
            "test_negative_r2_fraction",
            "elapsed_seconds",
        ],
        rowcap_rows,
    )
    write_csv(
        PCA_CSV,
        [
            "variant_id",
            "pca_k",
            "test_mean_per_protein_r2",
            "test_median_per_protein_r2",
            "test_negative_r2_fraction",
            "elapsed_seconds",
        ],
        pca_rows,
    )
    write_csv(
        BUCKET_R2_CSV,
        [
            "variant_id",
            "variant_label",
            "split",
            "bucket_label",
            "n_proteins",
            "mean_per_protein_r2",
            "median_per_protein_r2",
            "negative_r2_fraction",
            "mean_n_residues",
        ],
        bucket_r2_rows,
    )
    write_csv(
        SIZE_CORR_CSV,
        [
            "variant_id",
            "variant_label",
            "split",
            "pearson_size_vs_r2",
            "spearman_size_vs_r2",
            "mean_r2_bucket_le_255",
            "mean_r2_bucket_256_371",
            "mean_r2_bucket_372_500",
            "mean_r2_bucket_501_plus",
            "small_vs_large_gap",
        ],
        size_corr_rows,
    )
    write_csv(
        BUCKET_COSINE_CSV,
        [
            "coefficient_space",
            "left_bucket_label",
            "right_bucket_label",
            "cosine_similarity",
            "left_norm",
            "right_norm",
        ],
        bucket_cos_rows,
    )
    write_csv(
        BUCKET_GLOBAL_COSINE_CSV,
        [
            "coefficient_space",
            "bucket_label",
            "cosine_vs_global_128d_float64",
            "bucket_norm",
            "global_norm",
        ],
        bucket_global_rows,
    )
    write_csv(
        TOP_SHIFT_CSV,
        [
            "coefficient_space",
            "left_bucket_label",
            "right_bucket_label",
            "rank",
            "feature_index",
            "feature_name",
            "abs_difference",
            "left_value",
            "right_value",
        ],
        top_shift_rows,
    )
    write_csv(
        SPECTRUM_INTERPRET_CSV,
        [
            "feature_set",
            "weighting_scheme",
            "row_cap_per_train_protein",
            "feature_dim",
            "cond_float32",
            "cutoff_float32",
            "n_cutoff_float32",
            "n_cutoff_float64",
            "n_directions_kept_by_float32",
            "fraction_cut_by_float32",
            "max_singular_value",
            "min_positive_singular_value",
            "effective_rank",
        ],
        spectrum_interpret_rows,
    )

    plot_rowcap_precision(rowcap_rows)
    plot_pca_sweep(pca_rows)
    plot_bucket_r2(bucket_r2_rows)
    plot_bucket_cosine_heatmap(bucket_cos_rows)
    plot_size_vs_r2(per_rows)

    write_report(
        summary_lookup=summary_lookup,
        bucket_rows=bucket_r2_rows,
        size_corr_rows=size_corr_rows,
        bucket_cos_rows=bucket_cos_rows,
        bucket_global_rows=bucket_global_rows,
        top_shift_rows=top_shift_rows,
        spectrum_rows=spectrum_interpret_rows,
    )

    print("Wrote analysis outputs to:")
    print(f"  {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_delta_per_protein_visualizations.py
===========================================
Generate fixed-path visualizations and summaries for Delta per-protein regression outputs.

Inputs (hardcoded):
    delta_files/per_protein_files/
        per_protein_regression_metrics.csv
        per_protein_mds_tm_scores.csv
        per_protein_coefficients.csv

Outputs (hardcoded):
    delta_files/per_protein_files/visualizations/
"""

from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

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
INPUT_DIR = PROJECT_ROOT / "delta_files" / "per_protein_files"
OUTPUT_DIR = INPUT_DIR / "visualizations"
PAIR_GRID_DIR = OUTPUT_DIR / "protein_pair_grids"
MEAN_LAYER_DIR = OUTPUT_DIR / "mean_layer_similarity"

METRICS_CSV = INPUT_DIR / "per_protein_regression_metrics.csv"
TM_CSV = INPUT_DIR / "per_protein_mds_tm_scores.csv"
COEF_CSV = INPUT_DIR / "per_protein_coefficients.csv"

EXPECTED_LAYERS = list(range(48))
SELECTED_GRID_LAYERS = [0, 5, 10, 20, 30, 40, 47]
EXPECTED_N_PROTEINS = 184

BUCKET_ORDER = ["small", "medium", "large", "very_large"]
BUCKET_COLORS = {
    "small": "#4c78a8",
    "medium": "#f58518",
    "large": "#54a24b",
    "very_large": "#b279a2",
}


def read_csv_rows(path: Path) -> List[dict]:
    with path.open("r", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows


def ensure_required_columns(path: Path, rows: List[dict], required: Iterable[str]) -> None:
    columns = set(rows[0].keys())
    missing = sorted(set(required) - columns)
    if missing:
        raise ValueError(f"{path.name} missing required columns: {missing}")


def parse_metrics(rows: List[dict]) -> Dict[Tuple[str, int], dict]:
    required = {
        "protein_id",
        "layer",
        "n_residues",
        "r2_train",
        "r2_test",
    }
    ensure_required_columns(METRICS_CSV, rows, required)

    parsed: Dict[Tuple[str, int], dict] = {}
    for row in rows:
        protein_id = row["protein_id"]
        layer = int(row["layer"])
        key = (protein_id, layer)
        if key in parsed:
            raise ValueError(f"Duplicate metrics row for {key}")
        parsed[key] = {
            "protein_id": protein_id,
            "layer": layer,
            "n_residues": int(row["n_residues"]),
            "r2_train": float(row["r2_train"]),
            "r2_test": float(row["r2_test"]),
        }
    return parsed


def parse_tm(rows: List[dict]) -> Dict[Tuple[str, int], dict]:
    required = {"protein_id", "layer", "n_residues", "tm_score", "rmsd"}
    ensure_required_columns(TM_CSV, rows, required)

    parsed: Dict[Tuple[str, int], dict] = {}
    for row in rows:
        protein_id = row["protein_id"]
        layer = int(row["layer"])
        key = (protein_id, layer)
        if key in parsed:
            raise ValueError(f"Duplicate TM row for {key}")
        parsed[key] = {
            "protein_id": protein_id,
            "layer": layer,
            "n_residues": int(row["n_residues"]),
            "tm_score": float(row["tm_score"]),
            "rmsd": float(row["rmsd"]),
        }
    return parsed


def parse_coefficients(rows: List[dict]) -> Dict[Tuple[str, int], np.ndarray]:
    coef_cols = [f"coef_{i}" for i in range(256)]
    required = {"protein_id", "layer", *coef_cols}
    ensure_required_columns(COEF_CSV, rows, required)

    parsed: Dict[Tuple[str, int], np.ndarray] = {}
    for row in rows:
        protein_id = row["protein_id"]
        layer = int(row["layer"])
        key = (protein_id, layer)
        if key in parsed:
            raise ValueError(f"Duplicate coefficients row for {key}")
        vector = np.array([float(row[c]) for c in coef_cols], dtype=np.float64)
        parsed[key] = vector
    return parsed


def assign_size_bucket(n_residues: int) -> str:
    if n_residues < 100:
        return "small"
    if n_residues < 250:
        return "medium"
    if n_residues < 500:
        return "large"
    return "very_large"


def validate_and_index_data(
    metrics: Dict[Tuple[str, int], dict],
    tm: Dict[Tuple[str, int], dict],
    coefficients: Dict[Tuple[str, int], np.ndarray],
):
    metric_keys = set(metrics.keys())
    tm_keys = set(tm.keys())
    coef_keys = set(coefficients.keys())

    if metric_keys != tm_keys:
        raise ValueError(
            "Metrics/TM key mismatch: "
            f"only_metrics={len(metric_keys - tm_keys)}, only_tm={len(tm_keys - metric_keys)}"
        )
    if metric_keys != coef_keys:
        raise ValueError(
            "Metrics/Coefficients key mismatch: "
            f"only_metrics={len(metric_keys - coef_keys)}, only_coef={len(coef_keys - metric_keys)}"
        )

    proteins = sorted({pid for pid, _ in metric_keys})
    if len(proteins) != EXPECTED_N_PROTEINS:
        raise ValueError(
            f"Expected {EXPECTED_N_PROTEINS} proteins, found {len(proteins)}"
        )

    protein_sizes = {}
    for pid in proteins:
        layers = sorted(layer for p, layer in metric_keys if p == pid)
        if layers != EXPECTED_LAYERS:
            raise ValueError(f"Protein {pid} missing layers or out-of-order layers: {layers}")

        residue_counts = {
            metrics[(pid, layer)]["n_residues"]
            for layer in EXPECTED_LAYERS
        }
        if len(residue_counts) != 1:
            raise ValueError(f"Protein {pid} has inconsistent n_residues in metrics")

        tm_counts = {
            tm[(pid, layer)]["n_residues"]
            for layer in EXPECTED_LAYERS
        }
        if len(tm_counts) != 1:
            raise ValueError(f"Protein {pid} has inconsistent n_residues in TM table")

        n_metrics = next(iter(residue_counts))
        n_tm = next(iter(tm_counts))
        if n_metrics != n_tm:
            raise ValueError(f"Protein {pid} n_residues mismatch metrics({n_metrics}) vs TM({n_tm})")

        protein_sizes[pid] = n_metrics

    sorted_proteins = [
        pid for pid, _ in sorted(protein_sizes.items(), key=lambda item: (item[1], item[0]))
    ]
    protein_to_idx = {pid: idx for idx, pid in enumerate(sorted_proteins)}

    bucket_by_protein = {pid: assign_size_bucket(protein_sizes[pid]) for pid in sorted_proteins}
    bucket_indices = {
        bucket: [protein_to_idx[pid] for pid in sorted_proteins if bucket_by_protein[pid] == bucket]
        for bucket in BUCKET_ORDER
    }

    if sum(len(v) for v in bucket_indices.values()) != len(sorted_proteins):
        raise ValueError("Size bucket indexing failed: bucket counts do not sum to total proteins")

    return sorted_proteins, protein_sizes, protein_to_idx, bucket_by_protein, bucket_indices


def build_layer_matrices(
    sorted_proteins: List[str],
    metrics: Dict[Tuple[str, int], dict],
    tm: Dict[Tuple[str, int], dict],
    coefficients: Dict[Tuple[str, int], np.ndarray],
):
    n_proteins = len(sorted_proteins)
    n_layers = len(EXPECTED_LAYERS)

    r2_train = np.full((n_proteins, n_layers), np.nan, dtype=np.float64)
    r2_test = np.full((n_proteins, n_layers), np.nan, dtype=np.float64)
    tm_score = np.full((n_proteins, n_layers), np.nan, dtype=np.float64)
    coef_tensor = np.full((n_layers, n_proteins, 256), np.nan, dtype=np.float64)

    for p_idx, pid in enumerate(sorted_proteins):
        for layer in EXPECTED_LAYERS:
            key = (pid, layer)
            r2_train[p_idx, layer] = metrics[key]["r2_train"]
            r2_test[p_idx, layer] = metrics[key]["r2_test"]
            tm_score[p_idx, layer] = tm[key]["tm_score"]
            coef_tensor[layer, p_idx] = coefficients[key]

    for name, arr in [
        ("r2_train", r2_train),
        ("r2_test", r2_test),
        ("tm_score", tm_score),
        ("coef_tensor", coef_tensor),
    ]:
        if np.isnan(arr).any():
            raise ValueError(f"NaN found in computed {name} matrix")

    return r2_train, r2_test, tm_score, coef_tensor


def summarize_values(values: np.ndarray) -> dict:
    abs_vals = np.abs(values)
    return {
        "count": int(values.size),
        "mean": float(np.mean(values)),
        "median": float(np.median(values)),
        "std": float(np.std(values)),
        "mean_abs": float(np.mean(abs_vals)),
        "p95_abs": float(np.percentile(abs_vals, 95.0)),
        "min": float(np.min(values)),
        "max": float(np.max(values)),
        "frac_negative": float(np.mean(values < 0.0)),
    }


def write_gap_outputs(gaps: np.ndarray) -> None:
    flat = gaps.reshape(-1)
    global_stats = summarize_values(flat)
    layer47_stats = summarize_values(gaps[:, 47])

    summary_path = OUTPUT_DIR / "r2_train_vs_test_gap_summary.txt"
    with summary_path.open("w") as handle:
        handle.write("Train vs Test R2 Gap Summary\n")
        handle.write("============================\n\n")
        handle.write("Definition: gap = r2_train - r2_test\n\n")

        handle.write("All protein-layer rows\n")
        handle.write(f"  Count:         {global_stats['count']}\n")
        handle.write(f"  Mean:          {global_stats['mean']:.6f}\n")
        handle.write(f"  Median:        {global_stats['median']:.6f}\n")
        handle.write(f"  Std:           {global_stats['std']:.6f}\n")
        handle.write(f"  Mean abs:      {global_stats['mean_abs']:.6f}\n")
        handle.write(f"  P95 abs:       {global_stats['p95_abs']:.6f}\n")
        handle.write(f"  Min:           {global_stats['min']:.6f}\n")
        handle.write(f"  Max:           {global_stats['max']:.6f}\n")
        handle.write(f"  Fraction < 0:  {global_stats['frac_negative']:.6f}\n\n")

        handle.write("Layer 47 across proteins\n")
        handle.write(f"  Count:         {layer47_stats['count']}\n")
        handle.write(f"  Mean:          {layer47_stats['mean']:.6f}\n")
        handle.write(f"  Median:        {layer47_stats['median']:.6f}\n")
        handle.write(f"  Std:           {layer47_stats['std']:.6f}\n")
        handle.write(f"  Mean abs:      {layer47_stats['mean_abs']:.6f}\n")
        handle.write(f"  P95 abs:       {layer47_stats['p95_abs']:.6f}\n")
        handle.write(f"  Min:           {layer47_stats['min']:.6f}\n")
        handle.write(f"  Max:           {layer47_stats['max']:.6f}\n")
        handle.write(f"  Fraction < 0:  {layer47_stats['frac_negative']:.6f}\n")

    csv_path = OUTPUT_DIR / "r2_train_vs_test_gap_by_layer.csv"
    with csv_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "layer",
            "mean_gap",
            "median_gap",
            "std_gap",
            "mean_abs_gap",
            "p95_abs_gap",
            "min_gap",
            "max_gap",
            "frac_negative_gap",
        ])
        for layer in EXPECTED_LAYERS:
            stats = summarize_values(gaps[:, layer])
            writer.writerow([
                layer,
                f"{stats['mean']:.6f}",
                f"{stats['median']:.6f}",
                f"{stats['std']:.6f}",
                f"{stats['mean_abs']:.6f}",
                f"{stats['p95_abs']:.6f}",
                f"{stats['min']:.6f}",
                f"{stats['max']:.6f}",
                f"{stats['frac_negative']:.6f}",
            ])


def plot_layer_trajectories(
    matrix: np.ndarray,
    bucket_indices: Dict[str, List[int]],
    y_label: str,
    title: str,
    out_path: Path,
) -> None:
    layers = np.array(EXPECTED_LAYERS, dtype=int)
    fig, ax = plt.subplots(figsize=(11, 6))

    for idx in range(matrix.shape[0]):
        ax.plot(layers, matrix[idx], color="#9aa5b1", linewidth=0.6, alpha=0.35)

    overall = np.mean(matrix, axis=0)
    ax.plot(layers, overall, color="black", linewidth=3.2, label="All proteins mean")

    for bucket in BUCKET_ORDER:
        indices = bucket_indices[bucket]
        if not indices:
            continue
        bucket_mean = np.mean(matrix[indices], axis=0)
        ax.plot(
            layers,
            bucket_mean,
            color=BUCKET_COLORS[bucket],
            linewidth=2.2,
            label=f"{bucket.replace('_', ' ').title()} mean (n={len(indices)})",
        )

    ax.set_xlabel("Layer")
    ax.set_ylabel(y_label)
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best", fontsize=9)
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = vectors / safe_norms
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        result = normalized @ normalized.T
    return np.clip(result, -1.0, 1.0)


def euclidean_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    sq = np.sum(vectors * vectors, axis=1, keepdims=True)
    with np.errstate(over="ignore", invalid="ignore", divide="ignore"):
        dist_sq = sq + sq.T - 2.0 * (vectors @ vectors.T)
    dist_sq = np.maximum(dist_sq, 0.0)
    return np.sqrt(dist_sq)


def manhattan_distance_matrix(vectors: np.ndarray) -> np.ndarray:
    return np.sum(np.abs(vectors[:, None, :] - vectors[None, :, :]), axis=2)


def compute_pair_grids(
    coef_tensor: np.ndarray,
) -> Dict[str, Dict[int, np.ndarray]]:
    by_metric: Dict[str, Dict[int, np.ndarray]] = {
        "cosine_similarity": {},
        "euclidean_distance": {},
        "manhattan_distance": {},
    }

    for layer in SELECTED_GRID_LAYERS:
        vectors = coef_tensor[layer]
        by_metric["cosine_similarity"][layer] = cosine_similarity_matrix(vectors)
        by_metric["euclidean_distance"][layer] = euclidean_distance_matrix(vectors)
        by_metric["manhattan_distance"][layer] = manhattan_distance_matrix(vectors)

    return by_metric


def plot_pair_grid_pages(
    metric_name: str,
    matrices_by_layer: Dict[int, np.ndarray],
    cmap: str,
) -> None:
    layers = sorted(matrices_by_layer.keys())
    stack = np.stack([matrices_by_layer[layer] for layer in layers], axis=0)
    vmin = float(np.min(stack))
    vmax = float(np.max(stack))

    metric_dir = PAIR_GRID_DIR / metric_name
    metric_dir.mkdir(parents=True, exist_ok=True)

    chunks = [layers[i : i + 2] for i in range(0, len(layers), 2)]
    for page_idx, chunk in enumerate(chunks, start=1):
        fig, axes = plt.subplots(1, len(chunk), figsize=(6.2 * len(chunk), 5.8))
        if len(chunk) == 1:
            axes = [axes]

        images = []
        for ax, layer in zip(axes, chunk):
            mat = matrices_by_layer[layer]
            im = ax.imshow(
                mat,
                cmap=cmap,
                vmin=vmin,
                vmax=vmax,
                interpolation="nearest",
                aspect="equal",
                origin="lower",
            )
            images.append(im)
            ax.set_title(f"Layer {layer}")
            ax.set_xlabel("Protein index (size-sorted)")
            ax.set_ylabel("Protein index (size-sorted)")
            ax.set_xticks([])
            ax.set_yticks([])

        cbar = fig.colorbar(images[-1], ax=axes, fraction=0.035, pad=0.02)
        cbar.set_label(metric_name.replace("_", " ").title())
        fig.suptitle(
            f"Protein x Protein {metric_name.replace('_', ' ').title()}\n"
            f"(fixed color scale across layers {SELECTED_GRID_LAYERS})",
            y=0.98,
        )
        fig.subplots_adjust(left=0.06, right=0.90, top=0.88, bottom=0.08, wspace=0.16)

        out_path = metric_dir / f"{metric_name}_page_{page_idx:02d}.png"
        fig.savefig(out_path, dpi=180)
        plt.close(fig)


def validate_pair_grids(pair_grids: Dict[str, Dict[int, np.ndarray]]) -> None:
    for metric, layer_map in pair_grids.items():
        for layer, matrix in layer_map.items():
            if not np.isfinite(matrix).all():
                raise ValueError(f"{metric} layer {layer}: matrix contains non-finite values")
            if not np.allclose(matrix, matrix.T, atol=1e-8):
                raise ValueError(f"{metric} layer {layer}: matrix is not symmetric")

            diag = np.diag(matrix)
            if metric == "cosine_similarity":
                if not np.allclose(diag, 1.0, atol=1e-5):
                    raise ValueError(f"{metric} layer {layer}: diagonal not ~1")
            else:
                if not np.allclose(diag, 0.0, atol=1e-4):
                    raise ValueError(f"{metric} layer {layer}: diagonal not 0")


def compute_mean_layer_relation_matrices(
    coef_tensor: np.ndarray,
    bucket_indices: Dict[str, List[int]],
):
    n_layers = coef_tensor.shape[0]
    n_proteins = coef_tensor.shape[1]

    sum_cos = np.zeros((n_layers, n_layers), dtype=np.float64)
    sum_euc = np.zeros((n_layers, n_layers), dtype=np.float64)
    sum_man = np.zeros((n_layers, n_layers), dtype=np.float64)

    bucket_cos_sums = {
        bucket: np.zeros((n_layers, n_layers), dtype=np.float64)
        for bucket in BUCKET_ORDER
    }
    bucket_cos_counts = {bucket: 0 for bucket in BUCKET_ORDER}

    idx_to_bucket = {}
    for bucket, indices in bucket_indices.items():
        for idx in indices:
            idx_to_bucket[idx] = bucket

    for protein_idx in range(n_proteins):
        layer_vectors = coef_tensor[:, protein_idx, :]
        cos = cosine_similarity_matrix(layer_vectors)
        euc = euclidean_distance_matrix(layer_vectors)
        man = manhattan_distance_matrix(layer_vectors)

        sum_cos += cos
        sum_euc += euc
        sum_man += man

        bucket = idx_to_bucket[protein_idx]
        bucket_cos_sums[bucket] += cos
        bucket_cos_counts[bucket] += 1

    mean_cos = sum_cos / n_proteins
    mean_euc = sum_euc / n_proteins
    mean_man = sum_man / n_proteins

    bucket_cos_means = {}
    for bucket in BUCKET_ORDER:
        count = bucket_cos_counts[bucket]
        if count == 0:
            bucket_cos_means[bucket] = None
        else:
            bucket_cos_means[bucket] = bucket_cos_sums[bucket] / count

    return mean_cos, mean_euc, mean_man, bucket_cos_means, bucket_cos_counts


def plot_single_heatmap(
    matrix: np.ndarray,
    title: str,
    out_path: Path,
    cmap: str,
) -> None:
    fig, ax = plt.subplots(figsize=(7.2, 6.3))
    im = ax.imshow(
        matrix,
        cmap=cmap,
        interpolation="nearest",
        aspect="equal",
        origin="lower",
    )
    ax.set_title(title)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Layer")
    tick_positions = list(range(0, 48, 4)) + [47]
    tick_positions = sorted(set(tick_positions))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Value")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def plot_cosine_by_bucket(
    bucket_cos_means: Dict[str, np.ndarray],
    bucket_cos_counts: Dict[str, int],
    out_path: Path,
) -> None:
    mats = [m for m in bucket_cos_means.values() if m is not None]
    if not mats:
        raise ValueError("No bucket cosine matrices available")

    vmin = float(min(np.min(m) for m in mats))
    vmax = float(max(np.max(m) for m in mats))

    fig, axes = plt.subplots(2, 2, figsize=(11, 10))
    axes_list = axes.flatten()

    images = []
    for ax, bucket in zip(axes_list, BUCKET_ORDER):
        matrix = bucket_cos_means[bucket]
        if matrix is None:
            ax.axis("off")
            ax.set_title(f"{bucket} (n=0)")
            continue
        im = ax.imshow(
            matrix,
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            interpolation="nearest",
            aspect="equal",
            origin="lower",
        )
        images.append(im)
        ax.set_title(f"{bucket.replace('_', ' ').title()} (n={bucket_cos_counts[bucket]})")
        ax.set_xlabel("Layer")
        ax.set_ylabel("Layer")
        tick_positions = list(range(0, 48, 8)) + [47]
        tick_positions = sorted(set(tick_positions))
        ax.set_xticks(tick_positions)
        ax.set_yticks(tick_positions)

    cbar = fig.colorbar(images[-1], ax=axes_list, fraction=0.03, pad=0.02)
    cbar.set_label("Cosine similarity")
    fig.suptitle("Mean Within-Protein Layer Cosine Similarity by Size Bucket", y=0.98)
    fig.subplots_adjust(left=0.07, right=0.92, top=0.92, bottom=0.07, wspace=0.22, hspace=0.22)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def rankdata_average(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)

    i = 0
    n = values.size
    while i < n:
        j = i
        while j + 1 < n and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg_rank = 0.5 * (i + j) + 1.0
        ranks[order[i : j + 1]] = avg_rank
        i = j + 1

    return ranks


def safe_pearson(x: np.ndarray, y: np.ndarray) -> float:
    if x.size < 2:
        return float("nan")
    if np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def compute_layer47_tm_r2(
    sorted_proteins: List[str],
    metrics: Dict[Tuple[str, int], dict],
    tm: Dict[Tuple[str, int], dict],
):
    xs = np.array([metrics[(pid, 47)]["r2_test"] for pid in sorted_proteins], dtype=np.float64)
    ys = np.array([tm[(pid, 47)]["tm_score"] for pid in sorted_proteins], dtype=np.float64)

    pearson = safe_pearson(xs, ys)
    spearman = safe_pearson(rankdata_average(xs), rankdata_average(ys))

    slope = float("nan")
    intercept = float("nan")
    if xs.size >= 2 and not np.allclose(xs, xs[0]):
        slope, intercept = np.polyfit(xs, ys, 1)

    return xs, ys, pearson, spearman, slope, intercept


def plot_layer47_tm_r2_scatter(
    xs: np.ndarray,
    ys: np.ndarray,
    pearson: float,
    spearman: float,
    slope: float,
    intercept: float,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize=(7.4, 6.0))
    ax.scatter(xs, ys, s=45, alpha=0.8, color="#4c78a8", edgecolors="black", linewidths=0.4)

    if not math.isnan(slope):
        x_line = np.linspace(float(np.min(xs)), float(np.max(xs)), 100)
        y_line = slope * x_line + intercept
        ax.plot(x_line, y_line, color="#e45756", linewidth=2.0, label="Linear fit")

    ax.set_title(
        "Layer 47: TM-score vs Test R2\n"
        f"Pearson={pearson:.4f}  Spearman={spearman:.4f}"
    )
    ax.set_xlabel("Test R2")
    ax.set_ylabel("TM-score")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def write_layer47_corr_csv(
    n_points: int,
    pearson: float,
    spearman: float,
    slope: float,
    intercept: float,
    out_path: Path,
) -> None:
    with out_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow([
            "layer",
            "n_points",
            "pearson",
            "spearman",
            "slope_tm_vs_r2",
            "intercept_tm_vs_r2",
        ])
        writer.writerow([
            47,
            n_points,
            f"{pearson:.6f}",
            f"{spearman:.6f}",
            f"{slope:.6f}",
            f"{intercept:.6f}",
        ])


def ensure_output_dirs() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    PAIR_GRID_DIR.mkdir(parents=True, exist_ok=True)
    MEAN_LAYER_DIR.mkdir(parents=True, exist_ok=True)


def validate_output_contract() -> None:
    expected_files = [
        OUTPUT_DIR / "r2_train_vs_test_gap_summary.txt",
        OUTPUT_DIR / "r2_train_vs_test_gap_by_layer.csv",
        OUTPUT_DIR / "r2_test_across_layers_by_protein_and_size.png",
        OUTPUT_DIR / "tm_score_across_layers_by_protein_and_size.png",
        OUTPUT_DIR / "layer47_tm_vs_r2_scatter.png",
        OUTPUT_DIR / "layer47_tm_vs_r2_correlation.csv",
        MEAN_LAYER_DIR / "all_proteins_cosine_48x48.png",
        MEAN_LAYER_DIR / "all_proteins_euclidean_48x48.png",
        MEAN_LAYER_DIR / "all_proteins_manhattan_48x48.png",
        MEAN_LAYER_DIR / "cosine_by_size_bucket_48x48.png",
    ]

    for metric in ["cosine_similarity", "euclidean_distance", "manhattan_distance"]:
        metric_dir = PAIR_GRID_DIR / metric
        pages = sorted(metric_dir.glob("*.png"))
        if len(pages) != 4:
            raise ValueError(
                f"Expected 4 page images in {metric_dir}, found {len(pages)}"
            )
        expected_files.extend(pages)

    missing = [p for p in expected_files if not p.exists()]
    if missing:
        raise ValueError(f"Missing expected outputs: {[str(p) for p in missing]}")


def main() -> None:
    print("Loading input CSVs...")
    ensure_output_dirs()

    metrics_rows = read_csv_rows(METRICS_CSV)
    tm_rows = read_csv_rows(TM_CSV)
    coef_rows = read_csv_rows(COEF_CSV)

    metrics = parse_metrics(metrics_rows)
    tm = parse_tm(tm_rows)
    coefficients = parse_coefficients(coef_rows)

    (
        sorted_proteins,
        protein_sizes,
        _,
        _,
        bucket_indices,
    ) = validate_and_index_data(metrics, tm, coefficients)

    print(
        f"Validated keys: {len(metrics)} rows "
        f"({len(sorted_proteins)} proteins x {len(EXPECTED_LAYERS)} layers)"
    )
    print(
        "Size buckets: "
        + ", ".join(f"{b}={len(bucket_indices[b])}" for b in BUCKET_ORDER)
    )

    r2_train, r2_test, tm_score, coef_tensor = build_layer_matrices(
        sorted_proteins,
        metrics,
        tm,
        coefficients,
    )

    gaps = r2_train - r2_test
    write_gap_outputs(gaps)

    plot_layer_trajectories(
        r2_test,
        bucket_indices,
        y_label="Test R2",
        title="Per-Protein Test R2 Across Layers",
        out_path=OUTPUT_DIR / "r2_test_across_layers_by_protein_and_size.png",
    )
    plot_layer_trajectories(
        tm_score,
        bucket_indices,
        y_label="TM-score",
        title="Per-Protein TM-score Across Layers",
        out_path=OUTPUT_DIR / "tm_score_across_layers_by_protein_and_size.png",
    )

    pair_grids = compute_pair_grids(coef_tensor)
    validate_pair_grids(pair_grids)

    plot_pair_grid_pages(
        metric_name="cosine_similarity",
        matrices_by_layer=pair_grids["cosine_similarity"],
        cmap="viridis",
    )
    plot_pair_grid_pages(
        metric_name="euclidean_distance",
        matrices_by_layer=pair_grids["euclidean_distance"],
        cmap="magma",
    )
    plot_pair_grid_pages(
        metric_name="manhattan_distance",
        matrices_by_layer=pair_grids["manhattan_distance"],
        cmap="magma",
    )

    mean_cos, mean_euc, mean_man, bucket_cos_means, bucket_cos_counts = (
        compute_mean_layer_relation_matrices(coef_tensor, bucket_indices)
    )

    plot_single_heatmap(
        mean_cos,
        title="Mean Within-Protein Layer Cosine Similarity (All Proteins)",
        out_path=MEAN_LAYER_DIR / "all_proteins_cosine_48x48.png",
        cmap="viridis",
    )
    plot_single_heatmap(
        mean_euc,
        title="Mean Within-Protein Layer Euclidean Distance (All Proteins)",
        out_path=MEAN_LAYER_DIR / "all_proteins_euclidean_48x48.png",
        cmap="magma",
    )
    plot_single_heatmap(
        mean_man,
        title="Mean Within-Protein Layer Manhattan Distance (All Proteins)",
        out_path=MEAN_LAYER_DIR / "all_proteins_manhattan_48x48.png",
        cmap="magma",
    )

    plot_cosine_by_bucket(
        bucket_cos_means,
        bucket_cos_counts,
        out_path=MEAN_LAYER_DIR / "cosine_by_size_bucket_48x48.png",
    )

    xs, ys, pearson, spearman, slope, intercept = compute_layer47_tm_r2(
        sorted_proteins,
        metrics,
        tm,
    )
    plot_layer47_tm_r2_scatter(
        xs,
        ys,
        pearson,
        spearman,
        slope,
        intercept,
        out_path=OUTPUT_DIR / "layer47_tm_vs_r2_scatter.png",
    )
    write_layer47_corr_csv(
        n_points=int(xs.size),
        pearson=pearson,
        spearman=spearman,
        slope=slope,
        intercept=intercept,
        out_path=OUTPUT_DIR / "layer47_tm_vs_r2_correlation.csv",
    )

    validate_output_contract()

    print("\nDone.")
    print(f"Output directory: {OUTPUT_DIR}")
    print(
        "Layer-47 TM vs test R2: "
        f"pearson={pearson:.4f}, spearman={spearman:.4f}, n={xs.size}"
    )
    print(
        "Global train-test gap: "
        f"mean={np.mean(gaps):.6f}, median={np.median(gaps):.6f}, "
        f"mean_abs={np.mean(np.abs(gaps)):.6f}"
    )


if __name__ == "__main__":
    main()

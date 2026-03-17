#!/usr/bin/env python3
"""
Analyze Delta context linear-tree diagnostics and generate fixed-path figures.

Inputs:
  - delta_files/diagnostic_lin_tree_context_reg_csv_files/
    - diagnostic_lin_tree_context_reg_threshold_search.csv
    - diagnostic_lin_tree_context_reg_rules.csv
    - diagnostic_lin_tree_context_reg_variant_summary.csv
    - diagnostic_lin_tree_context_reg_leaf_coefficients_raw.csv
    - diagnostic_lin_tree_context_reg_leaf_weight_diagnostics.csv

Outputs:
  - delta_files/diagnostic_lin_tree_context_reg_csv_files/analysis/
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
INPUT_DIR = PROJECT_ROOT / "delta_files" / "diagnostic_lin_tree_context_reg_csv_files"
OUTPUT_DIR = INPUT_DIR / "analysis"

THRESHOLD_CSV = INPUT_DIR / "diagnostic_lin_tree_context_reg_threshold_search.csv"
RULES_CSV = INPUT_DIR / "diagnostic_lin_tree_context_reg_rules.csv"
SUMMARY_CSV = INPUT_DIR / "diagnostic_lin_tree_context_reg_variant_summary.csv"
LEAF_COEF_CSV = INPUT_DIR / "diagnostic_lin_tree_context_reg_leaf_coefficients_raw.csv"
LEAF_WEIGHT_CSV = INPUT_DIR / "diagnostic_lin_tree_context_reg_leaf_weight_diagnostics.csv"

VARIANT_ORDER = [
    "lower_triangle_center_only_linear_tree",
    "lower_triangle_plus_axial_context_linear_tree",
]

VARIANT_META = {
    "lower_triangle_center_only_linear_tree": {
        "label": "Center-Only Context",
        "slug": "center_only",
        "color": "#9c2f2f",
    },
    "lower_triangle_plus_axial_context_linear_tree": {
        "label": "Axial-Context",
        "slug": "plus_axial_context",
        "color": "#1f5a91",
    },
}

NODE_PATHS = ["", "L", "R"]
NODE_LABELS = {"": "ROOT", "L": "L", "R": "R"}
LEAF_ORDER = ["LL", "LR", "RL", "RR"]
CHILD_COLORS = {"L": "#2a7f62", "R": "#b86922"}


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
    return value


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: format_value(row.get(name, "")) for name in fieldnames})


def cosine_similarity_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    safe_norms = np.where(norms == 0.0, 1.0, norms)
    normalized = vectors / safe_norms
    matrix = normalized @ normalized.T
    return np.clip(matrix, -1.0, 1.0)


def validate_inputs(
    threshold_rows: list[dict[str, str]],
    rule_rows: list[dict[str, str]],
    summary_rows: list[dict[str, str]],
    coef_rows: list[dict[str, str]],
    weight_rows: list[dict[str, str]],
) -> None:
    ensure_required_columns(
        THRESHOLD_CSV,
        threshold_rows,
        {
            "variant",
            "node_path",
            "threshold",
            "test_eval_scope",
            "test_eval_n_proteins",
            "test_eval_n_pairs",
            "test_mean_per_protein_r2",
        },
    )
    ensure_required_columns(
        RULES_CSV,
        rule_rows,
        {"variant", "node_path", "threshold"},
    )
    ensure_required_columns(
        SUMMARY_CSV,
        summary_rows,
        {
            "variant",
            "test_mean_per_protein_r2",
            "selected_root_test_mean_per_protein_r2",
            "selected_root_threshold",
        },
    )
    ensure_required_columns(
        LEAF_COEF_CSV,
        coef_rows,
        {"variant", "node_type", "node_path", "coef_index", "coefficient_raw"},
    )
    ensure_required_columns(
        LEAF_WEIGHT_CSV,
        weight_rows,
        {"variant", "node_type", "node_path", "protein_equivalent_weight_mass"},
    )


def index_summary_rows(summary_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    indexed = {}
    for row in summary_rows:
        variant = row["variant"]
        indexed[variant] = {
            "final_depth2_r2": as_float(row["test_mean_per_protein_r2"]),
            "trained_root_depth1_r2": as_float(row["selected_root_test_mean_per_protein_r2"]),
            "trained_root_threshold": as_float(row["selected_root_threshold"]),
        }
    missing = set(VARIANT_ORDER) - set(indexed)
    if missing:
        raise ValueError(f"Missing variants in summary CSV: {sorted(missing)}")
    return indexed


def index_rule_thresholds(rule_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    per_variant: dict[str, dict[str, float]] = {variant: {} for variant in VARIANT_ORDER}
    for row in rule_rows:
        variant = row["variant"]
        if variant not in per_variant:
            continue
        node_path = row["node_path"]
        if node_path in NODE_PATHS:
            per_variant[variant][node_path] = as_float(row["threshold"])

    for variant in VARIANT_ORDER:
        missing = [NODE_LABELS[path] for path in NODE_PATHS if path not in per_variant[variant]]
        if missing:
            raise ValueError(f"{variant} missing selected thresholds for {missing}")
    return per_variant


def index_threshold_rows(
    threshold_rows: list[dict[str, str]],
) -> dict[str, dict[str, list[dict[str, float | int | str]]]]:
    indexed: dict[str, dict[str, list[dict[str, float | int | str]]]] = {
        variant: {path: [] for path in NODE_PATHS}
        for variant in VARIANT_ORDER
    }
    observed_paths = set()

    for row in threshold_rows:
        variant = row["variant"]
        if variant not in indexed:
            continue
        node_path = row["node_path"]
        observed_paths.add(node_path)
        if node_path not in NODE_PATHS:
            raise ValueError(f"Unexpected node_path {node_path!r} in threshold search")
        indexed[variant][node_path].append(
            {
                "variant": variant,
                "node_path": node_path,
                "threshold": as_float(row["threshold"]),
                "scope": row["test_eval_scope"],
                "n_proteins": as_int(row["test_eval_n_proteins"]),
                "n_pairs": as_int(row["test_eval_n_pairs"]),
                "r2": as_float(row["test_mean_per_protein_r2"]),
            }
        )

    if observed_paths != set(NODE_PATHS):
        raise ValueError(
            f"Threshold search node paths mismatch: expected {set(NODE_PATHS)}, found {observed_paths}"
        )

    for variant in VARIANT_ORDER:
        for node_path in NODE_PATHS:
            rows = indexed[variant][node_path]
            if not rows:
                raise ValueError(f"{variant} has no threshold-search rows for {NODE_LABELS[node_path]}")
            rows.sort(key=lambda row: float(row["threshold"]))

            expected_scope = "full_test" if node_path == "" else "node_routed_subset"
            scopes = {str(row["scope"]) for row in rows}
            if scopes != {expected_scope}:
                raise ValueError(
                    f"{variant} {NODE_LABELS[node_path]} scope mismatch: expected {expected_scope}, found {scopes}"
                )

    return indexed


def row_at_threshold(
    rows: list[dict[str, float | int | str]],
    threshold: float,
    tolerance: float = 1e-6,
) -> dict[str, float | int | str] | None:
    for row in rows:
        if abs(float(row["threshold"]) - threshold) <= tolerance:
            return row
    return None


def best_r2_row(rows: list[dict[str, float | int | str]]) -> dict[str, float | int | str]:
    return max(rows, key=lambda row: float(row["r2"]))


def build_depth_summary_rows(
    summary_by_variant: dict[str, dict[str, float]],
    rule_thresholds: dict[str, dict[str, float]],
    threshold_by_variant: dict[str, dict[str, list[dict[str, float | int | str]]]],
) -> list[dict[str, object]]:
    rows = []
    for variant in VARIANT_ORDER:
        root_rows = threshold_by_variant[variant][""]
        best_root_row = best_r2_row(root_rows)
        root_selected_row = row_at_threshold(root_rows, rule_thresholds[variant][""])
        if root_selected_row is None:
            raise ValueError(f"{variant} missing threshold-search row for trained root threshold")

        row = {
            "variant": variant,
            "variant_label": VARIANT_META[variant]["label"],
            "final_depth2_test_mean_per_protein_r2": summary_by_variant[variant]["final_depth2_r2"],
            "trained_root_depth1_proxy_r2": summary_by_variant[variant]["trained_root_depth1_r2"],
            "best_root_search_r2": float(best_root_row["r2"]),
            "gain_vs_trained_root_depth1": (
                summary_by_variant[variant]["final_depth2_r2"]
                - summary_by_variant[variant]["trained_root_depth1_r2"]
            ),
            "gain_vs_best_root_search": (
                summary_by_variant[variant]["final_depth2_r2"] - float(best_root_row["r2"])
            ),
            "selected_threshold_root": rule_thresholds[variant][""],
            "selected_threshold_left": rule_thresholds[variant]["L"],
            "selected_threshold_right": rule_thresholds[variant]["R"],
            "best_root_search_threshold": float(best_root_row["threshold"]),
            "trained_root_threshold_r2_from_search": float(root_selected_row["r2"]),
        }
        rows.append(row)
    return rows


def plot_threshold_profiles(
    variant: str,
    summary_metrics: dict[str, float],
    selected_thresholds: dict[str, float],
    threshold_rows: dict[str, list[dict[str, float | int | str]]],
) -> None:
    meta = VARIANT_META[variant]
    root_rows = threshold_rows[""]
    child_rows_left = threshold_rows["L"]
    child_rows_right = threshold_rows["R"]
    best_root_row = best_r2_row(root_rows)
    selected_root_row = row_at_threshold(root_rows, selected_thresholds[""])
    selected_left_row = row_at_threshold(child_rows_left, selected_thresholds["L"])
    selected_right_row = row_at_threshold(child_rows_right, selected_thresholds["R"])

    if selected_root_row is None or selected_left_row is None or selected_right_row is None:
        raise ValueError(f"{variant} missing a threshold-search row for one of the selected thresholds")

    fig, axes = plt.subplots(1, 2, figsize=(14.0, 5.8))
    root_ax, child_ax = axes

    root_x = np.asarray([float(row["threshold"]) for row in root_rows], dtype=np.float64)
    root_y = np.asarray([float(row["r2"]) for row in root_rows], dtype=np.float64)
    root_ax.plot(root_x, root_y, color=meta["color"], linewidth=2.2, marker="o", markersize=5)
    root_ax.axhline(
        summary_metrics["final_depth2_r2"],
        color="black",
        linestyle="--",
        linewidth=1.4,
        label=f"Final depth-2 model ({summary_metrics['final_depth2_r2']:.3f})",
    )
    root_ax.scatter(
        [float(selected_root_row["threshold"])],
        [float(selected_root_row["r2"])],
        color="#111111",
        s=70,
        marker="D",
        label=(
            "Trained depth-2 root "
            f"({float(selected_root_row['threshold']):.1f}, {float(selected_root_row['r2']):.3f})"
        ),
        zorder=5,
    )
    if abs(float(best_root_row["threshold"]) - float(selected_root_row["threshold"])) <= 1e-6:
        root_ax.annotate(
            "Trained root = best standalone root",
            xy=(float(selected_root_row["threshold"]), float(selected_root_row["r2"])),
            xytext=(12, 16),
            textcoords="offset points",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#666666", "alpha": 0.9},
        )
    else:
        root_ax.scatter(
            [float(best_root_row["threshold"])],
            [float(best_root_row["r2"])],
            color="#f1c40f",
            edgecolors="black",
            s=80,
            marker="*",
            label=(
                "Best standalone root "
                f"({float(best_root_row['threshold']):.1f}, {float(best_root_row['r2']):.3f})"
            ),
            zorder=6,
        )

    root_ax.set_title("Root Threshold Search\nscope: full_test")
    root_ax.set_xlabel("n_residues threshold")
    root_ax.set_ylabel("Mean per-protein $R^2$")
    root_ax.grid(True, alpha=0.25)
    root_ax.legend(loc="lower right", fontsize=8)

    for node_path, rows in [("L", child_rows_left), ("R", child_rows_right)]:
        xs = np.asarray([float(row["threshold"]) for row in rows], dtype=np.float64)
        ys = np.asarray([float(row["r2"]) for row in rows], dtype=np.float64)
        proteins = int(rows[0]["n_proteins"])
        pairs = int(rows[0]["n_pairs"])
        child_ax.plot(
            xs,
            ys,
            color=CHILD_COLORS[node_path],
            linewidth=2.0,
            marker="o",
            markersize=5,
            label=f"{node_path} subset ({proteins} proteins, {pairs:,} pairs)",
        )

    child_ax.scatter(
        [float(selected_left_row["threshold"])],
        [float(selected_left_row["r2"])],
        color=CHILD_COLORS["L"],
        edgecolors="black",
        s=75,
        marker="D",
        zorder=6,
    )
    child_ax.scatter(
        [float(selected_right_row["threshold"])],
        [float(selected_right_row["r2"])],
        color=CHILD_COLORS["R"],
        edgecolors="black",
        s=75,
        marker="D",
        zorder=6,
    )
    child_ax.annotate(
        f"Selected L: {float(selected_left_row['threshold']):.1f}",
        xy=(float(selected_left_row["threshold"]), float(selected_left_row["r2"])),
        xytext=(10, 12),
        textcoords="offset points",
        fontsize=9,
        color=CHILD_COLORS["L"],
    )
    child_ax.annotate(
        f"Selected R: {float(selected_right_row['threshold']):.1f}",
        xy=(float(selected_right_row["threshold"]), float(selected_right_row["r2"])),
        xytext=(10, -18),
        textcoords="offset points",
        fontsize=9,
        color=CHILD_COLORS["R"],
    )
    child_ax.set_title("Child Threshold Search\nscope: node_routed_subset")
    child_ax.set_xlabel("n_residues threshold")
    child_ax.set_ylabel("Mean per-protein $R^2$")
    child_ax.grid(True, alpha=0.25)
    child_ax.legend(loc="lower right", fontsize=8)
    child_ax.text(
        0.02,
        0.02,
        "Child-node scores are evaluated only on proteins/pairs\nalready routed into that node.",
        transform=child_ax.transAxes,
        fontsize=8.5,
        ha="left",
        va="bottom",
        bbox={"boxstyle": "round,pad=0.3", "fc": "white", "ec": "#bbbbbb", "alpha": 0.9},
    )

    fig.suptitle(f"{meta['label']} Context Linear-Tree Split Profiles", y=0.98, fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(OUTPUT_DIR / f"root_child_threshold_profiles_{meta['slug']}.png", dpi=180)
    plt.close(fig)


def plot_depth_comparison(depth_rows: list[dict[str, object]]) -> None:
    fig, ax = plt.subplots(figsize=(10.5, 4.8))
    y_positions = np.arange(len(depth_rows), dtype=float)

    metric_order = [
        ("trained_root_depth1_proxy_r2", "Trained-root depth-1 proxy", "#555555"),
        ("best_root_search_r2", "Best standalone root-only", "#d4a017"),
        ("final_depth2_test_mean_per_protein_r2", "Final depth-2 model", "#1b7f3b"),
    ]

    for idx, row in enumerate(depth_rows):
        values = [float(row[key]) for key, _, _ in metric_order]
        ax.plot(values, [y_positions[idx]] * len(values), color="#c7c7c7", linewidth=2.0, zorder=1)
        for value, (_, label, color) in zip(values, metric_order):
            ax.scatter(value, y_positions[idx], s=85, color=color, edgecolors="black", zorder=3)
        ax.text(
            float(row["final_depth2_test_mean_per_protein_r2"]) + 0.002,
            y_positions[idx] + 0.06,
            f"+{float(row['gain_vs_trained_root_depth1']):.3f} vs trained root",
            fontsize=9,
            color="#1b7f3b",
            ha="left",
            va="bottom",
        )

    ax.set_yticks(y_positions)
    ax.set_yticklabels([str(row["variant_label"]) for row in depth_rows])
    ax.set_xlabel("Mean test per-protein $R^2$")
    ax.set_title("Depth-1 vs Depth-2 Comparison")
    ax.grid(True, axis="x", alpha=0.25)

    handles = [
        plt.Line2D([0], [0], marker="o", color="w", markerfacecolor=color, markeredgecolor="black", markersize=8, label=label)
        for _, label, color in metric_order
    ]
    ax.legend(handles=handles, loc="lower right", fontsize=9)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "depth_comparison_plot.png", dpi=180)
    plt.close(fig)


def build_leaf_vectors(coef_rows: list[dict[str, str]]) -> dict[str, dict[str, np.ndarray]]:
    grouped: dict[str, dict[str, dict[int, float]]] = {
        variant: {leaf: {} for leaf in LEAF_ORDER}
        for variant in VARIANT_ORDER
    }

    for row in coef_rows:
        variant = row["variant"]
        if variant not in grouped or row["node_type"] != "leaf":
            continue
        node_path = row["node_path"]
        if node_path not in LEAF_ORDER:
            continue
        grouped[variant][node_path][as_int(row["coef_index"])] = as_float(row["coefficient_raw"])

    vectors: dict[str, dict[str, np.ndarray]] = {variant: {} for variant in VARIANT_ORDER}
    for variant in VARIANT_ORDER:
        expected_indices = None
        for leaf in LEAF_ORDER:
            coef_map = grouped[variant][leaf]
            if not coef_map:
                raise ValueError(f"{variant} missing coefficients for leaf {leaf}")
            indices = sorted(coef_map.keys())
            if expected_indices is None:
                expected_indices = indices
            elif indices != expected_indices:
                raise ValueError(f"{variant} coefficient index mismatch for leaf {leaf}")
            vectors[variant][leaf] = np.asarray([coef_map[idx] for idx in indices], dtype=np.float64)
    return vectors


def validate_leaf_weights(weight_rows: list[dict[str, str]]) -> dict[str, dict[str, float]]:
    masses: dict[str, dict[str, float]] = {variant: {} for variant in VARIANT_ORDER}
    for row in weight_rows:
        variant = row["variant"]
        if variant not in masses or row["node_type"] != "leaf":
            continue
        node_path = row["node_path"]
        if node_path in LEAF_ORDER:
            masses[variant][node_path] = as_float(row["protein_equivalent_weight_mass"])

    for variant in VARIANT_ORDER:
        missing = [leaf for leaf in LEAF_ORDER if leaf not in masses[variant]]
        if missing:
            raise ValueError(f"{variant} missing leaf weight diagnostics for {missing}")
    return masses


def write_cosine_matrix_csv(variant: str, matrix: np.ndarray) -> None:
    rows = []
    for row_idx, leaf in enumerate(LEAF_ORDER):
        row = {"leaf": leaf}
        for col_idx, col_leaf in enumerate(LEAF_ORDER):
            row[col_leaf] = float(matrix[row_idx, col_idx])
        rows.append(row)

    write_csv(
        OUTPUT_DIR / f"leaf_coefficient_cosine_similarity_{VARIANT_META[variant]['slug']}.csv",
        ["leaf", *LEAF_ORDER],
        rows,
    )


def plot_leaf_cosine_heatmap(variant: str, matrix: np.ndarray, vmin: float, vmax: float) -> None:
    fig, ax = plt.subplots(figsize=(5.8, 5.2))
    im = ax.imshow(
        matrix,
        cmap="viridis",
        vmin=vmin,
        vmax=vmax,
        interpolation="nearest",
        aspect="equal",
        origin="upper",
    )
    ax.set_title(f"{VARIANT_META[variant]['label']} Leaf Coefficient Cosine Similarity")
    ax.set_xticks(range(len(LEAF_ORDER)))
    ax.set_xticklabels(LEAF_ORDER)
    ax.set_yticks(range(len(LEAF_ORDER)))
    ax.set_yticklabels(LEAF_ORDER)

    for row_idx in range(matrix.shape[0]):
        for col_idx in range(matrix.shape[1]):
            ax.text(
                col_idx,
                row_idx,
                f"{matrix[row_idx, col_idx]:.3f}",
                ha="center",
                va="center",
                color="white" if matrix[row_idx, col_idx] < 0.82 else "black",
                fontsize=9,
            )

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Cosine similarity")
    fig.tight_layout()
    fig.savefig(
        OUTPUT_DIR / f"leaf_coefficient_cosine_similarity_{VARIANT_META[variant]['slug']}.png",
        dpi=180,
    )
    plt.close(fig)


def build_cosine_outputs(
    leaf_vectors: dict[str, dict[str, np.ndarray]],
) -> tuple[dict[str, np.ndarray], list[dict[str, object]]]:
    matrices = {}
    interpretation_rows = []

    for variant in VARIANT_ORDER:
        stacked = np.stack([leaf_vectors[variant][leaf] for leaf in LEAF_ORDER], axis=0)
        matrix = cosine_similarity_matrix(stacked)
        np.fill_diagonal(matrix, 1.0)
        if not np.allclose(matrix, matrix.T, atol=1e-8):
            raise ValueError(f"{variant} leaf cosine matrix is not symmetric")
        if not np.allclose(np.diag(matrix), 1.0, atol=1e-8):
            raise ValueError(f"{variant} leaf cosine diagonal is not 1")
        matrices[variant] = matrix

        pairs = []
        for i in range(len(LEAF_ORDER)):
            for j in range(i + 1, len(LEAF_ORDER)):
                pairs.append((LEAF_ORDER[i], LEAF_ORDER[j], float(matrix[i, j])))
        most_similar = max(pairs, key=lambda item: item[2])
        least_similar = min(pairs, key=lambda item: item[2])
        interpretation_rows.append(
            {
                "variant": variant,
                "variant_label": VARIANT_META[variant]["label"],
                "mean_offdiag_cosine": float(np.mean([pair[2] for pair in pairs])),
                "min_offdiag_cosine": float(min(pair[2] for pair in pairs)),
                "max_offdiag_cosine": float(max(pair[2] for pair in pairs)),
                "most_similar_pair": f"{most_similar[0]} vs {most_similar[1]}",
                "most_similar_pair_cosine": most_similar[2],
                "least_similar_pair": f"{least_similar[0]} vs {least_similar[1]}",
                "least_similar_pair_cosine": least_similar[2],
            }
        )

    return matrices, interpretation_rows


def write_analysis_summary(
    depth_rows: list[dict[str, object]],
    cosine_rows: list[dict[str, object]],
    leaf_masses: dict[str, dict[str, float]],
    threshold_by_variant: dict[str, dict[str, list[dict[str, float | int | str]]]],
) -> None:
    cosine_by_variant = {str(row["variant"]): row for row in cosine_rows}
    with (OUTPUT_DIR / "analysis_summary.txt").open("w") as handle:
        handle.write("Context Linear-Tree Diagnostic Analysis\n")
        handle.write("=====================================\n\n")
        handle.write("Scope caveat\n")
        handle.write("-----------\n")
        handle.write(
            "Root threshold-search rows report full-test mean per-protein R^2.\n"
            "Child threshold-search rows report routed-subset mean per-protein R^2 only\n"
            "for proteins/pairs that already entered that node, so child curves should be\n"
            "read as local split-quality diagnostics rather than direct global comparisons.\n\n"
        )

        handle.write("Depth payoff\n")
        handle.write("------------\n")
        for row in depth_rows:
            variant = str(row["variant"])
            handle.write(f"{row['variant_label']}\n")
            handle.write(
                f"  final depth-2 mean test R^2:      {float(row['final_depth2_test_mean_per_protein_r2']):.6f}\n"
            )
            handle.write(
                f"  trained-root depth-1 proxy:       {float(row['trained_root_depth1_proxy_r2']):.6f}\n"
            )
            handle.write(
                f"  best standalone root-only:        {float(row['best_root_search_r2']):.6f}\n"
            )
            handle.write(
                f"  gain vs trained-root proxy:       {float(row['gain_vs_trained_root_depth1']):+.6f}\n"
            )
            handle.write(
                f"  gain vs best root-only threshold: {float(row['gain_vs_best_root_search']):+.6f}\n"
            )
            handle.write(
                "  selected thresholds: "
                f"ROOT={float(row['selected_threshold_root']):.1f}, "
                f"L={float(row['selected_threshold_left']):.1f}, "
                f"R={float(row['selected_threshold_right']):.1f}\n"
            )
            root_counts = threshold_by_variant[variant][""][0]
            left_counts = threshold_by_variant[variant]["L"][0]
            right_counts = threshold_by_variant[variant]["R"][0]
            handle.write(
                "  routed test scopes: "
                f"ROOT={int(root_counts['n_proteins'])} proteins/{int(root_counts['n_pairs']):,} pairs, "
                f"L={int(left_counts['n_proteins'])} proteins/{int(left_counts['n_pairs']):,} pairs, "
                f"R={int(right_counts['n_proteins'])} proteins/{int(right_counts['n_pairs']):,} pairs\n\n"
            )

        handle.write("Leaf coefficient cosine summary\n")
        handle.write("-------------------------------\n")
        for variant in VARIANT_ORDER:
            row = cosine_by_variant[variant]
            handle.write(f"{row['variant_label']}\n")
            handle.write(f"  mean off-diagonal cosine: {float(row['mean_offdiag_cosine']):.6f}\n")
            handle.write(
                f"  most similar pair:        {row['most_similar_pair']} ({float(row['most_similar_pair_cosine']):.6f})\n"
            )
            handle.write(
                f"  least similar pair:       {row['least_similar_pair']} ({float(row['least_similar_pair_cosine']):.6f})\n"
            )
            handle.write(
                "  leaf protein-equivalent mass: "
                + ", ".join(f"{leaf}={leaf_masses[variant][leaf]:.1f}" for leaf in LEAF_ORDER)
                + "\n\n"
            )


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    threshold_rows = load_csv_rows(THRESHOLD_CSV)
    rule_rows = load_csv_rows(RULES_CSV)
    summary_rows = load_csv_rows(SUMMARY_CSV)
    coef_rows = load_csv_rows(LEAF_COEF_CSV)
    weight_rows = load_csv_rows(LEAF_WEIGHT_CSV)
    validate_inputs(threshold_rows, rule_rows, summary_rows, coef_rows, weight_rows)

    summary_by_variant = index_summary_rows(summary_rows)
    rule_thresholds = index_rule_thresholds(rule_rows)
    threshold_by_variant = index_threshold_rows(threshold_rows)
    depth_rows = build_depth_summary_rows(summary_by_variant, rule_thresholds, threshold_by_variant)

    for variant in VARIANT_ORDER:
        plot_threshold_profiles(
            variant,
            summary_by_variant[variant],
            rule_thresholds[variant],
            threshold_by_variant[variant],
        )

    write_csv(
        OUTPUT_DIR / "depth_comparison_summary.csv",
        [
            "variant",
            "variant_label",
            "final_depth2_test_mean_per_protein_r2",
            "trained_root_depth1_proxy_r2",
            "best_root_search_r2",
            "gain_vs_trained_root_depth1",
            "gain_vs_best_root_search",
            "selected_threshold_root",
            "selected_threshold_left",
            "selected_threshold_right",
            "best_root_search_threshold",
            "trained_root_threshold_r2_from_search",
        ],
        depth_rows,
    )
    plot_depth_comparison(depth_rows)

    leaf_vectors = build_leaf_vectors(coef_rows)
    leaf_masses = validate_leaf_weights(weight_rows)
    cosine_matrices, cosine_rows = build_cosine_outputs(leaf_vectors)
    vmin = float(min(np.min(matrix) for matrix in cosine_matrices.values()))
    vmax = 1.0

    for variant in VARIANT_ORDER:
        write_cosine_matrix_csv(variant, cosine_matrices[variant])
        plot_leaf_cosine_heatmap(variant, cosine_matrices[variant], vmin=vmin, vmax=vmax)

    write_csv(
        OUTPUT_DIR / "leaf_coefficient_cosine_interpretation.csv",
        [
            "variant",
            "variant_label",
            "mean_offdiag_cosine",
            "min_offdiag_cosine",
            "max_offdiag_cosine",
            "most_similar_pair",
            "most_similar_pair_cosine",
            "least_similar_pair",
            "least_similar_pair_cosine",
        ],
        cosine_rows,
    )
    write_analysis_summary(depth_rows, cosine_rows, leaf_masses, threshold_by_variant)


if __name__ == "__main__":
    main()

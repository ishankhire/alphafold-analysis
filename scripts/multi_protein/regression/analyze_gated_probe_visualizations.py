#!/usr/bin/env python3
"""Visualize held-out trajectories and coefficient similarity for gated probes.

This local post-processing script reads the published consolidated outputs for
the sequence-gated (A1), amino-acid-pair-gated (A2), and joint AA-pair ×
sequence-gated (A4) probes, plus the ungated general base256 probe.  It writes
PNGs and audit CSVs to ``delta_files/consolidated/gated_probe_visualizations``.

Analysis 3 in the gated-probe suite contains true-distance statistics rather
than fitted probes, so the third trajectory family here is A4.
"""

from __future__ import annotations

import math
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


DELTA_DIR = PROJECT_ROOT / "delta_files"
CONSOLIDATED_DIR = DELTA_DIR / "consolidated"
GENERAL_DIR = DELTA_DIR / "general_base256_combined"
OUT_DIR = CONSOLIDATED_DIR / "gated_probe_visualizations"

GATED_LAYERS = (0, 10, 20, 30, 40, 47)
FEATURE_DIM = 256
EPS = 1e-12

ANALYSES = {
    "analysis_1_sequence_gated": {
        "short_name": "A1 sequence-gated",
        "group_column": "train_gate_sequence_bin",
        "group_label": "Sequence-separation gate",
        "expected_groups": 6,
        "color": "#4c78a8",
    },
    "analysis_2_aa_pair_gated": {
        "short_name": "A2 AA-pair-gated",
        "group_column": "train_gate_aa_pair",
        "group_label": "AA-pair gate",
        "expected_groups": 210,
        "color": "#e45756",
    },
    "analysis_4_aa_sequence_gated": {
        "short_name": "A4 AA-pair × sequence-gated",
        "group_column": "joint_gate",
        "group_label": "AA-pair × sequence gate",
        "expected_groups": None,
        "color": "#54a24b",
    },
}
MODEL_ORDER = (
    "General base256",
    "A1 sequence-gated",
    "A2 AA-pair-gated",
    "A4 AA-pair × sequence-gated",
)
MODEL_COLORS = {
    "General base256": "#4c78a8",
    "A1 sequence-gated": "#f58518",
    "A2 AA-pair-gated": "#e45756",
    "A4 AA-pair × sequence-gated": "#54a24b",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 240,
            "savefig.bbox": "tight",
        }
    )


def require_columns(path: Path, frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def finite_rows(frame: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    numeric = frame.copy()
    for column in columns:
        numeric[column] = pd.to_numeric(numeric[column], errors="coerce")
    return numeric[np.isfinite(numeric[columns]).all(axis=1)].copy()


def load_gate_summary(analysis_name: str) -> pd.DataFrame:
    path = CONSOLIDATED_DIR / analysis_name / "gate_summary.csv"
    frame = pd.read_csv(path)
    required = {
        "layer", "split", "evaluation_scope", "fit_status", "train_gate_aa_pair",
        "train_gate_sequence_bin", "n_pairs", "pooled_r2", "pooled_rmse",
        "pooled_mae", "pooled_mean_true_distance", "pooled_target_variance",
    }
    require_columns(path, frame, required)
    frame = frame[
        (frame["split"] == "test")
        & (frame["evaluation_scope"] == "gate_overall")
        & frame["fit_status"].astype(str).str.startswith("fitted")
    ].copy()
    frame = finite_rows(
        frame,
        ["layer", "n_pairs", "pooled_r2", "pooled_rmse", "pooled_mae",
         "pooled_mean_true_distance", "pooled_target_variance"],
    )
    frame["layer"] = frame["layer"].astype(int)
    frame["n_pairs"] = frame["n_pairs"].astype(int)
    frame["joint_gate"] = (
        frame["train_gate_aa_pair"].astype(str)
        + " × "
        + frame["train_gate_sequence_bin"].astype(str)
    )
    layers = tuple(sorted(frame["layer"].unique()))
    if layers != GATED_LAYERS:
        raise ValueError(f"{path} expected layers {GATED_LAYERS}, found {layers}")
    return frame


def plot_gate_trajectories(analysis_name: str, frame: pd.DataFrame) -> None:
    meta = ANALYSES[analysis_name]
    group_column = meta["group_column"]
    groups = sorted(frame[group_column].unique())
    expected = meta["expected_groups"]
    if expected is not None and len(groups) != expected:
        raise ValueError(f"{analysis_name} expected {expected} groups, found {len(groups)}")

    if analysis_name == "analysis_4_aa_sequence_gated":
        plot_joint_gate_trajectories(frame, "pooled_r2", "$R^2$", "higher is better")
        plot_joint_gate_trajectories(frame, "pooled_rmse", "RMSE (Å)", "lower is better")
        return

    for metric, label, direction in (
        ("pooled_r2", "$R^2$", "higher is better"),
        ("pooled_rmse", "RMSE (Å)", "lower is better"),
    ):
        figure, axis = plt.subplots(figsize=(10, 6))
        for group, subset in frame.groupby(group_column, sort=True):
            subset = subset.sort_values("layer")
            axis.plot(
                subset["layer"], subset[metric], color=meta["color"], alpha=0.22,
                linewidth=0.85,
            )
        median = frame.groupby("layer", sort=True)[metric].median()
        axis.plot(median.index, median.values, color="#111111", linewidth=2.6, marker="o",
                  markersize=4, label="gate median")
        axis.set(
            title=f"{meta['short_name']}: held-out {label} for every {meta['group_label'].lower()}",
            xlabel="AlphaFold layer",
            ylabel=label,
            xticks=GATED_LAYERS,
        )
        axis.grid(axis="y", alpha=0.25)
        axis.legend(loc="best")
        figure.text(
            0.5, 0.01,
            f"{len(groups)} superimposed held-out trajectories; {direction}.",
            ha="center", va="bottom", fontsize=9,
        )
        figure.tight_layout(rect=(0, 0.04, 1, 1))
        output = OUT_DIR / f"{analysis_name}_{metric}_trajectories.png"
        figure.savefig(output)
        plt.close(figure)


def plot_joint_gate_trajectories(frame: pd.DataFrame, metric: str, label: str, direction: str) -> None:
    sequence_bins = ("1", "2-4", "5-9", "10-19", "20-49", "50+")
    figure, axes = plt.subplots(2, 3, figsize=(15, 8), sharex=True)
    for axis, sequence_bin in zip(axes.flat, sequence_bins):
        subset = frame[frame["train_gate_sequence_bin"] == sequence_bin]
        for _, group in subset.groupby("train_gate_aa_pair", sort=True):
            group = group.sort_values("layer")
            values = group[metric].clip(lower=-1.0) if metric == "pooled_r2" else group[metric]
            axis.plot(group["layer"], values, color=ANALYSES[
                "analysis_4_aa_sequence_gated"]["color"], alpha=0.12, linewidth=0.65)
        median = subset.groupby("layer", sort=True)[metric].median()
        axis.plot(median.index, median.values, color="#111111", linewidth=2.0, marker="o",
                  markersize=3)
        axis.set_title(f"Sequence separation {sequence_bin} ({subset['train_gate_aa_pair'].nunique()} gates)")
        axis.grid(axis="y", alpha=0.22)
        if metric == "pooled_r2":
            axis.set_ylim(-1.05, 1.05)
    for axis in axes[-1]:
        axis.set_xlabel("AlphaFold layer")
        axis.set_xticks(GATED_LAYERS)
    for axis in axes[:, 0]:
        axis.set_ylabel(label)
    figure.suptitle(f"A4 AA-pair × sequence-gated: held-out {label} trajectories", y=0.98)
    figure.text(
        0.5, 0.015,
        ("Each panel overlays fitted AA-pair gates in one separation bin; R² values below −1 "
         "are clipped for display." if metric == "pooled_r2" else
         f"Each panel overlays fitted AA-pair gates in one separation bin; {direction}."),
        ha="center", va="bottom", fontsize=9,
    )
    figure.tight_layout(rect=(0, 0.04, 1, 0.95))
    figure.savefig(OUT_DIR / f"analysis_4_aa_sequence_gated_{metric}_trajectories.png")
    plt.close(figure)


def load_coefficients(analysis_name: str, group_column: str) -> pd.DataFrame:
    path = CONSOLIDATED_DIR / analysis_name / "coefficients.csv"
    frame = pd.read_csv(path)
    require_columns(path, frame, {"layer", "train_gate_aa_pair", "train_gate_sequence_bin",
                                  "feature_index", "coefficient_raw_scale"})
    frame = finite_rows(frame, ["layer", "feature_index", "coefficient_raw_scale"])
    frame["layer"] = frame["layer"].astype(int)
    frame["feature_index"] = frame["feature_index"].astype(int)
    frame = frame[frame["feature_index"] >= 0].copy()
    frame["joint_gate"] = (
        frame["train_gate_aa_pair"].astype(str)
        + " × "
        + frame["train_gate_sequence_bin"].astype(str)
    )
    if group_column not in frame:
        raise ValueError(f"Unknown coefficient group column {group_column}")
    counts = frame.groupby(["layer", group_column], sort=True)["feature_index"].nunique()
    if not (counts == FEATURE_DIM).all():
        raise ValueError(f"{path} must contain {FEATURE_DIM} non-intercept coefficients per gate")
    return frame


def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= EPS):
        raise ValueError("Cannot compute cosine similarity for a zero coefficient vector")
    normalized = vectors / norms
    return np.clip(normalized @ normalized.T, -1.0, 1.0)


def coefficient_cosine_outputs(analysis_name: str, group_column: str, group_label: str) -> None:
    coefficients = load_coefficients(analysis_name, group_column)
    group_order = sorted(coefficients[group_column].unique())
    matrices = {}
    rows = []
    for layer in GATED_LAYERS:
        subset = coefficients[coefficients["layer"] == layer]
        pivot = subset.pivot(index=group_column, columns="feature_index", values="coefficient_raw_scale")
        pivot = pivot.reindex(index=group_order, columns=range(FEATURE_DIM))
        if pivot.isna().any().any():
            raise ValueError(f"Missing coefficient values for {analysis_name}, layer {layer}")
        matrix = cosine_matrix(pivot.to_numpy(dtype=np.float64))
        matrices[layer] = matrix
        for left_index, left in enumerate(group_order):
            for right_index, right in enumerate(group_order):
                rows.append(
                    {
                        "analysis_name": analysis_name,
                        "layer": layer,
                        "left_group": left,
                        "right_group": right,
                        "coefficient_cosine_similarity": matrix[left_index, right_index],
                    }
                )

    pd.DataFrame(rows).to_csv(
        OUT_DIR / f"{analysis_name}_coefficient_cosine_similarity.csv", index=False
    )
    figure, axes = plt.subplots(2, 3, figsize=(15, 9), constrained_layout=True)
    image = None
    for axis, layer in zip(axes.flat, GATED_LAYERS):
        image = axis.imshow(matrices[layer], cmap="coolwarm", vmin=-1.0, vmax=1.0,
                            interpolation="nearest")
        axis.set_title(f"Layer {layer}")
        if len(group_order) <= 12:
            axis.set_xticks(range(len(group_order)), group_order, rotation=45, ha="right")
            axis.set_yticks(range(len(group_order)), group_order)
        else:
            axis.set_xticks([])
            axis.set_yticks([])
    figure.colorbar(image, ax=axes.ravel().tolist(), shrink=0.8, label="Raw-coefficient cosine similarity")
    figure.suptitle(f"{group_label} coefficient cosine similarity by layer", y=1.02)
    figure.savefig(OUT_DIR / f"{analysis_name}_coefficient_cosine_similarity_by_layer.png")
    plt.close(figure)


def recombine_gated_metrics(analysis_name: str, gate_summary: pd.DataFrame) -> pd.DataFrame:
    """Recombine disjoint gate cells into whole-layer test metrics."""
    per_path = CONSOLIDATED_DIR / analysis_name / "test_per_protein_gate_metrics.csv.gz"
    per = pd.read_csv(per_path)
    require_columns(
        per_path,
        per,
        {"layer", "protein_id", "fit_status", "n_pairs", "target_variance",
         "per_protein_r2", "per_protein_mae", "per_protein_rmse", "mean_true_distance"},
    )
    per = per[per["fit_status"].astype(str).str.startswith("fitted")].copy()
    per = finite_rows(
        per,
        ["layer", "n_pairs", "target_variance", "per_protein_mae",
         "per_protein_rmse", "mean_true_distance"],
    )
    per["layer"] = per["layer"].astype(int)
    per["n_pairs"] = per["n_pairs"].astype(int)

    output_rows = []
    for layer in GATED_LAYERS:
        pooled = gate_summary[gate_summary["layer"] == layer]
        all_test = pd.read_csv(CONSOLIDATED_DIR / analysis_name / "gate_summary.csv")
        all_test = all_test[
            (all_test["layer"] == layer)
            & (all_test["split"] == "test")
            & (all_test["evaluation_scope"] == "gate_overall")
        ]
        total_pairs = int(pd.to_numeric(all_test["n_pairs"], errors="raise").sum())
        n_pairs = int(pooled["n_pairs"].sum())
        sum_y = float((pooled["n_pairs"] * pooled["pooled_mean_true_distance"]).sum())
        sum_y2 = float((pooled["n_pairs"] * (
            pooled["pooled_target_variance"] + pooled["pooled_mean_true_distance"] ** 2
        )).sum())
        sum_abs = float((pooled["n_pairs"] * pooled["pooled_mae"]).sum())
        sum_sq_error = float((pooled["n_pairs"] * pooled["pooled_rmse"] ** 2).sum())
        total_ss = sum_y2 - sum_y ** 2 / n_pairs
        pooled_r2 = float("nan") if total_ss <= EPS else 1.0 - sum_sq_error / total_ss

        protein_rows = []
        for protein_id, protein in per[per["layer"] == layer].groupby("protein_id", sort=True):
            protein_n = int(protein["n_pairs"].sum())
            protein_sum_y = float((protein["n_pairs"] * protein["mean_true_distance"]).sum())
            protein_sum_y2 = float((protein["n_pairs"] * (
                protein["target_variance"] + protein["mean_true_distance"] ** 2
            )).sum())
            protein_sse = float((protein["n_pairs"] * protein["per_protein_rmse"] ** 2).sum())
            protein_abs = float((protein["n_pairs"] * protein["per_protein_mae"]).sum())
            protein_ss = protein_sum_y2 - protein_sum_y ** 2 / protein_n
            protein_rows.append(
                {
                    "protein_id": protein_id,
                    "r2": float("nan") if protein_ss <= EPS else 1.0 - protein_sse / protein_ss,
                    "rmse": math.sqrt(protein_sse / protein_n),
                    "mae": protein_abs / protein_n,
                }
            )
        protein_frame = pd.DataFrame(protein_rows)
        output_rows.append(
            {
                "model": ANALYSES[analysis_name]["short_name"],
                "layer": layer,
                "pooled_r2": pooled_r2,
                "pooled_rmse": math.sqrt(sum_sq_error / n_pairs),
                "pooled_mae": sum_abs / n_pairs,
                "mean_per_protein_r2": protein_frame["r2"].mean(skipna=True),
                "mean_per_protein_rmse": protein_frame["rmse"].mean(),
                "mean_per_protein_mae": protein_frame["mae"].mean(),
                "n_test_pairs": n_pairs,
                "total_test_pairs": total_pairs,
                "test_pair_coverage": n_pairs / total_pairs,
                "n_test_proteins": len(protein_frame),
            }
        )
    return pd.DataFrame(output_rows)


def load_general_metrics() -> pd.DataFrame:
    summary_path = GENERAL_DIR / "split_summary.csv"
    per_path = GENERAL_DIR / "per_protein.csv"
    summary = pd.read_csv(summary_path)
    per = pd.read_csv(per_path)
    require_columns(
        summary_path,
        summary,
        {"layer", "test_pair_weighted_r2", "test_pair_weighted_rmse", "test_pair_weighted_mae",
         "test_mean_per_protein_r2", "n_test_rows", "n_test_proteins"},
    )
    require_columns(per_path, per, {"layer", "split", "per_protein_rmse", "per_protein_mae"})
    per = per[per["split"] == "test"].copy()
    rows = []
    for _, row in summary.sort_values("layer").iterrows():
        layer = int(row["layer"])
        protein = per[per["layer"] == layer]
        rows.append(
            {
                "model": "General base256",
                "layer": layer,
                "pooled_r2": float(row["test_pair_weighted_r2"]),
                "pooled_rmse": float(row["test_pair_weighted_rmse"]),
                "pooled_mae": float(row["test_pair_weighted_mae"]),
                "mean_per_protein_r2": float(row["test_mean_per_protein_r2"]),
                "mean_per_protein_rmse": float(protein["per_protein_rmse"].mean()),
                "mean_per_protein_mae": float(protein["per_protein_mae"].mean()),
                "n_test_pairs": int(row["n_test_rows"]),
                "total_test_pairs": int(row["n_test_rows"]),
                "test_pair_coverage": 1.0,
                "n_test_proteins": int(row["n_test_proteins"]),
            }
        )
    return pd.DataFrame(rows)


def plot_whole_protein_comparison(metrics: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5), sharex=True)
    panels = (
        ("mean_per_protein_r2", "Mean held-out per-protein $R^2$", "higher is better"),
        ("mean_per_protein_rmse", "Mean held-out per-protein RMSE (Å)", "lower is better"),
    )
    for axis, (metric, ylabel, direction) in zip(axes, panels):
        for model in MODEL_ORDER:
            subset = metrics[metrics["model"] == model].sort_values("layer")
            if subset.empty:
                continue
            marker = "o" if model != "General base256" else None
            linewidth = 2.4 if model != "General base256" else 1.6
            alpha = 1.0 if model != "General base256" else 0.75
            axis.plot(subset["layer"], subset[metric], label=model, color=MODEL_COLORS[model],
                      marker=marker, markersize=4, linewidth=linewidth, alpha=alpha)
        axis.set(xlabel="AlphaFold layer", ylabel=ylabel)
        axis.grid(alpha=0.25)
        axis.text(0.02, 0.03, direction, transform=axis.transAxes, fontsize=9,
                  bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.75})
    axes[0].legend(loc="best")
    figure.suptitle("Whole-protein held-out performance: gated probes vs general base256", y=1.01)
    figure.tight_layout()
    figure.savefig(OUT_DIR / "whole_protein_r2_rmse_comparison.png")
    plt.close(figure)


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    configure_style()

    gate_frames = {name: load_gate_summary(name) for name in ANALYSES}
    for analysis_name, frame in gate_frames.items():
        plot_gate_trajectories(analysis_name, frame)

    coefficient_cosine_outputs(
        "analysis_1_sequence_gated", "train_gate_sequence_bin", "A1 sequence-gate"
    )
    coefficient_cosine_outputs(
        "analysis_2_aa_pair_gated", "train_gate_aa_pair", "A2 AA-pair-gate"
    )

    comparison = pd.concat(
        [
            load_general_metrics(),
            *(recombine_gated_metrics(name, frame) for name, frame in gate_frames.items()),
        ],
        ignore_index=True,
    )
    comparison["model"] = pd.Categorical(comparison["model"], categories=MODEL_ORDER, ordered=True)
    comparison = comparison.sort_values(["model", "layer"])
    comparison.to_csv(OUT_DIR / "whole_protein_comparison_metrics.csv", index=False)
    plot_whole_protein_comparison(comparison)

    print(f"Wrote visualizations and audit CSVs to {OUT_DIR}")


if __name__ == "__main__":
    main()

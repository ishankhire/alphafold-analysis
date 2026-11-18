#!/usr/bin/env python3
"""Figures for the probe layer-similarity diagnostic.

Reads the compact tables written by ``probe_layer_similarity.py`` and, when
present, the all-layer transfer run's ``transfer_matrix.csv``. Every plotted
value is a per-protein metric averaged over proteins with equal weight, or a
within-fold weight cosine averaged over the five outer folds.

Inputs: ``$PROBE_LAYER_SIMILARITY_OUTPUT_ROOT`` (tables) and
``$ALL_LAYER_TRANSFER_OUTPUT_ROOT/transfer_matrix.csv`` (optional).
Outputs: ``<figure root>/NN_<name>.{png,pdf}`` where the figure root is
``$PROBE_LAYER_SIMILARITY_FIGURE_ROOT`` or ``<tables>/figures``.
"""
from __future__ import annotations

import csv
import json
import os
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

N = 48
PROBES = {"shared_ols_base256": ("Shared linear", "#2a78d6"),
          "per_protein_ols_base256": ("Per-protein linear", "#eb6834"),
          "sequence_gated_ols_base256": ("Sequence-gated linear", "#1baf7a"),
          "shared_xgboost_base256_fixed_s4": ("Shared XGBoost", "#eda100")}
GATES = {"1": "1", "2_to_4": "2–4", "5_to_9": "5–9", "10_to_19": "10–19",
         "20_to_49": "20–49", "50_plus": "50+"}
GATE_COLORS = ("#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#52514e", "#e1e0d9", "#fcfcfb"
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "blue", ["#f4f8fe", "#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])
DIVERGING = LinearSegmentedColormap.from_list(
    "red_blue", ["#a12a2a", "#e34948", "#f0efec", "#3987e5", "#0d366b"])
TICKS = [0, 8, 16, 24, 32, 40, 47]

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "font.size": 11, "axes.titlesize": 12, "axes.labelsize": 11, "axes.edgecolor": "#c3c2b7",
    "axes.labelcolor": INK, "text.color": INK, "xtick.color": MUTED, "ytick.color": MUTED,
    "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "lines.linewidth": 2.0, "legend.frameon": False,
})


def tables_root() -> Path:
    default = Path.home() / "diagnostic_experiments" / "probe_layer_similarity_v1"
    return Path(os.environ.get("PROBE_LAYER_SIMILARITY_OUTPUT_ROOT", default)).expanduser().resolve()


def read_csv(path: Path):
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def load_matrices(path: Path) -> dict[tuple[str, str, str], np.ndarray]:
    """``{(probe, gate, measure): symmetric [48, 48] mean}``."""
    matrices: dict[tuple[str, str, str], np.ndarray] = {}
    for row in read_csv(path):
        key = (row["probe"], row["gate"], row["measure"])
        matrix = matrices.setdefault(key, np.full((N, N), np.nan))
        a, b = int(row["layer_a"]), int(row["layer_b"])
        matrix[a, b] = matrix[b, a] = float(row["mean"])
    return matrices


def load_effectiveness(path: Path) -> dict[tuple[str, str], np.ndarray]:
    curves: dict[tuple[str, str], np.ndarray] = {}
    for row in read_csv(path):
        curves.setdefault((row["probe"], row["metric"]), np.full(N, np.nan))[int(row["layer"])] = float(row["mean"])
    return curves


def load_transfer(path: Path) -> np.ndarray | None:
    """``[source, target]`` mean per-protein R2 of the shared linear probe."""
    if not path.is_file():
        return None
    matrix = np.full((N, N), np.nan)
    for row in read_csv(path):
        matrix[int(row["source_layer"]), int(row["target_layer"])] = float(row["mean_protein_r2"])
    return matrix


def offset_profile(matrix: np.ndarray) -> np.ndarray:
    """Mean over layer pairs at each absolute offset 0..47."""
    return np.array([np.nanmean(np.diagonal(matrix, k)) for k in range(N)])


def color_scale(matrices) -> tuple:
    """Diverging if any value is negative; otherwise sequential from a rounded-down minimum."""
    low = min(float(np.nanmin(matrix)) for matrix in matrices)
    if low < 0:
        return DIVERGING, -1.0, 1.0
    return SEQUENTIAL, (np.floor(low * 10) / 10 if low >= 0.5 else 0.0), 1.0


def heatmap(ax, matrix, cmap, vmin, vmax, title, xlabel="Layer", ylabel="Layer"):
    image = ax.imshow(matrix, origin="lower", cmap=cmap, vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set(title=title, xlabel=xlabel, ylabel=ylabel, xticks=TICKS, yticks=TICKS)
    ax.grid(False)
    return image


def save(fig, root: Path, name: str) -> None:
    for suffix in ("png", "pdf"):
        fig.savefig(root / f"{name}.{suffix}", dpi=170)
    plt.close(fig)


def probe_legend(fig, keys=PROBES):
    handles = [plt.Line2D([], [], color=PROBES[k][1], lw=2.5, label=PROBES[k][0]) for k in keys]
    fig.legend(handles=handles, loc="outside lower center", ncol=len(handles))


def figure_effectiveness(curves, root, n_proteins):
    panels = (("r2", "R²"), ("mae", "Absolute error (Å)"), ("tm_score", "TM-score"))
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.6), layout="constrained")
    for ax, (metric, label) in zip(axes, panels):
        for probe, (_, color) in PROBES.items():
            ax.plot(range(N), curves[(probe, metric)], color=color)
        ax.set(title=f"Mean per-protein {label}", xlabel="Layer", ylabel=label, xticks=TICKS)
    probe_legend(fig)
    fig.suptitle(f"Probe effectiveness by layer · mean over {n_proteins} proteins", fontsize=13)
    save(fig, root, "01_effectiveness_by_layer")


def figure_weight_shared(matrices, root):
    pair = [matrices[("shared_ols_base256", "", f"weight_cosine_{space}")] for space in ("standardized", "raw_scale")]
    cmap, vmin, vmax = color_scale(pair)
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.6), layout="constrained")
    for ax, matrix, label in zip(axes, pair, ("Standardized coefficients", "Raw-scale coefficients")):
        image = heatmap(ax, matrix, cmap, vmin, vmax, label)
    fig.colorbar(image, ax=axes, label="Cosine similarity, mean over five folds", shrink=0.85)
    fig.suptitle("Shared linear probe: weight similarity between layers", fontsize=13)
    save(fig, root, "02_weight_cosine_shared")


def figure_weight_gated(matrices, root):
    stack = [matrices[("sequence_gated_ols_base256", gate, "weight_cosine_standardized")] for gate in GATES]
    cmap, vmin, vmax = color_scale(stack)
    fig, axes = plt.subplots(2, 3, figsize=(15, 9.6), layout="constrained", sharex=True, sharey=True)
    for ax, matrix, label in zip(axes.flat, stack, GATES.values()):
        image = heatmap(ax, matrix, cmap, vmin, vmax, f"Sequence separation {label}", "", "")
    for ax in axes[-1]:
        ax.set_xlabel("Layer")
    for ax in axes[:, 0]:
        ax.set_ylabel("Layer")
    fig.colorbar(image, ax=axes, label="Cosine similarity, standardized coefficients, mean over five folds", shrink=0.7)
    fig.suptitle("Sequence-gated linear probe: weight similarity between layers, one panel per gate", fontsize=13)
    save(fig, root, "03_weight_cosine_gated")


def figure_pair_similarity(matrices, root, kind, name, n_proteins):
    rows = (("pearson", "Pearson"), ("spearman", "Spearman"))
    stack = [matrices[(probe, "", f"{kind}_{stat}")] for stat, _ in rows for probe in PROBES]
    cmap, vmin, vmax = color_scale(stack)
    fig, axes = plt.subplots(2, 4, figsize=(18, 9.2), layout="constrained", sharex=True, sharey=True)
    for r, (stat, stat_label) in enumerate(rows):
        for c, (probe, (label, _)) in enumerate(PROBES.items()):
            image = heatmap(axes[r, c], matrices[(probe, "", f"{kind}_{stat}")], cmap, vmin, vmax,
                            label if r == 0 else "", "Layer" if r == 1 else "", f"{stat_label}\nLayer" if c == 0 else "")
    fig.colorbar(image, ax=axes, label=f"Correlation, mean over {n_proteins} proteins", shrink=0.7)
    what = ("predicted distances" if kind == "prediction"
            else "errors (probe distance minus final structure distance)")
    fig.suptitle(f"Similarity of {what} between layers", fontsize=13)
    save(fig, root, name)


def figure_transfer(transfer, root):
    fig, ax = plt.subplots(figsize=(7.4, 6.2), layout="constrained")
    image = heatmap(ax, np.clip(transfer, 0, 1), SEQUENTIAL, 0, 1,
                    "Shared linear probe transfer\nprobe from one layer read on another",
                    "Layer the probe is applied to", "Layer the probe was trained on")
    fig.colorbar(image, ax=ax, label="Mean per-protein R² (values below 0 shown as 0)")
    save(fig, root, "06_transfer_mean_protein_r2")


def figure_offset(matrices, root):
    panels = (("prediction_pearson", "Predicted distances, Pearson"),
              ("prediction_spearman", "Predicted distances, Spearman"),
              ("error_pearson", "Errors, Pearson"), ("error_spearman", "Errors, Spearman"))
    fig, axes = plt.subplots(1, 5, figsize=(21, 4.6), layout="constrained", sharey=True)
    for ax, (measure, label) in zip(axes, panels):
        for probe, (_, color) in PROBES.items():
            ax.plot(range(N), offset_profile(matrices[(probe, "", measure)]), color=color)
        ax.set(title=label, xlabel="Layers apart", xticks=TICKS)
    ax = axes[-1]
    ax.plot(range(N), offset_profile(matrices[("shared_ols_base256", "", "weight_cosine_standardized")]),
            color=PROBES["shared_ols_base256"][1])
    for (gate, label), color in zip(GATES.items(), GATE_COLORS):
        ax.plot(range(N), offset_profile(matrices[("sequence_gated_ols_base256", gate, "weight_cosine_standardized")]),
                color=color, lw=1.3, ls="--", label=f"Gate {label}")
    ax.set(title="Weight cosine (standardized)", xlabel="Layers apart", xticks=TICKS)
    ax.legend(title="Sequence-gated", fontsize=8.5, title_fontsize=9)
    axes[0].set_ylabel("Similarity, mean over layer pairs")
    probe_legend(fig)
    fig.suptitle("Similarity against the number of layers apart", fontsize=13)
    save(fig, root, "07_similarity_vs_offset")


def figure_adjacent(matrices, curves, root):
    shared = "shared_ols_base256"
    lines = (("weight_cosine_standardized", "Weight cosine", "#2a78d6"),
             ("prediction_pearson", "Predicted distances, Pearson", "#eb6834"),
             ("error_pearson", "Errors, Pearson", "#1baf7a"))
    x = np.arange(N - 1) + 0.5
    fig, axes = plt.subplots(2, 1, figsize=(11, 7.4), layout="constrained", sharex=True)
    for measure, label, color in lines:
        axes[0].plot(x, np.diagonal(matrices[(shared, "", measure)], 1), color=color, label=label)
    axes[0].set(title="Similarity between each layer and the next", ylabel="Similarity")
    axes[0].legend(loc="lower right")
    gain = np.diff(curves[(shared, "r2")])
    axes[1].bar(x, gain, width=0.8, color=np.where(gain >= 0, "#2a78d6", "#e34948"))
    axes[1].axhline(0, color="#c3c2b7", lw=1)
    axes[1].set(title="Change in mean per-protein R² from each layer to the next (blue up, red down)",
                xlabel="Layer", ylabel="Change in R²", xticks=TICKS)
    fig.suptitle("Shared linear probe: where consecutive layers differ", fontsize=13)
    save(fig, root, "08_adjacent_layer_similarity")


def figure_comparison(matrices, transfer, curves, root):
    shared = "shared_ols_base256"
    upper = np.triu_indices(N, k=1)
    offset = (upper[1] - upper[0]).astype(float)
    cosine = matrices[(shared, "", "weight_cosine_standardized")][upper]
    pearson = matrices[(shared, "", "prediction_pearson")][upper]
    panels = [(cosine, pearson, "Weight cosine", "Predicted distances, Pearson")]
    if transfer is not None:
        # Symmetrized transfer: mean of the two directions for each layer pair.
        symmetric = 0.5 * (transfer + transfer.T)[upper]
        panels += [(cosine, symmetric, "Weight cosine", "Transfer mean per-protein R²"),
                   (pearson, symmetric, "Predicted distances, Pearson", "Transfer mean per-protein R²")]
    fig, axes = plt.subplots(1, len(panels), figsize=(6.2 * len(panels), 5.4), layout="constrained", squeeze=False)
    for ax, (xv, yv, xl, yl) in zip(axes[0], panels):
        points = ax.scatter(xv, yv, c=offset, cmap=SEQUENTIAL, vmin=-8, vmax=47, s=9, linewidths=0)
        ax.set(xlabel=xl, ylabel=yl)
        if "Transfer" in yl:
            ax.set_ylim(max(-1.0, float(np.nanmin(yv)) - 0.05), 1.0)
    fig.colorbar(points, ax=axes, label="Layers apart", shrink=0.85)
    fig.suptitle("Shared linear probe: the similarity measures against each other, one point per layer pair",
                 fontsize=13)
    save(fig, root, "09_measure_comparison")


def main() -> None:
    tables = tables_root()
    manifest = json.loads((tables / "run_manifest.json").read_text())
    if manifest["analysis_status"] != "complete" and not os.environ.get("PROBE_LAYER_SIMILARITY_ALLOW_PARTIAL"):
        raise SystemExit(f"Refusing to draw: analysis_status={manifest['analysis_status']}")
    root = Path(os.environ.get("PROBE_LAYER_SIMILARITY_FIGURE_ROOT", tables / "figures")).expanduser().resolve()
    root.mkdir(parents=True, exist_ok=True)
    matrices = load_matrices(tables / "layer_pair_similarity.csv")
    curves = load_effectiveness(tables / "effectiveness_by_layer.csv")
    default_transfer = Path.home() / "diagnostic_experiments" / "cross_layer_probe_transfer_all_layers_v1"
    transfer = load_transfer(Path(os.environ.get("ALL_LAYER_TRANSFER_OUTPUT_ROOT", default_transfer)).expanduser()
                             / "transfer_matrix.csv")
    n = manifest["n_proteins"]
    figure_effectiveness(curves, root, n)
    figure_weight_shared(matrices, root)
    figure_weight_gated(matrices, root)
    figure_pair_similarity(matrices, root, "prediction", "04_prediction_similarity", n)
    figure_pair_similarity(matrices, root, "error", "05_error_similarity", n)
    if transfer is not None:
        figure_transfer(transfer, root)
    figure_offset(matrices, root)
    figure_adjacent(matrices, curves, root)
    figure_comparison(matrices, transfer, curves, root)
    print(f"figures written to {root}; transfer {'included' if transfer is not None else 'missing'}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Render the all-layer transfer figures from the aggregated outputs.

Signed error is probe distance minus final structure distance, where the final
structure is OpenFold's own prediction. Each figure writes independently, so one
failure does not stop the others; the process exits nonzero if any figure failed.
"""

from __future__ import annotations

import json
import os
import sys
import traceback
from pathlib import Path
from typing import Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap, LogNorm, Normalize, PowerNorm, to_rgb  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

OUTPUT_ENV = "ALL_LAYER_TRANSFER_OUTPUT_ROOT"
OUTLIER = "7stz_C"
V1_BLOCKS = ((22, 22), (22, 43), (43, 22), (43, 43))
V1_BLOCK_SIZE = 5
# (source layer, target layer): two native cells, then the two directions between them.
SELECTED_CELLS = ((22, 22), (47, 47), (22, 47), (47, 22))
RIDGELINE_WINDOW = 40.0
DPI = int(os.environ.get("FIGURE_DPI", "170"))

SURFACE, INK, INK2, MUTED, RULE = "#fcfcfb", "#0b0b0b", "#52514e", "#8b8a85", "#dedcd6"
BLUE, ORANGE, AQUA, YELLOW = "#2a78d6", "#eb6834", "#1baf7a", "#eda100"
DIV = LinearSegmentedColormap.from_list("red_gray_blue", ["#8f1d1c", "#e34948", "#f0efec", "#2a78d6", "#123f75"])
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#f1f6fd", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
SEQUENCE_ORDER = ("1", "2-4", "5-9", "10-19", "20-49", "50+")
DISTANCE_ORDER = ("0-5", "5-10", "10-20", "20-40", "40+")
RESULT_NOTE = ""  # set from the run manifest, so partial cohorts label themselves
plt.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "axes.facecolor": SURFACE,
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.edgecolor": RULE, "axes.linewidth": 0.8,
    "xtick.major.size": 0, "ytick.major.size": 0, "axes.spines.top": False, "axes.spines.right": False,
})


def output_root() -> Path:
    value = os.environ.get(OUTPUT_ENV)
    if not value:
        raise ValueError(f"{OUTPUT_ENV} must name the aggregated run to draw")
    return Path(value).expanduser().resolve()


def header(fig, title: str, note: str, top: float = 0.985) -> None:
    fig.text(0.012, top, title, fontsize=13.5, va="top", weight="semibold")
    fig.text(0.012, top - 0.26 / fig.get_figheight(), note, fontsize=9.5, va="top", color=INK2)


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=DPI)
    fig.savefig(path.with_suffix(".pdf"), dpi=DPI)
    plt.close(fig)
    print("wrote", path.name, flush=True)


def clean(ax) -> None:
    for spine in ax.spines.values():
        spine.set_visible(False)


def ink_for(cmap, norm, value: float) -> str:
    r, g, b = to_rgb(cmap(norm(value)))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "#222222"


def cell_name(source: int, target: int) -> str:
    return f"Native probe, layer {source}" if source == target else f"Layer {source} probe on layer {target}"


def layer_ticks(ax, layers: Sequence[int]) -> None:
    wanted = [layer for layer in (0, 8, 16, 24, 32, 40, 47) if layer in layers] or list(layers)
    positions = [list(layers).index(layer) for layer in wanted]
    ax.set_xticks(positions, wanted), ax.set_yticks(positions, wanted)


def v1_blocks(ax, layers: Sequence[int]) -> None:
    for row, column in V1_BLOCKS:
        if row in layers and column in layers:
            ax.add_patch(Rectangle((list(layers).index(column) - .5, list(layers).index(row) - .5),
                                   V1_BLOCK_SIZE, V1_BLOCK_SIZE, fill=False, ec=INK, lw=0.9))


def matrix(table: pd.DataFrame, value: str, layers: Sequence[int]) -> np.ndarray:
    """Rows are source layers and columns are target layers, as in the v1 heatmaps."""
    return table.pivot(index="source_layer", columns="target_layer", values=value).loc[list(layers), list(layers)].to_numpy()


def fig_r2(data, out: Path) -> None:
    table, layers = data["matrix"], data["layers"]
    fig, ax = plt.subplots(figsize=(7.9, 7.2))
    fig.subplots_adjust(left=.09, right=.86, top=.86, bottom=.08)
    image = ax.imshow(matrix(table, "pooled_r2", layers), cmap=SEQ, norm=Normalize(0, 1), interpolation="nearest")
    v1_blocks(ax, layers), layer_ticks(ax, layers), clean(ax)
    ax.set_xlabel("Target layer (features evaluated)"), ax.set_ylabel("Source layer (probe fitted)")
    bar = fig.colorbar(image, cax=fig.add_axes([.88, .08, .022, .78]), extend="min", ticks=[0, .25, .5, .75, 1])
    bar.set_label("Pooled R²"), bar.outline.set_visible(False)
    header(fig, "01  Transfer matrix, all 48 × 48 cells", RESULT_NOTE + "\nOutlined blocks: the cells v1 covers.")
    save(fig, out / "01_transfer_pooled_r2_48x48.png")


def fig_offset(data, out: Path) -> None:
    table = data["matrix"]
    fig, ax = plt.subplots(figsize=(9.2, 5.0))
    fig.subplots_adjust(left=.08, right=.97, top=.8, bottom=.125)
    grouped = table.groupby("layer_offset")["delta_pooled_vs_native_target_mae"]
    offsets = np.array(sorted(table.layer_offset.unique()))
    ax.fill_between(offsets, grouped.quantile(.25).loc[offsets], grouped.quantile(.75).loc[offsets],
                    color=BLUE, alpha=.18, lw=0)
    ax.plot(offsets, grouped.median().loc[offsets], color=BLUE, lw=2)
    ax.grid(axis="y", color=RULE, lw=.6), ax.set_axisbelow(True), ax.axvline(0, color=MUTED, lw=.8)
    ax.axhline(0, color=MUTED, lw=.8), ax.set_xlim(offsets.min(), offsets.max())
    ax.set_xlabel("Layer offset (target − source). Negative: the probe came from a later layer")
    ax.set_ylabel("Excess MAE over native target probe (Å)")
    clean(ax)
    header(fig, "02  How far does a probe travel?", RESULT_NOTE + "\nLine: median across cells at that offset. Band: IQR.")
    save(fig, out / "02_transfer_offset_decay.png")


def fig_shift_spread(data, out: Path) -> None:
    quantiles, layers = data["quantiles"], data["layers"]
    whole = quantiles[quantiles.category_scope.eq("all_pairs") & quantiles.weighting.eq("pair")]
    median, spread = matrix(whole, "signed_error_q50", layers), matrix(whole, "signed_error_iqr", layers)
    limit = max(1.0, float(np.ceil(np.nanquantile(np.abs(median), .97))))
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.3))
    fig.subplots_adjust(left=.05, right=.925, top=.88, bottom=.09, wspace=.36)
    left = axes[0].imshow(median, cmap=DIV, norm=Normalize(-limit, limit), interpolation="nearest")
    norm = LogNorm(max(float(np.nanmin(spread)), .5), max(float(np.nanmax(spread)), 1.0))
    right = axes[1].imshow(spread, cmap=SEQ, norm=norm, interpolation="nearest")
    for ax, image, label, extend in ((axes[0], left, "Median signed error, probe − final structure (Å)", "both"),
                                     (axes[1], right, "IQR of signed error (Å, log scale)", "neither")):
        layer_ticks(ax, layers), clean(ax), v1_blocks(ax, layers)
        ax.set_xlabel("Target layer"), ax.set_ylabel("Source layer")
        bar = fig.colorbar(image, ax=ax, fraction=.046, pad=.03, extend=extend)
        bar.set_label(label), bar.outline.set_visible(False)
        if ax is axes[1]:
            ticks = [v for v in (1, 2, 3, 5, 10, 15, 20, 30, 50) if norm.vmin <= v <= norm.vmax]
            bar.ax.minorticks_off()
            bar.set_ticks(ticks, labels=[str(v) for v in ticks])
    header(fig, "03  Every probe's error distribution, reduced to shift and spread", RESULT_NOTE)
    save(fig, out / "03_error_median_and_iqr_48x48.png")


def fig_ridgelines(data, out: Path) -> None:
    histograms, layers = data["histograms"], [int(v) for v in data["histograms"]["layers"]]
    edges = histograms["error_edges"]
    clipped = np.clip(edges, -250.0, 250.0)
    widths = np.diff(clipped)
    inside = (edges[:-1] >= -RIDGELINE_WINDOW) & (edges[1:] <= RIDGELINE_WINDOW)
    counts = histograms["pair"].astype(np.float64)
    x = np.column_stack([clipped[:-1], clipped[1:]]).ravel()
    keep = (x >= -RIDGELINE_WINDOW) & (x <= RIDGELINE_WINDOW)
    scale = max(float((counts[t, t] / counts[t, t].sum() / widths).max()) for t in range(len(layers))) / 2.0
    for s, source in enumerate(layers):
        targets = sorted(set(range(0, max(layers) + 1, 4)) | {source, max(layers)} & set(layers))
        targets = [target for target in targets if target in layers]
        fig, ax = plt.subplots(figsize=(7.2, 7.4))
        fig.subplots_adjust(left=.095, right=.975, top=.87, bottom=.08)
        for row, target in enumerate(targets):
            cell = counts[layers.index(target), s]
            density = cell / cell.sum() / widths
            height = np.minimum(np.repeat(density, 2) / scale, 2.2)
            base = len(targets) - 1 - row
            ax.fill_between(x[keep], base, base + height[keep], color=BLUE if target == source else "#c9c8c1",
                            lw=0, zorder=3 + row * 2)
            ax.plot(x[keep], base + height[keep], color=SURFACE, lw=.8, zorder=4 + row * 2)
            outside = 100 * (1 - cell[inside].sum() / cell.sum())
            if outside >= 1:
                ax.text(RIDGELINE_WINDOW, base + .07, f"{outside:.0f}% beyond ±{RIDGELINE_WINDOW:.0f} Å", ha="right",
                        fontsize=7.5, color=INK2, zorder=200)
        ax.axvline(0, color=MUTED, lw=.7, zorder=150)
        ax.set_xlim(-RIDGELINE_WINDOW, RIDGELINE_WINDOW)
        ax.set_yticks(range(len(targets)), targets[::-1])
        ax.set_xlabel("Signed error, probe − final structure (Å)"), ax.set_ylabel("Target layer"), clean(ax)
        header(fig, f"04  Error distribution of the layer {source} probe across target layers",
               RESULT_NOTE + "\nBlue: the probe's own layer. Pair-weighted.")
        save(fig, out / "per_probe" / f"probe_layer_{source:02d}_error_ridgeline.png")


def fig_fan(data, out: Path) -> None:
    quantiles = data["quantiles"]
    native = quantiles[quantiles.is_same_layer & quantiles.category_scope.eq("all_pairs")]
    pair = native[native.weighting.eq("pair")].sort_values("target_layer")
    without = native[native.weighting.eq(f"pair_excl_{OUTLIER}")].sort_values("target_layer")
    layers = pair.target_layer.to_numpy()
    fig, ax = plt.subplots(figsize=(10, 5.1))
    fig.subplots_adjust(left=.075, right=.975, top=.84, bottom=.115)
    ax.fill_between(layers, pair.signed_error_q01, pair.signed_error_q99, color="#cde2fb", lw=0, label="1st to 99th percentile")
    ax.fill_between(layers, pair.signed_error_q05, pair.signed_error_q95, color="#86b6ef", lw=0, label="5th to 95th percentile")
    ax.fill_between(layers, pair.signed_error_q25, pair.signed_error_q75, color=BLUE, lw=0, label="Interquartile range")
    ax.plot(layers, pair.signed_error_q50, color=INK, lw=2, label="Median")
    ax.plot(layers, without.signed_error_q01, color=INK2, lw=1.3, ls=(0, (4, 3)), label=f"1st and 99th without {OUTLIER}")
    ax.plot(layers, without.signed_error_q99, color=INK2, lw=1.3, ls=(0, (4, 3)))
    ax.axhline(0, color=MUTED, lw=.8), ax.set_xlim(layers.min(), layers.max())
    ax.set_ylim(float(pair.signed_error_q01.min()) - 1, float(pair.signed_error_q99.max()) + 12)
    ax.set_xlabel("Layer"), ax.set_ylabel("Signed error, probe − final structure (Å)")
    ax.grid(axis="y", color=RULE, lw=.6), ax.set_axisbelow(True), clean(ax)
    ax.legend(frameon=False, ncol=3, loc="upper center", fontsize=9.3)
    header(fig, "05  Native error distribution by layer", RESULT_NOTE + " Pair-weighted.")
    save(fig, out / "05_native_error_fan_by_layer.png")


def fig_error_vs_distance(data, out: Path) -> None:
    root, layers = data["root"], data["layers"]
    edges = data["histograms"]["error_edges"]
    window = RIDGELINE_WINDOW
    keep = (edges[:-1] >= -window) & (edges[1:] <= window)
    y_edges = edges[np.concatenate([keep, [False]]) | np.concatenate([[False], keep])]
    bin_widths = np.diff(edges[1:-1])
    fig, axes = plt.subplots(1, len(SELECTED_CELLS), figsize=(16.4, 5.2), sharey=True)
    fig.subplots_adjust(left=.05, right=.93, top=.82, bottom=.115, wspace=.07)
    norm = PowerNorm(.5, vmin=0, vmax=.5)
    image = None
    for ax, (source, target) in zip(axes, SELECTED_CELLS):
        with np.load(root / "layer_shards" / f"target_layer_{target:02d}.npz") as shard:
            position = layers.index(source)
            joint = shard["hist_joint"][position].astype(np.int64).sum(axis=0) - shard["outlier_hist_joint"][position]
            distance = shard["distance_edges"].astype(np.float64).copy()
            low, high = float(shard["error_min"][position]), float(shard["error_max"][position])
        distance[-1] = 120.0
        counts = joint.sum(axis=1).astype(np.float64)                # (error, distance)
        with np.errstate(divide="ignore", invalid="ignore"):
            column = np.nan_to_num(counts / counts.sum(axis=0, keepdims=True))
        image = ax.pcolormesh(distance, y_edges, column[keep] / bin_widths[keep[1:-1]][:, None], cmap=SEQ, norm=norm,
                              rasterized=True)
        bounded = edges.astype(np.float64).copy()
        bounded[0], bounded[-1] = min(low, bounded[1]), max(high, bounded[-2])
        cumulative = np.cumsum(counts, axis=0)
        median = np.array([
            np.interp(.5, np.r_[0.0, cumulative[:, d]] / cumulative[-1, d], bounded) if cumulative[-1, d] else np.nan
            for d in range(counts.shape[1])
        ])
        ax.step(distance, np.clip(np.append(median, median[-1]), -window, window), where="post", color=INK, lw=1.8,
                label="Median error")
        if np.any(np.abs(median) > window):
            ax.text(.98, .96, "Median leaves the plot at long distances", transform=ax.transAxes, ha="right", va="top",
                    fontsize=9, color=INK2)
        ax.axhline(0, color=MUTED, lw=.8), ax.set_title(cell_name(source, target), loc="left")
        ax.set_xlabel("Final structure Cα distance (Å); last bin is 100+")
        ax.set_ylim(-window, window), ax.set_xlim(0, 120), clean(ax)
    axes[0].set_ylabel("Signed error, probe − final structure (Å)")
    axes[0].legend(frameon=False, loc="lower left", fontsize=9.3)
    bar = fig.colorbar(image, cax=fig.add_axes([.94, .115, .011, .705]), extend="max")
    bar.set_label("Density within each distance bin (per Å)"), bar.outline.set_visible(False)
    header(fig, "06  Where the error is: signed error against final structure distance", RESULT_NOTE + f" Excludes {OUTLIER}.")
    save(fig, out / "06_error_vs_final_structure_distance.png")


def annotated(ax, values: np.ndarray, n: np.ndarray, cmap, norm, fmt):
    image = ax.imshow(np.ma.masked_where(n == 0, values), cmap=cmap, norm=norm, aspect="auto")
    for r in range(values.shape[0]):
        for c in range(values.shape[1]):
            if n[r, c] == 0:
                ax.text(c, r, "–", ha="center", va="center", color=MUTED)
            else:
                ax.text(c, r, fmt(values[r, c]), ha="center", va="center", fontsize=8.2,
                        color=ink_for(cmap, norm, values[r, c]))
    ax.set_xticks(range(len(DISTANCE_ORDER)), DISTANCE_ORDER), ax.set_yticks(range(len(SEQUENCE_ORDER)), SEQUENCE_ORDER)
    clean(ax)
    return image


def fig_strata(data, out: Path) -> None:
    strata = data["strata"]
    fig, axes = plt.subplots(3, len(SELECTED_CELLS), figsize=(17.6, 10.2), sharex=True, sharey=True)
    fig.subplots_adjust(left=.06, right=.925, top=.9, bottom=.06, wspace=.05, hspace=.16)
    share_norm, bias_norm = PowerNorm(.5, vmin=0, vmax=70), Normalize(-8, 8)
    first, last = None, None
    for column, (source, target) in enumerate(SELECTED_CELLS):
        cell = strata[strata.source_layer.eq(source) & strata.target_layer.eq(target)]
        grouped = cell.groupby(["canonical_sequence_bin", "canonical_distance_bin"]).sum(numeric_only=True)

        def grid(name: str) -> np.ndarray:
            return grouped[name].unstack(fill_value=0).reindex(index=SEQUENCE_ORDER, columns=DISTANCE_ORDER,
                                                               fill_value=0).to_numpy(dtype=np.float64)

        n, n_kept = grid("n_pairs"), grid(f"n_pairs_excl_{OUTLIER}")
        squared, squared_kept = grid("sum_squared_error"), grid(f"sum_squared_error_excl_{OUTLIER}")
        mean = grid(f"sum_signed_error_excl_{OUTLIER}") / np.maximum(n_kept, 1)
        first = annotated(axes[0, column], 100 * squared / squared.sum(), n, SEQ, share_norm, lambda v: f"{v:.1f}")
        annotated(axes[1, column], 100 * squared_kept / squared_kept.sum(), n_kept, SEQ, share_norm, lambda v: f"{v:.1f}")
        last = annotated(axes[2, column], mean, n_kept, DIV, bias_norm, lambda v: f"{v:+.1f}")
        axes[0, column].set_title(f"{cell_name(source, target)}\nShare of squared error (%)", loc="left", fontsize=10.5)
        axes[1, column].set_title(f"Same, excluding {OUTLIER} (%)", loc="left", fontsize=10.5)
        axes[2, column].set_title(f"Mean signed error, excluding {OUTLIER} (Å)", loc="left", fontsize=10.5)
        axes[2, column].set_xlabel("Final structure Cα distance (Å)")
    for ax in axes[:, 0]:
        ax.set_ylabel("Sequence separation")
    top_bar = fig.colorbar(first, cax=fig.add_axes([.935, .39, .011, .51]), extend="max")
    top_bar.set_label("Share of squared error (%)"), top_bar.outline.set_visible(False)
    bottom_bar = fig.colorbar(last, cax=fig.add_axes([.935, .06, .011, .26]), extend="both")
    bottom_bar.set_label("Mean signed error (Å)"), bottom_bar.outline.set_visible(False)
    header(fig, "07  Where the error budget sits: sequence separation × final structure distance", RESULT_NOTE, top=.99)
    save(fig, out / "07_error_budget_strata.png")


def fig_chain(data, out: Path) -> None:
    position = data["position"]
    fig, ax = plt.subplots(figsize=(9.6, 5.0))
    fig.subplots_adjust(left=.085, right=.97, top=.83, bottom=.125)
    placements = {(22, 22): (6, 9), (47, 47): (6, 9), (22, 47): (12, -15), (47, 22): (12, 9)}
    for (source, target), color in zip(SELECTED_CELLS, (AQUA, BLUE, ORANGE, YELLOW)):
        cell = position[position.source_layer.eq(source) & position.target_layer.eq(target)].sort_values("position_bin")
        x = (cell.position_start + cell.position_end) / 2
        y = cell[f"mae_excl_{OUTLIER}"].to_numpy()
        label = f"Layer {source}" if source == target else f"{source} on {target}"
        ax.plot(x, y, color=color, lw=2, marker="o", ms=4.5, mec=SURFACE, mew=.8, label=label)
        where, offset = placements.get((source, target), (len(y) // 2, 9))
        ax.annotate(label, (x.iloc[where], y[where]), xytext=(0, offset), textcoords="offset points", ha="center", fontsize=9.5)
    ax.set_xlim(0, 1), ax.set_ylim(bottom=0)
    ax.set_xlabel("Residue position along the chain (N-terminus = 0, C-terminus = 1)")
    ax.set_ylabel("Mean absolute error of pairs\ninvolving that residue (Å)")
    ax.grid(axis="y", color=RULE, lw=.6), ax.set_axisbelow(True), clean(ax)
    ax.legend(frameon=False, ncol=4, loc="upper right", fontsize=9.5)
    header(fig, "08  Where along the chain the error sits", RESULT_NOTE + f" Excludes {OUTLIER}.")
    save(fig, out / "08_error_along_chain.png")


def fig_concentration(data, out: Path) -> None:
    table, layers = data["matrix"], data["layers"]
    native = data["native_proteins"]
    fig, axes = plt.subplots(1, 2, figsize=(12.6, 5.9), gridspec_kw={"width_ratios": [1.12, 1]})
    fig.subplots_adjust(left=.065, right=.895, top=.83, bottom=.115, wspace=.2)
    ax = axes[0]
    n_proteins = native.protein_id.nunique()
    rank = np.arange(1, n_proteins + 1)
    for layer, color in ((22, ORANGE), (47, BLUE)):
        cell = native[native.target_layer.eq(layer)]
        squared = np.sort((cell.rmse ** 2 * cell.n_evaluated_pairs).to_numpy())[::-1]
        share = 100 * np.cumsum(squared) / squared.sum()
        ax.plot(rank, share, color=color, lw=2, label=f"Native layer {layer}")
        ax.annotate(f"Layer {layer}", (rank[0], share[0]), xytext=(6, 8 if layer == 47 else -15),
                    textcoords="offset points", fontsize=9.5)
    ax.plot(rank, 100 * rank / n_proteins, color=MUTED, lw=1.2, ls=(0, (4, 3)), label="Equal shares")
    ax.set_xscale("log"), ax.set_xlim(1, n_proteins), ax.set_ylim(0, 100), ax.minorticks_off()
    ax.set_xticks([1, 10, 100, n_proteins], ["1", "10", "100", str(n_proteins)])
    ax.set_xlabel("Proteins, ranked by contribution to squared error")
    ax.set_ylabel("Cumulative share of squared error (%)")
    ax.grid(axis="y", color=RULE, lw=.6), ax.set_axisbelow(True), clean(ax)
    ax.legend(frameon=False, loc="lower right", fontsize=9.5)
    ax.set_title("Native cells", loc="left")
    ax = axes[1]
    delta = matrix(table, f"pooled_r2_excl_{OUTLIER}", layers) - matrix(table, "pooled_r2", layers)
    limit = max(.02, float(np.ceil(np.nanquantile(np.abs(delta), .99) * 50) / 50))
    image = ax.imshow(delta, cmap=DIV, norm=Normalize(-limit, limit), interpolation="nearest")
    layer_ticks(ax, layers), clean(ax), v1_blocks(ax, layers)
    ax.set_xlabel("Target layer"), ax.set_ylabel("Source layer")
    ax.set_title(f"Pooled R² gained by dropping {OUTLIER}", loc="left")
    bar = fig.colorbar(image, cax=fig.add_axes([.91, .115, .013, .715]), extend="both")
    bar.set_label("Change in pooled R²"), bar.outline.set_visible(False)
    header(fig, "09  How much one protein shapes the pooled numbers", RESULT_NOTE)
    save(fig, out / "09_protein_concentration.png")


def load_data(root: Path) -> dict[str, object]:
    global RESULT_NOTE
    manifest = json.loads((root / "run_manifest.json").read_text())
    if manifest["analysis_status"] != "complete":
        raise SystemExit(f"Refusing to draw a run with status {manifest['analysis_status']!r}: {root}")
    cohort = "CAMEO proteins only" if manifest["cohort"] == "cameo" else "proteins"
    RESULT_NOTE = (f"Target-restandardized, {manifest['n_proteins']} {cohort}, OOF. "
                   "Error is probe minus final structure.")
    with np.load(root / "error_histograms_by_cell.npz") as handle:
        histograms = {key: handle[key] for key in handle.files}
    native = pd.read_csv(root / "per_protein.csv",
                         usecols=["protein_id", "target_layer", "is_same_layer", "rmse", "n_evaluated_pairs"])
    return {
        "root": root, "layers": [int(v) for v in histograms["layers"]], "histograms": histograms,
        "matrix": pd.read_csv(root / "transfer_matrix.csv"), "quantiles": pd.read_csv(root / "error_quantiles.csv"),
        "strata": pd.read_csv(root / "error_strata.csv"), "position": pd.read_csv(root / "error_position.csv"),
        "native_proteins": native[native.is_same_layer],
    }


def main() -> None:
    root = output_root()
    data = load_data(root)
    out = root / "figures"
    failed = []
    for figure in (fig_r2, fig_offset, fig_shift_spread, fig_ridgelines, fig_fan, fig_error_vs_distance,
                   fig_strata, fig_chain, fig_concentration):
        try:
            figure(data, out)
        except Exception:  # keep drawing the rest; reported through the exit status
            traceback.print_exc()
            failed.append(figure.__name__)
    if failed:
        print("FAILED:", failed, flush=True)
        sys.exit(1)


if __name__ == "__main__":
    main()

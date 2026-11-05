#!/usr/bin/env python3
"""Pooled-R2 heatmaps for the cross-layer shared-OLS probe transfer run."""
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, TwoSlopeNorm, to_rgb

HERE = Path(__file__).resolve().parent
SOURCE = HERE.parents[1] / "diagnostic_experiments/cross_layer_probe_transfer_v1/transfer_matrix.csv"
LAYERS = (22, 23, 24, 25, 26, 43, 44, 45, 46, 47)
MODES = {"raw": "Raw (source scaler)", "target_restandardized": "Target-restandardized"}
CMAP = LinearSegmentedColormap.from_list(
    "red_gray_blue", ["#8f1d1c", "#e34948", "#f0efec", "#2a78d6", "#123f75"])
NORM = TwoSlopeNorm(vmin=-1.6, vcenter=0.0, vmax=1.0)

table = pd.read_csv(SOURCE)
plt.rcParams.update({"font.size": 11, "axes.titlesize": 13})
for mode, title in MODES.items():
    m = (table[table.transfer_mode.eq(mode)]
         .pivot(index="source_layer", columns="target_layer", values="pooled_r2")
         .loc[list(LAYERS), list(LAYERS)].to_numpy())
    fig, ax = plt.subplots(figsize=(7.2, 6.2))
    image = ax.imshow(m, cmap=CMAP, norm=NORM)
    for i in range(len(LAYERS)):
        for j in range(len(LAYERS)):
            r, g, b = to_rgb(CMAP(NORM(m[i, j])))
            ink = "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "#222222"
            ax.text(j, i, f"{m[i, j]:.2f}", ha="center", va="center", fontsize=9,
                    color=ink, fontweight="bold" if i == j else "normal")
    for k in (4.5,):
        ax.axhline(k, color="#222222", linewidth=1.5)
        ax.axvline(k, color="#222222", linewidth=1.5)
    ax.set_xticks(range(len(LAYERS)), LAYERS)
    ax.set_yticks(range(len(LAYERS)), LAYERS)
    ax.tick_params(length=0)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlabel("Target layer (features evaluated)")
    ax.set_ylabel("Source layer (probe fitted)")
    ax.set_title(f"Shared-OLS probe transfer, pooled OOF $R^2$\n{title}")
    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04, ticks=[-1.5, -1, -0.5, 0, 0.5, 1])
    bar.set_label("Pooled $R^2$")
    bar.outline.set_visible(False)
    fig.tight_layout()
    for ext in ("png", "pdf"):
        fig.savefig(HERE / f"transfer_pooled_r2_{mode}.{ext}", dpi=200)
    plt.close(fig)
print("wrote", sorted(p.name for p in HERE.glob("transfer_pooled_r2_*")))

"""
Publication figure: R^2 and TM-score across Evoformer layers.

2 rows x 3 columns:
  Col 1: general_base256_combined (shared/generalizing probe, held-out split)
  Col 2: per_protein_base256_combined (per-protein model, held-out pair split)
  Col 3: general vs per-protein mean trajectories on shared axes
  Row 1: held-out per-protein R^2     Row 2: TM-score

For the single-variant panels (cols 1 & 2) the bold line is the mean across
proteins at each layer; faint background lines are individual protein
trajectories. The general column uses the held-out (test) proteins. The
per-protein column uses held-out residue-pair rows within each protein.

Y-axis is fixed to [0, 1] in every panel; individual trajectories dipping
below 0 (early-layer R^2) are simply clipped by the axis limits.

Run:  source ../../../.venv/bin/activate && python make_r2_tm_layer_grid.py
"""

import os
import numpy as np
import pandas as pd
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ----------------------------------------------------------------------------
# Paths
# ----------------------------------------------------------------------------
HERE = os.path.dirname(os.path.abspath(__file__))
DELTA = os.path.normpath(os.path.join(HERE, "..", ".."))
GENERAL_DIR = os.path.join(DELTA, "general_base256_combined")
PER_PROTEIN_DIR = os.path.join(DELTA, "per_protein_base256_combined")
OUT_PNG = os.path.join(HERE, "r2_tm_layer_grid.png")
OUT_PDF = os.path.join(HERE, "r2_tm_layer_grid.pdf")

# ----------------------------------------------------------------------------
# Styling
# ----------------------------------------------------------------------------
GENERAL_COLOR = "#1f5fa8"      # blue  -> general / shared probe
PER_PROTEIN_COLOR = "#c0392b"  # red   -> per-protein
BG_ALPHA = 0.06
BG_LW = 0.5
MEAN_LW = 2.6

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 11,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 120,
})


# ----------------------------------------------------------------------------
# Data loading -> per-layer mean line + wide matrix of individual trajectories
# ----------------------------------------------------------------------------
def load_trajectories(geometry_csv, metric, split=None):
    """Return (layers, mean_per_layer, traj_matrix).

    traj_matrix has shape (n_proteins, n_layers); rows are individual proteins.
    The mean line is computed directly from the same per-protein rows that are
    drawn in the background, so the bold line is exactly their mean.
    """
    df = pd.read_csv(geometry_csv)
    if metric not in df.columns:
        raise ValueError(f"{geometry_csv} is missing required metric column {metric!r}")
    if split is not None:
        df = df[df["split"] == split]
    pivot = df.pivot_table(index="protein_id", columns="layer", values=metric)
    pivot = pivot.sort_index(axis=1)
    layers = pivot.columns.to_numpy()
    traj = pivot.to_numpy(dtype=float)
    mean_line = np.nanmean(traj, axis=0)
    return layers, mean_line, traj


gen_geom = os.path.join(GENERAL_DIR, "geometry_metrics.csv")
pp_geom = os.path.join(PER_PROTEIN_DIR, "geometry_metrics.csv")

# General -> held-out (test) proteins only
g_layers, g_r2_mean, g_r2_traj = load_trajectories(gen_geom, "per_protein_r2", split="test")
_, g_tm_mean, g_tm_traj = load_trajectories(gen_geom, "tm_score", split="test")

# Per-protein -> all proteins; R^2 is held-out residue-pair score.
p_layers, p_r2_mean, p_r2_traj = load_trajectories(pp_geom, "test_r2")
_, p_tm_mean, p_tm_traj = load_trajectories(pp_geom, "tm_score")

g_n = g_r2_traj.shape[0]
p_n = p_r2_traj.shape[0]


def peak(layers, mean_line):
    i = int(np.nanargmax(mean_line))
    return layers[i], mean_line[i]


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def style_axis(ax):
    ax.set_xlim(0, 47)
    ax.set_ylim(0, 1)
    ax.set_xticks(np.arange(0, 48, 8))
    ax.set_yticks(np.arange(0, 1.01, 0.2))
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_single(ax, layers, mean_line, traj, color, n, label_metric, title):
    # individual trajectories (background); clipped to [0,1] by axis limits
    for row in traj:
        ax.plot(layers, row, color=color, alpha=BG_ALPHA, lw=BG_LW,
                solid_capstyle="round", zorder=1)
    # bold mean
    ax.plot(layers, mean_line, color=color, lw=MEAN_LW, zorder=3,
            label=f"mean over proteins (n={n})")
    # peak marker
    px, py = peak(layers, mean_line)
    ax.plot([px], [py], marker="o", ms=5.5, color=color, mec="white",
            mew=0.8, zorder=4)
    ax.annotate(f"peak {py:.3f}\n@ L{int(px)}",
                xy=(px, py), xytext=(-6, -20), textcoords="offset points",
                ha="right", va="top", fontsize=8, color=color)
    # proxy for background legend entry
    ax.plot([], [], color=color, alpha=0.5, lw=1.2, label="individual proteins")
    ax.set_title(title)
    ax.set_ylabel(label_metric)
    ax.legend(loc="lower right", framealpha=0.9)
    style_axis(ax)


def plot_compare(ax, layers, g_mean, p_mean, label_metric, title):
    ax.plot(layers, p_mean, color=PER_PROTEIN_COLOR, lw=MEAN_LW, zorder=3,
            label=f"per-protein (n={p_n})")
    ax.plot(layers, g_mean, color=GENERAL_COLOR, lw=MEAN_LW, zorder=3,
            label=f"general / held-out (n={g_n})")
    for mean_line, color in ((p_mean, PER_PROTEIN_COLOR), (g_mean, GENERAL_COLOR)):
        px, py = peak(layers, mean_line)
        ax.plot([px], [py], marker="o", ms=5.5, color=color, mec="white",
                mew=0.8, zorder=4)
    ax.set_title(title)
    ax.set_ylabel(label_metric)
    ax.legend(loc="lower right", framealpha=0.9)
    style_axis(ax)


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(16, 9), constrained_layout=True)

# Row 1: R^2
plot_single(axes[0, 0], g_layers, g_r2_mean, g_r2_traj, GENERAL_COLOR, g_n,
            r"held-out $R^2$", r"General base256 — $R^2$ (held-out proteins)")
plot_single(axes[0, 1], p_layers, p_r2_mean, p_r2_traj, PER_PROTEIN_COLOR, p_n,
            r"held-out $R^2$", r"Per-protein base256 — $R^2$ (held-out pairs)")
plot_compare(axes[0, 2], g_layers, g_r2_mean, p_r2_mean,
             r"held-out $R^2$", r"$R^2$ — general vs per-protein")

# Row 2: TM-score
plot_single(axes[1, 0], g_layers, g_tm_mean, g_tm_traj, GENERAL_COLOR, g_n,
            "TM-score", "General base256 — TM-score (held-out)")
plot_single(axes[1, 1], p_layers, p_tm_mean, p_tm_traj, PER_PROTEIN_COLOR, p_n,
            "TM-score", "Per-protein base256 — TM-score")
plot_compare(axes[1, 2], g_layers, g_tm_mean, p_tm_mean,
             "TM-score", "TM-score — general vs per-protein")

for ax in axes[1, :]:
    ax.set_xlabel("Evoformer layer")

fig.suptitle(
    "Linear model for predicting structures from pair representation across layers using MDS",
    fontsize=14, fontweight="bold",
)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")
print(f"general held-out proteins: {g_n}")
print(f"per-protein proteins:      {p_n}")
print(f"general  R^2 peak: L{int(peak(g_layers, g_r2_mean)[0])} = {peak(g_layers, g_r2_mean)[1]:.4f}")
print(f"general  TM  peak: L{int(peak(g_layers, g_tm_mean)[0])} = {peak(g_layers, g_tm_mean)[1]:.4f}")
print(f"perprot  test R^2 peak: L{int(peak(p_layers, p_r2_mean)[0])} = {peak(p_layers, p_r2_mean)[1]:.4f}")
print(f"perprot  TM  peak: L{int(peak(p_layers, p_tm_mean)[0])} = {peak(p_layers, p_tm_mean)[1]:.4f}")
print(f"saved: {OUT_PNG}")
print(f"saved: {OUT_PDF}")

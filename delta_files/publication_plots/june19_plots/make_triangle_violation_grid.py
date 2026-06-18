"""
Publication figure: exact triangle-inequality violations across Evoformer layers.

2 rows x 4 columns, split into two 2x2 blocks:
  Left  2x2: general_base256_combined  (held-out / test split)
  Right 2x2: per_protein_base256_combined  (all proteins)

Within each 2x2 block the four panels are the exact unordered-triangle
violation fractions of the MDS-reconstructed distance geometry:
  top-left  : total violation fraction
  top-right : violations of magnitude 0–5 Å
  bot-left  : violations of magnitude 5–10 Å
  bot-right : violations of magnitude 10+ Å
(total = sum of the three severity bins).

Each metric uses its own y-scale (the bins are orders of magnitude smaller than
the total), but the general and per-protein panels for the SAME metric share a
y-limit so the two variants are directly comparable. Bottom is anchored at 0;
the bold line is the mean across proteins, faint lines are individual proteins.

Run:  source ../../../.venv/bin/activate && python make_triangle_violation_grid.py
"""

import math
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
GENERAL_CSV = os.path.join(
    DELTA, "general_base256_combined",
    "exact_triangle_metrics", "exact_triangle_metrics.csv")
PER_PROTEIN_CSV = os.path.join(
    DELTA, "per_protein_base256_combined",
    "exact_triangle_metrics", "exact_triangle_metrics.csv")
OUT_PNG = os.path.join(HERE, "triangle_violation_grid.png")
OUT_PDF = os.path.join(HERE, "triangle_violation_grid.pdf")

# ----------------------------------------------------------------------------
# Styling (kept consistent with make_r2_tm_layer_grid.py)
# ----------------------------------------------------------------------------
GENERAL_COLOR = "#1f5fa8"      # blue  -> general / held-out
PER_PROTEIN_COLOR = "#c0392b"  # red   -> per-protein
# Fewer general proteins (111) than per-protein (552), so give the general
# cloud a higher per-line alpha to keep the two blocks visually balanced.
GENERAL_BG_ALPHA = 0.11
PER_PROTEIN_BG_ALPHA = 0.05
BG_LW = 0.5
MEAN_LW = 2.6

plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 11.5,
    "axes.labelsize": 11,
    "axes.linewidth": 0.9,
    "xtick.labelsize": 9.5,
    "ytick.labelsize": 9.5,
    "legend.fontsize": 8.5,
    "figure.dpi": 120,
})

# Metric -> (column name, panel title, position in the 2x2 block)
METRICS = [
    ("triangle_violation_fraction",        "Total violation fraction", (0, 0)),
    ("triangle_violation_0_5_fraction",    "Violations 0–5 Å", (0, 1)),
    ("triangle_violation_5_10_fraction",   "Violations 5–10 Å", (1, 0)),
    ("triangle_violation_10_plus_fraction", "Violations 10+ Å", (1, 1)),
]


# ----------------------------------------------------------------------------
# Data loading -> per-layer mean line + matrix of individual trajectories
# ----------------------------------------------------------------------------
def load_trajectories(df, metric):
    """Return (layers, mean_per_layer, traj_matrix) from a pre-filtered frame."""
    pivot = df.pivot_table(index="protein_id", columns="layer", values=metric)
    pivot = pivot.sort_index(axis=1)
    layers = pivot.columns.to_numpy()
    traj = pivot.to_numpy(dtype=float)
    mean_line = np.nanmean(traj, axis=0)
    return layers, mean_line, traj


gen_df = pd.read_csv(GENERAL_CSV)
gen_df = gen_df[gen_df["split"] == "test"]           # held-out, matches figure 1
pp_df = pd.read_csv(PER_PROTEIN_CSV)                  # all proteins

g_n = gen_df["protein_id"].nunique()
p_n = pp_df["protein_id"].nunique()


def nice_ceil(x):
    """Smallest 'nice' number >= x for a clean shared y-limit."""
    if x <= 0:
        return 1.0
    exp = math.floor(math.log10(x))
    base = 10.0 ** exp
    for m in (1, 1.5, 2, 2.5, 3, 4, 5, 6, 8, 10):
        if m * base >= x:
            return m * base
    return 10 * base


# Shared per-metric y-top across both variants (99th pct of individuals + headroom).
shared_top = {}
for col, _title, _pos in METRICS:
    combined = np.concatenate([gen_df[col].to_numpy(), pp_df[col].to_numpy()])
    shared_top[col] = nice_ceil(np.nanpercentile(combined, 99))


# ----------------------------------------------------------------------------
# Plot helpers
# ----------------------------------------------------------------------------
def style_axis(ax, ytop):
    ax.set_xlim(0, 47)
    ax.set_ylim(0, ytop)
    ax.set_xticks(np.arange(0, 48, 8))
    ax.grid(True, alpha=0.25, linewidth=0.6)
    ax.set_axisbelow(True)


def plot_violation(ax, layers, mean_line, traj, color, title, ytop,
                   show_ylabel, show_xlabel, n, bg_alpha):
    for row in traj:  # individual proteins; clipped to [0, ytop] by limits
        ax.plot(layers, row, color=color, alpha=bg_alpha, lw=BG_LW,
                solid_capstyle="round", zorder=1)
    ax.plot(layers, mean_line, color=color, lw=MEAN_LW, zorder=3,
            label=f"mean (n={n})")
    ax.set_title(title)
    if show_ylabel:
        ax.set_ylabel("fraction of triangles")
    if show_xlabel:
        ax.set_xlabel("Evoformer layer")
    ax.legend(loc="upper right", framealpha=0.9)
    style_axis(ax, ytop)


def draw_block(subfig, df, color, header, n, bg_alpha):
    axs = subfig.subplots(2, 2)
    for col, title, (r, c) in METRICS:
        layers, mean_line, traj = load_trajectories(df, col)
        plot_violation(
            axs[r, c], layers, mean_line, traj, color, title,
            shared_top[col],
            show_ylabel=(c == 0),
            show_xlabel=(r == 1),
            n=n,
            bg_alpha=bg_alpha,
        )
    subfig.suptitle(header, fontsize=12.5, fontweight="bold", color=color)
    return axs


# ----------------------------------------------------------------------------
# Figure
# ----------------------------------------------------------------------------
fig = plt.figure(figsize=(18, 9), layout="constrained")
sub_left, sub_right = fig.subfigures(1, 2, wspace=0.04)

draw_block(sub_left, gen_df, GENERAL_COLOR,
           f"General base256 — held-out (n={g_n})", g_n, GENERAL_BG_ALPHA)
draw_block(sub_right, pp_df, PER_PROTEIN_COLOR,
           f"Per-protein base256 (n={p_n})", p_n, PER_PROTEIN_BG_ALPHA)

fig.suptitle(
    "Exact triangle-inequality violations in MDS-reconstructed geometry across layers",
    fontsize=14, fontweight="bold",
)

fig.savefig(OUT_PNG, dpi=300, bbox_inches="tight")
fig.savefig(OUT_PDF, bbox_inches="tight")

print(f"general held-out proteins: {g_n}")
print(f"per-protein proteins:      {p_n}")
for col, title, _pos in METRICS:
    print(f"{title:24s} shared y-top = {shared_top[col]:.3f}")
print(f"saved: {OUT_PNG}")
print(f"saved: {OUT_PDF}")

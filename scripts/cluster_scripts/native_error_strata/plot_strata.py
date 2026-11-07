#!/usr/bin/env python3
"""Plot mean signed error and SD for all 48 native probes from native_error_strata.csv."""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, Normalize, SymLogNorm, to_rgb
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from strata_bins import DIST_LABELS, N_D, N_S, SCOPES, SEQ_LABELS, output_root  # noqa: E402

WORK = output_root()
FIGS = WORK / "figures"
N_LAYERS = 48
SURFACE, INK, INK2, RULE, NO_DATA = "#fcfcfb", "#0b0b0b", "#52514e", "#dedcd6", "#c9c8c1"
DIV = LinearSegmentedColormap.from_list("red_gray_blue", ["#8f1d1c", "#e34948", "#f0efec", "#2a78d6", "#123f75"])
SEQ = LinearSegmentedColormap.from_list("blue_seq", ["#f1f6fd", "#cde2fb", "#86b6ef", "#2a78d6", "#184f95", "#0d366b"])
NOTE = {"excluding_7stz_C": "Real data: native probes, 551 proteins, OOF. Excludes 7stz_C.",
        "all_552": "Real data: native probes, all 552 proteins, OOF. Includes 7stz_C."}
MEAN_TICKS = [-50, -20, -10, -5, -2, 0, 2, 5, 10, 20, 50]
plt.rcParams.update({
    "figure.facecolor": SURFACE, "savefig.facecolor": SURFACE, "font.size": 10, "axes.labelcolor": INK2,
    "xtick.color": INK2, "ytick.color": INK2, "text.color": INK, "xtick.major.size": 0, "ytick.major.size": 0,
})


def load(scope):
    table = pd.read_csv(WORK / "data/native_error_strata.csv",
                        dtype={"sequence_separation": str, "final_structure_distance": str})
    table = table[table.cohort_scope.eq(scope)]
    if len(table) != N_LAYERS * N_S * N_D:
        raise ValueError(f"{scope}: expected {N_LAYERS * N_S * N_D} rows, found {len(table)}")
    s = table.sequence_separation.map({label: k for k, label in enumerate(SEQ_LABELS)}).to_numpy()
    d = table.final_structure_distance.map({label: k for k, label in enumerate(DIST_LABELS)}).to_numpy()
    grid = lambda column, kind=float: np.array(  # noqa: E731
        np.zeros((N_LAYERS, N_S, N_D), kind))
    mean, sd, n, ok = grid("m"), grid("s"), grid("n", np.int64), grid("o", bool)
    where = (table.layer.to_numpy(), s, d)
    mean[where], sd[where] = table.mean_signed_error.to_numpy(), table.sd_signed_error.to_numpy()
    n[where], ok[where] = table.n_pairs.to_numpy(), table.support_status.eq("sufficient").to_numpy()
    return mean, sd, n, ok


def header(fig, title, note, top=.988):
    fig.text(.012, top, title, fontsize=14, va="top", weight="semibold")
    fig.text(.012, top - .27 / fig.get_figheight(), note, fontsize=9.8, va="top", color=INK2)


def ink_for(norm, cmap, value):
    r, g, b = to_rgb(cmap(norm(value)))
    return "white" if 0.2126 * r + 0.7152 * g + 0.0722 * b < 0.5 else "#222222"


def signed(value):
    text = f"{value:+.1f}"
    return "0.0" if text in ("+0.0", "-0.0") else text


def mean_colorbar(bar):
    bar.set_ticks(MEAN_TICKS, labels=[f"{tick:+d}" if tick else "0" for tick in MEAN_TICKS])
    bar.ax.minorticks_off()


def overview(values, ok, cmap, norm, extend, title, note, bar_label, path, log_ticks=False):
    fig, axes = plt.subplots(6, 8, figsize=(15.5, 12.8), sharex=True, sharey=True)
    fig.subplots_adjust(left=.058, right=.93, top=.905, bottom=.09, wspace=.07, hspace=.17)
    for layer, ax in enumerate(axes.flat):
        ax.set_facecolor(NO_DATA)
        image = ax.imshow(np.ma.masked_where(~ok[layer], values[layer]), cmap=cmap, norm=norm, aspect="auto",
                          interpolation="nearest")
        ax.set_title(f"Layer {layer}", loc="left", fontsize=9.5, pad=3)
        for spine in ax.spines.values():
            spine.set_visible(False)
    x_ticks, y_ticks = [0, 4, 8, 12, 16], [0, 3, 6, 9, 12]
    axes[0, 0].set_xticks(x_ticks, [DIST_LABELS[k] for k in x_ticks])
    axes[0, 0].set_yticks(y_ticks, [SEQ_LABELS[k] for k in y_ticks])
    for ax in axes.flat:
        ax.tick_params(labelsize=7.8)
    for ax in axes[-1]:
        plt.setp(ax.get_xticklabels(), rotation=50, ha="right", rotation_mode="anchor")
    fig.supxlabel("Final structure Cα distance (Å)", fontsize=11, color=INK2, y=.012)
    fig.supylabel("Sequence separation (residues)", fontsize=11, color=INK2, x=.012)
    bar = fig.colorbar(image, cax=fig.add_axes([.943, .09, .012, .815]), extend=extend)
    bar.set_label(bar_label), bar.outline.set_visible(False)
    if log_ticks:
        mean_colorbar(bar)
    header(fig, title, note)
    fig.savefig(path, dpi=150)
    fig.savefig(path.with_suffix(".pdf"), dpi=150)
    plt.close(fig)


def layer_page(layer, mean, sd, n, ok, norm, scope, paths):
    fig, ax = plt.subplots(figsize=(14.2, 8.5))
    fig.subplots_adjust(left=.08, right=.925, top=.885, bottom=.075)
    ax.set_facecolor(NO_DATA)
    image = ax.imshow(np.ma.masked_where(~ok, mean), cmap=DIV, norm=norm, aspect="auto", interpolation="nearest")
    for s in range(N_S):
        for d in range(N_D):
            if n[s, d] == 0:
                ax.text(d, s, "–", ha="center", va="center", color=INK2, fontsize=8)
                continue
            color = ink_for(norm, DIV, mean[s, d]) if ok[s, d] else INK2
            ax.text(d, s, f"{signed(mean[s, d])}\n±{sd[s, d]:.1f}", ha="center", va="center", color=color,
                    fontsize=7.6, linespacing=1.05)
    ax.set_xticks(range(N_D), DIST_LABELS, fontsize=9), ax.set_yticks(range(N_S), SEQ_LABELS, fontsize=9)
    ax.set_xlabel("Final structure Cα distance (Å)"), ax.set_ylabel("Sequence separation (residues)")
    for spine in ax.spines.values():
        spine.set_visible(False)
    bar = fig.colorbar(image, cax=fig.add_axes([.937, .075, .014, .81]), extend="both")
    bar.set_label("Mean signed error, probe − final structure (Å)"), bar.outline.set_visible(False)
    mean_colorbar(bar)
    header(fig, f"Layer {layer}: mean signed error ± SD by sequence separation and final structure distance",
           NOTE[scope] + " SD is across pairs. Grey: under 100 pairs or 5 proteins. Colour shared, logarithmic beyond ±2 Å.")
    for path, dpi in paths:
        fig.savefig(path, dpi=dpi)
    plt.close(fig)


def main():
    limits = None
    for scope in SCOPES:
        mean, sd, n, ok = load(scope)
        if limits is None:  # scales come from the primary scope so both scopes stay comparable
            limits = (float(np.ceil(np.nanmax(np.abs(mean[ok])))), float(np.ceil(np.nanquantile(sd[ok], .98))))
            print("colour limits", limits, flush=True)
        mean_norm = SymLogNorm(linthresh=2.0, linscale=.8, vmin=-limits[0], vmax=limits[0], base=10)
        sd_norm = Normalize(0, limits[1])
        out = FIGS / scope
        (out / "by_layer").mkdir(parents=True, exist_ok=True)
        page = FIGS / "page"
        page.mkdir(exist_ok=True)
        overview(mean, ok, DIV, mean_norm, "both", "Mean signed error for all 48 native probes",
                 NOTE[scope] + " Colour shared by all panels, logarithmic beyond ±2 Å. Grey: no pairs, or under 100 pairs or 5 proteins.",
                 "Mean signed error, probe − final structure (Å)", out / "mean_signed_error_all_layers.png", log_ticks=True)
        overview(sd, ok, SEQ, sd_norm, "max", "Standard deviation of signed error for all 48 native probes",
                 NOTE[scope] + " SD is across pairs in each bin. Grey: no pairs, or fewer than 100 pairs or 5 proteins.",
                 "SD of signed error (Å)", out / "sd_signed_error_all_layers.png")
        for layer in range(N_LAYERS):
            paths = [(out / "by_layer" / f"layer_{layer:02d}.png", 150)]
            if scope == SCOPES[0]:
                paths.append((page / f"layer_{layer:02d}.png", 100))
            layer_page(layer, mean[layer], sd[layer], n[layer], ok[layer], mean_norm, scope, paths)
        print("rendered", scope, flush=True)
    # Smaller copies for the web page; the full-resolution files stay as written above.
    sources = [(FIGS / SCOPES[0] / f"{name}_all_layers.png", FIGS / "page" / f"{name}.png")
               for name in ("mean_signed_error", "sd_signed_error")]
    sources += [(path, path) for path in sorted((FIGS / "page").glob("layer_*.png"))]
    for source, target in sources:
        Image.open(source).convert("RGB").quantize(colors=160, dither=Image.Dither.NONE).save(target, optimize=True)
    print("wrote page images", flush=True)


if __name__ == "__main__":
    main()

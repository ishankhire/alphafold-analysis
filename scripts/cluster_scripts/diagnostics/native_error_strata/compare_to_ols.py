#!/usr/bin/env python3
"""Compare a native probe with shared OLS on the same CAMEO proteins, excluding 7stz_C.

Reads the two runs' per-protein strata and writes a by-bin table and one overview figure.
Nothing is fitted or predicted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import SymLogNorm
from PIL import Image

SCRIPT_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(SCRIPT_DIR))
from plot_strata import DIV, FIGS, LABEL, N_LAYERS, WORK, overview  # noqa: E402
from strata_bins import DIST_LABELS, N_D, N_S, SEQ_LABELS, ols_output_root  # noqa: E402

OUTLIER = "7stz_C"
TICKS = [-20, -10, -5, -2, -1, 0, 1, 2, 5, 10, 20]


def load(root):
    """Pair counts, protein counts, mean signed error and MAE by layer and bin."""
    with np.load(root / "data/protein_strata_cameo.npz") as d:
        cell_index, cell_n, offsets, sums, ids = (d[key] for key in ("cell_index", "cell_n", "offsets", "sums", "protein_ids"))
    owner = np.repeat(np.arange(len(ids)), np.diff(offsets))
    keep = ids[owner] != OUTLIER
    cells = cell_index[keep].astype(np.int64)
    n, proteins = np.zeros(N_S * N_D), np.zeros(N_S * N_D)
    np.add.at(n, cells, cell_n[keep].astype(float))
    np.add.at(proteins, cells, 1)
    total = np.zeros((N_LAYERS, N_S * N_D, 3))
    np.add.at(total, (np.arange(N_LAYERS)[:, None], cells[None, :]), sums[keep].transpose(1, 0, 2))
    with np.errstate(divide="ignore", invalid="ignore"):
        mean, mae = total[..., 0] / n, total[..., 1] / n
    shape = (N_LAYERS, N_S, N_D)
    return n.reshape(N_S, N_D), proteins.reshape(N_S, N_D), mean.reshape(shape), mae.reshape(shape), total


def main():
    n, proteins, mean, mae, total = load(WORK)
    n_ols, _, mean_ols, mae_ols, total_ols = load(ols_output_root())
    if not np.array_equal(n, n_ols):
        raise ValueError("The two probes do not cover the same pairs")
    supported = (n >= 100) & (proteins >= 5)
    rows = [{"layer": layer, "sequence_separation": SEQ_LABELS[s], "final_structure_distance": DIST_LABELS[d],
             "n_pairs": int(n[s, d]), "n_proteins": int(proteins[s, d]), "supported": bool(supported[s, d]),
             "mean_signed_error": mean[layer, s, d], "mean_signed_error_ols": mean_ols[layer, s, d],
             "mae": mae[layer, s, d], "mae_ols": mae_ols[layer, s, d]}
            for layer in range(N_LAYERS) for s in range(N_S) for d in range(N_D) if n[s, d] > 0]
    pd.DataFrame(rows).to_csv(WORK / "data/strata_vs_ols_cameo.csv", index=False, float_format="%.6f")

    difference = mae - mae_ols
    shown = np.repeat(supported[None], N_LAYERS, axis=0)
    limit = float(np.ceil(np.nanmax(np.abs(difference[shown]))))
    ticks = [tick for tick in TICKS if abs(tick) <= limit]
    path = FIGS / "mae_difference_vs_ols_all_layers.png"
    overview(difference, shown, DIV.reversed(), SymLogNorm(linthresh=1.0, linscale=.8, vmin=-limit, vmax=limit, base=10), "neither",
             f"{LABEL}MAE minus shared OLS MAE for all 48 native probes",
             "Real data: same 153 CAMEO proteins, OOF. Excludes 7stz_C. Blue: lower error than OLS. Grey: no pairs, or under 100 pairs or 5 proteins.",
             f"{LABEL}MAE − OLS MAE (Å)", path, ticks=ticks)
    (FIGS / "page").mkdir(exist_ok=True)
    Image.open(path).convert("RGB").quantize(colors=160, dither=Image.Dither.NONE).save(
        FIGS / "page/mae_difference_vs_ols.png", optimize=True)

    pairs = n.sum()
    for layer in (0, 22, 47):
        better = int((difference[layer][supported] < 0).sum())
        print(f"layer {layer}: pooled MAE {total[layer, :, 1].sum() / pairs:.3f} vs OLS {total_ols[layer, :, 1].sum() / pairs:.3f}; "
              f"lower error in {better}/{int(supported.sum())} supported bins", flush=True)


if __name__ == "__main__":
    main()

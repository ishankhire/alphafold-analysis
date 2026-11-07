#!/usr/bin/env python3
"""Sum the per-protein CAMEO strata into the table plot_strata.py reads.

The all-protein table needs the ECOD final structures, so this is the cohort view
available without them. Nothing is fitted or predicted.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import build_strata as bs  # noqa: E402
from strata_bins import N_D, N_S  # noqa: E402

N_CELLS = N_S * N_D


def add_up(cell_index, cell_n, sums, keep):
    n, proteins = np.zeros(N_CELLS, np.int64), np.zeros(N_CELLS, np.int64)
    np.add.at(n, cell_index[keep], cell_n[keep])
    np.add.at(proteins, cell_index[keep], 1)
    total = np.zeros((bs.N_LAYERS, N_CELLS, 3))
    np.add.at(total, (np.arange(bs.N_LAYERS)[:, None], cell_index[keep][None, :]), sums[keep].transpose(1, 0, 2))
    return n.reshape(N_S, N_D), proteins.reshape(N_S, N_D), total.reshape(bs.N_LAYERS, N_S, N_D, 3)


def main():
    with np.load(bs.DATA / "protein_strata_cameo.npz") as d:
        cell_index, cell_n = d["cell_index"].astype(np.int64), d["cell_n"].astype(np.int64)
        offsets, sums, ids, pairs = d["offsets"], d["sums"], d["protein_ids"], d["n_pairs"]
    owner = np.repeat(np.arange(len(ids)), np.diff(offsets))
    n, proteins, total = add_up(cell_index, cell_n, sums, np.ones(cell_index.size, bool))
    n_outlier, _, total_outlier = add_up(cell_index, cell_n, sums, ids[owner] == bs.OUTLIER)
    if n.sum() != pairs.sum() or n_outlier.sum() == 0:
        raise ValueError("Per-protein strata do not account for every CAMEO pair and 7stz_C")
    table = bs.tidy_rows({"strata_n": n, "strata_n_outlier": n_outlier, "n_proteins": proteins,
                          "strata_sum": total, "strata_sum_outlier": total_outlier}, all_scope="all_cameo")
    table.to_csv(bs.DATA / "native_error_strata_cameo.csv", index=False, float_format="%.6f")
    print(f"wrote native_error_strata_cameo.csv with {len(table)} rows for {bs.PROBE}", flush=True)


if __name__ == "__main__":
    main()

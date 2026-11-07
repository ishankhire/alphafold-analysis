#!/usr/bin/env python3
"""Test each stated claim per protein and pack the per-protein strata for the web page."""
from __future__ import annotations

import base64
import gzip
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
from canonical_storage import csv_root  # noqa: E402
from strata_bins import DIST_LABELS, N_D, N_S, OLS_PROBE, SEQ_LABELS, ols_output_root, output_root, probe_name  # noqa: E402

DATA = output_root() / "data"
FOLDS = csv_root() / "regression_5fold_v1/metadata/regression_outer_folds_v1.csv"
N_LAYERS, UNIT, MIN_PAIRS = 48, 0.05, 20
ALL_SEQ, ALL_DIST = list(range(N_S)), list(range(N_D))
MID_DIST = [5, 6, 7, 8, 9, 10, 11]  # 8-40 Å
FAR_DIST = [14, 15, 16]             # 60 Å and beyond


def b64(data):
    return base64.b64encode(data).decode("ascii")


class Strata:
    def __init__(self, path):
        with np.load(path) as d:
            self.cell_index, self.cell_n = d["cell_index"].astype(np.int64), d["cell_n"].astype(np.float64)
            self.offsets, self.sums = d["offsets"], d["sums"]
            self.ids, self.source = d["protein_ids"], d["source"]
            self.length, self.fold, self.pairs = d["n_residues"], d["outer_fold"], d["n_pairs"]
        self.owner = np.repeat(np.arange(len(self.ids)), np.diff(self.offsets))

    def select(self, seq, dist, layer):
        """Per-protein pair count, mean, SD and MAE over a set of bins at one layer."""
        chosen = np.zeros(N_S * N_D, bool)
        chosen[[s * N_D + d for s in seq for d in dist]] = True
        keep = chosen[self.cell_index]
        n, e, a, q = (np.zeros(len(self.ids)) for _ in range(4))
        np.add.at(n, self.owner[keep], self.cell_n[keep])
        for target, k in ((e, 0), (a, 1), (q, 2)):
            np.add.at(target, self.owner[keep], self.sums[keep, layer, k])
        with np.errstate(divide="ignore", invalid="ignore"):
            mean, mae = e / n, a / n
            sd = np.sqrt(np.clip(q / n - mean * mean, 0, None))
        return n, mean, sd, mae


def claims(s):
    """Each claim returns (statement, proteins evaluated, proteins where it holds, median value)."""
    rows = []

    def add(text, evaluated, holds, value):
        rows.append((text, evaluated, holds & evaluated, float(np.median(value[evaluated])) if evaluated.any() else None))

    for layer in (0, 22, 47):
        n, mean, _, _ = s.select(ALL_SEQ, [16], layer)
        add(f"Layer {layer}: pairs beyond 100 Å read short (mean below 0)", n >= MIN_PAIRS, mean < 0, mean)
    n, mean, _, _ = s.select(ALL_SEQ, [14, 15], 47)
    add("Layer 47: pairs at 60–100 Å read short (mean below 0)", n >= MIN_PAIRS, mean < 0, mean)
    n, mean, _, _ = s.select(ALL_SEQ, MID_DIST, 47)
    add("Layer 47: pairs at 8–40 Å are near zero (mean within ±1 Å)", n >= MIN_PAIRS, np.abs(mean) <= 1, mean)
    n, mean, _, _ = s.select(ALL_SEQ[1:], [1], 47)
    add("Layer 47: pairs at 4–5 Å read long (mean above 0, separation 2 or more)", n >= MIN_PAIRS, mean > 0, mean)
    n, mean, _, _ = s.select(ALL_SEQ[5:], [3], 47)
    add("Layer 47: pairs at 6–7 Å read short (mean below 0, separation 7 or more)", n >= MIN_PAIRS, mean < 0, mean)
    bins = [12], [6, 7, 8, 9]
    n, early, _, _ = s.select(*bins, 0)
    _, middle, _, _ = s.select(*bins, 22)
    _, late, _, _ = s.select(*bins, 47)
    add("Layer 0: separation 200+ at 10–25 Å reads long (mean above 0)", n >= MIN_PAIRS, early > 0, early)
    add("Separation 200+ at 10–25 Å: the mean falls from layer 0 to 22 to 47", n >= MIN_PAIRS,
        (early > middle) & (middle > late), early - late)
    n_far, _, sd_far, _ = s.select(ALL_SEQ, FAR_DIST, 47)
    n_mid, _, sd_mid, _ = s.select(ALL_SEQ, MID_DIST, 47)
    add("Layer 47: spread beyond 60 Å exceeds spread at 8–40 Å", (n_far >= MIN_PAIRS) & (n_mid >= MIN_PAIRS),
        sd_far > sd_mid, sd_far - sd_mid)
    n, _, _, mae_early = s.select(ALL_SEQ, ALL_DIST, 0)
    _, _, _, mae_late = s.select(ALL_SEQ, ALL_DIST, 47)
    add("Whole protein: MAE is lower at layer 47 than at layer 0", n >= MIN_PAIRS, mae_late < mae_early, mae_late - mae_early)
    return rows


def main():
    s = Strata(DATA / "protein_strata_cameo.npz")
    folds = pd.read_csv(FOLDS, dtype={"protein_id": str}).set_index("protein_id")
    group = folds.length_bin.reindex(s.ids).to_numpy().astype(int)
    # Labels use the whole cohort's length bins, so a later run with ECOD keeps the same groups.
    ranges = folds.groupby("length_bin").n_residues.agg(["min", "max"]).sort_index()
    group_labels = [f"{low}–{high}" for low, high in ranges.to_numpy()]

    table, csv_rows = [], []
    for text, evaluated, holds, median in claims(s):
        by_group = [[int((evaluated & (group == g)).sum()), int((holds & (group == g)).sum())]
                    for g in range(len(group_labels))]
        failing = s.ids[evaluated & ~holds]
        table.append({"claim": text, "evaluated": int(evaluated.sum()), "holds": int(holds.sum()), "median": median,
                      "groups": by_group, "exceptions": s.ids[evaluated & ~holds][:6].tolist()})
        row = {"claim": text, "proteins_evaluated": int(evaluated.sum()), "proteins_holding": int(holds.sum()),
               "median_protein_value": median, "exceptions": " ".join(failing.tolist())}
        for label, (n, h) in zip(group_labels, by_group):
            row[f"evaluated_{label}"], row[f"holding_{label}"] = n, h
        csv_rows.append(row)
    pd.DataFrame(csv_rows).to_csv(DATA / "claims_check_cameo.csv", index=False, float_format="%.4f")
    comparison = {}
    if probe_name() != OLS_PROBE:
        ols = pd.read_csv(ols_output_root() / "data/claims_check_cameo.csv")
        comparison = {r.claim: [int(r.proteins_holding), int(r.proteins_evaluated)] for r in ols.itertuples()}
        for row in table:
            row["comparison"] = comparison.get(row["claim"])

    n = s.cell_n[:, None]
    mean = s.sums[:, :, 0] / n
    sd = np.sqrt(np.clip(s.sums[:, :, 2] / n - mean * mean, 0, None))
    packed = [np.rint(values / UNIT).astype(np.int16) for values in (mean, sd)]
    start = np.zeros((packed[0].shape[0], 1), np.int16)
    deltas = [np.diff(values, axis=1, prepend=start) for values in packed]  # smooth across layers, so it compresses
    raw = np.concatenate([values.ravel() for values in deltas]).astype("<i2").tobytes()
    values = gzip.compress(raw, compresslevel=9, mtime=0)
    payload = {
        "seq_labels": SEQ_LABELS, "dist_labels": DIST_LABELS, "n_layers": N_LAYERS, "unit": UNIT,
        "min_pairs": MIN_PAIRS, "color": {"limit": 71.0, "linthresh": 2.0, "linscale": 0.8},
        "proteins": {"id": s.ids.tolist(), "source": s.source.tolist(), "length": s.length.tolist(),
                     "fold": s.fold.tolist(), "pairs": s.pairs.tolist(), "group": group.tolist()},
        "group_labels": group_labels, "offsets": s.offsets.tolist(), "claims": table,
        "cells": b64(s.cell_index.astype(np.uint8).tobytes()), "cell_n": b64(s.cell_n.astype("<u4").tobytes()),
        "values": b64(values),
    }
    if comparison:
        payload["comparison_label"] = "OLS"
    (DATA / "protein_page_payload_cameo.json").write_text(json.dumps(payload, separators=(",", ":")))
    print(f"{s.cell_index.size} protein-bin rows; raw {len(raw) / 1e6:.1f} MB, gzip {len(values) / 1e6:.1f} MB", flush=True)
    for row in table:
        print(f"{row['holds']}/{row['evaluated']} {row['claim']} (median {row['median']:.2f})", flush=True)


if __name__ == "__main__":
    sys.exit(main())

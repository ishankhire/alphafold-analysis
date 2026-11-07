#!/usr/bin/env python3
"""Collect the rows for the report's charts from the stored result tables."""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, CLUSTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from followup_paths import ABLATION_CSV_DIR, GAIN_ROOT, PRODUCER, TRANSFER_ROOT, TRIANGLE_ROOT  # noqa: E402

ROOT = TRIANGLE_ROOT
R = TRANSFER_ROOT
G = GAIN_ROOT
L = np.arange(48)
out = {}

M = pd.read_csv(R / "transfer_matrix.csv")
med = M.pivot(index="source_layer", columns="target_layer", values="median_protein_r2").reindex(index=L, columns=L).to_numpy(float)
out["heat"] = [{"source": int(s), "v": [round(float(x), 3) for x in med[s]]} for s in L]

geom = pd.read_csv(PRODUCER / "shared_ols_base256/geometry_metrics.csv", dtype={"protein_id": str},
                   usecols=["layer", "protein_id", "source", "tm_score", "triangle_violation_fraction"])
cameo = geom[geom.source.eq("cameo")]
ids = set(cameo.protein_id)
sm = pd.read_csv(ABLATION_CSV_DIR / "structure_module_tm_scores.csv", dtype={"protein": str})
sm = sm[sm.protein.isin(ids)][[f"L{t}" for t in L]].apply(pd.to_numeric, errors="coerce")
mds = cameo.groupby("layer").tm_score.mean().reindex(L).to_numpy()
out["timelines"] = [{"layer": int(t), "native_r2": round(float(med[t, t]), 3), "probe47_r2": round(float(med[47, t]), 3),
                     "mds_tm": round(float(mds[t]), 3), "sm_tm": round(float(sm[f"L{t}"].mean()), 3)} for t in L]
half = lambda y: int(np.argmax(y >= (y.max() + y.min()) / 2))
out["halfway"] = {"native_r2": half(np.diag(med)), "probe47_r2": half(med[47]), "mds_tm": half(mds), "sm_tm": half(sm.mean().to_numpy())}
out["cameo_proteins"] = len(sm)

cell = pd.read_csv(G / "gain_direction_by_cell_excl_7stz_C.csv")
cell = cell[(cell.source_layer >= 16) & (cell.target_layer >= 16) & (cell.source_layer != cell.target_layer)].copy()
cell["offset"] = (cell.target_layer - cell.source_layer).abs()
cell["way"] = np.where(cell.source_layer < cell.target_layer, "earlier probe on later layer", "later probe on earlier layer")
rows = []
for (offset, way), part in cell.groupby(["offset", "way"]):
    if offset in (1, 2, 3, 5, 8, 12, 16, 20, 25, 31):
        rows.append({"offset": int(offset), "way": way, "gain": round(float(part.gain_part.mean()), 4), "direction": round(float(part.direction_part.mean()), 4),
                     "loss": round(float(part.loss.mean()), 4), "cells": int(len(part))})
out["gain"] = rows
out["gain_overall"] = {"loss_sum": round(float(cell.loss.sum()), 2), "gain_share": round(float(cell.gain_part.sum() / cell.loss.sum()), 3)}

shards = [np.load(ROOT / "layer_shards" / f"layer_{t:02d}.npz") for t in L]
names = shards[0]["protein_ids"].astype(str)
models = list(shards[0]["models"].astype(str))
counts = np.stack([s["counts"] for s in shards])
keep = names != "7stz_C"
curve = (counts[..., 1] / counts[..., 0]).mean(axis=4)[:, :, keep].mean(axis=2)
use = {"observed": "observed", "pooled_strata": "pooled strata", "protein_strata": "per-protein strata", "residues_plus_permuted": "shuffled plus residue bias"}
rows = []
for a, probe in enumerate(("OLS", "XGBoost")):
    for name, label in use.items():
        rows.extend({"probe": probe, "model": label, "layer": int(t), "rate": round(float(curve[t, a, models.index(name)]), 4)} for t in L)
out["triangle"] = rows

v = pd.read_csv(PRODUCER / "per_protein_ols_base256/geometry_metrics.csv", dtype={"protein_id": str}, usecols=["layer", "protein_id", "triangle_violation_fraction"])
wide = v.pivot(index="protein_id", columns="layer", values="triangle_violation_fraction")
steps = wide.diff(axis=1).iloc[:, 1:]
falling = (steps < 0).mean()
out["events"] = [{"block": int(b), "falling": round(float(falling[b]), 3), "step": round(float(steps[b].mean()), 4)} for b in falling.index]
out["events_n"] = len(wide)
(ROOT / "report_chart_data.json").write_text(json.dumps(out))
print({k: (len(x) if hasattr(x, "__len__") else x) for k, x in out.items()})
print(out["halfway"], out["gain_overall"], out["cameo_proteins"], out["events_n"])

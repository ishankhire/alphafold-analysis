#!/usr/bin/env python3
"""Split the violation drops by cohort and length, list event blocks, and test the two-force account."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import spearmanr

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
OUT = "7stz_C"
FAMILIES = ("shared_ols_base256", "sequence_gated_ols_base256", "per_protein_ols_base256", "shared_xgboost_base256_fixed_s4")
BLOCKS = [8, 13, 16, 17, 18, 35, 39, 40, 42, 43, 44, 47]
pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 200)


def h(title):
    print(f"\n{'=' * 8} {title}", flush=True)


wide, meta = {}, None
for family in FAMILIES:
    t = pd.read_csv(PRODUCER / family / "geometry_metrics.csv", dtype={"protein_id": str},
                    usecols=["layer", "protein_id", "source", "n_residues", "triangle_violation_fraction"])
    wide[family] = t.pivot(index="protein_id", columns="layer", values="triangle_violation_fraction")
    meta = t.drop_duplicates("protein_id").set_index("protein_id")[["source", "n_residues"]]

h("1. VIOLATION STEP BY COHORT: mean step, then share of proteins falling")
print(meta.groupby("source").n_residues.describe().round(0).to_string())
for family in FAMILIES:
    steps = wide[family].diff(axis=1).iloc[:, 1:]
    rows = {}
    for name, ids in (("all", meta.index), ("cameo", meta.index[meta.source.eq("cameo")]), ("ecod", meta.index[meta.source.eq("ecod")])):
        s = steps.loc[steps.index.intersection(ids)]
        rows[name + "_mean"] = s.mean()[BLOCKS].round(4)
        rows[name + "_fall"] = (s < 0).mean()[BLOCKS].round(2)
    print(family)
    print(pd.DataFrame(rows).T.to_string())

h("2. SAME BLOCKS BY LENGTH TERTILE, shared OLS and per-protein OLS, 552 proteins")
tertile = pd.qcut(meta.n_residues.astype(float), 3, labels=["short", "mid", "long"])
print("edges", np.round(meta.n_residues.astype(float).quantile([1 / 3, 2 / 3]).to_numpy(), 0))
print(pd.crosstab(meta.source, tertile).to_string())
for family in (FAMILIES[0], FAMILIES[2]):
    steps = wide[family].diff(axis=1).iloc[:, 1:]
    print(family)
    print(pd.DataFrame({f"{a}_{b}": steps.loc[meta.index[(meta.source.eq(a)) & (tertile.eq(b))]].mean()[[13, 17, 35, 39, 43, 44]].round(4)
                        for a in ("cameo", "ecod") for b in ("short", "mid", "long")}).T.to_string())

h("3. EVENT BLOCKS: share of proteins falling, per-protein OLS, both cohorts separately")
steps = wide[FAMILIES[2]].diff(axis=1).iloc[:, 1:]
fall_all = (steps < 0).mean()
fall_c = (steps.loc[steps.index.intersection(meta.index[meta.source.eq("cameo")])] < 0).mean()
fall_e = (steps.loc[steps.index.intersection(meta.index[meta.source.eq("ecod")])] < 0).mean()
ordered = fall_all.sort_values(ascending=False)
print(pd.DataFrame({"all": ordered, "cameo": fall_c[ordered.index], "ecod": fall_e[ordered.index], "mean_step": steps.mean()[ordered.index]}).head(16).round(4).to_string())
print("share of blocks at or above 0.90:", int((fall_all >= .9).sum()), "| at or below 0.10:", int((fall_all <= .1).sum()), sorted(fall_all.index[fall_all <= .1].tolist()))
events = sorted(fall_all.index[fall_all >= .9].tolist())
print("events:", events, "| same list at 0.85:", sorted(fall_all.index[fall_all >= .85].tolist()))
pp = pd.read_csv(PRODUCER / FAMILIES[2] / "per_protein.csv", usecols=["layer", "protein_id", "r2"], dtype={"protein_id": str}).pivot(index="protein_id", columns="layer", values="r2")
ppd = pp.diff(axis=1).iloc[:, 1:]
print("per-protein OLS, share of proteins with R2 rising at events:", (ppd > 0).mean()[events].round(2).to_dict(), "| median block", round(float((ppd > 0).mean().median()), 2))
print("per-protein OLS, violation steps vs R2 steps, spearman", round(spearmanr(steps.mean(), ppd.mean()).statistic, 3))
print("per-protein OLS, violation steps vs share falling in the other three families:", {f[:12]: round(spearmanr(fall_all, (wide[f].diff(axis=1).iloc[:, 1:] < 0).mean()).statistic, 3) for f in FAMILIES if f != FAMILIES[2]})

h("4. DO THE EVENTS LINE UP WITH EARLIER SERIES? Mean at event blocks, against all 46 circular shifts")
pr = pd.read_csv(PRODUCER / FAMILIES[0] / "per_protein.csv", usecols=["layer", "protein_id", "r2"], dtype={"protein_id": str})
r2 = pr.groupby("layer").r2.mean().reindex(L).to_numpy()
cell = pd.read_csv(G / "gain_direction_by_cell_excl_7stz_C.csv")
diag = cell[cell.source_layer.eq(cell.target_layer)].set_index("target_layer").sort_index()
M = pd.read_csv(R / "transfer_matrix.csv").pivot(index="source_layer", columns="target_layer", values="pooled_mae_excl_7stz_C").to_numpy(float)
cost = np.array([((M[t, t + 1] - M[t + 1, t + 1]) + (M[t + 1, t] - M[t, t])) / 2 for t in L[:-1]])
sm = pd.read_csv(ABLATION_CSV_DIR / "structure_module_tm_scores.csv", dtype={"protein": str})[[f"L{t}" for t in L]].apply(pd.to_numeric, errors="coerce").mean().to_numpy()
abl = {}
for part in ("pair_transition", "tri_start_ablate_tri_end_ablate_tmu"):
    for way in ("fwd", "rev"):
        a = pd.read_csv(ABLATION_CSV_DIR / f"ablate_{part}_layerwise_{way}.csv")[[f"L{t}" for t in L]].apply(pd.to_numeric, errors="coerce").mean().to_numpy()
        abl[f"ablation {part[:14]} {way}"] = (a[:-1] - a[1:]) if way == "fwd" else (a[1:] - a[:-1])
series = {"native R2 step": np.diff(r2), "native amplitude step": np.diff(diag.amp.to_numpy()), "adjacent transfer cost": cost, "structure module TM step": np.diff(sm), **abl}
position = np.array([b - 1 for b in events])
for name, s in series.items():
    observed = s[position].mean()
    shifted = np.array([s[(position + k) % 47].mean() for k in range(1, 47)])
    print(f"{name:40s} events {observed:+.4f} other shifts mean {shifted.mean():+.4f} | exact p {(np.sum(shifted >= observed) + 1) / 47:.3f}")

h("5. WHEN MODELS MISS: absolute steps at the largest CAMEO changes, 153 proteins")
shards = [np.load(ROOT / "layer_shards" / f"layer_{t:02d}.npz") for t in L]
ids, models = shards[0]["protein_ids"].astype(str), list(shards[0]["models"].astype(str))
counts = np.stack([s["counts"] for s in shards])
keep = ids != OUT
rate = (counts[..., 1] / counts[..., 0]).mean(axis=4)
curve = rate[:, :, keep].mean(axis=2)
for a, probe in enumerate(("shared_ols_base256", "shared_xgboost_base256_fixed_s4")):
    d = np.diff(curve[:, a], axis=0)
    top = np.argsort(-np.abs(d[:, 0]))[:8] + 1
    print(probe)
    print(pd.DataFrame(d[top - 1][:, [0, 4, 6]], index=[f"block {b}" for b in top], columns=["observed", "protein_strata", "residues_plus_permuted"]).round(4).to_string())
    ratio = curve[:, a, 0][:, None] / curve[:, a]
    print("observed / model, by layer window:")
    print(pd.DataFrame({m: [ratio[list(w), k].mean() for w in (range(0, 13), range(13, 22), range(22, 35), range(35, 44), range(44, 48))] for k, m in enumerate(models) if k}, index=["L0-12", "L13-21", "L22-34", "L35-43", "L44-47"]).round(3).to_string())

h("6. TWO-FORCE TEST: does margin mean and SD account for violations?")
edges = np.stack([s["edges"] for s in shards])
for a, probe in enumerate(("shared_ols_base256", "shared_xgboost_base256_fixed_s4")):
    v = rate[:, a, :, 0][:, keep]
    mean_m, sd_m = edges[:, a, keep, 3], edges[:, a, keep, 4]
    print(probe, "cohort means by layer: r(violation, margin mean)", round(np.corrcoef(curve[:, a, 0], mean_m.mean(1))[0, 1], 3), "r(violation, margin SD)", round(np.corrcoef(curve[:, a, 0], sd_m.mean(1))[0, 1], 3))
    X = np.column_stack([mean_m.mean(1), sd_m.mean(1), np.ones(48)])
    b, *_ = np.linalg.lstsq(X, curve[:, a, 0], rcond=None)
    fit = X @ b
    print("   cohort curve ~ margin mean + margin SD: coefficients", np.round(b, 4), "R2", round(1 - ((curve[:, a, 0] - fit) ** 2).sum() / ((curve[:, a, 0] - curve[:, a, 0].mean()) ** 2).sum(), 3))
    dv, dm, ds = v - v.mean(0, keepdims=True), mean_m - mean_m.mean(0, keepdims=True), sd_m - sd_m.mean(0, keepdims=True)
    Xp = np.column_stack([dm.ravel(), ds.ravel()])
    bp, *_ = np.linalg.lstsq(Xp, dv.ravel(), rcond=None)
    print("   within protein across layers: coefficients", np.round(bp, 4), "R2", round(1 - ((dv.ravel() - Xp @ bp) ** 2).sum() / (dv.ravel() ** 2).sum(), 3))
    print("   violation share of each force, L0 to L47:", "mean", round(b[0] * (mean_m.mean(1)[47] - mean_m.mean(1)[0]), 4), "SD", round(b[1] * (sd_m.mean(1)[47] - sd_m.mean(1)[0]), 4), "observed", round(curve[47, a, 0] - curve[0, a, 0], 4))

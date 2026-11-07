#!/usr/bin/env python3
"""Compare representation changes by block with ablation and the structure module readout.

Definitions inferred from the data: FWD at L ablates blocks 0..L, and REV at L ablates L..47.
structure_module_tm_scores is read as the structure module run on the saved layer L representation.
That reading comes from the file name, the load_from_saved_layer output paths, and L47 matching the
intact TM. Its generating script is in vizfold-main.zip, which was not found.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, CLUSTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from followup_paths import ABLATION_CSV_DIR, PRODUCER, TRANSFER_ROOT  # noqa: E402

R = TRANSFER_ROOT
FAM = PRODUCER / "shared_ols_base256"
COEF = FAM / "model_artifacts/coefficients"
LAYER_COLS = [f"L{i}" for i in range(48)]
L = np.arange(48)
NAMED = (8, 13, 35, 44)
SURROGATES = 5000
RNG = np.random.default_rng(0)
pd.set_option("display.width", 190, "display.max_columns", 40, "display.max_rows", 200)


def h(title):
    print(f"\n{'=' * 8} {title}", flush=True)


folds = pd.read_csv(PRODUCER / "metadata/regression_outer_folds_v1.csv", dtype={"protein_id": str})
cohort = set(folds.protein_id)
cameo = set(folds.protein_id[folds.source.eq("cameo")])
lengths = folds.set_index("protein_id").n_residues


def load(stem):
    d = pd.read_csv(ABLATION_CSV_DIR / f"{stem}.csv", dtype={"protein": str})
    d["protein"] = [p if ds == "CAMEO" else p.zfill(9) for p, ds in zip(d.protein, d.dataset)]
    d = d[d.protein.isin(cohort)].set_index("protein")
    return d[LAYER_COLS].apply(pd.to_numeric, errors="coerce")


PARTS = {"pair_transition": "ablate_pair_transition", "msa_transition": "ablate_msa_transition", "msa_row": "ablate_msa_row", "msa_col": "ablate_msa_col",
         "tmu": "ablate_tmu", "tri_attention": "ablate_tri_start_ablate_tri_end", "tmu_plus_tri": "ablate_tri_start_ablate_tri_end_ablate_tmu"}
curves = {}
for name, stem in PARTS.items():
    for direction in ("fwd", "rev"):
        curves[(name, direction)] = load(f"{stem}_layerwise_{direction}")
sm = load("structure_module_tm_scores")
print("proteins per table:", {k: len(v) for k, v in list(curves.items())[:2]}, "structure module", len(sm), "missing cells", int(sm.isna().sum().sum()))

h("1. CURVES: mean TM by boundary layer, 552 proteins (CAMEO in brackets for structure module)")
show = [0, 4, 8, 12, 13, 16, 20, 24, 28, 32, 34, 35, 36, 40, 43, 44, 46, 47]
print(pd.DataFrame({f"{a}_{b}": curves[(a, b)].mean().to_numpy()[show] for a, b in curves}, index=show).round(2).T.to_string())
print("structure module:", dict(zip(show, np.round(sm.mean().to_numpy()[show], 3))))
print("structure module CAMEO:", dict(zip(show, np.round(sm[sm.index.isin(cameo)].mean().to_numpy()[show], 3))))


def crossing(y):
    y = np.asarray(y)
    mid = (y.max() + y.min()) / 2
    return int(np.argmax(y >= mid)) if y[-1] > y[0] else int(np.argmax(y <= mid))


print("half-way layer:", {f"{a}_{b}": crossing(c.mean()) for (a, b), c in curves.items()}, "structure module", crossing(sm.mean()))
print("intact reference, mean TM of REV at 47 and FWD at 0:", {a: (round(float(curves[(a, 'rev')]["L47"].mean()), 3), round(float(curves[(a, 'fwd')]["L0"].mean()), 3)) for a in PARTS})

h("2. PROBE SERIES BY LAYER")
M = pd.read_csv(R / "transfer_matrix.csv")
mat = lambda col: M.pivot(index="source_layer", columns="target_layer", values=col).to_numpy(float)
med, r2x, maex = mat("median_protein_r2"), mat("pooled_r2_excl_7stz_C"), mat("pooled_mae_excl_7stz_C")
pp = pd.read_csv(FAM / "per_protein.csv", usecols=["layer", "protein_id", "r2"], dtype={"protein_id": str})
full_r2 = pp.groupby("layer").r2.mean().reindex(L).to_numpy()
W = np.stack([[np.load(COEF / f"fold_{f}_layer_{l:02d}.npz")["coefficient_standardized"].astype(float) for f in range(5)] for l in L])
cos = np.mean([(W[:, f] @ W[:, f].T) / np.outer(np.linalg.norm(W[:, f], axis=1), np.linalg.norm(W[:, f], axis=1)) for f in range(5)], axis=0)
geom = pd.read_csv(FAM / "geometry_summary.csv")
tm = geom[geom.geometry_metric.eq("tm_score")].set_index("layer")["mean"].reindex(L).to_numpy()
tri = geom[geom.geometry_metric.eq("triangle_violation_fraction")].set_index("layer")["mean"].reindex(L).to_numpy()
adjust = lambda x: np.r_[np.nan, x]
series = {
    "dR2_median_cameo": adjust(np.diff(np.diag(med))), "dR2_mean_552": adjust(np.diff(full_r2)),
    "rotation": adjust(1 - np.array([cos[l, l + 1] for l in L[:-1]])),
    "adjacent_cost": adjust(np.array([((maex[l, l + 1] - maex[l + 1, l + 1]) + (maex[l + 1, l] - maex[l, l])) / 2 for l in L[:-1]])),
    "dTM_mds_552": adjust(np.diff(tm)), "dTriangle_552": adjust(np.diff(tri)),
    "dStructureModule": adjust(np.diff(sm.mean().to_numpy())),
}
blocks = np.arange(1, 48)
print(pd.DataFrame({k: v[list(NAMED)] for k, v in series.items()}, index=[f"block {b}" for b in NAMED]).round(3).to_string())
print("rank among 47 blocks, 1 = largest:", {k: {b: int(pd.Series(v[1:], index=blocks).rank(ascending=False)[b]) for b in NAMED} for k, v in series.items()})
print("top blocks:", {k: pd.Series(v[1:], index=blocks).sort_values(ascending=False).head(5).round(3).to_dict() for k, v in series.items()})


def marginals(curve, direction):
    """Mean TM lost by adding block b to the ablated set, per block, over the given proteins."""
    y = curve.to_numpy(float)
    out = np.full(48, np.nan)
    if direction == "fwd":
        out[1:] = np.nanmean(y[:, :-1] - y[:, 1:], axis=0)
    else:
        out[:-1] = np.nanmean(y[:, 1:] - y[:, :-1], axis=0)
    return out


def ranks(a):
    return rankdata(a, axis=-1)


def surrogate_p(x, y):
    """Two-sided p for Spearman under phase-randomized surrogates of x that keep its spectrum."""
    ok = ~(np.isnan(x) | np.isnan(y))
    x, y = x[ok], y[ok]
    rx, ry = ranks(x), ranks(y)
    observed = np.corrcoef(rx, ry)[0, 1]
    spectrum = np.fft.rfft(rx - rx.mean())
    phases = RNG.uniform(0, 2 * np.pi, (SURROGATES, spectrum.size))
    phases[:, 0] = 0
    if rx.size % 2 == 0:
        phases[:, -1] = 0
    fake = np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=rx.size, axis=1)
    rf = ranks(fake)
    rf -= rf.mean(1, keepdims=True)
    rc = ry - ry.mean()
    null = (rf @ rc) / (np.linalg.norm(rf, axis=1) * np.linalg.norm(rc))
    return observed, float((np.sum(np.abs(null) >= abs(observed)) + 1) / (SURROGATES + 1))


h("3. DO BLOCK-LEVEL REPRESENTATION CHANGES MATCH ABLATION SENSITIVITY? Spearman, p from spectrum-preserving surrogates")
rows, margin = [], {}
for (name, direction), curve in curves.items():
    y = marginals(curve, direction)
    margin[(name, direction)] = y
    for key in ("dR2_mean_552", "rotation", "adjacent_cost", "dStructureModule"):
        rho, p = surrogate_p(series[key], y)
        rows.append((name, direction, key, round(rho, 2), round(p, 3)))
table = pd.DataFrame(rows, columns=["part", "direction", "series", "rho", "p"])
print(table.pivot_table(index=["part", "direction"], columns="series", values="rho").round(2).to_string())
print(table.pivot_table(index=["part", "direction"], columns="series", values="p").round(3).to_string())
print(f"{len(table)} tests; smallest p {table.p.min():.4f}; Bonferroni threshold {0.05 / len(table):.4f}; p<0.05: {(table.p < .05).sum()}, expected by chance {0.05 * len(table):.1f}")
for b in NAMED:
    print(f"block {b}: marginal TM loss, rank of 47 in brackets:", {f"{a}_{d}": (round(float(y[b]), 3), int(pd.Series(y).rank(ascending=False)[b])) for (a, d), y in margin.items() if not np.isnan(y[b]) and np.nanmax(y) > .01})
print("TM lost at each block where the curve is informative, largest five blocks:")
for (a, d), y in margin.items():
    print(f"  {a}_{d}:", pd.Series(y).sort_values(ascending=False).head(5).round(3).to_dict())

h("4. WHAT DOES THE STRUCTURE MODULE'S USABLE SIGNAL TRACK? Levels by layer")
sm_mean = sm.mean().to_numpy()
sm_cameo = sm[sm.index.isin(cameo)].mean().to_numpy()
levels = {
    "native median R2": np.diag(med), "native mean R2 (552)": full_r2, "MDS TM (552)": tm,
    "47 probe on L, median protein R2": med[47], "47 probe on L, pooled R2 excl": r2x[47],
    "L probe on 47, median protein R2": med[:, 47], "cosine to layer 47 probe": cos[:, 47],
    "native triangle violation": tri,
}
print("half-way layer of structure module:", crossing(sm_mean), "(CAMEO", crossing(sm_cameo), ")")
for name, y in levels.items():
    print(f"{name:36s} spearman with SM TM {pd.Series(y).corr(pd.Series(sm_mean), method='spearman'):6.3f} (CAMEO {pd.Series(y).corr(pd.Series(sm_cameo), method='spearman'):6.3f}) | half-way layer {crossing(y)} | values at L16 {y[16]:.2f} L34 {y[34]:.2f} L36 {y[36]:.2f} L40 {y[40]:.2f} L44 {y[44]:.2f}")
print("largest structure module steps:", pd.Series(np.diff(sm_mean), index=[f"{a}>{a + 1}" for a in L[:-1]]).sort_values(ascending=False).head(8).round(3).to_dict())

h("5. SAME TEST ACROSS PROTEINS, within each layer (CAMEO, 154 proteins)")
P = pd.read_csv(R / "per_protein.csv", usecols=["protein_id", "source_layer", "target_layer", "r2", "mae"], dtype={"protein_id": str})
native = P[P.source_layer.eq(P.target_layer)].pivot(index="protein_id", columns="target_layer", values="r2")
late_on = P[P.source_layer.eq(47)].pivot(index="protein_id", columns="target_layer", values="r2")
to_late = P[P.target_layer.eq(47)].pivot(index="protein_id", columns="source_layer", values="r2")
smc = sm[sm.index.isin(cameo)].copy()
smc.columns = L


def rank_residual(a, b):
    a, b = rankdata(a), rankdata(b)
    return a - np.polyval(np.polyfit(b, a, 1), b)


rows = []
for layer in (16, 20, 24, 28, 30, 32, 34, 35, 36, 37, 38, 39, 40, 42, 44, 46):
    y = smc[layer].dropna()
    ids = y.index
    size = lengths.reindex(ids).astype(float)
    row = {"layer": layer, "SM_mean": y.mean(), "SM_sd": y.std()}
    for name, table_ in (("native_r2", native), ("47_on_L", late_on), ("L_on_47", to_late)):
        x = table_[layer].reindex(ids)
        ok = x.notna().to_numpy() & size.notna().to_numpy()
        row[f"rho_{name}"] = x[ok].corr(y[ok], method="spearman")
        row[f"partial_{name}"] = np.corrcoef(rank_residual(x[ok], size[ok]), rank_residual(y[ok], size[ok]))[0, 1]
    row["rho_length"] = size.corr(y, method="spearman")
    rows.append(row)
print(pd.DataFrame(rows).round(2).to_string(index=False))
print("per-protein half-way layer of structure module: quantiles", np.round(pd.Series([crossing(r.to_numpy()) for _, r in smc.iterrows()]).quantile([.1, .25, .5, .75, .9]).to_numpy(), 1),
      "| spearman with length", round(float(pd.Series([crossing(r.to_numpy()) for _, r in smc.iterrows()], index=smc.index).corr(lengths.reindex(smc.index).astype(float), method="spearman")), 3))

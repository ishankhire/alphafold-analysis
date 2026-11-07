#!/usr/bin/env python3
"""Read-only interpretation of the CAMEO all-layer transfer results. Prints a text report."""
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, CLUSTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from followup_paths import PRODUCER, TRANSFER_ROOT  # noqa: E402

R = TRANSFER_ROOT
FAM = PRODUCER / "shared_ols_base256"
COEF = FAM / "model_artifacts/coefficients"
OUT = "7stz_C"
L = np.arange(48)
pd.set_option("display.width", 190, "display.max_columns", 40, "display.max_rows", 400)


def h(title):
    print(f"\n{'=' * 8} {title}", flush=True)


M = pd.read_csv(R / "transfer_matrix.csv")


def mat(col, table=M):
    """Rows are source layers, columns are target layers."""
    return table.pivot(index="source_layer", columns="target_layer", values=col).to_numpy(dtype=float)


r2, r2x, med, mean, mae, maex, medmae = (mat(c) for c in (
    "pooled_r2", f"pooled_r2_excl_{OUT}", "median_protein_r2", "mean_protein_r2", "pooled_mae",
    f"pooled_mae_excl_{OUT}", "median_protein_mae"))
rmse = mat("pooled_rmse")
nat = lambda m: np.diag(m)

h("A. NATIVE TRAJECTORY (CAMEO)")
full = pd.read_csv(FAM / "pooled_metrics.csv").sort_values("layer")
geom = pd.read_csv(FAM / "geometry_summary.csv")
tm = geom[geom.geometry_metric.eq("tm_score")].sort_values("layer")
tri = geom[geom.geometry_metric.eq("triangle_violation_fraction")].sort_values("layer")
table = pd.DataFrame({
    "pool_r2": nat(r2), "pool_r2_x": nat(r2x), "med_prot_r2": nat(med), "mean_prot_r2": nat(mean),
    "pool_mae_x": nat(maex), "med_prot_mae": nat(medmae), "full552_pool_r2": full.pooled_r2.to_numpy(),
    "full552_tm_mean": tm["mean"].to_numpy(), "full552_tri_viol": tri["mean"].to_numpy(),
}, index=L).round(3)
print(table.loc[[0, 1, 2, 4, 8, 12, 13, 14, 15, 16, 17, 18, 20, 22, 26, 30, 33, 34, 35, 36, 40, 42, 43, 44, 45, 46, 47]])
step = pd.Series(np.diff(nat(med)), index=[f"{a}>{a + 1}" for a in L[:-1]])
print("largest increases, median protein R2:", step.sort_values(ascending=False).head(7).round(3).to_dict())
print("largest decreases, median protein R2:", step.sort_values().head(6).round(3).to_dict())
step = pd.Series(np.diff(nat(r2x)), index=[f"{a}>{a + 1}" for a in L[:-1]])
print("largest increases, pooled R2 excl:", step.sort_values(ascending=False).head(6).round(3).to_dict())
print("non-monotonic layers (median protein R2 falls):", int((np.diff(nat(med)) < 0).sum()), "of 47")


def segment(X, k):
    """Optimal contiguous segmentation of the rows of X into k blocks."""
    n = len(X)
    cum, cum2 = np.vstack([np.zeros(X.shape[1]), np.cumsum(X, 0)]), np.r_[0, np.cumsum((X * X).sum(1))]

    def cost(i, j):
        s = cum[j] - cum[i]
        return cum2[j] - cum2[i] - (s @ s) / (j - i)
    best = np.full((k + 1, n + 1), np.inf)
    back = np.zeros((k + 1, n + 1), dtype=int)
    best[0, 0] = 0
    for c in range(1, k + 1):
        for j in range(c, n + 1):
            options = [best[c - 1, i] + cost(i, j) for i in range(c - 1, j)]
            arg = int(np.argmin(options))
            best[c, j], back[c, j] = options[arg], arg + c - 1
    cuts, j = [], n
    for c in range(k, 0, -1):
        j = back[c, j]
        cuts.append(j)
    total = ((X - X.mean(0)) ** 2).sum()
    return sorted(c for c in cuts if c > 0), 1 - best[k, n] / total


h("B. BLOCK STRUCTURE (optimal contiguous segmentation; cut value = first layer of a new block)")
clip = np.clip(med, -1, 1)
for name, X in (("sources (median protein R2)", clip), ("targets (median protein R2)", clip.T),
                ("sources (pooled R2 excl)", np.clip(r2x, -1, 1)), ("targets (pooled R2 excl)", np.clip(r2x, -1, 1).T)):
    print(name, {k: (segment(X, k)[0], round(segment(X, k)[1], 3)) for k in (2, 3, 4, 5, 6)})
F = pd.read_csv(R / "fold_metrics.csv")
for fold in range(5):
    Y = np.clip(mat("pooled_r2", F[F.outer_fold.eq(fold)]), -1, 1)
    print(f"fold {fold}: sources k=3 {segment(Y, 3)[0]}, targets k=3 {segment(Y.T, 3)[0]}, k=4 {segment(Y, 4)[0]} / {segment(Y.T, 4)[0]}")

h("C. ROW AND COLUMN PROFILES, off-diagonal median protein R2")
off = med.copy()
np.fill_diagonal(off, np.nan)
print("source s, mean on targets 16-47:", {int(s): round(float(np.nanmean(off[s, 16:])), 2) for s in L[::3]})
print("target t, mean from sources 16-47:", {int(t): round(float(np.nanmean(off[16:, t])), 2) for t in L[::3]})
print("targets t=10..20 from sources 22-47:", {int(t): round(float(np.nanmean(off[22:, t])), 3) for t in range(10, 21)})
print("sources s=8..20 on targets 22-47:", {int(s): round(float(np.nanmean(off[s, 22:])), 3) for s in range(8, 21)})

h("D. DIRECTION. eff = transfer R2 / target's native R2 (median protein)")
eff = med / nat(med)[None, :]
effx = r2x / nat(r2x)[None, :]
print(" |off|  fwd(early probe on later) back(later probe on earlier), layers 16-47 only; pooled R2 excl in brackets")
for k in (1, 2, 3, 5, 8, 12, 16, 20, 25, 31):
    f = [(eff[s, s + k], effx[s, s + k]) for s in range(16, 48 - k)]
    b = [(eff[s + k, s], effx[s + k, s]) for s in range(16, 48 - k)]
    print(f"{k:5d}  {np.median([x[0] for x in f]):.3f} ({np.median([x[1] for x in f]):.3f})   {np.median([x[0] for x in b]):.3f} ({np.median([x[1] for x in b]):.3f})")
print("whole matrix, absolute pooled MAE excl by |offset| (fwd vs back) and excess over native target")
for k in (5, 10, 20, 30, 40):
    a = [maex[s, s + k] for s in range(48 - k)]
    b = [maex[s + k, s] for s in range(48 - k)]
    x = [maex[s, s + k] - maex[s + k, s + k] for s in range(48 - k)]
    y = [maex[s + k, s] - maex[s, s] for s in range(48 - k)]
    print(f"{k:3d}  abs {np.median(a):.2f} vs {np.median(b):.2f}   excess {np.median(x):.2f} vs {np.median(y):.2f}")
same = [(s, t) for s in range(16, 48) for t in range(s + 1, 48)]
print("layers 16-47: fraction of pairs where early-on-late eff > late-on-early eff:",
      round(float(np.mean([eff[s, t] > eff[t, s] for s, t in same])), 3))

h("E. DOES ANY NON-NATIVE PROBE BEAT THE TARGET'S OWN PROBE?")
for name, X, sign in (("pooled MAE excl", maex, 1), ("median protein MAE", medmae, 1), ("pooled R2 excl", r2x, -1), ("median protein R2", med, -1)):
    D = sign * (X - nat(X)[None, :])
    np.fill_diagonal(D, np.nan)
    wins = D < 0
    print(name, "cells better than native:", int(np.nansum(wins)), "of 2256; by |offset| 1:", int(sum(wins[s, t] for s in L for t in L if abs(s - t) == 1)),
          "2:", int(sum(wins[s, t] for s in L for t in L if abs(s - t) == 2)), "3-5:", int(sum(wins[s, t] for s in L for t in L if 3 <= abs(s - t) <= 5)),
          "6+:", int(sum(wins[s, t] for s in L for t in L if abs(s - t) >= 6)))
D = med - nat(med)[None, :]
np.fill_diagonal(D, -9)
print("best margin cells (median protein R2):", [(int(s), int(t), round(float(D[s, t]), 4)) for s, t in zip(*np.unravel_index(np.argsort(-D, axis=None)[:10], D.shape))])

h("F. REACH: targets where a probe holds >=95% / >=90% of the target's native median protein R2")
print({int(s): (int((eff[s] >= .95).sum()), int((eff[s] >= .90).sum())) for s in L[::3]})
print("targets t: sources holding >=95%:", {int(t): int((eff[:, t] >= .95).sum()) for t in L[::3]})
ranked = pd.Series([np.nanmean(off[s, 16:]) for s in L], index=L).sort_values(ascending=False)
print("best universal probes on targets 16-47 (mean median protein R2):", ranked.head(8).round(3).to_dict())
print("best on all targets:", pd.Series([np.nanmean(off[s]) for s in L], index=L).sort_values(ascending=False).head(5).round(3).to_dict())

ST = pd.read_csv(R / "error_strata.csv")
XN, XE, XA, XS = f"n_pairs_excl_{OUT}", f"sum_signed_error_excl_{OUT}", f"sum_abs_error_excl_{OUT}", f"sum_squared_error_excl_{OUT}"
DIST = ["0-4", "4-5", "5-6", "6-7", "7-8", "8-10", "10-12", "12-15", "15-20", "20-25", "25-30", "30-40", "40-50", "50-60", "60-80", "80-100", "100+"]
MID = dict(zip(DIST, [3, 4.5, 5.5, 6.5, 7.5, 9, 11, 13.5, 17.5, 22.5, 27.5, 35, 45, 55, 70, 90, 120]))
CD, CS = ["0-5", "5-10", "10-20", "20-40", "40+"], ["1", "2-4", "5-9", "10-19", "20-49", "50+"]

h("G. ORDER OF EMERGENCE, native, excl 7stz_C. t50 = first layer with half of the MAE drop from layer 0 to layer 47")
native = ST[ST.is_same_layer]


def traj(frame):
    g = frame.groupby("target_layer")[[XN, XA]].sum()
    return (g[XA] / g[XN]).reindex(L).to_numpy()


def t_half(y):
    return int(np.argmax(y <= y[0] - .5 * (y[0] - y[47])))


rows = []
for name, sel in [(f"dist {d}", native.final_structure_distance_bin.eq(d)) for d in DIST] + \
                 [(f"canon dist {d}", native.canonical_distance_bin.eq(d)) for d in CD] + \
                 [(f"seq {s}", native.canonical_sequence_bin.eq(s)) for s in CS] + \
                 [("contacts seq>=20 d<8", native.canonical_sequence_bin.isin(["20-49", "50+"]) & native.final_structure_distance_bin.isin(DIST[:5])),
                  ("local seq<=4", native.canonical_sequence_bin.isin(["1", "2-4"]))]:
    y = traj(native[sel])
    rows.append((name, round(y[0], 2), round(y[8], 2), round(y[16], 2), round(y[22], 2), round(y[33], 2), round(y[43], 2), round(y[47], 2), t_half(y), round(1 - y[47] / y[0], 2)))
print(pd.DataFrame(rows, columns=["stratum", "L0", "L8", "L16", "L22", "L33", "L43", "L47", "t50", "frac_drop"]).to_string(index=False))

h("H. CALIBRATION: mean signed error by fine distance, excl 7stz_C")
cells = [(0, 0), (22, 22), (47, 47), (22, 47), (47, 22), (0, 47), (47, 0), (30, 47), (47, 30), (16, 47)]
rows = {}
for s, t in cells:
    c = ST[ST.source_layer.eq(s) & ST.target_layer.eq(t)].groupby("final_structure_distance_bin")[[XN, XE]].sum()
    rows[f"{s}>{t}"] = (c[XE] / c[XN]).reindex(DIST).round(2)
print(pd.DataFrame(rows).to_string())
slopes = []
for s, t in cells:
    c = ST[ST.source_layer.eq(s) & ST.target_layer.eq(t)].groupby("final_structure_distance_bin")[[XN, XE]].sum().reindex(DIST)
    d = np.array([MID[b] for b in DIST])
    w = c[XN].to_numpy(dtype=float)
    e = (c[XE] / c[XN]).to_numpy()
    low = d < 60
    slopes.append((f"{s}>{t}", round(np.polyfit(d[low], e[low], 1, w=np.sqrt(w[low]))[0], 3), round(np.polyfit(d, e, 1, w=np.sqrt(w))[0], 3)))
print("slope of error on distance (0 = calibrated, negative = compressed): (cell, under 60 A, all)", slopes)

h("I. WHERE TRANSFER COST SITS: share of excess squared error over the target's native probe, excl 7stz_C")
base = native.set_index(["target_layer", "canonical_sequence_bin", "canonical_distance_bin", "sequence_separation_bin", "final_structure_distance_bin"])[XS]
for name, sel in (("within 16-47, |offset|>=5", ST.source_layer.ge(16) & ST.target_layer.ge(16) & (ST.layer_offset.abs() >= 5)),
                  ("late probe on early layer", ST.source_layer.ge(16) & ST.target_layer.le(13)),
                  ("early probe on late layer", ST.source_layer.le(12) & ST.target_layer.ge(16))):
    f = ST[sel]
    native_sse = base.reindex(pd.MultiIndex.from_frame(f[["target_layer", "canonical_sequence_bin", "canonical_distance_bin", "sequence_separation_bin", "final_structure_distance_bin"]])).to_numpy()
    f = f.assign(extra=f[XS].to_numpy() - native_sse)
    g = f.groupby(["canonical_sequence_bin", "canonical_distance_bin"]).extra.sum().unstack().reindex(index=CS, columns=CD)
    print(name, "| total excess relative to native SSE:", round(float(f.extra.sum() / np.nansum(native_sse)), 2))
    print((100 * g / f.extra.sum()).round(1).to_string())

h("J. CHAIN ENDS, excl 7stz_C. MAE ratio of end bin to median of bins 5-14")
POS = pd.read_csv(R / "error_position.csv")
rows = []
for s, t in [(l, l) for l in (0, 8, 16, 22, 33, 43, 47)] + [(22, 47), (47, 22), (0, 47), (47, 0)]:
    c = POS[POS.source_layer.eq(s) & POS.target_layer.eq(t)].sort_values("position_bin")
    y, m = c[f"mae_excl_{OUT}"].to_numpy(), c[f"mean_signed_error_excl_{OUT}"].to_numpy()
    mid = np.median(y[5:15])
    rows.append((f"{s}>{t}", round(mid, 2), round(y[0], 2), round(y[1], 2), round(y[19], 2), round(y[0] / mid, 2), round(y[19] / mid, 2), round(m[0], 2), round(m[19], 2), round(np.median(m[5:15]), 2)))
print(pd.DataFrame(rows, columns=["cell", "mid", "first5%", "next5%", "last5%", "Nratio", "Cratio", "Nbias", "Cbias", "midbias"]).to_string(index=False))

h("K. PROTEINS")
PP = pd.read_csv(R / "per_protein.csv", usecols=["protein_id", "source_layer", "target_layer", "is_same_layer", "outer_fold", "n_residues", "r2", "mae", "rmse", "bias_pred_minus_true", "mean_true_distance", "target_variance"], dtype={"protein_id": str})
NP = PP[PP.is_same_layer]
rows = []
for layer in (0, 8, 16, 22, 33, 47):
    c = NP[NP.target_layer.eq(layer)]
    rows.append((layer, *np.round(c.r2.quantile([.1, .25, .5, .75, .9]).to_numpy(), 3), round(c.mae.median(), 2),
                 round(c[["n_residues", "mae"]].corr("spearman").iloc[0, 1], 2), round(c[["mean_true_distance", "mae"]].corr("spearman").iloc[0, 1], 2),
                 round(c[["n_residues", "r2"]].corr("spearman").iloc[0, 1], 2)))
print(pd.DataFrame(rows, columns=["layer", "r2_q10", "q25", "q50", "q75", "q90", "med_mae", "rho(len,mae)", "rho(meand,mae)", "rho(len,r2)"]).to_string(index=False))
c = NP[NP.target_layer.eq(47)].copy()
c["tertile"] = pd.qcut(c.n_residues, 3, labels=["short", "mid", "long"])
print(c.groupby("tertile", observed=True).agg(R_min=("n_residues", "min"), R_max=("n_residues", "max"), r2=("r2", "median"), mae=("mae", "median"), rmse=("rmse", "median"), bias=("bias_pred_minus_true", "median")).round(3).to_string())
print("worst native R2 at layer 47:")
print(c.nsmallest(8, "r2")[["protein_id", "outer_fold", "n_residues", "r2", "mae", "bias_pred_minus_true", "mean_true_distance"]].round(2).to_string(index=False))
print("highest MAE at layer 47:")
print(c.nlargest(6, "mae")[["protein_id", "outer_fold", "n_residues", "r2", "mae", "bias_pred_minus_true", "mean_true_distance"]].round(2).to_string(index=False))
x = NP[NP.protein_id.eq(OUT)].sort_values("target_layer")
print(f"{OUT} native: ", {int(l): (round(float(a), 2), round(float(b), 1), round(float(d), 1)) for l, a, b, d in zip(x.target_layer, x.r2, x.mae, x.bias_pred_minus_true) if l in (0, 16, 22, 33, 43, 44, 47)})
share = mat(f"squared_error_share_{OUT}")
print(f"{OUT} squared-error share: native min/median/max", np.round([nat(share).min(), np.median(nat(share)), nat(share).max()], 3), "all cells", np.round([share.min(), np.median(share), share.max()], 3))
A = pd.read_csv(R / "protein_anomalies.csv", dtype={"protein_id": str})
g = A.groupby("protein_id").agg(cells=("rank", "size"), mean_share=("squared_error_share", "mean"), median_z=("mae_robust_z", "median"), R=("n_residues", "first"), fold=("outer_fold", "first")).sort_values("cells", ascending=False).head(9)
print(g.round(3).to_string())

h("L. PAIRED PROTEIN VIEW OF TRANSFER")
wide = PP.pivot_table(index="protein_id", columns=["source_layer", "target_layer"], values="mae")
for s, t in ((22, 47), (47, 22), (30, 47), (33, 34), (44, 45), (46, 47), (47, 46), (43, 44), (44, 43), (16, 47), (47, 16)):
    d = wide[(s, t)] - wide[(t, t)]
    print(f"{s}>{t}: median excess MAE {d.median():.2f}, IQR {d.quantile(.25):.2f} to {d.quantile(.75):.2f}, proteins better than native {100 * (d < 0).mean():.0f}%, worst {d.idxmax()} +{d.max():.1f}")

h("M. WHAT DOES SEQUENCE ALONE EXPLAIN? in-sample oracle R2 for the 154 CAMEO proteins")
with np.load(R / "layer_shards/target_layer_47.npz") as shard:
    probe = shard["probe"][0]
    ids = shard["protein_ids"].astype(str)
keep_sets = {"all": np.ones(len(ids), bool), f"excl {OUT}": ids != OUT}
for name, keep in keep_sets.items():
    p = probe[keep]
    n, sy, sy2 = p[:, 0, 0].sum(), p[:, 0, 1].sum(), p[:, 0, 2].sum()
    sst = sy2 - sy * sy / n
    seq = p[:, 1:7]
    nb, sb, s2b = seq[..., 0].sum(0), seq[..., 1].sum(0), seq[..., 2].sum(0)
    seq_only = 1 - (s2b - sb * sb / nb).sum() / sst
    prot_only = 1 - (p[:, 0, 2] - p[:, 0, 1] ** 2 / p[:, 0, 0]).sum() / sst
    with np.errstate(divide="ignore", invalid="ignore"):
        both = np.nan_to_num(seq[..., 2] - seq[..., 1] ** 2 / seq[..., 0])
    print(f"{name}: six sequence bins {seq_only:.3f}; protein mean {prot_only:.3f}; protein x six bins {1 - both.sum() / sst:.3f}")

h("N. PROBE WEIGHTS")
fits = {}
for layer in L:
    for fold in range(5):
        with np.load(COEF / f"fold_{fold}_layer_{layer:02d}.npz") as z:
            fits[(layer, fold)] = (z["coefficient_standardized"].astype(float), z["feature_scale"].astype(float), z["feature_mean"].astype(float))
W = np.stack([[fits[(l, f)][0] for f in range(5)] for l in L])            # (layers, folds, 256)
S = np.stack([[fits[(l, f)][1] for f in range(5)] for l in L])
U = np.stack([[fits[(l, f)][2] for f in range(5)] for l in L])


def cosine(a, b):
    return a @ b.T / np.outer(np.linalg.norm(a, axis=1), np.linalg.norm(b, axis=1))


folds = np.stack([cosine(W[:, f], W[:, f]) for f in range(5)]).mean(0)
print("norm of standardized weights by layer:", {int(l): round(float(np.linalg.norm(W[l], axis=1).mean()), 1) for l in L[::6]}, "L47", round(float(np.linalg.norm(W[47], axis=1).mean()), 1))
adj = np.array([folds[l, l + 1] for l in L[:-1]])
print("cosine of adjacent-layer probes: median", round(float(np.median(adj)), 3), "min", {f"{int(a)}>{int(a) + 1}": round(float(adj[a]), 3) for a in np.argsort(adj)[:6]})
across = []
for l in L:
    c = cosine(W[l], W[l])
    across.append(c[np.triu_indices(5, 1)].mean())
print("cosine between folds at same layer:", {int(l): round(float(across[l]), 3) for l in L[::6]}, "L47", round(float(across[47]), 3))
for l in (0, 22, 33, 47):
    c = cosine(W[l], W[l])
    print(f"layer {l}: fold 3 (trained with no {OUT} held out differently) mean cos to others {c[3, [0, 1, 2, 4]].mean():.3f}; others among themselves {c[np.ix_([0, 1, 2, 4], [0, 1, 2, 4])][np.triu_indices(4, 1)].mean():.3f}")
print("cosine by |offset|:", {k: round(float(np.mean([folds[s, s + k] for s in range(48 - k)])), 3) for k in (1, 2, 3, 5, 10, 20, 30, 40)})
print("mean cosine 22 vs 47:", round(float(folds[22, 47]), 3), "| within 16-47 off-diagonal median", round(float(np.median(folds[16:, 16:][np.triu_indices(32, 1)])), 3))
print("segmentation of weight cosine (k=3,4):", segment(folds, 3)[0], segment(folds, 4)[0])
iu = np.triu_indices(48, 1)
tril_ok = ~np.eye(48, dtype=bool)
print("Spearman, cosine vs median protein R2 over off-diagonal cells:", round(float(pd.Series(folds[tril_ok]).corr(pd.Series(clip[tril_ok]), method="spearman")), 3))
sel = np.zeros((48, 48), bool)
sel[16:, 16:] = True
sel &= tril_ok
print("  within 16-47:", round(float(pd.Series(folds[sel]).corr(pd.Series(clip[sel]), method="spearman")), 3))
print("  within 16-47 vs efficiency:", round(float(pd.Series(folds[sel]).corr(pd.Series(eff[sel]), method="spearman")), 3))
Wm = W.mean(1)
u, sv, vt = np.linalg.svd(Wm - Wm.mean(0), full_matrices=False)
print("variance of mean weights across layers by component:", np.round((sv ** 2 / (sv ** 2).sum())[:6], 3))
print("PC1 scores by layer:", {int(l): round(float(u[l, 0] * sv[0]), 1) for l in L[::4]}, "PC2:", {int(l): round(float(u[l, 1] * sv[1]), 1) for l in L[::8]})
print("direction symmetry, cosine of w_ij and w_ji:", {int(l): round(float(np.mean([cosine(W[l, f, :128][None], W[l, f, 128:][None])[0, 0] for f in range(5)])), 3) for l in (0, 8, 16, 22, 33, 47)})
top = lambda l, k=20: set(np.argsort(-np.abs(Wm[l]))[:k])
print("top-20 weight overlap:", {f"{a}&{b}": len(top(a) & top(b)) for a, b in ((22, 23), (22, 26), (22, 47), (33, 47), (46, 47), (0, 47), (16, 22))})
logs = np.log(S).mean(1)
sc = np.corrcoef(logs)
meansim = cosine(U.mean(1), U.mean(1))
print("correlation of log feature scales: 0-47", round(float(sc[0, 47]), 3), "22-47", round(float(sc[22, 47]), 3), "22-23", round(float(sc[22, 23]), 3), "| median scale by layer", {int(l): round(float(np.median(S[l])), 3) for l in L[::8]})
print("largest step in feature means (1 - cosine):", {f"{int(a)}>{int(a) + 1}": round(float(1 - meansim[a, a + 1]), 4) for a in np.argsort([-(1 - meansim[l, l + 1]) for l in L[:-1]])[:6]})
print("largest step in log scales (RMS):", {f"{int(a)}>{int(a) + 1}": round(float(np.sqrt(np.mean((logs[a + 1] - logs[a]) ** 2))), 4) for a in np.argsort([-np.mean((logs[l + 1] - logs[l]) ** 2) for l in L[:-1]])[:6]})

h("O. SHAPE OF ERROR DISTRIBUTIONS")
Q = pd.read_csv(R / "error_quantiles.csv")
W2 = Q[Q.category_scope.eq("all_pairs") & Q.weighting.eq(f"pair_excl_{OUT}")]
rows = []
for s, t in cells:
    r = W2[W2.source_layer.eq(s) & W2.target_layer.eq(t)].iloc[0]
    rows.append((f"{s}>{t}", r.signed_error_q01, r.signed_error_q05, r.signed_error_q25, r.signed_error_q50, r.signed_error_q75, r.signed_error_q95, r.signed_error_q99, r.signed_error_iqr, r.share_within_2A, r.share_within_5A, r.share_within_10A))
print(pd.DataFrame(rows, columns=["cell", "q01", "q05", "q25", "q50", "q75", "q95", "q99", "iqr", "w2A", "w5A", "w10A"]).round(2).to_string(index=False))
Pm = mat("signed_error_q50", Q[Q.category_scope.eq("all_pairs") & Q.weighting.eq("pair")])
rm = pd.Series([np.mean(Pm[s, 16:]) for s in L], index=L)
print("median error by source, mean over targets 16-47:", rm.round(2).to_dict())
print("biggest row jumps:", rm.diff().abs().sort_values(ascending=False).head(6).round(2).to_dict())
print("mean cohort bias of cells (should be near 0 by construction of restandardization): median |bias|", round(float(np.median(np.abs(mat("pooled_bias_pred_minus_true")))), 3))
print("open tail share:", float((Q[Q.weighting.eq("pair") & Q.category_scope.eq("all_pairs")].share_in_open_tail_bins).max()))

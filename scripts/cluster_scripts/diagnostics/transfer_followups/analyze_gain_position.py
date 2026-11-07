#!/usr/bin/env python3
"""Separate gain from direction in transfer loss, and test chain ends against distance mix."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, CLUSTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from followup_paths import GAIN_ROOT, TRANSFER_ROOT  # noqa: E402

PRODUCTION = TRANSFER_ROOT
ROOT = GAIN_ROOT
OUT = "7stz_C"
L = np.arange(48)
DIST = ["0-4", "4-5", "5-6", "6-7", "7-8", "8-10", "10-12", "12-15", "15-20", "20-25", "25-30", "30-40", "40-50", "50-60", "60-80", "80-100", "100+"]
SEQ = ["1", "2", "3", "4", "5-6", "7-9", "10-14", "15-19", "20-29", "30-49", "50-99", "100-199", "200+"]
UNDER60, UNDER20 = 14, 9
MIN_PAIRS = 20
pd.set_option("display.width", 190, "display.max_columns", 40, "display.max_rows", 200)


def h(title):
    print(f"\n{'=' * 8} {title}", flush=True)


shards = [np.load(ROOT / "layer_shards" / f"target_layer_{t:02d}.npz") for t in L]
ids = shards[0]["protein_ids"].astype(str)
fold = shards[0]["outer_fold"]
moments = np.stack([s["moments"] for s in shards])                 # (target, source, protein, distance, 3)
targets = shards[0]["targets"]
keep_all, keep = np.ones(len(ids), bool), ids != OUT
print("proteins", len(ids), "pairs", int(targets[..., 0].sum()), "outlier present", OUT in ids)


def stats(mask, bins):
    """Whole-cell sums of shape (target, source): n, y, yy, p, pp, py."""
    m = moments[:, :, mask][:, :, :, bins].sum(axis=(2, 3))
    t = targets[mask][:, bins].sum(axis=(0, 1))
    return t, m


def pooled(mask, bins=slice(None)):
    t, m = stats(mask, bins)
    n, sy, syy = t
    sp, spp, spy = m[..., 0], m[..., 1], m[..., 2]
    vy = syy - sy * sy / n
    vp = spp - sp * sp / n
    cov = spy - sp * sy / n
    return {"r": cov / np.sqrt(vp * vy), "gain": cov / vy, "amp": np.sqrt(vp / vy),
            "r2": 1 - (spp - 2 * spy + syy) / vy, "bias": (sp - sy) / n}


def within_protein(mask, bins=slice(None)):
    """Correlation and gain after removing each protein's own means."""
    m = moments[:, :, mask][:, :, :, bins].sum(axis=3)
    t = targets[mask][:, bins].sum(axis=1)
    n, sy, syy = t[..., 0], t[..., 1], t[..., 2]
    sp, spp, spy = m[..., 0], m[..., 1], m[..., 2]
    vy = (syy - sy * sy / n).sum()
    vp = (spp - sp * sp / n).sum(axis=2)
    cov = (spy - sp * sy / n).sum(axis=2)
    return {"r": cov / np.sqrt(vp * vy), "gain": cov / vy, "amp": np.sqrt(vp / vy)}


def crossfit(mask):
    """Affine recalibration fit on other folds' proteins, protein-equal, then applied out of fold."""
    m = moments[:, :, :, :, :].sum(axis=3)                        # (target, source, protein, 3)
    t = targets.sum(axis=1)
    n, sy, syy = t[:, 0], t[:, 1], t[:, 2]
    sp, spp, spy = m[..., 0], m[..., 1], m[..., 2]
    sse = np.zeros(sp.shape)
    a_all, b_all = np.zeros((48, 48, 5)), np.zeros((48, 48, 5))
    for f in range(5):
        train = mask & (fold != f)
        w = 1.0 / n[train]
        W = train.sum()
        wp, wy = (sp[:, :, train] * w).sum(2), (sy[train] * w).sum()
        wpp, wpy = (spp[:, :, train] * w).sum(2), (spy[:, :, train] * w).sum(2)
        b = (wpy - wp * wy / W) / (wpp - wp * wp / W)
        a = (wy - b * wp) / W
        a_all[:, :, f], b_all[:, :, f] = a, b
        test = fold == f
        sse[:, :, test] = (syy[test] - 2 * a[..., None] * sy[test] - 2 * b[..., None] * spy[:, :, test]
                           + n[test] * a[..., None] ** 2 + 2 * (a * b)[..., None] * sp[:, :, test]
                           + b[..., None] ** 2 * spp[:, :, test])
    raw = spp - 2 * spy + syy
    tss_p = syy - sy * sy / n
    total = syy[mask].sum() - sy[mask].sum() ** 2 / n[mask].sum()
    return {"r2": 1 - raw[:, :, mask].sum(2) / total, "r2_recal": 1 - sse[:, :, mask].sum(2) / total,
            "med_protein": np.median(1 - raw[:, :, mask] / tss_p[mask], axis=2),
            "med_protein_recal": np.median(1 - sse[:, :, mask] / tss_p[mask], axis=2),
            "b": b_all.mean(2), "a": a_all.mean(2), "b_fold_sd": b_all.std(2)}


h("1. GAIN VERSUS DIRECTION. Each matrix is indexed [target, source]")
P = pooled(keep)
PA = pooled(keep_all)
U = pooled(keep, np.arange(UNDER60))
WP = within_protein(keep)
WU = within_protein(keep, np.arange(UNDER60))
C = crossfit(keep)
CA = crossfit(keep_all)
print("control, native cells, excl 7stz_C: recal slope b, median over layers",
      round(float(np.median(np.diag(C["b"]))), 3), "range", np.round([np.diag(C["b"]).min(), np.diag(C["b"]).max()], 3),
      "| mean R2 change from recalibration", round(float(np.mean(np.diag(C["r2_recal"]) - np.diag(C["r2"]))), 4))
print("largest fold SD of recal slope within layers 16-47:", round(float(C["b_fold_sd"][16:, 16:].max()), 3))


def split(res):
    """Loss = native R2 - transfer R2, split into a gain part and a direction part."""
    r2, recal = res["r2"], res["r2_recal"]
    nat, nat_recal = np.diag(r2)[:, None], np.diag(recal)[:, None]
    direction = nat_recal - recal
    gain = (recal - r2) - (nat_recal - nat)
    return nat - r2, gain, direction


matrix = pd.read_csv(PRODUCTION / "transfer_matrix.csv")
produced = lambda col: matrix.pivot(index="target_layer", columns="source_layer", values=col).to_numpy(float)
for name, ours, theirs in (("pooled R2 excl 7stz_C", C["r2"], produced("pooled_r2_excl_7stz_C")),
                           ("pooled R2 all", CA["r2"], produced("pooled_r2")),
                           ("median protein R2", CA["med_protein"], produced("median_protein_r2"))):
    worst = float(np.abs(ours - theirs).max())
    print(f"gate {name}: worst absolute difference from production {worst:.2e}")
    if worst > 1e-7:
        raise ValueError(f"{name} does not reproduce the production table")
loss, gain, direction = split(C)
med_loss, med_gain, med_dir = split({"r2": C["med_protein"], "r2_recal": C["med_protein_recal"]})


def describe(name, cells):
    t = np.array([c[0] for c in cells])
    s = np.array([c[1] for c in cells])
    l, g, d = loss[t, s], gain[t, s], direction[t, s]
    ml, mg, md = med_loss[t, s], med_gain[t, s], med_dir[t, s]
    b = C["b"][t, s]
    print(f"{name:34s} n={len(cells):4d} | pooled loss {np.median(l):6.3f} gain {np.median(g):6.3f} dir {np.median(d):6.3f} share_gain {100 * g.sum() / l.sum():5.1f}%"
          f" | protein loss {np.median(ml):6.3f} gain {np.median(mg):6.3f} dir {np.median(md):6.3f} share {100 * mg.sum() / ml.sum():5.1f}% | recal b {np.median(b):5.2f} r {np.median(P['r'][t, s]):.3f} amp {np.median(P['amp'][t, s]):.2f}")


print("loss = native R2 - transfer R2; gain = recovered by one global affine map, beyond native; dir = not recoverable by that map. Excl 7stz_C.")
late = range(16, 48)
for k in (1, 2, 3, 5, 8, 12, 16, 20, 25, 31):
    describe(f"16-47, early on later, offset {k}", [(s + k, s) for s in range(16, 48 - k)])
    describe(f"16-47, later on early, offset {k}", [(s, s + k) for s in range(16, 48 - k)])
describe("late on early (s>=16, t<=13)", [(t, s) for t in range(14) for s in late])
describe("early on late (s<=12, t>=16)", [(t, s) for t in late for s in range(13)])
describe("within 0-13 off-diagonal", [(t, s) for t in range(14) for s in range(14) if s != t])
for s, t in ((22, 47), (47, 22), (30, 47), (47, 30), (16, 47), (47, 16), (44, 47), (47, 44), (33, 35), (35, 33), (12, 13), (13, 12), (0, 47), (47, 0)):
    print(f"cell s={s:2d} t={t:2d}: R2 {C['r2'][t, s]:6.3f} recal {C['r2_recal'][t, s]:6.3f} native {C['r2'][t, t]:6.3f} | r {P['r'][t, s]:.3f} amp {P['amp'][t, s]:.3f} gain-slope {P['gain'][t, s]:.3f} recal b {C['b'][t, s]:.3f}"
          f" | under60 r {U['r'][t, s]:.3f} amp {U['amp'][t, s]:.3f} | within-protein r {WP['r'][t, s]:.3f} amp {WP['amp'][t, s]:.3f} u60 r {WU['r'][t, s]:.3f}")


def conditional(mask):
    """Share of prediction variance fixed by the distance bin, and mean prediction by bin."""
    t = targets[mask].sum(axis=0)
    m = moments[:, :, mask].sum(axis=2)
    n_bin, sp_bin = t[:, 0], m[..., 0]
    used = n_bin > 0
    n, sp, spp = n_bin.sum(), sp_bin.sum(axis=2), m[..., 1].sum(axis=2)
    between = (sp_bin[:, :, used] ** 2 / n_bin[used]).sum(axis=2) - sp * sp / n
    with np.errstate(invalid="ignore", divide="ignore"):
        return between / (spp - sp * sp / n), sp_bin / n_bin, t[:, 1] / n_bin, n_bin


eta2, mean_p, mean_y, n_bin = conditional(keep)
print("\nIs the leftover curvature or scatter? r2 is the linear share of prediction variance tied to distance; eta2 uses all 17 distance bins.")
for s, t in ((22, 22), (47, 47), (22, 47), (47, 22), (30, 47), (47, 30), (0, 47), (47, 0)):
    print(f"cell s={s:2d} t={t:2d}: r2 {P['r'][t, s] ** 2:.3f} eta2 {eta2[t, s]:.3f} curvature share of unexplained {(eta2[t, s] - P['r'][t, s] ** 2) / (1 - P['r'][t, s] ** 2):.3f}")
print("mean prediction by distance bin; bins with 200+ pairs")
use = n_bin >= 200
print(pd.DataFrame({"pairs": n_bin[use].astype(int), "mean_y": mean_y[use], "native47": mean_p[47, 47, use], "22on47": mean_p[47, 22, use], "native22": mean_p[22, 22, use], "47on22": mean_p[22, 47, use], "0on47": mean_p[47, 0, use]},
                   index=np.array(DIST)[use]).round(1).to_string())
print("native amp (SD prediction / SD target) by layer:", {int(l): round(float(P["amp"][l, l]), 3) for l in L[::6]}, "L47", round(float(P["amp"][47, 47]), 3))
print("native r by layer:", {int(l): round(float(P["r"][l, l]), 3) for l in L[::6]}, "L47", round(float(P["r"][47, 47]), 3))
print("with 7stz_C included, share of pooled loss from gain, 22 on 47 and 47 on 22:",
      [round(float((CA["r2_recal"][t, s] - CA["r2"][t, s]) / max(1e-9, CA["r2"][t, t] - CA["r2"][t, s])), 3) for s, t in ((22, 47), (47, 22))])
frame = pd.DataFrame({
    "target_layer": np.repeat(L, 48), "source_layer": np.tile(L, 48), "r2": C["r2"].ravel(), "r2_recal": C["r2_recal"].ravel(),
    "pearson_r": P["r"].ravel(), "amp": P["amp"].ravel(), "recal_slope": C["b"].ravel(), "recal_intercept": C["a"].ravel(),
    "loss": loss.ravel(), "gain_part": gain.ravel(), "direction_part": direction.ravel(),
    "within_protein_r": WP["r"].ravel(), "within_protein_amp": WP["amp"].ravel(),
    "median_protein_r2": C["med_protein"].ravel(), "median_protein_r2_recal": C["med_protein_recal"].ravel(),
})
frame.to_csv(ROOT / "gain_direction_by_cell_excl_7stz_C.csv", index=False)

h("2. WHAT CHANGES ACROSS THE BOUNDARIES")
print("best-source view: for each target, the best non-native source after recalibration, median protein R2")
off = C["med_protein_recal"].copy()
np.fill_diagonal(off, -9)
print({int(t): (int(off[t].argmax()), round(float(off[t].max()), 3), round(float(C["med_protein_recal"][t, t]), 3)) for t in L[::4]})
print("47 probe on layer t, recal R2 / r:", {int(t): (round(float(C["r2_recal"][t, 47]), 3), round(float(P["r"][t, 47]), 3)) for t in (0, 8, 13, 16, 20, 24, 28, 32, 34, 35, 36, 38, 40, 42, 44, 46)})
print("layer s probe on 47, recal R2 / r:", {int(s): (round(float(C["r2_recal"][47, s]), 3), round(float(P["r"][47, s]), 3)) for s in (0, 8, 13, 16, 20, 24, 28, 32, 34, 35, 36, 38, 40, 42, 44, 46)})

h("3. CHAIN POSITION AGAINST DISTANCE AND SEQUENCE MIX, excl 7stz_C. Reference mix: bins 5-14")
pd_stats = np.stack([s["pos_dist"] for s in shards])               # (target, source, fold, position, stratum, 3)
ps_stats = np.stack([s["pos_seq"] for s in shards])
pd_n, ps_n = shards[0]["pos_dist_n"], shards[0]["pos_seq_n"]
out_d, out_s = np.stack([s["out_dist"] for s in shards]), np.stack([s["out_seq"] for s in shards])
out_dn, out_sn = shards[0]["out_dist_n"], shards[0]["out_seq_n"]
MID = np.arange(5, 15)


def standardized(values, counts):
    """values (position, stratum): rate per pair, reweighted to the mid-chain stratum mix."""
    mix = counts[MID].sum(axis=0).astype(float)
    mix /= mix.sum()
    out, coverage = np.full(values.shape[0], np.nan), np.zeros(values.shape[0])
    for b in range(values.shape[0]):
        use = counts[b] >= MIN_PAIRS
        out[b] = (values[b][use] / counts[b][use] * mix[use]).sum() / mix[use].sum()
        coverage[b] = mix[use].sum()
    return out, coverage


def chain_row(label, A, counts, k):
    raw = A[..., k].sum(axis=1) / counts.sum(axis=1)
    std, coverage = standardized(A[..., k], counts)
    return raw, std, coverage


def mid_rate(A, counts, k=1):
    """Rate pooled over the reference bins; it equals their standardized rate by construction."""
    return A[MID][..., k].sum() / counts[MID].sum()


rows = []
for s, t in [(l, l) for l in (0, 8, 16, 22, 33, 43, 47)] + [(22, 47), (47, 22), (0, 47), (47, 0)]:
    Ad = pd_stats[t, s].sum(axis=0) - out_d[t, s]
    Nd = pd_n.sum(axis=0) - out_dn
    As = ps_stats[t, s].sum(axis=0) - out_s[t, s]
    Ns = ps_n.sum(axis=0) - out_sn
    raw, std, cov = chain_row("d", Ad, Nd, 1)
    _, stds, covs = chain_row("s", As, Ns, 1)
    mid = mid_rate(Ad, Nd)
    srd, sst, _ = chain_row("d", Ad, Nd, 0)
    rows.append((f"{s}>{t}", round(mid, 2), round(raw[0] / mid, 2), round(std[0] / mid, 2), round(stds[0] / mid, 2), round(raw[19] / mid, 2), round(std[19] / mid, 2), round(stds[19] / mid, 2),
                 round(raw[1] / mid, 2), round(std[1] / mid, 2), round(srd[0], 2), round(sst[0], 2), round(srd[19], 2), round(sst[19], 2), round(float(cov[0]), 3), round(float(cov[19]), 3)))
print(pd.DataFrame(rows, columns=["cell", "midMAE", "Nraw", "Nstd_d", "Nstd_seq", "Craw", "Cstd_d", "Cstd_seq", "N2raw", "N2std_d", "Nbias", "Nbias_std", "Cbias", "Cbias_std", "cover_N", "cover_C"]).to_string(index=False))
Nd = pd_n.sum(axis=0) - out_dn
share = Nd / Nd.sum(axis=1, keepdims=True)
print("share of pairs by distance group: bin0, bin19, mid")
groups = {"<8": range(0, 5), "8-20": range(5, 9), "20-40": range(9, 12), "40-60": range(12, 14), "60+": range(14, 17)}
print(pd.DataFrame({name: [share[0, list(g)].sum(), share[19, list(g)].sum(), (Nd[MID][:, list(g)].sum() / Nd[MID].sum())] for name, g in groups.items()}, index=["N end", "C end", "mid"]).round(3).to_string())
print("MAE by distance group, native 47, N end, C end, mid")
A = pd_stats[47, 47].sum(axis=0) - out_d[47, 47]
print(pd.DataFrame({name: [A[b][list(g), 1].sum() / Nd[b][list(g)].sum() for b in (0, 19)] + [A[MID][:, list(g), 1].sum() / Nd[MID][:, list(g)].sum()] for name, g in groups.items()}, index=["N end", "C end", "mid"]).round(2).to_string())
raw47, std47, _ = chain_row("d", A, Nd, 1)
print("native 47 profile, raw:", np.round(raw47, 2).tolist())
print("native 47 profile, distance-standardized:", np.round(std47, 2).tolist())
A = pd_stats[22, 22].sum(axis=0) - out_d[22, 22]
raw22, std22, _ = chain_row("d", A, Nd, 1)
print("native 22 profile, raw:", np.round(raw22, 2).tolist())
print("native 22 profile, distance-standardized:", np.round(std22, 2).tolist())
print("by fold, distance-standardized N and C ratios, native 47 and 22 (fold 3 excludes 7stz_C):")
for t in (22, 47):
    out = []
    for f in range(5):
        Af, Nf = pd_stats[t, t, f].copy(), pd_n[f].copy()
        if f == int(fold[ids == OUT][0]):
            Af, Nf = Af - out_d[t, t], Nf - out_dn
        raw, std, _ = chain_row("d", Af, Nf, 1)
        mid = mid_rate(Af, Nf)
        out.append((round(float(std[0] / mid), 2), round(float(std[19] / mid), 2)))
    print(t, out)

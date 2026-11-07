#!/usr/bin/env python3
"""Compare observed triangle violations with the error models, and with every earlier result."""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, CLUSTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from followup_paths import ABLATION_CSV_DIR, GAIN_ROOT, PRODUCER, TRANSFER_ROOT, TRIANGLE_ROOT  # noqa: E402

ROOT = TRIANGLE_ROOT
R = TRANSFER_ROOT
G = GAIN_ROOT
L, NAMED, SHOW = np.arange(48), (8, 13, 35, 44), (0, 8, 12, 13, 17, 22, 34, 35, 42, 43, 44, 47)
SURROGATES, OUT = 5000, "7stz_C"
RNG = np.random.default_rng(0)
pd.set_option("display.width", 200, "display.max_columns", 40, "display.max_rows", 200)


def h(title):
    print(f"\n{'=' * 8} {title}", flush=True)


def surrogate_p(x, y):
    """Spearman with a p from spectrum-preserving surrogates of x."""
    ok = ~(np.isnan(x) | np.isnan(y))
    rx, ry = rankdata(x[ok]), rankdata(y[ok])
    observed = np.corrcoef(rx, ry)[0, 1]
    spectrum = np.fft.rfft(rx - rx.mean())
    phases = RNG.uniform(0, 2 * np.pi, (SURROGATES, spectrum.size))
    phases[:, 0] = 0
    if rx.size % 2 == 0:
        phases[:, -1] = 0
    fake = rankdata(np.fft.irfft(np.abs(spectrum) * np.exp(1j * phases), n=rx.size, axis=1), axis=1)
    fake -= fake.mean(1, keepdims=True)
    rc = ry - ry.mean()
    null = (fake @ rc) / (np.linalg.norm(fake, axis=1) * np.linalg.norm(rc))
    return observed, float((np.sum(np.abs(null) >= abs(observed)) + 1) / (SURROGATES + 1))


shards = [np.load(ROOT / "layer_shards" / f"layer_{t:02d}.npz") for t in L]
ids, models, probes = shards[0]["protein_ids"].astype(str), shards[0]["models"].astype(str), shards[0]["probes"].astype(str)
n_res = shards[0]["n_residues"]
counts = np.stack([s["counts"] for s in shards])                 # (layer, probe, protein, model, rep, 6)
keep = ids != OUT
frac = counts[..., 1] / counts[..., 0]
mean_rate = frac.mean(axis=4)                                    # (layer, probe, protein, model)
curve = mean_rate[:, :, keep].mean(axis=2)                       # (layer, probe, model)
index = {m: k for k, m in enumerate(models)}
print("proteins", len(ids), "exact triangle sets", int(shards[0]["exact"].sum()), "models", list(models))
mc = np.abs(frac[..., 0] - frac[..., 1])[:, :, keep].mean(axis=2)
print("Monte Carlo: mean difference between the two replicates, per protein:", dict(zip(models, np.round(mc.mean(axis=(0, 1)), 5))))

for a, probe in enumerate(probes):
    h(f"1. {probe}: VIOLATION FRACTION BY LAYER, 153 proteins, protein-equal")
    print(pd.DataFrame(curve[:, a][list(SHOW)], index=[f"L{t}" for t in SHOW], columns=models).round(4).to_string())
    obs = curve[:, a, 0]
    rows = []
    for m, name in enumerate(models):
        y = curve[:, a, m]
        step_o, step_m = np.diff(obs), np.diff(y)
        rho, p = surrogate_p(step_m, step_o)
        lvl = np.corrcoef(y, obs)[0, 1]
        row = {"model": name, "mean_minus_obs": (y - obs).mean(), "mean_abs_gap": np.abs(y - obs).mean(), "level_r": lvl, "step_pearson": np.corrcoef(step_m, step_o)[0, 1], "step_spearman": rho, "p": p}
        for b in NAMED[1:]:
            row[f"drop{b}"] = step_m[b - 1] / step_o[b - 1]
        row["rise_0_11"] = (y[11] - y[0]) / (obs[11] - obs[0])
        row["rise_22_34"] = (y[34] - y[22]) / (obs[34] - obs[22])
        rows.append(row)
    print("\ndrop and rise columns: model change as a share of the observed change. 1 is full recovery.")
    print(pd.DataFrame(rows).round(3).to_string(index=False))
    print("\nacross proteins within a layer, Spearman with observed:")
    print(pd.DataFrame({name: [spearmanr(mean_rate[t, a, keep, m], mean_rate[t, a, keep, 0]).statistic for t in (13, 22, 34, 47)] for m, name in enumerate(models)}, index=["L13", "L22", "L34", "L47"]).round(3).to_string())
    print("\nabove 5 A (share of triangles):")
    big = (counts[..., 3] + counts[..., 4]).mean(axis=4) / counts[..., 0].mean(axis=4)
    print(pd.DataFrame(big[:, a][:, keep].mean(axis=1)[list((0, 13, 22, 34, 35, 44, 47))], index=["L0", "L13", "L22", "L34", "L35", "L44", "L47"], columns=models).round(4).to_string())

h("2. WHICH TRIANGLES: violation rate by relative slack in the final structure, 153 proteins")
sv = np.stack([s["slack_viol"] for s in shards])
sn = shards[0]["slack_n"]
labels = ["<0.02", "0.02-0.1", "0.1-0.3", ">=0.3"]
print("share of triangles per slack bin:", dict(zip(labels, np.round((sn[keep] / sn[keep].sum(1, keepdims=True)).mean(0), 4))))
for a, probe in enumerate(probes):
    for t in (0, 22, 34, 47):
        rate = (sv[t, a][keep] / np.maximum(sn[keep][:, None], 1)).mean(axis=0)
        print(probe[:11], f"L{t}", pd.DataFrame(rate[[0, 4, 5, 6]], index=[models[k] for k in (0, 4, 5, 6)], columns=labels).round(3).to_string().replace("\n", " | "))

h("3. WHY: error correlation across edges, and the margin (longest edge error minus the other two)")
edges = np.stack([s["edges"] for s in shards])[:, :, keep].mean(axis=2)     # (layer, probe, 7)
for a, probe in enumerate(probes):
    print(probe)
    print(pd.DataFrame(edges[:, a][list(SHOW)], index=[f"L{t}" for t in SHOW], columns=["z_ab", "z_ac", "z_bc", "margin_mean", "margin_sd", "perm_mean", "perm_sd"]).round(3).to_string())
    sd_step = np.diff(edges[:, a, 4])
    print("margin SD, step vs observed violation step: spearman", np.round(surrogate_p(sd_step, np.diff(curve[:, a, 0])), 3), "| margin mean:", np.round(surrogate_p(np.diff(edges[:, a, 3]), np.diff(curve[:, a, 0])), 3))

h("4. CHAIN POSITION: share of violating triangle corners against share of all corners, 153 proteins")
pv = np.stack([s["position_viol"] for s in shards])
pn = shards[0]["position_n"]
for a, probe in enumerate(probes):
    for t in (0, 22, 47):
        out = []
        for slot, name in ((0, "observed"), (1, "residuals_permuted")):
            share = pv[t, a][keep, slot] / np.maximum(pv[t, a][keep, slot].sum(1, keepdims=True), 1)
            base = pn[keep] / pn[keep].sum(1, keepdims=True)
            out.append((name, np.round((share / base).mean(0)[[0, 1, 2, 10, 17, 18, 19]], 2).tolist()))
        print(probe[:11], f"L{t} enrichment at bins 0,1,2,10,17,18,19:", out)

h("5. ARE THE DROPS REAL ACROSS PROTEINS? 552 proteins, stored geometry, four probe families")
families = ("shared_ols_base256", "sequence_gated_ols_base256", "per_protein_ols_base256", "shared_xgboost_base256_fixed_s4")
violation = {}
for family in families:
    table = pd.read_csv(PRODUCER / family / "geometry_metrics.csv", usecols=["layer", "protein_id", "triangle_violation_fraction"], dtype={"protein_id": str})
    wide = table.pivot(index="protein_id", columns="layer", values="triangle_violation_fraction")
    violation[family] = wide
    steps = wide.diff(axis=1).iloc[:, 1:]
    falling = (steps < 0).mean()
    print(f"{family:34s} proteins {len(wide)} | share falling, median block {falling.median():.2f} | " + " ".join(f"b{b} {falling[b]:.2f} ({steps[b].mean():+.4f})" for b in (8, 13, 17, 35, 43, 44)) + " | top falling blocks " + str(falling.sort_values(ascending=False).head(4).round(2).to_dict()))
base = violation[families[0]].diff(axis=1).iloc[:, 1:].mean().to_numpy()
print("family step correlation with shared OLS, Spearman and p:", {f[:12]: tuple(np.round(surrogate_p(violation[f].diff(axis=1).iloc[:, 1:].mean().to_numpy(), base), 3)) for f in families[1:]})

h("6. AGAINST EVERY EARLIER RESULT, by block. Step in violation fraction is the second series.")
pp = pd.read_csv(PRODUCER / "shared_ols_base256/per_protein.csv", usecols=["layer", "protein_id", "r2"], dtype={"protein_id": str})
r2 = pp.groupby("layer").r2.mean().reindex(L).to_numpy()
cell = pd.read_csv(G / "gain_direction_by_cell_excl_7stz_C.csv")
diag = cell[cell.source_layer.eq(cell.target_layer)].set_index("target_layer").sort_index()
amp = diag.amp.to_numpy()
M = pd.read_csv(R / "transfer_matrix.csv")
maex = M.pivot(index="source_layer", columns="target_layer", values="pooled_mae_excl_7stz_C").to_numpy(float)
cost = np.array([((maex[t, t + 1] - maex[t + 1, t + 1]) + (maex[t + 1, t] - maex[t, t])) / 2 for t in L[:-1]])
W = np.stack([[np.load(PRODUCER / f"shared_ols_base256/model_artifacts/coefficients/fold_{f}_layer_{l:02d}.npz")["coefficient_standardized"].astype(float) for f in range(5)] for l in L])
cos = np.mean([(W[:, f] @ W[:, f].T) / np.outer(np.linalg.norm(W[:, f], axis=1), np.linalg.norm(W[:, f], axis=1)) for f in range(5)], axis=0)
rotation = np.array([1 - cos[t, t + 1] for t in L[:-1]])
sm = pd.read_csv(ABLATION_CSV_DIR / "structure_module_tm_scores.csv", dtype={"protein": str})
sm = sm[sm.protein.isin(ids)].set_index("protein")[[f"L{t}" for t in L]].apply(pd.to_numeric, errors="coerce")
sm_mean = sm.mean().to_numpy()
geom = pd.read_csv(PRODUCER / "shared_ols_base256/geometry_summary.csv")
mds = geom[geom.geometry_metric.eq("tm_score")].set_index("layer")["mean"].reindex(L).to_numpy()
wide = violation[families[0]]
v552, vstep = wide.mean().reindex(L).to_numpy(), wide.mean().reindex(L).diff().to_numpy()[1:]
vobs = curve[:, 0, 0]
for name, series in (("native R2 step (552)", np.diff(r2)), ("MDS TM step (552)", np.diff(mds)), ("native amplitude step (CAMEO)", np.diff(amp)), ("weight rotation", rotation), ("adjacent transfer cost", cost), ("structure module TM step (CAMEO)", np.diff(sm_mean))):
    print(f"{name:34s} spearman, p with violation step:", tuple(np.round(surrogate_p(series, vstep), 3)))
print("levels: violation vs native R2, spearman", round(spearmanr(v552, r2).statistic, 3), "| vs amplitude", round(spearmanr(vobs, amp).statistic, 3), "| vs SM TM", round(spearmanr(vobs, sm_mean).statistic, 3))
print("within layers 16-47: violation vs R2", round(spearmanr(v552[16:], r2[16:]).statistic, 3), "| vs amplitude", round(spearmanr(vobs[16:], amp[16:]).statistic, 3))
A = np.column_stack([np.ones(47), np.diff(r2), np.diff(amp)])
fit, *_ = np.linalg.lstsq(A, np.diff(vobs), rcond=None)
pred = A @ fit
print("violation step ~ R2 step + amplitude step:", np.round(fit, 3), "r2", round(1 - ((np.diff(vobs) - pred) ** 2).sum() / ((np.diff(vobs) - np.diff(vobs).mean()) ** 2).sum(), 3))
print("by block:")
print(pd.DataFrame({"dViolation_552": vstep, "dViolation_cameo": np.diff(vobs), "dR2": np.diff(r2), "dAmp": np.diff(amp), "rotation": rotation, "cost": cost, "dSM": np.diff(sm_mean)}, index=L[1:]).loc[[b for b in (8, 13, 15, 16, 17, 19, 34, 35, 36, 39, 40, 42, 43, 44, 46, 47)]].round(4).to_string())

h("7. PROTEIN BY PROTEIN: violation drop at block 35 against the structure module")
c = wide.reindex(sm.index)
lengths = pd.Series(n_res, index=ids).reindex(sm.index).astype(float)
for b, window in ((35, (34, 36)), (35, (34, 40)), (44, (43, 44)), (13, (12, 13))):
    drop = (c[b] - c[b - 1]).to_numpy()
    gain = (sm[f"L{window[1]}"] - sm[f"L{window[0]}"]).to_numpy()
    print(f"block {b} violation step vs SM TM gain L{window[0]}->L{window[1]}: spearman", tuple(np.round(spearmanr(drop, gain), 3)))
for t in (35, 36, 38, 40, 44):
    print(f"L{t}: SM TM vs violation fraction, spearman {spearmanr(sm[f'L{t}'], c[t]).statistic:.3f} | violation vs length {spearmanr(c[t], lengths).statistic:.3f}")

pd.DataFrame([{"probe": probe, "layer": int(t), "model": name, "violation_fraction": float(curve[t, a, m])}
              for a, probe in enumerate(probes) for m, name in enumerate(models) for t in L]
             ).to_csv(ROOT / "triangle_models_by_layer.csv", index=False, float_format="%.6f")
margin_columns = ["edge_correlation_short_mid", "edge_correlation_short_long", "edge_correlation_mid_long", "margin_mean",
                  "margin_sd", "shuffled_margin_mean", "shuffled_margin_sd"]
pd.DataFrame([{"probe": probe, "layer": int(t), **dict(zip(margin_columns, map(float, edges[t, a])))}
              for a, probe in enumerate(probes) for t in L]
             ).to_csv(ROOT / "triangle_margin_by_layer.csv", index=False, float_format="%.6f")

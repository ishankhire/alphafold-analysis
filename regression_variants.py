"""
Regression variant analyses on layer 47 pair representations.

Analysis 1 — Top-k feature selection:
  Train the full 256-dim linear regression on layer 47, then re-train using
  only the k features with the largest |coefficient|, for k = 10, 20, 32, 64,
  128, 256. Plot R² vs k.

Analysis 1b — Random-k feature baseline:
  Same k values, but select k channels uniformly at random (20 trials each).
  Plotted alongside top-k to test whether the top-k channels are genuinely
  special or any random subset encodes distance equally well.

Analysis 2 — Upper vs lower triangle of feature vectors:
  Train separate regressions using only pair_block[i,j,:] (upper, 128-dim)
  vs only pair_block[j,i,:] (lower, 128-dim). Report R² and show scatterplots.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "visualizations"), exist_ok=True)

PAIR_OFFSET = 4
C = 128  # channels per direction
LAYER = 47  # final layer (0-indexed)

# ---------------------------------------------------------------------------
# 1. Load distance matrix and build target vector
# ---------------------------------------------------------------------------
dist_csv = os.path.join(BASE_DIR, "residue_distances.csv")
raw = np.genfromtxt(dist_csv, delimiter=",", skip_header=1)
n = raw.shape[0]  # 276 resolved residues

n_pairs = n * (n - 1) // 2
y = np.empty(n_pairs, dtype=np.float64)
pair_i = np.empty(n_pairs, dtype=np.int32)
pair_j = np.empty(n_pairs, dtype=np.int32)
idx = 0
for i in range(n):
    for j in range(i + 1, n):
        y[idx] = raw[i, j]
        pair_i[idx] = i
        pair_j[idx] = j
        idx += 1

print(f"Distance matrix: {n}x{n} | Pairs: {n_pairs}")

# Fixed train/test split (same as probe_layer_regression.py)
all_idx = np.arange(n_pairs)
idx_train, idx_test = train_test_split(all_idx, test_size=0.2, random_state=42)
y_train = y[idx_train]
y_test = y[idx_test]

# ---------------------------------------------------------------------------
# 2. Load layer 47 pair block and build feature matrix
# ---------------------------------------------------------------------------
pb_path = os.path.join(BASE_DIR, "7b3a_A", f"7b3a_A_pair_block_{LAYER}.npy")
pair_block = np.load(pb_path)

pi = pair_i + PAIR_OFFSET
pj = pair_j + PAIR_OFFSET
X_upper = pair_block[pi, pj, :]  # (n_pairs, 128) — pair_block[i,j]
X_lower = pair_block[pj, pi, :]  # (n_pairs, 128) — pair_block[j,i]
X_full = np.concatenate([X_upper, X_lower], axis=1)  # (n_pairs, 256)

del pair_block  # free memory

# =========================================================================
# ANALYSIS 1: Top-k feature selection
# =========================================================================
print("\n" + "=" * 60)
print("ANALYSIS 1: Top-k Feature Selection (Layer 47)")
print("=" * 60)

# First, fit the full model to get coefficients
model_full = LinearRegression()
model_full.fit(X_full[idx_train], y_train)
full_coefs = model_full.coef_

# Rank features by |coefficient|
ranked_features = np.argsort(np.abs(full_coefs))[::-1]

k_values = [10, 20, 32, 64, 128, 256]
topk_r2 = []

for k in k_values:
    selected = ranked_features[:k]
    X_sel_train = X_full[idx_train][:, selected]
    X_sel_test = X_full[idx_test][:, selected]

    model = LinearRegression()
    model.fit(X_sel_train, y_train)
    y_pred = model.predict(X_sel_test)
    r2 = r2_score(y_test, y_pred)
    topk_r2.append(r2)
    print(f"  k = {k:>3d}  →  R² = {r2:.6f}")

# =========================================================================
# ANALYSIS 1b: Random-k feature baseline
# =========================================================================
print("\n" + "=" * 60)
print("ANALYSIS 1b: Random-k Feature Baseline (Layer 47)")
print("=" * 60)

N_TRIALS = 20
rng = np.random.RandomState(123)
# k=256 is trivially the full model, so skip random trials for it
random_r2_mean = []
random_r2_std = []
random_r2_all = []  # store per-trial results

for k in k_values:
    if k >= 256:
        # All features selected — no randomness
        random_r2_mean.append(topk_r2[-1])
        random_r2_std.append(0.0)
        random_r2_all.append([topk_r2[-1]] * N_TRIALS)
        print(f"  k = {k:>3d}  →  R² = {topk_r2[-1]:.6f} (all features, no randomness)")
        continue

    trial_r2s = []
    for t in range(N_TRIALS):
        selected = rng.choice(256, size=k, replace=False)
        X_sel_train = X_full[idx_train][:, selected]
        X_sel_test = X_full[idx_test][:, selected]

        model = LinearRegression()
        model.fit(X_sel_train, y_train)
        y_pred = model.predict(X_sel_test)
        trial_r2s.append(r2_score(y_test, y_pred))

    mean_r2 = np.mean(trial_r2s)
    std_r2 = np.std(trial_r2s)
    random_r2_mean.append(mean_r2)
    random_r2_std.append(std_r2)
    random_r2_all.append(trial_r2s)
    print(f"  k = {k:>3d}  →  R² = {mean_r2:.6f} ± {std_r2:.6f}")

random_r2_mean = np.array(random_r2_mean)
random_r2_std = np.array(random_r2_std)

# Plot: Top-k vs Random-k comparison
fig, ax = plt.subplots(figsize=(9, 5.5))
ax.plot(k_values, topk_r2, "o-", markersize=7, linewidth=2,
        color="steelblue", label="Top-k (by |coefficient|)", zorder=3)
ax.plot(k_values, random_r2_mean, "s-", markersize=7, linewidth=2,
        color="darkorange", label=f"Random-k (mean of {N_TRIALS} trials)", zorder=3)
ax.fill_between(k_values,
                random_r2_mean - random_r2_std,
                random_r2_mean + random_r2_std,
                color="darkorange", alpha=0.2, label="Random-k ±1 std")
for k, r2_top, r2_rand in zip(k_values, topk_r2, random_r2_mean):
    ax.annotate(f"{r2_top:.4f}", (k, r2_top), textcoords="offset points",
                xytext=(0, 10), ha="center", fontsize=7, color="steelblue")
    ax.annotate(f"{r2_rand:.4f}", (k, r2_rand), textcoords="offset points",
                xytext=(0, -14), ha="center", fontsize=7, color="darkorange")
ax.set_xlabel("Number of Features (k)")
ax.set_ylabel("Test R²")
ax.set_title("Layer 47: Top-k vs Random-k Feature Selection")
ax.set_xscale("log", base=2)
ax.set_xticks(k_values)
ax.get_xaxis().set_major_formatter(plt.ScalarFormatter())
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.legend(loc="lower right")
ax.grid(True, alpha=0.3)
plt.tight_layout()
topk_path = os.path.join(BASE_DIR, "visualizations", "regression_topk_r2.png")
plt.savefig(topk_path, dpi=150)
plt.close()
print(f"  Plot saved to {topk_path}")

# Save CSV
csv_path = os.path.join(BASE_DIR, "regression_topk_r2.csv")
with open(csv_path, "w") as f:
    f.write("k,topk_r2,random_r2_mean,random_r2_std\n")
    for k, r2_top, r2_rand, r2_std in zip(k_values, topk_r2, random_r2_mean, random_r2_std):
        f.write(f"{k},{r2_top:.6f},{r2_rand:.6f},{r2_std:.6f}\n")
print(f"  CSV saved to {csv_path}")

# =========================================================================
# ANALYSIS 2: Upper vs Lower triangle of feature vectors
# =========================================================================
print("\n" + "=" * 60)
print("ANALYSIS 2: Upper vs Lower Feature Triangle (Layer 47)")
print("=" * 60)

results_tri = {}
for label, X_tri in [("upper", X_upper), ("lower", X_lower)]:
    model = LinearRegression()
    model.fit(X_tri[idx_train], y_train)
    y_pred_train = model.predict(X_tri[idx_train])
    y_pred_test = model.predict(X_tri[idx_test])
    r2_train = r2_score(y_train, y_pred_train)
    r2_test = r2_score(y_test, y_pred_test)
    results_tri[label] = {
        "r2_train": r2_train,
        "r2_test": r2_test,
        "y_pred_test": y_pred_test,
    }
    print(f"  {label:>5s}: Train R² = {r2_train:.6f}  |  Test R² = {r2_test:.6f}")

# Also report full model for reference
y_pred_full = model_full.predict(X_full[idx_test])
r2_full = r2_score(y_test, y_pred_full)
print(f"   full: Train R² = {r2_score(y_train, model_full.predict(X_full[idx_train])):.6f}"
      f"  |  Test R² = {r2_full:.6f}")

# Scatterplots
fig, axes = plt.subplots(1, 2, figsize=(14, 6), sharex=True, sharey=True)

for ax, label in zip(axes, ["upper", "lower"]):
    y_pred = results_tri[label]["y_pred_test"]
    r2 = results_tri[label]["r2_test"]

    ax.scatter(y_test, y_pred, s=0.5, alpha=0.15, color="steelblue", rasterized=True)
    lims = [0, y_test.max() * 1.05]
    ax.plot(lims, lims, "r--", linewidth=1, alpha=0.7)
    ax.set_xlabel("True CA-CA Distance (Å)")
    ax.set_ylabel("Predicted Distance (Å)")
    ax.set_title(f"{label.capitalize()} triangle — pair_block[{'i,j' if label == 'upper' else 'j,i'},:]\n"
                 f"R² = {r2:.4f}")
    ax.set_aspect("equal")
    ax.set_xlim(lims)
    ax.set_ylim(lims)
    ax.grid(True, alpha=0.3)

plt.tight_layout()
tri_path = os.path.join(BASE_DIR, "visualizations", "regression_upper_lower_scatter.png")
plt.savefig(tri_path, dpi=150)
plt.close()
print(f"  Scatterplot saved to {tri_path}")

# =========================================================================
# Console summary
# =========================================================================
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"\nAnalysis 1 — Top-k vs Random-k features (layer 47):")
print(f"  {'k':>5s}  {'Top-k R²':>10s}  {'Random R²':>10s}  {'Gap':>8s}")
for k, r2_top, r2_rand in zip(k_values, topk_r2, random_r2_mean):
    gap = r2_top - r2_rand
    print(f"  {k:>5d}  {r2_top:>10.4f}  {r2_rand:>10.4f}  {gap:>+8.4f}")
print(f"\nAnalysis 2 — Upper vs Lower feature triangle:")
print(f"  Upper (pair_block[i,j,:]):  R² = {results_tri['upper']['r2_test']:.4f}")
print(f"  Lower (pair_block[j,i,:]):  R² = {results_tri['lower']['r2_test']:.4f}")
print(f"  Full  (concatenated):       R² = {r2_full:.4f}")

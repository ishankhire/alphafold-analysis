"""
PCA dimensionality probe: how many principal components of the layer-47 pair
representations are needed to linearly predict CA-CA spatial distance?

Runs linear regression on top-k PCs for k in {1, 2, 3, 5, 10, 20, 32, 64, 128, 256}.
If R² saturates at small k, spatial distance lives in a low-dimensional subspace.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score
from sklearn.decomposition import PCA

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
PROTEIN_DIR = os.path.join(ROOT_DIR, "proteins", "7b3a")
os.makedirs(os.path.join(PROTEIN_DIR, "visualizations"), exist_ok=True)
os.makedirs(os.path.join(PROTEIN_DIR, "csv_files"), exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load distance matrix
# ---------------------------------------------------------------------------
dist_csv = os.path.join(PROTEIN_DIR, "csv_files", "residue_distances.csv")
raw = np.genfromtxt(dist_csv, delimiter=",", skip_header=1)
n = raw.shape[0]  # 276 resolved residues
print(f"Distance matrix: {n} x {n} residues")

PAIR_OFFSET = 4
C = 128

# ---------------------------------------------------------------------------
# 2. Load layer-47 pair block and build features
# ---------------------------------------------------------------------------
pb_path = os.path.join(PROTEIN_DIR, "pair_blocks", "7b3a_A_pair_block_47.npy")
pair_block = np.load(pb_path)
print(f"Pair block shape: {pair_block.shape}")

n_pairs = n * (n - 1) // 2
X = np.empty((n_pairs, 2 * C), dtype=np.float32)
y = np.empty(n_pairs, dtype=np.float64)

idx = 0
for i in range(n):
    for j in range(i + 1, n):
        pi, pj = i + PAIR_OFFSET, j + PAIR_OFFSET
        X[idx, :C] = pair_block[pi, pj, :]
        X[idx, C:] = pair_block[pj, pi, :]
        y[idx] = raw[i, j]
        idx += 1

print(f"Samples: {n_pairs}  |  Features: {2 * C}  |  Distance range: {y.min():.2f} – {y.max():.2f} Å")
del pair_block

# ---------------------------------------------------------------------------
# 3. Fixed train/test split
# ---------------------------------------------------------------------------
all_idx = np.arange(n_pairs)
idx_train, idx_test = train_test_split(all_idx, test_size=0.2, random_state=42)
y_train = y[idx_train]
y_test = y[idx_test]

# ---------------------------------------------------------------------------
# 4. Fit PCA on training features (max components = 256)
# ---------------------------------------------------------------------------
print("\nFitting PCA on training set (256 components)...")
pca = PCA(n_components=256)
pca.fit(X[idx_train])

X_pca_train = pca.transform(X[idx_train])  # (n_train, 256)
X_pca_test = pca.transform(X[idx_test])    # (n_test, 256)

cumvar = np.cumsum(pca.explained_variance_ratio_)
print(f"Variance explained by top-10 PCs: {cumvar[9]:.4f}")
print(f"Variance explained by top-50 PCs: {cumvar[49]:.4f}")
print(f"Variance explained by all 256 PCs: {cumvar[-1]:.4f}")

# ---------------------------------------------------------------------------
# 5. Regression for each k
# ---------------------------------------------------------------------------
ks = [1, 2, 3, 5, 10, 20, 32, 64, 128, 256]
r2_results = []

print(f"\n{'k':>5s}  {'R²':>8s}  {'Cumulative Var':>15s}")
for k in ks:
    model = LinearRegression()
    model.fit(X_pca_train[:, :k], y_train)
    y_pred = model.predict(X_pca_test[:, :k])
    r2 = r2_score(y_test, y_pred)
    r2_results.append(r2)
    print(f"{k:5d}  {r2:8.4f}  {cumvar[k-1]:15.4f}")

# ---------------------------------------------------------------------------
# 6. Save CSV
# ---------------------------------------------------------------------------
csv_path = os.path.join(PROTEIN_DIR, "csv_files", "pca_regression_r2.csv")
with open(csv_path, "w") as f:
    f.write("n_components,r2,cumulative_variance\n")
    for k, r2 in zip(ks, r2_results):
        f.write(f"{k},{r2:.6f},{cumvar[k-1]:.6f}\n")
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# 7. Plot R² vs number of PCs
# ---------------------------------------------------------------------------
fig, ax1 = plt.subplots(figsize=(9, 5))

color_r2 = "steelblue"
color_var = "darkorange"

ax1.semilogx(ks, r2_results, "o-", color=color_r2, markersize=6, label="Test R²")
ax1.set_xlabel("Number of Principal Components (log scale)")
ax1.set_ylabel("Test R²", color=color_r2)
ax1.tick_params(axis="y", labelcolor=color_r2)
ax1.set_ylim(0, 1.05)
ax1.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

ax2 = ax1.twinx()
ax2.semilogx(ks, [cumvar[k - 1] for k in ks], "s--", color=color_var, markersize=5, label="Cumul. Var.")
ax2.set_ylabel("Cumulative Variance Explained", color=color_var)
ax2.tick_params(axis="y", labelcolor=color_var)
ax2.set_ylim(0, 1.05)

# Combine legends
lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right")

ax1.set_title("PCA Dimensionality Probe: Test R² vs Number of PCs (Layer 47)")
ax1.set_xticks(ks)
ax1.set_xticklabels(ks)
plt.tight_layout()
png_path = os.path.join(PROTEIN_DIR, "visualizations", "pca_regression_r2.png")
plt.savefig(png_path, dpi=150)
plt.close()
print(f"Plot saved to {png_path}")

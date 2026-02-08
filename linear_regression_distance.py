"""
Linear regression to predict CA-CA spatial distance from AlphaFold pair representations.

Input: For each residue pair (i, j) where i < j, concatenate pair_block_47[i,j,:]
       and pair_block_47[j,i,:] → 256-dim feature vector.
Output: Spatial distance in Angstroms.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "visualizations"), exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load data
# ---------------------------------------------------------------------------
# Distance matrix: first row is residue labels, remaining rows are distances
dist_csv = os.path.join(BASE_DIR, "residue_distances.csv")
raw = np.genfromtxt(dist_csv, delimiter=",", skip_header=1)
n_residues = raw.shape[0]  # 280
print(f"Distance matrix shape: ({n_residues}, {raw.shape[1]})")

# Last-layer pair representation
pair_path = os.path.join(BASE_DIR, "7b3a_A", "7b3a_A_pair_block_47.npy")
pair_block = np.load(pair_path)
r, _, C = pair_block.shape
print(f"Pair block shape: {pair_block.shape}")

# Alignment: the entity sequence (280 residues) has a 4-residue N-terminal prefix
# (GHMA) not present in the resolved structure (276 residues). CIF residue i
# corresponds to pair block index i + PAIR_OFFSET.
PAIR_OFFSET = 4
n = n_residues  # 276 resolved residues

# ---------------------------------------------------------------------------
# 2. Extract features and targets (upper triangle, i < j)
# ---------------------------------------------------------------------------
n_pairs = n * (n - 1) // 2
X = np.empty((n_pairs, 2 * C), dtype=np.float32)
y = np.empty(n_pairs, dtype=np.float64)
pair_indices = []  # (i, j) in CIF numbering for each sample

idx = 0
for i in range(n):
    for j in range(i + 1, n):
        pi, pj = i + PAIR_OFFSET, j + PAIR_OFFSET
        X[idx, :C] = pair_block[pi, pj, :]
        X[idx, C:] = pair_block[pj, pi, :]
        y[idx] = raw[i, j]
        pair_indices.append((i, j))
        idx += 1

print(f"Samples: {n_pairs}  |  Features per sample: {2 * C}")
print(f"Distance range: {y.min():.2f} – {y.max():.2f} Å")

# ---------------------------------------------------------------------------
# 3. Train / test split
# ---------------------------------------------------------------------------
X_train, X_test, y_train, y_test, idx_train, idx_test = train_test_split(
    X, y, np.arange(n_pairs), test_size=0.2, random_state=42
)

# ---------------------------------------------------------------------------
# 4. Fit linear regression
# ---------------------------------------------------------------------------
model = LinearRegression()
model.fit(X_train, y_train)

y_pred_train = model.predict(X_train)
y_pred_test = model.predict(X_test)

# ---------------------------------------------------------------------------
# 5. Evaluate
# ---------------------------------------------------------------------------
def print_metrics(label, y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    print(f"{label:>10s}  R²={r2:.4f}  MAE={mae:.2f} Å  RMSE={rmse:.2f} Å")

print()
print_metrics("Train", y_train, y_pred_train)
print_metrics("Test", y_test, y_pred_test)

# ---------------------------------------------------------------------------
# 6. Top-10 features by absolute coefficient weight
# ---------------------------------------------------------------------------
coefs = model.coef_
abs_coefs = np.abs(coefs)
top10 = np.argsort(abs_coefs)[::-1][:10]

print("\nTop 10 features by |coefficient|:")
print(f"{'Rank':>4s}  {'Feature':>10s}  {'Coeff':>12s}")
for rank, fi in enumerate(top10, 1):
    half = "upper" if fi < C else "lower"
    ch = fi if fi < C else fi - C
    print(f"{rank:4d}  {half}[{ch:3d}]     {coefs[fi]:12.4f}")

# ---------------------------------------------------------------------------
# 7. Scatter plot (test set)
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(7, 7))
ax.scatter(y_test, y_pred_test, s=1, alpha=0.3)
lims = [0, max(y_test.max(), y_pred_test.max()) * 1.05]
ax.plot(lims, lims, "r--", linewidth=1, label="y = x")
ax.set_xlabel("Actual distance (Å)")
ax.set_ylabel("Predicted distance (Å)")
ax.set_title("Linear Regression: Predicted vs Actual CA-CA Distance (Test Set)")
r2_test = r2_score(y_test, y_pred_test)
ax.legend(title=f"R² = {r2_test:.4f}")
ax.set_xlim(lims)
ax.set_ylim(lims)
ax.set_aspect("equal")
plt.tight_layout()
scatter_path = os.path.join(BASE_DIR, "visualizations", "linear_regression_scatter.png")
plt.savefig(scatter_path, dpi=150)
plt.close()
print(f"\nScatter plot saved to {scatter_path}")

# ---------------------------------------------------------------------------
# 8. Save per-pair results CSV (test set)
# ---------------------------------------------------------------------------
csv_path = os.path.join(BASE_DIR, "linear_regression_results.csv")
with open(csv_path, "w") as f:
    f.write("residue_i,residue_j,actual_distance,predicted_distance\n")
    for k, test_idx in enumerate(idx_test):
        i, j = pair_indices[test_idx]
        f.write(f"{i},{j},{y_test[k]:.4f},{y_pred_test[k]:.4f}\n")

print(f"Results CSV saved to {csv_path}")

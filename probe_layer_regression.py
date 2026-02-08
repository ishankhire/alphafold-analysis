"""
Layer-wise linear regression probe: how well can each layer's pair representations
predict CA-CA spatial distance?

For each of the 48 layers, concatenate pair_block[i,j,:] and pair_block[j,i,:]
(256-dim), train a linear regression, and record test-set R². Uses the same
fixed train/test split (random_state=42) across all layers for fair comparison.
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
os.makedirs(os.path.join(BASE_DIR, "visualizations"), exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load distance matrix
# ---------------------------------------------------------------------------
dist_csv = os.path.join(BASE_DIR, "residue_distances.csv")
raw = np.genfromtxt(dist_csv, delimiter=",", skip_header=1)
n = raw.shape[0]  # 276 resolved residues
print(f"Distance matrix: {n} x {n} residues")

PAIR_OFFSET = 4
C = 128  # channels per pair representation

# ---------------------------------------------------------------------------
# 2. Build target vector y (same across all layers)
# ---------------------------------------------------------------------------
n_pairs = n * (n - 1) // 2
y = np.empty(n_pairs, dtype=np.float64)
idx = 0
for i in range(n):
    for j in range(i + 1, n):
        y[idx] = raw[i, j]
        idx += 1

print(f"Pairs: {n_pairs}  |  Distance range: {y.min():.2f} – {y.max():.2f} Å")

# ---------------------------------------------------------------------------
# 3. Fixed train/test split on indices (same split for every layer)
# ---------------------------------------------------------------------------
all_idx = np.arange(n_pairs)
idx_train, idx_test = train_test_split(all_idx, test_size=0.2, random_state=42)
y_train = y[idx_train]
y_test = y[idx_test]

# Precompute pair (i, j) index arrays to avoid recomputing each loop iteration
pair_i = np.empty(n_pairs, dtype=np.int32)
pair_j = np.empty(n_pairs, dtype=np.int32)
k = 0
for i in range(n):
    for j in range(i + 1, n):
        pair_i[k] = i
        pair_j[k] = j
        k += 1

# ---------------------------------------------------------------------------
# 4. Loop over all 48 layers
# ---------------------------------------------------------------------------
results = []
n_layers = 48
for layer in range(n_layers):
    pb_path = os.path.join(BASE_DIR, "7b3a_A", f"7b3a_A_pair_block_{layer}.npy")
    pair_block = np.load(pb_path)

    # Extract features: concatenate [pair_block[i+4, j+4], pair_block[j+4, i+4]]
    pi = pair_i + PAIR_OFFSET
    pj = pair_j + PAIR_OFFSET
    X = np.concatenate([pair_block[pi, pj, :], pair_block[pj, pi, :]], axis=1)

    X_train = X[idx_train]
    X_test = X[idx_test]

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    r2 = r2_score(y_test, y_pred)

    results.append(r2)
    print(f"Layer {layer:2d}  R² = {r2:.4f}")

    del pair_block, X  # free memory before next load

# ---------------------------------------------------------------------------
# 5. Save CSV
# ---------------------------------------------------------------------------
csv_path = os.path.join(BASE_DIR, "layer_regression_r2.csv")
with open(csv_path, "w") as f:
    f.write("layer,r2\n")
    for layer, r2 in enumerate(results):
        f.write(f"{layer},{r2:.6f}\n")
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# 6. Plot R² across layers
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(n_layers), results, "o-", markersize=4)
ax.set_xlabel("Layer")
ax.set_ylabel("Test R²")
ax.set_title("Linear Regression Probe: Distance Predictability Across Layers")
ax.set_xticks(range(0, n_layers, 4))
ax.set_ylim(0, 1.05)
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
plt.tight_layout()
png_path = os.path.join(BASE_DIR, "visualizations", "layer_regression_r2.png")
plt.savefig(png_path, dpi=150)
plt.close()
print(f"Plot saved to {png_path}")

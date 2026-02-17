"""
Layer-wise linear regression probe with standardized features.

Same as probe_layer_regression.py, but standardizes features (zero mean, unit
variance) before fitting. This makes regression coefficients directly comparable
across features — a large |coefficient| genuinely means the feature contributes
more to the prediction, regardless of the feature's original scale.

R² values are identical to the unstandardized version (standardization is a
linear transform that doesn't affect linear regression fit quality).
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(os.path.dirname(_SCRIPT_DIR))
PROTEIN_DIR = os.path.join(ROOT_DIR, "proteins", "7b3a")
os.makedirs(os.path.join(PROTEIN_DIR, "visualizations"), exist_ok=True)
os.makedirs(os.path.join(PROTEIN_DIR, "csv_files"), exist_ok=True)

# ---------------------------------------------------------------------------
# 1. Load distance matrix
# ---------------------------------------------------------------------------
dist_csv = os.path.join(PROTEIN_DIR, "csv_files", "residue_distances.csv")

# np.genfromtxt reads a CSV into a 2D numpy array.
# delimiter="," splits on commas; skip_header=1 skips the first row (column labels).
# Result: raw has shape (276, 276) — a symmetric matrix of CA-CA distances in Angstroms.
raw = np.genfromtxt(dist_csv, delimiter=",", skip_header=1)

# raw.shape returns a tuple (rows, cols). shape[0] gives the number of rows.
n = raw.shape[0]  # 276 resolved residues
print(f"Distance matrix: {n} x {n} residues")

PAIR_OFFSET = 4
C = 128  # channels per pair representation

# ---------------------------------------------------------------------------
# 2. Build target vector y (same across all layers)
# ---------------------------------------------------------------------------

# Number of unique pairs in the upper triangle of an n×n matrix (excluding diagonal).
# For n=276: 276 * 275 / 2 = 37,950 pairs.
n_pairs = n * (n - 1) // 2

# np.empty allocates an array of shape (37950,) WITHOUT initializing values (faster
# than np.zeros when you plan to fill every element). dtype=np.float64 is 64-bit float.
y = np.empty(n_pairs, dtype=np.float64)
idx = 0
for i in range(n):
    for j in range(i + 1, n):
        # Flatten the upper triangle of the distance matrix into a 1D vector.
        # Each entry y[idx] is the CA-CA distance between residue i and residue j.
        y[idx] = raw[i, j]
        idx += 1

# y.min() and y.max() return the single smallest/largest element in the array.
print(f"Pairs: {n_pairs}  |  Distance range: {y.min():.2f} – {y.max():.2f} Å")

# ---------------------------------------------------------------------------
# 3. Fixed train/test split on indices (same split for every layer)
# ---------------------------------------------------------------------------

# np.arange(n_pairs) creates a 1D array [0, 1, 2, ..., 37949] — one index per pair.
all_idx = np.arange(n_pairs)

# train_test_split randomly partitions all_idx into two arrays:
#   idx_train: shape ~(30360,) — 80% of indices
#   idx_test:  shape ~(7590,)  — 20% of indices
# random_state=42 makes the split deterministic/reproducible.
idx_train, idx_test = train_test_split(all_idx, test_size=0.2, random_state=42)

# Fancy indexing: y[idx_train] selects elements at the positions given by idx_train.
# If idx_train = [5, 100, 23, ...], then y_train = [y[5], y[100], y[23], ...].
# y_train shape: (30360,), y_test shape: (7590,)
y_train = y[idx_train]
y_test = y[idx_test]

# Precompute pair (i, j) index arrays — maps each of the 37,950 pair indices back to
# which residue pair (i, j) it corresponds to, so we don't recompute this every layer.
# pair_i and pair_j each have shape (37950,).
pair_i = np.empty(n_pairs, dtype=np.int32)
pair_j = np.empty(n_pairs, dtype=np.int32)
k = 0
for i in range(n):
    for j in range(i + 1, n):
        pair_i[k] = i
        pair_j[k] = j
        k += 1

# ---------------------------------------------------------------------------
# 4. Loop over all 48 layers (with standardization)
# ---------------------------------------------------------------------------
results = []
n_layers = 48

# np.zeros creates an array filled with 0.0. Shape (48, 256): one row per layer,
# one column per feature (2 * 128 = 256 because we concatenate upper and lower).
all_coefs = np.zeros((n_layers, 2 * C))

for layer in range(n_layers):
    pb_path = os.path.join(PROTEIN_DIR, "pair_blocks", f"7b3a_A_pair_block_{layer}.npy")

    # np.load reads a .npy file (numpy's binary format) back into an array.
    # pair_block shape: (280, 280, 128) — for every pair (i, j), 128 feature channels.
    pair_block = np.load(pb_path)

    # Apply the 4-residue alignment offset. pair_i and pair_j are shape (37950,);
    # adding a scalar broadcasts — adds 4 to every element.
    # pi, pj shape: (37950,) with values in range [4, 279].
    pi = pair_i + PAIR_OFFSET
    pj = pair_j + PAIR_OFFSET

    # Advanced (fancy) indexing: pair_block[pi, pj, :] uses the arrays pi and pj
    # to index the first two dimensions simultaneously. For each k in 0..37949,
    # it selects pair_block[pi[k], pj[k], :] — a 128-dim vector.
    # Result shape: (37950, 128) — the "upper" features (i→j direction).
    #
    # pair_block[pj, pi, :] is the same but with indices swapped — the "lower"
    # features (j→i direction). Also shape (37950, 128).
    #
    # np.concatenate joins arrays along a given axis.
    # axis=1 concatenates along columns: (37950, 128) + (37950, 128) → (37950, 256).
    # Each row is now a 256-dim feature vector for one residue pair.
    X = np.concatenate([pair_block[pi, pj, :], pair_block[pj, pi, :]], axis=1)

    # Fancy indexing again: select only the training/test rows from X.
    # X_train shape: (30360, 256), X_test shape: (7590, 256).
    X_train = X[idx_train]
    X_test = X[idx_test]

    # StandardScaler standardizes each feature (column) independently:
    #   x_scaled = (x - mean) / std
    # After this, each of the 256 columns has mean≈0 and std≈1 in the training set.
    #
    # fit_transform does two things in one call:
    #   1) fit: computes mean and std for each of the 256 columns from X_train
    #   2) transform: applies (x - mean) / std to X_train
    # Result: X_train shape unchanged (30360, 256), but values are standardized.
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)

    # scaler.transform applies the SAME mean/std (learned from training data) to
    # X_test. This avoids data leakage — test data doesn't influence the scaling.
    # X_test shape unchanged: (7590, 256).
    X_test = scaler.transform(X_test)

    # LinearRegression fits: y = X @ coef + intercept, minimizing squared error.
    # model.fit(X_train, y_train) learns:
    #   model.coef_:      shape (256,) — one weight per feature
    #   model.intercept_: scalar — the bias term
    # Because features are standardized, coef_[k] directly tells you: "how much does
    # the predicted distance change when feature k moves by one standard deviation?"
    model = LinearRegression()
    model.fit(X_train, y_train)

    # model.predict computes X_test @ model.coef_ + model.intercept_.
    # y_pred shape: (7590,) — one predicted distance per test pair.
    y_pred = model.predict(X_test)

    # r2_score computes the coefficient of determination R²:
    #   R² = 1 - sum((y_true - y_pred)²) / sum((y_true - mean(y_true))²)
    # R²=1 means perfect prediction, R²=0 means no better than predicting the mean.
    r2 = r2_score(y_test, y_pred)

    # Store this layer's 256 coefficients into the corresponding row of all_coefs.
    # model.coef_ shape: (256,), all_coefs[layer, :] selects row `layer` (all columns).
    all_coefs[layer, :] = model.coef_
    results.append(r2)
    print(f"Layer {layer:2d}  R² = {r2:.4f}")

    # Free memory: pair_block is (280, 280, 128) ≈ 38 MB, X is (37950, 256) ≈ 78 MB.
    del pair_block, X

# ---------------------------------------------------------------------------
# 5. Save CSV
# ---------------------------------------------------------------------------
csv_path = os.path.join(PROTEIN_DIR, "csv_files", "layer_regression_r2_standardized.csv")
with open(csv_path, "w") as f:
    f.write("layer,r2\n")
    for layer, r2 in enumerate(results):
        f.write(f"{layer},{r2:.6f}\n")
print(f"\nResults saved to {csv_path}")

# ---------------------------------------------------------------------------
# 6. Plot R² across layers
# ---------------------------------------------------------------------------

# plt.subplots creates a figure and one set of axes. figsize=(width, height) in inches.
# Returns (fig, ax) where fig is the overall figure and ax is the plotting area.
fig, ax = plt.subplots(figsize=(10, 5))

# ax.plot draws a line chart. "o-" means circle markers connected by lines.
# range(n_layers) = [0, 1, ..., 47] for x-axis, results = list of 48 R² values for y.
ax.plot(range(n_layers), results, "o-", markersize=4)
ax.set_xlabel("Layer")
ax.set_ylabel("Test R²")
ax.set_title("Linear Regression Probe (Standardized): Distance Predictability Across Layers")

# set_xticks controls which x positions get labeled. range(0, 48, 4) = [0, 4, 8, ...44].
ax.set_xticks(range(0, n_layers, 4))
ax.set_ylim(0, 1.05)

# axhline draws a horizontal reference line across the full width of the plot.
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)

# tight_layout adjusts spacing so labels don't get clipped.
plt.tight_layout()
png_path = os.path.join(PROTEIN_DIR, "visualizations", "layer_regression_r2_standardized.png")

# savefig writes the current figure to a file. dpi=150 controls resolution (dots per inch).
plt.savefig(png_path, dpi=150)

# plt.close frees the figure's memory (important when generating many plots in a loop).
plt.close()
print(f"Plot saved to {png_path}")

# ===========================================================================
# RANK EVOLUTION OF FINAL LAYER'S TOP 10 (STANDARDIZED COEFFICIENTS)
# ===========================================================================

def feature_label(fi):
    """Return human-readable label for feature index (0-255).
    Indices 0-127 are "upper" (pair_block[i,j]), 128-255 are "lower" (pair_block[j,i])."""
    if fi < C:
        return f"upper[{fi}]"
    return f"lower[{fi - C}]"

# np.abs computes element-wise absolute value. all_coefs shape: (48, 256),
# so abs_coefs also has shape (48, 256). We use absolute values because sign
# indicates direction (positive/negative relationship with distance) but not importance.
abs_coefs = np.abs(all_coefs)  # (48, 256)
TOP_K = 50

# For each layer, find the indices of the TOP_K features with largest |coefficient|.
# top_k_per_layer shape: (48, 50) — each row holds 50 feature indices.
top_k_per_layer = np.zeros((n_layers, TOP_K), dtype=int)
for layer in range(n_layers):
    # np.argsort returns indices that would sort the array in ascending order.
    # Shape: (256,). [::-1] reverses to descending order (largest first).
    ranked = np.argsort(abs_coefs[layer])[::-1]
    # ranked[:TOP_K] takes the first 50 elements — the 50 most important feature indices.
    top_k_per_layer[layer, :] = ranked[:TOP_K]

# top_k_per_layer[-1] indexes the last row (layer 47). [:10] takes first 10 entries.
# final_top_10 shape: (10,) — the 10 most important feature indices at the final layer.
final_top_10 = top_k_per_layer[-1][:10]

# Track how each of those 10 features ranked at every layer.
# rank_evolution shape: (48, 10). Entry [layer, i] = rank of final_top_10[i] at that layer.
rank_evolution = np.zeros((n_layers, len(final_top_10)), dtype=int)
for layer in range(n_layers):
    # Same descending sort as above — ranked[0] is the most important feature at this layer.
    ranked = np.argsort(abs_coefs[layer])[::-1]
    # Invert the permutation: build a dict mapping feature_index → its rank.
    # enumerate(ranked) yields (rank, feature_index) pairs: (0, best_feature), (1, second), ...
    rank_lookup = {ch: rank for rank, ch in enumerate(ranked)}
    for i, fi in enumerate(final_top_10):
        rank_evolution[layer, i] = rank_lookup[fi]

# ---------------------------------------------------------------------------
# 7. Plot rank evolution
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))

# plt.cm.tab10 is a colormap with 10 distinct colors. np.linspace(0, 1, 10) generates
# 10 evenly spaced values from 0 to 1. Passing these to the colormap returns an array
# of shape (10, 4) — 10 RGBA color tuples, one per line we'll plot.
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for i, fi in enumerate(final_top_10):
    # rank_evolution[:, i] selects column i — all 48 layers for the i-th tracked feature.
    # Shape: (48,). Plotted against range(48) = [0, 1, ..., 47].
    ax.plot(range(n_layers), rank_evolution[:, i], 'o-',
            markersize=3, linewidth=1.5, color=colors[i],
            label=feature_label(fi))

# axhline draws a horizontal line at rank=50 to show the top-50 cutoff.
ax.axhline(y=TOP_K, color='red', linestyle='--', linewidth=1, alpha=0.5,
           label=f'Top-{TOP_K} cutoff')

# axvspan shades a vertical band (layers 8-18, the R² emergence zone).
ax.axvspan(8, 18, alpha=0.1, color='orange')

# invert_yaxis flips the y-axis so rank 0 (most important) is at the TOP of the plot,
# which is more intuitive — "rising" on the chart means becoming more important.
ax.invert_yaxis()
ax.set_xlabel("Layer")
ax.set_ylabel("Rank (lower = more important)")
ax.set_title("Rank Evolution of Final Layer's Top 10 Features (Standardized)")
ax.set_xticks(range(0, n_layers, 4))

# bbox_to_anchor=(1.02, 1) places the legend just outside the right edge of the plot.
# loc='upper left' means the upper-left corner of the legend box sits at that anchor point.
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
rank_path = os.path.join(PROTEIN_DIR, "visualizations", "regression_coef_rank_evolution_standardized.png")
plt.savefig(rank_path, dpi=150)
plt.close()
print(f"Rank evolution plot saved to {rank_path}")

# ---------------------------------------------------------------------------
# 8. Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("STANDARDIZED COEFFICIENT SUMMARY")
print("=" * 70)

print(f"\nTop 10 features at layer 47 (by |standardized coefficient|):")
print(f"{'Rank':>4s}  {'Feature':>12s}  {'|Coef|':>10s}  {'Coef':>10s}")
for rank, fi in enumerate(final_top_10):
    # abs_coefs[47, fi] indexes row 47 (last layer), column fi (specific feature).
    # Returns a single scalar — the absolute standardized coefficient for that feature.
    print(f"{rank:4d}  {feature_label(fi):>12s}  "
          f"{abs_coefs[47, fi]:10.4f}  {all_coefs[47, fi]:10.4f}")

"""
Similarity analysis of regression coefficient vectors across layers.

Produces three plots:
1. Cosine similarity heatmap (48x48)
2. Euclidean distance heatmap (48x48)
3. Euclidean distance between successive layers (line plot)
"""

import os
import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
MULTI_DIR = os.path.join(ROOT_DIR, "proteins", "multi_protein_results")
CSV_PATH = os.path.join(MULTI_DIR, "csv_files", "layer_regression_coefficients_predicted.csv")
VIS_DIR = os.path.join(MULTI_DIR, "visualizations")
os.makedirs(VIS_DIR, exist_ok=True)

# Load CSV: skip header, first col = layer, second col = intercept, rest = coefficients
data = np.loadtxt(CSV_PATH, delimiter=",", skiprows=1)
layers = data[:, 0].astype(int)
coefficients = data[:, 2:]  # skip layer and intercept columns
n_layers = len(layers)

tick_positions = list(range(0, n_layers, 5))

# --- 1. Cosine similarity heatmap ---
norms = np.linalg.norm(coefficients, axis=1, keepdims=True)
normed = coefficients / norms
cosine_sim = normed @ normed.T

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(cosine_sim, cmap="RdBu_r", vmin=-1, vmax=1, aspect="equal")
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)
ax.set_title("Cosine Similarity of Regression Coefficient Vectors Across Layers", fontsize=13)
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.set_xticklabels(tick_positions)
ax.set_yticklabels(tick_positions)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Cosine Similarity", fontsize=11)
plt.tight_layout()
out = os.path.join(VIS_DIR, "coefficient_cosine_similarity.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# --- 2. Euclidean distance heatmap ---
euc_dist = cdist(coefficients, coefficients, metric="euclidean")

fig, ax = plt.subplots(figsize=(10, 8))
im = ax.imshow(euc_dist, cmap="viridis_r", aspect="equal")
ax.set_xlabel("Layer", fontsize=12)
ax.set_ylabel("Layer", fontsize=12)
ax.set_title("Euclidean Distance Between Regression Coefficient Vectors Across Layers", fontsize=13)
ax.set_xticks(tick_positions)
ax.set_yticks(tick_positions)
ax.set_xticklabels(tick_positions)
ax.set_yticklabels(tick_positions)
cbar = fig.colorbar(im, ax=ax, shrink=0.8)
cbar.set_label("Euclidean Distance", fontsize=11)
plt.tight_layout()
out = os.path.join(VIS_DIR, "coefficient_euclidean_distance.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

# --- 3. Euclidean distance between successive layers ---
successive_dist = np.array([
    np.linalg.norm(coefficients[i + 1] - coefficients[i])
    for i in range(n_layers - 1)
])

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(n_layers - 1), successive_dist, "o-", markersize=4, linewidth=1.5)
ax.set_xlabel("Layer Transition (i → i+1)", fontsize=12)
ax.set_ylabel("Euclidean Distance", fontsize=12)
ax.set_title("Euclidean Distance Between Successive Layer Coefficient Vectors", fontsize=13)
ax.set_xticks(range(0, n_layers - 1, 5))
ax.set_xticklabels([f"{i}→{i+1}" for i in range(0, n_layers - 1, 5)], fontsize=9)
ax.grid(True, alpha=0.3)
plt.tight_layout()
out = os.path.join(VIS_DIR, "coefficient_successive_euclidean.png")
plt.savefig(out, dpi=150)
plt.close()
print(f"Saved: {out}")

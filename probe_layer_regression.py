"""
Layer-wise linear regression probe: how well can each layer's pair representations
predict CA-CA spatial distance?

For each of the 48 layers, concatenate pair_block[i,j,:] and pair_block[j,i,:]
(256-dim), train a linear regression, and record test-set R². Uses the same
fixed train/test split (random_state=42) across all layers for fair comparison.

Also tracks how regression coefficients evolve across layers: which features
(out of 256) are most important at each layer, how much the top-50 set overlaps
with the final layer's top-50, and when specific features rise to prominence.
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
TOP_K = 50
all_coefs = np.zeros((n_layers, 2 * C))  # store all 256 coefficients per layer
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

    all_coefs[layer, :] = model.coef_
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

# ===========================================================================
# COEFFICIENT EVOLUTION ANALYSIS
# ===========================================================================

def feature_label(fi):
    """Return human-readable label for feature index (0-255)."""
    if fi < C:
        return f"upper[{fi}]"
    return f"lower[{fi - C}]"

# ---------------------------------------------------------------------------
# 7. Derived metrics
# ---------------------------------------------------------------------------
abs_coefs = np.abs(all_coefs)  # (48, 256)

# Top-K features per layer (by |coefficient|)
top_k_per_layer = np.zeros((n_layers, TOP_K), dtype=int)
for layer in range(n_layers):
    ranked = np.argsort(abs_coefs[layer])[::-1]
    top_k_per_layer[layer, :] = ranked[:TOP_K]

# Final layer's top-K and top-10 for reference
final_top_k = set(top_k_per_layer[-1])
final_top_10 = top_k_per_layer[-1][:10]

# Overlap with final layer: fraction of each layer's top-K also in final's top-K
overlap_with_final = np.zeros(n_layers)
for layer in range(n_layers):
    layer_set = set(top_k_per_layer[layer])
    overlap_with_final[layer] = len(layer_set & final_top_k) / TOP_K

# Jaccard similarity between consecutive layers' top-K sets
jaccard_consecutive = np.zeros(n_layers - 1)
for layer in range(1, n_layers):
    set_prev = set(top_k_per_layer[layer - 1])
    set_curr = set(top_k_per_layer[layer])
    union = len(set_prev | set_curr)
    jaccard_consecutive[layer - 1] = len(set_prev & set_curr) / union if union > 0 else 0

# Rank evolution of final layer's top-10 features across all layers
rank_evolution = np.zeros((n_layers, len(final_top_10)), dtype=int)
for layer in range(n_layers):
    ranked = np.argsort(abs_coefs[layer])[::-1]
    rank_lookup = {ch: rank for rank, ch in enumerate(ranked)}
    for i, fi in enumerate(final_top_10):
        rank_evolution[layer, i] = rank_lookup[fi]

# ---------------------------------------------------------------------------
# 8. Save CSVs
# ---------------------------------------------------------------------------
# CSV 1: All coefficients per layer (long format)
csv1_path = os.path.join(BASE_DIR, "regression_coefficients_by_layer.csv")
with open(csv1_path, "w") as f:
    f.write("layer,feature_index,feature_label,coefficient,abs_coefficient,rank\n")
    for layer in range(n_layers):
        ranked = np.argsort(abs_coefs[layer])[::-1]
        rank_lookup = {fi: r for r, fi in enumerate(ranked)}
        for fi in range(2 * C):
            f.write(f"{layer},{fi},{feature_label(fi)},"
                    f"{all_coefs[layer, fi]:.6f},{abs_coefs[layer, fi]:.6f},"
                    f"{rank_lookup[fi]}\n")
print(f"Coefficients CSV saved to {csv1_path}")

# CSV 2: Overlap summary per layer
csv2_path = os.path.join(BASE_DIR, "regression_coefficient_overlap.csv")
with open(csv2_path, "w") as f:
    f.write("layer,r2,overlap_with_final_top50,jaccard_with_prev\n")
    for layer in range(n_layers):
        jacc = jaccard_consecutive[layer - 1] if layer > 0 else float('nan')
        f.write(f"{layer},{results[layer]:.6f},"
                f"{overlap_with_final[layer]:.4f},{jacc:.4f}\n")
print(f"Overlap CSV saved to {csv2_path}")

# ---------------------------------------------------------------------------
# 9. Visualization: Coefficient heatmap (top 30 features)
# ---------------------------------------------------------------------------
# Select top 30 features by mean |coefficient| in layers 20-47 (high-R² regime)
mean_abs = abs_coefs[20:, :].mean(axis=0)
top_30_idx = np.sort(np.argsort(mean_abs)[::-1][:30])

fig, ax = plt.subplots(figsize=(14, 8))
im = ax.imshow(abs_coefs[:, top_30_idx], aspect='auto',
               cmap='viridis', interpolation='nearest')
ax.set_yticks(range(0, n_layers, 4))
ax.set_xticks(range(len(top_30_idx)))
ax.set_xticklabels([feature_label(fi) for fi in top_30_idx], fontsize=7, rotation=60, ha='right')
ax.set_xlabel("Feature")
ax.set_ylabel("Layer")
ax.set_title("Top 30 Feature |Coefficients| Across Layers")
ax.axhline(y=8, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
ax.axhline(y=18, color='white', linestyle='--', linewidth=0.8, alpha=0.7)
plt.colorbar(im, ax=ax, label="|coefficient|")
plt.tight_layout()
heatmap_path = os.path.join(BASE_DIR, "visualizations", "regression_coef_heatmap.png")
plt.savefig(heatmap_path, dpi=150)
plt.close()
print(f"Heatmap saved to {heatmap_path}")

# ---------------------------------------------------------------------------
# 10. Visualization: Overlap and Jaccard
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top: overlap with final layer
ax1.plot(range(n_layers), overlap_with_final, 'o-', markersize=4, color='steelblue')
ax1.axvspan(8, 18, alpha=0.1, color='orange', label='R² emergence zone')
ax1.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax1.set_ylabel("Fraction Overlap")
ax1.set_title(f"Overlap of Top-{TOP_K} Features with Final Layer (47)")
ax1.set_ylim(0, 1.05)
ax1.legend()
ax1.grid(True, alpha=0.3)

# Bottom: Jaccard consecutive
ax2.plot(range(1, n_layers), jaccard_consecutive, 'o-', markersize=4, color='darkorange')
ax2.axvspan(8, 18, alpha=0.1, color='orange')
ax2.axhline(1.0, color='gray', linestyle='--', linewidth=0.8, alpha=0.5)
ax2.set_xlabel("Layer")
ax2.set_ylabel("Jaccard Similarity")
ax2.set_title(f"Jaccard Similarity of Top-{TOP_K} Features (Consecutive Layers)")
ax2.set_ylim(0, 1.05)
ax2.set_xticks(range(0, n_layers, 4))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
overlap_path = os.path.join(BASE_DIR, "visualizations", "regression_coef_overlap.png")
plt.savefig(overlap_path, dpi=150)
plt.close()
print(f"Overlap plot saved to {overlap_path}")

# ---------------------------------------------------------------------------
# 11. Visualization: Rank evolution of final layer's top 10 features
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(12, 6))
colors = plt.cm.tab10(np.linspace(0, 1, 10))

for i, fi in enumerate(final_top_10):
    ax.plot(range(n_layers), rank_evolution[:, i], 'o-',
            markersize=3, linewidth=1.5, color=colors[i],
            label=feature_label(fi))

ax.axhline(y=TOP_K, color='red', linestyle='--', linewidth=1, alpha=0.5,
           label=f'Top-{TOP_K} cutoff')
ax.axvspan(8, 18, alpha=0.1, color='orange')
ax.invert_yaxis()
ax.set_xlabel("Layer")
ax.set_ylabel("Rank (lower = more important)")
ax.set_title("Rank Evolution of Final Layer's Top 10 Features")
ax.set_xticks(range(0, n_layers, 4))
ax.legend(bbox_to_anchor=(1.02, 1), loc='upper left', fontsize=8)
ax.grid(True, alpha=0.3)
plt.tight_layout()
rank_path = os.path.join(BASE_DIR, "visualizations", "regression_coef_rank_evolution.png")
plt.savefig(rank_path, dpi=150)
plt.close()
print(f"Rank evolution plot saved to {rank_path}")

# ---------------------------------------------------------------------------
# 12. Visualization: Top 5 feature traces + R² vs total top-50 importance
# ---------------------------------------------------------------------------
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))

# Top: |coefficient| traces for top 5 features
top_5 = final_top_10[:5]
for fi in top_5:
    ax1.plot(range(n_layers), abs_coefs[:, fi], 'o-',
             markersize=4, linewidth=1.5, label=feature_label(fi))
ax1.set_xlabel("Layer")
ax1.set_ylabel("|Coefficient|")
ax1.set_title("Top 5 Feature |Coefficients| Across Layers")
ax1.legend()
ax1.grid(True, alpha=0.3)
ax1.set_xticks(range(0, n_layers, 4))
ax1.axvspan(8, 18, alpha=0.1, color='orange')

# Bottom: R² vs total top-50 importance (dual y-axis)
top50_frac = np.array([abs_coefs[l, top_k_per_layer[l]].sum() / abs_coefs[l].sum() for l in range(n_layers)])
ax2.plot(range(n_layers), results, 'o-', color='steelblue', markersize=4, label='R²')
ax2.set_ylabel("Test R²", color='steelblue')
ax2.set_xlabel("Layer")
ax2.set_ylim(0, 1.05)

ax2_right = ax2.twinx()
ax2_right.plot(range(n_layers), top50_frac, 's-', color='darkorange', markersize=4, label='Top-50 fraction')
ax2_right.set_ylabel("Top-50 / Total |Coefficient| Fraction", color='darkorange')

lines1, labels1 = ax2.get_legend_handles_labels()
lines2, labels2 = ax2_right.get_legend_handles_labels()
ax2.legend(lines1 + lines2, labels1 + labels2, loc='center right')
ax2.set_title("R² and Top-50 Coefficient Concentration Across Layers")
ax2.set_xticks(range(0, n_layers, 4))
ax2.grid(True, alpha=0.3)

plt.tight_layout()
top_path = os.path.join(BASE_DIR, "visualizations", "regression_coef_top_channels.png")
plt.savefig(top_path, dpi=150)
plt.close()
print(f"Top channels plot saved to {top_path}")

# ---------------------------------------------------------------------------
# 13. Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 70)
print("COEFFICIENT EVOLUTION SUMMARY")
print("=" * 70)

print(f"\nTop 10 features at layer 47 (by |coefficient|):")
print(f"{'Rank':>4s}  {'Feature':>12s}  {'|Coef|':>10s}  {'Coef':>10s}")
for rank, fi in enumerate(final_top_10):
    print(f"{rank:4d}  {feature_label(fi):>12s}  "
          f"{abs_coefs[47, fi]:10.4f}  {all_coefs[47, fi]:10.4f}")

print(f"\nOverlap with final layer's top-{TOP_K} features:")
for layer in [0, 5, 10, 15, 20, 30, 40, 47]:
    print(f"  Layer {layer:2d}: {overlap_with_final[layer]:.0%} overlap, "
          f"R² = {results[layer]:.4f}")

idx_80 = np.argmax(overlap_with_final >= 0.8)
if overlap_with_final[idx_80] >= 0.8:
    print(f"\nFirst layer with >=80% overlap: layer {idx_80} "
          f"(R² = {results[idx_80]:.4f})")

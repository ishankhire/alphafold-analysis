#!/usr/bin/env python3
"""
probe_layer_regression_multi.py
================================
Trains a single linear regression model per layer across all available proteins,
pooling all residue-pair embeddings and their CA-CA distances.

For each of the 48 AlphaFold layers:
  - Feature: concatenated pair-block vectors [pair_block[i,j,:], pair_block[j,i,:]] → 256-dim
  - Target: CA-CA distance in Angstroms for that residue pair
  - Split: fixed 80/20 train/test across the pooled dataset (random_state=42)
  - Standardization: StandardScaler fit on training rows only

Outputs
-------
  csv_files/layer_regression_r2_multi.csv           — per-layer R² (test set)
  csv_files/layer_regression_coefficients_multi.csv — coefficients for all 48 layers
  csv_files/predicted_distances_last_layer_multi.csv — predicted vs actual for last layer
  visualizations/layer_regression_r2_multi.png
  visualizations/pred_vs_actual_all_last_layer_multi.png
  visualizations/pred_vs_actual_per_protein_last_layer_multi.png
"""

import sys
import os
import csv
import math
from pathlib import Path

# scripts/multi_protein/regression/ is 4 levels deep; climb to project root.
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

from lib.alignment import build_alignment, validate_alignment
from lib.structure import extract_residue_coordinates, compute_distance_matrix, detect_chain

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
RESULTS_DIR = PROJECT_ROOT / "proteins" / "multi_protein_results"
CSV_DIR     = RESULTS_DIR / "csv_files"
VIS_DIR     = RESULTS_DIR / "visualizations"
CSV_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

PROTEINS_ROOT = PROJECT_ROOT / "proteins"

N_LAYERS = 48

# ---------------------------------------------------------------------------
# Pre-computation pass: load alignment + distances for each protein
# ---------------------------------------------------------------------------
protein_dirs = sorted([
    p for p in PROTEINS_ROOT.iterdir()
    if p.is_dir() and p.name != "multi_protein_results"
])

protein_data  = []   # list of dicts, one per successfully loaded protein
skipped       = []   # list of (protein_id, reason) for the summary

for protein_dir in protein_dirs:
    protein_id = protein_dir.name

    cif_files = list(protein_dir.glob("*.cif")) + list(protein_dir.glob("*.CIF"))
    if not cif_files:
        skipped.append((protein_id, "no CIF file"))
        print(f"[SKIP] {protein_id}: no CIF file")
        continue
    cif_path = str(cif_files[0])

    chain_id = detect_chain(protein_dir)
    if chain_id is None:
        skipped.append((protein_id, "could not detect chain"))
        print(f"[SKIP] {protein_id}: could not detect chain")
        continue

    first_pb = protein_dir / "pair_blocks" / f"{protein_id}_pair_block_0.npy"
    if not first_pb.exists():
        skipped.append((protein_id, "pair blocks not found"))
        print(f"[SKIP] {protein_id}: pair blocks not found")
        continue

    try:
        alignment = build_alignment(cif_path, chain_id, protein_id)

        ca_result = extract_residue_coordinates(cif_path, chain_id)
        n_ca = len(ca_result["residues"])
        validate_alignment(alignment, str(first_pb), n_ca)

        coords      = ca_result["coordinates"]
        dist_matrix = compute_distance_matrix(coords)

        struct_i, struct_j = np.triu_indices(alignment.n_modeled, k=1)
        pb_i      = alignment.pb_indices[struct_i]
        pb_j      = alignment.pb_indices[struct_j]
        y_protein = dist_matrix[struct_i, struct_j]

        protein_data.append({
            "protein_id": protein_id,
            "chain_id":   chain_id,
            "pb_dir":     protein_dir / "pair_blocks",
            "pb_i":       pb_i,
            "pb_j":       pb_j,
            "struct_i":   struct_i,
            "struct_j":   struct_j,
            "y":          y_protein,
        })
        print(f"[OK]   {protein_id}: {alignment.n_modeled} residues, {len(y_protein)} pairs")

    except AssertionError as e:
        skipped.append((protein_id, f"alignment failed — {e}"))
        print(f"[SKIP] {protein_id}: alignment failed — {e}")
    except Exception as e:
        skipped.append((protein_id, f"unexpected error — {e}"))
        print(f"[SKIP] {protein_id}: unexpected error — {e}")

if not protein_data:
    raise RuntimeError("No proteins loaded successfully.")

# ---------------------------------------------------------------------------
# Pooled target + per-protein offset tracking
# ---------------------------------------------------------------------------
y_all         = np.concatenate([d["y"] for d in protein_data])
n_total_pairs = len(y_all)
n_proteins    = len(protein_data)

protein_offsets = []   # (protein_id, start, end) inclusive of end-exclusive
offset = 0
for d in protein_data:
    n = len(d["y"])
    protein_offsets.append((d["protein_id"], offset, offset + n))
    offset += n

print(f"\nTotal pairs across {n_proteins} protein(s): {n_total_pairs}")

# Fixed 80/20 train/test split over pooled pairs
all_idx               = np.arange(n_total_pairs)
idx_train, idx_test   = train_test_split(all_idx, test_size=0.2, random_state=42)
y_train               = y_all[idx_train]
y_test                = y_all[idx_test]

# Split label array (for predictions CSV)
split_labels          = np.empty(n_total_pairs, dtype=object)
split_labels[idx_train] = "train"
split_labels[idx_test]  = "test"

# ---------------------------------------------------------------------------
# Layer loop
# ---------------------------------------------------------------------------
results         = []   # test R² per layer
all_coefs       = []   # list of dicts, one per layer
last_layer_preds = None

print()
for layer in range(N_LAYERS):
    X_chunks = []
    for d in protein_data:
        pb_path = d["pb_dir"] / f"{d['protein_id']}_pair_block_{layer}.npy"
        pb      = np.load(pb_path)
        X_chunk = np.concatenate([
            pb[d["pb_i"], d["pb_j"], :],   # (n_pairs, 128)
            pb[d["pb_j"], d["pb_i"], :],   # (n_pairs, 128)
        ], axis=1)                         # (n_pairs, 256)
        X_chunks.append(X_chunk)
        del pb

    X_all = np.vstack(X_chunks)           # (n_total_pairs, 256)

    scaler  = StandardScaler()
    X_train = scaler.fit_transform(X_all[idx_train])
    X_test  = scaler.transform(X_all[idx_test])

    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    results.append(r2)
    print(f"Layer {layer:2d}  R² = {r2:.4f}")

    # Save coefficients
    coef_row = {"layer": layer, "intercept": float(model.intercept_)}
    coef_row.update({f"coef_{i}": float(c) for i, c in enumerate(model.coef_)})
    all_coefs.append(coef_row)

    # Last layer: predict on ALL pairs and keep for plots/CSV
    if layer == N_LAYERS - 1:
        X_all_std    = scaler.transform(X_all)
        last_layer_preds = model.predict(X_all_std)   # (n_total_pairs,)

    del X_all, X_train, X_test

# ---------------------------------------------------------------------------
# Save CSV 1: layer R²
# ---------------------------------------------------------------------------
r2_csv = CSV_DIR / "layer_regression_r2_multi.csv"
with open(r2_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["layer", "r2", "n_proteins", "n_total_pairs"])
    for layer, r2 in enumerate(results):
        writer.writerow([layer, f"{r2:.6f}", n_proteins, n_total_pairs])
print(f"\nSaved R² CSV → {r2_csv}")

# ---------------------------------------------------------------------------
# Save CSV 2: coefficients
# ---------------------------------------------------------------------------
coef_csv = CSV_DIR / "layer_regression_coefficients_multi.csv"
fieldnames = ["layer", "intercept"] + [f"coef_{i}" for i in range(256)]
with open(coef_csv, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(all_coefs)
print(f"Saved coefficients CSV → {coef_csv}")

# ---------------------------------------------------------------------------
# Save CSV 3: predicted distances (last layer, all pairs)
# ---------------------------------------------------------------------------
pred_csv = CSV_DIR / "predicted_distances_last_layer_multi.csv"
with open(pred_csv, "w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(["protein_id", "struct_i", "struct_j",
                     "actual_distance_A", "predicted_distance_A", "split"])
    for pid, start, end in protein_offsets:
        d = next(x for x in protein_data if x["protein_id"] == pid)
        for k in range(end - start):
            g = start + k
            writer.writerow([
                pid,
                int(d["struct_i"][k]),
                int(d["struct_j"][k]),
                f"{y_all[g]:.4f}",
                f"{last_layer_preds[g]:.4f}",
                split_labels[g],
            ])
print(f"Saved predicted distances CSV → {pred_csv}")

# ---------------------------------------------------------------------------
# Plot 1: Layer-wise R²
# ---------------------------------------------------------------------------
fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(N_LAYERS), results, marker="o", markersize=3, linewidth=1.5)
ax.set_xlabel("Layer")
ax.set_ylabel("Test R²")
ax.set_title(
    f"Layer-wise Regression R² ({n_proteins} protein(s), {n_total_pairs:,} pairs)"
)
ax.set_ylim(0, 1)
ax.grid(True, alpha=0.3)
fig.tight_layout()
r2_png = VIS_DIR / "layer_regression_r2_multi.png"
fig.savefig(r2_png, dpi=150)
plt.close(fig)
print(f"Saved R² plot → {r2_png}")

# ---------------------------------------------------------------------------
# Plot 2: Predicted vs actual — all proteins combined (last layer)
# ---------------------------------------------------------------------------
COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

fig, ax = plt.subplots(figsize=(8, 7))
for idx_p, (pid, start, end) in enumerate(protein_offsets):
    y_act  = y_all[start:end]
    y_pred = last_layer_preds[start:end]
    ax.scatter(y_act, y_pred,
               s=2, alpha=0.3,
               color=COLORS[idx_p % len(COLORS)],
               label=f"{pid} ({end - start:,} pairs)")

# y = x diagonal
lim_min = 0
lim_max = y_all.max() * 1.05
ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", linewidth=1, label="y = x")
ax.set_xlim(lim_min, lim_max)
ax.set_ylim(lim_min, lim_max)
ax.set_xlabel("Actual CA-CA Distance (Å)")
ax.set_ylabel("Predicted CA-CA Distance (Å)")
ax.set_title(f"Layer 47: Predicted vs Actual Distance (all proteins)")
ax.legend(markerscale=4, fontsize=9)
fig.tight_layout()
all_png = VIS_DIR / "pred_vs_actual_all_last_layer_multi.png"
fig.savefig(all_png, dpi=150)
plt.close(fig)
print(f"Saved combined pred vs actual plot → {all_png}")

# ---------------------------------------------------------------------------
# Plot 3: Predicted vs actual — per-protein grid (last layer)
# ---------------------------------------------------------------------------
n_cols  = 2
n_rows  = math.ceil(n_proteins / n_cols)
fig, axes = plt.subplots(n_rows, n_cols,
                          figsize=(7 * n_cols, 6 * n_rows),
                          squeeze=False)

for idx_p, (pid, start, end) in enumerate(protein_offsets):
    row, col = divmod(idx_p, n_cols)
    ax = axes[row][col]
    y_act  = y_all[start:end]
    y_pred = last_layer_preds[start:end]
    r2_p   = r2_score(y_act, y_pred)
    ax.scatter(y_act, y_pred, s=2, alpha=0.4,
               color=COLORS[idx_p % len(COLORS)])
    lim_max_p = max(y_act.max(), y_pred.max()) * 1.05
    ax.plot([0, lim_max_p], [0, lim_max_p], "k--", linewidth=1)
    ax.set_xlim(0, lim_max_p)
    ax.set_ylim(0, lim_max_p)
    ax.set_xlabel("Actual CA-CA Distance (Å)")
    ax.set_ylabel("Predicted CA-CA Distance (Å)")
    ax.set_title(f"{pid}  (R² = {r2_p:.4f})")
    ax.grid(True, alpha=0.3)

# Hide unused panels
for idx_p in range(n_proteins, n_rows * n_cols):
    row, col = divmod(idx_p, n_cols)
    axes[row][col].set_visible(False)

fig.suptitle("Layer 47: Predicted vs Actual Distance (per protein)", fontsize=14)
fig.tight_layout()
per_png = VIS_DIR / "pred_vs_actual_per_protein_last_layer_multi.png"
fig.savefig(per_png, dpi=150)
plt.close(fig)
print(f"Saved per-protein grid plot → {per_png}")

# ---------------------------------------------------------------------------
# Console summary
# ---------------------------------------------------------------------------
print("\n" + "=" * 60)
print("SUMMARY")
print("=" * 60)
print(f"Included proteins ({n_proteins}):")
for pid, start, end in protein_offsets:
    print(f"  {pid}: {end - start:,} pairs")
if skipped:
    print(f"\nSkipped proteins ({len(skipped)}):")
    for pid, reason in skipped:
        print(f"  {pid}: {reason}")
print(f"\nBest layer: {int(np.argmax(results))}  R² = {max(results):.4f}")
print(f"Layer 47 R² = {results[-1]:.4f}")

# ---------------------------------------------------------------------------
# Coefficient analysis
# ---------------------------------------------------------------------------
C_DIM = 128  # channels per pair representation


def feature_label_multi(fi, c=C_DIM):
    return f"upper[{fi}]" if fi < c else f"lower[{fi - c}]"


# Build (N_LAYERS, 256) coefficient matrix from stored list of dicts
coef_matrix = np.array([
    [row[f"coef_{i}"] for i in range(256)]
    for row in all_coefs
], dtype=np.float64)

abs_coefs_m = np.abs(coef_matrix)

# ── Top-10 coefficients at layer 47 ────────────────────────────────────────
last_coefs = coef_matrix[-1]
abs_last   = abs_coefs_m[-1]
top10_idx  = np.argsort(abs_last)[::-1][:10]

print("\n" + "=" * 60)
print("TOP 10 COEFFICIENTS AT LAYER 47  (standardized features)")
print("=" * 60)
print(f"{'Rank':>4}  {'Feature':>12}  {'|Coef|':>10}  {'Coef':>10}")
for rank, fi in enumerate(top10_idx):
    print(f"{rank:4d}  {feature_label_multi(fi):>12}  "
          f"{abs_last[fi]:10.4f}  {last_coefs[fi]:10.4f}")

# ── Cosine similarity between consecutive layer coefficient vectors ──────────
cosine_sim = np.zeros(N_LAYERS - 1)
for layer in range(1, N_LAYERS):
    v1, v2 = coef_matrix[layer - 1], coef_matrix[layer]
    n1, n2 = np.linalg.norm(v1), np.linalg.norm(v2)
    cosine_sim[layer - 1] = np.dot(v1, v2) / (n1 * n2) if (n1 > 0 and n2 > 0) else 0.0

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, N_LAYERS), cosine_sim, "o-", markersize=4, color="steelblue")
ax.set_xlabel("Layer")
ax.set_ylabel("Cosine Similarity")
ax.set_title("Cosine Similarity of Coefficient Vectors Between Consecutive Layers")
ax.set_xlim(0, N_LAYERS - 1)
ax.set_xticks(range(0, N_LAYERS, 4))
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
cosine_png = VIS_DIR / "coef_cosine_similarity_multi.png"
fig.savefig(cosine_png, dpi=150)
plt.close(fig)
print(f"\nSaved cosine similarity plot → {cosine_png}")

# ── Jaccard similarity of top-50 features between consecutive layers ─────────
TOP_K = 50
top_k_per_layer = np.zeros((N_LAYERS, TOP_K), dtype=int)
for layer in range(N_LAYERS):
    top_k_per_layer[layer] = np.argsort(abs_coefs_m[layer])[::-1][:TOP_K]

jaccard_consecutive = np.zeros(N_LAYERS - 1)
for layer in range(1, N_LAYERS):
    set_prev = set(top_k_per_layer[layer - 1])
    set_curr = set(top_k_per_layer[layer])
    union = len(set_prev | set_curr)
    jaccard_consecutive[layer - 1] = (
        len(set_prev & set_curr) / union if union > 0 else 0.0
    )

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(range(1, N_LAYERS), jaccard_consecutive, "o-", markersize=4, color="darkorange")
ax.set_xlabel("Layer")
ax.set_ylabel("Jaccard Similarity")
ax.set_title(f"Jaccard Similarity of Top-{TOP_K} Features Between Consecutive Layers")
ax.set_xlim(0, N_LAYERS - 1)
ax.set_ylim(0, 1.05)
ax.set_xticks(range(0, N_LAYERS, 4))
ax.axhline(1.0, color="gray", linestyle="--", linewidth=0.8, alpha=0.5)
ax.grid(True, alpha=0.3)
fig.tight_layout()
jaccard_png = VIS_DIR / "coef_jaccard_similarity_multi.png"
fig.savefig(jaccard_png, dpi=150)
plt.close(fig)
print(f"Saved Jaccard similarity plot → {jaccard_png}")

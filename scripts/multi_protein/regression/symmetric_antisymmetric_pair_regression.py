#!/usr/bin/env python3
"""
symmetric_antisymmetric_pair_regression.py
=========================================
Compare four pooled linear probes for predicting CA-CA distances from
AlphaFold pair representations on a fixed 4-protein subset:

1. Baseline: [pb[i,j,:], pb[j,i,:]]
2. Symmetric-only: S = 0.5 * (pb[i,j,:] + pb[j,i,:])
3. Antisymmetric-only: A = 0.5 * (pb[i,j,:] - pb[j,i,:])
4. Symmetric+Antisymmetric: [S, A]

Outputs (inside repo proteins/multi_protein_results/symmetric_antisymmetric_pair_features/):
    csv_files/layer_r2_comparison.csv
    csv_files/selected_protein_pair_counts.csv
    csv_files/layer_regression_coefficients_comparison.csv
    csv_files/predicted_distances_layer47_comparison.csv
    visualizations/layer_r2_comparison.png
    visualizations/pred_vs_actual_comparison_layer47.png
"""

import csv
import math
import os
import sys
import tempfile
from pathlib import Path

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))
os.environ.setdefault("XDG_CACHE_HOME", os.path.join(tempfile.gettempdir(), "xdg-cache"))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

sys.path.insert(
    0,
    os.path.dirname(
        os.path.dirname(
            os.path.dirname(
                os.path.dirname(os.path.abspath(__file__))
            )
        )
    ),
)
from lib.data_loading import discover_proteins, find_pdb_file, get_pair_block_path
from lib.structure import extract_ca_coordinates, compute_distance_matrix


# ===========================================================================
# Configuration
# ===========================================================================

SELECTED_PROTEIN_IDS = ["6tf4_A", "6wmk_A", "6wqc_A", "6xqj_A"]
N_LAYERS = 48
N_CHANNELS = 128
BASELINE_FEATURE_DIM = 2 * N_CHANNELS
DECOMPOSED_FEATURE_DIM = 2 * N_CHANNELS
REDUCED_FEATURE_DIM = N_CHANNELS
COEFFICIENT_EXPORT_DIM = max(BASELINE_FEATURE_DIM, DECOMPOSED_FEATURE_DIM)
TEST_SIZE = 0.2
RANDOM_STATE = 42
DECOMPOSITION_ATOL = 5e-5
MODEL_ORDER = [
    "baseline",
    "symmetric_only",
    "antisymmetric_only",
    "symmetric_antisymmetric",
]
MODEL_LABELS = {
    "baseline": "Baseline",
    "symmetric_only": "Symmetric Only",
    "antisymmetric_only": "Antisymmetric Only",
    "symmetric_antisymmetric": "Symmetric + Antisymmetric",
}
MODEL_COLORS = {
    "baseline": "steelblue",
    "symmetric_only": "darkorange",
    "antisymmetric_only": "forestgreen",
    "symmetric_antisymmetric": "firebrick",
}


# ===========================================================================
# Paths
# ===========================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "protein_data"
OUTPUT_ROOT = PROJECT_ROOT / "proteins" / "multi_protein_results"
OUTPUT_DIR = OUTPUT_ROOT / "symmetric_antisymmetric_pair_features"


# ===========================================================================
# Helpers
# ===========================================================================

def require_selected_proteins(all_proteins):
    """Return the selected proteins in the configured order or raise."""
    protein_map = {protein_id: protein_dir for protein_id, protein_dir in all_proteins}
    missing = [protein_id for protein_id in SELECTED_PROTEIN_IDS if protein_id not in protein_map]
    if missing:
        missing_str = ", ".join(missing)
        raise RuntimeError(f"Missing selected protein directories: {missing_str}")
    return [(protein_id, protein_map[protein_id]) for protein_id in SELECTED_PROTEIN_IDS]


def build_ordered_features(pb, row_idx, col_idx):
    """Return ordered upper/lower triangle feature blocks for all upper-triangle pairs."""
    upper = pb[row_idx, col_idx, :]
    lower = pb[col_idx, row_idx, :]
    return upper, lower


def build_symmetric_features(upper, lower):
    """Return the symmetric component S = 0.5 * (upper + lower)."""
    return 0.5 * (upper + lower)


def build_antisymmetric_features(upper, lower):
    """Return the antisymmetric component A = 0.5 * (upper - lower)."""
    return 0.5 * (upper - lower)


def validate_decomposition(upper, lower, symmetric, antisymmetric, baseline, combined):
    """Check shapes and algebraic reconstruction for the decomposition."""
    n_pairs = upper.shape[0]

    if upper.shape != (n_pairs, N_CHANNELS):
        raise RuntimeError(f"Upper feature shape mismatch: expected ({n_pairs}, {N_CHANNELS}), got {upper.shape}")
    if lower.shape != (n_pairs, N_CHANNELS):
        raise RuntimeError(f"Lower feature shape mismatch: expected ({n_pairs}, {N_CHANNELS}), got {lower.shape}")
    if symmetric.shape != (n_pairs, N_CHANNELS):
        raise RuntimeError(
            f"Symmetric feature shape mismatch: expected ({n_pairs}, {N_CHANNELS}), got {symmetric.shape}"
        )
    if antisymmetric.shape != (n_pairs, N_CHANNELS):
        raise RuntimeError(
            f"Antisymmetric feature shape mismatch: expected ({n_pairs}, {N_CHANNELS}), got {antisymmetric.shape}"
        )
    if baseline.shape != (n_pairs, BASELINE_FEATURE_DIM):
        raise RuntimeError(
            f"Baseline feature shape mismatch: expected ({n_pairs}, {BASELINE_FEATURE_DIM}), got {baseline.shape}"
        )
    if combined.shape != (n_pairs, DECOMPOSED_FEATURE_DIM):
        raise RuntimeError(
            f"Combined feature shape mismatch: expected ({n_pairs}, {DECOMPOSED_FEATURE_DIM}), got {combined.shape}"
        )

    if not np.allclose(upper, symmetric + antisymmetric, atol=DECOMPOSITION_ATOL):
        raise RuntimeError("Symmetric/antisymmetric decomposition failed to reconstruct upper features.")
    if not np.allclose(lower, symmetric - antisymmetric, atol=DECOMPOSITION_ATOL):
        raise RuntimeError("Symmetric/antisymmetric decomposition failed to reconstruct lower features.")


def fit_probe(X_all, idx_train, idx_test, y_train, y_test):
    """Fit a standardized linear probe and return metrics plus the fitted objects."""
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[idx_train])
    X_test = scaler.transform(X_all[idx_test])

    model = LinearRegression()
    model.fit(X_train, y_train)
    y_pred_test = model.predict(X_test)
    r2 = r2_score(y_test, y_pred_test)
    return scaler, model, r2


# ===========================================================================
# Main
# ===========================================================================

def main():
    proteins_dir = DATA_ROOT.resolve()
    output_dir = OUTPUT_DIR.resolve()
    csv_dir = output_dir / "csv_files"
    vis_dir = output_dir / "visualizations"
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Protein data root:  {proteins_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Selected proteins:  {', '.join(SELECTED_PROTEIN_IDS)}")
    print(f"Layers:             {N_LAYERS}")
    print(f"Split strategy:     pooled pair-level {1 - TEST_SIZE:.1%}/{TEST_SIZE:.1%}")

    all_proteins = discover_proteins(proteins_dir)
    selected_proteins = require_selected_proteins(all_proteins)

    protein_data = []
    pair_count_rows = []
    for protein_id, protein_dir in selected_proteins:
        pdb_path = find_pdb_file(protein_dir, protein_id)
        if pdb_path is None:
            raise RuntimeError(f"Selected protein {protein_id} has no predicted PDB file.")

        pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
        if not pb0_path.exists():
            raise RuntimeError(f"Selected protein {protein_id} is missing pair block 0: {pb0_path}")

        ca_data = extract_ca_coordinates(pdb_path)
        n_ca = len(ca_data["residues"])
        if n_ca == 0:
            raise RuntimeError(f"Selected protein {protein_id} has no CA atoms in {pdb_path}.")

        pb0 = np.load(str(pb0_path), mmap_mode="r")
        pb_dim = pb0.shape[0]
        del pb0

        if n_ca != pb_dim:
            raise RuntimeError(
                f"Selected protein {protein_id} has CA count {n_ca} but pair block dimension {pb_dim}."
            )

        dist_matrix = compute_distance_matrix(ca_data["coordinates"])
        pb_i, pb_j = np.triu_indices(n_ca, k=1)
        y_protein = dist_matrix[pb_i, pb_j]

        protein_data.append(
            {
                "protein_id": protein_id,
                "protein_dir": protein_dir,
                "n_residues": n_ca,
                "pb_i": pb_i,
                "pb_j": pb_j,
                "y": y_protein,
            }
        )
        pair_count_rows.append(
            {
                "protein_id": protein_id,
                "n_residues": n_ca,
                "n_pairs_upper_triangle": len(y_protein),
            }
        )
        print(f"[OK]   {protein_id}: {n_ca} residues, {len(y_protein):,} upper-triangle pairs")

    y_all = np.concatenate([d["y"] for d in protein_data])
    n_total_pairs = len(y_all)
    n_proteins = len(protein_data)

    protein_offsets = []
    offset = 0
    for d in protein_data:
        n_pairs = len(d["y"])
        protein_offsets.append((d["protein_id"], offset, offset + n_pairs))
        offset += n_pairs

    all_idx = np.arange(n_total_pairs)
    idx_train, idx_test = train_test_split(
        all_idx,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    y_train = y_all[idx_train]
    y_test = y_all[idx_test]

    split_labels = np.empty(n_total_pairs, dtype=object)
    split_labels[idx_train] = "train"
    split_labels[idx_test] = "test"

    results = {model_name: [] for model_name in MODEL_ORDER}
    coefficient_rows = []
    last_layer_preds = {}

    print(f"\nTotal pairs across {n_proteins} proteins: {n_total_pairs:,}")
    print()

    for layer in range(N_LAYERS):
        feature_chunks = {model_name: [] for model_name in MODEL_ORDER}

        for d in protein_data:
            pb_path = get_pair_block_path(d["protein_dir"], d["protein_id"], layer)
            pb = np.load(str(pb_path))

            upper, lower = build_ordered_features(pb, d["pb_i"], d["pb_j"])
            baseline_features = np.concatenate([upper, lower], axis=1)
            symmetric_features = build_symmetric_features(upper, lower)
            antisymmetric_features = build_antisymmetric_features(upper, lower)
            combined_features = np.concatenate([symmetric_features, antisymmetric_features], axis=1)

            validate_decomposition(
                upper,
                lower,
                symmetric_features,
                antisymmetric_features,
                baseline_features,
                combined_features,
            )

            feature_chunks["baseline"].append(baseline_features)
            feature_chunks["symmetric_only"].append(symmetric_features)
            feature_chunks["antisymmetric_only"].append(antisymmetric_features)
            feature_chunks["symmetric_antisymmetric"].append(combined_features)
            del pb

        feature_matrices = {
            model_name: np.vstack(chunks)
            for model_name, chunks in feature_chunks.items()
        }

        layer_r2_strings = []
        for model_name in MODEL_ORDER:
            X_all = feature_matrices[model_name]
            scaler, model, r2 = fit_probe(X_all, idx_train, idx_test, y_train, y_test)
            results[model_name].append(r2)
            layer_r2_strings.append(f"{model_name}: {r2:.4f}")

            coefficient_rows.append(
                {
                    "layer": layer,
                    "model": model_name,
                    "feature_dim": int(X_all.shape[1]),
                    "intercept": float(model.intercept_),
                    "coefs": [float(value) for value in model.coef_],
                }
            )

            if layer == N_LAYERS - 1:
                last_layer_preds[model_name] = model.predict(scaler.transform(X_all))

        print(f"Layer {layer:2d}  {'  '.join(layer_r2_strings)}")
        del feature_matrices

    r2_csv = csv_dir / "layer_r2_comparison.csv"
    with open(r2_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "layer",
                "baseline_r2",
                "symmetric_only_r2",
                "antisymmetric_only_r2",
                "symmetric_antisymmetric_r2",
                "symmetric_minus_baseline",
                "antisymmetric_minus_baseline",
                "combined_minus_baseline",
                "n_proteins",
                "n_total_pairs",
            ]
        )
        for layer in range(N_LAYERS):
            baseline_r2 = results["baseline"][layer]
            symmetric_r2 = results["symmetric_only"][layer]
            antisymmetric_r2 = results["antisymmetric_only"][layer]
            combined_r2 = results["symmetric_antisymmetric"][layer]
            writer.writerow(
                [
                    layer,
                    f"{baseline_r2:.6f}",
                    f"{symmetric_r2:.6f}",
                    f"{antisymmetric_r2:.6f}",
                    f"{combined_r2:.6f}",
                    f"{symmetric_r2 - baseline_r2:.6f}",
                    f"{antisymmetric_r2 - baseline_r2:.6f}",
                    f"{combined_r2 - baseline_r2:.6f}",
                    n_proteins,
                    n_total_pairs,
                ]
            )
    print(f"\nSaved R^2 CSV -> {r2_csv}")

    pair_count_csv = csv_dir / "selected_protein_pair_counts.csv"
    with open(pair_count_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "n_residues", "n_pairs_upper_triangle"])
        for row in pair_count_rows:
            writer.writerow([row["protein_id"], row["n_residues"], row["n_pairs_upper_triangle"]])
    print(f"Saved pair count CSV -> {pair_count_csv}")

    coef_csv = csv_dir / "layer_regression_coefficients_comparison.csv"
    coef_header = [
        "layer",
        "model",
        "feature_dim",
        "intercept",
    ] + [f"coef_{i}" for i in range(COEFFICIENT_EXPORT_DIM)]
    with open(coef_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(coef_header)
        for row in coefficient_rows:
            out_row = [
                row["layer"],
                row["model"],
                row["feature_dim"],
                row["intercept"],
            ]
            out_row.extend(row["coefs"])
            if len(row["coefs"]) < COEFFICIENT_EXPORT_DIM:
                out_row.extend([""] * (COEFFICIENT_EXPORT_DIM - len(row["coefs"])))
            writer.writerow(out_row)
    print(f"Saved coefficient CSV -> {coef_csv}")

    pred_csv = csv_dir / "predicted_distances_layer47_comparison.csv"
    with open(pred_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            [
                "protein_id",
                "struct_i",
                "struct_j",
                "actual_distance_A",
                "baseline_predicted_A",
                "symmetric_only_predicted_A",
                "antisymmetric_only_predicted_A",
                "symmetric_antisymmetric_predicted_A",
                "split",
            ]
        )
        for protein_id, start, end in protein_offsets:
            protein_record = next(d for d in protein_data if d["protein_id"] == protein_id)
            for local_idx in range(end - start):
                global_idx = start + local_idx
                writer.writerow(
                    [
                        protein_id,
                        int(protein_record["pb_i"][local_idx]),
                        int(protein_record["pb_j"][local_idx]),
                        f"{y_all[global_idx]:.4f}",
                        f"{last_layer_preds['baseline'][global_idx]:.4f}",
                        f"{last_layer_preds['symmetric_only'][global_idx]:.4f}",
                        f"{last_layer_preds['antisymmetric_only'][global_idx]:.4f}",
                        f"{last_layer_preds['symmetric_antisymmetric'][global_idx]:.4f}",
                        split_labels[global_idx],
                    ]
                )
    print(f"Saved prediction CSV -> {pred_csv}")

    fig, ax = plt.subplots(figsize=(10, 5))
    for model_name in MODEL_ORDER:
        ax.plot(
            range(N_LAYERS),
            results[model_name],
            marker="o",
            markersize=3,
            linewidth=1.5,
            color=MODEL_COLORS[model_name],
            label=MODEL_LABELS[model_name],
        )
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test R^2")
    ax.set_title(f"Symmetric vs Antisymmetric Pair Probe ({n_proteins} proteins, {n_total_pairs:,} pairs)")
    ax.set_ylim(0, 1)
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    layer_plot = vis_dir / "layer_r2_comparison.png"
    fig.savefig(layer_plot, dpi=150)
    plt.close(fig)
    print(f"Saved R^2 plot -> {layer_plot}")

    fig, axes = plt.subplots(1, len(MODEL_ORDER), figsize=(7 * len(MODEL_ORDER), 6), squeeze=False)
    lim_max = y_all.max() * 1.05
    for idx, model_name in enumerate(MODEL_ORDER):
        ax = axes[0][idx]
        y_pred = last_layer_preds[model_name][idx_test]
        ax.scatter(
            y_test,
            y_pred,
            s=2,
            alpha=0.3,
            color=MODEL_COLORS[model_name],
            rasterized=True,
        )
        ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1)
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_xlabel("Actual CA-CA Distance (A)")
        ax.set_ylabel("Predicted CA-CA Distance (A)")
        ax.set_title(f"{MODEL_LABELS[model_name]}\nR^2 = {results[model_name][-1]:.4f}")
        ax.set_aspect("equal")
        ax.grid(True, alpha=0.3)
    fig.suptitle("Layer 47: Predicted vs Actual Distance", fontsize=14)
    fig.tight_layout()
    pred_plot = vis_dir / "pred_vs_actual_comparison_layer47.png"
    fig.savefig(pred_plot, dpi=150)
    plt.close(fig)
    print(f"Saved prediction plot -> {pred_plot}")

    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print("Included proteins:")
    for row in pair_count_rows:
        print(
            f"  {row['protein_id']}: {row['n_residues']} residues, "
            f"{row['n_pairs_upper_triangle']:,} pairs"
        )

    print("\nBest layer per model:")
    for model_name in MODEL_ORDER:
        valid_results = [value for value in results[model_name] if not math.isnan(value)]
        if not valid_results:
            continue
        best_layer = int(np.nanargmax(results[model_name]))
        print(f"  {MODEL_LABELS[model_name]:25s} layer {best_layer:2d}  R^2 = {results[model_name][best_layer]:.4f}")

    print("\nLayer 47 R^2:")
    for model_name in MODEL_ORDER:
        print(f"  {MODEL_LABELS[model_name]:25s} R^2 = {results[model_name][-1]:.4f}")

    best_symmetric_layer = int(np.nanargmax(results["symmetric_only"]))
    best_antisymmetric_layer = int(np.nanargmax(results["antisymmetric_only"]))
    best_combined_layer = int(np.nanargmax(results["symmetric_antisymmetric"]))
    print("\nDelta vs baseline:")
    print(
        f"  Best symmetric-only layer {best_symmetric_layer:2d}: "
        f"{results['symmetric_only'][best_symmetric_layer] - results['baseline'][best_symmetric_layer]:+.4f}"
    )
    print(
        f"  Best antisymmetric-only layer {best_antisymmetric_layer:2d}: "
        f"{results['antisymmetric_only'][best_antisymmetric_layer] - results['baseline'][best_antisymmetric_layer]:+.4f}"
    )
    print(
        f"  Best combined layer {best_combined_layer:2d}: "
        f"{results['symmetric_antisymmetric'][best_combined_layer] - results['baseline'][best_combined_layer]:+.4e}"
    )
    print(
        f"  Layer 47 symmetric-only vs baseline: "
        f"{results['symmetric_only'][-1] - results['baseline'][-1]:+.4f}"
    )
    print(
        f"  Layer 47 antisymmetric-only vs baseline: "
        f"{results['antisymmetric_only'][-1] - results['baseline'][-1]:+.4f}"
    )
    print(
        f"  Layer 47 combined vs baseline (should be ~0): "
        f"{results['symmetric_antisymmetric'][-1] - results['baseline'][-1]:+.4e}"
    )


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
neighbor_context_pair_regression.py
==================================
Compare three pooled linear probes for predicting CA-CA distances from
AlphaFold pair representations on a fixed 4-protein subset:

1. Baseline: [pb[i,j,:], pb[j,i,:]]
2. Neighbor-context: baseline + four orthogonal neighbor cells
3. Fixed-random control: baseline + four fixed random off-diagonal cells

Edge pairs are retained. Missing orthogonal neighbors are zero-padded.

Outputs (inside repo proteins/multi_protein_results/neighbor_context_pair_features/):
    csv_files/layer_r2_comparison.csv
    csv_files/selected_protein_pair_counts.csv
    csv_files/random_control_cells.csv
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
AUGMENTED_FEATURE_DIM = 6 * N_CHANNELS
TEST_SIZE = 0.2
RANDOM_STATE = 42
NEIGHBOR_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]
MODEL_ORDER = ["baseline", "neighbor", "random_fixed"]
MODEL_LABELS = {
    "baseline": "Baseline",
    "neighbor": "Neighbor Context",
    "random_fixed": "Fixed Random Control",
}
MODEL_COLORS = {
    "baseline": "steelblue",
    "neighbor": "darkorange",
    "random_fixed": "forestgreen",
}


# ===========================================================================
# Paths
# ===========================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_ROOT = PROJECT_ROOT.parent / "protein_data"
OUTPUT_ROOT = PROJECT_ROOT / "proteins" / "multi_protein_results"
OUTPUT_DIR = OUTPUT_ROOT / "neighbor_context_pair_features"


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


def extract_neighbor_block(pb, row_idx, col_idx, delta_row, delta_col):
    """Extract a neighbor block with zero-padding for out-of-bounds cells."""
    n_pairs = len(row_idx)
    n_residues = pb.shape[0]
    rows = row_idx + delta_row
    cols = col_idx + delta_col

    out = np.zeros((n_pairs, N_CHANNELS), dtype=pb.dtype)
    valid = (
        (rows >= 0)
        & (rows < n_residues)
        & (cols >= 0)
        & (cols < n_residues)
    )
    if np.any(valid):
        out[valid] = pb[rows[valid], cols[valid], :]
    return out


def build_baseline_features(pb, row_idx, col_idx):
    """Build the 256-dim baseline feature matrix for all upper-triangle pairs."""
    return np.concatenate(
        [
            pb[row_idx, col_idx, :],
            pb[col_idx, row_idx, :],
        ],
        axis=1,
    )


def build_neighbor_features(pb, row_idx, col_idx, baseline_features):
    """Append orthogonal neighbor cells to the baseline feature matrix."""
    blocks = [baseline_features]
    for delta_row, delta_col in NEIGHBOR_OFFSETS:
        blocks.append(extract_neighbor_block(pb, row_idx, col_idx, delta_row, delta_col))
    return np.concatenate(blocks, axis=1)


def build_random_features(pb, baseline_features, random_cells, n_pairs):
    """Append four fixed random off-diagonal cells to the baseline feature matrix."""
    control_blocks = [pb[row_idx, col_idx, :] for row_idx, col_idx in random_cells]
    control_flat = np.concatenate(control_blocks, axis=0).reshape(1, -1)
    repeated_control = np.repeat(control_flat, n_pairs, axis=0)
    return np.concatenate([baseline_features, repeated_control], axis=1)


def build_random_cells(n_residues, protein_index):
    """Sample deterministic fixed-random control cells for each layer."""
    row_idx, col_idx = np.where(~np.eye(n_residues, dtype=bool))
    n_choices = len(row_idx)
    if n_choices < 4:
        raise RuntimeError(
            f"Protein with {n_residues} residues has only {n_choices} off-diagonal cells; need at least 4."
        )

    random_cells = {}
    for layer in range(N_LAYERS):
        rng = np.random.default_rng(RANDOM_STATE + 1000 * protein_index + layer)
        picked = rng.choice(n_choices, size=4, replace=False)
        random_cells[layer] = [
            (int(row_idx[idx]), int(col_idx[idx]))
            for idx in picked
        ]
    return random_cells


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
    for protein_index, (protein_id, protein_dir) in enumerate(selected_proteins):
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
        random_cells = build_random_cells(n_ca, protein_index)

        protein_data.append(
            {
                "protein_id": protein_id,
                "protein_dir": protein_dir,
                "n_residues": n_ca,
                "pb_i": pb_i,
                "pb_j": pb_j,
                "y": y_protein,
                "random_cells": random_cells,
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
            baseline_features = build_baseline_features(pb, d["pb_i"], d["pb_j"])
            neighbor_features = build_neighbor_features(pb, d["pb_i"], d["pb_j"], baseline_features)
            random_features = build_random_features(
                pb,
                baseline_features,
                d["random_cells"][layer],
                len(d["y"]),
            )

            feature_chunks["baseline"].append(baseline_features)
            feature_chunks["neighbor"].append(neighbor_features)
            feature_chunks["random_fixed"].append(random_features)
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
                "neighbor_r2",
                "random_fixed_r2",
                "neighbor_minus_baseline",
                "random_minus_baseline",
                "neighbor_minus_random",
                "n_proteins",
                "n_total_pairs",
            ]
        )
        for layer in range(N_LAYERS):
            baseline_r2 = results["baseline"][layer]
            neighbor_r2 = results["neighbor"][layer]
            random_r2 = results["random_fixed"][layer]
            writer.writerow(
                [
                    layer,
                    f"{baseline_r2:.6f}",
                    f"{neighbor_r2:.6f}",
                    f"{random_r2:.6f}",
                    f"{neighbor_r2 - baseline_r2:.6f}",
                    f"{random_r2 - baseline_r2:.6f}",
                    f"{neighbor_r2 - random_r2:.6f}",
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

    random_cells_csv = csv_dir / "random_control_cells.csv"
    with open(random_cells_csv, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["protein_id", "layer", "cell_rank", "row_idx", "col_idx"])
        for d in protein_data:
            for layer in range(N_LAYERS):
                for cell_rank, (row_idx, col_idx) in enumerate(d["random_cells"][layer], start=1):
                    writer.writerow([d["protein_id"], layer, cell_rank, row_idx, col_idx])
    print(f"Saved random control CSV -> {random_cells_csv}")

    coef_csv = csv_dir / "layer_regression_coefficients_comparison.csv"
    coef_header = [
        "layer",
        "model",
        "feature_dim",
        "intercept",
    ] + [f"coef_{i}" for i in range(AUGMENTED_FEATURE_DIM)]
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
            if len(row["coefs"]) < AUGMENTED_FEATURE_DIM:
                out_row.extend([""] * (AUGMENTED_FEATURE_DIM - len(row["coefs"])))
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
                "neighbor_predicted_A",
                "random_fixed_predicted_A",
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
                        f"{last_layer_preds['neighbor'][global_idx]:.4f}",
                        f"{last_layer_preds['random_fixed'][global_idx]:.4f}",
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
    ax.set_title(f"Neighbor Context Pair Probe ({n_proteins} proteins, {n_total_pairs:,} pairs)")
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
        print(f"  {MODEL_LABELS[model_name]:20s} layer {best_layer:2d}  R^2 = {results[model_name][best_layer]:.4f}")

    print("\nLayer 47 R^2:")
    for model_name in MODEL_ORDER:
        print(f"  {MODEL_LABELS[model_name]:20s} R^2 = {results[model_name][-1]:.4f}")

    best_neighbor_layer = int(np.nanargmax(results["neighbor"]))
    print("\nNeighbor gains:")
    print(
        f"  Best neighbor layer {best_neighbor_layer:2d}: "
        f"vs baseline = {results['neighbor'][best_neighbor_layer] - results['baseline'][best_neighbor_layer]:+.4f}, "
        f"vs random = {results['neighbor'][best_neighbor_layer] - results['random_fixed'][best_neighbor_layer]:+.4f}"
    )
    print(
        f"  Layer 47: "
        f"vs baseline = {results['neighbor'][-1] - results['baseline'][-1]:+.4f}, "
        f"vs random = {results['neighbor'][-1] - results['random_fixed'][-1]:+.4f}"
    )


if __name__ == "__main__":
    main()

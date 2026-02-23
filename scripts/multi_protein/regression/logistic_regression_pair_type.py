#!/usr/bin/env python3
"""
logistic_regression_pair_type.py
================================
Local script that trains a logistic regression classifier to predict
amino acid pair type from AlphaFold pair representations across 48 layers.

Pooled model: one logistic regression per layer, trained on all proteins combined.
For each residue pair (i, j), the label is the alphabetically-sorted amino acid
pair type (e.g., "ALA-GLY").  Features are the concatenated pair-block vectors
[pair_block[i,j,:], pair_block[j,i,:]] -> 256-dim.

Dependencies (all pip-installable):
    pip install numpy scipy matplotlib scikit-learn biopython

Outputs (inside proteins/multi_protein_results/logistic_regression/):
    csv_files/layer_balanced_accuracy.csv
    csv_files/per_class_f1_layer47.csv
    csv_files/confusion_matrix_layer47.csv
    csv_files/top_confused_pairs_layer47.csv
    visualizations/layer_balanced_accuracy.png
    visualizations/confusion_matrix_layer47.png
    visualizations/top_confused_pairs_layer47.png
"""

import csv
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)

from Bio.PDB import PDBParser


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48          # AlphaFold pair representation layers
N_CHANNELS = 128       # channels per pair representation direction
FEATURE_DIM = 2 * N_CHANNELS  # 256 (concatenated [i,j] and [j,i])
TEST_SIZE = 0.2
RANDOM_STATE = 42
MAX_ITER = 2000        # LogisticRegression max iterations (210 classes)
TOP_N_CONFUSED = 20    # Number of top confused pairs to report

# ===========================================================================
# Hardcoded directories
# ===========================================================================

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
PROTEINS_DIR = PROJECT_ROOT / "proteins"
OUTPUT_DIR   = PROTEINS_DIR / "multi_protein_results" / "logistic_regression"


# ===========================================================================
# Helper functions — data loading (copied from multi_protein_regression.py)
# ===========================================================================

def find_pdb_file(protein_dir, protein_id):
    """Locate the predicted PDB file for a protein.

    Expected path: <protein_dir>/predicted_pdb/<protein_id>_model_1_ptm_unrelaxed.pdb

    Returns the Path object, or None if not found.
    """
    pdb_path = protein_dir / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    return pdb_path if pdb_path.exists() else None


def get_pair_block_path(protein_dir, protein_id, layer):
    """Return the path to a pair block .npy file for a given layer.

    Expected: <protein_dir>/pair_blocks/<protein_id>_pair_block_<layer>.npy
    """
    return protein_dir / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy"


def discover_proteins(proteins_dir):
    """Find all protein subdirectories under the given root.

    Returns a sorted list of (protein_id, protein_dir) tuples.
    Skips 'multi_protein_results' and any non-directory entries.
    """
    proteins = []
    for entry in sorted(proteins_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "multi_protein_results":
            continue
        proteins.append((entry.name, entry))
    return proteins


def extract_ca_coordinates(pdb_path, chain_id):
    """Extract CA atom coordinates from a PDB file.

    Extracts ALL residues with a CA atom (no standard-residue filtering).

    Parameters
    ----------
    pdb_path : str or Path
        Path to the PDB file.
    chain_id : str
        Chain identifier to extract (e.g. 'A', 'B').

    Returns
    -------
    dict with keys:
        'residues'    : list of (chain_id, residue_name, residue_number) tuples
        'coordinates' : np.ndarray of shape (n_residues, 3)
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", str(pdb_path))

    residues = []
    coordinates = []

    model = structure[0]
    for chain in model:
        if chain.get_id() != chain_id:
            continue
        for residue in chain:
            het_flag = residue.get_id()[0]
            if het_flag == "W":
                continue
            if "CA" in residue:
                ca = residue["CA"]
                residues.append((
                    chain.get_id(),
                    residue.get_resname(),
                    residue.get_id()[1],
                ))
                coordinates.append(ca.get_coord())

    return {
        "residues": residues,
        "coordinates": np.array(coordinates) if coordinates else np.empty((0, 3)),
    }


# ===========================================================================
# Helper functions — pair type labels
# ===========================================================================

def build_pair_labels(residue_data):
    """Build sorted amino acid pair type labels for all upper-triangle pairs.

    Parameters
    ----------
    residue_data : dict
        Output of extract_ca_coordinates(). Has 'residues' key with list of
        (chain_id, residue_name, residue_number) tuples.

    Returns
    -------
    labels : np.ndarray of str, shape (n_pairs,)
        Sorted pair type labels like "ALA-GLY" for each upper-triangle pair.
    struct_i : np.ndarray of int
    struct_j : np.ndarray of int
    """
    res_names = [r[1] for r in residue_data["residues"]]
    n = len(res_names)
    struct_i, struct_j = np.triu_indices(n, k=1)

    labels = np.array([
        "-".join(sorted([res_names[i], res_names[j]]))
        for i, j in zip(struct_i, struct_j)
    ])

    return labels, struct_i, struct_j


def extract_top_confused_pairs(cm, class_names, top_n=20):
    """Extract the top-N most confused off-diagonal pairs from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray, shape (n_classes, n_classes)
    class_names : list of str
    top_n : int

    Returns
    -------
    list of dicts with keys: true_class, predicted_class, count, true_total
    """
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)

    flat_indices = np.argsort(cm_off.ravel())[::-1][:top_n]
    rows, cols = np.unravel_index(flat_indices, cm.shape)

    results = []
    for r, c in zip(rows, cols):
        if cm_off[r, c] == 0:
            break
        results.append({
            "true_class": class_names[r],
            "predicted_class": class_names[c],
            "count": int(cm_off[r, c]),
            "true_total": int(cm[r].sum()),
        })
    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    proteins_dir = PROTEINS_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    n_layers = N_LAYERS

    csv_dir = output_dir / "csv_files"
    vis_dir = output_dir / "visualizations"
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Proteins directory: {proteins_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Layers:             {n_layers}")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------
    all_proteins = discover_proteins(proteins_dir)
    print(f"\nDiscovered {len(all_proteins)} protein directories.")

    protein_data = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        try:
            pdb_path = find_pdb_file(protein_dir, protein_id)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                print(f"[SKIP] {protein_id}: no predicted PDB file found")
                continue

            pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
            if not pb0_path.exists():
                skipped.append((protein_id, "pair block 0 not found"))
                print(f"[SKIP] {protein_id}: pair block 0 not found at {pb0_path}")
                continue

            chain_id = protein_id.split("_")[-1]

            ca_data = extract_ca_coordinates(pdb_path, chain_id=chain_id)
            n_ca = len(ca_data["residues"])
            if n_ca == 0:
                skipped.append((protein_id, "no CA atoms found in PDB"))
                print(f"[SKIP] {protein_id}: no CA atoms found in PDB")
                continue

            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = pb0.shape[0]
            del pb0

            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                print(f"[SKIP] {protein_id}: CA count {n_ca} != pair block dim {pb_dim}")
                continue

            labels, struct_i, struct_j = build_pair_labels(ca_data)
            n_pair_types = len(set(labels))

            protein_data.append({
                "protein_id": protein_id,
                "protein_dir": protein_dir,
                "n_residues": n_ca,
                "pb_i": struct_i,
                "pb_j": struct_j,
                "labels": labels,
            })
            print(f"[OK]   {protein_id}: {n_ca} residues, {len(labels):,} pairs, {n_pair_types} pair types")

        except Exception as e:
            skipped.append((protein_id, f"unexpected error -- {e}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {e}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    n_proteins = len(protein_data)

    # ------------------------------------------------------------------
    # Phase 2: Pool labels, encode, and split
    # ------------------------------------------------------------------
    labels_all = np.concatenate([d["labels"] for d in protein_data])
    n_total_pairs = len(labels_all)

    protein_offsets = []
    offset = 0
    for d in protein_data:
        n = len(d["labels"])
        protein_offsets.append((d["protein_id"], offset, offset + n))
        offset += n

    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(labels_all)
    n_classes = len(label_encoder.classes_)
    class_names = list(label_encoder.classes_)

    print(f"\nTotal pairs across {n_proteins} protein(s): {n_total_pairs:,}")
    print(f"Unique pair types: {n_classes}")

    all_idx = np.arange(n_total_pairs)
    idx_train, idx_test = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y_all,
    )
    y_train = y_all[idx_train]
    y_test = y_all[idx_test]

    print(f"Train: {len(idx_train):,}  Test: {len(idx_test):,}")

    # ------------------------------------------------------------------
    # Phase 3: Layer-wise classification
    # ------------------------------------------------------------------
    results = []
    last_layer_cm = None
    last_layer_report = None

    print()
    for layer in range(n_layers):
        try:
            X_chunks = []
            for d in protein_data:
                pb_path = get_pair_block_path(d["protein_dir"], d["protein_id"], layer)
                pb = np.load(str(pb_path))
                X_chunk = np.concatenate([
                    pb[d["pb_i"], d["pb_j"], :],
                    pb[d["pb_j"], d["pb_i"], :],
                ], axis=1)
                X_chunks.append(X_chunk)
                del pb

            X_all = np.vstack(X_chunks)
            del X_chunks

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_all[idx_train])
            X_test = scaler.transform(X_all[idx_test])
            del X_all

            model = LogisticRegression(
                C=1e12,
                solver="lbfgs",
                max_iter=MAX_ITER,
            )
            model.fit(X_train, y_train)
            y_pred_test = model.predict(X_test)

            bal_acc = balanced_accuracy_score(y_test, y_pred_test)
            acc = accuracy_score(y_test, y_pred_test)

            results.append({
                "layer": layer,
                "balanced_accuracy": bal_acc,
                "accuracy": acc,
                "n_classes": n_classes,
                "n_proteins": n_proteins,
                "n_total_pairs": n_total_pairs,
            })
            print(f"Layer {layer:2d}  balanced_acc = {bal_acc:.4f}  acc = {acc:.4f}")

            if layer == n_layers - 1:
                last_layer_cm = confusion_matrix(
                    y_test, y_pred_test,
                    labels=np.arange(n_classes),
                )
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_test, y_pred_test,
                    labels=np.arange(n_classes),
                    zero_division=0,
                )
                last_layer_report = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                }

            del X_train, X_test

        except Exception as e:
            print(f"Layer {layer:2d}  ERROR: {e}")
            traceback.print_exc()
            results.append({
                "layer": layer,
                "balanced_accuracy": float("nan"),
                "accuracy": float("nan"),
                "n_classes": n_classes,
                "n_proteins": n_proteins,
                "n_total_pairs": n_total_pairs,
            })

    # ------------------------------------------------------------------
    # Phase 4: Save CSV outputs
    # ------------------------------------------------------------------

    # CSV 1: Layer balanced accuracy
    ba_csv = csv_dir / "layer_balanced_accuracy.csv"
    with open(ba_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "balanced_accuracy", "accuracy", "n_classes", "n_proteins", "n_total_pairs"])
        for r in results:
            writer.writerow([
                r["layer"], f"{r['balanced_accuracy']:.6f}", f"{r['accuracy']:.6f}",
                r["n_classes"], r["n_proteins"], r["n_total_pairs"],
            ])
    print(f"\nSaved balanced accuracy CSV -> {ba_csv}")

    # CSV 2: Per-class F1 (layer 47)
    if last_layer_report is not None:
        f1_csv = csv_dir / "per_class_f1_layer47.csv"
        with open(f1_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_type", "f1", "precision", "recall", "support"])
            for i in range(n_classes):
                sup = int(last_layer_report["support"][i])
                if sup == 0:
                    continue
                writer.writerow([
                    class_names[i],
                    f"{last_layer_report['f1'][i]:.4f}",
                    f"{last_layer_report['precision'][i]:.4f}",
                    f"{last_layer_report['recall'][i]:.4f}",
                    sup,
                ])
        print(f"Saved per-class F1 CSV -> {f1_csv}")

    # CSV 3: Confusion matrix (layer 47)
    if last_layer_cm is not None:
        cm_csv = csv_dir / "confusion_matrix_layer47.csv"
        with open(cm_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + class_names)
            for i, row in enumerate(last_layer_cm):
                writer.writerow([class_names[i]] + [int(x) for x in row])
        print(f"Saved confusion matrix CSV -> {cm_csv}")

    # CSV 4: Top confused pairs (layer 47)
    if last_layer_cm is not None:
        top_confused = extract_top_confused_pairs(last_layer_cm, class_names, TOP_N_CONFUSED)
        tc_csv = csv_dir / "top_confused_pairs_layer47.csv"
        with open(tc_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["true_class", "predicted_class", "count", "true_total"])
            for row in top_confused:
                writer.writerow([row["true_class"], row["predicted_class"], row["count"], row["true_total"]])
        print(f"Saved top confused pairs CSV -> {tc_csv}")

    # ------------------------------------------------------------------
    # Phase 5: Visualizations
    # ------------------------------------------------------------------

    # Plot 1: Balanced accuracy vs layer
    bal_accs = [r["balanced_accuracy"] for r in results]
    accs = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_layers), bal_accs, marker="o", markersize=3, linewidth=1.5, label="Balanced Accuracy")
    ax.plot(range(n_layers), accs, marker="s", markersize=3, linewidth=1.5, alpha=0.7, label="Raw Accuracy")
    ax.axhline(y=1.0 / n_classes, color="gray", linestyle="--", linewidth=1, label=f"Random baseline ({1.0/n_classes:.4f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Layer-wise Logistic Regression ({n_proteins} proteins, {n_total_pairs:,} pairs, {n_classes} classes)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "layer_balanced_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Saved accuracy plot -> {vis_dir / 'layer_balanced_accuracy.png'}")

    # Plot 2: Full confusion matrix heatmap (layer 47, row-normalized)
    if last_layer_cm is not None:
        row_sums = last_layer_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1  # avoid division by zero
        cm_norm = last_layer_cm / row_sums

        fig, ax = plt.subplots(figsize=(24, 22))
        im = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="Fraction of true class", shrink=0.8)

        tick_step = 10
        tick_positions = list(range(0, n_classes, tick_step))
        tick_labels = [class_names[i] for i in tick_positions]
        ax.set_xticks(tick_positions)
        ax.set_xticklabels(tick_labels, rotation=90, fontsize=6)
        ax.set_yticks(tick_positions)
        ax.set_yticklabels(tick_labels, fontsize=6)

        ax.set_xlabel("Predicted pair type")
        ax.set_ylabel("True pair type")
        ax.set_title(f"Confusion Matrix — Layer 47 ({n_classes} classes, row-normalized)")
        fig.tight_layout()
        fig.savefig(vis_dir / "confusion_matrix_layer47.png", dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix plot -> {vis_dir / 'confusion_matrix_layer47.png'}")

    # Plot 3: Top confused pairs bar chart
    if last_layer_cm is not None and top_confused:
        fig, ax = plt.subplots(figsize=(10, 8))
        bar_labels = [f"{r['true_class']} -> {r['predicted_class']}" for r in top_confused]
        bar_values = [r["count"] for r in top_confused]

        y_pos = range(len(bar_labels))
        ax.barh(y_pos, bar_values, color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_labels, fontsize=9)
        ax.invert_yaxis()
        ax.set_xlabel("Misclassification count")
        ax.set_title(f"Top {len(top_confused)} Most Confused Pair Types — Layer 47")

        for i, v in enumerate(bar_values):
            ax.text(v + 0.5, i, str(v), va="center", fontsize=8)

        ax.grid(True, alpha=0.3, axis="x")
        fig.tight_layout()
        fig.savefig(vis_dir / "top_confused_pairs_layer47.png", dpi=150)
        plt.close(fig)
        print(f"Saved top confused pairs plot -> {vis_dir / 'top_confused_pairs_layer47.png'}")

    # ------------------------------------------------------------------
    # Phase 6: Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Included proteins ({n_proteins}):")
    for pid, start, end in protein_offsets:
        n_res = next(d["n_residues"] for d in protein_data if d["protein_id"] == pid)
        print(f"  {pid}: {n_res} residues, {end - start:,} pairs")
    if skipped:
        print(f"\nSkipped proteins ({len(skipped)}):")
        for pid, reason in skipped:
            print(f"  {pid}: {reason}")

    print(f"\nPair types: {n_classes}")
    print(f"Total pairs: {n_total_pairs:,} (train: {len(idx_train):,}, test: {len(idx_test):,})")

    valid_bal_accs = [r["balanced_accuracy"] for r in results if not math.isnan(r["balanced_accuracy"])]
    if valid_bal_accs:
        best_layer = int(np.nanargmax([r["balanced_accuracy"] for r in results]))
        print(f"\nBest layer: {best_layer}  balanced_acc = {results[best_layer]['balanced_accuracy']:.4f}")
        print(f"Layer {n_layers - 1}:   balanced_acc = {results[-1]['balanced_accuracy']:.4f}  acc = {results[-1]['accuracy']:.4f}")
        print(f"Random baseline: {1.0 / n_classes:.4f}")


if __name__ == "__main__":
    main()

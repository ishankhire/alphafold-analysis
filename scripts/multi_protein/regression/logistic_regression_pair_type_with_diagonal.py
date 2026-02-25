#!/usr/bin/env python3
"""
logistic_regression_pair_type_with_diagonal.py
==============================================
Local script that trains a logistic regression classifier to predict
amino acid pair type from AlphaFold pair representations across 48 layers.

Same as logistic_regression_pair_type.py, but with DIAGONAL features added.
For each residue pair (i, j), the feature vector is:
    [pair_block[i,j,:], pair_block[j,i,:], pair_block[i,i,:], pair_block[j,j,:]] -> 512-dim.

The diagonal entries pb[i,i,:] and pb[j,j,:] encode each residue's
self-representation, giving the model direct information about each
residue's identity in addition to the cross-pair interaction.

Pooled model: one logistic regression per layer, trained on all proteins combined.

Dependencies (all pip-installable):
    pip install numpy scipy matplotlib scikit-learn biopython

Outputs (inside proteins/multi_protein_results/logistic_regression_with_diagonal/):
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
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    balanced_accuracy_score,
    accuracy_score,
    confusion_matrix,
    precision_recall_fscore_support,
)


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48          # AlphaFold outputs 48 pair representation layers (0-47)
N_CHANNELS = 128       # Each pair block entry has 128 channels per direction
FEATURE_DIM = 4 * N_CHANNELS  # 512: pb[i,j,:] (128) + pb[j,i,:] (128) + pb[i,i,:] (128) + pb[j,j,:] (128)
TEST_SIZE = 0.2        # 20% of all residue pairs held out for test set
RANDOM_STATE = 42      # Fixed seed for reproducible train/test splits
MAX_ITER = 2000        # Max optimizer iterations for LogisticRegression
REG_C = 10             # Inverse regularization strength. Light regularization (C=10)
                       # helps L-BFGS converge faster with 512-dim features while
                       # having negligible impact on accuracy.
TOP_N_CONFUSED = 20    # How many top misclassified pair types to report in output

# ===========================================================================
# Hardcoded directories
# ===========================================================================

# PROJECT_ROOT: Path — resolves to the 'protein/' project root by going up 4 levels
#   from scripts/multi_protein/regression/logistic_regression_pair_type_with_diagonal.py
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# PROTEINS_DIR: Path — contains all protein subdirectories (e.g. 7b3a_A/, 6tf4_A/)
PROTEINS_DIR = PROJECT_ROOT / "proteins"

# OUTPUT_DIR: Path — where all outputs (CSVs, plots) are saved for this script
OUTPUT_DIR   = PROTEINS_DIR / "multi_protein_results" / "logistic_regression_with_diagonal"

sys.path.insert(0, str(PROJECT_ROOT))
from lib.classification_common import (
    build_pair_type_labels as _build_pair_type_labels,
    extract_top_confused_pairs as _extract_top_confused_pairs,
    write_confusion_matrix_csv,
    write_per_class_report_csv,
)
from lib.output_writers import ensure_output_dirs
from lib.protein_dataset import load_local_protein_data
from lib.regression_common import (
    build_layer_feature_matrix,
    concatenate_labels,
    compute_protein_offsets_for_key,
    encode_labels,
    split_indices,
)


# ===========================================================================
# Helper functions — data loading
# ===========================================================================

def find_pdb_file(protein_dir, protein_id):
    """Locate the AlphaFold-predicted PDB file for a protein.

    Parameters
    ----------
    protein_dir : Path
        Directory for this protein (e.g. proteins/7b3a_A/).
    protein_id : str
        Protein identifier (e.g. "7b3a_A").

    Returns
    -------
    Path or None
        Full path to the PDB file if it exists, otherwise None.
        Expected location: <protein_dir>/predicted_pdb/<protein_id>_model_1_ptm_unrelaxed.pdb
    """
    pdb_path = protein_dir / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    return pdb_path if pdb_path.exists() else None


def get_pair_block_path(protein_dir, protein_id, layer):
    """Construct the file path to a pair block .npy file for a given layer.

    Each pair block is a numpy array of shape (R, R, 128) where R is the number
    of residues in the AlphaFold entity sequence. There are 48 layers (0-47).

    Parameters
    ----------
    protein_dir : Path
        Directory for this protein (e.g. proteins/7b3a_A/).
    protein_id : str
        Protein identifier (e.g. "7b3a_A").
    layer : int
        Layer index (0-47).

    Returns
    -------
    Path
        Path to the .npy file. Does NOT check if the file exists.
    """
    return protein_dir / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy"


def discover_proteins(proteins_dir):
    """Scan the proteins/ directory and find all protein subdirectories.

    Iterates over proteins_dir in alphabetical order. Skips non-directories and
    the 'multi_protein_results' folder (which holds output, not protein data).

    Parameters
    ----------
    proteins_dir : Path
        Root proteins directory (e.g. proteins/).

    Returns
    -------
    list of (str, Path)
        Sorted list of (protein_id, protein_dir) tuples.
        Example: [("6tf4_A", Path("proteins/6tf4_A")), ("7b3a_A", Path("proteins/7b3a_A"))]
    """
    proteins = []
    for entry in sorted(proteins_dir.iterdir()):
        if not entry.is_dir():
            continue
        if entry.name == "multi_protein_results":
            continue
        proteins.append((entry.name, entry))
    return proteins


def extract_ca_coordinates(pdb_path):
    """Extract CA (alpha-carbon) atom coordinates and residue metadata from a PDB file.

    Extracts all residues with a CA atom across all chains. Chain filtering is
    unnecessary because each PDB file is already specific to a single protein.
    Water molecules (het_flag == "W") are skipped.
    """
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", str(pdb_path))

    residues = []
    coordinates = []

    model = structure[0]
    for chain in model:
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
    """Build sorted amino acid pair-type labels for every unique residue pair (i, j).

    For a protein with N residues, generates N*(N-1)/2 pairs from the upper triangle
    of the NxN pair matrix (excluding the diagonal, since a residue paired with itself
    is not meaningful). Each pair is labeled by alphabetically sorting the two amino
    acid names and joining with a hyphen, so that e.g. (GLY, ALA) and (ALA, GLY) both
    map to the same label "ALA-GLY".

    Parameters
    ----------
    residue_data : dict
        Output of extract_ca_coordinates(). Must have a 'residues' key containing
        a list of (chain_id, residue_name, residue_number) tuples.

    Returns
    -------
    labels : np.ndarray of str, shape (n_pairs,)
        One label per pair, e.g. ["ALA-GLY", "ALA-VAL", ...].
        Length = N*(N-1)/2 where N = number of residues.
    struct_i : np.ndarray of int, shape (n_pairs,)
        Row indices from np.triu_indices — the first residue index in each pair.
    struct_j : np.ndarray of int, shape (n_pairs,)
        Column indices from np.triu_indices — the second residue index in each pair.
    """
    n = len(residue_data["residues"])
    struct_i, struct_j = np.triu_indices(n, k=1)
    labels = _build_pair_type_labels(residue_data, struct_i, struct_j)
    return labels, struct_i, struct_j


def extract_top_confused_pairs(cm, class_names, top_n=20):
    """Find the most common misclassifications from a confusion matrix.

    Parameters
    ----------
    cm : np.ndarray of int, shape (n_classes, n_classes)
        Raw (unnormalized) confusion matrix.
    class_names : list of str, length n_classes
        Human-readable class name for each index.
    top_n : int
        How many top misclassifications to return. Default 20.

    Returns
    -------
    list of dict
        Each dict has keys: 'true_class', 'predicted_class', 'count', 'true_total'.
        Sorted by count descending.
    """
    return _extract_top_confused_pairs(cm, class_names, top_n=top_n)


# ===========================================================================
# Main
# ===========================================================================

def main():
    proteins_dir = PROTEINS_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    n_layers = N_LAYERS

    csv_dir, vis_dir = ensure_output_dirs(output_dir)

    print(f"Proteins directory: {proteins_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Layers:             {n_layers}")
    print(f"Feature dim:        {FEATURE_DIM} (cross-pair 256 + diagonal 256)")
    print(f"Regularization C:   {REG_C}")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------

    protein_data, skipped = load_local_protein_data(
        proteins_dir,
        include_pair_labels=True,
    )
    print(f"\nDiscovered {len(protein_data) + len(skipped)} protein directories.")
    for d in protein_data:
        n_pair_types = len(set(d["labels"]))
        print(
            f"[OK]   {d['protein_id']}: {d['n_residues']} residues, "
            f"{len(d['labels']):,} pairs, {n_pair_types} pair types"
        )
    for protein_id, reason in skipped:
        print(f"[SKIP] {protein_id}: {reason}")

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    n_proteins = len(protein_data)

    # ------------------------------------------------------------------
    # Phase 2: Pool labels, encode, and split
    # ------------------------------------------------------------------

    labels_all = concatenate_labels(protein_data)
    n_total_pairs = len(labels_all)

    protein_offsets = compute_protein_offsets_for_key(protein_data, "labels")

    y_all, label_encoder = encode_labels(labels_all)
    n_classes = len(label_encoder.classes_)
    class_names = list(label_encoder.classes_)

    print(f"\nTotal pairs across {n_proteins} protein(s): {n_total_pairs:,}")
    print(f"Unique pair types: {n_classes}")

    idx_train, idx_test = split_indices(
        n_samples=n_total_pairs,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y_all,
    )

    y_train = y_all[idx_train]
    y_test = y_all[idx_test]

    print(f"Train: {len(idx_train):,}  Test: {len(idx_test):,}")

    # ------------------------------------------------------------------
    # Phase 3: Layer-wise classification
    # ------------------------------------------------------------------
    # Features per pair (i, j) are now 512-dim:
    #   [pb[i,j,:], pb[j,i,:], pb[i,i,:], pb[j,j,:]]
    # The first 256 dims are the cross-pair features (same as original).
    # The last 256 dims are diagonal self-representations for each residue.

    results = []
    last_layer_cm = None
    last_layer_report = None

    print()
    for layer in range(n_layers):
        try:
            X_all = build_layer_feature_matrix(protein_data, layer, mode="diagonal")

            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_all[idx_train])
            X_test = scaler.transform(X_all[idx_test])
            del X_all

            model = LogisticRegression(
                C=REG_C,
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

    if last_layer_report is not None:
        f1_csv = csv_dir / "per_class_f1_layer47.csv"
        write_per_class_report_csv(f1_csv, class_names, last_layer_report)
        print(f"Saved per-class F1 CSV -> {f1_csv}")

    if last_layer_cm is not None:
        cm_csv = csv_dir / "confusion_matrix_layer47.csv"
        write_confusion_matrix_csv(cm_csv, last_layer_cm, class_names)
        print(f"Saved confusion matrix CSV -> {cm_csv}")

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

    bal_accs = [r["balanced_accuracy"] for r in results]
    accs = [r["accuracy"] for r in results]

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_layers), bal_accs, marker="o", markersize=3, linewidth=1.5, label="Balanced Accuracy")
    ax.plot(range(n_layers), accs, marker="s", markersize=3, linewidth=1.5, alpha=0.7, label="Raw Accuracy")
    ax.axhline(y=1.0 / n_classes, color="gray", linestyle="--", linewidth=1, label=f"Random baseline ({1.0/n_classes:.4f})")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Accuracy")
    ax.set_title(f"Layer-wise Logistic Regression w/ Diagonal ({n_proteins} proteins, {n_total_pairs:,} pairs, {n_classes} classes)")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "layer_balanced_accuracy.png", dpi=150)
    plt.close(fig)
    print(f"Saved accuracy plot -> {vis_dir / 'layer_balanced_accuracy.png'}")

    if last_layer_cm is not None:
        row_sums = last_layer_cm.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
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
        ax.set_title(f"Confusion Matrix — Layer 47 w/ Diagonal ({n_classes} classes, row-normalized)")
        fig.tight_layout()
        fig.savefig(vis_dir / "confusion_matrix_layer47.png", dpi=150)
        plt.close(fig)
        print(f"Saved confusion matrix plot -> {vis_dir / 'confusion_matrix_layer47.png'}")

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
        ax.set_title(f"Top {len(top_confused)} Most Confused Pair Types — Layer 47 w/ Diagonal")

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
    print(f"Feature vector: 512-dim (cross-pair 256 + diagonal 256)")
    print(f"Regularization: C={REG_C}")
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

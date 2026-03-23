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

Outputs (inside repo proteins/multi_protein_results/logistic_regression/):
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
import os
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

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from lib.data_loading import discover_proteins, find_pdb_file, get_pair_block_path
from lib.structure import extract_ca_coordinates


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48          # AlphaFold outputs 48 pair representation layers (0-47)
N_CHANNELS = 128       # Each pair block entry has 128 channels per direction
FEATURE_DIM = 2 * N_CHANNELS  # 256: we concatenate pb[i,j,:] (128) and pb[j,i,:] (128)
TEST_SIZE = 0.2        # 20% of all residue pairs held out for test set
RANDOM_STATE = 42      # Fixed seed for reproducible train/test splits
MAX_ITER = 2000        # Max optimizer iterations for LogisticRegression (default 100 is
                       # too few for ~210 classes; lbfgs may not converge otherwise)
TOP_N_CONFUSED = 20    # How many top misclassified pair types to report in output

# ===========================================================================
# Hardcoded directories
# ===========================================================================

# PROJECT_ROOT: Path — resolves to the 'protein/' project root by going up 4 levels
#   from scripts/multi_protein/regression/logistic_regression_pair_type.py
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# DATA_ROOT: Path — contains all local protein subdirectories under ../protein_data/
DATA_ROOT = PROJECT_ROOT.parent / "protein_data"

# OUTPUT_ROOT: Path — aggregated multi-protein outputs stay in the repo workspace.
OUTPUT_ROOT = PROJECT_ROOT / "proteins" / "multi_protein_results"

# OUTPUT_DIR: Path — where all outputs (CSVs, plots) are saved for this script
OUTPUT_DIR = OUTPUT_ROOT / "logistic_regression"


# ===========================================================================
# Helper functions — pair type labels (script-specific, not shared)
# ===========================================================================

def build_pair_labels(residue_data):
    """Build sorted amino acid pair-type labels for every unique residue pair (i, j).

    For a protein with N residues, generates N*(N-1)/2 pairs from the upper triangle
    of the NxN pair matrix (excluding the diagonal, since a residue paired with itself
    is not meaningful). Each pair is labeled by alphabetically sorting the two amino
    acid names and joining with a hyphen, so that e.g. (GLY, ALA) and (ALA, GLY) both
    map to the same label "ALA-GLY". This ensures symmetric pairs get the same class.

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
        These are later used to index into pair blocks as pb[struct_i, struct_j, :].
    struct_j : np.ndarray of int, shape (n_pairs,)
        Column indices from np.triu_indices — the second residue index in each pair.
    """
    # Extract just the 3-letter residue names (e.g. "ALA", "GLY") from each tuple
    res_names = [r[1] for r in residue_data["residues"]]  # list of str, length N
    n = len(res_names)  # int: number of residues in this protein

    # Generate all (i, j) index pairs where i < j (upper triangle, no diagonal).
    # struct_i and struct_j are 1D int arrays of length N*(N-1)/2.
    # Example for n=4: struct_i=[0,0,0,1,1,2], struct_j=[1,2,3,2,3,3]
    struct_i, struct_j = np.triu_indices(n, k=1)

    # For each pair, alphabetically sort the two amino acid names and join with "-".
    # This ensures the label is order-independent: (GLY, ALA) -> "ALA-GLY".
    # Result is a 1D numpy array of strings, same length as struct_i/struct_j.
    labels = np.array([
        "-".join(sorted([res_names[i], res_names[j]]))
        for i, j in zip(struct_i, struct_j)
    ])

    return labels, struct_i, struct_j


def extract_top_confused_pairs(cm, class_names, top_n=20):
    """Find the most common misclassifications from a confusion matrix.

    Looks at all off-diagonal entries in the confusion matrix (which represent
    incorrect predictions) and returns the top_n largest ones. This reveals
    which pair types the model most frequently confuses with each other.

    Parameters
    ----------
    cm : np.ndarray of int, shape (n_classes, n_classes)
        Raw (unnormalized) confusion matrix. cm[i, j] = number of samples
        whose true class is i but were predicted as class j.
    class_names : list of str, length n_classes
        Human-readable class name for each index (e.g. "ALA-GLY").
    top_n : int
        How many top misclassifications to return. Default 20.

    Returns
    -------
    list of dict
        Each dict has keys:
        - 'true_class' (str): the class that was misclassified
        - 'predicted_class' (str): what it was misclassified as
        - 'count' (int): how many times this specific confusion occurred
        - 'true_total' (int): total test samples of the true class
        Sorted by count descending. May be shorter than top_n if fewer
        off-diagonal entries are nonzero.
    """
    # Copy the confusion matrix and zero out the diagonal (correct predictions),
    # leaving only the misclassification counts
    cm_off = cm.copy()  # np.ndarray shape (n_classes, n_classes)
    np.fill_diagonal(cm_off, 0)

    # Flatten to 1D, sort descending, take the top_n largest values.
    # flat_indices: 1D int array of indices into the flattened matrix.
    flat_indices = np.argsort(cm_off.ravel())[::-1][:top_n]

    # Convert flat indices back to 2D (row, col) indices.
    # rows[k], cols[k] = (true_class_index, predicted_class_index) for the k-th
    # largest misclassification.
    rows, cols = np.unravel_index(flat_indices, cm.shape)

    results = []
    for r, c in zip(rows, cols):
        if cm_off[r, c] == 0:
            break  # All remaining entries are zero; stop early
        results.append({
            "true_class": class_names[r],
            "predicted_class": class_names[c],
            "count": int(cm_off[r, c]),
            "true_total": int(cm[r].sum()),  # Total samples with true label r
        })
    return results


# ===========================================================================
# Main
# ===========================================================================

def main():
    # Resolve to absolute paths for clearer logging
    proteins_dir = DATA_ROOT.resolve()     # Path: absolute path to ../protein_data/
    output_dir = OUTPUT_DIR.resolve()      # Path: absolute path to output directory
    n_layers = N_LAYERS                    # int: 48

    # Create output subdirectories for CSVs and visualizations
    csv_dir = output_dir / "csv_files"         # Path: where CSV results are saved
    vis_dir = output_dir / "visualizations"    # Path: where plot PNGs are saved
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Protein data root:  {proteins_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Layers:             {n_layers}")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------
    # For each protein, we need:
    #   1. The PDB file — to extract residue names (amino acid identities)
    #   2. The pair blocks — to verify dimensions match and to later extract features
    # We build pair-type labels here but defer loading the actual pair block data
    # until Phase 3 (one layer at a time to manage memory).

    # all_proteins: list of (str, Path) — e.g. [("6tf4_A", Path("../protein_data/6tf4_A")), ...]
    all_proteins = discover_proteins(proteins_dir)
    print(f"\nDiscovered {len(all_proteins)} protein directories.")

    # protein_data: list of dict — one entry per successfully loaded protein.
    #   Each dict contains metadata and index arrays needed for feature extraction.
    protein_data = []

    # skipped: list of (str, str) — (protein_id, reason) for proteins that couldn't be loaded
    skipped = []

    for protein_id, protein_dir in all_proteins:
        try:
            # --- Check that required files exist ---

            # pdb_path: Path or None — location of the AlphaFold predicted PDB file
            pdb_path = find_pdb_file(protein_dir, protein_id)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                print(f"[SKIP] {protein_id}: no predicted PDB file found")
                continue

            # pb0_path: Path — path to layer-0 pair block (used to check dimensions)
            pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
            if not pb0_path.exists():
                skipped.append((protein_id, "pair block 0 not found"))
                print(f"[SKIP] {protein_id}: pair block 0 not found at {pb0_path}")
                continue

            # --- Extract residue information from PDB ---

            # ca_data: dict with 'residues' (list of tuples) and 'coordinates' (ndarray).
            # We only use 'residues' to get amino acid names for labeling.
            ca_data = extract_ca_coordinates(pdb_path)
            n_ca = len(ca_data["residues"])  # int: number of CA atoms (= number of residues)
            if n_ca == 0:
                skipped.append((protein_id, "no CA atoms found in PDB"))
                print(f"[SKIP] {protein_id}: no CA atoms found in PDB")
                continue

            # --- Verify PDB residue count matches pair block dimensions ---
            # The pair block has shape (R, R, 128) where R must equal n_ca.
            # If they don't match, the pair block indices would be wrong.

            # pb0: np.ndarray shape (R, R, 128) — loaded read-only (mmap) just to check shape
            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = pb0.shape[0]  # int: first dimension R of the pair block
            del pb0  # Release the memory-mapped file immediately

            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                print(f"[SKIP] {protein_id}: CA count {n_ca} != pair block dim {pb_dim}")
                continue

            # --- Build pair-type labels and index arrays ---

            # labels: np.ndarray of str, shape (n_pairs,) — e.g. ["ALA-GLY", "ALA-VAL", ...]
            # struct_i: np.ndarray of int, shape (n_pairs,) — row indices for upper-triangle pairs
            # struct_j: np.ndarray of int, shape (n_pairs,) — col indices for upper-triangle pairs
            # n_pairs = n_ca * (n_ca - 1) / 2
            labels, struct_i, struct_j = build_pair_labels(ca_data)
            n_pair_types = len(set(labels))  # int: number of unique pair types for this protein

            # Store everything needed for this protein in a dict.
            # pb_i and pb_j will be used in Phase 3 to index into pair blocks:
            #   pb[pb_i, pb_j, :] extracts the (i->j) direction features
            #   pb[pb_j, pb_i, :] extracts the (j->i) direction features
            protein_data.append({
                "protein_id": protein_id,    # str: e.g. "7b3a_A"
                "protein_dir": protein_dir,  # Path: e.g. ../protein_data/7b3a_A/
                "n_residues": n_ca,          # int: number of residues in this protein
                "pb_i": struct_i,            # np.ndarray of int: row indices for all pairs
                "pb_j": struct_j,            # np.ndarray of int: col indices for all pairs
                "labels": labels,            # np.ndarray of str: pair-type label per pair
            })
            print(f"[OK]   {protein_id}: {n_ca} residues, {len(labels):,} pairs, {n_pair_types} pair types")

        except Exception as e:
            skipped.append((protein_id, f"unexpected error -- {e}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {e}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    n_proteins = len(protein_data)  # int: how many proteins were successfully loaded

    # ------------------------------------------------------------------
    # Phase 2: Pool labels, encode, and split
    # ------------------------------------------------------------------
    # Concatenate pair-type labels from ALL proteins into one flat array.
    # This is NOT unique labels — it contains every single pair from every protein,
    # with many duplicates (e.g. "ALA-GLY" appears thousands of times).
    # If protein A has 37,950 pairs and protein B has 20,000 pairs, labels_all has 57,950 entries.
    labels_all = np.concatenate([d["labels"] for d in protein_data])  # np.ndarray of str, shape (n_total_pairs,)
    n_total_pairs = len(labels_all)  # int: total number of residue pairs across all proteins

    # protein_offsets: list of (str, int, int) — tracks where each protein's pairs
    # sit within the concatenated labels_all array.
    # Example: [("6tf4_A", 0, 37950), ("7b3a_A", 37950, 75900)]
    # This is used in Phase 6 for the console summary to report per-protein pair counts.
    protein_offsets = []
    offset = 0
    for d in protein_data:
        n = len(d["labels"])
        protein_offsets.append((d["protein_id"], offset, offset + n))
        offset += n

    # LabelEncoder converts string labels to integer class IDs.
    # This is NOT one-hot encoding — it maps each unique string to an integer (0 to n_classes-1).
    # Example: "ALA-ALA" -> 0, "ALA-ARG" -> 1, ..., "VAL-VAL" -> 209
    # Sklearn's LogisticRegression internally handles these as categorical classes
    # (using softmax + cross-entropy), NOT as ordinal values with magnitude.
    label_encoder = LabelEncoder()
    y_all = label_encoder.fit_transform(labels_all)  # np.ndarray of int, shape (n_total_pairs,)
    n_classes = len(label_encoder.classes_)  # int: number of unique pair types (e.g. ~210)
    class_names = list(label_encoder.classes_)  # list of str: sorted unique class names

    print(f"\nTotal pairs across {n_proteins} protein(s): {n_total_pairs:,}")
    print(f"Unique pair types: {n_classes}")

    # Create a train/test split of INDICES (not the data itself, since features
    # are loaded one layer at a time in Phase 3 to save memory).
    # all_idx: np.ndarray of int, shape (n_total_pairs,) — [0, 1, 2, ..., n_total_pairs-1]
    all_idx = np.arange(n_total_pairs)

    # idx_train, idx_test: np.ndarray of int — indices into labels_all / y_all / X_all.
    # stratify=y_all ensures each class has proportional representation in both splits,
    # which is important because some pair types are much rarer than others.
    idx_train, idx_test = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE,
        stratify=y_all,
    )

    # y_train, y_test: np.ndarray of int — integer-encoded class labels for each split
    y_train = y_all[idx_train]  # shape (n_train,)
    y_test = y_all[idx_test]    # shape (n_test,)

    print(f"Train: {len(idx_train):,}  Test: {len(idx_test):,}")

    # ------------------------------------------------------------------
    # Phase 3: Layer-wise classification
    # ------------------------------------------------------------------
    # Train one independent logistic regression model per layer (48 total).
    # Each layer's model gets a fresh set of 256-dim features extracted from
    # that layer's pair block, but the train/test split and labels stay the same.
    # This lets us see how classification accuracy evolves through the network.

    # results: list of dict — one entry per layer with accuracy metrics.
    # Used in Phase 4 to write the CSV and Phase 5 to plot accuracy vs layer.
    results = []

    # last_layer_cm: np.ndarray or None — confusion matrix for the final layer (47).
    # Saved separately because we generate extra diagnostics (F1, top confused pairs)
    # only for the last layer.
    last_layer_cm = None

    # last_layer_report: dict or None — per-class precision/recall/f1/support for layer 47.
    last_layer_report = None

    print()
    for layer in range(n_layers):
        try:
            # --- Build the feature matrix X_all for this layer ---
            # We load each protein's pair block for this layer, extract the 256-dim
            # feature vector for each pair, and stack them into one big matrix.

            X_chunks = []  # list of np.ndarray — one chunk per protein, each shape (n_pairs_i, 256)
            for d in protein_data:
                # pb_path: Path — path to this protein's pair block for the current layer
                pb_path = get_pair_block_path(d["protein_dir"], d["protein_id"], layer)

                # pb: np.ndarray shape (R, R, 128) — the full pair representation matrix.
                # R = number of residues, 128 = channels per direction.
                pb = np.load(str(pb_path))

                # For each pair (i, j), concatenate the two directions:
                #   pb[i, j, :] = 128-dim vector for the i->j direction
                #   pb[j, i, :] = 128-dim vector for the j->i direction
                # Result: X_chunk has shape (n_pairs, 256) where n_pairs = len(pb_i).
                # The pair block is asymmetric (pb[i,j] != pb[j,i]), so both directions
                # carry distinct information and are both included as features.
                X_chunk = np.concatenate([
                    pb[d["pb_i"], d["pb_j"], :],  # shape (n_pairs, 128): i->j features
                    pb[d["pb_j"], d["pb_i"], :],  # shape (n_pairs, 128): j->i features
                ], axis=1)  # shape (n_pairs, 256)
                X_chunks.append(X_chunk)
                del pb  # Free the full (R, R, 128) array to save memory

            # Stack all proteins' feature chunks into one matrix.
            # X_all: np.ndarray shape (n_total_pairs, 256) — all pairs from all proteins.
            # The row ordering matches labels_all: first all pairs from protein 0,
            # then protein 1, etc.
            X_all = np.vstack(X_chunks)
            del X_chunks  # Free the list of chunks

            # --- Standardize features ---
            # StandardScaler: zero-mean, unit-variance per feature (column).
            # fit_transform on training data learns mean/std, then transform on test data
            # uses the SAME mean/std (no data leakage from test set).
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_all[idx_train])  # np.ndarray shape (n_train, 256)
            X_test = scaler.transform(X_all[idx_test])         # np.ndarray shape (n_test, 256)
            del X_all  # Free the full feature matrix

            # --- Train logistic regression ---
            # LogisticRegression with solver="lbfgs" defaults to multi_class="multinomial",
            # which means it uses SOFTMAX over all ~210 classes and minimizes
            # CROSS-ENTROPY loss. This is equivalent to multinomial logistic regression.
            #
            # C=1e12: inverse regularization strength. A huge C means effectively NO
            # regularization — the model fits the training data without penalty on weights.
            #
            # max_iter=2000: L-BFGS optimizer iterations. With ~210 classes and 256
            # features, the default 100 iterations is often insufficient for convergence.
            model = LogisticRegression(
                C=1e12,
                solver="lbfgs",
                max_iter=MAX_ITER,
            )
            model.fit(X_train, y_train)  # Train on (n_train, 256) features and int labels

            # y_pred_test: np.ndarray of int, shape (n_test,) — predicted class IDs for test set
            y_pred_test = model.predict(X_test)

            # --- Compute accuracy metrics ---

            # bal_acc (float): balanced accuracy = mean of per-class recall values.
            # Each class contributes equally regardless of how many samples it has.
            # This is the fairer metric since pair types have very unequal frequencies.
            bal_acc = balanced_accuracy_score(y_test, y_pred_test)

            # acc (float): raw accuracy = fraction of correct predictions.
            # Can be misleading if common pair types dominate the test set.
            acc = accuracy_score(y_test, y_pred_test)

            # Store results for this layer (used in Phase 4 CSV and Phase 5 plots)
            results.append({
                "layer": layer,                # int: layer index (0-47)
                "balanced_accuracy": bal_acc,   # float: balanced accuracy (0 to 1)
                "accuracy": acc,                # float: raw accuracy (0 to 1)
                "n_classes": n_classes,          # int: number of unique pair types
                "n_proteins": n_proteins,        # int: number of proteins used
                "n_total_pairs": n_total_pairs,  # int: total pairs across all proteins
            })
            print(f"Layer {layer:2d}  balanced_acc = {bal_acc:.4f}  acc = {acc:.4f}")

            # --- Extra diagnostics for the final layer (47) ---
            # Save confusion matrix and per-class F1 only for the last layer,
            # since these are expensive to compute and most interesting at the final layer.
            if layer == n_layers - 1:
                # last_layer_cm: np.ndarray of int, shape (n_classes, n_classes).
                # cm[i, j] = number of test samples with true class i predicted as class j.
                # Diagonal = correct predictions, off-diagonal = misclassifications.
                last_layer_cm = confusion_matrix(
                    y_test, y_pred_test,
                    labels=np.arange(n_classes),  # Ensure all classes appear even if absent from test
                )

                # Per-class metrics:
                # precision: np.ndarray of float, shape (n_classes,) — of all samples predicted as class k,
                #   what fraction actually are class k? (measures false positive rate)
                # recall: np.ndarray of float, shape (n_classes,) — of all true class k samples,
                #   what fraction were correctly predicted? (measures false negative rate)
                # f1: np.ndarray of float, shape (n_classes,) — harmonic mean of precision and recall.
                # support: np.ndarray of int, shape (n_classes,) — number of true test samples per class.
                precision, recall, f1, support = precision_recall_fscore_support(
                    y_test, y_pred_test,
                    labels=np.arange(n_classes),
                    zero_division=0,  # Return 0 for classes with no predictions/samples
                )
                last_layer_report = {
                    "precision": precision,
                    "recall": recall,
                    "f1": f1,
                    "support": support,
                }

            del X_train, X_test  # Free train/test feature matrices before next layer

        except Exception as e:
            print(f"Layer {layer:2d}  ERROR: {e}")
            traceback.print_exc()
            # Record NaN for failed layers so the CSV and plots still have 48 rows
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

    # CSV 1: Layer balanced accuracy — one row per layer (48 rows total).
    # Columns: layer, balanced_accuracy, accuracy, n_classes, n_proteins, n_total_pairs.
    # This is the primary output: tracks how well each layer's representations
    # can distinguish amino acid pair types.
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

    # CSV 2: Per-class F1 scores for layer 47 only.
    # One row per pair type that has at least one test sample (support > 0).
    # Columns: pair_type (e.g. "ALA-GLY"), f1, precision, recall, support (int).
    # Useful for identifying which pair types are easy/hard to classify.
    if last_layer_report is not None:
        f1_csv = csv_dir / "per_class_f1_layer47.csv"
        with open(f1_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["pair_type", "f1", "precision", "recall", "support"])
            for i in range(n_classes):
                sup = int(last_layer_report["support"][i])  # int: test samples for this class
                if sup == 0:
                    continue  # Skip classes with no test samples
                writer.writerow([
                    class_names[i],
                    f"{last_layer_report['f1'][i]:.4f}",
                    f"{last_layer_report['precision'][i]:.4f}",
                    f"{last_layer_report['recall'][i]:.4f}",
                    sup,
                ])
        print(f"Saved per-class F1 CSV -> {f1_csv}")

    # CSV 3: Full confusion matrix for layer 47.
    # Shape: (n_classes+1) x (n_classes+1) including header row/column.
    # Row = true class, Column = predicted class. Diagonal = correct predictions.
    if last_layer_cm is not None:
        cm_csv = csv_dir / "confusion_matrix_layer47.csv"
        with open(cm_csv, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow([""] + class_names)  # Header: blank corner + class names
            for i, row in enumerate(last_layer_cm):
                # Each row: true class name, then count for each predicted class
                writer.writerow([class_names[i]] + [int(x) for x in row])
        print(f"Saved confusion matrix CSV -> {cm_csv}")

    # CSV 4: Top confused pairs for layer 47.
    # Lists the most frequent misclassifications (off-diagonal confusion matrix entries).
    # Columns: true_class, predicted_class, count, true_total.
    # Example: "ALA-LEU" often misclassified as "ALA-ILE" (similar hydrophobic residues).
    if last_layer_cm is not None:
        # top_confused: list of dict — see extract_top_confused_pairs() docstring
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

    # Plot 1: Accuracy vs layer — the main result plot.
    # Shows how well each layer's pair representations can predict amino acid pair type.
    # Two lines: balanced accuracy (fair metric) and raw accuracy (biased toward common classes).
    # A horizontal dashed line shows the random-guess baseline (1/n_classes).
    bal_accs = [r["balanced_accuracy"] for r in results]  # list of float, length 48
    accs = [r["accuracy"] for r in results]                # list of float, length 48

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

    # Plot 2: Confusion matrix heatmap for layer 47 (row-normalized).
    # Each cell (i, j) shows what fraction of true-class-i samples were predicted as class j.
    # Row normalization means each row sums to 1.0, making it easy to see per-class recall
    # (diagonal) and which classes get confused with which (off-diagonal).
    if last_layer_cm is not None:
        # Row-normalize: divide each row by its sum so values are fractions (0 to 1)
        row_sums = last_layer_cm.sum(axis=1, keepdims=True)  # shape (n_classes, 1)
        row_sums[row_sums == 0] = 1  # Avoid division by zero for classes with no test samples
        cm_norm = last_layer_cm / row_sums  # np.ndarray of float, shape (n_classes, n_classes)

        fig, ax = plt.subplots(figsize=(24, 22))
        im = ax.imshow(cm_norm, cmap="Blues", interpolation="nearest", vmin=0, vmax=1)
        fig.colorbar(im, ax=ax, label="Fraction of true class", shrink=0.8)

        # With ~210 classes, labeling every tick is unreadable — show every 10th class name
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

    # Plot 3: Horizontal bar chart of the top-N most confused pair types (layer 47).
    # Each bar shows how many times a specific (true -> predicted) misclassification occurred.
    # Bars are sorted descending (most confused at top).
    if last_layer_cm is not None and top_confused:
        fig, ax = plt.subplots(figsize=(10, 8))
        # bar_labels: list of str — e.g. ["ALA-LEU -> ALA-ILE", ...]
        bar_labels = [f"{r['true_class']} -> {r['predicted_class']}" for r in top_confused]
        # bar_values: list of int — misclassification counts
        bar_values = [r["count"] for r in top_confused]

        y_pos = range(len(bar_labels))
        ax.barh(y_pos, bar_values, color="steelblue")
        ax.set_yticks(y_pos)
        ax.set_yticklabels(bar_labels, fontsize=9)
        ax.invert_yaxis()  # Most confused pair at the top
        ax.set_xlabel("Misclassification count")
        ax.set_title(f"Top {len(top_confused)} Most Confused Pair Types — Layer 47")

        # Annotate each bar with its count value
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
    # Print a human-readable summary of the run: which proteins were included,
    # which were skipped, overall statistics, and the best-performing layer.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Included proteins ({n_proteins}):")
    for pid, start, end in protein_offsets:
        # Look up the residue count for this protein from protein_data
        n_res = next(d["n_residues"] for d in protein_data if d["protein_id"] == pid)
        print(f"  {pid}: {n_res} residues, {end - start:,} pairs")
    if skipped:
        print(f"\nSkipped proteins ({len(skipped)}):")
        for pid, reason in skipped:
            print(f"  {pid}: {reason}")

    print(f"\nPair types: {n_classes}")
    print(f"Total pairs: {n_total_pairs:,} (train: {len(idx_train):,}, test: {len(idx_test):,})")

    # Find the layer with the highest balanced accuracy across all 48 layers
    valid_bal_accs = [r["balanced_accuracy"] for r in results if not math.isnan(r["balanced_accuracy"])]
    if valid_bal_accs:
        # best_layer: int — the layer index (0-47) with the highest balanced accuracy
        best_layer = int(np.nanargmax([r["balanced_accuracy"] for r in results]))
        print(f"\nBest layer: {best_layer}  balanced_acc = {results[best_layer]['balanced_accuracy']:.4f}")
        print(f"Layer {n_layers - 1}:   balanced_acc = {results[-1]['balanced_accuracy']:.4f}  acc = {results[-1]['accuracy']:.4f}")
        # Random baseline: if predicting uniformly at random, expected accuracy = 1/n_classes
        print(f"Random baseline: {1.0 / n_classes:.4f}")


if __name__ == "__main__":
    main()

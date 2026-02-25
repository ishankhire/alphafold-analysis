#!/usr/bin/env python3
"""
aa_augmented_regression.py
==========================
Local regression script that compares two models per AlphaFold layer:

  1. Baseline  (256-dim): pair-block features only [pb[i,j,:], pb[j,i,:]]
  2. Augmented (296-dim): pair-block features + 40-dim one-hot amino acid encoding

For each residue pair (i, j), the amino acid names are sorted alphabetically
(so ALA-GLY == GLY-ALA), then each is one-hot encoded into 20 dims and
concatenated to form a 40-dim vector appended to the pair-block features.

This tests whether amino acid identity provides information about pairwise
distances beyond what the pair representation already encodes.

Dependencies (all pip-installable):
    pip install numpy scipy matplotlib scikit-learn biopython

Outputs (inside proteins/multi_protein_results/aa_augmented_regression/):
    csv_files/layer_regression_r2_comparison.csv
    csv_files/layer_regression_coefficients_augmented.csv
    visualizations/layer_r2_comparison.png
    visualizations/pred_vs_actual_augmented.png
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
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))))
from lib.data_loading import discover_proteins, find_pdb_file, get_pair_block_path
from lib.structure import extract_ca_coordinates, compute_distance_matrix


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48          # AlphaFold outputs 48 pair representation layers (0-47)
N_CHANNELS = 128       # Each pair block entry has 128 channels per direction
FEATURE_DIM = 2 * N_CHANNELS  # 256: we concatenate pb[i,j,:] (128) and pb[j,i,:] (128)
TEST_SIZE = 0.2        # 20% of all residue pairs held out for test set
RANDOM_STATE = 42      # Fixed seed for reproducible train/test splits

# STANDARD_AAS: list of str — the 20 standard amino acids in alphabetical order.
# This canonical ordering is used for one-hot encoding: each amino acid maps to
# a unique position (0-19). The alphabetical sort ensures deterministic encoding.
STANDARD_AAS = [
    "ALA", "ARG", "ASN", "ASP", "CYS",
    "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO",
    "SER", "THR", "TRP", "TYR", "VAL",
]

# AA_TO_INDEX: dict mapping str -> int — maps each 3-letter amino acid code
# to its one-hot index (0-19). E.g. {"ALA": 0, "ARG": 1, ..., "VAL": 19}.
# Non-standard amino acids (e.g. selenomethionine "MSE") are NOT in this dict
# and will get all-zero encodings (handled gracefully in build_aa_onehot_features).
AA_TO_INDEX = {aa: i for i, aa in enumerate(STANDARD_AAS)}

N_AA = 20              # int: number of standard amino acids (always 20)

# AA_FEATURE_DIM: int — 40. For each pair (i, j), we create two one-hot vectors
# of length 20 (one per amino acid in the pair, sorted alphabetically), then
# concatenate them: 20 + 20 = 40. Positions 0-19 encode the alphabetically-first
# amino acid; positions 20-39 encode the alphabetically-second amino acid.
AA_FEATURE_DIM = 2 * N_AA  # 40

# AUGMENTED_FEATURE_DIM: int — 296. The augmented model uses the 256-dim pair-block
# features PLUS the 40-dim amino acid one-hot encoding: 256 + 40 = 296.
# This lets us measure how much additional predictive power amino acid identity
# adds on top of the learned pair representations.
AUGMENTED_FEATURE_DIM = FEATURE_DIM + AA_FEATURE_DIM  # 296

# ===========================================================================
# Hardcoded directories
# ===========================================================================

# PROJECT_ROOT: Path — resolves to the 'protein/' project root by going up 4 levels
#   from scripts/multi_protein/regression/aa_augmented_regression.py
#   Level 1: regression/ -> multi_protein/
#   Level 2: multi_protein/ -> scripts/
#   Level 3: scripts/ -> protein/
#   Level 4: (parent of __file__ itself, since __file__ is a file not a dir)
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent

# PROTEINS_DIR: Path — contains all protein subdirectories (e.g. 7b3a_A/, 6tf4_A/)
# and the multi_protein_results/ folder for cross-protein outputs.
PROTEINS_DIR = PROJECT_ROOT / "proteins"

# OUTPUT_DIR: Path — where all outputs (CSVs, plots) are saved for this script.
# Stored under multi_protein_results/ because this analysis pools data across proteins.
# Uses a dedicated "aa_augmented_regression" subfolder to avoid conflicting with
# other multi-protein regression outputs.
OUTPUT_DIR   = PROTEINS_DIR / "multi_protein_results" / "aa_augmented_regression"


# ===========================================================================
# Helper functions — amino acid one-hot encoding (script-specific, not shared)
# ===========================================================================

def build_aa_onehot_features(residue_data, struct_i, struct_j):
    """Build a 40-dim one-hot amino acid encoding for each residue pair.

    For each pair (i, j), the two residue names are sorted alphabetically.
    The first (alphabetically earlier) AA is one-hot encoded into positions 0-19,
    the second into positions 20-39. This symmetrization ensures that a pair
    (GLY, ALA) gets the exact same encoding as (ALA, GLY), removing directional
    ambiguity in the amino acid identity features.

    Non-standard amino acids (e.g. "MSE", "SEP") get all-zero entries because
    they are not in AA_TO_INDEX. This gracefully handles modified residues
    without crashing — they simply contribute no amino acid identity signal.

    This 40-dim vector is later concatenated with the 256-dim pair-block features
    to form the 296-dim augmented feature vector for each pair.

    Parameters
    ----------
    residue_data : dict
        Output of extract_ca_coordinates(). Must have a 'residues' key containing
        a list of (chain_id, residue_name, residue_number) tuples. The residue_name
        field (index 1) is used to determine the amino acid type.
    struct_i : np.ndarray of int, shape (n_pairs,)
        Row indices from np.triu_indices — the first residue index in each pair.
        These index into residue_data["residues"] to get the amino acid name.
    struct_j : np.ndarray of int, shape (n_pairs,)
        Column indices from np.triu_indices — the second residue index in each pair.

    Returns
    -------
    aa_features : np.ndarray of float32, shape (n_pairs, 40)
        One-hot encoded amino acid pair features. Each row has exactly 0 or 2
        nonzero entries (0 if both AAs are non-standard, 1 if one is non-standard,
        2 if both are standard — the typical case).
        Columns 0-19: one-hot for the alphabetically-first amino acid.
        Columns 20-39: one-hot for the alphabetically-second amino acid.
    """
    # Extract just the 3-letter residue names (e.g. "ALA", "GLY") from each tuple.
    # res_names: list of str, length = number of residues in this protein.
    res_names = [r[1] for r in residue_data["residues"]]

    n_pairs = len(struct_i)  # int: number of unique residue pairs (upper triangle)

    # Initialize the output matrix as all zeros. For each pair, we will set exactly
    # two entries to 1.0 (one in columns 0-19, one in columns 20-39) if both
    # amino acids are standard. Using float32 to save memory (matches pair block dtype).
    aa_features = np.zeros((n_pairs, AA_FEATURE_DIM), dtype=np.float32)  # shape (n_pairs, 40)

    for k in range(n_pairs):
        # aa1: str — 3-letter code of the first residue in this pair (e.g. "GLY")
        aa1 = res_names[struct_i[k]]
        # aa2: str — 3-letter code of the second residue in this pair (e.g. "ALA")
        aa2 = res_names[struct_j[k]]

        # Sort alphabetically so that (GLY, ALA) -> ["ALA", "GLY"].
        # This ensures the encoding is symmetric: pair (i,j) == pair (j,i).
        sorted_pair = sorted([aa1, aa2])  # list of 2 str, alphabetically ordered

        # Look up the one-hot index for each amino acid.
        # idx1: int or None — index (0-19) for the alphabetically-first AA, or None if non-standard.
        idx1 = AA_TO_INDEX.get(sorted_pair[0])
        # idx2: int or None — index (0-19) for the alphabetically-second AA, or None if non-standard.
        idx2 = AA_TO_INDEX.get(sorted_pair[1])

        # Set the one-hot entries. Positions 0-19 are for the first (sorted) AA,
        # positions 20-39 are for the second (sorted) AA.
        if idx1 is not None:
            aa_features[k, idx1] = 1.0        # Set bit in columns 0-19
        if idx2 is not None:
            aa_features[k, N_AA + idx2] = 1.0  # Set bit in columns 20-39 (offset by 20)

    return aa_features


# ===========================================================================
# Main
# ===========================================================================

def main():
    """Main entry point: runs the full baseline vs. augmented regression pipeline.

    The pipeline has 6 phases:
      Phase 1: Load PDB structures and pair block metadata for each protein.
               Extract CA coordinates, compute distance matrices, build AA features.
      Phase 2: Pool all residue pairs across proteins, create train/test split.
      Phase 3: For each of 48 layers, train two linear regression models:
               (a) baseline (256-dim pair-block only) and
               (b) augmented (296-dim pair-block + amino acid one-hot).
               Record R-squared for both models.
      Phase 4: Save CSV outputs (R-squared comparison, augmented model coefficients).
      Phase 5: Generate visualizations (R-squared vs layer, predicted vs actual scatter).
      Phase 6: Print a console summary of the run.
    """
    # Resolve to absolute paths for clearer logging and deterministic path handling
    proteins_dir = PROTEINS_DIR.resolve()  # Path: absolute path to proteins/
    output_dir = OUTPUT_DIR.resolve()      # Path: absolute path to output directory
    n_layers = N_LAYERS                    # int: 48 — local alias for the layer count constant

    # Create output subdirectories for CSVs and visualizations.
    # parents=True creates intermediate directories if needed.
    # exist_ok=True prevents errors if directories already exist from a previous run.
    csv_dir = output_dir / "csv_files"         # Path: where CSV results are saved
    vis_dir = output_dir / "visualizations"    # Path: where plot PNGs are saved
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Proteins directory: {proteins_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Layers:             {n_layers}")
    print(f"Feature dims:       baseline={FEATURE_DIM}, augmented={AUGMENTED_FEATURE_DIM}")

    # ------------------------------------------------------------------
    # Phase 1: Load data for each protein
    # ------------------------------------------------------------------
    # For each protein, we need:
    #   1. The PDB file — to extract residue names (for AA one-hot encoding) AND
    #      CA coordinates (for computing pairwise distances as regression targets)
    #   2. The pair blocks — to verify dimensions match (actual features loaded in Phase 3)
    #
    # We compute distance matrices and amino acid one-hot features here, but defer
    # loading the actual pair block data until Phase 3 (one layer at a time to manage
    # memory, since each pair block is shape (R, R, 128) and can be large).

    # all_proteins: list of (str, Path) — e.g. [("6tf4_A", Path("proteins/6tf4_A")), ...]
    all_proteins = discover_proteins(proteins_dir)
    print(f"\nDiscovered {len(all_proteins)} protein directories.")

    # protein_data: list of dict — one entry per successfully loaded protein.
    #   Each dict contains metadata, index arrays, distance targets, and AA features
    #   needed for regression in Phase 3.
    protein_data = []

    # skipped: list of (str, str) — (protein_id, reason) for proteins that couldn't be loaded.
    # Reported in Phase 6 summary so the user knows why a protein was excluded.
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

            # pb0_path: Path — path to layer-0 pair block (used only to check dimensions,
            # not to extract features; actual feature extraction happens in Phase 3).
            pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
            if not pb0_path.exists():
                skipped.append((protein_id, "pair block 0 not found"))
                print(f"[SKIP] {protein_id}: pair block 0 not found at {pb0_path}")
                continue

            # --- Extract residue information and coordinates from PDB ---

            # ca_data: dict with 'residues' (list of tuples) and 'coordinates' (ndarray).
            # 'residues' provides amino acid names for one-hot encoding.
            # 'coordinates' provides CA atom positions for distance matrix computation.
            ca_data = extract_ca_coordinates(pdb_path)
            n_ca = len(ca_data["residues"])  # int: number of CA atoms (= number of residues)
            if n_ca == 0:
                skipped.append((protein_id, "no CA atoms found in PDB"))
                print(f"[SKIP] {protein_id}: no CA atoms found in PDB")
                continue

            # --- Verify PDB residue count matches pair block dimensions ---
            # The pair block has shape (R, R, 128) where R must equal n_ca.
            # If they don't match, indexing pb[struct_i, struct_j, :] would be wrong
            # because the pair block and the distance matrix would refer to different residues.

            # pb0: np.ndarray shape (R, R, 128) — loaded read-only (mmap) just to check shape.
            # mmap_mode="r" avoids reading the entire array into RAM.
            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = pb0.shape[0]  # int: first dimension R of the pair block
            del pb0  # Release the memory-mapped file immediately — we only needed the shape

            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                print(f"[SKIP] {protein_id}: CA count {n_ca} != pair block dim {pb_dim}")
                continue

            # --- Compute pairwise CA-CA distance matrix (regression targets) ---

            # dist_matrix: np.ndarray of float, shape (n_ca, n_ca) — symmetric distance
            # matrix in Angstroms. dist_matrix[i, j] = Euclidean distance between the CA
            # atoms of residue i and residue j. Diagonal is 0.
            dist_matrix = compute_distance_matrix(ca_data["coordinates"])

            # --- Extract upper-triangle indices and distance targets ---

            # struct_i, struct_j: np.ndarray of int, shape (n_pairs,) — row and column
            # indices for all unique pairs from the upper triangle (excluding diagonal).
            # k=1 means offset 1 above the diagonal, so we get i < j for all pairs.
            # n_pairs = n_ca * (n_ca - 1) / 2.
            # Example for n_ca=4: struct_i=[0,0,0,1,1,2], struct_j=[1,2,3,2,3,3].
            struct_i, struct_j = np.triu_indices(n_ca, k=1)

            # y_protein: np.ndarray of float, shape (n_pairs,) — the CA-CA distance
            # for each unique pair (i, j). This is the regression target.
            # Indexed from the distance matrix using the upper-triangle indices.
            y_protein = dist_matrix[struct_i, struct_j]

            # --- Build amino acid one-hot features for this protein ---

            # aa_features: np.ndarray of float32, shape (n_pairs, 40) — the 40-dim
            # one-hot amino acid encoding for each pair. See build_aa_onehot_features()
            # for details on the encoding scheme (alphabetically sorted, two 20-dim vectors).
            aa_features = build_aa_onehot_features(ca_data, struct_i, struct_j)

            # Store everything needed for this protein in a dict.
            # pb_i and pb_j will be used in Phase 3 to index into pair blocks:
            #   pb[pb_i, pb_j, :] extracts the (i->j) direction features (128-dim)
            #   pb[pb_j, pb_i, :] extracts the (j->i) direction features (128-dim)
            # y is the regression target (CA-CA distances).
            # aa_features is the 40-dim one-hot encoding appended in the augmented model.
            protein_data.append({
                "protein_id": protein_id,      # str: e.g. "7b3a_A"
                "protein_dir": protein_dir,    # Path: e.g. proteins/7b3a_A/
                "n_residues": n_ca,            # int: number of residues in this protein
                "pb_i": struct_i,              # np.ndarray of int, shape (n_pairs,): row indices for all pairs
                "pb_j": struct_j,              # np.ndarray of int, shape (n_pairs,): col indices for all pairs
                "y": y_protein,                # np.ndarray of float, shape (n_pairs,): CA-CA distances in Angstroms
                "aa_features": aa_features,    # np.ndarray of float32, shape (n_pairs, 40): one-hot AA encoding
            })
            print(f"[OK]   {protein_id}: {n_ca} residues, {len(y_protein):,} pairs")

        except Exception as e:
            skipped.append((protein_id, f"unexpected error -- {e}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {e}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    n_proteins = len(protein_data)  # int: how many proteins were successfully loaded

    # ------------------------------------------------------------------
    # Phase 2: Pool targets and split
    # ------------------------------------------------------------------
    # Concatenate regression targets and amino acid features from ALL proteins into
    # single flat arrays. This pools data across proteins so that a single model
    # is trained on pairs from all proteins combined. The train/test split is done
    # on indices (not data) because the pair-block features are loaded one layer
    # at a time in Phase 3 to manage memory.

    # y_all: np.ndarray of float, shape (n_total_pairs,) — concatenated CA-CA distances
    # from all proteins. This is the regression target for the pooled model.
    # If protein A has 37,950 pairs and protein B has 20,000 pairs, y_all has 57,950 entries.
    y_all = np.concatenate([d["y"] for d in protein_data])

    # aa_all: np.ndarray of float32, shape (n_total_pairs, 40) — concatenated one-hot
    # amino acid features from all proteins, in the same order as y_all.
    # Stacked vertically since each protein contributes a matrix of shape (n_pairs_i, 40).
    aa_all = np.vstack([d["aa_features"] for d in protein_data])

    n_total_pairs = len(y_all)  # int: total number of residue pairs across all proteins

    # protein_offsets: list of (str, int, int) — tracks where each protein's pairs
    # sit within the concatenated y_all / aa_all arrays.
    # Example: [("6tf4_A", 0, 37950), ("7b3a_A", 37950, 75900)]
    # Used in Phase 5 for the predicted-vs-actual scatter plot (to color points by protein)
    # and in Phase 6 for the console summary.
    protein_offsets = []
    offset = 0  # int: running offset into the concatenated arrays
    for d in protein_data:
        n = len(d["y"])  # int: number of pairs for this protein
        protein_offsets.append((d["protein_id"], offset, offset + n))
        offset += n

    print(f"\nTotal pairs across {n_proteins} protein(s): {n_total_pairs:,}")

    # Create a train/test split of INDICES (not the data itself, since pair-block
    # features are loaded one layer at a time in Phase 3 to save memory).
    # all_idx: np.ndarray of int, shape (n_total_pairs,) — [0, 1, 2, ..., n_total_pairs-1]
    all_idx = np.arange(n_total_pairs)

    # idx_train, idx_test: np.ndarray of int — indices into y_all / aa_all / X_all.
    # These define which pairs are used for training and which for evaluation.
    # The same split is reused for every layer so results are comparable across layers.
    # Note: unlike the logistic regression script, we do NOT stratify here because
    # this is a regression task (continuous y), not classification.
    idx_train, idx_test = train_test_split(all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE)

    # y_train: np.ndarray of float, shape (n_train,) — CA-CA distances for training pairs
    y_train = y_all[idx_train]
    # y_test: np.ndarray of float, shape (n_test,) — CA-CA distances for test pairs
    y_test = y_all[idx_test]

    # ------------------------------------------------------------------
    # Phase 3: Layer-wise regression (baseline + augmented)
    # ------------------------------------------------------------------
    # For each of the 48 AlphaFold layers, we train TWO independent linear regression
    # models on the same train/test split:
    #
    #   1. Baseline model (256-dim features):
    #      Features = [pb[i,j,:], pb[j,i,:]] — pair-block vectors only.
    #      This measures how much pairwise distance information is encoded in the
    #      pair representation at each layer.
    #
    #   2. Augmented model (296-dim features):
    #      Features = [pb[i,j,:], pb[j,i,:], aa_onehot(i,j)] — pair-block vectors
    #      plus the 40-dim amino acid one-hot encoding.
    #      This measures whether knowing the amino acid identities of the two residues
    #      adds predictive power beyond what the pair representation already captures.
    #
    # If the augmented model's R-squared is significantly higher than the baseline,
    # it means the pair representation does NOT fully encode amino acid identity
    # information relevant to distance prediction.

    # results_baseline: list of float — R-squared values for the baseline model, one per layer.
    # Length will be 48 after the loop (or contain NaN for any failed layers).
    results_baseline = []

    # results_augmented: list of float — R-squared values for the augmented model, one per layer.
    results_augmented = []

    # all_coefs_augmented: list of dict — regression coefficients for the augmented model
    # at each layer. Each dict has keys: "layer" (int), "intercept" (float), and
    # "coef_0" through "coef_295" (float). Saved to CSV in Phase 4 for downstream analysis
    # (e.g., to see which features have the largest weights).
    all_coefs_augmented = []

    # last_layer_preds_augmented: np.ndarray of float or None — predicted distances from
    # the augmented model at the final layer (layer 47), for ALL pairs (not just test set).
    # Used in Phase 5 to generate the predicted-vs-actual scatter plot. Set to None
    # initially and populated only when we reach the last layer.
    last_layer_preds_augmented = None

    print()
    for layer in range(n_layers):
        try:
            # --- Build the feature matrix X_pb for this layer ---
            # We load each protein's pair block for this layer, extract the 256-dim
            # feature vector for each pair, and stack them into one big matrix.

            X_chunks = []  # list of np.ndarray — one chunk per protein, each shape (n_pairs_i, 256)
            for d in protein_data:
                # pb_path: Path — path to this protein's pair block for the current layer
                pb_path = get_pair_block_path(d["protein_dir"], d["protein_id"], layer)

                # pb: np.ndarray shape (R, R, 128) — the full pair representation matrix
                # for this protein at this layer. R = number of residues, 128 = channels.
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
            # X_pb: np.ndarray shape (n_total_pairs, 256) — all pairs from all proteins.
            # The row ordering matches y_all and aa_all: first all pairs from protein 0,
            # then protein 1, etc.
            X_pb = np.vstack(X_chunks)
            del X_chunks  # Free the list of chunks

            # ============================================================
            # --- Baseline model (256-dim) ---
            # ============================================================
            # Uses ONLY pair-block features. This is the control: how well can
            # the pair representation alone predict CA-CA distances?

            # scaler_base: StandardScaler — zero-mean, unit-variance normalization.
            # fit_transform on training data learns mean/std per feature (column),
            # then transform on test data uses the SAME mean/std (no data leakage).
            scaler_base = StandardScaler()
            X_train_base = scaler_base.fit_transform(X_pb[idx_train])  # np.ndarray shape (n_train, 256)
            X_test_base = scaler_base.transform(X_pb[idx_test])         # np.ndarray shape (n_test, 256)

            # model_base: LinearRegression — ordinary least squares (OLS) regression.
            # Minimizes sum of squared residuals: ||y - Xw||^2.
            # No regularization (unlike Ridge/Lasso), so this finds the exact OLS solution.
            model_base = LinearRegression()
            model_base.fit(X_train_base, y_train)  # Train on (n_train, 256) features

            # r2_base: float — R-squared on the test set for the baseline model.
            # R-squared = 1 - (SS_res / SS_tot), where SS_res is the sum of squared
            # residuals and SS_tot is the total variance of y_test.
            # R^2 = 1.0 means perfect prediction; R^2 = 0.0 means no better than predicting
            # the mean distance for every pair.
            r2_base = r2_score(y_test, model_base.predict(X_test_base))
            results_baseline.append(r2_base)

            del X_train_base, X_test_base  # Free baseline train/test matrices

            # ============================================================
            # --- Augmented model (296-dim) ---
            # ============================================================
            # Uses pair-block features (256-dim) PLUS amino acid one-hot encoding (40-dim).
            # By comparing R-squared with the baseline, we measure the marginal
            # contribution of amino acid identity beyond what the pair representation captures.

            # X_aug: np.ndarray shape (n_total_pairs, 296) — horizontal concatenation of
            # the 256-dim pair-block features and the 40-dim AA one-hot features.
            # Columns 0-255: pair-block features (same as baseline).
            # Columns 256-295: AA one-hot encoding (from aa_all, computed in Phase 1).
            X_aug = np.hstack([X_pb, aa_all])

            # scaler_aug: StandardScaler — fit on the augmented training features.
            # Note: the one-hot columns (all 0s and 1s) will also be standardized,
            # which is fine — it puts them on the same scale as the pair-block features.
            scaler_aug = StandardScaler()
            X_train_aug = scaler_aug.fit_transform(X_aug[idx_train])  # shape (n_train, 296)
            X_test_aug = scaler_aug.transform(X_aug[idx_test])         # shape (n_test, 296)

            # model_aug: LinearRegression — OLS regression on the augmented feature space.
            model_aug = LinearRegression()
            model_aug.fit(X_train_aug, y_train)  # Train on (n_train, 296) features

            # y_pred_aug: np.ndarray of float, shape (n_test,) — predicted CA-CA distances
            # from the augmented model on the test set.
            y_pred_aug = model_aug.predict(X_test_aug)

            # r2_aug: float — R-squared on the test set for the augmented model.
            # If r2_aug > r2_base, the amino acid identity features add predictive value
            # beyond what the pair representation already encodes.
            r2_aug = r2_score(y_test, y_pred_aug)
            results_augmented.append(r2_aug)

            # --- Save augmented model coefficients ---
            # Store the learned weights for every feature (296 coefficients + 1 intercept).
            # This allows post-hoc analysis of which features (pair-block channels or
            # specific amino acid indicators) have the largest influence on distance prediction.

            # coef_row: dict — one row for the coefficients CSV. Keys are "layer", "intercept",
            # and "coef_0" through "coef_295".
            coef_row = {"layer": layer, "intercept": float(model_aug.intercept_)}
            # model_aug.coef_: np.ndarray of float, shape (296,) — the learned regression
            # weights. coef_[i] is the weight for feature i after standardization.
            coef_row.update({f"coef_{i}": float(c) for i, c in enumerate(model_aug.coef_)})
            all_coefs_augmented.append(coef_row)

            # --- Generate predictions for the scatter plot (last layer only) ---
            # For the final layer, we want to plot predicted vs actual for ALL pairs
            # (not just test), so we standardize and predict on the full augmented matrix.
            if layer == n_layers - 1:
                # X_aug_std: np.ndarray shape (n_total_pairs, 296) — all pairs standardized
                # using the scaler fitted on training data only (consistent normalization).
                X_aug_std = scaler_aug.transform(X_aug)
                # last_layer_preds_augmented: np.ndarray of float, shape (n_total_pairs,)
                # Predicted CA-CA distances for every pair from every protein.
                last_layer_preds_augmented = model_aug.predict(X_aug_std)

            # Print per-layer results: baseline R², augmented R², and the improvement (delta).
            # A positive delta means the AA features add information; negative would be surprising.
            print(f"Layer {layer:2d}  R2_baseline={r2_base:.4f}  R2_augmented={r2_aug:.4f}  delta={r2_aug - r2_base:+.4f}")

            # Free large arrays before the next layer iteration to manage memory.
            # X_pb is the biggest one: shape (n_total_pairs, 256).
            del X_pb, X_aug, X_train_aug, X_test_aug

        except Exception as e:
            print(f"Layer {layer:2d}  ERROR: {e}")
            traceback.print_exc()
            # Record NaN for failed layers so the CSV and plots still have 48 rows.
            # This prevents index errors when accessing results_baseline[layer] later.
            results_baseline.append(float("nan"))
            results_augmented.append(float("nan"))
            # Still need a coef row for the CSV; use zeros as placeholders.
            all_coefs_augmented.append({"layer": layer, "intercept": 0.0})

    # ------------------------------------------------------------------
    # Phase 4: Save CSV outputs
    # ------------------------------------------------------------------
    # Two CSV files are produced:
    #   1. R-squared comparison: one row per layer showing baseline vs augmented R².
    #   2. Augmented model coefficients: one row per layer with all 296 weights + intercept.
    #
    # These CSVs enable downstream analysis without re-running the regression.

    # CSV 1: R² comparison — the primary output of this script.
    # Columns: layer (int), r2_baseline (float), r2_augmented (float),
    # r2_improvement (float = augmented - baseline), n_proteins (int), n_total_pairs (int).
    # 48 rows, one per layer.
    r2_csv = csv_dir / "layer_regression_r2_comparison.csv"
    with open(r2_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["layer", "r2_baseline", "r2_augmented", "r2_improvement", "n_proteins", "n_total_pairs"])
        for layer in range(n_layers):
            r2_b = results_baseline[layer]    # float: baseline R² for this layer
            r2_a = results_augmented[layer]   # float: augmented R² for this layer
            improvement = r2_a - r2_b         # float: R² improvement from adding AA features
            writer.writerow([
                layer, f"{r2_b:.6f}", f"{r2_a:.6f}", f"{improvement:.6f}",
                n_proteins, n_total_pairs,
            ])
    print(f"\nSaved R² comparison CSV -> {r2_csv}")

    # CSV 2: Augmented model coefficients — allows post-hoc analysis of feature importance.
    # One row per layer (48 rows). Columns: "layer", "intercept", "coef_0" through "coef_295".
    # Coefficients 0-255 correspond to the 256 pair-block features (standardized).
    # Coefficients 256-295 correspond to the 40 amino acid one-hot features (standardized).
    coef_csv = csv_dir / "layer_regression_coefficients_augmented.csv"
    # fieldnames: list of str — column headers for the CSV. 298 columns total:
    # "layer" + "intercept" + 296 coefficient columns.
    fieldnames = ["layer", "intercept"] + [f"coef_{i}" for i in range(AUGMENTED_FEATURE_DIM)]
    with open(coef_csv, "w", newline="") as f:
        # DictWriter writes dicts as CSV rows, matching keys to fieldnames.
        # extrasaction="ignore" silently drops any extra keys not in fieldnames
        # (shouldn't happen, but defensive coding).
        writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(all_coefs_augmented)
    print(f"Saved augmented coefficients CSV -> {coef_csv}")

    # ------------------------------------------------------------------
    # Phase 5: Visualizations
    # ------------------------------------------------------------------
    # Two plots are generated:
    #   1. R² vs layer comparison: shows how baseline and augmented R² evolve across layers.
    #   2. Predicted vs actual scatter: last layer only, colored by protein.
    #
    # Input: results_baseline (list of float), results_augmented (list of float),
    #        last_layer_preds_augmented (np.ndarray), y_all, protein_offsets.
    # Output: PNG files saved to vis_dir.

    # Plot 1: R² vs layer comparison — the main result visualization.
    # Two lines show baseline (pair-block only) and augmented (pair-block + AA) R².
    # The gap between the lines reveals how much amino acid identity adds at each layer.
    # If the lines converge at later layers, it means the pair representation gradually
    # learns to encode amino acid identity information internally.
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(range(n_layers), results_baseline, marker="o", markersize=3, linewidth=1.5,
            color="steelblue", label="Baseline (256-dim, pair block only)")
    ax.plot(range(n_layers), results_augmented, marker="s", markersize=3, linewidth=1.5,
            color="darkorange", label="Augmented (296-dim, pair block + AA)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("Test R²")
    ax.set_title(f"AA-Augmented Regression: R² vs Layer ({n_proteins} proteins, {n_total_pairs:,} pairs)")
    ax.set_ylim(0, 1)  # R² ranges from 0 to 1 for a good model
    ax.legend()
    ax.grid(True, alpha=0.3)  # Light grid lines for readability
    fig.tight_layout()  # Adjust layout so labels don't get clipped
    fig.savefig(vis_dir / "layer_r2_comparison.png", dpi=150)  # 150 DPI for publication-quality
    plt.close(fig)  # Close the figure to free memory (important in loops or long scripts)
    print(f"Saved R² comparison plot -> {vis_dir / 'layer_r2_comparison.png'}")

    # Plot 2: Predicted vs actual CA-CA distances (last layer, augmented model).
    # Each point is one residue pair. X-axis = actual distance, Y-axis = predicted distance.
    # A perfect model would put all points on the y=x diagonal.
    # Points are colored by protein to reveal whether some proteins are harder to predict.
    if last_layer_preds_augmented is not None:
        # COLORS: list of str — matplotlib's default color cycle. Used to assign a
        # distinct color to each protein's scatter points. If there are more proteins
        # than colors, they wrap around (using modulo).
        COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

        fig, ax = plt.subplots(figsize=(8, 7))
        # Iterate over proteins using protein_offsets to slice the concatenated arrays.
        # idx_p: int — index of the current protein (0, 1, 2, ...).
        # pid: str — protein identifier (e.g. "7b3a_A").
        # start, end: int — slice boundaries in y_all / last_layer_preds_augmented.
        for idx_p, (pid, start, end) in enumerate(protein_offsets):
            # y_act: np.ndarray of float, shape (end - start,) — actual CA-CA distances
            # for this protein's pairs.
            y_act = y_all[start:end]
            # y_pred: np.ndarray of float, shape (end - start,) — predicted CA-CA distances
            # for this protein's pairs from the augmented model.
            y_pred = last_layer_preds_augmented[start:end]
            ax.scatter(
                y_act, y_pred,
                s=2, alpha=0.3,  # Small markers with transparency to handle dense overlaps
                color=COLORS[idx_p % len(COLORS)],  # Wrap color index if more proteins than colors
                label=f"{pid} ({end - start:,} pairs)",  # Legend entry with pair count
            )

        # Draw y=x reference line (perfect prediction diagonal).
        # lim_max: float — upper limit for both axes, 5% above the max actual distance
        # to give visual padding.
        lim_max = y_all.max() * 1.05
        ax.plot([0, lim_max], [0, lim_max], "k--", linewidth=1, label="y = x")
        ax.set_xlim(0, lim_max)
        ax.set_ylim(0, lim_max)
        ax.set_xlabel("Actual CA-CA Distance (Å)")
        ax.set_ylabel("Predicted CA-CA Distance (Å)")

        # r2_last_aug: float — R-squared for the last layer's augmented model.
        # Displayed in the plot title for quick reference.
        r2_last_aug = results_augmented[-1]
        ax.set_title(f"Layer {n_layers - 1}: Augmented Model (R² = {r2_last_aug:.4f})")
        ax.legend(markerscale=4, fontsize=9)  # markerscale=4 makes legend markers larger for visibility
        fig.tight_layout()
        fig.savefig(vis_dir / "pred_vs_actual_augmented.png", dpi=150)
        plt.close(fig)
        print(f"Saved pred vs actual plot -> {vis_dir / 'pred_vs_actual_augmented.png'}")

    # ------------------------------------------------------------------
    # Phase 6: Console summary
    # ------------------------------------------------------------------
    # Print a human-readable summary of the run: which proteins were included,
    # which were skipped, feature dimensions, pair counts, and the best-performing
    # layer for both baseline and augmented models.
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Included proteins ({n_proteins}):")
    for pid, start, end in protein_offsets:
        # Look up the residue count for this protein from protein_data.
        # n_res: int — number of residues in this protein (same as n_ca from Phase 1).
        n_res = next(d["n_residues"] for d in protein_data if d["protein_id"] == pid)
        print(f"  {pid}: {n_res} residues, {end - start:,} pairs")
    if skipped:
        print(f"\nSkipped proteins ({len(skipped)}):")
        for pid, reason in skipped:
            print(f"  {pid}: {reason}")

    print(f"\nFeature dimensions: baseline={FEATURE_DIM}, augmented={AUGMENTED_FEATURE_DIM}")
    print(f"Total pairs: {n_total_pairs:,} (train: {len(idx_train):,}, test: {len(idx_test):,})")

    # Filter out NaN results (from failed layers) before computing summary statistics.
    # valid_base: list of float — non-NaN baseline R² values.
    valid_base = [r for r in results_baseline if not math.isnan(r)]
    # valid_aug: list of float — non-NaN augmented R² values.
    valid_aug = [r for r in results_augmented if not math.isnan(r)]
    if valid_base and valid_aug:
        # best_base: int — layer index (0-47) with the highest baseline R².
        # np.nanargmax ignores NaN values when finding the argmax.
        best_base = int(np.nanargmax(results_baseline))
        # best_aug: int — layer index (0-47) with the highest augmented R².
        best_aug = int(np.nanargmax(results_augmented))
        print(f"\nBaseline best:  layer {best_base}  R² = {results_baseline[best_base]:.4f}")
        print(f"Augmented best: layer {best_aug}  R² = {results_augmented[best_aug]:.4f}")
        # Also report the final layer (47) specifically, since the last layer's
        # representations are what AlphaFold uses to make its structure prediction.
        print(f"\nLayer {n_layers - 1}:")
        print(f"  Baseline:    R² = {results_baseline[-1]:.4f}")
        print(f"  Augmented:   R² = {results_augmented[-1]:.4f}")
        # Improvement: positive means AA features helped; negative would be surprising.
        print(f"  Improvement: {results_augmented[-1] - results_baseline[-1]:+.4f}")


if __name__ == "__main__":
    main()

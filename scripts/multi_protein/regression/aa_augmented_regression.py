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
import traceback
from pathlib import Path

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48          # AlphaFold outputs 48 pair representation layers (0-47)
N_CHANNELS = 128       # Each pair block entry has 128 channels per direction
FEATURE_DIM = 2 * N_CHANNELS  # 256: we concatenate pb[i,j,:] (128) and pb[j,i,:] (128)
TEST_SIZE = 0.2        # 20% of all residue pairs held out for test set
RANDOM_STATE = 42      # Fixed seed for reproducible train/test splits

# AUGMENTED_FEATURE_DIM: int — 296. The augmented model uses the 256-dim pair-block
# features PLUS the 40-dim amino acid one-hot encoding: 256 + 40 = 296.
# This lets us measure how much additional predictive power amino acid identity
# adds on top of the learned pair representations.
AUGMENTED_FEATURE_DIM = FEATURE_DIM + 40  # 296

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

sys.path.insert(0, str(PROJECT_ROOT))
from lib.aa_features import AA_FEATURE_DIM, build_aa_onehot_features as _build_aa_onehot_features
from lib.output_writers import ensure_output_dirs
from lib.protein_dataset import load_local_protein_data
from lib.regression_common import (
    build_layer_feature_matrix,
    concatenate_targets,
    compute_protein_offsets,
    split_indices,
)


# ===========================================================================
# Helper functions — amino acid one-hot encoding
# ===========================================================================

def build_aa_onehot_features(residue_data, struct_i, struct_j):
    """Backward-compatible wrapper around shared AA one-hot helper."""
    return _build_aa_onehot_features(residue_data, struct_i, struct_j)


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
    csv_dir, vis_dir = ensure_output_dirs(output_dir)

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

    protein_data, skipped = load_local_protein_data(proteins_dir, include_distances=True)
    print(f"\nDiscovered {len(protein_data) + len(skipped)} protein directories.")
    for d in protein_data:
        ca_data = {"residues": d["residues"]}
        d["aa_features"] = build_aa_onehot_features(ca_data, d["pb_i"], d["pb_j"])
        print(f"[OK]   {d['protein_id']}: {d['n_residues']} residues, {len(d['y']):,} pairs")
    for protein_id, reason in skipped:
        print(f"[SKIP] {protein_id}: {reason}")

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
    y_all = concatenate_targets(protein_data)

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
    protein_offsets = compute_protein_offsets(protein_data)

    print(f"\nTotal pairs across {n_proteins} protein(s): {n_total_pairs:,}")

    # Create a train/test split of INDICES (not the data itself, since pair-block
    # features are loaded one layer at a time in Phase 3 to save memory).
    # all_idx: np.ndarray of int, shape (n_total_pairs,) — [0, 1, 2, ..., n_total_pairs-1]
    # idx_train, idx_test: np.ndarray of int — indices into y_all / aa_all / X_all.
    # These define which pairs are used for training and which for evaluation.
    # The same split is reused for every layer so results are comparable across layers.
    # Note: unlike the logistic regression script, we do NOT stratify here because
    # this is a regression task (continuous y), not classification.
    idx_train, idx_test = split_indices(
        n_samples=n_total_pairs,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

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

            X_pb = build_layer_feature_matrix(protein_data, layer, mode="base")

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

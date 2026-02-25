#!/usr/bin/env python3
"""
mds_reconstruction.py
=====================
For each of 48 AlphaFold layers, trains a pooled linear regression to predict
CA-CA distances from pair-block features, then uses classical MDS to reconstruct
3D coordinates from the predicted distance matrix. Compares MDS-reconstructed
structures to actual PDB coordinates via TM-score and RMSD.

This goes beyond R² by testing whether predicted distances are *geometrically
coherent* — i.e., do they correspond to a realizable 3D protein structure?

Outputs (inside proteins/multi_protein_results/mds_reconstruction/):
    csv_files/mds_reconstruction_metrics.csv
    csv_files/mds_eigenvalue_spectrum.csv
    csv_files/mds_reconstructed_coords_layer47.csv
    visualizations/mds_tmscores_vs_layer.png
    visualizations/mds_rmsd_vs_layer.png
    visualizations/mds_eigenvalue_fraction_vs_layer.png
    visualizations/mds_reconstruction_overlay_layer47.png
    visualizations/mds_r2_vs_tmscore.png
"""

import csv
import math
import sys
import traceback
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401 (needed for 3D projection)
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
# lib/ imports
PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
from lib.data_loading import discover_proteins, find_pdb_file, get_pair_block_path
from lib.structure import extract_ca_coordinates, compute_distance_matrix, superimpose_structures, compute_tm_score


# ===========================================================================
# Configuration
# ===========================================================================

N_LAYERS = 48
N_CHANNELS = 128
FEATURE_DIM = 2 * N_CHANNELS  # 256
TEST_SIZE = 0.2
RANDOM_STATE = 42

# Layers to save full eigenvalue spectra for
EIGENVALUE_LAYERS = [0, 11, 23, 35, 47]

PROTEINS_DIR = PROJECT_ROOT / "proteins"
OUTPUT_DIR = PROTEINS_DIR / "multi_protein_results" / "mds_reconstruction"


# ===========================================================================
# Classical MDS
# ===========================================================================

def classical_mds(distance_matrix, n_dims=3):
    """Classical (Torgerson) MDS: recover n_dims-dimensional coordinates from a
    pairwise distance matrix.

    The core idea: given only the distances between N points, reconstruct where
    those points are in space. This is the inverse of computing a distance
    matrix — instead of coords -> distances, we go distances -> coords.

    Mathematical overview:
        1. Square the distances to get D².
        2. "Double center" D² to convert it into a Gram matrix B (inner products).
           This removes the arbitrary choice of origin that coordinates depend on.
        3. Eigendecompose B. The top eigenvalues tell us which dimensions carry
           the most variance (like PCA). For points that truly live in 3D, only
           3 eigenvalues will be large; the rest will be ~zero.
        4. Reconstruct coordinates: eigenvectors scaled by sqrt(eigenvalues).

    Parameters
    ----------
    distance_matrix : np.ndarray, shape (n, n)
        Symmetric matrix of pairwise distances. D[i,j] = distance between
        point i and point j. Diagonal should be zero.
    n_dims : int
        Number of dimensions to embed into (default 3 for 3D reconstruction).

    Returns
    -------
    coords : np.ndarray, shape (n, n_dims)
        Reconstructed coordinates. These are determined up to rotation and
        reflection — MDS cannot recover the original orientation, only the
        relative geometry. We fix this later with Kabsch alignment.
    eigenvalues : np.ndarray, shape (n,)
        ALL eigenvalues of the Gram matrix B, sorted largest to smallest.
        Diagnostic: if the top 3 dominate, the distances are well-explained
        by 3D geometry. Large negative eigenvalues indicate the distance
        matrix is non-Euclidean (can't be realized by any set of 3D points).
    """
    n = distance_matrix.shape[0]

    # STEP 1: Square all distances.
    # Why? The fundamental identity connecting distances to coordinates is:
    #   d(i,j)² = ||x_i||² + ||x_j||² - 2 * x_i · x_j
    # We need d² (not d) because the dot product relationship is quadratic.
    D_sq = distance_matrix ** 2

    # STEP 2: Double centering — convert squared distances into inner products.
    #
    # The squared distance d²_ij = ||x_i||² + ||x_j||² - 2*(x_i · x_j)
    # contains three terms, but we only want the dot product x_i · x_j.
    # Double centering removes the ||x_i||² and ||x_j||² terms by subtracting
    # row means, column means, and adding back the grand mean:
    #
    #   B_ij = -0.5 * (D²_ij - mean_of_row_i - mean_of_col_j + grand_mean)
    #
    # After this transformation, B_ij = x_i · x_j (inner product), where the
    # coordinates x are centered at the origin (centroid = 0). This is called
    # the Gram matrix.
    #
    # Mathematically equivalent to B = -0.5 * H @ D_sq @ H where
    # H = I - (1/n)*11^T is the centering matrix, but computing via means
    # is more memory-efficient (avoids creating the n×n matrix H).
    row_mean = D_sq.mean(axis=1, keepdims=True)   # mean of each row (n×1)
    col_mean = D_sq.mean(axis=0, keepdims=True)   # mean of each column (1×n)
    grand_mean = D_sq.mean()                       # overall mean (scalar)
    B = -0.5 * (D_sq - row_mean - col_mean + grand_mean)

    # STEP 3: Eigendecomposition of the Gram matrix.
    #
    # Since B = X @ X^T (coordinates times their transpose), its eigenvalues
    # tell us the variance along each principal axis — exactly like PCA.
    #   - For perfect 3D distances: exactly 3 positive eigenvalues, rest = 0
    #   - For noisy/predicted distances: eigenvalues taper off gradually
    #   - Negative eigenvalues: the distances are "non-Euclidean" — no set of
    #     points in any dimension could produce exactly these distances. This
    #     happens when predictions violate the triangle inequality.
    #
    # np.linalg.eigh is used because B is symmetric (B_ij = B_ji, since
    # dot products are symmetric). eigh is faster and more numerically stable
    # than the general eig for symmetric matrices.
    eigenvalues, eigenvectors = np.linalg.eigh(B)

    # eigh returns eigenvalues in ascending order (smallest first).
    # We want descending order (largest first) so the top-n_dims are first.
    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    # STEP 4: Reconstruct coordinates from the top eigenvalues/eigenvectors.
    #
    # Since B = X @ X^T, and the eigendecomposition gives B = V @ Λ @ V^T,
    # we can reconstruct X = V @ sqrt(Λ). Each eigenvector gives one axis of
    # the coordinate system, scaled by the square root of its eigenvalue
    # (which represents the spread of points along that axis).
    #
    # We take only the top n_dims eigenvalues/eigenvectors (typically 3),
    # discarding the rest. This is the best rank-n_dims approximation of B
    # (by the Eckart-Young theorem — same principle as truncated SVD/PCA).
    #
    # Negative eigenvalues are clamped to zero because sqrt of a negative
    # number is undefined. These correspond to non-Euclidean "phantom
    # dimensions" that arise from prediction errors.
    top_evals = np.maximum(eigenvalues[:n_dims], 0.0)
    coords = eigenvectors[:, :n_dims] * np.sqrt(top_evals)

    return coords, eigenvalues


# ===========================================================================
# Main
# ===========================================================================

def main():
    proteins_dir = PROTEINS_DIR.resolve()
    output_dir = OUTPUT_DIR.resolve()
    csv_dir = output_dir / "csv_files"
    vis_dir = output_dir / "visualizations"
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)

    print(f"Proteins directory: {proteins_dir}")
    print(f"Output directory:   {output_dir}")
    print(f"Layers:             {N_LAYERS}")

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

            ca_data = extract_ca_coordinates(pdb_path)
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

            dist_matrix = compute_distance_matrix(ca_data["coordinates"])
            struct_i, struct_j = np.triu_indices(n_ca, k=1)
            y_protein = dist_matrix[struct_i, struct_j]

            protein_data.append({
                "protein_id": protein_id,
                "protein_dir": protein_dir,
                "n_residues": n_ca,
                "coordinates": ca_data["coordinates"],  # ground truth 3D coords
                "residues": ca_data["residues"],
                "pb_i": struct_i,
                "pb_j": struct_j,
                "y": y_protein,
            })
            print(f"[OK]   {protein_id}: {n_ca} residues, {len(y_protein):,} pairs")

        except Exception as e:
            skipped.append((protein_id, f"unexpected error -- {e}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {e}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    n_proteins = len(protein_data)

    # ------------------------------------------------------------------
    # Phase 2: Pool targets and create train/test split
    # ------------------------------------------------------------------
    y_all = np.concatenate([d["y"] for d in protein_data])
    n_total_pairs = len(y_all)

    protein_offsets = []
    offset = 0
    for d in protein_data:
        n = len(d["y"])
        protein_offsets.append((d["protein_id"], offset, offset + n))
        offset += n

    print(f"\nTotal pairs across {n_proteins} protein(s): {n_total_pairs:,}")

    all_idx = np.arange(n_total_pairs)
    idx_train, idx_test = train_test_split(
        all_idx, test_size=TEST_SIZE, random_state=RANDOM_STATE
    )
    y_train = y_all[idx_train]

    # ------------------------------------------------------------------
    # Phase 3: Layer-wise regression + MDS reconstruction
    # ------------------------------------------------------------------
    # Storage for per-(layer, protein) metrics
    metrics_rows = []
    eigenvalue_rows = []
    last_layer_coords = {}  # protein_id -> (mds_aligned, pdb_coords)

    print()
    for layer in range(N_LAYERS):
        try:
            # --- Load features ---
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

            # --- Train regression ---
            scaler = StandardScaler()
            X_train = scaler.fit_transform(X_all[idx_train])

            model = LinearRegression()
            model.fit(X_train, y_train)

            # --- Predict on ALL pairs ---
            X_all_std = scaler.transform(X_all)
            y_pred_all = model.predict(X_all_std)

            # Test R² for context
            r2_test = r2_score(y_all[idx_test], y_pred_all[idx_test])

            del X_all, X_train, X_all_std

            # --- Per-protein MDS reconstruction ---
            protein_metrics = []
            for p_idx, (pid, start, end) in enumerate(protein_offsets):
                d = protein_data[p_idx]
                n = d["n_residues"]
                true_coords = d["coordinates"]

                # Reconstruct full symmetric predicted distance matrix
                pred_dist = np.zeros((n, n))
                y_pred_protein = y_pred_all[start:end]
                pred_dist[d["pb_i"], d["pb_j"]] = y_pred_protein
                pred_dist[d["pb_j"], d["pb_i"]] = y_pred_protein

                # Clamp negative predictions to zero
                n_negative_preds = int(np.sum(y_pred_protein < 0))
                pred_dist = np.maximum(pred_dist, 0.0)

                # Classical MDS
                mds_coords, eigenvalues = classical_mds(pred_dist, n_dims=3)

                # Eigenvalue diagnostics
                pos_mask = eigenvalues > 0
                sum_pos = np.sum(eigenvalues[pos_mask]) if np.any(pos_mask) else 1.0
                frac_top3 = float(np.sum(np.maximum(eigenvalues[:3], 0.0)) / sum_pos)
                n_neg_evals = int(np.sum(eigenvalues < 0))
                neg_frac = float(abs(np.sum(eigenvalues[eigenvalues < 0])) / sum_pos) if n_neg_evals > 0 else 0.0

                # Kabsch alignment + metrics.
                # MDS is ambiguous up to reflection (mirror image). The Kabsch
                # algorithm only applies proper rotations (det=+1), so if MDS
                # produced a mirror image, rotation-only alignment will be poor.
                # Fix: try both orientations, keep the one with lower RMSD.
                mds_reflected = mds_coords.copy()
                mds_reflected[:, 2] *= -1  # flip z-axis to get mirror image

                transformed_orig, rmsd_orig = superimpose_structures(true_coords, mds_coords)
                transformed_flip, rmsd_flip = superimpose_structures(true_coords, mds_reflected)

                if rmsd_flip < rmsd_orig:
                    transformed, rmsd = transformed_flip, rmsd_flip
                else:
                    transformed, rmsd = transformed_orig, rmsd_orig

                tm = compute_tm_score(true_coords, transformed, n)

                # Stress: how well MDS coords reproduce the predicted distances
                mds_dist = compute_distance_matrix(transformed)
                triu_i, triu_j = np.triu_indices(n, k=1)
                d_pred = pred_dist[triu_i, triu_j]
                d_mds = mds_dist[triu_i, triu_j]
                stress = float(np.sqrt(np.sum((d_pred - d_mds) ** 2) / np.sum(d_pred ** 2)))

                metrics_rows.append({
                    "layer": layer,
                    "protein_id": pid,
                    "n_residues": n,
                    "r2_test": r2_test,
                    "rmsd": rmsd,
                    "tm_score": tm,
                    "frac_top3_eigenvalues": frac_top3,
                    "n_negative_eigenvalues": n_neg_evals,
                    "neg_eigenvalue_fraction": neg_frac,
                    "n_negative_predictions": n_negative_preds,
                    "stress": stress,
                })
                protein_metrics.append((pid, tm, rmsd))

                # Save eigenvalue spectrum for selected layers
                if layer in EIGENVALUE_LAYERS:
                    cum_var = np.cumsum(np.maximum(eigenvalues, 0.0)) / sum_pos
                    for rank, ev in enumerate(eigenvalues):
                        eigenvalue_rows.append({
                            "layer": layer,
                            "protein_id": pid,
                            "eigenvalue_rank": rank,
                            "eigenvalue": float(ev),
                            "cumulative_variance_fraction": float(cum_var[rank]),
                        })

                # Save last-layer coordinates for overlay plots
                if layer == N_LAYERS - 1:
                    last_layer_coords[pid] = (transformed, true_coords, d["residues"])

            # Print summary for this layer
            tm_vals = [m[1] for m in protein_metrics]
            rmsd_vals = [m[2] for m in protein_metrics]
            print(f"Layer {layer:2d}  R²={r2_test:.4f}  "
                  f"TM={np.mean(tm_vals):.4f}  RMSD={np.mean(rmsd_vals):.2f}Å")

            del y_pred_all

        except Exception as e:
            print(f"Layer {layer:2d}  ERROR: {e}")
            traceback.print_exc()

    # ------------------------------------------------------------------
    # Phase 4: Save CSV outputs
    # ------------------------------------------------------------------

    # CSV 1: Metrics
    metrics_csv = csv_dir / "mds_reconstruction_metrics.csv"
    fieldnames = [
        "layer", "protein_id", "n_residues", "r2_test", "rmsd", "tm_score",
        "frac_top3_eigenvalues", "n_negative_eigenvalues", "neg_eigenvalue_fraction",
        "n_negative_predictions", "stress",
    ]
    with open(metrics_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in metrics_rows:
            row_fmt = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
            writer.writerow(row_fmt)
    print(f"\nSaved metrics CSV -> {metrics_csv}")

    # CSV 2: Eigenvalue spectrum
    ev_csv = csv_dir / "mds_eigenvalue_spectrum.csv"
    ev_fields = ["layer", "protein_id", "eigenvalue_rank", "eigenvalue", "cumulative_variance_fraction"]
    with open(ev_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=ev_fields)
        writer.writeheader()
        for row in eigenvalue_rows:
            row_fmt = {k: (f"{v:.6f}" if isinstance(v, float) else v) for k, v in row.items()}
            writer.writerow(row_fmt)
    print(f"Saved eigenvalue CSV -> {ev_csv}")

    # CSV 3: Reconstructed coordinates (last layer)
    coords_csv = csv_dir / "mds_reconstructed_coords_layer47.csv"
    with open(coords_csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "protein_id", "residue_idx", "residue_name", "residue_number",
            "mds_x", "mds_y", "mds_z", "pdb_x", "pdb_y", "pdb_z",
            "deviation_angstrom",
        ])
        for pid, (mds_c, pdb_c, residues) in last_layer_coords.items():
            deviations = np.sqrt(np.sum((pdb_c - mds_c) ** 2, axis=1))
            for i in range(len(residues)):
                writer.writerow([
                    pid, i, residues[i][1], residues[i][2],
                    f"{mds_c[i, 0]:.3f}", f"{mds_c[i, 1]:.3f}", f"{mds_c[i, 2]:.3f}",
                    f"{pdb_c[i, 0]:.3f}", f"{pdb_c[i, 1]:.3f}", f"{pdb_c[i, 2]:.3f}",
                    f"{deviations[i]:.4f}",
                ])
    print(f"Saved reconstructed coords CSV -> {coords_csv}")

    # ------------------------------------------------------------------
    # Phase 5: Visualizations
    # ------------------------------------------------------------------

    # Organize metrics by protein for plotting
    protein_ids = [d["protein_id"] for d in protein_data]
    COLORS = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    # Build per-protein arrays: [layer] -> value
    tm_by_protein = {pid: [] for pid in protein_ids}
    rmsd_by_protein = {pid: [] for pid in protein_ids}
    frac3_by_protein = {pid: [] for pid in protein_ids}
    r2_by_layer = []

    for row in metrics_rows:
        tm_by_protein[row["protein_id"]].append(row["tm_score"])
        rmsd_by_protein[row["protein_id"]].append(row["rmsd"])
        frac3_by_protein[row["protein_id"]].append(row["frac_top3_eigenvalues"])

    # Deduplicate r2 (same across proteins for a given layer)
    seen_layers = set()
    for row in metrics_rows:
        if row["layer"] not in seen_layers:
            r2_by_layer.append(row["r2_test"])
            seen_layers.add(row["layer"])

    layers = list(range(len(r2_by_layer)))

    # --- Plot 1: TM-score vs layer ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pid in enumerate(protein_ids):
        ax.plot(layers, tm_by_protein[pid], marker=".", markersize=4,
                linewidth=1.2, color=COLORS[i % len(COLORS)], label=pid, alpha=0.8)
    # Mean
    mean_tm = np.mean([tm_by_protein[pid] for pid in protein_ids], axis=0)
    ax.plot(layers, mean_tm, "k-", linewidth=2.5, label="mean", alpha=0.9)
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="TM=0.5 (fold threshold)")
    ax.set_xlabel("Layer")
    ax.set_ylabel("TM-score")
    ax.set_title("MDS Reconstruction: TM-score vs AlphaFold Layer")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "mds_tmscores_vs_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved TM-score plot -> {vis_dir / 'mds_tmscores_vs_layer.png'}")

    # --- Plot 2: RMSD vs layer ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pid in enumerate(protein_ids):
        ax.plot(layers, rmsd_by_protein[pid], marker=".", markersize=4,
                linewidth=1.2, color=COLORS[i % len(COLORS)], label=pid, alpha=0.8)
    mean_rmsd = np.mean([rmsd_by_protein[pid] for pid in protein_ids], axis=0)
    ax.plot(layers, mean_rmsd, "k-", linewidth=2.5, label="mean", alpha=0.9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("RMSD (Å)")
    ax.set_title("MDS Reconstruction: RMSD vs AlphaFold Layer")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "mds_rmsd_vs_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved RMSD plot -> {vis_dir / 'mds_rmsd_vs_layer.png'}")

    # --- Plot 3: Top-3 eigenvalue fraction vs layer ---
    fig, ax = plt.subplots(figsize=(10, 5))
    for i, pid in enumerate(protein_ids):
        ax.plot(layers, frac3_by_protein[pid], marker=".", markersize=4,
                linewidth=1.2, color=COLORS[i % len(COLORS)], label=pid, alpha=0.8)
    mean_frac = np.mean([frac3_by_protein[pid] for pid in protein_ids], axis=0)
    ax.plot(layers, mean_frac, "k-", linewidth=2.5, label="mean", alpha=0.9)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Fraction of Variance (Top 3 Eigenvalues)")
    ax.set_title("MDS Reconstruction: 3D Embeddability vs AlphaFold Layer")
    ax.set_ylim(0, 1.05)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "mds_eigenvalue_fraction_vs_layer.png", dpi=150)
    plt.close(fig)
    print(f"Saved eigenvalue fraction plot -> {vis_dir / 'mds_eigenvalue_fraction_vs_layer.png'}")

    # --- Plot 4: 3D overlay (last layer, 2x2 grid) ---
    n_cols = 2
    n_rows = math.ceil(n_proteins / n_cols)
    fig = plt.figure(figsize=(7 * n_cols, 6 * n_rows))

    for p_idx, pid in enumerate(protein_ids):
        if pid not in last_layer_coords:
            continue
        mds_c, pdb_c, _ = last_layer_coords[pid]
        ax = fig.add_subplot(n_rows, n_cols, p_idx + 1, projection="3d")

        ax.scatter(pdb_c[:, 0], pdb_c[:, 1], pdb_c[:, 2],
                   s=8, alpha=0.6, color="steelblue", label="PDB (actual)")
        ax.scatter(mds_c[:, 0], mds_c[:, 1], mds_c[:, 2],
                   s=8, alpha=0.6, color="tomato", label="MDS (reconstructed)")

        # Find layer-47 metrics for this protein
        l47_row = next(r for r in metrics_rows
                       if r["layer"] == N_LAYERS - 1 and r["protein_id"] == pid)
        ax.set_title(f"{pid}\nTM={l47_row['tm_score']:.3f}  RMSD={l47_row['rmsd']:.2f}Å",
                     fontsize=10)
        ax.legend(fontsize=8, loc="upper left")
        ax.set_xlabel("X (Å)", fontsize=8)
        ax.set_ylabel("Y (Å)", fontsize=8)
        ax.set_zlabel("Z (Å)", fontsize=8)

    fig.suptitle("Layer 47: PDB vs MDS-Reconstructed Coordinates", fontsize=14)
    fig.tight_layout()
    fig.savefig(vis_dir / "mds_reconstruction_overlay_layer47.png", dpi=150)
    plt.close(fig)
    print(f"Saved overlay plot -> {vis_dir / 'mds_reconstruction_overlay_layer47.png'}")

    # --- Plot 5: R² vs TM-score scatter ---
    fig, ax = plt.subplots(figsize=(8, 7))
    for i, pid in enumerate(protein_ids):
        r2_vals = [r2_by_layer[row["layer"]] for row in metrics_rows if row["protein_id"] == pid]
        tm_vals = [row["tm_score"] for row in metrics_rows if row["protein_id"] == pid]
        ax.scatter(r2_vals, tm_vals, s=15, alpha=0.6,
                   color=COLORS[i % len(COLORS)], label=pid)

    ax.set_xlabel("Test R²")
    ax.set_ylabel("TM-score")
    ax.set_title("Regression R² vs MDS Reconstruction TM-score")
    ax.axhline(y=0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6)
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(vis_dir / "mds_r2_vs_tmscore.png", dpi=150)
    plt.close(fig)
    print(f"Saved R² vs TM-score plot -> {vis_dir / 'mds_r2_vs_tmscore.png'}")

    # ------------------------------------------------------------------
    # Console summary
    # ------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"Included proteins ({n_proteins}):")
    for d in protein_data:
        print(f"  {d['protein_id']}: {d['n_residues']} residues")
    if skipped:
        print(f"\nSkipped proteins ({len(skipped)}):")
        for pid, reason in skipped:
            print(f"  {pid}: {reason}")

    # Last layer summary
    print(f"\nLayer {N_LAYERS - 1} results:")
    for pid in protein_ids:
        row = next(r for r in metrics_rows
                   if r["layer"] == N_LAYERS - 1 and r["protein_id"] == pid)
        print(f"  {pid}: TM={row['tm_score']:.4f}  RMSD={row['rmsd']:.2f}Å  "
              f"Top3EV={row['frac_top3_eigenvalues']:.4f}  Stress={row['stress']:.4f}")


if __name__ == "__main__":
    main()

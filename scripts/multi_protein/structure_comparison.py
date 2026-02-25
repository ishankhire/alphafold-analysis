#!/usr/bin/env python3
"""
structure_comparison.py
=======================
Compare AlphaFold predicted structures (PDB) against experimental ground truth (CIF)
for all available proteins. Computes RMSD, TM-score, per-residue CA deviation, and
pairwise distance matrix comparison after Kabsch superposition.

Outputs
-------
  Per-protein:
    proteins/<id>/csv_files/structure_comparison.csv
    proteins/<id>/visualizations/structure_comparison_deviation.png

  Multi-protein summary:
    proteins/multi_protein_results/csv_files/structure_comparison_summary.csv
    proteins/multi_protein_results/visualizations/structure_comparison_per_residue.png
    proteins/multi_protein_results/visualizations/structure_comparison_summary.png
"""

import sys
import os
import csv
from pathlib import Path

# scripts/multi_protein/ is 3 levels deep; climb to project root.
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import pearsonr

from lib.alignment import build_alignment, validate_alignment
from lib.structure import (
    extract_residue_coordinates,
    extract_pdb_ca_coordinates,
    compute_distance_matrix,
    detect_chain,
    superimpose_structures,
    compute_tm_score,
)

# ---------------------------------------------------------------------------
# Output directories
# ---------------------------------------------------------------------------
PROTEINS_ROOT = PROJECT_ROOT / "proteins"
RESULTS_DIR = PROTEINS_ROOT / "multi_protein_results"
CSV_DIR = RESULTS_DIR / "csv_files"
VIS_DIR = RESULTS_DIR / "visualizations"
CSV_DIR.mkdir(parents=True, exist_ok=True)
VIS_DIR.mkdir(parents=True, exist_ok=True)

PROTEINS = ["7b3a_A", "6tf4_A", "6wmk_A", "6wqc_A"]


def find_cif_file(protein_dir):
    """Find the CIF file in a protein directory (case-insensitive)."""
    cif_files = list(protein_dir.glob("*.cif"))
    if len(cif_files) == 1:
        return cif_files[0]
    raise FileNotFoundError(f"Expected 1 CIF file in {protein_dir}, found {len(cif_files)}")


def find_pdb_file(protein_dir):
    """Find the predicted PDB file."""
    pdb_dir = protein_dir / "predicted_pdb"
    pdb_files = list(pdb_dir.glob("*.pdb"))
    if len(pdb_files) == 1:
        return pdb_files[0]
    raise FileNotFoundError(f"Expected 1 PDB file in {pdb_dir}, found {len(pdb_files)}")


def process_protein(protein_dir, protein_id):
    """Process one protein: align, superimpose, compute metrics.

    Returns a summary dict or None on failure.
    """
    pdb_id = protein_id.split("_")[0]  # e.g. "7b3a" from "7b3a_A"

    # --- Locate files ---
    cif_path = find_cif_file(protein_dir)
    pdb_path = find_pdb_file(protein_dir)
    chain_id = detect_chain(protein_dir)
    if chain_id is None:
        chain_id = "A"

    print(f"\n{'='*60}")
    print(f"Processing {protein_id}")
    print(f"  CIF: {cif_path.name}")
    print(f"  PDB: {pdb_path.name}")
    print(f"  Chain: {chain_id}")

    # --- Build alignment ---
    alignment = build_alignment(str(cif_path), chain_id, pdb_id)

    # Validate alignment (need a pair block path)
    pb_dir = protein_dir / "pair_blocks"
    first_pb = sorted(pb_dir.glob("*.npy"))[0]

    # --- Extract CIF CA coordinates (ground truth, resolved residues only) ---
    cif_data = extract_residue_coordinates(str(cif_path), chain_id)
    n_cif = len(cif_data["residues"])

    validate_alignment(alignment, str(first_pb), n_cif)

    print(f"  CIF residues: {n_cif}")
    print(f"  Entity length: {alignment.n_entity}")
    print(f"  pb_indices range: [{alignment.pb_indices[0]}..{alignment.pb_indices[-1]}]")

    cif_coords = cif_data["coordinates"]  # shape (n_cif, 3)

    # --- Extract PDB CA coordinates (predicted, all entity residues) ---
    pdb_data = extract_pdb_ca_coordinates(str(pdb_path))
    n_pdb = len(pdb_data["residues"])
    print(f"  PDB CA atoms: {n_pdb}")

    # Build lookup: PDB residue number -> index in pdb_data arrays
    pdb_resnum_to_idx = {res[2]: i for i, res in enumerate(pdb_data["residues"])}

    # --- Match residues: CIF residue i -> PDB residue at (pb_indices[i] + 1) ---
    matched_pdb_coords = []
    matched_indices = []
    for i, pb_idx in enumerate(alignment.pb_indices):
        pdb_resnum = pb_idx + 1  # PDB is 1-indexed
        if pdb_resnum not in pdb_resnum_to_idx:
            print(f"  WARNING: PDB residue {pdb_resnum} not found (CIF residue {i})")
            continue
        matched_pdb_coords.append(pdb_data["coordinates"][pdb_resnum_to_idx[pdb_resnum]])
        matched_indices.append(i)

    matched_pdb_coords = np.array(matched_pdb_coords)
    matched_cif_coords = cif_coords[matched_indices]
    n_matched = len(matched_indices)
    print(f"  Matched residues: {n_matched}")

    # --- Kabsch superposition ---
    transformed_coords, rmsd = superimpose_structures(matched_cif_coords, matched_pdb_coords)
    print(f"  RMSD: {rmsd:.3f} Å")

    # --- TM-score ---
    tm = compute_tm_score(matched_cif_coords, transformed_coords, n_matched)
    print(f"  TM-score: {tm:.4f}")

    # --- Per-residue CA deviation ---
    per_res_dist = np.sqrt(np.sum((matched_cif_coords - transformed_coords) ** 2, axis=1))
    print(f"  Mean deviation: {np.mean(per_res_dist):.3f} Å")
    print(f"  Max deviation:  {np.max(per_res_dist):.3f} Å")
    print(f"  Median deviation: {np.median(per_res_dist):.3f} Å")

    # --- Distance matrix comparison ---
    dist_cif = compute_distance_matrix(matched_cif_coords)
    dist_pred = compute_distance_matrix(transformed_coords)
    dist_diff = np.abs(dist_pred - dist_cif)

    # Exclude diagonal for statistics
    mask = ~np.eye(n_matched, dtype=bool)
    dist_mae = np.mean(dist_diff[mask])
    dist_max_err = np.max(dist_diff[mask])

    # Pearson correlation on upper triangle (excluding diagonal)
    triu_idx = np.triu_indices(n_matched, k=1)
    r_corr, _ = pearsonr(dist_cif[triu_idx], dist_pred[triu_idx])
    print(f"  Distance matrix MAE: {dist_mae:.3f} Å")
    print(f"  Distance matrix max error: {dist_max_err:.3f} Å")
    print(f"  Distance matrix Pearson r: {r_corr:.6f}")

    # --- Save per-protein CSV ---
    csv_dir = protein_dir / "csv_files"
    csv_dir.mkdir(parents=True, exist_ok=True)
    csv_path = csv_dir / "structure_comparison.csv"

    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "cif_residue_idx", "residue_name", "residue_number", "entity_position",
            "ca_deviation_angstrom",
            "cif_x", "cif_y", "cif_z",
            "pred_x", "pred_y", "pred_z",
        ])
        for k, i in enumerate(matched_indices):
            res = cif_data["residues"][i]
            writer.writerow([
                i, res[1], res[2], int(alignment.pb_indices[i]),
                f"{per_res_dist[k]:.4f}",
                f"{matched_cif_coords[k, 0]:.3f}",
                f"{matched_cif_coords[k, 1]:.3f}",
                f"{matched_cif_coords[k, 2]:.3f}",
                f"{transformed_coords[k, 0]:.3f}",
                f"{transformed_coords[k, 1]:.3f}",
                f"{transformed_coords[k, 2]:.3f}",
            ])
    print(f"  Saved: {csv_path}")

    # --- Per-protein deviation plot ---
    vis_dir = protein_dir / "visualizations"
    vis_dir.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(matched_indices, per_res_dist, linewidth=0.8)
    ax.set_xlabel("Residue index")
    ax.set_ylabel("CA deviation (Å)")
    ax.set_title(f"{protein_id}: Per-residue CA deviation (RMSD={rmsd:.2f} Å, TM={tm:.4f})")
    ax.axhline(rmsd, color="red", linestyle="--", alpha=0.5, label=f"RMSD={rmsd:.2f}")
    ax.legend()
    fig.tight_layout()
    fig.savefig(vis_dir / "structure_comparison_deviation.png", dpi=150)
    plt.close(fig)
    print(f"  Saved: {vis_dir / 'structure_comparison_deviation.png'}")

    return {
        "protein_id": protein_id,
        "n_resolved": n_matched,
        "n_entity": alignment.n_entity,
        "rmsd_angstrom": rmsd,
        "tm_score": tm,
        "mean_deviation_angstrom": float(np.mean(per_res_dist)),
        "median_deviation_angstrom": float(np.median(per_res_dist)),
        "max_deviation_angstrom": float(np.max(per_res_dist)),
        "dist_matrix_mae": dist_mae,
        "dist_matrix_max_error": dist_max_err,
        "dist_matrix_pearson_r": r_corr,
        "per_res_dist": per_res_dist,
        "matched_indices": matched_indices,
    }


def save_summary_csv(results):
    """Save multi-protein summary CSV."""
    path = CSV_DIR / "structure_comparison_summary.csv"
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "protein_id", "n_resolved", "n_entity",
            "rmsd_angstrom", "tm_score",
            "mean_deviation_angstrom", "median_deviation_angstrom", "max_deviation_angstrom",
            "dist_matrix_mae", "dist_matrix_max_error", "dist_matrix_pearson_r",
        ])
        for r in results:
            writer.writerow([
                r["protein_id"], r["n_resolved"], r["n_entity"],
                f"{r['rmsd_angstrom']:.4f}", f"{r['tm_score']:.4f}",
                f"{r['mean_deviation_angstrom']:.4f}",
                f"{r['median_deviation_angstrom']:.4f}",
                f"{r['max_deviation_angstrom']:.4f}",
                f"{r['dist_matrix_mae']:.4f}",
                f"{r['dist_matrix_max_error']:.4f}",
                f"{r['dist_matrix_pearson_r']:.6f}",
            ])
    print(f"\nSummary CSV saved: {path}")


def plot_per_residue_grid(results):
    """2x2 grid of per-residue deviation plots."""
    n = len(results)
    cols = 2
    rows = (n + 1) // 2
    fig, axes = plt.subplots(rows, cols, figsize=(14, 4 * rows))
    axes = axes.flatten()

    for idx, r in enumerate(results):
        ax = axes[idx]
        ax.plot(r["matched_indices"], r["per_res_dist"], linewidth=0.8)
        ax.axhline(r["rmsd_angstrom"], color="red", linestyle="--", alpha=0.5)
        ax.set_title(f"{r['protein_id']} (RMSD={r['rmsd_angstrom']:.2f} Å, TM={r['tm_score']:.4f})")
        ax.set_xlabel("Residue index")
        ax.set_ylabel("CA deviation (Å)")

    # Hide unused axes
    for idx in range(len(results), len(axes)):
        axes[idx].set_visible(False)

    fig.tight_layout()
    path = VIS_DIR / "structure_comparison_per_residue.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Per-residue grid saved: {path}")


def plot_summary_bars(results):
    """Bar chart comparing RMSD and TM-score across proteins."""
    names = [r["protein_id"] for r in results]
    rmsds = [r["rmsd_angstrom"] for r in results]
    tms = [r["tm_score"] for r in results]
    maes = [r["dist_matrix_mae"] for r in results]

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    x = np.arange(len(names))

    axes[0].bar(x, rmsds, color="steelblue")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(names, rotation=30, ha="right")
    axes[0].set_ylabel("RMSD (Å)")
    axes[0].set_title("RMSD after Kabsch superposition")

    axes[1].bar(x, tms, color="seagreen")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(names, rotation=30, ha="right")
    axes[1].set_ylabel("TM-score")
    axes[1].set_title("TM-score")
    axes[1].set_ylim(0, 1)
    axes[1].axhline(0.5, color="red", linestyle="--", alpha=0.5, label="Same fold threshold")
    axes[1].legend()

    axes[2].bar(x, maes, color="darkorange")
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(names, rotation=30, ha="right")
    axes[2].set_ylabel("MAE (Å)")
    axes[2].set_title("Distance matrix MAE")

    fig.tight_layout()
    path = VIS_DIR / "structure_comparison_summary.png"
    fig.savefig(path, dpi=150)
    plt.close(fig)
    print(f"Summary bar chart saved: {path}")


def main():
    results = []
    for protein_id in PROTEINS:
        protein_dir = PROTEINS_ROOT / protein_id
        if not protein_dir.exists():
            print(f"Skipping {protein_id}: directory not found")
            continue
        try:
            result = process_protein(protein_dir, protein_id)
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR processing {protein_id}: {e}")
            import traceback
            traceback.print_exc()

    if not results:
        print("No proteins processed successfully.")
        return

    save_summary_csv(results)
    plot_per_residue_grid(results)
    plot_summary_bars(results)

    print(f"\n{'='*60}")
    print("Summary:")
    print(f"{'Protein':<12} {'RMSD (Å)':>10} {'TM-score':>10} {'Dist MAE':>10} {'Pearson r':>10}")
    for r in results:
        print(f"{r['protein_id']:<12} {r['rmsd_angstrom']:>10.3f} {r['tm_score']:>10.4f} "
              f"{r['dist_matrix_mae']:>10.3f} {r['dist_matrix_pearson_r']:>10.6f}")


if __name__ == "__main__":
    main()

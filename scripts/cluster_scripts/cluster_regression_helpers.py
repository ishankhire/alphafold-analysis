#!/usr/bin/env python3
"""
Shared helper utilities for future cluster regression and diagnostic scripts.

This module centralizes only stable, duplicated helpers. Existing scripts are
not required to adopt it immediately.
"""

import os
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser


STANDARD_AAS = [
    "ALA",
    "ARG",
    "ASN",
    "ASP",
    "CYS",
    "GLU",
    "GLN",
    "GLY",
    "HIS",
    "ILE",
    "LEU",
    "LYS",
    "MET",
    "PHE",
    "PRO",
    "SER",
    "THR",
    "TRP",
    "TYR",
    "VAL",
]
AA_TO_INDEX = {aa: idx for idx, aa in enumerate(STANDARD_AAS)}


def _get_pdb_root():
    """Resolve the cluster PDB root lazily so imports work outside sbatch."""
    return Path(os.environ["PDB_ROOT"])


def find_pdb_file(protein_id):
    """Locate the predicted PDB file for a protein."""
    pdb_root = _get_pdb_root()
    pdb_path = (
        pdb_root
        / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions"
        / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    )
    return pdb_path if pdb_path.exists() else None


def get_pair_block_path(protein_dir, protein_id, layer):
    """Return the path to a pair block .npy file."""
    return protein_dir / f"{protein_id}_pair_block_{layer}.npy"


def discover_proteins(pair_blocks_root):
    """Find all protein directories under the pair blocks root."""
    proteins = []
    for entry in sorted(pair_blocks_root.iterdir()):
        if entry.is_dir():
            proteins.append((entry.name, entry))
    return proteins


def extract_ca_coordinates(pdb_path):
    """Extract CA atoms from all chains in the PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", str(pdb_path))

    residues = []
    coordinates = []
    for chain in structure[0]:
        for residue in chain:
            if residue.get_id()[0] == "W":
                continue
            if "CA" not in residue:
                continue
            residues.append((chain.get_id(), residue.get_resname(), residue.get_id()[1]))
            coordinates.append(residue["CA"].get_coord())

    return {
        "residues": residues,
        "coordinates": (
            np.array(coordinates, dtype=np.float32)
            if coordinates
            else np.empty((0, 3), dtype=np.float32)
        ),
    }


def compute_distance_matrix(coords):
    """Vectorized pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def build_sorted_aa_pair_indices(residues, pair_i, pair_j):
    """Build per-pair AA indices after alphabetical sorting of residue names."""
    residue_names = [record[1] for record in residues]
    n_pairs = len(pair_i)
    aa_first = np.full(n_pairs, -1, dtype=np.int16)
    aa_second = np.full(n_pairs, -1, dtype=np.int16)

    for idx in range(n_pairs):
        aa_i = residue_names[pair_i[idx]]
        aa_j = residue_names[pair_j[idx]]
        if aa_i <= aa_j:
            aa_left = aa_i
            aa_right = aa_j
        else:
            aa_left = aa_j
            aa_right = aa_i

        left_idx = AA_TO_INDEX.get(aa_left)
        right_idx = AA_TO_INDEX.get(aa_right)
        if left_idx is not None:
            aa_first[idx] = left_idx
        if right_idx is not None:
            aa_second[idx] = right_idx

    return aa_first, aa_second


def classical_mds(distance_matrix, n_dims=3):
    """Classical (Torgerson) MDS: recover coordinates from pairwise distances."""
    D_sq = distance_matrix ** 2

    row_mean = D_sq.mean(axis=1, keepdims=True)
    col_mean = D_sq.mean(axis=0, keepdims=True)
    grand_mean = D_sq.mean()
    B = -0.5 * (D_sq - row_mean - col_mean + grand_mean)

    eigenvalues, eigenvectors = np.linalg.eigh(B)

    idx = np.argsort(eigenvalues)[::-1]
    eigenvalues = eigenvalues[idx]
    eigenvectors = eigenvectors[:, idx]

    top_evals = np.maximum(eigenvalues[:n_dims], 0.0)
    coords = eigenvectors[:, :n_dims] * np.sqrt(top_evals)

    return coords, eigenvalues


def superimpose_structures(fixed_coords, moving_coords):
    """Kabsch algorithm for optimal rotation + translation."""
    fixed_center = fixed_coords.mean(axis=0)
    moving_center = moving_coords.mean(axis=0)
    fixed_centered = fixed_coords - fixed_center
    moving_centered = moving_coords - moving_center

    covariance = moving_centered.T @ fixed_centered
    U, _, Vt = np.linalg.svd(covariance)

    det_sign = np.linalg.det(Vt.T @ U.T)
    sign_matrix = np.diag([1.0, 1.0, np.sign(det_sign)])
    rotation = Vt.T @ sign_matrix @ U.T

    transformed = (moving_centered @ rotation.T) + fixed_center

    diff = fixed_coords - transformed
    rmsd = np.sqrt(np.mean(np.sum(diff ** 2, axis=1)))

    return transformed, rmsd


def compute_tm_score(fixed_coords, transformed_coords, length_target):
    """Compute TM-score between two superimposed structures."""
    d0 = 1.24 * (max(length_target, 16) - 15) ** (1.0 / 3.0) - 1.8
    d0 = max(d0, 0.5)

    distances = np.sqrt(np.sum((fixed_coords - transformed_coords) ** 2, axis=1))
    scores = 1.0 / (1.0 + (distances / d0) ** 2)
    return float(np.sum(scores) / length_target)


def compute_triangle_violations(pred_dist):
    """Compute triangle inequality violations using min-indirect-path approach."""
    n_residues = pred_dist.shape[0]
    min_indirect = np.full((n_residues, n_residues), np.inf)
    for j in range(n_residues):
        candidate = pred_dist[:, j:j + 1] + pred_dist[j:j + 1, :]
        min_indirect = np.minimum(min_indirect, candidate)
    np.fill_diagonal(min_indirect, 0.0)

    mask = pred_dist > min_indirect
    np.fill_diagonal(mask, False)
    n_violations = mask.sum()
    total_pairs = n_residues * (n_residues - 1)

    violation_fraction = float(n_violations / total_pairs) if total_pairs > 0 else 0.0

    if n_violations > 0:
        violations = pred_dist[mask] - min_indirect[mask]
        mean_violation = float(violations.mean())
        max_violation = float(violations.max())
    else:
        mean_violation = 0.0
        max_violation = 0.0

    return violation_fraction, mean_violation, max_violation


def analyze_protein_geometry(n_residues, true_coords, pb_i, pb_j, y_pred):
    """Run MDS reconstruction + triangle inequality analysis on one protein."""
    pred_dist = np.zeros((n_residues, n_residues))
    pred_dist[pb_i, pb_j] = y_pred
    pred_dist[pb_j, pb_i] = y_pred

    n_negative_preds = int(np.sum(y_pred < 0))
    pred_dist = np.maximum(pred_dist, 0.0)

    mds_coords, eigenvalues = classical_mds(pred_dist, n_dims=3)

    pos_mask = eigenvalues > 0
    sum_pos = np.sum(eigenvalues[pos_mask]) if np.any(pos_mask) else 1.0
    frac_top3 = float(np.sum(np.maximum(eigenvalues[:3], 0.0)) / sum_pos)
    n_neg_evals = int(np.sum(eigenvalues < 0))
    neg_frac = (
        float(abs(np.sum(eigenvalues[eigenvalues < 0])) / sum_pos)
        if n_neg_evals > 0 else 0.0
    )

    mds_reflected = mds_coords.copy()
    mds_reflected[:, 2] *= -1

    transformed_orig, rmsd_orig = superimpose_structures(true_coords, mds_coords)
    transformed_flip, rmsd_flip = superimpose_structures(true_coords, mds_reflected)

    if rmsd_flip < rmsd_orig:
        transformed, rmsd = transformed_flip, rmsd_flip
    else:
        transformed, rmsd = transformed_orig, rmsd_orig

    tm_score = compute_tm_score(true_coords, transformed, n_residues)

    mds_dist = compute_distance_matrix(transformed)
    triu_i, triu_j = np.triu_indices(n_residues, k=1)
    d_pred = pred_dist[triu_i, triu_j]
    d_mds = mds_dist[triu_i, triu_j]
    stress = float(np.sqrt(np.sum((d_pred - d_mds) ** 2) / np.sum(d_pred ** 2)))

    viol_frac, viol_mean, viol_max = compute_triangle_violations(pred_dist)

    return {
        "rmsd": rmsd,
        "tm_score": tm_score,
        "frac_top3_eigenvalues": frac_top3,
        "n_negative_eigenvalues": n_neg_evals,
        "neg_eigenvalue_fraction": neg_frac,
        "n_negative_predictions": n_negative_preds,
        "stress": stress,
        "triangle_violation_fraction": viol_frac,
        "triangle_mean_violation": viol_mean,
        "triangle_max_violation": viol_max,
    }


def make_failed_geometry_row(fold_idx, layer, protein_id, split, n_residues, per_protein_r2):
    """Return a placeholder per-protein row when geometry analysis fails."""
    return {
        "fold": fold_idx,
        "layer": layer,
        "protein_id": protein_id,
        "split": split,
        "n_residues": n_residues,
        "per_protein_r2": per_protein_r2,
        "rmsd": float("nan"),
        "tm_score": float("nan"),
        "frac_top3_eigenvalues": float("nan"),
        "n_negative_eigenvalues": -1,
        "neg_eigenvalue_fraction": float("nan"),
        "n_negative_predictions": -1,
        "stress": float("nan"),
        "triangle_violation_fraction": float("nan"),
        "triangle_mean_violation": float("nan"),
        "triangle_max_violation": float("nan"),
    }


__all__ = [
    "AA_TO_INDEX",
    "STANDARD_AAS",
    "analyze_protein_geometry",
    "build_sorted_aa_pair_indices",
    "classical_mds",
    "compute_distance_matrix",
    "compute_tm_score",
    "compute_triangle_violations",
    "discover_proteins",
    "extract_ca_coordinates",
    "find_pdb_file",
    "get_pair_block_path",
    "make_failed_geometry_row",
    "superimpose_structures",
]

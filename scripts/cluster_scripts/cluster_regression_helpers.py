#!/usr/bin/env python3
"""
Shared helper utilities for future cluster regression and diagnostic scripts.

This module centralizes only stable, duplicated helpers. Existing scripts are
not required to adopt it immediately.
"""

import os
from itertools import combinations_with_replacement
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
STANDARD_AA_PAIR_LABELS = tuple(
    f"{aa_left}-{aa_right}"
    for aa_left, aa_right in combinations_with_replacement(sorted(STANDARD_AAS), 2)
)
AA_PAIR_TO_CODE = {label: code for code, label in enumerate(STANDARD_AA_PAIR_LABELS)}
AA_INDEX_PAIR_TO_CODE = np.full((len(STANDARD_AAS), len(STANDARD_AAS)), -1, dtype=np.int16)
for _aa_i, _index_i in AA_TO_INDEX.items():
    for _aa_j, _index_j in AA_TO_INDEX.items():
        AA_INDEX_PAIR_TO_CODE[_index_i, _index_j] = AA_PAIR_TO_CODE[
            "-".join(sorted((_aa_i, _aa_j)))
        ]

TRIANGLE_VIOLATION_TOLERANCE = 1e-12
TRIANGLE_VIOLATION_KEYS = [
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]

CAMEO_PAIR_BLOCKS_ROOT = Path("/work/hdd/bekh/thayes/openfold_reps")
CAMEO_PDB_ROOT = Path("/work/hdd/bekh/thayes/openfold_reps_outputs_sm")
ECOD_PAIR_BLOCKS_ROOT = Path(
    "/work/hdd/bfna/thayes/ecod/experiments/representation_saves"
)
ECOD_PDB_ROOT = Path(
    "/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm"
)


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


def resolve_combined_pair_block_path(protein_id, source, layer):
    """Resolve one canonical combined-cohort pair tensor on Delta."""
    protein_id = str(protein_id)
    source = str(source)
    if source == "cameo":
        protein_dir = CAMEO_PAIR_BLOCKS_ROOT / protein_id
    elif source == "ecod":
        protein_dir = ECOD_PAIR_BLOCKS_ROOT / f"{protein_id}_save_layers_0_47"
    else:
        raise ValueError(f"Unknown combined-cohort source: {source!r}")
    candidates = (
        get_pair_block_path(protein_dir, protein_id, layer),
        protein_dir / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy",
    )
    return next((path for path in candidates if path.is_file()), candidates[0])


def resolve_combined_pdb_path(protein_id, source):
    """Resolve one canonical combined-cohort predicted PDB on Delta."""
    protein_id = str(protein_id)
    source = str(source)
    if source == "cameo":
        pdb_root = CAMEO_PDB_ROOT
        protein_dir = CAMEO_PAIR_BLOCKS_ROOT / protein_id
    elif source == "ecod":
        pdb_root = ECOD_PDB_ROOT
        protein_dir = ECOD_PAIR_BLOCKS_ROOT / f"{protein_id}_save_layers_0_47"
    else:
        raise ValueError(f"Unknown combined-cohort source: {source!r}")
    filename = f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    candidates = (
        protein_dir / "predicted_pdb" / filename,
        pdb_root
        / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions"
        / filename,
        pdb_root / protein_id / "predicted_pdb" / filename,
    )
    return next((path for path in candidates if path.is_file()), candidates[1])


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


def build_unordered_aa_pair_codes(residues, pair_i, pair_j):
    """Encode canonical unordered residue pairs with stable codes; invalid pairs are -1."""
    residue_codes = np.asarray(
        [AA_TO_INDEX.get(record[1], -1) for record in residues], dtype=np.int16
    )
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)
    if pair_i.shape != pair_j.shape:
        raise ValueError("pair_i and pair_j must have identical shapes")

    codes = np.full(pair_i.shape, -1, dtype=np.int16)
    left = residue_codes[pair_i]
    right = residue_codes[pair_j]
    valid = (left >= 0) & (right >= 0)
    codes[valid] = AA_INDEX_PAIR_TO_CODE[left[valid], right[valid]]
    return codes


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


def make_triangle_violation_metrics(fill_value=0.0):
    """Return a complete triangle-violation metric dictionary."""
    return {key: float(fill_value) for key in TRIANGLE_VIOLATION_KEYS}


def compute_triangle_violations(pred_dist):
    """Compute exact unordered-triangle inequality violation metrics."""
    n_residues = pred_dist.shape[0]
    total_triangles = n_residues * (n_residues - 1) * (n_residues - 2) // 6
    if total_triangles <= 0:
        return make_triangle_violation_metrics(0.0)

    n_violations = 0
    n_0_5 = 0
    n_5_10 = 0
    n_10_plus = 0
    violation_sum = 0.0
    max_violation = 0.0

    for i in range(n_residues - 2):
        local_j, local_k = np.triu_indices(n_residues - i - 1, k=1)
        j = local_j + i + 1
        k = local_k + i + 1

        side_ij = pred_dist[i, j]
        side_ik = pred_dist[i, k]
        side_jk = pred_dist[j, k]
        longest = np.maximum(np.maximum(side_ij, side_ik), side_jk)
        violations = longest - (side_ij + side_ik + side_jk - longest)

        positive = violations > TRIANGLE_VIOLATION_TOLERANCE
        if not np.any(positive):
            continue

        positive_violations = violations[positive]
        n_positive = int(positive_violations.size)
        n_violations += n_positive
        violation_sum += float(np.sum(positive_violations, dtype=np.float64))
        max_violation = max(max_violation, float(np.max(positive_violations)))

        n_0_5 += int(np.count_nonzero(positive_violations < 5.0))
        n_5_10 += int(
            np.count_nonzero(
                (positive_violations >= 5.0) & (positive_violations < 10.0)
            )
        )
        n_10_plus += int(np.count_nonzero(positive_violations >= 10.0))

    n_no_violation = total_triangles - n_violations
    return {
        "triangle_violation_fraction": float(n_violations / total_triangles),
        "triangle_mean_violation": float(violation_sum / total_triangles),
        "triangle_max_violation": float(max_violation),
        "triangle_no_violation_fraction": float(n_no_violation / total_triangles),
        "triangle_violation_0_5_fraction": float(n_0_5 / total_triangles),
        "triangle_violation_5_10_fraction": float(n_5_10 / total_triangles),
        "triangle_violation_10_plus_fraction": float(n_10_plus / total_triangles),
    }


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

    triangle_metrics = compute_triangle_violations(pred_dist)

    return {
        "rmsd": rmsd,
        "tm_score": tm_score,
        "frac_top3_eigenvalues": frac_top3,
        "n_negative_eigenvalues": n_neg_evals,
        "neg_eigenvalue_fraction": neg_frac,
        "n_negative_predictions": n_negative_preds,
        "stress": stress,
        **triangle_metrics,
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
        **make_triangle_violation_metrics(float("nan")),
    }


__all__ = [
    "AA_INDEX_PAIR_TO_CODE",
    "AA_PAIR_TO_CODE",
    "AA_TO_INDEX",
    "STANDARD_AA_PAIR_LABELS",
    "STANDARD_AAS",
    "analyze_protein_geometry",
    "build_sorted_aa_pair_indices",
    "build_unordered_aa_pair_codes",
    "classical_mds",
    "compute_distance_matrix",
    "compute_tm_score",
    "compute_triangle_violations",
    "discover_proteins",
    "extract_ca_coordinates",
    "find_pdb_file",
    "get_pair_block_path",
    "resolve_combined_pair_block_path",
    "resolve_combined_pdb_path",
    "make_failed_geometry_row",
    "make_triangle_violation_metrics",
    "superimpose_structures",
    "TRIANGLE_VIOLATION_KEYS",
    "TRIANGLE_VIOLATION_TOLERANCE",
]

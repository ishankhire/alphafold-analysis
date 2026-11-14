#!/usr/bin/env python3
"""Shared pieces of the SMACOF error decomposition: SMACOF, the four terms, and bin sums.

A decoded distance matrix is fitted by a 3D point set that minimises raw, unweighted stress
on distances in angstroms (metric SMACOF, one start from the classical-MDS coordinates).
With ``d_hat`` the decoded distances after clipping negatives to zero, ``p`` the distances
of the fitted points and ``d`` the distances of OpenFold's final structure, means over
unordered pairs give

    total_sq        = mean (d_hat - d)^2
    inconsistent_sq = mean (d_hat - p)^2
    consistent_sq   = mean (p - d)^2
    cross           = 2 mean (d_hat - p)(p - d)
    total_sq = inconsistent_sq + consistent_sq + cross      (exact)

The input is never rescaled or normalised.
"""
from __future__ import annotations

import numpy as np

EXPERIMENT_NAME = "smacof_error_decomposition_5fold_v1"
DEFAULT_REL_TOL = 1e-10
DEFAULT_MAX_ITER = 20_000
SNAPSHOT_TOLS = (1e-6, 1e-8, 1e-10, 1e-12)   # tolerances whose terms are recorded on the way
SNAPSHOT_LABELS = {1e-6: "1e-6", 1e-8: "1e-8", 1e-10: "1e-10", 1e-12: "1e-12"}
EXACT_FIT_FRACTION = 1e-24               # raw stress below this share of sum(d_hat^2) is an exact fit
IDENTITY_RTOL = 1e-9
STRESS_SLACK_RTOL = 1e-9                 # rounding slack when comparing SMACOF and classical raw stress

# Triangle-experiment model names -> the labels this experiment writes.
MATRIX_TYPES = {"observed": "observed", "null_a": "control_gaussian", "null_b": "control_shuffled"}
CONTROL_BINNING = "B10"
BIN_SYSTEMS = ("sequence_separation", "true_distance_b10")


def packed_to_matrix(values: np.ndarray, n_residues: int) -> np.ndarray:
    """Symmetric zero-diagonal matrix from a packed np.tril_indices(n, k=-1) vector."""
    pair_i, pair_j = np.tril_indices(n_residues, k=-1)
    values = np.asarray(values, dtype=np.float64)
    if values.shape != pair_i.shape:
        raise ValueError("Packed vector length does not match n_residues")
    matrix = np.zeros((n_residues, n_residues), dtype=np.float64)
    matrix[pair_i, pair_j] = matrix[pair_j, pair_i] = values
    return matrix


def packed_distances(coordinates: np.ndarray) -> np.ndarray:
    """Pairwise distances of a point set in np.tril_indices(n, k=-1) order."""
    coordinates = np.asarray(coordinates, dtype=np.float64)
    pair_i, pair_j = np.tril_indices(coordinates.shape[0], k=-1)
    return np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)


def _distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    """Fast distance matrix for the inner loop (Gram form; coordinates are kept centred)."""
    gram = coordinates @ coordinates.T
    squared = np.diag(gram)
    matrix = squared[:, None] + squared[None, :] - 2.0 * gram
    np.maximum(matrix, 0.0, out=matrix)
    np.sqrt(matrix, out=matrix)
    np.fill_diagonal(matrix, 0.0)
    return matrix


def raw_stress(d_hat: np.ndarray, p: np.ndarray) -> float:
    """Sum over unordered pairs of (d_hat - p)^2, in squared angstroms."""
    return float(np.sum((np.asarray(d_hat, np.float64) - np.asarray(p, np.float64)) ** 2))


def smacof_metric(dissimilarities: np.ndarray, start: np.ndarray, *, rel_tol: float = DEFAULT_REL_TOL,
                  max_iter: int = DEFAULT_MAX_ITER, snapshot_tols: tuple[float, ...] = SNAPSHOT_TOLS) -> dict:
    """Metric SMACOF on raw unweighted stress, from one given start.

    ``dissimilarities`` is a symmetric, zero-diagonal, nonnegative matrix in angstroms and is
    used as given. Each iteration is the Guttman transform ``X <- B(X) X / n`` with
    ``B_ij = -delta_ij / d_ij(X)`` (0 where the fitted distance is 0). The run has converged
    when one iteration lowers raw stress by at most ``rel_tol`` times its value, or when the
    fit is exact. ``snapshots`` holds the coordinates and iteration at which each looser
    tolerance in ``snapshot_tols`` was first met.
    """
    delta = np.asarray(dissimilarities, dtype=np.float64)
    n = delta.shape[0]
    if delta.shape != (n, n) or not np.all(np.isfinite(delta)) or np.any(delta < 0.0):
        raise ValueError("Dissimilarities must be a finite, nonnegative square matrix")
    if not np.array_equal(delta, delta.T) or np.any(np.diag(delta) != 0.0):
        raise ValueError("Dissimilarities must be symmetric with a zero diagonal")
    coordinates = np.asarray(start, dtype=np.float64)
    if coordinates.shape != (n, 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("The start must be a finite (n, 3) array")
    coordinates = coordinates - coordinates.mean(axis=0)
    exact = EXACT_FIT_FRACTION * 0.5 * float(np.sum(delta * delta))

    distances = _distance_matrix(coordinates)
    stress = 0.5 * float(np.sum((delta - distances) ** 2))
    start_stress = stress
    pending = sorted((t for t in snapshot_tols if t >= rel_tol), reverse=True)   # loosest first
    snapshots: dict[float, tuple[int, np.ndarray]] = {}
    iterations, converged, relative = 0, stress <= exact, float("nan")
    if converged:
        relative = 0.0
    ratio = np.empty_like(delta)
    while not converged and iterations < max_iter:
        iterations += 1
        ratio.fill(0.0)
        np.divide(delta, distances, out=ratio, where=distances > 0.0)
        coordinates = (ratio.sum(axis=1)[:, None] * coordinates - ratio @ coordinates) / n
        distances = _distance_matrix(coordinates)
        new_stress = 0.5 * float(np.sum((delta - distances) ** 2))
        relative = (stress - new_stress) / stress if stress > 0.0 else 0.0
        stress = new_stress
        while pending and relative <= pending[0]:
            snapshots[pending.pop(0)] = (iterations, coordinates.copy())
        converged = relative <= rel_tol or stress <= exact
    if converged:
        for tolerance in pending:                       # an exact fit meets every tolerance
            snapshots[tolerance] = (iterations, coordinates.copy())
    return {"coordinates": coordinates, "iterations": iterations, "converged": bool(converged),
            "final_relative_decrease": float(relative), "start_raw_stress": start_stress,
            "raw_stress": stress, "snapshots": snapshots}


def decompose(d_hat: np.ndarray, p: np.ndarray, d: np.ndarray) -> dict[str, float]:
    """The four terms (means over unordered pairs), the identity residual, and the scale ratio."""
    d_hat, p, d = (np.asarray(v, dtype=np.float64) for v in (d_hat, p, d))
    if not d_hat.shape == p.shape == d.shape or d_hat.ndim != 1 or d_hat.size == 0:
        raise ValueError("d_hat, p and d must be packed vectors of one length")
    outer, inner = d_hat - p, p - d
    total = float(np.mean((d_hat - d) ** 2))
    inconsistent = float(np.mean(outer * outer))
    consistent = float(np.mean(inner * inner))
    cross = 2.0 * float(np.mean(outer * inner))
    p_sq = float(np.sum(p * p))
    d_hat_sq = float(np.sum(d_hat * d_hat))
    return {
        "total_sq": total, "inconsistent_sq": inconsistent, "consistent_sq": consistent, "cross": cross,
        "identity_residual": total - (inconsistent + consistent + cross),
        "raw_stress": float(np.sum(outer * outer)),
        "normalized_stress": float(np.sqrt(np.sum(outer * outer) / d_hat_sq)) if d_hat_sq > 0.0 else float("nan"),
        "scale_ratio": float(np.sum(d_hat * p)) / p_sq if p_sq > 0.0 else float("nan"),
    }


def identity_holds(terms: dict[str, float]) -> bool:
    return abs(terms["identity_residual"]) <= IDENTITY_RTOL * max(terms["total_sq"], 1.0)


def bin_sums(d_hat: np.ndarray, p: np.ndarray, d: np.ndarray, codes: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    """Per-bin pair count and sums of (d_hat-d)^2, (d_hat-p)^2 and (p-d)^2."""
    d_hat, p, d = (np.asarray(v, dtype=np.float64) for v in (d_hat, p, d))
    codes = np.asarray(codes, dtype=np.int64)
    if np.any(codes < 0) or np.any(codes >= n_bins):
        raise ValueError("Every pair needs a bin in range")
    return {
        "n_pairs": np.bincount(codes, minlength=n_bins).astype(np.int64),
        "sum_total_sq": np.bincount(codes, weights=(d_hat - d) ** 2, minlength=n_bins),
        "sum_inconsistent_sq": np.bincount(codes, weights=(d_hat - p) ** 2, minlength=n_bins),
        "sum_consistent_sq": np.bincount(codes, weights=(p - d) ** 2, minlength=n_bins),
    }

#!/usr/bin/env python3
"""Shared pieces of the null-triangle experiment: bins, streaming scoring, and nulls.

Each null adds errors to the final-structure distances (d_true). The errors match the
probe's residuals bin by bin but are rearranged among pairs. The nulls and the observed
probe output are scored on every triangle with the paper's all-triangles metrics
(violation fraction, normalized mean violation).

The triangle metric follows ``triangle_violation_subgroup_common.compute_metric_rows``
operation by operation. It reuses that module's triangle order and tolerance, and keeps
only the all-triangles reduction. ``tests/test_null_triangles.py`` and the replay's
in-job gate assert bit-identical results against ``compute_metric_rows``. Every vector of
a protein is scored in one pass over the triangles, batched as ``iter_triangle_indices``
batches them, so each vector's summation order matches ``compute_metric_rows``.
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent / "triangle_violation_subgroups"):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from triangle_violation_subgroup_common import (  # noqa: E402
    DEFAULT_TRIANGLE_BATCH_SIZE,
    TRIANGLE_TOLERANCE_ANGSTROM,
    iter_triangle_indices,
    triangle_count,
)

EXPERIMENT_NAME = "null_triangles_5fold_v1"
SEED = 20261109
PROBES = (
    "shared_ols_base256",
    "per_protein_ols_base256",
    "sequence_gated_ols_base256",
    "shared_xgboost_base256_fixed_s4",
)
SENSITIVITY_PROBE = "shared_ols_base256"
N_DRAWS = 3
NULL_A_MIN_PAIRS = 30
BITWISE_MAX_TRIANGLES = 1_000_000   # proteins this small are also checked against compute_metric_rows
LAYERS = tuple(range(48))
EXPECTED_COHORT = {"cameo": 154, "ecod": 398}

# Distance bins on d_true, half-open [lo, hi) in angstroms.
B10_EDGES = np.array([0, 4, 6, 8, 10, 14, 20, 30, 40, 55, np.inf], dtype=np.float64)
B5_EDGES = np.array([0, 5, 10, 20, 40, np.inf], dtype=np.float64)
B20_EDGES = np.array(
    [0, 2, 4, 5, 6, 7, 8, 9, 10, 12, 14, 17, 20, 25, 30, 35, 40, 47.5, 55, 70, np.inf],
    dtype=np.float64,
)
SEQ_LABELS = ("1", "2-4", "5-9", "10-19", "20-49", "50+")
BINNINGS = ("B10", "B10xS6", "B5", "B20")
BINNING_CODES = {name: code for code, name in enumerate(BINNINGS)}
MODELS = ("observed", "null_a", "null_b")
MODEL_CODES = {name: code for code, name in enumerate(MODELS)}


def edge_labels(edges: np.ndarray) -> tuple[str, ...]:
    def fmt(value: float) -> str:
        return "inf" if not math.isfinite(value) else f"{value:g}"
    return tuple(f"[{fmt(lo)},{fmt(hi)})" for lo, hi in zip(edges[:-1], edges[1:]))


B10_LABELS = edge_labels(B10_EDGES)
BIN_LABELS = {
    "B10": B10_LABELS,
    "B5": edge_labels(B5_EDGES),
    "B20": edge_labels(B20_EDGES),
    "B10xS6": tuple(f"{d}|{s}" for d in B10_LABELS for s in SEQ_LABELS),
}

# Which (model, binning) runs are made, per probe. Observed is scored separately.
NULL_RUNS = (("null_a", "B10"), ("null_b", "B10"))
SENSITIVITY_RUNS = (("null_b", "B5"), ("null_b", "B20"), ("null_b", "B10xS6"))


def null_runs(probe: str) -> tuple[tuple[str, str], ...]:
    return NULL_RUNS + (SENSITIVITY_RUNS if probe == SENSITIVITY_PROBE else ())


# ---------------------------------------------------------------- bins

def distance_codes(distances: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """Half-open [lo, hi) bin index for each nonnegative distance."""
    distances = np.asarray(distances, dtype=np.float64)
    if np.any(distances < 0) or not np.all(np.isfinite(distances)):
        raise ValueError("Reference distances must be finite and nonnegative")
    return (np.searchsorted(edges, distances, side="right") - 1).astype(np.int64)


def all_bin_codes(d_true: np.ndarray, seq_codes: np.ndarray) -> dict[str, np.ndarray]:
    seq_codes = np.asarray(seq_codes, dtype=np.int64)
    if np.any(seq_codes < 0) or np.any(seq_codes >= len(SEQ_LABELS)):
        raise ValueError("Every pair needs a defined sequence-separation bin")
    b10 = distance_codes(d_true, B10_EDGES)
    return {
        "B10": b10,
        "B10xS6": b10 * len(SEQ_LABELS) + seq_codes,
        "B5": distance_codes(d_true, B5_EDGES),
        "B20": distance_codes(d_true, B20_EDGES),
    }


def bin_moments(residual: np.ndarray, codes: np.ndarray, n_bins: int) -> dict[str, np.ndarray]:
    """Per-bin count, mean, population SD, RMSE, skew, and excess kurtosis."""
    r = np.asarray(residual, dtype=np.float64)
    n = np.bincount(codes, minlength=n_bins).astype(np.float64)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = np.bincount(codes, weights=r, minlength=n_bins) / n
        centred = r - np.nan_to_num(mean)[codes]
        m2 = np.bincount(codes, weights=centred ** 2, minlength=n_bins) / n
        m3 = np.bincount(codes, weights=centred ** 3, minlength=n_bins) / n
        m4 = np.bincount(codes, weights=centred ** 4, minlength=n_bins) / n
        mse = np.bincount(codes, weights=r * r, minlength=n_bins) / n
        sd = np.sqrt(m2)
        skew = np.where(m2 > 0, m3 / m2 ** 1.5, np.nan)
        kurt = np.where(m2 > 0, m4 / m2 ** 2 - 3.0, np.nan)
    return {"n": n, "mean": mean, "sd": sd, "rmse": np.sqrt(mse), "skew": skew, "excess_kurtosis": kurt}


# ---------------------------------------------------------------- triangles

def pair_position(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Index of pair (high > low) in np.tril_indices(n, k=-1) order."""
    return high * (high - 1) // 2 + low


def side_positions(tri_i: np.ndarray, tri_j: np.ndarray, tri_k: np.ndarray) -> np.ndarray:
    """Packed positions of sides (i,j), (i,k), (j,k), the order compute_metric_rows gathers."""
    return np.stack((pair_position(tri_i, tri_j), pair_position(tri_i, tri_k),
                     pair_position(tri_j, tri_k)), axis=1)


def reference_rms(d_true: np.ndarray, positions: np.ndarray) -> np.ndarray:
    """RMS side length in the reference structure, as triangle_geometry computes it."""
    sides = d_true[positions]
    sides.sort(axis=1)
    a, b, c = sides.T
    if np.any(a <= 0.0):
        raise ValueError("Reference triangles must have three positive side lengths")
    return np.sqrt((a * a + b * b + c * c) / 3.0)


def score_batch(clipped: np.ndarray, positions: np.ndarray, rms: np.ndarray,
                negative: np.ndarray | None = None) -> tuple[int, int, float, float, int]:
    """One batch of the all-triangles reduction in compute_metric_rows, step for step."""
    sides = clipped[positions]
    sides.sort(axis=1)
    raw_delta = sides[:, 2] - sides[:, 0] - sides[:, 1]
    delta = np.where(raw_delta > TRIANGLE_TOLERANCE_ANGSTROM, raw_delta, 0.0)
    relative = delta / rms
    violation = delta > 0.0
    d, r = delta[violation], relative[violation]
    n_clipped = 0
    if negative is not None and d.size:
        n_clipped = int(np.count_nonzero(negative[positions[violation]].any(axis=1)))
    return (int(positions.shape[0]), int(d.size), float(np.sum(d, dtype=np.float64)),
            float(np.sum(r, dtype=np.float64)), n_clipped)


class Totals:
    __slots__ = ("n", "nv", "abs_sum", "rel_sum", "n_clipped")

    def __init__(self) -> None:
        self.n = self.nv = self.n_clipped = 0
        self.abs_sum = self.rel_sum = 0.0

    def add(self, part: tuple[int, int, float, float, int]) -> None:
        n, nv, a, r, c = part
        self.n += n
        self.nv += nv
        self.abs_sum += a
        self.rel_sum += r
        self.n_clipped += c

    def metrics(self) -> dict[str, float]:
        return {
            "n_triangles": self.n, "n_violations": self.nv,
            "violation_sum_angstrom": self.abs_sum, "relative_violation_sum": self.rel_sum,
            "violation_fraction": self.nv / self.n if self.n else float("nan"),
            "mean_relative_violation_all": self.rel_sum / self.n if self.n else float("nan"),
            "n_violations_with_clipped_side": self.n_clipped,
        }


def score_streaming(vectors: dict, d_true: np.ndarray, n_residues: int,
                    track_clipped: frozenset | set = frozenset()) -> dict:
    """Every triple for many packed distance vectors at once, without holding the triangle list.

    Each vector is clipped at zero once; triangle indices and reference RMS are made once
    per batch and shared. ``n_violations_with_clipped_side`` is counted for keys in
    ``track_clipped`` and left at 0 for the rest.
    """
    clipped = {k: np.maximum(np.asarray(v, dtype=np.float64), 0.0) for k, v in vectors.items()}
    negative = {k: np.asarray(vectors[k]) < 0.0 for k in vectors if k in track_clipped}
    totals = {k: Totals() for k in vectors}
    for tri_i, tri_j, tri_k in iter_triangle_indices(n_residues, DEFAULT_TRIANGLE_BATCH_SIZE):
        positions = side_positions(tri_i, tri_j, tri_k)
        rms = reference_rms(d_true, positions)
        for key in vectors:
            totals[key].add(score_batch(clipped[key], positions, rms, negative.get(key)))
    return {k: t.metrics() for k, t in totals.items()}


# ---------------------------------------------------------------- nulls

def within_bin_permutation(rng: np.random.Generator, codes: np.ndarray) -> np.ndarray:
    """A permutation that only moves values among pairs in the same bin."""
    shuffled = np.lexsort((rng.random(codes.size), codes))
    positions = np.argsort(codes, kind="stable")
    permutation = np.empty(codes.size, np.int64)
    permutation[positions] = shuffled
    return permutation


def same_bin_histograms(original: np.ndarray, permuted: np.ndarray, codes: np.ndarray) -> bool:
    a = original[np.lexsort((original, codes))]
    b = permuted[np.lexsort((permuted, codes))]
    return bool(np.array_equal(a, b))


def null_draw_rng(layer: int, probe: str, protein_index: int, model: str, binning: str, draw: int) -> np.random.Generator:
    return np.random.default_rng([SEED, 1, layer, PROBES.index(probe), protein_index,
                                  MODEL_CODES[model], BINNING_CODES[binning], draw])


def make_null(model: str, rng: np.random.Generator, d_true: np.ndarray, residual: np.ndarray,
              codes: np.ndarray, mu: np.ndarray, sd: np.ndarray) -> tuple[np.ndarray, bool | None]:
    """One null distance vector (unclipped) and, for Null B, the histogram check."""
    if model == "null_a":
        return d_true + mu[codes] + sd[codes] * rng.standard_normal(d_true.size), None
    if model == "null_b":
        permuted = residual[within_bin_permutation(rng, codes)]
        return d_true + permuted, same_bin_histograms(residual, permuted, codes)
    raise ValueError(f"Unknown null model {model!r}")


def calibration_fit(d_true: np.ndarray, d_pred: np.ndarray) -> tuple[float, float]:
    """OLS of d_pred on d_true within one protein: (alpha, beta)."""
    x, y = np.asarray(d_true, np.float64), np.asarray(d_pred, np.float64)
    xc = x - x.mean()
    beta = float(np.dot(xc, y - y.mean()) / np.dot(xc, xc))
    return float(y.mean() - beta * x.mean()), beta

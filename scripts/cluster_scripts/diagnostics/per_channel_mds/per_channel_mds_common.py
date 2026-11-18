"""Per-channel classical MDS of the pair representation: the maths, no I/O.

Each of the 128 channels of a pair block is treated as a dissimilarity matrix. The channel is
symmetrised, oriented by a sign, shifted so its smallest off-diagonal value is zero, and put
through the same classical MDS and the same geometry formulas the probe families use
(``cluster_regression_helpers.classical_mds`` and the ``geometry_metrics.csv`` columns).
See ``README.md`` for the design.
"""
from __future__ import annotations

import numpy as np
from scipy.stats import rankdata

EXPERIMENT_NAME = "per_channel_mds_cameo_v1"
SEED = 20261118
N_DIMS = 3
CHANNEL_CHUNK = 8                   # channels per batched eigendecomposition (bounds worker memory)
SIGNS = (1, -1)
GEOMETRY = ("stress", "frac_top1_eigenvalue", "frac_top3_eigenvalues", "n_negative_eigenvalues",
            "negative_eigenvalue_fraction")
PERMUTED = ("stress", "frac_top3_eigenvalues", "n_negative_eigenvalues", "negative_eigenvalue_fraction")
CHANNEL_COLUMNS = (
    "status", "best_sign", "stress", "stress_other_sign", "frac_top1_eigenvalue", "frac_top3_eigenvalues",
    "n_negative_eigenvalues", "negative_eigenvalue_fraction",
    "perm_best_sign", *(f"perm_{name}" for name in PERMUTED),
    "spearman_sequence_separation", "spearman_true_distance", "embedding_true_distance_pearson",
    "antisymmetric_fraction", "offdiag_mean", "offdiag_std",
)
REFERENCES = ("true_distance", "sequence_separation")
SHIFTS = ("raw", "minshift")


def batched_classical_mds(distances: np.ndarray, n_dims: int = N_DIMS) -> tuple[np.ndarray, np.ndarray]:
    """``helpers.classical_mds`` over a stack of matrices ``(C, R, R)``.

    Returns coordinates ``(C, R, n_dims)`` and all eigenvalues ``(C, R)`` in descending order.
    """
    d_sq = distances ** 2
    row_mean = d_sq.mean(axis=2, keepdims=True)
    col_mean = d_sq.mean(axis=1, keepdims=True)
    grand_mean = d_sq.mean(axis=(1, 2), keepdims=True)
    gram = -0.5 * (d_sq - row_mean - col_mean + grand_mean)
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    eigenvalues = eigenvalues[:, ::-1]
    eigenvectors = eigenvectors[:, :, ::-1]
    top = np.maximum(eigenvalues[:, :n_dims], 0.0)
    coordinates = eigenvectors[:, :, :n_dims] * np.sqrt(top)[:, None, :]
    return coordinates, eigenvalues


def mds_metrics(distances: np.ndarray, upper: tuple[np.ndarray, np.ndarray]) -> tuple[dict, np.ndarray]:
    """Geometry metrics of each matrix in ``(C, R, R)`` and its embedded upper-triangle distances ``(C, P)``.

    The formulas are those of ``diagnostic_shared_ols_mds_pdb_export.reconstruct_mds_coordinates``.
    """
    coordinates, eigenvalues = batched_classical_mds(distances)
    positive_mass = np.where(eigenvalues > 0.0, eigenvalues, 0.0).sum(axis=1)
    positive_mass = np.where(positive_mass > 0.0, positive_mass, 1.0)
    negative = eigenvalues < 0.0
    top = np.maximum(eigenvalues[:, :N_DIMS], 0.0)
    difference = coordinates[:, upper[0], :] - coordinates[:, upper[1], :]
    embedded = np.sqrt((difference ** 2).sum(axis=2))
    source = distances[:, upper[0], upper[1]]
    denominator = (source ** 2).sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        stress = np.sqrt(((source - embedded) ** 2).sum(axis=1) / denominator)
    metrics = {
        "stress": np.where(denominator > 0.0, stress, np.nan),
        "frac_top1_eigenvalue": top[:, 0] / positive_mass,
        "frac_top3_eigenvalues": top.sum(axis=1) / positive_mass,
        "n_negative_eigenvalues": negative.sum(axis=1),
        "negative_eigenvalue_fraction": np.abs(np.where(negative, eigenvalues, 0.0).sum(axis=1)) / positive_mass,
    }
    return metrics, embedded


def to_matrices(values: np.ndarray, upper: tuple[np.ndarray, np.ndarray], n_residues: int) -> np.ndarray:
    """Symmetric zero-diagonal matrices ``(C, R, R)`` from upper-triangle values ``(C, P)``."""
    matrices = np.zeros((values.shape[0], n_residues, n_residues), dtype=np.float64)
    matrices[:, upper[0], upper[1]] = values
    matrices[:, upper[1], upper[0]] = values
    return matrices


def min_shift(values: np.ndarray, sign: int) -> np.ndarray:
    """Orient by ``sign`` and subtract the smallest value, per row of ``(C, P)``."""
    oriented = sign * values
    return oriented - oriented.min(axis=1, keepdims=True)


def best_sign_metrics(values: np.ndarray, upper: tuple[np.ndarray, np.ndarray],
                      n_residues: int) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    """MDS under both signs; per row keep the lower-stress one (ties and NaN go to +1).

    Returns the kept metrics, the kept sign, the other sign's stress and the kept embedded distances.
    """
    results = [mds_metrics(to_matrices(min_shift(values, sign), upper, n_residues), upper) for sign in SIGNS]
    (plus, plus_embedded), (minus, minus_embedded) = results
    use_minus = minus["stress"] < plus["stress"]
    kept = {name: np.where(use_minus, minus[name], plus[name]) for name in GEOMETRY}
    other_stress = np.where(use_minus, plus["stress"], minus["stress"])
    embedded = np.where(use_minus[:, None], minus_embedded, plus_embedded)
    return kept, np.where(use_minus, -1, 1), other_stress, embedded


def centred_unit(values: np.ndarray) -> np.ndarray:
    """Rows of ``(C, P)`` centred and scaled to unit norm; a constant row becomes NaN."""
    centred = values - values.mean(axis=1, keepdims=True)
    norm = np.sqrt((centred ** 2).sum(axis=1, keepdims=True))
    with np.errstate(invalid="ignore", divide="ignore"):
        return centred / np.where(norm > 0.0, norm, np.nan)


def permute_rows(values: np.ndarray, layer: int, protein_index: int, first_channel: int) -> np.ndarray:
    """Each channel's upper-triangle values in a random order, seeded per layer, protein and channel."""
    permuted = np.empty_like(values)
    for offset in range(values.shape[0]):
        rng = np.random.default_rng([SEED, layer, protein_index, first_channel + offset])
        permuted[offset] = rng.permutation(values[offset])
    return permuted


def channel_metrics(block: np.ndarray, true_distances: np.ndarray, layer: int, protein_index: int,
                    chunk: int = CHANNEL_CHUNK) -> dict[str, np.ndarray]:
    """All per-channel columns for one pair block ``(R, R, C)``; each value has shape ``(C,)``."""
    n_residues, n_channels = block.shape[0], block.shape[2]
    upper = np.triu_indices(n_residues, k=1)
    separation_unit = centred_unit(rankdata((upper[1] - upper[0]).astype(np.float64))[None, :])
    true_upper = np.asarray(true_distances, dtype=np.float64)[upper][None, :]
    true_rank_unit = centred_unit(rankdata(true_upper, axis=1))
    true_unit = centred_unit(true_upper)
    columns: dict[str, list] = {name: [] for name in CHANNEL_COLUMNS}
    for start in range(0, n_channels, chunk):
        stop = min(start + chunk, n_channels)
        forward = block[upper[0], upper[1], start:stop].astype(np.float64).T
        backward = block[upper[1], upper[0], start:stop].astype(np.float64).T
        symmetric, antisymmetric = (forward + backward) / 2.0, (forward - backward) / 2.0
        del forward, backward
        mean, std = symmetric.mean(axis=1), symmetric.std(axis=1)
        constant = np.ptp(symmetric, axis=1) == 0.0
        antisymmetric_sq = (antisymmetric ** 2).sum(axis=1)
        total = antisymmetric_sq + ((symmetric - mean[:, None]) ** 2).sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            antisymmetric_fraction = np.where(total > 0.0, antisymmetric_sq / total, np.nan)
        rank_unit = centred_unit(rankdata(symmetric, axis=1))
        kept, sign, other_stress, embedded = best_sign_metrics(symmetric, upper, n_residues)
        embedding_pearson = (centred_unit(embedded) * true_unit).sum(axis=1)
        permuted = permute_rows(symmetric, layer, protein_index, start)
        perm_kept, perm_sign, _, _ = best_sign_metrics(permuted, upper, n_residues)

        def blank(array: np.ndarray) -> np.ndarray:
            return np.where(constant, np.nan, array)

        columns["status"].append(np.where(constant, "constant", "ok"))
        columns["best_sign"].append(sign)
        columns["stress_other_sign"].append(blank(other_stress))
        for name in GEOMETRY:
            columns[name].append(blank(kept[name]))
        columns["perm_best_sign"].append(perm_sign)
        for name in PERMUTED:
            columns[f"perm_{name}"].append(blank(perm_kept[name]))
        columns["spearman_sequence_separation"].append((rank_unit * separation_unit).sum(axis=1))
        columns["spearman_true_distance"].append((rank_unit * true_rank_unit).sum(axis=1))
        columns["embedding_true_distance_pearson"].append(blank(embedding_pearson))
        columns["antisymmetric_fraction"].append(antisymmetric_fraction)
        columns["offdiag_mean"].append(mean)
        columns["offdiag_std"].append(std)
    return {name: np.concatenate(parts) for name, parts in columns.items()}


def single_matrix_metrics(distance_matrix: np.ndarray, classical_mds) -> dict[str, float]:
    """The same metrics for one matrix through ``helpers.classical_mds``; the parity reference."""
    coordinates, eigenvalues = classical_mds(distance_matrix, n_dims=N_DIMS)
    positive_mass = float(eigenvalues[eigenvalues > 0.0].sum()) or 1.0
    negative = eigenvalues < 0.0
    upper = np.triu_indices(distance_matrix.shape[0], k=1)
    embedded = np.linalg.norm(coordinates[upper[0]] - coordinates[upper[1]], axis=1)
    source = distance_matrix[upper]
    denominator = float((source ** 2).sum())
    return {
        "stress": float(np.sqrt(((source - embedded) ** 2).sum() / denominator)) if denominator > 0 else float("nan"),
        "frac_top1_eigenvalue": float(max(eigenvalues[0], 0.0) / positive_mass),
        "frac_top3_eigenvalues": float(np.maximum(eigenvalues[:N_DIMS], 0.0).sum() / positive_mass),
        "n_negative_eigenvalues": int(negative.sum()),
        "negative_eigenvalue_fraction": float(abs(eigenvalues[negative].sum()) / positive_mass),
    }


def channel_distance_matrix(block: np.ndarray, channel: int, sign: int) -> np.ndarray:
    """One channel's dissimilarity matrix: symmetrised, oriented, min-shifted, zero diagonal."""
    n_residues = block.shape[0]
    upper = np.triu_indices(n_residues, k=1)
    plane = block[:, :, channel].astype(np.float64)
    values = ((plane + plane.T) / 2.0)[upper][None, :]
    return to_matrices(min_shift(values, sign), upper, n_residues)[0]


def reference_rows(true_distances: np.ndarray) -> list[dict]:
    """The final-structure distances and ``|i - j|`` through the same MDS, unshifted and min-shifted."""
    n_residues = true_distances.shape[0]
    upper = np.triu_indices(n_residues, k=1)
    sources = {"true_distance": np.asarray(true_distances, dtype=np.float64)[upper],
               "sequence_separation": (upper[1] - upper[0]).astype(np.float64)}
    rows = []
    for reference in REFERENCES:
        for shift in SHIFTS:
            values = sources[reference][None, :]
            if shift == "minshift":
                values = min_shift(values, 1)
            metrics, embedded = mds_metrics(to_matrices(values, upper, n_residues), upper)
            row = {"reference": reference, "shift": shift}
            row.update({name: metrics[name][0] for name in GEOMETRY})
            row["embedding_true_distance_pearson"] = float(
                (centred_unit(embedded) * centred_unit(sources["true_distance"][None, :])).sum())
            rows.append(row)
    return rows

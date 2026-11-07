#!/usr/bin/env python3
"""Can the native error strata reproduce triangle violations? One layer, two probes, CAMEO.

A triangle violates when its longest side exceeds the sum of the other two, which depends on how
errors co-occur across three edges. The strata describe one edge at a time. Each model below adds
error to the final structure distances and keeps one more property of the real error. Models are
scored on the same fixed triangles per protein, across layers and probes.

No model is run: this reads stored predictions, stored strata, and the CAMEO final structures.
"""
from __future__ import annotations

import math
import os
import resource
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, CLUSTER_DIR, SCRIPT_DIR.parent / "native_error_strata"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from followup_paths import PRODUCER, PROTEIN_STRATA, TRIANGLE_ROOT  # noqa: E402

import cluster_regression_helpers as helpers  # noqa: E402
from regression_output_helpers import atomic_savez_compressed  # noqa: E402
from regression_protocol import strict_lower_triangle_indices  # noqa: E402
from strata_bins import DIST_EDGES, N_D, N_S, SEQ_EDGES  # noqa: E402

ROOT = TRIANGLE_ROOT
EXPORT = PRODUCER / "predicted_distances_5fold_v1"
STRATA = PROTEIN_STRATA
PROBES = tuple(STRATA)
MODELS = ("observed", "iid_matched", "pooled_mean_only", "pooled_strata", "protein_strata",
          "residuals_permuted", "residues_plus_permuted")
STOCHASTIC = {"iid_matched", "pooled_strata", "protein_strata", "residuals_permuted", "residues_plus_permuted"}
OUTLIER, N_TRIANGLES, N_REPS, N_POSITION = "7stz_C", 1_000_000, 2, 20
TOLERANCE = 1e-12                       # the geometry helper's violation tolerance
SLACK_EDGES = np.array([0.0, 0.02, 0.1, 0.3, np.inf])
N_SLACK = SLACK_EDGES.size - 1
STARTED = time.perf_counter()


def log(message: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[{time.perf_counter() - STARTED:7.1f}s] [peak_rss={rss:.0f}MiB] {message}", flush=True)


def sample_triangles(n: int, seed: int) -> tuple[np.ndarray, bool]:
    """Sorted residue triples: every triple if few enough, else a fixed uniform sample."""
    if math.comb(n, 3) <= N_TRIANGLES:
        parts = []
        for top in range(2, n):
            mid, low = np.tril_indices(top, k=-1)
            parts.append(np.stack((low, mid, np.full(low.size, top)), axis=1))
        return np.concatenate(parts).astype(np.int64), True
    draws = np.sort(np.random.default_rng(seed).integers(0, n, (int(N_TRIANGLES * 1.25), 3)), axis=1)
    distinct = draws[(draws[:, 0] < draws[:, 1]) & (draws[:, 1] < draws[:, 2])]
    if distinct.shape[0] < N_TRIANGLES:
        raise ValueError("Too few distinct triples were drawn")
    return distinct[:N_TRIANGLES].astype(np.int64), False


def pair_position(high: np.ndarray, low: np.ndarray) -> np.ndarray:
    """Index in strict-lower-triangle order: (1,0), (2,0), (2,1), (3,0), ..."""
    return high * (high - 1) // 2 + low


def score(values: np.ndarray, sides: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Violating triangles and their sizes, with negative distances clipped as the geometry code does."""
    lengths = np.maximum(values, 0.0)[sides]
    excess = 2.0 * lengths.max(axis=1) - lengths.sum(axis=1)
    violating = excess > TOLERANCE
    return violating, excess[violating]


def within_cell_permutation(rng: np.random.Generator, cell: np.ndarray) -> np.ndarray:
    shuffled = np.lexsort((rng.random(cell.size), cell))
    positions = np.argsort(cell, kind="stable")
    permutation = np.empty(cell.size, np.int64)
    permutation[positions] = shuffled
    return permutation


def load_strata(probe: str, layer: int) -> dict[str, object]:
    with np.load(STRATA[probe]) as data:
        ids = data["protein_ids"].astype(str)
        offsets = data["offsets"]
        cell_index = data["cell_index"].astype(np.int64)
        cell_n = data["cell_n"].astype(np.float64)
        sums = data["sums"][:, layer].astype(np.float64)
    size = N_S * N_D
    keep = np.repeat(ids != OUTLIER, np.diff(offsets))
    pool_n = np.bincount(cell_index[keep], weights=cell_n[keep], minlength=size)
    pool_error = np.bincount(cell_index[keep], weights=sums[keep, 0], minlength=size)
    pool_square = np.bincount(cell_index[keep], weights=sums[keep, 2], minlength=size)
    with np.errstate(invalid="ignore", divide="ignore"):
        pool_mean = np.nan_to_num(pool_error / pool_n)
        pool_sd = np.sqrt(np.maximum(np.nan_to_num(pool_square / pool_n) - pool_mean ** 2, 0.0))
    return {"index": {p: k for k, p in enumerate(ids)}, "offsets": offsets, "cell_index": cell_index,
            "cell_n": cell_n, "sums": sums, "pool_mean": pool_mean, "pool_sd": pool_sd}


def protein_strata(strata: dict[str, object], protein: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    k = strata["index"][protein]
    start, stop = strata["offsets"][k], strata["offsets"][k + 1]
    cells, n, sums = strata["cell_index"][start:stop], strata["cell_n"][start:stop], strata["sums"][start:stop]
    mean, sd = np.zeros(N_S * N_D), np.zeros(N_S * N_D)
    mean[cells] = sums[:, 0] / n
    sd[cells] = np.sqrt(np.maximum(sums[:, 2] / n - mean[cells] ** 2, 0.0))
    return mean, sd, cells, n


def main() -> None:
    layer = int(os.environ["SLURM_ARRAY_TASK_ID"])
    final = ROOT / "layer_shards" / f"layer_{layer:02d}.npz"
    if final.is_file():
        log(f"layer {layer} already complete")
        return
    meta = pd.read_csv(PRODUCER / "metadata/regression_outer_folds_v1.csv", dtype={"protein_id": str})
    meta = meta[meta.source.eq("cameo")].sort_values("protein_id").reset_index(drop=True)
    if len(meta) != 154:
        raise ValueError(f"Expected 154 CAMEO proteins, found {len(meta)}")
    stored = {}
    for probe in PROBES:
        table = pd.read_csv(PRODUCER / probe / "geometry_metrics.csv", dtype={"protein_id": str},
                            usecols=["layer", "protein_id", "triangle_violation_fraction"])
        stored[probe] = table[table.layer.eq(layer)].set_index("protein_id").triangle_violation_fraction
    strata = {probe: load_strata(probe, layer) for probe in PROBES}
    log(f"layer {layer}: loaded metadata and strata")

    A, P, M = len(PROBES), len(meta), len(MODELS)
    counts = np.zeros((A, P, M, N_REPS, 6))              # triangles, violations, <5, 5-10, >=10, sum of violations
    slack_viol = np.zeros((A, P, M, N_SLACK))
    slack_n = np.zeros((P, N_SLACK))
    position_viol = np.zeros((A, P, 2, N_POSITION))      # observed, residuals_permuted
    position_n = np.zeros((P, N_POSITION))
    edges = np.zeros((A, P, 3 + 4))                      # z_ab, z_ac, z_bc, margin mean and SD: observed, then permuted
    exact = np.zeros(P, bool)
    worst_strata, worst_exact, worst_z = 0.0, 0.0, 0.0

    for p, row in meta.iterrows():
        n = int(row.n_residues)
        ca = helpers.extract_ca_coordinates(helpers.resolve_combined_pdb_path(row.protein_id, row.source))
        coordinates = np.asarray(ca["coordinates"], dtype=np.float64)
        if coordinates.shape[0] != n:
            raise ValueError(f"{row.protein_id}: residue mismatch")
        i, j = strict_lower_triangle_indices(n)
        y = helpers.compute_distance_matrix(coordinates)[i, j]
        sequence = np.searchsorted(SEQ_EDGES, np.abs(i - j), side="right") - 1
        distance = np.searchsorted(DIST_EDGES, y, side="right") - 1
        cell = sequence * N_D + distance

        triples, exact[p] = sample_triangles(n, p)
        low, mid, top = triples.T
        sides = np.stack((pair_position(mid, low), pair_position(top, low), pair_position(top, mid)), axis=1)
        order = np.argsort(y[sides], axis=1)
        sorted_sides = np.take_along_axis(sides, order, axis=1)   # shortest, middle, longest in the final structure
        a, b, c = y[sorted_sides].T
        slack_bin = np.searchsorted(SLACK_EDGES, np.clip((a + b - c) / c, 0.0, None), side="right") - 1
        slack_bin = np.minimum(slack_bin, N_SLACK - 1)
        slack_n[p] = np.bincount(slack_bin, minlength=N_SLACK)
        position = np.minimum(N_POSITION * triples // n, N_POSITION - 1)
        position_n[p] = np.bincount(position.ravel(), minlength=N_POSITION)

        for a_index, probe in enumerate(PROBES):
            prediction = np.load(EXPORT / f"layers/layer_{layer:02d}/{row.protein_id}/{probe}.npy").astype(np.float64)
            if prediction.shape != y.shape:
                raise ValueError(f"{row.protein_id}: prediction shape {prediction.shape}")
            residual = prediction - y
            mean, sd, cells, cell_n = protein_strata(strata[probe], row.protein_id)
            pool_mean, pool_sd = strata[probe]["pool_mean"], strata[probe]["pool_sd"]

            # The stored strata must match the errors recomputed here.
            error_sum = np.bincount(cell, weights=residual, minlength=N_S * N_D)
            worst_strata = max(worst_strata, float(np.max(np.abs(error_sum[cells] / cell_n - mean[cells]))))

            centred = residual - mean[cell]
            totals = np.bincount(i, weights=centred, minlength=n) + np.bincount(j, weights=centred, minlength=n)
            residue_effect = totals / max(n - 2, 1)
            remainder = centred - residue_effect[i] - residue_effect[j]
            bias, spread = residual.mean(), residual.std()

            for r in range(N_REPS):
                rng = np.random.default_rng([layer, a_index, p, r])
                noise = rng.standard_normal(y.size)
                permutation = within_cell_permutation(rng, cell)
                remainder_permutation = within_cell_permutation(rng, cell)
                permuted = residual[permutation]
                models = {
                    "observed": prediction,
                    "iid_matched": y + bias + spread * noise,
                    "pooled_mean_only": y + pool_mean[cell],
                    "pooled_strata": y + pool_mean[cell] + pool_sd[cell] * noise,
                    "protein_strata": y + mean[cell] + sd[cell] * noise,
                    "residuals_permuted": y + permuted,
                    "residues_plus_permuted": y + mean[cell] + residue_effect[i] + residue_effect[j]
                    + remainder[remainder_permutation],
                }
                for m, name in enumerate(MODELS):
                    if r > 0 and name not in STOCHASTIC:
                        counts[a_index, p, m, r] = counts[a_index, p, m, 0]
                        continue
                    violating, excess = score(models[name], sides)
                    counts[a_index, p, m, r] = (
                        sides.shape[0], violating.sum(), np.count_nonzero(excess < 5.0),
                        np.count_nonzero((excess >= 5.0) & (excess < 10.0)), np.count_nonzero(excess >= 10.0), excess.sum())
                    if r == 0:
                        slack_viol[a_index, p, m] = np.bincount(slack_bin[violating], minlength=N_SLACK)
                        if name in ("observed", "residuals_permuted"):
                            slot = 0 if name == "observed" else 1
                            position_viol[a_index, p, slot] = np.bincount(position[violating].ravel(), minlength=N_POSITION)
                if r == 0:
                    z = np.divide(centred, sd[cell], out=np.zeros_like(centred), where=sd[cell] > 0)
                    za, zb, zc = z[sorted_sides].T
                    ra, rb, rc = residual[sorted_sides].T
                    pa, pb, pc = permuted[sorted_sides].T
                    margin, margin_permuted = rc - ra - rb, pc - pa - pb
                    edges[a_index, p] = ((za * zb).mean(), (za * zc).mean(), (zb * zc).mean(),
                                         margin.mean(), margin.std(), margin_permuted.mean(), margin_permuted.std())

            fraction = counts[a_index, p, 0, 0, 1] / counts[a_index, p, 0, 0, 0]
            reference = float(stored[probe].loc[row.protein_id])
            if exact[p]:
                worst_exact = max(worst_exact, abs(fraction - reference))
            else:
                worst_z = max(worst_z, abs(fraction - reference) / math.sqrt(max(reference * (1 - reference), 1e-9) / N_TRIANGLES))
        if p % 50 == 0:
            log(f"layer {layer}: protein {p + 1}/{P}")

    log(f"check strata: worst mean error difference {worst_strata:.2e} A")
    log(f"check exact proteins ({int(exact.sum())}): worst violation fraction difference {worst_exact:.2e}")
    log(f"check sampled proteins ({int((~exact).sum())}): worst z {worst_z:.2f}")
    if worst_strata > 1e-5 or worst_exact > 5e-5 or worst_z > 6.0:
        raise ValueError("Observed values do not reproduce the stored strata or geometry tables")
    atomic_savez_compressed(
        final, counts=counts, slack_viol=slack_viol, slack_n=slack_n, position_viol=position_viol, position_n=position_n,
        edges=edges, exact=exact, protein_ids=meta.protein_id.to_numpy().astype(str),
        n_residues=meta.n_residues.to_numpy(), probes=np.asarray(PROBES), models=np.asarray(MODELS))
    log(f"layer {layer} complete")


if __name__ == "__main__":
    main()

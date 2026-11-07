#!/usr/bin/env python3
"""One target layer: prediction moments, and chain position crossed with distance and sequence.

Replays the 240 frozen shared-OLS probes in target_restandardized mode on the 154 CAMEO
proteins, as the production worker does, and stores what its shards lack:

- per-protein sum_p, sum_p2, sum_py by fine distance bin, which give correlation and gain;
- signed, absolute, and squared error by chain position x distance, and x sequence separation.

Every task must reproduce the production shard before it writes anything.
"""
from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, CLUSTER_DIR, CLUSTER_DIR / "cross_layer_transfer",
             CLUSTER_DIR / "cross_validation", CLUSTER_DIR / "triangle_violation_subgroups",
             SCRIPT_DIR.parent / "cross_layer_transfer_all_layers"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from followup_paths import GAIN_ROOT, PRODUCER, TRANSFER_ROOT  # noqa: E402

from all_layer_transfer_common import (  # noqa: E402
    LAYERS, N_DISTANCE_BINS, N_OUTER_FOLDS, N_POSITION_BINS, N_SEQUENCE_BINS, OUTLIER_PROTEIN,
    distance_fine_bins, load_probe_bank, position_bins, predict_all_sources, select_cohort,
    sequence_fine_bins,
)
from cross_validation_common import load_cv_folds  # noqa: E402
from regression_output_helpers import atomic_savez_compressed  # noqa: E402
from triangle_violation_subgroup_replay import load_protein  # noqa: E402

ROOT = GAIN_ROOT
PRODUCTION = TRANSFER_ROOT / "layer_shards"
TOLERANCE = 1e-8
STARTED = time.perf_counter()


def log(message: str) -> None:
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024
    print(f"[{time.perf_counter() - STARTED:7.1f}s] [peak_rss={rss:.0f}MiB] {message}", flush=True)


def grouped_sums(keys: np.ndarray, columns: list[np.ndarray]) -> tuple[np.ndarray, list[np.ndarray]]:
    """Sum each (rows, sources) column by key; returns occupied keys and (keys, sources) sums."""
    order = np.argsort(keys, kind="stable")
    ordered = keys[order]
    starts = np.flatnonzero(np.r_[True, ordered[1:] != ordered[:-1]])
    return ordered[starts], [np.add.reduceat(column[order], starts, axis=0) for column in columns]


def main() -> None:
    task = int(os.environ["SLURM_ARRAY_TASK_ID"])
    target = int(LAYERS[task])
    final = ROOT / "layer_shards" / f"target_layer_{target:02d}.npz"
    if final.is_file():
        log(f"target_layer={target} already complete")
        return
    bundle = load_cv_folds(metadata_dir=PRODUCER / "metadata")
    proteins = select_cohort(bundle, "cameo")
    bank, _, _ = load_probe_bank(PRODUCER, bundle, LAYERS)
    log(f"target_layer={target} loaded probes for {len(proteins)} proteins")

    S, P, D, Q, B, F = len(LAYERS), len(proteins), N_DISTANCE_BINS, N_SEQUENCE_BINS, N_POSITION_BINS, N_OUTER_FOLDS
    moments = np.zeros((S, P, D, 3))                        # sum_p, sum_p2, sum_py
    targets = np.zeros((P, D, 3))                           # n, sum_y, sum_y2
    pos_dist, pos_seq = np.zeros((S, F, B, D, 3)), np.zeros((S, F, B, Q, 3))   # sum_e, sum_abs, sum_sq
    pos_dist_n, pos_seq_n = np.zeros((F, B, D), np.int64), np.zeros((F, B, Q), np.int64)
    out_dist, out_seq = np.zeros((S, B, D, 3)), np.zeros((S, B, Q, 3))
    out_dist_n, out_seq_n = np.zeros((B, D), np.int64), np.zeros((B, Q), np.int64)

    for index, protein in enumerate(proteins):
        outer = bundle.outer_by_protein[protein]
        fold = int(outer.outer_fold)
        coordinates, _, pair_i, pair_j, features = load_protein(outer, target)
        weights, constants = bank.restandardized_weights(target, fold)
        predictions = predict_all_sources(features, weights, constants)
        del features
        y = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
        distance, sequence = distance_fine_bins(y), sequence_fine_bins(pair_i, pair_j)

        keys, (sum_p, sum_p2, sum_py) = grouped_sums(distance, [
            predictions, predictions * predictions, predictions * y[:, None]])
        moments[:, index, keys, 0], moments[:, index, keys, 1], moments[:, index, keys, 2] = sum_p.T, sum_p2.T, sum_py.T
        order = np.argsort(distance, kind="stable")
        starts = np.flatnonzero(np.r_[True, distance[order][1:] != distance[order][:-1]])
        targets[index, keys, 0] = np.diff(np.r_[starts, y.size])
        targets[index, keys, 1] = np.add.reduceat(y[order], starts)
        targets[index, keys, 2] = np.add.reduceat((y * y)[order], starts)

        errors = predictions - y[:, None]
        doubled = np.vstack([errors, errors])                # a pair counts for both of its residues
        position = np.r_[position_bins(pair_i, outer.n_residues), position_bins(pair_j, outer.n_residues)]
        columns = [doubled, np.abs(doubled), doubled * doubled]
        for strata, size, stats, counts, out_stats, out_counts in (
            (np.r_[distance, distance], D, pos_dist, pos_dist_n, out_dist, out_dist_n),
            (np.r_[sequence, sequence], Q, pos_seq, pos_seq_n, out_seq, out_seq_n),
        ):
            cell = position * size + strata
            keys, sums = grouped_sums(cell, columns)
            occupied = np.bincount(cell, minlength=B * size).reshape(B, size)
            counts[fold] += occupied
            for k, values in enumerate(sums):
                stats[:, fold, keys // size, keys % size, k] += values.T
            if protein == OUTLIER_PROTEIN:
                out_counts += occupied
                for k, values in enumerate(sums):
                    out_stats[:, keys // size, keys % size, k] += values.T
        if index % 50 == 0:
            log(f"target_layer={target} protein {index + 1}/{P}")

    with np.load(PRODUCTION / f"target_layer_{target:02d}.npz") as shard:
        if tuple(str(p) for p in shard["protein_ids"]) != tuple(proteins):
            raise ValueError("Protein order differs from the production shard")
        whole = shard["probe"][:, :, 0]
        position_stats = shard["position_stats"]
        position_n = shard["position_n"]
    sum_p, sum_p2, sum_py = (moments[..., k].sum(axis=2) for k in range(3))
    n, sum_y, sum_y2 = (targets[..., k].sum(axis=1) for k in range(3))
    identities = {
        "pair count": (whole[:, :, 0], np.broadcast_to(n, whole.shape[:2])),
        "sum_y": (whole[:, :, 1], np.broadcast_to(sum_y, whole.shape[:2])),
        "sum_y2": (whole[:, :, 2], np.broadcast_to(sum_y2, whole.shape[:2])),
        "sum_prediction": (whole[:, :, 3], sum_p),
        "sum_squared_error": (whole[:, :, 6], sum_p2 - 2 * sum_py + sum_y2),
        "position error sums": (position_stats, pos_dist.sum(axis=3)),
        "position by sequence sums": (position_stats, pos_seq.sum(axis=3)),
        "position counts by distance": (position_n, pos_dist_n.sum(axis=2)),
        "position counts by sequence": (position_n, pos_seq_n.sum(axis=2)),
    }
    for name, (expected, observed) in identities.items():
        worst = float(np.max(np.abs(expected - observed) / np.maximum(1.0, np.abs(expected))))
        log(f"check {name}: worst relative difference {worst:.2e}")
        if worst > TOLERANCE:
            raise ValueError(f"{name} disagrees with the production shard: {worst}")

    atomic_savez_compressed(
        final, moments=moments, targets=targets, pos_dist=pos_dist, pos_seq=pos_seq, pos_dist_n=pos_dist_n,
        pos_seq_n=pos_seq_n, out_dist=out_dist, out_seq=out_seq, out_dist_n=out_dist_n, out_seq_n=out_seq_n,
        protein_ids=np.asarray(proteins), outer_fold=np.asarray([int(bundle.outer_by_protein[p].outer_fold) for p in proteins]),
    )
    log(f"target_layer={target} complete")


if __name__ == "__main__":
    main()

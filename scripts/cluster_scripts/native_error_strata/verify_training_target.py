#!/usr/bin/env python3
"""Check the distances we measured against using fingerprints written during training.

The frozen probes are not readable, but each fit recorded statistics of its own
training targets. This recomputes them with the training code's own helpers from
the PDBs used for the error analysis, and compares every fold and layer. It also
compares each PDB with the SHA-256 pinned by the prediction export. Nothing is fitted.
"""
from __future__ import annotations

import hashlib
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import cluster_regression_helpers as helpers  # noqa: E402
from canonical_storage import csv_root  # noqa: E402
from cluster_regression_helpers import compute_distance_matrix, extract_ca_coordinates  # noqa: E402
from regression_protocol import strict_lower_triangle_indices  # noqa: E402
from strata_bins import output_root  # noqa: E402

ECOD = os.environ.get("CROSS_LAYER_ECOD_EXPERIMENTS_ROOT")
if ECOD:
    helpers.ECOD_PAIR_BLOCKS_ROOT = Path(ECOD) / "representation_saves"
    helpers.ECOD_PDB_ROOT = Path(ECOD) / "openfold_reps_outputs_sm"
EXPORT = csv_root() / "regression_5fold_v1/predicted_distances_5fold_v1"
TRAINING = csv_root() / "regression_5fold_v1/shared_ols_base256/training_metrics.csv"
STATUS = csv_root() / "diagnostic_experiments/cross_layer_probe_transfer_v1/source_probe_status.csv"
ANALYSIS, N_FOLDS, N_LAYERS, TOLERANCE = "shared_ols_base256", 5, 48, 1e-8
STARTED = time.perf_counter()


def log(message):
    print(f"[{time.perf_counter() - STARTED:7.1f}s] {message}", flush=True)


def scan(meta):
    """Per-protein target statistics, built exactly as load_target_records builds targets."""
    rows, mismatched = [], []
    for k, row in meta.iterrows():
        path = helpers.resolve_combined_pdb_path(row.protein_id, row.source)
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != row.pdb_sha256:
            mismatched.append(row.protein_id)
        coordinates = np.asarray(extract_ca_coordinates(path)["coordinates"], dtype=np.float64)
        pair_i, pair_j = strict_lower_triangle_indices(int(row.n_residues))
        targets = np.asarray(compute_distance_matrix(coordinates)[pair_i, pair_j], dtype=np.float64)
        if targets.size != row.n_pairs:
            raise ValueError(f"{row.protein_id}: expected {row.n_pairs} pairs, found {targets.size}")
        rows.append((targets.size, targets.sum(), (targets * targets).sum(), targets.mean()))
        if k % 100 == 0:
            log(f"protein {k + 1}/{len(meta)}")
    return np.array(rows), mismatched


def main():
    meta = pd.read_csv(EXPORT / "metadata/proteins.csv", dtype={"protein_id": str})
    if len(meta) != 552:
        raise ValueError(f"Expected the 552-protein cohort, found {len(meta)}")
    stats, mismatched = scan(meta)
    training = pd.read_csv(TRAINING)
    training = training[training.analysis_name.eq(ANALYSIS)]
    status = pd.read_csv(STATUS)
    folds = []
    for fold in range(N_FOLDS):
        train = meta.outer_fold.to_numpy() != fold
        n, sum_y, sum_y2, protein_means = stats[train, 0], stats[train, 1], stats[train, 2], stats[train, 3]
        mean = sum_y.sum() / n.sum()
        ours = {"n_training_proteins": int(train.sum()), "n_training_pairs": int(n.sum()),
                "mean_true_distance": float(mean), "target_variance": float(sum_y2.sum() / n.sum() - mean * mean),
                "global_baseline_value": float(protein_means.mean())}
        recorded = training[training.outer_fold.eq(fold)]
        baselines = status[status.outer_fold.eq(fold)].global_baseline_value
        folds.append({
            "outer_fold": fold, "ours": ours, "training_rows": len(recorded), "baseline_rows": len(baselines),
            "recorded_mean_true_distance": float(recorded.mean_true_distance.iloc[0]),
            "recorded_target_variance": float(recorded.target_variance.iloc[0]),
            "recorded_global_baseline": float(baselines.iloc[0]),
            "max_diff_proteins": int((recorded.n_training_proteins - ours["n_training_proteins"]).abs().max()),
            "max_diff_pairs": int((recorded.n_training_pairs - ours["n_training_pairs"]).abs().max()),
            "max_diff_mean": float((recorded.mean_true_distance - ours["mean_true_distance"]).abs().max()),
            "max_diff_variance": float((recorded.target_variance - ours["target_variance"]).abs().max()),
            "max_diff_baseline": float((baselines - ours["global_baseline_value"]).abs().max()),
        })
        log(json.dumps(folds[-1]))
    passed = not mismatched and all(
        f["training_rows"] == N_LAYERS and f["max_diff_proteins"] == 0 and f["max_diff_pairs"] == 0
        and max(f["max_diff_mean"], f["max_diff_variance"], f["max_diff_baseline"]) <= TOLERANCE
        for f in folds)
    summary = {"passed": passed, "tolerance": TOLERANCE, "pdbs_hashed": len(meta), "pdb_hash_mismatches": mismatched,
               "training_rows_compared": sum(f["training_rows"] for f in folds),
               "baseline_rows_compared": sum(f["baseline_rows"] for f in folds), "folds": folds,
               "total_seconds": time.perf_counter() - STARTED}
    (output_root() / "data").mkdir(parents=True, exist_ok=True)
    (output_root() / "data/training_target_check.json").write_text(json.dumps(summary, indent=2))
    log(f"training target check passed={passed}; pdb hash mismatches={len(mismatched)}")
    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()

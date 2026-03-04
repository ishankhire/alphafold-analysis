#!/usr/bin/env python3
"""
benchmark_kfold_mds_runtime.py
==============================
Standalone local benchmark harness for the k-fold MDS regression workflow.

It compares two execution modes:
  1. baseline    -- current fold->layer order with stacked OLS
  2. layer_cache -- layer->fold order, cache one layer at a time

The script computes geometry on all proteins in each fold, uses an optimized
but exact triangle-violation routine, and runs a sampled legacy-vs-optimized
geometry validation pass after the benchmark completes.

Run this script with the repo venv:
    ./.venv/bin/python scripts/multi_protein/regression/benchmark_kfold_mds_runtime.py
"""

import argparse
import csv
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import KFold
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from lib.data_loading import discover_proteins, find_pdb_file, get_pair_block_path
from lib.structure import (
    compute_distance_matrix,
    compute_tm_score,
    extract_ca_coordinates,
    superimpose_structures,
)


N_LAYERS_DEFAULT = 48
N_CHANNELS = 128
FEATURE_DIM = 2 * N_CHANNELS
RANDOM_STATE = 42

PROTEINS_DIR = PROJECT_ROOT / "proteins"
OUTPUT_ROOT = PROTEINS_DIR / "multi_protein_results"

VALID_MODES = ("baseline", "layer_cache")
REGRESSION_METRIC_KEYS = ("r2_train", "r2_test", "mae_test", "rmse_test")
GEOMETRY_KEYS = (
    "rmsd",
    "tm_score",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "neg_eigenvalue_fraction",
    "n_negative_predictions",
    "stress",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
)
GEOMETRY_INT_KEYS = ("n_negative_eigenvalues", "n_negative_predictions")
TRIANGLE_BLOCK_SIZE = 16
GEOMETRY_OPT_COMPARISON = "legacy_vs_optimized_geometry_sampled"
STAGE_COLUMNS = (
    "feature_build_train_total",
    "feature_build_test_total",
    "assemble_train_from_cache_total",
    "assemble_test_from_cache_total",
    "cache_build_layer_total",
    "scale_fit_transform_train_total",
    "scale_transform_test_total",
    "regression_fit_total",
    "predict_train_total",
    "predict_test_total",
    "geometry_total",
)


# ===========================================================================
# Geometry helpers (copied from kfold_mds_regression.py without behavior change)
# ===========================================================================

def classical_mds(distance_matrix, n_dims=3):
    """Classical (Torgerson) MDS: recover coordinates from pairwise distances.

    Returns (coords, eigenvalues) where eigenvalues are sorted descending.
    """
    n = distance_matrix.shape[0]
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


def compute_triangle_violations_legacy(pred_dist):
    """Legacy exact triangle-violation calculation using a per-j Python loop."""
    R = pred_dist.shape[0]
    min_indirect = np.full((R, R), np.inf)
    for j in range(R):
        candidate = pred_dist[:, j:j + 1] + pred_dist[j:j + 1, :]
        min_indirect = np.minimum(min_indirect, candidate)
    np.fill_diagonal(min_indirect, 0.0)

    mask = pred_dist > min_indirect
    np.fill_diagonal(mask, False)
    n_violations = mask.sum()
    total_pairs = R * (R - 1)

    violation_fraction = float(n_violations / total_pairs) if total_pairs > 0 else 0.0

    if n_violations > 0:
        violations = pred_dist[mask] - min_indirect[mask]
        mean_violation = float(violations.mean())
        max_violation = float(violations.max())
    else:
        mean_violation = 0.0
        max_violation = 0.0

    return violation_fraction, mean_violation, max_violation


def compute_triangle_violations_optimized(pred_dist, block_size=TRIANGLE_BLOCK_SIZE):
    """Exact triangle-violation calculation using blocked vectorized reduction."""
    R = pred_dist.shape[0]
    min_indirect = np.full((R, R), np.inf)
    for j_start in range(0, R, block_size):
        j_end = min(R, j_start + block_size)
        left = pred_dist[:, j_start:j_end]
        right = pred_dist[j_start:j_end, :]
        candidates = left[:, :, np.newaxis] + right[np.newaxis, :, :]
        block_min = np.min(candidates, axis=1)
        min_indirect = np.minimum(min_indirect, block_min)
    np.fill_diagonal(min_indirect, 0.0)

    mask = pred_dist > min_indirect
    np.fill_diagonal(mask, False)
    n_violations = mask.sum()
    total_pairs = R * (R - 1)

    violation_fraction = float(n_violations / total_pairs) if total_pairs > 0 else 0.0

    if n_violations > 0:
        violations = pred_dist[mask] - min_indirect[mask]
        mean_violation = float(violations.mean())
        max_violation = float(violations.max())
    else:
        mean_violation = 0.0
        max_violation = 0.0

    return violation_fraction, mean_violation, max_violation


def analyze_protein_geometry_impl(n_residues, true_coords, pb_i, pb_j, y_pred, triangle_fn):
    """Run MDS reconstruction + triangle inequality analysis on one protein."""
    R = n_residues

    pred_dist = np.zeros((R, R))
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

    tm = compute_tm_score(true_coords, transformed, R)

    mds_dist = compute_distance_matrix(transformed)
    triu_i, triu_j = np.triu_indices(R, k=1)
    d_pred = pred_dist[triu_i, triu_j]
    d_mds = mds_dist[triu_i, triu_j]
    stress = float(np.sqrt(np.sum((d_pred - d_mds) ** 2) / np.sum(d_pred ** 2)))

    viol_frac, viol_mean, viol_max = triangle_fn(pred_dist)

    return {
        "rmsd": rmsd,
        "tm_score": tm,
        "frac_top3_eigenvalues": frac_top3,
        "n_negative_eigenvalues": n_neg_evals,
        "neg_eigenvalue_fraction": neg_frac,
        "n_negative_predictions": n_negative_preds,
        "stress": stress,
        "triangle_violation_fraction": viol_frac,
        "triangle_mean_violation": viol_mean,
        "triangle_max_violation": viol_max,
    }


def analyze_protein_geometry_legacy(n_residues, true_coords, pb_i, pb_j, y_pred):
    """Legacy geometry path using the original triangle-violation algorithm."""
    return analyze_protein_geometry_impl(
        n_residues,
        true_coords,
        pb_i,
        pb_j,
        y_pred,
        triangle_fn=compute_triangle_violations_legacy,
    )


def analyze_protein_geometry_optimized(n_residues, true_coords, pb_i, pb_j, y_pred):
    """Optimized geometry path using exact blocked triangle-violation reduction."""
    return analyze_protein_geometry_impl(
        n_residues,
        true_coords,
        pb_i,
        pb_j,
        y_pred,
        triangle_fn=compute_triangle_violations_optimized,
    )


# ===========================================================================
# Shared helpers
# ===========================================================================

def parse_args():
    parser = argparse.ArgumentParser(
        description="Benchmark k-fold MDS regression runtime across execution modes."
    )
    parser.add_argument("--max-proteins", type=int, default=None)
    parser.add_argument("--max-layers", type=int, default=N_LAYERS_DEFAULT)
    parser.add_argument("--n-folds", type=int, default=3)
    parser.add_argument(
        "--modes",
        type=str,
        default="baseline,layer_cache",
    )
    parser.add_argument(
        "--output-subdir",
        type=str,
        default="kfold_mds_runtime_benchmark",
    )
    parser.add_argument(
        "--geometry-workers",
        type=int,
        default=min(4, os.cpu_count() or 1),
    )
    args = parser.parse_args()

    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    unknown = [m for m in modes if m not in VALID_MODES]
    if not modes:
        parser.error("At least one mode must be selected via --modes.")
    if unknown:
        parser.error(
            "Unknown mode(s): {}. Valid modes: {}".format(
                ", ".join(unknown), ", ".join(VALID_MODES)
            )
        )
    if args.max_layers < 1 or args.max_layers > N_LAYERS_DEFAULT:
        parser.error(f"--max-layers must be between 1 and {N_LAYERS_DEFAULT}.")
    if args.n_folds < 2:
        parser.error("--n-folds must be at least 2.")
    if args.geometry_workers < 1:
        parser.error("--geometry-workers must be at least 1.")

    args.modes = modes
    return args


def build_feature_chunk(protein_record, layer):
    """Load one protein/layer pair block and extract its dense feature chunk."""
    pb_path = get_pair_block_path(
        protein_record["protein_dir"], protein_record["protein_id"], layer
    )
    pb = np.load(str(pb_path))
    chunk = np.concatenate(
        [
            pb[protein_record["pb_i"], protein_record["pb_j"], :],
            pb[protein_record["pb_j"], protein_record["pb_i"], :],
        ],
        axis=1,
    )
    del pb
    return chunk


def stack_cached_chunks(protein_records, chunk_map):
    """Stack cached per-protein chunks in the provided protein order."""
    return np.vstack([chunk_map[d["protein_id"]] for d in protein_records])


def compute_regression_metrics(y_true, y_pred):
    """Compute regression metrics for a true/predicted vector pair."""
    return {
        "r2": float(r2_score(y_true, y_pred)),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "rmse": float(math.sqrt(np.mean((y_true - y_pred) ** 2))),
    }


def record_stage(timing_rows, mode, fold, layer, stage, seconds):
    timing_rows.append(
        {
            "mode": mode,
            "fold": int(fold),
            "layer": int(layer),
            "stage": stage,
            "seconds": float(seconds),
        }
    )


def summarize_stage_times(timing_rows):
    stage_totals = defaultdict(lambda: defaultdict(float))
    fold_layer_counts = defaultdict(int)
    for row in timing_rows:
        stage_totals[row["mode"]][row["stage"]] += row["seconds"]
        if row["stage"] == "total_fold_layer":
            fold_layer_counts[row["mode"]] += 1
    return stage_totals, fold_layer_counts


def build_offsets(protein_records):
    offsets = []
    offset = 0
    for d in protein_records:
        n_pairs = len(d["y"])
        offsets.append((d["protein_id"], offset, offset + n_pairs))
        offset += n_pairs
    return offsets


def format_seconds(value):
    return f"{value:.6f}"


def make_geometry_work_items(protein_records, prediction_map, split):
    """Build deterministic per-protein geometry work items for one split."""
    work_items = []
    for d in protein_records:
        work_items.append(
            {
                "protein_id": d["protein_id"],
                "split": split,
                "record": d,
                "y_pred": prediction_map[d["protein_id"]],
            }
        )
    return work_items


def run_geometry_work_item(work_item, analyze_fn):
    """Analyze one protein and return an ordered geometry row."""
    d = work_item["record"]
    metrics = analyze_fn(
        d["n_residues"],
        d["coordinates"],
        d["pb_i"],
        d["pb_j"],
        work_item["y_pred"],
    )
    row = {
        "split": work_item["split"],
        **metrics,
    }
    return work_item["protein_id"], row


def run_geometry_batch(records_with_predictions, n_jobs, analyze_fn):
    """Run ordered per-protein geometry, optionally in parallel."""
    if n_jobs == 1:
        ordered_results = [
            run_geometry_work_item(work_item, analyze_fn)
            for work_item in records_with_predictions
        ]
    else:
        ordered_results = Parallel(n_jobs=n_jobs, prefer="threads")(
            delayed(run_geometry_work_item)(work_item, analyze_fn)
            for work_item in records_with_predictions
        )
    return {protein_id: row for protein_id, row in ordered_results}


def load_protein_data(max_proteins):
    proteins_dir = PROTEINS_DIR.resolve()
    all_proteins = discover_proteins(proteins_dir)
    print(f"Discovered {len(all_proteins)} protein directories.")
    if max_proteins is not None:
        all_proteins = all_proteins[:max_proteins]
    print(f"Selected {len(all_proteins)} protein directories for loading.")

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
                print(f"[SKIP] {protein_id}: pair block 0 not found")
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
            pb_i, pb_j = np.triu_indices(n_ca, k=1)
            y = dist_matrix[pb_i, pb_j]

            protein_data.append(
                {
                    "protein_id": protein_id,
                    "protein_dir": protein_dir,
                    "n_residues": n_ca,
                    "coordinates": ca_data["coordinates"],
                    "pb_i": pb_i,
                    "pb_j": pb_j,
                    "y": y,
                    "n_pairs": len(y),
                }
            )
            print(f"[OK]   {protein_id}: {n_ca} residues, {len(y):,} pairs")
        except Exception as exc:
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {exc}")
            traceback.print_exc()

    return protein_data, skipped


def build_folds(protein_data, n_folds):
    if len(protein_data) < n_folds:
        raise ValueError(
            f"Need at least {n_folds} loaded proteins for {n_folds}-fold CV; "
            f"got {len(protein_data)}."
        )
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=RANDOM_STATE)
    return list(kf.split(np.arange(len(protein_data))))

def extract_prediction_map(protein_records, y_pred):
    prediction_map = {}
    for d, (pid, start, end) in zip(protein_records, build_offsets(protein_records)):
        prediction_map[pid] = np.array(y_pred[start:end], copy=True)
    return prediction_map


def run_stacked_regression_common(
    mode,
    fold_idx,
    layer,
    train_records,
    test_records,
    X_train_raw,
    X_test_raw,
    timing_rows,
    geometry_workers,
):
    y_train = np.concatenate([d["y"] for d in train_records])
    y_test = np.concatenate([d["y"] for d in test_records])

    t_stage = time.perf_counter()
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train_raw)
    record_stage(
        timing_rows,
        mode,
        fold_idx,
        layer,
        "scale_fit_transform_train",
        time.perf_counter() - t_stage,
    )

    t_stage = time.perf_counter()
    X_test = scaler.transform(X_test_raw)
    record_stage(
        timing_rows,
        mode,
        fold_idx,
        layer,
        "scale_transform_test",
        time.perf_counter() - t_stage,
    )

    t_stage = time.perf_counter()
    model = LinearRegression()
    model.fit(X_train, y_train)
    record_stage(
        timing_rows,
        mode,
        fold_idx,
        layer,
        "regression_fit",
        time.perf_counter() - t_stage,
    )

    t_stage = time.perf_counter()
    y_pred_train = model.predict(X_train)
    record_stage(
        timing_rows,
        mode,
        fold_idx,
        layer,
        "predict_train",
        time.perf_counter() - t_stage,
    )

    t_stage = time.perf_counter()
    y_pred_test = model.predict(X_test)
    record_stage(
        timing_rows,
        mode,
        fold_idx,
        layer,
        "predict_test",
        time.perf_counter() - t_stage,
    )

    train_metrics = compute_regression_metrics(y_train, y_pred_train)
    test_metrics = compute_regression_metrics(y_test, y_pred_test)
    train_prediction_map = extract_prediction_map(train_records, y_pred_train)
    test_prediction_map = extract_prediction_map(test_records, y_pred_test)
    protein_predictions = {}
    protein_predictions.update(train_prediction_map)
    protein_predictions.update(test_prediction_map)

    t_stage = time.perf_counter()
    geometry_work_items = []
    geometry_work_items.extend(
        make_geometry_work_items(train_records, train_prediction_map, "train")
    )
    geometry_work_items.extend(
        make_geometry_work_items(test_records, test_prediction_map, "test")
    )
    geometry_map = run_geometry_batch(
        geometry_work_items,
        geometry_workers,
        analyze_protein_geometry_optimized,
    )
    record_stage(
        timing_rows,
        mode,
        fold_idx,
        layer,
        "geometry",
        time.perf_counter() - t_stage,
    )

    result_row = {
        "r2_train": train_metrics["r2"],
        "r2_test": test_metrics["r2"],
        "mae_test": test_metrics["mae"],
        "rmse_test": test_metrics["rmse"],
        "intercept": float(model.intercept_),
        "coef": np.array(model.coef_, copy=True),
    }

    del X_train, X_test, y_pred_train, y_pred_test
    return result_row, protein_predictions, geometry_map


def run_mode_baseline(protein_data, folds, n_layers, geometry_workers):
    mode = "baseline"
    results_grid = {}
    protein_predictions = {}
    geometry_rows = {}
    timing_rows = []
    t_mode = time.perf_counter()

    print(f"\n=== Mode: {mode} ===")
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        train_records = [protein_data[i] for i in train_idx]
        test_records = [protein_data[i] for i in test_idx]
        print(
            f"[{mode}] fold {fold_idx}: {len(train_records)} train proteins, "
            f"{len(test_records)} test proteins"
        )

        for layer in range(n_layers):
            print(f"[{mode}] fold {fold_idx} layer {layer + 1}/{n_layers}")
            t_total = time.perf_counter()

            t_stage = time.perf_counter()
            train_chunks = [build_feature_chunk(d, layer) for d in train_records]
            X_train_raw = np.vstack(train_chunks)
            record_stage(
                timing_rows,
                mode,
                fold_idx,
                layer,
                "feature_build_train",
                time.perf_counter() - t_stage,
            )

            t_stage = time.perf_counter()
            test_chunks = [build_feature_chunk(d, layer) for d in test_records]
            X_test_raw = np.vstack(test_chunks)
            record_stage(
                timing_rows,
                mode,
                fold_idx,
                layer,
                "feature_build_test",
                time.perf_counter() - t_stage,
            )

            result_row, prediction_map, geometry_map = run_stacked_regression_common(
                mode,
                fold_idx,
                layer,
                train_records,
                test_records,
                X_train_raw,
                X_test_raw,
                timing_rows,
                geometry_workers,
            )
            results_grid[(fold_idx, layer)] = result_row
            for pid, pred in prediction_map.items():
                protein_predictions[(fold_idx, layer, pid)] = pred
            for pid, geom in geometry_map.items():
                geometry_rows[(fold_idx, layer, pid)] = geom

            record_stage(
                timing_rows,
                mode,
                fold_idx,
                layer,
                "total_fold_layer",
                time.perf_counter() - t_total,
            )

            del train_chunks, test_chunks, X_train_raw, X_test_raw

    mode_total_seconds = time.perf_counter() - t_mode
    print(f"[{mode}] total runtime: {mode_total_seconds:.2f}s")
    return {
        "mode": mode,
        "results_grid": results_grid,
        "protein_predictions": protein_predictions,
        "geometry_rows": geometry_rows,
        "timing_rows": timing_rows,
        "mode_total_seconds": mode_total_seconds,
    }


def run_mode_layer_cache(protein_data, folds, n_layers, geometry_workers):
    mode = "layer_cache"
    results_grid = {}
    protein_predictions = {}
    geometry_rows = {}
    timing_rows = []
    t_mode = time.perf_counter()

    print(f"\n=== Mode: {mode} ===")
    for layer in range(n_layers):
        print(f"[{mode}] layer {layer + 1}/{n_layers}")
        t_stage = time.perf_counter()
        chunk_map = {
            d["protein_id"]: build_feature_chunk(d, layer)
            for d in protein_data
        }
        record_stage(
            timing_rows,
            mode,
            -1,
            layer,
            "cache_build_layer",
            time.perf_counter() - t_stage,
        )

        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            train_records = [protein_data[i] for i in train_idx]
            test_records = [protein_data[i] for i in test_idx]
            t_total = time.perf_counter()

            t_stage = time.perf_counter()
            X_train_raw = stack_cached_chunks(train_records, chunk_map)
            record_stage(
                timing_rows,
                mode,
                fold_idx,
                layer,
                "assemble_train_from_cache",
                time.perf_counter() - t_stage,
            )

            t_stage = time.perf_counter()
            X_test_raw = stack_cached_chunks(test_records, chunk_map)
            record_stage(
                timing_rows,
                mode,
                fold_idx,
                layer,
                "assemble_test_from_cache",
                time.perf_counter() - t_stage,
            )

            result_row, prediction_map, geometry_map = run_stacked_regression_common(
                mode,
                fold_idx,
                layer,
                train_records,
                test_records,
                X_train_raw,
                X_test_raw,
                timing_rows,
                geometry_workers,
            )
            results_grid[(fold_idx, layer)] = result_row
            for pid, pred in prediction_map.items():
                protein_predictions[(fold_idx, layer, pid)] = pred
            for pid, geom in geometry_map.items():
                geometry_rows[(fold_idx, layer, pid)] = geom

            record_stage(
                timing_rows,
                mode,
                fold_idx,
                layer,
                "total_fold_layer",
                time.perf_counter() - t_total,
            )

            del X_train_raw, X_test_raw

        del chunk_map

    mode_total_seconds = time.perf_counter() - t_mode
    print(f"[{mode}] total runtime: {mode_total_seconds:.2f}s")
    return {
        "mode": mode,
        "results_grid": results_grid,
        "protein_predictions": protein_predictions,
        "geometry_rows": geometry_rows,
        "timing_rows": timing_rows,
        "mode_total_seconds": mode_total_seconds,
    }


def get_tolerances(comparison_name):
    return {
        "metric": 1e-10,
        "intercept": 1e-10,
        "coef": 1e-10,
        "pred": 1e-10,
        "geometry_float": 1e-10,
    }


def validate_modes(mode_a_result, mode_b_result, comparison_name):
    tolerances = get_tolerances(comparison_name)
    failure_rows = []

    max_abs_metric_delta = 0.0
    max_abs_intercept_delta = 0.0
    max_abs_coef_delta = 0.0
    max_abs_pred_delta = 0.0
    pred_abs_sum = 0.0
    pred_count = 0

    result_keys_a = set(mode_a_result["results_grid"].keys())
    result_keys_b = set(mode_b_result["results_grid"].keys())
    if result_keys_a != result_keys_b:
        failure_rows.append(
            {
                "comparison": comparison_name,
                "kind": "regression",
                "fold": -1,
                "layer": -1,
                "protein_id": "",
                "metric": "result_key_set",
                "observed_delta": float(abs(len(result_keys_a) - len(result_keys_b))),
                "tolerance": 0.0,
            }
        )

    for key in sorted(result_keys_a & result_keys_b):
        fold_idx, layer = key
        a_row = mode_a_result["results_grid"][key]
        b_row = mode_b_result["results_grid"][key]

        for metric in REGRESSION_METRIC_KEYS:
            delta = abs(a_row[metric] - b_row[metric])
            max_abs_metric_delta = max(max_abs_metric_delta, delta)
            if delta > tolerances["metric"]:
                failure_rows.append(
                    {
                        "comparison": comparison_name,
                        "kind": "regression",
                        "fold": fold_idx,
                        "layer": layer,
                        "protein_id": "",
                        "metric": metric,
                        "observed_delta": float(delta),
                        "tolerance": tolerances["metric"],
                    }
                )

        intercept_delta = abs(a_row["intercept"] - b_row["intercept"])
        max_abs_intercept_delta = max(max_abs_intercept_delta, intercept_delta)
        if intercept_delta > tolerances["intercept"]:
            failure_rows.append(
                {
                    "comparison": comparison_name,
                    "kind": "regression",
                    "fold": fold_idx,
                    "layer": layer,
                    "protein_id": "",
                    "metric": "intercept",
                    "observed_delta": float(intercept_delta),
                    "tolerance": tolerances["intercept"],
                }
            )

        coef_delta = float(np.max(np.abs(a_row["coef"] - b_row["coef"])))
        max_abs_coef_delta = max(max_abs_coef_delta, coef_delta)
        if coef_delta > tolerances["coef"]:
            failure_rows.append(
                {
                    "comparison": comparison_name,
                    "kind": "regression",
                    "fold": fold_idx,
                    "layer": layer,
                    "protein_id": "",
                    "metric": "coef",
                    "observed_delta": coef_delta,
                    "tolerance": tolerances["coef"],
                }
            )

    pred_keys_a = set(mode_a_result["protein_predictions"].keys())
    pred_keys_b = set(mode_b_result["protein_predictions"].keys())
    if pred_keys_a != pred_keys_b:
        failure_rows.append(
            {
                "comparison": comparison_name,
                "kind": "regression",
                "fold": -1,
                "layer": -1,
                "protein_id": "",
                "metric": "prediction_key_set",
                "observed_delta": float(abs(len(pred_keys_a) - len(pred_keys_b))),
                "tolerance": 0.0,
            }
        )

    for key in sorted(pred_keys_a & pred_keys_b):
        fold_idx, layer, protein_id = key
        a_pred = mode_a_result["protein_predictions"][key]
        b_pred = mode_b_result["protein_predictions"][key]
        diff = np.abs(a_pred - b_pred)
        local_max = float(diff.max()) if diff.size else 0.0
        max_abs_pred_delta = max(max_abs_pred_delta, local_max)
        pred_abs_sum += float(diff.sum())
        pred_count += int(diff.size)
        if local_max > tolerances["pred"]:
            failure_rows.append(
                {
                    "comparison": comparison_name,
                    "kind": "regression",
                    "fold": fold_idx,
                    "layer": layer,
                    "protein_id": protein_id,
                    "metric": "prediction",
                    "observed_delta": local_max,
                    "tolerance": tolerances["pred"],
                }
            )

    max_geometry_deltas = {key: 0.0 for key in GEOMETRY_KEYS}
    geom_keys_a = set(mode_a_result["geometry_rows"].keys())
    geom_keys_b = set(mode_b_result["geometry_rows"].keys())
    if geom_keys_a != geom_keys_b:
        failure_rows.append(
            {
                "comparison": comparison_name,
                "kind": "geometry",
                "fold": -1,
                "layer": -1,
                "protein_id": "",
                "metric": "geometry_key_set",
                "observed_delta": float(abs(len(geom_keys_a) - len(geom_keys_b))),
                "tolerance": 0.0,
            }
        )

    for key in sorted(geom_keys_a & geom_keys_b):
        fold_idx, layer, protein_id = key
        a_geom = mode_a_result["geometry_rows"][key]
        b_geom = mode_b_result["geometry_rows"][key]
        if a_geom["split"] != b_geom["split"]:
            failure_rows.append(
                {
                    "comparison": comparison_name,
                    "kind": "geometry",
                    "fold": fold_idx,
                    "layer": layer,
                    "protein_id": protein_id,
                    "metric": "split",
                    "observed_delta": 1.0,
                    "tolerance": 0.0,
                }
            )
        for metric in GEOMETRY_KEYS:
            delta = abs(a_geom[metric] - b_geom[metric])
            max_geometry_deltas[metric] = max(max_geometry_deltas[metric], float(delta))
            tol = 0.0 if metric in GEOMETRY_INT_KEYS else tolerances["geometry_float"]
            if delta > tol:
                failure_rows.append(
                    {
                        "comparison": comparison_name,
                        "kind": "geometry",
                        "fold": fold_idx,
                        "layer": layer,
                        "protein_id": protein_id,
                        "metric": metric,
                        "observed_delta": float(delta),
                        "tolerance": tol,
                    }
                )

    regression_summary = {
        "comparison": comparison_name,
        "max_abs_metric_delta": max_abs_metric_delta,
        "max_abs_intercept_delta": max_abs_intercept_delta,
        "max_abs_coef_delta": max_abs_coef_delta,
        "max_abs_pred_delta": max_abs_pred_delta,
        "mean_abs_pred_delta": (pred_abs_sum / pred_count) if pred_count else 0.0,
        "passed": not any(row["kind"] == "regression" for row in failure_rows),
    }

    geometry_summary = {
        "comparison": comparison_name,
        "max_abs_rmsd_delta": max_geometry_deltas["rmsd"],
        "max_abs_tm_score_delta": max_geometry_deltas["tm_score"],
        "max_abs_frac_top3_eigenvalues_delta": max_geometry_deltas["frac_top3_eigenvalues"],
        "max_abs_n_negative_eigenvalues_delta": max_geometry_deltas["n_negative_eigenvalues"],
        "max_abs_neg_eigenvalue_fraction_delta": max_geometry_deltas["neg_eigenvalue_fraction"],
        "max_abs_n_negative_predictions_delta": max_geometry_deltas["n_negative_predictions"],
        "max_abs_stress_delta": max_geometry_deltas["stress"],
        "max_abs_triangle_violation_fraction_delta": max_geometry_deltas["triangle_violation_fraction"],
        "max_abs_triangle_mean_violation_delta": max_geometry_deltas["triangle_mean_violation"],
        "max_abs_triangle_max_violation_delta": max_geometry_deltas["triangle_max_violation"],
        "passed": not any(row["kind"] == "geometry" for row in failure_rows),
    }

    return regression_summary, geometry_summary, failure_rows


def run_sampled_geometry_optimization_validation(
    source_mode,
    source_result,
    protein_records_by_id,
):
    failure_rows = []
    max_geometry_deltas = {key: 0.0 for key in GEOMETRY_KEYS}
    geometry_keys = sorted(source_result["geometry_rows"].keys())
    sampled_keys = geometry_keys[::20] if geometry_keys else []

    for fold_idx, layer, protein_id in sampled_keys:
        optimized_geom = source_result["geometry_rows"][(fold_idx, layer, protein_id)]
        protein_record = protein_records_by_id[protein_id]
        y_pred = source_result["protein_predictions"][(fold_idx, layer, protein_id)]
        legacy_geom = analyze_protein_geometry_legacy(
            protein_record["n_residues"],
            protein_record["coordinates"],
            protein_record["pb_i"],
            protein_record["pb_j"],
            y_pred,
        )

        for metric in GEOMETRY_KEYS:
            delta = abs(optimized_geom[metric] - legacy_geom[metric])
            max_geometry_deltas[metric] = max(max_geometry_deltas[metric], float(delta))
            tol = 0.0 if metric in GEOMETRY_INT_KEYS else 1e-10
            if delta > tol:
                failure_rows.append(
                    {
                        "comparison": GEOMETRY_OPT_COMPARISON,
                        "kind": "geometry_optimization",
                        "fold": fold_idx,
                        "layer": layer,
                        "protein_id": protein_id,
                        "metric": metric,
                        "observed_delta": float(delta),
                        "tolerance": tol,
                    }
                )

    summary_row = {
        "source_mode": source_mode,
        "sampled_cases": len(sampled_keys),
        "max_abs_rmsd_delta": max_geometry_deltas["rmsd"],
        "max_abs_tm_score_delta": max_geometry_deltas["tm_score"],
        "max_abs_frac_top3_eigenvalues_delta": max_geometry_deltas["frac_top3_eigenvalues"],
        "max_abs_n_negative_eigenvalues_delta": max_geometry_deltas["n_negative_eigenvalues"],
        "max_abs_neg_eigenvalue_fraction_delta": max_geometry_deltas["neg_eigenvalue_fraction"],
        "max_abs_n_negative_predictions_delta": max_geometry_deltas["n_negative_predictions"],
        "max_abs_stress_delta": max_geometry_deltas["stress"],
        "max_abs_triangle_violation_fraction_delta": max_geometry_deltas["triangle_violation_fraction"],
        "max_abs_triangle_mean_violation_delta": max_geometry_deltas["triangle_mean_violation"],
        "max_abs_triangle_max_violation_delta": max_geometry_deltas["triangle_max_violation"],
        "passed": not failure_rows,
    }
    return summary_row, failure_rows


def build_runtime_summary_rows(mode_results):
    all_timing_rows = []
    for result in mode_results.values():
        all_timing_rows.extend(result["timing_rows"])
    stage_totals, fold_layer_counts = summarize_stage_times(all_timing_rows)

    summary_rows = []
    for mode in VALID_MODES:
        if mode not in mode_results:
            continue
        row = {
            "mode": mode,
            "total_seconds": mode_results[mode]["mode_total_seconds"],
            "mean_seconds_per_fold_layer": (
                mode_results[mode]["mode_total_seconds"] / fold_layer_counts[mode]
                if fold_layer_counts[mode] else 0.0
            ),
            "n_fold_layer_runs": fold_layer_counts[mode],
            "percent_of_reference": "",
        }
        for col in STAGE_COLUMNS:
            stage_name = col[:-6]
            row[col] = stage_totals[mode].get(stage_name, 0.0)
        summary_rows.append(row)

    derived_specs = (
        ("cache_only_savings", "baseline", "layer_cache"),
    )
    for label, reference, target in derived_specs:
        if reference in mode_results and target in mode_results:
            saved = mode_results[reference]["mode_total_seconds"] - mode_results[target]["mode_total_seconds"]
            row = {
                "mode": label,
                "total_seconds": saved,
                "mean_seconds_per_fold_layer": "",
                "n_fold_layer_runs": "",
                "percent_of_reference": (
                    (saved / mode_results[reference]["mode_total_seconds"]) * 100.0
                    if mode_results[reference]["mode_total_seconds"] else 0.0
                ),
            }
            for col in STAGE_COLUMNS:
                row[col] = ""
            summary_rows.append(row)
    return summary_rows, all_timing_rows


def write_runtime_stage_breakdown(csv_path, timing_rows):
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["mode", "fold", "layer", "stage", "seconds"],
        )
        writer.writeheader()
        for row in timing_rows:
            writer.writerow(
                {
                    "mode": row["mode"],
                    "fold": row["fold"],
                    "layer": row["layer"],
                    "stage": row["stage"],
                    "seconds": format_seconds(row["seconds"]),
                }
            )


def write_runtime_mode_summary(csv_path, summary_rows):
    fieldnames = [
        "mode",
        "total_seconds",
        "mean_seconds_per_fold_layer",
        "n_fold_layer_runs",
        *STAGE_COLUMNS,
        "percent_of_reference",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            out_row = {}
            for field in fieldnames:
                value = row.get(field, "")
                if isinstance(value, float):
                    out_row[field] = format_seconds(value)
                else:
                    out_row[field] = value
            writer.writerow(out_row)


def write_validation_regression_summary(csv_path, summary_rows):
    fieldnames = [
        "comparison",
        "max_abs_metric_delta",
        "max_abs_intercept_delta",
        "max_abs_coef_delta",
        "max_abs_pred_delta",
        "mean_abs_pred_delta",
        "passed",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            writer.writerow(
                {
                    "comparison": row["comparison"],
                    "max_abs_metric_delta": format_seconds(row["max_abs_metric_delta"]),
                    "max_abs_intercept_delta": format_seconds(row["max_abs_intercept_delta"]),
                    "max_abs_coef_delta": format_seconds(row["max_abs_coef_delta"]),
                    "max_abs_pred_delta": format_seconds(row["max_abs_pred_delta"]),
                    "mean_abs_pred_delta": format_seconds(row["mean_abs_pred_delta"]),
                    "passed": str(bool(row["passed"])),
                }
            )


def write_validation_geometry_summary(csv_path, summary_rows):
    fieldnames = [
        "comparison",
        "max_abs_rmsd_delta",
        "max_abs_tm_score_delta",
        "max_abs_frac_top3_eigenvalues_delta",
        "max_abs_n_negative_eigenvalues_delta",
        "max_abs_neg_eigenvalue_fraction_delta",
        "max_abs_n_negative_predictions_delta",
        "max_abs_stress_delta",
        "max_abs_triangle_violation_fraction_delta",
        "max_abs_triangle_mean_violation_delta",
        "max_abs_triangle_max_violation_delta",
        "passed",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in summary_rows:
            out_row = {"comparison": row["comparison"], "passed": str(bool(row["passed"]))}
            for field in fieldnames:
                if field in ("comparison", "passed"):
                    continue
                out_row[field] = format_seconds(float(row[field]))
            writer.writerow(out_row)


def write_validation_failures(csv_path, failure_rows):
    fieldnames = [
        "comparison",
        "kind",
        "fold",
        "layer",
        "protein_id",
        "metric",
        "observed_delta",
        "tolerance",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in failure_rows:
            writer.writerow(
                {
                    "comparison": row["comparison"],
                    "kind": row["kind"],
                    "fold": row["fold"],
                    "layer": row["layer"],
                    "protein_id": row["protein_id"],
                    "metric": row["metric"],
                    "observed_delta": format_seconds(float(row["observed_delta"])),
                    "tolerance": format_seconds(float(row["tolerance"])),
                }
            )


def write_geometry_optimization_validation(csv_path, summary_row):
    fieldnames = [
        "source_mode",
        "sampled_cases",
        "max_abs_rmsd_delta",
        "max_abs_tm_score_delta",
        "max_abs_frac_top3_eigenvalues_delta",
        "max_abs_n_negative_eigenvalues_delta",
        "max_abs_neg_eigenvalue_fraction_delta",
        "max_abs_n_negative_predictions_delta",
        "max_abs_stress_delta",
        "max_abs_triangle_violation_fraction_delta",
        "max_abs_triangle_mean_violation_delta",
        "max_abs_triangle_max_violation_delta",
        "passed",
    ]
    with open(csv_path, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        out_row = {
            "source_mode": summary_row["source_mode"],
            "sampled_cases": summary_row["sampled_cases"],
            "passed": str(bool(summary_row["passed"])),
        }
        for field in fieldnames:
            if field in ("source_mode", "sampled_cases", "passed"):
                continue
            out_row[field] = format_seconds(float(summary_row[field]))
        writer.writerow(out_row)


def write_run_summary(
    summary_path,
    args,
    protein_data,
    skipped,
    folds,
    mode_results,
    regression_summaries,
    geometry_summaries,
    geometry_optimization_summary,
    runtime_summary_rows,
):
    savings_lookup = {
        row["mode"]: row
        for row in runtime_summary_rows
        if row["mode"] == "cache_only_savings"
    }

    with open(summary_path, "w") as handle:
        handle.write("K-fold MDS Runtime Benchmark\n")
        handle.write("=" * 30 + "\n\n")
        handle.write(f"Timestamp: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
        handle.write(f"Args: {' '.join(sys.argv[1:])}\n")
        handle.write(f"Modes: {', '.join(args.modes)}\n")
        handle.write(
            f"Loaded proteins ({len(protein_data)}): "
            f"{', '.join(d['protein_id'] for d in protein_data)}\n"
        )
        handle.write(f"Skipped proteins: {len(skipped)}\n")
        if skipped:
            for protein_id, reason in skipped:
                handle.write(f"  - {protein_id}: {reason}\n")

        handle.write("\nFold breakdown:\n")
        for fold_idx, (train_idx, test_idx) in enumerate(folds):
            train_ids = [protein_data[i]["protein_id"] for i in train_idx]
            test_ids = [protein_data[i]["protein_id"] for i in test_idx]
            handle.write(
                f"  Fold {fold_idx}: train={', '.join(train_ids)} | "
                f"test={', '.join(test_ids)}\n"
            )

        handle.write("\nMode runtimes:\n")
        for mode in VALID_MODES:
            if mode in mode_results:
                handle.write(
                    f"  {mode}: {mode_results[mode]['mode_total_seconds']:.3f}s\n"
                )

        handle.write("\nSavings:\n")
        for label in ("cache_only_savings",):
            if label in savings_lookup:
                row = savings_lookup[label]
                handle.write(
                    f"  {label}: {float(row['total_seconds']):.3f}s "
                    f"({float(row['percent_of_reference']):.2f}% of reference)\n"
                )

        handle.write("\nValidation summary:\n")
        for row in regression_summaries:
            handle.write(
                f"  Regression {row['comparison']}: "
                f"{'PASS' if row['passed'] else 'FAIL'}\n"
            )
        for row in geometry_summaries:
            handle.write(
                f"  Geometry {row['comparison']}: "
                f"{'PASS' if row['passed'] else 'FAIL'}\n"
            )
        handle.write(
            "  Geometry optimization sample "
            f"({geometry_optimization_summary['sampled_cases']} cases from "
            f"{geometry_optimization_summary['source_mode']}): "
            f"{'PASS' if geometry_optimization_summary['passed'] else 'FAIL'}\n"
        )

        handle.write("\nNotes:\n")
        handle.write("  - All compared runs reused the same folds.\n")
        handle.write("  - Geometry runs on all proteins in each fold (train + test).\n")
        handle.write(
            "  - A 5% sampled legacy-vs-optimized geometry equivalence check was run.\n"
        )
        handle.write(
            "  - Run this script with ./.venv/bin/python, not system python3.\n"
        )


def main():
    args = parse_args()

    output_dir = OUTPUT_ROOT / args.output_subdir
    csv_dir = output_dir / "csv_files"
    csv_dir.mkdir(parents=True, exist_ok=True)

    protein_data, skipped = load_protein_data(args.max_proteins)
    if not protein_data:
        print("ERROR: No proteins loaded successfully.")
        sys.exit(1)

    folds = build_folds(protein_data, args.n_folds)
    total_pairs = sum(d["n_pairs"] for d in protein_data)
    print(f"\nLoaded {len(protein_data)} proteins with {total_pairs:,} total pairs.")
    print(f"Using {args.n_folds}-fold CV across {args.max_layers} layer(s).")
    for fold_idx, (train_idx, test_idx) in enumerate(folds):
        n_train_pairs = sum(protein_data[i]["n_pairs"] for i in train_idx)
        n_test_pairs = sum(protein_data[i]["n_pairs"] for i in test_idx)
        print(
            f"  Fold {fold_idx}: {len(train_idx)} train / {len(test_idx)} test proteins, "
            f"{n_train_pairs:,} train / {n_test_pairs:,} test pairs"
        )

    runners = {
        "baseline": run_mode_baseline,
        "layer_cache": run_mode_layer_cache,
    }
    mode_results = {}

    for mode in args.modes:
        try:
            mode_results[mode] = runners[mode](
                protein_data,
                folds,
                args.max_layers,
                args.geometry_workers,
            )
        except Exception:
            print(f"\nERROR while running mode {mode}:")
            traceback.print_exc()
            sys.exit(1)

    regression_summaries = []
    geometry_summaries = []
    failure_rows = []

    comparisons = []
    if "baseline" in mode_results and "layer_cache" in mode_results:
        comparisons.append(("baseline", "layer_cache", "baseline_vs_layer_cache"))

    print("\nValidation:")
    for mode_a, mode_b, name in comparisons:
        regression_summary, geometry_summary, failures = validate_modes(
            mode_results[mode_a],
            mode_results[mode_b],
            name,
        )
        regression_summaries.append(regression_summary)
        geometry_summaries.append(geometry_summary)
        failure_rows.extend(failures)
        print(
            f"  {name}: regression={'PASS' if regression_summary['passed'] else 'FAIL'}, "
            f"geometry={'PASS' if geometry_summary['passed'] else 'FAIL'}"
        )

    protein_records_by_id = {d["protein_id"]: d for d in protein_data}
    geometry_opt_source_mode = "layer_cache" if "layer_cache" in mode_results else "baseline"
    geometry_optimization_summary, geometry_opt_failures = (
        run_sampled_geometry_optimization_validation(
            geometry_opt_source_mode,
            mode_results[geometry_opt_source_mode],
            protein_records_by_id,
        )
    )
    failure_rows.extend(geometry_opt_failures)
    print(
        "  legacy_vs_optimized_geometry_sampled: "
        f"{'PASS' if geometry_optimization_summary['passed'] else 'FAIL'} "
        f"({geometry_optimization_summary['sampled_cases']} sampled cases)"
    )

    runtime_summary_rows, all_timing_rows = build_runtime_summary_rows(mode_results)

    for label in ("cache_only_savings",):
        row = next((r for r in runtime_summary_rows if r["mode"] == label), None)
        if row is not None:
            print(
                f"  {label}: {float(row['total_seconds']):.3f}s "
                f"({float(row['percent_of_reference']):.2f}%)"
            )

    write_runtime_stage_breakdown(csv_dir / "runtime_stage_breakdown.csv", all_timing_rows)
    write_runtime_mode_summary(csv_dir / "runtime_mode_summary.csv", runtime_summary_rows)
    write_validation_regression_summary(
        csv_dir / "validation_regression_summary.csv",
        regression_summaries,
    )
    write_validation_geometry_summary(
        csv_dir / "validation_geometry_summary.csv",
        geometry_summaries,
    )
    write_geometry_optimization_validation(
        csv_dir / "geometry_optimization_validation.csv",
        geometry_optimization_summary,
    )
    write_validation_failures(csv_dir / "validation_failures.csv", failure_rows)
    write_run_summary(
        output_dir / "run_summary.txt",
        args,
        protein_data,
        skipped,
        folds,
        mode_results,
        regression_summaries,
        geometry_summaries,
        geometry_optimization_summary,
        runtime_summary_rows,
    )

    print(f"\nWrote benchmark outputs to {output_dir}")
    if failure_rows:
        print(f"Validation failures recorded: {len(failure_rows)}")
    else:
        print("All validations passed.")


if __name__ == "__main__":
    main()

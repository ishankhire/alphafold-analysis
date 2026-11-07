#!/usr/bin/env python3
"""Per-protein signed error by sequence separation and final structure distance.

Covers the 154 CAMEO proteins and 48 layers, stored sparsely. ECOD is left out because
its final structures are currently unavailable. Reads the existing packed OOF predictions
and PDBs; nothing is fitted or predicted, and neither probes nor pair tensors are opened.
Self-check: each protein's fine bins must reproduce its stored whole-protein metrics at
all 48 layers and its canonical subgroup rows at v1's 10 layers.
"""
from __future__ import annotations

import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import build_strata as bs  # noqa: E402
from canonical_storage import csv_root  # noqa: E402
from regression_output_helpers import atomic_savez_compressed  # noqa: E402
from strata_bins import CANONICAL_DIST, CANONICAL_SEQ, DIST_EDGES, N_D, OLS_PROBE, SEQ_EDGES  # noqa: E402

DATA = bs.DATA
SHARDS = csv_root() / "diagnostic_experiments/cross_layer_probe_transfer_v1/layer_shards"
PUBLISHED_SUBGROUPS = (csv_root() / "regression_5fold_v1/analysis/publication_subgroup_mae_5fold_v1"
                       / "per_protein_subgroup_mae.csv")
V1_LAYERS = (22, 23, 24, 25, 26, 43, 44, 45, 46, 47)
N_LAYERS, N_PROTEINS, TOLERANCE = bs.N_LAYERS, 154, 1e-5  # stored predictions are float32
STARTED = time.perf_counter()


def log(message):
    print(f"[{time.perf_counter() - STARTED:7.1f}s] {message}", flush=True)


def scan(meta):
    cell_index, cell_n, blocks, offsets = [], [], [], [0]
    read_seconds = bin_seconds = 0.0
    with ThreadPoolExecutor(bs.THREADS) as pool:
        for p, row in meta.iterrows():
            ca = bs.helpers.extract_ca_coordinates(bs.helpers.resolve_combined_pdb_path(row.protein_id, row.source))
            coords = np.asarray(ca["coordinates"], dtype=np.float64)
            chains = tuple(str(residue[0]) for residue in ca["residues"])
            if coords.shape[0] != row.n_residues:
                raise ValueError(f"{row.protein_id}: residue mismatch")
            i, j = bs.strict_lower_triangle_indices(int(row.n_residues))
            if np.any(bs.sequence_separation_bin_codes(i, j, chain_ids=chains) < 0):
                raise ValueError(f"{row.protein_id}: interchain pairs are not handled here")
            y = bs.helpers.compute_distance_matrix(coords)[i, j]
            seq = np.searchsorted(SEQ_EDGES, np.abs(i - j), side="right") - 1
            dist = np.searchsorted(DIST_EDGES, y, side="right") - 1
            present, compact, counts = np.unique(seq * N_D + dist, return_inverse=True, return_counts=True)
            tick = time.perf_counter()
            predictions = list(pool.map(lambda layer: bs.read_prediction(row.protein_id, layer), range(N_LAYERS)))
            read_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            block = np.empty((present.size, N_LAYERS, 3))
            for layer, pred in enumerate(predictions):
                if pred.shape != y.shape:
                    raise ValueError(f"{row.protein_id} layer {layer}: shape {pred.shape}")
                err = pred.astype(np.float64) - y
                for k, values in enumerate((err, np.abs(err), err * err)):
                    block[:, layer, k] = np.bincount(compact, weights=values, minlength=present.size)
            bin_seconds += time.perf_counter() - tick
            cell_index.append(present), cell_n.append(counts), blocks.append(block)
            offsets.append(offsets[-1] + present.size)
            if p % 100 == 0:
                log(f"protein {p + 1}/{len(meta)}")
    log(f"scan done; read {read_seconds:.0f}s, binning {bin_seconds:.0f}s")
    return (np.concatenate(cell_index), np.concatenate(cell_n), np.asarray(offsets), np.concatenate(blocks))


def summarize(rows, expected):
    worst = np.array(rows).max(axis=0) if rows else np.full(4, np.inf)
    return {"rows_compared": len(rows), "expected_rows": expected, "max_abs_diff_n": float(worst[0]),
            "max_abs_diff_mae": float(worst[1]), "max_abs_diff_rmse": float(worst[2]),
            "max_abs_diff_bias": float(worst[3])}


def check_export(meta, cell_n, offsets, sums):
    """Summing a protein's fine bins must give its stored whole-protein metrics."""
    index = {pid: k for k, pid in enumerate(meta.protein_id)}
    stored = pd.read_csv(bs.EXPORT / "validation.csv", usecols=[
        "protein_id", "analysis_name", "layer", "n_predicted_pairs", "stored_mae", "stored_rmse",
        "stored_bias_pred_minus_true"], dtype={"protein_id": str})
    stored = stored[stored.analysis_name.eq(bs.PROBE) & stored.protein_id.isin(index)]
    rows = []
    for r in stored.itertuples():
        k = index[r.protein_id]
        span = slice(offsets[k], offsets[k + 1])
        n = cell_n[span].sum()
        e, a, q = sums[span][:, int(r.layer)].sum(axis=0)
        rows.append((abs(n - r.n_predicted_pairs), abs(a / n - r.stored_mae), abs((q / n) ** .5 - r.stored_rmse),
                     abs(e / n - r.stored_bias_pred_minus_true)))
    return summarize(rows, N_LAYERS * len(meta))


def check_v1_categories(meta, cell_index, cell_n, offsets, sums):
    """Summing fine bins to canonical bins must match each protein's published v1 subgroup row."""
    index = {pid: k for k, pid in enumerate(meta.protein_id)}
    columns = ["target_layer", "source_layer", "transfer_mode", "predictor", "protein_id", "category_scope",
               "category_label", "n", "sum_error", "sum_abs_error", "sum_squared_error"]
    rows = []
    for layer in V1_LAYERS:
        shard = pd.read_csv(SHARDS / f"cross_layer_primitives_target_layer_{layer:02d}.csv", usecols=columns,
                            dtype={"protein_id": str, "category_label": str})
        shard = shard[shard.predictor.eq("probe") & shard.source_layer.eq(layer) & shard.transfer_mode.eq("raw")
                      & shard.protein_id.isin(index) & shard.category_scope.ne("all_pairs")]
        for r in shard.itertuples():
            k = index[r.protein_id]
            span = slice(offsets[k], offsets[k + 1])
            by_sequence = r.category_scope == "sequence_separation"
            coordinate = cell_index[span] // N_D if by_sequence else cell_index[span] % N_D
            keep = np.isin(coordinate, (CANONICAL_SEQ if by_sequence else CANONICAL_DIST)[r.category_label])
            n = cell_n[span][keep].sum()
            if n != r.n:
                rows.append((abs(n - r.n), np.inf, np.inf, np.inf))
            elif n == 0:
                rows.append((0, 0, 0, 0))
            else:
                e, a, q = sums[span][keep, layer].sum(axis=0)
                rows.append((0, abs(a - r.sum_abs_error) / n, abs((q / n) ** .5 - (r.sum_squared_error / n) ** .5),
                             abs(e - r.sum_error) / n))
    return summarize(rows, len(meta) * len(V1_LAYERS) * 11)


def check_published_subgroups(meta, cell_index, cell_n, offsets, sums):
    """Summing fine bins to canonical bins must match each protein's published subgroup MAE at every layer."""
    index = {pid: k for k, pid in enumerate(meta.protein_id)}
    table = pd.read_csv(PUBLISHED_SUBGROUPS, usecols=[
        "analysis_name", "layer", "protein_id", "category_scope", "category_label", "n_pairs", "sum_absolute_error"],
        dtype={"protein_id": str, "category_label": str})
    table = table[table.analysis_name.eq(bs.PROBE) & table.protein_id.isin(index)]
    rows = []
    for r in table.itertuples():
        k = index[r.protein_id]
        span = slice(offsets[k], offsets[k + 1])
        by_sequence = r.category_scope == "sequence_separation"
        coordinate = cell_index[span] // N_D if by_sequence else cell_index[span] % N_D
        keep = np.isin(coordinate, (CANONICAL_SEQ if by_sequence else CANONICAL_DIST)[r.category_label])
        n = cell_n[span][keep].sum()
        if n != r.n_pairs:
            rows.append((abs(n - r.n_pairs), np.inf))
        elif n == 0:
            rows.append((0, 0))
        else:
            rows.append((0, abs(sums[span][keep, int(r.layer), 1].sum() - r.sum_absolute_error) / n))
    worst = np.array(rows).max(axis=0) if rows else np.full(2, np.inf)
    return {"rows_compared": len(rows), "expected_rows": len(meta) * N_LAYERS * 11,
            "max_abs_diff_n": float(worst[0]), "max_abs_diff_mae": float(worst[1])}


def largest_metric_difference(check):
    return max(value for key, value in check.items() if key.startswith("max_abs_diff_") and key != "max_abs_diff_n")


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(bs.EXPORT / "metadata/proteins.csv", dtype={"protein_id": str})
    meta = meta[meta.source.eq("cameo")].reset_index(drop=True)
    if len(meta) != N_PROTEINS:
        raise ValueError(f"Expected {N_PROTEINS} CAMEO proteins, found {len(meta)}")
    cell_index, cell_n, offsets, sums = scan(meta)
    log(f"{cell_index.size} protein-bin rows")
    checks = {"stored_whole_protein": check_export(meta, cell_n, offsets, sums)}
    if bs.PROBE == OLS_PROBE:
        checks["v1_canonical_subgroups"] = check_v1_categories(meta, cell_index, cell_n, offsets, sums)
    else:  # v1's signed shards exist only for shared OLS
        checks["published_subgroup_mae"] = check_published_subgroups(meta, cell_index, cell_n, offsets, sums)
    passed = all(c["rows_compared"] == c["expected_rows"] and c["max_abs_diff_n"] == 0
                 and largest_metric_difference(c) <= TOLERANCE for c in checks.values())
    check = {"passed": passed, "tolerance_angstrom": TOLERANCE, "proteins": len(meta), "checks": checks}
    (DATA / "protein_strata_cameo_check.json").write_text(json.dumps(check, indent=2))
    log(f"checks: {json.dumps(check)}")
    atomic_savez_compressed(
        DATA / "protein_strata_cameo.npz", cell_index=cell_index.astype(np.uint8), cell_n=cell_n.astype(np.uint32),
        offsets=offsets, sums=sums, protein_ids=meta.protein_id.to_numpy().astype(str),
        source=meta.source.to_numpy().astype(str), n_residues=meta.n_residues.to_numpy(),
        outer_fold=meta.outer_fold.to_numpy(), n_pairs=meta.n_pairs.to_numpy())
    log("wrote protein_strata_cameo.npz")
    sys.exit(0 if check["passed"] else 1)


if __name__ == "__main__":
    main()

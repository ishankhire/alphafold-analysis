#!/usr/bin/env python3
"""Native shared-OLS signed error by fine sequence separation x final structure distance.

Reads the existing packed OOF predictions for all 48 layers and the existing PDBs.
Nothing is fitted or predicted, and neither probes nor pair tensors are opened.
Writes native_error_strata.csv only after the bins reconcile with validation.csv
and with the published canonical subgroup metrics at every layer.
"""
from __future__ import annotations

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

SCRIPT_DIR = Path(__file__).resolve().parent
for path in (SCRIPT_DIR, SCRIPT_DIR.parent):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))
import cluster_regression_helpers as helpers  # noqa: E402
from canonical_storage import csv_root  # noqa: E402
from regression_output_helpers import atomic_savez_compressed  # noqa: E402
from regression_protocol import sequence_separation_bin_codes, strict_lower_triangle_indices  # noqa: E402
from strata_bins import (  # noqa: E402
    CANONICAL_DIST, CANONICAL_SEQ, DIST_EDGES, DIST_LABELS, N_D, N_S, SEQ_EDGES, SEQ_LABELS, output_root,
)

ECOD = os.environ.get("CROSS_LAYER_ECOD_EXPERIMENTS_ROOT")
if ECOD:
    helpers.ECOD_PAIR_BLOCKS_ROOT = Path(ECOD) / "representation_saves"
    helpers.ECOD_PDB_ROOT = Path(ECOD) / "openfold_reps_outputs_sm"
PRODUCER = csv_root() / "regression_5fold_v1"
EXPORT = PRODUCER / "predicted_distances_5fold_v1"
SUBGROUPS = PRODUCER / "shared_ols_base256/subgroup_metrics"
DATA = output_root() / "data"
PROBE, OUTLIER, N_LAYERS, TOLERANCE = "shared_ols_base256", "7stz_C", 48, 1e-3
THREADS = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
STARTED = time.perf_counter()


def log(message):
    print(f"[{time.perf_counter() - STARTED:7.1f}s] {message}", flush=True)


def read_prediction(protein_id, layer):
    return np.load(EXPORT / f"layers/layer_{layer:02d}/{protein_id}/{PROBE}.npy")


def scan(meta):
    strata_n = np.zeros((N_S, N_D), np.int64)
    strata_n_outlier = np.zeros((N_S, N_D), np.int64)
    n_proteins = np.zeros((N_S, N_D), np.int64)
    strata_sum = np.zeros((N_LAYERS, N_S, N_D, 3))
    strata_sum_outlier = np.zeros((N_LAYERS, N_S, N_D, 3))
    protein_sum = np.zeros((N_LAYERS, len(meta), 3))
    protein_n = np.zeros(len(meta), np.int64)
    extremes = np.tile([np.inf, -np.inf], (N_LAYERS, 1))
    read_seconds = bin_seconds = 0.0
    with ThreadPoolExecutor(THREADS) as pool:
        for p, row in meta.iterrows():
            ca = helpers.extract_ca_coordinates(helpers.resolve_combined_pdb_path(row.protein_id, row.source))
            coords = np.asarray(ca["coordinates"], dtype=np.float64)
            chains = tuple(str(residue[0]) for residue in ca["residues"])
            if coords.shape[0] != row.n_residues:
                raise ValueError(f"{row.protein_id}: residue mismatch")
            i, j = strict_lower_triangle_indices(int(row.n_residues))
            if np.any(sequence_separation_bin_codes(i, j, chain_ids=chains) < 0):
                raise ValueError(f"{row.protein_id}: interchain pairs are not handled here")
            y = helpers.compute_distance_matrix(coords)[i, j]
            seq = np.searchsorted(SEQ_EDGES, np.abs(i - j), side="right") - 1
            dist = np.searchsorted(DIST_EDGES, y, side="right") - 1
            cell = seq * N_D + dist
            n_cell = np.bincount(cell, minlength=N_S * N_D).reshape(N_S, N_D)
            strata_n += n_cell
            n_proteins += n_cell > 0
            protein_n[p] = i.size
            is_outlier = row.protein_id == OUTLIER
            if is_outlier:
                strata_n_outlier += n_cell
            tick = time.perf_counter()
            predictions = list(pool.map(lambda layer: read_prediction(row.protein_id, layer), range(N_LAYERS)))
            read_seconds += time.perf_counter() - tick
            tick = time.perf_counter()
            for layer, pred in enumerate(predictions):
                if pred.shape != y.shape:
                    raise ValueError(f"{row.protein_id} layer {layer}: shape {pred.shape}")
                err = pred.astype(np.float64) - y
                for k, values in enumerate((err, np.abs(err), err * err)):
                    summed = np.bincount(cell, weights=values, minlength=N_S * N_D).reshape(N_S, N_D)
                    strata_sum[layer, :, :, k] += summed
                    protein_sum[layer, p, k] = values.sum()
                    if is_outlier:
                        strata_sum_outlier[layer, :, :, k] += summed
                extremes[layer] = min(extremes[layer, 0], err.min()), max(extremes[layer, 1], err.max())
            bin_seconds += time.perf_counter() - tick
            if p % 100 == 0:
                log(f"protein {p + 1}/{len(meta)}")
    log(f"scan done; read {read_seconds:.0f}s, binning {bin_seconds:.0f}s")
    return dict(strata_n=strata_n, strata_n_outlier=strata_n_outlier, n_proteins=n_proteins, strata_sum=strata_sum,
                strata_sum_outlier=strata_sum_outlier, protein_sum=protein_sum, protein_n=protein_n, extremes=extremes)


def check_export(arrays, meta):
    index = {pid: k for k, pid in enumerate(meta.protein_id)}
    stored = pd.read_csv(EXPORT / "validation.csv", usecols=[
        "protein_id", "analysis_name", "layer", "n_predicted_pairs", "stored_mae", "stored_rmse",
        "stored_bias_pred_minus_true"], dtype={"protein_id": str})
    stored = stored[stored.analysis_name.eq(PROBE)]
    diffs = np.zeros(4)
    for r in stored.itertuples():
        k, layer = index[r.protein_id], int(r.layer)
        n = arrays["protein_n"][k]
        e, a, s = arrays["protein_sum"][layer, k]
        row = (abs(n - r.n_predicted_pairs), abs(a / n - r.stored_mae), abs((s / n) ** .5 - r.stored_rmse),
               abs(e / n - r.stored_bias_pred_minus_true))
        diffs = np.maximum(diffs, row)
    return {"rows_compared": len(stored), "expected_rows": N_LAYERS * len(meta), "max_abs_diff_n": float(diffs[0]),
            "max_abs_diff_mae": float(diffs[1]), "max_abs_diff_rmse": float(diffs[2]),
            "max_abs_diff_bias": float(diffs[3])}


def check_canonical(arrays, filename, groups, axis):
    """Sum fine bins back to canonical bins and compare with the published CSV."""
    table = pd.read_csv(SUBGROUPS / filename, dtype={"category_label": str})
    table = table[table.analysis_name.eq(PROBE)]
    diffs = np.zeros(4)
    for r in table.itertuples():
        layer, group = int(r.layer), list(groups[r.category_label])
        pick = (lambda a: a[..., group, :, :]) if axis == "sequence" else (lambda a: a[..., :, group, :])
        n = (arrays["strata_n"][group, :] if axis == "sequence" else arrays["strata_n"][:, group]).sum()
        e, a, s = pick(arrays["strata_sum"][layer]).sum(axis=(0, 1))
        row = (abs(n - r.n_evaluated_pairs), abs(a / n - r.pooled_mae), abs((s / n) ** .5 - r.pooled_rmse),
               abs(e / n - r.pooled_bias_pred_minus_true))
        diffs = np.maximum(diffs, row)
    return {"rows_compared": len(table), "expected_rows": N_LAYERS * len(groups), "max_abs_diff_n": float(diffs[0]),
            "max_abs_diff_mae": float(diffs[1]), "max_abs_diff_rmse": float(diffs[2]),
            "max_abs_diff_bias": float(diffs[3])}


def support_status(n_pairs, n_proteins):
    """Same labels and thresholds as the canonical subgroup tables."""
    if n_pairs == 0:
        return "empty"
    if n_pairs < 100 and n_proteins < 5:
        return "low_pair_and_protein_support"
    if n_pairs < 100:
        return "low_pair_support"
    if n_proteins < 5:
        return "low_protein_support"
    return "sufficient"


def tidy_rows(arrays):
    rows = []
    outlier_cells = arrays["strata_n_outlier"] > 0
    scopes = {
        "all_552": (arrays["strata_n"], arrays["n_proteins"], arrays["strata_sum"]),
        "excluding_7stz_C": (arrays["strata_n"] - arrays["strata_n_outlier"], arrays["n_proteins"] - outlier_cells,
                             arrays["strata_sum"] - arrays["strata_sum_outlier"]),
    }
    for scope, (n_pairs, n_prot, sums) in scopes.items():
        for layer in range(N_LAYERS):
            for s in range(N_S):
                for d in range(N_D):
                    n = int(n_pairs[s, d])
                    e, a, q = sums[layer, s, d]
                    mean = e / n if n else float("nan")
                    rows.append({
                        "layer": layer, "cohort_scope": scope, "sequence_separation": SEQ_LABELS[s],
                        "final_structure_distance": DIST_LABELS[d], "n_pairs": n, "n_proteins": int(n_prot[s, d]),
                        "support_status": support_status(n, int(n_prot[s, d])),
                        "mean_signed_error": mean,
                        "sd_signed_error": float(np.sqrt(max(q / n - mean * mean, 0.0))) if n else float("nan"),
                        "mae": a / n if n else float("nan"), "rmse": float(np.sqrt(q / n)) if n else float("nan"),
                    })
    return pd.DataFrame(rows)


def main():
    DATA.mkdir(parents=True, exist_ok=True)
    meta = pd.read_csv(EXPORT / "metadata/proteins.csv", dtype={"protein_id": str})
    if len(meta) != 552:
        raise ValueError(f"Expected the 552-protein cohort, found {len(meta)}")
    arrays = scan(meta)
    atomic_savez_compressed(DATA / "native_strata.npz", **arrays, protein_ids=meta.protein_id.to_numpy().astype(str))
    log("saved native_strata.npz")
    checks = {
        "export_validation": check_export(arrays, meta),
        "canonical_by_sequence_separation": check_canonical(arrays, "by_sequence_separation.csv", CANONICAL_SEQ, "sequence"),
        "canonical_by_spatial_distance": check_canonical(arrays, "by_spatial_distance.csv", CANONICAL_DIST, "distance"),
    }
    passed = all(c["rows_compared"] == c["expected_rows"] and c["max_abs_diff_n"] == 0
                 and max(c["max_abs_diff_mae"], c["max_abs_diff_rmse"], c["max_abs_diff_bias"]) <= TOLERANCE
                 for c in checks.values())
    summary = {"passed": passed, "tolerance_angstrom": TOLERANCE, "checks": checks,
               "error_min": float(arrays["extremes"][:, 0].min()), "error_max": float(arrays["extremes"][:, 1].max()),
               "pairs": int(arrays["protein_n"].sum()), "total_seconds": time.perf_counter() - STARTED}
    (DATA / "checks.json").write_text(json.dumps(summary, indent=2))
    log(f"checks passed={passed}: {json.dumps(checks)}")
    if not passed:
        sys.exit(1)
    temporary = DATA / ".native_error_strata.csv.tmp"
    tidy_rows(arrays).to_csv(temporary, index=False, float_format="%.6f")
    os.replace(temporary, DATA / "native_error_strata.csv")
    log("wrote native_error_strata.csv")


if __name__ == "__main__":
    main()

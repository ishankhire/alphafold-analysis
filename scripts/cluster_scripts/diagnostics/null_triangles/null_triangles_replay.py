#!/usr/bin/env python3
"""Null models for triangle violations: one layer, four probes, 154 CAMEO proteins.

Task i scores layer i. For each probe and protein it scores:

- the observed OOF prediction on every triple (exact) and on the protein's fixed
  triangle sample (all triples if C(R,3) <= 1M, else 1M distinct triples);
- Null A (Gaussian per bin), Null B (residuals permuted within bin), and Null C
  (bin mean and per-residue offsets kept, remainder permuted within bin) on B10;
- Nulls B and C on B10 x the six sequence-separation bins;
- for shared OLS only, Null B on B5 and B20;

each with ``N_DRAWS`` draws, on the triangle sample. Pass 1 collects pair-level
error diagnostics and the cohort statistics Null A falls back to; pass 2 scores
triangles. No model is fitted or replayed: inputs are the cached OOF predictions
and the final OpenFold structures.

Gates before anything is written:
- the slim reducer equals ``compute_metric_rows`` bit for bit on every protein
  whose triangle sample is every triple (same float32-loaded inputs);
- exact observed metrics match the paper's stored per-protein values within the
  float32-cache tolerance (the paper replayed float64 predictions);
- Null B keeps every bin's residual histogram on every draw.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import math
import multiprocessing as mp
import os
import resource
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SCRIPT_DIR, SCRIPT_DIR.parents[1], SCRIPT_DIR.parents[1] / "triangle_violation_subgroups"):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cluster_regression_helpers as helpers  # noqa: E402
import null_triangles_common as common  # noqa: E402
from canonical_storage import csv_root  # noqa: E402
from regression_output_helpers import sha256_file  # noqa: E402
from regression_protocol import sequence_separation_bin_codes, strict_lower_triangle_indices  # noqa: E402
from triangle_violation_subgroup_common import build_reference_cache, compute_metric_rows  # noqa: E402

PRODUCER = csv_root() / "regression_5fold_v1"
EXPORT = Path(os.environ.get("NULL_TRIANGLES_EXPORT_ROOT", PRODUCER / "predicted_distances_5fold_v1"))
OUTPUT_ROOT = Path(os.environ.get(
    "NULL_TRIANGLES_OUTPUT_ROOT", csv_root() / "diagnostic_experiments" / common.EXPERIMENT_NAME))
COHORT_CSV = PRODUCER / "metadata" / "regression_outer_folds_v1.csv"
PARITY_CSV = SCRIPT_DIR / "paper_triangle_parity_cameo.csv"
PARITY_KEYS = ("n_triangles", "n_violations", "violation_sum_angstrom", "relative_violation_sum")
PAPER_FRACTION_TOL = 5e-5          # Exp 7's gate for exact proteins
PAPER_RELATIVE_SUM_RTOL = 1e-4
PAPER_NEGATIVE_TOL = 2
STARTED = time.perf_counter()

METRIC_COLUMNS = (
    "probe", "layer", "protein_id", "n_residues", "model", "binning", "triangle_set", "draw",
    "n_triangles", "n_violations", "violation_sum_angstrom", "relative_violation_sum",
    "violation_fraction", "mean_relative_violation_all", "n_violations_with_clipped_side",
    "n_negative_values",
)
DIAGNOSTIC_COLUMNS = (
    "probe", "layer", "protein_id", "binning", "cell", "cell_label", "n_pairs", "mean_r", "sd_r",
    "rmse", "skew", "excess_kurtosis", "null_a_fallback",
)
CALIBRATION_COLUMNS = (
    "probe", "layer", "protein_id", "n_residues", "n_pairs", "alpha", "beta", "mean_r", "rmse",
    "n_negative_predictions",
)


def log(message: str) -> None:
    scale = 1024 ** 2 if sys.platform == "darwin" else 1024
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale
    child = resource.getrusage(resource.RUSAGE_CHILDREN).ru_maxrss / scale
    print(f"[{time.strftime('%Y-%m-%dT%H:%M:%S')}] [{time.perf_counter() - STARTED:8.1f}s] "
          f"[peak_rss={rss:.0f}MiB worker_peak_rss={child:.0f}MiB] {message}", flush=True)


def fmt(value: object) -> str:
    if isinstance(value, (float, np.floating)):
        return repr(float(value)) if math.isfinite(value) else ""
    if isinstance(value, (bool, np.bool_)):
        return "1" if value else "0"
    return str(value)


def atomic_write(path: Path, columns: tuple[str, ...], rows: list[dict], compress: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    buffer = io.StringIO()
    writer = csv.writer(buffer, lineterminator="\n")
    writer.writerow(columns)
    for row in rows:
        writer.writerow([fmt(row[c]) for c in columns])
    data = buffer.getvalue().encode()
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "wb") as handle:
        handle.write(gzip.compress(data, compresslevel=6, mtime=0) if compress else data)
    os.replace(temporary, path)


def atomic_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
    with os.fdopen(descriptor, "w") as handle:
        json.dump(payload, handle, indent=2, sort_keys=True, default=float)
        handle.write("\n")
    os.replace(temporary, path)


# ---------------------------------------------------------------- inputs

def load_cohort() -> list[dict]:
    with open(COHORT_CSV, newline="") as handle:
        rows = [r for r in csv.DictReader(handle) if r["source"] == "cameo"]
    rows.sort(key=lambda r: r["protein_id"])
    selected = os.environ.get("NULL_TRIANGLES_PROTEINS")       # local smoke tests only
    if selected:
        keep = set(selected.split(","))
        rows = [r for r in rows if r["protein_id"] in keep]
    elif len(rows) != common.EXPECTED_CAMEO:
        raise ValueError(f"Expected {common.EXPECTED_CAMEO} CAMEO proteins, found {len(rows)}")
    return [{"protein_index": k, "protein_id": r["protein_id"], "n_residues": int(r["n_residues"]),
             "n_pairs": int(r["n_pairs"])} for k, r in enumerate(rows)]


def load_prediction_index(layer: int) -> dict[tuple[str, str], dict]:
    path = EXPORT / "layers" / f"layer_{layer:02d}" / "prediction_index.csv"
    with open(path, newline="") as handle:
        return {(r["protein_id"], r["analysis_name"]): r for r in csv.DictReader(handle)}


def load_paper_parity(layer: int) -> dict[tuple[str, str], dict]:
    with open(PARITY_CSV, newline="") as handle:
        return {(r["protein_id"], r["analysis_name"]): r for r in csv.DictReader(handle)
                if int(r["layer"]) == layer}


def load_prediction(layer: int, protein: dict, probe: str, index_row: dict | None) -> np.ndarray:
    path = EXPORT / "layers" / f"layer_{layer:02d}" / protein["protein_id"] / f"{probe}.npy"
    if index_row is not None:
        if (index_row["status"] != "complete" or int(index_row["n_values"]) != protein["n_pairs"]
                or sha256_file(path) != index_row["array_sha256"]):
            raise ValueError(f"Prediction checksum/coverage mismatch: {path}")
    values = np.load(path, allow_pickle=False)
    if values.dtype != np.float32 or values.shape != (protein["n_pairs"],) or not np.all(np.isfinite(values)):
        raise ValueError(f"Prediction shape/dtype/finiteness mismatch: {path}")
    return values


def load_structure(protein: dict) -> dict:
    pdb = helpers.resolve_combined_pdb_path(protein["protein_id"], "cameo")
    ca = helpers.extract_ca_coordinates(pdb)
    coordinates = np.asarray(ca["coordinates"], dtype=np.float64)
    if coordinates.shape[0] != protein["n_residues"]:
        raise ValueError(f"{protein['protein_id']}: {coordinates.shape[0]} CA atoms, expected {protein['n_residues']}")
    chains = [record[0] for record in ca["residues"]]
    return {"coordinates": coordinates, "chains": chains, "pdb": str(pdb)}


def pair_geometry(structure: dict) -> dict:
    coordinates = structure["coordinates"]
    pair_i, pair_j = strict_lower_triangle_indices(coordinates.shape[0])
    d_true = helpers.compute_distance_matrix(coordinates)[pair_i, pair_j]
    seq = sequence_separation_bin_codes(pair_i, pair_j, chain_ids=structure["chains"])
    return {"pair_i": pair_i, "pair_j": pair_j, "d_true": d_true, "codes": common.all_bin_codes(d_true, seq)}


# ---------------------------------------------------------------- paper metric (parity only)

def paper_all_triangles(prediction: np.ndarray, coordinates: np.ndarray, directory: Path) -> dict:
    """compute_metric_rows on one packed prediction; returns the all-triangles row."""
    n = coordinates.shape[0]
    pair_i, pair_j = np.tril_indices(n, k=-1)
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[pair_i, pair_j] = matrix[pair_j, pair_i] = np.asarray(prediction, dtype=np.float64)
    work = Path(tempfile.mkdtemp(prefix="paper_ref_", dir=directory))
    cache, _ = build_reference_cache(coordinates, work)
    try:
        identity = {
            "analysis_name": "parity", "model_family": "parity", "layer": 0, "protein_id": "parity",
            "source": "cameo", "generalization_scope": "parity", "n_residues": n,
            "protein_length_bin": "parity",
        }
        rows = compute_metric_rows(matrix, cache, identity)
    finally:
        cache.cleanup()
        shutil.rmtree(work, ignore_errors=True)
    row = next(r for r in rows if r["category_system"] == "all_triangles")
    return {**{k: row[k] for k in PARITY_KEYS}, "n_negative_predictions": row["n_negative_predictions"]}


# ---------------------------------------------------------------- pass 1

def pass_one(task: tuple) -> dict:
    layer, protein, index_rows = task
    began = time.perf_counter()
    structure = load_structure(protein)
    geometry = pair_geometry(structure)
    d_true = geometry["d_true"]
    out = {"protein": protein, "structure": structure, "moments": {}, "calibration": []}
    for probe in common.PROBES:
        prediction = load_prediction(layer, protein, probe, index_rows[probe]).astype(np.float64)
        residual = prediction - d_true
        for binning in ("B10", "B10xS6"):
            out["moments"][(probe, binning)] = common.bin_moments(
                residual, geometry["codes"][binning], len(common.BIN_LABELS[binning]))
        alpha, beta = common.calibration_fit(d_true, prediction)
        out["calibration"].append({
            "probe": probe, "layer": layer, "protein_id": protein["protein_id"],
            "n_residues": protein["n_residues"], "n_pairs": d_true.size, "alpha": alpha, "beta": beta,
            "mean_r": float(residual.mean()), "rmse": float(np.sqrt(np.mean(residual ** 2))),
            "n_negative_predictions": int(np.count_nonzero(prediction < 0.0)),
        })
    out["seconds"] = time.perf_counter() - began
    return out


def cohort_fallback(results: list[dict], probe: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    """Protein-equal mean of per-protein bin mean and SD over proteins with >= 30 pairs."""
    n_bins = len(common.B10_LABELS)
    n = np.stack([r["moments"][(probe, "B10")]["n"] for r in results])
    mu = np.stack([r["moments"][(probe, "B10")]["mean"] for r in results])
    sd = np.stack([r["moments"][(probe, "B10")]["sd"] for r in results])
    cohort_mu, cohort_sd, support = np.full(n_bins, np.nan), np.full(n_bins, np.nan), []
    for b in range(n_bins):
        keep = n[:, b] >= common.NULL_A_MIN_PAIRS
        if not keep.any():                    # no protein qualifies: use every protein with pairs
            keep = n[:, b] >= 2
        support.append(int(keep.sum()))
        if keep.any():
            cohort_mu[b], cohort_sd[b] = mu[keep, b].mean(), sd[keep, b].mean()
    return cohort_mu, cohort_sd, support


def null_a_parameters(n_pairs, mu, sd, cohort_mu, cohort_sd):
    used = (np.asarray(n_pairs) > 0) & (np.asarray(n_pairs) < common.NULL_A_MIN_PAIRS)
    return np.where(used, cohort_mu, mu), np.where(used, cohort_sd, sd), used


# ---------------------------------------------------------------- pass 2

def metric_row(probe, layer, protein, model, binning, triangle_set, draw, metrics, values) -> dict:
    return {"probe": probe, "layer": layer, "protein_id": protein["protein_id"],
            "n_residues": protein["n_residues"], "model": model, "binning": binning,
            "triangle_set": triangle_set, "draw": draw, **metrics,
            "n_negative_values": int(np.count_nonzero(np.asarray(values) < 0.0))}


def pass_two(task: tuple) -> dict:
    layer, protein, structure, fallback, scratch = task
    began = time.perf_counter()
    geometry = pair_geometry(structure)
    d_true, pair_i, pair_j, codes = geometry["d_true"], geometry["pair_i"], geometry["pair_j"], geometry["codes"]
    n_res = protein["n_residues"]
    triangles = common.sampled_triangle_set(d_true, n_res, protein["protein_index"])
    predictions = {p: load_prediction(layer, protein, p, None) for p in common.PROBES}
    rows, checks = [], {"bitwise_compared": 0, "bitwise_failures": [], "histogram_checks": 0,
                        "histogram_failures": 0}

    # Observed, exact and on the sample. A full sample is the exact set, batched identically.
    if triangles.exact:
        exact = {p: common.score_set(v, triangles, track_clipped=True) for p, v in predictions.items()}
        for probe, values in predictions.items():
            paper = paper_all_triangles(values, structure["coordinates"], scratch)
            checks["bitwise_compared"] += 1
            for key in PARITY_KEYS:
                if exact[probe][key] != paper[key]:
                    checks["bitwise_failures"].append([probe, key, exact[probe][key], paper[key]])
    else:
        exact = common.score_exact_streaming(predictions, d_true, n_res)
    for probe, values in predictions.items():
        rows.append(metric_row(probe, layer, protein, "observed", "none", "exact", -1, exact[probe], values))
        sample = exact[probe] if triangles.exact else common.score_set(values, triangles, track_clipped=True)
        rows.append(metric_row(probe, layer, protein, "observed", "none", "sample_1m", -1, sample, values))
    t_observed = time.perf_counter() - began

    # Nulls, on the sample.
    fallback_used = {}
    for probe in common.PROBES:
        residual = predictions[probe].astype(np.float64) - d_true
        moments = {b: common.bin_moments(residual, codes[b], len(common.BIN_LABELS[b])) for b in common.BINNINGS}
        a_mu, a_sd, used = null_a_parameters(moments["B10"]["n"], moments["B10"]["mean"], moments["B10"]["sd"],
                                             *fallback[probe])
        fallback_used[probe] = used.tolist()
        for model, binning in common.null_runs(probe):
            mu, sd = (a_mu, a_sd) if model == "null_a" else (moments[binning]["mean"], moments[binning]["sd"])
            for draw in range(common.N_DRAWS):
                rng = common.null_draw_rng(layer, probe, protein["protein_index"], model, binning, draw)
                values, same = common.make_null(model, rng, d_true, residual, codes[binning], pair_i, pair_j,
                                                n_res, mu, sd)
                if not np.all(np.isfinite(values)):
                    raise ValueError(f"{protein['protein_id']} {probe} {model} {binning}: nonfinite null values")
                if same is not None:
                    checks["histogram_checks"] += 1
                    checks["histogram_failures"] += int(not same)
                rows.append(metric_row(probe, layer, protein, model, binning, "sample_1m", draw,
                                       common.score_set(values, triangles), values))
    return {"protein_id": protein["protein_id"], "rows": rows, "checks": checks,
            "fallback_used": fallback_used, "sample_exact": triangles.exact, "n_sample": triangles.n,
            "seconds_observed": t_observed, "seconds": time.perf_counter() - began}


# ---------------------------------------------------------------- main

def paper_parity(rows: list[dict], paper: dict, cohort: list[dict]) -> dict:
    worst = {"fraction": 0.0, "relative_sum_rtol": 0.0, "negative": 0, "n_compared": 0}
    failures = []
    observed = {(r["protein_id"], r["probe"]): r for r in rows
                if r["model"] == "observed" and r["triangle_set"] == "exact"}
    for protein in cohort:
        for probe in common.PROBES:
            key = (protein["protein_id"], probe)
            ours, ref = observed[key], paper.get(key)
            if ref is None:
                failures.append([*key, "missing paper row"])
                continue
            worst["n_compared"] += 1
            n_ref = int(ref["n_triangles"])
            if ours["n_triangles"] != n_ref:
                failures.append([*key, "n_triangles", ours["n_triangles"], n_ref])
                continue
            fraction = abs(ours["n_violations"] - int(ref["n_violations"])) / n_ref
            ref_sum = float(ref["relative_violation_sum"])
            rtol = abs(ours["relative_violation_sum"] - ref_sum) / ref_sum if ref_sum else abs(ours["relative_violation_sum"])
            negative = abs(ours["n_negative_values"] - int(ref["n_negative_predictions"]))
            worst["fraction"] = max(worst["fraction"], fraction)
            worst["relative_sum_rtol"] = max(worst["relative_sum_rtol"], rtol)
            worst["negative"] = max(worst["negative"], negative)
            if fraction > PAPER_FRACTION_TOL or rtol > PAPER_RELATIVE_SUM_RTOL or negative > PAPER_NEGATIVE_TOL:
                failures.append([*key, "tolerance", fraction, rtol, negative])
    return {"worst": worst, "failures": failures}


def main() -> None:
    layer = int(os.environ.get("NULL_TRIANGLES_LAYER", os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    if layer not in common.LAYERS:
        raise ValueError(f"Layer {layer} is not in 0-47")
    shards, validation_dir = OUTPUT_ROOT / "layer_shards", OUTPUT_ROOT / "validation"
    marker = validation_dir / f"layer_{layer:02d}.json"
    if marker.is_file():
        log(f"layer {layer} already complete ({marker})")
        return
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    skip_paper = os.environ.get("NULL_TRIANGLES_SKIP_PAPER_PARITY") == "1"   # local smoke tests only
    cohort = load_cohort()
    index = load_prediction_index(layer)
    paper = load_paper_parity(layer)
    log(f"layer {layer}: {len(cohort)} proteins, {workers} workers, output {OUTPUT_ROOT}")

    context = mp.get_context("fork")
    order = sorted(cohort, key=lambda p: -p["n_residues"])
    with context.Pool(workers) as pool:
        tasks = [(layer, p, {probe: index[(p["protein_id"], probe)] for probe in common.PROBES}) for p in order]
        first = {r["protein"]["protein_id"]: r for r in pool.imap_unordered(pass_one, tasks, chunksize=1)}
        log(f"pass 1 done: pair statistics for {len(first)} proteins")
        results = [first[p["protein_id"]] for p in cohort]
        fallback, fallback_support = {}, {}
        for probe in common.PROBES:
            cohort_mu, cohort_sd, support = cohort_fallback(results, probe)
            fallback[probe], fallback_support[probe] = (cohort_mu, cohort_sd), support
        with tempfile.TemporaryDirectory(prefix="null_tri_", dir=os.environ.get("TMPDIR")) as scratch:
            tasks = [(layer, p, first[p["protein_id"]]["structure"], fallback, Path(scratch)) for p in order]
            second = {}
            for done, result in enumerate(pool.imap_unordered(pass_two, tasks, chunksize=1), 1):
                second[result["protein_id"]] = result
                if done % 10 == 0 or done == len(tasks):
                    log(f"pass 2: {done}/{len(tasks)} proteins (last {result['protein_id']}, "
                        f"{result['seconds']:.0f}s)")

    metric_rows = [row for p in cohort for row in second[p["protein_id"]]["rows"]]
    bitwise_failures = [f for r in second.values() for f in r["checks"]["bitwise_failures"]]
    histogram_failures = sum(r["checks"]["histogram_failures"] for r in second.values())
    parity = {"worst": {}, "failures": [], "skipped": True} if skip_paper else paper_parity(metric_rows, paper, cohort)
    log(f"bitwise vs compute_metric_rows: {sum(r['checks']['bitwise_compared'] for r in second.values())} "
        f"compared, {len(bitwise_failures)} failures")
    log(f"paper parity: worst {parity['worst']}, {len(parity['failures'])} failures")
    log(f"Null B histogram checks: {sum(r['checks']['histogram_checks'] for r in second.values())}, "
        f"{histogram_failures} failures")
    if bitwise_failures or parity["failures"] or histogram_failures:
        atomic_json(validation_dir / f"layer_{layer:02d}_FAILED.json", {
            "bitwise_failures": bitwise_failures, "paper_parity": parity,
            "histogram_failures": histogram_failures})
        raise SystemExit(f"layer {layer}: validation failed; nothing written")

    diagnostic_rows = []
    for result in results:
        pid = result["protein"]["protein_id"]
        for probe in common.PROBES:
            used = second[pid]["fallback_used"][probe]
            for binning in ("B10", "B10xS6"):
                m = result["moments"][(probe, binning)]
                for cell in np.flatnonzero(m["n"] > 0):
                    diagnostic_rows.append({
                        "probe": probe, "layer": layer, "protein_id": pid, "binning": binning,
                        "cell": int(cell), "cell_label": common.BIN_LABELS[binning][cell],
                        "n_pairs": int(m["n"][cell]), "mean_r": m["mean"][cell], "sd_r": m["sd"][cell],
                        "rmse": m["rmse"][cell], "skew": m["skew"][cell],
                        "excess_kurtosis": m["excess_kurtosis"][cell],
                        "null_a_fallback": bool(used[cell]) if binning == "B10" else False,
                    })
    calibration_rows = [row for result in results for row in result["calibration"]]
    atomic_write(shards / f"null_metrics_layer_{layer:02d}.csv.gz", METRIC_COLUMNS, metric_rows, True)
    atomic_write(shards / f"pair_diagnostics_layer_{layer:02d}.csv.gz", DIAGNOSTIC_COLUMNS, diagnostic_rows, True)
    atomic_write(shards / f"calibration_layer_{layer:02d}.csv", CALIBRATION_COLUMNS, calibration_rows, False)
    fallback_counts = {probe: [int(sum(second[p["protein_id"]]["fallback_used"][probe][b] for p in cohort))
                               for b in range(len(common.B10_LABELS))] for probe in common.PROBES}
    atomic_json(marker, {
        "layer": layer, "status": "complete", "n_proteins": len(cohort),
        "proteins_with_every_triple": sum(r["sample_exact"] for r in second.values()),
        "bitwise_compared": sum(r["checks"]["bitwise_compared"] for r in second.values()),
        "bitwise_failures": 0, "paper_parity": parity,
        "paper_tolerances": {"violation_fraction": PAPER_FRACTION_TOL,
                             "relative_violation_sum_rtol": PAPER_RELATIVE_SUM_RTOL,
                             "n_negative_predictions": PAPER_NEGATIVE_TOL},
        "histogram_checks": sum(r["checks"]["histogram_checks"] for r in second.values()),
        "histogram_failures": 0,
        "null_a_fallback_proteins_by_bin": fallback_counts,
        "null_a_cohort_support_by_bin": fallback_support,
        "b10_labels": list(common.B10_LABELS),
        "n_draws": common.N_DRAWS, "n_sample_triangles": common.N_SAMPLE_TRIANGLES, "seed": common.SEED,
        "rows": {"null_metrics": len(metric_rows), "pair_diagnostics": len(diagnostic_rows),
                 "calibration": len(calibration_rows)},
        "seconds": {"total": time.perf_counter() - STARTED,
                    "pass_one_protein_max": max(r["seconds"] for r in results),
                    "pass_two_protein_max": max(r["seconds"] for r in second.values()),
                    "pass_two_protein_sum": sum(r["seconds"] for r in second.values())},
        "inputs": {"export_root": str(EXPORT), "cohort_csv": str(COHORT_CSV),
                   "parity_csv_sha256": sha256_file(PARITY_CSV)},
    })
    log(f"layer {layer} complete: {len(metric_rows)} metric rows")


if __name__ == "__main__":
    main()

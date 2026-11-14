#!/usr/bin/env python3
"""SMACOF error decomposition: one layer, four probes, the 552-protein cohort.

Task i handles layer i. For each probe and protein it takes the observed OOF decoded
distances and the triangle experiment's two matched-noise controls on B10 (Gaussian-matched
noise and shuffled errors, 3 draws each), clips negatives to zero, fits each matrix with
metric SMACOF started from its own classical-MDS coordinates, and splits the error against
OpenFold's final structure into an inconsistent part, a consistent part and a cross term
(see ``smacof_error_decomposition_common``). The same split is saved for the classical-MDS
start. No model is fitted: inputs are the cached OOF predictions and the final structures.

The controls are the triangle experiment's own matrices. They are rebuilt by
``null_triangles_replay.build_vectors`` from the same seeds, predictions and structures.

Gates before anything is written (a failure writes only ``validation/layer_LL_FAILED.json``):
- the four-term identity holds on every row, for the SMACOF and the classical split;
- SMACOF raw stress is no higher than the classical-MDS raw stress it started from;
- every rebuilt vector has the negative-value count the triangle run saved, and on proteins
  with at most ``BITWISE_MAX_TRIANGLES`` triangles the same violation count; the shuffled
  control keeps every bin's residual histogram;
- the classical-MDS start reproduces the stress, RMSD and negative-prediction count saved in
  the canonical ``geometry_metrics.csv`` within the float32-cache tolerance.
"""
from __future__ import annotations

import csv
import gzip
import os
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parents[1]
# In the repository the helpers live in sibling folders; on Delta everything is deployed flat.
for _path, _module in ((CLUSTER_DIR / "triangle_violation_subgroups", "triangle_violation_subgroup_common.py"),
                       (CLUSTER_DIR / "null_triangles", "null_triangles_replay.py"),
                       (CLUSTER_DIR, "cluster_regression_helpers.py"),
                       (SCRIPT_DIR, "smacof_error_decomposition_common.py")):
    if (_path / _module).is_file() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cluster_regression_helpers as helpers  # noqa: E402
import null_triangles_common as triangles  # noqa: E402
import null_triangles_replay as replay  # noqa: E402
import smacof_error_decomposition_common as common  # noqa: E402
from canonical_storage import csv_root  # noqa: E402
from null_triangles_replay import atomic_json, atomic_write, log  # noqa: E402
from regression_output_helpers import sha256_file  # noqa: E402
from triangle_violation_subgroup_common import triangle_count  # noqa: E402

OUTPUT_ROOT = Path(os.environ.get(
    "SMACOF_DECOMPOSITION_OUTPUT_ROOT", csv_root() / "diagnostic_experiments" / common.EXPERIMENT_NAME))
TRIANGLE_ROOT = Path(os.environ.get(
    "SMACOF_DECOMPOSITION_TRIANGLE_ROOT", csv_root() / "main_experiments" / triangles.EXPERIMENT_NAME))
# classical-MDS columns of the four families' canonical geometry_metrics.csv (extraction in the README).
PARITY_CSV = SCRIPT_DIR / "classical_geometry_parity_5fold_v1.csv.gz"
REL_TOL = float(os.environ.get("SMACOF_REL_TOL", common.DEFAULT_REL_TOL))
MAX_ITER = int(os.environ.get("SMACOF_MAX_ITER", common.DEFAULT_MAX_ITER))
PARITY_STRESS_TOL = 1e-4            # float32-cache tolerance on classical normalized stress
PARITY_RMSD_TOL = 1e-2              # angstroms
PARITY_NEGATIVE_TOL = 2
STARTED = time.perf_counter()

SNAPSHOT_COLUMNS = tuple(
    f"smacof_{name}_at_{common.SNAPSHOT_LABELS[t]}"
    for t in common.SNAPSHOT_TOLS for name in ("iterations", "inconsistent_sq", "consistent_sq"))
DECOMPOSITION_COLUMNS = (
    "probe", "layer", "protein_id", "source", "n_residues", "n_pairs", "matrix_type", "binning", "draw",
    "n_negative_values", "total_sq",
    "smacof_inconsistent_sq", "smacof_consistent_sq", "smacof_cross", "smacof_identity_residual",
    "smacof_raw_stress", "classical_raw_stress", "smacof_normalized_stress", "classical_normalized_stress",
    "smacof_scale_ratio", "smacof_iterations", "smacof_converged", "smacof_final_relative_decrease",
    *SNAPSHOT_COLUMNS,
    "classical_inconsistent_sq", "classical_consistent_sq", "classical_cross", "classical_identity_residual",
    "classical_scale_ratio", "smacof_rmsd", "classical_rmsd", "smacof_tm_score", "smacof_seconds",
)
BIN_COLUMNS = (
    "probe", "layer", "protein_id", "source", "matrix_type", "bin_system", "bin", "bin_label", "n_pairs",
    "n_draws", "sum_total_sq", "sum_smacof_inconsistent_sq", "sum_smacof_consistent_sq",
    "sum_classical_inconsistent_sq", "sum_classical_consistent_sq",
)
BIN_LABELS = {"sequence_separation": triangles.SEQ_LABELS, "true_distance_b10": triangles.B10_LABELS}


# ---------------------------------------------------------------- inputs

def load_triangle_reference(layer: int) -> dict[tuple, dict]:
    """The triangle run's per-vector rows for this layer, keyed (probe, protein, model, binning, draw)."""
    path = TRIANGLE_ROOT / "layer_shards" / f"null_metrics_layer_{layer:02d}.csv.gz"
    with gzip.open(path, "rt", newline="") as handle:
        return {(r["probe"], r["protein_id"], r["model"], r["binning"], int(r["draw"])):
                {"n_negative_values": int(r["n_negative_values"]), "n_violations": int(r["n_violations"])}
                for r in csv.DictReader(handle)}


def load_classical_parity(layer: int) -> dict[tuple[str, str], dict]:
    with gzip.open(PARITY_CSV, "rt", newline="") as handle:
        return {(r["protein_id"], r["analysis_name"]):
                {"stress": float(r["stress"]), "rmsd": float(r["rmsd"]),
                 "n_negative_predictions": int(r["n_negative_predictions"])}
                for r in csv.DictReader(handle) if int(r["layer"]) == layer}


# ---------------------------------------------------------------- one protein

def aligned_rmsd(reference: np.ndarray, coordinates: np.ndarray) -> tuple[float, np.ndarray]:
    """Kabsch RMSD and aligned coordinates for the better of the two z-reflections."""
    reflected = coordinates.copy()
    reflected[:, 2] *= -1.0
    fits = [helpers.superimpose_structures(reference, candidate) for candidate in (coordinates, reflected)]
    aligned, rmsd = min(fits, key=lambda fit: fit[1])
    return float(rmsd), aligned


def selected_keys(vectors: dict) -> list[tuple]:
    """Observed vectors and the B10 controls; the triangle run's sensitivity binnings are left out."""
    return [key for key in vectors if key[1] == "observed" or key[2] == common.CONTROL_BINNING]


def fit_vector(values: np.ndarray, d_true: np.ndarray, reference: np.ndarray, with_tm_score: bool) -> dict:
    """Clip, classical MDS, SMACOF, both splits, RMSD and (optionally) TM-score for one vector."""
    began = time.perf_counter()
    n = reference.shape[0]
    d_hat = np.maximum(np.asarray(values, dtype=np.float64), 0.0)
    matrix = common.packed_to_matrix(d_hat, n)
    start, _ = helpers.classical_mds(matrix, n_dims=3)
    fit = common.smacof_metric(matrix, start, rel_tol=REL_TOL, max_iter=MAX_ITER)
    p_classical = common.packed_distances(start)
    p_smacof = common.packed_distances(fit["coordinates"])
    classical = common.decompose(d_hat, p_classical, d_true)
    smacof = common.decompose(d_hat, p_smacof, d_true)
    snapshots = {}
    for tolerance in common.SNAPSHOT_TOLS:
        label = common.SNAPSHOT_LABELS[tolerance]
        if tolerance in fit["snapshots"]:
            iterations, coordinates = fit["snapshots"][tolerance]
            terms = common.decompose(d_hat, common.packed_distances(coordinates), d_true)
            values_at = (iterations, terms["inconsistent_sq"], terms["consistent_sq"])
        else:                                   # never met before the cap, or tighter than rel_tol
            values_at = ("", float("nan"), float("nan"))
        for name, value in zip(("iterations", "inconsistent_sq", "consistent_sq"), values_at):
            snapshots[f"smacof_{name}_at_{label}"] = value
    smacof_rmsd, aligned = aligned_rmsd(reference, fit["coordinates"])
    classical_rmsd, _ = aligned_rmsd(reference, start)
    tm_score = (float(helpers.compute_mds_tm_score(reference, fit["coordinates"], n))
                if with_tm_score else float("nan"))
    return {"d_hat": d_hat, "p_smacof": p_smacof, "p_classical": p_classical, "smacof": smacof,
            "classical": classical, "fit": fit, "snapshots": snapshots, "smacof_rmsd": smacof_rmsd,
            "classical_rmsd": classical_rmsd, "tm_score": tm_score, "aligned": aligned,
            "seconds": time.perf_counter() - began}


def decomposition_row(key: tuple, layer: int, protein: dict, values: np.ndarray, result: dict) -> dict:
    probe, model, binning, draw = key
    smacof, classical, fit = result["smacof"], result["classical"], result["fit"]
    return {
        "probe": probe, "layer": layer, "protein_id": protein["protein_id"], "source": protein["source"],
        "n_residues": protein["n_residues"], "n_pairs": protein["n_pairs"],
        "matrix_type": common.MATRIX_TYPES[model], "binning": binning, "draw": draw,
        "n_negative_values": int(np.count_nonzero(np.asarray(values) < 0.0)),
        "total_sq": smacof["total_sq"],
        "smacof_inconsistent_sq": smacof["inconsistent_sq"], "smacof_consistent_sq": smacof["consistent_sq"],
        "smacof_cross": smacof["cross"], "smacof_identity_residual": smacof["identity_residual"],
        "smacof_raw_stress": smacof["raw_stress"], "classical_raw_stress": classical["raw_stress"],
        "smacof_normalized_stress": smacof["normalized_stress"],
        "classical_normalized_stress": classical["normalized_stress"],
        "smacof_scale_ratio": smacof["scale_ratio"], "smacof_iterations": fit["iterations"],
        "smacof_converged": fit["converged"], "smacof_final_relative_decrease": fit["final_relative_decrease"],
        **result["snapshots"],
        "classical_inconsistent_sq": classical["inconsistent_sq"],
        "classical_consistent_sq": classical["consistent_sq"], "classical_cross": classical["cross"],
        "classical_identity_residual": classical["identity_residual"],
        "classical_scale_ratio": classical["scale_ratio"],
        "smacof_rmsd": result["smacof_rmsd"], "classical_rmsd": result["classical_rmsd"],
        "smacof_tm_score": result["tm_score"], "smacof_seconds": result["seconds"],
    }


def row_failures(row: dict) -> list[list]:
    """Gate 1 (identity, both splits) and gate 2 (SMACOF stress no higher than its start)."""
    where = [row["protein_id"], row["probe"], row["matrix_type"], row["draw"]]
    failures = []
    for split in ("smacof", "classical"):
        if abs(row[f"{split}_identity_residual"]) > common.IDENTITY_RTOL * max(row["total_sq"], 1.0):
            failures.append([*where, f"{split}_identity", row[f"{split}_identity_residual"]])
    if row["smacof_raw_stress"] > row["classical_raw_stress"] * (1.0 + common.STRESS_SLACK_RTOL):
        failures.append([*where, "stress_above_start", row["smacof_raw_stress"], row["classical_raw_stress"]])
    return failures


def process_protein(task: tuple) -> dict:
    layer, protein, structure, fallback, triangle_reference, parity, with_tm_score = task
    began = time.perf_counter()
    pid, n = protein["protein_id"], protein["n_residues"]
    geometry = replay.pair_geometry(structure)
    d_true, reference = geometry["d_true"], structure["coordinates"]
    predictions = {probe: replay.load_prediction(layer, protein, probe, None) for probe in triangles.PROBES}
    vectors, _, checks = replay.build_vectors(layer, protein, geometry, predictions, fallback)
    keys = selected_keys(vectors)
    failures = {"rows": [], "pairing": [], "parity": []}
    worst = {"parity_stress": 0.0, "parity_rmsd": 0.0, "parity_negative": 0}

    # Gate 3: these are the vectors the triangle run scored.
    pairing_compared = 0
    if triangle_reference is not None:
        violations = None
        if triangle_count(n) <= triangles.BITWISE_MAX_TRIANGLES:
            violations = triangles.score_streaming({k: vectors[k] for k in keys}, d_true, n)
        for key in keys:
            saved = triangle_reference.get((key[0], pid, key[1], key[2], key[3]))
            negative = int(np.count_nonzero(np.asarray(vectors[key]) < 0.0))
            pairing_compared += 1
            if saved is None:
                failures["pairing"].append([pid, *key, "missing triangle row"])
            elif negative != saved["n_negative_values"]:
                failures["pairing"].append([pid, *key, "n_negative_values", negative, saved["n_negative_values"]])
            elif violations is not None and violations[key]["n_violations"] != saved["n_violations"]:
                failures["pairing"].append(
                    [pid, *key, "n_violations", violations[key]["n_violations"], saved["n_violations"]])

    codes = {"sequence_separation": geometry["codes"]["B10xS6"] % len(triangles.SEQ_LABELS),
             "true_distance_b10": geometry["codes"]["B10"]}
    rows, coordinates, sums = [], {}, {}
    for key in keys:
        probe, model, _, _ = key
        result = fit_vector(vectors[key], d_true, reference, with_tm_score and model == "observed")
        row = decomposition_row(key, layer, protein, vectors[key], result)
        rows.append(row)
        failures["rows"] += row_failures(row)
        if model == "observed":
            coordinates[f"{pid}/{probe}"] = result["aligned"].astype(np.float32)
            if parity is not None:              # gate 4
                saved = parity.get((pid, probe))
                if saved is None:
                    failures["parity"].append([pid, probe, "missing parity row"])
                else:
                    gaps = (abs(row["classical_normalized_stress"] - saved["stress"]),
                            abs(row["classical_rmsd"] - saved["rmsd"]),
                            abs(row["n_negative_values"] - saved["n_negative_predictions"]))
                    worst["parity_stress"] = max(worst["parity_stress"], gaps[0])
                    worst["parity_rmsd"] = max(worst["parity_rmsd"], gaps[1])
                    worst["parity_negative"] = max(worst["parity_negative"], gaps[2])
                    if gaps[0] > PARITY_STRESS_TOL or gaps[1] > PARITY_RMSD_TOL or gaps[2] > PARITY_NEGATIVE_TOL:
                        failures["parity"].append([pid, probe, "tolerance", *gaps])
        for system, system_codes in codes.items():
            n_bins = len(BIN_LABELS[system])
            by_smacof = common.bin_sums(result["d_hat"], result["p_smacof"], d_true, system_codes, n_bins)
            by_classical = common.bin_sums(result["d_hat"], result["p_classical"], d_true, system_codes, n_bins)
            entry = sums.setdefault((probe, model, system), {"n_draws": 0, "n_pairs": by_smacof["n_pairs"],
                                                             "values": np.zeros((5, n_bins))})
            entry["n_draws"] += 1
            entry["values"] += np.stack((by_smacof["sum_total_sq"], by_smacof["sum_inconsistent_sq"],
                                         by_smacof["sum_consistent_sq"], by_classical["sum_inconsistent_sq"],
                                         by_classical["sum_consistent_sq"]))

    bin_rows = []
    for (probe, model, system), entry in sums.items():
        means = entry["values"] / entry["n_draws"]          # control draws are averaged
        for b in np.flatnonzero(entry["n_pairs"] > 0):
            bin_rows.append({
                "probe": probe, "layer": layer, "protein_id": pid, "source": protein["source"],
                "matrix_type": common.MATRIX_TYPES[model], "bin_system": system, "bin": int(b),
                "bin_label": BIN_LABELS[system][b], "n_pairs": int(entry["n_pairs"][b]),
                "n_draws": entry["n_draws"], "sum_total_sq": means[0, b],
                "sum_smacof_inconsistent_sq": means[1, b], "sum_smacof_consistent_sq": means[2, b],
                "sum_classical_inconsistent_sq": means[3, b], "sum_classical_consistent_sq": means[4, b],
            })
    return {"protein_id": pid, "rows": rows, "bin_rows": bin_rows, "coordinates": coordinates,
            "failures": failures, "worst": worst, "pairing_compared": pairing_compared,
            "pairing_violation_checked": triangle_reference is not None
            and triangle_count(n) <= triangles.BITWISE_MAX_TRIANGLES,
            "histogram_checks": checks["histogram_checks"], "histogram_failures": checks["histogram_failures"],
            "n_vectors": len(keys), "seconds": time.perf_counter() - began}


# ---------------------------------------------------------------- main

def write_coordinates(path: Path, arrays: dict[str, np.ndarray]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".npz", dir=path.parent)
    with os.fdopen(descriptor, "wb") as handle:
        np.savez_compressed(handle, **arrays)
    os.replace(temporary, path)


def convergence_report(rows: list[dict]) -> dict:
    report = {}
    for matrix_type in common.MATRIX_TYPES.values():
        members = [r for r in rows if r["matrix_type"] == matrix_type]
        iterations = np.array([r["smacof_iterations"] for r in members], dtype=np.float64)
        ratio = np.array([r["smacof_scale_ratio"] for r in members], dtype=np.float64)
        report[matrix_type] = {
            "n_runs": len(members), "n_converged": int(sum(r["smacof_converged"] for r in members)),
            "iterations_median": float(np.median(iterations)), "iterations_p95": float(np.quantile(iterations, 0.95)),
            "iterations_max": int(iterations.max()),
            "scale_ratio_min": float(np.nanmin(ratio)), "scale_ratio_max": float(np.nanmax(ratio)),
        }
    return report


def snapshot_report(rows: list[dict]) -> dict:
    """Largest change in a row's inconsistent or consistent RMS (angstroms) between each snapshot and the end."""
    report = {}
    for tolerance in common.SNAPSHOT_TOLS:
        label = common.SNAPSHOT_LABELS[tolerance]
        worst, missing = 0.0, 0
        for r in rows:
            for part in ("inconsistent_sq", "consistent_sq"):
                at = r[f"smacof_{part}_at_{label}"]
                if not np.isfinite(at):
                    missing += 1
                    continue
                worst = max(worst, abs(np.sqrt(at) - np.sqrt(r[f"smacof_{part}"])))
        report[label] = {"max_abs_rms_change_angstrom": worst, "n_missing_values": missing}
    return report


def main() -> None:
    import multiprocessing as mp

    layer = int(os.environ.get("SMACOF_DECOMPOSITION_LAYER", os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    if layer not in triangles.LAYERS:
        raise ValueError(f"Layer {layer} is not in 0-47")
    shards, validation_dir = OUTPUT_ROOT / "layer_shards", OUTPUT_ROOT / "validation"
    marker = validation_dir / f"layer_{layer:02d}.json"
    if marker.is_file():
        log(f"layer {layer} already complete ({marker})")
        return
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    # The three switches below are for local smoke tests only.
    skip_pairing = os.environ.get("SMACOF_DECOMPOSITION_SKIP_TRIANGLE_PAIRING") == "1"
    skip_parity = os.environ.get("SMACOF_DECOMPOSITION_SKIP_CLASSICAL_PARITY") == "1"
    with_tm_score = os.environ.get("SMACOF_DECOMPOSITION_SKIP_TM_SCORE") != "1"
    cohort = replay.load_cohort()
    index = replay.load_prediction_index(layer)
    triangle_reference = None if skip_pairing else load_triangle_reference(layer)
    parity = None if skip_parity else load_classical_parity(layer)
    tm_provenance = helpers.tmscore_provenance() if with_tm_score else None
    log(f"layer {layer}: {len(cohort)} proteins, {workers} workers, rel_tol {REL_TOL:g}, "
        f"max_iter {MAX_ITER}, output {OUTPUT_ROOT}")

    context = mp.get_context("fork")
    order = sorted(cohort, key=lambda p: -p["n_residues"])
    with context.Pool(workers) as pool:
        tasks = [(layer, p, {probe: index[(p["protein_id"], probe)] for probe in triangles.PROBES}) for p in order]
        first = {r["protein"]["protein_id"]: r for r in pool.imap_unordered(replay.pass_one, tasks, chunksize=1)}
        log(f"pass 1 done: pair statistics for {len(first)} proteins")
        ordered = [first[p["protein_id"]] for p in cohort]          # cohort order, as the triangle run
        fallback = {probe: replay.cohort_fallback(ordered, probe)[:2] for probe in triangles.PROBES}
        by_protein: dict[str, dict] = {}
        for key, value in (triangle_reference or {}).items():
            by_protein.setdefault(key[1], {})[key] = value
        tasks = []
        for p in order:
            pid = p["protein_id"]
            reference = None if triangle_reference is None else by_protein.get(pid, {})
            protein_parity = None if parity is None else {
                (pid, probe): parity[(pid, probe)] for probe in triangles.PROBES if (pid, probe) in parity}
            tasks.append((layer, p, first[pid]["structure"], fallback, reference, protein_parity, with_tm_score))
        second = {}
        for done, result in enumerate(pool.imap_unordered(process_protein, tasks, chunksize=1), 1):
            second[result["protein_id"]] = result
            if done % 25 == 0 or done == len(tasks):
                log(f"pass 2: {done}/{len(tasks)} proteins (last {result['protein_id']}, "
                    f"{result['n_vectors']} matrices, {result['seconds']:.0f}s)")

    results = [second[p["protein_id"]] for p in cohort]
    rows = [row for r in results for row in r["rows"]]
    bin_rows = [row for r in results for row in r["bin_rows"]]
    failures = {name: [f for r in results for f in r["failures"][name]] for name in ("rows", "pairing", "parity")}
    histogram_failures = sum(r["histogram_failures"] for r in results)
    worst_parity = {name: max(r["worst"][name] for r in results)
                    for name in ("parity_stress", "parity_rmsd", "parity_negative")}
    convergence, snapshots = convergence_report(rows), snapshot_report(rows)
    gates = {
        "identity_and_stress": {"rows_checked": len(rows), "failures": failures["rows"]},
        "triangle_pairing": {"skipped": skip_pairing, "vectors_compared": sum(r["pairing_compared"] for r in results),
                             "proteins_with_violation_count_check":
                                 sum(r["pairing_violation_checked"] for r in results),
                             "histogram_checks": sum(r["histogram_checks"] for r in results),
                             "histogram_failures": histogram_failures, "failures": failures["pairing"]},
        "classical_parity": {"skipped": skip_parity, "worst": worst_parity, "failures": failures["parity"],
                             "tolerances": {"stress": PARITY_STRESS_TOL, "rmsd_angstrom": PARITY_RMSD_TOL,
                                            "n_negative_predictions": PARITY_NEGATIVE_TOL}},
    }
    log(f"identity/stress: {len(rows)} rows, {len(failures['rows'])} failures")
    log(f"triangle pairing: {gates['triangle_pairing']['vectors_compared']} vectors, "
        f"{len(failures['pairing'])} failures; histogram failures {histogram_failures}")
    log(f"classical parity: worst {worst_parity}, {len(failures['parity'])} failures")
    log(f"convergence: {convergence}")
    log(f"snapshots vs final: {snapshots}")
    if any(failures.values()) or histogram_failures:
        atomic_json(validation_dir / f"layer_{layer:02d}_FAILED.json", {"layer": layer, "gates": gates})
        raise SystemExit(f"layer {layer}: validation failed; nothing written")

    atomic_write(shards / f"decomposition_layer_{layer:02d}.csv.gz", DECOMPOSITION_COLUMNS, rows, True)
    atomic_write(shards / f"bin_sums_layer_{layer:02d}.csv.gz", BIN_COLUMNS, bin_rows, True)
    coordinates_path = shards / f"smacof_coordinates_layer_{layer:02d}.npz"
    write_coordinates(coordinates_path, {key: value for r in results for key, value in r["coordinates"].items()})
    atomic_json(marker, {
        "layer": layer, "status": "complete", "experiment": common.EXPERIMENT_NAME,
        "code_revision": os.environ.get("SMACOF_DECOMPOSITION_CODE_REVISION", ""),
        "n_proteins": len(cohort),
        "n_proteins_by_source": {s: sum(p["source"] == s for p in cohort) for s in triangles.EXPECTED_COHORT},
        "smacof": {"rel_tol": REL_TOL, "max_iter": MAX_ITER, "snapshot_tols": list(common.SNAPSHOT_TOLS),
                   "start": "classical_mds_top_three_nonnegative_eigenvalues", "n_starts": 1,
                   "objective": "raw_unweighted_stress_angstrom", "input_rescaling": "none",
                   "negative_handling": "clip_to_zero"},
        "controls": {"source": triangles.EXPERIMENT_NAME, "seed": triangles.SEED, "n_draws": triangles.N_DRAWS,
                     "binning": common.CONTROL_BINNING, "matrix_types": common.MATRIX_TYPES},
        "gates": gates, "convergence": convergence, "snapshots_vs_final": snapshots,
        "tm_score": tm_provenance,
        "rows": {"decomposition": len(rows), "bin_sums": len(bin_rows),
                 "coordinates": sum(len(r["coordinates"]) for r in results)},
        "sha256": {name: sha256_file(shards / name) for name in (
            f"decomposition_layer_{layer:02d}.csv.gz", f"bin_sums_layer_{layer:02d}.csv.gz",
            coordinates_path.name)},
        "seconds": {"total": time.perf_counter() - STARTED,
                    "protein_max": max(r["seconds"] for r in results),
                    "protein_sum": sum(r["seconds"] for r in results),
                    "smacof_sum": sum(row["smacof_seconds"] for row in rows)},
        "workers": workers,
        "inputs": {"export_root": str(replay.EXPORT), "cohort_csv": str(replay.COHORT_CSV),
                   "triangle_root": str(TRIANGLE_ROOT), "parity_csv_sha256": sha256_file(PARITY_CSV),
                   "ecod_pdb_root": str(helpers.ECOD_PDB_ROOT), "cameo_pdb_root": str(helpers.CAMEO_PDB_ROOT)},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
    })
    log(f"layer {layer} complete: {len(rows)} decomposition rows, {len(bin_rows)} bin rows")


if __name__ == "__main__":
    main()

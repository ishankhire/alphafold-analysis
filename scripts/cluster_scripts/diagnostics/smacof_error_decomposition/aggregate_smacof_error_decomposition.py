#!/usr/bin/env python3
"""Collect the SMACOF error-decomposition layer shards into analysis-ready tables.

Control draws are averaged within each protein first; the draw SD of each squared term is
kept so Monte Carlo noise can be reported. Cohort bin terms are protein-equal means of
per-bin mean squared terms. Bootstraps, text and figures are made locally by
``scripts/multi_protein/smacof_error_decomposition/analyze_smacof_error_decomposition.py``.
"""
from __future__ import annotations

import csv
import gzip
import io
import json
import os
import subprocess
import sys
import tempfile
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

import smacof_error_decomposition as worker
import smacof_error_decomposition_common as common
from smacof_error_decomposition import OUTPUT_ROOT, atomic_json, atomic_write, log, sha256_file, triangles

LAYERS = tuple(int(x) for x in os.environ.get("SMACOF_DECOMPOSITION_AGGREGATE_LAYERS", "").split(",") if x) \
    or triangles.LAYERS
BIN_MIN_PAIRS = 10                  # per protein, to contribute to a cohort bin term
TERMS = ("total_sq", "smacof_inconsistent_sq", "smacof_consistent_sq", "smacof_cross",
         "classical_inconsistent_sq", "classical_consistent_sq", "classical_cross")
PER_PROTEIN_COLUMNS = (
    "probe", "layer", "protein_id", "source", "n_residues", "n_pairs", "matrix_type", "n_draws",
    *TERMS, *(f"{t}_draw_sd" for t in TERMS),
    "smacof_raw_stress", "classical_raw_stress", "smacof_normalized_stress", "classical_normalized_stress",
    "smacof_scale_ratio", "smacof_iterations_mean", "smacof_iterations_max", "n_unconverged",
    "smacof_rmsd", "classical_rmsd", "smacof_tm_score", "n_negative_values",
)
BIN_SUM_COLUMNS = ("sum_total_sq", "sum_smacof_inconsistent_sq", "sum_smacof_consistent_sq",
                   "sum_classical_inconsistent_sq", "sum_classical_consistent_sq")
COHORT_BIN_COLUMNS = (
    "probe", "layer", "matrix_type", "bin_system", "bin", "bin_label", "n_proteins", "cohort_n_pairs",
    "protein_mean_total_sq", "protein_mean_smacof_inconsistent_sq", "protein_mean_smacof_consistent_sq",
    "protein_mean_smacof_cross", "protein_mean_classical_inconsistent_sq", "protein_mean_classical_consistent_sq",
    "protein_mean_classical_cross",
)
CONVERGENCE_COLUMNS = ("probe", "layer", "matrix_type", "n_runs", "n_converged", "iterations_median",
                       "iterations_p95", "iterations_max", "scale_ratio_min", "scale_ratio_max")
BIN_FILES = {"sequence_separation": "bin_sums_by_sequence_separation.csv.gz",
             "true_distance_b10": "bin_sums_by_true_distance.csv.gz"}


def num(value: str) -> float:
    return float(value) if value != "" else float("nan")


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


class GzipTable:
    """A gzip CSV written shard by shard to a temporary file, then moved into place."""

    def __init__(self, path: Path, columns: tuple[str, ...]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        self.path, self.columns, self.n_rows = path, columns, 0
        descriptor, self.temporary = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent)
        self.raw = os.fdopen(descriptor, "wb")
        self.zipped = gzip.GzipFile(fileobj=self.raw, mode="wb", compresslevel=6, mtime=0)
        self.text = io.TextIOWrapper(self.zipped, newline="")
        self.writer = csv.writer(self.text, lineterminator="\n")
        self.writer.writerow(columns)

    def add(self, rows: list[dict], formatted: bool = False) -> None:
        for row in rows:
            self.writer.writerow([row[c] if formatted else worker.replay.fmt(row[c]) for c in self.columns])
        self.n_rows += len(rows)

    def close(self) -> int:
        self.text.flush()
        self.zipped.close()
        self.raw.close()
        os.replace(self.temporary, self.path)
        return self.n_rows


def draw_sd(values: np.ndarray) -> float:
    return float(np.std(values, ddof=1)) if values.size > 1 else float("nan")


def per_protein_rows(layer: int, rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["probe"], r["protein_id"], r["matrix_type"])].append(r)
    out = []
    for (probe, pid, matrix_type), members in sorted(groups.items()):
        expected = 1 if matrix_type == "observed" else triangles.N_DRAWS
        if len(members) != expected:
            raise ValueError(f"layer {layer} {probe} {pid} {matrix_type}: {len(members)} rows, expected {expected}")
        column = lambda name: np.array([num(m[name]) for m in members])  # noqa: E731
        iterations = column("smacof_iterations")
        row = {"probe": probe, "layer": layer, "protein_id": pid, "source": members[0]["source"],
               "n_residues": int(members[0]["n_residues"]), "n_pairs": int(members[0]["n_pairs"]),
               "matrix_type": matrix_type, "n_draws": len(members)}
        for term in TERMS:
            values = column(term)
            row[term], row[f"{term}_draw_sd"] = float(values.mean()), draw_sd(values)
        for name in ("smacof_raw_stress", "classical_raw_stress", "smacof_normalized_stress",
                     "classical_normalized_stress", "smacof_scale_ratio", "smacof_rmsd", "classical_rmsd",
                     "n_negative_values"):
            row[name] = float(column(name).mean())
        row["smacof_tm_score"] = num(members[0]["smacof_tm_score"]) if matrix_type == "observed" else float("nan")
        row["smacof_iterations_mean"] = float(iterations.mean())
        row["smacof_iterations_max"] = int(iterations.max())
        row["n_unconverged"] = int(sum(m["smacof_converged"] != "1" for m in members))
        gap = abs(row["total_sq"] - row["smacof_inconsistent_sq"] - row["smacof_consistent_sq"] - row["smacof_cross"])
        if gap > 10 * common.IDENTITY_RTOL * max(row["total_sq"], 1.0):
            raise ValueError(f"layer {layer} {probe} {pid} {matrix_type}: draw-averaged identity fails")
        out.append(row)
    return out


def cohort_bin_rows(layer: int, rows: list[dict]) -> list[dict]:
    """Protein-equal mean of per-bin mean squared terms, over proteins with >= BIN_MIN_PAIRS pairs."""
    cells = defaultdict(list)
    for r in rows:
        cells[(r["probe"], r["matrix_type"], r["bin_system"], int(r["bin"]))].append(r)
    out = []
    for (probe, matrix_type, system, b), members in sorted(cells.items()):
        keep = [m for m in members if int(m["n_pairs"]) >= BIN_MIN_PAIRS]
        row = {"probe": probe, "layer": layer, "matrix_type": matrix_type, "bin_system": system, "bin": b,
               "bin_label": members[0]["bin_label"], "n_proteins": len(keep),
               "cohort_n_pairs": sum(int(m["n_pairs"]) for m in members)}
        means = {}
        for name in BIN_SUM_COLUMNS:
            values = [num(m[name]) / int(m["n_pairs"]) for m in keep]
            means[name] = float(np.mean(values)) if values else float("nan")
        row.update({
            "protein_mean_total_sq": means["sum_total_sq"],
            "protein_mean_smacof_inconsistent_sq": means["sum_smacof_inconsistent_sq"],
            "protein_mean_smacof_consistent_sq": means["sum_smacof_consistent_sq"],
            "protein_mean_smacof_cross": means["sum_total_sq"] - means["sum_smacof_inconsistent_sq"]
            - means["sum_smacof_consistent_sq"],
            "protein_mean_classical_inconsistent_sq": means["sum_classical_inconsistent_sq"],
            "protein_mean_classical_consistent_sq": means["sum_classical_consistent_sq"],
            "protein_mean_classical_cross": means["sum_total_sq"] - means["sum_classical_inconsistent_sq"]
            - means["sum_classical_consistent_sq"],
        })
        out.append(row)
    return out


def convergence_rows(layer: int, rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["probe"], r["matrix_type"])].append(r)
    out = []
    for (probe, matrix_type), members in sorted(groups.items()):
        iterations = np.array([num(m["smacof_iterations"]) for m in members])
        ratio = np.array([num(m["smacof_scale_ratio"]) for m in members])
        out.append({"probe": probe, "layer": layer, "matrix_type": matrix_type, "n_runs": len(members),
                    "n_converged": int(sum(m["smacof_converged"] == "1" for m in members)),
                    "iterations_median": float(np.median(iterations)),
                    "iterations_p95": float(np.quantile(iterations, 0.95)), "iterations_max": int(iterations.max()),
                    "scale_ratio_min": float(np.nanmin(ratio)), "scale_ratio_max": float(np.nanmax(ratio))})
    return out


def check_bin_totals(layer: int, decomposition: list[dict], bins: list[dict]) -> None:
    """Per-bin sums of the observed matrices add up to the whole-protein terms, in both bin systems."""
    totals = defaultdict(lambda: np.zeros(3))
    for r in bins:
        if r["matrix_type"] == "observed":
            totals[(r["probe"], r["protein_id"], r["bin_system"])] += [
                num(r["sum_total_sq"]), num(r["sum_smacof_inconsistent_sq"]), num(r["sum_smacof_consistent_sq"])]
    for r in decomposition:
        if r["matrix_type"] != "observed":
            continue
        expected = np.array([num(r["total_sq"]), num(r["smacof_inconsistent_sq"]), num(r["smacof_consistent_sq"])])
        for system in common.BIN_SYSTEMS:
            got = totals[(r["probe"], r["protein_id"], system)] / int(r["n_pairs"])
            if not np.allclose(got, expected, rtol=1e-9, atol=1e-12):
                raise ValueError(f"layer {layer} {r['probe']} {r['protein_id']} {system}: bin sums do not add up")


def code_revision() -> str:
    """The deployed revision (passed at submission on Delta), else the checkout's HEAD."""
    if os.environ.get("SMACOF_DECOMPOSITION_CODE_REVISION"):
        return os.environ["SMACOF_DECOMPOSITION_CODE_REVISION"]
    try:
        return subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def main() -> None:
    started = time.perf_counter()
    shards, validation, tables = OUTPUT_ROOT / "layer_shards", OUTPUT_ROOT / "validation", OUTPUT_ROOT / "tables"
    missing = [layer for layer in LAYERS if not (validation / f"layer_{layer:02d}.json").is_file()]
    if missing:
        raise SystemExit(f"Incomplete layers: {missing}")
    reports = {layer: json.loads((validation / f"layer_{layer:02d}.json").read_text()) for layer in LAYERS}
    settings = {json.dumps(r["smacof"], sort_keys=True) for r in reports.values()}
    if len(settings) != 1:
        raise SystemExit("Layers were run with different SMACOF settings; refusing to mix them")
    for layer, report in reports.items():
        gates = report["gates"]
        if report["status"] != "complete" or any(gates[g]["failures"] for g in gates) \
                or gates["triangle_pairing"]["histogram_failures"]:
            raise SystemExit(f"layer {layer}: validation report is not clean")

    per_draw = GzipTable(tables / "decomposition_per_draw.csv.gz", worker.DECOMPOSITION_COLUMNS)
    bin_tables = {system: GzipTable(tables / name, worker.BIN_COLUMNS) for system, name in BIN_FILES.items()}
    per_protein, cohort_bins, convergence = [], [], []
    for layer in LAYERS:
        report = reports[layer]
        names = (f"decomposition_layer_{layer:02d}.csv.gz", f"bin_sums_layer_{layer:02d}.csv.gz")
        for name in names:
            if sha256_file(shards / name) != report["sha256"][name]:
                raise SystemExit(f"layer {layer}: {name} changed since it was validated")
        decomposition, bins = read_gz(shards / names[0]), read_gz(shards / names[1])
        if len(decomposition) != report["rows"]["decomposition"] or len(bins) != report["rows"]["bin_sums"]:
            raise SystemExit(f"layer {layer}: shard row counts do not match the validation report")
        keys = {(r["probe"], r["protein_id"], r["matrix_type"], r["draw"]) for r in decomposition}
        if len(keys) != len(decomposition):
            raise SystemExit(f"layer {layer}: duplicate decomposition keys")
        check_bin_totals(layer, decomposition, bins)
        per_draw.add(decomposition, formatted=True)
        for system, table in bin_tables.items():
            table.add([r for r in bins if r["bin_system"] == system], formatted=True)
        per_protein += per_protein_rows(layer, decomposition)
        cohort_bins += cohort_bin_rows(layer, bins)
        convergence += convergence_rows(layer, decomposition)
        log(f"aggregated layer {layer}")

    n_per_draw = per_draw.close()
    n_bins = {system: table.close() for system, table in bin_tables.items()}
    atomic_write(tables / "decomposition_per_protein.csv.gz", PER_PROTEIN_COLUMNS, per_protein, True)
    atomic_write(tables / "bin_terms_cohort.csv", COHORT_BIN_COLUMNS, cohort_bins, False)
    atomic_write(tables / "convergence_summary.csv", CONVERGENCE_COLUMNS, convergence, False)
    proteins = {(r["layer"], r["protein_id"]) for r in per_protein}
    n_proteins = reports[LAYERS[0]]["n_proteins"]
    if len(proteins) != n_proteins * len(LAYERS):
        raise SystemExit("A protein is missing at some layer")
    first = reports[LAYERS[0]]
    atomic_json(tables / "run_manifest.json", {
        "experiment": common.EXPERIMENT_NAME, "layers": list(LAYERS), "probes": list(triangles.PROBES),
        "n_proteins": n_proteins, "n_proteins_by_source": first["n_proteins_by_source"],
        "evaluation_scope": "oof_all_pairs", "smacof": first["smacof"], "controls": first["controls"],
        "bin_systems": {name: list(labels) for name, labels in worker.BIN_LABELS.items()},
        "bin_min_pairs": BIN_MIN_PAIRS, "code_revision": code_revision(),
        "tm_score": first["tm_score"], "versions": first["versions"], "inputs": first["inputs"],
        "checks": {
            "identity_and_stress_rows": sum(r["gates"]["identity_and_stress"]["rows_checked"] for r in reports.values()),
            "triangle_pairing_vectors": sum(r["gates"]["triangle_pairing"]["vectors_compared"] for r in reports.values()),
            "histogram_checks": sum(r["gates"]["triangle_pairing"]["histogram_checks"] for r in reports.values()),
            "classical_parity_worst": {name: max(r["gates"]["classical_parity"]["worst"][name] for r in reports.values())
                                       for name in ("parity_stress", "parity_rmsd", "parity_negative")},
            "failures": 0,
        },
        "unconverged_runs": sum(c["n_runs"] - c["n_converged"] for r in reports.values()
                                for c in r["convergence"].values()),
        "snapshots_vs_final": {layer: r["snapshots_vs_final"] for layer, r in reports.items()},
        "layer_seconds": {layer: r["seconds"]["total"] for layer, r in reports.items()},
        "rows": {"decomposition_per_draw": n_per_draw, "decomposition_per_protein": len(per_protein),
                 **{BIN_FILES[system]: n for system, n in n_bins.items()},
                 "bin_terms_cohort": len(cohort_bins), "convergence_summary": len(convergence)},
        "shard_sha256": {layer: r["sha256"] for layer, r in reports.items()},
        "aggregation_seconds": time.perf_counter() - started,
    })
    log(f"tables written to {tables}")


if __name__ == "__main__":
    sys.exit(main())

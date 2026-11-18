#!/usr/bin/env python3
"""Collect the per-channel MDS layer shards into analysis-ready tables.

Writes the concatenated per-channel table, a per-protein-and-layer summary over the 128
channels, the reference rows (final-structure distances and ``|i - j|`` through the same MDS)
and a runtime table. Cohort statistics and figures are made locally by
``scripts/multi_protein/per_channel_mds/analyze_per_channel_mds.py``.
"""
from __future__ import annotations

import csv
import gzip
import json
import os
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

import per_channel_mds as worker
import per_channel_mds_common as common
from per_channel_mds import OUTPUT_ROOT, atomic_json, atomic_write, helpers, log, replay, sha256_file

LAYERS = tuple(int(x) for x in os.environ.get("PER_CHANNEL_MDS_AGGREGATE_LAYERS", "").split(",") if x) \
    or worker.LAYERS
SUMMARISED = ("stress", "negative_eigenvalue_fraction", "frac_top3_eigenvalues",
              "perm_stress", "perm_negative_eigenvalue_fraction", "perm_frac_top3_eigenvalues")
STATISTICS = ("mean", "var", "min", "p10", "median", "max")
SUMMARY_COLUMNS = ("layer", "protein_id", "source", "n_residues", "n_channels_ok", "best_channel",
                   *(f"{name}_{statistic}" for name in SUMMARISED for statistic in STATISTICS))
REFERENCE_COLUMNS = ("protein_id", "source", "n_residues", "reference", "shift", *common.GEOMETRY,
                     "embedding_true_distance_pearson")
RUNTIME_COLUMNS = ("analysis_name", "layer", "status", "elapsed_seconds", "protein_seconds_sum",
                   "load_seconds_sum", "peak_rss_mib", "workers")


def read_shard(path: Path) -> tuple[list[str], list[dict]]:
    with gzip.open(path, "rt", newline="") as handle:
        lines = handle.read().splitlines()
    return lines, list(csv.DictReader(lines))


def summary_rows(layer: int, rows: list[dict]) -> list[dict]:
    by_protein: dict[str, list[dict]] = {}
    for row in rows:
        by_protein.setdefault(row["protein_id"], []).append(row)
    out = []
    for pid, group in by_protein.items():
        ok = [r for r in group if r["status"] == "ok"]
        summary = {"layer": layer, "protein_id": pid, "source": group[0]["source"],
                   "n_residues": int(group[0]["n_residues"]), "n_channels_ok": len(ok),
                   "best_channel": int(min(ok, key=lambda r: float(r["stress"]))["channel"]) if ok else -1}
        for name in SUMMARISED:
            values = np.array([float(r[name]) for r in ok], dtype=np.float64)
            statistics = {"mean": np.mean, "var": lambda v: np.var(v, ddof=1) if v.size > 1 else np.nan,
                          "min": np.min, "p10": lambda v: np.percentile(v, 10), "median": np.median,
                          "max": np.max}
            for statistic in STATISTICS:
                summary[f"{name}_{statistic}"] = float(statistics[statistic](values)) if values.size else np.nan
        out.append(summary)
    return out


def reference_table(cohort: list[dict]) -> list[dict]:
    rows = []
    for protein in cohort:
        structure = replay.load_structure(protein)
        for row in common.reference_rows(helpers.compute_distance_matrix(structure["coordinates"])):
            rows.append({"protein_id": protein["protein_id"], "source": protein["source"],
                         "n_residues": protein["n_residues"], **row})
    return rows


def code_revision() -> str:
    """The deployed files' commit: the environment variable, else the deployment manifest."""
    revision = os.environ.get("PER_CHANNEL_MDS_CODE_REVISION", "")
    manifest = Path(__file__).resolve().parent / "deployment_manifest.json"
    if not revision and manifest.is_file():
        revision = json.loads(manifest.read_text()).get("git_commit", "")
    if not revision:
        try:
            revision = subprocess.run(["git", "-C", str(Path(__file__).resolve().parent), "rev-parse", "HEAD"],
                                      capture_output=True, text=True, check=True).stdout.strip()
        except (OSError, subprocess.CalledProcessError):
            revision = ""
    return revision


def main() -> None:
    started = time.perf_counter()
    shards, validation, tables = OUTPUT_ROOT / "layer_shards", OUTPUT_ROOT / "validation", OUTPUT_ROOT / "tables"
    missing = [layer for layer in LAYERS if not (validation / f"layer_{layer:02d}.json").is_file()]
    if missing:
        raise SystemExit(f"Incomplete layers: {missing}")
    reports = {layer: json.loads((validation / f"layer_{layer:02d}.json").read_text()) for layer in LAYERS}
    if len({json.dumps(r["settings"], sort_keys=True) for r in reports.values()}) != 1:
        raise SystemExit("Layers were run with different settings; refusing to mix them")
    if len({r["n_proteins"] for r in reports.values()}) != 1:
        raise SystemExit("Layers cover different numbers of proteins")
    for layer, report in reports.items():
        if report["status"] != "complete" or report["gates"]["failures"]:
            raise SystemExit(f"layer {layer}: validation report is not clean")

    tables.mkdir(parents=True, exist_ok=True)
    cohort = worker.load_cohort()
    expected = {p["protein_id"] for p in cohort}
    summaries, runtime, n_rows = [], [], 0
    descriptor, temporary = tempfile.mkstemp(prefix=".channel_metrics.", dir=tables)
    with os.fdopen(descriptor, "wb") as raw, gzip.GzipFile(fileobj=raw, mode="wb", compresslevel=6, mtime=0) as out:
        out.write((",".join(worker.SHARD_COLUMNS) + "\n").encode())
        for layer in LAYERS:
            report = reports[layer]
            name = f"channel_metrics_layer_{layer:02d}.csv.gz"
            if sha256_file(shards / name) != report["sha256"][name]:
                raise SystemExit(f"layer {layer}: {name} changed since it was validated")
            lines, rows = read_shard(shards / name)
            if lines[0].split(",") != list(worker.SHARD_COLUMNS) or len(rows) != report["rows"]:
                raise SystemExit(f"layer {layer}: shard header or row count does not match the validation report")
            keys = {(r["protein_id"], r["channel"]) for r in rows}
            if len(keys) != len(rows) or {r["protein_id"] for r in rows} != expected \
                    or any(int(r["layer"]) != layer for r in rows):
                raise SystemExit(f"layer {layer}: duplicate keys, wrong layer or wrong proteins")
            out.write(("\n".join(lines[1:]) + "\n").encode())
            n_rows += len(rows)
            summaries += summary_rows(layer, rows)
            runtime.append({"analysis_name": common.EXPERIMENT_NAME, "layer": layer, "status": report["status"],
                            "elapsed_seconds": report["seconds"]["total"],
                            "protein_seconds_sum": report["seconds"]["protein_sum"],
                            "load_seconds_sum": report["seconds"]["load_sum"],
                            "peak_rss_mib": report["peak_rss_mib"]["worker_max"], "workers": report["workers"]})
            log(f"aggregated layer {layer}")
    os.replace(temporary, tables / "channel_metrics.csv.gz")
    if n_rows != len(cohort) * worker.N_CHANNELS * len(LAYERS):
        raise SystemExit(f"{n_rows} rows for {len(cohort)} proteins and {len(LAYERS)} layers")

    references = reference_table(cohort)
    atomic_write(tables / "protein_layer_summary.csv", SUMMARY_COLUMNS, summaries, False)
    atomic_write(tables / "reference_metrics.csv", REFERENCE_COLUMNS, references, False)
    atomic_write(tables / "runtime.csv", RUNTIME_COLUMNS, runtime, False)
    first = reports[LAYERS[0]]
    atomic_json(tables / "run_manifest.json", {
        "experiment": common.EXPERIMENT_NAME, "analysis_status": "complete", "layers": list(LAYERS),
        "n_proteins": len(cohort), "source": worker.SOURCE, "smoke": first["smoke"],
        "settings": first["settings"], "references": list(common.REFERENCES), "shifts": list(common.SHIFTS),
        "code_revision": code_revision(), "layer_code_revisions": sorted({r["code_revision"] for r in reports.values()}),
        "versions": first["versions"], "inputs": first["inputs"],
        "cohort_csv_sha256": sha256_file(replay.COHORT_CSV),
        "checks": {"rows_checked": sum(r["gates"]["rows_checked"] for r in reports.values()),
                   "parity_max_abs_difference": max(r["gates"]["parity_max_abs_difference"] for r in reports.values()),
                   "n_constant_channels": sum(r["gates"]["n_constant_channels"] for r in reports.values()),
                   "failures": 0},
        "rows": {"channel_metrics": n_rows, "protein_layer_summary": len(summaries),
                 "reference_metrics": len(references), "runtime": len(runtime)},
        "shard_sha256": {layer: r["sha256"] for layer, r in reports.items()},
        "layer_seconds": {layer: r["seconds"]["total"] for layer, r in reports.items()},
        "aggregation_seconds": time.perf_counter() - started,
    })
    log(f"tables written to {tables}")


if __name__ == "__main__":
    sys.exit(main())

#!/usr/bin/env python3
"""Collect the 48 null-triangle layer shards into compact, analysis-ready tables.

Every summary is a protein-equal mean of per-protein statistics. Draws are averaged
within each protein first; the draw SD is kept so Monte Carlo noise can be reported.
Bootstraps and figures are made locally by
``scripts/multi_protein/null_triangles/analyze_null_triangles.py``.
"""
from __future__ import annotations

import csv
import gzip
import json
import math
import os
import sys
import time
from collections import defaultdict
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SCRIPT_DIR, SCRIPT_DIR.parents[1], SCRIPT_DIR.parents[1] / "triangle_violation_subgroups"):
    if _path.is_dir() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import null_triangles_common as common  # noqa: E402
from null_triangles_replay import OUTPUT_ROOT, atomic_json, atomic_write, log  # noqa: E402

DIAGNOSTIC_MIN_PAIRS = 10          # per protein, to contribute to a bin or cell
CROSSTAB_MIN_COHORT_PAIRS = 100    # cohort-wide, below which a cross-tab cell is left empty
LAYERS = tuple(int(x) for x in os.environ.get("NULL_TRIANGLES_AGGREGATE_LAYERS", "").split(",") if x) or common.LAYERS

PER_PROTEIN_COLUMNS = (
    "probe", "layer", "protein_id", "n_residues", "model", "binning", "triangle_set", "n_draws",
    "n_triangles", "violation_fraction", "violation_fraction_draw_sd", "mean_relative_violation_all",
    "mean_relative_violation_all_draw_sd", "n_violations", "clipped_side_fraction",
)
STAT_COLUMNS = ("n_pairs", "bias", "sd", "rmse", "bias_sq", "sd_sq", "mse", "skew", "excess_kurtosis")
BIN_COLUMNS = ("probe", "layer", "binning", "cell", "cell_label", "distance_bin", "sequence_bin",
               "n_proteins_contributing", "cohort_n_pairs",
               *(f"protein_mean_{c}" for c in STAT_COLUMNS))
FALLBACK_COLUMNS = ("probe", "layer", "bin", "bin_label", "n_proteins_fallback", "n_proteins_cohort_support")


def read_gz_csv(path: Path) -> list[dict]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def num(value: str) -> float:
    return float(value) if value != "" else float("nan")


def draw_sd(values: list[float]) -> float:
    return float(np.std(values, ddof=1)) if len(values) > 1 else float("nan")


def per_protein_rows(layer: int, rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for r in rows:
        groups[(r["probe"], r["protein_id"], r["model"], r["binning"], r["triangle_set"])].append(r)
    out = []
    for (probe, pid, model, binning, triangle_set), members in sorted(groups.items()):
        vf = [num(m["violation_fraction"]) for m in members]
        rel = [num(m["mean_relative_violation_all"]) for m in members]
        expected = 1 if model == "observed" else common.N_DRAWS
        if len(members) != expected:
            raise ValueError(f"layer {layer} {probe} {pid} {model} {binning}: {len(members)} rows, expected {expected}")
        nv = sum(int(m["n_violations"]) for m in members) / len(members)
        clipped = (int(members[0]["n_violations_with_clipped_side"]) / int(members[0]["n_violations"])
                   if model == "observed" and int(members[0]["n_violations"]) else float("nan"))
        out.append({
            "probe": probe, "layer": layer, "protein_id": pid, "n_residues": int(members[0]["n_residues"]),
            "model": model, "binning": binning, "triangle_set": triangle_set, "n_draws": len(members),
            "n_triangles": int(members[0]["n_triangles"]),
            "violation_fraction": float(np.mean(vf)), "violation_fraction_draw_sd": draw_sd(vf),
            "mean_relative_violation_all": float(np.mean(rel)),
            "mean_relative_violation_all_draw_sd": draw_sd(rel),
            "n_violations": nv, "clipped_side_fraction": clipped,
        })
    return out


def bin_summary(layer: int, rows: list[dict], binning: str, labels: tuple[str, ...]) -> list[dict]:
    """Protein-equal means over proteins with >= DIAGNOSTIC_MIN_PAIRS pairs in the cell."""
    cells = defaultdict(list)
    for r in rows:
        if r["binning"] == binning:
            cells[(r["probe"], int(r["cell"]))].append(r)
    out = []
    for probe in common.PROBES:
        for cell, label in enumerate(labels):
            members = cells.get((probe, cell), [])
            cohort_pairs = sum(int(m["n_pairs"]) for m in members)
            keep = [m for m in members if int(m["n_pairs"]) >= DIAGNOSTIC_MIN_PAIRS]
            distance_bin, _, sequence_bin = label.partition("|")
            row = {"probe": probe, "layer": layer, "binning": binning, "cell": cell, "cell_label": label,
                   "distance_bin": distance_bin, "sequence_bin": sequence_bin,
                   "n_proteins_contributing": len(keep), "cohort_n_pairs": cohort_pairs}
            empty = not keep or (binning == "B10xS6" and cohort_pairs < CROSSTAB_MIN_COHORT_PAIRS)
            if empty:
                row.update({f"protein_mean_{c}": float("nan") for c in STAT_COLUMNS})
            else:
                bias = np.array([num(m["mean_r"]) for m in keep])
                sd = np.array([num(m["sd_r"]) for m in keep])
                rmse = np.array([num(m["rmse"]) for m in keep])
                mse = rmse ** 2
                if not np.allclose(mse, bias ** 2 + sd ** 2, rtol=1e-9, atol=1e-9):
                    raise ValueError(f"layer {layer} {probe} {label}: RMSE^2 != bias^2 + sd^2 per protein")
                row.update({
                    "protein_mean_n_pairs": float(np.mean([int(m["n_pairs"]) for m in keep])),
                    "protein_mean_bias": float(bias.mean()), "protein_mean_sd": float(sd.mean()),
                    "protein_mean_rmse": float(rmse.mean()), "protein_mean_bias_sq": float(np.mean(bias ** 2)),
                    "protein_mean_sd_sq": float(np.mean(sd ** 2)), "protein_mean_mse": float(mse.mean()),
                    "protein_mean_skew": float(np.nanmean([num(m["skew"]) for m in keep])),
                    "protein_mean_excess_kurtosis": float(np.nanmean([num(m["excess_kurtosis"]) for m in keep])),
                })
                identity_gap = abs(row["protein_mean_mse"] - row["protein_mean_bias_sq"] - row["protein_mean_sd_sq"])
                if identity_gap > 1e-9 * max(1.0, row["protein_mean_mse"]):
                    raise ValueError(f"layer {layer} {probe} {label}: mean(mse) != mean(bias^2) + mean(sd^2)")
            out.append(row)
    return out


def main() -> None:
    started = time.perf_counter()
    shards, validation = OUTPUT_ROOT / "layer_shards", OUTPUT_ROOT / "validation"
    tables = OUTPUT_ROOT / "tables"
    missing = [layer for layer in LAYERS if not (validation / f"layer_{layer:02d}.json").is_file()]
    if missing:
        raise SystemExit(f"Incomplete layers: {missing}")
    per_protein, by_bin, crosstab, calibration, fallback, layer_reports = [], [], [], [], [], {}
    calibration_columns = None
    for layer in LAYERS:
        report = json.loads((validation / f"layer_{layer:02d}.json").read_text())
        if report["status"] != "complete" or report["bitwise_failures"] or report["histogram_failures"] \
                or report["paper_parity"]["failures"]:
            raise SystemExit(f"layer {layer}: validation report is not clean")
        layer_reports[layer] = report
        per_protein += per_protein_rows(layer, read_gz_csv(shards / f"null_metrics_layer_{layer:02d}.csv.gz"))
        diagnostics = read_gz_csv(shards / f"pair_diagnostics_layer_{layer:02d}.csv.gz")
        by_bin += bin_summary(layer, diagnostics, "B10", common.BIN_LABELS["B10"])
        crosstab += bin_summary(layer, diagnostics, "B10xS6", common.BIN_LABELS["B10xS6"])
        with open(shards / f"calibration_layer_{layer:02d}.csv", newline="") as handle:
            reader = csv.DictReader(handle)
            calibration_columns = tuple(reader.fieldnames)
            calibration += list(reader)
        for probe in common.PROBES:
            for b, label in enumerate(common.B10_LABELS):
                fallback.append({"probe": probe, "layer": layer, "bin": b, "bin_label": label,
                                 "n_proteins_fallback": report["null_a_fallback_proteins_by_bin"][probe][b],
                                 "n_proteins_cohort_support": report["null_a_cohort_support_by_bin"][probe][b]})
        log(f"aggregated layer {layer}")

    atomic_write(tables / "null_per_protein.csv", PER_PROTEIN_COLUMNS, per_protein, False)
    atomic_write(tables / "error_diagnostics_by_bin.csv", BIN_COLUMNS, by_bin, False)
    atomic_write(tables / "crosstab_b10_seqsep.csv", BIN_COLUMNS, crosstab, False)
    atomic_write(tables / "calibration_per_protein.csv", calibration_columns, calibration, False)
    atomic_write(tables / "null_a_fallback_counts.csv", FALLBACK_COLUMNS, fallback, False)
    worst = {
        "paper_fraction": max(r["paper_parity"]["worst"]["fraction"] for r in layer_reports.values()),
        "paper_relative_sum_rtol": max(r["paper_parity"]["worst"]["relative_sum_rtol"] for r in layer_reports.values()),
        "paper_negative": max(r["paper_parity"]["worst"]["negative"] for r in layer_reports.values()),
    }
    atomic_json(tables / "run_manifest.json", {
        "experiment": common.EXPERIMENT_NAME, "layers": list(LAYERS), "probes": list(common.PROBES),
        "n_proteins": layer_reports[LAYERS[0]]["n_proteins"], "seed": common.SEED, "n_draws": common.N_DRAWS,
        "n_sample_triangles": common.N_SAMPLE_TRIANGLES, "null_a_min_pairs": common.NULL_A_MIN_PAIRS,
        "diagnostic_min_pairs": DIAGNOSTIC_MIN_PAIRS, "crosstab_min_cohort_pairs": CROSSTAB_MIN_COHORT_PAIRS,
        "null_runs": [list(r) for r in common.NULL_RUNS], "sensitivity_runs": [list(r) for r in common.SENSITIVITY_RUNS],
        "sensitivity_probe": common.SENSITIVITY_PROBE,
        "bins": {name: list(labels) for name, labels in common.BIN_LABELS.items()},
        "checks": {
            "bitwise_compared": sum(r["bitwise_compared"] for r in layer_reports.values()),
            "bitwise_failures": 0,
            "paper_parity_compared": sum(r["paper_parity"]["worst"]["n_compared"] for r in layer_reports.values()),
            "paper_parity_worst": worst,
            "histogram_checks": sum(r["histogram_checks"] for r in layer_reports.values()),
            "histogram_failures": 0,
        },
        "layer_seconds": {layer: r["seconds"]["total"] for layer, r in layer_reports.items()},
        "rows": {"null_per_protein": len(per_protein), "error_diagnostics_by_bin": len(by_bin),
                 "crosstab_b10_seqsep": len(crosstab), "calibration_per_protein": len(calibration),
                 "null_a_fallback_counts": len(fallback)},
        "aggregation_seconds": time.perf_counter() - started,
    })
    log(f"tables written to {tables}")


if __name__ == "__main__":
    main()

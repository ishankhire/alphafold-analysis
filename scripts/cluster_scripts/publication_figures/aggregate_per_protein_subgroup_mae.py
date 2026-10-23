#!/usr/bin/env python3
"""Validate, consolidate, summarize, and reconcile publication subgroup MAE."""

from __future__ import annotations

import csv
import math
import os
import resource
import sys
import tempfile
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
CV_DIR = CLUSTER_DIR / "cross_validation"
TRIANGLE_DIR = CLUSTER_DIR / "triangle_violation_subgroups"
for path in (SCRIPT_DIR, CLUSTER_DIR, CV_DIR, TRIANGLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from publication_subgroup_mae_common import (  # noqa: E402
    ANALYSIS_NAME, CATEGORY_LABELS, EXPECTED_PROTEINS, LAYERS, PER_PROTEIN_COLUMNS,
    RECONCILIATION_COLUMNS, RUNTIME_COLUMNS, SUMMARY_COLUMNS, validate_complete_layer,
)
from regression_output_helpers import atomic_write_csv, atomic_write_json, sha256_file  # noqa: E402
from triangle_violation_subgroup_common import FAMILY_BY_NAME, FAMILY_SPECS  # noqa: E402


from canonical_storage import assert_writable, csv_root as storage_csv_root, workflow_root, normalize_shared_tree


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def csv_root() -> Path:
    return storage_csv_root()


def producer_root() -> Path:
    return csv_root() / "regression_5fold_v1"


def output_root() -> Path:
    return workflow_root(ANALYSIS_NAME, "PUBLICATION_SUBGROUP_OUTPUT_ROOT", analysis=True)


def convert_row(source: dict[str, str]) -> dict[str, object]:
    row: dict[str, object] = dict(source)
    for column in ("feature_dim", "layer", "outer_fold", "n_residues", "n_pairs"):
        row[column] = int(source[column])
    for column in ("sum_absolute_error", "mae"):
        row[column] = float(source[column])
    return row


def load_producer_metrics() -> dict[tuple[str, int, str, str], tuple[int, float]]:
    result: dict[tuple[str, int, str, str], tuple[int, float]] = {}
    filenames = {
        "sequence_separation": "subgroup_metrics/by_sequence_separation.csv",
        "spatial_distance": "subgroup_metrics/by_spatial_distance.csv",
    }
    for spec in FAMILY_SPECS:
        for scope, filename in filenames.items():
            path = producer_root() / spec.family_dir / filename
            with path.open(newline="") as handle:
                rows = list(csv.DictReader(handle))
            for row in rows:
                key = (spec.analysis_name, int(row["layer"]), scope, row["category_label"])
                if key in result:
                    raise ValueError(f"Duplicate producer subgroup key {key}")
                result[key] = (int(row["n_evaluated_pairs"]), float(row["pooled_mae"]))
    expected = len(FAMILY_SPECS) * len(LAYERS) * sum(map(len, CATEGORY_LABELS.values()))
    if len(result) != expected:
        raise ValueError(f"Producer subgroup inventory has {len(result)} rows, expected {expected}")
    return result


def main() -> None:
    assert_writable(output_root())
    started = time.perf_counter()
    root = output_root()
    shards = root / "layer_shards"
    expected_shards = [shards / f"per_protein_subgroup_mae_layer_{layer:02d}.csv" for layer in LAYERS]
    missing = [str(path) for path in expected_shards if not path.is_file()]
    if missing:
        raise FileNotFoundError(f"Missing layer shards: {missing}")

    destination = root / "per_protein_subgroup_mae.csv"
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{destination.name}.", suffix=".tmp", dir=root)
    values: dict[tuple[str, int, str, str], list[float]] = defaultdict(list)
    pair_totals: dict[tuple[str, int, str, str], list[float]] = defaultdict(lambda: [0.0, 0.0])
    runtime_rows: list[dict[str, object]] = []
    total_rows = 0
    try:
        with os.fdopen(descriptor, "w", newline="") as output:
            writer = csv.DictWriter(output, fieldnames=PER_PROTEIN_COLUMNS)
            writer.writeheader()
            for layer, shard in zip(LAYERS, expected_shards):
                with shard.open(newline="") as handle:
                    rows = [convert_row(row) for row in csv.DictReader(handle)]
                validate_complete_layer(rows, layer)
                for row in rows:
                    writer.writerow({column: row[column] for column in PER_PROTEIN_COLUMNS})
                    key = (
                        str(row["analysis_name"]), int(row["layer"]),
                        str(row["category_scope"]), str(row["category_label"]),
                    )
                    mae = float(row["mae"])
                    if math.isfinite(mae):
                        values[key].append(mae)
                    pair_totals[key][0] += float(row["n_pairs"])
                    pair_totals[key][1] += float(row["sum_absolute_error"])
                total_rows += len(rows)
                runtime_path = shards / f"runtime_layer_{layer:02d}.csv"
                if runtime_path.is_file():
                    with runtime_path.open(newline="") as handle:
                        runtime_rows.extend(dict(row) for row in csv.DictReader(handle))
            output.flush()
            os.fsync(output.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise

    expected_rows = EXPECTED_PROTEINS * len(FAMILY_SPECS) * len(LAYERS) * 11
    if total_rows != expected_rows:
        raise ValueError(f"Consolidated {total_rows} rows, expected {expected_rows}")

    producer = load_producer_metrics()
    summary_rows: list[dict[str, object]] = []
    reconciliation_rows: list[dict[str, object]] = []
    for key in sorted(pair_totals):
        analysis, layer, scope, label = key
        spec = FAMILY_BY_NAME[analysis]
        finite = np.asarray(values[key], dtype=np.float64)
        if finite.size == 0:
            raise ValueError(f"No protein has finite support for {key}")
        n_pairs = int(pair_totals[key][0])
        replayed_pooled = pair_totals[key][1] / n_pairs
        producer_pairs, producer_pooled = producer[key]
        difference = abs(replayed_pooled - producer_pooled)
        status = "complete" if n_pairs == producer_pairs and math.isclose(
            replayed_pooled, producer_pooled, rel_tol=1e-10, abs_tol=1e-10,
        ) else "mismatch"
        reconciliation_rows.append({
            "analysis_name": analysis, "model_family": spec.model_family, "layer": layer,
            "category_scope": scope, "category_label": label, "replayed_n_pairs": n_pairs,
            "producer_n_pairs": producer_pairs, "replayed_pooled_mae": replayed_pooled,
            "producer_pooled_mae": producer_pooled, "absolute_difference": difference,
            "status": status,
        })
        if status != "complete":
            raise ValueError(f"Producer pooled-MAE reconciliation failed for {key}")
        sd = float(np.std(finite, ddof=1)) if finite.size > 1 else math.nan
        summary_rows.append({
            "analysis_name": analysis, "model_family": spec.model_family, "feature_dim": 256,
            "layer": layer, "cv_protocol_version": "2.0",
            "evaluation_design": spec.evaluation_design, "evaluation_scope": "oof_all_pairs",
            "oof_prediction_scope": spec.oof_prediction_scope, "category_scope": scope,
            "category_label": label, "n_proteins_total": EXPECTED_PROTEINS,
            "n_proteins_finite": int(finite.size), "protein_equal_mean_mae": float(finite.mean()),
            "protein_sd_mae": sd, "protein_se_mae": sd / math.sqrt(finite.size),
            "pair_pooled_mae": replayed_pooled, "n_pairs": n_pairs,
        })

    atomic_write_csv(root / "subgroup_mae_summary.csv", SUMMARY_COLUMNS, summary_rows)
    atomic_write_csv(root / "pooled_reconciliation.csv", RECONCILIATION_COLUMNS, reconciliation_rows)
    atomic_write_csv(root / "runtime.csv", RUNTIME_COLUMNS, runtime_rows)
    output_files = (
        "per_protein_subgroup_mae.csv", "subgroup_mae_summary.csv",
        "pooled_reconciliation.csv", "runtime.csv",
    )
    atomic_write_json(root / "run_manifest.json", {
        "analysis_name": ANALYSIS_NAME,
        "analysis_status": "complete",
        "created_at": utc_now(),
        "n_layers": len(LAYERS),
        "n_proteins": EXPECTED_PROTEINS,
        "n_model_families": len(FAMILY_SPECS),
        "n_per_protein_subgroup_rows": total_rows,
        "n_summary_rows": len(summary_rows),
        "aggregation_elapsed_seconds": time.perf_counter() - started,
        "aggregation_peak_rss_mb": peak_rss_mb(),
        "layer_shards": {path.name: sha256_file(path) for path in expected_shards},
        "files": {name: sha256_file(root / name) for name in output_files},
    })
    print(f"Consolidated {total_rows} rows into {destination}", flush=True)

    normalize_shared_tree(output_root())


if __name__ == "__main__":
    main()

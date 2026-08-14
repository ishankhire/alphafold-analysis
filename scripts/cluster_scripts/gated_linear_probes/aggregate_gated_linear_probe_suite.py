#!/usr/bin/env python3
"""Validate gated-probe task shards and publish deterministic result tables."""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from gated_linear_probe_common import (  # noqa: E402
    FEATURE_DIM,
    GATE_STATUS_FIELDNAMES,
    METRIC_FIELDNAMES,
    PER_PROTEIN_FIELDNAMES,
    atomic_savez,
    atomic_write_csv,
    atomic_write_text,
    configured_layers,
    consolidated_root,
    expected_gates,
    feature_names,
    gate_slug,
    joint_task_for_id,
    layer_for_task,
    log,
    metrics_from_moments,
    run_mode,
    sha256_file,
    suite_root,
    validate_unique_rows,
)


COEFFICIENT_FIELDNAMES = [
    "analysis_name", "layer", "train_gate_aa_pair", "train_gate_sequence_bin",
    "feature_index", "feature_name", "coefficient_standardized", "coefficient_raw_scale",
]
COEFFICIENT_INDEX_FIELDNAMES = [
    "analysis_name", "layer", "train_gate_aa_pair", "train_gate_sequence_bin",
    "task_id", "npz_file", "npz_row", "feature_dim", "fit_status", "sha256",
]
METRIC_KEY = [
    "analysis_name", "layer", "split", "train_gate_aa_pair", "train_gate_sequence_bin",
    "evaluation_scope", "eval_aa_pair", "eval_sequence_bin", "eval_distance_bin",
]
AUDIT_REPLAY_ABS_TOL = 1e-8
AUDIT_REPLAY_REL_TOL = 1e-9


def _assert_close(actual, expected, label):
    actual = float(actual); expected = float(expected)
    if math.isnan(actual) and math.isnan(expected):
        return
    if not math.isclose(
        actual,
        expected,
        rel_tol=AUDIT_REPLAY_REL_TOL,
        abs_tol=AUDIT_REPLAY_ABS_TOL,
    ):
        raise ValueError(f"Audit replay mismatch for {label}: {actual} != {expected}")


def _verify_audit_rows(metric_rows, audit_path):
    with np.load(audit_path) as data:
        audits = {}
        for split in ("train", "test"):
            cell_code = np.asarray(data[f"{split}_cell_code"], dtype=np.int32)
            protein_code = np.asarray(data[f"{split}_protein_code"], dtype=np.int32)
            moments = np.asarray(data[f"{split}_moments"], dtype=np.float64)
            audits[split] = np.column_stack((cell_code, protein_code, moments))
    for split in ("train", "test"):
        rows = [row for row in metric_rows if row["split"] == split]
        audit = audits[split]
        for cell_code, row in enumerate(rows):
            cell = audit[audit[:, 0] == cell_code] if audit.size else np.empty((0, 8))
            if not cell.size:
                if str(row["fit_status"]).startswith("insufficient"):
                    continue
                if int(row["n_pairs"]) == 0:
                    continue
                raise ValueError(f"Missing audit data for {split} cell {cell_code}")
            pooled_moment = np.sum(cell[:, 2:8], axis=0, dtype=np.float64)
            pooled = metrics_from_moments(pooled_moment)
            per = [metrics_from_moments(item[2:8]) for item in cell]
            finite_r2 = [item["r2"] for item in per if math.isfinite(item["r2"])]
            _assert_close(row["pooled_r2"], pooled["r2"], "pooled_r2")
            _assert_close(row["pooled_mae"], pooled["mae"], "pooled_mae")
            _assert_close(row["pooled_rmse"], pooled["rmse"], "pooled_rmse")
            _assert_close(row["mean_per_protein_r2"],
                          np.mean(finite_r2) if finite_r2 else float("nan"), "macro_r2")
            _assert_close(row["mean_per_protein_mae"], np.mean([item["mae"] for item in per]), "macro_mae")
            _assert_close(row["mean_per_protein_rmse"], np.mean([item["rmse"] for item in per]), "macro_rmse")
            if int(row["n_pairs"]) != int(pooled_moment[0]) or int(row["n_proteins"]) != len(cell):
                raise ValueError("Audit support counts do not match metric row")


def _load_analysis(root, analysis_name, task_count):
    metric_rows, status_rows, per_rows, coefficient_records = [], [], [], []
    run_manifest = json.loads((root / "run_manifest.json").read_text())
    for task_id in range(task_count):
        task_dir = root / analysis_name / "task_shards" / f"task_{task_id:03d}"
        manifest_path = task_dir / "task_manifest.json"
        if not manifest_path.exists():
            raise ValueError(f"Missing task manifest: {manifest_path}")
        task_manifest = json.loads(manifest_path.read_text())
        mode = run_manifest["run_mode"]
        if analysis_name == "analysis_4_aa_sequence_gated":
            expected_layer, expected_sequence = joint_task_for_id(task_id, mode)
        else:
            expected_layer, expected_sequence = layer_for_task(task_id, mode), None
        expected_slugs = [gate_slug(*gate) for gate in expected_gates(analysis_name, expected_sequence)]
        if (task_manifest.get("layer") != expected_layer
                or task_manifest.get("sequence_task_code") != expected_sequence
                or task_manifest.get("expected_gate_keys") != expected_slugs):
            raise ValueError(f"Task mapping mismatch in {manifest_path}")
        if not task_manifest.get("completed"):
            raise ValueError(f"Task is incomplete: {manifest_path}")
        if task_manifest["configuration_digest"] != run_manifest["configuration_digest"]:
            raise ValueError(f"Mixed configuration digest in task {task_id}")
        if task_manifest["split_digest"] != run_manifest["split_digest"]:
            raise ValueError(f"Mixed split digest in task {task_id}")
        if task_manifest["completed_gate_keys"] != task_manifest["expected_gate_keys"]:
            raise ValueError(f"Incomplete or reordered gate list in task {task_id}")
        for slug in task_manifest["expected_gate_keys"]:
            result_path = task_dir / "gates" / f"{slug}.json"
            coefficient_path = task_dir / "gates" / f"{slug}_coefficients.npz"
            audit_path = task_dir / "gates" / f"{slug}_audit.npz"
            artifacts = task_manifest["artifacts"][slug]
            for path, key in ((result_path, "result_sha256"), (coefficient_path, "coefficient_sha256"),
                              (audit_path, "audit_sha256")):
                if not path.exists() or sha256_file(path) != artifacts[key]:
                    raise ValueError(f"Missing or corrupt artifact: {path}")
            result = json.loads(result_path.read_text())
            _verify_audit_rows(result["metric_rows"], audit_path)
            metric_rows.extend(result["metric_rows"])
            status_rows.append(result["gate_status"])
            per_rows.extend(result["per_protein_rows"])
            coefficient_records.append((task_id, slug, result["gate_status"], coefficient_path))
    validate_unique_rows(metric_rows, METRIC_KEY, f"{analysis_name} metric rows")
    validate_unique_rows(status_rows,
                         ["analysis_name", "layer", "train_gate_aa_pair", "train_gate_sequence_bin"],
                         f"{analysis_name} gate status")
    return metric_rows, status_rows, per_rows, coefficient_records


def _write_summary(path, analysis_name, status_rows, metric_rows):
    counts = {}
    for row in status_rows:
        counts[row["fit_status"]] = counts.get(row["fit_status"], 0) + 1
    test_overall = [row for row in metric_rows if row["split"] == "test"
                    and row["evaluation_scope"] == "gate_overall"
                    and math.isfinite(float(row["pooled_rmse_skill"]))]
    mean_skill = float(np.mean([float(row["pooled_rmse_skill"]) for row in test_overall])) if test_overall else float("nan")
    lines = [
        f"# {analysis_name.replace('_', ' ').title()}", "",
        f"Consolidated {len(status_rows):,} training gates and {len(metric_rows):,} train/test metric rows.", "",
        f"Fit-status counts: `{json.dumps(counts, sort_keys=True)}`", "",
        f"Mean held-out gate-overall pooled RMSE skill across defined cells: `{mean_skill:.6g}`", "",
        "Rank-deficient, ill-conditioned, and sparse cells remain in the CSV outputs with explicit status flags.", "",
    ]
    atomic_write_text(path, "\n".join(lines))


def _publish_analysis(destination, analysis_name, metric_rows, status_rows, per_rows, coefficient_records):
    output = destination / analysis_name
    overall = [row for row in metric_rows if row["evaluation_scope"] == "gate_overall"]
    atomic_write_csv(output / "gate_summary.csv", METRIC_FIELDNAMES, overall)
    if analysis_name == "analysis_1_sequence_gated":
        atomic_write_csv(output / "subgroup_by_amino_acid_pair.csv", METRIC_FIELDNAMES,
                         [row for row in metric_rows if row["evaluation_scope"] == "aa_pair"])
        atomic_write_csv(output / "subgroup_by_spatial_distance.csv", METRIC_FIELDNAMES,
                         [row for row in metric_rows if row["evaluation_scope"] == "spatial_distance"])
    elif analysis_name == "analysis_2_aa_pair_gated":
        atomic_write_csv(output / "subgroup_by_sequence_separation.csv", METRIC_FIELDNAMES,
                         [row for row in metric_rows if row["evaluation_scope"] == "sequence_separation"])
        atomic_write_csv(output / "subgroup_by_spatial_distance.csv", METRIC_FIELDNAMES,
                         [row for row in metric_rows if row["evaluation_scope"] == "spatial_distance"])
    else:
        atomic_write_csv(output / "subgroup_by_spatial_distance.csv", METRIC_FIELDNAMES,
                         [row for row in metric_rows if row["evaluation_scope"] == "spatial_distance"])
    atomic_write_csv(output / "test_per_protein_gate_metrics.csv.gz", PER_PROTEIN_FIELDNAMES, per_rows, compress=True)
    atomic_write_csv(output / "gate_status.csv", GATE_STATUS_FIELDNAMES, status_rows)

    if analysis_name in {"analysis_1_sequence_gated", "analysis_2_aa_pair_gated"}:
        coefficient_rows = []
        names = feature_names()
        for _, _, status, coefficient_path in coefficient_records:
            with np.load(coefficient_path) as data:
                coef_std = np.asarray(data["coef_std"], dtype=np.float64)
                if not coef_std.size:
                    continue
                coef_raw = np.asarray(data["coef_raw"], dtype=np.float64)
                coefficient_rows.append({
                    "analysis_name": analysis_name, "layer": status["layer"],
                    "train_gate_aa_pair": status["train_gate_aa_pair"],
                    "train_gate_sequence_bin": status["train_gate_sequence_bin"],
                    "feature_index": -1, "feature_name": "intercept",
                    "coefficient_standardized": float(data["intercept_std"]),
                    "coefficient_raw_scale": float(data["intercept_raw"]),
                })
                coefficient_rows.extend({
                    "analysis_name": analysis_name, "layer": status["layer"],
                    "train_gate_aa_pair": status["train_gate_aa_pair"],
                    "train_gate_sequence_bin": status["train_gate_sequence_bin"],
                    "feature_index": index, "feature_name": names[index],
                    "coefficient_standardized": coef_std[index],
                    "coefficient_raw_scale": coef_raw[index],
                } for index in range(FEATURE_DIM))
        atomic_write_csv(output / "coefficients.csv", COEFFICIENT_FIELDNAMES, coefficient_rows)
    else:
        shard_dir = output / "coefficient_shards"
        index_rows = []
        by_task = {}
        for record in coefficient_records:
            by_task.setdefault(record[0], []).append(record)
        for task_id, task_records in sorted(by_task.items()):
            coef_std = np.full((len(task_records), FEATURE_DIM), np.nan, dtype=np.float64)
            coef_raw = np.full_like(coef_std, np.nan)
            means = np.full_like(coef_std, np.nan); scales = np.full_like(coef_std, np.nan)
            intercept_std = np.full(len(task_records), np.nan); intercept_raw = np.full(len(task_records), np.nan)
            statuses, labels = [], []
            for row_index, (_, _, status, path) in enumerate(task_records):
                with np.load(path) as data:
                    if np.asarray(data["coef_std"]).size:
                        coef_std[row_index] = data["coef_std"]; coef_raw[row_index] = data["coef_raw"]
                        means[row_index] = data["scaler_mean"]; scales[row_index] = data["scaler_scale"]
                        intercept_std[row_index] = data["intercept_std"]; intercept_raw[row_index] = data["intercept_raw"]
                statuses.append(status["fit_status"])
                labels.append(f"{status['train_gate_aa_pair']}|{status['train_gate_sequence_bin']}")
            shard_path = shard_dir / f"task_{task_id:03d}.npz"
            atomic_savez(shard_path, gate_labels=np.asarray(labels), coef_std=coef_std, coef_raw=coef_raw,
                         intercept_std=intercept_std, intercept_raw=intercept_raw, scaler_mean=means,
                         scaler_scale=scales, fit_status=np.asarray(statuses))
            checksum = sha256_file(shard_path)
            for row_index, (_, _, status, _) in enumerate(task_records):
                index_rows.append({
                    "analysis_name": analysis_name, "layer": status["layer"],
                    "train_gate_aa_pair": status["train_gate_aa_pair"],
                    "train_gate_sequence_bin": status["train_gate_sequence_bin"],
                    "task_id": task_id, "npz_file": f"coefficient_shards/{shard_path.name}",
                    "npz_row": row_index, "feature_dim": FEATURE_DIM,
                    "fit_status": status["fit_status"], "sha256": checksum,
                })
        atomic_write_csv(output / "coefficient_index.csv", COEFFICIENT_INDEX_FIELDNAMES, index_rows)
    _write_summary(output / "summary.md", analysis_name, status_rows, metric_rows)


def main():
    mode = run_mode()
    root = suite_root(mode)
    destination = consolidated_root(mode)
    log(f"Aggregating {root} -> {destination}")
    preparation_files = []
    for filename in ("run_manifest.json", "split_assignments.csv", "skipped_proteins.csv", "excluded_pairs_summary.csv"):
        source = root / filename
        if not source.exists():
            raise ValueError(f"Missing preparation output: {source}")
        preparation_files.append((source, destination / filename))
    analysis3_source = root / "analysis_3_true_distance_statistics"
    if not analysis3_source.exists():
        raise ValueError("Missing Analysis 3 outputs")

    task_counts = {
        "analysis_1_sequence_gated": len(configured_layers(mode)),
        "analysis_2_aa_pair_gated": len(configured_layers(mode)),
        "analysis_4_aa_sequence_gated": len(configured_layers(mode)) * 6,
    }
    validated = {}
    for analysis_name, task_count in task_counts.items():
        validated[analysis_name] = _load_analysis(root, analysis_name, task_count)
    raw_baseline = root / "analysis_1_sequence_gated" / "raw_sequence_baseline_summary.csv"
    if not raw_baseline.exists():
        raise ValueError("Missing raw sequence baseline summary")

    destination.mkdir(parents=True, exist_ok=True)
    for source, target in preparation_files:
        shutil.copy2(source, target)
    shutil.copytree(analysis3_source, destination / "analysis_3_true_distance_statistics", dirs_exist_ok=True)
    for analysis_name, loaded in validated.items():
        _publish_analysis(destination, analysis_name, *loaded)
    shutil.copy2(raw_baseline, destination / "analysis_1_sequence_gated" / raw_baseline.name)
    log("Aggregation completed successfully")


if __name__ == "__main__":
    main()

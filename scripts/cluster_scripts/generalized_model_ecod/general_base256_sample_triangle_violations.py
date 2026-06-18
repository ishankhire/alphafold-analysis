#!/usr/bin/env python3
"""
Replay general_base256_combined coefficients and compute exact triangle metrics.

This script does not refit regression models. It reconstructs the predicted
pairwise distance matrix for each protein/layer, calls the shared exact
triangle-violation helper, and writes one triangle-only metrics row per protein.
"""

import csv
import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(CLUSTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_SCRIPTS_DIR))

from cluster_regression_helpers import (  # noqa: E402
    TRIANGLE_VIOLATION_KEYS,
    compute_triangle_violations,
)
from general_train_test_common import (  # noqa: E402
    N_CHANNELS,
    N_LAYERS,
    SPLIT_ID,
    build_dataset_protein_manifest,
    feature_names_for_variant,
    format_layer_indices,
    log_progress,
    make_variant,
    resolve_pair_block_path,
)


VARIANT = make_variant("general_base256_combined", pair_mode="concat_directional")
VARIANT_NAME = VARIANT["name"]
FEATURE_DIM = 2 * N_CHANNELS
PAIR_CHUNK_SIZE = 250_000

COEFFICIENTS_CSV_NAME = "coefficients.csv"
SPLIT_ASSIGNMENTS_CSV_NAME = "split_assignments.csv"
OUTPUT_SUBDIR_NAME = "exact_triangle_metrics"
SUMMARY_TXT_NAME = "summary.txt"

TRIANGLE_METRIC_FIELDNAMES = [
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "n_triangles",
    "n_negative_predictions",
    *TRIANGLE_VIOLATION_KEYS,
    "elapsed_seconds",
]


def format_csv_value(value):
    """Format numeric CSV values consistently with the regression helpers."""
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_csv_rows(path, fieldnames, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_csv_value(row.get(key, "")) for key in fieldnames})


def normalize_requested_layers():
    """Use one Slurm array task as one layer; run all layers outside arrays."""
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None or task_id == "":
        return list(range(N_LAYERS))

    layer = int(task_id)
    if layer < 0 or layer >= N_LAYERS:
        raise ValueError(f"SLURM_ARRAY_TASK_ID={layer} is outside layer range 0-{N_LAYERS - 1}")
    return [layer]


def load_coefficients(coefficient_path, requested_layers):
    expected_feature_names = feature_names_for_variant(VARIANT)
    if len(expected_feature_names) != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {len(expected_feature_names)}")

    requested = set(int(layer) for layer in requested_layers)
    bundles = {
        layer: {
            "intercept": None,
            "coef": np.full((FEATURE_DIM,), np.nan, dtype=np.float64),
            "feature_names": [None] * FEATURE_DIM,
            "feature_count": 0,
            "intercept_count": 0,
        }
        for layer in requested
    }

    with coefficient_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if row["variant"] != VARIANT_NAME:
                continue
            if int(row["split_id"]) != SPLIT_ID:
                continue
            layer = int(row["layer"])
            if layer not in bundles:
                continue
            feature_dim = int(row["feature_dim"])
            if feature_dim != FEATURE_DIM:
                raise ValueError(
                    f"Layer {layer} has feature_dim={feature_dim}; expected {FEATURE_DIM}"
                )

            feature_index = int(row["feature_index"])
            raw_value = float(row["coefficient_raw_scale"])
            bundle = bundles[layer]
            if feature_index == -1:
                bundle["intercept"] = raw_value
                bundle["intercept_count"] += 1
            else:
                if feature_index < 0 or feature_index >= FEATURE_DIM:
                    raise ValueError(f"Unexpected feature index {feature_index} at layer {layer}")
                bundle["coef"][feature_index] = raw_value
                bundle["feature_names"][feature_index] = row["feature_name"]
                bundle["feature_count"] += 1

    for layer in requested_layers:
        bundle = bundles[int(layer)]
        if bundle["intercept_count"] != 1:
            raise ValueError(
                f"Layer {layer} expected one intercept row, found {bundle['intercept_count']}"
            )
        if bundle["feature_count"] != FEATURE_DIM:
            raise ValueError(
                f"Layer {layer} expected {FEATURE_DIM} coefficient rows, "
                f"found {bundle['feature_count']}"
            )
        if bundle["feature_names"] != expected_feature_names:
            raise ValueError(f"Layer {layer} feature names do not match {VARIANT_NAME}")
        if not np.all(np.isfinite(bundle["coef"])):
            raise ValueError(f"Layer {layer} has non-finite raw coefficients")

        bundle["coef_ij"] = np.asarray(bundle["coef"][:N_CHANNELS], dtype=np.float64)
        bundle["coef_ji"] = np.asarray(bundle["coef"][N_CHANNELS:], dtype=np.float64)

    return bundles


def load_assignment_records(assignment_path):
    manifest_by_id = {
        entry["protein_id"]: entry
        for entry in build_dataset_protein_manifest()
    }
    records = []
    seen = set()

    with assignment_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["split_id"]) != SPLIT_ID:
                continue
            protein_id = row["protein_id"]
            if protein_id in seen:
                raise ValueError(f"Duplicate split assignment for {protein_id}")
            if protein_id not in manifest_by_id:
                raise ValueError(f"{protein_id} is absent from the combined manifest")

            manifest_entry = manifest_by_id[protein_id]
            n_residues = int(row["n_residues"])
            records.append(
                {
                    "split": row["split"],
                    "protein_id": protein_id,
                    "source": manifest_entry["source"],
                    "protein_dir": manifest_entry["protein_dir"],
                    "n_residues": n_residues,
                    "size_bin": int(row["size_bin"]),
                    "n_pairs": int(row["n_pairs"]),
                    "n_triangles": n_residues * (n_residues - 1) * (n_residues - 2) // 6,
                }
            )
            seen.add(protein_id)

    if not records:
        raise ValueError(f"No split_id={SPLIT_ID} rows found in {assignment_path}")
    return records


def predict_all_distances(pb, record, coefficient_bundle):
    n_residues = int(record["n_residues"])
    pair_i, pair_j = np.tril_indices(n_residues, k=-1)
    if pair_i.size != int(record["n_pairs"]):
        raise ValueError(
            f"{record['protein_id']} expected {record['n_pairs']} pairs, "
            f"built {pair_i.size}"
        )

    pred_dist = np.zeros((n_residues, n_residues), dtype=np.float64)
    coef_ij = coefficient_bundle["coef_ij"]
    coef_ji = coefficient_bundle["coef_ji"]
    intercept = float(coefficient_bundle["intercept"])
    n_negative = 0

    for start in range(0, pair_i.size, PAIR_CHUNK_SIZE):
        end = min(start + PAIR_CHUNK_SIZE, pair_i.size)
        idx_i = pair_i[start:end]
        idx_j = pair_j[start:end]
        y_pred = np.full((end - start,), intercept, dtype=np.float64)
        y_pred += np.asarray(pb[idx_i, idx_j, :], dtype=np.float64) @ coef_ij
        y_pred += np.asarray(pb[idx_j, idx_i, :], dtype=np.float64) @ coef_ji
        n_negative += int(np.sum(y_pred < 0.0))
        y_pred = np.maximum(y_pred, 0.0)
        pred_dist[idx_i, idx_j] = y_pred
        pred_dist[idx_j, idx_i] = y_pred

    return pred_dist, n_negative


def compute_record_layer_metrics(record, layer, coefficient_bundle):
    protein_t0 = time.time()
    pb_path = resolve_pair_block_path(record["protein_dir"], record["protein_id"], layer)
    if not pb_path.exists():
        raise FileNotFoundError(f"Missing pair block for {record['protein_id']} layer {layer}: {pb_path}")

    pb = np.load(str(pb_path))
    expected_shape = (int(record["n_residues"]), int(record["n_residues"]), N_CHANNELS)
    if tuple(pb.shape) != expected_shape:
        raise ValueError(
            f"{record['protein_id']} layer {layer} pair block shape {pb.shape} "
            f"!= expected {expected_shape}"
        )

    pred_dist, n_negative = predict_all_distances(pb, record, coefficient_bundle)
    del pb

    triangle_metrics = compute_triangle_violations(pred_dist)
    del pred_dist

    return {
        "variant": VARIANT_NAME,
        "feature_dim": FEATURE_DIM,
        "split_id": SPLIT_ID,
        "layer": int(layer),
        "split": record["split"],
        "protein_id": record["protein_id"],
        "n_residues": int(record["n_residues"]),
        "size_bin": int(record["size_bin"]),
        "n_pairs": int(record["n_pairs"]),
        "n_triangles": int(record["n_triangles"]),
        "n_negative_predictions": int(n_negative),
        **triangle_metrics,
        "elapsed_seconds": float(time.time() - protein_t0),
    }


def process_layer(layer, records, coefficient_bundle, output_dir):
    layer_t0 = time.time()
    tmp_path = output_dir / f"exact_triangle_metrics_layer_{layer:02d}.csv.tmp"
    final_path = output_dir / f"exact_triangle_metrics_layer_{layer:02d}.csv"
    rows = []

    log_progress(f"Layer {layer:02d}: starting exact triangle replay for {len(records)} proteins")
    for protein_index, record in enumerate(records, start=1):
        rows.append(compute_record_layer_metrics(record, layer, coefficient_bundle))
        if protein_index % 10 == 0 or protein_index == len(records):
            elapsed = time.time() - layer_t0
            log_progress(
                f"Layer {layer:02d}: processed {protein_index}/{len(records)} proteins "
                f"in {elapsed:.1f}s"
            )
        gc.collect()

    write_csv_rows(tmp_path, TRIANGLE_METRIC_FIELDNAMES, rows)
    os.replace(tmp_path, final_path)
    layer_elapsed = time.time() - layer_t0
    log_progress(
        f"Layer {layer:02d}: wrote {final_path.name} "
        f"({len(rows)} proteins) in {layer_elapsed:.1f}s"
    )
    return final_path, layer_elapsed


def write_summary_text(path, csv_root, output_dir, records, completed_layers, elapsed_seconds):
    source_counts = {}
    split_counts = {}
    for record in records:
        source_counts[record["source"]] = source_counts.get(record["source"], 0) + 1
        split_counts[record["split"]] = split_counts.get(record["split"], 0) + 1

    with path.open("w") as handle:
        handle.write("general_base256_combined exact triangle violation metrics\n")
        handle.write("========================================================\n\n")
        handle.write(f"CSV root:          {csv_root}\n")
        handle.write(f"Output dir:        {output_dir}\n")
        handle.write(f"Variant:           {VARIANT_NAME}\n")
        handle.write(f"Feature dimension: {FEATURE_DIM}\n")
        handle.write(f"Split id:          {SPLIT_ID}\n")
        handle.write(f"Proteins:          {len(records)}\n")
        handle.write(f"Proteins by split: {split_counts}\n")
        handle.write(f"Proteins by source:{source_counts}\n")
        handle.write(f"Completed layers:  {format_layer_indices(completed_layers) if completed_layers else 'none'}\n")
        handle.write(f"Elapsed seconds:   {elapsed_seconds:.2f}\n\n")
        handle.write("Each layer shard has one triangle-only metrics row per protein.\n")
        handle.write("Distances are clipped at zero before exact unordered-triangle metrics.\n")


def main():
    t0 = time.time()
    csv_root = Path(os.environ.get("CSV_ROOT") or (Path.home() / "csv_files")).expanduser().resolve()
    variant_dir = csv_root / VARIANT_NAME
    output_dir = variant_dir / OUTPUT_SUBDIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    requested_layers = normalize_requested_layers()
    coefficient_path = variant_dir / COEFFICIENTS_CSV_NAME
    assignment_path = variant_dir / SPLIT_ASSIGNMENTS_CSV_NAME

    log_progress("Starting exact triangle violation replay")
    log_progress(f"CSV root:        {csv_root}")
    log_progress(f"Coefficient CSV: {coefficient_path}")
    log_progress(f"Assignments CSV: {assignment_path}")
    log_progress(f"Output dir:      {output_dir}")
    log_progress(f"Layers:          {format_layer_indices(requested_layers)}")

    coefficient_bundles = load_coefficients(coefficient_path, requested_layers)
    records = load_assignment_records(assignment_path)
    log_progress(f"Loaded split-assignment proteins: {len(records)}")

    completed_layers = []
    write_summary_text(
        output_dir / SUMMARY_TXT_NAME,
        csv_root,
        output_dir,
        records,
        completed_layers,
        time.time() - t0,
    )

    for layer in requested_layers:
        process_layer(layer, records, coefficient_bundles[int(layer)], output_dir)
        completed_layers.append(int(layer))
        write_summary_text(
            output_dir / SUMMARY_TXT_NAME,
            csv_root,
            output_dir,
            records,
            completed_layers,
            time.time() - t0,
        )
        gc.collect()

    log_progress(f"Completed exact triangle replay in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

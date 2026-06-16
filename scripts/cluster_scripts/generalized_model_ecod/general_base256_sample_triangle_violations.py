#!/usr/bin/env python3
"""
Sample granular triangle inequality violations for held-out proteins.

This replays saved raw-scale general_base256_combined coefficients onto sampled
residue triples. It does not refit regression models or run MDS.
"""

import csv
import gc
import hashlib
import math
import os
import sys
import time
from itertools import combinations
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from general_train_test_common import (  # noqa: E402
    N_CHANNELS,
    RANDOM_STATE,
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
LAYERS = (0, 10, 20, 30, 40, 47)
TRIANGLES_PER_PROTEIN = 2000
SPLIT_NAME = "test"

COEFFICIENTS_CSV_NAME = "coefficients.csv"
SPLIT_ASSIGNMENTS_CSV_NAME = "split_assignments.csv"
OUTPUT_SUBDIR_NAME = "triangle_violation_samples_test"
SUMMARY_CSV_NAME = "triangle_sample_summary.csv"
SUMMARY_TXT_NAME = "summary.txt"

TRIANGLE_FIELDNAMES = [
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "sample_index",
    "residue_a",
    "residue_b",
    "residue_c",
    "pred_ab_raw",
    "pred_ac_raw",
    "pred_bc_raw",
    "dist_ab",
    "dist_ac",
    "dist_bc",
    "margin_ab_le_ac_bc",
    "margin_ac_le_ab_bc",
    "margin_bc_le_ab_ac",
    "violation_ab",
    "violation_ac",
    "violation_bc",
    "max_violation",
    "n_violated_inequalities",
    "negative_edge_count",
]

SUMMARY_FIELDNAMES = [
    "scope",
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_triangles_sampled",
    "n_inequalities",
    "n_violated_inequalities",
    "triangle_with_any_violation_fraction",
    "inequality_violation_fraction",
    "mean_positive_violation",
    "max_violation",
    "mean_max_violation_per_triangle",
    "n_negative_edges",
    "negative_edge_fraction",
    "elapsed_seconds",
]


def format_csv_value(value):
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, (int, np.integer)):
        return int(value)
    return value


def write_summary_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {key: format_csv_value(row.get(key, "")) for key in SUMMARY_FIELDNAMES}
            )


def stable_seed(protein_id):
    payload = f"{RANDOM_STATE}:{protein_id}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def sample_triangles(n_residues, protein_id, sample_size):
    total = n_residues * (n_residues - 1) * (n_residues - 2) // 6
    if total <= 0:
        return np.empty((0, 3), dtype=np.int32)
    if total <= sample_size:
        return np.asarray(list(combinations(range(n_residues), 3)), dtype=np.int32)

    rng = np.random.default_rng(stable_seed(protein_id))
    seen = set()
    out = []
    while len(out) < sample_size:
        remaining = sample_size - len(out)
        draw_size = max(1024, remaining * 3)
        draws = rng.integers(0, n_residues, size=(draw_size, 3), dtype=np.int32)
        draws.sort(axis=1)
        valid = (draws[:, 0] < draws[:, 1]) & (draws[:, 1] < draws[:, 2])
        for a, b, c in draws[valid]:
            key = (int(a), int(b), int(c))
            if key in seen:
                continue
            seen.add(key)
            out.append(key)
            if len(out) == sample_size:
                break

    return np.asarray(out, dtype=np.int32)


def load_coefficients(coefficient_path):
    expected_feature_names = feature_names_for_variant(VARIANT)
    if len(expected_feature_names) != FEATURE_DIM:
        raise ValueError(f"Expected feature dim {FEATURE_DIM}, got {len(expected_feature_names)}")

    bundles = {
        layer: {
            "intercept": None,
            "coef": np.full((FEATURE_DIM,), np.nan, dtype=np.float64),
            "feature_names": [None] * FEATURE_DIM,
        }
        for layer in LAYERS
    }
    feature_counts = {layer: 0 for layer in LAYERS}
    intercept_counts = {layer: 0 for layer in LAYERS}

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
            if feature_index == -1:
                bundles[layer]["intercept"] = raw_value
                intercept_counts[layer] += 1
            else:
                if feature_index < 0 or feature_index >= FEATURE_DIM:
                    raise ValueError(f"Unexpected feature index {feature_index} at layer {layer}")
                bundles[layer]["coef"][feature_index] = raw_value
                bundles[layer]["feature_names"][feature_index] = row["feature_name"]
                feature_counts[layer] += 1

    for layer in LAYERS:
        bundle = bundles[layer]
        if intercept_counts[layer] != 1:
            raise ValueError(
                f"Layer {layer} expected one intercept row, found {intercept_counts[layer]}"
            )
        if feature_counts[layer] != FEATURE_DIM:
            raise ValueError(
                f"Layer {layer} expected {FEATURE_DIM} coefficient rows, "
                f"found {feature_counts[layer]}"
            )
        if bundle["feature_names"] != expected_feature_names:
            raise ValueError(f"Layer {layer} feature names do not match {VARIANT_NAME}")
        if not np.all(np.isfinite(bundle["coef"])):
            raise ValueError(f"Layer {layer} has non-finite raw coefficients")

        bundle["coef_ij"] = np.asarray(bundle["coef"][:N_CHANNELS], dtype=np.float64)
        bundle["coef_ji"] = np.asarray(bundle["coef"][N_CHANNELS:], dtype=np.float64)

    return bundles


def load_test_records(assignment_path):
    manifest_by_id = {
        entry["protein_id"]: entry
        for entry in build_dataset_protein_manifest()
    }
    rows = []
    seen = set()

    with assignment_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["split_id"]) != SPLIT_ID:
                continue
            if row["split"] != SPLIT_NAME:
                continue
            protein_id = row["protein_id"]
            if protein_id in seen:
                raise ValueError(f"Duplicate held-out assignment for {protein_id}")
            if protein_id not in manifest_by_id:
                raise ValueError(f"{protein_id} is absent from the combined manifest")

            manifest_entry = manifest_by_id[protein_id]
            n_residues = int(row["n_residues"])
            rows.append(
                {
                    "split": row["split"],
                    "protein_id": protein_id,
                    "source": manifest_entry["source"],
                    "protein_dir": manifest_entry["protein_dir"],
                    "n_residues": n_residues,
                    "size_bin": int(row["size_bin"]),
                    "n_pairs": int(row["n_pairs"]),
                    "triangles": sample_triangles(
                        n_residues,
                        protein_id,
                        TRIANGLES_PER_PROTEIN,
                    ),
                }
            )
            seen.add(protein_id)

    if not rows:
        raise ValueError(f"No split={SPLIT_NAME!r} rows found in {assignment_path}")
    return rows


def predict_edges(pb, idx_a, idx_b, coefficient_bundle):
    coef_ij = coefficient_bundle["coef_ij"]
    coef_ji = coefficient_bundle["coef_ji"]
    pred = np.full((idx_a.size,), float(coefficient_bundle["intercept"]), dtype=np.float64)
    pred += np.asarray(pb[idx_a, idx_b, :], dtype=np.float64) @ coef_ij
    pred += np.asarray(pb[idx_b, idx_a, :], dtype=np.float64) @ coef_ji
    return pred


def compute_triangle_arrays(pb, triangles, coefficient_bundle):
    a = triangles[:, 0]
    b = triangles[:, 1]
    c = triangles[:, 2]

    pred_ab = predict_edges(pb, a, b, coefficient_bundle)
    pred_ac = predict_edges(pb, a, c, coefficient_bundle)
    pred_bc = predict_edges(pb, b, c, coefficient_bundle)

    negative_edge_count = (
        (pred_ab < 0.0).astype(np.int16)
        + (pred_ac < 0.0).astype(np.int16)
        + (pred_bc < 0.0).astype(np.int16)
    )

    dist_ab = np.maximum(pred_ab, 0.0)
    dist_ac = np.maximum(pred_ac, 0.0)
    dist_bc = np.maximum(pred_bc, 0.0)

    margin_ab = dist_ab - (dist_ac + dist_bc)
    margin_ac = dist_ac - (dist_ab + dist_bc)
    margin_bc = dist_bc - (dist_ab + dist_ac)

    violation_ab = np.maximum(margin_ab, 0.0)
    violation_ac = np.maximum(margin_ac, 0.0)
    violation_bc = np.maximum(margin_bc, 0.0)

    max_violation = np.maximum.reduce([violation_ab, violation_ac, violation_bc])
    n_violated = (
        (violation_ab > 0.0).astype(np.int16)
        + (violation_ac > 0.0).astype(np.int16)
        + (violation_bc > 0.0).astype(np.int16)
    )

    return {
        "pred_ab": pred_ab,
        "pred_ac": pred_ac,
        "pred_bc": pred_bc,
        "dist_ab": dist_ab,
        "dist_ac": dist_ac,
        "dist_bc": dist_bc,
        "margin_ab": margin_ab,
        "margin_ac": margin_ac,
        "margin_bc": margin_bc,
        "violation_ab": violation_ab,
        "violation_ac": violation_ac,
        "violation_bc": violation_bc,
        "max_violation": max_violation,
        "n_violated": n_violated,
        "negative_edge_count": negative_edge_count,
    }


def summarize_protein(layer, record, arrays, elapsed_seconds):
    n_triangles = int(record["triangles"].shape[0])
    n_inequalities = 3 * n_triangles
    n_violated = int(np.sum(arrays["n_violated"], dtype=np.int64))
    n_negative_edges = int(np.sum(arrays["negative_edge_count"], dtype=np.int64))
    max_violation = np.asarray(arrays["max_violation"], dtype=np.float64)
    positive_values = np.concatenate(
        [
            arrays["violation_ab"][arrays["violation_ab"] > 0.0],
            arrays["violation_ac"][arrays["violation_ac"] > 0.0],
            arrays["violation_bc"][arrays["violation_bc"] > 0.0],
        ]
    )

    positive_sum = float(np.sum(positive_values, dtype=np.float64))
    positive_count = int(positive_values.size)
    max_sum = float(np.sum(max_violation, dtype=np.float64))
    triangles_with_any = int(np.sum(max_violation > 0.0, dtype=np.int64))

    return {
        "scope": "protein",
        "variant": VARIANT_NAME,
        "feature_dim": FEATURE_DIM,
        "split_id": SPLIT_ID,
        "layer": int(layer),
        "split": SPLIT_NAME,
        "protein_id": record["protein_id"],
        "n_residues": int(record["n_residues"]),
        "size_bin": int(record["size_bin"]),
        "n_triangles_sampled": n_triangles,
        "n_inequalities": n_inequalities,
        "n_violated_inequalities": n_violated,
        "triangle_with_any_violation_fraction": (
            float(triangles_with_any / n_triangles) if n_triangles else float("nan")
        ),
        "inequality_violation_fraction": (
            float(n_violated / n_inequalities) if n_inequalities else float("nan")
        ),
        "mean_positive_violation": (
            float(positive_sum / positive_count) if positive_count else 0.0
        ),
        "max_violation": float(np.max(max_violation)) if n_triangles else 0.0,
        "mean_max_violation_per_triangle": (
            float(max_sum / n_triangles) if n_triangles else float("nan")
        ),
        "n_negative_edges": n_negative_edges,
        "negative_edge_fraction": (
            float(n_negative_edges / (3 * n_triangles)) if n_triangles else float("nan")
        ),
        "elapsed_seconds": float(elapsed_seconds),
        "_positive_violation_sum": positive_sum,
        "_positive_violation_count": positive_count,
        "_max_violation_sum": max_sum,
        "_triangles_with_any_violation": triangles_with_any,
    }


def summarize_layer(layer, protein_summaries, elapsed_seconds):
    n_triangles = int(sum(row["n_triangles_sampled"] for row in protein_summaries))
    n_inequalities = int(sum(row["n_inequalities"] for row in protein_summaries))
    n_violated = int(sum(row["n_violated_inequalities"] for row in protein_summaries))
    n_negative_edges = int(sum(row["n_negative_edges"] for row in protein_summaries))
    positive_sum = float(sum(row["_positive_violation_sum"] for row in protein_summaries))
    positive_count = int(sum(row["_positive_violation_count"] for row in protein_summaries))
    max_sum = float(sum(row["_max_violation_sum"] for row in protein_summaries))
    triangles_with_any = int(
        sum(row["_triangles_with_any_violation"] for row in protein_summaries)
    )

    max_values = [
        float(row["max_violation"])
        for row in protein_summaries
        if row["n_triangles_sampled"] > 0
    ]

    return {
        "scope": "layer",
        "variant": VARIANT_NAME,
        "feature_dim": FEATURE_DIM,
        "split_id": SPLIT_ID,
        "layer": int(layer),
        "split": SPLIT_NAME,
        "protein_id": "ALL",
        "n_residues": "",
        "size_bin": "",
        "n_triangles_sampled": n_triangles,
        "n_inequalities": n_inequalities,
        "n_violated_inequalities": n_violated,
        "triangle_with_any_violation_fraction": (
            float(triangles_with_any / n_triangles) if n_triangles else float("nan")
        ),
        "inequality_violation_fraction": (
            float(n_violated / n_inequalities) if n_inequalities else float("nan")
        ),
        "mean_positive_violation": (
            float(positive_sum / positive_count) if positive_count else 0.0
        ),
        "max_violation": max(max_values) if max_values else 0.0,
        "mean_max_violation_per_triangle": (
            float(max_sum / n_triangles) if n_triangles else float("nan")
        ),
        "n_negative_edges": n_negative_edges,
        "negative_edge_fraction": (
            float(n_negative_edges / (3 * n_triangles)) if n_triangles else float("nan")
        ),
        "elapsed_seconds": float(elapsed_seconds),
    }


def write_triangle_rows(writer, layer, record, arrays):
    triangles = record["triangles"]
    for sample_index, (a, b, c) in enumerate(triangles):
        writer.writerow(
            [
                VARIANT_NAME,
                FEATURE_DIM,
                SPLIT_ID,
                int(layer),
                SPLIT_NAME,
                record["protein_id"],
                int(record["n_residues"]),
                int(record["size_bin"]),
                int(sample_index),
                int(a),
                int(b),
                int(c),
                format_csv_value(arrays["pred_ab"][sample_index]),
                format_csv_value(arrays["pred_ac"][sample_index]),
                format_csv_value(arrays["pred_bc"][sample_index]),
                format_csv_value(arrays["dist_ab"][sample_index]),
                format_csv_value(arrays["dist_ac"][sample_index]),
                format_csv_value(arrays["dist_bc"][sample_index]),
                format_csv_value(arrays["margin_ab"][sample_index]),
                format_csv_value(arrays["margin_ac"][sample_index]),
                format_csv_value(arrays["margin_bc"][sample_index]),
                format_csv_value(arrays["violation_ab"][sample_index]),
                format_csv_value(arrays["violation_ac"][sample_index]),
                format_csv_value(arrays["violation_bc"][sample_index]),
                format_csv_value(arrays["max_violation"][sample_index]),
                int(arrays["n_violated"][sample_index]),
                int(arrays["negative_edge_count"][sample_index]),
            ]
        )


def process_layer(layer, records, coefficient_bundle, output_dir):
    layer_t0 = time.time()
    tmp_path = output_dir / f"triangle_samples_layer_{layer:02d}.csv.tmp"
    final_path = output_dir / f"triangle_samples_layer_{layer:02d}.csv"
    protein_summaries = []

    with tmp_path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(TRIANGLE_FIELDNAMES)

        for protein_index, record in enumerate(records, start=1):
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

            arrays = compute_triangle_arrays(pb, record["triangles"], coefficient_bundle)
            del pb

            write_triangle_rows(writer, layer, record, arrays)
            protein_summaries.append(
                summarize_protein(layer, record, arrays, time.time() - protein_t0)
            )
            del arrays

            if protein_index % 10 == 0 or protein_index == len(records):
                log_progress(
                    f"Layer {layer:02d}: processed {protein_index}/{len(records)} "
                    f"held-out proteins"
                )

    os.replace(tmp_path, final_path)
    layer_elapsed = time.time() - layer_t0
    layer_summary = summarize_layer(layer, protein_summaries, layer_elapsed)
    log_progress(
        f"Layer {layer:02d}: wrote {final_path.name} "
        f"({layer_summary['n_triangles_sampled']:,} triangles) in {layer_elapsed:.1f}s"
    )
    return protein_summaries + [layer_summary]


def write_summary_text(path, csv_root, output_dir, records, completed_layers, elapsed_seconds):
    with path.open("w") as handle:
        handle.write("general_base256_combined held-out sampled triangle violations\n")
        handle.write("============================================================\n\n")
        handle.write(f"CSV root:              {csv_root}\n")
        handle.write(f"Output dir:            {output_dir}\n")
        handle.write(f"Variant:               {VARIANT_NAME}\n")
        handle.write(f"Feature dimension:     {FEATURE_DIM}\n")
        handle.write(f"Split:                 {SPLIT_NAME}\n")
        handle.write(f"Split id:              {SPLIT_ID}\n")
        handle.write(f"Layers:                {format_layer_indices(LAYERS)}\n")
        handle.write(f"Held-out proteins:     {len(records)}\n")
        handle.write(f"Triangles per protein: {TRIANGLES_PER_PROTEIN}\n")
        handle.write(f"Completed layers:      {format_layer_indices(completed_layers) if completed_layers else 'none'}\n")
        handle.write(f"Elapsed seconds:       {elapsed_seconds:.2f}\n\n")
        handle.write("Triangle distances are clipped at zero before inequality margins, matching\n")
        handle.write("the existing aggregate geometry analysis. Raw predictions are also saved.\n")


def main():
    t0 = time.time()
    csv_root = Path(os.environ.get("CSV_ROOT") or (Path.home() / "csv_files")).expanduser().resolve()
    variant_dir = csv_root / VARIANT_NAME
    output_dir = variant_dir / OUTPUT_SUBDIR_NAME
    output_dir.mkdir(parents=True, exist_ok=True)

    coefficient_path = variant_dir / COEFFICIENTS_CSV_NAME
    assignment_path = variant_dir / SPLIT_ASSIGNMENTS_CSV_NAME

    log_progress("Starting held-out sampled triangle violation replay")
    log_progress(f"CSV root:         {csv_root}")
    log_progress(f"Coefficient CSV:  {coefficient_path}")
    log_progress(f"Assignments CSV:  {assignment_path}")
    log_progress(f"Output dir:       {output_dir}")
    log_progress(f"Layers:           {format_layer_indices(LAYERS)}")
    log_progress(f"Samples/protein:  {TRIANGLES_PER_PROTEIN}")

    coefficient_bundles = load_coefficients(coefficient_path)
    records = load_test_records(assignment_path)
    log_progress(f"Loaded held-out proteins: {len(records)}")

    total_requested = int(sum(record["triangles"].shape[0] for record in records))
    log_progress(f"Sampled triangles per layer: {total_requested:,}")

    all_summary_rows = []
    completed_layers = []
    write_summary_text(
        output_dir / SUMMARY_TXT_NAME,
        csv_root,
        output_dir,
        records,
        completed_layers,
        time.time() - t0,
    )

    for layer in LAYERS:
        layer_rows = process_layer(layer, records, coefficient_bundles[layer], output_dir)
        all_summary_rows.extend(layer_rows)
        completed_layers.append(layer)
        write_summary_csv(output_dir / SUMMARY_CSV_NAME, all_summary_rows)
        write_summary_text(
            output_dir / SUMMARY_TXT_NAME,
            csv_root,
            output_dir,
            records,
            completed_layers,
            time.time() - t0,
        )
        gc.collect()

    log_progress(f"Completed sampled triangle replay in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

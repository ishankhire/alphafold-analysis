#!/usr/bin/env python3
"""Train distance probes selected only by residue sequence separation.

One SLURM array task runs one layer from ``LAYERS``.  Within that layer, the
script trains an independent base256 linear probe for each sequence-separation
bin.  A matched sequence-separation-only linear baseline is fit on the same
rows.  Protein membership, never residue-pair membership, defines the split.

Outputs are written to
``$CSV_ROOT/diagnostic_sequence_separation_gated_probes/layer_batches``:

* ``split_summary_layer_XX.csv``: one row per model and separation bin.
* ``per_protein_layer_XX.csv``: one row per model/bin/split/protein.
* ``coefficients_layer_XX.csv``: intercept and feature coefficients per model/bin.
* ``split_assignments_layer_XX.csv``: one row per available protein.
* ``summary_layer_XX.txt``: configuration, skipped inputs, and timing.
"""

import csv
import gc
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import StratifiedShuffleSplit


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
COMBINED_MODEL_DIR = CLUSTER_SCRIPTS_DIR / "generalized_model_ecod"
for path in (CLUSTER_SCRIPTS_DIR, COMBINED_MODEL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from general_train_test_common import (  # noqa: E402
    FLOAT_EPS,
    N_SIZE_BINS,
    RANDOM_STATE,
    assign_size_bins,
    build_feature_chunk_from_pb,
    build_strict_stratified_train_test_split,
    feature_names_for_variant,
    load_protein_records,
    make_variant,
    resolve_pair_block_path,
)


ANALYSIS_NAME = "diagnostic_sequence_separation_gated_probes"
LAYERS = (0, 10, 20, 30, 40, 47)
MAX_TRAIN_PROTEINS = 100
VARIANT = make_variant("base256", pair_mode="concat_directional")
REPRESENTATION_MODEL = "base256_gated_by_sequence_separation"
BASELINE_MODEL = "sequence_separation_only"
SEPARATION_BINS = (
    (1, 1, "1"),
    (2, 4, "2-4"),
    (5, 9, "5-9"),
    (10, 19, "10-19"),
    (20, 49, "20-49"),
    (50, sys.maxsize, "50+"),
)

SUMMARY_FIELDS = [
    "analysis_name", "model", "feature_dim", "layer", "sequence_separation_bin",
    "sequence_separation_min", "sequence_separation_max", "n_train_proteins",
    "n_test_proteins", "n_train_rows", "n_test_rows", "train_mean_per_protein_r2",
    "test_mean_per_protein_r2", "train_pair_weighted_r2", "test_pair_weighted_r2",
    "train_pair_weighted_mae", "test_pair_weighted_mae", "train_pair_weighted_rmse",
    "test_pair_weighted_rmse", "train_mean_bias", "test_mean_bias", "elapsed_seconds",
]
PER_PROTEIN_FIELDS = [
    "analysis_name", "model", "feature_dim", "layer", "sequence_separation_bin",
    "split", "protein_id", "source", "n_residues", "size_bin", "n_pairs",
    "target_variance", "per_protein_r2", "per_protein_mse", "per_protein_mae",
    "per_protein_rmse", "mean_true_distance", "mean_predicted_distance",
    "mean_error_pred_minus_true",
]
COEFFICIENT_FIELDS = [
    "analysis_name", "model", "feature_dim", "layer", "sequence_separation_bin",
    "feature_index", "feature_name", "coefficient_standardized", "coefficient_raw_scale",
]
ASSIGNMENT_FIELDS = [
    "protein_id", "source", "split", "n_residues", "size_bin", "n_pairs",
    "target_variance",
]


def now_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"[{now_timestamp()}] {message}", flush=True)


def format_value(value):
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_csv(path, fields, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: format_value(row.get(field, "")) for field in fields})
    temporary.replace(path)


def selected_layer():
    raw_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "0")
    task_id = int(raw_task_id)
    if task_id < 0 or task_id >= len(LAYERS):
        raise ValueError(
            f"SLURM_ARRAY_TASK_ID={task_id} is outside 0..{len(LAYERS) - 1}"
        )
    return LAYERS[task_id], task_id


def cap_training_records(train_pool):
    """Select at most 100 training proteins while preserving size strata."""
    if len(train_pool) <= MAX_TRAIN_PROTEINS:
        return list(train_pool), []
    labels = np.asarray([row["size_bin"] for row in train_pool], dtype=np.int32)
    indices = np.arange(len(train_pool), dtype=np.int64)
    splitter = StratifiedShuffleSplit(
        n_splits=1,
        train_size=MAX_TRAIN_PROTEINS,
        random_state=RANDOM_STATE,
    )
    selected_idx, unused_idx = next(splitter.split(indices, labels))
    selected = [train_pool[int(index)] for index in sorted(selected_idx)]
    unused = [train_pool[int(index)] for index in sorted(unused_idx)]
    return selected, unused


def separation_mask(record, lower, upper):
    separation = np.asarray(record["lt_i"] - record["lt_j"], dtype=np.int64)
    return (separation >= lower) & (separation <= upper)


def load_layer_cache(records, layer):
    """Materialize base256 features once for each selected train/test protein."""
    cache = {}
    total_rows = 0
    for index, record in enumerate(records, start=1):
        pair_path = resolve_pair_block_path(
            record["protein_dir"], record["protein_id"], layer
        )
        pair_block = np.load(str(pair_path))
        cache[record["protein_id"]] = build_feature_chunk_from_pb(
            record, pair_block, VARIANT
        )
        del pair_block
        total_rows += int(record["n_pairs"])
        if index % 20 == 0 or index == len(records):
            log(
                f"Layer {layer:02d}: cached {index}/{len(records)} proteins "
                f"({total_rows:,} total pair rows)"
            )
    return cache


def protein_chunks(records, feature_cache, lower, upper, model_name):
    """Return model inputs and targets for proteins that have rows in one gate."""
    chunks = []
    for record in records:
        mask = separation_mask(record, lower, upper)
        if not np.any(mask):
            continue
        if model_name == REPRESENTATION_MODEL:
            x = np.asarray(feature_cache[record["protein_id"]][mask], dtype=np.float32)
        elif model_name == BASELINE_MODEL:
            seqsep = np.asarray(record["lt_i"] - record["lt_j"], dtype=np.float64)
            x = seqsep[mask, None]
        else:
            raise ValueError(f"Unknown model: {model_name}")
        y = np.asarray(record["y_abs"][mask], dtype=np.float64)
        chunks.append((record, x, y))
    return chunks


def compute_protein_equal_scaler(chunks):
    feature_dim = int(chunks[0][1].shape[1])
    mean = np.zeros(feature_dim, dtype=np.float64)
    second = np.zeros(feature_dim, dtype=np.float64)
    target_mean = 0.0
    for _record, x, y in chunks:
        x64 = np.asarray(x, dtype=np.float64)
        mean += np.mean(x64, axis=0, dtype=np.float64)
        second += np.mean(x64 * x64, axis=0, dtype=np.float64)
        target_mean += float(np.mean(y, dtype=np.float64))
    mean /= len(chunks)
    second /= len(chunks)
    target_mean /= len(chunks)
    scale = np.sqrt(np.maximum(second - mean * mean, 0.0))
    scale[scale < FLOAT_EPS] = 1.0
    return mean, scale, target_mean


def materialize_chunks(chunks, mean, scale):
    offsets = []
    total_rows = int(sum(len(y) for _record, _x, y in chunks))
    feature_dim = int(chunks[0][1].shape[1])
    x_all = np.empty((total_rows, feature_dim), dtype=np.float64)
    y_all = np.empty(total_rows, dtype=np.float64)
    weights = np.empty(total_rows, dtype=np.float64)
    start = 0
    for record, x, y in chunks:
        end = start + len(y)
        x_all[start:end] = x
        x_all[start:end] -= mean
        x_all[start:end] /= scale
        y_all[start:end] = y
        weights[start:end] = 1.0 / len(y)
        offsets.append((record, start, end))
        start = end
    weights *= len(weights) / np.sum(weights)
    return x_all, y_all, weights, offsets


def compute_r2(y_true, y_pred):
    ss_res = float(np.sum((y_true - y_pred) ** 2, dtype=np.float64))
    centered = y_true - float(np.mean(y_true, dtype=np.float64))
    ss_tot = float(np.sum(centered * centered, dtype=np.float64))
    return float(1.0 - ss_res / ss_tot) if ss_tot > FLOAT_EPS else float("nan")


def summarize_predictions(offsets, y_true, y_pred):
    per_rows = []
    finite_r2 = []
    sum_y = 0.0
    sum_y2 = 0.0
    sum_abs = 0.0
    sum_sqerr = 0.0
    total_n = 0
    for record, start, end in offsets:
        truth = y_true[start:end]
        pred = y_pred[start:end]
        error = pred - truth
        r2 = compute_r2(truth, pred)
        if math.isfinite(r2):
            finite_r2.append(r2)
        per_rows.append(
            {
                "protein_id": record["protein_id"],
                "source": record["source"],
                "n_residues": record["n_residues"],
                "size_bin": record["size_bin"],
                "n_pairs": len(truth),
                "target_variance": float(np.var(truth, dtype=np.float64)),
                "per_protein_r2": r2,
                "per_protein_mse": float(np.mean(error * error)),
                "per_protein_mae": float(np.mean(np.abs(error))),
                "per_protein_rmse": float(np.sqrt(np.mean(error * error))),
                "mean_true_distance": float(np.mean(truth)),
                "mean_predicted_distance": float(np.mean(pred)),
                "mean_error_pred_minus_true": float(np.mean(error)),
            }
        )
        sum_y += float(np.sum(truth, dtype=np.float64))
        sum_y2 += float(np.sum(truth * truth, dtype=np.float64))
        sum_abs += float(np.sum(np.abs(error), dtype=np.float64))
        sum_sqerr += float(np.sum(error * error, dtype=np.float64))
        total_n += len(truth)
    ss_tot = sum_y2 - sum_y * sum_y / total_n
    return {
        "n_proteins": len(offsets),
        "n_rows": total_n,
        "mean_per_protein_r2": float(np.mean(finite_r2)) if finite_r2 else float("nan"),
        "pair_weighted_r2": 1.0 - sum_sqerr / ss_tot if ss_tot > FLOAT_EPS else float("nan"),
        "pair_weighted_mae": sum_abs / total_n,
        "pair_weighted_rmse": math.sqrt(sum_sqerr / total_n),
        "mean_bias": float(np.mean([row["mean_error_pred_minus_true"] for row in per_rows])),
        "per_rows": per_rows,
    }


def fit_and_evaluate(model_name, train_chunks, test_chunks):
    mean, scale, target_mean = compute_protein_equal_scaler(train_chunks)
    x_train, y_train, weights, train_offsets = materialize_chunks(
        train_chunks, mean, scale
    )
    model = LinearRegression(fit_intercept=False, copy_X=False)
    model.fit(x_train, y_train - target_mean, sample_weight=weights)
    coef_std = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    # Sample-weight processing may rescale X in place when copy_X=False.
    # Rebuild it before scoring so train metrics use the original inputs.
    del x_train, weights, model
    x_train, _unused_y, _unused_weights, _unused_offsets = materialize_chunks(
        train_chunks, mean, scale
    )
    train_pred = np.asarray(x_train @ coef_std + target_mean, dtype=np.float64)
    train_metrics = summarize_predictions(train_offsets, y_train, train_pred)
    del x_train, y_train, _unused_y, _unused_weights, _unused_offsets, train_pred

    x_test, y_test, _unused_weights, test_offsets = materialize_chunks(
        test_chunks, mean, scale
    )
    test_pred = np.asarray(x_test @ coef_std + target_mean, dtype=np.float64)
    test_metrics = summarize_predictions(test_offsets, y_test, test_pred)
    del x_test, y_test, _unused_weights, test_pred

    coef_raw = coef_std / scale
    intercept_raw = float(target_mean - np.sum(coef_std * mean / scale))
    feature_names = (
        feature_names_for_variant(VARIANT)
        if model_name == REPRESENTATION_MODEL
        else ["sequence_separation_raw"]
    )
    return train_metrics, test_metrics, coef_std, coef_raw, target_mean, intercept_raw, feature_names


def attach_per_protein_metadata(rows, model_name, feature_dim, layer, label, split):
    output = []
    for row in rows:
        output.append(
            {
                "analysis_name": ANALYSIS_NAME,
                "model": model_name,
                "feature_dim": feature_dim,
                "layer": layer,
                "sequence_separation_bin": label,
                "split": split,
                **row,
            }
        )
    return output


def run():
    started = time.time()
    layer, task_id = selected_layer()
    csv_root = Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files")))
    output_dir = csv_root / ANALYSIS_NAME / "layer_batches"
    suffix = f"layer_{layer:02d}"
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Starting array task {task_id} for layer {layer}")
    all_manifest, records, skipped = load_protein_records()
    if not records:
        raise RuntimeError("No combined CAMEO+ECOD proteins loaded")
    assign_size_bins(records, N_SIZE_BINS)
    train_pool, test_records, _base_assignments, size_bin_counts = (
        build_strict_stratified_train_test_split(records)
    )
    train_records, unused_train_records = cap_training_records(train_pool)
    active_records = train_records + test_records
    log(
        f"Loaded {len(records)} proteins; using {len(train_records)} train and "
        f"{len(test_records)} held-out test proteins; size bins={size_bin_counts}"
    )

    split_lookup = {row["protein_id"]: "train" for row in train_records}
    split_lookup.update({row["protein_id"]: "test" for row in test_records})
    split_lookup.update(
        {row["protein_id"]: "unused_train_pool" for row in unused_train_records}
    )
    assignment_rows = [
        {
            "protein_id": row["protein_id"],
            "source": row["source"],
            "split": split_lookup[row["protein_id"]],
            "n_residues": row["n_residues"],
            "size_bin": row["size_bin"],
            "n_pairs": row["n_pairs"],
            "target_variance": row["target_variance"],
        }
        for row in records
    ]
    write_csv(output_dir / f"split_assignments_{suffix}.csv", ASSIGNMENT_FIELDS, assignment_rows)

    cache = load_layer_cache(active_records, layer)
    summary_rows = []
    per_protein_rows = []
    coefficient_rows = []
    completed_bins = []

    for lower, upper, label in SEPARATION_BINS:
        bin_started = time.time()
        log(f"Layer {layer:02d}, separation {label}: starting")
        for model_name in (REPRESENTATION_MODEL, BASELINE_MODEL):
            model_started = time.time()
            train_chunks = protein_chunks(train_records, cache, lower, upper, model_name)
            test_chunks = protein_chunks(test_records, cache, lower, upper, model_name)
            if not train_chunks or not test_chunks:
                raise RuntimeError(f"No rows available for {model_name}, separation {label}")
            result = fit_and_evaluate(model_name, train_chunks, test_chunks)
            train_metrics, test_metrics, coef_std, coef_raw, intercept_std, intercept_raw, names = result
            feature_dim = len(names)
            summary_rows.append(
                {
                    "analysis_name": ANALYSIS_NAME,
                    "model": model_name,
                    "feature_dim": feature_dim,
                    "layer": layer,
                    "sequence_separation_bin": label,
                    "sequence_separation_min": lower,
                    "sequence_separation_max": "inf" if upper == sys.maxsize else upper,
                    "n_train_proteins": train_metrics["n_proteins"],
                    "n_test_proteins": test_metrics["n_proteins"],
                    "n_train_rows": train_metrics["n_rows"],
                    "n_test_rows": test_metrics["n_rows"],
                    "train_mean_per_protein_r2": train_metrics["mean_per_protein_r2"],
                    "test_mean_per_protein_r2": test_metrics["mean_per_protein_r2"],
                    "train_pair_weighted_r2": train_metrics["pair_weighted_r2"],
                    "test_pair_weighted_r2": test_metrics["pair_weighted_r2"],
                    "train_pair_weighted_mae": train_metrics["pair_weighted_mae"],
                    "test_pair_weighted_mae": test_metrics["pair_weighted_mae"],
                    "train_pair_weighted_rmse": train_metrics["pair_weighted_rmse"],
                    "test_pair_weighted_rmse": test_metrics["pair_weighted_rmse"],
                    "train_mean_bias": train_metrics["mean_bias"],
                    "test_mean_bias": test_metrics["mean_bias"],
                    "elapsed_seconds": time.time() - model_started,
                }
            )
            per_protein_rows.extend(
                attach_per_protein_metadata(
                    train_metrics["per_rows"], model_name, feature_dim, layer, label, "train"
                )
            )
            per_protein_rows.extend(
                attach_per_protein_metadata(
                    test_metrics["per_rows"], model_name, feature_dim, layer, label, "test"
                )
            )
            coefficient_rows.append(
                {
                    "analysis_name": ANALYSIS_NAME,
                    "model": model_name,
                    "feature_dim": feature_dim,
                    "layer": layer,
                    "sequence_separation_bin": label,
                    "feature_index": -1,
                    "feature_name": "intercept",
                    "coefficient_standardized": intercept_std,
                    "coefficient_raw_scale": intercept_raw,
                }
            )
            for index, name in enumerate(names):
                coefficient_rows.append(
                    {
                        "analysis_name": ANALYSIS_NAME,
                        "model": model_name,
                        "feature_dim": feature_dim,
                        "layer": layer,
                        "sequence_separation_bin": label,
                        "feature_index": index,
                        "feature_name": name,
                        "coefficient_standardized": coef_std[index],
                        "coefficient_raw_scale": coef_raw[index],
                    }
                )
            gc.collect()

        completed_bins.append(label)
        write_csv(output_dir / f"split_summary_{suffix}.csv", SUMMARY_FIELDS, summary_rows)
        write_csv(output_dir / f"per_protein_{suffix}.csv", PER_PROTEIN_FIELDS, per_protein_rows)
        write_csv(output_dir / f"coefficients_{suffix}.csv", COEFFICIENT_FIELDS, coefficient_rows)
        log(
            f"Layer {layer:02d}, separation {label}: complete in "
            f"{time.time() - bin_started:.1f}s; partial outputs saved"
        )

    summary_path = output_dir / f"summary_{suffix}.txt"
    summary_path.write_text(
        "\n".join(
            [
                "Sequence-separation-gated distance probe diagnostic",
                f"generated_at={now_timestamp()}",
                f"layer={layer}",
                f"array_task_id={task_id}",
                f"layers={','.join(map(str, LAYERS))}",
                f"separation_bins={','.join(row[2] for row in SEPARATION_BINS)}",
                f"models={REPRESENTATION_MODEL},{BASELINE_MODEL}",
                f"n_manifest_proteins={len(all_manifest)}",
                f"n_loaded_proteins={len(records)}",
                f"n_train_pool_proteins={len(train_pool)}",
                f"n_train_proteins={len(train_records)}",
                f"n_test_proteins={len(test_records)}",
                f"n_unused_train_pool_proteins={len(unused_train_records)}",
                f"random_state={RANDOM_STATE}",
                "weighting=equal_total_weight_per_training_protein_within_each_bin",
                "precision=float64_scaling_and_fit",
                f"completed_bins={','.join(completed_bins)}",
                f"n_skipped_inputs={len(skipped)}",
                f"elapsed_seconds={time.time() - started:.3f}",
                "",
                "Skipped inputs:",
                *[f"{protein_id}: {reason}" for protein_id, reason in skipped],
                "",
            ]
        )
    )
    log(f"Layer {layer:02d}: all outputs complete in {time.time() - started:.1f}s")


if __name__ == "__main__":
    try:
        run()
    except Exception:
        traceback.print_exc()
        raise

#!/usr/bin/env python3
"""
Shared runner for combined CAMEO+ECOD per-protein base256 regressions.

Each fitted model is independent for one protein and one layer. Features use
one unordered residue-pair row with both pair-block directions concatenated:
[pair_i_j, pair_j_i].
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

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
GENERALIZED_ECOD_DIR = CLUSTER_SCRIPTS_DIR / "generalized_model_ecod"
for _path in (CLUSTER_SCRIPTS_DIR, GENERALIZED_ECOD_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cluster_regression_helpers import analyze_protein_geometry  # noqa: E402
from general_train_test_common import (  # noqa: E402
    CAMEO_PAIR_BLOCKS_ROOT,
    CAMEO_PDB_ROOT,
    ECOD_PAIR_BLOCKS_ROOT,
    ECOD_PDB_ROOT,
    FLOAT_EPS,
    N_CHANNELS,
    N_SIZE_BINS,
    assign_size_bins,
    compute_mae,
    compute_r2,
    compute_rmse,
    feature_dim_for_variant,
    feature_names_for_variant,
    format_layer_indices,
    format_value,
    load_protein_records,
    make_variant,
    normalize_layer_indices,
    resolve_pair_block_path,
)


PER_PROTEIN_CSV_NAME = "per_protein.csv"
COEFFICIENTS_CSV_NAME = "coefficients.csv"
GEOMETRY_METRICS_CSV_NAME = "geometry_metrics.csv"
GEOMETRY_SUMMARY_CSV_NAME = "geometry_summary.csv"
SUMMARY_TXT_NAME = "summary.txt"

GEOMETRY_METRIC_KEYS = [
    "per_protein_r2",
    "tm_score",
    "rmsd",
    "stress",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "neg_eigenvalue_fraction",
    "n_negative_predictions",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
]

PER_PROTEIN_FIELDNAMES = [
    "variant",
    "feature_dim",
    "layer",
    "source",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
    "model_rank",
    "model_singular_min",
    "model_singular_max",
    "model_condition_number",
    "per_protein_r2",
    "per_protein_mse",
    "per_protein_mae",
    "per_protein_rmse",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
    "elapsed_seconds",
]

COEFFICIENT_FIELDNAMES = [
    "variant",
    "feature_dim",
    "layer",
    "source",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "feature_index",
    "feature_name",
    "coefficient_standardized",
    "coefficient_raw_scale",
]

GEOMETRY_METRICS_FIELDNAMES = [
    "variant",
    "feature_dim",
    "layer",
    "source",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "per_protein_r2",
    "rmsd",
    "tm_score",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "neg_eigenvalue_fraction",
    "n_negative_predictions",
    "stress",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
]

GEOMETRY_SUMMARY_FIELDNAMES = [
    "variant",
    "feature_dim",
    "layer",
    "n_proteins",
]
for _metric_key in GEOMETRY_METRIC_KEYS:
    GEOMETRY_SUMMARY_FIELDNAMES.extend([f"{_metric_key}_mean", f"{_metric_key}_std"])


def now_timestamp():
    """Return a readable local timestamp for progress logs."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_progress(message):
    """Emit one timestamped progress message."""
    print(f"[{now_timestamp()}] {message}", flush=True)


def resolve_output_filename(base_name, output_filename_suffix=""):
    """Optionally append one stable suffix before the filename extension."""
    if not output_filename_suffix:
        return base_name
    path = Path(base_name)
    return f"{path.stem}_{output_filename_suffix}{path.suffix}"


def write_csv(path, fieldnames, rows):
    """Write complete CSV rows with stable formatting."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def open_streaming_csv(path, fieldnames):
    """Open one streaming CSV writer and emit its header."""
    handle = path.open("w", newline="")
    writer = csv.DictWriter(handle, fieldnames=fieldnames)
    writer.writeheader()
    handle.flush()
    return handle, writer


def write_streaming_row(writer, fieldnames, row):
    """Write one formatted row to an already-open CSV writer."""
    writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def flush_streams(handles):
    """Flush all open CSV handles."""
    for handle in handles:
        handle.flush()


def source_counts(records):
    """Count records by source label."""
    counts = {}
    for record in records:
        source = record.get("source", "unknown")
        counts[source] = counts.get(source, 0) + 1
    return counts


def build_base256_feature_matrix(record, layer):
    """Build one protein/layer base256 matrix in float64."""
    protein_id = record["protein_id"]
    pb_path = resolve_pair_block_path(record["protein_dir"], protein_id, layer)
    if not pb_path.exists():
        raise FileNotFoundError(f"pair block layer {layer} not found at {pb_path}")

    pair_i = record["lt_i"]
    pair_j = record["lt_j"]
    n_pairs = int(record["n_pairs"])
    x = np.empty((n_pairs, 2 * N_CHANNELS), dtype=np.float64)

    pb = np.load(str(pb_path))
    try:
        if pb.ndim != 3 or int(pb.shape[2]) != N_CHANNELS:
            raise ValueError(
                f"unexpected pair block shape for {protein_id} layer {layer}: {pb.shape}"
            )
        if int(pb.shape[0]) != int(record["n_residues"]) or int(pb.shape[1]) != int(
            record["n_residues"]
        ):
            raise ValueError(
                f"pair block dim {pb.shape[:2]} != {record['n_residues']} residues "
                f"for {protein_id} layer {layer}"
            )

        x[:, :N_CHANNELS] = pb[pair_i, pair_j, :]
        x[:, N_CHANNELS:] = pb[pair_j, pair_i, :]
    finally:
        del pb

    return x


def standardize_features_in_place(x):
    """Standardize features in place and return the raw-space scaler."""
    mean = np.mean(x, axis=0, dtype=np.float64)
    scale = np.std(x, axis=0, dtype=np.float64)
    scale[scale < FLOAT_EPS] = 1.0
    x -= mean
    x /= scale
    return mean, scale


def standardized_to_raw_coefficients(coef_std, intercept_std, scaler_mean, scaler_scale):
    """Convert standardized-space coefficients back to raw feature space."""
    coef_std = np.asarray(coef_std, dtype=np.float64)
    scaler_mean = np.asarray(scaler_mean, dtype=np.float64)
    scaler_scale = np.asarray(scaler_scale, dtype=np.float64)
    coef_raw = coef_std / scaler_scale
    intercept_raw = float(
        intercept_std - np.sum((coef_std * scaler_mean) / scaler_scale, dtype=np.float64)
    )
    return coef_raw, intercept_raw


def build_coefficient_rows(variant, record, layer, fitted, scaler_mean, scaler_scale):
    """Emit intercept plus one row per feature coefficient for one protein model."""
    coef_std = np.asarray(fitted["coef_std"], dtype=np.float64)
    intercept_std = float(fitted["intercept_std"])
    coef_raw, intercept_raw = standardized_to_raw_coefficients(
        coef_std,
        intercept_std,
        scaler_mean,
        scaler_scale,
    )
    feature_names = feature_names_for_variant(variant)
    feature_dim = int(len(feature_names))
    base = {
        "variant": variant["name"],
        "feature_dim": feature_dim,
        "layer": int(layer),
        "source": record.get("source", ""),
        "protein_id": record["protein_id"],
        "n_residues": int(record["n_residues"]),
        "size_bin": int(record["size_bin"]),
        "n_pairs": int(record["n_pairs"]),
    }

    rows = [
        {
            **base,
            "feature_index": -1,
            "feature_name": "intercept",
            "coefficient_standardized": float(intercept_std),
            "coefficient_raw_scale": float(intercept_raw),
        }
    ]

    for feature_index, feature_name in enumerate(feature_names):
        rows.append(
            {
                **base,
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "coefficient_standardized": float(coef_std[feature_index]),
                "coefficient_raw_scale": float(coef_raw[feature_index]),
            }
        )

    return rows


def finite_or_nan(value):
    """Return a finite float value, otherwise nan."""
    value = float(value)
    if math.isfinite(value):
        return value
    return float("nan")


def model_rank_metadata(model):
    """Return rank and singular-value diagnostics for one fitted dense model."""
    rank = int(getattr(model, "rank_", -1))
    singular = np.asarray(getattr(model, "singular_", []), dtype=np.float64)
    if singular.size == 0:
        return rank, float("nan"), float("nan"), float("nan")

    singular_min = finite_or_nan(np.min(singular))
    singular_max = finite_or_nan(np.max(singular))
    if math.isnan(singular_min) or singular_min <= FLOAT_EPS:
        condition_number = float("nan")
    else:
        condition_number = finite_or_nan(singular_max / singular_min)
    return rank, singular_min, singular_max, condition_number


def build_per_protein_row(variant, record, layer, model, y_true, y_pred, elapsed_seconds):
    """Build the per-protein regression metrics row."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    rank, singular_min, singular_max, condition_number = model_rank_metadata(model)
    mse_value = float(np.mean((y_pred - y_true) ** 2)) if y_true.size else float("nan")
    return {
        "variant": variant["name"],
        "feature_dim": int(feature_dim_for_variant(variant)),
        "layer": int(layer),
        "source": record.get("source", ""),
        "protein_id": record["protein_id"],
        "n_residues": int(record["n_residues"]),
        "size_bin": int(record["size_bin"]),
        "n_pairs": int(record["n_pairs"]),
        "target_variance": float(record["target_variance"]),
        "model_rank": rank,
        "model_singular_min": singular_min,
        "model_singular_max": singular_max,
        "model_condition_number": condition_number,
        "per_protein_r2": compute_r2(y_true, y_pred),
        "per_protein_mse": mse_value,
        "per_protein_mae": compute_mae(y_true, y_pred),
        "per_protein_rmse": compute_rmse(y_true, y_pred),
        "mean_true_distance": float(np.mean(y_true)) if y_true.size else float("nan"),
        "mean_predicted_distance": float(np.mean(y_pred)) if y_pred.size else float("nan"),
        "mean_error_pred_minus_true": (
            float(np.mean(y_pred - y_true)) if y_true.size else float("nan")
        ),
        "elapsed_seconds": float(elapsed_seconds),
    }


def make_failed_geometry_row(variant, record, layer, per_r2):
    """Return a placeholder geometry row when reconstruction fails."""
    return {
        "variant": variant["name"],
        "feature_dim": int(feature_dim_for_variant(variant)),
        "layer": int(layer),
        "source": record.get("source", ""),
        "protein_id": record["protein_id"],
        "n_residues": int(record["n_residues"]),
        "size_bin": int(record["size_bin"]),
        "n_pairs": int(record["n_pairs"]),
        "per_protein_r2": float(per_r2),
        "rmsd": float("nan"),
        "tm_score": float("nan"),
        "frac_top3_eigenvalues": float("nan"),
        "n_negative_eigenvalues": -1,
        "neg_eigenvalue_fraction": float("nan"),
        "n_negative_predictions": -1,
        "stress": float("nan"),
        "triangle_violation_fraction": float("nan"),
        "triangle_mean_violation": float("nan"),
        "triangle_max_violation": float("nan"),
    }


def build_geometry_row(variant, record, layer, y_pred, per_r2):
    """Run MDS/TM-score geometry analysis for one protein/layer."""
    try:
        metrics = analyze_protein_geometry(
            int(record["n_residues"]),
            np.asarray(record["coordinates"], dtype=np.float64),
            record["lt_i"],
            record["lt_j"],
            np.asarray(y_pred, dtype=np.float64),
        )
    except Exception:
        traceback.print_exc()
        return make_failed_geometry_row(variant, record, layer, per_r2)

    return {
        "variant": variant["name"],
        "feature_dim": int(feature_dim_for_variant(variant)),
        "layer": int(layer),
        "source": record.get("source", ""),
        "protein_id": record["protein_id"],
        "n_residues": int(record["n_residues"]),
        "size_bin": int(record["size_bin"]),
        "n_pairs": int(record["n_pairs"]),
        "per_protein_r2": float(per_r2),
        **metrics,
    }


def fit_single_protein_layer(variant, record, layer):
    """Fit and evaluate one protein-specific model for one layer."""
    fit_t0 = time.time()
    x = build_base256_feature_matrix(record, layer)
    y = np.asarray(record["y_abs"], dtype=np.float64)
    scaler_mean, scaler_scale = standardize_features_in_place(x)

    # X is already standardized, so fitting centered y without an intercept is
    # equivalent to an intercept model and avoids sklearn mutating X for centering.
    intercept_std = float(np.mean(y)) if y.size else 0.0
    y_centered = y - intercept_std
    model = LinearRegression(copy_X=False, fit_intercept=False)
    model.fit(x, y_centered)
    coef_std = np.asarray(model.coef_, dtype=np.float64).copy()
    y_pred = np.asarray((x @ coef_std) + intercept_std, dtype=np.float64)

    fitted = {
        "coef_std": coef_std,
        "intercept_std": intercept_std,
    }
    per_row = build_per_protein_row(
        variant,
        record,
        layer,
        model,
        y,
        y_pred,
        elapsed_seconds=float(time.time() - fit_t0),
    )
    coefficient_rows = build_coefficient_rows(
        variant,
        record,
        layer,
        fitted,
        scaler_mean,
        scaler_scale,
    )
    geometry_row = build_geometry_row(
        variant,
        record,
        layer,
        y_pred,
        per_row["per_protein_r2"],
    )

    del x
    del model
    del y_pred
    del y_centered
    del fitted
    return per_row, coefficient_rows, geometry_row


def summarize_geometry_rows(geometry_rows):
    """Aggregate geometry metrics per variant and layer."""
    grouped = {}
    for row in geometry_rows:
        key = (row["variant"], int(row["feature_dim"]), int(row["layer"]))
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (variant_name, feature_dim, layer), rows in sorted(grouped.items()):
        out = {
            "variant": variant_name,
            "feature_dim": int(feature_dim),
            "layer": int(layer),
            "n_proteins": len(rows),
        }
        for metric_key in GEOMETRY_METRIC_KEYS:
            values = [
                float(row[metric_key])
                for row in rows
                if isinstance(row[metric_key], (int, float, np.integer, np.floating))
                and not math.isnan(float(row[metric_key]))
            ]
            out[f"{metric_key}_mean"] = float(np.mean(values)) if values else float("nan")
            out[f"{metric_key}_std"] = float(np.std(values)) if values else float("nan")
        summary_rows.append(out)

    return summary_rows


def numeric_summary(rows, key):
    """Return min/median/mean/max for a numeric row key."""
    values = [
        float(row[key])
        for row in rows
        if isinstance(row.get(key), (int, float, np.integer, np.floating))
        and not math.isnan(float(row[key]))
    ]
    if not values:
        return None
    values = np.asarray(values, dtype=np.float64)
    return {
        "min": float(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "max": float(np.max(values)),
    }


def write_summary_text(
    path,
    title,
    variant,
    layer_indices,
    elapsed_seconds,
    all_proteins,
    loaded_records,
    skipped,
    per_protein_rows,
    geometry_rows,
):
    """Write a compact human-readable summary."""
    layer_indices = normalize_layer_indices(layer_indices)
    completed_by_layer = {}
    for row in per_protein_rows:
        completed_by_layer[int(row["layer"])] = completed_by_layer.get(int(row["layer"]), 0) + 1

    with path.open("w") as handle:
        handle.write(f"{title}\n")
        handle.write(f"{'=' * len(title)}\n\n")
        handle.write(f"Variant:             {variant['name']}\n")
        handle.write(f"Feature dimension:   {feature_dim_for_variant(variant)}\n")
        handle.write("Feature convention:  unordered base256 [pair_i_j, pair_j_i]\n")
        handle.write("Dataset:             combined CAMEO+ECOD, TM-score > 0.7\n")
        handle.write(
            f"Layers:              {format_layer_indices(layer_indices)} "
            f"({len(layer_indices)} total)\n"
        )
        handle.write("Splits:              none; each model uses all unordered residue pairs\n")
        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(loaded_records)}\n")
        handle.write(f"Loaded by source:    {source_counts(loaded_records)}\n")
        handle.write(f"Completed rows:      {len(per_protein_rows)}\n")
        handle.write(f"Completed by layer:  {completed_by_layer}\n")
        handle.write(f"Skipped events:      {len(skipped)}\n")
        handle.write(f"Elapsed seconds:     {elapsed_seconds:.2f}\n\n")

        for layer in layer_indices:
            layer_rows = [row for row in per_protein_rows if int(row["layer"]) == layer]
            geom_rows = [row for row in geometry_rows if int(row["layer"]) == layer]
            r2_stats = numeric_summary(layer_rows, "per_protein_r2")
            tm_stats = numeric_summary(geom_rows, "tm_score")
            if r2_stats is not None:
                handle.write(
                    f"Layer {layer:02d} R2: "
                    f"min={r2_stats['min']:.6f}, "
                    f"median={r2_stats['median']:.6f}, "
                    f"mean={r2_stats['mean']:.6f}, "
                    f"max={r2_stats['max']:.6f}\n"
                )
            if tm_stats is not None:
                handle.write(
                    f"Layer {layer:02d} TM-score: "
                    f"min={tm_stats['min']:.6f}, "
                    f"median={tm_stats['median']:.6f}, "
                    f"mean={tm_stats['mean']:.6f}, "
                    f"max={tm_stats['max']:.6f}\n"
                )

        if skipped:
            handle.write("\nSkipped proteins/events:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")


def write_checkpoint_outputs(
    output_dir,
    title,
    variant,
    layer_indices,
    output_filename_suffix,
    all_proteins,
    loaded_records,
    skipped,
    per_protein_rows,
    geometry_rows,
    elapsed_seconds,
):
    """Write checkpoint summary artifacts derived from streamed row outputs."""
    write_csv(
        output_dir / resolve_output_filename(GEOMETRY_SUMMARY_CSV_NAME, output_filename_suffix),
        GEOMETRY_SUMMARY_FIELDNAMES,
        summarize_geometry_rows(geometry_rows),
    )
    write_summary_text(
        output_dir / resolve_output_filename(SUMMARY_TXT_NAME, output_filename_suffix),
        title=title,
        variant=variant,
        layer_indices=layer_indices,
        elapsed_seconds=elapsed_seconds,
        all_proteins=all_proteins,
        loaded_records=loaded_records,
        skipped=skipped,
        per_protein_rows=per_protein_rows,
        geometry_rows=geometry_rows,
    )


def run_per_protein_job(
    variant,
    title,
    layer_indices=None,
    output_filename_suffix="",
    checkpoint_every=25,
):
    """Run one combined-dataset per-protein regression job."""
    t0 = time.time()
    csv_root_env = os.environ.get("CSV_ROOT") or str(Path.home() / "csv_files")
    layer_indices = normalize_layer_indices(layer_indices)
    output_dir = Path(csv_root_env).expanduser().resolve() / variant["name"]
    output_dir.mkdir(parents=True, exist_ok=True)

    print(title)
    print("=" * len(title))
    log_progress(f"CAMEO pair root:  {CAMEO_PAIR_BLOCKS_ROOT}")
    log_progress(f"CAMEO PDB root:   {CAMEO_PDB_ROOT}")
    log_progress(f"ECOD pair root:   {ECOD_PAIR_BLOCKS_ROOT}")
    log_progress(f"ECOD PDB root:    {ECOD_PDB_ROOT}")
    log_progress(f"Output dir:       {output_dir}")
    log_progress(
        f"Layer subset:     {format_layer_indices(layer_indices)} "
        f"({len(layer_indices)} total)"
    )

    all_proteins, records, skipped = load_protein_records()
    if not records:
        log_progress("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    assign_size_bins(records, N_SIZE_BINS)
    log_progress(f"Loaded proteins:  {len(records)}")
    log_progress(f"Loaded by source: {source_counts(records)}")

    per_protein_rows = []
    geometry_rows = []
    runtime_skipped = []

    per_path = output_dir / resolve_output_filename(PER_PROTEIN_CSV_NAME, output_filename_suffix)
    coef_path = output_dir / resolve_output_filename(COEFFICIENTS_CSV_NAME, output_filename_suffix)
    geom_path = output_dir / resolve_output_filename(
        GEOMETRY_METRICS_CSV_NAME,
        output_filename_suffix,
    )

    handles = []
    try:
        per_handle, per_writer = open_streaming_csv(per_path, PER_PROTEIN_FIELDNAMES)
        coef_handle, coef_writer = open_streaming_csv(coef_path, COEFFICIENT_FIELDNAMES)
        geom_handle, geom_writer = open_streaming_csv(geom_path, GEOMETRY_METRICS_FIELDNAMES)
        handles = [per_handle, coef_handle, geom_handle]

        write_checkpoint_outputs(
            output_dir=output_dir,
            title=title,
            variant=variant,
            layer_indices=layer_indices,
            output_filename_suffix=output_filename_suffix,
            all_proteins=all_proteins,
            loaded_records=records,
            skipped=[*skipped, *runtime_skipped],
            per_protein_rows=per_protein_rows,
            geometry_rows=geometry_rows,
            elapsed_seconds=float(time.time() - t0),
        )

        for layer in layer_indices:
            layer_t0 = time.time()
            log_progress(f"Layer {layer:02d}: starting per-protein fits")
            for record_index, record in enumerate(records, start=1):
                protein_t0 = time.time()
                protein_id = record["protein_id"]
                log_progress(
                    f"Layer {layer:02d}: fitting {record_index}/{len(records)} "
                    f"{protein_id} ({record['n_residues']} residues, "
                    f"{record['n_pairs']:,} pairs)"
                )
                try:
                    per_row, coefficient_rows, geometry_row = fit_single_protein_layer(
                        variant,
                        record,
                        layer,
                    )
                except Exception as exc:
                    reason = f"layer {layer}: {exc}"
                    runtime_skipped.append((protein_id, reason))
                    log_progress(f"Layer {layer:02d}: SKIP {protein_id} -- {exc}")
                    traceback.print_exc()
                    continue

                write_streaming_row(per_writer, PER_PROTEIN_FIELDNAMES, per_row)
                for coef_row in coefficient_rows:
                    write_streaming_row(coef_writer, COEFFICIENT_FIELDNAMES, coef_row)
                write_streaming_row(geom_writer, GEOMETRY_METRICS_FIELDNAMES, geometry_row)
                flush_streams(handles)

                per_protein_rows.append(per_row)
                geometry_rows.append(geometry_row)
                log_progress(
                    f"Layer {layer:02d}: finished {protein_id} "
                    f"R2={per_row['per_protein_r2']:.4f}, "
                    f"TM={geometry_row['tm_score']:.4f}, "
                    f"elapsed={time.time() - protein_t0:.1f}s"
                )

                if record_index % checkpoint_every == 0:
                    write_checkpoint_outputs(
                        output_dir=output_dir,
                        title=title,
                        variant=variant,
                        layer_indices=layer_indices,
                        output_filename_suffix=output_filename_suffix,
                        all_proteins=all_proteins,
                        loaded_records=records,
                        skipped=[*skipped, *runtime_skipped],
                        per_protein_rows=per_protein_rows,
                        geometry_rows=geometry_rows,
                        elapsed_seconds=float(time.time() - t0),
                    )
                    log_progress(
                        f"Layer {layer:02d}: checkpointed after {record_index} proteins"
                    )

                gc.collect()

            write_checkpoint_outputs(
                output_dir=output_dir,
                title=title,
                variant=variant,
                layer_indices=layer_indices,
                output_filename_suffix=output_filename_suffix,
                all_proteins=all_proteins,
                loaded_records=records,
                skipped=[*skipped, *runtime_skipped],
                per_protein_rows=per_protein_rows,
                geometry_rows=geometry_rows,
                elapsed_seconds=float(time.time() - t0),
            )
            log_progress(
                f"Layer {layer:02d}: completed in {time.time() - layer_t0:.1f}s "
                f"(total {time.time() - t0:.1f}s)"
            )
    finally:
        for handle in handles:
            handle.close()

    log_progress(f"Saved per-protein metrics -> {per_path}")
    log_progress(f"Saved coefficients        -> {coef_path}")
    log_progress(f"Saved geometry metrics    -> {geom_path}")
    log_progress(f"Completed {variant['name']} run in {time.time() - t0:.1f}s")


def make_base256_per_protein_variant(name="per_protein_base256_combined"):
    """Create the fixed base256 per-protein variant."""
    return make_variant(name, pair_mode="concat_directional")

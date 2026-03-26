#!/usr/bin/env python3
"""
Temporary single train/test split runner for generalized context-feature jobs.
"""

import gc
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
GENERALIZED_MODEL_DIR = SCRIPT_DIR.parent / "generalized_model"
if str(GENERALIZED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(GENERALIZED_MODEL_DIR))

import context_feature_kfold_common as base  # noqa: E402


TEST_SIZE = 0.2
TEMP_OUTPUT_SUBDIR = "temp_context_feature_train_test"

SPLIT_SUMMARY_CSV_NAME = "split_summary.csv"
PER_PROTEIN_CSV_NAME = "per_protein.csv"
COEFFICIENTS_CSV_NAME = "coefficients.csv"
SPLIT_ASSIGNMENTS_CSV_NAME = "split_assignments.csv"
GEOMETRY_METRICS_CSV_NAME = "geometry_metrics.csv"
GEOMETRY_SUMMARY_CSV_NAME = "geometry_summary.csv"
SUMMARY_TXT_NAME = "summary.txt"

SPLIT_SUMMARY_FIELDNAMES = [
    "variant",
    "feature_dim",
    "layer",
    "n_train_proteins",
    "n_test_proteins",
    "n_train_rows",
    "n_test_rows",
    "train_mean_per_protein_r2",
    "test_mean_per_protein_r2",
    "train_pair_weighted_r2",
    "test_pair_weighted_r2",
    "train_pair_weighted_mae",
    "test_pair_weighted_mae",
    "train_pair_weighted_rmse",
    "test_pair_weighted_rmse",
    "train_mean_bias",
    "test_mean_bias",
    "elapsed_seconds",
]

PER_PROTEIN_FIELDNAMES = [
    "variant",
    "feature_dim",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
    "per_protein_r2",
    "per_protein_mse",
    "per_protein_mae",
    "per_protein_rmse",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
]

COEFFICIENT_FIELDNAMES = [
    "variant",
    "feature_dim",
    "layer",
    "feature_index",
    "feature_name",
    "coefficient_standardized",
    "coefficient_raw_scale",
]

SPLIT_ASSIGNMENT_FIELDNAMES = [
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
]

GEOMETRY_METRICS_FIELDNAMES = [
    "variant",
    "feature_dim",
    "layer",
    "split",
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

GEOMETRY_SUMMARY_FIELDNAMES = ["variant", "feature_dim", "layer", "split", "n_proteins"]
for _metric_key in base.GEOMETRY_METRIC_KEYS:
    GEOMETRY_SUMMARY_FIELDNAMES.extend([f"{_metric_key}_mean", f"{_metric_key}_std"])


make_variant = base.make_variant


def choose_stratify_labels(records):
    """Only stratify when the requested outer split is feasible."""
    if len(records) < 2:
        return None

    labels = np.asarray([record["size_bin"] for record in records], dtype=np.int32)
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2 or np.min(counts) < 2:
        return None

    n_total = len(records)
    n_test = max(1, int(math.ceil(TEST_SIZE * n_total)))
    n_train = n_total - n_test
    if n_test < unique.size or n_train < unique.size:
        return None
    return labels


def split_records(records):
    """Create one deterministic protein-level train/test split."""
    if len(records) < 2:
        raise ValueError("Need at least two proteins for train/test splitting.")

    indices = np.arange(len(records), dtype=np.int64)
    labels = choose_stratify_labels(records)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=base.RANDOM_STATE,
        stratify=labels,
    )

    train_set = set(int(idx) for idx in train_idx)
    test_set = set(int(idx) for idx in test_idx)
    train_records = [record for idx, record in enumerate(records) if idx in train_set]
    test_records = [record for idx, record in enumerate(records) if idx in test_set]
    return train_records, test_records


def resolve_variant_output_dir(csv_root, variant):
    """Return the isolated output directory for one temporary variant."""
    return Path(csv_root).resolve() / TEMP_OUTPUT_SUBDIR / variant["name"]


def build_split_assignment_rows(train_records, test_records):
    """Build one single-split assignment table."""
    rows = []
    for split_name, records in [("train", train_records), ("test", test_records)]:
        for record in records:
            rows.append(
                {
                    "split": split_name,
                    "protein_id": record["protein_id"],
                    "n_residues": int(record["n_residues"]),
                    "size_bin": int(record["size_bin"]),
                    "n_pairs": int(record["n_pairs"]),
                    "target_variance": float(record["target_variance"]),
                }
            )
    return rows


def build_split_summary_row(
    variant,
    layer,
    train_records,
    test_records,
    train_metrics,
    test_metrics,
    elapsed_seconds,
):
    """Build one per-layer single-split summary row."""
    return {
        "variant": variant["name"],
        "feature_dim": int(base.feature_dim_for_variant(variant)),
        "layer": int(layer),
        "n_train_proteins": int(len(train_records)),
        "n_test_proteins": int(len(test_records)),
        "n_train_rows": int(sum(record["n_pairs"] for record in train_records)),
        "n_test_rows": int(sum(record["n_pairs"] for record in test_records)),
        "train_mean_per_protein_r2": float(train_metrics["mean_per_protein_r2"]),
        "test_mean_per_protein_r2": float(test_metrics["mean_per_protein_r2"]),
        "train_pair_weighted_r2": float(train_metrics["pair_weighted_r2"]),
        "test_pair_weighted_r2": float(test_metrics["pair_weighted_r2"]),
        "train_pair_weighted_mae": float(train_metrics["pair_weighted_mae"]),
        "test_pair_weighted_mae": float(test_metrics["pair_weighted_mae"]),
        "train_pair_weighted_rmse": float(train_metrics["pair_weighted_rmse"]),
        "test_pair_weighted_rmse": float(test_metrics["pair_weighted_rmse"]),
        "train_mean_bias": float(train_metrics["mean_bias"]),
        "test_mean_bias": float(test_metrics["mean_bias"]),
        "elapsed_seconds": float(elapsed_seconds),
    }


def build_per_protein_output_rows(variant, layer, split_name, metric_rows):
    """Attach single-split metadata to per-protein rows."""
    rows = []
    for row in metric_rows:
        out = dict(row)
        out["variant"] = variant["name"]
        out["feature_dim"] = int(base.feature_dim_for_variant(variant))
        out["layer"] = int(layer)
        out["split"] = split_name
        rows.append(out)
    return rows


def build_coefficient_rows(variant, layer, fitted, scaler_mean, scaler_scale):
    """Emit one intercept row plus one row per feature coefficient."""
    coef_std = np.asarray(fitted["coef_std"], dtype=np.float64)
    intercept_std = float(fitted["intercept_std"])
    coef_raw, intercept_raw = base.standardized_to_raw_coefficients(
        coef_std,
        intercept_std,
        scaler_mean,
        scaler_scale,
    )
    feature_names = base.feature_names_for_variant(variant)
    feature_dim = int(len(feature_names))

    rows = [
        {
            "variant": variant["name"],
            "feature_dim": feature_dim,
            "layer": int(layer),
            "feature_index": -1,
            "feature_name": "intercept",
            "coefficient_standardized": float(intercept_std),
            "coefficient_raw_scale": float(intercept_raw),
        }
    ]

    for feature_index, feature_name in enumerate(feature_names):
        rows.append(
            {
                "variant": variant["name"],
                "feature_dim": feature_dim,
                "layer": int(layer),
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "coefficient_standardized": float(coef_std[feature_index]),
                "coefficient_raw_scale": float(coef_raw[feature_index]),
            }
        )
    return rows


def make_failed_geometry_row(variant, layer, split_name, record, per_r2):
    """Return a placeholder geometry row when reconstruction fails."""
    return {
        "variant": variant["name"],
        "feature_dim": int(base.feature_dim_for_variant(variant)),
        "layer": int(layer),
        "split": split_name,
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


def collect_geometry_rows_for_split(variant, layer, split_name, split_offsets, y_pred):
    """Run per-protein geometry analysis for one single split."""
    rows = []
    for record, start, end in split_offsets:
        y_true = np.asarray(record["y_abs"], dtype=np.float64)
        y_pred_p = np.asarray(y_pred[start:end], dtype=np.float64)
        per_r2 = base.compute_r2(y_true, y_pred_p)
        try:
            metrics = base.analyze_protein_geometry(
                int(record["n_residues"]),
                np.asarray(record["coordinates"], dtype=np.float64),
                record["lt_i"],
                record["lt_j"],
                y_pred_p,
            )
            rows.append(
                {
                    "variant": variant["name"],
                    "feature_dim": int(base.feature_dim_for_variant(variant)),
                    "layer": int(layer),
                    "split": split_name,
                    "protein_id": record["protein_id"],
                    "n_residues": int(record["n_residues"]),
                    "size_bin": int(record["size_bin"]),
                    "n_pairs": int(record["n_pairs"]),
                    "per_protein_r2": float(per_r2),
                    **metrics,
                }
            )
        except Exception:
            rows.append(make_failed_geometry_row(variant, layer, split_name, record, per_r2))
    return rows


def summarize_geometry_rows(geometry_rows):
    """Aggregate geometry metrics per variant, layer, and split."""
    grouped = {}
    for row in geometry_rows:
        key = (
            row["variant"],
            int(row["feature_dim"]),
            int(row["layer"]),
            row["split"],
        )
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (variant_name, feature_dim, layer, split_name), rows in sorted(grouped.items()):
        out = {
            "variant": variant_name,
            "feature_dim": int(feature_dim),
            "layer": int(layer),
            "split": split_name,
            "n_proteins": len(rows),
        }
        for metric_key in base.GEOMETRY_METRIC_KEYS:
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


def build_summary_text(
    path,
    title,
    variant,
    layer_indices,
    elapsed_seconds,
    all_proteins,
    loaded_records,
    skipped,
    size_bin_counts,
    train_records,
    test_records,
    split_summary_rows,
    geometry_summary_rows,
):
    """Write a compact human-readable summary for one temporary train/test run."""
    layer_means = [
        (int(row["layer"]), float(row["test_mean_per_protein_r2"]))
        for row in split_summary_rows
    ]
    best_layer = None
    best_key = None
    for layer, mean_r2 in layer_means:
        safe_score = -1e18 if math.isnan(mean_r2) else mean_r2
        key = (safe_score, -layer)
        if best_layer is None or key > best_key:
            best_layer = (layer, mean_r2)
            best_key = key

    geom_by_layer = {}
    for row in geometry_summary_rows:
        if row["split"] != "test":
            continue
        geom_by_layer.setdefault(int(row["layer"]), []).append(float(row["tm_score_mean"]))

    best_tm_layer = None
    best_tm_key = None
    for layer, values in sorted(geom_by_layer.items()):
        clean_values = [value for value in values if not math.isnan(value)]
        mean_tm = float(np.mean(clean_values)) if clean_values else float("nan")
        safe_score = -1e18 if math.isnan(mean_tm) else mean_tm
        key = (safe_score, -layer)
        if best_tm_layer is None or key > best_tm_key:
            best_tm_layer = (layer, mean_tm)
            best_tm_key = key

    with path.open("w") as handle:
        handle.write(f"{title}\n")
        handle.write(f"{'=' * len(title)}\n\n")
        handle.write(f"Variant:             {variant['name']}\n")
        handle.write(f"Feature dimension:   {base.feature_dim_for_variant(variant)}\n")
        handle.write(
            f"Layers:              {base.format_layer_indices(layer_indices)} "
            f"({len(base.normalize_layer_indices(layer_indices))} total)\n"
        )
        handle.write(
            "Train/test split:    80/20 protein-level, deterministic, "
            "size-stratified when feasible\n"
        )
        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(loaded_records)}\n")
        handle.write(f"Train proteins:      {len(train_records)}\n")
        handle.write(f"Test proteins:       {len(test_records)}\n")
        handle.write(f"Size bin counts:     {size_bin_counts}\n")
        handle.write(f"Elapsed seconds:     {elapsed_seconds:.2f}\n\n")

        if best_layer is not None:
            handle.write(
                "Best layer by held-out per-protein R2:\n"
                f"  Layer {best_layer[0]} -> {best_layer[1]:.4f}\n\n"
            )

        if best_tm_layer is not None:
            handle.write(
                "Best test geometry layer by mean TM-score:\n"
                f"  Layer {best_tm_layer[0]} -> {best_tm_layer[1]:.4f}\n\n"
            )

        if layer_means:
            handle.write("Layer means (held-out per-protein R2):\n")
            for layer, mean_r2 in sorted(layer_means):
                handle.write(f"  Layer {layer:02d}: {mean_r2:.4f}\n")

        if skipped:
            handle.write("\nSkipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")


def flush_outputs(
    output_dir,
    title,
    variant,
    layer_indices,
    output_filename_suffix,
    all_proteins,
    loaded_records,
    skipped,
    size_bin_counts,
    train_records,
    test_records,
    split_summary_rows,
    per_protein_rows,
    coefficient_rows,
    split_assignment_rows,
    geometry_rows,
    elapsed_seconds,
):
    """Persist the current partial or final outputs to disk."""
    geometry_summary_rows = summarize_geometry_rows(geometry_rows)
    base.write_csv(
        output_dir / base.resolve_output_filename(SPLIT_SUMMARY_CSV_NAME, output_filename_suffix),
        SPLIT_SUMMARY_FIELDNAMES,
        split_summary_rows,
    )
    base.write_csv(
        output_dir / base.resolve_output_filename(PER_PROTEIN_CSV_NAME, output_filename_suffix),
        PER_PROTEIN_FIELDNAMES,
        per_protein_rows,
    )
    base.write_csv(
        output_dir / base.resolve_output_filename(COEFFICIENTS_CSV_NAME, output_filename_suffix),
        COEFFICIENT_FIELDNAMES,
        coefficient_rows,
    )
    base.write_csv(
        output_dir / base.resolve_output_filename(
            SPLIT_ASSIGNMENTS_CSV_NAME,
            output_filename_suffix,
        ),
        SPLIT_ASSIGNMENT_FIELDNAMES,
        split_assignment_rows,
    )
    base.write_csv(
        output_dir / base.resolve_output_filename(GEOMETRY_METRICS_CSV_NAME, output_filename_suffix),
        GEOMETRY_METRICS_FIELDNAMES,
        geometry_rows,
    )
    base.write_csv(
        output_dir / base.resolve_output_filename(GEOMETRY_SUMMARY_CSV_NAME, output_filename_suffix),
        GEOMETRY_SUMMARY_FIELDNAMES,
        geometry_summary_rows,
    )
    build_summary_text(
        output_dir / base.resolve_output_filename(SUMMARY_TXT_NAME, output_filename_suffix),
        title=title,
        variant=variant,
        layer_indices=layer_indices,
        elapsed_seconds=elapsed_seconds,
        all_proteins=all_proteins,
        loaded_records=loaded_records,
        skipped=skipped,
        size_bin_counts=size_bin_counts,
        train_records=train_records,
        test_records=test_records,
        split_summary_rows=split_summary_rows,
        geometry_summary_rows=geometry_summary_rows,
    )


def run_variant_train_test_job(variant, title, layer_indices=None, output_filename_suffix=""):
    """Run one temporary single-split context-feature job for a layer subset."""
    t0 = time.time()
    pair_blocks_root_env = os.environ.get("PAIR_BLOCKS_ROOT")
    pdb_root_env = os.environ.get("PDB_ROOT")
    csv_root_env = os.environ.get("CSV_ROOT")
    layer_indices = base.normalize_layer_indices(layer_indices)

    if not pair_blocks_root_env or not pdb_root_env or not csv_root_env:
        raise ValueError("PAIR_BLOCKS_ROOT, PDB_ROOT, and CSV_ROOT must all be set.")

    pair_blocks_root = Path(pair_blocks_root_env).resolve()
    pdb_root = Path(pdb_root_env).resolve()
    output_dir = resolve_variant_output_dir(csv_root_env, variant)
    output_dir.mkdir(parents=True, exist_ok=True)

    print(title)
    print("=" * len(title))
    base.log_progress(f"Pair blocks root: {pair_blocks_root}")
    base.log_progress(f"PDB root:         {pdb_root}")
    base.log_progress(f"Output dir:       {output_dir}")
    base.log_progress(
        f"Layer subset:     {base.format_layer_indices(layer_indices)} "
        f"({len(layer_indices)} total)"
    )

    all_proteins, records, skipped = base.load_protein_records(pair_blocks_root, pdb_root)
    if not records:
        base.log_progress("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    base.assign_size_bins(records, base.N_SIZE_BINS)
    train_records, test_records = split_records(records)
    size_bin_counts = {
        int(label): int(count)
        for label, count in zip(
            *np.unique(
                np.asarray([record["size_bin"] for record in records], dtype=np.int32),
                return_counts=True,
            )
        )
    }
    split_assignment_rows = build_split_assignment_rows(train_records, test_records)

    base.log_progress(f"Loaded proteins:  {len(records)}")
    base.log_progress(f"Train proteins:   {len(train_records)}")
    base.log_progress(f"Test proteins:    {len(test_records)}")
    base.log_progress(f"Size bin counts:  {size_bin_counts}")

    split_summary_rows = []
    per_protein_rows = []
    coefficient_rows = []
    geometry_rows = []

    flush_outputs(
        output_dir=output_dir,
        title=title,
        variant=variant,
        layer_indices=layer_indices,
        output_filename_suffix=output_filename_suffix,
        all_proteins=all_proteins,
        loaded_records=records,
        skipped=skipped,
        size_bin_counts=size_bin_counts,
        train_records=train_records,
        test_records=test_records,
        split_summary_rows=split_summary_rows,
        per_protein_rows=per_protein_rows,
        coefficient_rows=coefficient_rows,
        split_assignment_rows=split_assignment_rows,
        geometry_rows=geometry_rows,
        elapsed_seconds=float(time.time() - t0),
    )

    for layer in layer_indices:
        layer_t0 = time.time()
        base.log_progress(f"Layer {layer:02d}: starting cache build")
        layer_cache = base.load_layer_feature_cache(records, layer, variant)
        layer_rows = int(sum(chunk.shape[0] for chunk in layer_cache.values()))
        base.log_progress(
            f"Layer {layer:02d}: finished cache build with {len(layer_cache)} proteins "
            f"({layer_rows:,} rows) in {time.time() - layer_t0:.1f}s"
        )
        try:
            n_train_rows = int(sum(record["n_pairs"] for record in train_records))
            n_test_rows = int(sum(record["n_pairs"] for record in test_records))
            base.log_progress(
                f"Layer {layer:02d}: starting fit/eval "
                f"({n_train_rows:,} train rows, {n_test_rows:,} test rows)"
            )

            scaler_mean, scaler_scale = base.compute_scaler_from_cache(
                train_records,
                layer_cache,
                variant,
            )
            train_split = base.build_standardized_split(
                train_records,
                layer_cache,
                scaler_mean,
                scaler_scale,
                variant,
            )
            sample_weight = base.compute_protein_equal_weights(train_records)
            fitted = base.fit_model_from_matrix(
                train_split["x"],
                train_split["y"],
                sample_weight,
            )
            y_pred_train = np.asarray(
                fitted["model"].predict(train_split["x"]),
                dtype=np.float64,
            )
            train_metrics = base.summarize_split_from_predictions(
                train_split["offsets"],
                train_split["y"],
                y_pred_train,
            )
            del train_split["x"]
            del sample_weight

            test_split = base.build_standardized_split(
                test_records,
                layer_cache,
                scaler_mean,
                scaler_scale,
                variant,
            )
            y_pred_test = np.asarray(
                fitted["model"].predict(test_split["x"]),
                dtype=np.float64,
            )
            test_metrics = base.summarize_split_from_predictions(
                test_split["offsets"],
                test_split["y"],
                y_pred_test,
            )
            del test_split["x"]

            split_summary_rows.append(
                build_split_summary_row(
                    variant=variant,
                    layer=layer,
                    train_records=train_records,
                    test_records=test_records,
                    train_metrics=train_metrics,
                    test_metrics=test_metrics,
                    elapsed_seconds=float(time.time() - layer_t0),
                )
            )
            per_protein_rows.extend(
                build_per_protein_output_rows(
                    variant,
                    layer,
                    "train",
                    train_metrics["per_rows"],
                )
            )
            per_protein_rows.extend(
                build_per_protein_output_rows(
                    variant,
                    layer,
                    "test",
                    test_metrics["per_rows"],
                )
            )
            coefficient_rows.extend(
                build_coefficient_rows(
                    variant,
                    layer,
                    fitted,
                    scaler_mean,
                    scaler_scale,
                )
            )
            geometry_rows.extend(
                collect_geometry_rows_for_split(
                    variant,
                    layer,
                    "train",
                    train_split["offsets"],
                    y_pred_train,
                )
            )
            geometry_rows.extend(
                collect_geometry_rows_for_split(
                    variant,
                    layer,
                    "test",
                    test_split["offsets"],
                    y_pred_test,
                )
            )

            base.log_progress(
                f"Layer {layer:02d}: "
                f"test_R2={test_metrics['mean_per_protein_r2']:.4f}, "
                f"test_pair_R2={test_metrics['pair_weighted_r2']:.4f}, "
                f"elapsed={time.time() - layer_t0:.1f}s"
            )
            del train_split["y"]
            del train_split["offsets"]
            del test_split["y"]
            del test_split["offsets"]
            del y_pred_train
            del y_pred_test
            del fitted["model"]
            del fitted

            base.log_progress(f"Layer {layer:02d}: writing checkpoint outputs")
            flush_outputs(
                output_dir=output_dir,
                title=title,
                variant=variant,
                layer_indices=layer_indices,
                output_filename_suffix=output_filename_suffix,
                all_proteins=all_proteins,
                loaded_records=records,
                skipped=skipped,
                size_bin_counts=size_bin_counts,
                train_records=train_records,
                test_records=test_records,
                split_summary_rows=split_summary_rows,
                per_protein_rows=per_protein_rows,
                coefficient_rows=coefficient_rows,
                split_assignment_rows=split_assignment_rows,
                geometry_rows=geometry_rows,
                elapsed_seconds=float(time.time() - t0),
            )
            base.log_progress(
                f"Finished layer {layer:02d} "
                f"(elapsed {time.time() - layer_t0:.1f}s, total {time.time() - t0:.1f}s)"
            )
        finally:
            layer_cache.clear()
            del layer_cache
            gc.collect()

    flush_outputs(
        output_dir=output_dir,
        title=title,
        variant=variant,
        layer_indices=layer_indices,
        output_filename_suffix=output_filename_suffix,
        all_proteins=all_proteins,
        loaded_records=records,
        skipped=skipped,
        size_bin_counts=size_bin_counts,
        train_records=train_records,
        test_records=test_records,
        split_summary_rows=split_summary_rows,
        per_protein_rows=per_protein_rows,
        coefficient_rows=coefficient_rows,
        split_assignment_rows=split_assignment_rows,
        geometry_rows=geometry_rows,
        elapsed_seconds=float(time.time() - t0),
    )
    base.log_progress(f"Completed {variant['name']} temporary train/test run in {time.time() - t0:.1f}s")

#!/usr/bin/env python3
"""
Diagnostic for `LinearRegression(copy_X=False)` in the strict baseline256
layer-47 size-stratified 5-fold regression pipeline.

This keeps the shared K-fold data path, fold construction, float conversions,
and protein-equal weighting exactly aligned with the production baseline256
workflow. The only comparison is sklearn `copy_X=False` versus `copy_X=True`.

Outputs inside `CSV_ROOT`:
    diagnostic_copy_x_kfold_fold_summary.csv
    diagnostic_copy_x_kfold_mutation_audit.csv
    diagnostic_copy_x_kfold_comparison_summary.csv
    diagnostic_copy_x_kfold_fold_assignments.csv
    diagnostic_copy_x_kfold_summary.txt
"""

import gc
import math
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
K_FOLD_DIR = CLUSTER_SCRIPTS_DIR / "generalized_model" / "mar25_kfold_regressions"
for candidate in [CLUSTER_SCRIPTS_DIR, K_FOLD_DIR]:
    if str(candidate) not in sys.path:
        sys.path.insert(0, str(candidate))

from context_feature_kfold_common import (  # noqa: E402
    FOLD_ASSIGNMENT_FIELDNAMES,
    N_FOLDS,
    N_SIZE_BINS,
    assign_size_bins,
    build_standardized_split,
    build_strict_stratified_folds,
    compute_protein_equal_weights,
    compute_scaler_from_cache,
    feature_dim_for_variant,
    load_layer_feature_cache,
    load_protein_records,
    make_variant,
    summarize_split_from_predictions,
    write_csv,
)


TITLE = "Diagnostic copy_X K-Fold Regression - baseline256 layer 47"
VARIANT = make_variant("baseline256")
LAYER = 47
FEATURE_DIM = feature_dim_for_variant(VARIANT)
COPY_SETTINGS = [False, True]

FOLD_SUMMARY_CSV_NAME = "diagnostic_copy_x_kfold_fold_summary.csv"
MUTATION_AUDIT_CSV_NAME = "diagnostic_copy_x_kfold_mutation_audit.csv"
COMPARISON_SUMMARY_CSV_NAME = "diagnostic_copy_x_kfold_comparison_summary.csv"
FOLD_ASSIGNMENTS_CSV_NAME = "diagnostic_copy_x_kfold_fold_assignments.csv"
SUMMARY_TXT_NAME = "diagnostic_copy_x_kfold_summary.txt"

FOLD_SUMMARY_FIELDNAMES = [
    "variant",
    "layer",
    "feature_dim",
    "copy_x",
    "fold",
    "n_train_proteins",
    "n_test_proteins",
    "n_train_rows",
    "n_test_rows",
    "train_live_mean_per_protein_r2",
    "train_live_pair_weighted_r2",
    "train_live_pair_weighted_mae",
    "train_live_pair_weighted_rmse",
    "train_pristine_mean_per_protein_r2",
    "train_pristine_pair_weighted_r2",
    "train_pristine_pair_weighted_mae",
    "train_pristine_pair_weighted_rmse",
    "test_mean_per_protein_r2",
    "test_pair_weighted_r2",
    "test_pair_weighted_mae",
    "test_pair_weighted_rmse",
    "elapsed_seconds",
]

MUTATION_AUDIT_FIELDNAMES = [
    "variant",
    "layer",
    "feature_dim",
    "copy_x",
    "fold",
    "n_train_rows",
    "any_x_change",
    "n_changed_entries",
    "changed_entry_fraction",
    "max_abs_x_change",
    "mean_abs_x_change",
    "rms_x_change",
    "train_live_vs_pristine_pred_max_abs_diff",
    "train_live_vs_pristine_pred_mean_abs_diff",
    "train_live_vs_pristine_pred_rmse",
    "train_live_vs_pristine_mean_per_protein_r2_delta",
    "train_live_vs_pristine_pair_weighted_r2_delta",
    "coef_l2_norm",
    "intercept",
    "elapsed_seconds",
]

COMPARISON_SUMMARY_FIELDNAMES = [
    "variant",
    "layer",
    "feature_dim",
    "fold",
    "n_train_proteins",
    "n_test_proteins",
    "n_train_rows",
    "n_test_rows",
    "false_any_x_change",
    "true_any_x_change",
    "false_max_abs_x_change",
    "true_max_abs_x_change",
    "false_mean_abs_x_change",
    "true_mean_abs_x_change",
    "false_train_live_vs_pristine_pair_weighted_r2_delta",
    "true_train_live_vs_pristine_pair_weighted_r2_delta",
    "false_train_live_vs_pristine_mean_per_protein_r2_delta",
    "true_train_live_vs_pristine_mean_per_protein_r2_delta",
    "coef_max_abs_delta_false_minus_true",
    "coef_l2_delta_false_minus_true",
    "intercept_delta_false_minus_true",
    "train_live_mean_per_protein_r2_delta_false_minus_true",
    "train_pristine_mean_per_protein_r2_delta_false_minus_true",
    "test_mean_per_protein_r2_delta_false_minus_true",
    "train_live_pair_weighted_r2_delta_false_minus_true",
    "train_pristine_pair_weighted_r2_delta_false_minus_true",
    "test_pair_weighted_r2_delta_false_minus_true",
]


def now_timestamp():
    """Return one readable local timestamp."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_progress(message):
    """Emit one timestamped progress line."""
    print(f"[{now_timestamp()}] {message}", flush=True)


def format_metric(value):
    """Stable human-readable formatting for summary text."""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, (np.integer, int)):
        return str(int(value))
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def safe_mean(rows, key):
    """Mean across existing numeric values for one key."""
    values = [float(row[key]) for row in rows if key in row]
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_mean_abs(rows, key):
    """Mean absolute value across one numeric column."""
    values = [abs(float(row[key])) for row in rows if key in row]
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def safe_max_abs(rows, key):
    """Max absolute value across one numeric column."""
    values = [abs(float(row[key])) for row in rows if key in row]
    if not values:
        return float("nan")
    return float(np.max(np.asarray(values, dtype=np.float64)))


def compute_vector_delta_stats(a, b):
    """Summarize differences between two dense vectors."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    diff = a - b
    abs_diff = np.abs(diff)
    return {
        "max_abs_diff": float(np.max(abs_diff)) if abs_diff.size else 0.0,
        "mean_abs_diff": float(np.mean(abs_diff)) if abs_diff.size else 0.0,
        "rmse_diff": float(np.sqrt(np.mean(diff * diff))) if diff.size else 0.0,
    }


def compute_matrix_mutation_stats(before, after, row_chunk_size=16384):
    """Summarize dense matrix mutation without materializing one full diff copy."""
    if before.shape != after.shape:
        raise ValueError(f"Shape mismatch for mutation audit: {before.shape} != {after.shape}")

    n_rows = int(before.shape[0])
    total_entries = int(before.size)
    n_changed = 0
    max_abs = 0.0
    sum_abs = 0.0
    sum_sq = 0.0

    for start in range(0, n_rows, row_chunk_size):
        end = min(start + row_chunk_size, n_rows)
        diff = np.asarray(after[start:end, :] - before[start:end, :], dtype=np.float64)
        if diff.size == 0:
            continue
        abs_diff = np.abs(diff)
        n_changed += int(np.count_nonzero(abs_diff))
        max_abs = max(max_abs, float(np.max(abs_diff)))
        sum_abs += float(np.sum(abs_diff, dtype=np.float64))
        sum_sq += float(np.sum(diff * diff, dtype=np.float64))

    changed_fraction = float(n_changed / total_entries) if total_entries else 0.0
    mean_abs = float(sum_abs / total_entries) if total_entries else 0.0
    rms = float(math.sqrt(sum_sq / total_entries)) if total_entries else 0.0
    return {
        "any_x_change": bool(n_changed > 0),
        "n_changed_entries": int(n_changed),
        "changed_entry_fraction": float(changed_fraction),
        "max_abs_x_change": float(max_abs),
        "mean_abs_x_change": float(mean_abs),
        "rms_x_change": float(rms),
    }


def fit_copy_setting(
    copy_x,
    fold_index,
    train_split,
    test_split,
    sample_weight,
):
    """Fit one sklearn LinearRegression copy_X setting and summarize its outputs."""
    fit_t0 = time.time()
    x_train_pristine = train_split["x"]
    x_train_work = x_train_pristine.copy()
    y_train = train_split["y"]
    x_test = test_split["x"]
    y_test = test_split["y"]

    model = LinearRegression(copy_X=copy_x)
    model.fit(x_train_work, y_train, sample_weight=sample_weight)

    mutation_stats = compute_matrix_mutation_stats(x_train_pristine, x_train_work)

    train_pred_live = np.asarray(model.predict(x_train_work), dtype=np.float64)
    train_pred_pristine = np.asarray(model.predict(x_train_pristine), dtype=np.float64)
    test_pred = np.asarray(model.predict(x_test), dtype=np.float64)

    train_live_metrics = summarize_split_from_predictions(
        train_split["offsets"],
        y_train,
        train_pred_live,
    )
    train_pristine_metrics = summarize_split_from_predictions(
        train_split["offsets"],
        y_train,
        train_pred_pristine,
    )
    test_metrics = summarize_split_from_predictions(
        test_split["offsets"],
        y_test,
        test_pred,
    )
    live_vs_pristine_pred_delta = compute_vector_delta_stats(train_pred_live, train_pred_pristine)

    elapsed_seconds = float(time.time() - fit_t0)
    result = {
        "copy_x": bool(copy_x),
        "coef": np.asarray(model.coef_, dtype=np.float64).copy(),
        "coef_l2_norm": float(np.linalg.norm(model.coef_)),
        "intercept": float(model.intercept_),
        "mutation_stats": mutation_stats,
        "train_live_metrics": train_live_metrics,
        "train_pristine_metrics": train_pristine_metrics,
        "test_metrics": test_metrics,
        "live_vs_pristine_pred_delta": live_vs_pristine_pred_delta,
        "elapsed_seconds": elapsed_seconds,
        "fold_summary_row": {
            "variant": VARIANT["name"],
            "layer": int(LAYER),
            "feature_dim": int(FEATURE_DIM),
            "copy_x": bool(copy_x),
            "fold": int(fold_index),
            "n_train_proteins": int(len(train_split["offsets"])),
            "n_test_proteins": int(len(test_split["offsets"])),
            "n_train_rows": int(train_split["n_rows"]),
            "n_test_rows": int(test_split["n_rows"]),
            "train_live_mean_per_protein_r2": float(train_live_metrics["mean_per_protein_r2"]),
            "train_live_pair_weighted_r2": float(train_live_metrics["pair_weighted_r2"]),
            "train_live_pair_weighted_mae": float(train_live_metrics["pair_weighted_mae"]),
            "train_live_pair_weighted_rmse": float(train_live_metrics["pair_weighted_rmse"]),
            "train_pristine_mean_per_protein_r2": float(
                train_pristine_metrics["mean_per_protein_r2"]
            ),
            "train_pristine_pair_weighted_r2": float(
                train_pristine_metrics["pair_weighted_r2"]
            ),
            "train_pristine_pair_weighted_mae": float(
                train_pristine_metrics["pair_weighted_mae"]
            ),
            "train_pristine_pair_weighted_rmse": float(
                train_pristine_metrics["pair_weighted_rmse"]
            ),
            "test_mean_per_protein_r2": float(test_metrics["mean_per_protein_r2"]),
            "test_pair_weighted_r2": float(test_metrics["pair_weighted_r2"]),
            "test_pair_weighted_mae": float(test_metrics["pair_weighted_mae"]),
            "test_pair_weighted_rmse": float(test_metrics["pair_weighted_rmse"]),
            "elapsed_seconds": float(elapsed_seconds),
        },
        "mutation_row": {
            "variant": VARIANT["name"],
            "layer": int(LAYER),
            "feature_dim": int(FEATURE_DIM),
            "copy_x": bool(copy_x),
            "fold": int(fold_index),
            "n_train_rows": int(train_split["n_rows"]),
            "any_x_change": bool(mutation_stats["any_x_change"]),
            "n_changed_entries": int(mutation_stats["n_changed_entries"]),
            "changed_entry_fraction": float(mutation_stats["changed_entry_fraction"]),
            "max_abs_x_change": float(mutation_stats["max_abs_x_change"]),
            "mean_abs_x_change": float(mutation_stats["mean_abs_x_change"]),
            "rms_x_change": float(mutation_stats["rms_x_change"]),
            "train_live_vs_pristine_pred_max_abs_diff": float(
                live_vs_pristine_pred_delta["max_abs_diff"]
            ),
            "train_live_vs_pristine_pred_mean_abs_diff": float(
                live_vs_pristine_pred_delta["mean_abs_diff"]
            ),
            "train_live_vs_pristine_pred_rmse": float(
                live_vs_pristine_pred_delta["rmse_diff"]
            ),
            "train_live_vs_pristine_mean_per_protein_r2_delta": float(
                train_live_metrics["mean_per_protein_r2"]
                - train_pristine_metrics["mean_per_protein_r2"]
            ),
            "train_live_vs_pristine_pair_weighted_r2_delta": float(
                train_live_metrics["pair_weighted_r2"]
                - train_pristine_metrics["pair_weighted_r2"]
            ),
            "coef_l2_norm": float(np.linalg.norm(model.coef_)),
            "intercept": float(model.intercept_),
            "elapsed_seconds": float(elapsed_seconds),
        },
    }

    del x_train_work
    del train_pred_live
    del train_pred_pristine
    del test_pred
    return result


def build_comparison_row(fold, false_result, true_result):
    """Compare copy_X=False and copy_X=True for one fold."""
    false_row = false_result["fold_summary_row"]
    true_row = true_result["fold_summary_row"]
    false_mut = false_result["mutation_row"]
    true_mut = true_result["mutation_row"]
    coef_false = false_result["coef"]
    coef_true = true_result["coef"]
    coef_diff = np.asarray(coef_false - coef_true, dtype=np.float64)

    return {
        "variant": VARIANT["name"],
        "layer": int(LAYER),
        "feature_dim": int(FEATURE_DIM),
        "fold": int(fold["fold_index"]),
        "n_train_proteins": int(len(fold["train_records"])),
        "n_test_proteins": int(len(fold["test_records"])),
        "n_train_rows": int(false_row["n_train_rows"]),
        "n_test_rows": int(false_row["n_test_rows"]),
        "false_any_x_change": bool(false_mut["any_x_change"]),
        "true_any_x_change": bool(true_mut["any_x_change"]),
        "false_max_abs_x_change": float(false_mut["max_abs_x_change"]),
        "true_max_abs_x_change": float(true_mut["max_abs_x_change"]),
        "false_mean_abs_x_change": float(false_mut["mean_abs_x_change"]),
        "true_mean_abs_x_change": float(true_mut["mean_abs_x_change"]),
        "false_train_live_vs_pristine_pair_weighted_r2_delta": float(
            false_mut["train_live_vs_pristine_pair_weighted_r2_delta"]
        ),
        "true_train_live_vs_pristine_pair_weighted_r2_delta": float(
            true_mut["train_live_vs_pristine_pair_weighted_r2_delta"]
        ),
        "false_train_live_vs_pristine_mean_per_protein_r2_delta": float(
            false_mut["train_live_vs_pristine_mean_per_protein_r2_delta"]
        ),
        "true_train_live_vs_pristine_mean_per_protein_r2_delta": float(
            true_mut["train_live_vs_pristine_mean_per_protein_r2_delta"]
        ),
        "coef_max_abs_delta_false_minus_true": (
            float(np.max(np.abs(coef_diff))) if coef_diff.size else 0.0
        ),
        "coef_l2_delta_false_minus_true": float(np.linalg.norm(coef_diff)),
        "intercept_delta_false_minus_true": float(
            false_result["intercept"] - true_result["intercept"]
        ),
        "train_live_mean_per_protein_r2_delta_false_minus_true": float(
            false_row["train_live_mean_per_protein_r2"]
            - true_row["train_live_mean_per_protein_r2"]
        ),
        "train_pristine_mean_per_protein_r2_delta_false_minus_true": float(
            false_row["train_pristine_mean_per_protein_r2"]
            - true_row["train_pristine_mean_per_protein_r2"]
        ),
        "test_mean_per_protein_r2_delta_false_minus_true": float(
            false_row["test_mean_per_protein_r2"] - true_row["test_mean_per_protein_r2"]
        ),
        "train_live_pair_weighted_r2_delta_false_minus_true": float(
            false_row["train_live_pair_weighted_r2"] - true_row["train_live_pair_weighted_r2"]
        ),
        "train_pristine_pair_weighted_r2_delta_false_minus_true": float(
            false_row["train_pristine_pair_weighted_r2"]
            - true_row["train_pristine_pair_weighted_r2"]
        ),
        "test_pair_weighted_r2_delta_false_minus_true": float(
            false_row["test_pair_weighted_r2"] - true_row["test_pair_weighted_r2"]
        ),
    }


def build_summary_text(
    path,
    output_dir,
    loaded_records,
    skipped,
    size_bin_counts,
    fold_summary_rows,
    mutation_rows,
    comparison_rows,
    elapsed_seconds,
):
    """Write one compact text summary for the diagnostic."""
    false_rows = [row for row in fold_summary_rows if not bool(row["copy_x"])]
    true_rows = [row for row in fold_summary_rows if bool(row["copy_x"])]
    false_mut = [row for row in mutation_rows if not bool(row["copy_x"])]
    true_mut = [row for row in mutation_rows if bool(row["copy_x"])]

    interpretation = "No completed folds yet."
    if comparison_rows:
        mean_false_mut = safe_mean(false_mut, "max_abs_x_change")
        mean_true_mut = safe_mean(true_mut, "max_abs_x_change")
        mean_abs_train_live_delta = safe_mean_abs(
            comparison_rows,
            "train_live_pair_weighted_r2_delta_false_minus_true",
        )
        mean_abs_train_pristine_delta = safe_mean_abs(
            comparison_rows,
            "train_pristine_pair_weighted_r2_delta_false_minus_true",
        )
        mean_abs_test_delta = safe_mean_abs(
            comparison_rows,
            "test_pair_weighted_r2_delta_false_minus_true",
        )
        if mean_false_mut > 0.0 and mean_abs_train_live_delta > 0.0:
            if mean_abs_train_pristine_delta <= 1e-12 and mean_abs_test_delta <= 1e-12:
                interpretation = (
                    "copy_X=False mutates dense x_train in place, and the observed "
                    "difference is confined to train metrics computed on the mutated "
                    "post-fit matrix. Coefficients, pristine-train metrics, and test "
                    "metrics match copy_X=True to numerical precision."
                )
            else:
                interpretation = (
                    "copy_X=False mutates dense x_train in place, and at least one "
                    "downstream metric delta extends beyond live post-fit train "
                    "evaluation. Inspect the comparison CSV for fold-level details."
                )
        elif mean_false_mut > 0.0:
            interpretation = (
                "copy_X=False mutates dense x_train in place, but the downstream "
                "metric deltas are negligible at the fold-summary level."
            )
        else:
            interpretation = (
                "No in-place x_train mutation was detected for copy_X=False in the "
                "completed folds."
            )

    with path.open("w") as handle:
        handle.write(f"{TITLE}\n")
        handle.write(f"{'=' * len(TITLE)}\n\n")
        handle.write(f"Timestamp:        {now_timestamp()}\n")
        handle.write(f"Output dir:       {output_dir}\n")
        handle.write(f"Variant:          {VARIANT['name']}\n")
        handle.write(f"Layer:            {LAYER}\n")
        handle.write(f"Feature dim:      {FEATURE_DIM}\n")
        handle.write(f"Outer folds:      {N_FOLDS} (strict size-stratified)\n")
        handle.write(f"Size bins:        {N_SIZE_BINS}\n")
        handle.write(f"sklearn version:  {sklearn.__version__}\n")
        handle.write(f"Loaded proteins:  {len(loaded_records)}\n")
        handle.write(f"Skipped proteins: {len(skipped)}\n")
        handle.write(f"Size bin counts:  {size_bin_counts}\n")
        handle.write(f"Completed folds:  {len(comparison_rows)} / {N_FOLDS}\n")
        handle.write(f"Elapsed seconds:  {elapsed_seconds:.1f}\n\n")

        for label, rows in [("copy_X=False", false_rows), ("copy_X=True", true_rows)]:
            handle.write(f"{label}\n")
            handle.write(f"{'-' * len(label)}\n")
            handle.write(
                f"Mean train live pair-R2:      {format_metric(safe_mean(rows, 'train_live_pair_weighted_r2'))}\n"
            )
            handle.write(
                f"Mean train pristine pair-R2:  {format_metric(safe_mean(rows, 'train_pristine_pair_weighted_r2'))}\n"
            )
            handle.write(
                f"Mean test pair-R2:            {format_metric(safe_mean(rows, 'test_pair_weighted_r2'))}\n"
            )
            handle.write(
                f"Mean train live mean-R2:      {format_metric(safe_mean(rows, 'train_live_mean_per_protein_r2'))}\n"
            )
            handle.write(
                f"Mean train pristine mean-R2:  {format_metric(safe_mean(rows, 'train_pristine_mean_per_protein_r2'))}\n"
            )
            handle.write(
                f"Mean test mean-R2:            {format_metric(safe_mean(rows, 'test_mean_per_protein_r2'))}\n"
            )
            handle.write("\n")

        handle.write("Mutation Audit\n")
        handle.write("--------------\n")
        handle.write(
            f"Mean max |x_after - x_before|, copy_X=False: {format_metric(safe_mean(false_mut, 'max_abs_x_change'))}\n"
        )
        handle.write(
            f"Mean max |x_after - x_before|, copy_X=True:  {format_metric(safe_mean(true_mut, 'max_abs_x_change'))}\n"
        )
        handle.write(
            f"Mean live-vs-pristine train pair-R2 delta, copy_X=False: {format_metric(safe_mean(false_mut, 'train_live_vs_pristine_pair_weighted_r2_delta'))}\n"
        )
        handle.write(
            f"Mean live-vs-pristine train pair-R2 delta, copy_X=True:  {format_metric(safe_mean(true_mut, 'train_live_vs_pristine_pair_weighted_r2_delta'))}\n"
        )
        handle.write("\n")

        handle.write("copy_X=False minus copy_X=True\n")
        handle.write("-----------------------------\n")
        handle.write(
            f"Mean |coef delta|_max:        {format_metric(safe_mean(comparison_rows, 'coef_max_abs_delta_false_minus_true'))}\n"
        )
        handle.write(
            f"Max  |coef delta|_max:        {format_metric(safe_max_abs(comparison_rows, 'coef_max_abs_delta_false_minus_true'))}\n"
        )
        handle.write(
            f"Mean |train live pair-R2 delta|:     {format_metric(safe_mean_abs(comparison_rows, 'train_live_pair_weighted_r2_delta_false_minus_true'))}\n"
        )
        handle.write(
            f"Mean |train pristine pair-R2 delta|: {format_metric(safe_mean_abs(comparison_rows, 'train_pristine_pair_weighted_r2_delta_false_minus_true'))}\n"
        )
        handle.write(
            f"Mean |test pair-R2 delta|:           {format_metric(safe_mean_abs(comparison_rows, 'test_pair_weighted_r2_delta_false_minus_true'))}\n"
        )
        handle.write(
            f"Mean |train live mean-R2 delta|:     {format_metric(safe_mean_abs(comparison_rows, 'train_live_mean_per_protein_r2_delta_false_minus_true'))}\n"
        )
        handle.write(
            f"Mean |train pristine mean-R2 delta|: {format_metric(safe_mean_abs(comparison_rows, 'train_pristine_mean_per_protein_r2_delta_false_minus_true'))}\n"
        )
        handle.write(
            f"Mean |test mean-R2 delta|:           {format_metric(safe_mean_abs(comparison_rows, 'test_mean_per_protein_r2_delta_false_minus_true'))}\n"
        )
        handle.write("\n")

        handle.write("Interpretation\n")
        handle.write("--------------\n")
        handle.write(f"{interpretation}\n")


def flush_outputs(
    output_dir,
    loaded_records,
    skipped,
    size_bin_counts,
    fold_summary_rows,
    mutation_rows,
    comparison_rows,
    fold_assignment_rows,
    elapsed_seconds,
):
    """Rewrite all compact outputs after each completed fold."""
    write_csv(output_dir / FOLD_SUMMARY_CSV_NAME, FOLD_SUMMARY_FIELDNAMES, fold_summary_rows)
    write_csv(output_dir / MUTATION_AUDIT_CSV_NAME, MUTATION_AUDIT_FIELDNAMES, mutation_rows)
    write_csv(
        output_dir / COMPARISON_SUMMARY_CSV_NAME,
        COMPARISON_SUMMARY_FIELDNAMES,
        comparison_rows,
    )
    write_csv(
        output_dir / FOLD_ASSIGNMENTS_CSV_NAME,
        FOLD_ASSIGNMENT_FIELDNAMES,
        fold_assignment_rows,
    )
    build_summary_text(
        output_dir / SUMMARY_TXT_NAME,
        output_dir=output_dir,
        loaded_records=loaded_records,
        skipped=skipped,
        size_bin_counts=size_bin_counts,
        fold_summary_rows=fold_summary_rows,
        mutation_rows=mutation_rows,
        comparison_rows=comparison_rows,
        elapsed_seconds=elapsed_seconds,
    )


def main():
    """Run the fixed baseline256 layer-47 copy_X diagnostic."""
    t0 = time.time()
    pair_blocks_root_env = os.environ.get("PAIR_BLOCKS_ROOT")
    pdb_root_env = os.environ.get("PDB_ROOT")
    csv_root_env = os.environ.get("CSV_ROOT")
    if not pair_blocks_root_env or not pdb_root_env or not csv_root_env:
        raise ValueError("PAIR_BLOCKS_ROOT, PDB_ROOT, and CSV_ROOT must all be set.")

    pair_blocks_root = Path(pair_blocks_root_env).resolve()
    pdb_root = Path(pdb_root_env).resolve()
    output_dir = Path(csv_root_env).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))
    log_progress(f"Pair blocks root: {pair_blocks_root}")
    log_progress(f"PDB root:         {pdb_root}")
    log_progress(f"Output dir:       {output_dir}")
    log_progress(f"Variant:          {VARIANT['name']}")
    log_progress(f"Layer:            {LAYER}")
    log_progress(f"Feature dim:      {FEATURE_DIM}")
    log_progress(f"sklearn version:  {sklearn.__version__}")

    all_proteins, records, skipped = load_protein_records(pair_blocks_root, pdb_root)
    if not records:
        log_progress("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    assign_size_bins(records, N_SIZE_BINS)
    folds, fold_assignment_rows, size_bin_counts = build_strict_stratified_folds(records)
    log_progress(f"Loaded proteins:  {len(records)}")
    log_progress(f"Size bin counts:  {size_bin_counts}")

    fold_summary_rows = []
    mutation_rows = []
    comparison_rows = []
    flush_outputs(
        output_dir=output_dir,
        loaded_records=records,
        skipped=skipped,
        size_bin_counts=size_bin_counts,
        fold_summary_rows=fold_summary_rows,
        mutation_rows=mutation_rows,
        comparison_rows=comparison_rows,
        fold_assignment_rows=fold_assignment_rows,
        elapsed_seconds=float(time.time() - t0),
    )

    log_progress(f"Layer {LAYER:02d}: starting cache build")
    cache_t0 = time.time()
    layer_cache = load_layer_feature_cache(records, LAYER, VARIANT)
    log_progress(
        f"Layer {LAYER:02d}: finished cache build with {len(layer_cache)} proteins in "
        f"{time.time() - cache_t0:.1f}s"
    )

    for fold in folds:
        fold_index = int(fold["fold_index"])
        fold_t0 = time.time()
        train_records = fold["train_records"]
        test_records = fold["test_records"]
        n_train_rows = int(sum(record["n_pairs"] for record in train_records))
        n_test_rows = int(sum(record["n_pairs"] for record in test_records))
        log_progress(
            f"Fold {fold_index}: starting comparison "
            f"({len(train_records)} train / {len(test_records)} test proteins, "
            f"{n_train_rows:,} train rows, {n_test_rows:,} test rows)"
        )

        scaler_mean, scaler_scale = compute_scaler_from_cache(train_records, layer_cache, VARIANT)
        train_split = build_standardized_split(
            train_records,
            layer_cache,
            scaler_mean,
            scaler_scale,
            VARIANT,
        )
        test_split = build_standardized_split(
            test_records,
            layer_cache,
            scaler_mean,
            scaler_scale,
            VARIANT,
        )
        sample_weight = compute_protein_equal_weights(train_records)

        per_setting_results = {}
        for copy_x in COPY_SETTINGS:
            log_progress(f"Fold {fold_index}: fitting copy_X={copy_x}")
            result = fit_copy_setting(
                copy_x=copy_x,
                fold_index=fold_index,
                train_split=train_split,
                test_split=test_split,
                sample_weight=sample_weight,
            )
            per_setting_results[bool(copy_x)] = result
            fold_summary_rows.append(result["fold_summary_row"])
            mutation_rows.append(result["mutation_row"])
            log_progress(
                f"Fold {fold_index}: copy_X={copy_x} complete in "
                f"{result['elapsed_seconds']:.1f}s "
                f"(max |dx|={result['mutation_row']['max_abs_x_change']:.6g})"
            )

        comparison_rows.append(
            build_comparison_row(
                fold,
                false_result=per_setting_results[False],
                true_result=per_setting_results[True],
            )
        )

        flush_outputs(
            output_dir=output_dir,
            loaded_records=records,
            skipped=skipped,
            size_bin_counts=size_bin_counts,
            fold_summary_rows=fold_summary_rows,
            mutation_rows=mutation_rows,
            comparison_rows=comparison_rows,
            fold_assignment_rows=fold_assignment_rows,
            elapsed_seconds=float(time.time() - t0),
        )
        log_progress(f"Fold {fold_index}: flushed outputs in {time.time() - fold_t0:.1f}s")

        del train_split
        del test_split
        del sample_weight
        del per_setting_results
        gc.collect()

    del layer_cache
    gc.collect()
    log_progress(f"Completed diagnostic in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()

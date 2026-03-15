#!/usr/bin/env python3
"""
diagnostic_sampling_weighting_kfold_regression.py
================================================
Diagnostic ablation for layer-47 lower-triangle linear regression across
protein-level 5-fold CV. The goal is to isolate whether strong capped
baselines come from train-pair caps, protein-equal weighting, or fold design.

Outputs (inside CSV_ROOT/):
    diagnostic_sampling_weighting_kfold_variant_summary.csv
    diagnostic_sampling_weighting_kfold_fold_summary.csv
    diagnostic_sampling_weighting_kfold_per_protein.csv
    diagnostic_sampling_weighting_kfold_fold_assignments.csv
    diagnostic_sampling_weighting_kfold_train_sampling_audit.csv
    diagnostic_sampling_weighting_kfold_summary.txt
"""

import gc
import math
import time

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, StratifiedKFold

from diagnostic_tree_regression_common import (
    CSV_DIR,
    FLOAT_EPS,
    LAYER,
    MAX_TRAIN_PAIRS_PER_PROTEIN,
    N_CHANNELS,
    RANDOM_STATE,
    compute_pair_weights,
    deterministic_sample_indices,
    get_pair_block_path,
    load_protein_records,
    summarize_predictions,
    write_csv,
)


TITLE = "Diagnostic Sampling/Weighting K-Fold Regression — Layer 47"

N_FOLDS = 5
N_SIZE_BINS = 5
FOLD_FAMILY_RANDOM = "random"
FOLD_FAMILY_STRATIFIED = "stratified_size_bins"

SUMMARY_CSV_NAME = "diagnostic_sampling_weighting_kfold_variant_summary.csv"
FOLD_SUMMARY_CSV_NAME = "diagnostic_sampling_weighting_kfold_fold_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_sampling_weighting_kfold_per_protein.csv"
FOLD_ASSIGNMENTS_CSV_NAME = "diagnostic_sampling_weighting_kfold_fold_assignments.csv"
TRAIN_AUDIT_CSV_NAME = "diagnostic_sampling_weighting_kfold_train_sampling_audit.csv"
SUMMARY_TXT_NAME = "diagnostic_sampling_weighting_kfold_summary.txt"

VARIANTS = [
    {
        "name": "all_pairs_protein_equal_stratified_kfold",
        "fold_family": FOLD_FAMILY_STRATIFIED,
        "sample_cap": None,
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "all_pairs_protein_equal_random_kfold",
        "fold_family": FOLD_FAMILY_RANDOM,
        "sample_cap": None,
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "cap5k_protein_equal_stratified_kfold",
        "fold_family": FOLD_FAMILY_STRATIFIED,
        "sample_cap": MAX_TRAIN_PAIRS_PER_PROTEIN,
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "cap5k_protein_equal_random_kfold",
        "fold_family": FOLD_FAMILY_RANDOM,
        "sample_cap": MAX_TRAIN_PAIRS_PER_PROTEIN,
        "uses_protein_equal_weighting": True,
    },
    {
        "name": "cap5k_pair_weighted_random_kfold",
        "fold_family": FOLD_FAMILY_RANDOM,
        "sample_cap": MAX_TRAIN_PAIRS_PER_PROTEIN,
        "uses_protein_equal_weighting": False,
    },
]

SUMMARY_FIELDNAMES = [
    "variant",
    "fold_family",
    "sample_cap_per_protein",
    "uses_protein_equal_weighting",
    "n_folds",
    "n_loaded_proteins",
    "mean_n_train_proteins",
    "mean_n_test_proteins",
    "mean_n_train_rows_used",
    "mean_n_test_rows_used",
    "mean_test_mean_per_protein_r2",
    "std_test_mean_per_protein_r2",
    "mean_test_median_per_protein_r2",
    "std_test_median_per_protein_r2",
    "mean_test_negative_r2_fraction",
    "std_test_negative_r2_fraction",
    "mean_test_pair_weighted_r2",
    "std_test_pair_weighted_r2",
    "mean_test_pair_weighted_mae",
    "std_test_pair_weighted_mae",
    "mean_test_pair_weighted_rmse",
    "std_test_pair_weighted_rmse",
]

FOLD_SUMMARY_FIELDNAMES = [
    "variant",
    "fold_family",
    "fold",
    "sample_cap_per_protein",
    "uses_protein_equal_weighting",
    "n_train_proteins",
    "n_test_proteins",
    "n_train_rows_used",
    "n_test_rows_used",
    "test_mean_per_protein_r2",
    "test_median_per_protein_r2",
    "test_negative_r2_fraction",
    "test_pair_weighted_r2",
    "test_pair_weighted_mae",
    "test_pair_weighted_rmse",
    "elapsed_seconds",
]

PER_PROTEIN_FIELDNAMES = [
    "variant",
    "fold_family",
    "fold",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs_total",
    "n_pairs_used_test",
    "per_protein_r2",
    "mae",
    "rmse",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
]

FOLD_ASSIGNMENT_FIELDNAMES = [
    "fold_family",
    "fold",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
]

TRAIN_AUDIT_FIELDNAMES = [
    "variant",
    "fold_family",
    "fold",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs_available",
    "n_pairs_used",
    "protein_total_train_weight",
]


def assign_rank_size_bins(records, n_bins):
    """Assign rank-based size bins in place."""
    sorted_records = sorted(
        records,
        key=lambda record: (record["n_residues"], record["protein_id"]),
    )
    total = len(sorted_records)
    for rank, record in enumerate(sorted_records):
        size_bin = int((rank * n_bins) // max(total, 1))
        if size_bin >= n_bins:
            size_bin = n_bins - 1
        record["size_bin"] = int(size_bin)


def build_fold_family(records, family_name, stratified):
    """Build one reusable protein-level fold family."""
    indices = np.arange(len(records), dtype=np.int64)
    labels = np.asarray([record["size_bin"] for record in records], dtype=np.int32)

    if stratified:
        _, counts = np.unique(labels, return_counts=True)
        if np.any(counts < N_FOLDS):
            raise ValueError(
                f"Need at least {N_FOLDS} proteins per size bin for stratified folds"
            )
        splitter = StratifiedKFold(
            n_splits=N_FOLDS,
            shuffle=True,
            random_state=RANDOM_STATE,
        )
        split_iter = splitter.split(indices, labels)
    else:
        splitter = KFold(n_splits=N_FOLDS, shuffle=True, random_state=RANDOM_STATE)
        split_iter = splitter.split(indices)

    folds = []
    assignment_rows = []
    for fold_index, (train_idx, test_idx) in enumerate(split_iter):
        train_idx = np.asarray(train_idx, dtype=np.int64)
        test_idx = np.asarray(test_idx, dtype=np.int64)
        train_records = [records[int(idx)] for idx in train_idx]
        test_records = [records[int(idx)] for idx in test_idx]
        folds.append(
            {
                "fold_family": family_name,
                "fold_index": int(fold_index),
                "train_records": train_records,
                "test_records": test_records,
            }
        )
        for split_name, split_idx in [("train", train_idx), ("test", test_idx)]:
            for idx in split_idx:
                record = records[int(idx)]
                assignment_rows.append(
                    {
                        "fold_family": family_name,
                        "fold": int(fold_index),
                        "split": split_name,
                        "protein_id": record["protein_id"],
                        "n_residues": int(record["n_residues"]),
                        "size_bin": int(record["size_bin"]),
                        "n_pairs": int(record["n_pairs"]),
                    }
                )
    return {"name": family_name, "folds": folds, "assignment_rows": assignment_rows}


def preload_lower_triangle_chunks(records):
    """Load layer-47 lower-triangle features once per protein."""
    chunk_map = {}
    total_rows = 0
    for record_index, record in enumerate(records, start=1):
        pb_path = get_pair_block_path(record["protein_dir"], record["protein_id"], LAYER)
        pb = np.load(str(pb_path), mmap_mode="r")
        chunk = np.asarray(pb[record["lt_i"], record["lt_j"], :], dtype=np.float32)
        del pb

        if chunk.shape[0] != int(record["n_pairs"]):
            raise ValueError(
                f"Lower-triangle chunk row mismatch for {record['protein_id']}: "
                f"{chunk.shape[0]} != {record['n_pairs']}"
            )

        chunk_map[record["protein_id"]] = chunk
        total_rows += int(chunk.shape[0])

        if record_index % 25 == 0 or record_index == len(records):
            print(
                f"  Cached lower-triangle chunks for {record_index}/{len(records)} proteins"
            )
    return chunk_map, int(total_rows)


def build_lower_triangle_bundle(records, chunk_map, sample_cap):
    """Materialize one train/test fold into dense lower-triangle matrices."""
    row_counts = []
    selections = []
    total_rows = 0

    for record in records:
        local_idx = deterministic_sample_indices(
            label=f"{record['protein_id']}:{record['n_pairs']}",
            n_rows=record["n_pairs"],
            cap=sample_cap,
            offset=0,
        )
        selections.append(local_idx)
        row_counts.append(int(local_idx.size))
        total_rows += int(local_idx.size)

    x = np.empty((total_rows, N_CHANNELS), dtype=np.float32)
    y = np.empty((total_rows,), dtype=np.float64)
    offsets = []

    start = 0
    for record, local_idx, n_rows in zip(records, selections, row_counts):
        end = start + n_rows
        chunk = chunk_map[record["protein_id"]]
        x[start:end, :] = chunk[local_idx, :]
        y[start:end] = record["y_abs"][local_idx]
        offsets.append((start, end))
        start = end

    row_counts_arr = np.asarray(row_counts, dtype=np.int32)
    return {
        "x": x,
        "y": y,
        "offsets": offsets,
        "row_counts": row_counts_arr,
        "weights": compute_pair_weights(row_counts),
    }


def compute_standardization_from_bundle(bundle, protein_equal_weighting):
    """Compute train-only feature moments from the actual rows used."""
    mean_acc = np.zeros((N_CHANNELS,), dtype=np.float64)
    sq_acc = np.zeros((N_CHANNELS,), dtype=np.float64)
    total_weight = 0.0

    for (start, end), n_rows in zip(bundle["offsets"], bundle["row_counts"]):
        block = bundle["x"][start:end, :]
        block64 = np.asarray(block, dtype=np.float64)
        block_mean = np.mean(block64, axis=0, dtype=np.float64)
        block_sq_mean = np.mean(block64 * block64, axis=0, dtype=np.float64)
        weight = 1.0 if protein_equal_weighting else float(n_rows)
        mean_acc += weight * block_mean
        sq_acc += weight * block_sq_mean
        total_weight += weight

    mean = mean_acc / max(total_weight, 1.0)
    second_moment = sq_acc / max(total_weight, 1.0)
    variance = np.maximum(second_moment - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale[scale < FLOAT_EPS] = 1.0
    return {
        "mean": mean.astype(np.float32, copy=False),
        "scale": scale.astype(np.float32, copy=False),
    }


def apply_standardization_in_place(bundle, standardization):
    """Apply a train-fitted standardization transform to a bundle."""
    bundle["x"] -= standardization["mean"]
    bundle["x"] /= standardization["scale"]


def fit_linear_model(bundle, use_protein_equal_weighting):
    """Fit one lower-triangle linear regression model."""
    model = LinearRegression(copy_X=False)
    if use_protein_equal_weighting:
        model.fit(bundle["x"], bundle["y"], sample_weight=bundle["weights"])
    else:
        model.fit(bundle["x"], bundle["y"])
    return model


def build_per_protein_rows(variant, fold_info, test_records, metrics):
    """Format long-form test-protein rows for one fold."""
    rows = []
    for record, metric_row in zip(test_records, metrics["per_rows"]):
        rows.append(
            {
                "variant": variant["name"],
                "fold_family": variant["fold_family"],
                "fold": int(fold_info["fold_index"]),
                "protein_id": record["protein_id"],
                "n_residues": int(record["n_residues"]),
                "size_bin": int(record["size_bin"]),
                "n_pairs_total": int(record["n_pairs"]),
                "n_pairs_used_test": int(metric_row["n_pairs_used"]),
                "per_protein_r2": float(metric_row["per_protein_r2"]),
                "mae": float(metric_row["per_protein_mae"]),
                "rmse": float(metric_row["per_protein_rmse"]),
                "mean_true_distance": float(metric_row["mean_true_distance"]),
                "mean_predicted_distance": float(metric_row["mean_predicted_distance"]),
                "mean_error_pred_minus_true": float(metric_row["mean_error_pred_minus_true"]),
            }
        )
    return rows


def build_train_sampling_audit_rows(variant, fold_info, train_records, train_bundle):
    """Record how many train pairs and how much total weight each protein received."""
    rows = []
    use_protein_equal = bool(variant["uses_protein_equal_weighting"])
    for record, (start, end), n_rows in zip(
        train_records,
        train_bundle["offsets"],
        train_bundle["row_counts"],
    ):
        if use_protein_equal:
            total_train_weight = float(np.sum(train_bundle["weights"][start:end]))
        else:
            total_train_weight = float(n_rows)
        rows.append(
            {
                "variant": variant["name"],
                "fold_family": variant["fold_family"],
                "fold": int(fold_info["fold_index"]),
                "protein_id": record["protein_id"],
                "n_residues": int(record["n_residues"]),
                "size_bin": int(record["size_bin"]),
                "n_pairs_available": int(record["n_pairs"]),
                "n_pairs_used": int(n_rows),
                "protein_total_train_weight": total_train_weight,
            }
        )
    return rows


def evaluate_variant_on_fold(variant, fold_info, chunk_map):
    """Run one variant on one fold and collect outputs."""
    t0 = time.time()
    train_records = fold_info["train_records"]
    test_records = fold_info["test_records"]

    train_bundle = build_lower_triangle_bundle(
        train_records,
        chunk_map,
        sample_cap=variant["sample_cap"],
    )
    test_bundle = build_lower_triangle_bundle(
        test_records,
        chunk_map,
        sample_cap=None,
    )

    standardization = compute_standardization_from_bundle(
        train_bundle,
        protein_equal_weighting=bool(variant["uses_protein_equal_weighting"]),
    )
    apply_standardization_in_place(train_bundle, standardization)
    apply_standardization_in_place(test_bundle, standardization)

    model = fit_linear_model(
        train_bundle,
        use_protein_equal_weighting=bool(variant["uses_protein_equal_weighting"]),
    )
    y_pred_test = np.asarray(model.predict(test_bundle["x"]), dtype=np.float64)
    test_metrics = summarize_predictions(
        "test",
        test_records,
        test_bundle["offsets"],
        test_bundle["y"],
        y_pred_test,
    )

    fold_summary_row = {
        "variant": variant["name"],
        "fold_family": variant["fold_family"],
        "fold": int(fold_info["fold_index"]),
        "sample_cap_per_protein": (
            ""
            if variant["sample_cap"] is None
            else int(variant["sample_cap"])
        ),
        "uses_protein_equal_weighting": bool(variant["uses_protein_equal_weighting"]),
        "n_train_proteins": int(len(train_records)),
        "n_test_proteins": int(len(test_records)),
        "n_train_rows_used": int(train_bundle["x"].shape[0]),
        "n_test_rows_used": int(test_bundle["x"].shape[0]),
        "test_mean_per_protein_r2": float(test_metrics["mean_per_protein_r2"]),
        "test_median_per_protein_r2": float(test_metrics["median_per_protein_r2"]),
        "test_negative_r2_fraction": float(test_metrics["negative_r2_fraction"]),
        "test_pair_weighted_r2": float(test_metrics["pair_weighted_r2"]),
        "test_pair_weighted_mae": float(test_metrics["pair_weighted_mae"]),
        "test_pair_weighted_rmse": float(test_metrics["pair_weighted_rmse"]),
        "elapsed_seconds": float(time.time() - t0),
    }

    per_rows = build_per_protein_rows(variant, fold_info, test_records, test_metrics)
    audit_rows = build_train_sampling_audit_rows(
        variant,
        fold_info,
        train_records,
        train_bundle,
    )

    del train_bundle
    del test_bundle
    del y_pred_test
    gc.collect()

    return {
        "fold_summary_row": fold_summary_row,
        "per_rows": per_rows,
        "audit_rows": audit_rows,
    }


def aggregate_variant_rows(variant, fold_rows, n_loaded_proteins):
    """Aggregate one variant's fold rows into a single summary row."""
    def mean_of(field_name):
        values = np.asarray([float(row[field_name]) for row in fold_rows], dtype=np.float64)
        return float(np.mean(values))

    def std_of(field_name):
        values = np.asarray([float(row[field_name]) for row in fold_rows], dtype=np.float64)
        return float(np.std(values))

    return {
        "variant": variant["name"],
        "fold_family": variant["fold_family"],
        "sample_cap_per_protein": (
            ""
            if variant["sample_cap"] is None
            else int(variant["sample_cap"])
        ),
        "uses_protein_equal_weighting": bool(variant["uses_protein_equal_weighting"]),
        "n_folds": int(len(fold_rows)),
        "n_loaded_proteins": int(n_loaded_proteins),
        "mean_n_train_proteins": mean_of("n_train_proteins"),
        "mean_n_test_proteins": mean_of("n_test_proteins"),
        "mean_n_train_rows_used": mean_of("n_train_rows_used"),
        "mean_n_test_rows_used": mean_of("n_test_rows_used"),
        "mean_test_mean_per_protein_r2": mean_of("test_mean_per_protein_r2"),
        "std_test_mean_per_protein_r2": std_of("test_mean_per_protein_r2"),
        "mean_test_median_per_protein_r2": mean_of("test_median_per_protein_r2"),
        "std_test_median_per_protein_r2": std_of("test_median_per_protein_r2"),
        "mean_test_negative_r2_fraction": mean_of("test_negative_r2_fraction"),
        "std_test_negative_r2_fraction": std_of("test_negative_r2_fraction"),
        "mean_test_pair_weighted_r2": mean_of("test_pair_weighted_r2"),
        "std_test_pair_weighted_r2": std_of("test_pair_weighted_r2"),
        "mean_test_pair_weighted_mae": mean_of("test_pair_weighted_mae"),
        "std_test_pair_weighted_mae": std_of("test_pair_weighted_mae"),
        "mean_test_pair_weighted_rmse": mean_of("test_pair_weighted_rmse"),
        "std_test_pair_weighted_rmse": std_of("test_pair_weighted_rmse"),
    }


def size_bin_delta_summary(per_rows, improved_variant, baseline_variant):
    """Summarize per-size-bin deltas between two variants."""
    bin_values = {}
    for row in per_rows:
        variant = row["variant"]
        if variant not in {improved_variant, baseline_variant}:
            continue
        size_bin = int(row["size_bin"])
        bin_values.setdefault(size_bin, {}).setdefault(variant, []).append(
            float(row["per_protein_r2"])
        )

    deltas = []
    for size_bin in sorted(bin_values):
        improved_values = bin_values[size_bin].get(improved_variant, [])
        baseline_values = bin_values[size_bin].get(baseline_variant, [])
        if not improved_values or not baseline_values:
            continue
        delta = float(np.mean(improved_values) - np.mean(baseline_values))
        deltas.append((size_bin, delta))

    if not deltas:
        return "no shared size bins", None

    formatted = ", ".join(f"bin{size_bin}={delta:+.4f}" for size_bin, delta in deltas)
    strongest_bin, strongest_delta = max(deltas, key=lambda item: abs(item[1]))
    strongest = f"strongest=bin{strongest_bin} ({strongest_delta:+.4f})"
    return f"{formatted}; {strongest}", (strongest_bin, strongest_delta)


def build_summary_text(
    path,
    elapsed_seconds,
    all_proteins,
    protein_data,
    skipped,
    total_cached_rows,
    summary_rows,
    per_rows,
):
    """Write a compact human-readable summary."""
    summary_by_variant = {row["variant"]: row for row in summary_rows}
    ranked_rows = sorted(
        summary_rows,
        key=lambda row: float(row["mean_test_mean_per_protein_r2"]),
        reverse=True,
    )

    def metric_delta(variant_a, variant_b):
        return (
            float(summary_by_variant[variant_a]["mean_test_mean_per_protein_r2"])
            - float(summary_by_variant[variant_b]["mean_test_mean_per_protein_r2"])
        )

    stratified_cap_by_bin, _ = size_bin_delta_summary(
        per_rows,
        "cap5k_protein_equal_stratified_kfold",
        "all_pairs_protein_equal_stratified_kfold",
    )
    random_cap_by_bin, _ = size_bin_delta_summary(
        per_rows,
        "cap5k_protein_equal_random_kfold",
        "all_pairs_protein_equal_random_kfold",
    )

    with path.open("w") as handle:
        handle.write(f"{TITLE}\n")
        handle.write("=" * len(TITLE) + "\n\n")
        handle.write(f"Output CSV dir:   {CSV_DIR.resolve()}\n")
        handle.write(f"Layer:            {LAYER}\n")
        handle.write("Representation:   lower triangle only, pb[i,j,:] only\n")
        handle.write(f"Cross-validation: protein-level {N_FOLDS}-fold\n")
        handle.write(f"Size bins:        {N_SIZE_BINS} rank-based bins for stratified folds\n")
        handle.write(
            f"Train cap:        {MAX_TRAIN_PAIRS_PER_PROTEIN:,} pairs per train protein when capped\n"
        )
        handle.write(f"Elapsed seconds:  {elapsed_seconds:.1f}\n\n")

        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(protein_data)}\n")
        handle.write(f"Proteins skipped:    {len(skipped)}\n")
        handle.write(f"Cached LT rows:      {total_cached_rows:,}\n\n")

        handle.write("Ranked by fold-mean test mean per-protein R2:\n")
        for row in ranked_rows:
            handle.write(
                f"  {row['variant']}: mean={float(row['mean_test_mean_per_protein_r2']):.4f} "
                f"+/- {float(row['std_test_mean_per_protein_r2']):.4f}, "
                f"pairR2={float(row['mean_test_pair_weighted_r2']):.4f}\n"
            )

        handle.write("\nKey ablation deltas (mean test mean per-protein R2):\n")
        handle.write(
            "  uncapped stratified - uncapped random: "
            f"{metric_delta('all_pairs_protein_equal_stratified_kfold', 'all_pairs_protein_equal_random_kfold'):+.4f}\n"
        )
        handle.write(
            "  capped stratified - capped random: "
            f"{metric_delta('cap5k_protein_equal_stratified_kfold', 'cap5k_protein_equal_random_kfold'):+.4f}\n"
        )
        handle.write(
            "  capped - uncapped within stratified: "
            f"{metric_delta('cap5k_protein_equal_stratified_kfold', 'all_pairs_protein_equal_stratified_kfold'):+.4f}\n"
        )
        handle.write(
            "  capped - uncapped within random: "
            f"{metric_delta('cap5k_protein_equal_random_kfold', 'all_pairs_protein_equal_random_kfold'):+.4f}\n"
        )
        handle.write(
            "  capped pair-weighted - capped protein-equal within random: "
            f"{metric_delta('cap5k_pair_weighted_random_kfold', 'cap5k_protein_equal_random_kfold'):+.4f}\n"
        )

        handle.write("\nCapping effect by size bin:\n")
        handle.write(f"  Stratified folds: {stratified_cap_by_bin}\n")
        handle.write(f"  Random folds:     {random_cap_by_bin}\n")


def run_synthetic_sanity_checks():
    """Quick local sanity checks for folds, capping, and weighting."""
    synthetic_records = []
    for index in range(25):
        n_pairs = 6 + (index % 7)
        synthetic_records.append(
            {
                "protein_id": f"synthetic_{index:02d}",
                "n_residues": 60 + (index * 10),
                "n_pairs": n_pairs,
                "y_abs": np.linspace(1.0, 2.0, n_pairs, dtype=np.float32),
            }
        )

    assign_rank_size_bins(synthetic_records, N_SIZE_BINS)
    stratified = build_fold_family(
        synthetic_records,
        FOLD_FAMILY_STRATIFIED,
        stratified=True,
    )
    random = build_fold_family(
        synthetic_records,
        FOLD_FAMILY_RANDOM,
        stratified=False,
    )

    def count_test_appearances(fold_family):
        counts = {}
        for fold in fold_family["folds"]:
            train_ids = {record["protein_id"] for record in fold["train_records"]}
            test_ids = {record["protein_id"] for record in fold["test_records"]}
            if train_ids & test_ids:
                raise AssertionError("Protein leakage detected between train and test")
            for record in fold["test_records"]:
                protein_id = record["protein_id"]
                counts[protein_id] = counts.get(protein_id, 0) + 1
        return counts

    stratified_counts = count_test_appearances(stratified)
    random_counts = count_test_appearances(random)

    bundle_records = [
        {
            "protein_id": "bundle_a",
            "n_residues": 100,
            "n_pairs": 9,
            "y_abs": np.linspace(1.0, 9.0, 9, dtype=np.float32),
        },
        {
            "protein_id": "bundle_b",
            "n_residues": 150,
            "n_pairs": 4,
            "y_abs": np.linspace(2.0, 5.0, 4, dtype=np.float32),
        },
    ]
    chunk_map = {
        "bundle_a": np.tile(np.arange(N_CHANNELS, dtype=np.float32), (9, 1)),
        "bundle_b": np.tile(np.arange(N_CHANNELS, dtype=np.float32) * 0.5, (4, 1)),
    }
    capped_bundle = build_lower_triangle_bundle(
        bundle_records,
        chunk_map,
        sample_cap=5,
    )
    capped_scaler = compute_standardization_from_bundle(
        capped_bundle,
        protein_equal_weighting=True,
    )

    weight_totals = []
    for start, end in capped_bundle["offsets"]:
        weight_totals.append(float(np.sum(capped_bundle["weights"][start:end])))

    return {
        "stratified_test_appearances_min": min(stratified_counts.values()),
        "stratified_test_appearances_max": max(stratified_counts.values()),
        "random_test_appearances_min": min(random_counts.values()),
        "random_test_appearances_max": max(random_counts.values()),
        "capped_max_rows_per_protein": int(np.max(capped_bundle["row_counts"])),
        "protein_equal_weight_total_min": float(min(weight_totals)),
        "protein_equal_weight_total_max": float(max(weight_totals)),
        "scale_min": float(np.min(capped_scaler["scale"])),
    }


def main():
    """Run the full k-fold ablation diagnostic."""
    t0 = time.time()
    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))
    print(f"Output CSV dir: {csv_dir}")
    print(f"Variants:       {len(VARIANTS)}")
    print(f"Train cap:      {MAX_TRAIN_PAIRS_PER_PROTEIN:,} pairs per train protein")
    print(f"Cross-validation: {N_FOLDS}-fold protein-level CV")

    all_proteins, protein_data, skipped = load_protein_records()
    if not protein_data:
        print("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    assign_rank_size_bins(protein_data, N_SIZE_BINS)
    chunk_map, total_cached_rows = preload_lower_triangle_chunks(protein_data)

    fold_families = {
        FOLD_FAMILY_STRATIFIED: build_fold_family(
            protein_data,
            FOLD_FAMILY_STRATIFIED,
            stratified=True,
        ),
        FOLD_FAMILY_RANDOM: build_fold_family(
            protein_data,
            FOLD_FAMILY_RANDOM,
            stratified=False,
        ),
    }

    fold_assignment_rows = (
        fold_families[FOLD_FAMILY_STRATIFIED]["assignment_rows"]
        + fold_families[FOLD_FAMILY_RANDOM]["assignment_rows"]
    )
    write_csv(
        csv_dir / FOLD_ASSIGNMENTS_CSV_NAME,
        FOLD_ASSIGNMENT_FIELDNAMES,
        fold_assignment_rows,
    )

    fold_summary_rows = []
    summary_rows = []
    per_protein_rows = []
    train_audit_rows = []

    for variant in VARIANTS:
        print(f"\nRunning variant: {variant['name']}")
        fold_family = fold_families[variant["fold_family"]]
        variant_fold_rows = []
        for fold_info in fold_family["folds"]:
            fold_number = int(fold_info["fold_index"]) + 1
            print(
                f"  Fold {fold_number}/{N_FOLDS} "
                f"({len(fold_info['train_records'])} train / {len(fold_info['test_records'])} test proteins)"
            )
            result = evaluate_variant_on_fold(variant, fold_info, chunk_map)
            fold_summary_rows.append(result["fold_summary_row"])
            variant_fold_rows.append(result["fold_summary_row"])
            per_protein_rows.extend(result["per_rows"])
            train_audit_rows.extend(result["audit_rows"])
            print(
                "    test mean per-protein R2="
                f"{float(result['fold_summary_row']['test_mean_per_protein_r2']):.4f}, "
                "test pair-weighted R2="
                f"{float(result['fold_summary_row']['test_pair_weighted_r2']):.4f}"
            )

        summary_row = aggregate_variant_rows(
            variant,
            variant_fold_rows,
            n_loaded_proteins=len(protein_data),
        )
        summary_rows.append(summary_row)
        print(
            "  fold-mean test mean per-protein R2="
            f"{float(summary_row['mean_test_mean_per_protein_r2']):.4f}"
        )

    write_csv(csv_dir / SUMMARY_CSV_NAME, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(csv_dir / FOLD_SUMMARY_CSV_NAME, FOLD_SUMMARY_FIELDNAMES, fold_summary_rows)
    write_csv(csv_dir / PER_PROTEIN_CSV_NAME, PER_PROTEIN_FIELDNAMES, per_protein_rows)
    write_csv(csv_dir / TRAIN_AUDIT_CSV_NAME, TRAIN_AUDIT_FIELDNAMES, train_audit_rows)

    build_summary_text(
        csv_dir / SUMMARY_TXT_NAME,
        elapsed_seconds=float(time.time() - t0),
        all_proteins=all_proteins,
        protein_data=protein_data,
        skipped=skipped,
        total_cached_rows=total_cached_rows,
        summary_rows=summary_rows,
        per_rows=per_protein_rows,
    )

    print("\nWrote outputs:")
    print(f"  {csv_dir / SUMMARY_CSV_NAME}")
    print(f"  {csv_dir / FOLD_SUMMARY_CSV_NAME}")
    print(f"  {csv_dir / PER_PROTEIN_CSV_NAME}")
    print(f"  {csv_dir / FOLD_ASSIGNMENTS_CSV_NAME}")
    print(f"  {csv_dir / TRAIN_AUDIT_CSV_NAME}")
    print(f"  {csv_dir / SUMMARY_TXT_NAME}")
    print(f"Elapsed seconds: {time.time() - t0:.1f}")


if __name__ == "__main__":
    main()

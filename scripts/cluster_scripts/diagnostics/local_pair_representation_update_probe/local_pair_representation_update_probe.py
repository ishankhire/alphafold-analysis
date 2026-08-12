#!/usr/bin/env python3
"""Fit one expanded local pair-representation Ridge update-probe shard."""

from __future__ import annotations

import gc
import math
import os
import resource
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from scipy.sparse.linalg import LinearOperator, lsqr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from local_pair_representation_update_common import (
    AA_PAIRS, BOOTSTRAP_REPLICATES, LSQR_MAX_ITER, LSQR_TOLERANCE,
    N_CHANNELS, OUTPUT_NAME, RELATIVE_RIDGE_ALPHA, SOURCE_VARIANT,
    aa_pair_indices, bootstrap_assignments, build_feature_chunk, build_offsets,
    decode_selected_distances, extract_pdb_residue_names, feature_dim,
    feature_schema, load_distance_coefficients, load_split_records, local_pair_indices,
    log, now_timestamp, pair_block_path, pdb_path, protein_equal_scaler,
    protein_equal_weights, standardize_in_place, task_spec, write_csv_atomic,
    write_json_atomic, write_text_atomic,
)


N_AA_PAIRS = len(AA_PAIRS)
EXPECTED_COUNTS = (441, 111, 812097, 207900)


def finite_mean(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def safe_pearson(y, prediction):
    y, prediction = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    if y.size < 2 or np.std(y) <= 0.0 or np.std(prediction) <= 0.0:
        return float("nan")
    return float(np.corrcoef(y, prediction)[0, 1])


def regression_metrics(y, prediction):
    y, prediction = np.asarray(y, np.float64), np.asarray(prediction, np.float64)
    if y.size == 0:
        return {key: float("nan") for key in (
            "r2", "mae", "rmse", "pearson_r", "bias", "sign_accuracy",
        )}
    nonzero = y != 0.0
    return {
        "r2": float(r2_score(y, prediction)) if y.size > 1 and np.var(y) > 0 else float("nan"),
        "mae": float(mean_absolute_error(y, prediction)),
        "rmse": float(np.sqrt(mean_squared_error(y, prediction))),
        "pearson_r": safe_pearson(y, prediction),
        "bias": float(np.mean(prediction - y)),
        "sign_accuracy": (
            float(np.mean(np.sign(prediction[nonzero]) == np.sign(y[nonzero])))
            if np.any(nonzero) else float("nan")
        ),
    }


def max_rss_mb():
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    if sys.platform == "darwin":
        return value / (1024.0 * 1024.0)
    return value / 1024.0


def fit_weighted_ridge_lsqr(x, y, weights, relative_alpha=RELATIVE_RIDGE_ALPHA,
                            tolerance=LSQR_TOLERANCE, max_iter=LSQR_MAX_ITER):
    """Solve weighted Ridge without materializing sqrt(W)X or X'X."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    sqrt_weights = np.sqrt(weights)
    weight_sum = float(np.sum(weights))
    intercept = float(np.dot(weights, y) / weight_sum)
    rhs = sqrt_weights * (y - intercept)
    operator = LinearOperator(
        x.shape,
        matvec=lambda vector: sqrt_weights * (x @ vector),
        rmatvec=lambda vector: x.T @ (sqrt_weights * vector),
        dtype=np.float64,
    )
    alpha = float(relative_alpha * weight_sum)
    result = lsqr(
        operator, rhs, damp=math.sqrt(alpha), atol=tolerance, btol=tolerance,
        iter_lim=max_iter, show=False,
    )
    coefficient = np.asarray(result[0], dtype=np.float64)
    return {
        "coefficient": coefficient,
        "intercept": intercept,
        "relative_alpha": relative_alpha,
        "actual_alpha": alpha,
        "istop": int(result[1]),
        "n_iter": int(result[2]),
        "residual_norm": float(result[3]),
        "regularized_residual_norm": float(result[4]),
        "operator_norm": float(result[5]),
        "condition_estimate": float(result[6]),
        "normal_equation_residual_norm": float(result[7]),
        "coefficient_norm": float(result[8]),
        "iteration_cap_reached": int(result[2]) >= int(max_iter),
        "converged": int(result[1]) in {1, 2, 4, 5},
    }


def evaluate_model(layer_from, layer_to, variant, targets, predictions, offsets, fit):
    summary_rows, per_rows = [], []
    for split in ("train", "test"):
        y, prediction = targets[split], predictions[split]
        per_metrics = []
        for record, start, end in offsets[split]:
            metrics = regression_metrics(y[start:end], prediction[start:end])
            per_metrics.append(metrics)
            per_rows.append({
                "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                "feature_dim": feature_dim(variant), "split": split,
                "source": record["source"], "protein_id": record["protein_id"],
                "n_residues": record["n_residues"], "n_pairs": end - start,
                **metrics,
            })
        pooled = regression_metrics(y, prediction)
        row = {
            "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
            "feature_dim": feature_dim(variant), "split": split,
            "n_proteins": len(offsets[split]), "n_pairs": y.size,
            **{f"pair_{key}": value for key, value in pooled.items()},
        }
        for key in pooled:
            values = [metrics[key] for metrics in per_metrics]
            row[f"mean_per_protein_{key}"] = finite_mean(values)
            row[f"n_valid_proteins_{key}"] = sum(math.isfinite(value) for value in values)
        row.update({
            "relative_alpha": fit["relative_alpha"], "actual_alpha": fit["actual_alpha"],
            "solver": "scipy_lsqr_linear_operator", "tolerance": LSQR_TOLERANCE,
            "max_iter": LSQR_MAX_ITER, "n_iter": fit["n_iter"], "istop": fit["istop"],
            "converged": fit["converged"],
            "iteration_cap_reached": fit["iteration_cap_reached"],
            "condition_estimate": fit["condition_estimate"],
            "fit_elapsed_seconds": fit.get("fit_elapsed_seconds", float("nan")),
        })
        summary_rows.append(row)
    return summary_rows, per_rows


def magnitude_metrics(layer_from, layer_to, variant, split, target, prediction, offsets):
    signed = (
        ("lt_neg4", target < -4.0, "signed"),
        ("neg4_neg2", (target >= -4.0) & (target < -2.0), "signed"),
        ("neg2_neg1", (target >= -2.0) & (target < -1.0), "signed"),
        ("neg1_neg0_5", (target >= -1.0) & (target < -0.5), "signed"),
        ("neg0_5_neg0_25", (target >= -0.5) & (target < -0.25), "signed"),
        ("neg0_25_0", (target >= -0.25) & (target < 0.0), "signed"),
        ("zero_0_25", (target > 0.0) & (target <= 0.25), "signed"),
        ("0_25_0_5", (target > 0.25) & (target <= 0.5), "signed"),
        ("0_5_1", (target > 0.5) & (target <= 1.0), "signed"),
        ("1_2", (target > 1.0) & (target <= 2.0), "signed"),
        ("2_4", (target > 2.0) & (target <= 4.0), "signed"),
        ("gt4", target > 4.0, "signed"),
        ("exact_zero", target == 0.0, "audit"),
    )
    masks = list(signed)
    masks.extend(
        (f"abs_gt_{str(value).replace('.', '_')}", np.abs(target) > value, "cumulative")
        for value in (0.5, 1.0, 2.0, 4.0)
    )
    protein_ids = np.empty(target.size, dtype=object)
    for record, start, end in offsets:
        protein_ids[start:end] = record["protein_id"]
    rows = []
    for name, mask, bin_type in masks:
        row = {
            "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
            "split": split, "bin_name": name, "bin_type": bin_type,
            "n_pairs": int(np.count_nonzero(mask)),
            "n_proteins": len(set(protein_ids[mask])) if np.any(mask) else 0,
        }
        row.update(regression_metrics(target[mask], prediction[mask]))
        rows.append(row)
    return rows


def target_statistics(layer_from, layer_to, split, target, categories, offsets):
    protein_ids = np.empty(target.size, dtype=object)
    for record, start, end in offsets:
        protein_ids[start:end] = record["protein_id"]
    overall_mean = float(np.mean(target))
    rows = []
    for category, (aa_first, aa_second) in enumerate(AA_PAIRS):
        mask = categories == category
        values = target[mask]
        row = {
            "layer_from": layer_from, "layer_to": layer_to, "split": split,
            "aa_pair_index": category, "aa_first": aa_first, "aa_second": aa_second,
            "aa_pair": f"{aa_first}-{aa_second}", "n_pairs": values.size,
            "n_proteins": len(set(protein_ids[mask])) if values.size else 0,
            "overall_mean_delta_d": overall_mean,
        }
        if values.size:
            quantiles = np.quantile(values, (0.05, 0.25, 0.5, 0.75, 0.95))
            row.update({
                "mean_delta_d": np.mean(values), "std_delta_d": np.std(values, ddof=1) if values.size > 1 else float("nan"),
                "mean_absolute_delta_d": np.mean(np.abs(values)), "minimum_delta_d": np.min(values),
                "q05_delta_d": quantiles[0], "q25_delta_d": quantiles[1],
                "median_delta_d": quantiles[2], "q75_delta_d": quantiles[3],
                "q95_delta_d": quantiles[4], "maximum_delta_d": np.max(values),
                "fraction_shrink": np.mean(values < 0.0), "fraction_expand": np.mean(values > 0.0),
                "fraction_within_0_5": np.mean(np.abs(values) <= 0.5),
                "mean_delta_d_minus_all_pairs": np.mean(values) - overall_mean,
            })
        rows.append(row)
    return rows


def aa_regression_metrics(layer_from, layer_to, variant, split, target, prediction,
                          categories, offsets):
    protein_ids = np.empty(target.size, dtype=object)
    for record, start, end in offsets:
        protein_ids[start:end] = record["protein_id"]
    rows = []
    for category, (aa_first, aa_second) in enumerate(AA_PAIRS):
        mask = categories == category
        rows.append({
            "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
            "split": split, "aa_pair_index": category, "aa_first": aa_first,
            "aa_second": aa_second, "aa_pair": f"{aa_first}-{aa_second}",
            "n_pairs": int(np.count_nonzero(mask)),
            "n_proteins": len(set(protein_ids[mask])) if np.any(mask) else 0,
            **regression_metrics(target[mask], prediction[mask]),
        })
    return rows


def sufficient_statistics(target, prediction, categories, offsets):
    n_proteins = len(offsets)
    shape = (n_proteins, N_AA_PAIRS)
    n = np.zeros(shape, np.int64)
    sum_y = np.zeros(shape, np.float64)
    sum_y2 = np.zeros(shape, np.float64)
    sse = np.zeros(shape, np.float64)
    protein_r2 = np.full(n_proteins, np.nan, np.float64)
    overall = np.zeros((n_proteins, 4), np.float64)
    for protein_index, (_record, start, end) in enumerate(offsets):
        y, pred, category = target[start:end], prediction[start:end], categories[start:end]
        residual2 = (pred - y) ** 2
        overall[protein_index] = (y.size, np.sum(y), np.dot(y, y), np.sum(residual2))
        protein_r2[protein_index] = regression_metrics(y, pred)["r2"]
        valid = category >= 0
        if not np.any(valid):
            continue
        cat = category[valid]
        n[protein_index] = np.bincount(cat, minlength=N_AA_PAIRS)
        sum_y[protein_index] = np.bincount(cat, weights=y[valid], minlength=N_AA_PAIRS)
        sum_y2[protein_index] = np.bincount(cat, weights=y[valid] ** 2, minlength=N_AA_PAIRS)
        sse[protein_index] = np.bincount(cat, weights=residual2[valid], minlength=N_AA_PAIRS)
    return n, sum_y, sum_y2, sse, overall, protein_r2


def r2_from_sums(n, sum_y, sum_y2, sse):
    variance = sum_y2 - np.divide(sum_y * sum_y, n, out=np.zeros_like(sum_y), where=n > 0)
    return np.divide(-sse, variance, out=np.full_like(sum_y, np.nan), where=(n > 1) & (variance > 0)) + 1.0


def bootstrap_outputs(layer_from, layer_to, variant, target, prediction, categories,
                      offsets, bootstrap_counts):
    n, sum_y, sum_y2, sse, overall, protein_r2 = sufficient_statistics(
        target, prediction, categories, offsets
    )
    category_r2 = np.full((BOOTSTRAP_REPLICATES, N_AA_PAIRS), np.nan)
    category_mean = np.full_like(category_r2, np.nan)
    category_rank = np.full_like(category_r2, np.nan)
    overall_r2 = np.full(BOOTSTRAP_REPLICATES, np.nan)
    overall_mean = np.full(BOOTSTRAP_REPLICATES, np.nan)
    mean_protein_r2 = np.full(BOOTSTRAP_REPLICATES, np.nan)
    bootstrap_rows = []
    for replicate, counts in enumerate(bootstrap_counts):
        cat_n = counts @ n
        cat_sum = counts @ sum_y
        cat_sum2 = counts @ sum_y2
        cat_sse = counts @ sse
        category_r2[replicate] = r2_from_sums(cat_n, cat_sum, cat_sum2, cat_sse)
        category_mean[replicate] = np.divide(
            cat_sum, cat_n, out=np.full(N_AA_PAIRS, np.nan), where=cat_n > 0
        )
        totals = counts @ overall
        overall_r2[replicate] = r2_from_sums(*totals)[()]
        overall_mean[replicate] = totals[1] / totals[0]
        valid_protein = (counts > 0) & np.isfinite(protein_r2)
        if np.any(valid_protein):
            mean_protein_r2[replicate] = np.average(
                protein_r2[valid_protein], weights=counts[valid_protein]
            )
        valid_categories = np.flatnonzero(np.isfinite(category_r2[replicate]))
        if valid_categories.size:
            order = valid_categories[np.argsort(-category_r2[replicate, valid_categories])]
            category_rank[replicate, order] = np.arange(1, order.size + 1)
        for metric, value in (
            ("pair_r2", overall_r2[replicate]),
            ("mean_per_protein_r2", mean_protein_r2[replicate]),
        ):
            bootstrap_rows.append({
                "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                "bootstrap_replicate": replicate, "metric": metric, "value": value,
            })

    point_overall = regression_metrics(target, prediction)["r2"]
    aa_summary, target_uncertainty = [], []
    for category, (aa_first, aa_second) in enumerate(AA_PAIRS):
        valid_r2 = np.isfinite(category_r2[:, category])
        r2_values = category_r2[valid_r2, category]
        deltas = r2_values - overall_r2[valid_r2]
        ranks = category_rank[np.isfinite(category_rank[:, category]), category]
        top_quartile = []
        for replicate in range(BOOTSTRAP_REPLICATES):
            if not np.isfinite(category_rank[replicate, category]):
                continue
            valid_count = int(np.count_nonzero(np.isfinite(category_rank[replicate])))
            top_quartile.append(category_rank[replicate, category] <= math.ceil(valid_count / 4))
        point_mask = categories == category
        point_r2 = regression_metrics(target[point_mask], prediction[point_mask])["r2"]
        row = {
            "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
            "split": "test", "aa_pair_index": category, "aa_first": aa_first,
            "aa_second": aa_second, "aa_pair": f"{aa_first}-{aa_second}",
            "n_pairs": int(np.count_nonzero(point_mask)),
            "point_r2": point_r2, "point_overall_r2": point_overall,
            "point_r2_minus_overall": point_r2 - point_overall,
            "n_valid_bootstrap_replicates": r2_values.size,
        }
        if r2_values.size:
            row.update({
                "bootstrap_mean_r2": np.mean(r2_values),
                "r2_ci_lower_95": np.quantile(r2_values, 0.025),
                "r2_ci_upper_95": np.quantile(r2_values, 0.975),
                "bootstrap_mean_r2_minus_overall": np.mean(deltas),
                "r2_minus_overall_ci_lower_95": np.quantile(deltas, 0.025),
                "r2_minus_overall_ci_upper_95": np.quantile(deltas, 0.975),
                "median_rank": np.median(ranks), "rank_ci_lower_95": np.quantile(ranks, 0.025),
                "rank_ci_upper_95": np.quantile(ranks, 0.975),
                "probability_top_quartile": np.mean(top_quartile),
            })
        aa_summary.append(row)

        valid_mean = np.isfinite(category_mean[:, category])
        means = category_mean[valid_mean, category]
        contrasts = means - overall_mean[valid_mean]
        target_row = {
            "layer_from": layer_from, "layer_to": layer_to, "split": "test",
            "aa_pair_index": category, "aa_first": aa_first, "aa_second": aa_second,
            "aa_pair": f"{aa_first}-{aa_second}",
            "n_valid_bootstrap_replicates": means.size,
        }
        if means.size:
            target_row.update({
                "bootstrap_mean_delta_d": np.mean(means),
                "mean_delta_d_ci_lower_95": np.quantile(means, 0.025),
                "mean_delta_d_ci_upper_95": np.quantile(means, 0.975),
                "bootstrap_mean_delta_d_minus_all_pairs": np.mean(contrasts),
                "contrast_ci_lower_95": np.quantile(contrasts, 0.025),
                "contrast_ci_upper_95": np.quantile(contrasts, 0.975),
            })
        target_uncertainty.append(target_row)
    return bootstrap_rows, aa_summary, target_uncertainty


def change_statistics(layer_from, layer_to, split, values, n_proteins):
    quantiles = np.quantile(values, (0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99))
    return {
        "layer_from": layer_from, "layer_to": layer_to, "split": split,
        "n_proteins": n_proteins, "n_pairs": values.size,
        "mean": np.mean(values), "std": np.std(values, ddof=1), "minimum": np.min(values),
        "q01": quantiles[0], "q05": quantiles[1], "q25": quantiles[2],
        "median": quantiles[3], "q75": quantiles[4], "q95": quantiles[5],
        "q99": quantiles[6], "maximum": np.max(values),
        "mean_absolute": np.mean(np.abs(values)), "rms": np.sqrt(np.mean(values ** 2)),
        "n_shrink": np.count_nonzero(values < -0.5),
        "n_within_0_5": np.count_nonzero(np.abs(values) <= 0.5),
        "n_expand": np.count_nonzero(values > 0.5), "threshold_angstrom": 0.5,
    }


def main():
    total_started = time.time()
    group = os.environ.get("VARIANT_GROUP", "all")
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    layer_from, layer_to, variant = task_spec(task_id, group)
    csv_root = Path(os.environ.get("CSV_ROOT", Path.home() / "csv_files")).expanduser()
    source_root = csv_root / SOURCE_VARIANT
    output_root = Path(
        os.environ.get("LOCAL_PAIR_REP_UPDATE_OUTPUT_ROOT", csv_root / OUTPUT_NAME)
    ).expanduser()
    output_dir = output_root / "shards" / f"transition_{layer_from:02d}_{layer_to:02d}" / variant
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Starting task={task_id}, group={group}, transition={layer_from}->{layer_to}, variant={variant}")

    assignment_path = source_root / "split_assignments.csv"
    coefficient_path = source_root / "coefficients.csv"
    records, split_rows = load_split_records(assignment_path)
    intercept_from, coefficients_from, coefficient_rows_from, hash_from = load_distance_coefficients(
        coefficient_path, layer_from
    )
    intercept_to, coefficients_to, coefficient_rows_to, hash_to = load_distance_coefficients(
        coefficient_path, layer_to
    )
    offsets, pair_counts = {}, {}
    for split in ("train", "test"):
        offsets[split], pair_counts[split] = build_offsets(records, split)
    actual_counts = (
        len(offsets["train"]), len(offsets["test"]), pair_counts["train"], pair_counts["test"],
    )
    if actual_counts != EXPECTED_COUNTS:
        raise ValueError(f"Frozen split counts {actual_counts} do not match {EXPECTED_COUNTS}")

    x = {split: np.empty((pair_counts[split], feature_dim(variant)), np.float64) for split in ("train", "test")}
    target = {split: np.empty(pair_counts[split], np.float64) for split in ("train", "test")}
    categories = {split: np.empty(pair_counts[split], np.int16) for split in ("train", "test")}
    positions = {"train": 0, "test": 0}
    manifest_rows, skipped_rows, timing_rows = [], [], []
    unknown_residues, total_residues, unknown_proteins = 0, 0, set()
    feature_started = time.time()
    try:
        for record_index, record in enumerate(records, start=1):
            protein_started = time.time()
            protein_id, split = record["protein_id"], record["split"]
            from_path, to_path, structure_path = (
                pair_block_path(protein_id, layer_from), pair_block_path(protein_id, layer_to),
                pdb_path(protein_id),
            )
            try:
                for path in (from_path, to_path, structure_path):
                    if not path.exists():
                        raise FileNotFoundError(path)
                residue_names = extract_pdb_residue_names(structure_path)
                if len(residue_names) != record["n_residues"]:
                    raise ValueError(
                        f"PDB CA count {len(residue_names)} != assigned {record['n_residues']}"
                    )
                protein_unknown = sum(value is None for value in residue_names)
                total_residues += len(residue_names)
                unknown_residues += protein_unknown
                if protein_unknown:
                    unknown_proteins.add(protein_id)
                pb_from, pb_to = np.load(from_path), np.load(to_path)
                expected_shape = (record["n_residues"], record["n_residues"], N_CHANNELS)
                if pb_from.shape != expected_shape or pb_to.shape != expected_shape:
                    raise ValueError(f"Pair-block shapes {pb_from.shape}/{pb_to.shape} != {expected_shape}")
                pair_i, pair_j, _separation = local_pair_indices(record["n_residues"])
                distance_from = decode_selected_distances(
                    pb_from, pair_i, pair_j, intercept_from, coefficients_from
                )
                distance_to = decode_selected_distances(
                    pb_to, pair_i, pair_j, intercept_to, coefficients_to
                )
                x_chunk = build_feature_chunk(pb_from, pair_i, pair_j, variant)
                category_chunk = aa_pair_indices(residue_names, pair_i, pair_j)
                start, end = positions[split], positions[split] + pair_i.size
                x[split][start:end] = x_chunk
                target[split][start:end] = distance_to - distance_from
                categories[split][start:end] = category_chunk
                positions[split] = end
                manifest_rows.append({
                    "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                    "split": split, "source": record["source"], "protein_id": protein_id,
                    "n_residues": record["n_residues"], "n_local_pairs": pair_i.size,
                    "n_unknown_residues": protein_unknown,
                    "n_unknown_aa_pairs": int(np.count_nonzero(category_chunk < 0)),
                    "pair_block_from": str(from_path), "pair_block_to": str(to_path),
                    "pdb_path": str(structure_path), "status": "success",
                    "elapsed_seconds": time.time() - protein_started,
                })
                del pb_from, pb_to, pair_i, pair_j, distance_from, distance_to, x_chunk, category_chunk
                gc.collect()
            except Exception as exc:
                skipped_rows.append({
                    "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                    "split": split, "source": record["source"], "protein_id": protein_id,
                    "reason": type(exc).__name__, "message": str(exc),
                })
                raise
            if record_index % 10 == 0 or record_index == len(records):
                log(
                    f"Feature build {record_index}/{len(records)} proteins; "
                    f"elapsed={time.time() - feature_started:.1f}s; max_rss={max_rss_mb():.0f} MB"
                )
    finally:
        write_csv_atomic(output_dir / "protein_manifest.csv", manifest_rows)
        write_csv_atomic(output_dir / "skipped_proteins.csv", skipped_rows, (
            "layer_from", "layer_to", "variant", "split", "source", "protein_id", "reason", "message",
        ))
    if positions != pair_counts:
        raise AssertionError(f"Filled {positions}, expected {pair_counts}")
    if unknown_residues / max(total_residues, 1) > 0.001 or len(unknown_proteins) > 5:
        raise ValueError(
            f"Unknown residues exceed policy: {unknown_residues}/{total_residues} "
            f"across {len(unknown_proteins)} proteins"
        )
    timing_rows.append({
        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
        "stage": "feature_build", "elapsed_seconds": time.time() - feature_started,
        "max_rss_mb": max_rss_mb(),
    })

    write_csv_atomic(output_dir / "feature_schema.csv", feature_schema(variant))
    write_csv_atomic(output_dir / "coefficients_used.csv", coefficient_rows_from + coefficient_rows_to)
    write_csv_atomic(output_dir / "split_assignments.csv", split_rows)
    change_rows = [
        change_statistics(layer_from, layer_to, split, target[split], len(offsets[split]))
        for split in ("train", "test")
    ]
    write_csv_atomic(output_dir / "change_statistics.csv", change_rows)
    target_rows = []
    for split in ("train", "test"):
        target_rows.extend(target_statistics(
            layer_from, layer_to, split, target[split], categories[split], offsets[split]
        ))
    write_csv_atomic(output_dir / "aa_pair_target_statistics.csv", target_rows)

    bootstrap_counts, assignment_rows = bootstrap_assignments(offsets["test"], layer_from, layer_to)
    write_csv_atomic(output_dir / "bootstrap_assignments.csv", assignment_rows)

    scale_started = time.time()
    mean, scale = protein_equal_scaler(x["train"], offsets["train"])
    transform_rows = []
    for index, schema in enumerate(feature_schema(variant)):
        transform_rows.append({
            "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
            "feature_index": index, "feature_name": schema["feature_name"],
            "training_mean": mean[index], "training_scale": scale[index],
        })
    write_csv_atomic(output_dir / "feature_transforms.csv", transform_rows)
    for split in ("train", "test"):
        standardize_in_place(x[split], mean, scale)
    timing_rows.append({
        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
        "stage": "standardization", "elapsed_seconds": time.time() - scale_started,
        "max_rss_mb": max_rss_mb(),
    })

    metadata = {
        "status": "running_regression", "started_at": now_timestamp(),
        "task_id": task_id, "variant_group": group,
        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
        "feature_dim": feature_dim(variant), "feature_dtype": "float64",
        "n_train_proteins": len(offsets["train"]), "n_test_proteins": len(offsets["test"]),
        "n_train_pairs": pair_counts["train"], "n_test_pairs": pair_counts["test"],
        "sequence_separation": [4, 12], "distance_type": "pre_mds_raw",
        "distance_probe_policy": "saved_pooled_own_layer",
        "distance_probe_layers": [layer_from, layer_to],
        "distance_probe_sha256": {str(layer_from): hash_from, str(layer_to): hash_to},
        "ridge_relative_alpha": RELATIVE_RIDGE_ALPHA, "lsqr_tolerance": LSQR_TOLERANCE,
        "lsqr_max_iter": LSQR_MAX_ITER, "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "endpoint_collision_policy": "zero_edges_and_zero_validity",
        "unknown_residue_count": unknown_residues,
        "unknown_residue_protein_count": len(unknown_proteins),
    }
    write_json_atomic(output_dir / "run_metadata.json", metadata)

    fit_started = time.time()
    weights = protein_equal_weights(offsets["train"])
    fit = fit_weighted_ridge_lsqr(x["train"], target["train"], weights)
    fit["fit_elapsed_seconds"] = time.time() - fit_started
    timing_rows.append({
        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
        "stage": "ridge_fit", "elapsed_seconds": fit["fit_elapsed_seconds"],
        "max_rss_mb": max_rss_mb(),
    })
    log(
        f"LSQR complete: istop={fit['istop']}, iterations={fit['n_iter']}, "
        f"converged={fit['converged']}, elapsed={time.time() - fit_started:.1f}s"
    )

    evaluation_started = time.time()
    predictions = {
        split: fit["intercept"] + x[split] @ fit["coefficient"]
        for split in ("train", "test")
    }
    model_rows, per_rows = evaluate_model(
        layer_from, layer_to, variant, target, predictions, offsets, fit
    )
    aa_metric_rows, magnitude_rows = [], []
    for split in ("train", "test"):
        aa_metric_rows.extend(aa_regression_metrics(
            layer_from, layer_to, variant, split, target[split], predictions[split],
            categories[split], offsets[split],
        ))
        magnitude_rows.extend(magnitude_metrics(
            layer_from, layer_to, variant, split, target[split], predictions[split], offsets[split]
        ))
    bootstrap_rows, aa_bootstrap_rows, target_uncertainty_rows = bootstrap_outputs(
        layer_from, layer_to, variant, target["test"], predictions["test"],
        categories["test"], offsets["test"], bootstrap_counts,
    )
    raw_coefficient = fit["coefficient"] / scale
    raw_intercept = fit["intercept"] - float(np.dot(raw_coefficient, mean))
    coefficient_rows = [{
        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
        "feature_dim": feature_dim(variant), "feature_index": -1,
        "feature_name": "intercept", "coefficient_standardized": fit["intercept"],
        "coefficient_raw_scale": raw_intercept,
    }]
    for index, schema in enumerate(feature_schema(variant)):
        coefficient_rows.append({
            "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
            "feature_dim": feature_dim(variant), "feature_index": index,
            "feature_name": schema["feature_name"],
            "coefficient_standardized": fit["coefficient"][index],
            "coefficient_raw_scale": raw_coefficient[index],
        })
    timing_rows.append({
        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
        "stage": "evaluation_and_bootstrap", "elapsed_seconds": time.time() - evaluation_started,
        "max_rss_mb": max_rss_mb(),
    })
    timing_rows.append({
        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
        "stage": "total", "elapsed_seconds": time.time() - total_started,
        "max_rss_mb": max_rss_mb(),
    })

    write_csv_atomic(output_dir / "model_summary.csv", model_rows)
    write_csv_atomic(output_dir / "per_protein_metrics.csv", per_rows)
    write_csv_atomic(output_dir / "regression_magnitude_metrics.csv", magnitude_rows)
    write_csv_atomic(output_dir / "aa_pair_regression_metrics.csv", aa_metric_rows)
    write_csv_atomic(output_dir / "aa_pair_bootstrap_summary.csv", aa_bootstrap_rows)
    write_csv_atomic(output_dir / "aa_pair_target_uncertainty.csv", target_uncertainty_rows)
    write_csv_atomic(output_dir / "bootstrap_metrics.csv", bootstrap_rows)
    write_csv_atomic(output_dir / "coefficients.csv", coefficient_rows)
    write_csv_atomic(output_dir / "timing.csv", timing_rows)
    metadata.update({
        "status": "complete", "completed_at": now_timestamp(),
        "elapsed_seconds": time.time() - total_started, "max_rss_mb": max_rss_mb(),
        "lsqr": {key: value for key, value in fit.items() if key != "coefficient"},
    })
    write_json_atomic(output_dir / "run_metadata.json", metadata)
    test_row = next(row for row in model_rows if row["split"] == "test")
    write_text_atomic(output_dir / "summary.txt", (
        "Local pair-representation distance-update Ridge probe\n"
        f"Transition: {layer_from}->{layer_to}\n"
        f"Variant: {variant} ({feature_dim(variant)} features)\n"
        f"Held-out pair R2: {test_row['pair_r2']:.6f}\n"
        f"Held-out mean per-protein R2: {test_row['mean_per_protein_r2']:.6f}\n"
        f"LSQR iterations: {fit['n_iter']}/{LSQR_MAX_ITER}; istop={fit['istop']}\n"
        f"Elapsed seconds: {time.time() - total_started:.1f}\n"
    ))
    log(f"Completed {layer_from}->{layer_to}/{variant} in {time.time() - total_started:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

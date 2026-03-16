#!/usr/bin/env python3
"""
diagnostic_protein_fairness_regression.py
=========================================
Layer-47 lower-triangle diagnostic for protein-fair linear objectives.

This script compares three closely related lower-triangle linear probes:
1. baseline_protein_equal_mse_linear
2. protein_equal_nmse_linear
3. protein_fairness_reweighted_nmse_linear

All variants:
- use only lower-triangle unordered residue pairs
- use only the center pair representation pb[i, j, :] at layer 47
- fit one shared linear model across proteins
- keep the predictor linear in the input features

The difference is only in the training objective:
- baseline_protein_equal_mse_linear minimizes mean per-protein MSE
- protein_equal_nmse_linear minimizes mean per-protein variance-normalized MSE
- protein_fairness_reweighted_nmse_linear repeatedly refits a linear model while
  upweighting train proteins with higher normalized train loss

Outputs (inside CSV_ROOT/):
    diagnostic_protein_fairness_variant_summary.csv
    diagnostic_protein_fairness_per_protein.csv
    diagnostic_protein_fairness_split.csv
    diagnostic_protein_fairness_round_summary.csv
    diagnostic_protein_fairness_round_history.csv
    diagnostic_protein_fairness_coefficients.csv
    diagnostic_protein_fairness_summary.txt
"""

import math
import os
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(CLUSTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_SCRIPTS_DIR))

from cluster_regression_helpers import get_pair_block_path  # noqa: E402
from diagnostic_tree_regression_common import (  # noqa: E402
    CSV_DIR,
    FLOAT_EPS,
    LAYER,
    N_CHANNELS,
    RANDOM_STATE,
    TEST_SIZE,
    compute_r2,
    load_protein_records,
    write_csv,
)


TITLE = "Diagnostic Protein Fairness Regression - Layer 47"
N_SIZE_BINS = 5
BLOCK_PAIRS = 100_000
RIDGE_ALPHA = 1e-8
FAIRNESS_MAX_ROUNDS = 6
FAIRNESS_POWER = 0.5
FAIRNESS_MIN_MULTIPLIER = 0.25
FAIRNESS_MAX_MULTIPLIER = 4.0
FAIRNESS_CONVERGENCE_TOL = 0.02

SUMMARY_CSV_NAME = "diagnostic_protein_fairness_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_protein_fairness_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_protein_fairness_split.csv"
ROUND_SUMMARY_CSV_NAME = "diagnostic_protein_fairness_round_summary.csv"
ROUND_HISTORY_CSV_NAME = "diagnostic_protein_fairness_round_history.csv"
COEFFICIENT_CSV_NAME = "diagnostic_protein_fairness_coefficients.csv"
SUMMARY_TXT_NAME = "diagnostic_protein_fairness_summary.txt"

FEATURE_NAMES = [f"pair_ch_{idx}" for idx in range(N_CHANNELS)]

VARIANTS = [
    {
        "name": "baseline_protein_equal_mse_linear",
        "objective_mode": "protein_equal_mse",
        "uses_iterative_fairness": False,
    },
    {
        "name": "protein_equal_nmse_linear",
        "objective_mode": "protein_equal_nmse",
        "uses_iterative_fairness": False,
    },
    {
        "name": "protein_fairness_reweighted_nmse_linear",
        "objective_mode": "protein_equal_nmse",
        "uses_iterative_fairness": True,
    },
]

SUMMARY_FIELDNAMES = [
    "variant",
    "objective_mode",
    "uses_iterative_fairness",
    "n_train_proteins",
    "n_test_proteins",
    "n_train_rows",
    "n_test_rows",
    "train_pair_weighted_r2",
    "test_pair_weighted_r2",
    "train_protein_weighted_r2",
    "test_protein_weighted_r2",
    "train_pair_weighted_mae",
    "test_pair_weighted_mae",
    "train_pair_weighted_rmse",
    "test_pair_weighted_rmse",
    "train_mean_per_protein_r2",
    "test_mean_per_protein_r2",
    "test_median_per_protein_r2",
    "test_negative_r2_fraction",
    "train_mean_per_protein_nmse",
    "test_mean_per_protein_nmse",
    "test_median_per_protein_nmse",
    "test_max_per_protein_nmse",
    "fairness_rounds_completed",
    "final_train_objective_mean",
    "final_train_objective_std",
    "final_train_objective_max",
    "elapsed_seconds",
]

PER_PROTEIN_FIELDNAMES = [
    "variant",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
    "per_protein_r2",
    "per_protein_mse",
    "per_protein_nmse",
    "per_protein_mae",
    "per_protein_rmse",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
]

SPLIT_FIELDNAMES = [
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
]

ROUND_SUMMARY_FIELDNAMES = [
    "variant",
    "round_index",
    "objective_mode",
    "mean_train_objective",
    "std_train_objective",
    "median_train_objective",
    "max_train_objective",
    "min_multiplier",
    "max_multiplier",
    "max_log_multiplier_change",
    "converged_after_round",
]

ROUND_HISTORY_FIELDNAMES = [
    "variant",
    "round_index",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
    "train_objective_loss",
    "train_r2",
    "fairness_multiplier",
    "effective_protein_mass",
]

COEFFICIENT_FIELDNAMES = [
    "variant",
    "feature_index",
    "feature_name",
    "coefficient_raw_scale",
]


def assign_size_bins(records, n_bins):
    """Assign stable rank-based size bins across all loaded proteins."""
    ordered = sorted(records, key=lambda row: (row["n_residues"], row["protein_id"]))
    total = len(ordered)
    for rank, record in enumerate(ordered):
        size_bin = int((rank * n_bins) // max(total, 1))
        if size_bin >= n_bins:
            size_bin = n_bins - 1
        record["size_bin"] = int(size_bin)


def split_global_records(records):
    """Create a single deterministic protein-level train/test split."""
    if len(records) < 2:
        raise ValueError("Need at least two proteins to split into train/test.")

    indices = np.arange(len(records), dtype=np.int64)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_set = set(int(idx) for idx in train_idx)
    test_set = set(int(idx) for idx in test_idx)
    train_records = [record for idx, record in enumerate(records) if idx in train_set]
    test_records = [record for idx, record in enumerate(records) if idx in test_set]
    return train_records, test_records


def prepare_records(records):
    """Attach layer-47 lower-triangle metadata used by the diagnostic."""
    prepared = []
    for record in records:
        target_variance = float(np.var(record["y_abs"], dtype=np.float64))
        if target_variance <= FLOAT_EPS:
            target_variance = FLOAT_EPS

        prepared_record = dict(record)
        prepared_record["pb_path"] = get_pair_block_path(
            record["protein_dir"],
            record["protein_id"],
            LAYER,
        )
        prepared_record["target_variance"] = target_variance
        prepared_record["target_mean"] = float(np.mean(record["y_abs"], dtype=np.float64))
        prepared.append(prepared_record)
    return prepared


def iter_record_blocks(record):
    """Yield lower-triangle feature/target blocks for one protein."""
    pb = np.load(str(record["pb_path"]), mmap_mode="r")
    lt_i = record["lt_i"]
    lt_j = record["lt_j"]
    y_abs = record["y_abs"]
    n_pairs = int(record["n_pairs"])

    for start in range(0, n_pairs, BLOCK_PAIRS):
        end = min(start + BLOCK_PAIRS, n_pairs)
        x_block = np.asarray(pb[lt_i[start:end], lt_j[start:end], :], dtype=np.float64)
        y_block = np.asarray(y_abs[start:end], dtype=np.float64)
        yield x_block, y_block

    del pb


def compute_feature_standardization(train_records):
    """Compute protein-equal feature mean/std from train proteins only."""
    mean_acc = np.zeros((N_CHANNELS,), dtype=np.float64)
    sq_acc = np.zeros((N_CHANNELS,), dtype=np.float64)

    for record in train_records:
        feat_sum = np.zeros((N_CHANNELS,), dtype=np.float64)
        feat_sq_sum = np.zeros((N_CHANNELS,), dtype=np.float64)
        total_rows = 0
        for x_block, _ in iter_record_blocks(record):
            feat_sum += np.sum(x_block, axis=0, dtype=np.float64)
            feat_sq_sum += np.sum(x_block * x_block, axis=0, dtype=np.float64)
            total_rows += int(x_block.shape[0])

        if total_rows <= 0:
            raise ValueError(f"Protein {record['protein_id']} has no lower-triangle rows.")

        feature_mean = feat_sum / float(total_rows)
        feature_sq_mean = feat_sq_sum / float(total_rows)
        record["feature_mean_raw"] = feature_mean
        record["feature_sq_mean_raw"] = feature_sq_mean

        mean_acc += feature_mean
        sq_acc += feature_sq_mean

    mean = mean_acc / float(len(train_records))
    second_moment = sq_acc / float(len(train_records))
    variance = np.maximum(second_moment - mean * mean, FLOAT_EPS)
    scale = np.sqrt(variance)
    scale[scale < FLOAT_EPS] = 1.0
    return {"mean": mean, "scale": scale}


def build_protein_mass_lookup(records, objective_mode, fairness_lookup=None):
    """Build the total mass each protein contributes to the weighted loss."""
    masses = {}
    for record in records:
        if objective_mode == "protein_equal_mse":
            base_mass = 1.0
        elif objective_mode == "protein_equal_nmse":
            base_mass = 1.0 / max(float(record["target_variance"]), FLOAT_EPS)
        else:
            raise ValueError(f"Unsupported objective_mode: {objective_mode}")

        fairness_multiplier = 1.0
        if fairness_lookup is not None:
            fairness_multiplier = float(fairness_lookup[record["protein_id"]])

        masses[record["protein_id"]] = base_mass * fairness_multiplier

    mean_mass = float(np.mean(list(masses.values())))
    if mean_mass > 0.0:
        for protein_id in masses:
            masses[protein_id] /= mean_mass
    return masses


def fit_weighted_linear_model(train_records, protein_mass_lookup, standardization):
    """Fit one weighted linear model with a streaming normal-equation solve."""
    feat_mean_std_num = np.zeros((N_CHANNELS,), dtype=np.float64)
    y_mean_num = 0.0
    total_mass = 0.0

    for record in train_records:
        protein_id = record["protein_id"]
        mass = float(protein_mass_lookup[protein_id])
        feat_mean_std = (record["feature_mean_raw"] - standardization["mean"]) / standardization["scale"]
        feat_mean_std_num += mass * feat_mean_std
        y_mean_num += mass * float(record["target_mean"])
        total_mass += mass

    if total_mass <= 0.0:
        raise ValueError("Total protein mass is non-positive.")

    x_mean = feat_mean_std_num / total_mass
    y_mean = y_mean_num / total_mass

    xtx = np.zeros((N_CHANNELS, N_CHANNELS), dtype=np.float64)
    xty = np.zeros((N_CHANNELS,), dtype=np.float64)

    for record in train_records:
        protein_id = record["protein_id"]
        mass = float(protein_mass_lookup[protein_id])
        row_weight = mass / float(record["n_pairs"])

        for x_block, y_block in iter_record_blocks(record):
            x_std = (x_block - standardization["mean"]) / standardization["scale"]
            x_centered = x_std - x_mean
            y_centered = y_block - y_mean
            xtx += row_weight * (x_centered.T @ x_centered)
            xty += row_weight * (x_centered.T @ y_centered)

    xtx.flat[:: N_CHANNELS + 1] += RIDGE_ALPHA
    try:
        coef = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        coef = np.linalg.pinv(xtx) @ xty

    intercept = float(y_mean - np.dot(x_mean, coef))
    return {"coef": coef, "intercept": intercept}


def predict_block(model, x_block, standardization):
    """Predict one raw feature block with a standardized linear model."""
    x_std = (x_block - standardization["mean"]) / standardization["scale"]
    return np.asarray(x_std @ model["coef"] + model["intercept"], dtype=np.float64)


def compute_per_protein_objectives(records, model, standardization, objective_mode):
    """Compute train-side per-protein losses for fairness reweighting."""
    losses = {}
    r2_scores = {}

    for record in records:
        sum_y = 0.0
        sum_y2 = 0.0
        sum_sqerr = 0.0
        n_rows = 0

        for x_block, y_block in iter_record_blocks(record):
            pred_block = predict_block(model, x_block, standardization)
            err_block = pred_block - y_block
            sum_y += float(np.sum(y_block, dtype=np.float64))
            sum_y2 += float(np.sum(y_block * y_block, dtype=np.float64))
            sum_sqerr += float(np.sum(err_block * err_block, dtype=np.float64))
            n_rows += int(y_block.size)

        if n_rows <= 0:
            losses[record["protein_id"]] = float("nan")
            r2_scores[record["protein_id"]] = float("nan")
            continue

        mse_value = sum_sqerr / float(n_rows)
        if objective_mode == "protein_equal_mse":
            loss_value = mse_value
        elif objective_mode == "protein_equal_nmse":
            loss_value = mse_value / max(float(record["target_variance"]), FLOAT_EPS)
        else:
            raise ValueError(f"Unsupported objective_mode: {objective_mode}")

        mean_y = sum_y / float(n_rows)
        ss_tot = sum_y2 - 2.0 * mean_y * sum_y + float(n_rows) * mean_y * mean_y
        if ss_tot <= FLOAT_EPS:
            r2_value = float("nan")
        else:
            r2_value = float(1.0 - (sum_sqerr / ss_tot))

        losses[record["protein_id"]] = float(loss_value)
        r2_scores[record["protein_id"]] = float(r2_value)

    return losses, r2_scores


def update_fairness_lookup(loss_lookup):
    """Convert per-protein train losses into the next fairness multipliers."""
    values = np.array(
        [float(loss) for loss in loss_lookup.values() if not math.isnan(float(loss))],
        dtype=np.float64,
    )
    if values.size == 0:
        return {protein_id: 1.0 for protein_id in loss_lookup}

    mean_loss = max(float(np.mean(values)), FLOAT_EPS)
    out = {}
    for protein_id, loss_value in loss_lookup.items():
        if math.isnan(float(loss_value)):
            multiplier = 1.0
        else:
            ratio = max(float(loss_value), FLOAT_EPS) / mean_loss
            multiplier = float(ratio ** FAIRNESS_POWER)
            multiplier = float(
                np.clip(multiplier, FAIRNESS_MIN_MULTIPLIER, FAIRNESS_MAX_MULTIPLIER)
            )
        out[protein_id] = multiplier

    mean_multiplier = float(np.mean(list(out.values())))
    if mean_multiplier > 0.0:
        for protein_id in out:
            out[protein_id] /= mean_multiplier
    return out


def fit_variant_model(variant, train_records, standardization):
    """Fit one variant and return the final model plus fairness diagnostics."""
    variant_name = variant["name"]
    objective_mode = variant["objective_mode"]
    uses_iterative_fairness = bool(variant["uses_iterative_fairness"])

    fairness_lookup = {record["protein_id"]: 1.0 for record in train_records}
    round_summary_rows = []
    round_history_rows = []
    final_model = None
    final_losses = None

    n_rounds = 1 if not uses_iterative_fairness else FAIRNESS_MAX_ROUNDS
    for round_index in range(n_rounds):
        protein_mass_lookup = build_protein_mass_lookup(
            train_records,
            objective_mode,
            fairness_lookup=fairness_lookup,
        )
        model = fit_weighted_linear_model(train_records, protein_mass_lookup, standardization)
        losses, r2_scores = compute_per_protein_objectives(
            train_records,
            model,
            standardization,
            objective_mode,
        )

        loss_values = np.array(
            [float(value) for value in losses.values() if not math.isnan(float(value))],
            dtype=np.float64,
        )
        mean_loss = float(np.mean(loss_values)) if loss_values.size else float("nan")
        std_loss = float(np.std(loss_values)) if loss_values.size else float("nan")
        median_loss = float(np.median(loss_values)) if loss_values.size else float("nan")
        max_loss = float(np.max(loss_values)) if loss_values.size else float("nan")

        next_fairness_lookup = fairness_lookup
        max_log_change = 0.0
        converged = True
        if uses_iterative_fairness:
            next_fairness_lookup = update_fairness_lookup(losses)
            deltas = []
            for protein_id, old_value in fairness_lookup.items():
                new_value = float(next_fairness_lookup[protein_id])
                old_value = max(float(old_value), FLOAT_EPS)
                deltas.append(abs(math.log(new_value / old_value)))
            max_log_change = float(max(deltas)) if deltas else 0.0
            converged = bool(max_log_change <= FAIRNESS_CONVERGENCE_TOL)

        round_summary_rows.append(
            {
                "variant": variant_name,
                "round_index": int(round_index),
                "objective_mode": objective_mode,
                "mean_train_objective": mean_loss,
                "std_train_objective": std_loss,
                "median_train_objective": median_loss,
                "max_train_objective": max_loss,
                "min_multiplier": float(min(fairness_lookup.values())),
                "max_multiplier": float(max(fairness_lookup.values())),
                "max_log_multiplier_change": max_log_change,
                "converged_after_round": bool(converged),
            }
        )

        for record in train_records:
            protein_id = record["protein_id"]
            round_history_rows.append(
                {
                    "variant": variant_name,
                    "round_index": int(round_index),
                    "protein_id": protein_id,
                    "n_residues": int(record["n_residues"]),
                    "size_bin": int(record["size_bin"]),
                    "n_pairs": int(record["n_pairs"]),
                    "target_variance": float(record["target_variance"]),
                    "train_objective_loss": float(losses[protein_id]),
                    "train_r2": float(r2_scores[protein_id]),
                    "fairness_multiplier": float(fairness_lookup[protein_id]),
                    "effective_protein_mass": float(protein_mass_lookup[protein_id]),
                }
            )

        final_model = model
        final_losses = losses
        if not uses_iterative_fairness:
            break
        if converged and round_index >= 1:
            break
        fairness_lookup = next_fairness_lookup

    if final_model is None or final_losses is None:
        raise RuntimeError(f"Variant {variant_name} did not produce a fitted model.")

    return final_model, round_summary_rows, round_history_rows, final_losses


def summarize_split(records, model, standardization):
    """Compute per-protein rows and pooled metrics for one split."""
    per_rows = []
    per_r2_values = []
    per_nmse_values = []
    negative_r2 = 0

    pair_sum_y = 0.0
    pair_sum_y2 = 0.0
    pair_sum_abs = 0.0
    pair_sum_sqerr = 0.0
    pair_total_n = 0

    protein_weight_sum = 0.0
    protein_weighted_sum_y = 0.0
    protein_weighted_sum_y2 = 0.0
    protein_weighted_sum_abs = 0.0
    protein_weighted_sum_sqerr = 0.0

    for record in records:
        sum_y = 0.0
        sum_y2 = 0.0
        sum_pred = 0.0
        sum_abs = 0.0
        sum_sqerr = 0.0
        n_rows = 0

        for x_block, y_block in iter_record_blocks(record):
            pred_block = predict_block(model, x_block, standardization)
            err_block = pred_block - y_block

            sum_y += float(np.sum(y_block, dtype=np.float64))
            sum_y2 += float(np.sum(y_block * y_block, dtype=np.float64))
            sum_pred += float(np.sum(pred_block, dtype=np.float64))
            sum_abs += float(np.sum(np.abs(err_block), dtype=np.float64))
            sum_sqerr += float(np.sum(err_block * err_block, dtype=np.float64))
            n_rows += int(y_block.size)

        if n_rows <= 0:
            continue

        mean_y = sum_y / float(n_rows)
        mean_pred = sum_pred / float(n_rows)
        mse_value = sum_sqerr / float(n_rows)
        mae_value = sum_abs / float(n_rows)
        rmse_value = float(math.sqrt(mse_value))
        ss_tot = sum_y2 - 2.0 * mean_y * sum_y + float(n_rows) * mean_y * mean_y
        if ss_tot <= FLOAT_EPS:
            r2_value = float("nan")
        else:
            r2_value = float(1.0 - (sum_sqerr / ss_tot))

        nmse_value = float(mse_value / max(float(record["target_variance"]), FLOAT_EPS))
        if not math.isnan(r2_value):
            per_r2_values.append(float(r2_value))
            if r2_value < 0.0:
                negative_r2 += 1
        per_nmse_values.append(nmse_value)

        per_rows.append(
            {
                "protein_id": record["protein_id"],
                "n_residues": int(record["n_residues"]),
                "size_bin": int(record["size_bin"]),
                "n_pairs": int(record["n_pairs"]),
                "target_variance": float(record["target_variance"]),
                "per_protein_r2": float(r2_value),
                "per_protein_mse": float(mse_value),
                "per_protein_nmse": float(nmse_value),
                "per_protein_mae": float(mae_value),
                "per_protein_rmse": float(rmse_value),
                "mean_true_distance": float(mean_y),
                "mean_predicted_distance": float(mean_pred),
                "mean_error_pred_minus_true": float(mean_pred - mean_y),
            }
        )

        pair_sum_y += sum_y
        pair_sum_y2 += sum_y2
        pair_sum_abs += sum_abs
        pair_sum_sqerr += sum_sqerr
        pair_total_n += int(n_rows)

        row_weight = 1.0 / float(n_rows)
        protein_weight_sum += 1.0
        protein_weighted_sum_y += row_weight * sum_y
        protein_weighted_sum_y2 += row_weight * sum_y2
        protein_weighted_sum_abs += row_weight * sum_abs
        protein_weighted_sum_sqerr += row_weight * sum_sqerr

    pair_mean = pair_sum_y / max(float(pair_total_n), 1.0)
    pair_ss_tot = (
        pair_sum_y2
        - 2.0 * pair_mean * pair_sum_y
        + float(pair_total_n) * pair_mean * pair_mean
    )
    if pair_ss_tot <= FLOAT_EPS:
        pair_r2 = float("nan")
    else:
        pair_r2 = float(1.0 - (pair_sum_sqerr / pair_ss_tot))

    pair_mae = pair_sum_abs / max(float(pair_total_n), 1.0)
    pair_rmse = float(math.sqrt(pair_sum_sqerr / max(float(pair_total_n), 1.0)))

    protein_mean_y = protein_weighted_sum_y / max(protein_weight_sum, 1.0)
    protein_ss_tot = (
        protein_weighted_sum_y2
        - 2.0 * protein_mean_y * protein_weighted_sum_y
        + protein_weight_sum * protein_mean_y * protein_mean_y
    )
    if protein_ss_tot <= FLOAT_EPS:
        protein_weighted_r2 = float("nan")
    else:
        protein_weighted_r2 = float(1.0 - (protein_weighted_sum_sqerr / protein_ss_tot))

    return {
        "pair_weighted_r2": pair_r2,
        "protein_weighted_r2": protein_weighted_r2,
        "pair_weighted_mae": float(pair_mae),
        "pair_weighted_rmse": float(pair_rmse),
        "mean_per_protein_r2": (
            float(np.mean(per_r2_values)) if per_r2_values else float("nan")
        ),
        "median_per_protein_r2": (
            float(np.median(per_r2_values)) if per_r2_values else float("nan")
        ),
        "negative_r2_fraction": (
            float(negative_r2 / len(per_r2_values)) if per_r2_values else float("nan")
        ),
        "mean_per_protein_nmse": (
            float(np.mean(per_nmse_values)) if per_nmse_values else float("nan")
        ),
        "median_per_protein_nmse": (
            float(np.median(per_nmse_values)) if per_nmse_values else float("nan")
        ),
        "max_per_protein_nmse": (
            float(np.max(per_nmse_values)) if per_nmse_values else float("nan")
        ),
        "per_rows": per_rows,
    }


def build_split_rows(train_records, test_records):
    """Format protein-level split assignments."""
    rows = []
    for split_name, split_records in [("train", train_records), ("test", test_records)]:
        for record in split_records:
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


def build_coefficient_rows(variant_name, model, standardization):
    """Convert standardized coefficients back to the raw feature scale."""
    coef_raw = np.asarray(model["coef"], dtype=np.float64) / standardization["scale"]
    intercept_raw = float(
        model["intercept"]
        - np.sum((standardization["mean"] / standardization["scale"]) * model["coef"])
    )

    rows = [
        {
            "variant": variant_name,
            "feature_index": -1,
            "feature_name": "intercept",
            "coefficient_raw_scale": intercept_raw,
        }
    ]
    for feature_index, feature_name in enumerate(FEATURE_NAMES):
        rows.append(
            {
                "variant": variant_name,
                "feature_index": int(feature_index),
                "feature_name": feature_name,
                "coefficient_raw_scale": float(coef_raw[feature_index]),
            }
        )
    return rows


def evaluate_variant(variant, train_records, test_records, standardization):
    """Fit one variant and collect summary/per-protein outputs."""
    t0 = time.time()
    model, round_summary_rows, round_history_rows, final_losses = fit_variant_model(
        variant,
        train_records,
        standardization,
    )
    train_metrics = summarize_split(train_records, model, standardization)
    test_metrics = summarize_split(test_records, model, standardization)

    per_protein_rows = []
    for split_name, metric_group in [("train", train_metrics), ("test", test_metrics)]:
        for row in metric_group["per_rows"]:
            out = dict(row)
            out["variant"] = variant["name"]
            out["split"] = split_name
            per_protein_rows.append(out)

    loss_values = np.array(
        [float(value) for value in final_losses.values() if not math.isnan(float(value))],
        dtype=np.float64,
    )
    final_mean_loss = float(np.mean(loss_values)) if loss_values.size else float("nan")
    final_std_loss = float(np.std(loss_values)) if loss_values.size else float("nan")
    final_max_loss = float(np.max(loss_values)) if loss_values.size else float("nan")

    summary_row = {
        "variant": variant["name"],
        "objective_mode": variant["objective_mode"],
        "uses_iterative_fairness": bool(variant["uses_iterative_fairness"]),
        "n_train_proteins": int(len(train_records)),
        "n_test_proteins": int(len(test_records)),
        "n_train_rows": int(sum(record["n_pairs"] for record in train_records)),
        "n_test_rows": int(sum(record["n_pairs"] for record in test_records)),
        "train_pair_weighted_r2": float(train_metrics["pair_weighted_r2"]),
        "test_pair_weighted_r2": float(test_metrics["pair_weighted_r2"]),
        "train_protein_weighted_r2": float(train_metrics["protein_weighted_r2"]),
        "test_protein_weighted_r2": float(test_metrics["protein_weighted_r2"]),
        "train_pair_weighted_mae": float(train_metrics["pair_weighted_mae"]),
        "test_pair_weighted_mae": float(test_metrics["pair_weighted_mae"]),
        "train_pair_weighted_rmse": float(train_metrics["pair_weighted_rmse"]),
        "test_pair_weighted_rmse": float(test_metrics["pair_weighted_rmse"]),
        "train_mean_per_protein_r2": float(train_metrics["mean_per_protein_r2"]),
        "test_mean_per_protein_r2": float(test_metrics["mean_per_protein_r2"]),
        "test_median_per_protein_r2": float(test_metrics["median_per_protein_r2"]),
        "test_negative_r2_fraction": float(test_metrics["negative_r2_fraction"]),
        "train_mean_per_protein_nmse": float(train_metrics["mean_per_protein_nmse"]),
        "test_mean_per_protein_nmse": float(test_metrics["mean_per_protein_nmse"]),
        "test_median_per_protein_nmse": float(test_metrics["median_per_protein_nmse"]),
        "test_max_per_protein_nmse": float(test_metrics["max_per_protein_nmse"]),
        "fairness_rounds_completed": int(len(round_summary_rows)),
        "final_train_objective_mean": final_mean_loss,
        "final_train_objective_std": final_std_loss,
        "final_train_objective_max": final_max_loss,
        "elapsed_seconds": float(time.time() - t0),
    }

    coefficient_rows = build_coefficient_rows(variant["name"], model, standardization)
    return {
        "summary_row": summary_row,
        "per_protein_rows": per_protein_rows,
        "round_summary_rows": round_summary_rows,
        "round_history_rows": round_history_rows,
        "coefficient_rows": coefficient_rows,
    }


def build_summary_text(
    path,
    elapsed_seconds,
    all_proteins,
    loaded_records,
    skipped,
    train_records,
    test_records,
    summary_rows,
):
    """Write a short human-readable summary for quick inspection."""
    best_row = max(
        summary_rows,
        key=lambda row: (
            -1e18
            if math.isnan(float(row["test_mean_per_protein_r2"]))
            else float(row["test_mean_per_protein_r2"])
        ),
    )

    with path.open("w") as handle:
        handle.write(f"{TITLE}\n")
        handle.write(f"{'=' * len(TITLE)}\n\n")
        handle.write("Model family:      shared linear regression\n")
        handle.write("Feature policy:    lower-triangle center only, layer 47, 128 channels\n")
        handle.write("Target policy:     absolute CA distances\n")
        handle.write(f"Train/test split:  protein-level 80/20, seed={RANDOM_STATE}\n")
        handle.write(
            "Objectives:        mean protein MSE, mean protein NMSE, and iterative "
            "fairness-reweighted mean protein NMSE\n"
        )
        handle.write(
            "Linearity:         every fitted predictor is still y = x @ beta + b; "
            "the fairness variant changes refit weights across rounds, not the "
            "prediction form\n\n"
        )
        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(loaded_records)}\n")
        handle.write(f"Train proteins:      {len(train_records)}\n")
        handle.write(f"Test proteins:       {len(test_records)}\n")
        handle.write(f"Elapsed seconds:     {elapsed_seconds:.2f}\n\n")

        handle.write("Variant summary:\n")
        for row in summary_rows:
            handle.write(
                f"  {row['variant']}: "
                f"test mean per-protein R2={float(row['test_mean_per_protein_r2']):.4f}, "
                f"test protein-weighted R2={float(row['test_protein_weighted_r2']):.4f}, "
                f"test mean per-protein NMSE={float(row['test_mean_per_protein_nmse']):.4f}, "
                f"fairness rounds={int(row['fairness_rounds_completed'])}\n"
            )

        handle.write(
            f"\nBest variant by test mean per-protein R2: {best_row['variant']} "
            f"({float(best_row['test_mean_per_protein_r2']):.4f})\n"
        )

        if skipped:
            handle.write("\nSkipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")


def main():
    t0 = time.time()
    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))
    print(f"Output CSV dir: {csv_dir}")
    print(f"Feature policy: lower-triangle center only, layer={LAYER}, dim={N_CHANNELS}")
    print(f"Variants:       {len(VARIANTS)}")

    all_proteins, protein_data, skipped = load_protein_records()
    if not protein_data:
        print("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    prepared_records = prepare_records(protein_data)
    assign_size_bins(prepared_records, N_SIZE_BINS)
    train_records, test_records = split_global_records(prepared_records)
    standardization = compute_feature_standardization(train_records)

    print(f"Loaded proteins: {len(prepared_records)}")
    print(f"Train proteins:  {len(train_records)}")
    print(f"Test proteins:   {len(test_records)}")

    summary_rows = []
    per_protein_rows = []
    round_summary_rows = []
    round_history_rows = []
    coefficient_rows = []

    for variant in VARIANTS:
        print(f"\n[Variant] {variant['name']}")
        result = evaluate_variant(variant, train_records, test_records, standardization)
        summary_rows.append(result["summary_row"])
        per_protein_rows.extend(result["per_protein_rows"])
        round_summary_rows.extend(result["round_summary_rows"])
        round_history_rows.extend(result["round_history_rows"])
        coefficient_rows.extend(result["coefficient_rows"])
        print(
            "  "
            f"test mean per-protein R2={result['summary_row']['test_mean_per_protein_r2']:.4f}, "
            f"test mean per-protein NMSE={result['summary_row']['test_mean_per_protein_nmse']:.4f}, "
            f"rounds={result['summary_row']['fairness_rounds_completed']}"
        )

    split_rows = build_split_rows(train_records, test_records)

    write_csv(csv_dir / SUMMARY_CSV_NAME, SUMMARY_FIELDNAMES, summary_rows)
    write_csv(csv_dir / PER_PROTEIN_CSV_NAME, PER_PROTEIN_FIELDNAMES, per_protein_rows)
    write_csv(csv_dir / SPLIT_CSV_NAME, SPLIT_FIELDNAMES, split_rows)
    write_csv(
        csv_dir / ROUND_SUMMARY_CSV_NAME,
        ROUND_SUMMARY_FIELDNAMES,
        round_summary_rows,
    )
    write_csv(
        csv_dir / ROUND_HISTORY_CSV_NAME,
        ROUND_HISTORY_FIELDNAMES,
        round_history_rows,
    )
    write_csv(
        csv_dir / COEFFICIENT_CSV_NAME,
        COEFFICIENT_FIELDNAMES,
        coefficient_rows,
    )
    build_summary_text(
        csv_dir / SUMMARY_TXT_NAME,
        elapsed_seconds=float(time.time() - t0),
        all_proteins=all_proteins,
        loaded_records=prepared_records,
        skipped=skipped,
        train_records=train_records,
        test_records=test_records,
        summary_rows=summary_rows,
    )

    print(f"\nSaved summary CSV -> {csv_dir / SUMMARY_CSV_NAME}")
    print(f"Saved per-protein CSV -> {csv_dir / PER_PROTEIN_CSV_NAME}")
    print(f"Saved split CSV -> {csv_dir / SPLIT_CSV_NAME}")
    print(f"Saved round summary CSV -> {csv_dir / ROUND_SUMMARY_CSV_NAME}")
    print(f"Saved round history CSV -> {csv_dir / ROUND_HISTORY_CSV_NAME}")
    print(f"Saved coefficient CSV -> {csv_dir / COEFFICIENT_CSV_NAME}")
    print(f"Saved text summary -> {csv_dir / SUMMARY_TXT_NAME}")


if __name__ == "__main__":
    main()

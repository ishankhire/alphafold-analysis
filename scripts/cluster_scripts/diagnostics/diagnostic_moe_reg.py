#!/usr/bin/env python3
"""
diagnostic_moe_reg.py
=====================
Layer-47 diagnostic regression script for lower-triangle linear
mixture-of-experts variants on Delta.

Requested variants:
1. lower_triangle_linear_baseline
2. soft_size_gated_linear_moe_k2
3. soft_size_gated_linear_moe_k3
4. shared_global_residual_moe_k2
5. shared_global_residual_moe_k3

Model policy:
- lower-triangle pair representation only, pb[i,j,:] for i > j
- gate on log(n_residues) only
- 1D Gaussian gate
- protein-equal generalized EM
- plain LinearRegression experts

Outputs (inside CSV_ROOT/):
    diagnostic_moe_reg_variant_summary.csv
    diagnostic_moe_reg_per_protein.csv
    diagnostic_moe_reg_split.csv
    diagnostic_moe_reg_seq_sep_diagnostics.csv
    diagnostic_moe_reg_coefficients_raw.csv
    diagnostic_moe_reg_gate_parameters.csv
    diagnostic_moe_reg_protein_responsibilities.csv
    diagnostic_moe_reg_em_trace.csv
    diagnostic_moe_reg_gate_grid.csv
    diagnostic_moe_reg_summary.txt
"""

import math
import time

import numpy as np
from sklearn.linear_model import LinearRegression

from diagnostic_tree_regression_common import (
    CSV_DIR,
    FLOAT_EPS,
    LAYER,
    MAX_TRAIN_PAIRS_PER_PROTEIN,
    N_CHANNELS,
    compute_pair_weights,
    deterministic_sample_indices,
    format_value,
    get_pair_block_path,
    load_protein_records,
    build_seq_sep_diagnostics,
    build_split_rows,
    split_records,
    summarize_predictions,
    write_csv,
)


TITLE = "Diagnostic Mixture-of-Experts Regression - Layer 47"
MODEL_TYPE_BASELINE = "linear_baseline"
MODEL_TYPE_SOFT_MOE = "soft_size_gated_linear_moe"
MODEL_TYPE_SHARED_RESIDUAL = "shared_global_residual_moe"

FEATURE_NAMES = [f"pair_ch_{idx}" for idx in range(N_CHANNELS)]

SUMMARY_CSV_NAME = "diagnostic_moe_reg_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_moe_reg_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_moe_reg_split.csv"
SEQ_DIAGNOSTICS_CSV_NAME = "diagnostic_moe_reg_seq_sep_diagnostics.csv"
COEFFICIENTS_CSV_NAME = "diagnostic_moe_reg_coefficients_raw.csv"
GATE_PARAMS_CSV_NAME = "diagnostic_moe_reg_gate_parameters.csv"
RESPONSIBILITIES_CSV_NAME = "diagnostic_moe_reg_protein_responsibilities.csv"
EM_TRACE_CSV_NAME = "diagnostic_moe_reg_em_trace.csv"
GATE_GRID_CSV_NAME = "diagnostic_moe_reg_gate_grid.csv"
SUMMARY_TXT_NAME = "diagnostic_moe_reg_summary.txt"

EM_MAX_ITERS = 50
EM_TOL = 1e-4
N_RESTARTS = 5
MIN_GATE_WEIGHT = 1e-6
MIN_GATE_VARIANCE = 1e-4
MIN_NOISE_VARIANCE = 1e-4
MIN_EFFECTIVE_MASS = 1e-6
INIT_MEAN_JITTER_FRAC = 0.05
GATE_GRID_SIZE = 100

VARIANTS = [
    {
        "name": "lower_triangle_linear_baseline",
        "model_type": MODEL_TYPE_BASELINE,
        "n_experts": 1,
        "uses_shared_global_model": False,
    },
    {
        "name": "soft_size_gated_linear_moe_k2",
        "model_type": MODEL_TYPE_SOFT_MOE,
        "n_experts": 2,
        "uses_shared_global_model": False,
    },
    {
        "name": "soft_size_gated_linear_moe_k3",
        "model_type": MODEL_TYPE_SOFT_MOE,
        "n_experts": 3,
        "uses_shared_global_model": False,
    },
    {
        "name": "shared_global_residual_moe_k2",
        "model_type": MODEL_TYPE_SHARED_RESIDUAL,
        "n_experts": 2,
        "uses_shared_global_model": True,
    },
    {
        "name": "shared_global_residual_moe_k3",
        "model_type": MODEL_TYPE_SHARED_RESIDUAL,
        "n_experts": 3,
        "uses_shared_global_model": True,
    },
]


class FixedLinearModel:
    """Small predict-only linear model for collapsed expert fallbacks."""

    def __init__(self, coef, intercept):
        self.coef_ = np.asarray(coef, dtype=np.float64)
        self.intercept_ = float(intercept)

    def predict(self, x):
        return np.asarray(x, dtype=np.float64) @ self.coef_ + self.intercept_


def logsumexp_rows(scores):
    """Stable row-wise logsumexp."""
    scores = np.asarray(scores, dtype=np.float64)
    max_scores = np.max(scores, axis=1, keepdims=True)
    stable = np.exp(scores - max_scores)
    return max_scores[:, 0] + np.log(np.sum(stable, axis=1) + FLOAT_EPS)


def normalize_log_scores(scores):
    """Convert log-scores to row-normalized probabilities."""
    norm = logsumexp_rows(scores)
    probs = np.exp(scores - norm[:, np.newaxis])
    probs_sum = np.sum(probs, axis=1, keepdims=True)
    probs /= np.clip(probs_sum, FLOAT_EPS, None)
    return probs


def compute_entropy(probabilities):
    """Compute Shannon entropy row-wise."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    safe = np.clip(probabilities, FLOAT_EPS, 1.0)
    return -np.sum(safe * np.log(safe), axis=1)


def gaussian_logpdf(x, mean, variance):
    """Compute 1D Gaussian log-density."""
    variance = max(float(variance), MIN_GATE_VARIANCE)
    return -0.5 * (
        math.log(2.0 * math.pi * variance)
        + ((np.asarray(x, dtype=np.float64) - float(mean)) ** 2) / variance
    )


def build_lower_triangle_bundle(records, sample_cap=None, sample_offset=0):
    """Materialize lower-triangle-only features for one split."""
    row_counts = []
    selections = []
    for record in records:
        local_idx = deterministic_sample_indices(
            label=f"{record['protein_id']}:{record['n_pairs']}",
            n_rows=record["n_pairs"],
            cap=sample_cap,
            offset=sample_offset,
        )
        selections.append(local_idx)
        row_counts.append(int(local_idx.size))

    total_rows = int(sum(row_counts))
    x = np.empty((total_rows, N_CHANNELS), dtype=np.float32)
    y = np.empty((total_rows,), dtype=np.float64)
    seq_sep_rows = np.empty((total_rows,), dtype=np.int32)
    offsets = []
    protein_ids = []
    n_residues = np.empty((len(records),), dtype=np.int32)
    log_n_residues = np.empty((len(records),), dtype=np.float64)

    start = 0
    for protein_index, (record, local_idx, n_rows) in enumerate(
        zip(records, selections, row_counts)
    ):
        end = start + n_rows
        pb_path = get_pair_block_path(record["protein_dir"], record["protein_id"], LAYER)
        pb = np.load(str(pb_path), mmap_mode="r")

        lt_i = record["lt_i"][local_idx]
        lt_j = record["lt_j"][local_idx]

        x[start:end, :] = pb[lt_i, lt_j, :]
        y[start:end] = record["y_abs"][local_idx]
        seq_sep_rows[start:end] = record["seq_sep"][local_idx].astype(
            np.int32,
            copy=False,
        )
        offsets.append((start, end))

        protein_ids.append(record["protein_id"])
        n_residues[protein_index] = int(record["n_residues"])
        log_n_residues[protein_index] = math.log(float(record["n_residues"]))

        del pb
        start = end

    return {
        "x": x,
        "y": y,
        "seq_sep": seq_sep_rows,
        "offsets": offsets,
        "row_counts": np.asarray(row_counts, dtype=np.int32),
        "weights": compute_pair_weights(row_counts),
        "protein_ids": protein_ids,
        "n_residues": n_residues,
        "log_n_residues": log_n_residues,
    }


def standardize_features(train_bundle, eval_bundles):
    """Standardize feature columns using train-only moments."""
    mean = np.mean(train_bundle["x"], axis=0, dtype=np.float64)
    scale = np.std(train_bundle["x"], axis=0, dtype=np.float64)
    scale[scale < FLOAT_EPS] = 1.0

    train_bundle["x"] -= mean
    train_bundle["x"] /= scale
    for bundle in eval_bundles:
        bundle["x"] -= mean
        bundle["x"] /= scale

    return {"mean": mean, "scale": scale}


def to_raw_linear_params(model, standardization):
    """Convert standardized linear params back to raw feature scale."""
    coef_std = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    intercept_std = float(model.intercept_)
    mean = np.asarray(standardization["mean"], dtype=np.float64)
    scale = np.asarray(standardization["scale"], dtype=np.float64)
    coef_raw = coef_std / scale
    intercept_raw = intercept_std - float(np.dot(coef_std, mean / scale))
    return coef_raw, intercept_raw


def fit_linear_model(x, y, sample_weight):
    """Fit a weighted linear regression."""
    model = LinearRegression(copy_X=False)
    model.fit(x, y, sample_weight=sample_weight)
    return model


def fit_linear_model_with_fallback(x, y, sample_weight):
    """Fit a weighted linear regression or fall back to an intercept-only model."""
    total_weight = float(np.sum(sample_weight))
    if total_weight <= MIN_EFFECTIVE_MASS:
        intercept = float(np.mean(y)) if y.size > 0 else 0.0
        return FixedLinearModel(np.zeros(x.shape[1], dtype=np.float64), intercept)

    try:
        return fit_linear_model(x, y, sample_weight=sample_weight)
    except Exception:
        intercept = float(np.average(y, weights=sample_weight))
        return FixedLinearModel(np.zeros(x.shape[1], dtype=np.float64), intercept)


def compute_gate_log_scores(log_n_residues, gate_weights, gate_means, gate_variances):
    """Compute gate-only log-scores for each protein and expert."""
    n_proteins = len(log_n_residues)
    n_experts = len(gate_weights)
    scores = np.empty((n_proteins, n_experts), dtype=np.float64)
    for expert_index in range(n_experts):
        scores[:, expert_index] = (
            math.log(max(float(gate_weights[expert_index]), MIN_GATE_WEIGHT))
            + gaussian_logpdf(
                log_n_residues,
                gate_means[expert_index],
                gate_variances[expert_index],
            )
        )
    return scores


def compute_gate_predictive_weights(log_n_residues, gate_weights, gate_means, gate_variances):
    """Compute gate weights for inference without target leakage."""
    scores = compute_gate_log_scores(
        log_n_residues,
        gate_weights,
        gate_means,
        gate_variances,
    )
    return normalize_log_scores(scores)


def compute_expert_prediction_matrix(models, x):
    """Predict one column per expert."""
    rows = x.shape[0]
    n_experts = len(models)
    predictions = np.empty((rows, n_experts), dtype=np.float64)
    for expert_index, model in enumerate(models):
        predictions[:, expert_index] = np.asarray(model.predict(x), dtype=np.float64)
    return predictions


def compute_protein_mean_sq_error_matrix(y_true, expert_predictions, offsets):
    """Compute one mean squared error per protein and expert."""
    n_proteins = len(offsets)
    n_experts = expert_predictions.shape[1]
    mse = np.empty((n_proteins, n_experts), dtype=np.float64)
    for protein_index, (start, end) in enumerate(offsets):
        residual = y_true[start:end, np.newaxis] - expert_predictions[start:end, :]
        mse[protein_index, :] = np.mean(residual * residual, axis=0)
    return mse


def expand_protein_weights_to_rows(protein_weights, row_counts):
    """Repeat per-protein weights across its rows with equal row mass."""
    row_counts = np.asarray(row_counts, dtype=np.float64)
    per_row = np.asarray(protein_weights, dtype=np.float64) / np.clip(row_counts, 1.0, None)
    return np.repeat(per_row, row_counts.astype(np.int32))


def initialize_responsibilities(log_n_residues, n_experts, seed):
    """Initialize soft assignments from a jittered 1D Gaussian gate."""
    rng = np.random.default_rng(seed)
    global_std = float(np.std(log_n_residues, dtype=np.float64))
    jitter_scale = max(global_std * INIT_MEAN_JITTER_FRAC, 1e-3)
    quantiles = np.linspace(0.2, 0.8, n_experts)
    means = np.quantile(log_n_residues, quantiles)
    means += rng.normal(loc=0.0, scale=jitter_scale, size=n_experts)
    means = np.clip(
        means,
        float(np.min(log_n_residues)),
        float(np.max(log_n_residues)),
    )
    variances = np.full(
        n_experts,
        max(float(np.var(log_n_residues, dtype=np.float64)), MIN_GATE_VARIANCE),
        dtype=np.float64,
    )
    weights = np.full(n_experts, 1.0 / n_experts, dtype=np.float64)
    return compute_gate_predictive_weights(log_n_residues, weights, means, variances)


def update_gate_parameters(log_n_residues, responsibilities, previous_gate=None):
    """Update Gaussian gate params from protein-level responsibilities."""
    n_experts = responsibilities.shape[1]
    global_mean = float(np.mean(log_n_residues))
    global_var = max(float(np.var(log_n_residues, dtype=np.float64)), MIN_GATE_VARIANCE)

    effective_mass = np.sum(responsibilities, axis=0, dtype=np.float64)
    gate_weights = effective_mass / max(float(responsibilities.shape[0]), 1.0)
    gate_weights = np.clip(gate_weights, MIN_GATE_WEIGHT, None)
    gate_weights /= np.sum(gate_weights)

    means = np.empty((n_experts,), dtype=np.float64)
    variances = np.empty((n_experts,), dtype=np.float64)
    for expert_index in range(n_experts):
        mass = float(effective_mass[expert_index])
        if mass > MIN_EFFECTIVE_MASS:
            means[expert_index] = float(
                np.sum(responsibilities[:, expert_index] * log_n_residues) / mass
            )
            diff = log_n_residues - means[expert_index]
            variances[expert_index] = max(
                float(np.sum(responsibilities[:, expert_index] * diff * diff) / mass),
                MIN_GATE_VARIANCE,
            )
        elif previous_gate is None:
            means[expert_index] = global_mean
            variances[expert_index] = global_var
        else:
            means[expert_index] = float(previous_gate["gate_means"][expert_index])
            variances[expert_index] = float(previous_gate["gate_variances"][expert_index])

    order = np.argsort(means)
    return {
        "gate_weights": gate_weights[order],
        "gate_means": means[order],
        "gate_variances": variances[order],
        "effective_mass": effective_mass[order],
        "order": order,
    }


def fit_experts_for_responsibilities(bundle, target, responsibilities):
    """Fit one expert per responsibility column."""
    models = []
    n_experts = responsibilities.shape[1]
    fallback_variance = max(float(np.var(target, dtype=np.float64)), MIN_NOISE_VARIANCE)

    for expert_index in range(n_experts):
        row_weights = expand_protein_weights_to_rows(
            responsibilities[:, expert_index],
            bundle["row_counts"],
        )
        model = fit_linear_model_with_fallback(bundle["x"], target, row_weights)
        models.append(model)

    expert_predictions = compute_expert_prediction_matrix(models, bundle["x"])
    mse_matrix = compute_protein_mean_sq_error_matrix(
        target,
        expert_predictions,
        bundle["offsets"],
    )

    noise_variances = np.empty((n_experts,), dtype=np.float64)
    for expert_index in range(n_experts):
        mass = float(np.sum(responsibilities[:, expert_index]))
        if mass > MIN_EFFECTIVE_MASS:
            noise_variances[expert_index] = max(
                float(
                    np.sum(responsibilities[:, expert_index] * mse_matrix[:, expert_index])
                    / mass
                ),
                MIN_NOISE_VARIANCE,
            )
        else:
            noise_variances[expert_index] = fallback_variance

    return models, expert_predictions, mse_matrix, noise_variances


def compute_posterior_responsibilities(
    log_n_residues,
    gate_weights,
    gate_means,
    gate_variances,
    mse_matrix,
    noise_variances,
):
    """Compute training-time posterior responsibilities."""
    gate_scores = compute_gate_log_scores(
        log_n_residues,
        gate_weights,
        gate_means,
        gate_variances,
    )
    noise_variances = np.asarray(noise_variances, dtype=np.float64)
    likelihood_scores = -0.5 * (
        np.log(2.0 * math.pi * noise_variances[np.newaxis, :])
        + (mse_matrix / noise_variances[np.newaxis, :])
    )
    total_scores = gate_scores + likelihood_scores
    responsibilities = normalize_log_scores(total_scores)
    objective = float(np.mean(logsumexp_rows(total_scores)))
    return responsibilities, objective


def predict_moe_target(model_state, bundle):
    """Predict the active target using gate-only inference weights."""
    gate_predict = compute_gate_predictive_weights(
        bundle["log_n_residues"],
        model_state["gate_weights"],
        model_state["gate_means"],
        model_state["gate_variances"],
    )
    expert_predictions = compute_expert_prediction_matrix(
        model_state["expert_models"],
        bundle["x"],
    )
    y_pred = np.empty((bundle["x"].shape[0],), dtype=np.float64)
    for protein_index, (start, end) in enumerate(bundle["offsets"]):
        y_pred[start:end] = expert_predictions[start:end, :] @ gate_predict[protein_index, :]
    return y_pred, gate_predict


def fit_gaussian_moe(bundle, target, n_experts, restart_seed, variant_name, stage_name):
    """Fit a protein-level Gaussian-gated MoE with generalized EM."""
    trace_rows = []
    failed = False
    failure_reason = ""
    converged = False
    responsibilities = initialize_responsibilities(
        bundle["log_n_residues"],
        n_experts,
        seed=restart_seed,
    )
    previous_gate = None
    previous_objective = None

    try:
        for iteration in range(EM_MAX_ITERS):
            gate_update = update_gate_parameters(
                bundle["log_n_residues"],
                responsibilities,
                previous_gate=previous_gate,
            )
            responsibilities = responsibilities[:, gate_update["order"]]
            gate_weights = gate_update["gate_weights"]
            gate_means = gate_update["gate_means"]
            gate_variances = gate_update["gate_variances"]

            expert_models, expert_predictions, mse_matrix, noise_variances = (
                fit_experts_for_responsibilities(bundle, target, responsibilities)
            )
            posterior, objective = compute_posterior_responsibilities(
                bundle["log_n_residues"],
                gate_weights,
                gate_means,
                gate_variances,
                mse_matrix,
                noise_variances,
            )
            max_resp_change = float(np.max(np.abs(posterior - responsibilities)))
            mean_entropy = float(np.mean(compute_entropy(posterior)))
            effective_mass = np.sum(posterior, axis=0, dtype=np.float64)

            if (
                not np.isfinite(objective)
                or not np.all(np.isfinite(posterior))
                or not np.all(np.isfinite(noise_variances))
            ):
                failed = True
                failure_reason = "non-finite objective or responsibility encountered"

            trace_rows.append(
                {
                    "variant": variant_name,
                    "stage": stage_name,
                    "restart_index": int(restart_seed),
                    "iteration": int(iteration),
                    "objective": objective,
                    "max_responsibility_change": max_resp_change,
                    "mean_posterior_entropy": mean_entropy,
                    "min_effective_mass": float(np.min(effective_mass)),
                    "max_effective_mass": float(np.max(effective_mass)),
                    "converged": False,
                    "failed": failed,
                    "failure_reason": failure_reason,
                }
            )

            previous_gate = {
                "gate_weights": gate_weights,
                "gate_means": gate_means,
                "gate_variances": gate_variances,
            }
            responsibilities = posterior

            if failed:
                break
            if previous_objective is not None:
                if (
                    abs(objective - previous_objective) <= EM_TOL
                    and max_resp_change <= EM_TOL
                ):
                    converged = True
                    break
            previous_objective = objective
    except Exception as exc:
        failed = True
        failure_reason = str(exc)

    final_gate = update_gate_parameters(
        bundle["log_n_residues"],
        responsibilities,
        previous_gate=previous_gate,
    )
    responsibilities = responsibilities[:, final_gate["order"]]
    expert_models, expert_predictions, mse_matrix, noise_variances = fit_experts_for_responsibilities(
        bundle,
        target,
        responsibilities,
    )
    posterior, final_objective = compute_posterior_responsibilities(
        bundle["log_n_residues"],
        final_gate["gate_weights"],
        final_gate["gate_means"],
        final_gate["gate_variances"],
        mse_matrix,
        noise_variances,
    )

    if trace_rows:
        trace_rows[-1]["converged"] = converged
        trace_rows[-1]["failed"] = failed
        trace_rows[-1]["failure_reason"] = failure_reason

    return {
        "expert_models": expert_models,
        "gate_weights": final_gate["gate_weights"],
        "gate_means": final_gate["gate_means"],
        "gate_variances": final_gate["gate_variances"],
        "noise_variances": noise_variances,
        "posterior_train": posterior,
        "train_objective": final_objective,
        "converged": converged,
        "failed": failed,
        "failure_reason": failure_reason,
        "iterations_completed": len(trace_rows),
        "trace_rows": trace_rows,
        "restart_seed": int(restart_seed),
    }


def build_metrics_rows(variant, split_name, records, bundle, y_pred, gate_weights):
    """Build per-protein metric rows with gate diagnostics."""
    metrics = summarize_predictions(
        split_name,
        records,
        bundle["offsets"],
        bundle["y"],
        y_pred,
    )
    entropy = compute_entropy(gate_weights)
    top_expert = np.argmax(gate_weights, axis=1)
    top_weight = np.max(gate_weights, axis=1)

    rows = []
    for record, metric_row, protein_entropy, protein_top_expert, protein_top_weight in zip(
        records,
        metrics["per_rows"],
        entropy,
        top_expert,
        top_weight,
    ):
        out = dict(metric_row)
        out["variant"] = variant["name"]
        out["model_type"] = variant["model_type"]
        out["log_n_residues"] = math.log(float(record["n_residues"]))
        out["gate_entropy"] = float(protein_entropy)
        out["top_expert"] = int(protein_top_expert)
        out["top_expert_weight"] = float(protein_top_weight)
        rows.append(out)
    return metrics, rows


def build_responsibility_rows(
    variant,
    split_name,
    responsibility_source,
    records,
    responsibilities,
):
    """Build long-form responsibility rows."""
    entropy = compute_entropy(responsibilities)
    top_expert = np.argmax(responsibilities, axis=1)
    rows = []
    for protein_index, record in enumerate(records):
        for expert_index in range(responsibilities.shape[1]):
            rows.append(
                {
                    "variant": variant["name"],
                    "model_type": variant["model_type"],
                    "split": split_name,
                    "responsibility_source": responsibility_source,
                    "protein_id": record["protein_id"],
                    "n_residues": int(record["n_residues"]),
                    "log_n_residues": math.log(float(record["n_residues"])),
                    "expert_index": int(expert_index),
                    "responsibility": float(responsibilities[protein_index, expert_index]),
                    "gate_entropy": float(entropy[protein_index]),
                    "top_expert": int(top_expert[protein_index]),
                }
            )
    return rows


def build_seq_diag_rows(variant, split_name, bundle, y_pred):
    """Build sequence-separation diagnostics for one split."""
    rows = build_seq_sep_diagnostics(split_name, bundle["seq_sep"], bundle["y"], y_pred)
    out = []
    for row in rows:
        enriched = dict(row)
        enriched["variant"] = variant["name"]
        enriched["model_type"] = variant["model_type"]
        out.append(enriched)
    return out


def build_coefficient_rows(
    variant,
    component_type,
    model,
    standardization,
    expert_index=-1,
):
    """Build long-form raw-scale coefficient rows for one linear component."""
    coef_raw, intercept_raw = to_raw_linear_params(model, standardization)
    rows = [
        {
            "variant": variant["name"],
            "model_type": variant["model_type"],
            "component_type": component_type,
            "expert_index": int(expert_index),
            "feature_index": -1,
            "feature_name": "intercept",
            "coefficient_raw": float(intercept_raw),
            "intercept_raw": float(intercept_raw),
        }
    ]
    for feature_index, coef_value in enumerate(coef_raw):
        rows.append(
            {
                "variant": variant["name"],
                "model_type": variant["model_type"],
                "component_type": component_type,
                "expert_index": int(expert_index),
                "feature_index": int(feature_index),
                "feature_name": FEATURE_NAMES[feature_index],
                "coefficient_raw": float(coef_value),
                "intercept_raw": float(intercept_raw),
            }
        )
    return rows


def build_gate_parameter_rows(variant, fitted_model, train_bundle):
    """Build final gate parameter rows for one MoE variant."""
    gate_predict = compute_gate_predictive_weights(
        train_bundle["log_n_residues"],
        fitted_model["gate_weights"],
        fitted_model["gate_means"],
        fitted_model["gate_variances"],
    )
    posterior = np.asarray(fitted_model["posterior_train"], dtype=np.float64)

    rows = []
    for expert_index in range(len(fitted_model["gate_weights"])):
        rows.append(
            {
                "variant": variant["name"],
                "model_type": variant["model_type"],
                "expert_index": int(expert_index),
                "gate_weight": float(fitted_model["gate_weights"][expert_index]),
                "gate_mean_log_n": float(fitted_model["gate_means"][expert_index]),
                "gate_variance_log_n": float(fitted_model["gate_variances"][expert_index]),
                "gate_std_log_n": float(math.sqrt(fitted_model["gate_variances"][expert_index])),
                "expert_noise_variance": float(fitted_model["noise_variances"][expert_index]),
                "posterior_effective_protein_mass_train": float(np.sum(posterior[:, expert_index])),
                "posterior_average_responsibility_train": float(np.mean(posterior[:, expert_index])),
                "gate_predict_effective_protein_mass_train": float(np.sum(gate_predict[:, expert_index])),
                "gate_predict_average_weight_train": float(np.mean(gate_predict[:, expert_index])),
            }
        )
    return rows


def build_gate_grid_rows(variant, fitted_model, min_n, max_n):
    """Evaluate the final gate across a smooth size grid."""
    if min_n < 2:
        min_n = 2
    log_grid = np.linspace(math.log(float(min_n)), math.log(float(max_n)), GATE_GRID_SIZE)
    gate_weights = compute_gate_predictive_weights(
        log_grid,
        fitted_model["gate_weights"],
        fitted_model["gate_means"],
        fitted_model["gate_variances"],
    )
    rows = []
    for grid_index, log_n_value in enumerate(log_grid):
        n_value = math.exp(float(log_n_value))
        for expert_index in range(gate_weights.shape[1]):
            rows.append(
                {
                    "variant": variant["name"],
                    "model_type": variant["model_type"],
                    "grid_index": int(grid_index),
                    "n_residues_grid": float(n_value),
                    "log_n_residues_grid": float(log_n_value),
                    "expert_index": int(expert_index),
                    "gate_weight": float(gate_weights[grid_index, expert_index]),
                }
            )
    return rows


def summarize_variant_row(
    variant,
    elapsed_seconds,
    fit_bundle,
    val_bundle,
    train_bundle,
    test_bundle,
    fit_metrics,
    val_metrics,
    train_metrics,
    test_metrics,
    chosen_restart_index,
    selected_restart_model=None,
    fitted_model=None,
):
    """Build one summary row for baseline or MoE variant."""
    gate_entropy_test = 0.0
    min_effective_mass = 1.0
    max_effective_mass = 1.0
    chosen_restart_converged = True
    chosen_restart_iterations = 1
    final_train_iterations = 1
    train_objective = float("nan")
    failure_reason = ""

    if selected_restart_model is not None:
        chosen_restart_converged = bool(selected_restart_model["converged"])
        chosen_restart_iterations = int(selected_restart_model["iterations_completed"])
        failure_reason = str(selected_restart_model.get("failure_reason", ""))

    if fitted_model is not None:
        gate_predict_test = compute_gate_predictive_weights(
            test_bundle["log_n_residues"],
            fitted_model["gate_weights"],
            fitted_model["gate_means"],
            fitted_model["gate_variances"],
        )
        gate_entropy_test = float(np.mean(compute_entropy(gate_predict_test)))
        effective_mass = np.sum(fitted_model["posterior_train"], axis=0, dtype=np.float64)
        min_effective_mass = float(np.min(effective_mass))
        max_effective_mass = float(np.max(effective_mass))
        final_train_iterations = int(fitted_model["iterations_completed"])
        train_objective = float(fitted_model["train_objective"])
        if not failure_reason:
            failure_reason = str(fitted_model.get("failure_reason", ""))

    return {
        "variant": variant["name"],
        "model_type": variant["model_type"],
        "n_experts": int(variant["n_experts"]),
        "uses_shared_global_model": bool(variant["uses_shared_global_model"]),
        "gate_feature_name": "log_n_residues",
        "gate_family": "gaussian_1d" if variant["n_experts"] > 1 else "none",
        "training_objective": (
            "protein_equal_generalized_em" if variant["n_experts"] > 1 else "protein_equal_weighted_linear_regression"
        ),
        "n_loaded_proteins": int(len(fit_bundle["protein_ids"]) + len(val_bundle["protein_ids"]) + len(test_bundle["protein_ids"])),
        "n_fit_proteins": int(len(fit_bundle["protein_ids"])),
        "n_validation_proteins": int(len(val_bundle["protein_ids"])),
        "n_train_proteins": int(len(train_bundle["protein_ids"])),
        "n_test_proteins": int(len(test_bundle["protein_ids"])),
        "n_fit_rows_used": int(fit_bundle["x"].shape[0]),
        "n_validation_rows_used": int(val_bundle["x"].shape[0]),
        "n_train_rows_used": int(train_bundle["x"].shape[0]),
        "n_test_rows_used": int(test_bundle["x"].shape[0]),
        "train_cap_per_protein": int(MAX_TRAIN_PAIRS_PER_PROTEIN),
        "validation_mean_per_protein_r2": float(val_metrics["mean_per_protein_r2"]),
        "validation_median_per_protein_r2": float(val_metrics["median_per_protein_r2"]),
        "validation_negative_r2_fraction": float(val_metrics["negative_r2_fraction"]),
        "fit_mean_per_protein_r2": float(fit_metrics["mean_per_protein_r2"]),
        "train_mean_per_protein_r2": float(train_metrics["mean_per_protein_r2"]),
        "test_mean_per_protein_r2": float(test_metrics["mean_per_protein_r2"]),
        "test_median_per_protein_r2": float(test_metrics["median_per_protein_r2"]),
        "test_negative_r2_fraction": float(test_metrics["negative_r2_fraction"]),
        "train_pair_weighted_r2": float(train_metrics["pair_weighted_r2"]),
        "test_pair_weighted_r2": float(test_metrics["pair_weighted_r2"]),
        "train_pair_weighted_mae": float(train_metrics["pair_weighted_mae"]),
        "test_pair_weighted_mae": float(test_metrics["pair_weighted_mae"]),
        "train_pair_weighted_rmse": float(train_metrics["pair_weighted_rmse"]),
        "test_pair_weighted_rmse": float(test_metrics["pair_weighted_rmse"]),
        "chosen_restart_index": int(chosen_restart_index),
        "chosen_restart_converged": bool(chosen_restart_converged),
        "chosen_restart_iterations": int(chosen_restart_iterations),
        "final_train_iterations": int(final_train_iterations),
        "n_restarts": int(N_RESTARTS if variant["n_experts"] > 1 else 1),
        "em_max_iters": int(EM_MAX_ITERS if variant["n_experts"] > 1 else 0),
        "em_tol": float(EM_TOL if variant["n_experts"] > 1 else 0.0),
        "train_objective": float(train_objective),
        "mean_gate_entropy_test": float(gate_entropy_test),
        "min_effective_mass_train": float(min_effective_mass),
        "max_effective_mass_train": float(max_effective_mass),
        "failure_reason": failure_reason,
        "elapsed_seconds": float(elapsed_seconds),
    }


def evaluate_baseline_variant(
    variant,
    fit_records,
    val_records,
    train_records,
    test_records,
    fit_bundle,
    val_bundle,
    train_bundle,
    test_bundle,
    train_standardization,
):
    """Fit baseline search/final models and collect outputs."""
    t0 = time.time()
    fit_model = fit_linear_model(fit_bundle["x"], fit_bundle["y"], fit_bundle["weights"])
    fit_pred_search = np.asarray(fit_model.predict(fit_bundle["x"]), dtype=np.float64)
    val_pred_search = np.asarray(fit_model.predict(val_bundle["x"]), dtype=np.float64)

    single_gate_fit = np.ones((len(fit_records), 1), dtype=np.float64)
    single_gate_val = np.ones((len(val_records), 1), dtype=np.float64)
    fit_metrics, fit_rows = build_metrics_rows(
        variant,
        "fit_search",
        fit_records,
        fit_bundle,
        fit_pred_search,
        single_gate_fit,
    )
    val_metrics, val_rows = build_metrics_rows(
        variant,
        "validation_search",
        val_records,
        val_bundle,
        val_pred_search,
        single_gate_val,
    )

    final_model = fit_linear_model(train_bundle["x"], train_bundle["y"], train_bundle["weights"])
    train_pred_final = np.asarray(final_model.predict(train_bundle["x"]), dtype=np.float64)
    test_pred_final = np.asarray(final_model.predict(test_bundle["x"]), dtype=np.float64)

    single_gate_train = np.ones((len(train_records), 1), dtype=np.float64)
    single_gate_test = np.ones((len(test_records), 1), dtype=np.float64)
    train_metrics, train_rows = build_metrics_rows(
        variant,
        "train_final",
        train_records,
        train_bundle,
        train_pred_final,
        single_gate_train,
    )
    test_metrics, test_rows = build_metrics_rows(
        variant,
        "test_final",
        test_records,
        test_bundle,
        test_pred_final,
        single_gate_test,
    )

    summary_row = summarize_variant_row(
        variant,
        elapsed_seconds=float(time.time() - t0),
        fit_bundle=fit_bundle,
        val_bundle=val_bundle,
        train_bundle=train_bundle,
        test_bundle=test_bundle,
        fit_metrics=fit_metrics,
        val_metrics=val_metrics,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        chosen_restart_index=-1,
        selected_restart_model=None,
        fitted_model=None,
    )

    coefficient_rows = build_coefficient_rows(
        variant,
        component_type="baseline",
        model=final_model,
        standardization=train_standardization,
        expert_index=-1,
    )
    seq_rows = []
    seq_rows.extend(build_seq_diag_rows(variant, "fit_search", fit_bundle, fit_pred_search))
    seq_rows.extend(build_seq_diag_rows(variant, "validation_search", val_bundle, val_pred_search))
    seq_rows.extend(build_seq_diag_rows(variant, "train_final", train_bundle, train_pred_final))
    seq_rows.extend(build_seq_diag_rows(variant, "test_final", test_bundle, test_pred_final))

    per_rows = fit_rows + val_rows + train_rows + test_rows
    return {
        "summary_row": summary_row,
        "per_rows": per_rows,
        "seq_rows": seq_rows,
        "coefficient_rows": coefficient_rows,
        "gate_rows": [],
        "responsibility_rows": [],
        "trace_rows": [],
        "gate_grid_rows": [],
    }


def evaluate_moe_variant(
    variant,
    fit_records,
    val_records,
    train_records,
    test_records,
    fit_bundle,
    val_bundle,
    train_bundle,
    test_bundle,
    train_standardization,
):
    """Fit MoE search/final models and collect outputs."""
    t0 = time.time()

    shared_fit_model = None
    shared_train_model = None
    fit_target = fit_bundle["y"]
    val_shared_pred = np.zeros_like(val_bundle["y"], dtype=np.float64)
    fit_shared_pred = np.zeros_like(fit_bundle["y"], dtype=np.float64)
    train_target = train_bundle["y"]
    test_shared_pred = np.zeros_like(test_bundle["y"], dtype=np.float64)
    train_shared_pred = np.zeros_like(train_bundle["y"], dtype=np.float64)

    if variant["uses_shared_global_model"]:
        shared_fit_model = fit_linear_model(fit_bundle["x"], fit_bundle["y"], fit_bundle["weights"])
        fit_shared_pred = np.asarray(shared_fit_model.predict(fit_bundle["x"]), dtype=np.float64)
        val_shared_pred = np.asarray(shared_fit_model.predict(val_bundle["x"]), dtype=np.float64)
        fit_target = fit_bundle["y"] - fit_shared_pred

        shared_train_model = fit_linear_model(
            train_bundle["x"],
            train_bundle["y"],
            train_bundle["weights"],
        )
        train_shared_pred = np.asarray(shared_train_model.predict(train_bundle["x"]), dtype=np.float64)
        test_shared_pred = np.asarray(shared_train_model.predict(test_bundle["x"]), dtype=np.float64)
        train_target = train_bundle["y"] - train_shared_pred

    best_search = None
    trace_rows = []

    for restart_index in range(N_RESTARTS):
        search_model = fit_gaussian_moe(
            fit_bundle,
            fit_target,
            n_experts=variant["n_experts"],
            restart_seed=restart_index,
            variant_name=variant["name"],
            stage_name="fit_search",
        )
        trace_rows.extend(search_model["trace_rows"])
        if search_model["failed"]:
            continue

        fit_pred_target, fit_gate_predict = predict_moe_target(search_model, fit_bundle)
        val_pred_target, val_gate_predict = predict_moe_target(search_model, val_bundle)
        fit_pred = fit_pred_target + fit_shared_pred
        val_pred = val_pred_target + val_shared_pred

        fit_metrics, fit_rows = build_metrics_rows(
            variant,
            "fit_search",
            fit_records,
            fit_bundle,
            fit_pred,
            fit_gate_predict,
        )
        val_metrics, val_rows = build_metrics_rows(
            variant,
            "validation_search",
            val_records,
            val_bundle,
            val_pred,
            val_gate_predict,
        )

        candidate_score = float(val_metrics["mean_per_protein_r2"])
        if best_search is None or candidate_score > best_search["score"]:
            best_search = {
                "score": candidate_score,
                "restart_index": restart_index,
                "search_model": search_model,
                "fit_pred": fit_pred,
                "val_pred": val_pred,
                "fit_gate_predict": fit_gate_predict,
                "val_gate_predict": val_gate_predict,
                "fit_metrics": fit_metrics,
                "val_metrics": val_metrics,
                "fit_rows": fit_rows,
                "val_rows": val_rows,
            }

    if best_search is None:
        raise RuntimeError(f"All EM restarts failed for {variant['name']}")

    final_model = fit_gaussian_moe(
        train_bundle,
        train_target,
        n_experts=variant["n_experts"],
        restart_seed=best_search["restart_index"],
        variant_name=variant["name"],
        stage_name="train_final",
    )
    trace_rows.extend(final_model["trace_rows"])
    if final_model["failed"]:
        raise RuntimeError(
            f"Final EM refit failed for {variant['name']}: {final_model['failure_reason']}"
        )

    train_pred_target, train_gate_predict = predict_moe_target(final_model, train_bundle)
    test_pred_target, test_gate_predict = predict_moe_target(final_model, test_bundle)
    train_pred = train_pred_target + train_shared_pred
    test_pred = test_pred_target + test_shared_pred

    train_metrics, train_rows = build_metrics_rows(
        variant,
        "train_final",
        train_records,
        train_bundle,
        train_pred,
        train_gate_predict,
    )
    test_metrics, test_rows = build_metrics_rows(
        variant,
        "test_final",
        test_records,
        test_bundle,
        test_pred,
        test_gate_predict,
    )

    summary_row = summarize_variant_row(
        variant,
        elapsed_seconds=float(time.time() - t0),
        fit_bundle=fit_bundle,
        val_bundle=val_bundle,
        train_bundle=train_bundle,
        test_bundle=test_bundle,
        fit_metrics=best_search["fit_metrics"],
        val_metrics=best_search["val_metrics"],
        train_metrics=train_metrics,
        test_metrics=test_metrics,
        chosen_restart_index=best_search["restart_index"],
        selected_restart_model=best_search["search_model"],
        fitted_model=final_model,
    )

    coefficient_rows = []
    if variant["uses_shared_global_model"]:
        coefficient_rows.extend(
            build_coefficient_rows(
                variant,
                component_type="shared_global",
                model=shared_train_model,
                standardization=train_standardization,
                expert_index=-1,
            )
        )
        component_type = "expert_residual"
    else:
        component_type = "expert"

    for expert_index, expert_model in enumerate(final_model["expert_models"]):
        coefficient_rows.extend(
            build_coefficient_rows(
                variant,
                component_type=component_type,
                model=expert_model,
                standardization=train_standardization,
                expert_index=expert_index,
            )
        )

    gate_rows = build_gate_parameter_rows(variant, final_model, train_bundle)
    gate_grid_rows = build_gate_grid_rows(
        variant,
        final_model,
        min_n=min(int(np.min(train_bundle["n_residues"])), int(np.min(test_bundle["n_residues"]))),
        max_n=max(int(np.max(train_bundle["n_residues"])), int(np.max(test_bundle["n_residues"]))),
    )

    seq_rows = []
    seq_rows.extend(build_seq_diag_rows(variant, "fit_search", fit_bundle, best_search["fit_pred"]))
    seq_rows.extend(build_seq_diag_rows(variant, "validation_search", val_bundle, best_search["val_pred"]))
    seq_rows.extend(build_seq_diag_rows(variant, "train_final", train_bundle, train_pred))
    seq_rows.extend(build_seq_diag_rows(variant, "test_final", test_bundle, test_pred))

    responsibility_rows = []
    responsibility_rows.extend(
        build_responsibility_rows(
            variant,
            "fit_search",
            "posterior_training",
            fit_records,
            best_search["search_model"]["posterior_train"],
        )
    )
    responsibility_rows.extend(
        build_responsibility_rows(
            variant,
            "fit_search",
            "gate_predict",
            fit_records,
            best_search["fit_gate_predict"],
        )
    )
    responsibility_rows.extend(
        build_responsibility_rows(
            variant,
            "validation_search",
            "gate_predict",
            val_records,
            best_search["val_gate_predict"],
        )
    )
    responsibility_rows.extend(
        build_responsibility_rows(
            variant,
            "train_final",
            "posterior_training",
            train_records,
            final_model["posterior_train"],
        )
    )
    responsibility_rows.extend(
        build_responsibility_rows(
            variant,
            "train_final",
            "gate_predict",
            train_records,
            train_gate_predict,
        )
    )
    responsibility_rows.extend(
        build_responsibility_rows(
            variant,
            "test_final",
            "gate_predict",
            test_records,
            test_gate_predict,
        )
    )

    per_rows = best_search["fit_rows"] + best_search["val_rows"] + train_rows + test_rows
    return {
        "summary_row": summary_row,
        "per_rows": per_rows,
        "seq_rows": seq_rows,
        "coefficient_rows": coefficient_rows,
        "gate_rows": gate_rows,
        "responsibility_rows": responsibility_rows,
        "trace_rows": trace_rows,
        "gate_grid_rows": gate_grid_rows,
    }


def write_summary_text(
    path,
    elapsed_seconds,
    all_proteins,
    protein_data,
    skipped,
    fit_records,
    val_records,
    train_records,
    test_records,
    summary_rows,
    per_rows,
):
    """Write a compact human-readable summary."""
    ranked_by_test = sorted(
        summary_rows,
        key=lambda row: float(row["test_mean_per_protein_r2"]),
        reverse=True,
    )
    ranked_by_validation = sorted(
        summary_rows,
        key=lambda row: float(row["validation_mean_per_protein_r2"]),
        reverse=True,
    )

    with path.open("w") as handle:
        handle.write(f"{TITLE}\n")
        handle.write("=" * len(TITLE) + "\n\n")
        handle.write(f"Output CSV dir:   {CSV_DIR.resolve()}\n")
        handle.write(f"Layer:            {LAYER}\n")
        handle.write("Representation:   lower triangle only, pb[i,j,:] only\n")
        handle.write("Gate feature:     log(n_residues)\n")
        handle.write("Gate family:      1D Gaussian\n")
        handle.write("Objective:        protein-equal generalized EM\n")
        handle.write(
            f"Train cap:        {MAX_TRAIN_PAIRS_PER_PROTEIN:,} pairs per fit/train protein\n"
        )
        handle.write(
            f"EM defaults:      restarts={N_RESTARTS}, max_iters={EM_MAX_ITERS}, tol={EM_TOL}\n"
        )
        handle.write(f"Elapsed seconds:  {elapsed_seconds:.1f}\n\n")

        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(protein_data)}\n")
        handle.write(f"Proteins skipped:    {len(skipped)}\n")
        handle.write(f"Fit proteins:        {len(fit_records)}\n")
        handle.write(f"Validation proteins: {len(val_records)}\n")
        handle.write(f"Train proteins:      {len(train_records)}\n")
        handle.write(f"Test proteins:       {len(test_records)}\n\n")

        handle.write("Ranked by validation mean per-protein R2:\n")
        for row in ranked_by_validation:
            handle.write(
                f"  {row['variant']}: val={float(row['validation_mean_per_protein_r2']):.4f}, "
                f"test={float(row['test_mean_per_protein_r2']):.4f}, "
                f"restart={int(row['chosen_restart_index'])}\n"
            )
        handle.write("\nRanked by test mean per-protein R2:\n")
        for row in ranked_by_test:
            collapse_note = ""
            if float(row["min_effective_mass_train"]) < 0.75 and int(row["n_experts"]) > 1:
                collapse_note = " [possible expert collapse]"
            handle.write(
                f"  {row['variant']}: test={float(row['test_mean_per_protein_r2']):.4f}, "
                f"val={float(row['validation_mean_per_protein_r2']):.4f}, "
                f"min_mass={float(row['min_effective_mass_train']):.3f}{collapse_note}\n"
            )

        if ranked_by_test:
            best_variant = ranked_by_test[0]["variant"]
            best_test_rows = [
                row
                for row in per_rows
                if row["variant"] == best_variant and row["split"] == "test_final"
            ]
            best_test_rows = sorted(
                best_test_rows,
                key=lambda row: (
                    math.inf
                    if math.isnan(float(row["per_protein_r2"]))
                    else float(row["per_protein_r2"])
                ),
            )

            handle.write(f"\nBest variant by test R2: {best_variant}\n")
            handle.write("Worst 10 test proteins:\n")
            for row in best_test_rows[:10]:
                handle.write(
                    f"  {row['protein_id']}: R2={float(row['per_protein_r2']):.4f}, "
                    f"gate_entropy={float(row['gate_entropy']):.4f}, "
                    f"top_expert={int(row['top_expert'])}\n"
                )

            handle.write("\nBest 10 test proteins:\n")
            for row in reversed(best_test_rows[-10:]):
                handle.write(
                    f"  {row['protein_id']}: R2={float(row['per_protein_r2']):.4f}, "
                    f"gate_entropy={float(row['gate_entropy']):.4f}, "
                    f"top_expert={int(row['top_expert'])}\n"
                )

        if skipped:
            handle.write("\nSkipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")


def run_synthetic_sanity_checks():
    """Simple local sanity checks for the Gaussian gate implementation."""
    rng = np.random.default_rng(7)
    row_counts = np.array([40, 40, 40, 40, 40, 40], dtype=np.int32)
    offsets = []
    start = 0
    for n_rows in row_counts:
        end = start + int(n_rows)
        offsets.append((start, end))
        start = end

    x = rng.normal(size=(int(np.sum(row_counts)), 4)).astype(np.float32)
    y = np.empty((x.shape[0],), dtype=np.float64)
    seq_sep = np.ones((x.shape[0],), dtype=np.int32)
    n_residues = np.array([50, 60, 70, 300, 360, 420], dtype=np.int32)
    log_n = np.log(n_residues.astype(np.float64))
    small_coef = np.array([1.0, -0.5, 0.3, 0.0], dtype=np.float64)
    large_coef = np.array([-0.8, 0.6, 0.2, 0.4], dtype=np.float64)

    for protein_index, (start, end) in enumerate(offsets):
        coef = small_coef if protein_index < 3 else large_coef
        y[start:end] = x[start:end, :] @ coef + rng.normal(scale=0.1, size=end - start)

    bundle = {
        "x": x,
        "y": y,
        "seq_sep": seq_sep,
        "offsets": offsets,
        "row_counts": row_counts,
        "weights": compute_pair_weights(row_counts.tolist()),
        "protein_ids": [f"synthetic_{idx}" for idx in range(len(row_counts))],
        "n_residues": n_residues,
        "log_n_residues": log_n,
    }
    standardization = standardize_features(bundle, [])
    model = fit_gaussian_moe(
        bundle,
        bundle["y"],
        n_experts=2,
        restart_seed=0,
        variant_name="synthetic_moe",
        stage_name="synthetic",
    )
    _, gate_predict = predict_moe_target(model, bundle)
    row_sums = np.sum(gate_predict, axis=1)
    shared_pred = np.array([1.0, -2.0, 0.5], dtype=np.float64)
    zero_residual = np.zeros_like(shared_pred)

    return {
        "gate_row_sum_min": float(np.min(row_sums)),
        "gate_row_sum_max": float(np.max(row_sums)),
        "small_mean_weight_expert0": float(np.mean(gate_predict[:3, 0])),
        "large_mean_weight_expert1": float(np.mean(gate_predict[3:, 1])),
        "shared_plus_zero_residual_matches": bool(
            np.allclose(shared_pred + zero_residual, shared_pred)
        ),
        "standardization_scale_min": float(np.min(standardization["scale"])),
    }


def main():
    t_start = time.time()
    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))
    print(f"Output CSV dir: {csv_dir}")
    print(f"Train cap:      {MAX_TRAIN_PAIRS_PER_PROTEIN:,} pairs per fit/train protein")
    print(f"Variants:       {len(VARIANTS)}")
    print(f"EM defaults:    restarts={N_RESTARTS}, max_iters={EM_MAX_ITERS}, tol={EM_TOL}")

    all_proteins, protein_data, skipped = load_protein_records()
    if not protein_data:
        print("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    fit_records, val_records, train_records, test_records = split_records(protein_data)
    split_rows = build_split_rows(fit_records, val_records, train_records, test_records)
    write_csv(
        csv_dir / SPLIT_CSV_NAME,
        ["protein_id", "split_group", "split_role", "n_residues", "n_pairs"],
        split_rows,
    )

    fit_bundle = build_lower_triangle_bundle(
        fit_records,
        sample_cap=MAX_TRAIN_PAIRS_PER_PROTEIN,
        sample_offset=0,
    )
    val_bundle = build_lower_triangle_bundle(val_records, sample_cap=None, sample_offset=0)
    train_bundle = build_lower_triangle_bundle(
        train_records,
        sample_cap=MAX_TRAIN_PAIRS_PER_PROTEIN,
        sample_offset=1000,
    )
    test_bundle = build_lower_triangle_bundle(test_records, sample_cap=None, sample_offset=0)

    fit_standardization = standardize_features(fit_bundle, [val_bundle])
    train_standardization = standardize_features(train_bundle, [test_bundle])

    summary_rows = []
    per_protein_rows = []
    seq_diag_rows = []
    coefficient_rows = []
    gate_rows = []
    responsibility_rows = []
    trace_rows = []
    gate_grid_rows = []

    for variant in VARIANTS:
        print(f"\nRunning variant: {variant['name']}")
        if variant["model_type"] == MODEL_TYPE_BASELINE:
            result = evaluate_baseline_variant(
                variant,
                fit_records,
                val_records,
                train_records,
                test_records,
                fit_bundle,
                val_bundle,
                train_bundle,
                test_bundle,
                train_standardization,
            )
        else:
            result = evaluate_moe_variant(
                variant,
                fit_records,
                val_records,
                train_records,
                test_records,
                fit_bundle,
                val_bundle,
                train_bundle,
                test_bundle,
                train_standardization,
            )

        summary_rows.append(result["summary_row"])
        per_protein_rows.extend(result["per_rows"])
        seq_diag_rows.extend(result["seq_rows"])
        coefficient_rows.extend(result["coefficient_rows"])
        gate_rows.extend(result["gate_rows"])
        responsibility_rows.extend(result["responsibility_rows"])
        trace_rows.extend(result["trace_rows"])
        gate_grid_rows.extend(result["gate_grid_rows"])

        print(
            f"  validation mean per-protein R2={float(result['summary_row']['validation_mean_per_protein_r2']):.4f}, "
            f"test mean per-protein R2={float(result['summary_row']['test_mean_per_protein_r2']):.4f}"
        )

    write_csv(csv_dir / SUMMARY_CSV_NAME, list(summary_rows[0].keys()), summary_rows)
    write_csv(
        csv_dir / PER_PROTEIN_CSV_NAME,
        [
            "variant",
            "model_type",
            "split",
            "protein_id",
            "n_residues",
            "log_n_residues",
            "n_pairs_used",
            "per_protein_r2",
            "per_protein_mae",
            "per_protein_rmse",
            "mean_true_distance",
            "mean_predicted_distance",
            "mean_error_pred_minus_true",
            "gate_entropy",
            "top_expert",
            "top_expert_weight",
        ],
        per_protein_rows,
    )
    write_csv(
        csv_dir / SEQ_DIAGNOSTICS_CSV_NAME,
        [
            "variant",
            "model_type",
            "split",
            "seq_bucket",
            "n_pairs",
            "mean_true_distance",
            "mean_predicted_distance",
            "mean_error_pred_minus_true",
            "r2",
            "fraction_pred_lt_3A",
        ],
        seq_diag_rows,
    )
    write_csv(
        csv_dir / COEFFICIENTS_CSV_NAME,
        [
            "variant",
            "model_type",
            "component_type",
            "expert_index",
            "feature_index",
            "feature_name",
            "coefficient_raw",
            "intercept_raw",
        ],
        coefficient_rows,
    )
    if gate_rows:
        write_csv(
            csv_dir / GATE_PARAMS_CSV_NAME,
            [
                "variant",
                "model_type",
                "expert_index",
                "gate_weight",
                "gate_mean_log_n",
                "gate_variance_log_n",
                "gate_std_log_n",
                "expert_noise_variance",
                "posterior_effective_protein_mass_train",
                "posterior_average_responsibility_train",
                "gate_predict_effective_protein_mass_train",
                "gate_predict_average_weight_train",
            ],
            gate_rows,
        )
    if responsibility_rows:
        write_csv(
            csv_dir / RESPONSIBILITIES_CSV_NAME,
            [
                "variant",
                "model_type",
                "split",
                "responsibility_source",
                "protein_id",
                "n_residues",
                "log_n_residues",
                "expert_index",
                "responsibility",
                "gate_entropy",
                "top_expert",
            ],
            responsibility_rows,
        )
    if trace_rows:
        write_csv(
            csv_dir / EM_TRACE_CSV_NAME,
            [
                "variant",
                "stage",
                "restart_index",
                "iteration",
                "objective",
                "max_responsibility_change",
                "mean_posterior_entropy",
                "min_effective_mass",
                "max_effective_mass",
                "converged",
                "failed",
                "failure_reason",
            ],
            trace_rows,
        )
    if gate_grid_rows:
        write_csv(
            csv_dir / GATE_GRID_CSV_NAME,
            [
                "variant",
                "model_type",
                "grid_index",
                "n_residues_grid",
                "log_n_residues_grid",
                "expert_index",
                "gate_weight",
            ],
            gate_grid_rows,
        )

    write_summary_text(
        csv_dir / SUMMARY_TXT_NAME,
        elapsed_seconds=float(time.time() - t_start),
        all_proteins=all_proteins,
        protein_data=protein_data,
        skipped=skipped,
        fit_records=fit_records,
        val_records=val_records,
        train_records=train_records,
        test_records=test_records,
        summary_rows=summary_rows,
        per_rows=per_protein_rows,
    )

    print(f"\nSaved summary CSV -> {csv_dir / SUMMARY_CSV_NAME}")
    print(f"Saved per-protein CSV -> {csv_dir / PER_PROTEIN_CSV_NAME}")
    print(f"Saved split CSV -> {csv_dir / SPLIT_CSV_NAME}")
    print(f"Saved seq-sep CSV -> {csv_dir / SEQ_DIAGNOSTICS_CSV_NAME}")
    print(f"Saved coefficients CSV -> {csv_dir / COEFFICIENTS_CSV_NAME}")
    if gate_rows:
        print(f"Saved gate params CSV -> {csv_dir / GATE_PARAMS_CSV_NAME}")
    if responsibility_rows:
        print(f"Saved responsibilities CSV -> {csv_dir / RESPONSIBILITIES_CSV_NAME}")
    if trace_rows:
        print(f"Saved EM trace CSV -> {csv_dir / EM_TRACE_CSV_NAME}")
    if gate_grid_rows:
        print(f"Saved gate-grid CSV -> {csv_dir / GATE_GRID_CSV_NAME}")
    print(f"Saved summary text -> {csv_dir / SUMMARY_TXT_NAME}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Fit one transition/bundle shard for the biological-context update probe."""

from __future__ import annotations

import gc
import json
import math
import os
import time
import traceback
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score, average_precision_score, balanced_accuracy_score,
    f1_score, mean_absolute_error, mean_squared_error, precision_score,
    r2_score, recall_score, roc_auc_score,
)
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor, export_graphviz, export_text,
)

from local_pair_biological_context_common import (
    ALL_FEATURE_SPECS, BOOTSTRAP_REPLICATES, BUNDLE_FAMILIES, BUNDLES,
    BUNDLE_WIDTHS, CLASS_THRESHOLD, FAMILY_ORDER, HIST_GB_MAX_ITER,
    LOGISTIC_C_GRID, OUTPUT_NAME, RANDOM_SEED, SOURCE_VARIANT,
    TRIANGLE_FEATURE_NAMES, assemble_bundle, balanced_binary_weights,
    bootstrap_assignments, build_all_feature_families, build_offsets,
    bundle_indices, class_labels, correlation_label, decode_distance_matrix,
    decode_selected_distances, extract_pdb_residue_names, feature_schema,
    load_distance_coefficients, load_split_records, logistic_validation_assignments,
    log, metadata_constants, now_timestamp, pair_block_path, pdb_path,
    protein_equal_scaler, protein_equal_weights, safe_pearson, standardize,
    subset_offsets, task_spec, triangle_distance_correlations, write_csv_atomic,
    write_json_atomic, write_text_atomic, local_pair_indices,
)


SKIPPED_FIELDS = (
    "layer_from", "layer_to", "bundle", "split", "source", "protein_id",
    "reason", "message",
)
MAGNITUDE_BINS = (
    ("lt_neg4", -np.inf, -4.0, False),
    ("neg4_neg2", -4.0, -2.0, False),
    ("neg2_neg1", -2.0, -1.0, False),
    ("neg1_neg0_5", -1.0, -0.5, False),
    ("neg0_5_neg0_25", -0.5, -0.25, True),
    ("neg0_25_0", -0.25, 0.0, True),
    ("zero_0_25", 0.0, 0.25, True),
    ("0_25_0_5", 0.25, 0.5, True),
    ("0_5_1", 0.5, 1.0, False),
    ("1_2", 1.0, 2.0, False),
    ("2_4", 2.0, 4.0, False),
    ("gt4", 4.0, np.inf, False),
)


def finite_mean(values):
    values = [float(value) for value in values if math.isfinite(float(value))]
    return float(np.mean(values)) if values else float("nan")


def safe_roc_auc(y, score):
    return float(roc_auc_score(y, score)) if np.unique(y).size == 2 else float("nan")


def safe_average_precision(y, score):
    return float(average_precision_score(y, score)) if np.unique(y).size == 2 else float("nan")


def regression_metric_dict(y, pred):
    y, pred = np.asarray(y, np.float64), np.asarray(pred, np.float64)
    if y.size == 0:
        return {key: float("nan") for key in ("r2", "mae", "rmse", "pearson_r", "bias", "sign_accuracy")}
    nonzero = y != 0.0
    return {
        "r2": float(r2_score(y, pred)) if y.size > 1 and np.var(y) > 0 else float("nan"),
        "mae": float(mean_absolute_error(y, pred)),
        "rmse": float(np.sqrt(mean_squared_error(y, pred))),
        "pearson_r": safe_pearson(y, pred),
        "bias": float(np.mean(pred - y)),
        "sign_accuracy": float(np.mean(np.sign(pred[nonzero]) == np.sign(y[nonzero]))) if np.any(nonzero) else float("nan"),
    }


def classification_metric_dict(y, pred, score):
    y, pred, score = np.asarray(y), np.asarray(pred), np.asarray(score, np.float64)
    if y.size == 0:
        return {key: float("nan") for key in (
            "accuracy", "balanced_accuracy", "shrink_precision", "shrink_recall",
            "shrink_f1", "auroc", "average_precision",
        )}
    return {
        "accuracy": float(accuracy_score(y, pred)),
        "balanced_accuracy": float(balanced_accuracy_score(y, pred)) if np.unique(y).size == 2 else float("nan"),
        "shrink_precision": float(precision_score(y, pred, pos_label=1, zero_division=0)),
        "shrink_recall": float(recall_score(y, pred, pos_label=1, zero_division=0)),
        "shrink_f1": float(f1_score(y, pred, pos_label=1, zero_division=0)),
        "auroc": safe_roc_auc(y, score),
        "average_precision": safe_average_precision(y, score),
    }


def estimator_score(model, x):
    if hasattr(model, "predict_proba"):
        classes = list(model.classes_)
        return model.predict_proba(x)[:, classes.index(1)]
    value = model.decision_function(x)
    return np.asarray(value, dtype=np.float64)


def evaluate_model(layer_from, layer_to, bundle, target_name, estimator_name,
                   target_by_split, prediction_by_split, score_by_split,
                   offsets_by_split, total_pair_counts):
    summary_rows, per_rows = [], []
    for split in ("train", "test"):
        target = target_by_split[split]
        prediction = prediction_by_split[split]
        score = score_by_split.get(split)
        offsets = offsets_by_split[split]
        protein_metrics = []
        if target_name == "classification":
            pooled = classification_metric_dict(target, prediction, score)
        else:
            pooled = regression_metric_dict(target, prediction)
        for record, start, end in offsets:
            y_p, pred_p = target[start:end], prediction[start:end]
            if target_name == "classification":
                metrics = classification_metric_dict(y_p, pred_p, score[start:end])
                counts = {
                    "n_shrink": int(np.count_nonzero(y_p == 1)),
                    "n_expand": int(np.count_nonzero(y_p == 0)),
                }
            else:
                metrics = regression_metric_dict(y_p, pred_p)
                counts = {}
            protein_metrics.append(metrics)
            per_rows.append({
                "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
                "target": target_name, "estimator": estimator_name, "split": split,
                "source": record["source"], "protein_id": record["protein_id"],
                "n_residues": record["n_residues"], "n_pairs": end - start,
                **counts, **metrics,
            })
        row = {
            "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
            "feature_dim": BUNDLE_WIDTHS[bundle], "target": target_name,
            "estimator": estimator_name, "split": split, "n_proteins": len(offsets),
            "n_pairs": target.size, "coverage_of_all_local_pairs": target.size / max(total_pair_counts[split], 1),
        }
        if target_name == "classification":
            row.update({
                "n_shrink": int(np.count_nonzero(target == 1)),
                "n_expand": int(np.count_nonzero(target == 0)),
            })
        row.update({f"pair_{key}": value for key, value in pooled.items()})
        metric_names = pooled.keys()
        for key in metric_names:
            values = [metrics[key] for metrics in protein_metrics]
            row[f"mean_per_protein_{key}"] = finite_mean(values)
            row[f"n_valid_proteins_{key}"] = sum(math.isfinite(float(value)) for value in values)
        summary_rows.append(row)
    return summary_rows, per_rows


def filtered_classification_arrays(labels, arrays, offsets):
    row_parts, new_offsets, position = [], [], 0
    for record, start, end in offsets:
        local = np.flatnonzero(labels[start:end] >= 0) + start
        row_parts.append(local)
        new_offsets.append((record, position, position + local.size))
        position += local.size
    rows = np.concatenate(row_parts) if row_parts else np.empty(0, np.int64)
    return [np.asarray(array)[rows] for array in arrays], new_offsets, rows


def magnitude_rows(layer_from, layer_to, bundle, estimator_name, target_name,
                   split, true_delta, prediction, score, offsets):
    rows = []
    signed_masks = (
        ("lt_neg4", true_delta < -4.0, False),
        ("neg4_neg2", (true_delta >= -4.0) & (true_delta < -2.0), False),
        ("neg2_neg1", (true_delta >= -2.0) & (true_delta < -1.0), False),
        ("neg1_neg0_5", (true_delta >= -1.0) & (true_delta < -0.5), False),
        ("neg0_5_neg0_25", (true_delta >= -0.5) & (true_delta < -0.25), True),
        ("neg0_25_0", (true_delta >= -0.25) & (true_delta < 0.0), True),
        ("zero_0_25", (true_delta > 0.0) & (true_delta <= 0.25), True),
        ("0_25_0_5", (true_delta > 0.25) & (true_delta <= 0.5), True),
        ("0_5_1", (true_delta > 0.5) & (true_delta <= 1.0), False),
        ("1_2", (true_delta > 1.0) & (true_delta <= 2.0), False),
        ("2_4", (true_delta > 2.0) & (true_delta <= 4.0), False),
        ("gt4", true_delta > 4.0, False),
    )
    masks = [(name, mask, excluded, "signed") for name, mask, excluded in signed_masks]
    masks.append(("exact_zero", true_delta == 0.0, True, "audit"))
    for threshold in (0.5, 1.0, 2.0, 4.0):
        masks.append((f"abs_gt_{str(threshold).replace('.', '_')}", np.abs(true_delta) > threshold, False, "cumulative"))
    protein_ids = np.empty(true_delta.size, dtype=object)
    for record, start, end in offsets:
        protein_ids[start:end] = record["protein_id"]
    for name, mask, excluded, bin_type in masks:
        n = int(np.count_nonzero(mask))
        base = {
            "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
            "target": target_name, "estimator": estimator_name, "split": split,
            "bin_name": name, "bin_type": bin_type,
            "excluded_from_classifier_training": bool(excluded), "n_pairs": n,
            "n_proteins": len(set(protein_ids[mask])) if n else 0,
        }
        if not n:
            rows.append(base)
            continue
        delta = true_delta[mask]
        if name == "exact_zero":
            base.update({"sign_accuracy": float("nan"), "note": "sign undefined for exact-zero target"})
        elif target_name == "regression":
            pooled_metrics = regression_metric_dict(delta, prediction[mask])
            base.update(pooled_metrics)
        else:
            y = (delta < 0).astype(np.int8)
            pooled_metrics = classification_metric_dict(y, prediction[mask], score[mask])
            base.update(pooled_metrics)
            base["mean_score"] = float(np.mean(score[mask]))
        if name != "exact_zero":
            per_protein_metrics = []
            for _record, start, end in offsets:
                local_mask = mask[start:end]
                if not np.any(local_mask):
                    continue
                if target_name == "regression":
                    metrics = regression_metric_dict(
                        true_delta[start:end][local_mask], prediction[start:end][local_mask]
                    )
                else:
                    local_delta = true_delta[start:end][local_mask]
                    metrics = classification_metric_dict(
                        (local_delta < 0).astype(np.int8),
                        prediction[start:end][local_mask], score[start:end][local_mask],
                    )
                per_protein_metrics.append(metrics)
            for metric_name in pooled_metrics:
                base[f"mean_per_protein_{metric_name}"] = finite_mean(
                    metrics[metric_name] for metrics in per_protein_metrics
                )
                base[f"n_valid_proteins_{metric_name}"] = sum(
                    math.isfinite(float(metrics[metric_name])) for metrics in per_protein_metrics
                )
        rows.append(base)
    return rows


def feature_transform_rows(layer_from, layer_to, bundle, x_train, mean, scale):
    rows = []
    for index, schema in enumerate(feature_schema(bundle)):
        rows.append({
            "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
            "feature_index": index, "feature_name": schema["feature_name"],
            "training_mean": mean[index], "training_scale": scale[index],
            "training_minimum": np.min(x_train[:, index]),
            "training_maximum": np.max(x_train[:, index]), "missing_policy": "none",
        })
    return rows


def linear_coefficient_rows(layer_from, layer_to, bundle, target_name,
                            estimator_name, model, mean, scale):
    coefficients = np.ravel(np.asarray(model.coef_, dtype=np.float64))
    standardized_intercept = float(np.ravel(np.asarray(model.intercept_))[0])
    raw_coefficients = coefficients / scale
    raw_intercept = standardized_intercept - float(np.dot(raw_coefficients, mean))
    rows = [{
        "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
        "target": target_name, "estimator": estimator_name, "feature_index": -1,
        "feature_name": "intercept", "feature_family": "intercept",
        "coefficient_standardized": standardized_intercept,
        "coefficient_raw_scale": raw_intercept,
    }]
    schema = feature_schema(bundle)
    for index, coefficient in enumerate(coefficients):
        rows.append({
            "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
            "target": target_name, "estimator": estimator_name, "feature_index": index,
            "feature_name": schema[index]["feature_name"],
            "feature_family": schema[index]["feature_family"],
            "coefficient_standardized": coefficient,
            "coefficient_raw_scale": raw_coefficients[index],
        })
    return rows


def atomic_joblib_dump(path, model, feature_names):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    joblib.dump({"model": model, "feature_names": list(feature_names)}, temporary)
    os.replace(temporary, path)


def tree_artifacts(layer_from, layer_to, bundle, target_name, estimator_name,
                   model, x_by_split, target_by_split, prediction_by_split,
                   score_by_split, offsets_by_split, correlation_lookup, output_dir):
    names = [row["feature_name"] for row in feature_schema(bundle)]
    structure_dir = output_dir / "tree_structures"
    structure_dir.mkdir(parents=True, exist_ok=True)
    write_text_atomic(structure_dir / f"{target_name}_{estimator_name}.txt", export_text(model, feature_names=names))
    export_graphviz(model, out_file=str(structure_dir / f"{target_name}_{estimator_name}.dot"), feature_names=names, filled=True)
    tree = model.tree_
    parent = np.full(tree.node_count, -1, dtype=np.int64)
    branch = np.full(tree.node_count, "root", dtype=object)
    depth = np.zeros(tree.node_count, dtype=np.int64)
    for node in range(tree.node_count):
        for child, label in ((tree.children_left[node], "left"), (tree.children_right[node], "right")):
            if child >= 0:
                parent[child] = node
                branch[child] = label
                depth[child] = depth[node] + 1
    node_rows = []
    for node in range(tree.node_count):
        feature_index = int(tree.feature[node])
        value = np.ravel(tree.value[node])
        node_rows.append({
            "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
            "target": target_name, "estimator": estimator_name, "node_id": node,
            "parent_id": int(parent[node]), "branch_from_parent": branch[node],
            "depth": int(depth[node]), "is_leaf": tree.children_left[node] < 0,
            "feature_index": feature_index,
            "feature_name": names[feature_index] if feature_index >= 0 else "leaf",
            "threshold": float(tree.threshold[node]) if feature_index >= 0 else float("nan"),
            "impurity": float(tree.impurity[node]), "n_node_samples": int(tree.n_node_samples[node]),
            "weighted_n_node_samples": float(tree.weighted_n_node_samples[node]),
            "prediction": int(model.classes_[np.argmax(value)]) if target_name == "classification" else float(value[0]),
        })
    leaf_ids = [node for node in range(tree.node_count) if tree.children_left[node] < 0]
    paths = {}
    for leaf in leaf_ids:
        clauses, node = [], leaf
        while parent[node] >= 0:
            p = int(parent[node])
            sign = "<=" if branch[node] == "left" else ">"
            clauses.append(f"{names[tree.feature[p]]} {sign} {tree.threshold[p]:.9g}")
            node = p
        paths[leaf] = list(reversed(clauses))
    rule_rows = []
    for split in ("train", "test"):
        leaves = model.apply(x_by_split[split])
        protein_ids = np.empty(leaves.size, dtype=object)
        for record, start, end in offsets_by_split[split]:
            protein_ids[start:end] = record["protein_id"]
        for leaf in leaf_ids:
            mask = leaves == leaf
            pressure_names = [name for name in TRIANGLE_FEATURE_NAMES if any(name in clause for clause in paths[leaf])]
            correlations = {name: correlation_lookup.get((split, name), float("nan")) for name in pressure_names}
            labels = [correlation_label(value) for value in correlations.values()]
            worst = "high" if "high" in labels else "moderate" if "moderate" in labels else "weak" if labels else "none"
            row = {
                "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
                "target": target_name, "estimator": estimator_name, "split": split,
                "leaf_id": leaf, "rule_path": " AND ".join(paths[leaf]) or "root",
                "n_pairs": int(np.count_nonzero(mask)),
                "n_proteins": len(set(protein_ids[mask])) if np.any(mask) else 0,
                "pressure_features": json.dumps(pressure_names),
                "pressure_distance_correlations": json.dumps(correlations, sort_keys=True),
                "distance_collinearity_label": worst,
                "distance_collinearity_warning": worst == "high",
            }
            if np.any(mask):
                if target_name == "classification":
                    row.update(classification_metric_dict(
                        target_by_split[split][mask], prediction_by_split[split][mask], score_by_split[split][mask]
                    ))
                    row["shrink_prevalence"] = float(np.mean(target_by_split[split][mask]))
                else:
                    row.update(regression_metric_dict(target_by_split[split][mask], prediction_by_split[split][mask]))
                    row["mean_target"] = float(np.mean(target_by_split[split][mask]))
            rule_rows.append(row)
    return node_rows, rule_rows


def bootstrap_model_metrics(layer_from, layer_to, bundle, target_name, estimator_name,
                            per_protein_rows, bootstrap_counts):
    test_rows = [row for row in per_protein_rows if row["split"] == "test"]
    metrics = (
        ("balanced_accuracy", "mean_per_protein") if target_name == "classification" else ("r2", "mean_per_protein"),
        ("accuracy", "mean_per_protein") if target_name == "classification" else ("mae", "mean_per_protein"),
    )
    rows = []
    for replicate in range(bootstrap_counts.shape[0]):
        counts = bootstrap_counts[replicate]
        for metric, scope in metrics:
            values, weights = [], []
            for index, row in enumerate(test_rows):
                value = float(row.get(metric, float("nan")))
                if counts[index] > 0 and math.isfinite(value):
                    values.append(value)
                    weights.append(int(counts[index]))
            rows.append({
                "layer_from": layer_from, "layer_to": layer_to,
                "bootstrap_replicate": replicate, "bundle": bundle,
                "target": target_name, "estimator": estimator_name,
                "metric": metric, "aggregation_scope": scope,
                "value": float(np.average(values, weights=weights)) if values else float("nan"),
            })
    return rows


def bootstrap_tree_rule_metrics(layer_from, layer_to, bundle, target_name,
                                estimator_name, model, x_test, y_test,
                                offsets, bootstrap_counts):
    """Bootstrap leaf target effects with the transition's shared protein draws."""
    leaves = model.apply(x_test)
    leaf_ids = np.unique(leaves)
    per_leaf = {}
    for leaf in leaf_ids:
        protein_values = np.full(len(offsets), np.nan, dtype=np.float64)
        for protein_index, (_record, start, end) in enumerate(offsets):
            mask = leaves[start:end] == leaf
            if np.any(mask):
                protein_values[protein_index] = float(np.mean(y_test[start:end][mask]))
        per_leaf[int(leaf)] = protein_values
    rows = []
    metric_prefix = "rule_shrink_prevalence_leaf" if target_name == "classification" else "rule_mean_delta_leaf"
    for replicate in range(bootstrap_counts.shape[0]):
        counts = bootstrap_counts[replicate]
        for leaf, values in per_leaf.items():
            valid = np.isfinite(values) & (counts > 0)
            rows.append({
                "layer_from": layer_from, "layer_to": layer_to,
                "bootstrap_replicate": replicate, "bundle": bundle,
                "target": target_name, "estimator": estimator_name,
                "metric": f"{metric_prefix}_{leaf}", "aggregation_scope": "tree_rule",
                "leaf_id": leaf,
                "value": float(np.average(values[valid], weights=counts[valid])) if np.any(valid) else float("nan"),
            })
    return rows


def permutation_family_rows(layer_from, layer_to, bundle, target_name, estimator_name,
                            model, x_test, y_test, offsets, max_rows=25000):
    rng = np.random.default_rng(RANDOM_SEED + layer_from * 31 + len(bundle))
    selected, selected_offsets, position = [], [], 0
    per_protein_cap = max(1, max_rows // max(len(offsets), 1))
    for record, start, end in offsets:
        count = min(per_protein_cap, end - start)
        local = np.sort(rng.choice(np.arange(start, end), size=count, replace=False))
        selected.append(local)
        selected_offsets.append((record, position, position + count))
        position += count
    indices = np.concatenate(selected) if selected else np.empty(0, np.int64)
    x, y = x_test[indices].copy(), y_test[indices]
    if target_name == "classification":
        baseline = balanced_accuracy_score(y, model.predict(x))
    else:
        baseline = r2_score(y, model.predict(x))
    schema = feature_schema(bundle)
    families = sorted(set(row["feature_family"] for row in schema))
    rows = []
    for family in families:
        columns = [index for index, row in enumerate(schema) if row["feature_family"] == family]
        for repeat in range(3):
            permuted = x.copy()
            for _record, start, end in selected_offsets:
                order = rng.permutation(end - start)
                permuted[start:end, columns] = x[start:end, :][:, columns][order]
            if target_name == "classification":
                score = balanced_accuracy_score(y, model.predict(permuted))
            else:
                score = r2_score(y, model.predict(permuted))
            rows.append({
                "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
                "target": target_name, "estimator": estimator_name,
                "feature_family": family, "repeat": repeat, "n_sampled_pairs": y.size,
                "baseline_metric": baseline, "permuted_metric": score,
                "metric_decrease": baseline - score,
            })
    return rows


def fit_logistic_grid(x_train_raw, labels_train, train_offsets, validation_roles):
    fit_rows, fit_offsets = subset_offsets(train_offsets, validation_roles, "fit")
    validation_rows, validation_offsets = subset_offsets(train_offsets, validation_roles, "validation")
    x_fit_raw, x_validation_raw = x_train_raw[fit_rows], x_train_raw[validation_rows]
    y_fit, y_validation = labels_train[fit_rows], labels_train[validation_rows]
    mean, scale = protein_equal_scaler(x_fit_raw, fit_offsets)
    x_fit, x_validation = standardize(x_fit_raw, mean, scale), standardize(x_validation_raw, mean, scale)
    weights = balanced_binary_weights(protein_equal_weights(fit_offsets), y_fit)
    rows, candidates = [], []
    for c_value in LOGISTIC_C_GRID:
        started = time.time()
        model = LogisticRegression(C=c_value, solver="lbfgs", max_iter=200, random_state=RANDOM_SEED)
        model.fit(x_fit, y_fit, sample_weight=weights)
        converged = int(np.max(model.n_iter_)) < 200
        prediction = model.predict(x_validation)
        protein_scores = []
        for _record, start, end in validation_offsets:
            if np.unique(y_validation[start:end]).size == 2:
                protein_scores.append(balanced_accuracy_score(y_validation[start:end], prediction[start:end]))
        score = finite_mean(protein_scores)
        rows.append({
            "candidate_c": c_value, "mean_per_protein_balanced_accuracy": score,
            "n_validation_proteins": len(validation_offsets), "n_valid_validation_proteins": len(protein_scores),
            "converged": converged, "n_iter": int(np.max(model.n_iter_)),
            "elapsed_seconds": time.time() - started, "selected": False,
        })
        if converged and math.isfinite(score):
            candidates.append((score, -c_value, c_value))
    if not candidates:
        raise RuntimeError("No logistic C candidate converged with a finite validation score")
    selected_c = max(candidates)[2]
    for row in rows:
        row["selected"] = row["candidate_c"] == selected_c
    return selected_c, rows


def main():
    total_started = time.time()
    task_id = int(os.environ["SLURM_ARRAY_TASK_ID"])
    layer_from, layer_to, bundle = task_spec(task_id)
    csv_root = Path(os.environ.get("CSV_ROOT", Path.home() / "csv_files")).expanduser()
    source_root = csv_root / SOURCE_VARIANT
    output_root = Path(os.environ.get("LOCAL_BIO_CONTEXT_OUTPUT_ROOT", csv_root / OUTPUT_NAME)).expanduser()
    output_dir = output_root / "shards" / f"transition_{layer_from:02d}_{layer_to:02d}" / bundle
    output_dir.mkdir(parents=True, exist_ok=True)
    log(f"Starting task={task_id}, transition={layer_from}->{layer_to}, bundle={bundle}")

    assignment_path = source_root / "split_assignments.csv"
    coefficient_path = source_root / "coefficients.csv"
    records, split_source_rows = load_split_records(assignment_path)
    validation_roles, validation_rows = logistic_validation_assignments(records)
    intercept_from, coefficients_from, coef_rows_from, hash_from = load_distance_coefficients(coefficient_path, layer_from)
    intercept_to, coefficients_to, coef_rows_to, hash_to = load_distance_coefficients(coefficient_path, layer_to)
    coefficient_rows = coef_rows_from + coef_rows_to
    offsets_by_split, pair_counts = {}, {}
    for split in ("train", "test"):
        offsets_by_split[split], pair_counts[split] = build_offsets(records, split)

    x_by_split = {split: np.empty((pair_counts[split], BUNDLE_WIDTHS[bundle]), np.float64) for split in ("train", "test")}
    delta_by_split = {split: np.empty(pair_counts[split], np.float64) for split in ("train", "test")}
    triangle_by_split = {split: np.empty((pair_counts[split], 6), np.float64) for split in ("train", "test")}
    current_by_split = {split: np.empty(pair_counts[split], np.float64) for split in ("train", "test")}
    positions = {"train": 0, "test": 0}
    manifest_rows, skipped_rows, timing_rows = [], [], []
    unknown_residues, total_residues, unknown_proteins = 0, 0, set()
    feature_started = time.time()
    try:
        for record_index, record in enumerate(records, start=1):
            protein_started = time.time()
            protein_id, split = record["protein_id"], record["split"]
            from_path, to_path, structure_path = (
                pair_block_path(protein_id, layer_from), pair_block_path(protein_id, layer_to), pdb_path(protein_id)
            )
            try:
                for path in (from_path, to_path, structure_path):
                    if not path.exists():
                        raise FileNotFoundError(path)
                residues = extract_pdb_residue_names(structure_path)
                if len(residues) != record["n_residues"]:
                    raise ValueError(f"PDB CA count {len(residues)} != assigned residues {record['n_residues']}")
                protein_unknown = sum(aa is None for aa in residues)
                total_residues += len(residues)
                unknown_residues += protein_unknown
                if protein_unknown:
                    unknown_proteins.add(protein_id)
                pb_from, pb_to = np.load(from_path), np.load(to_path)
                expected = (record["n_residues"], record["n_residues"], 128)
                if pb_from.shape != expected or pb_to.shape != expected:
                    raise ValueError(f"Pair-block shape mismatch: {pb_from.shape}, {pb_to.shape}, expected {expected}")
                i, j, separation = local_pair_indices(record["n_residues"])
                raw_matrix = decode_distance_matrix(pb_from, intercept_from, coefficients_from)
                next_distance = decode_selected_distances(pb_to, i, j, intercept_to, coefficients_to)
                families = build_all_feature_families(pb_from, raw_matrix, i, j, separation, residues)
                x_chunk = assemble_bundle(families, bundle)
                delta = next_distance - raw_matrix[i, j]
                start, end = positions[split], positions[split] + i.size
                x_by_split[split][start:end] = x_chunk
                delta_by_split[split][start:end] = delta
                triangle_by_split[split][start:end] = families["triangle_pressure"]
                current_by_split[split][start:end] = raw_matrix[i, j]
                positions[split] = end
                manifest_rows.append({
                    "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
                    "split": split, "source": record["source"], "protein_id": protein_id,
                    "n_residues": record["n_residues"], "n_local_pairs": i.size,
                    "pdb_path": str(structure_path), "pair_block_from": str(from_path),
                    "pair_block_to": str(to_path), "n_unknown_residues": protein_unknown,
                    "n_negative_decoded_distances": int(np.count_nonzero(raw_matrix[np.triu_indices(record["n_residues"], 1)] < 0)),
                    "status": "success", "elapsed_seconds": time.time() - protein_started,
                })
                del pb_from, pb_to, raw_matrix, next_distance, families, x_chunk, delta, i, j, separation
                gc.collect()
            except Exception as exc:
                skipped_rows.append({
                    "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
                    "split": split, "source": record["source"], "protein_id": protein_id,
                    "reason": type(exc).__name__, "message": str(exc),
                })
                raise
            if record_index % 10 == 0 or record_index == len(records):
                log(f"Feature build {record_index}/{len(records)} proteins; elapsed={time.time()-feature_started:.1f}s")
    finally:
        write_csv_atomic(output_dir / "protein_manifest.csv", manifest_rows)
        write_csv_atomic(output_dir / "skipped_proteins.csv", skipped_rows, SKIPPED_FIELDS)
    if positions != pair_counts:
        raise AssertionError(f"Filled {positions}, expected {pair_counts}")
    if unknown_residues / max(total_residues, 1) > 0.001 or len(unknown_proteins) > 5:
        raise ValueError(f"Unknown residues exceed policy: {unknown_residues}/{total_residues}, proteins={len(unknown_proteins)}")
    timing_rows.append({
        "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
        "stage": "feature_build", "elapsed_seconds": time.time() - feature_started,
    })

    write_csv_atomic(output_dir / "feature_schema.csv", feature_schema(bundle))
    write_csv_atomic(output_dir / "coefficients_used.csv", coefficient_rows)
    write_csv_atomic(output_dir / "split_assignments.csv", split_source_rows)
    write_csv_atomic(output_dir / "logistic_validation_assignments.csv", validation_rows)
    correlation_rows = []
    for split in ("train", "test"):
        correlation_rows.extend(triangle_distance_correlations(
            layer_from, layer_to, split, current_by_split[split], triangle_by_split[split], offsets_by_split[split]
        ))
    write_csv_atomic(output_dir / "triangle_distance_correlations.csv", correlation_rows)
    correlation_lookup = {(row["split"], row["feature_name"]): row["pair_pearson_r"] for row in correlation_rows}

    bootstrap_counts, bootstrap_assignment_rows = bootstrap_assignments(offsets_by_split["test"], layer_from, layer_to)
    write_csv_atomic(output_dir / "bootstrap_assignments.csv", bootstrap_assignment_rows)
    mean, scale = protein_equal_scaler(x_by_split["train"], offsets_by_split["train"])
    transform_rows = feature_transform_rows(layer_from, layer_to, bundle, x_by_split["train"], mean, scale)
    write_csv_atomic(output_dir / "feature_transforms.csv", transform_rows)
    x_standardized = {split: standardize(x_by_split[split], mean, scale) for split in ("train", "test")}
    regression_weights = protein_equal_weights(offsets_by_split["train"])
    labels_all = {split: class_labels(delta_by_split[split]) for split in ("train", "test")}
    class_x_raw, class_x_std, class_y, class_offsets, class_rows = {}, {}, {}, {}, {}
    for split in ("train", "test"):
        arrays, class_offsets[split], class_rows[split] = filtered_classification_arrays(
            labels_all[split], [x_by_split[split], x_standardized[split], labels_all[split]], offsets_by_split[split]
        )
        class_x_raw[split], class_x_std[split], class_y[split] = arrays
    classification_weights = balanced_binary_weights(protein_equal_weights(class_offsets["train"]), class_y["train"])

    selected_c, search_rows = fit_logistic_grid(class_x_raw["train"], class_y["train"], class_offsets["train"], validation_roles)
    for row in search_rows:
        row.update({"layer_from": layer_from, "layer_to": layer_to, "bundle": bundle})
    write_csv_atomic(output_dir / "logistic_regularization_search.csv", search_rows)

    model_rows, per_protein_rows, coefficient_output_rows = [], [], []
    classification_magnitude_rows, regression_magnitude_rows = [], []
    tree_node_rows, tree_rule_rows, importance_rows, bootstrap_metric_rows = [], [], [], []
    feature_names = [row["feature_name"] for row in feature_schema(bundle)]
    models_to_fit = [
        ("classification", "logistic", LogisticRegression(C=selected_c, solver="lbfgs", max_iter=200, random_state=RANDOM_SEED), class_x_std, class_y, class_offsets, classification_weights),
        ("classification", "decision_tree_depth4", DecisionTreeClassifier(max_depth=4, min_samples_leaf=max(1000, math.ceil(0.002 * class_y["train"].size)), criterion="log_loss", random_state=RANDOM_SEED), class_x_raw, class_y, class_offsets, classification_weights),
        ("classification", "hist_gradient_boosting", HistGradientBoostingClassifier(max_iter=HIST_GB_MAX_ITER, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=RANDOM_SEED), class_x_raw, class_y, class_offsets, classification_weights),
        ("regression", "ridge", Ridge(alpha=0.001 * float(np.sum(regression_weights)), solver="lsqr", tol=1e-4, max_iter=100), x_standardized, delta_by_split, offsets_by_split, regression_weights),
        ("regression", "decision_tree_depth4", DecisionTreeRegressor(max_depth=4, min_samples_leaf=max(1000, math.ceil(0.002 * delta_by_split["train"].size)), criterion="squared_error", random_state=RANDOM_SEED), x_by_split, delta_by_split, offsets_by_split, regression_weights),
        ("regression", "hist_gradient_boosting", HistGradientBoostingRegressor(max_iter=HIST_GB_MAX_ITER, learning_rate=0.08, max_leaf_nodes=15, min_samples_leaf=200, l2_regularization=1.0, early_stopping=False, random_state=RANDOM_SEED), x_by_split, delta_by_split, offsets_by_split, regression_weights),
    ]
    if bundle == "biology_local_triangle":
        models_to_fit.extend([
            ("classification", "decision_tree_depth6", DecisionTreeClassifier(max_depth=6, min_samples_leaf=max(1000, math.ceil(0.002 * class_y["train"].size)), criterion="log_loss", random_state=RANDOM_SEED), class_x_raw, class_y, class_offsets, classification_weights),
            ("regression", "decision_tree_depth6", DecisionTreeRegressor(max_depth=6, min_samples_leaf=max(1000, math.ceil(0.002 * delta_by_split["train"].size)), criterion="squared_error", random_state=RANDOM_SEED), x_by_split, delta_by_split, offsets_by_split, regression_weights),
        ])

    metadata = {
        "status": "running_models", "layer_from": layer_from, "layer_to": layer_to,
        "bundle": bundle, "feature_dim": BUNDLE_WIDTHS[bundle], "task_id": task_id,
        "distance_probe_policy": "saved_pooled_own_layer", "distance_probe_layers": [layer_from, layer_to],
        "distance_probe_sha256": {str(layer_from): hash_from, str(layer_to): hash_to},
        "hist_gradient_boosting_max_iter": HIST_GB_MAX_ITER,
        "hist_gradient_boosting_early_stopping": False,
        "selected_logistic_c": selected_c, "started_at": now_timestamp(),
        "sequence_separation": [4, 12], "local_sequence_radius": 5,
        "classification_threshold_angstrom": CLASS_THRESHOLD,
        "classification_boundary_policy": "exclude_abs_delta_le_0.5",
        "feature_dtype": "float64", "bundle_families": list(BUNDLE_FAMILIES[bundle]),
        "coefficient_source": str(coefficient_path), "split_source": str(assignment_path),
        "model_configuration": {
            "ridge_relative_alpha": 0.001, "ridge_solver": "lsqr",
            "logistic_c_grid": list(LOGISTIC_C_GRID), "logistic_solver": "lbfgs",
            "decision_tree_depth": 4, "decision_tree_sensitivity_depth": 6,
            "hist_gradient_boosting_iterations": HIST_GB_MAX_ITER,
            "hist_gradient_boosting_learning_rate": 0.08,
            "hist_gradient_boosting_max_leaf_nodes": 15,
            "hist_gradient_boosting_min_samples_leaf": 200,
            "hist_gradient_boosting_l2_regularization": 1.0,
            "hist_gradient_boosting_early_stopping": False,
        },
        "unknown_residue_count": unknown_residues, "unknown_residue_protein_count": len(unknown_proteins),
        "constants": metadata_constants(),
    }
    write_json_atomic(output_dir / "run_metadata.json", metadata)

    for model_index, (target_name, estimator_name, model, x_inputs, y_inputs, model_offsets, weights) in enumerate(models_to_fit, start=1):
        started = time.time()
        log(f"Fitting {target_name}/{estimator_name} ({model_index}/{len(models_to_fit)})")
        model.fit(x_inputs["train"], y_inputs["train"], sample_weight=weights)
        if estimator_name == "hist_gradient_boosting":
            if model.early_stopping is not False or int(model.n_iter_) != HIST_GB_MAX_ITER:
                raise AssertionError(f"HistGB configuration drift: early_stopping={model.early_stopping}, n_iter={model.n_iter_}")
        predictions = {split: model.predict(x_inputs[split]) for split in ("train", "test")}
        scores = {split: estimator_score(model, x_inputs[split]) for split in ("train", "test")} if target_name == "classification" else {}
        summary, per_rows = evaluate_model(
            layer_from, layer_to, bundle, target_name, estimator_name,
            y_inputs, predictions, scores, model_offsets, pair_counts,
        )
        for row in summary:
            row.update({
                "fit_elapsed_seconds": time.time() - started,
                "selected_logistic_c": selected_c if estimator_name == "logistic" else "",
                "n_iter": int(np.max(np.atleast_1d(getattr(model, "n_iter_", -1)))),
                "converged": not (estimator_name == "logistic" and int(np.max(model.n_iter_)) >= 200),
            })
        model_rows.extend(summary)
        per_protein_rows.extend(per_rows)
        bootstrap_metric_rows.extend(bootstrap_model_metrics(
            layer_from, layer_to, bundle, target_name, estimator_name, per_rows, bootstrap_counts
        ))
        if estimator_name in {"logistic", "ridge"}:
            coefficient_output_rows.extend(linear_coefficient_rows(
                layer_from, layer_to, bundle, target_name, estimator_name, model, mean, scale
            ))
        if "decision_tree" in estimator_name:
            nodes, rules = tree_artifacts(
                layer_from, layer_to, bundle, target_name, estimator_name, model,
                x_inputs, y_inputs, predictions, scores, model_offsets,
                correlation_lookup, output_dir,
            )
            tree_node_rows.extend(nodes)
            tree_rule_rows.extend(rules)
            bootstrap_metric_rows.extend(bootstrap_tree_rule_metrics(
                layer_from, layer_to, bundle, target_name, estimator_name, model,
                x_inputs["test"], y_inputs["test"], model_offsets["test"],
                bootstrap_counts,
            ))
        if estimator_name == "hist_gradient_boosting":
            importance_rows.extend(permutation_family_rows(
                layer_from, layer_to, bundle, target_name, estimator_name,
                model, x_inputs["test"], y_inputs["test"], model_offsets["test"]
            ))
        if target_name == "classification":
            for split in ("train", "test"):
                all_prediction = model.predict(x_standardized[split] if estimator_name == "logistic" else x_by_split[split])
                all_score = estimator_score(model, x_standardized[split] if estimator_name == "logistic" else x_by_split[split])
                classification_magnitude_rows.extend(magnitude_rows(
                    layer_from, layer_to, bundle, estimator_name, target_name, split,
                    delta_by_split[split], all_prediction, all_score, offsets_by_split[split]
                ))
        else:
            for split in ("train", "test"):
                regression_magnitude_rows.extend(magnitude_rows(
                    layer_from, layer_to, bundle, estimator_name, target_name, split,
                    delta_by_split[split], predictions[split], None, offsets_by_split[split]
                ))
        atomic_joblib_dump(output_dir / "models" / f"{target_name}_{estimator_name}.joblib", model, feature_names)
        timing_rows.append({
            "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
            "stage": f"fit_evaluate_{target_name}_{estimator_name}",
            "elapsed_seconds": time.time() - started,
        })
        write_csv_atomic(output_dir / "model_summary.csv", model_rows)
        write_csv_atomic(output_dir / "per_protein_metrics.csv", per_protein_rows)
        write_csv_atomic(output_dir / "bootstrap_metrics.csv", bootstrap_metric_rows)
        metadata["completed_models"] = model_index
        write_json_atomic(output_dir / "run_metadata.json", metadata)
        del predictions, scores
        gc.collect()

    change_rows = []
    for split in ("train", "test"):
        values = delta_by_split[split]
        labels = class_labels(values)
        quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99])
        change_rows.append({
            "layer_from": layer_from, "layer_to": layer_to, "split": split,
            "n_proteins": len(offsets_by_split[split]), "n_pairs": values.size,
            "mean": np.mean(values), "std": np.std(values, ddof=1), "minimum": np.min(values),
            "q01": quantiles[0], "q05": quantiles[1], "q25": quantiles[2],
            "median": quantiles[3], "q75": quantiles[4], "q95": quantiles[5],
            "q99": quantiles[6], "maximum": np.max(values),
            "n_shrink": np.count_nonzero(labels == 1), "n_excluded": np.count_nonzero(labels == -1),
            "n_expand": np.count_nonzero(labels == 0), "threshold_angstrom": CLASS_THRESHOLD,
        })
    timing_rows.append({
        "layer_from": layer_from, "layer_to": layer_to, "bundle": bundle,
        "stage": "total", "elapsed_seconds": time.time() - total_started,
    })
    write_csv_atomic(output_dir / "change_statistics.csv", change_rows)
    write_csv_atomic(output_dir / "model_summary.csv", model_rows)
    write_csv_atomic(output_dir / "per_protein_metrics.csv", per_protein_rows)
    write_csv_atomic(output_dir / "linear_coefficients.csv", coefficient_output_rows)
    write_csv_atomic(output_dir / "tree_nodes.csv", tree_node_rows)
    write_csv_atomic(output_dir / "tree_rule_metrics.csv", tree_rule_rows)
    write_csv_atomic(output_dir / "feature_family_importance.csv", importance_rows)
    write_csv_atomic(output_dir / "classification_magnitude_metrics.csv", classification_magnitude_rows)
    write_csv_atomic(output_dir / "regression_magnitude_metrics.csv", regression_magnitude_rows)
    write_csv_atomic(output_dir / "bootstrap_metrics.csv", bootstrap_metric_rows)
    write_csv_atomic(output_dir / "timing.csv", timing_rows)
    metadata.update({"status": "complete", "completed_at": now_timestamp(), "elapsed_seconds": time.time() - total_started})
    write_json_atomic(output_dir / "run_metadata.json", metadata)
    write_text_atomic(output_dir / "summary.txt", (
        f"Local biological-context update probe shard\n"
        f"Transition: {layer_from}->{layer_to}\nBundle: {bundle} ({BUNDLE_WIDTHS[bundle]} features)\n"
        f"Status: complete\nHistGradientBoosting iterations: {HIST_GB_MAX_ITER}; early_stopping=False\n"
        f"Selected logistic C: {selected_c}\nElapsed seconds: {time.time()-total_started:.1f}\n"
    ))
    log(f"Completed {layer_from}->{layer_to}/{bundle} in {time.time()-total_started:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

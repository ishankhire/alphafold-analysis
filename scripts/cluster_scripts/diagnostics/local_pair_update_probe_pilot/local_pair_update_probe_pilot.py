#!/usr/bin/env python3
"""Pre-MDS local pair-update classification and regression pilot.

One Slurm array task fits one feature variant for one transition.  The frozen
general_base256_combined protein split and saved distance coefficients for
both endpoint layers are read from CSV_ROOT.  Each endpoint representation is
decoded by its own layer's probe.  Only pairs with inclusive sequence
separation 4..12 are used, and only layer-x representations are exposed to the
update-prediction probes.

The file is deliberately self-contained so this directory can be copied to
$HOME/openfold_interpretability_files/ on Delta without copying repository
Python helpers.
"""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import Ridge, RidgeClassifier
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    mean_absolute_error,
    mean_squared_error,
    precision_recall_fscore_support,
    r2_score,
)


N_CHANNELS = 128
DISTANCE_FEATURE_DIM = 2 * N_CHANNELS
TRANSITIONS = ((10, 11), (20, 21), (30, 31))
VARIANTS = ("distance_seqsep", "center", "triangle_r1", "triangle_r2", "triangle_r3")
CLASS_VALUES = (-1, 0, 1)
CLASS_NAMES = {-1: "decrease", 0: "no_change", 1: "increase"}
CLASS_THRESHOLD = 0.5
MIN_SEQUENCE_SEPARATION = 4
MAX_SEQUENCE_SEPARATION = 12
RELATIVE_RIDGE_ALPHA = 1e-3
LSQR_TOLERANCE = 1e-3
LSQR_MAX_ITER = 30
SPLIT_ID = 0
OUTPUT_NAME = "local_pair_update_probe_pilot"
SOURCE_VARIANT = "general_base256_combined"

CAMEO_PAIR_BLOCKS_ROOT = Path("/work/hdd/bekh/thayes/openfold_reps")
ECOD_PAIR_BLOCKS_ROOT = Path(
    "/work/hdd/bfna/thayes/ecod/experiments/representation_saves"
)

CLASSIFICATION_SUMMARY_FIELDS = [
    "layer_from", "layer_to", "variant", "feature_dim", "split", "n_proteins",
    "n_pairs", "n_decrease", "n_no_change", "n_increase", "pair_accuracy",
    "pair_balanced_accuracy", "pair_macro_f1", "mean_per_protein_accuracy",
    "mean_per_protein_balanced_accuracy", "mean_per_protein_macro_f1",
    "relative_alpha", "actual_alpha", "solver", "tolerance", "max_iter",
    "n_iter", "iteration_cap_reached",
]
REGRESSION_SUMMARY_FIELDS = [
    "layer_from", "layer_to", "variant", "feature_dim", "split", "n_proteins",
    "n_pairs", "pair_r2", "pair_mae", "pair_rmse", "pair_pearson_r",
    "pair_bias", "mean_per_protein_r2", "mean_per_protein_mae",
    "mean_per_protein_rmse", "mean_per_protein_pearson_r",
    "mean_per_protein_bias", "relative_alpha", "actual_alpha", "solver",
    "tolerance", "max_iter", "n_iter", "iteration_cap_reached",
]
PER_PROTEIN_FIELDS = [
    "layer_from", "layer_to", "variant", "feature_dim", "probe_type", "split",
    "source", "protein_id", "n_residues", "n_pairs", "n_decrease",
    "n_no_change", "n_increase", "accuracy", "balanced_accuracy", "macro_f1",
    "r2", "mae", "rmse", "pearson_r", "bias",
]
CLASS_METRIC_FIELDS = [
    "layer_from", "layer_to", "variant", "feature_dim", "split", "class_value",
    "class_name", "precision", "recall", "f1", "support", "prevalence",
]
CONFUSION_FIELDS = [
    "layer_from", "layer_to", "variant", "feature_dim", "split",
    "true_class_value", "true_class_name", "predicted_class_value",
    "predicted_class_name", "count", "row_fraction",
]
COEFFICIENT_FIELDS = [
    "layer_from", "layer_to", "variant", "feature_dim", "probe_type",
    "output_class_value", "output_class_name", "feature_index", "feature_name",
    "coefficient_standardized", "relative_alpha", "actual_alpha",
]
CHANGE_STATISTIC_FIELDS = [
    "layer_from", "layer_to", "distance_type", "split", "n_proteins", "n_pairs",
    "mean", "std", "minimum", "q01", "q05", "q25", "median", "q75", "q95",
    "q99", "maximum", "mean_absolute", "rms", "n_decrease", "n_no_change",
    "n_increase", "fraction_decrease", "fraction_no_change", "fraction_increase",
    "threshold_angstrom",
]
FEATURE_SCHEMA_FIELDS = [
    "variant", "feature_dim", "feature_index", "feature_name", "feature_kind",
    "shell", "slot_role", "leg", "channel", "validity_for",
]
MANIFEST_FIELDS = [
    "layer_from", "layer_to", "variant", "split", "source", "protein_id",
    "n_residues", "n_local_pairs", "pair_block_from", "pair_block_to", "status",
    "elapsed_seconds", "message",
]
SKIPPED_FIELDS = [
    "layer_from", "layer_to", "variant", "split", "source", "protein_id",
    "reason", "message",
]
TIMING_FIELDS = ["layer_from", "layer_to", "variant", "stage", "elapsed_seconds"]


def now_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"[{now_timestamp()}] {message}", flush=True)


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        if not math.isfinite(value):
            return "nan"
        return f"{value:.9g}"
    return value


def write_csv_atomic(path, fieldnames, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})
    os.replace(temporary, path)


def write_json_atomic(path, payload):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_text_atomic(path, text):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(text)
    os.replace(temporary, path)


def task_spec(task_id):
    task_id = int(task_id)
    if task_id < 0 or task_id >= len(TRANSITIONS) * len(VARIANTS):
        raise ValueError(f"Task ID {task_id} is outside 0..14")
    transition_index, variant_index = divmod(task_id, len(VARIANTS))
    layer_from, layer_to = TRANSITIONS[transition_index]
    return layer_from, layer_to, VARIANTS[variant_index]


def requested_task_spec():
    raw = os.environ.get("SLURM_ARRAY_TASK_ID")
    if raw is None or raw == "":
        raise ValueError("SLURM_ARRAY_TASK_ID is required and must be in 0..14")
    return task_spec(int(raw))


def local_pair_indices(n_residues):
    i_parts = []
    j_parts = []
    separation_parts = []
    for separation in range(MIN_SEQUENCE_SEPARATION, MAX_SEQUENCE_SEPARATION + 1):
        count = max(int(n_residues) - separation, 0)
        if count == 0:
            continue
        i = np.arange(count, dtype=np.int64)
        i_parts.append(i)
        j_parts.append(i + separation)
        separation_parts.append(np.full(count, separation, dtype=np.int16))
    if not i_parts:
        return (
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int64),
            np.empty(0, dtype=np.int16),
        )
    return np.concatenate(i_parts), np.concatenate(j_parts), np.concatenate(separation_parts)


def local_pair_count(n_residues):
    return sum(
        max(int(n_residues) - separation, 0)
        for separation in range(MIN_SEQUENCE_SEPARATION, MAX_SEQUENCE_SEPARATION + 1)
    )


def class_labels(changes, threshold=CLASS_THRESHOLD):
    values = np.asarray(changes, dtype=np.float64)
    labels = np.zeros(values.shape, dtype=np.int8)
    labels[values < -float(threshold)] = -1
    labels[values > float(threshold)] = 1
    return labels


def source_for_protein(protein_id):
    return "cameo" if "_" in str(protein_id) else "ecod"


def pair_block_path(protein_id, layer):
    source = source_for_protein(protein_id)
    if source == "cameo":
        root = Path(os.environ.get("CAMEO_PAIR_BLOCKS_ROOT", CAMEO_PAIR_BLOCKS_ROOT))
        return root / protein_id / f"{protein_id}_pair_block_{layer}.npy"
    root = Path(os.environ.get("ECOD_PAIR_BLOCKS_ROOT", ECOD_PAIR_BLOCKS_ROOT))
    return (
        root
        / f"{protein_id}_save_layers_0_47"
        / f"{protein_id}_pair_block_{layer}.npy"
    )


def load_split_records(path):
    records = []
    source_rows = []
    seen = set()
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["split_id"]) != SPLIT_ID:
                continue
            protein_id = row["protein_id"]
            if protein_id in seen:
                raise ValueError(f"Duplicate split assignment for {protein_id}")
            split = row["split"]
            if split not in {"train", "test"}:
                raise ValueError(f"Unexpected split {split!r} for {protein_id}")
            record = {
                "protein_id": protein_id,
                "source": source_for_protein(protein_id),
                "split": split,
                "n_residues": int(row["n_residues"]),
                "n_local_pairs": local_pair_count(int(row["n_residues"])),
            }
            records.append(record)
            source_rows.append(dict(row))
            seen.add(protein_id)
    if not records:
        raise ValueError(f"No split_id={SPLIT_ID} records found in {path}")
    return records, source_rows


def load_distance_coefficients(path, layer):
    coefficients = np.full(DISTANCE_FEATURE_DIM, np.nan, dtype=np.float64)
    intercept = None
    source_rows = []
    seen = set()
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != SOURCE_VARIANT:
                continue
            if int(row["split_id"]) != SPLIT_ID or int(row["layer"]) != int(layer):
                continue
            if int(row["feature_dim"]) != DISTANCE_FEATURE_DIM:
                raise ValueError(f"Unexpected distance feature dimension in row {row}")
            index = int(row["feature_index"])
            value = float(row["coefficient_raw_scale"])
            source_rows.append(dict(row))
            if index == -1:
                if intercept is not None:
                    raise ValueError(f"Duplicate intercept for layer {layer}")
                if row["feature_name"] != "intercept":
                    raise ValueError(f"Unexpected intercept feature name: {row}")
                intercept = value
                continue
            if index in seen or index < 0 or index >= DISTANCE_FEATURE_DIM:
                raise ValueError(f"Invalid or duplicate coefficient index {index}")
            expected = (
                f"pair_ij_ch_{index}"
                if index < N_CHANNELS
                else f"pair_ji_ch_{index - N_CHANNELS}"
            )
            if row["feature_name"] != expected:
                raise ValueError(
                    f"Coefficient {index} is {row['feature_name']!r}; expected {expected!r}"
                )
            coefficients[index] = value
            seen.add(index)
    if intercept is None or len(seen) != DISTANCE_FEATURE_DIM:
        raise ValueError(f"Layer {layer} is missing saved distance coefficients in {path}")
    if not np.all(np.isfinite(coefficients)):
        raise ValueError(f"Layer {layer} contains non-finite saved coefficients")
    return float(intercept), coefficients, source_rows


def symmetric_edge(pb, a, b):
    return 0.5 * (
        np.asarray(pb[a, b, :], dtype=np.float64)
        + np.asarray(pb[b, a, :], dtype=np.float64)
    )


def triangle_radius(variant):
    if variant.startswith("triangle_r"):
        return int(variant.rsplit("r", 1)[1])
    return 0


def feature_schema(variant):
    rows = []

    def append(name, kind, shell="", slot_role="", leg="", channel="", validity_for=""):
        rows.append(
            {
                "variant": variant,
                "feature_dim": 0,
                "feature_index": len(rows),
                "feature_name": name,
                "feature_kind": kind,
                "shell": shell,
                "slot_role": slot_role,
                "leg": leg,
                "channel": channel,
                "validity_for": validity_for,
            }
        )

    if variant == "distance_seqsep":
        append("predicted_distance_from", "ordinary")
        append("sequence_separation", "ordinary")
    else:
        if variant != "center" and triangle_radius(variant) == 0:
            raise ValueError(f"Unknown feature variant {variant!r}")
        for channel in range(N_CHANNELS):
            append(f"center_avg_ch_{channel}", "center", channel=channel)
        for shell in range(1, triangle_radius(variant) + 1):
            roles = (
                ("i_minus", -shell, "i"),
                ("i_plus", shell, "i"),
                ("j_minus", -shell, "j"),
                ("j_plus", shell, "j"),
            )
            for role, offset, anchor in roles:
                slot = f"shell_{shell}_{role}"
                for leg in ("i_to_k", "k_to_j"):
                    for channel in range(N_CHANNELS):
                        append(
                            f"{slot}_{leg}_avg_ch_{channel}",
                            "triangle_edge",
                            shell=shell,
                            slot_role=f"{anchor}{offset:+d}",
                            leg=leg,
                            channel=channel,
                        )
                append(
                    f"{slot}_valid",
                    "validity_mask",
                    shell=shell,
                    slot_role=f"{anchor}{offset:+d}",
                    validity_for=slot,
                )
    dimension = len(rows)
    for row in rows:
        row["feature_dim"] = dimension
    return rows


def feature_names(variant):
    return [row["feature_name"] for row in feature_schema(variant)]


def feature_dim(variant):
    return len(feature_schema(variant))


def validate_pair_block(pb, n_residues, path):
    expected = (int(n_residues), int(n_residues), N_CHANNELS)
    if tuple(pb.shape) != expected:
        raise ValueError(f"Pair block {path} has shape {pb.shape}; expected {expected}")


def distance_change_and_current(
    pb_from,
    pb_to,
    i,
    j,
    intercept_from,
    coefficients_from,
    intercept_to,
    coefficients_to,
):
    """Decode each endpoint with its own saved probe for canonical i < j pairs."""
    coef_lower_from = coefficients_from[:N_CHANNELS]
    coef_upper_from = coefficients_from[N_CHANNELS:]
    coef_lower_to = coefficients_to[:N_CHANNELS]
    coef_upper_to = coefficients_to[N_CHANNELS:]
    lower_from = np.asarray(pb_from[j, i, :], dtype=np.float64)
    upper_from = np.asarray(pb_from[i, j, :], dtype=np.float64)
    lower_to = np.asarray(pb_to[j, i, :], dtype=np.float64)
    upper_to = np.asarray(pb_to[i, j, :], dtype=np.float64)
    distance_from = (
        float(intercept_from)
        + lower_from @ coef_lower_from
        + upper_from @ coef_upper_from
    )
    distance_to = (
        float(intercept_to)
        + lower_to @ coef_lower_to
        + upper_to @ coef_upper_to
    )
    change = distance_to - distance_from
    return np.asarray(distance_from, dtype=np.float64), np.asarray(change, dtype=np.float64)


def build_feature_chunk(pb_from, i, j, separation, variant, distance_from=None):
    schema = feature_schema(variant)
    x = np.zeros((i.size, len(schema)), dtype=np.float64)
    if variant == "distance_seqsep":
        if distance_from is None:
            raise ValueError("distance_seqsep requires distance_from")
        x[:, 0] = np.asarray(distance_from, dtype=np.float64)
        x[:, 1] = np.asarray(separation, dtype=np.float64)
        return x

    x[:, :N_CHANNELS] = symmetric_edge(pb_from, i, j)
    write_col = N_CHANNELS
    n_residues = int(pb_from.shape[0])
    for shell in range(1, triangle_radius(variant) + 1):
        slot_specs = (
            (i - shell),
            (i + shell),
            (j - shell),
            (j + shell),
        )
        for k in slot_specs:
            valid = (k >= 0) & (k < n_residues) & (k != i) & (k != j)
            if np.any(valid):
                iv = i[valid]
                jv = j[valid]
                kv = k[valid]
                x[valid, write_col:write_col + N_CHANNELS] = symmetric_edge(
                    pb_from, iv, kv
                )
                write_col += N_CHANNELS
                x[valid, write_col:write_col + N_CHANNELS] = symmetric_edge(
                    pb_from, kv, jv
                )
                write_col += N_CHANNELS
                x[valid, write_col] = 1.0
                write_col += 1
            else:
                write_col += (2 * N_CHANNELS) + 1
    if write_col != x.shape[1]:
        raise AssertionError(f"Feature writer stopped at {write_col}, expected {x.shape[1]}")
    return x


def build_offsets(records, split):
    offsets = []
    start = 0
    for record in records:
        if record["split"] != split:
            continue
        end = start + int(record["n_local_pairs"])
        offsets.append((record, start, end))
        start = end
    return offsets, start


def protein_equal_weights(offsets):
    total_rows = offsets[-1][2] if offsets else 0
    weights = np.empty(total_rows, dtype=np.float64)
    n_proteins = len(offsets)
    for _record, start, end in offsets:
        weights[start:end] = total_rows / (n_proteins * max(end - start, 1))
    return weights


def protein_equal_scaler(x_train, offsets):
    if not offsets:
        raise ValueError("Cannot scale without training proteins")
    mean = np.zeros(x_train.shape[1], dtype=np.float64)
    second = np.zeros(x_train.shape[1], dtype=np.float64)
    for _record, start, end in offsets:
        block = x_train[start:end]
        mean += np.mean(block, axis=0, dtype=np.float64)
        second += np.einsum("ij,ij->j", block, block, dtype=np.float64) / max(end - start, 1)
    mean /= len(offsets)
    second /= len(offsets)
    scale = np.sqrt(np.maximum(second - mean * mean, 0.0))
    scale[scale < 1e-12] = 1.0
    return mean, scale


def standardize_in_place(x, mean, scale, offsets):
    for _record, start, end in offsets:
        x[start:end] -= mean
        x[start:end] /= scale


def balanced_classification_weights(base_weights, labels):
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    labels = np.asarray(labels, dtype=np.int8)
    total_mass = float(np.sum(weights))
    for class_value in CLASS_VALUES:
        mask = labels == class_value
        class_mass = float(np.sum(weights[mask]))
        if class_mass <= 0.0:
            raise ValueError(f"Training data contains no examples of class {class_value}")
        weights[mask] *= total_mass / (len(CLASS_VALUES) * class_mass)
    weights *= weights.size / np.sum(weights)
    return weights


def balanced_accuracy(y_true, y_pred):
    recalls = []
    for class_value in CLASS_VALUES:
        mask = np.asarray(y_true) == class_value
        if np.any(mask):
            recalls.append(float(np.mean(np.asarray(y_pred)[mask] == class_value)))
    return float(np.mean(recalls)) if recalls else float("nan")


def pearson_r(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size < 2 or np.std(y_true) <= 0.0 or np.std(y_pred) <= 0.0:
        return float("nan")
    return float(np.corrcoef(y_true, y_pred)[0, 1])


def safe_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    return float(r2_score(y_true, y_pred)) if y_true.size > 1 and np.var(y_true) > 0 else float("nan")


def iteration_info(model):
    raw = getattr(model, "n_iter_", None)
    if raw is None:
        return -1, False
    values = np.atleast_1d(raw).astype(np.int64)
    n_iter = int(np.max(values)) if values.size else -1
    return n_iter, bool(n_iter >= LSQR_MAX_ITER)


def classification_outputs(layer_from, layer_to, variant, model, labels_by_split, predictions_by_split, offsets_by_split, alpha):
    summary_rows = []
    per_rows = []
    class_rows = []
    confusion_rows = []
    n_iter, cap_reached = iteration_info(model)
    for split in ("train", "test"):
        y_true = labels_by_split[split]
        y_pred = predictions_by_split[split]
        offsets = offsets_by_split[split]
        protein_accuracy = []
        protein_balanced = []
        protein_macro_f1 = []
        for record, start, end in offsets:
            true_p = y_true[start:end]
            pred_p = y_pred[start:end]
            accuracy = float(accuracy_score(true_p, pred_p))
            balanced = balanced_accuracy(true_p, pred_p)
            macro = float(f1_score(true_p, pred_p, labels=CLASS_VALUES, average="macro", zero_division=0))
            protein_accuracy.append(accuracy)
            protein_balanced.append(balanced)
            protein_macro_f1.append(macro)
            counts = {value: int(np.count_nonzero(true_p == value)) for value in CLASS_VALUES}
            per_rows.append(
                {
                    "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                    "feature_dim": feature_dim(variant), "probe_type": "classification",
                    "split": split, "source": record["source"],
                    "protein_id": record["protein_id"], "n_residues": record["n_residues"],
                    "n_pairs": end - start, "n_decrease": counts[-1],
                    "n_no_change": counts[0], "n_increase": counts[1],
                    "accuracy": accuracy, "balanced_accuracy": balanced, "macro_f1": macro,
                }
            )
        counts = {value: int(np.count_nonzero(y_true == value)) for value in CLASS_VALUES}
        summary_rows.append(
            {
                "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                "feature_dim": feature_dim(variant), "split": split,
                "n_proteins": len(offsets), "n_pairs": y_true.size,
                "n_decrease": counts[-1], "n_no_change": counts[0], "n_increase": counts[1],
                "pair_accuracy": accuracy_score(y_true, y_pred),
                "pair_balanced_accuracy": balanced_accuracy(y_true, y_pred),
                "pair_macro_f1": f1_score(y_true, y_pred, labels=CLASS_VALUES, average="macro", zero_division=0),
                "mean_per_protein_accuracy": np.mean(protein_accuracy),
                "mean_per_protein_balanced_accuracy": np.mean(protein_balanced),
                "mean_per_protein_macro_f1": np.mean(protein_macro_f1),
                "relative_alpha": RELATIVE_RIDGE_ALPHA, "actual_alpha": alpha,
                "solver": "lsqr", "tolerance": LSQR_TOLERANCE,
                "max_iter": LSQR_MAX_ITER, "n_iter": n_iter,
                "iteration_cap_reached": cap_reached,
            }
        )
        precision, recall, f1_values, support = precision_recall_fscore_support(
            y_true, y_pred, labels=CLASS_VALUES, zero_division=0
        )
        for index, class_value in enumerate(CLASS_VALUES):
            class_rows.append(
                {
                    "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                    "feature_dim": feature_dim(variant), "split": split,
                    "class_value": class_value, "class_name": CLASS_NAMES[class_value],
                    "precision": precision[index], "recall": recall[index], "f1": f1_values[index],
                    "support": int(support[index]),
                    "prevalence": int(support[index]) / max(y_true.size, 1),
                }
            )
        matrix = confusion_matrix(y_true, y_pred, labels=CLASS_VALUES)
        for true_index, true_value in enumerate(CLASS_VALUES):
            row_total = int(np.sum(matrix[true_index]))
            for predicted_index, predicted_value in enumerate(CLASS_VALUES):
                count = int(matrix[true_index, predicted_index])
                confusion_rows.append(
                    {
                        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                        "feature_dim": feature_dim(variant), "split": split,
                        "true_class_value": true_value, "true_class_name": CLASS_NAMES[true_value],
                        "predicted_class_value": predicted_value,
                        "predicted_class_name": CLASS_NAMES[predicted_value], "count": count,
                        "row_fraction": count / row_total if row_total else float("nan"),
                    }
                )
    return summary_rows, per_rows, class_rows, confusion_rows


def regression_outputs(layer_from, layer_to, variant, model, targets_by_split, predictions_by_split, offsets_by_split, alpha):
    summary_rows = []
    per_rows = []
    n_iter, cap_reached = iteration_info(model)
    for split in ("train", "test"):
        y_true = targets_by_split[split]
        y_pred = predictions_by_split[split]
        offsets = offsets_by_split[split]
        metric_lists = {key: [] for key in ("r2", "mae", "rmse", "pearson_r", "bias")}
        for record, start, end in offsets:
            true_p = y_true[start:end]
            pred_p = y_pred[start:end]
            metrics = {
                "r2": safe_r2(true_p, pred_p),
                "mae": mean_absolute_error(true_p, pred_p),
                "rmse": math.sqrt(mean_squared_error(true_p, pred_p)),
                "pearson_r": pearson_r(true_p, pred_p),
                "bias": float(np.mean(pred_p - true_p)),
            }
            for key, value in metrics.items():
                if math.isfinite(float(value)):
                    metric_lists[key].append(float(value))
            per_rows.append(
                {
                    "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                    "feature_dim": feature_dim(variant), "probe_type": "regression",
                    "split": split, "source": record["source"],
                    "protein_id": record["protein_id"], "n_residues": record["n_residues"],
                    "n_pairs": end - start, **metrics,
                }
            )
        summary_rows.append(
            {
                "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                "feature_dim": feature_dim(variant), "split": split,
                "n_proteins": len(offsets), "n_pairs": y_true.size,
                "pair_r2": safe_r2(y_true, y_pred),
                "pair_mae": mean_absolute_error(y_true, y_pred),
                "pair_rmse": math.sqrt(mean_squared_error(y_true, y_pred)),
                "pair_pearson_r": pearson_r(y_true, y_pred),
                "pair_bias": float(np.mean(y_pred - y_true)),
                **{
                    f"mean_per_protein_{key}": (
                        float(np.mean(values)) if values else float("nan")
                    )
                    for key, values in metric_lists.items()
                },
                "relative_alpha": RELATIVE_RIDGE_ALPHA, "actual_alpha": alpha,
                "solver": "lsqr", "tolerance": LSQR_TOLERANCE,
                "max_iter": LSQR_MAX_ITER, "n_iter": n_iter,
                "iteration_cap_reached": cap_reached,
            }
        )
    return summary_rows, per_rows


def coefficient_rows(layer_from, layer_to, variant, model, probe_type, alpha):
    names = feature_names(variant)
    rows = []
    if probe_type == "classification":
        coef = np.asarray(model.coef_, dtype=np.float64)
        intercept = np.atleast_1d(model.intercept_).astype(np.float64)
        outputs = [(int(value), CLASS_NAMES[int(value)]) for value in model.classes_]
    else:
        coef = np.atleast_2d(np.asarray(model.coef_, dtype=np.float64))
        intercept = np.atleast_1d(model.intercept_).astype(np.float64)
        outputs = [("", "")]
    for output_index, (class_value, class_name) in enumerate(outputs):
        rows.append(
            {
                "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                "feature_dim": len(names), "probe_type": probe_type,
                "output_class_value": class_value, "output_class_name": class_name,
                "feature_index": -1, "feature_name": "intercept",
                "coefficient_standardized": intercept[output_index],
                "relative_alpha": RELATIVE_RIDGE_ALPHA, "actual_alpha": alpha,
            }
        )
        for index, name in enumerate(names):
            rows.append(
                {
                    "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                    "feature_dim": len(names), "probe_type": probe_type,
                    "output_class_value": class_value, "output_class_name": class_name,
                    "feature_index": index, "feature_name": name,
                    "coefficient_standardized": coef[output_index, index],
                    "relative_alpha": RELATIVE_RIDGE_ALPHA, "actual_alpha": alpha,
                }
            )
    return rows


def summarize_change(layer_from, layer_to, split, values, n_proteins):
    values = np.asarray(values, dtype=np.float64)
    labels = class_labels(values)
    quantiles = np.quantile(values, [0.01, 0.05, 0.25, 0.50, 0.75, 0.95, 0.99])
    counts = {value: int(np.count_nonzero(labels == value)) for value in CLASS_VALUES}
    return {
        "layer_from": layer_from, "layer_to": layer_to, "distance_type": "pre_mds_raw",
        "split": split, "n_proteins": n_proteins, "n_pairs": values.size,
        "mean": np.mean(values), "std": np.std(values, ddof=1), "minimum": np.min(values),
        "q01": quantiles[0], "q05": quantiles[1], "q25": quantiles[2],
        "median": quantiles[3], "q75": quantiles[4], "q95": quantiles[5],
        "q99": quantiles[6], "maximum": np.max(values),
        "mean_absolute": np.mean(np.abs(values)), "rms": np.sqrt(np.mean(values * values)),
        "n_decrease": counts[-1], "n_no_change": counts[0], "n_increase": counts[1],
        "fraction_decrease": counts[-1] / values.size,
        "fraction_no_change": counts[0] / values.size,
        "fraction_increase": counts[1] / values.size,
        "threshold_angstrom": CLASS_THRESHOLD,
    }


def write_task_summary(path, layer_from, layer_to, variant, status, counts, timings, classifier_rows=None, regression_rows=None):
    lines = [
        "Local pair-update classification and regression pilot",
        "=====================================================",
        "",
        f"Status:             {status}",
        f"Transition:         {layer_from} -> {layer_to}",
        f"Feature variant:    {variant}",
        f"Feature dimension:  {feature_dim(variant)}",
        "Distance type:      raw pre-MDS",
        "Distance probes:    saved own-layer probes at both endpoints",
        f"Threshold:          +/-{CLASS_THRESHOLD:.1f} Angstrom",
        f"Train proteins:     {counts.get('train_proteins', 0)}",
        f"Test proteins:      {counts.get('test_proteins', 0)}",
        f"Train pairs:        {counts.get('train_pairs', 0)}",
        f"Test pairs:         {counts.get('test_pairs', 0)}",
        "",
    ]
    for row in classifier_rows or []:
        if row["split"] == "test":
            lines.append(
                "Classifier test:   "
                f"macro_F1={float(row['pair_macro_f1']):.4f}, "
                f"balanced_accuracy={float(row['pair_balanced_accuracy']):.4f}"
            )
    for row in regression_rows or []:
        if row["split"] == "test":
            lines.append(
                "Regression test:   "
                f"pair_R2={float(row['pair_r2']):.4f}, "
                f"protein_mean_R2={float(row['mean_per_protein_r2']):.4f}"
            )
    lines.extend(["", "Stage timings:"])
    for row in timings:
        lines.append(f"  {row['stage']}: {float(row['elapsed_seconds']):.2f} seconds")
    write_text_atomic(path, "\n".join(lines) + "\n")


def main():
    total_t0 = time.time()
    layer_from, layer_to, variant = requested_task_spec()
    csv_root = Path(os.environ.get("CSV_ROOT", Path.home() / "csv_files")).expanduser()
    source_root = csv_root / SOURCE_VARIANT
    coefficient_path = source_root / "coefficients.csv"
    assignment_path = source_root / "split_assignments.csv"
    output_root = Path(
        os.environ.get("LOCAL_PAIR_UPDATE_OUTPUT_ROOT", csv_root / OUTPUT_NAME)
    ).expanduser()
    output_dir = output_root / "shards" / f"transition_{layer_from:02d}_{layer_to:02d}" / variant
    output_dir.mkdir(parents=True, exist_ok=True)

    log(f"Starting transition {layer_from}->{layer_to}, variant={variant}")
    log(f"Output directory: {output_dir}")
    timings = []
    manifest_rows = []
    skipped_rows = []
    classifier_summary_rows = []
    regression_summary_rows = []
    per_protein_rows = []
    class_metric_rows = []
    confusion_rows = []
    coefficient_output_rows = []

    records, split_source_rows = load_split_records(assignment_path)
    intercept_from, distance_coef_from, coefficient_source_rows_from = load_distance_coefficients(
        coefficient_path, layer_from
    )
    intercept_to, distance_coef_to, coefficient_source_rows_to = load_distance_coefficients(
        coefficient_path, layer_to
    )
    coefficient_source_rows = coefficient_source_rows_from + coefficient_source_rows_to
    log(f"Loaded saved distance probes for layers {layer_from} and {layer_to}")
    offsets_by_split = {}
    row_counts = {}
    for split in ("train", "test"):
        offsets_by_split[split], row_counts[split] = build_offsets(records, split)
    counts = {
        "train_proteins": len(offsets_by_split["train"]),
        "test_proteins": len(offsets_by_split["test"]),
        "train_pairs": row_counts["train"],
        "test_pairs": row_counts["test"],
    }
    expected_counts = (441, 111, 812097, 207900)
    actual_counts = (
        counts["train_proteins"], counts["test_proteins"],
        counts["train_pairs"], counts["test_pairs"],
    )
    if actual_counts != expected_counts:
        raise ValueError(f"Frozen split counts {actual_counts} do not match expected {expected_counts}")

    x_by_split = {
        split: np.empty((row_counts[split], feature_dim(variant)), dtype=np.float64)
        for split in ("train", "test")
    }
    target_by_split = {
        split: np.empty(row_counts[split], dtype=np.float64)
        for split in ("train", "test")
    }

    feature_t0 = time.time()
    split_positions = {"train": 0, "test": 0}
    try:
        for record_index, record in enumerate(records, start=1):
            protein_t0 = time.time()
            protein_id = record["protein_id"]
            from_path = pair_block_path(protein_id, layer_from)
            to_path = pair_block_path(protein_id, layer_to)
            try:
                if not from_path.exists() or not to_path.exists():
                    raise FileNotFoundError(f"Missing pair blocks: {from_path}, {to_path}")
                pb_from = np.load(str(from_path))
                pb_to = np.load(str(to_path))
                validate_pair_block(pb_from, record["n_residues"], from_path)
                validate_pair_block(pb_to, record["n_residues"], to_path)
                i, j, separation = local_pair_indices(record["n_residues"])
                distance_from, change = distance_change_and_current(
                    pb_from,
                    pb_to,
                    i,
                    j,
                    intercept_from,
                    distance_coef_from,
                    intercept_to,
                    distance_coef_to,
                )
                x_chunk = build_feature_chunk(
                    pb_from, i, j, separation, variant, distance_from=distance_from
                )
                split = record["split"]
                start = split_positions[split]
                end = start + i.size
                x_by_split[split][start:end] = x_chunk
                target_by_split[split][start:end] = change
                split_positions[split] = end
                manifest_rows.append(
                    {
                        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                        "split": split, "source": record["source"], "protein_id": protein_id,
                        "n_residues": record["n_residues"], "n_local_pairs": i.size,
                        "pair_block_from": str(from_path), "pair_block_to": str(to_path),
                        "status": "success", "elapsed_seconds": time.time() - protein_t0,
                        "message": "",
                    }
                )
                del pb_from, pb_to, i, j, separation, distance_from, change, x_chunk
                gc.collect()
            except Exception as exc:
                skipped_rows.append(
                    {
                        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
                        "split": record["split"], "source": record["source"],
                        "protein_id": protein_id, "reason": type(exc).__name__,
                        "message": str(exc),
                    }
                )
                write_csv_atomic(output_dir / "skipped_proteins.csv", SKIPPED_FIELDS, skipped_rows)
                raise
            if record_index % 10 == 0 or record_index == len(records):
                log(
                    f"Built features for {record_index}/{len(records)} proteins; "
                    f"elapsed={time.time() - feature_t0:.1f}s"
                )
        if split_positions != row_counts:
            raise AssertionError(f"Filled row counts {split_positions}, expected {row_counts}")
    finally:
        write_csv_atomic(output_dir / "protein_manifest.csv", MANIFEST_FIELDS, manifest_rows)
        write_csv_atomic(output_dir / "skipped_proteins.csv", SKIPPED_FIELDS, skipped_rows)

    timings.append(
        {"layer_from": layer_from, "layer_to": layer_to, "variant": variant,
         "stage": "feature_build", "elapsed_seconds": time.time() - feature_t0}
    )
    write_csv_atomic(output_dir / "feature_schema.csv", FEATURE_SCHEMA_FIELDS, feature_schema(variant))
    write_csv_atomic(
        output_dir / "coefficients_used.csv",
        list(coefficient_source_rows[0]),
        coefficient_source_rows,
    )
    write_csv_atomic(
        output_dir / "split_assignments.csv", list(split_source_rows[0]), split_source_rows
    )

    labels_by_split = {split: class_labels(values) for split, values in target_by_split.items()}
    all_values = np.concatenate([target_by_split["train"], target_by_split["test"]])
    change_rows = [
        summarize_change(layer_from, layer_to, "train", target_by_split["train"], counts["train_proteins"]),
        summarize_change(layer_from, layer_to, "test", target_by_split["test"], counts["test_proteins"]),
        summarize_change(layer_from, layer_to, "all", all_values, len(records)),
    ]
    del all_values
    write_csv_atomic(output_dir / "change_statistics.csv", CHANGE_STATISTIC_FIELDS, change_rows)

    scale_t0 = time.time()
    mean, scale = protein_equal_scaler(x_by_split["train"], offsets_by_split["train"])
    for split in ("train", "test"):
        standardize_in_place(x_by_split[split], mean, scale, offsets_by_split[split])
    base_weights = protein_equal_weights(offsets_by_split["train"])
    classifier_weights = balanced_classification_weights(base_weights, labels_by_split["train"])
    alpha = RELATIVE_RIDGE_ALPHA * float(np.sum(base_weights))
    timings.append(
        {"layer_from": layer_from, "layer_to": layer_to, "variant": variant,
         "stage": "standardization", "elapsed_seconds": time.time() - scale_t0}
    )

    metadata = {
        "status": "running_classifier",
        "layer_from": layer_from, "layer_to": layer_to, "variant": variant,
        "feature_dim": feature_dim(variant), "distance_type": "pre_mds_raw",
        "threshold_angstrom": CLASS_THRESHOLD, "sequence_separation_min": MIN_SEQUENCE_SEPARATION,
        "sequence_separation_max": MAX_SEQUENCE_SEPARATION,
        "n_train_proteins": counts["train_proteins"], "n_test_proteins": counts["test_proteins"],
        "n_train_pairs": counts["train_pairs"], "n_test_pairs": counts["test_pairs"],
        "relative_alpha": RELATIVE_RIDGE_ALPHA, "actual_alpha": alpha,
        "solver": "lsqr", "tolerance": LSQR_TOLERANCE, "max_iter": LSQR_MAX_ITER,
        "coefficient_source": str(coefficient_path), "split_source": str(assignment_path),
        "distance_probe_policy": "saved_own_layer",
        "distance_probe_layers": [layer_from, layer_to],
        "feature_dtype": "float64", "started_at": now_timestamp(),
    }
    write_json_atomic(output_dir / "run_metadata.json", metadata)
    write_task_summary(
        output_dir / "summary.txt", layer_from, layer_to, variant,
        "running_classifier", counts, timings,
    )

    classifier_t0 = time.time()
    classifier = RidgeClassifier(
        alpha=alpha, solver="lsqr", tol=LSQR_TOLERANCE, max_iter=LSQR_MAX_ITER,
        # Preserve the standardized matrix for the subsequent regression fit.
        # sklearn centers dense inputs in place when copy_X=False.
        fit_intercept=True, copy_X=True,
    )
    classifier.fit(
        x_by_split["train"], labels_by_split["train"], sample_weight=classifier_weights
    )
    timings.append(
        {"layer_from": layer_from, "layer_to": layer_to, "variant": variant,
         "stage": "classification_fit", "elapsed_seconds": time.time() - classifier_t0}
    )
    classifier_eval_t0 = time.time()
    classifier_predictions = {
        split: classifier.predict(x_by_split[split]) for split in ("train", "test")
    }
    (
        classifier_summary_rows,
        classifier_per_rows,
        class_metric_rows,
        confusion_rows,
    ) = classification_outputs(
        layer_from, layer_to, variant, classifier, labels_by_split,
        classifier_predictions, offsets_by_split, alpha,
    )
    per_protein_rows.extend(classifier_per_rows)
    coefficient_output_rows.extend(
        coefficient_rows(layer_from, layer_to, variant, classifier, "classification", alpha)
    )
    timings.append(
        {"layer_from": layer_from, "layer_to": layer_to, "variant": variant,
         "stage": "classification_evaluation", "elapsed_seconds": time.time() - classifier_eval_t0}
    )
    write_csv_atomic(
        output_dir / "classification_summary.csv",
        CLASSIFICATION_SUMMARY_FIELDS,
        classifier_summary_rows,
    )
    write_csv_atomic(output_dir / "class_metrics.csv", CLASS_METRIC_FIELDS, class_metric_rows)
    write_csv_atomic(output_dir / "confusion_matrix.csv", CONFUSION_FIELDS, confusion_rows)
    write_csv_atomic(output_dir / "per_protein_metrics.csv", PER_PROTEIN_FIELDS, per_protein_rows)
    write_csv_atomic(output_dir / "coefficients.csv", COEFFICIENT_FIELDS, coefficient_output_rows)
    write_csv_atomic(output_dir / "timing.csv", TIMING_FIELDS, timings)
    metadata.update(
        {
            "status": "classifier_complete_running_regression",
            "classifier_n_iter": iteration_info(classifier)[0],
            "classifier_iteration_cap_reached": iteration_info(classifier)[1],
        }
    )
    write_json_atomic(output_dir / "run_metadata.json", metadata)
    write_task_summary(
        output_dir / "summary.txt", layer_from, layer_to, variant,
        "classifier_complete_running_regression", counts, timings,
        classifier_rows=classifier_summary_rows,
    )

    regression_t0 = time.time()
    regressor = Ridge(
        alpha=alpha, solver="lsqr", tol=LSQR_TOLERANCE, max_iter=LSQR_MAX_ITER,
        # Preserve X for correctly evaluated train predictions after fitting.
        fit_intercept=True, copy_X=True,
    )
    regressor.fit(
        x_by_split["train"], target_by_split["train"], sample_weight=base_weights
    )
    timings.append(
        {"layer_from": layer_from, "layer_to": layer_to, "variant": variant,
         "stage": "regression_fit", "elapsed_seconds": time.time() - regression_t0}
    )
    regression_eval_t0 = time.time()
    regression_predictions = {
        split: regressor.predict(x_by_split[split]) for split in ("train", "test")
    }
    regression_summary_rows, regression_per_rows = regression_outputs(
        layer_from, layer_to, variant, regressor, target_by_split,
        regression_predictions, offsets_by_split, alpha,
    )
    per_protein_rows.extend(regression_per_rows)
    coefficient_output_rows.extend(
        coefficient_rows(layer_from, layer_to, variant, regressor, "regression", alpha)
    )
    timings.append(
        {"layer_from": layer_from, "layer_to": layer_to, "variant": variant,
         "stage": "regression_evaluation", "elapsed_seconds": time.time() - regression_eval_t0}
    )
    timings.append(
        {"layer_from": layer_from, "layer_to": layer_to, "variant": variant,
         "stage": "total", "elapsed_seconds": time.time() - total_t0}
    )

    write_csv_atomic(
        output_dir / "regression_summary.csv", REGRESSION_SUMMARY_FIELDS, regression_summary_rows
    )
    write_csv_atomic(output_dir / "per_protein_metrics.csv", PER_PROTEIN_FIELDS, per_protein_rows)
    write_csv_atomic(output_dir / "coefficients.csv", COEFFICIENT_FIELDS, coefficient_output_rows)
    write_csv_atomic(output_dir / "timing.csv", TIMING_FIELDS, timings)
    metadata.update(
        {
            "status": "complete", "completed_at": now_timestamp(),
            "classifier_n_iter": iteration_info(classifier)[0],
            "classifier_iteration_cap_reached": iteration_info(classifier)[1],
            "regression_n_iter": iteration_info(regressor)[0],
            "regression_iteration_cap_reached": iteration_info(regressor)[1],
            "elapsed_seconds": time.time() - total_t0,
        }
    )
    write_json_atomic(output_dir / "run_metadata.json", metadata)
    write_task_summary(
        output_dir / "summary.txt", layer_from, layer_to, variant, "complete",
        counts, timings, classifier_summary_rows, regression_summary_rows,
    )
    log(f"Completed transition {layer_from}->{layer_to}, {variant} in {time.time() - total_t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

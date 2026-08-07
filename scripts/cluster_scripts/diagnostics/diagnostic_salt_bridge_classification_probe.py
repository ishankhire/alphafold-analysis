#!/usr/bin/env python3
"""Layer-47 salt-bridge classification probe for the cached CAMEO-50 annotations.

The probe uses one-to-one within-protein matched controls for fitting and also
evaluates the selected models on the natural pool of chemically eligible
acidic--basic pairs.  It deliberately consumes, but never recomputes or
modifies, the structural annotations.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import json
import math
import os
import sys
import time
import traceback
import warnings
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import scipy
import sklearn
from scipy.optimize import linear_sum_assignment
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    average_precision_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


PROBE_VERSION = "salt_bridge_layer47_v1"
LAYER = 47
RANDOM_STATE = 42
N_CHANNELS = 128
MAX_ITER = 5000
TOLERANCE = 1e-6
C_GRID = (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)
SPLIT_TARGETS = {"train": 30, "validation": 10, "test": 10}
SPLIT_PRIORITY = {"train": 0, "validation": 1, "test": 2}
ACIDIC = frozenset(("ASP", "GLU"))
BASIC = frozenset(("LYS", "ARG"))
MIN_MATCHED_POSITIVES = {"train": 100, "validation": 20, "test": 20}
MIN_CONTRIBUTING_PROTEINS = 5
UNMATCHED_COST = 10_000.0
DEFAULT_DEADLINE_SECONDS = 6_000

VARIANT_ORDER = (
    "ordinary",
    "diagonal_only",
    "cross_only",
    "cross_plus_diagonal",
    "ordinary_plus_cross",
    "ordinary_plus_cross_plus_diagonal",
)
COMBINED_VARIANTS = frozenset(
    ("ordinary_plus_cross", "ordinary_plus_cross_plus_diagonal")
)

ORDINARY_FEATURE_NAMES = [
    "ca_distance",
    "log1p_sequence_separation",
    "acid_is_glu",
    "base_is_lys",
]
CROSS_FEATURE_NAMES = [f"pair_acid_base_ch_{idx}" for idx in range(N_CHANNELS)] + [
    f"pair_base_acid_ch_{idx}" for idx in range(N_CHANNELS)
]
DIAGONAL_FEATURE_NAMES = [f"pair_acid_acid_ch_{idx}" for idx in range(N_CHANNELS)] + [
    f"pair_base_base_ch_{idx}" for idx in range(N_CHANNELS)
]

SUMMARY_FIELDS = [
    "interaction_type",
    "layer",
    "model_variant",
    "split",
    "evaluation_scope",
    "n_proteins",
    "n_metric_proteins",
    "n_pairs",
    "n_positive",
    "n_negative",
    "prevalence",
    "pair_pooled_auroc",
    "pair_pooled_average_precision",
    "mean_per_protein_auroc",
    "mean_per_protein_average_precision",
    "pair_pooled_balanced_accuracy",
    "mean_per_protein_balanced_accuracy",
    "pair_pooled_precision",
    "pair_pooled_recall",
    "selected_C",
    "selected_threshold",
    "gain_over_ordinary_auroc",
    "gain_over_ordinary_average_precision",
]
PER_PROTEIN_FIELDS = [
    "interaction_type",
    "layer",
    "model_variant",
    "split",
    "evaluation_scope",
    "protein_id",
    "n_pairs",
    "n_positive",
    "n_negative",
    "prevalence",
    "auroc",
    "average_precision",
    "balanced_accuracy",
    "precision",
    "recall",
    "ranking_metrics_valid",
    "threshold_metrics_valid",
]
SEARCH_FIELDS = [
    "interaction_type",
    "layer",
    "model_variant",
    "C",
    "validation_mean_per_protein_average_precision",
    "validation_pair_pooled_average_precision",
    "validation_mean_per_protein_auroc",
    "validation_pair_pooled_auroc",
    "converged",
    "n_iterations",
    "fit_seconds",
    "selected",
]
COEFFICIENT_FIELDS = [
    "interaction_type",
    "layer",
    "model_variant",
    "selected_C",
    "feature_index",
    "feature_name",
    "coefficient_standardized",
    "coefficient_raw_scale",
]
PREDICTION_FIELDS = [
    "interaction_type",
    "layer",
    "model_variant",
    "split",
    "evaluation_scope",
    "protein_id",
    "pair_id",
    "i",
    "j",
    "acid_index",
    "base_index",
    "acid_name",
    "base_name",
    "label",
    "predicted_probability",
    "predicted_label",
    "selected_threshold",
    "match_set_id",
    "match_tier",
    "match_cost",
]


def log(message):
    """Print one timestamped progress message."""
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def format_value(value):
    """Return stable CSV-friendly values."""
    if value is None:
        return ""
    if isinstance(value, (np.bool_, bool)):
        return "true" if bool(value) else "false"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if not math.isfinite(value):
            return "nan"
        return f"{value:.8f}"
    return value


def write_csv_atomic(path, fieldnames, rows):
    """Atomically write a CSV from dictionaries."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})
    temporary.replace(path)


def write_json_atomic(path, payload):
    """Atomically write formatted JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    temporary.replace(path)


def stable_hash(payload):
    """Hash JSON-compatible configuration data."""
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def finite_float(value):
    """Parse one required finite float."""
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError("non-finite value")
    return parsed


def separation_bin(value):
    """Map sequence separation to the fixed matching bins."""
    value = int(value)
    if value == 1:
        return 0
    if value == 2:
        return 1
    if value <= 4:
        return 2
    if value <= 8:
        return 3
    if value <= 16:
        return 4
    if value <= 32:
        return 5
    if value <= 64:
        return 6
    return 7


def parse_candidate_row(row):
    """Convert an annotation row to a canonical acid--base candidate record.

    Return ``(candidate, reason)``.  Non-candidates and incomplete ordinary
    covariates are excluded before any model variant is constructed so all
    variants use identical rows. DSSP, SASA, and DSSP-derived annotations are
    deliberately ignored because they are not required to define salt bridges.
    """
    res_i, res_j = row.get("resname_i", ""), row.get("resname_j", "")
    if res_i in ACIDIC and res_j in BASIC:
        acid_side, base_side = "i", "j"
    elif res_j in ACIDIC and res_i in BASIC:
        acid_side, base_side = "j", "i"
    else:
        return None, "not_acidic_basic_candidate"
    if row.get("salt_bridge_valid") != "1" or row.get("salt_bridge") not in {"0", "1"}:
        return None, "invalid_salt_bridge_label"
    try:
        acid_index = int(row[acid_side])
        base_index = int(row[base_side])
        sequence_separation = int(row["sequence_separation"])
        if sequence_separation < 1:
            raise ValueError("invalid sequence separation")
        candidate = {
            "protein_id": row["protein_id"],
            "pair_id": int(row["pair_id"]),
            "n_residues": int(row["n_residues"]),
            "i": int(row["i"]),
            "j": int(row["j"]),
            "acid_index": acid_index,
            "base_index": base_index,
            "acid_name": row[f"resname_{acid_side}"],
            "base_name": row[f"resname_{base_side}"],
            "sequence_separation": sequence_separation,
            "sequence_bin": separation_bin(sequence_separation),
            "ca_distance": finite_float(row["ca_distance"]),
            "label": int(row["salt_bridge"]),
            "match_set_id": "",
            "match_tier": None,
            "match_cost": None,
        }
    except (KeyError, TypeError, ValueError):
        return None, "missing_or_invalid_ordinary_covariate"
    return candidate, None


def load_annotation_records(annotation_root):
    """Stream each all-pairs annotation once and retain compact candidates."""
    manifest_path = annotation_root / "protein_manifest.csv"
    config_path = annotation_root / "annotation_config.json"
    if not manifest_path.exists() or not config_path.exists():
        raise FileNotFoundError(f"annotation cache is incomplete beneath {annotation_root}")
    annotation_config = json.loads(config_path.read_text())
    with manifest_path.open(newline="") as handle:
        manifest_rows = list(csv.DictReader(handle))
    successful = [row for row in manifest_rows if row.get("status") == "success"]
    if len(successful) != sum(SPLIT_TARGETS.values()):
        raise RuntimeError(
            f"expected 50 successful annotation proteins for a 30/10/10 split; got {len(successful)}"
        )
    records = []
    exclusion_rows = []
    for manifest in sorted(successful, key=lambda item: item["protein_id"]):
        protein_id = manifest["protein_id"]
        pair_path = annotation_root / "annotations" / "per_protein" / f"{protein_id}.csv.gz"
        if not pair_path.exists():
            raise FileNotFoundError(f"missing annotation pair file: {pair_path}")
        reason_counts = Counter()
        candidates = []
        with gzip.open(pair_path, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                candidate, reason = parse_candidate_row(row)
                if candidate is None:
                    reason_counts[reason] += 1
                else:
                    candidates.append(candidate)
        for reason, count in sorted(reason_counts.items()):
            exclusion_rows.append(
                {
                    "protein_id": protein_id,
                    "entity_type": "pair_group",
                    "entity_identifier": reason,
                    "reason": reason,
                    "count": count,
                }
            )
        n_residues = int(
            manifest.get("residue_count")
            or (candidates[0]["n_residues"] if candidates else 0)
        )
        records.append(
            {
                "protein_id": protein_id,
                "n_residues": n_residues,
                "annotation_path": pair_path,
                "candidates": candidates,
                "n_candidates": len(candidates),
                "n_positive": sum(item["label"] for item in candidates),
                "n_negative": sum(1 - item["label"] for item in candidates),
            }
        )
        log(
            f"{protein_id}: retained {len(candidates):,} valid acid-base candidates "
            f"({records[-1]['n_positive']} positive)"
        )
    return annotation_config, records, exclusion_rows


def assign_protein_splits(records):
    """Deterministically balance protein count, positives, and residue counts."""
    if len(records) != sum(SPLIT_TARGETS.values()):
        raise ValueError("the fixed split requires exactly 50 proteins")
    total_positive = float(sum(row["n_positive"] for row in records))
    total_residues = float(sum(row["n_residues"] for row in records))
    total_count = float(len(records))
    fractions = {name: target / total_count for name, target in SPLIT_TARGETS.items()}
    target_positive = {name: fractions[name] * total_positive for name in SPLIT_TARGETS}
    target_residues = {name: fractions[name] * total_residues for name in SPLIT_TARGETS}
    state = {
        name: {"count": 0, "positive": 0.0, "residues": 0.0}
        for name in SPLIT_TARGETS
    }

    def objective(hypothetical):
        score = 0.0
        for name in SPLIT_TARGETS:
            values = hypothetical[name]
            score += ((values["count"] - SPLIT_TARGETS[name]) / SPLIT_TARGETS[name]) ** 2
            score += (
                (values["positive"] - target_positive[name])
                / max(1.0, target_positive[name])
            ) ** 2
            score += (
                (values["residues"] - target_residues[name])
                / max(1.0, target_residues[name])
            ) ** 2
        return score

    assignments = {}
    ordered = sorted(
        records,
        key=lambda row: (-row["n_positive"], -row["n_residues"], row["protein_id"]),
    )
    for record in ordered:
        choices = []
        for name in SPLIT_TARGETS:
            if state[name]["count"] >= SPLIT_TARGETS[name]:
                continue
            hypothetical = {key: value.copy() for key, value in state.items()}
            hypothetical[name]["count"] += 1
            hypothetical[name]["positive"] += record["n_positive"]
            hypothetical[name]["residues"] += record["n_residues"]
            choices.append((objective(hypothetical), SPLIT_PRIORITY[name], name, hypothetical))
        _, _, selected, state = min(choices, key=lambda item: (item[0], item[1]))
        assignments[record["protein_id"]] = selected
    counts = Counter(assignments.values())
    if any(counts[name] != target for name, target in SPLIT_TARGETS.items()):
        raise RuntimeError(f"split allocation failed: {dict(counts)}")
    return assignments


def matching_tier(positive, negative):
    """Return the progressive matching tier and within-tier cost."""
    if (positive["acid_name"], positive["base_name"]) != (
        negative["acid_name"],
        negative["base_name"],
    ):
        return None
    ca_diff = abs(positive["ca_distance"] - negative["ca_distance"])
    bin_diff = abs(positive["sequence_bin"] - negative["sequence_bin"])
    specs = (
        (0, 0.5, 0),
        (1, 1.0, 0),
        (2, 1.5, 1),
        (3, 2.0, 1),
    )
    for tier, ca_limit, bin_limit in specs:
        if (
            ca_diff <= ca_limit
            and bin_diff <= bin_limit
        ):
            continuous_cost = (
                ca_diff / ca_limit
                + abs(
                    math.log1p(positive["sequence_separation"])
                    - math.log1p(negative["sequence_separation"])
                )
                / math.log(2.0)
            )
            return tier, 100.0 * tier + continuous_cost
    return None


def standardized_mean_difference(first, second):
    """Compute a stable standardized mean difference."""
    first = np.asarray(first, dtype=np.float64)
    second = np.asarray(second, dtype=np.float64)
    if first.size == 0 or second.size == 0:
        return float("nan")
    pooled = math.sqrt((float(np.var(first)) + float(np.var(second))) / 2.0)
    if pooled <= 1e-12:
        return 0.0 if math.isclose(float(np.mean(first)), float(np.mean(second))) else float("inf")
    return float((np.mean(first) - np.mean(second)) / pooled)


def match_protein_candidates(protein_id, candidates):
    """Perform globally unique one-to-one matching within one protein."""
    positives = [row for row in candidates if row["label"] == 1]
    negatives = [row for row in candidates if row["label"] == 0]
    matched_rows = []
    unmatched_rows = []
    tier_counts = Counter()

    for identity in sorted({(row["acid_name"], row["base_name"]) for row in positives}):
        group_positive = [
            row for row in positives if (row["acid_name"], row["base_name"]) == identity
        ]
        group_negative = [
            row for row in negatives if (row["acid_name"], row["base_name"]) == identity
        ]
        n_positive, n_negative = len(group_positive), len(group_negative)
        if n_positive == 0:
            continue
        costs = np.full(
            (n_positive, n_negative + n_positive), UNMATCHED_COST, dtype=np.float64
        )
        tiers = np.full((n_positive, n_negative), -1, dtype=np.int8)
        for row_index, positive in enumerate(group_positive):
            for column_index, negative in enumerate(group_negative):
                match = matching_tier(positive, negative)
                if match is None:
                    continue
                tier, cost = match
                costs[row_index, column_index] = cost + column_index * 1e-9
                tiers[row_index, column_index] = tier
            costs[row_index, n_negative + row_index] = UNMATCHED_COST - 1.0 + row_index * 1e-9
        selected_rows, selected_columns = linear_sum_assignment(costs)
        assignments = dict(zip(selected_rows.tolist(), selected_columns.tolist()))
        for row_index, positive in enumerate(group_positive):
            column_index = assignments[row_index]
            if column_index >= n_negative or tiers[row_index, column_index] < 0:
                unmatched_rows.append(
                    {
                        "protein_id": protein_id,
                        "pair_id": positive["pair_id"],
                        "reason": "no_eligible_unique_within_protein_control",
                    }
                )
                continue
            negative = group_negative[column_index]
            tier = int(tiers[row_index, column_index])
            cost = float(costs[row_index, column_index])
            match_set_id = f"{protein_id}:salt_bridge:{positive['pair_id']}"
            for role, source in (("positive", positive), ("negative", negative)):
                row = source.copy()
                row.update(
                    {
                        "match_set_id": match_set_id,
                        "match_role": role,
                        "match_tier": tier,
                        "match_cost": cost,
                    }
                )
                matched_rows.append(row)
            tier_counts[tier] += 1

    matched_positive = [row for row in matched_rows if row["label"] == 1]
    matched_negative = [row for row in matched_rows if row["label"] == 0]
    numeric = {
        "ca_distance": lambda row: row["ca_distance"],
        "log1p_sequence_separation": lambda row: math.log1p(row["sequence_separation"]),
    }
    diagnostics = {
        "protein_id": protein_id,
        "n_candidate_positive": len(positives),
        "n_candidate_negative": len(negatives),
        "n_matched_positive": len(matched_positive),
        "n_unmatched_positive": len(unmatched_rows),
        **{f"n_matches_tier_{tier}": tier_counts[tier] for tier in range(4)},
    }
    for name, getter in numeric.items():
        diagnostics[f"smd_before_{name}"] = standardized_mean_difference(
            [getter(row) for row in positives], [getter(row) for row in negatives]
        )
        diagnostics[f"smd_after_{name}"] = standardized_mean_difference(
            [getter(row) for row in matched_positive],
            [getter(row) for row in matched_negative],
        )
    return matched_rows, diagnostics, unmatched_rows


def build_ordinary_features(candidates):
    """Build the intentionally simple linear ordinary-information baseline."""
    result = np.empty((len(candidates), len(ORDINARY_FEATURE_NAMES)), dtype=np.float64)
    for row_index, row in enumerate(candidates):
        result[row_index] = (
            row["ca_distance"],
            math.log1p(row["sequence_separation"]),
            int(row["acid_name"] == "GLU"),
            int(row["base_name"] == "LYS"),
        )
    return result


def extract_representation_features(pair_block, candidates):
    """Extract canonical cross and diagonal features with preallocated writes."""
    acid = np.asarray([row["acid_index"] for row in candidates], dtype=np.int64)
    base = np.asarray([row["base_index"] for row in candidates], dtype=np.int64)
    cross = np.empty((len(candidates), 2 * N_CHANNELS), dtype=np.float64)
    diagonal = np.empty((len(candidates), 2 * N_CHANNELS), dtype=np.float64)
    if candidates:
        cross[:, :N_CHANNELS] = pair_block[acid, base, :]
        cross[:, N_CHANNELS:] = pair_block[base, acid, :]
        diagonal[:, :N_CHANNELS] = pair_block[acid, acid, :]
        diagonal[:, N_CHANNELS:] = pair_block[base, base, :]
    return cross, diagonal


def build_feature_bundle(records, assignments, matched_by_protein, pair_blocks_root, deadline):
    """Read every layer-47 block once and cache only required base feature arrays."""
    chunks = {
        split: {"metadata": [], "ordinary": [], "cross": [], "diagonal": []}
        for split in SPLIT_TARGETS
    }
    for record in sorted(records, key=lambda row: row["protein_id"]):
        protein_id = record["protein_id"]
        split = assignments[protein_id]
        selected = (
            matched_by_protein[protein_id] if split == "train" else record["candidates"]
        )
        block_path = pair_blocks_root / protein_id / f"{protein_id}_pair_block_{LAYER}.npy"
        if not block_path.exists():
            raise FileNotFoundError(f"missing layer-{LAYER} pair block: {block_path}")
        started = time.time()
        pair_block = np.load(block_path)
        expected = (record["n_residues"], record["n_residues"], N_CHANNELS)
        if tuple(pair_block.shape) != expected:
            raise ValueError(
                f"{protein_id}: layer-{LAYER} shape {pair_block.shape} != expected {expected}"
            )
        ordinary = build_ordinary_features(selected)
        cross, diagonal = extract_representation_features(pair_block, selected)
        del pair_block
        match_lookup = {
            row["pair_id"]: row
            for row in matched_by_protein[protein_id]
        }
        metadata = []
        for row in selected:
            item = row.copy()
            matched = match_lookup.get(row["pair_id"])
            item["is_matched"] = matched is not None
            if matched is not None:
                item["match_set_id"] = matched["match_set_id"]
                item["match_tier"] = matched["match_tier"]
                item["match_cost"] = matched["match_cost"]
            metadata.append(item)
        chunks[split]["metadata"].extend(metadata)
        chunks[split]["ordinary"].append(ordinary)
        chunks[split]["cross"].append(cross)
        chunks[split]["diagonal"].append(diagonal)
        log(
            f"{protein_id}: extracted {len(selected):,} layer-{LAYER} examples "
            f"for {split} in {time.time() - started:.1f}s"
        )
        deadline.check(f"feature extraction for {protein_id}")

    bundles = {}
    for split, split_chunks in chunks.items():
        metadata = split_chunks["metadata"]
        bundles[split] = {
            "metadata": metadata,
            "ordinary": np.vstack(split_chunks["ordinary"]),
            "cross": np.vstack(split_chunks["cross"]),
            "diagonal": np.vstack(split_chunks["diagonal"]),
            "y": np.asarray([row["label"] for row in metadata], dtype=np.int8),
            "protein_ids": np.asarray([row["protein_id"] for row in metadata], dtype=object),
            "matched_mask": np.asarray([row["is_matched"] for row in metadata], dtype=bool),
        }
    return bundles


def feature_names_for_variant(variant):
    """Return deterministic transformed feature names."""
    if variant == "ordinary":
        return list(ORDINARY_FEATURE_NAMES)
    if variant == "diagonal_only":
        return list(DIAGONAL_FEATURE_NAMES)
    if variant == "cross_only":
        return list(CROSS_FEATURE_NAMES)
    if variant == "cross_plus_diagonal":
        return list(CROSS_FEATURE_NAMES + DIAGONAL_FEATURE_NAMES)
    if variant == "ordinary_plus_cross":
        return list(ORDINARY_FEATURE_NAMES + CROSS_FEATURE_NAMES)
    if variant == "ordinary_plus_cross_plus_diagonal":
        return list(ORDINARY_FEATURE_NAMES + CROSS_FEATURE_NAMES + DIAGONAL_FEATURE_NAMES)
    raise KeyError(variant)


def matrix_for_variant(bundle, variant):
    """Materialize one model variant from cached base arrays."""
    if variant == "ordinary":
        return bundle["ordinary"]
    if variant == "diagonal_only":
        return bundle["diagonal"]
    if variant == "cross_only":
        return bundle["cross"]
    if variant == "cross_plus_diagonal":
        return np.concatenate((bundle["cross"], bundle["diagonal"]), axis=1)
    if variant == "ordinary_plus_cross":
        return np.concatenate((bundle["ordinary"], bundle["cross"]), axis=1)
    if variant == "ordinary_plus_cross_plus_diagonal":
        return np.concatenate(
            (bundle["ordinary"], bundle["cross"], bundle["diagonal"]), axis=1
        )
    raise KeyError(variant)


def equal_protein_weights(protein_ids):
    """Give each represented protein equal total sample weight."""
    protein_ids = np.asarray(protein_ids, dtype=object)
    counts = Counter(protein_ids.tolist())
    weights = np.asarray([1.0 / counts[item] for item in protein_ids], dtype=np.float64)
    weights *= len(weights) / float(np.sum(weights))
    return weights


def fit_weighted_standardizer(matrix, weights):
    """Fit weighted train-only means and scales."""
    matrix = np.asarray(matrix, dtype=np.float64)
    weights = np.asarray(weights, dtype=np.float64)
    normalized = weights / np.sum(weights)
    mean = np.sum(matrix * normalized[:, None], axis=0)
    variance = np.sum(((matrix - mean) ** 2) * normalized[:, None], axis=0)
    scale = np.sqrt(np.maximum(variance, 0.0))
    scale[scale <= 1e-12] = 1.0
    return mean, scale


def transform_matrix(matrix, mean, scale):
    """Apply a fitted standardizer as float64."""
    return (np.asarray(matrix, dtype=np.float64) - mean) / scale


def safe_auroc(labels, probabilities):
    labels = np.asarray(labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(roc_auc_score(labels, probabilities))


def safe_average_precision(labels, probabilities):
    labels = np.asarray(labels)
    if labels.size == 0 or np.unique(labels).size < 2:
        return float("nan")
    return float(average_precision_score(labels, probabilities))


def metric_value_rows(labels, probabilities, protein_ids, threshold=None):
    """Compute one metric row per protein."""
    rows = []
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    protein_ids = np.asarray(protein_ids, dtype=object)
    for protein_id in sorted(set(protein_ids.tolist())):
        mask = protein_ids == protein_id
        local_y, local_probability = labels[mask], probabilities[mask]
        ranking_valid = np.unique(local_y).size == 2
        row = {
            "protein_id": protein_id,
            "n_pairs": int(mask.sum()),
            "n_positive": int(np.sum(local_y == 1)),
            "n_negative": int(np.sum(local_y == 0)),
            "prevalence": float(np.mean(local_y)) if local_y.size else float("nan"),
            "auroc": safe_auroc(local_y, local_probability),
            "average_precision": safe_average_precision(local_y, local_probability),
            "balanced_accuracy": float("nan"),
            "precision": float("nan"),
            "recall": float("nan"),
            "ranking_metrics_valid": bool(ranking_valid),
            "threshold_metrics_valid": False,
        }
        if threshold is not None and ranking_valid:
            predicted = (local_probability >= threshold).astype(np.int8)
            row.update(
                balanced_accuracy=float(balanced_accuracy_score(local_y, predicted)),
                precision=float(precision_score(local_y, predicted, zero_division=0)),
                recall=float(recall_score(local_y, predicted, zero_division=0)),
                threshold_metrics_valid=True,
            )
        rows.append(row)
    return rows


def finite_mean(values):
    values = np.asarray(list(values), dtype=np.float64)
    values = values[np.isfinite(values)]
    return float(np.mean(values)) if values.size else float("nan")


def summarize_metrics(labels, probabilities, protein_ids, threshold=None):
    """Compute pooled and macro metrics plus per-protein rows."""
    labels = np.asarray(labels, dtype=np.int8)
    probabilities = np.asarray(probabilities, dtype=np.float64)
    protein_ids = np.asarray(protein_ids, dtype=object)
    protein_rows = metric_value_rows(labels, probabilities, protein_ids, threshold)
    summary = {
        "n_proteins": len(set(protein_ids.tolist())),
        "n_metric_proteins": sum(row["ranking_metrics_valid"] for row in protein_rows),
        "n_pairs": int(labels.size),
        "n_positive": int(np.sum(labels == 1)),
        "n_negative": int(np.sum(labels == 0)),
        "prevalence": float(np.mean(labels)) if labels.size else float("nan"),
        "pair_pooled_auroc": safe_auroc(labels, probabilities),
        "pair_pooled_average_precision": safe_average_precision(labels, probabilities),
        "mean_per_protein_auroc": finite_mean(row["auroc"] for row in protein_rows),
        "mean_per_protein_average_precision": finite_mean(
            row["average_precision"] for row in protein_rows
        ),
        "pair_pooled_balanced_accuracy": float("nan"),
        "mean_per_protein_balanced_accuracy": finite_mean(
            row["balanced_accuracy"] for row in protein_rows
        ),
        "pair_pooled_precision": float("nan"),
        "pair_pooled_recall": float("nan"),
    }
    if threshold is not None and np.unique(labels).size == 2:
        predicted = (probabilities >= threshold).astype(np.int8)
        summary.update(
            pair_pooled_balanced_accuracy=float(
                balanced_accuracy_score(labels, predicted)
            ),
            pair_pooled_precision=float(
                precision_score(labels, predicted, zero_division=0)
            ),
            pair_pooled_recall=float(recall_score(labels, predicted, zero_division=0)),
        )
    return summary, protein_rows


def choose_threshold(labels, probabilities, protein_ids):
    """Select the threshold maximizing macro per-protein balanced accuracy."""
    probabilities = np.asarray(probabilities, dtype=np.float64)
    unique = np.unique(probabilities)
    if unique.size == 0:
        raise ValueError("cannot select a threshold from empty validation predictions")
    candidates = np.unique(np.concatenate(([0.0, 0.5, 1.0], unique)))
    best = None
    for threshold in candidates:
        _, protein_rows = summarize_metrics(labels, probabilities, protein_ids, threshold)
        score = finite_mean(row["balanced_accuracy"] for row in protein_rows)
        key = (-score, abs(float(threshold) - 0.5), -float(threshold))
        if best is None or key < best[0]:
            best = (key, float(threshold), score)
    return best[1], best[2]


def fit_variant(
    variant,
    train_matrix,
    validation_matrix,
    train_bundle,
    validation_bundle,
    deadline,
):
    """Scale, search C, and return the selected train-only model."""
    train_weights = equal_protein_weights(train_bundle["protein_ids"])
    mean, scale = fit_weighted_standardizer(train_matrix, train_weights)
    train_scaled = transform_matrix(train_matrix, mean, scale)
    validation_scaled = transform_matrix(validation_matrix, mean, scale)
    validation_mask = validation_bundle["matched_mask"]
    validation_y = validation_bundle["y"][validation_mask]
    validation_proteins = validation_bundle["protein_ids"][validation_mask]
    search_rows = []
    best = None
    for c_value in C_GRID:
        started = time.time()
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(
                C=c_value,
                solver="lbfgs",
                max_iter=MAX_ITER,
                tol=TOLERANCE,
                random_state=RANDOM_STATE,
            )
            model.fit(train_scaled, train_bundle["y"], sample_weight=train_weights)
        converged = not any(issubclass(item.category, ConvergenceWarning) for item in caught)
        probabilities = model.predict_proba(validation_scaled[validation_mask])[:, 1]
        summary, _ = summarize_metrics(
            validation_y, probabilities, validation_proteins, threshold=None
        )
        macro_ap = summary["mean_per_protein_average_precision"]
        row = {
            "interaction_type": "salt_bridge",
            "layer": LAYER,
            "model_variant": variant,
            "C": c_value,
            "validation_mean_per_protein_average_precision": macro_ap,
            "validation_pair_pooled_average_precision": summary[
                "pair_pooled_average_precision"
            ],
            "validation_mean_per_protein_auroc": summary["mean_per_protein_auroc"],
            "validation_pair_pooled_auroc": summary["pair_pooled_auroc"],
            "converged": converged,
            "n_iterations": int(model.n_iter_[0]),
            "fit_seconds": time.time() - started,
            "selected": False,
        }
        search_rows.append(row)
        if not math.isfinite(macro_ap):
            continue
        selection_key = (-macro_ap, c_value)
        if best is None or selection_key < best[0]:
            best = (selection_key, model, c_value, probabilities)
        log(
            f"{variant}: C={c_value:g}, validation macro AP={macro_ap:.4f}, "
            f"iterations={model.n_iter_[0]}, converged={converged}"
        )
        deadline.check(f"{variant} C={c_value:g}")
    if best is None:
        raise RuntimeError(f"{variant}: no finite validation macro AP")
    _, selected_model, selected_c, selected_validation_probability = best
    for row in search_rows:
        row["selected"] = math.isclose(row["C"], selected_c)
    selected_threshold, threshold_score = choose_threshold(
        validation_y, selected_validation_probability, validation_proteins
    )
    log(
        f"{variant}: selected C={selected_c:g}, threshold={selected_threshold:.6f}, "
        f"validation macro balanced accuracy={threshold_score:.4f}"
    )
    return {
        "model": selected_model,
        "selected_C": selected_c,
        "threshold": selected_threshold,
        "mean": mean,
        "scale": scale,
        "train_scaled": train_scaled,
        "validation_scaled": validation_scaled,
        "search_rows": search_rows,
    }


def evaluation_slices(bundle, split):
    """Return requested evaluation scopes for one split."""
    if split == "train":
        return (("matched", np.ones(len(bundle["y"]), dtype=bool), True),)
    return (
        ("matched", bundle["matched_mask"], True),
        ("natural_candidate_pool", np.ones(len(bundle["y"]), dtype=bool), False),
    )


def add_evaluation_rows(
    variant,
    split,
    bundle,
    probabilities,
    selected_c,
    threshold,
    summary_rows,
    per_protein_rows,
):
    """Append summary and per-protein evaluation rows."""
    for scope, mask, thresholded in evaluation_slices(bundle, split):
        local_threshold = threshold if thresholded else None
        summary, protein_metrics = summarize_metrics(
            bundle["y"][mask],
            probabilities[mask],
            bundle["protein_ids"][mask],
            local_threshold,
        )
        summary_rows.append(
            {
                "interaction_type": "salt_bridge",
                "layer": LAYER,
                "model_variant": variant,
                "split": split,
                "evaluation_scope": scope,
                **summary,
                "selected_C": selected_c,
                "selected_threshold": threshold if thresholded else None,
                "gain_over_ordinary_auroc": None,
                "gain_over_ordinary_average_precision": None,
            }
        )
        for row in protein_metrics:
            per_protein_rows.append(
                {
                    "interaction_type": "salt_bridge",
                    "layer": LAYER,
                    "model_variant": variant,
                    "split": split,
                    "evaluation_scope": scope,
                    **row,
                }
            )


def coefficient_rows(variant, fitted, feature_names):
    """Build standardized and transformed raw-scale coefficient rows."""
    coefficients = np.asarray(fitted["model"].coef_[0], dtype=np.float64)
    raw_coefficients = coefficients / fitted["scale"]
    raw_intercept = float(
        fitted["model"].intercept_[0]
        - np.sum(coefficients * fitted["mean"] / fitted["scale"])
    )
    rows = [
        {
            "interaction_type": "salt_bridge",
            "layer": LAYER,
            "model_variant": variant,
            "selected_C": fitted["selected_C"],
            "feature_index": -1,
            "feature_name": "intercept",
            "coefficient_standardized": float(fitted["model"].intercept_[0]),
            "coefficient_raw_scale": raw_intercept,
        }
    ]
    for index, name in enumerate(feature_names):
        rows.append(
            {
                "interaction_type": "salt_bridge",
                "layer": LAYER,
                "model_variant": variant,
                "selected_C": fitted["selected_C"],
                "feature_index": index,
                "feature_name": name,
                "coefficient_standardized": coefficients[index],
                "coefficient_raw_scale": raw_coefficients[index],
            }
        )
    return rows


def write_prediction_shard(path, variant, bundles, probabilities, threshold):
    """Write validation/test predictions for one completed model variant."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for split in ("validation", "test"):
            bundle = bundles[split]
            for scope, mask, thresholded in evaluation_slices(bundle, split):
                for index in np.flatnonzero(mask):
                    meta = bundle["metadata"][int(index)]
                    probability = float(probabilities[split][index])
                    writer.writerow(
                        {
                            "interaction_type": "salt_bridge",
                            "layer": LAYER,
                            "model_variant": variant,
                            "split": split,
                            "evaluation_scope": scope,
                            "protein_id": meta["protein_id"],
                            "pair_id": meta["pair_id"],
                            "i": meta["i"],
                            "j": meta["j"],
                            "acid_index": meta["acid_index"],
                            "base_index": meta["base_index"],
                            "acid_name": meta["acid_name"],
                            "base_name": meta["base_name"],
                            "label": meta["label"],
                            "predicted_probability": format_value(probability),
                            "predicted_label": (
                                int(probability >= threshold) if thresholded else ""
                            ),
                            "selected_threshold": (
                                format_value(threshold) if thresholded else ""
                            ),
                            "match_set_id": meta.get("match_set_id", ""),
                            "match_tier": format_value(meta.get("match_tier")),
                            "match_cost": format_value(meta.get("match_cost")),
                        }
                    )
    temporary.replace(path)


def merge_prediction_shards(shard_paths, destination):
    """Merge checkpointed gzip prediction shards into the stable output."""
    temporary = destination.with_suffix(destination.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=PREDICTION_FIELDS)
        writer.writeheader()
        for path in shard_paths:
            with gzip.open(path, "rt", newline="") as source:
                for row in csv.DictReader(source):
                    writer.writerow(row)
    temporary.replace(destination)


def add_ordinary_gains(summary_rows):
    """Populate held-out gains for the two combined variants."""
    ordinary = {
        (row["split"], row["evaluation_scope"]): row
        for row in summary_rows
        if row["model_variant"] == "ordinary"
    }
    for row in summary_rows:
        if row["model_variant"] not in COMBINED_VARIANTS:
            continue
        baseline = ordinary.get((row["split"], row["evaluation_scope"]))
        if baseline is None:
            continue
        row["gain_over_ordinary_auroc"] = (
            row["pair_pooled_auroc"] - baseline["pair_pooled_auroc"]
        )
        row["gain_over_ordinary_average_precision"] = (
            row["pair_pooled_average_precision"]
            - baseline["pair_pooled_average_precision"]
        )


def write_comparison_plot(summary_rows, destination):
    """Plot layer-47 held-out AUROC and AP for matched and natural scopes."""
    lookup = {
        (row["model_variant"], row["evaluation_scope"]): row
        for row in summary_rows
        if row["split"] == "test"
    }
    labels = list(VARIANT_ORDER)
    x_values = np.arange(len(labels))
    figure, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True)
    for column, scope in enumerate(("matched", "natural_candidate_pool")):
        for row_index, (field, title) in enumerate(
            (("pair_pooled_auroc", "AUROC"), ("pair_pooled_average_precision", "Average precision"))
        ):
            values = [lookup[(variant, scope)][field] for variant in labels]
            axes[row_index, column].bar(x_values, values, color="#4c78a8")
            axes[row_index, column].set_ylim(0.0, 1.0)
            axes[row_index, column].set_ylabel(title)
            axes[row_index, column].set_title(scope.replace("_", " ").title())
            axes[row_index, column].grid(axis="y", alpha=0.25)
    for axis in axes[-1]:
        axis.set_xticks(x_values)
        axis.set_xticklabels(labels, rotation=35, ha="right")
    figure.suptitle("Layer-47 salt-bridge classification on held-out proteins")
    figure.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(destination, dpi=180)
    plt.close(figure)


class Deadline:
    """Internal wall-clock guard that leaves time for Slurm output flushing."""

    def __init__(self, seconds):
        self.started = time.time()
        self.seconds = float(seconds)

    def check(self, stage):
        elapsed = time.time() - self.started
        if elapsed > self.seconds:
            raise TimeoutError(
                f"internal deadline exceeded after {elapsed:.1f}s at {stage}; "
                "completed checkpoint outputs are preserved"
            )


def matching_assignment_rows(matched_by_protein, assignments):
    fields = [
        "interaction_type",
        "protein_id",
        "split",
        "match_set_id",
        "match_role",
        "label",
        "pair_id",
        "i",
        "j",
        "acid_index",
        "base_index",
        "acid_name",
        "base_name",
        "ca_distance",
        "sequence_separation",
        "sequence_bin",
        "match_tier",
        "match_cost",
    ]
    rows = []
    for protein_id in sorted(matched_by_protein):
        for row in matched_by_protein[protein_id]:
            rows.append(
                {
                    "interaction_type": "salt_bridge",
                    "protein_id": protein_id,
                    "split": assignments[protein_id],
                    **row,
                }
            )
    return fields, rows


def write_gzip_csv_atomic(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with gzip.open(temporary, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})
    temporary.replace(path)


def build_probe_config(annotation_config, annotation_root, pair_blocks_root):
    """Build the complete immutable classification configuration."""
    config = {
        "probe_version": PROBE_VERSION,
        "annotation_config_hash": annotation_config.get("config_hash"),
        "layer": LAYER,
        "split_targets": SPLIT_TARGETS,
        "split_balance_fields": ["valid_salt_bridge_positives", "n_residues"],
        "negative_policy": "one_to_one_unique_within_protein_acidic_basic",
        "matching_tiers": [
            {"tier": 0, "sequence_bin_delta": 0, "ca_delta": 0.5},
            {"tier": 1, "sequence_bin_delta": 0, "ca_delta": 1.0},
            {"tier": 2, "sequence_bin_delta": 1, "ca_delta": 1.5},
            {"tier": 3, "sequence_bin_delta": 1, "ca_delta": 2.0},
        ],
        "ordinary_baseline": {
            "design": "simple_linear_salt_only_no_dssp_or_sasa",
            "features": ORDINARY_FEATURE_NAMES,
        },
        "ignored_annotation_fields": [
            "dssp_code_i",
            "dssp_code_j",
            "dssp_3state_i",
            "dssp_3state_j",
            "absolute_sasa_i",
            "absolute_sasa_j",
            "relative_sasa_i",
            "relative_sasa_j",
            "backbone_hbond",
            "beta_bridge",
            "hydrophobic_core_contact",
        ],
        "variants": {name: feature_names_for_variant(name) for name in VARIANT_ORDER},
        "training_weighting": "equal_total_weight_per_protein",
        "regularization": "L2",
        "solver": "lbfgs",
        "max_iter": MAX_ITER,
        "tolerance": TOLERANCE,
        "C_grid": C_GRID,
        "selection_metric": "validation_mean_per_protein_average_precision",
        "threshold_metric": "validation_mean_per_protein_balanced_accuracy",
        "natural_pool_threshold_metrics": False,
        "random_state": RANDOM_STATE,
        "pair_blocks_root": str(pair_blocks_root),
        "annotation_root": str(annotation_root),
        "software": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    config["classification_config_hash"] = stable_hash(config)
    return config


def main():
    deadline = Deadline(float(os.environ.get("PROBE_DEADLINE_SECONDS", DEFAULT_DEADLINE_SECONDS)))
    annotation_root = Path(
        os.environ.get(
            "CSV_ROOT",
            str(Path.home() / "csv_files" / "pair_interaction_classification" / "cameo50_v1"),
        )
    ).resolve()
    pair_blocks_root = Path(
        os.environ.get("PAIR_BLOCKS_ROOT", "/work/hdd/bekh/thayes/openfold_reps")
    ).resolve()
    output_dir = annotation_root / "classification" / "salt_bridge" / "layer_47"
    checkpoint_dir = output_dir / "prediction_shards"
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_dir.mkdir(parents=True, exist_ok=True)

    log(f"annotation root: {annotation_root}")
    log(f"pair-block root: {pair_blocks_root}")
    log(f"output directory: {output_dir}")
    log(f"internal deadline: {deadline.seconds:.0f}s")

    annotation_config, records, exclusion_rows = load_annotation_records(annotation_root)
    deadline.check("annotation loading")
    probe_config = build_probe_config(
        annotation_config, annotation_root, pair_blocks_root
    )
    config_path = output_dir / "classification_config.json"
    if config_path.exists():
        existing = json.loads(config_path.read_text())
        if existing.get("classification_config_hash") != probe_config["classification_config_hash"]:
            raise RuntimeError("classification output configuration differs; refusing to mix results")
    write_json_atomic(config_path, probe_config)

    assignments = assign_protein_splits(records)
    log(f"protein split counts: {dict(Counter(assignments.values()))}")

    matched_by_protein = {}
    matching_diagnostics = []
    unmatched = []
    for record in records:
        protein_id = record["protein_id"]
        matched, diagnostics, local_unmatched = match_protein_candidates(
            protein_id, record["candidates"]
        )
        matched_by_protein[protein_id] = matched
        diagnostics["split"] = assignments[protein_id]
        matching_diagnostics.append(diagnostics)
        unmatched.extend(local_unmatched)
        log(
            f"{protein_id}: matched {diagnostics['n_matched_positive']} / "
            f"{diagnostics['n_candidate_positive']} positives"
        )
    deadline.check("matching")

    split_rows = []
    for record in sorted(records, key=lambda row: row["protein_id"]):
        protein_id = record["protein_id"]
        matched_positive = sum(
            row["label"] for row in matched_by_protein[protein_id]
        )
        split_rows.append(
            {
                "protein_id": protein_id,
                "split": assignments[protein_id],
                "n_residues": record["n_residues"],
                "n_natural_candidates": record["n_candidates"],
                "n_natural_positive": record["n_positive"],
                "n_natural_negative": record["n_negative"],
                "n_matched_positive": matched_positive,
                "n_matched_negative": matched_positive,
                "contributes_matched_examples": matched_positive > 0,
            }
        )
    write_csv_atomic(
        output_dir / "split_assignments.csv",
        [
            "protein_id",
            "split",
            "n_residues",
            "n_natural_candidates",
            "n_natural_positive",
            "n_natural_negative",
            "n_matched_positive",
            "n_matched_negative",
            "contributes_matched_examples",
        ],
        split_rows,
    )
    diagnostic_fields = list(matching_diagnostics[0].keys())
    write_csv_atomic(output_dir / "matching_diagnostics.csv", diagnostic_fields, matching_diagnostics)
    match_fields, match_rows = matching_assignment_rows(matched_by_protein, assignments)
    write_gzip_csv_atomic(
        output_dir / "matched_pair_assignments.csv.gz", match_fields, match_rows
    )
    exclusion_rows.extend(
        {
            "protein_id": row["protein_id"],
            "entity_type": "positive_pair",
            "entity_identifier": row["pair_id"],
            "reason": row["reason"],
            "count": 1,
        }
        for row in unmatched
    )
    write_csv_atomic(
        output_dir / "classification_exclusions.csv",
        ["protein_id", "entity_type", "entity_identifier", "reason", "count"],
        exclusion_rows,
    )

    for split, minimum in MIN_MATCHED_POSITIVES.items():
        selected = [row for row in split_rows if row["split"] == split]
        matched_count = sum(row["n_matched_positive"] for row in selected)
        contributing = sum(row["contributes_matched_examples"] for row in selected)
        if matched_count < minimum or contributing < MIN_CONTRIBUTING_PROTEINS:
            raise RuntimeError(
                f"{split}: only {matched_count} matched positives from {contributing} proteins; "
                f"requires at least {minimum} positives and {MIN_CONTRIBUTING_PROTEINS} proteins"
            )

    bundles = build_feature_bundle(
        records, assignments, matched_by_protein, pair_blocks_root, deadline
    )
    log(
        "feature row counts: "
        + ", ".join(f"{split}={len(bundle['y']):,}" for split, bundle in bundles.items())
    )

    summary_rows = []
    per_protein_rows = []
    search_rows = []
    all_coefficient_rows = []
    shard_paths = []
    selected_settings = {}
    for variant in VARIANT_ORDER:
        variant_started = time.time()
        log(f"{variant}: model search started")
        train_matrix = matrix_for_variant(bundles["train"], variant)
        validation_matrix = matrix_for_variant(bundles["validation"], variant)
        test_matrix = matrix_for_variant(bundles["test"], variant)
        fitted = fit_variant(
            variant,
            train_matrix,
            validation_matrix,
            bundles["train"],
            bundles["validation"],
            deadline,
        )
        test_scaled = transform_matrix(test_matrix, fitted["mean"], fitted["scale"])
        probabilities = {
            "train": fitted["model"].predict_proba(fitted["train_scaled"])[:, 1],
            "validation": fitted["model"].predict_proba(fitted["validation_scaled"])[:, 1],
            "test": fitted["model"].predict_proba(test_scaled)[:, 1],
        }
        for split in SPLIT_TARGETS:
            add_evaluation_rows(
                variant,
                split,
                bundles[split],
                probabilities[split],
                fitted["selected_C"],
                fitted["threshold"],
                summary_rows,
                per_protein_rows,
            )
        search_rows.extend(fitted["search_rows"])
        all_coefficient_rows.extend(
            coefficient_rows(variant, fitted, feature_names_for_variant(variant))
        )
        selected_settings[variant] = {
            "C": fitted["selected_C"],
            "threshold": fitted["threshold"],
        }
        shard_path = checkpoint_dir / f"{variant}.csv.gz"
        write_prediction_shard(
            shard_path, variant, bundles, probabilities, fitted["threshold"]
        )
        shard_paths.append(shard_path)
        add_ordinary_gains(summary_rows)
        write_csv_atomic(output_dir / "regularization_search.csv", SEARCH_FIELDS, search_rows)
        write_csv_atomic(output_dir / "interaction_type_summary.csv", SUMMARY_FIELDS, summary_rows)
        write_csv_atomic(
            output_dir / "interaction_type_per_protein.csv",
            PER_PROTEIN_FIELDS,
            per_protein_rows,
        )
        write_csv_atomic(
            output_dir / "coefficients.csv", COEFFICIENT_FIELDS, all_coefficient_rows
        )
        log(f"{variant}: completed and checkpointed in {time.time() - variant_started:.1f}s")
        del train_matrix, validation_matrix, test_matrix, test_scaled, probabilities
        deadline.check(f"completed variant {variant}")

    add_ordinary_gains(summary_rows)
    write_csv_atomic(output_dir / "interaction_type_summary.csv", SUMMARY_FIELDS, summary_rows)
    merge_prediction_shards(shard_paths, output_dir / "predictions.csv.gz")
    write_comparison_plot(
        summary_rows, output_dir / "visualizations" / "layer47_model_comparison.png"
    )

    elapsed = time.time() - deadline.started
    lines = [
        "Layer-47 salt-bridge classification probe",
        f"probe_version={PROBE_VERSION}",
        f"classification_config_hash={probe_config['classification_config_hash']}",
        f"annotation_config_hash={annotation_config.get('config_hash', '')}",
        "interpretation=interactions in OpenFold unrelaxed predicted structures",
        "ordinary_baseline_design=simple linear amino-acid identity, C-alpha distance, and sequence separation",
        "dssp_sasa_policy=ignored; unavailable DSSP-derived fields do not affect salt-bridge eligibility",
        "training_data=one-to-one within-protein matched acid-base pairs",
        "test_scopes=matched,natural_candidate_pool",
        f"runtime_seconds={elapsed:.1f}",
    ]
    for split in SPLIT_TARGETS:
        local = [row for row in split_rows if row["split"] == split]
        lines.append(f"{split}.proteins={len(local)}")
        lines.append(
            f"{split}.matched_positives={sum(row['n_matched_positive'] for row in local)}"
        )
    for variant in VARIANT_ORDER:
        lines.append(f"{variant}.selected_C={selected_settings[variant]['C']}")
        lines.append(f"{variant}.selected_threshold={selected_settings[variant]['threshold']}")
        for scope in ("matched", "natural_candidate_pool"):
            row = next(
                item
                for item in summary_rows
                if item["model_variant"] == variant
                and item["split"] == "test"
                and item["evaluation_scope"] == scope
            )
            lines.append(f"{variant}.test.{scope}.auroc={row['pair_pooled_auroc']}")
            lines.append(
                f"{variant}.test.{scope}.average_precision={row['pair_pooled_average_precision']}"
            )
    (output_dir / "summary.txt").write_text("\n".join(lines) + "\n")
    log(f"classification probe complete in {elapsed:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

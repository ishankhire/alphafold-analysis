#!/usr/bin/env python3
"""
piecewise_linear_regression.py
==============================
All-layer fixed-threshold piecewise linear regression diagnostic for AlphaFold
pair representations with per-protein geometry analysis.

Requested variants (all with protein-equal training weights):
1. lower_triangle_center_only_piecewise_linear
2. lower_triangle_plus_axial_context_piecewise_linear

Model policy for every variant and every layer:
- route each protein's lower-triangle residue-pair rows to a fixed leaf region
  using raw n_residues only
- fit one ordinary linear regression per final region (LL, LR, RL, RR)
- exclude n_residues from the fitted linear features
- evaluate by routing every row to its region's fitted model

Outputs (inside CSV_ROOT/):
    piecewise_linear_regression_variant_summary.csv
    piecewise_linear_regression_per_protein.csv
    piecewise_linear_regression_split.csv
    piecewise_linear_regression_rules.csv
    piecewise_linear_regression_coefficients_raw.csv
    piecewise_linear_regression_region_diagnostics.csv
    piecewise_linear_regression_test_scatter_sample.csv
    piecewise_linear_regression_geometry_metrics.csv
    piecewise_linear_regression_geometry_summary.csv
    piecewise_linear_regression_summary.txt
"""

import csv
import math
import os
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from cluster_regression_helpers import (  # noqa: E402
    analyze_protein_geometry,
    compute_distance_matrix,
    discover_proteins,
    extract_ca_coordinates,
    find_pdb_file,
    get_pair_block_path,
)


N_LAYERS = 48
N_CHANNELS = 128
TEST_SIZE = 0.2
RANDOM_STATE = 42
FLOAT_EPS = 1e-12
SCATTER_SAMPLE_LAYER = 47
SCATTER_SAMPLE_PER_VARIANT = 100_000

CENTER_OFFSETS = [(0, 0)]
AXIAL_OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

PAIR_BLOCKS_ROOT = Path(os.environ["PAIR_BLOCKS_ROOT"])
PDB_ROOT = Path(os.environ["PDB_ROOT"])
CSV_DIR = Path(os.environ["CSV_ROOT"])

SUMMARY_CSV_NAME = "piecewise_linear_regression_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "piecewise_linear_regression_per_protein.csv"
SPLIT_CSV_NAME = "piecewise_linear_regression_split.csv"
RULES_CSV_NAME = "piecewise_linear_regression_rules.csv"
COEF_CSV_NAME = "piecewise_linear_regression_coefficients_raw.csv"
REGION_DIAGNOSTICS_CSV_NAME = "piecewise_linear_regression_region_diagnostics.csv"
SCATTER_SAMPLE_CSV_NAME = "piecewise_linear_regression_test_scatter_sample.csv"
GEOMETRY_METRICS_CSV_NAME = "piecewise_linear_regression_geometry_metrics.csv"
GEOMETRY_SUMMARY_CSV_NAME = "piecewise_linear_regression_geometry_summary.csv"
SUMMARY_TXT_NAME = "piecewise_linear_regression_summary.txt"


def now_timestamp():
    """Return a readable local timestamp for progress logs."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _feature_dim_for_slots(n_slots, bidirectional):
    """Return feature width including final n_residues split column."""
    slot_dim = 2 * N_CHANNELS if bidirectional else N_CHANNELS
    return (n_slots * slot_dim) + 1


def make_region(label, min_exclusive, max_inclusive, path_rule):
    """Create one fixed leaf-region specification."""
    return {
        "label": label,
        "node_path": label,
        "min_exclusive": min_exclusive,
        "max_inclusive": max_inclusive,
        "path_rule": path_rule,
    }


def build_variant_specs():
    """Define requested fixed-threshold context variants."""
    return [
        {
            "name": "lower_triangle_center_only_piecewise_linear",
            "feature_mode": "lower_triangle_center_only",
            "offsets": CENTER_OFFSETS,
            "bidirectional": False,
            "feature_dim": _feature_dim_for_slots(len(CENTER_OFFSETS), False),
            "root_threshold": 371.5,
            "left_child_threshold": 255.5,
            "right_child_threshold": 503.0,
            "regions": [
                make_region(
                    "LL",
                    None,
                    255.5,
                    "n_residues <= 371.500000 AND n_residues <= 255.500000",
                ),
                make_region(
                    "LR",
                    255.5,
                    371.5,
                    "n_residues <= 371.500000 AND n_residues > 255.500000",
                ),
                make_region(
                    "RL",
                    371.5,
                    503.0,
                    "n_residues > 371.500000 AND n_residues <= 503.000000",
                ),
                make_region(
                    "RR",
                    503.0,
                    None,
                    "n_residues > 371.500000 AND n_residues > 503.000000",
                ),
            ],
        },
        {
            "name": "lower_triangle_plus_axial_context_piecewise_linear",
            "feature_mode": "lower_triangle_plus_axial_context",
            "offsets": AXIAL_OFFSETS,
            "bidirectional": False,
            "feature_dim": _feature_dim_for_slots(len(AXIAL_OFFSETS), False),
            "root_threshold": 371.5,
            "left_child_threshold": 232.5,
            "right_child_threshold": 503.0,
            "regions": [
                make_region(
                    "LL",
                    None,
                    232.5,
                    "n_residues <= 371.500000 AND n_residues <= 232.500000",
                ),
                make_region(
                    "LR",
                    232.5,
                    371.5,
                    "n_residues <= 371.500000 AND n_residues > 232.500000",
                ),
                make_region(
                    "RL",
                    371.5,
                    503.0,
                    "n_residues > 371.500000 AND n_residues <= 503.000000",
                ),
                make_region(
                    "RR",
                    503.0,
                    None,
                    "n_residues > 371.500000 AND n_residues > 503.000000",
                ),
            ],
        },
    ]


VARIANTS = build_variant_specs()
for variant in VARIANTS:
    variant["split_feature_index"] = variant["feature_dim"] - 1


def resolve_pdb_file(protein_id, protein_dir):
    """Prefer cluster PDB layout, with a local-repo fallback for smoke tests."""
    pdb_path = find_pdb_file(protein_id)
    if pdb_path is not None and pdb_path.exists():
        return pdb_path

    local_path = protein_dir / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    return local_path if local_path.exists() else None


def resolve_pair_block_path(protein_dir, protein_id, layer):
    """Prefer cluster pair-block layout, with a local-repo fallback."""
    cluster_path = get_pair_block_path(protein_dir, protein_id, layer)
    if cluster_path.exists():
        return cluster_path

    local_path = protein_dir / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy"
    if local_path.exists():
        return local_path
    return cluster_path


def split_global_records(records):
    """Create one deterministic protein-level train/test split."""
    if len(records) < 2:
        raise ValueError("Need at least two proteins to split into train/test.")

    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_set = set(int(i) for i in train_idx)
    test_set = set(int(i) for i in test_idx)
    train_records = [record for idx, record in enumerate(records) if idx in train_set]
    test_records = [record for idx, record in enumerate(records) if idx in test_set]
    return train_records, test_records


def compute_split_pair_weights(split_records, normalize_mean=False):
    """Return per-pair weights so each protein contributes equally."""
    pieces = [
        np.full(record["n_pairs"], 1.0 / record["n_pairs"], dtype=np.float64)
        for record in split_records
    ]
    if not pieces:
        return np.empty((0,), dtype=np.float64)

    weights = np.concatenate(pieces)
    if normalize_mean and weights.size > 0:
        weights *= weights.size / np.sum(weights)
    return weights


def build_offsets(split_records):
    """Build global row offsets for one split."""
    offsets = []
    start = 0
    for record in split_records:
        end = start + record["n_pairs"]
        offsets.append((start, end))
        start = end
    return offsets


def format_offset_label(offset):
    """Create a stable string label for one offset."""
    di, dj = offset
    return f"di_{di:+d}_dj_{dj:+d}".replace("+", "p").replace("-", "m")


def get_feature_names(variant):
    """Return ordered feature names for one variant."""
    names = []
    for offset in variant["offsets"]:
        offset_label = format_offset_label(offset)
        if variant["bidirectional"]:
            names.extend([f"{offset_label}_pb_ab_{idx}" for idx in range(N_CHANNELS)])
            names.extend([f"{offset_label}_pb_ba_{idx}" for idx in range(N_CHANNELS)])
        else:
            names.extend([f"{offset_label}_pb_ab_{idx}" for idx in range(N_CHANNELS)])
    names.append("n_residues")
    return names


def region_mask(values, region):
    """Return a boolean mask for one fixed region."""
    values = np.asarray(values, dtype=np.float64)
    mask = np.ones(values.shape, dtype=bool)
    if region["min_exclusive"] is not None:
        mask &= values > float(region["min_exclusive"])
    if region["max_inclusive"] is not None:
        mask &= values <= float(region["max_inclusive"])
    return mask


def assign_region_label(variant, n_residues):
    """Return the fixed region label for one protein size."""
    for region in variant["regions"]:
        if region_mask(np.asarray([n_residues], dtype=np.float64), region)[0]:
            return region["label"]
    raise ValueError(
        f"Unable to route n_residues={n_residues} for variant {variant['name']}"
    )


def build_protein_feature_chunk(record, variant, layer):
    """Load one protein pair block and build dense feature rows for one variant."""
    pb_path = resolve_pair_block_path(record["protein_dir"], record["protein_id"], layer)
    pb = np.load(str(pb_path), mmap_mode="r")

    pair_i = record["pair_i"]
    pair_j = record["pair_j"]
    n_pairs = record["n_pairs"]
    n_residues = float(record["n_residues"])
    feature_dim = int(variant["feature_dim"])
    chunk = np.zeros((n_pairs, feature_dim), dtype=np.float32)

    write_col = 0
    for di, dj in variant["offsets"]:
        src_i = pair_i + di
        src_j = pair_j + dj
        in_bounds = (
            (src_i >= 0)
            & (src_i < record["n_residues"])
            & (src_j >= 0)
            & (src_j < record["n_residues"])
        )

        if variant["bidirectional"]:
            valid = in_bounds
            if np.any(valid):
                chunk[valid, write_col:write_col + N_CHANNELS] = pb[src_i[valid], src_j[valid], :]
                write_col += N_CHANNELS
                chunk[valid, write_col:write_col + N_CHANNELS] = pb[src_j[valid], src_i[valid], :]
                write_col += N_CHANNELS
            else:
                write_col += 2 * N_CHANNELS
        else:
            valid = in_bounds & (src_i > src_j)
            if np.any(valid):
                chunk[valid, write_col:write_col + N_CHANNELS] = pb[src_i[valid], src_j[valid], :]
            write_col += N_CHANNELS

    chunk[:, -1] = n_residues
    del pb
    return chunk


def build_split_matrix(split_records, variant, layer):
    """Build dense matrix and target vector for one split, variant, and layer."""
    n_rows = int(sum(record["n_pairs"] for record in split_records))
    x = np.empty((n_rows, variant["feature_dim"]), dtype=np.float32)
    y = np.empty((n_rows,), dtype=np.float64)

    offset = 0
    for record in split_records:
        chunk = build_protein_feature_chunk(record, variant, layer)
        end = offset + record["n_pairs"]
        x[offset:end, :] = chunk
        y[offset:end] = record["y_abs"]
        offset = end
        del chunk

    return x, y


def fit_and_apply_standardization(x_train, x_test, n_scale_features):
    """Standardize leading feature columns using train-only moments."""
    if n_scale_features <= 0:
        return

    mean = np.mean(x_train[:, :n_scale_features], axis=0, dtype=np.float64)
    scale = np.std(x_train[:, :n_scale_features], axis=0, dtype=np.float64)
    scale[scale < FLOAT_EPS] = 1.0

    x_train[:, :n_scale_features] -= mean
    x_train[:, :n_scale_features] /= scale
    x_test[:, :n_scale_features] -= mean
    x_test[:, :n_scale_features] /= scale


def compute_r2(y_true, y_pred):
    """Compute vanilla R-squared."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")

    mean_true = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot <= FLOAT_EPS:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def summarize_predictions(y_true, y_pred, split_records, variant, layer, split_name):
    """Compute per-protein R2 rows and split-level summaries."""
    offsets = build_offsets(split_records)
    per_rows = []
    per_r2 = []
    negative_r2_count = 0

    for record, (start, end) in zip(split_records, offsets):
        y_true_p = y_true[start:end]
        y_pred_p = y_pred[start:end]
        per_r2_val = compute_r2(y_true_p, y_pred_p)

        if not math.isnan(per_r2_val):
            per_r2.append(per_r2_val)
            if per_r2_val < 0.0:
                negative_r2_count += 1

        error = y_pred_p - y_true_p
        per_rows.append(
            {
                "variant": variant["name"],
                "model_type": "fixed_threshold_piecewise_linear",
                "feature_mode": variant["feature_mode"],
                "layer": layer,
                "split": split_name,
                "protein_id": record["protein_id"],
                "region_label": assign_region_label(variant, record["n_residues"]),
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
                "per_protein_r2": per_r2_val,
                "mean_true_distance": float(np.mean(y_true_p)),
                "std_true_distance": float(np.std(y_true_p)),
                "mean_predicted_distance": float(np.mean(y_pred_p)),
                "std_predicted_distance": float(np.std(y_pred_p)),
                "mean_error_pred_minus_true": float(np.mean(error)),
            }
        )

    return {
        "mean_per_protein_r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
        "median_per_protein_r2": float(np.median(per_r2)) if per_r2 else float("nan"),
        "negative_r2_count": int(negative_r2_count),
        "negative_r2_fraction": (
            float(negative_r2_count / len(per_r2)) if per_r2 else float("nan")
        ),
        "per_rows": per_rows,
    }


def summarize_geometry_rows(geometry_rows):
    """Aggregate geometry metrics per variant, layer, and split."""
    metric_keys = [
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

    grouped = {}
    for row in geometry_rows:
        key = (row["variant"], int(row["layer"]), row["split"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (variant_name, layer, split_name), rows in sorted(grouped.items()):
        out = {
            "variant": variant_name,
            "layer": layer,
            "split": split_name,
            "n_proteins": len(rows),
        }
        for metric_key in metric_keys:
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


def make_failed_geometry_row(variant_name, feature_mode, layer, split_name, record, per_r2):
    """Return a placeholder geometry row when reconstruction fails."""
    return {
        "variant": variant_name,
        "model_type": "fixed_threshold_piecewise_linear",
        "feature_mode": feature_mode,
        "layer": layer,
        "split": split_name,
        "protein_id": record["protein_id"],
        "region_label": "",
        "n_residues": record["n_residues"],
        "n_pairs": record["n_pairs"],
        "per_protein_r2": per_r2,
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


def collect_geometry_rows(variant, layer, split_name, split_records, y_true, y_pred):
    """Run per-protein geometry analysis for one split."""
    rows = []
    offsets = build_offsets(split_records)
    for record, (start, end) in zip(split_records, offsets):
        per_protein_r2 = compute_r2(y_true[start:end], y_pred[start:end])
        try:
            metrics = analyze_protein_geometry(
                record["n_residues"],
                record["coordinates"],
                record["pair_i"],
                record["pair_j"],
                y_pred[start:end],
            )
            rows.append(
                {
                    "variant": variant["name"],
                    "model_type": "fixed_threshold_piecewise_linear",
                    "feature_mode": variant["feature_mode"],
                    "layer": layer,
                    "split": split_name,
                    "protein_id": record["protein_id"],
                    "region_label": assign_region_label(variant, record["n_residues"]),
                    "n_residues": record["n_residues"],
                    "n_pairs": record["n_pairs"],
                    "per_protein_r2": per_protein_r2,
                    **metrics,
                }
            )
        except Exception:
            rows.append(
                make_failed_geometry_row(
                    variant["name"],
                    variant["feature_mode"],
                    layer,
                    split_name,
                    record,
                    per_protein_r2,
                )
            )
    return rows


def allocate_protein_equal_sample_counts(test_records, sample_size, rng):
    """Allocate sample counts approximately equally per protein."""
    if not test_records:
        return np.empty((0,), dtype=np.int64)

    capacities = np.array([record["n_pairs"] for record in test_records], dtype=np.int64)
    total_capacity = int(np.sum(capacities))
    target = min(int(sample_size), total_capacity)
    n_proteins = len(test_records)

    if target <= 0:
        return np.zeros((n_proteins,), dtype=np.int64)

    counts = np.full(n_proteins, target // n_proteins, dtype=np.int64)
    remainder = target % n_proteins
    if remainder > 0:
        bump = rng.permutation(n_proteins)[:remainder]
        counts[bump] += 1

    np.minimum(counts, capacities, out=counts)
    assigned = int(np.sum(counts))
    deficit = target - assigned

    while deficit > 0:
        eligible = np.flatnonzero(capacities > counts)
        if eligible.size == 0:
            break
        step = min(deficit, eligible.size)
        chosen = rng.choice(eligible, size=step, replace=False)
        counts[chosen] += 1
        deficit -= step

    return counts


def sample_test_rows_for_variant(variant, layer, test_records, y_true, y_pred, sample_size, rng):
    """Sample test prediction rows with protein-equal allocation."""
    offsets = build_offsets(test_records)
    sample_counts = allocate_protein_equal_sample_counts(test_records, sample_size, rng)

    rows = []
    for idx, (record, (start, end)) in enumerate(zip(test_records, offsets)):
        take_n = int(sample_counts[idx])
        if take_n <= 0:
            continue

        n_pairs = int(record["n_pairs"])
        if take_n >= n_pairs:
            chosen_local = np.arange(n_pairs, dtype=np.int64)
        else:
            chosen_local = rng.choice(n_pairs, size=take_n, replace=False)

        seq_sep = record["seq_sep"]
        region_label = assign_region_label(variant, record["n_residues"])
        for local_idx in chosen_local:
            global_idx = start + int(local_idx)
            rows.append(
                {
                    "variant": variant["name"],
                    "model_type": "fixed_threshold_piecewise_linear",
                    "feature_mode": variant["feature_mode"],
                    "layer": layer,
                    "split": "test",
                    "protein_id": record["protein_id"],
                    "region_label": region_label,
                    "n_residues": int(record["n_residues"]),
                    "seq_sep": float(seq_sep[local_idx]),
                    "actual_distance": float(y_true[global_idx]),
                    "predicted_distance": float(y_pred[global_idx]),
                }
            )

    return rows


def fit_region_models(x_train, y_train, train_weights, variant, feature_names, layer):
    """Fit one linear regression per fixed region."""
    split_idx = int(variant["split_feature_index"])
    split_values = x_train[:, split_idx]
    models = {}
    coef_rows = []

    for region in variant["regions"]:
        mask = region_mask(split_values, region)
        raw_row_count = int(mask.sum())
        if raw_row_count == 0:
            raise RuntimeError(
                f"No training rows for variant={variant['name']} layer={layer} "
                f"region={region['label']}"
            )

        model = LinearRegression(copy_X=False)
        model.fit(
            x_train[mask, :split_idx],
            y_train[mask],
            sample_weight=train_weights[mask],
        )
        models[region["label"]] = model

        coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
        intercept = float(model.intercept_)
        weight_mass = float(train_weights[mask].sum())

        for coef_idx, coef_value in enumerate(coef):
            coef_rows.append(
                {
                    "variant": variant["name"],
                    "model_type": "fixed_threshold_piecewise_linear",
                    "feature_mode": variant["feature_mode"],
                    "layer": layer,
                    "region_label": region["label"],
                    "node_path": region["node_path"],
                    "path_rule": region["path_rule"],
                    "n_train_rows": raw_row_count,
                    "train_protein_equivalent_weight_mass": weight_mass,
                    "intercept_raw": intercept,
                    "coef_index": int(coef_idx),
                    "feature_name": feature_names[coef_idx],
                    "coefficient_raw": float(coef_value),
                }
            )

    return models, coef_rows


def predict_piecewise(models, x, variant):
    """Predict with one fixed-region linear model per route."""
    split_idx = int(variant["split_feature_index"])
    split_values = x[:, split_idx]
    y_pred = np.empty((x.shape[0],), dtype=np.float64)

    for region in variant["regions"]:
        mask = region_mask(split_values, region)
        if not np.any(mask):
            continue
        y_pred[mask] = np.asarray(
            models[region["label"]].predict(x[mask, :split_idx]),
            dtype=np.float64,
        )

    return y_pred


def collect_region_diagnostics(split_records, weights, variant, layer, split_name):
    """Summarize mass and row counts for each fixed region."""
    rows = []
    offsets = build_offsets(split_records)
    total_rows = int(sum(record["n_pairs"] for record in split_records))
    total_mass = float(weights.sum()) if weights.size > 0 else 0.0

    for region in variant["regions"]:
        raw_row_count = 0
        weight_mass = 0.0
        protein_ids = []

        for record, (start, end) in zip(split_records, offsets):
            if assign_region_label(variant, record["n_residues"]) != region["label"]:
                continue
            raw_row_count += int(record["n_pairs"])
            protein_ids.append(record["protein_id"])
            if weights.size > 0:
                weight_mass += float(weights[start:end].sum())

        rows.append(
            {
                "variant": variant["name"],
                "model_type": "fixed_threshold_piecewise_linear",
                "feature_mode": variant["feature_mode"],
                "layer": layer,
                "split": split_name,
                "region_label": region["label"],
                "node_path": region["node_path"],
                "path_rule": region["path_rule"],
                "n_proteins": len(protein_ids),
                "raw_row_count": raw_row_count,
                "raw_row_fraction": (
                    float(raw_row_count / total_rows) if total_rows > 0 else float("nan")
                ),
                "protein_equivalent_weight_mass": weight_mass,
                "protein_equivalent_weight_fraction": (
                    float(weight_mass / total_mass) if total_mass > 0 else float("nan")
                ),
            }
        )

    return rows


def summarize_region_coverage(records, variant):
    """Summarize fixed-region coverage for one protein split."""
    summary = []
    for region in variant["regions"]:
        matched = [
            record
            for record in records
            if assign_region_label(variant, record["n_residues"]) == region["label"]
        ]
        summary.append(
            {
                "region_label": region["label"],
                "n_proteins": len(matched),
                "n_pairs": int(sum(record["n_pairs"] for record in matched)),
            }
        )
    return summary


def validate_split_region_coverage(train_records, test_records):
    """Ensure all fixed regions are represented in the training split."""
    for variant in VARIANTS:
        train_summary = summarize_region_coverage(train_records, variant)
        test_summary = summarize_region_coverage(test_records, variant)

        train_text = ", ".join(
            f"{row['region_label']}={row['n_proteins']} proteins/{row['n_pairs']} pairs"
            for row in train_summary
        )
        test_text = ", ".join(
            f"{row['region_label']}={row['n_proteins']} proteins/{row['n_pairs']} pairs"
            for row in test_summary
        )
        print(f"Region coverage [{variant['name']}] train: {train_text}")
        print(f"Region coverage [{variant['name']}] test:  {test_text}")

        missing_train = [
            row["region_label"] for row in train_summary if int(row["n_proteins"]) == 0
        ]
        if missing_train:
            raise RuntimeError(
                f"Training split is missing fixed regions for {variant['name']}: "
                f"{', '.join(missing_train)}. "
                "The fixed-threshold setup requires at least one training protein per region."
            )


def build_rules_rows():
    """Export fixed-threshold metadata for each variant and region."""
    rows = []
    for variant in VARIANTS:
        for region in variant["regions"]:
            rows.append(
                {
                    "variant": variant["name"],
                    "model_type": "fixed_threshold_piecewise_linear",
                    "feature_mode": variant["feature_mode"],
                    "split_feature_name": "n_residues",
                    "root_threshold": variant["root_threshold"],
                    "left_child_threshold": variant["left_child_threshold"],
                    "right_child_threshold": variant["right_child_threshold"],
                    "region_label": region["label"],
                    "node_path": region["node_path"],
                    "min_n_residues_exclusive": (
                        "" if region["min_exclusive"] is None else region["min_exclusive"]
                    ),
                    "max_n_residues_inclusive": (
                        "" if region["max_inclusive"] is None else region["max_inclusive"]
                    ),
                    "path_rule": region["path_rule"],
                }
            )
    return rows


def run_variant_for_layer(variant, layer, train_records, test_records, variant_index):
    """Fit one variant for one layer and return all outputs."""
    t0 = time.time()

    feature_names = get_feature_names(variant)
    split_idx = int(variant["split_feature_index"])

    x_train, y_train = build_split_matrix(train_records, variant, layer)
    x_test, y_test = build_split_matrix(test_records, variant, layer)

    fit_and_apply_standardization(x_train, x_test, n_scale_features=split_idx)
    if feature_names[split_idx] != "n_residues":
        raise ValueError(
            f"Expected split feature to be n_residues, got {feature_names[split_idx]}"
        )

    train_weights = compute_split_pair_weights(train_records, normalize_mean=False)
    test_weights = compute_split_pair_weights(test_records, normalize_mean=False)

    models, coef_rows = fit_region_models(
        x_train,
        y_train,
        train_weights,
        variant,
        feature_names,
        layer,
    )

    y_pred_train = predict_piecewise(models, x_train, variant)
    y_pred_test = predict_piecewise(models, x_test, variant)

    train_metrics = summarize_predictions(
        y_train, y_pred_train, train_records, variant, layer, "train"
    )
    test_metrics = summarize_predictions(
        y_test, y_pred_test, test_records, variant, layer, "test"
    )

    train_geometry_rows = collect_geometry_rows(
        variant, layer, "train", train_records, y_train, y_pred_train
    )
    test_geometry_rows = collect_geometry_rows(
        variant, layer, "test", test_records, y_test, y_pred_test
    )
    geometry_rows = train_geometry_rows + test_geometry_rows

    geometry_lookup = {
        (row["variant"], int(row["layer"]), row["split"]): row
        for row in summarize_geometry_rows(geometry_rows)
    }
    train_geom_summary = geometry_lookup.get((variant["name"], int(layer), "train"), {})
    test_geom_summary = geometry_lookup.get((variant["name"], int(layer), "test"), {})

    region_rows = []
    region_rows.extend(
        collect_region_diagnostics(train_records, train_weights, variant, layer, "train")
    )
    region_rows.extend(
        collect_region_diagnostics(test_records, test_weights, variant, layer, "test")
    )

    scatter_rows = []
    if layer == SCATTER_SAMPLE_LAYER:
        rng = np.random.default_rng((RANDOM_STATE * 1000) + (layer * 10) + variant_index)
        scatter_rows = sample_test_rows_for_variant(
            variant=variant,
            layer=layer,
            test_records=test_records,
            y_true=y_test,
            y_pred=y_pred_test,
            sample_size=SCATTER_SAMPLE_PER_VARIANT,
            rng=rng,
        )

    summary_row = {
        "variant": variant["name"],
        "model_type": "fixed_threshold_piecewise_linear",
        "feature_mode": variant["feature_mode"],
        "layer": layer,
        "feature_dim": variant["feature_dim"],
        "uses_protein_equal_weighting": True,
        "weight_normalization": "per_protein_sum_1",
        "threshold_policy": "fixed_layer47_leaf_thresholds",
        "n_regions": len(variant["regions"]),
        "root_threshold": variant["root_threshold"],
        "left_child_threshold": variant["left_child_threshold"],
        "right_child_threshold": variant["right_child_threshold"],
        "split_feature_index": split_idx,
        "split_feature_name": feature_names[split_idx],
        "n_train_proteins": len(train_records),
        "n_test_proteins": len(test_records),
        "n_train_pairs": int(y_train.size),
        "n_test_pairs": int(y_test.size),
        "scatter_test_samples": int(len(scatter_rows)),
        "train_mean_per_protein_r2": train_metrics["mean_per_protein_r2"],
        "train_median_per_protein_r2": train_metrics["median_per_protein_r2"],
        "train_negative_r2_count": train_metrics["negative_r2_count"],
        "train_negative_r2_fraction": train_metrics["negative_r2_fraction"],
        "test_mean_per_protein_r2": test_metrics["mean_per_protein_r2"],
        "test_median_per_protein_r2": test_metrics["median_per_protein_r2"],
        "test_negative_r2_count": test_metrics["negative_r2_count"],
        "test_negative_r2_fraction": test_metrics["negative_r2_fraction"],
        "train_mean_tm_score": train_geom_summary.get("tm_score_mean", float("nan")),
        "test_mean_tm_score": test_geom_summary.get("tm_score_mean", float("nan")),
        "train_mean_rmsd": train_geom_summary.get("rmsd_mean", float("nan")),
        "test_mean_rmsd": test_geom_summary.get("rmsd_mean", float("nan")),
        "train_mean_stress": train_geom_summary.get("stress_mean", float("nan")),
        "test_mean_stress": test_geom_summary.get("stress_mean", float("nan")),
        "train_mean_triangle_violation_fraction": train_geom_summary.get(
            "triangle_violation_fraction_mean",
            float("nan"),
        ),
        "test_mean_triangle_violation_fraction": test_geom_summary.get(
            "triangle_violation_fraction_mean",
            float("nan"),
        ),
        "train_mean_neg_eigenvalue_fraction": train_geom_summary.get(
            "neg_eigenvalue_fraction_mean",
            float("nan"),
        ),
        "test_mean_neg_eigenvalue_fraction": test_geom_summary.get(
            "neg_eigenvalue_fraction_mean",
            float("nan"),
        ),
        "elapsed_seconds": float(time.time() - t0),
    }

    del x_train, x_test, y_train, y_test, y_pred_train, y_pred_test

    return summary_row, train_metrics["per_rows"] + test_metrics["per_rows"], coef_rows, geometry_rows, region_rows, scatter_rows


def format_value(value):
    """Stable CSV float formatting."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_csv(path, fieldnames, rows):
    """Write CSV with stable field ordering."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def main():
    t_start = time.time()

    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 72}")
    print("PIECEWISE LINEAR REGRESSION — FIXED THRESHOLD CONTEXT DIAGNOSTIC")
    print(f"{'=' * 72}")
    print(f"Pair blocks root:     {PAIR_BLOCKS_ROOT}")
    print(f"PDB root:             {PDB_ROOT}")
    print(f"Output CSV dir:       {csv_dir}")
    print(f"Layers:               0..{N_LAYERS - 1}")
    print(f"Train/test split:     80/20 protein-level, seed={RANDOM_STATE}")
    print("Training weights:     protein-equal (per protein sums to 1.0)")
    print("Model family:         fixed-threshold piecewise linear regression")
    print("Scaling policy:       z-score all non-n_residues features")
    print(
        "Scatter sample:       test-only, protein-equal, up to "
        f"{SCATTER_SAMPLE_PER_VARIANT:,} rows per variant for layer {SCATTER_SAMPLE_LAYER}"
    )
    print(f"Variants:             {len(VARIANTS)}")

    all_proteins = discover_proteins(PAIR_BLOCKS_ROOT)
    print(f"\nDiscovered {len(all_proteins)} protein directories.")

    protein_data = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        try:
            pdb_path = resolve_pdb_file(protein_id, protein_dir)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                print(f"[SKIP] {protein_id}: no predicted PDB file found")
                continue

            pb0_path = resolve_pair_block_path(protein_dir, protein_id, 0)
            if not pb0_path.exists():
                skipped.append((protein_id, "pair block 0 not found"))
                print(f"[SKIP] {protein_id}: pair block 0 not found at {pb0_path}")
                continue

            ca_data = extract_ca_coordinates(pdb_path)
            n_ca = len(ca_data["residues"])
            if n_ca < 2:
                skipped.append((protein_id, "fewer than 2 CA atoms"))
                print(f"[SKIP] {protein_id}: fewer than 2 CA atoms")
                continue

            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = int(pb0.shape[0])
            del pb0

            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                print(f"[SKIP] {protein_id}: CA count {n_ca} != pair block dim {pb_dim}")
                continue

            pair_i, pair_j = np.tril_indices(n_ca, k=-1)
            if pair_i.size == 0:
                skipped.append((protein_id, "no lower-triangle residue pairs"))
                print(f"[SKIP] {protein_id}: no lower-triangle residue pairs")
                continue

            dist_matrix = compute_distance_matrix(ca_data["coordinates"])
            y_abs = np.asarray(dist_matrix[pair_i, pair_j], dtype=np.float32)
            seq_sep = (pair_i - pair_j).astype(np.float32, copy=False)

            protein_data.append(
                {
                    "protein_id": protein_id,
                    "protein_dir": protein_dir,
                    "n_residues": n_ca,
                    "pair_i": pair_i,
                    "pair_j": pair_j,
                    "n_pairs": int(y_abs.size),
                    "y_abs": y_abs,
                    "seq_sep": seq_sep,
                    "coordinates": ca_data["coordinates"],
                }
            )
            print(
                f"[OK]   {protein_id}: {n_ca} residues, {y_abs.size:,} lower-triangle pairs"
            )

        except Exception as exc:
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {exc}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(protein_data)} proteins (skipped {len(skipped)}).")

    train_records, test_records = split_global_records(protein_data)
    print(
        f"Global split: {len(train_records)} train / {len(test_records)} test proteins"
    )
    validate_split_region_coverage(train_records, test_records)

    split_rows = []
    for record in train_records:
        split_rows.append(
            {
                "scope": "global_shared",
                "protein_id": record["protein_id"],
                "split": "train",
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
            }
        )
    for record in test_records:
        split_rows.append(
            {
                "scope": "global_shared",
                "protein_id": record["protein_id"],
                "split": "test",
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
            }
        )

    split_fields = ["scope", "protein_id", "split", "n_residues", "n_pairs"]
    write_csv(csv_dir / SPLIT_CSV_NAME, split_fields, split_rows)
    print(f"Saved split CSV -> {csv_dir / SPLIT_CSV_NAME}")

    summary_rows = []
    per_protein_rows = []
    coef_rows = []
    geometry_rows = []
    region_rows = []
    scatter_rows = []

    for layer in range(N_LAYERS):
        layer_start = time.time()
        print(f"\n{'-' * 72}")
        print(f"[LAYER {layer:02d}] start={now_timestamp()}")

        for variant_index, variant in enumerate(VARIANTS):
            variant_start = time.time()
            print(
                f"  Running variant: {variant['name']} "
                f"(feature_mode={variant['feature_mode']})"
            )
            (
                summary_row,
                variant_per_rows,
                variant_coef_rows,
                variant_geometry_rows,
                variant_region_rows,
                variant_scatter_rows,
            ) = run_variant_for_layer(
                variant=variant,
                layer=layer,
                train_records=train_records,
                test_records=test_records,
                variant_index=variant_index,
            )
            summary_rows.append(summary_row)
            per_protein_rows.extend(variant_per_rows)
            coef_rows.extend(variant_coef_rows)
            geometry_rows.extend(variant_geometry_rows)
            region_rows.extend(variant_region_rows)
            scatter_rows.extend(variant_scatter_rows)

            print(
                f"  Completed {variant['name']}: "
                f"test mean per-protein R2={summary_row['test_mean_per_protein_r2']:.4f}, "
                f"test median={summary_row['test_median_per_protein_r2']:.4f}, "
                f"TM_test={summary_row['test_mean_tm_score']:.4f}, "
                f"stress_test={summary_row['test_mean_stress']:.4f}, "
                f"elapsed={time.time() - variant_start:.1f}s"
            )

        print(
            f"[LAYER {layer:02d}] end={now_timestamp()} "
            f"elapsed={time.time() - layer_start:.1f}s"
        )

    geometry_summary_rows = summarize_geometry_rows(geometry_rows)
    rules_rows = build_rules_rows()

    summary_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "layer",
        "feature_dim",
        "uses_protein_equal_weighting",
        "weight_normalization",
        "threshold_policy",
        "n_regions",
        "root_threshold",
        "left_child_threshold",
        "right_child_threshold",
        "split_feature_index",
        "split_feature_name",
        "n_train_proteins",
        "n_test_proteins",
        "n_train_pairs",
        "n_test_pairs",
        "scatter_test_samples",
        "train_mean_per_protein_r2",
        "train_median_per_protein_r2",
        "train_negative_r2_count",
        "train_negative_r2_fraction",
        "test_mean_per_protein_r2",
        "test_median_per_protein_r2",
        "test_negative_r2_count",
        "test_negative_r2_fraction",
        "train_mean_tm_score",
        "test_mean_tm_score",
        "train_mean_rmsd",
        "test_mean_rmsd",
        "train_mean_stress",
        "test_mean_stress",
        "train_mean_triangle_violation_fraction",
        "test_mean_triangle_violation_fraction",
        "train_mean_neg_eigenvalue_fraction",
        "test_mean_neg_eigenvalue_fraction",
        "elapsed_seconds",
    ]
    write_csv(csv_dir / SUMMARY_CSV_NAME, summary_fields, summary_rows)
    print(f"\nSaved summary CSV -> {csv_dir / SUMMARY_CSV_NAME}")

    per_protein_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "layer",
        "split",
        "protein_id",
        "region_label",
        "n_residues",
        "n_pairs",
        "per_protein_r2",
        "mean_true_distance",
        "std_true_distance",
        "mean_predicted_distance",
        "std_predicted_distance",
        "mean_error_pred_minus_true",
    ]
    write_csv(csv_dir / PER_PROTEIN_CSV_NAME, per_protein_fields, per_protein_rows)
    print(f"Saved per-protein CSV -> {csv_dir / PER_PROTEIN_CSV_NAME}")

    rules_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "split_feature_name",
        "root_threshold",
        "left_child_threshold",
        "right_child_threshold",
        "region_label",
        "node_path",
        "min_n_residues_exclusive",
        "max_n_residues_inclusive",
        "path_rule",
    ]
    write_csv(csv_dir / RULES_CSV_NAME, rules_fields, rules_rows)
    print(f"Saved rules CSV -> {csv_dir / RULES_CSV_NAME}")

    coef_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "layer",
        "region_label",
        "node_path",
        "path_rule",
        "n_train_rows",
        "train_protein_equivalent_weight_mass",
        "intercept_raw",
        "coef_index",
        "feature_name",
        "coefficient_raw",
    ]
    write_csv(csv_dir / COEF_CSV_NAME, coef_fields, coef_rows)
    print(f"Saved coefficients CSV -> {csv_dir / COEF_CSV_NAME}")

    region_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "layer",
        "split",
        "region_label",
        "node_path",
        "path_rule",
        "n_proteins",
        "raw_row_count",
        "raw_row_fraction",
        "protein_equivalent_weight_mass",
        "protein_equivalent_weight_fraction",
    ]
    write_csv(csv_dir / REGION_DIAGNOSTICS_CSV_NAME, region_fields, region_rows)
    print(f"Saved region diagnostics CSV -> {csv_dir / REGION_DIAGNOSTICS_CSV_NAME}")

    scatter_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "layer",
        "split",
        "protein_id",
        "region_label",
        "n_residues",
        "seq_sep",
        "actual_distance",
        "predicted_distance",
    ]
    write_csv(csv_dir / SCATTER_SAMPLE_CSV_NAME, scatter_fields, scatter_rows)
    print(f"Saved scatter sample CSV -> {csv_dir / SCATTER_SAMPLE_CSV_NAME}")

    geometry_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "layer",
        "split",
        "protein_id",
        "region_label",
        "n_residues",
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
    write_csv(csv_dir / GEOMETRY_METRICS_CSV_NAME, geometry_fields, geometry_rows)
    print(f"Saved geometry metrics CSV -> {csv_dir / GEOMETRY_METRICS_CSV_NAME}")

    geometry_summary_fields = ["variant", "layer", "split", "n_proteins"]
    geometry_metrics = [
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
    for metric_key in geometry_metrics:
        geometry_summary_fields.extend([f"{metric_key}_mean", f"{metric_key}_std"])
    write_csv(
        csv_dir / GEOMETRY_SUMMARY_CSV_NAME,
        geometry_summary_fields,
        geometry_summary_rows,
    )
    print(f"Saved geometry summary CSV -> {csv_dir / GEOMETRY_SUMMARY_CSV_NAME}")

    summary_path = csv_dir / SUMMARY_TXT_NAME
    with summary_path.open("w") as handle:
        handle.write("Piecewise Linear Regression — Fixed Threshold Context Diagnostic\n")
        handle.write("=" * 72 + "\n\n")
        handle.write(f"Pair blocks root:   {PAIR_BLOCKS_ROOT}\n")
        handle.write(f"PDB root:           {PDB_ROOT}\n")
        handle.write(f"Output CSV dir:     {csv_dir}\n")
        handle.write(f"Layers:             0..{N_LAYERS - 1}\n")
        handle.write("Representation:     one lower-triangle row per unordered pair\n")
        handle.write("Training weighting: protein-equal (each protein sums to 1.0)\n")
        handle.write("Model family:       fixed-threshold piecewise linear regression\n")
        handle.write("Threshold policy:   fixed layer-47 leaf thresholds\n")
        handle.write("Scaling policy:     z-score all non-n_residues features\n")
        handle.write(
            "Scatter sampling:   test-only, protein-equal, up to "
            f"{SCATTER_SAMPLE_PER_VARIANT:,} rows per variant for layer {SCATTER_SAMPLE_LAYER}\n"
        )
        handle.write(
            f"Train/test split:   protein-level 80/20, seed={RANDOM_STATE}\n\n"
        )
        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(protein_data)}\n")
        handle.write(f"Proteins skipped:    {len(skipped)}\n")
        handle.write(f"Train proteins:      {len(train_records)}\n")
        handle.write(f"Test proteins:       {len(test_records)}\n\n")

        best_row = max(
            summary_rows,
            key=lambda row: float(row["test_mean_per_protein_r2"]),
        )
        handle.write(
            "Best variant/layer by test mean per-protein R2: "
            f"{best_row['variant']} @ layer {int(best_row['layer'])} "
            f"({best_row['test_mean_per_protein_r2']:.4f})\n\n"
        )

        handle.write("Best layer per variant:\n")
        for variant in VARIANTS:
            variant_rows = [row for row in summary_rows if row["variant"] == variant["name"]]
            best_variant_row = max(
                variant_rows,
                key=lambda row: float(row["test_mean_per_protein_r2"]),
            )
            handle.write(
                f"  {variant['name']}: layer {int(best_variant_row['layer'])}, "
                f"meanPP_R2={best_variant_row['test_mean_per_protein_r2']:.4f}, "
                f"medianPP_R2={best_variant_row['test_median_per_protein_r2']:.4f}, "
                f"TM={best_variant_row['test_mean_tm_score']:.4f}, "
                f"Stress={best_variant_row['test_mean_stress']:.4f}\n"
            )

        handle.write("\nPer-layer summary:\n")
        handle.write(
            "  Variant"
            + " " * 36
            + "Layer  TestMeanPP_R2  TestMedian   TestTM    Stress   TriViol   Elapsed\n"
        )
        for row in summary_rows:
            handle.write(
                f"  {row['variant']:<51}"
                f"{int(row['layer']):>5d}"
                f"{row['test_mean_per_protein_r2']:>15.4f}"
                f"{row['test_median_per_protein_r2']:>12.4f}"
                f"{row['test_mean_tm_score']:>10.4f}"
                f"{row['test_mean_stress']:>10.4f}"
                f"{row['test_mean_triangle_violation_fraction']:>10.6f}"
                f"{row['elapsed_seconds']:>10.1f}\n"
            )

        handle.write("\nTop 5 worst/best test proteins at each variant's best layer:\n")
        for variant in VARIANTS:
            variant_rows = [row for row in summary_rows if row["variant"] == variant["name"]]
            best_variant_row = max(
                variant_rows,
                key=lambda row: float(row["test_mean_per_protein_r2"]),
            )
            best_layer = int(best_variant_row["layer"])
            test_rows = [
                row
                for row in per_protein_rows
                if row["variant"] == variant["name"]
                and int(row["layer"]) == best_layer
                and row["split"] == "test"
                and not math.isnan(float(row["per_protein_r2"]))
            ]
            ranked = sorted(test_rows, key=lambda row: float(row["per_protein_r2"]))
            handle.write(f"\n  {variant['name']} @ layer {best_layer}\n")
            handle.write("    Worst:\n")
            for row in ranked[:5]:
                handle.write(
                    f"      {row['protein_id']}: R2={row['per_protein_r2']:.4f}, "
                    f"region={row['region_label']}, n_res={row['n_residues']}\n"
                )
            handle.write("    Best:\n")
            for row in reversed(ranked[-5:]):
                handle.write(
                    f"      {row['protein_id']}: R2={row['per_protein_r2']:.4f}, "
                    f"region={row['region_label']}, n_res={row['n_residues']}\n"
                )

        if skipped:
            handle.write("\nSkipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")

        handle.write(
            f"\nNote: {GEOMETRY_METRICS_CSV_NAME} includes both train and test "
            "proteins with per-protein predictive and geometry diagnostics.\n"
        )
        handle.write(
            f"Note: {REGION_DIAGNOSTICS_CSV_NAME} reports raw row counts and "
            "protein-equivalent mass for each fixed region on both train and test.\n"
        )
        handle.write(
            f"Note: {RULES_CSV_NAME} documents the fixed thresholds used for LL/LR/RL/RR.\n"
        )
        handle.write(
            f"Note: {SCATTER_SAMPLE_CSV_NAME} contains layer-{SCATTER_SAMPLE_LAYER} "
            "test samples only.\n"
        )

    print(f"Saved summary text -> {summary_path}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 72}")
    print("PIECEWISE LINEAR REGRESSION RUN COMPLETE")
    print(f"Elapsed time: {elapsed:.1f} seconds ({elapsed / 60.0:.1f} minutes)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

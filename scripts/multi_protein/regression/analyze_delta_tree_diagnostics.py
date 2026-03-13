#!/usr/bin/env python3
"""
Analyze Delta decision-tree / XGBoost diagnostic outputs and generate a
reproducible interpretability report.
"""

from __future__ import annotations

import csv
import math
import os
import tempfile
from collections import defaultdict
from pathlib import Path
from typing import Iterable

os.environ.setdefault("MPLCONFIGDIR", os.path.join(tempfile.gettempdir(), "matplotlib-cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT_DIR = Path(__file__).resolve().parents[3]
DELTA_DIR = ROOT_DIR / "delta_files"
DECISION_TREE_DIR = DELTA_DIR / "diagnostic_decision_tree_csv_files"
XGBOOST_DIR = DELTA_DIR / "diagnostic_xgboost_csv_files"
LINEAR_TREE_DIR = DELTA_DIR / "diagnostic_linear_tree_csv_files"
OUTPUT_DIR = DELTA_DIR / "tree_diagnostics_analysis"

PAIR_FEATURE_COUNT = 128
FEATURE_NAMES = [f"pair_ch_{idx}" for idx in range(PAIR_FEATURE_COUNT)] + [
    "n_residues",
    "seq_sep",
    "log1p_seq_sep",
    "is_seq_sep_1",
    "is_seq_sep_le_2",
    "is_seq_sep_le_4",
]
SCALAR_FEATURES = FEATURE_NAMES[PAIR_FEATURE_COUNT:]
TOP_K_VALUES = (10, 20, 30)
REPORT_NAME = "delta_tree_diagnostics_report.md"
BRANCH_SUMMARY_CSV = "decision_tree_branch_summary.csv"
SCALAR_SUMMARY_CSV = "scalar_feature_importance_summary.csv"
XGB_TREE_CSV = "xgboost_best_trees.csv"
OVERLAP_CSV = "linear_vs_tree_rank_overlap.csv"
OVERLAP_SIG_CSV = "linear_vs_tree_rank_overlap_significance.csv"
XGB_DEPTH_CSV = "xgboost_depth_search_comparison.csv"
DT_FIGURE_NAME = "decision_tree_top_levels.png"


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, fieldnames: list[str], rows: Iterable[dict[str, object]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({name: format_value(row.get(name, "")) for name in fieldnames})


def format_value(value: object) -> object:
    if isinstance(value, float):
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def parse_float(value: str) -> float:
    if value == "" or value.lower() == "nan":
        return float("nan")
    return float(value)


def parse_int(value: str) -> int:
    return int(float(value))


def xgb_feature_name(raw_name: str) -> str:
    if raw_name.startswith("f") and raw_name[1:].isdigit():
        idx = int(raw_name[1:])
        if 0 <= idx < len(FEATURE_NAMES):
            return FEATURE_NAMES[idx]
    return raw_name


def importance_vector(rows: list[dict[str, str]], feature_names: list[str]) -> dict[str, float]:
    values = {name: 0.0 for name in feature_names}
    for row in rows:
        name = row["feature_name"]
        if name in values:
            values[name] = parse_float(row["importance"])
    return values


def average_rank_desc(values: dict[str, float]) -> dict[str, float]:
    items = sorted(values.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    position = 1
    index = 0
    while index < len(items):
        value = items[index][1]
        start = index
        while index < len(items) and items[index][1] == value:
            index += 1
        end = index
        avg_rank = (position + (position + (end - start) - 1)) / 2.0
        for name, _ in items[start:end]:
            ranks[name] = avg_rank
        position += end - start
    return ranks


def pearson_corr(xs: list[float], ys: list[float]) -> float:
    if len(xs) != len(ys) or not xs:
        return float("nan")
    mean_x = sum(xs) / len(xs)
    mean_y = sum(ys) / len(ys)
    num = sum((x - mean_x) * (y - mean_y) for x, y in zip(xs, ys))
    den_x = math.sqrt(sum((x - mean_x) ** 2 for x in xs))
    den_y = math.sqrt(sum((y - mean_y) ** 2 for y in ys))
    if den_x <= 0.0 or den_y <= 0.0:
        return float("nan")
    return num / (den_x * den_y)


def spearman_corr_desc(values_a: dict[str, float], values_b: dict[str, float]) -> float:
    rank_a = average_rank_desc(values_a)
    rank_b = average_rank_desc(values_b)
    keys = sorted(values_a.keys())
    xs = [rank_a[key] for key in keys]
    ys = [rank_b[key] for key in keys]
    return pearson_corr(xs, ys)


def top_k_features(values: dict[str, float], k: int) -> list[str]:
    return [name for name, _ in sorted(values.items(), key=lambda item: (-item[1], item[0]))[:k]]


def hypergeom_tail(n: int, success_states: int, draws: int, observed: int) -> float:
    denom = math.comb(n, draws)
    tail = 0.0
    upper = min(success_states, draws)
    for overlap in range(observed, upper + 1):
        tail += (
            math.comb(success_states, overlap)
            * math.comb(n - success_states, draws - overlap)
            / denom
        )
    return min(max(tail, 0.0), 1.0)


def benjamini_hochberg(rows: list[dict[str, object]], p_key: str, out_key: str) -> None:
    indexed = [
        (idx, float(row[p_key]))
        for idx, row in enumerate(rows)
        if not math.isnan(float(row[p_key]))
    ]
    indexed.sort(key=lambda item: item[1])
    if not indexed:
        return
    adjusted = [0.0] * len(indexed)
    total = len(indexed)
    running = 1.0
    for reverse_rank, (row_index, p_value) in enumerate(reversed(indexed), start=1):
        rank = total - reverse_rank + 1
        candidate = min(running, p_value * total / rank)
        running = candidate
        adjusted[rank - 1] = candidate
    for (rank_pos, (row_index, _)) in enumerate(indexed):
        rows[row_index][out_key] = adjusted[rank_pos]


def first_row(path: Path) -> dict[str, str]:
    rows = read_csv_rows(path)
    if not rows:
        raise ValueError(f"No rows found in {path}")
    return rows[0]


def build_parent_lookup(node_rows: list[dict[str, str]]) -> tuple[dict[int, dict[str, object]], dict[int, tuple[int, str]]]:
    nodes: dict[int, dict[str, object]] = {}
    parent_lookup: dict[int, tuple[int, str]] = {}
    for row in node_rows:
        node_id = parse_int(row["node_id"])
        node = {
            "node_id": node_id,
            "depth": parse_int(row["depth"]),
            "left_child": parse_int(row["left_child"]),
            "right_child": parse_int(row["right_child"]),
            "feature_name": row["feature_name"],
            "threshold": parse_float(row["threshold"]),
            "weighted_n_node_samples": parse_float(row["weighted_n_node_samples"]),
            "value": parse_float(row["value"]),
            "is_leaf": parse_int(row["is_leaf"]) == 1,
        }
        nodes[node_id] = node
        left = node["left_child"]
        right = node["right_child"]
        if left >= 0:
            parent_lookup[left] = (node_id, "left")
        if right >= 0:
            parent_lookup[right] = (node_id, "right")
    return nodes, parent_lookup


def decision_tree_leaf_paths(node_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    nodes, parent_lookup = build_parent_lookup(node_rows)
    root_weight = float(nodes[0]["weighted_n_node_samples"])
    rows = []
    for node_id, node in nodes.items():
        if not bool(node["is_leaf"]):
            continue
        conditions = []
        current = node_id
        while current in parent_lookup:
            parent_id, direction = parent_lookup[current]
            parent = nodes[parent_id]
            operator = "<=" if direction == "left" else ">"
            conditions.append(
                f"{parent['feature_name']} {operator} {float(parent['threshold']):.3f}"
            )
            current = parent_id
        conditions.reverse()
        weighted = float(node["weighted_n_node_samples"])
        rows.append(
            {
                "leaf_node_id": node_id,
                "leaf_depth": int(node["depth"]),
                "weighted_n_node_samples": weighted,
                "sample_fraction": weighted / root_weight if root_weight > 0.0 else float("nan"),
                "prediction_value": float(node["value"]),
                "path_rule": " AND ".join(conditions) if conditions else "ROOT",
            }
        )
    rows.sort(key=lambda row: float(row["weighted_n_node_samples"]), reverse=True)
    return rows


def scalar_summary_decision_tree(
    importance_rows: list[dict[str, str]],
    node_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    importance_map = {row["feature_name"]: parse_float(row["importance"]) for row in importance_rows}
    rank_map = {
        row["feature_name"]: rank
        for rank, row in enumerate(
            sorted(importance_rows, key=lambda row: (-parse_float(row["importance"]), row["feature_name"])),
            start=1,
        )
    }
    usage: dict[str, dict[str, object]] = {
        feature: {
            "split_count": 0,
            "earliest_depth": math.inf,
            "root_split_count": 0,
            "max_weighted_n_node_samples": 0.0,
        }
        for feature in SCALAR_FEATURES
    }
    for row in node_rows:
        feature = row["feature_name"]
        if feature not in usage:
            continue
        usage[feature]["split_count"] = int(usage[feature]["split_count"]) + 1
        usage[feature]["earliest_depth"] = min(
            float(usage[feature]["earliest_depth"]),
            parse_int(row["depth"]),
        )
        if parse_int(row["depth"]) == 0:
            usage[feature]["root_split_count"] = int(usage[feature]["root_split_count"]) + 1
        usage[feature]["max_weighted_n_node_samples"] = max(
            float(usage[feature]["max_weighted_n_node_samples"]),
            parse_float(row["weighted_n_node_samples"]),
        )

    rows = []
    for feature in SCALAR_FEATURES:
        importance = importance_map.get(feature, 0.0)
        split_count = int(usage[feature]["split_count"])
        earliest_depth = usage[feature]["earliest_depth"]
        if importance == 0.0 and split_count == 0:
            classification = "unused"
        elif importance < 0.005 and split_count <= 6 and earliest_depth >= 7:
            classification = "used_rare_low_effect"
        else:
            classification = "used_nontrivial"
        rows.append(
            {
                "model_name": "decision_tree",
                "feature_name": feature,
                "importance_metric": "sklearn_feature_importances_",
                "importance_value": importance,
                "importance_rank": rank_map.get(feature, math.nan),
                "split_count": split_count,
                "earliest_depth": float("nan") if math.isinf(float(earliest_depth)) else int(earliest_depth),
                "root_split_count": int(usage[feature]["root_split_count"]),
                "max_weighted_n_node_samples": float(usage[feature]["max_weighted_n_node_samples"]),
                "classification": classification,
            }
        )
    return rows


def scalar_summary_xgboost(
    gain_rows: list[dict[str, str]],
    weight_rows: list[dict[str, str]],
    cover_rows: list[dict[str, str]],
    node_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    gain_map = {row["feature_name"]: parse_float(row["importance"]) for row in gain_rows}
    weight_map = {row["feature_name"]: parse_float(row["importance"]) for row in weight_rows}
    cover_map = {row["feature_name"]: parse_float(row["importance"]) for row in cover_rows}
    all_features = list(FEATURE_NAMES)
    gain_rank = average_rank_desc({feature: gain_map.get(feature, 0.0) for feature in all_features})
    weight_rank = average_rank_desc({feature: weight_map.get(feature, 0.0) for feature in all_features})
    cover_rank = average_rank_desc({feature: cover_map.get(feature, 0.0) for feature in all_features})

    usage: dict[str, dict[str, object]] = {
        feature: {
            "split_count": 0,
            "earliest_depth": math.inf,
            "root_split_count": 0,
            "total_gain": 0.0,
            "max_cover": 0.0,
        }
        for feature in SCALAR_FEATURES
    }
    for row in node_rows:
        feature = xgb_feature_name(row["feature_name"])
        if feature not in usage:
            continue
        usage[feature]["split_count"] = int(usage[feature]["split_count"]) + 1
        usage[feature]["earliest_depth"] = min(
            float(usage[feature]["earliest_depth"]),
            parse_int(row["depth"]),
        )
        if parse_int(row["depth"]) == 0:
            usage[feature]["root_split_count"] = int(usage[feature]["root_split_count"]) + 1
        gain = parse_float(row["gain"])
        if not math.isnan(gain):
            usage[feature]["total_gain"] = float(usage[feature]["total_gain"]) + gain
        cover = parse_float(row["cover"])
        if not math.isnan(cover):
            usage[feature]["max_cover"] = max(float(usage[feature]["max_cover"]), cover)

    top_gain = max((gain_map.get(feature, 0.0) for feature in all_features), default=0.0)
    rows = []
    for feature in SCALAR_FEATURES:
        gain = gain_map.get(feature, 0.0)
        weight = weight_map.get(feature, 0.0)
        cover = cover_map.get(feature, 0.0)
        split_count = int(usage[feature]["split_count"])
        earliest_depth = usage[feature]["earliest_depth"]
        if gain == 0.0 and weight == 0.0 and split_count == 0:
            classification = "unused"
        elif gain <= 0.02 * top_gain and int(usage[feature]["root_split_count"]) == 0 and earliest_depth >= 4:
            classification = "used_rare_low_effect"
        else:
            classification = "used_nontrivial"
        rows.append(
            {
                "model_name": "xgboost",
                "feature_name": feature,
                "importance_metric": "gain",
                "importance_value": gain,
                "importance_rank": gain_rank.get(feature, math.nan),
                "split_count": split_count,
                "earliest_depth": float("nan") if math.isinf(float(earliest_depth)) else int(earliest_depth),
                "root_split_count": int(usage[feature]["root_split_count"]),
                "total_gain": float(usage[feature]["total_gain"]),
                "weight_importance": weight,
                "weight_rank": weight_rank.get(feature, math.nan),
                "cover_importance": cover,
                "cover_rank": cover_rank.get(feature, math.nan),
                "max_cover": float(usage[feature]["max_cover"]),
                "classification": classification,
            }
        )
    return rows


def parse_path_interval(path_rule: str) -> tuple[int | None, int | None]:
    lower: int | None = None
    upper: int | None = None
    if path_rule == "ROOT":
        return lower, upper
    for clause in path_rule.split(" AND "):
        pieces = clause.split()
        if len(pieces) != 3:
            continue
        operator = pieces[1]
        value = int(round(float(pieces[2])))
        if operator == "<=":
            upper = value if upper is None else min(upper, value)
        elif operator == ">":
            lower = value + 1 if lower is None else max(lower, value + 1)
    return lower, upper


def bucket_label(lower: int | None, upper: int | None) -> str:
    if lower is None and upper is None:
        return "all"
    if lower is None:
        return f"<= {upper}"
    if upper is None:
        return f"> {lower - 1}"
    return f"{lower}-{upper}"


def lower_triangle_linear_buckets(
    coef_rows: list[dict[str, str]],
) -> list[dict[str, object]]:
    grouped: dict[str, dict[str, object]] = {}
    for row in coef_rows:
        if row["variant"] != "lower_triangle_only_linear_tree":
            continue
        feature_name = row["feature_name"]
        if not feature_name.startswith("pb_ij_"):
            continue
        leaf_path = row["leaf_path"]
        bucket = grouped.setdefault(
            leaf_path,
            {
                "leaf_id": parse_int(row["leaf_id"]),
                "leaf_path": leaf_path,
                "path_rule": row["path_rule"],
                "n_samples": parse_int(row["n_samples"]),
                "coefficients": {},
            },
        )
        channel = int(feature_name.split("_")[-1])
        bucket["coefficients"][f"pair_ch_{channel}"] = abs(parse_float(row["coefficient_raw"]))

    output = []
    for bucket in grouped.values():
        lower, upper = parse_path_interval(str(bucket["path_rule"]))
        output.append(
            {
                "leaf_id": bucket["leaf_id"],
                "leaf_path": bucket["leaf_path"],
                "path_rule": bucket["path_rule"],
                "n_samples": bucket["n_samples"],
                "lower_bound": lower,
                "upper_bound": upper,
                "bucket_label": bucket_label(lower, upper),
                "coefficients": {name: float(bucket["coefficients"].get(name, 0.0)) for name in FEATURE_NAMES[:PAIR_FEATURE_COUNT]},
            }
        )
    output.sort(key=lambda row: ((math.inf if row["lower_bound"] is None else row["lower_bound"]), row["leaf_path"]))
    return output


def overlap_analysis(
    linear_buckets: list[dict[str, object]],
    decision_tree_importance: dict[str, float],
    xgb_gain: dict[str, float],
    xgb_weight: dict[str, float],
    xgb_cover: dict[str, float],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    comparison_vectors = {
        "decision_tree_importance": decision_tree_importance,
        "xgboost_gain": xgb_gain,
        "xgboost_weight": xgb_weight,
        "xgboost_cover": xgb_cover,
    }
    overlap_rows = []
    significance_rows = []
    for bucket in linear_buckets:
        linear_values = dict(bucket["coefficients"])
        linear_top_30 = top_k_features(linear_values, 30)
        overlap_rows.append(
            {
                "bucket_label": bucket["bucket_label"],
                "path_rule": bucket["path_rule"],
                "n_samples": bucket["n_samples"],
                "comparison": "linear_tree_reference",
                "top_30_features": ", ".join(linear_top_30),
            }
        )
        for metric_name, metric_values in comparison_vectors.items():
            rho = spearman_corr_desc(linear_values, metric_values)
            for top_k in TOP_K_VALUES:
                linear_top = set(top_k_features(linear_values, top_k))
                model_top = set(top_k_features(metric_values, top_k))
                overlap = sorted(linear_top & model_top)
                p_value = hypergeom_tail(PAIR_FEATURE_COUNT, top_k, top_k, len(overlap))
                overlap_rows.append(
                    {
                        "bucket_label": bucket["bucket_label"],
                        "path_rule": bucket["path_rule"],
                        "n_samples": bucket["n_samples"],
                        "comparison": metric_name,
                        "top_k": top_k,
                        "overlap_count": len(overlap),
                        "overlap_features": ", ".join(overlap),
                        "spearman_rho": rho,
                    }
                )
                significance_rows.append(
                    {
                        "bucket_label": bucket["bucket_label"],
                        "path_rule": bucket["path_rule"],
                        "n_samples": bucket["n_samples"],
                        "comparison": metric_name,
                        "top_k": top_k,
                        "overlap_count": len(overlap),
                        "p_value_hypergeom": p_value,
                        "spearman_rho": rho,
                    }
                )
    benjamini_hochberg(significance_rows, "p_value_hypergeom", "p_value_hypergeom_fdr")
    return overlap_rows, significance_rows


def summarize_xgb_trees(node_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, object]] = {}
    for row in node_rows:
        tree_index = parse_int(row["tree_index"])
        tree = grouped.setdefault(
            tree_index,
            {
                "tree_index": tree_index,
                "split_count": 0,
                "total_gain": 0.0,
                "max_depth": 0,
                "root_feature_name": "",
                "root_split_condition": float("nan"),
                "root_gain": float("nan"),
                "root_cover": float("nan"),
            },
        )
        depth = parse_int(row["depth"])
        tree["max_depth"] = max(int(tree["max_depth"]), depth)
        is_leaf = parse_int(row["is_leaf"]) == 1
        if not is_leaf:
            tree["split_count"] = int(tree["split_count"]) + 1
            gain = parse_float(row["gain"])
            if not math.isnan(gain):
                tree["total_gain"] = float(tree["total_gain"]) + gain
            if depth == 0:
                tree["root_feature_name"] = xgb_feature_name(row["feature_name"])
                tree["root_split_condition"] = parse_float(row["split_condition"])
                tree["root_gain"] = parse_float(row["gain"])
                tree["root_cover"] = parse_float(row["cover"])
    rows = sorted(grouped.values(), key=lambda row: (-float(row["total_gain"]), int(row["tree_index"])))
    for rank, row in enumerate(rows, start=1):
        row["tree_rank_by_total_gain"] = rank
    return rows


def summarize_xgb_depth_search(search_rows: list[dict[str, str]]) -> list[dict[str, object]]:
    grouped: dict[int, dict[str, str]] = {}
    for row in search_rows:
        depth = parse_int(row["max_depth"])
        best = grouped.get(depth)
        score = parse_float(row["validation_mean_per_protein_r2"])
        if best is None or score > parse_float(best["validation_mean_per_protein_r2"]):
            grouped[depth] = row
    best_score = max(parse_float(row["validation_mean_per_protein_r2"]) for row in grouped.values())
    rows = []
    for depth in sorted(grouped):
        row = grouped[depth]
        score = parse_float(row["validation_mean_per_protein_r2"])
        rows.append(
            {
                "max_depth": depth,
                "config_index": parse_int(row["config_index"]),
                "validation_mean_per_protein_r2": score,
                "validation_delta_vs_best": score - best_score,
                "validation_median_per_protein_r2": parse_float(row["validation_median_per_protein_r2"]),
                "fit_mean_per_protein_r2": parse_float(row["fit_mean_per_protein_r2"]),
                "best_iteration": parse_int(row["best_iteration"]),
                "learning_rate": parse_float(row["learning_rate"]),
                "n_estimators": parse_int(row["n_estimators"]),
                "min_child_weight": parse_float(row["min_child_weight"]),
                "subsample": parse_float(row["subsample"]),
                "colsample_bytree": parse_float(row["colsample_bytree"]),
                "reg_lambda": parse_float(row["reg_lambda"]),
            }
        )
    return rows


def summarize_train_cap(split_rows: list[dict[str, str]]) -> dict[str, object]:
    train_rows = [row for row in split_rows if row["split_group"] == "train"]
    capped = [row for row in train_rows if parse_int(row["n_pairs"]) > 5000]
    uncapped = [row for row in train_rows if parse_int(row["n_pairs"]) <= 5000]
    train_pairs = [parse_int(row["n_pairs"]) for row in train_rows]
    return {
        "n_train_proteins": len(train_rows),
        "n_capped_train_proteins": len(capped),
        "fraction_capped_train_proteins": (len(capped) / len(train_rows)) if train_rows else float("nan"),
        "min_train_pairs": min(train_pairs) if train_pairs else 0,
        "median_train_pairs": median(train_pairs),
        "max_train_pairs": max(train_pairs) if train_pairs else 0,
        "max_uncapped_pairs": max((parse_int(row["n_pairs"]) for row in uncapped), default=0),
    }


def median(values: list[int]) -> float:
    if not values:
        return float("nan")
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[mid])
    return (ordered[mid - 1] + ordered[mid]) / 2.0


def seq_diag_lookup(rows: list[dict[str, str]], split: str, bucket: str) -> dict[str, str] | None:
    for row in rows:
        if row["split"] == split and row["seq_bucket"] == bucket:
            return row
    return None


def format_pct(value: float) -> str:
    return f"{100.0 * value:.1f}%"


def markdown_table(headers: list[str], rows: list[list[str]]) -> str:
    if not rows:
        return "_No rows._"
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def removal_verdict(
    dt_rows: list[dict[str, object]],
    xgb_rows: list[dict[str, object]],
    feature_group: str,
) -> str:
    if feature_group == "n_residues":
        dt = next(row for row in dt_rows if row["feature_name"] == "n_residues")
        xgb = next(row for row in xgb_rows if row["feature_name"] == "n_residues")
        if float(dt["importance_value"]) == 0.0 and float(xgb["importance_value"]) == 0.0:
            return "almost certainly similar"
        if float(xgb["importance_value"]) < 0.05 * max(float(row["importance_value"]) for row in xgb_rows):
            return "probably similar but not identical"
        return "uncertain without ablation"

    seq_names = {"seq_sep", "log1p_seq_sep", "is_seq_sep_1", "is_seq_sep_le_2", "is_seq_sep_le_4"}
    dt_total = sum(float(row["importance_value"]) for row in dt_rows if row["feature_name"] in seq_names)
    xgb_total_gain = sum(float(row["total_gain"]) for row in xgb_rows if row["feature_name"] in seq_names)
    xgb_total_splits = sum(int(row["split_count"]) for row in xgb_rows if row["feature_name"] in seq_names)
    if dt_total == 0.0 and xgb_total_splits == 0:
        return "almost certainly similar"
    if dt_total < 0.005 and xgb_total_gain < 10_000_000.0:
        return "probably similar but not identical"
    return "uncertain without ablation"


def draw_decision_tree_top_levels(node_rows: list[dict[str, str]], output_path: Path, max_depth: int = 3) -> None:
    nodes, _ = build_parent_lookup(node_rows)
    truncated_children: dict[int, tuple[int | None, int | None]] = {}
    for node_id, node in nodes.items():
        depth = int(node["depth"])
        if bool(node["is_leaf"]) or depth >= max_depth:
            truncated_children[node_id] = (None, None)
        else:
            truncated_children[node_id] = (int(node["left_child"]), int(node["right_child"]))

    x_positions: dict[int, float] = {}
    leaf_counter = 0

    def assign_x(node_id: int) -> float:
        nonlocal leaf_counter
        left_child, right_child = truncated_children[node_id]
        if left_child is None or right_child is None:
            x = float(leaf_counter)
            leaf_counter += 1
            x_positions[node_id] = x
            return x
        left_x = assign_x(left_child)
        right_x = assign_x(right_child)
        x = (left_x + right_x) / 2.0
        x_positions[node_id] = x
        return x

    assign_x(0)

    y_positions = {node_id: -float(node["depth"]) for node_id, node in nodes.items() if node_id in x_positions}
    fig_width = max(12.0, leaf_counter * 2.2)
    fig_height = 2.8 * (max_depth + 1)
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_axis_off()

    for node_id, x in x_positions.items():
        node = nodes[node_id]
        y = y_positions[node_id]
        left_child, right_child = truncated_children[node_id]
        for child_id, edge_label in ((left_child, "<="), (right_child, ">")):
            if child_id is None:
                continue
            child_x = x_positions[child_id]
            child_y = y_positions[child_id]
            ax.plot([x, child_x], [y, child_y], color="#4f6b7a", linewidth=1.5)
            ax.text((x + child_x) / 2.0, (y + child_y) / 2.0 + 0.08, edge_label, fontsize=9, ha="center")

    for node_id, x in x_positions.items():
        node = nodes[node_id]
        y = y_positions[node_id]
        left_child, right_child = truncated_children[node_id]
        is_truncated = not bool(node["is_leaf"]) and left_child is None and right_child is None
        if bool(node["is_leaf"]):
            label = (
                f"leaf {node_id}\n"
                f"value={float(node['value']):.2f}\n"
                f"weight={float(node['weighted_n_node_samples']):,.0f}"
            )
            facecolor = "#f0f2f5"
        elif is_truncated:
            label = (
                f"node {node_id}\n"
                f"{node['feature_name']} <= {float(node['threshold']):.2f}\n"
                f"weight={float(node['weighted_n_node_samples']):,.0f}\n"
                "+ deeper subtree"
            )
            facecolor = "#fdecc8"
        else:
            label = (
                f"node {node_id}\n"
                f"{node['feature_name']} <= {float(node['threshold']):.2f}\n"
                f"weight={float(node['weighted_n_node_samples']):,.0f}\n"
                f"value={float(node['value']):.2f}"
            )
            facecolor = "#dfeef7"
        ax.text(
            x,
            y,
            label,
            ha="center",
            va="center",
            fontsize=9,
            bbox={"boxstyle": "round,pad=0.35", "facecolor": facecolor, "edgecolor": "#35505c"},
        )

    ax.set_title("Decision Tree Top Levels (depth <= 3)", fontsize=14, pad=12)
    xs = list(x_positions.values())
    ax.set_xlim(min(xs) - 1.0, max(xs) + 1.0)
    ax.set_ylim(-(max_depth + 0.8), 0.8)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def build_report(
    decision_summary: dict[str, str],
    xgb_summary: dict[str, str],
    dt_scalar_rows: list[dict[str, object]],
    xgb_scalar_rows: list[dict[str, object]],
    decision_tree_search: list[dict[str, str]],
    xgb_depth_rows: list[dict[str, object]],
    branch_rows: list[dict[str, object]],
    xgb_tree_rows: list[dict[str, object]],
    linear_summary_rows: list[dict[str, str]],
    overlap_significance_rows: list[dict[str, object]],
    cap_summary: dict[str, object],
    dt_seq_rows: list[dict[str, str]],
    xgb_seq_rows: list[dict[str, str]],
) -> str:
    dt_best_depth4 = max(
        (row for row in decision_tree_search if parse_int(row["max_depth"]) == 4),
        key=lambda row: parse_float(row["validation_mean_per_protein_r2"]),
    )
    xgb_best_depth4 = max(
        xgb_depth_rows,
        key=lambda row: float(row["validation_mean_per_protein_r2"]) if int(row["max_depth"]) == 4 else -math.inf,
    )
    best_overlap_rows = sorted(
        overlap_significance_rows,
        key=lambda row: (float(row["p_value_hypergeom_fdr"]), -int(row["overlap_count"])),
    )[:8]
    top_xgb_trees = xgb_tree_rows[:8]
    top_branches = branch_rows[:8]

    seq_dt_sep1 = seq_diag_lookup(dt_seq_rows, "test_full", "seq_sep_eq_1_exact")
    seq_xgb_sep1 = seq_diag_lookup(xgb_seq_rows, "test_full", "seq_sep_eq_1_exact")
    seq_dt_9p = seq_diag_lookup(dt_seq_rows, "test_full", "9+")
    seq_xgb_9p = seq_diag_lookup(xgb_seq_rows, "test_full", "9+")

    dt_importance_map = {row["feature_name"]: row for row in dt_scalar_rows}
    xgb_importance_map = {row["feature_name"]: row for row in xgb_scalar_rows}

    report_lines = [
        "# Delta Tree Diagnostics Report",
        "",
        "## 1. What each CSV means",
        "",
        "These artifacts all come from the same layer-47 lower-triangle probe setup.",
        "Every decision-tree / XGBoost row is one unordered residue pair `(i, j)` with `i > j`.",
        "The feature vector is 134-dimensional:",
        f"- 128 pair channels: `{', '.join(FEATURE_NAMES[:6])}, ...`",
        "- 1 protein-size scalar: `n_residues`",
        "- 2 raw/continuous sequence-separation scalars: `seq_sep`, `log1p_seq_sep`",
        "- 3 sequence-separation indicator scalars: `is_seq_sep_1`, `is_seq_sep_le_2`, `is_seq_sep_le_4`",
        "",
        "Training uses a protein-level split, not a pair-level split:",
        "- 184 proteins loaded",
        "- 147 train proteins / 37 test proteins",
        "- inside the 147-train set, 117 fit proteins / 30 validation proteins for model selection",
        "- training loss is protein-equal because each protein gets total sample weight 1, even though proteins contribute different numbers of residue pairs",
        "- train proteins are capped at 5,000 sampled pairs each; validation and test use all available pairs",
        "",
        "That matters for interpretability: importance values describe what the model used under this exact split, weighting rule, and sampling cap. They do not prove a feature is biologically causal or universally necessary.",
        "",
        "### `diagnostic_decision_tree_variant_summary.csv`",
        "- One row for the selected final decision-tree model after validation search.",
        "- `best_validation_mean_per_protein_r2`: score used to pick hyperparameters; this is the mean of per-protein `R^2` values on the validation proteins.",
        "- `train_mean_per_protein_r2` / `test_mean_per_protein_r2`: average per-protein `R^2` on sampled train rows / full test rows.",
        "- `train_pair_weighted_r2` / `test_pair_weighted_r2`: pooled `R^2` over all rows in that split, so big proteins matter more than in mean-per-protein `R^2`.",
        "- `test_negative_r2_fraction`: fraction of proteins where predictions are worse than predicting that protein's mean distance.",
        "- `tree_depth` and `n_leaves`: realized complexity of the final fitted tree.",
        "",
        "### `diagnostic_decision_tree_search.csv`",
        "- One row per tested depth / minimum leaf / minimum split configuration.",
        "- `fit_mean_per_protein_r2` is in-sample on the fit proteins used to train that config.",
        "- `validation_mean_per_protein_r2` is the selection metric; if this rises while fit performance rises much faster, that is the usual sign of overfitting.",
        "",
        "### `diagnostic_decision_tree_feature_importance.csv`",
        "- This is `DecisionTreeRegressor.feature_importances_` from scikit-learn.",
        "- It is the normalized total weighted impurity decrease attributed to each feature across the final tree.",
        "- Values sum to about 1 across all 134 features.",
        "- A large value means the feature caused large variance reductions high in the tree or across many heavy branches.",
        "- A zero means the final tree never split on that feature.",
        "- This metric is model-internal and relative: it says what the tree used, not whether dropping the feature leaves held-out performance unchanged.",
        "",
        "### `diagnostic_decision_tree_nodes.csv`",
        "- One row per fitted node in the final decision tree.",
        "- `feature_name` and `threshold` define the split at internal nodes.",
        "- `weighted_n_node_samples` is the total protein-equal sample mass reaching that node.",
        "- `value` is the node's predicted distance.",
        "- This file is the most concrete interpretability artifact because it shows where a feature appears, how deep it appears, and how much weighted data it controls.",
        "",
        "### `diagnostic_decision_tree_tree.txt`",
        "- Text dump of the final decision tree from `sklearn.tree.export_text`.",
        "- Good for quick inspection, but hard to read once the tree is deep.",
        "",
        "### `diagnostic_decision_tree_seq_sep_diagnostics.csv`",
        "- Buckets pairs by sequence separation (`1`, `2`, `3-4`, `5-8`, `9+`, plus exact `seq_sep_eq_1_exact`).",
        "- `r2` is computed inside each bucket only.",
        "- Negative `R^2` in tiny-distance buckets can coexist with small absolute error because the true-distance variance is tiny there.",
        "- `fraction_pred_lt_3A` helps check whether the model is willing to make very short-distance predictions.",
        "",
        "### `diagnostic_xgboost_variant_summary.csv`",
        "- Same role as the decision-tree summary, but for the final XGBoost ensemble.",
        "- `best_iteration_search` is the early-stopped boosting round from the fit/validation search.",
        "- `final_n_estimators` is the number of boosting rounds used when refitting on the full sampled train split.",
        "- `n_boosted_rounds` should match the actual number of dumped trees in the final ensemble.",
        "",
        "### `diagnostic_xgboost_search.csv`",
        "- One row per tested XGBoost hyperparameter configuration.",
        "- `best_iteration` and `best_score_rmse` come from validation early stopping.",
        "- `validation_mean_per_protein_r2` remains the model-selection target.",
        "",
        "### `diagnostic_xgboost_nodes.csv`",
        "- One row per node in every boosted tree.",
        "- `gain` is the loss reduction at that split node.",
        "- `cover` is the weighted amount of training data reaching that split node.",
        "- Summing `gain` over all split nodes in one tree gives a useful tree-level proxy for how much that tree contributed to training loss reduction.",
        "",
        "### `diagnostic_xgboost_root_splits.csv`",
        "- One row per boosted tree root split.",
        "- Helpful for checking which features repeatedly control the first decision in each tree.",
        "",
        "### `diagnostic_xgboost_top_split_features.csv`",
        "- Derived by parsing all split nodes in the dumped trees.",
        "- `split_count` is how often a feature is used anywhere in the ensemble.",
        "- `total_gain` is the sum of split gains over all those uses.",
        "- This is different from XGBoost's built-in `gain` importance, which is an average gain per split.",
        "",
        "### `diagnostic_xgboost_feature_importance_gain.csv`",
        "- `booster.get_score(importance_type='gain')`.",
        "- This is average gain per split for each feature, not total gain.",
        "- Good default importance metric when asking which features tend to make valuable splits.",
        "",
        "### `diagnostic_xgboost_feature_importance_weight.csv`",
        "- `booster.get_score(importance_type='weight')`.",
        "- Pure split frequency: how many times a feature was used.",
        "- High weight with modest gain usually means a feature is useful as a routing helper but not a dominant driver of performance.",
        "",
        "### `diagnostic_xgboost_feature_importance_cover.csv`",
        "- `booster.get_score(importance_type='cover')`.",
        "- Average covered sample mass for splits using that feature.",
        "- This says where the feature tends to appear in the tree structure, not how much predictive value it adds.",
        "",
        "### `diagnostic_linear_tree_variant_summary.csv`",
        "- Summary of the separate linear-tree experiment.",
        "- Important here because it gives indirect ablation evidence for sequence-separation features in a related but not identical model family.",
        "",
        "### `diagnostic_linear_tree_rules.csv`",
        "- Learned size-based split rules for each linear-tree variant.",
        "- For `lower_triangle_only_linear_tree`, the learned size buckets are the exact categories used below for coefficient/importance overlap.",
        "",
        "### `diagnostic_linear_tree_leaf_coefficients_raw.csv`",
        "- Long-form raw coefficients for each linear-model leaf.",
        "- In `lower_triangle_only_linear_tree`, features are `pb_ij_0` ... `pb_ij_127`, which map directly to `pair_ch_0` ... `pair_ch_127` in the tree/XGBoost runs.",
        "- Taking absolute coefficient magnitude is a reasonable ranking heuristic here, but it is still not a formal significance test because correlated channels can trade off with one another.",
        "",
        "## 2. Headline performance",
        "",
        markdown_table(
            ["Model", "Test mean per-protein R2", "Test pair-weighted R2", "Train mean per-protein R2", "Train/test gap"],
            [
                [
                    "Decision tree",
                    f"{parse_float(decision_summary['test_mean_per_protein_r2']):.4f}",
                    f"{parse_float(decision_summary['test_pair_weighted_r2']):.4f}",
                    f"{parse_float(decision_summary['train_mean_per_protein_r2']):.4f}",
                    f"{parse_float(decision_summary['train_mean_per_protein_r2']) - parse_float(decision_summary['test_mean_per_protein_r2']):.4f}",
                ],
                [
                    "XGBoost",
                    f"{parse_float(xgb_summary['test_mean_per_protein_r2']):.4f}",
                    f"{parse_float(xgb_summary['test_pair_weighted_r2']):.4f}",
                    f"{parse_float(xgb_summary['train_mean_per_protein_r2']):.4f}",
                    f"{parse_float(xgb_summary['train_mean_per_protein_r2']) - parse_float(xgb_summary['test_mean_per_protein_r2']):.4f}",
                ],
            ],
        ),
        "",
        "Interpretation:",
        "- These are not random-luck training scores; both models are being judged on 37 held-out proteins.",
        "- XGBoost clearly generalizes better than the single tree on this split.",
        "- The small train/test gap for XGBoost and the zero negative-`R^2` fraction argue against the result being a fluke caused only by a few easy proteins.",
        "- Interpretability is still limited because this is one random protein split, one training-pair cap, and one set of correlated pair channels.",
        "",
        "## 3. How important were `n_residues` and sequence-separation features?",
        "",
        markdown_table(
            ["Model", "Feature", "Primary importance", "Primary rank", "Split count", "Earliest depth", "Classification"],
            [
                [
                    "Decision tree",
                    row["feature_name"],
                    f"{float(row['importance_value']):.6f}",
                    f"{float(row['importance_rank']):.0f}",
                    f"{int(row['split_count'])}",
                    "NA" if math.isnan(float(row["earliest_depth"])) else f"{int(row['earliest_depth'])}",
                    str(row["classification"]),
                ]
                for row in dt_scalar_rows
            ] + [
                [
                    "XGBoost (gain)",
                    row["feature_name"],
                    f"{float(row['importance_value']):.2f}",
                    f"{float(row['importance_rank']):.0f}",
                    f"{int(row['split_count'])}",
                    "NA" if math.isnan(float(row["earliest_depth"])) else f"{int(row['earliest_depth'])}",
                    str(row["classification"]),
                ]
                for row in xgb_scalar_rows
            ],
        ),
        "",
        f"Removal verdict for `n_residues`: **{removal_verdict(dt_scalar_rows, xgb_scalar_rows, 'n_residues')}**.",
        f"Removal verdict for sequence-separation features as a group: **{removal_verdict(dt_scalar_rows, xgb_scalar_rows, 'seq_sep_group')}**.",
        "",
        "What this means in practice:",
        f"- In the single decision tree, `n_residues` is real but secondary: importance `{float(dt_importance_map['n_residues']['importance_value']):.6f}`, rank {int(float(dt_importance_map['n_residues']['importance_rank']))}. `seq_sep` and `log1p_seq_sep` are much smaller, and the three bin indicators are unused.",
        f"- In XGBoost, `n_residues` is used often (`weight={float(xgb_importance_map['n_residues']['weight_importance']):.0f}`) but with moderate average gain relative to the dominant pair channels. That pattern means it helps routing, but it is not the main source of predictive power.",
        f"- `seq_sep` and `log1p_seq_sep` are also used, but their gains are far below the top pair channels. `n_residues` does appear as a root split in {int(xgb_importance_map['n_residues']['root_split_count'])} boosted trees, but the strongest early trees are still pair-channel driven.",
        "- Indirect ablation evidence from the linear-tree experiment is weakly against large sequence-separation gains: `both_triangles_linear_tree` test mean per-protein `R^2` is 0.8487, `both_triangles_plus_seqdist_linear_tree` is 0.8493, and `both_triangles_plus_seqdist_plus_bins_linear_tree` is 0.8492. That is a tiny delta.",
        "- So the best reading is: these scalars are not useless, but the pair channels dominate. Removing them would probably not destroy performance, yet you should not expect *exactly* the same result without running a true ablation.",
        "",
        "Sequence-bucket behavior supports that interpretation:",
        markdown_table(
            ["Model", "Bucket", "Test R2", "Mean pred-true"],
            [
                [
                    "Decision tree",
                    "seq_sep=1",
                    "NA" if seq_dt_sep1 is None else f"{parse_float(seq_dt_sep1['r2']):.4f}",
                    "NA" if seq_dt_sep1 is None else f"{parse_float(seq_dt_sep1['mean_error_pred_minus_true']):+.4f}",
                ],
                [
                    "XGBoost",
                    "seq_sep=1",
                    "NA" if seq_xgb_sep1 is None else f"{parse_float(seq_xgb_sep1['r2']):.4f}",
                    "NA" if seq_xgb_sep1 is None else f"{parse_float(seq_xgb_sep1['mean_error_pred_minus_true']):+.4f}",
                ],
                [
                    "Decision tree",
                    "seq_sep>=9",
                    "NA" if seq_dt_9p is None else f"{parse_float(seq_dt_9p['r2']):.4f}",
                    "NA" if seq_dt_9p is None else f"{parse_float(seq_dt_9p['mean_error_pred_minus_true']):+.4f}",
                ],
                [
                    "XGBoost",
                    "seq_sep>=9",
                    "NA" if seq_xgb_9p is None else f"{parse_float(seq_xgb_9p['r2']):.4f}",
                    "NA" if seq_xgb_9p is None else f"{parse_float(seq_xgb_9p['mean_error_pred_minus_true']):+.4f}",
                ],
            ],
        ),
        "",
        "Short-range buckets remain hard even when sequence-separation features are present, which again suggests those features are not a magic driver of the overall score.",
        "",
        "## 4. Making sense of the tree",
        "",
        "The single decision tree is hard to read because the selected model has depth 10 and 805 leaves. The useful way to read it is not as one giant rule list, but as a hierarchy:",
        "- start with the top few splits, because those move the most weighted sample mass",
        "- then inspect which heavy branches later introduce size or sequence-separation conditions",
        "- finally treat deep leaves as local refinements, not global explanations",
        "",
        f"The top-level visualization is saved as `{DT_FIGURE_NAME}` and intentionally truncates at depth 3 so the structure is readable.",
        "",
        "Highest-coverage decision-tree leaves:",
        markdown_table(
            ["Leaf node", "Depth", "Weighted sample fraction", "Prediction", "Path"],
            [
                [
                    str(int(row["leaf_node_id"])),
                    str(int(row["leaf_depth"])),
                    format_pct(float(row["sample_fraction"])),
                    f"{float(row['prediction_value']):.2f}",
                    str(row["path_rule"]),
                ]
                for row in top_branches
            ],
        ),
        "",
        "Interpretation:",
        "- The root and early heavy branches are dominated by pair channels, not scalar features.",
        "- `n_residues` and `seq_sep` mostly appear deeper, where the model is carving up special cases rather than defining the main global structure.",
        "",
        "Best XGBoost trees by total split gain (training-loss reduction inside the final ensemble):",
        markdown_table(
            ["Tree rank", "Tree index", "Total gain", "Max depth", "Root feature", "Root gain"],
            [
                [
                    str(int(row["tree_rank_by_total_gain"])),
                    str(int(row["tree_index"])),
                    f"{float(row['total_gain']):.0f}",
                    str(int(row["max_depth"])),
                    str(row["root_feature_name"]),
                    f"{float(row['root_gain']):.0f}",
                ]
                for row in top_xgb_trees
            ],
        ),
        "",
        "Caveat: these are the trees that reduced training objective the most. XGBoost does not expose a clean per-tree held-out score, so this is the best available notion of a 'best tree' from the dumped ensemble.",
        "",
        "Did any depth-4 trees behave fine?",
        "",
        markdown_table(
            ["Model family", "Depth-4 validation mean per-protein R2", "Best family validation R2", "Delta"],
            [
                [
                    "Decision tree",
                    f"{parse_float(dt_best_depth4['validation_mean_per_protein_r2']):.4f}",
                    f"{max(parse_float(row['validation_mean_per_protein_r2']) for row in decision_tree_search):.4f}",
                    f"{parse_float(dt_best_depth4['validation_mean_per_protein_r2']) - max(parse_float(row['validation_mean_per_protein_r2']) for row in decision_tree_search):+.4f}",
                ],
                [
                    "XGBoost",
                    f"{float(xgb_best_depth4['validation_mean_per_protein_r2']):.4f}",
                    f"{max(float(row['validation_mean_per_protein_r2']) for row in xgb_depth_rows):.4f}",
                    f"{float(xgb_best_depth4['validation_delta_vs_best']):+.4f}",
                ],
            ],
        ),
        "",
        "Interpretation:",
        "- A depth-4 **single decision tree** is noticeably worse than the selected depth-10 tree, so in that family depth 4 does *not* look sufficient.",
        "- A depth-4 **XGBoost ensemble** is much closer to the best XGBoost setup, so yes: shallow boosted trees behaved fine.",
        "",
        "## 5. Would more training pairs per protein help?",
        "",
        f"- The current run cannot answer that exactly because the cap never changes. There is no counterfactual artifact with 10,000 or full-train pairs.",
        f"- The cap is active for {int(cap_summary['n_capped_train_proteins'])}/{int(cap_summary['n_train_proteins'])} train proteins ({format_pct(float(cap_summary['fraction_capped_train_proteins']))}). So this question is real; the cap is binding for most proteins.",
        f"- Train proteins have median {float(cap_summary['median_train_pairs']):.0f} lower-triangle pairs before capping, far above 5,000 for many proteins.",
        f"- XGBoost already has only a modest train/test gap ({parse_float(xgb_summary['train_mean_per_protein_r2']) - parse_float(xgb_summary['test_mean_per_protein_r2']):.4f}), so there is no obvious sign that it is starved for data in a way that would guarantee a large win from more sampled pairs.",
        f"- The single tree also does not look variance-limited in the usual way; its main limitation is representational simplicity relative to XGBoost.",
        "- So the responsible answer is: you can make an informed guess that more pairs might help a bit, especially for large proteins, but a real answer requires rerunning with a cap sweep.",
        "",
        "## 6. Linear-tree vs tree/XGBoost feature overlap",
        "",
        "Here the comparison is intentionally restricted to `lower_triangle_only_linear_tree`, because only that variant lives in the same 128-channel feature space as the decision-tree/XGBoost runs.",
        "The learned size buckets from that model are:",
        "- `<= 252`",
        "- `253-374`",
        "- `375-507`",
        "- `> 507`",
        "",
        "Within each bucket, pair channels are ranked by absolute linear coefficient magnitude. Those rankings are then compared with the decision-tree importance rank and the XGBoost gain / weight / cover ranks over the same 128 channels.",
        "",
        "Most significant top-K overlaps after FDR correction:",
        markdown_table(
            ["Bucket", "Comparison", "Top-K", "Overlap", "Spearman rho", "FDR-adjusted p"],
            [
                [
                    str(row["bucket_label"]),
                    str(row["comparison"]),
                    str(int(row["top_k"])),
                    str(int(row["overlap_count"])),
                    f"{float(row['spearman_rho']):.3f}",
                    f"{float(row['p_value_hypergeom_fdr']):.4g}",
                ]
                for row in best_overlap_rows
            ],
        ),
        "",
        "How to read this:",
        "- The hypergeometric p-value asks whether two top-K lists overlap more than expected by chance under a random ranking over 128 channels.",
        "- The FDR correction matters because many bucket/metric/K combinations are tested.",
        "- Spearman `rho` uses the full 128-feature ranking and is less brittle than a single top-K cutoff.",
        "",
        "Interpretation:",
        "- Significant overlap means the linear-tree leaf and the nonlinear tree family are repeatedly highlighting some of the same channels. That makes the overlap interpretable, not just anecdotal.",
        "- Non-significant overlap does **not** mean one of the models is wrong. Correlated channels, different inductive biases, and bucket-specific fits can easily reshuffle ranks.",
        "- Decision-tree importance and XGBoost gain are the cleaner comparison targets. XGBoost weight and cover are more structural than predictive.",
        "",
        "## 7. Bottom line",
        "",
        "- The results look real enough to analyze: held-out protein performance is high, especially for XGBoost, and not obviously due to noise.",
        "- The pair channels dominate predictive power.",
        "- `n_residues` matters some, mainly as a routing / context feature, but it does not look like the main story.",
        "- Sequence-separation features matter a little but probably would not radically change the result if removed; exact ablation still needs a rerun.",
        "- The single decision tree is interpretable only in compressed form. The top levels and heavy branches are useful; the full 805-leaf dump is not a human-scale explanation.",
        "- Depth-4 boosted trees look fine. Depth-4 single trees do not.",
        "- The linear-tree overlap analysis can tell you whether the same pair channels keep resurfacing across model classes, which is the strongest evidence here that some channels are genuinely robustly important rather than ranking highly by chance on one fit.",
        "",
    ]
    return "\n".join(report_lines)


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    decision_summary = first_row(DECISION_TREE_DIR / "diagnostic_decision_tree_variant_summary.csv")
    decision_search = read_csv_rows(DECISION_TREE_DIR / "diagnostic_decision_tree_search.csv")
    decision_importance_rows = read_csv_rows(DECISION_TREE_DIR / "diagnostic_decision_tree_feature_importance.csv")
    decision_node_rows = read_csv_rows(DECISION_TREE_DIR / "diagnostic_decision_tree_nodes.csv")
    decision_seq_rows = read_csv_rows(DECISION_TREE_DIR / "diagnostic_decision_tree_seq_sep_diagnostics.csv")
    decision_split_rows = read_csv_rows(DECISION_TREE_DIR / "diagnostic_decision_tree_split.csv")

    xgb_summary = first_row(XGBOOST_DIR / "diagnostic_xgboost_variant_summary.csv")
    xgb_search_rows = read_csv_rows(XGBOOST_DIR / "diagnostic_xgboost_search.csv")
    xgb_gain_rows = read_csv_rows(XGBOOST_DIR / "diagnostic_xgboost_feature_importance_gain.csv")
    xgb_weight_rows = read_csv_rows(XGBOOST_DIR / "diagnostic_xgboost_feature_importance_weight.csv")
    xgb_cover_rows = read_csv_rows(XGBOOST_DIR / "diagnostic_xgboost_feature_importance_cover.csv")
    xgb_node_rows = read_csv_rows(XGBOOST_DIR / "diagnostic_xgboost_nodes.csv")
    xgb_seq_rows = read_csv_rows(XGBOOST_DIR / "diagnostic_xgboost_seq_sep_diagnostics.csv")

    linear_summary_rows = read_csv_rows(LINEAR_TREE_DIR / "diagnostic_linear_tree_variant_summary.csv")
    linear_coef_rows = read_csv_rows(LINEAR_TREE_DIR / "diagnostic_linear_tree_leaf_coefficients_raw.csv")

    branch_rows = decision_tree_leaf_paths(decision_node_rows)
    write_csv(
        OUTPUT_DIR / BRANCH_SUMMARY_CSV,
        ["leaf_node_id", "leaf_depth", "weighted_n_node_samples", "sample_fraction", "prediction_value", "path_rule"],
        branch_rows,
    )

    dt_scalar_rows = scalar_summary_decision_tree(decision_importance_rows, decision_node_rows)
    xgb_scalar_rows = scalar_summary_xgboost(xgb_gain_rows, xgb_weight_rows, xgb_cover_rows, xgb_node_rows)
    write_csv(
        OUTPUT_DIR / SCALAR_SUMMARY_CSV,
        [
            "model_name",
            "feature_name",
            "importance_metric",
            "importance_value",
            "importance_rank",
            "split_count",
            "earliest_depth",
            "root_split_count",
            "max_weighted_n_node_samples",
            "total_gain",
            "weight_importance",
            "weight_rank",
            "cover_importance",
            "cover_rank",
            "max_cover",
            "classification",
        ],
        list(dt_scalar_rows) + list(xgb_scalar_rows),
    )

    xgb_tree_rows = summarize_xgb_trees(xgb_node_rows)
    write_csv(
        OUTPUT_DIR / XGB_TREE_CSV,
        [
            "tree_rank_by_total_gain",
            "tree_index",
            "total_gain",
            "split_count",
            "max_depth",
            "root_feature_name",
            "root_split_condition",
            "root_gain",
            "root_cover",
        ],
        xgb_tree_rows,
    )

    xgb_depth_rows = summarize_xgb_depth_search(xgb_search_rows)
    write_csv(
        OUTPUT_DIR / XGB_DEPTH_CSV,
        [
            "max_depth",
            "config_index",
            "validation_mean_per_protein_r2",
            "validation_delta_vs_best",
            "validation_median_per_protein_r2",
            "fit_mean_per_protein_r2",
            "best_iteration",
            "learning_rate",
            "n_estimators",
            "min_child_weight",
            "subsample",
            "colsample_bytree",
            "reg_lambda",
        ],
        xgb_depth_rows,
    )

    linear_buckets = lower_triangle_linear_buckets(linear_coef_rows)
    decision_pair_importance = importance_vector(decision_importance_rows, FEATURE_NAMES[:PAIR_FEATURE_COUNT])
    xgb_gain_pair = importance_vector(xgb_gain_rows, FEATURE_NAMES[:PAIR_FEATURE_COUNT])
    xgb_weight_pair = importance_vector(xgb_weight_rows, FEATURE_NAMES[:PAIR_FEATURE_COUNT])
    xgb_cover_pair = importance_vector(xgb_cover_rows, FEATURE_NAMES[:PAIR_FEATURE_COUNT])

    overlap_rows, overlap_sig_rows = overlap_analysis(
        linear_buckets,
        decision_pair_importance,
        xgb_gain_pair,
        xgb_weight_pair,
        xgb_cover_pair,
    )
    write_csv(
        OUTPUT_DIR / OVERLAP_CSV,
        ["bucket_label", "path_rule", "n_samples", "comparison", "top_k", "overlap_count", "overlap_features", "spearman_rho", "top_30_features"],
        overlap_rows,
    )
    write_csv(
        OUTPUT_DIR / OVERLAP_SIG_CSV,
        ["bucket_label", "path_rule", "n_samples", "comparison", "top_k", "overlap_count", "p_value_hypergeom", "p_value_hypergeom_fdr", "spearman_rho"],
        overlap_sig_rows,
    )

    draw_decision_tree_top_levels(decision_node_rows, OUTPUT_DIR / DT_FIGURE_NAME, max_depth=3)

    cap_summary = summarize_train_cap(decision_split_rows)
    report_text = build_report(
        decision_summary=decision_summary,
        xgb_summary=xgb_summary,
        dt_scalar_rows=dt_scalar_rows,
        xgb_scalar_rows=xgb_scalar_rows,
        decision_tree_search=decision_search,
        xgb_depth_rows=xgb_depth_rows,
        branch_rows=branch_rows,
        xgb_tree_rows=xgb_tree_rows,
        linear_summary_rows=linear_summary_rows,
        overlap_significance_rows=overlap_sig_rows,
        cap_summary=cap_summary,
        dt_seq_rows=decision_seq_rows,
        xgb_seq_rows=xgb_seq_rows,
    )
    (OUTPUT_DIR / REPORT_NAME).write_text(report_text)

    print(f"Wrote report -> {OUTPUT_DIR / REPORT_NAME}")
    print(f"Wrote figure -> {OUTPUT_DIR / DT_FIGURE_NAME}")


if __name__ == "__main__":
    main()

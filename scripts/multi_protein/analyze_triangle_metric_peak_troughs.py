#!/usr/bin/env python3
"""Test search-adjusted local extrema in focused triangle-metric trajectories.

This local-only workflow reads the canonical per-protein triangle metrics,
tests every eligible layer before candidate filtering, and writes statistical
tables plus annotated layer-trajectory figures. It never reads decodability or
R2 results.
"""

from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


REPO_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(REPO_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(REPO_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


DEFAULT_INPUT_PATH = (
    REPO_ROOT
    / "delta_files"
    / "triangle_violation_subgroups"
    / "per_protein_triangle_metrics.csv"
)
DEFAULT_OUTPUT_ROOT = (
    REPO_ROOT
    / "delta_files"
    / "analysis"
    / "triangle_violation_subgroups"
    / "peak_trough_significance"
)
LAYERS = tuple(range(48))
ELIGIBLE_LAYERS = tuple(range(5, 43))
WINDOW = 5
EXPECTED_PROTEINS = 111
EXPECTED_FOCUSED_ROWS = 277_056
EXPECTED_TRAJECTORIES = 52
EXPECTED_TEST_ROWS = 7_904
EXPECTED_BH_FAMILY_SIZE = 3_952
ALPHA = 0.05
ALIGNMENT_TOLERANCE = 2

MODEL_SPECS = {
    "general_base256_combined": ("Shared linear", "#4C78A8"),
    "per_protein_base256_combined": ("Per-protein linear", "#7F7F7F"),
    "shared_xgboost_base256_combined": ("XGBoost", "#E45756"),
    "analysis_1_sequence_gated": ("Sequence gated", "#F58518"),
}
MODEL_ORDER = tuple(MODEL_SPECS)

CATEGORY_SPECS = {
    "all_triangles": {
        "figure_number": 1,
        "title": "All triangles",
        "labels": ("all",),
    },
    "rms_size": {
        "figure_number": 2,
        "title": "Reference RMS-size tertiles",
        "labels": ("small", "medium", "large"),
    },
    "area": {
        "figure_number": 3,
        "title": "Reference area tertiles",
        "labels": ("low", "medium", "high"),
    },
    "relative_slack": {
        "figure_number": 4,
        "title": "Reference relative-slack tertiles",
        "labels": ("low", "medium", "high"),
    },
    "angle_shape": {
        "figure_number": 5,
        "title": "Reference largest-angle classes",
        "labels": ("clearly_acute", "near_right", "clearly_obtuse"),
    },
}

METRIC_SPECS = {
    "violation_fraction": {
        "role": "secondary",
        "title": "Triangle-inequality violation fraction",
        "ylabel": "Fraction of triangles",
        "display_scale": 100.0,
        "display_unit": "percentage points",
    },
    "mean_relative_violation_all": {
        "role": "primary",
        "title": "Normalized mean triangle violation",
        "ylabel": "Mean relative violation per triangle",
        "display_scale": 100.0,
        "display_unit": "normalized-burden points",
    },
}
METRIC_ORDER = tuple(METRIC_SPECS)
DIRECTIONS = ("peak", "trough")

INPUT_COLUMNS = (
    "analysis_name",
    "model_family",
    "layer",
    "protein_id",
    "evaluation_scope",
    "category_system",
    "category_label",
    "violation_fraction",
    "mean_relative_violation_all",
    "support_status",
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def relative(path: Path) -> str:
    return path.resolve().relative_to(REPO_ROOT).as_posix()


def display_category(label: str) -> str:
    if label == "all":
        return "All triangles"
    return label.replace("__", " · ").replace("_", " ").title()


def expected_category_pairs() -> set[tuple[str, str]]:
    return {
        (system, label)
        for system, spec in CATEGORY_SPECS.items()
        for label in spec["labels"]
    }


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 9,
            "axes.titlesize": 10,
            "axes.labelsize": 9,
            "xtick.labelsize": 8,
            "ytick.labelsize": 8,
            "legend.fontsize": 8,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.dpi": 220,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_and_validate_input(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path, usecols=INPUT_COLUMNS, dtype={"protein_id": str})
    frame = frame[
        frame["analysis_name"].isin(MODEL_ORDER)
        & frame["category_system"].isin(CATEGORY_SPECS)
    ].copy()
    if len(frame) != EXPECTED_FOCUSED_ROWS:
        raise ValueError(
            f"Expected {EXPECTED_FOCUSED_ROWS} focused rows; found {len(frame)}"
        )
    if set(frame["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError("Input does not contain exactly the four focused models")
    if set(frame["layer"].astype(int)) != set(LAYERS):
        raise ValueError("Input does not contain exactly layers 0-47")
    if set(zip(frame["category_system"], frame["category_label"])) != expected_category_pairs():
        raise ValueError("Focused category labels do not match the analysis contract")
    if set(frame["evaluation_scope"]) != {"all_edges"}:
        raise ValueError("Focused rows must use only the all_edges scope")

    key = [
        "analysis_name",
        "layer",
        "protein_id",
        "category_system",
        "category_label",
    ]
    if frame.duplicated(key).any():
        raise ValueError("Input contains duplicate model-layer-protein-category rows")
    outcomes = frame[list(METRIC_ORDER)].to_numpy(dtype=float)
    if not np.isfinite(outcomes).all():
        raise ValueError("Focused outcome values must all be finite")

    protein_sets = frame.groupby(
        ["analysis_name", "category_system", "category_label"], sort=False
    )["protein_id"].agg(lambda values: frozenset(values))
    if len(protein_sets) != EXPECTED_TRAJECTORIES:
        raise ValueError(
            f"Expected {EXPECTED_TRAJECTORIES} trajectories; found {len(protein_sets)}"
        )
    first_set = protein_sets.iloc[0]
    if len(first_set) != EXPECTED_PROTEINS or any(ids != first_set for ids in protein_sets):
        raise ValueError("All focused trajectories must use the same 111 proteins")

    coverage = frame.groupby(
        ["analysis_name", "category_system", "category_label"], sort=False
    )["layer"].nunique()
    if not coverage.eq(len(LAYERS)).all():
        raise ValueError("Every focused trajectory must contain all 48 layers")
    return frame


def ols_slope(values: Sequence[float]) -> float:
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    if y.ndim != 1 or len(y) < 2 or not np.isfinite(y).all():
        return math.nan
    centered_x = x - x.mean()
    return float(np.dot(centered_x, y - y.mean()) / np.dot(centered_x, centered_x))


def one_sample_greater(values: Sequence[float]) -> dict[str, float | int | str]:
    sample = np.asarray(values, dtype=float)
    if sample.ndim != 1 or len(sample) < 2:
        return {
            "n": int(sample.size),
            "mean": math.nan,
            "sd": math.nan,
            "se": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "t": math.nan,
            "df": max(int(sample.size) - 1, 0),
            "p": math.nan,
            "cohen_dz": math.nan,
            "status": "insufficient_n",
        }
    if not np.isfinite(sample).all():
        return {
            "n": int(sample.size),
            "mean": math.nan,
            "sd": math.nan,
            "se": math.nan,
            "ci_low": math.nan,
            "ci_high": math.nan,
            "t": math.nan,
            "df": int(sample.size) - 1,
            "p": math.nan,
            "cohen_dz": math.nan,
            "status": "nonfinite_input",
        }

    n = int(sample.size)
    mean = float(sample.mean())
    sd = float(sample.std(ddof=1))
    df = n - 1
    if sd == 0.0:
        if mean > 0.0:
            p_value = 0.0
            status = "zero_variance_positive"
        elif mean < 0.0:
            p_value = 1.0
            status = "zero_variance_negative"
        else:
            p_value = 1.0
            status = "zero_variance_zero"
        return {
            "n": n,
            "mean": mean,
            "sd": sd,
            "se": 0.0,
            "ci_low": mean,
            "ci_high": mean,
            "t": math.nan,
            "df": df,
            "p": p_value,
            "cohen_dz": math.nan,
            "status": status,
        }

    se = sd / math.sqrt(n)
    t_statistic = mean / se
    critical = float(stats.t.ppf(0.975, df))
    return {
        "n": n,
        "mean": mean,
        "sd": sd,
        "se": se,
        "ci_low": mean - critical * se,
        "ci_high": mean + critical * se,
        "t": t_statistic,
        "df": df,
        "p": float(stats.t.sf(t_statistic, df)),
        "cohen_dz": mean / sd,
        "status": "complete",
    }


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    adjusted = np.full(values.shape, np.nan, dtype=float)
    finite_positions = np.flatnonzero(np.isfinite(values))
    if finite_positions.size == 0:
        return adjusted
    finite_values = values[finite_positions]
    if ((finite_values < 0.0) | (finite_values > 1.0)).any():
        raise ValueError("BH p-values must lie in [0, 1]")
    order = np.argsort(finite_values, kind="stable")
    ranked = finite_values[order]
    scale = len(ranked) / np.arange(1, len(ranked) + 1, dtype=float)
    ranked_adjusted = np.minimum.accumulate((ranked * scale)[::-1])[::-1]
    ranked_adjusted = np.clip(ranked_adjusted, 0.0, 1.0)
    inverse = np.empty_like(order)
    inverse[order] = np.arange(len(order))
    adjusted[finite_positions] = ranked_adjusted[inverse]
    return adjusted


def analyze_metric_wide(
    wide: pd.DataFrame,
    *,
    analysis_name: str,
    model_family: str,
    category_system: str,
    category_label: str,
    metric: str,
) -> list[dict[str, object]]:
    expected_columns = list(LAYERS)
    if list(wide.columns) != expected_columns:
        raise ValueError("Wide trajectory must contain ordered layers 0-47")
    if len(wide) != EXPECTED_PROTEINS:
        raise ValueError(f"Wide trajectory must contain {EXPECTED_PROTEINS} proteins")
    matrix = wide.to_numpy(dtype=float)
    if not np.isfinite(matrix).all():
        raise ValueError("Wide trajectory contains nonfinite outcomes")
    group_means = matrix.mean(axis=0)
    spec = METRIC_SPECS[metric]
    records: list[dict[str, object]] = []

    for layer in ELIGIBLE_LAYERS:
        left_layers = tuple(range(layer - WINDOW, layer))
        right_layers = tuple(range(layer + 1, layer + WINDOW + 1))
        center_values = matrix[:, layer]
        left_values = matrix[:, left_layers].mean(axis=1)
        right_values = matrix[:, right_layers].mean(axis=1)
        center_mean = float(group_means[layer])
        left_mean = float(left_values.mean())
        right_mean = float(right_values.mean())
        neighbor_means = group_means[list(left_layers + right_layers)]
        left_slope = ols_slope(group_means[layer - WINDOW : layer + 1])
        right_slope = ols_slope(group_means[layer : layer + WINDOW + 1])

        for direction in DIRECTIONS:
            sign = 1.0 if direction == "peak" else -1.0
            strict_window = bool(
                center_mean > float(neighbor_means.max())
                if direction == "peak"
                else center_mean < float(neighbor_means.min())
            )
            slope_shape = bool(
                (left_slope > 0.0 and right_slope < 0.0)
                if direction == "peak"
                else (left_slope < 0.0 and right_slope > 0.0)
            )
            left_test = one_sample_greater(sign * (center_values - left_values))
            right_test = one_sample_greater(sign * (center_values - right_values))
            component_p_values = [float(left_test["p"]), float(right_test["p"])]
            combined_p = (
                max(component_p_values)
                if np.isfinite(component_p_values).all()
                else math.nan
            )
            dz_values = [float(left_test["cohen_dz"]), float(right_test["cohen_dz"])]
            standardized_prominence = (
                min(dz_values) if np.isfinite(dz_values).all() else math.nan
            )
            test_status = (
                "complete"
                if left_test["status"] == right_test["status"] == "complete"
                else f"left={left_test['status']};right={right_test['status']}"
            )
            display_scale = float(spec["display_scale"])
            records.append(
                {
                    "analysis_name": analysis_name,
                    "model_family": model_family,
                    "model": MODEL_SPECS[analysis_name][0],
                    "category_system": category_system,
                    "category_label": category_label,
                    "metric": metric,
                    "metric_role": spec["role"],
                    "direction": direction,
                    "layer": layer,
                    "window_layers_per_side": WINDOW,
                    "n_proteins": int(left_test["n"]),
                    "center_mean": center_mean,
                    "left_neighborhood_mean": left_mean,
                    "right_neighborhood_mean": right_mean,
                    "left_slope": left_slope,
                    "right_slope": right_slope,
                    "strict_window_extremum": strict_window,
                    "slope_shape_pass": slope_shape,
                    "shape_qualified_candidate": strict_window and slope_shape,
                    "left_contrast_mean": left_test["mean"],
                    "left_contrast_sd": left_test["sd"],
                    "left_contrast_se": left_test["se"],
                    "left_contrast_ci95_low": left_test["ci_low"],
                    "left_contrast_ci95_high": left_test["ci_high"],
                    "left_t_statistic": left_test["t"],
                    "left_df": left_test["df"],
                    "left_p_value": left_test["p"],
                    "left_cohen_dz": left_test["cohen_dz"],
                    "left_test_status": left_test["status"],
                    "right_contrast_mean": right_test["mean"],
                    "right_contrast_sd": right_test["sd"],
                    "right_contrast_se": right_test["se"],
                    "right_contrast_ci95_low": right_test["ci_low"],
                    "right_contrast_ci95_high": right_test["ci_high"],
                    "right_t_statistic": right_test["t"],
                    "right_df": right_test["df"],
                    "right_p_value": right_test["p"],
                    "right_cohen_dz": right_test["cohen_dz"],
                    "right_test_status": right_test["status"],
                    "combined_p_value": combined_p,
                    "raw_prominence": min(
                        float(left_test["mean"]), float(right_test["mean"])
                    ),
                    "standardized_prominence": standardized_prominence,
                    "display_scale": display_scale,
                    "display_unit": spec["display_unit"],
                    "center_mean_x100": center_mean * display_scale,
                    "left_neighborhood_mean_x100": left_mean * display_scale,
                    "right_neighborhood_mean_x100": right_mean * display_scale,
                    "left_contrast_mean_x100": float(left_test["mean"]) * display_scale,
                    "right_contrast_mean_x100": float(right_test["mean"]) * display_scale,
                    "raw_prominence_x100": min(
                        float(left_test["mean"]), float(right_test["mean"])
                    )
                    * display_scale,
                    "test_status": test_status,
                }
            )
    return records


def analyze_all_trajectories(frame: pd.DataFrame) -> pd.DataFrame:
    records: list[dict[str, object]] = []
    group_columns = [
        "analysis_name",
        "model_family",
        "category_system",
        "category_label",
    ]
    for key, group in frame.groupby(group_columns, sort=False):
        analysis_name, model_family, category_system, category_label = key
        for metric in METRIC_ORDER:
            wide = group.pivot(index="protein_id", columns="layer", values=metric)
            wide = wide.sort_index().reindex(columns=LAYERS)
            records.extend(
                analyze_metric_wide(
                    wide,
                    analysis_name=str(analysis_name),
                    model_family=str(model_family),
                    category_system=str(category_system),
                    category_label=str(category_label),
                    metric=metric,
                )
            )
    tests = pd.DataFrame(records)
    if len(tests) != EXPECTED_TEST_ROWS:
        raise ValueError(f"Expected {EXPECTED_TEST_ROWS} test rows; found {len(tests)}")

    tests["bh_q_value"] = math.nan
    tests["bh_family_size"] = 0
    for metric in METRIC_ORDER:
        mask = tests["metric"].eq(metric)
        family_size = int(mask.sum())
        if family_size != EXPECTED_BH_FAMILY_SIZE:
            raise ValueError(
                f"Expected {EXPECTED_BH_FAMILY_SIZE} BH rows for {metric}; found {family_size}"
            )
        tests.loc[mask, "bh_q_value"] = benjamini_hochberg(
            tests.loc[mask, "combined_p_value"].to_numpy(dtype=float)
        )
        tests.loc[mask, "bh_family_size"] = family_size

    tests["fdr_pass"] = tests["bh_q_value"].le(ALPHA)
    tests["supported_extremum"] = (
        tests["shape_qualified_candidate"] & tests["fdr_pass"]
    )
    tests["result_status"] = np.select(
        [
            tests["supported_extremum"],
            tests["shape_qualified_candidate"],
        ],
        ["supported", "candidate_not_fdr_supported"],
        default="not_candidate",
    )
    return sort_tests(tests)


def sort_tests(frame: pd.DataFrame) -> pd.DataFrame:
    result = frame.copy()
    model_order = {name: index for index, name in enumerate(MODEL_ORDER)}
    system_order = {name: index for index, name in enumerate(CATEGORY_SPECS)}
    metric_order = {name: index for index, name in enumerate(METRIC_ORDER)}
    direction_order = {name: index for index, name in enumerate(DIRECTIONS)}
    result["_model_order"] = result["analysis_name"].map(model_order)
    result["_system_order"] = result["category_system"].map(system_order)
    result["_metric_order"] = result["metric"].map(metric_order)
    result["_direction_order"] = result["direction"].map(direction_order)
    result = result.sort_values(
        [
            "_metric_order",
            "_model_order",
            "_system_order",
            "category_label",
            "_direction_order",
            "layer",
        ],
        kind="stable",
    ).drop(columns=["_model_order", "_system_order", "_metric_order", "_direction_order"])
    return result.reset_index(drop=True)


def summarize_trajectories(tests: pd.DataFrame) -> pd.DataFrame:
    keys = [
        "analysis_name",
        "model_family",
        "model",
        "category_system",
        "category_label",
        "metric",
        "metric_role",
    ]
    records: list[dict[str, object]] = []
    for key, group in tests.groupby(keys, sort=False):
        record = dict(zip(keys, key))
        for direction in DIRECTIONS:
            directional = group[group["direction"].eq(direction)]
            candidates = directional[directional["shape_qualified_candidate"]]
            supported = directional[directional["supported_extremum"]]
            prefix = direction
            record[f"n_candidate_{prefix}s"] = len(candidates)
            record[f"candidate_{prefix}_layers"] = ";".join(
                str(value) for value in candidates["layer"].astype(int)
            )
            record[f"n_supported_{prefix}s"] = len(supported)
            record[f"supported_{prefix}_layers"] = ";".join(
                str(value) for value in supported["layer"].astype(int)
            )
            if supported.empty:
                record[f"largest_supported_{prefix}_prominence"] = math.nan
                record[f"largest_supported_{prefix}_layer"] = pd.NA
            else:
                maximum_index = supported["raw_prominence"].idxmax()
                maximum = supported.loc[maximum_index]
                record[f"largest_supported_{prefix}_prominence"] = maximum[
                    "raw_prominence"
                ]
                record[f"largest_supported_{prefix}_layer"] = int(maximum["layer"])
        records.append(record)
    return pd.DataFrame(records)


def event_payload(row: pd.Series | None, suffix: str) -> dict[str, object]:
    if row is None:
        return {
            f"layer_{suffix}": pd.NA,
            f"raw_prominence_{suffix}": math.nan,
            f"standardized_prominence_{suffix}": math.nan,
            f"bh_q_value_{suffix}": math.nan,
        }
    return {
        f"layer_{suffix}": int(row["layer"]),
        f"raw_prominence_{suffix}": float(row["raw_prominence"]),
        f"standardized_prominence_{suffix}": float(row["standardized_prominence"]),
        f"bh_q_value_{suffix}": float(row["bh_q_value"]),
    }


def match_event_rows(
    events_a: pd.DataFrame,
    events_b: pd.DataFrame,
    base: dict[str, object],
) -> tuple[list[dict[str, object]], dict[str, object]]:
    a_rows = [row for _, row in events_a.sort_values("layer").iterrows()]
    b_rows = [row for _, row in events_b.sort_values("layer").iterrows()]
    possibilities = sorted(
        (
            abs(int(a["layer"]) - int(b["layer"])),
            int(a["layer"]),
            int(b["layer"]),
            a_index,
            b_index,
        )
        for a_index, a in enumerate(a_rows)
        for b_index, b in enumerate(b_rows)
        if abs(int(a["layer"]) - int(b["layer"])) <= ALIGNMENT_TOLERANCE
    )
    used_a: set[int] = set()
    used_b: set[int] = set()
    matches: list[tuple[int, int]] = []
    for _, _, _, a_index, b_index in possibilities:
        if a_index not in used_a and b_index not in used_b:
            matches.append((a_index, b_index))
            used_a.add(a_index)
            used_b.add(b_index)

    records: list[dict[str, object]] = []
    for a_index, b_index in matches:
        a = a_rows[a_index]
        b = b_rows[b_index]
        difference = int(b["layer"]) - int(a["layer"])
        records.append(
            {
                **base,
                **event_payload(a, "a"),
                **event_payload(b, "b"),
                "layer_difference_b_minus_a": difference,
                "absolute_layer_difference": abs(difference),
                "exact_match": difference == 0,
                "within_2_match": True,
                "match_status": "exact_match" if difference == 0 else "within_2_match",
            }
        )
    for a_index, a in enumerate(a_rows):
        if a_index not in used_a:
            records.append(
                {
                    **base,
                    **event_payload(a, "a"),
                    **event_payload(None, "b"),
                    "layer_difference_b_minus_a": math.nan,
                    "absolute_layer_difference": math.nan,
                    "exact_match": False,
                    "within_2_match": False,
                    "match_status": "only_a",
                }
            )
    for b_index, b in enumerate(b_rows):
        if b_index not in used_b:
            records.append(
                {
                    **base,
                    **event_payload(None, "a"),
                    **event_payload(b, "b"),
                    "layer_difference_b_minus_a": math.nan,
                    "absolute_layer_difference": math.nan,
                    "exact_match": False,
                    "within_2_match": False,
                    "match_status": "only_b",
                }
            )
    if not records:
        records.append(
            {
                **base,
                **event_payload(None, "a"),
                **event_payload(None, "b"),
                "layer_difference_b_minus_a": math.nan,
                "absolute_layer_difference": math.nan,
                "exact_match": False,
                "within_2_match": False,
                "match_status": "neither_has_supported_extremum",
            }
        )

    summary = {
        **base,
        "n_supported_extrema_a": len(a_rows),
        "n_supported_extrema_b": len(b_rows),
        "n_exact_matches": sum(
            int(record["match_status"] == "exact_match") for record in records
        ),
        "n_within_2_matches_including_exact": len(matches),
        "n_within_2_nonexact_matches": sum(
            int(record["match_status"] == "within_2_match") for record in records
        ),
        "n_unmatched_a": len(a_rows) - len(matches),
        "n_unmatched_b": len(b_rows) - len(matches),
        "any_exact_match": any(record["match_status"] == "exact_match" for record in records),
        "any_within_2_match": bool(matches),
    }
    return records, summary


def build_alignment_tables(tests: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    supported = tests[tests["supported_extremum"]].copy()
    comparison_records: list[dict[str, object]] = []
    summary_records: list[dict[str, object]] = []
    comparison_index = 0

    for metric in METRIC_ORDER:
        for direction in DIRECTIONS:
            selected = supported[
                supported["metric"].eq(metric) & supported["direction"].eq(direction)
            ]
            for system, spec in CATEGORY_SPECS.items():
                for label in spec["labels"]:
                    for model_a, model_b in itertools.combinations(MODEL_ORDER, 2):
                        comparison_index += 1
                        base = {
                            "comparison_id": f"comparison_{comparison_index:05d}",
                            "comparison_scope": "across_models_same_category",
                            "metric": metric,
                            "metric_role": METRIC_SPECS[metric]["role"],
                            "direction": direction,
                            "category_system": system,
                            "model_a": model_a,
                            "model_b": model_b,
                            "category_label_a": label,
                            "category_label_b": label,
                        }
                        common = selected[
                            selected["category_system"].eq(system)
                            & selected["category_label"].eq(label)
                        ]
                        rows, summary = match_event_rows(
                            common[common["analysis_name"].eq(model_a)],
                            common[common["analysis_name"].eq(model_b)],
                            base,
                        )
                        comparison_records.extend(rows)
                        summary_records.append(summary)

                labels = tuple(spec["labels"])
                if len(labels) < 2:
                    continue
                for model in MODEL_ORDER:
                    for label_a, label_b in itertools.combinations(labels, 2):
                        comparison_index += 1
                        base = {
                            "comparison_id": f"comparison_{comparison_index:05d}",
                            "comparison_scope": "across_categories_same_model",
                            "metric": metric,
                            "metric_role": METRIC_SPECS[metric]["role"],
                            "direction": direction,
                            "category_system": system,
                            "model_a": model,
                            "model_b": model,
                            "category_label_a": label_a,
                            "category_label_b": label_b,
                        }
                        common = selected[
                            selected["analysis_name"].eq(model)
                            & selected["category_system"].eq(system)
                        ]
                        rows, summary = match_event_rows(
                            common[common["category_label"].eq(label_a)],
                            common[common["category_label"].eq(label_b)],
                            base,
                        )
                        comparison_records.extend(rows)
                        summary_records.append(summary)

    return pd.DataFrame(comparison_records), pd.DataFrame(summary_records)


def category_colors(labels: Sequence[str]) -> dict[str, object]:
    palette = plt.get_cmap("tab10")
    return {label: palette(index) for index, label in enumerate(labels)}


def save_figure(
    figure: plt.Figure,
    output_root: Path,
    stem: str,
    metadata: dict[str, object],
) -> list[dict[str, object]]:
    destination = output_root / "annotated_trajectories"
    destination.mkdir(parents=True, exist_ok=True)
    records: list[dict[str, object]] = []
    for extension in ("png", "pdf"):
        path = destination / f"{stem}.{extension}"
        if extension == "pdf":
            figure.savefig(
                path,
                metadata={
                    "Creator": "analyze_triangle_metric_peak_troughs.py",
                    "CreationDate": None,
                    "ModDate": None,
                },
            )
        else:
            figure.savefig(path)
        records.append(
            {
                **metadata,
                "format": extension,
                "path": path.relative_to(output_root).as_posix(),
                "sha256": sha256_file(path),
            }
        )
    plt.close(figure)
    return records


def plot_category_system(
    source: pd.DataFrame,
    candidates: pd.DataFrame,
    system: str,
    output_root: Path,
) -> list[dict[str, object]]:
    spec = CATEGORY_SPECS[system]
    labels = tuple(spec["labels"])
    colors = category_colors(labels)
    figure, axes = plt.subplots(
        len(METRIC_ORDER),
        len(MODEL_ORDER),
        figsize=(4.0 * len(MODEL_ORDER), 3.7 * len(METRIC_ORDER)),
        sharex=True,
        sharey="row",
        squeeze=False,
    )
    selected_source = source[source["category_system"].eq(system)]
    selected_candidates = candidates[candidates["category_system"].eq(system)]
    for row_index, metric in enumerate(METRIC_ORDER):
        metric_spec = METRIC_SPECS[metric]
        for column_index, model in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            model_source = selected_source[selected_source["analysis_name"].eq(model)]
            model_candidates = selected_candidates[
                selected_candidates["analysis_name"].eq(model)
                & selected_candidates["metric"].eq(metric)
            ]
            for label in labels:
                points = (
                    model_source[model_source["category_label"].eq(label)]
                    .groupby("layer", as_index=False)[metric]
                    .mean()
                    .sort_values("layer")
                )
                axis.plot(
                    points["layer"],
                    points[metric],
                    color=colors[label],
                    linewidth=1.8,
                )
                label_candidates = model_candidates[
                    model_candidates["category_label"].eq(label)
                ]
                for direction, marker in (("peak", "^"), ("trough", "v")):
                    directional = label_candidates[
                        label_candidates["direction"].eq(direction)
                    ]
                    unsupported = directional[~directional["supported_extremum"]]
                    supported = directional[directional["supported_extremum"]]
                    axis.scatter(
                        unsupported["layer"],
                        unsupported["center_mean"],
                        marker=marker,
                        s=35,
                        facecolors="none",
                        edgecolors=colors[label],
                        linewidths=1.1,
                        zorder=4,
                    )
                    axis.scatter(
                        supported["layer"],
                        supported["center_mean"],
                        marker=marker,
                        s=45,
                        facecolors=colors[label],
                        edgecolors="black",
                        linewidths=0.55,
                        zorder=5,
                    )
            axis.set(xlim=(0, 47), xticks=(0, 10, 20, 30, 40, 47))
            axis.set_ylim(bottom=0)
            if row_index == 0:
                axis.set_title(MODEL_SPECS[model][0])
            if row_index == len(METRIC_ORDER) - 1:
                axis.set_xlabel("Evoformer layer")
            if column_index == 0:
                axis.set_ylabel(f"{metric_spec['title']}\n({metric_spec['ylabel']})")

    category_handles = [
        Line2D([0], [0], color=colors[label], linewidth=2.2, label=display_category(label))
        for label in labels
    ]
    marker_handles = [
        Line2D([0], [0], marker="^", color="black", linestyle="none", markerfacecolor="none", label="Candidate peak"),
        Line2D([0], [0], marker="v", color="black", linestyle="none", markerfacecolor="none", label="Candidate trough"),
        Line2D([0], [0], marker="o", color="black", linestyle="none", markerfacecolor="black", label="BH-supported"),
    ]
    figure.legend(
        handles=category_handles + marker_handles,
        loc="outside lower center",
        ncol=len(category_handles) + len(marker_handles),
        frameon=False,
    )
    figure.suptitle(f"{spec['title']}: search-adjusted local extrema", y=1.01)
    figure.tight_layout(rect=(0, 0.07, 1, 0.99))
    stem = f"figure_{spec['figure_number']:02d}_{system}_peak_troughs"
    return save_figure(
        figure,
        output_root,
        stem,
        {
            "figure_number": spec["figure_number"],
            "category_system": system,
            "figure_title": spec["title"],
            "metrics": ";".join(METRIC_ORDER),
        },
    )


def markdown_table(headers: Sequence[str], rows: Iterable[Sequence[object]]) -> list[str]:
    output = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join("---" for _ in headers) + "|",
    ]
    output.extend("| " + " | ".join(str(value) for value in row) + " |" for row in rows)
    return output


def format_layers(values: pd.Series) -> str:
    return ", ".join(str(value) for value in values.astype(int)) or "None"


def write_readme(
    output_root: Path,
    tests: pd.DataFrame,
    candidates: pd.DataFrame,
    alignment_summary: pd.DataFrame,
) -> None:
    supported = candidates[candidates["supported_extremum"]]
    count_rows = []
    for metric in METRIC_ORDER:
        for direction in DIRECTIONS:
            selected = candidates[
                candidates["metric"].eq(metric) & candidates["direction"].eq(direction)
            ]
            count_rows.append(
                (
                    METRIC_SPECS[metric]["role"].title(),
                    metric,
                    direction.title(),
                    len(selected),
                    int(selected["supported_extremum"].sum()),
                )
            )

    recurring_rows: list[tuple[object, ...]] = []
    frequencies = (
        supported.groupby(["metric", "direction", "layer"], as_index=False)
        .size()
        .sort_values(["metric", "direction", "size", "layer"], ascending=[True, True, False, True])
    )
    for (metric, direction), group in frequencies.groupby(["metric", "direction"], sort=False):
        for _, row in group.head(8).iterrows():
            recurring_rows.append((metric, direction.title(), int(row["layer"]), int(row["size"])))

    model_rows = []
    for model in MODEL_ORDER:
        for metric in METRIC_ORDER:
            selected = supported[
                supported["analysis_name"].eq(model) & supported["metric"].eq(metric)
            ]
            model_rows.append(
                (
                    MODEL_SPECS[model][0],
                    metric,
                    int(selected["direction"].eq("peak").sum()),
                    int(selected["direction"].eq("trough").sum()),
                )
            )

    all_triangle_rows = []
    for model in MODEL_ORDER:
        for metric in METRIC_ORDER:
            selected = supported[
                supported["analysis_name"].eq(model)
                & supported["category_system"].eq("all_triangles")
                & supported["metric"].eq(metric)
            ]
            all_triangle_rows.append(
                (
                    MODEL_SPECS[model][0],
                    metric,
                    format_layers(selected.loc[selected["direction"].eq("peak"), "layer"]),
                    format_layers(selected.loc[selected["direction"].eq("trough"), "layer"]),
                )
            )

    effect_rows: list[tuple[object, ...]] = []
    for metric in METRIC_ORDER:
        for direction in DIRECTIONS:
            selected = supported[
                supported["metric"].eq(metric) & supported["direction"].eq(direction)
            ].nlargest(5, "raw_prominence")
            for _, row in selected.iterrows():
                effect_rows.append(
                    (
                        metric,
                        direction.title(),
                        row["model"],
                        f"{row['category_system']}:{display_category(row['category_label'])}",
                        int(row["layer"]),
                        f"{row['raw_prominence_x100']:.3f}",
                        f"{row['standardized_prominence']:.3f}",
                        f"{row['bh_q_value']:.2e}",
                    )
                )

    alignment_aggregate = (
        alignment_summary.groupby(
            ["comparison_scope", "metric", "direction"], as_index=False
        )
        .agg(
            n_pairs=("comparison_id", "size"),
            pairs_any_exact=("any_exact_match", "sum"),
            pairs_any_within2=("any_within_2_match", "sum"),
            exact_matches=("n_exact_matches", "sum"),
            within2_matches=("n_within_2_matches_including_exact", "sum"),
        )
    )
    alignment_rows = []
    for _, row in alignment_aggregate.iterrows():
        scope_label = (
            "Across models"
            if row["comparison_scope"] == "across_models_same_category"
            else "Across categories"
        )
        alignment_rows.append(
            (
                scope_label,
                row["metric"],
                str(row["direction"]).title(),
                int(row["n_pairs"]),
                int(row["pairs_any_exact"]),
                int(row["pairs_any_within2"]),
                int(row["exact_matches"]),
                int(row["within2_matches"]),
            )
        )

    primary_peak_alignment = alignment_aggregate[
        alignment_aggregate["metric"].eq("mean_relative_violation_all")
        & alignment_aggregate["direction"].eq("peak")
    ].set_index("comparison_scope")
    category_primary = primary_peak_alignment.loc["across_categories_same_model"]
    model_primary = primary_peak_alignment.loc["across_models_same_category"]

    lines = [
        "# Search-adjusted triangle-metric peaks and troughs",
        "",
        "## Scope and result definition",
        "",
        (
            "This analysis tests local extrema in the five focused triangle-consistency "
            "figure families using paired measurements from the same 111 proteins. It uses "
            "only `violation_fraction` and `mean_relative_violation_all`; R2 is not read."
        ),
        "",
        (
            "Every layer 5--42 was tested as a possible peak and trough before candidate "
            "filtering. A reported supported extremum is the strict maximum/minimum in its "
            "+/-5 neighborhood, has the required fitted slope directions, and passes an "
            "outcome-specific Benjamini--Hochberg threshold of q <= 0.05."
        ),
        "",
        "## Candidate and supported counts",
        "",
        *markdown_table(
            ("Role", "Metric", "Direction", "Shape-qualified candidates", "BH-supported"),
            count_rows,
        ),
        "",
        "## Supported extrema by model",
        "",
        *markdown_table(("Model", "Metric", "Peaks", "Troughs"), model_rows),
        "",
        "## All-triangle supported layers",
        "",
        (
            "This compact view answers the original unstratified question. `None` means that "
            "no layer met all shape and search-adjusted significance requirements."
        ),
        "",
        *markdown_table(("Model", "Metric", "Peak layers", "Trough layers"), all_triangle_rows),
        "",
        "## Most recurrent exact layers",
        "",
        (
            "Counts below are numbers of the 52 trajectories with a supported extremum at "
            "that exact layer. Nearby-event comparisons within two layers are retained in "
            "the alignment tables."
        ),
        "",
        *markdown_table(("Metric", "Direction", "Layer", "Trajectory count"), recurring_rows),
        "",
        "## Largest supported raw effects",
        "",
        (
            "The table shows the five largest conservative raw prominences for each metric "
            "and direction. The x100 magnitude is in percentage points for violation fraction "
            "and normalized-burden points for mean relative violation. Raw magnitudes are most "
            "interpretable within a metric; paired Cohen's dz aids comparison across differently "
            "scaled categories."
        ),
        "",
        *markdown_table(
            (
                "Metric",
                "Direction",
                "Model",
                "Category",
                "Layer",
                "Prominence x100",
                "Cohen dz",
                "BH q",
            ),
            effect_rows,
        ),
        "",
        "## Interpretation of magnitude",
        "",
        (
            "Each row reports separate center-versus-left and center-versus-right paired "
            "mean contrasts. `raw_prominence` is the smaller of those two contrasts, so it "
            "is a conservative descriptive magnitude. `standardized_prominence` is the "
            "smaller paired Cohen's dz. The side-specific 95% intervals are ordinary "
            "unadjusted t intervals; BH-adjusted q-values, rather than those intervals, "
            "determine search-adjusted support."
        ),
        "",
        (
            "Violation-fraction x100 columns are percentage-point effects. Relative-burden "
            "x100 columns express the average normalized triangle-violation burden on a "
            "100-point scale; they are not percent changes from baseline."
        ),
        "",
        "## Cross-model and cross-category agreement",
        "",
        (
            f"For the primary metric's peaks, {int(category_primary['pairs_any_within2'])} of "
            f"{int(category_primary['n_pairs'])} within-model category pairs and "
            f"{int(model_primary['pairs_any_within2'])} of {int(model_primary['n_pairs'])} "
            "same-category model pairs shared at least one supported event within two layers. "
            "This indicates substantial category alignment but only partial model alignment. "
            "Unmatched events are descriptive differences and are not formal tests that peak "
            "locations differ."
        ),
        "",
        *markdown_table(
            (
                "Comparison",
                "Metric",
                "Direction",
                "Trajectory pairs",
                "Pairs with exact match",
                "Pairs with +/-2 match",
                "Exact event matches",
                "+/-2 event matches",
            ),
            alignment_rows,
        ),
        "",
        (
            "`alignment_comparisons.csv` records exact, within-two-layer, and unmatched "
            "supported events. `alignment_summary.csv` has one row per trajectory pair. "
            "Exact selected layers are always retained; the tolerance is only a secondary "
            "broad-stage label."
        ),
        "",
        "## Files",
        "",
        "- `all_layer_tests.csv`: all 7,904 predefined layer/direction hypotheses.",
        "- `candidate_extrema.csv`: strict-window candidates with the required slope shape.",
        "- `trajectory_summary.csv`: candidate and supported layers per trajectory.",
        "- `alignment_comparisons.csv`: event-level model/category comparisons.",
        "- `alignment_summary.csv`: pair-level agreement counts.",
        "- `annotated_trajectories/`: hollow candidate markers and filled supported markers.",
        "",
        "## Caveats",
        "",
        (
            "These are search-adjusted local-extremum findings, not independent replication. "
            "BH targets the expected false-discovery proportion across the predefined search; "
            "it does not assign a probability that an individual extremum is true. Selection "
            "can still make reported effect magnitudes optimistic. Independent proteins would "
            "be needed for fully external confirmation."
        ),
        "",
        (
            "The t-tests treat proteins, not triangles, as independent units. The analysis "
            "does not bootstrap, test ten neighbors individually, or make formal claims that "
            "extremum locations differ between models or categories. The per-protein linear "
            "probe retains its mixed fitted/held-out edge interpretation from the source workflow."
        ),
        "",
        f"All component tests used n={EXPECTED_PROTEINS}; all BH families contained {EXPECTED_BH_FAMILY_SIZE} hypotheses.",
        "",
    ]
    (output_root / "README.md").write_text("\n".join(lines))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-path", type=Path, default=DEFAULT_INPUT_PATH)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    input_path = args.input_path.resolve()
    output_root = args.output_root.resolve()
    output_root.mkdir(parents=True, exist_ok=True)
    configure_style()

    source = load_and_validate_input(input_path)
    tests = analyze_all_trajectories(source)
    candidates = tests[tests["shape_qualified_candidate"]].copy()
    trajectory_summary = summarize_trajectories(tests)
    alignment_comparisons, alignment_summary = build_alignment_tables(tests)

    tests.to_csv(output_root / "all_layer_tests.csv", index=False)
    candidates.to_csv(output_root / "candidate_extrema.csv", index=False)
    trajectory_summary.to_csv(output_root / "trajectory_summary.csv", index=False)
    alignment_comparisons.to_csv(output_root / "alignment_comparisons.csv", index=False)
    alignment_summary.to_csv(output_root / "alignment_summary.csv", index=False)

    figure_records: list[dict[str, object]] = []
    for system in CATEGORY_SPECS:
        figure_records.extend(plot_category_system(source, candidates, system, output_root))
    figure_manifest = pd.DataFrame(figure_records)
    figure_manifest.to_csv(output_root / "figure_manifest.csv", index=False)
    write_readme(output_root, tests, candidates, alignment_summary)

    generated_paths = [
        output_root / "README.md",
        output_root / "all_layer_tests.csv",
        output_root / "candidate_extrema.csv",
        output_root / "trajectory_summary.csv",
        output_root / "alignment_comparisons.csv",
        output_root / "alignment_summary.csv",
        output_root / "figure_manifest.csv",
        *[output_root / record["path"] for record in figure_records],
    ]
    generated_at = datetime.now(timezone.utc).isoformat()
    supported = candidates[candidates["supported_extremum"]]
    generation_summary = {
        "analysis_name": "triangle_metric_peak_trough_significance",
        "generated_at": generated_at,
        "input_path": relative(input_path),
        "output_root": relative(output_root),
        "evaluation_scope": "all_edges",
        "aggregation_unit": "protein",
        "n_input_rows": len(source),
        "n_proteins": source["protein_id"].nunique(),
        "models": list(MODEL_ORDER),
        "category_systems": list(CATEGORY_SPECS),
        "n_trajectories": EXPECTED_TRAJECTORIES,
        "metrics": list(METRIC_ORDER),
        "primary_metric": "mean_relative_violation_all",
        "secondary_metric": "violation_fraction",
        "eligible_layers": list(ELIGIBLE_LAYERS),
        "window_layers_per_side": WINDOW,
        "directions": list(DIRECTIONS),
        "n_all_layer_tests": len(tests),
        "bh_method": "Benjamini-Hochberg",
        "bh_families": "separate_by_metric_across_all_models_categories_directions_layers",
        "bh_family_size": EXPECTED_BH_FAMILY_SIZE,
        "alpha": ALPHA,
        "alignment_tolerance_layers": ALIGNMENT_TOLERANCE,
        "n_shape_qualified_candidates": len(candidates),
        "n_supported_extrema": len(supported),
        "candidate_counts": {
            metric: {
                direction: int(
                    (
                        candidates["metric"].eq(metric)
                        & candidates["direction"].eq(direction)
                    ).sum()
                )
                for direction in DIRECTIONS
            }
            for metric in METRIC_ORDER
        },
        "supported_counts": {
            metric: {
                direction: int(
                    (
                        supported["metric"].eq(metric)
                        & supported["direction"].eq(direction)
                    ).sum()
                )
                for direction in DIRECTIONS
            }
            for metric in METRIC_ORDER
        },
        "distinct_figures": len(figure_records) // 2,
        "rendered_figure_files": len(figure_records),
        "excluded_methods": [
            "R2",
            "bootstrap_resampling",
            "individual_neighbor_t_tests",
        ],
        "source_sha256": {relative(input_path): sha256_file(input_path)},
        "artifact_sha256": {
            path.relative_to(output_root).as_posix(): sha256_file(path)
            for path in generated_paths
        },
    }
    with (output_root / "generation_summary.json").open("w") as handle:
        json.dump(generation_summary, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")

    print(
        f"Wrote {len(tests)} tests, {len(candidates)} candidates, "
        f"{len(supported)} supported extrema, and {len(figure_records) // 2} figures "
        f"beneath {output_root}"
    )


if __name__ == "__main__":
    main()

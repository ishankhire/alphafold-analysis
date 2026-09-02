#!/usr/bin/env python3
"""Generate the September 2 publication bundle from fivefold outputs only."""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path
from typing import Collection, Iterable, Mapping, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


GENERATOR_VERSION = "sept_2_publication_plots_v2"
MAIN_FIGURE_FONT_SCALE = 1.5
BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260902
EXPECTED_PROTEINS = 552
LAYERS = tuple(range(48))
ELIGIBLE_EXTREMA_LAYERS = tuple(range(5, 43))
EXTREMA_WINDOW = 5
ALPHA = 0.05
MILESTONE_LAYERS = (0, 10, 20, 30, 40, 47)

REGRESSION_ROOT = PROJECT_ROOT / "delta_files/regression_5fold_v1"
OUTPUT_ROOT = PROJECT_ROOT / "delta_files/publication_plots/sept_2_plots"
MAIN_DIR = OUTPUT_ROOT / "main_figures"
SUPPLEMENT_DIR = OUTPUT_ROOT / "supplemental_figures"
ANNOTATED_DIR = SUPPLEMENT_DIR / "triangle_annotated_trajectories"
INDIVIDUAL_DIR = SUPPLEMENT_DIR / "individual_protein_trajectories"
SOURCE_DIR = OUTPUT_ROOT / "source_data"

SUBGROUP_DIR = REGRESSION_ROOT / "analysis/publication_subgroup_mae_5fold_v1"
TRIANGLE_DIR = REGRESSION_ROOT / "analysis/triangle_violation_subgroups_5fold_v1"
OUTER_FOLD_PATH = REGRESSION_ROOT / "metadata/regression_outer_folds_v1.csv"

MODEL_SPECS = {
    "shared_ols_base256": {
        "label": "Shared linear",
        "color": "#4C78A8",
        "linestyle": "-",
        "generalization_scope": "unseen_protein_ids",
    },
    "per_protein_ols_base256": {
        "label": "Per-protein linear",
        "color": "#7F7F7F",
        "linestyle": (0, (5, 2)),
        "generalization_scope": "unseen_pairs_within_seen_protein",
    },
    "shared_xgboost_base256_fixed_s4": {
        "label": "XGBoost",
        "color": "#E45756",
        "linestyle": "-",
        "generalization_scope": "unseen_protein_ids",
    },
    "sequence_gated_ols_base256": {
        "label": "Sequence gated",
        "color": "#F58518",
        "linestyle": "-",
        "generalization_scope": "unseen_protein_ids",
    },
}
MODEL_ORDER = tuple(MODEL_SPECS)

SPATIAL_BINS = ("0-5", "5-10", "10-20", "20-40", "40+")
SEQUENCE_BINS = ("1", "2-4", "5-9", "10-19", "20-49", "50+")
EXPECTED_BIN_SUPPORT = {
    ("spatial_distance", "0-5"): 552,
    ("spatial_distance", "5-10"): 552,
    ("spatial_distance", "10-20"): 552,
    ("spatial_distance", "20-40"): 552,
    ("spatial_distance", "40+"): 479,
    ("sequence_separation", "1"): 552,
    ("sequence_separation", "2-4"): 552,
    ("sequence_separation", "5-9"): 552,
    ("sequence_separation", "10-19"): 552,
    ("sequence_separation", "20-49"): 552,
    ("sequence_separation", "50+"): 549,
}

TRIANGLE_CATEGORY_SPECS = {
    "all_triangles": {"title": "All triangles", "labels": ("all",)},
    "rms_size": {
        "title": "Reference RMS-size tertiles",
        "labels": ("small", "medium", "large"),
        "stem": "figure_02_rms_size_peak_troughs",
    },
    "area": {
        "title": "Reference area tertiles",
        "labels": ("low", "medium", "high"),
        "stem": "figure_03_area_peak_troughs",
    },
    "relative_slack": {
        "title": "Reference relative-slack tertiles",
        "labels": ("low", "medium", "high"),
        "stem": "figure_04_relative_slack_peak_troughs",
    },
    "angle_shape": {
        "title": "Reference largest-angle classes",
        "labels": ("clearly_acute", "near_right", "clearly_obtuse"),
        "stem": "figure_05_angle_shape_peak_troughs",
    },
}
TRIANGLE_METRICS = {
    "mean_relative_violation_all": {
        "title": "Normalized mean triangle violation",
        "ylabel": "Mean relative violation per triangle",
        "domain": "nonnegative",
    },
    "violation_fraction": {
        "title": "Triangle-inequality violation fraction",
        "ylabel": "Fraction of triangles",
        "domain": "unit",
    },
}


def configure_style(*, font_scale: float = 1.0) -> None:
    """Configure figure styling, with enlarged type for publication main figures."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10 * font_scale,
            "axes.titlesize": 12 * font_scale,
            "axes.labelsize": 11 * font_scale,
            "xtick.labelsize": 9 * font_scale,
            "ytick.labelsize": 9 * font_scale,
            "legend.fontsize": 8.5 * font_scale,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_directories() -> None:
    for path in (MAIN_DIR, SUPPLEMENT_DIR, ANNOTATED_DIR, INDIVIDUAL_DIR, SOURCE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def require_input_path(path: Path) -> Path:
    """Resolve and validate an analytical input under the fivefold root."""
    resolved = path.resolve()
    root = REGRESSION_ROOT.resolve()
    if not resolved.is_relative_to(root):
        raise ValueError(f"Analytical input is outside regression_5fold_v1: {resolved}")
    if not resolved.is_file():
        raise FileNotFoundError(f"Required fivefold input is missing: {resolved}")
    return resolved


def require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}")


def normalize_protein_ids(values: pd.Series) -> pd.Series:
    return values.astype("string").str.zfill(9)


def bin_slug(label: str) -> str:
    return label.replace("+", "_plus").replace("-", "_")


def display_category(label: str) -> str:
    return label.replace("__", " · ").replace("_", " ").title()


def clip_interval(
    low: np.ndarray | pd.Series,
    high: np.ndarray | pd.Series,
    domain: str,
) -> tuple[np.ndarray, np.ndarray]:
    """Clip intervals for display only; source endpoints remain untouched."""
    low_array = np.asarray(low, dtype=float).copy()
    high_array = np.asarray(high, dtype=float).copy()
    if domain == "unit":
        return np.clip(low_array, 0.0, 1.0), np.clip(high_array, 0.0, 1.0)
    if domain == "nonnegative":
        return np.maximum(low_array, 0.0), np.maximum(high_array, 0.0)
    if domain == "unbounded":
        return low_array, high_array
    raise ValueError(f"Unknown metric domain: {domain}")


def expected_figure_ids() -> tuple[str, ...]:
    spatial = tuple(
        f"main_figure_15_spatial_distance_mae_{bin_slug(label)}"
        for label in SPATIAL_BINS
    )
    sequence = tuple(
        f"supp_figure_sequence_separation_mae_{bin_slug(label)}"
        for label in SEQUENCE_BINS
    )
    annotated = tuple(
        str(spec["stem"])
        for system, spec in TRIANGLE_CATEGORY_SPECS.items()
        if system != "all_triangles"
    )
    return (
        "main_figure_01_mean_per_protein_r2",
        "main_figure_03_tm_score",
        "main_figure_15_spatial_distance_mae_combined",
        *spatial,
        "main_figure_all_triangles_peak_troughs_overlay",
        "main_figure_all_triangles_decodability_consistency",
        "supp_figure_sequence_separation_mae_combined",
        *sequence,
        "supp_figure_all_triangles_violation_fraction_peak_troughs_overlay",
        "supp_figure_all_triangles_violation_fraction_decodability_consistency",
        "supp_figure_top3_eigenvalue_fraction",
        "supp_figure_mds_stress",
        *annotated,
        "supp_figure_individual_proteins_r2_by_model",
        "supp_figure_individual_proteins_tm_score_by_model",
        "supp_figure_individual_proteins_normalized_triangle_violation_by_model",
        "supp_figure_individual_proteins_triangle_violation_fraction_by_model",
    )


def load_cohort() -> tuple[str, ...]:
    path = require_input_path(OUTER_FOLD_PATH)
    frame = pd.read_csv(path, dtype={"protein_id": "string"})
    require_columns(frame, {"protein_id", "outer_fold"}, relative(path))
    frame["protein_id"] = normalize_protein_ids(frame["protein_id"])
    if len(frame) != EXPECTED_PROTEINS or frame["protein_id"].nunique() != EXPECTED_PROTEINS:
        raise ValueError("Fivefold cohort does not contain exactly 552 unique proteins")
    if set(frame["outer_fold"].astype(int)) != set(range(5)):
        raise ValueError("Fivefold cohort does not contain outer folds 0-4")
    return tuple(sorted(frame["protein_id"]))


def validate_core_values(frame: pd.DataFrame, context: str, cohort: Sequence[str]) -> None:
    if len(frame) != len(MODEL_ORDER) * len(LAYERS) * EXPECTED_PROTEINS:
        raise ValueError(f"{context} has {len(frame)} rows; expected 105984")
    if set(frame["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError(f"{context} does not contain exactly the four whitelisted models")
    if set(frame["layer"].astype(int)) != set(LAYERS):
        raise ValueError(f"{context} does not contain layers 0-47")
    if set(frame["protein_id"]) != set(cohort):
        raise ValueError(f"{context} does not use the frozen 552-protein cohort")
    if frame.duplicated(["analysis_name", "layer", "protein_id"]).any():
        raise ValueError(f"{context} contains duplicate model/layer/protein keys")
    if not np.isfinite(pd.to_numeric(frame["value"], errors="coerce")).all():
        raise ValueError(f"{context} contains nonfinite plotted values")


def load_core_metric(
    cohort: Sequence[str], filename: str, metric: str, status_column: str
) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    columns = [
        "analysis_name",
        "model_family",
        "layer",
        "protein_id",
        "source",
        "cv_protocol_version",
        "evaluation_design",
        "outer_fold",
        "evaluation_scope",
        "oof_prediction_scope",
        status_column,
        metric,
    ]
    for analysis in MODEL_ORDER:
        path = require_input_path(REGRESSION_ROOT / analysis / filename)
        frame = pd.read_csv(path, usecols=columns, dtype={"protein_id": "string"})
        if set(frame["analysis_name"]) != {analysis}:
            raise ValueError(f"{relative(path)} has the wrong analysis name")
        if set(frame[status_column]) != {"complete"}:
            raise ValueError(f"{relative(path)} contains incomplete rows")
        frame["protein_id"] = normalize_protein_ids(frame["protein_id"])
        frame = frame.rename(columns={metric: "value"}).drop(columns=[status_column])
        frame["model"] = MODEL_SPECS[analysis]["label"]
        frame["generalization_scope"] = MODEL_SPECS[analysis]["generalization_scope"]
        frames.append(frame)
        paths.append(path)
    result = pd.concat(frames, ignore_index=True)
    validate_core_values(result, metric, cohort)
    return result, paths


def validate_subgroup_values(frame: pd.DataFrame, cohort: Sequence[str]) -> None:
    expected_rows = len(MODEL_ORDER) * len(LAYERS) * EXPECTED_PROTEINS * 11
    if len(frame) != expected_rows:
        raise ValueError(f"Per-protein subgroup MAE has {len(frame)} rows; expected {expected_rows}")
    if set(frame["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError("Per-protein subgroup MAE has the wrong model set")
    if set(frame["protein_id"]) != set(cohort):
        raise ValueError("Per-protein subgroup MAE has the wrong protein cohort")
    key = ["analysis_name", "layer", "protein_id", "category_scope", "category_label"]
    if frame.duplicated(key).any():
        raise ValueError("Per-protein subgroup MAE contains duplicate keys")
    expected_labels = {
        "spatial_distance": set(SPATIAL_BINS),
        "sequence_separation": set(SEQUENCE_BINS),
    }
    for scope, labels in expected_labels.items():
        selected = frame[frame["category_scope"].eq(scope)]
        if set(selected["category_label"]) != labels:
            raise ValueError(f"Per-protein subgroup MAE has wrong labels for {scope}")
    supported = frame["n_pairs"].gt(0)
    if not frame.loc[supported, "support_status"].eq("complete").all():
        raise ValueError("Supported subgroup rows are not complete")
    if not frame.loc[~supported, "support_status"].eq("empty_category").all():
        raise ValueError("Empty subgroup rows are not explicitly unsupported")
    if not np.isfinite(frame.loc[supported, "mae"]).all():
        raise ValueError("Supported subgroup rows contain nonfinite MAE")
    if frame.loc[~supported, "mae"].notna().any():
        raise ValueError("Empty subgroup rows contain a finite MAE")
    support = (
        frame.groupby(["category_scope", "category_label", "analysis_name", "layer"])["mae"]
        .count()
        .groupby(level=[0, 1])
        .agg(["min", "max"])
    )
    for key_value, expected in EXPECTED_BIN_SUPPORT.items():
        observed = support.loc[key_value]
        if int(observed["min"]) != expected or int(observed["max"]) != expected:
            raise ValueError(f"Unexpected finite protein support for {key_value}: {observed.to_dict()}")


def load_subgroup_values(cohort: Sequence[str]) -> tuple[pd.DataFrame, list[Path]]:
    path = require_input_path(SUBGROUP_DIR / "per_protein_subgroup_mae.csv")
    columns = [
        "analysis_name",
        "model_family",
        "layer",
        "protein_id",
        "source",
        "cv_protocol_version",
        "evaluation_design",
        "outer_fold",
        "evaluation_scope",
        "oof_prediction_scope",
        "generalization_scope",
        "category_scope",
        "category_label",
        "n_pairs",
        "sum_absolute_error",
        "mae",
        "support_status",
    ]
    frame = pd.read_csv(
        path,
        usecols=columns,
        dtype={"protein_id": "string", "category_label": "string"},
    )
    frame["protein_id"] = normalize_protein_ids(frame["protein_id"])
    frame["category_label"] = frame["category_label"].astype(str)
    frame["value"] = frame["mae"]
    frame["model"] = frame["analysis_name"].map(
        lambda value: MODEL_SPECS[str(value)]["label"]
    )
    validate_subgroup_values(frame, cohort)

    reconciliation_path = require_input_path(SUBGROUP_DIR / "pooled_reconciliation.csv")
    reconciliation = pd.read_csv(reconciliation_path, dtype={"category_label": "string"})
    if len(reconciliation) != 2_112 or set(reconciliation["status"]) != {"complete"}:
        raise ValueError("Pooled reconciliation is not complete for all 2,112 diagnostics")
    if not np.allclose(
        reconciliation["replayed_n_pairs"], reconciliation["producer_n_pairs"], rtol=0, atol=0
    ):
        raise ValueError("Replayed subgroup pair counts do not reconcile")
    if not np.allclose(
        reconciliation["replayed_pooled_mae"],
        reconciliation["producer_pooled_mae"],
        rtol=1e-12,
        atol=1e-12,
    ):
        raise ValueError("Replayed subgroup pooled MAE does not reconcile")
    return frame, [path, reconciliation_path]


def validate_triangle_values(frame: pd.DataFrame, cohort: Sequence[str]) -> None:
    expected_rows = len(MODEL_ORDER) * len(LAYERS) * EXPECTED_PROTEINS * 13
    if len(frame) != expected_rows:
        raise ValueError(f"Focused triangle table has {len(frame)} rows; expected {expected_rows}")
    if set(frame["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError("Focused triangle table has the wrong model set")
    if set(frame["protein_id"]) != set(cohort):
        raise ValueError("Focused triangle table has the wrong protein cohort")
    key = ["analysis_name", "layer", "protein_id", "category_system", "category_label"]
    if frame.duplicated(key).any():
        raise ValueError("Focused triangle table contains duplicate keys")
    for system, spec in TRIANGLE_CATEGORY_SPECS.items():
        labels = set(frame.loc[frame["category_system"].eq(system), "category_label"])
        if labels != set(spec["labels"]):
            raise ValueError(f"Focused triangle table has wrong labels for {system}: {labels}")
    for metric in TRIANGLE_METRICS:
        counts = frame.groupby(
            ["analysis_name", "category_system", "category_label", "layer"]
        )[metric].count()
        if len(counts) != 2_496 or not counts.eq(EXPECTED_PROTEINS).all():
            raise ValueError(f"Triangle metric {metric} lacks 552 finite proteins per trajectory")
        if not np.isfinite(frame[metric]).all():
            raise ValueError(f"Triangle metric {metric} contains nonfinite values")


def load_triangle_values(cohort: Sequence[str]) -> tuple[pd.DataFrame, list[Path]]:
    path = require_input_path(TRIANGLE_DIR / "per_protein_triangle_metrics.csv")
    columns = [
        "analysis_name",
        "model_family",
        "layer",
        "protein_id",
        "source",
        "cv_protocol_version",
        "evaluation_design",
        "outer_fold",
        "evaluation_scope",
        "oof_prediction_scope",
        "generalization_scope",
        "category_system",
        "category_label",
        "n_triangles",
        "n_violations",
        "relative_violation_sum",
        "violation_fraction",
        "mean_relative_violation_all",
        "support_status",
    ]
    frame = pd.read_csv(path, usecols=columns, dtype={"protein_id": "string"})
    frame = frame[frame["category_system"].isin(TRIANGLE_CATEGORY_SPECS)].copy()
    frame["protein_id"] = normalize_protein_ids(frame["protein_id"])
    frame["model"] = frame["analysis_name"].map(
        lambda value: MODEL_SPECS[str(value)]["label"]
    )
    validate_triangle_values(frame, cohort)
    return frame, [path]


def bootstrap_counts(
    n_proteins: int = EXPECTED_PROTEINS,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
) -> np.ndarray:
    rng = np.random.default_rng(seed)
    probabilities = np.full(n_proteins, 1.0 / n_proteins)
    return rng.multinomial(n_proteins, probabilities, size=replicates).astype(np.int16)


def bootstrap_matrix_summary(matrix: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    """Protein-equal point means and paired-bootstrap intervals by column."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or counts.ndim != 2 or counts.shape[1] != values.shape[0]:
        raise ValueError("Bootstrap matrix and resample counts have incompatible shapes")
    finite = np.isfinite(values)
    finite_values = np.where(finite, values, 0.0)
    denominators = counts @ finite.astype(np.int16)
    if np.any(denominators.sum(axis=0) == 0):
        raise ValueError("A bootstrap column has no finite protein support")
    boot = np.full(denominators.shape, np.nan, dtype=float)
    np.divide(
        counts @ finite_values,
        denominators,
        out=boot,
        where=denominators > 0,
    )
    low, high = np.nanquantile(boot, [0.025, 0.975], axis=0)
    n_finite = finite.sum(axis=0)
    return {
        "mean": finite_values.sum(axis=0) / n_finite,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "n_proteins_finite": n_finite.astype(int),
    }


def bootstrap_summary(
    frame: pd.DataFrame,
    cohort: Sequence[str],
    counts: np.ndarray,
    category_columns: Sequence[str] = (),
) -> pd.DataFrame:
    required = {"analysis_name", "model", "protein_id", "layer", "value"} | set(
        category_columns
    )
    require_columns(frame, required, "bootstrap input")
    groups = [((), frame)]
    if category_columns:
        groups = list(frame.groupby(list(category_columns), sort=True, dropna=False))
    records: list[dict[str, object]] = []
    expected_columns = pd.MultiIndex.from_product([MODEL_ORDER, LAYERS])
    for category_key, group in groups:
        keys = category_key if isinstance(category_key, tuple) else (category_key,)
        pivot = group.pivot(index="protein_id", columns=["analysis_name", "layer"], values="value")
        pivot = pivot.reindex(index=cohort, columns=expected_columns)
        result = bootstrap_matrix_summary(pivot.to_numpy(dtype=float), counts)
        for column_index, (analysis, layer) in enumerate(expected_columns):
            selected = group[group["analysis_name"].eq(analysis)]
            row: dict[str, object] = {
                "analysis_name": analysis,
                "model": MODEL_SPECS[analysis]["label"],
                "layer": int(layer),
                "mean": float(result["mean"][column_index]),
                "bootstrap_ci_low": float(result["bootstrap_ci_low"][column_index]),
                "bootstrap_ci_high": float(result["bootstrap_ci_high"][column_index]),
                "n_proteins_total": EXPECTED_PROTEINS,
                "n_proteins_finite": int(result["n_proteins_finite"][column_index]),
                "aggregation": "protein_equal_mean",
                "evaluation_design": selected["evaluation_design"].drop_duplicates().item(),
                "evaluation_scope": selected["evaluation_scope"].drop_duplicates().item(),
                "oof_prediction_scope": selected["oof_prediction_scope"].drop_duplicates().item(),
                "generalization_scope": MODEL_SPECS[analysis]["generalization_scope"],
            }
            row.update(dict(zip(category_columns, keys)))
            records.append(row)
    return pd.DataFrame(records)


def ols_slope(values: np.ndarray) -> float:
    y = np.asarray(values, dtype=float)
    x = np.arange(len(y), dtype=float)
    centered = x - x.mean()
    return float(centered @ (y - y.mean()) / (centered @ centered))


def one_sample_greater(values: np.ndarray) -> dict[str, float | int | str]:
    data = np.asarray(values, dtype=float)
    data = data[np.isfinite(data)]
    n = len(data)
    if n < 2:
        return {"n": n, "mean": math.nan, "sd": math.nan, "t": math.nan, "p": math.nan, "status": "insufficient"}
    mean = float(data.mean())
    sd = float(data.std(ddof=1))
    if sd == 0.0:
        p = 0.0 if mean > 0 else 1.0
        t_value = math.inf if mean > 0 else (-math.inf if mean < 0 else 0.0)
        status = "constant"
    else:
        t_value = mean / (sd / math.sqrt(n))
        p = float(stats.t.sf(t_value, n - 1))
        status = "complete"
    return {"n": n, "mean": mean, "sd": sd, "t": t_value, "p": p, "status": status}


def benjamini_hochberg(p_values: Sequence[float]) -> np.ndarray:
    values = np.asarray(p_values, dtype=float)
    if not np.isfinite(values).all() or np.any((values < 0) | (values > 1)):
        raise ValueError("BH p-values must be finite and lie in [0, 1]")
    order = np.argsort(values, kind="stable")
    ranked = values[order]
    adjusted_ranked = ranked * len(values) / np.arange(1, len(values) + 1)
    adjusted_ranked = np.minimum.accumulate(adjusted_ranked[::-1])[::-1]
    adjusted = np.empty_like(adjusted_ranked)
    adjusted[order] = np.clip(adjusted_ranked, 0.0, 1.0)
    return adjusted


def peak_trough_test_at_layer(
    matrix: np.ndarray, layer: int, direction: str, window: int = EXTREMA_WINDOW
) -> dict[str, object]:
    """Evaluate one predefined peak or trough hypothesis for a protein matrix."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Peak/trough matrix must be a finite protein-by-layer matrix")
    if direction not in {"peak", "trough"}:
        raise ValueError(f"Unknown extremum direction: {direction}")
    if layer < window or layer + window >= values.shape[1]:
        raise ValueError("Peak/trough center lacks the requested neighbor window")
    means = values.mean(axis=0)
    left_layers = tuple(range(layer - window, layer))
    right_layers = tuple(range(layer + 1, layer + window + 1))
    left_values = values[:, left_layers].mean(axis=1)
    right_values = values[:, right_layers].mean(axis=1)
    neighbors = means[list(left_layers + right_layers)]
    left_slope = ols_slope(means[layer - window : layer + 1])
    right_slope = ols_slope(means[layer : layer + window + 1])
    sign = 1.0 if direction == "peak" else -1.0
    strict = bool(
        means[layer] > neighbors.max()
        if direction == "peak"
        else means[layer] < neighbors.min()
    )
    slope_shape = bool(
        left_slope > 0 and right_slope < 0
        if direction == "peak"
        else left_slope < 0 and right_slope > 0
    )
    left_test = one_sample_greater(sign * (values[:, layer] - left_values))
    right_test = one_sample_greater(sign * (values[:, layer] - right_values))
    return {
        "center_mean": float(means[layer]),
        "left_neighborhood_mean": float(left_values.mean()),
        "right_neighborhood_mean": float(right_values.mean()),
        "left_slope": left_slope,
        "right_slope": right_slope,
        "strict_window_extremum": strict,
        "slope_shape_pass": slope_shape,
        "shape_qualified_candidate": strict and slope_shape,
        "left_test": left_test,
        "right_test": right_test,
        "combined_p_value": max(float(left_test["p"]), float(right_test["p"])),
    }


def analyze_peak_troughs(triangle: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    records: list[dict[str, object]] = []
    group_columns = ["analysis_name", "model_family", "category_system", "category_label"]
    for key, group in triangle.groupby(group_columns, sort=True):
        analysis, model_family, category_system, category_label = key
        for metric in TRIANGLE_METRICS:
            wide = group.pivot(index="protein_id", columns="layer", values=metric).reindex(columns=LAYERS)
            matrix = wide.to_numpy(dtype=float)
            if matrix.shape != (EXPECTED_PROTEINS, len(LAYERS)) or not np.isfinite(matrix).all():
                raise ValueError(f"Incomplete peak/trough trajectory: {key}/{metric}")
            for layer in ELIGIBLE_EXTREMA_LAYERS:
                for direction in ("peak", "trough"):
                    test = peak_trough_test_at_layer(matrix, layer, direction)
                    left_test = test["left_test"]
                    right_test = test["right_test"]
                    records.append(
                        {
                            "analysis_name": analysis,
                            "model_family": model_family,
                            "model": MODEL_SPECS[str(analysis)]["label"],
                            "category_system": category_system,
                            "category_label": category_label,
                            "metric": metric,
                            "direction": direction,
                            "layer": layer,
                            "window_layers_per_side": EXTREMA_WINDOW,
                            "n_proteins": int(left_test["n"]),
                            "center_mean": test["center_mean"],
                            "left_neighborhood_mean": test["left_neighborhood_mean"],
                            "right_neighborhood_mean": test["right_neighborhood_mean"],
                            "left_slope": test["left_slope"],
                            "right_slope": test["right_slope"],
                            "strict_window_extremum": test["strict_window_extremum"],
                            "slope_shape_pass": test["slope_shape_pass"],
                            "shape_qualified_candidate": test["shape_qualified_candidate"],
                            "left_contrast_mean": float(left_test["mean"]),
                            "left_contrast_sd": float(left_test["sd"]),
                            "left_t_statistic": float(left_test["t"]),
                            "left_p_value": float(left_test["p"]),
                            "right_contrast_mean": float(right_test["mean"]),
                            "right_contrast_sd": float(right_test["sd"]),
                            "right_t_statistic": float(right_test["t"]),
                            "right_p_value": float(right_test["p"]),
                            "combined_p_value": test["combined_p_value"],
                        }
                    )
    tests = pd.DataFrame(records)
    if len(tests) != 7_904:
        raise ValueError(f"Expected 7,904 peak/trough tests; found {len(tests)}")
    tests["bh_q_value"] = math.nan
    tests["bh_family_size"] = 0
    for metric in TRIANGLE_METRICS:
        mask = tests["metric"].eq(metric)
        if int(mask.sum()) != 3_952:
            raise ValueError(f"Expected 3,952 BH hypotheses for {metric}")
        tests.loc[mask, "bh_q_value"] = benjamini_hochberg(
            tests.loc[mask, "combined_p_value"].to_numpy(dtype=float)
        )
        tests.loc[mask, "bh_family_size"] = 3_952
    tests["fdr_pass"] = tests["bh_q_value"].le(ALPHA)
    tests["supported_extremum"] = tests["shape_qualified_candidate"] & tests["fdr_pass"]
    candidates = tests[tests["shape_qualified_candidate"]].copy()

    summary_records: list[dict[str, object]] = []
    trajectory_columns = [
        "analysis_name", "model_family", "category_system", "category_label", "metric"
    ]
    for key, group in tests.groupby(trajectory_columns, sort=True):
        record = dict(zip(trajectory_columns, key))
        for direction in ("peak", "trough"):
            directional = group[group["direction"].eq(direction)]
            candidate_rows = directional[directional["shape_qualified_candidate"]]
            supported_rows = directional[directional["supported_extremum"]]
            record[f"candidate_{direction}_layers"] = ";".join(
                map(str, candidate_rows["layer"].astype(int))
            )
            record[f"supported_{direction}_layers"] = ";".join(
                map(str, supported_rows["layer"].astype(int))
            )
            record[f"n_candidate_{direction}s"] = len(candidate_rows)
            record[f"n_supported_{direction}s"] = len(supported_rows)
        summary_records.append(record)
    return tests, candidates, pd.DataFrame(summary_records)


def phase_axis_summary(matrix: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray | float]:
    """Summarize raw and fixed-range normalized bootstrap coordinates."""
    values = np.asarray(matrix, dtype=float)
    if values.ndim != 2 or not np.isfinite(values).all():
        raise ValueError("Phase-axis matrix must be finite and two-dimensional")
    if counts.shape[1] != values.shape[0]:
        raise ValueError("Phase-axis bootstrap counts have incompatible protein support")
    mean = values.mean(axis=0)
    boot = (counts @ values) / values.shape[0]
    low, high = np.quantile(boot, [0.025, 0.975], axis=0)
    minimum = float(mean.min())
    maximum = float(mean.max())
    span = maximum - minimum
    if not span > 0:
        raise ValueError("Phase-axis point estimates have zero span")
    return {
        "mean": mean,
        "bootstrap_ci_low": low,
        "bootstrap_ci_high": high,
        "minimum": minimum,
        "maximum": maximum,
        "normalized_mean": (mean - minimum) / span,
        "normalized_bootstrap_ci_low": (low - minimum) / span,
        "normalized_bootstrap_ci_high": (high - minimum) / span,
    }


def build_phase_points(
    r2: pd.DataFrame,
    triangle: pd.DataFrame,
    cohort: Sequence[str],
    counts: np.ndarray,
) -> pd.DataFrame:
    all_triangle = triangle[
        triangle["category_system"].eq("all_triangles")
        & triangle["category_label"].eq("all")
    ]
    records: list[dict[str, object]] = []
    for analysis in MODEL_ORDER:
        r2_matrix = (
            r2[r2["analysis_name"].eq(analysis)]
            .pivot(index="protein_id", columns="layer", values="value")
            .reindex(index=cohort, columns=LAYERS)
            .to_numpy(dtype=float)
        )
        r2_result = phase_axis_summary(r2_matrix, counts)
        r2_mean = np.asarray(r2_result["mean"])
        r2_min = float(r2_result["minimum"])
        r2_max = float(r2_result["maximum"])
        r2_span = r2_max - r2_min
        r2_low = np.asarray(r2_result["bootstrap_ci_low"])
        r2_high = np.asarray(r2_result["bootstrap_ci_high"])
        for metric in TRIANGLE_METRICS:
            consistency_matrix = (
                all_triangle[all_triangle["analysis_name"].eq(analysis)]
                .pivot(index="protein_id", columns="layer", values=metric)
                .reindex(index=cohort, columns=LAYERS)
                .to_numpy(dtype=float)
            )
            consistency_boot = (counts @ consistency_matrix) / EXPECTED_PROTEINS
            consistency_mean = consistency_matrix.mean(axis=0)
            consistency_low, consistency_high = np.quantile(
                consistency_boot, [0.025, 0.975], axis=0
            )
            for layer in LAYERS:
                records.append(
                    {
                        "analysis_name": analysis,
                        "model": MODEL_SPECS[analysis]["label"],
                        "layer": layer,
                        "raw_mean_per_protein_r2": float(r2_mean[layer]),
                        "raw_r2_bootstrap_ci_low": float(r2_low[layer]),
                        "raw_r2_bootstrap_ci_high": float(r2_high[layer]),
                        "model_r2_min": r2_min,
                        "model_r2_max": r2_max,
                        "normalized_r2": float((r2_mean[layer] - r2_min) / r2_span),
                        "normalized_r2_bootstrap_ci_low": float((r2_low[layer] - r2_min) / r2_span),
                        "normalized_r2_bootstrap_ci_high": float((r2_high[layer] - r2_min) / r2_span),
                        "consistency_metric": metric,
                        "consistency_value": float(consistency_mean[layer]),
                        "consistency_bootstrap_ci_low": float(consistency_low[layer]),
                        "consistency_bootstrap_ci_high": float(consistency_high[layer]),
                        "milestone": layer in MILESTONE_LAYERS,
                        "n_proteins": EXPECTED_PROTEINS,
                        "aggregation": "protein_equal_mean",
                        "evaluation_scope": "oof_all_pairs",
                    }
                )
    points = pd.DataFrame(records)
    if len(points) != len(MODEL_ORDER) * len(LAYERS) * len(TRIANGLE_METRICS):
        raise ValueError("Decodability-consistency point count is incomplete")
    return points


def plot_model_lines(axis: plt.Axes, summary: pd.DataFrame, domain: str) -> None:
    for analysis in MODEL_ORDER:
        data = summary[summary["analysis_name"].eq(analysis)].sort_values("layer")
        spec = MODEL_SPECS[analysis]
        axis.plot(
            data["layer"], data["mean"], color=spec["color"],
            linestyle=spec["linestyle"], linewidth=2.0, label=spec["label"],
        )
        low, high = clip_interval(data["bootstrap_ci_low"], data["bootstrap_ci_high"], domain)
        axis.fill_between(data["layer"], low, high, color=spec["color"], alpha=0.13, linewidth=0)


def style_layer_axis(axis: plt.Axes, ylabel: str, domain: str) -> None:
    axis.set(xlabel="Evoformer layer", ylabel=ylabel, xlim=(0, 47))
    axis.set_xticks(MILESTONE_LAYERS)
    if domain == "unit":
        axis.set_ylim(0, min(1.0, axis.get_ylim()[1]))
    elif domain == "nonnegative":
        axis.set_ylim(bottom=0)
    elif domain == "unbounded":
        axis.axhline(0, color="#666666", linewidth=0.7, alpha=0.45, zorder=0)


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> tuple[Path, Path]:
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def line_figure(
    summary: pd.DataFrame, ylabel: str, stem: str, directory: Path, domain: str
) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(8.4, 5.3))
    plot_model_lines(axis, summary, domain)
    style_layer_axis(axis, ylabel, domain)
    axis.legend(frameon=True)
    figure.tight_layout()
    return save_figure(figure, directory, stem)


def faceted_subgroup_figure(
    summary: pd.DataFrame,
    categories: Sequence[str],
    ylabel: str,
    stem: str,
    directory: Path,
    *,
    ncols: int = 3,
    figsize: tuple[float, float] | None = None,
    ylimits: Mapping[str, tuple[float, float]] | None = None,
    yticks: Mapping[str, Sequence[float]] | None = None,
    visible_y_tick_categories: Collection[str] | None = None,
    shared_axis_labels: bool = False,
    title_pad: float | None = None,
) -> tuple[Path, Path]:
    nrows = math.ceil(len(categories) / ncols)
    if figsize is None:
        figsize = (15, 3.8 * nrows)
    figure, axes = plt.subplots(nrows, ncols, figsize=figsize, sharex=True)
    flat = np.atleast_1d(axes).ravel()
    for axis, category in zip(flat, categories):
        selected = summary[summary["category_label"].eq(category)]
        plot_model_lines(axis, selected, "nonnegative")
        style_layer_axis(axis, ylabel, "nonnegative")
        if ylimits is not None and category in ylimits:
            axis.set_ylim(*ylimits[category])
        if yticks is not None and category in yticks:
            axis.set_yticks(yticks[category])
        if shared_axis_labels:
            axis.set(xlabel="", ylabel="")
        if (
            visible_y_tick_categories is not None
            and category not in visible_y_tick_categories
        ):
            axis.tick_params(axis="y", left=False, labelleft=False)
        if title_pad is None:
            axis.set_title(category)
        else:
            axis.set_title(category, pad=title_pad)
    for axis in flat[len(categories) :]:
        axis.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    if shared_axis_labels:
        shared_ylabel = figure.supylabel(
            ylabel, x=0.045, multialignment="center"
        )
        figure.supxlabel("Evoformer layer", y=0.14)
        figure.legend(
            handles, labels, loc="lower center", bbox_to_anchor=(0.5, 0.01),
            ncol=4, frameon=False,
        )
        figure.tight_layout(rect=(0.025, 0.22, 1, 0.98))
        first_axis_bounds = flat[0].get_position()
        shared_ylabel.set_y((first_axis_bounds.y0 + first_axis_bounds.y1) / 2)
    else:
        figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
        figure.tight_layout(rect=(0, 0.08, 1, 1))
    return save_figure(figure, directory, stem)


def standalone_subgroup_figures(
    summary: pd.DataFrame,
    categories: Sequence[str],
    prefix: str,
    directory: Path,
) -> list[tuple[str, Path, Path]]:
    outputs = []
    for category in categories:
        stem = f"{prefix}_{bin_slug(category)}"
        selected = summary[summary["category_label"].eq(category)]
        png, pdf = line_figure(selected, "Mean per-protein MAE (Å)", stem, directory, "nonnegative")
        outputs.append((stem, png, pdf))
    return outputs


def add_extrema_markers(
    axis: plt.Axes,
    candidates: pd.DataFrame,
    analysis: str,
    color: object,
    category_label: str | None = None,
) -> None:
    selected = candidates[candidates["analysis_name"].eq(analysis)]
    if category_label is not None:
        selected = selected[selected["category_label"].eq(category_label)]
    for row in selected.itertuples():
        marker = "^" if row.direction == "peak" else "v"
        axis.scatter(
            row.layer,
            row.center_mean,
            marker=marker,
            s=40,
            facecolor=color if bool(row.supported_extremum) else "white",
            edgecolor=color,
            linewidth=1.1,
            zorder=6,
        )


def triangle_overlay_figure(
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    metric: str,
    stem: str,
    directory: Path,
) -> tuple[Path, Path]:
    spec = TRIANGLE_METRICS[metric]
    selected_summary = summary[
        summary["category_system"].eq("all_triangles")
        & summary["category_label"].eq("all")
        & summary["metric"].eq(metric)
    ]
    selected_candidates = candidates[
        candidates["category_system"].eq("all_triangles")
        & candidates["category_label"].eq("all")
        & candidates["metric"].eq(metric)
    ]
    figure, axis = plt.subplots(figsize=(8.6, 5.6))
    plot_model_lines(axis, selected_summary, str(spec["domain"]))
    for analysis in MODEL_ORDER:
        add_extrema_markers(
            axis, selected_candidates, analysis, MODEL_SPECS[analysis]["color"]
        )
    style_layer_axis(axis, str(spec["ylabel"]), str(spec["domain"]))
    axis.set_title(str(spec["title"]))
    handles, labels = axis.get_legend_handles_labels()
    handles.extend(
        [
            Line2D([], [], marker="^", linestyle="none", markerfacecolor="white", markeredgecolor="black", label="Candidate peak/trough"),
            Line2D([], [], marker="o", linestyle="none", markerfacecolor="black", markeredgecolor="black", label="BH-supported"),
        ]
    )
    labels.extend(["Candidate peak/trough", "BH-supported"])
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.15, 1, 1))
    return save_figure(figure, directory, stem)


def triangle_stratified_figure(
    summary: pd.DataFrame,
    candidates: pd.DataFrame,
    system: str,
    stem: str,
) -> tuple[Path, Path]:
    category_spec = TRIANGLE_CATEGORY_SPECS[system]
    labels = tuple(category_spec["labels"])
    colors = {label: plt.get_cmap("tab10")(index) for index, label in enumerate(labels)}
    figure, axes = plt.subplots(2, 4, figsize=(16.5, 8.0), sharex=True, sharey="row")
    for row_index, metric in enumerate(TRIANGLE_METRICS):
        metric_spec = TRIANGLE_METRICS[metric]
        for column_index, analysis in enumerate(MODEL_ORDER):
            axis = axes[row_index, column_index]
            for label in labels:
                data = summary[
                    summary["category_system"].eq(system)
                    & summary["category_label"].eq(label)
                    & summary["metric"].eq(metric)
                    & summary["analysis_name"].eq(analysis)
                ].sort_values("layer")
                color = colors[label]
                axis.plot(data["layer"], data["mean"], color=color, linewidth=1.8, label=display_category(label))
                low, high = clip_interval(
                    data["bootstrap_ci_low"], data["bootstrap_ci_high"], str(metric_spec["domain"])
                )
                axis.fill_between(data["layer"], low, high, color=color, alpha=0.11, linewidth=0)
                marker_rows = candidates[
                    candidates["category_system"].eq(system)
                    & candidates["metric"].eq(metric)
                ]
                add_extrema_markers(axis, marker_rows, analysis, color, label)
            style_layer_axis(axis, str(metric_spec["ylabel"]), str(metric_spec["domain"]))
            if row_index == 0:
                axis.set_title(str(MODEL_SPECS[analysis]["label"]))
    category_handles = [Line2D([], [], color=colors[label], linewidth=2, label=display_category(label)) for label in labels]
    marker_handles = [
        Line2D([], [], marker="^", linestyle="none", markerfacecolor="white", markeredgecolor="black", label="Candidate peak/trough"),
        Line2D([], [], marker="o", linestyle="none", markerfacecolor="black", markeredgecolor="black", label="BH-supported"),
    ]
    figure.legend(category_handles + marker_handles, [h.get_label() for h in category_handles + marker_handles], loc="lower center", ncol=5, frameon=False)
    figure.suptitle(str(category_spec["title"]), y=0.99, fontweight="semibold")
    figure.tight_layout(rect=(0, 0.09, 1, 0.96))
    return save_figure(figure, ANNOTATED_DIR, stem)


def phase_figure(
    points: pd.DataFrame, metric: str, stem: str, directory: Path
) -> tuple[Path, Path]:
    metric_spec = TRIANGLE_METRICS[metric]
    selected = points[points["consistency_metric"].eq(metric)]
    figure, axis = plt.subplots(figsize=(8.7, 5.9))
    label_shifts = (-8, 8, -7, 7)
    for model_index, analysis in enumerate(MODEL_ORDER):
        spec = MODEL_SPECS[analysis]
        data = selected[selected["analysis_name"].eq(analysis)].sort_values("layer")
        axis.plot(
            data["normalized_r2"], data["consistency_value"], color=spec["color"],
            linestyle=spec["linestyle"], linewidth=2.0, label=spec["label"], zorder=2,
        )
        axis.scatter(data["normalized_r2"], data["consistency_value"], color=spec["color"], s=10, zorder=3)
        milestones = data[data["milestone"]]
        x_low, x_high = clip_interval(
            milestones["normalized_r2_bootstrap_ci_low"],
            milestones["normalized_r2_bootstrap_ci_high"],
            "unit",
        )
        y_low, y_high = clip_interval(
            milestones["consistency_bootstrap_ci_low"],
            milestones["consistency_bootstrap_ci_high"],
            str(metric_spec["domain"]),
        )
        x = milestones["normalized_r2"].to_numpy(dtype=float)
        y = milestones["consistency_value"].to_numpy(dtype=float)
        axis.errorbar(
            x, y, xerr=np.vstack([np.maximum(0, x - x_low), np.maximum(0, x_high - x)]),
            yerr=np.vstack([np.maximum(0, y - y_low), np.maximum(0, y_high - y)]), fmt="none",
            ecolor=spec["color"], elinewidth=0.8, alpha=0.42, capsize=1.5, zorder=1,
        )
        axis.scatter(x, y, color=spec["color"], s=31, edgecolors="white", linewidths=0.65, zorder=4)
        for row in milestones.itertuples():
            axis.annotate(
                str(row.layer), (row.normalized_r2, row.consistency_value),
                xytext=(4, label_shifts[model_index]), textcoords="offset points",
                fontsize=6.2 * (plt.rcParams["font.size"] / 10),
                color=spec["color"], bbox={"facecolor": "white", "edgecolor": "none", "alpha": 0.65, "pad": 0.1},
            )
    axis.set(
        title=str(metric_spec["title"]),
        xlabel="Normalized mean per-protein all-pairs $R^2$",
        ylabel=str(metric_spec["ylabel"]),
        xlim=(0, 1),
    )
    axis.set_xticks((0, 0.25, 0.5, 0.75, 1))
    if metric_spec["domain"] == "unit":
        axis.set_ylim(0, min(1.0, axis.get_ylim()[1]))
    else:
        axis.set_ylim(bottom=0)
    axis.legend(frameon=True)
    figure.tight_layout()
    return save_figure(figure, directory, stem)


def individual_by_model_figure(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    ylabel: str,
    domain: str,
    stem: str,
) -> tuple[Path, Path]:
    figure, axes = plt.subplots(2, 2, figsize=(12.5, 8.5), sharex=True, sharey=True)
    for axis, analysis in zip(axes.ravel(), MODEL_ORDER):
        spec = MODEL_SPECS[analysis]
        model_raw = raw[raw["analysis_name"].eq(analysis)]
        for _, trajectory in model_raw.groupby("protein_id", sort=False):
            trajectory = trajectory.sort_values("layer")
            axis.plot(trajectory["layer"], trajectory["value"], color=spec["color"], alpha=0.025, linewidth=0.28)
        model_summary = summary[summary["analysis_name"].eq(analysis)].sort_values("layer")
        low, high = clip_interval(
            model_summary["bootstrap_ci_low"], model_summary["bootstrap_ci_high"], domain
        )
        axis.fill_between(model_summary["layer"], low, high, color=spec["color"], alpha=0.18, linewidth=0)
        axis.plot(
            model_summary["layer"], model_summary["mean"], color=spec["color"],
            linestyle=spec["linestyle"], linewidth=2.3,
        )
        style_layer_axis(axis, ylabel, domain)
        axis.set_title(str(spec["label"]))
    figure.tight_layout()
    return save_figure(figure, INDIVIDUAL_DIR, stem)


def write_source_csv(frame: pd.DataFrame, name: str, *, compressed: bool = False) -> Path:
    suffix = ".csv.gz" if compressed else ".csv"
    path = SOURCE_DIR / f"{name}{suffix}"
    if compressed:
        frame.to_csv(
            path,
            index=False,
            compression={"method": "gzip", "compresslevel": 6, "mtime": 0},
        )
    else:
        frame.to_csv(path, index=False)
    return path


def manifest_row(
    *,
    stem: str,
    section: str,
    title: str,
    png: Path,
    pdf: Path,
    source_csvs: Sequence[Path],
    uncertainty: str,
) -> dict[str, object]:
    return {
        "figure_id": stem,
        "section": section,
        "title": title,
        "png_path": relative(png),
        "pdf_path": relative(pdf),
        "source_csvs": ";".join(relative(path) for path in source_csvs),
        "cohort": "552 protocol-2.0 proteins",
        "evaluation_scope": "oof_all_pairs",
        "aggregation": "protein_equal_mean",
        "uncertainty": uncertainty,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "historical_placeholder": False,
        "aa_gated_included": False,
        "png_sha256": sha256_file(png),
        "pdf_sha256": sha256_file(pdf),
        "generator_version": GENERATOR_VERSION,
    }


def main() -> None:
    configure_style()
    ensure_directories()
    generated_at = datetime.now(timezone.utc).isoformat()
    cohort = load_cohort()
    counts = bootstrap_counts()

    print("Loading fivefold R2 and geometry values...")
    r2, r2_inputs = load_core_metric(cohort, "per_protein.csv", "r2", "prediction_status")
    tm, tm_inputs = load_core_metric(cohort, "geometry_metrics.csv", "tm_score", "geometry_status")
    top3, top3_inputs = load_core_metric(
        cohort, "geometry_metrics.csv", "frac_top3_eigenvalues", "geometry_status"
    )
    stress, stress_inputs = load_core_metric(
        cohort, "geometry_metrics.csv", "stress", "geometry_status"
    )
    print("Loading per-protein subgroup MAE values...")
    subgroup, subgroup_inputs = load_subgroup_values(cohort)
    spatial = subgroup[subgroup["category_scope"].eq("spatial_distance")].copy()
    sequence = subgroup[subgroup["category_scope"].eq("sequence_separation")].copy()
    print("Loading focused fivefold triangle values...")
    triangle, triangle_inputs = load_triangle_values(cohort)

    print("Computing paired-protein bootstrap summaries...")
    r2_summary = bootstrap_summary(r2, cohort, counts)
    tm_summary = bootstrap_summary(tm, cohort, counts)
    top3_summary = bootstrap_summary(top3, cohort, counts)
    stress_summary = bootstrap_summary(stress, cohort, counts)
    spatial_summary = bootstrap_summary(
        spatial, cohort, counts, ("category_scope", "category_label")
    )
    sequence_summary = bootstrap_summary(
        sequence, cohort, counts, ("category_scope", "category_label")
    )
    triangle_summaries = []
    for metric in TRIANGLE_METRICS:
        values = triangle.copy()
        values["value"] = values[metric]
        summary = bootstrap_summary(
            values, cohort, counts, ("category_system", "category_label")
        )
        summary["metric"] = metric
        triangle_summaries.append(summary)
    triangle_summary = pd.concat(triangle_summaries, ignore_index=True)

    print("Running peak/trough tests and BH correction...")
    all_tests, candidates, trajectory_summary = analyze_peak_troughs(triangle)
    phase_points = build_phase_points(r2, triangle, cohort, counts)

    print("Writing source-data tables...")
    source_paths: dict[str, Path] = {}
    source_paths["r2_raw"] = write_source_csv(r2, "r2_per_protein_values")
    source_paths["r2_summary"] = write_source_csv(r2_summary, "r2_plot_summary")
    source_paths["tm_raw"] = write_source_csv(tm, "tm_score_per_protein_values")
    source_paths["tm_summary"] = write_source_csv(tm_summary, "tm_score_plot_summary")
    source_paths["top3_raw"] = write_source_csv(
        top3, "top3_eigenvalue_fraction_per_protein_values"
    )
    source_paths["top3_summary"] = write_source_csv(
        top3_summary, "top3_eigenvalue_fraction_plot_summary"
    )
    source_paths["stress_raw"] = write_source_csv(
        stress, "mds_stress_per_protein_values"
    )
    source_paths["stress_summary"] = write_source_csv(
        stress_summary, "mds_stress_plot_summary"
    )
    source_paths["spatial_raw"] = write_source_csv(
        spatial, "spatial_distance_mae_per_protein_values", compressed=True
    )
    source_paths["spatial_summary"] = write_source_csv(
        spatial_summary, "spatial_distance_mae_plot_summary"
    )
    source_paths["sequence_raw"] = write_source_csv(
        sequence, "sequence_separation_mae_per_protein_values", compressed=True
    )
    source_paths["sequence_summary"] = write_source_csv(
        sequence_summary, "sequence_separation_mae_plot_summary"
    )
    source_paths["triangle_raw"] = write_source_csv(
        triangle, "triangle_per_protein_values", compressed=True
    )
    source_paths["triangle_summary"] = write_source_csv(
        triangle_summary, "triangle_plot_summary"
    )
    source_paths["all_tests"] = write_source_csv(all_tests, "triangle_all_layer_tests")
    source_paths["candidates"] = write_source_csv(candidates, "triangle_candidate_extrema")
    source_paths["trajectory"] = write_source_csv(
        trajectory_summary, "triangle_trajectory_summary"
    )
    source_paths["phase"] = write_source_csv(
        phase_points, "decodability_consistency_points"
    )
    reconciliation = pd.read_csv(
        require_input_path(SUBGROUP_DIR / "pooled_reconciliation.csv"),
        dtype={"category_label": "string"},
    )
    source_paths["reconciliation"] = write_source_csv(
        reconciliation, "pooled_reconciliation_diagnostic"
    )

    manifest: list[dict[str, object]] = []
    standard_uncertainty = "paired protein bootstrap percentile 95% CI"
    phase_uncertainty = "joint paired-protein bootstrap 95% x/y intervals at milestone layers"

    configure_style(font_scale=MAIN_FIGURE_FONT_SCALE)
    png, pdf = line_figure(
        r2_summary, "Mean per-protein $R^2$", "main_figure_01_mean_per_protein_r2",
        MAIN_DIR, "unbounded",
    )
    manifest.append(manifest_row(stem="main_figure_01_mean_per_protein_r2", section="main", title="Mean per-protein fivefold OOF R2", png=png, pdf=pdf, source_csvs=[source_paths["r2_raw"], source_paths["r2_summary"]], uncertainty=standard_uncertainty))
    png, pdf = line_figure(
        tm_summary, "Mean reconstruction TM-score", "main_figure_03_tm_score",
        MAIN_DIR, "unit",
    )
    manifest.append(manifest_row(stem="main_figure_03_tm_score", section="main", title="Mean fivefold OOF reconstruction TM-score", png=png, pdf=pdf, source_csvs=[source_paths["tm_raw"], source_paths["tm_summary"]], uncertainty=standard_uncertainty))

    png, pdf = faceted_subgroup_figure(
        spatial_summary, SPATIAL_BINS, "Mean per-protein\nMAE (Å)",
        "main_figure_15_spatial_distance_mae_combined", MAIN_DIR,
        ncols=len(SPATIAL_BINS),
        figsize=(15, 3.9),
        ylimits={category: (0, 5) for category in SPATIAL_BINS[:-1]}
        | {SPATIAL_BINS[-1]: (0, 15)},
        yticks={category: tuple(range(6)) for category in SPATIAL_BINS[:-1]}
        | {SPATIAL_BINS[-1]: (0, 5, 10, 15)},
        visible_y_tick_categories=(SPATIAL_BINS[0], SPATIAL_BINS[-1]),
        shared_axis_labels=True,
        title_pad=14,
    )
    manifest.append(manifest_row(stem="main_figure_15_spatial_distance_mae_combined", section="main", title="Protein-equal MAE by true spatial distance", png=png, pdf=pdf, source_csvs=[source_paths["spatial_raw"], source_paths["spatial_summary"], source_paths["reconciliation"]], uncertainty=standard_uncertainty))
    for stem, png, pdf in standalone_subgroup_figures(
        spatial_summary, SPATIAL_BINS, "main_figure_15_spatial_distance_mae", MAIN_DIR
    ):
        manifest.append(manifest_row(stem=stem, section="main", title=f"Protein-equal spatial-bin MAE: {stem.rsplit('_', 2)[-1]}", png=png, pdf=pdf, source_csvs=[source_paths["spatial_raw"], source_paths["spatial_summary"]], uncertainty=standard_uncertainty))

    png, pdf = triangle_overlay_figure(
        triangle_summary, candidates, "mean_relative_violation_all",
        "main_figure_all_triangles_peak_troughs_overlay", MAIN_DIR,
    )
    manifest.append(manifest_row(stem="main_figure_all_triangles_peak_troughs_overlay", section="main", title="Normalized mean all-triangle violation with peak/trough annotations", png=png, pdf=pdf, source_csvs=[source_paths["triangle_raw"], source_paths["triangle_summary"], source_paths["all_tests"], source_paths["candidates"]], uncertainty=standard_uncertainty))
    png, pdf = phase_figure(
        phase_points, "mean_relative_violation_all",
        "main_figure_all_triangles_decodability_consistency", MAIN_DIR,
    )
    manifest.append(manifest_row(stem="main_figure_all_triangles_decodability_consistency", section="main", title="Normalized decodability versus normalized mean triangle violation", png=png, pdf=pdf, source_csvs=[source_paths["r2_raw"], source_paths["triangle_raw"], source_paths["phase"]], uncertainty=phase_uncertainty))

    configure_style()
    png, pdf = faceted_subgroup_figure(
        sequence_summary, SEQUENCE_BINS, "Mean per-protein MAE (Å)",
        "supp_figure_sequence_separation_mae_combined", SUPPLEMENT_DIR,
    )
    manifest.append(manifest_row(stem="supp_figure_sequence_separation_mae_combined", section="supplemental", title="Protein-equal MAE by sequence separation", png=png, pdf=pdf, source_csvs=[source_paths["sequence_raw"], source_paths["sequence_summary"], source_paths["reconciliation"]], uncertainty=standard_uncertainty))
    for stem, png, pdf in standalone_subgroup_figures(
        sequence_summary, SEQUENCE_BINS, "supp_figure_sequence_separation_mae", SUPPLEMENT_DIR
    ):
        manifest.append(manifest_row(stem=stem, section="supplemental", title=f"Protein-equal sequence-bin MAE: {stem.rsplit('_', 2)[-1]}", png=png, pdf=pdf, source_csvs=[source_paths["sequence_raw"], source_paths["sequence_summary"]], uncertainty=standard_uncertainty))

    fraction_peak_stem = "supp_figure_all_triangles_violation_fraction_peak_troughs_overlay"
    png, pdf = triangle_overlay_figure(
        triangle_summary, candidates, "violation_fraction", fraction_peak_stem, SUPPLEMENT_DIR
    )
    manifest.append(manifest_row(stem=fraction_peak_stem, section="supplemental", title="All-triangle violation fraction with peak/trough annotations", png=png, pdf=pdf, source_csvs=[source_paths["triangle_raw"], source_paths["triangle_summary"], source_paths["all_tests"], source_paths["candidates"]], uncertainty=standard_uncertainty))
    fraction_phase_stem = "supp_figure_all_triangles_violation_fraction_decodability_consistency"
    png, pdf = phase_figure(
        phase_points, "violation_fraction", fraction_phase_stem, SUPPLEMENT_DIR
    )
    manifest.append(manifest_row(stem=fraction_phase_stem, section="supplemental", title="Normalized decodability versus all-triangle violation fraction", png=png, pdf=pdf, source_csvs=[source_paths["r2_raw"], source_paths["triangle_raw"], source_paths["phase"]], uncertainty=phase_uncertainty))

    top3_stem = "supp_figure_top3_eigenvalue_fraction"
    png, pdf = line_figure(
        top3_summary, "Mean top-3 eigenvalue fraction", top3_stem,
        SUPPLEMENT_DIR, "unit",
    )
    manifest.append(manifest_row(stem=top3_stem, section="supplemental", title="Mean top-3 MDS eigenvalue fraction across Evoformer layers", png=png, pdf=pdf, source_csvs=[source_paths["top3_raw"], source_paths["top3_summary"]], uncertainty=standard_uncertainty))

    stress_stem = "supp_figure_mds_stress"
    png, pdf = line_figure(
        stress_summary, "Mean normalized MDS stress", stress_stem,
        SUPPLEMENT_DIR, "nonnegative",
    )
    manifest.append(manifest_row(stem=stress_stem, section="supplemental", title="Mean normalized MDS stress across Evoformer layers", png=png, pdf=pdf, source_csvs=[source_paths["stress_raw"], source_paths["stress_summary"]], uncertainty=standard_uncertainty))

    for system, category_spec in TRIANGLE_CATEGORY_SPECS.items():
        if system == "all_triangles":
            continue
        stem = str(category_spec["stem"])
        png, pdf = triangle_stratified_figure(
            triangle_summary, candidates, system, stem
        )
        manifest.append(manifest_row(stem=stem, section="supplemental", title=f"{category_spec['title']} annotated triangle trajectories", png=png, pdf=pdf, source_csvs=[source_paths["triangle_raw"], source_paths["triangle_summary"], source_paths["all_tests"], source_paths["candidates"], source_paths["trajectory"]], uncertainty=standard_uncertainty))

    all_triangle_raw = triangle[
        triangle["category_system"].eq("all_triangles")
        & triangle["category_label"].eq("all")
    ].copy()
    individual_specs = (
        (r2, r2_summary, "Mean per-protein $R^2$", "unbounded", "supp_figure_individual_proteins_r2_by_model", [source_paths["r2_raw"], source_paths["r2_summary"]]),
        (tm, tm_summary, "Reconstruction TM-score", "unit", "supp_figure_individual_proteins_tm_score_by_model", [source_paths["tm_raw"], source_paths["tm_summary"]]),
    )
    for raw, summary, ylabel, domain, stem, sources in individual_specs:
        png, pdf = individual_by_model_figure(raw, summary, ylabel, domain, stem)
        manifest.append(manifest_row(stem=stem, section="supplemental", title=f"Model-separated individual-protein trajectories: {ylabel}", png=png, pdf=pdf, source_csvs=sources, uncertainty=standard_uncertainty))
    for metric, stem in (
        ("mean_relative_violation_all", "supp_figure_individual_proteins_normalized_triangle_violation_by_model"),
        ("violation_fraction", "supp_figure_individual_proteins_triangle_violation_fraction_by_model"),
    ):
        raw = all_triangle_raw.copy()
        raw["value"] = raw[metric]
        summary = triangle_summary[
            triangle_summary["category_system"].eq("all_triangles")
            & triangle_summary["category_label"].eq("all")
            & triangle_summary["metric"].eq(metric)
        ]
        metric_spec = TRIANGLE_METRICS[metric]
        png, pdf = individual_by_model_figure(
            raw, summary, str(metric_spec["ylabel"]), str(metric_spec["domain"]), stem
        )
        manifest.append(manifest_row(stem=stem, section="supplemental", title=f"Model-separated individual-protein trajectories: {metric_spec['title']}", png=png, pdf=pdf, source_csvs=[source_paths["triangle_raw"], source_paths["triangle_summary"]], uncertainty=standard_uncertainty))

    manifest_frame = pd.DataFrame(manifest)
    expected_ids = set(expected_figure_ids())
    if len(manifest_frame) != 29 or set(manifest_frame["figure_id"]) != expected_ids:
        missing = sorted(expected_ids - set(manifest_frame["figure_id"]))
        extra = sorted(set(manifest_frame["figure_id"]) - expected_ids)
        raise ValueError(f"Figure inventory mismatch; missing={missing}, extra={extra}")
    if manifest_frame["figure_id"].duplicated().any():
        raise ValueError("Figure manifest IDs are not unique")
    manifest_path = OUTPUT_ROOT / "figure_manifest.csv"
    manifest_frame.to_csv(manifest_path, index=False)

    input_paths = [
        OUTER_FOLD_PATH,
        *r2_inputs,
        *tm_inputs,
        *top3_inputs,
        *stress_inputs,
        *subgroup_inputs,
        *triangle_inputs,
        SUBGROUP_DIR / "run_manifest.json",
        TRIANGLE_DIR / "run_manifest.json",
    ]
    resolved_inputs = [require_input_path(path) for path in input_paths]
    generation_summary = {
        "generator_version": GENERATOR_VERSION,
        "generated_at": generated_at,
        "input_root": relative(REGRESSION_ROOT),
        "input_policy": "all analytical and provenance inputs resolve below regression_5fold_v1",
        "bootstrap": {
            "method": "paired-protein percentile, pointwise 95% intervals",
            "replicates": BOOTSTRAP_REPLICATES,
            "seed": BOOTSTRAP_SEED,
        },
        "peak_trough": {
            "eligible_layers": list(ELIGIBLE_EXTREMA_LAYERS),
            "window_layers_per_side": EXTREMA_WINDOW,
            "bh_method": "Benjamini-Hochberg",
            "families": "separate_by_metric_across_all_plotted_models_categories_directions_layers",
            "family_size_per_metric": 3_952,
            "alpha": ALPHA,
        },
        "expected_counts": {
            "models": 4,
            "proteins": EXPECTED_PROTEINS,
            "layers": 48,
            "r2_rows": 105_984,
            "tm_score_rows": 105_984,
            "top3_eigenvalue_fraction_rows": 105_984,
            "mds_stress_rows": 105_984,
            "spatial_mae_rows": 529_920,
            "sequence_mae_rows": 635_904,
            "focused_triangle_rows": 1_377_792,
            "peak_trough_tests": 7_904,
            "figures": 29,
        },
        "observed_counts": {
            "models": len(MODEL_ORDER),
            "proteins": len(cohort),
            "layers": len(LAYERS),
            "r2_rows": len(r2),
            "tm_score_rows": len(tm),
            "top3_eigenvalue_fraction_rows": len(top3),
            "mds_stress_rows": len(stress),
            "spatial_mae_rows": len(spatial),
            "sequence_mae_rows": len(sequence),
            "focused_triangle_rows": len(triangle),
            "peak_trough_tests": len(all_tests),
            "figures": len(manifest_frame),
        },
        "models": list(MODEL_ORDER),
        "aa_gated_included": False,
        "historical_inputs_used": False,
        "pooled_mae_used_for_plots": False,
        "inputs": {relative(path): sha256_file(path) for path in resolved_inputs},
        "source_data": {relative(path): sha256_file(path) for path in source_paths.values()},
        "manifest": relative(manifest_path),
        "output_inventory": {
            row.figure_id: {"png": row.png_path, "pdf": row.pdf_path}
            for row in manifest_frame.itertuples()
        },
    }
    (OUTPUT_ROOT / "generation_summary.json").write_text(
        json.dumps(generation_summary, indent=2) + "\n"
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "# September 2 publication plots\n\n"
        "This bundle is derived exclusively from `delta_files/regression_5fold_v1/`. "
        "It contains four protocol-2.0 fivefold OOF probe families, 552 proteins, and "
        "layers 0–47; no AA-gated model or historical triangle placeholder is used. "
        "Every plotted mean—including R2, TM-score, top-3 MDS eigenvalue fraction, "
        "normalized MDS stress, subgroup MAE, normalized triangle violation, and "
        "triangle violation fraction—is protein-equal. Bands are "
        "pointwise paired-protein bootstrap percentile 95% intervals from 2,000 "
        "replicates (seed 20260902). Long-range bins average finite supported proteins "
        "and report their counts in source data. Pooled subgroup MAE is retained only "
        "as a reconciliation diagnostic and never supplies a plotted value. Peak/trough "
        "BH correction is separate by triangle metric and spans all plotted models, "
        "categories, directions, and eligible layers (3,952 hypotheses per metric). "
        "Per-protein OLS measures unseen pairs within seen proteins; the other families "
        "measure unseen proteins. See `figure_manifest.csv`, `generation_summary.json`, "
        "and `source_data/` for exact provenance.\n"
    )
    print(f"Generated {len(manifest_frame)} figure pairs in {relative(OUTPUT_ROOT)}")


if __name__ == "__main__":
    main()

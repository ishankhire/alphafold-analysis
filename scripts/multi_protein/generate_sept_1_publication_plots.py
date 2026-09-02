#!/usr/bin/env python3
"""Generate the self-contained September 1 publication figure bundle."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Sequence


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D


GENERATOR_VERSION = "sept_1_publication_plots_v2"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260901
EXPECTED_PROTEINS = 552
LAYERS = tuple(range(48))
SEQUENCE_BINS = ("1", "2-4", "5-9", "10-19", "20-49", "50+")
SPATIAL_BINS = ("0-5", "5-10", "10-20", "20-40", "40+")

DELTA_DIR = PROJECT_ROOT / "delta_files"
REGRESSION_DIR = DELTA_DIR / "regression_5fold_v1"
SUBGROUP_PATH = (
    DELTA_DIR / "analysis/publication_subgroup_mae_5fold_v1/per_protein_subgroup_mae.csv"
)
POOLED_SUBGROUP_FILENAMES = {
    "spatial_distance": "by_spatial_distance.csv",
    "sequence_separation": "by_sequence_separation.csv",
}
HISTORICAL_RAW = DELTA_DIR / "triangle_violation_subgroups/per_protein_triangle_metrics.csv"
HISTORICAL_ANALYSIS = DELTA_DIR / "analysis/triangle_violation_subgroups"
PEAK_ROOT = HISTORICAL_ANALYSIS / "peak_trough_significance"
OUTPUT_ROOT = DELTA_DIR / "publication_plots/sept_1_plots"
MAIN_DIR = OUTPUT_ROOT / "main_figures"
SUPPLEMENT_DIR = OUTPUT_ROOT / "supplemental_figures"
VARIANT_DIR = SUPPLEMENT_DIR / "uncertainty_variants"
ANNOTATED_DIR = SUPPLEMENT_DIR / "triangle_annotated_trajectories"
SOURCE_DIR = OUTPUT_ROOT / "source_data"

MODEL_SPECS = {
    "shared_ols_base256": {
        "label": "Shared linear", "color": "#4C78A8", "linestyle": "-",
        "generalization_scope": "unseen_protein_ids",
    },
    "per_protein_ols_base256": {
        "label": "Per-protein linear", "color": "#7F7F7F",
        "linestyle": (0, (5, 2)), "generalization_scope": "unseen_pairs_within_seen_protein",
    },
    "shared_xgboost_base256_fixed_s4": {
        "label": "XGBoost", "color": "#E45756", "linestyle": "-",
        "generalization_scope": "unseen_protein_ids",
    },
    "sequence_gated_ols_base256": {
        "label": "Sequence gated", "color": "#F58518", "linestyle": "-",
        "generalization_scope": "unseen_protein_ids",
    },
}
MODEL_ORDER = tuple(MODEL_SPECS)
HISTORICAL_NAME_MAP = {
    "general_base256_combined": "shared_ols_base256",
    "per_protein_base256_combined": "per_protein_ols_base256",
    "shared_xgboost_base256_combined": "shared_xgboost_base256_fixed_s4",
    "analysis_1_sequence_gated": "sequence_gated_ols_base256",
}


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 10, "axes.titlesize": 12,
        "axes.labelsize": 11, "xtick.labelsize": 9, "ytick.labelsize": 9,
        "legend.fontsize": 8.5, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.alpha": 0.22, "savefig.dpi": 300,
        "savefig.bbox": "tight", "pdf.fonttype": 42, "ps.fonttype": 42,
    })


def ensure_directories() -> None:
    for path in (MAIN_DIR, SUPPLEMENT_DIR, VARIANT_DIR, ANNOTATED_DIR, SOURCE_DIR):
        path.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bin_slug(label: str) -> str:
    return label.replace("+", "_plus").replace("-", "_")


def require_columns(frame: pd.DataFrame, columns: Iterable[str], context: str) -> None:
    missing = sorted(set(columns) - set(frame.columns))
    if missing:
        raise ValueError(f"{context} is missing columns: {missing}")


def validate_layer_protein_values(frame: pd.DataFrame, context: str, expected: int = EXPECTED_PROTEINS) -> None:
    if set(frame["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError(f"{context} does not contain exactly the four planned models")
    if set(pd.to_numeric(frame["layer"]).astype(int)) != set(LAYERS):
        raise ValueError(f"{context} does not contain layers 0-47")
    key = ["analysis_name", "layer", "protein_id"]
    if frame.duplicated(key).any():
        raise ValueError(f"{context} contains duplicate model/layer/protein rows")
    for analysis, group in frame.groupby("analysis_name"):
        counts = group.groupby("layer")["protein_id"].nunique()
        if len(counts) != 48 or not (counts == expected).all():
            raise ValueError(f"{context}/{analysis} does not have {expected} proteins at every layer")


def load_metric_values(filename: str, metric: str, status_column: str) -> pd.DataFrame:
    frames = []
    common_ids: set[str] | None = None
    for analysis in MODEL_ORDER:
        path = REGRESSION_DIR / analysis / filename
        frame = pd.read_csv(path, dtype={"protein_id": str})
        require_columns(
            frame,
            {"analysis_name", "layer", "protein_id", "source", "outer_fold", metric, status_column,
             "evaluation_design", "evaluation_scope", "oof_prediction_scope"},
            relative(path),
        )
        if set(frame[status_column]) != {"complete"}:
            raise ValueError(f"{relative(path)} contains incomplete rows")
        frame["protein_id"] = frame["protein_id"].str.zfill(9)
        ids = set(frame["protein_id"])
        common_ids = ids if common_ids is None else common_ids
        if ids != common_ids:
            raise ValueError(f"{relative(path)} does not use the common fivefold cohort")
        selected = frame[[
            "analysis_name", "layer", "protein_id", "source", "outer_fold",
            "evaluation_design", "evaluation_scope", "oof_prediction_scope", metric,
        ]].rename(columns={metric: "value"})
        selected["model"] = MODEL_SPECS[analysis]["label"]
        selected["generalization_scope"] = MODEL_SPECS[analysis]["generalization_scope"]
        frames.append(selected)
    result = pd.concat(frames, ignore_index=True)
    validate_layer_protein_values(result, metric)
    if not np.isfinite(pd.to_numeric(result["value"], errors="coerce")).all():
        raise ValueError(f"{metric} values contain nonfinite entries")
    return result


def validate_subgroup_values(frame: pd.DataFrame, expected_proteins: int = EXPECTED_PROTEINS) -> None:
    required = {
        "analysis_name", "model_family", "layer", "protein_id", "source", "outer_fold",
        "evaluation_design", "evaluation_scope", "oof_prediction_scope", "generalization_scope",
        "category_scope", "category_label", "n_pairs", "sum_absolute_error", "mae", "support_status",
    }
    require_columns(frame, required, relative(SUBGROUP_PATH))
    if set(frame["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError("Per-protein subgroup MAE does not contain exactly the four planned models")
    if set(frame["layer"].astype(int)) != set(LAYERS):
        raise ValueError("Per-protein subgroup MAE does not contain layers 0-47")
    expected_rows = expected_proteins * 4 * 48 * 11
    if len(frame) != expected_rows:
        raise ValueError(f"Per-protein subgroup MAE has {len(frame)} rows, expected {expected_rows}")
    key = ["analysis_name", "layer", "protein_id", "category_scope", "category_label"]
    if frame.duplicated(key).any():
        raise ValueError("Per-protein subgroup MAE contains duplicate keys")
    expected_labels = {"sequence_separation": set(SEQUENCE_BINS), "spatial_distance": set(SPATIAL_BINS)}
    for scope, labels in expected_labels.items():
        subset = frame[frame["category_scope"].eq(scope)]
        if set(subset["category_label"].astype(str)) != labels:
            raise ValueError(f"Per-protein subgroup MAE has wrong labels for {scope}")
    supported = frame["n_pairs"].gt(0)
    if not np.isfinite(frame.loc[supported, "mae"]).all() or frame.loc[~supported, "mae"].notna().any():
        raise ValueError("Per-protein subgroup MAE has invalid supported/empty values")


def load_subgroup_values() -> pd.DataFrame:
    if not SUBGROUP_PATH.is_file():
        raise FileNotFoundError(
            f"Required per-protein subgroup replay is missing: {SUBGROUP_PATH}. "
            "Run and copy publication_subgroup_mae_5fold_v1; pooled MAE is not an allowed fallback."
        )
    frame = pd.read_csv(SUBGROUP_PATH, dtype={"protein_id": str, "category_label": str})
    frame["protein_id"] = frame["protein_id"].str.zfill(9)
    validate_subgroup_values(frame)
    frame["model"] = frame["analysis_name"].map(lambda value: MODEL_SPECS[value]["label"])
    frame["value"] = frame["mae"]
    frame["aggregation"] = "protein_equal_mean"
    frame["temporary_pooled_mae_placeholder"] = False
    return frame


def validate_pooled_subgroup_values(frame: pd.DataFrame) -> None:
    required = {
        "analysis_name", "model_family", "layer", "evaluation_design",
        "evaluation_scope", "oof_prediction_scope", "category_scope",
        "category_label", "n_proteins", "n_expected_pairs",
        "n_evaluated_pairs", "pair_coverage", "support_status", "pooled_mae",
    }
    require_columns(frame, required, "fivefold pooled subgroup MAE")
    if set(frame["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError("Pooled subgroup MAE does not contain exactly the four planned models")
    if set(pd.to_numeric(frame["layer"]).astype(int)) != set(LAYERS):
        raise ValueError("Pooled subgroup MAE does not contain layers 0-47")
    expected_rows = len(MODEL_ORDER) * len(LAYERS) * (len(SPATIAL_BINS) + len(SEQUENCE_BINS))
    if len(frame) != expected_rows:
        raise ValueError(f"Pooled subgroup MAE has {len(frame)} rows, expected {expected_rows}")
    key = ["analysis_name", "layer", "category_scope", "category_label"]
    if frame.duplicated(key).any():
        raise ValueError("Pooled subgroup MAE contains duplicate keys")
    expected_labels = {"sequence_separation": set(SEQUENCE_BINS), "spatial_distance": set(SPATIAL_BINS)}
    for scope, labels in expected_labels.items():
        subset = frame[frame["category_scope"].eq(scope)]
        if set(subset["category_label"].astype(str)) != labels:
            raise ValueError(f"Pooled subgroup MAE has wrong labels for {scope}")
    if frame["n_proteins"].le(0).any() or frame["n_proteins"].gt(EXPECTED_PROTEINS).any():
        raise ValueError("Pooled subgroup MAE has invalid per-category protein support")
    support_counts = frame.groupby(["category_scope", "category_label"])["n_proteins"].nunique()
    if not support_counts.eq(1).all():
        raise ValueError("Pooled subgroup MAE protein support differs across models or layers")
    if set(frame["support_status"]) != {"sufficient"}:
        raise ValueError("Pooled subgroup MAE contains unsupported rows")
    if not np.isfinite(pd.to_numeric(frame["pooled_mae"], errors="coerce")).all():
        raise ValueError("Pooled subgroup MAE contains nonfinite values")
    if not frame["n_evaluated_pairs"].eq(frame["n_expected_pairs"]).all():
        raise ValueError("Pooled subgroup MAE does not have complete pair coverage")


def load_pooled_subgroup_values() -> tuple[pd.DataFrame, list[Path]]:
    """Load the temporary pair-pooled fallback from canonical fivefold outputs."""
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for analysis in MODEL_ORDER:
        for scope, filename in POOLED_SUBGROUP_FILENAMES.items():
            path = REGRESSION_DIR / analysis / "subgroup_metrics" / filename
            frame = pd.read_csv(path, dtype={"category_label": str})
            if set(frame["category_scope"]) != {scope}:
                raise ValueError(f"{relative(path)} has the wrong category scope")
            frames.append(frame)
            paths.append(path)
    result = pd.concat(frames, ignore_index=True)
    validate_pooled_subgroup_values(result)
    result["model"] = result["analysis_name"].map(lambda value: MODEL_SPECS[value]["label"])
    result["generalization_scope"] = result["analysis_name"].map(
        lambda value: MODEL_SPECS[value]["generalization_scope"]
    )
    result["value"] = result["pooled_mae"]
    result["aggregation"] = "pair_pooled_mae"
    result["temporary_pooled_mae_placeholder"] = True
    return result, paths


def pooled_subgroup_summary(frame: pd.DataFrame) -> pd.DataFrame:
    """Put pair-pooled trajectories into the publication summary schema."""
    rows = frame.copy()
    rows["mean"] = rows["pooled_mae"]
    for column in (
        "sd", "se", "bootstrap_ci_low", "bootstrap_ci_high", "two_sd_low", "two_sd_high",
    ):
        rows[column] = np.nan
    rows["n_proteins_total"] = EXPECTED_PROTEINS
    rows["n_proteins_finite"] = rows["n_proteins"].astype(int)
    rows["aggregation"] = "pair_pooled_mae"
    rows["historical_placeholder"] = False
    rows["temporary_pooled_mae_placeholder"] = True
    return rows


def paired_bootstrap_summary(
    frame: pd.DataFrame,
    category_columns: Sequence[str] = (),
    *,
    replicates: int = BOOTSTRAP_REPLICATES,
    seed: int = BOOTSTRAP_SEED,
    placeholder: bool = False,
) -> pd.DataFrame:
    """Summarize trajectories with one paired protein bootstrap per category."""
    required = {"analysis_name", "model", "protein_id", "layer", "value"} | set(category_columns)
    require_columns(frame, required, "bootstrap input")
    group_columns = list(category_columns)
    category_groups = [((), frame)] if not group_columns else list(frame.groupby(group_columns, sort=True, dropna=False))
    rng = np.random.default_rng(seed)
    rows: list[dict[str, object]] = []
    for category_key, group in category_groups:
        keys = category_key if isinstance(category_key, tuple) else (category_key,)
        pivot = group.pivot(index="protein_id", columns=["analysis_name", "layer"], values="value")
        expected_columns = pd.MultiIndex.from_product([MODEL_ORDER, LAYERS])
        pivot = pivot.reindex(columns=expected_columns)
        support = pivot.notna().all(axis=1)
        finite = pivot.loc[support].to_numpy(dtype=float)
        if finite.size == 0 or not np.isfinite(finite).all():
            raise ValueError(f"No common finite protein support for category {keys}")
        n = finite.shape[0]
        counts = rng.multinomial(n, np.full(n, 1.0 / n), size=replicates)
        bootstrap = counts @ finite / n
        low, high = np.quantile(bootstrap, [0.025, 0.975], axis=0)
        for column_index, (analysis, layer) in enumerate(expected_columns):
            values = finite[:, column_index]
            sd = float(np.std(values, ddof=1)) if n > 1 else math.nan
            model_rows = group[group["analysis_name"].eq(analysis)]
            evaluation_scope = (
                str(model_rows["evaluation_scope"].drop_duplicates().item())
                if "evaluation_scope" in model_rows and model_rows["evaluation_scope"].nunique() == 1
                else "not_recorded"
            )
            oof_prediction_scope = (
                str(model_rows["oof_prediction_scope"].drop_duplicates().item())
                if "oof_prediction_scope" in model_rows and model_rows["oof_prediction_scope"].nunique() == 1
                else "historical_protocol_1_not_applicable"
            )
            generalization_scope = (
                str(model_rows["generalization_scope"].drop_duplicates().item())
                if "generalization_scope" in model_rows and model_rows["generalization_scope"].nunique() == 1
                else str(MODEL_SPECS[analysis]["generalization_scope"])
            )
            row = {
                "analysis_name": analysis,
                "model": MODEL_SPECS[analysis]["label"],
                "layer": int(layer),
                "mean": float(np.mean(values)),
                "sd": sd,
                "se": sd / math.sqrt(n) if n else math.nan,
                "bootstrap_ci_low": float(low[column_index]),
                "bootstrap_ci_high": float(high[column_index]),
                "two_sd_low": float(np.mean(values) - 2 * sd),
                "two_sd_high": float(np.mean(values) + 2 * sd),
                "n_proteins_total": int(group["protein_id"].nunique()),
                "n_proteins_finite": n,
                "aggregation": "protein_equal_mean",
                "evaluation_scope": evaluation_scope,
                "oof_prediction_scope": oof_prediction_scope,
                "generalization_scope": generalization_scope,
                "historical_placeholder": bool(placeholder),
                "temporary_pooled_mae_placeholder": False,
            }
            row.update(dict(zip(group_columns, keys)))
            rows.append(row)
    return pd.DataFrame(rows)


def plot_model_lines(axis: plt.Axes, summary: pd.DataFrame, band: str | None = "bootstrap") -> None:
    for analysis in MODEL_ORDER:
        data = summary[summary["analysis_name"].eq(analysis)].sort_values("layer")
        spec = MODEL_SPECS[analysis]
        axis.plot(data["layer"], data["mean"], color=spec["color"], linestyle=spec["linestyle"], linewidth=2, label=spec["label"])
        if band is not None:
            low, high = (
                ("bootstrap_ci_low", "bootstrap_ci_high") if band == "bootstrap"
                else ("two_sd_low", "two_sd_high")
            )
            axis.fill_between(data["layer"].to_numpy(float), data[low].to_numpy(float), data[high].to_numpy(float), color=spec["color"], alpha=0.13 if band == "bootstrap" else 0.09, linewidth=0)


def plot_individual_lines(axis: plt.Axes, raw: pd.DataFrame, summary: pd.DataFrame) -> None:
    for analysis in MODEL_ORDER:
        spec = MODEL_SPECS[analysis]
        model_raw = raw[raw["analysis_name"].eq(analysis)]
        for _, trajectory in model_raw.groupby("protein_id", sort=False):
            trajectory = trajectory.sort_values("layer")
            axis.plot(trajectory["layer"], trajectory["value"], color=spec["color"], alpha=0.025, linewidth=0.28)
        mean = summary[summary["analysis_name"].eq(analysis)].sort_values("layer")
        axis.plot(mean["layer"], mean["mean"], color=spec["color"], linestyle=spec["linestyle"], linewidth=2.2, label=spec["label"])


def style_axis(axis: plt.Axes, ylabel: str, *, bounded: bool = False) -> None:
    axis.set(xlabel="Evoformer layer", ylabel=ylabel, xlim=(0, 47))
    axis.set_xticks([0, 10, 20, 30, 40, 47])
    if bounded:
        axis.set_ylim(0, 1)
    else:
        axis.set_ylim(bottom=0)


def save_figure(figure: plt.Figure, directory: Path, stem: str) -> tuple[Path, Path]:
    png = directory / f"{stem}.png"
    pdf = directory / f"{stem}.pdf"
    figure.savefig(png)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def line_figure(summary: pd.DataFrame, ylabel: str, stem: str, directory: Path, *, band: str | None = "bootstrap", bounded: bool = False) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(8.4, 5.3))
    plot_model_lines(axis, summary, band=band)
    style_axis(axis, ylabel, bounded=bounded)
    axis.legend(frameon=True)
    figure.tight_layout()
    return save_figure(figure, directory, stem)


def spaghetti_figure(raw: pd.DataFrame, summary: pd.DataFrame, ylabel: str, stem: str, *, bounded: bool = False) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(8.4, 5.3))
    plot_individual_lines(axis, raw, summary)
    style_axis(axis, ylabel, bounded=bounded)
    axis.legend(frameon=True)
    figure.tight_layout()
    return save_figure(figure, VARIANT_DIR, stem)


def faceted_figure(
    raw: pd.DataFrame,
    summary: pd.DataFrame,
    categories: Sequence[str],
    ylabel: str,
    stem: str,
    directory: Path,
    *,
    band: str = "bootstrap",
    individual: bool = False,
) -> tuple[Path, Path]:
    ncols = 3
    nrows = math.ceil(len(categories) / ncols)
    figure, axes = plt.subplots(nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), sharex=True)
    flat = np.atleast_1d(axes).ravel()
    for axis, category in zip(flat, categories):
        raw_category = raw[raw["category_label"].astype(str).eq(category)]
        summary_category = summary[summary["category_label"].astype(str).eq(category)]
        if individual:
            plot_individual_lines(axis, raw_category, summary_category)
        else:
            plot_model_lines(axis, summary_category, band=band)
        style_axis(axis, ylabel)
        axis.set_title(category)
    for axis in flat[len(categories):]:
        axis.set_visible(False)
    handles, labels = flat[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=4, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    return save_figure(figure, directory, stem)


def standalone_subgroup_figures(
    summary: pd.DataFrame, categories: Sequence[str], prefix: str, directory: Path,
    *, ylabel: str, band: str | None,
) -> list[tuple[str, Path, Path]]:
    outputs = []
    for category in categories:
        stem = f"{prefix}_{bin_slug(category)}"
        subset = summary[summary["category_label"].astype(str).eq(category)]
        png, pdf = line_figure(subset, ylabel, stem, directory, band=band)
        outputs.append((stem, png, pdf))
    return outputs


def load_historical_triangle_values() -> pd.DataFrame:
    systems = {"all_triangles", "rms_size", "area", "relative_slack", "angle_shape"}
    columns = [
        "analysis_name", "model_family", "layer", "protein_id", "source", "protein_split",
        "evaluation_scope", "category_system", "category_label", "violation_fraction",
        "mean_relative_violation_all", "support_status", "generalization_scope",
    ]
    frame = pd.read_csv(HISTORICAL_RAW, usecols=columns, dtype={"protein_id": str})
    frame = frame[frame["category_system"].isin(systems)].copy()
    frame["analysis_name_historical"] = frame["analysis_name"]
    frame["analysis_name"] = frame["analysis_name"].map(HISTORICAL_NAME_MAP)
    if frame["analysis_name"].isna().any() or set(frame["analysis_name"]) != set(MODEL_ORDER):
        raise ValueError("Historical triangle rows do not map to the four planned models")
    frame["model"] = frame["analysis_name"].map(lambda value: MODEL_SPECS[value]["label"])
    frame["protein_id"] = frame["protein_id"].str.zfill(9)
    return frame


def plot_historical_overlay(
    summaries: dict[str, pd.DataFrame], candidates: pd.DataFrame,
) -> tuple[Path, Path]:
    panels = (
        ("violation_fraction", "Triangle-inequality violation fraction", "Fraction of triangles"),
        ("mean_relative_violation_all", "Normalized mean triangle violation", "Mean relative violation per triangle"),
    )
    figure, axes = plt.subplots(1, 2, figsize=(15.2, 5.5), sharex=True)
    for axis, (metric, title, ylabel) in zip(axes, panels):
        plot_model_lines(axis, summaries[metric])
        markers = candidates[
            candidates["category_system"].eq("all_triangles") & candidates["metric"].eq(metric)
        ]
        for row in markers.itertuples():
            analysis = HISTORICAL_NAME_MAP[str(row.analysis_name)]
            marker = "^" if row.direction == "peak" else "v"
            supported = bool(row.supported_extremum)
            axis.scatter(
                row.layer, row.center_mean, marker=marker, s=48,
                facecolor=MODEL_SPECS[analysis]["color"] if supported else "white",
                edgecolor=MODEL_SPECS[analysis]["color"], linewidth=1.2, zorder=5,
            )
        style_axis(axis, ylabel)
        axis.set_title(title)
    handles, labels = axes[0].get_legend_handles_labels()
    handles.extend([
        Line2D([], [], marker="^", linestyle="None", markerfacecolor="white", markeredgecolor="black", label="Candidate peak/trough"),
        Line2D([], [], marker="o", linestyle="None", markerfacecolor="black", markeredgecolor="black", label="BH-supported"),
    ])
    labels.extend(["Candidate peak/trough", "BH-supported"])
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.12, 1, 1))
    return save_figure(figure, MAIN_DIR, "main_figure_all_triangles_peak_troughs_overlay")


def manifest_row(
    *, stem: str, section: str, title: str, png: Path, pdf: Path,
    source_csvs: Sequence[Path], uncertainty: str, cohort: str,
    scope: str, placeholder: bool, copied: bool,
    aggregation: str = "protein_equal_mean",
    temporary_pooled_mae_placeholder: bool = False,
) -> dict[str, object]:
    return {
        "figure_id": stem, "section": section, "title": title,
        "png_path": relative(png), "pdf_path": relative(pdf),
        "source_csvs": ";".join(relative(path) for path in source_csvs),
        "cohort": cohort, "evaluation_scope": scope, "aggregation": aggregation,
        "uncertainty": uncertainty, "historical_placeholder": placeholder,
        "temporary_pooled_mae_placeholder": temporary_pooled_mae_placeholder,
        "copied_verbatim": copied, "png_sha256": sha256_file(png), "pdf_sha256": sha256_file(pdf),
        "generator_version": GENERATOR_VERSION,
    }


def copy_pair(source_directory: Path, stem: str, destination: Path, output_stem: str | None = None) -> tuple[Path, Path]:
    target_stem = output_stem or stem
    png = destination / f"{target_stem}.png"
    pdf = destination / f"{target_stem}.pdf"
    shutil.copy2(source_directory / f"{stem}.png", png)
    shutil.copy2(source_directory / f"{stem}.pdf", pdf)
    if sha256_file(png) != sha256_file(source_directory / f"{stem}.png") or sha256_file(pdf) != sha256_file(source_directory / f"{stem}.pdf"):
        raise RuntimeError(f"Verbatim copy hash mismatch for {stem}")
    return png, pdf


def main() -> None:
    configure_style()
    ensure_directories()
    generated_at = datetime.now(timezone.utc).isoformat()

    r2 = load_metric_values("per_protein.csv", "r2", "prediction_status")
    tm = load_metric_values("geometry_metrics.csv", "tm_score", "geometry_status")
    if SUBGROUP_PATH.is_file():
        subgroup = load_subgroup_values()
        subgroup_mode = "per_protein_mae"
        subgroup_inputs = [SUBGROUP_PATH]
        subgroup_summary = None
        subgroup_band = "bootstrap"
        subgroup_ylabel = "Mean per-protein MAE (Å)"
        subgroup_aggregation = "protein_equal_mean"
        subgroup_uncertainty = "paired protein bootstrap percentile 95% CI"
        subgroup_placeholder = False
        subgroup_raw_suffix = "per_protein_values"
        print(f"Using per-protein subgroup MAE: {relative(SUBGROUP_PATH)}")
    else:
        subgroup, subgroup_inputs = load_pooled_subgroup_values()
        subgroup_mode = "temporary_pair_pooled_mae"
        subgroup_summary = pooled_subgroup_summary(subgroup)
        subgroup_band = None
        subgroup_ylabel = "Pooled MAE (Å)"
        subgroup_aggregation = "pair_pooled_mae"
        subgroup_uncertainty = "none (single pair-pooled estimate)"
        subgroup_placeholder = True
        subgroup_raw_suffix = "pooled_values"
        print(
            "Per-protein subgroup replay is unavailable; using the explicit temporary "
            "pair-pooled fivefold fallback."
        )
    spatial = subgroup[subgroup["category_scope"].eq("spatial_distance")].copy()
    sequence = subgroup[subgroup["category_scope"].eq("sequence_separation")].copy()
    historical = load_historical_triangle_values()

    r2_summary = paired_bootstrap_summary(r2)
    tm_summary = paired_bootstrap_summary(tm)
    if subgroup_summary is None:
        spatial_summary = paired_bootstrap_summary(spatial, ("category_scope", "category_label"))
        sequence_summary = paired_bootstrap_summary(sequence, ("category_scope", "category_label"))
    else:
        spatial_summary = subgroup_summary[
            subgroup_summary["category_scope"].eq("spatial_distance")
        ].copy()
        sequence_summary = subgroup_summary[
            subgroup_summary["category_scope"].eq("sequence_separation")
        ].copy()
    historical_summaries = {}
    all_triangle = historical[historical["category_system"].eq("all_triangles")].copy()
    for metric in ("violation_fraction", "mean_relative_violation_all"):
        metric_values = all_triangle.rename(columns={metric: "value"})
        historical_summaries[metric] = paired_bootstrap_summary(metric_values, placeholder=True)

    source_paths = {}
    spatial_raw_key = f"spatial_distance_mae_{subgroup_raw_suffix}"
    sequence_raw_key = f"sequence_separation_mae_{subgroup_raw_suffix}"
    for name, frame in (
        ("r2_per_protein_values", r2), ("r2_plot_summary", r2_summary),
        ("tm_score_per_protein_values", tm), ("tm_score_plot_summary", tm_summary),
        (spatial_raw_key, spatial), ("spatial_distance_mae_plot_summary", spatial_summary),
        (sequence_raw_key, sequence), ("sequence_separation_mae_plot_summary", sequence_summary),
        ("triangle_peak_trough_values_historical", historical),
    ):
        path = SOURCE_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        source_paths[name] = path
    historical_summary = pd.concat([
        frame.assign(metric=metric) for metric, frame in historical_summaries.items()
    ], ignore_index=True)
    source_paths["all_triangles_plot_summary_historical"] = SOURCE_DIR / "all_triangles_plot_summary_historical.csv"
    historical_summary.to_csv(source_paths["all_triangles_plot_summary_historical"], index=False)

    candidates = pd.read_csv(PEAK_ROOT / "candidate_extrema.csv")
    all_tests = pd.read_csv(PEAK_ROOT / "all_layer_tests.csv")
    trajectory = pd.read_csv(PEAK_ROOT / "trajectory_summary.csv")
    decodability = pd.read_csv(HISTORICAL_ANALYSIS / "focused_decodability_consistency/decodability_consistency_points.csv")
    for name, frame in (
        ("triangle_candidate_extrema_historical", candidates),
        ("triangle_all_layer_tests_historical", all_tests),
        ("triangle_trajectory_summary_historical", trajectory),
        ("triangle_decodability_consistency_historical", decodability),
    ):
        path = SOURCE_DIR / f"{name}.csv"
        frame.to_csv(path, index=False)
        source_paths[name] = path

    manifest = []
    png, pdf = line_figure(r2_summary, "Mean per-protein $R^2$", "main_figure_01_mean_per_protein_r2", MAIN_DIR, bounded=True)
    manifest.append(manifest_row(stem="main_figure_01_mean_per_protein_r2", section="main", title="Mean per-protein fivefold OOF R2", png=png, pdf=pdf, source_csvs=[source_paths["r2_per_protein_values"], source_paths["r2_plot_summary"]], uncertainty="paired protein bootstrap percentile 95% CI", cohort="552 protocol-2.0 proteins", scope="oof_all_pairs", placeholder=False, copied=False))
    png, pdf = line_figure(tm_summary, "Mean reconstruction TM-score", "main_figure_03_tm_score", MAIN_DIR, bounded=True)
    manifest.append(manifest_row(stem="main_figure_03_tm_score", section="main", title="Mean fivefold OOF reconstruction TM-score", png=png, pdf=pdf, source_csvs=[source_paths["tm_score_per_protein_values"], source_paths["tm_score_plot_summary"]], uncertainty="paired protein bootstrap percentile 95% CI", cohort="552 protocol-2.0 proteins", scope="oof_all_pairs", placeholder=False, copied=False))

    png, pdf = faceted_figure(spatial, spatial_summary, SPATIAL_BINS, subgroup_ylabel, "main_figure_15_spatial_distance_mae_combined", MAIN_DIR, band=subgroup_band)
    manifest.append(manifest_row(stem="main_figure_15_spatial_distance_mae_combined", section="main", title=f"{subgroup_ylabel} by true spatial distance", png=png, pdf=pdf, source_csvs=[source_paths[spatial_raw_key], source_paths["spatial_distance_mae_plot_summary"]], uncertainty=subgroup_uncertainty, cohort="552 protocol-2.0 proteins", scope="oof_all_pairs", placeholder=False, copied=False, aggregation=subgroup_aggregation, temporary_pooled_mae_placeholder=subgroup_placeholder))
    for stem, png, pdf in standalone_subgroup_figures(spatial_summary, SPATIAL_BINS, "main_figure_15_spatial_distance_mae", MAIN_DIR, ylabel=subgroup_ylabel, band=subgroup_band):
        manifest.append(manifest_row(stem=stem, section="main", title=f"Spatial-bin {subgroup_ylabel}: {stem.rsplit('_', 1)[-1]}", png=png, pdf=pdf, source_csvs=[source_paths[spatial_raw_key], source_paths["spatial_distance_mae_plot_summary"]], uncertainty=subgroup_uncertainty, cohort="552 protocol-2.0 proteins", scope="oof_all_pairs", placeholder=False, copied=False, aggregation=subgroup_aggregation, temporary_pooled_mae_placeholder=subgroup_placeholder))

    png, pdf = faceted_figure(sequence, sequence_summary, SEQUENCE_BINS, subgroup_ylabel, "supp_figure_sequence_separation_mae_combined", SUPPLEMENT_DIR, band=subgroup_band)
    manifest.append(manifest_row(stem="supp_figure_sequence_separation_mae_combined", section="supplemental", title=f"{subgroup_ylabel} by sequence separation", png=png, pdf=pdf, source_csvs=[source_paths[sequence_raw_key], source_paths["sequence_separation_mae_plot_summary"]], uncertainty=subgroup_uncertainty, cohort="552 protocol-2.0 proteins", scope="oof_all_pairs", placeholder=False, copied=False, aggregation=subgroup_aggregation, temporary_pooled_mae_placeholder=subgroup_placeholder))
    for stem, png, pdf in standalone_subgroup_figures(sequence_summary, SEQUENCE_BINS, "supp_figure_sequence_separation_mae", SUPPLEMENT_DIR, ylabel=subgroup_ylabel, band=subgroup_band):
        manifest.append(manifest_row(stem=stem, section="supplemental", title=f"Sequence-bin {subgroup_ylabel}: {stem.rsplit('_', 1)[-1]}", png=png, pdf=pdf, source_csvs=[source_paths[sequence_raw_key], source_paths["sequence_separation_mae_plot_summary"]], uncertainty=subgroup_uncertainty, cohort="552 protocol-2.0 proteins", scope="oof_all_pairs", placeholder=False, copied=False, aggregation=subgroup_aggregation, temporary_pooled_mae_placeholder=subgroup_placeholder))

    png, pdf = plot_historical_overlay(historical_summaries, candidates)
    manifest.append(manifest_row(stem="main_figure_all_triangles_peak_troughs_overlay", section="main", title="Historical all-triangles peak/trough overlay", png=png, pdf=pdf, source_csvs=[source_paths["triangle_peak_trough_values_historical"], source_paths["all_triangles_plot_summary_historical"], source_paths["triangle_candidate_extrema_historical"]], uncertainty="paired protein bootstrap percentile 95% CI", cohort="111 historical locked-test proteins", scope="all_edges", placeholder=True, copied=False))
    png, pdf = copy_pair(HISTORICAL_ANALYSIS / "focused_decodability_consistency", "figure_01_all_triangles_decodability_consistency", MAIN_DIR, "main_figure_all_triangles_decodability_consistency")
    manifest.append(manifest_row(stem="main_figure_all_triangles_decodability_consistency", section="main", title="Historical all-triangles decodability consistency", png=png, pdf=pdf, source_csvs=[source_paths["triangle_decodability_consistency_historical"]], uncertainty="none", cohort="111 historical locked-test proteins", scope="all_edges", placeholder=True, copied=True))

    for stem in (
        "figure_02_rms_size_peak_troughs", "figure_03_area_peak_troughs",
        "figure_04_relative_slack_peak_troughs", "figure_05_angle_shape_peak_troughs",
    ):
        png, pdf = copy_pair(PEAK_ROOT / "annotated_trajectories", stem, ANNOTATED_DIR)
        manifest.append(manifest_row(stem=stem, section="supplemental", title=f"Historical annotated triangle trajectories: {stem}", png=png, pdf=pdf, source_csvs=[source_paths["triangle_peak_trough_values_historical"], source_paths["triangle_all_layer_tests_historical"], source_paths["triangle_candidate_extrema_historical"], source_paths["triangle_trajectory_summary_historical"]], uncertainty="none", cohort="111 historical locked-test proteins", scope="all_edges", placeholder=True, copied=True))

    variant_specs = (
        (r2, r2_summary, "Mean per-protein $R^2$", "main_figure_01_mean_per_protein_r2", True, None),
        (tm, tm_summary, "Mean reconstruction TM-score", "main_figure_03_tm_score", True, None),
    )
    for raw, summary, ylabel, base, bounded, _ in variant_specs:
        png, pdf = line_figure(summary, ylabel, f"{base}_2sd", VARIANT_DIR, band="2sd", bounded=bounded)
        manifest.append(manifest_row(stem=f"{base}_2sd", section="supplemental_variant", title=f"{base}: mean plus or minus 2 SD", png=png, pdf=pdf, source_csvs=[source_paths["r2_per_protein_values"] if base.startswith("main_figure_01") else source_paths["tm_score_per_protein_values"], source_paths["r2_plot_summary"] if base.startswith("main_figure_01") else source_paths["tm_score_plot_summary"]], uncertainty="mean +/- 2 sample SD", cohort="552 protocol-2.0 proteins", scope="oof_all_pairs", placeholder=False, copied=False))
        png, pdf = spaghetti_figure(raw, summary, ylabel, f"{base}_individual_proteins", bounded=bounded)
        manifest.append(manifest_row(stem=f"{base}_individual_proteins", section="supplemental_variant", title=f"{base}: individual protein trajectories", png=png, pdf=pdf, source_csvs=[source_paths["r2_per_protein_values"] if base.startswith("main_figure_01") else source_paths["tm_score_per_protein_values"]], uncertainty="raw individual proteins", cohort="552 protocol-2.0 proteins", scope="oof_all_pairs", placeholder=False, copied=False))

    if subgroup_mode == "per_protein_mae":
        for raw, summary, categories, base, raw_key, summary_key in (
            (spatial, spatial_summary, SPATIAL_BINS, "main_figure_15_spatial_distance_mae_combined", spatial_raw_key, "spatial_distance_mae_plot_summary"),
            (sequence, sequence_summary, SEQUENCE_BINS, "supp_figure_sequence_separation_mae_combined", sequence_raw_key, "sequence_separation_mae_plot_summary"),
        ):
            png, pdf = faceted_figure(raw, summary, categories, "Mean per-protein MAE (Å)", f"{base}_2sd", VARIANT_DIR, band="2sd")
            manifest.append(manifest_row(stem=f"{base}_2sd", section="supplemental_variant", title=f"{base}: mean plus or minus 2 SD", png=png, pdf=pdf, source_csvs=[source_paths[raw_key], source_paths[summary_key]], uncertainty="mean +/- 2 sample SD", cohort="552 protocol-2.0 proteins", scope="oof_all_pairs", placeholder=False, copied=False))
            png, pdf = faceted_figure(raw, summary, categories, "Mean per-protein MAE (Å)", f"{base}_individual_proteins", VARIANT_DIR, individual=True)
            manifest.append(manifest_row(stem=f"{base}_individual_proteins", section="supplemental_variant", title=f"{base}: individual protein trajectories", png=png, pdf=pdf, source_csvs=[source_paths[raw_key]], uncertainty="raw individual proteins", cohort="552 protocol-2.0 proteins", scope="oof_all_pairs", placeholder=False, copied=False))

    manifest_frame = pd.DataFrame(manifest)
    if manifest_frame["figure_id"].duplicated().any():
        raise ValueError("Figure manifest IDs are not unique")
    manifest_frame.to_csv(OUTPUT_ROOT / "figure_manifest.csv", index=False)
    input_paths = [
        *(REGRESSION_DIR / analysis / filename for analysis in MODEL_ORDER for filename in ("per_protein.csv", "geometry_metrics.csv")),
        *subgroup_inputs, HISTORICAL_RAW, PEAK_ROOT / "candidate_extrema.csv",
    ]
    summary = {
        "generator_version": GENERATOR_VERSION, "generated_at": generated_at,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES, "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_method": "paired protein percentile, pointwise 95% intervals",
        "figures": len(manifest_frame), "models": list(MODEL_ORDER),
        "aa_gated_included": False, "fivefold_proteins": EXPECTED_PROTEINS,
        "historical_triangle_proteins": 111,
        "subgroup_mae_mode": subgroup_mode,
        "subgroup_mae_temporary_placeholder": subgroup_placeholder,
        "subgroup_mae_uncertainty": subgroup_uncertainty,
        "inputs": {relative(path): sha256_file(path) for path in input_paths},
        "manifest": relative(OUTPUT_ROOT / "figure_manifest.csv"),
    }
    (OUTPUT_ROOT / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    subgroup_readme = (
        "Spatial- and sequence-bin MAE currently use pair-pooled fivefold estimates as an "
        "explicit temporary placeholder; those figures have no uncertainty bands or individual-"
        "protein variants. Regenerate after copying the per-protein replay CSV to replace them "
        "automatically."
        if subgroup_placeholder else
        "Spatial- and sequence-bin MAE use equal-weight per-protein means from the replay output."
    )
    (OUTPUT_ROOT / "README.md").write_text(
        "# September 1 publication plots\n\n"
        "This bundle uses protocol-2.0 fivefold OOF results for four probe families and "
        "protein-equal means for R2 and TM-score. Their main uncertainty bands are paired-protein "
        "percentile bootstrap 95% intervals. The triangle figures are explicitly historical "
        "111-protein placeholders "
        "in the manifest and source tables. Per-protein OLS measures unseen pairs within seen "
        "proteins; the other families measure unseen proteins. No AA-gated model is included. "
        f"{subgroup_readme} See `figure_manifest.csv` and `source_data/` for exact provenance.\n"
    )
    print(f"Generated {len(manifest_frame)} publication figures in {relative(OUTPUT_ROOT)}")


if __name__ == "__main__":
    main()

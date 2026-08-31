#!/usr/bin/env python3
"""Generate main-paper figures 1-9 and 11-18 from canonical probe outputs.

Figures 5 and 6 from the catalog are deliberately consolidated into one
six-panel triangle-violation diagnostic.  The script is local-only: it reads
canonical CSV artifacts already copied into ``delta_files`` and writes derived
tables, PNG/PDF figures, and a source-hashed figure manifest.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from datetime import datetime, timezone
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd
from scipy import stats


matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D


DELTA_DIR = PROJECT_ROOT / "delta_files"
ANALYSIS_DIR = DELTA_DIR / "analysis"
COMPARING_DIR = ANALYSIS_DIR / "comparing_variants"
GEOMETRY_DIR = ANALYSIS_DIR / "geometry_consistency"
SEQUENCE_DIR = ANALYSIS_DIR / "sequence_gated"
AA_DIR = ANALYSIS_DIR / "amino_acid_gated"
REFINEMENT_DIR = ANALYSIS_DIR / "per_protein_refinement"
PAPER_DIR = ANALYSIS_DIR / "paper_figures"
MAIN_DIR = PAPER_DIR / "main"

BOOTSTRAP_REPLICATES = 2_000
BOOTSTRAP_SEED = 20260818
TIMING_GAIN_EPS = 0.02
LAYERS = tuple(range(48))

MODEL_SPECS = {
    "Shared linear": {
        "root": DELTA_DIR
        / "general_base256_combined_canonical"
        / "general_base256_combined_canonical",
        "color": "#4C78A8",
        "linestyle": "-",
        "main_scope": "all_pairs",
        "geometry_scope": "all_pairs",
    },
    "Per-protein linear": {
        "root": DELTA_DIR / "per_protein_base256_combined_canonical",
        "color": "#7F7F7F",
        "linestyle": (0, (5, 2)),
        "main_scope": "heldout_pairs",
        "geometry_scope": "all_pairs",
    },
    "XGBoost": {
        "root": DELTA_DIR / "shared_xgboost_base256_combined",
        "color": "#E45756",
        "linestyle": "-",
        "main_scope": "all_pairs",
        "geometry_scope": "all_pairs",
    },
    "Sequence gated": {
        "root": DELTA_DIR
        / "gated_linear_probes"
        / "consolidated"
        / "analysis_1_sequence_gated",
        "color": "#F58518",
        "linestyle": "-",
        "main_scope": "all_pairs",
        "geometry_scope": "all_pairs",
    },
    "AA gated": {
        "root": DELTA_DIR
        / "gated_linear_probes"
        / "consolidated"
        / "analysis_2_aa_pair_gated",
        "color": "#54A24B",
        "linestyle": "-",
        "main_scope": "all_pairs",
        "geometry_scope": "all_pairs",
    },
}
MODEL_ORDER = tuple(MODEL_SPECS)

SEQUENCE_BINS = ("1", "2-4", "5-9", "10-19", "20-49", "50+")
DISTANCE_BINS = ("0-5", "5-10", "10-20", "20-40", "40+")
ECOD_CATEGORIES = ("alpha", "beta", "alpha_beta", "alpha_plus_beta")
CATEGORY_LABELS = {
    "alpha": "All-alpha",
    "beta": "All-beta",
    "alpha_beta": "Alpha/beta",
    "alpha_plus_beta": "Alpha+beta",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 8.5,
            "figure.titlesize": 15,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.dpi": 260,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def ensure_directories() -> None:
    for path in (
        COMPARING_DIR,
        GEOMETRY_DIR,
        SEQUENCE_DIR,
        AA_DIR,
        REFINEMENT_DIR,
        MAIN_DIR,
    ):
        path.mkdir(parents=True, exist_ok=True)


def relative(path: Path) -> str:
    return path.resolve().relative_to(PROJECT_ROOT).as_posix()


def require_columns(path: Path, frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{relative(path)} is missing columns: {missing}")


def require_layers(frame: pd.DataFrame, context: str) -> None:
    observed = tuple(sorted(pd.to_numeric(frame["layer"]).dropna().astype(int).unique()))
    if observed != LAYERS:
        raise ValueError(f"{context} does not contain all layers 0-47: {observed}")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def combined_digest(paths: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths):
        digest.update(relative(path).encode())
        digest.update(sha256_file(path).encode())
    return digest.hexdigest()


def save_figure(figure: plt.Figure, stem: str) -> tuple[Path, Path]:
    png_path = MAIN_DIR / f"{stem}.png"
    pdf_path = MAIN_DIR / f"{stem}.pdf"
    figure.savefig(png_path)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def model_legend(axis: plt.Axes, **kwargs: object) -> None:
    handles = [
        Line2D(
            [0],
            [0],
            color=MODEL_SPECS[model]["color"],
            linestyle=MODEL_SPECS[model]["linestyle"],
            linewidth=2.3,
            label=model,
        )
        for model in MODEL_ORDER
    ]
    axis.legend(handles=handles, **kwargs)


def plot_model_lines(
    axis: plt.Axes,
    frame: pd.DataFrame,
    value_column: str,
    *,
    ci: bool = False,
    linewidth: float = 2.0,
) -> None:
    for model in MODEL_ORDER:
        subset = frame[frame["model"] == model].sort_values("layer")
        if subset.empty:
            continue
        color = MODEL_SPECS[model]["color"]
        axis.plot(
            subset["layer"],
            subset[value_column],
            color=color,
            linestyle=MODEL_SPECS[model]["linestyle"],
            linewidth=linewidth,
            label=model,
        )
        if ci and {"ci_low", "ci_high"}.issubset(subset.columns):
            axis.fill_between(
                subset["layer"].to_numpy(dtype=float),
                subset["ci_low"].to_numpy(dtype=float),
                subset["ci_high"].to_numpy(dtype=float),
                color=color,
                alpha=0.13,
                linewidth=0,
            )


def bootstrap_mean_ci(values: np.ndarray, rng: np.random.Generator) -> tuple[float, float]:
    values = np.asarray(values, dtype=float)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return math.nan, math.nan
    samples = values[rng.integers(0, values.size, size=(BOOTSTRAP_REPLICATES, values.size))]
    boot_means = samples.mean(axis=1)
    return tuple(np.quantile(boot_means, [0.025, 0.975]))


def load_model_csv(model: str, name: str, **read_kwargs: object) -> tuple[Path, pd.DataFrame]:
    path = Path(MODEL_SPECS[model]["root"]) / name
    if not path.exists():
        raise FileNotFoundError(path)
    return path, pd.read_csv(path, **read_kwargs)


def load_per_protein_models() -> tuple[dict[str, pd.DataFrame], list[Path]]:
    frames: dict[str, pd.DataFrame] = {}
    paths: list[Path] = []
    test_ids: set[str] | None = None
    for model in MODEL_ORDER:
        path, frame = load_model_csv(model, "per_protein.csv", dtype={"protein_id": str})
        paths.append(path)
        require_columns(
            path,
            frame,
            {
                "layer",
                "protein_id",
                "source",
                "protein_split",
                "evaluation_scope",
                "n_residues",
                "r2",
                "mae",
            },
        )
        frame["protein_id"] = frame["protein_id"].str.zfill(9)
        frame["model"] = model
        require_layers(frame, f"{model} per-protein metrics")
        ids = set(frame.loc[frame["protein_split"].eq("test"), "protein_id"])
        if len(ids) != 111:
            raise ValueError(f"{model} has {len(ids)} test proteins; expected 111")
        if test_ids is None:
            test_ids = ids
        elif ids != test_ids:
            raise ValueError(f"{model} does not use the common locked test protein set")
        frames[model] = frame
    return frames, paths


def derive_mean_trajectories(
    frames: dict[str, pd.DataFrame], rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, frame in frames.items():
        scope = str(MODEL_SPECS[model]["main_scope"])
        subset = frame[
            frame["protein_split"].eq("test") & frame["evaluation_scope"].eq(scope)
        ].copy()
        for metric in ("r2", "mae"):
            for layer, group in subset.groupby("layer", sort=True):
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                ci_low, ci_high = bootstrap_mean_ci(values, rng)
                rows.append(
                    {
                        "model": model,
                        "layer": int(layer),
                        "metric": metric,
                        "mean": float(values.mean()),
                        "median": float(np.median(values)),
                        "ci_low": ci_low,
                        "ci_high": ci_high,
                        "n_proteins": int(values.size),
                        "protein_split": "test",
                        "evaluation_scope": scope,
                        "generalization_note": (
                            "unseen pairs within seen proteins"
                            if model == "Per-protein linear"
                            else "unseen proteins"
                        ),
                    }
                )
    result = pd.DataFrame(rows)
    require_layers(result, "mean per-protein trajectories")
    result.to_csv(COMPARING_DIR / "mean_per_protein_trajectories.csv", index=False)
    return result


def derive_layerwise_gains(
    frames: dict[str, pd.DataFrame], rng: np.random.Generator
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model, frame in frames.items():
        scope = str(MODEL_SPECS[model]["main_scope"])
        subset = frame[
            frame["protein_split"].eq("test") & frame["evaluation_scope"].eq(scope)
        ][["protein_id", "layer", "r2"]]
        pivot = subset.pivot(index="protein_id", columns="layer", values="r2")
        for to_layer in range(1, 48):
            paired = pivot[[to_layer - 1, to_layer]].dropna()
            gains = (paired[to_layer] - paired[to_layer - 1]).to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(gains, rng)
            rows.append(
                {
                    "model": model,
                    "from_layer": to_layer - 1,
                    "to_layer": to_layer,
                    "metric": "r2",
                    "value_from": float(paired[to_layer - 1].mean()),
                    "value_to": float(paired[to_layer].mean()),
                    "gain": float(gains.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_paired_proteins": int(len(paired)),
                    "evaluation_scope": scope,
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(COMPARING_DIR / "layerwise_gains.csv", index=False)
    return result


TRIANGLE_COLUMNS = (
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
)


def load_geometry_models() -> tuple[dict[str, pd.DataFrame], list[Path]]:
    frames: dict[str, pd.DataFrame] = {}
    paths: list[Path] = []
    for model in MODEL_ORDER:
        path, frame = load_model_csv(model, "geometry_metrics.csv", dtype={"protein_id": str})
        paths.append(path)
        require_columns(
            path,
            frame,
            {
                "layer",
                "protein_id",
                "protein_split",
                "evaluation_scope",
                "geometry_status",
                "tm_score",
                "frac_top3_eigenvalues",
                "negative_eigenvalue_fraction",
                "stress",
                *TRIANGLE_COLUMNS,
            },
        )
        frame["protein_id"] = frame["protein_id"].str.zfill(9)
        frame["model"] = model
        subset = frame[
            frame["protein_split"].eq("test")
            & frame["evaluation_scope"].eq(MODEL_SPECS[model]["geometry_scope"])
            & frame["geometry_status"].eq("complete")
        ].copy()
        require_layers(subset, f"{model} test geometry")
        if subset["protein_id"].nunique() != 111:
            raise ValueError(f"{model} geometry does not contain 111 test proteins")
        denominator = pd.to_numeric(subset["triangle_violation_fraction"], errors="coerce")
        numerator = pd.to_numeric(subset["triangle_mean_violation"], errors="coerce")
        subset["conditional_triangle_violation_magnitude"] = np.where(
            denominator > 0, numerator / denominator, np.nan
        )
        frames[model] = subset
    return frames, paths


def derive_geometry_trajectories(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    metrics = (
        "tm_score",
        "frac_top3_eigenvalues",
        "negative_eigenvalue_fraction",
        "stress",
        *TRIANGLE_COLUMNS,
        "conditional_triangle_violation_magnitude",
    )
    rows: list[dict[str, object]] = []
    for model, frame in frames.items():
        for metric in metrics:
            for layer, group in frame.groupby("layer", sort=True):
                values = pd.to_numeric(group[metric], errors="coerce").to_numpy(dtype=float)
                values = values[np.isfinite(values)]
                rows.append(
                    {
                        "model": model,
                        "layer": int(layer),
                        "geometry_metric": metric,
                        "mean": float(values.mean()),
                        "std": float(values.std(ddof=0)),
                        "median": float(np.median(values)),
                        "q25": float(np.quantile(values, 0.25)),
                        "q75": float(np.quantile(values, 0.75)),
                        "n_proteins": int(values.size),
                        "protein_split": "test",
                        "evaluation_scope": "all_pairs",
                    }
                )
    result = pd.DataFrame(rows)
    result.to_csv(COMPARING_DIR / "geometry_model_trajectories.csv", index=False)
    return result


def derive_decodability_consistency(
    per_protein: dict[str, pd.DataFrame], geometry: dict[str, pd.DataFrame]
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for model in MODEL_ORDER:
        accuracy_rows = per_protein[model][
            per_protein[model]["protein_split"].eq("test")
            & per_protein[model]["evaluation_scope"].eq("all_pairs")
        ][["protein_id", "layer", "r2"]]
        merged = accuracy_rows.merge(
            geometry[model][["protein_id", "layer", "triangle_violation_fraction"]],
            on=["protein_id", "layer"],
            validate="one_to_one",
        )
        for layer, group in merged.groupby("layer", sort=True):
            finite = group[
                np.isfinite(group["r2"]) & np.isfinite(group["triangle_violation_fraction"])
            ]
            rows.append(
                {
                    "model": model,
                    "layer": int(layer),
                    "accuracy_metric": "mean_per_protein_r2",
                    "accuracy_value": float(finite["r2"].mean()),
                    "consistency_metric": "triangle_violation_fraction",
                    "consistency_value": float(finite["triangle_violation_fraction"].mean()),
                    "protein_split": "test",
                    "evaluation_scope": "all_pairs",
                    "n_proteins": int(len(finite)),
                }
            )
    result = pd.DataFrame(rows)
    result.to_csv(GEOMETRY_DIR / "decodability_consistency.csv", index=False)
    return result


def load_subgroups(filename: str) -> tuple[pd.DataFrame, list[Path]]:
    frames: list[pd.DataFrame] = []
    paths: list[Path] = []
    for model in MODEL_ORDER:
        path, frame = load_model_csv(model, f"subgroup_metrics/{filename}")
        paths.append(path)
        require_columns(
            path,
            frame,
            {
                "layer",
                "protein_split",
                "evaluation_scope",
                "category_label",
                "n_proteins",
                "n_evaluated_pairs",
                "support_status",
                "pooled_r2",
                "pooled_mae",
                "pooled_rmse",
                "pooled_bias_pred_minus_true",
            },
        )
        filtered = frame[frame["protein_split"].eq("test")].copy()
        expected_scope = str(MODEL_SPECS[model]["main_scope"])
        if set(filtered["evaluation_scope"]) != {expected_scope}:
            raise ValueError(f"Unexpected {model} scope in {filename}")
        require_layers(filtered, f"{model} {filename}")
        filtered["model"] = model
        frames.append(filtered)
    return pd.concat(frames, ignore_index=True), paths


def canonical_subgroup_table(frame: pd.DataFrame, category_name: str) -> pd.DataFrame:
    result = frame.rename(
        columns={
            "category_label": category_name,
            "n_evaluated_pairs": "n_pairs",
            "pooled_bias_pred_minus_true": "pooled_bias",
        }
    ).copy()
    columns = [
        "model",
        "layer",
        category_name,
        "n_proteins",
        "n_pairs",
        "pooled_r2",
        "pooled_mae",
        "pooled_rmse",
        "pooled_bias",
        "protein_split",
        "evaluation_scope",
        "support_status",
    ]
    return result[columns]


def derive_aa_tables(aa_subgroups: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    trajectories = canonical_subgroup_table(aa_subgroups, "aa_pair")
    split = trajectories["aa_pair"].str.split("-", n=1, expand=True)
    trajectories.insert(3, "aa_1", split[0])
    trajectories.insert(4, "aa_2", split[1])
    if trajectories["aa_pair"].nunique() != 210:
        raise ValueError("Expected all 210 unordered amino-acid pairs")
    trajectories.to_csv(AA_DIR / "amino_acid_pair_trajectories.csv", index=False)

    primary = trajectories[trajectories["model"].eq("AA gated")].copy()
    timing_rows: list[dict[str, object]] = []
    for aa_pair, group in primary.groupby("aa_pair", sort=True):
        group = group.sort_values("layer")
        values = group.set_index("layer")["pooled_r2"].reindex(LAYERS).to_numpy(dtype=float)
        if not np.isfinite(values).all():
            raise ValueError(f"AA gated trajectory is incomplete for {aa_pair}")
        peak_r2 = float(values.max())
        peak_layer = int(np.flatnonzero(values == peak_r2)[0])
        total_gain = peak_r2 - float(values[0])
        if total_gain < TIMING_GAIN_EPS:
            l50 = math.nan
            l90 = math.nan
            status = "negligible_peak_gain"
        else:
            l50 = int(np.flatnonzero(values >= values[0] + 0.50 * total_gain)[0])
            l90 = int(np.flatnonzero(values >= values[0] + 0.90 * total_gain)[0])
            status = "timing_defined"
        timing_rows.append(
            {
                "model": "AA gated",
                "aa_pair": aa_pair,
                "r2_layer0": float(values[0]),
                "r2_layer47": float(values[-1]),
                "peak_r2": peak_r2,
                "peak_layer": peak_layer,
                "total_peak_gain": total_gain,
                "L50": l50,
                "L90": l90,
                "late_reversal": peak_r2 - float(values[-1]),
                "n_pairs_min": int(group["n_pairs"].min()),
                "n_proteins_min": int(group["n_proteins"].min()),
                "timing_status": status,
            }
        )
    timing = pd.DataFrame(timing_rows).sort_values(
        ["L50", "L90", "peak_layer", "aa_pair"], na_position="last"
    )
    timing.to_csv(AA_DIR / "amino_acid_pair_refinement_timing.csv", index=False)

    shared = trajectories[trajectories["model"].eq("Shared linear")].copy()
    gated = trajectories[trajectories["model"].eq("AA gated")].copy()
    join_columns = ["layer", "aa_pair"]
    gain = shared.merge(
        gated,
        on=join_columns,
        suffixes=("_shared", "_aa_gated"),
        validate="one_to_one",
    )
    gain = pd.DataFrame(
        {
            "layer": gain["layer"],
            "aa_pair": gain["aa_pair"],
            "shared_pooled_r2": gain["pooled_r2_shared"],
            "aa_gated_pooled_r2": gain["pooled_r2_aa_gated"],
            "delta_r2": gain["pooled_r2_aa_gated"] - gain["pooled_r2_shared"],
            "shared_pooled_mae": gain["pooled_mae_shared"],
            "aa_gated_pooled_mae": gain["pooled_mae_aa_gated"],
            "mae_reduction": gain["pooled_mae_shared"] - gain["pooled_mae_aa_gated"],
            "n_pairs_shared": gain["n_pairs_shared"],
            "n_pairs_aa_gated": gain["n_pairs_aa_gated"],
            "support_status": np.where(
                gain["support_status_shared"].eq("sufficient")
                & gain["support_status_aa_gated"].eq("sufficient"),
                "sufficient",
                "limited",
            ),
        }
    )
    gain.to_csv(AA_DIR / "aa_gating_pairwise_gain.csv", index=False)
    return trajectories, timing, gain


def bh_adjust(p_values: pd.Series) -> pd.Series:
    values = pd.to_numeric(p_values, errors="coerce").to_numpy(dtype=float)
    adjusted = np.full(values.shape, np.nan)
    finite_idx = np.flatnonzero(np.isfinite(values))
    if finite_idx.size == 0:
        return pd.Series(adjusted, index=p_values.index)
    order = finite_idx[np.argsort(values[finite_idx])]
    ranked = values[order] * finite_idx.size / np.arange(1, finite_idx.size + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    adjusted[order] = np.minimum(ranked, 1.0)
    return pd.Series(adjusted, index=p_values.index)


def fit_categorical_length_model(
    frame: pd.DataFrame, category_column: str, level_order: tuple[str, ...]
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    data = frame[[category_column, "r2", "n_residues"]].dropna().copy()
    data = data[data[category_column].isin(level_order)]
    levels = [level for level in level_order if level in set(data[category_column])]
    reference = levels[0]
    log_length = np.log(data["n_residues"].to_numpy(dtype=float))
    design_parts = [np.ones(len(data))]
    design_names = ["intercept"]
    for level in levels[1:]:
        design_parts.append(data[category_column].eq(level).to_numpy(dtype=float))
        design_names.append(level)
    design_parts.append(log_length)
    design_names.append("log_length")
    design = np.column_stack(design_parts)
    response = data["r2"].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank != design.shape[1]:
        raise ValueError(f"Rank-deficient {category_column} model")
    residual = response - design @ coefficients
    degrees_freedom = len(response) - design.shape[1]
    sigma2 = float(residual @ residual / degrees_freedom)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    mean_log_length = float(log_length.mean())
    adjusted_rows: list[dict[str, object]] = []
    effect_rows: list[dict[str, object]] = []
    for level in levels:
        contrast = np.zeros(design.shape[1])
        contrast[0] = 1.0
        contrast[-1] = mean_log_length
        if level != reference:
            contrast[design_names.index(level)] = 1.0
        estimate = float(contrast @ coefficients)
        standard_error = float(np.sqrt(contrast @ covariance @ contrast))
        adjusted_rows.append(
            {
                category_column: level,
                "mean": estimate,
                "ci_low": estimate - 1.96 * standard_error,
                "ci_high": estimate + 1.96 * standard_error,
                "n_proteins": int(data[category_column].eq(level).sum()),
                "mean_length": float(
                    data.loc[data[category_column].eq(level), "n_residues"].mean()
                ),
            }
        )
        if level == reference:
            continue
        coefficient_index = design_names.index(level)
        effect = float(coefficients[coefficient_index])
        effect_se = float(np.sqrt(covariance[coefficient_index, coefficient_index]))
        t_value = effect / effect_se
        p_value = float(2 * stats.t.sf(abs(t_value), degrees_freedom))
        effect_rows.append(
            {
                "reference_category": reference,
                "category": level,
                "effect": effect,
                "standard_error": effect_se,
                "ci_low": effect - 1.96 * effect_se,
                "ci_high": effect + 1.96 * effect_se,
                "p_value": p_value,
                "adjustment_formula": f"r2 ~ C({category_column}) + log(n_residues)",
                "n_proteins": int(len(data)),
            }
        )
    return adjusted_rows, effect_rows


def derive_ecod_category_tables(
    per_protein_frame: pd.DataFrame, rng: np.random.Generator
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    metadata_path = PROJECT_ROOT / "ecod_protein_categories.csv"
    metadata = pd.read_csv(metadata_path, dtype={"prot": str})
    require_columns(
        metadata_path,
        metadata,
        {"prot", "category", "length", "primary_arch"},
    )
    metadata["protein_id"] = metadata["prot"].str.zfill(9)
    base = per_protein_frame[
        per_protein_frame["source"].eq("ecod")
        & per_protein_frame["evaluation_scope"].eq("heldout_pairs")
    ].merge(
        metadata[["protein_id", "category", "primary_arch"]],
        on="protein_id",
        validate="many_to_one",
    )
    if base["protein_id"].nunique() != 398:
        raise ValueError(f"Expected 398 usable ECOD proteins, found {base['protein_id'].nunique()}")

    broad_trajectory_rows: list[dict[str, object]] = []
    broad_effect_rows: list[dict[str, object]] = []
    architecture_trajectory_rows: list[dict[str, object]] = []
    architecture_effect_rows: list[dict[str, object]] = []
    architecture_order = tuple(sorted(metadata["primary_arch"].dropna().unique()))
    architecture_to_category = (
        metadata[["primary_arch", "category"]]
        .drop_duplicates()
        .set_index("primary_arch")["category"]
        .to_dict()
    )

    for layer, layer_frame in base.groupby("layer", sort=True):
        for category in ECOD_CATEGORIES:
            group = layer_frame[layer_frame["category"].eq(category)]
            values = group["r2"].to_numpy(dtype=float)
            ci_low, ci_high = bootstrap_mean_ci(values, rng)
            broad_trajectory_rows.append(
                {
                    "category": category,
                    "layer": int(layer),
                    "metric": "r2",
                    "adjustment": "raw",
                    "mean": float(values.mean()),
                    "ci_low": ci_low,
                    "ci_high": ci_high,
                    "n_proteins": int(len(group)),
                    "mean_length": float(group["n_residues"].mean()),
                    "evaluation_scope": "heldout_pairs",
                }
            )
        adjusted, effects = fit_categorical_length_model(
            layer_frame, "category", ECOD_CATEGORIES
        )
        for row in adjusted:
            row.update(
                {
                    "layer": int(layer),
                    "metric": "r2",
                    "adjustment": "log_length_adjusted",
                    "evaluation_scope": "heldout_pairs",
                }
            )
            broad_trajectory_rows.append(row)
        for row in effects:
            row.update({"layer": int(layer), "metric": "r2"})
            broad_effect_rows.append(row)

        adjusted_arch, effects_arch = fit_categorical_length_model(
            layer_frame, "primary_arch", architecture_order
        )
        for row in adjusted_arch:
            row.update(
                {
                    "category": architecture_to_category[row["primary_arch"]],
                    "layer": int(layer),
                    "metric": "r2",
                    "adjustment": "log_length_adjusted",
                    "evaluation_scope": "heldout_pairs",
                }
            )
            architecture_trajectory_rows.append(row)
        for row in effects_arch:
            row.update({"layer": int(layer), "metric": "r2"})
            architecture_effect_rows.append(row)

    broad_trajectories = pd.DataFrame(broad_trajectory_rows)
    broad_effects = pd.DataFrame(broad_effect_rows)
    broad_effects["p_value_fdr_bh"] = bh_adjust(broad_effects["p_value"])
    broad_effects["multiple_test_method"] = "Benjamini-Hochberg across all layer contrasts"
    broad_trajectories.to_csv(REFINEMENT_DIR / "ecod_category_trajectories.csv", index=False)
    broad_effects.to_csv(REFINEMENT_DIR / "ecod_category_effects.csv", index=False)

    architecture_trajectories = pd.DataFrame(architecture_trajectory_rows)
    architecture_effects = pd.DataFrame(architecture_effect_rows)
    architecture_effects["p_value_fdr_bh"] = bh_adjust(architecture_effects["p_value"])
    architecture_effects["multiple_test_method"] = (
        "Benjamini-Hochberg across all layer contrasts"
    )
    architecture_trajectories.to_csv(
        REFINEMENT_DIR / "ecod_architecture_trajectories.csv", index=False
    )
    architecture_effects.to_csv(
        REFINEMENT_DIR / "ecod_architecture_effects.csv", index=False
    )
    return broad_trajectories, broad_effects, architecture_trajectories, architecture_effects


def timing_summary(values: np.ndarray, metric: str) -> dict[str, object]:
    values = np.asarray(values, dtype=float)
    if not np.isfinite(values).all():
        return {
            "value_layer0": values[0],
            "value_layer47": values[-1],
            "best_value": math.nan,
            "peak_layer": math.nan,
            "total_gain": math.nan,
            "L50": math.nan,
            "L90": math.nan,
            "late_reversal": math.nan,
            "timing_status": "incomplete_trajectory",
        }
    if metric == "r2":
        best = float(values.max())
        peak_layer = int(np.flatnonzero(values == best)[0])
        gain = best - float(values[0])
        reached = lambda fraction: np.flatnonzero(values >= values[0] + fraction * gain)
        late_reversal = best - float(values[-1])
    else:
        best = float(values.min())
        peak_layer = int(np.flatnonzero(values == best)[0])
        gain = float(values[0]) - best
        reached = lambda fraction: np.flatnonzero(values <= values[0] - fraction * gain)
        late_reversal = float(values[-1]) - best
    if gain < TIMING_GAIN_EPS:
        l50 = math.nan
        l90 = math.nan
        status = "negligible_gain"
    else:
        l50 = int(reached(0.50)[0])
        l90 = int(reached(0.90)[0])
        status = "timing_defined"
    return {
        "value_layer0": float(values[0]),
        "value_layer47": float(values[-1]),
        "best_value": best,
        "peak_layer": peak_layer,
        "total_gain": gain,
        "L50": l50,
        "L90": l90,
        "late_reversal": late_reversal,
        "timing_status": status,
    }


def fit_continuous_predictor(
    frame: pd.DataFrame,
    outcome: str,
    predictor: str,
    population: str,
    adjustment_columns: tuple[str, ...] = (),
) -> dict[str, object]:
    columns = [outcome, predictor, *adjustment_columns]
    data = frame[columns].dropna().copy()
    continuous = np.log(data[predictor].to_numpy(dtype=float))
    design_parts = [np.ones(len(data)), continuous]
    formula_terms = [f"log({predictor})"]
    for column in adjustment_columns:
        levels = sorted(data[column].unique())
        for level in levels[1:]:
            design_parts.append(data[column].eq(level).to_numpy(dtype=float))
        formula_terms.append(f"C({column})")
    design = np.column_stack(design_parts)
    response = data[outcome].to_numpy(dtype=float)
    coefficients, _, rank, _ = np.linalg.lstsq(design, response, rcond=None)
    if rank != design.shape[1]:
        raise ValueError(f"Rank-deficient predictor model for {outcome}")
    residual = response - design @ coefficients
    degrees_freedom = len(response) - design.shape[1]
    sigma2 = float(residual @ residual / degrees_freedom)
    covariance = sigma2 * np.linalg.inv(design.T @ design)
    standard_error = float(np.sqrt(covariance[1, 1]))
    coefficient = float(coefficients[1])
    p_value = float(2 * stats.t.sf(abs(coefficient / standard_error), degrees_freedom))
    return {
        "outcome": outcome,
        "predictor": f"log({predictor})",
        "population": population,
        "model": "ordinary_least_squares",
        "coefficient": coefficient,
        "standard_error": standard_error,
        "ci_low": coefficient - 1.96 * standard_error,
        "ci_high": coefficient + 1.96 * standard_error,
        "p_value": p_value,
        "adjustment_formula": f"{outcome} ~ " + " + ".join(formula_terms),
        "n_proteins": int(len(data)),
    }


def derive_protein_timing(per_protein_frame: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    metadata = pd.read_csv(PROJECT_ROOT / "ecod_protein_categories.csv", dtype={"prot": str})
    metadata["protein_id"] = metadata["prot"].str.zfill(9)
    subset = per_protein_frame[per_protein_frame["evaluation_scope"].eq("heldout_pairs")].copy()
    subset = subset.merge(
        metadata[["protein_id", "category", "primary_arch"]].rename(
            columns={"category": "ecod_category"}
        ),
        on="protein_id",
        how="left",
        validate="many_to_one",
    )
    rows: list[dict[str, object]] = []
    for protein_id, group in subset.groupby("protein_id", sort=True):
        group = group.sort_values("layer")
        if tuple(group["layer"].astype(int)) != LAYERS:
            raise ValueError(f"Incomplete held-out trajectory for {protein_id}")
        first = group.iloc[0]
        for metric in ("r2", "mae"):
            timing = timing_summary(group[metric].to_numpy(dtype=float), metric)
            rows.append(
                {
                    "protein_id": protein_id,
                    "source": first["source"],
                    "n_residues": int(first["n_residues"]),
                    "model": "Per-protein linear",
                    "evaluation_scope": "heldout_pairs",
                    "metric": metric,
                    **timing,
                    "ecod_category": first["ecod_category"],
                    "primary_arch": first["primary_arch"],
                }
            )
    timing_frame = pd.DataFrame(rows)
    timing_frame.to_csv(REFINEMENT_DIR / "protein_refinement_timing.csv", index=False)

    predictor_rows: list[dict[str, object]] = []
    for metric in ("r2", "mae"):
        metric_timing = timing_frame[
            timing_frame["metric"].eq(metric)
            & timing_frame["timing_status"].eq("timing_defined")
        ]
        for outcome in ("L50", "L90", "peak_layer", "value_layer47", "total_gain"):
            unadjusted = fit_continuous_predictor(
                metric_timing, outcome, "n_residues", "all usable proteins"
            )
            unadjusted.update({"outcome": f"{metric}_{outcome}", "metric": metric})
            predictor_rows.append(unadjusted)
            ecod = metric_timing[metric_timing["source"].eq("ecod")]
            adjusted = fit_continuous_predictor(
                ecod,
                outcome,
                "n_residues",
                "ECOD only",
                adjustment_columns=("ecod_category",),
            )
            adjusted.update({"outcome": f"{metric}_{outcome}", "metric": metric})
            predictor_rows.append(adjusted)
    predictors = pd.DataFrame(predictor_rows)
    predictors.to_csv(REFINEMENT_DIR / "protein_refinement_predictors.csv", index=False)
    return timing_frame, predictors


def plot_figures_1_2(mean_trajectories: pd.DataFrame) -> list[tuple[str, Path, Path]]:
    outputs = []
    specifications = (
        ("r2", "Mean per-protein $R^2$", (0, 1), "main_figure_01_mean_per_protein_r2"),
        ("mae", "Mean per-protein MAE (Å)", None, "main_figure_02_mean_per_protein_mae"),
    )
    for metric, ylabel, ylim, stem in specifications:
        figure, axis = plt.subplots(figsize=(8.4, 5.3))
        data = mean_trajectories[mean_trajectories["metric"].eq(metric)]
        plot_model_lines(axis, data, "mean", ci=True)
        axis.set(xlabel="Evoformer layer", ylabel=ylabel, xlim=(0, 47), ylim=ylim)
        if metric == "mae":
            axis.set_ylim(bottom=0)
        axis.set_xticks([0, 10, 20, 30, 40, 47])
        model_legend(axis, loc="best")
        figure.tight_layout()
        png, pdf = save_figure(figure, stem)
        outputs.append((stem, png, pdf))
    return outputs


def plot_figures_3_4(geometry: pd.DataFrame) -> list[tuple[str, Path, Path]]:
    outputs = []

    figure, axis = plt.subplots(figsize=(8.4, 5.3))
    data = geometry[geometry["geometry_metric"].eq("tm_score")]
    plot_model_lines(axis, data, "mean")
    axis.set(
        xlabel="Evoformer layer",
        ylabel="Mean reconstruction TM-score",
        xlim=(0, 47),
        ylim=(0, 1),
    )
    axis.set_xticks([0, 10, 20, 30, 40, 47])
    model_legend(axis, loc="best")
    figure.tight_layout()
    stem = "main_figure_03_tm_score"
    png, pdf = save_figure(figure, stem)
    outputs.append((stem, png, pdf))

    outputs.append(plot_mds_diagnostics(geometry))
    return outputs


def plot_mds_diagnostics(
    geometry: pd.DataFrame,
) -> tuple[str, Path, Path]:
    panels = (
        (
            "frac_top3_eigenvalues",
            "Top-three positive eigenvalue mass",
            "Fraction of positive eigenvalue mass",
            (0, 1),
        ),
        (
            "negative_eigenvalue_fraction",
            "Negative eigenvalue mass",
            "|Negative mass| / positive mass",
            None,
        ),
        (
            "stress",
            "Three-dimensional reconstruction stress",
            "Normalized distance discrepancy",
            None,
        ),
    )
    figure, axes = plt.subplots(1, 3, figsize=(18.0, 5.2), sharex=True)
    for panel_label, (axis, (metric, title, ylabel, ylim)) in zip(
        ("A", "B", "C"), zip(axes, panels)
    ):
        data = geometry[geometry["geometry_metric"].eq(metric)]
        plot_model_lines(axis, data, "mean")
        axis.set(
            title=f"{panel_label}  {title}",
            xlabel="Evoformer layer",
            ylabel=ylabel,
            xlim=(0, 47),
        )
        axis.set_xticks([0, 10, 20, 30, 40, 47])
        if ylim is None:
            axis.set_ylim(bottom=0)
        else:
            axis.set_ylim(*ylim)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.005),
        ncol=len(MODEL_ORDER),
        frameon=False,
    )
    figure.suptitle(
        "Classical-MDS diagnostics across layers "
        "(mean across 111 locked test proteins)",
        y=0.98,
    )
    figure.tight_layout(rect=(0, 0.12, 1, 0.93))
    stem = "main_figure_04_top3_eigenvalue_fraction"
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


def plot_triangle_diagnostics(geometry: pd.DataFrame) -> tuple[str, Path, Path]:
    panels = (
        ("triangle_violation_fraction", "All violations", "Fraction of triangles"),
        (
            "conditional_triangle_violation_magnitude",
            "Conditional mean magnitude",
            "Magnitude (Å)",
        ),
        ("triangle_max_violation", "Maximum magnitude", "Magnitude (Å)"),
        ("triangle_violation_0_5_fraction", "Magnitude 0–5 Å", "Fraction of triangles"),
        ("triangle_violation_5_10_fraction", "Magnitude 5–10 Å", "Fraction of triangles"),
        ("triangle_violation_10_plus_fraction", "Magnitude 10+ Å", "Fraction of triangles"),
    )
    figure, axes = plt.subplots(2, 3, figsize=(15.5, 8.5), sharex=True)
    for axis, (metric, title, ylabel) in zip(axes.flat, panels):
        data = geometry[geometry["geometry_metric"].eq(metric)]
        plot_model_lines(axis, data, "mean", linewidth=1.8)
        axis.set(title=title, xlabel="Evoformer layer", ylabel=ylabel, xlim=(0, 47))
        axis.set_xticks([0, 10, 20, 30, 40, 47])
        axis.set_ylim(bottom=0)
    model_legend(axes[0, 0], loc="upper right")
    figure.suptitle("Triangle-inequality violations across layers", y=0.995)
    figure.tight_layout()
    stem = "main_figure_05_triangle_violation_diagnostics"
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


def plot_layerwise_gain(gains: pd.DataFrame) -> tuple[str, Path, Path]:
    figure, axis = plt.subplots(figsize=(9.2, 5.4))
    plot_data = gains.rename(columns={"to_layer": "layer"})
    plot_model_lines(axis, plot_data, "gain", ci=True)
    axis.axhline(0, color="black", linewidth=1, alpha=0.7)
    axis.set(
        xlabel="Destination layer",
        ylabel="Mean per-protein $R^2$ gain",
        xlim=(1, 47),
    )
    axis.set_xticks([1, 10, 20, 30, 40, 47])
    model_legend(axis, loc="best")
    figure.tight_layout()
    stem = "main_figure_07_layerwise_r2_gain"
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


def plot_phase(decodability: pd.DataFrame) -> tuple[str, Path, Path]:
    figure, axis = plt.subplots(figsize=(8.2, 6.6))
    cmap = plt.get_cmap("viridis")
    selected_layers = {0, 10, 20, 30, 40, 47}
    for model in MODEL_ORDER:
        data = decodability[decodability["model"].eq(model)].sort_values("layer")
        axis.plot(
            data["accuracy_value"],
            data["consistency_value"],
            color=MODEL_SPECS[model]["color"],
            linewidth=1.2,
            alpha=0.7,
        )
        axis.scatter(
            data["accuracy_value"],
            data["consistency_value"],
            c=data["layer"],
            cmap=cmap,
            vmin=0,
            vmax=47,
            s=20,
            edgecolors="none",
        )
        for row in data[data["layer"].isin(selected_layers)].itertuples():
            axis.annotate(
                str(row.layer),
                (row.accuracy_value, row.consistency_value),
                xytext=(3, 3),
                textcoords="offset points",
                fontsize=6.5,
                color=MODEL_SPECS[model]["color"],
            )
    colorbar = figure.colorbar(
        plt.cm.ScalarMappable(norm=plt.Normalize(0, 47), cmap=cmap), ax=axis, pad=0.02
    )
    colorbar.set_label("Evoformer layer")
    axis.set(
        xlabel="Mean per-protein all-pairs $R^2$",
        ylabel="Mean triangle violation fraction",
    )
    model_legend(axis, loc="best")
    figure.tight_layout()
    stem = "main_figure_08_decodability_consistency"
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


def plot_faceted_subgroups(
    frame: pd.DataFrame,
    category_column: str,
    categories: tuple[str, ...],
    ylabel: str,
    stem: str,
    ncols: int = 3,
) -> tuple[str, Path, Path]:
    nrows = math.ceil(len(categories) / ncols)
    figure, axes = plt.subplots(
        nrows, ncols, figsize=(5.0 * ncols, 3.8 * nrows), sharex=True, sharey=False
    )
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, category in zip(flat_axes, categories):
        data = frame[frame[category_column].astype(str).eq(category)]
        plot_model_lines(axis, data, "pooled_mae", linewidth=1.7)
        axis.set(title=category, xlabel="Evoformer layer", ylabel=ylabel, xlim=(0, 47))
        axis.set_xticks([0, 10, 20, 30, 40, 47])
        axis.set_ylim(bottom=0)
    for axis in flat_axes[len(categories) :]:
        axis.set_visible(False)
    model_legend(flat_axes[0], loc="best")
    figure.tight_layout()
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


AA_HIGHLIGHTS = {
    "CYS-CYS": "#D62728",
    "GLY-GLY": "#1F77B4",
    "PRO-PRO": "#9467BD",
    "ASP-LYS": "#FF7F0E",
    "PHE-TYR": "#2CA02C",
}


def plot_all_aa_trajectories(
    trajectories: pd.DataFrame, metric: str, ylabel: str, stem: str, ylim: tuple[float, float] | None
) -> tuple[str, Path, Path]:
    data = trajectories[trajectories["model"].eq("AA gated")]
    figure, axis = plt.subplots(figsize=(10.5, 6.3))
    for _, group in data.groupby("aa_pair", sort=True):
        group = group.sort_values("layer")
        axis.plot(group["layer"], group[metric], color="#9E9E9E", alpha=0.12, linewidth=0.65)
    for aa_pair, color in AA_HIGHLIGHTS.items():
        group = data[data["aa_pair"].eq(aa_pair)].sort_values("layer")
        axis.plot(group["layer"], group[metric], color=color, linewidth=2.2, label=aa_pair)
    median = data.groupby("layer", sort=True)[metric].median()
    axis.plot(median.index, median.values, color="black", linewidth=2.2, linestyle="--", label="Pair median")
    axis.set(xlabel="Evoformer layer", ylabel=ylabel, xlim=(0, 47), ylim=ylim)
    if ylim is None:
        axis.set_ylim(bottom=0)
    axis.set_xticks([0, 10, 20, 30, 40, 47])
    axis.legend(loc="best", ncols=2)
    figure.tight_layout()
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


def plot_aa_timing(timing: pd.DataFrame) -> tuple[str, Path, Path]:
    ordered = timing.reset_index(drop=True).copy()
    ordered["rank"] = np.arange(1, len(ordered) + 1)
    defined = ordered[ordered["timing_status"].eq("timing_defined")]
    negligible = ordered[~ordered["timing_status"].eq("timing_defined")]
    figure, axis = plt.subplots(figsize=(9.8, 7.2))
    axis.scatter(defined["L50"], defined["rank"], s=13, color="#4C78A8", label="$L_{50}$")
    axis.scatter(defined["L90"], defined["rank"], s=13, color="#F58518", label="$L_{90}$")
    axis.scatter(
        defined["peak_layer"], defined["rank"], s=11, color="#54A24B", label="Peak layer"
    )
    if not negligible.empty:
        axis.scatter(
            negligible["peak_layer"],
            negligible["rank"],
            marker="x",
            s=24,
            color="#777777",
            label=f"Peak gain < {TIMING_GAIN_EPS:.2f}",
        )
    axis.set(
        xlabel="Layer",
        ylabel="AA-pair rank (sorted by $L_{50}$)",
        xlim=(-0.5, 47.5),
        ylim=(0, 211),
    )
    axis.set_xticks([0, 10, 20, 30, 40, 47])
    axis.legend(loc="best")
    figure.tight_layout()
    stem = "main_figure_13_aa_pair_refinement_timing"
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


def plot_aa_gating_gain(gain: pd.DataFrame, timing: pd.DataFrame) -> tuple[str, Path, Path]:
    order = timing["aa_pair"].tolist()
    matrix = gain.pivot(index="aa_pair", columns="layer", values="mae_reduction").reindex(order)
    limit = float(np.nanquantile(np.abs(matrix.to_numpy(dtype=float)), 0.99))
    figure, axis = plt.subplots(figsize=(12.2, 8.2))
    image = axis.imshow(
        matrix.to_numpy(dtype=float),
        aspect="auto",
        interpolation="none",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0, vmax=limit),
        origin="upper",
    )
    axis.set(
        xlabel="Evoformer layer",
        ylabel="AA pairs, sorted by refinement timing",
        xticks=[0, 10, 20, 30, 40, 47],
        yticks=[],
    )
    colorbar = figure.colorbar(image, ax=axis, pad=0.015)
    colorbar.set_label("Shared MAE − AA-gated MAE (Å)")
    figure.tight_layout()
    stem = "main_figure_14_aa_gating_pairwise_gain"
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


def plot_ecod_categories(
    trajectories: pd.DataFrame, effects: pd.DataFrame
) -> tuple[str, Path, Path]:
    colors = dict(zip(ECOD_CATEGORIES, ("#4C78A8", "#E45756", "#54A24B", "#F58518")))
    figure, axes = plt.subplots(2, 1, figsize=(9.5, 8.8), sharex=True)
    raw = trajectories[trajectories["adjustment"].eq("raw")]
    for category in ECOD_CATEGORIES:
        data = raw[raw["category"].eq(category)].sort_values("layer")
        axes[0].plot(data["layer"], data["mean"], color=colors[category], linewidth=2, label=CATEGORY_LABELS[category])
        axes[0].fill_between(
            data["layer"].to_numpy(dtype=float),
            data["ci_low"].to_numpy(dtype=float),
            data["ci_high"].to_numpy(dtype=float),
            color=colors[category],
            alpha=0.12,
            linewidth=0,
        )
    axes[0].set(ylabel="Mean held-out $R^2$", ylim=(0, 1), title="Raw category means")
    axes[0].legend(loc="best", ncols=2)
    for category in ECOD_CATEGORIES[1:]:
        data = effects[effects["category"].eq(category)].sort_values("layer")
        axes[1].plot(data["layer"], data["effect"], color=colors[category], linewidth=2, label=f"{CATEGORY_LABELS[category]} − All-alpha")
        axes[1].fill_between(
            data["layer"].to_numpy(dtype=float),
            data["ci_low"].to_numpy(dtype=float),
            data["ci_high"].to_numpy(dtype=float),
            color=colors[category],
            alpha=0.12,
            linewidth=0,
        )
    axes[1].axhline(0, color="black", linewidth=1)
    axes[1].set(
        xlabel="Evoformer layer",
        ylabel="Length-adjusted $R^2$ effect",
        title="Category contrasts adjusted for log residue count",
        xlim=(0, 47),
    )
    axes[1].set_xticks([0, 10, 20, 30, 40, 47])
    axes[1].legend(loc="best", ncols=2)
    figure.tight_layout()
    stem = "main_figure_16_ecod_category_refinement"
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


def pretty_architecture(name: str) -> str:
    if name.startswith("a_b_"):
        return "α/β " + name.removeprefix("a_b_").replace("_", " ")
    if name.startswith("alpha_"):
        return "α " + name.removeprefix("alpha_").replace("_", " ")
    if name.startswith("beta_"):
        return "β " + name.removeprefix("beta_").replace("_", " ")
    return name.replace("_", " ")


def plot_ecod_architectures(trajectories: pd.DataFrame) -> tuple[str, Path, Path]:
    architectures = sorted(trajectories["primary_arch"].unique())
    cmap = plt.get_cmap("tab20")
    colors = {architecture: cmap(index) for index, architecture in enumerate(architectures)}
    figure, axes = plt.subplots(2, 2, figsize=(13.0, 8.5), sharex=True, sharey=True)
    for axis, category in zip(axes.flat, ECOD_CATEGORIES):
        category_data = trajectories[trajectories["category"].eq(category)]
        for architecture in sorted(category_data["primary_arch"].unique()):
            data = category_data[category_data["primary_arch"].eq(architecture)].sort_values(
                "layer"
            )
            axis.plot(
                data["layer"],
                data["mean"],
                color=colors[architecture],
                linewidth=1.9,
                label=pretty_architecture(architecture),
            )
        axis.set(
            title=CATEGORY_LABELS[category],
            xlabel="Evoformer layer",
            ylabel="Length-adjusted mean held-out $R^2$",
            xlim=(0, 47),
            ylim=(0.6, 1.0),
        )
        axis.set_xticks([0, 10, 20, 30, 40, 47])
        axis.legend(loc="lower right", fontsize=7.5)
    figure.tight_layout()
    stem = "main_figure_17_ecod_architecture_refinement"
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


def plot_length_timing(timing: pd.DataFrame, predictors: pd.DataFrame) -> tuple[str, Path, Path]:
    data = timing[
        timing["metric"].eq("r2") & timing["timing_status"].eq("timing_defined")
    ].copy()
    colors = {"cameo": "#4C78A8", "ecod": "#F58518"}
    figure, axis = plt.subplots(figsize=(9.0, 6.3))
    for source in ("cameo", "ecod"):
        subset = data[data["source"].eq(source)]
        axis.scatter(
            subset["n_residues"],
            subset["L50"],
            s=18,
            alpha=0.48,
            color=colors[source],
            label=source.upper(),
            edgecolors="none",
        )
    x = np.linspace(data["n_residues"].min(), data["n_residues"].max(), 250)
    slope = float(
        predictors[
            predictors["outcome"].eq("r2_L50")
            & predictors["population"].eq("all usable proteins")
        ].iloc[0]["coefficient"]
    )
    intercept = float((data["L50"] - slope * np.log(data["n_residues"])).mean())
    axis.plot(x, intercept + slope * np.log(x), color="black", linewidth=2.2, label="OLS trend")
    axis.set(
        xlabel="Residue count",
        ylabel="$L_{50}$ for held-out per-protein $R^2$",
        ylim=(-1, 48),
    )
    axis.legend(loc="best")
    figure.tight_layout()
    stem = "main_figure_18_length_refinement_timing"
    png, pdf = save_figure(figure, stem)
    return stem, png, pdf


def manifest_row(
    figure_id: str,
    title: str,
    metric: str,
    sources: list[Path],
    plot_data: list[Path],
    png: Path,
    pdf: Path,
    scope: str,
    notes: str,
    generated_at: str,
) -> dict[str, object]:
    return {
        "figure_id": figure_id,
        "figure_set": "main",
        "title": title,
        "readiness": "generated",
        "evaluation_population": (
            "locked 111 test proteins" if figure_id not in {"16", "17", "18"} else "all usable proteins as specified"
        ),
        "evaluation_scope": scope,
        "metric": metric,
        "source_files": ";".join(relative(path) for path in sources),
        "plot_data_file": ";".join(relative(path) for path in plot_data),
        "png_file": relative(png),
        "pdf_file": relative(pdf),
        "script_file": relative(Path(__file__)),
        "generated_at": generated_at,
        "source_digest": combined_digest(sources),
        "notes": notes,
    }


def validate_outputs(manifest: pd.DataFrame) -> None:
    if manifest["figure_id"].duplicated().any():
        raise ValueError("Duplicate figure IDs in manifest")
    for column in ("png_file", "pdf_file", "plot_data_file"):
        for entry in manifest[column]:
            for item in str(entry).split(";"):
                path = PROJECT_ROOT / item
                if not path.exists() or path.stat().st_size == 0:
                    raise ValueError(f"Missing or empty manifest artifact: {item}")


def main() -> None:
    configure_style()
    ensure_directories()
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    generated_at = datetime.now(timezone.utc).isoformat()

    per_protein, per_protein_paths = load_per_protein_models()
    mean_trajectories = derive_mean_trajectories(per_protein, rng)
    gains = derive_layerwise_gains(per_protein, rng)
    geometry_frames, geometry_paths = load_geometry_models()
    geometry_trajectories = derive_geometry_trajectories(geometry_frames)
    decodability = derive_decodability_consistency(per_protein, geometry_frames)

    sequence_raw, sequence_paths = load_subgroups("by_sequence_separation.csv")
    sequence = canonical_subgroup_table(sequence_raw, "sequence_bin")
    sequence.to_csv(SEQUENCE_DIR / "sequence_separation_trajectories.csv", index=False)
    aa_raw, aa_paths = load_subgroups("by_amino_acid_pair.csv")
    aa_trajectories, aa_timing, aa_gain = derive_aa_tables(aa_raw)
    spatial_raw, spatial_paths = load_subgroups("by_spatial_distance.csv")
    spatial = canonical_subgroup_table(spatial_raw, "distance_bin")
    spatial.to_csv(COMPARING_DIR / "spatial_distance_trajectories.csv", index=False)

    per_protein_model = per_protein["Per-protein linear"]
    broad, broad_effects, architectures, architecture_effects = derive_ecod_category_tables(
        per_protein_model, rng
    )
    protein_timing, protein_predictors = derive_protein_timing(per_protein_model)

    figure_paths: dict[str, tuple[Path, Path]] = {}
    for stem, png, pdf in plot_figures_1_2(mean_trajectories):
        figure_paths[stem] = (png, pdf)
    for stem, png, pdf in plot_figures_3_4(geometry_trajectories):
        figure_paths[stem] = (png, pdf)
    for result in (
        plot_triangle_diagnostics(geometry_trajectories),
        plot_layerwise_gain(gains),
        plot_phase(decodability),
        plot_faceted_subgroups(
            sequence,
            "sequence_bin",
            SEQUENCE_BINS,
            "Pooled MAE (Å)",
            "main_figure_09_sequence_separation_mae",
        ),
        plot_all_aa_trajectories(
            aa_trajectories,
            "pooled_r2",
            "Pooled $R^2$",
            "main_figure_11_aa_pair_r2_210",
            (0, 1),
        ),
        plot_all_aa_trajectories(
            aa_trajectories,
            "pooled_mae",
            "Pooled MAE (Å)",
            "main_figure_12_aa_pair_mae_210",
            None,
        ),
        plot_aa_timing(aa_timing),
        plot_aa_gating_gain(aa_gain, aa_timing),
        plot_faceted_subgroups(
            spatial,
            "distance_bin",
            DISTANCE_BINS,
            "Pooled MAE (Å)",
            "main_figure_15_spatial_distance_mae",
        ),
        plot_ecod_categories(broad, broad_effects),
        plot_ecod_architectures(architectures),
        plot_length_timing(protein_timing, protein_predictors),
    ):
        stem, png, pdf = result
        figure_paths[stem] = (png, pdf)

    script_path = Path(__file__)
    mean_path = COMPARING_DIR / "mean_per_protein_trajectories.csv"
    geometry_path = COMPARING_DIR / "geometry_model_trajectories.csv"
    gain_path = COMPARING_DIR / "layerwise_gains.csv"
    decodability_path = GEOMETRY_DIR / "decodability_consistency.csv"
    sequence_path = SEQUENCE_DIR / "sequence_separation_trajectories.csv"
    aa_trajectory_path = AA_DIR / "amino_acid_pair_trajectories.csv"
    aa_timing_path = AA_DIR / "amino_acid_pair_refinement_timing.csv"
    aa_gain_path = AA_DIR / "aa_gating_pairwise_gain.csv"
    spatial_path = COMPARING_DIR / "spatial_distance_trajectories.csv"
    broad_path = REFINEMENT_DIR / "ecod_category_trajectories.csv"
    broad_effect_path = REFINEMENT_DIR / "ecod_category_effects.csv"
    architecture_path = REFINEMENT_DIR / "ecod_architecture_trajectories.csv"
    architecture_effect_path = REFINEMENT_DIR / "ecod_architecture_effects.csv"
    timing_path = REFINEMENT_DIR / "protein_refinement_timing.csv"
    predictor_path = REFINEMENT_DIR / "protein_refinement_predictors.csv"

    entries = [
        ("1", "Mean per-protein R2 across layers", "r2", per_protein_paths, [mean_path], "main_figure_01_mean_per_protein_r2", "model-specific primary scopes", "Per-protein linear uses held-out pairs; shared models use all pairs."),
        ("2", "Mean per-protein MAE across layers", "mae", per_protein_paths, [mean_path], "main_figure_02_mean_per_protein_mae", "model-specific primary scopes", "Per-protein linear uses held-out pairs; shared models use all pairs."),
        ("3", "Mean reconstruction TM-score across layers", "tm_score", geometry_paths, [geometry_path], "main_figure_03_tm_score", "all_pairs", "Reference is OpenFold's final predicted structure; per-protein complete matrices include fitting and held-out pairs."),
        ("4", "MDS eigenvalue fractions and 3D stress", "frac_top3_eigenvalues; negative_eigenvalue_fraction; stress", geometry_paths, [geometry_path], "main_figure_04_top3_eigenvalue_fraction", "all_pairs", "Top-three positive mass diagnoses low dimensionality, negative mass diagnoses non-Euclidean structure, and stress measures the normalized 3D distance discrepancy."),
        ("5-6", "Triangle-violation diagnostics", "fraction; conditional mean; maximum; severity-bin fractions", geometry_paths, [geometry_path], "main_figure_05_triangle_violation_diagnostics", "all_pairs", "Catalog Figures 5 and 6 consolidated into one six-panel figure at user request."),
        ("7", "Layer-to-layer R2 gain", "r2 gain", per_protein_paths, [gain_path], "main_figure_07_layerwise_r2_gain", "model-specific primary scopes", "Intervals bootstrap paired protein-level layer differences."),
        ("8", "Decodability versus geometric consistency", "all-pairs r2 vs triangle violation fraction", per_protein_paths + geometry_paths, [decodability_path], "main_figure_08_decodability_consistency", "all_pairs", "Per-protein all-pairs R2 mixes fitting and held-out pairs."),
        ("9", "Sequence-separation MAE trajectories", "pooled_mae", sequence_paths, [sequence_path], "main_figure_09_sequence_separation_mae", "model-specific primary scopes", "Six panels retain all 30 model-by-bin trajectories in one numbered figure."),
        ("11", "All 210 amino-acid-pair R2 trajectories", "pooled_r2", aa_paths, [aa_trajectory_path], "main_figure_11_aa_pair_r2_210", "all_pairs AA-gated", "Highlighted pairs were fixed before plotting; all 210 curves remain visible."),
        ("12", "All 210 amino-acid-pair MAE trajectories", "pooled_mae", aa_paths, [aa_trajectory_path], "main_figure_12_aa_pair_mae_210", "all_pairs AA-gated", "Highlighted pairs were fixed before plotting; all 210 curves remain visible."),
        ("13", "AA-pair refinement-stage distribution", "L50; L90; peak_layer", [aa_paths[-1]], [aa_timing_path], "main_figure_13_aa_pair_refinement_timing", "all_pairs AA-gated", f"Timing is undefined when total peak R2 gain is below {TIMING_GAIN_EPS:.2f}."),
        ("14", "Where AA gating helps over shared linear", "shared MAE - AA-gated MAE", [aa_paths[0], aa_paths[-1]], [aa_gain_path, aa_timing_path], "main_figure_14_aa_gating_pairwise_gain", "all_pairs", "Heatmap color limits use the 99th percentile of absolute values."),
        ("15", "Accuracy by true spatial-distance bin", "pooled_mae", spatial_paths, [spatial_path], "main_figure_15_spatial_distance_mae", "model-specific primary scopes", "Five panels retain all 25 model-by-bin trajectories in one numbered figure."),
        ("16", "Per-protein refinement by broad ECOD category", "r2", [per_protein_paths[1], PROJECT_ROOT / "ecod_protein_categories.csv"], [broad_path, broad_effect_path], "main_figure_16_ecod_category_refinement", "heldout_pairs", "Raw means and log-length-adjusted category contrasts; 398 usable ECOD proteins."),
        ("17", "Per-protein refinement by ECOD architecture", "length-adjusted r2", [per_protein_paths[1], PROJECT_ROOT / "ecod_protein_categories.csv"], [architecture_path, architecture_effect_path], "main_figure_17_ecod_architecture_refinement", "heldout_pairs", "Twelve predeclared ECOD architectures; adjusted at the population mean log length."),
        ("18", "Protein length and refinement timing", "L50", [per_protein_paths[1], PROJECT_ROOT / "ecod_protein_categories.csv"], [timing_path, predictor_path], "main_figure_18_length_refinement_timing", "heldout_pairs", "Source color is a technical diagnostic; associations use continuous log residue count."),
    ]
    manifest_rows = []
    for figure_id, title, metric, sources, plot_data, stem, scope, notes in entries:
        png, pdf = figure_paths[stem]
        manifest_rows.append(
            manifest_row(
                figure_id,
                title,
                metric,
                sources,
                plot_data,
                png,
                pdf,
                scope,
                notes,
                generated_at,
            )
        )
    manifest = pd.DataFrame(manifest_rows)
    validate_outputs(manifest)
    manifest.to_csv(PAPER_DIR / "figure_manifest.csv", index=False)

    summary = {
        "generated_at": generated_at,
        "figures_generated": int(len(manifest)),
        "catalog_items_covered": [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 11, 12, 13, 14, 15, 16, 17, 18
        ],
        "triangle_figure_consolidation": "Catalog items 5 and 6 are one six-panel figure.",
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "timing_gain_threshold": TIMING_GAIN_EPS,
        "manifest": relative(PAPER_DIR / "figure_manifest.csv"),
    }
    (PAPER_DIR / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(
        f"Generated {len(manifest)} numbered figure files covering 17 catalog items in "
        f"{relative(MAIN_DIR)}"
    )


if __name__ == "__main__":
    main()

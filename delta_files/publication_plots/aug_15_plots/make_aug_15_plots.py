#!/usr/bin/env python3
"""Create the August 15 first-draft publication plots.

The figures compare saved distance-probe and secondary-structure diagnostics
without refitting any model. Sparse diagnostics remain sparse: sequence-gated
and secondary-structure results use layers 0/10/20/30/40/47, while shared
XGBoost uses anchor layers 0/23/47.

Run from any working directory with::

    .venv/bin/python delta_files/publication_plots/aug_15_plots/make_aug_15_plots.py

Each figure is written as a 300-DPI PNG and a vector PDF. The exact plotted
values are also exported as companion CSV files in this directory.
"""

from __future__ import annotations

import os
from pathlib import Path


HERE = Path(__file__).resolve().parent
PROJECT_ROOT = Path(__file__).resolve().parents[3]
DELTA_DIR = PROJECT_ROOT / "delta_files"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


GENERAL_COLOR = "#4C78A8"
PER_PROTEIN_COLOR = "#E45756"
GATED_COLOR = "#F58518"
XGBOOST_COLOR = "#7A5195"
BASELINE_COLOR = "#6B6B6B"

GENERAL_GEOMETRY = DELTA_DIR / "general_base256_combined" / "geometry_summary.csv"
PER_PROTEIN_GEOMETRY = (
    DELTA_DIR / "per_protein_base256_combined" / "geometry_summary.csv"
)
WHOLE_PROTEIN_COMPARISON = (
    DELTA_DIR
    / "consolidated"
    / "gated_probe_visualizations"
    / "whole_protein_comparison_metrics.csv"
)
GATED_SUMMARY = (
    DELTA_DIR / "consolidated" / "analysis_1_sequence_gated" / "gate_summary.csv"
)
OLD_SEQUENCE_BASELINE = (
    DELTA_DIR
    / "diagnostics"
    / "diagnostic_sequence_separation_gated_probes"
    / "aggregate_layer_summary.csv"
)
GENERAL_SEQUENCE_ACCURACY = (
    DELTA_DIR
    / "diagnostics"
    / "diagnostic_secondary_structure_prediction"
    / "accuracy_by_sequence_separation.csv"
)
GENERAL_SECONDARY_SUMMARY = (
    DELTA_DIR
    / "diagnostics"
    / "diagnostic_secondary_structure_prediction"
    / "layer_summary.csv"
)
GATED_SECONDARY_SUMMARY = (
    DELTA_DIR
    / "diagnostics"
    / "diagnostic_secondary_structure_sequence_gated"
    / "layer_summary.csv"
)
GENERAL_HELIX_SCORES = (
    DELTA_DIR
    / "diagnostics"
    / "diagnostic_secondary_structure_prediction"
    / "helix_residue_scores.csv"
)
GATED_HELIX_SCORES = (
    DELTA_DIR
    / "diagnostics"
    / "diagnostic_secondary_structure_sequence_gated"
    / "helix_residue_scores.csv"
)
XGBOOST_SEARCH_ROOT = DELTA_DIR / "xgboost_hyperparameter_search" / "shared" / "search"

GATED_LAYERS = (0, 10, 20, 30, 40, 47)
XGBOOST_LAYERS = (0, 23, 47)
SHORT_RANGE_BINS = ("1", "2-4", "5-9")


def configure_style() -> None:
    """Apply a compact, colorblind-friendly publication style."""
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 14,
            "axes.labelsize": 12,
            "axes.linewidth": 0.9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "xtick.labelsize": 10,
            "ytick.labelsize": 10,
            "legend.fontsize": 9,
            "figure.titlesize": 15,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def load_csv(path: Path, required: set[str]) -> pd.DataFrame:
    """Load a CSV and fail with a useful message if its schema changed."""
    if not path.is_file():
        raise FileNotFoundError(f"Required input does not exist: {path}")
    frame = pd.read_csv(path)
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")
    return frame


def relative_source(path: Path) -> str:
    return str(path.relative_to(PROJECT_ROOT))


def style_layer_axis(axis: plt.Axes, ylabel: str, ylim: tuple[float, float]) -> None:
    # Keep endpoint markers fully visible at layers 0 and 47.
    axis.set_xlim(-0.6, 47.6)
    axis.set_ylim(*ylim)
    axis.set_xticks([0, 10, 20, 30, 40, 47])
    axis.set_xlabel("OpenFold pair-stack layer")
    axis.set_ylabel(ylabel)
    axis.grid(True, alpha=0.25, linewidth=0.7)
    axis.set_axisbelow(True)


def save_figure(figure: plt.Figure, stem: str) -> None:
    figure.savefig(HERE / f"{stem}.png", dpi=300, bbox_inches="tight")
    figure.savefig(HERE / f"{stem}.pdf", bbox_inches="tight")
    plt.close(figure)


def add_figure_note(figure: plt.Figure, text: str) -> None:
    figure.text(0.01, 0.012, text, ha="left", va="bottom", fontsize=8, color="#444444")


def geometry_series(
    path: Path,
    value_column: str,
    model: str,
    evaluation: str,
    split: str | None = None,
) -> pd.DataFrame:
    required = {"layer", "n_proteins", value_column}
    if split is not None:
        required.add("split")
    frame = load_csv(path, required)
    if split is not None:
        frame = frame[frame["split"] == split].copy()
    if len(frame) != 48 or frame["layer"].nunique() != 48:
        raise ValueError(f"Expected one row for each of 48 layers in {path}")
    return pd.DataFrame(
        {
            "model": model,
            "layer": frame["layer"].astype(int),
            "value": pd.to_numeric(frame[value_column], errors="raise"),
            "n_proteins": frame["n_proteins"].astype(int),
            "evaluation": evaluation,
            "source_file": relative_source(path),
        }
    ).sort_values("layer")


def build_r2_data() -> pd.DataFrame:
    general = geometry_series(
        GENERAL_GEOMETRY,
        "per_protein_r2_mean",
        "Shared OLS",
        "held-out proteins",
        split="test",
    )
    per_protein = geometry_series(
        PER_PROTEIN_GEOMETRY,
        "per_protein_r2_mean",
        "Per-protein OLS",
        "in-sample protein-specific fits",
    )

    gated_raw = load_csv(
        WHOLE_PROTEIN_COMPARISON,
        {"model", "layer", "mean_per_protein_r2", "n_test_proteins"},
    )
    gated_raw = gated_raw[gated_raw["model"] == "A1 sequence-gated"].copy()
    if tuple(sorted(gated_raw["layer"].astype(int))) != GATED_LAYERS:
        raise ValueError("Sequence-gated whole-protein output does not have the expected layers")
    gated = pd.DataFrame(
        {
            "model": "Sequence-gated OLS",
            "layer": gated_raw["layer"].astype(int),
            "value": pd.to_numeric(gated_raw["mean_per_protein_r2"], errors="raise"),
            "n_proteins": gated_raw["n_test_proteins"].astype(int),
            "evaluation": "held-out proteins; six anchor layers",
            "source_file": relative_source(WHOLE_PROTEIN_COMPARISON),
        }
    ).sort_values("layer")

    xgboost_rows: list[dict[str, object]] = []
    for layer in XGBOOST_LAYERS:
        path = XGBOOST_SEARCH_ROOT / f"layer_{layer:02d}" / "search_summary.csv"
        search = load_csv(
            path,
            {
                "config_id",
                "layer",
                "validation_mean_per_protein_r2",
                "test_mean_per_protein_r2",
                "n_test_proteins",
            },
        )
        validation = pd.to_numeric(
            search["validation_mean_per_protein_r2"], errors="raise"
        )
        selected = search.loc[validation.idxmax()]
        xgboost_rows.append(
            {
                "model": "Shared XGBoost",
                "layer": layer,
                "value": float(selected["test_mean_per_protein_r2"]),
                "n_proteins": int(selected["n_test_proteins"]),
                "evaluation": (
                    "held-out proteins; validation-selected; 128 symmetrized features; "
                    "different protein split"
                ),
                "source_file": relative_source(path),
                "selected_config": str(selected["config_id"]),
            }
        )
    xgboost = pd.DataFrame(xgboost_rows)

    baseline_raw = load_csv(
        OLD_SEQUENCE_BASELINE,
        {"model", "split", "mean_per_protein_r2", "n_proteins"},
    )
    baseline_raw = baseline_raw[
        (baseline_raw["model"] == "sequence_separation_only")
        & (baseline_raw["split"] == "test")
    ].copy()
    baseline_values = pd.to_numeric(
        baseline_raw["mean_per_protein_r2"], errors="raise"
    )
    if baseline_raw.empty or not np.allclose(baseline_values, baseline_values.iloc[0]):
        raise ValueError("Old sequence-only held-out baseline is missing or not layer-independent")
    baseline = pd.DataFrame(
        {
            "model": "Sequence separation only",
            "layer": np.arange(48, dtype=int),
            "value": float(baseline_values.iloc[0]),
            "n_proteins": int(baseline_raw["n_proteins"].iloc[0]),
            "evaluation": "held-out proteins; old 100-training-protein diagnostic",
            "source_file": relative_source(OLD_SEQUENCE_BASELINE),
        }
    )

    data = pd.concat([general, per_protein, gated, xgboost, baseline], ignore_index=True)
    data["metric"] = "mean_per_protein_r2"
    columns = [
        "metric",
        "model",
        "layer",
        "value",
        "n_proteins",
        "evaluation",
        "selected_config",
        "source_file",
    ]
    for column in columns:
        if column not in data:
            data[column] = ""
    return data[columns]


def plot_r2(data: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(9.4, 5.9))

    def rows(model: str) -> pd.DataFrame:
        return data[data["model"] == model].sort_values("layer")

    general = rows("Shared OLS")
    per_protein = rows("Per-protein OLS")
    gated = rows("Sequence-gated OLS")
    xgboost = rows("Shared XGBoost")
    baseline = rows("Sequence separation only")

    axis.plot(
        general["layer"], general["value"], color=GENERAL_COLOR, lw=2.6,
        label="Shared OLS — held-out",
    )
    axis.plot(
        per_protein["layer"], per_protein["value"], color=PER_PROTEIN_COLOR,
        lw=2.3, ls="--", label="Per-protein OLS — in-sample",
    )
    axis.plot(
        gated["layer"], gated["value"], color=GATED_COLOR, lw=1.8, ls=":",
        marker="o", ms=6, label="Sequence-gated OLS — held-out anchors",
    )
    axis.scatter(
        xgboost["layer"], xgboost["value"], color=XGBOOST_COLOR, marker="D",
        s=55, edgecolor="white", linewidth=0.8, zorder=5,
        label="Shared XGBoost — held-out anchors",
    )
    axis.axhline(
        float(baseline["value"].iloc[0]), color=BASELINE_COLOR, lw=1.7,
        ls="-.", label="Sequence separation only — old held-out baseline",
    )

    style_layer_axis(axis, r"Mean per-protein $R^2$", (0.0, 1.02))
    axis.set_title(r"Mean per-protein distance-decoding $R^2$ across layers")
    axis.legend(loc="lower right", frameon=True, framealpha=0.95)
    add_figure_note(
        figure,
        "Shared and sequence-gated OLS use the same 111 held-out proteins. "
        "Per-protein OLS is an in-sample upper bound. XGBoost uses a different split "
        "and 128 symmetrized features; markers are validation-selected test results.",
    )
    figure.tight_layout(rect=(0, 0.065, 1, 1))
    save_figure(figure, "mean_per_protein_r2_across_layers")


def build_geometry_plot_data(value_column: str, metric: str) -> pd.DataFrame:
    general = geometry_series(
        GENERAL_GEOMETRY,
        value_column,
        "Shared OLS",
        "held-out proteins",
        split="test",
    )
    per_protein = geometry_series(
        PER_PROTEIN_GEOMETRY,
        value_column,
        "Per-protein OLS",
        "in-sample protein-specific fits",
    )
    data = pd.concat([general, per_protein], ignore_index=True)
    data.insert(0, "metric", metric)
    return data


def plot_two_geometry_curves(
    data: pd.DataFrame,
    ylabel: str,
    title: str,
    stem: str,
    ylim: tuple[float, float],
) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.5))
    general = data[data["model"] == "Shared OLS"].sort_values("layer")
    per_protein = data[data["model"] == "Per-protein OLS"].sort_values("layer")
    axis.plot(
        general["layer"], general["value"], color=GENERAL_COLOR, lw=2.6,
        label="Shared OLS — held-out (n=111)",
    )
    axis.plot(
        per_protein["layer"], per_protein["value"], color=PER_PROTEIN_COLOR,
        lw=2.3, ls="--", label="Per-protein OLS — in-sample (n=552)",
    )
    style_layer_axis(axis, ylabel, ylim)
    axis.set_title(title)
    axis.legend(loc="lower right", frameon=True, framealpha=0.95)
    add_figure_note(
        figure,
        "Values are means over proteins. The per-protein curve is an in-sample, "
        "protein-specific upper bound rather than a held-out generalization estimate.",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(figure, stem)


def weighted_short_range_rows(
    frame: pd.DataFrame,
    model: str,
    bin_column: str,
    mae_column: str,
    source: Path,
) -> pd.DataFrame:
    frame = frame[frame[bin_column].astype(str).isin(SHORT_RANGE_BINS)].copy()
    frame["layer"] = pd.to_numeric(frame["layer"], errors="raise").astype(int)
    frame["n_pairs"] = pd.to_numeric(frame["n_pairs"], errors="raise").astype(int)
    frame[mae_column] = pd.to_numeric(frame[mae_column], errors="raise")
    rows: list[dict[str, object]] = []
    for layer, group in frame.groupby("layer", sort=True):
        observed_bins = set(group[bin_column].astype(str))
        if observed_bins != set(SHORT_RANGE_BINS):
            raise ValueError(f"Layer {layer} for {model} has short-range bins {observed_bins}")
        n_pairs = int(group["n_pairs"].sum())
        pooled_mae = float(np.average(group[mae_column], weights=group["n_pairs"]))
        rows.append(
            {
                "metric": "pooled_mae_sequence_separation_lt_10",
                "model": model,
                "layer": int(layer),
                "value_angstrom": pooled_mae,
                "n_pairs": n_pairs,
                "included_bins": ";".join(SHORT_RANGE_BINS),
                "evaluation": "held-out residue pairs",
                "source_file": relative_source(source),
            }
        )
    result = pd.DataFrame(rows)
    if tuple(result["layer"]) != GATED_LAYERS:
        raise ValueError(f"{model} short-range output does not have the expected layers")
    return result


def build_short_range_mae_data() -> pd.DataFrame:
    general_raw = load_csv(
        GENERAL_SEQUENCE_ACCURACY,
        {"layer", "split", "group", "distance_bin", "n_pairs", "mae"},
    )
    general_raw = general_raw[
        (general_raw["split"] == "test") & (general_raw["distance_bin"] == "all")
    ].copy()
    general = weighted_short_range_rows(
        general_raw,
        "Shared OLS",
        "group",
        "mae",
        GENERAL_SEQUENCE_ACCURACY,
    )

    gated_raw = load_csv(
        GATED_SUMMARY,
        {
            "layer",
            "split",
            "evaluation_scope",
            "fit_status",
            "train_gate_sequence_bin",
            "n_pairs",
            "pooled_mae",
        },
    )
    gated_raw = gated_raw[
        (gated_raw["split"] == "test")
        & (gated_raw["evaluation_scope"] == "gate_overall")
        & gated_raw["fit_status"].astype(str).str.startswith("fitted")
    ].copy()
    gated = weighted_short_range_rows(
        gated_raw,
        "Sequence-gated OLS",
        "train_gate_sequence_bin",
        "pooled_mae",
        GATED_SUMMARY,
    )
    if not np.array_equal(general["n_pairs"].to_numpy(), gated["n_pairs"].to_numpy()):
        raise ValueError("General and sequence-gated short-range pair counts do not match")
    return pd.concat([general, gated], ignore_index=True)


def plot_short_range_mae(data: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(8.6, 5.5))
    general = data[data["model"] == "Shared OLS"].sort_values("layer")
    gated = data[data["model"] == "Sequence-gated OLS"].sort_values("layer")
    axis.plot(
        general["layer"], general["value_angstrom"], color=GENERAL_COLOR,
        lw=2.2, ls=":", marker="o", ms=6, label="Shared OLS",
    )
    axis.plot(
        gated["layer"], gated["value_angstrom"], color=GATED_COLOR,
        lw=2.2, ls=":", marker="o", ms=6, label="Sequence-gated OLS",
    )
    style_layer_axis(axis, "Pooled MAE (Å)", (0.0, 4.05))
    axis.set_title("Short-range distance error across layers")
    axis.legend(loc="upper right", frameon=True, framealpha=0.95)
    add_figure_note(
        figure,
        "Residue pairs satisfy |i−j| < 10. MAE is recomputed exactly by summing "
        "absolute-error contributions from bins 1, 2–4, and 5–9 (210,897 test pairs).",
    )
    figure.tight_layout(rect=(0, 0.055, 1, 1))
    save_figure(figure, "sequence_separation_under10_mae_across_layers")


def helix_metrics(
    summary_path: Path,
    scores_path: Path,
    model: str,
) -> pd.DataFrame:
    summary = load_csv(
        summary_path,
        {"layer", "split", "n_helix_residues", "n_strand_residues", "helix_strand_auroc"},
    )
    summary = summary[summary["split"] == "test"].copy()
    summary["layer"] = pd.to_numeric(summary["layer"], errors="raise").astype(int)
    if tuple(summary.sort_values("layer")["layer"]) != GATED_LAYERS:
        raise ValueError(f"{model} secondary-structure summary has unexpected layers")

    scores = load_csv(scores_path, {"layer", "split", "dssp_code", "looks_helical"})
    scores = scores[scores["split"] == "test"].copy()
    observed_codes = set(scores["dssp_code"].astype(str))
    if observed_codes != {"H", "E"}:
        raise ValueError(f"Expected only DSSP H/E candidates in {scores_path}; got {observed_codes}")
    scores["layer"] = pd.to_numeric(scores["layer"], errors="raise").astype(int)
    scores["looks_helical"] = pd.to_numeric(
        scores["looks_helical"], errors="raise"
    ).astype(int)

    recall = (
        scores[scores["dssp_code"] == "H"]
        .groupby("layer", as_index=False)
        .agg(true_helix_recall=("looks_helical", "mean"), n_helix=("looks_helical", "size"))
    )
    specificity = (
        scores[scores["dssp_code"] == "E"]
        .groupby("layer", as_index=False)
        .agg(false_positive_rate=("looks_helical", "mean"), n_strand=("looks_helical", "size"))
    )
    specificity["strand_specificity"] = 1.0 - specificity["false_positive_rate"]
    merged = summary.merge(recall, on="layer", validate="one_to_one").merge(
        specificity[["layer", "n_strand", "strand_specificity"]],
        on="layer",
        validate="one_to_one",
    )
    if not np.array_equal(merged["n_helix_residues"], merged["n_helix"]):
        raise ValueError(f"Helix counts disagree between {summary_path} and {scores_path}")
    if not np.array_equal(merged["n_strand_residues"], merged["n_strand"]):
        raise ValueError(f"Strand counts disagree between {summary_path} and {scores_path}")
    return pd.DataFrame(
        {
            "model": model,
            "layer": merged["layer"].astype(int),
            "helix_strand_auroc": pd.to_numeric(
                merged["helix_strand_auroc"], errors="raise"
            ),
            "true_helix_recall": merged["true_helix_recall"].astype(float),
            "strand_specificity": merged["strand_specificity"].astype(float),
            "n_helix_residues": merged["n_helix"].astype(int),
            "n_strand_residues": merged["n_strand"].astype(int),
            "evaluation": "held-out DSSP H-versus-E candidates; frozen decision rule",
            "summary_source_file": relative_source(summary_path),
            "score_source_file": relative_source(scores_path),
        }
    ).sort_values("layer")


def build_helix_data() -> pd.DataFrame:
    general = helix_metrics(
        GENERAL_SECONDARY_SUMMARY,
        GENERAL_HELIX_SCORES,
        "Shared OLS",
    )
    gated = helix_metrics(
        GATED_SECONDARY_SUMMARY,
        GATED_HELIX_SCORES,
        "Sequence-gated OLS",
    )
    return pd.concat([general, gated], ignore_index=True)


def plot_helix_metrics(data: pd.DataFrame) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(12.4, 5.3), sharex=True)
    styles = {
        "Shared OLS": (GENERAL_COLOR, "Shared OLS"),
        "Sequence-gated OLS": (GATED_COLOR, "Sequence-gated OLS"),
    }
    for model, (color, label) in styles.items():
        selected = data[data["model"] == model].sort_values("layer")
        axes[0].plot(
            selected["layer"], selected["helix_strand_auroc"], color=color,
            lw=2.2, ls=":", marker="o", ms=6, label=label,
        )
        axes[1].plot(
            selected["layer"], selected["true_helix_recall"], color=color,
            lw=2.2, ls=":", marker="o", ms=6, label=label,
        )

    style_layer_axis(axes[0], "AUROC", (0.45, 1.01))
    style_layer_axis(axes[1], "True helix recall", (0.0, 1.01))
    axes[0].axhline(0.5, color="#999999", lw=1.0, ls="--", zorder=0)
    axes[0].set_title("Helix versus beta-strand discrimination")
    axes[1].set_title("True helix recall at the frozen threshold")
    axes[0].legend(loc="lower right", frameon=True, framealpha=0.95)
    axes[1].legend(loc="lower right", frameon=True, framealpha=0.95)
    add_figure_note(
        figure,
        "Held-out evaluation includes 6,541 DSSP H residues and 4,726 DSSP E residues. "
        "Recall is the fraction of H residues passing both frozen 'looks helical' gates; "
        "coil residues are not part of this diagnostic.",
    )
    figure.tight_layout(rect=(0, 0.075, 1, 1))
    save_figure(figure, "helix_classification_across_layers")


def write_outputs() -> None:
    configure_style()
    HERE.mkdir(parents=True, exist_ok=True)

    r2 = build_r2_data()
    r2.to_csv(HERE / "mean_per_protein_r2_plot_data.csv", index=False)
    plot_r2(r2)

    tm_score = build_geometry_plot_data("tm_score_mean", "mean_tm_score")
    tm_score.to_csv(HERE / "tm_score_plot_data.csv", index=False)
    plot_two_geometry_curves(
        tm_score,
        "Mean TM-score",
        "MDS reconstruction TM-score across layers",
        "tm_score_across_layers",
        (0.35, 0.97),
    )

    top3 = build_geometry_plot_data(
        "frac_top3_eigenvalues_mean",
        "mean_fraction_positive_mds_eigenvalue_mass_top3",
    )
    top3.to_csv(HERE / "top3_eigenvalue_fraction_plot_data.csv", index=False)
    plot_two_geometry_curves(
        top3,
        "Mean top-3 eigenvalue fraction",
        "Positive MDS eigenvalue mass captured by three dimensions",
        "top3_eigenvalue_fraction_across_layers",
        (0.30, 0.80),
    )

    short_range = build_short_range_mae_data()
    short_range.to_csv(
        HERE / "sequence_separation_under10_mae_plot_data.csv", index=False
    )
    plot_short_range_mae(short_range)

    helix = build_helix_data()
    helix.to_csv(HERE / "helix_classification_plot_data.csv", index=False)
    plot_helix_metrics(helix)

    print(f"Wrote five PNGs, five PDFs, and five plot-data CSVs to {HERE}")


if __name__ == "__main__":
    write_outputs()

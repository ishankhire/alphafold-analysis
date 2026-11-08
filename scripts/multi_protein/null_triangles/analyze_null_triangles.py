#!/usr/bin/env python3
"""Bootstraps, figures, tables, and sanity checks for the null-triangle experiment.

Reads the compact tables written by the cluster aggregation
(``scripts/cluster_scripts/diagnostics/null_triangles/aggregate_null_triangles.py``), copied to
``delta_files/diagnostic_experiments/null_triangles_cameo_v1/``. Every curve is a
protein-equal mean with a paired-protein bootstrap: one multinomial count matrix (2000
replicates, seed 20260902) is shared by every probe, layer, model, and metric. Gaps
compare a null with observed scored on the same triangle sample, protein by protein.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/multi_protein"))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from generate_sept_2_publication_plots import (  # noqa: E402
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, bootstrap_counts, bootstrap_matrix_summary,
)

TABLES = PROJECT_ROOT / "delta_files/diagnostic_experiments/null_triangles_cameo_v1"
FIGURES = PROJECT_ROOT / "delta_files/diagnostic_plots/null_triangles_cameo_v1"
LAYERS = np.arange(48)
TABLE_LAYERS = (0, 10, 21, 32, 40, 47)
CROSSTAB_LAYERS = (10, 21, 32, 47)
TROUGH, SECOND_PEAK = range(20, 23), range(30, 35)
OUTLIER = "7stz_C"
PROBES = ("shared_ols_base256", "per_protein_ols_base256", "sequence_gated_ols_base256",
          "shared_xgboost_base256_fixed_s4")
PROBE_LABELS = {"shared_ols_base256": "Shared OLS", "per_protein_ols_base256": "Per-protein OLS",
                "sequence_gated_ols_base256": "Seq-gated OLS", "shared_xgboost_base256_fixed_s4": "XGBoost"}
PROBE_COLORS = {"shared_ols_base256": "#2a6fb0", "per_protein_ols_base256": "#d9822b",
                "sequence_gated_ols_base256": "#3a9a5b", "shared_xgboost_base256_fixed_s4": "#9b4dca"}
PROBE_STYLES = {"shared_ols_base256": "-", "per_protein_ols_base256": "--",
                "sequence_gated_ols_base256": "-.", "shared_xgboost_base256_fixed_s4": ":"}
METRICS = {"mean_relative_violation_all": "Normalized mean violation", "violation_fraction": "Violation fraction"}
SERIES_COLORS = {"observed": "#222222", "null_a": "#d9822b", "null_b": "#2a6fb0", "null_c": "#3a9a5b"}
SERIES_LABELS = {"observed": "Observed", "null_a": "Null A (Gaussian)", "null_b": "Null B (permuted)",
                 "null_c": "Null C (residue offsets + permuted)"}


# ---------------------------------------------------------------- data

def load() -> dict:
    per = pd.read_csv(TABLES / "null_per_protein.csv", dtype={"protein_id": str})
    cohort = sorted(per["protein_id"].unique())
    return {
        "per": per, "cohort": cohort,
        "bins": pd.read_csv(TABLES / "error_diagnostics_by_bin.csv"),
        "crosstab": pd.read_csv(TABLES / "crosstab_b10_seqsep.csv"),
        "calibration": pd.read_csv(TABLES / "calibration_per_protein.csv", dtype={"protein_id": str}),
        "fallback": pd.read_csv(TABLES / "null_a_fallback_counts.csv"),
        "manifest": json.loads((TABLES / "run_manifest.json").read_text()),
    }


def matrix(per: pd.DataFrame, cohort: list[str], probe: str, model: str, binning: str,
           triangle_set: str, metric: str) -> np.ndarray:
    """Proteins x layers for one series."""
    rows = per[(per.probe == probe) & (per.model == model) & (per.binning == binning)
               & (per.triangle_set == triangle_set)]
    if rows.empty:
        raise ValueError(f"No rows for {probe} {model} {binning} {triangle_set}")
    return rows.pivot(index="protein_id", columns="layer", values=metric).reindex(
        index=cohort, columns=LAYERS).to_numpy(float)


def observed(data, probe, metric, triangle_set="exact"):
    return matrix(data["per"], data["cohort"], probe, "observed", "none", triangle_set, metric)


def null(data, probe, model, binning, metric):
    return matrix(data["per"], data["cohort"], probe, model, binning, "sample_1m", metric)


def gap(data, probe, model, binning, metric):
    """Observed minus null, both on the triangle sample, per protein."""
    return observed(data, probe, metric, "sample_1m") - null(data, probe, model, binning, metric)


def summary(values: np.ndarray, counts: np.ndarray) -> dict[str, np.ndarray]:
    return bootstrap_matrix_summary(values, counts)


def contiguous_ranges(layers: np.ndarray) -> str:
    if layers.size == 0:
        return "none"
    parts, start = [], layers[0]
    for prev, cur in zip(layers[:-1], layers[1:]):
        if cur != prev + 1:
            parts.append(f"L{start}" if start == prev else f"L{start}–L{prev}")
            start = cur
    parts.append(f"L{start}" if start == layers[-1] else f"L{start}–L{layers[-1]}")
    return ", ".join(parts)


def markdown_table(frame: pd.DataFrame) -> str:
    def cell(value):
        return f"{value:.4g}" if isinstance(value, float) else str(value)
    lines = ["| " + " | ".join(map(str, frame.columns)) + " |", "|" + "---|" * len(frame.columns)]
    lines += ["| " + " | ".join(cell(v) for v in row) + " |" for row in frame.itertuples(index=False)]
    return "\n".join(lines)


def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=200, bbox_inches="tight")
    plt.close(fig)


def band(ax, s, color, label, style="-"):
    ax.plot(LAYERS, s["mean"], style, color=color, lw=1.6, label=label)
    ax.fill_between(LAYERS, s["bootstrap_ci_low"], s["bootstrap_ci_high"], color=color, alpha=0.18, lw=0)


# ---------------------------------------------------------------- figures

def figure_observed_vs_nulls(data, counts) -> None:
    fig, axes = plt.subplots(2, 4, figsize=(18, 7.5), sharex=True)
    for col, probe in enumerate(PROBES):
        for row, (metric, label) in enumerate(METRICS.items()):
            ax = axes[row, col]
            band(ax, summary(observed(data, probe, metric), counts), SERIES_COLORS["observed"], SERIES_LABELS["observed"])
            for model in ("null_a", "null_b", "null_c"):
                band(ax, summary(null(data, probe, model, "B10", metric), counts), SERIES_COLORS[model], SERIES_LABELS[model])
            if row == 0:
                ax.set_title(PROBE_LABELS[probe])
            if col == 0:
                ax.set_ylabel(label)
            if row == 1:
                ax.set_xlabel("Layer")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Observed vs null models (B10), {len(data['cohort'])} CAMEO proteins, protein-equal mean, 95% paired bootstrap")
    save(fig, "observed_vs_nulls")


def figure_gap(data, counts, model: str, binning: str, name: str) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    for ax, (metric, label) in zip(axes, METRICS.items()):
        for probe in PROBES:
            s = summary(gap(data, probe, model, binning, metric), counts)
            band(ax, s, PROBE_COLORS[probe], PROBE_LABELS[probe])
            excludes = (s["bootstrap_ci_low"] > 0) | (s["bootstrap_ci_high"] < 0)
            ax.plot(LAYERS[excludes], s["mean"][excludes], "o", ms=3, color=PROBE_COLORS[probe])
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Observed − {SERIES_LABELS[model].split(' (')[0]}: {label.lower()}")
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle(f"Observed − {SERIES_LABELS[model]} on {binning}; dots: 95% band excludes 0")
    save(fig, name)


def figure_sensitivity(data, counts) -> None:
    probe = "shared_ols_base256"
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
    styles = {"B5": "--", "B10": "-", "B20": ":"}
    for ax, (metric, label) in zip(axes, METRICS.items()):
        band(ax, summary(observed(data, probe, metric, "sample_1m"), counts), SERIES_COLORS["observed"], "Observed (sample)")
        for binning, style in styles.items():
            band(ax, summary(null(data, probe, "null_b", binning, metric), counts), SERIES_COLORS["null_b"],
                 f"Null B, {binning}", style)
        ax.set_xlabel("Layer")
        ax.set_ylabel(label)
    axes[0].legend(fontsize=8, frameon=False)
    fig.suptitle("Shared OLS: Null B under B5 / B10 / B20 bins")
    save(fig, "sensitivity_binning")


def figure_bias_sd(data) -> None:
    bins = data["bins"]
    labels = ("[0,4)", "[4,6)", "[6,8)", "[8,10)")
    fig, axes = plt.subplots(1, 4, figsize=(18, 4.2), sharex=True)
    for ax, label in zip(axes, labels):
        for probe in PROBES:
            rows = bins[(bins.probe == probe) & (bins.cell_label == label)].set_index("layer").reindex(LAYERS)
            style = PROBE_STYLES[probe]
            ax.plot(LAYERS, rows["protein_mean_sd_sq"], style, color="#2a6fb0", lw=1.4)
            ax.plot(LAYERS, rows["protein_mean_mse"], style, color="#c0392b", lw=1.4)
        ax.set_title(f"d_true in {label} Å")
        ax.set_xlabel("Layer")
    axes[0].set_ylabel("Å²: lower line sd², upper line sd² + bias² (= MSE)")
    handles = [Line2D([], [], color="#2a6fb0", label="sd²"), Line2D([], [], color="#c0392b", label="sd² + bias²")]
    handles += [Line2D([], [], color="#444444", ls=PROBE_STYLES[p], label=PROBE_LABELS[p]) for p in PROBES]
    axes[0].legend(handles=handles, fontsize=8, frameon=False)
    fig.suptitle("Error decomposition by distance bin, protein-equal means (proteins with ≥10 pairs in the bin)")
    save(fig, "bias_sd_stacked")


def calibration_table(data, counts) -> pd.DataFrame:
    cal, records = data["calibration"], []
    for probe in PROBES:
        for column in ("beta", "alpha"):
            values = cal[cal.probe == probe].pivot(index="protein_id", columns="layer", values=column).reindex(
                index=data["cohort"], columns=LAYERS).to_numpy(float)
            s = summary(values, counts)
            for layer in LAYERS:
                records.append({"probe": probe, "layer": int(layer), "parameter": column, "protein_mean": s["mean"][layer],
                                "ci_low": s["bootstrap_ci_low"][layer], "ci_high": s["bootstrap_ci_high"][layer]})
    return pd.DataFrame(records)


def figure_calibration(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for probe in PROBES:
        rows = table[(table.probe == probe) & (table.parameter == "beta")].sort_values("layer")
        ax.plot(rows.layer, rows.protein_mean, color=PROBE_COLORS[probe], label=PROBE_LABELS[probe])
        ax.fill_between(rows.layer, rows.ci_low, rows.ci_high, color=PROBE_COLORS[probe], alpha=0.18, lw=0)
    ax.axhline(1, color="#888888", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel("β in d_pred = α + β·d_true (protein-equal mean)")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "calibration_beta")


def clipped_table(data, counts) -> pd.DataFrame:
    records = []
    for probe in PROBES:
        s = summary(observed(data, probe, "clipped_side_fraction"), counts)
        for layer in LAYERS:
            records.append({"probe": probe, "layer": int(layer), "protein_mean": s["mean"][layer],
                            "ci_low": s["bootstrap_ci_low"][layer], "ci_high": s["bootstrap_ci_high"][layer],
                            "n_proteins": int(s["n_proteins_finite"][layer])})
    return pd.DataFrame(records)


def figure_clipped(table: pd.DataFrame) -> None:
    fig, ax = plt.subplots(figsize=(7, 4.5))
    for probe in PROBES:
        rows = table[table.probe == probe].sort_values("layer")
        ax.plot(rows.layer, rows.protein_mean, color=PROBE_COLORS[probe], label=PROBE_LABELS[probe])
        ax.fill_between(rows.layer, rows.ci_low, rows.ci_high, color=PROBE_COLORS[probe], alpha=0.18, lw=0)
    ax.set_xlabel("Layer")
    ax.set_ylabel("Share of violating triples with a side clipped to 0")
    ax.legend(fontsize=8, frameon=False)
    save(fig, "clipped_side_fraction")


def figure_crosstab(data) -> None:
    table = data["crosstab"]
    distance_labels = list(dict.fromkeys(table.distance_bin))
    sequence_labels = list(dict.fromkeys(table.sequence_bin))
    fig, axes = plt.subplots(len(PROBES), len(CROSSTAB_LAYERS), figsize=(16, 16), sharex=True, sharey=True)
    finite = table[table.layer.isin(CROSSTAB_LAYERS)].protein_mean_sd
    vmax = float(np.nanpercentile(finite, 99))
    for r, probe in enumerate(PROBES):
        for c, layer in enumerate(CROSSTAB_LAYERS):
            rows = table[(table.probe == probe) & (table.layer == layer)]
            grid = rows.pivot(index="distance_bin", columns="sequence_bin", values="protein_mean_sd").reindex(
                index=distance_labels, columns=sequence_labels)
            ax = axes[r, c]
            image = ax.imshow(grid.to_numpy(float), cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            for (y, x), value in np.ndenumerate(grid.to_numpy(float)):
                if np.isfinite(value):
                    ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=6, color="white")
            ax.set_xticks(range(len(sequence_labels)), sequence_labels, fontsize=7)
            ax.set_yticks(range(len(distance_labels)), distance_labels, fontsize=7)
            if r == 0:
                ax.set_title(f"Layer {layer}")
            if c == 0:
                ax.set_ylabel(f"{PROBE_LABELS[probe]}\nd_true bin (Å)")
            if r == len(PROBES) - 1:
                ax.set_xlabel("Sequence separation")
    fig.colorbar(image, ax=axes, shrink=0.5, label="Protein-equal mean residual SD (Å)")
    fig.suptitle("Residual SD by distance × sequence separation (empty: < 100 pairs cohort-wide)")
    save(fig, "crosstab_sd_heatmap")


# ---------------------------------------------------------------- tables and text

def summary_table(data, counts) -> pd.DataFrame:
    records = []
    for probe in PROBES:
        for metric in METRICS:
            series = {"observed": observed(data, probe, metric)}
            for model in ("null_a", "null_b", "null_c"):
                series[model] = null(data, probe, model, "B10", metric)
            series["observed_minus_null_b"] = gap(data, probe, "null_b", "B10", metric)
            series["observed_minus_null_c"] = gap(data, probe, "null_c", "B10", metric)
            for name, values in series.items():
                s = summary(values, counts)
                for layer in TABLE_LAYERS:
                    records.append({"probe": probe, "metric": metric, "quantity": name, "layer": layer,
                                    "protein_mean": s["mean"][layer], "ci_low": s["bootstrap_ci_low"][layer],
                                    "ci_high": s["bootstrap_ci_high"][layer]})
    return pd.DataFrame(records)


def window_contrast(values: np.ndarray, counts: np.ndarray) -> dict[str, float]:
    """Paired bootstrap of mean(L30-34) - mean(L20-22) per protein."""
    per_protein = np.nanmean(values[:, list(SECOND_PEAK)], axis=1) - np.nanmean(values[:, list(TROUGH)], axis=1)
    s = summary(per_protein[:, None], counts)
    return {"mean": float(s["mean"][0]), "ci_low": float(s["bootstrap_ci_low"][0]),
            "ci_high": float(s["bootstrap_ci_high"][0])}


def results_markdown(data, counts) -> str:
    lines = ["# Null models for triangle violations: results", "",
             f"{len(data['cohort'])} CAMEO proteins, layers 0–47, protein-equal means, 95% paired-protein "
             f"bootstrap ({BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}). Gaps compare observed and "
             "null on the same triangle sample. Below / above: the 95% band of observed − null lies entirely "
             "below / above 0. Second-peak contrast: per-protein mean over L30–34 minus mean over L20–22, "
             "paired bootstrap; it survives when the interval lies above 0.", ""]
    for probe in PROBES:
        lines += [f"## {PROBE_LABELS[probe]}", ""]
        for metric, label in METRICS.items():
            lines.append(f"**{label}**")
            lines.append("")
            for model in ("null_b", "null_c"):
                values = gap(data, probe, model, "B10", metric)
                s = summary(values, counts)
                below = LAYERS[s["bootstrap_ci_high"] < 0]
                above = LAYERS[s["bootstrap_ci_low"] > 0]
                within = LAYERS[(s["bootstrap_ci_low"] <= 0) & (s["bootstrap_ci_high"] >= 0)]
                lines.append(f"- Observed vs {SERIES_LABELS[model]}: below in {contiguous_ranges(below)}; "
                             f"above in {contiguous_ranges(above)}; matches (band includes 0) in "
                             f"{contiguous_ranges(within)}.")
            obs = window_contrast(observed(data, probe, metric), counts)
            gap_b = window_contrast(gap(data, probe, "null_b", "B10", metric), counts)
            null_b = window_contrast(null(data, probe, "null_b", "B10", metric), counts)
            lines.append(f"- Second peak in observed: {obs['mean']:.4g} [{obs['ci_low']:.4g}, {obs['ci_high']:.4g}]; "
                         f"in Null B alone: {null_b['mean']:.4g} [{null_b['ci_low']:.4g}, {null_b['ci_high']:.4g}]; "
                         f"in observed − Null B: {gap_b['mean']:.4g} [{gap_b['ci_low']:.4g}, {gap_b['ci_high']:.4g}] "
                         f"({'survives' if gap_b['ci_low'] > 0 else 'does not survive'}).")
            lines.append("")
    return "\n".join(lines)


def sanity_markdown(data, counts) -> str:
    manifest, per = data["manifest"], data["per"]
    checks = manifest["checks"]
    lines = ["# Sanity checks", "",
             f"- Slim reducer vs `compute_metric_rows`, bit-identical: {checks['bitwise_compared']} comparisons, "
             f"{checks['bitwise_failures']} failures.",
             f"- Exact observed vs the paper's stored per-protein values: {checks['paper_parity_compared']} "
             f"comparisons, worst violation-fraction gap {checks['paper_parity_worst']['paper_fraction']:.3g}, worst "
             f"normalized-sum relative gap {checks['paper_parity_worst']['paper_relative_sum_rtol']:.3g}, worst "
             f"negative-count gap {checks['paper_parity_worst']['paper_negative']} (float32 cache vs float64 replay).",
             f"- Null B kept each bin's residual histogram in all {checks['histogram_checks']} draws "
             f"({checks['histogram_failures']} failures).", ""]
    # Null A vs Null B, XGBoost layer 47.
    probe = "shared_xgboost_base256_fixed_s4"
    lines.append("## Null A vs Null B, XGBoost, layer 47")
    for metric, label in METRICS.items():
        diff = null(data, probe, "null_a", "B10", metric) - null(data, probe, "null_b", "B10", metric)
        s = summary(diff, counts)
        lines.append(f"- {label}: A − B = {s['mean'][47]:.4g} [{s['bootstrap_ci_low'][47]:.4g}, "
                     f"{s['bootstrap_ci_high'][47]:.4g}] ({'agree' if s['bootstrap_ci_low'][47] <= 0 <= s['bootstrap_ci_high'][47] else 'differ'}).")
    shape = data["bins"][(data["bins"].probe == probe) & (data["bins"].layer == 47)][
        ["cell_label", "protein_mean_skew", "protein_mean_excess_kurtosis", "n_proteins_contributing"]]
    lines += ["", "Per-bin residual shape at layer 47 (protein-equal means):", "", markdown_table(shape), ""]
    # Monte Carlo noise.
    lines += ["## Monte Carlo noise (K draws per protein)", "",
              "| Model | Binning | Metric | Mean per-protein draw SD | Cohort-mean MC SE | Median bootstrap CI half-width |",
              "|---|---|---|---|---|---|"]
    for model, binning in (("null_a", "B10"), ("null_b", "B10"), ("null_c", "B10"), ("null_b", "B10xS6"), ("null_c", "B10xS6")):
        for metric, label in METRICS.items():
            rows = per[(per.model == model) & (per.binning == binning)]
            draw_sd = rows[f"{metric}_draw_sd"]
            k = int(rows.n_draws.iloc[0])
            se = float(np.sqrt(np.nanmean(draw_sd ** 2) / k / len(data["cohort"])))
            widths = [np.median((lambda s: (s["bootstrap_ci_high"] - s["bootstrap_ci_low"]) / 2)(
                summary(null(data, p, model, binning, metric), counts))) for p in PROBES]
            lines.append(f"| {model} | {binning} | {label} | {np.nanmean(draw_sd):.3g} | {se:.3g} | {np.median(widths):.3g} |")
    # 7stz_C robustness.
    keep = np.array([pid != OUTLIER for pid in data["cohort"]])
    counts_wo = bootstrap_counts(n_proteins=int(keep.sum()))
    lines += ["", f"## Robustness: observed − Null B (B10) without {OUTLIER}", "",
              "| Probe | Metric | Layer | All proteins | Without 7stz_C |", "|---|---|---|---|---|"]
    for probe in PROBES:
        for metric, label in METRICS.items():
            values = gap(data, probe, "null_b", "B10", metric)
            s_all, s_wo = summary(values, counts), summary(values[keep], counts_wo)
            for layer in TABLE_LAYERS:
                lines.append(f"| {PROBE_LABELS[probe]} | {label} | {layer} | "
                             f"{s_all['mean'][layer]:.4g} [{s_all['bootstrap_ci_low'][layer]:.4g}, {s_all['bootstrap_ci_high'][layer]:.4g}] | "
                             f"{s_wo['mean'][layer]:.4g} [{s_wo['bootstrap_ci_low'][layer]:.4g}, {s_wo['bootstrap_ci_high'][layer]:.4g}] |")
    # Null A fallback.
    fallback = data["fallback"].groupby(["probe", "bin_label"], sort=False).n_proteins_fallback.agg(["mean", "max"])
    lines += ["", "## Null A fallback (proteins per layer using cohort bin statistics)", "",
              markdown_table(fallback.reset_index()), ""]
    return "\n".join(lines)


def main() -> None:
    data = load()
    counts = bootstrap_counts(n_proteins=len(data["cohort"]))
    figure_observed_vs_nulls(data, counts)
    figure_gap(data, counts, "null_b", "B10", "observed_minus_nullB")
    figure_gap(data, counts, "null_c", "B10", "observed_minus_nullC")
    figure_gap(data, counts, "null_b", "B10xS6", "seqsep_variant_observed_minus_nullB")
    figure_gap(data, counts, "null_c", "B10xS6", "seqsep_variant_observed_minus_nullC")
    figure_sensitivity(data, counts)
    figure_bias_sd(data)
    calibration = calibration_table(data, counts)
    calibration.to_csv(TABLES / "calibration_by_layer.csv", index=False)
    figure_calibration(calibration)
    clipped = clipped_table(data, counts)
    clipped.to_csv(TABLES / "clipped_side_fraction.csv", index=False)
    figure_clipped(clipped)
    figure_crosstab(data)
    summary_table(data, counts).to_csv(TABLES / "summary_table.csv", index=False)
    (TABLES / "sanity_checks.md").write_text(sanity_markdown(data, counts))
    (TABLES / "RESULTS.md").write_text(results_markdown(data, counts))
    print(f"figures in {FIGURES}; tables and text in {TABLES}")


if __name__ == "__main__":
    main()

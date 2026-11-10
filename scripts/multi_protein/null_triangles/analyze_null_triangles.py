#!/usr/bin/env python3
"""Bootstraps, figures, tables, and sanity checks for the null-triangle experiment (552 proteins).

Reads the compact tables written by the cluster aggregation
(``scripts/cluster_scripts/null_triangles/aggregate_null_triangles.py``), copied to
``delta_files/regression_5fold_v1/analysis/null_triangles_5fold_v1/tables/``. Every curve is
a protein-equal mean over the combined CAMEO + ECOD cohort with a paired-protein bootstrap:
one multinomial count matrix (2000 replicates, seed 20260902) is shared by every probe,
layer, model, and metric. Observed and every null are scored on every triangle, so gaps
are exact per-protein differences.

Figures go to ``figures/main/``, ``figures/supplement/`` and ``figures/diagnostics/``.
Every y-axis starts at 0 except the observed − null gap plots, whose values are negative.
"""
from __future__ import annotations

import json
import os
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
from matplotlib.ticker import MaxNLocator  # noqa: E402

from generate_sept_2_publication_plots import (  # noqa: E402
    BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, bootstrap_counts, bootstrap_matrix_summary,
    configure_style, MAIN_FIGURE_FONT_SCALE,
)

ROOT = Path(os.environ.get("NULL_TRIANGLES_ANALYSIS_ROOT",       # override for local smoke tests only
                           PROJECT_ROOT / "delta_files/regression_5fold_v1/analysis/null_triangles_5fold_v1"))
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
LAYERS = np.arange(48)
TABLE_LAYERS = (0, 10, 21, 32, 40, 47)
CROSSTAB_LAYERS = (10, 21, 32, 47)
TROUGH, SECOND_PEAK = range(20, 23), range(30, 35)
OUTLIER = "7stz_C"
EXPECTED_PROTEINS = 552
# Paper order and names: shared linear, per-protein linear, XGBoost, sequence-gated.
PROBES = ("shared_ols_base256", "per_protein_ols_base256", "shared_xgboost_base256_fixed_s4",
          "sequence_gated_ols_base256")
SENSITIVITY_PROBE = "shared_ols_base256"
PROBE_LABELS = {"shared_ols_base256": "Shared linear", "per_protein_ols_base256": "Per-protein linear",
                "sequence_gated_ols_base256": "Sequence-gated", "shared_xgboost_base256_fixed_s4": "XGBoost"}
# Validated categorical order (CVD ΔE ≥ 8 on adjacent pairs); line styles are the secondary encoding.
PROBE_COLORS = {"shared_ols_base256": "#2a6fb0", "per_protein_ols_base256": "#d9822b",
                "sequence_gated_ols_base256": "#2a9d8f", "shared_xgboost_base256_fixed_s4": "#9b4dca"}
PROBE_STYLES = {"shared_ols_base256": "-", "per_protein_ols_base256": "--",
                "sequence_gated_ols_base256": "-.", "shared_xgboost_base256_fixed_s4": ":"}
METRICS = {"mean_relative_violation_all": "Normalized mean violation", "violation_fraction": "Violation fraction"}
NULLS = ("null_a", "null_b")
SERIES_COLORS = {"observed": "#222222", "null_a": "#d9822b", "null_b": "#2a6fb0"}
SERIES_LABELS = {"observed": "Observed", "null_a": "Null A (Gaussian)", "null_b": "Null B (permuted)"}
# Supplement 2: coarse → fine distance bins as a light → dark blue ramp; the sequence-separation
# variant gets its own hue. Each also has its own line style.
SENSITIVITY_SERIES = {
    "B5": ("#8fb8de", "--", "Null B, 5 bins"),
    "B10": ("#2a6fb0", "-", "Null B, 10 bins"),
    "B20": ("#163d66", ":", "Null B, 20 bins"),
    "B10xS6": ("#9b4dca", "-.", "Null B, 10 × 6 seq-sep bins"),
}

configure_style(font_scale=MAIN_FIGURE_FONT_SCALE)
plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                     "grid.color": "#e6e6e6", "grid.linewidth": 0.6, "axes.axisbelow": True,
                     "legend.frameon": False, "figure.titlesize": 21,
                     "axes.linewidth": 1.2, "xtick.major.size": 6, "ytick.major.size": 6})


# ---------------------------------------------------------------- data

def load() -> dict:
    per = pd.read_csv(TABLES / "null_per_protein.csv.gz", dtype={"protein_id": str})
    cohort = sorted(per["protein_id"].unique())
    if len(cohort) != EXPECTED_PROTEINS and "NULL_TRIANGLES_ANALYSIS_ROOT" not in os.environ:
        raise ValueError(f"Expected {EXPECTED_PROTEINS} proteins, found {len(cohort)}")
    return {
        "per": per, "cohort": cohort,
        "sources": per.drop_duplicates("protein_id").set_index("protein_id")["source"].reindex(cohort),
        "bins": pd.read_csv(TABLES / "error_diagnostics_by_bin.csv"),
        "crosstab": pd.read_csv(TABLES / "crosstab_b10_seqsep.csv"),
        "calibration": pd.read_csv(TABLES / "calibration_per_protein.csv", dtype={"protein_id": str}),
        "fallback": pd.read_csv(TABLES / "null_a_fallback_counts.csv"),
        "manifest": json.loads((TABLES / "run_manifest.json").read_text()),
    }


def matrix(per: pd.DataFrame, cohort: list[str], probe: str, model: str, binning: str, metric: str) -> np.ndarray:
    """Proteins x layers for one series."""
    rows = per[(per.probe == probe) & (per.model == model) & (per.binning == binning)]
    if rows.empty:
        raise ValueError(f"No rows for {probe} {model} {binning}")
    return rows.pivot(index="protein_id", columns="layer", values=metric).reindex(
        index=cohort, columns=LAYERS).to_numpy(float)


def observed(data, probe, metric):
    return matrix(data["per"], data["cohort"], probe, "observed", "none", metric)


def null(data, probe, model, binning, metric):
    return matrix(data["per"], data["cohort"], probe, model, binning, metric)


def gap(data, probe, model, binning, metric):
    """Observed minus null, per protein, on the same (every) triangle."""
    return observed(data, probe, metric) - null(data, probe, model, binning, metric)


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


def save(fig, folder: str, name: str) -> None:
    # Heatmaps use categorical bin labels; keep those and their colorbar intact.
    for ax in fig.axes:
        if not ax.images and ax.get_label() != "<colorbar>":
            ax.yaxis.set_major_locator(MaxNLocator(nbins=4, min_n_ticks=3))
    directory = FIGURES / folder
    directory.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(directory / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def band(ax, s, color, label, style="-"):
    ax.plot(LAYERS, s["mean"], style, color=color, lw=1.6, label=label)
    ax.fill_between(LAYERS, s["bootstrap_ci_low"], s["bootstrap_ci_high"], color=color, alpha=0.18, lw=0)


def from_zero(ax) -> None:
    ax.set_ylim(bottom=0)


# ---------------------------------------------------------------- main and supplement figures

def figure_observed_vs_nulls(data, counts, metric: str, folder: str, name: str) -> None:
    """One row of four probes, observed vs Null A vs Null B on B10, shared y-axis from 0."""
    fig, axes = plt.subplots(1, len(PROBES), figsize=(20, 5.5), sharex=True, sharey=True, layout="constrained")
    for ax, probe in zip(axes, PROBES):
        band(ax, summary(observed(data, probe, metric), counts), SERIES_COLORS["observed"], SERIES_LABELS["observed"])
        for model in NULLS:
            band(ax, summary(null(data, probe, model, "B10", metric), counts), SERIES_COLORS[model], SERIES_LABELS[model])
        ax.set_title(PROBE_LABELS[probe])
        ax.set_xlabel("Layer")
        ax.set_xlim(0, 47)
    from_zero(axes[0])
    axes[0].set_ylabel(METRICS[metric])
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, folder, name)


def figure_sensitivity(data, counts) -> None:
    """Shared OLS: Null B under B5 / B10 / B20 / B10×S6, both metrics, y from 0.

    Bands for observed and the B10 reference only; the other binnings are lines so the
    panel stays readable.
    """
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), sharex=True, layout="constrained")
    for ax, (metric, label) in zip(axes, METRICS.items()):
        band(ax, summary(observed(data, SENSITIVITY_PROBE, metric), counts), SERIES_COLORS["observed"], "Observed")
        for binning, (color, style, series_label) in SENSITIVITY_SERIES.items():
            s = summary(null(data, SENSITIVITY_PROBE, "null_b", binning, metric), counts)
            if binning == "B10":
                band(ax, s, color, series_label, style)
            else:
                ax.plot(LAYERS, s["mean"], style, color=color, lw=1.6, label=series_label)
        ax.set_xlabel("Layer")
        ax.set_ylabel(label)
        ax.set_xlim(0, 47)
        from_zero(ax)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    save(fig, "supplement", "sensitivity_binning_shared_ols")


# ---------------------------------------------------------------- diagnostic figures

def figure_gap(data, counts, model: str, binning: str, probes: tuple[str, ...], name: str) -> None:
    """Observed − null with 95% bands; dots where the band excludes 0. Centred on 0, not from 0."""
    fig, axes = plt.subplots(1, 2, figsize=(16, 6), layout="constrained")
    for ax, (metric, label) in zip(axes, METRICS.items()):
        for probe in probes:
            s = summary(gap(data, probe, model, binning, metric), counts)
            band(ax, s, PROBE_COLORS[probe], PROBE_LABELS[probe], PROBE_STYLES[probe])
            excludes = (s["bootstrap_ci_low"] > 0) | (s["bootstrap_ci_high"] < 0)
            ax.plot(LAYERS[excludes], s["mean"][excludes], "o", ms=3, color=PROBE_COLORS[probe])
        ax.axhline(0, color="#888888", lw=0.8)
        ax.set_xlabel("Layer")
        ax.set_ylabel(f"Observed − {SERIES_LABELS[model].split(' (')[0]}:\n{label.lower()}")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=len(probes), bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Observed − {SERIES_LABELS[model]} on {binning}\nDots: 95% band excludes 0")
    save(fig, "diagnostics", name)


def figure_bias_sd(data) -> None:
    bins = data["bins"]
    labels = ("[0,4)", "[4,6)", "[6,8)", "[8,10)")
    fig, axes = plt.subplots(1, 4, figsize=(22, 6), sharex=True, layout="constrained")
    for ax, label in zip(axes, labels):
        for probe in PROBES:
            rows = bins[(bins.probe == probe) & (bins.cell_label == label)].set_index("layer").reindex(LAYERS)
            style = PROBE_STYLES[probe]
            ax.plot(LAYERS, rows["protein_mean_sd_sq"], style, color="#2a6fb0", lw=1.4)
            ax.plot(LAYERS, rows["protein_mean_mse"], style, color="#c0392b", lw=1.4)
        ax.set_title(f"d_true in {label} Å")
        ax.set_xlabel("Layer")
        from_zero(ax)
    axes[0].set_ylabel("Squared error (Å²)")
    handles = [Line2D([], [], color="#2a6fb0", label="sd²"), Line2D([], [], color="#c0392b", label="sd² + bias²")]
    handles += [Line2D([], [], color="#444444", ls=PROBE_STYLES[p], label=PROBE_LABELS[p]) for p in PROBES]
    fig.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle("Error decomposition by distance bin\nProtein-equal means (proteins with ≥10 pairs in the bin)")
    save(fig, "diagnostics", "bias_sd_stacked")


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


def figure_line_table(table: pd.DataFrame, ylabel: str, name: str, reference: float | None = None) -> None:
    fig, ax = plt.subplots(figsize=(10, 6), layout="constrained")
    for probe in PROBES:
        rows = table[table.probe == probe].sort_values("layer")
        ax.plot(rows.layer, rows.protein_mean, PROBE_STYLES[probe], color=PROBE_COLORS[probe], label=PROBE_LABELS[probe])
        ax.fill_between(rows.layer, rows.ci_low, rows.ci_high, color=PROBE_COLORS[probe], alpha=0.18, lw=0)
    if reference is not None:
        ax.axhline(reference, color="#888888", lw=0.8)
    ax.set_xlabel("Layer")
    ax.set_ylabel(ylabel)
    from_zero(ax)
    handles, labels = ax.get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, bbox_to_anchor=(0.5, -0.02))
    save(fig, "diagnostics", name)


def clipped_table(data, counts) -> pd.DataFrame:
    records = []
    for probe in PROBES:
        s = summary(observed(data, probe, "clipped_side_fraction"), counts)
        for layer in LAYERS:
            records.append({"probe": probe, "layer": int(layer), "protein_mean": s["mean"][layer],
                            "ci_low": s["bootstrap_ci_low"][layer], "ci_high": s["bootstrap_ci_high"][layer],
                            "n_proteins": int(s["n_proteins_finite"][layer])})
    return pd.DataFrame(records)


def figure_crosstab(data) -> None:
    table = data["crosstab"]
    distance_labels = list(dict.fromkeys(table.distance_bin))
    sequence_labels = list(dict.fromkeys(table.sequence_bin))
    fig, axes = plt.subplots(len(PROBES), len(CROSSTAB_LAYERS), figsize=(20, 20), sharex=True, sharey=True, layout="constrained")
    finite = table[table.layer.isin(CROSSTAB_LAYERS)].protein_mean_sd
    vmax = float(np.nanpercentile(finite, 99))
    for r, probe in enumerate(PROBES):
        for c, layer in enumerate(CROSSTAB_LAYERS):
            rows = table[(table.probe == probe) & (table.layer == layer)]
            grid = rows.pivot(index="distance_bin", columns="sequence_bin", values="protein_mean_sd").reindex(
                index=distance_labels, columns=sequence_labels)
            ax = axes[r, c]
            ax.grid(False)
            image = ax.imshow(grid.to_numpy(float), cmap="viridis", vmin=0, vmax=vmax, aspect="auto")
            for (y, x), value in np.ndenumerate(grid.to_numpy(float)):
                if np.isfinite(value):
                    ax.text(x, y, f"{value:.1f}", ha="center", va="center", fontsize=12, color="white")
            ax.set_xticks(range(len(sequence_labels)), sequence_labels, fontsize=14, rotation=45, ha="right")
            ax.set_yticks(range(len(distance_labels)), distance_labels, fontsize=14)
            if r == 0:
                ax.set_title(f"Layer {layer}")
            if c == 0:
                ax.set_ylabel(f"{PROBE_LABELS[probe]}\nd_true bin (Å)")
            if r == len(PROBES) - 1:
                ax.set_xlabel("Sequence separation")
    colorbar = fig.colorbar(image, ax=axes, shrink=0.5, label="Protein-equal mean residual SD (Å)")
    colorbar.locator = MaxNLocator(nbins=4)
    colorbar.update_ticks()
    fig.suptitle("Residual SD by distance × sequence separation\nEmpty: < 100 pairs cohort-wide")
    save(fig, "diagnostics", "crosstab_sd_heatmap")


# ---------------------------------------------------------------- tables and text

def summary_table(data, counts) -> pd.DataFrame:
    records = []
    for probe in PROBES:
        for metric in METRICS:
            series = {"observed": observed(data, probe, metric)}
            for model in NULLS:
                series[model] = null(data, probe, model, "B10", metric)
                series[f"observed_minus_{model}"] = gap(data, probe, model, "B10", metric)
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


def interval(c: dict) -> str:
    return f"{c['mean']:.4g} [{c['ci_low']:.4g}, {c['ci_high']:.4g}]"


def results_markdown(data, counts) -> str:
    n_by_source = data["sources"].value_counts().to_dict()
    lines = ["# Null models for triangle violations: results", "",
             f"{len(data['cohort'])} proteins ({n_by_source.get('cameo', 0)} CAMEO + {n_by_source.get('ecod', 0)} ECOD), "
             f"layers 0–47, every triangle, K={data['manifest']['n_draws']} draws per null, protein-equal means, "
             f"95% paired-protein bootstrap ({BOOTSTRAP_REPLICATES} replicates, seed {BOOTSTRAP_SEED}). "
             "Below / above: the 95% band of observed − null lies entirely below / above 0. Second-peak contrast: "
             "per-protein mean over L30–34 minus mean over L20–22, paired bootstrap; it survives when the interval "
             "lies above 0.", ""]
    for probe in PROBES:
        lines += [f"## {PROBE_LABELS[probe]}", ""]
        for metric, label in METRICS.items():
            lines += [f"**{label}**", ""]
            for model in NULLS:
                s = summary(gap(data, probe, model, "B10", metric), counts)
                below = LAYERS[s["bootstrap_ci_high"] < 0]
                above = LAYERS[s["bootstrap_ci_low"] > 0]
                within = LAYERS[(s["bootstrap_ci_low"] <= 0) & (s["bootstrap_ci_high"] >= 0)]
                lines.append(f"- Observed vs {SERIES_LABELS[model]}, B10: below in {contiguous_ranges(below)}; "
                             f"above in {contiguous_ranges(above)}; matches (band includes 0) in "
                             f"{contiguous_ranges(within)}.")
            obs = window_contrast(observed(data, probe, metric), counts)
            null_b = window_contrast(null(data, probe, "null_b", "B10", metric), counts)
            gap_b = window_contrast(gap(data, probe, "null_b", "B10", metric), counts)
            lines.append(f"- Second peak in observed: {interval(obs)}; in Null B alone: {interval(null_b)}; "
                         f"in observed − Null B: {interval(gap_b)} "
                         f"({'survives' if gap_b['ci_low'] > 0 else 'does not survive'}).")
            lines.append("")
    lines += [f"## Binning sensitivity ({PROBE_LABELS[SENSITIVITY_PROBE]}, Null B)", ""]
    for metric, label in METRICS.items():
        for binning in SENSITIVITY_SERIES:
            s = summary(gap(data, SENSITIVITY_PROBE, "null_b", binning, metric), counts)
            below = LAYERS[s["bootstrap_ci_high"] < 0]
            lines.append(f"- {label}, {binning}: observed below Null B in {contiguous_ranges(below)}; "
                         f"gap at L47 {s['mean'][47]:.4g} [{s['bootstrap_ci_low'][47]:.4g}, {s['bootstrap_ci_high'][47]:.4g}].")
    lines.append("")
    return "\n".join(lines)


def sanity_markdown(data, counts) -> str:
    manifest, per = data["manifest"], data["per"]
    checks = manifest["checks"]
    worst = checks["paper_parity_worst"]
    lines = ["# Sanity checks", "",
             f"- Streaming reducer vs `compute_metric_rows`, bit-identical (proteins with ≤ "
             f"{manifest['bitwise_max_triangles']:,} triangles): {checks['bitwise_compared']} comparisons, "
             f"{checks['bitwise_failures']} failures.",
             f"- Observed vs the paper's stored per-protein values: {checks['paper_parity_compared']} comparisons, "
             f"worst violation-fraction gap {worst['paper_fraction']:.3g}, worst normalized-sum relative gap "
             f"{worst['paper_relative_sum_rtol']:.3g}, worst negative-count gap {worst['paper_negative']} "
             "(float32 cache vs float64 replay).",
             f"- Null B kept each bin's residual histogram in all {checks['histogram_checks']} draws "
             f"({checks['histogram_failures']} failures).",
             f"- Triangles scored per layer per vector: {manifest['n_triangles_per_layer']:,}.", ""]
    probe = "shared_xgboost_base256_fixed_s4"
    lines.append("## Null A vs Null B, XGBoost, layer 47")
    for metric, label in METRICS.items():
        s = summary(null(data, probe, "null_a", "B10", metric) - null(data, probe, "null_b", "B10", metric), counts)
        agree = s["bootstrap_ci_low"][47] <= 0 <= s["bootstrap_ci_high"][47]
        lines.append(f"- {label}: A − B = {s['mean'][47]:.4g} [{s['bootstrap_ci_low'][47]:.4g}, "
                     f"{s['bootstrap_ci_high'][47]:.4g}] ({'agree' if agree else 'differ'}).")
    shape = data["bins"][(data["bins"].probe == probe) & (data["bins"].layer == 47)][
        ["cell_label", "protein_mean_skew", "protein_mean_excess_kurtosis", "n_proteins_contributing"]]
    lines += ["", "Per-bin residual shape at layer 47 (protein-equal means):", "", markdown_table(shape), ""]
    lines += ["## Monte Carlo noise (K draws per protein)", "",
              "| Model | Binning | Metric | Mean per-protein draw SD | Cohort-mean MC SE | Median bootstrap CI half-width |",
              "|---|---|---|---|---|---|"]
    runs = [("null_a", "B10", PROBES), ("null_b", "B10", PROBES)]
    runs += [("null_b", b, (SENSITIVITY_PROBE,)) for b in ("B5", "B20", "B10xS6")]
    for model, binning, probes in runs:
        for metric, label in METRICS.items():
            rows = per[(per.model == model) & (per.binning == binning)]
            draw_sd = rows[f"{metric}_draw_sd"]
            k = int(rows.n_draws.iloc[0])
            se = float(np.sqrt(np.nanmean(draw_sd ** 2) / k / len(data["cohort"])))
            widths = [np.median((lambda s: (s["bootstrap_ci_high"] - s["bootstrap_ci_low"]) / 2)(
                summary(null(data, p, model, binning, metric), counts))) for p in probes]
            lines.append(f"| {model} | {binning} | {label} | {np.nanmean(draw_sd):.3g} | {se:.3g} | {np.median(widths):.3g} |")
    keep = np.array([pid != OUTLIER for pid in data["cohort"]])
    if not keep.all():
        counts_wo = bootstrap_counts(n_proteins=int(keep.sum()))
        lines += ["", f"## Robustness: observed − Null B (B10) without {OUTLIER}", "",
                  f"| Probe | Metric | Layer | All proteins | Without {OUTLIER} |", "|---|---|---|---|---|"]
        for probe in PROBES:
            for metric, label in METRICS.items():
                values = gap(data, probe, "null_b", "B10", metric)
                s_all, s_wo = summary(values, counts), summary(values[keep], counts_wo)
                for layer in TABLE_LAYERS:
                    lines.append(f"| {PROBE_LABELS[probe]} | {label} | {layer} | "
                                 f"{s_all['mean'][layer]:.4g} [{s_all['bootstrap_ci_low'][layer]:.4g}, {s_all['bootstrap_ci_high'][layer]:.4g}] | "
                                 f"{s_wo['mean'][layer]:.4g} [{s_wo['bootstrap_ci_low'][layer]:.4g}, {s_wo['bootstrap_ci_high'][layer]:.4g}] |")
    fallback = data["fallback"].groupby(["probe", "bin_label"], sort=False).n_proteins_fallback.agg(["mean", "max"])
    lines += ["", "## Null A fallback (proteins per layer using cohort bin statistics)", "",
              markdown_table(fallback.reset_index()), ""]
    return "\n".join(lines)


def main() -> None:
    data = load()
    counts = bootstrap_counts(n_proteins=len(data["cohort"]))
    figure_observed_vs_nulls(data, counts, "mean_relative_violation_all", "main", "observed_vs_nulls_normalized_violation")
    figure_observed_vs_nulls(data, counts, "violation_fraction", "supplement", "observed_vs_nulls_violation_fraction")
    figure_sensitivity(data, counts)
    figure_gap(data, counts, "null_a", "B10", PROBES, "observed_minus_nullA")
    figure_gap(data, counts, "null_b", "B10", PROBES, "observed_minus_nullB")
    figure_gap(data, counts, "null_b", "B10xS6", (SENSITIVITY_PROBE,), "seqsep_variant_observed_minus_nullB")
    figure_bias_sd(data)
    calibration = calibration_table(data, counts)
    calibration.to_csv(ROOT / "calibration_by_layer.csv", index=False)
    figure_line_table(calibration[calibration.parameter == "beta"],
                      "β in d_pred = α + β·d_true\n(protein-equal mean)", "calibration_beta", reference=1.0)
    clipped = clipped_table(data, counts)
    clipped.to_csv(ROOT / "clipped_side_fraction.csv", index=False)
    figure_line_table(clipped, "Share of violating triples\nwith a side clipped to 0", "clipped_side_fraction")
    figure_crosstab(data)
    summary_table(data, counts).to_csv(ROOT / "summary_table.csv", index=False)
    (ROOT / "sanity_checks.md").write_text(sanity_markdown(data, counts))
    (ROOT / "RESULTS.md").write_text(results_markdown(data, counts))
    print(f"figures in {FIGURES}; tables and text in {ROOT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Local analysis of the per-channel classical MDS of the pair representation (154 CAMEO proteins).

Reads the aggregated tables under
``delta_files/diagnostic_experiments/per_channel_mds_cameo_v1/tables`` and the shared linear
probe's CAMEO rows of ``regression_5fold_v1/shared_ols_base256/geometry_metrics.csv``. Writes
``summary_by_layer.csv``, ``channel_summary.csv``, ``RESULTS.md``, ``sanity_checks.md`` and
the figures.

Every cohort curve first reduces the 128 channels within a protein (mean, decile, best
channel) and then averages proteins equally. Bands are the paper's paired-protein bootstrap.
The probe rows use clipped OOF distances with no shift; the channel rows are min-shifted, so
the min-shifted true-distance reference is the floor a channel can reach.
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
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.lines import Line2D  # noqa: E402

from generate_sept_2_publication_plots import (  # noqa: E402
    MAIN_FIGURE_FONT_SCALE, bootstrap_counts, bootstrap_matrix_summary, configure_style,
)

SMOKE = "PER_CHANNEL_MDS_ANALYSIS_ROOT" in os.environ               # local smoke tests only
ROOT = Path(os.environ.get(
    "PER_CHANNEL_MDS_ANALYSIS_ROOT", PROJECT_ROOT / "delta_files/diagnostic_experiments/per_channel_mds_cameo_v1"))
TABLES = ROOT / "tables"
FIGURES = ROOT / "figures"
PROBE = "shared_ols_base256"
PROBE_TABLE = PROJECT_ROOT / "delta_files/regression_5fold_v1" / PROBE / "geometry_metrics.csv"
EXPECTED_PROTEINS, EXPECTED_LAYERS, N_CHANNELS = 154, 48, 128
SNAPSHOT_LAYERS = (0, 12, 24, 36, 47)
TABLE_LAYERS = (0, 10, 20, 30, 40, 47)

# metric -> (axis label, statistic of the better decile, statistic of the best channel)
METRICS = {
    "stress": ("Stress", "p10", "min"),
    "negative_eigenvalue_fraction": ("Negative eigenvalue fraction", "p10", "min"),
    "frac_top3_eigenvalues": ("Top-3 eigenvalue fraction", "p90", "max"),
}
BLUE, ORANGE, AQUA, GREY = "#2a78d6", "#eb6834", "#1baf7a", "#52514e"   # grey is the neutral null
# Drawn series -> (label, colour, line style, drawn with a band). The better decile and the
# permuted best channel are in summary_by_layer.csv and RESULTS.md but not in the figure.
SERIES = {
    "channel_mean": ("Channel mean", BLUE, "-", True),
    "channel_best": ("Best channel", BLUE, ":", False),
    "permuted_mean": ("Permuted channel mean", GREY, "-", True),
    "shared_ols_probe": ("Shared linear probe", ORANGE, "-", True),
    "true_distance_minshift": ("True distances, min-shifted", AQUA, "-.", False),
}
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "sequential_blue", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


def load() -> dict:
    channels = pd.read_csv(TABLES / "channel_metrics.csv.gz")
    summary = pd.read_csv(TABLES / "protein_layer_summary.csv")
    references = pd.read_csv(TABLES / "reference_metrics.csv")
    manifest = json.loads((TABLES / "run_manifest.json").read_text())
    cohort = sorted(summary["protein_id"].unique())
    layers = sorted(int(x) for x in summary["layer"].unique())
    probe = pd.read_csv(PROBE_TABLE, dtype={"protein_id": str})
    probe = probe[(probe["source"] == "cameo") & probe["protein_id"].isin(cohort) & probe["layer"].isin(layers)]
    ok = channels[channels["status"] == "ok"]
    decile = ok.groupby(["layer", "protein_id"])["frac_top3_eigenvalues"].quantile(0.9)
    summary = summary.merge(decile.rename("frac_top3_eigenvalues_p90").reset_index(), on=["layer", "protein_id"])
    return {"channels": channels, "summary": summary, "references": references, "manifest": manifest,
            "cohort": cohort, "layers": layers, "probe": probe}


def pivot(frame: pd.DataFrame, column: str, data: dict) -> np.ndarray:
    """Values as proteins x layers, in cohort and layer order."""
    wide = frame.pivot(index="protein_id", columns="layer", values=column)
    return wide.reindex(index=data["cohort"], columns=data["layers"]).to_numpy(dtype=float)


def series_matrices(data: dict, metric: str) -> dict[str, np.ndarray]:
    _, decile, best = METRICS[metric]
    summary, references = data["summary"], data["references"]
    floor = references[(references["reference"] == "true_distance") & (references["shift"] == "minshift")]
    floor = floor.set_index("protein_id")[metric].reindex(data["cohort"]).to_numpy(dtype=float)
    return {
        "channel_mean": pivot(summary, f"{metric}_mean", data),
        "channel_better_decile": pivot(summary, f"{metric}_{decile}", data),
        "channel_best": pivot(summary, f"{metric}_{best}", data),
        "channel_variance": pivot(summary, f"{metric}_var", data),
        "permuted_mean": pivot(summary, f"perm_{metric}_mean", data),
        "permuted_best": pivot(summary, f"perm_{metric}_{best}", data),
        "shared_ols_probe": pivot(data["probe"], metric, data),
        "true_distance_minshift": np.repeat(floor[:, None], len(data["layers"]), axis=1),
    }


def summary_by_layer(data: dict, counts: np.ndarray) -> pd.DataFrame:
    rows = []
    for metric in METRICS:
        for series, matrix in series_matrices(data, metric).items():
            s = bootstrap_matrix_summary(matrix, counts)
            for k, layer in enumerate(data["layers"]):
                rows.append({"layer": layer, "metric": metric, "series": series, "mean": s["mean"][k],
                             "bootstrap_ci_low": s["bootstrap_ci_low"][k],
                             "bootstrap_ci_high": s["bootstrap_ci_high"][k],
                             "n_proteins": int(s["n_proteins_finite"][k])})
    return pd.DataFrame(rows)


def channel_summary(data: dict) -> pd.DataFrame:
    """Cohort means for every layer and channel; proteins weigh equally."""
    ok = data["channels"][data["channels"]["status"] == "ok"].copy()
    ok["abs_spearman_true_distance"] = ok["spearman_true_distance"].abs()
    ok["abs_spearman_sequence_separation"] = ok["spearman_sequence_separation"].abs()
    ok["positive_sign"] = (ok["best_sign"] == 1).astype(float)
    columns = ["stress", "perm_stress", "negative_eigenvalue_fraction", "frac_top3_eigenvalues",
               "frac_top1_eigenvalue", "spearman_true_distance", "abs_spearman_true_distance",
               "spearman_sequence_separation", "abs_spearman_sequence_separation",
               "embedding_true_distance_pearson", "antisymmetric_fraction", "positive_sign"]
    grouped = ok.groupby(["layer", "channel"])
    out = grouped[columns].mean()
    out.insert(0, "n_proteins", grouped.size())
    return out.reset_index()


# ---------------------------------------------------------------- figures

def save(fig, name: str) -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    for suffix in ("pdf", "png"):
        fig.savefig(FIGURES / f"{name}.{suffix}", dpi=300, bbox_inches="tight")
    plt.close(fig)


def layer_axis(ax, layers: list[int]) -> None:
    ax.set_xlim(min(layers), max(layers))
    ax.set_xticks([x for x in range(0, 48, 10) if min(layers) <= x <= max(layers)] or layers)
    ax.set_xlabel("Layer")
    ax.locator_params(axis="y", nbins=5)


def figure_metric_by_layer(summary: pd.DataFrame, layers: list[int]) -> None:
    configure_style(font_scale=MAIN_FIGURE_FONT_SCALE)
    fig, axes = plt.subplots(1, 3, figsize=(24, 7.2))
    for ax, (metric, (label, _, _)) in zip(axes, METRICS.items()):
        for series, (_, colour, style, band) in SERIES.items():
            s = summary[(summary["metric"] == metric) & (summary["series"] == series)].sort_values("layer")
            ax.plot(s["layer"], s["mean"], color=colour, linestyle=style, linewidth=2.4)
            if band:
                ax.fill_between(s["layer"], s["bootstrap_ci_low"], s["bootstrap_ci_high"], color=colour,
                                alpha=0.18, linewidth=0)
        ax.set_ylabel(label)
        ax.set_ylim(bottom=0)
        layer_axis(ax, layers)
    handles = [Line2D([0], [0], color=colour, linestyle=style, linewidth=2.4, label=label)
               for label, colour, style, _ in SERIES.values()]
    fig.legend(handles=handles, loc="lower center", ncol=len(handles), frameon=False, bbox_to_anchor=(0.5, -0.1))
    fig.tight_layout(w_pad=2.5)
    save(fig, "metric_by_layer")


def figure_heatmap(channels: pd.DataFrame, layers: list[int]) -> None:
    configure_style(font_scale=MAIN_FIGURE_FONT_SCALE)
    wide = channels.pivot(index="channel", columns="layer", values="stress").reindex(columns=layers)
    wide = wide.sort_values(layers[-1])
    fig, ax = plt.subplots(figsize=(13, 9.5))
    image = ax.imshow(wide.to_numpy(), aspect="auto", cmap=SEQUENTIAL, interpolation="nearest",
                      extent=(layers[0] - 0.5, layers[-1] + 0.5, len(wide) + 0.5, 0.5))
    ax.grid(False)
    ax.set_xlabel("Layer")
    ax.set_ylabel(f"Channel, ranked by layer-{layers[-1]} stress")
    ax.set_xticks([x for x in range(0, 48, 10) if layers[0] <= x <= layers[-1]] or layers)
    ax.set_yticks([1, 32, 64, 96, 128])
    bar = fig.colorbar(image, ax=ax, pad=0.02)
    bar.set_label("Cohort-mean stress")
    bar.ax.locator_params(nbins=5)
    save(fig, "stress_heatmap_layer_by_channel")


def figure_scatter(channels: pd.DataFrame, snapshots: list[int], floor: float) -> None:
    configure_style(font_scale=MAIN_FIGURE_FONT_SCALE)
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4.3 * len(snapshots) + 1.5, 6.4), sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    top = float(np.ceil(channels["abs_spearman_sequence_separation"].max() * 10) / 10)
    for ax, layer in zip(axes[0], snapshots):
        s = channels[channels["layer"] == layer]
        points = ax.scatter(s["abs_spearman_true_distance"], s["stress"], c=s["abs_spearman_sequence_separation"],
                            cmap=SEQUENTIAL, vmin=0, vmax=top, s=70, edgecolors="white", linewidths=0.8)
        ax.axhline(floor, color=AQUA, linestyle="-.", linewidth=2.2)
        ax.set_title(f"Layer {layer}")
        ax.set_xlim(0, 1)
        ax.set_xticks([0, 0.5, 1])
        ax.locator_params(axis="y", nbins=5)
    axes[0, len(snapshots) // 2].set_xlabel("|Spearman| with true distance")
    axes[0, 0].set_ylabel("Cohort-mean stress")
    axes[0, 0].set_ylim(bottom=0)
    bar = fig.colorbar(points, ax=axes[0].tolist(), pad=0.01, aspect=18)
    bar.set_label("|Spearman| with sequence separation")
    bar.ax.locator_params(nbins=5)
    fig.legend(handles=[Line2D([0], [0], color=AQUA, linestyle="-.", linewidth=2.2,
                               label="True distances, min-shifted (floor)"),
                        Line2D([0], [0], marker="o", linestyle="", color=BLUE, markersize=10, label="One channel")],
               loc="outside lower center", ncol=2, frameon=False)
    save(fig, "stress_vs_distance_correlation")


def figure_distributions(channels: pd.DataFrame, snapshots: list[int]) -> None:
    configure_style(font_scale=MAIN_FIGURE_FONT_SCALE)
    fig, axes = plt.subplots(1, len(snapshots), figsize=(4.3 * len(snapshots), 6.0), sharex=True, sharey=True,
                             squeeze=False, constrained_layout=True)
    ok = channels[channels["status"] == "ok"]
    upper = float(np.ceil(np.nanpercentile(ok[["stress", "perm_stress"]].to_numpy(), 99.9) * 10) / 10)
    bins = np.linspace(0, upper, 61)
    for ax, layer in zip(axes[0], snapshots):
        s = ok[ok["layer"] == layer]
        for column, colour in (("perm_stress", GREY), ("stress", BLUE)):
            ax.hist(s[column], bins=bins, density=True, histtype="stepfilled", color=colour, alpha=0.30)
            ax.hist(s[column], bins=bins, density=True, histtype="step", color=colour, linewidth=2.2)
        ax.set_title(f"Layer {layer}")
        ax.set_xlim(0, upper)
        ax.locator_params(axis="x", nbins=3)
        ax.locator_params(axis="y", nbins=5)
    axes[0, 0].set_ylabel("Density over\nprotein-channel pairs")
    axes[0, len(snapshots) // 2].set_xlabel("Stress")
    fig.legend(handles=[Line2D([0], [0], color=BLUE, linewidth=2.2, label="Channel"),
                        Line2D([0], [0], color=GREY, linewidth=2.2, label="Permuted channel")],
               loc="outside lower center", ncol=2, frameon=False)
    save(fig, "stress_distribution_real_vs_permuted")


# ---------------------------------------------------------------- text

def value(summary: pd.DataFrame, metric: str, series: str, layer: int, interval: bool = False) -> str:
    row = summary[(summary["metric"] == metric) & (summary["series"] == series) & (summary["layer"] == layer)].iloc[0]
    text = f"{row['mean']:.3f}"
    return f"{text} [{row['bootstrap_ci_low']:.3f}, {row['bootstrap_ci_high']:.3f}]" if interval else text


def headline(data: dict, summary: pd.DataFrame, floor: float) -> list[str]:
    """The stress numbers at the last layer and the shape of the channel-mean curve."""
    layers, last = data["layers"], data["layers"][-1]
    stress = summary[summary["metric"] == "stress"]

    def at(series: str, layer: int) -> float:
        return float(stress[(stress["series"] == series) & (stress["layer"] == layer)]["mean"].iloc[0])

    null = at("permuted_mean", last)
    curve = stress[stress["series"] == "channel_mean"].set_index("layer")["mean"]
    lowest = int(curve.idxmin())
    later = curve.loc[lowest:]
    lines = [
        "## Headline", "",
        f"- At layer {last} the channel mean stress is {at('channel_mean', last):.3f}, against {null:.3f} for permuted",
        f"  channels, {at('shared_ols_probe', last):.3f} for the shared linear probe and {floor:.3f} for the shift floor.",
        f"  From the permuted null to the floor, the channel mean covers "
        f"{100 * (null - at('channel_mean', last)) / (null - floor):.0f}% of the gap and the probe "
        f"{100 * (null - at('shared_ols_probe', last)) / (null - floor):.0f}%.",
        f"- The best channel of each protein reaches {at('channel_best', last):.3f} at layer {last} "
        f"({at('channel_best', layers[0]):.3f} at layer {layers[0]}); the best of 128 permuted channels reaches "
        f"{at('permuted_best', last):.3f}.",
        f"- The channel mean is lowest at layer {lowest} ({curve.loc[lowest]:.3f}); after it the highest value is "
        f"{later.max():.3f} at layer {int(later.idxmax())}.",
        "",
    ]
    return lines


def write_results(data: dict, summary: pd.DataFrame, channels: pd.DataFrame) -> None:
    layers, last = data["layers"], data["layers"][-1]
    table_layers = [x for x in TABLE_LAYERS if x in layers] or layers
    references = data["references"].groupby(["reference", "shift"])[list(METRICS)].mean()
    floor = float(references.loc[("true_distance", "minshift"), "stress"])
    ok = data["channels"][data["channels"]["status"] == "ok"]
    lines = [
        "# Per-channel classical MDS of the pair representation",
        "",
        f"{len(data['cohort'])} CAMEO proteins, layers {layers[0]}–{last}, 128 channels. Each channel is symmetrised,",
        "oriented by the sign with lower stress, shifted so its smallest off-diagonal value is zero, and",
        "embedded in 3D with the classical MDS used for the probes. Generated by",
        "`scripts/multi_protein/per_channel_mds/analyze_per_channel_mds.py`; design in",
        "`scripts/cluster_scripts/diagnostics/per_channel_mds/README.md`.",
        "",
        "Channels are one axis of an arbitrary basis, so the channel means below describe this basis,",
        "not the representation. Cohort numbers reduce the 128 channels within each protein first and",
        "then average proteins equally; brackets are 95% paired-protein bootstrap intervals.",
        "",
        *headline(data, summary, floor),
    ]
    for metric, (label, decile, best) in METRICS.items():
        direction = "lower" if best == "min" else "higher"
        lines += [f"## {label}", "",
                  f"Better decile is the channel `{decile}` within a protein and best channel the `{best}` "
                  f"({direction} is more Euclidean).", "",
                  "| Layer | Channel mean | Variance across channels | Better decile | Best channel | "
                  "Permuted mean | Permuted best channel | Shared linear probe |", "|---|---|---|---|---|---|---|---|"]
        for layer in table_layers:
            lines.append(
                f"| {layer} | {value(summary, metric, 'channel_mean', layer, True)} | "
                f"{value(summary, metric, 'channel_variance', layer)} | "
                f"{value(summary, metric, 'channel_better_decile', layer)} | "
                f"{value(summary, metric, 'channel_best', layer)} | "
                f"{value(summary, metric, 'permuted_mean', layer)} | "
                f"{value(summary, metric, 'permuted_best', layer)} | "
                f"{value(summary, metric, 'shared_ols_probe', layer, True)} |")
        lines.append("")
    lines += ["## Reference matrices through the same MDS", "",
              "Cohort means. The min-shifted true distances are what a channel exactly affine in distance",
              "would score, so they are the floor for the channel rows.", "",
              "| Matrix | Shift | Stress | Negative eigenvalue fraction | Top-3 eigenvalue fraction |", "|---|---|---|---|---|"]
    names = {"true_distance": "Final-structure distances", "sequence_separation": "Sequence separation \\|i − j\\|"}
    for (reference, shift), row in references.iterrows():
        lines.append(f"| {names[reference]} | {shift} | {row['stress']:.3f} | "
                     f"{row['negative_eigenvalue_fraction']:.3f} | {row['frac_top3_eigenvalues']:.3f} |")
    lines += ["", "## Which channels are geometric", "",
              "Per channel, cohort means over proteins. `|ρ| distance` and `|ρ| separation` are mean absolute",
              "Spearman correlations of the symmetrised channel with final-structure distance and with",
              "`|i − j|`; `embedding r` is the Pearson correlation between the channel's 3D embedding",
              "distances and the final-structure distances.", ""]
    for layer in (layers[0], last):
        top = channels[channels["layer"] == layer].nsmallest(8, "stress")
        lines += [f"Lowest-stress channels at layer {layer}:", "",
                  "| Channel | Stress | Permuted stress | \\|ρ\\| distance | \\|ρ\\| separation | Embedding r | "
                  "Positive sign share |", "|---|---|---|---|---|---|---|"]
        for _, r in top.iterrows():
            lines.append(f"| {int(r['channel'])} | {r['stress']:.3f} | {r['perm_stress']:.3f} | "
                         f"{r['abs_spearman_true_distance']:.3f} | {r['abs_spearman_sequence_separation']:.3f} | "
                         f"{r['embedding_true_distance_pearson']:.3f} | {r['positive_sign']:.2f} |")
        lines.append("")
    lines += ["Channel counts by cohort-mean values:", "",
              "| Layer | Stress below permuted | Stress < 0.3 | \\|ρ\\| distance > 0.5 | \\|ρ\\| separation > 0.5 | "
              "Embedding r > 0.5 | Mean antisymmetric fraction | r(stress, \\|ρ\\| distance) | "
              "r(stress, \\|ρ\\| separation) |", "|---|---|---|---|---|---|---|---|---|"]
    for layer in table_layers:
        s = channels[channels["layer"] == layer]
        lines.append(f"| {layer} | {int((s['stress'] < s['perm_stress']).sum())} | {int((s['stress'] < 0.3).sum())} | "
                     f"{int((s['abs_spearman_true_distance'] > 0.5).sum())} | "
                     f"{int((s['abs_spearman_sequence_separation'] > 0.5).sum())} | "
                     f"{int((s['embedding_true_distance_pearson'] > 0.5).sum())} | "
                     f"{s['antisymmetric_fraction'].mean():.3f} | "
                     f"{np.corrcoef(s['stress'], s['abs_spearman_true_distance'])[0, 1]:.2f} | "
                     f"{np.corrcoef(s['stress'], s['abs_spearman_sequence_separation'])[0, 1]:.2f} |")
    relation = ok.groupby("layer")[["stress", "spearman_true_distance"]].apply(
        lambda g: np.corrcoef(g["stress"], g["spearman_true_distance"].abs())[0, 1])
    lines += ["", "The last two columns are Pearson correlations over the 128 channels between cohort-mean stress",
              "and each cohort-mean absolute Spearman.",
              "", f"Across protein-channel pairs, the correlation between stress and `|ρ| distance` is "
                  f"{relation.loc[layers[0]]:.2f} at layer {layers[0]} and {relation.loc[last]:.2f} at layer {last}. "
                  f"The stress floor from the shift rule is {floor:.3f}.", "",
              "## Figures", "",
              "- `figures/metric_by_layer`: the three metrics by layer, channel mean and best channel, with",
              "  the permuted channel mean, the shared linear probe and the min-shifted true distances.",
              "- `figures/stress_heatmap_layer_by_channel`: cohort-mean stress per channel and layer.",
              "- `figures/stress_vs_distance_correlation`: one point per channel at selected layers.",
              "- `figures/stress_distribution_real_vs_permuted`: stress over protein-channel pairs.", ""]
    (ROOT / "RESULTS.md").write_text("\n".join(lines))


def write_sanity(data: dict, summary: pd.DataFrame) -> None:
    channels, references, manifest = data["channels"], data["references"], data["manifest"]
    n_proteins, n_layers = len(data["cohort"]), len(data["layers"])
    raw = references[(references["reference"] == "true_distance") & (references["shift"] == "raw")]
    line = references[(references["reference"] == "sequence_separation") & (references["shift"] == "raw")]
    ok = channels[channels["status"] == "ok"]
    probe_rows = len(data["probe"])
    checks = [
        ("Proteins and layers", f"{n_proteins} proteins, {n_layers} layers",
         SMOKE or (n_proteins == EXPECTED_PROTEINS and n_layers == EXPECTED_LAYERS)),
        ("Channel rows", f"{len(channels)} (expected {n_proteins * n_layers * N_CHANNELS})",
         len(channels) == n_proteins * n_layers * N_CHANNELS),
        ("Unique (layer, protein, channel) keys", str(not channels.duplicated(["layer", "protein_id", "channel"]).any()),
         not channels.duplicated(["layer", "protein_id", "channel"]).any()),
        ("Constant channels", str(int((channels["status"] != "ok").sum())), True),
        ("Stress finite and non-negative on ok rows",
         f"min {ok['stress'].min():.4f}, max {ok['stress'].max():.4f}",
         bool(np.isfinite(ok["stress"]).all() and (ok["stress"] >= 0).all())),
        ("Kept sign has the lower stress", f"max excess {(ok['stress'] - ok['stress_other_sign']).max():.2e}",
         bool((ok["stress"] <= ok["stress_other_sign"] + 1e-12).all())),
        ("True distances embed exactly (raw)", f"max stress {raw['stress'].max():.2e}", bool(raw["stress"].max() < 1e-6)),
        ("Sequence separation is one-dimensional (raw)", f"min top-1 fraction {line['frac_top1_eigenvalue'].min():.6f}",
         bool(line["frac_top1_eigenvalue"].min() > 1 - 1e-6)),
        ("Batched MDS parity with helpers.classical_mds",
         f"max difference {manifest['checks']['parity_max_abs_difference']:.2e}",
         manifest["checks"]["parity_max_abs_difference"] < 1e-8),
        ("Probe rows joined", f"{probe_rows} of {n_proteins * n_layers}", probe_rows == n_proteins * n_layers),
        ("Summary matches the channel table",
         f"max difference {summary_mismatch(data):.2e}", summary_mismatch(data) < 1e-9),
    ]
    lines = ["# Sanity checks: per-channel MDS", "", f"Code revision: `{manifest.get('code_revision', '')}`.", "",
             "| Check | Value | Pass |", "|---|---|---|"]
    lines += [f"| {name} | {text} | {'yes' if passed else 'NO'} |" for name, text, passed in checks]
    share = (ok["stress"] < ok["perm_stress"]).groupby(ok["layer"]).mean()
    lines += ["", "Share of protein-channel pairs whose stress is below their permuted stress: "
              + ", ".join(f"layer {layer} {share.loc[layer]:.3f}" for layer in data["layers"]
                          if layer in SNAPSHOT_LAYERS or SMOKE) + ".", ""]
    (ROOT / "sanity_checks.md").write_text("\n".join(lines))
    failed = [name for name, _, passed in checks if not passed]
    if failed:
        raise SystemExit(f"Sanity checks failed: {failed}")


def summary_mismatch(data: dict) -> float:
    """The aggregator's per-protein channel mean of stress against the channel table."""
    ok = data["channels"][data["channels"]["status"] == "ok"]
    recomputed = ok.groupby(["layer", "protein_id"])["stress"].mean().rename("recomputed").reset_index()
    merged = data["summary"].merge(recomputed, on=["layer", "protein_id"])
    return float((merged["stress_mean"] - merged["recomputed"]).abs().max())


def main() -> None:
    data = load()
    counts = bootstrap_counts(n_proteins=len(data["cohort"]))
    summary = summary_by_layer(data, counts)
    channels = channel_summary(data)
    summary.to_csv(ROOT / "summary_by_layer.csv", index=False)
    channels.to_csv(ROOT / "channel_summary.csv", index=False)
    snapshots = [x for x in SNAPSHOT_LAYERS if x in data["layers"]] or data["layers"]
    references = data["references"]
    floor = float(references[(references["reference"] == "true_distance")
                             & (references["shift"] == "minshift")]["stress"].mean())
    figure_metric_by_layer(summary, data["layers"])
    figure_heatmap(channels, data["layers"])
    figure_scatter(channels, snapshots, floor)
    figure_distributions(data["channels"], snapshots)
    write_results(data, summary, channels)
    write_sanity(data, summary)
    print(f"wrote {ROOT}")


if __name__ == "__main__":
    main()

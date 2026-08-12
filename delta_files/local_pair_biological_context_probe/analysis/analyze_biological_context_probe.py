#!/usr/bin/env python3
"""Create compact summaries and figures for the biological-context probe."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ANALYSIS_DIR = Path(__file__).resolve().parent
RESULTS_DIR = ANALYSIS_DIR.parent

TRANSITIONS = ((10, 11), (15, 16), (20, 21), (30, 31), (40, 41))
TRANSITION_LABELS = [f"{start}→{end}" for start, end in TRANSITIONS]

BUNDLES = (
    "controls",
    "endpoint_biology",
    "local_biology",
    "local_triangle",
    "biology_local_triangle",
    "biology_local_triangle_mediator",
    "biology_local_triangle_plus_center",
)
BUNDLE_LABELS = {
    "controls": "Controls",
    "endpoint_biology": "Endpoint\nbiology",
    "local_biology": "Local\nbiology",
    "local_triangle": "Local\ntriangle",
    "biology_local_triangle": "Biology +\ntriangle",
    "biology_local_triangle_mediator": "+ mediator\nidentity",
    "biology_local_triangle_plus_center": "+ pair-center\nchannels",
}
BUNDLE_COLORS = {
    "controls": "#8c8c8c",
    "endpoint_biology": "#4c78a8",
    "local_biology": "#72b7b2",
    "local_triangle": "#f58518",
    "biology_local_triangle": "#54a24b",
    "biology_local_triangle_mediator": "#b279a2",
    "biology_local_triangle_plus_center": "#e45756",
}

ESTIMATORS = {
    "classification": ("logistic", "decision_tree_depth4", "hist_gradient_boosting"),
    "regression": ("ridge", "decision_tree_depth4", "hist_gradient_boosting"),
}
ESTIMATOR_LABELS = {
    "logistic": "Logistic",
    "ridge": "Ridge",
    "decision_tree_depth4": "Depth-4 tree",
    "hist_gradient_boosting": "HistGB",
}
ESTIMATOR_COLORS = {
    "logistic": "#4c78a8",
    "ridge": "#4c78a8",
    "decision_tree_depth4": "#f58518",
    "hist_gradient_boosting": "#54a24b",
}

CONTRASTS = (
    "local_biology_minus_endpoint_biology",
    "local_triangle_minus_controls",
    "biology_local_triangle_minus_local_triangle",
    "biology_local_triangle_mediator_minus_biology_local_triangle",
    "biology_local_triangle_plus_center_minus_biology_local_triangle",
)
CONTRAST_LABELS = {
    "local_biology_minus_endpoint_biology": "Nearby sequence\ncomposition",
    "local_triangle_minus_controls": "Local triangle\npressure",
    "biology_local_triangle_minus_local_triangle": "Biology beyond\ntriangle geometry",
    "biology_local_triangle_mediator_minus_biology_local_triangle": "Strongest mediator\nidentity",
    "biology_local_triangle_plus_center_minus_biology_local_triangle": "Pair-center\nchannels",
}

FAMILY_LABELS = {
    "controls": "Controls",
    "endpoint_aa": "Endpoint identity",
    "endpoint_properties": "Endpoint properties",
    "pair_chemistry": "Pair chemistry",
    "local_biology": "Local biology",
    "triangle_pressure": "Triangle pressure",
    "mediator_identity": "Mediator identity",
    "center_representation": "Pair-center channels",
}
FAMILY_COLORS = {
    "controls": "#8c8c8c",
    "endpoint_aa": "#4c78a8",
    "endpoint_properties": "#72b7b2",
    "pair_chemistry": "#b279a2",
    "local_biology": "#54a24b",
    "triangle_pressure": "#f58518",
    "mediator_identity": "#ff9da6",
    "center_representation": "#e45756",
}


def configure_style() -> None:
    plt.rcParams.update({
        "figure.dpi": 140,
        "savefig.dpi": 220,
        "font.size": 9,
        "axes.titlesize": 11,
        "axes.labelsize": 9,
        "legend.fontsize": 8,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "axes.grid": False,
    })


def save_figure(fig: plt.Figure, name: str) -> None:
    fig.savefig(ANALYSIS_DIR / name, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def load_results() -> dict[str, pd.DataFrame]:
    names = (
        "model_summary",
        "feature_ablation_summary",
        "feature_family_importance",
        "change_statistics",
    )
    return {name: pd.read_csv(RESULTS_DIR / f"{name}.csv") for name in names}


def write_headline_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "classification": "mean_per_protein_balanced_accuracy",
        "regression": "mean_per_protein_r2",
    }
    rows = []
    test = model_summary[
        (model_summary["split"] == "test")
        & (model_summary["estimator"] != "decision_tree_depth6")
    ]
    for target, metric in metrics.items():
        selected = test[test["target"] == target]
        for (estimator, bundle), group in selected.groupby(["estimator", "bundle"]):
            values = group[metric].to_numpy(dtype=np.float64)
            rows.append({
                "target": target,
                "metric": metric,
                "estimator": estimator,
                "bundle": bundle,
                "mean_across_transitions": np.mean(values),
                "minimum_across_transitions": np.min(values),
                "maximum_across_transitions": np.max(values),
                "n_transitions": values.size,
            })
    result = pd.DataFrame(rows).sort_values(["target", "estimator", "bundle"])
    result.to_csv(ANALYSIS_DIR / "headline_performance_summary.csv", index=False)
    return result


def plot_headline_performance(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(12, 8), constrained_layout=True)
    panels = (
        ("classification", "Mean per-protein balanced accuracy", 0.5),
        ("regression", r"Mean per-protein $R^2$", 0.0),
    )
    x = np.arange(len(BUNDLES), dtype=np.float64)
    width = 0.24
    for axis, (target, ylabel, baseline) in zip(axes, panels):
        estimators = ESTIMATORS[target]
        for offset_index, estimator in enumerate(estimators):
            selected = summary[
                (summary["target"] == target) & (summary["estimator"] == estimator)
            ].set_index("bundle").reindex(BUNDLES)
            center = x + (offset_index - 1) * width
            mean = selected["mean_across_transitions"].to_numpy()
            lower = mean - selected["minimum_across_transitions"].to_numpy()
            upper = selected["maximum_across_transitions"].to_numpy() - mean
            axis.bar(
                center,
                mean,
                width=width,
                label=ESTIMATOR_LABELS[estimator],
                color=ESTIMATOR_COLORS[estimator],
                alpha=0.86,
                yerr=np.vstack((lower, upper)),
                error_kw={"elinewidth": 0.8, "capsize": 2, "capthick": 0.8},
            )
        axis.axhline(baseline, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
        axis.set_ylabel(ylabel)
        axis.set_xticks(x, [BUNDLE_LABELS[bundle] for bundle in BUNDLES])
        axis.legend(frameon=False, ncol=3, loc="upper left")
        axis.set_title(
            "Shrink-versus-expand classification" if target == "classification"
            else "Continuous decoded-distance change regression"
        )
    fig.suptitle(
        "Held-out biological-context probe performance\n"
        "Bars average five transitions; whiskers span the transition minimum and maximum",
        fontsize=13,
    )
    save_figure(fig, "biological_context_headline_performance.png")


def write_ablation_summary(ablation: pd.DataFrame) -> pd.DataFrame:
    data = ablation.copy()
    data["ci_excludes_zero_positive"] = data["ci_lower_95"] > 0.0
    data["ci_excludes_zero_negative"] = data["ci_upper_95"] < 0.0
    result = (
        data.groupby(["target", "estimator", "contrast"], as_index=False)
        .agg(
            mean_point_gain=("point_gain", "mean"),
            minimum_point_gain=("point_gain", "min"),
            maximum_point_gain=("point_gain", "max"),
            transitions_with_positive_95ci=("ci_excludes_zero_positive", "sum"),
            transitions_with_negative_95ci=("ci_excludes_zero_negative", "sum"),
            n_transitions=("point_gain", "size"),
        )
        .sort_values(["target", "contrast", "estimator"])
    )
    result.to_csv(ANALYSIS_DIR / "ablation_gain_summary.csv", index=False)
    return result


def plot_ablation_gains(ablation: pd.DataFrame) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(18, 7), sharex="row", constrained_layout=True)
    for row_index, target in enumerate(("classification", "regression")):
        for column_index, contrast in enumerate(CONTRASTS):
            axis = axes[row_index, column_index]
            selected = ablation[
                (ablation["target"] == target) & (ablation["contrast"] == contrast)
            ]
            for estimator in ESTIMATORS[target]:
                values = selected[selected["estimator"] == estimator].sort_values("layer_from")
                y = values["point_gain"].to_numpy()
                low = y - values["ci_lower_95"].to_numpy()
                high = values["ci_upper_95"].to_numpy() - y
                axis.errorbar(
                    np.arange(len(TRANSITIONS)),
                    y,
                    yerr=np.vstack((low, high)),
                    marker="o",
                    markersize=3.5,
                    linewidth=1.1,
                    capsize=2,
                    color=ESTIMATOR_COLORS[estimator],
                    label=ESTIMATOR_LABELS[estimator],
                )
            axis.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
            axis.set_xticks(np.arange(len(TRANSITIONS)), TRANSITION_LABELS, rotation=45)
            axis.set_title(CONTRAST_LABELS[contrast])
            if column_index == 0:
                axis.set_ylabel(
                    "Gain in balanced accuracy" if target == "classification"
                    else r"Gain in mean per-protein $R^2$"
                )
            if row_index == 0 and column_index == 4:
                axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle(
        "Declared feature-ablation gains on held-out proteins\n"
        "Points are transition-specific gains; whiskers are paired 95% protein-bootstrap intervals",
        fontsize=13,
    )
    save_figure(fig, "biological_context_ablation_gains.png")


def write_transition_summary(model_summary: pd.DataFrame) -> pd.DataFrame:
    metrics = {
        "classification": "mean_per_protein_balanced_accuracy",
        "regression": "mean_per_protein_r2",
    }
    bundles = ("controls", "biology_local_triangle", "biology_local_triangle_plus_center")
    rows = []
    test = model_summary[model_summary["split"] == "test"]
    for target, metric in metrics.items():
        selected = test[
            (test["target"] == target)
            & test["estimator"].isin(ESTIMATORS[target])
            & test["bundle"].isin(bundles)
        ]
        for _, row in selected.iterrows():
            rows.append({
                "layer_from": row["layer_from"],
                "layer_to": row["layer_to"],
                "target": target,
                "metric": metric,
                "estimator": row["estimator"],
                "bundle": row["bundle"],
                "value": row[metric],
            })
    result = pd.DataFrame(rows).sort_values(
        ["target", "estimator", "bundle", "layer_from"]
    )
    result.to_csv(ANALYSIS_DIR / "transition_performance_summary.csv", index=False)
    return result


def plot_transition_performance(summary: pd.DataFrame) -> None:
    bundles = ("controls", "biology_local_triangle", "biology_local_triangle_plus_center")
    fig, axes = plt.subplots(2, 3, figsize=(13, 7), sharex=True, constrained_layout=True)
    for row_index, target in enumerate(("classification", "regression")):
        for column_index, estimator in enumerate(ESTIMATORS[target]):
            axis = axes[row_index, column_index]
            selected = summary[
                (summary["target"] == target) & (summary["estimator"] == estimator)
            ]
            for bundle in bundles:
                values = selected[selected["bundle"] == bundle].sort_values("layer_from")
                axis.plot(
                    np.arange(len(TRANSITIONS)),
                    values["value"],
                    marker="o",
                    linewidth=1.7,
                    color=BUNDLE_COLORS[bundle],
                    label=BUNDLE_LABELS[bundle].replace("\n", " "),
                )
            baseline = 0.5 if target == "classification" else 0.0
            axis.axhline(baseline, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
            axis.set_title(ESTIMATOR_LABELS[estimator])
            axis.set_xticks(np.arange(len(TRANSITIONS)), TRANSITION_LABELS)
            if column_index == 0:
                axis.set_ylabel(
                    "Balanced accuracy" if target == "classification"
                    else r"Mean per-protein $R^2$"
                )
            if row_index == 0 and column_index == 2:
                axis.legend(frameon=False, loc="upper left", bbox_to_anchor=(1.02, 1.0))
    fig.suptitle("Where interpretable and pair-representation signal appears across depth", fontsize=13)
    save_figure(fig, "biological_context_performance_by_transition.png")


def write_importance_summary(importance: pd.DataFrame) -> pd.DataFrame:
    result = (
        importance.groupby(
            ["bundle", "target", "estimator", "feature_family"], as_index=False
        )["metric_decrease"]
        .agg(
            mean_metric_decrease="mean",
            minimum_metric_decrease="min",
            maximum_metric_decrease="max",
            n_permutations="size",
        )
        .sort_values(["bundle", "target", "mean_metric_decrease"], ascending=[True, True, False])
    )
    result.to_csv(ANALYSIS_DIR / "feature_family_importance_summary.csv", index=False)
    return result


def plot_feature_importance(importance: pd.DataFrame) -> None:
    bundles = ("biology_local_triangle", "biology_local_triangle_plus_center")
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    for row_index, target in enumerate(("classification", "regression")):
        for column_index, bundle in enumerate(bundles):
            axis = axes[row_index, column_index]
            selected = importance[
                (importance["target"] == target) & (importance["bundle"] == bundle)
            ].sort_values("mean_metric_decrease")
            labels = [FAMILY_LABELS[value] for value in selected["feature_family"]]
            colors = [FAMILY_COLORS[value] for value in selected["feature_family"]]
            axis.barh(labels, selected["mean_metric_decrease"], color=colors, alpha=0.9)
            axis.axvline(0.0, color="black", linewidth=0.8)
            axis.set_xlabel(
                "Decrease in balanced accuracy" if target == "classification"
                else r"Decrease in $R^2$"
            )
            axis.set_title(
                ("Interpretable biology + triangle" if column_index == 0 else "+ pair-center channels")
                + ("\nClassification" if target == "classification" else "\nRegression")
            )
    fig.suptitle(
        "Held-out HistGB feature-family permutation importance\n"
        "Means across five transitions, three permutations each",
        fontsize=13,
    )
    save_figure(fig, "biological_context_feature_family_importance.png")


def write_target_summary(change_statistics: pd.DataFrame) -> pd.DataFrame:
    result = change_statistics[change_statistics["split"] == "test"].copy()
    meaningful = result["n_shrink"] + result["n_expand"]
    result["meaningful_change_coverage"] = meaningful / result["n_pairs"]
    result["shrink_fraction_among_meaningful"] = result["n_shrink"] / meaningful
    columns = (
        "layer_from", "layer_to", "n_pairs", "mean", "std", "q05", "q25",
        "median", "q75", "q95", "n_shrink", "n_expand", "n_excluded",
        "meaningful_change_coverage", "shrink_fraction_among_meaningful",
    )
    result = result.loc[:, columns]
    result.to_csv(ANALYSIS_DIR / "target_distribution_summary.csv", index=False)
    return result


def plot_target_distributions(summary: pd.DataFrame) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), constrained_layout=True)
    x = np.arange(len(summary), dtype=np.float64)

    left = axes[0]
    left.vlines(x, summary["q05"], summary["q95"], color="#9ecae1", linewidth=5, label="5th–95th percentile")
    left.vlines(x, summary["q25"], summary["q75"], color="#2171b5", linewidth=9, label="25th–75th percentile")
    left.scatter(x, summary["median"], color="white", edgecolor="black", zorder=3, s=28, label="Median")
    left.axhline(0.0, color="black", linewidth=0.8, linestyle="--", alpha=0.6)
    left.set_xticks(x, TRANSITION_LABELS)
    left.set_ylabel(r"Decoded distance change $\Delta d$ (Å)")
    left.set_title("Update distributions on held-out proteins")
    left.legend(frameon=False, loc="lower left")

    right = axes[1]
    coverage = summary["meaningful_change_coverage"].to_numpy()
    shrink_fraction = summary["shrink_fraction_among_meaningful"].to_numpy()
    right.bar(x, coverage, color="#4c78a8", alpha=0.86, label=r"Coverage: $|\Delta d|>0.5$ Å")
    right.plot(x, shrink_fraction, color="#e45756", marker="o", linewidth=1.8, label="Shrink fraction among included")
    right.axhline(0.5, color="black", linewidth=0.8, linestyle="--", alpha=0.55)
    right.set_ylim(0.0, 1.0)
    right.set_xticks(x, TRANSITION_LABELS)
    right.set_ylabel("Fraction of held-out pairs")
    right.set_title("Classification coverage and class balance")
    right.legend(frameon=False, loc="lower left")

    fig.suptitle("The scale and direction of decoded-distance updates vary with depth", fontsize=13)
    save_figure(fig, "biological_context_target_distributions.png")


def main() -> None:
    configure_style()
    data = load_results()

    headline = write_headline_summary(data["model_summary"])
    plot_headline_performance(headline)

    write_ablation_summary(data["feature_ablation_summary"])
    plot_ablation_gains(data["feature_ablation_summary"])

    transition = write_transition_summary(data["model_summary"])
    plot_transition_performance(transition)

    importance = write_importance_summary(data["feature_family_importance"])
    plot_feature_importance(importance)

    targets = write_target_summary(data["change_statistics"])
    plot_target_distributions(targets)

    print(f"Wrote biological-context analysis to {ANALYSIS_DIR}")


if __name__ == "__main__":
    main()

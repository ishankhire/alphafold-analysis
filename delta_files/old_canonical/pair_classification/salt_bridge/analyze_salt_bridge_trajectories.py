#!/usr/bin/env python3
"""Aggregate and visualize the completed 48-layer salt-bridge probes."""

from __future__ import annotations

import csv
import json
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


HERE = Path(__file__).resolve().parent
SOURCE_ROOT = (
    HERE.parents[1]
    / "pair_interaction_classification"
    / "cameo50_v1"
    / "classification"
    / "salt_bridge"
)
EXPECTED_LAYERS = tuple(range(48))
VARIANTS = (
    "ordinary",
    "diagonal_only",
    "cross_only",
    "cross_plus_diagonal",
    "ordinary_plus_cross",
    "ordinary_plus_cross_plus_diagonal",
)
LABELS = {
    "ordinary": "Ordinary",
    "diagonal_only": "Diagonal only",
    "cross_only": "Cross only",
    "cross_plus_diagonal": "Cross + diagonal",
    "ordinary_plus_cross": "Ordinary + cross",
    "ordinary_plus_cross_plus_diagonal": "Ordinary + cross + diagonal",
}
COLORS = {
    "ordinary": "#6b7280",
    "diagonal_only": "#e69f00",
    "cross_only": "#0072b2",
    "cross_plus_diagonal": "#009e73",
    "ordinary_plus_cross": "#cc79a7",
    "ordinary_plus_cross_plus_diagonal": "#d55e00",
}
LINESTYLES = {
    "ordinary": "--",
    "diagonal_only": "--",
    "cross_only": "-",
    "cross_plus_diagonal": "-",
    "ordinary_plus_cross": "-.",
    "ordinary_plus_cross_plus_diagonal": "-.",
}
CROSS_FAMILY = tuple(item for item in VARIANTS if "cross" in item)
GRID_SUBSETS = {
    "full_7": (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0),
    "drop_100": (1e-4, 1e-3, 1e-2, 1e-1, 1.0, 10.0),
    "compact_5": (1e-4, 1e-3, 1e-2, 1e-1, 1.0),
    "compact_4": (1e-4, 1e-3, 1e-2, 1e-1),
    "central_3": (1e-3, 1e-2, 1e-1),
    "central_2": (1e-3, 1e-2),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, rows: list[dict], fieldnames: list[str]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def validate_and_load() -> tuple[list[dict[str, str]], list[dict[str, str]]]:
    metric_rows: list[dict[str, str]] = []
    search_rows: list[dict[str, str]] = []
    reference_split = None
    for layer in EXPECTED_LAYERS:
        layer_dir = SOURCE_ROOT / f"layer_{layer}"
        required = (
            "classification_config.json",
            "interaction_type_summary.csv",
            "regularization_search.csv",
            "split_assignments.csv",
            "summary.txt",
        )
        missing = [name for name in required if not (layer_dir / name).is_file()]
        if missing:
            raise FileNotFoundError(f"layer {layer} is incomplete: {missing}")
        config = json.loads((layer_dir / "classification_config.json").read_text())
        if int(config["layer"]) != layer:
            raise ValueError(f"layer {layer} configuration reports layer {config['layer']}")
        split_bytes = (layer_dir / "split_assignments.csv").read_bytes()
        if reference_split is None:
            reference_split = split_bytes
        elif split_bytes != reference_split:
            raise ValueError(f"layer {layer} uses different protein splits")

        local_metrics = read_csv(layer_dir / "interaction_type_summary.csv")
        local_search = read_csv(layer_dir / "regularization_search.csv")
        if len(local_metrics) != 30 or len(local_search) != 42:
            raise ValueError(
                f"layer {layer}: expected 30 metric and 42 search rows, got "
                f"{len(local_metrics)} and {len(local_search)}"
            )
        if any(row["converged"] != "true" for row in local_search):
            raise ValueError(f"layer {layer} contains a non-converged fit")
        if Counter(row["model_variant"] for row in local_metrics) != Counter(
            {variant: 5 for variant in VARIANTS}
        ):
            raise ValueError(f"layer {layer} has incomplete metric variant coverage")
        metric_rows.extend(local_metrics)
        search_rows.extend(local_search)
    return metric_rows, search_rows


def metric_lookup(rows: list[dict[str, str]]) -> dict[tuple, dict[str, str]]:
    return {
        (
            int(row["layer"]),
            row["model_variant"],
            row["split"],
            row["evaluation_scope"],
        ): row
        for row in rows
    }


def trajectory_plot(
    lookup: dict[tuple, dict[str, str]],
    destination: Path,
    split: str,
    scope: str,
    panels: tuple[tuple[str, str], ...],
    title: str,
) -> None:
    columns = 2
    rows = int(np.ceil(len(panels) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(14, 4.2 * rows), sharex=True)
    axes = np.atleast_1d(axes).ravel()
    for axis, (field, panel_title) in zip(axes, panels):
        for variant in VARIANTS:
            values = [
                float(lookup[(layer, variant, split, scope)][field])
                for layer in EXPECTED_LAYERS
            ]
            axis.plot(
                EXPECTED_LAYERS,
                values,
                label=LABELS[variant],
                color=COLORS[variant],
                linestyle=LINESTYLES[variant],
                linewidth=2.0,
                alpha=0.95,
            )
        axis.set_title(panel_title)
        axis.set_ylim(0.0, 1.01)
        axis.set_ylabel("Score")
        axis.set_xticks(range(0, 48, 5))
        axis.grid(alpha=0.25)
    for axis in axes[len(panels) :]:
        axis.set_visible(False)
    for axis in axes[-columns:]:
        axis.set_xlabel("Pair-representation layer")
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.suptitle(title, fontsize=15)
    figure.tight_layout(rect=(0, 0.09, 1, 0.96))
    figure.savefig(destination, dpi=180)
    plt.close(figure)


def hyperparameter_outputs(search_rows: list[dict[str, str]]) -> None:
    score = {
        (int(row["layer"]), row["model_variant"], float(row["C"])): float(
            row["validation_mean_per_protein_average_precision"]
        )
        for row in search_rows
    }
    c_grid = sorted({key[2] for key in score})
    best = {
        (layer, variant): max(score[layer, variant, c] for c in c_grid)
        for layer in EXPECTED_LAYERS
        for variant in VARIANTS
    }
    selected_counts = Counter(
        (row["model_variant"], float(row["C"]))
        for row in search_rows
        if row["selected"] == "true"
    )

    summary_rows = []
    for variant in VARIANTS:
        for c_value in c_grid:
            values = np.asarray(
                [score[layer, variant, c_value] for layer in EXPECTED_LAYERS]
            )
            regrets = np.asarray(
                [best[layer, variant] - score[layer, variant, c_value] for layer in EXPECTED_LAYERS]
            )
            summary_rows.append(
                {
                    "model_variant": variant,
                    "C": f"{c_value:.4g}",
                    "selected_layer_count": selected_counts[variant, c_value],
                    "selection_fraction": f"{selected_counts[variant, c_value] / 48:.8f}",
                    "mean_validation_macro_ap": f"{values.mean():.8f}",
                    "std_validation_macro_ap": f"{values.std():.8f}",
                    "min_validation_macro_ap": f"{values.min():.8f}",
                    "max_validation_macro_ap": f"{values.max():.8f}",
                    "mean_regret_to_layer_best": f"{regrets.mean():.8f}",
                    "max_regret_to_layer_best": f"{regrets.max():.8f}",
                }
            )
    write_csv(
        HERE / "hyperparameter_summary.csv",
        summary_rows,
        list(summary_rows[0]),
    )

    grid_rows = []
    groups = {
        "cross_family": CROSS_FAMILY,
        "all_representation_variants": tuple(v for v in VARIANTS if v != "ordinary"),
    }
    for grid_name, subset in GRID_SUBSETS.items():
        for group_name, variants in groups.items():
            regrets = []
            for layer in EXPECTED_LAYERS:
                for variant in variants:
                    subset_best = max(score[layer, variant, c] for c in subset)
                    regrets.append(best[layer, variant] - subset_best)
            values = np.asarray(regrets)
            grid_rows.append(
                {
                    "grid_name": grid_name,
                    "C_values": ";".join(f"{value:.4g}" for value in subset),
                    "n_C_values": len(subset),
                    "variant_group": group_name,
                    "n_layer_variant_cases": values.size,
                    "fraction_same_optimum_as_full_grid": f"{np.mean(values <= 1e-12):.8f}",
                    "mean_validation_macro_ap_regret": f"{values.mean():.8f}",
                    "p95_validation_macro_ap_regret": f"{np.quantile(values, 0.95):.8f}",
                    "max_validation_macro_ap_regret": f"{values.max():.8f}",
                }
            )
    write_csv(HERE / "reduced_grid_analysis.csv", grid_rows, list(grid_rows[0]))

    transfer_rows = []
    for source_layer in EXPECTED_LAYERS:
        chosen = {
            variant: max(c_grid, key=lambda c: score[source_layer, variant, c])
            for variant in CROSS_FAMILY
        }
        regrets = np.asarray(
            [
                best[target_layer, variant]
                - score[target_layer, variant, chosen[variant]]
                for variant in CROSS_FAMILY
                for target_layer in EXPECTED_LAYERS
            ]
        )
        transfer_rows.append(
            {
                "source_layer": source_layer,
                **{f"selected_C_{variant}": f"{chosen[variant]:.4g}" for variant in CROSS_FAMILY},
                "n_target_layer_variant_cases": regrets.size,
                "mean_validation_macro_ap_regret": f"{regrets.mean():.8f}",
                "p95_validation_macro_ap_regret": f"{np.quantile(regrets, 0.95):.8f}",
                "max_validation_macro_ap_regret": f"{regrets.max():.8f}",
            }
        )
    write_csv(
        HERE / "single_layer_C_transfer.csv",
        transfer_rows,
        list(transfer_rows[0]),
    )

    figure, axes = plt.subplots(3, 2, figsize=(15, 12), sharex=True)
    log_grid = np.log10(c_grid)
    for axis, variant in zip(axes.ravel(), VARIANTS):
        regret_matrix = np.asarray(
            [
                [best[layer, variant] - score[layer, variant, c] for layer in EXPECTED_LAYERS]
                for c in c_grid
            ]
        )
        image = axis.imshow(
            regret_matrix,
            aspect="auto",
            origin="lower",
            interpolation="nearest",
            extent=(-0.5, 47.5, log_grid[0] - 0.5, log_grid[-1] + 0.5),
            cmap="magma_r",
            vmin=0.0,
            vmax=0.04,
        )
        axis.set_title(LABELS[variant])
        axis.set_yticks(log_grid)
        axis.set_yticklabels([f"{c:g}" for c in c_grid])
        axis.set_ylabel("C")
        axis.set_xticks(range(0, 48, 5))
        axis.set_xlabel("Layer")
    figure.subplots_adjust(left=0.07, right=0.86, bottom=0.07, top=0.93, hspace=0.30)
    colorbar_axis = figure.add_axes((0.89, 0.16, 0.018, 0.68))
    colorbar = figure.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Validation macro-AP loss relative to best C at that layer")
    figure.suptitle("Regularization sensitivity across layers", fontsize=15)
    figure.savefig(HERE / "hyperparameter_sensitivity.png", dpi=180)
    plt.close(figure)

    figure, axis = plt.subplots(figsize=(13, 6))
    width = 0.12
    x_values = np.arange(len(c_grid))
    for index, variant in enumerate(VARIANTS):
        counts = [selected_counts[variant, c] for c in c_grid]
        axis.bar(
            x_values + (index - 2.5) * width,
            counts,
            width=width,
            label=LABELS[variant],
            color=COLORS[variant],
        )
    axis.set_xticks(x_values)
    axis.set_xticklabels([f"{value:g}" for value in c_grid])
    axis.set_xlabel("Selected C")
    axis.set_ylabel("Number of layers (out of 48)")
    axis.set_title("Validation-selected regularization values across layers")
    axis.grid(axis="y", alpha=0.25)
    axis.legend(ncol=2, frameon=False)
    figure.tight_layout()
    figure.savefig(HERE / "selected_C_counts.png", dpi=180)
    plt.close(figure)


def selected_hyperparameter_plot(lookup: dict[tuple, dict[str, str]]) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(14, 9), sharex=True)
    for variant in VARIANTS:
        selected_c = [
            float(lookup[layer, variant, "validation", "matched"]["selected_C"])
            for layer in EXPECTED_LAYERS
        ]
        thresholds = [
            float(lookup[layer, variant, "validation", "matched"]["selected_threshold"])
            for layer in EXPECTED_LAYERS
        ]
        axes[0].plot(
            EXPECTED_LAYERS,
            selected_c,
            label=LABELS[variant],
            color=COLORS[variant],
            linestyle=LINESTYLES[variant],
            linewidth=2,
        )
        axes[1].plot(
            EXPECTED_LAYERS,
            thresholds,
            label=LABELS[variant],
            color=COLORS[variant],
            linestyle=LINESTYLES[variant],
            linewidth=2,
        )
    axes[0].set_yscale("log")
    axes[0].set_ylabel("Selected C (log scale)")
    axes[0].set_title("L2 regularization selected by validation macro AP")
    axes[1].set_ylabel("Selected probability threshold")
    axes[1].set_xlabel("Pair-representation layer")
    axes[1].set_title("Threshold selected by validation macro balanced accuracy")
    axes[1].set_ylim(0.0, 1.01)
    for axis in axes:
        axis.set_xticks(range(0, 48, 5))
        axis.grid(alpha=0.25)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="lower center", ncol=3, frameon=False)
    figure.tight_layout(rect=(0, 0.08, 1, 1))
    figure.savefig(HERE / "selected_hyperparameters.png", dpi=180)
    plt.close(figure)


def main() -> None:
    metric_rows, search_rows = validate_and_load()
    write_csv(HERE / "layer_metrics.csv", metric_rows, list(metric_rows[0]))
    write_csv(HERE / "hyperparameter_search.csv", search_rows, list(search_rows[0]))
    lookup = metric_lookup(metric_rows)
    trajectory_plot(
        lookup,
        HERE / "matched_test_trajectories.png",
        "test",
        "matched",
        (
            ("pair_pooled_auroc", "Pooled AUROC"),
            ("pair_pooled_average_precision", "Pooled average precision"),
            ("mean_per_protein_auroc", "Mean per-protein AUROC"),
            ("mean_per_protein_average_precision", "Mean per-protein average precision"),
            ("pair_pooled_balanced_accuracy", "Balanced accuracy"),
            ("pair_pooled_precision", "Precision"),
            ("pair_pooled_recall", "Recall"),
        ),
        "Matched held-out test trajectories",
    )
    trajectory_plot(
        lookup,
        HERE / "natural_test_trajectories.png",
        "test",
        "natural_candidate_pool",
        (
            ("pair_pooled_auroc", "Pooled AUROC"),
            ("pair_pooled_average_precision", "Pooled average precision"),
            ("mean_per_protein_auroc", "Mean per-protein AUROC"),
            ("mean_per_protein_average_precision", "Mean per-protein average precision"),
        ),
        "Natural candidate-pool held-out test trajectories",
    )
    trajectory_plot(
        lookup,
        HERE / "validation_matched_trajectories.png",
        "validation",
        "matched",
        (
            ("pair_pooled_auroc", "Pooled AUROC"),
            ("pair_pooled_average_precision", "Pooled average precision"),
            ("mean_per_protein_auroc", "Mean per-protein AUROC"),
            ("mean_per_protein_average_precision", "Selection metric: macro AP"),
            ("pair_pooled_balanced_accuracy", "Balanced accuracy at selected threshold"),
        ),
        "Matched validation trajectories",
    )
    selected_hyperparameter_plot(lookup)
    hyperparameter_outputs(search_rows)
    print(f"Wrote salt-bridge trajectory analysis to {HERE}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Plot layer-to-layer probe changes against layerwise-ablation TM changes."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Mapping


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from plot_layerwise_ablation_tm_scores import (
    ABLATIONS,
    DIRECTIONS,
    EXPECTED_PROTEINS,
    INPUT_DIR,
    LAYERS,
    MODEL_SPECS,
    OUTPUT_DIR,
    configure_style,
    load_cohort,
    load_probe_means,
    load_tm_score_matrix,
)


SHARED_LINEAR_MODEL = "shared_ols_base256"
TRANSITION_LAYERS = np.asarray(LAYERS[1:], dtype=int)
ANNOTATED_LAYERS = frozenset({1, 5, 10, 15, 20, 25, 30, 35, 40, 45, 47})


def compute_layer_changes(
    matrix: np.ndarray, triangle_means: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    """Return changes for each transition from layer L-1 to layer L."""
    matrix = np.asarray(matrix, dtype=float)
    triangle_means = np.asarray(triangle_means, dtype=float)
    expected_shape = (EXPECTED_PROTEINS, len(LAYERS))
    if matrix.shape != expected_shape:
        raise ValueError(f"TM-score matrix has shape {matrix.shape}; expected {expected_shape}")
    if triangle_means.shape != (len(LAYERS),):
        raise ValueError("Triangle trajectory must contain one value for each layer")
    if not np.isfinite(matrix).all() or not np.isfinite(triangle_means).all():
        raise ValueError("Layer-change inputs must be finite")

    triangle_change = np.diff(triangle_means)
    per_protein_tm_change = np.diff(matrix, axis=1)
    mean_tm_change = per_protein_tm_change.mean(axis=0)
    return triangle_change, mean_tm_change


def plot_change_scatter(
    *,
    triangle_change: np.ndarray,
    mean_tm_change: np.ndarray,
    label: str,
    direction: str,
    stem: str,
) -> tuple[Path, Path]:
    """Draw and save one connected layer-transition scatterplot."""
    if triangle_change.shape != TRANSITION_LAYERS.shape:
        raise ValueError("Triangle changes must contain 47 layer transitions")
    if mean_tm_change.shape != TRANSITION_LAYERS.shape:
        raise ValueError("TM-score changes must contain 47 layer transitions")

    figure, axis = plt.subplots(figsize=(8.2, 6.2))
    axis.axhline(0.0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    axis.axvline(0.0, color="#777777", linewidth=0.9, linestyle="--", zorder=0)
    axis.plot(
        triangle_change,
        mean_tm_change,
        color="#4A4A4A",
        linewidth=1.15,
        alpha=0.72,
        zorder=1,
    )
    points = axis.scatter(
        triangle_change,
        mean_tm_change,
        c=TRANSITION_LAYERS,
        cmap="viridis",
        vmin=1,
        vmax=47,
        s=42,
        edgecolor="white",
        linewidth=0.45,
        zorder=2,
    )
    for layer, x_value, y_value in zip(
        TRANSITION_LAYERS, triangle_change, mean_tm_change
    ):
        if int(layer) in ANNOTATED_LAYERS:
            axis.annotate(
                str(layer),
                (x_value, y_value),
                xytext=(4, 4),
                textcoords="offset points",
                fontsize=7.5,
                color="#303030",
                zorder=3,
            )

    model_label = str(MODEL_SPECS[SHARED_LINEAR_MODEL]["label"])
    axis.set(
        title=f"{label}: {DIRECTIONS[direction]} layerwise changes",
        xlabel=(
            r"Change in mean normalized triangle violation, "
            r"$\Delta_L$ (shared linear)"
        ),
        ylabel=r"Mean per-protein TM-score change, $\Delta_L$",
    )
    axis.ticklabel_format(axis="both", style="sci", scilimits=(-2, 2))
    colorbar = figure.colorbar(points, ax=axis, pad=0.02)
    colorbar.set_label(r"Transition endpoint layer $L$ ($L-1 \rightarrow L$)")
    axis.text(
        0.01,
        0.99,
        f"{model_label}; {EXPECTED_PROTEINS} proteins",
        transform=axis.transAxes,
        ha="left",
        va="top",
        color="#555555",
        fontsize=9,
    )
    figure.tight_layout()

    png_path = OUTPUT_DIR / f"{stem}.png"
    pdf_path = OUTPUT_DIR / f"{stem}.pdf"
    figure.savefig(png_path)
    figure.savefig(pdf_path)
    plt.close(figure)
    return png_path, pdf_path


def expected_output_filenames() -> set[str]:
    names: set[str] = set()
    for ablation in ABLATIONS:
        for direction in DIRECTIONS:
            stem = f"{ablation['key']}_{direction}_tm_change_vs_triangle_change"
            names.update({f"{stem}.png", f"{stem}.pdf"})
    return names


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort_keys = load_cohort()
    probe_means: Mapping[str, Mapping[str, np.ndarray]] = load_probe_means(
        cohort_keys
    )
    triangle_means = probe_means["mean_relative_violation_all"][SHARED_LINEAR_MODEL]

    generated: list[Path] = []
    for ablation in ABLATIONS:
        for direction in DIRECTIONS:
            filename = f"{ablation['source_stem']}_{direction}.csv"
            keys, matrix = load_tm_score_matrix(INPUT_DIR / filename, cohort_keys)
            if not keys.equals(cohort_keys):
                raise ValueError(f"{filename} does not match the frozen protein cohort")
            triangle_change, mean_tm_change = compute_layer_changes(
                matrix, triangle_means
            )
            stem = f"{ablation['key']}_{direction}_tm_change_vs_triangle_change"
            generated.extend(
                plot_change_scatter(
                    triangle_change=triangle_change,
                    mean_tm_change=mean_tm_change,
                    label=str(ablation["label"]),
                    direction=direction,
                    stem=stem,
                )
            )

    generated_names = {path.name for path in generated}
    if generated_names != expected_output_filenames() or not all(
        path.is_file() for path in generated
    ):
        raise RuntimeError("The six expected PNG/PDF scatterplot pairs were not generated")
    print(
        "Generated 6 connected layer-change scatterplot pairs from "
        f"{EXPECTED_PROTEINS} proteins"
    )


if __name__ == "__main__":
    main()

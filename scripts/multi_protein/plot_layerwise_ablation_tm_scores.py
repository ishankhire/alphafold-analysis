#!/usr/bin/env python3
"""Render the combined Pair/MSA transition ablation figure used in draft_4.tex."""

from __future__ import annotations

import os
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import LineCollection


INPUT_DIR = PROJECT_ROOT / "delta_files/layerwise_ablation_csvs"
COHORT_PATH = (
    PROJECT_ROOT / "delta_files/regression_5fold_v1/metadata/regression_outer_folds_v1.csv"
)
OUTPUT_DIR = (
    PROJECT_ROOT
    / "delta_files/publication_plots/sept_2_plots/layerwise_ablation_trajectories"
)

LAYERS = tuple(range(48))
LAYER_COLUMNS = tuple(f"L{layer}" for layer in LAYERS)
EXPECTED_SOURCE_PROTEINS = 554
EXPECTED_PROTEINS = 552
EXPECTED_EXCLUSIONS = frozenset(
    {("ecod", "000196893"), ("ecod", "001681437")}
)
ABLATIONS = (
    {
        "label": "Pair Transition",
        "source_stem": "ablate_pair_transition_layerwise",
    },
    {
        "label": "MSA MLP",
        "source_stem": "ablate_msa_transition_layerwise",
    },
)
DIRECTIONS = {"fwd": "FWD", "rev": "REV"}
OUTPUT_STEM = "combined_layerwise_ablation_tm_score_trajectories"


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def normalize_protein_ids(values: pd.Series) -> pd.Series:
    normalized = values.astype("string").str.strip().str.zfill(9)
    if normalized.isna().any() or normalized.eq("").any():
        raise ValueError("Protein IDs contain missing or empty values")
    return normalized


def load_cohort() -> pd.MultiIndex:
    frame = pd.read_csv(COHORT_PATH, dtype={"protein_id": "string"})
    required = {"source", "protein_id"}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"Cohort manifest is missing columns: {missing}")
    frame["source"] = frame["source"].astype("string").str.strip().str.lower()
    frame["protein_id"] = normalize_protein_ids(frame["protein_id"])
    if len(frame) != EXPECTED_PROTEINS:
        raise ValueError(f"Cohort has {len(frame)} rows; expected {EXPECTED_PROTEINS}")
    if frame.duplicated(["source", "protein_id"]).any():
        raise ValueError("Cohort contains duplicate source/protein keys")
    return pd.MultiIndex.from_frame(
        frame.sort_values(["source", "protein_id"], kind="stable")[
            ["source", "protein_id"]
        ]
    )


def load_tm_score_matrix(path: Path, cohort: pd.MultiIndex) -> np.ndarray:
    frame = pd.read_csv(path, dtype={"protein": "string"})
    required = {"dataset", "protein", "experiment", *LAYER_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if len(frame) != EXPECTED_SOURCE_PROTEINS:
        raise ValueError(
            f"{path.name} has {len(frame)} rows; expected {EXPECTED_SOURCE_PROTEINS}"
        )
    frame["source"] = frame["dataset"].astype("string").str.strip().str.lower()
    frame["protein_id"] = normalize_protein_ids(frame["protein"])
    if frame.duplicated(["source", "protein_id"]).any():
        raise ValueError(f"{path.name} contains duplicate source/protein keys")

    source_keys = pd.MultiIndex.from_frame(frame[["source", "protein_id"]])
    missing_cohort = set(cohort.tolist()) - set(source_keys.tolist())
    exclusions = set(source_keys.tolist()) - set(cohort.tolist())
    if missing_cohort:
        raise ValueError(f"{path.name} is missing frozen-cohort proteins")
    if exclusions != EXPECTED_EXCLUSIONS:
        raise ValueError(f"{path.name} has unexpected cohort exclusions: {exclusions}")

    matrix = (
        frame.set_index(["source", "protein_id"])
        .reindex(cohort)
        .loc[:, LAYER_COLUMNS]
        .to_numpy(dtype=float)
    )
    if matrix.shape != (EXPECTED_PROTEINS, len(LAYERS)):
        raise ValueError(f"{path.name} has an unexpected TM-score shape")
    if not np.isfinite(matrix).all() or np.any((matrix < 0) | (matrix > 1)):
        raise ValueError(f"{path.name} contains invalid TM-scores")
    return matrix


def draw_trajectory(axis: plt.Axes, matrix: np.ndarray) -> None:
    layers = np.asarray(LAYERS)
    segments = np.stack((np.broadcast_to(layers, matrix.shape), matrix), axis=-1)
    axis.add_collection(
        LineCollection(
            segments,
            colors="#8A8A8A",
            linewidths=0.55,
            alpha=0.10,
            rasterized=True,
        )
    )
    axis.plot(
        layers,
        matrix.mean(axis=0),
        color="black",
        linewidth=2.4,
        marker="o",
        markersize=3.5,
        markeredgewidth=0,
        zorder=5,
    )


def render(panels: list[tuple[str, str, np.ndarray]]) -> Path:
    if len(panels) != 4:
        raise ValueError(f"Expected four panels; received {len(panels)}")
    figure, axes = plt.subplots(1, 4, figsize=(14, 4.8), sharex=True, sharey=True)
    for index, (label, direction, matrix) in enumerate(panels):
        axis = axes[index]
        draw_trajectory(axis, matrix)
        axis.set(
            xlim=(0, 47),
            ylim=(0, 1),
            title=f"({chr(ord('a') + index)}) {label}\n({DIRECTIONS[direction]})",
        )
        axis.set_xticks((0, 10, 20, 30, 40, 47))
        axis.tick_params(axis="both", labelsize=17)
        axis.title.set_size(20)
    figure.supxlabel("Layer", fontsize=25, y=0.025)
    figure.supylabel("TM-score", fontsize=25, x=0.015)
    figure.subplots_adjust(
        left=0.085, right=0.995, bottom=0.22, top=0.76, wspace=0.08
    )
    output = OUTPUT_DIR / f"{OUTPUT_STEM}.pdf"
    figure.savefig(output)
    plt.close(figure)
    return output


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    cohort = load_cohort()
    panels = []
    for ablation in ABLATIONS:
        for direction in DIRECTIONS:
            path = INPUT_DIR / f"{ablation['source_stem']}_{direction}.csv"
            panels.append((str(ablation["label"]), direction, load_tm_score_matrix(path, cohort)))
    output = render(panels)
    actual = {path.name for path in OUTPUT_DIR.glob("*") if path.is_file()}
    if actual != {output.name}:
        raise RuntimeError(f"Unexpected ablation output inventory: {sorted(actual)}")
    print(f"Generated {output} from four 554-protein ablation tables")


if __name__ == "__main__":
    main()

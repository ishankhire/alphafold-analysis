#!/usr/bin/env python3
"""Plot individual-protein trajectories for six selected layerwise ablations."""

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


INPUT_DIR = PROJECT_ROOT / "delta_files/layerwise_ablation_csvs"
OUTPUT_DIR = (
    PROJECT_ROOT
    / "delta_files/publication_plots/sept_2_plots/layerwise_ablation_trajectories"
)
LAYERS = tuple(range(48))
LAYER_COLUMNS = tuple(f"L{layer}" for layer in LAYERS)
EXPECTED_PROTEINS = 554

ABLATIONS = (
    {
        "key": "pair_transition_mlp",
        "label": "Pair transition MLPs",
        "source_stem": "ablate_pair_transition_layerwise",
    },
    {
        "key": "msa_mlp",
        "label": "MSA MLPs",
        "source_stem": "ablate_msa_transition_layerwise",
    },
    {
        "key": "tmu_triangle_attention",
        "label": "TMU + triangle attention",
        "source_stem": "ablate_tri_start_ablate_tri_end_ablate_tmu_layerwise",
    },
)
DIRECTIONS = {
    "fwd": "FWD",
    "rev": "REV",
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
            "legend.fontsize": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.22,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def load_tm_score_matrix(path: Path) -> tuple[pd.MultiIndex, np.ndarray]:
    """Load one wide TM-score table and return a consistently ordered matrix."""
    if not path.is_file():
        raise FileNotFoundError(f"Missing required ablation input: {path}")

    frame = pd.read_csv(path)
    required = {"dataset", "protein", "experiment", *LAYER_COLUMNS}
    missing = sorted(required - set(frame.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")
    if len(frame) != EXPECTED_PROTEINS:
        raise ValueError(
            f"{path.name} has {len(frame)} proteins; expected {EXPECTED_PROTEINS}"
        )
    if frame.duplicated(["dataset", "protein"]).any():
        raise ValueError(f"{path.name} contains duplicate dataset/protein keys")

    frame = frame.sort_values(["dataset", "protein"], kind="stable").reset_index(drop=True)
    matrix = frame.loc[:, LAYER_COLUMNS].to_numpy(dtype=float)
    if matrix.shape != (EXPECTED_PROTEINS, len(LAYERS)):
        raise ValueError(f"{path.name} has an unexpected TM-score matrix shape")
    if not np.isfinite(matrix).all():
        raise ValueError(f"{path.name} contains nonfinite TM-scores")
    if np.any((matrix < 0.0) | (matrix > 1.0)):
        raise ValueError(f"{path.name} contains TM-scores outside [0, 1]")

    keys = pd.MultiIndex.from_frame(frame[["dataset", "protein"]])
    return keys, matrix


def plot_trajectory(
    *,
    matrix: np.ndarray,
    label: str,
    direction: str,
    stem: str,
) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(8.4, 5.3))
    layers = np.asarray(LAYERS)

    for protein_values in matrix:
        axis.plot(
            layers,
            protein_values,
            color="#8A8A8A",
            linewidth=0.55,
            alpha=0.12,
            rasterized=True,
        )
    axis.plot(
        layers,
        matrix.mean(axis=0),
        color="black",
        linewidth=2.4,
        marker="o",
        markersize=3.8,
        markerfacecolor="black",
        markeredgecolor="black",
        markeredgewidth=0,
        zorder=5,
        label="Mean",
    )

    axis.set(
        title=f"{label}: {DIRECTIONS[direction]} layerwise ablation",
        xlabel="Ablation boundary layer",
        ylabel="TM-score",
        xlim=(0, 47),
        ylim=(0, 1),
    )
    axis.set_xticks((0, 10, 20, 30, 40, 47))
    axis.legend(loc="best", frameon=True)
    axis.text(
        0.99,
        0.02,
        f"{matrix.shape[0]} proteins",
        transform=axis.transAxes,
        ha="right",
        va="bottom",
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


def write_readme() -> None:
    text = f"""# Selected layerwise-ablation TM-score trajectories

This folder contains six conceptual figures: FWD and REV layerwise
ablations for Pair transition MLPs, MSA MLPs (`msa_transition`), and the combined
TMU + triangle-attention (`tri_start`, `tri_end`, and `tmu`) intervention. Each
figure is supplied as both PNG and PDF.

Each graph plots all {EXPECTED_PROTEINS} proteins individually as thin,
semi-transparent gray lines. The thicker black line is the protein-equal mean,
with a black dot at every layer; no uncertainty band is shown. The x-axis is the
boundary layer recorded by each FWD or REV source CSV.

Figures are regenerated by
`scripts/multi_protein/plot_layerwise_ablation_tm_scores.py`.
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="utf-8")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    cohort_keys: pd.MultiIndex | None = None
    generated: list[Path] = []
    for ablation in ABLATIONS:
        for direction in DIRECTIONS:
            filename = f"{ablation['source_stem']}_{direction}.csv"
            keys, matrix = load_tm_score_matrix(INPUT_DIR / filename)
            if cohort_keys is None:
                cohort_keys = keys
            elif not keys.equals(cohort_keys):
                raise ValueError(f"{filename} does not match the shared protein cohort")

            stem = f"{ablation['key']}_{direction}_tm_score_trajectory"
            generated.extend(
                plot_trajectory(
                    matrix=matrix,
                    label=str(ablation["label"]),
                    direction=direction,
                    stem=stem,
                )
            )
    write_readme()
    print(f"Generated {len(generated) // 2} figures in PNG and PDF: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()

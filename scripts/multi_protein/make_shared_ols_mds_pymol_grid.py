#!/usr/bin/env python3
"""Compose the shared-OLS PyMOL panels into the supplemental figure grid."""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


EXPORT_ROOT = PROJECT_ROOT / "delta_files" / "shared_ols_mds_pdb_exports"
MANIFEST_PATH = EXPORT_ROOT / "run_manifest.json"
PANEL_ROOT = EXPORT_ROOT / "pymol_renders"
OUTPUT_NAME = "shared_ols_mds_structures_visualized.png"
CANONICAL_OUTPUT = EXPORT_ROOT / OUTPUT_NAME
OVERLEAF_OUTPUTS = (
    PROJECT_ROOT / "overleaf" / "figures" / OUTPUT_NAME,
    PROJECT_ROOT / "overleaf" / "supplemental_figures" / OUTPUT_NAME,
)

COLUMNS = (
    ("SM", None),
    ("L0", 0),
    ("L10", 10),
    ("L20", 20),
    ("L30", 30),
    ("L47", 47),
)

DPI = 300
CROP_PADDING = 28
FIG_WIDTH = 13.8
ROW_HEIGHT = 1.45
FONT_SIZE = 10
LABEL_SIZE = 10


def load_protein_ids() -> tuple[str, ...]:
    with MANIFEST_PATH.open() as handle:
        manifest = json.load(handle)
    protein_ids = manifest.get("protein_ids")
    if manifest.get("run_status") != "complete":
        raise ValueError("The shared-OLS MDS export manifest is not complete")
    if not isinstance(protein_ids, list) or len(protein_ids) != 10:
        raise ValueError(f"Expected ten protein IDs, found: {protein_ids!r}")
    return tuple(protein_ids)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": FONT_SIZE,
            "savefig.dpi": DPI,
            "savefig.facecolor": "white",
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )


def panel_path(protein_id: str, layer: int | None) -> Path:
    protein_dir = PANEL_ROOT / protein_id
    if layer is None:
        return protein_dir / f"{protein_id}_original_full.png"
    return protein_dir / f"{protein_id}_shared_ols_mds_layer_{layer:02d}.png"


def validate_inputs(protein_ids: tuple[str, ...]) -> None:
    expected_paths = {
        panel_path(protein_id, layer)
        for protein_id in protein_ids
        for _label, layer in COLUMNS
    }
    missing = sorted(path for path in expected_paths if not path.is_file())
    actual_paths = set(PANEL_ROOT.rglob("*.png"))
    unexpected = sorted(actual_paths - expected_paths)
    if missing or unexpected:
        details = [*(f"Missing panel: {path}" for path in missing)]
        details.extend(f"Unexpected panel: {path}" for path in unexpected)
        raise FileNotFoundError("PyMOL panel inventory is invalid:\n" + "\n".join(details))


def load_image(path: Path) -> np.ndarray:
    image = mpimg.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4]
        image = image[:, :, :3] * alpha + (1.0 - alpha)
    return image[:, :, :3]


def content_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    content_mask = np.any(image < 0.98, axis=2)
    if not np.any(content_mask):
        raise ValueError("Encountered a blank PyMOL panel")
    rows, cols = np.where(content_mask)
    return (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)


def common_crop_box(images: dict[tuple[str, int | None], np.ndarray]) -> tuple[int, int, int, int]:
    shapes = {image.shape for image in images.values()}
    if shapes != {(1200, 1600, 3)}:
        raise ValueError(f"Expected 1600x1200 RGB panels, found shapes: {sorted(shapes)}")

    bboxes = [content_bbox(image) for image in images.values()]
    top = max(min(bbox[0] for bbox in bboxes) - CROP_PADDING, 0)
    bottom = min(max(bbox[1] for bbox in bboxes) + CROP_PADDING, 1200)
    left = max(min(bbox[2] for bbox in bboxes) - CROP_PADDING, 0)
    right = min(max(bbox[3] for bbox in bboxes) + CROP_PADDING, 1600)
    return (top, bottom, left, right)


def load_panels(protein_ids: tuple[str, ...]) -> dict[tuple[str, int | None], np.ndarray]:
    images = {
        (protein_id, layer): load_image(panel_path(protein_id, layer))
        for protein_id in protein_ids
        for _label, layer in COLUMNS
    }
    top, bottom, left, right = common_crop_box(images)
    return {
        key: image[top:bottom, left:right]
        for key, image in images.items()
    }


def make_figure(
    protein_ids: tuple[str, ...],
    panels: dict[tuple[str, int | None], np.ndarray],
) -> plt.Figure:
    fig_height = ROW_HEIGHT * len(protein_ids)
    fig, axes = plt.subplots(
        len(protein_ids),
        len(COLUMNS),
        figsize=(FIG_WIDTH, fig_height),
    )

    for row_index, protein_id in enumerate(protein_ids):
        for column_index, (label, layer) in enumerate(COLUMNS):
            axis = axes[row_index, column_index]
            axis.imshow(panels[(protein_id, layer)])
            axis.set_axis_off()
            axis.set_title(label, pad=5, fontsize=FONT_SIZE)

        axes[row_index, 0].text(
            -0.22,
            0.5,
            protein_id,
            transform=axes[row_index, 0].transAxes,
            rotation=90,
            ha="center",
            va="center",
            fontsize=LABEL_SIZE,
        )

    fig.patch.set_facecolor("white")
    fig.subplots_adjust(
        left=0.055,
        right=0.995,
        top=0.985,
        bottom=0.01,
        wspace=0.10,
        hspace=0.48,
    )
    return fig


def save_outputs(fig: plt.Figure) -> None:
    CANONICAL_OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(CANONICAL_OUTPUT, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    for output_path in OVERLEAF_OUTPUTS:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(CANONICAL_OUTPUT, output_path)


def main() -> None:
    configure_style()
    protein_ids = load_protein_ids()
    validate_inputs(protein_ids)
    panels = load_panels(protein_ids)
    fig = make_figure(protein_ids, panels)
    try:
        save_outputs(fig)
    finally:
        plt.close(fig)

    print(f"Saved canonical figure: {CANONICAL_OUTPUT}")
    for output_path in OVERLEAF_OUTPUTS:
        print(f"Copied Overleaf figure: {output_path}")


if __name__ == "__main__":
    main()

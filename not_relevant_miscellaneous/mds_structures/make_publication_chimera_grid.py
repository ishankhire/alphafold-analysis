#!/usr/bin/env python3
"""Compose Chimera-rendered protein panels into a publication grid."""

from __future__ import annotations

import os
from pathlib import Path

EARLY_PROJECT_ROOT = Path(__file__).resolve().parents[1]
os.makedirs(EARLY_PROJECT_ROOT / ".mplconfig", exist_ok=True)
os.makedirs(EARLY_PROJECT_ROOT / ".cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(EARLY_PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(EARLY_PROJECT_ROOT / ".cache"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


PROJECT_ROOT = EARLY_PROJECT_ROOT
CHIMERA_DIR = PROJECT_ROOT / "chimera_files"
OUTPUT_PNG = CHIMERA_DIR / "publication_chimera_grid.png"
OUTPUT_PDF = CHIMERA_DIR / "publication_chimera_grid.pdf"

PROTEIN_IDS = (
    "7b3a_A",
    "7dkk_A",
    "7dq9_A",
    "7ebt_B",
    "7f6e_B",
    "7kdx_B",
    "7mro_A",
    "7pbk_A",
    "7tbs_A",
    "7tkv_A",
)

COLUMNS = (
    ("SM", "original_full"),
    ("L10", "layer10"),
    ("L15", "layer15"),
    ("L22", "layer22"),
    ("L47", "layer47"),
)

DPI = 300
CROP_PADDING = 28
FIG_WIDTH = 11.5
ROW_HEIGHT = 1.45
FONT_SIZE = 10
LABEL_SIZE = 10


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


def image_path(protein_id: str, image_stem: str) -> Path:
    return CHIMERA_DIR / protein_id / f"{protein_id}_{image_stem}.png"


def validate_inputs() -> None:
    missing = [
        image_path(protein_id, image_stem)
        for protein_id in PROTEIN_IDS
        for _label, image_stem in COLUMNS
        if not image_path(protein_id, image_stem).exists()
    ]
    if missing:
        missing_text = "\n".join(str(path) for path in missing)
        raise FileNotFoundError(f"Missing required Chimera PNG inputs:\n{missing_text}")


def load_image(path: Path) -> np.ndarray:
    image = mpimg.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4]
        image = image[:, :, :3] * alpha + (1.0 - alpha)
    return image[:, :, :3]


def content_bbox(image: np.ndarray) -> tuple[int, int, int, int]:
    content_mask = np.any(image < 0.965, axis=2)
    if not np.any(content_mask):
        return (0, image.shape[0], 0, image.shape[1])

    rows, cols = np.where(content_mask)
    return (int(rows.min()), int(rows.max()) + 1, int(cols.min()), int(cols.max()) + 1)


def common_crop_box(images: dict[tuple[str, str], np.ndarray]) -> tuple[int, int, int, int]:
    shapes = {image.shape for image in images.values()}
    if len(shapes) != 1:
        shape_text = ", ".join(str(shape) for shape in sorted(shapes))
        raise ValueError(f"Expected all input PNGs to share one shape, found: {shape_text}")

    image_height, image_width, _channels = next(iter(shapes))
    bboxes = [content_bbox(image) for image in images.values()]
    top = max(min(bbox[0] for bbox in bboxes) - CROP_PADDING, 0)
    bottom = min(max(bbox[1] for bbox in bboxes) + CROP_PADDING, image_height)
    left = max(min(bbox[2] for bbox in bboxes) - CROP_PADDING, 0)
    right = min(max(bbox[3] for bbox in bboxes) + CROP_PADDING, image_width)
    return (top, bottom, left, right)


def apply_crop(image: np.ndarray, crop_box: tuple[int, int, int, int]) -> np.ndarray:
    top, bottom, left, right = crop_box
    return image[top:bottom, left:right]


def load_panels() -> dict[tuple[str, str], np.ndarray]:
    images = {
        (protein_id, image_stem): load_image(image_path(protein_id, image_stem))
        for protein_id in PROTEIN_IDS
        for _label, image_stem in COLUMNS
    }
    crop_box = common_crop_box(images)
    return {key: apply_crop(image, crop_box) for key, image in images.items()}


def make_figure(panels: dict[tuple[str, str], np.ndarray]) -> plt.Figure:
    n_rows = len(PROTEIN_IDS)
    n_cols = len(COLUMNS)
    fig_height = ROW_HEIGHT * n_rows
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(FIG_WIDTH, fig_height))

    for row_index, protein_id in enumerate(PROTEIN_IDS):
        for col_index, (label, image_stem) in enumerate(COLUMNS):
            ax = axes[row_index, col_index]
            ax.imshow(panels[(protein_id, image_stem)])
            ax.set_axis_off()
            ax.set_title(label, pad=5, fontsize=FONT_SIZE)

        axes[row_index, 0].text(
            -0.24,
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
        left=0.065,
        right=0.995,
        top=0.985,
        bottom=0.01,
        wspace=0.18,
        hspace=0.48,
    )
    return fig


def main() -> None:
    configure_style()
    validate_inputs()
    panels = load_panels()
    fig = make_figure(panels)
    fig.savefig(OUTPUT_PNG, dpi=DPI, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(OUTPUT_PDF, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)
    print(f"Saved {OUTPUT_PNG}")
    print(f"Saved {OUTPUT_PDF}")


if __name__ == "__main__":
    main()

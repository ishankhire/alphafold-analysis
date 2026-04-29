#!/usr/bin/env python3
"""
make_chimera_image_strip.py
===========================
Compose Chimera-rendered PNGs into a labeled horizontal strip.

Default input:
    chimera_files/*.png

Default output:
    chimera_files/chimera_image_strip.png
"""

from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_DIR = PROJECT_ROOT / "chimera_files"
DEFAULT_OUTPUT = DEFAULT_INPUT_DIR / "chimera_image_strip.png"

os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib
matplotlib.use("Agg")
import matplotlib.image as mpimg
import matplotlib.pyplot as plt


@dataclass(frozen=True)
class StripImage:
    path: Path
    label: str
    sort_key: tuple[int, int, str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a labeled strip plot from Chimera PNG exports."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=DEFAULT_INPUT_DIR,
        help="Directory containing Chimera PNG files.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_OUTPUT,
        help="Output image path.",
    )
    parser.add_argument(
        "--files",
        nargs="+",
        type=Path,
        default=None,
        help=(
            "Optional explicit image paths. If omitted, all PNGs in --input-dir are used, "
            "excluding previous strip outputs."
        ),
    )
    parser.add_argument(
        "--labels",
        nargs="+",
        default=None,
        help="Optional labels matching --files order. Example: GT L10 L15 L22 L47",
    )
    parser.add_argument(
        "--side-label",
        default=None,
        help="Optional vertical label shown at the left of the strip.",
    )
    parser.add_argument(
        "--no-crop",
        action="store_true",
        help="Keep each PNG exactly as exported instead of trimming white margins.",
    )
    parser.add_argument(
        "--background",
        choices=("white", "original"),
        default="white",
        help="Use a white plot background by replacing near-black Chimera canvas pixels.",
    )
    parser.add_argument(
        "--crop-padding",
        type=int,
        default=30,
        help="Pixel padding retained around detected non-white content.",
    )
    parser.add_argument(
        "--dpi",
        type=int,
        default=250,
        help="DPI for the saved strip image.",
    )
    return parser.parse_args()


def infer_label(path: Path) -> str:
    stem = path.stem
    if stem.endswith("_final_pdb") or stem.endswith("_gt") or stem.endswith("_ground_truth"):
        return "GT"

    match = re.search(r"(?:^|_)layer[_-]?(\d+)(?:_|$)", stem, flags=re.IGNORECASE)
    if match:
        return f"L{int(match.group(1))}"

    return stem


def label_sort_key(label: str, path: Path) -> tuple[int, int, str]:
    if label == "GT":
        return (0, -1, path.name)

    match = re.fullmatch(r"L(\d+)", label)
    if match:
        return (1, int(match.group(1)), path.name)

    return (2, 0, path.name)


def infer_side_label(images: list[StripImage]) -> str | None:
    if not images:
        return None

    first_stem = images[0].path.stem
    for suffix in ("_final_pdb", "_gt", "_ground_truth"):
        if first_stem.endswith(suffix):
            return first_stem[: -len(suffix)]

    layer_match = re.search(r"(.+?)_layer[_-]?\d+$", first_stem, flags=re.IGNORECASE)
    if layer_match:
        return layer_match.group(1)

    return None


def collect_images(input_dir: Path, files: list[Path] | None, labels: list[str] | None) -> list[StripImage]:
    if files is None:
        paths = [
            path
            for path in input_dir.glob("*.png")
            if "strip" not in path.stem.lower()
        ]
    else:
        paths = [path if path.is_absolute() else PROJECT_ROOT / path for path in files]

    if not paths:
        raise FileNotFoundError(f"No PNG files found for strip plot in {input_dir}")

    if labels is not None and len(labels) != len(paths):
        raise ValueError(
            f"--labels count ({len(labels)}) must match image count ({len(paths)})"
        )

    images: list[StripImage] = []
    for index, path in enumerate(paths):
        if not path.exists():
            raise FileNotFoundError(f"Image not found: {path}")
        label = labels[index] if labels is not None else infer_label(path)
        images.append(StripImage(path=path, label=label, sort_key=label_sort_key(label, path)))

    if labels is None:
        images.sort(key=lambda image: image.sort_key)

    return images


def load_image(path: Path) -> np.ndarray:
    image = mpimg.imread(path)
    if image.ndim == 2:
        image = np.repeat(image[:, :, None], 3, axis=2)
    if image.shape[2] == 4:
        alpha = image[:, :, 3:4]
        image = image[:, :, :3] * alpha + (1.0 - alpha)
    return image[:, :, :3]


def normalize_background(image: np.ndarray, background: str) -> np.ndarray:
    if background == "original":
        return image

    normalized = image.copy()
    black_background = np.all(normalized < 0.08, axis=2)
    normalized[black_background] = 1.0
    return normalized


def crop_white_margin(image: np.ndarray, padding: int) -> np.ndarray:
    content_mask = np.any(image < 0.96, axis=2)
    if not np.any(content_mask):
        return image

    rows, cols = np.where(content_mask)
    top = max(int(rows.min()) - padding, 0)
    bottom = min(int(rows.max()) + padding + 1, image.shape[0])
    left = max(int(cols.min()) - padding, 0)
    right = min(int(cols.max()) + padding + 1, image.shape[1])
    return image[top:bottom, left:right]


def make_strip(
    images: list[StripImage],
    output: Path,
    side_label: str | None,
    background: str,
    crop: bool,
    crop_padding: int,
    dpi: int,
) -> None:
    rendered = []
    for strip_image in images:
        image = load_image(strip_image.path)
        image = normalize_background(image, background=background)
        if crop:
            image = crop_white_margin(image, padding=crop_padding)
        rendered.append(image)

    n_images = len(rendered)
    fig_width = max(2.3 * n_images, 6.0)
    fig_height = 1.9
    left_margin = 0.075 if side_label else 0.02

    fig, axes = plt.subplots(1, n_images, figsize=(fig_width, fig_height))
    if n_images == 1:
        axes = [axes]

    for ax, image, strip_image in zip(axes, rendered, images):
        ax.imshow(image)
        ax.set_axis_off()
        ax.text(
            0.5,
            -0.12,
            strip_image.label,
            transform=ax.transAxes,
            ha="center",
            va="top",
            fontsize=11,
        )

    if side_label:
        fig.text(
            0.022,
            0.56,
            side_label,
            rotation=90,
            ha="center",
            va="center",
            fontsize=11,
        )

    fig.subplots_adjust(
        left=left_margin,
        right=0.995,
        top=0.96,
        bottom=0.24,
        wspace=0.24,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=dpi, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def main() -> None:
    args = parse_args()
    images = collect_images(args.input_dir, args.files, args.labels)
    side_label = args.side_label
    if side_label is None:
        side_label = infer_side_label(images)

    make_strip(
        images=images,
        output=args.output,
        side_label=side_label,
        background=args.background,
        crop=not args.no_crop,
        crop_padding=args.crop_padding,
        dpi=args.dpi,
    )
    print(f"Saved strip plot: {args.output}")


if __name__ == "__main__":
    main()

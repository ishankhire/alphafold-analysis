#!/usr/bin/env python3
"""Combine the per_protein_base256_combined exact-triangle layer shards.

Mirrors the combine/move half of `analyze_delta_exact_triangle_metrics.py`
(general variant), adapted for the per-protein layout: there is no train/test
split, each shard carries a `source` column, and there are 552 proteins per layer.

Produces, under `delta_files/per_protein_base256_combined/exact_triangle_metrics/`:
- `exact_triangle_metrics.csv` (all 48 layers concatenated, validated)
- `layer_batches/` holding the moved per-layer shards plus `summary.txt`

Fixed configuration; no CLI arguments. Re-running after the shards have already
been moved reads the existing combined CSV instead.
"""

from __future__ import annotations

import filecmp
import re
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VARIANT = "per_protein_base256_combined"
EXPECTED_LAYERS = tuple(range(48))
EXPECTED_PROTEINS_PER_LAYER = 552

EXACT_DIR = PROJECT_ROOT / "delta_files" / VARIANT / "exact_triangle_metrics"
LAYER_BATCH_DIR = EXACT_DIR / "layer_batches"
COMBINED_CSV = EXACT_DIR / "exact_triangle_metrics.csv"

LAYER_FILE_RE = re.compile(r"exact_triangle_metrics_layer_(\d+)\.csv$")
VALIDATION_TOLERANCE = 5e-5

REQUIRED_COLUMNS = {
    "variant",
    "feature_dim",
    "layer",
    "source",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "n_triangles",
    "n_negative_predictions",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
}
PROPORTION_COLUMNS = [
    "triangle_violation_fraction",
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]
CATEGORY_COLUMNS = [
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]
VIOLATION_BIN_COLUMNS = [
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]


def parse_layer(path: Path) -> int:
    match = LAYER_FILE_RE.match(path.name)
    if match is None:
        raise ValueError(f"Could not parse layer from {path}")
    return int(match.group(1))


def map_layer_files(paths: list[Path]) -> dict[int, Path]:
    mapped: dict[int, Path] = {}
    for path in paths:
        layer = parse_layer(path)
        if layer in mapped:
            raise ValueError(f"Duplicate layer {layer} files: {mapped[layer]} and {path}")
        mapped[layer] = path
    return mapped


def find_layer_shards() -> list[Path]:
    top_level = map_layer_files(sorted(EXACT_DIR.glob("exact_triangle_metrics_layer_*.csv")))
    batched = map_layer_files(sorted(LAYER_BATCH_DIR.glob("exact_triangle_metrics_layer_*.csv")))
    expected = set(EXPECTED_LAYERS)

    if set(top_level) == expected:
        return [top_level[layer] for layer in EXPECTED_LAYERS]
    if set(batched) == expected:
        return [batched[layer] for layer in EXPECTED_LAYERS]

    combined = dict(batched)
    for layer, path in top_level.items():
        if layer in combined:
            raise ValueError(
                f"Layer {layer} exists in both {EXACT_DIR} and {LAYER_BATCH_DIR}; "
                "remove one copy or rerun after cleaning duplicates."
            )
        combined[layer] = path

    if set(combined) == expected:
        return [combined[layer] for layer in EXPECTED_LAYERS]

    if COMBINED_CSV.exists():
        return []

    missing = sorted(expected.difference(combined))
    extra = sorted(set(combined).difference(expected))
    raise FileNotFoundError(
        "Could not find a complete set of exact triangle layer shards; "
        f"missing={missing}, extra={extra}"
    )


def read_exact_metrics(shard_paths: list[Path]) -> pd.DataFrame:
    if shard_paths:
        frames = []
        for path in shard_paths:
            frame = pd.read_csv(path)
            file_layer = parse_layer(path)
            row_layers = tuple(sorted(int(value) for value in frame["layer"].unique()))
            if row_layers != (file_layer,):
                raise ValueError(f"{path} contains layer values {row_layers}, expected {file_layer}")
            frames.append(frame)
        df = pd.concat(frames, ignore_index=True)
    else:
        df = pd.read_csv(COMBINED_CSV)

    validate_exact_metrics(df)
    return df.sort_values(["layer", "source", "protein_id"]).reset_index(drop=True)


def validate_exact_metrics(df: pd.DataFrame) -> None:
    missing = sorted(REQUIRED_COLUMNS.difference(df.columns))
    if missing:
        raise ValueError(f"Exact triangle metrics are missing required columns: {missing}")

    layers = tuple(sorted(int(value) for value in df["layer"].dropna().unique()))
    if layers != EXPECTED_LAYERS:
        raise ValueError(f"Expected layers 0-47, found {layers}")

    variants = tuple(sorted(str(value) for value in df["variant"].dropna().unique()))
    if variants != (VARIANT,):
        raise ValueError(f"Expected variant {VARIANT!r}, found {variants}")

    row_counts = df.groupby("layer", sort=True).size()
    bad_row_counts = row_counts[row_counts != EXPECTED_PROTEINS_PER_LAYER]
    if not bad_row_counts.empty:
        raise ValueError(f"Expected {EXPECTED_PROTEINS_PER_LAYER} rows per layer, found {bad_row_counts.to_dict()}")

    protein_counts = df.groupby("layer", sort=True)["protein_id"].nunique()
    bad_protein_counts = protein_counts[protein_counts != EXPECTED_PROTEINS_PER_LAYER]
    if not bad_protein_counts.empty:
        raise ValueError(
            f"Expected {EXPECTED_PROTEINS_PER_LAYER} unique proteins per layer, found {bad_protein_counts.to_dict()}"
        )

    n_duplicates = int(df.duplicated(["layer", "protein_id"]).sum())
    if n_duplicates:
        raise ValueError(f"Found {n_duplicates} duplicate protein-layer rows")

    expected_triangles = (
        df["n_residues"].astype(np.int64)
        * (df["n_residues"].astype(np.int64) - 1)
        * (df["n_residues"].astype(np.int64) - 2)
        // 6
    )
    if not np.array_equal(expected_triangles.to_numpy(), df["n_triangles"].astype(np.int64).to_numpy()):
        raise ValueError("n_triangles does not match n_residues choose 3 for every row")

    for column in PROPORTION_COLUMNS:
        values = df[column].astype(np.float64)
        if ((values < -VALIDATION_TOLERANCE) | (values > 1.0 + VALIDATION_TOLERANCE)).any():
            raise ValueError(f"{column} contains values outside [0, 1]")

    category_sum = df[CATEGORY_COLUMNS].astype(np.float64).sum(axis=1)
    category_error = float(np.max(np.abs(category_sum - 1.0)))
    if category_error > VALIDATION_TOLERANCE:
        raise ValueError(f"Triangle category proportions do not sum to 1; max error={category_error}")

    violation_bin_sum = df[VIOLATION_BIN_COLUMNS].astype(np.float64).sum(axis=1)
    violation_error = float(
        np.max(np.abs(df["triangle_violation_fraction"].astype(np.float64) - violation_bin_sum))
    )
    if violation_error > VALIDATION_TOLERANCE:
        raise ValueError(
            "Triangle violation bin proportions do not match total violation fraction; "
            f"max error={violation_error}"
        )


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp_path, index=False, float_format="%.6f")
    tmp_path.replace(path)


def move_top_level_layer_artifacts(shard_paths: list[Path]) -> None:
    LAYER_BATCH_DIR.mkdir(parents=True, exist_ok=True)

    for path in shard_paths:
        if path.parent != EXACT_DIR:
            continue
        target = LAYER_BATCH_DIR / path.name
        if target.exists():
            if filecmp.cmp(path, target, shallow=False):
                path.unlink()
            else:
                raise FileExistsError(f"Refusing to overwrite non-identical {target}")
        else:
            path.replace(target)

    summary_path = EXACT_DIR / "summary.txt"
    if summary_path.exists():
        target = LAYER_BATCH_DIR / summary_path.name
        if target.exists():
            if filecmp.cmp(summary_path, target, shallow=False):
                summary_path.unlink()
            else:
                raise FileExistsError(f"Refusing to overwrite non-identical {target}")
        else:
            summary_path.replace(target)


def normalize_line_endings(path: Path) -> None:
    data = path.read_bytes()
    normalized = data.replace(b"\r\n", b"\n")
    if normalized != data:
        path.write_bytes(normalized)


def normalize_layer_batch_artifacts() -> None:
    for path in sorted(LAYER_BATCH_DIR.glob("exact_triangle_metrics_layer_*.csv")):
        normalize_line_endings(path)
    summary_path = LAYER_BATCH_DIR / "summary.txt"
    if summary_path.exists():
        normalize_line_endings(summary_path)


def main() -> None:
    EXACT_DIR.mkdir(parents=True, exist_ok=True)

    shard_paths = find_layer_shards()
    metrics = read_exact_metrics(shard_paths)
    write_csv(metrics, COMBINED_CSV)
    move_top_level_layer_artifacts(shard_paths)
    normalize_layer_batch_artifacts()

    print(f"Wrote {COMBINED_CSV.relative_to(PROJECT_ROOT)}")
    print(f"  rows={len(metrics)} layers={metrics['layer'].nunique()} "
          f"proteins/layer={sorted(metrics.groupby('layer').size().unique().tolist())}")
    print(f"Moved layer shards to: {LAYER_BATCH_DIR.relative_to(PROJECT_ROOT)}")
    layer_means = metrics.groupby("layer")["triangle_violation_fraction"].mean()
    print(f"  violation_fraction layer-mean range: [{layer_means.min():.4f}, {layer_means.max():.4f}]")


if __name__ == "__main__":
    main()

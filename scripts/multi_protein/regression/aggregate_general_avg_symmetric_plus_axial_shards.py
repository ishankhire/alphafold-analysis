"""Aggregate split layer shards for general_base256_avg_symmetric_plus_axial.

This merges the existing layer-range CSV outputs in
delta_files/general_base256_avg_symmetric_plus_axial/ into the canonical
single-file layout used by the other generalized train/test runs.
"""

from __future__ import annotations

import ast
import csv
import math
import re
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = ROOT_DIR / "delta_files" / "general_base256_avg_symmetric_plus_axial"
VARIANT_NAME = "general_base256_avg_symmetric_plus_axial"
FEATURE_DIM = 640
RANDOM_STATE = 42

LAYERED_OUTPUTS = {
    "coefficients": "coefficients.csv",
    "geometry_metrics": "geometry_metrics.csv",
    "geometry_summary": "geometry_summary.csv",
    "per_protein": "per_protein.csv",
    "split_summary": "split_summary.csv",
}
SHARED_OUTPUTS = {
    "split_assignments": "split_assignments.csv",
}


def read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open("r", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = list(reader.fieldnames or [])
        return fieldnames, list(reader)


def write_csv(path: Path, fieldnames: list[str], rows: list[dict[str, str]]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def shard_sort_key(path: Path) -> tuple[int, int]:
    match = re.search(r"_(\d+)-(\d+)\.csv$", path.name)
    if match is None:
        raise ValueError(f"Could not parse shard layer range from {path.name}")
    return int(match.group(1)), int(match.group(2))


def summary_sort_key(path: Path) -> tuple[int, int]:
    match = re.search(r"_(\d+)-(\d+)\.txt$", path.name)
    if match is None:
        raise ValueError(f"Could not parse shard layer range from {path.name}")
    return int(match.group(1)), int(match.group(2))


def load_and_merge_layered_shards(prefix: str) -> tuple[list[str], list[dict[str, str]], list[Path]]:
    shard_paths = sorted(INPUT_DIR.glob(f"{prefix}_*.csv"), key=shard_sort_key)
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found for {prefix!r} in {INPUT_DIR}")

    merged_rows: list[dict[str, str]] = []
    expected_fieldnames: list[str] | None = None
    seen_layers: set[int] = set()
    for shard_path in shard_paths:
        fieldnames, rows = read_csv(shard_path)
        if expected_fieldnames is None:
            expected_fieldnames = fieldnames
        elif fieldnames != expected_fieldnames:
            raise ValueError(
                f"Header mismatch for {shard_path.name}: expected {expected_fieldnames}, got {fieldnames}"
            )
        if "layer" not in fieldnames:
            raise ValueError(f"{shard_path.name} is missing required 'layer' column")
        shard_layers = {int(row["layer"]) for row in rows}
        overlap = seen_layers.intersection(shard_layers)
        if overlap:
            raise ValueError(f"Duplicate layers for {prefix}: {sorted(overlap)}")
        seen_layers.update(shard_layers)
        merged_rows.extend(rows)

    expected_layers = set(range(48))
    if seen_layers != expected_layers:
        missing = sorted(expected_layers.difference(seen_layers))
        extra = sorted(seen_layers.difference(expected_layers))
        raise ValueError(f"{prefix} layer coverage mismatch; missing={missing}, extra={extra}")

    return expected_fieldnames or [], merged_rows, shard_paths


def load_identical_shared_shards(prefix: str) -> tuple[list[str], list[dict[str, str]], list[Path]]:
    shard_paths = sorted(INPUT_DIR.glob(f"{prefix}_*.csv"), key=shard_sort_key)
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found for {prefix!r} in {INPUT_DIR}")

    first_fieldnames: list[str] | None = None
    first_rows: list[dict[str, str]] | None = None
    for shard_path in shard_paths:
        fieldnames, rows = read_csv(shard_path)
        if first_fieldnames is None:
            first_fieldnames = fieldnames
            first_rows = rows
            continue
        if fieldnames != first_fieldnames or rows != first_rows:
            raise ValueError(f"{prefix} shard {shard_path.name} does not match the first shard exactly")

    return first_fieldnames or [], first_rows or [], shard_paths


def parse_summary_metadata(path: Path) -> dict[str, object]:
    values: dict[str, object] = {}
    for line in path.read_text().splitlines():
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if not raw_value:
            continue
        values[key] = raw_value

    required = [
        "Variant",
        "Feature dimension",
        "Train/test split",
        "Random state",
        "Proteins discovered",
        "Proteins loaded",
        "Train proteins",
        "Test proteins",
        "Skipped proteins",
        "Size bin counts",
        "Elapsed seconds",
    ]
    for key in required:
        if key not in values:
            raise ValueError(f"Summary {path.name} is missing required field {key!r}")

    return {
        "variant": str(values["Variant"]),
        "feature_dim": int(str(values["Feature dimension"])),
        "split_description": str(values["Train/test split"]),
        "random_state": int(str(values["Random state"])),
        "proteins_discovered": int(str(values["Proteins discovered"])),
        "proteins_loaded": int(str(values["Proteins loaded"])),
        "train_proteins": int(str(values["Train proteins"])),
        "test_proteins": int(str(values["Test proteins"])),
        "skipped_proteins": int(str(values["Skipped proteins"])),
        "size_bin_counts": ast.literal_eval(str(values["Size bin counts"])),
        "elapsed_seconds": float(str(values["Elapsed seconds"])),
    }


def load_summary_metadata() -> dict[str, object]:
    summary_paths = sorted(INPUT_DIR.glob("summary_*.txt"), key=summary_sort_key)
    if not summary_paths:
        raise FileNotFoundError(f"No shard summaries found in {INPUT_DIR}")

    parsed = [parse_summary_metadata(path) for path in summary_paths]
    first = parsed[0]
    for other in parsed[1:]:
        for key in (
            "variant",
            "feature_dim",
            "split_description",
            "random_state",
            "proteins_discovered",
            "proteins_loaded",
            "train_proteins",
            "test_proteins",
            "skipped_proteins",
            "size_bin_counts",
        ):
            if other[key] != first[key]:
                raise ValueError(f"Summary metadata mismatch for {key}: {first[key]!r} != {other[key]!r}")

    return {
        **first,
        "elapsed_seconds": sum(float(item["elapsed_seconds"]) for item in parsed),
    }


def compute_best_r2(split_summary_rows: list[dict[str, str]]) -> tuple[int, float]:
    best_layer: int | None = None
    best_score = -math.inf
    for row in split_summary_rows:
        score = float(row["test_mean_per_protein_r2"])
        layer = int(row["layer"])
        if math.isnan(score):
            continue
        if score > best_score or (score == best_score and (best_layer is None or layer < best_layer)):
            best_layer = layer
            best_score = score
    if best_layer is None:
        raise ValueError("No valid held-out per-protein R2 values found")
    return best_layer, best_score


def compute_best_tm(geometry_summary_rows: list[dict[str, str]]) -> tuple[int, float]:
    best_layer: int | None = None
    best_score = -math.inf
    for row in geometry_summary_rows:
        if row["split"] != "test":
            continue
        score = float(row["tm_score_mean"])
        layer = int(row["layer"])
        if math.isnan(score):
            continue
        if score > best_score or (score == best_score and (best_layer is None or layer < best_layer)):
            best_layer = layer
            best_score = score
    if best_layer is None:
        raise ValueError("No valid held-out TM-score means found")
    return best_layer, best_score


def write_summary(
    path: Path,
    metadata: dict[str, object],
    split_summary_rows: list[dict[str, str]],
    geometry_summary_rows: list[dict[str, str]],
) -> None:
    short_variant = VARIANT_NAME.removeprefix("general_")
    title = f"General Train/Test Regression - {short_variant}"
    best_r2_layer, best_r2_value = compute_best_r2(split_summary_rows)
    best_tm_layer, best_tm_value = compute_best_tm(geometry_summary_rows)
    heldout_rows = sorted(split_summary_rows, key=lambda row: int(row["layer"]))

    with path.open("w") as handle:
        handle.write(f"{title}\n")
        handle.write(f"{'=' * len(title)}\n\n")
        handle.write(f"Variant:             {metadata['variant']}\n")
        handle.write(f"Feature dimension:   {metadata['feature_dim']}\n")
        handle.write("Layers:              0-47 (48 total)\n")
        handle.write(f"Train/test split:    {metadata['split_description']}\n")
        handle.write(f"Random state:        {metadata['random_state']}\n")
        handle.write(f"Proteins discovered: {metadata['proteins_discovered']}\n")
        handle.write(f"Proteins loaded:     {metadata['proteins_loaded']}\n")
        handle.write(f"Train proteins:      {metadata['train_proteins']}\n")
        handle.write(f"Test proteins:       {metadata['test_proteins']}\n")
        handle.write(f"Skipped proteins:    {metadata['skipped_proteins']}\n")
        handle.write(f"Size bin counts:     {metadata['size_bin_counts']}\n")
        handle.write(f"Elapsed seconds:     {float(metadata['elapsed_seconds']):.2f}\n\n")
        handle.write("Best layer by held-out per-protein R2:\n")
        handle.write(f"  Layer {best_r2_layer} -> {best_r2_value:.4f}\n\n")
        handle.write("Best layer by held-out mean TM-score:\n")
        handle.write(f"  Layer {best_tm_layer} -> {best_tm_value:.4f}\n\n")
        handle.write("Layer means (held-out per-protein R2):\n")
        for row in heldout_rows:
            handle.write(f"  Layer {int(row['layer']):02d}: {float(row['test_mean_per_protein_r2']):.4f}\n")


def verify_split_assignments(rows: list[dict[str, str]]) -> None:
    split_pairs = {(int(row["split_id"]), row["split"]) for row in rows}
    if split_pairs != {(0, "train"), (0, "test")}:
        raise ValueError(f"Unexpected split assignments present: {sorted(split_pairs)}")


def main() -> None:
    merged_tables: dict[str, tuple[list[str], list[dict[str, str]], list[Path]]] = {}
    for prefix in LAYERED_OUTPUTS:
        merged_tables[prefix] = load_and_merge_layered_shards(prefix)
    for prefix in SHARED_OUTPUTS:
        merged_tables[prefix] = load_identical_shared_shards(prefix)

    metadata = load_summary_metadata()
    if metadata["variant"] != VARIANT_NAME:
        raise ValueError(f"Unexpected variant {metadata['variant']!r}")
    if int(metadata["feature_dim"]) != FEATURE_DIM:
        raise ValueError(f"Unexpected feature dimension {metadata['feature_dim']!r}")

    verify_split_assignments(merged_tables["split_assignments"][1])

    for prefix, output_name in {**LAYERED_OUTPUTS, **SHARED_OUTPUTS}.items():
        fieldnames, rows, _ = merged_tables[prefix]
        write_csv(INPUT_DIR / output_name, fieldnames, rows)

    write_summary(
        INPUT_DIR / "summary.txt",
        metadata,
        merged_tables["split_summary"][1],
        merged_tables["geometry_summary"][1],
    )

    print("Wrote aggregated outputs:")
    for output_name in [
        "coefficients.csv",
        "geometry_metrics.csv",
        "geometry_summary.csv",
        "per_protein.csv",
        "split_assignments.csv",
        "split_summary.csv",
        "summary.txt",
    ]:
        print(f"  {INPUT_DIR / output_name}")


if __name__ == "__main__":
    main()

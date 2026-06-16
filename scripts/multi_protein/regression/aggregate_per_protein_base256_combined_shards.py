"""Aggregate layer shards for per_protein_base256_combined.

This merges the layer-range CSV and text outputs in
delta_files/per_protein_base256_combined/ into the canonical single-file
layout used by downstream local analyses.
"""

from __future__ import annotations

import ast
import csv
import os
import re
import tempfile
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
INPUT_DIR = ROOT_DIR / "delta_files" / "per_protein_base256_combined"
VARIANT_NAME = "per_protein_base256_combined"
FEATURE_DIM = 256
EXPECTED_LAYERS = set(range(48))

CSV_OUTPUTS = {
    "coefficients": "coefficients.csv",
    "geometry_metrics": "geometry_metrics.csv",
    "geometry_summary": "geometry_summary.csv",
    "per_protein": "per_protein.csv",
}


def shard_sort_key(path: Path) -> tuple[int, int]:
    match = re.search(r"_layers(\d+)_(\d+)\.(?:csv|txt)$", path.name)
    if match is None:
        raise ValueError(f"Could not parse shard layer range from {path.name}")
    return int(match.group(1)), int(match.group(2))


def expected_layers_for_shard(path: Path) -> set[int]:
    start, end = shard_sort_key(path)
    return set(range(start, end + 1))


def aggregate_csv(prefix: str, output_name: str) -> tuple[int, list[Path]]:
    shard_paths = sorted(INPUT_DIR.glob(f"{prefix}_layers*.csv"), key=shard_sort_key)
    if not shard_paths:
        raise FileNotFoundError(f"No shard files found for {prefix!r} in {INPUT_DIR}")

    output_path = INPUT_DIR / output_name
    expected_header: list[str] | None = None
    seen_layers: set[int] = set()
    rows_written = 0

    fd, tmp_name = tempfile.mkstemp(
        prefix=f".{output_name}.", suffix=".tmp", dir=str(INPUT_DIR)
    )
    os.close(fd)
    tmp_path = Path(tmp_name)

    try:
        with tmp_path.open("w", newline="") as output_handle:
            writer: csv.writer | None = None
            for shard_path in shard_paths:
                shard_layers: set[int] = set()
                with shard_path.open("r", newline="") as input_handle:
                    reader = csv.reader(input_handle)
                    try:
                        header = next(reader)
                    except StopIteration as exc:
                        raise ValueError(f"{shard_path.name} is empty") from exc

                    if expected_header is None:
                        expected_header = header
                        if "layer" not in header:
                            raise ValueError(f"{shard_path.name} is missing required 'layer' column")
                        writer = csv.writer(output_handle)
                        writer.writerow(header)
                    elif header != expected_header:
                        raise ValueError(
                            f"Header mismatch for {shard_path.name}: "
                            f"expected {expected_header}, got {header}"
                        )

                    assert writer is not None
                    layer_index = header.index("layer")
                    for row_number, row in enumerate(reader, start=2):
                        if len(row) != len(header):
                            raise ValueError(
                                f"{shard_path.name}:{row_number} has {len(row)} fields; "
                                f"expected {len(header)}"
                            )
                        try:
                            layer = int(row[layer_index])
                        except ValueError as exc:
                            raise ValueError(
                                f"{shard_path.name}:{row_number} has non-integer layer "
                                f"{row[layer_index]!r}"
                            ) from exc
                        shard_layers.add(layer)
                        writer.writerow(row)
                        rows_written += 1

                overlap = seen_layers.intersection(shard_layers)
                if overlap:
                    raise ValueError(f"Duplicate layers for {prefix}: {sorted(overlap)}")
                expected_shard_layers = expected_layers_for_shard(shard_path)
                if shard_layers != expected_shard_layers:
                    missing = sorted(expected_shard_layers.difference(shard_layers))
                    extra = sorted(shard_layers.difference(expected_shard_layers))
                    raise ValueError(
                        f"{shard_path.name} layer coverage mismatch; "
                        f"missing={missing}, extra={extra}"
                    )
                seen_layers.update(shard_layers)

        if seen_layers != EXPECTED_LAYERS:
            missing = sorted(EXPECTED_LAYERS.difference(seen_layers))
            extra = sorted(seen_layers.difference(EXPECTED_LAYERS))
            raise ValueError(f"{prefix} layer coverage mismatch; missing={missing}, extra={extra}")

        tmp_path.replace(output_path)
    except Exception:
        tmp_path.unlink(missing_ok=True)
        raise

    return rows_written, shard_paths


def parse_summary(path: Path) -> tuple[dict[str, object], list[str], list[str]]:
    metadata: dict[str, object] = {}
    layer_lines: list[str] = []
    skipped_lines: list[str] = []
    in_skipped = False

    for line in path.read_text().splitlines():
        if re.match(r"Layer \d{2} (?:R2|TM-score):", line):
            layer_lines.append(line)
            in_skipped = False
            continue
        if line == "Skipped proteins/events:":
            in_skipped = True
            continue
        if in_skipped and line.startswith("  "):
            skipped_lines.append(line)
            continue
        if ":" not in line:
            continue
        key, raw_value = line.split(":", 1)
        key = key.strip()
        raw_value = raw_value.strip()
        if raw_value:
            metadata[key] = raw_value

    required = [
        "Variant",
        "Feature dimension",
        "Feature convention",
        "Dataset",
        "Layers",
        "Splits",
        "Proteins discovered",
        "Proteins loaded",
        "Loaded by source",
        "Completed rows",
        "Completed by layer",
        "Skipped events",
        "Elapsed seconds",
    ]
    missing = [key for key in required if key not in metadata]
    if missing:
        raise ValueError(f"Summary {path.name} is missing required fields: {missing}")

    parsed = {
        "variant": str(metadata["Variant"]),
        "feature_dim": int(str(metadata["Feature dimension"])),
        "feature_convention": str(metadata["Feature convention"]),
        "dataset": str(metadata["Dataset"]),
        "splits": str(metadata["Splits"]),
        "proteins_discovered": int(str(metadata["Proteins discovered"])),
        "proteins_loaded": int(str(metadata["Proteins loaded"])),
        "loaded_by_source": ast.literal_eval(str(metadata["Loaded by source"])),
        "completed_rows": int(str(metadata["Completed rows"])),
        "completed_by_layer": {
            int(layer): int(count)
            for layer, count in ast.literal_eval(str(metadata["Completed by layer"])).items()
        },
        "skipped_events": int(str(metadata["Skipped events"])),
        "elapsed_seconds": float(str(metadata["Elapsed seconds"])),
    }
    return parsed, layer_lines, skipped_lines


def load_summary_metadata() -> tuple[dict[str, object], list[str], list[str], list[Path]]:
    summary_paths = sorted(INPUT_DIR.glob("summary_layers*.txt"), key=shard_sort_key)
    if not summary_paths:
        raise FileNotFoundError(f"No shard summaries found in {INPUT_DIR}")

    parsed_items = [parse_summary(path) for path in summary_paths]
    first = parsed_items[0][0]
    constant_keys = [
        "variant",
        "feature_dim",
        "feature_convention",
        "dataset",
        "splits",
        "proteins_discovered",
        "proteins_loaded",
        "loaded_by_source",
        "skipped_events",
    ]
    for metadata, _, _ in parsed_items[1:]:
        for key in constant_keys:
            if metadata[key] != first[key]:
                raise ValueError(
                    f"Summary metadata mismatch for {key}: {first[key]!r} != {metadata[key]!r}"
                )

    completed_by_layer: dict[int, int] = {}
    layer_lines: list[str] = []
    skipped_lines: list[str] = []
    for metadata, shard_layer_lines, shard_skipped_lines in parsed_items:
        for layer, count in metadata["completed_by_layer"].items():
            if layer in completed_by_layer:
                raise ValueError(f"Duplicate completed-by-layer entry for layer {layer}")
            completed_by_layer[layer] = count
        layer_lines.extend(shard_layer_lines)
        skipped_lines.extend(shard_skipped_lines)

    seen_layers = set(completed_by_layer)
    if seen_layers != EXPECTED_LAYERS:
        missing = sorted(EXPECTED_LAYERS.difference(seen_layers))
        extra = sorted(seen_layers.difference(EXPECTED_LAYERS))
        raise ValueError(f"Summary layer coverage mismatch; missing={missing}, extra={extra}")

    metadata = {
        **first,
        "completed_rows": sum(item[0]["completed_rows"] for item in parsed_items),
        "completed_by_layer": dict(sorted(completed_by_layer.items())),
        "elapsed_seconds": sum(float(item[0]["elapsed_seconds"]) for item in parsed_items),
    }
    unique_skipped_lines = list(dict.fromkeys(skipped_lines))
    layer_lines.sort(key=lambda line: (int(line.split()[1]), line))
    return metadata, layer_lines, unique_skipped_lines, summary_paths


def write_summary(
    path: Path,
    metadata: dict[str, object],
    layer_lines: list[str],
    skipped_lines: list[str],
) -> None:
    title = "Combined CAMEO+ECOD Per-Protein Regression - base256"
    with path.open("w") as handle:
        handle.write(f"{title}\n")
        handle.write(f"{'=' * len(title)}\n\n")
        handle.write(f"Variant:             {metadata['variant']}\n")
        handle.write(f"Feature dimension:   {metadata['feature_dim']}\n")
        handle.write(f"Feature convention:  {metadata['feature_convention']}\n")
        handle.write(f"Dataset:             {metadata['dataset']}\n")
        handle.write("Layers:              0-47 (48 total)\n")
        handle.write(f"Splits:              {metadata['splits']}\n")
        handle.write(f"Proteins discovered: {metadata['proteins_discovered']}\n")
        handle.write(f"Proteins loaded:     {metadata['proteins_loaded']}\n")
        handle.write(f"Loaded by source:    {metadata['loaded_by_source']}\n")
        handle.write(f"Completed rows:      {metadata['completed_rows']}\n")
        handle.write(f"Completed by layer:  {metadata['completed_by_layer']}\n")
        handle.write(f"Skipped events:      {metadata['skipped_events']}\n")
        handle.write(f"Elapsed seconds:     {float(metadata['elapsed_seconds']):.2f}\n\n")

        for line in layer_lines:
            handle.write(f"{line}\n")

        handle.write("\nSkipped proteins/events:\n")
        for line in skipped_lines:
            handle.write(f"{line}\n")


def main() -> None:
    output_counts: dict[str, tuple[int, list[Path]]] = {}
    for prefix, output_name in CSV_OUTPUTS.items():
        output_counts[output_name] = aggregate_csv(prefix, output_name)

    metadata, layer_lines, skipped_lines, summary_paths = load_summary_metadata()
    if metadata["variant"] != VARIANT_NAME:
        raise ValueError(f"Unexpected variant {metadata['variant']!r}")
    if int(metadata["feature_dim"]) != FEATURE_DIM:
        raise ValueError(f"Unexpected feature dimension {metadata['feature_dim']!r}")
    write_summary(INPUT_DIR / "summary.txt", metadata, layer_lines, skipped_lines)

    print("Wrote aggregated outputs:")
    for output_name, (row_count, shard_paths) in output_counts.items():
        print(f"  {INPUT_DIR / output_name} ({row_count} rows from {len(shard_paths)} shards)")
    print(f"  {INPUT_DIR / 'summary.txt'} (from {len(summary_paths)} shard summaries)")


if __name__ == "__main__":
    main()

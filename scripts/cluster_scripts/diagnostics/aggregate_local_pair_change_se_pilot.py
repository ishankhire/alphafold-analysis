#!/usr/bin/env python3
"""Aggregate completed local-pair change SE transition shards."""

import csv
import os
from pathlib import Path


OUTPUT_NAME = "local_pair_change_se_pilot_layers10_15"
TRANSITIONS = tuple((layer, layer + 1) for layer in range(10, 16))
AGGREGATE_FILES = (
    "change_statistics.csv",
    "pair_se_statistics.csv",
    "change_buckets.csv",
    "threshold_stability.csv",
    "per_protein_statistics.csv",
    "bootstrap_fit_summary.csv",
    "bootstrap_coefficients.csv",
    "bootstrap_assignments.csv",
    "protein_manifest.csv",
    "coefficients_used.csv",
)


def aggregate_csv(output_root, filename):
    all_rows = []
    fieldnames = None
    for layer_from, layer_to in TRANSITIONS:
        shard = (
            output_root
            / "layer_shards"
            / f"transition_{layer_from:02d}_{layer_to:02d}"
            / filename
        )
        if not shard.exists():
            raise FileNotFoundError(f"Missing required transition shard: {shard}")
        with shard.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise ValueError(f"Schema mismatch in {shard}")
            all_rows.extend(reader)

    tmp_path = output_root / f"{filename}.tmp"
    with tmp_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(all_rows)
    os.replace(tmp_path, output_root / filename)
    return len(all_rows)


def main():
    csv_root = Path(
        os.environ.get("CSV_ROOT") or (Path.home() / "csv_files")
    ).expanduser().resolve()
    output_root = csv_root / OUTPUT_NAME
    counts = {
        filename: aggregate_csv(output_root, filename)
        for filename in AGGREGATE_FILES
    }
    summary_tmp = output_root / "aggregate_summary.txt.tmp"
    with summary_tmp.open("w") as handle:
        handle.write("Local pair change SE pilot aggregate\n")
        handle.write("====================================\n\n")
        handle.write("Completed transitions: 10->11 through 15->16\n")
        handle.write("Pair-specific CSV.gz files remain in their transition shards.\n\n")
        for filename, count in counts.items():
            handle.write(f"{filename}: {count} rows\n")
    os.replace(summary_tmp, output_root / "aggregate_summary.txt")


if __name__ == "__main__":
    main()

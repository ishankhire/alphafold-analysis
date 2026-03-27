#!/usr/bin/env python3
"""Repair missing aggregate geometry rows for general_base256_plus_diagonal."""

from __future__ import annotations

import sys
from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[3]
TRAIN_TEST_DIR = (
    ROOT_DIR
    / "scripts"
    / "cluster_scripts"
    / "generalized_model"
    / "mar25_train_test_split_regressions"
)
if str(TRAIN_TEST_DIR) not in sys.path:
    sys.path.insert(0, str(TRAIN_TEST_DIR))

import general_train_test_common as common  # noqa: E402


VARIANT_DIR = ROOT_DIR / "delta_files" / "general_base256_plus_diagonal"
METRICS_PATH = VARIANT_DIR / "geometry_metrics.csv"
SUMMARY_PATH = VARIANT_DIR / "geometry_summary.csv"


def summary_key(row: dict[str, object]) -> tuple[object, ...]:
    return (
        row["variant"],
        int(row["feature_dim"]),
        int(row["split_id"]),
        int(row["layer"]),
        row["split"],
    )


def row_has_missing_aggregates(row: dict[str, object]) -> bool:
    return all(str(row[field_name]).lower() == "nan" for field_name in common.GEOMETRY_SUMMARY_FIELDNAMES[6:])


def main() -> None:
    summary_rows = common.read_csv_rows(SUMMARY_PATH, common.GEOMETRY_SUMMARY_FIELDNAMES)
    metrics_rows = common.read_csv_rows(METRICS_PATH, common.GEOMETRY_METRICS_FIELDNAMES)
    metrics_rows = common.coerce_geometry_metric_rows(metrics_rows)
    rebuilt_rows = common.summarize_geometry_rows(metrics_rows)
    rebuilt_by_key = {summary_key(row): row for row in rebuilt_rows}

    repaired_rows = []
    repaired_count = 0
    for row in summary_rows:
        key = summary_key(row)
        if row_has_missing_aggregates(row):
            repaired_rows.append(rebuilt_by_key[key])
            repaired_count += 1
            continue
        repaired_rows.append(row)

    if repaired_count == 0:
        print(f"No missing geometry summary rows found in {SUMMARY_PATH}")
        return

    common.write_csv(SUMMARY_PATH, common.GEOMETRY_SUMMARY_FIELDNAMES, repaired_rows)

    validated_rows = common.read_csv_rows(SUMMARY_PATH, common.GEOMETRY_SUMMARY_FIELDNAMES)
    still_missing = sum(1 for row in validated_rows if row_has_missing_aggregates(row))
    if still_missing:
        raise ValueError(f"{SUMMARY_PATH.name} still has {still_missing} broken geometry summary rows")

    print(f"Repaired {repaired_count} geometry summary rows in {SUMMARY_PATH}")


if __name__ == "__main__":
    main()

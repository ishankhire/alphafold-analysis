#!/usr/bin/env python3
"""Validate and aggregate all 15 local pair-update probe shards."""

from __future__ import annotations

import csv
import json
import math
import os
from datetime import datetime
from pathlib import Path


TRANSITIONS = ((10, 11), (20, 21), (30, 31))
VARIANTS = ("distance_seqsep", "center", "triangle_r1", "triangle_r2", "triangle_r3")
OUTPUT_NAME = "local_pair_update_probe_pilot"
REQUIRED_TABLES = (
    "classification_summary.csv",
    "regression_summary.csv",
    "class_metrics.csv",
    "confusion_matrix.csv",
    "per_protein_metrics.csv",
    "coefficients.csv",
    "change_statistics.csv",
    "feature_schema.csv",
    "protein_manifest.csv",
    "coefficients_used.csv",
    "split_assignments.csv",
    "timing.csv",
    "skipped_proteins.csv",
)


def now_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def format_value(value):
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, float):
        if not math.isfinite(value):
            return "nan"
        return f"{value:.9g}"
    return value


def write_csv_atomic(path, fieldnames, rows):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})
    os.replace(temporary, path)


def write_json_atomic(path, payload):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_text_atomic(path, value):
    temporary = Path(path).with_name(Path(path).name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def expected_specs():
    return [
        (layer_from, layer_to, variant)
        for layer_from, layer_to in TRANSITIONS
        for variant in VARIANTS
    ]


def shard_path(output_root, layer_from, layer_to, variant):
    return (
        Path(output_root)
        / "shards"
        / f"transition_{layer_from:02d}_{layer_to:02d}"
        / variant
    )


def validate_shards(output_root, required_tables=REQUIRED_TABLES):
    validated = []
    problems = []
    for layer_from, layer_to, variant in expected_specs():
        directory = shard_path(output_root, layer_from, layer_to, variant)
        metadata_path = directory / "run_metadata.json"
        if not metadata_path.exists():
            problems.append(f"missing metadata: {metadata_path}")
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except (OSError, json.JSONDecodeError) as exc:
            problems.append(f"invalid metadata {metadata_path}: {exc}")
            continue
        if metadata.get("status") != "complete":
            problems.append(
                f"incomplete shard {directory}: status={metadata.get('status')!r}"
            )
        if (
            int(metadata.get("layer_from", -1)) != layer_from
            or int(metadata.get("layer_to", -1)) != layer_to
            or metadata.get("variant") != variant
        ):
            problems.append(f"metadata identity mismatch: {metadata_path}")
        for filename in required_tables:
            if not (directory / filename).exists():
                problems.append(f"missing table: {directory / filename}")
        validated.append((layer_from, layer_to, variant, directory, metadata))
    if problems:
        raise RuntimeError(
            "Cannot aggregate local pair-update pilot:\n  - " + "\n  - ".join(problems)
        )
    if len(validated) != 15:
        raise RuntimeError(f"Expected 15 complete shards, found {len(validated)}")
    return validated


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def merge_rows(validated, filename):
    rows = []
    fieldnames = None
    for _layer_from, _layer_to, _variant, directory, _metadata in validated:
        path = directory / filename
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if fieldnames is None:
                fieldnames = list(reader.fieldnames or [])
            elif list(reader.fieldnames or []) != fieldnames:
                raise ValueError(f"Schema mismatch in {path}")
            rows.extend(reader)
    return fieldnames or [], rows


def deduplicate(rows, key_fields, label):
    unique = {}
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in unique and unique[key] != row:
            raise ValueError(f"Conflicting duplicate {label} row for key={key}")
        unique[key] = row
    return list(unique.values())


def finite(row, key):
    try:
        value = float(row[key])
    except (KeyError, TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def mean_finite(values):
    clean = [float(value) for value in values if math.isfinite(float(value))]
    return sum(clean) / len(clean) if clean else float("nan")


def ranked_variants(rows, metric):
    by_variant = {}
    for row in rows:
        if row.get("split") != "test":
            continue
        by_variant.setdefault(row["variant"], []).append(finite(row, metric))
    scores = [
        (variant, mean_finite(values)) for variant, values in by_variant.items()
    ]
    return sorted(scores, key=lambda item: (-item[1], item[0]))


def gains(rows, metric, variant):
    by_transition = {}
    for row in rows:
        if row.get("split") != "test":
            continue
        key = (int(row["layer_from"]), int(row["layer_to"]))
        by_transition.setdefault(key, {})[row["variant"]] = finite(row, metric)
    gains_by_baseline = {}
    for baseline in ("distance_seqsep", "center"):
        values = []
        for scores in by_transition.values():
            if variant in scores and baseline in scores:
                values.append(scores[variant] - scores[baseline])
        gains_by_baseline[baseline] = mean_finite(values)
    return gains_by_baseline


def summary_text(classification_rows, regression_rows, metadata_rows):
    classifier_rank = ranked_variants(classification_rows, "pair_macro_f1")
    classifier_balanced_rank = ranked_variants(
        classification_rows, "pair_balanced_accuracy"
    )
    regression_pair_rank = ranked_variants(regression_rows, "pair_r2")
    regression_protein_rank = ranked_variants(
        regression_rows, "mean_per_protein_r2"
    )
    lines = [
        "Local pair-update probe pilot aggregate",
        "========================================",
        "",
        "Status: complete",
        f"Completed shards: {len(metadata_rows)}/15",
        "Transitions: 10->11, 20->21, 30->31",
        "Distance type: raw pre-MDS",
        "Classification threshold: +/-0.5 Angstrom",
        "",
        "Mean held-out classifier ranking by pair macro-F1:",
    ]
    for rank, (variant, score) in enumerate(classifier_rank, start=1):
        gain = gains(classification_rows, "pair_macro_f1", variant)
        lines.append(
            f"  {rank}. {variant}: {score:.4f} "
            f"(gain vs distance={gain['distance_seqsep']:+.4f}, "
            f"vs center={gain['center']:+.4f})"
        )
    lines.append("")
    lines.append("Mean held-out classifier ranking by pair balanced accuracy:")
    for rank, (variant, score) in enumerate(classifier_balanced_rank, start=1):
        lines.append(f"  {rank}. {variant}: {score:.4f}")
    lines.append("")
    lines.append("Mean held-out regression ranking by pair R2:")
    for rank, (variant, score) in enumerate(regression_pair_rank, start=1):
        gain = gains(regression_rows, "pair_r2", variant)
        lines.append(
            f"  {rank}. {variant}: {score:.4f} "
            f"(gain vs distance={gain['distance_seqsep']:+.4f}, "
            f"vs center={gain['center']:+.4f})"
        )
    lines.append("")
    lines.append("Mean held-out regression ranking by protein-equal mean R2:")
    for rank, (variant, score) in enumerate(regression_protein_rank, start=1):
        lines.append(f"  {rank}. {variant}: {score:.4f}")
    lines.append("")
    lines.append("Iteration-cap flags:")
    for metadata in metadata_rows:
        flags = []
        if metadata.get("classifier_iteration_cap_reached"):
            flags.append("classifier")
        if metadata.get("regression_iteration_cap_reached"):
            flags.append("regression")
        if flags:
            lines.append(
                f"  {metadata['layer_from']}->{metadata['layer_to']} "
                f"{metadata['variant']}: {','.join(flags)}"
            )
    if lines[-1] == "Iteration-cap flags:":
        lines.append("  none")
    return "\n".join(lines) + "\n"


def main():
    csv_root = Path(os.environ.get("CSV_ROOT", Path.home() / "csv_files")).expanduser()
    output_root = Path(
        os.environ.get("LOCAL_PAIR_UPDATE_OUTPUT_ROOT", csv_root / OUTPUT_NAME)
    ).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    validated = validate_shards(output_root)

    simple_merges = (
        "classification_summary.csv",
        "regression_summary.csv",
        "class_metrics.csv",
        "confusion_matrix.csv",
        "per_protein_metrics.csv",
        "coefficients.csv",
        "protein_manifest.csv",
        "timing.csv",
        "skipped_proteins.csv",
    )
    merged = {}
    for filename in simple_merges:
        fieldnames, rows = merge_rows(validated, filename)
        write_csv_atomic(output_root / filename, fieldnames, rows)
        merged[filename] = rows

    dedupe_specs = {
        "change_statistics.csv": ("layer_from", "layer_to", "split"),
        "feature_schema.csv": ("variant", "feature_index"),
        "coefficients_used.csv": ("layer", "feature_index"),
        "split_assignments.csv": ("split_id", "protein_id"),
    }
    for filename, key_fields in dedupe_specs.items():
        fieldnames, rows = merge_rows(validated, filename)
        rows = deduplicate(rows, key_fields, filename)
        write_csv_atomic(output_root / filename, fieldnames, rows)
        merged[filename] = rows

    metadata_rows = [metadata for *_prefix, metadata in validated]
    metadata_payload = {
        "status": "complete",
        "completed_at": now_timestamp(),
        "n_shards": len(validated),
        "transitions": [list(value) for value in TRANSITIONS],
        "variants": list(VARIANTS),
        "n_classifier_fits": len(validated),
        "n_regression_fits": len(validated),
        "all_train_pairs_per_fit": 812097,
        "all_test_pairs_per_fit": 207900,
        "shards": metadata_rows,
    }
    write_json_atomic(output_root / "run_metadata.json", metadata_payload)
    write_text_atomic(
        output_root / "summary.txt",
        summary_text(
            merged["classification_summary.csv"],
            merged["regression_summary.csv"],
            metadata_rows,
        ),
    )
    print(f"Aggregated 15 complete shards into {output_root}", flush=True)


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Validate and aggregate all expanded local pair-update Ridge shards."""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

import numpy as np

from local_pair_representation_update_common import (
    BOOTSTRAP_REPLICATES, LSQR_MAX_ITER, OUTPUT_NAME, TRANSITIONS, VARIANTS,
    feature_dim, now_timestamp, write_csv_atomic, write_json_atomic,
    write_text_atomic,
)


REQUIRED_FILES = (
    "model_summary.csv", "per_protein_metrics.csv",
    "regression_magnitude_metrics.csv", "aa_pair_target_statistics.csv",
    "aa_pair_regression_metrics.csv", "aa_pair_target_uncertainty.csv",
    "aa_pair_bootstrap_summary.csv", "bootstrap_assignments.csv",
    "bootstrap_metrics.csv", "coefficients.csv", "feature_schema.csv",
    "feature_transforms.csv", "change_statistics.csv", "coefficients_used.csv",
    "split_assignments.csv", "protein_manifest.csv", "skipped_proteins.csv",
    "timing.csv", "summary.txt",
)
CONTRASTS = tuple(zip(VARIANTS[1:], VARIANTS[:-1]))


def expected_specs():
    return [(*transition, variant) for transition in TRANSITIONS for variant in VARIANTS]


def shard_path(output_root, layer_from, layer_to, variant):
    return Path(output_root) / "shards" / f"transition_{layer_from:02d}_{layer_to:02d}" / variant


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_shards(output_root, required_files=REQUIRED_FILES):
    validated, problems = [], []
    for layer_from, layer_to, variant in expected_specs():
        directory = shard_path(output_root, layer_from, layer_to, variant)
        metadata_path = directory / "run_metadata.json"
        if not metadata_path.exists():
            problems.append(f"missing metadata: {metadata_path}")
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception as exc:
            problems.append(f"invalid metadata {metadata_path}: {exc}")
            continue
        identity = (
            int(metadata.get("layer_from", -1)), int(metadata.get("layer_to", -1)),
            metadata.get("variant"),
        )
        if identity != (layer_from, layer_to, variant):
            problems.append(f"identity mismatch: {metadata_path}")
        if metadata.get("status") != "complete":
            problems.append(f"incomplete shard {directory}: {metadata.get('status')!r}")
        if int(metadata.get("feature_dim", -1)) != feature_dim(variant):
            problems.append(f"feature width mismatch: {metadata_path}")
        if metadata.get("distance_probe_layers") != [layer_from, layer_to]:
            problems.append(f"distance endpoint provenance mismatch: {metadata_path}")
        if int(metadata.get("lsqr_max_iter", -1)) != LSQR_MAX_ITER:
            problems.append(f"LSQR iteration configuration mismatch: {metadata_path}")
        for filename in required_files:
            if not (directory / filename).exists():
                problems.append(f"missing output: {directory / filename}")
        validated.append((layer_from, layer_to, variant, directory, metadata))
    if problems:
        raise RuntimeError(
            "Cannot aggregate local pair-representation update probe:\n  - "
            + "\n  - ".join(problems)
        )
    if len(validated) != len(TRANSITIONS) * len(VARIANTS):
        raise RuntimeError(f"Expected 42 complete shards, found {len(validated)}")
    validate_shared_rows(validated)
    return validated


def canonical_rows(path, key_fields):
    rows = read_csv(path)
    return sorted(
        [{key: row[key] for key in sorted(row)} for row in rows],
        key=lambda row: tuple(row[key] for key in key_fields),
    )


def validate_shared_rows(validated):
    bootstrap_reference = {}
    split_reference = None
    for layer_from, layer_to, _variant, directory, _metadata in validated:
        assignments = canonical_rows(
            directory / "bootstrap_assignments.csv",
            ("bootstrap_replicate", "protein_id"),
        )
        transition = (layer_from, layer_to)
        if transition not in bootstrap_reference:
            bootstrap_reference[transition] = assignments
        elif assignments != bootstrap_reference[transition]:
            raise ValueError(f"Bootstrap assignments differ within {transition}: {directory}")
        if len(assignments) != BOOTSTRAP_REPLICATES * 111:
            raise ValueError(f"Unexpected bootstrap assignment count in {directory}")
        splits = canonical_rows(directory / "split_assignments.csv", ("protein_id",))
        if split_reference is None:
            split_reference = splits
        elif splits != split_reference:
            raise ValueError(f"Split assignments differ in {directory}")


def merge_rows(validated, filename):
    rows, fieldnames = [], None
    for _layer_from, _layer_to, _variant, directory, _metadata in validated:
        with (directory / filename).open(newline="") as handle:
            reader = csv.DictReader(handle)
            current = list(reader.fieldnames or [])
            if fieldnames is None:
                fieldnames = current
            elif current != fieldnames:
                raise ValueError(f"Schema mismatch in {directory / filename}")
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


def finite(value):
    try:
        value = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return value if math.isfinite(value) else float("nan")


def variant_gain_rows(model_rows, bootstrap_rows):
    point = {}
    for row in model_rows:
        if row["split"] != "test":
            continue
        transition = (int(row["layer_from"]), int(row["layer_to"]))
        point[(transition, row["variant"], "pair_r2")] = finite(row["pair_r2"])
        point[(transition, row["variant"], "mean_per_protein_r2")] = finite(
            row["mean_per_protein_r2"]
        )
    replicate = {}
    for row in bootstrap_rows:
        key = (
            (int(row["layer_from"]), int(row["layer_to"])), row["variant"],
            row["metric"], int(row["bootstrap_replicate"]),
        )
        replicate[key] = finite(row["value"])
    rows = []
    for numerator, denominator in CONTRASTS:
        for metric in ("pair_r2", "mean_per_protein_r2"):
            transition_differences = {}
            for transition in TRANSITIONS:
                values = np.asarray([
                    replicate.get((transition, numerator, metric, index), np.nan)
                    - replicate.get((transition, denominator, metric, index), np.nan)
                    for index in range(BOOTSTRAP_REPLICATES)
                ])
                values = values[np.isfinite(values)]
                transition_differences[transition] = values
                rows.append({
                    "scope": "transition", "layer_from": transition[0],
                    "layer_to": transition[1], "metric": metric,
                    "contrast": f"{numerator}_minus_{denominator}",
                    "numerator_variant": numerator, "denominator_variant": denominator,
                    "point_gain": point[(transition, numerator, metric)] - point[(transition, denominator, metric)],
                    "bootstrap_mean_gain": np.mean(values) if values.size else float("nan"),
                    "ci_lower_95": np.quantile(values, 0.025) if values.size else float("nan"),
                    "ci_upper_95": np.quantile(values, 0.975) if values.size else float("nan"),
                    "n_paired_replicates": values.size,
                })
            common = min((values.size for values in transition_differences.values()), default=0)
            aggregate = np.mean(
                np.stack([transition_differences[transition][:common] for transition in TRANSITIONS]),
                axis=0,
            ) if common else np.empty(0)
            point_values = [
                point[(transition, numerator, metric)] - point[(transition, denominator, metric)]
                for transition in TRANSITIONS
            ]
            rows.append({
                "scope": "equal_transition_mean", "layer_from": "all", "layer_to": "all",
                "metric": metric, "contrast": f"{numerator}_minus_{denominator}",
                "numerator_variant": numerator, "denominator_variant": denominator,
                "point_gain": np.mean(point_values),
                "bootstrap_mean_gain": np.mean(aggregate) if aggregate.size else float("nan"),
                "ci_lower_95": np.quantile(aggregate, 0.025) if aggregate.size else float("nan"),
                "ci_upper_95": np.quantile(aggregate, 0.975) if aggregate.size else float("nan"),
                "n_paired_replicates": aggregate.size,
            })
    return rows


def summary_text(model_rows, gain_rows, metadata_rows):
    lines = [
        "Expanded local pair-representation update probe",
        "===============================================",
        "", "Status: complete", f"Completed shards: {len(metadata_rows)}/42",
        "Transitions: 0->1, 10->11, 20->21, 30->31, 40->41, 46->47",
        "Variants: center, triangle radii 1, 2, 3, 5, 7, 9",
        "Model: protein-equal float64 Ridge regression via LSQR", "",
        "Mean held-out pair R2 across transitions:",
    ]
    scores = {}
    for row in model_rows:
        if row["split"] == "test":
            scores.setdefault(row["variant"], []).append(finite(row["pair_r2"]))
    for variant in VARIANTS:
        lines.append(f"  {variant}: {np.mean(scores[variant]):.6f}")
    lines.extend(["", "Adjacent-radius held-out pair-R2 gains:"])
    for row in gain_rows:
        if row["scope"] == "equal_transition_mean" and row["metric"] == "pair_r2":
            lines.append(
                f"  {row['contrast']}: {float(row['point_gain']):+.6f} "
                f"[95% {float(row['ci_lower_95']):+.6f}, {float(row['ci_upper_95']):+.6f}]"
            )
    capped = [
        metadata for metadata in metadata_rows
        if metadata.get("lsqr", {}).get("iteration_cap_reached")
    ]
    lines.extend(["", "LSQR iteration-cap flags:"])
    if capped:
        for metadata in capped:
            lines.append(
                f"  {metadata['layer_from']}->{metadata['layer_to']} {metadata['variant']}: "
                f"{metadata['lsqr']['n_iter']}/{metadata['lsqr_max_iter']}"
            )
    else:
        lines.append("  none")
    lines.extend([
        "", "Amino-acid category intervals resample held-out proteins without refitting.",
        "Interpret category R2 together with category support and target variance.",
    ])
    return "\n".join(lines) + "\n"


def main():
    csv_root = Path(os.environ.get("CSV_ROOT", Path.home() / "csv_files")).expanduser()
    output_root = Path(
        os.environ.get("LOCAL_PAIR_REP_UPDATE_OUTPUT_ROOT", csv_root / OUTPUT_NAME)
    ).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    validated = validate_shards(output_root)

    simple = (
        "model_summary.csv", "per_protein_metrics.csv", "regression_magnitude_metrics.csv",
        "aa_pair_regression_metrics.csv", "aa_pair_bootstrap_summary.csv",
        "bootstrap_metrics.csv", "coefficients.csv", "protein_manifest.csv",
        "skipped_proteins.csv", "timing.csv",
    )
    merged = {}
    for filename in simple:
        fields, rows = merge_rows(validated, filename)
        write_csv_atomic(output_root / filename, rows, fields)
        merged[filename] = rows

    dedupe_specs = {
        "aa_pair_target_statistics.csv": ("layer_from", "layer_to", "split", "aa_pair_index"),
        "aa_pair_target_uncertainty.csv": ("layer_from", "layer_to", "split", "aa_pair_index"),
        "bootstrap_assignments.csv": ("layer_from", "layer_to", "bootstrap_replicate", "protein_id"),
        "feature_schema.csv": ("variant", "feature_index"),
        "feature_transforms.csv": ("layer_from", "layer_to", "variant", "feature_index"),
        "change_statistics.csv": ("layer_from", "layer_to", "split"),
        "coefficients_used.csv": ("layer", "feature_index"),
        "split_assignments.csv": ("split_id", "protein_id"),
    }
    for filename, keys in dedupe_specs.items():
        fields, rows = merge_rows(validated, filename)
        rows = deduplicate(rows, keys, filename)
        write_csv_atomic(output_root / filename, rows, fields)
        merged[filename] = rows

    gains = variant_gain_rows(merged["model_summary.csv"], merged["bootstrap_metrics.csv"])
    write_csv_atomic(output_root / "variant_gain_summary.csv", gains)
    metadata_rows = [metadata for *_prefix, metadata in validated]
    metadata = {
        "status": "complete", "completed_at": now_timestamp(),
        "n_shards": len(validated), "transitions": [list(value) for value in TRANSITIONS],
        "variants": list(VARIANTS), "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "shards": metadata_rows,
    }
    write_json_atomic(output_root / "run_metadata.json", metadata)
    write_text_atomic(
        output_root / "summary.txt",
        summary_text(merged["model_summary.csv"], gains, metadata_rows),
    )
    print(f"Aggregated {len(validated)} complete shards into {output_root}", flush=True)


if __name__ == "__main__":
    main()

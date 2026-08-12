#!/usr/bin/env python3
"""Validate and aggregate all biological-context update-probe shards."""

from __future__ import annotations

import csv
import json
import math
import os
import shutil
from pathlib import Path

import numpy as np

from local_pair_biological_context_common import (
    BOOTSTRAP_REPLICATES, BUNDLES, BUNDLE_WIDTHS, HIST_GB_MAX_ITER,
    OUTPUT_NAME, TRANSITIONS, now_timestamp, write_csv_atomic,
    write_json_atomic, write_text_atomic,
)


REQUIRED_TABLES = (
    "model_summary.csv", "feature_schema.csv", "feature_transforms.csv",
    "logistic_regularization_search.csv", "logistic_validation_assignments.csv",
    "classification_magnitude_metrics.csv", "regression_magnitude_metrics.csv",
    "per_protein_metrics.csv", "linear_coefficients.csv", "tree_nodes.csv",
    "tree_rule_metrics.csv", "feature_family_importance.csv",
    "triangle_distance_correlations.csv", "bootstrap_assignments.csv",
    "bootstrap_metrics.csv", "change_statistics.csv", "protein_manifest.csv",
    "skipped_proteins.csv", "split_assignments.csv", "coefficients_used.csv",
    "timing.csv", "summary.txt",
)
CONTRASTS = (
    ("local_biology", "endpoint_biology"),
    ("local_triangle", "controls"),
    ("biology_local_triangle", "local_triangle"),
    ("biology_local_triangle_mediator", "biology_local_triangle"),
    ("biology_local_triangle_plus_center", "biology_local_triangle"),
)


def expected_specs():
    return [(*transition, bundle) for transition in TRANSITIONS for bundle in BUNDLES]


def shard_path(output_root, layer_from, layer_to, bundle):
    return Path(output_root) / "shards" / f"transition_{layer_from:02d}_{layer_to:02d}" / bundle


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def validate_shards(output_root, required_tables=REQUIRED_TABLES):
    validated, problems = [], []
    for layer_from, layer_to, bundle in expected_specs():
        directory = shard_path(output_root, layer_from, layer_to, bundle)
        metadata_path = directory / "run_metadata.json"
        if not metadata_path.exists():
            problems.append(f"missing metadata: {metadata_path}")
            continue
        try:
            metadata = json.loads(metadata_path.read_text())
        except Exception as exc:
            problems.append(f"invalid metadata {metadata_path}: {exc}")
            continue
        if metadata.get("status") != "complete":
            problems.append(f"incomplete shard {directory}: {metadata.get('status')}")
        if (int(metadata.get("layer_from", -1)), int(metadata.get("layer_to", -1)), metadata.get("bundle")) != (layer_from, layer_to, bundle):
            problems.append(f"identity mismatch: {metadata_path}")
        if int(metadata.get("feature_dim", -1)) != BUNDLE_WIDTHS[bundle]:
            problems.append(f"feature width mismatch: {metadata_path}")
        if metadata.get("hist_gradient_boosting_early_stopping") is not False:
            problems.append(f"HistGB early stopping not false: {metadata_path}")
        if int(metadata.get("hist_gradient_boosting_max_iter", -1)) != HIST_GB_MAX_ITER:
            problems.append(f"HistGB max_iter mismatch: {metadata_path}")
        if metadata.get("distance_probe_layers") != [layer_from, layer_to]:
            problems.append(f"endpoint layer provenance mismatch: {metadata_path}")
        hashes = metadata.get("distance_probe_sha256", {})
        if not hashes.get(str(layer_from)) or not hashes.get(str(layer_to)) or hashes.get(str(layer_from)) == hashes.get(str(layer_to)):
            problems.append(f"missing or identical endpoint hashes: {metadata_path}")
        for filename in required_tables:
            if not (directory / filename).exists():
                problems.append(f"missing output: {directory / filename}")
        validated.append((layer_from, layer_to, bundle, directory, metadata))
    if problems:
        raise RuntimeError("Cannot aggregate biological-context probe:\n  - " + "\n  - ".join(problems))
    if len(validated) != len(TRANSITIONS) * len(BUNDLES):
        raise RuntimeError(f"Expected 35 shards, found {len(validated)}")
    validate_shared_assignments(validated)
    return validated


def canonical_rows(path, key_fields):
    rows = read_csv(path)
    return sorted([{key: row[key] for key in sorted(row)} for row in rows], key=lambda row: tuple(row[key] for key in key_fields))


def validate_shared_assignments(validated):
    validation_reference = None
    bootstrap_reference = {}
    for layer_from, layer_to, _bundle, directory, _metadata in validated:
        validation = canonical_rows(directory / "logistic_validation_assignments.csv", ("protein_id",))
        if validation_reference is None:
            validation_reference = validation
        elif validation != validation_reference:
            raise ValueError(f"Logistic validation assignments differ in {directory}")
        bootstrap = canonical_rows(
            directory / "bootstrap_assignments.csv",
            ("bootstrap_replicate", "protein_id"),
        )
        key = (layer_from, layer_to)
        if key not in bootstrap_reference:
            bootstrap_reference[key] = bootstrap
        elif bootstrap != bootstrap_reference[key]:
            raise ValueError(f"Bootstrap assignments differ within transition {key}: {directory}")
        expected = BOOTSTRAP_REPLICATES * 111
        if len(bootstrap) != expected:
            raise ValueError(f"Expected {expected} bootstrap assignment rows in {directory}, found {len(bootstrap)}")


def merge_rows(validated, filename):
    rows, fieldnames = [], None
    for _lf, _lt, _bundle, directory, _metadata in validated:
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
            raise ValueError(f"Conflicting duplicate {label}: {key}")
        unique[key] = row
    return list(unique.values())


def finite(value):
    try:
        result = float(value)
    except (TypeError, ValueError):
        return float("nan")
    return result if math.isfinite(result) else float("nan")


def feature_ablation_rows(model_rows, bootstrap_rows):
    point_lookup = {}
    for row in model_rows:
        if row["split"] != "test" or row["estimator"] == "decision_tree_depth6":
            continue
        metric = "mean_per_protein_balanced_accuracy" if row["target"] == "classification" else "mean_per_protein_r2"
        point_lookup[(row["layer_from"], row["layer_to"], row["bundle"], row["target"], row["estimator"], metric)] = finite(row.get(metric))
    replicate_lookup = {}
    for row in bootstrap_rows:
        if row["aggregation_scope"] != "mean_per_protein":
            continue
        metric = "balanced_accuracy" if row["target"] == "classification" else "r2"
        if row["metric"] != metric:
            continue
        key = (row["layer_from"], row["layer_to"], row["bundle"], row["target"], row["estimator"])
        replicate_lookup.setdefault(key, {})[int(row["bootstrap_replicate"])] = finite(row["value"])
    output = []
    for layer_from, layer_to in TRANSITIONS:
        for target in ("classification", "regression"):
            metric = "mean_per_protein_balanced_accuracy" if target == "classification" else "mean_per_protein_r2"
            for estimator in ("logistic" if target == "classification" else "ridge", "decision_tree_depth4", "hist_gradient_boosting"):
                for numerator, denominator in CONTRASTS:
                    point_num = point_lookup.get((str(layer_from), str(layer_to), numerator, target, estimator, metric), float("nan"))
                    point_den = point_lookup.get((str(layer_from), str(layer_to), denominator, target, estimator, metric), float("nan"))
                    first = replicate_lookup.get((str(layer_from), str(layer_to), numerator, target, estimator), {})
                    second = replicate_lookup.get((str(layer_from), str(layer_to), denominator, target, estimator), {})
                    replicate_ids = sorted(set(first) & set(second))
                    differences = np.asarray([first[index] - second[index] for index in replicate_ids], dtype=np.float64)
                    differences = differences[np.isfinite(differences)]
                    output.append({
                        "layer_from": layer_from, "layer_to": layer_to, "target": target,
                        "estimator": estimator, "contrast": f"{numerator}_minus_{denominator}",
                        "numerator_bundle": numerator, "denominator_bundle": denominator,
                        "metric": metric, "point_gain": point_num - point_den,
                        "bootstrap_mean_gain": np.mean(differences) if differences.size else float("nan"),
                        "ci_lower_95": np.quantile(differences, 0.025) if differences.size else float("nan"),
                        "ci_upper_95": np.quantile(differences, 0.975) if differences.size else float("nan"),
                        "n_paired_replicates": differences.size,
                    })
    return output


def copy_artifacts(validated, output_root):
    for layer_from, layer_to, bundle, directory, _metadata in validated:
        transition = f"transition_{layer_from:02d}_{layer_to:02d}"
        for source_name in ("models", "tree_structures"):
            source = directory / source_name
            destination = Path(output_root) / source_name / transition / bundle
            destination.mkdir(parents=True, exist_ok=True)
            for path in source.iterdir():
                shutil.copy2(path, destination / path.name)


def summary_text(model_rows, ablation_rows, rule_rows, correlations):
    lines = [
        "Local biological-context distance-update probe", "=" * 47, "",
        "Status: complete", "Transitions: 10->11, 15->16, 20->21, 30->31, 40->41",
        "Completed shards: 35/35", "HistGradientBoosting: 100 iterations; early_stopping=False", "",
        "Held-out bundle rankings:",
    ]
    for target, estimator, metric in (
        ("classification", "logistic", "mean_per_protein_balanced_accuracy"),
        ("classification", "decision_tree_depth4", "mean_per_protein_balanced_accuracy"),
        ("classification", "hist_gradient_boosting", "mean_per_protein_balanced_accuracy"),
        ("regression", "ridge", "mean_per_protein_r2"),
        ("regression", "decision_tree_depth4", "mean_per_protein_r2"),
        ("regression", "hist_gradient_boosting", "mean_per_protein_r2"),
    ):
        scores = {}
        for row in model_rows:
            if row["split"] == "test" and row["target"] == target and row["estimator"] == estimator:
                scores.setdefault(row["bundle"], []).append(finite(row.get(metric)))
        ranked = sorted(((bundle, np.nanmean(values)) for bundle, values in scores.items()), key=lambda item: (-item[1], item[0]))
        lines.append(f"  {target}/{estimator} ({metric}): " + ", ".join(f"{bundle}={score:.4f}" for bundle, score in ranked))
    lines.extend(["", "Declared paired bundle gains (held-out protein bootstrap):"])
    for row in ablation_rows:
        if row["estimator"] not in {"logistic", "ridge", "decision_tree_depth4", "hist_gradient_boosting"}:
            continue
        lines.append(
            f"  {row['layer_from']}->{row['layer_to']} {row['target']}/{row['estimator']} "
            f"{row['contrast']}: {float(row['point_gain']):+.4f} "
            f"[95% {float(row['ci_lower_95']):+.4f}, {float(row['ci_upper_95']):+.4f}]"
        )
    lines.extend(["", "Most directional held-out depth-4 rules (minimum 10 proteins):"])
    eligible_rules = [
        row for row in rule_rows
        if row.get("split") == "test"
        and row.get("estimator") == "decision_tree_depth4"
        and int(row.get("n_proteins", 0)) >= 10
    ]
    def rule_strength(row):
        if row["target"] == "classification":
            return abs(finite(row.get("shrink_prevalence")) - 0.5)
        return abs(finite(row.get("mean_target")))
    eligible_rules.sort(key=lambda row: (-rule_strength(row), -int(row.get("n_proteins", 0))))
    for row in eligible_rules[:15]:
        effect = (
            f"shrink prevalence={finite(row.get('shrink_prevalence')):.3f}"
            if row["target"] == "classification"
            else f"mean delta={finite(row.get('mean_target')):+.3f} A"
        )
        lines.append(
            f"  {row['layer_from']}->{row['layer_to']} {row['bundle']} {row['target']} "
            f"leaf {row['leaf_id']} ({row['n_proteins']} proteins): {effect}; "
            f"pressure-distance label={row.get('distance_collinearity_label', 'none')}; "
            f"rule={row['rule_path']}"
        )
    warning_rules = [row for row in rule_rows if row.get("split") == "test" and row.get("distance_collinearity_label") in {"moderate", "high"}]
    lines.extend(["", "Triangle-pressure rule collinearity:"])
    if not warning_rules:
        lines.append("  No held-out pressure-based rules had moderate/high current-distance correlation.")
    else:
        for row in warning_rules[:30]:
            lines.append(
                f"  {row['layer_from']}->{row['layer_to']} {row['bundle']} {row['target']} "
                f"leaf {row['leaf_id']}: {row['distance_collinearity_label']} correlation; "
                f"interpret geometric attribution as {'ambiguous' if row['distance_collinearity_label']=='high' else 'cautious'}."
            )
        if len(warning_rules) > 30:
            lines.append(f"  ... {len(warning_rules) - 30} additional flagged rule rows in tree_rule_metrics.csv")
    lines.extend(["", "Interpretation: predictive associations are diagnostic, not causal evidence."])
    return "\n".join(lines) + "\n"


def main():
    csv_root = Path(os.environ.get("CSV_ROOT", Path.home() / "csv_files")).expanduser()
    output_root = Path(os.environ.get("LOCAL_BIO_CONTEXT_OUTPUT_ROOT", csv_root / OUTPUT_NAME)).expanduser()
    output_root.mkdir(parents=True, exist_ok=True)
    validated = validate_shards(output_root)
    simple = (
        "model_summary.csv", "logistic_regularization_search.csv",
        "classification_magnitude_metrics.csv", "regression_magnitude_metrics.csv",
        "per_protein_metrics.csv", "linear_coefficients.csv", "tree_nodes.csv",
        "tree_rule_metrics.csv", "feature_family_importance.csv", "bootstrap_metrics.csv",
        "protein_manifest.csv", "skipped_proteins.csv", "timing.csv",
    )
    merged = {}
    for filename in simple:
        fields, rows = merge_rows(validated, filename)
        write_csv_atomic(output_root / filename, rows, fields)
        merged[filename] = rows
    dedupe_specs = {
        "feature_schema.csv": ("bundle", "feature_index"),
        "feature_transforms.csv": ("layer_from", "layer_to", "bundle", "feature_index"),
        "logistic_validation_assignments.csv": ("protein_id",),
        "triangle_distance_correlations.csv": ("layer_from", "layer_to", "split", "feature_name"),
        "bootstrap_assignments.csv": ("layer_from", "layer_to", "bootstrap_replicate", "protein_id"),
        "change_statistics.csv": ("layer_from", "layer_to", "split"),
        "split_assignments.csv": ("split_id", "protein_id"),
        "coefficients_used.csv": ("layer", "feature_index"),
    }
    for filename, keys in dedupe_specs.items():
        fields, rows = merge_rows(validated, filename)
        rows = deduplicate(rows, keys, filename)
        write_csv_atomic(output_root / filename, rows, fields)
        merged[filename] = rows
    ablation_rows = feature_ablation_rows(merged["model_summary.csv"], merged["bootstrap_metrics.csv"])
    write_csv_atomic(output_root / "feature_ablation_summary.csv", ablation_rows)
    copy_artifacts(validated, output_root)
    metadata = {
        "status": "complete", "completed_at": now_timestamp(), "n_shards": len(validated),
        "transitions": [list(value) for value in TRANSITIONS], "bundles": list(BUNDLES),
        "bundle_widths": BUNDLE_WIDTHS, "hist_gradient_boosting_max_iter": HIST_GB_MAX_ITER,
        "hist_gradient_boosting_early_stopping": False,
        "shards": [metadata for *_prefix, metadata in validated],
    }
    write_json_atomic(output_root / "run_metadata.json", metadata)
    write_text_atomic(output_root / "summary.txt", summary_text(
        merged["model_summary.csv"], ablation_rows, merged["tree_rule_metrics.csv"],
        merged["triangle_distance_correlations.csv"],
    ))
    print(f"Aggregated {len(validated)} complete shards into {output_root}", flush=True)


if __name__ == "__main__":
    main()

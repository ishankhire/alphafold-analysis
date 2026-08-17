#!/usr/bin/env python3
"""Validate gated task artifacts, finish joint recombination, and consolidate outputs."""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
for import_dir in (SCRIPT_DIR, CLUSTER_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from gated_linear_probe_common import (  # noqa: E402
    ANALYSIS_AA,
    ANALYSIS_JOINT,
    ANALYSIS_NAMES,
    ANALYSIS_SEQUENCE,
    FEATURE_DIM,
    MODEL_FAMILY,
    analysis_configuration_digest,
    analysis_configuration_payload,
    build_layer_shard,
    configured_layers,
    expected_gates,
    gate_slug,
    joint_task_for_id,
    layer_for_task,
    load_context,
    log,
    merge_prediction_fragments,
    peak_rss_mb,
    record_offsets,
    run_mode,
    suite_root,
    validate_unique_rows,
)
from regression_output_helpers import (  # noqa: E402
    COEFFICIENT_INDEX_COLUMNS,
    CSV_SCHEMAS,
    GEOMETRY_METRIC_NAMES,
    atomic_savez_compressed,
    atomic_write_csv,
    atomic_write_json,
    build_run_manifest,
    build_runtime_row,
    sha256_file,
)
from regression_protocol import MASTER_MANIFEST_DIGEST  # noqa: E402


SHARD_FILENAMES = (
    "per_protein.csv",
    "pooled_metrics.csv",
    "model_status.csv",
    "runtime.csv",
    "geometry_metrics.csv",
    "geometry_summary.csv",
    "subgroup_metrics/by_sequence_separation.csv",
    "subgroup_metrics/by_amino_acid_pair.csv",
    "subgroup_metrics/by_spatial_distance.csv",
)


def aggregation_destination(working_root: str | Path) -> Path:
    return Path(working_root) / "consolidated"


def task_count(analysis_name: str, mode: str) -> int:
    layers = configured_layers(mode)
    return len(layers) * (len(expected_gates(ANALYSIS_SEQUENCE)) if analysis_name == ANALYSIS_JOINT else 1)


def _task_mapping(analysis_name: str, task_id: int, mode: str) -> tuple[int, int | None]:
    if analysis_name == ANALYSIS_JOINT:
        return joint_task_for_id(task_id, mode)
    return layer_for_task(task_id, mode), None


def validate_task(
    root: Path,
    analysis_name: str,
    task_id: int,
    preparation_manifest: Mapping[str, object],
) -> dict[str, object]:
    path = root / analysis_name / "task_shards" / f"task_{task_id:03d}" / "task_manifest.json"
    if not path.exists():
        raise ValueError(f"Missing task manifest: {path}")
    manifest = json.loads(path.read_text())
    layer, sequence_code = _task_mapping(
        analysis_name, task_id, str(preparation_manifest["run_mode"]),
    )
    expected_slugs = [gate_slug(*gate) for gate in expected_gates(analysis_name, sequence_code)]
    invariants = {
        "analysis_name": analysis_name,
        "task_id": task_id,
        "layer": layer,
        "sequence_task_code": sequence_code,
        "configuration_digest": preparation_manifest["configuration_digest"],
        "analysis_configuration_digest": analysis_configuration_digest(
            analysis_name,
            str(preparation_manifest["run_mode"]),
            preparation_manifest,
        ),
        "master_split_digest": MASTER_MANIFEST_DIGEST,
        "expected_gate_keys": expected_slugs,
    }
    for key, expected in invariants.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Task {task_id} mismatch for {key}: {manifest.get(key)!r} != {expected!r}")
    if not manifest.get("completed"):
        raise ValueError(f"Task is incomplete: {path}")
    if manifest.get("completed_gate_keys") != expected_slugs:
        raise ValueError(f"Task {task_id} gate checkpoint list is incomplete or reordered")
    task_dir = path.parent
    prediction_path = task_dir / "task_predictions.npz"
    if not prediction_path.exists() or sha256_file(prediction_path) != manifest["task_prediction_sha256"]:
        raise ValueError(f"Missing or corrupt task prediction artifact: {prediction_path}")
    for slug in expected_slugs:
        artifacts = manifest["artifacts"][slug]
        for suffix, digest_key in (
            ("_status.json", "status_sha256"),
            ("_coefficients.npz", "coefficient_sha256"),
            ("_predictions.npz", "prediction_sha256"),
        ):
            artifact = task_dir / "gates" / f"{slug}{suffix}"
            if not artifact.exists() or sha256_file(artifact) != artifacts[digest_key]:
                raise ValueError(f"Missing or corrupt gate artifact: {artifact}")
    return manifest


def _status_rows_for_task(root: Path, analysis_name: str, task_manifest: Mapping[str, object]) -> list[dict[str, object]]:
    task_dir = root / analysis_name / "task_shards" / f"task_{int(task_manifest['task_id']):03d}"
    return [
        json.loads((task_dir / "gates" / f"{slug}_status.json").read_text())
        for slug in task_manifest["expected_gate_keys"]
    ]


def finish_joint_layer_shards(
    root: Path,
    records: Sequence[Mapping[str, object]],
    preparation_manifest: Mapping[str, object],
    task_manifests: Sequence[Mapping[str, object]],
    *,
    layers: Sequence[int] | None = None,
    skip_existing: bool = False,
) -> None:
    offsets, total_pairs = record_offsets(records)
    development_ids = {
        str(record["protein_id"])
        for record in records
        if record["master_role"] in {"fit", "validation"}
    }
    requested_layers = tuple(layers or configured_layers(str(preparation_manifest["run_mode"])))
    for layer in requested_layers:
        shard_manifest = (
            root / ANALYSIS_JOINT / "layer_shards" / f"layer_{layer:02d}"
            / "shard_manifest.json"
        )
        if skip_existing and shard_manifest.exists():
            log(f"Joint layer {layer}: reusing existing canonical layer shard")
            continue
        layer_tasks = sorted(
            (manifest for manifest in task_manifests if int(manifest["layer"]) == layer),
            key=lambda manifest: int(manifest["sequence_task_code"]),
        )
        if len(layer_tasks) != 6:
            raise ValueError(f"Joint layer {layer} has {len(layer_tasks)} task fragments instead of 6")
        prediction_paths = [
            root / ANALYSIS_JOINT / "task_shards" / f"task_{int(manifest['task_id']):03d}"
            / "task_predictions.npz"
            for manifest in layer_tasks
        ]
        flat_predictions, flat_conditional = merge_prediction_fragments(total_pairs, prediction_paths)
        status_rows = []
        for manifest in layer_tasks:
            status_rows.extend(_status_rows_for_task(root, ANALYSIS_JOINT, manifest))
        per_record_evaluated = [
            int(np.count_nonzero(np.isfinite(flat_predictions[offsets[index]:offsets[index + 1]])))
            for index in range(len(records))
        ]
        n_fitted_pairs = sum(
            per_record_evaluated[index]
            for index, record in enumerate(records)
            if str(record["protein_id"]) in development_ids
        )
        runtime = build_runtime_row(
            analysis_name=ANALYSIS_JOINT,
            model_family=MODEL_FAMILY,
            layer=layer,
            completion_status="complete",
            n_fitted_proteins=sum(
                per_record_evaluated[index] > 0
                for index, record in enumerate(records)
                if str(record["protein_id"]) in development_ids
            ),
            n_evaluated_proteins=sum(count > 0 for count in per_record_evaluated),
            n_fitted_pairs=n_fitted_pairs,
            n_evaluated_pairs=int(np.count_nonzero(np.isfinite(flat_predictions))),
            elapsed_seconds=sum(float(manifest["elapsed_seconds"]) for manifest in layer_tasks),
            peak_rss_mb=peak_rss_mb(),
        )
        build_layer_shard(
            root,
            ANALYSIS_JOINT,
            layer,
            records,
            flat_predictions,
            flat_conditional,
            status_rows,
            runtime,
            preparation_manifest,
        )
        log(f"Joint layer {layer}: checkpointed recombined canonical layer shard")


def _read_shard_rows(path: Path, expected_columns: Sequence[str]) -> list[dict[str, str]]:
    if not path.exists():
        raise ValueError(f"Missing canonical shard file: {path}")
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise ValueError(f"Canonical schema mismatch in {path}: {reader.fieldnames}")
        return list(reader)


def validate_layer_shard(
    root: Path,
    analysis_name: str,
    layer: int,
    preparation_manifest: Mapping[str, object],
) -> dict[str, list[dict[str, str]]]:
    shard = root / analysis_name / "layer_shards" / f"layer_{layer:02d}"
    manifest_path = shard / "shard_manifest.json"
    if not manifest_path.exists():
        raise ValueError(f"Missing layer shard manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_configuration_digest = analysis_configuration_digest(
        analysis_name,
        str(preparation_manifest["run_mode"]),
        preparation_manifest,
    )
    if manifest.get("configuration_digest") != expected_configuration_digest:
        raise ValueError(f"Layer {layer} configuration digest mismatch")
    if manifest.get("master_split_digest") != MASTER_MANIFEST_DIGEST:
        raise ValueError(f"Layer {layer} split digest mismatch")
    result = {}
    for filename in SHARD_FILENAMES:
        path = shard / filename
        if manifest["files"].get(filename) != sha256_file(path):
            raise ValueError(f"Layer shard checksum mismatch: {path}")
        result[filename] = _read_shard_rows(path, CSV_SCHEMAS[filename])
    return result


def build_coefficient_artifacts(
    root: Path,
    destination: Path,
    analysis_name: str,
    task_manifests: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    index_rows = []
    shard_dir = destination / analysis_name / "model_artifacts" / "coefficient_shards"
    for manifest in sorted(task_manifests, key=lambda row: int(row["task_id"])):
        task_id = int(manifest["task_id"])
        task_dir = root / analysis_name / "task_shards" / f"task_{task_id:03d}"
        statuses = _status_rows_for_task(root, analysis_name, manifest)
        n_models = len(statuses)
        coefficient_standardized = np.full((n_models, FEATURE_DIM), np.nan, dtype=np.float64)
        coefficient_raw_scale = np.full_like(coefficient_standardized, np.nan)
        feature_mean = np.full_like(coefficient_standardized, np.nan)
        feature_scale = np.full_like(coefficient_standardized, np.nan)
        intercept_standardized = np.full(n_models, np.nan, dtype=np.float64)
        intercept_raw_scale = np.full(n_models, np.nan, dtype=np.float64)
        for row_index, (slug, status) in enumerate(zip(manifest["expected_gate_keys"], statuses)):
            with np.load(task_dir / "gates" / f"{slug}_coefficients.npz") as data:
                coefficient_standardized[row_index] = data["coefficient_standardized"]
                coefficient_raw_scale[row_index] = data["coefficient_raw_scale"]
                feature_mean[row_index] = data["feature_mean"]
                feature_scale[row_index] = data["feature_scale"]
                intercept_standardized[row_index] = data["intercept_standardized"]
                intercept_raw_scale[row_index] = data["intercept_raw_scale"]
        shard_path = shard_dir / f"task_{task_id:03d}.npz"
        atomic_savez_compressed(
            shard_path,
            coefficient_standardized=coefficient_standardized,
            coefficient_raw_scale=coefficient_raw_scale,
            intercept_standardized=intercept_standardized,
            intercept_raw_scale=intercept_raw_scale,
            feature_mean=feature_mean,
            feature_scale=feature_scale,
            layer=np.asarray([int(row["layer"]) for row in statuses], dtype=np.int16),
            gate_aa_pair=np.asarray([row["gate_aa_pair"] for row in statuses]),
            gate_sequence_bin=np.asarray([row["gate_sequence_bin"] for row in statuses]),
            fit_status=np.asarray([row["fit_status"] for row in statuses]),
        )
        digest = sha256_file(shard_path)
        relative_path = str(shard_path.relative_to(destination / analysis_name))
        for row_index, status in enumerate(statuses):
            index_rows.append({
                "analysis_name": analysis_name,
                "model_family": MODEL_FAMILY,
                "layer": int(status["layer"]),
                "protein_id": "",
                "gate_aa_pair": status["gate_aa_pair"],
                "gate_sequence_bin": status["gate_sequence_bin"],
                "shard_path": relative_path,
                "array_row": row_index,
                "fit_status": status["fit_status"],
                "feature_dim": FEATURE_DIM,
                "file_sha256": digest,
            })
    atomic_write_csv(
        destination / analysis_name / "model_artifacts" / "coefficient_index.csv",
        COEFFICIENT_INDEX_COLUMNS,
        index_rows,
    )
    return index_rows


def _validate_consolidated_rows(
    analysis_name: str,
    layers: Sequence[int],
    n_proteins: int,
    tables: Mapping[str, Sequence[Mapping[str, object]]],
) -> None:
    expected_layers = set(layers)
    for filename, rows in tables.items():
        observed = {int(row["layer"]) for row in rows}
        if observed != expected_layers:
            raise ValueError(f"{analysis_name} {filename} layer coverage {observed} != {expected_layers}")
    if len(tables["per_protein.csv"]) != len(layers) * n_proteins:
        raise ValueError(f"{analysis_name} per-protein shard row count mismatch")
    if len(tables["geometry_metrics.csv"]) != len(layers) * n_proteins:
        raise ValueError(f"{analysis_name} geometry shard row count mismatch")
    expected_gates_per_layer = {
        ANALYSIS_SEQUENCE: 6,
        ANALYSIS_AA: 210,
        ANALYSIS_JOINT: 1260,
    }[analysis_name]
    if len(tables["model_status.csv"]) != len(layers) * expected_gates_per_layer:
        raise ValueError(f"{analysis_name} model-status row count mismatch")
    if len(tables["pooled_metrics.csv"]) != 2 * len(layers):
        raise ValueError(f"{analysis_name} pooled row count mismatch")
    if len(tables["runtime.csv"]) != len(layers):
        raise ValueError(f"{analysis_name} runtime row count mismatch")
    if len(tables["geometry_summary.csv"]) != 2 * len(GEOMETRY_METRIC_NAMES) * len(layers):
        raise ValueError(f"{analysis_name} geometry-summary row count mismatch")
    expected_subgroup_rows = {
        "subgroup_metrics/by_sequence_separation.csv": 6,
        "subgroup_metrics/by_amino_acid_pair.csv": 210,
        "subgroup_metrics/by_spatial_distance.csv": 5,
    }
    for filename, rows_per_layer in expected_subgroup_rows.items():
        if len(tables[filename]) != rows_per_layer * len(layers):
            raise ValueError(f"{analysis_name} {filename} row count mismatch")
    validate_unique_rows(
        tables["per_protein.csv"],
        ("analysis_name", "layer", "protein_id", "evaluation_scope"),
        f"{analysis_name} per_protein.csv",
    )
    validate_unique_rows(
        tables["model_status.csv"],
        ("analysis_name", "layer", "gate_aa_pair", "gate_sequence_bin"),
        f"{analysis_name} model_status.csv",
    )
    validate_unique_rows(
        tables["pooled_metrics.csv"],
        ("analysis_name", "layer", "protein_split", "evaluation_scope"),
        f"{analysis_name} pooled_metrics.csv",
    )
    validate_unique_rows(
        tables["geometry_metrics.csv"],
        ("analysis_name", "layer", "protein_id", "evaluation_scope"),
        f"{analysis_name} geometry_metrics.csv",
    )
    validate_unique_rows(
        tables["geometry_summary.csv"],
        ("analysis_name", "layer", "protein_split", "evaluation_scope", "geometry_metric"),
        f"{analysis_name} geometry_summary.csv",
    )
    validate_unique_rows(
        tables["runtime.csv"],
        ("analysis_name", "layer"),
        f"{analysis_name} runtime.csv",
    )
    for filename in expected_subgroup_rows:
        validate_unique_rows(
            tables[filename],
            ("analysis_name", "layer", "protein_split", "evaluation_scope", "category_label"),
            f"{analysis_name} {filename}",
        )


def consolidate_analysis(
    root: Path,
    destination: Path,
    analysis_name: str,
    records: Sequence[Mapping[str, object]],
    preparation_manifest: Mapping[str, object],
    task_manifests: Sequence[Mapping[str, object]],
) -> None:
    layers = configured_layers(str(preparation_manifest["run_mode"]))
    tables = {filename: [] for filename in SHARD_FILENAMES}
    output_shards = destination / analysis_name / "layer_shards"
    for layer in layers:
        validated = validate_layer_shard(
            root, analysis_name, layer, preparation_manifest,
        )
        for filename, rows in validated.items():
            tables[filename].extend(rows)
        source = root / analysis_name / "layer_shards" / f"layer_{layer:02d}"
        shutil.copytree(
            source, output_shards / f"layer_{layer:02d}", dirs_exist_ok=True,
        )
    _validate_consolidated_rows(analysis_name, layers, len(records), tables)
    for filename, rows in tables.items():
        atomic_write_csv(destination / analysis_name / filename, CSV_SCHEMAS[filename], rows)
    coefficient_index = build_coefficient_artifacts(
        root, destination, analysis_name, task_manifests,
    )
    if len(coefficient_index) != len(tables["model_status.csv"]):
        raise ValueError(f"{analysis_name} coefficient index does not cover every model-status row")
    try:
        code_revision = subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        code_revision = ""
    run_manifest = build_run_manifest(
        analysis_name=analysis_name,
        model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM,
        requested_layers=layers,
        completed_layers=layers,
        split_digest=MASTER_MANIFEST_DIGEST,
        configuration=analysis_configuration_payload(
            analysis_name,
            str(preparation_manifest["run_mode"]),
            preparation_manifest,
        ),
        versions=preparation_manifest.get("versions", {}),
        code_revision=code_revision,
        run_status="complete",
    )
    atomic_write_json(destination / analysis_name / "run_manifest.json", run_manifest)
    log(f"Consolidated {analysis_name}: {len(layers)} layers")


def main() -> None:
    mode = run_mode()
    root = suite_root(mode)
    destination = aggregation_destination(root)
    preparation_manifest, records = load_context(root)
    if preparation_manifest["run_mode"] != mode:
        raise ValueError("Requested aggregation mode differs from preparation mode")
    manifests_by_analysis = {}
    for analysis_name in ANALYSIS_NAMES:
        manifests_by_analysis[analysis_name] = [
            validate_task(root, analysis_name, task_id, preparation_manifest)
            for task_id in range(task_count(analysis_name, mode))
        ]
    finish_joint_layer_shards(
        root,
        records,
        preparation_manifest,
        manifests_by_analysis[ANALYSIS_JOINT],
        skip_existing=True,
    )
    destination.mkdir(parents=True, exist_ok=True)
    for analysis_name in ANALYSIS_NAMES:
        consolidate_analysis(
            root,
            destination,
            analysis_name,
            records,
            preparation_manifest,
            manifests_by_analysis[analysis_name],
        )
    log(f"Canonical gated-suite aggregation complete: {destination}")


if __name__ == "__main__":
    main()

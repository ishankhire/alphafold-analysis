#!/usr/bin/env python3
"""Validate and consolidate canonical per-protein base256 layer shards."""

from __future__ import annotations

import csv
import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[3]
CLUSTER_DIR = ROOT_DIR / "scripts" / "cluster_scripts"
PER_PROTEIN_DIR = CLUSTER_DIR / "per_protein_scripts"
for import_dir in (CLUSTER_DIR, PER_PROTEIN_DIR):
    if str(import_dir) not in sys.path:
        sys.path.append(str(import_dir))

from per_protein_base256_common import (  # noqa: E402
    ANALYSIS_NAME,
    EVALUATION_SCOPES,
    MODEL_FAMILY,
    PAIR_SPLIT_COLUMNS,
    configuration_payload,
    pair_split_manifest_digest,
    version_payload,
)
from regression_output_helpers import (  # noqa: E402
    COEFFICIENT_INDEX_COLUMNS,
    CSV_SCHEMAS,
    GEOMETRY_METRIC_NAMES,
    OUTPUT_SCHEMA_VERSION,
    atomic_write_csv,
    atomic_write_json,
    build_run_manifest,
    semantic_digest,
    sha256_file,
)
from regression_protocol import (  # noqa: E402
    AMINO_ACID_PAIR_LABELS,
    FEATURE_DIM,
    FEATURE_NAMES,
    LAYERS,
    MASTER_MANIFEST_DIGEST,
    SEQUENCE_BIN_LABELS,
    SPATIAL_DISTANCE_BIN_LABELS,
)


OUTPUT_PARENT = Path(os.environ.get("CSV_ROOT", ROOT_DIR / "delta_files"))
INPUT_DIR = OUTPUT_PARENT / ANALYSIS_NAME
EXPECTED_PROTEINS = 552
EXPECTED_LAYERS = tuple(LAYERS)

TABLES = (
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

PRIMARY_KEYS = {
    "per_protein.csv": ("analysis_name", "layer", "protein_id", "evaluation_scope"),
    "pooled_metrics.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope"),
    "model_status.csv": ("analysis_name", "layer", "protein_id"),
    "runtime.csv": ("analysis_name", "layer"),
    "geometry_metrics.csv": ("analysis_name", "layer", "protein_id"),
    "geometry_summary.csv": (
        "analysis_name", "layer", "protein_split", "evaluation_scope", "geometry_metric",
    ),
    "subgroup_metrics/by_sequence_separation.csv": (
        "analysis_name", "layer", "protein_split", "evaluation_scope", "category_label",
    ),
    "subgroup_metrics/by_amino_acid_pair.csv": (
        "analysis_name", "layer", "protein_split", "evaluation_scope", "category_label",
    ),
    "subgroup_metrics/by_spatial_distance.csv": (
        "analysis_name", "layer", "protein_split", "evaluation_scope", "category_label",
    ),
    "model_artifacts/coefficient_index.csv": ("analysis_name", "layer", "protein_id"),
}


def read_csv_exact(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(
                f"{path} schema mismatch: expected {tuple(columns)}, got {reader.fieldnames}"
            )
        return list(reader)


def read_json(path: Path) -> dict[str, object]:
    with path.open() as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} must contain a JSON object")
    return payload


def validate_unique(filename: str, rows: Sequence[Mapping[str, str]]) -> None:
    key_fields = PRIMARY_KEYS[filename]
    seen = set()
    for row in rows:
        key = tuple(row[field] for field in key_fields)
        if key in seen:
            raise ValueError(f"Duplicate {filename} primary key: {key}")
        seen.add(key)


def expected_layer_counts(n_proteins: int) -> dict[str, int]:
    return {
        "per_protein.csv": n_proteins * 3,
        "pooled_metrics.csv": 6,
        "model_status.csv": n_proteins,
        "runtime.csv": 1,
        "geometry_metrics.csv": n_proteins,
        "geometry_summary.csv": 2 * len(GEOMETRY_METRIC_NAMES),
        "subgroup_metrics/by_sequence_separation.csv": len(SEQUENCE_BIN_LABELS),
        "subgroup_metrics/by_amino_acid_pair.csv": len(AMINO_ACID_PAIR_LABELS),
        "subgroup_metrics/by_spatial_distance.csv": len(SPATIAL_DISTANCE_BIN_LABELS),
        "coefficient_index.csv": n_proteins,
    }


def _validate_layer_rows(
    filename: str,
    rows: Sequence[Mapping[str, str]],
    layer: int,
    expected_count: int,
) -> None:
    if len(rows) != expected_count:
        raise ValueError(
            f"Layer {layer:02d} {filename} has {len(rows)} rows; expected {expected_count}"
        )
    if any(row.get("analysis_name") != ANALYSIS_NAME for row in rows):
        raise ValueError(f"Layer {layer:02d} {filename} contains a wrong analysis name")
    if any(int(row["layer"]) != layer for row in rows):
        raise ValueError(f"Layer {layer:02d} {filename} contains a wrong layer")


def _validate_finite_success_metrics(rows: Sequence[Mapping[str, str]]) -> None:
    metric_fields = ("mae", "rmse", "bias_pred_minus_true")
    for row in rows:
        if row["prediction_status"] != "complete":
            continue
        for field in metric_fields:
            if not math.isfinite(float(row[field])):
                raise ValueError(
                    f"Complete per-protein row has nonfinite {field}: "
                    f"layer={row['layer']} protein={row['protein_id']} "
                    f"scope={row['evaluation_scope']}"
                )


def _validate_coefficient_index(
    root: Path,
    layer: int,
    rows: Sequence[Mapping[str, str]],
    coefficient_relative: str,
    expected_digest: str,
    expected_proteins: int,
) -> None:
    if len(rows) != expected_proteins:
        raise ValueError(f"Layer {layer:02d} coefficient index is incomplete")
    if {row["shard_path"] for row in rows} != {coefficient_relative}:
        raise ValueError(f"Layer {layer:02d} coefficient paths are inconsistent")
    if {row["file_sha256"] for row in rows} != {expected_digest}:
        raise ValueError(f"Layer {layer:02d} coefficient digests are inconsistent")
    coefficient_path = root / coefficient_relative
    if sha256_file(coefficient_path) != expected_digest:
        raise ValueError(f"Layer {layer:02d} coefficient shard digest mismatch")
    array_rows = sorted(int(row["array_row"]) for row in rows)
    if array_rows != list(range(expected_proteins)):
        raise ValueError(f"Layer {layer:02d} coefficient array rows are not exhaustive")
    with np.load(coefficient_path) as archive:
        required = {
            "protein_id", "fit_status", "coefficient_standardized", "coefficient_raw_scale",
            "feature_mean", "feature_scale", "intercept_standardized",
            "intercept_raw_scale", "feature_names", "layer",
        }
        if required - set(archive.files):
            raise ValueError(
                f"Layer {layer:02d} coefficient shard lacks arrays {sorted(required - set(archive.files))}"
            )
        expected_matrix_shape = (expected_proteins, FEATURE_DIM)
        for name in (
            "coefficient_standardized", "coefficient_raw_scale", "feature_mean", "feature_scale",
        ):
            if archive[name].shape != expected_matrix_shape:
                raise ValueError(f"Layer {layer:02d} coefficient array {name} has wrong shape")
        if np.asarray(archive["protein_id"]).astype(str).tolist() != [
            row["protein_id"] for row in sorted(rows, key=lambda row: int(row["array_row"]))
        ]:
            raise ValueError(f"Layer {layer:02d} coefficient protein IDs differ from the index")
        ordered_rows = sorted(rows, key=lambda row: int(row["array_row"]))
        archive_status = np.asarray(archive["fit_status"]).astype(str).tolist()
        if archive_status != [row["fit_status"] for row in ordered_rows]:
            raise ValueError(f"Layer {layer:02d} coefficient fit statuses differ from the index")
        if int(np.asarray(archive["layer"]).item()) != layer:
            raise ValueError(f"Layer {layer:02d} coefficient shard carries the wrong layer")
        if np.asarray(archive["feature_names"]).astype(str).tolist() != list(FEATURE_NAMES):
            raise ValueError(f"Layer {layer:02d} coefficient feature order is noncanonical")
        fitted_rows = np.asarray(
            [row["fit_status"].startswith("fitted") for row in ordered_rows], dtype=bool,
        )
        for name in (
            "coefficient_standardized", "coefficient_raw_scale", "feature_mean", "feature_scale",
        ):
            if fitted_rows.any() and not np.all(np.isfinite(archive[name][fitted_rows])):
                raise ValueError(f"Layer {layer:02d} fitted coefficient array {name} is nonfinite")
        for name in ("intercept_standardized", "intercept_raw_scale"):
            if fitted_rows.any() and not np.all(np.isfinite(archive[name][fitted_rows])):
                raise ValueError(f"Layer {layer:02d} fitted coefficient array {name} is nonfinite")
        if fitted_rows.any() and np.any(archive["feature_scale"][fitted_rows] <= 0.0):
            raise ValueError(f"Layer {layer:02d} fitted feature scales must be positive")


def validate_layer_shard(
    root: Path,
    layer: int,
    pair_digest: str,
    expected_configuration_digest: str | None,
    expected_proteins: int = EXPECTED_PROTEINS,
    pair_rows: Sequence[Mapping[str, str]] | None = None,
) -> tuple[dict[str, list[dict[str, str]]], list[dict[str, str]], dict[str, object]]:
    shard = root / "layer_shards" / f"layer_{layer:02d}"
    manifest = read_json(shard / "shard_manifest.json")
    required_manifest = {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "master_split_digest": MASTER_MANIFEST_DIGEST,
        "pair_split_manifest_digest": pair_digest,
        "layer": layer,
        "expected_proteins": expected_proteins,
        "processed_proteins": expected_proteins,
    }
    for key, expected in required_manifest.items():
        if manifest.get(key) != expected:
            raise ValueError(
                f"Layer {layer:02d} manifest {key}={manifest.get(key)!r}; expected {expected!r}"
            )
    configuration_digest = str(manifest["configuration_digest"])
    if expected_configuration_digest is not None and configuration_digest != expected_configuration_digest:
        raise ValueError(f"Layer {layer:02d} configuration digest differs across shards")
    if semantic_digest(manifest["configuration"]) != configuration_digest:
        raise ValueError(f"Layer {layer:02d} manifest configuration digest is invalid")
    tables: dict[str, list[dict[str, str]]] = {}
    expected_counts = expected_layer_counts(expected_proteins)
    for filename in TABLES:
        path = shard / filename
        expected_hash = manifest["artifact_sha256"].get(filename)
        if expected_hash != sha256_file(path):
            raise ValueError(f"Layer {layer:02d} {filename} digest mismatch")
        rows = read_csv_exact(path, CSV_SCHEMAS[filename])
        _validate_layer_rows(filename, rows, layer, expected_counts[filename])
        validate_unique(filename, rows)
        tables[filename] = rows
    _validate_finite_success_metrics(tables["per_protein.csv"])
    if pair_rows is not None:
        split_by_id = {row["protein_id"]: row["protein_split"] for row in pair_rows}
        expected_ids = set(split_by_id)
        per_rows = tables["per_protein.csv"]
        if {row["protein_id"] for row in per_rows} != expected_ids:
            raise ValueError(f"Layer {layer:02d} per-protein IDs differ from pair metadata")
        if {row["evaluation_scope"] for row in per_rows} != set(EVALUATION_SCOPES):
            raise ValueError(f"Layer {layer:02d} per-protein evaluation scopes are noncanonical")
        if any(row["protein_split"] != split_by_id[row["protein_id"]] for row in per_rows):
            raise ValueError(f"Layer {layer:02d} per-protein split labels differ from metadata")
        expected_counts_by_id = {
            row["protein_id"]: {
                "train_pairs": int(row["n_train_pairs"]),
                "heldout_pairs": int(row["n_heldout_pairs"]),
                "all_pairs": int(row["n_protein_pairs"]),
            }
            for row in pair_rows
        }
        if any(
            int(row["n_expected_pairs"])
            != expected_counts_by_id[row["protein_id"]][row["evaluation_scope"]]
            for row in per_rows
        ):
            raise ValueError(f"Layer {layer:02d} expected pair counts differ from split metadata")
        for filename in ("model_status.csv", "geometry_metrics.csv"):
            if {row["protein_id"] for row in tables[filename]} != expected_ids:
                raise ValueError(f"Layer {layer:02d} {filename} protein IDs are incomplete")
        pooled_keys = {
            (row["protein_split"], row["evaluation_scope"])
            for row in tables["pooled_metrics.csv"]
        }
        if pooled_keys != {
            (protein_split, scope)
            for protein_split in ("development", "test")
            for scope in EVALUATION_SCOPES
        }:
            raise ValueError(f"Layer {layer:02d} pooled scopes are noncanonical")
        for filename in (
            "subgroup_metrics/by_sequence_separation.csv",
            "subgroup_metrics/by_amino_acid_pair.csv",
            "subgroup_metrics/by_spatial_distance.csv",
        ):
            if any(
                row["protein_split"] != "test" or row["evaluation_scope"] != "heldout_pairs"
                for row in tables[filename]
            ):
                raise ValueError(f"Layer {layer:02d} {filename} is not locked-test held-out")
        if any(
            row["evaluation_scope"] != "all_pairs"
            for row in tables["geometry_metrics.csv"]
        ):
            raise ValueError(f"Layer {layer:02d} geometry is not labeled all-pairs")
    coefficient_rows = read_csv_exact(
        shard / "coefficient_index.csv", COEFFICIENT_INDEX_COLUMNS,
    )
    if manifest["artifact_sha256"].get("coefficient_index.csv") != sha256_file(
        shard / "coefficient_index.csv"
    ):
        raise ValueError(f"Layer {layer:02d} coefficient index digest mismatch")
    _validate_layer_rows(
        "coefficient_index.csv", coefficient_rows, layer, expected_counts["coefficient_index.csv"],
    )
    coefficient_relative = str(manifest["coefficient_shard"])
    _validate_coefficient_index(
        root, layer, coefficient_rows, coefficient_relative, str(manifest["coefficient_sha256"]),
        expected_proteins,
    )
    validate_unique("model_artifacts/coefficient_index.csv", coefficient_rows)
    return tables, coefficient_rows, manifest


def _code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=ROOT_DIR, text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def aggregate_result_family(
    root: Path = INPUT_DIR,
    expected_layers: Sequence[int] = EXPECTED_LAYERS,
    expected_proteins: int = EXPECTED_PROTEINS,
    require_canonical_coverage: bool = True,
) -> dict[str, int]:
    layers = tuple(int(layer) for layer in expected_layers)
    if require_canonical_coverage and (
        layers != EXPECTED_LAYERS or expected_proteins != EXPECTED_PROTEINS
    ):
        raise ValueError("Canonical consolidation requires all layers 0-47")
    pair_rows = read_csv_exact(root / "metadata" / "pair_splits.csv", PAIR_SPLIT_COLUMNS)
    if len(pair_rows) != expected_proteins:
        raise ValueError(
            f"Pair-split manifest has {len(pair_rows)} rows; expected {expected_proteins}"
        )
    if len({row["protein_id"] for row in pair_rows}) != expected_proteins:
        raise ValueError("Pair-split manifest protein IDs are not unique")
    pair_digest = pair_split_manifest_digest(pair_rows)
    consolidated = {filename: [] for filename in TABLES}
    coefficient_rows: list[dict[str, str]] = []
    manifests = []
    configuration_digest = None
    expected_versions = None
    expected_code_revision = None
    for layer in layers:
        layer_tables, layer_coefficients, manifest = validate_layer_shard(
            root, layer, pair_digest, configuration_digest, expected_proteins, pair_rows,
        )
        configuration_digest = str(manifest["configuration_digest"])
        versions = manifest.get("versions")
        revision = manifest.get("code_revision")
        if expected_versions is None:
            expected_versions = versions
            expected_code_revision = revision
        elif versions != expected_versions or revision != expected_code_revision:
            raise ValueError(f"Layer {layer:02d} version or code-revision provenance differs")
        manifests.append(manifest)
        for filename, rows in layer_tables.items():
            consolidated[filename].extend(rows)
        coefficient_rows.extend(layer_coefficients)
    for filename, rows in consolidated.items():
        validate_unique(filename, rows)
        atomic_write_csv(root / filename, CSV_SCHEMAS[filename], rows)
    validate_unique("model_artifacts/coefficient_index.csv", coefficient_rows)
    atomic_write_csv(
        root / "model_artifacts" / "coefficient_index.csv",
        COEFFICIENT_INDEX_COLUMNS,
        coefficient_rows,
    )
    completed_layers = [
        int(manifest["layer"])
        for manifest in manifests
        if manifest["completion_status"] == "complete"
    ]
    run_status = "complete" if len(completed_layers) == len(layers) else "incomplete"
    configuration = dict(manifests[0]["configuration"])
    manifest_versions = manifests[0].get("versions", version_payload())
    run_manifest = build_run_manifest(
        analysis_name=ANALYSIS_NAME,
        model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM,
        requested_layers=layers,
        completed_layers=completed_layers,
        split_digest=MASTER_MANIFEST_DIGEST,
        configuration=configuration,
        versions=manifest_versions,
        code_revision=str(manifests[0].get("code_revision") or _code_revision()),
        run_status=run_status,
    )
    run_manifest["pair_split_manifest_path"] = "metadata/pair_splits.csv"
    run_manifest["pair_split_manifest_digest"] = pair_digest
    run_manifest["preparation_failures_path"] = "metadata/preparation_failures.csv"
    atomic_write_json(root / "run_manifest.json", run_manifest)
    counts = {filename: len(rows) for filename, rows in consolidated.items()}
    counts["model_artifacts/coefficient_index.csv"] = len(coefficient_rows)
    counts["metadata/pair_splits.csv"] = len(pair_rows)
    return counts


def main() -> None:
    counts = aggregate_result_family()
    print(f"Validated canonical schema version {OUTPUT_SCHEMA_VERSION}")
    print(f"Wrote consolidated outputs under {INPUT_DIR}")
    for filename, count in sorted(counts.items()):
        print(f"  {filename}: {count} rows")


if __name__ == "__main__":
    main()

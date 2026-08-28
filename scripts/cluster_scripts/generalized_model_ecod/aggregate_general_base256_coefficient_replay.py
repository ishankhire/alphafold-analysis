#!/usr/bin/env python3
"""Validate and consolidate all canonical shared-OLS coefficient-replay shards."""

from __future__ import annotations

import hashlib
import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, CLUSTER_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from general_base256_coefficient_replay_common import (  # noqa: E402
    ANALYSIS_NAME,
    EXCLUDED_COLUMNS,
    FEATURE_DIM,
    FINAL_FAMILY_NAME,
    LEGACY_COMPARISON_COLUMNS,
    MODEL_FAMILY,
    SHARD_CSV_SCHEMAS,
    SPLIT_ASSIGNMENT_COLUMNS,
    CoefficientBundle,
    HistoricalSplitBundle,
    code_digest,
    code_revision,
    expected_layer_counts,
    final_output_root,
    load_coefficient_bundle,
    load_historical_split_bundle,
    load_legacy_summary,
    predict_raw,
    read_csv_strict,
    source_output_root,
    split_assignment_rows,
    utc_now,
    version_payload,
)
from general_train_test_common import (  # noqa: E402
    build_dataset_protein_manifest,
    resolve_pair_block_path,
)
from regression_output_helpers import (  # noqa: E402
    OUTPUT_SCHEMA_VERSION,
    atomic_write_csv,
    atomic_write_json,
    build_run_manifest,
    sha256_file,
)
from regression_protocol import (  # noqa: E402
    AMINO_ACID_PAIR_LABELS,
    FEATURE_NAMES,
    LAYERS,
    MASTER_MANIFEST_DIGEST,
    HISTORICAL_PROTOCOL_VERSION as PROTOCOL_VERSION,
    SEQUENCE_BIN_LABELS,
    SPATIAL_DISTANCE_BIN_LABELS,
    extract_directional_base256,
    load_master_manifest,
    strict_lower_triangle_indices,
)


PRIMARY_KEYS = {
    "per_protein.csv": ("analysis_name", "layer", "protein_id", "evaluation_scope"),
    "pooled_metrics.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope"),
    "model_status.csv": ("analysis_name", "layer"),
    "runtime.csv": ("analysis_name", "layer"),
    "geometry_metrics.csv": ("analysis_name", "layer", "protein_id", "evaluation_scope"),
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
    "model_artifacts/coefficients.csv": ("analysis_name", "layer", "feature_index"),
    "metadata/legacy_metric_comparison.csv": (
        "analysis_name", "layer", "protein_split", "metric",
    ),
}


def _as_float(row: Mapping[str, object], key: str) -> float:
    value = str(row[key])
    return float("nan") if value.lower() == "nan" else float(value)


def validate_unique_rows(
    rows: Sequence[Mapping[str, object]], keys: Sequence[str], label: str,
) -> None:
    values = [tuple(row[key] for key in keys) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate primary key in {label}")


def _validate_pooled_reconstruction(
    per_rows: Sequence[Mapping[str, object]],
    pooled_rows: Sequence[Mapping[str, object]],
    layer: int,
) -> None:
    for split in ("development", "test"):
        rows = [row for row in per_rows if row["protein_split"] == split]
        pooled = next(row for row in pooled_rows if row["protein_split"] == split)
        n = sum(int(row["n_evaluated_pairs"]) for row in rows)
        if n <= 0 or n != int(pooled["n_evaluated_pairs"]) or n != int(pooled["n_expected_pairs"]):
            raise ValueError(f"Layer {layer} {split} pooled support mismatch")
        sum_y = sum(int(row["n_evaluated_pairs"]) * _as_float(row, "mean_true_distance") for row in rows)
        sum_y2 = sum(int(row["n_evaluated_pairs"]) * (
            _as_float(row, "target_variance") + _as_float(row, "mean_true_distance") ** 2
        ) for row in rows)
        sse = sum(int(row["n_evaluated_pairs"]) * _as_float(row, "rmse") ** 2 for row in rows)
        sae = sum(int(row["n_evaluated_pairs"]) * _as_float(row, "mae") for row in rows)
        error = sum(
            int(row["n_evaluated_pairs"]) * _as_float(row, "bias_pred_minus_true")
            for row in rows
        )
        sst = sum_y2 - sum_y * sum_y / n
        reconstructed = {
            "pooled_r2": float("nan") if sst <= 1e-12 else 1.0 - sse / sst,
            "pooled_rmse": math.sqrt(sse / n),
            "pooled_mae": sae / n,
            "pooled_bias_pred_minus_true": error / n,
        }
        for key, expected in reconstructed.items():
            observed = _as_float(pooled, key)
            if not np.isclose(observed, expected, rtol=1e-9, atol=1e-10, equal_nan=True):
                raise ValueError(f"Layer {layer} {split} {key} does not reconstruct")
        if pooled["conditional_baseline_type"] != "not_applicable":
            raise ValueError(f"Layer {layer} shared OLS unexpectedly has a conditional baseline")
        for key in pooled:
            if key.startswith("conditional_baseline_pooled_") and str(pooled[key]).lower() != "nan":
                raise ValueError(f"Layer {layer} conditional baseline metric must be NaN")


def _validate_coefficients(
    rows: Sequence[Mapping[str, object]], layer: int, expected_model,
) -> None:
    observed = [(int(row["feature_index"]), row["feature_name"]) for row in rows]
    expected = [(-1, "intercept"), *list(enumerate(FEATURE_NAMES))]
    if observed != expected:
        raise ValueError(f"Layer {layer} canonical coefficient feature mapping is malformed")
    expected_rows = list(expected_model.canonical_rows)
    for row, expected_row in zip(rows, expected_rows):
        for key in ("coefficient_standardized", "coefficient_raw_scale"):
            if _as_float(row, key) != float(expected_row[key]):
                raise ValueError(f"Layer {layer} coefficient value changed for feature {row['feature_index']}")


def validate_layer_shard(
    root: Path,
    layer: int,
    entries: Sequence[object],
    coefficients: CoefficientBundle,
    historical_split: HistoricalSplitBundle,
) -> tuple[dict[str, list[dict[str, str]]], dict[str, object]]:
    shard = root / "layer_shards" / f"layer_{layer:02d}"
    manifest_path = shard / "shard_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing shard manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_invariants = {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "layer": layer,
        "status": "complete",
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "protocol_version": PROTOCOL_VERSION,
        "master_manifest_digest": MASTER_MANIFEST_DIGEST,
        "historical_split_digest": historical_split.semantic_digest,
        "coefficient_source_sha256": coefficients.sha256,
        "historical_split_source_sha256": historical_split.sha256,
        "code_digest": code_digest(),
    }
    for key, expected in expected_invariants.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Layer {layer} shard invariant mismatch for {key}")
    expected_ids = {entry.protein_id for entry in entries if entry.eligible}
    entry_by_id = {entry.protein_id: entry for entry in entries if entry.eligible}
    split_by_id = {entry.protein_id: entry.protein_split for entry in entries if entry.eligible}
    tables: dict[str, list[dict[str, str]]] = {}
    expected_counts = expected_layer_counts(len(expected_ids))
    if set(manifest.get("files", {})) != set(SHARD_CSV_SCHEMAS):
        raise ValueError(f"Layer {layer} shard file inventory mismatch")
    for filename, columns in SHARD_CSV_SCHEMAS.items():
        path = shard / filename
        if not path.is_file() or manifest["files"].get(filename) != sha256_file(path):
            raise ValueError(f"Layer {layer} checksum mismatch: {filename}")
        rows = read_csv_strict(path, columns)
        if len(rows) != expected_counts[filename]:
            raise ValueError(f"Layer {layer} row-count mismatch for {filename}")
        if {int(row["layer"]) for row in rows} != {layer}:
            raise ValueError(f"Layer identity mismatch in {filename}")
        if any(row["analysis_name"] != ANALYSIS_NAME for row in rows):
            raise ValueError(f"Analysis identity mismatch in {filename}")
        if rows and "model_family" in rows[0] and any(
            row["model_family"] != MODEL_FAMILY for row in rows
        ):
            raise ValueError(f"Model-family identity mismatch in {filename}")
        validate_unique_rows(rows, PRIMARY_KEYS[filename], filename)
        tables[filename] = rows

    per_rows = tables["per_protein.csv"]
    if {row["protein_id"] for row in per_rows} != expected_ids:
        raise ValueError(f"Layer {layer} per-protein membership mismatch")
    if any(row["protein_split"] != split_by_id[row["protein_id"]] for row in per_rows):
        raise ValueError(f"Layer {layer} per-protein split mismatch")
    if any(
        int(row["n_residues"]) != entry_by_id[row["protein_id"]].n_residues
        or int(row["n_protein_pairs"]) != entry_by_id[row["protein_id"]].n_pairs
        for row in per_rows
    ):
        raise ValueError(f"Layer {layer} per-protein support differs from the frozen manifest")
    if any(
        row["prediction_status"] != "complete"
        or row["evaluation_scope"] != "all_pairs"
        or float(row["pair_coverage"]) != 1.0
        or int(row["n_expected_pairs"]) != int(row["n_evaluated_pairs"])
        for row in per_rows
    ):
        raise ValueError(f"Layer {layer} has incomplete per-protein predictions")
    finite_fields = (
        "target_variance", "mae", "rmse", "bias_pred_minus_true",
        "mean_true_distance", "mean_predicted_distance",
    )
    if any(not all(np.isfinite(_as_float(row, key)) for key in finite_fields) for row in per_rows):
        raise ValueError(f"Layer {layer} has nonfinite complete per-protein metrics")

    geometry_rows = tables["geometry_metrics.csv"]
    if {row["protein_id"] for row in geometry_rows} != expected_ids or any(
        row["geometry_status"] != "complete"
        or row["evaluation_scope"] != "all_pairs"
        or float(row["pair_coverage"]) != 1.0
        or row["protein_split"] != split_by_id[row["protein_id"]]
        for row in geometry_rows
    ):
        raise ValueError(f"Layer {layer} has incomplete canonical geometry")

    pooled_rows = tables["pooled_metrics.csv"]
    if {row["protein_split"] for row in pooled_rows} != {"development", "test"}:
        raise ValueError(f"Layer {layer} pooled split coverage mismatch")
    for row in pooled_rows:
        expected_split_entries = [
            entry for entry in entries if entry.eligible and entry.protein_split == row["protein_split"]
        ]
        if (
            int(row["n_proteins"]) != len(expected_split_entries)
            or int(row["n_expected_pairs"]) != sum(entry.n_pairs for entry in expected_split_entries)
        ):
            raise ValueError(f"Layer {layer} pooled support differs from the frozen manifest")
        if float(row["pair_coverage"]) != 1.0:
            raise ValueError(f"Layer {layer} pooled coverage is incomplete")
        for key in (
            "pooled_r2", "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true",
            "global_baseline_pooled_r2", "global_baseline_pooled_mae",
            "global_baseline_pooled_rmse", "global_baseline_pooled_bias_pred_minus_true",
        ):
            if not np.isfinite(_as_float(row, key)):
                raise ValueError(f"Layer {layer} pooled field {key} is nonfinite")
    _validate_pooled_reconstruction(per_rows, pooled_rows, layer)

    subgroup_expectations = {
        "subgroup_metrics/by_sequence_separation.csv": set(SEQUENCE_BIN_LABELS),
        "subgroup_metrics/by_amino_acid_pair.csv": set(AMINO_ACID_PAIR_LABELS),
        "subgroup_metrics/by_spatial_distance.csv": set(SPATIAL_DISTANCE_BIN_LABELS),
    }
    for filename, labels in subgroup_expectations.items():
        rows = tables[filename]
        if {row["category_label"] for row in rows} != labels or any(
            row["protein_split"] != "test" or row["evaluation_scope"] != "all_pairs"
            for row in rows
        ):
            raise ValueError(f"Layer {layer} subgroup identity mismatch in {filename}")

    status = tables["model_status.csv"][0]
    if status["fit_status"] != "historical_fit_replayed":
        raise ValueError(f"Layer {layer} model status does not identify historical replay")
    if "design_diagnostics_unavailable" not in status["support_flags"]:
        raise ValueError(f"Layer {layer} model status omits unavailable diagnostics")
    for field in (
        "design_rank", "largest_singular_value", "smallest_singular_value", "condition_number",
    ):
        if str(status[field]).lower() != "nan":
            raise ValueError(f"Layer {layer} historical diagnostic {field} must be NaN")
    expected_development = [entry for entry in entries if entry.eligible and entry.protein_split == "development"]
    if int(status["n_training_proteins"]) != len(expected_development) or int(status["n_training_pairs"]) != sum(entry.n_pairs for entry in expected_development):
        raise ValueError(f"Layer {layer} historical training support mismatch")

    runtime = tables["runtime.csv"][0]
    if (
        runtime["completion_status"] != "complete"
        or int(runtime["n_evaluated_proteins"]) != len(expected_ids)
        or int(runtime["n_evaluated_pairs"]) != sum(entry.n_pairs for entry in entries if entry.eligible)
    ):
        raise ValueError(f"Layer {layer} runtime completion mismatch")
    comparison = tables["metadata/legacy_metric_comparison.csv"]
    if any(
        row["comparison_status"] != "pass"
        or _as_float(row, "absolute_difference") > _as_float(row, "tolerance")
        for row in comparison
    ):
        raise ValueError(f"Layer {layer} failed legacy metric comparison")
    _validate_coefficients(
        tables["model_artifacts/coefficients.csv"], layer, coefficients.by_layer[layer],
    )
    return tables, manifest


def verify_layer_replay(
    layer: int,
    manifest: Mapping[str, object],
    coefficients: CoefficientBundle,
    entries: Sequence[object],
) -> None:
    replay = manifest.get("replay")
    if not isinstance(replay, dict):
        raise ValueError(f"Layer {layer} replay metadata is missing")
    protein_id = str(replay["protein_id"])
    entry_by_id = {entry.protein_id: entry for entry in entries if entry.eligible}
    entry = entry_by_id.get(protein_id)
    if entry is None or entry.protein_split != "development":
        raise ValueError(f"Layer {layer} replay protein is not in development")
    dataset = {str(row["protein_id"]): row for row in build_dataset_protein_manifest()}
    source = dataset[protein_id]
    pair_path = resolve_pair_block_path(Path(source["protein_dir"]), protein_id, layer)
    pair_i, pair_j = strict_lower_triangle_indices(entry.n_residues)
    indices = np.asarray(replay["pair_indices"], dtype=np.int64)
    if (
        indices.ndim != 1 or indices.size == 0 or len(set(indices.tolist())) != indices.size
        or np.any(indices < 0) or np.any(indices >= entry.n_pairs)
    ):
        raise ValueError(f"Layer {layer} replay indices are invalid")
    pair_block = np.load(pair_path)
    try:
        features = extract_directional_base256(pair_block, pair_i[indices], pair_j[indices])
    finally:
        del pair_block
    if hashlib.sha256(features.tobytes()).hexdigest() != replay["feature_sha256"]:
        raise ValueError(f"Layer {layer} replay feature checksum mismatch")
    observed = predict_raw(features, coefficients.by_layer[layer])
    if not np.allclose(
        observed,
        np.asarray(replay["predictions"], dtype=np.float64),
        rtol=0.0,
        atol=float(replay["absolute_tolerance"]),
    ):
        raise ValueError(f"Layer {layer} raw coefficient replay mismatch")


def _publish_staged_files(staging: Path, root: Path) -> None:
    for path in sorted(staging.rglob("*")):
        if path.is_dir():
            continue
        destination = root / path.relative_to(staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)


def aggregate_result_family(
    root: Path,
    *,
    expected_layers: Sequence[int],
    entries: Sequence[object],
    coefficients: CoefficientBundle,
    historical_split: HistoricalSplitBundle,
    verify_replay: bool = True,
) -> dict[str, int]:
    requested = tuple(int(layer) for layer in expected_layers)
    if len(requested) != len(set(requested)):
        raise ValueError("Requested replay layers contain duplicates")
    layers = tuple(sorted(requested))
    if not layers:
        raise ValueError("At least one replay layer is required")
    if any(layer not in LAYERS for layer in layers):
        raise ValueError("Requested replay layers must be in 0..47")
    tables = {filename: [] for filename in SHARD_CSV_SCHEMAS}
    manifests = []
    for layer in layers:
        layer_tables, manifest = validate_layer_shard(
            root, layer, entries, coefficients, historical_split,
        )
        for filename, rows in layer_tables.items():
            tables[filename].extend(rows)
        manifests.append(manifest)
    n_proteins = sum(entry.eligible for entry in entries)
    for filename, rows in tables.items():
        expected = expected_layer_counts(n_proteins)[filename] * len(layers)
        if len(rows) != expected:
            raise ValueError(f"Consolidated row-count mismatch for {filename}: {len(rows)} != {expected}")
        validate_unique_rows(rows, PRIMARY_KEYS[filename], f"consolidated {filename}")
    for invariant in (
        "code_digest", "coefficient_source_sha256", "historical_split_source_sha256",
        "historical_split_digest", "master_manifest_digest",
    ):
        if len({manifest[invariant] for manifest in manifests}) != 1:
            raise ValueError(f"Layer manifests disagree on {invariant}")
    if verify_replay:
        for layer, manifest in zip(layers, manifests):
            verify_layer_replay(layer, manifest, coefficients, entries)

    split_rows, split_digest = split_assignment_rows(entries)
    exclusions = [{
        "protein_id": entry.protein_id,
        "source": entry.source,
        "eligibility_status": entry.eligibility_status,
        "stage": "frozen_manifest",
        "layer": "",
        "reason": entry.exclusion_reason,
    } for entry in entries if not entry.eligible]
    comparison_rows = tables["metadata/legacy_metric_comparison.csv"]
    max_differences = {
        metric: max(
            _as_float(row, "absolute_difference")
            for row in comparison_rows if row["metric"] == metric
        )
        for metric in sorted({row["metric"] for row in comparison_rows})
    }
    configuration = {
        "source_model": "historical_shared_ols_general_base256_combined",
        "fit_action": "none_coefficient_replay_only",
        "feature_mode": "directional_base256",
        "prediction_formula": "raw_intercept + directional_base256 @ raw_coefficients",
        "coefficient_serialization": "historical_csv_six_decimal",
        "development_proteins": sum(entry.eligible and entry.protein_split == "development" for entry in entries),
        "test_proteins": sum(entry.eligible and entry.protein_split == "test" for entry in entries),
        "legacy_metric_tolerances": {"r2": 1e-4, "mae_angstrom": 1e-3, "rmse_angstrom": 1e-3},
        "methodology_compliance": "historical_partial_output_schema_complete",
        "known_methodology_deviations": [
            "P2: original fit did not record the later fit/validation subdivision",
            "P5: original scaler second moments were formed from float32 cached features",
            "P8: original fit did not save solver-exposed rank or singular values",
        ],
    }
    staging = Path(tempfile.mkdtemp(prefix=f".{FINAL_FAMILY_NAME}.aggregate.", dir=root.parent))
    try:
        for filename, rows in tables.items():
            atomic_write_csv(staging / filename, SHARD_CSV_SCHEMAS[filename], rows)
        atomic_write_csv(
            staging / "metadata" / "split_assignments.csv",
            SPLIT_ASSIGNMENT_COLUMNS,
            split_rows,
        )
        atomic_write_csv(
            staging / "metadata" / "excluded_proteins.csv",
            EXCLUDED_COLUMNS,
            exclusions,
        )
        provenance = {
            "analysis_name": ANALYSIS_NAME,
            "model_family": MODEL_FAMILY,
            "source_directory": str(source_output_root()),
            "coefficient_source_path": str(coefficients.path),
            "coefficient_source_sha256": coefficients.sha256,
            "historical_split_source_path": str(historical_split.path),
            "historical_split_source_sha256": historical_split.sha256,
            "historical_split_digest": historical_split.semantic_digest,
            "canonical_master_manifest_digest": MASTER_MANIFEST_DIGEST,
            "canonical_split_assignment_digest": split_digest,
            "coefficient_precision": "six digits after the decimal point in the historical CSV",
            "prediction_formula": configuration["prediction_formula"],
            "fit_performed": False,
            "known_methodology_deviations": configuration["known_methodology_deviations"],
            "legacy_metric_max_absolute_differences": max_differences,
            "created_at": utc_now(),
        }
        atomic_write_json(staging / "metadata" / "replay_provenance.json", provenance)
        manifest = build_run_manifest(
            analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            requested_layers=layers,
            completed_layers=layers,
            split_digest=MASTER_MANIFEST_DIGEST,
            configuration=configuration,
            versions=version_payload(),
            code_revision=code_revision(),
            run_status="complete",
        )
        manifest["master_manifest_digest"] = manifest.pop("master_split_digest")
        manifest.update({
            "split_assignment_digest": split_digest,
            "historical_split_digest": historical_split.semantic_digest,
            "coefficient_source_sha256": coefficients.sha256,
            "historical_split_source_sha256": historical_split.sha256,
            "code_digest": code_digest(),
            "output_contract_status": "complete_with_unavailable_historical_fit_diagnostics",
            "methodology_compliance": configuration["methodology_compliance"],
            "known_methodology_deviations": configuration["known_methodology_deviations"],
            "legacy_metric_max_absolute_differences": max_differences,
            "input_root_identifiers": {
                "csv_root": str(root.parent),
                "historical_result_family": str(source_output_root()),
                "dataset_sources": sorted({entry.source for entry in entries}),
            },
            "finalized_at": utc_now(),
        })
        artifact_paths = sorted(path for path in staging.rglob("*") if path.is_file())
        manifest["artifact_sha256"] = {
            str(path.relative_to(staging)): sha256_file(path) for path in artifact_paths
        }
        atomic_write_json(staging / "run_manifest.json", manifest)
        _publish_staged_files(staging, root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return {filename: len(rows) for filename, rows in tables.items()}


def run(root: Path | None = None) -> Path:
    destination = final_output_root() if root is None else Path(root)
    entries = load_master_manifest()
    coefficients = load_coefficient_bundle()
    historical_split = load_historical_split_bundle(entries=entries)
    load_legacy_summary()  # Revalidate the source comparison table before publication.
    counts = aggregate_result_family(
        destination,
        expected_layers=LAYERS,
        entries=entries,
        coefficients=coefficients,
        historical_split=historical_split,
        verify_replay=True,
    )
    print(f"Published canonical shared-OLS replay to {destination}")
    print(json.dumps(counts, indent=2, sort_keys=True))
    return destination


if __name__ == "__main__":
    run()

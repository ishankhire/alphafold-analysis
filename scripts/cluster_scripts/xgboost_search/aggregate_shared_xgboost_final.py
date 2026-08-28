#!/usr/bin/env python3
"""Validate and consolidate all 48 canonical shared-XGBoost layer shards."""

from __future__ import annotations

import json
import math
import os
import shutil
import sys
import tempfile
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
GENERALIZED_DIR = CLUSTER_DIR / "generalized_model_ecod"
for _path in (SCRIPT_DIR, CLUSTER_DIR, GENERALIZED_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from general_train_test_common import build_dataset_protein_manifest, resolve_pair_block_path
from regression_output_helpers import atomic_write_csv, build_run_manifest, sha256_file
from regression_protocol import (
    AMINO_ACID_PAIR_LABELS,
    FEATURE_DIM,
    FEATURE_NAMES,
    MASTER_MANIFEST_DIGEST,
    HISTORICAL_PROTOCOL_VERSION as PROTOCOL_VERSION,
    SEQUENCE_BIN_LABELS,
    SPATIAL_DISTANCE_BIN_LABELS,
    extract_directional_base256,
    load_master_manifest,
)
from shared_xgboost_final_common import (
    ANALYSIS_NAME,
    CONFIGURATION_ID,
    EXCLUDED_COLUMNS,
    MODEL_FAMILY,
    OUTPUT_SCHEMA_VERSION,
    RANDOM_SEED,
    ROW_CAP,
    BOOSTING_ROUNDS,
    SHARD_CSV_SCHEMAS,
    SPLIT_ASSIGNMENT_COLUMNS,
    TRAINING_SAMPLE_COLUMNS,
    atomic_write_json,
    build_final_estimator,
    build_training_sample_manifest,
    code_revision,
    expected_layer_counts,
    final_output_root,
    load_final_configuration,
    read_csv_strict,
    split_assignment_rows,
    utc_now,
    validate_unique_rows,
    version_payload,
)


def _as_float(row, key):
    value = str(row[key])
    return float("nan") if value == "NaN" else float(value)


def _validate_pooled_reconstruction(per_rows, pooled_rows, layer):
    for split in ("development", "test"):
        rows = [row for row in per_rows if row["protein_split"] == split]
        pooled = next(row for row in pooled_rows if row["protein_split"] == split)
        n = sum(int(row["n_evaluated_pairs"]) for row in rows)
        if n != int(pooled["n_evaluated_pairs"]) or n != int(pooled["n_expected_pairs"]):
            raise ValueError(f"Layer {layer} {split} pooled support mismatch")
        sum_y = sum(int(row["n_evaluated_pairs"]) * _as_float(row, "mean_true_distance") for row in rows)
        sum_y2 = sum(int(row["n_evaluated_pairs"]) * (
            _as_float(row, "target_variance") + _as_float(row, "mean_true_distance") ** 2
        ) for row in rows)
        sse = sum(int(row["n_evaluated_pairs"]) * _as_float(row, "rmse") ** 2 for row in rows)
        sae = sum(int(row["n_evaluated_pairs"]) * _as_float(row, "mae") for row in rows)
        error = sum(int(row["n_evaluated_pairs"]) * _as_float(row, "bias_pred_minus_true") for row in rows)
        sst = sum_y2 - sum_y * sum_y / n
        reconstructed = {
            "pooled_r2": 1.0 - sse / sst,
            "pooled_rmse": math.sqrt(sse / n),
            "pooled_mae": sae / n,
            "pooled_bias_pred_minus_true": error / n,
        }
        for key, expected in reconstructed.items():
            if not np.isclose(_as_float(pooled, key), expected, rtol=1e-9, atol=1e-10):
                raise ValueError(f"Layer {layer} {split} {key} does not reconstruct")
        if pooled["conditional_baseline_type"] != "not_applicable":
            raise ValueError(f"Layer {layer} unexpectedly has a conditional baseline")
        for key in pooled:
            if key.startswith("conditional_baseline_pooled_") and str(pooled[key]) != "NaN":
                raise ValueError(f"Layer {layer} conditional baseline metric must be NaN")


def validate_layer_shard(root, layer, config, split_digest, sample_digest):
    shard = root / "layer_shards" / f"layer_{layer:02d}"
    manifest_path = shard / "shard_manifest.json"
    if not manifest_path.is_file():
        raise ValueError(f"Missing shard manifest: {manifest_path}")
    manifest = json.loads(manifest_path.read_text())
    expected_invariants = {
        "analysis_name": ANALYSIS_NAME,
        "layer": layer,
        "status": "complete",
        "configuration_digest": config["configuration_digest"],
        "master_manifest_digest": MASTER_MANIFEST_DIGEST,
        "split_assignment_digest": split_digest,
        "sample_policy_digest": sample_digest,
        "protocol_version": PROTOCOL_VERSION,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
    }
    for key, expected in expected_invariants.items():
        if manifest.get(key) != expected:
            raise ValueError(f"Layer {layer} shard invariant mismatch for {key}")
    tables = {}
    expected_counts = expected_layer_counts()
    for filename, columns in SHARD_CSV_SCHEMAS.items():
        path = shard / filename
        if manifest.get("files", {}).get(filename) != sha256_file(path):
            raise ValueError(f"Layer {layer} checksum mismatch: {filename}")
        rows = read_csv_strict(path, columns)
        if len(rows) != expected_counts[filename]:
            raise ValueError(f"Layer {layer} row-count mismatch for {filename}")
        if {int(row["layer"]) for row in rows} != {layer}:
            raise ValueError(f"Layer identity mismatch in {filename}")
        tables[filename] = rows
    booster_name = f"model_artifacts/booster_layer_{layer:02d}.json"
    booster_path = shard / booster_name
    if manifest.get("files", {}).get(booster_name) != sha256_file(booster_path):
        raise ValueError(f"Layer {layer} booster checksum mismatch")
    if set(manifest.get("files", {})) != {*SHARD_CSV_SCHEMAS, booster_name}:
        raise ValueError(f"Layer {layer} shard file inventory mismatch")
    per_rows = tables["per_protein.csv"]
    eligible = [entry for entry in load_master_manifest() if entry.eligible]
    expected_ids = {entry.protein_id for entry in eligible}
    expected_split = {entry.protein_id: entry.protein_split for entry in eligible}
    if {row["protein_id"] for row in per_rows} != expected_ids:
        raise ValueError(f"Layer {layer} protein membership mismatch")
    if any(row["protein_split"] != expected_split[row["protein_id"]] for row in per_rows):
        raise ValueError(f"Layer {layer} protein split mismatch")
    if sum(row["protein_split"] == "development" for row in per_rows) != 441:
        raise ValueError(f"Layer {layer} development/test support mismatch")
    if any(row["prediction_status"] != "complete" or float(row["pair_coverage"]) != 1.0 for row in per_rows):
        raise ValueError(f"Layer {layer} has incomplete per-protein predictions")
    core_fields = (
        "target_variance", "mae", "rmse", "bias_pred_minus_true",
        "mean_true_distance", "mean_predicted_distance",
    )
    if any(not all(np.isfinite(_as_float(row, field)) for field in core_fields) for row in per_rows):
        raise ValueError(f"Layer {layer} has nonfinite core per-protein metrics")
    if any(row["geometry_status"] != "complete" or float(row["pair_coverage"]) != 1.0 for row in tables["geometry_metrics.csv"]):
        raise ValueError(f"Layer {layer} has incomplete geometry")
    status = tables["model_status.csv"][0]
    if (status["selected_configuration_id"], int(float(status["training_row_cap"])),
        int(float(status["final_boosting_rounds"])), int(float(status["random_seed"]))) != (
        CONFIGURATION_ID, ROW_CAP, BOOSTING_ROUNDS, RANDOM_SEED,
    ):
        raise ValueError(f"Layer {layer} final-model status mismatch")
    expected_fitting_rows = sum(min(ROW_CAP, entry.n_pairs) for entry in eligible if entry.protein_split == "development")
    if int(status["n_training_proteins"]) != 441 or int(status["n_training_pairs"]) != expected_fitting_rows:
        raise ValueError(f"Layer {layer} fitting support mismatch")
    runtime = tables["runtime.csv"][0]
    if runtime["completion_status"] != "complete" or int(runtime["n_evaluated_proteins"]) != 552:
        raise ValueError(f"Layer {layer} runtime completion mismatch")
    pooled_rows = tables["pooled_metrics.csv"]
    if {row["protein_split"] for row in pooled_rows} != {"development", "test"}:
        raise ValueError(f"Layer {layer} pooled split coverage mismatch")
    for row in pooled_rows:
        if float(row["pair_coverage"]) != 1.0:
            raise ValueError(f"Layer {layer} pooled coverage is incomplete")
        for key in ("pooled_r2", "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true"):
            if not np.isfinite(_as_float(row, key)):
                raise ValueError(f"Layer {layer} pooled metric {key} is nonfinite")
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
    importance = tables["model_artifacts/feature_importance.csv"]
    if [(int(row["feature_index"]), row["feature_name"]) for row in importance] != list(enumerate(FEATURE_NAMES)):
        raise ValueError(f"Layer {layer} feature-importance mapping mismatch")
    _validate_pooled_reconstruction(per_rows, tables["pooled_metrics.csv"], layer)
    unique_keys = {
        "per_protein.csv": ("analysis_name", "layer", "protein_id", "evaluation_scope"),
        "pooled_metrics.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope"),
        "model_status.csv": ("analysis_name", "layer"),
        "runtime.csv": ("analysis_name", "layer"),
        "geometry_metrics.csv": ("analysis_name", "layer", "protein_id", "evaluation_scope"),
        "geometry_summary.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope", "geometry_metric"),
        "subgroup_metrics/by_sequence_separation.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope", "category_label"),
        "subgroup_metrics/by_amino_acid_pair.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope", "category_label"),
        "subgroup_metrics/by_spatial_distance.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope", "category_label"),
        "model_artifacts/feature_importance.csv": ("analysis_name", "layer", "feature_index"),
    }
    for filename, keys in unique_keys.items():
        validate_unique_rows(tables[filename], keys, filename)
    return tables, manifest, booster_path
def _verify_booster_replay(layer, manifest, booster_path, config, dataset_by_id):
    replay = manifest["replay"]
    protein_id = replay["protein_id"]
    roles = {entry.protein_id: entry.master_role for entry in load_master_manifest() if entry.eligible}
    if roles.get(protein_id) not in {"fit", "validation"}:
        raise ValueError(f"Layer {layer} replay batch is not a development protein")
    entry = dataset_by_id[protein_id]
    pair_path = resolve_pair_block_path(Path(entry["protein_dir"]), protein_id, layer)
    pair_block = np.load(pair_path)
    n_residues = pair_block.shape[0]
    pair_i, pair_j = np.tril_indices(n_residues, k=-1)
    indices = np.asarray(replay["pair_indices"], dtype=np.int64)
    features = extract_directional_base256(pair_block, pair_i[indices], pair_j[indices])
    if __import__("hashlib").sha256(features.tobytes()).hexdigest() != replay["feature_sha256"]:
        raise ValueError(f"Layer {layer} replay feature checksum mismatch")
    model = build_final_estimator(config, 1)
    model.load_model(str(booster_path))
    observed = np.asarray(model.predict(features), dtype=np.float64)
    if not np.allclose(observed, np.asarray(replay["predictions"], dtype=np.float64), rtol=0.0, atol=float(replay["absolute_tolerance"])):
        raise ValueError(f"Layer {layer} booster replay mismatch")
    return json.loads(model.get_booster().save_config())


def _publish_staged_files(staging, root):
    for path in sorted(staging.rglob("*")):
        if path.is_dir():
            continue
        relative = path.relative_to(staging)
        destination = root / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)


def run(root: Path | None = None) -> Path:
    root = final_output_root() if root is None else Path(root)
    config = load_final_configuration()
    split_rows, split_digest = split_assignment_rows()
    if split_digest != config["split_assignment_digest"]:
        raise ValueError("Frozen split-assignment digest does not match the canonical manifest")
    sample_rows, sample_digest, _ = build_training_sample_manifest()
    tables = {filename: [] for filename in SHARD_CSV_SCHEMAS}
    boosters = []
    manifests = []
    for layer in range(48):
        layer_tables, manifest, booster_path = validate_layer_shard(
            root, layer, config, split_digest, sample_digest,
        )
        for filename, rows in layer_tables.items():
            tables[filename].extend(rows)
        boosters.append((layer, booster_path))
        manifests.append(manifest)
    for filename, rows in tables.items():
        expected = expected_layer_counts()[filename] * 48
        if len(rows) != expected:
            raise ValueError(f"Consolidated row-count mismatch for {filename}: {len(rows)} != {expected}")
    consolidated_keys = {
        "per_protein.csv": ("analysis_name", "layer", "protein_id", "evaluation_scope"),
        "pooled_metrics.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope"),
        "model_status.csv": ("analysis_name", "layer"),
        "runtime.csv": ("analysis_name", "layer"),
        "geometry_metrics.csv": ("analysis_name", "layer", "protein_id", "evaluation_scope"),
        "geometry_summary.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope", "geometry_metric"),
        "subgroup_metrics/by_sequence_separation.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope", "category_label"),
        "subgroup_metrics/by_amino_acid_pair.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope", "category_label"),
        "subgroup_metrics/by_spatial_distance.csv": ("analysis_name", "layer", "protein_split", "evaluation_scope", "category_label"),
        "model_artifacts/feature_importance.csv": ("analysis_name", "layer", "feature_index"),
    }
    for filename, keys in consolidated_keys.items():
        validate_unique_rows(tables[filename], keys, f"consolidated {filename}")
    if len({manifest["configuration_digest"] for manifest in manifests}) != 1:
        raise ValueError("Layer configuration digests differ")
    dataset_by_id = {str(row["protein_id"]): row for row in build_dataset_protein_manifest()}
    booster_configs = [
        _verify_booster_replay(layer, manifest, booster, config, dataset_by_id)
        for (layer, booster), manifest in zip(boosters, manifests)
    ]

    staging = Path(tempfile.mkdtemp(prefix=f".{ANALYSIS_NAME}.aggregate.", dir=root.parent))
    try:
        for filename, rows in tables.items():
            if filename == "model_artifacts/feature_importance.csv":
                output_name = filename
            else:
                output_name = filename
            atomic_write_csv(staging / output_name, SHARD_CSV_SCHEMAS[filename], rows)
        for layer, booster in boosters:
            destination = staging / "model_artifacts" / f"booster_layer_{layer:02d}.json"
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(booster, destination)
        atomic_write_csv(staging / "metadata" / "split_assignments.csv", SPLIT_ASSIGNMENT_COLUMNS, split_rows)
        atomic_write_csv(staging / "metadata" / "training_sample_manifest.csv", TRAINING_SAMPLE_COLUMNS, sample_rows)
        exclusions = [{
            "protein_id": entry.protein_id, "source": entry.source,
            "eligibility_status": entry.eligibility_status, "stage": "frozen_manifest",
            "layer": "", "reason": entry.exclusion_reason,
        } for entry in load_master_manifest() if not entry.eligible]
        atomic_write_csv(staging / "metadata" / "excluded_proteins.csv", EXCLUDED_COLUMNS, exclusions)
        atomic_write_json(staging / "metadata" / "selection_provenance.json", config)
        manifest = build_run_manifest(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, requested_layers=range(48), completed_layers=range(48),
            split_digest=MASTER_MANIFEST_DIGEST,
            configuration={key: value for key, value in config.items() if key != "configuration_digest"},
            versions=version_payload(), code_revision=code_revision(), run_status="complete",
        )
        manifest["master_manifest_digest"] = manifest.pop("master_split_digest")
        manifest.update({
            "split_assignment_digest": split_digest,
            "sample_policy_digest": sample_digest,
            "selection_provenance": config,
            "xgboost_booster_config": booster_configs[0],
            "input_root_identifiers": {
                "csv_root": str(root.parent),
                "dataset_sources": sorted({entry.source for entry in load_master_manifest()}),
                "pair_block_roots": sorted({
                    str(Path(row["protein_dir"]).parent) for row in dataset_by_id.values()
                }),
                "pdb_roots": sorted({
                    str(Path(row["pdb_root"])) for row in dataset_by_id.values()
                }),
            },
            "target_definition": config["target"],
            "precision_definition": config["precision"],
            "finalized_at": utc_now(),
        })
        atomic_write_json(staging / "run_manifest.json", manifest)
        _publish_staged_files(staging, root)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    return root


if __name__ == "__main__":
    run()

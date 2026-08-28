#!/usr/bin/env python3
"""Strictly validate and consolidate fivefold sequence-gated OLS shards."""

from __future__ import annotations

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
for _path in (CLUSTER_DIR, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cross_validation_common import (  # noqa: E402
    FOLD_METRICS_FILENAME,
    FOLD_SUMMARY_FILENAME,
    GEOMETRY_METRICS_FILENAME,
    GEOMETRY_SUMMARY_FILENAME,
    MODEL_STABILITY_FILENAME,
    MODEL_STATUS_FILENAME,
    OOFComparisonAccumulator,
    OOFSubgroupAccumulator,
    PER_PROTEIN_FILENAME,
    POOLED_METRICS_FILENAME,
    REPLAY_INDEX_FILENAME,
    RUNTIME_FILENAME,
    STABLE_FIT_METRICS_FILENAME,
    SUBGROUP_SEQUENCE_FILENAME,
    SUBGROUP_SPATIAL_FILENAME,
    TRAINING_METRICS_FILENAME,
    build_fold_metrics_row,
    build_fold_summary_rows,
    build_model_stability_rows,
    build_pooled_metrics_row,
    build_replay_index_row,
    build_run_manifest,
    build_stable_fit_metrics_row,
    fold_layer_task_id,
    load_cv_folds,
    map_fold_layer_task,
    summarize_geometry_rows,
    validate_replay_index,
    validate_resume_checkpoint,
    write_v2_csv,
)
from regression_output_helpers import atomic_write_json  # noqa: E402
from regression_protocol import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_NAMES,
    LAYERS,
    N_OUTER_FOLDS,
    SEQUENCE_BIN_LABELS,
    SPATIAL_DISTANCE_BIN_LABELS,
)
from sequence_gated_ols_base256_common import (  # noqa: E402
    ANALYSIS_NAME,
    CONDITIONAL_BASELINE_TYPE,
    CONFIGURATION,
    CONFIGURATION_DIGEST,
    EVALUATION_DESIGN,
    MIN_TRAIN_PROTEINS,
    MIN_TRAIN_ROWS,
    MODEL_FAMILY,
    OOF_PREDICTION_SCOPE,
    TASK_CHECKPOINT_FILENAME,
    TASK_PAYLOAD_PATH,
    TASK_PAYLOAD_VERSION,
    TASK_TABLES,
    code_revision,
    expected_task_artifacts,
    family_output_root,
    gate_slug,
    load_coefficient_artifact,
    log,
    read_csv_strict,
    task_model_path,
    task_shard_path,
    version_payload,
)


def _float(row: Mapping[str, object], key: str) -> float:
    return float(row[key])


def _unique(rows: Sequence[Mapping[str, object]], keys: Sequence[str], label: str) -> None:
    values = [tuple(row[key] for key in keys) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate primary key in {label}")


def _publish_staging(staging: Path, root: Path) -> None:
    manifest = staging / "run_manifest.json"
    for path in sorted(staging.rglob("*")):
        if path.is_dir() or path == manifest:
            continue
        destination = root / path.relative_to(staging)
        destination.parent.mkdir(parents=True, exist_ok=True)
        os.replace(path, destination)
    destination = root / "run_manifest.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    os.replace(manifest, destination)


def validate_task_shard(root: str | Path, task, fold_bundle) -> dict[str, object]:
    shard = task_shard_path(root, task)
    checkpoint_path = shard / TASK_CHECKPOINT_FILENAME
    if not checkpoint_path.is_file():
        raise ValueError(f"Missing task checkpoint: {checkpoint_path}")
    with checkpoint_path.open() as handle:
        checkpoint = json.load(handle)
    invariants = checkpoint.get("invariants")
    if not isinstance(invariants, dict):
        raise ValueError(f"Invalid checkpoint invariants: {checkpoint_path}")
    expected_identity = {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "task_id": task.task_id,
        "outer_fold": task.outer_fold,
        "layer": task.layer,
        "configuration_digest": CONFIGURATION_DIGEST,
        "cohort_digest": fold_bundle.fold_digests["cohort_digest"],
        "outer_fold_digest": fold_bundle.fold_digests["outer_fold_digest"],
        "pair_fold_manifest_digest": fold_bundle.fold_digests["pair_fold_manifest_digest"],
    }
    if any(invariants.get(key) != value for key, value in expected_identity.items()):
        raise ValueError(f"Task {task.task_id} checkpoint invariant mismatch")
    if not str(invariants.get("input_identity_digest", "")):
        raise ValueError(f"Task {task.task_id} lacks an input identity digest")
    validate_resume_checkpoint(
        checkpoint_path, expected_invariants=invariants,
        expected_artifact_names=set(expected_task_artifacts()),
    )

    tables = {
        logical: read_csv_strict(shard / relative, logical)
        for logical, relative in TASK_TABLES.items()
    }
    expected_ids = set(fold_bundle.protein_ids_for_outer_fold(task.outer_fold))
    expected_pairs = {
        protein_id: int(fold_bundle.outer_by_protein[protein_id].n_pairs)
        for protein_id in expected_ids
    }
    per_rows = tables[PER_PROTEIN_FILENAME]
    geometry_rows = tables[GEOMETRY_METRICS_FILENAME]
    if {row["protein_id"] for row in per_rows} != expected_ids:
        raise ValueError(f"Task {task.task_id} held-out protein membership mismatch")
    if len(per_rows) != len(expected_ids) or len(geometry_rows) != len(expected_ids):
        raise ValueError(f"Task {task.task_id} held-out row count mismatch")
    for row in per_rows:
        if (
            int(row["layer"]) != task.layer
            or int(row["outer_fold"]) != task.outer_fold
            or row["evaluation_design"] != EVALUATION_DESIGN
            or row["oof_prediction_scope"] != OOF_PREDICTION_SCOPE
            or row["prediction_status"] not in {"complete", "complete_from_flagged_fit"}
            or int(row["n_evaluated_pairs"]) != expected_pairs[row["protein_id"]]
            or not math.isclose(float(row["pair_coverage"]), 1.0)
        ):
            raise ValueError(f"Task {task.task_id} has an invalid per-protein OOF row")
    if {row["protein_id"] for row in geometry_rows} != expected_ids or any(
        row["geometry_status"] != "complete"
        or not math.isclose(float(row["pair_coverage"]), 1.0)
        for row in geometry_rows
    ):
        raise ValueError(f"Task {task.task_id} has incomplete geometry")

    if (
        len(tables[FOLD_METRICS_FILENAME]) != 1
        or len(tables[RUNTIME_FILENAME]) != 1
        or len(tables[TRAINING_METRICS_FILENAME]) != len(SEQUENCE_BIN_LABELS)
        or len(tables[MODEL_STATUS_FILENAME]) != len(SEQUENCE_BIN_LABELS)
        or len(tables[SUBGROUP_SEQUENCE_FILENAME]) != len(SEQUENCE_BIN_LABELS)
        or len(tables[SUBGROUP_SPATIAL_FILENAME]) != len(SPATIAL_DISTANCE_BIN_LABELS)
    ):
        raise ValueError(f"Task {task.task_id} table row count mismatch")
    for rows in tables.values():
        if any(int(row["layer"]) != task.layer for row in rows):
            raise ValueError(f"Task {task.task_id} contains a different layer")
    status_by_gate = {row["sequence_gate"]: row for row in tables[MODEL_STATUS_FILENAME]}
    training_by_gate = {row["sequence_gate"]: row for row in tables[TRAINING_METRICS_FILENAME]}
    if set(status_by_gate) != set(SEQUENCE_BIN_LABELS) or set(training_by_gate) != set(SEQUENCE_BIN_LABELS):
        raise ValueError(f"Task {task.task_id} does not contain exactly six gate rows")
    accepted_statuses = {
        "fitted", "fitted_underdetermined", "fitted_rank_deficient", "fitted_ill_conditioned",
    }
    for label in SEQUENCE_BIN_LABELS:
        status = status_by_gate[label]
        training = training_by_gate[label]
        if (
            status["fit_status"] not in accepted_statuses
            or int(status["outer_fold"]) != task.outer_fold
            or int(status["n_training_proteins"]) < MIN_TRAIN_PROTEINS
            or int(status["n_training_pairs"]) < MIN_TRAIN_ROWS
            or training["fit_status"] != status["fit_status"]
            or int(training["outer_fold"]) != task.outer_fold
            or int(training["n_training_pairs"]) != int(status["n_training_pairs"])
            or int(training["n_training_proteins"]) != int(status["n_training_proteins"])
        ):
            raise ValueError(f"Task {task.task_id} gate support/status mismatch for {label}")
    expected_training_ids = set(fold_bundle.eligible_protein_ids) - expected_ids
    expected_training_pairs = sum(
        int(fold_bundle.outer_by_protein[protein_id].n_pairs)
        for protein_id in expected_training_ids
    )
    if sum(int(row["n_training_pairs"]) for row in status_by_gate.values()) != expected_training_pairs:
        raise ValueError(f"Task {task.task_id} gates do not partition training pairs")
    runtime = tables[RUNTIME_FILENAME][0]
    if (
        runtime["completion_status"] != "complete"
        or int(runtime["task_id"]) != task.task_id
        or int(runtime["n_fitted_proteins"]) != len(expected_training_ids)
        or int(runtime["n_fitted_pairs"]) != expected_training_pairs
        or int(runtime["n_evaluated_proteins"]) != len(expected_ids)
        or int(runtime["n_evaluated_pairs"]) != sum(expected_pairs.values())
    ):
        raise ValueError(f"Task {task.task_id} runtime support mismatch")

    with (shard / TASK_PAYLOAD_PATH).open() as handle:
        payload = json.load(handle)
    required_payload = {
        "task_payload_version", "task_id", "outer_fold", "layer",
        "configuration_digest", "global_baseline_digest", "all_gates_stable",
        "gate_metadata", "oof_accumulator", "sequence_subgroups", "spatial_subgroups",
    }
    if set(payload) != required_payload or (
        payload["task_payload_version"] != TASK_PAYLOAD_VERSION
        or int(payload["task_id"]) != task.task_id
        or int(payload["outer_fold"]) != task.outer_fold
        or int(payload["layer"]) != task.layer
        or payload["configuration_digest"] != CONFIGURATION_DIGEST
        or set(payload["gate_metadata"]) != set(SEQUENCE_BIN_LABELS)
    ):
        raise ValueError(f"Task {task.task_id} payload identity mismatch")
    expected_stable = all(status_by_gate[label]["fit_status"] == "fitted" for label in SEQUENCE_BIN_LABELS)
    if bool(payload["all_gates_stable"]) != expected_stable:
        raise ValueError(f"Task {task.task_id} all-gate stability mismatch")
    oof = OOFComparisonAccumulator.from_payload(payload["oof_accumulator"])
    sequence = OOFSubgroupAccumulator.from_payload(payload["sequence_subgroups"])
    spatial = OOFSubgroupAccumulator.from_payload(payload["spatial_subgroups"])
    oof.require_complete()
    if (
        set(oof.expected_pairs_by_protein) != expected_ids
        or set(sequence.protein_ids) != expected_ids
        or set(spatial.protein_ids) != expected_ids
        or set(oof.outer_fold_by_protein.values()) != {task.outer_fold}
        or oof.conditional_baseline is None
    ):
        raise ValueError(f"Task {task.task_id} accumulator scope mismatch")
    prediction_status = "complete" if expected_stable else "complete_from_flagged_fit"
    for row in per_rows:
        protein_id = row["protein_id"]
        metrics = oof.per_protein_model[protein_id].metrics()
        if row["prediction_status"] != prediction_status:
            raise ValueError(f"Task {task.task_id} prediction/fit status mismatch")
        for key in (
            "target_variance", "r2", "mae", "rmse", "bias_pred_minus_true",
            "mean_true_distance", "mean_predicted_distance",
        ):
            if not np.isclose(
                _float(row, key), float(metrics[key]),
                rtol=1e-12, atol=1e-12, equal_nan=True,
            ):
                raise ValueError(f"Task {task.task_id} per-protein reconstruction failed: {key}")
    fold_row = tables[FOLD_METRICS_FILENAME][0]
    rebuilt_fold = build_fold_metrics_row(
        oof, outer_fold=task.outer_fold, analysis_name=ANALYSIS_NAME,
        model_family=MODEL_FAMILY, feature_dim=FEATURE_DIM, layer=task.layer,
        evaluation_design=EVALUATION_DESIGN,
        oof_prediction_scope=OOF_PREDICTION_SCOPE,
        conditional_baseline_type=CONDITIONAL_BASELINE_TYPE,
    )
    for key in (
        "n_proteins", "n_expected_pairs", "n_evaluated_pairs", "pooled_r2",
        "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true",
        "global_baseline_pooled_r2", "global_baseline_pooled_mae",
        "global_baseline_pooled_rmse", "global_baseline_pooled_bias_pred_minus_true",
        "conditional_baseline_pooled_r2", "conditional_baseline_pooled_mae",
        "conditional_baseline_pooled_rmse", "conditional_baseline_pooled_bias_pred_minus_true",
    ):
        if not np.isclose(
            _float(fold_row, key), float(rebuilt_fold[key]),
            rtol=1e-12, atol=1e-12, equal_nan=True,
        ):
            raise ValueError(f"Task {task.task_id} fold metric reconstruction failed: {key}")

    artifacts = {}
    for label in SEQUENCE_BIN_LABELS:
        metadata = payload["gate_metadata"][label]
        if set(metadata) != {
            "scaler_digest", "conditional_baseline_digest", "fit_status", "fit_stable",
        } or metadata["fit_status"] != status_by_gate[label]["fit_status"]:
            raise ValueError(f"Task {task.task_id} gate metadata mismatch for {label}")
        artifact_path = shard / task_model_path(label)
        artifact = load_coefficient_artifact(
            artifact_path, expected_task=task, expected_sequence_gate=label,
            expected_scaler_digest=str(metadata["scaler_digest"]),
        )
        if (
            artifact["fit_status"] != metadata["fit_status"]
            or artifact["conditional_baseline_digest"] != metadata["conditional_baseline_digest"]
            or artifact["global_baseline_digest"] != payload["global_baseline_digest"]
            or bool(metadata["fit_stable"]) != (metadata["fit_status"] == "fitted")
        ):
            raise ValueError(f"Task {task.task_id} artifact metadata mismatch for {label}")
        artifacts[label] = {"artifact": artifact, "path": artifact_path}

    subgroup_identity = dict(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM, layer=task.layer,
        evaluation_design=EVALUATION_DESIGN,
        oof_prediction_scope=OOF_PREDICTION_SCOPE,
    )
    for logical, observed_rows, rebuilt_rows in (
        (SUBGROUP_SEQUENCE_FILENAME, tables[SUBGROUP_SEQUENCE_FILENAME], sequence.rows(**subgroup_identity)),
        (SUBGROUP_SPATIAL_FILENAME, tables[SUBGROUP_SPATIAL_FILENAME], spatial.rows(**subgroup_identity)),
    ):
        if [row["category_label"] for row in observed_rows] != [
            row["category_label"] for row in rebuilt_rows
        ]:
            raise ValueError(f"Task {task.task_id} subgroup labels mismatch in {logical}")
        for observed_row, rebuilt_row in zip(observed_rows, rebuilt_rows):
            for key in (
                "n_proteins", "n_expected_pairs", "n_evaluated_pairs", "pair_coverage",
                "pooled_r2", "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true",
            ):
                if not np.isclose(
                    _float(observed_row, key), float(rebuilt_row[key]),
                    rtol=1e-12, atol=1e-12, equal_nan=True,
                ):
                    raise ValueError(f"Task {task.task_id} subgroup reconstruction failed: {logical} {key}")
    return {
        "tables": tables, "payload": payload, "oof": oof,
        "sequence": sequence, "spatial": spatial, "artifacts": artifacts,
        "input_identity_digest": invariants["input_identity_digest"],
    }


def aggregate_sequence_gated_ols(
    *,
    root: str | Path | None = None,
    fold_bundle=None,
    layers: Sequence[int] = LAYERS,
    allow_partial: bool = False,
) -> Path:
    destination = family_output_root() if root is None else Path(root)
    bundle = load_cv_folds() if fold_bundle is None else fold_bundle
    requested_layers = tuple(int(layer) for layer in layers)
    if not requested_layers or len(set(requested_layers)) != len(requested_layers):
        raise ValueError("Aggregation layers must be nonempty and unique")
    if any(layer not in LAYERS for layer in requested_layers):
        raise ValueError("Aggregation layers must be canonical layers 0..47")
    if not allow_partial and set(requested_layers) != set(LAYERS):
        raise ValueError("Production sequence-gated aggregation requires all 48 layers")
    tasks = [
        map_fold_layer_task(fold_layer_task_id(fold, layer))
        for layer in requested_layers for fold in range(N_OUTER_FOLDS)
    ]
    validated = {}
    for task in tasks:
        log(f"validating task {task.task_id} (fold {task.outer_fold}, layer {task.layer})")
        validated[(task.outer_fold, task.layer)] = validate_task_shard(destination, task, bundle)

    consolidated = {
        PER_PROTEIN_FILENAME: [], POOLED_METRICS_FILENAME: [], FOLD_METRICS_FILENAME: [],
        FOLD_SUMMARY_FILENAME: [], TRAINING_METRICS_FILENAME: [],
        STABLE_FIT_METRICS_FILENAME: [], MODEL_STATUS_FILENAME: [], RUNTIME_FILENAME: [],
        GEOMETRY_METRICS_FILENAME: [], GEOMETRY_SUMMARY_FILENAME: [],
        MODEL_STABILITY_FILENAME: [], SUBGROUP_SEQUENCE_FILENAME: [],
        SUBGROUP_SPATIAL_FILENAME: [],
    }
    artifact_records = []
    for layer in requested_layers:
        layer_oof = OOFComparisonAccumulator()
        layer_sequence = OOFSubgroupAccumulator("sequence_separation", SEQUENCE_BIN_LABELS)
        layer_spatial = OOFSubgroupAccumulator("spatial_distance", SPATIAL_DISTANCE_BIN_LABELS)
        fold_rows = []
        geometry_rows = []
        coefficient_by_gate = {label: [] for label in SEQUENCE_BIN_LABELS}
        for fold in range(N_OUTER_FOLDS):
            result = validated[(fold, layer)]
            layer_oof.merge(result["oof"])
            layer_sequence.merge(result["sequence"])
            layer_spatial.merge(result["spatial"])
            fold_rows.append(build_fold_metrics_row(
                result["oof"], outer_fold=fold, analysis_name=ANALYSIS_NAME,
                model_family=MODEL_FAMILY, feature_dim=FEATURE_DIM, layer=layer,
                evaluation_design=EVALUATION_DESIGN,
                oof_prediction_scope=OOF_PREDICTION_SCOPE,
                conditional_baseline_type=CONDITIONAL_BASELINE_TYPE,
            ))
            tables = result["tables"]
            consolidated[PER_PROTEIN_FILENAME].extend(tables[PER_PROTEIN_FILENAME])
            consolidated[TRAINING_METRICS_FILENAME].extend(tables[TRAINING_METRICS_FILENAME])
            consolidated[MODEL_STATUS_FILENAME].extend(tables[MODEL_STATUS_FILENAME])
            consolidated[RUNTIME_FILENAME].extend(tables[RUNTIME_FILENAME])
            geometry_rows.extend(tables[GEOMETRY_METRICS_FILENAME])
            task = map_fold_layer_task(fold_layer_task_id(fold, layer))
            for label in SEQUENCE_BIN_LABELS:
                coefficient_by_gate[label].append(np.asarray(
                    result["artifacts"][label]["artifact"]["coefficient_standardized"],
                    dtype=np.float64,
                ))
                artifact_records.append({
                    "task": task, "sequence_gate": label,
                    "source": result["artifacts"][label]["path"],
                    "scaler_digest": result["payload"]["gate_metadata"][label]["scaler_digest"],
                })
        layer_oof.require_complete()
        if set(layer_oof.expected_pairs_by_protein) != set(bundle.eligible_protein_ids):
            raise ValueError(f"Layer {layer} OOF cohort is incomplete")
        consolidated[POOLED_METRICS_FILENAME].append(build_pooled_metrics_row(
            layer_oof, analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=layer, evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            conditional_baseline_type=CONDITIONAL_BASELINE_TYPE,
        ))
        consolidated[FOLD_METRICS_FILENAME].extend(fold_rows)
        consolidated[FOLD_SUMMARY_FILENAME].extend(build_fold_summary_rows(fold_rows))
        consolidated[STABLE_FIT_METRICS_FILENAME].append(build_stable_fit_metrics_row(
            layer_oof, analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=layer, evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            conditional_baseline_type=CONDITIONAL_BASELINE_TYPE,
        ))
        consolidated[GEOMETRY_METRICS_FILENAME].extend(geometry_rows)
        consolidated[GEOMETRY_SUMMARY_FILENAME].extend(summarize_geometry_rows(geometry_rows))
        identity = dict(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=layer,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
        )
        consolidated[SUBGROUP_SEQUENCE_FILENAME].extend(layer_sequence.rows(**identity))
        consolidated[SUBGROUP_SPATIAL_FILENAME].extend(layer_spatial.rows(**identity))
        for label in SEQUENCE_BIN_LABELS:
            consolidated[MODEL_STABILITY_FILENAME].extend(build_model_stability_rows(
                coefficient_by_gate[label], analysis_name=ANALYSIS_NAME,
                model_family=MODEL_FAMILY, layer=layer, feature_names=FEATURE_NAMES,
                sequence_gate=label,
            ))

    n_layers = len(requested_layers)
    expected_counts = {
        PER_PROTEIN_FILENAME: len(bundle.eligible_protein_ids) * n_layers,
        POOLED_METRICS_FILENAME: n_layers,
        FOLD_METRICS_FILENAME: N_OUTER_FOLDS * n_layers,
        FOLD_SUMMARY_FILENAME: 12 * n_layers,
        TRAINING_METRICS_FILENAME: N_OUTER_FOLDS * len(SEQUENCE_BIN_LABELS) * n_layers,
        STABLE_FIT_METRICS_FILENAME: n_layers,
        MODEL_STATUS_FILENAME: N_OUTER_FOLDS * len(SEQUENCE_BIN_LABELS) * n_layers,
        RUNTIME_FILENAME: N_OUTER_FOLDS * n_layers,
        GEOMETRY_METRICS_FILENAME: len(bundle.eligible_protein_ids) * n_layers,
        GEOMETRY_SUMMARY_FILENAME: 14 * n_layers,
        MODEL_STABILITY_FILENAME: len(SEQUENCE_BIN_LABELS) * FEATURE_DIM * n_layers,
        SUBGROUP_SEQUENCE_FILENAME: len(SEQUENCE_BIN_LABELS) * n_layers,
        SUBGROUP_SPATIAL_FILENAME: len(SPATIAL_DISTANCE_BIN_LABELS) * n_layers,
    }
    for filename, count in expected_counts.items():
        if len(consolidated[filename]) != count:
            raise ValueError(f"Consolidated row count mismatch for {filename}")
    unique_keys = {
        PER_PROTEIN_FILENAME: ("analysis_name", "layer", "protein_id"),
        POOLED_METRICS_FILENAME: ("analysis_name", "layer"),
        FOLD_METRICS_FILENAME: ("analysis_name", "layer", "outer_fold"),
        FOLD_SUMMARY_FILENAME: ("analysis_name", "layer", "fold_metric"),
        TRAINING_METRICS_FILENAME: ("analysis_name", "layer", "outer_fold", "sequence_gate"),
        STABLE_FIT_METRICS_FILENAME: ("analysis_name", "layer"),
        MODEL_STATUS_FILENAME: ("analysis_name", "layer", "outer_fold", "sequence_gate"),
        RUNTIME_FILENAME: ("analysis_name", "task_id"),
        GEOMETRY_METRICS_FILENAME: ("analysis_name", "layer", "protein_id"),
        GEOMETRY_SUMMARY_FILENAME: ("analysis_name", "layer", "geometry_metric"),
        MODEL_STABILITY_FILENAME: ("analysis_name", "layer", "sequence_gate", "feature_index"),
        SUBGROUP_SEQUENCE_FILENAME: ("analysis_name", "layer", "category_label"),
        SUBGROUP_SPATIAL_FILENAME: ("analysis_name", "layer", "category_label"),
    }
    for filename, keys in unique_keys.items():
        _unique(consolidated[filename], keys, filename)

    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix=f".{ANALYSIS_NAME}.aggregate.", dir=destination.parent))
    try:
        for filename, rows in consolidated.items():
            write_v2_csv(staging / filename, filename, rows)
        replay_rows = []
        for record in sorted(
            artifact_records,
            key=lambda item: (item["task"].layer, item["task"].outer_fold, item["sequence_gate"]),
        ):
            task = record["task"]
            label = record["sequence_gate"]
            relative = Path("model_artifacts") / "coefficients" / (
                f"gate_{gate_slug(label)}_fold_{task.outer_fold}_layer_{task.layer:02d}.npz"
            )
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["source"], target)
            load_coefficient_artifact(
                target, expected_task=task, expected_sequence_gate=label,
                expected_scaler_digest=str(record["scaler_digest"]),
            )
            replay_rows.append(build_replay_index_row(
                family_root=staging, artifact_path=relative,
                analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
                layer=task.layer, outer_fold=task.outer_fold, sequence_gate=label,
                artifact_type="sequence_gate_ols_coefficients_scaler_npz",
                configuration_digest=CONFIGURATION_DIGEST,
                scaler_digest=str(record["scaler_digest"]), status="complete",
            ))
        expected_models = N_OUTER_FOLDS * len(SEQUENCE_BIN_LABELS) * n_layers
        validate_replay_index(replay_rows, family_root=staging, expected_count=expected_models)
        write_v2_csv(staging / REPLAY_INDEX_FILENAME, REPLAY_INDEX_FILENAME, replay_rows)
        task_ids = sorted(task.task_id for task in tasks)
        manifest = build_run_manifest(
            fold_bundle=bundle, analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            requested_task_ids=task_ids, completed_task_ids=task_ids,
            configuration=CONFIGURATION, expected_replay_artifacts=expected_models,
            versions=version_payload(), code_revision=code_revision(), run_status="complete",
        )
        manifest.update({
            "requested_layers": list(requested_layers),
            "task_mapping": "fold_major_task_id=outer_fold*48+layer",
            "task_input_identity_digests": {
                str(task.task_id): validated[(task.outer_fold, task.layer)]["input_identity_digest"]
                for task in sorted(tasks, key=lambda item: item.task_id)
            },
            "sequence_gates": list(SEQUENCE_BIN_LABELS),
            "gate_failure_policy": "any_invalid_or_missing_gate_fails_the_fold_layer_shard",
            "coefficient_artifact_layout":
                "model_artifacts/coefficients/gate_<gate>_fold_<outer_fold>_layer_<layer>.npz",
            "raw_pair_predictions_saved": False,
        })
        atomic_write_json(staging / "run_manifest.json", manifest)
        _publish_staging(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log(f"published {n_layers} sequence-gated OLS layers to {destination}")
    return destination


if __name__ == "__main__":
    aggregate_sequence_gated_ols()

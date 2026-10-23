#!/usr/bin/env python3
"""Validate and consolidate fivefold fixed-S4 shared XGBoost shards."""

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
    FEATURE_IMPORTANCE_FILENAME,
    FOLD_METRICS_FILENAME,
    FOLD_SUMMARY_FILENAME,
    GEOMETRY_METRICS_FILENAME,
    GEOMETRY_SUMMARY_FILENAME,
    MODEL_STATUS_FILENAME,
    OOFComparisonAccumulator,
    OOFSubgroupAccumulator,
    PER_PROTEIN_FILENAME,
    POOLED_METRICS_FILENAME,
    REPLAY_INDEX_FILENAME,
    RUNTIME_FILENAME,
    SUBGROUP_SEQUENCE_FILENAME,
    SUBGROUP_SPATIAL_FILENAME,
    TRAINING_METRICS_FILENAME,
    build_feature_importance_rows,
    build_fold_metrics_row,
    build_fold_summary_rows,
    build_pooled_metrics_row,
    build_replay_index_row,
    build_run_manifest,
    fold_layer_task_id,
    load_cv_folds,
    map_fold_layer_task,
    semantic_digest,
    summarize_geometry_rows,
    validate_replay_index,
    validate_resume_checkpoint,
    validate_shard_inventory,
    write_v2_csv,
)
from regression_output_helpers import atomic_write_json, sha256_file  # noqa: E402
from regression_protocol import FEATURE_DIM, LAYERS, N_OUTER_FOLDS  # noqa: E402
from shared_xgboost_base256_fixed_s4_common import (  # noqa: E402
    ANALYSIS_NAME,
    BASE_RANDOM_SEED,
    BOOSTING_ROUNDS,
    CONFIGURATION,
    CONFIGURATION_DIGEST,
    CONFIGURATION_ID,
    CONFIG_PATH,
    EVALUATION_DESIGN,
    FIT_CHECKPOINT_FILENAME,
    MODEL_FAMILY,
    OOF_PREDICTION_SCOPE,
    ROW_CAP,
    TASK_BOOSTER_PATH,
    TASK_CHECKPOINT_FILENAME,
    TASK_IMPORTANCE_PATH,
    TASK_PAYLOAD_PATH,
    TASK_PAYLOAD_VERSION,
    TASK_TABLES,
    code_revision,
    expected_task_artifacts,
    family_output_root,
    fit_checkpoint_invariants,
    load_importance_artifact,
    log,
    read_csv_strict,
    sampling_manifest_digest,
    task_shard_path,
)


from canonical_storage import assert_writable, normalize_shared_tree


def _float(row: Mapping[str, object], key: str) -> float:
    return float(row[key])


def _training_assignment_digest(fold_bundle, outer_fold: int) -> str:
    from cross_validation_common import sampling_assignment_digest

    rows = []
    for protein_id in fold_bundle.training_protein_ids(outer_fold):
        entry = fold_bundle.outer_by_protein[protein_id]
        rows.append({
            "protein_id": protein_id,
            "n_pairs": int(entry.n_pairs),
            "n_selected_pairs": min(int(entry.n_pairs), ROW_CAP),
            "assignment_digest": sampling_assignment_digest(
                entry.n_pairs, protein_id, ROW_CAP, base_seed=BASE_RANDOM_SEED,
            ),
        })
    return semantic_digest(rows)


def validate_task_shard(
    root: str | Path,
    task,
    fold_bundle,
    *,
    expected_sampling_digest: str | None = None,
    expected_training_digest: str | None = None,
) -> dict[str, object]:
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
        "sampling_manifest_digest": (
            sampling_manifest_digest(fold_bundle)
            if expected_sampling_digest is None else str(expected_sampling_digest)
        ),
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
    training_ids = set(fold_bundle.eligible_protein_ids) - expected_ids
    expected_pairs = {
        protein_id: int(fold_bundle.outer_by_protein[protein_id].n_pairs)
        for protein_id in expected_ids
    }
    expected_training_pairs = sum(
        min(int(fold_bundle.outer_by_protein[protein_id].n_pairs), ROW_CAP)
        for protein_id in training_ids
    )
    per_rows = tables[PER_PROTEIN_FILENAME]
    geometry_rows = tables[GEOMETRY_METRICS_FILENAME]
    if (
        {row["protein_id"] for row in per_rows} != expected_ids
        or len(per_rows) != len(expected_ids)
        or len(geometry_rows) != len(expected_ids)
    ):
        raise ValueError(f"Task {task.task_id} held-out membership/count mismatch")
    for row in per_rows:
        if (
            int(row["layer"]) != task.layer
            or int(row["outer_fold"]) != task.outer_fold
            or row["evaluation_design"] != EVALUATION_DESIGN
            or row["oof_prediction_scope"] != OOF_PREDICTION_SCOPE
            or row["prediction_status"] != "complete"
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

    singleton_tables = (
        FOLD_METRICS_FILENAME, TRAINING_METRICS_FILENAME,
        MODEL_STATUS_FILENAME, RUNTIME_FILENAME,
    )
    if any(len(tables[name]) != 1 for name in singleton_tables):
        raise ValueError(f"Task {task.task_id} singleton table count mismatch")
    if len(tables[SUBGROUP_SEQUENCE_FILENAME]) != 6 or len(tables[SUBGROUP_SPATIAL_FILENAME]) != 5:
        raise ValueError(f"Task {task.task_id} subgroup row count mismatch")
    for rows in tables.values():
        if any(int(row["layer"]) != task.layer for row in rows):
            raise ValueError(f"Task {task.task_id} contains a different layer")
    fold_row = tables[FOLD_METRICS_FILENAME][0]
    training = tables[TRAINING_METRICS_FILENAME][0]
    status = tables[MODEL_STATUS_FILENAME][0]
    runtime = tables[RUNTIME_FILENAME][0]
    if (
        int(fold_row["outer_fold"]) != task.outer_fold
        or not math.isclose(float(fold_row["pair_coverage"]), 1.0)
        or status["fit_status"] != "fitted"
        or status["selected_configuration_id"] != CONFIGURATION_ID
        or int(status["final_boosting_rounds"]) != BOOSTING_ROUNDS
        or int(status["training_row_cap"]) != ROW_CAP
        or int(status["random_seed"]) != BASE_RANDOM_SEED + task.outer_fold
        or int(status["n_training_proteins"]) != len(training_ids)
        or int(status["n_training_pairs"]) != expected_training_pairs
        or training["fit_status"] != "fitted"
        or training["training_scope"] != "sampled_pairs_four_outer_training_folds"
        or int(training["n_training_pairs"]) != expected_training_pairs
        or runtime["completion_status"] != "complete"
        or int(runtime["task_id"]) != task.task_id
        or int(runtime["n_fitted_pairs"]) != expected_training_pairs
        or int(runtime["n_evaluated_proteins"]) != len(expected_ids)
        or int(runtime["n_evaluated_pairs"]) != sum(expected_pairs.values())
    ):
        raise ValueError(f"Task {task.task_id} fitting/runtime support mismatch")

    with (shard / TASK_PAYLOAD_PATH).open() as handle:
        payload = json.load(handle)
    required_payload = {
        "task_payload_version", "task_id", "outer_fold", "layer",
        "configuration_digest", "sampling_manifest_digest",
        "training_sample_assignment_digest", "global_baseline_digest",
        "global_baseline_value", "random_seed", "final_boosting_rounds",
        "versions", "oof_accumulator", "sequence_subgroups",
        "spatial_subgroups",
    }
    if set(payload) != required_payload or (
        payload["task_payload_version"] != TASK_PAYLOAD_VERSION
        or int(payload["task_id"]) != task.task_id
        or int(payload["outer_fold"]) != task.outer_fold
        or int(payload["layer"]) != task.layer
        or payload["configuration_digest"] != CONFIGURATION_DIGEST
        or payload["sampling_manifest_digest"] != expected_identity["sampling_manifest_digest"]
        or payload["training_sample_assignment_digest"]
        != (
            _training_assignment_digest(fold_bundle, task.outer_fold)
            if expected_training_digest is None else str(expected_training_digest)
        )
        or int(payload["random_seed"]) != BASE_RANDOM_SEED + task.outer_fold
        or int(payload["final_boosting_rounds"]) != BOOSTING_ROUNDS
        or not isinstance(payload["versions"], dict)
        or not str(payload["versions"].get("xgboost", ""))
        or not str(payload["global_baseline_digest"])
        or not math.isfinite(float(payload["global_baseline_value"]))
    ):
        raise ValueError(f"Task {task.task_id} payload identity mismatch")

    fit_invariants = fit_checkpoint_invariants(
        task, invariants,
        training_sample_digest=str(payload["training_sample_assignment_digest"]),
        xgboost_version=str(payload["versions"]["xgboost"]),
    )
    validate_resume_checkpoint(
        shard / FIT_CHECKPOINT_FILENAME, expected_invariants=fit_invariants,
        expected_artifact_names={"booster"},
    )
    booster_path = shard / TASK_BOOSTER_PATH
    try:
        booster_json = json.loads(booster_path.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"Task {task.task_id} booster is not native JSON") from exc
    if not isinstance(booster_json, dict) or not booster_json:
        raise ValueError(f"Task {task.task_id} booster JSON is empty")

    oof = OOFComparisonAccumulator.from_payload(payload["oof_accumulator"])
    sequence = OOFSubgroupAccumulator.from_payload(payload["sequence_subgroups"])
    spatial = OOFSubgroupAccumulator.from_payload(payload["spatial_subgroups"])
    oof.require_complete()
    if (
        set(oof.expected_pairs_by_protein) != expected_ids
        or set(sequence.protein_ids) != expected_ids
        or set(spatial.protein_ids) != expected_ids
        or set(oof.outer_fold_by_protein.values()) != {task.outer_fold}
        or not all(oof.stable_fit_by_protein.values())
    ):
        raise ValueError(f"Task {task.task_id} accumulator membership mismatch")
    for row in per_rows:
        metrics = oof.per_protein_model[row["protein_id"]].metrics()
        for key in (
            "target_variance", "r2", "mae", "rmse", "bias_pred_minus_true",
            "mean_true_distance", "mean_predicted_distance",
        ):
            if not np.isclose(
                _float(row, key), float(metrics[key]),
                rtol=1e-12, atol=1e-12, equal_nan=True,
            ):
                raise ValueError(f"Task {task.task_id} per-protein reconstruction failed: {key}")
    rebuilt_fold = build_fold_metrics_row(
        oof, outer_fold=task.outer_fold, analysis_name=ANALYSIS_NAME,
        model_family=MODEL_FAMILY, feature_dim=FEATURE_DIM, layer=task.layer,
        evaluation_design=EVALUATION_DESIGN,
        oof_prediction_scope=OOF_PREDICTION_SCOPE,
    )
    for key in (
        "n_proteins", "n_expected_pairs", "n_evaluated_pairs", "pooled_r2",
        "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true",
        "global_baseline_pooled_r2", "global_baseline_pooled_mae",
        "global_baseline_pooled_rmse", "global_baseline_pooled_bias_pred_minus_true",
    ):
        if not np.isclose(
            _float(fold_row, key), float(rebuilt_fold[key]),
            rtol=1e-12, atol=1e-12, equal_nan=True,
        ):
            raise ValueError(f"Task {task.task_id} fold metric reconstruction failed: {key}")
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
                    raise ValueError(f"Task {task.task_id} subgroup reconstruction failed")
    importance = load_importance_artifact(shard / TASK_IMPORTANCE_PATH, expected_task=task)
    return {
        "tables": tables, "payload": payload, "oof": oof,
        "sequence": sequence, "spatial": spatial, "importance": importance,
        "booster_path": booster_path,
        "input_identity_digest": invariants["input_identity_digest"],
    }


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


def aggregate_shared_xgboost(
    *,
    root: str | Path | None = None,
    fold_bundle=None,
    layers: Sequence[int] = LAYERS,
    allow_partial: bool = False,
) -> Path:
    assert_writable(family_output_root() if root is None else root)
    destination = family_output_root() if root is None else Path(root)
    bundle = load_cv_folds() if fold_bundle is None else fold_bundle
    requested_layers = tuple(int(layer) for layer in layers)
    if not requested_layers or len(set(requested_layers)) != len(requested_layers):
        raise ValueError("Aggregation layers must be nonempty and unique")
    if any(layer not in LAYERS for layer in requested_layers):
        raise ValueError("Aggregation layers must be canonical layers 0..47")
    if not allow_partial and set(requested_layers) != set(LAYERS):
        raise ValueError("Production shared-XGBoost aggregation requires all 48 layers")
    tasks = [
        map_fold_layer_task(fold_layer_task_id(fold, layer))
        for layer in requested_layers for fold in range(N_OUTER_FOLDS)
    ]
    validate_shard_inventory(
        destination,
        expected_shards={
            task_shard_path(destination, task).name: task.task_id for task in tasks
        },
        checkpoint_filename=TASK_CHECKPOINT_FILENAME,
    )
    validated = {}
    expected_sampling_digest = sampling_manifest_digest(bundle)
    training_digests = {
        fold: _training_assignment_digest(bundle, fold)
        for fold in range(N_OUTER_FOLDS)
    }
    for task in tasks:
        log(f"validating task {task.task_id} (fold {task.outer_fold}, layer {task.layer})")
        validated[(task.outer_fold, task.layer)] = validate_task_shard(
            destination, task, bundle,
            expected_sampling_digest=expected_sampling_digest,
            expected_training_digest=training_digests[task.outer_fold],
        )

    consolidated = {
        PER_PROTEIN_FILENAME: [], POOLED_METRICS_FILENAME: [],
        FOLD_METRICS_FILENAME: [], FOLD_SUMMARY_FILENAME: [],
        TRAINING_METRICS_FILENAME: [], MODEL_STATUS_FILENAME: [],
        RUNTIME_FILENAME: [], GEOMETRY_METRICS_FILENAME: [],
        GEOMETRY_SUMMARY_FILENAME: [], SUBGROUP_SEQUENCE_FILENAME: [],
        SUBGROUP_SPATIAL_FILENAME: [], FEATURE_IMPORTANCE_FILENAME: [],
    }
    artifact_records = []
    for layer in requested_layers:
        layer_oof = OOFComparisonAccumulator()
        layer_sequence = OOFSubgroupAccumulator(
            "sequence_separation", ("1", "2-4", "5-9", "10-19", "20-49", "50+"),
        )
        layer_spatial = OOFSubgroupAccumulator(
            "spatial_distance", ("0-5", "5-10", "10-20", "20-40", "40+"),
        )
        fold_rows = []
        geometry_rows = []
        gains = []
        splits = []
        for fold in range(N_OUTER_FOLDS):
            result = validated[(fold, layer)]
            layer_oof.merge(result["oof"])
            layer_sequence.merge(result["sequence"])
            layer_spatial.merge(result["spatial"])
            task = map_fold_layer_task(fold_layer_task_id(fold, layer))
            fold_rows.append(build_fold_metrics_row(
                result["oof"], outer_fold=fold, analysis_name=ANALYSIS_NAME,
                model_family=MODEL_FAMILY, feature_dim=FEATURE_DIM, layer=layer,
                evaluation_design=EVALUATION_DESIGN,
                oof_prediction_scope=OOF_PREDICTION_SCOPE,
            ))
            tables = result["tables"]
            consolidated[PER_PROTEIN_FILENAME].extend(tables[PER_PROTEIN_FILENAME])
            consolidated[TRAINING_METRICS_FILENAME].extend(tables[TRAINING_METRICS_FILENAME])
            consolidated[MODEL_STATUS_FILENAME].extend(tables[MODEL_STATUS_FILENAME])
            consolidated[RUNTIME_FILENAME].extend(tables[RUNTIME_FILENAME])
            geometry_rows.extend(tables[GEOMETRY_METRICS_FILENAME])
            gains.append(result["importance"]["gain"])
            splits.append(result["importance"]["split_count"])
            artifact_records.append({"task": task, "source": result["booster_path"]})
        layer_oof.require_complete()
        if set(layer_oof.expected_pairs_by_protein) != set(bundle.eligible_protein_ids):
            raise ValueError(f"Layer {layer} OOF cohort is incomplete")
        consolidated[POOLED_METRICS_FILENAME].append(build_pooled_metrics_row(
            layer_oof, analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=layer, evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
        ))
        consolidated[FOLD_METRICS_FILENAME].extend(fold_rows)
        consolidated[FOLD_SUMMARY_FILENAME].extend(build_fold_summary_rows(fold_rows))
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
        consolidated[FEATURE_IMPORTANCE_FILENAME].extend(build_feature_importance_rows(
            np.asarray(gains, dtype=np.float64), np.asarray(splits, dtype=np.float64),
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY, layer=layer,
        ))

    expected_counts = {
        PER_PROTEIN_FILENAME: len(bundle.eligible_protein_ids) * len(requested_layers),
        POOLED_METRICS_FILENAME: len(requested_layers),
        FOLD_METRICS_FILENAME: N_OUTER_FOLDS * len(requested_layers),
        FOLD_SUMMARY_FILENAME: 12 * len(requested_layers),
        TRAINING_METRICS_FILENAME: N_OUTER_FOLDS * len(requested_layers),
        MODEL_STATUS_FILENAME: N_OUTER_FOLDS * len(requested_layers),
        RUNTIME_FILENAME: N_OUTER_FOLDS * len(requested_layers),
        GEOMETRY_METRICS_FILENAME: len(bundle.eligible_protein_ids) * len(requested_layers),
        GEOMETRY_SUMMARY_FILENAME: 14 * len(requested_layers),
        SUBGROUP_SEQUENCE_FILENAME: 6 * len(requested_layers),
        SUBGROUP_SPATIAL_FILENAME: 5 * len(requested_layers),
        FEATURE_IMPORTANCE_FILENAME: FEATURE_DIM * len(requested_layers),
    }
    for filename, count in expected_counts.items():
        if len(consolidated[filename]) != count:
            raise ValueError(f"Consolidated row count mismatch for {filename}")
    unique_keys = {
        PER_PROTEIN_FILENAME: ("analysis_name", "layer", "protein_id"),
        POOLED_METRICS_FILENAME: ("analysis_name", "layer"),
        FOLD_METRICS_FILENAME: ("analysis_name", "layer", "outer_fold"),
        FOLD_SUMMARY_FILENAME: ("analysis_name", "layer", "fold_metric"),
        TRAINING_METRICS_FILENAME: ("analysis_name", "layer", "outer_fold"),
        MODEL_STATUS_FILENAME: ("analysis_name", "layer", "outer_fold"),
        RUNTIME_FILENAME: ("analysis_name", "task_id"),
        GEOMETRY_METRICS_FILENAME: ("analysis_name", "layer", "protein_id"),
        GEOMETRY_SUMMARY_FILENAME: ("analysis_name", "layer", "geometry_metric"),
        SUBGROUP_SEQUENCE_FILENAME: ("analysis_name", "layer", "category_label"),
        SUBGROUP_SPATIAL_FILENAME: ("analysis_name", "layer", "category_label"),
        FEATURE_IMPORTANCE_FILENAME: ("analysis_name", "layer", "feature_index"),
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
            key=lambda item: (item["task"].layer, item["task"].outer_fold),
        ):
            task = record["task"]
            relative = Path("model_artifacts") / "boosters" / (
                f"fold_{task.outer_fold}_layer_{task.layer:02d}.json"
            )
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(record["source"], target)
            if sha256_file(target) != sha256_file(record["source"]):
                raise ValueError("Copied booster checksum mismatch")
            replay_rows.append(build_replay_index_row(
                family_root=staging, artifact_path=relative,
                analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
                layer=task.layer, outer_fold=task.outer_fold,
                artifact_type="xgboost_native_json_booster",
                configuration_digest=CONFIGURATION_DIGEST,
                scaler_digest="", status="complete",
            ))
        validate_replay_index(
            replay_rows, family_root=staging,
            expected_count=N_OUTER_FOLDS * len(requested_layers),
        )
        write_v2_csv(staging / REPLAY_INDEX_FILENAME, REPLAY_INDEX_FILENAME, replay_rows)
        fixed_config_path = staging / "model_artifacts" / "fixed_configuration.json"
        fixed_config_path.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CONFIG_PATH, fixed_config_path)
        task_ids = sorted(task.task_id for task in tasks)
        task_version_payloads = {
            semantic_digest(result["payload"]["versions"]): dict(
                result["payload"]["versions"]
            )
            for result in validated.values()
        }
        if len(task_version_payloads) != 1:
            raise ValueError("Task shards disagree on fitted library versions")
        # Import-free aggregation is supported locally; fitted versions come
        # authoritatively from the validated task payloads.
        versions = next(iter(task_version_payloads.values()))
        manifest = build_run_manifest(
            fold_bundle=bundle, analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            requested_task_ids=task_ids, completed_task_ids=task_ids,
            configuration=CONFIGURATION,
            expected_replay_artifacts=N_OUTER_FOLDS * len(requested_layers),
            versions=versions, code_revision=code_revision(), run_status="complete",
        )
        manifest.update({
            "requested_layers": list(requested_layers),
            "task_mapping": "fold_major_task_id=outer_fold*48+layer",
            "task_input_identity_digests": {
                str(task.task_id): validated[(task.outer_fold, task.layer)][
                    "input_identity_digest"
                ]
                for task in sorted(tasks, key=lambda item: item.task_id)
            },
            "sampling_manifest_digest": expected_sampling_digest,
            "fixed_configuration_path": "model_artifacts/fixed_configuration.json",
            "fixed_configuration_sha256": sha256_file(fixed_config_path),
            "selection_source_path": CONFIGURATION["selection_source_path"],
            "selection_source_sha256": CONFIGURATION["selection_source_sha256"],
            "selection_is_nested": False,
            "fully_selection_independent": False,
            "historical_test_exposure_disclosed": True,
            "booster_artifact_layout":
                "model_artifacts/boosters/fold_<outer_fold>_layer_<layer>.json",
            "raw_pair_predictions_saved": False,
        })
        atomic_write_json(staging / "run_manifest.json", manifest)
        _publish_staging(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log(f"published {len(requested_layers)} shared-XGBoost layers to {destination}")
    normalize_shared_tree(destination)
    return destination


if __name__ == "__main__":
    aggregate_shared_xgboost()

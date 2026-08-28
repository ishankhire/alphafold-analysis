#!/usr/bin/env python3
"""Validate and consolidate fivefold within-protein base256 OLS layer shards."""

from __future__ import annotations

import csv
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
    load_cv_folds,
    summarize_geometry_rows,
    validate_replay_index,
    validate_resume_checkpoint,
    write_v2_csv,
    V2_CSV_SCHEMAS,
)
from per_protein_ols_base256_common import (  # noqa: E402
    ANALYSIS_NAME,
    CONDITIONAL_BASELINE_TYPE,
    CONFIGURATION,
    CONFIGURATION_DIGEST,
    EVALUATION_DESIGN,
    MODEL_FAMILY,
    OOF_PREDICTION_SCOPE,
    TASK_CHECKPOINT_FILENAME,
    TASK_MODEL_PATH,
    TASK_PAYLOAD_PATH,
    TASK_PAYLOAD_VERSION,
    TASK_TABLES,
    expected_task_artifacts,
    family_output_root,
    load_coefficient_shard,
    read_csv_strict,
    task_shard_path,
)
from regression_output_helpers import atomic_write_json, sha256_file  # noqa: E402
from regression_protocol import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_NAMES,
    LAYERS,
    N_OUTER_FOLDS,
    N_PAIR_FOLDS,
)
from shared_ols_base256_common import code_revision, log, version_payload  # noqa: E402


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


def _concatenate_stability_shards(
    destination: Path,
    sources: Sequence[Path],
    *,
    expected_rows: int,
) -> None:
    """Stream validated layer stability tables without materializing all 48."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    columns = V2_CSV_SCHEMAS[MODEL_STABILITY_FILENAME]
    row_count = 0
    with destination.open("w", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=columns, extrasaction="raise")
        writer.writeheader()
        for source in sources:
            with source.open(newline="") as handle:
                reader = csv.DictReader(handle)
                if tuple(reader.fieldnames or ()) != columns:
                    raise ValueError(f"Schema mismatch in {source}")
                for row in reader:
                    writer.writerow(row)
                    row_count += 1
    if row_count != int(expected_rows):
        raise ValueError(
            f"Consolidated row count mismatch for {MODEL_STABILITY_FILENAME}: "
            f"{row_count} != {expected_rows}"
        )


def validate_task_shard(root: str | Path, layer: int, fold_bundle) -> dict[str, object]:
    shard = task_shard_path(root, layer)
    checkpoint_path = shard / TASK_CHECKPOINT_FILENAME
    if not checkpoint_path.is_file():
        raise ValueError(f"Missing layer checkpoint: {checkpoint_path}")
    with checkpoint_path.open() as handle:
        checkpoint = json.load(handle)
    invariants = checkpoint.get("invariants")
    expected_identity = {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "task_id": int(layer),
        "layer": int(layer),
        "configuration_digest": CONFIGURATION_DIGEST,
        "cohort_digest": fold_bundle.fold_digests["cohort_digest"],
        "outer_fold_digest": fold_bundle.fold_digests["outer_fold_digest"],
        "pair_fold_manifest_digest": fold_bundle.fold_digests["pair_fold_manifest_digest"],
    }
    if not isinstance(invariants, dict) or any(
        invariants.get(key) != value for key, value in expected_identity.items()
    ):
        raise ValueError(f"Layer {layer} checkpoint invariant mismatch")
    if not str(invariants.get("input_identity_digest", "")):
        raise ValueError(f"Layer {layer} lacks an input identity digest")
    validate_resume_checkpoint(
        checkpoint_path, expected_invariants=invariants,
        expected_artifact_names=set(expected_task_artifacts()),
    )

    tables = {
        logical: read_csv_strict(shard / relative, logical)
        for logical, relative in TASK_TABLES.items()
    }
    expected_ids = tuple(sorted(fold_bundle.eligible_protein_ids))
    expected_id_set = set(expected_ids)
    expected_model_keys = {
        (protein_id, str(pair_fold))
        for protein_id in expected_ids for pair_fold in range(N_PAIR_FOLDS)
    }
    per_rows = tables[PER_PROTEIN_FILENAME]
    geometry_rows = tables[GEOMETRY_METRICS_FILENAME]
    training_rows = tables[TRAINING_METRICS_FILENAME]
    status_rows = tables[MODEL_STATUS_FILENAME]
    stability_rows = tables[MODEL_STABILITY_FILENAME]
    if (
        len(per_rows) != len(expected_ids)
        or {row["protein_id"] for row in per_rows} != expected_id_set
        or len(geometry_rows) != len(expected_ids)
        or {row["protein_id"] for row in geometry_rows} != expected_id_set
    ):
        raise ValueError(f"Layer {layer} protein coverage mismatch")
    for row in per_rows:
        protein_id = row["protein_id"]
        expected_pairs = int(fold_bundle.outer_by_protein[protein_id].n_pairs)
        expected_fold = int(fold_bundle.outer_by_protein[protein_id].outer_fold)
        if (
            int(row["layer"]) != int(layer)
            or int(row["outer_fold"]) != expected_fold
            or row["evaluation_design"] != EVALUATION_DESIGN
            or row["oof_prediction_scope"] != OOF_PREDICTION_SCOPE
            or row["prediction_status"] not in {"complete", "complete_from_flagged_fit"}
            or int(row["n_evaluated_pairs"]) != expected_pairs
            or not math.isclose(float(row["pair_coverage"]), 1.0)
        ):
            raise ValueError(f"Layer {layer} has an invalid per-protein OOF row")
    if any(
        row["geometry_status"] != "complete"
        or not math.isclose(float(row["pair_coverage"]), 1.0)
        or int(row["layer"]) != int(layer)
        for row in geometry_rows
    ):
        raise ValueError(f"Layer {layer} has incomplete OOF geometry")
    if (
        len(training_rows) != len(expected_model_keys)
        or len(status_rows) != len(expected_model_keys)
        or {(row["protein_id"], row["pair_fold"]) for row in training_rows} != expected_model_keys
        or {(row["protein_id"], row["pair_fold"]) for row in status_rows} != expected_model_keys
    ):
        raise ValueError(f"Layer {layer} fitted-model coverage mismatch")
    allowed_statuses = {
        "fitted", "fitted_underdetermined", "fitted_rank_deficient",
        "fitted_ill_conditioned",
    }
    status_by_key = {(row["protein_id"], row["pair_fold"]): row for row in status_rows}
    for row in training_rows:
        key = (row["protein_id"], row["pair_fold"])
        status = status_by_key[key]
        protein_id, pair_fold_text = key
        pair_fold = int(pair_fold_text)
        pair_entry = fold_bundle.pair_by_protein[protein_id]
        expected_training = int(pair_entry.n_pairs) - int(pair_entry.fold_counts[pair_fold])
        expected_outer = int(fold_bundle.outer_by_protein[protein_id].outer_fold)
        if (
            status["fit_status"] not in allowed_statuses
            or row["fit_status"] != status["fit_status"]
            or int(row["n_training_pairs"]) != expected_training
            or int(status["n_training_pairs"]) != expected_training
            or int(row["outer_fold"]) != expected_outer
            or int(status["outer_fold"]) != expected_outer
        ):
            raise ValueError(f"Layer {layer} model support mismatch for {key}")
    if (
        len(tables[FOLD_METRICS_FILENAME]) != N_OUTER_FOLDS
        or {int(row["outer_fold"]) for row in tables[FOLD_METRICS_FILENAME]}
        != set(range(N_OUTER_FOLDS))
        or len(tables[RUNTIME_FILENAME]) != 1
        or len(tables[SUBGROUP_SEQUENCE_FILENAME]) != 6
        or len(tables[SUBGROUP_SPATIAL_FILENAME]) != 5
        or len(stability_rows) != len(expected_ids) * FEATURE_DIM
    ):
        raise ValueError(f"Layer {layer} table row count mismatch")
    _unique(stability_rows, ("protein_id", "feature_index"), "task model stability")
    if any(row["n_models"] != str(N_PAIR_FOLDS) for row in stability_rows):
        raise ValueError(f"Layer {layer} coefficient stability does not cover five models")
    runtime = tables[RUNTIME_FILENAME][0]
    if (
        runtime["completion_status"] != "complete"
        or int(runtime["task_id"]) != int(layer)
        or int(runtime["n_fitted_proteins"]) != len(expected_ids)
        or int(runtime["n_evaluated_proteins"]) != len(expected_ids)
        or int(runtime["n_evaluated_pairs"])
        != sum(int(fold_bundle.outer_by_protein[protein_id].n_pairs) for protein_id in expected_ids)
    ):
        raise ValueError(f"Layer {layer} runtime support mismatch")

    with (shard / TASK_PAYLOAD_PATH).open() as handle:
        payload = json.load(handle)
    required_payload = {
        "task_payload_version", "task_id", "layer", "configuration_digest",
        "n_models", "n_stable_proteins", "pair_assignment_digests",
        "oof_accumulator", "sequence_subgroups", "spatial_subgroups",
    }
    if set(payload) != required_payload or (
        payload["task_payload_version"] != TASK_PAYLOAD_VERSION
        or int(payload["task_id"]) != int(layer)
        or int(payload["layer"]) != int(layer)
        or payload["configuration_digest"] != CONFIGURATION_DIGEST
        or int(payload["n_models"]) != len(expected_model_keys)
        or set(payload["pair_assignment_digests"]) != expected_id_set
    ):
        raise ValueError(f"Layer {layer} payload identity mismatch")
    for protein_id in expected_ids:
        if payload["pair_assignment_digests"][protein_id] != str(
            fold_bundle.pair_by_protein[protein_id].assignment_digest
        ):
            raise ValueError(f"Layer {layer} pair-assignment digest mismatch")
    oof = OOFComparisonAccumulator.from_payload(payload["oof_accumulator"])
    sequence = OOFSubgroupAccumulator.from_payload(payload["sequence_subgroups"])
    spatial = OOFSubgroupAccumulator.from_payload(payload["spatial_subgroups"])
    oof.require_complete()
    if (
        set(oof.expected_pairs_by_protein) != expected_id_set
        or set(sequence.protein_ids) != expected_id_set
        or set(spatial.protein_ids) != expected_id_set
        or int(payload["n_stable_proteins"]) != sum(oof.stable_fit_by_protein.values())
    ):
        raise ValueError(f"Layer {layer} accumulator membership mismatch")
    rebuilt_fold_rows = [
        build_fold_metrics_row(
            oof.subset(fold_bundle.protein_ids_for_outer_fold(outer_fold)),
            outer_fold=outer_fold, analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY, feature_dim=FEATURE_DIM, layer=layer,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            conditional_baseline_type=CONDITIONAL_BASELINE_TYPE,
        )
        for outer_fold in range(N_OUTER_FOLDS)
    ]
    observed_fold_rows = sorted(
        tables[FOLD_METRICS_FILENAME], key=lambda row: int(row["outer_fold"]),
    )
    for observed, rebuilt in zip(observed_fold_rows, rebuilt_fold_rows):
        for key in (
            "n_proteins", "n_expected_pairs", "n_evaluated_pairs", "pair_coverage",
            "pooled_r2", "pooled_mae", "pooled_rmse",
            "pooled_bias_pred_minus_true", "global_baseline_pooled_r2",
            "global_baseline_pooled_mae", "global_baseline_pooled_rmse",
            "global_baseline_pooled_bias_pred_minus_true",
            "conditional_baseline_pooled_r2", "conditional_baseline_pooled_mae",
            "conditional_baseline_pooled_rmse",
            "conditional_baseline_pooled_bias_pred_minus_true",
        ):
            if not np.isclose(
                _float(observed, key), float(rebuilt[key]),
                rtol=1e-12, atol=1e-12, equal_nan=True,
            ):
                raise ValueError(f"Layer {layer} fold reconstruction failed: {key}")
    subgroup_identity = dict(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM, layer=layer,
        evaluation_design=EVALUATION_DESIGN,
        oof_prediction_scope=OOF_PREDICTION_SCOPE,
    )
    for logical, observed_rows, rebuilt_rows in (
        (
            SUBGROUP_SEQUENCE_FILENAME, tables[SUBGROUP_SEQUENCE_FILENAME],
            sequence.rows(**subgroup_identity),
        ),
        (
            SUBGROUP_SPATIAL_FILENAME, tables[SUBGROUP_SPATIAL_FILENAME],
            spatial.rows(**subgroup_identity),
        ),
    ):
        for observed, rebuilt in zip(observed_rows, rebuilt_rows):
            if (
                observed["category_label"] != rebuilt["category_label"]
                or observed["support_status"] != rebuilt["support_status"]
            ):
                raise ValueError(f"Layer {layer} subgroup identity mismatch in {logical}")
            for key in (
                "n_proteins", "n_expected_pairs", "n_evaluated_pairs", "pair_coverage",
                "pooled_r2", "pooled_mae", "pooled_rmse",
                "pooled_bias_pred_minus_true",
            ):
                if not np.isclose(
                    _float(observed, key), float(rebuilt[key]),
                    rtol=1e-12, atol=1e-12, equal_nan=True,
                ):
                    raise ValueError(
                        f"Layer {layer} subgroup reconstruction failed: {logical} {key}"
                    )
    for row in per_rows:
        protein_id = row["protein_id"]
        metrics = oof.per_protein_model[protein_id].metrics()
        expected_status = (
            "complete" if oof.stable_fit_by_protein[protein_id]
            else "complete_from_flagged_fit"
        )
        if row["prediction_status"] != expected_status:
            raise ValueError(f"Layer {layer} stable-fit status mismatch for {protein_id}")
        for key in (
            "target_variance", "r2", "mae", "rmse", "bias_pred_minus_true",
            "mean_true_distance", "mean_predicted_distance",
        ):
            if not np.isclose(
                _float(row, key), float(metrics[key]),
                rtol=1e-12, atol=1e-12, equal_nan=True,
            ):
                raise ValueError(f"Layer {layer} OOF reconstruction failed: {protein_id} {key}")

    responsibilities = [
        (protein_id, pair_fold)
        for protein_id in expected_ids for pair_fold in range(N_PAIR_FOLDS)
    ]
    artifact = load_coefficient_shard(
        shard / TASK_MODEL_PATH, expected_layer=layer,
        expected_responsibilities=responsibilities,
    )
    artifact_statuses = np.asarray(artifact["fit_status"]).astype(str)
    if any(
        artifact_statuses[index] != status_by_key[(protein_id, str(pair_fold))]["fit_status"]
        for index, (protein_id, pair_fold) in enumerate(responsibilities)
    ):
        raise ValueError(f"Layer {layer} artifact/model-status mismatch")
    stability_by_key = {
        (row["protein_id"], int(row["feature_index"])): row
        for row in stability_rows
    }
    coefficients = np.asarray(artifact["coefficient_standardized"], dtype=np.float64)
    for protein_index, protein_id in enumerate(expected_ids):
        start = protein_index * N_PAIR_FOLDS
        rebuilt_rows = build_model_stability_rows(
            coefficients[start:start + N_PAIR_FOLDS],
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            layer=layer, feature_names=FEATURE_NAMES, protein_id=protein_id,
        )
        for rebuilt in rebuilt_rows:
            observed = stability_by_key[(protein_id, int(rebuilt["feature_index"]))]
            for key in (
                "coefficient_standardized_mean", "coefficient_standardized_std",
                "coefficient_standardized_min", "coefficient_standardized_max",
                "sign_consistency_fraction",
            ):
                if not np.isclose(
                    _float(observed, key), float(rebuilt[key]),
                    rtol=1e-12, atol=1e-12, equal_nan=True,
                ):
                    raise ValueError(
                        f"Layer {layer} stability reconstruction failed: {protein_id} {key}"
                    )
    tables.pop(MODEL_STABILITY_FILENAME)
    return {
        "tables": tables, "payload": payload, "oof": oof,
        "sequence": sequence, "spatial": spatial,
        "artifact_path": shard / TASK_MODEL_PATH,
        "stability_path": shard / TASK_TABLES[MODEL_STABILITY_FILENAME],
        "input_identity_digest": invariants["input_identity_digest"],
    }


def aggregate_per_protein_ols(
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
        raise ValueError("Production per-protein OLS aggregation requires all 48 layers")
    validated = {}
    for layer in requested_layers:
        log(f"validating per-protein OLS layer {layer}")
        validated[layer] = validate_task_shard(destination, layer, bundle)

    consolidated = {
        PER_PROTEIN_FILENAME: [], POOLED_METRICS_FILENAME: [],
        FOLD_METRICS_FILENAME: [], FOLD_SUMMARY_FILENAME: [],
        TRAINING_METRICS_FILENAME: [], STABLE_FIT_METRICS_FILENAME: [],
        MODEL_STATUS_FILENAME: [], RUNTIME_FILENAME: [],
        GEOMETRY_METRICS_FILENAME: [], GEOMETRY_SUMMARY_FILENAME: [],
        SUBGROUP_SEQUENCE_FILENAME: [],
        SUBGROUP_SPATIAL_FILENAME: [],
    }
    for layer in requested_layers:
        result = validated[layer]
        tables = result["tables"]
        oof = result["oof"]
        consolidated[PER_PROTEIN_FILENAME].extend(tables[PER_PROTEIN_FILENAME])
        consolidated[TRAINING_METRICS_FILENAME].extend(tables[TRAINING_METRICS_FILENAME])
        consolidated[MODEL_STATUS_FILENAME].extend(tables[MODEL_STATUS_FILENAME])
        consolidated[RUNTIME_FILENAME].extend(tables[RUNTIME_FILENAME])
        consolidated[GEOMETRY_METRICS_FILENAME].extend(tables[GEOMETRY_METRICS_FILENAME])
        consolidated[SUBGROUP_SEQUENCE_FILENAME].extend(tables[SUBGROUP_SEQUENCE_FILENAME])
        consolidated[SUBGROUP_SPATIAL_FILENAME].extend(tables[SUBGROUP_SPATIAL_FILENAME])
        consolidated[POOLED_METRICS_FILENAME].append(build_pooled_metrics_row(
            oof, analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=layer,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            conditional_baseline_type=CONDITIONAL_BASELINE_TYPE,
        ))
        fold_rows = [
            build_fold_metrics_row(
                oof.subset(bundle.protein_ids_for_outer_fold(outer_fold)),
                outer_fold=outer_fold, analysis_name=ANALYSIS_NAME,
                model_family=MODEL_FAMILY, feature_dim=FEATURE_DIM, layer=layer,
                evaluation_design=EVALUATION_DESIGN,
                oof_prediction_scope=OOF_PREDICTION_SCOPE,
                conditional_baseline_type=CONDITIONAL_BASELINE_TYPE,
            )
            for outer_fold in range(N_OUTER_FOLDS)
        ]
        consolidated[FOLD_METRICS_FILENAME].extend(fold_rows)
        consolidated[FOLD_SUMMARY_FILENAME].extend(build_fold_summary_rows(fold_rows))
        consolidated[STABLE_FIT_METRICS_FILENAME].append(build_stable_fit_metrics_row(
            oof, analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=layer,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            conditional_baseline_type=CONDITIONAL_BASELINE_TYPE,
        ))
        consolidated[GEOMETRY_SUMMARY_FILENAME].extend(
            summarize_geometry_rows(tables[GEOMETRY_METRICS_FILENAME])
        )

    n_proteins = len(bundle.eligible_protein_ids)
    expected_counts = {
        PER_PROTEIN_FILENAME: n_proteins * len(requested_layers),
        POOLED_METRICS_FILENAME: len(requested_layers),
        FOLD_METRICS_FILENAME: N_OUTER_FOLDS * len(requested_layers),
        FOLD_SUMMARY_FILENAME: 12 * len(requested_layers),
        TRAINING_METRICS_FILENAME: n_proteins * N_PAIR_FOLDS * len(requested_layers),
        STABLE_FIT_METRICS_FILENAME: len(requested_layers),
        MODEL_STATUS_FILENAME: n_proteins * N_PAIR_FOLDS * len(requested_layers),
        RUNTIME_FILENAME: len(requested_layers),
        GEOMETRY_METRICS_FILENAME: n_proteins * len(requested_layers),
        GEOMETRY_SUMMARY_FILENAME: 14 * len(requested_layers),
        SUBGROUP_SEQUENCE_FILENAME: 6 * len(requested_layers),
        SUBGROUP_SPATIAL_FILENAME: 5 * len(requested_layers),
    }
    for filename, count in expected_counts.items():
        if len(consolidated[filename]) != count:
            raise ValueError(f"Consolidated row count mismatch for {filename}")
    unique_keys = {
        PER_PROTEIN_FILENAME: ("analysis_name", "layer", "protein_id"),
        POOLED_METRICS_FILENAME: ("analysis_name", "layer"),
        FOLD_METRICS_FILENAME: ("analysis_name", "layer", "outer_fold"),
        FOLD_SUMMARY_FILENAME: ("analysis_name", "layer", "fold_metric"),
        TRAINING_METRICS_FILENAME: ("analysis_name", "layer", "protein_id", "pair_fold"),
        STABLE_FIT_METRICS_FILENAME: ("analysis_name", "layer"),
        MODEL_STATUS_FILENAME: ("analysis_name", "layer", "protein_id", "pair_fold"),
        RUNTIME_FILENAME: ("analysis_name", "task_id"),
        GEOMETRY_METRICS_FILENAME: ("analysis_name", "layer", "protein_id"),
        GEOMETRY_SUMMARY_FILENAME: ("analysis_name", "layer", "geometry_metric"),
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
        _concatenate_stability_shards(
            staging / MODEL_STABILITY_FILENAME,
            [validated[layer]["stability_path"] for layer in requested_layers],
            expected_rows=n_proteins * FEATURE_DIM * len(requested_layers),
        )
        replay_rows = []
        for layer in requested_layers:
            result = validated[layer]
            relative = Path("model_artifacts") / "coefficient_shards" / f"layer_{layer:02d}.npz"
            target = staging / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(result["artifact_path"], target)
            responsibilities = [
                (protein_id, pair_fold)
                for protein_id in sorted(bundle.eligible_protein_ids)
                for pair_fold in range(N_PAIR_FOLDS)
            ]
            artifact = load_coefficient_shard(
                target, expected_layer=layer,
                expected_responsibilities=responsibilities,
            )
            artifact_digest = sha256_file(target)
            scaler_digests = np.asarray(artifact["scaler_digest"]).astype(str)
            for index, (protein_id, pair_fold) in enumerate(responsibilities):
                replay_rows.append(build_replay_index_row(
                    family_root=staging, artifact_path=relative,
                    analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
                    layer=layer, protein_id=protein_id, pair_fold=pair_fold,
                    artifact_type="ols_coefficients_scaler_layer_shard_npz",
                    configuration_digest=CONFIGURATION_DIGEST,
                    scaler_digest=scaler_digests[index],
                    artifact_sha256=artifact_digest, status="complete",
                ))
        validate_replay_index(
            replay_rows, family_root=staging,
            expected_count=n_proteins * N_PAIR_FOLDS * len(requested_layers),
        )
        write_v2_csv(staging / REPLAY_INDEX_FILENAME, REPLAY_INDEX_FILENAME, replay_rows)
        task_ids = sorted(requested_layers)
        manifest = build_run_manifest(
            fold_bundle=bundle, analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY, feature_dim=FEATURE_DIM,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            requested_task_ids=task_ids, completed_task_ids=task_ids,
            configuration=CONFIGURATION,
            expected_replay_artifacts=n_proteins * N_PAIR_FOLDS * len(requested_layers),
            versions=version_payload(), code_revision=code_revision(),
            run_status="complete",
        )
        manifest.update({
            "requested_layers": list(requested_layers),
            "task_mapping": "layer_task_id=layer",
            "task_input_identity_digests": {
                str(layer): validated[layer]["input_identity_digest"]
                for layer in requested_layers
            },
            "coefficient_artifact_layout":
                "model_artifacts/coefficient_shards/layer_<layer>.npz",
            "coefficient_shard_indexing":
                "locate unique protein_id,pair_fold arrays inside each layer archive",
            "outer_fold_role": "reporting_only_does_not_affect_within_protein_fit",
            "raw_pair_predictions_saved": False,
        })
        atomic_write_json(staging / "run_manifest.json", manifest)
        _publish_staging(staging, destination)
    finally:
        shutil.rmtree(staging, ignore_errors=True)
    log(f"published {len(requested_layers)} per-protein OLS layers to {destination}")
    return destination


if __name__ == "__main__":
    aggregate_per_protein_ols()

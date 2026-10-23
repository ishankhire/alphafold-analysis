#!/usr/bin/env python3
"""Protocol-2.0 fivefold sequence-gated base256 OLS runner primitives."""

from __future__ import annotations

import gc
import math
import os
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
for _path in (CLUSTER_DIR, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cross_validation_common import (  # noqa: E402
    CV_PROTOCOL_VERSION,
    EVALUATION_SCOPE,
    FOLD_METRICS_FILENAME,
    GEOMETRY_METRICS_FILENAME,
    MODEL_STATUS_FILENAME,
    OOFComparisonAccumulator,
    OOFSubgroupAccumulator,
    PER_PROTEIN_FILENAME,
    RUNTIME_FILENAME,
    SUBGROUP_SEQUENCE_FILENAME,
    SUBGROUP_SPATIAL_FILENAME,
    TRAINING_METRICS_FILENAME,
    FoldLayerTask,
    build_fold_metrics_row,
    build_geometry_row,
    build_model_status_row,
    build_per_protein_row,
    build_runtime_row,
    build_training_metrics_row,
    commit_checkpoint,
    fit_train_only_baseline,
    fivefold_result_root,
    load_cv_folds,
    map_fold_layer_task,
    prepare_training_fold,
    semantic_digest,
    utc_now,
    validate_resume_checkpoint,
    write_v2_csv,
    write_v2_npz,
)
from regression_output_helpers import atomic_write_json  # noqa: E402
from regression_protocol import (  # noqa: E402
    FEATURE_DIM,
    FEATURE_NAMES,
    LAYERS,
    N_OUTER_FOLDS,
    SEQUENCE_BIN_LABELS,
    SPATIAL_DISTANCE_BIN_LABELS,
    sequence_separation_bin_codes,
    spatial_distance_bin_codes,
)
from shared_ols_base256_common import (  # noqa: E402
    ProteinLayerRecord,
    code_revision,
    load_feature_block,
    load_target_records,
    peak_rss_mb,
    read_csv_strict,
    resolve_task_inputs,
    version_payload,
)


ANALYSIS_NAME = "sequence_gated_ols_base256"
MODEL_FAMILY = "gated_ols"
FAMILY_DIR_NAME = "sequence_gated_ols_base256"
EVALUATION_DESIGN = "protein_outer_5fold"
OOF_PREDICTION_SCOPE = "heldout_proteins_all_pairs_recombined_sequence_gates"
CONDITIONAL_BASELINE_TYPE = "sequence_gate_fold_training_mean"
CONDITION_LIMIT = 1e12
MIN_TRAIN_ROWS = FEATURE_DIM + 1
MIN_TRAIN_PROTEINS = 5
TASK_PAYLOAD_VERSION = "sequence_gated_ols_base256_task_payload_v1"
ARTIFACT_VERSION = "sequence_gated_ols_base256_coefficients_v1"

CONFIGURATION = {
    "analysis_name": ANALYSIS_NAME,
    "model_family": MODEL_FAMILY,
    "cv_protocol_version": CV_PROTOCOL_VERSION,
    "evaluation_design": EVALUATION_DESIGN,
    "evaluation_scope": EVALUATION_SCOPE,
    "oof_prediction_scope": OOF_PREDICTION_SCOPE,
    "outer_folds": N_OUTER_FOLDS,
    "layers": list(LAYERS),
    "sequence_gates": list(SEQUENCE_BIN_LABELS),
    "feature_mode": "directional_base256",
    "feature_dim": FEATURE_DIM,
    "feature_order": list(FEATURE_NAMES),
    "feature_cache_dtype": "float32",
    "fit_dtype": "float64",
    "target": "final_predicted_pdb_absolute_ca_distance_angstrom",
    "pair_order": "np.tril_indices(n_residues, k=-1)",
    "estimator": "sklearn.linear_model.LinearRegression",
    "fit_intercept": True,
    "copy_X": False,
    "training_pairs": "all_gate_pairs_from_four_outer_training_folds",
    "training_weighting": "equal_total_weight_per_contributing_training_protein_within_gate",
    "scaler": "gate_and_fold_training_only_protein_equal_float64_moments",
    "global_baseline": "fold_training_only_protein_equal_target_mean",
    "conditional_baseline": CONDITIONAL_BASELINE_TYPE,
    "minimum_training_rows_per_gate": MIN_TRAIN_ROWS,
    "minimum_training_proteins_per_gate": MIN_TRAIN_PROTEINS,
    "invalid_gate_policy": "fail_fold_layer_shard_without_fallback",
    "condition_number_flag_threshold": CONDITION_LIMIT,
    "stable_fit_policy": "all_six_gate_fits_rank_256_and_finite_condition_at_most_1e12",
    "flagged_fit_primary_policy": "include_finite_predictions_with_explicit_status",
    "post_cv_all_data_refit": False,
    "raw_pair_predictions_saved": False,
}
CONFIGURATION_DIGEST = semantic_digest(CONFIGURATION)

TASK_TABLES = {
    PER_PROTEIN_FILENAME: "per_protein.csv",
    FOLD_METRICS_FILENAME: "fold_metrics.csv",
    TRAINING_METRICS_FILENAME: "training_metrics.csv",
    MODEL_STATUS_FILENAME: "model_status.csv",
    RUNTIME_FILENAME: "runtime.csv",
    GEOMETRY_METRICS_FILENAME: "geometry_metrics.csv",
    SUBGROUP_SEQUENCE_FILENAME: "subgroup_metrics/by_sequence_separation.csv",
    SUBGROUP_SPATIAL_FILENAME: "subgroup_metrics/by_spatial_distance.csv",
}
TASK_PAYLOAD_PATH = "task_payload.json"
TASK_CHECKPOINT_FILENAME = "checkpoint.json"


from canonical_storage import assert_writable


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def family_output_root(csv_root: str | Path | None = None) -> Path:
    return fivefold_result_root(csv_root) / FAMILY_DIR_NAME


def task_shard_path(root: str | Path, task: FoldLayerTask) -> Path:
    return Path(root) / "layer_shards" / f"fold_{task.outer_fold}_layer_{task.layer:02d}"


def gate_slug(sequence_gate: str) -> str:
    label = str(sequence_gate)
    if label not in SEQUENCE_BIN_LABELS:
        raise ValueError(f"Unknown sequence gate: {label}")
    return label.replace("+", "_plus").replace("-", "_to_")


def task_model_path(sequence_gate: str) -> str:
    return f"model_artifacts/coefficients/gate_{gate_slug(sequence_gate)}.npz"


def expected_task_artifacts() -> dict[str, str]:
    artifacts = {
        **{logical: relative for logical, relative in TASK_TABLES.items()},
        "task_payload": TASK_PAYLOAD_PATH,
    }
    artifacts.update({
        f"coefficient_model_{gate_slug(label)}": task_model_path(label)
        for label in SEQUENCE_BIN_LABELS
    })
    return artifacts


@dataclass(frozen=True)
class SequenceGateOLSFit:
    sequence_gate: str
    coefficient_standardized: np.ndarray
    coefficient_raw_scale: np.ndarray
    intercept_standardized: float
    intercept_raw_scale: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    scaler_digest: str
    conditional_baseline_value: float
    conditional_baseline_digest: str
    global_baseline_value: float
    global_baseline_digest: str
    design_rank: int
    largest_singular_value: float
    smallest_singular_value: float
    condition_number: float
    fit_status: str
    support_flags: str
    n_training_proteins: int
    n_training_pairs: int

    @property
    def stable(self) -> bool:
        return self.fit_status == "fitted"


@dataclass(frozen=True)
class GateFitResult:
    fit: SequenceGateOLSFit
    training_targets: np.ndarray
    training_predictions: np.ndarray


def validated_sequence_gate_codes(record: ProteinLayerRecord) -> np.ndarray:
    codes = sequence_separation_bin_codes(
        record.pair_i, record.pair_j, chain_ids=record.chain_ids,
    )
    if codes.shape != record.targets.shape:
        raise ValueError(f"Sequence-gate codes do not match targets for {record.protein_id}")
    undefined = int(np.count_nonzero(codes < 0))
    if undefined:
        raise ValueError(
            f"Sequence-gated OLS requires intrachain coverage; {record.protein_id} "
            f"has {undefined} pairs with undefined sequence separation"
        )
    if set(np.unique(codes)) - set(range(len(SEQUENCE_BIN_LABELS))):
        raise ValueError(f"Invalid sequence-gate code for {record.protein_id}")
    return codes


def _fit_status(n_rows: int, rank: int, condition: float) -> tuple[str, str]:
    flags = []
    if n_rows < MIN_TRAIN_ROWS:
        flags.append("underdetermined")
    if rank < FEATURE_DIM:
        flags.append("rank_deficient")
    if not math.isfinite(condition) or condition > CONDITION_LIMIT:
        flags.append("ill_conditioned")
    if "underdetermined" in flags:
        status = "fitted_underdetermined"
    elif "rank_deficient" in flags:
        status = "fitted_rank_deficient"
    elif "ill_conditioned" in flags:
        status = "fitted_ill_conditioned"
    else:
        status = "fitted"
    return status, ";".join(flags)


def fit_sequence_gate_ols(
    feature_blocks: Mapping[str, np.ndarray],
    target_blocks: Mapping[str, np.ndarray],
    code_blocks: Mapping[str, np.ndarray],
    *,
    sequence_gate: str,
    global_baseline_value: float,
    global_baseline_digest: str,
) -> GateFitResult:
    """Fit one isolated gate with gate-specific train-only preprocessing."""
    label = str(sequence_gate)
    if label not in SEQUENCE_BIN_LABELS:
        raise ValueError(f"Unknown sequence gate: {label}")
    gate_code = SEQUENCE_BIN_LABELS.index(label)
    if set(feature_blocks) != set(target_blocks) or set(feature_blocks) != set(code_blocks):
        raise ValueError("Gate features, targets, and codes must cover identical proteins")
    gated_features: dict[str, np.ndarray] = {}
    gated_targets: dict[str, np.ndarray] = {}
    for protein_id in sorted(feature_blocks):
        features = np.asarray(feature_blocks[protein_id])
        targets = np.asarray(target_blocks[protein_id], dtype=np.float64)
        codes = np.asarray(code_blocks[protein_id])
        if features.shape != (targets.size, FEATURE_DIM) or codes.shape != targets.shape:
            raise ValueError(f"Gate block shape mismatch for {protein_id}")
        mask = codes == gate_code
        if np.any(mask):
            gated_features[protein_id] = features[mask]
            gated_targets[protein_id] = targets[mask]
    n_training_proteins = len(gated_features)
    n_training_pairs = sum(values.size for values in gated_targets.values())
    if n_training_proteins < MIN_TRAIN_PROTEINS or n_training_pairs < MIN_TRAIN_ROWS:
        raise ValueError(
            f"Invalid sequence gate {label}: {n_training_pairs} rows from "
            f"{n_training_proteins} proteins; requires at least "
            f"{MIN_TRAIN_ROWS} rows and {MIN_TRAIN_PROTEINS} proteins"
        )
    prepared = prepare_training_fold(gated_features, gated_targets)
    model = LinearRegression(copy_X=False, fit_intercept=True)
    model.fit(
        prepared.standardized_features,
        prepared.targets,
        sample_weight=prepared.sample_weights,
    )
    coefficient_standardized = np.asarray(model.coef_, dtype=np.float64).copy()
    intercept_standardized = float(model.intercept_)
    singular = np.asarray(getattr(model, "singular_", []), dtype=np.float64)
    rank = int(getattr(model, "rank_", -1))
    largest = float(np.max(singular)) if singular.size else float("nan")
    smallest = float(np.min(singular)) if singular.size else float("nan")
    condition = largest / smallest if singular.size and smallest > 0 else float("inf")
    status, flags = _fit_status(prepared.targets.size, rank, condition)
    coefficient_raw = coefficient_standardized / prepared.standardizer.feature_scale
    intercept_raw = intercept_standardized - float(
        np.dot(coefficient_raw, prepared.standardizer.feature_mean)
    )
    fit = SequenceGateOLSFit(
        sequence_gate=label,
        coefficient_standardized=coefficient_standardized,
        coefficient_raw_scale=np.asarray(coefficient_raw, dtype=np.float64),
        intercept_standardized=intercept_standardized,
        intercept_raw_scale=intercept_raw,
        feature_mean=prepared.standardizer.feature_mean.copy(),
        feature_scale=prepared.standardizer.feature_scale.copy(),
        scaler_digest=prepared.standardizer.scaler_digest,
        conditional_baseline_value=prepared.global_baseline.value,
        conditional_baseline_digest=prepared.global_baseline.baseline_digest,
        global_baseline_value=float(global_baseline_value),
        global_baseline_digest=str(global_baseline_digest),
        design_rank=rank,
        largest_singular_value=largest,
        smallest_singular_value=smallest,
        condition_number=condition,
        fit_status=status,
        support_flags=flags,
        n_training_proteins=n_training_proteins,
        n_training_pairs=n_training_pairs,
    )
    training_predictions = np.concatenate([
        predict_sequence_gate_ols(fit, gated_features[protein_id])
        for protein_id in sorted(gated_features)
    ])
    return GateFitResult(fit, prepared.targets.copy(), training_predictions)


def predict_sequence_gate_ols(fit: SequenceGateOLSFit, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM or not np.all(np.isfinite(values)):
        raise ValueError("Sequence-gate prediction features are invalid")
    predictions = fit.intercept_standardized + (
        (values - fit.feature_mean) / fit.feature_scale
    ) @ fit.coefficient_standardized
    result = np.asarray(predictions, dtype=np.float64)
    if not np.all(np.isfinite(result)):
        raise ValueError(f"Sequence gate {fit.sequence_gate} produced nonfinite predictions")
    return result


def recombine_sequence_gate_predictions(
    fits: Mapping[str, SequenceGateOLSFit],
    features: np.ndarray,
    codes: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    if set(fits) != set(SEQUENCE_BIN_LABELS):
        raise ValueError("Recombination requires exactly the six fitted sequence gates")
    values = np.asarray(features)
    gate_codes = np.asarray(codes)
    if values.shape != (gate_codes.size, FEATURE_DIM):
        raise ValueError("Recombination feature/code shapes do not match")
    predictions = np.full(gate_codes.size, np.nan, dtype=np.float64)
    conditional = np.full(gate_codes.size, np.nan, dtype=np.float64)
    assignments = np.zeros(gate_codes.size, dtype=np.int8)
    for code, label in enumerate(SEQUENCE_BIN_LABELS):
        mask = gate_codes == code
        if np.any(mask):
            fit = fits[label]
            predictions[mask] = predict_sequence_gate_ols(fit, values[mask])
            conditional[mask] = fit.conditional_baseline_value
            assignments[mask] += 1
    if not np.all(assignments == 1) or not np.all(np.isfinite(predictions)):
        raise ValueError("Sequence-gate recombination lacks exact-once complete prediction coverage")
    if not np.all(np.isfinite(conditional)):
        raise ValueError("Sequence-gate conditional baseline coverage is incomplete")
    return predictions, conditional


def write_coefficient_artifact(path: Path, fit: SequenceGateOLSFit, task: FoldLayerTask) -> None:
    write_v2_npz(
        path,
        artifact_version=np.asarray(ARTIFACT_VERSION),
        analysis_name=np.asarray(ANALYSIS_NAME),
        model_family=np.asarray(MODEL_FAMILY),
        layer=np.asarray(task.layer, dtype=np.int16),
        outer_fold=np.asarray(task.outer_fold, dtype=np.int16),
        sequence_gate=np.asarray(fit.sequence_gate),
        coefficient_standardized=fit.coefficient_standardized,
        coefficient_raw_scale=fit.coefficient_raw_scale,
        intercept_standardized=np.asarray(fit.intercept_standardized, dtype=np.float64),
        intercept_raw_scale=np.asarray(fit.intercept_raw_scale, dtype=np.float64),
        feature_mean=fit.feature_mean,
        feature_scale=fit.feature_scale,
        scaler_digest=np.asarray(fit.scaler_digest),
        conditional_baseline_value=np.asarray(fit.conditional_baseline_value, dtype=np.float64),
        conditional_baseline_digest=np.asarray(fit.conditional_baseline_digest),
        global_baseline_value=np.asarray(fit.global_baseline_value, dtype=np.float64),
        global_baseline_digest=np.asarray(fit.global_baseline_digest),
        configuration_digest=np.asarray(CONFIGURATION_DIGEST),
        fit_status=np.asarray(fit.fit_status),
    )


def load_coefficient_artifact(
    path: str | Path,
    *,
    expected_task: FoldLayerTask | None = None,
    expected_sequence_gate: str | None = None,
    expected_scaler_digest: str | None = None,
) -> dict[str, object]:
    expected_keys = {
        "artifact_version", "analysis_name", "model_family", "layer", "outer_fold",
        "sequence_gate", "coefficient_standardized", "coefficient_raw_scale",
        "intercept_standardized", "intercept_raw_scale", "feature_mean", "feature_scale",
        "scaler_digest", "conditional_baseline_value", "conditional_baseline_digest",
        "global_baseline_value", "global_baseline_digest", "configuration_digest", "fit_status",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(f"Coefficient artifact schema mismatch: {path}")
        result = {name: archive[name].copy() for name in archive.files}
    scalar_names = expected_keys - {
        "coefficient_standardized", "coefficient_raw_scale", "feature_mean", "feature_scale",
    }
    scalars = {name: np.asarray(result[name]).item() for name in scalar_names}
    if (
        scalars["artifact_version"] != ARTIFACT_VERSION
        or scalars["analysis_name"] != ANALYSIS_NAME
        or scalars["model_family"] != MODEL_FAMILY
        or scalars["configuration_digest"] != CONFIGURATION_DIGEST
        or scalars["sequence_gate"] not in SEQUENCE_BIN_LABELS
    ):
        raise ValueError("Coefficient artifact identity mismatch")
    if expected_task is not None and (
        int(scalars["layer"]), int(scalars["outer_fold"])
    ) != (expected_task.layer, expected_task.outer_fold):
        raise ValueError("Coefficient artifact task identity mismatch")
    if expected_sequence_gate is not None and scalars["sequence_gate"] != expected_sequence_gate:
        raise ValueError("Coefficient artifact sequence-gate mismatch")
    if expected_scaler_digest is not None and scalars["scaler_digest"] != expected_scaler_digest:
        raise ValueError("Coefficient artifact scaler digest mismatch")
    if any(not str(scalars[name]) for name in (
        "scaler_digest", "conditional_baseline_digest", "global_baseline_digest",
    )) or any(not math.isfinite(float(scalars[name])) for name in (
        "intercept_standardized", "intercept_raw_scale", "conditional_baseline_value",
        "global_baseline_value",
    )):
        raise ValueError("Coefficient artifact scalar state is invalid")
    arrays = [
        np.asarray(result[name], dtype=np.float64)
        for name in ("coefficient_standardized", "coefficient_raw_scale", "feature_mean", "feature_scale")
    ]
    if any(array.shape != (FEATURE_DIM,) or not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Coefficient artifact arrays are invalid")
    coefficient_standardized, coefficient_raw, feature_mean, feature_scale = arrays
    if np.any(feature_scale <= 0) or not np.allclose(
        coefficient_raw, coefficient_standardized / feature_scale,
        rtol=1e-12, atol=1e-12,
    ):
        raise ValueError("Raw and standardized coefficient artifacts disagree")
    expected_intercept = float(scalars["intercept_standardized"]) - float(
        np.dot(coefficient_raw, feature_mean)
    )
    if not math.isclose(
        float(scalars["intercept_raw_scale"]), expected_intercept,
        rel_tol=1e-12, abs_tol=1e-12,
    ):
        raise ValueError("Raw and standardized intercept artifacts disagree")
    return {**result, **scalars}


def task_invariants(
    task: FoldLayerTask,
    *,
    fold_digests: Mapping[str, object],
    input_digest: str,
) -> dict[str, object]:
    return {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "task_id": task.task_id,
        "outer_fold": task.outer_fold,
        "layer": task.layer,
        "configuration_digest": CONFIGURATION_DIGEST,
        "cohort_digest": fold_digests["cohort_digest"],
        "outer_fold_digest": fold_digests["outer_fold_digest"],
        "pair_fold_manifest_digest": fold_digests["pair_fold_manifest_digest"],
        "input_identity_digest": str(input_digest),
    }


def execute_fold_layer_task(
    *,
    task: FoldLayerTask,
    records: Sequence[ProteinLayerRecord],
    training_protein_ids: Sequence[str],
    heldout_protein_ids: Sequence[str],
    fold_digests: Mapping[str, object],
    input_digest: str,
    root: str | Path,
    feature_loader: Callable[[ProteinLayerRecord], np.ndarray] = load_feature_block,
) -> Path:
    """Fit six isolated gates, recombine held-out predictions, and checkpoint."""
    assert_writable(root)
    started = time.monotonic()
    shard = task_shard_path(root, task)
    invariants = task_invariants(task, fold_digests=fold_digests, input_digest=input_digest)
    checkpoint = shard / TASK_CHECKPOINT_FILENAME
    artifact_names = set(expected_task_artifacts())
    if checkpoint.is_file():
        validate_resume_checkpoint(
            checkpoint, expected_invariants=invariants,
            expected_artifact_names=artifact_names,
        )
        log(f"task {task.task_id} fold {task.outer_fold} layer {task.layer}: valid checkpoint; skipping")
        return shard

    by_id = {record.protein_id: record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Task records contain duplicate protein IDs")
    training_ids = tuple(sorted(map(str, training_protein_ids)))
    heldout_ids = tuple(sorted(map(str, heldout_protein_ids)))
    if (
        not training_ids or not heldout_ids
        or set(training_ids) & set(heldout_ids)
        or set(training_ids) | set(heldout_ids) != set(by_id)
    ):
        raise ValueError("Training and held-out protein IDs must be disjoint and exhaustive")
    if any(by_id[protein_id].outer_fold != task.outer_fold for protein_id in heldout_ids):
        raise ValueError("Held-out record has the wrong outer-fold label")
    if any(by_id[protein_id].outer_fold == task.outer_fold for protein_id in training_ids):
        raise ValueError("Training record belongs to the held-out outer fold")

    log(f"task {task.task_id}: loading {len(training_ids)} training feature blocks")
    feature_blocks: dict[str, np.ndarray] = {}
    target_blocks: dict[str, np.ndarray] = {}
    code_blocks: dict[str, np.ndarray] = {}
    for index, protein_id in enumerate(training_ids, start=1):
        record = by_id[protein_id]
        feature_blocks[protein_id] = feature_loader(record)
        target_blocks[protein_id] = record.targets
        code_blocks[protein_id] = validated_sequence_gate_codes(record)
        if index % 25 == 0 or index == len(training_ids):
            log(f"task {task.task_id}: cached {index}/{len(training_ids)} training proteins")
    global_baseline = fit_train_only_baseline(target_blocks)
    fits: dict[str, SequenceGateOLSFit] = {}
    training_rows = []
    status_rows = []
    for gate_number, label in enumerate(SEQUENCE_BIN_LABELS, start=1):
        log(f"task {task.task_id}: fitting gate {gate_number}/6 ({label})")
        result = fit_sequence_gate_ols(
            feature_blocks, target_blocks, code_blocks,
            sequence_gate=label,
            global_baseline_value=global_baseline.value,
            global_baseline_digest=global_baseline.baseline_digest,
        )
        fit = result.fit
        fits[label] = fit
        status_rows.append(build_model_status_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=task.layer, outer_fold=task.outer_fold,
            sequence_gate=label, fit_status=fit.fit_status,
            support_flags=fit.support_flags,
            n_training_proteins=fit.n_training_proteins,
            n_training_pairs=fit.n_training_pairs, design_rank=fit.design_rank,
            largest_singular_value=fit.largest_singular_value,
            smallest_singular_value=fit.smallest_singular_value,
            condition_number=fit.condition_number,
        ))
        training_rows.append(build_training_metrics_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=task.layer,
            evaluation_design=EVALUATION_DESIGN,
            training_scope="gate_pairs_four_outer_training_folds",
            y_true=result.training_targets,
            predictions=result.training_predictions,
            fit_status=fit.fit_status, support_flags=fit.support_flags,
            outer_fold=task.outer_fold, sequence_gate=label,
            n_training_proteins=fit.n_training_proteins,
            design_rank=fit.design_rank,
            largest_singular_value=fit.largest_singular_value,
            smallest_singular_value=fit.smallest_singular_value,
            condition_number=fit.condition_number,
        ))
    del feature_blocks, target_blocks, code_blocks
    gc.collect()

    all_gates_stable = all(fit.stable for fit in fits.values())
    oof = OOFComparisonAccumulator()
    sequence = OOFSubgroupAccumulator("sequence_separation", SEQUENCE_BIN_LABELS)
    spatial = OOFSubgroupAccumulator("spatial_distance", SPATIAL_DISTANCE_BIN_LABELS)
    per_rows = []
    geometry_rows = []
    log(f"task {task.task_id}: streaming {len(heldout_ids)} held-out proteins")
    for index, protein_id in enumerate(heldout_ids, start=1):
        record = by_id[protein_id]
        codes = validated_sequence_gate_codes(record)
        features = feature_loader(record)
        predictions, conditional_predictions = recombine_sequence_gate_predictions(
            fits, features, codes,
        )
        del features
        prediction_status = "complete" if all_gates_stable else "complete_from_flagged_fit"
        per_rows.append(build_per_protein_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=task.layer, protein_id=protein_id,
            source=record.source, outer_fold=task.outer_fold,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            n_residues=record.n_residues, y_true=record.targets,
            predictions=predictions, prediction_status=prediction_status,
        ))
        oof.update_protein(
            protein_id, task.outer_fold, record.targets, predictions,
            global_baseline.value,
            conditional_baseline_predictions=conditional_predictions,
            stable_fit=all_gates_stable,
        )
        sequence.update_protein(protein_id, record.targets, predictions, codes)
        spatial.update_protein(
            protein_id, record.targets, predictions,
            spatial_distance_bin_codes(record.targets),
        )
        geometry_rows.append(build_geometry_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=task.layer, protein_id=protein_id,
            source=record.source, outer_fold=task.outer_fold,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            n_residues=record.n_residues, true_coordinates=record.coordinates,
            predictions=predictions,
        ))
        if index % 20 == 0 or index == len(heldout_ids):
            log(f"task {task.task_id}: evaluated {index}/{len(heldout_ids)} held-out proteins")
    oof.require_complete()
    failed_geometry = [
        row["protein_id"] for row in geometry_rows
        if row["geometry_status"] != "complete"
    ]
    if failed_geometry:
        raise RuntimeError(f"Task {task.task_id} geometry is incomplete for {failed_geometry[:5]}")

    fold_row = build_fold_metrics_row(
        oof, outer_fold=task.outer_fold, analysis_name=ANALYSIS_NAME,
        model_family=MODEL_FAMILY, feature_dim=FEATURE_DIM, layer=task.layer,
        evaluation_design=EVALUATION_DESIGN,
        oof_prediction_scope=OOF_PREDICTION_SCOPE,
        conditional_baseline_type=CONDITIONAL_BASELINE_TYPE,
    )
    subgroup_identity = dict(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM, layer=task.layer,
        evaluation_design=EVALUATION_DESIGN,
        oof_prediction_scope=OOF_PREDICTION_SCOPE,
    )
    elapsed = time.monotonic() - started
    runtime_row = build_runtime_row(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        task_id=task.task_id, outer_fold=task.outer_fold, layer=task.layer,
        completion_status="complete", n_fitted_proteins=len(training_ids),
        n_evaluated_proteins=len(heldout_ids),
        n_fitted_pairs=sum(fit.n_training_pairs for fit in fits.values()),
        n_evaluated_pairs=oof.n_evaluated_pairs,
        elapsed_seconds=elapsed, peak_rss_mb=peak_rss_mb(),
    )
    tables = {
        PER_PROTEIN_FILENAME: per_rows,
        FOLD_METRICS_FILENAME: [fold_row],
        TRAINING_METRICS_FILENAME: training_rows,
        MODEL_STATUS_FILENAME: status_rows,
        RUNTIME_FILENAME: [runtime_row],
        GEOMETRY_METRICS_FILENAME: geometry_rows,
        SUBGROUP_SEQUENCE_FILENAME: sequence.rows(**subgroup_identity),
        SUBGROUP_SPATIAL_FILENAME: spatial.rows(**subgroup_identity),
    }
    for logical, relative in TASK_TABLES.items():
        write_v2_csv(shard / relative, logical, tables[logical])
    for label, fit in fits.items():
        write_coefficient_artifact(shard / task_model_path(label), fit, task)
    atomic_write_json(shard / TASK_PAYLOAD_PATH, {
        "task_payload_version": TASK_PAYLOAD_VERSION,
        "task_id": task.task_id,
        "outer_fold": task.outer_fold,
        "layer": task.layer,
        "configuration_digest": CONFIGURATION_DIGEST,
        "global_baseline_digest": global_baseline.baseline_digest,
        "all_gates_stable": all_gates_stable,
        "gate_metadata": {
            label: {
                "scaler_digest": fits[label].scaler_digest,
                "conditional_baseline_digest": fits[label].conditional_baseline_digest,
                "fit_status": fits[label].fit_status,
                "fit_stable": fits[label].stable,
            }
            for label in SEQUENCE_BIN_LABELS
        },
        "oof_accumulator": oof.to_payload(),
        "sequence_subgroups": sequence.to_payload(),
        "spatial_subgroups": spatial.to_payload(),
    })
    commit_checkpoint(
        checkpoint, invariants=invariants, artifacts=expected_task_artifacts(),
    )
    log(f"task {task.task_id}: complete in {elapsed:.1f}s")
    return shard


def run_fold_layer_task(
    task_id: int | None = None,
    *,
    root: str | Path | None = None,
) -> Path:
    assert_writable(family_output_root() if root is None else root)
    if task_id is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise RuntimeError("Provide task_id or set SLURM_ARRAY_TASK_ID")
        task_id = int(raw)
    task = map_fold_layer_task(task_id)
    fold_bundle = load_cv_folds()
    destination = family_output_root() if root is None else Path(root)
    paths, input_digest = resolve_task_inputs(fold_bundle, task.layer)
    invariants = task_invariants(
        task, fold_digests=fold_bundle.fold_digests, input_digest=input_digest,
    )
    checkpoint = task_shard_path(destination, task) / TASK_CHECKPOINT_FILENAME
    if checkpoint.is_file():
        validate_resume_checkpoint(
            checkpoint, expected_invariants=invariants,
            expected_artifact_names=set(expected_task_artifacts()),
        )
        log(f"task {task.task_id}: valid checkpoint; no inputs opened")
        return checkpoint.parent
    try:
        records = load_target_records(fold_bundle, paths)
        return execute_fold_layer_task(
            task=task, records=records,
            training_protein_ids=fold_bundle.training_protein_ids(task.outer_fold),
            heldout_protein_ids=fold_bundle.protein_ids_for_outer_fold(task.outer_fold),
            fold_digests=fold_bundle.fold_digests, input_digest=input_digest,
            root=destination,
        )
    except Exception as exc:
        atomic_write_json(destination / "failures" / f"task_{task.task_id:03d}.json", {
            "analysis_name": ANALYSIS_NAME, "task_id": task.task_id,
            "outer_fold": task.outer_fold, "layer": task.layer,
            "exception_type": type(exc).__name__, "message": str(exc),
            "configuration_digest": CONFIGURATION_DIGEST, "timestamp": utc_now(),
        })
        raise


__all__ = [
    "ANALYSIS_NAME", "ARTIFACT_VERSION", "CONDITION_LIMIT", "CONDITIONAL_BASELINE_TYPE",
    "CONFIGURATION", "CONFIGURATION_DIGEST", "EVALUATION_DESIGN", "FAMILY_DIR_NAME",
    "GateFitResult", "MIN_TRAIN_PROTEINS", "MIN_TRAIN_ROWS", "MODEL_FAMILY",
    "OOF_PREDICTION_SCOPE", "ProteinLayerRecord", "SequenceGateOLSFit",
    "TASK_CHECKPOINT_FILENAME", "TASK_PAYLOAD_PATH", "TASK_PAYLOAD_VERSION", "TASK_TABLES",
    "code_revision", "execute_fold_layer_task", "expected_task_artifacts",
    "family_output_root", "fit_sequence_gate_ols", "gate_slug", "load_coefficient_artifact",
    "load_feature_block", "load_target_records", "log", "peak_rss_mb",
    "predict_sequence_gate_ols", "read_csv_strict", "recombine_sequence_gate_predictions",
    "resolve_task_inputs", "run_fold_layer_task", "task_invariants", "task_model_path",
    "task_shard_path", "validated_sequence_gate_codes", "version_payload",
    "write_coefficient_artifact",
]

#!/usr/bin/env python3
"""Protocol-2.0 fivefold within-protein base256 OLS workflow."""

from __future__ import annotations

import csv
import gc
import math
import os
import sys
import time
from dataclasses import dataclass
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
    GEOMETRY_METRICS_FILENAME,
    MODEL_STABILITY_FILENAME,
    MODEL_STATUS_FILENAME,
    OOFComparisonAccumulator,
    OOFPairAssembler,
    OOFSubgroupAccumulator,
    PER_PROTEIN_FILENAME,
    RUNTIME_FILENAME,
    SUBGROUP_SEQUENCE_FILENAME,
    SUBGROUP_SPATIAL_FILENAME,
    TRAINING_METRICS_FILENAME,
    build_fold_metrics_row,
    build_geometry_row,
    build_model_stability_rows,
    build_model_status_row,
    build_per_protein_row,
    build_runtime_row,
    build_training_metrics_row,
    commit_checkpoint,
    fit_train_only_baseline,
    fit_train_only_standardizer,
    fivefold_result_root,
    load_cv_folds,
    map_layer_task,
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
    N_OUTER_FOLDS,
    N_PAIR_FOLDS,
    pair_fold_assignment_digest,
    sequence_separation_bin_codes,
    spatial_distance_bin_codes,
    within_protein_pair_folds,
)
from shared_ols_base256_common import (  # noqa: E402
    ProteinLayerRecord,
    load_feature_block,
    load_target_records,
    log,
    peak_rss_mb,
    resolve_task_inputs,
)


ANALYSIS_NAME = "per_protein_ols_base256"
MODEL_FAMILY = "per_protein_ols"
FAMILY_DIR_NAME = "per_protein_ols_base256"
EVALUATION_DESIGN = "within_protein_pair_5fold"
OOF_PREDICTION_SCOPE = "heldout_pair_folds_all_pairs"
CONDITIONAL_BASELINE_TYPE = "per_protein_pair_fold_training_mean"
CONDITION_LIMIT = 1e12
MIN_TRAIN_ROWS = FEATURE_DIM + 1
TASK_PAYLOAD_VERSION = "per_protein_ols_base256_layer_payload_v1"
ARTIFACT_VERSION = "per_protein_ols_base256_coefficient_shard_v1"

CONFIGURATION = {
    "analysis_name": ANALYSIS_NAME,
    "model_family": MODEL_FAMILY,
    "feature_definition": "directional_base256_strict_lower_triangle",
    "feature_dim": FEATURE_DIM,
    "feature_dtype_for_fit": "float64",
    "target": "final_predicted_pdb_ca_distance_angstrom",
    "model": "sklearn.linear_model.LinearRegression",
    "fit_intercept": True,
    "pair_fold_count": N_PAIR_FOLDS,
    "pair_fold_source": "per_protein_pair_folds_v1.csv",
    "scaler": "pair_fold_training_rows_only_population_std",
    "scale_floor": 1e-12,
    "evaluation_design": EVALUATION_DESIGN,
    "oof_prediction_scope": OOF_PREDICTION_SCOPE,
    "global_baseline": "each_pair_fold_training_mean",
    "conditional_baseline": CONDITIONAL_BASELINE_TYPE,
    "stable_fit_policy": {
        "minimum_training_rows": MIN_TRAIN_ROWS,
        "required_design_rank": FEATURE_DIM,
        "condition_number_max": CONDITION_LIMIT,
        "protein_row_requires_all_five_fits": True,
    },
    "geometry_scope": "one_complete_oof_vector_per_protein_layer",
    "raw_pair_predictions_saved": False,
    "coefficient_storage": "one_compressed_npz_shard_per_layer",
}
CONFIGURATION_DIGEST = semantic_digest(CONFIGURATION)

TASK_TABLES = {
    PER_PROTEIN_FILENAME: "per_protein.csv",
    "fold_metrics.csv": "fold_metrics.csv",
    TRAINING_METRICS_FILENAME: "training_metrics.csv",
    MODEL_STATUS_FILENAME: "model_status.csv",
    RUNTIME_FILENAME: "runtime.csv",
    GEOMETRY_METRICS_FILENAME: "geometry_metrics.csv",
    MODEL_STABILITY_FILENAME: "model_stability.csv",
    SUBGROUP_SEQUENCE_FILENAME: "subgroup_metrics/by_sequence_separation.csv",
    SUBGROUP_SPATIAL_FILENAME: "subgroup_metrics/by_spatial_distance.csv",
}
TASK_PAYLOAD_PATH = "task_payload.json"
TASK_MODEL_PATH = "model_artifacts/coefficient_shard.npz"
TASK_CHECKPOINT_FILENAME = "checkpoint.json"


def family_output_root(csv_root: str | Path | None = None) -> Path:
    return fivefold_result_root(csv_root) / FAMILY_DIR_NAME


def task_shard_path(root: str | Path, layer: int) -> Path:
    return Path(root) / "layer_shards" / f"layer_{int(layer):02d}"


def expected_task_artifacts() -> dict[str, str]:
    return {
        **{logical: relative for logical, relative in TASK_TABLES.items()},
        "task_payload": TASK_PAYLOAD_PATH,
        "coefficient_shard": TASK_MODEL_PATH,
    }


@dataclass(frozen=True)
class PerProteinOLSFit:
    protein_id: str
    pair_fold: int
    coefficient_standardized: np.ndarray
    coefficient_raw_scale: np.ndarray
    intercept_standardized: float
    intercept_raw_scale: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    scaler_digest: str
    baseline_value: float
    baseline_digest: str
    design_rank: int
    largest_singular_value: float
    smallest_singular_value: float
    condition_number: float
    fit_status: str
    support_flags: str
    n_training_pairs: int

    @property
    def stable(self) -> bool:
        return self.fit_status == "fitted"


def _fit_status(n_rows: int, rank: int, condition: float) -> tuple[str, str]:
    flags = []
    if int(n_rows) < MIN_TRAIN_ROWS:
        flags.append("underdetermined")
    if int(rank) < FEATURE_DIM:
        flags.append("rank_deficient")
    if not math.isfinite(float(condition)) or float(condition) > CONDITION_LIMIT:
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


def validate_pair_assignments(
    record: ProteinLayerRecord,
    pair_manifest_entry,
) -> np.ndarray:
    assignments = within_protein_pair_folds(record.n_residues, record.protein_id)
    counts = tuple(int(value) for value in np.bincount(assignments, minlength=N_PAIR_FOLDS))
    observed_digest = pair_fold_assignment_digest(record.n_residues, record.protein_id)
    if (
        int(pair_manifest_entry.n_residues) != record.n_residues
        or int(pair_manifest_entry.n_pairs) != record.targets.size
        or counts != tuple(int(value) for value in pair_manifest_entry.fold_counts)
        or observed_digest != str(pair_manifest_entry.assignment_digest)
    ):
        raise ValueError(f"Frozen pair-fold assignment mismatch for {record.protein_id}")
    if (
        assignments.shape != record.targets.shape
        or set(np.unique(assignments)) != set(range(N_PAIR_FOLDS))
    ):
        raise ValueError(f"Pair folds are incomplete for {record.protein_id}")
    return assignments


def fit_pair_fold_ols(
    protein_id: str,
    features: np.ndarray,
    targets: np.ndarray,
    assignments: np.ndarray,
    pair_fold: int,
) -> tuple[PerProteinOLSFit, np.ndarray, np.ndarray, np.ndarray]:
    """Fit one four-pair-fold OLS model and predict only its held-out fold."""
    fold = int(pair_fold)
    if fold not in range(N_PAIR_FOLDS):
        raise ValueError("pair_fold must be in 0..4")
    values = np.asarray(features)
    y = np.asarray(targets, dtype=np.float64)
    labels = np.asarray(assignments, dtype=np.int16)
    if (
        values.shape != (y.size, FEATURE_DIM)
        or labels.shape != y.shape
        or not np.all(np.isfinite(values))
        or not np.all(np.isfinite(y))
    ):
        raise ValueError("Per-protein OLS inputs are invalid")
    heldout = np.flatnonzero(labels == fold).astype(np.int64, copy=False)
    training = np.flatnonzero(labels != fold).astype(np.int64, copy=False)
    if heldout.size < 1 or training.size < 1 or heldout.size + training.size != y.size:
        raise ValueError("Pair-fold training and held-out rows must be disjoint and exhaustive")

    training_features = np.asarray(values[training], dtype=np.float64)
    training_targets = np.asarray(y[training], dtype=np.float64)
    standardizer = fit_train_only_standardizer({str(protein_id): training_features})
    baseline = fit_train_only_baseline({str(protein_id): training_targets})
    standardized = standardizer.transform(training_features)
    model = LinearRegression(copy_X=False, fit_intercept=True)
    model.fit(standardized, training_targets)
    coefficient_standardized = np.asarray(model.coef_, dtype=np.float64).copy()
    intercept_standardized = float(model.intercept_)
    singular = np.asarray(getattr(model, "singular_", []), dtype=np.float64)
    rank = int(getattr(model, "rank_", -1))
    largest = float(np.max(singular)) if singular.size else float("nan")
    smallest = float(np.min(singular)) if singular.size else float("nan")
    condition = largest / smallest if singular.size and smallest > 0 else float("inf")
    status, flags = _fit_status(training.size, rank, condition)
    coefficient_raw = coefficient_standardized / standardizer.feature_scale
    intercept_raw = intercept_standardized - float(
        np.dot(coefficient_raw, standardizer.feature_mean)
    )
    fit = PerProteinOLSFit(
        protein_id=str(protein_id), pair_fold=fold,
        coefficient_standardized=coefficient_standardized,
        coefficient_raw_scale=np.asarray(coefficient_raw, dtype=np.float64),
        intercept_standardized=intercept_standardized,
        intercept_raw_scale=intercept_raw,
        feature_mean=standardizer.feature_mean.copy(),
        feature_scale=standardizer.feature_scale.copy(),
        scaler_digest=standardizer.scaler_digest,
        baseline_value=baseline.value,
        baseline_digest=baseline.baseline_digest,
        design_rank=rank, largest_singular_value=largest,
        smallest_singular_value=smallest, condition_number=condition,
        fit_status=status, support_flags=flags,
        n_training_pairs=int(training.size),
    )
    training_predictions = predict_pair_fold_ols(fit, values[training])
    heldout_predictions = predict_pair_fold_ols(fit, values[heldout])
    del standardized, model
    return fit, training, training_predictions, heldout_predictions


def predict_pair_fold_ols(fit: PerProteinOLSFit, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM or not np.all(np.isfinite(values)):
        raise ValueError("Per-protein OLS prediction features are invalid")
    predictions = fit.intercept_standardized + (
        (values - fit.feature_mean) / fit.feature_scale
    ) @ fit.coefficient_standardized
    predictions = np.asarray(predictions, dtype=np.float64)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Per-protein OLS produced nonfinite predictions")
    return predictions


def write_coefficient_shard(path: Path, layer: int, fits: Sequence[PerProteinOLSFit]) -> None:
    ordered = tuple(fits)
    if not ordered:
        raise ValueError("Coefficient shard cannot be empty")
    responsibilities = [(fit.protein_id, fit.pair_fold) for fit in ordered]
    if len(set(responsibilities)) != len(responsibilities):
        raise ValueError("Coefficient shard contains duplicate model responsibilities")
    write_v2_npz(
        path,
        artifact_version=np.asarray(ARTIFACT_VERSION),
        analysis_name=np.asarray(ANALYSIS_NAME),
        model_family=np.asarray(MODEL_FAMILY),
        layer=np.asarray(layer, dtype=np.int16),
        protein_id=np.asarray([fit.protein_id for fit in ordered]),
        pair_fold=np.asarray([fit.pair_fold for fit in ordered], dtype=np.int16),
        coefficient_standardized=np.asarray(
            [fit.coefficient_standardized for fit in ordered], dtype=np.float64,
        ),
        coefficient_raw_scale=np.asarray(
            [fit.coefficient_raw_scale for fit in ordered], dtype=np.float64,
        ),
        intercept_standardized=np.asarray(
            [fit.intercept_standardized for fit in ordered], dtype=np.float64,
        ),
        intercept_raw_scale=np.asarray(
            [fit.intercept_raw_scale for fit in ordered], dtype=np.float64,
        ),
        feature_mean=np.asarray([fit.feature_mean for fit in ordered], dtype=np.float64),
        feature_scale=np.asarray([fit.feature_scale for fit in ordered], dtype=np.float64),
        scaler_digest=np.asarray([fit.scaler_digest for fit in ordered]),
        baseline_value=np.asarray([fit.baseline_value for fit in ordered], dtype=np.float64),
        baseline_digest=np.asarray([fit.baseline_digest for fit in ordered]),
        fit_status=np.asarray([fit.fit_status for fit in ordered]),
        configuration_digest=np.asarray(CONFIGURATION_DIGEST),
    )


def load_coefficient_shard(
    path: str | Path,
    *,
    expected_layer: int | None = None,
    expected_responsibilities: Sequence[tuple[str, int]] | None = None,
) -> dict[str, np.ndarray | str | int]:
    expected_keys = {
        "artifact_version", "analysis_name", "model_family", "layer", "protein_id",
        "pair_fold", "coefficient_standardized", "coefficient_raw_scale",
        "intercept_standardized", "intercept_raw_scale", "feature_mean", "feature_scale",
        "scaler_digest", "baseline_value", "baseline_digest", "fit_status",
        "configuration_digest",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(f"Coefficient shard schema mismatch: {path}")
        result = {name: archive[name].copy() for name in archive.files}
    scalars = {
        name: np.asarray(result[name]).item()
        for name in (
            "artifact_version", "analysis_name", "model_family", "layer",
            "configuration_digest",
        )
    }
    if (
        scalars["artifact_version"] != ARTIFACT_VERSION
        or scalars["analysis_name"] != ANALYSIS_NAME
        or scalars["model_family"] != MODEL_FAMILY
        or scalars["configuration_digest"] != CONFIGURATION_DIGEST
    ):
        raise ValueError("Coefficient shard identity mismatch")
    layer = int(scalars["layer"])
    if expected_layer is not None and layer != int(expected_layer):
        raise ValueError("Coefficient shard layer mismatch")
    protein_ids = np.asarray(result["protein_id"]).astype(str)
    pair_folds = np.asarray(result["pair_fold"], dtype=np.int16)
    n_models = protein_ids.size
    responsibilities = list(zip(protein_ids.tolist(), pair_folds.tolist()))
    if n_models < 1 or len(set(responsibilities)) != n_models:
        raise ValueError("Coefficient shard responsibilities are empty or duplicated")
    if (
        expected_responsibilities is not None
        and responsibilities != list(expected_responsibilities)
    ):
        raise ValueError("Coefficient shard responsibility ordering mismatch")
    matrix_names = (
        "coefficient_standardized", "coefficient_raw_scale", "feature_mean", "feature_scale",
    )
    matrices = [np.asarray(result[name], dtype=np.float64) for name in matrix_names]
    vector_names = (
        "intercept_standardized", "intercept_raw_scale", "baseline_value",
        "scaler_digest", "baseline_digest", "fit_status",
    )
    if any(matrix.shape != (n_models, FEATURE_DIM) for matrix in matrices):
        raise ValueError("Coefficient shard matrix shape mismatch")
    if any(np.asarray(result[name]).shape != (n_models,) for name in vector_names):
        raise ValueError("Coefficient shard vector shape mismatch")
    if any(not np.all(np.isfinite(matrix)) for matrix in matrices[:3]):
        raise ValueError("Coefficient shard contains nonfinite model arrays")
    if not np.all(np.isfinite(matrices[3])) or np.any(matrices[3] <= 0):
        raise ValueError("Coefficient shard feature scales are invalid")
    standardized, raw, means, scales = matrices
    if not np.allclose(raw, standardized / scales, rtol=1e-12, atol=1e-12):
        raise ValueError("Coefficient shard raw and standardized coefficients disagree")
    expected_intercepts = np.asarray(
        result["intercept_standardized"], dtype=np.float64,
    ) - np.einsum("ij,ij->i", raw, means)
    if not np.allclose(
        np.asarray(result["intercept_raw_scale"], dtype=np.float64), expected_intercepts,
        rtol=1e-12, atol=1e-12,
    ):
        raise ValueError("Coefficient shard raw and standardized intercepts disagree")
    if not np.all(np.isfinite(np.asarray(result["baseline_value"], dtype=np.float64))):
        raise ValueError("Coefficient shard baselines are nonfinite")
    return {**result, **scalars, "layer": layer}


def task_invariants(
    layer: int,
    *,
    fold_digests: Mapping[str, object],
    input_digest: str,
) -> dict[str, object]:
    return {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "task_id": int(layer),
        "layer": int(layer),
        "configuration_digest": CONFIGURATION_DIGEST,
        "cohort_digest": fold_digests["cohort_digest"],
        "outer_fold_digest": fold_digests["outer_fold_digest"],
        "pair_fold_manifest_digest": fold_digests["pair_fold_manifest_digest"],
        "input_identity_digest": str(input_digest),
    }


def execute_layer_task(
    *,
    layer: int,
    records: Sequence[ProteinLayerRecord],
    fold_bundle,
    input_digest: str,
    root: str | Path,
    feature_loader: Callable[[ProteinLayerRecord], np.ndarray] = load_feature_block,
) -> Path:
    """Fit five pair-fold models per protein and checkpoint one complete layer."""
    started = time.monotonic()
    task_layer = map_layer_task(int(layer))
    shard = task_shard_path(root, task_layer)
    invariants = task_invariants(
        task_layer, fold_digests=fold_bundle.fold_digests, input_digest=input_digest,
    )
    checkpoint = shard / TASK_CHECKPOINT_FILENAME
    if checkpoint.is_file():
        validate_resume_checkpoint(
            checkpoint, expected_invariants=invariants,
            expected_artifact_names=set(expected_task_artifacts()),
        )
        log(f"task {task_layer} layer {task_layer}: valid checkpoint; skipping")
        return shard

    by_id = {record.protein_id: record for record in records}
    expected_ids = tuple(sorted(fold_bundle.eligible_protein_ids))
    if len(by_id) != len(records) or set(by_id) != set(expected_ids):
        raise ValueError("Layer records must cover the frozen cohort exactly once")

    oof = OOFComparisonAccumulator()
    sequence = OOFSubgroupAccumulator(
        "sequence_separation", ("1", "2-4", "5-9", "10-19", "20-49", "50+"),
    )
    spatial = OOFSubgroupAccumulator(
        "spatial_distance", ("0-5", "5-10", "10-20", "20-40", "40+"),
    )
    per_rows: list[dict[str, object]] = []
    training_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    fits: list[PerProteinOLSFit] = []
    n_fitted_pairs = 0

    log(f"task {task_layer}: fitting five pair folds for {len(expected_ids)} proteins")
    for index, protein_id in enumerate(expected_ids, start=1):
        record = by_id[protein_id]
        outer_entry = fold_bundle.outer_by_protein[protein_id]
        pair_entry = fold_bundle.pair_by_protein[protein_id]
        if record.outer_fold != int(outer_entry.outer_fold):
            raise ValueError(f"Outer-fold reporting label mismatch for {protein_id}")
        assignments = validate_pair_assignments(record, pair_entry)
        features = feature_loader(record)
        assembler = OOFPairAssembler(record.targets, with_conditional_baseline=True)
        protein_fits: list[PerProteinOLSFit] = []
        for pair_fold in range(N_PAIR_FOLDS):
            fit, training_indices, training_predictions, heldout_predictions = fit_pair_fold_ols(
                protein_id, features, record.targets, assignments, pair_fold,
            )
            heldout_indices = np.flatnonzero(assignments == pair_fold).astype(np.int64, copy=False)
            assembler.add_fold(
                pair_fold, heldout_indices, heldout_predictions, fit.baseline_value,
                conditional_baseline_predictions=fit.baseline_value,
            )
            status_rows.append(build_model_status_row(
                analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
                feature_dim=FEATURE_DIM, layer=task_layer,
                outer_fold=record.outer_fold, protein_id=protein_id,
                pair_fold=pair_fold, fit_status=fit.fit_status,
                support_flags=fit.support_flags, n_training_proteins=1,
                n_training_pairs=fit.n_training_pairs,
                design_rank=fit.design_rank,
                largest_singular_value=fit.largest_singular_value,
                smallest_singular_value=fit.smallest_singular_value,
                condition_number=fit.condition_number,
                random_seed=int(pair_entry.pair_fold_seed),
            ))
            training_rows.append(build_training_metrics_row(
                analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
                feature_dim=FEATURE_DIM, layer=task_layer,
                evaluation_design=EVALUATION_DESIGN,
                training_scope="four_within_protein_pair_folds",
                y_true=record.targets[training_indices],
                predictions=training_predictions, fit_status=fit.fit_status,
                outer_fold=record.outer_fold, protein_id=protein_id,
                pair_fold=pair_fold, support_flags=fit.support_flags,
                n_training_proteins=1, design_rank=fit.design_rank,
                largest_singular_value=fit.largest_singular_value,
                smallest_singular_value=fit.smallest_singular_value,
                condition_number=fit.condition_number,
            ))
            n_fitted_pairs += fit.n_training_pairs
            fits.append(fit)
            protein_fits.append(fit)
        assembled = assembler.finalize()
        predictions = np.asarray(assembled["predictions"], dtype=np.float64)
        global_baseline = np.asarray(
            assembled["global_baseline_predictions"], dtype=np.float64,
        )
        conditional_baseline = np.asarray(
            assembled["conditional_baseline_predictions"], dtype=np.float64,
        )
        all_stable = all(fit.stable for fit in protein_fits)
        prediction_status = "complete" if all_stable else "complete_from_flagged_fit"
        per_rows.append(build_per_protein_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=task_layer, protein_id=protein_id,
            source=record.source, outer_fold=record.outer_fold,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            n_residues=record.n_residues, y_true=record.targets,
            predictions=predictions, prediction_status=prediction_status,
        ))
        oof.update_protein(
            protein_id, record.outer_fold, record.targets, predictions,
            global_baseline, conditional_baseline_predictions=conditional_baseline,
            stable_fit=all_stable,
        )
        sequence.update_protein(
            protein_id, record.targets, predictions,
            sequence_separation_bin_codes(
                record.pair_i, record.pair_j, chain_ids=record.chain_ids,
            ),
        )
        spatial.update_protein(
            protein_id, record.targets, predictions,
            spatial_distance_bin_codes(record.targets),
        )
        geometry_rows.append(build_geometry_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=task_layer, protein_id=protein_id,
            source=record.source, outer_fold=record.outer_fold,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            n_residues=record.n_residues, true_coordinates=record.coordinates,
            predictions=predictions,
        ))
        stability_rows.extend(build_model_stability_rows(
            [fit.coefficient_standardized for fit in protein_fits],
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            layer=task_layer, feature_names=FEATURE_NAMES, protein_id=protein_id,
        ))
        del features, assembler, predictions, global_baseline, conditional_baseline
        gc.collect()
        if index % 20 == 0 or index == len(expected_ids):
            log(f"task {task_layer}: completed {index}/{len(expected_ids)} proteins")

    oof.require_complete()
    if any(row["geometry_status"] != "complete" for row in geometry_rows):
        raise RuntimeError("Per-protein OOF geometry is incomplete")
    fold_rows = [
        build_fold_metrics_row(
            oof.subset(fold_bundle.protein_ids_for_outer_fold(outer_fold)),
            outer_fold=outer_fold, analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY, feature_dim=FEATURE_DIM, layer=task_layer,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            conditional_baseline_type=CONDITIONAL_BASELINE_TYPE,
        )
        for outer_fold in range(N_OUTER_FOLDS)
    ]
    elapsed = time.monotonic() - started
    runtime_row = build_runtime_row(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        task_id=task_layer, layer=task_layer, completion_status="complete",
        n_fitted_proteins=len(expected_ids), n_evaluated_proteins=len(expected_ids),
        n_fitted_pairs=n_fitted_pairs, n_evaluated_pairs=oof.n_evaluated_pairs,
        elapsed_seconds=elapsed, peak_rss_mb=peak_rss_mb(),
    )
    identity = dict(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM, layer=task_layer,
        evaluation_design=EVALUATION_DESIGN,
        oof_prediction_scope=OOF_PREDICTION_SCOPE,
    )
    tables = {
        PER_PROTEIN_FILENAME: per_rows,
        "fold_metrics.csv": fold_rows,
        TRAINING_METRICS_FILENAME: training_rows,
        MODEL_STATUS_FILENAME: status_rows,
        RUNTIME_FILENAME: [runtime_row],
        GEOMETRY_METRICS_FILENAME: geometry_rows,
        MODEL_STABILITY_FILENAME: stability_rows,
        SUBGROUP_SEQUENCE_FILENAME: sequence.rows(**identity),
        SUBGROUP_SPATIAL_FILENAME: spatial.rows(**identity),
    }
    for logical, relative in TASK_TABLES.items():
        write_v2_csv(shard / relative, logical, tables[logical])
    write_coefficient_shard(shard / TASK_MODEL_PATH, task_layer, fits)
    atomic_write_json(shard / TASK_PAYLOAD_PATH, {
        "task_payload_version": TASK_PAYLOAD_VERSION,
        "task_id": task_layer,
        "layer": task_layer,
        "configuration_digest": CONFIGURATION_DIGEST,
        "n_models": len(fits),
        "n_stable_proteins": sum(oof.stable_fit_by_protein.values()),
        "pair_assignment_digests": {
            protein_id: str(fold_bundle.pair_by_protein[protein_id].assignment_digest)
            for protein_id in expected_ids
        },
        "oof_accumulator": oof.to_payload(),
        "sequence_subgroups": sequence.to_payload(),
        "spatial_subgroups": spatial.to_payload(),
    })
    commit_checkpoint(
        checkpoint, invariants=invariants, artifacts=expected_task_artifacts(),
    )
    log(f"task {task_layer}: complete in {elapsed:.1f}s")
    return shard


def read_csv_strict(path: str | Path, logical_filename: str) -> list[dict[str, str]]:
    from cross_validation_common import V2_CSV_SCHEMAS

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != V2_CSV_SCHEMAS[logical_filename]:
            raise ValueError(f"Schema mismatch in {path}")
        return list(reader)


def run_layer_task(
    task_id: int | None = None,
    *,
    root: str | Path | None = None,
) -> Path:
    if task_id is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise RuntimeError("Provide task_id or set SLURM_ARRAY_TASK_ID")
        task_id = int(raw)
    layer = map_layer_task(task_id)
    fold_bundle = load_cv_folds()
    destination = family_output_root() if root is None else Path(root)
    paths, input_digest = resolve_task_inputs(fold_bundle, layer)
    invariants = task_invariants(
        layer, fold_digests=fold_bundle.fold_digests, input_digest=input_digest,
    )
    checkpoint = task_shard_path(destination, layer) / TASK_CHECKPOINT_FILENAME
    if checkpoint.is_file():
        validate_resume_checkpoint(
            checkpoint, expected_invariants=invariants,
            expected_artifact_names=set(expected_task_artifacts()),
        )
        log(f"task {task_id}: valid checkpoint; no inputs opened")
        return checkpoint.parent
    try:
        records = load_target_records(fold_bundle, paths)
        return execute_layer_task(
            layer=layer, records=records, fold_bundle=fold_bundle,
            input_digest=input_digest, root=destination,
        )
    except Exception as exc:
        atomic_write_json(destination / "failures" / f"layer_{layer:02d}.json", {
            "analysis_name": ANALYSIS_NAME, "task_id": int(task_id), "layer": layer,
            "exception_type": type(exc).__name__, "message": str(exc),
            "configuration_digest": CONFIGURATION_DIGEST, "timestamp": utc_now(),
        })
        raise


__all__ = [
    "ANALYSIS_NAME", "ARTIFACT_VERSION", "CONDITIONAL_BASELINE_TYPE",
    "CONFIGURATION", "CONFIGURATION_DIGEST", "EVALUATION_DESIGN",
    "FAMILY_DIR_NAME", "MODEL_FAMILY", "OOF_PREDICTION_SCOPE",
    "PerProteinOLSFit", "TASK_CHECKPOINT_FILENAME", "TASK_MODEL_PATH",
    "TASK_PAYLOAD_PATH", "TASK_PAYLOAD_VERSION", "TASK_TABLES",
    "execute_layer_task", "expected_task_artifacts", "family_output_root",
    "fit_pair_fold_ols", "load_coefficient_shard", "predict_pair_fold_ols",
    "read_csv_strict", "run_layer_task", "task_invariants", "task_shard_path",
    "validate_pair_assignments", "write_coefficient_shard",
]

#!/usr/bin/env python3
"""Protocol-2.0 fivefold shared base256 OLS runner primitives."""

from __future__ import annotations

import csv
import gc
import math
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np
import scipy
import sklearn
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
for _path in (CLUSTER_DIR, SCRIPT_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cluster_regression_helpers import (  # noqa: E402
    compute_distance_matrix,
    extract_ca_coordinates,
    resolve_combined_pair_block_path,
    resolve_combined_pdb_path,
)
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
    extract_directional_base256,
    sequence_separation_bin_codes,
    spatial_distance_bin_codes,
    strict_lower_triangle_indices,
)


ANALYSIS_NAME = "shared_ols_base256"
MODEL_FAMILY = "shared_ols"
FAMILY_DIR_NAME = "shared_ols_base256"
EVALUATION_DESIGN = "protein_outer_5fold"
OOF_PREDICTION_SCOPE = "heldout_proteins_all_pairs"
CONDITION_LIMIT = 1e12
MIN_TRAIN_ROWS = FEATURE_DIM + 1
TASK_PAYLOAD_VERSION = "shared_ols_base256_task_payload_v1"
ARTIFACT_VERSION = "shared_ols_base256_coefficients_v1"

CONFIGURATION = {
    "analysis_name": ANALYSIS_NAME,
    "model_family": MODEL_FAMILY,
    "cv_protocol_version": CV_PROTOCOL_VERSION,
    "evaluation_design": EVALUATION_DESIGN,
    "evaluation_scope": EVALUATION_SCOPE,
    "oof_prediction_scope": OOF_PREDICTION_SCOPE,
    "outer_folds": N_OUTER_FOLDS,
    "layers": list(LAYERS),
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
    "training_pairs": "all_pairs_from_four_outer_training_folds",
    "training_weighting": "equal_total_weight_per_training_protein",
    "scaler": "fold_training_only_protein_equal_float64_moments",
    "global_baseline": "fold_training_only_protein_equal_target_mean",
    "condition_number_flag_threshold": CONDITION_LIMIT,
    "stable_fit_policy": "rank_256_and_finite_condition_at_most_1e12",
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
TASK_MODEL_PATH = "model_artifacts/coefficients.npz"
TASK_CHECKPOINT_FILENAME = "checkpoint.json"


from canonical_storage import assert_writable, deployment_revision


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def family_output_root(csv_root: str | Path | None = None) -> Path:
    return fivefold_result_root(csv_root) / FAMILY_DIR_NAME


def task_shard_path(root: str | Path, task: FoldLayerTask) -> Path:
    return Path(root) / "layer_shards" / (
        f"fold_{task.outer_fold}_layer_{task.layer:02d}"
    )


def expected_task_artifacts() -> dict[str, str]:
    return {
        **{logical: relative for logical, relative in TASK_TABLES.items()},
        "task_payload": TASK_PAYLOAD_PATH,
        "coefficient_model": TASK_MODEL_PATH,
    }


@dataclass(frozen=True)
class ProteinLayerRecord:
    protein_id: str
    source: str
    n_residues: int
    outer_fold: int
    coordinates: np.ndarray
    chain_ids: tuple[str, ...]
    pair_i: np.ndarray
    pair_j: np.ndarray
    targets: np.ndarray
    pair_block_path: Path | None = None
    synthetic_features: np.ndarray | None = None


@dataclass(frozen=True)
class SharedOLSFit:
    coefficient_standardized: np.ndarray
    coefficient_raw_scale: np.ndarray
    intercept_standardized: float
    intercept_raw_scale: float
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    scaler_digest: str
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


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)


def version_payload() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scipy": scipy.__version__,
        "scikit_learn": sklearn.__version__,
    }


def code_revision() -> str:
    return deployment_revision(SCRIPT_DIR)


def _input_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen input is missing: {path}")
    stat = path.stat()
    return {
        "path": str(path), "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def resolve_task_inputs(fold_bundle, layer: int) -> tuple[dict[str, dict[str, Path]], str]:
    """Resolve all fold/layer inputs without opening tensors or PDBs."""
    paths: dict[str, dict[str, Path]] = {}
    identities = []
    for protein_id in fold_bundle.eligible_protein_ids:
        entry = fold_bundle.outer_by_protein[protein_id]
        pair_path = resolve_combined_pair_block_path(protein_id, entry.source, layer)
        pdb_path = resolve_combined_pdb_path(protein_id, entry.source)
        paths[protein_id] = {"pair_block": pair_path, "pdb": pdb_path}
        identities.append({
            "protein_id": protein_id,
            "source": entry.source,
            "n_residues": int(entry.n_residues),
            "n_pairs": int(entry.n_pairs),
            "outer_fold": int(entry.outer_fold),
            "pair_block": _input_identity(pair_path),
            "pdb": _input_identity(pdb_path),
        })
    return paths, semantic_digest(identities)


def load_target_records(fold_bundle, paths: Mapping[str, Mapping[str, Path]]) -> list[ProteinLayerRecord]:
    records = []
    for index, protein_id in enumerate(fold_bundle.eligible_protein_ids, start=1):
        entry = fold_bundle.outer_by_protein[protein_id]
        ca = extract_ca_coordinates(paths[protein_id]["pdb"])
        coordinates = np.asarray(ca["coordinates"], dtype=np.float64)
        residues = tuple(ca["residues"])
        if coordinates.shape != (int(entry.n_residues), 3) or len(residues) != entry.n_residues:
            raise ValueError(
                f"Frozen residue count mismatch for {protein_id}: "
                f"{coordinates.shape[0]} != {entry.n_residues}"
            )
        pair_i, pair_j = strict_lower_triangle_indices(entry.n_residues)
        targets = np.asarray(
            compute_distance_matrix(coordinates)[pair_i, pair_j], dtype=np.float64,
        )
        if targets.size != int(entry.n_pairs) or not np.all(np.isfinite(targets)):
            raise ValueError(f"Frozen target mismatch for {protein_id}")
        records.append(ProteinLayerRecord(
            protein_id=protein_id, source=str(entry.source),
            n_residues=int(entry.n_residues), outer_fold=int(entry.outer_fold),
            coordinates=coordinates,
            chain_ids=tuple(str(residue[0]) for residue in residues),
            pair_i=pair_i, pair_j=pair_j, targets=targets,
            pair_block_path=Path(paths[protein_id]["pair_block"]),
        ))
        if index % 50 == 0 or index == len(fold_bundle.eligible_protein_ids):
            log(f"loaded targets for {index}/{len(fold_bundle.eligible_protein_ids)} proteins")
    return records


def load_feature_block(record: ProteinLayerRecord) -> np.ndarray:
    """Load one directional base256 block, using a float32 temporary cache."""
    if record.synthetic_features is not None:
        values = np.asarray(record.synthetic_features)
    else:
        if record.pair_block_path is None:
            raise ValueError(f"No pair tensor path for {record.protein_id}")
        pair_block = np.load(str(record.pair_block_path))
        try:
            expected = (record.n_residues, record.n_residues, FEATURE_DIM // 2)
            if pair_block.shape != expected:
                raise ValueError(
                    f"Pair tensor shape mismatch for {record.protein_id}: "
                    f"{pair_block.shape} != {expected}"
                )
            values = extract_directional_base256(
                pair_block, record.pair_i, record.pair_j,
            )
        finally:
            del pair_block
    if values.shape != (record.targets.size, FEATURE_DIM) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid base256 features for {record.protein_id}")
    return np.ascontiguousarray(values, dtype=np.float32)


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


def fit_shared_ols(
    feature_blocks: Mapping[str, np.ndarray],
    target_blocks: Mapping[str, np.ndarray],
) -> tuple[SharedOLSFit, np.ndarray, np.ndarray]:
    """Fit the dense protein-equal fold model and return compact diagnostics."""
    prepared = prepare_training_fold(feature_blocks, target_blocks)
    if prepared.standardized_features.shape[1] != FEATURE_DIM:
        raise ValueError(f"Shared OLS requires exactly {FEATURE_DIM} features")
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
    fit = SharedOLSFit(
        coefficient_standardized=coefficient_standardized,
        coefficient_raw_scale=np.asarray(coefficient_raw, dtype=np.float64),
        intercept_standardized=intercept_standardized,
        intercept_raw_scale=intercept_raw,
        feature_mean=prepared.standardizer.feature_mean.copy(),
        feature_scale=prepared.standardizer.feature_scale.copy(),
        scaler_digest=prepared.standardizer.scaler_digest,
        global_baseline_value=prepared.global_baseline.value,
        global_baseline_digest=prepared.global_baseline.baseline_digest,
        design_rank=rank,
        largest_singular_value=largest,
        smallest_singular_value=smallest,
        condition_number=condition,
        fit_status=status,
        support_flags=flags,
        n_training_proteins=prepared.standardizer.n_training_proteins,
        n_training_pairs=prepared.standardizer.n_training_rows,
    )
    # ``copy_X=False`` permits sklearn to center the dense fit matrix in place.
    # Rebuild diagnostic predictions from the untouched float32 cache so they
    # use exactly the same replay transform as held-out predictions.
    training_predictions = np.concatenate([
        predict_shared_ols(fit, feature_blocks[protein_id])
        for protein_id in sorted(feature_blocks)
    ])
    return fit, prepared.targets.copy(), training_predictions


def predict_shared_ols(fit: SharedOLSFit, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM or not np.all(np.isfinite(values)):
        raise ValueError("Shared OLS prediction features are invalid")
    predictions = fit.intercept_standardized + (
        (values - fit.feature_mean) / fit.feature_scale
    ) @ fit.coefficient_standardized
    predictions = np.asarray(predictions, dtype=np.float64)
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Shared OLS produced nonfinite OOF predictions")
    return predictions


def write_coefficient_artifact(path: Path, fit: SharedOLSFit, task: FoldLayerTask) -> None:
    write_v2_npz(
        path,
        artifact_version=np.asarray(ARTIFACT_VERSION),
        analysis_name=np.asarray(ANALYSIS_NAME),
        model_family=np.asarray(MODEL_FAMILY),
        layer=np.asarray(task.layer, dtype=np.int16),
        outer_fold=np.asarray(task.outer_fold, dtype=np.int16),
        coefficient_standardized=fit.coefficient_standardized,
        coefficient_raw_scale=fit.coefficient_raw_scale,
        intercept_standardized=np.asarray(fit.intercept_standardized, dtype=np.float64),
        intercept_raw_scale=np.asarray(fit.intercept_raw_scale, dtype=np.float64),
        feature_mean=fit.feature_mean,
        feature_scale=fit.feature_scale,
        scaler_digest=np.asarray(fit.scaler_digest),
        global_baseline_value=np.asarray(fit.global_baseline_value, dtype=np.float64),
        global_baseline_digest=np.asarray(fit.global_baseline_digest),
        configuration_digest=np.asarray(CONFIGURATION_DIGEST),
        fit_status=np.asarray(fit.fit_status),
    )


def load_coefficient_artifact(
    path: str | Path,
    *,
    expected_task: FoldLayerTask | None = None,
    expected_scaler_digest: str | None = None,
) -> dict[str, object]:
    expected_keys = {
        "artifact_version", "analysis_name", "model_family", "layer", "outer_fold",
        "coefficient_standardized", "coefficient_raw_scale", "intercept_standardized",
        "intercept_raw_scale", "feature_mean", "feature_scale", "scaler_digest",
        "global_baseline_value", "global_baseline_digest", "configuration_digest",
        "fit_status",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(f"Coefficient artifact schema mismatch: {path}")
        result = {name: archive[name].copy() for name in archive.files}
    scalars = {
        name: np.asarray(result[name]).item()
        for name in (
            "artifact_version", "analysis_name", "model_family", "layer", "outer_fold",
            "intercept_standardized", "intercept_raw_scale", "scaler_digest",
            "global_baseline_value", "global_baseline_digest", "configuration_digest",
            "fit_status",
        )
    }
    if (
        scalars["artifact_version"] != ARTIFACT_VERSION
        or scalars["analysis_name"] != ANALYSIS_NAME
        or scalars["model_family"] != MODEL_FAMILY
        or scalars["configuration_digest"] != CONFIGURATION_DIGEST
    ):
        raise ValueError("Coefficient artifact identity mismatch")
    if expected_task is not None and (
        int(scalars["layer"]), int(scalars["outer_fold"])
    ) != (expected_task.layer, expected_task.outer_fold):
        raise ValueError("Coefficient artifact task identity mismatch")
    if expected_scaler_digest is not None and scalars["scaler_digest"] != expected_scaler_digest:
        raise ValueError("Coefficient artifact scaler digest mismatch")
    if (
        not str(scalars["scaler_digest"])
        or not str(scalars["global_baseline_digest"])
        or not math.isfinite(float(scalars["global_baseline_value"]))
        or not math.isfinite(float(scalars["intercept_standardized"]))
        or not math.isfinite(float(scalars["intercept_raw_scale"]))
    ):
        raise ValueError("Coefficient artifact scalar state is invalid")
    arrays = [
        np.asarray(result[name], dtype=np.float64)
        for name in (
            "coefficient_standardized", "coefficient_raw_scale", "feature_mean", "feature_scale",
        )
    ]
    if any(array.shape != (FEATURE_DIM,) or not np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("Coefficient artifact arrays are invalid")
    coefficient_standardized, coefficient_raw, feature_mean, feature_scale = arrays
    if np.any(feature_scale <= 0):
        raise ValueError("Coefficient artifact feature scales must be positive")
    if not np.allclose(
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
    """Fit, stream held-out evaluation, and atomically checkpoint one task."""
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
    feature_blocks = {}
    target_blocks = {}
    for index, protein_id in enumerate(training_ids, start=1):
        record = by_id[protein_id]
        feature_blocks[protein_id] = feature_loader(record)
        target_blocks[protein_id] = record.targets
        if index % 25 == 0 or index == len(training_ids):
            log(f"task {task.task_id}: cached {index}/{len(training_ids)} training proteins")
    log(f"task {task.task_id}: fitting dense float64 protein-equal OLS")
    fit, training_targets, training_predictions = fit_shared_ols(
        feature_blocks, target_blocks,
    )
    del feature_blocks, target_blocks
    gc.collect()

    status_row = build_model_status_row(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM, layer=task.layer, outer_fold=task.outer_fold,
        fit_status=fit.fit_status, support_flags=fit.support_flags,
        n_training_proteins=fit.n_training_proteins,
        n_training_pairs=fit.n_training_pairs, design_rank=fit.design_rank,
        largest_singular_value=fit.largest_singular_value,
        smallest_singular_value=fit.smallest_singular_value,
        condition_number=fit.condition_number,
    )
    training_row = build_training_metrics_row(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM, layer=task.layer,
        evaluation_design=EVALUATION_DESIGN,
        training_scope="all_pairs_four_outer_training_folds",
        y_true=training_targets, predictions=training_predictions,
        fit_status=fit.fit_status, support_flags=fit.support_flags,
        outer_fold=task.outer_fold, n_training_proteins=fit.n_training_proteins,
        design_rank=fit.design_rank,
        largest_singular_value=fit.largest_singular_value,
        smallest_singular_value=fit.smallest_singular_value,
        condition_number=fit.condition_number,
    )
    del training_targets, training_predictions
    gc.collect()

    oof = OOFComparisonAccumulator()
    sequence = OOFSubgroupAccumulator("sequence_separation", SEQUENCE_BIN_LABELS)
    spatial = OOFSubgroupAccumulator("spatial_distance", SPATIAL_DISTANCE_BIN_LABELS)
    per_rows = []
    geometry_rows = []
    log(f"task {task.task_id}: streaming {len(heldout_ids)} held-out proteins")
    for index, protein_id in enumerate(heldout_ids, start=1):
        record = by_id[protein_id]
        features = feature_loader(record)
        predictions = predict_shared_ols(fit, features)
        del features
        status = "complete" if fit.stable else "complete_from_flagged_fit"
        per_rows.append(build_per_protein_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=task.layer, protein_id=protein_id,
            source=record.source, outer_fold=task.outer_fold,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            n_residues=record.n_residues, y_true=record.targets,
            predictions=predictions, prediction_status=status,
        ))
        oof.update_protein(
            protein_id, task.outer_fold, record.targets, predictions,
            fit.global_baseline_value,
            stable_fit=fit.stable,
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
        raise RuntimeError(
            f"Task {task.task_id} geometry is incomplete for {failed_geometry[:5]}"
        )

    fold_row = build_fold_metrics_row(
        oof, outer_fold=task.outer_fold, analysis_name=ANALYSIS_NAME,
        model_family=MODEL_FAMILY, feature_dim=FEATURE_DIM, layer=task.layer,
        evaluation_design=EVALUATION_DESIGN,
        oof_prediction_scope=OOF_PREDICTION_SCOPE,
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
        n_fitted_pairs=fit.n_training_pairs,
        n_evaluated_pairs=oof.n_evaluated_pairs,
        elapsed_seconds=elapsed, peak_rss_mb=peak_rss_mb(),
    )
    tables = {
        PER_PROTEIN_FILENAME: per_rows,
        FOLD_METRICS_FILENAME: [fold_row],
        TRAINING_METRICS_FILENAME: [training_row],
        MODEL_STATUS_FILENAME: [status_row],
        RUNTIME_FILENAME: [runtime_row],
        GEOMETRY_METRICS_FILENAME: geometry_rows,
        SUBGROUP_SEQUENCE_FILENAME: sequence.rows(**subgroup_identity),
        SUBGROUP_SPATIAL_FILENAME: spatial.rows(**subgroup_identity),
    }
    for logical, relative in TASK_TABLES.items():
        write_v2_csv(shard / relative, logical, tables[logical])
    write_coefficient_artifact(shard / TASK_MODEL_PATH, fit, task)
    atomic_write_json(shard / TASK_PAYLOAD_PATH, {
        "task_payload_version": TASK_PAYLOAD_VERSION,
        "task_id": task.task_id, "outer_fold": task.outer_fold, "layer": task.layer,
        "configuration_digest": CONFIGURATION_DIGEST,
        "scaler_digest": fit.scaler_digest,
        "global_baseline_digest": fit.global_baseline_digest,
        "fit_stable": fit.stable,
        "oof_accumulator": oof.to_payload(),
        "sequence_subgroups": sequence.to_payload(),
        "spatial_subgroups": spatial.to_payload(),
    })
    artifacts = expected_task_artifacts()
    commit_checkpoint(
        checkpoint, invariants=invariants, artifacts=artifacts,
    )
    log(f"task {task.task_id}: complete in {elapsed:.1f}s")
    return shard


def read_csv_strict(path: str | Path, logical_filename: str) -> list[dict[str, str]]:
    from cross_validation_common import V2_CSV_SCHEMAS

    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != V2_CSV_SCHEMAS[logical_filename]:
            raise ValueError(f"Schema mismatch in {path}")
        return list(reader)


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
    "ANALYSIS_NAME", "ARTIFACT_VERSION", "CONDITION_LIMIT", "CONFIGURATION",
    "CONFIGURATION_DIGEST", "EVALUATION_DESIGN", "FAMILY_DIR_NAME", "MODEL_FAMILY",
    "OOF_PREDICTION_SCOPE", "ProteinLayerRecord", "SharedOLSFit", "TASK_CHECKPOINT_FILENAME",
    "TASK_MODEL_PATH", "TASK_PAYLOAD_PATH", "TASK_PAYLOAD_VERSION", "TASK_TABLES",
    "code_revision", "execute_fold_layer_task", "expected_task_artifacts",
    "family_output_root", "fit_shared_ols", "load_coefficient_artifact",
    "load_feature_block", "load_target_records", "log", "peak_rss_mb",
    "predict_shared_ols", "read_csv_strict", "resolve_task_inputs",
    "run_fold_layer_task", "task_invariants", "task_shard_path", "version_payload",
    "write_coefficient_artifact",
]

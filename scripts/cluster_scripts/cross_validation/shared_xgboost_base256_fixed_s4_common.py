#!/usr/bin/env python3
"""Protocol-2.0 fivefold fixed-S4 shared base256 XGBoost primitives."""

from __future__ import annotations

import csv
import gc
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
REPO_ROOT = SCRIPT_DIR.parents[2]
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
    SAMPLING_ALGORITHM,
    SAMPLING_NAMESPACE,
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
    deterministic_sample_indices,
    deterministic_sample_seed,
    fit_train_only_baseline,
    fivefold_result_root,
    load_cv_folds,
    map_fold_layer_task,
    sampling_assignment_digest,
    semantic_digest,
    utc_now,
    validate_resume_checkpoint,
    write_v2_csv,
    write_v2_npz,
)
from regression_output_helpers import atomic_write_json, sha256_file  # noqa: E402
from regression_protocol import (  # noqa: E402
    FEATURE_DIM,
    LAYERS,
    N_OUTER_FOLDS,
    SEQUENCE_BIN_LABELS,
    SPATIAL_DISTANCE_BIN_LABELS,
    extract_directional_base256,
    sequence_separation_bin_codes,
    spatial_distance_bin_codes,
    strict_lower_triangle_indices,
)


ANALYSIS_NAME = "shared_xgboost_base256_fixed_s4"
MODEL_FAMILY = "shared_xgboost"
FAMILY_DIR_NAME = "shared_xgboost_base256_fixed_s4"
EVALUATION_DESIGN = "protein_outer_5fold"
OOF_PREDICTION_SCOPE = "heldout_proteins_all_pairs"
CONFIGURATION_ID = "S4"
ROW_CAP = 5000
BOOSTING_ROUNDS = 700
BASE_RANDOM_SEED = 42
CONFIG_PATH = SCRIPT_DIR / "shared_xgboost_base256_fixed_s4_config.json"
SELECTION_SOURCE_PATH = (
    "results/xgboost_hyperparameter_search/shared/winner_cap_sweep/"
    "cross_layer_selection_table.csv"
)
SELECTION_SOURCE_SHA256 = (
    "4890bdd629a7ad29ec6ed166de533bd0ca8db1659af17cb2b4ea849b2b527569"
)
TASK_PAYLOAD_VERSION = "shared_xgboost_base256_fixed_s4_task_payload_v1"
IMPORTANCE_ARTIFACT_VERSION = "shared_xgboost_base256_fixed_s4_importance_v1"


def configuration_payload(config: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in config.items() if key != "configuration_digest"}


def load_fixed_configuration(path: str | Path = CONFIG_PATH) -> dict[str, object]:
    """Load and strictly validate the checksummed S4/5k/700 decision."""
    config = json.loads(Path(path).read_text())
    observed = semantic_digest(configuration_payload(config))
    if config.get("configuration_digest") != observed:
        raise ValueError("Fixed XGBoost configuration digest is missing or changed")
    expected = {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "configuration_id": CONFIGURATION_ID,
        "cv_protocol_version": CV_PROTOCOL_VERSION,
        "evaluation_design": EVALUATION_DESIGN,
        "evaluation_scope": EVALUATION_SCOPE,
        "oof_prediction_scope": OOF_PREDICTION_SCOPE,
        "outer_folds": N_OUTER_FOLDS,
        "layers": list(LAYERS),
        "feature_mode": "directional_base256",
        "feature_dim": FEATURE_DIM,
        "feature_cache_dtype": "float32",
        "fit_dtype": "float64",
        "target": "final_predicted_pdb_absolute_ca_distance_angstrom",
        "pair_order": "np.tril_indices(n_residues, k=-1)",
        "training_row_cap_per_protein": ROW_CAP,
        "training_sampling_algorithm": SAMPLING_ALGORITHM,
        "training_sampling_namespace": SAMPLING_NAMESPACE,
        "training_sampling_base_seed": BASE_RANDOM_SEED,
        "training_sampling_scope": "one_target_and_layer_independent_sample_per_protein",
        "training_weighting": "equal_total_weight_per_training_protein",
        "global_baseline": "sampled_fold_training_only_protein_equal_target_mean",
        "booster_seed_policy": "42 + outer_fold",
        "final_boosting_rounds": BOOSTING_ROUNDS,
        "final_fit_early_stopping": False,
        "final_fit_evaluation_set": False,
        "post_cv_all_data_refit": False,
        "raw_pair_predictions_saved": False,
        "selection_source_path": SELECTION_SOURCE_PATH,
        "selection_source_sha256": SELECTION_SOURCE_SHA256,
        "selection_source_selected_mean_validation_r2": 0.8209633288007732,
        "selection_source_runner_up_configuration_id": "S1",
        "selection_source_runner_up_mean_validation_r2": 0.8196613060608356,
        "same_cohort_preliminary_selection": True,
        "nested_selection": False,
        "fully_selection_independent": False,
        "source_runs_exposed_historical_test": True,
        "historical_test_exposure_disclosed": True,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Invalid fixed XGBoost configuration field {key}")
    expected_keys = {*expected, "xgboost_parameters", "configuration_digest"}
    if set(config) != expected_keys:
        raise ValueError("Fixed XGBoost configuration schema changed")
    expected_parameters = {
        "objective": "reg:squarederror",
        "tree_method": "hist",
        "max_depth": 6,
        "learning_rate": 0.1,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "max_bin": 256,
        "n_estimators": BOOSTING_ROUNDS,
    }
    if config.get("xgboost_parameters") != expected_parameters:
        raise ValueError("Fixed XGBoost parameters are not exactly S4/5k/700")
    forbidden = {"early_stopping_rounds", "eval_set"}
    if forbidden & set(config["xgboost_parameters"]):
        raise ValueError("Fixed XGBoost parameters contain evaluation-time controls")

    # This source artifact is available in the repository but intentionally is
    # not deployed as a new selection artifact. Its path and digest remain
    # mandatory provenance in the portable fixed configuration.
    source = REPO_ROOT / SELECTION_SOURCE_PATH
    if source.is_file():
        if sha256_file(source) != SELECTION_SOURCE_SHA256:
            raise ValueError("Frozen XGBoost selection source checksum changed")
        with source.open(newline="") as handle:
            rows = {row["config_id"]: row for row in csv.DictReader(handle)}
        selected = rows.get(CONFIGURATION_ID)
        runner_up = rows.get(str(config["selection_source_runner_up_configuration_id"]))
        if selected is None or runner_up is None:
            raise ValueError("Frozen XGBoost selection source lacks the required rows")
        if not np.isclose(
            float(selected["mean_validation_r2_across_layers"]),
            float(config["selection_source_selected_mean_validation_r2"]),
            rtol=0.0, atol=1e-15,
        ) or not np.isclose(
            float(runner_up["mean_validation_r2_across_layers"]),
            float(config["selection_source_runner_up_mean_validation_r2"]),
            rtol=0.0, atol=1e-15,
        ):
            raise ValueError("Frozen XGBoost selection values changed")
    return config


_FROZEN_CONFIGURATION = load_fixed_configuration()
CONFIGURATION = configuration_payload(_FROZEN_CONFIGURATION)
CONFIGURATION_DIGEST = str(_FROZEN_CONFIGURATION["configuration_digest"])

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
TASK_BOOSTER_PATH = "model_artifacts/booster.json"
TASK_IMPORTANCE_PATH = "model_artifacts/feature_importance.npz"
FIT_CHECKPOINT_FILENAME = "fit_checkpoint.json"
TASK_CHECKPOINT_FILENAME = "checkpoint.json"


def log(message: str) -> None:
    print(f"[{datetime.now().astimezone().isoformat(timespec='seconds')}] {message}", flush=True)


def family_output_root(csv_root: str | Path | None = None) -> Path:
    return fivefold_result_root(csv_root) / FAMILY_DIR_NAME


def task_shard_path(root: str | Path, task: FoldLayerTask) -> Path:
    return Path(root) / "layer_shards" / f"fold_{task.outer_fold}_layer_{task.layer:02d}"


def expected_task_artifacts() -> dict[str, str]:
    return {
        **{logical: relative for logical, relative in TASK_TABLES.items()},
        "task_payload": TASK_PAYLOAD_PATH,
        "booster": TASK_BOOSTER_PATH,
        "feature_importance": TASK_IMPORTANCE_PATH,
        "fit_checkpoint": FIT_CHECKPOINT_FILENAME,
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
class SampledTrainingData:
    features: np.ndarray
    targets: np.ndarray
    sample_weights: np.ndarray
    global_baseline_value: float
    global_baseline_digest: str
    sample_assignment_digest: str
    n_training_proteins: int


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)


def require_xgboost():
    try:
        import xgboost  # type: ignore
        from xgboost import XGBRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "XGBoost is required for this cluster workflow; activate ~/protein_venv on Delta"
        ) from exc
    return xgboost, XGBRegressor


def version_payload() -> dict[str, str]:
    xgboost, _ = require_xgboost()
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "xgboost": str(xgboost.__version__),
    }


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def _input_identity(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise FileNotFoundError(f"Required frozen input is missing: {path}")
    stat = path.stat()
    return {
        "path": str(path), "size_bytes": int(stat.st_size),
        "mtime_ns": int(stat.st_mtime_ns),
    }


def resolve_task_inputs(fold_bundle, layer: int) -> tuple[dict[str, dict[str, Path]], str]:
    """Resolve every fold/layer path without opening held-out inputs."""
    paths: dict[str, dict[str, Path]] = {}
    identities = []
    for protein_id in fold_bundle.eligible_protein_ids:
        entry = fold_bundle.outer_by_protein[protein_id]
        pair_path = resolve_combined_pair_block_path(protein_id, entry.source, layer)
        pdb_path = resolve_combined_pdb_path(protein_id, entry.source)
        paths[protein_id] = {"pair_block": pair_path, "pdb": pdb_path}
        identities.append({
            "protein_id": protein_id, "source": entry.source,
            "n_residues": int(entry.n_residues), "n_pairs": int(entry.n_pairs),
            "outer_fold": int(entry.outer_fold),
            "pair_block": _input_identity(pair_path), "pdb": _input_identity(pdb_path),
        })
    return paths, semantic_digest(identities)


def load_target_records(
    fold_bundle,
    paths: Mapping[str, Mapping[str, Path]],
    protein_ids: Sequence[str],
) -> list[ProteinLayerRecord]:
    """Open PDB targets for only the explicitly requested protein IDs."""
    requested = tuple(sorted(map(str, protein_ids)))
    if not requested or len(set(requested)) != len(requested):
        raise ValueError("Target loading requires nonempty unique protein IDs")
    records = []
    for index, protein_id in enumerate(requested, start=1):
        entry = fold_bundle.outer_by_protein[protein_id]
        ca = extract_ca_coordinates(paths[protein_id]["pdb"])
        coordinates = np.asarray(ca["coordinates"], dtype=np.float64)
        residues = tuple(ca["residues"])
        if coordinates.shape != (int(entry.n_residues), 3) or len(residues) != entry.n_residues:
            raise ValueError(f"Frozen residue count mismatch for {protein_id}")
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
        if index % 50 == 0 or index == len(requested):
            log(f"loaded targets for {index}/{len(requested)} requested proteins")
    return records


def load_feature_block(record: ProteinLayerRecord) -> np.ndarray:
    """Load one full base256 block into the temporary float32 cache."""
    if record.synthetic_features is not None:
        values = np.asarray(record.synthetic_features)
    else:
        if record.pair_block_path is None:
            raise ValueError(f"No pair tensor path for {record.protein_id}")
        pair_block = np.load(str(record.pair_block_path))
        try:
            expected = (record.n_residues, record.n_residues, FEATURE_DIM // 2)
            if pair_block.shape != expected:
                raise ValueError(f"Pair tensor shape mismatch for {record.protein_id}")
            values = extract_directional_base256(pair_block, record.pair_i, record.pair_j)
        finally:
            del pair_block
    if values.shape != (record.targets.size, FEATURE_DIM) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid base256 features for {record.protein_id}")
    return np.ascontiguousarray(values, dtype=np.float32)


def sampling_manifest_digest(fold_bundle) -> str:
    rows = []
    for protein_id in fold_bundle.eligible_protein_ids:
        entry = fold_bundle.outer_by_protein[protein_id]
        rows.append({
            "protein_id": protein_id,
            "n_pairs": int(entry.n_pairs),
            "n_selected_pairs": min(int(entry.n_pairs), ROW_CAP),
            "base_seed": BASE_RANDOM_SEED,
            "derived_seed": deterministic_sample_seed(
                protein_id, base_seed=BASE_RANDOM_SEED,
            ),
            "assignment_digest": sampling_assignment_digest(
                entry.n_pairs, protein_id, ROW_CAP, base_seed=BASE_RANDOM_SEED,
            ),
        })
    return semantic_digest(rows)


def materialize_sampled_training(
    records: Sequence[ProteinLayerRecord],
    *,
    feature_loader: Callable[[ProteinLayerRecord], np.ndarray] = load_feature_block,
) -> SampledTrainingData:
    ordered = sorted(records, key=lambda record: record.protein_id)
    if not ordered or len({record.protein_id for record in ordered}) != len(ordered):
        raise ValueError("Training records must have unique protein IDs")
    selected_by_id = {
        record.protein_id: deterministic_sample_indices(
            record.targets.size, record.protein_id, ROW_CAP, base_seed=BASE_RANDOM_SEED,
        )
        for record in ordered
    }
    total = sum(indices.size for indices in selected_by_id.values())
    features = np.empty((total, FEATURE_DIM), dtype=np.float64)
    targets = np.empty(total, dtype=np.float64)
    sample_weights = np.empty(total, dtype=np.float64)
    target_blocks = {}
    assignment_rows = []
    cursor = 0
    for index, record in enumerate(ordered, start=1):
        selected = selected_by_id[record.protein_id]
        stop = cursor + selected.size
        block = feature_loader(record)
        features[cursor:stop] = np.asarray(block[selected], dtype=np.float64)
        del block
        selected_targets = np.asarray(record.targets[selected], dtype=np.float64)
        targets[cursor:stop] = selected_targets
        sample_weights[cursor:stop] = 1.0 / float(selected.size)
        target_blocks[record.protein_id] = selected_targets
        assignment_rows.append({
            "protein_id": record.protein_id,
            "n_pairs": int(record.targets.size),
            "n_selected_pairs": int(selected.size),
            "assignment_digest": sampling_assignment_digest(
                record.targets.size, record.protein_id, ROW_CAP,
                base_seed=BASE_RANDOM_SEED,
            ),
        })
        cursor = stop
        if index % 25 == 0 or index == len(ordered):
            log(f"materialized samples for {index}/{len(ordered)} training proteins")
    if (
        cursor != total or total < 1 or not np.all(np.isfinite(features))
        or not np.all(np.isfinite(targets)) or not np.all(np.isfinite(sample_weights))
    ):
        raise ValueError("Sampled XGBoost training matrix is incomplete or nonfinite")
    baseline = fit_train_only_baseline(target_blocks)
    return SampledTrainingData(
        features=features, targets=targets, sample_weights=sample_weights,
        global_baseline_value=baseline.value,
        global_baseline_digest=baseline.baseline_digest,
        sample_assignment_digest=semantic_digest(assignment_rows),
        n_training_proteins=len(ordered),
    )


def build_fixed_estimator(outer_fold: int, n_jobs: int):
    _, estimator_type = require_xgboost()
    fold = int(outer_fold)
    if fold not in range(N_OUTER_FOLDS):
        raise ValueError("XGBoost outer fold must be 0..4")
    parameters = dict(CONFIGURATION["xgboost_parameters"])
    parameters["random_state"] = BASE_RANDOM_SEED + fold
    return estimator_type(**parameters, n_jobs=int(n_jobs))


def booster_round_count(model) -> int:
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    value = booster.num_boosted_rounds()
    return int(value)


def fit_fixed_estimator(model, training: SampledTrainingData) -> None:
    model.fit(
        training.features, training.targets,
        sample_weight=training.sample_weights,
    )
    rounds = booster_round_count(model)
    if rounds != BOOSTING_ROUNDS:
        raise RuntimeError(
            f"Fixed XGBoost fit produced {rounds} rounds; expected {BOOSTING_ROUNDS}"
        )


def predict_fixed_estimator(model, features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != FEATURE_DIM or not np.all(np.isfinite(values)):
        raise ValueError("XGBoost prediction features are invalid")
    predictions = np.asarray(model.predict(values), dtype=np.float64)
    if predictions.shape != (values.shape[0],) or not np.all(np.isfinite(predictions)):
        raise ValueError("XGBoost produced incomplete or nonfinite predictions")
    return predictions


def feature_importance_arrays(model) -> tuple[np.ndarray, np.ndarray]:
    booster = model.get_booster() if hasattr(model, "get_booster") else model
    gain_map = booster.get_score(importance_type="gain")
    split_map = booster.get_score(importance_type="weight")
    for mapping in (gain_map, split_map):
        for key, value in mapping.items():
            if (
                not key.startswith("f") or not key[1:].isdigit()
                or int(key[1:]) >= FEATURE_DIM or not np.isfinite(float(value))
                or float(value) < 0
            ):
                raise ValueError(f"Unexpected booster feature-importance item: {key!r}")
    gain = np.asarray([gain_map.get(f"f{i}", 0.0) for i in range(FEATURE_DIM)], dtype=np.float64)
    split = np.asarray([split_map.get(f"f{i}", 0.0) for i in range(FEATURE_DIM)], dtype=np.float64)
    return gain, split


def atomic_save_booster(model, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}.", suffix=".json", dir=destination.parent,
    )
    os.close(descriptor)
    try:
        saver = model if hasattr(model, "save_model") else model.get_booster()
        saver.save_model(temporary_name)
        os.replace(temporary_name, destination)
    except Exception:
        Path(temporary_name).unlink(missing_ok=True)
        raise


def write_importance_artifact(
    path: Path, gain: np.ndarray, split_count: np.ndarray, task: FoldLayerTask,
) -> None:
    write_v2_npz(
        path,
        artifact_version=np.asarray(IMPORTANCE_ARTIFACT_VERSION),
        analysis_name=np.asarray(ANALYSIS_NAME), model_family=np.asarray(MODEL_FAMILY),
        layer=np.asarray(task.layer, dtype=np.int16),
        outer_fold=np.asarray(task.outer_fold, dtype=np.int16),
        configuration_digest=np.asarray(CONFIGURATION_DIGEST),
        gain=np.asarray(gain, dtype=np.float64),
        split_count=np.asarray(split_count, dtype=np.float64),
    )


def load_importance_artifact(
    path: str | Path, *, expected_task: FoldLayerTask | None = None,
) -> dict[str, object]:
    expected_keys = {
        "artifact_version", "analysis_name", "model_family", "layer", "outer_fold",
        "configuration_digest", "gain", "split_count",
    }
    with np.load(path, allow_pickle=False) as archive:
        if set(archive.files) != expected_keys:
            raise ValueError(f"Feature-importance artifact schema mismatch: {path}")
        result = {name: archive[name].copy() for name in archive.files}
    scalars = {
        key: np.asarray(result[key]).item()
        for key in (
            "artifact_version", "analysis_name", "model_family", "layer", "outer_fold",
            "configuration_digest",
        )
    }
    if (
        scalars["artifact_version"] != IMPORTANCE_ARTIFACT_VERSION
        or scalars["analysis_name"] != ANALYSIS_NAME
        or scalars["model_family"] != MODEL_FAMILY
        or scalars["configuration_digest"] != CONFIGURATION_DIGEST
    ):
        raise ValueError("Feature-importance artifact identity mismatch")
    if expected_task is not None and (
        int(scalars["layer"]), int(scalars["outer_fold"])
    ) != (expected_task.layer, expected_task.outer_fold):
        raise ValueError("Feature-importance task identity mismatch")
    for key in ("gain", "split_count"):
        values = np.asarray(result[key], dtype=np.float64)
        if values.shape != (FEATURE_DIM,) or not np.all(np.isfinite(values)) or np.any(values < 0):
            raise ValueError("Feature-importance arrays are invalid")
        result[key] = values
    return {**result, **scalars}


def task_invariants(
    task: FoldLayerTask,
    *,
    fold_digests: Mapping[str, object],
    input_digest: str,
    sample_manifest_digest_value: str,
) -> dict[str, object]:
    return {
        "analysis_name": ANALYSIS_NAME, "model_family": MODEL_FAMILY,
        "task_id": task.task_id, "outer_fold": task.outer_fold, "layer": task.layer,
        "configuration_digest": CONFIGURATION_DIGEST,
        "cohort_digest": fold_digests["cohort_digest"],
        "outer_fold_digest": fold_digests["outer_fold_digest"],
        "pair_fold_manifest_digest": fold_digests["pair_fold_manifest_digest"],
        "sampling_manifest_digest": str(sample_manifest_digest_value),
        "input_identity_digest": str(input_digest),
    }


def fit_checkpoint_invariants(
    task: FoldLayerTask,
    task_identity: Mapping[str, object],
    *,
    training_sample_digest: str,
    xgboost_version: str,
) -> dict[str, object]:
    return {
        **dict(task_identity),
        "training_sample_assignment_digest": str(training_sample_digest),
        "random_seed": BASE_RANDOM_SEED + task.outer_fold,
        "final_boosting_rounds": BOOSTING_ROUNDS,
        "xgboost_version": str(xgboost_version),
    }


def _fit_or_resume(
    *,
    shard: Path,
    task: FoldLayerTask,
    training: SampledTrainingData,
    fit_invariants: Mapping[str, object],
    n_jobs: int,
    estimator_builder: Callable[[int, int], object],
) -> tuple[object, bool]:
    booster_path = shard / TASK_BOOSTER_PATH
    checkpoint_path = shard / FIT_CHECKPOINT_FILENAME
    if checkpoint_path.is_file():
        validate_resume_checkpoint(
            checkpoint_path, expected_invariants=fit_invariants,
            expected_artifact_names={"booster"},
        )
        model = estimator_builder(task.outer_fold, n_jobs)
        model.load_model(str(booster_path))
        if booster_round_count(model) != BOOSTING_ROUNDS:
            raise ValueError("Checkpointed booster does not contain exactly 700 rounds")
        log(f"task {task.task_id}: reused checksummed 700-round booster checkpoint")
        return model, True
    log(f"task {task.task_id}: fitting S4 for exactly {BOOSTING_ROUNDS} rounds")
    model = estimator_builder(task.outer_fold, n_jobs)
    fit_fixed_estimator(model, training)
    atomic_save_booster(model, booster_path)
    commit_checkpoint(
        checkpoint_path, invariants=fit_invariants,
        artifacts={"booster": TASK_BOOSTER_PATH},
    )
    return model, False


def execute_fold_layer_task(
    *,
    task: FoldLayerTask,
    training_records: Sequence[ProteinLayerRecord],
    heldout_protein_ids: Sequence[str],
    heldout_loader: Callable[[], Sequence[ProteinLayerRecord]],
    fold_digests: Mapping[str, object],
    input_digest: str,
    sample_manifest_digest_value: str,
    root: str | Path,
    feature_loader: Callable[[ProteinLayerRecord], np.ndarray] = load_feature_block,
    estimator_builder: Callable[[int, int], object] = build_fixed_estimator,
    versions: Mapping[str, str] | None = None,
    n_jobs: int = 1,
) -> Path:
    """Fit/checkpoint a booster, then open and evaluate its held-out fold."""
    started = time.monotonic()
    shard = task_shard_path(root, task)
    invariants = task_invariants(
        task, fold_digests=fold_digests, input_digest=input_digest,
        sample_manifest_digest_value=sample_manifest_digest_value,
    )
    checkpoint = shard / TASK_CHECKPOINT_FILENAME
    if checkpoint.is_file():
        validate_resume_checkpoint(
            checkpoint, expected_invariants=invariants,
            expected_artifact_names=set(expected_task_artifacts()),
        )
        log(f"task {task.task_id}: valid checkpoint; skipping")
        return shard

    training_ids = tuple(sorted(record.protein_id for record in training_records))
    heldout_ids = tuple(sorted(map(str, heldout_protein_ids)))
    if (
        not training_ids or not heldout_ids
        or len(set(training_ids)) != len(training_ids)
        or len(set(heldout_ids)) != len(heldout_ids)
        or set(training_ids) & set(heldout_ids)
    ):
        raise ValueError("Training and held-out protein IDs must be nonempty and disjoint")
    if any(record.outer_fold == task.outer_fold for record in training_records):
        raise ValueError("Training record belongs to the held-out outer fold")

    log(f"task {task.task_id}: materializing deterministic 5k training samples")
    training = materialize_sampled_training(
        training_records, feature_loader=feature_loader,
    )
    resolved_versions = dict(version_payload() if versions is None else versions)
    if not str(resolved_versions.get("xgboost", "")):
        raise ValueError("XGBoost version is required for booster provenance")
    fit_invariants = fit_checkpoint_invariants(
        task, invariants,
        training_sample_digest=training.sample_assignment_digest,
        xgboost_version=str(resolved_versions["xgboost"]),
    )
    model, reused = _fit_or_resume(
        shard=shard, task=task, training=training,
        fit_invariants=fit_invariants, n_jobs=n_jobs,
        estimator_builder=estimator_builder,
    )
    # This is the locked phase boundary: no held-out PDB or pair tensor may be
    # opened before the native booster and its checksum manifest both exist.
    validate_resume_checkpoint(
        shard / FIT_CHECKPOINT_FILENAME, expected_invariants=fit_invariants,
        expected_artifact_names={"booster"},
    )
    training_predictions = predict_fixed_estimator(model, training.features)
    training_row = build_training_metrics_row(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM, layer=task.layer,
        evaluation_design=EVALUATION_DESIGN,
        training_scope="sampled_pairs_four_outer_training_folds",
        y_true=training.targets, predictions=training_predictions,
        fit_status="fitted", outer_fold=task.outer_fold,
        n_training_proteins=training.n_training_proteins,
    )
    status_row = build_model_status_row(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM, layer=task.layer, outer_fold=task.outer_fold,
        fit_status="fitted", support_flags="fit_checkpoint_reused" if reused else "",
        n_training_proteins=training.n_training_proteins,
        n_training_pairs=training.targets.size,
        selected_configuration_id=CONFIGURATION_ID,
        final_boosting_rounds=BOOSTING_ROUNDS, training_row_cap=ROW_CAP,
        random_seed=BASE_RANDOM_SEED + task.outer_fold,
    )
    gain, split_count = feature_importance_arrays(model)
    global_baseline_value = training.global_baseline_value
    global_baseline_digest = training.global_baseline_digest
    n_training_pairs = int(training.targets.size)
    del training_predictions, training
    gc.collect()

    log(f"task {task.task_id}: loading held-out fold after booster checkpoint")
    heldout_records = list(heldout_loader())
    by_id = {record.protein_id: record for record in heldout_records}
    if set(by_id) != set(heldout_ids) or len(by_id) != len(heldout_records):
        raise ValueError("Held-out loader did not return the exact requested fold")
    if any(record.outer_fold != task.outer_fold for record in heldout_records):
        raise ValueError("Held-out record has the wrong outer-fold label")

    oof = OOFComparisonAccumulator()
    sequence = OOFSubgroupAccumulator("sequence_separation", SEQUENCE_BIN_LABELS)
    spatial = OOFSubgroupAccumulator("spatial_distance", SPATIAL_DISTANCE_BIN_LABELS)
    per_rows = []
    geometry_rows = []
    for index, protein_id in enumerate(heldout_ids, start=1):
        record = by_id[protein_id]
        features = feature_loader(record)
        predictions = predict_fixed_estimator(model, features)
        del features
        per_rows.append(build_per_protein_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=task.layer, protein_id=protein_id,
            source=record.source, outer_fold=task.outer_fold,
            evaluation_design=EVALUATION_DESIGN,
            oof_prediction_scope=OOF_PREDICTION_SCOPE,
            n_residues=record.n_residues, y_true=record.targets,
            predictions=predictions, prediction_status="complete",
        ))
        oof.update_protein(
            protein_id, task.outer_fold, record.targets, predictions,
            global_baseline_value, stable_fit=True,
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
    if any(row["geometry_status"] != "complete" for row in geometry_rows):
        raise RuntimeError(f"Task {task.task_id} has incomplete held-out geometry")

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
        n_evaluated_proteins=len(heldout_ids), n_fitted_pairs=n_training_pairs,
        n_evaluated_pairs=oof.n_evaluated_pairs, elapsed_seconds=elapsed,
        peak_rss_mb=peak_rss_mb(),
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
    write_importance_artifact(shard / TASK_IMPORTANCE_PATH, gain, split_count, task)
    atomic_write_json(shard / TASK_PAYLOAD_PATH, {
        "task_payload_version": TASK_PAYLOAD_VERSION,
        "task_id": task.task_id, "outer_fold": task.outer_fold, "layer": task.layer,
        "configuration_digest": CONFIGURATION_DIGEST,
        "sampling_manifest_digest": sample_manifest_digest_value,
        "training_sample_assignment_digest": fit_invariants[
            "training_sample_assignment_digest"
        ],
        "global_baseline_digest": global_baseline_digest,
        "global_baseline_value": global_baseline_value,
        "random_seed": BASE_RANDOM_SEED + task.outer_fold,
        "final_boosting_rounds": BOOSTING_ROUNDS,
        "versions": resolved_versions,
        "oof_accumulator": oof.to_payload(),
        "sequence_subgroups": sequence.to_payload(),
        "spatial_subgroups": spatial.to_payload(),
    })
    commit_checkpoint(
        checkpoint, invariants=invariants, artifacts=expected_task_artifacts(),
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
    if task_id is None:
        raw = os.environ.get("SLURM_ARRAY_TASK_ID")
        if raw is None:
            raise RuntimeError("Provide task_id or set SLURM_ARRAY_TASK_ID")
        task_id = int(raw)
    task = map_fold_layer_task(task_id)
    fold_bundle = load_cv_folds()
    destination = family_output_root() if root is None else Path(root)
    paths, input_digest = resolve_task_inputs(fold_bundle, task.layer)
    sample_digest = sampling_manifest_digest(fold_bundle)
    invariants = task_invariants(
        task, fold_digests=fold_bundle.fold_digests, input_digest=input_digest,
        sample_manifest_digest_value=sample_digest,
    )
    checkpoint = task_shard_path(destination, task) / TASK_CHECKPOINT_FILENAME
    if checkpoint.is_file():
        validate_resume_checkpoint(
            checkpoint, expected_invariants=invariants,
            expected_artifact_names=set(expected_task_artifacts()),
        )
        log(f"task {task.task_id}: valid checkpoint; no inputs opened")
        return checkpoint.parent
    training_ids = fold_bundle.training_protein_ids(task.outer_fold)
    heldout_ids = fold_bundle.protein_ids_for_outer_fold(task.outer_fold)
    try:
        # Fail with the explicit cluster-environment message before loading the
        # large training inputs when XGBoost is unavailable.
        versions = version_payload()
        # Only the four training folds are opened before booster checkpointing.
        training_records = load_target_records(fold_bundle, paths, training_ids)
        return execute_fold_layer_task(
            task=task, training_records=training_records,
            heldout_protein_ids=heldout_ids,
            heldout_loader=lambda: load_target_records(fold_bundle, paths, heldout_ids),
            fold_digests=fold_bundle.fold_digests, input_digest=input_digest,
            sample_manifest_digest_value=sample_digest, root=destination,
            versions=versions,
            n_jobs=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
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
    "ANALYSIS_NAME", "BASE_RANDOM_SEED", "BOOSTING_ROUNDS", "CONFIGURATION",
    "CONFIGURATION_DIGEST", "CONFIGURATION_ID", "CONFIG_PATH", "EVALUATION_DESIGN",
    "FAMILY_DIR_NAME", "FIT_CHECKPOINT_FILENAME", "IMPORTANCE_ARTIFACT_VERSION",
    "MODEL_FAMILY", "OOF_PREDICTION_SCOPE", "ProteinLayerRecord", "ROW_CAP",
    "SELECTION_SOURCE_PATH", "SELECTION_SOURCE_SHA256", "SampledTrainingData",
    "TASK_BOOSTER_PATH", "TASK_CHECKPOINT_FILENAME", "TASK_IMPORTANCE_PATH",
    "TASK_PAYLOAD_PATH", "TASK_PAYLOAD_VERSION", "TASK_TABLES", "_fit_or_resume",
    "atomic_save_booster", "booster_round_count", "build_fixed_estimator",
    "code_revision", "configuration_payload", "execute_fold_layer_task",
    "expected_task_artifacts", "family_output_root", "feature_importance_arrays",
    "fit_checkpoint_invariants", "fit_fixed_estimator", "load_feature_block",
    "load_fixed_configuration", "load_importance_artifact", "load_target_records",
    "log", "materialize_sampled_training", "peak_rss_mb", "predict_fixed_estimator",
    "read_csv_strict", "resolve_task_inputs", "run_fold_layer_task",
    "sampling_manifest_digest", "task_invariants", "task_shard_path", "version_payload",
    "write_importance_artifact",
]

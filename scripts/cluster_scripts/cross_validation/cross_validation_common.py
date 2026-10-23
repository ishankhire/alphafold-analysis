#!/usr/bin/env python3
"""Common engine for protocol-2.0 fivefold regression workflows.

This module owns fold/task mechanics, train-only preprocessing, mergeable OOF
statistics, v2 output rows, replay indexes, and checkpoint validation.  It does
not fit a model family or launch a cluster job; those responsibilities belong
to the later family-specific stages.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Sequence

import numpy as np

from cluster_regression_helpers import analyze_protein_geometry, TMScoreError
from regression_output_helpers import (
    FIVEFOLD_OUTPUT_SCHEMA_VERSION,
    GEOMETRY_METRIC_NAMES,
    GEOMETRY_METRIC_UNITS,
    atomic_savez_compressed,
    atomic_write_csv,
    atomic_write_json,
    sha256_file,
)
from regression_protocol import (
    ELIGIBLE_COHORT_DIGEST,
    FEATURE_NAMES,
    FLOAT_EPS,
    FOLD_DIGESTS_PATH,
    LAYERS,
    MASTER_MANIFEST_PATH,
    N_OUTER_FOLDS,
    N_PAIR_FOLDS,
    OUTER_FOLD_MANIFEST_DIGEST,
    OUTER_FOLD_MANIFEST_PATH,
    PAIR_FOLD_MANIFEST_DIGEST,
    PAIR_FOLD_MANIFEST_PATH,
    PROTOCOL_VERSION,
    ProteinEqualFeatureMoments,
    canonical_json_bytes,
    load_fold_digests,
    load_master_manifest,
    load_outer_fold_manifest,
    load_pair_fold_manifest,
    protein_equal_weights,
    stable_seed,
    strict_lower_triangle_indices,
)


OUTPUT_SCHEMA_VERSION = FIVEFOLD_OUTPUT_SCHEMA_VERSION
CV_PROTOCOL_VERSION = PROTOCOL_VERSION
RESULT_ROOT_NAME = "regression_5fold_v1"
METADATA_DIR_NAME = "metadata"
CHECKPOINT_MANIFEST_VERSION = "regression_5fold_checkpoint_v1"
REPLAY_INDEX_VERSION = "regression_5fold_replay_index_v1"
SAMPLING_ALGORITHM = "stable_sha256_numpy_permutation_sorted_v1"
SAMPLING_NAMESPACE = "regression_5fold_training_rows_v1"
GLOBAL_BASELINE_TYPE = "global_fold_training_mean"
EVALUATION_SCOPE = "oof_all_pairs"
EVALUATION_DESIGNS = ("protein_outer_5fold", "within_protein_pair_5fold")

RUN_MANIFEST_FILENAME = "run_manifest.json"
PER_PROTEIN_FILENAME = "per_protein.csv"
POOLED_METRICS_FILENAME = "pooled_metrics.csv"
FOLD_METRICS_FILENAME = "fold_metrics.csv"
FOLD_SUMMARY_FILENAME = "fold_summary.csv"
TRAINING_METRICS_FILENAME = "training_metrics.csv"
STABLE_FIT_METRICS_FILENAME = "stable_fit_metrics.csv"
MODEL_STATUS_FILENAME = "model_status.csv"
RUNTIME_FILENAME = "runtime.csv"
GEOMETRY_METRICS_FILENAME = "geometry_metrics.csv"
GEOMETRY_SUMMARY_FILENAME = "geometry_summary.csv"
MODEL_STABILITY_FILENAME = "model_stability.csv"
SUBGROUP_SEQUENCE_FILENAME = "subgroup_metrics/by_sequence_separation.csv"
SUBGROUP_SPATIAL_FILENAME = "subgroup_metrics/by_spatial_distance.csv"
REPLAY_INDEX_FILENAME = "model_artifacts/replay_index.csv"
FEATURE_IMPORTANCE_FILENAME = "model_artifacts/feature_importance.csv"

PER_PROTEIN_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_id", "source",
    "cv_protocol_version", "evaluation_design", "outer_fold", "evaluation_scope",
    "oof_prediction_scope", "prediction_status", "n_residues", "n_protein_pairs",
    "n_expected_pairs", "n_evaluated_pairs", "pair_coverage", "target_variance", "r2",
    "mae", "rmse", "bias_pred_minus_true", "mean_true_distance",
    "mean_predicted_distance",
)

_METRIC_IDENTITY_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "cv_protocol_version",
    "evaluation_design", "evaluation_scope", "oof_prediction_scope",
)
_METRIC_SUPPORT_COLUMNS = (
    "n_proteins", "n_proteins_with_finite_r2", "n_expected_pairs",
    "n_evaluated_pairs", "pair_coverage",
)
_METRIC_VALUE_COLUMNS = (
    "pooled_r2", "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true",
    "global_baseline_type", "global_baseline_pooled_r2", "global_baseline_pooled_mae",
    "global_baseline_pooled_rmse", "global_baseline_pooled_bias_pred_minus_true",
    "conditional_baseline_type", "conditional_baseline_pooled_r2",
    "conditional_baseline_pooled_mae", "conditional_baseline_pooled_rmse",
    "conditional_baseline_pooled_bias_pred_minus_true",
)
POOLED_METRICS_COLUMNS = (
    *_METRIC_IDENTITY_COLUMNS, *_METRIC_SUPPORT_COLUMNS, *_METRIC_VALUE_COLUMNS,
)
FOLD_METRICS_COLUMNS = (
    *_METRIC_IDENTITY_COLUMNS, "outer_fold", *_METRIC_SUPPORT_COLUMNS,
    *_METRIC_VALUE_COLUMNS,
)
FOLD_SUMMARY_COLUMNS = (
    *_METRIC_IDENTITY_COLUMNS, "fold_metric", "n_folds", "mean", "std",
)
TRAINING_METRICS_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "cv_protocol_version",
    "evaluation_design", "outer_fold", "protein_id", "pair_fold", "sequence_gate",
    "training_scope", "fit_status", "support_flags", "n_training_proteins",
    "n_training_pairs", "target_variance", "r2", "mae", "rmse",
    "bias_pred_minus_true", "mean_true_distance", "mean_predicted_distance",
    "design_rank", "largest_singular_value", "smallest_singular_value",
    "condition_number",
)
STABLE_FIT_METRICS_COLUMNS = (
    *_METRIC_IDENTITY_COLUMNS, "stability_filter", "n_proteins_total",
    "n_proteins_included", *_METRIC_SUPPORT_COLUMNS, *_METRIC_VALUE_COLUMNS,
)
MODEL_STATUS_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "cv_protocol_version",
    "outer_fold", "protein_id", "pair_fold", "sequence_gate", "fit_status",
    "support_flags", "n_training_proteins", "n_training_pairs", "design_rank",
    "largest_singular_value", "smallest_singular_value", "condition_number",
    "selected_configuration_id", "final_boosting_rounds", "training_row_cap",
    "random_seed",
)
RUNTIME_COLUMNS = (
    "analysis_name", "model_family", "task_id", "outer_fold", "layer",
    "completion_status", "n_fitted_proteins", "n_evaluated_proteins",
    "n_fitted_pairs", "n_evaluated_pairs", "elapsed_seconds", "peak_rss_mb",
)
GEOMETRY_METRICS_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_id", "source",
    "cv_protocol_version", "evaluation_design", "outer_fold", "evaluation_scope",
    "oof_prediction_scope", "n_residues", "n_protein_pairs", "pair_coverage",
    "geometry_status", "geometry_exclusion_reason", *GEOMETRY_METRIC_NAMES,
)
GEOMETRY_SUMMARY_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "cv_protocol_version",
    "evaluation_design", "evaluation_scope", "oof_prediction_scope", "geometry_metric",
    "units", "n_proteins_attempted", "n_proteins_included", "n_proteins_excluded",
    "mean", "std", "median", "q25", "q75",
)
SUBGROUP_METRICS_COLUMNS = (
    *_METRIC_IDENTITY_COLUMNS, "category_scope", "category_label", "n_proteins",
    "n_expected_pairs", "n_evaluated_pairs", "pair_coverage", "support_status",
    "pooled_r2", "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true",
)
MODEL_STABILITY_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_id",
    "sequence_gate", "feature_index", "feature_name", "n_models",
    "coefficient_standardized_mean", "coefficient_standardized_std",
    "coefficient_standardized_min", "coefficient_standardized_max",
    "sign_consistency_fraction",
)
REPLAY_INDEX_COLUMNS = (
    "replay_index_version", "analysis_name", "model_family", "layer", "outer_fold",
    "protein_id", "pair_fold", "sequence_gate", "artifact_type", "artifact_path",
    "artifact_sha256", "configuration_digest", "scaler_digest", "status",
)
FEATURE_IMPORTANCE_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "cv_protocol_version",
    "feature_index", "feature_name", "n_outer_folds", "gain_mean", "gain_std",
    "split_count_mean", "split_count_std",
)

V2_CSV_SCHEMAS = MappingProxyType({
    PER_PROTEIN_FILENAME: PER_PROTEIN_COLUMNS,
    POOLED_METRICS_FILENAME: POOLED_METRICS_COLUMNS,
    FOLD_METRICS_FILENAME: FOLD_METRICS_COLUMNS,
    FOLD_SUMMARY_FILENAME: FOLD_SUMMARY_COLUMNS,
    TRAINING_METRICS_FILENAME: TRAINING_METRICS_COLUMNS,
    STABLE_FIT_METRICS_FILENAME: STABLE_FIT_METRICS_COLUMNS,
    MODEL_STATUS_FILENAME: MODEL_STATUS_COLUMNS,
    RUNTIME_FILENAME: RUNTIME_COLUMNS,
    GEOMETRY_METRICS_FILENAME: GEOMETRY_METRICS_COLUMNS,
    GEOMETRY_SUMMARY_FILENAME: GEOMETRY_SUMMARY_COLUMNS,
    MODEL_STABILITY_FILENAME: MODEL_STABILITY_COLUMNS,
    SUBGROUP_SEQUENCE_FILENAME: SUBGROUP_METRICS_COLUMNS,
    SUBGROUP_SPATIAL_FILENAME: SUBGROUP_METRICS_COLUMNS,
    REPLAY_INDEX_FILENAME: REPLAY_INDEX_COLUMNS,
    FEATURE_IMPORTANCE_FILENAME: FEATURE_IMPORTANCE_COLUMNS,
})


from canonical_storage import result_root


def semantic_digest(value: object) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible values."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def numeric_array_digest(values: np.ndarray | Sequence[float]) -> str:
    """Hash a numeric array with explicit shape and little-endian dtype metadata."""
    array = np.asarray(values)
    if array.dtype.kind not in "biufc" or array.dtype.hasobject:
        raise TypeError("Only non-object numeric arrays have a canonical digest")
    dtype = array.dtype.newbyteorder("<")
    canonical = np.ascontiguousarray(array.astype(dtype, copy=False))
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"dtype": dtype.str, "shape": list(canonical.shape)}))
    digest.update(canonical.tobytes(order="C"))
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def fivefold_result_root(csv_root: str | Path | None = None) -> Path:
    """Resolve the local or Delta v2 result root without creating it."""
    return result_root(csv_root)


@dataclass(frozen=True)
class FoldLayerTask:
    task_id: int
    outer_fold: int
    layer: int


def _normalized_task_layers(layers: Sequence[int]) -> tuple[int, ...]:
    canonical_layers = tuple(int(layer) for layer in layers)
    if not canonical_layers or len(set(canonical_layers)) != len(canonical_layers):
        raise ValueError("Task mapping requires unique layers")
    return canonical_layers


def map_fold_layer_task(
    task_id: int,
    *,
    layers: Sequence[int] = LAYERS,
    n_outer_folds: int = N_OUTER_FOLDS,
) -> FoldLayerTask:
    """Map a fold-major array ID to one outer-fold/layer task."""
    canonical_layers = _normalized_task_layers(layers)
    fold_count = int(n_outer_folds)
    if fold_count < 1:
        raise ValueError("Task mapping requires at least one outer fold")
    task = int(task_id)
    n_tasks = fold_count * len(canonical_layers)
    if task < 0 or task >= n_tasks:
        raise ValueError(f"Fold/layer task must be in 0..{n_tasks - 1}; got {task_id}")
    outer_fold, layer_offset = divmod(task, len(canonical_layers))
    return FoldLayerTask(task, outer_fold, canonical_layers[layer_offset])


def fold_layer_task_id(
    outer_fold: int,
    layer: int,
    *,
    layers: Sequence[int] = LAYERS,
    n_outer_folds: int = N_OUTER_FOLDS,
) -> int:
    canonical_layers = _normalized_task_layers(layers)
    fold_count = int(n_outer_folds)
    if fold_count < 1:
        raise ValueError("Task mapping requires at least one outer fold")
    fold = int(outer_fold)
    if fold < 0 or fold >= fold_count:
        raise ValueError(f"outer_fold must be in 0..{fold_count - 1}")
    try:
        layer_offset = canonical_layers.index(int(layer))
    except ValueError as exc:
        raise ValueError(f"Layer {layer} is not in the requested task mapping") from exc
    return fold * len(canonical_layers) + layer_offset


def map_layer_task(task_id: int, *, layers: Sequence[int] = LAYERS) -> int:
    canonical_layers = _normalized_task_layers(layers)
    task = int(task_id)
    if task < 0 or task >= len(canonical_layers):
        raise ValueError(f"Layer task must be in 0..{len(canonical_layers) - 1}; got {task_id}")
    return canonical_layers[task]


@dataclass(frozen=True)
class CVFoldBundle:
    """Validated frozen fold metadata and convenient protein-ID indexes."""

    outer_entries: tuple[object, ...]
    pair_entries: tuple[object, ...]
    fold_digests: Mapping[str, object]
    outer_by_protein: Mapping[str, object]
    pair_by_protein: Mapping[str, object]

    @property
    def eligible_protein_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self.outer_by_protein))

    def protein_ids_for_outer_fold(self, outer_fold: int) -> tuple[str, ...]:
        fold = int(outer_fold)
        if fold not in range(N_OUTER_FOLDS):
            raise ValueError(f"outer_fold must be in 0..{N_OUTER_FOLDS - 1}")
        return tuple(sorted(
            protein_id for protein_id, entry in self.outer_by_protein.items()
            if int(entry.outer_fold) == fold
        ))

    def training_protein_ids(self, heldout_fold: int) -> tuple[str, ...]:
        heldout = set(self.protein_ids_for_outer_fold(heldout_fold))
        return tuple(protein_id for protein_id in self.eligible_protein_ids if protein_id not in heldout)


def load_cv_folds(
    *,
    metadata_dir: str | Path | None = None,
    master_manifest_path: str | Path = MASTER_MANIFEST_PATH,
) -> CVFoldBundle:
    """Load and cross-validate the master, outer-fold, pair-fold, and digest files."""
    if metadata_dir is None and (os.environ.get("CSV_ROOT") or (result_root() / "metadata").is_dir()):
        metadata_dir = fivefold_result_root() / METADATA_DIR_NAME
    if metadata_dir is None:
        outer_path = OUTER_FOLD_MANIFEST_PATH
        pair_path = PAIR_FOLD_MANIFEST_PATH
        digest_path = FOLD_DIGESTS_PATH
    else:
        root = Path(metadata_dir)
        outer_path = root / OUTER_FOLD_MANIFEST_PATH.name
        pair_path = root / PAIR_FOLD_MANIFEST_PATH.name
        digest_path = root / FOLD_DIGESTS_PATH.name
    master = load_master_manifest(master_manifest_path)
    outer = load_outer_fold_manifest(outer_path, master_entries=master)
    pairs = load_pair_fold_manifest(pair_path, master_entries=master)
    digests = load_fold_digests(
        digest_path, master_entries=master, outer_entries=outer, pair_entries=pairs,
    )
    outer_by_id = {entry.protein_id: entry for entry in outer}
    pair_by_id = {entry.protein_id: entry for entry in pairs}
    if set(outer_by_id) != set(pair_by_id):
        raise ValueError("Outer- and pair-fold manifests do not cover identical proteins")
    if digests["cohort_digest"] != ELIGIBLE_COHORT_DIGEST:
        raise ValueError("Fold metadata uses an unexpected eligible-cohort digest")
    if digests["outer_fold_digest"] != OUTER_FOLD_MANIFEST_DIGEST:
        raise ValueError("Fold metadata uses an unexpected outer-fold digest")
    if digests["pair_fold_manifest_digest"] != PAIR_FOLD_MANIFEST_DIGEST:
        raise ValueError("Fold metadata uses an unexpected pair-fold digest")
    return CVFoldBundle(
        tuple(outer), tuple(pairs), MappingProxyType(dict(digests)),
        MappingProxyType(outer_by_id), MappingProxyType(pair_by_id),
    )


def deterministic_sample_seed(
    protein_id: str,
    *,
    base_seed: int = 42,
    namespace: str = SAMPLING_NAMESPACE,
) -> int:
    if not str(protein_id):
        raise ValueError("protein_id cannot be empty")
    if not str(namespace):
        raise ValueError("sampling namespace cannot be empty")
    return stable_seed(str(namespace), int(base_seed), str(protein_id))


def deterministic_sample_indices(
    n_rows: int,
    protein_id: str,
    row_cap: int,
    *,
    base_seed: int = 42,
    namespace: str = SAMPLING_NAMESPACE,
) -> np.ndarray:
    """Select a target- and layer-independent sorted sample without replacement."""
    count = int(n_rows)
    cap = int(row_cap)
    if count < 0 or cap < 1:
        raise ValueError("n_rows must be nonnegative and row_cap must be positive")
    if count <= cap:
        return np.arange(count, dtype=np.int64)
    rng = np.random.default_rng(
        deterministic_sample_seed(protein_id, base_seed=base_seed, namespace=namespace)
    )
    selected = np.sort(rng.permutation(count)[:cap].astype(np.int64, copy=False))
    if np.unique(selected).size != cap:
        raise RuntimeError("Deterministic sampling produced duplicate indices")
    return selected


def sampling_assignment_digest(
    n_rows: int,
    protein_id: str,
    row_cap: int,
    *,
    base_seed: int = 42,
    namespace: str = SAMPLING_NAMESPACE,
) -> str:
    indices = deterministic_sample_indices(
        n_rows, protein_id, row_cap, base_seed=base_seed, namespace=namespace,
    )
    payload = {
        "algorithm": SAMPLING_ALGORITHM,
        "namespace": namespace,
        "base_seed": int(base_seed),
        "derived_seed": deterministic_sample_seed(
            protein_id, base_seed=base_seed, namespace=namespace,
        ),
        "protein_id": str(protein_id),
        "n_rows": int(n_rows),
        "row_cap": int(row_cap),
        "n_selected": int(indices.size),
        "indices_digest": numeric_array_digest(indices),
    }
    return semantic_digest(payload)


def _validated_feature_blocks(
    feature_blocks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not feature_blocks:
        raise ValueError("At least one training-protein feature block is required")
    normalized: dict[str, np.ndarray] = {}
    feature_dim: int | None = None
    for raw_id in sorted(feature_blocks, key=str):
        protein_id = str(raw_id)
        if not protein_id or protein_id in normalized:
            raise ValueError("Training protein identifiers must be nonempty and unique")
        block = np.asarray(feature_blocks[raw_id])
        if block.ndim != 2 or block.shape[0] < 1 or block.shape[1] < 1:
            raise ValueError(f"Feature block for {protein_id} must be a nonempty matrix")
        if block.dtype.kind not in "biufc" or block.dtype.hasobject:
            raise TypeError(f"Feature block for {protein_id} must be numeric")
        if feature_dim is None:
            feature_dim = int(block.shape[1])
        elif block.shape[1] != feature_dim:
            raise ValueError("Every training protein must use the same feature dimension")
        if not np.all(np.isfinite(block)):
            raise ValueError(f"Feature block for {protein_id} contains nonfinite values")
        normalized[protein_id] = block
    return normalized


def _validated_target_blocks(
    target_blocks: Mapping[str, np.ndarray],
) -> dict[str, np.ndarray]:
    if not target_blocks:
        raise ValueError("At least one training-protein target block is required")
    normalized: dict[str, np.ndarray] = {}
    for raw_id in sorted(target_blocks, key=str):
        protein_id = str(raw_id)
        if not protein_id or protein_id in normalized:
            raise ValueError("Training protein identifiers must be nonempty and unique")
        block = np.asarray(target_blocks[raw_id])
        if block.ndim != 1 or block.size < 1:
            raise ValueError(f"Target block for {protein_id} must be a nonempty vector")
        if block.dtype.kind not in "biufc" or block.dtype.hasobject:
            raise TypeError(f"Target block for {protein_id} must be numeric")
        if not np.all(np.isfinite(block)):
            raise ValueError(f"Target block for {protein_id} contains nonfinite values")
        normalized[protein_id] = block
    return normalized


@dataclass(frozen=True)
class FittedStandardizer:
    feature_mean: np.ndarray
    feature_scale: np.ndarray
    n_training_proteins: int
    n_training_rows: int
    scaler_digest: str

    def transform(self, features: np.ndarray) -> np.ndarray:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[1] != self.feature_mean.size:
            raise ValueError("Features do not match the fitted standardizer dimension")
        if not np.all(np.isfinite(values)):
            raise ValueError("Features to transform must be finite")
        return (values - self.feature_mean) / self.feature_scale


@dataclass(frozen=True)
class TrainingBaseline:
    value: float
    n_training_proteins: int
    n_training_rows: int
    baseline_digest: str


@dataclass(frozen=True)
class PreparedTrainingFold:
    """Materialized float64 fitting data with train-only preprocessing."""

    standardized_features: np.ndarray
    targets: np.ndarray
    protein_ids: np.ndarray
    sample_weights: np.ndarray
    row_slices: Mapping[str, tuple[int, int]]
    standardizer: FittedStandardizer
    global_baseline: TrainingBaseline


def fit_train_only_standardizer(
    feature_blocks: Mapping[str, np.ndarray],
) -> FittedStandardizer:
    """Fit protein-equal float64 moments from fitting blocks only."""
    blocks = _validated_feature_blocks(feature_blocks)
    moments = ProteinEqualFeatureMoments()
    for block in blocks.values():
        moments.update_protein(np.asarray(block, dtype=np.float64))
    mean, scale = moments.finalize()
    mean = np.ascontiguousarray(np.asarray(mean, dtype=np.float64))
    scale = np.ascontiguousarray(np.asarray(scale, dtype=np.float64))
    if np.any(scale < FLOAT_EPS) or not np.all(np.isfinite(mean)) or not np.all(np.isfinite(scale)):
        raise ValueError("Fitted standardizer is nonfinite or below the canonical scale floor")
    mean.setflags(write=False)
    scale.setflags(write=False)
    row_counts = {protein_id: int(block.shape[0]) for protein_id, block in blocks.items()}
    digest = semantic_digest({
        "algorithm": "protein_equal_float64_moments_v1",
        "protein_row_counts": row_counts,
        "feature_mean_digest": numeric_array_digest(mean),
        "feature_scale_digest": numeric_array_digest(scale),
    })
    return FittedStandardizer(
        mean, scale, len(blocks), sum(row_counts.values()), digest,
    )


def fit_train_only_baseline(
    target_blocks: Mapping[str, np.ndarray],
) -> TrainingBaseline:
    """Fit the protein-equal intercept baseline from fitting targets only."""
    blocks = _validated_target_blocks(target_blocks)
    protein_means = {
        protein_id: float(np.mean(np.asarray(block, dtype=np.float64), dtype=np.float64))
        for protein_id, block in blocks.items()
    }
    value = float(np.mean(np.fromiter(protein_means.values(), dtype=np.float64)))
    row_counts = {protein_id: int(block.size) for protein_id, block in blocks.items()}
    digest = semantic_digest({
        "algorithm": "protein_equal_training_mean_v1",
        "protein_row_counts": row_counts,
        "protein_means": protein_means,
        "value": value,
    })
    return TrainingBaseline(value, len(blocks), sum(row_counts.values()), digest)


def fit_conditional_train_only_baselines(
    target_blocks: Mapping[str, np.ndarray],
    category_code_blocks: Mapping[str, np.ndarray],
    category_labels: Sequence[str],
) -> dict[str, TrainingBaseline]:
    """Fit one protein-equal training mean per gate/category."""
    targets = _validated_target_blocks(target_blocks)
    labels = tuple(str(label) for label in category_labels)
    if not labels or len(set(labels)) != len(labels):
        raise ValueError("Conditional baseline labels must be nonempty and unique")
    if set(map(str, category_code_blocks)) != set(targets):
        raise ValueError("Category-code and target blocks must cover identical proteins")
    codes_by_id: dict[str, np.ndarray] = {}
    for raw_id, raw_codes in category_code_blocks.items():
        protein_id = str(raw_id)
        codes = np.asarray(raw_codes)
        if codes.shape != targets[protein_id].shape or not np.issubdtype(codes.dtype, np.integer):
            raise ValueError(f"Category codes for {protein_id} must be an integer target-shaped vector")
        if np.any((codes < -1) | (codes >= len(labels))):
            raise ValueError(f"Category codes for {protein_id} contain an invalid value")
        codes_by_id[protein_id] = codes.astype(np.int16, copy=False)
    baselines: dict[str, TrainingBaseline] = {}
    for category, label in enumerate(labels):
        gated = {
            protein_id: targets[protein_id][codes_by_id[protein_id] == category]
            for protein_id in targets
            if np.any(codes_by_id[protein_id] == category)
        }
        if gated:
            baselines[label] = fit_train_only_baseline(gated)
        else:
            baselines[label] = TrainingBaseline(
                float("nan"), 0, 0,
                semantic_digest({"algorithm": "protein_equal_training_mean_v1", "label": label, "empty": True}),
            )
    return baselines


def prepare_training_fold(
    feature_blocks: Mapping[str, np.ndarray],
    target_blocks: Mapping[str, np.ndarray],
) -> PreparedTrainingFold:
    """Materialize fitting rows, weights, scaler, and baseline from training blocks."""
    features = _validated_feature_blocks(feature_blocks)
    targets = _validated_target_blocks(target_blocks)
    if set(features) != set(targets):
        raise ValueError("Training feature and target blocks must cover identical proteins")
    for protein_id in features:
        if features[protein_id].shape[0] != targets[protein_id].size:
            raise ValueError(f"Feature/target row mismatch for {protein_id}")
    standardizer = fit_train_only_standardizer(features)
    baseline = fit_train_only_baseline(targets)
    n_rows = sum(block.shape[0] for block in features.values())
    n_features = standardizer.feature_mean.size
    matrix = np.empty((n_rows, n_features), dtype=np.float64)
    response = np.empty(n_rows, dtype=np.float64)
    protein_ids = np.empty(n_rows, dtype=object)
    row_slices: dict[str, tuple[int, int]] = {}
    cursor = 0
    for protein_id in sorted(features):
        stop = cursor + features[protein_id].shape[0]
        matrix[cursor:stop] = np.asarray(features[protein_id], dtype=np.float64)
        response[cursor:stop] = np.asarray(targets[protein_id], dtype=np.float64)
        protein_ids[cursor:stop] = protein_id
        row_slices[protein_id] = (cursor, stop)
        cursor = stop
    weights = protein_equal_weights(protein_ids)
    for protein_id, (start, stop) in row_slices.items():
        if not math.isclose(float(np.sum(weights[start:stop])), 1.0, rel_tol=1e-12, abs_tol=1e-12):
            raise RuntimeError(f"Protein-equal weights do not sum to one for {protein_id}")
    np.subtract(matrix, standardizer.feature_mean, out=matrix)
    np.divide(matrix, standardizer.feature_scale, out=matrix)
    return PreparedTrainingFold(
        matrix, response, protein_ids, weights,
        MappingProxyType(row_slices), standardizer, baseline,
    )


def _as_float_vector(values: np.ndarray | Sequence[float], label: str) -> np.ndarray:
    array = np.asarray(values, dtype=np.float64)
    if array.ndim != 1:
        raise ValueError(f"{label} must be a one-dimensional vector")
    return array


@dataclass
class RegressionSufficientStatistics:
    """Mergeable OOF statistics that reproduce direct concatenation metrics."""

    n: int = 0
    sum_y: float = 0.0
    sum_y2: float = 0.0
    sum_prediction: float = 0.0
    sum_error: float = 0.0
    sum_abs_error: float = 0.0
    sum_squared_error: float = 0.0

    def update(self, y_true: np.ndarray, predictions: np.ndarray) -> None:
        y = _as_float_vector(y_true, "y_true")
        pred = _as_float_vector(predictions, "predictions")
        if y.shape != pred.shape:
            raise ValueError("Targets and predictions must have identical shape")
        if not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
            raise ValueError("Sufficient statistics accept only finite evaluated rows")
        if y.size == 0:
            return
        error = pred - y
        self.n += int(y.size)
        self.sum_y += float(np.sum(y, dtype=np.float64))
        self.sum_y2 += float(np.sum(y * y, dtype=np.float64))
        self.sum_prediction += float(np.sum(pred, dtype=np.float64))
        self.sum_error += float(np.sum(error, dtype=np.float64))
        self.sum_abs_error += float(np.sum(np.abs(error), dtype=np.float64))
        self.sum_squared_error += float(np.sum(error * error, dtype=np.float64))

    def merge(self, other: "RegressionSufficientStatistics") -> "RegressionSufficientStatistics":
        if not isinstance(other, RegressionSufficientStatistics):
            raise TypeError("Can only merge regression sufficient statistics")
        for name in self.to_payload():
            setattr(self, name, getattr(self, name) + getattr(other, name))
        return self

    def metrics(self) -> dict[str, float]:
        if self.n == 0:
            return {
                "target_variance": float("nan"), "r2": float("nan"),
                "mae": float("nan"), "rmse": float("nan"),
                "bias_pred_minus_true": float("nan"), "mean_true_distance": float("nan"),
                "mean_predicted_distance": float("nan"),
            }
        mean_y = self.sum_y / self.n
        variance = max(0.0, self.sum_y2 / self.n - mean_y * mean_y)
        total_sum_squares = self.sum_y2 - self.sum_y * self.sum_y / self.n
        return {
            "target_variance": variance,
            "r2": float("nan") if total_sum_squares <= FLOAT_EPS else 1.0 - self.sum_squared_error / total_sum_squares,
            "mae": self.sum_abs_error / self.n,
            "rmse": math.sqrt(self.sum_squared_error / self.n),
            "bias_pred_minus_true": self.sum_error / self.n,
            "mean_true_distance": mean_y,
            "mean_predicted_distance": self.sum_prediction / self.n,
        }

    def to_payload(self) -> dict[str, int | float]:
        return {
            "n": int(self.n), "sum_y": float(self.sum_y), "sum_y2": float(self.sum_y2),
            "sum_prediction": float(self.sum_prediction), "sum_error": float(self.sum_error),
            "sum_abs_error": float(self.sum_abs_error),
            "sum_squared_error": float(self.sum_squared_error),
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "RegressionSufficientStatistics":
        expected = set(cls().to_payload())
        if set(payload) != expected:
            raise ValueError("Unexpected sufficient-statistic payload")
        result = cls(
            n=int(payload["n"]), sum_y=float(payload["sum_y"]),
            sum_y2=float(payload["sum_y2"]), sum_prediction=float(payload["sum_prediction"]),
            sum_error=float(payload["sum_error"]),
            sum_abs_error=float(payload["sum_abs_error"]),
            sum_squared_error=float(payload["sum_squared_error"]),
        )
        if result.n < 0 or not all(math.isfinite(float(value)) for key, value in result.to_payload().items() if key != "n"):
            raise ValueError("Invalid sufficient-statistic payload")
        return result


@dataclass
class OOFComparisonAccumulator:
    """Exact-once protein OOF metrics with model and train-derived baselines."""

    model: RegressionSufficientStatistics = field(default_factory=RegressionSufficientStatistics)
    global_baseline: RegressionSufficientStatistics = field(default_factory=RegressionSufficientStatistics)
    conditional_baseline: RegressionSufficientStatistics | None = None
    expected_pairs_by_protein: dict[str, int] = field(default_factory=dict)
    evaluated_pairs_by_protein: dict[str, int] = field(default_factory=dict)
    outer_fold_by_protein: dict[str, int] = field(default_factory=dict)
    stable_fit_by_protein: dict[str, bool] = field(default_factory=dict)
    per_protein_model: dict[str, RegressionSufficientStatistics] = field(default_factory=dict)
    per_protein_global: dict[str, RegressionSufficientStatistics] = field(default_factory=dict)
    per_protein_conditional: dict[str, RegressionSufficientStatistics] = field(default_factory=dict)

    def update_protein(
        self,
        protein_id: str,
        outer_fold: int,
        y_true: np.ndarray,
        predictions: np.ndarray,
        global_baseline_predictions: np.ndarray | float,
        *,
        conditional_baseline_predictions: np.ndarray | float | None = None,
        stable_fit: bool = True,
    ) -> np.ndarray:
        protein = str(protein_id)
        if not protein or protein in self.expected_pairs_by_protein:
            raise ValueError(f"OOF protein must be nonempty and appear exactly once: {protein!r}")
        fold = int(outer_fold)
        if fold not in range(N_OUTER_FOLDS):
            raise ValueError(f"outer_fold must be in 0..{N_OUTER_FOLDS - 1}")
        y = _as_float_vector(y_true, "y_true")
        pred = _as_float_vector(predictions, "predictions")
        if y.shape != pred.shape or y.size < 1 or not np.all(np.isfinite(y)):
            raise ValueError("OOF targets must be a nonempty finite vector matching predictions")
        global_pred = np.broadcast_to(
            np.asarray(global_baseline_predictions, dtype=np.float64), y.shape,
        )
        conditional_pred: np.ndarray | None = None
        if conditional_baseline_predictions is not None:
            conditional_pred = np.broadcast_to(
                np.asarray(conditional_baseline_predictions, dtype=np.float64), y.shape,
            )
        if self.conditional_baseline is None and self.expected_pairs_by_protein and conditional_pred is not None:
            raise ValueError("Conditional baseline applicability must be consistent")
        if self.conditional_baseline is not None and conditional_pred is None:
            raise ValueError("Conditional baseline applicability must be consistent")
        if not self.expected_pairs_by_protein and conditional_pred is not None:
            self.conditional_baseline = RegressionSufficientStatistics()

        evaluated = np.isfinite(pred)
        if np.any(evaluated) and not np.all(np.isfinite(global_pred[evaluated])):
            raise ValueError("Global baseline is nonfinite on evaluated model rows")
        if conditional_pred is not None and np.any(evaluated) and not np.all(np.isfinite(conditional_pred[evaluated])):
            raise ValueError("Conditional baseline is nonfinite on evaluated model rows")

        local_model = RegressionSufficientStatistics()
        local_global = RegressionSufficientStatistics()
        local_conditional = RegressionSufficientStatistics()
        if np.any(evaluated):
            local_model.update(y[evaluated], pred[evaluated])
            local_global.update(y[evaluated], global_pred[evaluated])
            if conditional_pred is not None:
                local_conditional.update(y[evaluated], conditional_pred[evaluated])
        self.model.merge(local_model)
        self.global_baseline.merge(local_global)
        if self.conditional_baseline is not None:
            self.conditional_baseline.merge(local_conditional)
        self.expected_pairs_by_protein[protein] = int(y.size)
        self.evaluated_pairs_by_protein[protein] = int(np.count_nonzero(evaluated))
        self.outer_fold_by_protein[protein] = fold
        self.stable_fit_by_protein[protein] = bool(stable_fit)
        self.per_protein_model[protein] = local_model
        self.per_protein_global[protein] = local_global
        if conditional_pred is not None:
            self.per_protein_conditional[protein] = local_conditional
        return evaluated

    @property
    def n_expected_pairs(self) -> int:
        return sum(self.expected_pairs_by_protein.values())

    @property
    def n_evaluated_pairs(self) -> int:
        return self.model.n

    @property
    def n_proteins(self) -> int:
        return len(self.expected_pairs_by_protein)

    @property
    def n_proteins_with_finite_r2(self) -> int:
        return sum(
            np.isfinite(stats.metrics()["r2"])
            for stats in self.per_protein_model.values()
        )

    @property
    def pair_coverage(self) -> float:
        return (
            float("nan") if self.n_expected_pairs == 0
            else self.n_evaluated_pairs / self.n_expected_pairs
        )

    @property
    def is_complete(self) -> bool:
        return bool(self.expected_pairs_by_protein) and all(
            self.evaluated_pairs_by_protein[protein] == expected
            for protein, expected in self.expected_pairs_by_protein.items()
        )

    def require_complete(self) -> None:
        if not self.is_complete:
            incomplete = sorted(
                protein for protein, expected in self.expected_pairs_by_protein.items()
                if self.evaluated_pairs_by_protein.get(protein, 0) != expected
            )
            raise ValueError(f"OOF accumulator has incomplete pair coverage: {incomplete[:5]}")
        if self.global_baseline.n != self.model.n:
            raise ValueError("Global baseline and model do not cover identical OOF rows")
        if self.conditional_baseline is not None and self.conditional_baseline.n != self.model.n:
            raise ValueError("Conditional baseline and model do not cover identical OOF rows")

    def merge(self, other: "OOFComparisonAccumulator") -> "OOFComparisonAccumulator":
        overlap = set(self.expected_pairs_by_protein) & set(other.expected_pairs_by_protein)
        if overlap:
            raise ValueError(f"Cannot merge duplicate OOF proteins: {sorted(overlap)[:5]}")
        if (self.conditional_baseline is None) != (other.conditional_baseline is None):
            if self.n_proteins == 0:
                self.conditional_baseline = (
                    None if other.conditional_baseline is None else RegressionSufficientStatistics()
                )
            elif other.n_proteins:
                raise ValueError("Cannot merge inconsistent conditional-baseline scopes")
        self.model.merge(other.model)
        self.global_baseline.merge(other.global_baseline)
        if self.conditional_baseline is not None and other.conditional_baseline is not None:
            self.conditional_baseline.merge(other.conditional_baseline)
        for destination_name in (
            "expected_pairs_by_protein", "evaluated_pairs_by_protein", "outer_fold_by_protein",
            "stable_fit_by_protein", "per_protein_model", "per_protein_global",
            "per_protein_conditional",
        ):
            getattr(self, destination_name).update(getattr(other, destination_name))
        return self

    def subset(self, protein_ids: Iterable[str]) -> "OOFComparisonAccumulator":
        selected = {str(protein_id) for protein_id in protein_ids}
        unknown = selected - set(self.expected_pairs_by_protein)
        if unknown:
            raise ValueError(f"Unknown OOF protein IDs: {sorted(unknown)[:5]}")
        result = OOFComparisonAccumulator()
        conditional = self.conditional_baseline is not None
        if conditional:
            result.conditional_baseline = RegressionSufficientStatistics()
        for protein in sorted(selected):
            result.expected_pairs_by_protein[protein] = self.expected_pairs_by_protein[protein]
            result.evaluated_pairs_by_protein[protein] = self.evaluated_pairs_by_protein[protein]
            result.outer_fold_by_protein[protein] = self.outer_fold_by_protein[protein]
            result.stable_fit_by_protein[protein] = self.stable_fit_by_protein[protein]
            result.per_protein_model[protein] = self.per_protein_model[protein]
            result.per_protein_global[protein] = self.per_protein_global[protein]
            result.model.merge(self.per_protein_model[protein])
            result.global_baseline.merge(self.per_protein_global[protein])
            if conditional:
                local = self.per_protein_conditional[protein]
                result.per_protein_conditional[protein] = local
                result.conditional_baseline.merge(local)
        return result

    def stable_only(self) -> "OOFComparisonAccumulator":
        return self.subset(
            protein for protein, stable in self.stable_fit_by_protein.items() if stable
        )

    def to_payload(self) -> dict[str, object]:
        return {
            "model": self.model.to_payload(),
            "global_baseline": self.global_baseline.to_payload(),
            "conditional_baseline": (
                None if self.conditional_baseline is None else self.conditional_baseline.to_payload()
            ),
            "proteins": [{
                "protein_id": protein,
                "outer_fold": self.outer_fold_by_protein[protein],
                "n_expected_pairs": self.expected_pairs_by_protein[protein],
                "n_evaluated_pairs": self.evaluated_pairs_by_protein[protein],
                "stable_fit": self.stable_fit_by_protein[protein],
                "model": self.per_protein_model[protein].to_payload(),
                "global_baseline": self.per_protein_global[protein].to_payload(),
                "conditional_baseline": (
                    self.per_protein_conditional[protein].to_payload()
                    if self.conditional_baseline is not None else None
                ),
            } for protein in sorted(self.expected_pairs_by_protein)],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "OOFComparisonAccumulator":
        if set(payload) != {"model", "global_baseline", "conditional_baseline", "proteins"}:
            raise ValueError("Unexpected OOF accumulator payload")
        conditional_payload = payload["conditional_baseline"]
        result = cls(
            model=RegressionSufficientStatistics.from_payload(payload["model"]),
            global_baseline=RegressionSufficientStatistics.from_payload(payload["global_baseline"]),
            conditional_baseline=(
                None if conditional_payload is None
                else RegressionSufficientStatistics.from_payload(conditional_payload)
            ),
        )
        rows = payload["proteins"]
        if not isinstance(rows, list):
            raise ValueError("OOF accumulator proteins payload must be a list")
        for row in rows:
            protein = str(row["protein_id"])
            if not protein or protein in result.expected_pairs_by_protein:
                raise ValueError("OOF accumulator payload contains duplicate proteins")
            result.expected_pairs_by_protein[protein] = int(row["n_expected_pairs"])
            result.evaluated_pairs_by_protein[protein] = int(row["n_evaluated_pairs"])
            result.outer_fold_by_protein[protein] = int(row["outer_fold"])
            result.stable_fit_by_protein[protein] = bool(row["stable_fit"])
            result.per_protein_model[protein] = RegressionSufficientStatistics.from_payload(row["model"])
            result.per_protein_global[protein] = RegressionSufficientStatistics.from_payload(row["global_baseline"])
            if (
                result.expected_pairs_by_protein[protein] < 1
                or result.evaluated_pairs_by_protein[protein] < 0
                or result.evaluated_pairs_by_protein[protein] > result.expected_pairs_by_protein[protein]
                or result.outer_fold_by_protein[protein] not in range(N_OUTER_FOLDS)
                or result.per_protein_model[protein].n != result.evaluated_pairs_by_protein[protein]
                or result.per_protein_global[protein].n != result.evaluated_pairs_by_protein[protein]
            ):
                raise ValueError("Invalid per-protein OOF accumulator payload")
            if result.conditional_baseline is not None:
                if row["conditional_baseline"] is None:
                    raise ValueError("Missing per-protein conditional baseline payload")
                result.per_protein_conditional[protein] = RegressionSufficientStatistics.from_payload(
                    row["conditional_baseline"]
                )
                if result.per_protein_conditional[protein].n != result.evaluated_pairs_by_protein[protein]:
                    raise ValueError("Conditional baseline payload has inconsistent row support")
        rebuilt = cls()
        if result.conditional_baseline is not None:
            rebuilt.conditional_baseline = RegressionSufficientStatistics()
        for protein in sorted(result.expected_pairs_by_protein):
            rebuilt.expected_pairs_by_protein[protein] = result.expected_pairs_by_protein[protein]
            rebuilt.evaluated_pairs_by_protein[protein] = result.evaluated_pairs_by_protein[protein]
            rebuilt.outer_fold_by_protein[protein] = result.outer_fold_by_protein[protein]
            rebuilt.stable_fit_by_protein[protein] = result.stable_fit_by_protein[protein]
            rebuilt.per_protein_model[protein] = result.per_protein_model[protein]
            rebuilt.per_protein_global[protein] = result.per_protein_global[protein]
            rebuilt.model.merge(result.per_protein_model[protein])
            rebuilt.global_baseline.merge(result.per_protein_global[protein])
            if rebuilt.conditional_baseline is not None:
                local = result.per_protein_conditional[protein]
                rebuilt.per_protein_conditional[protein] = local
                rebuilt.conditional_baseline.merge(local)
        def statistics_match(
            left: RegressionSufficientStatistics,
            right: RegressionSufficientStatistics,
        ) -> bool:
            if left.n != right.n:
                return False
            return all(
                math.isclose(
                    float(getattr(left, name)), float(getattr(right, name)),
                    rel_tol=1e-12, abs_tol=1e-12,
                )
                for name in left.to_payload() if name != "n"
            )

        if not statistics_match(rebuilt.model, result.model) or not statistics_match(
            rebuilt.global_baseline, result.global_baseline,
        ):
            raise ValueError("OOF accumulator totals do not match its per-protein payload")
        if result.conditional_baseline is not None and not statistics_match(
            rebuilt.conditional_baseline, result.conditional_baseline,
        ):
            raise ValueError("OOF conditional-baseline total does not match its protein payload")
        return result


class OOFPairAssembler:
    """Assemble disjoint pair-fold predictions in canonical pair order."""

    def __init__(self, y_true: np.ndarray, *, with_conditional_baseline: bool = False):
        y = _as_float_vector(y_true, "y_true")
        if y.size < 1 or not np.all(np.isfinite(y)):
            raise ValueError("Pair-level OOF assembly requires finite nonempty targets")
        self.y_true = y.copy()
        self.predictions = np.full(y.size, np.nan, dtype=np.float64)
        self.global_baseline_predictions = np.full(y.size, np.nan, dtype=np.float64)
        self.conditional_baseline_predictions = (
            np.full(y.size, np.nan, dtype=np.float64) if with_conditional_baseline else None
        )
        self.pair_folds = np.full(y.size, -1, dtype=np.int16)
        self.assigned = np.zeros(y.size, dtype=bool)

    def add_fold(
        self,
        pair_fold: int,
        pair_indices: np.ndarray,
        predictions: np.ndarray,
        global_baseline_predictions: np.ndarray | float,
        *,
        conditional_baseline_predictions: np.ndarray | float | None = None,
    ) -> None:
        fold = int(pair_fold)
        if fold not in range(N_PAIR_FOLDS):
            raise ValueError(f"pair_fold must be in 0..{N_PAIR_FOLDS - 1}")
        indices = np.asarray(pair_indices, dtype=np.int64)
        pred = _as_float_vector(predictions, "predictions")
        if indices.ndim != 1 or indices.size < 1 or indices.shape != pred.shape:
            raise ValueError("Pair indices and fold predictions must be matching nonempty vectors")
        if np.any(indices < 0) or np.any(indices >= self.y_true.size):
            raise ValueError("Pair-fold indices are out of bounds")
        if np.unique(indices).size != indices.size or np.any(self.assigned[indices]):
            raise ValueError("Every canonical pair may be assigned out of fold exactly once")
        global_pred = np.broadcast_to(
            np.asarray(global_baseline_predictions, dtype=np.float64), pred.shape,
        )
        if not np.all(np.isfinite(global_pred)):
            raise ValueError("Pair-fold global baseline predictions must be finite")
        conditional_pred: np.ndarray | None = None
        if conditional_baseline_predictions is not None:
            conditional_pred = np.broadcast_to(
                np.asarray(conditional_baseline_predictions, dtype=np.float64), pred.shape,
            )
        if (self.conditional_baseline_predictions is None) != (conditional_pred is None):
            raise ValueError("Conditional baseline applicability must match the assembler")
        if conditional_pred is not None and not np.all(np.isfinite(conditional_pred)):
            raise ValueError("Pair-fold conditional baseline predictions must be finite")
        self.predictions[indices] = pred
        self.global_baseline_predictions[indices] = global_pred
        if conditional_pred is not None:
            self.conditional_baseline_predictions[indices] = conditional_pred
        self.pair_folds[indices] = fold
        self.assigned[indices] = True

    @property
    def assignment_complete(self) -> bool:
        return bool(np.all(self.assigned))

    @property
    def prediction_complete(self) -> bool:
        return self.assignment_complete and bool(np.all(np.isfinite(self.predictions)))

    def finalize(self, *, require_finite_predictions: bool = True) -> dict[str, np.ndarray | None]:
        if not self.assignment_complete:
            raise ValueError("Pair-level OOF folds do not cover every canonical pair")
        counts = np.bincount(self.pair_folds, minlength=N_PAIR_FOLDS)
        if np.any(counts == 0):
            raise ValueError("Every pair fold must contribute at least one held-out pair")
        if require_finite_predictions and not self.prediction_complete:
            raise ValueError("Pair-level OOF predictions are incomplete or nonfinite")
        return {
            "y_true": self.y_true.copy(),
            "predictions": self.predictions.copy(),
            "global_baseline_predictions": self.global_baseline_predictions.copy(),
            "conditional_baseline_predictions": (
                None if self.conditional_baseline_predictions is None
                else self.conditional_baseline_predictions.copy()
            ),
            "pair_folds": self.pair_folds.copy(),
        }


def _identity(
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    evaluation_design: str,
    oof_prediction_scope: str,
) -> dict[str, object]:
    if not analysis_name or not model_family or not evaluation_design or not oof_prediction_scope:
        raise ValueError("V2 output identity strings cannot be empty")
    if evaluation_design not in EVALUATION_DESIGNS:
        raise ValueError(f"Unknown fivefold evaluation design: {evaluation_design}")
    if int(layer) not in LAYERS:
        raise ValueError("V2 outputs require a canonical layer 0..47")
    if int(feature_dim) < 1 or int(feature_dim) > len(FEATURE_NAMES):
        raise ValueError(f"feature_dim must be in 1..{len(FEATURE_NAMES)}")
    return {
        "analysis_name": analysis_name,
        "model_family": model_family,
        "feature_dim": int(feature_dim),
        "layer": int(layer),
        "cv_protocol_version": CV_PROTOCOL_VERSION,
        "evaluation_design": evaluation_design,
        "evaluation_scope": EVALUATION_SCOPE,
        "oof_prediction_scope": oof_prediction_scope,
    }


def _prefixed_metrics(metrics: Mapping[str, float], prefix: str) -> dict[str, float]:
    return {
        f"{prefix}r2": metrics["r2"],
        f"{prefix}mae": metrics["mae"],
        f"{prefix}rmse": metrics["rmse"],
        f"{prefix}bias_pred_minus_true": metrics["bias_pred_minus_true"],
    }


def build_per_protein_row(
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    protein_id: str,
    source: str,
    outer_fold: int,
    evaluation_design: str,
    oof_prediction_scope: str,
    n_residues: int,
    y_true: np.ndarray,
    predictions: np.ndarray,
    prediction_status: str | None = None,
) -> dict[str, object]:
    y = _as_float_vector(y_true, "y_true")
    pred = _as_float_vector(predictions, "predictions")
    residues = int(n_residues)
    expected_pairs = residues * (residues - 1) // 2
    if residues < 2 or y.size != expected_pairs or y.shape != pred.shape:
        raise ValueError("Per-protein OOF rows require every canonical unordered pair slot")
    if str(source) not in {"cameo", "ecod"}:
        raise ValueError("Per-protein source must be cameo or ecod")
    if not np.all(np.isfinite(y)):
        raise ValueError("Per-protein OOF targets must be finite")
    fold = int(outer_fold)
    if fold not in range(N_OUTER_FOLDS):
        raise ValueError(f"outer_fold must be in 0..{N_OUTER_FOLDS - 1}")
    evaluated = np.isfinite(pred)
    stats = RegressionSufficientStatistics()
    stats.update(y[evaluated], pred[evaluated])
    n_evaluated = int(np.count_nonzero(evaluated))
    automatic_status = (
        "complete" if n_evaluated == expected_pairs
        else "no_predictions" if n_evaluated == 0
        else "incomplete_pair_coverage"
    )
    status = prediction_status or automatic_status
    if status == "complete" and n_evaluated != expected_pairs:
        raise ValueError("prediction_status=complete requires 100% finite OOF coverage")
    row = {
        **_identity(
            analysis_name=analysis_name, model_family=model_family, feature_dim=feature_dim,
            layer=layer, evaluation_design=evaluation_design,
            oof_prediction_scope=oof_prediction_scope,
        ),
        "protein_id": str(protein_id), "source": str(source), "outer_fold": fold,
        "prediction_status": status, "n_residues": residues,
        "n_protein_pairs": expected_pairs, "n_expected_pairs": expected_pairs,
        "n_evaluated_pairs": n_evaluated,
        "pair_coverage": n_evaluated / expected_pairs,
        **stats.metrics(),
    }
    return {column: row[column] for column in PER_PROTEIN_COLUMNS}


def _comparison_row_values(
    accumulator: OOFComparisonAccumulator,
    *,
    global_baseline_type: str,
    conditional_baseline_type: str,
    require_complete: bool,
) -> dict[str, object]:
    if require_complete:
        accumulator.require_complete()
    if accumulator.global_baseline.n != accumulator.model.n:
        raise ValueError("Global baseline and model must use identical evaluated rows")
    if accumulator.conditional_baseline is None:
        if conditional_baseline_type != "not_applicable":
            raise ValueError("A conditional baseline type was supplied without predictions")
        conditional_metrics = RegressionSufficientStatistics().metrics()
    else:
        if conditional_baseline_type == "not_applicable":
            raise ValueError("Conditional predictions require an explicit baseline type")
        if accumulator.conditional_baseline.n != accumulator.model.n:
            raise ValueError("Conditional baseline and model must use identical evaluated rows")
        conditional_metrics = accumulator.conditional_baseline.metrics()
    return {
        "n_proteins": accumulator.n_proteins,
        "n_proteins_with_finite_r2": accumulator.n_proteins_with_finite_r2,
        "n_expected_pairs": accumulator.n_expected_pairs,
        "n_evaluated_pairs": accumulator.n_evaluated_pairs,
        "pair_coverage": accumulator.pair_coverage,
        **_prefixed_metrics(accumulator.model.metrics(), "pooled_"),
        "global_baseline_type": global_baseline_type,
        **_prefixed_metrics(accumulator.global_baseline.metrics(), "global_baseline_pooled_"),
        "conditional_baseline_type": conditional_baseline_type,
        **_prefixed_metrics(conditional_metrics, "conditional_baseline_pooled_"),
    }


def build_pooled_metrics_row(
    accumulator: OOFComparisonAccumulator,
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    evaluation_design: str,
    oof_prediction_scope: str,
    global_baseline_type: str = GLOBAL_BASELINE_TYPE,
    conditional_baseline_type: str = "not_applicable",
    require_complete: bool = True,
) -> dict[str, object]:
    row = {
        **_identity(
            analysis_name=analysis_name, model_family=model_family, feature_dim=feature_dim,
            layer=layer, evaluation_design=evaluation_design,
            oof_prediction_scope=oof_prediction_scope,
        ),
        **_comparison_row_values(
            accumulator, global_baseline_type=global_baseline_type,
            conditional_baseline_type=conditional_baseline_type,
            require_complete=require_complete,
        ),
    }
    return {column: row[column] for column in POOLED_METRICS_COLUMNS}


def build_fold_metrics_row(
    accumulator: OOFComparisonAccumulator,
    *,
    outer_fold: int,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    evaluation_design: str,
    oof_prediction_scope: str,
    global_baseline_type: str = GLOBAL_BASELINE_TYPE,
    conditional_baseline_type: str = "not_applicable",
    require_complete: bool = True,
) -> dict[str, object]:
    fold = int(outer_fold)
    if fold not in range(N_OUTER_FOLDS):
        raise ValueError(f"outer_fold must be in 0..{N_OUTER_FOLDS - 1}")
    observed_folds = set(accumulator.outer_fold_by_protein.values())
    if observed_folds and observed_folds != {fold}:
        raise ValueError(f"Fold metric row contains proteins from folds {sorted(observed_folds)}")
    base = build_pooled_metrics_row(
        accumulator, analysis_name=analysis_name, model_family=model_family,
        feature_dim=feature_dim, layer=layer, evaluation_design=evaluation_design,
        oof_prediction_scope=oof_prediction_scope,
        global_baseline_type=global_baseline_type,
        conditional_baseline_type=conditional_baseline_type,
        require_complete=require_complete,
    )
    row = {**base, "outer_fold": fold}
    return {column: row[column] for column in FOLD_METRICS_COLUMNS}


def build_fold_summary_rows(
    fold_rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    """Summarize five descriptive fold metrics without treating them as CIs."""
    if len(fold_rows) != N_OUTER_FOLDS:
        raise ValueError(f"Fold summary requires exactly {N_OUTER_FOLDS} rows")
    folds = {int(row["outer_fold"]) for row in fold_rows}
    if folds != set(range(N_OUTER_FOLDS)):
        raise ValueError("Fold summary requires one row for every outer fold")
    identity = {column: fold_rows[0][column] for column in _METRIC_IDENTITY_COLUMNS}
    if any(
        any(row[column] != identity[column] for column in _METRIC_IDENTITY_COLUMNS)
        for row in fold_rows[1:]
    ):
        raise ValueError("Fold summary rows do not share one analysis identity")
    metric_columns = (
        "pooled_r2", "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true",
        "global_baseline_pooled_r2", "global_baseline_pooled_mae",
        "global_baseline_pooled_rmse", "global_baseline_pooled_bias_pred_minus_true",
        "conditional_baseline_pooled_r2", "conditional_baseline_pooled_mae",
        "conditional_baseline_pooled_rmse", "conditional_baseline_pooled_bias_pred_minus_true",
    )
    result = []
    for metric in metric_columns:
        values = np.asarray([float(row[metric]) for row in fold_rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        result.append({
            **identity, "fold_metric": metric, "n_folds": int(finite.size),
            "mean": float(np.mean(finite)) if finite.size else float("nan"),
            "std": float(np.std(finite, ddof=0)) if finite.size else float("nan"),
        })
    return [{column: row[column] for column in FOLD_SUMMARY_COLUMNS} for row in result]


def build_stable_fit_metrics_row(
    accumulator: OOFComparisonAccumulator,
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    evaluation_design: str,
    oof_prediction_scope: str,
    global_baseline_type: str = GLOBAL_BASELINE_TYPE,
    conditional_baseline_type: str = "not_applicable",
) -> dict[str, object]:
    stable = accumulator.stable_only()
    values = _comparison_row_values(
        stable, global_baseline_type=global_baseline_type,
        conditional_baseline_type=conditional_baseline_type,
        require_complete=stable.n_proteins > 0,
    )
    row = {
        **_identity(
            analysis_name=analysis_name, model_family=model_family, feature_dim=feature_dim,
            layer=layer, evaluation_design=evaluation_design,
            oof_prediction_scope=oof_prediction_scope,
        ),
        "stability_filter": "all_contributing_ols_fits_pass_rank_condition_policy",
        "n_proteins_total": accumulator.n_proteins,
        "n_proteins_included": stable.n_proteins,
        **values,
    }
    return {column: row[column] for column in STABLE_FIT_METRICS_COLUMNS}


def build_training_metrics_row(
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    evaluation_design: str,
    training_scope: str,
    y_true: np.ndarray,
    predictions: np.ndarray,
    fit_status: str,
    outer_fold: int | None = None,
    protein_id: str = "",
    pair_fold: int | None = None,
    sequence_gate: str = "",
    support_flags: str = "",
    n_training_proteins: int = 0,
    design_rank: float = float("nan"),
    largest_singular_value: float = float("nan"),
    smallest_singular_value: float = float("nan"),
    condition_number: float = float("nan"),
) -> dict[str, object]:
    if int(layer) not in LAYERS or int(feature_dim) < 1 or evaluation_design not in EVALUATION_DESIGNS:
        raise ValueError("Invalid training-metric identity")
    if outer_fold is not None and int(outer_fold) not in range(N_OUTER_FOLDS):
        raise ValueError("Training-metric outer_fold is invalid")
    if pair_fold is not None and int(pair_fold) not in range(N_PAIR_FOLDS):
        raise ValueError("Training-metric pair_fold is invalid")
    if int(n_training_proteins) < 1:
        raise ValueError("Training metrics require at least one fitting protein")
    y = _as_float_vector(y_true, "y_true")
    pred = _as_float_vector(predictions, "predictions")
    if y.shape != pred.shape or not np.all(np.isfinite(y)) or not np.all(np.isfinite(pred)):
        raise ValueError("Training metrics require matching finite target and prediction vectors")
    stats = RegressionSufficientStatistics()
    stats.update(y, pred)
    row = {
        "analysis_name": analysis_name, "model_family": model_family,
        "feature_dim": int(feature_dim), "layer": int(layer),
        "cv_protocol_version": CV_PROTOCOL_VERSION, "evaluation_design": evaluation_design,
        "outer_fold": "" if outer_fold is None else int(outer_fold),
        "protein_id": str(protein_id), "pair_fold": "" if pair_fold is None else int(pair_fold),
        "sequence_gate": str(sequence_gate), "training_scope": str(training_scope),
        "fit_status": str(fit_status), "support_flags": str(support_flags),
        "n_training_proteins": int(n_training_proteins), "n_training_pairs": int(y.size),
        **stats.metrics(), "design_rank": design_rank,
        "largest_singular_value": largest_singular_value,
        "smallest_singular_value": smallest_singular_value,
        "condition_number": condition_number,
    }
    return {column: row[column] for column in TRAINING_METRICS_COLUMNS}


def build_model_status_row(
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    fit_status: str,
    outer_fold: int | None = None,
    protein_id: str = "",
    pair_fold: int | None = None,
    sequence_gate: str = "",
    support_flags: str = "",
    n_training_proteins: int = 0,
    n_training_pairs: int = 0,
    design_rank: float = float("nan"),
    largest_singular_value: float = float("nan"),
    smallest_singular_value: float = float("nan"),
    condition_number: float = float("nan"),
    selected_configuration_id: str = "",
    final_boosting_rounds: int | None = None,
    training_row_cap: int | None = None,
    random_seed: int | None = None,
) -> dict[str, object]:
    if int(layer) not in LAYERS or int(feature_dim) < 1 or int(feature_dim) > len(FEATURE_NAMES):
        raise ValueError("Invalid model-status layer or feature dimension")
    if outer_fold is not None and int(outer_fold) not in range(N_OUTER_FOLDS):
        raise ValueError("Model-status outer_fold is invalid")
    if pair_fold is not None and int(pair_fold) not in range(N_PAIR_FOLDS):
        raise ValueError("Model-status pair_fold is invalid")
    if int(n_training_proteins) < 0 or int(n_training_pairs) < 0:
        raise ValueError("Model-status training support cannot be negative")
    row = {
        "analysis_name": analysis_name, "model_family": model_family,
        "feature_dim": int(feature_dim), "layer": int(layer),
        "cv_protocol_version": CV_PROTOCOL_VERSION,
        "outer_fold": "" if outer_fold is None else int(outer_fold),
        "protein_id": str(protein_id), "pair_fold": "" if pair_fold is None else int(pair_fold),
        "sequence_gate": str(sequence_gate), "fit_status": str(fit_status),
        "support_flags": str(support_flags), "n_training_proteins": int(n_training_proteins),
        "n_training_pairs": int(n_training_pairs), "design_rank": design_rank,
        "largest_singular_value": largest_singular_value,
        "smallest_singular_value": smallest_singular_value,
        "condition_number": condition_number,
        "selected_configuration_id": str(selected_configuration_id),
        "final_boosting_rounds": "" if final_boosting_rounds is None else int(final_boosting_rounds),
        "training_row_cap": "" if training_row_cap is None else int(training_row_cap),
        "random_seed": "" if random_seed is None else int(random_seed),
    }
    return {column: row[column] for column in MODEL_STATUS_COLUMNS}


def build_runtime_row(
    *,
    analysis_name: str,
    model_family: str,
    task_id: int,
    layer: int,
    completion_status: str,
    n_fitted_proteins: int,
    n_evaluated_proteins: int,
    n_fitted_pairs: int,
    n_evaluated_pairs: int,
    elapsed_seconds: float,
    outer_fold: int | None = None,
    peak_rss_mb: float = float("nan"),
) -> dict[str, object]:
    if int(layer) not in LAYERS:
        raise ValueError("Runtime layer must be in 0..47")
    if outer_fold is not None and int(outer_fold) not in range(N_OUTER_FOLDS):
        raise ValueError("Runtime outer_fold is invalid")
    counts = (n_fitted_proteins, n_evaluated_proteins, n_fitted_pairs, n_evaluated_pairs)
    if int(task_id) < 0 or any(int(value) < 0 for value in counts):
        raise ValueError("Runtime task ID and support counts cannot be negative")
    if float(elapsed_seconds) < 0:
        raise ValueError("Elapsed time cannot be negative")
    row = {
        "analysis_name": analysis_name, "model_family": model_family,
        "task_id": int(task_id), "outer_fold": "" if outer_fold is None else int(outer_fold),
        "layer": int(layer), "completion_status": str(completion_status),
        "n_fitted_proteins": int(n_fitted_proteins),
        "n_evaluated_proteins": int(n_evaluated_proteins),
        "n_fitted_pairs": int(n_fitted_pairs), "n_evaluated_pairs": int(n_evaluated_pairs),
        "elapsed_seconds": float(elapsed_seconds), "peak_rss_mb": peak_rss_mb,
    }
    return {column: row[column] for column in RUNTIME_COLUMNS}


def build_model_stability_rows(
    coefficient_blocks: Sequence[np.ndarray],
    *,
    analysis_name: str,
    model_family: str,
    layer: int,
    feature_names: Sequence[str],
    protein_id: str = "",
    sequence_gate: str = "",
) -> list[dict[str, object]]:
    coefficients = np.asarray(coefficient_blocks, dtype=np.float64)
    names = tuple(str(name) for name in feature_names)
    if coefficients.ndim != 2 or coefficients.shape[0] < 1 or coefficients.shape[1] != len(names):
        raise ValueError("Coefficient stability requires a nonempty model-by-feature matrix")
    if not np.all(np.isfinite(coefficients)) or len(set(names)) != len(names):
        raise ValueError("Coefficient stability inputs must be finite with unique feature names")
    if int(layer) not in LAYERS:
        raise ValueError("Coefficient stability layer must be in 0..47")
    signs = np.sign(coefficients)
    rows = []
    for index, name in enumerate(names):
        values = coefficients[:, index]
        positive = float(np.mean(signs[:, index] > 0))
        negative = float(np.mean(signs[:, index] < 0))
        zero = float(np.mean(signs[:, index] == 0))
        rows.append({
            "analysis_name": analysis_name, "model_family": model_family,
            "feature_dim": len(names), "layer": int(layer), "protein_id": str(protein_id),
            "sequence_gate": str(sequence_gate), "feature_index": index, "feature_name": name,
            "n_models": int(coefficients.shape[0]),
            "coefficient_standardized_mean": float(np.mean(values)),
            "coefficient_standardized_std": float(np.std(values, ddof=0)),
            "coefficient_standardized_min": float(np.min(values)),
            "coefficient_standardized_max": float(np.max(values)),
            "sign_consistency_fraction": max(positive, negative, zero),
        })
    return [{column: row[column] for column in MODEL_STABILITY_COLUMNS} for row in rows]


def build_feature_importance_rows(
    gain_by_fold: np.ndarray,
    split_count_by_fold: np.ndarray,
    *,
    analysis_name: str,
    model_family: str,
    layer: int,
    feature_names: Sequence[str] = FEATURE_NAMES,
) -> list[dict[str, object]]:
    """Build fold-summarized XGBoost importance rows, including zero-use features."""
    gains = np.asarray(gain_by_fold, dtype=np.float64)
    splits = np.asarray(split_count_by_fold, dtype=np.float64)
    names = tuple(str(name) for name in feature_names)
    expected_shape = (N_OUTER_FOLDS, len(names))
    if gains.shape != expected_shape or splits.shape != expected_shape:
        raise ValueError(f"Feature importance arrays must have shape {expected_shape}")
    if not np.all(np.isfinite(gains)) or not np.all(np.isfinite(splits)):
        raise ValueError("Feature importance arrays must be finite")
    if np.any(gains < 0) or np.any(splits < 0) or len(set(names)) != len(names):
        raise ValueError("Feature importances must be nonnegative with unique names")
    if int(layer) not in LAYERS:
        raise ValueError("Feature-importance layer must be in 0..47")
    rows = []
    for index, name in enumerate(names):
        row = {
            "analysis_name": analysis_name, "model_family": model_family,
            "feature_dim": len(names), "layer": int(layer),
            "cv_protocol_version": CV_PROTOCOL_VERSION, "feature_index": index,
            "feature_name": name, "n_outer_folds": N_OUTER_FOLDS,
            "gain_mean": float(np.mean(gains[:, index])),
            "gain_std": float(np.std(gains[:, index], ddof=0)),
            "split_count_mean": float(np.mean(splits[:, index])),
            "split_count_std": float(np.std(splits[:, index], ddof=0)),
        }
        rows.append({column: row[column] for column in FEATURE_IMPORTANCE_COLUMNS})
    return rows


@dataclass
class _SubgroupCell:
    stats: RegressionSufficientStatistics = field(default_factory=RegressionSufficientStatistics)
    n_expected_pairs: int = 0
    protein_ids: set[str] = field(default_factory=set)


class OOFSubgroupAccumulator:
    """Pair-pooled OOF subgroup statistics without baseline duplication."""

    def __init__(self, category_scope: str, category_labels: Sequence[str]):
        labels = tuple(str(label) for label in category_labels)
        if not category_scope or not labels or len(set(labels)) != len(labels):
            raise ValueError("Subgroup scope and labels must be nonempty and unique")
        self.category_scope = str(category_scope)
        self.category_labels = labels
        self.cells = [_SubgroupCell() for _ in labels]
        self._seen_proteins: set[str] = set()

    @property
    def protein_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._seen_proteins))

    def update_protein(
        self,
        protein_id: str,
        y_true: np.ndarray,
        predictions: np.ndarray,
        category_codes: np.ndarray,
    ) -> None:
        protein = str(protein_id)
        if not protein or protein in self._seen_proteins:
            raise ValueError("Each protein may update a subgroup accumulator exactly once")
        y = _as_float_vector(y_true, "y_true")
        pred = _as_float_vector(predictions, "predictions")
        codes = np.asarray(category_codes)
        if y.shape != pred.shape or y.shape != codes.shape or not np.all(np.isfinite(y)):
            raise ValueError("Subgroup targets, predictions, and codes must match")
        if not np.issubdtype(codes.dtype, np.integer):
            raise ValueError("Subgroup codes must have an integer dtype")
        if np.any((codes < -1) | (codes >= len(self.category_labels))):
            raise ValueError("Subgroup codes must be -1 or a valid category index")
        self._seen_proteins.add(protein)
        for category, cell in enumerate(self.cells):
            expected = codes == category
            cell.n_expected_pairs += int(np.count_nonzero(expected))
            evaluated = expected & np.isfinite(pred)
            if np.any(evaluated):
                cell.protein_ids.add(protein)
                cell.stats.update(y[evaluated], pred[evaluated])

    def merge(self, other: "OOFSubgroupAccumulator") -> "OOFSubgroupAccumulator":
        if (
            self.category_scope != other.category_scope
            or self.category_labels != other.category_labels
        ):
            raise ValueError("Cannot merge subgroup accumulators with different categories")
        overlap = self._seen_proteins & other._seen_proteins
        if overlap:
            raise ValueError(f"Cannot merge duplicate subgroup proteins: {sorted(overlap)[:5]}")
        self._seen_proteins.update(other._seen_proteins)
        for destination, source in zip(self.cells, other.cells):
            destination.stats.merge(source.stats)
            destination.n_expected_pairs += source.n_expected_pairs
            destination.protein_ids.update(source.protein_ids)
        return self

    def to_payload(self) -> dict[str, object]:
        return {
            "category_scope": self.category_scope,
            "category_labels": list(self.category_labels),
            "seen_proteins": sorted(self._seen_proteins),
            "cells": [
                {
                    "stats": cell.stats.to_payload(),
                    "n_expected_pairs": int(cell.n_expected_pairs),
                    "protein_ids": sorted(cell.protein_ids),
                }
                for cell in self.cells
            ],
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, object]) -> "OOFSubgroupAccumulator":
        if set(payload) != {
            "category_scope", "category_labels", "seen_proteins", "cells",
        }:
            raise ValueError("Unexpected subgroup-accumulator payload")
        labels = payload["category_labels"]
        cells = payload["cells"]
        seen = payload["seen_proteins"]
        if not isinstance(labels, list) or not isinstance(cells, list) or not isinstance(seen, list):
            raise ValueError("Subgroup-accumulator payload lists are invalid")
        result = cls(str(payload["category_scope"]), [str(label) for label in labels])
        if len(cells) != len(result.cells):
            raise ValueError("Subgroup-accumulator cell count is invalid")
        result._seen_proteins = {str(protein_id) for protein_id in seen}
        if len(result._seen_proteins) != len(seen) or any(not value for value in result._seen_proteins):
            raise ValueError("Subgroup-accumulator seen proteins are invalid")
        for destination, raw in zip(result.cells, cells):
            if not isinstance(raw, Mapping) or set(raw) != {
                "stats", "n_expected_pairs", "protein_ids",
            }:
                raise ValueError("Subgroup-accumulator cell payload is invalid")
            proteins = raw["protein_ids"]
            if not isinstance(proteins, list):
                raise ValueError("Subgroup cell protein IDs must be a list")
            destination.stats = RegressionSufficientStatistics.from_payload(raw["stats"])
            destination.n_expected_pairs = int(raw["n_expected_pairs"])
            destination.protein_ids = {str(protein_id) for protein_id in proteins}
            if (
                destination.n_expected_pairs < destination.stats.n
                or len(destination.protein_ids) != len(proteins)
                or not destination.protein_ids.issubset(result._seen_proteins)
            ):
                raise ValueError("Subgroup-accumulator cell support is invalid")
        return result

    @staticmethod
    def _support_status(n_pairs: int, n_proteins: int) -> str:
        if n_pairs == 0:
            return "empty"
        if n_pairs < 100 and n_proteins < 5:
            return "low_pair_and_protein_support"
        if n_pairs < 100:
            return "low_pair_support"
        if n_proteins < 5:
            return "low_protein_support"
        return "sufficient"

    def rows(
        self,
        *,
        analysis_name: str,
        model_family: str,
        feature_dim: int,
        layer: int,
        evaluation_design: str,
        oof_prediction_scope: str,
    ) -> list[dict[str, object]]:
        identity = _identity(
            analysis_name=analysis_name, model_family=model_family, feature_dim=feature_dim,
            layer=layer, evaluation_design=evaluation_design,
            oof_prediction_scope=oof_prediction_scope,
        )
        result = []
        for label, cell in zip(self.category_labels, self.cells):
            evaluated = cell.stats.n
            coverage = (
                float("nan") if cell.n_expected_pairs == 0
                else evaluated / cell.n_expected_pairs
            )
            row = {
                **identity, "category_scope": self.category_scope, "category_label": label,
                "n_proteins": len(cell.protein_ids), "n_expected_pairs": cell.n_expected_pairs,
                "n_evaluated_pairs": evaluated, "pair_coverage": coverage,
                "support_status": self._support_status(evaluated, len(cell.protein_ids)),
                **_prefixed_metrics(cell.stats.metrics(), "pooled_"),
            }
            result.append({column: row[column] for column in SUBGROUP_METRICS_COLUMNS})
        return result


def _empty_geometry_metrics() -> dict[str, float]:
    return {metric: float("nan") for metric in GEOMETRY_METRIC_NAMES}


def build_geometry_row(
    *,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    layer: int,
    protein_id: str,
    source: str,
    outer_fold: int,
    evaluation_design: str,
    oof_prediction_scope: str,
    n_residues: int,
    true_coordinates: np.ndarray,
    predictions: np.ndarray,
) -> dict[str, object]:
    residues = int(n_residues)
    if int(outer_fold) not in range(N_OUTER_FOLDS):
        raise ValueError("Geometry outer_fold is invalid")
    if str(source) not in {"cameo", "ecod"}:
        raise ValueError("Geometry source must be cameo or ecod")
    pair_i, pair_j = strict_lower_triangle_indices(residues)
    pred = _as_float_vector(predictions, "predictions")
    if residues < 2 or pred.size != pair_i.size:
        raise ValueError("Geometry requires one prediction for every canonical unordered pair")
    n_evaluated = int(np.count_nonzero(np.isfinite(pred)))
    identity = {
        **_identity(
            analysis_name=analysis_name, model_family=model_family, feature_dim=feature_dim,
            layer=layer, evaluation_design=evaluation_design,
            oof_prediction_scope=oof_prediction_scope,
        ),
        "protein_id": str(protein_id), "source": str(source), "outer_fold": int(outer_fold),
        "n_residues": residues, "n_protein_pairs": int(pair_i.size),
        "pair_coverage": n_evaluated / pair_i.size,
    }
    if n_evaluated != pair_i.size:
        row = {
            **identity, "geometry_status": "incomplete_pair_coverage",
            "geometry_exclusion_reason": f"{n_evaluated}/{pair_i.size} finite OOF pair predictions",
            **_empty_geometry_metrics(),
        }
        return {column: row[column] for column in GEOMETRY_METRICS_COLUMNS}
    try:
        raw = analyze_protein_geometry(
            residues, np.asarray(true_coordinates, dtype=np.float64), pair_i, pair_j, pred,
        )
        metrics = {
            "rmsd": float(raw["rmsd"]), "tm_score": float(raw["tm_score"]),
            "frac_top3_eigenvalues": float(raw["frac_top3_eigenvalues"]),
            "n_negative_eigenvalues": int(raw["n_negative_eigenvalues"]),
            "negative_eigenvalue_fraction": float(raw["neg_eigenvalue_fraction"]),
            "stress": float(raw["stress"]),
            "n_negative_predictions": int(raw["n_negative_predictions"]),
            "n_triangles": math.comb(residues, 3),
            "triangle_violation_fraction": float(raw["triangle_violation_fraction"]),
            "triangle_mean_violation": float(raw["triangle_mean_violation"]),
            "triangle_max_violation": float(raw["triangle_max_violation"]),
            "triangle_violation_0_5_fraction": float(raw["triangle_violation_0_5_fraction"]),
            "triangle_violation_5_10_fraction": float(raw["triangle_violation_5_10_fraction"]),
            "triangle_violation_10_plus_fraction": float(raw["triangle_violation_10_plus_fraction"]),
        }
        if not all(np.isfinite(value) for value in metrics.values()):
            raise ValueError("Geometry calculation returned nonfinite metrics")
        row = {
            **identity, "geometry_status": "complete", "geometry_exclusion_reason": "",
            **metrics,
        }
    except TMScoreError:
        raise
    except Exception as exc:
        row = {
            **identity, "geometry_status": "geometry_failed",
            "geometry_exclusion_reason": f"{type(exc).__name__}: {exc}",
            **_empty_geometry_metrics(),
        }
    return {column: row[column] for column in GEOMETRY_METRICS_COLUMNS}


def summarize_geometry_rows(rows: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    group_fields = _METRIC_IDENTITY_COLUMNS
    grouped: dict[tuple[object, ...], list[Mapping[str, object]]] = {}
    for row in rows:
        key = tuple(row[field] for field in group_fields)
        grouped.setdefault(key, []).append(row)
    summaries = []
    for key in sorted(grouped, key=lambda value: tuple(map(str, value))):
        group = grouped[key]
        included = [row for row in group if row["geometry_status"] == "complete"]
        identity = dict(zip(group_fields, key))
        for metric in GEOMETRY_METRIC_NAMES:
            values = np.asarray([float(row[metric]) for row in included], dtype=np.float64)
            values = values[np.isfinite(values)]
            statistics = (
                {
                    "mean": float(np.mean(values)), "std": float(np.std(values, ddof=0)),
                    "median": float(np.median(values)), "q25": float(np.quantile(values, 0.25)),
                    "q75": float(np.quantile(values, 0.75)),
                }
                if values.size else
                {name: float("nan") for name in ("mean", "std", "median", "q25", "q75")}
            )
            row = {
                **identity, "geometry_metric": metric, "units": GEOMETRY_METRIC_UNITS[metric],
                "n_proteins_attempted": len(group), "n_proteins_included": len(included),
                "n_proteins_excluded": len(group) - len(included), **statistics,
            }
            summaries.append({column: row[column] for column in GEOMETRY_SUMMARY_COLUMNS})
    return summaries


def _relative_artifact_path(path: str | Path) -> Path:
    relative = Path(path)
    if relative.is_absolute() or not relative.parts or ".." in relative.parts:
        raise ValueError("Artifact paths must be safe paths relative to the family root")
    return relative


def _resolve_artifact_path(root: str | Path, relative: Path) -> Path:
    resolved_root = Path(root).resolve()
    resolved = (resolved_root / relative).resolve()
    if not resolved.is_relative_to(resolved_root):
        raise ValueError("Artifact path resolves outside its declared root")
    return resolved


def build_replay_index_row(
    *,
    family_root: str | Path,
    artifact_path: str | Path,
    analysis_name: str,
    model_family: str,
    layer: int,
    artifact_type: str,
    configuration_digest: str,
    scaler_digest: str,
    artifact_sha256: str | None = None,
    status: str = "complete",
    outer_fold: int | None = None,
    protein_id: str = "",
    pair_fold: int | None = None,
    sequence_gate: str = "",
) -> dict[str, object]:
    relative = _relative_artifact_path(artifact_path)
    resolved = _resolve_artifact_path(family_root, relative)
    if not resolved.is_file():
        raise FileNotFoundError(f"Replay artifact does not exist: {resolved}")
    if int(layer) not in LAYERS:
        raise ValueError("Replay artifact layer must be in 0..47")
    if not artifact_type or not configuration_digest or not status:
        raise ValueError("Replay artifact type, configuration digest, and status are required")
    if outer_fold is not None and int(outer_fold) not in range(N_OUTER_FOLDS):
        raise ValueError("Replay outer_fold is invalid")
    if pair_fold is not None and int(pair_fold) not in range(N_PAIR_FOLDS):
        raise ValueError("Replay pair_fold is invalid")
    row = {
        "replay_index_version": REPLAY_INDEX_VERSION,
        "analysis_name": analysis_name, "model_family": model_family, "layer": int(layer),
        "outer_fold": "" if outer_fold is None else int(outer_fold),
        "protein_id": str(protein_id), "pair_fold": "" if pair_fold is None else int(pair_fold),
        "sequence_gate": str(sequence_gate), "artifact_type": str(artifact_type),
        "artifact_path": relative.as_posix(),
        "artifact_sha256": (
            sha256_file(resolved) if artifact_sha256 is None else str(artifact_sha256)
        ),
        "configuration_digest": str(configuration_digest), "scaler_digest": str(scaler_digest),
        "status": str(status),
    }
    return {column: row[column] for column in REPLAY_INDEX_COLUMNS}


def validate_replay_index(
    rows: Sequence[Mapping[str, object]],
    *,
    family_root: str | Path,
    expected_count: int | None = None,
) -> None:
    if expected_count is not None and len(rows) != int(expected_count):
        raise ValueError(f"Expected {expected_count} replay rows, observed {len(rows)}")
    responsibilities: set[tuple[object, ...]] = set()
    artifact_digests: dict[str, str] = {}
    root = Path(family_root)
    for index, row in enumerate(rows):
        if tuple(row) != REPLAY_INDEX_COLUMNS:
            raise ValueError(f"Replay row {index} has an unexpected schema or column order")
        if row["replay_index_version"] != REPLAY_INDEX_VERSION:
            raise ValueError("Replay row uses an unexpected index version")
        relative = _relative_artifact_path(str(row["artifact_path"]))
        path_key = relative.as_posix()
        responsibility = tuple(row[column] for column in (
            "model_family", "layer", "outer_fold", "protein_id", "pair_fold", "sequence_gate",
        ))
        if responsibility in responsibilities:
            raise ValueError(f"Replay responsibility is duplicated: {responsibility}")
        responsibilities.add(responsibility)
        resolved = _resolve_artifact_path(root, relative)
        if not resolved.is_file():
            raise ValueError(f"Replay artifact is missing or checksum-mismatched: {resolved}")
        if path_key not in artifact_digests:
            artifact_digests[path_key] = sha256_file(resolved)
        observed_digest = artifact_digests[path_key]
        if observed_digest != row["artifact_sha256"]:
            raise ValueError(f"Replay artifact is missing or checksum-mismatched: {resolved}")


def _checkpoint_artifact_records(
    checkpoint_root: Path,
    artifacts: Mapping[str, str | Path],
) -> dict[str, dict[str, object]]:
    if not artifacts:
        raise ValueError("A complete checkpoint must contain at least one artifact")
    records: dict[str, dict[str, object]] = {}
    for name in sorted(artifacts):
        if not str(name):
            raise ValueError("Checkpoint artifact names cannot be empty")
        relative = _relative_artifact_path(artifacts[name])
        path = _resolve_artifact_path(checkpoint_root, relative)
        if not path.is_file():
            raise FileNotFoundError(f"Checkpoint artifact does not exist: {path}")
        records[str(name)] = {
            "path": relative.as_posix(),
            "size_bytes": int(path.stat().st_size),
            "sha256": sha256_file(path),
        }
    return records


def commit_checkpoint(
    manifest_path: str | Path,
    *,
    invariants: Mapping[str, object],
    artifacts: Mapping[str, str | Path],
    completed_at: str | None = None,
) -> dict[str, object]:
    """Atomically publish a completion manifest after checksumming all artifacts."""
    destination = Path(manifest_path)
    invariant_payload = dict(invariants)
    if not invariant_payload:
        raise ValueError("Checkpoint invariants cannot be empty")
    manifest = {
        "checkpoint_manifest_version": CHECKPOINT_MANIFEST_VERSION,
        "status": "complete",
        "invariants": invariant_payload,
        "invariants_digest": semantic_digest(invariant_payload),
        "artifacts": _checkpoint_artifact_records(destination.parent, artifacts),
        "completed_at": completed_at or utc_now(),
    }
    atomic_write_json(destination, manifest)
    return manifest


def validate_resume_checkpoint(
    manifest_path: str | Path,
    *,
    expected_invariants: Mapping[str, object],
    expected_artifact_names: Iterable[str] | None = None,
) -> dict[str, object]:
    """Require exact invariants plus present, size-matched, checksummed artifacts."""
    path = Path(manifest_path)
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint manifest does not exist: {path}")
    with path.open() as handle:
        manifest = json.load(handle)
    required = {
        "checkpoint_manifest_version", "status", "invariants", "invariants_digest",
        "artifacts", "completed_at",
    }
    if set(manifest) != required:
        raise ValueError("Checkpoint manifest schema is invalid")
    if manifest["checkpoint_manifest_version"] != CHECKPOINT_MANIFEST_VERSION:
        raise ValueError("Checkpoint manifest version is incompatible")
    if manifest["status"] != "complete":
        raise ValueError("Checkpoint is not complete")
    expected = dict(expected_invariants)
    if manifest["invariants"] != expected or manifest["invariants_digest"] != semantic_digest(expected):
        raise ValueError("Checkpoint invariants do not match the requested task")
    artifacts = manifest["artifacts"]
    if not isinstance(artifacts, dict) or not artifacts:
        raise ValueError("Checkpoint artifact manifest is empty or invalid")
    if expected_artifact_names is not None:
        expected_names = {str(name) for name in expected_artifact_names}
        if set(artifacts) != expected_names:
            raise ValueError("Checkpoint artifact names do not match the resume contract")
    for name, record in artifacts.items():
        if set(record) != {"path", "size_bytes", "sha256"}:
            raise ValueError(f"Checkpoint artifact record is invalid: {name}")
        relative = _relative_artifact_path(record["path"])
        artifact = _resolve_artifact_path(path.parent, relative)
        if not artifact.is_file():
            raise ValueError(f"Checkpoint artifact is missing: {artifact}")
        if artifact.stat().st_size != int(record["size_bytes"]):
            raise ValueError(f"Checkpoint artifact size mismatch: {artifact}")
        if sha256_file(artifact) != record["sha256"]:
            raise ValueError(f"Checkpoint artifact checksum mismatch: {artifact}")
    return manifest


def validate_shard_inventory(
    root: str | Path,
    *,
    expected_shards: Mapping[str, int],
    checkpoint_filename: str,
) -> None:
    """Require one complete physical shard directory per expected task identity."""
    expected = {str(name): int(task_id) for name, task_id in expected_shards.items()}
    if not expected:
        raise ValueError("Expected shard inventory cannot be empty")
    if any(Path(name).name != name or name in {"", ".", ".."} for name in expected):
        raise ValueError("Expected shard names must be immediate directory names")
    if len(set(expected.values())) != len(expected):
        raise ValueError("Expected shard inventory contains duplicate task identities")
    if not str(checkpoint_filename) or Path(checkpoint_filename).name != checkpoint_filename:
        raise ValueError("Checkpoint filename must be an immediate child name")

    shard_root = Path(root) / "layer_shards"
    if not shard_root.is_dir():
        raise ValueError(f"Missing layer shard directory: {shard_root}")
    observed_entries = {path.name: path for path in shard_root.iterdir()}
    observed_names = set(observed_entries)
    missing = sorted(set(expected) - observed_names)
    unexpected = sorted(observed_names - set(expected))

    identities: dict[int, str] = {}
    for name, shard in sorted(observed_entries.items()):
        if not shard.is_dir():
            continue
        checkpoint_path = shard / checkpoint_filename
        if not checkpoint_path.is_file():
            if name in expected:
                raise ValueError(
                    f"Shard {name} is incomplete: missing {checkpoint_filename}"
                )
            continue
        try:
            with checkpoint_path.open() as handle:
                checkpoint = json.load(handle)
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"Shard {name} has an invalid completion checkpoint") from exc
        if checkpoint.get("status") != "complete":
            raise ValueError(f"Shard {name} is incomplete")
        invariants = checkpoint.get("invariants")
        if not isinstance(invariants, dict) or "task_id" not in invariants:
            raise ValueError(f"Shard {name} checkpoint lacks a task identity")
        try:
            task_id = int(invariants["task_id"])
        except (TypeError, ValueError) as exc:
            raise ValueError(f"Shard {name} checkpoint has an invalid task identity") from exc
        if task_id in identities:
            raise ValueError(
                "Duplicate physical shard task identity "
                f"{task_id}: {identities[task_id]} and {name}"
            )
        identities[task_id] = name
        if name in expected and task_id != expected[name]:
            raise ValueError(
                f"Shard {name} has task identity {task_id}; expected {expected[name]}"
            )

    if missing or unexpected or len(observed_entries) != len(expected):
        raise ValueError(
            "Shard inventory mismatch: "
            f"expected={len(expected)}, observed={len(observed_entries)}, "
            f"missing={missing[:5]}, unexpected={unexpected[:5]}"
        )


def write_v2_csv(
    path: str | Path,
    logical_filename: str,
    rows: Sequence[Mapping[str, object]],
) -> None:
    """Atomically write one output-schema-2.0 table with strict column order."""
    if logical_filename not in V2_CSV_SCHEMAS:
        raise ValueError(f"Unknown v2 CSV output: {logical_filename}")
    columns = V2_CSV_SCHEMAS[logical_filename]
    normalized = []
    for index, row in enumerate(rows):
        if tuple(row) != columns:
            raise ValueError(f"V2 CSV row {index} does not match {logical_filename}")
        normalized.append(row)
    atomic_write_csv(path, columns, normalized)


def write_v2_npz(path: str | Path, **arrays: np.ndarray) -> None:
    """Expose the common atomic compressed-array writer to v2 runners."""
    if not arrays:
        raise ValueError("An NPZ artifact must contain at least one array")
    atomic_savez_compressed(path, **arrays)


def build_run_manifest(
    *,
    fold_bundle: CVFoldBundle,
    analysis_name: str,
    model_family: str,
    feature_dim: int,
    evaluation_design: str,
    oof_prediction_scope: str,
    requested_task_ids: Sequence[int],
    completed_task_ids: Sequence[int],
    configuration: Mapping[str, object],
    expected_replay_artifacts: int,
    versions: Mapping[str, str] | None = None,
    code_revision: str = "",
    run_status: str = "complete",
    created_at: str | None = None,
) -> dict[str, object]:
    requested_values = [int(task_id) for task_id in requested_task_ids]
    completed_values = [int(task_id) for task_id in completed_task_ids]
    if len(requested_values) != len(set(requested_values)):
        raise ValueError("Requested task IDs contain duplicates")
    if len(completed_values) != len(set(completed_values)):
        raise ValueError("Completed task IDs contain duplicates")
    requested = sorted(requested_values)
    completed = sorted(completed_values)
    if not requested or not set(completed).issubset(requested):
        raise ValueError("Completed task IDs must be a subset of a nonempty request")
    if int(feature_dim) < 1 or int(feature_dim) > len(FEATURE_NAMES):
        raise ValueError("Run-manifest feature dimension is invalid")
    configuration_payload = dict(configuration)
    if not configuration_payload:
        raise ValueError("Run configuration cannot be empty")
    return {
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "cv_protocol_version": CV_PROTOCOL_VERSION,
        "result_root_name": RESULT_ROOT_NAME,
        "analysis_name": analysis_name,
        "model_family": model_family,
        "feature_dim": int(feature_dim),
        "feature_names": list(FEATURE_NAMES[: int(feature_dim)]),
        "evaluation_design": evaluation_design,
        "evaluation_scope": EVALUATION_SCOPE,
        "oof_prediction_scope": oof_prediction_scope,
        "cohort_digest": fold_bundle.fold_digests["cohort_digest"],
        "outer_fold_digest": fold_bundle.fold_digests["outer_fold_digest"],
        "pair_fold_manifest_digest": fold_bundle.fold_digests["pair_fold_manifest_digest"],
        "requested_task_ids": requested,
        "completed_task_ids": completed,
        "expected_replay_artifacts": int(expected_replay_artifacts),
        "configuration": configuration_payload,
        "configuration_digest": semantic_digest(configuration_payload),
        "versions": dict(versions or {}),
        "code_revision": str(code_revision),
        "run_status": str(run_status),
        "created_at": created_at or utc_now(),
    }


__all__ = [
    "CHECKPOINT_MANIFEST_VERSION", "CVFoldBundle", "CV_PROTOCOL_VERSION",
    "EVALUATION_DESIGNS", "EVALUATION_SCOPE", "FEATURE_IMPORTANCE_COLUMNS",
    "FEATURE_IMPORTANCE_FILENAME", "FOLD_METRICS_COLUMNS", "FOLD_METRICS_FILENAME",
    "FOLD_SUMMARY_COLUMNS", "FOLD_SUMMARY_FILENAME", "FoldLayerTask",
    "FittedStandardizer", "GEOMETRY_METRICS_COLUMNS", "GEOMETRY_METRICS_FILENAME",
    "GEOMETRY_SUMMARY_COLUMNS", "GEOMETRY_SUMMARY_FILENAME", "GLOBAL_BASELINE_TYPE",
    "MODEL_STABILITY_COLUMNS", "MODEL_STABILITY_FILENAME", "MODEL_STATUS_COLUMNS",
    "MODEL_STATUS_FILENAME", "OOFComparisonAccumulator", "OOFPairAssembler",
    "OOFSubgroupAccumulator", "OUTPUT_SCHEMA_VERSION", "PER_PROTEIN_COLUMNS",
    "PER_PROTEIN_FILENAME", "POOLED_METRICS_COLUMNS", "POOLED_METRICS_FILENAME",
    "PreparedTrainingFold", "REPLAY_INDEX_COLUMNS", "REPLAY_INDEX_FILENAME",
    "REPLAY_INDEX_VERSION", "RESULT_ROOT_NAME", "RUNTIME_COLUMNS", "RUNTIME_FILENAME",
    "RegressionSufficientStatistics", "SAMPLING_ALGORITHM", "SAMPLING_NAMESPACE",
    "STABLE_FIT_METRICS_COLUMNS", "STABLE_FIT_METRICS_FILENAME",
    "SUBGROUP_METRICS_COLUMNS", "SUBGROUP_SEQUENCE_FILENAME", "SUBGROUP_SPATIAL_FILENAME",
    "TRAINING_METRICS_COLUMNS", "TRAINING_METRICS_FILENAME", "TrainingBaseline",
    "V2_CSV_SCHEMAS", "build_fold_metrics_row", "build_fold_summary_rows",
    "build_feature_importance_rows", "build_geometry_row", "build_model_stability_rows",
    "build_model_status_row",
    "build_per_protein_row", "build_pooled_metrics_row", "build_replay_index_row",
    "build_run_manifest", "build_runtime_row", "build_stable_fit_metrics_row",
    "build_training_metrics_row", "commit_checkpoint", "deterministic_sample_indices",
    "deterministic_sample_seed", "fit_conditional_train_only_baselines",
    "fit_train_only_baseline", "fit_train_only_standardizer", "fivefold_result_root",
    "fold_layer_task_id", "load_cv_folds", "map_fold_layer_task", "map_layer_task",
    "numeric_array_digest", "prepare_training_fold", "sampling_assignment_digest",
    "semantic_digest", "summarize_geometry_rows", "utc_now", "validate_replay_index",
    "validate_resume_checkpoint", "validate_shard_inventory", "write_v2_csv",
    "write_v2_npz",
]

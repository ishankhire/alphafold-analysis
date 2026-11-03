"""Contracts for replaying frozen shared-OLS probes across Evoformer layers.

A probe fitted at a *source* layer is applied, without refitting, to the pair
representation of a *target* layer for the same held-out outer-fold proteins.
Two transfer modes are evaluated:

``raw``
    The frozen source probe exactly as fitted (source scaler, standardized
    coefficients, and intercept).
``target_restandardized``
    The source standardized coefficients and intercept with the target layer's
    own same-fold, train-only feature mean and scale.  The target scaler comes
    from the frozen target-layer shared-OLS artifact, which was fitted on the
    identical training proteins, so this remains an out-of-fold replay.

Workers store additive sufficient-statistic primitives so every per-protein,
fold, pooled, and subgroup metric can be reconstructed exactly by summation.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from cross_validation_common import FoldLayerTask, RegressionSufficientStatistics
from regression_protocol import SEQUENCE_BIN_LABELS, SPATIAL_DISTANCE_BIN_LABELS
from shared_ols_base256_common import load_coefficient_artifact as load_shared_ols
from triangle_violation_subgroup_common import (
    FAMILY_BY_NAME,
    index_replay_rows,
    load_and_validate_replay_index,
    sha256_file,
    validate_family_manifest,
)
from triangle_violation_subgroup_replay import predict_ols


ANALYSIS_NAME = "cross_layer_probe_transfer_v1"
OUTPUT_ENV = "CROSS_LAYER_TRANSFER_OUTPUT_ROOT"
DEFAULT_OUTPUT_SUBDIR = Path("diagnostic_experiments") / ANALYSIS_NAME
PRODUCER_ANALYSIS = "shared_ols_base256"
PRODUCER_SPEC = FAMILY_BY_NAME[PRODUCER_ANALYSIS]
MODEL_FAMILY = "shared_ols"
FEATURE_DIM = 256
CV_PROTOCOL_VERSION = "2.0"
EVALUATION_DESIGN = "protein_outer_5fold"
EVALUATION_SCOPE = "oof_all_pairs"
OOF_PREDICTION_SCOPE = "heldout_proteins_all_pairs"
EXPECTED_PROTEINS = 552
N_OUTER_FOLDS = 5
N_CANONICAL_LAYERS = 48

MID_LAYERS = (22, 23, 24, 25, 26)
END_LAYERS = (43, 44, 45, 46, 47)
TRANSFER_LAYERS = MID_LAYERS + END_LAYERS
MODES = ("raw", "target_restandardized")
ANALYSIS_BY_MODE = {
    "raw": "shared_ols_base256_xfer_raw",
    "target_restandardized": "shared_ols_base256_xfer_target_restandardized",
}

CATEGORY_LABELS = {
    "all_pairs": ("all",),
    "sequence_separation": tuple(SEQUENCE_BIN_LABELS),
    "spatial_distance": tuple(SPATIAL_DISTANCE_BIN_LABELS),
}
CELLS_PER_PREDICTION = sum(len(labels) for labels in CATEGORY_LABELS.values())  # 12
ROWS_PER_PROTEIN = len(TRANSFER_LAYERS) * len(MODES) * CELLS_PER_PREDICTION + 1  # 241

STAT_FIELDS = (
    "n", "sum_y", "sum_y2", "sum_prediction", "sum_error", "sum_abs_error",
    "sum_squared_error",
)
PRIMITIVE_COLUMNS = (
    "analysis_name", "target_layer", "source_layer", "transfer_mode", "predictor",
    "protein_id", "source", "outer_fold", "n_residues", "category_scope",
    "category_label", *STAT_FIELDS,
)
TRANSFER_COLUMNS = (
    "source_layer", "target_layer", "transfer_mode", "source_group", "target_group",
    "transfer_block", "is_same_layer",
)
WORKER_RUNTIME_COLUMNS = (
    "stage", "target_layer", "protein_id", "started_at", "finished_at",
    "elapsed_seconds", "peak_rss_mb", "status", "detail",
)


def layer_group(layer: int) -> str:
    if int(layer) in MID_LAYERS:
        return "mid"
    if int(layer) in END_LAYERS:
        return "end"
    raise ValueError(f"Layer {layer} is not a cross-layer transfer layer")


def transfer_identity(source_layer: int, target_layer: int, mode: str) -> dict[str, object]:
    if mode not in MODES:
        raise ValueError(f"Unknown transfer mode {mode!r}")
    source_group = layer_group(source_layer)
    target_group = layer_group(target_layer)
    return {
        "source_layer": int(source_layer),
        "target_layer": int(target_layer),
        "transfer_mode": mode,
        "source_group": source_group,
        "target_group": target_group,
        "transfer_block": f"{source_group}_to_{target_group}",
        "is_same_layer": int(source_layer) == int(target_layer),
    }


def target_layer_for_task(task_id: int) -> int:
    task = int(task_id)
    if task not in range(len(TRANSFER_LAYERS)):
        raise ValueError(f"Task id must be in 0..{len(TRANSFER_LAYERS) - 1}; got {task_id}")
    return TRANSFER_LAYERS[task]


def _checked_artifact(root: Path, row: Mapping[str, str]) -> tuple[Path, str]:
    path = root / row["artifact_path"]
    if not path.is_file():
        raise FileNotFoundError(path)
    digest = sha256_file(path)
    if digest != row["artifact_sha256"]:
        raise ValueError(f"Artifact checksum mismatch: {path}")
    return path, digest


def load_transfer_artifacts(
    producer_root: str | Path, fold_bundle,
) -> tuple[dict[tuple[int, int], dict[str, object]], dict[str, object], list[dict[str, object]]]:
    """Load the 50 frozen shared-OLS fits (10 layers x 5 folds) with checks.

    Returns ``(models, provenance, status_rows)`` where ``models`` is keyed by
    ``(layer, outer_fold)``.
    """
    root = Path(producer_root) / PRODUCER_SPEC.family_dir
    manifest = validate_family_manifest(root, PRODUCER_SPEC, fold_bundle.fold_digests)
    rows = load_and_validate_replay_index(root, PRODUCER_SPEC, manifest, verify_artifacts=False)
    index = index_replay_rows(rows, ("layer", "outer_fold"))
    status_by_key = _producer_model_status(root)
    models: dict[tuple[int, int], dict[str, object]] = {}
    status_rows: list[dict[str, object]] = []
    for layer in TRANSFER_LAYERS:
        for fold in range(N_OUTER_FOLDS):
            row = index[(str(layer), str(fold))]
            path, digest = _checked_artifact(root, row)
            fit = load_shared_ols(
                path,
                expected_task=FoldLayerTask(fold * N_CANONICAL_LAYERS + layer, fold, layer),
                expected_scaler_digest=row["scaler_digest"],
            )
            if not str(fit["fit_status"]).startswith("fitted"):
                raise ValueError(f"Shared OLS layer {layer} fold {fold} is not a completed fit")
            models[(layer, fold)] = fit
            status = status_by_key.get((layer, fold), {})
            status_rows.append({
                "source_layer": layer, "outer_fold": fold,
                "fit_status": str(fit["fit_status"]),
                "support_flags": status.get("support_flags", ""),
                "condition_number": float(status.get("condition_number", "nan") or "nan"),
                "artifact_path": row["artifact_path"], "artifact_sha256": digest,
                "scaler_digest": row["scaler_digest"],
                "global_baseline_value": float(fit["global_baseline_value"]),
            })
    for fold in range(N_OUTER_FOLDS):
        values = {float(models[(layer, fold)]["global_baseline_value"]) for layer in TRANSFER_LAYERS}
        if len(values) != 1:
            raise ValueError(f"Fold {fold} global baseline differs across layers: {sorted(values)}")
    provenance = {
        "producer_root": str(root),
        "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "replay_index_sha256": sha256_file(root / "model_artifacts/replay_index.csv"),
        "configuration_digest": manifest["configuration_digest"],
        "artifacts": [
            {key: row[key] for key in (
                "source_layer", "outer_fold", "artifact_path", "artifact_sha256", "scaler_digest",
            )}
            for row in status_rows
        ],
    }
    return models, provenance, status_rows


def _producer_model_status(root: Path) -> dict[tuple[int, int], dict[str, str]]:
    import csv

    path = root / "model_status.csv"
    if not path.is_file():
        return {}
    with path.open(newline="") as handle:
        return {
            (int(row["layer"]), int(row["outer_fold"])): row
            for row in csv.DictReader(handle)
        }


def global_baseline_by_fold(models: Mapping[tuple[int, int], Mapping[str, object]]) -> dict[int, float]:
    return {
        fold: float(models[(TRANSFER_LAYERS[0], fold)]["global_baseline_value"])
        for fold in range(N_OUTER_FOLDS)
    }


def predict_transfer(
    features: np.ndarray,
    source_fit: Mapping[str, object],
    target_fit: Mapping[str, object],
    mode: str,
) -> np.ndarray:
    """Apply a frozen source probe to target-layer features."""
    if mode == "raw":
        return predict_ols(features, source_fit)
    if mode != "target_restandardized":
        raise ValueError(f"Unknown transfer mode {mode!r}")
    restandardized = {
        "intercept_standardized": source_fit["intercept_standardized"],
        "coefficient_standardized": source_fit["coefficient_standardized"],
        "feature_mean": target_fit["feature_mean"],
        "feature_scale": target_fit["feature_scale"],
    }
    return predict_ols(features, restandardized)


def stats_payload(y_true: np.ndarray, predictions: np.ndarray) -> dict[str, float | int]:
    stats = RegressionSufficientStatistics()
    if np.asarray(y_true).size:
        stats.update(y_true, predictions)
    return stats.to_payload()


def primitive_rows(
    *,
    identity: Mapping[str, object],
    y_true: np.ndarray,
    predictions: np.ndarray,
    sequence_codes: np.ndarray,
    spatial_codes: np.ndarray,
) -> list[dict[str, object]]:
    """Emit the twelve additive category cells for one protein prediction."""
    y = np.asarray(y_true, dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    if y.shape != pred.shape or y.ndim != 1 or not np.all(np.isfinite(pred)):
        raise ValueError("Predictions must be finite and aligned with targets")
    rows = []
    for scope, labels in CATEGORY_LABELS.items():
        codes = (
            np.zeros(y.size, dtype=np.int16) if scope == "all_pairs"
            else sequence_codes if scope == "sequence_separation" else spatial_codes
        )
        for code, label in enumerate(labels):
            selected = np.asarray(codes) == code
            rows.append({
                **identity, "category_scope": scope, "category_label": label,
                **stats_payload(y[selected], pred[selected]),
            })
    return rows


def row_stats(row: Mapping[str, object]) -> RegressionSufficientStatistics:
    return RegressionSufficientStatistics.from_payload({
        "n": int(row["n"]), **{field: float(row[field]) for field in STAT_FIELDS if field != "n"},
    })


def convert_primitive_row(source: Mapping[str, str]) -> dict[str, object]:
    row: dict[str, object] = dict(source)
    for column in ("target_layer", "outer_fold", "n_residues", "n"):
        row[column] = int(source[column])
    row["source_layer"] = int(source["source_layer"]) if source["source_layer"] not in ("", "NaN") else ""
    for column in STAT_FIELDS[1:]:
        row[column] = float(source[column])
    return row


def validate_target_shard(
    rows: Sequence[Mapping[str, object]],
    target_layer: int,
    *,
    expected_proteins: int | None = EXPECTED_PROTEINS,
) -> set[str]:
    """Validate one target-layer shard; return the complete protein IDs."""
    keys = set()
    by_protein: dict[str, int] = {}
    for row in rows:
        if int(row["target_layer"]) != int(target_layer) or row["analysis_name"] != ANALYSIS_NAME:
            raise ValueError("Shard row has the wrong target layer or analysis")
        predictor = row["predictor"]
        if predictor == "probe":
            if int(row["source_layer"]) not in TRANSFER_LAYERS or row["transfer_mode"] not in MODES:
                raise ValueError("Probe row has an invalid source layer or mode")
        elif predictor == "global_baseline":
            if row["source_layer"] != "" or row["transfer_mode"] != "" or row["category_scope"] != "all_pairs":
                raise ValueError("Baseline row has an invalid identity")
        else:
            raise ValueError(f"Unknown predictor {predictor!r}")
        scope, label = str(row["category_scope"]), str(row["category_label"])
        if scope not in CATEGORY_LABELS or label not in CATEGORY_LABELS[scope]:
            raise ValueError(f"Unknown category {scope}/{label}")
        key = (
            str(row["protein_id"]), predictor, str(row["source_layer"]),
            str(row["transfer_mode"]), scope, label,
        )
        if key in keys:
            raise ValueError(f"Duplicate shard key {key}")
        keys.add(key)
        by_protein[str(row["protein_id"])] = by_protein.get(str(row["protein_id"]), 0) + 1
    partial = {protein for protein, count in by_protein.items() if count != ROWS_PER_PROTEIN}
    if partial:
        raise ValueError(f"Shard has partial protein groups: {sorted(partial)[:5]}")
    if expected_proteins is not None and len(by_protein) != expected_proteins:
        raise ValueError(f"Shard has {len(by_protein)} proteins, expected {expected_proteins}")
    return set(by_protein)


def finite_or_nan(value: float) -> float:
    return float(value) if math.isfinite(float(value)) else float("nan")


__all__ = [name for name in globals() if name.isupper()] + [
    "convert_primitive_row", "finite_or_nan", "global_baseline_by_fold", "layer_group",
    "load_transfer_artifacts", "predict_transfer", "primitive_rows", "row_stats",
    "stats_payload", "target_layer_for_task", "transfer_identity", "validate_target_shard",
]

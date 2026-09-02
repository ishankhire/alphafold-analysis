"""Contracts and reducers for publication per-protein subgroup MAE replay."""

from __future__ import annotations

import math
from collections import defaultdict
from typing import Mapping, Sequence

import numpy as np

from regression_protocol import SEQUENCE_BIN_LABELS, SPATIAL_DISTANCE_BIN_LABELS
from triangle_violation_subgroup_common import CV_PROTOCOL_VERSION, FAMILY_BY_NAME, FAMILY_SPECS


ANALYSIS_NAME = "publication_subgroup_mae_5fold_v1"
EVALUATION_SCOPE = "oof_all_pairs"
FEATURE_DIM = 256
EXPECTED_PROTEINS = 552
LAYERS = tuple(range(48))
CATEGORY_LABELS = {
    "sequence_separation": tuple(SEQUENCE_BIN_LABELS),
    "spatial_distance": tuple(SPATIAL_DISTANCE_BIN_LABELS),
}
PER_PROTEIN_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "protein_id", "source",
    "cv_protocol_version", "evaluation_design", "outer_fold", "evaluation_scope",
    "oof_prediction_scope", "generalization_scope", "category_scope", "category_label",
    "n_residues", "n_pairs", "sum_absolute_error", "mae", "support_status",
)
SUMMARY_COLUMNS = (
    "analysis_name", "model_family", "feature_dim", "layer", "cv_protocol_version",
    "evaluation_design", "evaluation_scope", "oof_prediction_scope", "category_scope",
    "category_label", "n_proteins_total", "n_proteins_finite", "protein_equal_mean_mae",
    "protein_sd_mae", "protein_se_mae", "pair_pooled_mae", "n_pairs",
)
RECONCILIATION_COLUMNS = (
    "analysis_name", "model_family", "layer", "category_scope", "category_label",
    "replayed_n_pairs", "producer_n_pairs", "replayed_pooled_mae", "producer_pooled_mae",
    "absolute_difference", "status",
)
RUNTIME_COLUMNS = (
    "stage", "layer", "protein_id", "started_at", "finished_at", "elapsed_seconds",
    "peak_rss_mb", "status", "detail",
)


def subgroup_rows(
    *,
    spec,
    layer: int,
    protein_id: str,
    source: str,
    outer_fold: int,
    n_residues: int,
    absolute_error: np.ndarray,
    sequence_codes: np.ndarray,
    spatial_codes: np.ndarray,
) -> list[dict[str, object]]:
    """Reduce one protein/model prediction vector into all eleven subgroup rows."""
    error = np.asarray(absolute_error, dtype=np.float64)
    sequence = np.asarray(sequence_codes, dtype=np.int16)
    spatial = np.asarray(spatial_codes, dtype=np.int16)
    if error.ndim != 1 or sequence.shape != error.shape or spatial.shape != error.shape:
        raise ValueError("Error and category-code vectors must be aligned one-dimensional arrays")
    if not np.all(np.isfinite(error)) or np.any(error < 0):
        raise ValueError("Absolute errors must be finite and nonnegative")
    rows: list[dict[str, object]] = []
    for category_scope, codes in (
        ("sequence_separation", sequence),
        ("spatial_distance", spatial),
    ):
        for code, label in enumerate(CATEGORY_LABELS[category_scope]):
            selected = codes == code
            n_pairs = int(np.count_nonzero(selected))
            sae = float(np.sum(error[selected], dtype=np.float64)) if n_pairs else 0.0
            rows.append({
                "analysis_name": spec.analysis_name,
                "model_family": spec.model_family,
                "feature_dim": FEATURE_DIM,
                "layer": int(layer),
                "protein_id": str(protein_id),
                "source": str(source),
                "cv_protocol_version": CV_PROTOCOL_VERSION,
                "evaluation_design": spec.evaluation_design,
                "outer_fold": int(outer_fold),
                "evaluation_scope": EVALUATION_SCOPE,
                "oof_prediction_scope": spec.oof_prediction_scope,
                "generalization_scope": spec.generalization_scope,
                "category_scope": category_scope,
                "category_label": label,
                "n_residues": int(n_residues),
                "n_pairs": n_pairs,
                "sum_absolute_error": sae,
                "mae": sae / n_pairs if n_pairs else math.nan,
                "support_status": "complete" if n_pairs else "empty_category",
            })
    return rows


def validate_complete_layer(
    rows: Sequence[Mapping[str, object]],
    layer: int,
    *,
    expected_proteins: int = EXPECTED_PROTEINS,
) -> None:
    expected_rows = expected_proteins * len(FAMILY_SPECS) * sum(map(len, CATEGORY_LABELS.values()))
    if len(rows) != expected_rows:
        raise ValueError(f"Layer {layer} has {len(rows)} rows, expected {expected_rows}")
    keys: set[tuple[str, str, str, str]] = set()
    proteins_by_family: dict[str, set[str]] = defaultdict(set)
    for row in rows:
        if int(row["layer"]) != layer or row["evaluation_scope"] != EVALUATION_SCOPE:
            raise ValueError(f"Layer {layer} contains wrong layer/scope identity")
        analysis = str(row["analysis_name"])
        if analysis not in FAMILY_BY_NAME:
            raise ValueError(f"Unknown model analysis {analysis}")
        scope = str(row["category_scope"])
        label = str(row["category_label"])
        if scope not in CATEGORY_LABELS or label not in CATEGORY_LABELS[scope]:
            raise ValueError(f"Unknown subgroup {scope}/{label}")
        protein_id = str(row["protein_id"])
        key = (analysis, protein_id, scope, label)
        if key in keys:
            raise ValueError(f"Duplicate subgroup row {key}")
        keys.add(key)
        proteins_by_family[analysis].add(protein_id)
        n_pairs = int(row["n_pairs"])
        sae = float(row["sum_absolute_error"])
        mae = float(row["mae"])
        status = str(row["support_status"])
        if n_pairs == 0:
            if status != "empty_category" or sae != 0.0 or not math.isnan(mae):
                raise ValueError("Empty subgroup rows have an invalid metric/status")
        elif status != "complete" or not math.isclose(mae, sae / n_pairs, rel_tol=1e-12, abs_tol=1e-12):
            raise ValueError("Supported subgroup row has inconsistent additive primitives")
    protein_sets = list(proteins_by_family.values())
    if set(proteins_by_family) != set(FAMILY_BY_NAME):
        raise ValueError("Model-family coverage is incomplete")
    if any(len(ids) != expected_proteins or ids != protein_sets[0] for ids in protein_sets):
        raise ValueError("Model families do not contain identical complete protein sets")


__all__ = [
    "ANALYSIS_NAME", "CATEGORY_LABELS", "EVALUATION_SCOPE", "EXPECTED_PROTEINS",
    "FEATURE_DIM", "LAYERS", "PER_PROTEIN_COLUMNS", "RECONCILIATION_COLUMNS",
    "RUNTIME_COLUMNS", "SUMMARY_COLUMNS", "subgroup_rows", "validate_complete_layer",
]

"""Contracts for replaying every frozen shared-OLS probe across all Evoformer layers.

A probe fitted at a *source* layer is applied, without refitting, to the pair
representation of a *target* layer for the same held-out outer-fold proteins.
Only ``target_restandardized`` is evaluated: the source standardized coefficients
and intercept with the target layer's own same-fold, train-only feature mean and
scale. The target scaler comes from the frozen target-layer artifact, which was
fitted on the identical training proteins, so this stays an out-of-fold replay.

Signed error is probe distance minus final structure distance. The target is
OpenFold's own final predicted structure, so an error measures how far a probe
reads from that structure.

Each target layer produces one additive shard holding per-protein sufficient
statistics, signed-error histograms, and exact error sums, so nothing at pair
level is stored and every table can be rebuilt by summation.

The cohort is an explicit setting, ``ALL_LAYER_COHORT``. ``full`` is the frozen
552-protein cohort. ``cameo`` is its 154 CAMEO proteins, used while the ECOD pair
tensors and final structures are unavailable; those results use the same frozen
probes but a different cohort, so they keep their own analysis name and output
root and must be re-run as ``full`` once ECOD returns.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from cross_layer_transfer_common import (
    CATEGORY_LABELS, CV_PROTOCOL_VERSION, EVALUATION_DESIGN, EVALUATION_SCOPE, EXPECTED_PROTEINS,
    FEATURE_DIM, MODEL_FAMILY, N_CANONICAL_LAYERS, N_OUTER_FOLDS, OOF_PREDICTION_SCOPE,
    PRODUCER_SPEC, STAT_FIELDS, WORKER_RUNTIME_COLUMNS, _checked_artifact, _producer_model_status,
)
from cross_validation_common import FoldLayerTask
from regression_protocol import SEQUENCE_BIN_LABELS, SPATIAL_DISTANCE_BIN_LABELS
from shared_ols_base256_common import load_coefficient_artifact as load_shared_ols
from triangle_violation_subgroup_common import (
    index_replay_rows, load_and_validate_replay_index, sha256_file, validate_family_manifest,
)


COHORT_ENV = "ALL_LAYER_COHORT"
COHORT_SIZES = {"cameo": 154, "full": EXPECTED_PROTEINS}
OUTPUT_ENV = "ALL_LAYER_TRANSFER_OUTPUT_ROOT"
V1_ROOT_ENV = "CROSS_LAYER_V1_ROOT"
DEFAULT_V1_ROOT = Path("/work/hdd/bfna/ikhire/diagnostic_experiments/cross_layer_probe_transfer_v1")
TRANSFER_MODE = "target_restandardized"
OUTLIER_PROTEIN = "7stz_C"
LAYERS = tuple(range(N_CANONICAL_LAYERS))
V1_LAYERS = (22, 23, 24, 25, 26, 43, 44, 45, 46, 47)
SHARD_SCHEMA = 1


def _error_edges() -> np.ndarray:
    positive = np.concatenate([
        np.arange(0.5, 20.0 + 1e-9, 0.5),
        np.arange(22.0, 60.0 + 1e-9, 2.0),
        np.arange(70.0, 200.0 + 1e-9, 10.0),
    ])
    return np.concatenate([[-np.inf], -positive[::-1], [0.0], positive, [np.inf]])


ERROR_EDGES = _error_edges()
SEQUENCE_EDGES = np.array([1, 2, 3, 4, 5, 7, 10, 15, 20, 30, 50, 100, 200, np.inf])
SEQUENCE_LABELS = ("1", "2", "3", "4", "5-6", "7-9", "10-14", "15-19", "20-29", "30-49",
                   "50-99", "100-199", "200+")
DISTANCE_EDGES = np.array([0, 4, 5, 6, 7, 8, 10, 12, 15, 20, 25, 30, 40, 50, 60, 80, 100, np.inf])
DISTANCE_LABELS = ("0-4", "4-5", "5-6", "6-7", "7-8", "8-10", "10-12", "12-15", "15-20", "20-25",
                   "25-30", "30-40", "40-50", "50-60", "60-80", "80-100", "100+")
CANONICAL_SEQUENCE_GROUPS = {"1": (0,), "2-4": (1, 2, 3), "5-9": (4, 5), "10-19": (6, 7),
                             "20-49": (8, 9), "50+": (10, 11, 12)}
CANONICAL_DISTANCE_GROUPS = {"0-5": (0, 1), "5-10": (2, 3, 4, 5), "10-20": (6, 7, 8),
                             "20-40": (9, 10, 11), "40+": (12, 13, 14, 15, 16)}
N_ERROR_BINS = ERROR_EDGES.size - 1
N_SEQUENCE_BINS = SEQUENCE_EDGES.size - 1
N_DISTANCE_BINS = DISTANCE_EDGES.size - 1
N_POSITION_BINS = 20
MIN_SUPPORT_PAIRS, MIN_SUPPORT_PROTEINS = 100, 5

# Whole-protein, then canonical sequence, then canonical distance: 12 additive cells.
CATEGORY_CELLS = tuple(
    (scope, label) for scope, labels in CATEGORY_LABELS.items() for label in labels
)
CELL_INDEX = {cell: index for index, cell in enumerate(CATEGORY_CELLS)}
ERROR_SUM_FIELDS = ("sum_error", "sum_abs_error", "sum_squared_error")

TRANSFER_COLUMNS = ("source_layer", "target_layer", "transfer_mode", "layer_offset", "is_same_layer")


def cohort_name() -> str:
    """The cohort must be stated; a silent default could mix partial and full results."""
    value = os.environ.get(COHORT_ENV, "")
    if value not in COHORT_SIZES:
        raise ValueError(f"{COHORT_ENV} must be one of {sorted(COHORT_SIZES)}; got {value!r}")
    return value


def analysis_name(cohort: str) -> str:
    return "cross_layer_probe_transfer_all_layers_v1" if cohort == "full" else (
        f"cross_layer_probe_transfer_all_layers_{cohort}_v1")


def analysis_label(cohort: str) -> str:
    base = "shared_ols_base256_xfer_all_target_restandardized"
    return base if cohort == "full" else f"{base}_{cohort}"


def default_output_root(cohort: str) -> Path:
    return Path.home() / "diagnostic_experiments" / analysis_name(cohort)


def select_cohort(fold_bundle, cohort: str) -> tuple[str, ...]:
    """Cohort proteins in frozen order; every outer fold must stay represented."""
    protein_ids = tuple(fold_bundle.eligible_protein_ids)
    if cohort != "full":
        protein_ids = tuple(p for p in protein_ids if str(fold_bundle.outer_by_protein[p].source) == cohort)
    if len(protein_ids) != COHORT_SIZES[cohort]:
        raise ValueError(f"Cohort {cohort} has {len(protein_ids)} proteins, expected {COHORT_SIZES[cohort]}")
    if {int(fold_bundle.outer_by_protein[p].outer_fold) for p in protein_ids} != set(range(N_OUTER_FOLDS)):
        raise ValueError(f"Cohort {cohort} does not cover every outer fold")
    return protein_ids


def transfer_identity(source_layer: int, target_layer: int) -> dict[str, object]:
    return {
        "source_layer": int(source_layer), "target_layer": int(target_layer),
        "transfer_mode": TRANSFER_MODE, "layer_offset": int(target_layer) - int(source_layer),
        "is_same_layer": int(source_layer) == int(target_layer),
    }


def target_layer_for_task(task_id: int, layers: Sequence[int] = LAYERS) -> int:
    task = int(task_id)
    if task not in range(len(layers)):
        raise ValueError(f"Task id must be in 0..{len(layers) - 1}; got {task_id}")
    return int(layers[task])


@dataclass(frozen=True)
class ProbeBank:
    """Frozen probes indexed by (layer position, outer fold)."""

    intercept: np.ndarray            # (layers, folds)
    coefficient: np.ndarray          # (layers, folds, features)
    feature_mean: np.ndarray         # (layers, folds, features)
    feature_scale: np.ndarray        # (layers, folds, features)
    global_baseline: np.ndarray      # (folds,)

    def restandardized_weights(self, target_position: int, fold: int) -> tuple[np.ndarray, np.ndarray]:
        """Raw-feature weights and constants for every source at one target scaler.

        ``intercept + ((x - target_mean) / target_scale) @ coef`` rearranged so one
        matrix product serves all sources.
        """
        weights = self.coefficient[:, fold] / self.feature_scale[target_position, fold]
        constants = self.intercept[:, fold] - weights @ self.feature_mean[target_position, fold]
        return weights, constants


def load_probe_bank(
    producer_root: str | Path, fold_bundle, layers: Sequence[int] = LAYERS,
) -> tuple[ProbeBank, dict[str, object], list[dict[str, object]]]:
    """Load and checksum every frozen shared-OLS fit for the given layers."""
    root = Path(producer_root) / PRODUCER_SPEC.family_dir
    manifest = validate_family_manifest(root, PRODUCER_SPEC, fold_bundle.fold_digests)
    rows = load_and_validate_replay_index(root, PRODUCER_SPEC, manifest, verify_artifacts=False)
    index = index_replay_rows(rows, ("layer", "outer_fold"))
    status_by_key = _producer_model_status(root)
    shape = (len(layers), N_OUTER_FOLDS)
    intercept = np.empty(shape)
    coefficient, mean, scale = (np.empty((*shape, FEATURE_DIM)) for _ in range(3))
    baseline = np.full(shape, np.nan)
    status_rows: list[dict[str, object]] = []
    for position, layer in enumerate(layers):
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
            intercept[position, fold] = float(fit["intercept_standardized"])
            coefficient[position, fold] = np.asarray(fit["coefficient_standardized"], dtype=np.float64)
            mean[position, fold] = np.asarray(fit["feature_mean"], dtype=np.float64)
            scale[position, fold] = np.asarray(fit["feature_scale"], dtype=np.float64)
            baseline[position, fold] = float(fit["global_baseline_value"])
            status = status_by_key.get((layer, fold), {})
            status_rows.append({
                "source_layer": layer, "outer_fold": fold, "fit_status": str(fit["fit_status"]),
                "support_flags": status.get("support_flags", ""),
                "condition_number": float(status.get("condition_number", "nan") or "nan"),
                "artifact_path": row["artifact_path"], "artifact_sha256": digest,
                "scaler_digest": row["scaler_digest"],
                "global_baseline_value": float(fit["global_baseline_value"]),
            })
    if not np.all(baseline == baseline[0]):
        raise ValueError("Global baseline differs across layers within a fold")
    provenance = {
        "producer_root": str(root),
        "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
        "replay_index_sha256": sha256_file(root / "model_artifacts/replay_index.csv"),
        "configuration_digest": manifest["configuration_digest"],
        "artifacts": [
            {key: row[key] for key in ("source_layer", "outer_fold", "artifact_path",
                                       "artifact_sha256", "scaler_digest")}
            for row in status_rows
        ],
    }
    return ProbeBank(intercept, coefficient, mean, scale, baseline[0].copy()), provenance, status_rows


def predict_all_sources(features: np.ndarray, weights: np.ndarray, constants: np.ndarray) -> np.ndarray:
    """Predictions of shape (pairs, sources) for one protein at one target layer."""
    predictions = np.asarray(features, dtype=np.float64) @ weights.T + constants
    if not np.all(np.isfinite(predictions)):
        raise ValueError("Replay produced nonfinite predictions")
    return predictions


def sequence_fine_bins(pair_i: np.ndarray, pair_j: np.ndarray) -> np.ndarray:
    separation = np.abs(np.asarray(pair_i, dtype=np.int64) - np.asarray(pair_j, dtype=np.int64))
    if np.any(separation < 1):
        raise ValueError("Pairs must join distinct residues")
    return np.searchsorted(SEQUENCE_EDGES, separation, side="right") - 1


def distance_fine_bins(distances: np.ndarray) -> np.ndarray:
    values = np.asarray(distances, dtype=np.float64)
    if not np.all(np.isfinite(values)) or np.any(values < 0):
        raise ValueError("Final structure distances must be finite and nonnegative")
    return np.searchsorted(DISTANCE_EDGES, values, side="right") - 1


def position_bins(residues: np.ndarray, n_residues: int) -> np.ndarray:
    """Normalized chain position; N-terminus is bin 0."""
    return np.minimum((N_POSITION_BINS * np.asarray(residues, dtype=np.int64)) // int(n_residues),
                      N_POSITION_BINS - 1)


def _group_sum(values: np.ndarray, starts: np.ndarray) -> np.ndarray:
    return np.add.reduceat(values, starts, axis=0)


def summarize_protein(
    *, y: np.ndarray, predictions: np.ndarray, pair_i: np.ndarray, pair_j: np.ndarray,
    n_residues: int, baseline_value: float,
) -> dict[str, np.ndarray]:
    """Additive statistics for one protein at one target layer, for all sources."""
    y = np.asarray(y, dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    n_pairs, n_sources = predictions.shape
    if y.shape != (n_pairs,):
        raise ValueError("Targets and predictions are misaligned")
    errors = predictions - y[:, None]

    sequence, distance = sequence_fine_bins(pair_i, pair_j), distance_fine_bins(y)
    strata = sequence * N_DISTANCE_BINS + distance
    order = np.argsort(strata, kind="stable")
    ordered_strata = strata[order]
    starts = np.flatnonzero(np.r_[True, ordered_strata[1:] != ordered_strata[:-1]])
    present = ordered_strata[starts]
    ordered_errors = errors[order]
    ordered_y = y[order]

    shape = (N_SEQUENCE_BINS, N_DISTANCE_BINS)
    strata_n = np.zeros(shape, dtype=np.int64)
    strata_n.flat[present] = np.diff(np.r_[starts, n_pairs])
    strata_y = np.zeros((*shape, 2))
    strata_y[..., 0].flat[present] = _group_sum(ordered_y, starts)
    strata_y[..., 1].flat[present] = _group_sum(ordered_y * ordered_y, starts)
    strata_stats = np.zeros((n_sources, *shape, 3))
    grouped = (_group_sum(ordered_errors, starts), _group_sum(np.abs(ordered_errors), starts),
               _group_sum(ordered_errors * ordered_errors, starts))
    for k, values in enumerate(grouped):
        strata_stats[:, sequence_index(present), distance_index(present), k] = values.T

    probe = np.zeros((n_sources, len(CATEGORY_CELLS), len(STAT_FIELDS)))
    for cell, selector in _category_selectors().items():
        n = strata_n[selector].sum()
        sum_y, sum_y2 = strata_y[selector].reshape(-1, 2).sum(axis=0)
        summed = strata_stats[(slice(None), *selector)].reshape(n_sources, -1, 3).sum(axis=1)
        probe[:, CELL_INDEX[cell], 0] = n
        probe[:, CELL_INDEX[cell], 1] = sum_y
        probe[:, CELL_INDEX[cell], 2] = sum_y2
        probe[:, CELL_INDEX[cell], 3] = sum_y + summed[:, 0]
        probe[:, CELL_INDEX[cell], 4:7] = summed

    baseline_error = baseline_value - y
    baseline = np.array([
        n_pairs, y.sum(), (y * y).sum(), baseline_value * n_pairs, baseline_error.sum(),
        np.abs(baseline_error).sum(), (baseline_error * baseline_error).sum(),
    ])

    error_bins = np.searchsorted(ERROR_EDGES, errors.ravel(), side="right").reshape(errors.shape) - 1
    source_index = np.arange(n_sources, dtype=np.int64)
    n_strata = N_SEQUENCE_BINS * N_DISTANCE_BINS
    hist_joint = np.bincount(
        (source_index * (N_ERROR_BINS * n_strata) + error_bins * n_strata + strata[:, None]).ravel(),
        minlength=n_sources * N_ERROR_BINS * n_strata,
    ).reshape(n_sources, N_ERROR_BINS, *shape).astype(np.int32)
    hist_protein = np.bincount(
        (source_index * N_ERROR_BINS + error_bins).ravel(), minlength=n_sources * N_ERROR_BINS,
    ).reshape(n_sources, N_ERROR_BINS).astype(np.int32)

    position_i, position_j = position_bins(pair_i, n_residues), position_bins(pair_j, n_residues)
    pair_cell = position_i * N_POSITION_BINS + position_j
    order = np.argsort(pair_cell, kind="stable")
    ordered_cells = pair_cell[order]
    starts = np.flatnonzero(np.r_[True, ordered_cells[1:] != ordered_cells[:-1]])
    present = ordered_cells[starts]
    ordered_errors = errors[order]
    square = np.zeros((n_sources, N_POSITION_BINS, N_POSITION_BINS, 3))
    grouped = (_group_sum(ordered_errors, starts), _group_sum(np.abs(ordered_errors), starts),
               _group_sum(ordered_errors * ordered_errors, starts))
    for k, values in enumerate(grouped):
        square[:, present // N_POSITION_BINS, present % N_POSITION_BINS, k] = values.T
    # A pair counts once for each of its two residues.
    position_stats = square.sum(axis=1) + square.sum(axis=2)
    position_n = np.bincount(position_i, minlength=N_POSITION_BINS) + np.bincount(
        position_j, minlength=N_POSITION_BINS)

    return {
        "probe": probe, "baseline": baseline, "hist_joint": hist_joint, "hist_protein": hist_protein,
        "strata_stats": strata_stats, "strata_n": strata_n, "position_stats": position_stats,
        "position_n": position_n, "error_min": errors.min(axis=0), "error_max": errors.max(axis=0),
    }


def sequence_index(strata: np.ndarray) -> np.ndarray:
    return np.asarray(strata) // N_DISTANCE_BINS


def distance_index(strata: np.ndarray) -> np.ndarray:
    return np.asarray(strata) % N_DISTANCE_BINS


def _category_selectors() -> dict[tuple[str, str], tuple[object, object]]:
    selectors: dict[tuple[str, str], tuple[object, object]] = {
        ("all_pairs", "all"): (slice(None), slice(None)),
    }
    for label in SEQUENCE_BIN_LABELS:
        selectors[("sequence_separation", label)] = (list(CANONICAL_SEQUENCE_GROUPS[label]), slice(None))
    for label in SPATIAL_DISTANCE_BIN_LABELS:
        selectors[("spatial_distance", label)] = (slice(None), list(CANONICAL_DISTANCE_GROUPS[label]))
    return selectors


class TargetAccumulator:
    """Everything a worker saves for one target layer."""

    FOLD_ARRAYS = ("hist_joint", "strata_stats", "strata_n", "strata_proteins",
                   "position_stats", "position_n")

    def __init__(self, target_layer: int, protein_ids: Sequence[str], n_sources: int = len(LAYERS),
                 code_revision: str = ""):
        self.target_layer = int(target_layer)
        self.protein_ids = tuple(protein_ids)
        self.code_revision = str(code_revision)
        n_proteins = len(self.protein_ids)
        strata = (N_SEQUENCE_BINS, N_DISTANCE_BINS)
        self.arrays: dict[str, np.ndarray] = {
            "completed": np.zeros(n_proteins, dtype=bool),
            "probe": np.zeros((n_sources, n_proteins, len(CATEGORY_CELLS), len(STAT_FIELDS))),
            "baseline": np.zeros((n_proteins, len(STAT_FIELDS))),
            "hist_protein": np.zeros((n_sources, n_proteins, N_ERROR_BINS), dtype=np.int32),
            "error_min": np.full(n_sources, np.inf),
            "error_max": np.full(n_sources, -np.inf),
            "hist_joint": np.zeros((n_sources, N_OUTER_FOLDS, N_ERROR_BINS, *strata), dtype=np.int32),
            "strata_stats": np.zeros((n_sources, N_OUTER_FOLDS, *strata, 3)),
            "strata_n": np.zeros((N_OUTER_FOLDS, *strata), dtype=np.int64),
            "strata_proteins": np.zeros((N_OUTER_FOLDS, *strata), dtype=np.int32),
            "position_stats": np.zeros((n_sources, N_OUTER_FOLDS, N_POSITION_BINS, 3)),
            "position_n": np.zeros((N_OUTER_FOLDS, N_POSITION_BINS), dtype=np.int64),
            "outlier_hist_joint": np.zeros((n_sources, N_ERROR_BINS, *strata), dtype=np.int32),
            "outlier_strata_stats": np.zeros((n_sources, *strata, 3)),
            "outlier_strata_n": np.zeros(strata, dtype=np.int64),
            "outlier_position_stats": np.zeros((n_sources, N_POSITION_BINS, 3)),
            "outlier_position_n": np.zeros(N_POSITION_BINS, dtype=np.int64),
        }

    def add(self, protein_position: int, outer_fold: int, summary: Mapping[str, np.ndarray]) -> None:
        a = self.arrays
        if a["completed"][protein_position]:
            raise ValueError(f"Protein {self.protein_ids[protein_position]} added twice")
        a["probe"][:, protein_position] = summary["probe"]
        a["baseline"][protein_position] = summary["baseline"]
        a["hist_protein"][:, protein_position] = summary["hist_protein"]
        a["error_min"] = np.minimum(a["error_min"], summary["error_min"])
        a["error_max"] = np.maximum(a["error_max"], summary["error_max"])
        a["hist_joint"][:, outer_fold] += summary["hist_joint"]
        a["strata_stats"][:, outer_fold] += summary["strata_stats"]
        a["strata_n"][outer_fold] += summary["strata_n"]
        a["strata_proteins"][outer_fold] += (summary["strata_n"] > 0).astype(np.int32)
        a["position_stats"][:, outer_fold] += summary["position_stats"]
        a["position_n"][outer_fold] += summary["position_n"]
        if self.protein_ids[protein_position] == OUTLIER_PROTEIN:
            a["outlier_hist_joint"] += summary["hist_joint"]
            a["outlier_strata_stats"] += summary["strata_stats"]
            a["outlier_strata_n"] += summary["strata_n"]
            a["outlier_position_stats"] += summary["position_stats"]
            a["outlier_position_n"] += summary["position_n"]
        a["completed"][protein_position] = True

    def payload(self) -> dict[str, np.ndarray]:
        return {
            **self.arrays, "target_layer": np.int64(self.target_layer),
            "protein_ids": np.asarray(self.protein_ids), "shard_schema": np.int64(SHARD_SCHEMA),
            "code_revision": np.asarray(self.code_revision),
            "error_edges": ERROR_EDGES, "sequence_edges": SEQUENCE_EDGES,
            "distance_edges": DISTANCE_EDGES,
        }

    @classmethod
    def from_file(cls, path: str | Path, target_layer: int, protein_ids: Sequence[str],
                  n_sources: int = len(LAYERS), code_revision: str | None = None) -> "TargetAccumulator":
        """Load a shard; a given ``code_revision`` must match the one that wrote it."""
        accumulator = cls(target_layer, protein_ids, n_sources)
        with np.load(path) as shard:
            validate_shard_header(shard, target_layer, protein_ids)
            accumulator.code_revision = str(shard["code_revision"])
            if code_revision is not None and accumulator.code_revision != code_revision:
                raise ValueError(f"Shard was written by revision {accumulator.code_revision!r}, "
                                 f"not {code_revision!r}: {path}")
            for name, template in accumulator.arrays.items():
                loaded = shard[name]
                if loaded.shape != template.shape or loaded.dtype != template.dtype:
                    raise ValueError(f"Shard array {name} has the wrong shape or dtype: {path}")
                accumulator.arrays[name] = loaded.copy()
        return accumulator


def validate_shard_header(shard, target_layer: int, protein_ids: Sequence[str]) -> None:
    if int(shard["shard_schema"]) != SHARD_SCHEMA or int(shard["target_layer"]) != int(target_layer):
        raise ValueError("Shard has the wrong schema or target layer")
    if tuple(str(p) for p in shard["protein_ids"]) != tuple(protein_ids):
        raise ValueError("Shard protein order differs from the frozen cohort")
    for name, edges in (("error_edges", ERROR_EDGES), ("sequence_edges", SEQUENCE_EDGES),
                        ("distance_edges", DISTANCE_EDGES)):
        if not np.array_equal(shard[name], edges):
            raise ValueError(f"Shard {name} differ from this code")


def validate_final_shard(accumulator: TargetAccumulator) -> None:
    """Internal consistency a finished target layer must satisfy."""
    a = accumulator.arrays
    if not a["completed"].all():
        raise ValueError("Target layer is missing proteins")
    whole = CELL_INDEX[("all_pairs", "all")]
    pairs = a["probe"][0, :, whole, 0]
    if not np.all(a["probe"][..., whole, 0] == pairs):
        raise ValueError("Pair counts differ across sources")
    if not np.array_equal(a["baseline"][:, 0], pairs):
        raise ValueError("Baseline pair counts differ from probe pair counts")
    if not np.all(a["hist_protein"].sum(axis=2) == pairs.astype(np.int64)):
        raise ValueError("Per-protein histogram mass differs from pair counts")
    total = int(pairs.sum())
    if not np.all(a["hist_joint"].sum(axis=(1, 2, 3, 4), dtype=np.int64) == total):
        raise ValueError("Joint histogram mass differs from the cohort pair count")
    if int(a["strata_n"].sum()) != total or int(a["position_n"].sum()) != 2 * total:
        raise ValueError("Strata or position counts differ from the cohort pair count")


def strata_support_status(n_pairs: np.ndarray, n_proteins: np.ndarray) -> np.ndarray:
    """Same vocabulary as the canonical subgroup tables."""
    pairs, proteins = np.asarray(n_pairs), np.asarray(n_proteins)
    few_pairs, few_proteins = pairs < MIN_SUPPORT_PAIRS, proteins < MIN_SUPPORT_PROTEINS
    return np.select(
        [pairs == 0, few_pairs & few_proteins, few_pairs, few_proteins],
        ["empty", "low_pair_and_protein_support", "low_pair_support", "low_protein_support"],
        default="sufficient",
    )


def histogram_quantiles(counts: np.ndarray, quantiles: Sequence[float], low: float, high: float) -> np.ndarray:
    """Quantiles interpolated within bins; open tail bins clamp to the observed extremes."""
    masses = np.asarray(counts, dtype=np.float64)
    total = masses.sum()
    if total <= 0:
        return np.full(len(quantiles), np.nan)
    edges = ERROR_EDGES.astype(np.float64).copy()
    edges[0], edges[-1] = min(float(low), edges[1]), max(float(high), edges[-2])
    cumulative = np.concatenate([[0.0], np.cumsum(masses)]) / total
    return np.interp(quantiles, cumulative, edges)


__all__ = [name for name in globals() if name.isupper()] + [
    "ProbeBank", "TargetAccumulator", "analysis_label", "analysis_name", "cohort_name",
    "default_output_root", "distance_fine_bins", "distance_index", "histogram_quantiles",
    "load_probe_bank", "position_bins", "predict_all_sources", "select_cohort", "sequence_fine_bins",
    "sequence_index", "strata_support_status", "summarize_protein", "target_layer_for_task",
    "transfer_identity",
    "validate_final_shard", "validate_shard_header",
]

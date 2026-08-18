#!/usr/bin/env python3
"""Canonical output replay for the historical shared base256 OLS coefficients.

This module never fits a regression.  It validates the historical six-decimal
coefficient CSV, replays its raw-scale coefficients, and writes one canonical
output shard per pair-representation layer.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import platform
import resource
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable, Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
for _path in (SCRIPT_DIR, CLUSTER_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cluster_regression_helpers import (  # noqa: E402
    build_unordered_aa_pair_codes,
    compute_distance_matrix,
    extract_ca_coordinates,
)
from general_train_test_common import (  # noqa: E402
    build_dataset_protein_manifest,
    resolve_pair_block_path,
    resolve_pdb_file,
)
from regression_output_helpers import (  # noqa: E402
    COEFFICIENT_COLUMNS,
    GEOMETRY_METRICS_COLUMNS,
    GEOMETRY_SUMMARY_COLUMNS,
    MODEL_STATUS_COLUMNS,
    OUTPUT_SCHEMA_VERSION,
    PER_PROTEIN_COLUMNS,
    POOLED_METRICS_COLUMNS,
    RUNTIME_COLUMNS,
    SUBGROUP_METRICS_COLUMNS,
    PooledComparisonAccumulator,
    SubgroupAccumulator,
    atomic_write_csv,
    atomic_write_json,
    build_geometry_row,
    build_model_status_row,
    build_per_protein_row,
    build_pooled_metrics_row,
    build_runtime_row,
    semantic_digest,
    sha256_file,
    summarize_geometry_rows,
)
from regression_protocol import (  # noqa: E402
    AMINO_ACID_PAIR_LABELS,
    FEATURE_DIM,
    FEATURE_NAMES,
    LAYERS,
    MASTER_MANIFEST_DIGEST,
    PROTOCOL_VERSION,
    SEQUENCE_BIN_LABELS,
    SPATIAL_DISTANCE_BIN_LABELS,
    extract_directional_base256,
    load_master_manifest,
    protein_equal_mean,
    sequence_separation_bin_codes,
    spatial_distance_bin_codes,
    strict_lower_triangle_indices,
)


ANALYSIS_NAME = "general_base256_combined"
MODEL_FAMILY = "shared_ols"
SOURCE_FAMILY_NAME = "general_base256_combined"
FINAL_FAMILY_NAME = "general_base256_combined_canonical"
WORK_FAMILY_NAME = "general_base256_combined_canonical_work"
LEGACY_COEFFICIENT_FILENAME = "coefficients.csv"
LEGACY_SPLIT_FILENAME = "split_assignments.csv"
LEGACY_SUMMARY_FILENAME = "split_summary.csv"

LEGACY_COEFFICIENT_COLUMNS = (
    "variant", "feature_dim", "split_id", "layer", "feature_index",
    "feature_name", "coefficient_standardized", "coefficient_raw_scale",
)
LEGACY_SPLIT_COLUMNS = (
    "split_id", "split", "protein_id", "n_residues", "size_bin", "n_pairs",
    "target_variance",
)
LEGACY_SUMMARY_COLUMNS = (
    "variant", "feature_dim", "split_id", "layer", "n_train_proteins",
    "n_test_proteins", "n_train_rows", "n_test_rows",
    "train_mean_per_protein_r2", "test_mean_per_protein_r2",
    "train_pair_weighted_r2", "test_pair_weighted_r2",
    "train_pair_weighted_mae", "test_pair_weighted_mae",
    "train_pair_weighted_rmse", "test_pair_weighted_rmse",
    "train_mean_bias", "test_mean_bias", "elapsed_seconds",
)
SPLIT_ASSIGNMENT_COLUMNS = (
    "manifest_version", "master_manifest_digest", "protein_id", "source",
    "n_residues", "n_pairs", "size_bin", "master_role", "protein_split",
    "outer_split_seed", "development_split_seed", "split_assignment_digest",
)
EXCLUDED_COLUMNS = (
    "protein_id", "source", "eligibility_status", "stage", "layer", "reason",
)
LEGACY_COMPARISON_COLUMNS = (
    "analysis_name", "layer", "protein_split", "metric", "legacy_value",
    "replay_value", "absolute_difference", "tolerance", "comparison_status",
)

SHARD_CSV_SCHEMAS = {
    "per_protein.csv": PER_PROTEIN_COLUMNS,
    "pooled_metrics.csv": POOLED_METRICS_COLUMNS,
    "model_status.csv": MODEL_STATUS_COLUMNS,
    "runtime.csv": RUNTIME_COLUMNS,
    "geometry_metrics.csv": GEOMETRY_METRICS_COLUMNS,
    "geometry_summary.csv": GEOMETRY_SUMMARY_COLUMNS,
    "subgroup_metrics/by_sequence_separation.csv": SUBGROUP_METRICS_COLUMNS,
    "subgroup_metrics/by_amino_acid_pair.csv": SUBGROUP_METRICS_COLUMNS,
    "subgroup_metrics/by_spatial_distance.csv": SUBGROUP_METRICS_COLUMNS,
    "model_artifacts/coefficients.csv": COEFFICIENT_COLUMNS,
    "metadata/legacy_metric_comparison.csv": LEGACY_COMPARISON_COLUMNS,
}


@dataclass(frozen=True)
class LayerCoefficients:
    layer: int
    intercept_raw: float
    coefficient_raw: np.ndarray
    canonical_rows: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class CoefficientBundle:
    path: Path
    sha256: str
    by_layer: Mapping[int, LayerCoefficients]


@dataclass(frozen=True)
class HistoricalSplitBundle:
    path: Path
    sha256: str
    semantic_digest: str
    split_by_id: Mapping[str, str]


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def csv_root() -> Path:
    return Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))).expanduser()


def source_output_root() -> Path:
    return csv_root() / SOURCE_FAMILY_NAME


def final_output_root() -> Path:
    return csv_root() / FINAL_FAMILY_NAME


def working_output_root() -> Path:
    return csv_root() / WORK_FAMILY_NAME


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)


def version_payload() -> dict[str, str]:
    return {"python": platform.python_version(), "numpy": np.__version__}


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def code_digest() -> str:
    def shared_file(name: str) -> Path:
        local = SCRIPT_DIR / name
        return local if local.is_file() else CLUSTER_DIR / name

    paths = (
        Path(__file__),
        SCRIPT_DIR / "general_base256_coefficient_replay_layer.py",
        SCRIPT_DIR / "aggregate_general_base256_coefficient_replay.py",
        shared_file("regression_output_helpers.py"),
        shared_file("regression_protocol.py"),
    )
    return semantic_digest([
        (path.name, sha256_file(path)) for path in paths if path.is_file()
    ])


def read_csv_strict(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"Schema mismatch in {path}: {reader.fieldnames}")
        return list(reader)


def _expected_legacy_feature_name(feature_index: int) -> str:
    if feature_index < 128:
        return f"pair_ij_ch_{feature_index}"
    return f"pair_ji_ch_{feature_index - 128}"


def load_coefficient_bundle(
    path: str | Path | None = None,
    *,
    expected_layers: Sequence[int] = LAYERS,
) -> CoefficientBundle:
    coefficient_path = Path(path) if path is not None else source_output_root() / LEGACY_COEFFICIENT_FILENAME
    rows = read_csv_strict(coefficient_path, LEGACY_COEFFICIENT_COLUMNS)
    expected = tuple(sorted({int(layer) for layer in expected_layers}))
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in rows:
        if row["variant"] != ANALYSIS_NAME or int(row["feature_dim"]) != FEATURE_DIM:
            raise ValueError("Legacy coefficient identity does not match general_base256_combined/base256")
        if int(row["split_id"]) != 0:
            raise ValueError("Legacy coefficients must use split_id=0")
        layer = int(row["layer"])
        if layer not in expected:
            raise ValueError(f"Unexpected coefficient layer {layer}")
        grouped.setdefault(layer, []).append(row)
    if tuple(sorted(grouped)) != expected:
        raise ValueError(f"Coefficient layers do not match {expected}: {sorted(grouped)}")

    by_layer: dict[int, LayerCoefficients] = {}
    for layer in expected:
        layer_rows = grouped[layer]
        if len(layer_rows) != FEATURE_DIM + 1:
            raise ValueError(f"Layer {layer} has {len(layer_rows)} coefficient rows, expected 257")
        by_index: dict[int, dict[str, str]] = {}
        for row in layer_rows:
            index = int(row["feature_index"])
            if index in by_index:
                raise ValueError(f"Duplicate coefficient index {index} in layer {layer}")
            by_index[index] = row
        if set(by_index) != {-1, *range(FEATURE_DIM)}:
            raise ValueError(f"Layer {layer} coefficient indices are incomplete")
        if by_index[-1]["feature_name"] != "intercept":
            raise ValueError(f"Layer {layer} intercept row is malformed")
        raw = np.empty(FEATURE_DIM, dtype=np.float64)
        canonical_rows: list[dict[str, object]] = []
        for index in (-1, *range(FEATURE_DIM)):
            row = by_index[index]
            expected_name = "intercept" if index == -1 else _expected_legacy_feature_name(index)
            if row["feature_name"] != expected_name:
                raise ValueError(
                    f"Layer {layer} feature {index} is {row['feature_name']!r}, expected {expected_name!r}"
                )
            standardized = float(row["coefficient_standardized"])
            raw_value = float(row["coefficient_raw_scale"])
            if not math.isfinite(standardized) or not math.isfinite(raw_value):
                raise ValueError(f"Layer {layer} feature {index} has a nonfinite coefficient")
            if index >= 0:
                raw[index] = raw_value
            canonical_rows.append({
                "analysis_name": ANALYSIS_NAME,
                "model_family": MODEL_FAMILY,
                "layer": layer,
                "protein_id": "",
                "gate_aa_pair": "",
                "gate_sequence_bin": "",
                "feature_index": index,
                "feature_name": "intercept" if index == -1 else FEATURE_NAMES[index],
                "coefficient_standardized": standardized,
                "coefficient_raw_scale": raw_value,
            })
        raw.setflags(write=False)
        by_layer[layer] = LayerCoefficients(
            layer=layer,
            intercept_raw=float(by_index[-1]["coefficient_raw_scale"]),
            coefficient_raw=raw,
            canonical_rows=tuple(canonical_rows),
        )
    return CoefficientBundle(
        path=coefficient_path.resolve(),
        sha256=sha256_file(coefficient_path),
        by_layer=by_layer,
    )


def load_historical_split_bundle(
    path: str | Path | None = None,
    *,
    entries: Sequence[object] | None = None,
) -> HistoricalSplitBundle:
    split_path = Path(path) if path is not None else source_output_root() / LEGACY_SPLIT_FILENAME
    rows = read_csv_strict(split_path, LEGACY_SPLIT_COLUMNS)
    manifest_entries = tuple(load_master_manifest() if entries is None else entries)
    eligible = {entry.protein_id: entry for entry in manifest_entries if entry.eligible}
    if len(rows) != len(eligible):
        raise ValueError(f"Historical split has {len(rows)} rows, expected {len(eligible)}")
    split_by_id: dict[str, str] = {}
    normalized = []
    for row in rows:
        protein_id = row["protein_id"]
        if protein_id in split_by_id or protein_id not in eligible:
            raise ValueError(f"Duplicate or unknown historical split protein: {protein_id}")
        source_split = row["split"]
        if source_split not in {"train", "test"}:
            raise ValueError(f"Invalid historical split label for {protein_id}: {source_split}")
        protein_split = "development" if source_split == "train" else "test"
        entry = eligible[protein_id]
        if protein_split != entry.protein_split:
            raise ValueError(f"Historical/canonical outer split mismatch for {protein_id}")
        if (
            int(row["split_id"]) != 0
            or int(row["n_residues"]) != entry.n_residues
            or int(row["n_pairs"]) != entry.n_pairs
            or int(row["size_bin"]) != entry.size_bin
            or not math.isfinite(float(row["target_variance"]))
        ):
            raise ValueError(f"Historical split metadata mismatch for {protein_id}")
        split_by_id[protein_id] = protein_split
        normalized.append({
            "protein_id": protein_id,
            "protein_split": protein_split,
            "n_residues": int(row["n_residues"]),
            "n_pairs": int(row["n_pairs"]),
            "size_bin": int(row["size_bin"]),
        })
    if set(split_by_id) != set(eligible):
        raise ValueError("Historical split membership is incomplete")
    counts = {split: sum(value == split for value in split_by_id.values()) for split in ("development", "test")}
    canonical_counts = {
        split: sum(entry.protein_split == split for entry in eligible.values())
        for split in ("development", "test")
    }
    if counts != canonical_counts:
        raise ValueError(f"Historical split counts differ from the canonical outer split: {counts}")
    return HistoricalSplitBundle(
        path=split_path.resolve(),
        sha256=sha256_file(split_path),
        semantic_digest=semantic_digest(sorted(normalized, key=lambda row: row["protein_id"])),
        split_by_id=split_by_id,
    )


def load_legacy_summary(
    path: str | Path | None = None,
    *,
    expected_layers: Sequence[int] = LAYERS,
) -> dict[int, dict[str, str]]:
    summary_path = Path(path) if path is not None else source_output_root() / LEGACY_SUMMARY_FILENAME
    rows = read_csv_strict(summary_path, LEGACY_SUMMARY_COLUMNS)
    expected = set(map(int, expected_layers))
    by_layer: dict[int, dict[str, str]] = {}
    for row in rows:
        layer = int(row["layer"])
        if layer in by_layer or layer not in expected:
            raise ValueError(f"Duplicate or unexpected legacy summary layer {layer}")
        if row["variant"] != ANALYSIS_NAME or int(row["feature_dim"]) != FEATURE_DIM or int(row["split_id"]) != 0:
            raise ValueError(f"Legacy summary identity mismatch for layer {layer}")
        by_layer[layer] = row
    if set(by_layer) != expected:
        raise ValueError("Legacy split summary does not cover every requested layer")
    return by_layer


def split_assignment_rows(
    entries: Sequence[object] | None = None,
) -> tuple[list[dict[str, object]], str]:
    manifest_entries = tuple(load_master_manifest() if entries is None else entries)
    eligible = sorted((entry for entry in manifest_entries if entry.eligible), key=lambda entry: entry.protein_id)
    base = [{
        "manifest_version": entry.manifest_version,
        "master_manifest_digest": MASTER_MANIFEST_DIGEST,
        "protein_id": entry.protein_id,
        "source": entry.source,
        "n_residues": entry.n_residues,
        "n_pairs": entry.n_pairs,
        "size_bin": entry.size_bin,
        "master_role": entry.master_role,
        "protein_split": entry.protein_split,
        "outer_split_seed": entry.outer_split_seed,
        "development_split_seed": entry.development_split_seed,
    } for entry in eligible]
    digest = semantic_digest(base)
    return [{**row, "split_assignment_digest": digest} for row in base], digest


def validate_layer_task_id(task_id: int) -> int:
    layer = int(task_id)
    if layer not in LAYERS:
        raise ValueError(f"Coefficient replay layer task must be 0..47; got {task_id}")
    return layer


def load_layer_records(
    layer: int,
    *,
    entries: Sequence[object] | None = None,
) -> tuple[list[dict[str, object]], str]:
    """Load the complete frozen cohort and fail rather than changing membership."""
    manifest_entries = tuple(load_master_manifest() if entries is None else entries)
    eligible = sorted((entry for entry in manifest_entries if entry.eligible), key=lambda entry: entry.protein_id)
    dataset = {str(row["protein_id"]): row for row in build_dataset_protein_manifest()}
    records: list[dict[str, object]] = []
    identities = []
    for entry in eligible:
        source = dataset.get(entry.protein_id)
        if source is None or str(source["source"]) != entry.source:
            raise FileNotFoundError(f"Frozen protein is missing from dataset discovery: {entry.protein_id}")
        protein_dir = Path(source["protein_dir"])
        pair_path = resolve_pair_block_path(protein_dir, entry.protein_id, layer)
        pdb_path = resolve_pdb_file(entry.protein_id, protein_dir, Path(source["pdb_root"]))
        if not pair_path.is_file() or pdb_path is None or not Path(pdb_path).is_file():
            raise FileNotFoundError(
                f"Frozen layer input is missing for {entry.protein_id}: pair={pair_path}, pdb={pdb_path}"
            )
        ca = extract_ca_coordinates(pdb_path)
        coordinates = np.asarray(ca["coordinates"], dtype=np.float64)
        if coordinates.shape != (entry.n_residues, 3):
            raise ValueError(f"Canonical residue count mismatch for {entry.protein_id}")
        pair_shape = np.load(pair_path, mmap_mode="r").shape
        if pair_shape != (entry.n_residues, entry.n_residues, 128):
            raise ValueError(f"Pair-block shape mismatch for {entry.protein_id}: {pair_shape}")
        pair_i, pair_j = strict_lower_triangle_indices(entry.n_residues)
        y_true = np.asarray(compute_distance_matrix(coordinates)[pair_i, pair_j], dtype=np.float64)
        if y_true.size != entry.n_pairs or not np.all(np.isfinite(y_true)):
            raise ValueError(f"Target construction failed for {entry.protein_id}")
        records.append({
            "protein_id": entry.protein_id,
            "source": entry.source,
            "master_role": entry.master_role,
            "protein_split": entry.protein_split,
            "n_residues": entry.n_residues,
            "n_pairs": entry.n_pairs,
            "size_bin": entry.size_bin,
            "protein_dir": protein_dir,
            "pair_path": pair_path,
            "pdb_path": Path(pdb_path),
            "pair_i": pair_i,
            "pair_j": pair_j,
            "coordinates": coordinates,
            "residues": tuple(ca["residues"]),
            "y_true": y_true,
        })
        pair_stat = pair_path.stat()
        pdb_stat = Path(pdb_path).stat()
        identities.append((
            entry.protein_id, str(pair_path), pair_stat.st_size, pair_stat.st_mtime_ns,
            str(pdb_path), pdb_stat.st_size, pdb_stat.st_mtime_ns,
        ))
    if len(records) != len(eligible):
        raise RuntimeError("Frozen layer cohort was not loaded completely")
    return records, semantic_digest(identities)


def load_record_features(record: Mapping[str, object], layer: int) -> np.ndarray:
    pair_path = Path(record.get("pair_path") or resolve_pair_block_path(
        Path(record["protein_dir"]), str(record["protein_id"]), layer,
    ))
    pair_block = np.load(pair_path)
    try:
        return extract_directional_base256(
            pair_block,
            np.asarray(record["pair_i"], dtype=np.int64),
            np.asarray(record["pair_j"], dtype=np.int64),
        )
    finally:
        del pair_block


def predict_raw(features: np.ndarray, coefficients: LayerCoefficients) -> np.ndarray:
    matrix = np.asarray(features, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != FEATURE_DIM or not np.all(np.isfinite(matrix)):
        raise ValueError("Replay features must be a finite N x 256 float64 matrix")
    predictions = matrix @ coefficients.coefficient_raw + coefficients.intercept_raw
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.ndim != 1 or predictions.shape[0] != matrix.shape[0] or not np.all(np.isfinite(predictions)):
        raise ValueError("Raw coefficient replay returned incomplete or nonfinite predictions")
    return predictions


def evaluate_records(
    records: Sequence[Mapping[str, object]],
    layer: int,
    coefficients: LayerCoefficients,
    global_development_mean: float,
    *,
    feature_loader: Callable[[Mapping[str, object], int], np.ndarray] = load_record_features,
) -> dict[str, list[dict[str, object]]]:
    per_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    pooled = {split: PooledComparisonAccumulator() for split in ("development", "test")}
    sequence = SubgroupAccumulator("sequence_separation", SEQUENCE_BIN_LABELS)
    amino_acid = SubgroupAccumulator("amino_acid_pair", AMINO_ACID_PAIR_LABELS)
    spatial = SubgroupAccumulator("spatial_distance", SPATIAL_DISTANCE_BIN_LABELS)
    for index, record in enumerate(records, start=1):
        protein_id = str(record["protein_id"])
        y_true = np.asarray(record["y_true"], dtype=np.float64)
        features = feature_loader(record, layer)
        predictions = predict_raw(features, coefficients)
        del features
        if predictions.shape != y_true.shape:
            raise ValueError(f"Prediction/target shape mismatch for {protein_id}")
        split = str(record["protein_split"])
        per_rows.append(build_per_protein_row(
            analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            layer=layer,
            protein_id=protein_id,
            source=str(record["source"]),
            protein_split=split,
            evaluation_scope="all_pairs",
            n_residues=int(record["n_residues"]),
            y_true=y_true,
            predictions=predictions,
        ))
        pooled[split].update_protein(
            protein_id, y_true, predictions, global_development_mean,
        )
        pair_i = np.asarray(record["pair_i"], dtype=np.int64)
        pair_j = np.asarray(record["pair_j"], dtype=np.int64)
        if split == "test":
            sequence.update_protein(
                protein_id, y_true, predictions,
                sequence_separation_bin_codes(pair_i, pair_j),
            )
            amino_acid.update_protein(
                protein_id, y_true, predictions,
                build_unordered_aa_pair_codes(record["residues"], pair_i, pair_j),
            )
            spatial.update_protein(
                protein_id, y_true, predictions, spatial_distance_bin_codes(y_true),
            )
        geometry_rows.append(build_geometry_row(
            analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            layer=layer,
            protein_id=protein_id,
            source=str(record["source"]),
            protein_split=split,
            n_residues=int(record["n_residues"]),
            true_coordinates=np.asarray(record["coordinates"], dtype=np.float64),
            pair_i=pair_i,
            pair_j=pair_j,
            predictions=predictions,
        ))
        if index % 25 == 0 or index == len(records):
            log(f"Layer {layer:02d}: replayed and evaluated {index}/{len(records)} proteins")
    pooled_rows = [build_pooled_metrics_row(
        pooled[split],
        analysis_name=ANALYSIS_NAME,
        model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM,
        layer=layer,
        protein_split=split,
        evaluation_scope="all_pairs",
        conditional_baseline_type="not_applicable",
    ) for split in ("development", "test")]
    subgroup_identity = {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "feature_dim": FEATURE_DIM,
        "layer": layer,
        "protein_split": "test",
        "evaluation_scope": "all_pairs",
    }
    return {
        "per_protein.csv": per_rows,
        "pooled_metrics.csv": pooled_rows,
        "geometry_metrics.csv": geometry_rows,
        "geometry_summary.csv": summarize_geometry_rows(geometry_rows),
        "subgroup_metrics/by_sequence_separation.csv": sequence.rows(**subgroup_identity),
        "subgroup_metrics/by_amino_acid_pair.csv": amino_acid.rows(**subgroup_identity),
        "subgroup_metrics/by_spatial_distance.csv": spatial.rows(**subgroup_identity),
    }


def legacy_comparison_rows(
    layer: int,
    per_rows: Sequence[Mapping[str, object]],
    pooled_rows: Sequence[Mapping[str, object]],
    legacy_row: Mapping[str, str],
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    pooled_by_split = {str(row["protein_split"]): row for row in pooled_rows}
    for split, legacy_prefix in (("development", "train"), ("test", "test")):
        protein_rows = [row for row in per_rows if row["protein_split"] == split]
        finite_r2 = [float(row["r2"]) for row in protein_rows if np.isfinite(float(row["r2"]))]
        replay_values = {
            "mean_per_protein_r2": float(np.mean(finite_r2)) if finite_r2 else float("nan"),
            "pooled_r2": float(pooled_by_split[split]["pooled_r2"]),
            "pooled_mae": float(pooled_by_split[split]["pooled_mae"]),
            "pooled_rmse": float(pooled_by_split[split]["pooled_rmse"]),
        }
        legacy_columns = {
            "mean_per_protein_r2": f"{legacy_prefix}_mean_per_protein_r2",
            "pooled_r2": f"{legacy_prefix}_pair_weighted_r2",
            "pooled_mae": f"{legacy_prefix}_pair_weighted_mae",
            "pooled_rmse": f"{legacy_prefix}_pair_weighted_rmse",
        }
        for metric, replay_value in replay_values.items():
            legacy_value = float(legacy_row[legacy_columns[metric]])
            tolerance = 1e-4 if metric.endswith("r2") else 1e-3
            difference = abs(replay_value - legacy_value)
            rows.append({
                "analysis_name": ANALYSIS_NAME,
                "layer": layer,
                "protein_split": split,
                "metric": metric,
                "legacy_value": legacy_value,
                "replay_value": replay_value,
                "absolute_difference": difference,
                "tolerance": tolerance,
                "comparison_status": "pass" if difference <= tolerance else "fail",
            })
    return rows


def expected_layer_counts(n_proteins: int) -> dict[str, int]:
    return {
        "per_protein.csv": n_proteins,
        "pooled_metrics.csv": 2,
        "model_status.csv": 1,
        "runtime.csv": 1,
        "geometry_metrics.csv": n_proteins,
        "geometry_summary.csv": 28,
        "subgroup_metrics/by_sequence_separation.csv": 6,
        "subgroup_metrics/by_amino_acid_pair.csv": 210,
        "subgroup_metrics/by_spatial_distance.csv": 5,
        "model_artifacts/coefficients.csv": FEATURE_DIM + 1,
        "metadata/legacy_metric_comparison.csv": 8,
    }


def write_layer_shard(
    root: Path,
    layer: int,
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    *,
    invariants: Mapping[str, object],
    replay: Mapping[str, object],
) -> Path:
    shard = root / "layer_shards" / f"layer_{layer:02d}"
    for filename, columns in SHARD_CSV_SCHEMAS.items():
        atomic_write_csv(shard / filename, columns, list(tables[filename]))
    files = [shard / filename for filename in sorted(SHARD_CSV_SCHEMAS)]
    atomic_write_json(shard / "shard_manifest.json", {
        **dict(invariants),
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "layer": layer,
        "status": "complete",
        "files": {
            str(path.relative_to(shard)): sha256_file(path) for path in files
        },
        "replay": dict(replay),
        "completed_at": utc_now(),
    })
    return shard


def write_failure(
    root: Path,
    layer: int,
    phase: str,
    exc: Exception,
    *,
    coefficient_sha256: str = "",
    historical_split_sha256: str = "",
) -> None:
    atomic_write_json(root / "failures" / f"layer_{layer:02d}.json", {
        "analysis_name": ANALYSIS_NAME,
        "layer": layer,
        "phase": phase,
        "exception_type": type(exc).__name__,
        "message": str(exc),
        "coefficient_source_sha256": coefficient_sha256,
        "historical_split_source_sha256": historical_split_sha256,
        "master_manifest_digest": MASTER_MANIFEST_DIGEST,
        "timestamp": utc_now(),
    })


def run_layer(task_id: int | None = None) -> Path:
    started = time.perf_counter()
    layer = validate_layer_task_id(
        int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if task_id is None else task_id
    )
    root = final_output_root()
    phase = "coefficient_validation"
    coefficient_bundle: CoefficientBundle | None = None
    split_bundle: HistoricalSplitBundle | None = None
    try:
        coefficient_bundle = load_coefficient_bundle()
        phase = "historical_split_validation"
        entries = load_master_manifest()
        split_bundle = load_historical_split_bundle(entries=entries)
        phase = "legacy_summary_validation"
        legacy_summary = load_legacy_summary()
        phase = "layer_input_loading"
        records, layer_input_digest = load_layer_records(layer, entries=entries)
        development_records = [record for record in records if record["protein_split"] == "development"]
        global_mean = protein_equal_mean(
            np.asarray(record["y_true"], dtype=np.float64) for record in development_records
        )
        phase = "coefficient_replay_evaluation"
        model = coefficient_bundle.by_layer[layer]
        tables = evaluate_records(records, layer, model, global_mean)
        failed_geometry = [
            row for row in tables["geometry_metrics.csv"]
            if row["geometry_status"] != "complete"
        ]
        if failed_geometry:
            raise RuntimeError(
                f"Layer {layer} has {len(failed_geometry)} incomplete geometry rows"
            )
        comparisons = legacy_comparison_rows(
            layer, tables["per_protein.csv"], tables["pooled_metrics.csv"], legacy_summary[layer],
        )
        if any(row["comparison_status"] != "pass" for row in comparisons):
            failures = [row for row in comparisons if row["comparison_status"] != "pass"]
            raise ValueError(f"Legacy replay comparison exceeded tolerance: {failures}")
        training_pairs = sum(int(record["n_pairs"]) for record in development_records)
        status_row = build_model_status_row(
            analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            layer=layer,
            fit_status="historical_fit_replayed",
            support_flags=(
                "historical_coefficients_six_decimal;"
                "design_diagnostics_unavailable;historical_p2_p5_p8_deviations"
            ),
            n_training_proteins=len(development_records),
            n_training_pairs=training_pairs,
        )
        runtime_row = build_runtime_row(
            analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY,
            layer=layer,
            completion_status="complete",
            n_fitted_proteins=len(development_records),
            n_evaluated_proteins=len(records),
            n_fitted_pairs=training_pairs,
            n_evaluated_pairs=sum(int(record["n_pairs"]) for record in records),
            elapsed_seconds=time.perf_counter() - started,
            peak_rss_mb=peak_rss_mb(),
        )
        tables.update({
            "model_status.csv": [status_row],
            "runtime.csv": [runtime_row],
            "model_artifacts/coefficients.csv": list(model.canonical_rows),
            "metadata/legacy_metric_comparison.csv": comparisons,
        })
        replay_record = development_records[0]
        replay_features = load_record_features(replay_record, layer)[:32]
        replay_predictions = predict_raw(replay_features, model)
        replay = {
            "protein_id": str(replay_record["protein_id"]),
            "pair_indices": list(range(int(replay_features.shape[0]))),
            "feature_sha256": hashlib.sha256(replay_features.tobytes()).hexdigest(),
            "predictions": replay_predictions.tolist(),
            "absolute_tolerance": 1e-10,
        }
        phase = "layer_shard_write"
        shard = write_layer_shard(
            root,
            layer,
            tables,
            invariants={
                "output_schema_version": OUTPUT_SCHEMA_VERSION,
                "protocol_version": PROTOCOL_VERSION,
                "master_manifest_digest": MASTER_MANIFEST_DIGEST,
                "historical_split_digest": split_bundle.semantic_digest,
                "coefficient_source_sha256": coefficient_bundle.sha256,
                "historical_split_source_sha256": split_bundle.sha256,
                "layer_input_digest": layer_input_digest,
                "code_digest": code_digest(),
            },
            replay=replay,
        )
        log(f"Layer {layer:02d}: canonical coefficient replay shard complete at {shard}")
        return shard
    except Exception as exc:
        write_failure(
            root,
            layer,
            phase,
            exc,
            coefficient_sha256="" if coefficient_bundle is None else coefficient_bundle.sha256,
            historical_split_sha256="" if split_bundle is None else split_bundle.sha256,
        )
        raise


__all__ = [name for name in globals() if not name.startswith("_")]

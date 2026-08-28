#!/usr/bin/env python3
"""Canonical combined-cohort per-protein base256 OLS regression runner.

One model is fitted for every eligible protein and Evoformer layer.  The model
uses the canonical deterministic 80/20 within-protein pair split, reports
training, held-out, and mixed all-pair metrics separately, and writes canonical
layer shards that can be consolidated without reading pair-level predictions.
"""

from __future__ import annotations

import gc
import math
import os
import platform
import resource
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
GENERALIZED_ECOD_DIR = CLUSTER_SCRIPTS_DIR / "generalized_model_ecod"
for _path in (CLUSTER_SCRIPTS_DIR, GENERALIZED_ECOD_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cluster_regression_helpers import (  # noqa: E402
    build_unordered_aa_pair_codes,
    compute_distance_matrix,
    extract_ca_coordinates,
)
from general_train_test_common import (  # noqa: E402
    CAMEO_PAIR_BLOCKS_ROOT,
    CAMEO_PDB_ROOT,
    ECOD_PAIR_BLOCKS_ROOT,
    ECOD_PDB_ROOT,
    build_dataset_protein_manifest,
    format_layer_indices,
    make_variant,
    normalize_layer_indices,
    resolve_pair_block_path,
    resolve_pdb_file,
)
from regression_output_helpers import (  # noqa: E402
    COEFFICIENT_INDEX_COLUMNS,
    GEOMETRY_METRICS_COLUMNS,
    GEOMETRY_SUMMARY_COLUMNS,
    MODEL_STATUS_COLUMNS,
    PER_PROTEIN_COLUMNS,
    POOLED_METRICS_COLUMNS,
    RUNTIME_COLUMNS,
    SUBGROUP_METRICS_COLUMNS,
    PooledComparisonAccumulator,
    RegressionStats,
    SubgroupAccumulator,
    atomic_savez_compressed,
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
    FLOAT_EPS,
    HELDOUT_PAIR_FRACTION,
    MASTER_MANIFEST_DIGEST,
    HISTORICAL_PROTOCOL_VERSION as PROTOCOL_VERSION,
    SEQUENCE_BIN_LABELS,
    SPATIAL_DISTANCE_BIN_LABELS,
    WITHIN_PROTEIN_SPLIT_SEED,
    eligible_manifest,
    extract_directional_base256,
    load_master_manifest,
    protein_equal_mean,
    sequence_separation_bin_codes,
    spatial_distance_bin_codes,
    strict_lower_triangle_indices,
    within_protein_pair_split,
    within_protein_pair_split_digest,
    within_protein_split_seed,
)


ANALYSIS_NAME = "per_protein_base256_combined"
MODEL_FAMILY = "per_protein_ols"
CONDITION_LIMIT = 1e12
MIN_TRAIN_ROWS = FEATURE_DIM + 1
EVALUATION_SCOPES = ("train_pairs", "heldout_pairs", "all_pairs")

PAIR_SPLIT_COLUMNS = (
    "analysis_name", "protein_id", "source", "master_role", "protein_split",
    "n_protein_pairs", "n_train_pairs", "n_heldout_pairs", "split_seed",
    "split_digest",
)
PREPARATION_FAILURE_COLUMNS = ("protein_id", "source", "stage", "reason")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def log_progress(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f} MiB] {message}", flush=True)


def version_payload() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def configuration_payload(layer_indices: Sequence[int]) -> dict[str, object]:
    normalize_layer_indices(layer_indices)
    return {
        "requested_layers": list(range(48)),
        "analysis_cohort": "frozen_combined_cameo_ecod_tm_gt_0.7",
        "master_split_digest": MASTER_MANIFEST_DIGEST,
        "feature_definition": "strict_lower_triangle_[Z_ij|Z_ji]",
        "feature_dim": FEATURE_DIM,
        "feature_precision": "float64",
        "target_precision": "float64",
        "within_protein_training_fraction": 1.0 - HELDOUT_PAIR_FRACTION,
        "within_protein_heldout_fraction": HELDOUT_PAIR_FRACTION,
        "within_protein_split_seed": WITHIN_PROTEIN_SPLIT_SEED,
        "scaler_population": "training_pairs_only",
        "ols_intercept": True,
        "condition_number_flag_threshold": CONDITION_LIMIT,
        "unstable_fit_aggregation_policy": "include_with_flags",
        "global_baseline": "protein_equal_development_training_pair_mean",
        "conditional_baseline": "per_protein_training_pair_mean",
        "subgroup_population": "heldout_pairs_from_locked_test_proteins",
        "geometry_population": "all_pairs",
        "coefficient_storage": "indexed_compressed_npz_layer_shards",
        "protocol_version": PROTOCOL_VERSION,
        "input_roots": {
            "cameo_pair_blocks": str(CAMEO_PAIR_BLOCKS_ROOT),
            "cameo_pdb": str(CAMEO_PDB_ROOT),
            "ecod_pair_blocks": str(ECOD_PAIR_BLOCKS_ROOT),
            "ecod_pdb": str(ECOD_PDB_ROOT),
        },
    }


def output_root(variant_name: str = ANALYSIS_NAME) -> Path:
    csv_root = Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))).expanduser()
    return csv_root.resolve() / variant_name


def _source_entries() -> dict[str, Mapping[str, object]]:
    entries = build_dataset_protein_manifest()
    by_id = {str(entry["protein_id"]): entry for entry in entries}
    if len(by_id) != len(entries):
        raise ValueError("Dataset discovery returned duplicate protein IDs")
    return by_id


def _prepare_record(manifest_entry, source_entry: Mapping[str, object]) -> dict[str, object]:
    protein_id = manifest_entry.protein_id
    if str(source_entry["source"]) != manifest_entry.source:
        raise ValueError("discovered source differs from frozen manifest")
    protein_dir = Path(source_entry["protein_dir"])
    pdb_path = resolve_pdb_file(protein_id, protein_dir, Path(source_entry["pdb_root"]))
    if pdb_path is None:
        raise FileNotFoundError("predicted PDB not found")
    ca_data = extract_ca_coordinates(pdb_path)
    residues = tuple(ca_data["residues"])
    coordinates = np.asarray(ca_data["coordinates"], dtype=np.float64)
    if len(residues) != manifest_entry.n_residues:
        raise ValueError(
            f"PDB CA count {len(residues)} != frozen count {manifest_entry.n_residues}"
        )
    pair_i, pair_j = strict_lower_triangle_indices(manifest_entry.n_residues)
    if pair_i.size != manifest_entry.n_pairs:
        raise ValueError("strict-lower-triangle pair count differs from frozen manifest")
    reference_path = resolve_pair_block_path(protein_dir, protein_id, 0)
    if not reference_path.exists():
        raise FileNotFoundError(f"reference pair block not found at {reference_path}")
    pair_block = np.load(str(reference_path))
    expected_shape = (manifest_entry.n_residues, manifest_entry.n_residues, 128)
    try:
        if pair_block.shape != expected_shape:
            raise ValueError(f"reference pair-block shape {pair_block.shape} != {expected_shape}")
        extract_directional_base256(pair_block, pair_i, pair_j)
    finally:
        del pair_block
    y_true = np.asarray(
        compute_distance_matrix(coordinates)[pair_i, pair_j], dtype=np.float64,
    )
    if not np.all(np.isfinite(y_true)):
        raise ValueError("target distances contain nonfinite values")
    train_idx, heldout_idx = within_protein_pair_split(
        manifest_entry.n_pairs, protein_id,
    )
    chain_ids = np.asarray([str(residue[0]) for residue in residues], dtype="U16")
    return {
        "protein_id": protein_id,
        "source": manifest_entry.source,
        "master_role": manifest_entry.master_role,
        "protein_split": manifest_entry.protein_split,
        "protein_dir": protein_dir,
        "n_residues": manifest_entry.n_residues,
        "n_pairs": manifest_entry.n_pairs,
        "size_bin": manifest_entry.size_bin,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "coordinates": coordinates,
        "y_true": y_true,
        "train_idx": train_idx,
        "heldout_idx": heldout_idx,
        "sequence_codes": sequence_separation_bin_codes(
            pair_i, pair_j, chain_ids=chain_ids,
        ),
        "aa_codes": build_unordered_aa_pair_codes(residues, pair_i, pair_j),
        "spatial_codes": spatial_distance_bin_codes(y_true),
        "split_seed": within_protein_split_seed(protein_id),
        "split_digest": within_protein_pair_split_digest(
            manifest_entry.n_pairs, protein_id,
        ),
    }


def prepare_records() -> tuple[list[dict[str, object]], list[dict[str, str]]]:
    manifest_entries = eligible_manifest(load_master_manifest())
    discovered = _source_entries()
    records: list[dict[str, object]] = []
    failures: list[dict[str, str]] = []
    for number, entry in enumerate(manifest_entries, start=1):
        source_entry = discovered.get(entry.protein_id)
        try:
            if source_entry is None:
                raise FileNotFoundError("protein absent from dataset discovery")
            records.append(_prepare_record(entry, source_entry))
        except Exception as exc:
            failures.append({
                "protein_id": entry.protein_id,
                "source": entry.source,
                "stage": "cohort_preparation",
                "reason": f"{type(exc).__name__}: {exc}",
            })
            traceback.print_exc()
        if number % 25 == 0 or number == len(manifest_entries):
            log_progress(
                f"Prepared {number}/{len(manifest_entries)} frozen proteins; "
                f"failures={len(failures)}"
            )
    return records, failures


def build_pair_split_rows(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    return [
        {
            "analysis_name": ANALYSIS_NAME,
            "protein_id": str(record["protein_id"]),
            "source": str(record["source"]),
            "master_role": str(record["master_role"]),
            "protein_split": str(record["protein_split"]),
            "n_protein_pairs": int(record["n_pairs"]),
            "n_train_pairs": int(len(record["train_idx"])),
            "n_heldout_pairs": int(len(record["heldout_idx"])),
            "split_seed": int(record["split_seed"]),
            "split_digest": str(record["split_digest"]),
        }
        for record in records
    ]


def pair_split_manifest_digest(rows: Sequence[Mapping[str, object]]) -> str:
    normalized = [
        {
            "analysis_name": str(row["analysis_name"]),
            "protein_id": str(row["protein_id"]),
            "source": str(row["source"]),
            "master_role": str(row["master_role"]),
            "protein_split": str(row["protein_split"]),
            "n_protein_pairs": int(row["n_protein_pairs"]),
            "n_train_pairs": int(row["n_train_pairs"]),
            "n_heldout_pairs": int(row["n_heldout_pairs"]),
            "split_seed": int(row["split_seed"]),
            "split_digest": str(row["split_digest"]),
        }
        for row in rows
    ]
    normalized.sort(key=lambda row: str(row["protein_id"]))
    return semantic_digest(normalized)


def global_development_training_mean(records: Sequence[Mapping[str, object]]) -> float:
    return protein_equal_mean(
        np.asarray(record["y_true"], dtype=np.float64)[record["train_idx"]]
        for record in records
        if record["protein_split"] == "development"
    )


def build_base256_feature_matrix(record: Mapping[str, object], layer: int) -> np.ndarray:
    path = resolve_pair_block_path(
        Path(record["protein_dir"]), str(record["protein_id"]), layer,
    )
    if not path.exists():
        raise FileNotFoundError(f"pair block layer {layer} not found at {path}")
    pair_block = np.load(str(path))
    expected = (int(record["n_residues"]), int(record["n_residues"]), 128)
    try:
        if pair_block.shape != expected:
            raise ValueError(f"pair-block shape {pair_block.shape} != {expected}")
        return extract_directional_base256(
            pair_block,
            np.asarray(record["pair_i"], dtype=np.int64),
            np.asarray(record["pair_j"], dtype=np.int64),
        )
    finally:
        del pair_block


def training_scaler(features: np.ndarray, train_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    values = np.asarray(features, dtype=np.float64)
    train = values[np.asarray(train_idx, dtype=np.int64)]
    mean = np.mean(train, axis=0, dtype=np.float64)
    scale = np.std(train, axis=0, dtype=np.float64)
    scale[scale < FLOAT_EPS] = 1.0
    return mean, scale


def _rank_metadata(model: LinearRegression) -> tuple[int, float, float, float]:
    rank = int(getattr(model, "rank_", -1))
    singular = np.asarray(getattr(model, "singular_", []), dtype=np.float64)
    if singular.size == 0:
        return rank, float("nan"), float("nan"), float("nan")
    largest = float(np.max(singular))
    smallest = float(np.min(singular))
    condition = largest / smallest if smallest > 0.0 else float("inf")
    return rank, largest, smallest, condition


def fit_status_from_diagnostics(
    n_train: int,
    rank: int,
    condition_number: float,
) -> tuple[str, str]:
    flags = []
    if n_train < MIN_TRAIN_ROWS:
        flags.append("underdetermined")
    if rank < FEATURE_DIM:
        flags.append("rank_deficient")
    if not math.isfinite(condition_number) or condition_number > CONDITION_LIMIT:
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


def fit_single_protein_layer(
    record: Mapping[str, object], layer: int,
) -> tuple[dict[str, object], np.ndarray, dict[str, np.ndarray | float]]:
    features = build_base256_feature_matrix(record, layer)
    y_true = np.asarray(record["y_true"], dtype=np.float64)
    train_idx = np.asarray(record["train_idx"], dtype=np.int64)
    feature_mean, feature_scale = training_scaler(features, train_idx)
    standardized = np.empty_like(features)
    np.subtract(features, feature_mean, out=standardized)
    np.divide(standardized, feature_scale, out=standardized)
    model = LinearRegression(copy_X=False, fit_intercept=True)
    model.fit(standardized[train_idx], y_true[train_idx])
    predictions = np.asarray(model.predict(standardized), dtype=np.float64)
    coefficient_standardized = np.asarray(model.coef_, dtype=np.float64).copy()
    intercept_standardized = float(model.intercept_)
    coefficient_raw = coefficient_standardized / feature_scale
    intercept_raw = intercept_standardized - float(np.dot(coefficient_raw, feature_mean))
    rank, largest, smallest, condition = _rank_metadata(model)
    fit_status, support_flags = fit_status_from_diagnostics(
        len(train_idx), rank, condition,
    )
    status_row = build_model_status_row(
        analysis_name=ANALYSIS_NAME,
        model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM,
        layer=layer,
        protein_id=str(record["protein_id"]),
        fit_status=fit_status,
        support_flags=support_flags,
        n_training_proteins=1,
        n_training_pairs=len(train_idx),
        design_rank=rank,
        largest_singular_value=largest,
        smallest_singular_value=smallest,
        condition_number=condition,
        random_seed=int(record["split_seed"]),
    )
    fitted = {
        "coefficient_standardized": coefficient_standardized,
        "coefficient_raw_scale": coefficient_raw,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "intercept_standardized": intercept_standardized,
        "intercept_raw_scale": intercept_raw,
    }
    del features, standardized, model
    return status_row, predictions, fitted


def _scope_indices(record: Mapping[str, object], scope: str) -> np.ndarray:
    if scope == "train_pairs":
        return np.asarray(record["train_idx"], dtype=np.int64)
    if scope == "heldout_pairs":
        return np.asarray(record["heldout_idx"], dtype=np.int64)
    if scope == "all_pairs":
        return np.arange(int(record["n_pairs"]), dtype=np.int64)
    raise ValueError(f"Unknown evaluation scope {scope}")


def build_scope_rows(
    record: Mapping[str, object],
    layer: int,
    predictions: np.ndarray,
    prediction_status: str | None = None,
) -> list[dict[str, object]]:
    y_true = np.asarray(record["y_true"], dtype=np.float64)
    pred = np.asarray(predictions, dtype=np.float64)
    return [
        build_per_protein_row(
            analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            layer=layer,
            protein_id=str(record["protein_id"]),
            source=str(record["source"]),
            protein_split=str(record["protein_split"]),
            evaluation_scope=scope,
            n_residues=int(record["n_residues"]),
            y_true=y_true[indices],
            predictions=pred[indices],
            prediction_status=prediction_status,
        )
        for scope in EVALUATION_SCOPES
        for indices in (_scope_indices(record, scope),)
    ]


def _empty_coefficient_state(n_models: int) -> dict[str, np.ndarray]:
    matrix_shape = (n_models, FEATURE_DIM)
    return {
        "protein_id": np.full(n_models, "", dtype="U32"),
        "fit_status": np.full(n_models, "not_started", dtype="U64"),
        "coefficient_standardized": np.full(matrix_shape, np.nan, dtype=np.float64),
        "coefficient_raw_scale": np.full(matrix_shape, np.nan, dtype=np.float64),
        "feature_mean": np.full(matrix_shape, np.nan, dtype=np.float64),
        "feature_scale": np.full(matrix_shape, np.nan, dtype=np.float64),
        "intercept_standardized": np.full(n_models, np.nan, dtype=np.float64),
        "intercept_raw_scale": np.full(n_models, np.nan, dtype=np.float64),
    }


def _set_coefficient_state(
    state: dict[str, np.ndarray],
    index: int,
    record: Mapping[str, object],
    fit_status: str,
    fitted: Mapping[str, object] | None,
) -> None:
    state["protein_id"][index] = str(record["protein_id"])
    state["fit_status"][index] = fit_status
    if fitted is None:
        return
    for name in (
        "coefficient_standardized", "coefficient_raw_scale", "feature_mean", "feature_scale",
    ):
        state[name][index] = np.asarray(fitted[name], dtype=np.float64)
    state["intercept_standardized"][index] = float(fitted["intercept_standardized"])
    state["intercept_raw_scale"][index] = float(fitted["intercept_raw_scale"])


def _coefficient_index_rows(
    records: Sequence[Mapping[str, object]],
    statuses: Sequence[Mapping[str, object]],
    relative_path: str,
    digest: str,
) -> list[dict[str, object]]:
    record_index = {str(record["protein_id"]): index for index, record in enumerate(records)}
    return [
        {
            "analysis_name": ANALYSIS_NAME,
            "model_family": MODEL_FAMILY,
            "layer": int(status["layer"]),
            "protein_id": str(status["protein_id"]),
            "gate_aa_pair": "",
            "gate_sequence_bin": "",
            "shard_path": relative_path,
            "array_row": record_index[str(status["protein_id"])],
            "fit_status": str(status["fit_status"]),
            "feature_dim": FEATURE_DIM,
            "file_sha256": digest,
        }
        for status in statuses
    ]


def _pooled_rows(
    layer: int,
    pooled: Mapping[tuple[str, str], PooledComparisonAccumulator],
) -> list[dict[str, object]]:
    return [
        build_pooled_metrics_row(
            pooled[(protein_split, scope)],
            analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            layer=layer,
            protein_split=protein_split,
            evaluation_scope=scope,
            conditional_baseline_type="per_protein_training_pair_mean",
        )
        for protein_split in ("development", "test")
        for scope in EVALUATION_SCOPES
    ]


def _subgroup_rows(
    layer: int,
    subgroup_sequence: SubgroupAccumulator,
    subgroup_aa: SubgroupAccumulator,
    subgroup_spatial: SubgroupAccumulator,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    kwargs = {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "feature_dim": FEATURE_DIM,
        "layer": layer,
        "protein_split": "test",
        "evaluation_scope": "heldout_pairs",
    }
    return (
        subgroup_sequence.rows(**kwargs),
        subgroup_aa.rows(**kwargs),
        subgroup_spatial.rows(**kwargs),
    )


def _write_layer_checkpoint(
    *,
    root: Path,
    layer: int,
    records: Sequence[Mapping[str, object]],
    processed_count: int,
    per_rows: Sequence[Mapping[str, object]],
    pooled: Mapping[tuple[str, str], PooledComparisonAccumulator],
    status_rows: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    subgroup_sequence: SubgroupAccumulator,
    subgroup_aa: SubgroupAccumulator,
    subgroup_spatial: SubgroupAccumulator,
    coefficient_state: dict[str, np.ndarray],
    layer_started: float,
    n_fitted_pairs: int,
    n_evaluated_pairs: int,
    pair_split_digest: str,
    configuration: Mapping[str, object],
) -> None:
    shard = root / "layer_shards" / f"layer_{layer:02d}"
    coefficient_relative = f"model_artifacts/coefficient_shards/layer_{layer:02d}.npz"
    coefficient_path = root / coefficient_relative
    atomic_savez_compressed(
        coefficient_path,
        **coefficient_state,
        layer=np.asarray(layer, dtype=np.int16),
        feature_names=np.asarray(FEATURE_NAMES),
    )
    coefficient_digest = sha256_file(coefficient_path)
    coefficient_index = _coefficient_index_rows(
        records, status_rows, coefficient_relative, coefficient_digest,
    )
    subgroup_rows = _subgroup_rows(
        layer, subgroup_sequence, subgroup_aa, subgroup_spatial,
    )
    elapsed = time.monotonic() - layer_started
    n_failures = sum(not str(row["fit_status"]).startswith("fitted") for row in status_rows)
    completion_status = (
        "complete" if processed_count == len(records) and n_failures == 0
        else "partial_failure" if processed_count == len(records)
        else "in_progress"
    )
    runtime_row = build_runtime_row(
        analysis_name=ANALYSIS_NAME,
        model_family=MODEL_FAMILY,
        layer=layer,
        completion_status=completion_status,
        n_fitted_proteins=len(status_rows) - n_failures,
        n_evaluated_proteins=sum(
            row["evaluation_scope"] == "all_pairs" and int(row["n_evaluated_pairs"]) > 0
            for row in per_rows
        ),
        n_fitted_pairs=n_fitted_pairs,
        n_evaluated_pairs=n_evaluated_pairs,
        elapsed_seconds=elapsed,
        peak_rss_mb=peak_rss_mb(),
    )
    tables = {
        "per_protein.csv": (PER_PROTEIN_COLUMNS, list(per_rows)),
        "pooled_metrics.csv": (POOLED_METRICS_COLUMNS, _pooled_rows(layer, pooled)),
        "model_status.csv": (MODEL_STATUS_COLUMNS, list(status_rows)),
        "runtime.csv": (RUNTIME_COLUMNS, [runtime_row]),
        "geometry_metrics.csv": (GEOMETRY_METRICS_COLUMNS, list(geometry_rows)),
        "geometry_summary.csv": (
            GEOMETRY_SUMMARY_COLUMNS, summarize_geometry_rows(geometry_rows),
        ),
        "subgroup_metrics/by_sequence_separation.csv": (
            SUBGROUP_METRICS_COLUMNS, subgroup_rows[0],
        ),
        "subgroup_metrics/by_amino_acid_pair.csv": (
            SUBGROUP_METRICS_COLUMNS, subgroup_rows[1],
        ),
        "subgroup_metrics/by_spatial_distance.csv": (
            SUBGROUP_METRICS_COLUMNS, subgroup_rows[2],
        ),
        "coefficient_index.csv": (COEFFICIENT_INDEX_COLUMNS, coefficient_index),
    }
    artifact_hashes = {}
    for filename, (columns, rows) in tables.items():
        path = shard / filename
        atomic_write_csv(path, columns, rows)
        artifact_hashes[filename] = sha256_file(path)
    atomic_write_json(shard / "shard_manifest.json", {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "protocol_version": PROTOCOL_VERSION,
        "master_split_digest": MASTER_MANIFEST_DIGEST,
        "pair_split_manifest_digest": pair_split_digest,
        "configuration_digest": semantic_digest(configuration),
        "configuration": dict(configuration),
        "versions": version_payload(),
        "code_revision": code_revision(),
        "layer": layer,
        "expected_proteins": len(records),
        "processed_proteins": processed_count,
        "fit_failures": n_failures,
        "completion_status": completion_status,
        "coefficient_shard": coefficient_relative,
        "coefficient_sha256": coefficient_digest,
        "artifact_sha256": artifact_hashes,
        "updated_at": utc_now(),
    })


def run_layer(
    *,
    root: Path,
    layer: int,
    records: Sequence[Mapping[str, object]],
    global_mean: float,
    pair_split_digest: str,
    configuration: Mapping[str, object],
    checkpoint_every: int,
) -> int:
    layer_started = time.monotonic()
    per_rows: list[dict[str, object]] = []
    status_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    pooled = {
        (protein_split, scope): PooledComparisonAccumulator()
        for protein_split in ("development", "test")
        for scope in EVALUATION_SCOPES
    }
    for accumulator in pooled.values():
        accumulator.conditional_baseline = RegressionStats()
    subgroup_sequence = SubgroupAccumulator(
        "sequence_separation", tuple(SEQUENCE_BIN_LABELS),
    )
    subgroup_aa = SubgroupAccumulator(
        "amino_acid_pair", tuple(AMINO_ACID_PAIR_LABELS),
    )
    subgroup_spatial = SubgroupAccumulator(
        "spatial_distance", tuple(SPATIAL_DISTANCE_BIN_LABELS),
    )
    coefficient_state = _empty_coefficient_state(len(records))
    n_fitted_pairs = 0
    n_evaluated_pairs = 0
    failures = 0
    log_progress(f"Layer {layer:02d}: starting {len(records)} per-protein fits")
    for record_index, record in enumerate(records):
        protein_id = str(record["protein_id"])
        predictions = np.full(int(record["n_pairs"]), np.nan, dtype=np.float64)
        fitted = None
        prediction_status = None
        try:
            status_row, predictions, fitted = fit_single_protein_layer(record, layer)
            n_fitted_pairs += len(record["train_idx"])
        except Exception as exc:
            failures += 1
            prediction_status = "fit_failed"
            status_row = build_model_status_row(
                analysis_name=ANALYSIS_NAME,
                model_family=MODEL_FAMILY,
                feature_dim=FEATURE_DIM,
                layer=layer,
                protein_id=protein_id,
                fit_status="fit_failed",
                support_flags=f"{type(exc).__name__}: {exc}",
                n_training_proteins=1,
                n_training_pairs=len(record["train_idx"]),
                random_seed=int(record["split_seed"]),
            )
            log_progress(f"Layer {layer:02d}: FIT FAILED {protein_id}: {exc}")
            traceback.print_exc()
        status_rows.append(status_row)
        _set_coefficient_state(
            coefficient_state,
            record_index,
            record,
            str(status_row["fit_status"]),
            fitted,
        )
        scope_rows = build_scope_rows(
            record, layer, predictions, prediction_status=prediction_status,
        )
        per_rows.extend(scope_rows)
        y_true = np.asarray(record["y_true"], dtype=np.float64)
        conditional_mean = float(np.mean(y_true[record["train_idx"]], dtype=np.float64))
        for scope in EVALUATION_SCOPES:
            indices = _scope_indices(record, scope)
            pooled[(str(record["protein_split"]), scope)].update_protein(
                protein_id,
                y_true[indices],
                predictions[indices],
                global_mean,
                conditional_baseline_predictions=conditional_mean,
            )
        if record["protein_split"] == "test":
            heldout = np.asarray(record["heldout_idx"], dtype=np.int64)
            subgroup_sequence.update_protein(
                protein_id,
                y_true[heldout],
                predictions[heldout],
                np.asarray(record["sequence_codes"])[heldout],
            )
            subgroup_aa.update_protein(
                protein_id,
                y_true[heldout],
                predictions[heldout],
                np.asarray(record["aa_codes"])[heldout],
            )
            subgroup_spatial.update_protein(
                protein_id,
                y_true[heldout],
                predictions[heldout],
                np.asarray(record["spatial_codes"])[heldout],
            )
        geometry_rows.append(build_geometry_row(
            analysis_name=ANALYSIS_NAME,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            layer=layer,
            protein_id=protein_id,
            source=str(record["source"]),
            protein_split=str(record["protein_split"]),
            n_residues=int(record["n_residues"]),
            true_coordinates=np.asarray(record["coordinates"], dtype=np.float64),
            pair_i=np.asarray(record["pair_i"], dtype=np.int64),
            pair_j=np.asarray(record["pair_j"], dtype=np.int64),
            predictions=predictions,
        ))
        n_evaluated_pairs += int(np.count_nonzero(np.isfinite(predictions)))
        processed = record_index + 1
        if processed % checkpoint_every == 0 or processed == len(records):
            _write_layer_checkpoint(
                root=root,
                layer=layer,
                records=records,
                processed_count=processed,
                per_rows=per_rows,
                pooled=pooled,
                status_rows=status_rows,
                geometry_rows=geometry_rows,
                subgroup_sequence=subgroup_sequence,
                subgroup_aa=subgroup_aa,
                subgroup_spatial=subgroup_spatial,
                coefficient_state=coefficient_state,
                layer_started=layer_started,
                n_fitted_pairs=n_fitted_pairs,
                n_evaluated_pairs=n_evaluated_pairs,
                pair_split_digest=pair_split_digest,
                configuration=configuration,
            )
            log_progress(
                f"Layer {layer:02d}: checkpointed {processed}/{len(records)} proteins"
            )
        del predictions, fitted
        gc.collect()
    log_progress(
        f"Layer {layer:02d}: finished in {time.monotonic() - layer_started:.1f}s; "
        f"fit_failures={failures}"
    )
    return failures


def run_per_protein_job(
    variant,
    title: str,
    layer_indices=None,
    output_filename_suffix: str = "",
    checkpoint_every: int = 25,
) -> None:
    """Run canonical layer shards; the suffix argument remains for old entrypoint compatibility."""
    del output_filename_suffix
    layers = normalize_layer_indices(layer_indices)
    root = output_root(str(variant["name"]))
    root.mkdir(parents=True, exist_ok=True)
    print(title)
    print("=" * len(title))
    log_progress(f"Output root: {root}")
    log_progress(f"Layers: {format_layer_indices(layers)}")
    records, preparation_failures = prepare_records()
    atomic_write_csv(
        root / "metadata" / "preparation_failures.csv",
        PREPARATION_FAILURE_COLUMNS,
        preparation_failures,
    )
    if preparation_failures or len(records) != 552:
        raise RuntimeError(
            f"Frozen-cohort preparation incomplete: {len(records)}/552 records, "
            f"{len(preparation_failures)} failures"
        )
    pair_rows = build_pair_split_rows(records)
    pair_digest = pair_split_manifest_digest(pair_rows)
    atomic_write_csv(root / "metadata" / "pair_splits.csv", PAIR_SPLIT_COLUMNS, pair_rows)
    global_mean = global_development_training_mean(records)
    configuration = configuration_payload(layers)
    configuration["pair_split_manifest_digest"] = pair_digest
    configuration["global_development_training_mean"] = global_mean
    failures = 0
    for layer in layers:
        failures += run_layer(
            root=root,
            layer=layer,
            records=records,
            global_mean=global_mean,
            pair_split_digest=pair_digest,
            configuration=configuration,
            checkpoint_every=checkpoint_every,
        )
    if failures:
        raise RuntimeError(
            f"Saved explicit failure rows for {failures} protein-layer fits; run is incomplete"
        )


def make_base256_per_protein_variant(name: str = ANALYSIS_NAME):
    return make_variant(name, pair_mode="concat_directional")


__all__ = [
    "ANALYSIS_NAME", "CONDITION_LIMIT", "EVALUATION_SCOPES", "MIN_TRAIN_ROWS",
    "MODEL_FAMILY", "PAIR_SPLIT_COLUMNS", "build_base256_feature_matrix",
    "build_pair_split_rows", "build_scope_rows", "configuration_payload",
    "fit_single_protein_layer", "fit_status_from_diagnostics",
    "global_development_training_mean", "make_base256_per_protein_variant",
    "pair_split_manifest_digest", "prepare_records", "run_layer",
    "run_per_protein_job", "training_scaler",
]

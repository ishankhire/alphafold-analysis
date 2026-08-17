#!/usr/bin/env python3
"""Canonical preparation, fitting, recombination, and shard helpers for gated OLS."""

from __future__ import annotations

import csv
import json
import math
import os
import platform
import resource
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
GENERAL_DIR = CLUSTER_SCRIPTS_DIR / "generalized_model_ecod"
for import_dir in (CLUSTER_SCRIPTS_DIR, GENERAL_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

import general_train_test_common as general_common  # noqa: E402
from regression_output_helpers import (  # noqa: E402
    GEOMETRY_METRICS_COLUMNS,
    GEOMETRY_SUMMARY_COLUMNS,
    MODEL_STATUS_COLUMNS,
    PER_PROTEIN_COLUMNS,
    POOLED_METRICS_COLUMNS,
    RUNTIME_COLUMNS,
    SUBGROUP_METRICS_COLUMNS,
    PooledComparisonAccumulator,
    SubgroupAccumulator,
    atomic_savez_compressed,
    atomic_write_csv,
    atomic_write_json,
    build_geometry_row,
    build_model_status_row,
    build_per_protein_row,
    build_pooled_metrics_row,
    build_runtime_row,
    sha256_file,
    summarize_geometry_rows,
)
from regression_protocol import (  # noqa: E402
    AMINO_ACID_PAIR_LABELS,
    FEATURE_DIM,
    FEATURE_NAMES,
    FLOAT_EPS,
    LAYERS,
    MASTER_MANIFEST_DIGEST,
    PROTOCOL_VERSION,
    SEQUENCE_BIN_LABELS,
    SPATIAL_DISTANCE_BIN_LABELS,
    ProteinEqualFeatureMoments,
    extract_directional_base256,
    protein_equal_weights,
    semantic_digest,
)


ANALYSIS_SEQUENCE = "analysis_1_sequence_gated"
ANALYSIS_AA = "analysis_2_aa_pair_gated"
ANALYSIS_JOINT = "analysis_4_aa_sequence_gated"
ANALYSIS_NAMES = (ANALYSIS_SEQUENCE, ANALYSIS_AA, ANALYSIS_JOINT)
MODEL_FAMILY = "gated_ols"
MIN_TRAIN_ROWS = FEATURE_DIM + 1
MIN_TRAIN_PROTEINS = 5
CONDITION_LIMIT = 1e12
RUN_MODE_ENV = "GATED_PROBE_RUN_MODE"
VALID_RUN_MODES = {"production", "smoke"}
PREPARATION_MANIFEST_FILENAME = "preparation_manifest.json"
PREPARED_PROTEINS_FILENAME = "prepared_proteins.csv"
CONDITIONAL_BASELINES_FILENAME = "metadata/conditional_baselines.csv"

PREPARED_PROTEIN_COLUMNS = (
    "protein_id", "source", "master_role", "protein_split", "n_residues",
    "n_pairs", "size_bin",
)
BASELINE_COLUMNS = (
    "analysis_name", "gate_aa_pair", "gate_sequence_bin", "n_training_proteins",
    "n_training_pairs", "prediction",
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f} MiB] {message}", flush=True)


def run_mode() -> str:
    value = os.environ.get(RUN_MODE_ENV, "production").strip().lower()
    if value not in VALID_RUN_MODES:
        raise ValueError(f"{RUN_MODE_ENV} must be one of {sorted(VALID_RUN_MODES)}, got {value!r}")
    return value


def configured_layers(mode: str | None = None) -> tuple[int, ...]:
    return (0,) if (mode or run_mode()) == "smoke" else LAYERS


def suite_root(mode: str | None = None) -> Path:
    selected_mode = mode or run_mode()
    csv_root = Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))).expanduser()
    name = "gated_linear_probes_smoke" if selected_mode == "smoke" else "gated_linear_probes"
    return csv_root / name


def consolidated_root(mode: str | None = None) -> Path:
    return suite_root(mode) / "consolidated"


def configuration_payload(mode: str | None = None) -> dict[str, object]:
    selected_mode = mode or run_mode()
    return {
        "run_mode": selected_mode,
        "layers": list(configured_layers(selected_mode)),
        "cohort_reference_layer": 0,
        "feature_definition": "strict_lower_triangle_[Z_ij|Z_ji]",
        "feature_dim": FEATURE_DIM,
        "storage_precision": "float32_feature_cache",
        "fitting_and_statistics_precision": "float64",
        "fitting_roles": ["fit", "validation"],
        "evaluation_protein_splits": ["development", "test"],
        "protein_weighting": "protein_equal_within_gate",
        "sequence_bins": list(SEQUENCE_BIN_LABELS),
        "amino_acid_pairs": list(AMINO_ACID_PAIR_LABELS),
        "spatial_distance_bins": list(SPATIAL_DISTANCE_BIN_LABELS),
        "minimum_training_rows": MIN_TRAIN_ROWS,
        "minimum_training_proteins": MIN_TRAIN_PROTEINS,
        "condition_number_flag_threshold": CONDITION_LIMIT,
        "joint_fallback_policy": "none",
        "geometry_requires_complete_pair_coverage": True,
        "master_split_digest": MASTER_MANIFEST_DIGEST,
        "protocol_version": PROTOCOL_VERSION,
    }


def configuration_digest(mode: str | None = None) -> str:
    return semantic_digest(configuration_payload(mode))


def analysis_configuration_payload(
    analysis_name: str,
    mode: str | None = None,
    preparation_metadata: Mapping[str, object] | None = None,
) -> dict[str, object]:
    if analysis_name not in ANALYSIS_NAMES:
        raise ValueError(f"Unknown gated analysis {analysis_name}")
    preparation = dict(preparation_metadata or {})
    return {
        **configuration_payload(mode),
        "gate_family": analysis_name,
        "canonical_layer_shards": True,
        "coefficient_storage": "indexed_compressed_npz_shards",
        "input_roots": dict(preparation.get("input_roots", {})),
        "prepared_protein_digest": str(preparation.get("prepared_protein_digest", "")),
        "prepared_proteins": int(preparation.get("prepared_proteins", 0)),
        "prepared_role_counts": dict(preparation.get("prepared_role_counts", {})),
    }


def analysis_configuration_digest(
    analysis_name: str,
    mode: str | None = None,
    preparation_metadata: Mapping[str, object] | None = None,
) -> str:
    return semantic_digest(
        analysis_configuration_payload(analysis_name, mode, preparation_metadata)
    )


def version_payload() -> dict[str, str]:
    return {
        "python": platform.python_version(),
        "numpy": np.__version__,
        "scikit_learn": sklearn.__version__,
    }


def layer_for_task(task_id: int, mode: str | None = None) -> int:
    layers = configured_layers(mode)
    if task_id < 0 or task_id >= len(layers):
        raise ValueError(f"Layer task {task_id} outside [0, {len(layers) - 1}]")
    return layers[task_id]


def joint_task_for_id(task_id: int, mode: str | None = None) -> tuple[int, int]:
    layers = configured_layers(mode)
    task_count = len(layers) * len(SEQUENCE_BIN_LABELS)
    if task_id < 0 or task_id >= task_count:
        raise ValueError(f"Joint task {task_id} outside [0, {task_count - 1}]")
    return layers[task_id // len(SEQUENCE_BIN_LABELS)], task_id % len(SEQUENCE_BIN_LABELS)


def expected_gates(
    analysis_name: str,
    sequence_task_code: int | None = None,
) -> list[tuple[int | None, int | None]]:
    if analysis_name == ANALYSIS_SEQUENCE:
        return [(None, code) for code in range(len(SEQUENCE_BIN_LABELS))]
    if analysis_name == ANALYSIS_AA:
        return [(code, None) for code in range(len(AMINO_ACID_PAIR_LABELS))]
    if analysis_name == ANALYSIS_JOINT:
        if sequence_task_code is None:
            raise ValueError("Joint analysis requires a sequence-bin task code")
        return [(code, sequence_task_code) for code in range(len(AMINO_ACID_PAIR_LABELS))]
    raise ValueError(f"Unknown gated analysis {analysis_name}")


def gate_labels(aa_code: int | None, sequence_code: int | None) -> tuple[str, str]:
    aa_label = "all" if aa_code is None else AMINO_ACID_PAIR_LABELS[aa_code]
    sequence_label = "all" if sequence_code is None else SEQUENCE_BIN_LABELS[sequence_code]
    return aa_label, sequence_label


def gate_slug(aa_code: int | None, sequence_code: int | None) -> str:
    aa = "all" if aa_code is None else f"aa_{aa_code:03d}"
    sequence = "all" if sequence_code is None else f"seq_{sequence_code}"
    return f"{aa}__{sequence}"


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def load_context(root: str | Path | None = None) -> tuple[dict[str, object], list[dict[str, object]]]:
    root_path = Path(root or suite_root())
    manifest = json.loads((root_path / PREPARATION_MANIFEST_FILENAME).read_text())
    if manifest["configuration_digest"] != configuration_digest(str(manifest["run_mode"])):
        raise ValueError("Preparation configuration digest does not match current code")
    if manifest["master_split_digest"] != MASTER_MANIFEST_DIGEST:
        raise ValueError("Preparation does not use the canonical master split")
    source_manifest = {
        row["protein_id"]: row for row in general_common.build_dataset_protein_manifest()
    }
    records = []
    for row in _read_csv(root_path / PREPARED_PROTEINS_FILENAME):
        protein_id = row["protein_id"]
        source = source_manifest.get(protein_id)
        if source is None:
            raise ValueError(f"Prepared protein {protein_id} is absent from the input-path manifest")
        with np.load(root_path / "metadata" / "per_protein" / f"{protein_id}.npz") as data:
            metadata = {name: np.asarray(data[name]) for name in data.files}
        records.append({
            "protein_id": protein_id,
            "source": row["source"],
            "master_role": row["master_role"],
            "protein_split": row["protein_split"],
            "n_residues": int(row["n_residues"]),
            "n_pairs": int(row["n_pairs"]),
            "size_bin": int(row["size_bin"]),
            "protein_dir": Path(source["protein_dir"]),
            "metadata": metadata,
        })
    return manifest, records


def load_conditional_baselines(root: str | Path) -> dict[tuple[str, str, str], dict[str, object]]:
    result = {}
    for row in _read_csv(Path(root) / CONDITIONAL_BASELINES_FILENAME):
        key = (row["analysis_name"], row["gate_aa_pair"], row["gate_sequence_bin"])
        result[key] = {
            "prediction": float(row["prediction"]),
            "n_training_proteins": int(row["n_training_proteins"]),
            "n_training_pairs": int(row["n_training_pairs"]),
        }
    return result


def gate_mask(record: Mapping[str, object], aa_code: int | None, sequence_code: int | None) -> np.ndarray:
    metadata = record["metadata"]
    mask = np.ones(int(record["n_pairs"]), dtype=bool)
    if aa_code is not None:
        mask &= metadata["aa_pair_code"] == aa_code
    if sequence_code is not None:
        mask &= metadata["sequence_bin_code"] == sequence_code
    return mask


def record_offsets(records: Sequence[Mapping[str, object]]) -> tuple[np.ndarray, int]:
    offsets = np.zeros(len(records) + 1, dtype=np.int64)
    for index, record in enumerate(records):
        offsets[index + 1] = offsets[index] + int(record["n_pairs"])
    return offsets, int(offsets[-1])


def load_layer_feature_cache(records: Sequence[Mapping[str, object]], layer: int) -> dict[str, np.ndarray]:
    """Load each layer tensor once; cache storage may be float32, fitting never is."""
    cache: dict[str, np.ndarray] = {}
    for index, record in enumerate(records, start=1):
        path = general_common.resolve_pair_block_path(
            record["protein_dir"], record["protein_id"], layer,
        )
        pair_block = np.load(path)
        expected_shape = (int(record["n_residues"]), int(record["n_residues"]), 128)
        if pair_block.shape != expected_shape:
            raise ValueError(
                f"Unexpected pair-block shape for {record['protein_id']} layer {layer}: "
                f"{pair_block.shape} != {expected_shape}"
            )
        pair_i = record["metadata"]["pair_i"]
        pair_j = record["metadata"]["pair_j"]
        features = extract_directional_base256(pair_block, pair_i, pair_j).astype(np.float32)
        if not np.all(np.isfinite(features)):
            raise ValueError(f"Nonfinite pair features for {record['protein_id']} layer {layer}")
        cache[str(record["protein_id"])] = features
        if index % 25 == 0 or index == len(records):
            log(f"Layer {layer}: cached {index}/{len(records)} proteins")
    return cache


def materialize_gate(
    records: Sequence[Mapping[str, object]],
    feature_cache: Mapping[str, np.ndarray],
    aa_code: int | None,
    sequence_code: int | None,
    *,
    global_offsets: np.ndarray | None = None,
    record_codes: Sequence[int] | None = None,
) -> dict[str, object]:
    if record_codes is not None and len(record_codes) != len(records):
        raise ValueError("record_codes must contain one global code per supplied record")
    selected = []
    total = 0
    for local_record_code, record in enumerate(records):
        record_code = (
            local_record_code if record_codes is None else int(record_codes[local_record_code])
        )
        indices = np.flatnonzero(gate_mask(record, aa_code, sequence_code))
        if indices.size:
            selected.append((record_code, record, indices))
            total += int(indices.size)
    x = np.empty((total, FEATURE_DIM), dtype=np.float64)
    y = np.empty(total, dtype=np.float64)
    protein_ids = np.empty(total, dtype=object)
    flat_indices = np.empty(total, dtype=np.int64)
    blocks = []
    cursor = 0
    for record_code, record, indices in selected:
        end = cursor + indices.size
        x[cursor:end] = feature_cache[str(record["protein_id"])][indices]
        y[cursor:end] = record["metadata"]["y_true"][indices]
        protein_ids[cursor:end] = str(record["protein_id"])
        base = 0 if global_offsets is None else int(global_offsets[record_code])
        flat_indices[cursor:end] = base + indices
        blocks.append((record_code, cursor, end))
        cursor = end
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(y)):
        raise ValueError("Gate materialization produced nonfinite fitting values")
    return {
        "x": x,
        "y": y,
        "protein_ids": protein_ids,
        "flat_indices": flat_indices,
        "blocks": blocks,
    }


def gate_eligibility(n_rows: int, n_proteins: int) -> tuple[str, list[str]]:
    flags = []
    if n_rows < MIN_TRAIN_ROWS:
        flags.append("insufficient_train_rows")
    if n_proteins < MIN_TRAIN_PROTEINS:
        flags.append("insufficient_train_proteins")
    if "insufficient_train_rows" in flags:
        return "insufficient_train_rows", flags
    if "insufficient_train_proteins" in flags:
        return "insufficient_train_proteins", flags
    return "eligible", flags


def fit_ols(x: np.ndarray, y: np.ndarray, protein_ids: Sequence[object]) -> dict[str, object]:
    if x.dtype != np.float64 or y.dtype != np.float64:
        raise TypeError("OLS feature and target arrays must be float64")
    ids = np.asarray(protein_ids, dtype=object)
    if ids.shape != y.shape:
        raise ValueError("One protein identifier is required per fitting row")
    moments = ProteinEqualFeatureMoments()
    string_ids = ids.astype(str)
    for protein_id in sorted(set(string_ids)):
        moments.update_protein(x[string_ids == protein_id])
    feature_mean, feature_scale = moments.finalize()
    standardized = np.empty_like(x)
    np.subtract(x, feature_mean, out=standardized)
    np.divide(standardized, feature_scale, out=standardized)
    weights = protein_equal_weights(ids)
    model = LinearRegression(copy_X=False)
    model.fit(standardized, y, sample_weight=weights)
    coefficient_standardized = np.asarray(model.coef_, dtype=np.float64)
    intercept_standardized = float(model.intercept_)
    coefficient_raw = coefficient_standardized / feature_scale
    intercept_raw = intercept_standardized - float(np.dot(coefficient_raw, feature_mean))
    singular = np.asarray(model.singular_, dtype=np.float64)
    rank = int(model.rank_)
    largest = float(singular[0]) if singular.size else float("nan")
    smallest = float(singular[-1]) if singular.size else float("nan")
    condition = largest / smallest if singular.size and smallest > 0 else float("inf")
    flags = []
    if rank < FEATURE_DIM:
        flags.append("rank_deficient")
    if not math.isfinite(condition) or condition > CONDITION_LIMIT:
        flags.append("ill_conditioned")
    status = (
        "fitted_rank_deficient" if "rank_deficient" in flags
        else "fitted_ill_conditioned" if "ill_conditioned" in flags
        else "fitted"
    )
    return {
        "model": model,
        "feature_mean": feature_mean,
        "feature_scale": feature_scale,
        "coefficient_standardized": coefficient_standardized,
        "coefficient_raw_scale": coefficient_raw,
        "intercept_standardized": intercept_standardized,
        "intercept_raw_scale": intercept_raw,
        "design_rank": rank,
        "largest_singular_value": largest,
        "smallest_singular_value": smallest,
        "condition_number": condition,
        "fit_status": status,
        "support_flags": ";".join(flags),
    }


def predict_ols(fitted: Mapping[str, object], features: np.ndarray) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    standardized = (values - fitted["feature_mean"]) / fitted["feature_scale"]
    return np.asarray(fitted["model"].predict(standardized), dtype=np.float64)


def _empty_coefficient_arrays() -> dict[str, np.ndarray]:
    return {
        "coefficient_standardized": np.full(FEATURE_DIM, np.nan, dtype=np.float64),
        "coefficient_raw_scale": np.full(FEATURE_DIM, np.nan, dtype=np.float64),
        "feature_mean": np.full(FEATURE_DIM, np.nan, dtype=np.float64),
        "feature_scale": np.full(FEATURE_DIM, np.nan, dtype=np.float64),
        "intercept_standardized": np.asarray(np.nan, dtype=np.float64),
        "intercept_raw_scale": np.asarray(np.nan, dtype=np.float64),
    }


def _gate_paths(task_dir: Path, slug: str) -> tuple[Path, Path, Path]:
    gate_dir = task_dir / "gates"
    return (
        gate_dir / f"{slug}_status.json",
        gate_dir / f"{slug}_coefficients.npz",
        gate_dir / f"{slug}_predictions.npz",
    )


def _json_safe_row(row: Mapping[str, object]) -> dict[str, object]:
    return {
        key: (
            None
            if isinstance(value, (float, np.floating)) and not math.isfinite(float(value))
            else value.item() if isinstance(value, np.generic)
            else value
        )
        for key, value in row.items()
    }


def _load_prediction_fragment(path: Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(path) as data:
        return (
            np.asarray(data["flat_indices"], dtype=np.int64),
            np.asarray(data["predictions"], dtype=np.float64),
            np.asarray(data["conditional_predictions"], dtype=np.float64),
        )


def merge_prediction_fragments(
    total_pairs: int,
    fragment_paths: Sequence[Path],
) -> tuple[np.ndarray, np.ndarray]:
    predictions = np.full(total_pairs, np.nan, dtype=np.float64)
    conditional = np.full(total_pairs, np.nan, dtype=np.float64)
    assigned = np.zeros(total_pairs, dtype=bool)
    for path in fragment_paths:
        indices, fragment_predictions, fragment_conditional = _load_prediction_fragment(path)
        if indices.shape != fragment_predictions.shape or indices.shape != fragment_conditional.shape:
            raise ValueError(f"Prediction fragment arrays do not match in {path}")
        if np.any(assigned[indices]):
            raise ValueError(f"Prediction fragments overlap in {path}")
        assigned[indices] = True
        predictions[indices] = fragment_predictions
        conditional[indices] = fragment_conditional
    return predictions, conditional


def split_flat_by_protein(
    records: Sequence[Mapping[str, object]],
    flat_values: np.ndarray,
) -> dict[str, np.ndarray]:
    offsets, total_pairs = record_offsets(records)
    values = np.asarray(flat_values, dtype=np.float64)
    if values.shape != (total_pairs,):
        raise ValueError("Flat prediction vector does not match prepared pair count")
    return {
        str(record["protein_id"]): values[offsets[index]:offsets[index + 1]]
        for index, record in enumerate(records)
    }


def _global_baseline(manifest: Mapping[str, object]) -> float:
    value = float(manifest["global_development_mean"])
    if not math.isfinite(value):
        raise ValueError("Prepared global development baseline is not finite")
    return value


def build_layer_shard(
    root: str | Path,
    analysis_name: str,
    layer: int,
    records: Sequence[Mapping[str, object]],
    flat_predictions: np.ndarray,
    flat_conditional_predictions: np.ndarray,
    status_rows: Sequence[Mapping[str, object]],
    runtime_row: Mapping[str, object],
    preparation_manifest: Mapping[str, object],
) -> Path:
    """Evaluate one recombined piecewise model and atomically replace its CSV shard files."""
    if analysis_name not in ANALYSIS_NAMES:
        raise ValueError(f"Unknown analysis {analysis_name}")
    predictions = split_flat_by_protein(records, flat_predictions)
    conditional = split_flat_by_protein(records, flat_conditional_predictions)
    per_rows = []
    pooled = {
        split: PooledComparisonAccumulator() for split in ("development", "test")
    }
    subgroup_sequence = SubgroupAccumulator(
        "sequence_separation", tuple(SEQUENCE_BIN_LABELS),
    )
    subgroup_aa = SubgroupAccumulator("amino_acid_pair", tuple(AMINO_ACID_PAIR_LABELS))
    subgroup_spatial = SubgroupAccumulator(
        "spatial_distance", tuple(SPATIAL_DISTANCE_BIN_LABELS),
    )
    geometry_rows = []
    global_mean = _global_baseline(preparation_manifest)
    for record in records:
        protein_id = str(record["protein_id"])
        y_true = np.asarray(record["metadata"]["y_true"], dtype=np.float64)
        pred = predictions[protein_id]
        conditional_pred = conditional[protein_id]
        per_rows.append(build_per_protein_row(
            analysis_name=analysis_name,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            layer=layer,
            protein_id=protein_id,
            source=str(record["source"]),
            protein_split=str(record["protein_split"]),
            evaluation_scope="all_pairs",
            n_residues=int(record["n_residues"]),
            y_true=y_true,
            predictions=pred,
        ))
        pooled[str(record["protein_split"])].update_protein(
            protein_id,
            y_true,
            pred,
            global_mean,
            conditional_baseline_predictions=conditional_pred,
        )
        if record["protein_split"] == "test":
            subgroup_sequence.update_protein(
                protein_id, y_true, pred, record["metadata"]["sequence_bin_code"],
            )
            subgroup_aa.update_protein(
                protein_id, y_true, pred, record["metadata"]["aa_pair_code"],
            )
            subgroup_spatial.update_protein(
                protein_id, y_true, pred, record["metadata"]["distance_bin_code"],
            )
        geometry_rows.append(build_geometry_row(
            analysis_name=analysis_name,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            layer=layer,
            protein_id=protein_id,
            source=str(record["source"]),
            protein_split=str(record["protein_split"]),
            n_residues=int(record["n_residues"]),
            true_coordinates=record["metadata"]["coordinates"],
            pair_i=record["metadata"]["pair_i"],
            pair_j=record["metadata"]["pair_j"],
            predictions=pred,
        ))
    pooled_rows = [
        build_pooled_metrics_row(
            pooled[split],
            analysis_name=analysis_name,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            layer=layer,
            protein_split=split,
            evaluation_scope="all_pairs",
            conditional_baseline_type=(
                "sequence_gate_development_mean" if analysis_name == ANALYSIS_SEQUENCE
                else "amino_acid_pair_gate_development_mean" if analysis_name == ANALYSIS_AA
                else "joint_gate_development_mean"
            ),
        )
        for split in ("development", "test")
    ]
    subgroup_kwargs = {
        "analysis_name": analysis_name,
        "model_family": MODEL_FAMILY,
        "feature_dim": FEATURE_DIM,
        "layer": layer,
        "protein_split": "test",
        "evaluation_scope": "all_pairs",
    }
    shard = Path(root) / analysis_name / "layer_shards" / f"layer_{layer:02d}"
    atomic_write_csv(shard / "per_protein.csv", PER_PROTEIN_COLUMNS, per_rows)
    atomic_write_csv(shard / "pooled_metrics.csv", POOLED_METRICS_COLUMNS, pooled_rows)
    atomic_write_csv(shard / "model_status.csv", MODEL_STATUS_COLUMNS, list(status_rows))
    atomic_write_csv(shard / "runtime.csv", RUNTIME_COLUMNS, [runtime_row])
    atomic_write_csv(shard / "geometry_metrics.csv", GEOMETRY_METRICS_COLUMNS, geometry_rows)
    atomic_write_csv(
        shard / "geometry_summary.csv",
        GEOMETRY_SUMMARY_COLUMNS,
        summarize_geometry_rows(geometry_rows),
    )
    atomic_write_csv(
        shard / "subgroup_metrics" / "by_sequence_separation.csv",
        SUBGROUP_METRICS_COLUMNS,
        subgroup_sequence.rows(**subgroup_kwargs),
    )
    atomic_write_csv(
        shard / "subgroup_metrics" / "by_amino_acid_pair.csv",
        SUBGROUP_METRICS_COLUMNS,
        subgroup_aa.rows(**subgroup_kwargs),
    )
    atomic_write_csv(
        shard / "subgroup_metrics" / "by_spatial_distance.csv",
        SUBGROUP_METRICS_COLUMNS,
        subgroup_spatial.rows(**subgroup_kwargs),
    )
    shard_files = sorted(path for path in shard.rglob("*.csv"))
    atomic_write_json(shard / "shard_manifest.json", {
        "analysis_name": analysis_name,
        "layer": layer,
        "configuration_digest": analysis_configuration_digest(
            analysis_name,
            str(preparation_manifest["run_mode"]),
            preparation_manifest,
        ),
        "master_split_digest": preparation_manifest["master_split_digest"],
        "files": {
            str(path.relative_to(shard)): sha256_file(path) for path in shard_files
        },
        "completed_at": utc_now(),
    })
    return shard


def run_analysis_task(analysis_name: str, task_id: int | None = None) -> None:
    mode = run_mode()
    task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if task_id is None else int(task_id)
    if analysis_name == ANALYSIS_JOINT:
        layer, sequence_task_code = joint_task_for_id(task, mode)
    else:
        layer, sequence_task_code = layer_for_task(task, mode), None
    root = suite_root(mode)
    preparation_manifest, records = load_context(root)
    baselines = load_conditional_baselines(root)
    development = [record for record in records if record["master_role"] in {"fit", "validation"}]
    development_codes = [
        index for index, record in enumerate(records)
        if record["master_role"] in {"fit", "validation"}
    ]
    test_records = [record for record in records if record["master_role"] == "test"]
    test_codes = [
        index for index, record in enumerate(records) if record["master_role"] == "test"
    ]
    if {record["protein_split"] for record in development} != {"development"}:
        raise ValueError("Fitting records must be canonical development proteins")
    offsets, total_pairs = record_offsets(records)
    gates = expected_gates(analysis_name, sequence_task_code)
    task_dir = root / analysis_name / "task_shards" / f"task_{task:03d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_manifest_path = task_dir / "task_manifest.json"
    invariant = {
        "analysis_name": analysis_name,
        "task_id": task,
        "layer": layer,
        "sequence_task_code": sequence_task_code,
        "run_mode": mode,
        "configuration_digest": preparation_manifest["configuration_digest"],
        "analysis_configuration_digest": analysis_configuration_digest(
            analysis_name, mode, preparation_manifest,
        ),
        "master_split_digest": preparation_manifest["master_split_digest"],
        "expected_gate_keys": [gate_slug(*gate) for gate in gates],
        "protein_ids": [record["protein_id"] for record in records],
        "development_protein_ids": [record["protein_id"] for record in development],
    }
    if task_manifest_path.exists():
        task_manifest = json.loads(task_manifest_path.read_text())
        for key, expected in invariant.items():
            if task_manifest.get(key) != expected:
                raise ValueError(f"Existing task manifest mismatch for {key}")
    else:
        task_manifest = {
            **invariant,
            "completed_gate_keys": [],
            "artifacts": {},
            "started_at": utc_now(),
            "completed": False,
        }
        atomic_write_json(task_manifest_path, task_manifest)
    task_started = time.monotonic()
    feature_cache = None
    fragment_paths = []
    status_rows = []
    did_work = False
    for gate_number, (aa_code, sequence_code) in enumerate(gates, start=1):
        slug = gate_slug(aa_code, sequence_code)
        status_path, coefficient_path, prediction_path = _gate_paths(task_dir, slug)
        if slug in task_manifest["completed_gate_keys"]:
            artifacts = task_manifest["artifacts"][slug]
            for path, digest_key in (
                (status_path, "status_sha256"),
                (coefficient_path, "coefficient_sha256"),
                (prediction_path, "prediction_sha256"),
            ):
                if not path.exists() or sha256_file(path) != artifacts[digest_key]:
                    raise ValueError(f"Completed gate has missing or corrupt artifact: {path}")
            fragment_paths.append(prediction_path)
            status_rows.append(json.loads(status_path.read_text()))
            log(f"{analysis_name} layer {layer}: reused gate {gate_number}/{len(gates)} {slug}")
            continue
        if feature_cache is None:
            feature_cache = load_layer_feature_cache(records, layer)
        did_work = True
        gate_started = time.monotonic()
        aa_label, sequence_label = gate_labels(aa_code, sequence_code)
        development_gate = materialize_gate(
            development,
            feature_cache,
            aa_code,
            sequence_code,
            global_offsets=offsets,
            record_codes=development_codes,
        )
        test_gate = materialize_gate(
            test_records,
            feature_cache,
            aa_code,
            sequence_code,
            global_offsets=offsets,
            record_codes=test_codes,
        )
        n_training_pairs = int(len(development_gate["y"]))
        n_training_proteins = len(set(map(str, development_gate["protein_ids"])))
        eligibility, flags = gate_eligibility(n_training_pairs, n_training_proteins)
        fitted = None
        development_predictions = np.full(len(development_gate["y"]), np.nan, dtype=np.float64)
        test_predictions = np.full(len(test_gate["y"]), np.nan, dtype=np.float64)
        if eligibility == "eligible":
            log(
                f"{analysis_name} layer {layer} {slug}: fitting {n_training_pairs:,} rows "
                f"from {n_training_proteins} development proteins"
            )
            fitted = fit_ols(
                development_gate["x"], development_gate["y"], development_gate["protein_ids"],
            )
            development_predictions = predict_ols(fitted, development_gate["x"])
            if len(test_gate["y"]):
                test_predictions = predict_ols(fitted, test_gate["x"])
            fit_status = str(fitted["fit_status"])
            support_flags = str(fitted["support_flags"])
        else:
            fit_status = eligibility
            support_flags = ";".join(flags)
        baseline_key = (analysis_name, aa_label, sequence_label)
        baseline = baselines[baseline_key]
        if (
            baseline["n_training_pairs"] != n_training_pairs
            or baseline["n_training_proteins"] != n_training_proteins
        ):
            raise ValueError(f"Conditional baseline support mismatch for {baseline_key}")
        flat_indices = np.concatenate((
            development_gate["flat_indices"], test_gate["flat_indices"],
        ))
        predictions = np.concatenate((development_predictions, test_predictions))
        conditional_predictions = np.full(predictions.shape, float(baseline["prediction"]))
        status_row = build_model_status_row(
            analysis_name=analysis_name,
            model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM,
            layer=layer,
            fit_status=fit_status,
            support_flags=support_flags,
            gate_aa_pair=aa_label,
            gate_sequence_bin=sequence_label,
            n_training_proteins=n_training_proteins,
            n_training_pairs=n_training_pairs,
            design_rank=float("nan") if fitted is None else fitted["design_rank"],
            largest_singular_value=(
                float("nan") if fitted is None else fitted["largest_singular_value"]
            ),
            smallest_singular_value=(
                float("nan") if fitted is None else fitted["smallest_singular_value"]
            ),
            condition_number=float("nan") if fitted is None else fitted["condition_number"],
        )
        coefficient_arrays = _empty_coefficient_arrays() if fitted is None else {
            name: np.asarray(fitted[name]) for name in (
                "coefficient_standardized", "coefficient_raw_scale", "feature_mean",
                "feature_scale", "intercept_standardized", "intercept_raw_scale",
            )
        }
        atomic_write_json(status_path, _json_safe_row(status_row))
        atomic_savez_compressed(
            coefficient_path,
            **coefficient_arrays,
            fit_status=np.asarray(fit_status),
            gate_aa_pair=np.asarray(aa_label),
            gate_sequence_bin=np.asarray(sequence_label),
            layer=np.asarray(layer, dtype=np.int16),
        )
        atomic_savez_compressed(
            prediction_path,
            flat_indices=np.asarray(flat_indices, dtype=np.int64),
            predictions=predictions,
            conditional_predictions=conditional_predictions,
        )
        task_manifest["artifacts"][slug] = {
            "status_sha256": sha256_file(status_path),
            "coefficient_sha256": sha256_file(coefficient_path),
            "prediction_sha256": sha256_file(prediction_path),
        }
        task_manifest["completed_gate_keys"].append(slug)
        atomic_write_json(task_manifest_path, task_manifest)
        fragment_paths.append(prediction_path)
        status_rows.append(status_row)
        log(
            f"{analysis_name} layer {layer}: checkpointed gate {gate_number}/{len(gates)} "
            f"{slug} in {time.monotonic() - gate_started:.1f}s"
        )
    flat_predictions, flat_conditional = merge_prediction_fragments(total_pairs, fragment_paths)
    task_prediction_path = task_dir / "task_predictions.npz"
    claimed = np.isfinite(flat_conditional)
    atomic_savez_compressed(
        task_prediction_path,
        flat_indices=np.flatnonzero(claimed).astype(np.int64),
        predictions=flat_predictions[claimed],
        conditional_predictions=flat_conditional[claimed],
    )
    elapsed = (
        time.monotonic() - task_started
        if did_work or "elapsed_seconds" not in task_manifest
        else float(task_manifest["elapsed_seconds"])
    )
    task_manifest.update({
        "completed": True,
        "completed_at": utc_now(),
        "elapsed_seconds": elapsed,
        "task_prediction_sha256": sha256_file(task_prediction_path),
    })
    atomic_write_json(task_manifest_path, task_manifest)
    if analysis_name != ANALYSIS_JOINT:
        n_evaluated = int(np.count_nonzero(np.isfinite(flat_predictions)))
        per_record_evaluated = [
            int(np.count_nonzero(np.isfinite(flat_predictions[offsets[index]:offsets[index + 1]])))
            for index in range(len(records))
        ]
        runtime_row = build_runtime_row(
            analysis_name=analysis_name,
            model_family=MODEL_FAMILY,
            layer=layer,
            completion_status="complete",
            n_fitted_proteins=sum(
                per_record_evaluated[index] > 0
                for index, record in enumerate(records)
                if record["master_role"] in {"fit", "validation"}
            ),
            n_evaluated_proteins=sum(count > 0 for count in per_record_evaluated),
            n_fitted_pairs=sum(
                per_record_evaluated[index]
                for index, record in enumerate(records)
                if record["master_role"] in {"fit", "validation"}
            ),
            n_evaluated_pairs=n_evaluated,
            elapsed_seconds=elapsed,
            peak_rss_mb=peak_rss_mb(),
        )
        build_layer_shard(
            root, analysis_name, layer, records, flat_predictions, flat_conditional,
            status_rows, runtime_row, preparation_manifest,
        )
        log(f"{analysis_name} layer {layer}: checkpointed canonical layer shard")
    log(f"Completed {analysis_name} task {task}")


def validate_unique_rows(
    rows: Sequence[Mapping[str, object]],
    key_fields: Sequence[str],
    table_name: str,
) -> None:
    seen = set()
    for row in rows:
        key = tuple(str(row.get(field, "")) for field in key_fields)
        if key in seen:
            raise ValueError(f"Duplicate key in {table_name}: {key}")
        seen.add(key)


__all__ = [
    "ANALYSIS_AA", "ANALYSIS_JOINT", "ANALYSIS_NAMES", "ANALYSIS_SEQUENCE",
    "BASELINE_COLUMNS", "CONDITIONAL_BASELINES_FILENAME", "CONDITION_LIMIT",
    "FEATURE_DIM", "FEATURE_NAMES", "MIN_TRAIN_PROTEINS", "MIN_TRAIN_ROWS",
    "MODEL_FAMILY", "PREPARATION_MANIFEST_FILENAME", "PREPARED_PROTEINS_FILENAME",
    "PREPARED_PROTEIN_COLUMNS", "RUN_MODE_ENV", "analysis_configuration_digest",
    "analysis_configuration_payload", "build_layer_shard", "configuration_digest",
    "configuration_payload", "configured_layers",
    "consolidated_root", "expected_gates", "fit_ols", "gate_eligibility", "gate_labels",
    "gate_mask", "gate_slug", "joint_task_for_id", "layer_for_task", "load_context",
    "load_conditional_baselines", "log", "materialize_gate", "merge_prediction_fragments",
    "peak_rss_mb", "predict_ols", "record_offsets", "run_analysis_task", "run_mode",
    "split_flat_by_protein", "suite_root", "utc_now", "validate_unique_rows",
    "version_payload",
]

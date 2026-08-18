#!/usr/bin/env python3
"""Canonical final shared-XGBoost workflow primitives.

Selection is frozen in ``shared_xgboost_final_config.json``.  This module does
not read exploratory search outputs at runtime.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import platform
import resource
import subprocess
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
GENERALIZED_DIR = CLUSTER_DIR / "generalized_model_ecod"
for _path in (SCRIPT_DIR, CLUSTER_DIR, GENERALIZED_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cluster_regression_helpers import build_unordered_aa_pair_codes  # noqa: E402
from general_train_test_common import (  # noqa: E402
    build_dataset_protein_manifest,
    resolve_pair_block_path,
    resolve_pdb_file,
)
from regression_output_helpers import (  # noqa: E402
    FEATURE_IMPORTANCE_COLUMNS,
    GEOMETRY_METRICS_COLUMNS,
    GEOMETRY_SUMMARY_COLUMNS,
    MODEL_STATUS_COLUMNS,
    PER_PROTEIN_COLUMNS,
    POOLED_METRICS_COLUMNS,
    RUNTIME_COLUMNS,
    SUBGROUP_METRICS_COLUMNS,
    PooledComparisonAccumulator,
    SubgroupAccumulator,
    atomic_write_csv,
    atomic_write_json,
    build_geometry_row,
    build_per_protein_row,
    build_pooled_metrics_row,
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
    SEQUENCE_BIN_LABELS,
    SPATIAL_DISTANCE_BIN_LABELS,
    load_master_manifest,
    partition_manifest_by_role,
    protein_equal_mean,
    protein_equal_weights,
    sequence_separation_bin_codes,
    spatial_distance_bin_codes,
)
from xgboost_search_common import (  # noqa: E402
    capped_pair_indices,
    load_combined_records,
    load_pair_features,
    log,
)


ANALYSIS_NAME = "shared_xgboost_base256_combined"
MODEL_FAMILY = "shared_xgboost"
CONFIG_PATH = SCRIPT_DIR / "shared_xgboost_final_config.json"
ROW_CAP = 5000
BOOSTING_ROUNDS = 700
RANDOM_SEED = 42
CONFIGURATION_ID = "S4"
OUTPUT_SCHEMA_VERSION = "1.0"
CONFIG_DIGEST_FIELD = "configuration_digest"
SPLIT_ASSIGNMENT_COLUMNS = (
    "manifest_version", "master_manifest_digest", "protein_id", "source",
    "n_residues", "n_pairs", "size_bin", "master_role", "protein_split",
    "outer_split_seed", "development_split_seed", "split_assignment_digest",
)
TRAINING_SAMPLE_COLUMNS = (
    "analysis_name", "protein_id", "source", "master_role", "protein_split",
    "n_protein_pairs", "n_selected_pairs", "sampling_seed", "sampling_policy",
    "indices_sha256", "row_weight", "total_protein_weight", "sample_policy_digest",
)
EXCLUDED_COLUMNS = (
    "protein_id", "source", "eligibility_status", "stage", "layer", "reason",
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
    "model_artifacts/feature_importance.csv": FEATURE_IMPORTANCE_COLUMNS,
}


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def final_output_root() -> Path:
    name = (
        f"{ANALYSIS_NAME}_development_smoke"
        if os.environ.get("XGBOOST_FINAL_DEVELOPMENT_SMOKE") == "1"
        else ANALYSIS_NAME
    )
    return Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))) / name


def working_output_root() -> Path:
    name = (
        f"{ANALYSIS_NAME}_development_smoke_work"
        if os.environ.get("XGBOOST_FINAL_DEVELOPMENT_SMOKE") == "1"
        else f"{ANALYSIS_NAME}_work"
    )
    return Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))) / name


def configuration_payload(config: Mapping[str, object]) -> dict[str, object]:
    return {key: value for key, value in config.items() if key != CONFIG_DIGEST_FIELD}


def load_final_configuration(path: str | Path = CONFIG_PATH) -> dict[str, object]:
    """Load and strictly validate the frozen final-model decision artifact."""
    config_path = Path(path)
    config = json.loads(config_path.read_text())
    observed_digest = semantic_digest(configuration_payload(config))
    if config.get(CONFIG_DIGEST_FIELD) != observed_digest:
        raise ValueError("Frozen final configuration digest is missing or changed")
    expected = {
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "configuration_id": CONFIGURATION_ID,
        "training_row_cap_per_development_protein": ROW_CAP,
        "sampling_seed": RANDOM_SEED,
        "feature_mode": "directional_base256",
        "feature_dimension": FEATURE_DIM,
        "master_manifest_digest": MASTER_MANIFEST_DIGEST,
        "split_assignment_digest": "59865846507b7b90c1045ea18eab54d33c854a58c8cc4e3c362adf3a247f5404",
        "training_metrics_used": False,
        "test_metrics_used": False,
        "source_runs_exposed_test": True,
        "historical_test_exposure_disclosed": True,
        "round_selection_status": "validation_boundary_at_700",
        "final_fit_early_stopping": False,
        "final_fit_evaluation_set": False,
    }
    for key, value in expected.items():
        if config.get(key) != value:
            raise ValueError(f"Invalid frozen configuration field {key}: {config.get(key)!r} != {value!r}")
    parameters = config.get("xgboost_parameters")
    expected_parameters = {
        "objective": "reg:squarederror", "tree_method": "hist", "max_depth": 6,
        "learning_rate": 0.1, "min_child_weight": 10, "subsample": 0.8,
        "colsample_bytree": 0.8, "reg_lambda": 1.0, "max_bin": 256,
        "n_estimators": BOOSTING_ROUNDS, "random_state": RANDOM_SEED,
    }
    if parameters != expected_parameters:
        raise ValueError("Frozen XGBoost parameters are not the exact S4/5k/700 specification")
    for filename_key, digest_key in (
        ("validation_selection_table", "validation_selection_table_sha256"),
        ("validation_cap_sweep_table", "validation_cap_sweep_table_sha256"),
    ):
        table = config_path.parent / str(config[filename_key])
        if not table.is_file() or sha256_file(table) != config[digest_key]:
            raise ValueError(f"Frozen selection table is missing or changed: {table}")
        with table.open(newline="") as handle:
            columns = tuple(csv.DictReader(handle).fieldnames or ())
        if any(column.startswith("test_") or column == "split" for column in columns):
            raise ValueError(f"Frozen selection table exposes test data: {table}")
    return config


def split_assignment_rows() -> tuple[list[dict[str, object]], str]:
    entries = sorted(
        (entry for entry in load_master_manifest() if entry.eligible),
        key=lambda entry: entry.protein_id,
    )
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
    } for entry in entries]
    digest = semantic_digest(base)
    return [{**row, "split_assignment_digest": digest} for row in base], digest


def build_training_sample_manifest() -> tuple[list[dict[str, object]], str, dict[str, np.ndarray]]:
    roles = partition_manifest_by_role(load_master_manifest())
    entries = sorted((*roles["fit"], *roles["validation"]), key=lambda entry: entry.protein_id)
    rows: list[dict[str, object]] = []
    indices_by_id: dict[str, np.ndarray] = {}
    policy = "stable_protein_id_permutation_of_strict_lower_triangle"
    for entry in entries:
        indices = capped_pair_indices(entry.n_pairs, entry.protein_id, ROW_CAP, RANDOM_SEED)
        indices_by_id[entry.protein_id] = indices
        rows.append({
            "analysis_name": ANALYSIS_NAME,
            "protein_id": entry.protein_id,
            "source": entry.source,
            "master_role": entry.master_role,
            "protein_split": "development",
            "n_protein_pairs": entry.n_pairs,
            "n_selected_pairs": int(indices.size),
            "sampling_seed": RANDOM_SEED,
            "sampling_policy": policy,
            "indices_sha256": hashlib.sha256(indices.tobytes()).hexdigest(),
            "row_weight": 1.0 / float(indices.size),
            "total_protein_weight": 1.0,
        })
    digest = semantic_digest(rows)
    return [{**row, "sample_policy_digest": digest} for row in rows], digest, indices_by_id


def validate_layer_task_id(task_id: int) -> int:
    layer = int(task_id)
    if layer not in LAYERS:
        raise ValueError(f"Final XGBoost layer task must be 0..47; got {task_id}")
    return layer


def validate_frozen_layer_input_paths(layer: int) -> str:
    """Validate all eligible paths without opening any pair tensor or PDB."""
    dataset = {
        str(row["protein_id"]): row for row in build_dataset_protein_manifest()
    }
    identities = []
    for entry in sorted(
        (item for item in load_master_manifest() if item.eligible),
        key=lambda item: item.protein_id,
    ):
        record = dataset.get(entry.protein_id)
        if record is None or str(record["source"]) != entry.source:
            raise FileNotFoundError(f"Frozen protein is missing from dataset discovery: {entry.protein_id}")
        pair_path = resolve_pair_block_path(
            Path(record["protein_dir"]), entry.protein_id, layer,
        )
        pdb_path = resolve_pdb_file(
            entry.protein_id, Path(record["protein_dir"]), Path(record["pdb_root"]),
        )
        if not pair_path.is_file() or pdb_path is None or not Path(pdb_path).is_file():
            raise FileNotFoundError(
                f"Frozen layer input is missing for {entry.protein_id}: pair={pair_path}, pdb={pdb_path}"
            )
        pair_stat = pair_path.stat()
        pdb_stat = Path(pdb_path).stat()
        identities.append((
            entry.protein_id, str(pair_path), pair_stat.st_size, pair_stat.st_mtime_ns,
            str(pdb_path), pdb_stat.st_size, pdb_stat.st_mtime_ns,
        ))
    if len(identities) != 552:
        raise RuntimeError(f"Expected 552 frozen layer inputs; observed {len(identities)}")
    return semantic_digest(identities)


def materialize_development_matrix(
    records: Sequence[Mapping[str, object]],
    layer: int,
    indices_by_id: Mapping[str, np.ndarray],
) -> dict[str, object]:
    total_rows = sum(len(indices_by_id[str(record["protein_id"])]) for record in records)
    features = np.empty((total_rows, FEATURE_DIM), dtype=np.float64)
    targets = np.empty(total_rows, dtype=np.float64)
    protein_ids = np.empty(total_rows, dtype=object)
    offsets = []
    cursor = 0
    for record in records:
        protein_id = str(record["protein_id"])
        indices = np.asarray(indices_by_id[protein_id], dtype=np.int64)
        stop = cursor + indices.size
        features[cursor:stop] = load_pair_features(
            record, layer, indices, feature_mode="directional_base256",
        )
        targets[cursor:stop] = np.asarray(record["y"], dtype=np.float64)[indices]
        protein_ids[cursor:stop] = protein_id
        offsets.append((record, indices, cursor, stop))
        cursor = stop
    if cursor != total_rows or not np.all(np.isfinite(features)) or not np.all(np.isfinite(targets)):
        raise ValueError("Final development matrix is incomplete or nonfinite")
    weights = protein_equal_weights(protein_ids)
    return {
        "x": features, "y": targets, "weights": weights,
        "protein_ids": protein_ids, "offsets": offsets,
    }


def require_xgboost():
    try:
        import xgboost  # type: ignore
        from xgboost import XGBRegressor  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("XGBoost is required; activate ~/protein_venv on Delta") from exc
    return xgboost, XGBRegressor


def build_final_estimator(config: Mapping[str, object], n_jobs: int):
    """Construct the exact fixed estimator; fitting receives no evaluation set."""
    _, estimator_type = require_xgboost()
    parameters = dict(config["xgboost_parameters"])
    return estimator_type(**parameters, n_jobs=int(n_jobs))


def fit_final_estimator(model, bundle: Mapping[str, object]) -> None:
    model.fit(bundle["x"], bundle["y"], sample_weight=bundle["weights"])


def feature_importance_rows(model, layer: int) -> list[dict[str, object]]:
    booster = model.get_booster()
    gain = booster.get_score(importance_type="gain")
    weight = booster.get_score(importance_type="weight")
    for mapping in (gain, weight):
        for key in mapping:
            if not key.startswith("f") or not key[1:].isdigit() or int(key[1:]) >= FEATURE_DIM:
                raise ValueError(f"Unexpected booster feature key: {key!r}")
    return [{
        "analysis_name": ANALYSIS_NAME,
        "model_family": MODEL_FAMILY,
        "layer": layer,
        "feature_index": index,
        "feature_name": FEATURE_NAMES[index],
        "gain": float(gain.get(f"f{index}", 0.0)),
        "split_count": float(weight.get(f"f{index}", 0.0)),
    } for index in range(FEATURE_DIM)]


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


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 if sys.platform != "darwin" else 1024.0 * 1024.0)


def version_payload() -> dict[str, str]:
    xgboost, _ = require_xgboost()
    return {"python": platform.python_version(), "numpy": np.__version__, "xgboost": xgboost.__version__}


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=SCRIPT_DIR,
            text=True, stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def code_digest() -> str:
    paths = (
        SCRIPT_DIR / "shared_xgboost_final_common.py",
        SCRIPT_DIR / "shared_xgboost_final_layer.py",
        CONFIG_PATH,
        CLUSTER_DIR / "regression_output_helpers.py",
        CLUSTER_DIR / "regression_protocol.py",
    )
    return semantic_digest([(path.name, sha256_file(path)) for path in paths if path.exists()])


def evaluate_records(
    model,
    records: Sequence[Mapping[str, object]],
    layer: int,
    global_development_mean: float,
) -> dict[str, list[dict[str, object]]]:
    per_rows: list[dict[str, object]] = []
    geometry_rows: list[dict[str, object]] = []
    pooled = {split: PooledComparisonAccumulator() for split in ("development", "test")}
    sequence = SubgroupAccumulator("sequence_separation", SEQUENCE_BIN_LABELS)
    amino_acid = SubgroupAccumulator("amino_acid_pair", AMINO_ACID_PAIR_LABELS)
    spatial = SubgroupAccumulator("spatial_distance", SPATIAL_DISTANCE_BIN_LABELS)
    for record in records:
        protein_id = str(record["protein_id"])
        y_true = np.asarray(record["y"], dtype=np.float64)
        features = load_pair_features(record, layer, feature_mode="directional_base256")
        predictions = np.asarray(model.predict(features), dtype=np.float64)
        if predictions.shape != y_true.shape or not np.all(np.isfinite(predictions)):
            raise ValueError(f"Incomplete/nonfinite predictions for {protein_id}")
        split = "test" if record["master_role"] == "test" else "development"
        per_rows.append(build_per_protein_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=layer, protein_id=protein_id,
            source=str(record["source"]), protein_split=split,
            evaluation_scope="all_pairs", n_residues=int(record["n_residues"]),
            y_true=y_true, predictions=predictions,
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
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=layer, protein_id=protein_id,
            source=str(record["source"]), protein_split=split,
            n_residues=int(record["n_residues"]),
            true_coordinates=np.asarray(record["coordinates"], dtype=np.float64),
            pair_i=pair_i, pair_j=pair_j, predictions=predictions,
        ))
    pooled_rows = [build_pooled_metrics_row(
        pooled[split], analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM, layer=layer, protein_split=split,
        evaluation_scope="all_pairs", conditional_baseline_type="not_applicable",
    ) for split in ("development", "test")]
    subgroup_kwargs = dict(
        analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
        feature_dim=FEATURE_DIM, layer=layer, protein_split="test",
        evaluation_scope="all_pairs",
    )
    return {
        "per_protein.csv": per_rows,
        "pooled_metrics.csv": pooled_rows,
        "geometry_metrics.csv": geometry_rows,
        "geometry_summary.csv": summarize_geometry_rows(geometry_rows),
        "subgroup_metrics/by_sequence_separation.csv": sequence.rows(**subgroup_kwargs),
        "subgroup_metrics/by_amino_acid_pair.csv": amino_acid.rows(**subgroup_kwargs),
        "subgroup_metrics/by_spatial_distance.csv": spatial.rows(**subgroup_kwargs),
    }


def write_layer_shard(
    root: Path,
    layer: int,
    tables: Mapping[str, Sequence[Mapping[str, object]]],
    model,
    status_row: Mapping[str, object],
    runtime_row: Mapping[str, object],
    invariants: Mapping[str, object],
    replay: Mapping[str, object],
) -> Path:
    shard = root / "layer_shards" / f"layer_{layer:02d}"
    all_tables = {
        **tables,
        "model_status.csv": [status_row],
        "runtime.csv": [runtime_row],
        "model_artifacts/feature_importance.csv": feature_importance_rows(model, layer),
    }
    for filename, rows in all_tables.items():
        atomic_write_csv(shard / filename, SHARD_CSV_SCHEMAS[filename], list(rows))
    booster_path = shard / "model_artifacts" / f"booster_layer_{layer:02d}.json"
    atomic_save_booster(model, booster_path)
    files = sorted((*[shard / filename for filename in all_tables], booster_path))
    atomic_write_json(shard / "shard_manifest.json", {
        **dict(invariants), "analysis_name": ANALYSIS_NAME, "layer": layer,
        "status": "complete", "files": {
            str(path.relative_to(shard)): sha256_file(path) for path in files
        }, "replay": dict(replay), "completed_at": utc_now(),
    })
    return shard


def write_failure(root: Path, layer: int, phase: str, exc: Exception, config: Mapping[str, object]) -> None:
    atomic_write_json(root / "failures" / f"layer_{layer:02d}.json", {
        "layer": layer, "phase": phase, "exception_type": type(exc).__name__,
        "message": str(exc), "timestamp": utc_now(),
        "configuration_digest": config.get(CONFIG_DIGEST_FIELD, ""),
        "master_manifest_digest": MASTER_MANIFEST_DIGEST,
    })


def expected_layer_counts(n_proteins: int = 552) -> dict[str, int]:
    return {
        "per_protein.csv": n_proteins,
        "pooled_metrics.csv": 2,
        "model_status.csv": 1,
        "runtime.csv": 1,
        "geometry_metrics.csv": n_proteins,
        "geometry_summary.csv": 2 * 14,
        "subgroup_metrics/by_sequence_separation.csv": 6,
        "subgroup_metrics/by_amino_acid_pair.csv": 210,
        "subgroup_metrics/by_spatial_distance.csv": 5,
        "model_artifacts/feature_importance.csv": FEATURE_DIM,
    }


def read_csv_strict(path: Path, columns: Sequence[str]) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"Schema mismatch in {path}: {reader.fieldnames}")
        return list(reader)


def validate_unique_rows(rows: Sequence[Mapping[str, object]], keys: Sequence[str], label: str) -> None:
    values = [tuple(row[key] for key in keys) for row in rows]
    if len(values) != len(set(values)):
        raise ValueError(f"Duplicate primary key in {label}")


__all__ = [name for name in globals() if not name.startswith("_")]

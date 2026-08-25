#!/usr/bin/env python3
"""Replay four frozen probes for one layer and reduce exact triangle subgroups."""

from __future__ import annotations

import csv
import gc
import json
import math
import os
import resource
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
GENERAL_DIR = CLUSTER_DIR / "generalized_model_ecod"
for path in (SCRIPT_DIR, CLUSTER_DIR, GENERAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cluster_regression_helpers import extract_ca_coordinates  # noqa: E402
from general_train_test_common import (  # noqa: E402
    build_dataset_protein_manifest,
    resolve_pair_block_path,
    resolve_pdb_file,
)
from regression_output_helpers import atomic_write_csv, atomic_write_json  # noqa: E402
from regression_protocol import (  # noqa: E402
    FEATURE_DIM,
    MASTER_MANIFEST_DIGEST,
    SEQUENCE_BIN_LABELS,
    extract_directional_base256,
    load_master_manifest,
    semantic_digest,
    sequence_separation_bin_codes,
    strict_lower_triangle_indices,
    within_protein_pair_split,
    within_protein_pair_split_digest,
    within_protein_split_seed,
)
from triangle_violation_subgroup_common import (  # noqa: E402
    ANALYSIS_NAME,
    CATEGORY_LABELS,
    MODEL_IDENTITIES,
    PER_PROTEIN_COLUMNS,
    RUNTIME_COLUMNS,
    THRESHOLD_COLUMNS,
    build_reference_cache,
    compute_metric_rows,
    load_json,
    resolve_result_root,
    sha256_file,
    validate_producer_manifest,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def csv_root() -> Path:
    return Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))).expanduser()


def output_root() -> Path:
    return csv_root() / ANALYSIS_NAME


def configured_artifact_root(environment_name: str, relative: str) -> Path:
    configured = os.environ.get(environment_name)
    return resolve_result_root(Path(configured) if configured else csv_root() / relative)


def read_csv(path: Path, expected_columns: Sequence[str] | None = None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if expected_columns is not None and tuple(reader.fieldnames or ()) != tuple(expected_columns):
            raise ValueError(f"Schema mismatch in {path}: {reader.fieldnames}")
        return list(reader)


def load_reference_thresholds(root: Path) -> dict[str, dict[str, object]]:
    manifest = load_json(root / "run_manifest.json")
    if (
        manifest.get("analysis_name") != ANALYSIS_NAME
        or manifest.get("run_status") != "reference_complete"
        or manifest.get("master_manifest_digest") != MASTER_MANIFEST_DIGEST
        or manifest.get("evaluation_scopes") != ["all_edges"]
    ):
        raise ValueError("Triangle reference manifest is incomplete or incompatible")
    path = root / "metadata/protein_reference_thresholds.csv"
    expected_sha = manifest.get("artifact_sha256", {}).get(
        "metadata/protein_reference_thresholds.csv"
    )
    if expected_sha != sha256_file(path):
        raise ValueError("Frozen reference-threshold checksum mismatch")
    rows = read_csv(path, THRESHOLD_COLUMNS)
    if len(rows) != 111:
        raise ValueError(f"Expected 111 reference rows, observed {len(rows)}")
    numeric = set(THRESHOLD_COLUMNS) - {
        "protein_id", "source", "protein_split", "protein_length_bin",
        "reference_digest", "status",
    }
    converted: dict[str, dict[str, object]] = {}
    for source_row in rows:
        row: dict[str, object] = dict(source_row)
        for key in numeric:
            row[key] = float(source_row[key])
        for key in (
            "n_residues", "n_triangles", "n_rms_small", "n_rms_medium",
            "n_rms_large", "n_clearly_acute", "n_near_right",
            "n_clearly_obtuse", "n_exact90_acute",
            "n_exact90_right_within_tolerance", "n_exact90_obtuse",
            "n_area_low", "n_area_medium", "n_area_high", "n_slack_low",
            "n_slack_medium", "n_slack_high",
        ):
            row[key] = int(float(source_row[key]))
        if row["protein_split"] != "test" or row["status"] != "complete":
            raise ValueError(f"Invalid reference row for {row['protein_id']}")
        converted[str(row["protein_id"])] = row
    return converted


def _verify_manifest_artifact(root: Path, manifest: Mapping[str, object], relative: str) -> str:
    observed = sha256_file(root / relative)
    expected_map = manifest.get("artifact_sha256", {})
    if isinstance(expected_map, Mapping) and relative in expected_map and expected_map[relative] != observed:
        raise ValueError(f"Artifact checksum mismatch for {root / relative}")
    return observed


def validate_layer_model_status(
    root: Path,
    *,
    layer: int,
    analysis_name: str,
    model_family: str,
    accepted_statuses: set[str],
    expected_count: int,
    protein_ids: Sequence[str] | None = None,
) -> str:
    path = root / "model_status.csv"
    rows = [
        row for row in read_csv(path)
        if int(row["layer"]) == layer
        and (protein_ids is None or row["protein_id"] in set(protein_ids))
    ]
    if len(rows) != expected_count:
        raise ValueError(f"Model-status coverage mismatch for {analysis_name} layer {layer}")
    for row in rows:
        if (
            row["analysis_name"] != analysis_name
            or row["model_family"] != model_family
            or int(row["feature_dim"]) != FEATURE_DIM
            or row["fit_status"] not in accepted_statuses
        ):
            raise ValueError(f"Invalid producer model status for {analysis_name} layer {layer}")
    return sha256_file(path)


def load_shared_ols(root: Path, layer: int) -> tuple[float, np.ndarray, dict[str, object]]:
    manifest = validate_producer_manifest(
        root,
        analysis_name="general_base256_combined",
        model_family="shared_ols",
        master_digest=MASTER_MANIFEST_DIGEST,
    )
    relative = "model_artifacts/coefficients.csv"
    checksum = _verify_manifest_artifact(root, manifest, relative)
    rows = [
        row for row in read_csv(root / relative)
        if int(row["layer"]) == layer
    ]
    if len(rows) != FEATURE_DIM + 1:
        raise ValueError(f"Shared OLS layer {layer} does not have 257 coefficient rows")
    by_index = {int(row["feature_index"]): row for row in rows}
    if set(by_index) != {-1, *range(FEATURE_DIM)}:
        raise ValueError("Shared OLS coefficient indices are incomplete")
    coefficients = np.asarray(
        [float(by_index[index]["coefficient_raw_scale"]) for index in range(FEATURE_DIM)],
        dtype=np.float64,
    )
    intercept = float(by_index[-1]["coefficient_raw_scale"])
    if not np.all(np.isfinite(coefficients)) or not math.isfinite(intercept):
        raise ValueError("Shared OLS raw coefficients are nonfinite")
    return intercept, coefficients, {
        "root": str(root), "manifest_sha256": sha256_file(root / "run_manifest.json"),
        "coefficient_sha256": checksum,
        "model_status_sha256": validate_layer_model_status(
            root, layer=layer, analysis_name="general_base256_combined",
            model_family="shared_ols", accepted_statuses={"historical_fit_replayed"},
            expected_count=1,
        ),
        "coefficient_serialization": "historical_six_decimal",
    }


def load_indexed_coefficients(
    root: Path,
    *,
    layer: int,
    analysis_name: str,
    model_family: str,
    protein_ids: Sequence[str] | None = None,
    gated: bool = False,
) -> tuple[dict[str, tuple[float, np.ndarray]] | list[tuple[float, np.ndarray]], dict[str, object]]:
    validate_producer_manifest(
        root, analysis_name=analysis_name, model_family=model_family,
        master_digest=MASTER_MANIFEST_DIGEST,
    )
    index_path = root / "model_artifacts/coefficient_index.csv"
    rows = [row for row in read_csv(index_path) if int(row["layer"]) == layer]
    if gated:
        if len(rows) != len(SEQUENCE_BIN_LABELS):
            raise ValueError(f"Sequence-gated layer {layer} must contain six gates")
        by_gate = {row["gate_sequence_bin"]: row for row in rows}
        if set(by_gate) != set(SEQUENCE_BIN_LABELS):
            raise ValueError("Sequence-gated coefficient labels are incomplete")
        ordered_rows = [by_gate[label] for label in SEQUENCE_BIN_LABELS]
    else:
        expected = set(protein_ids or ())
        by_protein = {row["protein_id"]: row for row in rows if row["protein_id"] in expected}
        if set(by_protein) != expected:
            raise ValueError("Per-protein coefficient index does not cover every test protein")
        ordered_rows = [by_protein[protein_id] for protein_id in protein_ids or ()]

    npz_cache: dict[Path, object] = {}
    values: list[tuple[float, np.ndarray]] = []
    observed_shards: dict[str, str] = {}
    try:
        for row in ordered_rows:
            if row["fit_status"] != "fitted" or int(row["feature_dim"]) != FEATURE_DIM:
                raise ValueError("Indexed coefficient row is not a fitted base256 model")
            shard = root / row["shard_path"]
            checksum = observed_shards.get(str(shard))
            if checksum is None:
                checksum = sha256_file(shard)
                observed_shards[str(shard)] = checksum
            if checksum != row["file_sha256"]:
                raise ValueError(f"Indexed coefficient shard checksum mismatch: {shard}")
            if shard not in npz_cache:
                npz_cache[shard] = np.load(shard, allow_pickle=False)
            archive = npz_cache[shard]
            array_row = int(row["array_row"])
            coefficients = np.asarray(archive["coefficient_raw_scale"][array_row], dtype=np.float64)
            intercept = float(archive["intercept_raw_scale"][array_row])
            if coefficients.shape != (FEATURE_DIM,) or not np.all(np.isfinite(coefficients)):
                raise ValueError("Indexed raw coefficient vector is malformed")
            values.append((intercept, coefficients.copy()))
    finally:
        for archive in npz_cache.values():
            archive.close()
    provenance = {
        "root": str(root),
        "manifest_sha256": sha256_file(root / "run_manifest.json"),
        "coefficient_index_sha256": sha256_file(index_path),
        "coefficient_shards": observed_shards,
        "model_status_sha256": validate_layer_model_status(
            root, layer=layer, analysis_name=analysis_name, model_family=model_family,
            accepted_statuses={"fitted"}, expected_count=len(ordered_rows),
            protein_ids=protein_ids,
        ),
    }
    if gated:
        return values, provenance
    return dict(zip(protein_ids or (), values)), provenance


def validate_per_protein_pair_splits(root: Path, entries: Sequence[object]) -> dict[str, object]:
    """Reconstruct and verify the producer's canonical within-protein splits."""
    manifest = load_json(root / "run_manifest.json")
    path = root / "metadata/pair_splits.csv"
    rows = read_csv(path)
    normalized = []
    by_id = {}
    for row in rows:
        protein_id = row["protein_id"]
        if protein_id in by_id:
            raise ValueError(f"Duplicate per-protein split row for {protein_id}")
        by_id[protein_id] = row
        normalized.append({
            "analysis_name": row["analysis_name"],
            "protein_id": protein_id,
            "source": row["source"],
            "master_role": row["master_role"],
            "protein_split": row["protein_split"],
            "n_protein_pairs": int(row["n_protein_pairs"]),
            "n_train_pairs": int(row["n_train_pairs"]),
            "n_heldout_pairs": int(row["n_heldout_pairs"]),
            "split_seed": int(row["split_seed"]),
            "split_digest": row["split_digest"],
        })
    normalized.sort(key=lambda row: str(row["protein_id"]))
    observed_manifest_digest = semantic_digest(normalized)
    if observed_manifest_digest != manifest.get("pair_split_manifest_digest"):
        raise ValueError("Per-protein pair-split manifest digest mismatch")
    for entry in entries:
        row = by_id.get(entry.protein_id)
        if row is None:
            raise ValueError(f"Missing per-protein pair split for {entry.protein_id}")
        train, heldout = within_protein_pair_split(entry.n_pairs, entry.protein_id)
        if (
            row["analysis_name"] != "per_protein_base256_combined"
            or row["source"] != entry.source
            or row["protein_split"] != "test"
            or int(row["n_protein_pairs"]) != entry.n_pairs
            or int(row["n_train_pairs"]) != train.size
            or int(row["n_heldout_pairs"]) != heldout.size
            or int(row["split_seed"]) != within_protein_split_seed(entry.protein_id)
            or row["split_digest"]
            != within_protein_pair_split_digest(entry.n_pairs, entry.protein_id)
        ):
            raise ValueError(f"Canonical pair-split replay mismatch for {entry.protein_id}")
    return {
        "pair_splits_sha256": sha256_file(path),
        "pair_split_manifest_digest": observed_manifest_digest,
        "n_validated_test_proteins": len(entries),
    }


def load_artifacts(layer: int, protein_ids: Sequence[str]):
    shared_root = configured_artifact_root(
        "TRIANGLE_SHARED_OLS_ROOT", "general_base256_combined_canonical"
    )
    per_root = configured_artifact_root(
        "TRIANGLE_PER_PROTEIN_OLS_ROOT", "per_protein_base256_combined"
    )
    xgb_root = configured_artifact_root(
        "TRIANGLE_XGBOOST_ROOT", "shared_xgboost_base256_combined"
    )
    gated_root = configured_artifact_root(
        "TRIANGLE_SEQUENCE_GATED_ROOT",
        "gated_linear_probes/consolidated/analysis_1_sequence_gated",
    )
    shared_intercept, shared_coefficients, shared_provenance = load_shared_ols(
        shared_root, layer
    )
    shared = (shared_intercept, shared_coefficients)
    per, per_provenance = load_indexed_coefficients(
        per_root, layer=layer, analysis_name="per_protein_base256_combined",
        model_family="per_protein_ols", protein_ids=protein_ids,
    )
    entries_by_id = {entry.protein_id: entry for entry in test_entries()}
    per_provenance.update(
        validate_per_protein_pair_splits(
            per_root, [entries_by_id[protein_id] for protein_id in protein_ids]
        )
    )
    gated, gated_provenance = load_indexed_coefficients(
        gated_root, layer=layer, analysis_name="analysis_1_sequence_gated",
        model_family="gated_ols", gated=True,
    )
    xgb_manifest = validate_producer_manifest(
        xgb_root, analysis_name="shared_xgboost_base256_combined",
        model_family="shared_xgboost", master_digest=MASTER_MANIFEST_DIGEST,
    )
    booster_path = xgb_root / f"model_artifacts/booster_layer_{layer:02d}.json"
    if not booster_path.is_file():
        raise FileNotFoundError(booster_path)
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is required to replay the frozen booster") from exc
    booster = xgb.Booster()
    booster.load_model(booster_path)
    provenance = {
        "layer": layer,
        "master_manifest_digest": MASTER_MANIFEST_DIGEST,
        "shared_ols": shared_provenance,
        "per_protein_ols": per_provenance,
        "sequence_gated_ols": gated_provenance,
        "shared_xgboost": {
            "root": str(xgb_root),
            "manifest_sha256": sha256_file(xgb_root / "run_manifest.json"),
            "booster_sha256": sha256_file(booster_path),
            "booster_config": xgb_manifest.get("xgboost_booster_config"),
            "xgboost_version": xgb.__version__,
            "model_status_sha256": validate_layer_model_status(
                xgb_root, layer=layer,
                analysis_name="shared_xgboost_base256_combined",
                model_family="shared_xgboost", accepted_statuses={"complete"},
                expected_count=1,
            ),
        },
    }
    return shared, per, gated, booster, xgb, provenance


def test_entries():
    entries = sorted(
        (
            entry for entry in load_master_manifest()
            if entry.eligible and entry.protein_split == "test"
        ),
        key=lambda entry: entry.protein_id,
    )
    if len(entries) != 111:
        raise RuntimeError(f"Expected 111 locked-test proteins, observed {len(entries)}")
    return entries


def load_protein(entry, discovered: Mapping[str, object], layer: int):
    protein_dir = Path(discovered["protein_dir"])
    pdb_path = resolve_pdb_file(entry.protein_id, protein_dir, Path(discovered["pdb_root"]))
    if pdb_path is None:
        raise FileNotFoundError("Final predicted PDB not found")
    ca_data = extract_ca_coordinates(pdb_path)
    coordinates = np.asarray(ca_data["coordinates"], dtype=np.float64)
    residues = tuple(ca_data["residues"])
    if coordinates.shape != (entry.n_residues, 3):
        raise ValueError("Final-PDB C-alpha count differs from frozen manifest")
    pair_i, pair_j = strict_lower_triangle_indices(entry.n_residues)
    pair_path = resolve_pair_block_path(protein_dir, entry.protein_id, layer)
    pair_block = np.load(pair_path)
    try:
        expected = (entry.n_residues, entry.n_residues, 128)
        if pair_block.shape != expected:
            raise ValueError(f"Pair tensor shape {pair_block.shape} != {expected}")
        features = extract_directional_base256(pair_block, pair_i, pair_j)
    finally:
        del pair_block
    chain_ids = np.asarray([str(residue[0]) for residue in residues], dtype="U32")
    sequence_codes = sequence_separation_bin_codes(pair_i, pair_j, chain_ids=chain_ids)
    if np.any(sequence_codes < 0):
        raise ValueError("Sequence-gated replay encountered an interchain pair without a frozen gate")
    return coordinates, pair_i, pair_j, features, sequence_codes


def symmetric_matrix(n: int, pair_i: np.ndarray, pair_j: np.ndarray, values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != pair_i.shape or not np.all(np.isfinite(values)):
        raise ValueError("Prediction vector is malformed")
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[pair_i, pair_j] = values
    matrix[pair_j, pair_i] = values
    return matrix


def model_predictions(
    entry,
    features: np.ndarray,
    sequence_codes: np.ndarray,
    shared,
    per,
    gated,
    booster,
    xgb,
) -> list[tuple[str, str, str, np.ndarray]]:
    predictions = []
    shared_intercept, shared_coef = shared
    predictions.append((
        "general_base256_combined", "shared_ols", "unseen_proteins",
        shared_intercept + features @ shared_coef,
    ))
    per_intercept, per_coef = per[entry.protein_id]
    predictions.append((
        "per_protein_base256_combined", "per_protein_ols",
        "mixed_within_protein_train_and_heldout", per_intercept + features @ per_coef,
    ))
    predictions.append((
        "shared_xgboost_base256_combined", "shared_xgboost", "unseen_proteins",
        np.asarray(booster.predict(xgb.DMatrix(features)), dtype=np.float64),
    ))
    gated_prediction = np.empty(features.shape[0], dtype=np.float64)
    for code, (intercept, coefficients) in enumerate(gated):
        mask = sequence_codes == code
        gated_prediction[mask] = intercept + features[mask] @ coefficients
    predictions.append((
        "analysis_1_sequence_gated", "gated_ols", "unseen_proteins", gated_prediction,
    ))
    if tuple((name, family, scope) for name, family, scope, _ in predictions) != MODEL_IDENTITIES:
        raise RuntimeError("Replay model identity/order changed")
    return predictions


def checkpoint_valid_rows(path: Path, layer: int) -> tuple[list[dict[str, str]], set[str]]:
    if not path.is_file():
        return [], set()
    rows = read_csv(path, PER_PROTEIN_COLUMNS)
    grouped: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        if int(row["layer"]) != layer:
            raise ValueError("Checkpoint contains a different layer")
        grouped.setdefault(row["protein_id"], []).append(row)
    expected_per_protein = len(MODEL_IDENTITIES) * sum(map(len, CATEGORY_LABELS.values()))
    complete_ids = {
        protein_id for protein_id, protein_rows in grouped.items()
        if len(protein_rows) == expected_per_protein
    }
    if len(complete_ids) != len(grouped):
        raise ValueError("Checkpoint contains a partially written protein")
    return rows, complete_ids


def main() -> None:
    layer = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if layer not in range(48):
        raise ValueError(f"SLURM_ARRAY_TASK_ID must select layer 0..47; got {layer}")
    layer_started_wall = utc_now()
    layer_started = time.perf_counter()
    root = output_root()
    shards = root / "layer_shards"
    shards.mkdir(parents=True, exist_ok=True)
    final_path = shards / f"per_protein_triangle_metrics_layer_{layer:02d}.csv"
    checkpoint_path = shards / f".per_protein_triangle_metrics_layer_{layer:02d}.checkpoint.csv"
    runtime_path = shards / f"runtime_layer_{layer:02d}.csv"
    if final_path.exists():
        log(f"final layer shard already exists: {final_path}")
        return

    thresholds = load_reference_thresholds(root)
    entries = test_entries()
    protein_ids = [entry.protein_id for entry in entries]
    shared, per, gated, booster, xgb, provenance = load_artifacts(layer, protein_ids)
    atomic_write_json(shards / f"artifact_provenance_layer_{layer:02d}.json", provenance)
    discovered_rows = build_dataset_protein_manifest()
    discovered = {str(row["protein_id"]): row for row in discovered_rows}
    rows, complete_ids = checkpoint_valid_rows(checkpoint_path, layer)
    runtime_rows: list[dict[str, object]] = []
    if runtime_path.is_file():
        runtime_rows.extend(read_csv(runtime_path, RUNTIME_COLUMNS))
    failures: list[str] = []
    checkpoint_every = int(os.environ.get("TRIANGLE_CHECKPOINT_PROTEINS", "5"))
    temp_parent = Path(os.environ.get("TRIANGLE_TMPDIR", "/tmp"))

    for number, entry in enumerate(entries, start=1):
        if entry.protein_id in complete_ids:
            log(f"layer={layer} protein={entry.protein_id} resumed from checkpoint")
            continue
        protein_started_wall = utc_now()
        protein_started = time.perf_counter()
        try:
            source_record = discovered.get(entry.protein_id)
            if source_record is None:
                raise FileNotFoundError("Protein absent from dataset discovery")
            coordinates, pair_i, pair_j, features, sequence_codes = load_protein(
                entry, source_record, layer
            )
            with tempfile.TemporaryDirectory(
                prefix=f"triangle_replay_l{layer:02d}_{entry.protein_id}_", dir=temp_parent,
            ) as temporary:
                cache, observed_reference = build_reference_cache(
                    coordinates, temporary, thresholds=thresholds[entry.protein_id]
                )
                if observed_reference["reference_digest"] != thresholds[entry.protein_id]["reference_digest"]:
                    raise ValueError("Reference geometry digest changed since preparation")
                for analysis_name, model_family, generalization_scope, prediction in model_predictions(
                    entry, features, sequence_codes, shared, per, gated, booster, xgb
                ):
                    model_started = time.perf_counter()
                    matrix = symmetric_matrix(entry.n_residues, pair_i, pair_j, prediction)
                    identity = {
                        "analysis_name": analysis_name,
                        "model_family": model_family,
                        "layer": layer,
                        "protein_id": entry.protein_id,
                        "source": entry.source,
                        "protein_split": "test",
                        "generalization_scope": generalization_scope,
                        "n_residues": entry.n_residues,
                        "protein_length_bin": f"length_bin_{entry.size_bin + 1}",
                    }
                    rows.extend(compute_metric_rows(matrix, cache, identity))
                    elapsed_model = time.perf_counter() - model_started
                    runtime_rows.append({
                        "stage": f"replay_model:{analysis_name}", "layer": layer,
                        "protein_id": entry.protein_id, "started_at": protein_started_wall,
                        "finished_at": utc_now(), "elapsed_seconds": elapsed_model,
                        "peak_rss_mb": peak_rss_mb(), "status": "complete", "detail": "",
                    })
                    log(
                        f"layer={layer} protein={entry.protein_id} model={analysis_name} "
                        f"elapsed={elapsed_model:.1f}s"
                    )
                cache.cleanup()
            complete_ids.add(entry.protein_id)
            status = "complete"
        except Exception as exc:
            status = "failed"
            failures.append(f"{entry.protein_id}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            runtime_rows.append({
                "stage": "replay_protein", "layer": layer, "protein_id": entry.protein_id,
                "started_at": protein_started_wall, "finished_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - protein_started,
                "peak_rss_mb": peak_rss_mb(), "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}",
            })
        log(
            f"layer={layer} progress={number}/111 protein={entry.protein_id} status={status} "
            f"elapsed={time.perf_counter() - protein_started:.1f}s"
        )
        if number % checkpoint_every == 0 or status == "failed" or number == len(entries):
            atomic_write_csv(checkpoint_path, PER_PROTEIN_COLUMNS, rows)
            atomic_write_csv(runtime_path, RUNTIME_COLUMNS, runtime_rows)
        if "features" in locals():
            del features
        gc.collect()

    if failures or complete_ids != set(protein_ids):
        raise RuntimeError(f"Layer {layer} incomplete; failures={failures}")
    expected_rows = 111 * len(MODEL_IDENTITIES) * sum(map(len, CATEGORY_LABELS.values()))
    if len(rows) != expected_rows:
        raise RuntimeError(f"Layer {layer} has {len(rows)} rows, expected {expected_rows}")
    atomic_write_csv(final_path, PER_PROTEIN_COLUMNS, rows)
    checkpoint_path.unlink(missing_ok=True)
    runtime_rows.append({
        "stage": "layer_complete", "layer": layer, "protein_id": "",
        "started_at": layer_started_wall, "finished_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - layer_started,
        "peak_rss_mb": peak_rss_mb(), "status": "complete", "detail": "",
    })
    atomic_write_csv(runtime_path, RUNTIME_COLUMNS, runtime_rows)
    log(f"layer={layer} complete elapsed={time.perf_counter() - layer_started:.1f}s")


if __name__ == "__main__":
    main()

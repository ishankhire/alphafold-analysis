#!/usr/bin/env python3
"""Replay all four frozen model families OOF for one Evoformer layer."""

from __future__ import annotations

import csv
import gc
import os
import resource
import sys
import tempfile
import time
import traceback
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
CV_DIR = CLUSTER_DIR / "cross_validation"
for path in (SCRIPT_DIR, CLUSTER_DIR, CV_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cluster_regression_helpers import (  # noqa: E402
    extract_ca_coordinates,
    resolve_combined_pair_block_path,
    resolve_combined_pdb_path,
)
from cross_validation_common import FoldLayerTask, load_cv_folds  # noqa: E402
from per_protein_ols_base256_common import load_coefficient_shard  # noqa: E402
from regression_output_helpers import atomic_write_csv, atomic_write_json  # noqa: E402
from regression_protocol import (  # noqa: E402
    SEQUENCE_BIN_LABELS,
    extract_directional_base256,
    pair_fold_assignment_digest,
    sequence_separation_bin_codes,
    strict_lower_triangle_indices,
    within_protein_pair_folds,
)
from sequence_gated_ols_base256_common import load_coefficient_artifact as load_gated_ols  # noqa: E402
from shared_ols_base256_common import load_coefficient_artifact as load_shared_ols  # noqa: E402
from triangle_violation_subgroup_common import (  # noqa: E402
    ANALYSIS_NAME,
    CATEGORY_LABELS,
    CV_PROTOCOL_VERSION,
    EVALUATION_SCOPE,
    EXPECTED_PROTEINS,
    FAMILY_BY_NAME,
    FAMILY_SPECS,
    FEATURE_DIM,
    LAYERS,
    N_PAIR_FOLDS,
    PER_PROTEIN_COLUMNS,
    RUNTIME_COLUMNS,
    THRESHOLD_COLUMNS,
    assemble_oof_predictions,
    build_reference_cache,
    compute_metric_rows,
    index_replay_rows,
    load_and_validate_replay_index,
    load_json,
    read_csv_strict,
    sha256_file,
    symmetric_pair_matrix,
    validate_family_manifest,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def csv_root() -> Path:
    return Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))).expanduser()


def producer_root() -> Path:
    return csv_root() / "regression_5fold_v1"


def output_root() -> Path:
    return csv_root() / "analysis" / ANALYSIS_NAME


def _artifact_path(root: Path, row: Mapping[str, str]) -> Path:
    path = root / row["artifact_path"]
    if not path.is_file() or sha256_file(path) != row["artifact_sha256"]:
        raise ValueError(f"Artifact is missing or changed: {path}")
    return path


def load_layer_artifacts(layer: int, fold_bundle) -> tuple[dict[str, object], dict[str, object]]:
    """Validate all producers, then load only the models needed by this layer."""
    artifacts: dict[str, object] = {}
    provenance: dict[str, object] = {
        "analysis_name": ANALYSIS_NAME, "layer": layer, "created_at": utc_now(),
        "metadata_digests": dict(fold_bundle.fold_digests), "producers": {},
    }
    for spec in FAMILY_SPECS:
        root = producer_root() / spec.family_dir
        manifest = validate_family_manifest(root, spec, fold_bundle.fold_digests)
        # Preparation already audited the complete inventory. Each layer task
        # revalidates the full index contract, then checksums every artifact it
        # actually opens through ``_artifact_path`` below.
        rows = load_and_validate_replay_index(root, spec, manifest, verify_artifacts=False)
        selected = [row for row in rows if int(row["layer"]) == layer]
        provenance["producers"][spec.analysis_name] = {
            "root": str(root), "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
            "replay_index_sha256": sha256_file(root / "model_artifacts/replay_index.csv"),
            "configuration_digest": manifest["configuration_digest"],
            "replay_rows": selected,
        }
        if spec.analysis_name == "shared_ols_base256":
            index = index_replay_rows(selected, ("layer", "outer_fold"))
            models = {}
            for fold in range(5):
                row = index[(str(layer), str(fold))]
                fit = load_shared_ols(
                    _artifact_path(root, row),
                    expected_task=FoldLayerTask(fold * 48 + layer, fold, layer),
                    expected_scaler_digest=row["scaler_digest"],
                )
                if not str(fit["fit_status"]).startswith("fitted"):
                    raise ValueError("Shared OLS artifact is not a completed fit")
                models[fold] = fit
            artifacts[spec.analysis_name] = models
        elif spec.analysis_name == "sequence_gated_ols_base256":
            index = index_replay_rows(selected, ("layer", "outer_fold", "sequence_gate"))
            models = {}
            for fold in range(5):
                for gate in SEQUENCE_BIN_LABELS:
                    row = index[(str(layer), str(fold), gate)]
                    fit = load_gated_ols(
                        _artifact_path(root, row),
                        expected_task=FoldLayerTask(fold * 48 + layer, fold, layer),
                        expected_sequence_gate=gate,
                        expected_scaler_digest=row["scaler_digest"],
                    )
                    if not str(fit["fit_status"]).startswith("fitted"):
                        raise ValueError("Gated OLS artifact is not a completed fit")
                    models[(fold, gate)] = fit
            artifacts[spec.analysis_name] = models
        elif spec.analysis_name == "per_protein_ols_base256":
            expected = EXPECTED_PROTEINS * N_PAIR_FOLDS
            if len(selected) != expected:
                raise ValueError(f"Per-protein replay layer has {len(selected)} rows, expected {expected}")
            index = index_replay_rows(selected, ("layer", "protein_id", "pair_fold"))
            artifact_paths = {row["artifact_path"] for row in selected}
            if len(artifact_paths) != 1:
                raise ValueError("Per-protein layer must resolve to exactly one coefficient shard")
            shard_path = _artifact_path(root, selected[0])
            shard = load_coefficient_shard(shard_path, expected_layer=layer)
            responsibilities = list(zip(
                np.asarray(shard["protein_id"]).astype(str).tolist(),
                np.asarray(shard["pair_fold"], dtype=np.int16).tolist(),
            ))
            row_by_key = {(protein_id, int(pair_fold)): row_index for row_index, (protein_id, pair_fold) in enumerate(responsibilities)}
            if len(row_by_key) != expected:
                raise ValueError("Per-protein coefficient shard responsibilities are not unique/complete")
            for protein_id in fold_bundle.eligible_protein_ids:
                for pair_fold in range(N_PAIR_FOLDS):
                    replay = index[(str(layer), protein_id, str(pair_fold))]
                    position = row_by_key[(protein_id, pair_fold)]
                    if str(np.asarray(shard["scaler_digest"])[position]) != replay["scaler_digest"]:
                        raise ValueError("Per-protein shard scaler digest mismatch")
                    if not str(np.asarray(shard["fit_status"])[position]).startswith("fitted"):
                        raise ValueError("Per-protein artifact is not a completed fit")
            artifacts[spec.analysis_name] = (shard, row_by_key)
        elif spec.analysis_name == "shared_xgboost_base256_fixed_s4":
            try:
                import xgboost as xgb
            except ImportError as exc:
                raise RuntimeError("xgboost is required to replay the frozen booster family") from exc
            index = index_replay_rows(selected, ("layer", "outer_fold"))
            models = {}
            for fold in range(5):
                row = index[(str(layer), str(fold))]
                booster = xgb.Booster()
                booster.load_model(str(_artifact_path(root, row)))
                models[fold] = booster
            artifacts[spec.analysis_name] = models
        else:  # pragma: no cover - guarded by the frozen family table
            raise AssertionError(spec.analysis_name)
        log(f"layer={layer} validated/loaded producer={spec.analysis_name}")
    return artifacts, provenance


def load_thresholds(root: Path) -> dict[str, dict[str, str]]:
    rows = read_csv_strict(root / "metadata/protein_reference_thresholds.csv", THRESHOLD_COLUMNS)
    if len(rows) != EXPECTED_PROTEINS or any(row["status"] != "complete" for row in rows):
        raise ValueError("Frozen reference thresholds are incomplete")
    output = {row["protein_id"]: row for row in rows}
    if len(output) != EXPECTED_PROTEINS:
        raise ValueError("Frozen reference threshold protein IDs are duplicated")
    return output


def load_protein(outer, layer: int) -> tuple[np.ndarray, tuple[str, ...], np.ndarray, np.ndarray, np.ndarray]:
    pdb_path = resolve_combined_pdb_path(outer.protein_id, outer.source)
    pair_path = resolve_combined_pair_block_path(outer.protein_id, outer.source, layer)
    if not pdb_path.is_file() or not pair_path.is_file():
        raise FileNotFoundError(f"Missing frozen protein inputs for {outer.protein_id} layer {layer}")
    ca = extract_ca_coordinates(pdb_path)
    coordinates = np.asarray(ca["coordinates"], dtype=np.float64)
    chain_ids = tuple(str(residue[0]) for residue in ca["residues"])
    if coordinates.shape != (int(outer.n_residues), 3) or len(chain_ids) != outer.n_residues:
        raise ValueError("Final-PDB alignment changed after cohort freezing")
    pair_i, pair_j = strict_lower_triangle_indices(outer.n_residues)
    pair_block = np.load(pair_path)
    try:
        if pair_block.shape != (outer.n_residues, outer.n_residues, 128):
            raise ValueError(f"Pair tensor has wrong shape: {pair_block.shape}")
        features = extract_directional_base256(pair_block, pair_i, pair_j)
    finally:
        del pair_block
    if features.shape != (outer.n_pairs, FEATURE_DIM) or not np.all(np.isfinite(features)):
        raise ValueError("Extracted pair features are incomplete or nonfinite")
    return coordinates, chain_ids, pair_i, pair_j, features


def predict_ols(features: np.ndarray, fit: Mapping[str, object]) -> np.ndarray:
    values = np.asarray(features, dtype=np.float64)
    prediction = float(fit["intercept_standardized"]) + (
        (values - np.asarray(fit["feature_mean"], dtype=np.float64))
        / np.asarray(fit["feature_scale"], dtype=np.float64)
    ) @ np.asarray(fit["coefficient_standardized"], dtype=np.float64)
    if not np.all(np.isfinite(prediction)):
        raise ValueError("OLS replay produced nonfinite predictions")
    return prediction


def model_predictions(
    outer, chain_ids: Sequence[str], pair_i: np.ndarray, pair_j: np.ndarray,
    features: np.ndarray, artifacts: Mapping[str, object], fold_bundle,
) -> tuple[list[tuple[object, np.ndarray]], np.ndarray]:
    outer_fold = int(outer.outer_fold)
    predictions: list[tuple[object, np.ndarray]] = []
    shared = artifacts["shared_ols_base256"][outer_fold]
    predictions.append((FAMILY_BY_NAME["shared_ols_base256"], predict_ols(features, shared)))

    assignments = within_protein_pair_folds(outer.n_residues, outer.protein_id)
    pair_entry = fold_bundle.pair_by_protein[outer.protein_id]
    if (
        pair_fold_assignment_digest(outer.n_residues, outer.protein_id) != pair_entry.assignment_digest
        or tuple(np.bincount(assignments, minlength=5)) != tuple(pair_entry.fold_counts)
    ):
        raise ValueError("Regenerated pair folds disagree with the frozen manifest")
    shard, row_by_key = artifacts["per_protein_ols_base256"]
    per_models = {}
    for fold in range(N_PAIR_FOLDS):
        position = row_by_key[(outer.protein_id, fold)]
        per_models[fold] = (
            np.asarray(shard["coefficient_standardized"])[position],
            float(np.asarray(shard["intercept_standardized"])[position]),
            np.asarray(shard["feature_mean"])[position],
            np.asarray(shard["feature_scale"])[position],
        )
    predictions.append((
        FAMILY_BY_NAME["per_protein_ols_base256"],
        assemble_oof_predictions(features, assignments, per_models),
    ))

    booster = artifacts["shared_xgboost_base256_fixed_s4"][outer_fold]
    xgb_prediction = np.asarray(booster.inplace_predict(features), dtype=np.float64)
    if xgb_prediction.shape != (features.shape[0],) or not np.all(np.isfinite(xgb_prediction)):
        raise ValueError("XGBoost replay produced incomplete/nonfinite predictions")
    predictions.append((FAMILY_BY_NAME["shared_xgboost_base256_fixed_s4"], xgb_prediction))

    sequence_codes = sequence_separation_bin_codes(pair_i, pair_j, chain_ids=chain_ids)
    if set(np.unique(sequence_codes)) - set(range(len(SEQUENCE_BIN_LABELS))):
        raise ValueError("Every pair must route to exactly one canonical sequence gate")
    gated_prediction = np.full(features.shape[0], np.nan, dtype=np.float64)
    gate_writes = np.zeros(features.shape[0], dtype=np.uint8)
    for code, gate in enumerate(SEQUENCE_BIN_LABELS):
        selected = np.flatnonzero(sequence_codes == code)
        if selected.size == 0:
            continue
        gated_prediction[selected] = predict_ols(
            features[selected], artifacts["sequence_gated_ols_base256"][(outer_fold, gate)],
        )
        gate_writes[selected] += 1
    if not np.all(gate_writes == 1) or not np.all(np.isfinite(gated_prediction)):
        raise ValueError("Sequence-gated replay has missing/duplicate pair coverage")
    predictions.append((FAMILY_BY_NAME["sequence_gated_ols_base256"], gated_prediction))
    return predictions, assignments


def checkpoint_valid_rows(path: Path, layer: int) -> tuple[list[dict[str, str]], set[str]]:
    if not path.is_file():
        return [], set()
    rows = read_csv_strict(path, PER_PROTEIN_COLUMNS)
    grouped: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in rows:
        if int(row["layer"]) != layer:
            raise ValueError("Checkpoint contains the wrong layer")
        grouped[row["protein_id"]].append(row)
    complete = set()
    for protein_id, group in grouped.items():
        if len(group) != 103:
            raise ValueError(f"Checkpoint has partial protein group {protein_id}")
        by_family = defaultdict(list)
        for row in group:
            by_family[row["analysis_name"]].append(row)
        if set(by_family) != set(FAMILY_BY_NAME):
            raise ValueError("Checkpoint model-family coverage is incomplete")
        if len(by_family["per_protein_ols_base256"]) != 28 or any(
            len(by_family[name]) != 25 for name in by_family if name != "per_protein_ols_base256"
        ):
            raise ValueError("Checkpoint category coverage is incomplete")
        complete.add(protein_id)
    return rows, complete


def main() -> None:
    layer = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if layer not in LAYERS:
        raise ValueError(f"SLURM_ARRAY_TASK_ID must select layer 0..47; got {layer}")
    root = output_root()
    preparation = load_json(root / "run_manifest.json")
    if preparation.get("preparation_status") != "complete":
        raise ValueError("Reference preparation is not complete")
    fold_bundle = load_cv_folds(metadata_dir=producer_root() / "metadata")
    if len(fold_bundle.eligible_protein_ids) != EXPECTED_PROTEINS:
        raise ValueError("Frozen cohort size changed")
    thresholds = load_thresholds(root)
    layer_started_wall = utc_now()
    layer_started = time.perf_counter()
    shards = root / "layer_shards"
    shards.mkdir(parents=True, exist_ok=True)
    final_path = shards / f"per_protein_triangle_metrics_layer_{layer:02d}.csv"
    checkpoint_path = shards / f".per_protein_triangle_metrics_layer_{layer:02d}.checkpoint.csv"
    runtime_path = shards / f"runtime_layer_{layer:02d}.csv"
    if final_path.is_file():
        existing_rows, existing_ids = checkpoint_valid_rows(final_path, layer)
        if (
            len(existing_rows) != 56_856
            or existing_ids != set(fold_bundle.eligible_protein_ids)
        ):
            raise ValueError(f"Existing final layer shard is incomplete: {final_path}")
        log(f"layer={layer} valid final shard already exists")
        return
    artifacts, provenance = load_layer_artifacts(layer, fold_bundle)
    atomic_write_json(shards / f"artifact_provenance_layer_{layer:02d}.json", provenance)
    rows, complete_ids = checkpoint_valid_rows(checkpoint_path, layer)
    runtime_rows = read_csv_strict(runtime_path, RUNTIME_COLUMNS) if runtime_path.is_file() else []
    failures: list[str] = []
    checkpoint_every = int(os.environ.get("TRIANGLE_CHECKPOINT_PROTEINS", "5"))
    temp_parent = Path(os.environ.get("TRIANGLE_TMPDIR", os.environ.get("TMPDIR", "/tmp")))

    for number, protein_id in enumerate(fold_bundle.eligible_protein_ids, start=1):
        if protein_id in complete_ids:
            continue
        outer = fold_bundle.outer_by_protein[protein_id]
        protein_started_wall = utc_now()
        protein_started = time.perf_counter()
        try:
            completed_protein_rows: list[dict[str, object]] = []
            coordinates, chain_ids, pair_i, pair_j, features = load_protein(outer, layer)
            predictions, assignments = model_predictions(
                outer, chain_ids, pair_i, pair_j, features, artifacts, fold_bundle,
            )
            pair_fold_matrix = symmetric_pair_matrix(
                outer.n_residues, pair_i, pair_j, assignments, diagonal=-1,
            )
            with tempfile.TemporaryDirectory(
                prefix=f"triangle_replay_l{layer:02d}_{protein_id}_", dir=temp_parent,
            ) as temporary:
                cache = None
                try:
                    cache, reference = build_reference_cache(
                        coordinates, temporary, thresholds=thresholds[protein_id],
                    )
                    if reference["reference_digest"] != thresholds[protein_id]["reference_digest"]:
                        raise ValueError("Reference geometry digest changed since preparation")
                    for spec, prediction in predictions:
                        model_started = time.perf_counter()
                        identity = {
                            "analysis_name": spec.analysis_name, "model_family": spec.model_family,
                            "feature_dim": FEATURE_DIM, "layer": layer, "protein_id": protein_id,
                            "source": outer.source, "cv_protocol_version": CV_PROTOCOL_VERSION,
                            "evaluation_design": spec.evaluation_design,
                            "outer_fold": outer.outer_fold,
                            "oof_prediction_scope": spec.oof_prediction_scope,
                            "generalization_scope": spec.generalization_scope,
                            "n_residues": outer.n_residues,
                            "protein_length_bin": f"length_bin_{int(outer.length_bin) + 1}",
                        }
                        matrix = symmetric_pair_matrix(
                            outer.n_residues, pair_i, pair_j, prediction,
                        )
                        completed_protein_rows.extend(compute_metric_rows(
                            matrix, cache, identity,
                            pair_fold_matrix=pair_fold_matrix if spec.uses_pair_folds else None,
                        ))
                        runtime_rows.append({
                            "stage": f"replay_model:{spec.analysis_name}", "layer": layer,
                            "protein_id": protein_id, "started_at": protein_started_wall,
                            "finished_at": utc_now(),
                            "elapsed_seconds": time.perf_counter() - model_started,
                            "peak_rss_mb": peak_rss_mb(), "status": "complete", "detail": "",
                        })
                finally:
                    if cache is not None:
                        cache.cleanup()
            if len(completed_protein_rows) != 103:
                raise RuntimeError("Completed protein did not produce exactly 103 rows")
            rows.extend(completed_protein_rows)
            complete_ids.add(protein_id)
            status = "complete"
        except Exception as exc:
            status = "failed"
            failures.append(f"{protein_id}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
            runtime_rows.append({
                "stage": "replay_protein", "layer": layer, "protein_id": protein_id,
                "started_at": protein_started_wall, "finished_at": utc_now(),
                "elapsed_seconds": time.perf_counter() - protein_started,
                "peak_rss_mb": peak_rss_mb(), "status": "failed",
                "detail": f"{type(exc).__name__}: {exc}",
            })
        log(
            f"layer={layer} progress={number}/{EXPECTED_PROTEINS} protein={protein_id} "
            f"status={status} elapsed={time.perf_counter() - protein_started:.1f}s"
        )
        if number % checkpoint_every == 0 or status == "failed" or number == EXPECTED_PROTEINS:
            atomic_write_csv(checkpoint_path, PER_PROTEIN_COLUMNS, rows)
            atomic_write_csv(runtime_path, RUNTIME_COLUMNS, runtime_rows)
        if "features" in locals():
            del features
        gc.collect()

    if failures or complete_ids != set(fold_bundle.eligible_protein_ids):
        raise RuntimeError(f"Layer {layer} incomplete; failures={failures}")
    expected_rows = EXPECTED_PROTEINS * (4 * sum(map(len, CATEGORY_LABELS.values())) + 3)
    if len(rows) != expected_rows or expected_rows != 56_856:
        raise RuntimeError(f"Layer {layer} has {len(rows)} rows, expected 56,856")
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

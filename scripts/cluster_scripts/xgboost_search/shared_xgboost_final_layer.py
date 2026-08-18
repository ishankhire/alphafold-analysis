#!/usr/bin/env python3
"""Fit and evaluate one canonical final shared-XGBoost layer."""

from __future__ import annotations

import json
import os
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
GENERALIZED_DIR = CLUSTER_DIR / "generalized_model_ecod"
for _path in (SCRIPT_DIR, CLUSTER_DIR, GENERALIZED_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from regression_output_helpers import build_model_status_row, build_runtime_row, sha256_file
from regression_protocol import FEATURE_DIM, MASTER_MANIFEST_DIGEST, protein_equal_mean
from shared_xgboost_final_common import (
    ANALYSIS_NAME,
    BOOSTING_ROUNDS,
    CONFIGURATION_ID,
    MODEL_FAMILY,
    RANDOM_SEED,
    ROW_CAP,
    atomic_save_booster,
    atomic_write_json,
    build_final_estimator,
    build_training_sample_manifest,
    code_digest,
    evaluate_records,
    final_output_root,
    fit_final_estimator,
    load_combined_records,
    load_final_configuration,
    load_pair_features,
    log,
    materialize_development_matrix,
    peak_rss_mb,
    split_assignment_rows,
    utc_now,
    validate_layer_task_id,
    validate_frozen_layer_input_paths,
    version_payload,
    working_output_root,
    write_failure,
    write_layer_shard,
)
from xgboost_search_common import build_shared_split


def _checkpoint_invariants(config, split_digest, sample_digest, input_digest, versions):
    return {
        "configuration_digest": config["configuration_digest"],
        "master_manifest_digest": MASTER_MANIFEST_DIGEST,
        "split_assignment_digest": split_digest,
        "sample_policy_digest": sample_digest,
        "layer_input_digest": input_digest,
        "code_digest": code_digest(),
        "xgboost_version": versions["xgboost"],
    }


def _fit_or_resume(root, layer, config, bundle, invariants, n_jobs):
    checkpoint_dir = root / "fit_checkpoints" / f"layer_{layer:02d}"
    booster_path = checkpoint_dir / f"booster_layer_{layer:02d}.json"
    manifest_path = checkpoint_dir / "fit_checkpoint.json"
    model = build_final_estimator(config, n_jobs)
    if booster_path.is_file() and manifest_path.is_file():
        manifest = json.loads(manifest_path.read_text())
        if all(manifest.get(key) == value for key, value in invariants.items()) and manifest.get("booster_sha256") == sha256_file(booster_path):
            model.load_model(str(booster_path))
            log(f"Layer {layer}: reused matching fit checkpoint")
            return model, True
    log(f"Layer {layer}: fitting S4 for exactly {BOOSTING_ROUNDS} rounds")
    fit_final_estimator(model, bundle)
    atomic_save_booster(model, booster_path)
    atomic_write_json(manifest_path, {
        **invariants,
        "layer": layer,
        "status": "fitted",
        "booster_sha256": sha256_file(booster_path),
        "fitted_at": utc_now(),
    })
    return model, False


def run(task_id: int | None = None) -> Path:
    started = time.perf_counter()
    phase = "configuration_validation"
    layer = validate_layer_task_id(
        int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if task_id is None else task_id
    )
    root = final_output_root()
    work_root = working_output_root()
    config = {}
    try:
        # Frozen selection is validated before any pair matrix is opened.
        config = load_final_configuration()
        split_rows, split_digest = split_assignment_rows()
        if split_digest != config["split_assignment_digest"]:
            raise ValueError("Frozen split-assignment digest does not match the canonical manifest")
        sample_rows, sample_digest, indices_by_id = build_training_sample_manifest()
        if len(sample_rows) != 441:
            raise ValueError(f"Expected 441 development sampling rows; got {len(sample_rows)}")
        phase = "layer_input_path_validation"
        input_digest = validate_frozen_layer_input_paths(layer)

        phase = "development_input_loading"
        development_records, skipped = load_combined_records(
            (layer,), pair_order="lower", master_roles=("fit", "validation"),
        )
        development_parts, _ = build_shared_split(
            development_records, required_roles=("fit", "validation"),
        )
        development_records = sorted(
            (*development_parts["fit"], *development_parts["validation"]),
            key=lambda record: str(record["protein_id"]),
        )
        if skipped or len(development_records) != 441:
            raise RuntimeError(f"Incomplete frozen development cohort: n={len(development_records)}, skipped={skipped}")
        global_mean = protein_equal_mean(
            np.asarray(record["y"], dtype=np.float64) for record in development_records
        )

        phase = "development_materialization"
        bundle = materialize_development_matrix(development_records, layer, indices_by_id)
        if bundle["x"].dtype != np.float64 or bundle["y"].dtype != np.float64 or bundle["weights"].dtype != np.float64:
            raise AssertionError("All final XGBoost fit arrays must be float64")

        phase = "booster_fit"
        versions = version_payload()
        invariants = _checkpoint_invariants(
            config, split_digest, sample_digest, input_digest, versions,
        )
        model, reused = _fit_or_resume(
            work_root, layer, config, bundle, invariants,
            int(os.environ.get("SLURM_CPUS_PER_TASK", "1")),
        )
        replay_record, replay_indices, replay_start, replay_stop = bundle["offsets"][0]
        replay_features = np.asarray(bundle["x"][replay_start:min(replay_stop, replay_start + 32)], dtype=np.float64)
        replay_predictions = np.asarray(model.predict(replay_features), dtype=np.float64)
        replay = {
            "protein_id": str(replay_record["protein_id"]),
            "pair_indices": np.asarray(replay_indices[: replay_features.shape[0]], dtype=np.int64).tolist(),
            "feature_sha256": __import__("hashlib").sha256(replay_features.tobytes()).hexdigest(),
            "predictions": replay_predictions.tolist(),
            "absolute_tolerance": 1e-10,
        }

        if os.environ.get("XGBOOST_FINAL_DEVELOPMENT_SMOKE") == "1":
            phase = "development_smoke_write"
            smoke_path = root / f"layer_{layer:02d}_development_smoke.json"
            atomic_write_json(smoke_path, {
                **invariants,
                "layer": layer,
                "status": "development_smoke_complete",
                "locked_test_loaded": False,
                "n_development_proteins": len(development_records),
                "feature_shape": list(bundle["x"].shape),
                "feature_dtype": str(bundle["x"].dtype),
                "target_dtype": str(bundle["y"].dtype),
                "weight_dtype": str(bundle["weights"].dtype),
                "n_fitting_rows": int(bundle["y"].size),
                "checkpoint_reused": reused,
                "replay": replay,
                "peak_rss_mb": peak_rss_mb(),
                "elapsed_seconds": time.perf_counter() - started,
                "completed_at": utc_now(),
            })
            log(f"Layer {layer}: development-only smoke complete at {smoke_path}")
            return smoke_path

        # Locked-test targets and pair blocks are first opened only after the
        # final configuration is frozen and the final booster exists.
        phase = "test_input_loading"
        test_records, test_skipped = load_combined_records(
            (layer,), pair_order="lower", master_roles=("test",),
        )
        test_parts, _ = build_shared_split(test_records, required_roles=("test",))
        test_records = test_parts["test"]
        if test_skipped or len(test_records) != 111:
            raise RuntimeError(f"Incomplete frozen test cohort: n={len(test_records)}, skipped={test_skipped}")

        phase = "all_protein_evaluation"
        all_records = sorted((*development_records, *test_records), key=lambda record: str(record["protein_id"]))
        tables = evaluate_records(model, all_records, layer, global_mean)
        status = build_model_status_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            feature_dim=FEATURE_DIM, layer=layer, fit_status="complete",
            support_flags="fit_checkpoint_reused" if reused else "",
            n_training_proteins=441, n_training_pairs=int(bundle["y"].size),
            selected_configuration_id=CONFIGURATION_ID,
            final_boosting_rounds=BOOSTING_ROUNDS,
            training_row_cap=ROW_CAP, random_seed=RANDOM_SEED,
        )
        runtime = build_runtime_row(
            analysis_name=ANALYSIS_NAME, model_family=MODEL_FAMILY,
            layer=layer, completion_status="complete", n_fitted_proteins=441,
            n_evaluated_proteins=552, n_fitted_pairs=int(bundle["y"].size),
            n_evaluated_pairs=sum(int(row["n_evaluated_pairs"]) for row in tables["per_protein.csv"]),
            elapsed_seconds=time.perf_counter() - started, peak_rss_mb=peak_rss_mb(),
        )
        phase = "canonical_shard_write"
        shard = write_layer_shard(
            root, layer, tables, model, status, runtime,
            {
                **invariants,
                "protocol_version": "1.0",
                "output_schema_version": "1.0",
                "global_development_mean": global_mean,
                "versions": versions,
            },
            replay,
        )
        log(f"Layer {layer}: canonical shard complete at {shard}")
        return shard
    except Exception as exc:
        write_failure(work_root, layer, phase, exc, config)
        raise


if __name__ == "__main__":
    run()

#!/usr/bin/env python3
"""Array entrypoint for the three-layer shared XGBoost search."""

import gc
import os
import time
from pathlib import Path

import numpy as np

from xgboost_search_common import (
    ANCHOR_LAYERS, EARLY_STOPPING_ROUNDS, FIXED_XGBOOST_PARAMETERS, N_CHANNELS, N_ESTIMATORS,
    SEARCH_CAP, SHARED_CONFIGS, build_shared_split, cleanup_bundle,
    evaluate_model_records, fit_xgboost, load_combined_records, log,
    map_search_task, materialize_records, n_jobs_from_environment,
    output_root, shared_search_dir, split_digest, summarize_prediction_bundle,
    timestamp, write_csv, write_json, write_summary,
)


def run(task_id: int) -> None:
    layer = map_search_task(task_id)
    layer_dir = shared_search_dir() / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    task_start = time.perf_counter()
    log(f"Shared XGBoost search task {task_id}: layer {layer}")
    log(f"Output root: {output_root()}")

    records, skipped = load_combined_records(ANCHOR_LAYERS)
    split, assignments = build_shared_split(records)
    digest = split_digest(assignments)
    write_csv(layer_dir / "split_assignments.csv", assignments)
    write_csv(layer_dir / "skipped_proteins.csv", skipped, ["protein_id", "source", "reason"])
    write_json(layer_dir / "configurations.json", list(SHARED_CONFIGS))
    metadata = {
        "pipeline": "shared_search", "layer": layer, "task_id": task_id,
        "anchor_layers": list(ANCHOR_LAYERS), "split_digest": digest,
        "feature_definition": "float64 0.5*(pair_block[i,j,:]+pair_block[j,i,:])",
        "feature_count": N_CHANNELS, "standardized": False,
        "fit_cap_per_protein": SEARCH_CAP, "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "fixed_xgboost_parameters": FIXED_XGBOOST_PARAMETERS,
        "selection_metric": "validation_mean_per_protein_r2",
        "n_valid_proteins": len(records), "n_skipped_proteins": len(skipped),
        "status": "running", "started_at": timestamp(),
    }
    write_json(layer_dir / "search_metadata.json", metadata)

    log("Materializing the nested 10k/protein fit matrix and uncapped validation matrix")
    fit_bundle = materialize_records(split["fit"], layer, cap=SEARCH_CAP)
    validation_bundle = materialize_records(split["validation"], layer)
    matrix_materialization_seconds = fit_bundle["materialization_seconds"] + validation_bundle["materialization_seconds"]
    rows = []
    for config in SHARED_CONFIGS:
        config_start = time.perf_counter()
        config_id = config["config_id"]
        log(f"Fitting {config_id}: depth={config['max_depth']}, eta={config['learning_rate']}, min_child_weight={config['min_child_weight']}")
        model, fit_info = fit_xgboost(config, fit_bundle, validation_bundle, n_jobs_from_environment())

        tick = time.perf_counter()
        validation_pred = np.asarray(model.predict(validation_bundle["x"]), dtype=np.float64)
        validation_prediction_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        _, validation_metrics = summarize_prediction_bundle(validation_bundle, validation_pred)
        validation_metric_seconds = time.perf_counter() - tick
        del validation_pred

        _, test_metrics, _, test_times = evaluate_model_records(model, split["test"], layer, "test")
        row = {
            **config, "layer": layer,
            "validation_mean_per_protein_r2": validation_metrics["mean_per_protein_r2"],
            "validation_pooled_r2": validation_metrics["pooled_r2"],
            "test_mean_per_protein_r2": test_metrics["mean_per_protein_r2"],
            "test_pooled_r2": test_metrics["pooled_r2"],
            "best_iteration": fit_info["best_iteration"],
            "best_validation_rmse": fit_info["best_validation_rmse"],
            "n_fit_rows": int(fit_bundle["y"].size),
            "n_fit_proteins": len(split["fit"]),
            "n_validation_rows": int(validation_bundle["y"].size),
            "n_validation_proteins": len(split["validation"]),
            "n_test_rows": int(test_metrics["n_pairs"]),
            "n_test_proteins": len(split["test"]),
            "feature_materialization_seconds": matrix_materialization_seconds,
            "fit_seconds": fit_info["fit_seconds"],
            "validation_prediction_seconds": validation_prediction_seconds,
            "test_materialization_seconds": test_times["materialization_seconds"],
            "test_prediction_seconds": test_times["prediction_seconds"],
            "metric_seconds": validation_metric_seconds + test_times["metric_seconds"],
            "total_elapsed_seconds": matrix_materialization_seconds + time.perf_counter() - config_start,
            "split_digest": digest,
        }
        rows.append(row)
        write_csv(layer_dir / "search_summary.csv", rows)
        write_summary(
            layer_dir / f"summary_{datetime_label()}.txt",
            f"Shared XGBoost hyperparameter search — layer {layer}",
            [f"Completed configurations: {len(rows)}/{len(SHARED_CONFIGS)}", f"Latest configuration: {config_id}", f"Split digest: {digest}"],
        )
        del model
        gc.collect()
        log(f"Checkpointed {config_id}; validation mean protein R2={row['validation_mean_per_protein_r2']:.6f}")

    metadata.update({"status": "complete", "completed_at": timestamp(), "task_elapsed_seconds": time.perf_counter() - task_start})
    write_json(layer_dir / "search_metadata.json", metadata)
    write_summary(
        layer_dir / "summary.txt", f"Shared XGBoost hyperparameter search — layer {layer}",
        [f"Completed configurations: {len(rows)}", f"Split digest: {digest}", f"Task elapsed seconds: {metadata['task_elapsed_seconds']:.3f}"],
    )
    cleanup_bundle(fit_bundle)
    cleanup_bundle(validation_bundle)
    log(f"Shared layer {layer} search complete")


def datetime_label() -> str:
    return time.strftime("%Y%m%d_%H%M%S")


if __name__ == "__main__":
    run(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))

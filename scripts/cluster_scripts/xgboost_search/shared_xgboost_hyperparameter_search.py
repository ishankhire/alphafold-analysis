#!/usr/bin/env python3
"""Array entrypoint for the three-layer shared XGBoost search."""

import gc
import os
import time
from pathlib import Path

import numpy as np

from xgboost_search_common import (
    ANCHOR_LAYERS, DIRECTIONAL_BASE256_FEATURES, EARLY_STOPPING_ROUNDS,
    FEATURE_DIM, FIXED_XGBOOST_PARAMETERS, N_ESTIMATORS, SEARCH_CAP,
    SHARED_CONFIGS, SHARED_WEIGHTING_MODE, build_shared_split, cleanup_bundle,
    fit_xgboost, load_combined_records, log,
    map_search_task, materialize_records, n_jobs_from_environment,
    output_root, remove_obsolete_outputs, shared_search_dir, split_digest,
    summarize_prediction_bundle,
    timestamp, write_csv, write_json, write_summary,
)


def run(task_id: int) -> None:
    layer = map_search_task(task_id)
    layer_dir = shared_search_dir() / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    remove_obsolete_outputs(layer_dir, glob_patterns=("summary_*.txt",))
    task_start = time.perf_counter()
    log(f"Shared XGBoost search task {task_id}: layer {layer}")
    log(f"Output root: {output_root()}")

    records, skipped = load_combined_records(
        ANCHOR_LAYERS, pair_order="lower", master_roles=("fit", "validation"),
    )
    split, assignments = build_shared_split(
        records, required_roles=("fit", "validation"),
    )
    digest = split_digest(assignments)
    write_csv(layer_dir / "split_assignments.csv", assignments)
    write_csv(layer_dir / "skipped_proteins.csv", skipped, ["protein_id", "source", "reason"])
    write_json(layer_dir / "configurations.json", list(SHARED_CONFIGS))
    metadata = {
        "pipeline": "shared_search", "layer": layer, "task_id": task_id,
        "anchor_layers": list(ANCHOR_LAYERS), "split_digest": digest,
        "feature_definition": "float64 [pair_block[i,j,:] | pair_block[j,i,:]], i>j",
        "feature_count": FEATURE_DIM, "standardized": False,
        "fit_cap_per_protein": SEARCH_CAP, "n_estimators": N_ESTIMATORS,
        "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
        "fixed_xgboost_parameters": FIXED_XGBOOST_PARAMETERS,
        "weighting_mode": SHARED_WEIGHTING_MODE,
        "selection_metric": "validation_mean_per_protein_r2",
        "n_valid_proteins": len(records), "n_skipped_proteins": len(skipped),
        "status": "running", "started_at": timestamp(),
    }
    write_json(layer_dir / "search_metadata.json", metadata)

    log("Materializing the nested 10k/protein fit matrix and uncapped validation matrix")
    fit_bundle = materialize_records(
        split["fit"], layer, cap=SEARCH_CAP,
        feature_mode=DIRECTIONAL_BASE256_FEATURES,
    )
    validation_bundle = materialize_records(
        split["validation"], layer,
        feature_mode=DIRECTIONAL_BASE256_FEATURES,
    )
    matrix_materialization_seconds = fit_bundle["materialization_seconds"] + validation_bundle["materialization_seconds"]
    rows = []
    for config in SHARED_CONFIGS:
        config_start = time.perf_counter()
        config_id = config["config_id"]
        log(f"Fitting {config_id}: depth={config['max_depth']}, eta={config['learning_rate']}, min_child_weight={config['min_child_weight']}")
        model, fit_info = fit_xgboost(
            config, fit_bundle, validation_bundle, n_jobs_from_environment(),
            SHARED_WEIGHTING_MODE,
        )

        tick = time.perf_counter()
        train_pred = np.asarray(model.predict(fit_bundle["x"]), dtype=np.float64)
        train_prediction_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        _, train_metrics = summarize_prediction_bundle(
            fit_bundle, train_pred, "fit", "sampled_training_rows",
        )
        train_metric_seconds = time.perf_counter() - tick
        del train_pred

        tick = time.perf_counter()
        validation_pred = np.asarray(model.predict(validation_bundle["x"]), dtype=np.float64)
        validation_prediction_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        _, validation_metrics = summarize_prediction_bundle(
            validation_bundle, validation_pred, "validation", "full_protein_pairs",
        )
        validation_metric_seconds = time.perf_counter() - tick
        del validation_pred

        row = {
            **config, "layer": layer,
            "train_mean_per_protein_r2": train_metrics["per_protein_r2_mean"],
            "train_pooled_r2": train_metrics["pair_weighted_r2"],
            "validation_mean_per_protein_r2": validation_metrics["per_protein_r2_mean"],
            "validation_pooled_r2": validation_metrics["pair_weighted_r2"],
            "best_iteration": fit_info["best_iteration"],
            "best_validation_rmse": fit_info["best_validation_rmse"],
            "n_train_rows": int(fit_bundle["y"].size),
            "n_train_proteins": len(split["fit"]),
            "n_validation_rows": int(validation_bundle["y"].size),
            "n_validation_proteins": len(split["validation"]),
            "feature_materialization_seconds": matrix_materialization_seconds,
            "fit_seconds": fit_info["fit_seconds"],
            "train_prediction_seconds": train_prediction_seconds,
            "validation_prediction_seconds": validation_prediction_seconds,
            "metric_seconds": train_metric_seconds + validation_metric_seconds,
            "total_elapsed_seconds": matrix_materialization_seconds + time.perf_counter() - config_start,
            "split_digest": digest,
        }
        rows.append(row)
        write_csv(layer_dir / "search_summary.csv", rows)
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
if __name__ == "__main__":
    run(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))

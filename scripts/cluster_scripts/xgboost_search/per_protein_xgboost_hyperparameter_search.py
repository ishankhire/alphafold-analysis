#!/usr/bin/env python3
"""Array entrypoint for the 50-protein XGBoost hyperparameter search."""

import gc
import os
import time

import numpy as np

from xgboost_search_common import (
    ANCHOR_LAYERS, FIXED_XGBOOST_PARAMETERS, PER_PROTEIN_CONFIGS, PooledStats, attach_pair_splits,
    cleanup_bundle, fit_xgboost, load_combined_records, log, map_search_task,
    materialize_records, metric_row, n_jobs_from_environment,
    pair_split_summary, per_protein_search_dir, select_per_protein_manifest,
    timestamp, write_csv, write_json, write_summary,
)


def run(task_id: int) -> None:
    layer = map_search_task(task_id)
    layer_dir = per_protein_search_dir() / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    task_start = time.perf_counter()
    log(f"Per-protein XGBoost search task {task_id}: layer {layer}")

    records, skipped = load_combined_records(ANCHOR_LAYERS)
    selected_records = select_per_protein_manifest(records)
    if len(selected_records) != 50:
        raise RuntimeError(f"Expected exactly 50 per-protein search proteins; found {len(selected_records)}.")
    attach_pair_splits(selected_records)
    split_rows, digest = pair_split_summary(selected_records)
    manifest_rows = [{
        "protein_id": r["protein_id"], "source": r["source"],
        "n_residues": r["n_residues"], "n_pairs": r["n_pairs"], "size_bin": r["size_bin"],
    } for r in selected_records]
    write_csv(layer_dir / "protein_manifest.csv", manifest_rows)
    write_csv(layer_dir / "pair_split_summary.csv", split_rows)
    write_csv(layer_dir / "skipped_proteins.csv", skipped, ["protein_id", "source", "reason"])
    write_json(layer_dir / "configurations.json", list(PER_PROTEIN_CONFIGS))
    metadata = {
        "pipeline": "per_protein_search", "layer": layer, "task_id": task_id,
        "anchor_layers": list(ANCHOR_LAYERS), "split_digest": digest,
        "n_proteins": 50, "proteins_per_size_bin": 10,
        "pair_split": "deterministic 60/20/20", "fit_pair_cap": None,
        "feature_definition": "float64 0.5*(pair_block[i,j,:]+pair_block[j,i,:])",
        "selection_metric": "validation_mean_per_protein_r2",
        "fixed_xgboost_parameters": FIXED_XGBOOST_PARAMETERS,
        "status": "running", "started_at": timestamp(),
    }
    write_json(layer_dir / "search_metadata.json", metadata)

    summary_rows, timing_rows = [], []
    for config in PER_PROTEIN_CONFIGS:
        config_start = time.perf_counter()
        config_id = config["config_id"]
        log(f"Starting {config_id} across all 50 proteins")
        validation_pool, test_pool = PooledStats(), PooledStats()
        validation_r2s, test_r2s, best_iterations, best_scores = [], [], [], []
        config_timing_rows = []
        for protein_number, record in enumerate(selected_records, start=1):
            protein_start = time.perf_counter()
            fit_bundle = materialize_records([record], layer, split_name="fit")
            validation_bundle = materialize_records([record], layer, split_name="validation")
            test_bundle = materialize_records([record], layer, split_name="test")
            model, fit_info = fit_xgboost(config, fit_bundle, validation_bundle, n_jobs_from_environment())
            tick = time.perf_counter()
            val_pred = np.asarray(model.predict(validation_bundle["x"]), dtype=np.float64)
            validation_prediction_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            test_pred = np.asarray(model.predict(test_bundle["x"]), dtype=np.float64)
            test_prediction_seconds = time.perf_counter() - tick
            tick = time.perf_counter()
            val_metrics = metric_row(validation_bundle["y"], val_pred)
            test_metrics = metric_row(test_bundle["y"], test_pred)
            validation_pool.update(validation_bundle["y"], val_pred)
            test_pool.update(test_bundle["y"], test_pred)
            metric_seconds = time.perf_counter() - tick
            validation_r2s.append(val_metrics["r2"])
            test_r2s.append(test_metrics["r2"])
            best_iterations.append(fit_info["best_iteration"])
            best_scores.append(fit_info["best_validation_rmse"])
            config_timing_rows.append({
                "layer": layer, "config_id": config_id, "protein_id": record["protein_id"],
                "source": record["source"], "size_bin": record["size_bin"],
                "n_fit_rows": int(fit_bundle["y"].size),
                "n_validation_rows": int(validation_bundle["y"].size),
                "n_test_rows": int(test_bundle["y"].size),
                "fit_materialization_seconds": fit_bundle["materialization_seconds"],
                "validation_materialization_seconds": validation_bundle["materialization_seconds"],
                "test_materialization_seconds": test_bundle["materialization_seconds"],
                "fit_seconds": fit_info["fit_seconds"],
                "validation_prediction_seconds": validation_prediction_seconds,
                "test_prediction_seconds": test_prediction_seconds,
                "metric_seconds": metric_seconds,
                "best_iteration": fit_info["best_iteration"],
                "best_validation_rmse": fit_info["best_validation_rmse"],
                "total_seconds": time.perf_counter() - protein_start,
            })
            del model, val_pred, test_pred
            cleanup_bundle(fit_bundle)
            cleanup_bundle(validation_bundle)
            cleanup_bundle(test_bundle)
            gc.collect()
            if protein_number % 10 == 0:
                log(f"{config_id}: completed {protein_number}/50 proteins")

        if len(config_timing_rows) != 50:
            raise RuntimeError(f"Configuration {config_id} completed only {len(config_timing_rows)}/50 proteins.")
        timing_rows.extend(config_timing_rows)
        val_pooled, test_pooled = validation_pool.metrics(), test_pool.metrics()
        summary_rows.append({
            **config, "layer": layer, "n_proteins": 50,
            "validation_mean_per_protein_r2": float(np.mean(validation_r2s)),
            "validation_pooled_r2": val_pooled["r2"],
            "test_mean_per_protein_r2": float(np.mean(test_r2s)),
            "test_pooled_r2": test_pooled["r2"],
            "best_iteration_mean": float(np.mean(best_iterations)),
            "best_iteration_median": float(np.median(best_iterations)),
            "best_iteration_min": int(np.min(best_iterations)),
            "best_iteration_max": int(np.max(best_iterations)),
            "best_validation_rmse_mean": float(np.mean(best_scores)),
            "materialization_seconds": float(sum(r["fit_materialization_seconds"] + r["validation_materialization_seconds"] + r["test_materialization_seconds"] for r in config_timing_rows)),
            "fit_seconds": float(sum(r["fit_seconds"] for r in config_timing_rows)),
            "validation_prediction_seconds": float(sum(r["validation_prediction_seconds"] for r in config_timing_rows)),
            "test_prediction_seconds": float(sum(r["test_prediction_seconds"] for r in config_timing_rows)),
            "metric_seconds": float(sum(r["metric_seconds"] for r in config_timing_rows)),
            "total_elapsed_seconds": time.perf_counter() - config_start,
            "split_digest": digest,
        })
        write_csv(layer_dir / "search_summary.csv", summary_rows)
        write_csv(layer_dir / "per_protein_timing.csv", timing_rows)
        write_summary(
            layer_dir / "summary.txt", f"Per-protein XGBoost search — layer {layer}",
            [f"Completed configurations: {len(summary_rows)}/{len(PER_PROTEIN_CONFIGS)}", f"Latest configuration: {config_id}", f"Split digest: {digest}"],
        )
        log(f"Checkpointed {config_id}; validation mean protein R2={summary_rows[-1]['validation_mean_per_protein_r2']:.6f}")

    if len(summary_rows) != len(PER_PROTEIN_CONFIGS) or any(sum(r["config_id"] == c["config_id"] for r in timing_rows) != 50 for c in PER_PROTEIN_CONFIGS):
        raise RuntimeError("Refusing to complete an incomplete 50-protein search artifact.")
    metadata.update({"status": "complete", "completed_at": timestamp(), "task_elapsed_seconds": time.perf_counter() - task_start})
    write_json(layer_dir / "search_metadata.json", metadata)
    log(f"Per-protein layer {layer} search complete")


if __name__ == "__main__":
    run(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))

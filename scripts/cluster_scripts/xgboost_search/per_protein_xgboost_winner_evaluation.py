#!/usr/bin/env python3
"""Rerun and evaluate the globally selected per-protein XGBoost config."""

import gc
import json
import os
import time

import numpy as np

from xgboost_search_common import (
    ANCHOR_LAYERS, PER_PROTEIN_CONFIGS, PooledStats, attach_pair_splits,
    cleanup_bundle, fit_xgboost, geometry_metrics, importance_rows,
    load_combined_records, load_complete_search, load_pair_features, log,
    map_winner_layer_task, materialize_records, metric_row,
    n_jobs_from_environment, pair_split_summary, per_protein_search_dir,
    per_protein_winner_dir, select_per_protein_manifest, select_winner,
    summarize_geometry, timestamp, write_csv, write_json, write_summary,
)


def run(task_id: int) -> None:
    layer = map_winner_layer_task(task_id)
    task_dir = per_protein_winner_dir() / f"layer_{layer:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_start = time.perf_counter()
    log(f"Per-protein winner task {task_id}: layer={layer}")

    # Fail on missing/inconsistent artifacts before any fit can start.
    search_rows, expected_digest = load_complete_search(per_protein_search_dir(), PER_PROTEIN_CONFIGS)
    selected, selection_table, provenance = select_winner(search_rows, PER_PROTEIN_CONFIGS)
    write_json(task_dir / "selection_provenance.json", provenance)
    write_csv(task_dir / "cross_layer_selection_table.csv", selection_table)
    canonical_path = per_protein_winner_dir() / "selected_config.json"
    if task_id == 0:
        write_json(canonical_path, {**provenance, "cross_layer_table": selection_table})
    elif canonical_path.exists():
        canonical = json.loads(canonical_path.read_text())
        if canonical["selected_config"]["config_id"] != selected["config_id"]:
            raise RuntimeError("This task derived a different per-protein winner than canonical task 0.")

    records, skipped = load_combined_records(ANCHOR_LAYERS)
    selected_records = select_per_protein_manifest(records)
    attach_pair_splits(selected_records)
    split_rows, observed_digest = pair_split_summary(selected_records)
    if observed_digest != expected_digest:
        raise RuntimeError(f"Current pair split {observed_digest} differs from search split {expected_digest}.")
    write_csv(task_dir / "pair_split_summary.csv", split_rows)
    write_csv(task_dir / "skipped_proteins.csv", skipped, ["protein_id", "source", "reason"])

    regression_rows, geometry_rows, timing_rows, all_importance = [], [], [], []
    pooled = {name: PooledStats() for name in ("fit", "validation", "test")}
    r2_values = {name: [] for name in pooled}
    mae_values = {name: [] for name in pooled}
    rmse_values = {name: [] for name in pooled}
    for protein_number, record in enumerate(selected_records, start=1):
        protein_start = time.perf_counter()
        bundles = {name: materialize_records([record], layer, split_name=name) for name in pooled}
        model, fit_info = fit_xgboost(selected, bundles["fit"], bundles["validation"], n_jobs_from_environment())
        prediction_seconds = {}
        for split_name, bundle in bundles.items():
            tick = time.perf_counter()
            pred = np.asarray(model.predict(bundle["x"]), dtype=np.float64)
            prediction_seconds[f"{split_name}_prediction_seconds"] = time.perf_counter() - tick
            metrics = metric_row(bundle["y"], pred)
            pooled[split_name].update(bundle["y"], pred)
            r2_values[split_name].append(metrics["r2"])
            mae_values[split_name].append(metrics["mae"])
            rmse_values[split_name].append(metrics["rmse"])
            regression_rows.append({
                "layer": layer, "config_id": selected["config_id"], "split": split_name,
                "protein_id": record["protein_id"], "source": record["source"],
                "size_bin": record["size_bin"], "n_residues": record["n_residues"],
                "n_pairs": int(bundle["y"].size), **metrics,
            })
            del pred

        tick = time.perf_counter()
        all_x = load_pair_features(record, layer)
        all_pair_materialization_seconds = time.perf_counter() - tick
        tick = time.perf_counter()
        all_pred = np.asarray(model.predict(all_x), dtype=np.float64)
        all_pair_prediction_seconds = time.perf_counter() - tick
        del all_x
        tick = time.perf_counter()
        geometry_rows.append({
            "layer": layer, "config_id": selected["config_id"], "split": "all_pairs",
            "geometry_prediction_scope": "all_pairs_model_predictions",
            "regression_test_scope": "held_out_pair_predictions",
            "protein_id": record["protein_id"], "source": record["source"],
            "size_bin": record["size_bin"], "n_residues": record["n_residues"],
            "n_pairs": record["n_pairs"], **geometry_metrics(record, all_pred),
        })
        geometry_seconds = time.perf_counter() - tick
        del all_pred

        all_importance.extend(importance_rows(model, {
            "layer": layer, "config_id": selected["config_id"],
            "protein_id": record["protein_id"], "source": record["source"],
        }))
        timing_rows.append({
            "layer": layer, "config_id": selected["config_id"], "protein_id": record["protein_id"],
            "n_fit_rows": int(bundles["fit"]["y"].size),
            "fit_materialization_seconds": bundles["fit"]["materialization_seconds"],
            "validation_materialization_seconds": bundles["validation"]["materialization_seconds"],
            "test_materialization_seconds": bundles["test"]["materialization_seconds"],
            "fit_seconds": fit_info["fit_seconds"], **prediction_seconds,
            "all_pair_materialization_seconds": all_pair_materialization_seconds,
            "all_pair_prediction_seconds": all_pair_prediction_seconds,
            "geometry_seconds": geometry_seconds,
            "best_iteration": fit_info["best_iteration"],
            "best_validation_rmse": fit_info["best_validation_rmse"],
            "total_seconds": time.perf_counter() - protein_start,
        })
        del model
        for bundle in bundles.values():
            cleanup_bundle(bundle)
        gc.collect()
        if protein_number % 10 == 0:
            log(f"Completed {protein_number}/50 selected-protein winner fits")

    if len(timing_rows) != 50:
        raise RuntimeError(f"Winner evaluation completed only {len(timing_rows)}/50 proteins.")
    aggregate_rows = []
    for split_name in ("fit", "validation", "test"):
        pooled_metrics = pooled[split_name].metrics()
        aggregate_rows.append({
            "layer": layer, "config_id": selected["config_id"], "split": split_name,
            "n_proteins": 50, "n_pairs": pooled[split_name].n,
            "mean_per_protein_r2": float(np.mean(r2_values[split_name])),
            "mean_per_protein_mae": float(np.mean(mae_values[split_name])),
            "mean_per_protein_rmse": float(np.mean(rmse_values[split_name])),
            **{f"pooled_{key}": value for key, value in pooled_metrics.items()},
        })

    aggregate_importance = []
    for importance_type in ("gain", "weight", "cover"):
        for channel in range(128):
            values = [float(r["importance"]) for r in all_importance if r["importance_type"] == importance_type and int(r["channel"]) == channel]
            if len(values) != 50:
                raise RuntimeError("Importance aggregation did not receive exactly one row per protein/channel/type.")
            aggregate_importance.append({
                "layer": layer, "config_id": selected["config_id"],
                "importance_type": importance_type, "channel": channel,
                "feature_name": f"avg_symmetric_ch_{channel}",
                "equal_protein_mean_importance": float(np.mean(values)),
                "equal_protein_std_importance": float(np.std(values)),
            })

    write_csv(task_dir / "per_protein_regression_metrics.csv", regression_rows)
    write_csv(task_dir / "aggregate_regression_metrics.csv", aggregate_rows)
    write_csv(task_dir / "per_protein_timing.csv", timing_rows)
    write_csv(task_dir / "per_protein_channel_importance.csv", all_importance)
    write_csv(task_dir / "aggregate_channel_importance.csv", aggregate_importance)
    write_csv(task_dir / "per_protein_geometry_metrics.csv", geometry_rows)
    write_csv(task_dir / "aggregate_geometry_metrics.csv", summarize_geometry(geometry_rows))
    metadata = {
        "status": "complete", "completed_at": timestamp(), "task_id": task_id,
        "layer": layer, "selected_config": selected, "split_digest": observed_digest,
        "n_proteins": 50, "fit_pair_cap": None,
        "kept_early_stopped_60_percent_fit_model": True,
        "refit_on_fit_plus_validation": False,
        "geometry_prediction_scope": "all_pairs_model_predictions",
        "regression_test_scope": "held_out_pair_predictions",
        "geometry_includes_triangle_analysis": False,
        "booster_json_saved": False,
        "task_elapsed_seconds": time.perf_counter() - task_start,
    }
    write_json(task_dir / "run_metadata.json", metadata)
    write_summary(
        task_dir / "summary.txt", f"Per-protein XGBoost winner evaluation — layer {layer}",
        [f"Selected configuration: {selected['config_id']}", "Proteins: 50",
         "Geometry scope: all_pairs_model_predictions (not wholly held out)",
         "Regression test scope: held-out pairs", f"Split digest: {observed_digest}",
         f"Task elapsed seconds: {metadata['task_elapsed_seconds']:.3f}"],
    )
    log(f"Per-protein winner layer {layer} complete")


if __name__ == "__main__":
    run(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))

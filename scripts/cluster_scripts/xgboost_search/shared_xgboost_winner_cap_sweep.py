#!/usr/bin/env python3
"""Train/evaluate the selected shared configuration for one layer/cap task."""

import gc
import json
import os
import time

from xgboost_search_common import (
    ANCHOR_LAYERS, SHARED_CONFIGS, build_shared_split, cleanup_bundle,
    evaluate_model_records, fit_xgboost, importance_rows, load_combined_records,
    load_complete_search, log, map_shared_winner_task, materialize_records,
    n_jobs_from_environment, select_winner, shared_search_dir, shared_winner_dir,
    split_digest, summarize_geometry, timestamp, write_csv, write_json,
    write_summary,
)


def run(task_id: int) -> None:
    layer, cap, cap_label = map_shared_winner_task(task_id)
    task_dir = shared_winner_dir() / f"layer_{layer:02d}" / f"cap_{cap_label}"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_start = time.perf_counter()
    log(f"Shared winner task {task_id}: layer={layer}, cap={cap_label}")

    # This validation happens before importing XGBoost or loading feature matrices.
    search_rows, expected_digest = load_complete_search(shared_search_dir(), SHARED_CONFIGS)
    selected, selection_table, provenance = select_winner(search_rows, SHARED_CONFIGS)
    write_json(task_dir / "selection_provenance.json", provenance)
    write_csv(task_dir / "cross_layer_selection_table.csv", selection_table)

    canonical_path = shared_winner_dir() / "selected_config.json"
    if task_id == 0:
        write_json(canonical_path, {**provenance, "cross_layer_table": selection_table})
        write_csv(shared_winner_dir() / "cross_layer_selection_table.csv", selection_table)
    elif canonical_path.exists():
        canonical = json.loads(canonical_path.read_text())
        if canonical["selected_config"]["config_id"] != selected["config_id"]:
            raise RuntimeError("This task derived a different shared winner than canonical task 0.")

    records, skipped = load_combined_records(ANCHOR_LAYERS)
    split, assignments = build_shared_split(records)
    observed_digest = split_digest(assignments)
    if observed_digest != expected_digest:
        raise RuntimeError(f"Current shared split {observed_digest} differs from search split {expected_digest}.")
    write_csv(task_dir / "split_assignments.csv", assignments)
    write_csv(task_dir / "skipped_proteins.csv", skipped, ["protein_id", "source", "reason"])

    log(f"Training fresh {selected['config_id']} booster (cap={cap_label}) with validation early stopping")
    fit_bundle = materialize_records(split["fit"], layer, cap=cap)
    validation_bundle = materialize_records(split["validation"], layer)
    model, fit_info = fit_xgboost(selected, fit_bundle, validation_bundle, n_jobs_from_environment())
    model.get_booster().save_model(str(task_dir / "booster.json"))
    write_csv(task_dir / "channel_importance.csv", importance_rows(model, {"layer": layer, "cap": cap_label, "config_id": selected["config_id"]}))

    all_regression_rows, aggregate_rows, all_geometry_rows = [], [], []
    evaluation_times = {}
    for split_name in ("validation", "test"):
        rows, aggregate, geometry, timings = evaluate_model_records(
            model, split[split_name], layer, split_name, geometry=True,
            metadata={"cap": cap_label, "config_id": selected["config_id"]},
        )
        all_regression_rows.extend(rows)
        aggregate_rows.append(aggregate)
        all_geometry_rows.extend(geometry)
        for key, value in timings.items():
            evaluation_times[f"{split_name}_{key}"] = value

    write_csv(task_dir / "per_protein_regression_metrics.csv", all_regression_rows)
    write_csv(task_dir / "aggregate_regression_metrics.csv", aggregate_rows)
    write_csv(task_dir / "per_protein_geometry_metrics.csv", all_geometry_rows)
    write_csv(task_dir / "aggregate_geometry_metrics.csv", summarize_geometry(all_geometry_rows))
    timing = {
        "task_id": task_id, "layer": layer, "cap": cap_label,
        "config_id": selected["config_id"], "n_fit_rows": int(fit_bundle["y"].size),
        "n_fit_proteins": len(split["fit"]),
        "fit_feature_materialization_seconds": fit_bundle["materialization_seconds"],
        "validation_for_early_stopping_materialization_seconds": validation_bundle["materialization_seconds"],
        **fit_info, **evaluation_times, "total_elapsed_seconds": time.perf_counter() - task_start,
    }
    write_csv(task_dir / "timing.csv", [timing])
    write_json(task_dir / "run_metadata.json", {
        "status": "complete", "completed_at": timestamp(), "task_id": task_id,
        "layer": layer, "cap": cap_label, "cap_pairs_per_fit_protein": cap,
        "selected_config": selected, "split_digest": observed_digest,
        "kept_early_stopped_60_percent_fit_model": True,
        "refit_on_fit_plus_validation": False, "geometry_includes_triangle_analysis": False,
    })
    write_summary(
        task_dir / "summary.txt", f"Shared XGBoost winner evaluation — layer {layer}, cap {cap_label}",
        [f"Selected configuration: {selected['config_id']}", f"Best iteration: {fit_info['best_iteration']}",
         f"Best validation RMSE: {fit_info['best_validation_rmse']}", f"Fit rows: {fit_bundle['y'].size}",
         f"Split digest: {observed_digest}", f"Total elapsed seconds: {timing['total_elapsed_seconds']:.3f}"],
    )
    del model
    cleanup_bundle(fit_bundle)
    cleanup_bundle(validation_bundle)
    gc.collect()
    log(f"Shared winner task {task_id} complete")


if __name__ == "__main__":
    run(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))

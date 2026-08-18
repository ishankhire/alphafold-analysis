#!/usr/bin/env python3
"""Validation-only cap sensitivity for the selected shared configuration."""

import gc
import json
import os
import time

import numpy as np

from xgboost_search_common import (
    ANCHOR_LAYERS,
    DIRECTIONAL_BASE256_FEATURES,
    SHARED_VARIANT,
    SHARED_CONFIGS,
    SHARED_WEIGHTING_MODE,
    build_shared_split,
    cleanup_bundle,
    fit_xgboost,
    importance_rows,
    load_combined_records,
    load_complete_search,
    log,
    map_shared_winner_task,
    materialize_records,
    n_jobs_from_environment,
    remove_obsolete_outputs,
    select_winner,
    shared_search_dir,
    shared_winner_dir,
    split_digest,
    summarize_prediction_bundle,
    timestamp,
    write_csv,
    write_json,
    write_summary,
)


def run(task_id: int) -> None:
    layer, cap, cap_label = map_shared_winner_task(task_id)
    task_dir = shared_winner_dir() / f"layer_{layer:02d}" / f"cap_{cap_label}"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_start = time.perf_counter()
    log(f"Shared validation cap-sensitivity task {task_id}: layer={layer}, cap={cap_label}")

    # Validate all search artifacts before importing XGBoost or loading matrices.
    search_rows, expected_digest = load_complete_search(
        shared_search_dir(), SHARED_CONFIGS, SHARED_WEIGHTING_MODE
    )
    selected, selection_table, provenance = select_winner(
        search_rows, SHARED_CONFIGS
    )
    write_json(task_dir / "selection_provenance.json", provenance)
    write_csv(task_dir / "cross_layer_selection_table.csv", selection_table)

    canonical_path = shared_winner_dir() / "selected_config.json"
    if task_id == 0:
        write_json(canonical_path, {**provenance, "cross_layer_table": selection_table})
        write_csv(
            shared_winner_dir() / "cross_layer_selection_table.csv",
            selection_table,
        )
    elif canonical_path.exists():
        canonical = json.loads(canonical_path.read_text())
        if canonical["selected_config"]["config_id"] != selected["config_id"]:
            raise RuntimeError(
                "This task derived a different shared winner than canonical task 0."
            )

    records, skipped = load_combined_records(
        ANCHOR_LAYERS, pair_order="lower", master_roles=("fit", "validation"),
    )
    split, assignments = build_shared_split(
        records, required_roles=("fit", "validation"),
    )
    observed_digest = split_digest(assignments)
    if observed_digest != expected_digest:
        raise RuntimeError(
            f"Current shared split {observed_digest} differs from search split "
            f"{expected_digest}."
        )
    write_csv(task_dir / "split_assignments.csv", assignments)
    write_csv(
        task_dir / "skipped_proteins.csv",
        skipped,
        ["protein_id", "source", "reason"],
    )
    remove_obsolete_outputs(
        task_dir,
        filenames=(
            "per_protein_regression_metrics.csv",
            "aggregate_regression_metrics.csv",
            "per_protein_geometry_metrics.csv",
            "aggregate_geometry_metrics.csv",
        ),
    )

    log(
        f"Training fresh {selected['config_id']} booster (cap={cap_label}) "
        "with validation early stopping"
    )
    fit_bundle = materialize_records(
        split["fit"], layer, cap=cap,
        feature_mode=DIRECTIONAL_BASE256_FEATURES,
    )
    validation_bundle = materialize_records(
        split["validation"], layer,
        feature_mode=DIRECTIONAL_BASE256_FEATURES,
    )
    model, fit_info = fit_xgboost(
        selected,
        fit_bundle,
        validation_bundle,
        n_jobs_from_environment(),
        SHARED_WEIGHTING_MODE,
    )
    model.get_booster().save_model(str(task_dir / "booster.json"))
    identity = {
        "variant": SHARED_VARIANT,
        "config_id": selected["config_id"],
        "cap": cap_label,
        "layer": layer,
    }
    write_csv(
        task_dir / "channel_importance.csv",
        importance_rows(model, identity, DIRECTIONAL_BASE256_FEATURES),
    )

    regression_rows, split_summaries = [], []
    evaluation_times = {}

    tick = time.perf_counter()
    train_pred = np.asarray(model.predict(fit_bundle["x"]), dtype=np.float64)
    evaluation_times["train_prediction_seconds"] = time.perf_counter() - tick
    tick = time.perf_counter()
    rows, aggregate = summarize_prediction_bundle(
        fit_bundle,
        train_pred,
        "fit",
        "sampled_training_rows",
        identity,
    )
    evaluation_times["train_metric_seconds"] = time.perf_counter() - tick
    regression_rows.extend(rows)
    split_summaries.append(aggregate)
    del train_pred

    tick = time.perf_counter()
    validation_pred = np.asarray(
        model.predict(validation_bundle["x"]), dtype=np.float64
    )
    evaluation_times["validation_prediction_seconds"] = time.perf_counter() - tick
    tick = time.perf_counter()
    rows, aggregate = summarize_prediction_bundle(
        validation_bundle,
        validation_pred,
        "validation",
        "full_protein_pairs",
        identity,
    )
    evaluation_times["validation_metric_seconds"] = time.perf_counter() - tick
    regression_rows.extend(rows)
    split_summaries.append(aggregate)
    del validation_pred

    if len(regression_rows) != len(records):
        raise RuntimeError(
            f"Expected one selected-model regression row for each of "
            f"{len(records)} proteins; observed {len(regression_rows)}."
        )
    write_csv(task_dir / "per_protein.csv", regression_rows)
    write_csv(task_dir / "split_summary.csv", split_summaries)

    timing = {
        "task_id": task_id,
        "layer": layer,
        "cap": cap_label,
        "config_id": selected["config_id"],
        "n_train_rows": int(fit_bundle["y"].size),
        "n_train_proteins": len(split["fit"]),
        "n_validation_rows": int(validation_bundle["y"].size),
        "n_validation_proteins": len(split["validation"]),
        "train_feature_materialization_seconds": fit_bundle[
            "materialization_seconds"
        ],
        "validation_feature_materialization_seconds": validation_bundle[
            "materialization_seconds"
        ],
        **fit_info,
        **evaluation_times,
        "total_elapsed_seconds": time.perf_counter() - task_start,
    }
    write_csv(task_dir / "timing.csv", [timing])
    write_json(task_dir / "run_metadata.json", {
        "status": "complete",
        "completed_at": timestamp(),
        "task_id": task_id,
        "layer": layer,
        "cap": cap_label,
        "cap_pairs_per_training_protein": cap,
        "selected_config": selected,
        "split_digest": observed_digest,
        "weighting_mode": SHARED_WEIGHTING_MODE,
        "training_evaluation_scope": "sampled_training_rows",
        "validation_evaluation_scope": "full_protein_pairs",
        "kept_early_stopped_60_percent_training_model": True,
        "refit_on_train_plus_validation": False,
        "geometry_computed": False,
        "n_regression_rows": len(regression_rows),
    })
    write_summary(
        task_dir / "summary.txt",
        f"Shared XGBoost validation cap sensitivity — layer {layer}, cap {cap_label}",
        [
            f"Selected configuration: {selected['config_id']}",
            f"Best iteration: {fit_info['best_iteration']}",
            f"Best validation RMSE: {fit_info['best_validation_rmse']}",
            f"Training rows: {fit_bundle['y'].size}",
            f"Per-protein regression rows: {len(regression_rows)}",
            "Geometry computed: no",
            "Locked-test access: prohibited",
            f"Split digest: {observed_digest}",
            f"Total elapsed seconds: {timing['total_elapsed_seconds']:.3f}",
        ],
    )
    del model
    cleanup_bundle(fit_bundle)
    cleanup_bundle(validation_bundle)
    gc.collect()
    log(f"Shared winner task {task_id} complete")


if __name__ == "__main__":
    run(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))

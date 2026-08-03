#!/usr/bin/env python3
"""Rerun and evaluate the globally selected per-protein XGBoost config."""

import gc
import json
import os
import time

import numpy as np

from xgboost_search_common import (
    ANCHOR_LAYERS,
    PER_PROTEIN_VARIANT,
    PER_PROTEIN_CONFIGS,
    PER_PROTEIN_WEIGHTING_MODE,
    PooledStats,
    attach_pair_splits,
    cleanup_bundle,
    fit_xgboost,
    importance_rows,
    load_combined_records,
    load_complete_search,
    log,
    map_winner_layer_task,
    materialize_records,
    n_jobs_from_environment,
    pair_split_summary,
    per_protein_search_dir,
    per_protein_winner_dir,
    regression_output_row,
    remove_obsolete_outputs,
    require_non_degenerate_iterations,
    select_per_protein_manifest,
    select_winner,
    split_summary_row,
    timestamp,
    write_csv,
    write_json,
    write_summary,
)


def run(task_id: int) -> None:
    layer = map_winner_layer_task(task_id)
    task_dir = per_protein_winner_dir() / f"layer_{layer:02d}"
    task_dir.mkdir(parents=True, exist_ok=True)
    task_start = time.perf_counter()
    log(f"Per-protein winner task {task_id}: layer={layer}")

    # Fail on old, missing, incomplete, or inconsistent search artifacts first.
    search_rows, expected_digest = load_complete_search(
        per_protein_search_dir(),
        PER_PROTEIN_CONFIGS,
        PER_PROTEIN_WEIGHTING_MODE,
    )
    selected, selection_table, provenance = select_winner(
        search_rows, PER_PROTEIN_CONFIGS
    )
    write_json(task_dir / "selection_provenance.json", provenance)
    write_csv(task_dir / "cross_layer_selection_table.csv", selection_table)
    canonical_path = per_protein_winner_dir() / "selected_config.json"
    if task_id == 0:
        write_json(canonical_path, {**provenance, "cross_layer_table": selection_table})
    elif canonical_path.exists():
        canonical = json.loads(canonical_path.read_text())
        if canonical["selected_config"]["config_id"] != selected["config_id"]:
            raise RuntimeError(
                "This task derived a different per-protein winner than canonical task 0."
            )

    records, skipped = load_combined_records(ANCHOR_LAYERS)
    selected_records = select_per_protein_manifest(records)
    if len(selected_records) != 50:
        raise RuntimeError(
            f"Expected exactly 50 winner-evaluation proteins; found "
            f"{len(selected_records)}."
        )
    attach_pair_splits(selected_records)
    split_rows, observed_digest = pair_split_summary(selected_records)
    if observed_digest != expected_digest:
        raise RuntimeError(
            f"Current pair split {observed_digest} differs from search split "
            f"{expected_digest}."
        )
    write_csv(task_dir / "pair_split_summary.csv", split_rows)
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
            "per_protein_timing.csv",
            "per_protein_geometry_metrics.csv",
            "aggregate_geometry_metrics.csv",
        ),
    )

    identity = {
        "variant": PER_PROTEIN_VARIANT,
        "config_id": selected["config_id"],
        "layer": layer,
    }
    regression_rows, all_importance = [], []
    pools = {name: PooledStats() for name in ("train", "validation", "test")}
    split_regression_rows = {name: [] for name in pools}
    best_iterations, best_scores = [], []
    total_fit_seconds = 0.0

    for protein_number, record in enumerate(selected_records, start=1):
        bundles = {
            "train": materialize_records([record], layer, split_name="fit"),
            "validation": materialize_records(
                [record], layer, split_name="validation"
            ),
            "test": materialize_records([record], layer, split_name="test"),
        }
        model, fit_info = fit_xgboost(
            selected,
            bundles["train"],
            bundles["validation"],
            n_jobs_from_environment(),
            PER_PROTEIN_WEIGHTING_MODE,
        )
        best_iterations.append(fit_info["best_iteration"])
        best_scores.append(fit_info["best_validation_rmse"])
        total_fit_seconds += fit_info["fit_seconds"]

        for split_name, bundle in bundles.items():
            pred = np.asarray(model.predict(bundle["x"]), dtype=np.float64)
            pools[split_name].update(bundle["y"], pred)
            row = regression_output_row(
                record,
                bundle["y"],
                pred,
                "fit" if split_name == "train" else split_name,
                "pair_split_rows",
                identity,
            )
            regression_rows.append(row)
            split_regression_rows[split_name].append(row)
            del pred

        all_importance.extend(importance_rows(model, {
            **identity,
            "protein_id": record["protein_id"],
            "source": record["source"],
        }))
        del model
        for bundle in bundles.values():
            cleanup_bundle(bundle)
        gc.collect()
        if protein_number % 10 == 0:
            log(f"Completed {protein_number}/50 selected-protein winner fits")

    require_non_degenerate_iterations(
        selected["config_id"], best_iterations, len(selected_records)
    )
    if len(regression_rows) != 150:
        raise RuntimeError(
            f"Expected 150 train/validation/test regression rows; observed "
            f"{len(regression_rows)}."
        )

    split_summaries = [
        split_summary_row(split_regression_rows[name], pools[name], identity)
        for name in ("train", "validation", "test")
    ]
    aggregate_importance = []
    for importance_type in ("gain", "weight", "cover"):
        for channel in range(128):
            values = [
                float(row["importance"])
                for row in all_importance
                if row["importance_type"] == importance_type
                and int(row["channel"]) == channel
            ]
            if len(values) != 50:
                raise RuntimeError(
                    "Importance aggregation did not receive exactly one row per "
                    "protein/channel/type."
                )
            aggregate_importance.append({
                **identity,
                "importance_type": importance_type,
                "channel": channel,
                "feature_name": f"avg_symmetric_ch_{channel}",
                "equal_protein_mean_importance": float(np.mean(values)),
                "equal_protein_std_importance": float(np.std(values)),
            })

    write_csv(task_dir / "per_protein.csv", regression_rows)
    write_csv(task_dir / "split_summary.csv", split_summaries)
    write_csv(task_dir / "per_protein_channel_importance.csv", all_importance)
    write_csv(task_dir / "aggregate_channel_importance.csv", aggregate_importance)

    task_elapsed = time.perf_counter() - task_start
    metadata = {
        "status": "complete",
        "completed_at": timestamp(),
        "task_id": task_id,
        "layer": layer,
        "selected_config": selected,
        "split_digest": observed_digest,
        "n_proteins": 50,
        "n_regression_rows": len(regression_rows),
        "fit_pair_cap": None,
        "weighting_mode": PER_PROTEIN_WEIGHTING_MODE,
        "regression_evaluation_scope": "pair_split_rows",
        "kept_early_stopped_60_percent_training_model": True,
        "refit_on_train_plus_validation": False,
        "geometry_computed": False,
        "booster_json_saved": False,
        "best_iteration_mean": float(np.mean(best_iterations)),
        "best_iteration_median": float(np.median(best_iterations)),
        "best_iteration_min": int(np.min(best_iterations)),
        "best_iteration_max": int(np.max(best_iterations)),
        "best_validation_rmse_mean": float(np.mean(best_scores)),
        "total_fit_seconds": total_fit_seconds,
        "task_elapsed_seconds": task_elapsed,
    }
    write_json(task_dir / "run_metadata.json", metadata)
    write_summary(
        task_dir / "summary.txt",
        f"Per-protein XGBoost winner evaluation — layer {layer}",
        [
            f"Selected configuration: {selected['config_id']}",
            "Proteins: 50",
            f"Per-protein regression rows: {len(regression_rows)}",
            "Regression splits: train, validation, test",
            "Weighting mode: uniform_rows",
            "Geometry computed: no",
            f"Best iteration mean: {metadata['best_iteration_mean']:.3f}",
            f"Split digest: {observed_digest}",
            f"Task elapsed seconds: {task_elapsed:.3f}",
        ],
    )
    log(f"Per-protein winner layer {layer} complete in {task_elapsed:.3f}s")


if __name__ == "__main__":
    run(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))

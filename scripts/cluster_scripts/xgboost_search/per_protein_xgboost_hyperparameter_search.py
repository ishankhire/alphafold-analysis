#!/usr/bin/env python3
"""Array entrypoint for the 50-protein XGBoost hyperparameter search."""

import gc
import os
import time

import numpy as np

from xgboost_search_common import (
    ANCHOR_LAYERS,
    FIXED_XGBOOST_PARAMETERS,
    PER_PROTEIN_CONFIGS,
    PER_PROTEIN_WEIGHTING_MODE,
    PooledStats,
    attach_pair_splits,
    cleanup_bundle,
    fit_xgboost,
    load_combined_records,
    log,
    map_search_task,
    materialize_records,
    metric_row,
    n_jobs_from_environment,
    pair_split_summary,
    per_protein_search_dir,
    remove_obsolete_outputs,
    require_non_degenerate_iterations,
    select_per_protein_manifest,
    timestamp,
    write_csv,
    write_json,
    write_summary,
)


def run(task_id: int) -> None:
    layer = map_search_task(task_id)
    layer_dir = per_protein_search_dir() / f"layer_{layer:02d}"
    layer_dir.mkdir(parents=True, exist_ok=True)
    remove_obsolete_outputs(layer_dir, filenames=("per_protein_timing.csv",))
    task_start = time.perf_counter()
    log(f"Per-protein XGBoost search task {task_id}: layer {layer}")

    records, skipped = load_combined_records(ANCHOR_LAYERS)
    selected_records = select_per_protein_manifest(records)
    if len(selected_records) != 50:
        raise RuntimeError(
            f"Expected exactly 50 per-protein search proteins; found {len(selected_records)}."
        )
    attach_pair_splits(selected_records)
    split_rows, digest = pair_split_summary(selected_records)
    manifest_rows = [
        {
            "protein_id": record["protein_id"],
            "source": record["source"],
            "n_residues": record["n_residues"],
            "n_pairs": record["n_pairs"],
            "size_bin": record["size_bin"],
        }
        for record in selected_records
    ]
    write_csv(layer_dir / "protein_manifest.csv", manifest_rows)
    write_csv(layer_dir / "pair_split_summary.csv", split_rows)
    write_csv(
        layer_dir / "skipped_proteins.csv",
        skipped,
        ["protein_id", "source", "reason"],
    )
    write_json(layer_dir / "configurations.json", list(PER_PROTEIN_CONFIGS))
    metadata = {
        "pipeline": "per_protein_search",
        "layer": layer,
        "task_id": task_id,
        "anchor_layers": list(ANCHOR_LAYERS),
        "split_digest": digest,
        "n_proteins": 50,
        "proteins_per_size_bin": 10,
        "pair_split": "deterministic 60/20/20",
        "fit_pair_cap": None,
        "feature_definition": "float64 0.5*(pair_block[i,j,:]+pair_block[j,i,:])",
        "selection_metric": "validation_mean_per_protein_r2",
        "weighting_mode": PER_PROTEIN_WEIGHTING_MODE,
        "fixed_xgboost_parameters": FIXED_XGBOOST_PARAMETERS,
        "status": "running",
        "started_at": timestamp(),
    }
    write_json(layer_dir / "search_metadata.json", metadata)

    summary_rows = []
    completed_fits = 0
    for config in PER_PROTEIN_CONFIGS:
        config_start = time.perf_counter()
        config_id = config["config_id"]
        log(f"Starting {config_id} across all 50 proteins with uniform rows")
        pools = {name: PooledStats() for name in ("train", "validation", "test")}
        r2_values = {name: [] for name in pools}
        best_iterations, best_scores = [], []
        totals = {
            "materialization_seconds": 0.0,
            "fit_seconds": 0.0,
            "train_prediction_seconds": 0.0,
            "validation_prediction_seconds": 0.0,
            "test_prediction_seconds": 0.0,
            "metric_seconds": 0.0,
        }

        for protein_number, record in enumerate(selected_records, start=1):
            bundles = {
                "train": materialize_records([record], layer, split_name="fit"),
                "validation": materialize_records(
                    [record], layer, split_name="validation"
                ),
                "test": materialize_records([record], layer, split_name="test"),
            }
            totals["materialization_seconds"] += sum(
                bundle["materialization_seconds"] for bundle in bundles.values()
            )
            model, fit_info = fit_xgboost(
                config,
                bundles["train"],
                bundles["validation"],
                n_jobs_from_environment(),
                PER_PROTEIN_WEIGHTING_MODE,
            )
            totals["fit_seconds"] += fit_info["fit_seconds"]

            predictions = {}
            for split_name, bundle in bundles.items():
                tick = time.perf_counter()
                predictions[split_name] = np.asarray(
                    model.predict(bundle["x"]), dtype=np.float64
                )
                totals[f"{split_name}_prediction_seconds"] += (
                    time.perf_counter() - tick
                )

            tick = time.perf_counter()
            for split_name, bundle in bundles.items():
                metrics = metric_row(bundle["y"], predictions[split_name])
                pools[split_name].update(bundle["y"], predictions[split_name])
                r2_values[split_name].append(metrics["r2"])
            totals["metric_seconds"] += time.perf_counter() - tick
            best_iterations.append(fit_info["best_iteration"])
            best_scores.append(fit_info["best_validation_rmse"])
            completed_fits += 1

            del model
            predictions.clear()
            for bundle in bundles.values():
                cleanup_bundle(bundle)
            gc.collect()
            if protein_number % 10 == 0:
                log(f"{config_id}: completed {protein_number}/50 proteins")

        require_non_degenerate_iterations(config_id, best_iterations, 50)
        pooled_metrics = {name: pool.metrics() for name, pool in pools.items()}
        summary_rows.append({
            **config,
            "layer": layer,
            "n_proteins": 50,
            "train_mean_per_protein_r2": float(np.mean(r2_values["train"])),
            "train_pooled_r2": pooled_metrics["train"]["r2"],
            "validation_mean_per_protein_r2": float(
                np.mean(r2_values["validation"])
            ),
            "validation_pooled_r2": pooled_metrics["validation"]["r2"],
            "test_mean_per_protein_r2": float(np.mean(r2_values["test"])),
            "test_pooled_r2": pooled_metrics["test"]["r2"],
            "best_iteration_mean": float(np.mean(best_iterations)),
            "best_iteration_median": float(np.median(best_iterations)),
            "best_iteration_min": int(np.min(best_iterations)),
            "best_iteration_max": int(np.max(best_iterations)),
            "best_validation_rmse_mean": float(np.mean(best_scores)),
            **totals,
            "total_elapsed_seconds": time.perf_counter() - config_start,
            "split_digest": digest,
        })
        write_csv(layer_dir / "search_summary.csv", summary_rows)
        write_summary(
            layer_dir / "summary.txt",
            f"Per-protein XGBoost search — layer {layer}",
            [
                f"Completed configurations: {len(summary_rows)}/{len(PER_PROTEIN_CONFIGS)}",
                f"Completed fits: {completed_fits}/{50 * len(PER_PROTEIN_CONFIGS)}",
                f"Latest configuration: {config_id}",
                "Weighting mode: uniform_rows",
                "All-proteins-iteration-zero diagnostic: passed",
                f"Split digest: {digest}",
                f"Elapsed seconds so far: {time.perf_counter() - task_start:.3f}",
            ],
        )
        log(
            f"Checkpointed {config_id}; validation mean protein "
            f"R2={summary_rows[-1]['validation_mean_per_protein_r2']:.6f}"
        )

    if len(summary_rows) != len(PER_PROTEIN_CONFIGS) or completed_fits != 200:
        raise RuntimeError("Refusing to complete an incomplete 50-protein search artifact.")
    task_elapsed = time.perf_counter() - task_start
    metadata.update({
        "status": "complete",
        "completed_at": timestamp(),
        "completed_fits": completed_fits,
        "task_elapsed_seconds": task_elapsed,
    })
    write_json(layer_dir / "search_metadata.json", metadata)
    write_summary(
        layer_dir / "summary.txt",
        f"Per-protein XGBoost search — layer {layer}",
        [
            f"Completed configurations: {len(summary_rows)}/{len(PER_PROTEIN_CONFIGS)}",
            f"Completed fits: {completed_fits}",
            "Weighting mode: uniform_rows",
            "All-proteins-iteration-zero diagnostic: passed",
            f"Split digest: {digest}",
            f"Task elapsed seconds: {task_elapsed:.3f}",
        ],
    )
    log(f"Per-protein layer {layer} search complete in {task_elapsed:.3f}s")


if __name__ == "__main__":
    run(int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")))

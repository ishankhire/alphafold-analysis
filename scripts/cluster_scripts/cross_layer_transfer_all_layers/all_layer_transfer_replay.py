#!/usr/bin/env python3
"""Replay all 48 frozen shared-OLS source probes on one target layer (array task)."""

from __future__ import annotations

import csv
import gc
import os
import resource
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
for path in (SCRIPT_DIR, CLUSTER_DIR, CLUSTER_DIR / "cross_layer_transfer",
             CLUSTER_DIR / "cross_validation", CLUSTER_DIR / "triangle_violation_subgroups"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from aggregate_cross_layer_transfer import PARITY_TOLERANCE, stats_from_vector  # noqa: E402
from all_layer_transfer_common import (  # noqa: E402
    CANONICAL_SEQUENCE_GROUPS, CELL_INDEX, LAYERS, OUTPUT_ENV, PRODUCER_SPEC, WORKER_RUNTIME_COLUMNS,
    TargetAccumulator, analysis_name, cohort_name, default_output_root, load_probe_bank,
    predict_all_sources, select_cohort, sequence_fine_bins, summarize_protein, target_layer_for_task,
    validate_final_shard,
)
from canonical_storage import assert_writable, csv_root, deployment_revision  # noqa: E402
from cross_layer_transfer_replay import apply_ecod_root_override  # noqa: E402
from cross_validation_common import load_cv_folds  # noqa: E402
from regression_output_helpers import atomic_savez_compressed, atomic_write_csv, atomic_write_json  # noqa: E402
from regression_protocol import SEQUENCE_BIN_LABELS, sequence_separation_bin_codes  # noqa: E402
from triangle_violation_subgroup_replay import load_protein  # noqa: E402

CHECKPOINT_ENV = "ALL_LAYER_CHECKPOINT_PROTEINS"
DEFAULT_CHECKPOINT_PROTEINS = 100
NATIVE_CHECK_METRICS = ("r2", "mae", "rmse", "bias_pred_minus_true")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    return float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) / 1024.0


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def producer_root() -> Path:
    return csv_root() / "regression_5fold_v1"


def output_root(cohort: str) -> Path:
    value = os.environ.get(OUTPUT_ENV)
    return Path(value).expanduser().resolve() if value else default_output_root(cohort)


def read_runtime(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def require_single_chain_bins(pair_i: np.ndarray, pair_j: np.ndarray, chain_ids) -> None:
    """Fine sequence bins must nest in the canonical chain-aware bins for every pair."""
    canonical = sequence_separation_bin_codes(pair_i, pair_j, chain_ids=chain_ids)
    if np.any(canonical < 0):
        raise ValueError("Interchain pair found; fine sequence bins assume a single chain")
    lookup = np.empty(sum(len(group) for group in CANONICAL_SEQUENCE_GROUPS.values()), dtype=np.int64)
    for code, label in enumerate(SEQUENCE_BIN_LABELS):
        lookup[list(CANONICAL_SEQUENCE_GROUPS[label])] = code
    if not np.array_equal(lookup[sequence_fine_bins(pair_i, pair_j)], canonical):
        raise ValueError("Fine sequence bins do not nest in the canonical bins")


def protein_summary(outer, target_position: int, target_layer: int, bank) -> dict[str, np.ndarray]:
    coordinates, chain_ids, pair_i, pair_j, features = load_protein(outer, target_layer)
    fold = int(outer.outer_fold)
    weights, constants = bank.restandardized_weights(target_position, fold)
    predictions = predict_all_sources(features, weights, constants)
    del features
    require_single_chain_bins(pair_i, pair_j, chain_ids)
    return summarize_protein(
        y=np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1), predictions=predictions,
        pair_i=pair_i, pair_j=pair_j, n_residues=int(outer.n_residues),
        baseline_value=float(bank.global_baseline[fold]),
    )


def verify_native_cell(accumulator: TargetAccumulator, target_position: int, per_protein_path: Path) -> None:
    """Fail the task before writing a shard if the native probe disagrees with the producer."""
    with per_protein_path.open(newline="") as handle:
        producer = {row["protein_id"]: row for row in csv.DictReader(handle)
                    if int(row["layer"]) == accumulator.target_layer}
    whole = CELL_INDEX[("all_pairs", "all")]
    probe = accumulator.arrays["probe"][target_position]
    for position, protein in enumerate(accumulator.protein_ids):
        metrics = stats_from_vector(probe[position, whole]).metrics()
        for metric in NATIVE_CHECK_METRICS:
            expected = float(producer[protein][metric])
            if abs(metrics[metric] - expected) > PARITY_TOLERANCE * max(1.0, abs(expected)):
                raise ValueError(f"Native {metric} for {protein} at layer {accumulator.target_layer} is "
                                 f"{metrics[metric]}, producer has {expected}")


def main() -> None:
    cohort = cohort_name()
    root = output_root(cohort)
    assert_writable(root)
    target_layer = target_layer_for_task(int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")), LAYERS)
    target_position = LAYERS.index(target_layer)
    shards = root / "layer_shards"
    final_path = shards / f"target_layer_{target_layer:02d}.npz"
    checkpoint_path = shards / f".target_layer_{target_layer:02d}.checkpoint.npz"
    runtime_path = shards / f"runtime_target_layer_{target_layer:02d}.csv"

    started_wall, started = utc_now(), time.perf_counter()
    ecod_override = apply_ecod_root_override()
    if ecod_override:
        log(f"target_layer={target_layer} ECOD input override: {ecod_override}")
    fold_bundle = load_cv_folds(metadata_dir=producer_root() / "metadata")
    protein_ids = select_cohort(fold_bundle, cohort)
    revision = deployment_revision(SCRIPT_DIR)
    if final_path.is_file():
        validate_final_shard(TargetAccumulator.from_file(final_path, target_layer, protein_ids, len(LAYERS), revision))
        log(f"target_layer={target_layer} valid final shard already exists")
        return
    shards.mkdir(parents=True, exist_ok=True)

    bank, provenance, status_rows = load_probe_bank(producer_root(), fold_bundle, LAYERS)
    log(f"target_layer={target_layer} loaded {len(status_rows)} frozen shared-OLS fits")
    atomic_write_json(shards / f"artifact_provenance_target_layer_{target_layer:02d}.json", {
        "analysis_name": analysis_name(cohort), "cohort": cohort, "code_revision": revision,
        "target_layer": target_layer, "created_at": utc_now(),
        "metadata_digests": dict(fold_bundle.fold_digests), "producer": provenance,
        "source_probe_status": status_rows, "ecod_input_override": ecod_override,
    })

    if checkpoint_path.is_file():
        accumulator = TargetAccumulator.from_file(checkpoint_path, target_layer, protein_ids, len(LAYERS), revision)
        log(f"target_layer={target_layer} resuming with "
            f"{int(accumulator.arrays['completed'].sum())} completed proteins")
    else:
        accumulator = TargetAccumulator(target_layer, protein_ids, len(LAYERS), revision)
    runtime_rows = read_runtime(runtime_path)
    failures: list[str] = []
    checkpoint_every = int(os.environ.get(CHECKPOINT_ENV, str(DEFAULT_CHECKPOINT_PROTEINS)))

    def save_progress() -> None:
        atomic_savez_compressed(checkpoint_path, **accumulator.payload())
        atomic_write_csv(runtime_path, WORKER_RUNTIME_COLUMNS, runtime_rows)

    for number, protein_id in enumerate(protein_ids, start=1):
        if accumulator.arrays["completed"][number - 1]:
            continue
        outer = fold_bundle.outer_by_protein[protein_id]
        protein_wall, protein_started = utc_now(), time.perf_counter()
        try:
            accumulator.add(number - 1, int(outer.outer_fold),
                            protein_summary(outer, target_position, target_layer, bank))
            status, detail = "complete", ""
        except Exception as exc:  # recorded, then raised after the loop
            status, detail = "failed", f"{type(exc).__name__}: {exc}"
            failures.append(f"{protein_id}: {detail}")
            traceback.print_exc()
        elapsed = time.perf_counter() - protein_started
        runtime_rows.append({
            "stage": "replay_protein", "target_layer": target_layer, "protein_id": protein_id,
            "started_at": protein_wall, "finished_at": utc_now(), "elapsed_seconds": elapsed,
            "peak_rss_mb": peak_rss_mb(), "status": status, "detail": detail,
        })
        log(f"target_layer={target_layer} progress={number}/{len(protein_ids)} protein={protein_id} "
            f"n_residues={outer.n_residues} status={status} elapsed={elapsed:.1f}s")
        if number % checkpoint_every == 0 or status == "failed":
            save_progress()
        gc.collect()

    if failures or not accumulator.arrays["completed"].all():
        save_progress()
        raise RuntimeError(f"Target layer {target_layer} incomplete; failures={failures[:10]}")
    validate_final_shard(accumulator)
    verify_native_cell(accumulator, target_position, producer_root() / PRODUCER_SPEC.family_dir / "per_protein.csv")
    log(f"target_layer={target_layer} native cell reproduces the producer for {len(protein_ids)} proteins")
    atomic_savez_compressed(final_path, **accumulator.payload())
    checkpoint_path.unlink(missing_ok=True)
    runtime_rows.append({
        "stage": "target_layer_complete", "target_layer": target_layer, "protein_id": "",
        "started_at": started_wall, "finished_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb(),
        "status": "complete", "detail": "",
    })
    atomic_write_csv(runtime_path, WORKER_RUNTIME_COLUMNS, runtime_rows)
    log(f"target_layer={target_layer} complete elapsed={time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Replay ten frozen shared-OLS source probes on one target layer (array task)."""

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
CV_DIR = CLUSTER_DIR / "cross_validation"
TRIANGLE_DIR = CLUSTER_DIR / "triangle_violation_subgroups"
for path in (SCRIPT_DIR, CLUSTER_DIR, CV_DIR, TRIANGLE_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from canonical_storage import assert_writable, csv_root  # noqa: E402
from cross_layer_transfer_common import (  # noqa: E402
    ANALYSIS_NAME, DEFAULT_OUTPUT_SUBDIR, EXPECTED_PROTEINS, MODES, OUTPUT_ENV,
    PRIMITIVE_COLUMNS, ROWS_PER_PROTEIN, TRANSFER_LAYERS, WORKER_RUNTIME_COLUMNS,
    convert_primitive_row, global_baseline_by_fold, load_transfer_artifacts,
    predict_transfer, primitive_rows, stats_payload, target_layer_for_task,
    validate_target_shard,
)
from cross_validation_common import load_cv_folds  # noqa: E402
from regression_output_helpers import atomic_write_csv, atomic_write_json  # noqa: E402
from regression_protocol import (  # noqa: E402
    sequence_separation_bin_codes, spatial_distance_bin_codes,
)
from triangle_violation_subgroup_replay import load_protein  # noqa: E402
import cluster_regression_helpers  # noqa: E402

# One-off, unset by default: points this replay's ECOD inputs at a relocated
# copy of the ECOD experiments directory. Shared helpers and docs are unchanged.
ECOD_ROOT_OVERRIDE_ENV = "CROSS_LAYER_ECOD_EXPERIMENTS_ROOT"


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def producer_root() -> Path:
    return csv_root() / "regression_5fold_v1"


def output_root() -> Path:
    value = os.environ.get(OUTPUT_ENV)
    if value:
        return Path(value).expanduser().resolve()
    return csv_root() / DEFAULT_OUTPUT_SUBDIR


def read_shard(path: Path, target_layer: int, *, expected_proteins: int | None) -> tuple[list[dict[str, object]], set[str]]:
    if not path.is_file():
        return [], set()
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PRIMITIVE_COLUMNS:
            raise ValueError(f"Shard schema mismatch: {path}")
        rows = [convert_primitive_row(row) for row in reader]
    if not rows:
        return [], set()
    return rows, validate_target_shard(rows, target_layer, expected_proteins=expected_proteins)


def read_runtime(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def apply_ecod_root_override() -> dict[str, str] | None:
    value = os.environ.get(ECOD_ROOT_OVERRIDE_ENV)
    if not value:
        return None
    root = Path(value).expanduser()
    pair_root, pdb_root = root / "representation_saves", root / "openfold_reps_outputs_sm"
    for path in (pair_root, pdb_root):
        if not path.is_dir():
            raise FileNotFoundError(f"{ECOD_ROOT_OVERRIDE_ENV} directory missing: {path}")
    cluster_regression_helpers.ECOD_PAIR_BLOCKS_ROOT = pair_root
    cluster_regression_helpers.ECOD_PDB_ROOT = pdb_root
    return {"ecod_pair_blocks_root": str(pair_root), "ecod_pdb_root": str(pdb_root)}


def protein_rows(outer, target_layer: int, models, baselines) -> list[dict[str, object]]:
    coordinates, chain_ids, pair_i, pair_j, features = load_protein(outer, target_layer)
    try:
        y = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
        sequence_codes = sequence_separation_bin_codes(pair_i, pair_j, chain_ids=chain_ids)
        spatial_codes = spatial_distance_bin_codes(y)
        fold = int(outer.outer_fold)
        base = {
            "analysis_name": ANALYSIS_NAME, "target_layer": target_layer,
            "protein_id": outer.protein_id, "source": outer.source, "outer_fold": fold,
            "n_residues": int(outer.n_residues),
        }
        target_fit = models[(target_layer, fold)]
        rows: list[dict[str, object]] = []
        for source_layer in TRANSFER_LAYERS:
            source_fit = models[(source_layer, fold)]
            by_mode = {}
            for mode in MODES:
                prediction = predict_transfer(features, source_fit, target_fit, mode)
                by_mode[mode] = prediction
                rows.extend(primitive_rows(
                    identity={**base, "source_layer": source_layer, "transfer_mode": mode,
                              "predictor": "probe"},
                    y_true=y, predictions=prediction,
                    sequence_codes=sequence_codes, spatial_codes=spatial_codes,
                ))
            if source_layer == target_layer and not np.array_equal(
                by_mode["raw"], by_mode["target_restandardized"],
            ):
                raise RuntimeError("Same-layer raw and restandardized predictions differ")
        baseline = np.full(y.shape, baselines[fold], dtype=np.float64)
        rows.append({
            **base, "source_layer": "", "transfer_mode": "", "predictor": "global_baseline",
            "category_scope": "all_pairs", "category_label": "all", **stats_payload(y, baseline),
        })
    finally:
        del features
    if len(rows) != ROWS_PER_PROTEIN:
        raise RuntimeError(f"Protein produced {len(rows)} rows, expected {ROWS_PER_PROTEIN}")
    return rows


def main() -> None:
    root = output_root()
    assert_writable(root)
    target_layer = target_layer_for_task(int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    shards = root / "layer_shards"
    final_path = shards / f"cross_layer_primitives_target_layer_{target_layer:02d}.csv"
    checkpoint_path = shards / f".cross_layer_primitives_target_layer_{target_layer:02d}.checkpoint.csv"
    runtime_path = shards / f"runtime_target_layer_{target_layer:02d}.csv"
    if final_path.is_file():
        read_shard(final_path, target_layer, expected_proteins=EXPECTED_PROTEINS)
        log(f"target_layer={target_layer} valid final shard already exists")
        return
    shards.mkdir(parents=True, exist_ok=True)

    started_wall, started = utc_now(), time.perf_counter()
    ecod_override = apply_ecod_root_override()
    if ecod_override:
        log(f"target_layer={target_layer} ECOD input override: {ecod_override}")
    fold_bundle = load_cv_folds(metadata_dir=producer_root() / "metadata")
    n_proteins = len(fold_bundle.eligible_protein_ids)
    if n_proteins != EXPECTED_PROTEINS:
        raise ValueError("Frozen fivefold cohort size changed")
    models, provenance, status_rows = load_transfer_artifacts(producer_root(), fold_bundle)
    baselines = global_baseline_by_fold(models)
    log(f"target_layer={target_layer} loaded {len(models)} frozen shared-OLS fits")
    atomic_write_json(shards / f"artifact_provenance_target_layer_{target_layer:02d}.json", {
        "analysis_name": ANALYSIS_NAME, "target_layer": target_layer, "created_at": utc_now(),
        "metadata_digests": dict(fold_bundle.fold_digests), "producer": provenance,
        "source_probe_status": status_rows, "ecod_input_override": ecod_override,
    })

    rows, complete = read_shard(checkpoint_path, target_layer, expected_proteins=None)
    runtime_rows = read_runtime(runtime_path)
    failures: list[str] = []
    checkpoint_every = int(os.environ.get("CROSS_LAYER_CHECKPOINT_PROTEINS", "25"))
    if complete:
        log(f"target_layer={target_layer} resuming with {len(complete)} completed proteins")

    for number, protein_id in enumerate(fold_bundle.eligible_protein_ids, start=1):
        if protein_id in complete:
            continue
        outer = fold_bundle.outer_by_protein[protein_id]
        protein_wall, protein_started = utc_now(), time.perf_counter()
        try:
            rows.extend(protein_rows(outer, target_layer, models, baselines))
            complete.add(protein_id)
            status, detail = "complete", ""
        except Exception as exc:  # recorded and re-raised after the loop
            status, detail = "failed", f"{type(exc).__name__}: {exc}"
            failures.append(f"{protein_id}: {detail}")
            traceback.print_exc()
        elapsed = time.perf_counter() - protein_started
        runtime_rows.append({
            "stage": "replay_protein", "target_layer": target_layer, "protein_id": protein_id,
            "started_at": protein_wall, "finished_at": utc_now(), "elapsed_seconds": elapsed,
            "peak_rss_mb": peak_rss_mb(), "status": status, "detail": detail,
        })
        log(
            f"target_layer={target_layer} progress={number}/{n_proteins} "
            f"protein={protein_id} n_residues={outer.n_residues} status={status} elapsed={elapsed:.1f}s"
        )
        if number % checkpoint_every == 0 or status == "failed" or number == n_proteins:
            atomic_write_csv(checkpoint_path, PRIMITIVE_COLUMNS, rows)
            atomic_write_csv(runtime_path, WORKER_RUNTIME_COLUMNS, runtime_rows)
        gc.collect()

    if failures or complete != set(fold_bundle.eligible_protein_ids):
        atomic_write_csv(checkpoint_path, PRIMITIVE_COLUMNS, rows)
        atomic_write_csv(runtime_path, WORKER_RUNTIME_COLUMNS, runtime_rows)
        raise RuntimeError(f"Target layer {target_layer} incomplete; failures={failures[:10]}")
    validate_target_shard(rows, target_layer, expected_proteins=n_proteins)
    atomic_write_csv(final_path, PRIMITIVE_COLUMNS, rows)
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

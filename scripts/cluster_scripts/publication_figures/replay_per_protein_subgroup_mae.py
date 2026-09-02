#!/usr/bin/env python3
"""Replay all four protocol-2.0 models and save per-protein subgroup MAE."""

from __future__ import annotations

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

from cross_validation_common import load_cv_folds  # noqa: E402
from publication_subgroup_mae_common import (  # noqa: E402
    ANALYSIS_NAME, EXPECTED_PROTEINS, LAYERS, PER_PROTEIN_COLUMNS, RUNTIME_COLUMNS,
    subgroup_rows, validate_complete_layer,
)
from regression_output_helpers import atomic_write_csv, atomic_write_json  # noqa: E402
from regression_protocol import (  # noqa: E402
    sequence_separation_bin_codes, spatial_distance_bin_codes,
)
from triangle_violation_subgroup_replay import (  # noqa: E402
    load_layer_artifacts, load_protein, model_predictions,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def csv_root() -> Path:
    return Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))).expanduser()


def producer_root() -> Path:
    return csv_root() / "regression_5fold_v1"


def output_root() -> Path:
    return csv_root() / "analysis" / ANALYSIS_NAME


def read_checkpoint(path: Path, layer: int) -> tuple[list[dict[str, object]], set[str]]:
    if not path.is_file():
        return [], set()
    import csv
    rows: list[dict[str, object]] = []
    with path.open(newline="") as handle:
        for source in csv.DictReader(handle):
            row: dict[str, object] = dict(source)
            for column in ("feature_dim", "layer", "outer_fold", "n_residues", "n_pairs"):
                row[column] = int(source[column])
            for column in ("sum_absolute_error", "mae"):
                row[column] = float(source[column])
            rows.append(row)
    grouped: dict[str, list[dict[str, object]]] = {}
    for row in rows:
        grouped.setdefault(str(row["protein_id"]), []).append(row)
    expected = 4 * 11
    if any(len(group) != expected for group in grouped.values()):
        raise ValueError("Checkpoint contains a partial protein group")
    if rows:
        validate_complete_layer(rows, layer, expected_proteins=len(grouped))
    return rows, set(grouped)


def read_runtime(path: Path) -> list[dict[str, object]]:
    if not path.is_file():
        return []
    import csv
    with path.open(newline="") as handle:
        return [dict(row) for row in csv.DictReader(handle)]


def main() -> None:
    layer = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
    if layer not in LAYERS:
        raise ValueError(f"SLURM_ARRAY_TASK_ID must select layer 0..47; got {layer}")
    fold_bundle = load_cv_folds(metadata_dir=producer_root() / "metadata")
    if len(fold_bundle.eligible_protein_ids) != EXPECTED_PROTEINS:
        raise ValueError("Frozen fivefold cohort size changed")
    root = output_root()
    shards = root / "layer_shards"
    shards.mkdir(parents=True, exist_ok=True)
    final_path = shards / f"per_protein_subgroup_mae_layer_{layer:02d}.csv"
    checkpoint = shards / f".per_protein_subgroup_mae_layer_{layer:02d}.checkpoint.csv"
    runtime_path = shards / f"runtime_layer_{layer:02d}.csv"
    if final_path.is_file():
        rows, complete = read_checkpoint(final_path, layer)
        if len(rows) != EXPECTED_PROTEINS * 4 * 11 or len(complete) != EXPECTED_PROTEINS:
            raise ValueError("Existing final layer shard is incomplete")
        log(f"layer={layer} valid final shard already exists")
        return

    started_wall = utc_now()
    started = time.perf_counter()
    artifacts, provenance = load_layer_artifacts(layer, fold_bundle)
    provenance["analysis_name"] = ANALYSIS_NAME
    atomic_write_json(shards / f"artifact_provenance_layer_{layer:02d}.json", provenance)
    rows, complete = read_checkpoint(checkpoint, layer)
    runtime_rows: list[dict[str, object]] = read_runtime(runtime_path)
    failures: list[str] = []
    checkpoint_every = int(os.environ.get("PUBLICATION_CHECKPOINT_PROTEINS", "5"))

    for number, protein_id in enumerate(fold_bundle.eligible_protein_ids, start=1):
        if protein_id in complete:
            continue
        outer = fold_bundle.outer_by_protein[protein_id]
        protein_started_wall = utc_now()
        protein_started = time.perf_counter()
        try:
            coordinates, chain_ids, pair_i, pair_j, features = load_protein(outer, layer)
            true_distance = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
            sequence_codes = sequence_separation_bin_codes(pair_i, pair_j, chain_ids=chain_ids)
            spatial_codes = spatial_distance_bin_codes(true_distance)
            predictions, _ = model_predictions(
                outer, chain_ids, pair_i, pair_j, features, artifacts, fold_bundle,
            )
            protein_rows: list[dict[str, object]] = []
            for spec, prediction in predictions:
                protein_rows.extend(subgroup_rows(
                    spec=spec, layer=layer, protein_id=protein_id, source=outer.source,
                    outer_fold=outer.outer_fold, n_residues=outer.n_residues,
                    absolute_error=np.abs(np.asarray(prediction) - true_distance),
                    sequence_codes=sequence_codes, spatial_codes=spatial_codes,
                ))
            if len(protein_rows) != 44:
                raise RuntimeError("Completed protein did not produce exactly 44 rows")
            rows.extend(protein_rows)
            complete.add(protein_id)
            status = "complete"
        except Exception as exc:
            status = "failed"
            failures.append(f"{protein_id}: {type(exc).__name__}: {exc}")
            traceback.print_exc()
        runtime_rows.append({
            "stage": "replay_protein", "layer": layer, "protein_id": protein_id,
            "started_at": protein_started_wall, "finished_at": utc_now(),
            "elapsed_seconds": time.perf_counter() - protein_started,
            "peak_rss_mb": peak_rss_mb(), "status": status,
            "detail": "" if status == "complete" else failures[-1],
        })
        log(f"layer={layer} progress={number}/{EXPECTED_PROTEINS} protein={protein_id} status={status}")
        if number % checkpoint_every == 0 or status == "failed" or number == EXPECTED_PROTEINS:
            atomic_write_csv(checkpoint, PER_PROTEIN_COLUMNS, rows)
            atomic_write_csv(runtime_path, RUNTIME_COLUMNS, runtime_rows)
        gc.collect()

    if failures or complete != set(fold_bundle.eligible_protein_ids):
        raise RuntimeError(f"Layer {layer} incomplete; failures={failures}")
    validate_complete_layer(rows, layer)
    atomic_write_csv(final_path, PER_PROTEIN_COLUMNS, rows)
    checkpoint.unlink(missing_ok=True)
    runtime_rows.append({
        "stage": "layer_complete", "layer": layer, "protein_id": "",
        "started_at": started_wall, "finished_at": utc_now(),
        "elapsed_seconds": time.perf_counter() - started, "peak_rss_mb": peak_rss_mb(),
        "status": "complete", "detail": "",
    })
    atomic_write_csv(runtime_path, RUNTIME_COLUMNS, runtime_rows)
    log(f"layer={layer} complete elapsed={time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()

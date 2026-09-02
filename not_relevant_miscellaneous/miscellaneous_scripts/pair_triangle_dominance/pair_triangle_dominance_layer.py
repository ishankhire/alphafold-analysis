#!/usr/bin/env python3
"""Run one restart-safe pair-representation triangle-dominance layer."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Callable, Mapping

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "pair_triangle_dominance" else SCRIPT_DIR
for directory in (SCRIPT_DIR, CLUSTER_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from pair_triangle_dominance_common import (  # noqa: E402
    ANALYSIS_NAME,
    CHECKPOINT_BATCH_SIZE,
    CHECKPOINT_VERSION,
    CONFIGURATION_DIGEST,
    LAYER_STATUS_COLUMNS,
    N_CHANNELS,
    PER_PROTEIN_COLUMNS,
    PRIMARY_METRICS,
    PROTEIN_METRICS,
    RUNTIME_COLUMNS,
    SCHEMA_VERSION,
    STATUS_CODES,
    VARIANTS,
    SampleBundle,
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_json,
    default_pair_path,
    distribution_summary,
    load_sample_bundle,
    log,
    output_root,
    read_csv,
    read_json,
    reduce_protein_tensor,
    runtime_row,
    semantic_digest,
    sha256_file,
    utc_now,
)


CHECKPOINT_STATUS_COLUMNS = (
    "protein_id",
    "status",
    "failure_reason",
    "input_path",
    "input_size_bytes",
    "input_mtime_ns",
    "input_shape",
    "input_dtype",
    "n_sampled_triangles",
)


def _failure_rows(
    *, layer: int, sample_row: Mapping[str, str], status: str, reason: str
) -> list[dict[str, object]]:
    common = {
        "analysis_name": ANALYSIS_NAME,
        "schema_version": SCHEMA_VERSION,
        "layer": layer,
        "protein_id": sample_row["protein_id"],
        "source": "cameo",
        "n_residues": int(sample_row["n_residues"]),
        "n_population_triangles": int(sample_row["n_population_triangles"]),
        "n_sampled_triangles": int(sample_row["n_sampled_triangles"]),
        "sample_sha256": sample_row["sample_sha256"],
        "mean": float("nan"),
        "std": float("nan"),
        "median": float("nan"),
        "p90": float("nan"),
        "p99": float("nan"),
        "n_finite_triangles": 0,
        "zero_scale_channel_fraction": float("nan"),
        "status": status,
        "failure_reason": reason,
    }
    return [
        {**common, "representation_variant": variant, "metric": metric}
        for variant in VARIANTS
        for metric in PROTEIN_METRICS
    ]


def _complete_rows(
    *, layer: int, sample_row: Mapping[str, str], reduction: object
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for variant_index, variant in enumerate(VARIANTS):
        zero_fraction = float(
            np.mean(reduction.channel_zero_scale_fraction[variant_index])
        )
        distributions = [
            reduction.triangle_metrics[:, variant_index, metric_index]
            for metric_index in range(len(PRIMARY_METRICS))
        ] + [reduction.triangle_positive_fraction[:, variant_index]]
        for metric, values in zip(PROTEIN_METRICS, distributions):
            summary = distribution_summary(values)
            rows.append(
                {
                    "analysis_name": ANALYSIS_NAME,
                    "schema_version": SCHEMA_VERSION,
                    "layer": layer,
                    "protein_id": sample_row["protein_id"],
                    "source": "cameo",
                    "representation_variant": variant,
                    "metric": metric,
                    "n_residues": int(sample_row["n_residues"]),
                    "n_population_triangles": int(
                        sample_row["n_population_triangles"]
                    ),
                    "n_sampled_triangles": int(sample_row["n_sampled_triangles"]),
                    "sample_sha256": sample_row["sample_sha256"],
                    **summary,
                    "zero_scale_channel_fraction": zero_fraction,
                    "status": "complete",
                    "failure_reason": "",
                }
            )
    return rows


def _input_record(
    *, protein_id: str, path: Path, status: str, reason: str, n_sampled: int,
    shape: tuple[int, ...] | None = None, dtype: str = "",
) -> dict[str, object]:
    try:
        stat = path.stat()
        size, mtime = stat.st_size, stat.st_mtime_ns
    except FileNotFoundError:
        size, mtime = -1, -1
    return {
        "protein_id": protein_id,
        "status": status,
        "failure_reason": reason,
        "input_path": str(path),
        "input_size_bytes": size,
        "input_mtime_ns": mtime,
        "input_shape": "x".join(map(str, shape)) if shape else "",
        "input_dtype": dtype,
        "n_sampled_triangles": n_sampled,
    }


def _checkpoint_invariants(layer: int, bundle: SampleBundle) -> dict[str, object]:
    return {
        "checkpoint_version": CHECKPOINT_VERSION,
        "analysis_name": ANALYSIS_NAME,
        "schema_version": SCHEMA_VERSION,
        "configuration_digest": CONFIGURATION_DIGEST,
        "layer": layer,
        "protein_ids": list(bundle.protein_ids),
        "cohort_digest": bundle.provenance["cohort_digest"],
        "sample_semantic_digest": bundle.provenance["sample_semantic_digest"],
        "sample_npz_sha256": bundle.provenance["triangle_samples_npz_sha256"],
    }


def _write_checkpoint(
    checkpoint_dir: Path,
    *,
    processed_count: int,
    invariants: Mapping[str, object],
    rows: list[dict[str, object]],
    input_rows: list[dict[str, object]],
    channel_metric_mean: np.ndarray,
    channel_positive_fraction: np.ndarray,
    channel_zero_scale_fraction: np.ndarray,
    protein_status_code: np.ndarray,
) -> None:
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    stem = f"processed_{processed_count:03d}"
    metrics_path = checkpoint_dir / f"{stem}_metrics.csv"
    channels_path = checkpoint_dir / f"{stem}_channels.npz"
    status_path = checkpoint_dir / f"{stem}_status.csv"
    atomic_write_csv(metrics_path, PER_PROTEIN_COLUMNS, rows)
    atomic_write_csv(status_path, CHECKPOINT_STATUS_COLUMNS, input_rows)
    atomic_save_npz(
        channels_path,
        channel_metric_mean=channel_metric_mean,
        channel_positive_fraction=channel_positive_fraction,
        channel_zero_scale_fraction=channel_zero_scale_fraction,
        protein_status_code=protein_status_code,
    )
    artifacts = {
        "metrics": metrics_path.name,
        "channels": channels_path.name,
        "status": status_path.name,
    }
    manifest = {
        "checkpoint_version": CHECKPOINT_VERSION,
        "status": "partial",
        "processed_count": processed_count,
        "invariants": dict(invariants),
        "invariants_digest": semantic_digest(invariants),
        "artifacts": {
            key: {"filename": filename, "sha256": sha256_file(checkpoint_dir / filename)}
            for key, filename in artifacts.items()
        },
    }
    atomic_write_json(checkpoint_dir / "checkpoint.json", manifest)


def _load_checkpoint(
    checkpoint_dir: Path,
    *,
    expected_invariants: Mapping[str, object],
    n_proteins: int,
) -> tuple[
    int,
    list[dict[str, str]],
    list[dict[str, str]],
    np.ndarray,
    np.ndarray,
    np.ndarray,
    np.ndarray,
]:
    manifest_path = checkpoint_dir / "checkpoint.json"
    if not manifest_path.is_file():
        raise FileNotFoundError
    raw_manifest = read_json(manifest_path)
    if not isinstance(raw_manifest, dict):
        raise ValueError("Layer checkpoint must be a JSON object")
    manifest = dict(raw_manifest)
    if manifest.get("checkpoint_version") != CHECKPOINT_VERSION:
        raise ValueError("Layer checkpoint version mismatch")
    if manifest.get("invariants") != dict(expected_invariants):
        raise ValueError("Layer checkpoint invariant mismatch")
    if manifest.get("invariants_digest") != semantic_digest(expected_invariants):
        raise ValueError("Layer checkpoint invariant digest mismatch")
    processed = int(manifest.get("processed_count", -1))
    if processed < 0 or processed > n_proteins:
        raise ValueError("Invalid layer checkpoint processed count")

    resolved: dict[str, Path] = {}
    artifacts = manifest.get("artifacts")
    if not isinstance(artifacts, dict) or set(artifacts) != {
        "metrics", "channels", "status"
    }:
        raise ValueError("Invalid layer checkpoint artifact index")
    for key, record in artifacts.items():
        if not isinstance(record, dict):
            raise ValueError("Invalid layer checkpoint artifact record")
        path = checkpoint_dir / str(record.get("filename", ""))
        if not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"Corrupted layer checkpoint artifact: {key}")
        resolved[key] = path

    rows = read_csv(resolved["metrics"], PER_PROTEIN_COLUMNS)
    input_rows = read_csv(resolved["status"], CHECKPOINT_STATUS_COLUMNS)
    if len(rows) != processed * len(VARIANTS) * len(PROTEIN_METRICS):
        raise ValueError("Checkpoint metric row count does not match processed proteins")
    if len(input_rows) != processed:
        raise ValueError("Checkpoint status row count does not match processed proteins")
    with np.load(resolved["channels"], allow_pickle=False) as archive:
        if set(archive.files) != {
            "channel_metric_mean",
            "channel_positive_fraction",
            "channel_zero_scale_fraction",
            "protein_status_code",
        }:
            raise ValueError("Unexpected checkpoint channel arrays")
        metric_mean = archive["channel_metric_mean"].copy()
        positive = archive["channel_positive_fraction"].copy()
        zero = archive["channel_zero_scale_fraction"].copy()
        status_codes = archive["protein_status_code"].copy()
    expected_metric_shape = (n_proteins, len(VARIANTS), N_CHANNELS, len(PRIMARY_METRICS))
    expected_diag_shape = (n_proteins, len(VARIANTS), N_CHANNELS)
    if metric_mean.shape != expected_metric_shape:
        raise ValueError("Checkpoint channel metric shape mismatch")
    if positive.shape != expected_diag_shape or zero.shape != expected_diag_shape:
        raise ValueError("Checkpoint channel diagnostic shape mismatch")
    if status_codes.shape != (n_proteins,):
        raise ValueError("Checkpoint protein status shape mismatch")
    return processed, rows, input_rows, metric_mean, positive, zero, status_codes


def _cleanup_checkpoint(checkpoint_dir: Path) -> None:
    if not checkpoint_dir.is_dir() or checkpoint_dir.name[:6] != "layer_":
        return
    for path in checkpoint_dir.iterdir():
        if path.is_file():
            path.unlink()
    checkpoint_dir.rmdir()
    parent = checkpoint_dir.parent
    if parent.name == ".checkpoints" and parent.is_dir() and not any(parent.iterdir()):
        parent.rmdir()


def _validate_existing_complete_layer(
    root: Path, layer: int, invariants: Mapping[str, object]
) -> bool:
    shard_dir = root / "layer_shards"
    status_path = shard_dir / f"layer_status_{layer:02d}.csv"
    if not status_path.is_file():
        return False
    rows = read_csv(status_path, LAYER_STATUS_COLUMNS)
    if len(rows) != 1:
        raise ValueError(f"Layer {layer} status shard must contain exactly one row")
    row = rows[0]
    for field in (
        "configuration_digest",
        "cohort_digest",
        "sample_semantic_digest",
        "sample_npz_sha256",
    ):
        if row[field] != str(invariants[field]):
            raise ValueError(f"Existing layer {layer} invariant mismatch: {field}")
    if row["status"] != "complete":
        return False
    artifacts = {
        "per_protein_metrics_sha256": shard_dir
        / f"per_protein_metrics_layer_{layer:02d}.csv",
        "channel_metrics_sha256": shard_dir
        / f"per_protein_channel_metrics_layer_{layer:02d}.npz",
        "runtime_sha256": shard_dir / f"runtime_layer_{layer:02d}.csv",
    }
    for field, path in artifacts.items():
        if not path.is_file() or sha256_file(path) != row[field]:
            raise ValueError(f"Existing complete layer {layer} has corrupt artifact {path}")
    return True


def run_layer(
    layer: int,
    destination: Path,
    *,
    pair_path_resolver: Callable[[str, int], Path] = default_pair_path,
    expected_protein_count: int = 154,
    checkpoint_batch_size: int = CHECKPOINT_BATCH_SIZE,
    fail_on_incomplete: bool = True,
) -> dict[str, object]:
    if layer not in range(48):
        raise ValueError(f"Layer must lie in 0..47; observed {layer}")
    if checkpoint_batch_size <= 0:
        raise ValueError("checkpoint_batch_size must be positive")
    started_at = utc_now()
    started = time.perf_counter()
    destination = Path(destination)
    shard_dir = destination / "layer_shards"
    shard_dir.mkdir(parents=True, exist_ok=True)
    bundle = load_sample_bundle(
        destination, expected_protein_count=expected_protein_count
    )
    invariants = _checkpoint_invariants(layer, bundle)
    checkpoint_dir = destination / ".checkpoints" / f"layer_{layer:02d}"
    if _validate_existing_complete_layer(destination, layer, invariants):
        _cleanup_checkpoint(checkpoint_dir)
        log(f"Layer {layer}: validated complete final shards; skipping")
        return {"layer": layer, "status": "complete", "skipped": True}

    n_proteins = len(bundle.protein_ids)
    metric_shape = (n_proteins, len(VARIANTS), N_CHANNELS, len(PRIMARY_METRICS))
    diagnostic_shape = (n_proteins, len(VARIANTS), N_CHANNELS)
    channel_metric_mean = np.full(metric_shape, np.nan, dtype=np.float64)
    channel_positive_fraction = np.full(diagnostic_shape, np.nan, dtype=np.float64)
    channel_zero_scale_fraction = np.full(diagnostic_shape, np.nan, dtype=np.float64)
    protein_status_code = np.full(
        n_proteins, STATUS_CODES["not_processed"], dtype=np.int8
    )
    rows: list[dict[str, object]] = []
    input_rows: list[dict[str, object]] = []
    processed_count = 0
    if (checkpoint_dir / "checkpoint.json").is_file():
        (
            processed_count,
            loaded_rows,
            loaded_inputs,
            channel_metric_mean,
            channel_positive_fraction,
            channel_zero_scale_fraction,
            protein_status_code,
        ) = _load_checkpoint(
            checkpoint_dir,
            expected_invariants=invariants,
            n_proteins=n_proteins,
        )
        rows = list(loaded_rows)
        input_rows = list(loaded_inputs)
        log(f"Layer {layer}: resumed after {processed_count}/{n_proteins} proteins")

    for protein_index in range(processed_count, n_proteins):
        protein_id = bundle.protein_ids[protein_index]
        sample_row = bundle.sample_rows[protein_index]
        n_residues = int(sample_row["n_residues"])
        n_sampled = int(sample_row["n_sampled_triangles"])
        path = Path(pair_path_resolver(protein_id, layer))
        status, reason = "complete", ""
        shape: tuple[int, ...] | None = None
        dtype = ""

        if n_sampled == 0:
            status = "insufficient_residues"
            reason = "Protein has fewer than three residues"
            rows.extend(
                _failure_rows(
                    layer=layer, sample_row=sample_row, status=status, reason=reason
                )
            )
        elif not path.is_file():
            status = "missing_tensor"
            reason = f"Pair tensor not found: {path}"
            rows.extend(
                _failure_rows(
                    layer=layer, sample_row=sample_row, status=status, reason=reason
                )
            )
        else:
            try:
                tensor = np.load(path)
                shape, dtype = tuple(tensor.shape), str(tensor.dtype)
                expected_shape = (n_residues, n_residues, N_CHANNELS)
                if shape != expected_shape:
                    status = "invalid_tensor"
                    reason = f"Expected tensor shape {expected_shape}; observed {shape}"
                    rows.extend(
                        _failure_rows(
                            layer=layer,
                            sample_row=sample_row,
                            status=status,
                            reason=reason,
                        )
                    )
                elif not np.issubdtype(tensor.dtype, np.number):
                    status = "invalid_tensor"
                    reason = f"Pair tensor has nonnumeric dtype {tensor.dtype}"
                    rows.extend(
                        _failure_rows(
                            layer=layer,
                            sample_row=sample_row,
                            status=status,
                            reason=reason,
                        )
                    )
                elif not np.all(np.isfinite(tensor)):
                    status = "nonfinite_tensor"
                    reason = "Pair tensor contains nonfinite values"
                    rows.extend(
                        _failure_rows(
                            layer=layer,
                            sample_row=sample_row,
                            status=status,
                            reason=reason,
                        )
                    )
                else:
                    reduction = reduce_protein_tensor(
                        tensor, bundle.triples_for(protein_index)
                    )
                    rows.extend(
                        _complete_rows(
                            layer=layer, sample_row=sample_row, reduction=reduction
                        )
                    )
                    channel_metric_mean[protein_index] = reduction.channel_metric_mean
                    channel_positive_fraction[protein_index] = (
                        reduction.channel_positive_fraction
                    )
                    channel_zero_scale_fraction[protein_index] = (
                        reduction.channel_zero_scale_fraction
                    )
            except (OSError, ValueError) as exc:
                if status == "complete":
                    status = "processing_error"
                    reason = f"{type(exc).__name__}: {exc}"
                    rows.extend(
                        _failure_rows(
                            layer=layer,
                            sample_row=sample_row,
                            status=status,
                            reason=reason,
                        )
                    )

        protein_status_code[protein_index] = STATUS_CODES[status]
        input_rows.append(
            _input_record(
                protein_id=protein_id,
                path=path,
                status=status,
                reason=reason,
                n_sampled=n_sampled,
                shape=shape,
                dtype=dtype,
            )
        )
        processed_count = protein_index + 1
        if (
            processed_count % checkpoint_batch_size == 0
            or processed_count == n_proteins
        ):
            _write_checkpoint(
                checkpoint_dir,
                processed_count=processed_count,
                invariants=invariants,
                rows=rows,
                input_rows=input_rows,
                channel_metric_mean=channel_metric_mean,
                channel_positive_fraction=channel_positive_fraction,
                channel_zero_scale_fraction=channel_zero_scale_fraction,
                protein_status_code=protein_status_code,
            )
            log(f"Layer {layer}: checkpointed {processed_count}/{n_proteins} proteins")

    expected_rows = n_proteins * len(VARIANTS) * len(PROTEIN_METRICS)
    if len(rows) != expected_rows:
        raise ValueError(f"Layer {layer} produced {len(rows)} rows; expected {expected_rows}")
    expected_keys = [
        (layer, protein_id, variant, metric)
        for protein_id in bundle.protein_ids
        for variant in VARIANTS
        for metric in PROTEIN_METRICS
    ]
    observed_keys = [
        (
            int(row["layer"]),
            str(row["protein_id"]),
            str(row["representation_variant"]),
            str(row["metric"]),
        )
        for row in rows
    ]
    if observed_keys != expected_keys:
        raise ValueError("Layer metric rows are not in deterministic contract order")

    metrics_path = shard_dir / f"per_protein_metrics_layer_{layer:02d}.csv"
    channels_path = shard_dir / f"per_protein_channel_metrics_layer_{layer:02d}.npz"
    runtime_path = shard_dir / f"runtime_layer_{layer:02d}.csv"
    status_path = shard_dir / f"layer_status_{layer:02d}.csv"
    atomic_write_csv(metrics_path, PER_PROTEIN_COLUMNS, rows)
    atomic_save_npz(
        channels_path,
        channel_metric_mean=channel_metric_mean,
        channel_positive_fraction=channel_positive_fraction,
        channel_zero_scale_fraction=channel_zero_scale_fraction,
        protein_status_code=protein_status_code,
        protein_ids=np.asarray(bundle.protein_ids),
        variant_order=np.asarray(VARIANTS),
        primary_metric_order=np.asarray(PRIMARY_METRICS),
        status_code_labels=np.asarray(tuple(STATUS_CODES)),
        status_code_values=np.asarray(tuple(STATUS_CODES.values()), dtype=np.int8),
        layer=np.asarray(layer, dtype=np.int16),
        configuration_digest=np.asarray(CONFIGURATION_DIGEST),
        cohort_digest=np.asarray(bundle.provenance["cohort_digest"]),
        sample_semantic_digest=np.asarray(
            bundle.provenance["sample_semantic_digest"]
        ),
        sample_npz_sha256=np.asarray(
            bundle.provenance["triangle_samples_npz_sha256"]
        ),
    )
    completed = int(np.sum(protein_status_code == STATUS_CODES["complete"]))
    layer_status = "complete" if completed == n_proteins else "incomplete"
    finished_at = utc_now()
    runtime = runtime_row(
        stage="layer",
        layer=layer,
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=time.perf_counter() - started,
        processed_proteins=n_proteins,
        processed_sampled_triangles=sum(
            int(row["n_sampled_triangles"]) for row in bundle.sample_rows
        ),
        status=layer_status,
    )
    atomic_write_csv(runtime_path, RUNTIME_COLUMNS, [runtime])
    normalized_input_rows = [
        {column: str(row[column]) for column in CHECKPOINT_STATUS_COLUMNS}
        for row in input_rows
    ]
    input_identity_digest = semantic_digest(normalized_input_rows)
    status_row = {
        "analysis_name": ANALYSIS_NAME,
        "schema_version": SCHEMA_VERSION,
        "layer": layer,
        "n_proteins_expected": n_proteins,
        "n_proteins_completed": completed,
        "n_proteins_failed": n_proteins - completed,
        "n_variants_expected": len(VARIANTS),
        "n_channels_expected": N_CHANNELS,
        "configuration_digest": CONFIGURATION_DIGEST,
        "cohort_digest": bundle.provenance["cohort_digest"],
        "sample_semantic_digest": bundle.provenance["sample_semantic_digest"],
        "sample_npz_sha256": bundle.provenance["triangle_samples_npz_sha256"],
        "input_identity_digest": input_identity_digest,
        "per_protein_metrics_sha256": sha256_file(metrics_path),
        "channel_metrics_sha256": sha256_file(channels_path),
        "runtime_sha256": sha256_file(runtime_path),
        "status": layer_status,
    }
    atomic_write_csv(status_path, LAYER_STATUS_COLUMNS, [status_row])
    _cleanup_checkpoint(checkpoint_dir)
    log(
        f"Layer {layer}: completed={completed}/{n_proteins}; status={layer_status}; "
        f"elapsed={runtime['elapsed_seconds']:.1f}s"
    )
    if fail_on_incomplete and layer_status != "complete":
        raise RuntimeError(
            f"Layer {layer} is incomplete: {n_proteins - completed} protein(s) failed"
        )
    return status_row


def main() -> None:
    raw_layer = os.environ.get("SLURM_ARRAY_TASK_ID", os.environ.get("LAYER"))
    if raw_layer is None:
        raise RuntimeError("Set SLURM_ARRAY_TASK_ID or LAYER to a value in 0..47")
    run_layer(int(raw_layer), output_root())


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Strictly validate and consolidate all pair-triangle layer shards."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "pair_triangle_dominance" else SCRIPT_DIR
for directory in (SCRIPT_DIR, CLUSTER_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from pair_triangle_dominance_common import (  # noqa: E402
    ANALYSIS_NAME,
    CHANNEL_METRICS,
    CHANNEL_SUMMARY_COLUMNS,
    CONFIGURATION_DIGEST,
    LAYERS,
    LAYER_STATUS_COLUMNS,
    N_CHANNELS,
    PER_PROTEIN_COLUMNS,
    PRIMARY_METRICS,
    PROTEIN_METRICS,
    RUNTIME_COLUMNS,
    SCHEMA_VERSION,
    STATUS_CODES,
    SUMMARY_COLUMNS,
    VARIANTS,
    SampleBundle,
    atomic_write_csv,
    atomic_write_json,
    configuration_record,
    load_sample_bundle,
    log,
    output_root,
    protein_equal_summary,
    read_csv,
    runtime_row,
    sha256_file,
    utc_now,
)


def _scalar_string(archive: np.lib.npyio.NpzFile, name: str) -> str:
    value = archive[name]
    if value.shape != ():
        raise ValueError(f"NPZ metadata {name} must be scalar")
    return str(value.item())


def validate_layer_shard(
    destination: Path, layer: int, bundle: SampleBundle
) -> tuple[list[dict[str, str]], dict[str, np.ndarray], dict[str, str], dict[str, str]]:
    shard = destination / "layer_shards"
    metrics_path = shard / f"per_protein_metrics_layer_{layer:02d}.csv"
    channels_path = shard / f"per_protein_channel_metrics_layer_{layer:02d}.npz"
    status_path = shard / f"layer_status_{layer:02d}.csv"
    runtime_path = shard / f"runtime_layer_{layer:02d}.csv"
    for path in (metrics_path, channels_path, status_path, runtime_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing layer {layer} shard: {path}")

    status_rows = read_csv(status_path, LAYER_STATUS_COLUMNS)
    if len(status_rows) != 1:
        raise ValueError(f"Layer {layer} status shard must have one row")
    status = status_rows[0]
    expected_invariants = {
        "configuration_digest": CONFIGURATION_DIGEST,
        "cohort_digest": str(bundle.provenance["cohort_digest"]),
        "sample_semantic_digest": str(bundle.provenance["sample_semantic_digest"]),
        "sample_npz_sha256": str(bundle.provenance["triangle_samples_npz_sha256"]),
    }
    if status["analysis_name"] != ANALYSIS_NAME or status["schema_version"] != SCHEMA_VERSION:
        raise ValueError(f"Layer {layer} status identity mismatch")
    if int(status["layer"]) != layer:
        raise ValueError(f"Layer {layer} status records the wrong layer")
    for field, expected in expected_invariants.items():
        if status[field] != expected:
            raise ValueError(f"Layer {layer} invariant mismatch: {field}")
    if int(status["n_proteins_expected"]) != len(bundle.protein_ids):
        raise ValueError(f"Layer {layer} expected-protein count mismatch")
    if int(status["n_proteins_completed"]) != len(bundle.protein_ids):
        raise ValueError(f"Layer {layer} is incomplete")
    if int(status["n_proteins_failed"]) != 0 or status["status"] != "complete":
        raise ValueError(f"Layer {layer} has failed proteins")
    if int(status["n_variants_expected"]) != len(VARIANTS):
        raise ValueError(f"Layer {layer} variant count mismatch")
    if int(status["n_channels_expected"]) != N_CHANNELS:
        raise ValueError(f"Layer {layer} channel count mismatch")
    expected_hashes = {
        "per_protein_metrics_sha256": sha256_file(metrics_path),
        "channel_metrics_sha256": sha256_file(channels_path),
        "runtime_sha256": sha256_file(runtime_path),
    }
    for field, observed in expected_hashes.items():
        if status[field] != observed:
            raise ValueError(f"Layer {layer} artifact hash mismatch: {field}")

    rows = read_csv(metrics_path, PER_PROTEIN_COLUMNS)
    expected_row_count = len(bundle.protein_ids) * len(VARIANTS) * len(PROTEIN_METRICS)
    if len(rows) != expected_row_count:
        raise ValueError(
            f"Layer {layer} has {len(rows)} metric rows; expected {expected_row_count}"
        )
    expected_keys = [
        (layer, protein_id, variant, metric)
        for protein_id in bundle.protein_ids
        for variant in VARIANTS
        for metric in PROTEIN_METRICS
    ]
    observed_keys = [
        (
            int(row["layer"]),
            row["protein_id"],
            row["representation_variant"],
            row["metric"],
        )
        for row in rows
    ]
    if observed_keys != expected_keys:
        raise ValueError(f"Layer {layer} metric key order or coverage mismatch")
    sample_by_id = {row["protein_id"]: row for row in bundle.sample_rows}
    for row in rows:
        sample = sample_by_id[row["protein_id"]]
        if row["analysis_name"] != ANALYSIS_NAME or row["schema_version"] != SCHEMA_VERSION:
            raise ValueError(f"Layer {layer} metric identity mismatch")
        if row["status"] != "complete" or row["failure_reason"]:
            raise ValueError(f"Layer {layer} has an incomplete protein metric row")
        if int(row["n_residues"]) != int(sample["n_residues"]):
            raise ValueError(f"Layer {layer} residue count mismatch")
        if int(row["n_sampled_triangles"]) != int(sample["n_sampled_triangles"]):
            raise ValueError(f"Layer {layer} sample count mismatch")
        if row["sample_sha256"] != sample["sample_sha256"]:
            raise ValueError(f"Layer {layer} protein sample hash mismatch")
        if int(row["n_finite_triangles"]) != int(sample["n_sampled_triangles"]):
            raise ValueError(f"Layer {layer} finite triangle count mismatch")
        numeric = [
            float(row[column])
            for column in ("mean", "std", "median", "p90", "p99")
        ]
        if not np.all(np.isfinite(numeric)):
            raise ValueError(f"Layer {layer} has nonfinite complete metric summaries")

    required_arrays = {
        "channel_metric_mean",
        "channel_positive_fraction",
        "channel_zero_scale_fraction",
        "protein_status_code",
        "protein_ids",
        "variant_order",
        "primary_metric_order",
        "status_code_labels",
        "status_code_values",
        "layer",
        "configuration_digest",
        "cohort_digest",
        "sample_semantic_digest",
        "sample_npz_sha256",
    }
    with np.load(channels_path, allow_pickle=False) as archive:
        if set(archive.files) != required_arrays:
            raise ValueError(f"Layer {layer} channel shard array contract mismatch")
        metric_mean = archive["channel_metric_mean"].copy()
        positive = archive["channel_positive_fraction"].copy()
        zero = archive["channel_zero_scale_fraction"].copy()
        status_codes = archive["protein_status_code"].copy()
        if tuple(map(str, archive["protein_ids"])) != bundle.protein_ids:
            raise ValueError(f"Layer {layer} channel protein order mismatch")
        if tuple(map(str, archive["variant_order"])) != VARIANTS:
            raise ValueError(f"Layer {layer} channel variant order mismatch")
        if tuple(map(str, archive["primary_metric_order"])) != PRIMARY_METRICS:
            raise ValueError(f"Layer {layer} channel metric order mismatch")
        if int(archive["layer"].item()) != layer:
            raise ValueError(f"Layer {layer} NPZ layer mismatch")
        for field, expected in expected_invariants.items():
            if _scalar_string(archive, field) != expected:
                raise ValueError(f"Layer {layer} NPZ invariant mismatch: {field}")
    expected_metric_shape = (
        len(bundle.protein_ids),
        len(VARIANTS),
        N_CHANNELS,
        len(PRIMARY_METRICS),
    )
    expected_diag_shape = (len(bundle.protein_ids), len(VARIANTS), N_CHANNELS)
    if metric_mean.shape != expected_metric_shape:
        raise ValueError(f"Layer {layer} channel metric shape mismatch")
    if positive.shape != expected_diag_shape or zero.shape != expected_diag_shape:
        raise ValueError(f"Layer {layer} channel diagnostic shape mismatch")
    if status_codes.shape != (len(bundle.protein_ids),) or not np.all(
        status_codes == STATUS_CODES["complete"]
    ):
        raise ValueError(f"Layer {layer} channel status coverage mismatch")
    if not all(np.all(np.isfinite(array)) for array in (metric_mean, positive, zero)):
        raise ValueError(f"Layer {layer} complete channel shard contains nonfinite values")
    tolerance = 64 * np.finfo(np.float64).eps
    if np.any((positive < -tolerance) | (positive > 1.0 + tolerance)):
        raise ValueError(f"Layer {layer} positive fractions outside [0,1]")
    if np.any((zero < -tolerance) | (zero > 1.0 + tolerance)):
        raise ValueError(f"Layer {layer} zero-scale fractions outside [0,1]")

    # The protein-level mean triangle total must equal the channel means by linearity.
    row_lookup = {
        (row["protein_id"], row["representation_variant"], row["metric"]): float(
            row["mean"]
        )
        for row in rows
    }
    for protein_index, protein_id in enumerate(bundle.protein_ids):
        for variant_index, variant in enumerate(VARIANTS):
            for metric_index, metric in enumerate(PRIMARY_METRICS):
                observed = float(np.mean(metric_mean[protein_index, variant_index, :, metric_index]))
                expected = row_lookup[(protein_id, variant, metric)]
                if not np.isclose(observed, expected, rtol=2e-12, atol=2e-12):
                    raise ValueError(
                        f"Layer {layer} channel/protein mean disagreement for "
                        f"{protein_id}/{variant}/{metric}"
                    )
            observed_positive = float(np.mean(positive[protein_index, variant_index]))
            expected_positive = row_lookup[
                (protein_id, variant, "positive_channel_fraction")
            ]
            if not np.isclose(observed_positive, expected_positive, rtol=2e-12, atol=2e-12):
                raise ValueError(f"Layer {layer} positive-frequency mean disagreement")

    runtime_rows = read_csv(runtime_path, RUNTIME_COLUMNS)
    if len(runtime_rows) != 1 or runtime_rows[0]["stage"] != "layer":
        raise ValueError(f"Layer {layer} runtime shard mismatch")
    if runtime_rows[0]["status"] != "complete":
        raise ValueError(f"Layer {layer} runtime is not complete")
    arrays = {
        "channel_metric_mean": metric_mean,
        "channel_positive_fraction": positive,
        "channel_zero_scale_fraction": zero,
    }
    return rows, arrays, status, runtime_rows[0]


def aggregate_results(
    destination: Path,
    *,
    layers: Sequence[int] = LAYERS,
    expected_protein_count: int = 154,
) -> dict[str, object]:
    started_at = utc_now()
    started = time.perf_counter()
    destination = Path(destination)
    if tuple(sorted(layers)) != tuple(layers) or len(set(layers)) != len(layers):
        raise ValueError("layers must be unique and sorted")
    bundle = load_sample_bundle(
        destination, expected_protein_count=expected_protein_count
    )
    all_rows: list[dict[str, str]] = []
    all_status: list[dict[str, str]] = []
    layer_runtimes: list[dict[str, str]] = []
    arrays_by_layer: dict[int, dict[str, np.ndarray]] = {}
    for layer in layers:
        rows, arrays, status, runtime = validate_layer_shard(
            destination, layer, bundle
        )
        all_rows.extend(rows)
        arrays_by_layer[layer] = arrays
        all_status.append(status)
        layer_runtimes.append(runtime)
        log(f"Aggregation validated layer {layer}")

    expected_total = (
        len(layers) * expected_protein_count * len(VARIANTS) * len(PROTEIN_METRICS)
    )
    if len(all_rows) != expected_total:
        raise ValueError(f"Consolidated row count {len(all_rows)} != {expected_total}")
    atomic_write_csv(
        destination / "per_protein_metrics.csv", PER_PROTEIN_COLUMNS, all_rows
    )

    layer_summary: list[dict[str, object]] = []
    for layer in layers:
        layer_rows = [row for row in all_rows if int(row["layer"]) == layer]
        for variant in VARIANTS:
            for metric in PROTEIN_METRICS:
                values = [
                    float(row["mean"])
                    for row in layer_rows
                    if row["representation_variant"] == variant
                    and row["metric"] == metric
                ]
                layer_summary.append(
                    {
                        "analysis_name": ANALYSIS_NAME,
                        "schema_version": SCHEMA_VERSION,
                        "layer": layer,
                        "representation_variant": variant,
                        "metric": metric,
                        **protein_equal_summary(values, expected_protein_count),
                    }
                )
    atomic_write_csv(destination / "layer_summary.csv", SUMMARY_COLUMNS, layer_summary)

    channel_summary: list[dict[str, object]] = []
    for layer in layers:
        arrays = arrays_by_layer[layer]
        for variant_index, variant in enumerate(VARIANTS):
            metric_arrays = [
                arrays["channel_metric_mean"][:, variant_index, :, metric_index]
                for metric_index in range(len(PRIMARY_METRICS))
            ] + [
                arrays["channel_positive_fraction"][:, variant_index, :],
                arrays["channel_zero_scale_fraction"][:, variant_index, :],
            ]
            for channel in range(N_CHANNELS):
                for metric, values in zip(CHANNEL_METRICS, metric_arrays):
                    channel_summary.append(
                        {
                            "analysis_name": ANALYSIS_NAME,
                            "schema_version": SCHEMA_VERSION,
                            "layer": layer,
                            "representation_variant": variant,
                            "channel": channel,
                            "metric": metric,
                            **protein_equal_summary(
                                values[:, channel], expected_protein_count
                            ),
                        }
                    )
    atomic_write_csv(
        destination / "channel_summary.csv",
        CHANNEL_SUMMARY_COLUMNS,
        channel_summary,
    )
    atomic_write_csv(destination / "layer_status.csv", LAYER_STATUS_COLUMNS, all_status)

    existing_runtime = read_csv(destination / "runtime.csv", RUNTIME_COLUMNS)
    preparation_rows = [row for row in existing_runtime if row["stage"] == "preparation"]
    if len(preparation_rows) != 1 or preparation_rows[0]["status"] != "complete":
        raise ValueError("Expected exactly one complete preparation runtime row")
    aggregation_runtime = runtime_row(
        stage="aggregation",
        layer="",
        started_at=started_at,
        finished_at=utc_now(),
        elapsed_seconds=time.perf_counter() - started,
        processed_proteins=expected_protein_count * len(layers),
        processed_sampled_triangles=sum(
            int(row["n_sampled_triangles"]) for row in bundle.sample_rows
        )
        * len(layers),
        status="complete",
    )
    runtime_rows: list[Mapping[str, object]] = [
        preparation_rows[0],
        *layer_runtimes,
        aggregation_runtime,
    ]
    atomic_write_csv(destination / "runtime.csv", RUNTIME_COLUMNS, runtime_rows)

    output_paths = [
        "per_protein_metrics.csv",
        "layer_summary.csv",
        "channel_summary.csv",
        "layer_status.csv",
        "runtime.csv",
        "metadata/cameo_cohort.csv",
        "metadata/sample_manifest.csv",
        "metadata/triangle_samples.npz",
        "metadata/sample_provenance.json",
    ]
    for layer in layers:
        output_paths.extend(
            [
                f"layer_shards/per_protein_metrics_layer_{layer:02d}.csv",
                f"layer_shards/per_protein_channel_metrics_layer_{layer:02d}.npz",
                f"layer_shards/layer_status_{layer:02d}.csv",
                f"layer_shards/runtime_layer_{layer:02d}.csv",
            ]
        )
    source_hashes = {}
    for filename in (
        "pair_triangle_dominance_common.py",
        "prepare_pair_triangle_samples.py",
        "pair_triangle_dominance_layer.py",
        "aggregate_pair_triangle_dominance.py",
    ):
        path = SCRIPT_DIR / filename
        if path.is_file():
            source_hashes[filename] = sha256_file(path)
    manifest = {
        **configuration_record(),
        "cohort_digest": bundle.provenance["cohort_digest"],
        "sample_semantic_digest": bundle.provenance["sample_semantic_digest"],
        "sample_npz_sha256": bundle.provenance["triangle_samples_npz_sha256"],
        "protein_count": expected_protein_count,
        "layers_aggregated": list(layers),
        "per_protein_row_count": len(all_rows),
        "layer_summary_row_count": len(layer_summary),
        "channel_summary_row_count": len(channel_summary),
        "aggregation_completed_at": utc_now(),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
        "source_sha256": source_hashes,
        "output_sha256": {
            relative: sha256_file(destination / relative) for relative in output_paths
        },
        "scientific_limitations": [
            "Pair channels are signed and basis-dependent rather than physical lengths.",
            "Positive directional_raw margins are algebraic dominance, not metric violations.",
            "The analysis is descriptive and does not establish a learned metric space.",
            "Canonical-direction variants depend on residue ordering.",
        ],
        "storage_policy": "No pair, triangle, or channel-by-triangle measurement rows are saved.",
    }
    atomic_write_json(destination / "run_manifest.json", manifest)
    log(
        f"Aggregated {len(layers)} layers: {len(all_rows)} protein rows and "
        f"{len(channel_summary)} channel rows"
    )
    return manifest


def main() -> None:
    aggregate_results(output_root())


if __name__ == "__main__":
    main()

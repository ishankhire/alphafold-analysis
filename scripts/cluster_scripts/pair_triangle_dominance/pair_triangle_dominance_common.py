#!/usr/bin/env python3
"""Shared contract and numerical primitives for pair-triangle dominance."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import resource
import sys
import tempfile
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


ANALYSIS_NAME = "pair_representation_triangle_dominance"
SCHEMA_VERSION = "1.0"
SAMPLE_VERSION = "pair_triangle_samples_v1"
CONFIGURATION_VERSION = "pair_triangle_dominance_v1"
CHECKPOINT_VERSION = "pair_triangle_layer_checkpoint_v1"
BASE_SEED = 20260827
SAMPLE_CAP = 20_000
CHUNK_SIZE = 2_000
CHECKPOINT_BATCH_SIZE = 5
N_CHANNELS = 128
LAYERS = tuple(range(48))

VARIANTS = (
    "directional_raw",
    "directional_abs",
    "abs_then_sym",
    "sym_then_abs",
)
PRIMARY_METRICS = (
    "signed_margin",
    "positive_violation",
    "signed_normalized_margin",
    "positive_normalized_violation",
)
PROTEIN_METRICS = PRIMARY_METRICS + ("positive_channel_fraction",)
CHANNEL_METRICS = PRIMARY_METRICS + (
    "positive_channel_fraction",
    "zero_scale_fraction",
)

STATUS_CODES = {
    "not_processed": 0,
    "complete": 1,
    "insufficient_residues": 2,
    "missing_tensor": 3,
    "invalid_tensor": 4,
    "nonfinite_tensor": 5,
    "sample_mismatch": 6,
    "processing_error": 7,
}

COHORT_COLUMNS = (
    "protein_id",
    "source",
    "manifest_version",
    "manifest_eligibility_status",
    "expected_residue_count",
    "cohort_status",
)
SAMPLE_MANIFEST_COLUMNS = (
    "sample_version",
    "protein_id",
    "n_residues",
    "n_population_triangles",
    "n_sampled_triangles",
    "base_seed",
    "derived_seed",
    "rank_order",
    "unranking_algorithm",
    "rng_algorithm",
    "sample_offset_start",
    "sample_offset_stop",
    "sample_sha256",
    "status",
)
PER_PROTEIN_COLUMNS = (
    "analysis_name",
    "schema_version",
    "layer",
    "protein_id",
    "source",
    "representation_variant",
    "metric",
    "n_residues",
    "n_population_triangles",
    "n_sampled_triangles",
    "sample_sha256",
    "mean",
    "std",
    "median",
    "p90",
    "p99",
    "n_finite_triangles",
    "zero_scale_channel_fraction",
    "status",
    "failure_reason",
)
SUMMARY_COLUMNS = (
    "analysis_name",
    "schema_version",
    "layer",
    "representation_variant",
    "metric",
    "protein_equal_mean",
    "protein_std",
    "protein_median",
    "protein_q25",
    "protein_q75",
    "n_proteins_attempted",
    "n_proteins_finite",
    "support_status",
)
CHANNEL_SUMMARY_COLUMNS = (
    "analysis_name",
    "schema_version",
    "layer",
    "representation_variant",
    "channel",
    "metric",
    *SUMMARY_COLUMNS[5:],
)
LAYER_STATUS_COLUMNS = (
    "analysis_name",
    "schema_version",
    "layer",
    "n_proteins_expected",
    "n_proteins_completed",
    "n_proteins_failed",
    "n_variants_expected",
    "n_channels_expected",
    "configuration_digest",
    "cohort_digest",
    "sample_semantic_digest",
    "sample_npz_sha256",
    "input_identity_digest",
    "per_protein_metrics_sha256",
    "channel_metrics_sha256",
    "runtime_sha256",
    "status",
)
RUNTIME_COLUMNS = (
    "analysis_name",
    "stage",
    "layer",
    "started_at",
    "finished_at",
    "elapsed_seconds",
    "requested_cpus",
    "requested_memory",
    "requested_walltime",
    "processed_proteins",
    "processed_sampled_triangles",
    "peak_rss_mb",
    "status",
)


def canonical_json_bytes(value: object) -> bytes:
    return json.dumps(
        value, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")


def semantic_digest(value: object) -> str:
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_file(path: str | Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def output_root() -> Path:
    explicit = os.environ.get("TRIANGLE_OUTPUT_DIR")
    if explicit:
        return Path(explicit).expanduser()
    csv_root = Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files")))
    return csv_root.expanduser() / ANALYSIS_NAME


def configuration_record() -> dict[str, object]:
    return {
        "analysis_name": ANALYSIS_NAME,
        "schema_version": SCHEMA_VERSION,
        "configuration_version": CONFIGURATION_VERSION,
        "sample_version": SAMPLE_VERSION,
        "base_seed": BASE_SEED,
        "sample_cap": SAMPLE_CAP,
        "chunk_size": CHUNK_SIZE,
        "n_channels": N_CHANNELS,
        "layers": list(LAYERS),
        "variants": list(VARIANTS),
        "primary_metrics": list(PRIMARY_METRICS),
        "protein_metrics": list(PROTEIN_METRICS),
        "channel_metrics": list(CHANNEL_METRICS),
        "direction": "canonical_strict_lower_i_gt_j",
        "margin_formula": "2*max(a,b,d)-(a+b+d)",
        "normalizer": "abs(a)+abs(b)+abs(d)",
        "zero_scale_rule": "normalized_value_zero",
        "positive_boundary": "margin_gt_zero",
        "accumulation_dtype": "float64",
    }


CONFIGURATION_DIGEST = semantic_digest(configuration_record())


def atomic_write_json(path: str | Path, value: object) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as handle:
            json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_write_csv(
    path: str | Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]
) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp", dir=destination.parent
    )
    try:
        with os.fdopen(fd, "w", newline="", encoding="utf-8") as handle:
            writer = csv.DictWriter(handle, fieldnames=columns, extrasaction="raise")
            writer.writeheader()
            writer.writerows(rows)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def atomic_save_npz(path: str | Path, **arrays: np.ndarray) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.", suffix=".tmp.npz", dir=destination.parent
    )
    os.close(fd)
    try:
        np.savez_compressed(temporary_name, **arrays)
        os.replace(temporary_name, destination)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def read_csv(path: str | Path, columns: Sequence[str] | None = None) -> list[dict[str, str]]:
    with Path(path).open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if columns is not None and tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"Unexpected schema in {path}: {reader.fieldnames}")
        return list(reader)


def read_json(path: str | Path) -> object:
    with Path(path).open(encoding="utf-8") as handle:
        return json.load(handle)


def cohort_semantic_records(
    rows: Sequence[Mapping[str, object]],
) -> list[dict[str, object]]:
    return [
        {
            "protein_id": str(row["protein_id"]),
            "source": str(row["source"]),
            "manifest_version": str(row["manifest_version"]),
            "manifest_eligibility_status": str(row["manifest_eligibility_status"]),
            "expected_residue_count": int(row["expected_residue_count"]),
            "cohort_status": str(row["cohort_status"]),
        }
        for row in rows
    ]


@dataclass
class SampleBundle:
    protein_ids: tuple[str, ...]
    offsets: np.ndarray
    sampled_ranks: np.ndarray
    tri_i: np.ndarray
    tri_j: np.ndarray
    tri_k: np.ndarray
    cohort_rows: list[dict[str, str]]
    sample_rows: list[dict[str, str]]
    provenance: dict[str, object]

    def triples_for(self, protein_index: int) -> np.ndarray:
        start = int(self.offsets[protein_index])
        stop = int(self.offsets[protein_index + 1])
        return np.column_stack(
            (self.tri_i[start:stop], self.tri_j[start:stop], self.tri_k[start:stop])
        )

    def ranks_for(self, protein_index: int) -> np.ndarray:
        start = int(self.offsets[protein_index])
        stop = int(self.offsets[protein_index + 1])
        return self.sampled_ranks[start:stop]


def load_sample_bundle(
    root: str | Path, *, expected_protein_count: int | None = 154
) -> SampleBundle:
    destination = Path(root)
    metadata = destination / "metadata"
    cohort_path = metadata / "cameo_cohort.csv"
    manifest_path = metadata / "sample_manifest.csv"
    sample_path = metadata / "triangle_samples.npz"
    provenance_path = metadata / "sample_provenance.json"
    for path in (cohort_path, manifest_path, sample_path, provenance_path):
        if not path.is_file():
            raise FileNotFoundError(f"Missing frozen sampling artifact: {path}")

    raw_provenance = read_json(provenance_path)
    if not isinstance(raw_provenance, dict):
        raise ValueError("sample_provenance.json must contain an object")
    provenance = dict(raw_provenance)
    expected_hashes = {
        "cohort_csv_sha256": sha256_file(cohort_path),
        "sample_manifest_sha256": sha256_file(manifest_path),
        "triangle_samples_npz_sha256": sha256_file(sample_path),
    }
    for field, observed in expected_hashes.items():
        if provenance.get(field) != observed:
            raise ValueError(f"Sampling artifact hash mismatch for {field}")
    if provenance.get("configuration_digest") != CONFIGURATION_DIGEST:
        raise ValueError("Sampling configuration digest mismatch")

    cohort_rows = read_csv(cohort_path, COHORT_COLUMNS)
    sample_rows = read_csv(manifest_path, SAMPLE_MANIFEST_COLUMNS)
    if expected_protein_count is not None and len(sample_rows) != expected_protein_count:
        raise ValueError(
            f"Expected {expected_protein_count} sampled proteins; observed {len(sample_rows)}"
        )
    if len(cohort_rows) != len(sample_rows):
        raise ValueError("Cohort and sample manifests have different row counts")
    if semantic_digest(cohort_semantic_records(cohort_rows)) != provenance.get(
        "cohort_digest"
    ):
        raise ValueError("Cohort semantic digest mismatch")

    with np.load(sample_path, allow_pickle=False) as archive:
        expected_arrays = {
            "protein_ids",
            "offsets",
            "sampled_ranks",
            "tri_i",
            "tri_j",
            "tri_k",
        }
        if set(archive.files) != expected_arrays:
            raise ValueError(f"Unexpected triangle sample arrays: {archive.files}")
        protein_ids = tuple(str(value) for value in archive["protein_ids"])
        offsets = archive["offsets"].astype(np.int64, copy=True)
        sampled_ranks = archive["sampled_ranks"].astype(np.int64, copy=True)
        tri_i = archive["tri_i"].astype(np.int32, copy=True)
        tri_j = archive["tri_j"].astype(np.int32, copy=True)
        tri_k = archive["tri_k"].astype(np.int32, copy=True)

    manifest_ids = tuple(row["protein_id"] for row in sample_rows)
    cohort_ids = tuple(row["protein_id"] for row in cohort_rows)
    if protein_ids != manifest_ids or protein_ids != cohort_ids:
        raise ValueError("Protein ordering differs across sampling artifacts")
    if offsets.shape != (len(protein_ids) + 1,) or offsets[0] != 0:
        raise ValueError("Invalid sample offsets")
    if np.any(offsets[1:] < offsets[:-1]):
        raise ValueError("Sample offsets are not monotonic")
    total = int(offsets[-1])
    if not all(len(array) == total for array in (sampled_ranks, tri_i, tri_j, tri_k)):
        raise ValueError("Concatenated sample array lengths disagree with offsets")

    semantic_rows: list[dict[str, object]] = []
    for protein_index, row in enumerate(sample_rows):
        n_residues = int(row["n_residues"])
        population = math.comb(n_residues, 3) if n_residues >= 3 else 0
        expected_sample_count = min(SAMPLE_CAP, population)
        cohort_row = cohort_rows[protein_index]
        if int(cohort_row["expected_residue_count"]) != n_residues:
            raise ValueError(f"Cohort/sample residue mismatch for {row['protein_id']}")
        if row["sample_version"] != SAMPLE_VERSION:
            raise ValueError(f"Sample version mismatch for {row['protein_id']}")
        if int(row["n_population_triangles"]) != population:
            raise ValueError(f"Triangle population mismatch for {row['protein_id']}")
        if int(row["n_sampled_triangles"]) != expected_sample_count:
            raise ValueError(f"Triangle sample count mismatch for {row['protein_id']}")
        expected_status = "complete" if expected_sample_count else "insufficient_residues"
        if row["status"] != expected_status:
            raise ValueError(f"Sample support status mismatch for {row['protein_id']}")
        start, stop = int(offsets[protein_index]), int(offsets[protein_index + 1])
        if int(row["sample_offset_start"]) != start or int(row["sample_offset_stop"]) != stop:
            raise ValueError(f"Offset mismatch for {row['protein_id']}")
        triples = np.column_stack(
            (tri_i[start:stop], tri_j[start:stop], tri_k[start:stop])
        )
        ranks = sampled_ranks[start:stop]
        validate_samples(
            triples, ranks, n_residues, expected_count=int(row["n_sampled_triangles"])
        )
        observed_sample_hash = sample_sha256(row["protein_id"], n_residues, ranks, triples)
        if observed_sample_hash != row["sample_sha256"]:
            raise ValueError(f"Sample digest mismatch for {row['protein_id']}")
        semantic_rows.append(
            {
                "protein_id": row["protein_id"],
                "n_residues": n_residues,
                "n_population_triangles": int(row["n_population_triangles"]),
                "n_sampled_triangles": int(row["n_sampled_triangles"]),
                "sample_sha256": observed_sample_hash,
            }
        )
    if semantic_digest(semantic_rows) != provenance.get("sample_semantic_digest"):
        raise ValueError("Complete-sample semantic digest mismatch")
    if int(provenance.get("protein_count", -1)) != len(protein_ids):
        raise ValueError("Sampling provenance protein count mismatch")
    if int(provenance.get("total_sampled_triangles", -1)) != total:
        raise ValueError("Sampling provenance triangle count mismatch")

    return SampleBundle(
        protein_ids=protein_ids,
        offsets=offsets,
        sampled_ranks=sampled_ranks,
        tri_i=tri_i,
        tri_j=tri_j,
        tri_k=tri_k,
        cohort_rows=cohort_rows,
        sample_rows=sample_rows,
        provenance=provenance,
    )


def _largest_index_with_comb_at_most(
    target: int, choose: int, lower: int, upper: int
) -> int:
    """Return largest x in [lower, upper] with C(x, choose) <= target."""
    lo, hi = lower, upper
    while lo < hi:
        midpoint = (lo + hi + 1) // 2
        if math.comb(midpoint, choose) <= target:
            lo = midpoint
        else:
            hi = midpoint - 1
    return lo


def unrank_triple(rank: int, n_residues: int) -> tuple[int, int, int]:
    """Unrank a colex combinadic rank as a canonical triple i > j > k."""
    population = math.comb(n_residues, 3) if n_residues >= 3 else 0
    if rank < 0 or rank >= population:
        raise ValueError(f"Rank {rank} outside [0, {population})")
    i = _largest_index_with_comb_at_most(rank, 3, 2, n_residues - 1)
    remainder = rank - math.comb(i, 3)
    j = _largest_index_with_comb_at_most(remainder, 2, 1, i - 1)
    k = remainder - math.comb(j, 2)
    if not i > j > k >= 0:
        raise AssertionError(f"Unranking produced invalid triple {(i, j, k)}")
    return i, j, k


def derived_seed(protein_id: str, base_seed: int = BASE_SEED) -> int:
    payload = f"{SAMPLE_VERSION}|{base_seed}|{protein_id}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")


def sample_triples(
    protein_id: str,
    n_residues: int,
    *,
    base_seed: int = BASE_SEED,
    sample_cap: int = SAMPLE_CAP,
) -> tuple[np.ndarray, np.ndarray, int]:
    population = math.comb(n_residues, 3) if n_residues >= 3 else 0
    count = min(sample_cap, population)
    seed = derived_seed(protein_id, base_seed)
    if count == 0:
        return np.empty((0, 3), dtype=np.int32), np.empty(0, dtype=np.int64), seed
    generator = np.random.Generator(np.random.PCG64(seed))
    ranks = np.sort(generator.choice(population, size=count, replace=False)).astype(
        np.int64, copy=False
    )
    triples = np.asarray(
        [unrank_triple(int(rank), n_residues) for rank in ranks], dtype=np.int32
    )
    return triples, ranks, seed


def sample_sha256(
    protein_id: str, n_residues: int, ranks: np.ndarray, triples: np.ndarray
) -> str:
    digest = hashlib.sha256()
    digest.update(f"{SAMPLE_VERSION}|{protein_id}|{n_residues}|".encode("utf-8"))
    digest.update(np.asarray(ranks, dtype="<i8").tobytes(order="C"))
    digest.update(np.asarray(triples, dtype="<i4").tobytes(order="C"))
    return digest.hexdigest()


def validate_samples(
    triples: np.ndarray,
    ranks: np.ndarray,
    n_residues: int,
    *,
    expected_count: int | None = None,
) -> None:
    tri = np.asarray(triples)
    rank_array = np.asarray(ranks)
    population = math.comb(n_residues, 3) if n_residues >= 3 else 0
    count = min(SAMPLE_CAP, population) if expected_count is None else expected_count
    if tri.shape != (count, 3) or rank_array.shape != (count,):
        raise ValueError(
            f"Sample shape mismatch: triples={tri.shape}, ranks={rank_array.shape}, expected={count}"
        )
    if tri.dtype.kind not in "iu" or rank_array.dtype.kind not in "iu":
        raise ValueError("Sample arrays must use integer dtypes")
    if count == 0:
        return
    if not np.all((tri[:, 0] > tri[:, 1]) & (tri[:, 1] > tri[:, 2])):
        raise ValueError("Sample triples are not canonical i > j > k")
    if not np.all((tri >= 0) & (tri < n_residues)):
        raise ValueError("Sample triples contain out-of-range residue indices")
    if not np.all(rank_array[1:] > rank_array[:-1]):
        raise ValueError("Sample ranks must be unique and strictly increasing")
    if int(rank_array[0]) < 0 or int(rank_array[-1]) >= population:
        raise ValueError("Sample ranks lie outside the triangle population")
    reconstructed = np.asarray(
        [unrank_triple(int(rank), n_residues) for rank in rank_array], dtype=np.int32
    )
    if not np.array_equal(tri.astype(np.int32, copy=False), reconstructed):
        raise ValueError("Stored triples do not match their authoritative ranks")


def gather_directional_sides(
    tensor: np.ndarray, triples: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    tri = np.asarray(triples)
    if tri.ndim != 2 or tri.shape[1] != 3:
        raise ValueError("triples must have shape (n, 3)")
    i, j, k = tri.T
    lower = np.stack((tensor[i, j], tensor[i, k], tensor[j, k]), axis=1).astype(
        np.float64, copy=False
    )
    reverse = np.stack((tensor[j, i], tensor[k, i], tensor[k, j]), axis=1).astype(
        np.float64, copy=False
    )
    if not np.all(np.isfinite(lower)) or not np.all(np.isfinite(reverse)):
        raise ValueError("Gathered pair sides contain nonfinite values")
    return lower, reverse


def transform_variant(
    lower: np.ndarray, reverse: np.ndarray, variant: str
) -> np.ndarray:
    if variant == "directional_raw":
        return lower
    if variant == "directional_abs":
        return np.abs(lower)
    if variant == "abs_then_sym":
        return (np.abs(lower) + np.abs(reverse)) / 2.0
    if variant == "sym_then_abs":
        return np.abs((lower + reverse) / 2.0)
    raise ValueError(f"Unknown representation variant: {variant}")


def channel_measurements(sides: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(sides, dtype=np.float64)
    if values.ndim != 3 or values.shape[1] != 3:
        raise ValueError("sides must have shape (triangles, 3, channels)")
    if not np.all(np.isfinite(values)):
        raise ValueError("sides must be finite")
    maximum = np.max(values, axis=1)
    margin = 2.0 * maximum - np.sum(values, axis=1, dtype=np.float64)
    scale = np.sum(np.abs(values), axis=1, dtype=np.float64)
    normalized = np.divide(
        margin, scale, out=np.zeros_like(margin), where=scale > 0.0
    )
    metrics = np.stack(
        (
            margin,
            np.maximum(margin, 0.0),
            normalized,
            np.maximum(normalized, 0.0),
        ),
        axis=2,
    )
    if not np.all(np.isfinite(metrics)):
        raise ValueError("Calculated channel measurements are nonfinite")
    tolerance = 32 * np.finfo(np.float64).eps
    if np.any(np.abs(normalized) > 1.0 + tolerance):
        raise ValueError("Normalized margins exceed their L1 bound")
    return metrics, margin > 0.0, scale == 0.0


@dataclass
class ProteinReduction:
    triangle_metrics: np.ndarray
    triangle_positive_fraction: np.ndarray
    channel_metric_mean: np.ndarray
    channel_positive_fraction: np.ndarray
    channel_zero_scale_fraction: np.ndarray


def reduce_protein_tensor(
    tensor: np.ndarray, triples: np.ndarray, *, chunk_size: int = CHUNK_SIZE
) -> ProteinReduction:
    array = np.asarray(tensor)
    if array.ndim != 3 or array.shape[0] != array.shape[1] or array.shape[2] != N_CHANNELS:
        raise ValueError(f"Expected pair tensor shape (R, R, {N_CHANNELS}); observed {array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError("Pair tensor contains nonfinite values")
    tri = np.asarray(triples, dtype=np.int64)
    n_triangles = len(tri)
    if n_triangles == 0:
        raise ValueError("Cannot reduce an empty triangle sample")
    if chunk_size <= 0:
        raise ValueError("chunk_size must be positive")

    triangle_metrics = np.empty(
        (n_triangles, len(VARIANTS), len(PRIMARY_METRICS)), dtype=np.float64
    )
    triangle_positive = np.empty((n_triangles, len(VARIANTS)), dtype=np.float64)
    metric_sums = np.zeros(
        (len(VARIANTS), N_CHANNELS, len(PRIMARY_METRICS)), dtype=np.float64
    )
    positive_sums = np.zeros((len(VARIANTS), N_CHANNELS), dtype=np.float64)
    zero_sums = np.zeros((len(VARIANTS), N_CHANNELS), dtype=np.float64)

    for start in range(0, n_triangles, chunk_size):
        stop = min(start + chunk_size, n_triangles)
        lower, reverse = gather_directional_sides(array, tri[start:stop])
        for variant_index, variant in enumerate(VARIANTS):
            sides = transform_variant(lower, reverse, variant)
            metrics, positive, zero_scale = channel_measurements(sides)
            triangle_metrics[start:stop, variant_index] = np.mean(metrics, axis=1)
            triangle_positive[start:stop, variant_index] = np.mean(positive, axis=1)
            metric_sums[variant_index] += np.sum(metrics, axis=0, dtype=np.float64)
            positive_sums[variant_index] += np.sum(positive, axis=0, dtype=np.float64)
            zero_sums[variant_index] += np.sum(zero_scale, axis=0, dtype=np.float64)

    denominator = float(n_triangles)
    return ProteinReduction(
        triangle_metrics=triangle_metrics,
        triangle_positive_fraction=triangle_positive,
        channel_metric_mean=metric_sums / denominator,
        channel_positive_fraction=positive_sums / denominator,
        channel_zero_scale_fraction=zero_sums / denominator,
    )


def distribution_summary(values: np.ndarray) -> dict[str, float | int]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "mean": float("nan"),
            "std": float("nan"),
            "median": float("nan"),
            "p90": float("nan"),
            "p99": float("nan"),
            "n_finite_triangles": 0,
        }
    return {
        "mean": float(np.mean(finite)),
        "std": float(np.std(finite, ddof=0)),
        "median": float(np.median(finite)),
        "p90": float(np.quantile(finite, 0.90)),
        "p99": float(np.quantile(finite, 0.99)),
        "n_finite_triangles": int(finite.size),
    }


def protein_equal_summary(values: Sequence[float], attempted: int) -> dict[str, object]:
    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {
            "protein_equal_mean": float("nan"),
            "protein_std": float("nan"),
            "protein_median": float("nan"),
            "protein_q25": float("nan"),
            "protein_q75": float("nan"),
            "n_proteins_attempted": attempted,
            "n_proteins_finite": 0,
            "support_status": "no_finite_proteins",
        }
    return {
        "protein_equal_mean": float(np.mean(finite)),
        "protein_std": float(np.std(finite, ddof=0)),
        "protein_median": float(np.median(finite)),
        "protein_q25": float(np.quantile(finite, 0.25)),
        "protein_q75": float(np.quantile(finite, 0.75)),
        "n_proteins_attempted": attempted,
        "n_proteins_finite": int(finite.size),
        "support_status": "complete" if finite.size == attempted else "incomplete",
    }


def default_pair_path(protein_id: str, layer: int) -> Path:
    override = os.environ.get("PAIR_BLOCK_ROOT")
    if override:
        protein_dir = Path(override).expanduser() / protein_id
        candidates = (
            protein_dir / f"{protein_id}_pair_block_{layer}.npy",
            protein_dir / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy",
        )
        return next((path for path in candidates if path.is_file()), candidates[0])

    module_dir = Path(__file__).resolve().parent
    cluster_dir = module_dir.parent if module_dir.name == "pair_triangle_dominance" else module_dir
    if str(cluster_dir) not in sys.path:
        sys.path.insert(0, str(cluster_dir))
    from cluster_regression_helpers import resolve_combined_pair_block_path

    return resolve_combined_pair_block_path(protein_id, "cameo", layer)


def requested_resource(name: str, fallback: str) -> str:
    return os.environ.get(name, fallback)


def runtime_row(
    *,
    stage: str,
    layer: int | str,
    started_at: str,
    finished_at: str,
    elapsed_seconds: float,
    processed_proteins: int,
    processed_sampled_triangles: int,
    status: str,
) -> dict[str, object]:
    return {
        "analysis_name": ANALYSIS_NAME,
        "stage": stage,
        "layer": layer,
        "started_at": started_at,
        "finished_at": finished_at,
        "elapsed_seconds": elapsed_seconds,
        "requested_cpus": requested_resource("SLURM_CPUS_PER_TASK", "1"),
        "requested_memory": requested_resource("SLURM_MEM_PER_NODE", "unknown"),
        "requested_walltime": requested_resource("SLURM_TIMELIMIT", "unknown"),
        "processed_proteins": processed_proteins,
        "processed_sampled_triangles": processed_sampled_triangles,
        "peak_rss_mb": peak_rss_mb(),
        "status": status,
    }


__all__ = [name for name in globals() if not name.startswith("_")]

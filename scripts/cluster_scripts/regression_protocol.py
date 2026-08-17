#!/usr/bin/env python3
"""Canonical scientific protocol utilities for distance-regression probes.

This module is intentionally independent of any particular regression runner.
It fixes cohort, split, pair, feature, binning, weighting, and scaling behavior
that must remain identical across model families.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np

from cluster_regression_helpers import STANDARD_AA_PAIR_LABELS


PROTOCOL_VERSION = "1.0"
MASTER_MANIFEST_VERSION = "regression_master_split_v1"
N_CHANNELS = 128
FEATURE_DIM = 2 * N_CHANNELS
LAYERS = tuple(range(48))
N_SIZE_BINS = 5
OUTER_SPLIT_SEED = 42
DEVELOPMENT_SPLIT_SEED = 43
WITHIN_PROTEIN_SPLIT_SEED = 42
HELDOUT_PAIR_FRACTION = 0.2
FLOAT_EPS = 1e-12
FLOAT_DTYPE = np.dtype(np.float64)

MASTER_ROLES = ("fit", "validation", "test")
PROTEIN_SPLITS = ("development", "test")
SEQUENCE_BIN_LABELS = ("1", "2-4", "5-9", "10-19", "20-49", "50+")
SPATIAL_DISTANCE_BIN_LABELS = ("0-5", "5-10", "10-20", "20-40", "40+")
AMINO_ACID_PAIR_LABELS = tuple(STANDARD_AA_PAIR_LABELS)
FEATURE_NAMES = tuple(
    [f"pair_ij_ch_{index:03d}" for index in range(N_CHANNELS)]
    + [f"pair_ji_ch_{index:03d}" for index in range(N_CHANNELS)]
)

MODULE_DIR = Path(__file__).resolve().parent
MASTER_MANIFEST_PATH = (
    MODULE_DIR / "regression_protocol_data" / "regression_master_split_v1.csv"
)
MASTER_MANIFEST_COLUMNS = (
    "manifest_version",
    "protein_id",
    "source",
    "eligibility_status",
    "exclusion_reason",
    "n_residues",
    "n_pairs",
    "size_bin",
    "master_role",
    "protein_split",
    "outer_split_seed",
    "development_split_seed",
)

# Filled after the frozen manifest is generated. The loader checks this semantic
# digest rather than a byte-level file hash, so newline style is irrelevant.
MASTER_MANIFEST_DIGEST = "5561b786682d4408612c76d44c14ff6cf760291c2616159d3bea4cf13a149440"


@dataclass(frozen=True)
class ManifestEntry:
    manifest_version: str
    protein_id: str
    source: str
    eligibility_status: str
    exclusion_reason: str
    n_residues: int
    n_pairs: int
    size_bin: int | None
    master_role: str
    protein_split: str
    outer_split_seed: int
    development_split_seed: int

    @property
    def eligible(self) -> bool:
        return self.eligibility_status == "eligible"

    def semantic_record(self) -> dict[str, object]:
        return {
            column: getattr(self, column)
            for column in MASTER_MANIFEST_COLUMNS
        }


def canonical_json_bytes(value: object) -> bytes:
    """Serialize JSON deterministically and reject non-standard NaN values."""
    return json.dumps(
        value,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def semantic_digest(value: object) -> str:
    """Return a deterministic SHA-256 digest for JSON-compatible content."""
    return hashlib.sha256(canonical_json_bytes(value)).hexdigest()


def manifest_digest(entries: Sequence[ManifestEntry | Mapping[str, object]]) -> str:
    """Digest canonical manifest fields after sorting by protein identifier."""
    normalized = []
    for entry in entries:
        if isinstance(entry, ManifestEntry):
            record = entry.semantic_record()
        else:
            size_bin = entry["size_bin"]
            record = {
                "manifest_version": str(entry["manifest_version"]),
                "protein_id": str(entry["protein_id"]),
                "source": str(entry["source"]),
                "eligibility_status": str(entry["eligibility_status"]),
                "exclusion_reason": str(entry["exclusion_reason"]),
                "n_residues": int(entry["n_residues"]),
                "n_pairs": int(entry["n_pairs"]),
                "size_bin": None if size_bin in (None, "") else int(size_bin),
                "master_role": str(entry["master_role"]),
                "protein_split": str(entry["protein_split"]),
                "outer_split_seed": int(entry["outer_split_seed"]),
                "development_split_seed": int(entry["development_split_seed"]),
            }
        normalized.append(record)
    normalized.sort(key=lambda row: str(row["protein_id"]))
    return semantic_digest(normalized)


def _parse_manifest_row(row: Mapping[str, str]) -> ManifestEntry:
    missing = set(MASTER_MANIFEST_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Manifest row is missing columns: {sorted(missing)}")
    size_bin_text = row["size_bin"].strip()
    return ManifestEntry(
        manifest_version=row["manifest_version"].strip(),
        protein_id=row["protein_id"].strip(),
        source=row["source"].strip(),
        eligibility_status=row["eligibility_status"].strip(),
        exclusion_reason=row["exclusion_reason"].strip(),
        n_residues=int(row["n_residues"]),
        n_pairs=int(row["n_pairs"]),
        size_bin=int(size_bin_text) if size_bin_text else None,
        master_role=row["master_role"].strip(),
        protein_split=row["protein_split"].strip(),
        outer_split_seed=int(row["outer_split_seed"]),
        development_split_seed=int(row["development_split_seed"]),
    )


def validate_master_manifest(entries: Sequence[ManifestEntry]) -> None:
    """Enforce the frozen cohort and split invariants."""
    if len(entries) != 554:
        raise ValueError(f"Expected 554 manifest proteins, observed {len(entries)}")
    protein_ids = [entry.protein_id for entry in entries]
    if len(protein_ids) != len(set(protein_ids)):
        raise ValueError("Master manifest contains duplicate protein IDs")

    source_counts = {source: sum(e.source == source for e in entries) for source in ("cameo", "ecod")}
    if source_counts != {"cameo": 154, "ecod": 400}:
        raise ValueError(f"Unexpected manifest source counts: {source_counts}")

    eligible = [entry for entry in entries if entry.eligible]
    excluded = [entry for entry in entries if not entry.eligible]
    if len(eligible) != 552 or len(excluded) != 2:
        raise ValueError(
            f"Expected 552 eligible and 2 excluded proteins; observed {len(eligible)} and {len(excluded)}"
        )

    expected_exclusions = {"000196893", "001681437"}
    if {entry.protein_id for entry in excluded} != expected_exclusions:
        raise ValueError("Frozen technical exclusions do not match the canonical IDs")

    role_counts = {role: sum(e.master_role == role for e in eligible) for role in MASTER_ROLES}
    if role_counts != {"fit": 330, "validation": 111, "test": 111}:
        raise ValueError(f"Unexpected master-role counts: {role_counts}")

    for entry in entries:
        if entry.manifest_version != MASTER_MANIFEST_VERSION:
            raise ValueError(f"Unexpected manifest version for {entry.protein_id}")
        if entry.source not in {"cameo", "ecod"}:
            raise ValueError(f"Unexpected source for {entry.protein_id}: {entry.source}")
        if entry.n_residues < 2:
            raise ValueError(f"Invalid residue count for {entry.protein_id}")
        if entry.n_pairs != entry.n_residues * (entry.n_residues - 1) // 2:
            raise ValueError(f"Pair count does not match residue count for {entry.protein_id}")
        if entry.outer_split_seed != OUTER_SPLIT_SEED:
            raise ValueError(f"Unexpected outer split seed for {entry.protein_id}")
        if entry.development_split_seed != DEVELOPMENT_SPLIT_SEED:
            raise ValueError(f"Unexpected development split seed for {entry.protein_id}")
        if entry.eligible:
            if entry.size_bin not in range(N_SIZE_BINS):
                raise ValueError(f"Invalid size bin for {entry.protein_id}")
            if entry.master_role not in MASTER_ROLES:
                raise ValueError(f"Invalid master role for {entry.protein_id}")
            expected_split = "test" if entry.master_role == "test" else "development"
            if entry.protein_split != expected_split:
                raise ValueError(f"Inconsistent protein split for {entry.protein_id}")
            if entry.exclusion_reason:
                raise ValueError(f"Eligible protein {entry.protein_id} has an exclusion reason")
        else:
            if entry.eligibility_status != "excluded" or not entry.exclusion_reason:
                raise ValueError(f"Excluded protein {entry.protein_id} lacks a reason")
            if entry.size_bin is not None or entry.master_role != "excluded":
                raise ValueError(f"Excluded protein {entry.protein_id} has a split assignment")
            if entry.protein_split != "not_applicable":
                raise ValueError(f"Excluded protein {entry.protein_id} has a protein split")


def load_master_manifest(
    path: str | Path = MASTER_MANIFEST_PATH,
    *,
    verify_digest: bool = True,
) -> tuple[ManifestEntry, ...]:
    """Load and validate the frozen combined CAMEO+ECOD manifest."""
    manifest_path = Path(path)
    with manifest_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != MASTER_MANIFEST_COLUMNS:
            raise ValueError(
                f"Unexpected manifest columns in {manifest_path}: {reader.fieldnames}"
            )
        entries = tuple(_parse_manifest_row(row) for row in reader)
    validate_master_manifest(entries)
    observed_digest = manifest_digest(entries)
    if verify_digest and MASTER_MANIFEST_DIGEST and observed_digest != MASTER_MANIFEST_DIGEST:
        raise ValueError(
            f"Master manifest digest mismatch: {observed_digest} != {MASTER_MANIFEST_DIGEST}"
        )
    return entries


def eligible_manifest(entries: Sequence[ManifestEntry]) -> tuple[ManifestEntry, ...]:
    return tuple(entry for entry in entries if entry.eligible)


def partition_manifest_by_role(
    entries: Sequence[ManifestEntry],
) -> dict[str, tuple[ManifestEntry, ...]]:
    """Return eligible entries grouped into disjoint master roles."""
    eligible = eligible_manifest(entries)
    return {
        role: tuple(sorted((entry for entry in eligible if entry.master_role == role), key=lambda e: e.protein_id))
        for role in MASTER_ROLES
    }


def stable_seed(*parts: object) -> int:
    digest = hashlib.sha256("|".join(map(str, parts)).encode("utf-8")).digest()
    return int.from_bytes(digest[:8], "little") % (2**32)


def assign_rank_size_bins(records: Sequence[Mapping[str, object]]) -> dict[str, int]:
    """Assign deterministic equal-frequency length bins without mutating records."""
    if not records:
        raise ValueError("Cannot size-bin an empty protein collection")
    ordered = sorted(records, key=lambda row: (int(row["n_residues"]), str(row["protein_id"])))
    if len({str(row["protein_id"]) for row in ordered}) != len(ordered):
        raise ValueError("Protein records contain duplicate IDs")
    return {
        str(row["protein_id"]): min(N_SIZE_BINS - 1, rank * N_SIZE_BINS // len(ordered))
        for rank, row in enumerate(ordered)
    }


def _validation_counts_by_bin(bin_counts: Mapping[int, int], n_validation: int) -> dict[int, int]:
    total = sum(bin_counts.values())
    if total <= 0 or not 0 < n_validation < total:
        raise ValueError("Validation count must leave nonempty fit and validation sets")
    quotas = {size_bin: count * n_validation / total for size_bin, count in bin_counts.items()}
    allocated = {size_bin: int(math.floor(quota)) for size_bin, quota in quotas.items()}
    remaining = n_validation - sum(allocated.values())
    priority = sorted(bin_counts, key=lambda b: (-(quotas[b] - allocated[b]), b))
    for size_bin in priority[:remaining]:
        allocated[size_bin] += 1
    for size_bin, count in bin_counts.items():
        if allocated[size_bin] <= 0 or allocated[size_bin] >= count:
            raise ValueError(f"Size bin {size_bin} cannot support both development roles")
    return allocated


def build_master_roles(
    records: Sequence[Mapping[str, object]],
    locked_test_ids: Iterable[str],
    *,
    validation_seed: int = DEVELOPMENT_SPLIT_SEED,
) -> dict[str, str]:
    """Preserve explicit test IDs and deterministically subdivide development 75/25."""
    if not records:
        raise ValueError("Cannot split an empty protein collection")
    record_by_id = {str(row["protein_id"]): row for row in records}
    if len(record_by_id) != len(records):
        raise ValueError("Protein records contain duplicate IDs")
    test_ids = {str(protein_id) for protein_id in locked_test_ids}
    unknown = test_ids - set(record_by_id)
    if unknown:
        raise ValueError(f"Locked test IDs are absent from records: {sorted(unknown)}")
    if not test_ids or len(test_ids) == len(records):
        raise ValueError("Locked test membership must leave both development and test proteins")

    computed_bins = assign_rank_size_bins(records)
    size_bins = {}
    for protein_id, record in record_by_id.items():
        provided = record.get("size_bin")
        size_bins[protein_id] = computed_bins[protein_id] if provided in (None, "") else int(provided)
    if any(size_bin not in range(N_SIZE_BINS) for size_bin in size_bins.values()):
        raise ValueError("Protein records contain invalid size bins")

    development_ids = sorted(set(record_by_id) - test_ids)
    bin_members = {
        size_bin: [protein_id for protein_id in development_ids if size_bins[protein_id] == size_bin]
        for size_bin in range(N_SIZE_BINS)
    }
    if any(len(members) < 2 for members in bin_members.values()):
        raise ValueError("Every size bin must contain at least two development proteins")
    n_validation = int(math.ceil(0.25 * len(development_ids)))
    validation_counts = _validation_counts_by_bin(
        {size_bin: len(members) for size_bin, members in bin_members.items()},
        n_validation,
    )

    validation_ids: set[str] = set()
    for size_bin, members in bin_members.items():
        ordered = np.asarray(sorted(members), dtype=object)
        rng = np.random.default_rng(stable_seed(validation_seed, "development", size_bin))
        selected = rng.permutation(ordered.size)[: validation_counts[size_bin]]
        validation_ids.update(str(ordered[int(index)]) for index in selected)

    return {
        protein_id: (
            "test" if protein_id in test_ids
            else "validation" if protein_id in validation_ids
            else "fit"
        )
        for protein_id in sorted(record_by_id)
    }


def within_protein_split_seed(protein_id: str, seed: int = WITHIN_PROTEIN_SPLIT_SEED) -> int:
    digest = hashlib.blake2b(str(protein_id).encode("utf-8"), digest_size=8).digest()
    protein_seed = int.from_bytes(digest, byteorder="little", signed=False)
    return (protein_seed + int(seed)) % (2**63 - 1)


def within_protein_pair_split(
    n_pairs: int,
    protein_id: str,
    *,
    seed: int = WITHIN_PROTEIN_SPLIT_SEED,
) -> tuple[np.ndarray, np.ndarray]:
    """Return the canonical deterministic 80/20 training/held-out pair split."""
    n_pairs = int(n_pairs)
    if n_pairs < 2:
        raise ValueError(f"Protein {protein_id} has too few pairs to split: {n_pairs}")
    n_heldout = min(max(int(math.ceil(HELDOUT_PAIR_FRACTION * n_pairs)), 1), n_pairs - 1)
    permutation = np.random.default_rng(within_protein_split_seed(protein_id, seed)).permutation(n_pairs)
    heldout = np.sort(permutation[:n_heldout].astype(np.int64, copy=False))
    train = np.sort(permutation[n_heldout:].astype(np.int64, copy=False))
    return train, heldout


def within_protein_pair_split_digest(
    n_pairs: int,
    protein_id: str,
    *,
    seed: int = WITHIN_PROTEIN_SPLIT_SEED,
) -> str:
    train, heldout = within_protein_pair_split(n_pairs, protein_id, seed=seed)
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({"protein_id": str(protein_id), "n_pairs": int(n_pairs), "seed": int(seed)}))
    digest.update(train.astype("<i8", copy=False).tobytes())
    digest.update(heldout.astype("<i8", copy=False).tobytes())
    return digest.hexdigest()


def strict_lower_triangle_indices(n_residues: int) -> tuple[np.ndarray, np.ndarray]:
    n_residues = int(n_residues)
    if n_residues < 2:
        raise ValueError("At least two residues are required")
    pair_i, pair_j = np.tril_indices(n_residues, k=-1)
    return pair_i.astype(np.int64, copy=False), pair_j.astype(np.int64, copy=False)


def extract_directional_base256(
    pair_block: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
) -> np.ndarray:
    """Materialize canonical [Z_ij | Z_ji] features directly in float64."""
    pair_block = np.asarray(pair_block)
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)
    if pair_block.ndim != 3 or pair_block.shape[0] != pair_block.shape[1]:
        raise ValueError(f"Pair block must have shape (R, R, C); got {pair_block.shape}")
    if pair_block.shape[2] != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} channels; got {pair_block.shape[2]}")
    if pair_i.shape != pair_j.shape or pair_i.ndim != 1:
        raise ValueError("Pair index arrays must be one-dimensional with identical shape")
    if pair_i.size and (
        np.min(pair_i) < 0 or np.min(pair_j) < 0
        or np.max(pair_i) >= pair_block.shape[0] or np.max(pair_j) >= pair_block.shape[0]
    ):
        raise ValueError("Pair indices are out of bounds")
    features = np.empty((pair_i.size, FEATURE_DIM), dtype=np.float64)
    features[:, :N_CHANNELS] = pair_block[pair_i, pair_j, :]
    features[:, N_CHANNELS:] = pair_block[pair_j, pair_i, :]
    if not np.all(np.isfinite(features)):
        raise ValueError("Extracted base256 features contain nonfinite values")
    return features


def sequence_separation_bin_codes(
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    *,
    chain_ids: Sequence[object] | None = None,
) -> np.ndarray:
    """Map intrachain pair separations to the six canonical bins; undefined is -1."""
    pair_i = np.asarray(pair_i, dtype=np.int64)
    pair_j = np.asarray(pair_j, dtype=np.int64)
    if pair_i.shape != pair_j.shape:
        raise ValueError("Pair index arrays must have identical shape")
    separation = np.abs(pair_i - pair_j)
    if np.any(separation < 1):
        raise ValueError("Sequence separation is undefined for self-pairs")
    codes = np.select(
        [separation == 1, separation <= 4, separation <= 9, separation <= 19, separation <= 49],
        [0, 1, 2, 3, 4],
        default=5,
    ).astype(np.int16)
    if chain_ids is not None:
        chains = np.asarray(chain_ids)
        if chains.ndim != 1 or pair_i.size and max(int(np.max(pair_i)), int(np.max(pair_j))) >= chains.size:
            raise ValueError("Chain identifiers do not cover all pair indices")
        codes[chains[pair_i] != chains[pair_j]] = -1
    return codes


def spatial_distance_bin_codes(distances: np.ndarray) -> np.ndarray:
    """Map finite, nonnegative distances to the five canonical half-open bins."""
    distances = np.asarray(distances, dtype=np.float64)
    if not np.all(np.isfinite(distances)) or np.any(distances < 0):
        raise ValueError("Spatial distances must be finite and nonnegative")
    return np.select(
        [distances < 5.0, distances < 10.0, distances < 20.0, distances < 40.0],
        [0, 1, 2, 3],
        default=4,
    ).astype(np.int16)


def protein_equal_weights(protein_ids: Sequence[object]) -> np.ndarray:
    """Give every protein equal total row weight (one total unit per protein)."""
    ids = np.asarray(protein_ids)
    if ids.ndim != 1 or ids.size == 0:
        raise ValueError("Protein IDs must be a nonempty one-dimensional sequence")
    weights = np.empty(ids.size, dtype=np.float64)
    unique, counts = np.unique(ids.astype(str), return_counts=True)
    string_ids = ids.astype(str)
    for protein_id, count in zip(unique, counts):
        weights[string_ids == protein_id] = 1.0 / float(count)
    return weights


def protein_equal_mean(target_blocks: Iterable[np.ndarray]) -> float:
    """Return the arithmetic mean of per-protein target means."""
    means = []
    for block in target_blocks:
        values = np.asarray(block, dtype=np.float64)
        if values.ndim != 1 or values.size == 0 or not np.all(np.isfinite(values)):
            raise ValueError("Each target block must be a nonempty finite vector")
        means.append(float(np.mean(values, dtype=np.float64)))
    if not means:
        raise ValueError("At least one target block is required")
    return float(np.mean(np.asarray(means, dtype=np.float64), dtype=np.float64))


@dataclass
class ProteinEqualFeatureMoments:
    """Mergeable protein-equal float64 feature-moment accumulator."""

    n_proteins: int = 0
    feature_dim: int | None = None
    sum_protein_means: np.ndarray | None = None
    sum_protein_second_moments: np.ndarray | None = None

    def update_protein(self, features: np.ndarray) -> None:
        values = np.asarray(features, dtype=np.float64)
        if values.ndim != 2 or values.shape[0] == 0:
            raise ValueError("Feature block must be a nonempty two-dimensional array")
        if not np.all(np.isfinite(values)):
            raise ValueError("Feature block contains nonfinite values")
        if self.feature_dim is None:
            self.feature_dim = int(values.shape[1])
            self.sum_protein_means = np.zeros(self.feature_dim, dtype=np.float64)
            self.sum_protein_second_moments = np.zeros(self.feature_dim, dtype=np.float64)
        if values.shape[1] != self.feature_dim:
            raise ValueError(f"Feature width {values.shape[1]} != {self.feature_dim}")
        self.sum_protein_means += np.mean(values, axis=0, dtype=np.float64)
        self.sum_protein_second_moments += np.mean(values * values, axis=0, dtype=np.float64)
        self.n_proteins += 1

    def merge(self, other: "ProteinEqualFeatureMoments") -> "ProteinEqualFeatureMoments":
        if other.n_proteins == 0:
            return self
        if self.n_proteins == 0:
            self.n_proteins = other.n_proteins
            self.feature_dim = other.feature_dim
            self.sum_protein_means = other.sum_protein_means.copy()
            self.sum_protein_second_moments = other.sum_protein_second_moments.copy()
            return self
        if self.feature_dim != other.feature_dim:
            raise ValueError("Cannot merge feature moments with different widths")
        self.n_proteins += other.n_proteins
        self.sum_protein_means += other.sum_protein_means
        self.sum_protein_second_moments += other.sum_protein_second_moments
        return self

    def finalize(self, epsilon: float = FLOAT_EPS) -> tuple[np.ndarray, np.ndarray]:
        if self.n_proteins == 0 or self.sum_protein_means is None or self.sum_protein_second_moments is None:
            raise ValueError("Cannot finalize empty feature moments")
        mean = self.sum_protein_means / self.n_proteins
        variance = self.sum_protein_second_moments / self.n_proteins - mean * mean
        scale = np.sqrt(np.maximum(variance, 0.0))
        scale[scale < epsilon] = 1.0
        return mean, scale


__all__ = [
    "AMINO_ACID_PAIR_LABELS",
    "DEVELOPMENT_SPLIT_SEED",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FLOAT_DTYPE",
    "FLOAT_EPS",
    "LAYERS",
    "MASTER_MANIFEST_COLUMNS",
    "MASTER_MANIFEST_DIGEST",
    "MASTER_MANIFEST_PATH",
    "MASTER_MANIFEST_VERSION",
    "MASTER_ROLES",
    "ManifestEntry",
    "N_CHANNELS",
    "N_SIZE_BINS",
    "OUTER_SPLIT_SEED",
    "PROTOCOL_VERSION",
    "PROTEIN_SPLITS",
    "ProteinEqualFeatureMoments",
    "SEQUENCE_BIN_LABELS",
    "SPATIAL_DISTANCE_BIN_LABELS",
    "WITHIN_PROTEIN_SPLIT_SEED",
    "assign_rank_size_bins",
    "build_master_roles",
    "canonical_json_bytes",
    "eligible_manifest",
    "extract_directional_base256",
    "load_master_manifest",
    "manifest_digest",
    "partition_manifest_by_role",
    "protein_equal_mean",
    "protein_equal_weights",
    "semantic_digest",
    "sequence_separation_bin_codes",
    "spatial_distance_bin_codes",
    "stable_seed",
    "strict_lower_triangle_indices",
    "validate_master_manifest",
    "within_protein_pair_split",
    "within_protein_pair_split_digest",
    "within_protein_split_seed",
]

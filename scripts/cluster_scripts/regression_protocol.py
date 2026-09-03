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


HISTORICAL_PROTOCOL_VERSION = "1.0"
PROTOCOL_VERSION = "2.0"
MASTER_MANIFEST_VERSION = "regression_master_split_v1"
OUTER_FOLD_MANIFEST_VERSION = "regression_outer_folds_v1"
PAIR_FOLD_MANIFEST_VERSION = "per_protein_pair_folds_v1"
N_CHANNELS = 128
FEATURE_DIM = 2 * N_CHANNELS
LAYERS = tuple(range(48))
N_SIZE_BINS = 5
OUTER_SPLIT_SEED = 42
DEVELOPMENT_SPLIT_SEED = 43
WITHIN_PROTEIN_SPLIT_SEED = 42
HELDOUT_PAIR_FRACTION = 0.2
N_OUTER_FOLDS = 5
OUTER_FOLD_SEED = 44
N_PAIR_FOLDS = 5
PAIR_FOLD_BASE_SEED = 42
OUTER_FOLD_ALGORITHM = "numpy_rng44_length_bin_round_robin_v1"
PAIR_FOLD_ALGORITHM = "canonical_lower_triangle_numpy_shuffle_array_split_v1"
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
REPOSITORY_ROOT = MODULE_DIR.parents[1]
FIVEFOLD_METADATA_DIR = REPOSITORY_ROOT / "results" / "regression_5fold_v1" / "metadata"
OUTER_FOLD_MANIFEST_PATH = FIVEFOLD_METADATA_DIR / "regression_outer_folds_v1.csv"
PAIR_FOLD_MANIFEST_PATH = FIVEFOLD_METADATA_DIR / "per_protein_pair_folds_v1.csv"
FOLD_DIGESTS_PATH = FIVEFOLD_METADATA_DIR / "fold_digests.json"
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
OUTER_FOLD_MANIFEST_COLUMNS = (
    "cv_protocol_version",
    "manifest_version",
    "protein_id",
    "source",
    "n_residues",
    "n_pairs",
    "length_bin",
    "outer_fold",
    "historical_master_role",
    "historical_protein_split",
    "outer_fold_seed",
    "assignment_algorithm",
)
PAIR_FOLD_MANIFEST_COLUMNS = (
    "cv_protocol_version",
    "manifest_version",
    "protein_id",
    "source",
    "n_residues",
    "n_pairs",
    "pair_fold_algorithm",
    "pair_fold_base_seed",
    "pair_fold_seed",
    "fold_0_count",
    "fold_1_count",
    "fold_2_count",
    "fold_3_count",
    "fold_4_count",
    "assignment_digest",
)

# Filled after the frozen manifest is generated. The loader checks this semantic
# digest rather than a byte-level file hash, so newline style is irrelevant.
MASTER_MANIFEST_DIGEST = "5561b786682d4408612c76d44c14ff6cf760291c2616159d3bea4cf13a149440"
ELIGIBLE_COHORT_DIGEST = "ec65eeb90599de6d2b83c83a27df87efd420b5272fbf436ffaadfaee5025d9ea"
OUTER_FOLD_MANIFEST_DIGEST = "7d9432555ce0df9b141be8a7866061320d55f712b354cf7206ba55de6b9868cf"
PAIR_FOLD_MANIFEST_DIGEST = "ef239c80479ffd1bc4fb98a5660a73e8e714f5be0409bdf1fb8032eb342f8044"


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


@dataclass(frozen=True)
class OuterFoldEntry:
    cv_protocol_version: str
    manifest_version: str
    protein_id: str
    source: str
    n_residues: int
    n_pairs: int
    length_bin: int
    outer_fold: int
    historical_master_role: str
    historical_protein_split: str
    outer_fold_seed: int
    assignment_algorithm: str

    def semantic_record(self) -> dict[str, object]:
        return {
            column: getattr(self, column)
            for column in OUTER_FOLD_MANIFEST_COLUMNS
        }


@dataclass(frozen=True)
class PairFoldManifestEntry:
    cv_protocol_version: str
    manifest_version: str
    protein_id: str
    source: str
    n_residues: int
    n_pairs: int
    pair_fold_algorithm: str
    pair_fold_base_seed: int
    pair_fold_seed: int
    fold_0_count: int
    fold_1_count: int
    fold_2_count: int
    fold_3_count: int
    fold_4_count: int
    assignment_digest: str

    @property
    def fold_counts(self) -> tuple[int, ...]:
        return tuple(getattr(self, f"fold_{fold}_count") for fold in range(N_PAIR_FOLDS))

    def semantic_record(self) -> dict[str, object]:
        return {
            column: getattr(self, column)
            for column in PAIR_FOLD_MANIFEST_COLUMNS
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


def eligible_cohort_digest(entries: Sequence[ManifestEntry]) -> str:
    """Digest eligible cohort identity and geometry without historical split roles."""
    records = [
        {
            "protein_id": entry.protein_id,
            "source": entry.source,
            "n_residues": entry.n_residues,
            "n_pairs": entry.n_pairs,
            "length_bin": entry.size_bin,
        }
        for entry in eligible_manifest(entries)
    ]
    records.sort(key=lambda row: str(row["protein_id"]))
    return semantic_digest(records)


def outer_fold_manifest_digest(
    entries: Sequence[OuterFoldEntry | Mapping[str, object]],
) -> str:
    """Return the semantic digest of a compact outer-fold manifest."""
    normalized = []
    for entry in entries:
        if isinstance(entry, OuterFoldEntry):
            record = entry.semantic_record()
        else:
            record = {
                "cv_protocol_version": str(entry["cv_protocol_version"]),
                "manifest_version": str(entry["manifest_version"]),
                "protein_id": str(entry["protein_id"]),
                "source": str(entry["source"]),
                "n_residues": int(entry["n_residues"]),
                "n_pairs": int(entry["n_pairs"]),
                "length_bin": int(entry["length_bin"]),
                "outer_fold": int(entry["outer_fold"]),
                "historical_master_role": str(entry["historical_master_role"]),
                "historical_protein_split": str(entry["historical_protein_split"]),
                "outer_fold_seed": int(entry["outer_fold_seed"]),
                "assignment_algorithm": str(entry["assignment_algorithm"]),
            }
        normalized.append(record)
    normalized.sort(key=lambda row: str(row["protein_id"]))
    return semantic_digest(normalized)


def build_outer_fold_manifest(
    master_entries: Sequence[ManifestEntry],
    *,
    seed: int = OUTER_FOLD_SEED,
) -> tuple[OuterFoldEntry, ...]:
    """Freeze fold 0 and stratify the other proteins across folds 1 through 4."""
    eligible = sorted(eligible_manifest(master_entries), key=lambda entry: entry.protein_id)
    assignments = {
        entry.protein_id: 0
        for entry in eligible
        if entry.master_role == "test"
    }
    rng = np.random.default_rng(int(seed))
    for length_bin in range(N_SIZE_BINS):
        members = np.asarray(
            [
                entry.protein_id
                for entry in eligible
                if entry.master_role != "test" and entry.size_bin == length_bin
            ],
            dtype=object,
        )
        if members.size < N_OUTER_FOLDS - 1:
            raise ValueError(
                f"Length bin {length_bin} cannot populate four nonzero outer folds"
            )
        shuffled = members[rng.permutation(members.size)]
        for position, protein_id in enumerate(shuffled):
            assignments[str(protein_id)] = 1 + position % (N_OUTER_FOLDS - 1)

    if set(assignments) != {entry.protein_id for entry in eligible}:
        raise ValueError("Outer-fold construction did not assign every eligible protein")
    rows = tuple(
        OuterFoldEntry(
            cv_protocol_version=PROTOCOL_VERSION,
            manifest_version=OUTER_FOLD_MANIFEST_VERSION,
            protein_id=entry.protein_id,
            source=entry.source,
            n_residues=entry.n_residues,
            n_pairs=entry.n_pairs,
            length_bin=int(entry.size_bin),
            outer_fold=assignments[entry.protein_id],
            historical_master_role=entry.master_role,
            historical_protein_split=entry.protein_split,
            outer_fold_seed=int(seed),
            assignment_algorithm=OUTER_FOLD_ALGORITHM,
        )
        for entry in eligible
    )
    validate_outer_fold_manifest(rows, master_entries, verify_frozen_digest=False)
    return rows


def validate_outer_fold_manifest(
    entries: Sequence[OuterFoldEntry],
    master_entries: Sequence[ManifestEntry],
    *,
    verify_frozen_digest: bool = True,
) -> None:
    """Validate exhaustive, disjoint, stratified, deterministic outer folds."""
    eligible = sorted(eligible_manifest(master_entries), key=lambda entry: entry.protein_id)
    if len(entries) != len(eligible):
        raise ValueError(
            f"Expected {len(eligible)} outer-fold rows, observed {len(entries)}"
        )
    if len({entry.protein_id for entry in entries}) != len(entries):
        raise ValueError("Outer-fold manifest contains duplicate protein IDs")
    expected_by_id = {entry.protein_id: entry for entry in eligible}
    if {entry.protein_id for entry in entries} != set(expected_by_id):
        raise ValueError("Outer-fold manifest does not match the eligible cohort")

    for entry in entries:
        master = expected_by_id[entry.protein_id]
        if entry.cv_protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"Unexpected CV protocol version for {entry.protein_id}")
        if entry.manifest_version != OUTER_FOLD_MANIFEST_VERSION:
            raise ValueError(f"Unexpected outer-fold version for {entry.protein_id}")
        if entry.source != master.source or entry.n_residues != master.n_residues:
            raise ValueError(f"Cohort metadata mismatch for {entry.protein_id}")
        if entry.n_pairs != master.n_pairs or entry.length_bin != master.size_bin:
            raise ValueError(f"Pair or length-bin mismatch for {entry.protein_id}")
        if entry.historical_master_role != master.master_role:
            raise ValueError(f"Historical role mismatch for {entry.protein_id}")
        if entry.historical_protein_split != master.protein_split:
            raise ValueError(f"Historical split mismatch for {entry.protein_id}")
        if entry.outer_fold_seed != OUTER_FOLD_SEED:
            raise ValueError(f"Unexpected outer-fold seed for {entry.protein_id}")
        if entry.assignment_algorithm != OUTER_FOLD_ALGORITHM:
            raise ValueError(f"Unexpected outer-fold algorithm for {entry.protein_id}")
        expected_fold_range = {0} if master.master_role == "test" else set(range(1, 5))
        if entry.outer_fold not in expected_fold_range:
            raise ValueError(f"Invalid outer fold for {entry.protein_id}")

    for length_bin in range(N_SIZE_BINS):
        counts = [
            sum(
                entry.length_bin == length_bin and entry.outer_fold == fold
                for entry in entries
            )
            for fold in range(1, N_OUTER_FOLDS)
        ]
        if max(counts) - min(counts) > 1:
            raise ValueError(f"Length bin {length_bin} is not balanced: {counts}")

    if verify_frozen_digest and OUTER_FOLD_MANIFEST_DIGEST:
        observed = outer_fold_manifest_digest(entries)
        if observed != OUTER_FOLD_MANIFEST_DIGEST:
            raise ValueError(
                f"Outer-fold manifest digest mismatch: {observed} != "
                f"{OUTER_FOLD_MANIFEST_DIGEST}"
            )


def _parse_outer_fold_row(row: Mapping[str, str]) -> OuterFoldEntry:
    missing = set(OUTER_FOLD_MANIFEST_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Outer-fold row is missing columns: {sorted(missing)}")
    return OuterFoldEntry(
        cv_protocol_version=row["cv_protocol_version"].strip(),
        manifest_version=row["manifest_version"].strip(),
        protein_id=row["protein_id"].strip(),
        source=row["source"].strip(),
        n_residues=int(row["n_residues"]),
        n_pairs=int(row["n_pairs"]),
        length_bin=int(row["length_bin"]),
        outer_fold=int(row["outer_fold"]),
        historical_master_role=row["historical_master_role"].strip(),
        historical_protein_split=row["historical_protein_split"].strip(),
        outer_fold_seed=int(row["outer_fold_seed"]),
        assignment_algorithm=row["assignment_algorithm"].strip(),
    )


def load_outer_fold_manifest(
    path: str | Path = OUTER_FOLD_MANIFEST_PATH,
    *,
    master_entries: Sequence[ManifestEntry] | None = None,
    verify_digest: bool = True,
) -> tuple[OuterFoldEntry, ...]:
    """Load and validate the frozen fivefold protein assignments."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != OUTER_FOLD_MANIFEST_COLUMNS:
            raise ValueError(f"Unexpected outer-fold columns: {reader.fieldnames}")
        entries = tuple(_parse_outer_fold_row(row) for row in reader)
    validate_outer_fold_manifest(
        entries,
        master_entries or load_master_manifest(),
        verify_frozen_digest=verify_digest,
    )
    return entries


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


def pair_fold_seed(
    protein_id: str,
    base_seed: int = PAIR_FOLD_BASE_SEED,
) -> int:
    """Return the frozen protein-ID-derived seed for fivefold pair assignment."""
    return within_protein_split_seed(protein_id, seed=base_seed)


def within_protein_pair_folds(
    n_residues: int,
    protein_id: str,
    *,
    n_folds: int = N_PAIR_FOLDS,
    base_seed: int = PAIR_FOLD_BASE_SEED,
) -> np.ndarray:
    """Assign each canonical strict-lower-triangle pair to exactly one fold."""
    pair_i, _ = strict_lower_triangle_indices(n_residues)
    n_pairs = int(pair_i.size)
    if n_folds < 2 or n_pairs < n_folds:
        raise ValueError(
            f"Protein {protein_id} has {n_pairs} pairs and cannot support {n_folds} folds"
        )
    permutation = np.random.default_rng(
        pair_fold_seed(protein_id, base_seed)
    ).permutation(n_pairs)
    assignments = np.empty(n_pairs, dtype=np.int16)
    for fold, indices in enumerate(np.array_split(permutation, n_folds)):
        assignments[indices] = fold
    return assignments


def pair_fold_assignment_digest(
    n_residues: int,
    protein_id: str,
    *,
    n_folds: int = N_PAIR_FOLDS,
    base_seed: int = PAIR_FOLD_BASE_SEED,
) -> str:
    """Digest canonical pair coordinates and their deterministic fold labels."""
    pair_i, pair_j = strict_lower_triangle_indices(n_residues)
    assignments = within_protein_pair_folds(
        n_residues,
        protein_id,
        n_folds=n_folds,
        base_seed=base_seed,
    )
    digest = hashlib.sha256()
    digest.update(canonical_json_bytes({
        "algorithm": PAIR_FOLD_ALGORITHM,
        "base_seed": int(base_seed),
        "n_folds": int(n_folds),
        "n_pairs": int(pair_i.size),
        "n_residues": int(n_residues),
        "protein_id": str(protein_id),
        "seed": pair_fold_seed(protein_id, base_seed),
    }))
    digest.update(pair_i.astype("<i8", copy=False).tobytes())
    digest.update(pair_j.astype("<i8", copy=False).tobytes())
    digest.update(assignments.astype("<i2", copy=False).tobytes())
    return digest.hexdigest()


def pair_fold_manifest_digest(
    entries: Sequence[PairFoldManifestEntry | Mapping[str, object]],
) -> str:
    """Return the semantic digest of the compact per-protein pair-fold manifest."""
    normalized = []
    for entry in entries:
        if isinstance(entry, PairFoldManifestEntry):
            record = entry.semantic_record()
        else:
            record = {
                "cv_protocol_version": str(entry["cv_protocol_version"]),
                "manifest_version": str(entry["manifest_version"]),
                "protein_id": str(entry["protein_id"]),
                "source": str(entry["source"]),
                "n_residues": int(entry["n_residues"]),
                "n_pairs": int(entry["n_pairs"]),
                "pair_fold_algorithm": str(entry["pair_fold_algorithm"]),
                "pair_fold_base_seed": int(entry["pair_fold_base_seed"]),
                "pair_fold_seed": int(entry["pair_fold_seed"]),
                **{
                    f"fold_{fold}_count": int(entry[f"fold_{fold}_count"])
                    for fold in range(N_PAIR_FOLDS)
                },
                "assignment_digest": str(entry["assignment_digest"]),
            }
        normalized.append(record)
    normalized.sort(key=lambda row: str(row["protein_id"]))
    return semantic_digest(normalized)


def build_pair_fold_manifest(
    master_entries: Sequence[ManifestEntry],
) -> tuple[PairFoldManifestEntry, ...]:
    """Build one compact deterministic pair-fold row per eligible protein."""
    rows = []
    for entry in sorted(eligible_manifest(master_entries), key=lambda row: row.protein_id):
        assignments = within_protein_pair_folds(entry.n_residues, entry.protein_id)
        counts = np.bincount(assignments, minlength=N_PAIR_FOLDS)
        rows.append(PairFoldManifestEntry(
            cv_protocol_version=PROTOCOL_VERSION,
            manifest_version=PAIR_FOLD_MANIFEST_VERSION,
            protein_id=entry.protein_id,
            source=entry.source,
            n_residues=entry.n_residues,
            n_pairs=entry.n_pairs,
            pair_fold_algorithm=PAIR_FOLD_ALGORITHM,
            pair_fold_base_seed=PAIR_FOLD_BASE_SEED,
            pair_fold_seed=pair_fold_seed(entry.protein_id),
            fold_0_count=int(counts[0]),
            fold_1_count=int(counts[1]),
            fold_2_count=int(counts[2]),
            fold_3_count=int(counts[3]),
            fold_4_count=int(counts[4]),
            assignment_digest=pair_fold_assignment_digest(
                entry.n_residues, entry.protein_id
            ),
        ))
    result = tuple(rows)
    validate_pair_fold_manifest(result, master_entries, verify_frozen_digest=False)
    return result


def validate_pair_fold_manifest(
    entries: Sequence[PairFoldManifestEntry],
    master_entries: Sequence[ManifestEntry],
    *,
    verify_frozen_digest: bool = True,
) -> None:
    """Validate exact regeneration, evenness, and semantic pair-fold digests."""
    eligible = sorted(eligible_manifest(master_entries), key=lambda entry: entry.protein_id)
    if len(entries) != len(eligible):
        raise ValueError(
            f"Expected {len(eligible)} pair-fold rows, observed {len(entries)}"
        )
    if len({entry.protein_id for entry in entries}) != len(entries):
        raise ValueError("Pair-fold manifest contains duplicate protein IDs")
    expected_by_id = {entry.protein_id: entry for entry in eligible}
    if {entry.protein_id for entry in entries} != set(expected_by_id):
        raise ValueError("Pair-fold manifest does not match the eligible cohort")

    for entry in entries:
        master = expected_by_id[entry.protein_id]
        if entry.cv_protocol_version != PROTOCOL_VERSION:
            raise ValueError(f"Unexpected CV protocol version for {entry.protein_id}")
        if entry.manifest_version != PAIR_FOLD_MANIFEST_VERSION:
            raise ValueError(f"Unexpected pair-fold version for {entry.protein_id}")
        if entry.source != master.source or entry.n_residues != master.n_residues:
            raise ValueError(f"Cohort metadata mismatch for {entry.protein_id}")
        if entry.n_pairs != master.n_pairs:
            raise ValueError(f"Pair count mismatch for {entry.protein_id}")
        if entry.pair_fold_algorithm != PAIR_FOLD_ALGORITHM:
            raise ValueError(f"Unexpected pair-fold algorithm for {entry.protein_id}")
        if entry.pair_fold_base_seed != PAIR_FOLD_BASE_SEED:
            raise ValueError(f"Unexpected pair-fold base seed for {entry.protein_id}")
        if entry.pair_fold_seed != pair_fold_seed(entry.protein_id):
            raise ValueError(f"Derived pair-fold seed mismatch for {entry.protein_id}")
        if sum(entry.fold_counts) != entry.n_pairs:
            raise ValueError(f"Pair folds are not exhaustive for {entry.protein_id}")
        if max(entry.fold_counts) - min(entry.fold_counts) > 1:
            raise ValueError(f"Pair folds are uneven for {entry.protein_id}")
        expected_assignments = within_protein_pair_folds(
            entry.n_residues, entry.protein_id
        )
        expected_counts = tuple(
            int(value)
            for value in np.bincount(expected_assignments, minlength=N_PAIR_FOLDS)
        )
        if entry.fold_counts != expected_counts:
            raise ValueError(f"Pair-fold counts do not regenerate for {entry.protein_id}")
        expected_digest = pair_fold_assignment_digest(
            entry.n_residues, entry.protein_id
        )
        if entry.assignment_digest != expected_digest:
            raise ValueError(f"Pair-fold assignment digest mismatch for {entry.protein_id}")

    if verify_frozen_digest and PAIR_FOLD_MANIFEST_DIGEST:
        observed = pair_fold_manifest_digest(entries)
        if observed != PAIR_FOLD_MANIFEST_DIGEST:
            raise ValueError(
                f"Pair-fold manifest digest mismatch: {observed} != "
                f"{PAIR_FOLD_MANIFEST_DIGEST}"
            )


def _parse_pair_fold_row(row: Mapping[str, str]) -> PairFoldManifestEntry:
    missing = set(PAIR_FOLD_MANIFEST_COLUMNS) - set(row)
    if missing:
        raise ValueError(f"Pair-fold row is missing columns: {sorted(missing)}")
    return PairFoldManifestEntry(
        cv_protocol_version=row["cv_protocol_version"].strip(),
        manifest_version=row["manifest_version"].strip(),
        protein_id=row["protein_id"].strip(),
        source=row["source"].strip(),
        n_residues=int(row["n_residues"]),
        n_pairs=int(row["n_pairs"]),
        pair_fold_algorithm=row["pair_fold_algorithm"].strip(),
        pair_fold_base_seed=int(row["pair_fold_base_seed"]),
        pair_fold_seed=int(row["pair_fold_seed"]),
        fold_0_count=int(row["fold_0_count"]),
        fold_1_count=int(row["fold_1_count"]),
        fold_2_count=int(row["fold_2_count"]),
        fold_3_count=int(row["fold_3_count"]),
        fold_4_count=int(row["fold_4_count"]),
        assignment_digest=row["assignment_digest"].strip(),
    )


def load_pair_fold_manifest(
    path: str | Path = PAIR_FOLD_MANIFEST_PATH,
    *,
    master_entries: Sequence[ManifestEntry] | None = None,
    verify_digest: bool = True,
) -> tuple[PairFoldManifestEntry, ...]:
    """Load and validate the frozen compact per-protein pair-fold manifest."""
    with Path(path).open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != PAIR_FOLD_MANIFEST_COLUMNS:
            raise ValueError(f"Unexpected pair-fold columns: {reader.fieldnames}")
        entries = tuple(_parse_pair_fold_row(row) for row in reader)
    validate_pair_fold_manifest(
        entries,
        master_entries or load_master_manifest(),
        verify_frozen_digest=verify_digest,
    )
    return entries


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


def fold_digests_payload(
    master_entries: Sequence[ManifestEntry],
    outer_entries: Sequence[OuterFoldEntry],
    pair_entries: Sequence[PairFoldManifestEntry],
) -> dict[str, object]:
    """Build the authoritative compact provenance record for frozen folds."""
    validate_outer_fold_manifest(
        outer_entries, master_entries, verify_frozen_digest=False
    )
    validate_pair_fold_manifest(
        pair_entries, master_entries, verify_frozen_digest=False
    )
    return {
        "cv_protocol_version": PROTOCOL_VERSION,
        "construction_parameters": {
            "n_outer_folds": N_OUTER_FOLDS,
            "outer_fold_seed": OUTER_FOLD_SEED,
            "outer_fold_algorithm": OUTER_FOLD_ALGORITHM,
            "fold_0_source": "historical_master_role=test",
            "n_pair_folds": N_PAIR_FOLDS,
            "pair_fold_base_seed": PAIR_FOLD_BASE_SEED,
            "pair_fold_algorithm": PAIR_FOLD_ALGORITHM,
            "canonical_pair_order": "np.tril_indices(n_residues, k=-1)",
        },
        "master_manifest_digest": MASTER_MANIFEST_DIGEST,
        "cohort_digest": eligible_cohort_digest(master_entries),
        "outer_fold_digest": outer_fold_manifest_digest(outer_entries),
        "pair_fold_manifest_digest": pair_fold_manifest_digest(pair_entries),
        "per_fold_protein_counts": {
            str(fold): sum(entry.outer_fold == fold for entry in outer_entries)
            for fold in range(N_OUTER_FOLDS)
        },
        "per_fold_length_bin_counts": {
            str(fold): {
                str(length_bin): sum(
                    entry.outer_fold == fold and entry.length_bin == length_bin
                    for entry in outer_entries
                )
                for length_bin in range(N_SIZE_BINS)
            }
            for fold in range(N_OUTER_FOLDS)
        },
    }


def load_fold_digests(
    path: str | Path = FOLD_DIGESTS_PATH,
    *,
    master_entries: Sequence[ManifestEntry] | None = None,
    outer_entries: Sequence[OuterFoldEntry] | None = None,
    pair_entries: Sequence[PairFoldManifestEntry] | None = None,
) -> dict[str, object]:
    """Load the digest record and require exact agreement with both manifests."""
    with Path(path).open() as handle:
        payload = json.load(handle)
    master = tuple(master_entries or load_master_manifest())
    outer = tuple(outer_entries or load_outer_fold_manifest(master_entries=master))
    pairs = tuple(pair_entries or load_pair_fold_manifest(master_entries=master))
    expected = fold_digests_payload(master, outer, pairs)
    if payload != expected:
        raise ValueError("fold_digests.json does not match the frozen fold manifests")
    return payload


__all__ = [
    "AMINO_ACID_PAIR_LABELS",
    "DEVELOPMENT_SPLIT_SEED",
    "ELIGIBLE_COHORT_DIGEST",
    "FEATURE_DIM",
    "FEATURE_NAMES",
    "FIVEFOLD_METADATA_DIR",
    "FOLD_DIGESTS_PATH",
    "FLOAT_DTYPE",
    "FLOAT_EPS",
    "HISTORICAL_PROTOCOL_VERSION",
    "LAYERS",
    "MASTER_MANIFEST_COLUMNS",
    "MASTER_MANIFEST_DIGEST",
    "MASTER_MANIFEST_PATH",
    "MASTER_MANIFEST_VERSION",
    "MASTER_ROLES",
    "ManifestEntry",
    "N_CHANNELS",
    "N_OUTER_FOLDS",
    "N_PAIR_FOLDS",
    "N_SIZE_BINS",
    "OUTER_FOLD_ALGORITHM",
    "OUTER_FOLD_MANIFEST_COLUMNS",
    "OUTER_FOLD_MANIFEST_DIGEST",
    "OUTER_FOLD_MANIFEST_PATH",
    "OUTER_FOLD_MANIFEST_VERSION",
    "OUTER_FOLD_SEED",
    "OUTER_SPLIT_SEED",
    "OuterFoldEntry",
    "PAIR_FOLD_ALGORITHM",
    "PAIR_FOLD_BASE_SEED",
    "PAIR_FOLD_MANIFEST_COLUMNS",
    "PAIR_FOLD_MANIFEST_DIGEST",
    "PAIR_FOLD_MANIFEST_PATH",
    "PAIR_FOLD_MANIFEST_VERSION",
    "PairFoldManifestEntry",
    "PROTOCOL_VERSION",
    "PROTEIN_SPLITS",
    "ProteinEqualFeatureMoments",
    "SEQUENCE_BIN_LABELS",
    "SPATIAL_DISTANCE_BIN_LABELS",
    "WITHIN_PROTEIN_SPLIT_SEED",
    "assign_rank_size_bins",
    "build_master_roles",
    "build_outer_fold_manifest",
    "build_pair_fold_manifest",
    "canonical_json_bytes",
    "eligible_manifest",
    "eligible_cohort_digest",
    "extract_directional_base256",
    "fold_digests_payload",
    "load_fold_digests",
    "load_master_manifest",
    "load_outer_fold_manifest",
    "load_pair_fold_manifest",
    "manifest_digest",
    "outer_fold_manifest_digest",
    "pair_fold_assignment_digest",
    "pair_fold_manifest_digest",
    "pair_fold_seed",
    "partition_manifest_by_role",
    "protein_equal_mean",
    "protein_equal_weights",
    "semantic_digest",
    "sequence_separation_bin_codes",
    "spatial_distance_bin_codes",
    "stable_seed",
    "strict_lower_triangle_indices",
    "validate_master_manifest",
    "validate_outer_fold_manifest",
    "validate_pair_fold_manifest",
    "within_protein_pair_folds",
    "within_protein_pair_split",
    "within_protein_pair_split_digest",
    "within_protein_split_seed",
]

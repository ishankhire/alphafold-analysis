#!/usr/bin/env python3
"""Prepare frozen reference geometry for the fivefold OOF triangle replay."""

from __future__ import annotations

import os
import platform
import resource
import sys
import tempfile
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
CV_DIR = CLUSTER_DIR / "cross_validation"
for path in (SCRIPT_DIR, CLUSTER_DIR, CV_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cluster_regression_helpers import extract_ca_coordinates, resolve_combined_pdb_path  # noqa: E402
from cross_validation_common import load_cv_folds  # noqa: E402
from regression_output_helpers import atomic_write_csv, atomic_write_json  # noqa: E402
from regression_protocol import (  # noqa: E402
    pair_fold_assignment_digest,
    strict_lower_triangle_indices,
    within_protein_pair_folds,
)
from triangle_violation_subgroup_common import (  # noqa: E402
    ANALYSIS_NAME,
    ANGLE_ACUTE_MAX_DEGREES,
    ANGLE_OBTUSE_MIN_DEGREES,
    ANGLE_SIGN_RELATIVE_TOLERANCE,
    COHORT_COLUMNS,
    CV_PROTOCOL_VERSION,
    EDGE_REFERENCE_BALANCE_COLUMNS,
    EVALUATION_SCOPE,
    EXPECTED_PROTEINS,
    FAMILY_SPECS,
    OUTPUT_SCHEMA_VERSION,
    PREPARATION_FAILURE_COLUMNS,
    THRESHOLD_COLUMNS,
    TRIANGLE_TOLERANCE_ANGSTROM,
    build_reference_cache,
    edge_model_count_reference_balance,
    load_and_validate_replay_index,
    sha256_file,
    symmetric_pair_matrix,
    validate_family_manifest,
)


from canonical_storage import assert_writable, csv_root as storage_csv_root, workflow_root


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] [peak_rss={peak_rss_mb():.1f}MiB] {message}", flush=True)


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def csv_root() -> Path:
    return storage_csv_root()


def producer_root() -> Path:
    return csv_root() / "regression_5fold_v1"


def output_root() -> Path:
    return workflow_root(ANALYSIS_NAME, "TRIANGLE_OUTPUT_ROOT", analysis=False)


def validate_producers(fold_bundle) -> dict[str, object]:
    provenance: dict[str, object] = {
        "artifact_provenance_version": "triangle_violation_subgroups_5fold_v1_v1",
        "created_at": utc_now(),
        "metadata": dict(fold_bundle.fold_digests),
        "producers": {},
    }
    for spec in FAMILY_SPECS:
        root = producer_root() / spec.family_dir
        manifest = validate_family_manifest(root, spec, fold_bundle.fold_digests)
        replay_rows = load_and_validate_replay_index(root, spec, manifest)
        provenance["producers"][spec.analysis_name] = {
            "root": str(root),
            "run_manifest_sha256": sha256_file(root / "run_manifest.json"),
            "replay_index_sha256": sha256_file(root / "model_artifacts/replay_index.csv"),
            "configuration_digest": manifest["configuration_digest"],
            "n_replay_rows": len(replay_rows),
        }
        log(f"validated producer={spec.analysis_name} replay_rows={len(replay_rows)}")
    return provenance


def cohort_rows(fold_bundle) -> list[dict[str, object]]:
    rows = []
    for protein_id in fold_bundle.eligible_protein_ids:
        outer = fold_bundle.outer_by_protein[protein_id]
        pair = fold_bundle.pair_by_protein[protein_id]
        rows.append({
            "protein_id": protein_id,
            "source": outer.source,
            "n_residues": outer.n_residues,
            "n_pairs": outer.n_pairs,
            "protein_length_bin": f"length_bin_{int(outer.length_bin) + 1}",
            "outer_fold": outer.outer_fold,
            "historical_master_role": outer.historical_master_role,
            "historical_protein_split": outer.historical_protein_split,
            "cohort_digest": fold_bundle.fold_digests["cohort_digest"],
            "outer_fold_digest": fold_bundle.fold_digests["outer_fold_digest"],
            "pair_fold_manifest_digest": fold_bundle.fold_digests["pair_fold_manifest_digest"],
            "pair_fold_assignment_digest": pair.assignment_digest,
        })
    return rows


def prepare_one(outer, pair, temp_parent: Path) -> tuple[dict[str, object], list[dict[str, object]]]:
    pdb_path = resolve_combined_pdb_path(outer.protein_id, outer.source)
    if not pdb_path.is_file():
        raise FileNotFoundError(f"Final predicted PDB not found: {pdb_path}")
    ca = extract_ca_coordinates(pdb_path)
    coordinates = np.asarray(ca["coordinates"], dtype=np.float64)
    if coordinates.shape != (int(outer.n_residues), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("Final-PDB C-alpha coordinates disagree with the frozen cohort")
    pair_i, pair_j = strict_lower_triangle_indices(outer.n_residues)
    assignments = within_protein_pair_folds(outer.n_residues, outer.protein_id)
    observed_digest = pair_fold_assignment_digest(outer.n_residues, outer.protein_id)
    counts = tuple(int(value) for value in np.bincount(assignments, minlength=5))
    if (
        assignments.size != int(outer.n_pairs)
        or observed_digest != pair.assignment_digest
        or counts != tuple(int(value) for value in pair.fold_counts)
    ):
        raise ValueError("Canonical pair-fold assignment disagrees with frozen metadata")
    pair_fold_matrix = symmetric_pair_matrix(
        outer.n_residues, pair_i, pair_j, assignments, diagonal=-1,
    )
    with tempfile.TemporaryDirectory(
        prefix=f"triangle_reference_{outer.protein_id}_", dir=temp_parent,
    ) as temporary:
        cache = None
        try:
            cache, statistics = build_reference_cache(coordinates, temporary)
            balance = edge_model_count_reference_balance(
                cache, pair_fold_matrix,
                {
                    "protein_id": outer.protein_id,
                    "source": outer.source,
                    "n_residues": outer.n_residues,
                    "protein_length_bin": f"length_bin_{int(outer.length_bin) + 1}",
                },
            )
        finally:
            if cache is not None:
                cache.cleanup()
    threshold = {
        "protein_id": outer.protein_id,
        "source": outer.source,
        "n_residues": outer.n_residues,
        "protein_length_bin": f"length_bin_{int(outer.length_bin) + 1}",
        **statistics,
        "status": "complete",
    }
    if set(threshold) != set(THRESHOLD_COLUMNS):
        raise ValueError("Prepared reference threshold schema mismatch")
    return threshold, balance


def main() -> None:
    assert_writable(output_root())
    started = time.perf_counter()
    started_at = utc_now()
    root = output_root()
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    fold_bundle = load_cv_folds(metadata_dir=producer_root() / "metadata")
    if len(fold_bundle.eligible_protein_ids) != EXPECTED_PROTEINS:
        raise RuntimeError(
            f"Expected {EXPECTED_PROTEINS} eligible proteins; observed "
            f"{len(fold_bundle.eligible_protein_ids)}"
        )
    provenance = validate_producers(fold_bundle)
    atomic_write_json(metadata / "artifact_provenance.json", provenance)
    atomic_write_csv(metadata / "cohort.csv", COHORT_COLUMNS, cohort_rows(fold_bundle))

    threshold_path = metadata / "protein_reference_thresholds.csv"
    balance_path = root / "edge_model_count_reference_balance.csv"
    failure_path = metadata / "preparation_failures.csv"
    thresholds: list[dict[str, object]] = []
    balances: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    if threshold_path.is_file():
        import csv
        with threshold_path.open(newline="") as handle:
            thresholds = list(csv.DictReader(handle))
    if balance_path.is_file():
        import csv
        with balance_path.open(newline="") as handle:
            balances = list(csv.DictReader(handle))
    completed = {str(row["protein_id"]) for row in thresholds if row["status"] == "complete"}
    if len(thresholds) != len(completed) or len(balances) != 3 * len(completed):
        raise ValueError("Preparation checkpoint is partial or internally inconsistent")
    temp_parent = Path(os.environ.get("TRIANGLE_TMPDIR", os.environ.get("TMPDIR", "/tmp")))

    for index, protein_id in enumerate(fold_bundle.eligible_protein_ids, start=1):
        if protein_id in completed:
            log(f"reference protein={protein_id} resumed")
            continue
        outer = fold_bundle.outer_by_protein[protein_id]
        pair = fold_bundle.pair_by_protein[protein_id]
        protein_started = time.perf_counter()
        try:
            threshold, balance = prepare_one(outer, pair, temp_parent)
            thresholds.append(threshold)
            balances.extend(balance)
            completed.add(protein_id)
            status = "complete"
        except Exception as exc:
            status = "failed"
            failures.append({
                "protein_id": protein_id, "source": outer.source,
                "stage": "reference_geometry", "reason": f"{type(exc).__name__}: {exc}",
            })
            traceback.print_exc()
        atomic_write_csv(threshold_path, THRESHOLD_COLUMNS, thresholds)
        atomic_write_csv(balance_path, EDGE_REFERENCE_BALANCE_COLUMNS, balances)
        atomic_write_csv(failure_path, PREPARATION_FAILURE_COLUMNS, failures)
        log(
            f"reference progress={index}/{EXPECTED_PROTEINS} protein={protein_id} "
            f"status={status} elapsed={time.perf_counter() - protein_started:.1f}s"
        )

    if failures or completed != set(fold_bundle.eligible_protein_ids):
        raise RuntimeError(f"Reference preparation incomplete; failures={len(failures)}")
    if len(thresholds) != EXPECTED_PROTEINS or len(balances) != EXPECTED_PROTEINS * 3:
        raise RuntimeError("Reference output row counts are incomplete")
    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "cv_protocol_version": CV_PROTOCOL_VERSION,
        "evaluation_scope": EVALUATION_SCOPE,
        "preparation_status": "complete",
        "created_at": utc_now(),
        "started_at": started_at,
        "completed_at": utc_now(),
        "n_proteins": EXPECTED_PROTEINS,
        "n_reference_threshold_rows": len(thresholds),
        "n_edge_model_count_reference_balance_rows": len(balances),
        "cohort_digest": fold_bundle.fold_digests["cohort_digest"],
        "outer_fold_digest": fold_bundle.fold_digests["outer_fold_digest"],
        "pair_fold_manifest_digest": fold_bundle.fold_digests["pair_fold_manifest_digest"],
        "reference_definition": {
            "coordinate_source": "final_predicted_pdb_ca",
            "angle_broad_boundaries_degrees": [ANGLE_ACUTE_MAX_DEGREES, ANGLE_OBTUSE_MIN_DEGREES],
            "angle_sign_relative_tolerance": ANGLE_SIGN_RELATIVE_TOLERANCE,
            "triangle_violation_tolerance_angstrom": TRIANGLE_TOLERANCE_ANGSTROM,
            "tertile_intervals": "half_open_values_lt_q1_values_lt_q2_else_upper",
        },
        "software": {"python": platform.python_version(), "numpy": np.__version__},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mb": peak_rss_mb(),
        "files": {
            "metadata/cohort.csv": sha256_file(metadata / "cohort.csv"),
            "metadata/protein_reference_thresholds.csv": sha256_file(threshold_path),
            "metadata/preparation_failures.csv": sha256_file(failure_path),
            "metadata/artifact_provenance.json": sha256_file(metadata / "artifact_provenance.json"),
            "edge_model_count_reference_balance.csv": sha256_file(balance_path),
        },
    }
    atomic_write_json(root / "run_manifest.json", manifest)
    log(f"reference preparation complete elapsed={manifest['elapsed_seconds']:.1f}s")


if __name__ == "__main__":
    main()

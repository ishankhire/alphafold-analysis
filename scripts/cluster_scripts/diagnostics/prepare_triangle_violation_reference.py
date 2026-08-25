#!/usr/bin/env python3
"""Prepare exact frozen final-PDB triangle thresholds for locked-test proteins."""

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
GENERAL_DIR = CLUSTER_DIR / "generalized_model_ecod"
for path in (SCRIPT_DIR, CLUSTER_DIR, GENERAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from cluster_regression_helpers import extract_ca_coordinates  # noqa: E402
from general_train_test_common import (  # noqa: E402
    build_dataset_protein_manifest,
    resolve_pdb_file,
)
from regression_output_helpers import atomic_write_csv, atomic_write_json  # noqa: E402
from regression_protocol import (  # noqa: E402
    MASTER_MANIFEST_DIGEST,
    MASTER_MANIFEST_VERSION,
    load_master_manifest,
)
from triangle_violation_subgroup_common import (  # noqa: E402
    ANALYSIS_NAME,
    ANGLE_ACUTE_MAX_DEGREES,
    ANGLE_OBTUSE_MIN_DEGREES,
    ANGLE_SIGN_RELATIVE_TOLERANCE,
    EVALUATION_SCOPE,
    OUTPUT_SCHEMA_VERSION,
    PREPARATION_FAILURE_COLUMNS,
    SPLIT_ASSIGNMENT_COLUMNS,
    THRESHOLD_COLUMNS,
    TRIANGLE_TOLERANCE_ANGSTROM,
    build_reference_cache,
    sha256_file,
)


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def peak_rss_mb() -> float:
    value = float(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value / (1024.0 * 1024.0) if sys.platform == "darwin" else value / 1024.0


def output_root() -> Path:
    csv_root = Path(os.environ.get("CSV_ROOT", str(Path.home() / "csv_files"))).expanduser()
    return csv_root / ANALYSIS_NAME


def test_entries():
    entries = sorted(
        (
            entry for entry in load_master_manifest()
            if entry.eligible and entry.protein_split == "test"
        ),
        key=lambda entry: entry.protein_id,
    )
    if len(entries) != 111:
        raise RuntimeError(f"Expected 111 locked-test proteins, observed {len(entries)}")
    return entries


def discovered_records() -> dict[str, dict[str, object]]:
    records = build_dataset_protein_manifest()
    by_id = {str(record["protein_id"]): record for record in records}
    if len(by_id) != len(records):
        raise ValueError("Dataset discovery returned duplicate protein IDs")
    return by_id


def prepare_one(entry, discovered: dict[str, object]) -> dict[str, object]:
    if str(discovered["source"]) != entry.source:
        raise ValueError("Discovered source differs from frozen manifest")
    protein_dir = Path(discovered["protein_dir"])
    pdb_path = resolve_pdb_file(entry.protein_id, protein_dir, Path(discovered["pdb_root"]))
    if pdb_path is None or not Path(pdb_path).is_file():
        raise FileNotFoundError("Final predicted PDB not found")
    ca_data = extract_ca_coordinates(pdb_path)
    coordinates = np.asarray(ca_data["coordinates"], dtype=np.float64)
    if coordinates.shape != (entry.n_residues, 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError(
            f"C-alpha coordinate shape/values differ from frozen residue count {entry.n_residues}"
        )
    temp_parent = Path(os.environ.get("TRIANGLE_TMPDIR", "/tmp"))
    with tempfile.TemporaryDirectory(
        prefix=f"triangle_reference_{entry.protein_id}_", dir=temp_parent,
    ) as temporary:
        cache, statistics = build_reference_cache(coordinates, temporary)
        cache.cleanup()
    return {
        "protein_id": entry.protein_id,
        "source": entry.source,
        "protein_split": entry.protein_split,
        "n_residues": entry.n_residues,
        "protein_length_bin": f"length_bin_{entry.size_bin + 1}",
        **statistics,
        "status": "complete",
    }


def main() -> None:
    started = time.perf_counter()
    root = output_root()
    metadata = root / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)
    entries = test_entries()
    discovered = discovered_records()
    split_rows = [
        {
            "master_manifest_digest": MASTER_MANIFEST_DIGEST,
            "protein_id": entry.protein_id,
            "source": entry.source,
            "n_residues": entry.n_residues,
            "protein_length_bin": f"length_bin_{entry.size_bin + 1}",
            "master_role": entry.master_role,
            "protein_split": entry.protein_split,
        }
        for entry in entries
    ]
    atomic_write_csv(metadata / "split_assignments.csv", SPLIT_ASSIGNMENT_COLUMNS, split_rows)

    threshold_rows: list[dict[str, object]] = []
    failures: list[dict[str, object]] = []
    for number, entry in enumerate(entries, start=1):
        protein_started = time.perf_counter()
        try:
            source_record = discovered.get(entry.protein_id)
            if source_record is None:
                raise FileNotFoundError("Protein absent from dataset discovery")
            threshold_rows.append(prepare_one(entry, source_record))
            status = "complete"
        except Exception as exc:
            status = "failed"
            failures.append(
                {
                    "protein_id": entry.protein_id,
                    "source": entry.source,
                    "stage": "reference_preparation",
                    "reason": f"{type(exc).__name__}: {exc}",
                }
            )
            traceback.print_exc()
        log(
            f"reference {number}/111 protein={entry.protein_id} status={status} "
            f"elapsed={time.perf_counter() - protein_started:.1f}s peak_rss={peak_rss_mb():.1f}MiB"
        )

    atomic_write_csv(metadata / "protein_reference_thresholds.csv", THRESHOLD_COLUMNS, threshold_rows)
    atomic_write_csv(metadata / "preparation_failures.csv", PREPARATION_FAILURE_COLUMNS, failures)
    complete = len(threshold_rows) == 111 and not failures
    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "output_schema_version": OUTPUT_SCHEMA_VERSION,
        "run_status": "reference_complete" if complete else "reference_incomplete",
        "created_at": utc_now(),
        "master_manifest_version": MASTER_MANIFEST_VERSION,
        "master_manifest_digest": MASTER_MANIFEST_DIGEST,
        "protein_split": "test",
        "n_expected_proteins": 111,
        "n_prepared_proteins": len(threshold_rows),
        "evaluation_scopes": [EVALUATION_SCOPE],
        "requested_layers": list(range(48)),
        "models": [
            "general_base256_combined",
            "per_protein_base256_combined",
            "shared_xgboost_base256_combined",
            "analysis_1_sequence_gated",
        ],
        "category_configuration": {
            "near_right_degrees_inclusive": [ANGLE_ACUTE_MAX_DEGREES, ANGLE_OBTUSE_MIN_DEGREES],
            "angle_sign_relative_tolerance": ANGLE_SIGN_RELATIVE_TOLERANCE,
            "triangle_violation_tolerance_angstrom": TRIANGLE_TOLERANCE_ANGSTROM,
            "tertiles": "exact_within_protein_quantiles_half_open",
        },
        "bootstrap_resampling": False,
        "versions": {"python": platform.python_version(), "numpy": np.__version__},
        "elapsed_seconds": time.perf_counter() - started,
        "peak_rss_mb": peak_rss_mb(),
        "artifact_sha256": {
            "metadata/split_assignments.csv": sha256_file(metadata / "split_assignments.csv"),
            "metadata/protein_reference_thresholds.csv": sha256_file(
                metadata / "protein_reference_thresholds.csv"
            ),
            "metadata/preparation_failures.csv": sha256_file(metadata / "preparation_failures.csv"),
        },
    }
    atomic_write_json(root / "run_manifest.json", manifest)
    if not complete:
        raise RuntimeError(f"Reference preparation incomplete: {len(failures)} failures")
    log(f"reference preparation complete elapsed={time.perf_counter() - started:.1f}s")


if __name__ == "__main__":
    main()

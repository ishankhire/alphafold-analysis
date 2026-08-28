#!/usr/bin/env python3
"""Freeze the authoritative CAMEO triangle sample before layer analysis."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path
from typing import Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent if SCRIPT_DIR.name == "pair_triangle_dominance" else SCRIPT_DIR
for directory in (SCRIPT_DIR, CLUSTER_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from pair_triangle_dominance_common import (  # noqa: E402
    ANALYSIS_NAME,
    BASE_SEED,
    COHORT_COLUMNS,
    CONFIGURATION_DIGEST,
    SAMPLE_CAP,
    SAMPLE_MANIFEST_COLUMNS,
    SAMPLE_VERSION,
    RUNTIME_COLUMNS,
    atomic_save_npz,
    atomic_write_csv,
    atomic_write_json,
    cohort_semantic_records,
    load_sample_bundle,
    log,
    output_root,
    runtime_row,
    sample_sha256,
    sample_triples,
    semantic_digest,
    sha256_file,
    utc_now,
    validate_samples,
)
from regression_protocol import load_master_manifest  # noqa: E402


def prepare_samples(
    entries: Sequence[object],
    destination: Path,
    *,
    expected_protein_count: int = 154,
) -> dict[str, object]:
    started_at = utc_now()
    started = time.perf_counter()
    destination = Path(destination)
    metadata = destination / "metadata"
    metadata.mkdir(parents=True, exist_ok=True)

    cameo = sorted(
        (
            entry
            for entry in entries
            if str(getattr(entry, "source")) == "cameo"
            and str(getattr(entry, "eligibility_status")) == "eligible"
        ),
        key=lambda entry: str(getattr(entry, "protein_id")),
    )
    if len(cameo) != expected_protein_count:
        raise ValueError(
            f"Expected {expected_protein_count} eligible CAMEO proteins; observed {len(cameo)}"
        )
    protein_ids = [str(getattr(entry, "protein_id")) for entry in cameo]
    if len(set(protein_ids)) != len(protein_ids):
        raise ValueError("The CAMEO cohort contains duplicate protein IDs")

    cohort_rows: list[dict[str, object]] = []
    sample_rows: list[dict[str, object]] = []
    offsets = [0]
    rank_blocks: list[np.ndarray] = []
    triple_blocks: list[np.ndarray] = []
    semantic_rows: list[dict[str, object]] = []

    for entry in cameo:
        protein_id = str(getattr(entry, "protein_id"))
        n_residues = int(getattr(entry, "n_residues"))
        population = math.comb(n_residues, 3) if n_residues >= 3 else 0
        triples, ranks, seed = sample_triples(protein_id, n_residues)
        validate_samples(
            triples, ranks, n_residues, expected_count=min(SAMPLE_CAP, population)
        )
        digest = sample_sha256(protein_id, n_residues, ranks, triples)
        status = "complete" if len(triples) else "insufficient_residues"
        start = offsets[-1]
        stop = start + len(triples)
        offsets.append(stop)
        rank_blocks.append(ranks)
        triple_blocks.append(triples)

        cohort_rows.append(
            {
                "protein_id": protein_id,
                "source": "cameo",
                "manifest_version": str(getattr(entry, "manifest_version")),
                "manifest_eligibility_status": str(
                    getattr(entry, "eligibility_status")
                ),
                "expected_residue_count": n_residues,
                "cohort_status": "included",
            }
        )
        sample_rows.append(
            {
                "sample_version": SAMPLE_VERSION,
                "protein_id": protein_id,
                "n_residues": n_residues,
                "n_population_triangles": population,
                "n_sampled_triangles": len(triples),
                "base_seed": BASE_SEED,
                "derived_seed": seed,
                "rank_order": "ascending",
                "unranking_algorithm": "colex_combinadic_v1",
                "rng_algorithm": "numpy_Generator_PCG64",
                "sample_offset_start": start,
                "sample_offset_stop": stop,
                "sample_sha256": digest,
                "status": status,
            }
        )
        semantic_rows.append(
            {
                "protein_id": protein_id,
                "n_residues": n_residues,
                "n_population_triangles": population,
                "n_sampled_triangles": len(triples),
                "sample_sha256": digest,
            }
        )

    cohort_digest = semantic_digest(cohort_semantic_records(cohort_rows))
    sample_semantic_digest = semantic_digest(semantic_rows)
    provenance_path = metadata / "sample_provenance.json"
    if provenance_path.is_file():
        existing = load_sample_bundle(
            destination, expected_protein_count=expected_protein_count
        )
        if (
            existing.provenance.get("cohort_digest") != cohort_digest
            or existing.provenance.get("sample_semantic_digest")
            != sample_semantic_digest
        ):
            raise ValueError("Existing frozen sample differs from requested configuration")
        log("Existing frozen sample is valid and identical; preparation skipped")
        return existing.provenance

    all_ranks = (
        np.concatenate(rank_blocks).astype(np.int64, copy=False)
        if rank_blocks
        else np.empty(0, dtype=np.int64)
    )
    all_triples = (
        np.concatenate(triple_blocks).astype(np.int32, copy=False)
        if triple_blocks
        else np.empty((0, 3), dtype=np.int32)
    )
    cohort_path = metadata / "cameo_cohort.csv"
    manifest_path = metadata / "sample_manifest.csv"
    sample_path = metadata / "triangle_samples.npz"
    atomic_write_csv(cohort_path, COHORT_COLUMNS, cohort_rows)
    atomic_write_csv(manifest_path, SAMPLE_MANIFEST_COLUMNS, sample_rows)
    atomic_save_npz(
        sample_path,
        protein_ids=np.asarray(protein_ids, dtype=f"<U{max(map(len, protein_ids))}"),
        offsets=np.asarray(offsets, dtype=np.int64),
        sampled_ranks=all_ranks,
        tri_i=all_triples[:, 0],
        tri_j=all_triples[:, 1],
        tri_k=all_triples[:, 2],
    )

    provenance: dict[str, object] = {
        "analysis_name": ANALYSIS_NAME,
        "sample_version": SAMPLE_VERSION,
        "configuration_digest": CONFIGURATION_DIGEST,
        "base_seed": BASE_SEED,
        "sample_cap": SAMPLE_CAP,
        "protein_count": len(cameo),
        "total_sampled_triangles": int(len(all_ranks)),
        "rank_order": "ascending_within_protein",
        "unranking_algorithm": "colex_combinadic_v1",
        "rng_algorithm": "numpy_Generator_PCG64",
        "cohort_digest": cohort_digest,
        "sample_semantic_digest": sample_semantic_digest,
        "cohort_csv_sha256": sha256_file(cohort_path),
        "sample_manifest_sha256": sha256_file(manifest_path),
        "triangle_samples_npz_sha256": sha256_file(sample_path),
        "python_version": sys.version.split()[0],
        "numpy_version": np.__version__,
    }
    atomic_write_json(provenance_path, provenance)

    finished_at = utc_now()
    runtime = runtime_row(
        stage="preparation",
        layer="",
        started_at=started_at,
        finished_at=finished_at,
        elapsed_seconds=time.perf_counter() - started,
        processed_proteins=len(cameo),
        processed_sampled_triangles=len(all_ranks),
        status="complete",
    )
    atomic_write_csv(destination / "runtime.csv", RUNTIME_COLUMNS, [runtime])
    load_sample_bundle(destination, expected_protein_count=expected_protein_count)
    log(
        f"Prepared {len(cameo)} proteins and {len(all_ranks)} sampled triangles "
        f"under {destination}"
    )
    return provenance


def main() -> None:
    prepare_samples(load_master_manifest(), output_root())


if __name__ == "__main__":
    main()

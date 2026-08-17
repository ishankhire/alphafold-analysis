#!/usr/bin/env python3
"""Prepare canonical frozen-cohort metadata and development-only gated baselines."""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
GENERAL_DIR = CLUSTER_DIR / "generalized_model_ecod"
for import_dir in (SCRIPT_DIR, CLUSTER_DIR, GENERAL_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from cluster_regression_helpers import (  # noqa: E402
    build_unordered_aa_pair_codes,
    compute_distance_matrix,
    extract_ca_coordinates,
)
import general_train_test_common as general_common  # noqa: E402
from gated_linear_probe_common import (  # noqa: E402
    ANALYSIS_AA,
    ANALYSIS_JOINT,
    ANALYSIS_SEQUENCE,
    BASELINE_COLUMNS,
    CONDITIONAL_BASELINES_FILENAME,
    PREPARATION_MANIFEST_FILENAME,
    PREPARED_PROTEINS_FILENAME,
    PREPARED_PROTEIN_COLUMNS,
    configuration_digest,
    configuration_payload,
    gate_labels,
    log,
    run_mode,
    suite_root,
    utc_now,
    version_payload,
)
from regression_output_helpers import (  # noqa: E402
    atomic_savez_compressed,
    atomic_write_csv,
    atomic_write_json,
)
from regression_protocol import (  # noqa: E402
    AMINO_ACID_PAIR_LABELS,
    MASTER_MANIFEST_DIGEST,
    ManifestEntry,
    eligible_manifest,
    extract_directional_base256,
    load_master_manifest,
    protein_equal_mean,
    semantic_digest,
    sequence_separation_bin_codes,
    spatial_distance_bin_codes,
    strict_lower_triangle_indices,
)


FAILURE_COLUMNS = ("protein_id", "source", "stage", "reason")


def smoke_manifest_subset(entries: Sequence[ManifestEntry]) -> tuple[ManifestEntry, ...]:
    """Choose one frozen fit, validation, and test member from every size bin."""
    selected = []
    for size_bin in range(5):
        for role in ("fit", "validation", "test"):
            candidates = sorted(
                (
                    entry for entry in entries
                    if entry.eligible and entry.size_bin == size_bin and entry.master_role == role
                ),
                key=lambda entry: entry.protein_id,
            )
            if not candidates:
                raise ValueError(f"Smoke mode lacks a canonical {role} protein in size bin {size_bin}")
            selected.append(candidates[0])
    return tuple(sorted(selected, key=lambda entry: entry.protein_id))


def _validate_and_prepare(
    manifest_entry: ManifestEntry,
    source_entry: Mapping[str, object],
) -> dict[str, object]:
    protein_id = manifest_entry.protein_id
    protein_dir = Path(source_entry["protein_dir"])
    pdb_path = general_common.resolve_pdb_file(
        protein_id, protein_dir, Path(source_entry["pdb_root"]),
    )
    if pdb_path is None:
        raise ValueError("predicted PDB not found")
    ca_data = extract_ca_coordinates(pdb_path)
    residues = ca_data["residues"]
    coordinates = np.asarray(ca_data["coordinates"], dtype=np.float64)
    if len(residues) != manifest_entry.n_residues:
        raise ValueError(
            f"PDB CA count {len(residues)} != frozen manifest count {manifest_entry.n_residues}"
        )
    pair_i, pair_j = strict_lower_triangle_indices(manifest_entry.n_residues)
    if pair_i.size != manifest_entry.n_pairs:
        raise ValueError("Strict-lower-triangle pair count differs from frozen manifest")
    reference_path = general_common.resolve_pair_block_path(protein_dir, protein_id, 0)
    pair_block = np.load(reference_path)
    expected_shape = (manifest_entry.n_residues, manifest_entry.n_residues, 128)
    if pair_block.shape != expected_shape:
        raise ValueError(f"reference-layer shape {pair_block.shape} != {expected_shape}")
    # The frozen cohort contract validates finite canonical features at one reference layer.
    extract_directional_base256(pair_block, pair_i, pair_j)
    del pair_block
    y_true = np.asarray(
        compute_distance_matrix(coordinates)[pair_i, pair_j], dtype=np.float64,
    )
    if not np.all(np.isfinite(y_true)):
        raise ValueError("true distances contain nonfinite values")
    chain_ids = np.asarray([str(residue[0]) for residue in residues], dtype="U16")
    sequence_codes = sequence_separation_bin_codes(
        pair_i, pair_j, chain_ids=chain_ids,
    )
    aa_codes = build_unordered_aa_pair_codes(residues, pair_i, pair_j)
    distance_codes = spatial_distance_bin_codes(y_true)
    return {
        "protein_id": protein_id,
        "source": manifest_entry.source,
        "master_role": manifest_entry.master_role,
        "protein_split": manifest_entry.protein_split,
        "n_residues": manifest_entry.n_residues,
        "n_pairs": manifest_entry.n_pairs,
        "size_bin": manifest_entry.size_bin,
        "metadata": {
            "pair_i": pair_i,
            "pair_j": pair_j,
            "y_true": y_true,
            "coordinates": coordinates,
            "sequence_bin_code": sequence_codes,
            "aa_pair_code": aa_codes,
            "distance_bin_code": distance_codes,
        },
    }


def compute_conditional_baselines(records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    """Compute protein-equal gate means from canonical development proteins only."""
    development = [
        record for record in records if record["master_role"] in {"fit", "validation"}
    ]
    sizes = {
        ANALYSIS_SEQUENCE: 6,
        ANALYSIS_AA: len(AMINO_ACID_PAIR_LABELS),
        ANALYSIS_JOINT: len(AMINO_ACID_PAIR_LABELS) * 6,
    }
    accumulators = {
        analysis_name: {
            "pair_counts": np.zeros(size, dtype=np.int64),
            "protein_counts": np.zeros(size, dtype=np.int64),
            "sum_protein_means": np.zeros(size, dtype=np.float64),
        }
        for analysis_name, size in sizes.items()
    }
    for record in development:
        metadata = record["metadata"]
        y_true = np.asarray(metadata["y_true"], dtype=np.float64)
        aa_codes = np.asarray(metadata["aa_pair_code"], dtype=np.int32)
        sequence_codes = np.asarray(metadata["sequence_bin_code"], dtype=np.int32)
        code_specs = (
            (ANALYSIS_SEQUENCE, sequence_codes, sequence_codes >= 0),
            (ANALYSIS_AA, aa_codes, aa_codes >= 0),
            (
                ANALYSIS_JOINT,
                aa_codes * 6 + sequence_codes,
                (aa_codes >= 0) & (sequence_codes >= 0),
            ),
        )
        for analysis_name, codes, valid in code_specs:
            if not np.any(valid):
                continue
            accumulator = accumulators[analysis_name]
            counts = np.bincount(codes[valid], minlength=sizes[analysis_name])
            sums = np.bincount(
                codes[valid], weights=y_true[valid], minlength=sizes[analysis_name],
            )
            present = counts > 0
            accumulator["pair_counts"] += counts
            accumulator["protein_counts"][present] += 1
            accumulator["sum_protein_means"][present] += sums[present] / counts[present]

    rows = []
    for analysis_name in (ANALYSIS_SEQUENCE, ANALYSIS_AA, ANALYSIS_JOINT):
        accumulator = accumulators[analysis_name]
        for code in range(sizes[analysis_name]):
            if analysis_name == ANALYSIS_SEQUENCE:
                aa_code, sequence_code = None, code
            elif analysis_name == ANALYSIS_AA:
                aa_code, sequence_code = code, None
            else:
                aa_code, sequence_code = divmod(code, 6)
            n_proteins = int(accumulator["protein_counts"][code])
            aa_label, sequence_label = gate_labels(aa_code, sequence_code)
            rows.append({
                "analysis_name": analysis_name,
                "gate_aa_pair": aa_label,
                "gate_sequence_bin": sequence_label,
                "n_training_proteins": n_proteins,
                "n_training_pairs": int(accumulator["pair_counts"][code]),
                "prediction": (
                    float(accumulator["sum_protein_means"][code] / n_proteins)
                    if n_proteins else float("nan")
                ),
            })
    return rows


def write_prepared_suite(
    root: Path,
    records: Sequence[Mapping[str, object]],
    *,
    mode: str,
    total_manifest_entries: int,
) -> dict[str, object]:
    prepared_rows = [
        {column: record[column] for column in PREPARED_PROTEIN_COLUMNS}
        for record in records
    ]
    atomic_write_csv(root / PREPARED_PROTEINS_FILENAME, PREPARED_PROTEIN_COLUMNS, prepared_rows)
    for record in records:
        atomic_savez_compressed(
            root / "metadata" / "per_protein" / f"{record['protein_id']}.npz",
            **record["metadata"],
        )
    baseline_rows = compute_conditional_baselines(records)
    atomic_write_csv(root / CONDITIONAL_BASELINES_FILENAME, BASELINE_COLUMNS, baseline_rows)
    development_blocks = [
        record["metadata"]["y_true"]
        for record in records
        if record["master_role"] in {"fit", "validation"}
    ]
    role_counts = {
        role: sum(record["master_role"] == role for record in records)
        for role in ("fit", "validation", "test")
    }
    manifest = {
        "created_at": utc_now(),
        "run_mode": mode,
        "configuration": configuration_payload(mode),
        "configuration_digest": configuration_digest(mode),
        "master_split_digest": MASTER_MANIFEST_DIGEST,
        "total_frozen_manifest_entries": total_manifest_entries,
        "prepared_proteins": len(records),
        "prepared_role_counts": role_counts,
        "prepared_protein_digest": semantic_digest(prepared_rows),
        "global_development_mean": protein_equal_mean(development_blocks),
        "input_roots": {
            "cameo_pair_blocks": str(general_common.CAMEO_PAIR_BLOCKS_ROOT),
            "cameo_pdb": str(general_common.CAMEO_PDB_ROOT),
            "ecod_pair_blocks": str(general_common.ECOD_PAIR_BLOCKS_ROOT),
            "ecod_pdb": str(general_common.ECOD_PDB_ROOT),
        },
        "versions": version_payload(),
    }
    atomic_write_json(root / PREPARATION_MANIFEST_FILENAME, manifest)
    return manifest


def main() -> None:
    mode = run_mode()
    root = suite_root(mode)
    root.mkdir(parents=True, exist_ok=True)
    frozen = load_master_manifest()
    selected = eligible_manifest(frozen)
    if mode == "smoke":
        selected = smoke_manifest_subset(frozen)
    source_by_id = {
        row["protein_id"]: row for row in general_common.build_dataset_protein_manifest()
    }
    records = []
    failures = []
    log(f"Preparing {len(selected)} frozen-cohort proteins in {mode} mode")
    for index, entry in enumerate(selected, start=1):
        try:
            records.append(_validate_and_prepare(entry, source_by_id[entry.protein_id]))
        except Exception as exc:
            failures.append({
                "protein_id": entry.protein_id,
                "source": entry.source,
                "stage": "canonical_input_validation",
                "reason": f"{type(exc).__name__}: {exc}",
            })
        if index % 25 == 0 or index == len(selected):
            log(f"Validated {index}/{len(selected)} proteins; failures={len(failures)}")
    atomic_write_csv(root / "preparation_failures.csv", FAILURE_COLUMNS, failures)
    if failures:
        raise RuntimeError(
            f"Canonical cohort preparation failed for {len(failures)} proteins; "
            "repair inputs before fitting"
        )
    records.sort(key=lambda record: str(record["protein_id"]))
    manifest = write_prepared_suite(
        root, records, mode=mode, total_manifest_entries=len(frozen),
    )
    if mode == "production" and manifest["prepared_role_counts"] != {
        "fit": 330, "validation": 111, "test": 111,
    }:
        raise ValueError("Production preparation did not preserve canonical role counts")
    if not math.isfinite(float(manifest["global_development_mean"])):
        raise ValueError("Global development baseline is not finite")
    log(f"Preparation complete: {len(records)} proteins")


if __name__ == "__main__":
    main()

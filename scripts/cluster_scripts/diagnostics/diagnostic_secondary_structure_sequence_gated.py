#!/usr/bin/env python3
"""Replay sequence-separation-gated probes on helix and beta-sheet pairs.

This diagnostic deliberately reuses the frozen DSSP annotations and calibration
from ``diagnostic_secondary_structure_emergence``.  It does not rerun DSSP and
it decodes only the residue pairs required for helix offset triples and
beta-sheet partner/control contrasts.  The six layer jobs therefore avoid the
full all-pairs accuracy work and its sequence, distance, and amino-acid
disaggregations.

Set ``SUBSTRUCTURE_STAGE`` to ``prepare``, ``layer``, or ``consolidate``.  The
companion submit script runs the six layer tasks concurrently between the fast
validation and consolidation stages.
"""

from __future__ import annotations

import csv
import json
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
import diagnostic_secondary_structure_emergence as reference


PROJECT_ROOT = reference.PROJECT_ROOT
ANALYSIS_NAME = "diagnostic_secondary_structure_sequence_gated"
VARIANT = "analysis_1_sequence_gated"
ANALYSIS_LAYERS = reference.ANALYSIS_LAYERS
SEQUENCE_GATE_LABELS = tuple(row[2] for row in reference.SEPARATION_BINS)
FEATURE_DIM = reference.FEATURE_DIM
N_CHANNELS = reference.N_CHANNELS

# The imported row builders use this module global when labeling their output.
reference.VARIANT = VARIANT

LOCAL_SUITE_DIR = PROJECT_ROOT / "delta_files" / "consolidated"
LOCAL_REFERENCE_DIR = (
    PROJECT_ROOT / "delta_files" / "diagnostics" / "diagnostic_secondary_structure_prediction"
)
LOCAL_OUTPUT_PARENT = PROJECT_ROOT / "delta_files" / "diagnostics"

MANIFEST_FIELDS = [
    "analysis_name", "variant", "feature_dim", "layers", "analysis_split",
    "coefficient_results_dir", "suite_split_assignments", "reference_results_dir",
    "coefficients_sha256", "gate_status_sha256", "split_assignments_sha256",
    "reference_calibration_sha256", "reference_preparation_sha256", "decode_scope",
    "generated_at",
]
VALIDATION_FIELDS = [
    "category", "metric", "layer", "value", "threshold", "status", "details",
]
PER_PROTEIN_FIELDS = [
    "variant", "feature_dim", "layer", "split", "protein_id", "source",
    "n_residues", "n_decoded_pairs", "n_negative_predictions",
    "negative_prediction_fraction", "n_helix_residues", "n_strand_residues",
    "mean_helix_score", "mean_strand_score", "helix_strand_auroc",
    "mean_helix_centroid_distance", "n_helix_segments", "n_bridge_pairs",
    "n_parallel_bridge_pairs", "n_antiparallel_bridge_pairs",
]
LAYER_SUMMARY_FIELDS = [
    "variant", "feature_dim", "layer", "split", "n_proteins",
    "n_decoded_pairs", "n_negative_predictions", "negative_prediction_fraction",
    "n_helix_residues", "n_strand_residues", "mean_helix_score",
    "mean_strand_score", "helix_strand_auroc", "mean_helix_centroid_distance",
    "mean_helix_offset_2_decoded", "mean_helix_offset_3_decoded",
    "mean_helix_offset_4_decoded", "mean_helix_offset_2_true",
    "mean_helix_offset_3_true", "mean_helix_offset_4_true",
]
for _orientation in ("all", "parallel", "antiparallel"):
    LAYER_SUMMARY_FIELDS.extend(
        [
            f"sheet_{_orientation}_n_bridges",
            f"sheet_{_orientation}_mean_partner_distance",
            f"sheet_{_orientation}_mean_random_distance",
            f"sheet_{_orientation}_matched_random_margin",
            f"sheet_{_orientation}_matched_random_auroc",
            f"sheet_{_orientation}_mean_shift_1_distance",
            f"sheet_{_orientation}_shift_1_margin",
            f"sheet_{_orientation}_shift_1_auroc",
            f"sheet_{_orientation}_mean_shift_2_distance",
            f"sheet_{_orientation}_shift_2_margin",
            f"sheet_{_orientation}_shift_2_auroc",
        ]
    )


def log(message: str) -> None:
    reference.log(message)


def paths() -> tuple[Path, Path, Path, Path]:
    csv_root = Path(os.environ.get("CSV_ROOT", str(LOCAL_OUTPUT_PARENT))).expanduser().resolve()
    suite_dir = Path(
        os.environ.get("SEQUENCE_GATED_SUITE_DIR", str(LOCAL_SUITE_DIR))
    ).expanduser().resolve()
    coefficient_dir = Path(
        os.environ.get(
            "SEQUENCE_GATED_RESULTS_DIR", str(suite_dir / "analysis_1_sequence_gated")
        )
    ).expanduser().resolve()
    reference_dir = Path(
        os.environ.get("SUBSTRUCTURE_REFERENCE_DIR", str(LOCAL_REFERENCE_DIR))
    ).expanduser().resolve()
    output_dir = Path(
        os.environ.get(
            "SUBSTRUCTURE_OUTPUT_DIR", str(csv_root / ANALYSIS_NAME)
        )
    ).expanduser().resolve()
    return suite_dir, coefficient_dir, reference_dir, output_dir


def read_rows(path: Path, required_fields=()) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required_fields) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader)


def load_assignments(suite_dir: Path) -> list[dict]:
    rows = read_rows(
        suite_dir / "split_assignments.csv",
        {"protein_id", "source", "split", "n_residues"},
    )
    if len(rows) != len({row["protein_id"] for row in rows}):
        raise ValueError("Sequence-gated split contains duplicate protein IDs")
    for row in rows:
        row["source"] = row["source"].upper()
    return rows


def load_gated_coefficients(coefficient_dir: Path, layer: int) -> tuple[np.ndarray, np.ndarray]:
    rows = read_rows(
        coefficient_dir / "coefficients.csv",
        {
            "analysis_name", "layer", "train_gate_aa_pair",
            "train_gate_sequence_bin", "feature_index", "coefficient_raw_scale",
        },
    )
    selected = [
        row for row in rows
        if row["analysis_name"] == "analysis_1_sequence_gated"
        and int(row["layer"]) == layer
        and row["train_gate_aa_pair"] == "all"
    ]
    intercepts = np.empty(len(SEQUENCE_GATE_LABELS), dtype=np.float64)
    coefficients = np.empty((len(SEQUENCE_GATE_LABELS), FEATURE_DIM), dtype=np.float64)
    for gate_index, gate_label in enumerate(SEQUENCE_GATE_LABELS):
        gate_rows = [row for row in selected if row["train_gate_sequence_bin"] == gate_label]
        intercept_rows = [row for row in gate_rows if int(row["feature_index"]) == -1]
        feature_rows = sorted(
            (row for row in gate_rows if int(row["feature_index"]) >= 0),
            key=lambda row: int(row["feature_index"]),
        )
        indices = [int(row["feature_index"]) for row in feature_rows]
        if len(intercept_rows) != 1 or indices != list(range(FEATURE_DIM)):
            raise ValueError(
                f"Layer {layer} gate {gate_label} does not contain one intercept and "
                f"all {FEATURE_DIM} raw-scale coefficients"
            )
        intercepts[gate_index] = float(intercept_rows[0]["coefficient_raw_scale"])
        coefficients[gate_index] = [
            float(row["coefficient_raw_scale"]) for row in feature_rows
        ]
    if len(selected) != len(SEQUENCE_GATE_LABELS) * (FEATURE_DIM + 1):
        raise ValueError(
            f"Layer {layer} has {len(selected)} selected coefficient rows; expected "
            f"{len(SEQUENCE_GATE_LABELS) * (FEATURE_DIM + 1)}"
        )
    return intercepts, coefficients


def validate_gate_status(coefficient_dir: Path) -> None:
    rows = read_rows(
        coefficient_dir / "gate_status.csv",
        {
            "analysis_name", "layer", "train_gate_aa_pair",
            "train_gate_sequence_bin", "fit_status",
        },
    )
    selected = [
        row for row in rows
        if row["analysis_name"] == "analysis_1_sequence_gated"
        and row["train_gate_aa_pair"] == "all"
        and int(row["layer"]) in ANALYSIS_LAYERS
    ]
    expected = {
        (layer, label) for layer in ANALYSIS_LAYERS for label in SEQUENCE_GATE_LABELS
    }
    observed = {
        (int(row["layer"]), row["train_gate_sequence_bin"]) for row in selected
    }
    bad_status = [
        row for row in selected if row["fit_status"] != "fitted"
    ]
    if observed != expected or len(selected) != len(expected) or bad_status:
        raise ValueError(
            "Expected exactly 36 fully fitted sequence-gated models; "
            f"observed_keys={len(observed)}, rows={len(selected)}, bad_status={len(bad_status)}"
        )


def needed_secondary_structure_pairs(cache: dict[str, np.ndarray]) -> tuple[np.ndarray, np.ndarray]:
    """Return unique lower-triangle pairs used by the retained analyses."""
    pairs: set[tuple[int, int]] = set()
    dummy = np.zeros((len(cache["dssp_codes"]), len(cache["dssp_codes"])), dtype=np.float64)
    offset_indices, _ = reference.valid_offset_triples(
        dummy, cache["dssp_codes"], cache["chains"]
    )
    for index in offset_indices:
        for offset in (2, 3, 4):
            pairs.add((int(index + offset), int(index)))

    eligible_bridge_ids = {
        int(bridge_id)
        for bridge_id, eligible in zip(cache["bridge_id"], cache["bridge_main_eligible"])
        if int(eligible)
    }
    for _bridge_id, i, j, eligible in zip(
        cache["bridge_id"], cache["bridge_i"], cache["bridge_j"],
        cache["bridge_main_eligible"],
    ):
        if int(eligible):
            pairs.add((max(int(i), int(j)), min(int(i), int(j))))
    for bridge_id, i, j in zip(
        cache["control_bridge_id"], cache["control_i"], cache["control_j"]
    ):
        if int(bridge_id) in eligible_bridge_ids:
            pairs.add((max(int(i), int(j)), min(int(i), int(j))))

    ordered = sorted(pairs)
    return (
        np.asarray([row[0] for row in ordered], dtype=np.int32),
        np.asarray([row[1] for row in ordered], dtype=np.int32),
    )


def sequence_gate_codes(pair_i: np.ndarray, pair_j: np.ndarray, chains: np.ndarray) -> np.ndarray:
    separation = np.abs(np.asarray(pair_i) - np.asarray(pair_j))
    codes = np.full(len(separation), -1, dtype=np.int8)
    same_chain = chains[pair_i] == chains[pair_j]
    for gate_index, (lower, upper, _label) in enumerate(reference.SEPARATION_BINS):
        codes[same_chain & (separation >= lower) & (separation <= upper)] = gate_index
    return codes


def decode_needed_pairs(
    pair_block: np.ndarray,
    cache: dict[str, np.ndarray],
    intercepts: np.ndarray,
    coefficients: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Decode only retained helix/sheet pairs with their sequence gate."""
    if pair_block.ndim != 3 or pair_block.shape[:2] != (len(cache["chains"]),) * 2:
        raise ValueError(f"Unexpected pair-block/cache shapes: {pair_block.shape}")
    if pair_block.shape[2] != N_CHANNELS:
        raise ValueError(f"Expected {N_CHANNELS} pair channels, got {pair_block.shape[2]}")
    pair_i, pair_j = needed_secondary_structure_pairs(cache)
    gates = sequence_gate_codes(pair_i, pair_j, cache["chains"])
    if np.any(gates < 0):
        invalid = int(np.sum(gates < 0))
        raise ValueError(f"{invalid} retained secondary-structure pairs have no sequence gate")

    predicted = np.empty(len(pair_i), dtype=np.float64)
    for gate_index in range(len(SEQUENCE_GATE_LABELS)):
        mask = gates == gate_index
        if not np.any(mask):
            continue
        left = np.asarray(pair_block[pair_i[mask], pair_j[mask], :], dtype=np.float64)
        right = np.asarray(pair_block[pair_j[mask], pair_i[mask], :], dtype=np.float64)
        predicted[mask] = (
            intercepts[gate_index]
            + left @ coefficients[gate_index, :N_CHANNELS]
            + right @ coefficients[gate_index, N_CHANNELS:]
        )

    decoded = np.full(pair_block.shape[:2], np.nan, dtype=np.float64)
    np.fill_diagonal(decoded, 0.0)
    decoded[pair_i, pair_j] = predicted
    decoded[pair_j, pair_i] = predicted
    return decoded, predicted


def validate_reference_split(assignments: list[dict], reference_dir: Path) -> None:
    validation_rows = read_rows(
        reference_dir / "dssp_validation.csv",
        {"protein_id", "source", "split", "n_pair_residues", "usable_for_structure"},
    )
    expected = {
        row["protein_id"]: (row["source"], row["split"], int(row["n_residues"]))
        for row in assignments
    }
    observed = {
        row["protein_id"]: (
            row["source"].upper(), row["split"], int(row["n_pair_residues"])
        )
        for row in validation_rows
    }
    if expected != observed:
        missing = sorted(set(expected) - set(observed))
        extra = sorted(set(observed) - set(expected))
        mismatched = sorted(
            protein_id for protein_id in set(expected) & set(observed)
            if expected[protein_id] != observed[protein_id]
        )
        raise ValueError(
            "Sequence-gated and reference secondary-structure splits differ: "
            f"missing={missing[:5]}, extra={extra[:5]}, mismatched={mismatched[:5]}"
        )


def prepare_stage() -> None:
    started = time.time()
    suite_dir, coefficient_dir, reference_dir, output_dir = paths()
    output_dir.mkdir(parents=True, exist_ok=True)
    required_reference = (
        reference_dir / "preparation_complete.json",
        reference_dir / "calibration.json",
        reference_dir / "dssp_validation.csv",
        reference_dir / "validation.csv",
    )
    missing = [str(path) for path in required_reference if not path.is_file()]
    if missing:
        raise FileNotFoundError(
            "The frozen reference diagnostic is incomplete; rerun its preparation stage first. "
            f"Missing: {missing}"
        )

    assignments = load_assignments(suite_dir)
    validate_reference_split(assignments, reference_dir)
    test_rows = [row for row in assignments if row["split"] == "test"]
    missing_caches = [
        row["protein_id"] for row in test_rows
        if not reference.calibration_cache_path(reference_dir, row["protein_id"]).is_file()
    ]
    if missing_caches:
        raise FileNotFoundError(
            f"Reference annotation cache is missing for {len(missing_caches)} test proteins: "
            f"{missing_caches[:5]}"
        )
    for layer in ANALYSIS_LAYERS:
        load_gated_coefficients(coefficient_dir, layer)
    validate_gate_status(coefficient_dir)

    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "variant": VARIANT,
        "feature_dim": FEATURE_DIM,
        "layers": ",".join(map(str, ANALYSIS_LAYERS)),
        "analysis_split": "test",
        "coefficient_results_dir": str(coefficient_dir),
        "suite_split_assignments": str(suite_dir / "split_assignments.csv"),
        "reference_results_dir": str(reference_dir),
        "coefficients_sha256": reference.sha256_file(coefficient_dir / "coefficients.csv"),
        "gate_status_sha256": reference.sha256_file(coefficient_dir / "gate_status.csv"),
        "split_assignments_sha256": reference.sha256_file(suite_dir / "split_assignments.csv"),
        "reference_calibration_sha256": reference.sha256_file(reference_dir / "calibration.json"),
        "reference_preparation_sha256": reference.sha256_file(
            reference_dir / "preparation_complete.json"
        ),
        "decode_scope": "helix_offset_and_beta_bridge_control_pairs_only",
        "generated_at": reference.now_timestamp(),
    }
    manifest_path = output_dir / "run_manifest.csv"
    if manifest_path.is_file():
        previous = read_rows(manifest_path)
        immutable = set(MANIFEST_FIELDS) - {"generated_at"}
        mismatches = [
            key for key in immutable
            if previous and str(previous[0].get(key, "")) != str(manifest[key])
        ]
        if mismatches:
            raise RuntimeError(f"Existing output manifest differs in {sorted(mismatches)}")
    reference.atomic_write_csv(manifest_path, MANIFEST_FIELDS, [manifest])

    source_validation = read_rows(reference_dir / "validation.csv")
    true_helix_rows = [
        row for row in source_validation
        if row.get("category") == "helix_score" and row.get("metric") == "true_matrix_auroc"
    ]
    validation = [
        {
            "category": "input", "metric": "frozen_split_parity", "layer": "all",
            "value": len(assignments), "threshold": len(assignments), "status": "pass",
            "details": "protein IDs, source, split, and residue counts match the DSSP reference",
        },
        {
            "category": "input", "metric": "test_annotation_caches", "layer": "all",
            "value": len(test_rows), "threshold": len(test_rows), "status": "pass",
            "details": "all held-out proteins have frozen annotation caches",
        },
        {
            "category": "input", "metric": "fitted_sequence_gates", "layer": "all",
            "value": len(ANALYSIS_LAYERS) * len(SEQUENCE_GATE_LABELS),
            "threshold": len(ANALYSIS_LAYERS) * len(SEQUENCE_GATE_LABELS),
            "status": "pass", "details": "all raw-scale coefficient vectors loaded",
        },
    ]
    for row in true_helix_rows:
        validation.append(
            {
                "category": "reference_calibration", "metric": "true_matrix_auroc",
                "layer": "all", "value": row.get("value", ""),
                "threshold": row.get("threshold", ""), "status": row.get("status", ""),
                "details": f"split={row.get('split', '')}; inherited from reference validation",
            }
        )
    reference.atomic_write_csv(
        output_dir / "validation.csv", VALIDATION_FIELDS, validation
    )
    reference.atomic_write_csv(output_dir / "skipped_proteins.csv", reference.SKIPPED_FIELDS, [])
    completion = {
        "status": "complete", "n_assignments": len(assignments),
        "n_test_proteins": len(test_rows), "runtime_seconds": round(time.time() - started, 3),
    }
    reference.atomic_write_text(
        output_dir / "preparation_complete.json",
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
    )
    log(f"Sequence-gated preparation validation completed in {time.time() - started:.1f}s")


def layer_stage(layer: int) -> None:
    if layer not in ANALYSIS_LAYERS:
        raise ValueError(f"Layer must be one of {ANALYSIS_LAYERS}; got {layer}")
    started = time.time()
    suite_dir, coefficient_dir, reference_dir, output_dir = paths()
    if not (output_dir / "preparation_complete.json").is_file():
        raise RuntimeError("Preparation validation stage is incomplete")
    shard = output_dir / "layer_shards" / f"layer_{layer:02d}"
    shard.mkdir(parents=True, exist_ok=True)
    complete_path = shard / "complete.json"
    if complete_path.is_file():
        log(f"Layer {layer:02d} shard already complete; skipping")
        return

    calibration = json.loads((reference_dir / "calibration.json").read_text())
    intercepts, coefficients = load_gated_coefficients(coefficient_dir, layer)
    assignments = [row for row in load_assignments(suite_dir) if row["split"] == "test"]
    per_rows, helix_rows, segment_rows, bridge_rows, skipped = [], [], [], [], []

    for position, assignment in enumerate(assignments, start=1):
        protein_id = assignment["protein_id"]
        cache_path = reference.calibration_cache_path(reference_dir, protein_id)
        pair_path = reference.resolve_pair_block(protein_id, layer)
        if not cache_path.is_file() or not pair_path.is_file():
            skipped.append(
                {
                    "protein_id": protein_id, "source": assignment["source"], "split": "test",
                    "stage": f"layer_{layer}",
                    "reason": "missing_annotation_cache" if not cache_path.is_file() else "missing_pair_block",
                }
            )
            continue
        try:
            cache = reference.load_annotation_cache(cache_path)
            pair_block = np.load(pair_path)
            decoded, predicted = decode_needed_pairs(
                pair_block, cache, intercepts, coefficients
            )
            del pair_block
            true_distances = reference.distance_matrix(cache["coords"])
            protein_helix, protein_segments = reference.build_helix_rows(
                layer, assignment, cache, decoded, true_distances, calibration
            )
            protein_bridges = reference.build_bridge_rows(
                layer, assignment, cache, decoded, true_distances, calibration
            )
            h_rows = [row for row in protein_helix if row["dssp_code"] == "H"]
            e_rows = [row for row in protein_helix if row["dssp_code"] == "E"]
            per_rows.append(
                {
                    "variant": VARIANT, "feature_dim": FEATURE_DIM, "layer": layer,
                    "split": "test", "protein_id": protein_id, "source": assignment["source"],
                    "n_residues": len(cache["coords"]), "n_decoded_pairs": len(predicted),
                    "n_negative_predictions": int(np.sum(predicted < 0)),
                    "negative_prediction_fraction": float(np.mean(predicted < 0)) if len(predicted) else math.nan,
                    "n_helix_residues": len(h_rows), "n_strand_residues": len(e_rows),
                    "mean_helix_score": reference.finite_mean(row["score"] for row in h_rows),
                    "mean_strand_score": reference.finite_mean(row["score"] for row in e_rows),
                    "helix_strand_auroc": reference.safe_auc(
                        np.r_[np.ones(len(h_rows)), np.zeros(len(e_rows))],
                        np.r_[[row["score"] for row in h_rows], [row["score"] for row in e_rows]],
                    ),
                    "mean_helix_centroid_distance": reference.finite_mean(
                        row["distance_to_helix_centroid"] for row in h_rows
                    ),
                    "n_helix_segments": len(protein_segments),
                    "n_bridge_pairs": len(protein_bridges),
                    "n_parallel_bridge_pairs": sum(
                        row["orientation"] == "parallel" for row in protein_bridges
                    ),
                    "n_antiparallel_bridge_pairs": sum(
                        row["orientation"] == "antiparallel" for row in protein_bridges
                    ),
                }
            )
            helix_rows.extend(protein_helix)
            segment_rows.extend(protein_segments)
            bridge_rows.extend(protein_bridges)
        except Exception as exc:
            traceback.print_exc()
            skipped.append(
                {
                    "protein_id": protein_id, "source": assignment["source"], "split": "test",
                    "stage": f"layer_{layer}", "reason": f"{type(exc).__name__}: {exc}",
                }
            )
        if position % 10 == 0 or position == len(assignments):
            log(f"Layer {layer:02d}: processed {position}/{len(assignments)} test proteins")

    if not per_rows:
        raise RuntimeError(f"Layer {layer} produced no usable secondary-structure results")
    selected_h = [row for row in helix_rows if row["dssp_code"] == "H"]
    selected_e = [row for row in helix_rows if row["dssp_code"] == "E"]
    decoded_count = sum(int(row["n_decoded_pairs"]) for row in per_rows)
    negative_count = sum(int(row["n_negative_predictions"]) for row in per_rows)
    summary = {
        "variant": VARIANT, "feature_dim": FEATURE_DIM, "layer": layer, "split": "test",
        "n_proteins": len(per_rows), "n_decoded_pairs": decoded_count,
        "n_negative_predictions": negative_count,
        "negative_prediction_fraction": negative_count / max(decoded_count, 1),
        "n_helix_residues": len(selected_h), "n_strand_residues": len(selected_e),
        "mean_helix_score": reference.finite_mean(row["score"] for row in selected_h),
        "mean_strand_score": reference.finite_mean(row["score"] for row in selected_e),
        "helix_strand_auroc": reference.safe_auc(
            np.r_[np.ones(len(selected_h)), np.zeros(len(selected_e))],
            np.r_[[row["score"] for row in selected_h], [row["score"] for row in selected_e]],
        ),
        "mean_helix_centroid_distance": reference.finite_mean(
            row["distance_to_helix_centroid"] for row in selected_h
        ),
        **{
            f"mean_helix_offset_{offset}_decoded": reference.finite_mean(
                row[f"offset_{offset}_decoded"] for row in selected_h
            )
            for offset in (2, 3, 4)
        },
        **{
            f"mean_helix_offset_{offset}_true": reference.finite_mean(
                row[f"offset_{offset}_true"] for row in selected_h
            )
            for offset in (2, 3, 4)
        },
    }
    for orientation in ("all", "parallel", "antiparallel"):
        summary.update(reference.bridge_summary_fields(bridge_rows, orientation))

    reference.atomic_write_csv(shard / "per_protein.csv", PER_PROTEIN_FIELDS, per_rows)
    reference.atomic_write_csv(
        shard / "helix_residue_scores.csv", reference.HELIX_RESIDUE_FIELDS, helix_rows
    )
    reference.atomic_write_csv(
        shard / "helix_segment_trajectories.csv", reference.HELIX_SEGMENT_FIELDS, segment_rows
    )
    reference.atomic_write_csv(
        shard / "bridge_pair_trajectories.csv", reference.BRIDGE_FIELDS, bridge_rows
    )
    reference.atomic_write_csv(shard / "layer_summary.csv", LAYER_SUMMARY_FIELDS, [summary])
    reference.atomic_write_csv(shard / "skipped_proteins.csv", reference.SKIPPED_FIELDS, skipped)
    completion = {
        "status": "complete", "layer": layer, "n_proteins": len(per_rows),
        "n_decoded_pairs": decoded_count, "n_helix_rows": len(helix_rows),
        "n_bridge_rows": len(bridge_rows), "runtime_seconds": round(time.time() - started, 3),
    }
    reference.atomic_write_text(
        complete_path, json.dumps(completion, indent=2, sort_keys=True) + "\n"
    )
    log(f"Layer {layer:02d} complete in {time.time() - started:.1f}s; outputs={shard}")


def read_layer_rows(output_dir: Path, filename: str) -> list[dict]:
    rows = []
    for layer in ANALYSIS_LAYERS:
        path = output_dir / "layer_shards" / f"layer_{layer:02d}" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed layer output: {path}")
        rows.extend(read_rows(path))
    return rows


def consolidate_stage() -> None:
    started = time.time()
    _suite_dir, _coefficient_dir, _reference_dir, output_dir = paths()
    for layer in ANALYSIS_LAYERS:
        complete = output_dir / "layer_shards" / f"layer_{layer:02d}" / "complete.json"
        if not complete.is_file():
            raise RuntimeError(f"Layer {layer} is incomplete")

    fields = {
        "per_protein.csv": PER_PROTEIN_FIELDS,
        "helix_residue_scores.csv": reference.HELIX_RESIDUE_FIELDS,
        "helix_segment_trajectories.csv": reference.HELIX_SEGMENT_FIELDS,
        "bridge_pair_trajectories.csv": reference.BRIDGE_FIELDS,
        "layer_summary.csv": LAYER_SUMMARY_FIELDS,
    }
    combined = {filename: read_layer_rows(output_dir, filename) for filename in fields}
    reference.add_helix_trajectory_fields(combined["helix_segment_trajectories.csv"])
    reference.add_bridge_trajectory_fields(combined["bridge_pair_trajectories.csv"])
    for filename, fieldnames in fields.items():
        rows = combined[filename]
        rows.sort(
            key=lambda row: (
                int(row.get("layer", 0)), row.get("protein_id", ""),
                int(row.get("segment_id", row.get("bridge_id", 0)) or 0),
            )
        )
        reference.atomic_write_csv(output_dir / filename, fieldnames, rows)

    skipped = []
    for layer in ANALYSIS_LAYERS:
        skipped.extend(
            read_rows(output_dir / "layer_shards" / f"layer_{layer:02d}" / "skipped_proteins.csv")
        )
    deduplicated = {
        tuple(row.get(field, "") for field in reference.SKIPPED_FIELDS): row
        for row in skipped
    }
    reference.atomic_write_csv(
        output_dir / "skipped_proteins.csv", reference.SKIPPED_FIELDS,
        [deduplicated[key] for key in sorted(deduplicated)],
    )
    reference.make_plots(output_dir, combined["layer_summary.csv"])

    summaries = sorted(combined["layer_summary.csv"], key=lambda row: int(row["layer"]))
    text = [
        "Sequence-gated secondary-structure diagnostic",
        "=============================================",
        "",
        f"Generated: {reference.now_timestamp()}",
        "Coefficients: analysis_1_sequence_gated (six sequence-separation gates)",
        f"Layers: {', '.join(map(str, ANALYSIS_LAYERS))}",
        "Evaluation: frozen held-out test proteins",
        "Decode scope: helix offset and beta-sheet partner/control pairs only",
        "No sequence-separation, distance, or amino-acid-pair subgroup tables are produced.",
        "",
        "Held-out layer summary:",
    ]
    for row in summaries:
        text.append(
            f"  layer {int(row['layer']):02d}: helix/strand AUROC="
            f"{float(row['helix_strand_auroc']):.4f}; helix score="
            f"{float(row['mean_helix_score']):.4f}; strand score="
            f"{float(row['mean_strand_score']):.4f}; sheet random margin="
            f"{float(row['sheet_all_matched_random_margin']):.3f} A; register +/-1 margin="
            f"{float(row['sheet_all_shift_1_margin']):.3f} A"
        )
    text.extend(
        [
            "", "Interpretation safeguards:",
            "- Every decoded pair uses the coefficient vector for its own sequence-separation bin.",
            "- Helix and sheet calibration/controls are frozen from the original diagnostic.",
            "- Negative decoded distances are retained and counted over the sparse analysis pairs.",
            "", f"Runtime for consolidation: {time.time() - started:.1f}s",
        ]
    )
    reference.atomic_write_text(output_dir / "summary.txt", "\n".join(text) + "\n")
    completion = {
        "status": "complete", "layers": list(ANALYSIS_LAYERS),
        "runtime_seconds": round(time.time() - started, 3),
    }
    reference.atomic_write_text(
        output_dir / "consolidation_complete.json",
        json.dumps(completion, indent=2, sort_keys=True) + "\n",
    )
    log(f"Consolidation complete in {time.time() - started:.1f}s; outputs={output_dir}")


def main() -> None:
    stage = os.environ.get("SUBSTRUCTURE_STAGE", "").strip().lower()
    if stage == "prepare":
        prepare_stage()
    elif stage == "layer":
        value = os.environ.get("SUBSTRUCTURE_LAYER")
        if value is None:
            raise ValueError("SUBSTRUCTURE_LAYER is required for the layer stage")
        layer_stage(int(value))
    elif stage == "consolidate":
        consolidate_stage()
    else:
        raise ValueError("SUBSTRUCTURE_STAGE must be prepare, layer, or consolidate")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"FAILED: {type(error).__name__}: {error}")
        raise

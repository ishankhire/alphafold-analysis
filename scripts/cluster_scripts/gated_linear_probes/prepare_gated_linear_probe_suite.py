#!/usr/bin/env python3
"""Prepare the frozen split, pair metadata, baselines, and true-distance statistics."""

from __future__ import annotations

import platform
import sys
from pathlib import Path

import numpy as np
import sklearn
from sklearn.linear_model import LinearRegression

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
GENERAL_DIR = CLUSTER_DIR / "generalized_model_ecod"
for import_dir in (SCRIPT_DIR, CLUSTER_DIR, GENERAL_DIR):
    if str(import_dir) not in sys.path:
        sys.path.insert(0, str(import_dir))

from cluster_regression_helpers import (  # noqa: E402
    STANDARD_AA_PAIR_LABELS,
    build_unordered_aa_pair_codes,
    compute_distance_matrix,
    extract_ca_coordinates,
)
import general_train_test_common as general_common  # noqa: E402
from gated_linear_probe_common import (  # noqa: E402
    DISTANCE_BIN_LABELS,
    FLOAT_EPS,
    LAYERS,
    METRIC_FIELDNAMES,
    SEQUENCE_BIN_LABELS,
    atomic_savez,
    atomic_write_csv,
    atomic_write_json,
    atomic_write_text,
    canonical_json_bytes,
    configuration_digest,
    configuration_payload,
    distance_bin_codes,
    log,
    protein_equal_weights,
    repository_root,
    run_mode,
    sequence_bin_codes,
    sha256_bytes,
    suite_root,
    summarize_cell,
    utc_now,
)


SPLIT_FIELDNAMES = [
    "split", "protein_id", "source", "n_residues", "size_bin", "n_pairs", "target_variance",
]
SKIPPED_FIELDNAMES = ["protein_id", "source", "stage", "reason"]
EXCLUDED_FIELDNAMES = ["protein_id", "source", "split", "exclusion_reason", "n_pairs"]
BASELINE_FIELDNAMES = [
    "analysis_name", "train_gate_aa_pair", "train_gate_sequence_bin",
    "n_train_pairs", "n_train_proteins", "intercept_baseline_prediction",
]
RAW_MODEL_FIELDNAMES = [
    "train_gate_sequence_bin", "fit_status", "n_train_pairs", "n_train_proteins", "scaler_mean",
    "scaler_scale", "coefficient_standardized", "coefficient_raw_scale",
    "intercept_standardized", "intercept_raw_scale",
]
TRUE_STATS_FIELDNAMES = [
    "split", "category_scope", "aa_pair", "sequence_bin", "distance_bin",
    "n_pairs", "n_proteins", "pooled_mean_true_distance",
    "pooled_median_true_distance", "pooled_target_variance", "pooled_target_std",
    "mean_per_protein_true_distance", "median_per_protein_true_distance",
    "mean_per_protein_target_variance",
]
SUPPORT_FIELDNAMES = [
    "split", "category_scope", "n_expected_cells", "n_nonempty_cells",
    "n_sparse_cells", "n_pairs", "n_proteins",
]


def _validate_protein(entry):
    protein_id = entry["protein_id"]
    protein_dir = entry["protein_dir"]
    pdb_path = general_common.resolve_pdb_file(protein_id, protein_dir, entry["pdb_root"])
    if pdb_path is None:
        raise ValueError("predicted PDB not found")
    ca_data = extract_ca_coordinates(pdb_path)
    residues = ca_data["residues"]
    n_residues = len(residues)
    if n_residues < 2:
        raise ValueError("fewer than two CA residues")
    for layer in LAYERS:
        pair_path = general_common.resolve_pair_block_path(protein_dir, protein_id, layer)
        if not pair_path.exists():
            raise ValueError(f"pair block layer {layer} not found")
        pair_block = np.load(pair_path, mmap_mode="r")
        if pair_block.shape != (n_residues, n_residues, 128):
            raise ValueError(f"layer {layer} shape {pair_block.shape} != {(n_residues, n_residues, 128)}")
        del pair_block

    pair_i, pair_j = np.tril_indices(n_residues, k=-1)
    coordinates = np.asarray(ca_data["coordinates"], dtype=np.float64)
    y_true = np.asarray(compute_distance_matrix(coordinates)[pair_i, pair_j], dtype=np.float64)
    if not np.all(np.isfinite(y_true)):
        raise ValueError("true distances contain nonfinite values")
    chain_ids = np.asarray([str(row[0]) for row in residues], dtype="U16")
    chain_valid = chain_ids[pair_i] == chain_ids[pair_j]
    raw_sequence = np.abs(pair_i - pair_j).astype(np.int32)
    sequence_code = sequence_bin_codes(raw_sequence, chain_valid)
    aa_code = build_unordered_aa_pair_codes(residues, pair_i, pair_j)
    canonical_valid = aa_code >= 0
    distance_code = distance_bin_codes(y_true)
    if np.any(distance_code < 0):
        raise ValueError("valid true distance did not map to a spatial bin")
    return {
        "source": entry["source"], "protein_id": protein_id, "protein_dir": protein_dir,
        "n_residues": n_residues, "n_pairs": len(y_true),
        "target_variance": float(np.var(y_true, dtype=np.float64)),
        "metadata": {
            "y_true": y_true, "raw_sequence_separation": raw_sequence,
            "sequence_bin_code": sequence_code, "aa_pair_code": aa_code,
            "distance_bin_code": distance_code, "chain_valid_mask": chain_valid,
            "canonical_validity_mask": canonical_valid,
        },
    }


def _reuse_or_build_split(records):
    existing_path = repository_root() / "delta_files" / "general_base256_combined" / "split_assignments.csv"
    record_by_id = {row["protein_id"]: row for row in records}
    if existing_path.exists():
        import csv
        with existing_path.open(newline="") as handle:
            existing = list(csv.DictReader(handle))
        existing_ids = {row["protein_id"] for row in existing}
        compatible = existing_ids == set(record_by_id)
        if compatible:
            for row in existing:
                record = record_by_id[row["protein_id"]]
                if (int(row["n_residues"]) != record["n_residues"]
                        or int(row["size_bin"]) != record["size_bin"]):
                    compatible = False
                    break
        if compatible:
            return [
                {
                    "split": row["split"], "protein_id": row["protein_id"],
                    "source": record_by_id[row["protein_id"]]["source"],
                    "n_residues": record_by_id[row["protein_id"]]["n_residues"],
                    "size_bin": record_by_id[row["protein_id"]]["size_bin"],
                    "n_pairs": record_by_id[row["protein_id"]]["n_pairs"],
                    "target_variance": record_by_id[row["protein_id"]]["target_variance"],
                }
                for row in existing
            ], "reused_general_base256_combined"

    _, _, generated, _ = general_common.build_strict_stratified_train_test_split(records)
    return [
        {
            "split": row["split"], "protein_id": row["protein_id"],
            "source": record_by_id[row["protein_id"]]["source"],
            "n_residues": int(row["n_residues"]), "size_bin": int(row["size_bin"]),
            "n_pairs": int(row["n_pairs"]), "target_variance": float(row["target_variance"]),
        }
        for row in generated
    ], "generated_size_stratified"


def _smoke_subset(assignments):
    selected = []
    for size_bin in range(general_common.N_SIZE_BINS):
        train = sorted(
            (row for row in assignments if row["split"] == "train" and row["size_bin"] == size_bin),
            key=lambda row: row["protein_id"],
        )
        test = sorted(
            (row for row in assignments if row["split"] == "test" and row["size_bin"] == size_bin),
            key=lambda row: row["protein_id"],
        )
        if len(train) < 2 or not test:
            raise ValueError(f"Cannot construct smoke subset for size bin {size_bin}")
        selected.extend(train[:2])
        selected.append(test[0])
    return sorted(selected, key=lambda row: (row["split"], row["size_bin"], row["protein_id"]))


def _write_baselines(root, records):
    train = [row for row in records if row["split"] == "train"]
    test = [row for row in records if row["split"] == "test"]
    baseline_rows = []
    grouped = {
        "analysis_1_sequence_gated": {
            "size": 6, "counts": np.zeros(6, dtype=np.int64),
            "proteins": np.zeros(6, dtype=np.int64), "mean_sums": np.zeros(6, dtype=np.float64),
        },
        "analysis_2_aa_pair_gated": {
            "size": 210, "counts": np.zeros(210, dtype=np.int64),
            "proteins": np.zeros(210, dtype=np.int64), "mean_sums": np.zeros(210, dtype=np.float64),
        },
        "analysis_4_aa_sequence_gated": {
            "size": 1260, "counts": np.zeros(1260, dtype=np.int64),
            "proteins": np.zeros(1260, dtype=np.int64), "mean_sums": np.zeros(1260, dtype=np.float64),
        },
    }
    for record in train:
        metadata = record["metadata"]
        y = metadata["y_true"].astype(np.float64, copy=False)
        aa = metadata["aa_pair_code"].astype(np.int32, copy=False)
        sequence = metadata["sequence_bin_code"].astype(np.int32, copy=False)
        code_specs = (
            ("analysis_1_sequence_gated", sequence, sequence >= 0),
            ("analysis_2_aa_pair_gated", aa, aa >= 0),
            ("analysis_4_aa_sequence_gated", aa * 6 + sequence, (aa >= 0) & (sequence >= 0)),
        )
        for analysis, codes, valid in code_specs:
            accumulator = grouped[analysis]
            valid_codes = codes[valid]
            if not len(valid_codes):
                continue
            counts = np.bincount(valid_codes, minlength=accumulator["size"])
            sums = np.bincount(valid_codes, weights=y[valid], minlength=accumulator["size"])
            present = counts > 0
            accumulator["counts"] += counts
            accumulator["proteins"][present] += 1
            accumulator["mean_sums"][present] += sums[present] / counts[present]
    for analysis in ("analysis_1_sequence_gated", "analysis_2_aa_pair_gated",
                     "analysis_4_aa_sequence_gated"):
        accumulator = grouped[analysis]
        for code in range(accumulator["size"]):
            if analysis == "analysis_1_sequence_gated":
                aa_code, sequence_code = None, code
            elif analysis == "analysis_2_aa_pair_gated":
                aa_code, sequence_code = code, None
            else:
                aa_code, sequence_code = divmod(code, 6)
            n_proteins = int(accumulator["proteins"][code])
            prediction = (float(accumulator["mean_sums"][code] / n_proteins)
                          if n_proteins else float("nan"))
            baseline_rows.append({
                "analysis_name": analysis,
                "train_gate_aa_pair": "all" if aa_code is None else STANDARD_AA_PAIR_LABELS[aa_code],
                "train_gate_sequence_bin": "all" if sequence_code is None else SEQUENCE_BIN_LABELS[sequence_code],
                "n_train_pairs": int(accumulator["counts"][code]), "n_train_proteins": n_proteins,
                "intercept_baseline_prediction": prediction,
            })
    baseline_dir = root / "metadata" / "baselines"
    atomic_write_csv(baseline_dir / "intercept_baselines.csv", BASELINE_FIELDNAMES, baseline_rows)

    raw_models, summary_rows = [], []
    for sequence_code, label in enumerate(SEQUENCE_BIN_LABELS):
        train_x, train_y, train_offsets = [], [], []
        cursor = 0
        for protein_code, record in enumerate(train):
            metadata = record["metadata"]
            mask = metadata["sequence_bin_code"] == sequence_code
            y = metadata["y_true"][mask]
            x = metadata["raw_sequence_separation"][mask].astype(np.float64)
            if len(y):
                train_x.append(x); train_y.append(y)
                train_offsets.append((protein_code, cursor, cursor + len(y))); cursor += len(y)
        if not train_y:
            raw_models.append({
                "train_gate_sequence_bin": label, "fit_status": "insufficient_train_rows",
                "n_train_pairs": 0, "n_train_proteins": 0, "scaler_mean": float("nan"),
                "scaler_scale": float("nan"), "coefficient_standardized": float("nan"),
                "coefficient_raw_scale": float("nan"), "intercept_standardized": float("nan"),
                "intercept_raw_scale": float("nan"),
            })
            for split in ("train", "test"):
                identifiers = {
                    "analysis_name": "analysis_1_sequence_gated", "layer": "all", "split": split,
                    "train_gate_aa_pair": "all", "train_gate_sequence_bin": label,
                    "evaluation_scope": "gate_overall", "eval_aa_pair": "all",
                    "eval_sequence_bin": "all", "eval_distance_bin": "all",
                }
                row, _ = summarize_cell(
                    np.empty(0), None, float("nan"), [], identifiers,
                    "insufficient_train_rows", ["insufficient_train_rows"], 0, 0,
                )
                row["model"] = "raw_sequence_separation_ols"; row["feature_dim"] = 1
                summary_rows.append(row)
            continue
        x_train = np.concatenate(train_x).reshape(-1, 1)
        y_train = np.concatenate(train_y)
        weights = protein_equal_weights(train_offsets, len(y_train))
        per_protein_means = [np.mean(x_train[start:end], axis=0) for _, start, end in train_offsets]
        per_protein_seconds = [np.mean(x_train[start:end] ** 2, axis=0) for _, start, end in train_offsets]
        scaler_mean = np.mean(per_protein_means, axis=0)
        scaler_scale = np.sqrt(np.maximum(np.mean(per_protein_seconds, axis=0) - scaler_mean ** 2, 0.0))
        scaler_scale[scaler_scale < FLOAT_EPS] = 1.0
        model = LinearRegression().fit((x_train - scaler_mean) / scaler_scale, y_train, sample_weight=weights)
        coef_std = float(model.coef_[0]); intercept_std = float(model.intercept_)
        coef_raw = coef_std / float(scaler_scale[0])
        intercept_raw = intercept_std - coef_raw * float(scaler_mean[0])
        raw_models.append({
            "train_gate_sequence_bin": label, "fit_status": "fitted", "n_train_pairs": len(y_train),
            "n_train_proteins": len(train_offsets), "scaler_mean": scaler_mean[0],
            "scaler_scale": scaler_scale[0], "coefficient_standardized": coef_std,
            "coefficient_raw_scale": coef_raw, "intercept_standardized": intercept_std,
            "intercept_raw_scale": intercept_raw,
        })
        baseline_prediction = float(np.average(y_train, weights=weights))
        for split, split_records in (("train", train), ("test", test)):
            pieces_y, pieces_pred, slices = [], [], []
            cursor = 0
            for protein_code, record in enumerate(split_records):
                metadata = record["metadata"]
                mask = metadata["sequence_bin_code"] == sequence_code
                y = metadata["y_true"][mask]
                x = metadata["raw_sequence_separation"][mask].astype(np.float64)
                if len(y):
                    pieces_y.append(y); pieces_pred.append(model.predict(((x.reshape(-1, 1) - scaler_mean) / scaler_scale)))
                    slices.append((protein_code, cursor, cursor + len(y))); cursor += len(y)
            y = np.concatenate(pieces_y) if pieces_y else np.empty(0)
            pred = np.concatenate(pieces_pred) if pieces_pred else np.empty(0)
            identifiers = {
                "analysis_name": "analysis_1_sequence_gated", "layer": "all", "split": split,
                "train_gate_aa_pair": "all", "train_gate_sequence_bin": label,
                "evaluation_scope": "gate_overall", "eval_aa_pair": "all",
                "eval_sequence_bin": "all", "eval_distance_bin": "all",
            }
            row, _ = summarize_cell(y, pred, baseline_prediction, slices, identifiers,
                                    "fitted", [], len(y_train), len(train_offsets))
            row["model"] = "raw_sequence_separation_ols"; row["feature_dim"] = 1
            summary_rows.append(row)
    atomic_write_csv(baseline_dir / "raw_sequence_baseline_models.csv", RAW_MODEL_FIELDNAMES, raw_models)
    analysis_dir = root / "analysis_1_sequence_gated"
    atomic_write_csv(analysis_dir / "raw_sequence_baseline_summary.csv", METRIC_FIELDNAMES, summary_rows)


SCOPE_SPECS = (
    ("overall", 1),
    ("sequence_separation", 6),
    ("aa_pair", 210),
    ("spatial_distance", 5),
    ("aa_pair_x_sequence_separation", 210 * 6),
    ("sequence_separation_x_spatial_distance", 6 * 5),
    ("aa_pair_x_spatial_distance", 210 * 5),
    ("aa_pair_x_sequence_separation_x_spatial_distance", 210 * 6 * 5),
)


def _scope_codes(metadata, scope):
    aa = metadata["aa_pair_code"].astype(np.int32, copy=False)
    sequence = metadata["sequence_bin_code"].astype(np.int32, copy=False)
    distance = metadata["distance_bin_code"].astype(np.int32, copy=False)
    if scope == "overall":
        return np.zeros(len(aa), dtype=np.int32), np.ones(len(aa), dtype=bool)
    if scope == "sequence_separation":
        return sequence, sequence >= 0
    if scope == "aa_pair":
        return aa, aa >= 0
    if scope == "spatial_distance":
        return distance, distance >= 0
    if scope == "aa_pair_x_sequence_separation":
        return aa * 6 + sequence, (aa >= 0) & (sequence >= 0)
    if scope == "sequence_separation_x_spatial_distance":
        return sequence * 5 + distance, (sequence >= 0) & (distance >= 0)
    if scope == "aa_pair_x_spatial_distance":
        return aa * 5 + distance, (aa >= 0) & (distance >= 0)
    if scope == "aa_pair_x_sequence_separation_x_spatial_distance":
        return (aa * 6 + sequence) * 5 + distance, (aa >= 0) & (sequence >= 0) & (distance >= 0)
    raise ValueError(f"Unknown category scope {scope}")


def _decode_scope_code(scope, code):
    aa_code = sequence_code = distance_code = None
    if scope == "sequence_separation":
        sequence_code = code
    elif scope == "aa_pair":
        aa_code = code
    elif scope == "spatial_distance":
        distance_code = code
    elif scope == "aa_pair_x_sequence_separation":
        aa_code, sequence_code = divmod(code, 6)
    elif scope == "sequence_separation_x_spatial_distance":
        sequence_code, distance_code = divmod(code, 5)
    elif scope == "aa_pair_x_spatial_distance":
        aa_code, distance_code = divmod(code, 5)
    elif scope == "aa_pair_x_sequence_separation_x_spatial_distance":
        aa_sequence, distance_code = divmod(code, 5)
        aa_code, sequence_code = divmod(aa_sequence, 6)
    return aa_code, sequence_code, distance_code


def _write_true_distance_statistics(root, records):
    stats_rows, support_rows = [], []
    for scope, expected_count in SCOPE_SPECS:
        log(f"Analysis 3: computing {scope}")
        code_pieces, y_pieces, protein_pieces, split_pieces = [], [], [], []
        for protein_code, record in enumerate(records):
            codes, valid = _scope_codes(record["metadata"], scope)
            n_valid = int(np.count_nonzero(valid))
            if not n_valid:
                continue
            code_pieces.append(codes[valid])
            y_pieces.append(record["metadata"]["y_true"][valid].astype(np.float64, copy=False))
            protein_pieces.append(np.full(n_valid, protein_code, dtype=np.int32))
            split_pieces.append(np.full(n_valid, 0 if record["split"] == "train" else 1, dtype=np.int8))
        all_codes = np.concatenate(code_pieces) if code_pieces else np.empty(0, dtype=np.int32)
        all_y = np.concatenate(y_pieces) if y_pieces else np.empty(0, dtype=np.float64)
        all_proteins = np.concatenate(protein_pieces) if protein_pieces else np.empty(0, dtype=np.int32)
        all_splits = np.concatenate(split_pieces) if split_pieces else np.empty(0, dtype=np.int8)

        for split, split_mask in (
            ("all", np.ones(len(all_y), dtype=bool)),
            ("train", all_splits == 0),
            ("test", all_splits == 1),
        ):
            codes = all_codes[split_mask]
            y_values = all_y[split_mask]
            proteins = all_proteins[split_mask]
            nonempty = sparse = total_pairs = 0
            contributing_proteins = set()
            if len(codes):
                order = np.argsort(codes, kind="stable")
                codes = codes[order]; y_values = y_values[order]; proteins = proteins[order]
                unique_codes, starts = np.unique(codes, return_index=True)
                ends = np.r_[starts[1:], len(codes)]
                for code, start, end in zip(unique_codes, starts, ends):
                    cell_y = y_values[start:end]
                    cell_proteins = proteins[start:end]
                    protein_order = np.argsort(cell_proteins, kind="stable")
                    ordered_y = cell_y[protein_order]
                    ordered_proteins = cell_proteins[protein_order]
                    protein_ids, protein_starts = np.unique(ordered_proteins, return_index=True)
                    protein_ends = np.r_[protein_starts[1:], len(ordered_y)]
                    per_means = [float(np.mean(ordered_y[p_start:p_end]))
                                 for p_start, p_end in zip(protein_starts, protein_ends)]
                    per_variances = [float(np.var(ordered_y[p_start:p_end]))
                                     for p_start, p_end in zip(protein_starts, protein_ends)]
                    aa_code, sequence_code, distance_code = _decode_scope_code(scope, int(code))
                    stats_rows.append({
                        "split": split, "category_scope": scope,
                        "aa_pair": "all" if aa_code is None else STANDARD_AA_PAIR_LABELS[aa_code],
                        "sequence_bin": "all" if sequence_code is None else SEQUENCE_BIN_LABELS[sequence_code],
                        "distance_bin": "all" if distance_code is None else DISTANCE_BIN_LABELS[distance_code],
                        "n_pairs": len(cell_y), "n_proteins": len(protein_ids),
                        "pooled_mean_true_distance": float(np.mean(cell_y)),
                        "pooled_median_true_distance": float(np.median(cell_y)),
                        "pooled_target_variance": float(np.var(cell_y)),
                        "pooled_target_std": float(np.std(cell_y)),
                        "mean_per_protein_true_distance": float(np.mean(per_means)),
                        "median_per_protein_true_distance": float(np.median(per_means)),
                        "mean_per_protein_target_variance": float(np.mean(per_variances)),
                    })
                    nonempty += 1; total_pairs += len(cell_y); contributing_proteins.update(protein_ids.tolist())
                    if len(cell_y) < 100 or len(protein_ids) < 5:
                        sparse += 1
            support_rows.append({
                "split": split, "category_scope": scope, "n_expected_cells": expected_count,
                "n_nonempty_cells": nonempty, "n_sparse_cells": sparse,
                "n_pairs": total_pairs, "n_proteins": len(contributing_proteins),
            })
    output_dir = root / "analysis_3_true_distance_statistics"
    atomic_write_csv(output_dir / "true_distance_category_stats.csv", TRUE_STATS_FIELDNAMES, stats_rows)
    atomic_write_csv(output_dir / "category_support.csv", SUPPORT_FIELDNAMES, support_rows)
    atomic_write_text(output_dir / "summary.md", (
        "# True-distance category statistics\n\n"
        f"Prepared {len(stats_rows):,} nonempty category cells for all/train/test. "
        "See `true_distance_category_stats.csv` for exact statistics and `category_support.csv` for coverage.\n"
    ))


def main():
    mode = run_mode()
    root = suite_root(mode)
    root.mkdir(parents=True, exist_ok=True)
    log(f"Preparing gated linear probe suite in {root} ({mode})")
    valid, skipped = [], []
    manifest_entries = general_common.build_dataset_protein_manifest()
    for index, entry in enumerate(manifest_entries, start=1):
        try:
            valid.append(_validate_protein(entry))
        except Exception as exc:
            skipped.append({"protein_id": entry["protein_id"], "source": entry["source"],
                            "stage": "input_validation", "reason": str(exc)})
        if index % 25 == 0 or index == len(manifest_entries):
            log(f"Validated {index}/{len(manifest_entries)} proteins; {len(valid)} valid")
    general_common.assign_size_bins(valid, general_common.N_SIZE_BINS)
    assignments, split_source = _reuse_or_build_split(valid)
    if mode == "smoke":
        assignments = _smoke_subset(assignments)
    assignments = sorted(assignments, key=lambda row: (row["split"], row["protein_id"]))
    valid_by_id = {row["protein_id"]: row for row in valid}
    records = []
    excluded = []
    for assignment in assignments:
        record = valid_by_id[assignment["protein_id"]]
        record["split"] = assignment["split"]
        records.append(record)
        metadata = record["metadata"]
        atomic_savez(root / "metadata" / "per_protein" / f"{record['protein_id']}.npz", **metadata)
        for reason, mask in (
            ("noncanonical_or_unmapped_residue", ~metadata["canonical_validity_mask"]),
            ("interchain_sequence_separation_undefined", ~metadata["chain_valid_mask"]),
        ):
            count = int(np.count_nonzero(mask))
            if count:
                excluded.append({"protein_id": record["protein_id"], "source": record["source"],
                                 "split": record["split"], "exclusion_reason": reason, "n_pairs": count})
    split_payload = [{key: row[key] for key in SPLIT_FIELDNAMES} for row in assignments]
    split_digest = sha256_bytes(canonical_json_bytes(split_payload))
    run_manifest = {
        **configuration_payload(mode), "created_at": utc_now(),
        "configuration_digest": configuration_digest(mode), "split_digest": split_digest,
        "split_source": split_source, "manifest_proteins": len(manifest_entries),
        "valid_proteins_before_smoke_selection": len(valid), "analyzed_proteins": len(records),
        "input_roots": {
            "cameo_pair_blocks": str(general_common.CAMEO_PAIR_BLOCKS_ROOT),
            "cameo_pdb": str(general_common.CAMEO_PDB_ROOT),
            "ecod_pair_blocks": str(general_common.ECOD_PAIR_BLOCKS_ROOT),
            "ecod_pdb": str(general_common.ECOD_PDB_ROOT),
        },
        "versions": {
            "python": platform.python_version(), "numpy": np.__version__,
            "scikit_learn": sklearn.__version__,
        },
    }
    atomic_write_csv(root / "split_assignments.csv", SPLIT_FIELDNAMES, assignments)
    atomic_write_csv(root / "skipped_proteins.csv", SKIPPED_FIELDNAMES, skipped)
    atomic_write_csv(root / "excluded_pairs_summary.csv", EXCLUDED_FIELDNAMES, excluded)
    atomic_write_json(root / "run_manifest.json", run_manifest)
    log("Computing intercept and raw-sequence baselines")
    _write_baselines(root, records)
    log("Computing Analysis 3 true-distance statistics")
    _write_true_distance_statistics(root, records)
    log(f"Preparation complete: {len(records)} analyzed, {len(skipped)} skipped")


if __name__ == "__main__":
    main()

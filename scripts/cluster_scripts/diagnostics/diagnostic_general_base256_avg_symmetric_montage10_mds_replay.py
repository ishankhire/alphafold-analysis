#!/usr/bin/env python3
"""
Replay saved general_base256_avg_symmetric coefficients onto one fixed montage
protein set and export per-layer MDS reconstructions as ChimeraX-friendly PDB
files.

This script does not refit regression. It reuses saved raw-scale coefficients
and intercepts from the avg-symmetric train/test model, predicts residue-pair
distances for the montage proteins across all 48 AlphaFold layers, reconstructs
aligned CA coordinates with classical MDS, and writes both CSV diagnostics and
PDB outputs.
"""

from __future__ import annotations

import csv
import math
import os
import shutil
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPT_DIR = SCRIPT_DIR.parent
if str(CLUSTER_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_SCRIPT_DIR))

from cluster_regression_helpers import (  # noqa: E402
    classical_mds,
    compute_distance_matrix,
    compute_tm_score,
    compute_triangle_violations,
    extract_ca_coordinates,
    superimpose_structures,
)


PROJECT_ROOT = SCRIPT_DIR.parent.parent.parent

TARGET_PROTEIN_IDS = [
    "7tkv_A",
    "7tbs_A",
    "7pbk_A",
    "7mro_A",
    "7kdx_B",
    "7f6e_B",
    "7ebt_B",
    "7dq9_A",
    "7dkk_A",
    "7b3a_A",
]

VARIANT_NAME = "general_base256_avg_symmetric"
FEATURE_DIM = 128
N_CHANNELS = 128
N_LAYERS = 48
SPLIT_ID = 0
FLOAT_EPS = 1e-12
OUTPUT_BASENAME = "general_base256_avg_symmetric_montage10_mds_replay"

EXPECTED_FEATURE_NAMES = [f"pair_avg_ch_{idx}" for idx in range(N_CHANNELS)]

SEQ1_MIN_DISTANCE = 3.65
SEQ1_MAX_DISTANCE = 4.00
GLOBAL_MIN_DISTANCE = 3.65

RAW_RECONSTRUCTION_MODE = "raw"
SEQ1_RECONSTRUCTION_MODE = "seq1_clip_3p65_4p00"
GLOBAL_SEQ1_RECONSTRUCTION_MODE = "global_min_3p65_seq1_clip_3p65_4p00"

RECONSTRUCTION_MODES = [
    {
        "name": RAW_RECONSTRUCTION_MODE,
        "pdb_dir": "reconstructed_ca_layers",
        "remark": "raw predicted distances",
    },
    {
        "name": SEQ1_RECONSTRUCTION_MODE,
        "pdb_dir": "reconstructed_ca_layers_seq1_clip_3p65_4p00",
        "remark": "same-chain seq1 distances clipped to 3.65-4.00 A",
    },
    {
        "name": GLOBAL_SEQ1_RECONSTRUCTION_MODE,
        "pdb_dir": "reconstructed_ca_layers_global_min_3p65_seq1_clip_3p65_4p00",
        "remark": "global 3.65 A floor plus same-chain seq1 3.65-4.00 A clip",
    },
]
RECONSTRUCTION_MODE_BY_NAME = {mode["name"]: mode for mode in RECONSTRUCTION_MODES}

COEFFICIENT_DIR_CANDIDATES = [
    Path("/u/ikhire/general_train_test_split_csv_files/general_base256_avg_symmetric"),
    Path.home() / "general_train_test_split_csv_files" / "general_base256_avg_symmetric",
    PROJECT_ROOT / "delta_files" / "general_base256_avg_symmetric",
]

LOCAL_PAIR_BLOCKS_ROOT = PROJECT_ROOT.parent / "protein_data"
LOCAL_PDB_ROOT = PROJECT_ROOT.parent / "protein_data"
LOCAL_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "proteins"
    / "multi_protein_results"
    / "diagnostics"
    / OUTPUT_BASENAME
)

MANIFEST_CSV_NAME = "protein_manifest.csv"
COEFFICIENTS_USED_CSV_NAME = "coefficients_used.csv"
DISTANCE_METRICS_CSV_NAME = "distance_metrics.csv"
GEOMETRY_METRICS_CSV_NAME = "geometry_metrics.csv"
GEOMETRY_SUMMARY_CSV_NAME = "geometry_summary.csv"
MDS_COORDS_CSV_NAME = "mds_reconstructed_coords.csv"
SUMMARY_TXT_NAME = "summary.txt"

MANIFEST_FIELDNAMES = [
    "protein_id",
    "split",
    "size_bin",
    "n_residues",
    "n_pairs",
    "target_variance",
    "pair_block_dir",
    "pdb_source_path",
]

COEFFICIENT_FIELDNAMES = [
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "feature_index",
    "feature_name",
    "coefficient_standardized",
    "coefficient_raw_scale",
]

DISTANCE_METRIC_FIELDNAMES = [
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "target_variance",
    "per_protein_r2",
    "per_protein_mse",
    "per_protein_mae",
    "per_protein_rmse",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
]

GEOMETRY_METRIC_KEYS = [
    "per_protein_r2",
    "tm_score",
    "rmsd",
    "stress",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "neg_eigenvalue_fraction",
    "n_positive_eigenvalues",
    "positive_eigenvalue_fraction",
    "n_negative_predictions",
    "n_distances_modified",
    "modified_distance_fraction",
    "mean_abs_constraint_delta",
    "max_abs_constraint_delta",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]

GEOMETRY_METRIC_FIELDNAMES = [
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "reconstruction_mode",
    "protein_id",
    "n_residues",
    "size_bin",
    "n_pairs",
    "per_protein_r2",
    "rmsd",
    "tm_score",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "neg_eigenvalue_fraction",
    "n_positive_eigenvalues",
    "positive_eigenvalue_fraction",
    "n_negative_predictions",
    "n_distances_modified",
    "modified_distance_fraction",
    "mean_abs_constraint_delta",
    "max_abs_constraint_delta",
    "stress",
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]

GEOMETRY_SUMMARY_FIELDNAMES = [
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "reconstruction_mode",
    "n_proteins",
]
for _metric_key in GEOMETRY_METRIC_KEYS:
    GEOMETRY_SUMMARY_FIELDNAMES.extend([f"{_metric_key}_mean", f"{_metric_key}_std"])

MDS_COORD_FIELDNAMES = [
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "reconstruction_mode",
    "protein_id",
    "residue_idx",
    "chain_id",
    "residue_name",
    "residue_number",
    "mds_x",
    "mds_y",
    "mds_z",
    "pdb_x",
    "pdb_y",
    "pdb_z",
    "deviation_angstrom",
]


def now_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_progress(message):
    print(f"[{now_timestamp()}] {message}", flush=True)


def format_value(value):
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def atom_name_field(atom_name):
    if len(atom_name) >= 4:
        return atom_name[:4]
    return f" {atom_name:<3s}"


def format_atom_line(
    serial,
    residue_name,
    chain_id,
    residue_number,
    x,
    y,
    z,
    occupancy,
    bfactor,
):
    atom_field = atom_name_field("CA")
    residue_field = residue_name[:3].upper()
    safe_bfactor = min(max(float(bfactor), 0.0), 999.99)
    return (
        f"ATOM  {serial:5d} {atom_field}{' ':1s}{residue_field:>3s} {chain_id:1s}"
        f"{residue_number:4d}{' ':1s}   "
        f"{float(x):8.3f}{float(y):8.3f}{float(z):8.3f}"
        f"{float(occupancy):6.2f}{safe_bfactor:6.2f}          "
        f"{'C':>2s}\n"
    )


def format_ter_line(serial, residue_name, chain_id, residue_number):
    residue_field = residue_name[:3].upper()
    return (
        f"TER   {serial:5d}      {residue_field:>3s} {chain_id:1s}"
        f"{residue_number:4d}\n"
    )


def format_conect_line(serial_a, serial_b):
    return f"CONECT{serial_a:5d}{serial_b:5d}\n"


def write_ca_trace_pdb(output_path, residues, coordinates, b_factors, remark_line):
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        handle.write("REMARK CA trace exported for ChimeraX/PyMOL overlay\n")
        handle.write(f"REMARK {remark_line}\n")

        for serial, ((chain_id, residue_name, residue_number), coord, bfactor) in enumerate(
            zip(residues, coordinates, b_factors),
            start=1,
        ):
            handle.write(
                format_atom_line(
                    serial=serial,
                    residue_name=residue_name,
                    chain_id=chain_id,
                    residue_number=residue_number,
                    x=coord[0],
                    y=coord[1],
                    z=coord[2],
                    occupancy=1.0,
                    bfactor=bfactor,
                )
            )

        last_chain_id, last_residue_name, last_residue_number = residues[-1]
        handle.write(
            format_ter_line(
                serial=len(residues) + 1,
                residue_name=last_residue_name,
                chain_id=last_chain_id,
                residue_number=last_residue_number,
            )
        )
        for serial in range(1, len(residues)):
            handle.write(format_conect_line(serial, serial + 1))
        handle.write("END\n")


def resolve_output_root():
    csv_root = os.environ.get("CSV_ROOT")
    if csv_root:
        return Path(csv_root).expanduser().resolve() / OUTPUT_BASENAME
    return LOCAL_OUTPUT_ROOT.resolve()


def resolve_pair_blocks_root():
    root = os.environ.get("PAIR_BLOCKS_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return LOCAL_PAIR_BLOCKS_ROOT.resolve()


def resolve_pdb_root():
    root = os.environ.get("PDB_ROOT")
    if root:
        return Path(root).expanduser().resolve()
    return LOCAL_PDB_ROOT.resolve()


def resolve_coefficient_dir():
    for candidate in COEFFICIENT_DIR_CANDIDATES:
        if (candidate / "coefficients.csv").exists():
            return candidate.resolve()
    raise FileNotFoundError(
        "Could not find general_base256_avg_symmetric coefficient directory. Checked: "
        + ", ".join(str(path) for path in COEFFICIENT_DIR_CANDIDATES)
    )


def resolve_pair_block_path(pair_blocks_root, protein_id, layer):
    candidates = [
        pair_blocks_root / protein_id / f"{protein_id}_pair_block_{layer}.npy",
        pair_blocks_root / protein_id / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_pair_block_dir(pair_blocks_root, protein_id):
    candidates = [
        pair_blocks_root / protein_id / "pair_blocks",
        pair_blocks_root / protein_id,
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0]


def resolve_pdb_file(pdb_root, protein_id):
    local_dir = pdb_root / protein_id
    candidates = [
        local_dir / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
        pdb_root
        / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions"
        / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
        pdb_root / protein_id / "predicted_pdb" / f"{protein_id}_model_1_ptm_unrelaxed.pdb",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def compute_r2(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    mean_true = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot <= FLOAT_EPS:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def compute_mae(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true, y_pred):
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def build_geometry_summary_rows(geometry_rows):
    grouped = defaultdict(list)
    for row in geometry_rows:
        grouped[(int(row["layer"]), row["split"], row["reconstruction_mode"])].append(row)

    summary_rows = []
    for (layer, split, reconstruction_mode), rows in sorted(grouped.items()):
        out = {
            "variant": VARIANT_NAME,
            "feature_dim": FEATURE_DIM,
            "split_id": SPLIT_ID,
            "layer": layer,
            "split": split,
            "reconstruction_mode": reconstruction_mode,
            "n_proteins": len(rows),
        }
        for metric_key in GEOMETRY_METRIC_KEYS:
            values = [
                float(row[metric_key])
                for row in rows
                if not math.isnan(float(row[metric_key]))
            ]
            out[f"{metric_key}_mean"] = float(np.mean(values)) if values else float("nan")
            out[f"{metric_key}_std"] = float(np.std(values)) if values else float("nan")
        summary_rows.append(out)
    return summary_rows


def write_summary_text(
    output_path,
    coefficient_dir,
    pair_blocks_root,
    pdb_root,
    output_root,
    manifest_rows,
    distance_rows,
    geometry_rows,
    completed_layers,
):
    geometry_tm_by_mode_layer = defaultdict(list)
    geometry_r2_by_mode_layer = defaultdict(list)
    distance_by_layer = defaultdict(list)
    for row in geometry_rows:
        key = (row["reconstruction_mode"], int(row["layer"]))
        geometry_tm_by_mode_layer[key].append(float(row["tm_score"]))
        geometry_r2_by_mode_layer[key].append(float(row["per_protein_r2"]))
    for row in distance_rows:
        distance_by_layer[int(row["layer"])].append(float(row["per_protein_r2"]))

    def best_layer_mean(grouped_values, reconstruction_mode=None):
        best = None
        for key, values in grouped_values.items():
            if reconstruction_mode is None:
                layer = key
            else:
                mode, layer = key
                if mode != reconstruction_mode:
                    continue
            clean = [value for value in values if not math.isnan(value)]
            mean_value = float(np.mean(clean)) if clean else float("nan")
            score = -1e18 if math.isnan(mean_value) else mean_value
            if best is None or score > best[1]:
                best = (layer, score, mean_value)
        return best

    best_raw_probe_r2 = None
    for layer, values in distance_by_layer.items():
        clean = [value for value in values if not math.isnan(value)]
        mean_r2 = float(np.mean(clean)) if clean else float("nan")
        score = -1e18 if math.isnan(mean_r2) else mean_r2
        if best_raw_probe_r2 is None or score > best_raw_probe_r2[1]:
            best_raw_probe_r2 = (layer, score, mean_r2)

    with output_path.open("w") as handle:
        handle.write("general_base256_avg_symmetric montage10 MDS replay\n")
        handle.write("==================================================\n\n")
        handle.write(f"Variant:          {VARIANT_NAME}\n")
        handle.write(f"Feature dim:      {FEATURE_DIM}\n")
        handle.write(f"Split id:         {SPLIT_ID}\n")
        handle.write(f"Layers complete:  {len(completed_layers)}/{N_LAYERS}\n")
        handle.write(f"Coefficient dir:  {coefficient_dir}\n")
        handle.write(f"Pair blocks root: {pair_blocks_root}\n")
        handle.write(f"PDB root:         {pdb_root}\n")
        handle.write(f"Output root:      {output_root}\n\n")

        handle.write("Proteins\n")
        handle.write("--------\n")
        for row in manifest_rows:
            handle.write(
                f"{row['protein_id']}: split={row['split']}, residues={row['n_residues']}, "
                f"pairs={row['n_pairs']}, size_bin={row['size_bin']}\n"
            )
        handle.write("\n")

        if best_raw_probe_r2 is not None:
            handle.write(
                f"Best mean raw probe R2 layer: {best_raw_probe_r2[0]:02d} "
                f"({best_raw_probe_r2[2]:.4f})\n"
            )

        if geometry_rows:
            handle.write("\nBest geometry by reconstruction mode\n")
            handle.write("------------------------------------\n")
        for mode in RECONSTRUCTION_MODES:
            reconstruction_mode = mode["name"]
            best_r2 = best_layer_mean(geometry_r2_by_mode_layer, reconstruction_mode)
            best_tm = best_layer_mean(geometry_tm_by_mode_layer, reconstruction_mode)
            if best_r2 is None or best_tm is None:
                continue
            handle.write(
                f"{reconstruction_mode}: best input-distance R2 layer "
                f"{best_r2[0]:02d} ({best_r2[2]:.4f}); best TM-score layer "
                f"{best_tm[0]:02d} ({best_tm[2]:.4f})\n"
            )


def flush_outputs(
    output_root,
    coefficient_dir,
    pair_blocks_root,
    pdb_root,
    manifest_rows,
    coefficient_rows,
    distance_rows,
    geometry_rows,
    coord_rows,
    completed_layers,
):
    csv_root = output_root / "csv_files"
    write_csv(csv_root / MANIFEST_CSV_NAME, MANIFEST_FIELDNAMES, manifest_rows)
    write_csv(csv_root / COEFFICIENTS_USED_CSV_NAME, COEFFICIENT_FIELDNAMES, coefficient_rows)
    write_csv(csv_root / DISTANCE_METRICS_CSV_NAME, DISTANCE_METRIC_FIELDNAMES, distance_rows)
    write_csv(csv_root / GEOMETRY_METRICS_CSV_NAME, GEOMETRY_METRIC_FIELDNAMES, geometry_rows)
    write_csv(
        csv_root / GEOMETRY_SUMMARY_CSV_NAME,
        GEOMETRY_SUMMARY_FIELDNAMES,
        build_geometry_summary_rows(geometry_rows),
    )
    write_csv(csv_root / MDS_COORDS_CSV_NAME, MDS_COORD_FIELDNAMES, coord_rows)
    write_summary_text(
        output_root / SUMMARY_TXT_NAME,
        coefficient_dir=coefficient_dir,
        pair_blocks_root=pair_blocks_root,
        pdb_root=pdb_root,
        output_root=output_root,
        manifest_rows=manifest_rows,
        distance_rows=distance_rows,
        geometry_rows=geometry_rows,
        completed_layers=completed_layers,
    )


def load_coefficients(coefficient_dir):
    coeff_path = coefficient_dir / "coefficients.csv"
    coefficients_by_layer = {}
    coefficient_rows = []
    layer_feature_counts = defaultdict(int)
    layer_intercepts = defaultdict(int)

    with coeff_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["split_id"]) != SPLIT_ID:
                continue
            if row["variant"] != VARIANT_NAME:
                continue

            layer = int(row["layer"])
            feature_index = int(row["feature_index"])
            coefficient_rows.append(
                {
                    "variant": row["variant"],
                    "feature_dim": int(row["feature_dim"]),
                    "split_id": int(row["split_id"]),
                    "layer": layer,
                    "feature_index": feature_index,
                    "feature_name": row["feature_name"],
                    "coefficient_standardized": float(row["coefficient_standardized"]),
                    "coefficient_raw_scale": float(row["coefficient_raw_scale"]),
                }
            )

            bundle = coefficients_by_layer.setdefault(
                layer,
                {
                    "intercept": None,
                    "coef": np.zeros((FEATURE_DIM,), dtype=np.float64),
                    "feature_names": [None] * FEATURE_DIM,
                },
            )

            if feature_index == -1:
                bundle["intercept"] = float(row["coefficient_raw_scale"])
                layer_intercepts[layer] += 1
            else:
                if feature_index < 0 or feature_index >= FEATURE_DIM:
                    raise ValueError(f"Unexpected feature index {feature_index} at layer {layer}")
                bundle["coef"][feature_index] = float(row["coefficient_raw_scale"])
                bundle["feature_names"][feature_index] = row["feature_name"]
                layer_feature_counts[layer] += 1

    if len(coefficients_by_layer) != N_LAYERS:
        raise ValueError(
            f"Expected {N_LAYERS} layers in split-{SPLIT_ID} coefficients, "
            f"found {len(coefficients_by_layer)}"
        )

    for layer in range(N_LAYERS):
        if layer not in coefficients_by_layer:
            raise ValueError(f"Missing coefficient bundle for layer {layer}")
        if layer_feature_counts[layer] != FEATURE_DIM:
            raise ValueError(
                f"Layer {layer} expected {FEATURE_DIM} feature rows, found {layer_feature_counts[layer]}"
            )
        if layer_intercepts[layer] != 1:
            raise ValueError(
                f"Layer {layer} expected exactly one intercept row, found {layer_intercepts[layer]}"
            )

        bundle = coefficients_by_layer[layer]
        if bundle["intercept"] is None:
            raise ValueError(f"Layer {layer} is missing a raw-scale intercept")
        if bundle["feature_names"] != EXPECTED_FEATURE_NAMES:
            raise ValueError(
                f"Layer {layer} feature ordering does not match general_base256_avg_symmetric"
            )
        bundle["coef_avg"] = np.asarray(bundle["coef"], dtype=np.float64)

    coefficient_rows.sort(key=lambda row: (int(row["layer"]), int(row["feature_index"])))
    return coefficients_by_layer, coefficient_rows


def load_assignment_metadata(coefficient_dir):
    assignment_path = None
    candidate_dirs = [coefficient_dir] + [
        path.resolve() for path in COEFFICIENT_DIR_CANDIDATES if path.resolve() != coefficient_dir.resolve()
    ]
    for candidate_dir in candidate_dirs:
        candidate_path = candidate_dir / "split_assignments.csv"
        if candidate_path.exists():
            assignment_path = candidate_path
            break
    if assignment_path is None:
        raise FileNotFoundError(
            "Missing split_assignments.csv. Checked: "
            + ", ".join(str(path / "split_assignments.csv") for path in candidate_dirs)
        )

    metadata = {}
    with assignment_path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            if int(row["split_id"]) != SPLIT_ID:
                continue
            protein_id = row["protein_id"]
            if protein_id not in TARGET_PROTEIN_IDS:
                continue
            metadata[protein_id] = {
                "split": row["split"],
                "size_bin": int(row["size_bin"]),
                "n_pairs": int(row["n_pairs"]),
                "target_variance": float(row["target_variance"]),
                "n_residues": int(row["n_residues"]),
            }

    missing = sorted(set(TARGET_PROTEIN_IDS) - set(metadata))
    if missing:
        raise ValueError(f"Missing split-0 metadata for proteins: {', '.join(missing)}")
    return metadata


def predict_distances(pair_block, lt_i, lt_j, intercept, coef_avg):
    pair_avg = np.asarray(pair_block[lt_i, lt_j, :], dtype=np.float64)
    pair_avg += np.asarray(pair_block[lt_j, lt_i, :], dtype=np.float64)
    pair_avg *= 0.5
    y_pred = pair_avg @ coef_avg
    y_pred += float(intercept)
    return np.asarray(y_pred, dtype=np.float64)


def build_seq1_pair_mask(residues, lt_i, lt_j):
    mask = np.zeros((len(lt_i),), dtype=bool)
    for idx, (residue_i, residue_j) in enumerate(zip(lt_i, lt_j)):
        chain_i, _, number_i = residues[int(residue_i)]
        chain_j, _, number_j = residues[int(residue_j)]
        if chain_i == chain_j and abs(int(number_i) - int(number_j)) == 1:
            mask[idx] = True
    return mask


def build_raw_predicted_distance_matrix(n_residues, lt_i, lt_j, y_pred):
    pred_dist = np.zeros((n_residues, n_residues), dtype=np.float64)
    pred_dist[lt_i, lt_j] = y_pred
    pred_dist[lt_j, lt_i] = y_pred

    n_negative_predictions = int(np.sum(y_pred < 0.0))
    np.maximum(pred_dist, 0.0, out=pred_dist)
    np.fill_diagonal(pred_dist, 0.0)
    return pred_dist, n_negative_predictions


def summarize_constraint_delta(raw_pred_dist, constrained_pred_dist, lt_i, lt_j):
    raw_values = raw_pred_dist[lt_i, lt_j]
    constrained_values = constrained_pred_dist[lt_i, lt_j]
    deltas = constrained_values - raw_values
    abs_deltas = np.abs(deltas)
    modified_mask = abs_deltas > 1e-9
    n_modified = int(np.sum(modified_mask))
    n_pairs = int(len(raw_values))
    if n_modified == 0:
        mean_abs_delta = 0.0
        max_abs_delta = 0.0
    else:
        mean_abs_delta = float(np.mean(abs_deltas[modified_mask]))
        max_abs_delta = float(np.max(abs_deltas[modified_mask]))

    return {
        "n_distances_modified": n_modified,
        "modified_distance_fraction": float(n_modified / max(n_pairs, 1)),
        "mean_abs_constraint_delta": mean_abs_delta,
        "max_abs_constraint_delta": max_abs_delta,
    }


def build_constrained_distance_matrices(raw_pred_dist, lt_i, lt_j, seq1_pair_mask):
    matrices = {}
    raw_matrix = np.array(raw_pred_dist, dtype=np.float64, copy=True)
    np.fill_diagonal(raw_matrix, 0.0)
    matrices[RAW_RECONSTRUCTION_MODE] = raw_matrix

    seq1_matrix = np.array(raw_pred_dist, dtype=np.float64, copy=True)
    if np.any(seq1_pair_mask):
        seq1_i = lt_i[seq1_pair_mask]
        seq1_j = lt_j[seq1_pair_mask]
        seq1_values = np.clip(
            seq1_matrix[seq1_i, seq1_j],
            SEQ1_MIN_DISTANCE,
            SEQ1_MAX_DISTANCE,
        )
        seq1_matrix[seq1_i, seq1_j] = seq1_values
        seq1_matrix[seq1_j, seq1_i] = seq1_values
    np.fill_diagonal(seq1_matrix, 0.0)
    matrices[SEQ1_RECONSTRUCTION_MODE] = seq1_matrix

    global_matrix = np.array(raw_pred_dist, dtype=np.float64, copy=True)
    global_values = np.maximum(global_matrix[lt_i, lt_j], GLOBAL_MIN_DISTANCE)
    global_matrix[lt_i, lt_j] = global_values
    global_matrix[lt_j, lt_i] = global_values
    if np.any(seq1_pair_mask):
        seq1_i = lt_i[seq1_pair_mask]
        seq1_j = lt_j[seq1_pair_mask]
        seq1_values = np.clip(
            global_matrix[seq1_i, seq1_j],
            SEQ1_MIN_DISTANCE,
            SEQ1_MAX_DISTANCE,
        )
        global_matrix[seq1_i, seq1_j] = seq1_values
        global_matrix[seq1_j, seq1_i] = seq1_values
    np.fill_diagonal(global_matrix, 0.0)
    matrices[GLOBAL_SEQ1_RECONSTRUCTION_MODE] = global_matrix

    constraint_summaries = {
        mode["name"]: summarize_constraint_delta(
            raw_pred_dist,
            matrices[mode["name"]],
            lt_i,
            lt_j,
        )
        for mode in RECONSTRUCTION_MODES
    }
    return matrices, constraint_summaries


def reconstruct_geometry(true_coords, pred_dist, n_negative_predictions):
    n_residues = int(true_coords.shape[0])

    mds_coords, eigenvalues = classical_mds(pred_dist, n_dims=3)

    positive_mask = eigenvalues > 0.0
    n_positive = int(np.sum(positive_mask))
    positive_fraction = float(n_positive / max(len(eigenvalues), 1))
    sum_positive = float(np.sum(eigenvalues[positive_mask])) if np.any(positive_mask) else 1.0

    frac_top3 = float(np.sum(np.maximum(eigenvalues[:3], 0.0)) / sum_positive)
    negative_mask = eigenvalues < 0.0
    n_negative = int(np.sum(negative_mask))
    neg_fraction = (
        float(abs(np.sum(eigenvalues[negative_mask])) / sum_positive)
        if n_negative > 0
        else 0.0
    )

    reflected_coords = mds_coords.copy()
    reflected_coords[:, 2] *= -1.0

    transformed_orig, rmsd_orig = superimpose_structures(true_coords, mds_coords)
    transformed_flip, rmsd_flip = superimpose_structures(true_coords, reflected_coords)

    if rmsd_flip < rmsd_orig:
        transformed_coords = transformed_flip
        rmsd = float(rmsd_flip)
    else:
        transformed_coords = transformed_orig
        rmsd = float(rmsd_orig)

    tm_score = compute_tm_score(true_coords, transformed_coords, n_residues)

    mds_dist = compute_distance_matrix(transformed_coords)
    tri_i, tri_j = np.triu_indices(n_residues, k=1)
    d_pred = pred_dist[tri_i, tri_j]
    d_mds = mds_dist[tri_i, tri_j]
    denom = float(np.sum(d_pred * d_pred))
    if denom <= FLOAT_EPS:
        stress = float("nan")
    else:
        stress = float(np.sqrt(np.sum((d_pred - d_mds) ** 2) / denom))

    triangle_metrics = compute_triangle_violations(pred_dist)
    deviations = np.sqrt(np.sum((true_coords - transformed_coords) ** 2, axis=1))

    return {
        "transformed_coords": np.asarray(transformed_coords, dtype=np.float64),
        "deviations": np.asarray(deviations, dtype=np.float64),
        "rmsd": rmsd,
        "tm_score": float(tm_score),
        "frac_top3_eigenvalues": frac_top3,
        "n_negative_eigenvalues": n_negative,
        "neg_eigenvalue_fraction": neg_fraction,
        "n_positive_eigenvalues": n_positive,
        "positive_eigenvalue_fraction": positive_fraction,
        "n_negative_predictions": n_negative_predictions,
        "stress": stress,
        **triangle_metrics,
    }


def load_proteins(pair_blocks_root, pdb_root, assignment_metadata):
    manifest_rows = []
    protein_records = []

    for protein_id in TARGET_PROTEIN_IDS:
        pdb_path = resolve_pdb_file(pdb_root, protein_id)
        if pdb_path is None:
            raise FileNotFoundError(f"No predicted PDB found for {protein_id}")

        pair_block_path = resolve_pair_block_path(pair_blocks_root, protein_id, 0)
        if not pair_block_path.exists():
            raise FileNotFoundError(f"Pair block 0 not found for {protein_id}: {pair_block_path}")

        ca_data = extract_ca_coordinates(pdb_path)
        residues = list(ca_data["residues"])
        coords = np.asarray(ca_data["coordinates"], dtype=np.float64)
        n_residues = int(coords.shape[0])
        if n_residues < 2:
            raise ValueError(f"{protein_id} has fewer than 2 CA atoms")

        pb0 = np.load(str(pair_block_path), mmap_mode="r")
        pb_dim = int(pb0.shape[0])
        del pb0

        if pb_dim != n_residues:
            raise ValueError(
                f"{protein_id} CA count {n_residues} does not match pair-block dimension {pb_dim}"
            )

        lt_i, lt_j = np.tril_indices(n_residues, k=-1)
        seq1_pair_mask = build_seq1_pair_mask(residues, lt_i, lt_j)
        y_true = np.asarray(compute_distance_matrix(coords)[lt_i, lt_j], dtype=np.float64)
        n_pairs = int(y_true.size)
        target_variance = float(np.var(y_true, dtype=np.float64))
        if target_variance <= FLOAT_EPS:
            target_variance = FLOAT_EPS

        meta = assignment_metadata[protein_id]
        if meta["n_residues"] != n_residues:
            raise ValueError(
                f"{protein_id} split assignment residues {meta['n_residues']} != loaded {n_residues}"
            )
        if meta["n_pairs"] != n_pairs:
            raise ValueError(f"{protein_id} split assignment pairs {meta['n_pairs']} != loaded {n_pairs}")
        if not math.isclose(meta["target_variance"], target_variance, rel_tol=1e-6, abs_tol=1e-6):
            raise ValueError(
                f"{protein_id} split assignment variance {meta['target_variance']} != loaded {target_variance}"
            )

        manifest_rows.append(
            {
                "protein_id": protein_id,
                "split": meta["split"],
                "size_bin": meta["size_bin"],
                "n_residues": n_residues,
                "n_pairs": n_pairs,
                "target_variance": target_variance,
                "pair_block_dir": str(resolve_pair_block_dir(pair_blocks_root, protein_id)),
                "pdb_source_path": str(pdb_path),
            }
        )

        protein_records.append(
            {
                "protein_id": protein_id,
                "split": meta["split"],
                "size_bin": meta["size_bin"],
                "n_residues": n_residues,
                "n_pairs": n_pairs,
                "target_variance": target_variance,
                "lt_i": lt_i,
                "lt_j": lt_j,
                "seq1_pair_mask": seq1_pair_mask,
                "y_true": y_true,
                "coordinates": coords,
                "residues": residues,
                "pdb_path": pdb_path,
            }
        )

    return manifest_rows, protein_records


def copy_original_pdbs(output_root, protein_records):
    for record in protein_records:
        protein_root = output_root / "pdb_files" / record["protein_id"]
        original_dir = protein_root / "original_full"
        original_dir.mkdir(parents=True, exist_ok=True)
        shutil.copy2(record["pdb_path"], original_dir / record["pdb_path"].name)


def write_reference_ca_pdbs(output_root, protein_records):
    for record in protein_records:
        protein_root = output_root / "pdb_files" / record["protein_id"]
        reference_path = protein_root / "reference_ca" / f"{record['protein_id']}_reference_ca.pdb"
        zeros = np.zeros((record["n_residues"],), dtype=np.float64)
        write_ca_trace_pdb(
            reference_path,
            residues=record["residues"],
            coordinates=record["coordinates"],
            b_factors=zeros,
            remark_line="Reference CA coordinates from original predicted PDB",
        )


def write_reconstructed_pdb(
    output_root,
    record,
    layer,
    reconstruction_mode,
    transformed_coords,
    deviations,
):
    protein_root = output_root / "pdb_files" / record["protein_id"]
    mode_config = RECONSTRUCTION_MODE_BY_NAME[reconstruction_mode]
    reconstructed_path = (
        protein_root
        / mode_config["pdb_dir"]
        / f"{record['protein_id']}_mds_reconstructed_ca_layer{layer:02d}.pdb"
    )
    write_ca_trace_pdb(
        reconstructed_path,
        residues=record["residues"],
        coordinates=transformed_coords,
        b_factors=deviations,
        remark_line=(
            f"Split-0 general_base256_avg_symmetric MDS reconstruction "
            f"for layer {layer:02d}; mode={reconstruction_mode}; {mode_config['remark']}"
        ),
    )


def main():
    t0 = time.time()
    pair_blocks_root = resolve_pair_blocks_root()
    pdb_root = resolve_pdb_root()
    coefficient_dir = resolve_coefficient_dir()
    output_root = resolve_output_root()
    output_root.mkdir(parents=True, exist_ok=True)

    log_progress(f"Pair blocks root: {pair_blocks_root}")
    log_progress(f"PDB root:         {pdb_root}")
    log_progress(f"Coefficient dir:  {coefficient_dir}")
    log_progress(f"Output root:      {output_root}")
    log_progress(f"Proteins:         {', '.join(TARGET_PROTEIN_IDS)}")

    coefficients_by_layer, coefficient_rows = load_coefficients(coefficient_dir)
    assignment_metadata = load_assignment_metadata(coefficient_dir)
    manifest_rows, protein_records = load_proteins(
        pair_blocks_root=pair_blocks_root,
        pdb_root=pdb_root,
        assignment_metadata=assignment_metadata,
    )

    log_progress("Copying original full predicted PDB files")
    copy_original_pdbs(output_root, protein_records)
    log_progress("Writing reference CA-only PDB files")
    write_reference_ca_pdbs(output_root, protein_records)

    distance_rows = []
    geometry_rows = []
    coord_rows = []
    completed_layers = []

    flush_outputs(
        output_root=output_root,
        coefficient_dir=coefficient_dir,
        pair_blocks_root=pair_blocks_root,
        pdb_root=pdb_root,
        manifest_rows=manifest_rows,
        coefficient_rows=coefficient_rows,
        distance_rows=distance_rows,
        geometry_rows=geometry_rows,
        coord_rows=coord_rows,
        completed_layers=completed_layers,
    )

    for layer in range(N_LAYERS):
        layer_t0 = time.time()
        layer_bundle = coefficients_by_layer[layer]
        log_progress(f"Layer {layer:02d}: starting replay")

        for record in protein_records:
            protein_id = record["protein_id"]
            pair_block_path = resolve_pair_block_path(pair_blocks_root, protein_id, layer)
            if not pair_block_path.exists():
                raise FileNotFoundError(
                    f"Missing pair block for {protein_id} layer {layer}: {pair_block_path}"
                )

            pair_block = np.load(str(pair_block_path))
            y_pred = predict_distances(
                pair_block=pair_block,
                lt_i=record["lt_i"],
                lt_j=record["lt_j"],
                intercept=layer_bundle["intercept"],
                coef_avg=layer_bundle["coef_avg"],
            )
            del pair_block

            per_protein_r2 = compute_r2(record["y_true"], y_pred)
            mse_value = float(np.mean((y_pred - record["y_true"]) ** 2))
            distance_rows.append(
                {
                    "variant": VARIANT_NAME,
                    "feature_dim": FEATURE_DIM,
                    "split_id": SPLIT_ID,
                    "layer": layer,
                    "split": record["split"],
                    "protein_id": protein_id,
                    "n_residues": record["n_residues"],
                    "size_bin": record["size_bin"],
                    "n_pairs": record["n_pairs"],
                    "target_variance": record["target_variance"],
                    "per_protein_r2": per_protein_r2,
                    "per_protein_mse": mse_value,
                    "per_protein_mae": compute_mae(record["y_true"], y_pred),
                    "per_protein_rmse": compute_rmse(record["y_true"], y_pred),
                    "mean_true_distance": float(np.mean(record["y_true"])),
                    "mean_predicted_distance": float(np.mean(y_pred)),
                    "mean_error_pred_minus_true": float(np.mean(y_pred - record["y_true"])),
                }
            )

            raw_pred_dist, n_negative_predictions = build_raw_predicted_distance_matrix(
                n_residues=record["n_residues"],
                lt_i=record["lt_i"],
                lt_j=record["lt_j"],
                y_pred=y_pred,
            )
            distance_matrices, constraint_summaries = build_constrained_distance_matrices(
                raw_pred_dist=raw_pred_dist,
                lt_i=record["lt_i"],
                lt_j=record["lt_j"],
                seq1_pair_mask=record["seq1_pair_mask"],
            )

            mode_log_parts = []
            for mode in RECONSTRUCTION_MODES:
                reconstruction_mode = mode["name"]
                pred_dist_for_mds = distance_matrices[reconstruction_mode]
                constrained_y_pred = pred_dist_for_mds[record["lt_i"], record["lt_j"]]
                reconstruction_r2 = compute_r2(record["y_true"], constrained_y_pred)
                constraint_summary = constraint_summaries[reconstruction_mode]

                geometry = reconstruct_geometry(
                    true_coords=record["coordinates"],
                    pred_dist=pred_dist_for_mds,
                    n_negative_predictions=n_negative_predictions,
                )
                geometry_rows.append(
                    {
                        "variant": VARIANT_NAME,
                        "feature_dim": FEATURE_DIM,
                        "split_id": SPLIT_ID,
                        "layer": layer,
                        "split": record["split"],
                        "reconstruction_mode": reconstruction_mode,
                        "protein_id": protein_id,
                        "n_residues": record["n_residues"],
                        "size_bin": record["size_bin"],
                        "n_pairs": record["n_pairs"],
                        "per_protein_r2": reconstruction_r2,
                        "rmsd": geometry["rmsd"],
                        "tm_score": geometry["tm_score"],
                        "frac_top3_eigenvalues": geometry["frac_top3_eigenvalues"],
                        "n_negative_eigenvalues": geometry["n_negative_eigenvalues"],
                        "neg_eigenvalue_fraction": geometry["neg_eigenvalue_fraction"],
                        "n_positive_eigenvalues": geometry["n_positive_eigenvalues"],
                        "positive_eigenvalue_fraction": geometry["positive_eigenvalue_fraction"],
                        "n_negative_predictions": geometry["n_negative_predictions"],
                        "n_distances_modified": constraint_summary["n_distances_modified"],
                        "modified_distance_fraction": constraint_summary["modified_distance_fraction"],
                        "mean_abs_constraint_delta": constraint_summary[
                            "mean_abs_constraint_delta"
                        ],
                        "max_abs_constraint_delta": constraint_summary[
                            "max_abs_constraint_delta"
                        ],
                        "stress": geometry["stress"],
                        "triangle_violation_fraction": geometry["triangle_violation_fraction"],
                        "triangle_mean_violation": geometry["triangle_mean_violation"],
                        "triangle_max_violation": geometry["triangle_max_violation"],
                        "triangle_no_violation_fraction": geometry[
                            "triangle_no_violation_fraction"
                        ],
                        "triangle_violation_0_5_fraction": geometry[
                            "triangle_violation_0_5_fraction"
                        ],
                        "triangle_violation_5_10_fraction": geometry[
                            "triangle_violation_5_10_fraction"
                        ],
                        "triangle_violation_10_plus_fraction": geometry[
                            "triangle_violation_10_plus_fraction"
                        ],
                    }
                )

                for residue_idx, (
                    (chain_id, residue_name, residue_number),
                    true_coord,
                    recon_coord,
                    deviation,
                ) in enumerate(
                    zip(
                        record["residues"],
                        record["coordinates"],
                        geometry["transformed_coords"],
                        geometry["deviations"],
                    )
                ):
                    coord_rows.append(
                        {
                            "variant": VARIANT_NAME,
                            "feature_dim": FEATURE_DIM,
                            "split_id": SPLIT_ID,
                            "layer": layer,
                            "split": record["split"],
                            "reconstruction_mode": reconstruction_mode,
                            "protein_id": protein_id,
                            "residue_idx": residue_idx,
                            "chain_id": chain_id,
                            "residue_name": residue_name,
                            "residue_number": residue_number,
                            "mds_x": float(recon_coord[0]),
                            "mds_y": float(recon_coord[1]),
                            "mds_z": float(recon_coord[2]),
                            "pdb_x": float(true_coord[0]),
                            "pdb_y": float(true_coord[1]),
                            "pdb_z": float(true_coord[2]),
                            "deviation_angstrom": float(deviation),
                        }
                    )

                write_reconstructed_pdb(
                    output_root=output_root,
                    record=record,
                    layer=layer,
                    reconstruction_mode=reconstruction_mode,
                    transformed_coords=geometry["transformed_coords"],
                    deviations=geometry["deviations"],
                )

                mode_log_parts.append(
                    f"{reconstruction_mode}: R2={reconstruction_r2:.4f}, "
                    f"TM={geometry['tm_score']:.4f}, RMSD={geometry['rmsd']:.4f}"
                )

            log_progress(
                f"Layer {layer:02d} {protein_id}: probe_R2={per_protein_r2:.4f}; "
                + "; ".join(mode_log_parts)
            )

        completed_layers.append(layer)
        flush_outputs(
            output_root=output_root,
            coefficient_dir=coefficient_dir,
            pair_blocks_root=pair_blocks_root,
            pdb_root=pdb_root,
            manifest_rows=manifest_rows,
            coefficient_rows=coefficient_rows,
            distance_rows=distance_rows,
            geometry_rows=geometry_rows,
            coord_rows=coord_rows,
            completed_layers=completed_layers,
        )
        log_progress(f"Layer {layer:02d}: checkpoint written in {time.time() - layer_t0:.1f}s")

    log_progress(f"Finished replay in {time.time() - t0:.1f}s")


if __name__ == "__main__":
    try:
        main()
    except Exception as exc:  # pragma: no cover - diagnostic entrypoint
        log_progress(f"ERROR: {exc}")
        traceback.print_exc()
        raise

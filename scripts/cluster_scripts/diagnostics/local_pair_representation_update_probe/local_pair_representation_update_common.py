#!/usr/bin/env python3
"""Shared definitions for the expanded local pair-representation update probe."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
from datetime import datetime
from pathlib import Path

import numpy as np


N_CHANNELS = 128
DISTANCE_FEATURE_DIM = 256
TRANSITIONS = ((0, 1), (10, 11), (20, 21), (30, 31), (40, 41), (46, 47))
VARIANTS = (
    "center", "triangle_r1", "triangle_r2", "triangle_r3",
    "triangle_r5", "triangle_r7", "triangle_r9",
)
SMALL_VARIANTS = VARIANTS[:5]
LARGE_VARIANTS = VARIANTS[5:]
SOURCE_VARIANT = "general_base256_combined"
OUTPUT_NAME = "local_pair_representation_update_probe"
SPLIT_ID = 0
MIN_SEQUENCE_SEPARATION = 4
MAX_SEQUENCE_SEPARATION = 12
RELATIVE_RIDGE_ALPHA = 1e-3
LSQR_TOLERANCE = 1e-3
LSQR_MAX_ITER = 100
BOOTSTRAP_REPLICATES = 200
RANDOM_SEED = 20260812

CAMEO_PAIR_BLOCKS_ROOT = Path("/work/hdd/bekh/thayes/openfold_reps")
ECOD_PAIR_BLOCKS_ROOT = Path(
    "/work/hdd/bfna/thayes/ecod/experiments/representation_saves"
)
CAMEO_PDB_ROOT = Path("/work/hdd/bekh/thayes/openfold_reps_outputs_sm")
ECOD_PDB_ROOT = Path(
    "/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm"
)

STANDARD_AAS = (
    "ALA", "ARG", "ASN", "ASP", "CYS", "GLU", "GLN", "GLY", "HIS", "ILE",
    "LEU", "LYS", "MET", "PHE", "PRO", "SER", "THR", "TRP", "TYR", "VAL",
)
AA_TO_INDEX = {aa: index for index, aa in enumerate(STANDARD_AAS)}
AA_ALIASES = {
    "MSE": "MET", "SEC": "CYS", "PYL": "LYS",
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS",
    "HSD": "HIS", "HSE": "HIS",
}
AA_PAIRS = tuple(
    (STANDARD_AAS[first], STANDARD_AAS[second])
    for first in range(len(STANDARD_AAS))
    for second in range(first, len(STANDARD_AAS))
)
AA_PAIR_TO_INDEX = {pair: index for index, pair in enumerate(AA_PAIRS)}


def now_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message):
    print(f"[{now_timestamp()}] {message}", flush=True)


def format_value(value):
    if value is None:
        return ""
    if isinstance(value, (bool, np.bool_)):
        return "true" if bool(value) else "false"
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        value = float(value)
        return f"{value:.10g}" if math.isfinite(value) else "nan"
    return value


def write_csv_atomic(path, rows, fieldnames=None):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fieldnames is None:
        keys = set()
        for row in rows:
            keys.update(row)
        fieldnames = sorted(keys)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key)) for key in fieldnames})
    os.replace(temporary, path)


def write_json_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def write_text_atomic(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.write_text(value)
    os.replace(temporary, path)


def variants_for_group(group):
    if group == "small":
        return SMALL_VARIANTS
    if group == "large":
        return LARGE_VARIANTS
    if group == "all":
        return VARIANTS
    raise ValueError(f"Unknown VARIANT_GROUP {group!r}; expected small, large, or all")


def task_spec(task_id, group):
    variants = variants_for_group(group)
    total = len(TRANSITIONS) * len(variants)
    task_id = int(task_id)
    if task_id < 0 or task_id >= total:
        raise ValueError(f"Task ID {task_id} is outside 0..{total - 1} for {group}")
    transition_index, variant_index = divmod(task_id, len(variants))
    return (*TRANSITIONS[transition_index], variants[variant_index])


def triangle_radius(variant):
    if variant.startswith("triangle_r"):
        return int(variant.rsplit("r", 1)[1])
    if variant == "center":
        return 0
    raise ValueError(f"Unknown feature variant {variant!r}")


def feature_schema(variant):
    rows = []

    def add(name, kind, shell="", slot_role="", leg="", channel="", validity_for=""):
        rows.append({
            "variant": variant,
            "feature_index": len(rows),
            "feature_name": name,
            "feature_kind": kind,
            "shell": shell,
            "slot_role": slot_role,
            "leg": leg,
            "channel": channel,
            "validity_for": validity_for,
        })

    for channel in range(N_CHANNELS):
        add(f"center_avg_ch_{channel}", "center", channel=channel)
    for shell in range(1, triangle_radius(variant) + 1):
        roles = (
            ("i_minus", f"i-{shell}"), ("i_plus", f"i+{shell}"),
            ("j_minus", f"j-{shell}"), ("j_plus", f"j+{shell}"),
        )
        for role, slot_role in roles:
            slot = f"shell_{shell}_{role}"
            for leg in ("i_to_k", "k_to_j"):
                for channel in range(N_CHANNELS):
                    add(
                        f"{slot}_{leg}_avg_ch_{channel}", "triangle_edge",
                        shell=shell, slot_role=slot_role, leg=leg, channel=channel,
                    )
            add(
                f"{slot}_valid", "validity_mask", shell=shell,
                slot_role=slot_role, validity_for=slot,
            )
    width = len(rows)
    for row in rows:
        row["feature_dim"] = width
    return rows


def feature_dim(variant):
    return N_CHANNELS + 4 * triangle_radius(variant) * (2 * N_CHANNELS + 1)


def source_for_protein(protein_id):
    return "cameo" if "_" in str(protein_id) else "ecod"


def pair_block_path(protein_id, layer):
    if source_for_protein(protein_id) == "cameo":
        root = Path(os.environ.get("CAMEO_PAIR_BLOCKS_ROOT", CAMEO_PAIR_BLOCKS_ROOT))
        return root / protein_id / f"{protein_id}_pair_block_{layer}.npy"
    root = Path(os.environ.get("ECOD_PAIR_BLOCKS_ROOT", ECOD_PAIR_BLOCKS_ROOT))
    return root / f"{protein_id}_save_layers_0_47" / f"{protein_id}_pair_block_{layer}.npy"


def pdb_path(protein_id):
    if source_for_protein(protein_id) == "cameo":
        root = Path(os.environ.get("CAMEO_PDB_ROOT", CAMEO_PDB_ROOT))
    else:
        root = Path(os.environ.get("ECOD_PDB_ROOT", ECOD_PDB_ROOT))
    return (
        root / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions" / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    )


def normalize_aa(name):
    value = AA_ALIASES.get(str(name).upper(), str(name).upper())
    return value if value in AA_TO_INDEX else None


def extract_pdb_residue_names(path):
    residues, seen = [], set()
    with Path(path).open() as handle:
        for line in handle:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA" or line[16:17] not in {" ", "A"}:
                continue
            key = (line[21:22], line[22:26], line[26:27])
            if key in seen:
                continue
            seen.add(key)
            residues.append(normalize_aa(line[17:20].strip()))
    return residues


def aa_pair_indices(residue_names, pair_i, pair_j):
    result = np.full(pair_i.size, -1, dtype=np.int16)
    for row, (i, j) in enumerate(zip(pair_i.tolist(), pair_j.tolist())):
        first, second = normalize_aa(residue_names[i]), normalize_aa(residue_names[j])
        if first is None or second is None:
            continue
        pair = (
            (first, second)
            if AA_TO_INDEX[first] <= AA_TO_INDEX[second]
            else (second, first)
        )
        result[row] = AA_PAIR_TO_INDEX[pair]
    return result


def local_pair_indices(n_residues):
    i_parts, j_parts, separation_parts = [], [], []
    for separation in range(MIN_SEQUENCE_SEPARATION, MAX_SEQUENCE_SEPARATION + 1):
        count = max(int(n_residues) - separation, 0)
        if count:
            i = np.arange(count, dtype=np.int64)
            i_parts.append(i)
            j_parts.append(i + separation)
            separation_parts.append(np.full(count, separation, dtype=np.int16))
    if not i_parts:
        return np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int16)
    return np.concatenate(i_parts), np.concatenate(j_parts), np.concatenate(separation_parts)


def local_pair_count(n_residues):
    return sum(
        max(int(n_residues) - separation, 0)
        for separation in range(MIN_SEQUENCE_SEPARATION, MAX_SEQUENCE_SEPARATION + 1)
    )


def load_split_records(path):
    records, source_rows, seen = [], [], set()
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["split_id"]) != SPLIT_ID:
                continue
            protein_id = row["protein_id"]
            if protein_id in seen:
                raise ValueError(f"Duplicate split assignment for {protein_id}")
            seen.add(protein_id)
            source_rows.append(dict(row))
            records.append({
                "protein_id": protein_id,
                "source": source_for_protein(protein_id),
                "split": row["split"],
                "n_residues": int(row["n_residues"]),
                "n_local_pairs": local_pair_count(int(row["n_residues"])),
            })
    if not records:
        raise ValueError(f"No split_id={SPLIT_ID} records in {path}")
    return records, source_rows


def load_distance_coefficients(path, layer):
    coefficients = np.full(DISTANCE_FEATURE_DIM, np.nan, dtype=np.float64)
    intercept, source_rows, seen = None, [], set()
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if row["variant"] != SOURCE_VARIANT:
                continue
            if int(row["split_id"]) != SPLIT_ID or int(row["layer"]) != int(layer):
                continue
            if int(row["feature_dim"]) != DISTANCE_FEATURE_DIM:
                raise ValueError(f"Unexpected layer-{layer} feature width")
            source_rows.append(dict(row))
            index, value = int(row["feature_index"]), float(row["coefficient_raw_scale"])
            if index == -1:
                if intercept is not None or row["feature_name"] != "intercept":
                    raise ValueError(f"Invalid or duplicate layer-{layer} intercept")
                intercept = value
            else:
                if index in seen or index < 0 or index >= DISTANCE_FEATURE_DIM:
                    raise ValueError(f"Invalid or duplicate layer-{layer} coefficient {index}")
                expected = (
                    f"pair_ij_ch_{index}" if index < N_CHANNELS
                    else f"pair_ji_ch_{index - N_CHANNELS}"
                )
                if row["feature_name"] != expected:
                    raise ValueError(f"Layer-{layer} feature {index} is not {expected}")
                coefficients[index] = value
                seen.add(index)
    if intercept is None or len(seen) != DISTANCE_FEATURE_DIM or not np.all(np.isfinite(coefficients)):
        raise ValueError(f"Incomplete saved distance coefficients for layer {layer}")
    canonical = "\n".join(
        ",".join(f"{key}={row[key]}" for key in sorted(row))
        for row in sorted(source_rows, key=lambda item: int(item["feature_index"]))
    )
    digest = hashlib.sha256(canonical.encode()).hexdigest()
    for row in source_rows:
        row["coefficient_bundle_sha256"] = digest
    return float(intercept), coefficients, source_rows, digest


def decode_selected_distances(pb, i, j, intercept, coefficients):
    values = np.full(i.size, intercept, dtype=np.float64)
    values += np.asarray(pb[j, i, :], dtype=np.float64) @ coefficients[:N_CHANNELS]
    values += np.asarray(pb[i, j, :], dtype=np.float64) @ coefficients[N_CHANNELS:]
    return values


def symmetric_edge(pb, first, second):
    return 0.5 * (
        np.asarray(pb[first, second, :], dtype=np.float64)
        + np.asarray(pb[second, first, :], dtype=np.float64)
    )


def build_feature_chunk(pb, pair_i, pair_j, variant):
    x = np.zeros((pair_i.size, feature_dim(variant)), dtype=np.float64)
    x[:, :N_CHANNELS] = symmetric_edge(pb, pair_i, pair_j)
    column = N_CHANNELS
    n_residues = int(pb.shape[0])
    for shell in range(1, triangle_radius(variant) + 1):
        for k in (pair_i - shell, pair_i + shell, pair_j - shell, pair_j + shell):
            valid = (k >= 0) & (k < n_residues) & (k != pair_i) & (k != pair_j)
            if np.any(valid):
                i_valid, j_valid, k_valid = pair_i[valid], pair_j[valid], k[valid]
                x[valid, column:column + N_CHANNELS] = symmetric_edge(pb, i_valid, k_valid)
                x[valid, column + N_CHANNELS:column + 2 * N_CHANNELS] = symmetric_edge(
                    pb, k_valid, j_valid
                )
                x[valid, column + 2 * N_CHANNELS] = 1.0
            column += 2 * N_CHANNELS + 1
    if column != x.shape[1]:
        raise AssertionError(f"Feature writer stopped at {column}, expected {x.shape[1]}")
    return x


def build_offsets(records, split):
    offsets, position = [], 0
    for record in records:
        if record["split"] != split:
            continue
        end = position + record["n_local_pairs"]
        offsets.append((record, position, end))
        position = end
    return offsets, position


def protein_equal_weights(offsets):
    total = offsets[-1][2] if offsets else 0
    weights = np.empty(total, dtype=np.float64)
    for _record, start, end in offsets:
        weights[start:end] = total / (len(offsets) * max(end - start, 1))
    return weights


def protein_equal_scaler(x, offsets):
    mean = np.zeros(x.shape[1], dtype=np.float64)
    second = np.zeros(x.shape[1], dtype=np.float64)
    for _record, start, end in offsets:
        block = x[start:end]
        mean += np.mean(block, axis=0, dtype=np.float64)
        second += np.einsum("ij,ij->j", block, block, dtype=np.float64) / max(end - start, 1)
    mean /= len(offsets)
    second /= len(offsets)
    scale = np.sqrt(np.maximum(second - mean * mean, 0.0))
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    return mean, scale


def standardize_in_place(x, mean, scale, block_rows=16384):
    for start in range(0, x.shape[0], block_rows):
        end = min(start + block_rows, x.shape[0])
        x[start:end] -= mean
        x[start:end] /= scale


def bootstrap_assignments(test_offsets, layer_from, layer_to):
    proteins = [record["protein_id"] for record, _start, _end in test_offsets]
    rng = np.random.default_rng(RANDOM_SEED + 1009 * layer_from + 9176 * layer_to)
    counts, rows = [], []
    for replicate in range(BOOTSTRAP_REPLICATES):
        draw = rng.integers(0, len(proteins), size=len(proteins))
        multiplicity = np.bincount(draw, minlength=len(proteins)).astype(np.int64)
        counts.append(multiplicity)
        for index, protein_id in enumerate(proteins):
            rows.append({
                "layer_from": layer_from, "layer_to": layer_to,
                "bootstrap_replicate": replicate, "protein_id": protein_id,
                "draw_multiplicity": int(multiplicity[index]),
            })
    return np.stack(counts), rows

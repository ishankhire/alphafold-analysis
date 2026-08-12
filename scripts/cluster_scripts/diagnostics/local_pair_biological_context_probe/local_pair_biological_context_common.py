#!/usr/bin/env python3
"""Shared definitions for the local biological-context update probe.

This module is intentionally self-contained so its directory can be copied to
Delta without the rest of the repository.
"""

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
TRANSITIONS = ((10, 11), (15, 16), (20, 21), (30, 31), (40, 41))
BUNDLES = (
    "controls",
    "endpoint_biology",
    "local_biology",
    "local_triangle",
    "biology_local_triangle",
    "biology_local_triangle_mediator",
    "biology_local_triangle_plus_center",
)
BUNDLE_WIDTHS = {
    "controls": 5,
    "endpoint_biology": 47,
    "local_biology": 73,
    "local_triangle": 12,
    "biology_local_triangle": 79,
    "biology_local_triangle_mediator": 119,
    "biology_local_triangle_plus_center": 207,
}
OUTPUT_NAME = "local_pair_biological_context_probe"
SOURCE_VARIANT = "general_base256_combined"
SPLIT_ID = 0
MIN_SEQUENCE_SEPARATION = 4
MAX_SEQUENCE_SEPARATION = 12
LOCAL_RADIUS = 5
CLASS_THRESHOLD = 0.5
RANDOM_SEED = 20260812
BOOTSTRAP_REPLICATES = 200
LOGISTIC_C_GRID = (0.001, 0.01, 0.1, 1.0)
HIST_GB_MAX_ITER = 100

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
    "HID": "HIS", "HIE": "HIS", "HIP": "HIS", "HSD": "HIS", "HSE": "HIS",
}

# Kyte & Doolittle (1982), J Mol Biol 157:105-132.
HYDROPATHY = dict(zip(STANDARD_AAS, (
    1.8, -4.5, -3.5, -3.5, 2.5, -3.5, -3.5, -0.4, -3.2, 4.5,
    3.8, -3.9, 1.9, 2.8, -1.6, -0.8, -0.7, -0.9, -1.3, 4.2,
)))
# Grantham (1974), Science 185:862-864, Table 1.
GRANTHAM_VOLUME = dict(zip(STANDARD_AAS, (
    31.0, 124.0, 56.0, 54.0, 55.0, 83.0, 85.0, 3.0, 96.0, 111.0,
    111.0, 119.0, 105.0, 132.0, 32.5, 32.0, 61.0, 170.0, 136.0, 84.0,
)))
GRANTHAM_POLARITY = dict(zip(STANDARD_AAS, (
    8.1, 10.5, 11.6, 13.0, 5.5, 12.3, 10.5, 9.0, 10.4, 5.2,
    4.9, 11.3, 5.7, 5.2, 8.0, 9.2, 8.6, 5.4, 6.2, 5.9,
)))
NOMINAL_CHARGE = {aa: 0.0 for aa in STANDARD_AAS}
NOMINAL_CHARGE.update({"ASP": -1.0, "GLU": -1.0, "LYS": 1.0, "ARG": 1.0})

HYDROPHOBIC = frozenset(("ALA", "VAL", "ILE", "LEU", "MET", "PHE", "TRP", "TYR"))
AROMATIC = frozenset(("PHE", "TRP", "TYR"))
POLAR = frozenset(("CYS", "ASP", "GLU", "HIS", "LYS", "ASN", "GLN", "ARG", "SER", "THR", "TYR"))
CHARGED = frozenset(("ASP", "GLU", "LYS", "ARG"))
POSITIVE = frozenset(("LYS", "ARG"))
NEGATIVE = frozenset(("ASP", "GLU"))
SMALL = frozenset(("ALA", "CYS", "ASP", "GLY", "ASN", "PRO", "SER", "THR"))
BULKY = frozenset(("PHE", "LYS", "ARG", "TRP", "TYR"))

FAMILY_ORDER = (
    "controls", "endpoint_aa", "endpoint_properties", "pair_chemistry",
    "local_biology", "triangle_pressure", "mediator_identity", "center_representation",
)
TRIANGLE_FEATURE_NAMES = (
    "shrink_pressure_max", "shrink_pressure_mean", "shrink_violation_fraction",
    "grow_pressure_max", "grow_pressure_mean", "grow_violation_fraction",
)


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


def task_spec(task_id):
    task_id = int(task_id)
    total = len(TRANSITIONS) * len(BUNDLES)
    if task_id < 0 or task_id >= total:
        raise ValueError(f"Task ID {task_id} is outside 0..{total - 1}")
    transition_index, bundle_index = divmod(task_id, len(BUNDLES))
    return (*TRANSITIONS[transition_index], BUNDLES[bundle_index])


def source_for_protein(protein_id):
    return "cameo" if "_" in str(protein_id) else "ecod"


def pair_block_path(protein_id, layer):
    source = source_for_protein(protein_id)
    if source == "cameo":
        root = Path(os.environ.get("CAMEO_PAIR_BLOCKS_ROOT", CAMEO_PAIR_BLOCKS_ROOT))
        return root / protein_id / f"{protein_id}_pair_block_{layer}.npy"
    root = Path(os.environ.get("ECOD_PAIR_BLOCKS_ROOT", ECOD_PAIR_BLOCKS_ROOT))
    return root / f"{protein_id}_save_layers_0_47" / f"{protein_id}_pair_block_{layer}.npy"


def pdb_path(protein_id):
    source = source_for_protein(protein_id)
    if source == "cameo":
        root = Path(os.environ.get("CAMEO_PDB_ROOT", CAMEO_PDB_ROOT))
    else:
        root = Path(os.environ.get("ECOD_PDB_ROOT", ECOD_PDB_ROOT))
    return (
        root / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions" / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    )


def normalize_aa(name):
    name = str(name).upper()
    name = AA_ALIASES.get(name, name)
    return name if name in AA_TO_INDEX else None


def extract_pdb_residue_names(path):
    """Read ordered CA-bearing residues without depending on repository helpers."""
    residues = []
    seen = set()
    with Path(path).open() as handle:
        for line in handle:
            if not (line.startswith("ATOM  ") or line.startswith("HETATM")):
                continue
            if line[12:16].strip() != "CA":
                continue
            altloc = line[16:17]
            if altloc not in {" ", "A"}:
                continue
            residue_key = (line[21:22], line[22:26], line[26:27])
            if residue_key in seen:
                continue
            seen.add(residue_key)
            residues.append(normalize_aa(line[17:20].strip()))
    return residues


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
        return (np.empty(0, np.int64), np.empty(0, np.int64), np.empty(0, np.int16))
    return np.concatenate(i_parts), np.concatenate(j_parts), np.concatenate(separation_parts)


def local_pair_count(n_residues):
    return sum(max(int(n_residues) - separation, 0) for separation in range(4, 13))


def local_context_indices(i, j, n_residues, radius=LOCAL_RADIUS):
    values = set(range(max(0, i - radius), min(n_residues, i + radius + 1)))
    values.update(range(max(0, j - radius), min(n_residues, j + radius + 1)))
    values.discard(int(i))
    values.discard(int(j))
    return np.asarray(sorted(values), dtype=np.int64)


def load_split_records(path):
    records, source_rows = [], []
    with Path(path).open(newline="") as handle:
        for row in csv.DictReader(handle):
            if int(row["split_id"]) != SPLIT_ID:
                continue
            source_rows.append(dict(row))
            records.append({
                "protein_id": row["protein_id"],
                "source": source_for_protein(row["protein_id"]),
                "split": row["split"],
                "n_residues": int(row["n_residues"]),
                "size_bin": int(row.get("size_bin", 0)),
                "n_local_pairs": local_pair_count(int(row["n_residues"])),
            })
    if not records:
        raise ValueError(f"No split_id={SPLIT_ID} rows in {path}")
    if len({row["protein_id"] for row in records}) != len(records):
        raise ValueError("Duplicate protein IDs in split assignments")
    return records, source_rows


def logistic_validation_assignments(records):
    """Deterministic source/size-stratified 80/20 split of training proteins."""
    train = [row for row in records if row["split"] == "train"]
    groups = {}
    for row in train:
        groups.setdefault((row["source"], row["size_bin"]), []).append(row)
    assignment = {}
    rng = np.random.default_rng(RANDOM_SEED + 7919)
    for key in sorted(groups):
        group = sorted(groups[key], key=lambda row: row["protein_id"])
        order = rng.permutation(len(group))
        n_validation = max(1, int(round(0.2 * len(group)))) if len(group) > 1 else 0
        validation_indices = set(int(value) for value in order[:n_validation])
        for index, row in enumerate(group):
            assignment[row["protein_id"]] = "validation" if index in validation_indices else "fit"
    rows = [{
        "protein_id": row["protein_id"], "source": row["source"],
        "size_bin": row["size_bin"], "role": assignment[row["protein_id"]],
        "seed": RANDOM_SEED + 7919,
    } for row in sorted(train, key=lambda item: item["protein_id"])]
    return assignment, rows


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
                raise ValueError(f"Unexpected feature dimension in layer-{layer} coefficient row")
            source_rows.append(dict(row))
            index = int(row["feature_index"])
            value = float(row["coefficient_raw_scale"])
            if index == -1:
                if intercept is not None or row["feature_name"] != "intercept":
                    raise ValueError(f"Invalid or duplicate layer-{layer} intercept")
                intercept = value
            else:
                if index in seen or index < 0 or index >= DISTANCE_FEATURE_DIM:
                    raise ValueError(f"Invalid or duplicate layer-{layer} coefficient index {index}")
                expected_name = (
                    f"pair_ij_ch_{index}" if index < N_CHANNELS
                    else f"pair_ji_ch_{index - N_CHANNELS}"
                )
                if row["feature_name"] != expected_name:
                    raise ValueError(
                        f"Layer-{layer} coefficient {index} is {row['feature_name']!r}; expected {expected_name!r}"
                    )
                coefficients[index] = value
                seen.add(index)
    if intercept is None or not np.all(np.isfinite(coefficients)) or len(source_rows) != 257:
        raise ValueError(f"Incomplete layer-{layer} coefficient bundle in {path}")
    digest_source = "\n".join(
        ",".join(f"{key}={row[key]}" for key in sorted(row))
        for row in sorted(source_rows, key=lambda item: int(item["feature_index"]))
    )
    digest = hashlib.sha256(digest_source.encode()).hexdigest()
    for row in source_rows:
        row["coefficient_bundle_sha256"] = digest
    return float(intercept), coefficients, source_rows, digest


def decode_distance_matrix(pb, intercept, coefficients, chunk_size=65536):
    n_residues = int(pb.shape[0])
    i, j = np.triu_indices(n_residues, k=1)
    result = np.zeros((n_residues, n_residues), dtype=np.float64)
    for start in range(0, i.size, chunk_size):
        end = min(start + chunk_size, i.size)
        ii, jj = i[start:end], j[start:end]
        values = np.full(end - start, intercept, dtype=np.float64)
        values += np.asarray(pb[jj, ii, :], dtype=np.float64) @ coefficients[:N_CHANNELS]
        values += np.asarray(pb[ii, jj, :], dtype=np.float64) @ coefficients[N_CHANNELS:]
        result[ii, jj] = values
        result[jj, ii] = values
    return result


def decode_selected_distances(pb, i, j, intercept, coefficients):
    values = np.full(i.size, intercept, dtype=np.float64)
    values += np.asarray(pb[j, i, :], dtype=np.float64) @ coefficients[:N_CHANNELS]
    values += np.asarray(pb[i, j, :], dtype=np.float64) @ coefficients[N_CHANNELS:]
    return values


def feature_specs():
    specs = []
    def add(name, family, units="", definition=""):
        specs.append({"feature_name": name, "feature_family": family, "units": units, "definition": definition})
    for name, units in (
        ("current_distance", "angstrom"), ("sequence_separation", "residues"),
        ("protein_length", "residues"), ("normalized_pair_midpoint", "fraction"),
        ("normalized_nearest_terminus_distance", "fraction"),
    ):
        add(name, "controls", units)
    for aa in STANDARD_AAS:
        add(f"endpoint_count_{aa}", "endpoint_aa", "count")
    for prop in ("hydropathy", "volume", "polarity", "charge"):
        add(f"endpoint_{prop}_sum", "endpoint_properties")
        add(f"endpoint_{prop}_absolute_difference", "endpoint_properties")
    for group in ("hydrophobic", "aromatic", "small", "polar"):
        add(f"endpoint_{group}_count", "endpoint_properties", "count")
    for name in (
        "same_amino_acid", "cys_cys", "gly_pro", "both_hydrophobic", "both_aromatic",
        "opposite_charge", "both_positive", "both_negative", "both_small", "both_bulky",
    ):
        add(name, "pair_chemistry", "indicator")
    for aa in STANDARD_AAS:
        add(f"local_count_{aa}", "local_biology", "count")
    for group in ("hydrophobic", "aromatic", "charged", "small"):
        add(f"local_{group}_count", "local_biology", "count")
    add("local_third_residue_count", "local_biology", "count")
    add("local_between_residue_count", "local_biology", "count")
    for name in TRIANGLE_FEATURE_NAMES:
        add(name, "triangle_pressure", "angstrom" if "fraction" not in name else "fraction")
    for direction in ("shrink", "grow"):
        for aa in STANDARD_AAS:
            add(f"strongest_{direction}_mediator_{aa}", "mediator_identity", "indicator")
    for channel in range(N_CHANNELS):
        add(f"center_avg_ch_{channel}", "center_representation")
    return specs


ALL_FEATURE_SPECS = feature_specs()
FAMILY_INDICES = {
    family: np.asarray([i for i, row in enumerate(ALL_FEATURE_SPECS) if row["feature_family"] == family], dtype=np.int64)
    for family in FAMILY_ORDER
}
BUNDLE_FAMILIES = {
    "controls": ("controls",),
    "endpoint_biology": ("controls", "endpoint_aa", "endpoint_properties", "pair_chemistry"),
    "local_biology": ("controls", "endpoint_aa", "endpoint_properties", "pair_chemistry", "local_biology"),
    "local_triangle": ("controls", "local_count_only", "triangle_pressure"),
    "biology_local_triangle": ("controls", "endpoint_aa", "endpoint_properties", "pair_chemistry", "local_biology", "triangle_pressure"),
    "biology_local_triangle_mediator": ("controls", "endpoint_aa", "endpoint_properties", "pair_chemistry", "local_biology", "triangle_pressure", "mediator_identity"),
    "biology_local_triangle_plus_center": ("controls", "endpoint_aa", "endpoint_properties", "pair_chemistry", "local_biology", "triangle_pressure", "center_representation"),
}


def bundle_indices(bundle):
    if bundle not in BUNDLES:
        raise ValueError(f"Unknown bundle {bundle}")
    indices = []
    for family in BUNDLE_FAMILIES[bundle]:
        if family == "local_count_only":
            name_index = next(i for i, row in enumerate(ALL_FEATURE_SPECS) if row["feature_name"] == "local_third_residue_count")
            indices.append(name_index)
        else:
            indices.extend(FAMILY_INDICES[family].tolist())
    result = np.asarray(indices, dtype=np.int64)
    if result.size != BUNDLE_WIDTHS[bundle]:
        raise AssertionError(f"{bundle} schema has {result.size}, expected {BUNDLE_WIDTHS[bundle]}")
    return result


def feature_schema(bundle):
    rows = []
    for local_index, global_index in enumerate(bundle_indices(bundle)):
        spec = dict(ALL_FEATURE_SPECS[int(global_index)])
        spec.update({
            "bundle": bundle, "feature_dim": BUNDLE_WIDTHS[bundle],
            "feature_index": local_index, "global_feature_index": int(global_index),
            "source_layer": "layer_from", "transformation": "raw_then_training_standardized_for_linear_models",
        })
        rows.append(spec)
    return rows


def _property_value(table, aa):
    return table.get(aa, 0.0) if aa is not None else 0.0


def _count_members(names, members):
    return sum(name in members for name in names if name is not None)


def build_all_feature_families(pb_from, raw_distance_matrix, pair_i, pair_j, separation, residue_names):
    n_pairs = pair_i.size
    n_residues = len(residue_names)
    controls = np.empty((n_pairs, 5), dtype=np.float64)
    controls[:, 0] = raw_distance_matrix[pair_i, pair_j]
    controls[:, 1] = separation
    controls[:, 2] = n_residues
    denominator = max(n_residues - 1, 1)
    controls[:, 3] = (pair_i + pair_j) / (2.0 * denominator)
    controls[:, 4] = np.minimum(pair_i, n_residues - 1 - pair_j) / denominator

    endpoint_aa = np.zeros((n_pairs, 20), dtype=np.float64)
    endpoint_properties = np.zeros((n_pairs, 12), dtype=np.float64)
    pair_chemistry = np.zeros((n_pairs, 10), dtype=np.float64)
    local_biology = np.zeros((n_pairs, 26), dtype=np.float64)
    triangle = np.zeros((n_pairs, 6), dtype=np.float64)
    mediator = np.zeros((n_pairs, 40), dtype=np.float64)
    clipped = np.maximum(raw_distance_matrix, 0.0)

    for row_index, (i, j) in enumerate(zip(pair_i.tolist(), pair_j.tolist())):
        aa_i, aa_j = residue_names[i], residue_names[j]
        endpoints = (aa_i, aa_j)
        for aa in endpoints:
            if aa is not None:
                endpoint_aa[row_index, AA_TO_INDEX[aa]] += 1.0
        property_tables = (HYDROPATHY, GRANTHAM_VOLUME, GRANTHAM_POLARITY, NOMINAL_CHARGE)
        for prop_index, table in enumerate(property_tables):
            first, second = _property_value(table, aa_i), _property_value(table, aa_j)
            endpoint_properties[row_index, 2 * prop_index] = first + second
            endpoint_properties[row_index, 2 * prop_index + 1] = abs(first - second)
        for group_index, members in enumerate((HYDROPHOBIC, AROMATIC, SMALL, POLAR)):
            endpoint_properties[row_index, 8 + group_index] = _count_members(endpoints, members)
        charge_i, charge_j = _property_value(NOMINAL_CHARGE, aa_i), _property_value(NOMINAL_CHARGE, aa_j)
        pair_chemistry[row_index] = (
            aa_i is not None and aa_i == aa_j,
            aa_i == "CYS" and aa_j == "CYS",
            set(endpoints) == {"GLY", "PRO"},
            _count_members(endpoints, HYDROPHOBIC) == 2,
            _count_members(endpoints, AROMATIC) == 2,
            charge_i * charge_j < 0,
            aa_i in POSITIVE and aa_j in POSITIVE,
            aa_i in NEGATIVE and aa_j in NEGATIVE,
            _count_members(endpoints, SMALL) == 2,
            _count_members(endpoints, BULKY) == 2,
        )

        k_values = local_context_indices(i, j, n_residues)
        local_names = [residue_names[int(k)] for k in k_values]
        for aa in local_names:
            if aa is not None:
                local_biology[row_index, AA_TO_INDEX[aa]] += 1.0
        for group_index, members in enumerate((HYDROPHOBIC, AROMATIC, CHARGED, SMALL)):
            local_biology[row_index, 20 + group_index] = _count_members(local_names, members)
        local_biology[row_index, 24] = k_values.size
        local_biology[row_index, 25] = np.count_nonzero((k_values > i) & (k_values < j))

        if k_values.size:
            d_ij = clipped[i, j]
            shrink = np.maximum(d_ij - clipped[i, k_values] - clipped[k_values, j], 0.0)
            grow = np.maximum(np.abs(clipped[i, k_values] - clipped[k_values, j]) - d_ij, 0.0)
            triangle[row_index] = (
                np.max(shrink), np.mean(shrink), np.mean(shrink > 0.0),
                np.max(grow), np.mean(grow), np.mean(grow > 0.0),
            )
            if triangle[row_index, 0] > 0.0:
                k = int(k_values[int(np.argmax(shrink))])
                aa = residue_names[k]
                if aa is not None:
                    mediator[row_index, AA_TO_INDEX[aa]] = 1.0
            if triangle[row_index, 3] > 0.0:
                k = int(k_values[int(np.argmax(grow))])
                aa = residue_names[k]
                if aa is not None:
                    mediator[row_index, 20 + AA_TO_INDEX[aa]] = 1.0

    center = 0.5 * (
        np.asarray(pb_from[pair_i, pair_j, :], dtype=np.float64)
        + np.asarray(pb_from[pair_j, pair_i, :], dtype=np.float64)
    )
    return {
        "controls": controls,
        "endpoint_aa": endpoint_aa,
        "endpoint_properties": endpoint_properties,
        "pair_chemistry": pair_chemistry,
        "local_biology": local_biology,
        "triangle_pressure": triangle,
        "mediator_identity": mediator,
        "center_representation": center,
    }


def assemble_bundle(families, bundle):
    blocks = []
    for family in BUNDLE_FAMILIES[bundle]:
        if family == "local_count_only":
            blocks.append(families["local_biology"][:, 24:25])
        else:
            blocks.append(families[family])
    result = np.concatenate(blocks, axis=1).astype(np.float64, copy=False)
    if result.shape[1] != BUNDLE_WIDTHS[bundle]:
        raise AssertionError(f"Built {result.shape[1]} columns for {bundle}")
    return result


def build_offsets(records, split):
    offsets, position = [], 0
    for record in records:
        if record["split"] != split:
            continue
        end = position + record["n_local_pairs"]
        offsets.append((record, position, end))
        position = end
    return offsets, position


def subset_offsets(offsets, roles, desired_role):
    selected_rows, selected_offsets, position = [], [], 0
    for record, start, end in offsets:
        if roles[record["protein_id"]] != desired_role:
            continue
        indices = np.arange(start, end, dtype=np.int64)
        selected_rows.append(indices)
        selected_offsets.append((record, position, position + indices.size))
        position += indices.size
    return (np.concatenate(selected_rows) if selected_rows else np.empty(0, np.int64)), selected_offsets


def protein_equal_weights(offsets):
    total = offsets[-1][2] if offsets else 0
    weights = np.empty(total, dtype=np.float64)
    for _record, start, end in offsets:
        weights[start:end] = total / (len(offsets) * max(end - start, 1))
    return weights


def balanced_binary_weights(base_weights, labels):
    weights = np.asarray(base_weights, dtype=np.float64).copy()
    for value in (0, 1):
        mask = labels == value
        mass = float(np.sum(weights[mask]))
        if mass <= 0:
            raise ValueError(f"No classification examples for class {value}")
        weights[mask] *= np.sum(weights) / (2.0 * mass)
    weights *= weights.size / np.sum(weights)
    return weights


def protein_equal_scaler(x, offsets):
    means, second = [], []
    for _record, start, end in offsets:
        block = x[start:end]
        means.append(np.mean(block, axis=0, dtype=np.float64))
        second.append(np.mean(block * block, axis=0, dtype=np.float64))
    mean = np.mean(np.stack(means), axis=0)
    variance = np.maximum(np.mean(np.stack(second), axis=0) - mean * mean, 0.0)
    scale = np.sqrt(variance)
    scale[scale <= np.finfo(np.float64).eps] = 1.0
    return mean, scale


def standardize(x, mean, scale):
    return (np.asarray(x, dtype=np.float64) - mean) / scale


def class_labels(delta):
    delta = np.asarray(delta, dtype=np.float64)
    labels = np.full(delta.shape, -1, dtype=np.int8)
    labels[delta < -CLASS_THRESHOLD] = 1
    labels[delta > CLASS_THRESHOLD] = 0
    return labels


def safe_pearson(x, y):
    x, y = np.asarray(x, np.float64), np.asarray(y, np.float64)
    if x.size < 2 or np.std(x) <= 0 or np.std(y) <= 0:
        return float("nan")
    return float(np.corrcoef(x, y)[0, 1])


def bootstrap_assignments(test_offsets, layer_from, layer_to):
    proteins = [record["protein_id"] for record, _start, _end in test_offsets]
    rng = np.random.default_rng(RANDOM_SEED + 1009 * layer_from + 9176 * layer_to)
    rows, counts = [], []
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


def correlation_label(value):
    if not math.isfinite(float(value)):
        return "undefined"
    absolute = abs(float(value))
    if absolute < 0.3:
        return "weak"
    if absolute < 0.7:
        return "moderate"
    return "high"


def triangle_distance_correlations(layer_from, layer_to, split, current_distance, triangle, offsets):
    rows = []
    for index, name in enumerate(TRIANGLE_FEATURE_NAMES):
        per_protein = []
        for _record, start, end in offsets:
            value = safe_pearson(current_distance[start:end], triangle[start:end, index])
            if math.isfinite(value):
                per_protein.append(value)
        pooled = safe_pearson(current_distance, triangle[:, index])
        rows.append({
            "layer_from": layer_from, "layer_to": layer_to, "split": split,
            "feature_name": name, "pair_pearson_r": pooled,
            "mean_per_protein_pearson_r": np.mean(per_protein) if per_protein else float("nan"),
            "n_valid_proteins": len(per_protein), "correlation_label": correlation_label(pooled),
        })
    return rows


def metadata_constants():
    return {
        "standard_aas": list(STANDARD_AAS), "aa_aliases": AA_ALIASES,
        "hydropathy_kyte_doolittle": HYDROPATHY,
        "grantham_volume": GRANTHAM_VOLUME, "grantham_polarity": GRANTHAM_POLARITY,
        "nominal_charge": NOMINAL_CHARGE,
        "chemical_sets": {
            "hydrophobic": sorted(HYDROPHOBIC), "aromatic": sorted(AROMATIC),
            "polar": sorted(POLAR), "charged": sorted(CHARGED),
            "positive": sorted(POSITIVE), "negative": sorted(NEGATIVE),
            "small": sorted(SMALL), "bulky": sorted(BULKY),
        },
    }

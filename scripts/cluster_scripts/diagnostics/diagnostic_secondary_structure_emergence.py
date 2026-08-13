#!/usr/bin/env python3
"""Measure when final-PDB secondary structures become visible in decoded distances.

This diagnostic does not fit a new probe and does not discover secondary
structure.  A preparation stage runs DSSP on the final OpenFold PDBs, validates
residue indexing, freezes empirical helix centroids and visibility thresholds
from training proteins, and chooses deterministic beta-sheet controls.  Six
independent layer tasks then replay the saved ``general_base256_combined`` raw
coefficients on held-out proteins.  A final stage consolidates the layer shards,
adds trajectory stability fields, and writes plots.

Set ``SUBSTRUCTURE_STAGE`` to ``prepare``, ``layer``, or ``consolidate``.  The
companion submit script wires these stages together with Slurm dependencies.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
import tempfile
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import make_dssp_dict
from sklearn.metrics import roc_auc_score


SCRIPT_DIR = Path(__file__).resolve().parent


def find_project_root(script_dir: Path) -> Path:
    """Find a repository checkout, with a safe flat-copy fallback on Delta."""
    for candidate in (script_dir, *script_dir.parents):
        if (candidate / "datasets.md").is_file() and (
            candidate / "scripts" / "cluster_scripts"
        ).is_dir():
            return candidate
    return script_dir


PROJECT_ROOT = find_project_root(SCRIPT_DIR)
ANALYSIS_NAME = "diagnostic_secondary_structure_emergence"
ANALYSIS_LAYERS = (0, 10, 20, 30, 40, 47)
VARIANT = "general_base256_combined"
FEATURE_DIM = 256
N_CHANNELS = 128
SPLIT_ID = 0
RANDOM_SEED = 20260813
N_MATCHED_RANDOM = 20
MIN_TRUE_HELIX_AUROC = 0.98
REPLAY_R2_TOLERANCE = 1e-4
HELIX_MIN_SEGMENT_LENGTH = 6
ADJACENT_CA_MIN = 3.6
ADJACENT_CA_MAX = 4.0
FLOAT_EPS = 1e-12
STANDARD_AAS = tuple(
    "ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split()
)
AA_ALPHABETICAL = tuple(sorted(STANDARD_AAS))
AA_PAIR_LABELS = tuple(
    f"{left}-{right}"
    for left_index, left in enumerate(AA_ALPHABETICAL)
    for right in AA_ALPHABETICAL[left_index:]
)
AA_TO_CODE = {name: index for index, name in enumerate(AA_ALPHABETICAL)}
AA_PAIR_CODE = np.full((len(AA_ALPHABETICAL), len(AA_ALPHABETICAL)), -1, dtype=np.int16)
_aa_pair_to_code = {name: index for index, name in enumerate(AA_PAIR_LABELS)}
for _left_index, _left in enumerate(AA_ALPHABETICAL):
    for _right_index, _right in enumerate(AA_ALPHABETICAL):
        AA_PAIR_CODE[_left_index, _right_index] = _aa_pair_to_code[
            f"{min(_left, _right)}-{max(_left, _right)}"
        ]
DSSP_CODES = ("H", "B", "E", "G", "I", "T", "S", "-")
DSSP_PDB_RECORDS = frozenset(
    ("CRYST1", "MODEL ", "ATOM  ", "HETATM", "ANISOU", "TER   ", "ENDMDL", "END   ")
)
DISTANCE_BINS = (
    (0.0, 5.0, "0-5"),
    (5.0, 10.0, "5-10"),
    (10.0, 15.0, "10-15"),
    (15.0, 20.0, "15-20"),
    (20.0, 30.0, "20-30"),
    (30.0, math.inf, "30+"),
)
SEPARATION_BINS = (
    (1, 1, "1"),
    (2, 4, "2-4"),
    (5, 9, "5-9"),
    (10, 19, "10-19"),
    (20, 49, "20-49"),
    (50, sys.maxsize, "50+"),
)

LOCAL_RESULTS_DIR = PROJECT_ROOT / "delta_files" / "general_base256_combined"
LOCAL_DATA_ROOT = (
    PROJECT_ROOT.parent / "protein_data"
    if PROJECT_ROOT != SCRIPT_DIR
    else SCRIPT_DIR / "protein_data"
)
LOCAL_OUTPUT_PARENT = (
    PROJECT_ROOT / "proteins" / "multi_protein_results" / "diagnostics"
)
CAMEO_PAIR_ROOT = Path("/work/hdd/bekh/thayes/openfold_reps")
CAMEO_PDB_ROOT = Path("/work/hdd/bekh/thayes/openfold_reps_outputs_sm")
ECOD_PAIR_ROOT = Path("/work/hdd/bfna/thayes/ecod/experiments/representation_saves")
ECOD_PDB_ROOT = Path(
    "/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm"
)

DSSP_VALIDATION_FIELDS = [
    "protein_id", "source", "split", "status", "usable_for_structure",
    "dssp_exit_status", "n_pair_residues", "n_pdb_ca_residues",
    "n_dssp_records", "n_dssp_mapped", *[f"n_dssp_{code if code != '-' else 'coil'}" for code in DSSP_CODES],
    "helix_fraction", "strand_fraction", "coil_fraction", "n_helix_segments",
    "n_strand_segments", "helix_segment_lengths", "strand_segment_lengths",
    "n_bridge_pairs", "n_main_bridge_pairs", "n_parallel_bridge_pairs", "n_antiparallel_bridge_pairs",
    "n_unknown_orientation_bridge_pairs", "n_array_adjacencies",
    "n_chain_transitions", "n_numbering_gaps", "n_adjacent_ca_deviations",
    "mean_adjacent_ca_distance", "max_adjacent_ca_distance", "suspect_reasons",
    "pdb_path", "pair_block_0_path", "runtime_seconds",
]
DSSP_FAILURE_FIELDS = [
    "protein_id", "source", "split", "status", "suspect_reasons", "details"
]
SKIPPED_FIELDS = ["protein_id", "source", "split", "stage", "reason"]
VALIDATION_FIELDS = [
    "category", "metric", "split", "orientation", "value", "n", "threshold",
    "status", "details",
]
PER_PROTEIN_FIELDS = [
    "variant", "feature_dim", "split_id", "layer", "split", "protein_id",
    "source", "n_residues", "n_pairs", "per_protein_r2", "per_protein_mae",
    "per_protein_rmse", "mean_error_pred_minus_true", "saved_per_protein_r2",
    "replay_r2_difference", "n_negative_predictions", "negative_prediction_fraction",
    "n_helix_residues", "n_strand_residues", "mean_helix_score",
    "mean_strand_score", "helix_strand_auroc", "mean_helix_centroid_distance",
    "n_helix_segments", "n_bridge_pairs", "n_parallel_bridge_pairs",
    "n_antiparallel_bridge_pairs",
]
HELIX_RESIDUE_FIELDS = [
    "variant", "layer", "split", "protein_id", "source", "residue_index",
    "chain", "pdb_resseq", "insertion_code", "resname", "dssp_code",
    "segment_id", "segment_length", "offset_2_decoded", "offset_3_decoded",
    "offset_4_decoded", "offset_2_true", "offset_3_true", "offset_4_true",
    "score", "distance_to_helix_centroid", "distance_to_helix_centroid_z",
    "distance_to_strand_centroid_z", "raw_triple_rmse", "passes_relative_gate",
    "passes_absolute_gate", "looks_helical",
]
HELIX_SEGMENT_FIELDS = [
    "variant", "layer", "split", "protein_id", "source", "segment_id",
    "start_residue_index", "end_residue_index", "original_segment_length",
    "n_scored_residues", "mean_score", "mean_distance_to_helix_centroid",
    "mean_raw_triple_rmse", "fraction_residues_looking_helical", "visible",
    "first_visible_layer", "score_monotone_nondecreasing",
    "closeness_monotone_nonincreasing", "crossed_then_reverted",
]
BRIDGE_FIELDS = [
    "variant", "layer", "split", "protein_id", "source", "bridge_id", "i", "j",
    "orientation", "sequence_separation", "decoded_partner_distance",
    "true_partner_distance", "n_matched_random", "mean_decoded_random_distance",
    "mean_true_random_distance", "matched_random_margin", "n_shift_1",
    "mean_decoded_shift_1_distance", "mean_true_shift_1_distance",
    "shift_1_margin", "n_shift_2", "mean_decoded_shift_2_distance",
    "mean_true_shift_2_distance", "shift_2_margin", "visible",
    "first_visible_layer", "partner_distance_monotone_nonincreasing",
    "crossed_then_reverted",
]
LAYER_SUMMARY_FIELDS = [
    "variant", "feature_dim", "split_id", "layer", "split", "n_proteins",
    "n_pairs", "n_negative_predictions", "negative_prediction_fraction",
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
ACCURACY_FIELDS = [
    "variant", "layer", "split", "group", "distance_bin", "n_pairs", "mae",
    "rmse", "bias", "r2", "sparse", "metric_scope",
]
MANIFEST_FIELDS = [
    "analysis_name", "variant", "feature_dim", "split_id", "layers",
    "analysis_splits", "random_seed", "matched_random_per_bridge",
    "helix_min_segment_length", "helix_specificity", "minimum_true_helix_auroc",
    "replay_r2_tolerance", "input_results_dir", "coefficients_sha256",
    "split_assignments_sha256", "dssp_executable", "dssp_version",
    "generated_at",
]


def now_timestamp() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log(message: str) -> None:
    print(f"[{now_timestamp()}] {message}", flush=True)


def finite_mean(values) -> float:
    array = np.asarray(list(values), dtype=np.float64)
    array = array[np.isfinite(array)]
    return float(np.mean(array)) if array.size else float("nan")


def empirical_quantile(values, quantile: float, direction: str) -> float:
    """Return a deterministic order-statistic quantile without interpolation."""
    array = np.sort(np.asarray(values, dtype=np.float64))
    if array.size == 0:
        return float("nan")
    position = quantile * (array.size - 1)
    index = math.ceil(position) if direction == "higher" else math.floor(position)
    return float(array[int(index)])


def safe_auc(labels, scores) -> float:
    labels = np.asarray(labels, dtype=np.int8)
    scores = np.asarray(scores, dtype=np.float64)
    valid = np.isfinite(scores)
    labels, scores = labels[valid], scores[valid]
    if labels.size == 0 or np.unique(labels).size != 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_write_csv(path: Path, fieldnames, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def atomic_savez(path: Path, **arrays) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(handle, **arrays)
    temporary.replace(path)


def read_csv(path: Path, required_fields=None) -> list[dict]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required_fields or ()) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader)


def output_paths() -> tuple[Path, Path]:
    input_results = Path(
        os.environ.get("INPUT_RESULTS_DIR", str(LOCAL_RESULTS_DIR))
    ).expanduser().resolve()
    if os.environ.get("SUBSTRUCTURE_OUTPUT_DIR"):
        output_root = Path(os.environ["SUBSTRUCTURE_OUTPUT_DIR"]).expanduser().resolve()
    else:
        output_parent = Path(
            os.environ.get("CSV_ROOT", str(LOCAL_OUTPUT_PARENT))
        ).expanduser().resolve()
        output_root = output_parent / ANALYSIS_NAME
    return input_results, output_root


def is_ecod(protein_id: str) -> bool:
    return protein_id.isdigit()


def source_for(protein_id: str) -> str:
    return "ECOD" if is_ecod(protein_id) else "CAMEO"


def resolve_pair_block(protein_id: str, layer: int) -> Path:
    local = LOCAL_DATA_ROOT / protein_id / "pair_blocks" / f"{protein_id}_pair_block_{layer}.npy"
    if is_ecod(protein_id):
        root = Path(os.environ.get("ECOD_PAIR_BLOCKS_ROOT", str(ECOD_PAIR_ROOT))).expanduser()
        cluster = root / f"{protein_id}_save_layers_0_47" / f"{protein_id}_pair_block_{layer}.npy"
    else:
        root = Path(os.environ.get("CAMEO_PAIR_BLOCKS_ROOT", str(CAMEO_PAIR_ROOT))).expanduser()
        cluster = root / protein_id / f"{protein_id}_pair_block_{layer}.npy"
    for candidate in (local, cluster):
        if candidate.is_file():
            return candidate.resolve()
    return cluster


def resolve_pdb(protein_id: str) -> Path:
    filename = f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    local = LOCAL_DATA_ROOT / protein_id / "predicted_pdb" / filename
    root_name = "ECOD_PDB_ROOT" if is_ecod(protein_id) else "CAMEO_PDB_ROOT"
    default = ECOD_PDB_ROOT if is_ecod(protein_id) else CAMEO_PDB_ROOT
    root = Path(os.environ.get(root_name, str(default))).expanduser()
    candidates = (
        local,
        root / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47" / "predictions" / filename,
        root / protein_id / "predicted_pdb" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    return candidates[1]


def load_split_assignments(input_results: Path) -> list[dict]:
    rows = read_csv(
        input_results / "split_assignments.csv",
        {"split_id", "split", "protein_id", "n_residues"},
    )
    selected = [row for row in rows if int(row["split_id"]) == SPLIT_ID]
    if len(selected) != len({row["protein_id"] for row in selected}):
        raise ValueError("Frozen split contains duplicate protein IDs")
    for row in selected:
        row["source"] = source_for(row["protein_id"])
    return selected


def load_catalog_protein_ids() -> list[str]:
    """Read the two protein-ID code blocks in datasets.md when available."""
    path = PROJECT_ROOT / "datasets.md"
    if not path.is_file():
        return []
    ids, inside = [], False
    for line in path.read_text().splitlines():
        if line.strip() == "```text":
            inside = True
            continue
        if inside and line.strip() == "```":
            inside = False
            continue
        if inside:
            ids.extend(line.split())
    return ids


def load_saved_metrics(input_results: Path, layer: int) -> dict[str, dict]:
    rows = read_csv(
        input_results / "per_protein.csv",
        {"variant", "split_id", "layer", "split", "protein_id", "per_protein_r2"},
    )
    return {
        row["protein_id"]: row
        for row in rows
        if row["variant"] == VARIANT
        and int(row["split_id"]) == SPLIT_ID
        and int(row["layer"]) == layer
    }


def load_raw_coefficients(input_results: Path, layer: int) -> tuple[float, np.ndarray]:
    rows = read_csv(
        input_results / "coefficients.csv",
        {"variant", "feature_dim", "split_id", "layer", "feature_index", "coefficient_raw_scale"},
    )
    selected = [
        row for row in rows
        if row["variant"] == VARIANT
        and int(row["feature_dim"]) == FEATURE_DIM
        and int(row["split_id"]) == SPLIT_ID
        and int(row["layer"]) == layer
    ]
    if len(selected) != FEATURE_DIM + 1:
        raise ValueError(f"Layer {layer} has {len(selected)} coefficient rows, expected 257")
    intercept_rows = [row for row in selected if int(row["feature_index"]) == -1]
    feature_rows = sorted(
        (row for row in selected if int(row["feature_index"]) >= 0),
        key=lambda row: int(row["feature_index"]),
    )
    if len(intercept_rows) != 1 or [int(row["feature_index"]) for row in feature_rows] != list(range(FEATURE_DIM)):
        raise ValueError(f"Layer {layer} coefficient indices are incomplete")
    return (
        float(intercept_rows[0]["coefficient_raw_scale"]),
        np.asarray([float(row["coefficient_raw_scale"]) for row in feature_rows], dtype=np.float64),
    )


def parse_pdb_residues(path: Path) -> list[dict]:
    model = PDBParser(QUIET=True).get_structure("predicted", str(path))[0]
    rows = []
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] == "W" or "CA" not in residue:
                continue
            hetflag, resseq, insertion_code = residue.get_id()
            rows.append(
                {
                    "chain": chain.id,
                    "pdb_resseq": int(resseq),
                    "insertion_code": insertion_code.strip(),
                    "hetflag": hetflag,
                    "resname": residue.get_resname(),
                    "coordinate": np.asarray(residue["CA"].coord, dtype=np.float64),
                }
            )
    if not rows:
        raise ValueError(f"No CA residues found in {path}")
    return rows


def residue_key(row: dict) -> tuple:
    insertion = row["insertion_code"] or " "
    hetflag = " " if row["hetflag"] == " " else row["hetflag"]
    return row["chain"], (hetflag, int(row["pdb_resseq"]), insertion)


def prepare_dssp_input(source: Path) -> Path:
    filtered = [
        line for line in source.read_text().splitlines(keepends=True)
        if line[:6] in DSSP_PDB_RECORDS
    ]
    if not any(line.startswith(("ATOM  ", "HETATM")) for line in filtered):
        raise ValueError(f"No coordinate records remain after filtering {source}")
    descriptor, name = tempfile.mkstemp(
        prefix="substructure_dssp_", suffix=".pdb", dir=os.environ.get("TMPDIR")
    )
    with os.fdopen(descriptor, "w") as handle:
        if not any(line.startswith("CRYST1") for line in filtered):
            handle.write("CRYST1  999.000  999.000  999.000  90.00  90.00  90.00 P 1           1\n")
        handle.writelines(filtered)
    return Path(name)


def parse_dssp_bridge_rows(raw_text: str) -> list[dict]:
    """Parse fixed-width DSSP residue indices, partners, and bridge orientation.

    The two beta-bridge labels occupy columns immediately before BP1/BP2.
    Lower-case labels denote parallel bridges and upper-case labels denote
    antiparallel bridges.  This information is not retained by
    ``Bio.PDB.DSSP.make_dssp_dict``.
    """
    rows = []
    for line in raw_text.splitlines():
        if len(line) < 33 or not line[:5].strip().isdigit():
            continue
        try:
            dssp_index = int(line[:5])
            pdb_resseq = int(line[5:10])
        except ValueError:
            continue
        if len(line) > 13 and line[13] == "!":
            continue
        labels = (line[23] if len(line) > 23 else " ", line[24] if len(line) > 24 else " ")
        partner_fields = (line[25:29], line[29:33])
        partners = []
        for label, field in zip(labels, partner_fields):
            if not field.strip():
                continue
            try:
                partner = int(field)
            except ValueError:
                continue
            if partner <= 0:
                continue
            if label.islower():
                orientation = "parallel"
            elif label.isupper():
                orientation = "antiparallel"
            else:
                orientation = "unknown"
            partners.append((partner, orientation, label))
        rows.append(
            {
                "dssp_index": dssp_index,
                "chain": line[11].strip() or " ",
                "pdb_resseq": pdb_resseq,
                "insertion_code": line[10].strip(),
                "dssp_code": (line[16].strip() if len(line) > 16 else "") or "-",
                "partners": partners,
            }
        )
    return rows


def run_dssp(pdb_path: Path, executable: Path) -> tuple[dict, list[dict], int, str]:
    temporary = prepare_dssp_input(pdb_path)
    try:
        result = subprocess.run(
            [str(executable), "--output-format=dssp", str(temporary)],
            check=False,
            text=True,
            capture_output=True,
        )
    finally:
        temporary.unlink(missing_ok=True)
    if result.returncode != 0:
        raise RuntimeError(f"mkdssp exited {result.returncode}: {result.stderr.strip()}")
    descriptor, name = tempfile.mkstemp(prefix="substructure_raw_", suffix=".dssp")
    os.close(descriptor)
    raw_path = Path(name)
    try:
        raw_path.write_text(result.stdout)
        records, _ = make_dssp_dict(str(raw_path))
    finally:
        raw_path.unlink(missing_ok=True)
    return records, parse_dssp_bridge_rows(result.stdout), result.returncode, result.stderr.strip()


def build_segments(codes: np.ndarray, chains: np.ndarray, target_code: str) -> tuple[np.ndarray, list[dict]]:
    segment_ids = np.full(len(codes), -1, dtype=np.int32)
    segments = []
    start = None
    for index, code in enumerate(codes):
        continues = (
            start is not None
            and code == target_code
            and index > 0
            and chains[index] == chains[index - 1]
        )
        if code == target_code and start is None:
            start = index
        elif code == target_code and continues:
            pass
        elif code == target_code:
            end = index - 1
            segment_id = len(segments)
            segment_ids[start : end + 1] = segment_id
            segments.append({"segment_id": segment_id, "start": start, "end": end, "length": end - start + 1})
            start = index
        elif start is not None:
            end = index - 1
            segment_id = len(segments)
            segment_ids[start : end + 1] = segment_id
            segments.append({"segment_id": segment_id, "start": start, "end": end, "length": end - start + 1})
            start = None
    if start is not None:
        end = len(codes) - 1
        segment_id = len(segments)
        segment_ids[start : end + 1] = segment_id
        segments.append({"segment_id": segment_id, "start": start, "end": end, "length": end - start + 1})
    return segment_ids, segments


def deduplicate_bridges(raw_rows: list[dict], residues: list[dict], codes: np.ndarray) -> list[dict]:
    key_to_index = {
        (row["chain"] or " ", int(row["pdb_resseq"]), row["insertion_code"] or " "): index
        for index, row in enumerate(residues)
    }
    dssp_index_to_residue = {}
    for row in raw_rows:
        key = (row["chain"] or " ", int(row["pdb_resseq"]), row["insertion_code"] or " ")
        if key in key_to_index:
            dssp_index_to_residue[int(row["dssp_index"])] = key_to_index[key]
    collected = defaultdict(set)
    for row in raw_rows:
        left = dssp_index_to_residue.get(int(row["dssp_index"]))
        if left is None:
            continue
        for partner_index, orientation, _label in row["partners"]:
            right = dssp_index_to_residue.get(int(partner_index))
            if right is None or right == left:
                continue
            pair = tuple(sorted((left, right)))
            collected[pair].add(orientation)
    bridges = []
    for bridge_id, ((i, j), orientations) in enumerate(sorted(collected.items())):
        known = orientations & {"parallel", "antiparallel"}
        orientation = next(iter(known)) if len(known) == 1 else "unknown"
        main_eligible = bool(
            codes[i] == "E"
            and codes[j] == "E"
            and residues[i]["chain"] == residues[j]["chain"]
            and orientation in {"parallel", "antiparallel"}
        )
        bridges.append(
            {
                "bridge_id": bridge_id,
                "i": int(i),
                "j": int(j),
                "orientation": orientation,
                "main_eligible": int(main_eligible),
            }
        )
    return bridges


def stable_rng(*parts) -> np.random.Generator:
    payload = ":".join(str(part) for part in (RANDOM_SEED, *parts)).encode()
    seed = int.from_bytes(hashlib.sha256(payload).digest()[:8], "little")
    return np.random.default_rng(seed)


def choose_bridge_controls(
    protein_id: str,
    residues: list[dict],
    codes: np.ndarray,
    e_segment_ids: np.ndarray,
    bridges: list[dict],
) -> list[dict]:
    bridge_set = {tuple(sorted((row["i"], row["j"]))) for row in bridges}
    controls = []
    for bridge in bridges:
        if not bridge["main_eligible"]:
            continue
        i, j = bridge["i"], bridge["j"]
        separation = abs(i - j)
        candidates = []
        for left in range(len(residues) - separation):
            right = left + separation
            if residues[left]["chain"] != residues[right]["chain"]:
                continue
            if (left, right) in bridge_set:
                continue
            candidates.append((left, right))
        if candidates:
            order = stable_rng(protein_id, i, j, "matched_random").permutation(len(candidates))
            for candidate_index in order[:N_MATCHED_RANDOM]:
                left, right = candidates[int(candidate_index)]
                controls.append(
                    {
                        "bridge_id": bridge["bridge_id"], "type": "matched_random",
                        "i": left, "j": right, "offset": 0,
                    }
                )
        partner_segment = int(e_segment_ids[j])
        for magnitude in (1, 2):
            for signed_offset in (-magnitude, magnitude):
                shifted = j + signed_offset
                if shifted < 0 or shifted >= len(residues):
                    continue
                if int(e_segment_ids[shifted]) != partner_segment:
                    continue
                pair = tuple(sorted((i, shifted)))
                if pair in bridge_set or pair[0] == pair[1]:
                    continue
                controls.append(
                    {
                        "bridge_id": bridge["bridge_id"], "type": f"shift_{magnitude}",
                        "i": pair[0], "j": pair[1], "offset": signed_offset,
                    }
                )
    return controls


def distance_matrix(coords: np.ndarray) -> np.ndarray:
    delta = coords[:, None, :] - coords[None, :, :]
    return np.linalg.norm(delta, axis=2)


def valid_offset_triples(
    distances: np.ndarray,
    codes: np.ndarray,
    chains: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    indices, triples = [], []
    for index in range(len(codes)):
        if codes[index] not in {"H", "E"} or index + 4 >= len(codes):
            continue
        if any(chains[index + offset] != chains[index] for offset in (2, 3, 4)):
            continue
        indices.append(index)
        triples.append([distances[index, index + offset] for offset in (2, 3, 4)])
    return np.asarray(indices, dtype=np.int32), np.asarray(triples, dtype=np.float64).reshape((-1, 3))


def score_triples(triples: np.ndarray, calibration: dict) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    triples = np.asarray(triples, dtype=np.float64).reshape((-1, 3))
    mean = np.asarray(calibration["zscore_mean"], dtype=np.float64)
    scale = np.asarray(calibration["zscore_scale"], dtype=np.float64)
    helix = np.asarray(calibration["mu_helix_z"], dtype=np.float64)
    strand = np.asarray(calibration["mu_strand_z"], dtype=np.float64)
    z = (triples - mean) / scale
    distance_helix = np.linalg.norm(z - helix, axis=1)
    distance_strand = np.linalg.norm(z - strand, axis=1)
    return distance_strand - distance_helix, distance_helix, distance_strand


def metrics_from_arrays(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return {"n_pairs": 0, "mae": math.nan, "rmse": math.nan, "bias": math.nan, "r2": math.nan}
    error = y_pred - y_true
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2, dtype=np.float64))
    return {
        "n_pairs": int(y_true.size),
        "mae": float(np.mean(np.abs(error))),
        "rmse": float(np.sqrt(np.mean(error * error))),
        "bias": float(np.mean(error)),
        "r2": float(1.0 - np.sum(error * error, dtype=np.float64) / ss_tot) if ss_tot > FLOAT_EPS else math.nan,
    }


def calibration_cache_path(output_root: Path, protein_id: str) -> Path:
    return output_root / "annotation_cache" / f"{protein_id}.npz"


def encode_annotation_cache(
    path: Path,
    residues: list[dict],
    codes: np.ndarray,
    h_segment_ids: np.ndarray,
    e_segment_ids: np.ndarray,
    h_segments: list[dict],
    e_segments: list[dict],
    bridges: list[dict],
    controls: list[dict],
) -> None:
    def values(rows, key, dtype):
        return np.asarray([row[key] for row in rows], dtype=dtype)

    atomic_savez(
        path,
        coords=np.asarray([row["coordinate"] for row in residues], dtype=np.float64),
        chains=values(residues, "chain", "U4"),
        pdb_resseq=values(residues, "pdb_resseq", np.int32),
        insertion_codes=values(residues, "insertion_code", "U2"),
        resnames=values(residues, "resname", "U4"),
        dssp_codes=np.asarray(codes, dtype="U1"),
        h_segment_ids=np.asarray(h_segment_ids, dtype=np.int32),
        e_segment_ids=np.asarray(e_segment_ids, dtype=np.int32),
        h_segment_start=values(h_segments, "start", np.int32),
        h_segment_end=values(h_segments, "end", np.int32),
        h_segment_length=values(h_segments, "length", np.int32),
        e_segment_start=values(e_segments, "start", np.int32),
        e_segment_end=values(e_segments, "end", np.int32),
        e_segment_length=values(e_segments, "length", np.int32),
        bridge_id=values(bridges, "bridge_id", np.int32),
        bridge_i=values(bridges, "i", np.int32),
        bridge_j=values(bridges, "j", np.int32),
        bridge_orientation=values(bridges, "orientation", "U16"),
        bridge_main_eligible=values(bridges, "main_eligible", np.int8),
        control_bridge_id=values(controls, "bridge_id", np.int32),
        control_type=values(controls, "type", "U20"),
        control_i=values(controls, "i", np.int32),
        control_j=values(controls, "j", np.int32),
        control_offset=values(controls, "offset", np.int8),
    )


def load_annotation_cache(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {key: data[key] for key in data.files}


def dssp_version(executable: Path) -> str:
    result = subprocess.run([str(executable), "--version"], check=True, text=True, capture_output=True)
    return result.stdout.strip() or result.stderr.strip()


def write_manifest(input_results: Path, output_root: Path, executable: Path, version: str) -> dict:
    coefficients = input_results / "coefficients.csv"
    assignments = input_results / "split_assignments.csv"
    manifest = {
        "analysis_name": ANALYSIS_NAME,
        "variant": VARIANT,
        "feature_dim": FEATURE_DIM,
        "split_id": SPLIT_ID,
        "layers": ",".join(map(str, ANALYSIS_LAYERS)),
        "analysis_splits": os.environ.get("SUBSTRUCTURE_ANALYSIS_SPLITS", "test"),
        "random_seed": RANDOM_SEED,
        "matched_random_per_bridge": N_MATCHED_RANDOM,
        "helix_min_segment_length": HELIX_MIN_SEGMENT_LENGTH,
        "helix_specificity": 0.95,
        "minimum_true_helix_auroc": MIN_TRUE_HELIX_AUROC,
        "replay_r2_tolerance": REPLAY_R2_TOLERANCE,
        "input_results_dir": str(input_results),
        "coefficients_sha256": sha256_file(coefficients),
        "split_assignments_sha256": sha256_file(assignments),
        "dssp_executable": str(executable),
        "dssp_version": version,
        "generated_at": now_timestamp(),
    }
    existing_path = output_root / "run_manifest.csv"
    if existing_path.is_file():
        existing = read_csv(existing_path)
        if existing:
            immutable = set(MANIFEST_FIELDS) - {"generated_at"}
            mismatches = [key for key in immutable if str(existing[0].get(key)) != str(manifest.get(key))]
            if mismatches:
                raise RuntimeError(
                    f"Output directory has an incompatible run manifest: {sorted(mismatches)}"
                )
    atomic_write_csv(existing_path, MANIFEST_FIELDS, [manifest])
    return manifest


def prepare_stage() -> None:
    started = time.time()
    input_results, output_root = output_paths()
    output_root.mkdir(parents=True, exist_ok=True)
    executable = Path(os.environ.get("DSSP_EXECUTABLE", "mkdssp")).expanduser()
    if shutil.which(str(executable)) is None and not executable.is_file():
        raise FileNotFoundError(f"DSSP executable is unavailable: {executable}")
    version = dssp_version(executable)
    manifest = write_manifest(input_results, output_root, executable, version)
    for layer in ANALYSIS_LAYERS:
        load_raw_coefficients(input_results, layer)

    assignments = load_split_assignments(input_results)
    validation_rows, failure_rows, skipped_rows = [], [], []
    assignment_ids = {row["protein_id"] for row in assignments}
    for protein_id in sorted(set(load_catalog_protein_ids()) - assignment_ids):
        skipped_rows.append(
            {"protein_id": protein_id, "source": source_for(protein_id), "split": "",
             "stage": "frozen_split", "reason": "catalogued protein absent from saved split_assignments.csv"}
        )
    triples_by_split_code = defaultdict(list)
    sheet_values = defaultdict(lambda: {"partner": [], "matched_random": [], "shift_1": [], "shift_2": []})
    successful = 0

    for position, assignment in enumerate(assignments, start=1):
        protein_started = time.time()
        protein_id = assignment["protein_id"]
        split = assignment["split"]
        source = assignment["source"]
        pdb_path = resolve_pdb(protein_id)
        pair_path = resolve_pair_block(protein_id, 0)
        reasons = []
        details = ""
        status = "success"
        dssp_exit_status = ""
        if not pdb_path.is_file() or not pair_path.is_file():
            reason = "missing_pdb" if not pdb_path.is_file() else "missing_pair_block_0"
            skipped_rows.append({"protein_id": protein_id, "source": source, "split": split, "stage": "prepare", "reason": reason})
            failure_rows.append({"protein_id": protein_id, "source": source, "split": split, "status": "excluded", "suspect_reasons": reason, "details": ""})
            log(f"Preparation {position}/{len(assignments)} {protein_id}: {reason}")
            continue
        try:
            residues = parse_pdb_residues(pdb_path)
            pair_block = np.load(pair_path, mmap_mode="r")
            n_pair = int(pair_block.shape[0])
            if pair_block.ndim != 3 or pair_block.shape[1:] != (n_pair, N_CHANNELS):
                raise ValueError(f"unexpected pair-block shape {pair_block.shape}")
            del pair_block
            n_pdb = len(residues)
            if n_pair != n_pdb:
                reasons.append("residue_count_mismatch")
            if n_pair != int(assignment["n_residues"]):
                reasons.append("frozen_split_residue_count_mismatch")

            try:
                dssp_records, raw_rows, dssp_exit_status, dssp_stderr = run_dssp(pdb_path, executable)
            except Exception as exc:
                dssp_records, raw_rows = {}, []
                dssp_exit_status = "failed"
                status = "dssp_failed"
                reasons.append("dssp_failure")
                details = str(exc)
                dssp_stderr = str(exc)

            codes = np.full(n_pdb, "-", dtype="U1")
            mapped = np.zeros(n_pdb, dtype=np.int8)
            for index, residue in enumerate(residues):
                value = dssp_records.get(residue_key(residue))
                if value is not None:
                    mapped[index] = 1
                    codes[index] = value[1] or "-"
            n_mapped = int(np.sum(mapped))
            if n_mapped != n_pair:
                reasons.append("dssp_residue_count_mismatch")

            chains = np.asarray([row["chain"] for row in residues], dtype="U4")
            h_segment_ids, h_segments = build_segments(codes, chains, "H")
            e_segment_ids, e_segments = build_segments(codes, chains, "E")
            bridges = deduplicate_bridges(raw_rows, residues, codes)
            controls = choose_bridge_controls(protein_id, residues, codes, e_segment_ids, bridges)
            coords = np.asarray([row["coordinate"] for row in residues], dtype=np.float64)
            true_distances = distance_matrix(coords)

            counts = Counter(codes.tolist())
            helix_count, strand_count = counts["H"], counts["E"]
            coil_count = counts["-"]
            if helix_count == 0 and strand_count == 0:
                reasons.append("zero_helices_and_zero_strands")
            if n_pdb and coil_count / n_pdb > 0.9:
                reasons.append("coil_fraction_above_0.9")

            adjacent = np.diag(true_distances, k=1)
            chain_transitions = 0
            numbering_gaps = 0
            deviations = 0
            for index, value in enumerate(adjacent):
                same_chain = residues[index]["chain"] == residues[index + 1]["chain"]
                numbering_contiguous = residues[index + 1]["pdb_resseq"] == residues[index]["pdb_resseq"] + 1
                if not same_chain:
                    chain_transitions += 1
                elif not numbering_contiguous:
                    numbering_gaps += 1
                elif not (ADJACENT_CA_MIN <= value <= ADJACENT_CA_MAX):
                    deviations += 1
            if deviations:
                reasons.append("adjacent_ca_distance_deviation")

            usable = int(n_pair == n_pdb == n_mapped and status == "success")
            if usable:
                encode_annotation_cache(
                    calibration_cache_path(output_root, protein_id), residues, codes,
                    h_segment_ids, e_segment_ids, h_segments, e_segments, bridges, controls,
                )
                triple_indices, triples = valid_offset_triples(true_distances, codes, chains)
                for code in ("H", "E"):
                    selected = triples[codes[triple_indices] == code]
                    if selected.size:
                        triples_by_split_code[(split, code)].append(selected)
                bridge_lookup = {row["bridge_id"]: row for row in bridges if row["main_eligible"]}
                for bridge in bridge_lookup.values():
                    key = (split, bridge["orientation"])
                    sheet_values[key]["partner"].append(true_distances[bridge["i"], bridge["j"]])
                for control in controls:
                    bridge = bridge_lookup.get(control["bridge_id"])
                    if bridge is not None:
                        key = (split, bridge["orientation"])
                        sheet_values[key][control["type"]].append(true_distances[control["i"], control["j"]])
                successful += 1

            main_bridges = [row for row in bridges if row["main_eligible"]]
            validation_rows.append(
                {
                    "protein_id": protein_id, "source": source, "split": split,
                    "status": "suspect" if reasons else status, "usable_for_structure": usable,
                    "dssp_exit_status": dssp_exit_status, "n_pair_residues": n_pair,
                    "n_pdb_ca_residues": n_pdb, "n_dssp_records": len(dssp_records),
                    "n_dssp_mapped": n_mapped,
                    **{f"n_dssp_{code if code != '-' else 'coil'}": counts[code] for code in DSSP_CODES},
                    "helix_fraction": helix_count / n_pdb if n_pdb else math.nan,
                    "strand_fraction": strand_count / n_pdb if n_pdb else math.nan,
                    "coil_fraction": coil_count / n_pdb if n_pdb else math.nan,
                    "n_helix_segments": len(h_segments), "n_strand_segments": len(e_segments),
                    "helix_segment_lengths": ";".join(str(row["length"]) for row in h_segments),
                    "strand_segment_lengths": ";".join(str(row["length"]) for row in e_segments),
                    "n_bridge_pairs": len(bridges), "n_main_bridge_pairs": len(main_bridges),
                    "n_parallel_bridge_pairs": sum(row["orientation"] == "parallel" for row in main_bridges),
                    "n_antiparallel_bridge_pairs": sum(row["orientation"] == "antiparallel" for row in main_bridges),
                    "n_unknown_orientation_bridge_pairs": sum(row["orientation"] == "unknown" for row in bridges),
                    "n_array_adjacencies": len(adjacent), "n_chain_transitions": chain_transitions,
                    "n_numbering_gaps": numbering_gaps, "n_adjacent_ca_deviations": deviations,
                    "mean_adjacent_ca_distance": finite_mean(adjacent),
                    "max_adjacent_ca_distance": float(np.max(adjacent)) if adjacent.size else math.nan,
                    "suspect_reasons": ";".join(sorted(set(reasons))), "pdb_path": str(pdb_path),
                    "pair_block_0_path": str(pair_path),
                    "runtime_seconds": round(time.time() - protein_started, 3),
                }
            )
            if reasons:
                failure_rows.append(
                    {"protein_id": protein_id, "source": source, "split": split,
                     "status": status if not usable else "suspect",
                     "suspect_reasons": ";".join(sorted(set(reasons))),
                     "details": details or dssp_stderr}
                )
        except Exception as exc:
            traceback.print_exc()
            failure_rows.append(
                {"protein_id": protein_id, "source": source, "split": split,
                 "status": "excluded", "suspect_reasons": "unexpected_preparation_failure",
                 "details": f"{type(exc).__name__}: {exc}"}
            )
            skipped_rows.append(
                {"protein_id": protein_id, "source": source, "split": split,
                 "stage": "prepare", "reason": f"unexpected failure: {exc}"}
            )
        if position % 10 == 0 or position == len(assignments):
            log(f"Prepared {position}/{len(assignments)} proteins; structurally usable={successful}")

    atomic_write_csv(output_root / "dssp_validation.csv", DSSP_VALIDATION_FIELDS, validation_rows)
    atomic_write_csv(output_root / "dssp_failures.csv", DSSP_FAILURE_FIELDS, failure_rows)
    atomic_write_csv(output_root / "skipped_proteins.csv", SKIPPED_FIELDS, skipped_rows)

    def stacked(split, code):
        pieces = triples_by_split_code[(split, code)]
        return np.vstack(pieces) if pieces else np.empty((0, 3), dtype=np.float64)

    train_h, train_e = stacked("train", "H"), stacked("train", "E")
    if not len(train_h) or not len(train_e):
        raise RuntimeError("Training structures contain no valid H/E offset triples")
    combined = np.vstack((train_h, train_e))
    z_mean = np.mean(combined, axis=0, dtype=np.float64)
    z_scale = np.std(combined, axis=0, dtype=np.float64)
    z_scale[z_scale < FLOAT_EPS] = 1.0
    mu_h_raw = np.mean(train_h, axis=0, dtype=np.float64)
    mu_e_raw = np.mean(train_e, axis=0, dtype=np.float64)
    base_calibration = {
        "zscore_mean": z_mean.tolist(), "zscore_scale": z_scale.tolist(),
        "mu_helix_raw": mu_h_raw.tolist(), "mu_strand_raw": mu_e_raw.tolist(),
        "mu_helix_z": ((mu_h_raw - z_mean) / z_scale).tolist(),
        "mu_strand_z": ((mu_e_raw - z_mean) / z_scale).tolist(),
    }
    train_h_score, _train_h_close_z, _ = score_triples(train_h, base_calibration)
    train_e_score, _, _ = score_triples(train_e, base_calibration)
    train_h_close_raw = np.linalg.norm(train_h - mu_h_raw, axis=1)
    score_threshold = empirical_quantile(train_e_score, 0.95, "higher")
    closeness_threshold = empirical_quantile(train_h_close_raw, 0.95, "higher")
    matched_train = []
    for orientation in ("parallel", "antiparallel"):
        matched_train.extend(sheet_values[("train", orientation)]["matched_random"])
    sheet_distance_threshold = (
        empirical_quantile(matched_train, 0.05, "lower") if matched_train else math.nan
    )
    calibration = {
        **base_calibration,
        "helix_score_threshold_95_specificity": score_threshold,
        "helix_closeness_threshold_95_sensitivity": closeness_threshold,
        "sheet_partner_distance_threshold_95_specificity": sheet_distance_threshold,
        "random_seed": RANDOM_SEED,
        "matched_random_per_bridge": N_MATCHED_RANDOM,
    }
    atomic_write_text(output_root / "calibration.json", json.dumps(calibration, indent=2, sort_keys=True) + "\n")

    long_validation = []
    for index, value in enumerate(mu_h_raw, start=2):
        long_validation.append({"category": "helix_centroid", "metric": f"mu_helix_offset_{index}", "split": "train", "orientation": "", "value": value, "n": len(train_h), "threshold": "", "status": "check", "details": "raw Angstrom centroid"})
    for index, value in enumerate(mu_e_raw, start=2):
        long_validation.append({"category": "strand_centroid", "metric": f"mu_strand_offset_{index}", "split": "train", "orientation": "", "value": value, "n": len(train_e), "threshold": "", "status": "check", "details": "raw Angstrom centroid"})
    for split in ("train", "test"):
        h, e = stacked(split, "H"), stacked(split, "E")
        h_scores = score_triples(h, calibration)[0] if len(h) else np.empty(0)
        e_scores = score_triples(e, calibration)[0] if len(e) else np.empty(0)
        auc = safe_auc(np.r_[np.ones(len(h_scores)), np.zeros(len(e_scores))], np.r_[h_scores, e_scores])
        long_validation.append(
            {"category": "helix_score", "metric": "true_matrix_auroc", "split": split,
             "orientation": "", "value": auc, "n": len(h_scores) + len(e_scores),
             "threshold": MIN_TRUE_HELIX_AUROC, "status": "pass" if auc >= MIN_TRUE_HELIX_AUROC else "fail",
             "details": "H versus E using fixed training centroids"}
        )
    long_validation.extend(
        [
            {"category": "helix_gate", "metric": "score_threshold", "split": "train", "orientation": "", "value": score_threshold, "n": len(train_e_score), "threshold": 0.95, "status": "calibrated", "details": "95th percentile of true E scores"},
            {"category": "helix_gate", "metric": "closeness_threshold", "split": "train", "orientation": "", "value": closeness_threshold, "n": len(train_h_close_raw), "threshold": 0.95, "status": "calibrated", "details": "95th percentile of true H raw-Angstrom distances to the empirical helix centroid"},
            {"category": "sheet_gate", "metric": "partner_distance_threshold", "split": "train", "orientation": "all", "value": sheet_distance_threshold, "n": len(matched_train), "threshold": 0.95, "status": "calibrated", "details": "5th percentile of matched-random true distances"},
        ]
    )
    for split in ("train", "test"):
        for orientation in ("parallel", "antiparallel"):
            values = sheet_values[(split, orientation)]
            partner = np.asarray(values["partner"], dtype=np.float64)
            for contrast in ("matched_random", "shift_1", "shift_2"):
                negative = np.asarray(values[contrast], dtype=np.float64)
                auc = safe_auc(
                    np.r_[np.ones(len(partner)), np.zeros(len(negative))],
                    -np.r_[partner, negative],
                )
                long_validation.append(
                    {"category": "register_ceiling", "metric": f"partner_vs_{contrast}_auroc",
                     "split": split, "orientation": orientation, "value": auc,
                     "n": len(partner) + len(negative), "threshold": "", "status": "check",
                     "details": f"partner_mean={finite_mean(partner):.6g}; contrast_mean={finite_mean(negative):.6g}"}
                )
                long_validation.extend(
                    [
                        {"category": "register_ceiling", "metric": f"true_partner_mean_for_{contrast}",
                         "split": split, "orientation": orientation, "value": finite_mean(partner),
                         "n": len(partner), "threshold": "", "status": "check", "details": "Angstrom"},
                        {"category": "register_ceiling", "metric": f"true_{contrast}_mean",
                         "split": split, "orientation": orientation, "value": finite_mean(negative),
                         "n": len(negative), "threshold": "", "status": "check", "details": "Angstrom"},
                    ]
                )
    helix_fraction = sum(int(row["n_dssp_H"]) for row in validation_rows) / max(sum(int(row["n_pdb_ca_residues"]) for row in validation_rows), 1)
    strand_fraction = sum(int(row["n_dssp_E"]) for row in validation_rows) / max(sum(int(row["n_pdb_ca_residues"]) for row in validation_rows), 1)
    for metric, value, expected in (("aggregate_helix_fraction", helix_fraction, "roughly 0.30-0.35"), ("aggregate_strand_fraction", strand_fraction, "roughly 0.20-0.25")):
        long_validation.append({"category": "dssp_dataset", "metric": metric, "split": "all", "orientation": "", "value": value, "n": successful, "threshold": "advisory", "status": "check", "details": expected + "; dataset composition may shift this"})
    long_validation.extend(
        [
            {"category": "indexing", "metric": "total_adjacent_ca_deviations", "split": "all", "orientation": "",
             "value": sum(int(row["n_adjacent_ca_deviations"]) for row in validation_rows),
             "n": sum(int(row["n_array_adjacencies"]) for row in validation_rows), "threshold": 0,
             "status": "pass" if not any(int(row["n_adjacent_ca_deviations"]) for row in validation_rows) else "warning",
             "details": f"expected same-chain, numbering-contiguous CA distance in [{ADJACENT_CA_MIN}, {ADJACENT_CA_MAX}] Å"},
            {"category": "indexing", "metric": "proteins_with_residue_count_mismatch", "split": "all", "orientation": "",
             "value": sum("residue_count_mismatch" in row["suspect_reasons"] for row in validation_rows),
             "n": len(validation_rows), "threshold": 0,
             "status": "pass" if not any("residue_count_mismatch" in row["suspect_reasons"] for row in validation_rows) else "fail",
             "details": "PDB CA count versus pair-block dimension"},
        ]
    )
    atomic_write_csv(output_root / "validation.csv", VALIDATION_FIELDS, long_validation)

    test_auc_rows = [row for row in long_validation if row["category"] == "helix_score" and row["split"] == "test"]
    if not test_auc_rows:
        raise RuntimeError(
            "True-distance held-out helix AUROC could not be computed: "
            f"no test result was produced; see {output_root / 'validation.csv'}"
        )
    test_auc = float(test_auc_rows[0]["value"])
    if not np.isfinite(test_auc) or test_auc < MIN_TRUE_HELIX_AUROC:
        raise RuntimeError(
            "True-distance held-out helix AUROC "
            f"was {test_auc:.6f}, below the {MIN_TRUE_HELIX_AUROC:.2f} validation gate; "
            f"see {output_root / 'validation.csv'}"
        )
    completion = {"status": "complete", "manifest": manifest, "n_assignments": len(assignments), "n_structure_usable": successful, "runtime_seconds": round(time.time() - started, 3)}
    atomic_write_text(output_root / "preparation_complete.json", json.dumps(completion, indent=2, sort_keys=True) + "\n")
    log(f"Preparation complete in {time.time() - started:.1f}s; outputs={output_root}")


def decode_distances(pair_block: np.ndarray, intercept: float, coefficients: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    if pair_block.ndim != 3 or pair_block.shape[0] != pair_block.shape[1] or pair_block.shape[2] != N_CHANNELS:
        raise ValueError(f"Unexpected pair-block shape: {pair_block.shape}")
    n_residues = pair_block.shape[0]
    pair_i, pair_j = np.tril_indices(n_residues, k=-1)
    predicted = np.empty(len(pair_i), dtype=np.float64)
    chunk_size = int(os.environ.get("SUBSTRUCTURE_DECODE_CHUNK_ROWS", "250000"))
    for start in range(0, len(pair_i), chunk_size):
        end = min(start + chunk_size, len(pair_i))
        left = np.asarray(pair_block[pair_i[start:end], pair_j[start:end], :], dtype=np.float64)
        right = np.asarray(pair_block[pair_j[start:end], pair_i[start:end], :], dtype=np.float64)
        predicted[start:end] = intercept + left @ coefficients[:N_CHANNELS] + right @ coefficients[N_CHANNELS:]
    matrix = np.zeros((n_residues, n_residues), dtype=np.float64)
    matrix[pair_i, pair_j] = predicted
    matrix[pair_j, pair_i] = predicted
    return matrix, pair_i, pair_j, predicted


def segment_metadata(cache: dict, index: int) -> tuple[int, int, int, int]:
    segment_id = int(cache["h_segment_ids"][index])
    if segment_id < 0:
        return -1, -1, -1, 0
    return (
        segment_id,
        int(cache["h_segment_start"][segment_id]),
        int(cache["h_segment_end"][segment_id]),
        int(cache["h_segment_length"][segment_id]),
    )


def build_helix_rows(
    layer: int, assignment: dict, cache: dict, decoded: np.ndarray,
    true_distances: np.ndarray, calibration: dict,
) -> tuple[list[dict], list[dict]]:
    indices, decoded_triples = valid_offset_triples(decoded, cache["dssp_codes"], cache["chains"])
    true_triples = np.asarray(
        [[true_distances[index, index + offset] for offset in (2, 3, 4)] for index in indices],
        dtype=np.float64,
    ).reshape((-1, 3))
    scores, distance_h, distance_e = score_triples(decoded_triples, calibration)
    raw_helix_centroid = np.asarray(calibration["mu_helix_raw"], dtype=np.float64)
    raw_distance_h = np.linalg.norm(decoded_triples - raw_helix_centroid, axis=1)
    score_threshold = float(calibration["helix_score_threshold_95_specificity"])
    closeness_threshold = float(calibration["helix_closeness_threshold_95_sensitivity"])
    rows = []
    for position, index in enumerate(indices):
        code = str(cache["dssp_codes"][index])
        segment_id, segment_start, segment_end, segment_length = segment_metadata(cache, int(index))
        if code == "H":
            if segment_length < HELIX_MIN_SEGMENT_LENGTH or index in {segment_start, segment_end}:
                continue
        raw_rmse = float(np.sqrt(np.mean((decoded_triples[position] - true_triples[position]) ** 2)))
        relative = int(scores[position] > score_threshold)
        absolute = int(raw_distance_h[position] <= closeness_threshold)
        rows.append(
            {
                "variant": VARIANT, "layer": layer, "split": assignment["split"],
                "protein_id": assignment["protein_id"], "source": assignment["source"],
                "residue_index": int(index), "chain": str(cache["chains"][index]),
                "pdb_resseq": int(cache["pdb_resseq"][index]),
                "insertion_code": str(cache["insertion_codes"][index]),
                "resname": str(cache["resnames"][index]), "dssp_code": code,
                "segment_id": segment_id, "segment_length": segment_length,
                "offset_2_decoded": decoded_triples[position, 0],
                "offset_3_decoded": decoded_triples[position, 1],
                "offset_4_decoded": decoded_triples[position, 2],
                "offset_2_true": true_triples[position, 0], "offset_3_true": true_triples[position, 1],
                "offset_4_true": true_triples[position, 2], "score": scores[position],
                "distance_to_helix_centroid": raw_distance_h[position],
                "distance_to_helix_centroid_z": distance_h[position],
                "distance_to_strand_centroid_z": distance_e[position], "raw_triple_rmse": raw_rmse,
                "passes_relative_gate": relative, "passes_absolute_gate": absolute,
                "looks_helical": int(relative and absolute),
            }
        )
    grouped = defaultdict(list)
    for row in rows:
        if row["dssp_code"] == "H":
            grouped[row["segment_id"]].append(row)
    segment_rows = []
    for segment_id, selected in sorted(grouped.items()):
        mean_score = finite_mean(row["score"] for row in selected)
        mean_close = finite_mean(row["distance_to_helix_centroid"] for row in selected)
        segment_rows.append(
            {
                "variant": VARIANT, "layer": layer, "split": assignment["split"],
                "protein_id": assignment["protein_id"], "source": assignment["source"],
                "segment_id": segment_id,
                "start_residue_index": int(cache["h_segment_start"][segment_id]),
                "end_residue_index": int(cache["h_segment_end"][segment_id]),
                "original_segment_length": int(cache["h_segment_length"][segment_id]),
                "n_scored_residues": len(selected), "mean_score": mean_score,
                "mean_distance_to_helix_centroid": mean_close,
                "mean_raw_triple_rmse": finite_mean(row["raw_triple_rmse"] for row in selected),
                "fraction_residues_looking_helical": finite_mean(row["looks_helical"] for row in selected),
                "visible": int(mean_score > score_threshold and mean_close <= closeness_threshold),
                "first_visible_layer": "", "score_monotone_nondecreasing": "",
                "closeness_monotone_nonincreasing": "", "crossed_then_reverted": "",
            }
        )
    return rows, segment_rows


def build_bridge_rows(
    layer: int, assignment: dict, cache: dict, decoded: np.ndarray,
    true_distances: np.ndarray, calibration: dict,
) -> list[dict]:
    controls_by_bridge = defaultdict(lambda: defaultdict(list))
    for bridge_id, kind, i, j in zip(
        cache["control_bridge_id"], cache["control_type"], cache["control_i"], cache["control_j"]
    ):
        controls_by_bridge[int(bridge_id)][str(kind)].append((int(i), int(j)))
    rows = []
    threshold = float(calibration["sheet_partner_distance_threshold_95_specificity"])
    for bridge_id, i, j, orientation, eligible in zip(
        cache["bridge_id"], cache["bridge_i"], cache["bridge_j"],
        cache["bridge_orientation"], cache["bridge_main_eligible"],
    ):
        if not int(eligible):
            continue
        bridge_id, i, j = int(bridge_id), int(i), int(j)
        data = {}
        for kind in ("matched_random", "shift_1", "shift_2"):
            pairs = controls_by_bridge[bridge_id][kind]
            data[kind] = {
                "decoded": [decoded[a, b] for a, b in pairs],
                "true": [true_distances[a, b] for a, b in pairs],
            }
        partner_decoded = float(decoded[i, j])
        rows.append(
            {
                "variant": VARIANT, "layer": layer, "split": assignment["split"],
                "protein_id": assignment["protein_id"], "source": assignment["source"],
                "bridge_id": bridge_id, "i": i, "j": j, "orientation": str(orientation),
                "sequence_separation": abs(i - j), "decoded_partner_distance": partner_decoded,
                "true_partner_distance": true_distances[i, j],
                "n_matched_random": len(data["matched_random"]["decoded"]),
                "mean_decoded_random_distance": finite_mean(data["matched_random"]["decoded"]),
                "mean_true_random_distance": finite_mean(data["matched_random"]["true"]),
                "matched_random_margin": finite_mean(data["matched_random"]["decoded"]) - partner_decoded,
                "n_shift_1": len(data["shift_1"]["decoded"]),
                "mean_decoded_shift_1_distance": finite_mean(data["shift_1"]["decoded"]),
                "mean_true_shift_1_distance": finite_mean(data["shift_1"]["true"]),
                "shift_1_margin": finite_mean(data["shift_1"]["decoded"]) - partner_decoded,
                "n_shift_2": len(data["shift_2"]["decoded"]),
                "mean_decoded_shift_2_distance": finite_mean(data["shift_2"]["decoded"]),
                "mean_true_shift_2_distance": finite_mean(data["shift_2"]["true"]),
                "shift_2_margin": finite_mean(data["shift_2"]["decoded"]) - partner_decoded,
                "_decoded_matched_random_values": data["matched_random"]["decoded"],
                "_decoded_shift_1_values": data["shift_1"]["decoded"],
                "_decoded_shift_2_values": data["shift_2"]["decoded"],
                "visible": int(np.isfinite(threshold) and partner_decoded <= threshold),
                "first_visible_layer": "", "partner_distance_monotone_nonincreasing": "",
                "crossed_then_reverted": "",
            }
        )
    return rows


def distance_bin_indices(values: np.ndarray) -> np.ndarray:
    result = np.full(len(values), -1, dtype=np.int8)
    for index, (lower, upper, _label) in enumerate(DISTANCE_BINS):
        result[(values >= lower) & (values < upper)] = index
    return result


def separation_bin_indices(pair_i: np.ndarray, pair_j: np.ndarray, chains: np.ndarray) -> np.ndarray:
    separation = pair_i - pair_j
    result = np.full(len(pair_i), -1, dtype=np.int8)
    same_chain = chains[pair_i] == chains[pair_j]
    for index, (lower, upper, _label) in enumerate(SEPARATION_BINS):
        result[same_chain & (separation >= lower) & (separation <= upper)] = index
    result[~same_chain] = len(SEPARATION_BINS)
    return result


def grouped_accuracy_rows(
    layer: int, split: str, group_values: np.ndarray, group_labels: list[str],
    distance_bins: np.ndarray, y_true: np.ndarray, y_pred: np.ndarray,
    include_distance_bins: bool,
) -> list[dict]:
    rows = []

    def summarize_keys(keys, valid, n_keys):
        selected_keys = np.asarray(keys[valid], dtype=np.int64)
        selected_true = np.asarray(y_true[valid], dtype=np.float64)
        error = np.asarray(y_pred[valid], dtype=np.float64) - selected_true
        return {
            "n": np.bincount(selected_keys, minlength=n_keys).astype(np.int64),
            "sum_abs": np.bincount(selected_keys, weights=np.abs(error), minlength=n_keys),
            "sum_squared": np.bincount(selected_keys, weights=error * error, minlength=n_keys),
            "sum_error": np.bincount(selected_keys, weights=error, minlength=n_keys),
            "sum_true": np.bincount(selected_keys, weights=selected_true, minlength=n_keys),
            "sum_true_squared": np.bincount(selected_keys, weights=selected_true * selected_true, minlength=n_keys),
        }

    def row_from_moments(group, distance_bin, scope, moments, key):
        n = int(moments["n"][key])
        sum_squared = float(moments["sum_squared"][key])
        sum_true = float(moments["sum_true"][key])
        sum_true_squared = float(moments["sum_true_squared"][key])
        ss_total = sum_true_squared - sum_true * sum_true / max(n, 1)
        return {
            "variant": VARIANT, "layer": layer, "split": split, "group": group,
            "distance_bin": distance_bin, "n_pairs": n,
            "mae": float(moments["sum_abs"][key]) / n,
            "rmse": math.sqrt(sum_squared / n),
            "bias": float(moments["sum_error"][key]) / n,
            "r2": 1.0 - sum_squared / ss_total if ss_total > FLOAT_EPS else math.nan,
            "sparse": int(n < 100), "metric_scope": scope,
            "_sum_abs_error": float(moments["sum_abs"][key]),
            "_sum_squared_error": sum_squared,
            "_sum_error": float(moments["sum_error"][key]),
            "_sum_true": sum_true, "_sum_true_squared": sum_true_squared,
        }

    if include_distance_bins:
        composite = group_values.astype(np.int64) * len(DISTANCE_BINS) + distance_bins.astype(np.int64)
        valid = (group_values >= 0) & (distance_bins >= 0)
        composite_moments = summarize_keys(
            composite, valid, len(group_labels) * len(DISTANCE_BINS)
        )
        for key in np.flatnonzero(composite_moments["n"]):
            group_index, distance_index = divmod(int(key), len(DISTANCE_BINS))
            rows.append(
                row_from_moments(
                    group_labels[group_index], DISTANCE_BINS[distance_index][2],
                    "within_distance_bin", composite_moments, int(key),
                )
            )
        group_valid = group_values >= 0
        group_moments = summarize_keys(group_values, group_valid, len(group_labels))
        for group_index in np.flatnonzero(group_moments["n"]):
            rows.append(
                row_from_moments(
                    group_labels[int(group_index)], "all_raw", "raw_all_distances",
                    group_moments, int(group_index),
                )
            )
    else:
        valid = group_values >= 0
        moments = summarize_keys(group_values, valid, len(group_labels))
        for group_index in np.flatnonzero(moments["n"]):
            rows.append(
                row_from_moments(
                    group_labels[int(group_index)], "all", "raw_group",
                    moments, int(group_index),
                )
            )
    return rows


def accuracy_rows_for_protein(
    layer: int, split: str, cache: dict, pair_i: np.ndarray, pair_j: np.ndarray,
    y_true: np.ndarray, y_pred: np.ndarray,
) -> tuple[list[dict], list[dict], list[dict]]:
    residue_codes = np.asarray(
        [AA_TO_CODE.get(str(name), -1) for name in cache["resnames"]], dtype=np.int16
    )
    left_codes, right_codes = residue_codes[pair_i], residue_codes[pair_j]
    aa_values = np.full(len(pair_i), -1, dtype=np.int16)
    canonical = (left_codes >= 0) & (right_codes >= 0)
    aa_values[canonical] = AA_PAIR_CODE[left_codes[canonical], right_codes[canonical]]
    dist_bins = distance_bin_indices(y_true)
    sep_values = separation_bin_indices(pair_i, pair_j, cache["chains"])
    distance_labels = [row[2] for row in DISTANCE_BINS]
    separation_labels = [row[2] for row in SEPARATION_BINS] + ["interchain"]
    distance_group = dist_bins.copy()
    return (
        grouped_accuracy_rows(layer, split, aa_values, list(AA_PAIR_LABELS), dist_bins, y_true, y_pred, True),
        grouped_accuracy_rows(layer, split, distance_group, distance_labels, dist_bins, y_true, y_pred, False),
        grouped_accuracy_rows(layer, split, sep_values, separation_labels, dist_bins, y_true, y_pred, False),
    )


def bridge_summary_fields(rows: list[dict], orientation: str) -> dict:
    selected = rows if orientation == "all" else [row for row in rows if row["orientation"] == orientation]
    prefix = f"sheet_{orientation}_"
    result = {f"{prefix}n_bridges": len(selected)}
    partner = [float(row["decoded_partner_distance"]) for row in selected]
    result[f"{prefix}mean_partner_distance"] = finite_mean(partner)
    for kind, field, margin_field in (
        ("matched_random", "mean_decoded_random_distance", "matched_random_margin"),
        ("shift_1", "mean_decoded_shift_1_distance", "shift_1_margin"),
        ("shift_2", "mean_decoded_shift_2_distance", "shift_2_margin"),
    ):
        internal_field = f"_decoded_{kind}_values"
        negative = []
        for row in selected:
            if internal_field in row:
                negative.extend(float(value) for value in row[internal_field])
            elif str(row[field]) not in {"", "nan"} and np.isfinite(float(row[field])):
                negative.append(float(row[field]))
        result[f"{prefix}mean_{'random' if kind == 'matched_random' else kind}_distance"] = finite_mean(negative)
        result[f"{prefix}{margin_field}"] = finite_mean(float(row[margin_field]) for row in selected)
        result[f"{prefix}{kind}_auroc"] = safe_auc(
            np.r_[np.ones(len(partner)), np.zeros(len(negative))], -np.r_[partner, negative]
        )
    return result


def layer_stage(layer: int) -> None:
    if layer not in ANALYSIS_LAYERS:
        raise ValueError(f"Layer must be one of {ANALYSIS_LAYERS}; got {layer}")
    started = time.time()
    input_results, output_root = output_paths()
    if not (output_root / "preparation_complete.json").is_file():
        raise RuntimeError("Preparation stage is incomplete")
    calibration = json.loads((output_root / "calibration.json").read_text())
    intercept, coefficients = load_raw_coefficients(input_results, layer)
    saved_metrics = load_saved_metrics(input_results, layer)
    requested_splits = {
        token.strip() for token in os.environ.get("SUBSTRUCTURE_ANALYSIS_SPLITS", "test").split(",") if token.strip()
    }
    assignments = [row for row in load_split_assignments(input_results) if row["split"] in requested_splits]
    shard = output_root / "layer_shards" / f"layer_{layer:02d}"
    shard.mkdir(parents=True, exist_ok=True)
    complete_path = shard / "complete.json"
    if complete_path.is_file():
        log(f"Layer {layer:02d} shard already complete; skipping")
        return

    per_rows, helix_rows, segment_rows, bridge_rows, skipped = [], [], [], [], []
    replay_errors = []
    accuracy_rows = {"amino_acid_pair": [], "distance": [], "sequence_separation": []}
    for position, assignment in enumerate(assignments, start=1):
        protein_id = assignment["protein_id"]
        cache_path = calibration_cache_path(output_root, protein_id)
        pair_path = resolve_pair_block(protein_id, layer)
        if not cache_path.is_file() or not pair_path.is_file():
            skipped.append({"protein_id": protein_id, "source": assignment["source"], "split": assignment["split"], "stage": f"layer_{layer}", "reason": "missing_annotation_cache" if not cache_path.is_file() else "missing_pair_block"})
            continue
        try:
            cache = load_annotation_cache(cache_path)
            pair_block = np.load(pair_path)
            decoded, pair_i, pair_j, predicted = decode_distances(pair_block, intercept, coefficients)
            del pair_block
            if len(cache["coords"]) != decoded.shape[0]:
                raise ValueError(f"cache/pair dimension mismatch {len(cache['coords'])} != {decoded.shape[0]}")
            true_distances = distance_matrix(cache["coords"])
            y_true = true_distances[pair_i, pair_j]
            general = metrics_from_arrays(y_true, predicted)
            saved = saved_metrics.get(protein_id)
            if saved is None:
                raise RuntimeError("saved per-protein metric is missing for coefficient replay check")
            saved_r2 = float(saved["per_protein_r2"])
            difference = general["r2"] - saved_r2
            if np.isfinite(difference) and abs(difference) > REPLAY_R2_TOLERANCE:
                raise RuntimeError(f"coefficient replay R2 mismatch {difference:+.6g}")

            protein_helix, protein_segments = build_helix_rows(
                layer, assignment, cache, decoded, true_distances, calibration
            )
            protein_bridges = build_bridge_rows(
                layer, assignment, cache, decoded, true_distances, calibration
            )
            aa_rows, distance_rows, separation_rows = accuracy_rows_for_protein(
                layer, assignment["split"], cache, pair_i, pair_j, y_true, predicted
            )
            # Pair-level additive moments are accumulated below from each
            # protein's full arrays; retain these rows temporarily by protein.
            for row in aa_rows:
                row["protein_id"] = protein_id
            for row in distance_rows:
                row["protein_id"] = protein_id
            for row in separation_rows:
                row["protein_id"] = protein_id
            accuracy_rows["amino_acid_pair"].extend(aa_rows)
            accuracy_rows["distance"].extend(distance_rows)
            accuracy_rows["sequence_separation"].extend(separation_rows)

            h = [row for row in protein_helix if row["dssp_code"] == "H"]
            e = [row for row in protein_helix if row["dssp_code"] == "E"]
            per_rows.append(
                {
                    "variant": VARIANT, "feature_dim": FEATURE_DIM, "split_id": SPLIT_ID,
                    "layer": layer, "split": assignment["split"], "protein_id": protein_id,
                    "source": assignment["source"], "n_residues": len(cache["coords"]),
                    "n_pairs": general["n_pairs"], "per_protein_r2": general["r2"],
                    "per_protein_mae": general["mae"], "per_protein_rmse": general["rmse"],
                    "mean_error_pred_minus_true": general["bias"],
                    "saved_per_protein_r2": saved_r2, "replay_r2_difference": difference,
                    "n_negative_predictions": int(np.sum(predicted < 0)),
                    "negative_prediction_fraction": float(np.mean(predicted < 0)),
                    "n_helix_residues": len(h), "n_strand_residues": len(e),
                    "mean_helix_score": finite_mean(row["score"] for row in h),
                    "mean_strand_score": finite_mean(row["score"] for row in e),
                    "helix_strand_auroc": safe_auc(
                        np.r_[np.ones(len(h)), np.zeros(len(e))],
                        np.r_[[row["score"] for row in h], [row["score"] for row in e]],
                    ),
                    "mean_helix_centroid_distance": finite_mean(row["distance_to_helix_centroid"] for row in h),
                    "n_helix_segments": len(protein_segments), "n_bridge_pairs": len(protein_bridges),
                    "n_parallel_bridge_pairs": sum(row["orientation"] == "parallel" for row in protein_bridges),
                    "n_antiparallel_bridge_pairs": sum(row["orientation"] == "antiparallel" for row in protein_bridges),
                }
            )
            helix_rows.extend(protein_helix); segment_rows.extend(protein_segments); bridge_rows.extend(protein_bridges)
        except Exception as exc:
            traceback.print_exc()
            if "coefficient replay" in str(exc) or "saved per-protein metric" in str(exc):
                replay_errors.append(f"{protein_id}: {exc}")
            skipped.append({"protein_id": protein_id, "source": assignment["source"], "split": assignment["split"], "stage": f"layer_{layer}", "reason": f"{type(exc).__name__}: {exc}"})
        if position % 10 == 0 or position == len(assignments):
            log(f"Layer {layer:02d}: processed {position}/{len(assignments)} proteins")

    if replay_errors:
        raise RuntimeError(
            f"Layer {layer} failed coefficient replay validation for {len(replay_errors)} proteins; "
            + " | ".join(replay_errors[:5])
        )
    if not per_rows:
        raise RuntimeError(f"Layer {layer} produced no usable protein results")

    # Pool additive pair moments so raw group MAE/RMSE/bias/R2 are exact without
    # retaining pair-level predictions. Distance-standardized rows have no
    # coherent pooled R2 and deliberately leave it blank.
    pooled_accuracy = {}
    for kind, rows in accuracy_rows.items():
        grouped = defaultdict(list)
        for row in rows:
            grouped[(row["split"], row["group"], row["distance_bin"], row["metric_scope"])].append(row)
        output = []
        for (split, group, distance_bin, scope), selected in sorted(grouped.items()):
            counts = np.asarray([int(row["n_pairs"]) for row in selected], dtype=np.float64)
            total = int(np.sum(counts))
            weights = counts / max(total, 1)
            if scope == "global_distance_distribution_standardized":
                mae = float(np.sum(weights * [float(row["mae"]) for row in selected]))
                mse = float(np.sum(weights * [float(row["rmse"]) ** 2 for row in selected]))
                bias = float(np.sum(weights * [float(row["bias"]) for row in selected]))
                pooled_r2 = math.nan
            else:
                sum_abs = sum(float(row["_sum_abs_error"]) for row in selected)
                sum_squared = sum(float(row["_sum_squared_error"]) for row in selected)
                sum_error = sum(float(row["_sum_error"]) for row in selected)
                sum_true = sum(float(row["_sum_true"]) for row in selected)
                sum_true_squared = sum(float(row["_sum_true_squared"]) for row in selected)
                mae = sum_abs / max(total, 1)
                mse = sum_squared / max(total, 1)
                bias = sum_error / max(total, 1)
                ss_total = sum_true_squared - sum_true * sum_true / max(total, 1)
                pooled_r2 = 1.0 - sum_squared / ss_total if ss_total > FLOAT_EPS else math.nan
            output.append({"variant": VARIANT, "layer": layer, "split": split, "group": group,
                           "distance_bin": distance_bin, "n_pairs": total, "mae": mae,
                           "rmse": math.sqrt(mse), "bias": bias, "r2": pooled_r2,
                           "sparse": int(total < 100), "metric_scope": scope})
        pooled_accuracy[kind] = output

    # Distance-standardize amino-acid comparisons once across the entire
    # evaluation split. Every amino-acid combination therefore receives the
    # same global held-out distance-bin distribution rather than a different
    # protein- or amino-acid-specific mixture.
    aa_output = pooled_accuracy["amino_acid_pair"]
    for split in sorted(requested_splits):
        within = [
            row for row in aa_output
            if row["split"] == split and row["metric_scope"] == "within_distance_bin"
        ]
        global_counts = {
            label: sum(int(row["n_pairs"]) for row in within if row["distance_bin"] == label)
            for _lower, _upper, label in DISTANCE_BINS
        }
        total_global = max(sum(global_counts.values()), 1)
        global_weights = {label: count / total_global for label, count in global_counts.items()}
        by_group = defaultdict(list)
        for row in within:
            by_group[row["group"]].append(row)
        standardized = []
        for group, selected in sorted(by_group.items()):
            weights, maes, mses, biases = [], [], [], []
            for row in selected:
                weight = global_weights[row["distance_bin"]]
                weights.append(weight); maes.append(float(row["mae"])); mses.append(float(row["rmse"]) ** 2)
                biases.append(float(row["bias"]))
            weights = np.asarray(weights, dtype=np.float64)
            weights /= max(np.sum(weights), FLOAT_EPS)
            total = sum(int(row["n_pairs"]) for row in selected)
            standardized.append(
                {"variant": VARIANT, "layer": layer, "split": split, "group": group,
                 "distance_bin": "distance_standardized", "n_pairs": total,
                 "mae": float(np.sum(weights * maes)),
                 "rmse": float(np.sqrt(np.sum(weights * mses))),
                 "bias": float(np.sum(weights * biases)), "r2": math.nan,
                 "sparse": int(total < 100),
                 "metric_scope": "global_distance_distribution_standardized"}
            )
        aa_output.extend(standardized)

    summary_rows = []
    for split in sorted(requested_splits):
        protein_selected = [row for row in per_rows if row["split"] == split]
        helix_selected = [row for row in helix_rows if row["split"] == split and row["dssp_code"] == "H"]
        strand_selected = [row for row in helix_rows if row["split"] == split and row["dssp_code"] == "E"]
        bridge_selected = [row for row in bridge_rows if row["split"] == split]
        row = {
            "variant": VARIANT, "feature_dim": FEATURE_DIM, "split_id": SPLIT_ID,
            "layer": layer, "split": split, "n_proteins": len(protein_selected),
            "n_pairs": sum(int(item["n_pairs"]) for item in protein_selected),
            "n_negative_predictions": sum(int(item["n_negative_predictions"]) for item in protein_selected),
            "negative_prediction_fraction": sum(int(item["n_negative_predictions"]) for item in protein_selected) / max(sum(int(item["n_pairs"]) for item in protein_selected), 1),
            "n_helix_residues": len(helix_selected), "n_strand_residues": len(strand_selected),
            "mean_helix_score": finite_mean(item["score"] for item in helix_selected),
            "mean_strand_score": finite_mean(item["score"] for item in strand_selected),
            "helix_strand_auroc": safe_auc(
                np.r_[np.ones(len(helix_selected)), np.zeros(len(strand_selected))],
                np.r_[[item["score"] for item in helix_selected], [item["score"] for item in strand_selected]],
            ),
            "mean_helix_centroid_distance": finite_mean(item["distance_to_helix_centroid"] for item in helix_selected),
            **{f"mean_helix_offset_{offset}_decoded": finite_mean(item[f"offset_{offset}_decoded"] for item in helix_selected) for offset in (2, 3, 4)},
            **{f"mean_helix_offset_{offset}_true": finite_mean(item[f"offset_{offset}_true"] for item in helix_selected) for offset in (2, 3, 4)},
        }
        for orientation in ("all", "parallel", "antiparallel"):
            row.update(bridge_summary_fields(bridge_selected, orientation))
        summary_rows.append(row)

    atomic_write_csv(shard / "per_protein.csv", PER_PROTEIN_FIELDS, per_rows)
    atomic_write_csv(shard / "helix_residue_scores.csv", HELIX_RESIDUE_FIELDS, helix_rows)
    atomic_write_csv(shard / "helix_segment_trajectories.csv", HELIX_SEGMENT_FIELDS, segment_rows)
    atomic_write_csv(shard / "bridge_pair_trajectories.csv", BRIDGE_FIELDS, bridge_rows)
    atomic_write_csv(shard / "accuracy_by_amino_acid_pair.csv", ACCURACY_FIELDS, pooled_accuracy["amino_acid_pair"])
    atomic_write_csv(shard / "accuracy_by_distance.csv", ACCURACY_FIELDS, pooled_accuracy["distance"])
    atomic_write_csv(shard / "accuracy_by_sequence_separation.csv", ACCURACY_FIELDS, pooled_accuracy["sequence_separation"])
    atomic_write_csv(shard / "layer_summary.csv", LAYER_SUMMARY_FIELDS, summary_rows)
    atomic_write_csv(shard / "skipped_proteins.csv", SKIPPED_FIELDS, skipped)
    completion = {"status": "complete", "layer": layer, "n_proteins": len(per_rows), "n_helix_rows": len(helix_rows), "n_bridge_rows": len(bridge_rows), "runtime_seconds": round(time.time() - started, 3)}
    atomic_write_text(complete_path, json.dumps(completion, indent=2, sort_keys=True) + "\n")
    log(f"Layer {layer:02d} complete in {time.time() - started:.1f}s; outputs={shard}")


def read_layer_rows(output_root: Path, filename: str) -> list[dict]:
    rows = []
    for layer in ANALYSIS_LAYERS:
        path = output_root / "layer_shards" / f"layer_{layer:02d}" / filename
        if not path.is_file():
            raise FileNotFoundError(f"Missing completed layer output: {path}")
        rows.extend(read_csv(path))
    return rows


def add_helix_trajectory_fields(rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["protein_id"], row["segment_id"])].append(row)
    for selected in grouped.values():
        selected.sort(key=lambda row: int(row["layer"]))
        scores = np.asarray([float(row["mean_score"]) for row in selected])
        closeness = np.asarray([float(row["mean_distance_to_helix_centroid"]) for row in selected])
        visible = np.asarray([int(row["visible"]) for row in selected], dtype=np.int8)
        first = next((int(row["layer"]) for row in selected if int(row["visible"])), "")
        reverted = int(any(visible[:index].any() and not visible[index] for index in range(len(visible))))
        for row in selected:
            row["first_visible_layer"] = first
            row["score_monotone_nondecreasing"] = int(np.all(np.diff(scores) >= -1e-12))
            row["closeness_monotone_nonincreasing"] = int(np.all(np.diff(closeness) <= 1e-12))
            row["crossed_then_reverted"] = reverted


def add_bridge_trajectory_fields(rows: list[dict]) -> None:
    grouped = defaultdict(list)
    for row in rows:
        grouped[(row["split"], row["protein_id"], row["bridge_id"])].append(row)
    for selected in grouped.values():
        selected.sort(key=lambda row: int(row["layer"]))
        distances = np.asarray([float(row["decoded_partner_distance"]) for row in selected])
        visible = np.asarray([int(row["visible"]) for row in selected], dtype=np.int8)
        first = next((int(row["layer"]) for row in selected if int(row["visible"])), "")
        reverted = int(any(visible[:index].any() and not visible[index] for index in range(len(visible))))
        for row in selected:
            row["first_visible_layer"] = first
            row["partner_distance_monotone_nonincreasing"] = int(np.all(np.diff(distances) <= 1e-12))
            row["crossed_then_reverted"] = reverted


def make_plots(output_root: Path, layer_rows: list[dict]) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    plot_dir = output_root / "visualizations"
    plot_dir.mkdir(parents=True, exist_ok=True)
    primary = sorted((row for row in layer_rows if row["split"] == "test"), key=lambda row: int(row["layer"]))
    layers = [int(row["layer"]) for row in primary]

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(layers, [float(row["mean_helix_score"]) for row in primary], marker="o", label="Known H residues")
    axis.plot(layers, [float(row["mean_strand_score"]) for row in primary], marker="o", label="Known E residues")
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set(xlabel="Layer", ylabel="Nearest-centroid helix score", title="Held-out helix/strand score trajectories")
    axis.legend(); figure.tight_layout()
    figure.savefig(plot_dir / "helix_strand_scores_vs_layer.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    colors = {2: "tab:blue", 3: "tab:orange", 4: "tab:green"}
    for offset in (2, 3, 4):
        axis.plot(layers, [float(row[f"mean_helix_offset_{offset}_decoded"]) for row in primary], marker="o", color=colors[offset], label=f"decoded i,i+{offset}")
        axis.axhline(finite_mean(float(row[f"mean_helix_offset_{offset}_true"]) for row in primary), color=colors[offset], linestyle="--", alpha=0.7, label=f"true i,i+{offset}")
    axis.set(xlabel="Layer", ylabel="CA distance (Å)", title="Held-out helix distance triples")
    axis.legend(ncol=2, fontsize=8); figure.tight_layout()
    figure.savefig(plot_dir / "helix_offsets_vs_layer.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(layers, [float(row["sheet_all_mean_partner_distance"]) for row in primary], marker="o", label="DSSP partner")
    axis.plot(layers, [float(row["sheet_all_mean_random_distance"]) for row in primary], marker="o", label="matched random")
    axis.set(xlabel="Layer", ylabel="Decoded CA distance (Å)", title="Held-out sheet partner separation")
    axis.legend(); figure.tight_layout()
    figure.savefig(plot_dir / "sheet_partner_vs_random_vs_layer.png", dpi=180); plt.close(figure)

    figure, axis = plt.subplots(figsize=(7.5, 4.8))
    axis.plot(layers, [float(row["sheet_all_shift_1_margin"]) for row in primary], marker="o", label="shift ±1 minus partner")
    axis.plot(layers, [float(row["sheet_all_shift_2_margin"]) for row in primary], marker="o", label="shift ±2 minus partner")
    axis.axhline(0.0, color="black", linewidth=0.8, alpha=0.5)
    axis.set(xlabel="Layer", ylabel="Within-layer distance margin (Å)", title="Held-out beta-sheet register separation")
    axis.legend(); figure.tight_layout()
    figure.savefig(plot_dir / "sheet_register_separation_vs_layer.png", dpi=180); plt.close(figure)


def consolidate_stage() -> None:
    started = time.time()
    input_results, output_root = output_paths()
    if not (output_root / "preparation_complete.json").is_file():
        raise RuntimeError("Preparation stage is incomplete")
    for layer in ANALYSIS_LAYERS:
        if not (output_root / "layer_shards" / f"layer_{layer:02d}" / "complete.json").is_file():
            raise RuntimeError(f"Layer {layer} is incomplete")
    file_fields = {
        "per_protein.csv": PER_PROTEIN_FIELDS,
        "helix_residue_scores.csv": HELIX_RESIDUE_FIELDS,
        "helix_segment_trajectories.csv": HELIX_SEGMENT_FIELDS,
        "bridge_pair_trajectories.csv": BRIDGE_FIELDS,
        "accuracy_by_amino_acid_pair.csv": ACCURACY_FIELDS,
        "accuracy_by_distance.csv": ACCURACY_FIELDS,
        "accuracy_by_sequence_separation.csv": ACCURACY_FIELDS,
        "layer_summary.csv": LAYER_SUMMARY_FIELDS,
    }
    combined = {filename: read_layer_rows(output_root, filename) for filename in file_fields}
    add_helix_trajectory_fields(combined["helix_segment_trajectories.csv"])
    add_bridge_trajectory_fields(combined["bridge_pair_trajectories.csv"])
    for filename, fields in file_fields.items():
        rows = combined[filename]
        rows.sort(key=lambda row: (int(row.get("layer", 0)), row.get("split", ""), row.get("protein_id", ""), int(row.get("segment_id", row.get("bridge_id", 0)) or 0)))
        atomic_write_csv(output_root / filename, fields, rows)

    skipped = read_csv(output_root / "skipped_proteins.csv") if (output_root / "skipped_proteins.csv").is_file() else []
    skipped = [row for row in skipped if not row["stage"].startswith("layer_")]
    for layer in ANALYSIS_LAYERS:
        skipped.extend(read_csv(output_root / "layer_shards" / f"layer_{layer:02d}" / "skipped_proteins.csv"))
    deduplicated_skipped = {
        tuple(row.get(field, "") for field in SKIPPED_FIELDS): row for row in skipped
    }
    atomic_write_csv(
        output_root / "skipped_proteins.csv", SKIPPED_FIELDS,
        [deduplicated_skipped[key] for key in sorted(deduplicated_skipped)],
    )

    validation = read_csv(output_root / "validation.csv")
    validation = [row for row in validation if row["category"] != "coefficient_replay"]
    per_rows = combined["per_protein.csv"]
    for layer in ANALYSIS_LAYERS:
        selected = [row for row in per_rows if int(row["layer"]) == layer]
        differences = np.asarray([abs(float(row["replay_r2_difference"])) for row in selected], dtype=np.float64)
        validation.append(
            {"category": "coefficient_replay", "metric": "max_abs_per_protein_r2_difference",
             "split": ",".join(sorted({row["split"] for row in selected})), "orientation": "",
             "value": float(np.max(differences)) if differences.size else math.nan, "n": len(selected),
             "threshold": REPLAY_R2_TOLERANCE,
             "status": "pass" if differences.size and np.max(differences) <= REPLAY_R2_TOLERANCE else "fail",
             "details": f"layer={layer}; raw-scale coefficient replay versus saved per_protein.csv"}
        )
    atomic_write_csv(output_root / "validation.csv", VALIDATION_FIELDS, validation)
    make_plots(output_root, combined["layer_summary.csv"])

    summary_rows = [row for row in combined["layer_summary.csv"] if row["split"] == "test"]
    dssp_rows = read_csv(output_root / "dssp_validation.csv")
    failure_rows = read_csv(output_root / "dssp_failures.csv")
    text = [
        "Secondary-structure emergence diagnostic",
        "========================================",
        "",
        f"Generated: {now_timestamp()}",
        f"Variant: {VARIANT} ({FEATURE_DIM} directional features)",
        f"Layers: {', '.join(map(str, ANALYSIS_LAYERS))}",
        "Primary evaluation: frozen held-out test proteins",
        "Uncertainty: descriptive point estimates only; no bootstrap",
        f"DSSP validation rows: {len(dssp_rows)}; suspect/failure rows: {len(failure_rows)}",
        "",
        "Held-out layer summary:",
    ]
    for row in sorted(summary_rows, key=lambda item: int(item["layer"])):
        text.append(
            f"  layer {int(row['layer']):02d}: helix/strand AUROC={float(row['helix_strand_auroc']):.4f}; "
            f"helix score={float(row['mean_helix_score']):.4f}; strand score={float(row['mean_strand_score']):.4f}; "
            f"sheet random margin={float(row['sheet_all_matched_random_margin']):.3f} Å; "
            f"register ±1 margin={float(row['sheet_all_shift_1_margin']):.3f} Å"
        )
    text.extend(
        [
            "", "Interpretation safeguards:",
            "- Helix visibility requires both the fixed relative score gate and absolute centroid-closeness gate.",
            "- Sheet conclusions use within-layer contrast margins; the fixed distance threshold is only for trajectory crossings.",
            "- validation.csv contains true-distance register ceilings and coefficient-replay parity checks.",
            "- Negative decoded distances are retained and counted rather than clipped.",
            "", f"Runtime for consolidation: {time.time() - started:.1f}s",
        ]
    )
    atomic_write_text(output_root / "summary.txt", "\n".join(text) + "\n")
    atomic_write_text(output_root / "consolidation_complete.json", json.dumps({"status": "complete", "layers": ANALYSIS_LAYERS, "runtime_seconds": round(time.time() - started, 3)}, indent=2) + "\n")
    log(f"Consolidation complete in {time.time() - started:.1f}s; outputs={output_root}")


def main() -> None:
    stage = os.environ.get("SUBSTRUCTURE_STAGE", "").strip().lower()
    if stage == "prepare":
        prepare_stage()
    elif stage == "layer":
        value = os.environ.get("SUBSTRUCTURE_LAYER")
        if value is None:
            raise ValueError("SUBSTRUCTURE_LAYER is required for layer stage")
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

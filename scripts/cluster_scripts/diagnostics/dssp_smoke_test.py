#!/usr/bin/env python3
"""Run DSSP once on a final predicted PDB and save inspectable annotations.

This is intentionally a one-protein smoke test.  DSSP assigns per-residue
secondary structure (including helices, strands, and beta bridges), but it
does not itself assign disulfide bonds.  The latter are reported here as
geometric CYS SG--SG candidates from the predicted coordinates.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
import shutil
import subprocess
import tempfile
import time
from collections import Counter
from pathlib import Path

import Bio
from Bio.Data.IUPACData import protein_letters_3to1
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import make_dssp_dict


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]

# Change this constant or set DSSP_SAMPLE_PROTEIN in the sbatch file to inspect
# a different final predicted structure.
DEFAULT_PROTEIN_ID = "6tf4_A"
DEFAULT_CAMEO_PDB_ROOT = Path("/work/hdd/bekh/thayes/openfold_reps_outputs_sm")
DEFAULT_ECOD_PDB_ROOT = Path(
    "/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "proteins" / "multi_protein_results" / "diagnostics" / "dssp_smoke_test"

HELIX_CODES = frozenset(("H", "G", "I"))
BETA_CODES = frozenset(("E", "B"))
DISULFIDE_SG_SG_CUTOFF_ANGSTROM = 2.3


def log(message: str) -> None:
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def atomic_write_text(path: Path, text: str) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(text)
    temporary.replace(path)


def atomic_write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def is_ecod_id(protein_id: str) -> bool:
    return protein_id.isdigit()


def resolve_predicted_pdb(protein_id: str) -> Path:
    """Resolve a local or Delta final-predicted PDB for one protein."""
    filename = f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    local_root = PROJECT_ROOT.parent / "protein_data"
    root_name = "ECOD_PDB_ROOT" if is_ecod_id(protein_id) else "CAMEO_PDB_ROOT"
    default_root = DEFAULT_ECOD_PDB_ROOT if is_ecod_id(protein_id) else DEFAULT_CAMEO_PDB_ROOT
    cluster_root = Path(os.environ.get(root_name, str(default_root))).expanduser()
    candidates = (
        local_root / protein_id / "predicted_pdb" / filename,
        cluster_root / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47" / "predictions" / filename,
        cluster_root / protein_id / "predicted_pdb" / filename,
    )
    for candidate in candidates:
        if candidate.is_file():
            return candidate.resolve()
    checked = "\n  ".join(str(candidate) for candidate in candidates)
    raise FileNotFoundError(f"Final predicted PDB for {protein_id} was not found. Checked:\n  {checked}")


def make_dssp_input(source: Path) -> Path:
    """Add a synthetic CRYST1 record when needed, then return a temporary PDB.

    Some predicted OpenFold PDBs omit CRYST1.  Adding a large P1 box is the
    compatibility workaround used by the existing DSSP annotation pipeline.
    """
    raw = source.read_text()
    descriptor, temporary_name = tempfile.mkstemp(
        prefix="dssp_smoke_", suffix=".pdb", dir=os.environ.get("TMPDIR")
    )
    with os.fdopen(descriptor, "w") as handle:
        if not any(line.startswith("CRYST1") for line in raw.splitlines()):
            handle.write("CRYST1  999.000  999.000  999.000  90.00  90.00  90.00 P 1           1\n")
        handle.write(raw)
    return Path(temporary_name)


def run_dssp(source: Path, executable: Path, output_path: Path) -> dict:
    """Run mkdssp and save raw output; raise an informative error on failure."""
    temporary_input = make_dssp_input(source)
    try:
        result = subprocess.run(
            [str(executable), "--output-format=dssp", str(temporary_input)],
            check=False,
            text=True,
            capture_output=True,
        )
    finally:
        temporary_input.unlink(missing_ok=True)

    atomic_write_text(output_path, result.stdout)
    atomic_write_text(output_path.with_name("mkdssp.stderr.txt"), result.stderr)
    if result.returncode != 0:
        raise RuntimeError(
            f"mkdssp exited with status {result.returncode}; see "
            f"{output_path.with_name('mkdssp.stderr.txt')}"
        )
    if not result.stdout.strip():
        raise RuntimeError("mkdssp returned success but produced no DSSP output")
    records, keys = make_dssp_dict(str(output_path))
    return {"records": records, "keys": keys}


def residue_id(chain_id: str, residue) -> dict:
    hetflag, resseq, insertion_code = residue.get_id()
    return {
        "chain": chain_id,
        "pdb_resseq": int(resseq),
        "insertion_code": insertion_code.strip(),
        "hetflag": hetflag.strip(),
    }


def parse_polymer_residues(pdb_path: Path) -> list[dict]:
    model = PDBParser(QUIET=True).get_structure("predicted", str(pdb_path))[0]
    residues = []
    standard_residues = {name.upper() for name in protein_letters_3to1}
    for chain in model:
        for residue in chain:
            if residue.get_id()[0] != " " or residue.get_resname() not in standard_residues:
                continue
            if "CA" not in residue:
                continue
            residues.append({"residue": residue, **residue_id(chain.id, residue)})
    if not residues:
        raise ValueError(f"No standard amino-acid residues with CA atoms found in {pdb_path}")
    return residues


def secondary_structure_class(code: str) -> str:
    if code in HELIX_CODES:
        return "helix"
    if code == "E":
        return "beta_strand"
    if code == "B":
        return "beta_bridge"
    if code == "T":
        return "turn"
    if code == "S":
        return "bend"
    return "coil"


def dssp_key(record: dict) -> tuple:
    return (record["chain"], (" ", record["pdb_resseq"], record["insertion_code"] or " "))


def build_residue_rows(residues: list[dict], dssp: dict) -> list[dict]:
    rows = []
    for index, record in enumerate(residues):
        value = dssp.get(dssp_key(record))
        code = value[1] if value else ""
        rows.append(
            {
                "residue_index": index,
                "chain": record["chain"],
                "pdb_resseq": record["pdb_resseq"],
                "insertion_code": record["insertion_code"],
                "resname": record["residue"].get_resname(),
                "dssp_mapped": int(value is not None),
                "dssp_index": value[5] if value else "",
                "dssp_code": code or "-",
                "secondary_structure": secondary_structure_class(code),
                "absolute_sasa": value[2] if value else "",
                "phi": value[3] if value else "",
                "psi": value[4] if value else "",
            }
        )
    return rows


def build_segments(residue_rows: list[dict]) -> list[dict]:
    """Collapse contiguous DSSP assignments into helix/strand/bridge segments."""
    segments = []
    active = None
    for row in residue_rows:
        segment_type = row["secondary_structure"]
        if segment_type not in {"helix", "beta_strand", "beta_bridge"}:
            if active:
                segments.append(active)
                active = None
            continue
        same_segment = active and active["segment_type"] == segment_type and active["chain"] == row["chain"] and active["end_residue_index"] == row["residue_index"] - 1
        if not same_segment:
            if active:
                segments.append(active)
            active = {
                "segment_type": segment_type,
                "chain": row["chain"],
                "start_residue_index": row["residue_index"],
                "start_pdb_resseq": row["pdb_resseq"],
                "start_insertion_code": row["insertion_code"],
                "end_residue_index": row["residue_index"],
                "end_pdb_resseq": row["pdb_resseq"],
                "end_insertion_code": row["insertion_code"],
                "n_residues": 1,
            }
        else:
            active["end_residue_index"] = row["residue_index"]
            active["end_pdb_resseq"] = row["pdb_resseq"]
            active["end_insertion_code"] = row["insertion_code"]
            active["n_residues"] += 1
    if active:
        segments.append(active)
    return segments


def find_disulfide_candidates(residues: list[dict]) -> list[dict]:
    """Find CYS SG--SG contacts consistent with a disulfide geometry.

    A predicted PDB normally has no chemistry/oxidation information, so these
    coordinate-based calls are candidates, not experimentally confirmed bonds.
    """
    cysteines = [record for record in residues if record["residue"].get_resname() == "CYS" and "SG" in record["residue"]]
    candidates = []
    for left_index, left in enumerate(cysteines):
        for right in cysteines[left_index + 1 :]:
            distance = float(left["residue"]["SG"] - right["residue"]["SG"])
            if distance <= DISULFIDE_SG_SG_CUTOFF_ANGSTROM:
                candidates.append(
                    {
                        "chain_i": left["chain"],
                        "pdb_resseq_i": left["pdb_resseq"],
                        "insertion_code_i": left["insertion_code"],
                        "chain_j": right["chain"],
                        "pdb_resseq_j": right["pdb_resseq"],
                        "insertion_code_j": right["insertion_code"],
                        "sg_sg_distance_angstrom": distance,
                        "cutoff_angstrom": DISULFIDE_SG_SG_CUTOFF_ANGSTROM,
                        "interpretation": "coordinate_based_candidate_not_confirmed_bond",
                    }
                )
    return candidates


def count_ssbond_records(pdb_path: Path) -> int:
    return sum(line.startswith("SSBOND") for line in pdb_path.read_text().splitlines())


def main() -> None:
    started = time.time()
    protein_id = os.environ.get("DSSP_SAMPLE_PROTEIN", DEFAULT_PROTEIN_ID)
    output_root = Path(os.environ.get("DSSP_SMOKE_OUTPUT_DIR", str(DEFAULT_OUTPUT_ROOT))).expanduser() / protein_id
    output_root.mkdir(parents=True, exist_ok=True)
    executable = Path(os.environ.get("DSSP_EXECUTABLE", "mkdssp")).expanduser()
    if shutil.which(str(executable)) is None and not executable.is_file():
        raise FileNotFoundError(
            f"DSSP executable {executable} is unavailable. Run setup_dssp_delta.sbatch "
            "first, then set DSSP_EXECUTABLE to $HOME/dssp_env/bin/mkdssp."
        )

    pdb_path = resolve_predicted_pdb(protein_id)
    log(f"protein={protein_id}; final predicted PDB={pdb_path}")
    version = subprocess.run([str(executable), "--version"], check=True, text=True, capture_output=True).stdout.strip()
    log(f"mkdssp={version}")

    raw_dssp_path = output_root / "raw.dssp"
    dssp_result = run_dssp(pdb_path, executable, raw_dssp_path)
    residues = parse_polymer_residues(pdb_path)
    residue_rows = build_residue_rows(residues, dssp_result["records"])
    segments = build_segments(residue_rows)
    disulfides = find_disulfide_candidates(residues)
    secondary_counts = Counter(row["secondary_structure"] for row in residue_rows)

    atomic_write_csv(output_root / "residues.csv", list(residue_rows[0]), residue_rows)
    segment_fields = [
        "segment_type", "chain", "start_residue_index", "start_pdb_resseq",
        "start_insertion_code", "end_residue_index", "end_pdb_resseq",
        "end_insertion_code", "n_residues",
    ]
    atomic_write_csv(output_root / "secondary_structure_segments.csv", segment_fields, segments)
    disulfide_fields = [
        "chain_i", "pdb_resseq_i", "insertion_code_i", "chain_j", "pdb_resseq_j",
        "insertion_code_j", "sg_sg_distance_angstrom", "cutoff_angstrom", "interpretation",
    ]
    atomic_write_csv(output_root / "disulfide_candidates.csv", disulfide_fields, disulfides)

    summary = {
        "status": "success",
        "protein_id": protein_id,
        "pdb_path": str(pdb_path),
        "pdb_sha256": hashlib.sha256(pdb_path.read_bytes()).hexdigest(),
        "dssp_executable": str(executable),
        "dssp_version": version,
        "biopython_version": Bio.__version__,
        "n_polymer_residues": len(residues),
        "n_dssp_records": len(dssp_result["records"]),
        "n_residues_mapped_to_dssp": sum(row["dssp_mapped"] for row in residue_rows),
        "secondary_structure_residue_counts": dict(sorted(secondary_counts.items())),
        "n_helix_segments": sum(segment["segment_type"] == "helix" for segment in segments),
        "n_beta_strand_segments": sum(segment["segment_type"] == "beta_strand" for segment in segments),
        "n_beta_bridge_segments": sum(segment["segment_type"] == "beta_bridge" for segment in segments),
        "n_cysteines_with_sg": sum(record["residue"].get_resname() == "CYS" and "SG" in record["residue"] for record in residues),
        "n_coordinate_based_disulfide_candidates": len(disulfides),
        "disulfide_sg_sg_cutoff_angstrom": DISULFIDE_SG_SG_CUTOFF_ANGSTROM,
        "n_ssbond_records_in_pdb": count_ssbond_records(pdb_path),
        "raw_dssp_file": str(raw_dssp_path),
        "runtime_seconds": round(time.time() - started, 3),
    }
    atomic_write_text(output_root / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n")
    log(
        "success: "
        f"mapped={summary['n_residues_mapped_to_dssp']}/{summary['n_polymer_residues']}; "
        f"helices={summary['n_helix_segments']}; "
        f"beta_strands={summary['n_beta_strand_segments']}; "
        f"disulfide_candidates={len(disulfides)}"
    )
    log(f"outputs={output_root}")


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        log(f"FAILED: {type(error).__name__}: {error}")
        raise

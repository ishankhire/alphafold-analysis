#!/usr/bin/env python3
"""Annotate structural residue-pair interactions for the first 50 CAMEO proteins.

This intentionally stops before representation matching or model fitting.  It
uses the saved unrelaxed OpenFold PDB as the structural source of truth.
"""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import os
import shutil
import subprocess
import sys
import tempfile
import time
from collections import Counter
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from Bio.PDB.DSSP import make_dssp_dict
from Bio.Data.PDBData import residue_sasa_scales
from scipy.spatial.distance import cdist

ROOT_DIR = Path(__file__).resolve().parents[3]
ANNOTATION_VERSION = "cameo50_v1"
CAMEO50 = """6tf4_A 6wmk_A 6wqc_A 6zpp_A 7ab3_E 7ar0_B 7atr_A 7au7_A 7awk_A 7b0d_A
7b1k_B 7b1w_F 7b26_C 7b28_F 7b29_A 7b2a_A 7b2o_A 7b3a_A 7b4q_A 7b7t_A
7bbz_A 7bcz_A 7bew_A 7bhy_A 7cjs_B 7dcm_A 7dfe_A 7djy_A 7dk9_A 7dkk_A
7dko_A 7dmf_A 7dms_A 7dnu_A 7don_B 7dq9_A 7dru_C 7dtp_A 7dup_A 7dut_A
7dvn_A 7e0m_A 7e2v_A 7e37_A 7e3z_A 7ebt_B 7ee3_C 7ef6_A 7eqx_A 7esx_A""".split()
STANDARD = frozenset("ALA ARG ASN ASP CYS GLN GLU GLY HIS ILE LEU LYS MET PHE PRO SER THR TRP TYR VAL".split())
HYDROPHOBIC = frozenset("ALA VAL ILE LEU MET PHE TRP TYR".split())
CHARGED = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2"), "LYS": ("NZ",), "ARG": ("NH1", "NH2")}
ACIDIC, BASIC = frozenset(("ASP", "GLU")), frozenset(("LYS", "ARG"))
SANDER = residue_sasa_scales["Sander"]

PAIR_FIELDS = """protein_id source annotation_version config_hash pair_id n_residues i j chain_i chain_j pdb_resseq_i pdb_resseq_j insertion_code_i insertion_code_j resname_i resname_j same_chain sequence_separation ca_distance min_heavy_atom_distance min_sidechain_heavy_atom_distance charged_atom_distance dssp_code_i dssp_code_j dssp_3state_i dssp_3state_j absolute_sasa_i absolute_sasa_j relative_sasa_i relative_sasa_j hbond_i_donor_j hbond_j_donor_i hbond_i_donor_j_energy hbond_j_donor_i_energy best_hbond_energy salt_bridge salt_bridge_valid backbone_hbond backbone_hbond_valid beta_bridge beta_bridge_valid backbone_hbond_or_beta_bridge hydrophobic_contact hydrophobic_contact_valid hydrophobic_core_contact hydrophobic_core_contact_valid exclusion_reason""".split()
RESIDUE_FIELDS = """protein_id source annotation_version config_hash residue_index chain pdb_resseq insertion_code resname backbone_complete sidechain_complete dssp_mapped dssp_index dssp_code dssp_3state absolute_sasa relative_sasa phi psi dssp_failure_reason""".split()


def log(message):
    print(f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}", flush=True)


def null(value):
    return "" if value is None else value


def three_state(code):
    return "helix" if code in "HGI" else "beta" if code in "EB" else "coil"


def configuration():
    return {"annotation_version": ANNOTATION_VERSION, "proteins": CAMEO50,
            "pair_blocks_root": os.environ.get("PAIR_BLOCKS_ROOT", "/work/hdd/bekh/thayes/openfold_reps"),
            "pdb_root": os.environ.get("PDB_ROOT", "/work/hdd/bekh/thayes/openfold_reps_outputs_sm"),
            "salt_bridge": {"acidic_atoms": {k: CHARGED[k] for k in ACIDIC}, "basic_atoms": {k: CHARGED[k] for k in BASIC}, "cutoff_angstrom": 4.0},
            "hydrophobic_residues": sorted(HYDROPHOBIC), "hydrophobic_contact_cutoff_angstrom": 4.5,
            "hydrophobic_core_relative_sasa_cutoff": 0.20, "dssp_energy_cutoff_kcal_mol": -0.5,
            "sasa_scale": "Biopython Sander", "pair_block_layer": 0, "pair_block_channels": 128}


def config_hash(config):
    return hashlib.sha256(json.dumps(config, sort_keys=True).encode()).hexdigest()


def pdb_path(root, protein):
    return Path(root) / f"my_outputs_align_{protein}_representations_load_from_saved_layer_47" / "predictions" / f"{protein}_model_1_ptm_unrelaxed.pdb"


def pair_path(root, protein):
    return Path(root) / protein / f"{protein}_pair_block_0.npy"


def atom_coords(residue, names=None):
    result = []
    for atom in residue.get_atoms():
        if (names is None or atom.get_name() in names) and atom.element.strip().upper() not in {"H", "D"}:
            result.append(np.asarray(atom.coord, dtype=np.float64))
    return np.asarray(result, dtype=np.float64).reshape((-1, 3))


def parse_residues(path):
    model = PDBParser(QUIET=True).get_structure("predicted", str(path))[0]
    chains = []
    for chain in model:
        polymer = [r for r in chain if r.get_id()[0] == " " and r.get_resname() in STANDARD and "CA" in r]
        if polymer:
            chains.append((chain, polymer))
    if len(chains) != 1:
        raise ValueError(f"expected exactly one polymer chain, found {len(chains)}")
    chain, residues = chains[0]
    records = []
    for residue in residues:
        name = residue.get_resname()
        side = atom_coords(residue, {a.get_name() for a in residue.get_atoms() if a.get_name() not in {"N", "CA", "C", "O", "OXT"}})
        records.append({"chain": chain.id, "resseq": residue.id[1], "icode": residue.id[2].strip(), "resname": name,
                        "ca": np.asarray(residue["CA"].coord, dtype=np.float64), "heavy": atom_coords(residue), "side": side,
                        "charged": atom_coords(residue, CHARGED.get(name, ())),
                        "backbone_complete": int(all(a in residue for a in ("N", "CA", "C", "O"))),
                        "sidechain_complete": int(name == "GLY" or len(side) > 0)})
    return records


def prepare_dssp_input(source):
    raw = source.read_text()
    has_cryst1 = any(line.startswith("CRYST1") for line in raw.splitlines())
    fd, name = tempfile.mkstemp(prefix="dssp_", suffix=".pdb", dir=os.environ.get("TMPDIR"))
    with os.fdopen(fd, "w") as handle:
        if not has_cryst1:
            handle.write("CRYST1  999.000  999.000  999.000  90.00  90.00  90.00 P 1           1\n")
        handle.write(raw)
    return Path(name)


def run_dssp(source, executable, output):
    temporary = prepare_dssp_input(source)
    try:
        result = subprocess.run([str(executable), "--output-format=dssp", str(temporary)], check=True, text=True, capture_output=True)
        output.write_text(result.stdout)
        records, _ = make_dssp_dict(str(output))
        index_to_key, bp = {}, {}
        for line in result.stdout.splitlines():
            if len(line) >= 34 and line[:5].strip().isdigit() and line[9] != " ":
                key = (line[11], (" ", int(line[5:10]), line[10]))
                index_to_key[int(line[:5])] = key
                bp[key] = tuple(int(x) for x in (line[25:29], line[29:33]) if x.strip() and int(x) > 0)
        return records, index_to_key, bp, None
    finally:
        temporary.unlink(missing_ok=True)


def min_distance(first, second):
    if not len(first) or not len(second):
        return None
    return float(cdist(first, second).min())


def pair_labels(a, b, dssp, index_to_record, bridges):
    acidic_basic = (a["resname"] in ACIDIC and b["resname"] in BASIC) or (b["resname"] in ACIDIC and a["resname"] in BASIC)
    charged_distance = min_distance(a["charged"], b["charged"]) if acidic_basic else None
    salt_valid = int(not acidic_basic or charged_distance is not None)
    salt = int(charged_distance <= 4.0) if acidic_basic and charged_distance is not None else (0 if not acidic_basic else None)
    hydro_candidate = a["resname"] in HYDROPHOBIC and b["resname"] in HYDROPHOBIC
    side_distance = min_distance(a["side"], b["side"]) if hydro_candidate else None
    hydro_valid = int(not hydro_candidate or side_distance is not None)
    hydro = int(side_distance <= 4.5) if hydro_candidate and side_distance is not None else (0 if not hydro_candidate else None)
    key_a, key_b = a["dssp_key"], b["dssp_key"]
    dssp_valid = int(dssp is not None and key_a in dssp and key_b in dssp)
    h_ij = h_ji = e_ij = e_ji = beta = None
    if dssp_valid:
        da, db = dssp[key_a], dssp[key_b]
        # DSSP NH->O entries are donor at this residue and acceptor at relative index.
        donor_a = [(da[6], da[7]), (da[10], da[11])]
        donor_b = [(db[6], db[7]), (db[10], db[11])]
        energies_ij = [float(energy) for relative, energy in donor_a if index_to_record.get(da[5] + relative) == key_b]
        energies_ji = [float(energy) for relative, energy in donor_b if index_to_record.get(db[5] + relative) == key_a]
        e_ij, e_ji = min(energies_ij, default=None), min(energies_ji, default=None)
        h_ij, h_ji = int(e_ij is not None and e_ij < -0.5), int(e_ji is not None and e_ji < -0.5)
        beta = int(dssp[key_b][5] in bridges.get(key_a, ()) or dssp[key_a][5] in bridges.get(key_b, ()))
    rsasa_ok = dssp_valid and a["relative_sasa"] is not None and b["relative_sasa"] is not None
    core_valid = int(not hydro_candidate or (hydro_valid and rsasa_ok))
    core = int(hydro == 1 and a["relative_sasa"] <= .2 and b["relative_sasa"] <= .2) if core_valid and hydro_candidate else (0 if not hydro_candidate else None)
    return {"charged_atom_distance": charged_distance, "salt_bridge": salt, "salt_bridge_valid": salt_valid,
            "min_sidechain_heavy_atom_distance": side_distance, "hydrophobic_contact": hydro, "hydrophobic_contact_valid": hydro_valid,
            "hydrophobic_core_contact": core, "hydrophobic_core_contact_valid": core_valid,
            "backbone_hbond": (int(h_ij or h_ji) if dssp_valid else None), "backbone_hbond_valid": dssp_valid,
            "beta_bridge": beta, "beta_bridge_valid": dssp_valid,
            "backbone_hbond_or_beta_bridge": (int(h_ij or h_ji or beta) if dssp_valid else None),
            "hbond_i_donor_j": h_ij, "hbond_j_donor_i": h_ji, "hbond_i_donor_j_energy": e_ij, "hbond_j_donor_i_energy": e_ji,
            "best_hbond_energy": min((x for x in (e_ij, e_ji) if x is not None), default=None)}


def annotate(protein, config, digest, out_root):
    pair_root, pdb_root = config["pair_blocks_root"], config["pdb_root"]
    pdb, block = pdb_path(pdb_root, protein), pair_path(pair_root, protein)
    manifest = {"protein_id": protein, "source": "CAMEO", "pdb_path": str(pdb), "pair_block_path": str(block), "config_hash": digest}
    if not pdb.exists() or not block.exists():
        manifest.update(status="excluded", failure_reason="missing_pdb_or_pair_block"); return manifest, [], []
    try:
        residues = parse_residues(pdb); shape = tuple(np.load(block, mmap_mode="r").shape)
        if len(shape) != 3 or shape[0] != shape[1] or shape[2] != 128 or shape[0] != len(residues):
            raise ValueError(f"pair_block_shape={shape}, residue_count={len(residues)}")
    except Exception as exc:
        manifest.update(status="excluded", failure_reason=str(exc)); return manifest, [], []
    dssp_file = out_root / "annotations/dssp_raw" / f"{protein}.dssp.tmp"
    dssp = indices = bridges = None; dssp_failure = None
    try:
        dssp, indices, bridges, _ = run_dssp(pdb, os.environ["DSSP_EXECUTABLE"], dssp_file)
    except Exception as exc:
        dssp_failure = str(exc); dssp_file.unlink(missing_ok=True)
    for record in residues:
        record["dssp_key"] = (record["chain"], (" ", record["resseq"], record["icode"] or " "))
        value = dssp.get(record["dssp_key"]) if dssp else None
        record.update(dssp_index=value[5] if value else None, dssp_code=value[1] if value else None,
                      dssp_3state=three_state(value[1]) if value else None, absolute_sasa=value[2] if value else None,
                      relative_sasa=min(1.0, value[2] / SANDER[record["resname"]]) if value and record["resname"] in SANDER else None,
                      phi=value[3] if value else None, psi=value[4] if value else None)
    per_res = out_root / "annotations/per_residue" / f"{protein}.csv.gz.tmp"
    per_pair = out_root / "annotations/per_protein" / f"{protein}.csv.gz.tmp"
    counts, exclusions = Counter(), []
    with gzip.open(per_res, "wt", newline="") as handle, gzip.open(per_pair, "wt", newline="") as pairs:
        rw, pw = csv.DictWriter(handle, RESIDUE_FIELDS), csv.DictWriter(pairs, PAIR_FIELDS); rw.writeheader(); pw.writeheader()
        for i, r in enumerate(residues):
            rw.writerow({"protein_id": protein, "source": "CAMEO", "annotation_version": ANNOTATION_VERSION, "config_hash": digest, "residue_index": i,
                         **{k: null(r.get(k)) for k in RESIDUE_FIELDS if k not in {"protein_id", "source", "annotation_version", "config_hash", "residue_index", "dssp_mapped", "dssp_failure_reason"}}, "dssp_mapped": int(r["dssp_index"] is not None), "dssp_failure_reason": dssp_failure or ""})
        pair_id = 0
        for i, a in enumerate(residues[:-1]):
            for j, b in enumerate(residues[i + 1:], i + 1):
                labels = pair_labels(a, b, dssp, indices or {}, bridges or {})
                row = {"protein_id": protein, "source": "CAMEO", "annotation_version": ANNOTATION_VERSION, "config_hash": digest, "pair_id": pair_id, "n_residues": len(residues), "i": i, "j": j,
                       "chain_i": a["chain"], "chain_j": b["chain"], "pdb_resseq_i": a["resseq"], "pdb_resseq_j": b["resseq"], "insertion_code_i": a["icode"], "insertion_code_j": b["icode"], "resname_i": a["resname"], "resname_j": b["resname"], "same_chain": 1, "sequence_separation": j-i,
                       "ca_distance": float(np.linalg.norm(a["ca"]-b["ca"])), "min_heavy_atom_distance": min_distance(a["heavy"], b["heavy"]),
                       "dssp_code_i": a["dssp_code"], "dssp_code_j": b["dssp_code"], "dssp_3state_i": a["dssp_3state"], "dssp_3state_j": b["dssp_3state"], "absolute_sasa_i": a["absolute_sasa"], "absolute_sasa_j": b["absolute_sasa"], "relative_sasa_i": a["relative_sasa"], "relative_sasa_j": b["relative_sasa"], "exclusion_reason": dssp_failure or "", **labels}
                pw.writerow({k: null(row.get(k)) for k in PAIR_FIELDS}); pair_id += 1
                for name in ("salt_bridge", "backbone_hbond", "beta_bridge", "hydrophobic_contact", "hydrophobic_core_contact"):
                    counts[(name, "invalid" if not labels[f"{name}_valid"] else "positive" if labels[name] else "valid_negative")] += 1
    expected = len(residues)*(len(residues)-1)//2
    if pair_id != expected: raise RuntimeError("pair row validation failed")
    per_res.rename(per_res.with_suffix("")); per_pair.rename(per_pair.with_suffix(""))
    if dssp and dssp_file.exists(): dssp_file.rename(dssp_file.with_suffix(""))
    manifest.update(status="success", failure_reason=dssp_failure or "", residue_count=len(residues), pair_block_shape="x".join(map(str, shape)), expected_pair_count=expected,
                    pdb_sha256=hashlib.sha256(pdb.read_bytes()).hexdigest())
    return manifest, [{"protein_id": protein, "interaction_type": k[0], "count_type": k[1], "count": v} for k, v in counts.items()], exclusions


def main():
    started = time.time(); config, output = configuration(), Path(os.environ.get("CSV_ROOT", str(Path.home()/"csv_files/pair_interaction_classification/cameo50_v1")))
    executable = Path(os.environ["DSSP_EXECUTABLE"])
    try:
        config["dssp_version"] = subprocess.run([str(executable), "--version"], check=True, text=True, capture_output=True).stdout.strip()
    except Exception as exc:
        raise RuntimeError(f"DSSP preflight failed for {executable}: {exc}") from exc
    digest = config_hash(config); output.mkdir(parents=True, exist_ok=True)
    for directory in ("annotations/per_residue", "annotations/per_protein", "annotations/dssp_raw"):
        (output / directory).mkdir(parents=True, exist_ok=True)
    config_file = output / "annotation_config.json"
    if config_file.exists() and json.loads(config_file.read_text()).get("config_hash") != digest: raise RuntimeError("output configuration hash differs; refusing to mix definitions")
    config_file.write_text(json.dumps({**config, "config_hash": digest, "dssp_executable": str(executable)}, indent=2, sort_keys=True) + "\n")
    prior = {}
    if (output / "protein_manifest.csv").exists():
        with open(output / "protein_manifest.csv", newline="") as handle:
            prior = {row["protein_id"]: row for row in csv.DictReader(handle)}
    manifests=[]
    for protein in CAMEO50:
        existing = output / "annotations/per_protein" / f"{protein}.csv.gz"
        if existing.exists():
            log(f"{protein}: checkpoint exists; skipping")
            manifests.append(prior.get(protein, {"protein_id": protein, "source": "CAMEO", "status": "success", "failure_reason": "", "config_hash": digest}))
            continue
        log(f"{protein}: annotation started"); t=time.time(); manifest, _, _ = annotate(protein, config, digest, output); manifests.append(manifest); log(f"{protein}: {manifest['status']} in {time.time()-t:.1f}s")
    manifest_fields = sorted({k for row in manifests for k in row} | {"protein_id", "status", "failure_reason", "config_hash"})
    with open(output / "protein_manifest.csv", "w", newline="") as h: w=csv.DictWriter(h, manifest_fields); w.writeheader(); w.writerows(manifests)
    counts = Counter()
    for protein in CAMEO50:
        finished = output / "annotations/per_protein" / f"{protein}.csv.gz"
        if not finished.exists(): continue
        with gzip.open(finished, "rt", newline="") as handle:
            for row in csv.DictReader(handle):
                for name in ("salt_bridge", "backbone_hbond", "beta_bridge", "hydrophobic_contact", "hydrophobic_core_contact"):
                    counts[(protein, name, "invalid" if row[f"{name}_valid"] != "1" else "positive" if row[name] == "1" else "valid_negative")] += 1
    with open(output / "annotations/annotation_counts.csv", "w", newline="") as h:
        w=csv.DictWriter(h, ["protein_id","interaction_type","count_type","count"]); w.writeheader()
        w.writerows({"protein_id": k[0], "interaction_type": k[1], "count_type": k[2], "count": v} for k, v in sorted(counts.items()))
    with open(output / "annotations/exclusions.csv", "w", newline="") as h:
        w=csv.DictWriter(h, ["protein_id","entity_type","entity_identifier","affected_annotation","reason"]); w.writeheader()
        for m in manifests:
            if m["status"] != "success":
                w.writerow({"protein_id":m["protein_id"],"entity_type":"protein","entity_identifier":m["protein_id"],"affected_annotation":"all","reason":m["failure_reason"]})
            elif m.get("failure_reason"):
                w.writerow({"protein_id":m["protein_id"],"entity_type":"protein","entity_identifier":m["protein_id"],"affected_annotation":"dssp,backbone_hbond,beta_bridge,relative_sasa,hydrophobic_core_contact","reason":m["failure_reason"]})
    success=[m for m in manifests if m["status"] == "success"]
    summary = [f"requested_proteins={len(CAMEO50)}", f"successful_proteins={len(success)}", f"excluded_proteins={len(manifests)-len(success)}",
               f"total_residues={sum(int(m.get('residue_count') or 0) for m in success)}", f"total_pairs={sum(int(m.get('expected_pair_count') or 0) for m in success)}",
               f"annotation_version={ANNOTATION_VERSION}", f"dssp_version={config['dssp_version']}", f"configuration_hash={digest}", f"runtime_seconds={time.time()-started:.1f}"]
    summary.extend(f"{protein}.{interaction}.{kind}={count}" for (protein, interaction, kind), count in sorted(counts.items()))
    summary.extend(f"failure.{m['protein_id']}={m['failure_reason']}" for m in manifests if m.get("failure_reason"))
    (output / "summary.txt").write_text("\n".join(summary) + "\n")
    log("annotation complete")


if __name__ == "__main__": main()

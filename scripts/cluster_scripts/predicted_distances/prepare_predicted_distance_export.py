#!/usr/bin/env python3
"""Validate frozen models and prepare shared residue metadata; never fit a model."""
from __future__ import annotations

import subprocess
import time
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser

import predicted_distance_export_common as common


from canonical_storage import deployment_revision


def residue_metadata(outer, pair_entry):
    pdb_path = common.replay.resolve_combined_pdb_path(outer.protein_id, outer.source).resolve()
    before = common.sha256_file(pdb_path)
    ca = common.replay.extract_ca_coordinates(pdb_path)
    coords = np.asarray(ca["coordinates"], dtype=np.float64)
    if coords.shape != (outer.n_residues, 3) or not np.all(np.isfinite(coords)):
        raise ValueError(f"Frozen PDB alignment/finiteness mismatch: {outer.protein_id}")
    residues = []
    for chain in PDBParser(QUIET=True).get_structure("predicted", str(pdb_path))[0]:
        for residue in chain:
            if residue.id[0] == "W" or "CA" not in residue:
                continue
            residues.append((str(chain.id), str(residue.resname), int(residue.id[1]), str(residue.id[2]).strip()))
    if [(chain, name, number) for chain, name, number, _ in residues] != ca["residues"]:
        raise ValueError("Residue metadata order differs from canonical C-alpha extraction")
    if before != common.sha256_file(pdb_path):
        raise ValueError("PDB changed during preparation")
    protein = dict(protein_id=outer.protein_id, source=outer.source, outer_fold=outer.outer_fold,
                   n_residues=outer.n_residues, n_pairs=outer.n_pairs,
                   pair_assignment_digest=pair_entry.assignment_digest,
                   pdb_path=str(pdb_path), pdb_sha256=before)
    rows = [dict(protein_id=outer.protein_id, residue_index=i, chain_id=chain,
                 pdb_resseq=number, insertion_code=insertion, residue_name=name,
                 pdb_path=str(pdb_path), pdb_sha256=before)
            for i, (chain, name, number, insertion) in enumerate(residues)]
    return protein, rows


def prepare():
    started = time.perf_counter()
    root = common.output_root()
    with common.task_lock(root, "prepare"):
        bundle = common.load_bundle()
        if (root / "run_manifest.json").exists():
            common.load_preparation(root, bundle)
            common.log("compatible preparation already exists")
            return
        if (root / "layers").exists():
            raise ValueError("Existing prediction layers have no preparation manifest")
        common.log("preparation: validating complete frozen model inventories")
        identity = common.source_identity(bundle, verify_artifacts=True)
        proteins, residues = [], []
        for pid in sorted(bundle.eligible_protein_ids):
            protein, rows = residue_metadata(bundle.outer_by_protein[pid], bundle.pair_by_protein[pid])
            proteins.append(protein)
            residues.extend(rows)
        common.atomic_write_csv(root / "metadata/proteins.csv", common.PROTEIN_COLUMNS, proteins)
        common.atomic_write_csv(root / "metadata/residue_index.csv", common.RESIDUE_COLUMNS, residues)
        readme = common.readme_source().read_text()
        common.atomic_write_text(root / "README.md", readme)
        revision = deployment_revision(Path(__file__).parent)
        manifest = {
            "export_schema": common.EXPORT_SCHEMA, "preparation_status": "complete",
            "run_status": "prepared", "created_at": common.utc_now(), "completed_layers": [],
            "identity": identity, "identity_digest": common.semantic_digest(identity),
            "metadata_sha256": {name: common.sha256_file(root / name) for name in (
                "metadata/proteins.csv", "metadata/residue_index.csv",
            )}, "code_revision": revision,
            "preparation_elapsed_seconds": time.perf_counter() - started,
            "preparation_peak_rss_mb": common.peak_rss_mb(),
        }
        common.atomic_write_json(root / "run_manifest.json", manifest)
        common.log(f"preparation complete: proteins={len(proteins)} elapsed={time.perf_counter() - started:.2f}s root={root}")


if __name__ == "__main__":
    common.entrypoint(prepare)

#!/usr/bin/env python3
"""Rescore completed packed OOF distances without loading or fitting models."""
from __future__ import annotations

import argparse
import fcntl
import os
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from Bio.PDB import PDBParser

DIRECTORY = Path(__file__).resolve().parent
CLUSTER_DIRECTORY = DIRECTORY.parent
for directory in (DIRECTORY, CLUSTER_DIRECTORY, CLUSTER_DIRECTORY / "predicted_distances",
                  CLUSTER_DIRECTORY / "cross_validation"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import predicted_distance_export_common as export  # noqa: E402
from cluster_regression_helpers import (  # noqa: E402
    TM_SCORE_METHOD, classical_mds, compute_mds_tm_score_alignment, compute_tm_score,
    superimpose_structures, tmscore_provenance,
)
from cross_validation_common import (  # noqa: E402
    GEOMETRY_METRICS_COLUMNS, GEOMETRY_SUMMARY_COLUMNS, summarize_geometry_rows,
)

LAYERS = tuple(range(48))
EXPECTED_PROTEINS = 552
SCORE_COLUMNS = export.IDENTITY_COLUMNS + (
    "n_residues", "tm_score", "tm_score_kabsch", "tm_score_saved",
    "delta_tm_score_saved", "delta_tm_score_alignment", "array_sha256",
    "tm_score_method", "status",
)
RUNTIME_COLUMNS = ("layer", "started_at", "elapsed_seconds", "peak_rss_mb", "n_scores", "status")
PDB_COLUMNS = ("protein_id", "analysis_name", "layer", "n_residues", "pdb_path",
               "pdb_sha256", "array_sha256", "tm_score", "status")
PDB_SCHEMA = "tm_score_aligned_ca_pdb_v1"


from canonical_storage import assert_writable, result_root, workflow_root, normalize_shared_tree


def read_rows(path, columns=None):
    return export.read_csv(path, columns)


def load_context(predictions_root, regression_root):
    """Validate frozen export provenance, independently of current exporter code."""
    predictions_root, regression_root = Path(predictions_root).resolve(), Path(regression_root).resolve()
    run = export.load_json(predictions_root / "run_manifest.json")
    if (run.get("export_schema") != export.EXPORT_SCHEMA or run.get("run_status") != "complete"
            or run.get("preparation_status") != "complete"
            or run.get("completed_layers") != list(LAYERS)):
        raise ValueError("Rescoring requires the finalized, complete predicted-distance export")
    if export.semantic_digest(run["identity"]) != run["identity_digest"]:
        raise ValueError("Export identity digest mismatch")
    for relative, checksum in run["metadata_sha256"].items():
        if export.sha256_file(export.checked_path(predictions_root, relative)) != checksum:
            raise ValueError(f"Export metadata checksum mismatch: {relative}")
    proteins = read_rows(predictions_root / "metadata/proteins.csv", export.PROTEIN_COLUMNS)
    by_id = {row["protein_id"]: row for row in proteins}
    if (len(proteins) != EXPECTED_PROTEINS or len(by_id) != len(proteins)
            or sorted(by_id) != sorted(run["identity"]["protein_ids"])):
        raise ValueError("Export protein membership mismatch")
    residues = {pid: [] for pid in by_id}
    for row in read_rows(predictions_root / "metadata/residue_index.csv", export.RESIDUE_COLUMNS):
        residues[row["protein_id"]].append(row)
    for pid, protein in by_id.items():
        n = int(protein["n_residues"])
        if int(protein["n_pairs"]) != n * (n - 1) // 2 or n < 3:
            raise ValueError(f"Invalid residue/pair count: {pid}")
        if [int(row["residue_index"]) for row in residues[pid]] != list(range(n)):
            raise ValueError(f"Incomplete or reordered residue metadata: {pid}")
        if any(row["pdb_sha256"] != protein["pdb_sha256"] or row["pdb_path"] != protein["pdb_path"]
               for row in residues[pid]):
            raise ValueError(f"Inconsistent reference provenance: {pid}")
    geometry, inputs = {}, {
        "export_manifest_sha256": export.sha256_file(predictions_root / "run_manifest.json"),
        "pdb_export_schema": PDB_SCHEMA,
        "predictions_root": str(predictions_root), "regression_root": str(regression_root),
        "tm_score_backend": tmscore_provenance(), "source_geometry_sha256": {},
        "code_sha256": {
            "recompute_tm_scores.py": export.sha256_file(Path(__file__)),
            "cluster_regression_helpers.py": export.sha256_file(
                Path(sys.modules["cluster_regression_helpers"].__file__)),
            "predicted_distance_export_common.py": export.sha256_file(Path(export.__file__)),
            "cross_validation_common.py": export.sha256_file(Path(sys.modules["cross_validation_common"].__file__)),
        },
    }
    expected = {(layer, pid) for layer in LAYERS for pid in by_id}
    for spec in export.FAMILY_SPECS:
        path = regression_root / spec.family_dir / "geometry_metrics.csv"
        rows = read_rows(path, GEOMETRY_METRICS_COLUMNS)
        indexed = {(int(row["layer"]), row["protein_id"]): row for row in rows}
        if len(indexed) != len(rows) or set(indexed) != expected:
            raise ValueError(f"Source geometry coverage mismatch: {spec.analysis_name}")
        for (layer, pid), row in indexed.items():
            protein = by_id[pid]
            if (row["analysis_name"] != spec.analysis_name or row["geometry_status"] != "complete"
                    or row["evaluation_scope"] != "oof_all_pairs"
                    or int(row["n_residues"]) != int(protein["n_residues"])
                    or int(row["n_protein_pairs"]) != int(protein["n_pairs"])
                    or row["source"] != protein["source"]
                    or int(row["outer_fold"]) != int(protein["outer_fold"])
                    or float(row["pair_coverage"]) != 1
                    or not np.isfinite(float(row["tm_score"]))):
                raise ValueError(f"Invalid source geometry: {spec.analysis_name}/{layer}/{pid}")
        geometry[spec.analysis_name] = indexed
        inputs["source_geometry_sha256"][spec.analysis_name] = export.sha256_file(path)
    return SimpleNamespace(root=predictions_root, run=run, proteins=by_id, residues=residues,
                           geometry=geometry, inputs=inputs, digest=export.semantic_digest(inputs))


@contextmanager
def preparation_lock(output_root):
    """Serialize short manifest updates; concurrent layer starts must wait."""
    assert_writable(output_root)
    directory = Path(output_root) / ".locks"
    directory.mkdir(parents=True, exist_ok=True)
    with (directory / "prepare.lock").open("a") as handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        try:
            yield
        finally:
            fcntl.flock(handle, fcntl.LOCK_UN)


def ensure_output(output_root, context):
    assert_writable(output_root)
    output_root = Path(output_root).resolve()
    predictions_root = context.root
    regression_root = Path(context.inputs["regression_root"])
    protected = [predictions_root] + [regression_root / spec.family_dir for spec in export.FAMILY_SPECS]
    if (output_root == regression_root or regression_root.is_relative_to(output_root)
            or any(output_root.is_relative_to(root) or root.is_relative_to(output_root) for root in protected)):
        raise ValueError("Output root must be separate from prediction and source family trees")
    with preparation_lock(output_root):
        path = output_root / "run_manifest.json"
        if path.exists():
            if export.load_json(path).get("identity_digest") != context.digest:
                raise ValueError("Rescoring inputs/backend/code changed; use a new or explicitly cleared output root")
        else:
            export.atomic_write_json(path, {
                "tm_score_method": TM_SCORE_METHOD, "run_status": "prepared",
                "pdb_export_schema": PDB_SCHEMA,
                "identity": context.inputs, "identity_digest": context.digest,
                "created_at": export.utc_now(), "layers": list(LAYERS),
                "n_proteins": len(context.proteins), "evaluation_scope": "oof_all_pairs",
            })


def read_reference(context, pid):
    protein = context.proteins[pid]
    path = Path(protein["pdb_path"])
    if export.sha256_file(path) != protein["pdb_sha256"]:
        raise ValueError(f"Reference PDB checksum mismatch: {pid}")
    atoms, identities = [], []
    for chain in PDBParser(QUIET=True).get_structure(pid, str(path))[0]:
        for residue in chain:
            if residue.id[0] == "W" or "CA" not in residue:
                continue
            atoms.append(residue["CA"].get_coord())
            identities.append((str(chain.id), int(residue.id[1]), str(residue.id[2]).strip(), residue.resname))
    expected = [(row["chain_id"], int(row["pdb_resseq"]), row["insertion_code"], row["residue_name"])
                for row in context.residues[pid]]
    coordinates = np.asarray(atoms, dtype=np.float64)
    if (identities != expected or coordinates.shape != (int(protein["n_residues"]), 3)
            or not np.all(np.isfinite(coordinates))
            or export.sha256_file(path) != protein["pdb_sha256"]):
        raise ValueError(f"Reference residue identity/coordinate mismatch: {pid}")
    return coordinates


def load_layer(context, layer):
    directory = context.root / "layers" / f"layer_{layer:02d}"
    path = directory / "layer_manifest.json"
    if export.sha256_file(path) != context.run["layer_manifest_sha256"][str(layer)]:
        raise ValueError("Export layer manifest checksum mismatch")
    manifest = export.load_json(path)
    if (manifest["status"] != "complete" or manifest["layer"] != layer
            or manifest["identity_digest"] != context.run["identity_digest"]
            or set(manifest["records"]) != set(context.proteins)):
        raise ValueError("Export layer coverage/identity mismatch")
    for name, checksum in manifest["metadata_sha256"].items():
        if export.sha256_file(export.checked_path(directory, name)) != checksum:
            raise ValueError(f"Export layer metadata changed: {name}")
    return manifest


def score_distances(reference, values):
    n = len(reference)
    i, j = np.tril_indices(n, k=-1)
    matrix = np.zeros((n, n), dtype=np.float64)
    matrix[i, j] = matrix[j, i] = np.asarray(values, dtype=np.float64)
    np.maximum(matrix, 0, out=matrix)
    coordinates, _ = classical_mds(matrix, n_dims=3)
    reflected = coordinates.copy()
    reflected[:, 2] *= -1
    original_fit, original_rmsd = superimpose_structures(reference, coordinates)
    reflected_fit, reflected_rmsd = superimpose_structures(reference, reflected)
    kabsch = reflected_fit if reflected_rmsd < original_rmsd else original_fit
    score, aligned = compute_mds_tm_score_alignment(reference, coordinates, n)
    return score, compute_tm_score(reference, kabsch, n), aligned


def reconstruction_path(layer, pid, family):
    return f"layers/layer_{layer:02d}/pdb_files/{family}/{pid}_reconstructed_ca.pdb"


def write_reconstruction(output_root, context, layer, pid, family, coordinates, score, array_sha256):
    """Atomically retain a CA trace with the frozen reference residue identity."""
    residues = context.residues[pid]
    coordinates = np.asarray(coordinates, dtype=np.float64)
    if coordinates.shape != (len(residues), 3) or not np.all(np.isfinite(coordinates)):
        raise ValueError("Reconstructed PDB requires matching finite coordinates")
    relative = reconstruction_path(layer, pid, family)
    path = export.checked_path(output_root, relative)
    lines = ["REMARK CA-only MDS reconstruction; winning TM-score alignment\n",
             f"REMARK protein_id={pid} analysis_name={family} layer={layer}\n",
             f"REMARK tm_score_float64={score:.17g}; coordinates rounded to 0.001 Angstrom\n"]
    previous_chain = None
    for serial, (residue, point) in enumerate(zip(residues, coordinates), start=1):
        chain, insertion, name = residue["chain_id"], residue["insertion_code"], residue["residue_name"]
        number = int(residue["pdb_resseq"])
        fields = [f"{value:8.3f}" for value in point]
        if (len(chain) != 1 or len(insertion) > 1 or len(name) != 3
                or not -999 <= number <= 9999 or serial > 99999
                or any(len(field) != 8 for field in fields)):
            raise ValueError(f"Reconstruction exceeds standard PDB fields: {pid}")
        if previous_chain is not None and previous_chain != chain:
            lines.append("TER\n")
        lines.append(f"ATOM  {serial:5d}  CA  {name:3s} {chain}{number:4d}{insertion:1s}   "
                     f"{''.join(fields)}  1.00  0.00           C\n")
        previous_chain = chain
    lines.extend(("TER\n", "END\n"))
    path.parent.mkdir(parents=True, exist_ok=True)
    export.atomic_write_text(path, "".join(lines))
    return dict(protein_id=pid, analysis_name=family, layer=layer, n_residues=len(residues),
                pdb_path=relative, pdb_sha256=export.sha256_file(path),
                array_sha256=array_sha256, tm_score=score, status="complete")


def validate_pdb_rows(pdb_rows, score_rows, output_root):
    """Validate index identity and return whether every physical artifact is intact."""
    indexed = {(row["protein_id"], row["analysis_name"], int(row["layer"])): row for row in pdb_rows}
    scores = {(row["protein_id"], row["analysis_name"], int(row["layer"])): row for row in score_rows}
    if len(indexed) != len(pdb_rows) or set(indexed) != set(scores):
        raise ValueError("PDB index coverage/uniqueness mismatch")
    intact = True
    for (pid, family, layer), row in indexed.items():
        score = scores[pid, family, layer]
        if (set(row) != set(PDB_COLUMNS) or row["status"] != "complete"
                or row["pdb_path"] != reconstruction_path(layer, pid, family)
                or int(row["n_residues"]) != int(score["n_residues"])
                or row["array_sha256"] != score["array_sha256"]
                or float(row["tm_score"]) != float(score["tm_score"])):
            raise ValueError("PDB index identity/score mismatch")
        path = export.checked_path(output_root, row["pdb_path"])
        if not path.is_file() or export.sha256_file(path) != row["pdb_sha256"]:
            intact = False
    return intact


def invalidate_completion(output_root, directory=None):
    """A repair must not leave old completion claims visible after interruption."""
    with preparation_lock(output_root):
        path = output_root / "run_manifest.json"
        run = export.load_json(path)
        run["run_status"] = "prepared"
        for field in ("completed_at", "n_scores", "n_pdbs", "output_sha256", "layer_manifest_sha256"):
            run.pop(field, None)
        export.atomic_write_json(path, run)
        if directory is not None:
            (directory / "layer_manifest.json").unlink(missing_ok=True)


def validate_score_rows(rows, context, layer, protein_ids):
    expected = {(pid, spec.analysis_name) for pid in protein_ids for spec in export.FAMILY_SPECS}
    keys = {(row["protein_id"], row["analysis_name"]) for row in rows}
    if len(keys) != len(rows) or keys != expected:
        raise ValueError("Score coverage/uniqueness mismatch")
    for row in rows:
        source = context.geometry[row["analysis_name"]][(layer, row["protein_id"])]
        if (set(row) != set(SCORE_COLUMNS) or row["tm_score_method"] != TM_SCORE_METHOD
                or row["status"] != "complete" or int(row["layer"]) != layer
                or any(str(row[key]) != str(source[key]) for key in export.IDENTITY_COLUMNS)
                or int(row["n_residues"]) != int(source["n_residues"])):
            raise ValueError("Score row identity/status mismatch")
        score, baseline, saved = (float(row[name]) for name in ("tm_score", "tm_score_kabsch", "tm_score_saved"))
        if (not all(np.isfinite(value) and 0 <= value <= 1 for value in (score, baseline, saved))
                or score < baseline - 1e-12 or saved != float(source["tm_score"])
                or not np.isclose(float(row["delta_tm_score_saved"]), score - saved, atol=1e-14, rtol=0)
                or not np.isclose(float(row["delta_tm_score_alignment"]), score - baseline, atol=1e-14, rtol=0)):
            raise ValueError("Invalid score/baseline/difference")


def run_layer(context, output_root, layer):
    assert_writable(output_root)
    output_root = Path(output_root).resolve()
    if layer not in LAYERS:
        raise ValueError("Unknown layer")
    ensure_output(output_root, context)
    with export.task_lock(output_root, f"layer_{layer:02d}"):
        started, timestamp = time.perf_counter(), export.utc_now()
        manifest = load_layer(context, layer)
        directory = output_root / "layers" / f"layer_{layer:02d}"
        completed = directory / "layer_manifest.json"
        previous = None
        if completed.exists():
            previous = export.load_json(completed)
            if (previous["identity_digest"] != context.digest or previous["status"] != "complete"
                    or previous["layer"] != layer
                    or export.sha256_file(directory / "tm_scores.csv") != previous["tm_scores_sha256"]
                    or export.sha256_file(directory / "pdb_index.csv") != previous["pdb_index_sha256"]
                    or previous["n_pdbs"] != len(context.proteins) * len(export.FAMILY_SPECS)):
                raise ValueError("Completed scoring layer changed")
        rows, pdb_rows = [], []
        regenerated = False
        for number, pid in enumerate(sorted(context.proteins), start=1):
            reference = read_reference(context, pid)
            protein = context.proteins[pid]
            arrays = []
            for spec in export.FAMILY_SPECS:
                index = manifest["records"][pid]["families"][spec.analysis_name]["index"]
                source = context.geometry[spec.analysis_name][(layer, pid)]
                if any(str(index[key]) != str(source[key]) for key in export.IDENTITY_COLUMNS):
                    raise ValueError(f"Prediction/geometry identity mismatch: {pid}/{spec.analysis_name}")
                values = export.check_array(context.root, index, SimpleNamespace(
                    protein_id=pid, n_pairs=int(protein["n_pairs"])), spec, layer)
                arrays.append((spec, index, values))
            checkpoint = directory / "progress" / f"{pid}.json"
            checksums = {spec.analysis_name: index["array_sha256"] for spec, index, _ in arrays}
            reusable = False
            if checkpoint.exists():
                progress = export.load_json(checkpoint)
                if progress["identity_digest"] != context.digest or progress["array_sha256"] != checksums:
                    raise ValueError(f"Checkpoint provenance changed: {pid}")
                protein_rows = progress["rows"]
                validate_score_rows(protein_rows, context, layer, [pid])
                if any(row["array_sha256"] != checksums[row["analysis_name"]] for row in protein_rows):
                    raise ValueError("Checkpoint prediction checksum mismatch")
                protein_pdb_rows = progress["pdb_rows"]
                reusable = validate_pdb_rows(protein_pdb_rows, protein_rows, output_root)
            if not reusable:
                if not regenerated:
                    invalidate_completion(output_root, directory)
                    regenerated = True
                protein_rows, protein_pdb_rows = [], []
                for spec, index, values in arrays:
                    score, baseline, aligned = score_distances(reference, values)
                    source = context.geometry[spec.analysis_name][(layer, pid)]
                    saved = float(source["tm_score"])
                    protein_rows.append({
                        **{key: source[key] for key in export.IDENTITY_COLUMNS},
                        "n_residues": len(reference), "tm_score": score,
                        "tm_score_kabsch": baseline, "tm_score_saved": saved,
                        "delta_tm_score_saved": score - saved, "delta_tm_score_alignment": score - baseline,
                        "array_sha256": index["array_sha256"], "tm_score_method": TM_SCORE_METHOD,
                        "status": "complete",
                    })
                    protein_pdb_rows.append(write_reconstruction(
                        output_root, context, layer, pid, spec.analysis_name, aligned,
                        score, index["array_sha256"]))
                validate_score_rows(protein_rows, context, layer, [pid])
                export.atomic_write_json(checkpoint, {"identity_digest": context.digest,
                    "array_sha256": checksums, "rows": protein_rows, "pdb_rows": protein_pdb_rows})
            rows.extend(protein_rows)
            pdb_rows.extend(protein_pdb_rows)
            export.log(f"layer={layer} proteins={number}/{len(context.proteins)} elapsed={time.perf_counter()-started:.2f}s")
        validate_score_rows(rows, context, layer, context.proteins)
        if previous is not None and not regenerated:
            if (read_rows(directory / "tm_scores.csv", SCORE_COLUMNS) != export.read_csv_rows(SCORE_COLUMNS, rows)
                    or read_rows(directory / "pdb_index.csv", PDB_COLUMNS) != export.read_csv_rows(PDB_COLUMNS, pdb_rows)):
                raise ValueError("Completed scoring layer changed")
            export.log(f"layer={layer} completed checkpoint verified; retaining original runtime")
            return
        export.atomic_write_csv(directory / "tm_scores.csv", SCORE_COLUMNS, rows)
        export.atomic_write_csv(directory / "pdb_index.csv", PDB_COLUMNS, pdb_rows)
        runtime = dict(layer=layer, started_at=timestamp, elapsed_seconds=time.perf_counter()-started,
                       peak_rss_mb=export.peak_rss_mb(), n_scores=len(rows), status="complete")
        export.atomic_write_json(directory / "layer_manifest.json", {
            "identity_digest": context.digest, "status": "complete", "layer": layer,
            "tm_scores_sha256": export.sha256_file(directory / "tm_scores.csv"), "runtime": runtime,
            "pdb_index_sha256": export.sha256_file(directory / "pdb_index.csv"), "n_pdbs": len(pdb_rows),
        })
        export.log(f"layer={layer} complete: {len(rows)} scores")


def finalize(context, output_root):
    assert_writable(output_root)
    output_root = Path(output_root).resolve()
    ensure_output(output_root, context)
    with export.task_lock(output_root, "finalize"):
        # Invalidate a previous finalization before checking possibly damaged artifacts.
        invalidate_completion(output_root)
        started = time.perf_counter()
        rows, runtimes, pdb_rows, layer_hashes = [], [], [], {}
        for layer in LAYERS:
            source = load_layer(context, layer)
            directory = output_root / "layers" / f"layer_{layer:02d}"
            manifest = export.load_json(directory / "layer_manifest.json")
            if (manifest["identity_digest"] != context.digest or manifest["status"] != "complete"
                    or manifest["layer"] != layer
                    or export.sha256_file(directory / "tm_scores.csv") != manifest["tm_scores_sha256"]
                    or export.sha256_file(directory / "pdb_index.csv") != manifest["pdb_index_sha256"]):
                raise ValueError("Incomplete or changed scoring layer")
            layer_rows = read_rows(directory / "tm_scores.csv", SCORE_COLUMNS)
            validate_score_rows(layer_rows, context, layer, context.proteins)
            layer_pdb_rows = read_rows(directory / "pdb_index.csv", PDB_COLUMNS)
            if (manifest["n_pdbs"] != len(layer_rows)
                    or not validate_pdb_rows(layer_pdb_rows, layer_rows, output_root)):
                raise ValueError(f"Missing or changed reconstructed PDBs: layer={layer}")
            for row in layer_rows:
                pid, family = row["protein_id"], row["analysis_name"]
                index = source["records"][pid]["families"][family]["index"]
                if row["array_sha256"] != index["array_sha256"]:
                    raise ValueError("Scored prediction differs from export manifest")
                spec = next(spec for spec in export.FAMILY_SPECS if spec.analysis_name == family)
                export.check_array(context.root, index, SimpleNamespace(
                    protein_id=pid, n_pairs=int(context.proteins[pid]["n_pairs"])), spec, layer)
            rows.extend(layer_rows)
            pdb_rows.extend(layer_pdb_rows)
            runtimes.append(manifest["runtime"])
            layer_hashes[str(layer)] = export.sha256_file(directory / "layer_manifest.json")
            export.log(f"finalize layer={layer} verified {len(layer_pdb_rows)} PDBs elapsed={time.perf_counter()-started:.2f}s")
        for pid in context.proteins:
            read_reference(context, pid)
        export.atomic_write_csv(output_root / "tm_scores.csv", SCORE_COLUMNS, rows)
        export.atomic_write_csv(output_root / "pdb_index.csv", PDB_COLUMNS, pdb_rows)
        export.atomic_write_csv(output_root / "runtime.csv", RUNTIME_COLUMNS, runtimes)
        output_hashes = {}
        for spec in export.FAMILY_SPECS:
            family_rows = []
            for row in rows:
                if row["analysis_name"] == spec.analysis_name:
                    original = context.geometry[spec.analysis_name][(int(row["layer"]), row["protein_id"])]
                    family_rows.append({**original, "tm_score": float(row["tm_score"])})
            # CSV strings are preserved in the replacement table; summaries need
            # numeric identity fields to reproduce the canonical grouping.
            numeric = [{**row, **{key: int(row[key]) for key in ("feature_dim", "layer")}}
                       for row in family_rows]
            directory = output_root / spec.family_dir
            export.atomic_write_csv(directory / "geometry_metrics.csv", GEOMETRY_METRICS_COLUMNS, family_rows)
            export.atomic_write_csv(directory / "geometry_summary.csv", GEOMETRY_SUMMARY_COLUMNS,
                                    summarize_geometry_rows(numeric))
            export.atomic_write_json(directory / "run_manifest.json", {
                "tm_score_method": TM_SCORE_METHOD, "identity_digest": context.digest,
                "analysis_name": spec.analysis_name, "run_status": "complete",
                "source_geometry_sha256": context.inputs["source_geometry_sha256"][spec.analysis_name],
            })
            for name in ("geometry_metrics.csv", "geometry_summary.csv", "run_manifest.json"):
                relative = f"{spec.family_dir}/{name}"
                output_hashes[relative] = export.sha256_file(output_root / relative)
        for name in ("tm_scores.csv", "runtime.csv", "pdb_index.csv"):
            output_hashes[name] = export.sha256_file(output_root / name)
        run = export.load_json(output_root / "run_manifest.json")
        run.update(run_status="complete", completed_at=export.utc_now(), n_scores=len(rows),
                   n_pdbs=len(pdb_rows), output_sha256=output_hashes,
                   layer_manifest_sha256=layer_hashes)
        export.atomic_write_json(output_root / "run_manifest.json", run)
        export.log(f"finalized {len(rows)} optimized scores and {len(pdb_rows)} PDBs: "
                   f"{output_root} elapsed={time.perf_counter()-started:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--layer", type=int)
    mode.add_argument("--finalize", action="store_true")
    parser.add_argument("--predictions-root", type=Path, default=export.output_root())
    parser.add_argument("--regression-root", type=Path,
                        default=result_root())
    parser.add_argument("--output-root", type=Path, default=workflow_root("tm_score_optimized_v1", "TMSCORE_OUTPUT_ROOT"))
    args = parser.parse_args()
    output_root = args.output_root.expanduser().resolve()
    assert_writable(output_root)
    context = load_context(args.predictions_root, args.regression_root)
    if args.finalize:
        finalize(context, output_root)
        normalize_shared_tree(output_root)
    else:
        run_layer(context, output_root, args.layer)


if __name__ == "__main__":
    export.entrypoint(main)

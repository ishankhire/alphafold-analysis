#!/usr/bin/env python3
"""Consolidate completed TM-score layers without reopening individual arrays/PDBs."""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import recompute_tm_scores as worker

export = worker.export
VALIDATION_MODE = "consolidation_only"


from canonical_storage import assert_writable, result_root, workflow_root, normalize_shared_tree


def validate_pdb_index(pdb_rows, score_rows):
    """Check recorded artifact identities without inspecting the physical PDBs."""
    scores = {(row["protein_id"], row["analysis_name"]): row for row in score_rows}
    keys = [(row["protein_id"], row["analysis_name"]) for row in pdb_rows]
    if len(keys) != len(set(keys)) or set(keys) != set(scores):
        raise ValueError("PDB index coverage/uniqueness mismatch")
    for row in pdb_rows:
        pid, family, layer = row["protein_id"], row["analysis_name"], int(row["layer"])
        score = scores[pid, family]
        checksum = row["pdb_sha256"]
        if (set(row) != set(worker.PDB_COLUMNS) or row["status"] != "complete"
                or layer != int(score["layer"])
                or row["pdb_path"] != worker.reconstruction_path(layer, pid, family)
                or int(row["n_residues"]) != int(score["n_residues"])
                or row["array_sha256"] != score["array_sha256"]
                or float(row["tm_score"]) != float(score["tm_score"])
                or len(checksum) != 64 or any(char not in "0123456789abcdef" for char in checksum)):
            raise ValueError("PDB index identity/score/checksum mismatch")


def consolidate(context, output_root):
    assert_writable(output_root)
    output_root = Path(output_root).resolve()
    # The unchanged scoring worker verifies its original inputs/code identity.
    worker.ensure_output(output_root, context)
    with export.task_lock(output_root, "finalize"):
        worker.invalidate_completion(output_root)
        started, timestamp = time.perf_counter(), export.utc_now()
        validation = {
            "mode": VALIDATION_MODE,
            "layer_tables": "checksums_and_coverage_verified",
            "prediction_arrays": "not_revalidated",
            "reconstructed_pdbs": "not_revalidated",
            "reference_pdbs": "not_revalidated",
        }
        rows, pdb_rows, runtimes, layer_hashes = [], [], [], {}
        for layer in worker.LAYERS:
            source = worker.load_layer(context, layer)
            directory = output_root / "layers" / f"layer_{layer:02d}"
            path = directory / "layer_manifest.json"
            manifest = export.load_json(path)
            if (manifest["identity_digest"] != context.digest or manifest["status"] != "complete"
                    or manifest["layer"] != layer
                    or export.sha256_file(directory / "tm_scores.csv") != manifest["tm_scores_sha256"]
                    or export.sha256_file(directory / "pdb_index.csv") != manifest["pdb_index_sha256"]):
                raise ValueError(f"Incomplete or changed scoring layer: {layer}")
            layer_rows = worker.read_rows(directory / "tm_scores.csv", worker.SCORE_COLUMNS)
            worker.validate_score_rows(layer_rows, context, layer, context.proteins)
            layer_pdb_rows = worker.read_rows(directory / "pdb_index.csv", worker.PDB_COLUMNS)
            if manifest["n_pdbs"] != len(layer_rows):
                raise ValueError(f"PDB count mismatch: layer={layer}")
            validate_pdb_index(layer_pdb_rows, layer_rows)
            for row in layer_rows:
                index = source["records"][row["protein_id"]]["families"][row["analysis_name"]]["index"]
                if row["array_sha256"] != index["array_sha256"]:
                    raise ValueError("Scored prediction differs from export manifest")
            rows.extend(layer_rows)
            pdb_rows.extend(layer_pdb_rows)
            runtimes.append(manifest["runtime"])
            layer_hashes[str(layer)] = export.sha256_file(path)
            export.log(f"consolidate layer={layer} rows={len(layer_rows)} elapsed={time.perf_counter()-started:.2f}s")

        export.atomic_write_csv(output_root / "tm_scores.csv", worker.SCORE_COLUMNS, rows)
        export.atomic_write_csv(output_root / "pdb_index.csv", worker.PDB_COLUMNS, pdb_rows)
        export.atomic_write_csv(output_root / "runtime.csv", worker.RUNTIME_COLUMNS, runtimes)
        output_hashes = {}
        for spec in export.FAMILY_SPECS:
            family_rows = []
            for row in rows:
                if row["analysis_name"] == spec.analysis_name:
                    original = context.geometry[spec.analysis_name][(int(row["layer"]), row["protein_id"])]
                    family_rows.append({**original, "tm_score": float(row["tm_score"])})
            numeric = [{**row, **{key: int(row[key]) for key in ("feature_dim", "layer")}}
                       for row in family_rows]
            directory = output_root / spec.family_dir
            export.atomic_write_csv(directory / "geometry_metrics.csv", worker.GEOMETRY_METRICS_COLUMNS, family_rows)
            export.atomic_write_csv(directory / "geometry_summary.csv", worker.GEOMETRY_SUMMARY_COLUMNS,
                                    worker.summarize_geometry_rows(numeric))
            export.atomic_write_json(directory / "run_manifest.json", {
                "tm_score_method": worker.TM_SCORE_METHOD, "identity_digest": context.digest,
                "analysis_name": spec.analysis_name, "run_status": "complete",
                "source_geometry_sha256": context.inputs["source_geometry_sha256"][spec.analysis_name],
                "validation": validation,
            })
            for name in ("geometry_metrics.csv", "geometry_summary.csv", "run_manifest.json"):
                relative = f"{spec.family_dir}/{name}"
                output_hashes[relative] = export.sha256_file(output_root / relative)
        runtime = dict(started_at=timestamp, elapsed_seconds=time.perf_counter()-started,
                       peak_rss_mb=export.peak_rss_mb(), n_scores=len(rows), n_pdbs=len(pdb_rows),
                       validation_mode=VALIDATION_MODE, status="complete")
        export.atomic_write_json(output_root / "consolidation_runtime.json", runtime)
        for name in ("tm_scores.csv", "pdb_index.csv", "runtime.csv", "consolidation_runtime.json"):
            output_hashes[name] = export.sha256_file(output_root / name)
        run = export.load_json(output_root / "run_manifest.json")
        # Preserve the producer identity; record this distinct consumer separately.
        run.update(run_status="complete", completed_at=export.utc_now(), n_scores=len(rows),
                   n_pdbs=len(pdb_rows), output_sha256=output_hashes, layer_manifest_sha256=layer_hashes,
                   validation=validation, consolidation={
                       "script": Path(__file__).name, "code_sha256": export.sha256_file(Path(__file__)),
                       "runtime": runtime,
                   })
        export.atomic_write_json(output_root / "run_manifest.json", run)
        export.log(f"consolidated {len(rows)} scores and PDB index rows; bulk audit skipped; "
                   f"elapsed={time.perf_counter()-started:.2f}s")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--predictions-root", type=Path, default=export.output_root())
    parser.add_argument("--regression-root", type=Path,
                        default=result_root())
    parser.add_argument("--output-root", type=Path, default=workflow_root("tm_score_optimized_v1", "TMSCORE_OUTPUT_ROOT"))
    args = parser.parse_args()
    assert_writable(args.output_root)
    context = worker.load_context(args.predictions_root, args.regression_root)
    consolidate(context, args.output_root.expanduser())
    normalize_shared_tree(args.output_root)


if __name__ == "__main__":
    export.entrypoint(main)

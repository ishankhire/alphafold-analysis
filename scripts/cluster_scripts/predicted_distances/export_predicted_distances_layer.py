#!/usr/bin/env python3
"""Replay all four OOF probes for one layer, checkpointing each complete protein."""
from __future__ import annotations

import gc
import os
import time
import traceback

import numpy as np

import predicted_distance_export_common as common


def check_pair_input(outer, layer, record=None):
    path = common.replay.resolve_combined_pair_block_path(outer.protein_id, outer.source, layer).resolve()
    fingerprint = {"path": str(path), "sha256": common.sha256_file(path)}
    if record is not None and record != fingerprint:
        raise ValueError(f"Pair tensor changed since checkpoint: {outer.protein_id}/{layer}")
    return fingerprint


def write_progress(root, layer, progress):
    directory = root / ".checkpoints" / f"layer_{layer:02d}"
    common.atomic_write_json(directory / "progress.json", progress)
    # JSON snapshot is authoritative; these CSVs can be regenerated on resume.
    indexes = [v["index"] for r in progress["records"].values() for v in r["families"].values()]
    validation = [v["validation"] for r in progress["records"].values() for v in r["families"].values()]
    common.atomic_write_csv(directory / "prediction_index.csv", common.INDEX_COLUMNS, indexes)
    common.atomic_write_csv(directory / "validation.csv", common.VALIDATION_COLUMNS, validation)


def run_layer(layer: int):
    if layer not in common.LAYERS:
        raise ValueError("Layer must be 0..47 (SLURM_ARRAY_TASK_ID)")
    root = common.output_root()
    with common.task_lock(root, f"layer_{layer:02d}"):
        session_started = time.perf_counter()
        bundle = common.load_bundle()
        preparation, proteins = common.load_preparation(root, bundle)
        final_dir = root / "layers" / f"layer_{layer:02d}"
        checkpoint_dir = root / ".checkpoints" / f"layer_{layer:02d}"
        if (final_dir / "layer_manifest.json").exists():
            common.load_completed_layer(root, layer, bundle, preparation)
            common.log(f"layer={layer} completed export verified; skipping")
            return
        common.log(f"layer={layer} loading models")
        model_started = time.perf_counter()
        artifacts, provenance = common.replay.load_layer_artifacts(layer, bundle)
        for spec in common.FAMILY_SPECS:
            producer = provenance["producers"][spec.analysis_name]
            pinned = preparation["identity"]["producers"][spec.analysis_name]
            if (producer["run_manifest_sha256"] != pinned["files"]["run_manifest.json"]
                    or producer["replay_index_sha256"] != pinned["files"]["model_artifacts/replay_index.csv"]):
                raise ValueError("Loaded models disagree with prepared producers")
        common.log(f"layer={layer} model loading elapsed={time.perf_counter() - model_started:.2f}s")
        tables = {spec.analysis_name: common.producer_rows(spec, layer, bundle) for spec in common.FAMILY_SPECS}
        checkpoint_path = checkpoint_dir / "progress.json"
        progress = common.load_json(checkpoint_path) if checkpoint_path.exists() else {
            "layer": layer, "identity_digest": preparation["identity_digest"],
            "started_at": common.utc_now(), "elapsed_seconds": 0.0, "records": {},
        }
        if progress["layer"] != layer or progress["identity_digest"] != preparation["identity_digest"]:
            raise ValueError("Checkpoint configuration/provenance changed")
        common.validate_records(root, layer, progress["records"], bundle, preparation)
        for pid, record in progress["records"].items():
            check_pair_input(bundle.outer_by_protein[pid], layer, record["pair_input"])
        previous_elapsed = float(progress["elapsed_seconds"])
        active_protein = ""
        try:
            for pid in sorted(bundle.eligible_protein_ids):
                if pid in progress["records"]:
                    continue
                active_protein = pid
                outer = bundle.outer_by_protein[pid]
                protein_started = time.perf_counter()
                pair_input = check_pair_input(outer, layer)
                pdb_path = common.replay.resolve_combined_pdb_path(pid, outer.source).resolve()
                if str(pdb_path) != proteins[pid]["pdb_path"] or common.sha256_file(pdb_path) != proteins[pid]["pdb_sha256"]:
                    raise ValueError("Loaded PDB differs from prepared alignment")
                coords, chains, pair_i, pair_j, features = common.replay.load_protein(outer, layer)
                predictions, _ = common.replay.model_predictions(outer, chains, pair_i, pair_j, features, artifacts, bundle)
                if len(predictions) != len(common.FAMILY_SPECS) or {s.analysis_name for s, _ in predictions} != {s.analysis_name for s in common.FAMILY_SPECS}:
                    raise ValueError("Missing/duplicate predicted model family")
                targets = np.linalg.norm(coords[pair_i] - coords[pair_j], axis=1)
                # Validate every family before publishing any of this protein's files.
                entries = {}
                for spec, values in predictions:
                    validation, stats = common.validation_record(spec, outer, layer, targets, values, artifacts,
                                                                 tables[spec.analysis_name], preparation)
                    entries[spec.analysis_name] = {"validation": validation, "statistics": stats}
                check_pair_input(outer, layer, pair_input)
                for spec, values in predictions:
                    row = common.identity_row(spec, outer, layer, preparation)
                    row.update(common.atomic_save_prediction(root, layer, pid, spec, values))
                    entries[spec.analysis_name]["index"] = row
                progress["records"][pid] = {"pair_input": pair_input, "families": entries}
                progress["elapsed_seconds"] = previous_elapsed + time.perf_counter() - session_started
                progress.pop("last_failure", None)
                write_progress(root, layer, progress)
                common.log(f"layer={layer} protein={pid} complete={len(progress['records'])}/{len(proteins)} elapsed={time.perf_counter() - protein_started:.2f}s")
                del features, predictions, targets, coords, pair_i, pair_j, entries, values
                gc.collect()
            indexes, validation = common.validate_records(root, layer, progress["records"], bundle, preparation, require_complete=True)
            pooled = common.pooled_validation(layer, progress["records"], bundle)
            common.atomic_write_csv(final_dir / "prediction_index.csv", common.INDEX_COLUMNS, indexes)
            common.atomic_write_csv(final_dir / "validation.csv", common.VALIDATION_COLUMNS, validation)
            common.atomic_write_csv(final_dir / "pooled_validation.csv", common.POOLED_COLUMNS, pooled)
            runtime = dict(layer=layer, started_at=progress["started_at"], ended_at=common.utc_now(),
                           elapsed_seconds=previous_elapsed + time.perf_counter() - session_started,
                           n_proteins=len(proteins), n_arrays=len(indexes), output_bytes=sum(r["file_bytes"] for r in indexes),
                           peak_rss_mb=common.peak_rss_mb(), status="complete")
            common.atomic_write_json(final_dir / "runtime.json", runtime)
            layer_manifest = {"status": "complete", "layer": layer, "identity_digest": preparation["identity_digest"],
                              "records": progress["records"], "artifact_provenance": provenance,
                              "runtime": runtime, "metadata_sha256": {name: common.sha256_file(final_dir / name) for name in (
                                  "prediction_index.csv", "validation.csv", "pooled_validation.csv", "runtime.json",
                              )}}
            common.atomic_write_json(final_dir / "layer_manifest.json", layer_manifest)
            common.load_completed_layer(root, layer, bundle, preparation)
            # Only discard our own checkpoint after the final files were verified.
            for name in ("progress.json", "prediction_index.csv", "validation.csv"):
                (checkpoint_dir / name).unlink(missing_ok=True)
            for pid in progress["records"]:
                directory = checkpoint_dir / pid
                if directory.exists():
                    for path in directory.glob(".*.npy.tmp"):
                        path.unlink()
                    if not any(directory.iterdir()):
                        directory.rmdir()
            if checkpoint_dir.exists() and not any(checkpoint_dir.iterdir()):
                checkpoint_dir.rmdir()
            common.log(f"layer={layer} complete arrays={len(indexes)} elapsed={runtime['elapsed_seconds']:.2f}s")
        except Exception:
            progress["elapsed_seconds"] = previous_elapsed + time.perf_counter() - session_started
            progress["last_failure"] = {"protein_id": active_protein, "timestamp": common.utc_now(),
                                        "traceback": traceback.format_exc()}
            write_progress(root, layer, progress)
            raise


def main():
    run_layer(int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))


if __name__ == "__main__":
    common.entrypoint(main)

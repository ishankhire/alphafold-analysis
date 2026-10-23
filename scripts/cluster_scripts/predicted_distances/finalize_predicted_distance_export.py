#!/usr/bin/env python3
"""Verify every layer and consolidate CSV metadata; never concatenate NPY files."""
from __future__ import annotations

import os
import tempfile
import time
from pathlib import Path

import predicted_distance_export_common as common


from canonical_storage import normalize_shared_tree


def _finalize_locked(root):
    started = time.perf_counter()
    bundle = common.load_bundle()
    preparation, _ = common.load_preparation(root, bundle)
    if common.source_identity(bundle, verify_artifacts=True) != preparation["identity"]:
        raise ValueError("Producer identity changed during finalization")
    layers_dir = root / "layers"
    expected = {f"layer_{layer:02d}" for layer in common.LAYERS}
    if not layers_dir.is_dir() or {p.name for p in layers_dir.iterdir()} != expected:
        raise ValueError("Physical layer inventory is incomplete or contains unexpected entries")
    # Stream one layer at a time: no complete-cohort prediction or index matrix.
    with tempfile.TemporaryDirectory(prefix=".finalize-", dir=root) as temporary:
        staged = Path(temporary)
        handles = {}
        names = ("prediction_index.csv", "validation.csv", "pooled_validation.csv")
        runtime_rows, layer_hashes = [], {}
        try:
            handles = {name: (staged / name).open("w") for name in names}
            for position, layer in enumerate(common.LAYERS):
                with common.task_lock(root, f"layer_{layer:02d}"):
                    manifest, _, _ = common.load_completed_layer(root, layer, bundle, preparation)
                    pooled = common.pooled_validation(layer, manifest["records"], bundle)
                    directory = layers_dir / f"layer_{layer:02d}"
                    if common.read_csv(directory / "pooled_validation.csv", common.POOLED_COLUMNS) != common.read_csv_rows(common.POOLED_COLUMNS, pooled):
                        raise ValueError("Pooled validation differs from committed statistics")
                    for name in names:
                        with (directory / name).open() as source:
                            header = next(source)
                            if position == 0:
                                handles[name].write(header)
                            for line in source:
                                handles[name].write(line)
                    runtime_rows.append(manifest["runtime"])
                    layer_hashes[str(layer)] = common.sha256_file(directory / "layer_manifest.json")
                    common.log(f"finalization layer={layer} verified elapsed={time.perf_counter() - started:.2f}s")
            for handle in handles.values():
                handle.flush()
                os.fsync(handle.fileno())
        finally:
            for handle in handles.values():
                handle.close()
        common.atomic_write_csv(staged / "runtime.csv", common.RUNTIME_COLUMNS, runtime_rows)
        for name in (*names, "runtime.csv"):
            os.replace(staged / name, root / name)
    preparation.update(run_status="complete", completed_layers=list(common.LAYERS),
                       finalized_at=common.utc_now(), layer_manifest_sha256=layer_hashes,
                       finalization_elapsed_seconds=time.perf_counter() - started,
                       finalization_peak_rss_mb=common.peak_rss_mb(),
                       final_files_sha256={name: common.sha256_file(root / name) for name in (*names, "runtime.csv")})
    preparation.pop("last_finalization_failure", None)
    common.atomic_write_json(root / "run_manifest.json", preparation)
    common.log(f"export complete: layers={len(common.LAYERS)} files={len(bundle.eligible_protein_ids) * len(common.FAMILY_SPECS) * len(common.LAYERS)} root={root}")


def finalize():
    root = common.output_root()
    with common.task_lock(root, "finalize"):
        try:
            _finalize_locked(root)
            normalize_shared_tree(root)
        except Exception:
            import traceback
            path = root / "run_manifest.json"
            if path.exists():
                manifest = common.load_json(path)
                if manifest.get("export_schema") == common.EXPORT_SCHEMA:
                    manifest.update(run_status="validation_failed", last_finalization_failure={
                        "timestamp": common.utc_now(), "traceback": traceback.format_exc(),
                    })
                    common.atomic_write_json(path, manifest)
            raise


if __name__ == "__main__":
    common.entrypoint(finalize)

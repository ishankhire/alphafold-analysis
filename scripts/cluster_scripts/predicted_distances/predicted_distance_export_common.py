#!/usr/bin/env python3
"""Storage, provenance, and validation for packed protocol-2.0 OOF exports."""
from __future__ import annotations

import csv
import fcntl
import importlib.metadata
import math
import os
import resource
import sys
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
for directory in (SCRIPT_DIR, CLUSTER_DIR, CLUSTER_DIR / "cross_validation",
                  CLUSTER_DIR / "triangle_violation_subgroups"):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import triangle_violation_subgroup_replay as replay  # noqa: E402
from cross_validation_common import RegressionSufficientStatistics, load_cv_folds, semantic_digest, utc_now  # noqa: E402
from regression_output_helpers import atomic_write_csv, atomic_write_json  # noqa: E402
from triangle_violation_subgroup_common import (  # noqa: E402
    EXPECTED_PROTEINS, FAMILY_SPECS, LAYERS, load_and_validate_replay_index,
    load_json, sha256_file, validate_family_manifest,
)

EXPORT_SCHEMA = "predicted_distances_5fold_v1"
DEFAULT_OUTPUT_ROOT = Path("/work/hdd/bekh/ikhire/regression_5fold_v1") / EXPORT_SCHEMA
DTYPE = np.dtype("<f4")
METRICS = ("r2", "mae", "rmse", "bias_pred_minus_true", "mean_true_distance",
           "mean_predicted_distance")
RTOL = ATOL = 1e-8
IDENTITY_COLUMNS = ("protein_id", "analysis_name", "model_family", "layer", "source",
                    "outer_fold", "cv_protocol_version", "evaluation_design",
                    "evaluation_scope", "oof_prediction_scope")
INDEX_COLUMNS = IDENTITY_COLUMNS + (
    "array_path", "dtype", "n_values", "file_bytes", "array_sha256",
    "layer_manifest_path", "status",
)
VALIDATION_COLUMNS = IDENTITY_COLUMNS + (
    "n_expected_pairs", "n_predicted_pairs", "pair_coverage", "prediction_status",
    "contributing_fit_status", "n_negative_predictions", "stored_negative_predictions",
    "max_abs_cast_error", "mean_abs_cast_error",
) + tuple(f"{prefix}_{name}" for name in METRICS for prefix in (
    "replay", "producer_difference", "stored", "stored_minus_replay",
)) + ("validation_status", "failure_reason")
PROTEIN_COLUMNS = ("protein_id", "source", "outer_fold", "n_residues", "n_pairs",
                   "pair_assignment_digest", "pdb_path", "pdb_sha256")
RESIDUE_COLUMNS = ("protein_id", "residue_index", "chain_id", "pdb_resseq",
                   "insertion_code", "residue_name", "pdb_path", "pdb_sha256")
RUNTIME_COLUMNS = ("layer", "started_at", "ended_at", "elapsed_seconds",
                   "n_proteins", "n_arrays", "output_bytes", "peak_rss_mb", "status")
POOLED_COLUMNS = ("analysis_name", "layer", "n_pairs", "validation_status") + tuple(
    f"{prefix}_{name}" for name in METRICS for prefix in (
        "replay", "producer_difference", "stored", "stored_minus_replay",
    )
)


def output_root() -> Path:
    return Path(os.environ.get("PREDICTED_DISTANCE_OUTPUT_ROOT", str(DEFAULT_OUTPUT_ROOT))).expanduser().resolve()


def producer_root() -> Path:
    return replay.producer_root().resolve()


def load_bundle():
    bundle = load_cv_folds(metadata_dir=producer_root() / "metadata")
    if len(bundle.eligible_protein_ids) != EXPECTED_PROTEINS:
        raise ValueError("Export requires the complete frozen eligible cohort")
    return bundle


def log(message: str) -> None:
    print(f"[{utc_now()}] {message}", flush=True)


def atomic_write_text(path: Path, text: str):
    fd, temporary = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as handle:
            handle.write(text)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def peak_rss_mb() -> float:
    value = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    return value / (1024 ** 2 if sys.platform == "darwin" else 1024)


def read_csv(path: Path, columns=None) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if columns is not None and tuple(reader.fieldnames or ()) != tuple(columns):
            raise ValueError(f"Unexpected CSV schema: {path}")
        return list(reader)


def versions() -> dict[str, str]:
    result = {"python": sys.version.split()[0], "numpy": np.__version__}
    for package in ("xgboost", "scipy", "scikit-learn", "biopython"):
        try:
            result[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            result[package] = "unavailable"
    return result


def code_identity() -> dict[str, str]:
    # Imported modules have identical filenames in repository and flat deployment.
    names = ("cluster_regression_helpers", "regression_protocol", "regression_output_helpers",
             "cross_validation_common", "shared_ols_base256_common",
             "per_protein_ols_base256_common", "sequence_gated_ols_base256_common",
             "triangle_violation_subgroup_common", "triangle_violation_subgroup_replay")
    result = {name + ".py": sha256_file(Path(sys.modules[name].__file__)) for name in names}
    for name in ("predicted_distance_export_common.py", "prepare_predicted_distance_export.py",
                 "export_predicted_distances_layer.py", "finalize_predicted_distance_export.py"):
        result[name] = sha256_file(SCRIPT_DIR / name)
    result["predicted_distance_export_README.md"] = sha256_file(readme_source())
    return result


def readme_source() -> Path:
    deployed = SCRIPT_DIR / "predicted_distance_export_README.md"
    return deployed if deployed.exists() else SCRIPT_DIR / "README.md"


def source_identity(bundle, *, verify_artifacts=False) -> dict:
    producers = {}
    for spec in FAMILY_SPECS:
        root = producer_root() / spec.family_dir
        manifest = validate_family_manifest(root, spec, bundle.fold_digests)
        rows = load_and_validate_replay_index(root, spec, manifest, verify_artifacts=verify_artifacts)
        if spec.uses_pair_folds and {r["protein_id"] for r in rows} != set(bundle.eligible_protein_ids):
            raise ValueError("Per-protein replay index does not match frozen protein IDs")
        producers[spec.analysis_name] = {
            "root": str(root), "configuration_digest": manifest["configuration_digest"],
            "oof_prediction_scope": manifest["oof_prediction_scope"],
            "files": {name: sha256_file(root / name) for name in (
                "run_manifest.json", "model_artifacts/replay_index.csv", "per_protein.csv",
                "geometry_metrics.csv", "pooled_metrics.csv",
            )},
        }
        log(f"validated producer={spec.analysis_name} indexed_models={len(rows)}")
    return {
        "export_schema": EXPORT_SCHEMA, "cv_protocol_version": "2.0",
        "evaluation_scope": "oof_all_pairs", "dtype": DTYPE.str, "units": "angstrom",
        "pair_order": "np.tril_indices(R, k=-1)", "negative_prediction_handling": "raw_unclipped",
        "layers": list(LAYERS), "protein_ids": sorted(bundle.eligible_protein_ids),
        "output_root": str(output_root()), "producer_root": str(producer_root()),
        "fold_digests": dict(bundle.fold_digests), "producers": producers,
        "metadata_files": {name: sha256_file(producer_root() / "metadata" / name) for name in (
            "regression_outer_folds_v1.csv", "per_protein_pair_folds_v1.csv", "fold_digests.json",
        )}, "code_sha256": code_identity(), "versions": versions(),
        "metric_tolerance": {"rtol": RTOL, "atol": ATOL},
    }


def checked_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"Expected relative path: {relative}")
    result = (root / path).resolve()
    if not result.is_relative_to(root.resolve()):
        raise ValueError(f"Path escapes export root: {relative}")
    return result


@contextmanager
def task_lock(root: Path, name: str):
    lock_dir = root / ".locks"
    lock_dir.mkdir(parents=True, exist_ok=True)
    with (lock_dir / f"{name}.lock").open("a") as handle:
        try:
            fcntl.flock(handle, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError(f"Another {name} task is running") from exc
        yield


def load_preparation(root: Path, bundle) -> tuple[dict, dict[str, dict]]:
    manifest = load_json(root / "run_manifest.json")
    if manifest.get("preparation_status") != "complete":
        raise ValueError("Export preparation is not complete")
    identity = source_identity(bundle)
    if manifest.get("identity") != identity or manifest.get("identity_digest") != semantic_digest(identity):
        raise ValueError("Export input/configuration/code identity changed; refusing resume")
    for relative, checksum in manifest["metadata_sha256"].items():
        if sha256_file(checked_path(root, relative)) != checksum:
            raise ValueError(f"Prepared metadata changed: {relative}")
    rows = read_csv(root / "metadata/proteins.csv", PROTEIN_COLUMNS)
    proteins = {r["protein_id"]: r for r in rows}
    if len(proteins) != len(rows) or set(proteins) != set(bundle.eligible_protein_ids):
        raise ValueError("Prepared protein membership changed")
    for pid, row in proteins.items():
        outer = bundle.outer_by_protein[pid]
        if (row["source"] != outer.source or int(row["outer_fold"]) != outer.outer_fold
                or int(row["n_residues"]) != outer.n_residues or int(row["n_pairs"]) != outer.n_pairs
                or row["pair_assignment_digest"] != bundle.pair_by_protein[pid].assignment_digest):
            raise ValueError(f"Prepared protein/fold metadata mismatch: {pid}")
        if sha256_file(Path(row["pdb_path"])) != row["pdb_sha256"]:
            raise ValueError(f"Prepared PDB changed: {pid}")
    return manifest, proteins


def identity_row(spec, outer, layer, preparation) -> dict:
    return {
        "protein_id": outer.protein_id, "analysis_name": spec.analysis_name,
        "model_family": spec.model_family, "layer": layer, "source": outer.source,
        "outer_fold": outer.outer_fold, "cv_protocol_version": "2.0",
        "evaluation_design": spec.evaluation_design, "evaluation_scope": "oof_all_pairs",
        "oof_prediction_scope": preparation["identity"]["producers"][spec.analysis_name]["oof_prediction_scope"],
    }


def prediction_path(layer: int, pid: str, family: str) -> str:
    if layer not in LAYERS or family not in {s.analysis_name for s in FAMILY_SPECS}:
        raise ValueError("Unknown export layer/family")
    if not pid or Path(pid).name != pid or pid in {".", ".."}:
        raise ValueError("Invalid protein identifier")
    return f"layers/layer_{layer:02d}/{pid}/{family}.npy"


def atomic_save_prediction(root: Path, layer: int, pid: str, spec, values) -> dict:
    values = np.asarray(values, dtype=np.float64)
    if values.ndim != 1 or not values.size or not np.all(np.isfinite(values)):
        raise ValueError("Predictions must be a complete finite vector")
    with np.errstate(over="ignore"):
        stored = values.astype(DTYPE)
    if not np.all(np.isfinite(stored)):
        raise ValueError("Prediction overflow in float32 storage")
    relative = prediction_path(layer, pid, spec.analysis_name)
    destination = checked_path(root, relative)
    destination.parent.mkdir(parents=True, exist_ok=True)
    checkpoint_dir = root / ".checkpoints" / f"layer_{layer:02d}" / pid
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    fd, name = tempfile.mkstemp(prefix=f".{spec.analysis_name}.", suffix=".npy.tmp", dir=checkpoint_dir)
    try:
        with os.fdopen(fd, "wb") as handle:
            np.save(handle, stored, allow_pickle=False)
            handle.flush()
            os.fsync(handle.fileno())
        reloaded = np.load(name, allow_pickle=False)
        if reloaded.dtype != DTYPE or not np.array_equal(reloaded, stored):
            raise ValueError("Written prediction failed round-trip validation")
        checksum = sha256_file(Path(name))
        os.replace(name, destination)
        # Persist the rename before committing the progress snapshot.
        directory_fd = os.open(destination.parent, os.O_RDONLY)
        try:
            os.fsync(directory_fd)
        finally:
            os.close(directory_fd)
    finally:
        if os.path.exists(name):
            os.unlink(name)
    return {
        "array_path": relative, "dtype": DTYPE.str, "n_values": values.size,
        "file_bytes": destination.stat().st_size, "array_sha256": checksum,
        "layer_manifest_path": f"layers/layer_{layer:02d}/layer_manifest.json", "status": "complete",
    }


def check_array(root: Path, row: dict, outer, spec, layer: int) -> np.ndarray:
    expected = prediction_path(layer, outer.protein_id, spec.analysis_name)
    if (row["array_path"] != expected or row["dtype"] != DTYPE.str
            or int(row["n_values"]) != outer.n_pairs or row["status"] != "complete"
            or row["layer_manifest_path"] != f"layers/layer_{layer:02d}/layer_manifest.json"):
        raise ValueError("Prediction index identity/coverage mismatch")
    path = checked_path(root, expected)
    if path.stat().st_size != int(row["file_bytes"]) or sha256_file(path) != row["array_sha256"]:
        raise ValueError(f"Prediction checksum/size mismatch: {path}")
    array = np.load(path, allow_pickle=False)
    if array.dtype.str != DTYPE.str or array.shape != (outer.n_pairs,) or not np.all(np.isfinite(array)):
        raise ValueError(f"Prediction shape/dtype/coverage mismatch: {path}")
    return array


def number(value) -> float:
    return float("nan") if value is None or value == "" else float(value)


def json_number(value: float):
    return float(value) if math.isfinite(value) else None


def compare_metric(actual, expected, label: str) -> float:
    actual, expected = number(actual), number(expected)
    if not np.isclose(actual, expected, rtol=RTOL, atol=ATOL, equal_nan=True):
        raise ValueError(f"Replayed metric mismatch: {label}: {actual} vs {expected}")
    return actual - expected


def statistics(targets, predictions):
    stats = RegressionSufficientStatistics()
    stats.update(targets, predictions)
    return stats


def producer_rows(spec, layer: int, bundle) -> dict[str, dict]:
    root = producer_root() / spec.family_dir
    results = {}
    for filename in ("per_protein.csv", "geometry_metrics.csv"):
        selected = {}
        with (root / filename).open(newline="") as handle:
            for row in csv.DictReader(handle):
                if int(row["layer"]) != layer:
                    continue
                pid = row["protein_id"]
                if pid in selected:
                    raise ValueError(f"Duplicate producer row: {filename} {layer} {pid}")
                selected[pid] = row
        if set(selected) != set(bundle.eligible_protein_ids):
            raise ValueError(f"Incomplete producer table: {spec.analysis_name}/{filename}")
        results[filename] = selected
    return results


def fit_statuses(spec, outer, artifacts) -> list[str]:
    models = artifacts[spec.analysis_name]
    if spec.uses_pair_folds:
        shard, index = models
        return [str(shard["fit_status"][index[(outer.protein_id, f)]]) for f in range(5)]
    if spec.uses_sequence_gates:
        return [str(models[(outer.outer_fold, gate)]["fit_status"]) for gate in replay.SEQUENCE_BIN_LABELS]
    if spec.model_family == "shared_xgboost":
        booster = models[outer.outer_fold]
        if booster.num_boosted_rounds() != 700 or booster.num_features() != 256:
            raise ValueError("Frozen XGBoost must contain 700 rounds and 256 features")
        return ["fitted"]
    return [str(models[outer.outer_fold]["fit_status"])]


def validation_record(spec, outer, layer, targets, values, artifacts, tables, preparation) -> tuple[dict, dict]:
    values = np.asarray(values, dtype=np.float64)
    if values.shape != (outer.n_pairs,) or not np.all(np.isfinite(values)):
        raise ValueError("Replay prediction coverage is incomplete")
    statuses = fit_statuses(spec, outer, artifacts)
    if any(not s.startswith("fitted") for s in statuses):
        raise ValueError("Incomplete contributing fit")
    status = "complete" if all(s == "fitted" for s in statuses) else "complete_from_flagged_fit"
    expected = tables["per_protein.csv"][outer.protein_id]
    geometry = tables["geometry_metrics.csv"][outer.protein_id]
    identity = identity_row(spec, outer, layer, preparation)
    for table in (expected, geometry):
        if any(str(table[k]) != str(v) for k, v in identity.items()):
            raise ValueError("Producer table scope/fold identity mismatch")
        if (int(table["n_residues"]) != outer.n_residues
                or int(table["n_protein_pairs"]) != outer.n_pairs
                or float(table["pair_coverage"]) != 1):
            raise ValueError("Producer pair/residue coverage mismatch")
    if (expected["prediction_status"] != status or geometry["geometry_status"] != "complete"
            or int(expected["n_expected_pairs"]) != outer.n_pairs
            or int(expected["n_evaluated_pairs"]) != outer.n_pairs):
        raise ValueError("Producer prediction status/support mismatch")
    negative = int(np.count_nonzero(values < 0))
    if negative != int(geometry["n_negative_predictions"]):
        raise ValueError("Replayed negative-prediction count mismatch")
    raw = statistics(targets, values)
    with np.errstate(over="ignore"):
        stored_values = values.astype(DTYPE)
    if not np.all(np.isfinite(stored_values)):
        raise ValueError("Prediction overflow in float32 storage")
    stored = statistics(targets, stored_values)
    raw_metrics, stored_metrics = raw.metrics(), stored.metrics()
    row = dict(identity, n_expected_pairs=outer.n_pairs, n_predicted_pairs=values.size,
               pair_coverage=1, prediction_status=status, contributing_fit_status=";".join(statuses),
               n_negative_predictions=negative, stored_negative_predictions=int(np.count_nonzero(stored_values < 0)),
               max_abs_cast_error=float(np.max(np.abs(stored_values.astype(np.float64) - values))),
               mean_abs_cast_error=float(np.mean(np.abs(stored_values.astype(np.float64) - values))),
               validation_status="complete", failure_reason="")
    for name in METRICS:
        difference = compare_metric(raw_metrics[name], expected[name], f"{spec.analysis_name}/{outer.protein_id}/{name}")
        row.update({f"replay_{name}": json_number(raw_metrics[name]),
                    f"producer_difference_{name}": json_number(difference),
                    f"stored_{name}": json_number(stored_metrics[name]),
                    f"stored_minus_replay_{name}": json_number(stored_metrics[name] - raw_metrics[name])})
    return row, {"replay": raw.to_payload(), "stored": stored.to_payload()}


def validate_records(root, layer, records, bundle, preparation, *, require_complete=False):
    if set(records) - set(bundle.eligible_protein_ids):
        raise ValueError("Unknown checkpoint protein")
    if require_complete and set(records) != set(bundle.eligible_protein_ids):
        raise ValueError("Incomplete layer protein coverage")
    index_rows, validation_rows = [], []
    for pid in sorted(records):
        record = records[pid]
        outer = bundle.outer_by_protein[pid]
        if set(record["families"]) != {s.analysis_name for s in FAMILY_SPECS}:
            raise ValueError("Missing/duplicate family in checkpoint")
        for spec in FAMILY_SPECS:
            entry = record["families"][spec.analysis_name]
            index, validation = entry["index"], entry["validation"]
            identity = identity_row(spec, outer, layer, preparation)
            for row in (index, validation):
                if any(str(row[k]) != str(v) for k, v in identity.items()):
                    raise ValueError("Checkpoint identity/scope mismatch")
            if (set(index) != set(INDEX_COLUMNS) or set(validation) != set(VALIDATION_COLUMNS)
                    or validation["validation_status"] != "complete" or validation["failure_reason"]
                    or int(validation["n_expected_pairs"]) != outer.n_pairs
                    or int(validation["n_predicted_pairs"]) != outer.n_pairs
                    or float(validation["pair_coverage"]) != 1
                    or validation["prediction_status"] not in {"complete", "complete_from_flagged_fit"}):
                raise ValueError("Checkpoint validation schema/status/coverage mismatch")
            stored = check_array(root, index, outer, spec, layer)
            statuses = validation["contributing_fit_status"].split(";")
            expected_models = 5 if spec.uses_pair_folds else 6 if spec.uses_sequence_gates else 1
            if len(statuses) != expected_models or any(not s.startswith("fitted") for s in statuses):
                raise ValueError("Checkpoint contributing fit status mismatch")
            expected_status = "complete" if all(s == "fitted" for s in statuses) else "complete_from_flagged_fit"
            if validation["prediction_status"] != expected_status:
                raise ValueError("Checkpoint flagged-fit status mismatch")
            if int(np.count_nonzero(stored < 0)) != int(validation["stored_negative_predictions"]):
                raise ValueError("Stored negative-count mismatch")
            for name in METRICS:
                compare_metric(number(validation[f"replay_{name}"]) - number(validation[f"producer_difference_{name}"]),
                               number(validation[f"replay_{name}"]), "checkpoint producer parity")
            for kind in ("replay", "stored"):
                stats = RegressionSufficientStatistics.from_payload(entry["statistics"][kind])
                if stats.n != outer.n_pairs:
                    raise ValueError("Checkpoint sufficient-statistic support mismatch")
                for name, value in stats.metrics().items():
                    if name in METRICS:
                        compare_metric(value, validation[f"{kind}_{name}"], "checkpoint statistics")
            index_rows.append(index)
            validation_rows.append(validation)
    return index_rows, validation_rows


def pooled_validation(layer, records, bundle) -> list[dict]:
    output = []
    n_pairs = sum(bundle.outer_by_protein[pid].n_pairs for pid in bundle.eligible_protein_ids)
    for spec in FAMILY_SPECS:
        stats = {kind: RegressionSufficientStatistics() for kind in ("replay", "stored")}
        for record in records.values():
            for kind in stats:
                stats[kind].merge(RegressionSufficientStatistics.from_payload(record["families"][spec.analysis_name]["statistics"][kind]))
        saved = [r for r in read_csv(producer_root() / spec.family_dir / "pooled_metrics.csv") if int(r["layer"]) == layer]
        if len(saved) != 1 or int(saved[0]["n_proteins"]) != len(bundle.eligible_protein_ids) or int(saved[0]["n_evaluated_pairs"]) != n_pairs:
            raise ValueError("Producer pooled coverage mismatch")
        row = {"analysis_name": spec.analysis_name, "layer": layer, "n_pairs": n_pairs, "validation_status": "complete"}
        raw, stored = stats["replay"].metrics(), stats["stored"].metrics()
        for name in METRICS:
            difference = (compare_metric(raw[name], saved[0][f"pooled_{name}"], f"pooled/{spec.analysis_name}/{layer}/{name}")
                          if f"pooled_{name}" in saved[0] else float("nan"))
            row.update({f"replay_{name}": json_number(raw[name]), f"producer_difference_{name}": json_number(difference),
                        f"stored_{name}": json_number(stored[name]), f"stored_minus_replay_{name}": json_number(stored[name] - raw[name])})
        output.append(row)
    return output


def load_completed_layer(root, layer, bundle, preparation):
    directory = root / "layers" / f"layer_{layer:02d}"
    manifest = load_json(directory / "layer_manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("layer") != layer
            or manifest.get("identity_digest") != preparation["identity_digest"]):
        raise ValueError("Layer completion/identity mismatch")
    if set(manifest["metadata_sha256"]) != {"prediction_index.csv", "validation.csv", "pooled_validation.csv", "runtime.json"}:
        raise ValueError("Layer metadata inventory mismatch")
    for name, checksum in manifest["metadata_sha256"].items():
        if sha256_file(checked_path(directory, name)) != checksum:
            raise ValueError(f"Layer metadata checksum mismatch: {name}")
    records = manifest["records"]
    indexes, validation = validate_records(root, layer, records, bundle, preparation, require_complete=True)
    expected = {r["array_path"] for r in indexes}
    observed = {str(p.relative_to(root)) for p in directory.rglob("*.npy")}
    if observed != expected:
        raise ValueError("Unexpected/missing physical prediction files")
    runtime = manifest["runtime"]
    if (load_json(directory / "runtime.json") != runtime or set(runtime) != set(RUNTIME_COLUMNS)
            or runtime["status"] != "complete" or runtime["layer"] != layer
            or runtime["n_proteins"] != len(bundle.eligible_protein_ids)
            or runtime["n_arrays"] != len(indexes)
            or runtime["output_bytes"] != sum(r["file_bytes"] for r in indexes)):
        raise ValueError("Layer runtime/support mismatch")
    # CSVs are conveniences; manifest records are the authoritative checkpoint.
    if read_csv(directory / "prediction_index.csv", INDEX_COLUMNS) != read_csv_rows(INDEX_COLUMNS, indexes):
        raise ValueError("Layer index differs from committed records")
    if read_csv(directory / "validation.csv", VALIDATION_COLUMNS) != read_csv_rows(VALIDATION_COLUMNS, validation):
        raise ValueError("Layer validation differs from committed records")
    return manifest, indexes, validation


def read_csv_rows(columns, rows):
    # Match the shared writer's exact precision and NaN formatting in memory.
    import io
    from regression_output_helpers import _csv_value
    buffer = io.StringIO()
    writer = csv.DictWriter(buffer, fieldnames=columns)
    writer.writeheader()
    writer.writerows({k: _csv_value(row[k]) for k in columns} for row in rows)
    buffer.seek(0)
    return list(csv.DictReader(buffer))


def load_predicted_distances(root, layer, protein_id, probe, *, as_matrix=False):
    """Read one validated packed vector, optionally reconstructing a symmetric matrix."""
    root = Path(root).expanduser().resolve()
    relative = prediction_path(layer, protein_id, probe)
    manifest = load_json(root / "layers" / f"layer_{layer:02d}" / "layer_manifest.json")
    run = load_json(root / "run_manifest.json")
    if (manifest.get("status") != "complete" or manifest.get("layer") != layer
            or run.get("preparation_status") != "complete"
            or manifest.get("identity_digest") != run.get("identity_digest")):
        raise ValueError("Cannot read an incomplete/incompatible layer")
    if run.get("run_status") == "complete" and sha256_file(root / "layers" / f"layer_{layer:02d}" / "layer_manifest.json") != run["layer_manifest_sha256"][str(layer)]:
        raise ValueError("Completed layer manifest checksum mismatch")
    protein_path = root / "metadata/proteins.csv"
    if sha256_file(protein_path) != run["metadata_sha256"]["metadata/proteins.csv"]:
        raise ValueError("Protein metadata checksum mismatch")
    row = manifest["records"][protein_id]["families"][probe]["index"]
    proteins = {r["protein_id"]: r for r in read_csv(protein_path, PROTEIN_COLUMNS)}
    from types import SimpleNamespace
    outer = SimpleNamespace(protein_id=protein_id, n_pairs=int(proteins[protein_id]["n_pairs"]))
    spec = next(s for s in FAMILY_SPECS if s.analysis_name == probe)
    values = check_array(root, row, outer, spec, layer)
    if row["array_path"] != relative:
        raise ValueError("Unexpected prediction path")
    if not as_matrix:
        return values
    n_residues = int(proteins[protein_id]["n_residues"])
    matrix = np.zeros((n_residues, n_residues), dtype=DTYPE)
    i, j = np.tril_indices(n_residues, k=-1)
    matrix[i, j] = matrix[j, i] = values
    return matrix


def entrypoint(function):
    """Log exceptions; diagnostics stay beside outputs, never in the model root."""
    import traceback
    try:
        function()
    except Exception:
        log("FAILED\n" + traceback.format_exc())
        raise

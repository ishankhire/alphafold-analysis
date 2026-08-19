#!/usr/bin/env python3
"""Replay four frozen distance probes on DSSP-selected structure pairs.

Stages are selected with ``SECONDARY_STRUCTURE_STAGE=prepare|layer|consolidate``.
The implementation deliberately imports the established DSSP/geometry helpers
from ``diagnostic_secondary_structure_emergence`` so the primary comparison
uses exactly the same parser, bridge controls, calibration score, and segment
definitions as the audited exploratory analysis.
"""

from __future__ import annotations

import csv
import gzip
import hashlib
import itertools
import json
import math
import os
import platform
import resource
import shutil
import subprocess
import sys
import time
import traceback
from collections import Counter, defaultdict
from datetime import datetime, timezone
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

HERE = Path(__file__).resolve().parent
CLUSTER = HERE.parent
DIAGNOSTICS = CLUSTER / "diagnostics"
GENERAL = CLUSTER / "generalized_model_ecod"
for path in (CLUSTER, DIAGNOSTICS, GENERAL, HERE):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import diagnostic_secondary_structure_emergence as geometry  # noqa: E402
from general_train_test_common import (  # noqa: E402
    build_dataset_protein_manifest, resolve_pair_block_path, resolve_pdb_file,
)
from regression_protocol import (  # noqa: E402
    FEATURE_DIM, LAYERS, MASTER_MANIFEST_DIGEST, SEQUENCE_BIN_LABELS,
    WITHIN_PROTEIN_SPLIT_SEED, eligible_manifest, extract_directional_base256,
    load_master_manifest, sequence_separation_bin_codes,
    strict_lower_triangle_indices, within_protein_pair_split,
    within_protein_pair_split_digest, within_protein_split_seed,
)

ANALYSIS_NAME = "secondary_structure_model_comparison"
SCHEMA_VERSION = "1.0"
MODELS = ("sequence_gated_ols", "shared_global_ols", "per_protein_ols", "shared_xgboost")
SCOPES = ("all_selected_structure_pairs", "per_protein_heldout_matched")
CONTRASTS = ("matched_random", "shift_1", "shift_2")
ORIENTATIONS = ("all", "parallel", "antiparallel")
RANDOM_SEED = 20260813
MIN_TRUE_HELIX_AUROC = 0.95
HELIX_MIN_SEGMENT_LENGTH = 6

VALIDATION_FIELDS = ("category", "metric", "model_family", "layer", "split", "orientation", "value", "n", "threshold", "status", "details")
FAILURE_FIELDS = ("stage", "model_family", "layer", "protein_id", "source", "failure_type", "reason", "traceback_summary", "fatal")
RUNTIME_FIELDS = ("stage", "layer", "started_at", "ended_at", "elapsed_seconds", "n_proteins", "n_selected_pairs", "model_prediction_seconds", "peak_rss_mb", "status")
ARTIFACT_FIELDS = ("model_family", "layer", "protein_id", "artifact_role", "artifact_path", "array_row", "fit_status", "sha256", "validation_status", "details")


def utcnow():
    return datetime.now(timezone.utc).isoformat()


def code_revision():
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=HERE, text=True, stderr=subprocess.DEVNULL).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def package_versions():
    result={"python":platform.python_version()}
    for package in ("numpy","scikit-learn","biopython","xgboost"):
        try: result[package]=version(package)
        except PackageNotFoundError: result[package]="unavailable"
    return result


def finite_mean(values):
    a = np.asarray(list(values), dtype=np.float64)
    a = a[np.isfinite(a)]
    return float(a.mean()) if a.size else math.nan


def safe_auc(labels, scores):
    labels, scores = np.asarray(labels), np.asarray(scores, dtype=np.float64)
    ok = np.isfinite(scores)
    return float(roc_auc_score(labels[ok], scores[ok])) if ok.any() and np.unique(labels[ok]).size == 2 else math.nan


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def atomic_text(path, value):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(value); tmp.replace(path)


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def atomic_csv(path, rows, fields=None, gz=False):
    path = Path(path); path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if fields is None:
        fields = list(rows[0].keys()) if rows else []
        fields += sorted(set().union(*(row.keys() for row in rows)) - set(fields)) if rows else []
    fields = list(fields)
    tmp = path.with_name(path.name + ".tmp")
    opener = gzip.open if gz or path.suffix == ".gz" else open
    with opener(tmp, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader(); writer.writerows(rows)
    tmp.replace(path)


def read_csv(path):
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


def project_or_flat_root():
    """Return the repository root locally or the flat deployment directory on Delta."""
    return geometry.find_project_root(HERE)


def output_root():
    if os.environ.get("SECONDARY_STRUCTURE_OUTPUT_DIR"):
        return Path(os.environ["SECONDARY_STRUCTURE_OUTPUT_DIR"]).expanduser()
    base = project_or_flat_root()
    local_default = base / "delta_files" if base != HERE else HERE / "delta_files"
    return Path(os.environ.get("CSV_ROOT", str(local_default))).expanduser() / ANALYSIS_NAME


def input_roots():
    base = project_or_flat_root()
    local_default = base / "delta_files" if base != HERE else HERE / "delta_files"
    csv_root = Path(os.environ.get("CSV_ROOT", str(local_default))).expanduser()
    return {
        "shared_global_ols": Path(os.environ.get("SHARED_GLOBAL_OLS_ROOT", csv_root / "general_base256_combined")),
        "sequence_gated_ols": Path(os.environ.get("SEQUENCE_GATED_OLS_ROOT", csv_root / "gated_linear_probes/consolidated/analysis_1_sequence_gated")),
        "per_protein_ols": Path(os.environ.get("PER_PROTEIN_OLS_ROOT", csv_root / "per_protein_base256_combined")),
        "shared_xgboost": Path(os.environ.get("SHARED_XGBOOST_ROOT", csv_root / "shared_xgboost_base256_combined")),
    }


def _require_fields(rows, required, path):
    if not rows or set(required) - set(rows[0]):
        raise ValueError(f"{path} lacks required fields {sorted(set(required) - set(rows[0]) if rows else required)}")


def load_shared_ols(root, layer):
    suffix = "0-23" if layer < 24 else "24-47"
    path = root / f"coefficients_{suffix}.csv"
    rows = read_csv(path)
    _require_fields(rows, ("variant", "layer", "feature_index", "coefficient_raw_scale"), path)
    selected = [r for r in rows if r["variant"] == "general_base256_combined" and int(r["layer"]) == layer]
    indices = [int(r["feature_index"]) for r in selected]
    if len(selected) != FEATURE_DIM + 1 or Counter(indices) != Counter(range(-1, FEATURE_DIM)):
        raise ValueError(f"Layer {layer} shared OLS coefficients are missing or duplicated")
    by_index = {int(r["feature_index"]): float(r["coefficient_raw_scale"]) for r in selected}
    return by_index[-1], np.asarray([by_index[i] for i in range(FEATURE_DIM)]), path


def _load_index(root, predicates, expected, family):
    path = root / "model_artifacts/coefficient_index.csv"
    rows = read_csv(path)
    _require_fields(rows, ("layer", "protein_id", "shard_path", "array_row", "fit_status", "file_sha256"), path)
    selected = [r for r in rows if all(str(r.get(k, "")) == str(v) for k, v in predicates.items())]
    if len(selected) != expected:
        raise ValueError(f"{family} expected {expected} index rows for {predicates}, observed {len(selected)}")
    return selected


def _indexed_vectors(root, rows, family):
    result = {}
    by_path = defaultdict(list)
    for row in rows:
        by_path[row["shard_path"]].append(row)
    for relative, selected in by_path.items():
        path = root / relative
        digests = {r["file_sha256"] for r in selected}
        if len(digests) != 1 or sha256_file(path) not in digests:
            raise ValueError(f"{family} coefficient checksum mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            required = {"coefficient_raw_scale", "intercept_raw_scale"}
            if required - set(archive.files):
                raise ValueError(f"{path} lacks arrays {sorted(required - set(archive.files))}")
            for row in selected:
                index = int(row["array_row"])
                if not row["fit_status"].startswith("fitted"):
                    raise ValueError(f"{family} non-fitted artifact: {row}")
                coef = np.asarray(archive["coefficient_raw_scale"][index], dtype=np.float64)
                intercept = float(archive["intercept_raw_scale"][index])
                if coef.shape != (FEATURE_DIM,) or not np.isfinite(coef).all() or not math.isfinite(intercept):
                    raise ValueError(f"{family} invalid coefficient row {index} in {path}")
                if "protein_id" in archive.files and str(archive["protein_id"][index]) != row["protein_id"]:
                    raise ValueError(f"{family} protein/index mismatch in {path}")
                key = row.get("protein_id") or row.get("gate_sequence_bin")
                result[key] = (intercept, coef, row, path)
    return result


def load_sequence_gated(root, layer):
    rows = _load_index(root, {"layer": layer, "gate_aa_pair": "all"}, 6, "sequence_gated_ols")
    labels = {r.get("gate_sequence_bin") for r in rows}
    if labels != set(SEQUENCE_BIN_LABELS):
        raise ValueError(f"Layer {layer} has wrong sequence gates: {labels}")
    return _indexed_vectors(root, rows, "sequence_gated_ols")


def load_per_protein(root, layer, protein_ids):
    path = root / "model_artifacts/coefficient_index.csv"
    all_rows = read_csv(path)
    _require_fields(all_rows, ("layer", "protein_id", "shard_path", "array_row", "fit_status", "file_sha256"), path)
    rows = [r for r in all_rows if int(r["layer"]) == layer and r["protein_id"] in set(protein_ids)]
    if {r["protein_id"] for r in rows} != set(protein_ids):
        raise ValueError(f"Layer {layer} per-protein coefficient coverage differs from locked test")
    return _indexed_vectors(root, rows, "per_protein_ols")


def load_xgboost(root, layer):
    try:
        import xgboost as xgb
    except ImportError as exc:
        raise RuntimeError("xgboost is required to replay shared_xgboost_base256_combined") from exc
    path = root / f"model_artifacts/booster_layer_{layer:02d}.json"
    booster = xgb.Booster(); booster.load_model(path)
    if booster.num_features() != FEATURE_DIM:
        raise ValueError(f"Layer {layer} XGBoost feature width is {booster.num_features()}, expected {FEATURE_DIM}")
    return booster, path


def predict_xgboost(booster, features):
    import xgboost as xgb
    return np.asarray(booster.predict(xgb.DMatrix(np.asarray(features, dtype=np.float64))), dtype=np.float64)


def canonical_pair_position(i, j):
    hi, lo = (i, j) if i > j else (j, i)
    if hi == lo:
        raise ValueError("Self-pairs are not valid structure pairs")
    return hi * (hi - 1) // 2 + lo


def selected_pair_union(cache):
    pairs = set()
    for i, code in enumerate(cache["dssp_codes"].astype(str)):
        if code in {"H", "E"} and i + 4 < len(cache["chains"]) and all(cache["chains"][i] == cache["chains"][i + o] for o in (2, 3, 4)):
            pairs.update((min(i, i + o), max(i, i + o)) for o in (2, 3, 4))
    pairs.update(
        (min(int(i), int(j)), max(int(i), int(j)))
        for i, j, eligible in zip(cache["bridge_i"], cache["bridge_j"], cache["bridge_main_eligible"])
        if i != j and int(eligible)
    )
    pairs.update((min(int(i), int(j)), max(int(i), int(j))) for i, j in zip(cache["control_i"], cache["control_j"]) if i != j)
    return sorted(pairs)


def sparse_features(pair_block, pairs):
    if pair_block.ndim != 3 or pair_block.shape[0] != pair_block.shape[1] or pair_block.shape[2] != 128:
        raise ValueError(f"Unexpected pair tensor shape {pair_block.shape}")
    hi = np.asarray([max(p) for p in pairs], dtype=np.int64)
    lo = np.asarray([min(p) for p in pairs], dtype=np.int64)
    features = extract_directional_base256(pair_block, hi, lo)
    features = np.asarray(features, dtype=np.float64)
    if features.shape != (len(pairs), FEATURE_DIM) or not np.isfinite(features).all():
        raise ValueError("Selected directional features are nonfinite or incorrectly shaped")
    return features, hi, lo


def route_sequence_predictions(features, hi, lo, chains, models):
    codes = sequence_separation_bin_codes(hi, lo, chain_ids=chains)
    if np.any(codes < 0):
        raise ValueError("Sequence-gated replay received a cross-chain pair")
    result = np.empty(len(features), dtype=np.float64)
    for code, label in enumerate(SEQUENCE_BIN_LABELS):
        chosen = codes == code
        if chosen.any():
            intercept, coef, _, _ = models[label]
            result[chosen] = intercept + features[chosen] @ coef
    return result


def _dataset_records():
    records = {r["protein_id"]: dict(r) for r in build_dataset_protein_manifest()}
    base = project_or_flat_root()
    local = base.parent / "protein_data" if base != HERE else HERE / "protein_data"
    for protein_id, row in records.items():
        local_dir = local / f"{protein_id}_A"
        if local_dir.exists(): row["protein_dir"] = local_dir
    return records


def _resolve_inputs(record, layer):
    pair_root_key = "ECOD_PAIR_BLOCKS_ROOT" if record["source"] == "ecod" else "CAMEO_PAIR_BLOCKS_ROOT"
    pdb_root_key = "ECOD_PDB_ROOT" if record["source"] == "ecod" else "CAMEO_PDB_ROOT"
    protein_dir = Path(os.environ.get(pair_root_key, record["protein_dir"]))
    if record["source"] == "cameo" and os.environ.get(pair_root_key): protein_dir /= record["protein_id"]
    if record["source"] == "ecod" and os.environ.get(pair_root_key): protein_dir /= f"{record['protein_id']}_save_layers_0_47"
    pdb_root = Path(os.environ.get(pdb_root_key, record["pdb_root"]))
    return resolve_pair_block_path(protein_dir, record["protein_id"], layer), resolve_pdb_file(record["protein_id"], protein_dir, pdb_root)


def _pair_split_row(root, entry):
    rows = read_csv(root / "metadata/pair_splits.csv")
    matches = [r for r in rows if r["protein_id"] == entry.protein_id]
    if len(matches) != 1: raise ValueError(f"Missing/duplicate pair split for {entry.protein_id}")
    row = matches[0]
    expected = within_protein_pair_split_digest(entry.n_pairs, entry.protein_id)
    if (row["split_digest"] != expected or int(row["split_seed"]) != within_protein_split_seed(entry.protein_id)
            or int(row["n_protein_pairs"]) != entry.n_pairs):
        raise ValueError(f"Pair split metadata mismatch for {entry.protein_id}")
    return row


def validate_xgboost_model_status(rows):
    required = ("layer", "fit_status", "selected_configuration_id", "final_boosting_rounds", "training_row_cap")
    _require_fields(rows, required, "shared XGBoost model_status.csv")
    layers = [int(row["layer"]) for row in rows]
    statuses = {row["fit_status"] for row in rows}
    configurations = {row["selected_configuration_id"] for row in rows}
    rounds = {int(float(row["final_boosting_rounds"])) for row in rows}
    caps = {int(float(row["training_row_cap"])) for row in rows}
    if (len(rows) != len(LAYERS) or Counter(layers) != Counter(LAYERS)
            or statuses != {"complete"} or configurations != {"S4"}
            or rounds != {700} or caps != {5000}):
        raise ValueError(
            "Shared XGBoost model_status is not the frozen final model: "
            f"rows={len(rows)}, layers={sorted(set(layers))}, statuses={sorted(statuses)}, "
            f"configurations={sorted(configurations)}, rounds={sorted(rounds)}, caps={sorted(caps)}"
        )


def validate_model_inputs(roots, test_entries, artifact_rows):
    validations = []
    test_ids = [e.protein_id for e in test_entries]
    expected_counts = {e.protein_id: e.n_residues for e in test_entries}

    # Validate producer-level manifests and the two legacy OLS split files
    # before opening any layer artifact.
    for family in ("sequence_gated_ols", "per_protein_ols", "shared_xgboost"):
        path = roots[family] / "run_manifest.json"
        manifest = json.loads(path.read_text())
        digest = manifest.get("master_manifest_digest", manifest.get("master_split_digest"))
        if digest != MASTER_MANIFEST_DIGEST or manifest.get("run_status") != "complete":
            raise ValueError(f"{family} manifest is not a complete canonical-master run")
        if list(map(int, manifest.get("completed_layers", []))) != list(LAYERS) or int(manifest.get("feature_dim", -1)) != FEATURE_DIM:
            raise ValueError(f"{family} manifest lacks canonical 48-layer base256 coverage")
        artifact_rows.append({"model_family":family,"layer":"","protein_id":"","artifact_role":"run_manifest","artifact_path":str(path),"array_row":"","fit_status":"complete","sha256":sha256_file(path),"validation_status":"valid","details":"canonical digest and 48 layers"})
        for role, relative in (("model_status", "model_status.csv"), ("coefficient_index", "model_artifacts/coefficient_index.csv")):
            source_path = roots[family] / relative
            if source_path.is_file():
                artifact_rows.append({"model_family":family,"layer":"","protein_id":"","artifact_role":role,"artifact_path":str(source_path),"array_row":"","fit_status":"","sha256":sha256_file(source_path),"validation_status":"valid","details":"producer index/status"})
    xgb_status = read_csv(roots["shared_xgboost"] / "model_status.csv")
    validate_xgboost_model_status(xgb_status)
    split_paths = [roots["shared_global_ols"] / f"split_assignments_{suffix}.csv" for suffix in ("0-23", "24-47")]
    split_tables = [read_csv(path) for path in split_paths]
    for path, rows in zip(split_paths, split_tables):
        _require_fields(rows, ("split_id", "split", "protein_id", "n_residues"), path)
    normalized = [sorted((r["protein_id"], r["split"], int(r["n_residues"])) for r in rows if int(r["split_id"]) == 0) for rows in split_tables]
    if normalized[0] != normalized[1]:
        raise ValueError("Shared OLS split-assignment batches disagree")
    observed_test = {protein_id: n for protein_id, split, n in normalized[0] if split == "test"}
    if observed_test != expected_counts:
        raise ValueError("Shared OLS locked-test IDs/residue counts differ from canonical master manifest")
    for path in split_paths:
        artifact_rows.append({"model_family":"shared_global_ols","layer":"","protein_id":"","artifact_role":"split_assignments","artifact_path":str(path),"array_row":"","fit_status":"","sha256":sha256_file(path),"validation_status":"valid","details":"batch parity and canonical test membership"})
    for family, relative in (("per_protein_ols", "metadata/pair_splits.csv"), ("shared_xgboost", "metadata/split_assignments.csv")):
        path = roots[family] / relative
        artifact_rows.append({"model_family":family,"layer":"","protein_id":"","artifact_role":"split_metadata","artifact_path":str(path),"array_row":"","fit_status":"","sha256":sha256_file(path),"validation_status":"valid","details":"validated during preparation"})
    xgb_split = read_csv(roots["shared_xgboost"] / "metadata/split_assignments.csv")
    xgb_test = {r["protein_id"]: int(r["n_residues"]) for r in xgb_split if r["master_role"] == "test"}
    if xgb_test != expected_counts:
        raise ValueError("Shared XGBoost locked-test IDs/residue counts differ from canonical master manifest")
    for layer in LAYERS:
        intercept, coef, path = load_shared_ols(roots["shared_global_ols"], layer)
        artifact_rows.append({"model_family":"shared_global_ols","layer":layer,"protein_id":"","artifact_role":"coefficients","artifact_path":str(path),"array_row":"","fit_status":"fitted","sha256":sha256_file(path),"validation_status":"valid","details":""})
        gated = load_sequence_gated(roots["sequence_gated_ols"], layer)
        per = load_per_protein(roots["per_protein_ols"], layer, test_ids)
        booster, path = load_xgboost(roots["shared_xgboost"], layer)
        del booster
        artifact_rows.append({"model_family":"shared_xgboost","layer":layer,"protein_id":"","artifact_role":"booster","artifact_path":str(path),"array_row":"","fit_status":"fitted","sha256":sha256_file(path),"validation_status":"valid","details":""})
        for family, loaded in (("sequence_gated_ols", gated), ("per_protein_ols", per)):
            for key, (_, _, row, path) in loaded.items():
                artifact_rows.append({"model_family":family,"layer":layer,"protein_id":row.get("protein_id", ""),"artifact_role":"coefficient_shard","artifact_path":str(path),"array_row":row["array_row"],"fit_status":row["fit_status"],"sha256":row["file_sha256"],"validation_status":"valid","details":key})
        validations.append({"category":"artifact","metric":"layer_models_load","model_family":"all","layer":layer,"split":"test","orientation":"","value":4,"n":len(test_ids),"threshold":4,"status":"pass","details":"all frozen artifacts load"})
    return validations


def prepare_stage():
    started = time.time(); started_at=utcnow(); root = output_root(); root.mkdir(parents=True, exist_ok=True)
    entries = list(eligible_manifest(load_master_manifest())); test = [e for e in entries if e.master_role == "test"]
    if len(test) != 111: raise ValueError(f"Expected 111 locked-test proteins, observed {len(test)}")
    roots = input_roots(); artifacts = []
    validations = validate_model_inputs(roots, test, artifacts)
    records = _dataset_records(); executable = Path(os.environ.get("DSSP_EXECUTABLE", "mkdssp"))
    if shutil.which(str(executable)) is None and not executable.is_file(): raise FileNotFoundError(f"DSSP executable unavailable: {executable}")
    triples = defaultdict(list); sheet = defaultdict(list); sheet_margins=defaultdict(list); dssp_rows=[]; failures=[]; dssp_flags=[]
    pair_rows = read_csv(roots["per_protein_ols"] / "metadata/pair_splits.csv")
    pair_by_id = {r["protein_id"]: r for r in pair_rows}
    prepared_pair_count=0
    for number, entry in enumerate(entries, 1):
        try:
            split_row = pair_by_id.get(entry.protein_id)
            if split_row is None or split_row["split_digest"] != within_protein_pair_split_digest(entry.n_pairs, entry.protein_id) or int(split_row["split_seed"]) != within_protein_split_seed(entry.protein_id) or int(split_row["n_protein_pairs"]) != entry.n_pairs:
                raise ValueError("per-protein split seed/count/digest mismatch")
            record = records[entry.protein_id]; pair_path, pdb_path = _resolve_inputs(record, 0)
            if not pair_path.is_file() or pdb_path is None or not Path(pdb_path).is_file(): raise FileNotFoundError("missing PDB or reference pair tensor")
            residues = geometry.parse_pdb_residues(Path(pdb_path)); block = np.load(pair_path)
            if block.shape != (entry.n_residues, entry.n_residues, 128) or len(residues) != entry.n_residues: raise ValueError("PDB/pair/master residue-count mismatch")
            dssp, raw_bridges, exit_status, stderr = geometry.run_dssp(Path(pdb_path), executable)
            codes=np.full(entry.n_residues,"-",dtype="U1"); mapped=0
            for i,residue in enumerate(residues):
                value=dssp.get(geometry.residue_key(residue))
                if value is not None: codes[i]=value[1] or "-"; mapped += 1
            if mapped != entry.n_residues: raise ValueError(f"DSSP mapped {mapped}/{entry.n_residues} residues")
            chains=np.asarray([r["chain"] for r in residues]); hs,hseg=geometry.build_segments(codes,chains,"H"); es,eseg=geometry.build_segments(codes,chains,"E")
            bridges=geometry.deduplicate_bridges(raw_bridges,residues,codes); controls=geometry.choose_bridge_controls(entry.protein_id,residues,codes,es,bridges)
            cache_path=root/"annotation_cache"/f"{entry.protein_id}.npz"
            geometry.encode_annotation_cache(cache_path,residues,codes,hs,es,hseg,eseg,bridges,controls)
            with np.load(cache_path) as old: data={k:old[k] for k in old.files}
            _,heldout=within_protein_pair_split(entry.n_pairs,entry.protein_id); heldout_set=set(heldout.tolist())
            pairs=selected_pair_union(data); positions=np.asarray([canonical_pair_position(*p) for p in pairs],dtype=np.int64)
            prepared_pair_count += len(pairs)
            data.update(selected_pair_i=np.asarray([p[0] for p in pairs],dtype=np.int32),selected_pair_j=np.asarray([p[1] for p in pairs],dtype=np.int32),selected_pair_position=positions,selected_pair_heldout=np.asarray([p in heldout_set for p in positions],dtype=np.int8))
            geometry.atomic_savez(cache_path,**data)
            distances=geometry.distance_matrix(np.asarray(data["coords"],dtype=np.float64)); idx, tri=geometry.valid_offset_triples(distances,codes,chains)
            split="test" if entry.master_role=="test" else "development"
            for code in ("H","E"):
                chosen=tri[codes[idx]==code]
                if len(chosen): triples[(split,code)].append(chosen)
            eligible_bridges=[b for b in bridges if b["main_eligible"]]
            bridge_by_id={b["bridge_id"]:b for b in eligible_bridges}
            for b in eligible_bridges: sheet[(split,b["orientation"],"partner")].append(distances[b["i"],b["j"]])
            for c in controls:
                b=bridge_by_id.get(c["bridge_id"])
                if b: sheet[(split,b["orientation"],c["type"])].append(distances[c["i"],c["j"]])
            controls_by_bridge=defaultdict(lambda:defaultdict(list))
            for c in controls:
                if c["bridge_id"] in bridge_by_id:
                    controls_by_bridge[c["bridge_id"]][c["type"]].append(distances[c["i"],c["j"]])
            for bridge_id,b in bridge_by_id.items():
                partner_distance=distances[b["i"],b["j"]]
                for contrast in CONTRASTS:
                    values=controls_by_bridge[bridge_id][contrast]
                    if values: sheet_margins[(split,b["orientation"],contrast)].append(finite_mean(values)-partner_distance)
            counts=Counter(codes.tolist()); adjacent=np.linalg.norm(np.diff(data["coords"],axis=0),axis=1)
            contiguous=[i for i in range(len(residues)-1) if residues[i]["chain"]==residues[i+1]["chain"] and residues[i+1]["pdb_resseq"]==residues[i]["pdb_resseq"]+1]
            dssp_rows.append({"protein_id":entry.protein_id,"source":entry.source,"split":split,"status":"success","dssp_exit_status":exit_status,"n_pair_residues":entry.n_residues,"n_pdb_ca_residues":len(residues),"n_dssp_records":len(dssp),"n_dssp_mapped":mapped,"n_dssp_H":counts["H"],"n_dssp_E":counts["E"],"n_dssp_coil":counts["-"],"helix_fraction":counts["H"]/entry.n_residues,"strand_fraction":counts["E"]/entry.n_residues,"coil_fraction":counts["-"]/entry.n_residues,"n_helix_segments":len(hseg),"n_strand_segments":len(eseg),"helix_segment_lengths":";".join(str(s["length"]) for s in hseg),"strand_segment_lengths":";".join(str(s["length"]) for s in eseg),"n_bridge_pairs":len(eligible_bridges),"n_parallel_bridge_pairs":sum(b["orientation"]=="parallel" for b in eligible_bridges),"n_antiparallel_bridge_pairs":sum(b["orientation"]=="antiparallel" for b in eligible_bridges),"n_chain_transitions":sum(chains[i]!=chains[i+1] for i in range(len(chains)-1)),"n_numbering_gaps":sum(residues[i]["chain"]==residues[i+1]["chain"] and residues[i+1]["pdb_resseq"]!=residues[i]["pdb_resseq"]+1 for i in range(len(residues)-1)),"n_adjacent_ca_deviations":sum(not 3.6<=adjacent[i]<=4.0 for i in contiguous),"mean_adjacent_ca_distance":finite_mean(adjacent[i] for i in contiguous),"max_adjacent_ca_distance":max((adjacent[i] for i in contiguous),default=math.nan),"pdb_path":str(pdb_path),"pair_block_0_path":str(pair_path),"details":stderr})
            dssp_rows[-1].update({f"n_dssp_{code if code != '-' else 'coil'}":counts[code] for code in geometry.DSSP_CODES})
            suspect=[]
            if counts["H"]==0 and counts["E"]==0: suspect.append("zero_H_and_E")
            if counts["-"]/entry.n_residues>0.9: suspect.append("coil_fraction_above_0.9")
            if any(b["orientation"]=="unknown" for b in bridges): suspect.append("unknown_bridge_orientation")
            if dssp_rows[-1]["n_adjacent_ca_deviations"]: suspect.append("adjacent_ca_distance_outside_3.6_4.0")
            if suspect:
                dssp_rows[-1]["status"]="suspect"; dssp_rows[-1]["suspect_reasons"]=";".join(suspect)
                dssp_flags.append({"stage":"prepare","model_family":"","layer":"","protein_id":entry.protein_id,"source":entry.source,"failure_type":"dssp_validation_flag","reason":";".join(suspect),"traceback_summary":"","fatal":0})
        except Exception as exc:
            failures.append({"stage":"prepare","model_family":"","layer":"","protein_id":entry.protein_id,"source":entry.source,"failure_type":type(exc).__name__,"reason":str(exc),"traceback_summary":traceback.format_exc(limit=2),"fatal":1})
        if number % 25 == 0: print(f"[{utcnow()}] prepared {number}/{len(entries)}",flush=True)
    atomic_csv(root/"metadata/dssp_validation.csv",dssp_rows); atomic_csv(root/"metadata/dssp_failures.csv",dssp_flags+failures,FAILURE_FIELDS)
    atomic_csv(root/"metadata/input_artifacts.csv",artifacts,ARTIFACT_FIELDS)
    atomic_csv(root/"metadata/split_assignments.csv",[e.semantic_record() for e in entries])
    if failures: raise RuntimeError(f"Preparation failed for {len(failures)} proteins; see metadata/dssp_failures.csv")
    validations.extend([
        {"category":"cohort","metric":"eligible_proteins_prepared","model_family":"all","layer":"","split":"all","orientation":"","value":len(dssp_rows),"n":len(entries),"threshold":len(entries),"status":"pass" if len(dssp_rows)==len(entries) else "fail","details":"no silent cohort loss"},
        {"category":"indexing","metric":"per_protein_split_digests_valid","model_family":"per_protein_ols","layer":"","split":"all","orientation":"","value":len(entries),"n":len(entries),"threshold":len(entries),"status":"pass","details":f"seed={WITHIN_PROTEIN_SPLIT_SEED}; counts and digests reproduced"},
        {"category":"dssp","metric":"complete_residue_mappings","model_family":"","layer":"","split":"all","orientation":"","value":sum(int(r["n_dssp_mapped"])==int(r["n_pair_residues"]) for r in dssp_rows),"n":len(dssp_rows),"threshold":len(entries),"status":"pass","details":f"{len(dssp_flags)} advisory flags retained separately"},
    ])
    stack=lambda split,code: np.vstack(triples[(split,code)])
    dh,de,th,te=stack("development","H"),stack("development","E"),stack("test","H"),stack("test","E")
    combined=np.vstack((dh,de)); mean=combined.mean(0); scale=combined.std(0); scale[scale<1e-12]=1
    muh,mue=dh.mean(0),de.mean(0)
    calibration={"zscore_mean":mean.tolist(),"zscore_scale":scale.tolist(),"mu_helix_raw":muh.tolist(),"mu_strand_raw":mue.tolist(),"mu_helix_z":((muh-mean)/scale).tolist(),"mu_strand_z":((mue-mean)/scale).tolist()}
    dev_es=geometry.score_triples(de,calibration)[0]; dev_h_close=np.linalg.norm(dh-muh,axis=1)
    calibration["helix_score_threshold_95_specificity"]=geometry.empirical_quantile(dev_es,.95,"higher")
    calibration["helix_closeness_threshold_95_sensitivity"]=geometry.empirical_quantile(dev_h_close,.95,"higher")
    random_dev=np.asarray([v for (s,o,k),vals in sheet.items() if s=="development" and k=="matched_random" for v in vals])
    calibration["sheet_partner_distance_threshold_95_specificity"]=geometry.empirical_quantile(random_dev,.05,"lower")
    test_scores=np.r_[geometry.score_triples(th,calibration)[0],geometry.score_triples(te,calibration)[0]]
    test_auc=safe_auc(np.r_[np.ones(len(th)),np.zeros(len(te))],test_scores)
    validations.append({"category":"helix_score","metric":"true_matrix_auroc","model_family":"true_distance","layer":"","split":"test","orientation":"","value":test_auc,"n":len(test_scores),"threshold":MIN_TRUE_HELIX_AUROC,"status":"pass" if test_auc>=MIN_TRUE_HELIX_AUROC else "fail","details":"H versus E"})
    validations.extend([
        {"category":"helix_gate","metric":"score_threshold","model_family":"true_distance","layer":"","split":"development","orientation":"","value":calibration["helix_score_threshold_95_specificity"],"n":len(de),"threshold":0.95,"status":"calibrated","details":"empirical 95th percentile of development E scores"},
        {"category":"helix_gate","metric":"closeness_threshold","model_family":"true_distance","layer":"","split":"development","orientation":"","value":calibration["helix_closeness_threshold_95_sensitivity"],"n":len(dh),"threshold":0.95,"status":"calibrated","details":"empirical 95th percentile of development H raw closeness"},
        {"category":"sheet_gate","metric":"partner_distance_threshold","model_family":"true_distance","layer":"","split":"development","orientation":"all","value":calibration["sheet_partner_distance_threshold_95_specificity"],"n":len(random_dev),"threshold":0.05,"status":"calibrated","details":"empirical fifth percentile of development matched-random distances"},
    ])
    for orientation in ("parallel","antiparallel"):
        for split in ("development","test"):
            for contrast in CONTRASTS:
                partner=sheet[(split,orientation,"partner")]; control=sheet[(split,orientation,contrast)]
                validations.extend([
                    {"category":"beta_true_distance","metric":f"{contrast}_partner_mean","model_family":"true_distance","layer":"","split":split,"orientation":orientation,"value":finite_mean(partner),"n":len(partner),"threshold":"","status":"check","details":"true-distance ceiling"},
                    {"category":"beta_true_distance","metric":f"{contrast}_control_mean","model_family":"true_distance","layer":"","split":split,"orientation":orientation,"value":finite_mean(control),"n":len(control),"threshold":"","status":"check","details":"true-distance ceiling"},
                    {"category":"beta_true_distance","metric":f"{contrast}_auroc","model_family":"true_distance","layer":"","split":split,"orientation":orientation,"value":safe_auc(np.r_[np.ones(len(partner)),np.zeros(len(control))],-np.r_[partner,control]),"n":len(partner)+len(control),"threshold":"","status":"check","details":"true-distance ceiling"},
                    {"category":"beta_true_distance","metric":f"{contrast}_within_bridge_margin","model_family":"true_distance","layer":"","split":split,"orientation":orientation,"value":finite_mean(sheet_margins[(split,orientation,contrast)]),"n":len(sheet_margins[(split,orientation,contrast)]),"threshold":"","status":"check","details":"mean control minus partner"},
                    {"category":"beta_true_distance","metric":f"{contrast}_paired_correct_fraction","model_family":"true_distance","layer":"","split":split,"orientation":orientation,"value":finite_mean(float(v>0) for v in sheet_margins[(split,orientation,contrast)]),"n":len(sheet_margins[(split,orientation,contrast)]),"threshold":"","status":"check","details":"fraction of positive within-bridge margins"},
                ])
    atomic_json(root/"calibration.json",calibration); atomic_csv(root/"validation.csv",validations,VALIDATION_FIELDS)
    if not math.isfinite(test_auc) or test_auc<MIN_TRUE_HELIX_AUROC: raise RuntimeError(f"True-distance locked-test helix AUROC {test_auc:.6f} is below {MIN_TRUE_HELIX_AUROC}")
    unique_artifacts={r["artifact_path"]:r["sha256"] for r in artifacts}
    manifest={"analysis_name":ANALYSIS_NAME,"schema_version":SCHEMA_VERSION,"status":"prepared","master_manifest_digest":MASTER_MANIFEST_DIGEST,"model_roots":{k:str(v) for k,v in roots.items()},"models":MODELS,"layers":list(LAYERS),"evaluation_proteins":[e.protein_id for e in test],"evaluation_scopes":SCOPES,"feature_definition":"float64 [Z_ij|Z_ji] for i>j","random_seed":RANDOM_SEED,"matched_random_per_bridge":geometry.N_MATCHED_RANDOM,"dssp_executable":str(executable),"dssp_version":geometry.dssp_version(executable),"calibration_sha256":sha256_file(root/"calibration.json"),"preparation_artifact_table_sha256":sha256_file(root/"metadata/input_artifacts.csv"),"input_artifact_hashes":unique_artifacts,"implementation_revision":code_revision(),"created_at":utcnow(),"versions":package_versions()}
    atomic_json(root/"run_manifest.json",manifest)
    atomic_csv(root/"runtime.csv",[{"stage":"prepare","layer":"","started_at":started_at,"ended_at":utcnow(),"elapsed_seconds":time.time()-started,"n_proteins":len(entries),"n_selected_pairs":prepared_pair_count,"model_prediction_seconds":0,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"status":"complete"}],RUNTIME_FIELDS)
    atomic_json(root/"preparation_complete.json",{"status":"complete","configuration_digest":hashlib.sha256(json.dumps(manifest,sort_keys=True).encode()).hexdigest(),"elapsed_seconds":time.time()-started})


def _lookup(cache, predictions):
    return {(int(i),int(j)):float(v) for i,j,v in zip(cache["selected_pair_i"],cache["selected_pair_j"],predictions)}


def _pair_value(lookup,i,j): return lookup[(min(i,j),max(i,j))]


def build_rows(model, layer, scope, entry, cache, predicted, calibration):
    held={tuple((int(i),int(j))):bool(h) for i,j,h in zip(cache["selected_pair_i"],cache["selected_pair_j"],cache["selected_pair_heldout"])}
    lookup=_lookup(cache,predicted); coords=cache["coords"]; true=geometry.distance_matrix(coords); helix=[]; segments=[]; bridges=[]
    for i,code in enumerate(cache["dssp_codes"].astype(str)):
        if code not in {"H","E"} or i+4>=len(cache["chains"]) or not all(cache["chains"][i]==cache["chains"][i+o] for o in (2,3,4)): continue
        pairs=[(min(i,i+o),max(i,i+o)) for o in (2,3,4)]; all_held=all(held[p] for p in pairs)
        if scope==SCOPES[1] and not all_held: continue
        decoded=np.asarray([_pair_value(lookup,*p) for p in pairs]); target=np.asarray([true[p] for p in pairs]); score,zh,ze=geometry.score_triples(decoded[None],calibration); raw_close=float(np.linalg.norm(decoded-np.asarray(calibration["mu_helix_raw"])))
        segment=int(cache["h_segment_ids"][i]); seglen=int(cache["h_segment_length"][segment]) if segment>=0 else 0
        if code=="H" and (seglen<HELIX_MIN_SEGMENT_LENGTH or i in {int(cache["h_segment_start"][segment]),int(cache["h_segment_end"][segment])}): continue
        relative=score[0]>calibration["helix_score_threshold_95_specificity"]; absolute=raw_close<=calibration["helix_closeness_threshold_95_sensitivity"]
        helix.append({"model_family":model,"layer":layer,"evaluation_scope":scope,"protein_id":entry.protein_id,"source":entry.source,"residue_index":i,"chain":str(cache["chains"][i]),"pdb_resseq":int(cache["pdb_resseq"][i]),"insertion_code":str(cache["insertion_codes"][i]),"resname":str(cache["resnames"][i]),"dssp_code":code,"segment_id":segment,"segment_length":seglen,"offset_2_decoded":decoded[0],"offset_3_decoded":decoded[1],"offset_4_decoded":decoded[2],"offset_2_true":target[0],"offset_3_true":target[1],"offset_4_true":target[2],"score":score[0],"distance_to_helix_centroid":raw_close,"distance_to_helix_centroid_z":zh[0],"distance_to_strand_centroid_z":ze[0],"raw_triple_rmse":float(np.sqrt(np.mean((decoded-target)**2))),"passes_relative_gate":int(relative),"passes_absolute_gate":int(absolute),"looks_helical":int(relative and absolute),"all_three_pairs_heldout":int(all_held)})
    byseg=defaultdict(list)
    for row in helix:
        if row["dssp_code"]=="H": byseg[row["segment_id"]].append(row)
    for sid,start,end,length in zip(cache["h_segment_ids"][cache["h_segment_start"]],cache["h_segment_start"],cache["h_segment_end"],cache["h_segment_length"]):
        if int(length)<HELIX_MIN_SEGMENT_LENGTH: continue
        selected=byseg[int(sid)]; ms=finite_mean(r["score"] for r in selected); mc=finite_mean(r["distance_to_helix_centroid"] for r in selected)
        segments.append({"model_family":model,"layer":layer,"evaluation_scope":scope,"protein_id":entry.protein_id,"source":entry.source,"segment_id":int(sid),"start_residue_index":int(start),"end_residue_index":int(end),"original_segment_length":int(length),"n_scored_residues":len(selected),"mean_score":ms,"mean_distance_to_helix_centroid":mc,"mean_raw_triple_rmse":finite_mean(r["raw_triple_rmse"] for r in selected),"fraction_residues_looking_helical":finite_mean(r["looks_helical"] for r in selected),"visible":int(bool(selected) and ms>calibration["helix_score_threshold_95_specificity"] and mc<=calibration["helix_closeness_threshold_95_sensitivity"])})
    controls=defaultdict(lambda:defaultdict(list))
    for bid,kind,i,j in zip(cache["control_bridge_id"],cache["control_type"],cache["control_i"],cache["control_j"]): controls[int(bid)][str(kind)].append((int(i),int(j)))
    for bid,i,j,orientation,eligible in zip(cache["bridge_id"],cache["bridge_i"],cache["bridge_j"],cache["bridge_orientation"],cache["bridge_main_eligible"]):
        if not eligible: continue
        bid,i,j=int(bid),int(i),int(j); partner_pair=(min(i,j),max(i,j)); partner_held=held[partner_pair]
        row={"model_family":model,"layer":layer,"evaluation_scope":scope,"protein_id":entry.protein_id,"source":entry.source,"bridge_id":bid,"i":i,"j":j,"orientation":str(orientation),"sequence_separation":abs(i-j),"decoded_partner_distance":_pair_value(lookup,i,j),"true_partner_distance":true[i,j],"partner_pair_heldout":int(partner_held)}
        for contrast in CONTRASTS:
            pairs=[p for p in controls[bid][contrast] if scope==SCOPES[0] or (partner_held and held[(min(p),max(p))])]
            vals=[_pair_value(lookup,*p) for p in pairs]; targets=[true[p] for p in pairs]; key="random" if contrast=="matched_random" else contrast
            count_field = "n_matched_random" if contrast == "matched_random" else f"n_{key}"
            row.update({count_field:len(vals),f"mean_decoded_{key}_distance":finite_mean(vals),f"mean_true_{key}_distance":finite_mean(targets),f"{contrast}_margin":finite_mean(vals)-row["decoded_partner_distance"],f"{contrast}_paired_correct":int(bool(vals) and row["decoded_partner_distance"]<finite_mean(vals)) if vals else math.nan,f"_{contrast}_values":vals})
        row["visible"]=int(partner_held or scope==SCOPES[0]) and int(row["decoded_partner_distance"]<=calibration["sheet_partner_distance_threshold_95_specificity"]); bridges.append(row)
    return helix,segments,bridges


def summarize(model,layer,scope,protein_rows,helix,segments,bridges):
    h=[r for r in helix if r["dssp_code"]=="H"]; e=[r for r in helix if r["dssp_code"]=="E"]
    row={"analysis_name":ANALYSIS_NAME,"model_family":model,"layer":layer,"evaluation_scope":scope,"split":"test","n_proteins":len(protein_rows),"n_decoded_structure_pairs":sum(r["n_decoded_structure_pairs"] for r in protein_rows),"n_negative_predictions":sum(r["n_negative_predictions"] for r in protein_rows),"negative_prediction_fraction":sum(r["n_negative_predictions"] for r in protein_rows)/max(sum(r["n_decoded_structure_pairs"] for r in protein_rows),1),"n_helix_residues":len(h),"n_strand_residues":len(e),"mean_helix_score":finite_mean(r["score"] for r in h),"mean_strand_score":finite_mean(r["score"] for r in e),"helix_strand_auroc":safe_auc(np.r_[np.ones(len(h)),np.zeros(len(e))],np.r_[[r["score"] for r in h],[r["score"] for r in e]]),"mean_helix_centroid_distance":finite_mean(r["distance_to_helix_centroid"] for r in h),"helix_triple_rmse":float(np.sqrt(finite_mean(r["raw_triple_rmse"]**2 for r in h))),"helix_relative_gate_fraction":finite_mean(r["passes_relative_gate"] for r in h),"helix_absolute_gate_fraction":finite_mean(r["passes_absolute_gate"] for r in h),"helix_both_gates_fraction":finite_mean(r["looks_helical"] for r in h),"n_helix_segments_eligible":len(segments),"n_helix_segments_visible":sum(r["visible"] for r in segments),"helix_segment_visible_fraction":finite_mean(r["visible"] for r in segments)}
    for o in ORIENTATIONS:
        selected=bridges if o=="all" else [r for r in bridges if r["orientation"]==o]
        selected=[r for r in selected if scope==SCOPES[0] or int(r["partner_pair_heldout"])]
        prefix=f"sheet_{o}_"
        row[prefix+"n_bridges"]=len(selected); row[prefix+"n_visible_bridges"]=sum(r["visible"] for r in selected); row[prefix+"visible_bridge_fraction"]=finite_mean(r["visible"] for r in selected); row[prefix+"mean_partner_distance"]=finite_mean(r["decoded_partner_distance"] for r in selected)
        for c in CONTRASTS:
            key="random" if c=="matched_random" else c; controls=[v for r in selected for v in r[f"_{c}_values"]]; partners=[r["decoded_partner_distance"] for r in selected if r[f"_{c}_values"]]
            row[prefix+f"{c}_n_controls"]=len(controls); row[prefix+f"{c}_mean_distance"]=finite_mean(controls); row[prefix+f"{c}_margin"]=finite_mean(r[f"{c}_margin"] for r in selected); row[prefix+f"{c}_auroc"]=safe_auc(np.r_[np.ones(len(partners)),np.zeros(len(controls))],-np.r_[partners,controls]); row[prefix+f"{c}_paired_correct_fraction"]=finite_mean(r[f"{c}_paired_correct"] for r in selected)
    for offset in (2,3,4): row[f"mean_helix_offset_{offset}_decoded"]=finite_mean(r[f"offset_{offset}_decoded"] for r in h); row[f"mean_helix_offset_{offset}_true"]=finite_mean(r[f"offset_{offset}_true"] for r in h)
    return row


def layer_stage(layer):
    started=time.time(); started_at=utcnow(); root=output_root(); complete=root/f"layer_shards/layer_{layer:02d}/complete.json"; shard=complete.parent
    prep=json.loads((root/"preparation_complete.json").read_text()); calibration=json.loads((root/"calibration.json").read_text()); roots=input_roots()
    if complete.is_file() and (shard/"shard_manifest.json").is_file():
        prior=json.loads((shard/"shard_manifest.json").read_text())
        hashes_ok=all((shard/name).is_file() and sha256_file(shard/name)==digest for name,digest in prior.get("files",{}).items())
        marker=json.loads(complete.read_text())
        if prior.get("layer")==layer and prior.get("configuration_digest")==prep["configuration_digest"] and marker.get("manifest_sha256")==sha256_file(shard/"shard_manifest.json") and hashes_ok:
            print(f"Layer {layer:02d} already complete and configuration-matched",flush=True); return
    entries=[e for e in eligible_manifest(load_master_manifest()) if e.master_role=="test"]; ids=[e.protein_id for e in entries]
    shared_i,shared_c,_=load_shared_ols(roots["shared_global_ols"],layer); gated=load_sequence_gated(roots["sequence_gated_ols"],layer); per=load_per_protein(roots["per_protein_ols"],layer,ids); booster,_=load_xgboost(roots["shared_xgboost"],layer)
    records=_dataset_records(); all_per=[]; all_h=[]; all_s=[]; all_b=[]; failures=[]; prediction_seconds=0; selected_total=0
    for entry in entries:
        try:
            with np.load(root/"annotation_cache"/f"{entry.protein_id}.npz",allow_pickle=False) as archive: cache={k:archive[k] for k in archive.files}
            pair_path,_=_resolve_inputs(records[entry.protein_id],layer); block=np.load(pair_path); pairs=list(zip(cache["selected_pair_i"].tolist(),cache["selected_pair_j"].tolist())); features,hi,lo=sparse_features(block,pairs); del block
            tick=time.time(); pi,pc,_,_=per[entry.protein_id]
            predictions={"shared_global_ols":shared_i+features@shared_c,"sequence_gated_ols":route_sequence_predictions(features,hi,lo,cache["chains"],gated),"per_protein_ols":pi+features@pc,"shared_xgboost":predict_xgboost(booster,features)}; prediction_seconds+=time.time()-tick; selected_total+=len(pairs)
            for model,predicted in predictions.items():
                if not np.isfinite(predicted).all(): raise ValueError(f"{model} produced nonfinite predictions")
                for scope in SCOPES:
                    h,s,b=build_rows(model,layer,scope,entry,cache,predicted,calibration); all_h+=h; all_s+=s; all_b+=b
                    n_scope=len(predicted) if scope==SCOPES[0] else int(cache["selected_pair_heldout"].sum())
                    eligible_b=[r for r in b if scope==SCOPES[0] or int(r["partner_pair_heldout"])]
                    prow={"analysis_name":ANALYSIS_NAME,"model_family":model,"layer":layer,"evaluation_scope":scope,"split":"test","protein_id":entry.protein_id,"source":entry.source,"n_residues":entry.n_residues,"pair_scope_provenance":"mixed_within_protein_train_and_heldout" if model=="per_protein_ols" and scope==SCOPES[0] else "matched_per_protein_heldout" if scope==SCOPES[1] else "locked_test_protein","n_decoded_structure_pairs":n_scope,"n_negative_predictions":int((predicted<0).sum()) if scope==SCOPES[0] else int((predicted[cache["selected_pair_heldout"].astype(bool)]<0).sum()),"n_helix_residues":sum(r["dssp_code"]=="H" for r in h),"n_strand_residues":sum(r["dssp_code"]=="E" for r in h),"mean_helix_score":finite_mean(r["score"] for r in h if r["dssp_code"]=="H"),"mean_strand_score":finite_mean(r["score"] for r in h if r["dssp_code"]=="E"),"helix_strand_auroc":safe_auc([int(r["dssp_code"]=="H") for r in h],[r["score"] for r in h]),"mean_helix_centroid_distance":finite_mean(r["distance_to_helix_centroid"] for r in h if r["dssp_code"]=="H"),"helix_triple_rmse":float(np.sqrt(finite_mean(r["raw_triple_rmse"]**2 for r in h if r["dssp_code"]=="H"))),"n_helix_segments_eligible":len(s),"n_helix_segments_visible":sum(r["visible"] for r in s),"n_bridges":len(eligible_b),"n_parallel_bridges":sum(r["orientation"]=="parallel" for r in eligible_b),"n_antiparallel_bridges":sum(r["orientation"]=="antiparallel" for r in eligible_b),"prediction_status":"complete"}; all_per.append(prow)
        except Exception as exc:
            failures.append({"stage":"layer","model_family":"all","layer":layer,"protein_id":entry.protein_id,"source":entry.source,"failure_type":type(exc).__name__,"reason":str(exc),"traceback_summary":traceback.format_exc(limit=2),"fatal":1})
    atomic_csv(shard/"failures.csv",failures,FAILURE_FIELDS)
    if failures or len(all_per)!=len(entries)*len(MODELS)*len(SCOPES): raise RuntimeError(f"Layer {layer} incomplete: {len(failures)} failures, {len(all_per)} rows")
    summaries=[summarize(m,layer,s,[r for r in all_per if r["model_family"]==m and r["evaluation_scope"]==s],[r for r in all_h if r["model_family"]==m and r["evaluation_scope"]==s],[r for r in all_s if r["model_family"]==m and r["evaluation_scope"]==s],[r for r in all_b if r["model_family"]==m and r["evaluation_scope"]==s]) for m in MODELS for s in SCOPES]
    for rows in (all_b,):
        for r in rows:
            for k in list(r):
                if k.startswith("_"): del r[k]
    atomic_csv(shard/"layer_summary.csv",summaries); atomic_csv(shard/"per_protein.csv",all_per); atomic_csv(shard/"helix_residue_scores.csv.gz",all_h,gz=True); atomic_csv(shard/"helix_segment_rows.csv.gz",all_s,gz=True); atomic_csv(shard/"bridge_pair_rows.csv.gz",all_b,gz=True)
    runtime={"stage":"layer","layer":layer,"started_at":started_at,"ended_at":utcnow(),"elapsed_seconds":time.time()-started,"n_proteins":len(entries),"n_selected_pairs":selected_total,"model_prediction_seconds":prediction_seconds,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"status":"complete"}; atomic_csv(shard/"runtime.csv",[runtime],RUNTIME_FIELDS)
    manifest={"analysis_name":ANALYSIS_NAME,"layer":layer,"configuration_digest":prep["configuration_digest"],"files":{p.name:sha256_file(p) for p in shard.iterdir() if p.is_file() and p.name not in {"complete.json","shard_manifest.json"}}}; atomic_json(shard/"shard_manifest.json",manifest); atomic_json(complete,{"status":"complete","layer":layer,"manifest_sha256":sha256_file(shard/"shard_manifest.json")})


def trajectory_fields(rows, value_field):
    grouped=defaultdict(list)
    keys=("model_family","evaluation_scope","protein_id","segment_id" if "segment_id" in rows[0] else "bridge_id") if rows else ()
    for r in rows: grouped[tuple(r[k] for k in keys)].append(r)
    for selected in grouped.values():
        selected.sort(key=lambda r:int(r["layer"])); visible=np.asarray([int(r["visible"]) for r in selected]); values=np.asarray([float(r[value_field]) for r in selected]); finite=values[np.isfinite(values)]
        if len(selected)!=48 or [int(r["layer"]) for r in selected]!=list(LAYERS): raise ValueError("Trajectory lacks exactly layers 0..47")
        first=next((i for i,v in enumerate(visible) if v),""); transitions=int(np.sum(visible[1:]!=visible[:-1])); longest=max((len(list(g)) for v,g in itertools.groupby(visible) if v),default=0); reverted=int(any(visible[:i].any() and not visible[i] for i in range(len(visible))))
        common={"first_visible_layer":first,"ever_visible":int(visible.any()),"remains_visible_after_first":int(first!="" and visible[first:].all()),"crossed_then_reverted":reverted,"n_visibility_transitions":transitions,"longest_visible_run":longest,"final_layer_visible":int(visible[-1]),"visibility_auc":float(visible.mean())}
        monotone=int(len(finite)==len(values) and np.all(np.diff(values)>=-1e-12 if value_field=="mean_score" else np.diff(values)<=1e-12))
        closeness=np.asarray([float(r.get("mean_distance_to_helix_centroid", math.nan)) for r in selected]) if value_field=="mean_score" else None
        closeness_monotone=int(closeness is not None and np.isfinite(closeness).all() and np.all(np.diff(closeness)<=1e-12))
        for r in selected:
            r.update(common); r["score_monotone_nondecreasing" if value_field=="mean_score" else "partner_distance_monotone_nonincreasing"]=monotone
            if value_field=="mean_score": r["closeness_monotone_nonincreasing"]=closeness_monotone


def _pairwise(summary):
    rows=[]
    identity={"analysis_name","model_family","layer","evaluation_scope","split"}
    bykey={(r["model_family"],r["layer"],r["evaluation_scope"]):r for r in summary}
    for layer in LAYERS:
        for scope in SCOPES:
            for a,b in itertools.combinations(MODELS,2):
                ra=bykey[(a,str(layer),scope)]
                rb=bykey[(b,str(layer),scope)]
                for metric in sorted(set(ra)&set(rb)-identity):
                    try: va,vb=float(ra[metric]),float(rb[metric])
                    except (ValueError,TypeError): continue
                    if metric.startswith("n_") or metric.endswith("_n_controls"): continue
                    orientation=next((o for o in ORIENTATIONS if f"sheet_{o}_" in metric),"")
                    support=f"sheet_{orientation}_n_bridges" if orientation else "n_helix_residues"
                    rows.append({"model_a":a,"model_b":b,"layer":layer,"evaluation_scope":scope,"metric":metric,"orientation":orientation,"value_model_a":va,"value_model_b":vb,"difference_b_minus_a":vb-va,"support_model_a":ra.get(support,""),"support_model_b":rb.get(support,"")})
    return rows


def make_plots(root, summary):
    import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
    plots={"helix_strand_auroc_by_model.png":"helix_strand_auroc","helix_scores_by_model.png":"mean_helix_score","helix_offsets_by_model.png":"mean_helix_offset_3_decoded","helix_segment_visibility_by_model.png":"helix_segment_visible_fraction","sheet_partner_vs_random_auroc_by_model.png":"sheet_all_matched_random_auroc","sheet_register_auroc_by_model.png":"sheet_all_shift_1_auroc","sheet_register_margin_by_model.png":"sheet_all_shift_1_margin","sheet_register_paired_correct_by_model.png":"sheet_all_shift_1_paired_correct_fraction","sheet_visibility_by_model.png":"sheet_all_visible_bridge_fraction","heldout_sensitivity_support.png":"n_helix_residues"}
    root.mkdir(parents=True,exist_ok=True)
    for filename,metric in plots.items():
        fig,ax=plt.subplots(figsize=(8,4.5))
        for model in MODELS:
            chosen=sorted((r for r in summary if r["model_family"]==model and r["evaluation_scope"]==(SCOPES[1] if filename.startswith("heldout") else SCOPES[0])),key=lambda r:int(r["layer"])); ax.plot([int(r["layer"]) for r in chosen],[float(r.get(metric,"nan")) for r in chosen],label=model)
        ax.set(xlabel="Layer",ylabel=metric); ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(root/filename,dpi=180); plt.close(fig)


def consolidate_stage():
    started=time.time(); started_at=utcnow(); root=output_root(); summary=[]; per=[]; helix=[]; segments=[]; bridges=[]; runtimes=read_csv(root/"runtime.csv") if (root/"runtime.csv").is_file() else []; failures=[]
    mapping=(("layer_summary.csv",summary),("per_protein.csv",per),("helix_residue_scores.csv.gz",helix),("helix_segment_rows.csv.gz",segments),("bridge_pair_rows.csv.gz",bridges),("runtime.csv",runtimes),("failures.csv",failures))
    for layer in LAYERS:
        shard=root/f"layer_shards/layer_{layer:02d}"; complete=shard/"complete.json"; manifest=shard/"shard_manifest.json"
        if not complete.is_file() or json.loads(complete.read_text())["manifest_sha256"]!=sha256_file(manifest): raise RuntimeError(f"Layer {layer} has no valid completion marker")
        recorded=json.loads(manifest.read_text())["files"]
        for name,target in mapping:
            path=shard/name
            if sha256_file(path)!=recorded[name]: raise RuntimeError(f"Layer {layer} checksum mismatch for {name}")
            target.extend(read_csv(path))
    if len(summary)!=48*4*2 or len(per)!=48*4*2*111: raise ValueError("Consolidated coverage is incomplete")
    failures.extend(read_csv(root/"metadata/dssp_failures.csv"))
    summary_keys=[(r["model_family"],r["layer"],r["evaluation_scope"]) for r in summary]
    per_keys=[(r["model_family"],r["layer"],r["evaluation_scope"],r["protein_id"]) for r in per]
    if len(summary_keys)!=len(set(summary_keys)) or len(per_keys)!=len(set(per_keys)):
        raise ValueError("Consolidated primary keys are not unique")
    expected_ids=set(json.loads((root/"run_manifest.json").read_text())["evaluation_proteins"])
    for model in MODELS:
        for layer in LAYERS:
            for scope in SCOPES:
                group=[r for r in per if r["model_family"]==model and int(r["layer"])==layer and r["evaluation_scope"]==scope]
                if {r["protein_id"] for r in group}!=expected_ids:
                    raise ValueError(f"Protein coverage mismatch for {model} layer {layer} scope {scope}")
                total=next(r for r in summary if r["model_family"]==model and int(r["layer"])==layer and r["evaluation_scope"]==scope)
                detail_h=[r for r in helix if r["model_family"]==model and int(r["layer"])==layer and r["evaluation_scope"]==scope]
                detail_s=[r for r in segments if r["model_family"]==model and int(r["layer"])==layer and r["evaluation_scope"]==scope]
                if int(total["n_helix_residues"])!=sum(r["dssp_code"]=="H" for r in detail_h) or int(total["n_strand_residues"])!=sum(r["dssp_code"]=="E" for r in detail_h) or int(total["n_helix_segments_eligible"])!=len(detail_s):
                    raise ValueError(f"Detail support does not reconstruct summary for {model} layer {layer} scope {scope}")
    trajectory_fields(segments,"mean_score"); trajectory_fields(bridges,"decoded_partner_distance")
    runtimes.append({"stage":"consolidate","layer":"","started_at":started_at,"ended_at":utcnow(),"elapsed_seconds":time.time()-started,"n_proteins":111,"n_selected_pairs":"","model_prediction_seconds":0,"peak_rss_mb":resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024,"status":"complete"})
    atomic_csv(root/"layer_summary.csv",summary); atomic_csv(root/"model_pairwise_differences.csv",_pairwise(summary)); atomic_csv(root/"per_protein.csv",per); atomic_csv(root/"helix_residue_scores.csv.gz",helix,gz=True); atomic_csv(root/"helix_segment_trajectories.csv.gz",segments,gz=True); atomic_csv(root/"bridge_pair_trajectories.csv.gz",bridges,gz=True); atomic_csv(root/"runtime.csv",runtimes,RUNTIME_FIELDS); atomic_csv(root/"failures.csv",failures,FAILURE_FIELDS); make_plots(root/"visualizations",summary)
    manifest=json.loads((root/"run_manifest.json").read_text()); manifest["status"]="complete"; manifest["completed_layers"]=list(LAYERS); manifest["output_files"]={p.name:{"sha256":sha256_file(p),"rows":sum(1 for _ in read_csv(p))} for p in (root/"layer_summary.csv",root/"model_pairwise_differences.csv",root/"per_protein.csv",root/"helix_residue_scores.csv.gz",root/"helix_segment_trajectories.csv.gz",root/"bridge_pair_trajectories.csv.gz",root/"runtime.csv",root/"failures.csv")}; atomic_json(root/"run_manifest.json",manifest); atomic_json(root/"consolidation_complete.json",{"status":"complete","run_manifest_sha256":sha256_file(root/"run_manifest.json"),"completed_at":utcnow()})


def main():
    stage=os.environ.get("SECONDARY_STRUCTURE_STAGE","").strip().lower()
    if stage=="prepare": prepare_stage()
    elif stage=="layer":
        layer=int(os.environ.get("SLURM_ARRAY_TASK_ID",os.environ.get("SECONDARY_STRUCTURE_LAYER","-1")))
        if layer not in LAYERS: raise ValueError("Layer stage requires SLURM_ARRAY_TASK_ID in 0..47")
        layer_stage(layer)
    elif stage=="consolidate": consolidate_stage()
    else: raise ValueError("SECONDARY_STRUCTURE_STAGE must be prepare, layer, or consolidate")


if __name__ == "__main__": main()

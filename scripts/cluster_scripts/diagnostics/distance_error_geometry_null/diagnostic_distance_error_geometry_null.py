#!/usr/bin/env python3
"""Matched-R2 distance-noise null for layer-wise MDS geometry diagnostics.

The script starts from CA distances in each final OpenFold PDB and adds IID
Gaussian pairwise errors.  For every protein/layer/replicate, the errors are
rescaled after clipping distances at zero so that the perturbed distances have
the same R2 against the PDB distances as the saved linear probe.  Classical MDS
then reconstructs a CA trace, which is compared with the original PDB using the
same Kabsch-aligned TM-score approximation as the existing regression pipeline.

No probe is fitted or replayed.  The null uses per-protein summaries from
per_protein.csv and observed geometry from geometry_metrics.csv.
"""

from __future__ import annotations

import csv
import hashlib
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent.parent
if str(CLUSTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_SCRIPTS_DIR))

from cluster_regression_helpers import (  # noqa: E402
    classical_mds,
    compute_distance_matrix,
    compute_tm_score,
    extract_ca_coordinates,
    superimpose_structures,
)


PROJECT_ROOT = SCRIPT_DIR.parents[3]
ANALYSIS_NAME = "distance_error_geometry_null"
ANALYSIS_SPLIT = "test"
DEFAULT_N_REPLICATES = 20
DEFAULT_RANDOM_SEED = 20260813
N_LAYERS = 48
FLOAT_EPS = 1e-12
R2_MATCH_TOLERANCE = 1e-10
PDB_EXPORT_N_PROTEINS = 10
PDB_EXPORT_LAYERS = (0, 10, 20, 30, 40, 47)

LOCAL_RESULTS_DIR = PROJECT_ROOT / "delta_files" / "general_base256_combined"
LOCAL_PDB_ROOT = PROJECT_ROOT.parent / "protein_data"
LOCAL_OUTPUT_PARENT = (
    PROJECT_ROOT / "proteins" / "multi_protein_results" / "diagnostics"
)

CAMEO_PDB_ROOT = Path("/work/hdd/bekh/thayes/openfold_reps_outputs_sm")
ECOD_PDB_ROOT = Path(
    "/work/hdd/bfna/thayes/ecod/experiments/openfold_reps_outputs_sm"
)

PER_PROTEIN_REQUIRED_FIELDS = {
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "n_pairs",
    "target_variance",
    "per_protein_r2",
    "per_protein_mse",
    "per_protein_rmse",
    "mean_error_pred_minus_true",
}
OBSERVED_METRICS = (
    "tm_score",
    "rmsd",
    "stress",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "neg_eigenvalue_fraction",
)
GEOMETRY_REQUIRED_FIELDS = {
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "n_pairs",
    "per_protein_r2",
    *OBSERVED_METRICS,
}
HIGHER_IS_FAVORABLE = {
    "tm_score": True,
    "rmsd": False,
    "stress": False,
    "frac_top3_eigenvalues": True,
    "n_negative_eigenvalues": False,
    "neg_eigenvalue_fraction": False,
}

NULL_REPLICATE_FIELDS = [
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "n_pairs",
    "replicate",
    "random_seed",
    "saved_probe_r2",
    "saved_probe_mse",
    "saved_probe_rmse",
    "saved_target_variance",
    "pdb_target_variance",
    "target_null_mse",
    "requested_mean_error",
    "effective_mean_error_before_clipping",
    "achieved_mean_error",
    "noise_scale",
    "achieved_mse",
    "achieved_r2",
    *OBSERVED_METRICS,
]

COMPARISON_BASE_FIELDS = [
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "protein_id",
    "n_residues",
    "n_pairs",
    "n_null_replicates",
    "saved_probe_r2",
    "null_mean_achieved_r2",
    "null_max_abs_r2_error",
]
COMPARISON_FIELDS = list(COMPARISON_BASE_FIELDS)
for _metric in OBSERVED_METRICS:
    COMPARISON_FIELDS.extend(
        [
            f"observed_{_metric}",
            f"null_mean_{_metric}",
            f"null_std_{_metric}",
            f"observed_minus_null_{_metric}",
            f"null_tail_p_favorable_{_metric}",
        ]
    )

LAYER_BASE_FIELDS = [
    "variant",
    "feature_dim",
    "split_id",
    "layer",
    "split",
    "n_proteins",
    "n_null_replicates",
    "observed_mean_r2",
    "null_mean_achieved_r2",
    "null_max_abs_r2_error",
]
LAYER_SUMMARY_FIELDS = list(LAYER_BASE_FIELDS)
for _metric in OBSERVED_METRICS:
    LAYER_SUMMARY_FIELDS.extend(
        [
            f"observed_mean_{_metric}",
            f"null_mean_{_metric}",
            f"null_std_layer_mean_{_metric}",
            f"observed_minus_null_mean_{_metric}",
            f"null_tail_p_favorable_{_metric}",
        ]
    )

SKIPPED_FIELDS = ["protein_id", "reason"]
PDB_EXPORT_FIELDS = ["protein_id", "n_residues", "selection_rank"]
MANIFEST_FIELDS = [
    "analysis_name",
    "input_results_dir",
    "per_protein_csv",
    "geometry_metrics_csv",
    "analysis_split",
    "layers",
    "n_replicates",
    "random_seed",
    "noise_model",
    "matching_target",
    "pdb_export_layers",
    "pdb_export_n_proteins",
    "generated_at",
]


def now_timestamp():
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_progress(message):
    print(f"[{now_timestamp()}] {message}", flush=True)


def parse_layers(value):
    if value is None or not value.strip():
        return list(range(N_LAYERS))
    layers = sorted({int(token.strip()) for token in value.split(",") if token.strip()})
    if not layers or layers[0] < 0 or layers[-1] >= N_LAYERS:
        raise ValueError(f"NULL_LAYERS must contain integers from 0 to {N_LAYERS - 1}")
    return layers


def resolve_paths():
    input_results_dir = Path(
        os.environ.get("INPUT_RESULTS_DIR", str(LOCAL_RESULTS_DIR))
    ).expanduser().resolve()
    output_parent = Path(
        os.environ.get("CSV_ROOT", str(LOCAL_OUTPUT_PARENT))
    ).expanduser().resolve()
    return input_results_dir, output_parent / ANALYSIS_NAME


def read_csv(path, required_fields=None):
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        fields = set(reader.fieldnames or [])
        missing = set(required_fields or []) - fields
        if missing:
            raise ValueError(f"{path} is missing required columns: {sorted(missing)}")
        return list(reader)


def atomic_write_csv(path, fieldnames, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = path.with_suffix(path.suffix + ".tmp")
    with temporary_path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    temporary_path.replace(path)


def finite_float(value, label):
    result = float(value)
    if not math.isfinite(result):
        raise ValueError(f"Non-finite {label}: {value!r}")
    return result


def load_analysis_rows(input_results_dir, layers):
    per_path = input_results_dir / "per_protein.csv"
    geometry_path = input_results_dir / "geometry_metrics.csv"
    per_rows = read_csv(per_path, PER_PROTEIN_REQUIRED_FIELDS)
    geometry_rows = read_csv(geometry_path, GEOMETRY_REQUIRED_FIELDS)

    geometry_by_key = {}
    for row in geometry_rows:
        key = (row["split"], row["protein_id"], int(row["layer"]))
        if key in geometry_by_key:
            raise ValueError(f"Duplicate geometry row for key {key}")
        geometry_by_key[key] = row

    selected = []
    seen = set()
    layer_set = set(layers)
    for row in per_rows:
        layer = int(row["layer"])
        if row["split"] != ANALYSIS_SPLIT or layer not in layer_set:
            continue
        key = (row["split"], row["protein_id"], layer)
        if key in seen:
            raise ValueError(f"Duplicate per-protein row for key {key}")
        seen.add(key)
        observed = geometry_by_key.get(key)
        if observed is None:
            raise ValueError(f"No observed geometry row for {key}")

        parsed = dict(row)
        parsed["layer"] = layer
        parsed["feature_dim"] = int(row["feature_dim"])
        parsed["split_id"] = int(row["split_id"])
        parsed["n_residues"] = int(row["n_residues"])
        parsed["n_pairs"] = int(row["n_pairs"])
        for field in (
            "target_variance",
            "per_protein_r2",
            "per_protein_mse",
            "per_protein_rmse",
            "mean_error_pred_minus_true",
        ):
            parsed[field] = finite_float(row[field], field)
        parsed["observed"] = {
            metric: finite_float(observed[metric], f"observed {metric}")
            for metric in OBSERVED_METRICS
        }
        selected.append(parsed)

    if not selected:
        raise ValueError(
            f"No split={ANALYSIS_SPLIT!r} rows for layers {layers} in {per_path}"
        )
    return selected, per_path, geometry_path


def is_ecod_id(protein_id):
    return protein_id.isdigit()


def resolve_pdb_file(protein_id):
    filename = f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    candidates = [
        LOCAL_PDB_ROOT / protein_id / "predicted_pdb" / filename,
    ]

    env_name = "ECOD_PDB_ROOT" if is_ecod_id(protein_id) else "CAMEO_PDB_ROOT"
    default_root = ECOD_PDB_ROOT if is_ecod_id(protein_id) else CAMEO_PDB_ROOT
    root = Path(os.environ.get(env_name, str(default_root))).expanduser()
    candidates.extend(
        [
            root
            / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
            / "predictions"
            / filename,
            root / protein_id / "predicted_pdb" / filename,
        ]
    )
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    return None


def load_pdb_records(rows):
    metadata_by_protein = {}
    for row in rows:
        metadata_by_protein.setdefault(row["protein_id"], row)

    records = {}
    skipped = []
    for protein_id in sorted(metadata_by_protein):
        expected_n = metadata_by_protein[protein_id]["n_residues"]
        expected_pairs = metadata_by_protein[protein_id]["n_pairs"]
        pdb_path = resolve_pdb_file(protein_id)
        if pdb_path is None:
            skipped.append({"protein_id": protein_id, "reason": "predicted PDB not found"})
            continue
        try:
            ca_data = extract_ca_coordinates(pdb_path)
            coordinates = np.asarray(ca_data["coordinates"], dtype=np.float64)
            residues = list(ca_data["residues"])
            if coordinates.shape != (expected_n, 3):
                raise ValueError(
                    f"expected {expected_n} CA atoms, found {coordinates.shape[0]}"
                )
            true_dist = compute_distance_matrix(coordinates).astype(np.float64, copy=False)
            pair_i, pair_j = np.triu_indices(expected_n, k=1)
            if len(pair_i) != expected_pairs:
                raise ValueError(
                    f"expected {expected_pairs} residue pairs, found {len(pair_i)}"
                )
            records[protein_id] = {
                "protein_id": protein_id,
                "pdb_path": pdb_path,
                "coordinates": coordinates,
                "residues": residues,
                "true_dist": true_dist,
                "pair_i": pair_i,
                "pair_j": pair_j,
                "true_pair_dist": true_dist[pair_i, pair_j],
            }
        except Exception as exc:
            skipped.append({"protein_id": protein_id, "reason": str(exc)})

    if not records:
        raise ValueError("No PDB structures could be loaded for the selected probe rows")
    return records, skipped


def stable_replicate_seed(base_seed, protein_id, layer, replicate):
    payload = f"{base_seed}|{protein_id}|{layer}|{replicate}".encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, byteorder="little", signed=False)


def standardized_gaussian(n_values, seed):
    rng = np.random.default_rng(seed)
    values = rng.standard_normal(n_values).astype(np.float64)
    values -= float(np.mean(values))
    scale = float(np.sqrt(np.mean(values * values)))
    if scale <= FLOAT_EPS:
        raise ValueError("Degenerate Gaussian noise draw")
    return values / scale


def perturb_distances_matched_r2(true_dist, target_r2, requested_bias, seed):
    """Return nonnegative distances whose post-clip R2 matches target_r2."""
    true_dist = np.asarray(true_dist, dtype=np.float64)
    n_values = int(true_dist.size)
    if n_values < 2:
        raise ValueError("At least two pair distances are required")
    target_variance = float(np.mean((true_dist - np.mean(true_dist)) ** 2))
    if target_variance <= FLOAT_EPS:
        raise ValueError("PDB pair-distance variance is zero")
    if target_r2 > 1.0 + R2_MATCH_TOLERANCE:
        raise ValueError(f"Cannot match R2 greater than one: {target_r2}")

    target_mse = max((1.0 - target_r2) * target_variance, 0.0)
    max_bias = math.sqrt(target_mse) if target_mse > 0.0 else 0.0
    effective_bias = float(np.clip(requested_bias, -max_bias, max_bias))
    noise = standardized_gaussian(n_values, seed)

    def candidate(noise_scale):
        return np.maximum(true_dist + effective_bias + noise_scale * noise, 0.0)

    def mse_at(noise_scale):
        residual = candidate(noise_scale) - true_dist
        return float(np.mean(residual * residual))

    mse_zero = mse_at(0.0)
    if mse_zero > target_mse + max(R2_MATCH_TOLERANCE * target_variance, 1e-14):
        # Clipping can make an otherwise admissible negative bias too costly.
        effective_bias = 0.0
        mse_zero = 0.0

    if target_mse <= max(R2_MATCH_TOLERANCE * target_variance, 1e-14):
        noise_scale = 0.0
    elif abs(mse_zero - target_mse) <= R2_MATCH_TOLERANCE * target_variance:
        noise_scale = 0.0
    else:
        high = max(math.sqrt(target_mse - mse_zero), math.sqrt(target_mse), 1e-6)
        for _ in range(80):
            if mse_at(high) >= target_mse:
                break
            high *= 2.0
        else:  # pragma: no cover - defensive guard
            raise RuntimeError("Could not bracket a distance-noise scale")

        low = 0.0
        for _ in range(80):
            midpoint = 0.5 * (low + high)
            if mse_at(midpoint) < target_mse:
                low = midpoint
            else:
                high = midpoint
        noise_scale = 0.5 * (low + high)

    perturbed = candidate(noise_scale)
    residual = perturbed - true_dist
    achieved_mse = float(np.mean(residual * residual))
    achieved_r2 = float(1.0 - achieved_mse / target_variance)
    return perturbed, {
        "pdb_target_variance": target_variance,
        "target_null_mse": target_mse,
        "requested_mean_error": float(requested_bias),
        "effective_mean_error_before_clipping": effective_bias,
        "achieved_mean_error": float(np.mean(residual)),
        "noise_scale": float(noise_scale),
        "achieved_mse": achieved_mse,
        "achieved_r2": achieved_r2,
    }


def reconstruct_geometry(true_coords, pred_dist):
    mds_coords, eigenvalues = classical_mds(pred_dist, n_dims=3)
    reflected = np.asarray(mds_coords, dtype=np.float64).copy()
    reflected[:, 2] *= -1.0
    transformed_orig, rmsd_orig = superimpose_structures(true_coords, mds_coords)
    transformed_flip, rmsd_flip = superimpose_structures(true_coords, reflected)
    if rmsd_flip < rmsd_orig:
        transformed = np.asarray(transformed_flip, dtype=np.float64)
        rmsd = float(rmsd_flip)
    else:
        transformed = np.asarray(transformed_orig, dtype=np.float64)
        rmsd = float(rmsd_orig)

    positive = eigenvalues > 0.0
    sum_positive = float(np.sum(eigenvalues[positive])) if np.any(positive) else 1.0
    negative = eigenvalues < 0.0
    n_negative = int(np.count_nonzero(negative))
    neg_fraction = (
        float(abs(np.sum(eigenvalues[negative])) / sum_positive)
        if n_negative
        else 0.0
    )
    frac_top3 = float(np.sum(np.maximum(eigenvalues[:3], 0.0)) / sum_positive)

    reconstructed_dist = compute_distance_matrix(transformed)
    pair_i, pair_j = np.triu_indices(pred_dist.shape[0], k=1)
    source_values = pred_dist[pair_i, pair_j]
    residual = source_values - reconstructed_dist[pair_i, pair_j]
    denominator = float(np.sum(source_values * source_values))
    stress = (
        float(np.sqrt(np.sum(residual * residual) / denominator))
        if denominator > FLOAT_EPS
        else float("nan")
    )
    deviations = np.linalg.norm(true_coords - transformed, axis=1)
    return {
        "transformed_coords": transformed,
        "deviations": deviations,
        "tm_score": compute_tm_score(true_coords, transformed, len(true_coords)),
        "rmsd": rmsd,
        "stress": stress,
        "frac_top3_eigenvalues": frac_top3,
        "n_negative_eigenvalues": n_negative,
        "neg_eigenvalue_fraction": neg_fraction,
    }


def empirical_favorable_tail(null_values, observed_value, higher_is_favorable):
    values = np.asarray(null_values, dtype=np.float64)
    if higher_is_favorable:
        count = int(np.count_nonzero(values >= observed_value))
    else:
        count = int(np.count_nonzero(values <= observed_value))
    return float((count + 1) / (len(values) + 1))


def summarize_protein(row, null_rows):
    result = {
        "variant": row["variant"],
        "feature_dim": row["feature_dim"],
        "split_id": row["split_id"],
        "layer": row["layer"],
        "split": row["split"],
        "protein_id": row["protein_id"],
        "n_residues": row["n_residues"],
        "n_pairs": row["n_pairs"],
        "n_null_replicates": len(null_rows),
        "saved_probe_r2": row["per_protein_r2"],
    }
    achieved_r2 = np.asarray([item["achieved_r2"] for item in null_rows])
    result["null_mean_achieved_r2"] = float(np.mean(achieved_r2))
    result["null_max_abs_r2_error"] = float(
        np.max(np.abs(achieved_r2 - row["per_protein_r2"]))
    )
    for metric in OBSERVED_METRICS:
        observed = row["observed"][metric]
        values = np.asarray([item[metric] for item in null_rows], dtype=np.float64)
        result[f"observed_{metric}"] = observed
        result[f"null_mean_{metric}"] = float(np.mean(values))
        result[f"null_std_{metric}"] = float(np.std(values, ddof=0))
        result[f"observed_minus_null_{metric}"] = float(observed - np.mean(values))
        result[f"null_tail_p_favorable_{metric}"] = empirical_favorable_tail(
            values,
            observed,
            HIGHER_IS_FAVORABLE[metric],
        )
    return result


def summarize_layer(layer_rows, null_rows):
    first = layer_rows[0]
    by_replicate = defaultdict(list)
    for row in null_rows:
        by_replicate[int(row["replicate"])].append(row)
    replicate_ids = sorted(by_replicate)
    result = {
        "variant": first["variant"],
        "feature_dim": first["feature_dim"],
        "split_id": first["split_id"],
        "layer": first["layer"],
        "split": first["split"],
        "n_proteins": len(layer_rows),
        "n_null_replicates": len(replicate_ids),
        "observed_mean_r2": float(np.mean([row["per_protein_r2"] for row in layer_rows])),
        "null_mean_achieved_r2": float(np.mean([row["achieved_r2"] for row in null_rows])),
        "null_max_abs_r2_error": float(
            max(abs(row["achieved_r2"] - row["saved_probe_r2"]) for row in null_rows)
        ),
    }
    for metric in OBSERVED_METRICS:
        observed_mean = float(np.mean([row["observed"][metric] for row in layer_rows]))
        null_layer_means = np.asarray(
            [
                np.mean([row[metric] for row in by_replicate[replicate]])
                for replicate in replicate_ids
            ],
            dtype=np.float64,
        )
        result[f"observed_mean_{metric}"] = observed_mean
        result[f"null_mean_{metric}"] = float(np.mean(null_layer_means))
        result[f"null_std_layer_mean_{metric}"] = float(np.std(null_layer_means, ddof=0))
        result[f"observed_minus_null_mean_{metric}"] = float(
            observed_mean - np.mean(null_layer_means)
        )
        result[f"null_tail_p_favorable_{metric}"] = empirical_favorable_tail(
            null_layer_means,
            observed_mean,
            HIGHER_IS_FAVORABLE[metric],
        )
    return result


def format_atom_line(serial, residue, coord, bfactor):
    chain_id, residue_name, residue_number = residue
    chain_id = (chain_id or "A")[:1]
    safe_bfactor = min(max(float(bfactor), 0.0), 999.99)
    return (
        f"ATOM  {serial:5d}  CA  {residue_name[:3].upper():>3s} {chain_id:1s}"
        f"{int(residue_number):4d}    {coord[0]:8.3f}{coord[1]:8.3f}{coord[2]:8.3f}"
        f"  1.00{safe_bfactor:6.2f}           C\n"
    )


def write_ca_trace_pdb(path, residues, coordinates, deviations, remark):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        handle.write("REMARK CA-only trace for matched-R2 geometry null\n")
        handle.write(f"REMARK {remark}\n")
        previous_chain = None
        chain_connections = []
        for serial, (residue, coord, deviation) in enumerate(
            zip(residues, coordinates, deviations), start=1
        ):
            chain_id = residue[0]
            if previous_chain is not None and chain_id != previous_chain:
                handle.write("TER\n")
            handle.write(format_atom_line(serial, residue, coord, deviation))
            if previous_chain == chain_id:
                chain_connections.append((serial - 1, serial))
            previous_chain = chain_id
        handle.write("TER\n")
        for serial_a, serial_b in chain_connections:
            handle.write(f"CONECT{serial_a:5d}{serial_b:5d}\n")
        handle.write("END\n")


def choose_pdb_exports(pdb_records):
    ordered = sorted(
        pdb_records.values(),
        key=lambda row: (len(row["residues"]), row["protein_id"]),
    )
    n_export = min(PDB_EXPORT_N_PROTEINS, len(ordered))
    if n_export == 0:
        return []
    indices = np.linspace(0, len(ordered) - 1, num=n_export, dtype=int)
    return [ordered[int(index)]["protein_id"] for index in indices]


def make_manifest(input_results_dir, per_path, geometry_path, layers, n_replicates, seed):
    return {
        "analysis_name": ANALYSIS_NAME,
        "input_results_dir": str(input_results_dir),
        "per_protein_csv": str(per_path),
        "geometry_metrics_csv": str(geometry_path),
        "analysis_split": ANALYSIS_SPLIT,
        "layers": ",".join(str(layer) for layer in layers),
        "n_replicates": n_replicates,
        "random_seed": seed,
        "noise_model": "iid_standard_normal_rescaled_after_nonnegative_clipping",
        "matching_target": "per_protein_layer_saved_r2_against_final_pdb_ca_distances",
        "pdb_export_layers": ",".join(str(layer) for layer in PDB_EXPORT_LAYERS),
        "pdb_export_n_proteins": PDB_EXPORT_N_PROTEINS,
        "generated_at": now_timestamp(),
    }


def validate_or_write_manifest(path, manifest):
    if path.exists():
        existing_rows = read_csv(path)
        if len(existing_rows) != 1:
            raise ValueError(f"Expected one row in existing manifest: {path}")
        existing = existing_rows[0]
        for field in MANIFEST_FIELDS:
            if field == "generated_at":
                continue
            if str(existing.get(field, "")) != str(manifest[field]):
                raise ValueError(
                    f"Existing output uses a different {field}: "
                    f"{existing.get(field)!r} != {manifest[field]!r}. "
                    "Use a different CSV_ROOT or remove the old diagnostic output."
                )
        return existing
    atomic_write_csv(path, MANIFEST_FIELDS, [manifest])
    return manifest


def combine_layer_shards(output_root, layers):
    all_null = []
    all_comparison = []
    all_summary = []
    completed = []
    shard_root = output_root / "layer_shards"
    for layer in layers:
        null_path = shard_root / f"null_replicates_layer_{layer:02d}.csv"
        comparison_path = shard_root / f"per_protein_comparison_layer_{layer:02d}.csv"
        summary_path = shard_root / f"layer_summary_layer_{layer:02d}.csv"
        if not (null_path.exists() and comparison_path.exists() and summary_path.exists()):
            continue
        all_null.extend(read_csv(null_path))
        all_comparison.extend(read_csv(comparison_path))
        all_summary.extend(read_csv(summary_path))
        completed.append(layer)
    atomic_write_csv(output_root / "null_replicates.csv", NULL_REPLICATE_FIELDS, all_null)
    atomic_write_csv(
        output_root / "per_protein_comparison.csv", COMPARISON_FIELDS, all_comparison
    )
    atomic_write_csv(output_root / "layer_summary.csv", LAYER_SUMMARY_FIELDS, all_summary)
    return completed


def write_summary(output_root, manifest, completed_layers, n_available, n_skipped):
    layer_rows = read_csv(output_root / "layer_summary.csv")
    tm_deltas = [float(row["observed_minus_null_mean_tm_score"]) for row in layer_rows]
    stress_deltas = [float(row["observed_minus_null_mean_stress"]) for row in layer_rows]
    path = output_root / "summary.txt"
    temporary_path = path.with_suffix(".txt.tmp")
    with temporary_path.open("w") as handle:
        handle.write("Matched-R2 distance-error geometry null\n")
        handle.write("=======================================\n")
        handle.write(f"Generated: {now_timestamp()}\n")
        handle.write(f"Input: {manifest['input_results_dir']}\n")
        handle.write(f"Split: {ANALYSIS_SPLIT}\n")
        handle.write(f"Requested layers: {manifest['layers']}\n")
        handle.write(f"Completed layers: {','.join(map(str, completed_layers))}\n")
        handle.write(f"Null replicates per protein/layer: {manifest['n_replicates']}\n")
        handle.write(f"Proteins with PDBs: {n_available}\n")
        handle.write(f"Skipped proteins: {n_skipped}\n")
        if tm_deltas:
            handle.write(f"Mean layer TM observed-minus-null: {np.mean(tm_deltas):.6f}\n")
            handle.write(f"Mean layer stress observed-minus-null: {np.mean(stress_deltas):.6f}\n")
        handle.write("\nInterpretation\n")
        handle.write(
            "Positive observed-minus-null TM and negative observed-minus-null stress indicate "
            "that probe errors preserve more 3D structure than IID errors with the same R2.\n"
        )
        handle.write(
            "Near-zero differences support the explanation that marginal distance accuracy "
            "alone accounts for the observed geometry trajectory.\n"
        )
        handle.write("\nLimitations\n")
        handle.write(
            "The saved CSVs retain MSE/R2 and mean bias but not raw residual distributions, "
            "distance-dependent heteroscedasticity, or correlations among residue pairs. "
            "This is therefore an IID Gaussian matched-R2 null. A residual-permutation or "
            "correlation-preserving null requires replaying saved coefficients (no refit) or "
            "rerunning probes while saving pair-level residuals.\n"
        )
        handle.write(
            "TM-score here is the repository's fixed-correspondence, Kabsch-aligned TM-score "
            "calculation, not an external TM-align run. Exported PDBs are CA-only traces.\n"
        )
    temporary_path.replace(path)


def run_layer(layer, layer_rows, pdb_records, n_replicates, base_seed, output_root, export_ids):
    t0 = time.time()
    null_rows = []
    comparison_rows = []
    for protein_index, row in enumerate(layer_rows, start=1):
        record = pdb_records[row["protein_id"]]
        protein_null_rows = []
        for replicate in range(n_replicates):
            seed = stable_replicate_seed(base_seed, row["protein_id"], layer, replicate)
            perturbed_pairs, noise_metrics = perturb_distances_matched_r2(
                record["true_pair_dist"],
                row["per_protein_r2"],
                row["mean_error_pred_minus_true"],
                seed,
            )
            pred_dist = np.zeros_like(record["true_dist"], dtype=np.float64)
            pred_dist[record["pair_i"], record["pair_j"]] = perturbed_pairs
            pred_dist[record["pair_j"], record["pair_i"]] = perturbed_pairs
            geometry = reconstruct_geometry(record["coordinates"], pred_dist)
            null_row = {
                "variant": row["variant"],
                "feature_dim": row["feature_dim"],
                "split_id": row["split_id"],
                "layer": layer,
                "split": row["split"],
                "protein_id": row["protein_id"],
                "n_residues": row["n_residues"],
                "n_pairs": row["n_pairs"],
                "replicate": replicate,
                "random_seed": seed,
                "saved_probe_r2": row["per_protein_r2"],
                "saved_probe_mse": row["per_protein_mse"],
                "saved_probe_rmse": row["per_protein_rmse"],
                "saved_target_variance": row["target_variance"],
                **noise_metrics,
                **{metric: geometry[metric] for metric in OBSERVED_METRICS},
            }
            null_rows.append(null_row)
            protein_null_rows.append(null_row)

            if (
                replicate == 0
                and layer in PDB_EXPORT_LAYERS
                and row["protein_id"] in export_ids
            ):
                pdb_path = (
                    output_root
                    / "pdb_exports"
                    / "null_replicate_000"
                    / f"layer_{layer:02d}"
                    / f"{row['protein_id']}_matched_r2_mds_ca.pdb"
                )
                write_ca_trace_pdb(
                    pdb_path,
                    record["residues"],
                    geometry["transformed_coords"],
                    geometry["deviations"],
                    f"protein={row['protein_id']} layer={layer} replicate=0 "
                    f"matched_R2={noise_metrics['achieved_r2']:.6f}",
                )
        comparison_rows.append(summarize_protein(row, protein_null_rows))
        if protein_index == 1 or protein_index % 10 == 0 or protein_index == len(layer_rows):
            log_progress(
                f"Layer {layer:02d}: completed {protein_index}/{len(layer_rows)} proteins"
            )

    layer_summary = summarize_layer(layer_rows, null_rows)
    shard_root = output_root / "layer_shards"
    atomic_write_csv(
        shard_root / f"null_replicates_layer_{layer:02d}.csv",
        NULL_REPLICATE_FIELDS,
        null_rows,
    )
    atomic_write_csv(
        shard_root / f"per_protein_comparison_layer_{layer:02d}.csv",
        COMPARISON_FIELDS,
        comparison_rows,
    )
    atomic_write_csv(
        shard_root / f"layer_summary_layer_{layer:02d}.csv",
        LAYER_SUMMARY_FIELDS,
        [layer_summary],
    )
    log_progress(f"Layer {layer:02d}: checkpoint written in {time.time() - t0:.1f}s")


def main():
    t0 = time.time()
    layers = parse_layers(os.environ.get("NULL_LAYERS"))
    n_replicates = int(os.environ.get("NULL_N_REPLICATES", DEFAULT_N_REPLICATES))
    base_seed = int(os.environ.get("NULL_RANDOM_SEED", DEFAULT_RANDOM_SEED))
    if n_replicates < 1:
        raise ValueError("NULL_N_REPLICATES must be positive")

    input_results_dir, output_root = resolve_paths()
    output_root.mkdir(parents=True, exist_ok=True)
    log_progress(f"Input results: {input_results_dir}")
    log_progress(f"Output root:   {output_root}")
    log_progress(f"Layers:        {layers}")
    log_progress(f"Replicates:    {n_replicates}")

    analysis_rows, per_path, geometry_path = load_analysis_rows(input_results_dir, layers)
    pdb_records, skipped = load_pdb_records(analysis_rows)
    log_progress(
        f"Loaded {len(pdb_records)} held-out PDBs; skipped {len(skipped)} unavailable PDBs"
    )
    available_ids = set(pdb_records)
    usable_rows = [row for row in analysis_rows if row["protein_id"] in available_ids]
    rows_by_layer = defaultdict(list)
    for row in usable_rows:
        rows_by_layer[row["layer"]].append(row)
    for layer in layers:
        rows_by_layer[layer].sort(key=lambda row: row["protein_id"])
        if not rows_by_layer[layer]:
            raise ValueError(f"No usable proteins remain for layer {layer}")

    manifest = make_manifest(
        input_results_dir, per_path, geometry_path, layers, n_replicates, base_seed
    )
    manifest = validate_or_write_manifest(output_root / "run_manifest.csv", manifest)
    atomic_write_csv(output_root / "skipped_proteins.csv", SKIPPED_FIELDS, skipped)

    export_ids = choose_pdb_exports(pdb_records)
    export_rows = [
        {
            "protein_id": protein_id,
            "n_residues": len(pdb_records[protein_id]["residues"]),
            "selection_rank": rank,
        }
        for rank, protein_id in enumerate(export_ids, start=1)
    ]
    atomic_write_csv(output_root / "pdb_exports.csv", PDB_EXPORT_FIELDS, export_rows)
    for protein_id in export_ids:
        record = pdb_records[protein_id]
        write_ca_trace_pdb(
            output_root / "pdb_exports" / "reference_ca" / f"{protein_id}_reference_ca.pdb",
            record["residues"],
            record["coordinates"],
            np.zeros(len(record["coordinates"]), dtype=np.float64),
            f"protein={protein_id} source={record['pdb_path']}",
        )

    completed = combine_layer_shards(output_root, layers)
    if completed:
        log_progress(f"Resuming with completed layer checkpoints: {completed}")
    for layer in layers:
        if layer in completed:
            continue
        log_progress(f"Layer {layer:02d}: starting matched-R2 null")
        run_layer(
            layer,
            rows_by_layer[layer],
            pdb_records,
            n_replicates,
            base_seed,
            output_root,
            set(export_ids),
        )
        completed = combine_layer_shards(output_root, layers)
        write_summary(output_root, manifest, completed, len(pdb_records), len(skipped))

    completed = combine_layer_shards(output_root, layers)
    write_summary(output_root, manifest, completed, len(pdb_records), len(skipped))
    log_progress(
        f"Finished {len(completed)}/{len(layers)} layers in {time.time() - t0:.1f}s"
    )


if __name__ == "__main__":
    try:
        main()
    except Exception:
        traceback.print_exc()
        raise

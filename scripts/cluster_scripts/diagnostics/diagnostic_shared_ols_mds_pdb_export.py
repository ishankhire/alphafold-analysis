#!/usr/bin/env python3
"""Export selected canonical shared-OLS OOF MDS reconstructions as PDB files.

The protocol-2.0 shared OLS fit already stores one coefficient/scaler artifact
per outer fold and layer.  This diagnostic replays the artifact belonging to
each selected protein's held-out fold, reconstructs a three-dimensional CA
trace with the canonical classical-MDS procedure, and writes aligned CA-only
PDB files.  It never refits a regression or saves pair-level predictions.
"""

from __future__ import annotations

import csv
import os
import platform
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parent
CV_DIR = CLUSTER_DIR / "cross_validation"
for _path in (CLUSTER_DIR, CV_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from cluster_regression_helpers import (  # noqa: E402
    classical_mds,
    compute_distance_matrix,
    compute_tm_score,
    compute_mds_tm_score,
    TM_SCORE_METHOD,
    tmscore_provenance,
    extract_ca_coordinates,
    resolve_combined_pair_block_path,
    resolve_combined_pdb_path,
    superimpose_structures,
)
from cross_validation_common import (  # noqa: E402
    REPLAY_INDEX_FILENAME,
    REPLAY_INDEX_COLUMNS,
    fivefold_result_root,
    fold_layer_task_id,
    load_cv_folds,
    map_fold_layer_task,
)
from regression_output_helpers import (  # noqa: E402
    atomic_write_csv,
    atomic_write_json,
    semantic_digest,
    sha256_file,
)
from regression_protocol import (  # noqa: E402
    FEATURE_DIM,
    extract_directional_base256,
    strict_lower_triangle_indices,
)
from shared_ols_base256_common import load_coefficient_artifact  # noqa: E402


ANALYSIS_NAME = "shared_ols_mds_pdb_export"
SOURCE_ANALYSIS_NAME = "shared_ols_base256"
SOURCE_FAMILY_DIR = "shared_ols_base256"
OUTPUT_DIR_NAME = "shared_ols_mds_pdb_exports"
LAYERS = (0, 10, 20, 30, 40, 47)

# These are the ten rows, in figure order, from
# overleaf/figures/OUTDATED_mds_structures_visualized.png.
PROTEIN_IDS = (
    "7b3a_A",
    "7dkk_A",
    "7dq9_A",
    "7ebt_B",
    "7f6e_B",
    "7kdx_B",
    "7mro_A",
    "7pbk_A",
    "7tbs_A",
    "7tkv_A",
)

METRIC_COLUMNS = (
    "analysis_name",
    "source_analysis_name",
    "layer",
    "protein_id",
    "source",
    "outer_fold",
    "n_residues",
    "n_pairs",
    "r2",
    "mae",
    "rmse",
    "bias_pred_minus_true",
    "mean_predicted_distance",
    "rmsd",
    "tm_score",
    "frac_top3_eigenvalues",
    "n_negative_eigenvalues",
    "negative_eigenvalue_fraction",
    "stress",
    "n_negative_predictions",
    "reflection_selected",
    "model_fit_status",
    "coefficient_artifact_path",
    "coefficient_artifact_sha256",
    "pair_block_path",
    "pdb_source_path",
    "reconstructed_pdb_path",
    "reference_pdb_path",
)

CONFIGURATION = {
    "analysis_name": ANALYSIS_NAME,
    "source_analysis_name": SOURCE_ANALYSIS_NAME,
    "layers": list(LAYERS),
    "protein_ids": list(PROTEIN_IDS),
    "prediction_scope": "heldout_protein_outer_fold_artifact_replay",
    "feature_mode": "directional_base256",
    "feature_dim": FEATURE_DIM,
    "distance_matrix": "symmetric_zero_diagonal",
    "negative_prediction_handling": "count_raw_then_clip_to_zero_for_mds",
    "reconstruction": "classical_mds_top_three_nonnegative_eigenvalues",
    "alignment": "lower_rmsd_of_proper_kabsch_original_or_z_reflection",
    "pdb_contents": "aligned_ca_trace_with_deviation_angstrom_in_b_factor",
    "raw_pair_predictions_saved": False,
}
CONFIGURATION_DIGEST = semantic_digest(CONFIGURATION)


@dataclass(frozen=True)
class ProteinRecord:
    protein_id: str
    source: str
    outer_fold: int
    n_residues: int
    residues: tuple[tuple[str, str, int], ...]
    coordinates: np.ndarray
    pair_i: np.ndarray
    pair_j: np.ndarray
    targets: np.ndarray
    pdb_path: Path


@dataclass(frozen=True)
class Reconstruction:
    coordinates: np.ndarray
    deviations: np.ndarray
    rmsd: float
    tm_score: float
    frac_top3_eigenvalues: float
    n_negative_eigenvalues: int
    negative_eigenvalue_fraction: float
    stress: float
    reflection_selected: bool


def log(message: str) -> None:
    timestamp = datetime.now().astimezone().isoformat(timespec="seconds")
    print(f"[{timestamp}] {message}", flush=True)


def output_root() -> Path:
    override = os.environ.get("MDS_PDB_OUTPUT_ROOT")
    if override:
        return Path(override).expanduser().resolve()
    return fivefold_result_root() / "analysis" / OUTPUT_DIR_NAME


def source_family_root() -> Path:
    return fivefold_result_root() / SOURCE_FAMILY_DIR


def code_revision() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"],
            cwd=SCRIPT_DIR,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return ""


def load_replay_index(family_root: Path) -> dict[tuple[int, int], dict[str, str]]:
    path = family_root / REPLAY_INDEX_FILENAME
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if tuple(reader.fieldnames or ()) != REPLAY_INDEX_COLUMNS:
            raise ValueError(f"Unexpected replay-index schema: {path}")
        rows = list(reader)

    index: dict[tuple[int, int], dict[str, str]] = {}
    for row in rows:
        if (
            row["analysis_name"] != SOURCE_ANALYSIS_NAME
            or row["model_family"] != "shared_ols"
            or row["artifact_type"] != "ols_coefficients_scaler_npz"
            or row["status"] != "complete"
        ):
            raise ValueError(f"Unexpected shared-OLS replay-index row: {row}")
        if row["protein_id"] or row["pair_fold"] or row["sequence_gate"]:
            raise ValueError("Shared-OLS replay index contains model-specific routing fields")
        key = (int(row["outer_fold"]), int(row["layer"]))
        if key in index:
            raise ValueError(f"Duplicate shared-OLS replay-index identity: {key}")
        index[key] = row
    return index


def load_saved_rows(
    path: Path,
    *,
    required_columns: Sequence[str],
) -> dict[tuple[int, str], dict[str, str]]:
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = set(required_columns) - set(reader.fieldnames or ())
        if missing:
            raise ValueError(f"{path} is missing columns: {sorted(missing)}")
        rows = list(reader)
    keyed: dict[tuple[int, str], dict[str, str]] = {}
    for row in rows:
        key = (int(row["layer"]), row["protein_id"])
        if key in keyed:
            raise ValueError(f"Duplicate protein/layer row in {path}: {key}")
        keyed[key] = row
    return keyed


def load_protein_records(fold_bundle) -> dict[str, ProteinRecord]:
    records: dict[str, ProteinRecord] = {}
    for protein_id in PROTEIN_IDS:
        if protein_id not in fold_bundle.outer_by_protein:
            raise ValueError(f"Target protein is absent from the frozen cohort: {protein_id}")
        entry = fold_bundle.outer_by_protein[protein_id]
        if entry.source != "cameo":
            raise ValueError(f"Target protein is not a CAMEO protein: {protein_id}")
        pdb_path = resolve_combined_pdb_path(protein_id, entry.source)
        if not pdb_path.is_file():
            raise FileNotFoundError(f"Missing predicted PDB for {protein_id}: {pdb_path}")
        ca = extract_ca_coordinates(pdb_path)
        coordinates = np.asarray(ca["coordinates"], dtype=np.float64)
        residues = tuple(ca["residues"])
        n_residues = int(entry.n_residues)
        if coordinates.shape != (n_residues, 3) or len(residues) != n_residues:
            raise ValueError(
                f"Frozen residue count mismatch for {protein_id}: "
                f"{coordinates.shape[0]} != {n_residues}"
            )
        pair_i, pair_j = strict_lower_triangle_indices(n_residues)
        targets = np.asarray(
            compute_distance_matrix(coordinates)[pair_i, pair_j], dtype=np.float64
        )
        if targets.size != int(entry.n_pairs) or not np.all(np.isfinite(targets)):
            raise ValueError(f"Invalid target-distance vector for {protein_id}")
        records[protein_id] = ProteinRecord(
            protein_id=protein_id,
            source=str(entry.source),
            outer_fold=int(entry.outer_fold),
            n_residues=n_residues,
            residues=residues,
            coordinates=coordinates,
            pair_i=pair_i,
            pair_j=pair_j,
            targets=targets,
            pdb_path=pdb_path,
        )
    return records


def load_model_artifact(
    family_root: Path,
    replay_index: Mapping[tuple[int, int], Mapping[str, str]],
    *,
    outer_fold: int,
    layer: int,
) -> tuple[dict[str, object], Path, str]:
    key = (int(outer_fold), int(layer))
    if key not in replay_index:
        raise ValueError(f"Replay index has no artifact for fold/layer {key}")
    row = replay_index[key]
    artifact_path = (family_root / row["artifact_path"]).resolve()
    try:
        artifact_path.relative_to(family_root.resolve())
    except ValueError as exc:
        raise ValueError(f"Replay artifact escapes the family directory: {artifact_path}") from exc
    observed_sha256 = sha256_file(artifact_path)
    if observed_sha256 != row["artifact_sha256"]:
        raise ValueError(
            f"Coefficient artifact checksum mismatch for fold/layer {key}: "
            f"{observed_sha256} != {row['artifact_sha256']}"
        )
    task = map_fold_layer_task(fold_layer_task_id(outer_fold, layer))
    artifact = load_coefficient_artifact(
        artifact_path,
        expected_task=task,
        expected_scaler_digest=row["scaler_digest"],
    )
    return artifact, artifact_path, observed_sha256


def predict_distances(pair_block: np.ndarray, record: ProteinRecord, artifact) -> np.ndarray:
    expected_shape = (record.n_residues, record.n_residues, FEATURE_DIM // 2)
    if pair_block.shape != expected_shape:
        raise ValueError(
            f"Pair tensor shape mismatch for {record.protein_id}: "
            f"{pair_block.shape} != {expected_shape}"
        )
    features = extract_directional_base256(pair_block, record.pair_i, record.pair_j)
    values = np.asarray(features, dtype=np.float64)
    del features
    if values.shape != (record.targets.size, FEATURE_DIM) or not np.all(np.isfinite(values)):
        raise ValueError(f"Invalid base256 features for {record.protein_id}")
    predictions = float(artifact["intercept_standardized"]) + (
        (values - np.asarray(artifact["feature_mean"], dtype=np.float64))
        / np.asarray(artifact["feature_scale"], dtype=np.float64)
    ) @ np.asarray(artifact["coefficient_standardized"], dtype=np.float64)
    predictions = np.asarray(predictions, dtype=np.float64)
    if predictions.shape != record.targets.shape or not np.all(np.isfinite(predictions)):
        raise ValueError(f"Nonfinite shared-OLS predictions for {record.protein_id}")
    return predictions


def regression_metrics(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    residual = (
        np.asarray(predictions, dtype=np.float64)
        - np.asarray(targets, dtype=np.float64)
    )
    centered = targets - float(np.mean(targets, dtype=np.float64))
    denominator = float(np.sum(centered * centered, dtype=np.float64))
    return {
        "r2": float(1.0 - np.sum(residual * residual, dtype=np.float64) / denominator),
        "mae": float(np.mean(np.abs(residual), dtype=np.float64)),
        "rmse": float(np.sqrt(np.mean(residual * residual, dtype=np.float64))),
        "bias_pred_minus_true": float(np.mean(residual, dtype=np.float64)),
        "mean_predicted_distance": float(np.mean(predictions, dtype=np.float64)),
    }


def reconstruct_mds_coordinates(
    n_residues: int,
    true_coordinates: np.ndarray,
    pair_i: np.ndarray,
    pair_j: np.ndarray,
    predictions: np.ndarray,
) -> tuple[Reconstruction, int]:
    pred_dist = np.zeros((n_residues, n_residues), dtype=np.float64)
    pred_dist[pair_i, pair_j] = predictions
    pred_dist[pair_j, pair_i] = predictions
    n_negative_predictions = int(np.count_nonzero(predictions < 0.0))
    np.maximum(pred_dist, 0.0, out=pred_dist)
    np.fill_diagonal(pred_dist, 0.0)

    mds_coordinates, eigenvalues = classical_mds(pred_dist, n_dims=3)
    reflected = np.asarray(mds_coordinates, dtype=np.float64).copy()
    reflected[:, 2] *= -1.0
    aligned_original, rmsd_original = superimpose_structures(
        true_coordinates, mds_coordinates
    )
    aligned_reflected, rmsd_reflected = superimpose_structures(
        true_coordinates, reflected
    )
    reflection_selected = bool(rmsd_reflected < rmsd_original)
    if reflection_selected:
        aligned = np.asarray(aligned_reflected, dtype=np.float64)
        rmsd = float(rmsd_reflected)
    else:
        aligned = np.asarray(aligned_original, dtype=np.float64)
        rmsd = float(rmsd_original)

    positive = eigenvalues > 0.0
    positive_mass = float(np.sum(eigenvalues[positive], dtype=np.float64))
    if positive_mass <= 0.0:
        positive_mass = 1.0
    negative = eigenvalues < 0.0
    n_negative_eigenvalues = int(np.count_nonzero(negative))
    negative_fraction = (
        float(abs(np.sum(eigenvalues[negative], dtype=np.float64)) / positive_mass)
        if n_negative_eigenvalues
        else 0.0
    )
    frac_top3 = float(
        np.sum(np.maximum(eigenvalues[:3], 0.0), dtype=np.float64) / positive_mass
    )
    reconstructed_dist = compute_distance_matrix(aligned)
    upper_i, upper_j = np.triu_indices(n_residues, k=1)
    source_values = pred_dist[upper_i, upper_j]
    distance_residual = source_values - reconstructed_dist[upper_i, upper_j]
    stress_denominator = float(
        np.sum(source_values * source_values, dtype=np.float64)
    )
    stress = (
        float(
            np.sqrt(
                np.sum(distance_residual * distance_residual, dtype=np.float64)
                / stress_denominator
            )
        )
        if stress_denominator > 0.0
        else float("nan")
    )
    deviations = np.linalg.norm(true_coordinates - aligned, axis=1)
    reconstruction = Reconstruction(
        coordinates=aligned,
        deviations=np.asarray(deviations, dtype=np.float64),
        rmsd=rmsd,
        tm_score=compute_mds_tm_score(true_coordinates, mds_coordinates, n_residues),
        frac_top3_eigenvalues=frac_top3,
        n_negative_eigenvalues=n_negative_eigenvalues,
        negative_eigenvalue_fraction=negative_fraction,
        stress=stress,
        reflection_selected=reflection_selected,
    )
    return reconstruction, n_negative_predictions


def validate_saved_metrics(
    *,
    layer: int,
    record: ProteinRecord,
    regression: Mapping[str, float],
    reconstruction: Reconstruction,
    n_negative_predictions: int,
    saved_per_protein: Mapping[tuple[int, str], Mapping[str, str]],
    saved_geometry: Mapping[tuple[int, str], Mapping[str, str]],
    saved_tm_score_method: str = "legacy_or_optimized",
) -> None:
    key = (int(layer), record.protein_id)
    if key not in saved_per_protein or key not in saved_geometry:
        raise ValueError(f"Canonical outputs are missing {key}")
    per_row = saved_per_protein[key]
    geometry_row = saved_geometry[key]
    if per_row["prediction_status"] not in {"complete", "complete_from_flagged_fit"}:
        raise ValueError(f"Canonical prediction is incomplete for {key}")
    if geometry_row["geometry_status"] != "complete":
        raise ValueError(f"Canonical geometry is incomplete for {key}")
    for name in (
        "r2",
        "mae",
        "rmse",
        "bias_pred_minus_true",
        "mean_predicted_distance",
    ):
        if not np.isclose(
            regression[name], float(per_row[name]), rtol=1e-8, atol=1e-8
        ):
            raise ValueError(f"Replayed regression metric mismatch for {key}: {name}")
    geometry_values = {
        "rmsd": reconstruction.rmsd,
        "tm_score": reconstruction.tm_score,
        "frac_top3_eigenvalues": reconstruction.frac_top3_eigenvalues,
        "negative_eigenvalue_fraction": reconstruction.negative_eigenvalue_fraction,
        "stress": reconstruction.stress,
    }
    if saved_tm_score_method == "kabsch" or (
        saved_tm_score_method == "legacy_or_optimized" and not np.isclose(
            reconstruction.tm_score, float(geometry_row["tm_score"]), rtol=1e-8, atol=1e-8
        )
    ):
        geometry_values["tm_score"] = compute_tm_score(
            record.coordinates, reconstruction.coordinates, record.n_residues
        )
    elif saved_tm_score_method not in {TM_SCORE_METHOD, "legacy_or_optimized"}:
        raise ValueError(f"Unknown saved TM-score method: {saved_tm_score_method}")
    for name, value in geometry_values.items():
        if not np.isclose(
            value, float(geometry_row[name]), rtol=1e-8, atol=1e-8
        ):
            raise ValueError(f"Replayed geometry metric mismatch for {key}: {name}")
    if reconstruction.n_negative_eigenvalues != int(
        geometry_row["n_negative_eigenvalues"]
    ):
        raise ValueError(f"Replayed negative-eigenvalue count mismatch for {key}")
    if n_negative_predictions != int(geometry_row["n_negative_predictions"]):
        raise ValueError(f"Replayed negative-prediction count mismatch for {key}")


def format_atom_line(
    serial: int,
    residue: tuple[str, str, int],
    coordinate: np.ndarray,
    bfactor: float,
) -> str:
    chain_id, residue_name, residue_number = residue
    safe_chain = (str(chain_id).strip() or "A")[:1]
    safe_bfactor = min(max(float(bfactor), 0.0), 999.99)
    return (
        f"ATOM  {serial:5d}  CA  {str(residue_name)[:3].upper():>3s} {safe_chain:1s}"
        f"{int(residue_number):4d}    {float(coordinate[0]):8.3f}"
        f"{float(coordinate[1]):8.3f}{float(coordinate[2]):8.3f}"
        f"  1.00{safe_bfactor:6.2f}           C\n"
    )


def write_ca_trace_pdb(
    path: Path,
    *,
    residues: Sequence[tuple[str, str, int]],
    coordinates: np.ndarray,
    b_factors: np.ndarray,
    remarks: Sequence[str],
) -> None:
    if (
        not residues
        or len(residues) != len(coordinates)
        or len(residues) != len(b_factors)
    ):
        raise ValueError(
            "PDB residues, coordinates, and B factors must have equal nonzero length"
        )
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent
    )
    try:
        with os.fdopen(descriptor, "w") as handle:
            handle.write("REMARK CA-only trace for canonical shared-OLS MDS replay\n")
            for remark in remarks:
                handle.write(f"REMARK {remark}\n")
            previous_chain = None
            connections: list[tuple[int, int]] = []
            for serial, (residue, coordinate, bfactor) in enumerate(
                zip(residues, coordinates, b_factors), start=1
            ):
                chain_id = str(residue[0])
                if previous_chain is not None and chain_id != previous_chain:
                    handle.write("TER\n")
                handle.write(
                    format_atom_line(serial, residue, coordinate, float(bfactor))
                )
                if previous_chain == chain_id:
                    connections.append((serial - 1, serial))
                previous_chain = chain_id
            handle.write("TER\n")
            for serial_a, serial_b in connections:
                handle.write(f"CONECT{serial_a:5d}{serial_b:5d}\n")
            handle.write("END\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_name, path)
    except Exception:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def write_reference_pdbs(destination: Path, records: Mapping[str, ProteinRecord]) -> None:
    for protein_id in PROTEIN_IDS:
        record = records[protein_id]
        path = destination / "pdb_files" / protein_id / f"{protein_id}_reference_ca.pdb"
        write_ca_trace_pdb(
            path,
            residues=record.residues,
            coordinates=record.coordinates,
            b_factors=np.zeros(record.n_residues, dtype=np.float64),
            remarks=(
                f"protein={protein_id} source=final_predicted_pdb",
                "B-factor=0.00",
            ),
        )


def combine_layer_shards(destination: Path) -> list[dict[str, str]]:
    combined: list[dict[str, str]] = []
    for layer in LAYERS:
        path = destination / "layer_shards" / f"reconstruction_metrics_layer_{layer:02d}.csv"
        with path.open(newline="") as handle:
            reader = csv.DictReader(handle)
            if tuple(reader.fieldnames or ()) != METRIC_COLUMNS:
                raise ValueError(f"Unexpected reconstruction shard schema: {path}")
            rows = list(reader)
        if (
            len(rows) != len(PROTEIN_IDS)
            or {row["protein_id"] for row in rows} != set(PROTEIN_IDS)
        ):
            raise ValueError(f"Incomplete reconstruction shard: {path}")
        if {int(row["layer"]) for row in rows} != {layer}:
            raise ValueError(f"Wrong layer identity in reconstruction shard: {path}")
        combined.extend(rows)
    return combined


def main() -> int:
    started = time.monotonic()
    destination = output_root()
    family_root = source_family_root()
    log(f"Canonical shared-OLS family: {family_root}")
    log(f"Output root: {destination}")
    log(f"Layers: {', '.join(map(str, LAYERS))}")
    log(f"Proteins: {', '.join(PROTEIN_IDS)}")

    fold_bundle = load_cv_folds()
    replay_index = load_replay_index(family_root)
    records = load_protein_records(fold_bundle)
    saved_per_protein = load_saved_rows(
        family_root / "per_protein.csv",
        required_columns=(
            "layer",
            "protein_id",
            "prediction_status",
            "r2",
            "mae",
            "rmse",
            "bias_pred_minus_true",
            "mean_predicted_distance",
        ),
    )
    saved_geometry = load_saved_rows(
        family_root / "geometry_metrics.csv",
        required_columns=(
            "layer",
            "protein_id",
            "geometry_status",
            "rmsd",
            "tm_score",
            "frac_top3_eigenvalues",
            "n_negative_eigenvalues",
            "negative_eigenvalue_fraction",
            "stress",
            "n_negative_predictions",
        ),
    )
    saved_tm_score_method = json.loads((family_root / "run_manifest.json").read_text()).get(
        "tm_score_method", "legacy_or_optimized"
    )
    destination.mkdir(parents=True, exist_ok=True)
    write_reference_pdbs(destination, records)

    artifacts: dict[tuple[int, int], tuple[dict[str, object], Path, str]] = {}
    artifact_provenance: dict[str, dict[str, object]] = {}
    for layer in LAYERS:
        layer_started = time.monotonic()
        layer_rows: list[dict[str, object]] = []
        log(f"Layer {layer:02d}: starting ten-protein replay")
        for protein_id in PROTEIN_IDS:
            record = records[protein_id]
            artifact_key = (record.outer_fold, layer)
            if artifact_key not in artifacts:
                artifacts[artifact_key] = load_model_artifact(
                    family_root,
                    replay_index,
                    outer_fold=record.outer_fold,
                    layer=layer,
                )
            artifact, artifact_path, artifact_sha256 = artifacts[artifact_key]
            pair_block_path = resolve_combined_pair_block_path(
                protein_id, record.source, layer
            )
            if not pair_block_path.is_file():
                raise FileNotFoundError(
                    f"Missing pair tensor for {protein_id} layer {layer}: {pair_block_path}"
                )
            pair_block = np.load(str(pair_block_path))
            try:
                predictions = predict_distances(pair_block, record, artifact)
            finally:
                del pair_block
            regression = regression_metrics(record.targets, predictions)
            reconstruction, n_negative_predictions = reconstruct_mds_coordinates(
                record.n_residues,
                record.coordinates,
                record.pair_i,
                record.pair_j,
                predictions,
            )
            validate_saved_metrics(
                layer=layer,
                record=record,
                regression=regression,
                reconstruction=reconstruction,
                n_negative_predictions=n_negative_predictions,
                saved_per_protein=saved_per_protein,
                saved_geometry=saved_geometry,
                saved_tm_score_method=saved_tm_score_method,
            )

            protein_dir = destination / "pdb_files" / protein_id
            reconstructed_path = (
                protein_dir / f"{protein_id}_shared_ols_mds_layer_{layer:02d}.pdb"
            )
            reference_path = protein_dir / f"{protein_id}_reference_ca.pdb"
            write_ca_trace_pdb(
                reconstructed_path,
                residues=record.residues,
                coordinates=reconstruction.coordinates,
                b_factors=reconstruction.deviations,
                remarks=(
                    f"protein={protein_id} layer={layer} outer_fold={record.outer_fold}",
                    "source=protocol-2.0 heldout shared_ols_base256 prediction",
                    "negative predicted distances clipped to zero before classical MDS",
                    "B-factor=aligned per-residue CA deviation in angstrom",
                ),
            )
            artifact_provenance[f"fold_{record.outer_fold}_layer_{layer:02d}"] = {
                "path": str(artifact_path),
                "sha256": artifact_sha256,
                "fit_status": str(artifact["fit_status"]),
            }
            layer_rows.append(
                {
                    "analysis_name": ANALYSIS_NAME,
                    "source_analysis_name": SOURCE_ANALYSIS_NAME,
                    "layer": layer,
                    "protein_id": protein_id,
                    "source": record.source,
                    "outer_fold": record.outer_fold,
                    "n_residues": record.n_residues,
                    "n_pairs": record.targets.size,
                    **regression,
                    "rmsd": reconstruction.rmsd,
                    "tm_score": reconstruction.tm_score,
                    "frac_top3_eigenvalues": reconstruction.frac_top3_eigenvalues,
                    "n_negative_eigenvalues": reconstruction.n_negative_eigenvalues,
                    "negative_eigenvalue_fraction": (
                        reconstruction.negative_eigenvalue_fraction
                    ),
                    "stress": reconstruction.stress,
                    "n_negative_predictions": n_negative_predictions,
                    "reflection_selected": reconstruction.reflection_selected,
                    "model_fit_status": str(artifact["fit_status"]),
                    "coefficient_artifact_path": str(artifact_path),
                    "coefficient_artifact_sha256": artifact_sha256,
                    "pair_block_path": str(pair_block_path),
                    "pdb_source_path": str(record.pdb_path),
                    "reconstructed_pdb_path": str(reconstructed_path),
                    "reference_pdb_path": str(reference_path),
                }
            )
            log(
                f"Layer {layer:02d} {protein_id}: "
                f"R2={regression['r2']:.4f}, TM={reconstruction.tm_score:.4f}, "
                f"RMSD={reconstruction.rmsd:.3f} A"
            )
        atomic_write_csv(
            destination
            / "layer_shards"
            / f"reconstruction_metrics_layer_{layer:02d}.csv",
            METRIC_COLUMNS,
            layer_rows,
        )
        log(f"Layer {layer:02d}: checkpointed in {time.monotonic() - layer_started:.1f}s")

    combined_rows = combine_layer_shards(destination)
    atomic_write_csv(
        destination / "reconstruction_metrics.csv", METRIC_COLUMNS, combined_rows
    )
    elapsed = time.monotonic() - started
    atomic_write_json(
        destination / "run_manifest.json",
        {
            **CONFIGURATION,
            "configuration_digest": CONFIGURATION_DIGEST,
            "cv_protocol_version": "2.0",
            "cohort_digest": fold_bundle.fold_digests["cohort_digest"],
            "outer_fold_digest": fold_bundle.fold_digests["outer_fold_digest"],
            "source_family_root": str(family_root),
            "output_root": str(destination),
            "reconstruction_metrics": "reconstruction_metrics.csv",
            "layer_shard_layout": "layer_shards/reconstruction_metrics_layer_<layer>.csv",
            "reconstructed_pdb_layout": (
                "pdb_files/<protein_id>/<protein_id>_shared_ols_mds_layer_<layer>.pdb"
            ),
            "reference_pdb_layout": "pdb_files/<protein_id>/<protein_id>_reference_ca.pdb",
            "n_reconstructed_pdb_files": len(LAYERS) * len(PROTEIN_IDS),
            "n_reference_pdb_files": len(PROTEIN_IDS),
            "artifact_provenance": artifact_provenance,
            "python_version": platform.python_version(),
            "numpy_version": np.__version__,
            "code_revision": code_revision(),
            "tm_score_method": TM_SCORE_METHOD,
            "tm_score_backend": tmscore_provenance(),
            "elapsed_seconds": elapsed,
            "completed_at": datetime.now(timezone.utc).isoformat(),
            "run_status": "complete",
        },
    )
    log(
        f"Complete: wrote {len(combined_rows)} reconstructed structures and "
        f"{len(PROTEIN_IDS)} references in {elapsed:.1f}s"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

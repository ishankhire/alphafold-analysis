#!/usr/bin/env python3
"""Similarity between the frozen distance probes across the 48 layers.

Nothing is fitted. Three kinds of similarity are computed for every layer pair:

- weight cosine: coefficient vectors of the shared and sequence-gated linear
  probes, within each outer fold, then averaged over the five folds;
- prediction similarity: Pearson and Spearman correlation between the stored
  out-of-fold predicted distances at two layers, within each protein;
- error similarity: the same correlations on signed error, probe distance minus
  final structure distance.

Prediction and error similarity are computed per protein and summarized across
proteins with equal weight. No value is pooled over residue pairs.

Outputs under ``$PROBE_LAYER_SIMILARITY_OUTPUT_ROOT`` (default
``~/diagnostic_experiments/probe_layer_similarity_v1``):

- ``layer_pair_similarity.csv``: one row per probe, gate, measure and unordered
  layer pair (``layer_a <= layer_b``), with the mean, sd and quartiles across
  proteins (or folds, for weights).
- ``weight_cosine_by_fold.csv``: one row per probe, gate, coefficient space,
  outer fold and unordered layer pair.
- ``weight_fold_noise_floor.csv``: one row per probe, gate, coefficient space
  and layer; cosine between folds at the same layer.
- ``effectiveness_by_layer.csv``: one row per probe, layer and metric; protein
  mean, sd and quartiles of the producer's per-protein R2, MAE and TM-score.
- ``parity_checks.csv``: one row per probe and layer; per-protein MAE recomputed
  here against the producer's ``per_protein.csv``.
- ``per_protein_similarity.npz``: float32 ``[protein, probe, measure, 48, 48]``.
- ``run_manifest.json``.
"""
from __future__ import annotations

import csv
import hashlib
import json
import os
import sys
import time
from multiprocessing import Pool
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SCRIPT_DIR, SCRIPT_DIR.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

N_LAYERS = 48
N_FOLDS = 5
N_FEATURES = 256
PROBES = ("shared_ols_base256", "per_protein_ols_base256",
          "sequence_gated_ols_base256", "shared_xgboost_base256_fixed_s4")
GATES = ("1", "2_to_4", "5_to_9", "10_to_19", "20_to_49", "50_plus")
SPACES = ("standardized", "raw_scale")
PAIR_MEASURES = ("prediction_pearson", "prediction_spearman", "error_pearson", "error_spearman")
EFFECTIVENESS = (("r2", "per_protein.csv", "r2"), ("mae", "per_protein.csv", "mae"),
                 ("tm_score", "geometry_metrics.csv", "tm_score"))
PARITY_ATOL = 1e-3  # float32 predictions against float64 producer MAE, in angstroms
OUTPUT_ENV = "PROBE_LAYER_SIMILARITY_OUTPUT_ROOT"
SUMMARY_COLUMNS = ("probe", "gate", "measure", "averaged_over", "layer_a", "layer_b",
                   "layer_offset", "n", "mean", "sd", "median", "q25", "q75")


def csv_root() -> Path:
    return Path(os.environ.get("CSV_ROOT", "/work/hdd/bfna/ikhire")).expanduser()


def producer_root() -> Path:
    return csv_root() / "regression_5fold_v1"


def output_root() -> Path:
    default = Path.home() / "diagnostic_experiments" / "probe_layer_similarity_v1"
    return Path(os.environ.get(OUTPUT_ENV, default)).expanduser().resolve()


def log(message: str) -> None:
    print(f"[{time.strftime('%FT%TZ', time.gmtime())}] {message}", flush=True)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv(path: Path, columns, rows) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with open(temporary, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)
    temporary.replace(path)


# ---------------------------------------------------------------- similarity

def cosine_matrix(vectors: np.ndarray) -> np.ndarray:
    """Cosine similarity between the rows of ``vectors``."""
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or np.any(norms <= 0):
        raise ValueError("Cosine similarity requires finite, nonzero vectors")
    unit = vectors / norms
    return np.clip(unit @ unit.T, -1.0, 1.0)


def pearson_matrix(rows: np.ndarray) -> np.ndarray:
    """Pearson correlation between rows; NaN where a row has no variance."""
    rows = np.asarray(rows, dtype=np.float64)
    centered = rows - rows.mean(axis=1, keepdims=True)
    norms = np.linalg.norm(centered, axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        unit = centered / norms[:, None]
    result = np.clip(unit @ unit.T, -1.0, 1.0)
    result[norms == 0, :] = np.nan
    result[:, norms == 0] = np.nan
    return result


def spearman_matrix(rows: np.ndarray) -> np.ndarray:
    """Spearman correlation between rows: Pearson on average ranks."""
    from scipy.stats import rankdata
    return pearson_matrix(rankdata(np.asarray(rows, dtype=np.float64), axis=1))


def protein_similarity(predictions: np.ndarray, target: np.ndarray) -> np.ndarray:
    """``[measure, layer, layer]`` for one protein and probe.

    ``predictions`` is ``[layer, pair]`` and ``target`` the final structure
    distance per pair. Measures follow ``PAIR_MEASURES``.
    """
    predictions = np.asarray(predictions, dtype=np.float64)
    errors = predictions - np.asarray(target, dtype=np.float64)[None, :]
    return np.stack([pearson_matrix(predictions), spearman_matrix(predictions),
                     pearson_matrix(errors), spearman_matrix(errors)])


def summarize(values: np.ndarray) -> tuple:
    """n, mean, sd, median, q25, q75 over the finite entries of a 1-D array."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return (0, *(float("nan"),) * 5)
    q25, median, q75 = np.percentile(values, [25, 50, 75])
    sd = float(values.std(ddof=1)) if values.size > 1 else float("nan")
    return (int(values.size), float(values.mean()), sd, float(median), float(q25), float(q75))


def summary_rows(probe, gate, measure, averaged_over, stack):
    """Rows for the upper triangle of ``stack`` = ``[unit, layer, layer]``."""
    for a in range(N_LAYERS):
        for b in range(a, N_LAYERS):
            yield (probe, gate, measure, averaged_over, a, b, b - a, *summarize(stack[:, a, b]))


# ------------------------------------------------------------------- weights

def load_coefficients(family: str, gate: str | None) -> dict[str, np.ndarray]:
    """``{space: [fold, layer, feature]}`` with every artifact checksum verified."""
    root = producer_root() / family
    index = {(int(r["outer_fold"]), int(r["layer"]), r["sequence_gate"]): r
             for r in read_csv(root / "model_artifacts" / "replay_index.csv")}
    gate_label = "" if gate is None else gate.replace("_to_", "-").replace("_plus", "+")
    coefficients = {space: np.empty((N_FOLDS, N_LAYERS, N_FEATURES)) for space in SPACES}
    for fold in range(N_FOLDS):
        for layer in range(N_LAYERS):
            stem = f"fold_{fold}_layer_{layer:02d}.npz"
            relative = f"model_artifacts/coefficients/{stem if gate is None else f'gate_{gate}_{stem}'}"
            row = index[(fold, layer, gate_label)]
            path = root / relative
            if (row["artifact_path"] != relative or row["status"] != "complete"
                    or row["artifact_sha256"] != sha256_file(path)):
                raise ValueError(f"Replay-index mismatch: {path}")
            with np.load(path, allow_pickle=False) as archive:
                if int(archive["outer_fold"]) != fold or int(archive["layer"]) != layer:
                    raise ValueError(f"Artifact identity mismatch: {path}")
                for space in SPACES:
                    values = archive[f"coefficient_{space}"]
                    if values.shape != (N_FEATURES,):
                        raise ValueError(f"Unexpected coefficient shape: {path}")
                    coefficients[space][fold, layer] = values
    return coefficients


def fold_noise_floor(coefficients: np.ndarray) -> np.ndarray:
    """``[layer, fold pair]`` cosine between folds at the same layer."""
    upper = np.triu_indices(coefficients.shape[0], k=1)
    return np.stack([cosine_matrix(coefficients[:, layer])[upper]
                     for layer in range(coefficients.shape[1])])


def weight_tables():
    summary, by_fold, floor = [], [], []
    for family, gate in [("shared_ols_base256", None),
                         *(("sequence_gated_ols_base256", gate) for gate in GATES)]:
        coefficients = load_coefficients(family, gate)
        gate_name = gate or ""
        for space in SPACES:
            stack = np.stack([cosine_matrix(fold) for fold in coefficients[space]])
            summary.extend(summary_rows(family, gate_name, f"weight_cosine_{space}", "outer_fold", stack))
            by_fold.extend((family, gate_name, space, fold, a, b, stack[fold, a, b])
                           for fold in range(N_FOLDS) for a in range(N_LAYERS) for b in range(a, N_LAYERS))
            noise = fold_noise_floor(coefficients[space])
            floor.extend((family, gate_name, space, layer, noise.shape[1], noise[layer].mean(), noise[layer].min())
                         for layer in range(N_LAYERS))
        log(f"weights done: {family} gate={gate_name or '-'}")
    return summary, by_fold, floor


# --------------------------------------------------------------- predictions

def final_structure_distances(pdb_path: Path) -> np.ndarray:
    """Packed Calpha distances in ``np.tril_indices(R, k=-1)`` order."""
    from cluster_regression_helpers import compute_distance_matrix, extract_ca_coordinates
    coordinates = extract_ca_coordinates(pdb_path)["coordinates"]
    distances = compute_distance_matrix(coordinates)
    return distances[np.tril_indices(len(coordinates), k=-1)]


def process_protein(task):
    """Similarity matrices and per-layer MAE for one protein, all probes."""
    protein_id, source, n_pairs, pdb_sha256, prediction_root = task
    from cluster_regression_helpers import resolve_combined_pdb_path
    pdb_path = resolve_combined_pdb_path(protein_id, source)
    if sha256_file(pdb_path) != pdb_sha256:
        raise ValueError(f"Final structure checksum mismatch: {pdb_path}")
    target = final_structure_distances(pdb_path)
    if target.shape != (n_pairs,):
        raise ValueError(f"{protein_id}: {target.shape[0]} pairs in the PDB, expected {n_pairs}")
    similarity = np.empty((len(PROBES), len(PAIR_MEASURES), N_LAYERS, N_LAYERS), dtype=np.float32)
    mae = np.empty((len(PROBES), N_LAYERS))
    predictions = np.empty((N_LAYERS, n_pairs), dtype=np.float32)
    for p, probe in enumerate(PROBES):
        for layer in range(N_LAYERS):
            values = np.load(Path(prediction_root) / "layers" / f"layer_{layer:02d}" / protein_id / f"{probe}.npy",
                             allow_pickle=False)
            if values.shape != (n_pairs,) or not np.all(np.isfinite(values)):
                raise ValueError(f"Bad prediction array: {protein_id} {probe} layer {layer}")
            predictions[layer] = values
        similarity[p] = protein_similarity(predictions, target)
        mae[p] = np.abs(predictions.astype(np.float64) - target[None, :]).mean(axis=1)
    return protein_id, similarity, mae


def producer_metric(family: str, filename: str, column: str, protein_ids) -> np.ndarray:
    """``[protein, layer]`` of one producer per-protein column."""
    position = {pid: i for i, pid in enumerate(protein_ids)}
    values = np.full((len(protein_ids), N_LAYERS), np.nan)
    for row in read_csv(producer_root() / family / filename):
        if row["protein_id"] in position and row[column] not in ("", "NaN", "nan"):
            values[position[row["protein_id"]], int(row["layer"])] = float(row[column])
    return values


def main() -> None:
    started = time.time()
    root = output_root()
    from canonical_storage import assert_writable, deployment_revision
    assert_writable(root)
    root.mkdir(parents=True, exist_ok=True)
    prediction_root = producer_root() / "predicted_distances_5fold_v1"
    proteins = read_csv(prediction_root / "metadata" / "proteins.csv")
    limit = int(os.environ.get("PROBE_LAYER_SIMILARITY_MAX_PROTEINS", "0"))
    if limit:
        proteins = proteins[:limit]
    protein_ids = [row["protein_id"] for row in proteins]
    log(f"output={root} proteins={len(proteins)}")

    summary, by_fold, floor = weight_tables()

    tasks = [(row["protein_id"], row["source"], int(row["n_pairs"]), row["pdb_sha256"], str(prediction_root))
             for row in proteins]
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", "1"))
    similarity = np.empty((len(tasks), len(PROBES), len(PAIR_MEASURES), N_LAYERS, N_LAYERS), dtype=np.float32)
    mae = np.empty((len(tasks), len(PROBES), N_LAYERS))
    position = {pid: i for i, pid in enumerate(protein_ids)}
    with Pool(workers) as pool:
        # Longest proteins first, so the slowest tasks do not land at the end.
        ordered = sorted(tasks, key=lambda task: -task[2])
        for done, (pid, values, protein_mae) in enumerate(pool.imap_unordered(process_protein, ordered), 1):
            similarity[position[pid]], mae[position[pid]] = values, protein_mae
            if done % 50 == 0 or done == len(tasks):
                log(f"proteins done: {done}/{len(tasks)}")

    parity, effectiveness = [], []
    for p, probe in enumerate(PROBES):
        for m, measure in enumerate(PAIR_MEASURES):
            summary.extend(summary_rows(probe, "", measure, "protein", similarity[:, p, m]))
        for metric, filename, column in EFFECTIVENESS:
            values = producer_metric(probe, filename, column, protein_ids)
            effectiveness.extend((probe, layer, metric, *summarize(values[:, layer])) for layer in range(N_LAYERS))
            if metric == "mae":
                difference = np.abs(values - mae[:, p])
                parity.extend((probe, layer, len(protein_ids), float(np.nanmax(difference[:, layer])),
                               "pass" if np.all(difference[:, layer] <= PARITY_ATOL) else "fail")
                              for layer in range(N_LAYERS))
    failed = sum(row[-1] != "pass" for row in parity)
    nonfinite = int(np.sum(~np.isfinite(similarity)))

    write_csv(root / "layer_pair_similarity.csv", SUMMARY_COLUMNS, summary)
    write_csv(root / "weight_cosine_by_fold.csv",
              ("probe", "gate", "coefficient_space", "outer_fold", "layer_a", "layer_b", "cosine"), by_fold)
    write_csv(root / "weight_fold_noise_floor.csv",
              ("probe", "gate", "coefficient_space", "layer", "n_fold_pairs",
               "mean_cross_fold_cosine", "min_cross_fold_cosine"), floor)
    write_csv(root / "effectiveness_by_layer.csv",
              ("probe", "layer", "metric", "n", "mean", "sd", "median", "q25", "q75"), effectiveness)
    write_csv(root / "parity_checks.csv",
              ("probe", "layer", "n_proteins", "max_abs_mae_difference", "status"), parity)
    np.savez_compressed(root / "per_protein_similarity.npz", similarity=similarity,
                        protein_id=np.array(protein_ids), probe=np.array(PROBES),
                        measure=np.array(PAIR_MEASURES))
    status = "complete" if failed == 0 and not limit else ("partial_cohort" if failed == 0 else "parity_failed")
    import scipy
    manifest = {
        "analysis_name": "probe_layer_similarity_v1", "analysis_status": status,
        "code_revision": deployment_revision(SCRIPT_DIR), "n_proteins": len(protein_ids),
        "n_layers": N_LAYERS, "probes": PROBES, "sequence_gates": GATES, "pair_measures": PAIR_MEASURES,
        "aggregation": "per-protein correlation, then equal-weight statistics across proteins; "
                       "weight cosine within outer fold, then statistics across folds",
        "error_definition": "probe distance minus final structure distance",
        "parity": {"checks": len(parity), "failed": failed, "atol_angstrom": PARITY_ATOL,
                   "max_abs_mae_difference": max(row[3] for row in parity)},
        "nonfinite_similarity_values": nonfinite,
        "inputs": {"producer_root": str(producer_root()), "predicted_distances": str(prediction_root)},
        "versions": {"numpy": np.__version__, "scipy": scipy.__version__},
        "elapsed_seconds": round(time.time() - started, 1),
    }
    (root / "run_manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    log(f"status={status} parity_failed={failed} nonfinite={nonfinite}")
    if failed:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

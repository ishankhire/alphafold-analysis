#!/usr/bin/env python3
"""Per-channel classical MDS of the pair representation: one layer, the 154 CAMEO proteins.

Task i handles layer i. For each protein it loads that layer's pair block ``(R, R, 128)`` and,
for every channel, symmetrises it, orients it by the sign with the lower stress, shifts it so
the smallest off-diagonal value is zero and runs classical MDS in three dimensions. The same
is done on one random permutation of the channel's values. No probe is involved; see
``per_channel_mds_common`` for the maths and the README for the design.

Gates before anything is written (a failure writes only ``validation/layer_LL_FAILED.json``):
- the cohort is the 154 CAMEO proteins of the frozen manifest;
- every pair block is finite, ``(R, R, 128)``, with ``R`` equal to the manifest and the PDB;
- on one channel per protein the batched MDS reproduces ``helpers.classical_mds`` with the
  canonical geometry formulas;
- every non-constant channel has a finite stress >= 0 for the real and the permuted matrix.
"""
from __future__ import annotations

import os
import resource
import sys
import time
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_DIR = SCRIPT_DIR.parents[1]
# In the repository the helpers live in sibling folders; on Delta everything is deployed flat.
for _path, _module in ((CLUSTER_DIR / "triangle_violation_subgroups", "triangle_violation_subgroup_common.py"),
                       (CLUSTER_DIR / "null_triangles", "null_triangles_replay.py"),
                       (CLUSTER_DIR, "cluster_regression_helpers.py"),
                       (SCRIPT_DIR, "per_channel_mds_common.py")):
    if (_path / _module).is_file() and str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import cluster_regression_helpers as helpers  # noqa: E402
import null_triangles_replay as replay  # noqa: E402
import per_channel_mds_common as common  # noqa: E402
from canonical_storage import csv_root  # noqa: E402
from null_triangles_replay import atomic_json, atomic_write, log  # noqa: E402
from regression_output_helpers import sha256_file  # noqa: E402
from regression_protocol import N_CHANNELS  # noqa: E402

OUTPUT_ROOT = Path(os.environ.get(
    "PER_CHANNEL_MDS_OUTPUT_ROOT", csv_root() / "diagnostic_experiments" / common.EXPERIMENT_NAME))
# Local smoke tests only: a directory of <protein_id>/{pair_blocks,predicted_pdb}/ and a protein subset.
LOCAL_DATA_ROOT = os.environ.get("PER_CHANNEL_MDS_LOCAL_DATA_ROOT")
if LOCAL_DATA_ROOT:
    helpers.CAMEO_PAIR_BLOCKS_ROOT = Path(LOCAL_DATA_ROOT)
SMOKE = bool(os.environ.get("NULL_TRIANGLES_PROTEINS"))
LAYERS = tuple(range(48))
SOURCE = "cameo"
EXPECTED_PROTEINS = 154
PARITY_TOL = 1e-8
PARITY_CHANNEL = 0
STARTED = time.perf_counter()

SHARD_COLUMNS = ("layer", "protein_id", "source", "n_residues", "n_pairs", "channel", *common.CHANNEL_COLUMNS)


def load_cohort() -> list[dict]:
    """CAMEO rows of the frozen cohort; ``protein_index`` is the position in the full sorted cohort."""
    return [p for p in replay.load_cohort() if p["source"] == SOURCE]


def peak_rss_mib() -> float:
    scale = 1024 ** 2 if sys.platform == "darwin" else 1024
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / scale


def process_protein(task: tuple[int, dict]) -> dict:
    layer, protein = task
    started = time.perf_counter()
    pid, n_residues = protein["protein_id"], protein["n_residues"]
    failures = []
    structure = replay.load_structure(protein)          # raises unless the CA count equals n_residues
    true_distances = helpers.compute_distance_matrix(structure["coordinates"])
    path = helpers.resolve_combined_pair_block_path(pid, protein["source"], layer)
    block = np.load(str(path))
    if block.shape != (n_residues, n_residues, N_CHANNELS) or not np.all(np.isfinite(block)):
        raise ValueError(f"{pid}: pair block {path} has shape {block.shape} or non-finite values")
    loaded = time.perf_counter()
    metrics = common.channel_metrics(block, true_distances, layer, protein["protein_index"])

    # Parity: one channel through helpers.classical_mds and the canonical formulas.
    sign = int(metrics["best_sign"][PARITY_CHANNEL])
    reference = common.single_matrix_metrics(
        common.channel_distance_matrix(block, PARITY_CHANNEL, sign), helpers.classical_mds)
    parity = 0.0
    if metrics["status"][PARITY_CHANNEL] == "ok":
        parity = max(abs(float(metrics[name][PARITY_CHANNEL]) - reference[name]) for name in common.GEOMETRY)
        if not parity <= PARITY_TOL:
            failures.append(f"{pid}: batched MDS differs from helpers.classical_mds by {parity:.3g}")

    ok = metrics["status"] == "ok"
    for name in ("stress", "perm_stress"):
        values = metrics[name][ok]
        if not (np.all(np.isfinite(values)) and np.all(values >= 0.0)):
            failures.append(f"{pid}: {name} is not finite and non-negative on every non-constant channel")
    rows = []
    for channel in range(N_CHANNELS):
        row = {"layer": layer, "protein_id": pid, "source": protein["source"], "n_residues": n_residues,
               "n_pairs": protein["n_pairs"], "channel": channel}
        row.update({name: metrics[name][channel] for name in common.CHANNEL_COLUMNS})
        rows.append(row)
    return {"protein_id": pid, "rows": rows, "failures": failures, "parity": parity,
            "n_constant": int(np.count_nonzero(~ok)), "load_seconds": loaded - started,
            "seconds": time.perf_counter() - started, "peak_rss_mib": peak_rss_mib(),
            "pair_block": str(path), "pdb": structure["pdb"]}


def main() -> None:
    import multiprocessing as mp

    layer = int(os.environ.get("PER_CHANNEL_MDS_LAYER", os.environ.get("SLURM_ARRAY_TASK_ID", "-1")))
    if layer not in LAYERS:
        raise ValueError(f"Layer {layer} is not in 0-47")
    shards, validation_dir = OUTPUT_ROOT / "layer_shards", OUTPUT_ROOT / "validation"
    marker = validation_dir / f"layer_{layer:02d}.json"
    if marker.is_file():
        log(f"layer {layer} already complete ({marker})")
        return
    workers = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count() or 1))
    cohort = load_cohort()
    if not SMOKE and len(cohort) != EXPECTED_PROTEINS:
        raise ValueError(f"Expected {EXPECTED_PROTEINS} CAMEO proteins, found {len(cohort)}")
    log(f"layer {layer}: {len(cohort)} proteins, {workers} workers, output {OUTPUT_ROOT}")

    order = sorted(cohort, key=lambda p: -p["n_residues"])
    results = {}
    with mp.get_context("fork").Pool(workers) as pool:
        for done, result in enumerate(pool.imap_unordered(process_protein, [(layer, p) for p in order],
                                                          chunksize=1), 1):
            results[result["protein_id"]] = result
            if done % 25 == 0 or done == len(order):
                log(f"{done}/{len(order)} proteins (last {result['protein_id']}, {result['seconds']:.1f}s)")
    ordered = [results[p["protein_id"]] for p in cohort]
    rows = [row for r in ordered for row in r["rows"]]
    failures = [f for r in ordered for f in r["failures"]]
    if len(rows) != len(cohort) * N_CHANNELS:
        failures.append(f"{len(rows)} rows for {len(cohort)} proteins")
    gates = {"rows_checked": len(rows), "parity_max_abs_difference": max(r["parity"] for r in ordered),
             "parity_tolerance": PARITY_TOL, "n_constant_channels": sum(r["n_constant"] for r in ordered),
             "failures": failures}
    log(f"gates: {len(rows)} rows, parity {gates['parity_max_abs_difference']:.3g}, "
        f"{gates['n_constant_channels']} constant channels, {len(failures)} failures")
    if failures:
        atomic_json(validation_dir / f"layer_{layer:02d}_FAILED.json", {"layer": layer, "gates": gates})
        raise SystemExit(f"layer {layer}: validation failed; nothing written")

    shard = shards / f"channel_metrics_layer_{layer:02d}.csv.gz"
    atomic_write(shard, SHARD_COLUMNS, rows, True)
    first = cohort[0]
    atomic_json(marker, {
        "layer": layer, "status": "complete", "experiment": common.EXPERIMENT_NAME, "smoke": SMOKE,
        "code_revision": os.environ.get("PER_CHANNEL_MDS_CODE_REVISION", ""),
        "n_proteins": len(cohort), "source": SOURCE,
        "settings": {"n_channels": N_CHANNELS, "n_dims": common.N_DIMS, "seed": common.SEED,
                     "symmetrisation": "mean_of_ij_and_ji", "signs": list(common.SIGNS),
                     "sign_rule": "lower_stress", "shift": "subtract_minimum_off_diagonal",
                     "reconstruction": "classical_mds_top_three_nonnegative_eigenvalues",
                     "n_permutations": 1, "permutation": "upper_triangle_values_within_channel"},
        "gates": gates, "rows": len(rows), "sha256": {shard.name: sha256_file(shard)},
        "seconds": {"total": time.perf_counter() - STARTED,
                    "protein_max": max(r["seconds"] for r in ordered),
                    "protein_sum": sum(r["seconds"] for r in ordered),
                    "load_sum": sum(r["load_seconds"] for r in ordered)},
        "peak_rss_mib": {"parent": peak_rss_mib(), "worker_max": max(r["peak_rss_mib"] for r in ordered)},
        "workers": workers,
        "inputs": {"cohort_csv": str(replay.COHORT_CSV),
                   "cameo_pair_blocks_root": str(helpers.CAMEO_PAIR_BLOCKS_ROOT),
                   "cameo_pdb_root": str(helpers.CAMEO_PDB_ROOT),
                   "example_pair_block": results[first["protein_id"]]["pair_block"],
                   "example_pdb": results[first["protein_id"]]["pdb"]},
        "versions": {"python": sys.version.split()[0], "numpy": np.__version__},
    })
    log(f"layer {layer} complete: {len(rows)} rows")


if __name__ == "__main__":
    main()

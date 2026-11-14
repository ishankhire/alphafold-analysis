"""Checks for the SMACOF error decomposition: SMACOF itself, the four terms, controls, and a layer run."""
from __future__ import annotations

import csv
import gzip
import hashlib
import json
import sys
from pathlib import Path

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "smacof_error_decomposition"
CLUSTER_DIR = SCRIPT_DIR.parents[1]
for path in (CLUSTER_DIR / "triangle_violation_subgroups", CLUSTER_DIR / "null_triangles", CLUSTER_DIR, SCRIPT_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import cluster_regression_helpers as helpers  # noqa: E402
import null_triangles_common as triangles  # noqa: E402
import null_triangles_replay as replay  # noqa: E402
import smacof_error_decomposition as worker  # noqa: E402
import smacof_error_decomposition_common as common  # noqa: E402
import aggregate_smacof_error_decomposition as aggregate  # noqa: E402


def chain(n: int, seed: int) -> np.ndarray:
    """A random walk with 3.8 A steps, rounded as a PDB would store it."""
    rng = np.random.default_rng(seed)
    steps = rng.normal(size=(n, 3))
    steps *= 3.8 / np.linalg.norm(steps, axis=1, keepdims=True)
    return np.round(np.cumsum(steps, axis=0), 3)


def fit(d_hat: np.ndarray, n: int, **kwargs) -> tuple[dict, np.ndarray, np.ndarray]:
    matrix = common.packed_to_matrix(d_hat, n)
    start, _ = helpers.classical_mds(matrix, n_dims=3)
    result = common.smacof_metric(matrix, start, **kwargs)
    return result, common.packed_distances(result["coordinates"]), start


# ---------------------------------------------------------------- the three synthetic cases

def test_true_distances_give_all_zeros():
    coordinates = chain(80, 1)
    d = common.packed_distances(coordinates)
    result, p, _ = fit(d, 80)
    terms = common.decompose(d, p, d)
    assert result["converged"]
    for name in ("total_sq", "inconsistent_sq", "consistent_sq", "cross"):
        assert abs(terms[name]) < 1e-18, name


def test_random_noise_is_almost_all_inconsistent():
    n = 120
    d = common.packed_distances(chain(n, 2))
    d_hat = np.maximum(d + 2.0 * np.random.default_rng(3).standard_normal(d.size), 0.0)
    result, p, _ = fit(d_hat, n)
    terms = common.decompose(d_hat, p, d)
    assert result["converged"]
    # iid noise has about (3n - 6) of n(n-1)/2 directions inside the embeddable set: share ~ 1 - 6/n.
    share = terms["inconsistent_sq"] / terms["total_sq"]
    assert 0.9 < share < 1.0
    assert abs(share - (1.0 - 6.0 / n)) < 0.03
    assert abs(terms["cross"]) < 0.02 * terms["total_sq"]


def test_perturbed_valid_structure_is_all_consistent():
    n = 80
    coordinates = chain(n, 4)
    d = common.packed_distances(coordinates)
    moved = coordinates + 1.5 * np.random.default_rng(5).standard_normal(coordinates.shape)
    d_hat = common.packed_distances(moved)
    result, p, _ = fit(d_hat, n)
    terms = common.decompose(d_hat, p, d)
    assert result["converged"] and terms["total_sq"] > 1.0
    assert terms["inconsistent_sq"] < 1e-16
    assert abs(terms["cross"]) < 1e-8
    assert abs(terms["consistent_sq"] - terms["total_sq"]) < 1e-8 * terms["total_sq"]


# ---------------------------------------------------------------- SMACOF behaviour

def noisy_case(n: int = 60, seed: int = 6, scale: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    d = common.packed_distances(chain(n, seed))
    return d, np.maximum(d + scale * np.random.default_rng(seed + 100).standard_normal(d.size), 0.0)


def test_input_is_not_rescaled():
    n = 60
    d, d_hat = noisy_case(n)
    result, p, start = fit(d_hat, n, rel_tol=1e-12)
    big = common.smacof_metric(common.packed_to_matrix(10.0 * d_hat, n), 10.0 * start, rel_tol=1e-12)
    assert np.allclose(common.packed_distances(big["coordinates"]), 10.0 * p, rtol=1e-6, atol=1e-8)
    assert np.isclose(big["raw_stress"], 100.0 * result["raw_stress"], rtol=1e-6)
    # An embeddable input in angstroms comes back at its own scale, not a normalised one.
    exact, p_exact, _ = fit(d, n)
    assert np.allclose(p_exact, d, rtol=0, atol=1e-9)
    assert np.isclose(np.sum(p_exact ** 2), np.sum(d ** 2), rtol=1e-12)


def test_agrees_with_scikit_learn_metric_smacof():
    from sklearn.manifold import smacof

    n = 40
    _, d_hat = noisy_case(n, seed=7)
    matrix = common.packed_to_matrix(d_hat, n)
    start, _ = helpers.classical_mds(matrix, n_dims=3)
    ours = common.smacof_metric(matrix, start, rel_tol=1e-13, max_iter=50_000)
    theirs, their_stress = smacof(matrix, metric=True, n_components=3, init=start.copy(), n_init=1,
                                  max_iter=50_000, eps=1e-16, normalized_stress=False)
    assert np.isclose(ours["raw_stress"], their_stress, rtol=1e-6)
    assert np.allclose(common.packed_distances(ours["coordinates"]), common.packed_distances(theirs),
                       rtol=0, atol=1e-3)


def test_identity_scale_ratio_and_monotone_stress():
    n = 60
    d, d_hat = noisy_case(n, seed=8)
    result, p, start = fit(d_hat, n)
    terms = common.decompose(d_hat, p, d)
    classical = common.decompose(d_hat, common.packed_distances(start), d)
    assert common.identity_holds(terms) and common.identity_holds(classical)
    assert result["converged"] and abs(terms["scale_ratio"] - 1.0) < 1e-6
    assert terms["raw_stress"] <= classical["raw_stress"]
    assert np.isclose(result["start_raw_stress"], classical["raw_stress"], rtol=1e-9)
    assert np.isclose(result["raw_stress"], terms["raw_stress"], rtol=1e-9)
    assert terms["raw_stress"] == common.raw_stress(d_hat, p)


def test_iteration_cap_sets_the_converged_flag():
    n = 60
    _, d_hat = noisy_case(n, seed=9)
    capped, _, _ = fit(d_hat, n, max_iter=1)
    assert capped["iterations"] == 1 and not capped["converged"]
    unmoved, p, start = fit(d_hat, n, max_iter=0)
    assert unmoved["iterations"] == 0 and not unmoved["converged"]
    assert np.allclose(p, common.packed_distances(start))


def test_snapshots_are_recorded_in_order():
    n = 60
    _, d_hat = noisy_case(n, seed=10)
    result, _, _ = fit(d_hat, n, rel_tol=1e-10)
    loose, tight = result["snapshots"][1e-6][0], result["snapshots"][1e-8][0]
    assert 0 < loose <= tight <= result["iterations"]
    looser_run, _, _ = fit(d_hat, n, rel_tol=1e-6)
    assert looser_run["iterations"] == loose and 1e-8 not in looser_run["snapshots"]


def test_rejects_invalid_matrices():
    matrix = common.packed_to_matrix(np.ones(3), 3)
    start = np.eye(3)
    bad = matrix.copy()
    bad[0, 1] = 2.0
    with pytest.raises(ValueError):
        common.smacof_metric(bad, start)
    with pytest.raises(ValueError):
        common.smacof_metric(-matrix, start)


def test_bin_sums_add_up_to_whole_protein_terms():
    n = 50
    d, d_hat = noisy_case(n, seed=11)
    _, p, _ = fit(d_hat, n)
    terms = common.decompose(d_hat, p, d)
    pair_i, pair_j = np.tril_indices(n, k=-1)
    codes = triangles.all_bin_codes(d, replay.sequence_separation_bin_codes(pair_i, pair_j))
    for system_codes, n_bins in ((codes["B10xS6"] % 6, 6), (codes["B10"], 10)):
        sums = common.bin_sums(d_hat, p, d, system_codes, n_bins)
        assert sums["n_pairs"].sum() == d.size
        assert np.isclose(sums["sum_total_sq"].sum() / d.size, terms["total_sq"])
        assert np.isclose(sums["sum_inconsistent_sq"].sum() / d.size, terms["inconsistent_sq"])
        assert np.isclose(sums["sum_consistent_sq"].sum() / d.size, terms["consistent_sq"])


def test_selected_vectors_are_observed_plus_b10_controls():
    vectors = {(probe, "observed", "none", -1): 0 for probe in triangles.PROBES}
    for probe in triangles.PROBES:
        for model, binning in triangles.null_runs(probe):
            for draw in range(triangles.N_DRAWS):
                vectors[(probe, model, binning, draw)] = 0
    keys = worker.selected_keys(vectors)
    assert len(vectors) == 37 and len(keys) == 28
    assert {k[2] for k in keys} == {"none", "B10"}
    assert {(k[1], k[3]) for k in keys if k[0] == "shared_ols_base256"} == {
        ("observed", -1), *((m, draw) for m in ("null_a", "null_b") for draw in range(3))}


# ---------------------------------------------------------------- a layer on a fake export

PROTEINS = (("aaaa_A", "cameo", 34), ("bbbb_A", "cameo", 41), ("cccc", "ecod", 37))
TM_BINARY = pytest.mark.skipif(not (CLUSTER_DIR / "TMscore").is_file() and not __import__("shutil").which("TMscore"),
                               reason="TMscore executable unavailable")


def write_pdb(path: Path, coordinates: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    lines = [f"ATOM  {k:5d}  CA  ALA A{k:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n"
             for k, (x, y, z) in enumerate(coordinates, start=1)]
    path.write_text("".join(lines) + "END\n")


def fixture_structures() -> list[np.ndarray]:
    """Chains whose occupied B10 bins each hold at least two pairs in some protein.

    The Gaussian control needs a bin SD from somewhere; a three-protein cohort with a
    single pair in its farthest bin would have none.
    """
    for seed in range(20, 200):
        structures = [chain(n, seed + number) for number, (_, _, n) in enumerate(PROTEINS)]
        counts = np.stack([np.bincount(triangles.distance_codes(common.packed_distances(c), triangles.B10_EDGES),
                                       minlength=len(triangles.B10_LABELS)) for c in structures])
        if np.all((counts.sum(axis=0) == 0) | (counts.max(axis=0) >= 2)):
            return structures
    raise AssertionError("No suitable fixture structures found")


@pytest.fixture()
def export(tmp_path, monkeypatch):
    """A three-protein packed export, its structures, the triangle run on it, and a parity table."""
    roots = {"cameo": tmp_path / "cameo_pdb", "ecod": tmp_path / "ecod_pdb"}
    monkeypatch.setattr(helpers, "CAMEO_PDB_ROOT", roots["cameo"])
    monkeypatch.setattr(helpers, "ECOD_PDB_ROOT", roots["ecod"])
    monkeypatch.setattr(helpers, "CAMEO_PAIR_BLOCKS_ROOT", tmp_path / "none_cameo")
    monkeypatch.setattr(helpers, "ECOD_PAIR_BLOCKS_ROOT", tmp_path / "none_ecod")
    monkeypatch.setattr(triangles, "EXPECTED_COHORT", {"cameo": 2, "ecod": 1})
    export_root, layer_dir = tmp_path / "export", tmp_path / "export/layers/layer_00"
    cohort_csv = tmp_path / "cohort.csv"
    cohort_rows, index_rows, parity_rows = [], [], []
    structures = fixture_structures()
    for number, (pid, source, n) in enumerate(PROTEINS):
        coordinates = structures[number]
        write_pdb(roots[source] / pid / "predicted_pdb" / f"{pid}_model_1_ptm_unrelaxed.pdb", coordinates)
        d = common.packed_distances(coordinates)
        cohort_rows.append({"protein_id": pid, "source": source, "n_residues": n, "n_pairs": d.size})
        for code, probe in enumerate(triangles.PROBES):
            rng = np.random.default_rng([number, code])
            values = (d + rng.normal(-0.3, 1.0 + code, d.size)).astype(np.float32)
            path = layer_dir / pid / f"{probe}.npy"
            path.parent.mkdir(parents=True, exist_ok=True)
            np.save(path, values)
            index_rows.append({"protein_id": pid, "analysis_name": probe, "status": "complete",
                               "n_values": d.size, "array_sha256": hashlib.sha256(path.read_bytes()).hexdigest()})
            pair_i, pair_j = np.tril_indices(n, k=-1)
            canonical = helpers.analyze_protein_geometry(n, coordinates, pair_i, pair_j, values.astype(np.float64))
            parity_rows.append({"analysis_name": probe, "layer": 0, "protein_id": pid, "stress": canonical["stress"],
                                "rmsd": canonical["rmsd"],
                                "n_negative_predictions": canonical["n_negative_predictions"]})
    for path, rows in ((cohort_csv, cohort_rows), (layer_dir / "prediction_index.csv", index_rows)):
        with open(path, "w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
    parity_csv = tmp_path / "parity.csv.gz"
    with gzip.open(parity_csv, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(parity_rows[0]))
        writer.writeheader()
        writer.writerows(parity_rows)

    monkeypatch.setattr(replay, "EXPORT", export_root)
    monkeypatch.setattr(replay, "COHORT_CSV", cohort_csv)
    monkeypatch.setattr(replay, "OUTPUT_ROOT", tmp_path / "triangles")
    monkeypatch.setenv("SLURM_CPUS_PER_TASK", "2")
    monkeypatch.setenv("NULL_TRIANGLES_LAYER", "0")
    monkeypatch.setenv("NULL_TRIANGLES_SKIP_PAPER_PARITY", "1")
    replay.main()                                   # the real triangle run, on the fake export

    output = tmp_path / "smacof"
    monkeypatch.setattr(worker, "OUTPUT_ROOT", output)
    monkeypatch.setattr(worker, "TRIANGLE_ROOT", tmp_path / "triangles")
    monkeypatch.setattr(worker, "PARITY_CSV", parity_csv)
    monkeypatch.setattr(aggregate, "OUTPUT_ROOT", output)
    monkeypatch.setattr(aggregate, "LAYERS", (0,))
    monkeypatch.setenv("SMACOF_DECOMPOSITION_LAYER", "0")
    return {"output": output, "triangles": tmp_path / "triangles"}


def read_gz(path: Path) -> list[dict]:
    with gzip.open(path, "rt", newline="") as handle:
        return list(csv.DictReader(handle))


@TM_BINARY
def test_layer_run_and_aggregation(export):
    worker.main()
    output = export["output"]
    report = json.loads((output / "validation/layer_00.json").read_text())
    assert report["status"] == "complete"
    assert report["gates"]["triangle_pairing"]["vectors_compared"] == 3 * 28
    assert report["gates"]["triangle_pairing"]["proteins_with_violation_count_check"] == 3
    assert not any(report["gates"][gate]["failures"] for gate in report["gates"])
    assert report["gates"]["classical_parity"]["worst"]["parity_stress"] < 1e-9

    rows = read_gz(output / "layer_shards/decomposition_layer_00.csv.gz")
    assert len(rows) == 3 * 4 * 7
    assert {r["matrix_type"] for r in rows} == {"observed", "control_gaussian", "control_shuffled"}
    for r in rows:
        total = float(r["total_sq"])
        parts = float(r["smacof_inconsistent_sq"]) + float(r["smacof_consistent_sq"]) + float(r["smacof_cross"])
        assert abs(total - parts) <= 1e-9 * max(total, 1.0)
        assert float(r["smacof_raw_stress"]) <= float(r["classical_raw_stress"]) * (1 + 1e-9)
        assert r["smacof_converged"] == "1" and abs(float(r["smacof_scale_ratio"]) - 1.0) < 1e-5
        assert (r["smacof_tm_score"] != "") == (r["matrix_type"] == "observed")
        if r["matrix_type"] == "observed":
            assert 0.0 < float(r["smacof_tm_score"]) <= 1.0
    coordinates = np.load(output / "layer_shards/smacof_coordinates_layer_00.npz")
    assert len(coordinates.files) == 3 * 4
    assert coordinates["aaaa_A/shared_ols_base256"].shape == (34, 3)
    assert coordinates["aaaa_A/shared_ols_base256"].dtype == np.float32

    bins = read_gz(output / "layer_shards/bin_sums_layer_00.csv.gz")
    assert {r["bin_system"] for r in bins} == set(common.BIN_SYSTEMS)
    assert {r["n_draws"] for r in bins if r["matrix_type"] != "observed"} == {"3"}

    worker.main()                                   # a complete layer is skipped, not redone
    aggregate.main()
    tables = output / "tables"
    per_protein = read_gz(tables / "decomposition_per_protein.csv.gz")
    assert len(per_protein) == 3 * 4 * 3
    assert len(read_gz(tables / "decomposition_per_draw.csv.gz")) == len(rows)
    manifest = json.loads((tables / "run_manifest.json").read_text())
    assert manifest["rows"]["decomposition_per_protein"] == 36 and manifest["checks"]["failures"] == 0
    assert manifest["smacof"]["input_rescaling"] == "none"
    n_bin_rows = sum(len(read_gz(tables / name)) for name in aggregate.BIN_FILES.values())
    assert n_bin_rows == len(bins)
    with open(tables / "bin_terms_cohort.csv", newline="") as handle:
        cohort_bins = list(csv.DictReader(handle))
    assert {r["bin_system"] for r in cohort_bins} == set(common.BIN_SYSTEMS)
    with open(tables / "convergence_summary.csv", newline="") as handle:
        assert all(r["n_runs"] == r["n_converged"] for r in csv.DictReader(handle))


def test_failed_gate_writes_nothing(export, monkeypatch):
    monkeypatch.setenv("SMACOF_DECOMPOSITION_SKIP_TM_SCORE", "1")
    shard = export["triangles"] / "layer_shards/null_metrics_layer_00.csv.gz"
    rows = read_gz(shard)
    target = next(r for r in rows if r["model"] == "null_b" and r["binning"] == "B10")
    target["n_negative_values"] = str(int(target["n_negative_values"]) + 1)
    with gzip.open(shard, "wt", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    with pytest.raises(SystemExit):
        worker.main()
    output = export["output"]
    failed = json.loads((output / "validation/layer_00_FAILED.json").read_text())
    assert len(failed["gates"]["triangle_pairing"]["failures"]) == 1
    assert not (output / "validation/layer_00.json").exists()
    assert not (output / "layer_shards").exists()
    with pytest.raises(SystemExit):
        aggregate.main()


def test_launchers_follow_cluster_rules():
    for name in ("smacof_error_decomposition.sbatch", "aggregate_smacof_error_decomposition.sbatch"):
        text = (SCRIPT_DIR / name).read_text()
        assert "#SBATCH --mail-type=BEGIN,END,FAIL" in text
        assert "--mail-user" not in text
        assert "SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}" in text
        assert "diagnostic_experiments/smacof_error_decomposition_5fold_v1" in text
        assert "canonical_storage.py\" --check-write" in text


def test_parity_table_covers_cohort():
    with gzip.open(SCRIPT_DIR / "classical_geometry_parity_5fold_v1.csv.gz", "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 552 * 4 * 48
    assert {r["analysis_name"] for r in rows} == set(triangles.PROBES)
    assert len({(r["analysis_name"], r["layer"], r["protein_id"]) for r in rows}) == len(rows)

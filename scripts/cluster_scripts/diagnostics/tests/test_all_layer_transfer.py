from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "cross_layer_transfer_all_layers"
CLUSTER_DIR = SCRIPT_DIR.parents[1]
for path in (SCRIPT_DIR, CLUSTER_DIR, CLUSTER_DIR / "cross_layer_transfer",
             CLUSTER_DIR / "cross_validation", CLUSTER_DIR / "triangle_violation_subgroups"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import aggregate_all_layer_transfer as agg  # noqa: E402
import all_layer_transfer_common as common  # noqa: E402
import all_layer_transfer_replay as worker  # noqa: E402
import make_all_layer_transfer_figures as figures  # noqa: E402
from all_layer_transfer_common import (  # noqa: E402
    CANONICAL_DISTANCE_GROUPS, CANONICAL_SEQUENCE_GROUPS, CATEGORY_CELLS, DISTANCE_EDGES, ERROR_EDGES,
    N_ERROR_BINS, N_OUTER_FOLDS, N_POSITION_BINS, OUTLIER_PROTEIN, STAT_FIELDS, ProbeBank, TargetAccumulator,
    analysis_label, analysis_name, cohort_name, default_output_root, distance_fine_bins, histogram_quantiles,
    position_bins, predict_all_sources, select_cohort, sequence_fine_bins, strata_support_status,
    summarize_protein, target_layer_for_task, transfer_identity, validate_final_shard,
)
from cross_layer_transfer_common import predict_transfer, primitive_rows  # noqa: E402
from cross_validation_common import OOFComparisonAccumulator  # noqa: E402
from regression_protocol import (  # noqa: E402
    SEQUENCE_BIN_LABELS, SPATIAL_DISTANCE_BIN_LABELS, sequence_separation_bin_codes,
    spatial_distance_bin_codes, strict_lower_triangle_indices,
)

LAYERS = (22, 47)
NATIVE_CHECK = worker.verify_native_cell


def random_bank(rng, n_layers: int) -> ProbeBank:
    shape = (n_layers, N_OUTER_FOLDS)
    return ProbeBank(
        intercept=rng.normal(25, 3, size=shape), coefficient=rng.normal(0, .2, size=(*shape, 256)),
        feature_mean=rng.normal(0, 1, size=(*shape, 256)), feature_scale=rng.uniform(.5, 2, size=(*shape, 256)),
        global_baseline=rng.normal(22, 1, size=N_OUTER_FOLDS),
    )


def synthetic_protein(rng, n_residues: int):
    pair_i, pair_j = strict_lower_triangle_indices(n_residues)
    coordinates = np.cumsum(rng.normal(scale=2.2, size=(n_residues, 3)), axis=0)
    y = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
    return coordinates, pair_i, pair_j, y


def test_edges_and_task_mapping() -> None:
    assert N_ERROR_BINS == 150
    finite = ERROR_EDGES[1:-1]
    np.testing.assert_array_equal(finite, -finite[::-1])
    for angstroms in (1, 2, 5, 10):
        assert angstroms in finite and -angstroms in finite
    assert [target_layer_for_task(i) for i in (0, 47)] == [0, 47]
    with pytest.raises(ValueError):
        target_layer_for_task(48)
    assert transfer_identity(3, 47) == {
        "source_layer": 3, "target_layer": 47, "transfer_mode": "target_restandardized",
        "layer_offset": 44, "is_same_layer": False,
    }
    assert len(CATEGORY_CELLS) == 12


def test_fine_bins_nest_in_canonical_bins() -> None:
    separations = np.arange(1, 1200)
    canonical = sequence_separation_bin_codes(separations, np.zeros_like(separations))
    fine = sequence_fine_bins(separations, np.zeros_like(separations))
    for code, label in enumerate(SEQUENCE_BIN_LABELS):
        assert set(fine[canonical == code]) == set(CANONICAL_SEQUENCE_GROUPS[label])
    distances = np.unique(np.r_[DISTANCE_EDGES[:-1], DISTANCE_EDGES[1:-1] - 1e-9, np.linspace(0, 300, 6001)])
    canonical = spatial_distance_bin_codes(distances)
    fine = distance_fine_bins(distances)
    for code, label in enumerate(SPATIAL_DISTANCE_BIN_LABELS):
        assert set(fine[canonical == code]) == set(CANONICAL_DISTANCE_GROUPS[label])


def test_batched_weights_match_v1_target_restandardized() -> None:
    rng = np.random.default_rng(0)
    bank = random_bank(rng, 4)
    x = rng.normal(size=(37, 256))
    fold, target = 2, 3
    weights, constants = bank.restandardized_weights(target, fold)
    batched = predict_all_sources(x, weights, constants)

    def fit(layer):
        return {"intercept_standardized": bank.intercept[layer, fold],
                "coefficient_standardized": bank.coefficient[layer, fold],
                "feature_mean": bank.feature_mean[layer, fold], "feature_scale": bank.feature_scale[layer, fold]}

    for source in range(4):
        expected = predict_transfer(x, fit(source), fit(target), "target_restandardized")
        np.testing.assert_allclose(batched[:, source], expected, rtol=1e-11)


def test_nonfinite_predictions_are_rejected() -> None:
    with pytest.raises(ValueError, match="nonfinite"):
        predict_all_sources(np.full((2, 256), np.nan), np.ones((1, 256)), np.zeros(1))


def test_summary_matches_brute_force() -> None:
    rng = np.random.default_rng(1)
    n_residues, n_sources = 60, 3
    _, pair_i, pair_j, y = synthetic_protein(rng, n_residues)
    predictions = y[:, None] + rng.normal(scale=(2.0, 15.0, 90.0), size=(y.size, n_sources))
    predictions[0, 2] = y[0] + 900.0        # lands in the open upper tail
    predictions[1, 2] = y[1] - 900.0        # lands in the open lower tail
    summary = summarize_protein(y=y, predictions=predictions, pair_i=pair_i, pair_j=pair_j,
                                n_residues=n_residues, baseline_value=22.5)
    sequence = sequence_separation_bin_codes(pair_i, pair_j)
    spatial = spatial_distance_bin_codes(y)
    sequence_fine, distance_fine = sequence_fine_bins(pair_i, pair_j), distance_fine_bins(y)
    positions = np.stack([position_bins(pair_i, n_residues), position_bins(pair_j, n_residues)], axis=1)
    for source in range(n_sources):
        rows = primitive_rows(identity={}, y_true=y, predictions=predictions[:, source],
                              sequence_codes=sequence, spatial_codes=spatial)
        assert [(r["category_scope"], r["category_label"]) for r in rows] == list(CATEGORY_CELLS)
        for c, row in enumerate(rows):
            np.testing.assert_allclose(summary["probe"][source, c], [float(row[f]) for f in STAT_FIELDS],
                                       rtol=1e-12, atol=1e-9)
        error = predictions[:, source] - y
        bins = np.array([int(np.flatnonzero((ERROR_EDGES[:-1] <= e) & (e < ERROR_EDGES[1:]))[0]) for e in error])
        np.testing.assert_array_equal(summary["hist_protein"][source], np.bincount(bins, minlength=N_ERROR_BINS))
        joint = np.zeros(summary["hist_joint"].shape[1:], dtype=np.int64)
        stats = np.zeros(summary["strata_stats"].shape[1:])
        position = np.zeros((N_POSITION_BINS, 3))
        for k, value in enumerate(error):
            joint[bins[k], sequence_fine[k], distance_fine[k]] += 1
            stats[sequence_fine[k], distance_fine[k]] += (value, abs(value), value * value)
            for b in positions[k]:
                position[b] += (value, abs(value), value * value)
        np.testing.assert_array_equal(summary["hist_joint"][source], joint)
        np.testing.assert_allclose(summary["strata_stats"][source], stats, rtol=1e-12, atol=1e-9)
        np.testing.assert_allclose(summary["position_stats"][source], position, rtol=1e-12, atol=1e-9)
        assert summary["error_min"][source] == error.min() and summary["error_max"][source] == error.max()
    assert summary["hist_protein"][2, 0] >= 1 and summary["hist_protein"][2, -1] >= 1
    np.testing.assert_array_equal(summary["position_n"], np.bincount(positions.ravel(), minlength=N_POSITION_BINS))
    baseline = 22.5 - y
    np.testing.assert_allclose(summary["baseline"], [
        y.size, y.sum(), (y * y).sum(), 22.5 * y.size, baseline.sum(), np.abs(baseline).sum(), (baseline * baseline).sum(),
    ], rtol=1e-12)


def test_chain_guard_rejects_interchain_pairs() -> None:
    pair_i, pair_j = strict_lower_triangle_indices(12)
    worker.require_single_chain_bins(pair_i, pair_j, ["A"] * 12)
    with pytest.raises(ValueError, match="Interchain"):
        worker.require_single_chain_bins(pair_i, pair_j, ["A"] * 6 + ["B"] * 6)


def test_support_status_and_quantiles() -> None:
    status = strata_support_status(np.array([0, 50, 50, 500, 500]), np.array([0, 2, 9, 2, 9]))
    assert list(status) == ["empty", "low_pair_and_protein_support", "low_pair_support",
                            "low_protein_support", "sufficient"]
    values = np.random.default_rng(2).normal(scale=4, size=200_000)
    counts = np.bincount(np.searchsorted(ERROR_EDGES, values, side="right") - 1, minlength=N_ERROR_BINS)
    quantiles = histogram_quantiles(counts, (.05, .5, .95), values.min(), values.max())
    np.testing.assert_allclose(quantiles, np.quantile(values, (.05, .5, .95)), atol=.1)
    assert np.isnan(histogram_quantiles(np.zeros(N_ERROR_BINS), (.5,), -1, 1)).all()


def cohort(n_proteins: int = 14, n_ecod: int = 3):
    """A small cohort spanning all folds, with the known outlier and some ECOD proteins."""
    rng = np.random.default_rng(3)
    proteins = [OUTLIER_PROTEIN if index == 3 else f"p{index:02d}" for index in range(n_proteins)]
    entries = {}
    for index, protein in enumerate(proteins):
        n_residues = int(rng.integers(12, 34))
        entries[protein] = SimpleNamespace(
            protein_id=protein, source="ecod" if index >= n_proteins - n_ecod else "cameo",
            outer_fold=index % N_OUTER_FOLDS, n_residues=n_residues, n_pairs=n_residues * (n_residues - 1) // 2)
    return SimpleNamespace(
        eligible_protein_ids=tuple(proteins), outer_by_protein=entries, fold_digests={"cohort_digest": "x"},
        protein_ids_for_outer_fold=lambda fold: tuple(p for p in proteins if entries[p].outer_fold == fold),
    )


def fake_loader(seed: int = 4):
    """A deterministic stand-in for reading one protein at one layer."""
    def load(outer, layer):
        if outer.source != "cameo":
            raise AssertionError(f"Partial cohort must not read {outer.protein_id}")
        rng = np.random.default_rng([seed, sum(map(ord, outer.protein_id))])
        coordinates, pair_i, pair_j, _ = synthetic_protein(rng, outer.n_residues)
        layer_rng = np.random.default_rng([seed, sum(map(ord, outer.protein_id)), layer])
        return coordinates, tuple("A" for _ in range(outer.n_residues)), pair_i, pair_j, layer_rng.normal(size=(pair_i.size, 256))
    return load


@pytest.fixture
def environment(monkeypatch, tmp_path):
    bundle = cohort()
    bank = random_bank(np.random.default_rng(5), len(LAYERS))
    bank = ProbeBank(bank.intercept, bank.coefficient, bank.feature_mean, bank.feature_scale,
                     np.broadcast_to(bank.global_baseline, (N_OUTER_FOLDS,)).copy())
    output = tmp_path / "out"
    monkeypatch.setenv("ALL_LAYER_TRANSFER_OUTPUT_ROOT", str(output))
    monkeypatch.setenv("CSV_ROOT", str(tmp_path / "csv"))
    monkeypatch.delenv("CROSS_LAYER_ECOD_EXPERIMENTS_ROOT", raising=False)
    monkeypatch.setenv("ALL_LAYER_CHECKPOINT_PROTEINS", "4")
    monkeypatch.setenv("ALL_LAYER_COHORT", "cameo")
    monkeypatch.setitem(common.COHORT_SIZES, "cameo", 11)
    ids = tuple(p for p in bundle.eligible_protein_ids if bundle.outer_by_protein[p].source == "cameo")
    status = [{"source_layer": layer, "outer_fold": fold, "fit_status": "fitted", "support_flags": "",
               "condition_number": 1.0, "artifact_path": "x", "artifact_sha256": "x", "scaler_digest": "x",
               "global_baseline_value": 1.0} for layer in LAYERS for fold in range(N_OUTER_FOLDS)]
    for module in (worker, agg):
        monkeypatch.setattr(module, "LAYERS", LAYERS)
        monkeypatch.setattr(module, "load_cv_folds", lambda **_: bundle)
        monkeypatch.setattr(module, "load_probe_bank", lambda *_: (bank, {"artifacts": []}, status))
    monkeypatch.setattr(worker, "load_protein", fake_loader())
    monkeypatch.setattr(worker, "verify_native_cell", lambda *_: None)
    return SimpleNamespace(output=output, bundle=bundle, bank=bank, ids=ids, monkeypatch=monkeypatch)


def run_workers(environment) -> None:
    for task in range(len(LAYERS)):
        environment.monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task))
        worker.main()


def load_tables(environment):
    return agg.load_primitives(environment.output, environment.bundle, LAYERS, "cameo")


def test_worker_resume_matches_uninterrupted_run(environment, tmp_path) -> None:
    run_workers(environment)
    uninterrupted = {}
    with np.load(environment.output / "layer_shards/target_layer_47.npz") as shard:
        uninterrupted = {key: shard[key] for key in shard.files}
    (environment.output / "layer_shards/target_layer_47.npz").unlink()

    loader = fake_loader()
    target = environment.ids[6]

    def failing(outer, layer):
        if outer.protein_id == target:
            raise OSError("transient read failure")
        return loader(outer, layer)

    environment.monkeypatch.setenv("SLURM_ARRAY_TASK_ID", "1")
    environment.monkeypatch.setattr(worker, "load_protein", failing)
    with pytest.raises(RuntimeError, match="incomplete"):
        worker.main()
    checkpoint = environment.output / "layer_shards/.target_layer_47.checkpoint.npz"
    assert checkpoint.is_file() and not (environment.output / "layer_shards/target_layer_47.npz").exists()
    with np.load(checkpoint) as partial:
        assert partial["completed"].sum() == len(environment.ids) - 1

    environment.monkeypatch.setattr(worker, "load_protein", loader)
    worker.main()
    assert not checkpoint.exists()
    with np.load(environment.output / "layer_shards/target_layer_47.npz") as shard:
        for key, value in uninterrupted.items():
            np.testing.assert_array_equal(shard[key], value, err_msg=key)
    # A finished task exits without touching its shard.
    stamp = (environment.output / "layer_shards/target_layer_47.npz").stat().st_mtime_ns
    worker.main()
    assert (environment.output / "layer_shards/target_layer_47.npz").stat().st_mtime_ns == stamp


def test_final_shard_validation_rejects_damage(environment) -> None:
    run_workers(environment)
    path = environment.output / "layer_shards/target_layer_22.npz"
    ids = environment.ids
    accumulator = TargetAccumulator.from_file(path, 22, ids, len(LAYERS))
    validate_final_shard(accumulator)
    with pytest.raises(ValueError, match="written by revision"):
        TargetAccumulator.from_file(path, 22, ids, len(LAYERS), "another-revision")
    with pytest.raises(ValueError, match="wrong schema or target"):
        TargetAccumulator.from_file(path, 47, ids, len(LAYERS))
    with pytest.raises(ValueError, match="protein order"):
        TargetAccumulator.from_file(path, 22, ids[::-1], len(LAYERS))
    with pytest.raises(ValueError, match="added twice"):
        accumulator.add(0, 0, {})
    accumulator.arrays["hist_joint"][0, 0, 5, 3, 3] += 1
    with pytest.raises(ValueError, match="histogram mass"):
        validate_final_shard(accumulator)
    damaged = TargetAccumulator.from_file(path, 22, ids, len(LAYERS))
    damaged.arrays["completed"][2] = False
    with pytest.raises(ValueError, match="missing proteins"):
        validate_final_shard(damaged)


def test_consistency_checks_pass_then_catch_tampering(environment) -> None:
    run_workers(environment)
    primitives, _, consistency, _ = load_tables(environment)
    assert consistency and all(row["status"] == "pass" for row in consistency)
    assert {row["check"] for row in consistency} >= {
        "joint_histogram_matches_protein_histograms", "strata_error_sums_match_categories",
        "position_error_sums_are_twice_totals", "histogram_bounds_contain_exact_sums",
        "outlier_arrays_match_protein", "shards_share_this_code_revision",
    }
    path = environment.output / "layer_shards/target_layer_22.npz"
    ids = environment.ids

    def status_after(change) -> dict[str, str]:
        accumulator = TargetAccumulator.from_file(path, 22, ids, len(LAYERS))
        change(accumulator.arrays)
        return {row["check"]: row["status"] for row in agg.shard_consistency_checks(22, accumulator, primitives)}

    def move_count(a):
        a["hist_joint"][0, 0, 70, 3, 3] -= 1
        a["hist_joint"][0, 0, 90, 3, 3] += 1

    def bump_strata(a):
        a["strata_stats"][0, 0, 3, 3, 2] += 50.0

    def bump_position(a):
        a["position_stats"][0, 0, 1, 1] += 50.0

    def mirror_histograms(a):
        a["hist_joint"][:] = a["hist_joint"][:, :, ::-1]
        a["hist_protein"][:] = a["hist_protein"][:, :, ::-1]

    mirrored = status_after(mirror_histograms)
    assert mirrored["histogram_bounds_contain_exact_sums"] == "fail"
    assert mirrored["joint_histogram_matches_protein_histograms"] == "pass"
    assert status_after(move_count)["joint_histogram_matches_protein_histograms"] == "fail"
    assert status_after(bump_strata)["strata_error_sums_match_categories"] == "fail"
    assert status_after(bump_position)["position_error_sums_are_twice_totals"] == "fail"


def direct_cell(environment, source: int, target: int):
    """Recompute one cell directly from the synthetic inputs."""
    loader, bank = fake_loader(), environment.bank
    target_position, source_position = LAYERS.index(target), LAYERS.index(source)
    y_all, prediction_all = [], []
    direct = OOFComparisonAccumulator()
    for protein in environment.ids:
        outer = environment.bundle.outer_by_protein[protein]
        coordinates, _, pair_i, pair_j, features = loader(outer, target)
        y = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
        weights, constants = bank.restandardized_weights(target_position, outer.outer_fold)
        prediction = predict_all_sources(features, weights, constants)[:, source_position]
        direct.update_protein(protein, outer.outer_fold, y, prediction, float(bank.global_baseline[outer.outer_fold]))
        y_all.append(y), prediction_all.append(prediction)
    return direct, np.concatenate(y_all), np.concatenate(prediction_all)


def test_tables_equal_direct_recomputation(environment) -> None:
    run_workers(environment)
    primitives, runtime, _, histograms = load_tables(environment)
    n_cells = len(LAYERS) ** 2
    n_proteins = len(primitives.protein_ids)
    tables = agg.build_metric_tables(primitives, environment.bundle)
    retained: list[dict[str, object]] = []
    per_protein = list(agg.per_protein_rows(primitives, retained))
    assert len(per_protein) == n_cells * n_proteins and len(retained) == len(per_protein)  # 22 and 47 are v1 layers
    assert len(tables["pooled"]) == n_cells and len(tables["fold"]) == n_cells * 5
    assert len(tables["sequence_separation"]) == n_cells * 6 and len(tables["spatial_distance"]) == n_cells * 5

    source, target = 47, 22
    direct, y, prediction = direct_cell(environment, source, target)
    pooled = next(r for r in tables["pooled"] if (r["source_layer"], r["target_layer"]) == (source, target))
    expected = direct.model.metrics()
    for key in ("r2", "mae", "rmse", "bias_pred_minus_true"):
        assert math.isclose(pooled[f"pooled_{key}"], expected[key], rel_tol=1e-10, abs_tol=1e-9)
    assert math.isclose(pooled["global_baseline_pooled_mae"], direct.global_baseline.metrics()["mae"], rel_tol=1e-10)
    assert pooled["layer"] == target and pooled["transfer_mode"] == "target_restandardized"
    assert pooled["n_proteins"] == len(environment.ids) and pooled["analysis_name"].endswith("_cameo")

    matrix = agg.transfer_matrix_rows(tables["pooled"], primitives)
    assert len(matrix) == n_cells
    native = next(r for r in matrix if r["source_layer"] == r["target_layer"] == 47)
    assert native["delta_pooled_vs_native_target_r2"] == 0.0
    row = next(r for r in matrix if (r["source_layer"], r["target_layer"]) == (source, target))
    keep = []
    for protein in environment.ids:
        keep.extend([protein != OUTLIER_PROTEIN] * environment.bundle.outer_by_protein[protein].n_pairs)
    keep = np.asarray(keep)
    error = prediction[keep] - y[keep]
    assert math.isclose(row[f"pooled_mae_excl_{OUTLIER_PROTEIN}"], float(np.abs(error).mean()), rel_tol=1e-10)
    assert math.isclose(row[f"pooled_rmse_excl_{OUTLIER_PROTEIN}"], float(np.sqrt((error * error).mean())), rel_tol=1e-10)
    expected_r2 = 1 - (error * error).sum() / ((y[keep] - y[keep].mean()) ** 2).sum()
    assert math.isclose(row[f"pooled_r2_excl_{OUTLIER_PROTEIN}"], float(expected_r2), rel_tol=1e-10)

    anomalies = agg.protein_anomaly_rows(primitives)
    assert len(anomalies) == n_cells * 5
    first = [r for r in anomalies if (r["source_layer"], r["target_layer"]) == (source, target)]
    assert [r["rank"] for r in first] == [1, 2, 3, 4, 5]
    assert first[0]["protein_id"] == row["top_protein_id"]
    assert first[0]["squared_error_share"] == pytest.approx(row["top_protein_squared_error_share"])
    assert first[0]["squared_error_share"] >= first[1]["squared_error_share"]

    quantiles = list(agg.error_quantile_rows(environment.output, primitives))
    assert len(quantiles) == n_cells * 14
    whole = next(r for r in quantiles if (r["source_layer"], r["target_layer"], r["category_scope"], r["weighting"])
                 == (source, target, "all_pairs", "pair"))
    assert whole["n_pairs"] == y.size
    assert whole["signed_error_q50"] == pytest.approx(np.median(prediction - y), abs=1.0)
    assert whole["share_within_5A"] == pytest.approx(np.mean(np.abs(prediction - y) < 5), abs=1e-12)
    equal = next(r for r in quantiles if (r["source_layer"], r["target_layer"], r["weighting"])
                 == (source, target, "protein_equal"))
    assert equal["n_pairs"] == y.size

    strata = list(agg.error_strata_rows(environment.output, primitives))
    cell = [r for r in strata if (r["source_layer"], r["target_layer"]) == (source, target)]
    assert sum(r["n_pairs"] for r in cell) == y.size
    assert sum(r["share_of_squared_error"] for r in cell) == pytest.approx(1.0)
    assert sum(r["sum_abs_error"] for r in cell) == pytest.approx(float(np.abs(prediction - y).sum()))
    assert sum(r[f"n_pairs_excl_{OUTLIER_PROTEIN}"] for r in cell) == int(keep.sum())
    assert sum(r[f"sum_squared_error_excl_{OUTLIER_PROTEIN}"] for r in cell) == pytest.approx(float((error * error).sum()))

    position = list(agg.error_position_rows(environment.output, primitives))
    assert len(position) == n_cells * N_POSITION_BINS
    cell = [r for r in position if (r["source_layer"], r["target_layer"]) == (source, target)]
    assert sum(r["n_residue_pairs"] for r in cell) == 2 * y.size
    assert sum(r["mae"] * r["n_residue_pairs"] for r in cell) == pytest.approx(2 * float(np.abs(prediction - y).sum()))
    assert histograms["pair"][LAYERS.index(target), LAYERS.index(source)].sum() == y.size


def write_csv(path: Path, rows) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(rows[0])
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        writer.writerows(rows)


def v1_shard_rows(primitives, target: int) -> list[dict[str, object]]:
    """A ten-layer shard, with the raw rows and ECOD proteins the checks must ignore."""
    t = primitives.layers.index(target)
    rows = [{"target_layer": target, "source_layer": 22, "transfer_mode": "raw", "predictor": "probe",
             "protein_id": primitives.protein_ids[0], "category_scope": "all_pairs", "category_label": "all",
             **{field: 1.0 for field in STAT_FIELDS}},
            {"target_layer": target, "source_layer": 22, "transfer_mode": "target_restandardized", "predictor": "probe",
             "protein_id": "ecod_only", "category_scope": "all_pairs", "category_label": "all",
             **{field: 1.0 for field in STAT_FIELDS}}]
    for p, protein in enumerate(primitives.protein_ids):
        for s, source in enumerate(primitives.layers):
            for c, (scope, label) in enumerate(CATEGORY_CELLS):
                rows.append({"target_layer": target, "source_layer": source, "transfer_mode": "target_restandardized",
                             "predictor": "probe", "protein_id": protein, "category_scope": scope, "category_label": label,
                             **dict(zip(STAT_FIELDS, map(repr, map(float, primitives.probe[t, s, p, c]))))})
        rows.append({"target_layer": target, "source_layer": "", "transfer_mode": "", "predictor": "global_baseline",
                     "protein_id": protein, "category_scope": "all_pairs", "category_label": "all",
                     **dict(zip(STAT_FIELDS, map(repr, map(float, primitives.baseline[t, p]))))})
    return rows


def test_parity_gates_pass_and_flag_drift(environment, tmp_path) -> None:
    run_workers(environment)
    primitives, *_ = load_tables(environment)
    tables = agg.build_metric_tables(primitives, environment.bundle)
    retained: list[dict[str, object]] = []
    per_protein = list(agg.per_protein_rows(primitives, retained))

    producer = tmp_path / "producer"
    def native(rows):
        return [r for r in rows if r["source_layer"] == r["target_layer"]]
    write_csv(producer / "per_protein.csv", native(per_protein))
    write_csv(producer / "pooled_metrics.csv", native(tables["pooled"]))
    write_csv(producer / "fold_metrics.csv", native(tables["fold"]))
    write_csv(producer / "subgroup_metrics/by_sequence_separation.csv", native(tables["sequence_separation"]))
    write_csv(producer / "subgroup_metrics/by_spatial_distance.csv", native(tables["spatial_distance"]))
    def producer_checks(full):
        return agg.producer_parity_checks(producer, retained, tables, LAYERS, environment.ids, full)

    assert all(c["status"] == "pass" for c in producer_checks(True))
    partial = producer_checks(False)
    assert partial and {c["check"] for c in partial} == {"native_per_protein_vs_producer"}
    assert all(c["status"] == "pass" for c in partial)

    drifted = native(per_protein)
    drifted[0] = {**drifted[0], "mae": float(drifted[0]["mae"]) + 1e-5}
    write_csv(producer / "per_protein.csv", drifted)
    failing = [c for c in producer_checks(True) if c["status"] != "pass"]
    assert [(c["check"], c["metric"]) for c in failing] == [("native_per_protein_vs_producer", "mae")]
    assert [c["metric"] for c in producer_checks(False) if c["status"] != "pass"] == ["mae"]

    v1 = tmp_path / "v1"
    write_csv(v1 / "per_protein.csv", per_protein)
    write_csv(v1 / "pooled_metrics.csv", tables["pooled"])
    write_csv(v1 / "fold_metrics.csv", tables["fold"])
    write_csv(v1 / "subgroup_metrics/by_sequence_separation.csv", tables["sequence_separation"])
    write_csv(v1 / "subgroup_metrics/by_spatial_distance.csv", tables["spatial_distance"])
    for target in LAYERS:
        write_csv(v1 / f"layer_shards/cross_layer_primitives_target_layer_{target:02d}.csv",
                  v1_shard_rows(primitives, target))
    checks = agg.v1_overlap_checks(v1, retained, tables, primitives, True)
    assert checks and all(c["status"] == "pass" for c in checks)
    partial = agg.v1_overlap_checks(v1, retained, tables, primitives, False)
    assert {c["check"] for c in partial} == {"v1_overlap_per_protein", "v1_overlap_primitives_probe",
                                             "v1_overlap_primitives_baseline"}
    assert all(c["status"] == "pass" for c in partial)

    write_csv(v1 / "pooled_metrics.csv", tables["pooled"][1:])
    failing = [c for c in agg.v1_overlap_checks(v1, retained, tables, primitives, True) if c["status"] != "pass"]
    assert failing and {c["check"] for c in failing} == {"v1_overlap_pooled"}

    rows = v1_shard_rows(primitives, 47)
    drifted = next(r for r in rows if r["predictor"] == "probe" and r["category_scope"] == "spatial_distance")
    drifted["sum_abs_error"] = float(drifted["sum_abs_error"]) * 1.001
    write_csv(v1 / "layer_shards/cross_layer_primitives_target_layer_47.csv", rows)
    failing = [c for c in agg.v1_primitive_checks(v1, primitives) if c["status"] != "pass"]
    assert [(c["check"], c["metric"]) for c in failing] == [("v1_overlap_primitives_probe", "sum_abs_error")]

    write_csv(v1 / "layer_shards/cross_layer_primitives_target_layer_47.csv", [r for r in rows if r is not drifted])
    failing = [c for c in agg.v1_primitive_checks(v1, primitives) if c["status"] != "pass"]
    assert failing and all("expected" in c["detail"] for c in failing)


def test_aggregator_and_figures_end_to_end(environment) -> None:
    run_workers(environment)
    environment.monkeypatch.setattr(agg, "producer_parity_checks", lambda *_: [])
    environment.monkeypatch.setattr(agg, "v1_overlap_checks", lambda *_: [])
    agg.main()
    out = environment.output
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["analysis_status"] == "complete" and manifest["transfer_mode"] == "target_restandardized"
    assert manifest["cohort"] == "cameo" and manifest["n_proteins"] == len(environment.ids)
    assert manifest["analysis_name"] == analysis_name("cameo") and "Re-run" in manifest["cohort_note"]
    n_cells, n_proteins = len(LAYERS) ** 2, len(environment.ids)
    counts = manifest["row_counts"]
    assert counts["per_protein.csv"] == n_cells * n_proteins and counts["error_quantiles.csv"] == n_cells * 14
    assert counts["protein_anomalies.csv"] == n_cells * 5 and counts["error_position.csv"] == n_cells * N_POSITION_BINS
    assert 0 < counts["error_strata.csv"] <= n_cells * 13 * 17
    with (out / "per_protein.csv").open(newline="") as handle:
        assert sum(1 for _ in csv.DictReader(handle)) == n_cells * n_proteins

    environment.monkeypatch.setattr(figures, "DPI", 40)
    figures.main()
    written = {path.name for path in (out / "figures").iterdir()}
    assert {"01_transfer_pooled_r2_48x48.png", "02_transfer_offset_decay.png", "03_error_median_and_iqr_48x48.png",
            "05_native_error_fan_by_layer.png", "06_error_vs_final_structure_distance.png",
            "07_error_budget_strata.png", "08_error_along_chain.png", "09_protein_concentration.png"} <= written
    assert len(list((out / "figures/per_probe").glob("probe_layer_*.png"))) == len(LAYERS)
    assert "CAMEO proteins only" in figures.RESULT_NOTE

    manifest["analysis_status"] = "failed_parity"
    (out / "run_manifest.json").write_text(json.dumps(manifest))
    with pytest.raises(SystemExit, match="Refusing"):
        figures.main()


def test_aggregator_gate_fails_closed(environment) -> None:
    run_workers(environment)
    failure = [{"check": "x", "layer": 1, "metric": "m", "n_compared": 1, "max_abs_diff": 1.0,
                "tolerance": "t", "status": "fail", "detail": ""}]
    environment.monkeypatch.setattr(agg, "producer_parity_checks", lambda *_: failure)
    environment.monkeypatch.setattr(agg, "v1_overlap_checks", lambda *_: [])
    with pytest.raises(SystemExit):
        agg.main()
    manifest = json.loads((environment.output / "run_manifest.json").read_text())
    assert manifest["analysis_status"] == "failed_parity" and manifest["parity"]["n_failed"] == 1


def test_aggregation_marks_a_crashed_rerun_in_progress(environment) -> None:
    run_workers(environment)
    (environment.output / "run_manifest.json").write_text(json.dumps({"analysis_status": "complete"}))
    environment.monkeypatch.setattr(agg, "load_primitives", lambda *_: (_ for _ in ()).throw(OSError("crash")))
    with pytest.raises(OSError):
        agg.main()
    assert json.loads((environment.output / "run_manifest.json").read_text())["analysis_status"] == "in_progress"


def test_cohort_selection_and_naming(monkeypatch) -> None:
    bundle = cohort()
    monkeypatch.setitem(common.COHORT_SIZES, "cameo", 11)
    monkeypatch.setitem(common.COHORT_SIZES, "full", 14)
    assert len(select_cohort(bundle, "full")) == 14
    chosen = select_cohort(bundle, "cameo")
    assert len(chosen) == 11 and all(bundle.outer_by_protein[p].source == "cameo" for p in chosen)
    assert chosen == tuple(p for p in bundle.eligible_protein_ids if p in chosen)
    monkeypatch.setitem(common.COHORT_SIZES, "cameo", 10)
    with pytest.raises(ValueError, match="expected 10"):
        select_cohort(bundle, "cameo")
    monkeypatch.delenv("ALL_LAYER_COHORT", raising=False)
    with pytest.raises(ValueError, match="must be one of"):
        cohort_name()
    monkeypatch.setenv("ALL_LAYER_COHORT", "ecod")
    with pytest.raises(ValueError, match="must be one of"):
        cohort_name()
    assert analysis_name("full") == "cross_layer_probe_transfer_all_layers_v1"
    assert analysis_name("cameo") == "cross_layer_probe_transfer_all_layers_cameo_v1"
    assert analysis_label("cameo").endswith("_cameo") and not analysis_label("full").endswith("_cameo")
    assert default_output_root("cameo") != default_output_root("full")


def test_cohort_missing_a_fold_is_rejected(monkeypatch) -> None:
    bundle = cohort()
    for entry in bundle.outer_by_protein.values():
        entry.outer_fold = entry.outer_fold % 4
    monkeypatch.setitem(common.COHORT_SIZES, "full", 14)
    with pytest.raises(ValueError, match="every outer fold"):
        select_cohort(bundle, "full")


def test_native_self_check_detects_a_mismatch(environment, tmp_path) -> None:
    run_workers(environment)
    primitives, *_ = load_tables(environment)
    retained: list[dict[str, object]] = []
    rows = [r for r in agg.per_protein_rows(primitives, retained) if r["source_layer"] == r["target_layer"] == 47]
    path = tmp_path / "per_protein.csv"
    write_csv(path, rows)
    verify = NATIVE_CHECK
    shard = TargetAccumulator.from_file(environment.output / "layer_shards/target_layer_47.npz", 47,
                                        environment.ids, len(LAYERS))
    verify(shard, LAYERS.index(47), path)
    rows[3] = {**rows[3], "rmse": float(rows[3]["rmse"]) + 1e-4}
    write_csv(path, rows)
    with pytest.raises(ValueError, match="Native rmse"):
        verify(shard, LAYERS.index(47), path)


def test_common_exports_expected_scale() -> None:
    assert common.LAYERS == tuple(range(48)) and common.TRANSFER_MODE == "target_restandardized"
    assert len(common.SEQUENCE_LABELS) == common.N_SEQUENCE_BINS == 13
    assert len(common.DISTANCE_LABELS) == common.N_DISTANCE_BINS == 17

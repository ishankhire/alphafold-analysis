from __future__ import annotations

import csv
import math
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

SCRIPT_DIR = Path(__file__).resolve().parents[1]
CLUSTER_DIR = SCRIPT_DIR.parent
REPOSITORY = CLUSTER_DIR.parents[1]
for path in (SCRIPT_DIR, CLUSTER_DIR, CLUSTER_DIR / "cross_validation", CLUSTER_DIR / "triangle_violation_subgroups"):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import aggregate_cross_layer_transfer as agg  # noqa: E402
from cross_layer_transfer_common import (  # noqa: E402
    CATEGORY_LABELS, MODES, ROWS_PER_PROTEIN, TRANSFER_LAYERS, layer_group, predict_transfer,
    primitive_rows, row_stats, stats_payload, target_layer_for_task, transfer_identity,
    validate_target_shard,
)
from cross_validation_common import (  # noqa: E402
    OOFComparisonAccumulator, OOFSubgroupAccumulator, RegressionSufficientStatistics,
)
from regression_protocol import (  # noqa: E402
    SEQUENCE_BIN_LABELS, SPATIAL_DISTANCE_BIN_LABELS, sequence_separation_bin_codes,
    spatial_distance_bin_codes, strict_lower_triangle_indices,
)

PRODUCER = REPOSITORY / "delta_files/regression_5fold_v1"
LOCAL_DATA = REPOSITORY.parent / "protein_data"


def _fit(rng, mean_shift=0.0):
    return {
        "intercept_standardized": float(rng.normal(20, 1)),
        "coefficient_standardized": rng.normal(size=256),
        "feature_mean": rng.normal(size=256) + mean_shift,
        "feature_scale": rng.uniform(0.5, 2.0, size=256),
    }


def test_layer_layout_and_task_mapping() -> None:
    assert TRANSFER_LAYERS == (22, 23, 24, 25, 26, 43, 44, 45, 46, 47)
    assert [target_layer_for_task(i) for i in range(10)] == list(TRANSFER_LAYERS)
    with pytest.raises(ValueError):
        target_layer_for_task(10)
    assert layer_group(24) == "mid" and layer_group(45) == "end"
    with pytest.raises(ValueError):
        layer_group(30)
    identity = transfer_identity(43, 22, "raw")
    assert identity["transfer_block"] == "end_to_mid" and identity["is_same_layer"] is False
    assert ROWS_PER_PROTEIN == 241


def test_transfer_modes_match_hand_computation() -> None:
    rng = np.random.default_rng(0)
    source, target = _fit(rng), _fit(rng, mean_shift=3.0)
    x = rng.normal(size=(17, 256))
    raw = predict_transfer(x, source, target, "raw")
    restd = predict_transfer(x, source, target, "target_restandardized")
    expected_raw = source["intercept_standardized"] + ((x - source["feature_mean"]) / source["feature_scale"]) @ source["coefficient_standardized"]
    expected_restd = source["intercept_standardized"] + ((x - target["feature_mean"]) / target["feature_scale"]) @ source["coefficient_standardized"]
    np.testing.assert_allclose(raw, expected_raw, rtol=1e-13)
    np.testing.assert_allclose(restd, expected_restd, rtol=1e-13)
    assert not np.allclose(raw, restd)
    same = predict_transfer(x, source, source, "target_restandardized")
    assert np.array_equal(same, predict_transfer(x, source, source, "raw"))
    with pytest.raises(ValueError):
        predict_transfer(x, source, target, "bogus")


def test_primitive_rows_are_additive_and_reproduce_direct_metrics() -> None:
    rng = np.random.default_rng(1)
    n_residues = 30
    pair_i, pair_j = strict_lower_triangle_indices(n_residues)
    coordinates = rng.normal(scale=12, size=(n_residues, 3))
    y = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
    pred = y + rng.normal(scale=3, size=y.size)
    chains = ["A"] * 20 + ["B"] * 10
    sequence = sequence_separation_bin_codes(pair_i, pair_j, chain_ids=chains)
    spatial = spatial_distance_bin_codes(y)
    rows = primitive_rows(identity={"k": 1}, y_true=y, predictions=pred,
                          sequence_codes=sequence, spatial_codes=spatial)
    assert len(rows) == 12
    whole = next(r for r in rows if r["category_scope"] == "all_pairs")
    direct = RegressionSufficientStatistics()
    direct.update(y, pred)
    for key, value in direct.metrics().items():
        assert math.isclose(row_stats(whole).metrics()[key], value, rel_tol=1e-12, abs_tol=1e-12)
    spatial_total = sum(int(r["n"]) for r in rows if r["category_scope"] == "spatial_distance")
    sequence_total = sum(int(r["n"]) for r in rows if r["category_scope"] == "sequence_separation")
    assert spatial_total == y.size
    assert sequence_total == int(np.count_nonzero(sequence >= 0)) < y.size  # interchain excluded
    merged = RegressionSufficientStatistics()
    for r in rows:
        if r["category_scope"] == "spatial_distance":
            merged.merge(row_stats(r))
    assert math.isclose(merged.metrics()["rmse"], direct.metrics()["rmse"], rel_tol=1e-12)


def _synthetic_primitives(n_proteins: int = 10, seed: int = 2):
    """Build complete primitives plus the raw arrays needed for direct checks."""
    rng = np.random.default_rng(seed)
    proteins = [f"p{i:02d}" for i in range(n_proteins)]
    metadata, arrays = {}, {}
    for index, protein in enumerate(proteins):
        n_residues = int(rng.integers(6, 12))
        pair_i, pair_j = strict_lower_triangle_indices(n_residues)
        coordinates = rng.normal(scale=10, size=(n_residues, 3))
        y = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
        metadata[protein] = {"source": "ecod", "outer_fold": index % 5, "n_residues": n_residues}
        arrays[protein] = (y, sequence_separation_bin_codes(pair_i, pair_j), spatial_distance_bin_codes(y))
    primitives = agg.TransferPrimitives(proteins, metadata)
    predictions = {}
    for target in TRANSFER_LAYERS:
        rows = []
        for protein in proteins:
            y, sequence, spatial = arrays[protein]
            base = {"analysis_name": "cross_layer_probe_transfer_v1", "target_layer": target,
                    "protein_id": protein, **metadata[protein]}
            for source in TRANSFER_LAYERS:
                shared_noise = rng.normal(scale=2, size=y.size)
                for mode in MODES:
                    pred = y + (shared_noise if source == target else rng.normal(scale=4, size=y.size))
                    predictions[(source, target, mode, protein)] = pred
                    rows.extend(primitive_rows(
                        identity={**base, "source_layer": source, "transfer_mode": mode, "predictor": "probe"},
                        y_true=y, predictions=pred, sequence_codes=sequence, spatial_codes=spatial,
                    ))
            baseline = np.full(y.size, 22.5 + metadata[protein]["outer_fold"])
            rows.append({**base, "source_layer": "", "transfer_mode": "", "predictor": "global_baseline",
                         "category_scope": "all_pairs", "category_label": "all", **stats_payload(y, baseline)})
        assert validate_target_shard(rows, target, expected_proteins=n_proteins) == set(proteins)
        primitives.add_shard(target, rows)
    primitives.require_complete()
    fold_bundle = SimpleNamespace(protein_ids_for_outer_fold=lambda fold: tuple(
        p for p in proteins if metadata[p]["outer_fold"] == fold
    ))
    return primitives, fold_bundle, arrays, predictions, metadata


def test_aggregated_tables_equal_direct_concatenated_metrics() -> None:
    primitives, fold_bundle, arrays, predictions, metadata = _synthetic_primitives()
    tables = agg.build_metric_tables(primitives, fold_bundle)
    assert len(tables["pooled"]) == 200
    assert len(tables["fold"]) == 1000
    assert len(tables["fold_summary"]) == 200 * 12
    assert len(tables["sequence_separation"]) == 200 * len(SEQUENCE_BIN_LABELS)
    assert len(tables["spatial_distance"]) == 200 * len(SPATIAL_DISTANCE_BIN_LABELS)
    per_protein = list(agg.per_protein_rows(primitives))
    assert len(per_protein) == 200 * len(primitives.protein_ids)
    assert len(list(agg.per_protein_subgroup_rows(primitives))) == 200 * len(primitives.protein_ids) * 11

    source, target, mode = 43, 24, "target_restandardized"
    direct = OOFComparisonAccumulator()
    direct_sequence = OOFSubgroupAccumulator("sequence_separation", SEQUENCE_BIN_LABELS)
    for protein in primitives.protein_ids:
        y, sequence, _ = arrays[protein]
        pred = predictions[(source, target, mode, protein)]
        direct.update_protein(protein, metadata[protein]["outer_fold"], y, pred,
                              22.5 + metadata[protein]["outer_fold"])
        direct_sequence.update_protein(protein, y, pred, sequence)
    pooled = next(r for r in tables["pooled"] if (r["source_layer"], r["target_layer"], r["transfer_mode"]) == (source, target, mode))
    expected = direct.model.metrics()
    for key in ("r2", "mae", "rmse", "bias_pred_minus_true"):
        assert math.isclose(pooled[f"pooled_{key}"], expected[key], rel_tol=1e-12, abs_tol=1e-12)
    baseline = direct.global_baseline.metrics()
    assert math.isclose(pooled["global_baseline_pooled_mae"], baseline["mae"], rel_tol=1e-12)
    assert pooled["layer"] == target and pooled["analysis_name"].endswith("target_restandardized")
    sub = [r for r in tables["sequence_separation"] if (r["source_layer"], r["target_layer"], r["transfer_mode"]) == (source, target, mode)]
    identity = dict(analysis_name="x", model_family="shared_ols", feature_dim=256, layer=target,
                    evaluation_design="protein_outer_5fold", oof_prediction_scope="heldout_proteins_all_pairs")
    for ours, theirs in zip(sub, direct_sequence.rows(**identity)):
        assert ours["support_status"] == theirs["support_status"]
        assert ours["n_proteins"] == theirs["n_proteins"]
        for key in ("pooled_r2", "pooled_mae", "pooled_rmse", "pooled_bias_pred_minus_true"):
            if math.isnan(theirs[key]):
                assert math.isnan(ours[key])
            else:
                assert math.isclose(ours[key], theirs[key], rel_tol=1e-12, abs_tol=1e-12)

    matrix = agg.transfer_matrix_rows(tables["pooled"], per_protein)
    assert len(matrix) == 200
    diagonal = next(r for r in matrix if r["source_layer"] == r["target_layer"] == 47 and r["transfer_mode"] == "raw")
    assert diagonal["delta_pooled_vs_native_target_r2"] == 0.0
    assert agg.baseline_consistent(primitives)
    # Same-layer modes share identical noise, so their primitives agree exactly.
    t = TRANSFER_LAYERS.index(25)
    assert np.array_equal(primitives.probe[t, t, 0], primitives.probe[t, t, 1])


def test_shard_validation_rejects_partial_or_duplicate_rows() -> None:
    primitives, *_ = _synthetic_primitives(n_proteins=5, seed=3)
    rows = [{"analysis_name": "cross_layer_probe_transfer_v1", "target_layer": 22, "source_layer": 22,
             "transfer_mode": "raw", "predictor": "probe", "protein_id": "p0", "source": "ecod",
             "outer_fold": 0, "n_residues": 5, "category_scope": "all_pairs", "category_label": "all",
             **stats_payload(np.ones(10), np.ones(10))}]
    with pytest.raises(ValueError, match="partial"):
        validate_target_shard(rows, 22, expected_proteins=None)
    with pytest.raises(ValueError, match="Duplicate"):
        validate_target_shard(rows * 2, 22, expected_proteins=None)
    with pytest.raises(ValueError, match="wrong target"):
        validate_target_shard(rows, 23, expected_proteins=None)


def test_parity_comparison_flags_drift() -> None:
    assert agg._compare("c", 22, "r2", [(0.5, 0.5 + 1e-12)])["status"] == "pass"
    assert agg._compare("c", 22, "r2", [(0.5, 0.5 + 1e-6)])["status"] == "fail"
    assert agg._compare("c", 22, "r2", [(-6000.0, -6000.0 - 1e-6)])["status"] == "pass"  # scaled tolerance
    assert agg._compare("c", 22, "r2", [(float("nan"), 1.0)])["status"] == "fail"
    assert agg._compare("c", 22, "r2", [])["status"] == "fail"


def test_stream_writer_round_trips_floats(tmp_path) -> None:
    path = tmp_path / "out.csv"
    value = 0.1 + 0.2
    assert agg.stream_write_csv(path, ("a", "b"), iter([{"a": value, "b": "x"}])) == 1
    with path.open(newline="") as handle:
        row = next(csv.DictReader(handle))
    assert float(row["a"]) == value
    with pytest.raises(ValueError):
        agg.stream_write_csv(path, ("a",), iter([{"a": 1, "extra": 2}]))


needs_producer = pytest.mark.skipif(
    not (PRODUCER / "shared_ols_base256/model_artifacts/replay_index.csv").is_file(),
    reason="local frozen shared-OLS artifacts are unavailable",
)


@needs_producer
def test_real_artifacts_load_with_layer_invariant_baselines() -> None:
    from cross_layer_transfer_common import global_baseline_by_fold, load_transfer_artifacts
    from cross_validation_common import load_cv_folds

    fold_bundle = load_cv_folds(metadata_dir=PRODUCER / "metadata")
    models, provenance, status = load_transfer_artifacts(PRODUCER, fold_bundle)
    assert len(models) == 50 and len(status) == 50 and len(provenance["artifacts"]) == 50
    assert all(row["fit_status"].startswith("fitted") and not row["support_flags"] for row in status)
    assert set(global_baseline_by_fold(models)) == set(range(5))


@needs_producer
@pytest.mark.skipif(not (LOCAL_DATA / "7b3a_A/pair_blocks").is_dir(), reason="local pair blocks unavailable")
def test_real_diagonal_cells_reproduce_producer_per_protein(monkeypatch) -> None:
    import cross_layer_transfer_replay as worker
    import triangle_violation_subgroup_replay as replay
    from cross_layer_transfer_common import global_baseline_by_fold, load_transfer_artifacts
    from cross_validation_common import load_cv_folds

    monkeypatch.setattr(replay, "resolve_combined_pdb_path",
                        lambda pid, src: LOCAL_DATA / pid / "predicted_pdb" / f"{pid}_model_1_ptm_unrelaxed.pdb")
    monkeypatch.setattr(replay, "resolve_combined_pair_block_path",
                        lambda pid, src, layer: LOCAL_DATA / pid / "pair_blocks" / f"{pid}_pair_block_{layer}.npy")
    fold_bundle = load_cv_folds(metadata_dir=PRODUCER / "metadata")
    models, _, _ = load_transfer_artifacts(PRODUCER, fold_bundle)
    with (PRODUCER / "shared_ols_base256/per_protein.csv").open(newline="") as handle:
        producer = {(int(r["layer"]), r["protein_id"]): r for r in csv.DictReader(handle)}
    protein = "7b3a_A"
    for target in (22, 47):
        rows = worker.protein_rows(fold_bundle.outer_by_protein[protein], target, models,
                                   global_baseline_by_fold(models))
        assert len(rows) == ROWS_PER_PROTEIN
        whole = next(r for r in rows if r["predictor"] == "probe" and r["source_layer"] == target
                     and r["transfer_mode"] == "raw" and r["category_scope"] == "all_pairs")
        metrics = row_stats(whole).metrics()
        for key in ("r2", "mae", "rmse", "bias_pred_minus_true", "mean_predicted_distance"):
            assert abs(metrics[key] - float(producer[(target, protein)][key])) < 1e-8
        counts = {scope: sum(int(r["n"]) for r in rows if r["predictor"] == "probe" and r["source_layer"] == target
                             and r["transfer_mode"] == "raw" and r["category_scope"] == scope)
                  for scope in CATEGORY_LABELS}
        assert counts["spatial_distance"] == counts["all_pairs"]


LOCAL_E2E_PROTEINS = ("6tf4_A", "6wmk_A", "6wqc_A", "6zpp_A", "7ab3_E", "7ar0_B", "7b3a_A")


@needs_producer
@pytest.mark.skipif(
    not all((LOCAL_DATA / p / "pair_blocks").is_dir() for p in LOCAL_E2E_PROTEINS),
    reason="local pair blocks unavailable",
)
def test_end_to_end_worker_and_aggregator_on_local_subset(monkeypatch, tmp_path) -> None:
    """Run every array task and the aggregator on seven local cohort proteins.

    The subset spans all five outer folds. Diagonal per-protein metrics must
    match the producer; cohort-level parity correctly fails because the
    producer pools all 552 proteins.
    """
    import json

    import cross_layer_transfer_replay as worker
    import triangle_violation_subgroup_replay as replay
    from cross_validation_common import load_cv_folds

    full = load_cv_folds(metadata_dir=PRODUCER / "metadata")
    subset = {p: full.outer_by_protein[p] for p in LOCAL_E2E_PROTEINS}
    assert {int(entry.outer_fold) for entry in subset.values()} == set(range(5))
    reduced = SimpleNamespace(
        eligible_protein_ids=tuple(sorted(subset)), outer_by_protein=subset,
        fold_digests=full.fold_digests,
        protein_ids_for_outer_fold=lambda fold: tuple(sorted(
            p for p, entry in subset.items() if int(entry.outer_fold) == fold
        )),
    )
    monkeypatch.setenv("CSV_ROOT", str(REPOSITORY / "delta_files"))
    monkeypatch.setenv("CROSS_LAYER_TRANSFER_OUTPUT_ROOT", str(tmp_path / "out"))
    monkeypatch.setenv("CROSS_LAYER_CHECKPOINT_PROTEINS", "3")
    for module in (worker, agg):
        monkeypatch.setattr(module, "load_cv_folds", lambda **_: reduced)
        monkeypatch.setattr(module, "EXPECTED_PROTEINS", len(subset))
    monkeypatch.setattr(replay, "resolve_combined_pdb_path",
                        lambda pid, src: LOCAL_DATA / pid / "predicted_pdb" / f"{pid}_model_1_ptm_unrelaxed.pdb")
    monkeypatch.setattr(replay, "resolve_combined_pair_block_path",
                        lambda pid, src, layer: LOCAL_DATA / pid / "pair_blocks" / f"{pid}_pair_block_{layer}.npy")
    for task in range(len(TRANSFER_LAYERS)):
        monkeypatch.setenv("SLURM_ARRAY_TASK_ID", str(task))
        worker.main()
    shard = tmp_path / "out/layer_shards/cross_layer_primitives_target_layer_47.csv"
    assert shard.is_file()
    assert not (tmp_path / "out/layer_shards/.cross_layer_primitives_target_layer_47.checkpoint.csv").exists()

    with pytest.raises(SystemExit):
        agg.main()
    out = tmp_path / "out"
    manifest = json.loads((out / "run_manifest.json").read_text())
    assert manifest["analysis_status"] == "failed_parity"
    assert manifest["row_counts"]["per_protein.csv"] == 200 * len(subset)
    assert manifest["row_counts"]["per_protein_subgroup_mae.csv"] == 200 * len(subset) * 11
    assert manifest["row_counts"]["pooled_metrics.csv"] == 200
    with (out / "parity_checks.csv").open(newline="") as handle:
        checks = list(csv.DictReader(handle))
    per_protein_checks = [c for c in checks if c["check"] == "diagonal_per_protein_vs_producer"]
    assert len(per_protein_checks) == 80
    assert all(float(c["max_abs_diff"]) < 1e-8 for c in per_protein_checks)
    assert all(c["status"] == "pass" for c in checks if c["check"] in (
        "diagonal_restandardized_equals_raw", "global_baseline_identical_across_target_shards",
    ))
    with (out / "transfer_matrix.csv").open(newline="") as handle:
        matrix = list(csv.DictReader(handle))
    assert len(matrix) == 200
    assert {row["transfer_block"] for row in matrix} == {"mid_to_mid", "mid_to_end", "end_to_mid", "end_to_end"}

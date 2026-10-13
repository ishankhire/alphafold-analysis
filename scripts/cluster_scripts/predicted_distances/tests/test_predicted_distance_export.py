from __future__ import annotations

import csv
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
import predicted_distance_export_common as common
import prepare_predicted_distance_export as preparation
import export_predicted_distances_layer as worker
import finalize_predicted_distance_export as finalizer
from regression_protocol import pair_fold_assignment_digest, within_protein_pair_folds


class FakeBooster:
    def __init__(self, fold):
        self.fold = fold

    def inplace_predict(self, features):
        return (features[:, 0] * 0.7 + self.fold).astype(np.float32)

    def num_boosted_rounds(self):
        return 700

    def num_features(self):
        return 256


def ols(intercept):
    coefficient = np.zeros(256)
    coefficient[0] = 0.8
    return dict(coefficient_standardized=coefficient, intercept_standardized=intercept,
                feature_mean=np.zeros(256), feature_scale=np.ones(256), fit_status="fitted")


@pytest.fixture
def pipeline(tmp_path, monkeypatch):
    output, inputs, producers = (tmp_path / name for name in ("export", "inputs", "models"))
    inputs.mkdir()
    monkeypatch.setenv("PREDICTED_DISTANCE_OUTPUT_ROOT", str(output))
    monkeypatch.setattr(common, "LAYERS", (0, 1))
    # Upstream loaders remain strict in production; this fixture supplies frozen
    # model state without requiring 552 tensors or a native XGBoost installation.
    outers, pairs, pdbs, tensors = {}, {}, {}, {}
    for pid, residues, fold in (("000000022", 55, 1), ("toy_B", 11, 3)):
        outer = SimpleNamespace(protein_id=pid, source="ecod", outer_fold=fold,
                                n_residues=residues, n_pairs=residues * (residues - 1) // 2)
        outers[pid] = outer
        assignments = within_protein_pair_folds(residues, pid)
        pairs[pid] = SimpleNamespace(assignment_digest=pair_fold_assignment_digest(residues, pid),
                                     fold_counts=tuple(np.bincount(assignments, minlength=5)))
        pdb = inputs / f"{pid}.pdb"
        pdb.write_text("".join(
            f"ATOM  {i+1:5d}  CA  ALA A{i+1:4d}{'A' if i == 12 else ' '}   "
            f"{i*0.2:8.3f}{np.sin(i)*2:8.3f}{np.cos(i):8.3f}  1.00 20.00           C\n"
            for i in range(residues)
        ) + "END\n")
        pdbs[pid] = pdb
        for layer in common.LAYERS:
            path = inputs / f"{pid}-{layer}.npy"
            rng = np.random.default_rng(residues + layer)
            np.save(path, rng.normal(size=(residues, residues, 128)).astype(np.float32))
            tensors[(pid, layer)] = path
    bundle = SimpleNamespace(eligible_protein_ids=tuple(outers), outer_by_protein=outers,
                             pair_by_protein=pairs, fold_digests={"cohort_digest": "synthetic"})
    monkeypatch.setattr(common, "load_bundle", lambda: bundle)
    monkeypatch.setattr(common, "producer_root", lambda: producers)
    monkeypatch.setattr(common.replay, "resolve_combined_pdb_path", lambda pid, source: pdbs[pid])
    monkeypatch.setattr(common.replay, "resolve_combined_pair_block_path", lambda pid, source, layer: tensors[(pid, layer)])
    identity = {
        "export_schema": common.EXPORT_SCHEMA, "layers": list(common.LAYERS), "producers": {
            s.analysis_name: {"oof_prediction_scope": (
                "heldout_proteins_all_pairs_recombined_sequence_gates" if s.uses_sequence_gates else s.oof_prediction_scope),
                "files": {"run_manifest.json": "manifest", "model_artifacts/replay_index.csv": "index"}}
            for s in common.FAMILY_SPECS
        },
    }
    monkeypatch.setattr(common, "source_identity", lambda bundle, **kwargs: identity)

    def models(layer, bundle):
        shared = {f: ols(-2 + f + layer * 0.1) for f in range(5)}
        gated = {(f, gate): ols(f + code * 0.2 + layer * 0.1)
                 for f in range(5) for code, gate in enumerate(common.replay.SEQUENCE_BIN_LABELS)}
        ids, folds, coefficients, intercepts = [], [], [], []
        for pid in outers:
            for fold in range(5):
                ids.append(pid)
                folds.append(fold)
                coefficients.append(ols(0)["coefficient_standardized"])
                intercepts.append(fold * 0.3 + layer * 0.1)
        shard = dict(protein_id=np.array(ids), pair_fold=np.array(folds),
                     coefficient_standardized=np.array(coefficients),
                     intercept_standardized=np.array(intercepts),
                     feature_mean=np.zeros((len(ids), 256)), feature_scale=np.ones((len(ids), 256)),
                     fit_status=np.array(["fitted"] * len(ids)))
        artifacts = {
            "shared_ols_base256": shared, "per_protein_ols_base256": (shard, dict(zip(zip(ids, folds), range(len(ids))))),
            "sequence_gated_ols_base256": gated,
            "shared_xgboost_base256_fixed_s4": {f: FakeBooster(f) for f in range(5)},
        }
        provenance = {"producers": {s.analysis_name: {"run_manifest_sha256": "manifest", "replay_index_sha256": "index"}
                                     for s in common.FAMILY_SPECS}}
        return artifacts, provenance

    monkeypatch.setattr(common.replay, "load_layer_artifacts", models)
    expected = {}
    for layer in common.LAYERS:
        artifacts, _ = models(layer, bundle)
        aggregates = {s.analysis_name: common.RegressionSufficientStatistics() for s in common.FAMILY_SPECS}
        protein_rows = {s.analysis_name: [] for s in common.FAMILY_SPECS}
        geometry_rows = {s.analysis_name: [] for s in common.FAMILY_SPECS}
        for pid, outer in outers.items():
            coords, chains, i, j, features = common.replay.load_protein(outer, layer)
            predictions, _ = common.replay.model_predictions(outer, chains, i, j, features, artifacts, bundle)
            targets = np.linalg.norm(coords[i] - coords[j], axis=1)
            for spec, values in predictions:
                expected[(layer, pid, spec.analysis_name)] = values
                stats = common.statistics(targets, values)
                aggregates[spec.analysis_name].merge(stats)
                row = common.identity_row(spec, outer, layer, {"identity": identity})
                row.update(n_residues=outer.n_residues, pair_coverage=1,
                           prediction_status="complete", n_expected_pairs=outer.n_pairs,
                           n_evaluated_pairs=outer.n_pairs, n_protein_pairs=outer.n_pairs, **stats.metrics())
                protein_rows[spec.analysis_name].append(row)
                geometry_rows[spec.analysis_name].append(dict(row, geometry_status="complete",
                                                            n_negative_predictions=int(np.count_nonzero(values < 0))))
        for spec in common.FAMILY_SPECS:
            directory = producers / spec.family_dir
            directory.mkdir(parents=True, exist_ok=True)
            for filename, rows in (("per_protein.csv", protein_rows[spec.analysis_name]),
                                   ("geometry_metrics.csv", geometry_rows[spec.analysis_name])):
                path = directory / filename
                with path.open("a", newline="") as handle:
                    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
                    if layer == 0:
                        writer.writeheader()
                    writer.writerows(rows)
            row = dict(layer=layer, n_proteins=len(outers), n_evaluated_pairs=sum(o.n_pairs for o in outers.values()),
                       **{f"pooled_{k}": v for k, v in aggregates[spec.analysis_name].metrics().items()})
            with (directory / "pooled_metrics.csv").open("a", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=list(row))
                if layer == 0:
                    writer.writeheader()
                writer.writerow(row)
    return SimpleNamespace(root=output, bundle=bundle, expected=expected, identity=identity,
                           tensors=tensors, pdbs=pdbs, models=models, producers=producers)


def test_end_to_end_uses_one_file_per_pair_population_and_never_aggregates(pipeline):
    preparation.prepare()
    preparation.prepare()  # Compatible preparation is idempotent.
    residue_rows = common.read_csv(pipeline.root / "metadata/residue_index.csv")
    assert next(r for r in residue_rows if r["protein_id"] == "000000022" and r["residue_index"] == "12")["insertion_code"] == "A"
    for layer in common.LAYERS:
        worker.run_layer(layer)
        worker.run_layer(layer)  # Complete-layer resume verifies instead of rewriting.
    hashes = {p: common.sha256_file(p) for p in pipeline.root.rglob("*.npy")}
    finalizer.finalize()
    finalizer.finalize()
    assert hashes == {p: common.sha256_file(p) for p in pipeline.root.rglob("*.npy")}
    assert len(hashes) == 16
    manifest = common.load_json(pipeline.root / "run_manifest.json")
    assert manifest["run_status"] == "complete"
    assert manifest["completed_layers"] == [0, 1]
    assert len(common.read_csv(pipeline.root / "prediction_index.csv")) == 16
    assert len(common.read_csv(pipeline.root / "pooled_validation.csv")) == 8
    assert not list(pipeline.root.glob("*.npy"))
    assert not list((pipeline.root / ".checkpoints").rglob("*.npy*"))
    for (layer, pid, family), expected in pipeline.expected.items():
        actual = common.load_predicted_distances(pipeline.root, layer, pid, family)
        np.testing.assert_array_equal(actual, expected.astype("<f4"))
        matrix = common.load_predicted_distances(pipeline.root, layer, pid, family, as_matrix=True)
        np.testing.assert_array_equal(matrix, matrix.T)
        assert np.all(np.diag(matrix) == 0)
        i, j = np.tril_indices(matrix.shape[0], k=-1)
        np.testing.assert_array_equal(actual, matrix[i, j])
        np.testing.assert_array_equal(actual[i * (i - 1) // 2 + j], actual)
    assert np.any(common.load_predicted_distances(pipeline.root, 0, "000000022", "shared_ols_base256") < 0)


def test_actual_routing_uses_outer_fold_pair_fold_and_sequence_gate(pipeline):
    # These assertions use the real existing model_predictions, not a mock of
    # routing; exporter integration above verifies these vectors reach disk.
    outer = pipeline.bundle.outer_by_protein["000000022"]
    _, _, i, j, features = common.replay.load_protein(outer, 0)
    np.testing.assert_allclose(pipeline.expected[(0, outer.protein_id, "shared_ols_base256")],
                               features[:, 0] * 0.8 - 1)
    assignments = within_protein_pair_folds(outer.n_residues, outer.protein_id)
    np.testing.assert_allclose(pipeline.expected[(0, outer.protein_id, "per_protein_ols_base256")],
                               features[:, 0] * 0.8 + assignments * 0.3)
    codes = common.replay.sequence_separation_bin_codes(i, j, chain_ids=["A"] * outer.n_residues)
    assert set(codes) == set(range(6))
    np.testing.assert_allclose(pipeline.expected[(0, outer.protein_id, "sequence_gated_ols_base256")],
                               features[:, 0] * 0.8 + 1 + codes * 0.2)


def test_interrupt_resume_preserves_committed_protein_and_recomputes_uncommitted(pipeline, monkeypatch):
    preparation.prepare()
    original = common.atomic_save_prediction
    calls = []

    def interrupted(root, layer, pid, *args):
        calls.append(pid)
        if pid == "toy_B":
            if calls.count(pid) == 2:
                raise RuntimeError("simulated interrupted write")
        return original(root, layer, pid, *args)

    monkeypatch.setattr(common, "atomic_save_prediction", interrupted)
    with pytest.raises(RuntimeError, match="interrupted"):
        worker.run_layer(0)
    progress = common.load_json(pipeline.root / ".checkpoints/layer_00/progress.json")
    assert set(progress["records"]) == {"000000022"}
    assert "last_failure" in progress
    assert len(list((pipeline.root / "layers/layer_00/toy_B").glob("*.npy"))) == 1
    hashes = {p: common.sha256_file(p) for p in (pipeline.root / "layers/layer_00/000000022").glob("*.npy")}
    monkeypatch.setattr(common, "atomic_save_prediction", original)
    worker.run_layer(0)
    assert hashes == {p: common.sha256_file(p) for p in hashes}
    assert len(list((pipeline.root / "layers/layer_00").rglob("*.npy"))) == 8


@pytest.mark.parametrize("failure", ["missing_layer", "missing_array", "corrupt_array", "extra_array", "extra_layer", "duplicate_index"])
def test_finalization_rejects_incomplete_or_corrupt_inventory(pipeline, failure):
    preparation.prepare()
    worker.run_layer(0)
    if failure != "missing_layer":
        worker.run_layer(1)
    directory = pipeline.root / "layers/layer_00"
    array = directory / "000000022/shared_ols_base256.npy"
    if failure == "missing_array":
        array.unlink()
    elif failure == "corrupt_array":
        content = bytearray(array.read_bytes())
        content[-1] ^= 1
        array.write_bytes(content)
    elif failure == "extra_array":
        np.save(directory / "extra.npy", np.zeros(1, dtype=np.float32))
    elif failure == "extra_layer":
        (pipeline.root / "layers/layer_99").mkdir()
    elif failure == "duplicate_index":
        path = directory / "prediction_index.csv"
        path.write_text(path.read_text() + path.read_text().splitlines()[1] + "\n")
        manifest_path = directory / "layer_manifest.json"
        manifest = common.load_json(manifest_path)
        manifest["metadata_sha256"]["prediction_index.csv"] = common.sha256_file(path)
        common.atomic_write_json(manifest_path, manifest)
    with pytest.raises((ValueError, FileNotFoundError)):
        finalizer.finalize()
    assert common.load_json(pipeline.root / "run_manifest.json")["run_status"] == "validation_failed"


@pytest.mark.parametrize("change", ["producer_identity", "pdb", "checkpoint_array", "pair_tensor"])
def test_resume_rejects_changed_inputs_or_committed_predictions(pipeline, monkeypatch, change):
    preparation.prepare()
    original = common.replay.load_protein

    def interrupted(outer, layer):
        if outer.protein_id == "toy_B":
            raise RuntimeError("interrupt")
        return original(outer, layer)

    monkeypatch.setattr(common.replay, "load_protein", interrupted)
    with pytest.raises(RuntimeError):
        worker.run_layer(0)
    monkeypatch.setattr(common.replay, "load_protein", original)
    if change == "producer_identity":
        pipeline.identity["changed"] = True
    elif change == "pdb":
        with pipeline.pdbs["000000022"].open("a") as f:
            f.write("REMARK changed\n")
    else:
        path = (pipeline.tensors[("000000022", 0)] if change == "pair_tensor"
                else pipeline.root / "layers/layer_00/000000022/shared_ols_base256.npy")
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
    with pytest.raises(ValueError):
        worker.run_layer(0)


@pytest.mark.parametrize("failure", ["missing_gate", "bad_pair_digest", "metric_mismatch", "negative_count", "bad_booster"])
def test_prediction_failures_cannot_publish_complete_layer(pipeline, monkeypatch, failure):
    preparation.prepare()
    if failure == "bad_pair_digest":
        # Preparation identity still uses fixture metadata; the real router
        # separately validates the regenerated assignment digest.
        pipeline.bundle.pair_by_protein["000000022"].assignment_digest = "changed"
        # Exercise routing directly because worker preparation rejects this too.
        outer = pipeline.bundle.outer_by_protein["000000022"]
        _, chains, i, j, features = common.replay.load_protein(outer, 0)
        artifacts, _ = pipeline.models(0, pipeline.bundle)
        with pytest.raises(ValueError, match="pair folds"):
            common.replay.model_predictions(outer, chains, i, j, features, artifacts, pipeline.bundle)
        return
    if failure in {"missing_gate", "bad_booster"}:
        def models(layer, bundle):
            artifacts, provenance = pipeline.models(layer, bundle)
            if failure == "missing_gate":
                del artifacts["sequence_gated_ols_base256"][(1, "1")]
            else:
                artifacts["shared_xgboost_base256_fixed_s4"][1].num_boosted_rounds = lambda: 699
            return artifacts, provenance
        monkeypatch.setattr(common.replay, "load_layer_artifacts", models)
    else:
        path = pipeline.producers / "shared_ols_base256" / ("per_protein.csv" if failure == "metric_mismatch" else "geometry_metrics.csv")
        rows = common.read_csv(path)
        rows[0]["r2" if failure == "metric_mismatch" else "n_negative_predictions"] = "999"
        common.atomic_write_csv(path, list(rows[0]), rows)
    with pytest.raises((ValueError, KeyError)):
        worker.run_layer(0)
    assert not (pipeline.root / "layers/layer_00/layer_manifest.json").exists()


def test_float32_overflow_and_escaping_path_are_rejected(pipeline):
    spec = common.FAMILY_SPECS[0]
    with pytest.raises(ValueError, match="overflow"):
        common.atomic_save_prediction(pipeline.root, 0, "000000022", spec, np.array([1e100]))
    with pytest.raises(ValueError):
        common.prediction_path(0, "../escape", spec.analysis_name)


def test_independent_output_root_and_exclusive_task_lock(tmp_path, monkeypatch):
    monkeypatch.delenv("PREDICTED_DISTANCE_OUTPUT_ROOT", raising=False)
    monkeypatch.setenv("CSV_ROOT", str(tmp_path / "input_models"))
    assert str(common.output_root()) == "/work/hdd/bfna/ikhire/regression_5fold_v1/predicted_distances_5fold_v1"
    with common.task_lock(tmp_path, "layer_00"):
        with pytest.raises(RuntimeError, match="Another"):
            with common.task_lock(tmp_path, "layer_00"):
                pass


def test_flagged_finite_fit_is_retained_and_marked(pipeline, monkeypatch):
    path = pipeline.producers / "shared_ols_base256/per_protein.csv"
    rows = common.read_csv(path)
    for row in rows:
        if row["protein_id"] == "000000022":
            row["prediction_status"] = "complete_from_flagged_fit"
    common.atomic_write_csv(path, list(rows[0]), rows)

    def flagged_models(layer, bundle):
        artifacts, provenance = pipeline.models(layer, bundle)
        artifacts["shared_ols_base256"][1]["fit_status"] = "fitted_rank_deficient"
        return artifacts, provenance

    monkeypatch.setattr(common.replay, "load_layer_artifacts", flagged_models)
    preparation.prepare()
    worker.run_layer(0)
    rows = common.read_csv(pipeline.root / "layers/layer_00/validation.csv")
    row = next(r for r in rows if r["protein_id"] == "000000022" and r["analysis_name"] == "shared_ols_base256")
    assert row["prediction_status"] == "complete_from_flagged_fit"
    assert row["validation_status"] == "complete"
    np.testing.assert_array_equal(common.load_predicted_distances(pipeline.root, 0, "000000022", "shared_ols_base256"),
                                  pipeline.expected[(0, "000000022", "shared_ols_base256")].astype(np.float32))


@pytest.mark.parametrize("failure", ["nonfinite", "incomplete_pairs", "missing_family"])
def test_every_pair_requires_one_finite_prediction(pipeline, monkeypatch, failure):
    preparation.prepare()
    real_prediction = common.replay.model_predictions

    def broken_predictions(*args):
        predictions, assignments = real_prediction(*args)
        if failure == "nonfinite":
            predictions[0][1][-1] = np.nan
        elif failure == "incomplete_pairs":
            spec, values = predictions[0]
            predictions[0] = (spec, values[:-1])
        else:
            predictions.pop()
        return predictions, assignments

    monkeypatch.setattr(common.replay, "model_predictions", broken_predictions)
    with pytest.raises(ValueError):
        worker.run_layer(0)
    assert not (pipeline.root / "layers/layer_00/layer_manifest.json").exists()
    assert not list(pipeline.root.rglob("*.npy"))


def test_failed_revalidation_revokes_previously_complete_run(pipeline):
    preparation.prepare()
    for layer in common.LAYERS:
        worker.run_layer(layer)
    finalizer.finalize()
    path = pipeline.root / "layers/layer_00/toy_B/shared_ols_base256.npy"
    path.unlink()
    with pytest.raises(FileNotFoundError):
        finalizer.finalize()
    assert common.load_json(pipeline.root / "run_manifest.json")["run_status"] == "validation_failed"

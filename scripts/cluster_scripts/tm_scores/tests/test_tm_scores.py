from __future__ import annotations

import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Event
from types import SimpleNamespace

import numpy as np
import pytest

DIRECTORY = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(DIRECTORY))
import recompute_tm_scores as worker
import consolidate_tm_scores as consolidation
import cluster_regression_helpers as geometry


@pytest.fixture
def backend():
    try:
        return geometry.resolve_tmscore_binary()
    except geometry.TMScoreError:
        pytest.skip("Install the official TMscore binary or set TMSCORE_BINARY")


def test_native_alignment_preserves_rigid_and_mirror_equivalence(backend):
    rng = np.random.default_rng(54)
    fixed = rng.normal(size=(30, 3)) * 8
    rotation, _ = np.linalg.qr(rng.normal(size=(3, 3)))
    rotation[:, 0] *= np.linalg.det(rotation)
    moving = fixed @ rotation + [100, -200, 50]
    assert geometry.optimize_tm_score(fixed, moving, 30) == pytest.approx(1, abs=1e-12)
    moving[:, 2] *= -1
    assert geometry.compute_mds_tm_score(fixed, moving, 30) == pytest.approx(1, abs=1e-12)
    assert geometry.optimize_tm_score(fixed, moving, 30) < 0.99


def test_native_search_improves_outliers_and_keeps_correspondence(backend):
    fixed = np.random.default_rng(9).normal(size=(30, 3))
    moving = fixed.copy()
    moving[-5:] += 15
    aligned, _ = geometry.superimpose_structures(fixed, moving)
    baseline = geometry.compute_tm_score(fixed, aligned, 30)
    optimized = geometry.optimize_tm_score(fixed, moving, 30)
    assert optimized > baseline + 0.5
    assert geometry.optimize_tm_score(fixed, fixed, 60) == pytest.approx(0.5, abs=1e-12)
    permutation = fixed.copy()
    permutation[[0, 1, 2, 3]] = permutation[[3, 2, 1, 0]]
    assert geometry.optimize_tm_score(fixed, permutation, 30) < 0.999


def test_winning_coordinates_preserve_reference_frame_and_reflection(backend):
    rng = np.random.default_rng(18)
    reference = rng.normal(size=(30, 3)) * 8 + [110, -210, 70]
    moving = reference.copy()
    moving[:, 2] *= -1
    moving += [-300, 40, 90]
    score, aligned = geometry.compute_mds_tm_score_alignment(reference, moving, 30)
    assert score == pytest.approx(1, abs=1e-12)
    np.testing.assert_allclose(aligned, reference, atol=1e-8)
    assert geometry.compute_tm_score(reference, aligned, 30) == pytest.approx(score, abs=1e-12)


def test_winning_coordinates_use_native_search_for_outliers(backend):
    reference = np.random.default_rng(9).normal(size=(30, 3)) + [110, -210, 70]
    moving = reference.copy()
    moving[-5:] += 15
    kabsch, _ = geometry.superimpose_structures(reference, moving)
    score, aligned = geometry.optimize_tm_score_alignment(reference, moving, 30)
    assert score > geometry.compute_tm_score(reference, kabsch, 30) + 0.5
    assert geometry.compute_tm_score(reference, aligned, 30) == pytest.approx(score, abs=1e-12)


def test_winning_coordinates_keep_kabsch_when_native_candidate_is_worse(monkeypatch):
    monkeypatch.setattr(geometry, "resolve_tmscore_binary", lambda: Path("/fake/TMscore"))
    def poor_native_search(command, **kwargs):
        Path(command[4]).write_text("0 100 1 0 0\n1 100 0 1 0\n2 100 0 0 1\n")
        return SimpleNamespace(stdout="Number of residues in common = 30\nTM-score = 0.1\n")
    monkeypatch.setattr(geometry.subprocess, "run", poor_native_search)
    reference = np.random.default_rng(3).normal(size=(30, 3)) + [110, -210, 70]
    score, aligned = geometry.optimize_tm_score_alignment(reference, reference + 100, 30)
    assert score == pytest.approx(1, abs=1e-12)
    np.testing.assert_allclose(aligned, reference, atol=1e-10)


def test_backend_matches_native_report_with_coordinate_rounding(backend, tmp_path):
    fixed = np.random.default_rng(42).normal(size=(40, 3)) * 4
    moving = fixed.copy()
    moving[-7:] += 12
    fixed -= fixed.mean(axis=0)
    moving -= moving.mean(axis=0)
    model, reference = tmp_path / "model.pdb", tmp_path / "reference.pdb"
    geometry._write_tmscore_trace(model, moving)
    geometry._write_tmscore_trace(reference, fixed)
    result = subprocess.run([str(backend), str(model), str(reference)],
                            check=True, capture_output=True, text=True)
    import re
    reported = float(re.search(r"TM-score\s*=\s*([\d.]+)", result.stdout).group(1))
    assert geometry.optimize_tm_score(fixed, moving, 40) == pytest.approx(reported, abs=0.002)


def test_missing_backend_and_subprocess_errors_are_explicit(monkeypatch, tmp_path):
    monkeypatch.setenv("TMSCORE_BINARY", str(tmp_path / "missing"))
    with pytest.raises(geometry.TMScoreError, match="unavailable"):
        geometry.compute_mds_tm_score(np.ones((3, 3)), np.ones((3, 3)), 3)
    monkeypatch.setattr(geometry, "resolve_tmscore_binary", lambda: Path("/fake/TMscore"))
    def fail(*args, **kwargs):
        raise subprocess.CalledProcessError(1, args[0], stderr="backend failure")
    monkeypatch.setattr(geometry.subprocess, "run", fail)
    with pytest.raises(geometry.TMScoreError, match="backend failure"):
        geometry.optimize_tm_score(np.ones((3, 3)), np.ones((3, 3)), 3)


def test_invalid_and_short_coordinates_are_rejected():
    with pytest.raises(ValueError, match="matching finite"):
        geometry.optimize_tm_score(np.full((30, 3), np.nan), np.ones((30, 3)), 30)
    with pytest.raises(ValueError, match="between 3"):
        geometry.optimize_tm_score(np.ones((2, 3)), np.ones((2, 3)), 2)


@pytest.fixture
def packed(tmp_path, monkeypatch, backend):
    export = worker.export
    root, originals, output = [tmp_path / name for name in ("predictions", "originals", "scores")]
    monkeypatch.setattr(worker, "LAYERS", (0, 1))
    monkeypatch.setattr(worker, "EXPECTED_PROTEINS", 2)
    proteins, residues, coordinates = [], [], {}
    for number, pid in enumerate(("000000022", "toy_B")):
        n = 30
        points = np.random.default_rng(number).normal(size=(n, 3)) * 5
        pdb = tmp_path / f"{pid}.pdb"
        lines = []
        for i, (x, y, z) in enumerate(points):
            chain, insertion = ("A" if i < 15 else "B"), ("A" if i == 12 else " ")
            lines.append(f"ATOM  {i+1:5d}  CA  ALA {chain}{i+1:4d}{insertion}   "
                         f"{x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C\n")
        pdb.write_text("".join(lines) + "END\n")
        checksum = export.sha256_file(pdb)
        protein = dict(protein_id=pid, source="ecod", outer_fold=number,
                       n_residues=n, n_pairs=n*(n-1)//2, pair_assignment_digest="frozen",
                       pdb_path=str(pdb), pdb_sha256=checksum)
        proteins.append(protein)
        residues.extend(dict(protein_id=pid, residue_index=i, chain_id="A" if i<15 else "B",
                             pdb_resseq=i+1, insertion_code="A" if i==12 else "", residue_name="ALA",
                             pdb_path=str(pdb), pdb_sha256=checksum) for i in range(n))
        coordinates[pid] = geometry.extract_ca_coordinates(pdb)["coordinates"].astype(np.float64)
    identity = dict(protein_ids=[p["protein_id"] for p in proteins], layers=list(worker.LAYERS))
    digest = export.semantic_digest(identity)
    geometry_rows = {spec.analysis_name: [] for spec in export.FAMILY_SPECS}
    layer_hashes = {}
    for layer in worker.LAYERS:
        records = {}
        for protein in proteins:
            pid, n = protein["protein_id"], protein["n_residues"]
            points = coordinates[pid].copy()
            if layer == 1:
                points[-3:] += 5
            i, j = np.tril_indices(n, -1)
            values = geometry.compute_distance_matrix(points)[i, j]
            values[0] = -0.5  # Storage remains raw; reconstruction clips it.
            records[pid] = {"families": {}}
            for spec in export.FAMILY_SPECS:
                source = {key: 0 for key in worker.GEOMETRY_METRICS_COLUMNS}
                source.update(analysis_name=spec.analysis_name, model_family=spec.model_family,
                              feature_dim=256, protein_id=pid, source="ecod", layer=layer,
                              outer_fold=protein["outer_fold"], cv_protocol_version="2.0",
                              evaluation_design=spec.evaluation_design, evaluation_scope="oof_all_pairs",
                              oof_prediction_scope=spec.oof_prediction_scope, geometry_status="complete",
                              geometry_exclusion_reason="", n_residues=n, pair_coverage=1.0,
                              n_protein_pairs=len(values), tm_score=0.1)
                geometry_rows[spec.analysis_name].append(source)
                index = {**{key: source[key] for key in export.IDENTITY_COLUMNS},
                         **export.atomic_save_prediction(root, layer, pid, spec, values)}
                records[pid]["families"][spec.analysis_name] = {"index": index}
        path = root / "layers" / f"layer_{layer:02d}" / "layer_manifest.json"
        export.atomic_write_json(path, dict(status="complete", layer=layer, identity_digest=digest,
                                           records=records, metadata_sha256={}))
        layer_hashes[str(layer)] = export.sha256_file(path)
    export.atomic_write_csv(root / "metadata/proteins.csv", export.PROTEIN_COLUMNS, proteins)
    export.atomic_write_csv(root / "metadata/residue_index.csv", export.RESIDUE_COLUMNS, residues)
    export.atomic_write_json(root / "run_manifest.json", dict(
        export_schema=export.EXPORT_SCHEMA, preparation_status="complete", run_status="complete",
        completed_layers=list(worker.LAYERS), identity=identity, identity_digest=digest,
        layer_manifest_sha256=layer_hashes,
        metadata_sha256={name: export.sha256_file(root / name) for name in (
            "metadata/proteins.csv", "metadata/residue_index.csv")},
    ))
    for spec in export.FAMILY_SPECS:
        export.atomic_write_csv(originals / spec.family_dir / "geometry_metrics.csv",
                                worker.GEOMETRY_METRICS_COLUMNS, geometry_rows[spec.analysis_name])
    context = worker.load_context(root, originals)
    return context, output


def test_packed_rescoring_and_finalization_preserve_inputs_and_csv_schema(packed):
    context, output = packed
    inputs = {path: worker.export.sha256_file(path) for directory in (context.root, Path(context.inputs["regression_root"]))
              for path in directory.rglob("*") if path.is_file()}
    for layer in worker.LAYERS:
        worker.run_layer(context, output, layer)
    worker.finalize(context, output)
    assert inputs == {path: worker.export.sha256_file(path) for path in inputs}
    rows = worker.read_rows(output / "tm_scores.csv", worker.SCORE_COLUMNS)
    assert len(rows) == 16
    pdb_rows = worker.read_rows(output / "pdb_index.csv", worker.PDB_COLUMNS)
    assert len(pdb_rows) == len(rows)
    assert len(list(output.glob("layers/*/pdb_files/*/*.pdb"))) == len(rows)
    assert worker.validate_pdb_rows(pdb_rows, rows, output)
    for row in pdb_rows:
        pid = row["protein_id"]
        path = output / row["pdb_path"]
        structure = geometry.PDBParser(QUIET=True).get_structure(pid, str(path))[0]
        residues = [residue for chain in structure for residue in chain]
        assert [(residue.parent.id, residue.id[1], residue.id[2].strip(), residue.resname)
                for residue in residues] == [
                    (r["chain_id"], int(r["pdb_resseq"]), r["insertion_code"], r["residue_name"])
                    for r in context.residues[pid]]
        assert all(len(residue) == 1 and "CA" in residue for residue in residues)
        coordinates = np.asarray([r["CA"].coord for r in residues], dtype=np.float64)
        assert np.all(np.isfinite(coordinates))
        assert geometry.compute_tm_score(worker.read_reference(context, pid), coordinates, len(residues)) == pytest.approx(
            float(row["tm_score"]), abs=0.002)
    assert {r["protein_id"] for r in rows} == {"000000022", "toy_B"}
    for spec in worker.export.FAMILY_SPECS:
        for row in worker.read_rows(output / spec.family_dir / "geometry_metrics.csv", worker.GEOMETRY_METRICS_COLUMNS):
            original = context.geometry[spec.analysis_name][int(row["layer"]), row["protein_id"]]
            assert {k: v for k, v in row.items() if k != "tm_score"} == {k: v for k, v in original.items() if k != "tm_score"}
        summary = worker.read_rows(output / spec.family_dir / "geometry_summary.csv", worker.GEOMETRY_SUMMARY_COLUMNS)
        assert len([r for r in summary if r["geometry_metric"] == "tm_score"]) == 2
    manifest = worker.export.load_json(output / "run_manifest.json")
    assert manifest["run_status"] == "complete"
    assert manifest["n_pdbs"] == len(rows)
    assert manifest["output_sha256"]["pdb_index.csv"] == worker.export.sha256_file(output / "pdb_index.csv")
    assert set(manifest["layer_manifest_sha256"]) == {"0", "1"}


def test_checkpoint_resume_skips_scoring_and_finalizer_rejects_missing_layers(packed, monkeypatch):
    context, output = packed
    worker.run_layer(context, output, 0)
    def forbidden(*args):
        raise AssertionError("Completed predictions must not be rescored")
    monkeypatch.setattr(worker, "score_distances", forbidden)
    worker.run_layer(context, output, 0)
    with pytest.raises(FileNotFoundError):
        worker.finalize(context, output)
    assert worker.export.load_json(output / "run_manifest.json")["run_status"] == "prepared"


@pytest.mark.parametrize("damage", ["missing", "corrupted"])
def test_damaged_pdb_rejects_finalization_and_regenerates_only_affected_protein(packed, monkeypatch, damage):
    context, output = packed
    for layer in worker.LAYERS:
        worker.run_layer(context, output, layer)
    worker.finalize(context, output)
    rows = worker.read_rows(output / "pdb_index.csv", worker.PDB_COLUMNS)
    target = output / rows[0]["pdb_path"]
    original = target.read_bytes()
    if damage == "missing":
        target.unlink()
    else:
        target.write_bytes(b"corrupted PDB\n")
    with pytest.raises(ValueError, match="reconstructed PDBs"):
        worker.finalize(context, output)
    assert worker.export.load_json(output / "run_manifest.json")["run_status"] == "prepared"
    actual, calls = worker.score_distances, []
    def counted(*args):
        calls.append(1)
        return actual(*args)
    monkeypatch.setattr(worker, "score_distances", counted)
    worker.run_layer(context, output, 0)
    assert len(calls) == 4
    assert target.read_bytes() == original
    worker.finalize(context, output)
    assert worker.export.load_json(output / "run_manifest.json")["n_pdbs"] == 16


def test_interrupted_pdb_repair_invalidates_completion_before_writing(packed, monkeypatch):
    context, output = packed
    for layer in worker.LAYERS:
        worker.run_layer(context, output, layer)
    worker.finalize(context, output)
    target = next(output.glob("layers/layer_00/pdb_files/*/*.pdb"))
    target.unlink()
    def interrupt(*args):
        assert worker.export.load_json(output / "run_manifest.json")["run_status"] == "prepared"
        assert not (output / "layers/layer_00/layer_manifest.json").exists()
        raise RuntimeError("interrupted repair")
    monkeypatch.setattr(worker, "write_reconstruction", interrupt)
    with pytest.raises(RuntimeError, match="interrupted repair"):
        worker.run_layer(context, output, 0)


def test_pdb_write_failure_does_not_checkpoint_partial_protein(packed, monkeypatch):
    context, output = packed
    actual, written = worker.write_reconstruction, []
    def interrupt(*args):
        if len(written) == 2:
            raise RuntimeError("interrupted PDB export")
        written.append(actual(*args))
        return written[-1]
    monkeypatch.setattr(worker, "write_reconstruction", interrupt)
    with pytest.raises(RuntimeError, match="interrupted PDB export"):
        worker.run_layer(context, output, 0)
    assert len(list(output.glob("layers/layer_00/pdb_files/*/*.pdb"))) == 2
    assert not list(output.glob("layers/layer_00/progress/*.json"))
    monkeypatch.setattr(worker, "write_reconstruction", actual)
    worker.run_layer(context, output, 0)
    assert len(list(output.glob("layers/layer_00/pdb_files/*/*.pdb"))) == 8


def test_preparation_contention_waits_and_duplicate_layer_guard_remains_exclusive(tmp_path):
    output = tmp_path / "scores"
    context = SimpleNamespace(root=tmp_path / "predictions", digest="frozen", proteins={"toy": {}},
                              inputs={"regression_root": str(tmp_path / "regressions")})
    ready = [Event(), Event()]
    def prepare(index):
        ready[index].set()
        worker.ensure_output(output, context)
    with ThreadPoolExecutor(max_workers=2) as executor:
        with worker.preparation_lock(output):
            futures = [executor.submit(prepare, index) for index in range(2)]
            assert all(event.wait(timeout=5) for event in ready)
            assert all(not future.done() for future in futures)
        for future in futures:
            future.result(timeout=5)
    assert worker.export.load_json(output / "run_manifest.json")["identity_digest"] == "frozen"
    with worker.export.task_lock(output, "layer_00"):
        with pytest.raises(RuntimeError, match="Another layer_00 task"):
            with worker.export.task_lock(output, "layer_00"):
                pytest.fail("Duplicate layer guard was bypassed")


def test_rescoring_rejects_changed_predictions_references_and_input_output_overlap(packed):
    context, output = packed
    with pytest.raises(ValueError, match="separate"):
        worker.run_layer(context, context.root, 0)
    pid = "000000022"
    pdb = Path(context.proteins[pid]["pdb_path"])
    original = pdb.read_bytes()
    pdb.write_bytes(original + b"REMARK changed\n")
    with pytest.raises(ValueError, match="checksum"):
        worker.run_layer(context, output, 0)
    pdb.write_bytes(original)
    path = context.root / worker.export.prediction_path(0, pid, "shared_ols_base256")
    values = np.load(path)
    values[1] += 1
    np.save(path, values)
    with pytest.raises(ValueError, match="checksum"):
        worker.run_layer(context, output, 0)


def test_interrupted_layer_retains_completed_proteins(packed, monkeypatch):
    context, output = packed
    actual = worker.score_distances
    calls = []
    def interrupt_after_one_protein(*args):
        if len(calls) == 4:
            raise RuntimeError("interrupted")
        calls.append(1)
        return actual(*args)
    monkeypatch.setattr(worker, "score_distances", interrupt_after_one_protein)
    with pytest.raises(RuntimeError, match="interrupted"):
        worker.run_layer(context, output, 0)
    assert (output / "layers/layer_00/progress/000000022.json").is_file()
    assert not (output / "layers/layer_00/layer_manifest.json").exists()
    calls.clear()
    worker.run_layer(context, output, 0)
    assert len(calls) == 4


def test_backend_failures_abort_geometry_rows(monkeypatch):
    import cross_validation_common as cv
    def fail(*args):
        raise geometry.TMScoreError("backend unavailable")
    monkeypatch.setattr(cv, "analyze_protein_geometry", fail)
    with pytest.raises(geometry.TMScoreError, match="backend unavailable"):
        cv.build_geometry_row(analysis_name="test", model_family="shared_ols", feature_dim=256,
                              layer=0, protein_id="toy", source="ecod", outer_fold=0,
                              evaluation_design="protein_outer_5fold", oof_prediction_scope="heldout_proteins_all_pairs",
                              n_residues=3, true_coordinates=np.ones((3,3)), predictions=np.ones(3))


def test_consolidation_matches_full_finalization_without_bulk_file_reads(packed, monkeypatch):
    context, output = packed
    for layer in worker.LAYERS:
        worker.run_layer(context, output, layer)
    worker.finalize(context, output)
    csvs = ["tm_scores.csv", "pdb_index.csv", "runtime.csv"] + [
        f"{spec.family_dir}/{name}" for spec in worker.export.FAMILY_SPECS
        for name in ("geometry_metrics.csv", "geometry_summary.csv")]
    expected = {name: (output / name).read_bytes() for name in csvs}
    original_manifest = worker.export.load_json(output / "run_manifest.json")
    layer_files = {path: worker.export.sha256_file(path) for path in (output / "layers").rglob("*") if path.is_file()}
    def forbidden(*args, **kwargs):
        raise AssertionError("Consolidation must not reread individual arrays or PDBs")
    actual_hash = worker.export.sha256_file
    def metadata_hash(path):
        if Path(path).suffix in {".npy", ".pdb"}:
            forbidden()
        return actual_hash(path)
    monkeypatch.setattr(worker.export, "sha256_file", metadata_hash)
    monkeypatch.setattr(worker.export, "check_array", forbidden)
    monkeypatch.setattr(worker, "read_reference", forbidden)
    monkeypatch.setattr(worker, "validate_pdb_rows", forbidden)
    consolidation.consolidate(context, output)
    assert expected == {name: (output / name).read_bytes() for name in csvs}
    assert layer_files == {path: actual_hash(path) for path in layer_files}
    manifest = worker.export.load_json(output / "run_manifest.json")
    assert manifest["identity"] == original_manifest["identity"]
    assert manifest["identity_digest"] == original_manifest["identity_digest"]
    assert manifest["n_scores"] == manifest["n_pdbs"] == 16
    assert manifest["run_status"] == "complete"
    assert manifest["validation"]["mode"] == "consolidation_only"
    assert manifest["validation"]["reconstructed_pdbs"] == "not_revalidated"
    assert manifest["consolidation"]["code_sha256"] == actual_hash(Path(consolidation.__file__))


def test_consolidation_rejects_missing_layers_and_changed_identity(packed):
    context, output = packed
    worker.run_layer(context, output, 0)
    with pytest.raises(FileNotFoundError):
        consolidation.consolidate(context, output)
    assert worker.export.load_json(output / "run_manifest.json")["run_status"] == "prepared"
    with pytest.raises(ValueError, match="inputs/backend/code changed"):
        consolidation.consolidate(SimpleNamespace(**{**vars(context), "digest": "changed"}), output)


@pytest.mark.parametrize("damage", ["csv_bytes", "pdb_index_identity", "prediction_hash", "pdb_index_duplicate"])
def test_consolidation_rejects_inconsistent_layer_tables(packed, damage):
    context, output = packed
    for layer in worker.LAYERS:
        worker.run_layer(context, output, layer)
    directory = output / "layers/layer_00"
    manifest = worker.export.load_json(directory / "layer_manifest.json")
    if damage == "csv_bytes":
        path = directory / "tm_scores.csv"
        path.write_bytes(path.read_bytes() + b"\n")
    elif damage == "prediction_hash":
        path = directory / "tm_scores.csv"
        rows = worker.read_rows(path, worker.SCORE_COLUMNS)
        rows[0]["array_sha256"] = "0" * 64
        worker.export.atomic_write_csv(path, worker.SCORE_COLUMNS, rows)
        manifest["tm_scores_sha256"] = worker.export.sha256_file(path)
        path = directory / "pdb_index.csv"
        rows = worker.read_rows(path, worker.PDB_COLUMNS)
        rows[0]["array_sha256"] = "0" * 64
        worker.export.atomic_write_csv(path, worker.PDB_COLUMNS, rows)
        manifest["pdb_index_sha256"] = worker.export.sha256_file(path)
    else:
        path = directory / "pdb_index.csv"
        rows = worker.read_rows(path, worker.PDB_COLUMNS)
        if damage == "pdb_index_identity":
            rows[0]["pdb_path"] = "wrong.pdb"
        else:
            rows[-1] = rows[0].copy()
        worker.export.atomic_write_csv(path, worker.PDB_COLUMNS, rows)
        manifest["pdb_index_sha256"] = worker.export.sha256_file(path)
    worker.export.atomic_write_json(directory / "layer_manifest.json", manifest)
    with pytest.raises(ValueError):
        consolidation.consolidate(context, output)
    assert worker.export.load_json(output / "run_manifest.json")["run_status"] == "prepared"

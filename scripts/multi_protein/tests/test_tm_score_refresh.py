from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import analyze_tm_score_optimization as analysis
import generate_sept_2_publication_plots as plots


def write_rows(path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\r\n")
        writer.writeheader()
        writer.writerows(rows)


@pytest.fixture
def geometry(tmp_path, monkeypatch):
    family = "shared_ols_base256"
    monkeypatch.setattr(analysis, "MODELS", {family: ("Shared OLS", "blue")})
    root, incoming = tmp_path / "results", tmp_path / "results/optimized"
    ledger, old, new = {}, [], []
    proteins = ["000000160", "1abc_A"]
    for layer in (0, 1):
        for i, pid in enumerate(proteins):
            row = dict(analysis_name=family, layer=str(layer), protein_id=pid,
                       geometry_status="complete", evaluation_scope="oof_all_pairs",
                       tm_score=str(.4 + .1 * i), triangle_violation_fraction="0.125", rmsd="2.0")
            old.append(row)
            updated = {**row, "tm_score": str(.6 + .1 * i)}
            new.append(updated)
            ledger[family, pid, layer] = {**updated, "tm_score_saved": row["tm_score"]}
    write_rows(root / family / "geometry_metrics.csv", old)
    write_rows(incoming / family / "geometry_metrics.csv", new)
    def summaries(rows):
        result = []
        for layer in (0, 1):
            for metric in ("rmsd", "tm_score"):
                values = np.array([float(r[metric]) for r in rows if r["layer"] == str(layer)])
                result.append(dict(analysis_name=family, layer=str(layer), geometry_metric=metric,
                                   n_proteins_included="2", mean=str(values.mean()), std=str(values.std()),
                                   median=str(np.median(values)), q25=str(np.quantile(values, .25)),
                                   q75=str(np.quantile(values, .75))))
        return result
    write_rows(root / family / "geometry_summary.csv", summaries(old))
    write_rows(incoming / family / "geometry_summary.csv", summaries(new))
    (root / family / "run_manifest.json").write_text(json.dumps({"code_revision": "original-training"}))
    source_hash = analysis.sha256(root / family / "geometry_metrics.csv")
    (incoming / family / "run_manifest.json").write_text(json.dumps(dict(
        run_status="complete", identity_digest="producer", source_geometry_sha256=source_hash)))
    manifest = dict(output_sha256={}, identity_digest="producer", tm_score_method="optimized",
                    validation={"mode": "consolidation_only"},
                    identity={"source_geometry_sha256": {family: source_hash}})
    (incoming / "run_manifest.json").write_text(json.dumps(manifest))
    def update_hashes():
        for name in ("geometry_metrics.csv", "geometry_summary.csv", "run_manifest.json"):
            manifest["output_sha256"][f"{family}/{name}"] = analysis.sha256(incoming / family / name)
    update_hashes()
    return root, incoming, ledger, manifest, proteins, family, update_hashes


def prepare(geometry):
    root, incoming, ledger, manifest, proteins, _, _ = geometry
    return analysis.prepare_geometry_refresh(root, incoming, ledger, manifest, proteins, (0, 1))


def test_refresh_preserves_other_cells_ids_and_is_repeatable(geometry):
    root, _, _, _, proteins, family, _ = geometry
    path = root / family / "geometry_metrics.csv"
    original_bytes = path.read_bytes()
    _, original = analysis.read_csv(path)
    checks, updates, _ = prepare(geometry)
    assert all(r["passed"] for r in checks)
    assert path.read_bytes() == original_bytes  # preparation is read-only
    analysis.apply_geometry_refresh(updates)
    _, refreshed = analysis.read_csv(path)
    assert [r["protein_id"] for r in refreshed] == proteins * 2
    assert [{k: v for k, v in r.items() if k != "tm_score"} for r in refreshed] == [
        {k: v for k, v in r.items() if k != "tm_score"} for r in original]
    assert all(float(r["tm_score"]) > float(o["tm_score"]) for r, o in zip(refreshed, original))
    assert b"\r\n" in path.read_bytes()
    manifest = json.loads(path.with_name("run_manifest.json").read_text())
    assert manifest["code_revision"] == "original-training"
    before = {p: (p.read_bytes(), p.stat().st_mtime_ns) for p in updates}
    _, repeated, _ = prepare(geometry)
    analysis.apply_geometry_refresh(repeated)
    assert all((p.read_bytes(), p.stat().st_mtime_ns) == value for p, value in before.items())


@pytest.mark.parametrize("problem", ["triangle", "duplicate", "protein_id", "score", "summary"])
def test_refresh_rejects_inconsistent_inputs_without_writing(geometry, problem):
    root, incoming, _, _, _, family, update_hashes = geometry
    filename = "geometry_summary.csv" if problem == "summary" else "geometry_metrics.csv"
    path = incoming / family / filename
    _, rows = analysis.read_csv(path)
    if problem == "triangle":
        rows[0]["triangle_violation_fraction"] = "0.13"
    elif problem == "duplicate":
        rows.append(rows[0].copy())
    elif problem == "protein_id":
        rows[0]["protein_id"] = "160"
    elif problem == "score":
        rows[0]["tm_score"] = "0.99"
    else:
        next(r for r in rows if r["geometry_metric"] == "tm_score")["mean"] = "0.99"
    write_rows(path, rows)
    update_hashes()
    before = {p: p.read_bytes() for p in (root / family).iterdir()}
    with pytest.raises(ValueError):
        prepare(geometry)
    assert all(p.read_bytes() == value for p, value in before.items())


def test_targeted_figures_keep_other_artifacts_and_manifest_rows(tmp_path, monkeypatch):
    root = tmp_path / "results"
    output = tmp_path / "plots"
    output.mkdir()
    monkeypatch.setattr(plots, "OUTPUT_ROOT", output)
    monkeypatch.setattr(plots, "REGRESSION_ROOT", root)
    monkeypatch.setattr(plots, "relative", lambda p: str(p.relative_to(tmp_path)))
    monkeypatch.setattr(plots, "ensure_directories", lambda: None)
    monkeypatch.setattr(plots, "load_cohort", lambda: ("001", "002"))
    monkeypatch.setattr(plots, "bootstrap_counts", lambda: None)
    monkeypatch.setattr(plots, "bootstrap_summary", lambda *a: pd.DataFrame({"mean": [.6]}))
    path = root / "family/geometry_metrics.csv"
    path.parent.mkdir(parents=True)
    path.write_text("canonical")
    scoring_path = root / "tm_score_optimized_v1/run_manifest.json"
    scoring_path.parent.mkdir()
    scoring_path.write_text(json.dumps(dict(run_status="complete", tm_score_method="optimized",
                                            validation={"mode": "consolidation_only"})))
    path.with_name("run_manifest.json").write_text(json.dumps({"tm_score_refresh": {
        "source_manifest_sha256": plots.sha256_file(scoring_path),
        "output_sha256": {"geometry_metrics.csv": plots.sha256_file(path)}}}))
    monkeypatch.setattr(plots, "load_core_metric", lambda *a: (pd.DataFrame({"value": [.6]}), [path]))
    ids = plots.expected_figure_ids()
    rows = [dict(figure_id=stem, pdf_sha256="old-pdf", png_sha256="old-png", title=stem) for stem in ids]
    pd.DataFrame(rows).to_csv(output / "figure_manifest.csv", index=False)
    generation = dict(generated_at="original-timestamp", inputs={}, source_data={})
    (output / "generation_summary.json").write_text(json.dumps(generation))
    unrelated = output / "triangle.pdf"
    unrelated.write_bytes(b"original triangle")
    def save_sources(frame, name):
        target = output / f"{name}.csv"
        frame.to_csv(target, index=False)
        return target
    monkeypatch.setattr(plots, "write_source_csv", save_sources)
    def render(stem):
        paths = (output / f"{stem}.png", output / f"{stem}.pdf")
        for target in paths:
            target.write_bytes(b"new plot")
        return paths
    monkeypatch.setattr(plots, "line_figure", lambda *a, **kw: render(a[2]))
    monkeypatch.setattr(plots, "individual_by_model_figure", lambda *a: render(a[4]))
    plots.refresh_tm_score_figures()
    changed = {"main_figure_03_tm_score", "supp_figure_individual_proteins_tm_score_by_model"}
    actual = pd.read_csv(output / "figure_manifest.csv").to_dict("records")
    assert [r for r in actual if r["figure_id"] not in changed] == [r for r in rows if r["figure_id"] not in changed]
    assert unrelated.read_bytes() == b"original triangle"
    metadata = json.loads((output / "generation_summary.json").read_text())
    assert metadata["generated_at"] == "original-timestamp"
    assert set(metadata["tm_score_refresh"]["figure_ids"]) == changed

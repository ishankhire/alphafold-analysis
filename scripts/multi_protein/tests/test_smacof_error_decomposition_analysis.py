"""Checks for the SMACOF error-decomposition analysis: the cohort maths and a full run on synthetic tables."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(PROJECT_ROOT / "scripts/multi_protein/smacof_error_decomposition"))

import analyze_smacof_error_decomposition as analysis  # noqa: E402

TRIANGLE_TABLE = analysis.TRIANGLE_TABLE
needs_local_tables = pytest.mark.skipif(not TRIANGLE_TABLE.is_file(), reason="local triangle tables unavailable")


def test_rms_is_root_of_protein_mean_square():
    squares = np.array([[1.0, 4.0], [9.0, 16.0], [25.0, 36.0]])
    counts = analysis.bootstrap_counts(n_proteins=3, replicates=500)
    s = analysis.rms_summary(squares, counts)
    assert np.allclose(s["mean"], np.sqrt(squares.mean(axis=0)))
    assert not np.allclose(s["mean"], np.sqrt(squares).mean(axis=0))      # not the mean of per-protein RMS
    assert np.all(s["bootstrap_ci_low"] <= s["mean"]) and np.all(s["mean"] <= s["bootstrap_ci_high"])
    assert np.allclose(s["bootstrap_ci_low"] ** 2, s["sq_ci_low"])


def test_mean_per_protein_rms_takes_the_root_first():
    squares = np.array([[1.0], [4.0], [10000.0]])                       # one protein with a huge error
    counts = analysis.bootstrap_counts(n_proteins=3, replicates=500)
    assert np.isclose(analysis.mean_rms_summary(squares, counts)["mean"][0], (1 + 2 + 100) / 3)
    assert np.isclose(analysis.rms_summary(squares, counts)["mean"][0], np.sqrt(10005 / 3))


def test_cross_share_is_ratio_of_cohort_means():
    cross = np.array([[1.0], [-2.0], [4.0]])
    total = np.array([[10.0], [1.0], [100.0]])
    counts = analysis.bootstrap_counts(n_proteins=3, replicates=500)
    s = analysis.ratio_summary(cross, total, counts)
    assert np.isclose(s["mean"][0], cross.mean() / total.mean())
    assert not np.isclose(s["mean"][0], (cross / total).mean())           # not the mean of per-protein ratios
    assert s["bootstrap_ci_low"][0] <= s["mean"][0] <= s["bootstrap_ci_high"][0]
    with pytest.raises(ValueError):
        analysis.ratio_summary(np.array([[np.nan], [1.0], [1.0]]), total, counts)


def synthetic_tables(tables: Path, proteins: list[str]) -> None:
    rng = np.random.default_rng(0)
    draws = []
    for probe_code, probe in enumerate(analysis.PROBES):
        for layer in analysis.LAYERS:
            for pid in proteins:
                for matrix_type in analysis.MATRIX_TYPES:
                    for draw in ([-1] if matrix_type == "observed" else [0, 1, 2]):
                        inconsistent = (4.0 + probe_code) * np.exp(-layer / 20) * rng.uniform(0.8, 1.2)
                        consistent = 2.0 * np.exp(-layer / 30) * rng.uniform(0.8, 1.2)
                        cross = 0.1 * rng.normal()
                        draws.append({
                            "probe": probe, "layer": layer, "protein_id": pid, "source": "ecod", "n_residues": 50,
                            "n_pairs": 1225, "matrix_type": matrix_type, "draw": draw, "n_negative_values": 0,
                            "total_sq": inconsistent + consistent + cross, "smacof_inconsistent_sq": inconsistent,
                            "smacof_consistent_sq": consistent, "smacof_cross": cross, "smacof_identity_residual": 0.0,
                            "smacof_raw_stress": 1225 * inconsistent, "classical_raw_stress": 1500 * inconsistent,
                            "smacof_scale_ratio": 1.0, "classical_scale_ratio": 0.99,
                            "smacof_iterations": 500, "smacof_converged": int(not (layer == 0 and draw == 2)),
                            "smacof_inconsistent_sq_at_1e-6": inconsistent * 1.001,
                            "smacof_consistent_sq_at_1e-6": consistent,
                            "smacof_inconsistent_sq_at_1e-8": inconsistent, "smacof_consistent_sq_at_1e-8": consistent,
                            "classical_inconsistent_sq": 1.2 * inconsistent, "classical_consistent_sq": 1.5 * consistent,
                            "classical_cross": cross - 0.2 * inconsistent - 0.5 * consistent,
                            "classical_identity_residual": 0.0,
                            "smacof_rmsd": 3.0, "classical_rmsd": 3.5,
                            "smacof_tm_score": 0.6 if matrix_type == "observed" else np.nan,
                        })
    draws = pd.DataFrame(draws)
    tables.mkdir(parents=True)
    draws.to_csv(tables / "decomposition_per_draw.csv.gz", index=False)
    keys = ["probe", "layer", "protein_id", "matrix_type"]
    terms = ["total_sq", "smacof_inconsistent_sq", "smacof_consistent_sq", "smacof_cross",
             "classical_inconsistent_sq", "classical_consistent_sq", "classical_cross", "smacof_rmsd",
             "classical_rmsd", "smacof_tm_score", "n_negative_values"]
    grouped = draws.groupby(keys, sort=False)
    per = grouped[terms].mean().reset_index()
    per["n_draws"] = grouped.size().to_numpy()
    per["n_unconverged"] = (per["n_draws"] - grouped["smacof_converged"].sum().to_numpy()).astype(int)
    for term in ("smacof_inconsistent_sq", "smacof_consistent_sq"):
        per[f"{term}_draw_sd"] = grouped[term].std().to_numpy()
    per.to_csv(tables / "decomposition_per_protein.csv.gz", index=False)

    bins = []
    for system, labels in (("sequence_separation", ("1", "2-4", "5-9")), ("true_distance_b10", ("[0,4)", "[4,6)"))):
        for probe in analysis.PROBES:
            for layer in analysis.LAYERS:
                for b, label in enumerate(labels):
                    bins.append({"probe": probe, "layer": layer, "matrix_type": "observed", "bin_system": system,
                                 "bin": b, "bin_label": label, "protein_mean_smacof_inconsistent_sq": 1.0 + b,
                                 "protein_mean_smacof_consistent_sq": 0.5 + b})
    pd.DataFrame(bins).to_csv(tables / "bin_terms_cohort.csv", index=False)
    convergence = draws.groupby(["probe", "layer", "matrix_type"]).agg(
        n_runs=("draw", "size"), n_converged=("smacof_converged", "sum"),
        iterations_median=("smacof_iterations", "median"), iterations_p95=("smacof_iterations", "max")).reset_index()
    convergence.to_csv(tables / "convergence_summary.csv", index=False)
    (tables / "run_manifest.json").write_text(json.dumps({
        "smacof": {"rel_tol": 1e-8, "max_iter": 20000},
        "checks": {"identity_and_stress_rows": len(draws), "failures": 0, "triangle_pairing_vectors": len(draws),
                   "histogram_checks": 10, "classical_parity_worst": {"parity_stress": 1e-8}},
    }))


@needs_local_tables
def test_full_analysis_on_synthetic_tables(tmp_path, monkeypatch):
    proteins = sorted(pd.read_csv(TRIANGLE_TABLE, dtype={"protein_id": str}, usecols=["protein_id"])
                      .protein_id.unique())[:12]
    synthetic_tables(tmp_path / "tables", proteins)
    monkeypatch.setattr(analysis, "ROOT", tmp_path)
    monkeypatch.setattr(analysis, "TABLES", tmp_path / "tables")
    monkeypatch.setattr(analysis, "FIGURES", tmp_path / "figures")
    monkeypatch.setattr(analysis, "SMOKE", True)
    analysis.main()

    summary = pd.read_csv(tmp_path / "summary_table.csv")
    assert len(summary) == 4 * 3 * 7 * 48
    row = summary[(summary.probe == analysis.PROBES[0]) & (summary.matrix_type == "observed")
                  & (summary.quantity == "smacof_inconsistent_sq") & (summary.layer == 10)].iloc[0]
    per = pd.read_csv(tmp_path / "tables/decomposition_per_protein.csv.gz", dtype={"protein_id": str})
    expected = per[(per.probe == analysis.PROBES[0]) & (per.matrix_type == "observed")
                   & (per.layer == 10)].smacof_inconsistent_sq.mean()
    assert np.isclose(row.protein_mean_sq, expected) and np.isclose(row.rms_angstrom, np.sqrt(expected))
    shares = pd.read_csv(tmp_path / "cross_share_by_layer.csv")
    assert len(shares) == 4 * 3 * 2 * 48
    for name in ("inconsistent_consistent_by_layer", "cross_share_by_layer", "inconsistent_vs_triangle_violation",
                 "consistent_vs_inconsistent_phase", "inconsistent_consistent_by_layer_mean_per_protein_rms",
                 "inconsistent_vs_triangle_violation_mean_per_protein_rms",
                 "consistent_vs_inconsistent_phase_mean_per_protein_rms", "error_share_stacked"):
        for suffix in ("pdf", "png"):
            assert (tmp_path / "figures/main" / f"{name}.{suffix}").stat().st_size > 0
    assert len(list((tmp_path / "figures/diagnostics").glob("*.png"))) == 7
    results = (tmp_path / "RESULTS.md").read_text()
    assert "Unconverged SMACOF runs" in results and "hit the iteration cap" in results
    assert "| 1e-6 | inconsistent |" in (tmp_path / "sanity_checks.md").read_text()

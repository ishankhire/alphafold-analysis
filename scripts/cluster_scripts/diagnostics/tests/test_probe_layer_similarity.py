from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.spatial.distance import cosine as scipy_cosine
from scipy.stats import pearsonr, spearmanr

SCRIPT_DIR = Path(__file__).resolve().parents[1] / "probe_layer_similarity"
for path in (SCRIPT_DIR, SCRIPT_DIR.parents[1]):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import make_probe_layer_similarity_figures as figures  # noqa: E402
import probe_layer_similarity as sim  # noqa: E402


def test_cosine_matrix_matches_scipy_and_rejects_zero_vectors():
    rng = np.random.default_rng(0)
    vectors = rng.normal(size=(6, 20))
    matrix = sim.cosine_matrix(vectors)
    for a in range(6):
        for b in range(6):
            assert matrix[a, b] == pytest.approx(1 - scipy_cosine(vectors[a], vectors[b]), abs=1e-12)
    vectors[2] = 0
    with pytest.raises(ValueError):
        sim.cosine_matrix(vectors)


def test_protein_similarity_matches_scipy_on_predictions_and_errors():
    rng = np.random.default_rng(1)
    target = rng.uniform(4, 40, size=300)
    # Ties are included on purpose: Spearman must use average ranks.
    predictions = np.round(target[None, :] + rng.normal(scale=[[1], [3], [9]], size=(3, 300)), 1)
    result = sim.protein_similarity(predictions.astype(np.float32), target)
    predictions = predictions.astype(np.float32).astype(np.float64)
    errors = predictions - target
    assert result.shape == (4, 3, 3)
    for a in range(3):
        for b in range(3):
            expected = (pearsonr(predictions[a], predictions[b])[0], spearmanr(predictions[a], predictions[b])[0],
                        pearsonr(errors[a], errors[b])[0], spearmanr(errors[a], errors[b])[0])
            assert result[:, a, b] == pytest.approx(expected, abs=1e-10)


def test_error_similarity_differs_from_prediction_similarity():
    rng = np.random.default_rng(2)
    target = rng.uniform(4, 40, size=2000)
    predictions = target[None, :] + rng.normal(size=(2, 2000))  # independent errors
    result = sim.protein_similarity(predictions, target)
    assert result[0, 0, 1] > 0.98 and abs(result[2, 0, 1]) < 0.1


def test_pearson_matrix_marks_constant_rows_as_nan():
    matrix = sim.pearson_matrix(np.array([[1.0, 2, 3], [5, 5, 5], [3, 2, 1]]))
    assert np.isnan(matrix[1]).all() and np.isnan(matrix[:, 1]).all()
    assert matrix[0, 2] == pytest.approx(-1)


def test_summarize_is_protein_equal_and_ignores_nan():
    n, mean, sd, median, q25, q75 = sim.summarize(np.array([0.2, 0.4, np.nan, 0.9]))
    assert (n, mean, median) == (3, pytest.approx(0.5), pytest.approx(0.4))
    assert sd == pytest.approx(np.std([0.2, 0.4, 0.9], ddof=1)) and q25 < median < q75
    assert sim.summarize(np.array([np.nan]))[0] == 0


def test_summary_rows_cover_the_upper_triangle_once():
    stack = np.random.default_rng(3).uniform(size=(4, sim.N_LAYERS, sim.N_LAYERS))
    rows = list(sim.summary_rows("p", "", "m", "protein", stack))
    assert len(rows) == 48 * 49 // 2 and len(rows[0]) == len(sim.SUMMARY_COLUMNS)
    row = next(r for r in rows if (r[4], r[5]) == (3, 10))
    assert row[6] == 7 and row[7] == 4 and row[8] == pytest.approx(stack[:, 3, 10].mean())


def test_fold_noise_floor_is_one_for_identical_folds():
    vector = np.random.default_rng(4).normal(size=(1, 3, 8))
    noise = sim.fold_noise_floor(np.repeat(vector, 5, axis=0))
    assert noise.shape == (3, 10) and np.allclose(noise, 1)


def test_final_structure_distances_use_packed_lower_triangle_order(tmp_path):
    coordinates = np.array([[0, 0, 0], [3, 0, 0], [3, 4, 0], [3, 4, 12]], dtype=float)
    lines = [f"ATOM  {i + 1:5d}  CA  ALA A{i + 1:4d}    {x:8.3f}{y:8.3f}{z:8.3f}  1.00  0.00           C"
             for i, (x, y, z) in enumerate(coordinates)]
    path = tmp_path / "toy.pdb"
    path.write_text("\n".join(lines) + "\nEND\n")
    packed = sim.final_structure_distances(path)
    # For residues i > j, the packed index is i*(i-1)//2 + j.
    assert packed == pytest.approx([3, 5, 4, 13, np.sqrt(160), 12])


def _write(path, columns, rows):
    with open(path, "w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(columns)
        writer.writerows(rows)


def test_figures_render_from_tables(tmp_path, monkeypatch):
    rng = np.random.default_rng(5)
    layers = np.arange(48)
    decay = np.exp(-np.abs(layers[:, None] - layers[None, :]) / 20)
    summary = []
    keys = [("shared_ols_base256", "", f"weight_cosine_{s}", "outer_fold") for s in sim.SPACES]
    keys += [("sequence_gated_ols_base256", g, f"weight_cosine_{s}", "outer_fold") for g in sim.GATES for s in sim.SPACES]
    keys += [(p, "", m, "protein") for p in sim.PROBES for m in sim.PAIR_MEASURES]
    for probe, gate, measure, unit in keys:
        stack = np.repeat(decay[None], 3, axis=0) * (2 * decay[None] - 1 if measure.startswith("error") else 1)
        summary.extend(sim.summary_rows(probe, gate, measure, unit, stack))
    _write(tmp_path / "layer_pair_similarity.csv", sim.SUMMARY_COLUMNS, summary)
    _write(tmp_path / "effectiveness_by_layer.csv", ("probe", "layer", "metric", "n", "mean", "sd", "median", "q25", "q75"),
           [(p, layer, metric, 9, rng.uniform(), 0, 0, 0, 0)
            for p in sim.PROBES for metric, *_ in sim.EFFECTIVENESS for layer in range(48)])
    (tmp_path / "transfer").mkdir()
    _write(tmp_path / "transfer" / "transfer_matrix.csv", ("source_layer", "target_layer", "mean_protein_r2"),
           [(a, b, decay[a, b] - 0.5) for a in range(48) for b in range(48)])
    (tmp_path / "run_manifest.json").write_text(json.dumps({"analysis_status": "complete", "n_proteins": 9}))
    monkeypatch.setenv("PROBE_LAYER_SIMILARITY_OUTPUT_ROOT", str(tmp_path))
    monkeypatch.setenv("ALL_LAYER_TRANSFER_OUTPUT_ROOT", str(tmp_path / "transfer"))
    monkeypatch.delenv("PROBE_LAYER_SIMILARITY_FIGURE_ROOT", raising=False)
    figures.main()
    assert len(list((tmp_path / "figures").glob("*.png"))) == 9
    assert len(list((tmp_path / "figures").glob("*.pdf"))) == 9

    matrices = figures.load_matrices(tmp_path / "layer_pair_similarity.csv")
    matrix = matrices[("shared_ols_base256", "", "prediction_pearson")]
    assert np.allclose(matrix, matrix.T) and np.allclose(matrix, decay)
    assert figures.offset_profile(matrix)[5] == pytest.approx(np.exp(-5 / 20))
    assert figures.color_scale([matrices[("shared_ols_base256", "", "error_pearson")]])[1] == -1.0

    (tmp_path / "run_manifest.json").write_text(json.dumps({"analysis_status": "parity_failed", "n_proteins": 9}))
    with pytest.raises(SystemExit):
        figures.main()

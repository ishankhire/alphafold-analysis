from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1]
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from plot_layerwise_ablation_tm_scores import (  # noqa: E402
    EXPECTED_EXCLUDED_KEYS,
    LAYER_COLUMNS,
    MODEL_ORDER,
    expected_overleaf_filenames,
    load_tm_score_matrix,
    validate_probe_summary,
)


def synthetic_cohort() -> pd.MultiIndex:
    keys = [("ecod", f"{index:09d}") for index in range(552)]
    return pd.MultiIndex.from_tuples(keys, names=["source", "protein_id"])


def write_ablation_csv(path: Path, *, replace_exclusion: bool = False) -> None:
    rows = []
    for source, protein_id in synthetic_cohort():
        row = {"dataset": source.upper(), "protein": protein_id, "experiment": "test"}
        row.update({column: 0.5 for column in LAYER_COLUMNS})
        rows.append(row)
    exclusions = sorted(EXPECTED_EXCLUDED_KEYS)
    if replace_exclusion:
        exclusions[-1] = ("ecod", "999999999")
    for source, protein_id in exclusions:
        row = {"dataset": source.upper(), "protein": protein_id, "experiment": "test"}
        row.update({column: 0.25 for column in LAYER_COLUMNS})
        rows.append(row)
    pd.DataFrame(rows).to_csv(path, index=False)


def test_ablation_loader_excludes_exact_two_proteins(tmp_path: Path) -> None:
    path = tmp_path / "ablation.csv"
    write_ablation_csv(path)
    cohort = synthetic_cohort()
    keys, matrix = load_tm_score_matrix(path, cohort)
    assert keys.equals(cohort)
    assert matrix.shape == (552, 48)
    assert np.all(matrix == 0.5)


def test_ablation_loader_rejects_an_unexpected_cross_file_cohort(tmp_path: Path) -> None:
    path = tmp_path / "ablation.csv"
    write_ablation_csv(path, replace_exclusion=True)
    with pytest.raises(ValueError, match="unexpected cohort exclusions"):
        load_tm_score_matrix(path, synthetic_cohort())


def synthetic_probe_rows() -> pd.DataFrame:
    rows = []
    for model_index, analysis in enumerate(MODEL_ORDER):
        for layer in range(48):
            rows.append(
                {
                    "analysis_name": analysis,
                    "layer": layer,
                    "mean": model_index + layer + 1,
                    "n_proteins_total": 552,
                    "n_proteins_finite": 552,
                    "aggregation": "protein_equal_mean",
                }
            )
    return pd.DataFrame(rows)


def test_probe_summary_validates_four_models_and_protein_equal_means() -> None:
    means = validate_probe_summary(synthetic_probe_rows())
    assert set(means) == set(MODEL_ORDER)
    assert all(values.shape == (48,) for values in means.values())
    assert means[MODEL_ORDER[0]][0] == pytest.approx(1.0)
    assert means[MODEL_ORDER[2]][47] == pytest.approx(50.0)


def test_probe_summary_rejects_incomplete_model_coverage() -> None:
    rows = synthetic_probe_rows()
    rows = rows[~rows["analysis_name"].eq(MODEL_ORDER[-1])]
    with pytest.raises(ValueError, match="all four models"):
        validate_probe_summary(rows)


def test_overleaf_inventory_has_six_base_and_twelve_overlay_pdfs() -> None:
    names = expected_overleaf_filenames()
    assert len(names) == 18
    assert sum(name.endswith("_tm_score_trajectory.pdf") for name in names) == 6
    assert sum(name.endswith("_with_r2.pdf") for name in names) == 6
    assert sum(
        name.endswith("_with_normalized_triangle_violation.pdf") for name in names
    ) == 6

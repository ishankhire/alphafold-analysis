from pathlib import Path

import numpy as np

from lib.classification_common import (
    build_pair_type_labels,
    extract_top_confused_pairs,
    write_confusion_matrix_csv,
    write_per_class_report_csv,
)


def test_build_pair_type_labels_is_order_invariant():
    residue_data = {"residues": [("A", "GLY", 1), ("A", "ALA", 2)]}
    labels = build_pair_type_labels(residue_data, np.array([0]), np.array([1]))
    assert labels.tolist() == ["ALA-GLY"]


def test_extract_top_confused_pairs():
    cm = np.array([[3, 2], [1, 4]])
    rows = extract_top_confused_pairs(cm, ["A", "B"], top_n=2)
    assert rows[0]["true_class"] == "A"
    assert rows[0]["predicted_class"] == "B"
    assert rows[0]["count"] == 2


def test_csv_writers(tmp_path: Path):
    cm = np.array([[1, 0], [2, 3]])
    cm_path = tmp_path / "cm.csv"
    write_confusion_matrix_csv(cm_path, cm, ["A", "B"])
    assert cm_path.exists()

    report_path = tmp_path / "report.csv"
    report = {
        "precision": np.array([0.5, 0.7]),
        "recall": np.array([0.6, 0.8]),
        "f1": np.array([0.55, 0.75]),
        "support": np.array([10, 20]),
    }
    write_per_class_report_csv(report_path, ["A", "B"], report)
    assert report_path.exists()

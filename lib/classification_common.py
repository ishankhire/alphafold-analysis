"""
Shared helpers for classification-style regression scripts.
"""

import csv
import numpy as np


def build_pair_type_labels(residue_data, struct_i, struct_j):
    """Build sorted amino-acid pair labels for given pair indices."""
    res_names = [r[1] for r in residue_data["residues"]]
    return np.array(
        [
            "-".join(sorted([res_names[i], res_names[j]]))
            for i, j in zip(struct_i, struct_j)
        ]
    )


def extract_top_confused_pairs(cm, class_names, top_n=20):
    """Return the most frequent off-diagonal confusion entries."""
    cm_off = cm.copy()
    np.fill_diagonal(cm_off, 0)
    flat_indices = np.argsort(cm_off.ravel())[::-1][:top_n]
    rows, cols = np.unravel_index(flat_indices, cm.shape)

    results = []
    for r, c in zip(rows, cols):
        if cm_off[r, c] == 0:
            break
        results.append(
            {
                "true_class": class_names[r],
                "predicted_class": class_names[c],
                "count": int(cm_off[r, c]),
                "true_total": int(cm[r].sum()),
            }
        )
    return results


def write_confusion_matrix_csv(path, cm, class_names):
    """Write confusion matrix CSV with class labels in row/column headers."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([""] + list(class_names))
        for i, row in enumerate(cm):
            writer.writerow([class_names[i]] + [int(x) for x in row])


def write_per_class_report_csv(path, class_names, report):
    """Write per-class precision/recall/f1/support CSV."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["pair_type", "f1", "precision", "recall", "support"])
        for i, class_name in enumerate(class_names):
            sup = int(report["support"][i])
            if sup == 0:
                continue
            writer.writerow(
                [
                    class_name,
                    f"{report['f1'][i]:.4f}",
                    f"{report['precision'][i]:.4f}",
                    f"{report['recall'][i]:.4f}",
                    sup,
                ]
            )

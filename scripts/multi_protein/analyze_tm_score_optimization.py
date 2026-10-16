#!/usr/bin/env python3
"""Compare saved and optimized TM-scores over the complete frozen OOF cohort."""
from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

INPUT = ROOT / "delta_files/regression_5fold_v1/tm_score_optimized_v1"
OUTPUT = ROOT / "delta_files/regression_5fold_v1/analysis/tm_score_optimization_comparison_v1"
MODELS = {
    "shared_ols_base256": ("Shared OLS", "#377eb8"),
    "per_protein_ols_base256": ("Per-protein OLS", "#e68613"),
    "sequence_gated_ols_base256": ("Sequence-gated OLS", "#41956b"),
    "shared_xgboost_base256_fixed_s4": ("Shared XGBoost", "#9b61a8"),
}
TOLERANCE = 1e-6


def write_csv(path, rows):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def statistics(old, new):
    gain = new - old
    return {
        "tm_score_saved_mean": float(old.mean()),
        "tm_score_optimized_mean": float(new.mean()),
        "mean_gain": float(gain.mean()),
        "median_gain": float(np.median(gain)),
        "q90_gain": float(np.quantile(gain, 0.9)),
        "fraction_improved": float(np.mean(gain > TOLERANCE)),
        "fraction_decreased": float(np.mean(gain < -TOLERANCE)),
    }


def main():
    manifest = json.loads((INPUT / "run_manifest.json").read_text())
    checksum = hashlib.sha256((INPUT / "tm_scores.csv").read_bytes()).hexdigest()
    if manifest["run_status"] != "complete" or checksum != manifest["output_sha256"]["tm_scores.csv"]:
        raise ValueError("Score ledger is incomplete or differs from its completion manifest")
    with (INPUT / "tm_scores.csv").open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    proteins = sorted({row["protein_id"] for row in rows})
    expected = {(model, pid, layer) for model in MODELS for pid in proteins for layer in range(48)}
    indexed = {(row["analysis_name"], row["protein_id"], int(row["layer"])): row for row in rows}
    if len(proteins) != 552 or len(rows) != 105984 or len(indexed) != len(rows) or set(indexed) != expected:
        raise ValueError("Expected 552 proteins, four probes, and all 48 layers without duplicate rows")
    if any(row["status"] != "complete" or row["evaluation_scope"] != "oof_all_pairs" for row in rows):
        raise ValueError("Incomplete scores or inconsistent evaluation scope")

    model_rows, layer_rows = [], []
    figure, axes = plt.subplots(2, 2, figsize=(10, 7), sharex=True, sharey=True)
    for axis, (model, (label, color)) in zip(axes.flat, MODELS.items()):
        data = [[indexed[model, pid, layer] for layer in range(48)] for pid in proteins]
        old = np.array([[float(row["tm_score_saved"]) for row in protein] for protein in data])
        new = np.array([[float(row["tm_score"]) for row in protein] for protein in data])
        baseline = np.array([[float(row["tm_score_kabsch"]) for row in protein] for protein in data])
        if (not np.all(np.isfinite([old, new, baseline])) or np.any(new < baseline - 1e-12)
                or np.any(new < 0) or np.any(new > 1) or np.any(old < 0) or np.any(old > 1)):
            raise ValueError(f"Invalid score range or alignment floor: {model}")
        summary = dict(analysis_name=model, n_proteins=552, n_layers=48, n_scores=old.size,
                       **statistics(old, new),
                       layer_0_mean_gain=float((new[:, 0] - old[:, 0]).mean()),
                       layer_47_saved_mean=float(old[:, 47].mean()),
                       layer_47_optimized_mean=float(new[:, 47].mean()),
                       layer_47_mean_gain=float((new[:, 47] - old[:, 47]).mean()),
                       mean_kabsch_saved_difference=float((baseline - old).mean()),
                       max_abs_kabsch_saved_difference=float(np.max(np.abs(baseline - old))))
        model_rows.append(summary)
        for layer in range(48):
            layer_rows.append(dict(analysis_name=model, layer=layer, n_proteins=552,
                                   **statistics(old[:, layer], new[:, layer])))
        axis.plot(range(48), old.mean(axis=0), color="#777777", linestyle="--", label="Saved (Kabsch)")
        axis.plot(range(48), new.mean(axis=0), color=color, linewidth=2, label="Optimized TM-score")
        axis.set_title(label)
        axis.set_xlim(0, 47)
        axis.set_ylim(0.35, 1)
        axis.set_xticks([0, 10, 20, 30, 40, 47])
        axis.grid(alpha=0.18)
        axis.legend(frameon=False, fontsize=8, loc="lower right")
    figure.supxlabel("Evoformer layer")
    figure.supylabel("Mean TM-score across 552 proteins")
    figure.suptitle("TM-score optimization: same predicted distances, improved superposition")
    figure.tight_layout(rect=(0.02, 0.02, 1, 0.97))
    OUTPUT.mkdir(parents=True, exist_ok=True)
    write_csv(OUTPUT / "model_summary.csv", model_rows)
    write_csv(OUTPUT / "layer_summary.csv", layer_rows)
    figure.savefig(OUTPUT / "tm_score_comparison.png", dpi=180)
    figure.savefig(OUTPUT / "tm_score_comparison.pdf")
    plt.close(figure)
    summary = {
        "source_ledger": str((INPUT / "tm_scores.csv").relative_to(ROOT)),
        "source_sha256": checksum,
        "source_validation": manifest.get("validation"),
        "aggregation": "equal proteins within each layer; equal layers for the overall descriptive mean",
        "improvement_tolerance": TOLERANCE,
        "notes": ["Same regression predictions; the alignment objective changed.",
                  "All-layer means describe trajectories, not a designated primary layer.",
                  "No confidence intervals or hypothesis tests; improvement is favored by the scoring procedure."],
        "models": model_rows,
    }
    (OUTPUT / "comparison_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

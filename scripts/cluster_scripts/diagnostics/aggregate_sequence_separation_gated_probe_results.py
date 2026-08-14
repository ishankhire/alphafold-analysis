#!/usr/bin/env python3
"""Consolidate sequence-separation-gated probe outputs across all gates.

The source per-protein files contain one row per separation bin.  This script
reconstructs whole-protein metrics from additive error and target moments, then
summarizes those whole-protein rows by layer and split.  In particular, it does
not average R-squared values across sequence-separation bins.
"""

import csv
import math
from datetime import datetime
from pathlib import Path

import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[2]
RESULTS_DIR = (
    PROJECT_ROOT
    / "delta_files"
    / "diagnostics"
    / "diagnostic_sequence_separation_gated_probes"
)
LAYER_BATCHES_DIR = RESULTS_DIR / "layer_batches"
FLOAT_EPS = 1e-12

PER_PROTEIN_FIELDS = [
    "analysis_name",
    "model",
    "feature_dim",
    "layer",
    "split",
    "protein_id",
    "source",
    "n_residues",
    "size_bin",
    "n_sequence_separation_bins",
    "n_pairs",
    "target_variance",
    "per_protein_r2",
    "per_protein_mse",
    "per_protein_mae",
    "per_protein_rmse",
    "mean_true_distance",
    "mean_predicted_distance",
    "mean_error_pred_minus_true",
]

LAYER_SUMMARY_FIELDS = [
    "analysis_name",
    "model",
    "feature_dim",
    "layer",
    "split",
    "n_proteins",
    "n_pairs",
    "mean_per_protein_r2",
    "median_per_protein_r2",
    "std_per_protein_r2",
    "pooled_r2",
    "mean_per_protein_mae",
    "median_per_protein_mae",
    "pooled_mae",
    "mean_per_protein_rmse",
    "median_per_protein_rmse",
    "pooled_rmse",
    "mean_per_protein_bias",
    "pair_weighted_bias",
    "mean_true_distance",
    "mean_predicted_distance",
]


def parse_float(value):
    parsed = float(value)
    if not math.isfinite(parsed):
        raise ValueError(f"Expected a finite float, got {value!r}")
    return parsed


def format_value(value):
    if isinstance(value, (float, np.floating)):
        if not math.isfinite(float(value)):
            return "nan"
        return f"{float(value):.6f}"
    return value


def write_csv(path, fieldnames, rows):
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {name: format_value(row.get(name, "")) for name in fieldnames}
            )
    temporary.replace(path)


def read_source_rows():
    paths = sorted(LAYER_BATCHES_DIR.glob("per_protein_layer_*.csv"))
    if not paths:
        raise FileNotFoundError(f"No per-protein layer files found in {LAYER_BATCHES_DIR}")

    rows = []
    for path in paths:
        with path.open(newline="") as handle:
            rows.extend(csv.DictReader(handle))
    return paths, rows


def reconstruct_per_protein_rows(source_rows):
    grouped = {}
    for row in source_rows:
        key = (
            row["analysis_name"],
            row["model"],
            int(row["feature_dim"]),
            int(row["layer"]),
            row["split"],
            row["protein_id"],
        )
        grouped.setdefault(key, []).append(row)

    output_rows = []
    additive = {}
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        analysis_name, model, feature_dim, layer, split, protein_id = key
        first = rows[0]
        n_pairs = np.asarray([int(row["n_pairs"]) for row in rows], dtype=np.float64)
        total_pairs = float(np.sum(n_pairs))
        if total_pairs <= 0:
            raise ValueError(f"No pair rows for {model}, layer {layer}, {protein_id}")

        means_true = np.asarray(
            [parse_float(row["mean_true_distance"]) for row in rows], dtype=np.float64
        )
        means_pred = np.asarray(
            [parse_float(row["mean_predicted_distance"]) for row in rows],
            dtype=np.float64,
        )
        variances = np.asarray(
            [parse_float(row["target_variance"]) for row in rows], dtype=np.float64
        )
        mse = np.asarray(
            [parse_float(row["per_protein_mse"]) for row in rows], dtype=np.float64
        )
        mae = np.asarray(
            [parse_float(row["per_protein_mae"]) for row in rows], dtype=np.float64
        )

        sum_y = float(np.sum(n_pairs * means_true))
        sum_y2 = float(np.sum(n_pairs * (variances + means_true * means_true)))
        sum_pred = float(np.sum(n_pairs * means_pred))
        sum_squared_error = float(np.sum(n_pairs * mse))
        sum_absolute_error = float(np.sum(n_pairs * mae))
        mean_true = sum_y / total_pairs
        mean_pred = sum_pred / total_pairs
        ss_total = sum_y2 - sum_y * sum_y / total_pairs
        aggregate_mse = sum_squared_error / total_pairs
        aggregate_r2 = (
            1.0 - sum_squared_error / ss_total
            if ss_total > FLOAT_EPS
            else float("nan")
        )

        output_rows.append(
            {
                "analysis_name": analysis_name,
                "model": model,
                "feature_dim": feature_dim,
                "layer": layer,
                "split": split,
                "protein_id": protein_id,
                "source": first["source"],
                "n_residues": int(first["n_residues"]),
                "size_bin": int(first["size_bin"]),
                "n_sequence_separation_bins": len(rows),
                "n_pairs": int(total_pairs),
                "target_variance": ss_total / total_pairs,
                "per_protein_r2": aggregate_r2,
                "per_protein_mse": aggregate_mse,
                "per_protein_mae": sum_absolute_error / total_pairs,
                "per_protein_rmse": math.sqrt(aggregate_mse),
                "mean_true_distance": mean_true,
                "mean_predicted_distance": mean_pred,
                "mean_error_pred_minus_true": mean_pred - mean_true,
            }
        )
        additive[key] = {
            "n_pairs": total_pairs,
            "sum_y": sum_y,
            "sum_y2": sum_y2,
            "sum_pred": sum_pred,
            "sum_squared_error": sum_squared_error,
            "sum_absolute_error": sum_absolute_error,
        }

    return output_rows, additive


def summarize_layers(per_protein_rows, additive):
    grouped = {}
    for row in per_protein_rows:
        key = (
            row["analysis_name"],
            row["model"],
            int(row["feature_dim"]),
            int(row["layer"]),
            row["split"],
        )
        grouped.setdefault(key, []).append(row)

    output_rows = []
    for key, rows in sorted(grouped.items(), key=lambda item: item[0]):
        analysis_name, model, feature_dim, layer, split = key
        r2_values = np.asarray([row["per_protein_r2"] for row in rows], dtype=np.float64)
        r2_values = r2_values[np.isfinite(r2_values)]
        mae_values = np.asarray([row["per_protein_mae"] for row in rows], dtype=np.float64)
        rmse_values = np.asarray([row["per_protein_rmse"] for row in rows], dtype=np.float64)
        bias_values = np.asarray(
            [row["mean_error_pred_minus_true"] for row in rows], dtype=np.float64
        )

        moments = [
            additive[
                (
                    row["analysis_name"],
                    row["model"],
                    int(row["feature_dim"]),
                    int(row["layer"]),
                    row["split"],
                    row["protein_id"],
                )
            ]
            for row in rows
        ]
        total_pairs = float(sum(row["n_pairs"] for row in moments))
        sum_y = float(sum(row["sum_y"] for row in moments))
        sum_y2 = float(sum(row["sum_y2"] for row in moments))
        sum_pred = float(sum(row["sum_pred"] for row in moments))
        sum_squared_error = float(sum(row["sum_squared_error"] for row in moments))
        sum_absolute_error = float(sum(row["sum_absolute_error"] for row in moments))
        ss_total = sum_y2 - sum_y * sum_y / total_pairs

        output_rows.append(
            {
                "analysis_name": analysis_name,
                "model": model,
                "feature_dim": feature_dim,
                "layer": layer,
                "split": split,
                "n_proteins": len(rows),
                "n_pairs": int(total_pairs),
                "mean_per_protein_r2": float(np.mean(r2_values)),
                "median_per_protein_r2": float(np.median(r2_values)),
                "std_per_protein_r2": float(np.std(r2_values)),
                "pooled_r2": (
                    1.0 - sum_squared_error / ss_total
                    if ss_total > FLOAT_EPS
                    else float("nan")
                ),
                "mean_per_protein_mae": float(np.mean(mae_values)),
                "median_per_protein_mae": float(np.median(mae_values)),
                "pooled_mae": sum_absolute_error / total_pairs,
                "mean_per_protein_rmse": float(np.mean(rmse_values)),
                "median_per_protein_rmse": float(np.median(rmse_values)),
                "pooled_rmse": math.sqrt(sum_squared_error / total_pairs),
                "mean_per_protein_bias": float(np.mean(bias_values)),
                "pair_weighted_bias": (sum_pred - sum_y) / total_pairs,
                "mean_true_distance": sum_y / total_pairs,
                "mean_predicted_distance": sum_pred / total_pairs,
            }
        )
    return output_rows


def markdown_table(rows):
    lines = [
        "| Model | Layer | Mean protein R² | Median protein R² | Pooled R² | "
        "Mean protein MAE (Å) | Pooled MAE (Å) | Pooled RMSE (Å) |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for row in rows:
        lines.append(
            f"| {row['model']} | {row['layer']} | "
            f"{row['mean_per_protein_r2']:.4f} | "
            f"{row['median_per_protein_r2']:.4f} | "
            f"{row['pooled_r2']:.4f} | "
            f"{row['mean_per_protein_mae']:.3f} | "
            f"{row['pooled_mae']:.3f} | "
            f"{row['pooled_rmse']:.3f} |"
        )
    return "\n".join(lines)


def write_markdown_summary(paths, layer_rows):
    test_rows = sorted(
        [row for row in layer_rows if row["split"] == "test"],
        key=lambda row: (row["model"], row["layer"]),
    )
    content = "\n".join(
        [
            "# Sequence-separation-gated probe aggregate summary",
            "",
            f"Generated: {datetime.now().astimezone().isoformat(timespec='seconds')}",
            "",
            "## Held-out metrics",
            "",
            markdown_table(test_rows),
            "",
            "## Calculation",
            "",
            "The six sequence-separation bins are recombined within each protein by "
            "summing pair counts, squared errors, absolute errors, and target moments. "
            "Whole-protein R² is then computed from the combined residual and total "
            "sums of squares. It is not an average of bin-level R² values.",
            "",
            "`mean_per_protein_*` gives every protein equal weight. `pooled_*` gives "
            "every residue pair equal weight across the split.",
            "",
            "The source per-bin CSV values are rounded to six decimal places, so the "
            "reconstructed metrics can differ slightly from metrics computed from raw "
            "predictions.",
            "",
            f"Source files: {len(paths)} `per_protein_layer_XX.csv` files.",
            "",
        ]
    )
    (RESULTS_DIR / "aggregate_summary.md").write_text(content)


def main():
    paths, source_rows = read_source_rows()
    per_protein_rows, additive = reconstruct_per_protein_rows(source_rows)
    layer_rows = summarize_layers(per_protein_rows, additive)

    write_csv(
        RESULTS_DIR / "aggregate_per_protein.csv",
        PER_PROTEIN_FIELDS,
        per_protein_rows,
    )
    write_csv(
        RESULTS_DIR / "aggregate_layer_summary.csv",
        LAYER_SUMMARY_FIELDS,
        layer_rows,
    )
    write_markdown_summary(paths, layer_rows)

    print(f"Wrote {len(per_protein_rows)} per-protein aggregate rows")
    print(f"Wrote {len(layer_rows)} layer/split aggregate rows")
    print(f"Output directory: {RESULTS_DIR}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""
analyze_r2_geometry_correlation.py
===================================
Two analyses of whether regression R2 correlates with geometric quality
metrics from MDS reconstruction (TM-score, RMSD, eigenvalue concentration,
triangle inequality violations).

Analysis 1 — Layer-level means (48 points):
    Each layer's mean R2 vs mean metric across all proteins/folds.

Analysis 2 — Per-protein×layer (n_proteins × 48 points):
    Join kfold_mds_metrics.csv with kfold_regression_r2.csv on (fold, layer)
    so each protein×layer is one point. Includes partial correlations
    controlling for layer index to remove the depth-trend confound.

Outputs:
    csv_files/r2_geometry_correlations.csv           (Analysis 1)
    csv_files/r2_geometry_correlations_per_pt.csv    (Analysis 2 raw + partial)
    visualizations/r2_geometry_scatter.png           (Analysis 1)
    visualizations/r2_geometry_scatter_per_pt.png    (Analysis 2)
"""

import csv
from pathlib import Path

import numpy as np
from scipy import stats
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).parent.parent.parent.parent
DATA_DIR = PROJECT_ROOT / "proteins" / "multi_protein_results" / "kfold_mds" / "csv_files"
VIS_DIR = PROJECT_ROOT / "proteins" / "multi_protein_results" / "kfold_mds" / "visualizations"

METRICS = {
    "tm_score":                    ("TM-score",                "higher = better fold"),
    "rmsd":                        ("RMSD",                    "lower = better"),
    "frac_top3_eigenvalues":       ("Frac top-3 eigenvalues",  "higher = more 3D-like"),
    "neg_eigenvalue_fraction":     ("Neg eigenvalue fraction", "lower = more metric"),
    "stress":                      ("Stress",                  "lower = better MDS fit"),
    "triangle_violation_fraction": ("Triangle violation frac", "lower = more metric"),
    "triangle_mean_violation":     ("Triangle mean violation", "lower = better"),
}


def load_csv(path):
    """Load CSV into a dict of {column_name: np.array}."""
    with open(path) as f:
        rows = list(csv.DictReader(f))
    if not rows:
        return {}
    columns = {}
    for key in rows[0]:
        try:
            columns[key] = np.array([float(r[key]) for r in rows])
        except ValueError:
            columns[key] = np.array([r[key] for r in rows])
    return columns


def partial_correlation(x, y, covariate):
    """Pearson r between x and y after linearly partialling out covariate."""
    def resid(a, b):
        slope, intercept, _, _, _ = stats.linregress(b, a)
        return a - (slope * b + intercept)
    return stats.pearsonr(resid(x, covariate), resid(y, covariate))


# ---------------------------------------------------------------------------
# Analysis 1: layer-level means (48 points)
# ---------------------------------------------------------------------------

def analysis1():
    print(f"{'='*60}")
    print("ANALYSIS 1: Layer-level mean data points (48 points)")
    print(f"{'='*60}")

    r2_data = load_csv(DATA_DIR / "kfold_regression_r2_summary.csv")
    mds_data = load_csv(DATA_DIR / "kfold_mds_summary.csv")

    r2 = r2_data["r2_test_mean"]
    layers = r2_data["layer"]

    # Map metric col names in summary CSV (appended with _mean)
    summary_cols = {col: (f"{col}_mean", note) for col, (_, note) in METRICS.items()}

    print(f"{'Metric':<30s} {'Pearson r':>10s} {'p-value':>10s} {'Spearman r':>11s} {'p-value':>10s}")
    print("-" * 75)

    corr_rows = []
    for col, (summary_col, note) in summary_cols.items():
        label = METRICS[col][0]
        vals = mds_data[summary_col]
        mask = ~(np.isnan(r2) | np.isnan(vals))
        r2c, vc = r2[mask], vals[mask]

        pearson_r, pearson_p = stats.pearsonr(r2c, vc)
        spearman_r, spearman_p = stats.spearmanr(r2c, vc)

        print(f"{label:<30s} {pearson_r:>10.4f} {pearson_p:>10.2e} {spearman_r:>11.4f} {spearman_p:>10.2e}")
        corr_rows.append({
            "metric": label, "note": note,
            "pearson_r": pearson_r, "pearson_p": pearson_p,
            "spearman_r": spearman_r, "spearman_p": spearman_p,
            "n_layers": int(mask.sum()),
        })

    out_csv = DATA_DIR / "r2_geometry_correlations.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        writer.writeheader()
        for row in corr_rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})
    print(f"\nSaved -> {out_csv}")

    fig, axes = plt.subplots(3, 3, figsize=(15, 13))
    axes = axes.flatten()
    sc = None
    for idx, (col, (summary_col, note)) in enumerate(summary_cols.items()):
        label = METRICS[col][0]
        ax = axes[idx]
        vals = mds_data[summary_col]
        mask = ~(np.isnan(r2) | np.isnan(vals))
        sc = ax.scatter(r2[mask], vals[mask], c=layers[mask], cmap="viridis",
                        s=25, alpha=0.8, edgecolors="k", linewidths=0.3)
        slope, intercept, rval, pval, _ = stats.linregress(r2[mask], vals[mask])
        x_fit = np.linspace(r2[mask].min(), r2[mask].max(), 50)
        ax.plot(x_fit, slope * x_fit + intercept, "r--", linewidth=1.5,
                label=f"r={rval:.3f}, p={pval:.1e}")
        ax.set_xlabel("R² (test, mean across folds)")
        ax.set_ylabel(label)
        ax.set_title(f"R² vs {label}")
        ax.legend(fontsize=7)
        ax.grid(True, alpha=0.3)

    for idx in range(len(summary_cols), len(axes)):
        axes[idx].set_visible(False)

    fig.colorbar(sc, ax=axes.tolist(), shrink=0.5, label="Layer")
    fig.suptitle("Regression R² vs Geometric Quality Metrics (per-layer means)", fontsize=14, y=1.01)
    fig.tight_layout()
    out_png = VIS_DIR / "r2_geometry_scatter.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved scatter -> {out_png}")


# ---------------------------------------------------------------------------
# Analysis 2: per-protein×layer (n_proteins × 48 points)
# ---------------------------------------------------------------------------

def analysis2():
    print(f"\n{'='*60}")
    print("ANALYSIS 2: Per-protein×layer data points")
    print(f"{'='*60}")
    print("  Each protein×layer is one point; R² is the fold-level test R²\n"
          "  for the fold in which that protein was in the test set.\n"
          "  Partial correlation removes the linear layer-depth trend.\n")

    mds_rows = []
    with open(DATA_DIR / "kfold_mds_metrics.csv") as f:
        for row in csv.DictReader(f):
            mds_rows.append(row)

    r2_lookup = {}  # (fold, layer) -> r2_test
    with open(DATA_DIR / "kfold_regression_r2.csv") as f:
        for row in csv.DictReader(f):
            r2_lookup[(int(row["fold"]), int(row["layer"]))] = float(row["r2_test"])

    r2_vals, layer_vals = [], []
    metric_vals = {col: [] for col in METRICS}

    for row in mds_rows:
        key = (int(row["fold"]), int(row["layer"]))
        if key not in r2_lookup:
            continue
        try:
            r2_vals.append(r2_lookup[key])
            layer_vals.append(int(row["layer"]))
            for col in METRICS:
                metric_vals[col].append(float(row[col]))
        except (ValueError, KeyError):
            pass

    r2_arr = np.array(r2_vals)
    layer_arr = np.array(layer_vals, dtype=float)
    for col in METRICS:
        metric_vals[col] = np.array(metric_vals[col])

    print(f"  N data points: {len(r2_arr)}\n")
    header = (f"{'Metric':<30s} {'Raw r':>8s} {'p':>9s}  "
              f"{'Partial r':>9s} {'p':>9s}  (partial = controlling for layer)")
    print(header)
    print("-" * len(header))

    corr_rows = []
    for col, (label, note) in METRICS.items():
        vals = metric_vals[col]
        mask = ~(np.isnan(r2_arr) | np.isnan(vals))
        r2c, vc, lc = r2_arr[mask], vals[mask], layer_arr[mask]

        raw_r, raw_p = stats.pearsonr(r2c, vc)
        part_r, part_p = partial_correlation(r2c, vc, lc)

        print(f"  {label:<28s}  {raw_r:>8.4f} {raw_p:>9.2e}   {part_r:>9.4f} {part_p:>9.2e}")
        corr_rows.append({
            "metric": label, "note": note,
            "raw_pearson_r": raw_r, "raw_pearson_p": raw_p,
            "partial_pearson_r": part_r, "partial_pearson_p": part_p,
            "n_points": int(mask.sum()),
        })

    out_csv = DATA_DIR / "r2_geometry_correlations_per_pt.csv"
    with open(out_csv, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(corr_rows[0].keys()))
        writer.writeheader()
        for row in corr_rows:
            writer.writerow({k: (f"{v:.6f}" if isinstance(v, float) else v)
                             for k, v in row.items()})
    print(f"\n  Saved -> {out_csv}")

    # Scatter: raw (top row) and partial-residual (bottom row)
    n = len(METRICS)
    fig, axes = plt.subplots(2, n, figsize=(4 * n, 8))

    slope_r2, int_r2, _, _, _ = stats.linregress(layer_arr, r2_arr)
    r2_resid = r2_arr - (slope_r2 * layer_arr + int_r2)

    for idx, (col, (label, _)) in enumerate(METRICS.items()):
        vals = metric_vals[col]
        mask = ~(np.isnan(r2_arr) | np.isnan(vals))

        # Top: raw
        ax = axes[0, idx]
        ax.scatter(r2_arr[mask], vals[mask], c=layer_arr[mask], cmap="viridis",
                   s=10, alpha=0.5)
        raw_r, raw_p = stats.pearsonr(r2_arr[mask], vals[mask])
        ax.set_title(f"{label}\nr={raw_r:.3f}, p={raw_p:.1e}", fontsize=8)
        ax.set_xlabel("R² (fold-level)", fontsize=7)
        if idx == 0:
            ax.set_ylabel("Metric value (raw)", fontsize=8)
        ax.grid(True, alpha=0.3)

        # Bottom: partial residuals
        ax2 = axes[1, idx]
        slope_v, int_v, _, _, _ = stats.linregress(layer_arr[mask], vals[mask])
        v_resid = vals[mask] - (slope_v * layer_arr[mask] + int_v)
        ax2.scatter(r2_resid[mask], v_resid, c=layer_arr[mask], cmap="viridis",
                    s=10, alpha=0.5)
        part_r, part_p = partial_correlation(r2_arr[mask], vals[mask], layer_arr[mask])
        ax2.set_title(f"Layer trend removed\nr={part_r:.3f}, p={part_p:.1e}", fontsize=8)
        ax2.set_xlabel("R² residual", fontsize=7)
        if idx == 0:
            ax2.set_ylabel("Metric residual\n(layer partialled out)", fontsize=8)
        ax2.grid(True, alpha=0.3)

    fig.suptitle("Per-protein×layer: R² vs Geometric Metrics\n"
                 "Top: raw  |  Bottom: layer-depth trend removed",
                 fontsize=11)
    fig.tight_layout()
    out_png = VIS_DIR / "r2_geometry_scatter_per_pt.png"
    fig.savefig(out_png, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved scatter -> {out_png}")

    print(f"\n  KEY FINDINGS (Analysis 2)")
    for row in corr_rows:
        raw_sig = "sig" if row["raw_pearson_p"] < 0.05 else "n.s."
        part_sig = "sig" if row["partial_pearson_p"] < 0.05 else "n.s."
        print(f"  {row['metric']:<28s}  raw r={row['raw_pearson_r']:+.3f} ({raw_sig})  "
              f"partial r={row['partial_pearson_r']:+.3f} ({part_sig})")


def last_layer_scatter():
    """R² vs TM-score for each protein at layer 47."""
    LAST_LAYER = 47

    mds_rows = []
    with open(DATA_DIR / "kfold_mds_metrics.csv") as f:
        for row in csv.DictReader(f):
            if int(row["layer"]) == LAST_LAYER:
                mds_rows.append(row)

    r2_lookup = {}  # (fold, layer) -> r2_test
    with open(DATA_DIR / "kfold_regression_r2.csv") as f:
        for row in csv.DictReader(f):
            if int(row["layer"]) == LAST_LAYER:
                r2_lookup[int(row["fold"])] = float(row["r2_test"])

    proteins, r2_vals, tm_vals = [], [], []
    for row in mds_rows:
        fold = int(row["fold"])
        if fold not in r2_lookup:
            continue
        try:
            proteins.append(row["protein_id"])
            r2_vals.append(r2_lookup[fold])
            tm_vals.append(float(row["tm_score"]))
        except (ValueError, KeyError):
            pass

    r2_arr = np.array(r2_vals)
    tm_arr = np.array(tm_vals)

    fig, ax = plt.subplots(figsize=(7, 5))
    for pid, r2, tm in zip(proteins, r2_arr, tm_arr):
        ax.scatter(r2, tm, s=80, zorder=3)
        ax.annotate(pid, (r2, tm), textcoords="offset points", xytext=(5, 3), fontsize=8)

    ax.set_xlabel("R² (fold-level test, layer 47)")
    ax.set_ylabel("TM-score (layer 47)")
    ax.set_title("R² vs TM-score at Layer 47 (per protein)")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, alpha=0.6, label="TM=0.5")
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    out_png = VIS_DIR / "r2_vs_tmscore_layer47.png"
    fig.savefig(out_png, dpi=150)
    plt.close(fig)
    print(f"\nSaved layer-47 scatter -> {out_png}")
    print("  Proteins plotted:")
    for pid, r2, tm in zip(proteins, r2_arr, tm_arr):
        print(f"    {pid}: R²={r2:.4f}  TM={tm:.4f}")


def main():
    analysis1()
    analysis2()
    last_layer_scatter()


if __name__ == "__main__":
    main()

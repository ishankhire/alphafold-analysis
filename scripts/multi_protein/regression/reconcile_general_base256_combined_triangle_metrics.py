#!/usr/bin/env python3
"""Reconcile general_base256_combined geometry CSVs with the exact triangle metrics.

The original geometry CSVs carry triangle columns produced by the old
min-indirect-path definition. The authoritative exact per-triangle metrics now
live in `exact_triangle_metrics/exact_triangle_metrics.csv`. This script rewrites,
in place:

- `geometry_metrics.csv` (per-protein): drops the 3 old triangle columns and
  appends the 7 exact triangle fields, joined on the per-protein key. All other
  columns (including `n_negative_predictions`) are left untouched.
- `geometry_summary.csv` (per layer x split): drops the 6 old triangle summary
  columns and appends mean/std (population std, ddof=0) of the 7 exact fields,
  recomputed from the exact per-protein metrics. Non-triangle summary columns
  are preserved exactly.

Fixed configuration; no CLI arguments.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[3]
VARIANT = "general_base256_combined"
COMBINED_DIR = PROJECT_ROOT / "delta_files" / VARIANT

GEOM_METRICS = COMBINED_DIR / "geometry_metrics.csv"
GEOM_SUMMARY = COMBINED_DIR / "geometry_summary.csv"
EXACT_METRICS = COMBINED_DIR / "exact_triangle_metrics" / "exact_triangle_metrics.csv"

OLD_TRIANGLE_COLS = [
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
]
# Canonical exact-triangle field order (matches TRIANGLE_VIOLATION_KEYS).
NEW_TRIANGLE_COLS = [
    "triangle_violation_fraction",
    "triangle_mean_violation",
    "triangle_max_violation",
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]
OLD_SUMMARY_TRIANGLE_COLS = [
    f"{col}_{stat}" for col in OLD_TRIANGLE_COLS for stat in ("mean", "std")
]
NEW_SUMMARY_TRIANGLE_COLS = [
    f"{col}_{stat}" for col in NEW_TRIANGLE_COLS for stat in ("mean", "std")
]

PROTEIN_KEYS = ["variant", "feature_dim", "split_id", "layer", "split", "protein_id"]
SUMMARY_KEYS = ["variant", "feature_dim", "split_id", "layer", "split"]

CATEGORY_COLS = [
    "triangle_no_violation_fraction",
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]
BIN_COLS = [
    "triangle_violation_0_5_fraction",
    "triangle_violation_5_10_fraction",
    "triangle_violation_10_plus_fraction",
]
TOL = 5e-5


def atomic_write(df: pd.DataFrame, path: Path) -> None:
    tmp = path.with_name(f".{path.name}.tmp")
    df.to_csv(tmp, index=False, float_format="%.6f")
    tmp.replace(path)


def validate_exact(exact: pd.DataFrame) -> None:
    missing = sorted(set(PROTEIN_KEYS + NEW_TRIANGLE_COLS) - set(exact.columns))
    if missing:
        raise ValueError(f"{EXACT_METRICS} missing columns: {missing}")
    if exact.duplicated(PROTEIN_KEYS).any():
        raise ValueError("exact_triangle_metrics has duplicate per-protein keys")

    props = NEW_TRIANGLE_COLS[3:] + ["triangle_violation_fraction"]
    for col in props:
        values = exact[col].to_numpy(dtype=np.float64)
        if ((values < -TOL) | (values > 1.0 + TOL)).any():
            raise ValueError(f"{col} has values outside [0, 1]")

    cat_err = float(np.max(np.abs(exact[CATEGORY_COLS].to_numpy(np.float64).sum(axis=1) - 1.0)))
    if cat_err > TOL:
        raise ValueError(f"Category proportions do not sum to 1 (max err {cat_err})")

    bin_err = float(
        np.max(
            np.abs(
                exact[BIN_COLS].to_numpy(np.float64).sum(axis=1)
                - exact["triangle_violation_fraction"].to_numpy(np.float64)
            )
        )
    )
    if bin_err > TOL:
        raise ValueError(f"Bin proportions do not match violation fraction (max err {bin_err})")


def reconcile_metrics(exact: pd.DataFrame) -> pd.DataFrame:
    geom = pd.read_csv(GEOM_METRICS)
    missing = sorted(set(PROTEIN_KEYS + OLD_TRIANGLE_COLS) - set(geom.columns))
    if missing:
        raise ValueError(f"{GEOM_METRICS} missing expected columns: {missing}")

    triangle_block = exact[PROTEIN_KEYS + NEW_TRIANGLE_COLS]
    base = geom.drop(columns=OLD_TRIANGLE_COLS)
    merged = base.merge(triangle_block, on=PROTEIN_KEYS, how="left", validate="one_to_one")

    if len(merged) != len(geom):
        raise ValueError(f"Row count changed during merge: {len(geom)} -> {len(merged)}")
    if merged[NEW_TRIANGLE_COLS].isna().any().any():
        raise ValueError("Some geometry rows had no matching exact-triangle row")

    return merged


def reconcile_summary(exact: pd.DataFrame) -> pd.DataFrame:
    summary = pd.read_csv(GEOM_SUMMARY)
    missing = sorted(set(SUMMARY_KEYS + ["n_proteins"] + OLD_SUMMARY_TRIANGLE_COLS) - set(summary.columns))
    if missing:
        raise ValueError(f"{GEOM_SUMMARY} missing expected columns: {missing}")

    grouped = exact.groupby(SUMMARY_KEYS, sort=False)
    means = grouped[NEW_TRIANGLE_COLS].mean()
    stds = grouped[NEW_TRIANGLE_COLS].std(ddof=0)
    sizes = grouped.size().rename("n_proteins_exact")

    stats = pd.DataFrame(index=means.index)
    for col in NEW_TRIANGLE_COLS:
        stats[f"{col}_mean"] = means[col]
        stats[f"{col}_std"] = stds[col]
    stats = stats.join(sizes).reset_index()

    base = summary.drop(columns=OLD_SUMMARY_TRIANGLE_COLS)
    merged = base.merge(stats, on=SUMMARY_KEYS, how="left", validate="one_to_one")

    if len(merged) != len(summary):
        raise ValueError(f"Summary row count changed: {len(summary)} -> {len(merged)}")
    if merged[NEW_SUMMARY_TRIANGLE_COLS].isna().any().any():
        raise ValueError("Some summary rows had no matching exact-triangle group")
    if not (merged["n_proteins"].to_numpy() == merged["n_proteins_exact"].to_numpy()).all():
        raise ValueError("n_proteins disagrees between geometry_summary and exact metrics")

    return merged.drop(columns=["n_proteins_exact"])


def main() -> None:
    exact = pd.read_csv(EXACT_METRICS)
    validate_exact(exact)

    metrics = reconcile_metrics(exact)
    summary = reconcile_summary(exact)

    atomic_write(metrics, GEOM_METRICS)
    atomic_write(summary, GEOM_SUMMARY)

    print(f"Rewrote {GEOM_METRICS.relative_to(PROJECT_ROOT)}")
    print(f"  rows={len(metrics)} cols={len(metrics.columns)}")
    print(f"  triangle columns now: {NEW_TRIANGLE_COLS}")
    print(f"Rewrote {GEOM_SUMMARY.relative_to(PROJECT_ROOT)}")
    print(f"  rows={len(summary)} cols={len(summary.columns)}")
    layer_means = metrics.groupby("layer")["triangle_violation_fraction"].mean()
    print(
        "  exact violation_fraction layer-mean range: "
        f"[{layer_means.min():.4f}, {layer_means.max():.4f}]"
    )


if __name__ == "__main__":
    main()

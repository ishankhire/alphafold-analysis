#!/usr/bin/env python3
"""Plot saved split counts without fitting or replaying any models.

Writes xgboost_split_dimensions.{png,pdf}, dimension_summary.csv (one row per
input feature: index/name, mean splits over 48 layers and five folds, final-layer
mean, and corresponding fractions), and generation_summary.json under
delta_files/regression_5fold_v1/analysis/xgboost_split_dimensions/.
Counts describe training-tree structure, not held-out performance or gain.
"""

from __future__ import annotations

import csv
import hashlib
import json
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT / ".cache"))

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt

FAMILY = "shared_xgboost_base256_fixed_s4"
INPUT = ROOT / "delta_files/regression_5fold_v1" / FAMILY / "model_artifacts/feature_importance.csv"
OUTPUT = ROOT / "delta_files/regression_5fold_v1/analysis/xgboost_split_dimensions"


def main() -> None:
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    counts = np.full((48, 256), np.nan)
    names = [f"pair_{direction}_ch_{channel:03d}" for direction in ("ij", "ji") for channel in range(128)]
    for row in rows:
        layer, feature = int(row["layer"]), int(row["feature_index"])
        if not (0 <= layer < 48 and 0 <= feature < 256):
            raise ValueError("Unexpected layer or feature index")
        if (row["analysis_name"] != FAMILY or int(row["n_outer_folds"]) != 5
                or int(row["feature_dim"]) != 256 or row["feature_name"] != names[feature]):
            raise ValueError("Unexpected family, fold count, or feature mapping")
        if not np.isnan(counts[layer, feature]):
            raise ValueError("Duplicate layer/feature row")
        counts[layer, feature] = float(row["split_count_mean"])
    if not np.all(np.isfinite(counts)) or np.any(counts < 0) or np.any(counts.sum(axis=1) <= 0):
        raise ValueError("Incomplete or invalid split counts")
    if not np.allclose(counts * 5, np.rint(counts * 5), rtol=0, atol=1e-7):
        raise ValueError("Fold means do not reconstruct integer split totals")

    mean, final = counts.mean(axis=0), counts[-1]
    fractions = counts / counts.sum(axis=1, keepdims=True)
    OUTPUT.mkdir(parents=True, exist_ok=True)
    with (OUTPUT / "dimension_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["feature_index", "feature_name", "mean_splits_per_model_all_layers",
                         "all_layers_split_fraction", "layer47_mean_splits_per_model",
                         "layer47_split_fraction"])
        writer.writerows((i, names[i], mean[i], mean[i] / mean.sum(), final[i], fractions[-1, i])
                         for i in range(256))

    fig, axes = plt.subplots(3, 1, figsize=(13, 10), constrained_layout=True,
                             gridspec_kw={"height_ratios": [1, 1, 1.6]})
    colors = ["#3977AC"] * 128 + ["#DD8540"] * 128
    for ax, values, title in zip(axes[:2], (mean, final),
                               ("All layers: mean over 240 fitted models (48 layers × 5 folds)",
                                "Final layer 47: mean over five fitted models")):
        ax.bar(np.arange(256), values, width=1, color=colors)
        ax.set_title(title, loc="left", fontsize=11)
        ax.set_ylabel("Mean split count")
        ax.spines[["top", "right"]].set_visible(False)
    heat = axes[2].imshow(100 * fractions, origin="lower", aspect="auto", cmap="magma",
                          extent=(-0.5, 255.5, -0.5, 47.5), interpolation="nearest")
    axes[2].set_title("Split frequency by layer (each row normalized to 100%)", loc="left", fontsize=11)
    axes[2].set_ylabel("Layer")
    fig.colorbar(heat, ax=axes[2], label="Splits on feature (% of layer total)", pad=0.015)
    for ax in axes:
        ax.axvline(127.5, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xlim(-0.5, 255.5)
        ax.set_xticks([0, 32, 64, 96, 127, 160, 192, 224, 255])
    axes[-1].set_xlabel("Input feature index (0–127: Zᵢⱼ channels 0–127; 128–255: Zⱼᵢ channels 0–127)")
    fig.suptitle("Which dimensions does XGBoost split on?", fontsize=16)
    for suffix in ("png", "pdf"):
        fig.savefig(OUTPUT / f"xgboost_split_dimensions.{suffix}", dpi=180)
    plt.close(fig)
    summary = {
        "source": str(INPUT.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "n_layers": 48, "n_folds": 5, "n_features": 256,
        "count_definition": "Number of tree split nodes using a feature; XGBoost weight importance.",
        "all_layers_aggregation": "Arithmetic mean split count over all 240 fitted models.",
        "heatmap_aggregation": "Fold-mean counts divided by total fold-mean counts within each layer.",
        "total_splits_all_models": int(round(counts.sum() * 5)),
        "top_features_all_layers": [
            {"feature_index": int(i), "feature_name": names[i], "mean_splits_per_model": float(mean[i])}
            for i in np.argsort(-mean)[:10]
        ],
        "layer47_top_features": [int(i) for i in np.argsort(-final)[:10]],
        "top10_split_fraction_all_layers": float(np.sort(mean)[-10:].sum() / mean.sum()),
    }
    (OUTPUT / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

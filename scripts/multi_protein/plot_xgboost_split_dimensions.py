#!/usr/bin/env python3
"""Plot saved feature gain (default) or split counts without fitting models.

Writes xgboost_gain_dimensions.{png,pdf}, dimension_summary.csv (one row per
input feature: index/name, mean gain over 48 layers and five folds, final-layer
mean, and normalized gain fractions), and generation_summary.json under
delta_files/regression_5fold_v1/analysis/xgboost_gain_dimensions/.
Use --importance split_count to reproduce the original split-count outputs in
analysis/xgboost_split_dimensions/. Gain is average improvement per split using
the feature, with unused features zero-filled; it is not total_gain.
"""

from __future__ import annotations

import csv
import argparse
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
ANALYSIS_ROOT = ROOT / "delta_files/regression_5fold_v1/analysis"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--importance", choices=("gain", "split_count"), default="gain")
    metric = parser.parse_args().importance
    is_gain = metric == "gain"
    stem = "xgboost_gain_dimensions" if is_gain else "xgboost_split_dimensions"
    output = ANALYSIS_ROOT / stem
    with INPUT.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    importance = np.full((48, 256), np.nan)
    names = [f"pair_{direction}_ch_{channel:03d}" for direction in ("ij", "ji") for channel in range(128)]
    for row in rows:
        layer, feature = int(row["layer"]), int(row["feature_index"])
        if not (0 <= layer < 48 and 0 <= feature < 256):
            raise ValueError("Unexpected layer or feature index")
        if (row["analysis_name"] != FAMILY or int(row["n_outer_folds"]) != 5
                or int(row["feature_dim"]) != 256 or row["feature_name"] != names[feature]):
            raise ValueError("Unexpected family, fold count, or feature mapping")
        if not np.isnan(importance[layer, feature]):
            raise ValueError("Duplicate layer/feature row")
        importance[layer, feature] = float(row[f"{metric}_mean"])
    if (not np.all(np.isfinite(importance)) or np.any(importance < 0)
            or np.any(importance.sum(axis=1) <= 0)):
        raise ValueError(f"Incomplete or invalid {metric} values")
    if not is_gain and not np.allclose(importance * 5, np.rint(importance * 5), rtol=0, atol=1e-7):
        raise ValueError("Fold means do not reconstruct integer split totals")

    mean, final = importance.mean(axis=0), importance[-1]
    fractions = importance / importance.sum(axis=1, keepdims=True)
    output.mkdir(parents=True, exist_ok=True)
    value_name = "gain" if is_gain else "splits"
    fraction_name = "normalized_gain" if is_gain else "split"
    with (output / "dimension_summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["feature_index", "feature_name", f"mean_{value_name}_per_model_all_layers",
                         f"all_layers_{fraction_name}_fraction", f"layer47_mean_{value_name}_per_model",
                         f"layer47_{fraction_name}_fraction"])
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
        ax.set_ylabel("Mean gain" if is_gain else "Mean split count")
        ax.spines[["top", "right"]].set_visible(False)
    heat = axes[2].imshow(100 * fractions, origin="lower", aspect="auto", cmap="magma",
                          extent=(-0.5, 255.5, -0.5, 47.5), interpolation="nearest")
    heatmap_label = "Feature gain" if is_gain else "Split frequency"
    axes[2].set_title(f"{heatmap_label} by layer (each row normalized to 100%)", loc="left", fontsize=11)
    axes[2].set_ylabel("Layer")
    fig.colorbar(heat, ax=axes[2], label="Normalized gain (%)" if is_gain else "Splits on feature (% of layer total)", pad=0.015)
    for ax in axes:
        ax.axvline(127.5, color="#888888", linewidth=0.8, linestyle="--")
        ax.set_xlim(-0.5, 255.5)
        ax.set_xticks([0, 32, 64, 96, 127, 160, 192, 224, 255])
    axes[-1].set_xlabel("Input feature index (0–127: Zᵢⱼ channels 0–127; 128–255: Zⱼᵢ channels 0–127)")
    fig.suptitle("XGBoost gain by input dimension" if is_gain else "Which dimensions does XGBoost split on?", fontsize=16)
    for suffix in ("png", "pdf"):
        fig.savefig(output / f"{stem}.{suffix}", dpi=180)
    plt.close(fig)
    summary = {
        "source": str(INPUT.relative_to(ROOT)),
        "source_sha256": hashlib.sha256(INPUT.read_bytes()).hexdigest(),
        "n_layers": 48, "n_folds": 5, "n_features": 256,
        "importance_type": metric,
        "importance_definition": (
            "Average training-objective improvement per split using a feature; XGBoost gain, not total_gain. "
            "Unused features have gain zero."
            if is_gain else "Number of tree split nodes using a feature; XGBoost weight importance."
        ),
        "all_layers_aggregation": f"Arithmetic mean {metric} over all 240 fitted models.",
        "heatmap_aggregation": f"Fold-mean {metric} divided by sum over features within each layer.",
        "top_features_all_layers": [
            {"feature_index": int(i), "feature_name": names[i], f"mean_{value_name}_per_model": float(mean[i])}
            for i in np.argsort(-mean)[:10]
        ],
        "layer47_top_features": [int(i) for i in np.argsort(-final)[:10]],
        f"top10_{fraction_name}_fraction_all_layers": float(np.sort(mean)[-10:].sum() / mean.sum()),
    }
    if not is_gain:
        summary["total_splits_all_models"] = int(round(importance.sum() * 5))
    (output / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()

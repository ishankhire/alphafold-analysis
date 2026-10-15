#!/usr/bin/env python3
"""Compare saved shared-OLS coefficient vectors across all 48 layers.

Outputs under delta_files/regression_5fold_v1/analysis/linear_coefficients_analysis:
shared_ols_layer_cosine_{heatmap.png,heatmap.pdf,matrix.csv} (standardized),
shared_ols_raw_scale_layer_cosine_{heatmap.png,heatmap.pdf,matrix.csv},
fold_similarities.csv (fold/layer_a/layer_b plus both cosine similarities),
source_artifacts.csv (one row per fold/layer artifact with checksum/status),
generation_summary.json, and README.md. Each matrix averages within-fold
cosines over five folds; intercepts are excluded. No fitting or evaluation runs.
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

FAMILY = ROOT / "delta_files/regression_5fold_v1/shared_ols_base256"
OUTPUT = ROOT / "delta_files/regression_5fold_v1/analysis/linear_coefficients_analysis"
SPACES = ("standardized", "raw_scale")


def write_csv(path: Path, columns, rows) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(columns)
        writer.writerows(rows)


def layer_cosines(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if not np.all(np.isfinite(vectors)) or not np.all(np.isfinite(norms)) or np.any(norms <= 0):
        raise ValueError("Cosine similarity requires finite, nonzero coefficient vectors")
    unit = vectors / norms
    return np.clip(unit @ unit.T, -1, 1)


def load_coefficients():
    manifest = json.loads((FAMILY / "run_manifest.json").read_text())
    expected_names = [f"pair_{direction}_ch_{channel:03d}"
                      for direction in ("ij", "ji") for channel in range(128)]
    if manifest["configuration"]["feature_order"] != expected_names:
        raise ValueError("Unexpected directional feature order")
    with (FAMILY / "model_artifacts/replay_index.csv").open(newline="") as handle:
        index = list(csv.DictReader(handle))
    expected_keys = {(fold, layer) for fold in range(5) for layer in range(48)}
    if len(index) != 240 or {(int(r["outer_fold"]), int(r["layer"])) for r in index} != expected_keys:
        raise ValueError("Replay index must contain exactly five folds at every layer")
    coefficients = {space: np.empty((5, 48, 256), dtype=np.float64) for space in SPACES}
    sources = []
    for row in index:
        fold, layer = int(row["outer_fold"]), int(row["layer"])
        path = FAMILY / "model_artifacts/coefficients" / f"fold_{fold}_layer_{layer:02d}.npz"
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if (row["artifact_path"] != str(path.relative_to(FAMILY))
                or row["artifact_sha256"] != digest or row["status"] != "complete"):
            raise ValueError(f"Replay-index path/checksum/status mismatch: {path}")
        with np.load(path, allow_pickle=False) as archive:
            if (int(archive["outer_fold"]) != fold or int(archive["layer"]) != layer
                    or str(archive["analysis_name"]) != "shared_ols_base256"
                    or str(archive["model_family"]) != "shared_ols"
                    or str(archive["configuration_digest"]) != row["configuration_digest"]
                    or str(archive["scaler_digest"]) != row["scaler_digest"]):
                raise ValueError(f"Artifact identity mismatch: {path}")
            for space in SPACES:
                values = archive[f"coefficient_{space}"]
                if values.shape != (256,):
                    raise ValueError(f"Unexpected coefficient shape: {path}")
                coefficients[space][fold, layer] = values
            sources.append((fold, layer, str(path.relative_to(ROOT)), digest, str(archive["fit_status"])))
    return coefficients, sorted(sources)


def main() -> None:
    coefficients, sources = load_coefficients()
    similarities = {space: np.stack([layer_cosines(v) for v in coefficients[space]]) for space in SPACES}
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary = {
        "source_family": str(FAMILY.relative_to(ROOT)),
        "n_layers": 48, "n_outer_folds": 5, "n_features": 256,
        "intercept_included": False,
        "primary_coefficient_space": "standardized",
        "aggregation": "Arithmetic mean of five within-fold cosine similarities for each layer pair.",
        "source_replay_index_sha256": hashlib.sha256((FAMILY / "model_artifacts/replay_index.csv").read_bytes()).hexdigest(),
        "versions": {"numpy": np.__version__, "matplotlib": matplotlib.__version__},
        "statistics": {},
    }
    for space in SPACES:
        matrix = similarities[space].mean(axis=0)
        stem = "shared_ols_layer_cosine" if space == "standardized" else "shared_ols_raw_scale_layer_cosine"
        write_csv(OUTPUT / f"{stem}_matrix.csv", ["layer", *(f"layer_{i}" for i in range(48))],
                  ((i, *matrix[i]) for i in range(48)))
        fig, ax = plt.subplots(figsize=(9, 8), constrained_layout=True)
        heat = ax.imshow(matrix, origin="lower", cmap="coolwarm", vmin=-1, vmax=1,
                         interpolation="nearest")
        ax.set(title=f"Shared linear probe: layer coefficient cosine similarity\n"
                     f"{space.replace('_', ' ').capitalize()} coefficients · mean over five folds",
               xlabel="Layer", ylabel="Layer")
        ticks = [*range(0, 48, 4), 47]
        ax.set_xticks(ticks)
        ax.set_yticks(ticks)
        fig.colorbar(heat, ax=ax, label="Mean cosine similarity")
        for suffix in ("png", "pdf"):
            fig.savefig(OUTPUT / f"{stem}_heatmap.{suffix}", dpi=180)
        plt.close(fig)
        summary["statistics"][space] = {
            "mean_adjacent_layer_cosine": float(np.diag(matrix, k=1).mean()),
            "layer_0_vs_47_cosine": float(matrix[0, 47]),
            "minimum_off_diagonal_cosine": float(matrix[np.triu_indices(48, k=1)].min()),
        }
    write_csv(OUTPUT / "fold_similarities.csv",
              ["outer_fold", "layer_a", "layer_b", "cosine_standardized", "cosine_raw_scale"],
              ((f, a, b, *(similarities[s][f, a, b] for s in SPACES))
               for f in range(5) for a in range(48) for b in range(48)))
    write_csv(OUTPUT / "source_artifacts.csv", ["outer_fold", "layer", "artifact_path", "sha256", "fit_status"], sources)
    (OUTPUT / "generation_summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (OUTPUT / "README.md").write_text(
        "# Shared linear coefficient comparisons across layers\n\n"
        "The primary figure, `shared_ols_layer_cosine_heatmap.png` (also PDF), compares "
        "256 standardized coefficients, excluding intercepts. The raw-scale companion "
        "uses coefficients in original feature units. Features retain the fixed order "
        "[Z_ij channels 0–127, Z_ji channels 0–127].\n\n"
        "For each layer pair, compute cosine similarity within each outer fold, then "
        "average the five values equally. This compares models trained on matched protein "
        "sets; coefficient vectors are not averaged before calculating cosine similarity. "
        "The fixed color scale runs from −1 to +1. Similarity measures coefficient "
        "direction in the saved channel coordinates, not vector magnitude or prediction transfer.\n\n"
        "Each `*_matrix.csv` has 48 layer rows and 48 similarity columns plus a layer index. "
        "`fold_similarities.csv` retains one row per fold/ordered layer pair (11,520 rows), "
        "with standardized and raw-scale cosines. `source_artifacts.csv` records all 240 "
        "inputs, their verified replay-index checksums, and saved fit statuses; all fits "
        "are retained. `generation_summary.json` records aggregation and descriptive statistics.\n\n"
        "Regenerate from the repository root with "
        "`./.venv/bin/python scripts/multi_protein/plot_shared_linear_layer_cosines.py`. "
        "The source NPZ coefficient artifacts must be present locally; they are ignored by Git.\n"
    )
    print(json.dumps(summary["statistics"], indent=2))


if __name__ == "__main__":
    main()

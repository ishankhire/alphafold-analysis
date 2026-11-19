#!/usr/bin/env python3
"""Shared-linear probe transfer between layers, split by pair stratum.

Reads the per-protein sufficient statistics that the all-layer transfer run
stored in ``layer_shards/target_layer_TT.npz`` and replays nothing. For every
source layer, target layer and stratum it computes each protein's mean absolute
error, then averages over proteins with equal weight. Strata are the canonical
sequence-separation bins and final-structure-distance bins, one axis at a time.

Excess error is the transferred probe's per-protein absolute error minus that of
the target layer's own probe on the same protein and stratum.

Inputs: ``$ALL_LAYER_TRANSFER_OUTPUT_ROOT`` (``layer_shards/`` and
``transfer_matrix.csv``). Outputs under ``$PROBE_LAYER_SIMILARITY_OUTPUT_ROOT``:

- ``transfer_by_stratum.csv``: one row per stratum axis, stratum, source layer
  and target layer (25,344 rows), with ``n_proteins``, ``mean_protein_mae``,
  ``median_protein_mae``, ``mean_protein_excess_mae``,
  ``median_protein_excess_mae`` and ``native_target_mean_protein_mae``.
- ``figures/1{0,1}_transfer_mae_by_*.{png,pdf}`` and
  ``figures/1{2,3}_transfer_excess_mae_by_*.{png,pdf}``.
"""
from __future__ import annotations

import csv
import os
import sys
from pathlib import Path

import numpy as np

SCRIPT_DIR = Path(__file__).resolve().parent
for _path in (SCRIPT_DIR, SCRIPT_DIR.parents[1]):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

N_LAYERS = 48
# Category order of the shard's ``probe`` array: whole protein, then sequence, then distance.
CATEGORIES = (("all_pairs", "all"),
              *(("sequence_separation", label) for label in ("1", "2-4", "5-9", "10-19", "20-49", "50+")),
              *(("spatial_distance", label) for label in ("0-5", "5-10", "10-20", "20-40", "40+")))
STAT_N, STAT_SUM_ABS_ERROR = 0, 5
PARITY_RTOL = 1e-8
COLUMNS = ("stratum_axis", "stratum", "source_layer", "target_layer", "layer_offset", "n_proteins",
           "mean_protein_mae", "median_protein_mae", "mean_protein_excess_mae",
           "median_protein_excess_mae", "native_target_mean_protein_mae")


def transfer_root() -> Path:
    default = Path.home() / "diagnostic_experiments" / "cross_layer_probe_transfer_all_layers_v1"
    return Path(os.environ.get("ALL_LAYER_TRANSFER_OUTPUT_ROOT", default)).expanduser().resolve()


def output_root() -> Path:
    default = Path.home() / "diagnostic_experiments" / "probe_layer_similarity_v1"
    return Path(os.environ.get("PROBE_LAYER_SIMILARITY_OUTPUT_ROOT", default)).expanduser().resolve()


def protein_mae(probe: np.ndarray) -> np.ndarray:
    """``[source, protein, category]`` per-protein MAE; NaN where a protein has no pairs."""
    n = probe[..., STAT_N]
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n > 0, probe[..., STAT_SUM_ABS_ERROR] / n, np.nan)


def target_summary(probe: np.ndarray, target_layer: int) -> dict[str, np.ndarray]:
    """Protein-equal summaries for one target layer; arrays are ``[source, category]``."""
    mae = protein_mae(probe)
    excess = mae - mae[target_layer][None]
    return {"n_proteins": np.isfinite(mae).sum(axis=1),
            "mean_mae": np.nanmean(mae, axis=1), "median_mae": np.nanmedian(mae, axis=1),
            "mean_excess": np.nanmean(excess, axis=1), "median_excess": np.nanmedian(excess, axis=1)}


def load_summaries(root: Path, n_layers: int = N_LAYERS) -> dict[str, np.ndarray]:
    """``{name: [source, target, category]}`` from the per-target shards."""
    stacked: dict[str, list[np.ndarray]] = {}
    for target in range(n_layers):
        with np.load(root / "layer_shards" / f"target_layer_{target:02d}.npz", allow_pickle=False) as shard:
            if int(shard["target_layer"]) != target or not bool(np.all(shard["completed"])):
                raise ValueError(f"Shard for target layer {target} is incomplete or mislabeled")
            probe = shard["probe"]
        if probe.shape[0] != n_layers or probe.shape[2] != len(CATEGORIES):
            raise ValueError(f"Unexpected shard shape {probe.shape} for target layer {target}")
        for name, values in target_summary(probe, target).items():
            stacked.setdefault(name, []).append(values)
    return {name: np.stack(values, axis=1) for name, values in stacked.items()}


def check_parity(summaries, root: Path) -> float:
    """Whole-protein mean per-protein MAE must reproduce ``transfer_matrix.csv``."""
    worst = 0.0
    with open(root / "transfer_matrix.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            ours = summaries["mean_mae"][int(row["source_layer"]), int(row["target_layer"]), 0]
            expected = float(row["mean_protein_mae"])
            worst = max(worst, abs(ours - expected) / max(1.0, abs(expected)))
    if worst > PARITY_RTOL:
        raise ValueError(f"Mean per-protein MAE differs from transfer_matrix.csv by {worst:.3g}")
    return worst


def table_rows(summaries):
    n_layers = summaries["mean_mae"].shape[0]
    for c, (axis, label) in enumerate(CATEGORIES):
        for source in range(n_layers):
            for target in range(n_layers):
                yield (axis, label, source, target, target - source, int(summaries["n_proteins"][source, target, c]),
                       *(float(summaries[name][source, target, c])
                         for name in ("mean_mae", "median_mae", "mean_excess", "median_excess")),
                       float(summaries["mean_mae"][target, target, c]))


def draw(summaries, figure_root: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import make_probe_layer_similarity_figures as style

    axes_spec = (("sequence_separation", "sequence separation", "Sequence separation {}", (2, 3), (15, 10.2)),
                 ("spatial_distance", "final structure distance", "Final structure distance {} Å", (1, 5), (24, 5.6)))
    metrics = (("mean_mae", "transfer_mae", "Absolute error (Å)",
                "Shared linear probe transfer: mean per-protein absolute error by {}"),
               ("mean_excess", "transfer_excess_mae", "Excess absolute error over the layer's own probe (Å)",
                "Shared linear probe transfer: error added by transfer, by {}"))
    number = 10
    for key, stem, colorbar_label, title in metrics:
        for axis, axis_text, panel_title, shape, size in axes_spec:
            indices = [c for c, (a, _) in enumerate(CATEGORIES) if a == axis]
            fig, axes = plt.subplots(*shape, figsize=size, layout="constrained", squeeze=False)
            for ax, c in zip(axes.flat, indices):
                matrix = summaries[key][:, :, c]
                # Each stratum has its own scale; the top 5% of cells saturate.
                top = float(np.nanpercentile(matrix, 95))
                image = style.heatmap(ax, np.clip(matrix, 0, None), style.SEQUENTIAL, 0, top,
                                      panel_title.format(CATEGORIES[c][1].replace("-", "–")),
                                      "Layer the probe is applied to", "Layer the probe was trained on")
                fig.colorbar(image, ax=ax, shrink=0.8, extend="max")
            fig.suptitle(f"{title.format(axis_text)}\n{colorbar_label}, mean over proteins · "
                         "each panel has its own colour scale", fontsize=13)
            style.save(fig, figure_root, f"{number}_{stem}_by_{axis}")
            number += 1


def main() -> None:
    from canonical_storage import assert_writable
    source, root = transfer_root(), output_root()
    assert_writable(root)
    (root / "figures").mkdir(parents=True, exist_ok=True)
    summaries = load_summaries(source)
    worst = check_parity(summaries, source)
    temporary = root / "transfer_by_stratum.csv.tmp"
    with open(temporary, "w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(COLUMNS)
        writer.writerows(table_rows(summaries))
    temporary.replace(root / "transfer_by_stratum.csv")
    draw(summaries, root / "figures")
    print(f"wrote transfer_by_stratum.csv and four figures to {root}; parity max relative difference {worst:.2e}")


if __name__ == "__main__":
    main()

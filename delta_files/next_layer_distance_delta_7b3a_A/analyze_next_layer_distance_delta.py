#!/usr/bin/env python3
"""Analyze the 7b3a_A next-layer distance-delta diagnostic outputs."""

from __future__ import annotations

import csv
import math
import os
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = SCRIPT_DIR.parents[1]
os.makedirs(PROJECT_ROOT / ".mplconfig", exist_ok=True)
os.makedirs(PROJECT_ROOT / ".cache", exist_ok=True)
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


SUMMARY_CSV = SCRIPT_DIR / "next_layer_distance_delta_summary.csv"
R2_PLOT = SCRIPT_DIR / "next_layer_distance_delta_r2_over_layers.png"
GAP_PLOT = SCRIPT_DIR / "next_layer_distance_delta_train_test_gap.png"
ANALYSIS_CSV = SCRIPT_DIR / "next_layer_distance_delta_analysis_summary.csv"
REPORT_TXT = SCRIPT_DIR / "next_layer_distance_delta_analysis_report.txt"

EXPECTED_LAYERS = tuple(range(47))
EXPECTED_ROW_COUNT = 47 * 7
CENTER_KEY = ("center_pair_plus_current_distance", "")
FULL_CONTEXT_KEYS = (
    ("full_context", "5"),
    ("full_context", "10"),
    ("full_context", "20"),
)
FITTED_VARIANTS = {
    ("current_distance", ""),
    CENTER_KEY,
    *FULL_CONTEXT_KEYS,
}

VARIANT_ORDER = (
    ("zero_delta", ""),
    ("train_mean_delta", ""),
    ("current_distance", ""),
    CENTER_KEY,
    ("full_context", "5"),
    ("full_context", "10"),
    ("full_context", "20"),
)

LABELS = {
    ("zero_delta", ""): "Zero delta baseline",
    ("train_mean_delta", ""): "Train-mean delta baseline",
    ("current_distance", ""): "Current decoded distance",
    CENTER_KEY: "Center pair + current distance",
    ("full_context", "5"): "Full context k=5",
    ("full_context", "10"): "Full context k=10",
    ("full_context", "20"): "Full context k=20",
}

COLORS = {
    ("zero_delta", ""): "#8c8c8c",
    ("train_mean_delta", ""): "#4c78a8",
    ("current_distance", ""): "#72b7b2",
    CENTER_KEY: "#f58518",
    ("full_context", "5"): "#54a24b",
    ("full_context", "10"): "#e45756",
    ("full_context", "20"): "#b279a2",
}

SUMMARY_FIELDS = [
    "variant_label",
    "variant",
    "k_neighbors",
    "n_layers",
    "mean_train_r2",
    "mean_test_r2",
    "min_test_r2",
    "min_test_r2_layer",
    "max_test_r2",
    "max_test_r2_layer",
    "mean_train_minus_test_r2",
    "mean_test_r2_gain_vs_center",
    "winning_layers_vs_center",
]


@dataclass(frozen=True)
class SummaryRow:
    variant: str
    k_neighbors: str
    layer: int
    estimator: str
    train_r2: float
    test_r2: float

    @property
    def key(self) -> tuple[str, str]:
        return (self.variant, self.k_neighbors)


def parse_float(value: str) -> float:
    if value == "":
        return float("nan")
    return float(value)


def format_float(value: float) -> str:
    if not math.isfinite(value):
        return "nan"
    return f"{value:.6f}"


def variant_label(key: tuple[str, str]) -> str:
    return LABELS.get(key, key[0] if not key[1] else f"{key[0]} k={key[1]}")


def load_summary_rows(path: Path) -> list[SummaryRow]:
    if not path.exists():
        raise FileNotFoundError(f"Missing summary CSV: {path}")

    required = {"variant", "layer", "k_neighbors", "estimator", "train_r2", "test_r2"}
    rows: list[SummaryRow] = []
    with path.open(newline="") as handle:
        reader = csv.DictReader(handle)
        missing = sorted(required - set(reader.fieldnames or []))
        if missing:
            raise ValueError(f"{path} is missing columns: {missing}")
        for raw in reader:
            rows.append(
                SummaryRow(
                    variant=raw["variant"],
                    k_neighbors=raw["k_neighbors"],
                    layer=int(raw["layer"]),
                    estimator=raw["estimator"],
                    train_r2=parse_float(raw["train_r2"]),
                    test_r2=parse_float(raw["test_r2"]),
                )
            )

    if len(rows) != EXPECTED_ROW_COUNT:
        raise ValueError(f"Expected {EXPECTED_ROW_COUNT} summary rows, found {len(rows)}")

    observed_layers = tuple(sorted({row.layer for row in rows}))
    if observed_layers != EXPECTED_LAYERS:
        raise ValueError(f"Expected layers {EXPECTED_LAYERS}, found {observed_layers}")

    return rows


def group_by_variant(rows: list[SummaryRow]) -> dict[tuple[str, str], list[SummaryRow]]:
    grouped: dict[tuple[str, str], list[SummaryRow]] = defaultdict(list)
    for row in rows:
        grouped[row.key].append(row)

    for key, values in grouped.items():
        values.sort(key=lambda row: row.layer)
        layers = tuple(row.layer for row in values)
        if layers != EXPECTED_LAYERS:
            raise ValueError(f"{variant_label(key)} has layers {layers}, expected {EXPECTED_LAYERS}")

    expected_keys = set(VARIANT_ORDER)
    observed_keys = set(grouped)
    if observed_keys != expected_keys:
        raise ValueError(
            "Unexpected variant set: "
            f"expected {sorted(expected_keys)}, found {sorted(observed_keys)}"
        )

    return dict(grouped)


def finite_mean(values: list[float]) -> float:
    finite = [value for value in values if math.isfinite(value)]
    if not finite:
        return float("nan")
    return sum(finite) / len(finite)


def write_analysis_csv(grouped: dict[tuple[str, str], list[SummaryRow]]) -> None:
    center_by_layer = {row.layer: row.test_r2 for row in grouped[CENTER_KEY]}
    output_rows = []
    for key in VARIANT_ORDER:
        values = grouped[key]
        train_values = [row.train_r2 for row in values]
        test_values = [row.test_r2 for row in values]
        gaps = [row.train_r2 - row.test_r2 for row in values]
        min_row = min(values, key=lambda row: row.test_r2)
        max_row = max(values, key=lambda row: row.test_r2)

        if key in FULL_CONTEXT_KEYS:
            gains = [row.test_r2 - center_by_layer[row.layer] for row in values]
            mean_gain = finite_mean(gains)
            winning_layers = sum(1 for gain in gains if gain > 0.0)
        else:
            mean_gain = float("nan")
            winning_layers = ""

        output_rows.append(
            {
                "variant_label": variant_label(key),
                "variant": key[0],
                "k_neighbors": key[1],
                "n_layers": len(values),
                "mean_train_r2": format_float(finite_mean(train_values)),
                "mean_test_r2": format_float(finite_mean(test_values)),
                "min_test_r2": format_float(min_row.test_r2),
                "min_test_r2_layer": min_row.layer,
                "max_test_r2": format_float(max_row.test_r2),
                "max_test_r2_layer": max_row.layer,
                "mean_train_minus_test_r2": format_float(finite_mean(gaps)),
                "mean_test_r2_gain_vs_center": format_float(mean_gain),
                "winning_layers_vs_center": winning_layers,
            }
        )

    with ANALYSIS_CSV.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=SUMMARY_FIELDS)
        writer.writeheader()
        writer.writerows(output_rows)


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 11,
            "axes.titlesize": 15,
            "axes.labelsize": 13,
            "xtick.labelsize": 11,
            "ytick.labelsize": 11,
            "legend.fontsize": 10,
            "figure.titlesize": 16,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.linewidth": 1.0,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def style_axis(ax: plt.Axes) -> None:
    ax.set_xlim(0, 46)
    ax.set_xticks([0, 8, 16, 24, 32, 40, 46])
    ax.axhline(0.0, color="#4d4d4d", linewidth=0.9, alpha=0.75)
    ax.grid(True, color="#dddddd", linewidth=0.8, alpha=0.75)
    ax.tick_params(axis="both", width=1.0, length=5)


def save_r2_plot(grouped: dict[tuple[str, str], list[SummaryRow]]) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    for key in VARIANT_ORDER:
        values = grouped[key]
        ax.plot(
            [row.layer for row in values],
            [row.test_r2 for row in values],
            color=COLORS[key],
            linewidth=2.4 if key == CENTER_KEY else 1.9,
            alpha=0.95,
            label=variant_label(key),
        )

    ax.set_title("Held-out R$^2$ for decoded distance change")
    ax.set_xlabel("Input layer")
    ax.set_ylabel("Held-out delta R$^2$")
    style_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(R2_PLOT)
    plt.close(fig)


def save_gap_plot(grouped: dict[tuple[str, str], list[SummaryRow]]) -> None:
    fig, ax = plt.subplots(figsize=(10.8, 6.2), constrained_layout=True)
    for key in VARIANT_ORDER:
        if key not in FITTED_VARIANTS:
            continue
        values = grouped[key]
        ax.plot(
            [row.layer for row in values],
            [row.train_r2 - row.test_r2 for row in values],
            color=COLORS[key],
            linewidth=2.4 if key == CENTER_KEY else 1.9,
            alpha=0.95,
            label=variant_label(key),
        )

    ax.set_title("Train-test R$^2$ gap for fitted probes")
    ax.set_xlabel("Input layer")
    ax.set_ylabel("Train R$^2$ minus held-out R$^2$")
    style_axis(ax)
    ax.legend(loc="center left", bbox_to_anchor=(1.01, 0.5), frameon=False)
    fig.savefig(GAP_PLOT)
    plt.close(fig)


def context_gain_lines(grouped: dict[tuple[str, str], list[SummaryRow]]) -> list[str]:
    center_by_layer = {row.layer: row.test_r2 for row in grouped[CENTER_KEY]}
    lines = []
    for key in FULL_CONTEXT_KEYS:
        values = grouped[key]
        gains = [row.test_r2 - center_by_layer[row.layer] for row in values]
        winning_layers = [row.layer for row, gain in zip(values, gains) if gain > 0.0]
        lines.append(
            "- "
            f"{variant_label(key)}: mean held-out gain vs center = {finite_mean(gains):.4f}; "
            f"wins on {len(winning_layers)}/47 layers"
        )
    return lines


def write_report(grouped: dict[tuple[str, str], list[SummaryRow]]) -> None:
    best_by_mean = max(
        VARIANT_ORDER,
        key=lambda key: finite_mean([row.test_r2 for row in grouped[key]]),
    )
    center_mean = finite_mean([row.test_r2 for row in grouped[CENTER_KEY]])
    lines = [
        "Next-Layer Distance-Delta Analysis",
        "==================================",
        "",
        f"Source summary CSV: {SUMMARY_CSV.name}",
        f"Rows analyzed: {sum(len(values) for values in grouped.values())}",
        f"Layer range: 0-46",
        "",
        "Primary metric",
        "--------------",
        "R2 is the held-out R2 for predicted decoded-distance deltas, matching the "
        "target fitted by diagnostic_next_layer_distance_delta_regression.py.",
        "",
        "Best mean held-out R2",
        "---------------------",
        f"{variant_label(best_by_mean)}: "
        f"{finite_mean([row.test_r2 for row in grouped[best_by_mean]]):.4f}",
        f"Center pair + current distance mean held-out R2: {center_mean:.4f}",
        "",
        "Full-context gain over center-only",
        "----------------------------------",
        *context_gain_lines(grouped),
        "",
        "Interpretation",
        "--------------",
        "The context features are a global ridge probe over ranked endpoint-neighbor "
        "pair representations. They help only if those neighbor ranks expose a shared "
        "linear basis for distance-change information across residue pairs.",
        "In this run, larger context windows have much higher train R2 than held-out "
        "R2 and usually underperform the center-pair model, which is consistent with "
        "overfitting or heterogeneous neighbor roles rather than a clean shared "
        "distance-change readout.",
        "",
        "Outputs",
        "-------",
        f"- {R2_PLOT.name}",
        f"- {GAP_PLOT.name}",
        f"- {ANALYSIS_CSV.name}",
    ]
    REPORT_TXT.write_text("\n".join(lines) + "\n")


def main() -> None:
    rows = load_summary_rows(SUMMARY_CSV)
    grouped = group_by_variant(rows)
    configure_style()
    write_analysis_csv(grouped)
    save_r2_plot(grouped)
    save_gap_plot(grouped)
    write_report(grouped)
    print(f"Wrote {ANALYSIS_CSV}")
    print(f"Wrote {R2_PLOT}")
    print(f"Wrote {GAP_PLOT}")
    print(f"Wrote {REPORT_TXT}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Create compact paper-facing summary statistics from Delta regression outputs."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DELTA_DIR = PROJECT_ROOT / "delta_files"
OUT_DIR = Path(__file__).resolve().parent
EXPECTED_LAYERS = tuple(range(48))
TM_THRESHOLD = 0.6
VARIANT_ORDER = {"Variant A": 0, "Variant B": 1, "B minus A": 2}
METRIC_ORDER = {"r2": 0, "tm_score": 1, "r2_vs_tm_score": 2}
SPLIT_ORDER = {"test": 0, "train": 1, "all": 2}
ROW_TYPE_ORDER = {"ever_negative_summary": 0, "layer_negative_count": 1}
STATISTIC_ORDER = {
    "best_mean_layer": 0,
    "best_mean_value": 1,
    "layer47_mean": 2,
    "layer47_minus_best_mean": 3,
    "layer47_paired_delta_mean": 4,
    "layer47_paired_delta_median": 5,
    "layer47_paired_delta_std": 6,
    "layer47_n_improved": 7,
    "layer47_fraction_improved": 8,
    "layer47_pearson_correlation": 9,
}
INTEGER_COLUMNS = {
    "layer",
    "n",
    "n_proteins",
    "n_test_proteins",
    "n_negative_test_proteins",
    "n_test_proteins_ever_negative",
    "first_layer_mean_tm_ge_threshold",
    "first_layer_with_no_negative_r2_through_layer47",
}

VARIANTS = [
    {
        "folder": "general_base256_avg_symmetric",
        "short_label": "Variant A",
        "paper_label": "Variant A: averaged pair representations",
    },
    {
        "folder": "general_base256_avg_symmetric_plus_axial",
        "short_label": "Variant B",
        "paper_label": "Variant B: averaged pair + neighboring pair representations",
    },
]

METRICS = [
    {
        "metric": "r2",
        "display": "R^2",
        "source": "per_protein",
        "column": "per_protein_r2",
    },
    {
        "metric": "tm_score",
        "display": "TM-score",
        "source": "geometry",
        "column": "tm_score",
    },
]


def load_variant_metric(variant: dict[str, str], metric: dict[str, str]) -> pd.DataFrame:
    filename = "per_protein.csv" if metric["source"] == "per_protein" else "geometry_metrics.csv"
    path = DELTA_DIR / variant["folder"] / filename
    df = pd.read_csv(path)
    required = {"variant", "layer", "split", "protein_id", metric["column"]}
    missing = sorted(required - set(df.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")

    df = df[["variant", "layer", "split", "protein_id", metric["column"]]].copy()
    df["metric"] = metric["metric"]
    df["metric_display"] = metric["display"]
    df["value"] = pd.to_numeric(df[metric["column"]], errors="coerce")
    df["short_label"] = variant["short_label"]
    df["paper_label"] = variant["paper_label"]
    df["source_file"] = str(path.relative_to(PROJECT_ROOT))
    df = df.drop(columns=[metric["column"]])
    validate_metric_frame(df, path)
    return df


def validate_metric_frame(df: pd.DataFrame, path: Path) -> None:
    layers = tuple(sorted(int(value) for value in df["layer"].dropna().unique()))
    if layers != EXPECTED_LAYERS:
        raise ValueError(f"{path} expected layers 0-47, found {layers}")

    splits = tuple(sorted(str(value) for value in df["split"].dropna().unique()))
    if splits != ("test", "train"):
        raise ValueError(f"{path} expected train/test splits, found {splits}")

    counts = df.groupby(["split", "protein_id", "layer"], sort=True).size()
    if not (counts == 1).all():
        raise ValueError(f"{path} rows are not unique per (split, protein_id, layer)")

    if df["value"].isna().any():
        n_missing = int(df["value"].isna().sum())
        raise ValueError(f"{path} contains {n_missing} missing metric values")


def with_all_split(df: pd.DataFrame) -> pd.DataFrame:
    all_rows = df.copy()
    all_rows["split"] = "all"
    return pd.concat([df, all_rows], ignore_index=True)


def sample_std(values: pd.Series) -> float:
    return float(values.std(ddof=1))


def layer47_statistics(all_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    layer47 = all_metrics[all_metrics["layer"] == 47]
    grouped = layer47.groupby(
        ["paper_label", "short_label", "variant", "metric", "metric_display", "split"],
        sort=False,
    )
    for keys, group in grouped:
        paper_label, short_label, variant, metric, metric_display, split = keys
        values = group["value"]
        rows.append(
            {
                "paper_label": paper_label,
                "short_label": short_label,
                "variant": variant,
                "metric": metric,
                "metric_display": metric_display,
                "split_category": split,
                "layer": 47,
                "n_proteins": int(values.size),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "std": sample_std(values),
                "min": float(values.min()),
                "max": float(values.max()),
            }
        )
    return pd.DataFrame(rows)


def tm_thresholds(all_metrics: pd.DataFrame) -> pd.DataFrame:
    tm = all_metrics[all_metrics["metric"] == "tm_score"].copy()
    means = (
        tm.groupby(["paper_label", "short_label", "variant", "split", "layer"], sort=False)["value"]
        .mean()
        .reset_index(name="mean_tm_score")
    )
    rows = []
    grouped = means.groupby(["paper_label", "short_label", "variant", "split"], sort=False)
    for keys, group in grouped:
        paper_label, short_label, variant, split = keys
        crossed = group[group["mean_tm_score"] >= TM_THRESHOLD].sort_values("layer")
        if crossed.empty:
            first_layer = np.nan
            mean_at_first = np.nan
            crossed_text = "no"
        else:
            first = crossed.iloc[0]
            first_layer = int(first["layer"])
            mean_at_first = float(first["mean_tm_score"])
            crossed_text = "yes"
        rows.append(
            {
                "paper_label": paper_label,
                "short_label": short_label,
                "variant": variant,
                "split_category": split,
                "threshold": TM_THRESHOLD,
                "crossed_threshold": crossed_text,
                "first_layer_mean_tm_ge_threshold": first_layer,
                "mean_tm_score_at_first_layer": mean_at_first,
            }
        )
    return pd.DataFrame(rows)


def negative_r2_summary(all_metrics: pd.DataFrame) -> pd.DataFrame:
    r2_test = all_metrics[
        (all_metrics["metric"] == "r2")
        & (all_metrics["split"] == "test")
    ].copy()
    rows = []
    grouped = r2_test.groupby(["paper_label", "short_label", "variant"], sort=False)
    for keys, group in grouped:
        paper_label, short_label, variant = keys
        n_test = int(group["protein_id"].nunique())
        ever_negative = group.loc[group["value"] < 0.0, "protein_id"].drop_duplicates().sort_values()
        layer_counts = (
            group.assign(is_negative=group["value"] < 0.0)
            .groupby("layer", sort=True)["is_negative"]
            .sum()
            .astype(int)
        )
        zero_negative_layers = [int(layer) for layer, count in layer_counts.items() if int(count) == 0]
        first_persistent_zero = first_layer_zero_persistent(layer_counts)
        summary_values = {
            "paper_label": paper_label,
            "short_label": short_label,
            "variant": variant,
            "row_type": "ever_negative_summary",
            "layer": np.nan,
            "n_test_proteins": n_test,
            "n_negative_test_proteins": np.nan,
            "fraction_negative_test_proteins": np.nan,
            "n_test_proteins_ever_negative": int(ever_negative.size),
            "fraction_test_proteins_ever_negative": float(ever_negative.size / n_test),
            "zero_negative_layers": join_ints(zero_negative_layers),
            "first_layer_with_no_negative_r2_through_layer47": first_persistent_zero,
            "negative_protein_ids": ";".join(ever_negative.tolist()),
        }
        rows.append(summary_values)
        for layer, count in layer_counts.items():
            rows.append(
                {
                    "paper_label": paper_label,
                    "short_label": short_label,
                    "variant": variant,
                    "row_type": "layer_negative_count",
                    "layer": int(layer),
                    "n_test_proteins": n_test,
                    "n_negative_test_proteins": int(count),
                    "fraction_negative_test_proteins": float(count / n_test),
                    "n_test_proteins_ever_negative": int(ever_negative.size),
                    "fraction_test_proteins_ever_negative": float(ever_negative.size / n_test),
                    "zero_negative_layers": join_ints(zero_negative_layers),
                    "first_layer_with_no_negative_r2_through_layer47": first_persistent_zero,
                    "negative_protein_ids": "",
                }
            )
    return pd.DataFrame(rows)


def first_layer_zero_persistent(layer_counts: pd.Series) -> float:
    for layer in EXPECTED_LAYERS:
        tail = layer_counts[layer_counts.index >= layer]
        if not tail.empty and bool((tail == 0).all()):
            return int(layer)
    return np.nan


def join_ints(values: list[int]) -> str:
    return ";".join(str(value) for value in values)


def extra_paper_stats(all_metrics: pd.DataFrame) -> pd.DataFrame:
    rows = []
    rows.extend(best_layer_rows(all_metrics))
    rows.extend(variant_delta_rows(all_metrics))
    rows.extend(layer47_r2_tm_correlation_rows(all_metrics))
    return pd.DataFrame(rows)


def best_layer_rows(all_metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    means = (
        all_metrics.groupby(
            ["paper_label", "short_label", "variant", "metric", "metric_display", "split", "layer"],
            sort=False,
        )["value"]
        .mean()
        .reset_index(name="layer_mean")
    )
    grouped = means.groupby(
        ["paper_label", "short_label", "variant", "metric", "metric_display", "split"],
        sort=False,
    )
    for keys, group in grouped:
        paper_label, short_label, variant, metric, metric_display, split = keys
        best = group.sort_values(["layer_mean", "layer"], ascending=[False, True]).iloc[0]
        layer47 = group[group["layer"] == 47].iloc[0]
        rows.extend(
            [
                {
                    "paper_label": paper_label,
                    "short_label": short_label,
                    "variant": variant,
                    "metric": metric,
                    "metric_display": metric_display,
                    "split_category": split,
                    "statistic": "best_mean_layer",
                    "layer": int(best["layer"]),
                    "value": int(best["layer"]),
                    "n": np.nan,
                    "notes": "Layer with the highest mean per-protein metric.",
                },
                {
                    "paper_label": paper_label,
                    "short_label": short_label,
                    "variant": variant,
                    "metric": metric,
                    "metric_display": metric_display,
                    "split_category": split,
                    "statistic": "best_mean_value",
                    "layer": int(best["layer"]),
                    "value": float(best["layer_mean"]),
                    "n": np.nan,
                    "notes": "Highest mean per-protein metric across layers.",
                },
                {
                    "paper_label": paper_label,
                    "short_label": short_label,
                    "variant": variant,
                    "metric": metric,
                    "metric_display": metric_display,
                    "split_category": split,
                    "statistic": "layer47_mean",
                    "layer": 47,
                    "value": float(layer47["layer_mean"]),
                    "n": np.nan,
                    "notes": "Mean per-protein metric at the final layer.",
                },
                {
                    "paper_label": paper_label,
                    "short_label": short_label,
                    "variant": variant,
                    "metric": metric,
                    "metric_display": metric_display,
                    "split_category": split,
                    "statistic": "layer47_minus_best_mean",
                    "layer": 47,
                    "value": float(layer47["layer_mean"] - best["layer_mean"]),
                    "n": np.nan,
                    "notes": "Final-layer mean minus the best layer mean.",
                },
            ]
        )
    return rows


def variant_delta_rows(all_metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    layer47 = all_metrics[all_metrics["layer"] == 47].copy()
    for metric in METRICS:
        metric_rows = layer47[layer47["metric"] == metric["metric"]]
        for split in ["train", "test", "all"]:
            split_rows = metric_rows[metric_rows["split"] == split]
            wide = split_rows.pivot_table(
                index=["protein_id"],
                columns="short_label",
                values="value",
                aggfunc="first",
            )
            if {"Variant A", "Variant B"} - set(wide.columns):
                continue
            delta = wide["Variant B"] - wide["Variant A"]
            rows.append(
                {
                    "paper_label": "Variant B - Variant A",
                    "short_label": "B minus A",
                    "variant": "general_base256_avg_symmetric_plus_axial__minus__general_base256_avg_symmetric",
                    "metric": metric["metric"],
                    "metric_display": metric["display"],
                    "split_category": split,
                    "statistic": "layer47_paired_delta_mean",
                    "layer": 47,
                    "value": float(delta.mean()),
                    "n": int(delta.size),
                    "notes": "Mean paired protein-level layer-47 difference.",
                }
            )
            rows.append(
                {
                    "paper_label": "Variant B - Variant A",
                    "short_label": "B minus A",
                    "variant": "general_base256_avg_symmetric_plus_axial__minus__general_base256_avg_symmetric",
                    "metric": metric["metric"],
                    "metric_display": metric["display"],
                    "split_category": split,
                    "statistic": "layer47_paired_delta_median",
                    "layer": 47,
                    "value": float(delta.median()),
                    "n": int(delta.size),
                    "notes": "Median paired protein-level layer-47 difference.",
                }
            )
            rows.append(
                {
                    "paper_label": "Variant B - Variant A",
                    "short_label": "B minus A",
                    "variant": "general_base256_avg_symmetric_plus_axial__minus__general_base256_avg_symmetric",
                    "metric": metric["metric"],
                    "metric_display": metric["display"],
                    "split_category": split,
                    "statistic": "layer47_paired_delta_std",
                    "layer": 47,
                    "value": sample_std(delta),
                    "n": int(delta.size),
                    "notes": "Sample standard deviation of paired layer-47 differences.",
                }
            )
            rows.append(
                {
                    "paper_label": "Variant B - Variant A",
                    "short_label": "B minus A",
                    "variant": "general_base256_avg_symmetric_plus_axial__minus__general_base256_avg_symmetric",
                    "metric": metric["metric"],
                    "metric_display": metric["display"],
                    "split_category": split,
                    "statistic": "layer47_n_improved",
                    "layer": 47,
                    "value": int((delta > 0.0).sum()),
                    "n": int(delta.size),
                    "notes": "Number of proteins where Variant B exceeds Variant A at layer 47.",
                }
            )
            rows.append(
                {
                    "paper_label": "Variant B - Variant A",
                    "short_label": "B minus A",
                    "variant": "general_base256_avg_symmetric_plus_axial__minus__general_base256_avg_symmetric",
                    "metric": metric["metric"],
                    "metric_display": metric["display"],
                    "split_category": split,
                    "statistic": "layer47_fraction_improved",
                    "layer": 47,
                    "value": float((delta > 0.0).mean()),
                    "n": int(delta.size),
                    "notes": "Fraction of proteins where Variant B exceeds Variant A at layer 47.",
                }
            )
    return rows


def layer47_r2_tm_correlation_rows(all_metrics: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    layer47_test = all_metrics[
        (all_metrics["layer"] == 47)
        & (all_metrics["split"] == "test")
        & (all_metrics["metric"].isin(["r2", "tm_score"]))
    ]
    grouped = layer47_test.groupby(["paper_label", "short_label", "variant"], sort=False)
    for keys, group in grouped:
        paper_label, short_label, variant = keys
        wide = group.pivot_table(index="protein_id", columns="metric", values="value", aggfunc="first")
        corr = float(wide["r2"].corr(wide["tm_score"]))
        rows.append(
            {
                "paper_label": paper_label,
                "short_label": short_label,
                "variant": variant,
                "metric": "r2_vs_tm_score",
                "metric_display": "R^2 vs TM-score",
                "split_category": "test",
                "statistic": "layer47_pearson_correlation",
                "layer": 47,
                "value": corr,
                "n": int(wide.dropna().shape[0]),
                "notes": "Pearson correlation across held-out proteins at layer 47.",
            }
        )
    return rows


def write_csv(df: pd.DataFrame, filename: str) -> None:
    path = OUT_DIR / filename
    output = df.copy()
    for column in INTEGER_COLUMNS & set(output.columns):
        output[column] = output[column].map(format_integer_cell)
    output.to_csv(path, index=False, float_format="%.6f")


def sort_for_output(df: pd.DataFrame) -> pd.DataFrame:
    output = df.copy()
    output["_variant_order"] = mapped_order(output, "short_label", VARIANT_ORDER)
    output["_metric_order"] = mapped_order(output, "metric", METRIC_ORDER)
    split_column = "split_category" if "split_category" in output.columns else "split"
    output["_split_order"] = mapped_order(output, split_column, SPLIT_ORDER)
    output["_row_type_order"] = mapped_order(output, "row_type", ROW_TYPE_ORDER)
    output["_statistic_order"] = mapped_order(output, "statistic", STATISTIC_ORDER)
    if "layer" in output.columns:
        output["_layer_order"] = pd.to_numeric(output["layer"], errors="coerce").fillna(-1)
    else:
        output["_layer_order"] = -1
    sort_columns = [
        "_variant_order",
        "_metric_order",
        "_split_order",
        "_row_type_order",
        "_layer_order",
        "_statistic_order",
    ]
    output = output.sort_values(sort_columns, kind="mergesort").reset_index(drop=True)
    return output.drop(columns=sort_columns)


def mapped_order(df: pd.DataFrame, column: str, mapping: dict[str, int]) -> pd.Series:
    if column not in df.columns:
        return pd.Series(99, index=df.index)
    return df[column].map(mapping).fillna(99)


def format_integer_cell(value: object) -> str:
    if value == "" or pd.isna(value):
        return ""
    return str(int(value))


def fmt(value: object, digits: int = 3) -> str:
    if pd.isna(value):
        return "NA"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    value_f = float(value)
    return f"{value_f:.{digits}f}"


def fmt_layer(value: object) -> str:
    if pd.isna(value):
        return "NA"
    return str(int(value))


def markdown_table(df: pd.DataFrame, columns: list[str], headers: list[str]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[column]) for column in columns) + " |")
    return "\n".join(lines)


def make_markdown(
    layer47: pd.DataFrame,
    thresholds: pd.DataFrame,
    negative: pd.DataFrame,
    extras: pd.DataFrame,
) -> str:
    lines: list[str] = []
    lines.append("# Publication Summary Statistics")
    lines.append("")
    lines.append("Sources: `delta_files/general_base256_avg_symmetric/` and `delta_files/general_base256_avg_symmetric_plus_axial/`.")
    lines.append("All statistics are unweighted across proteins unless otherwise noted. `all` combines train and test proteins.")
    lines.append("")

    lines.append("## Layer 47")
    layer47_display = layer47.copy()
    layer47_display["mean"] = layer47_display["mean"].map(fmt)
    layer47_display["median"] = layer47_display["median"].map(fmt)
    layer47_display["std"] = layer47_display["std"].map(fmt)
    layer47_display["min"] = layer47_display["min"].map(fmt)
    layer47_display["max"] = layer47_display["max"].map(fmt)
    layer47_display = layer47_display[
        [
            "short_label",
            "metric_display",
            "split_category",
            "n_proteins",
            "mean",
            "median",
            "std",
            "min",
            "max",
        ]
    ]
    lines.append(
        markdown_table(
            layer47_display,
            list(layer47_display.columns),
            ["Variant", "Metric", "Split", "n", "Mean", "Median", "Std", "Min", "Max"],
        )
    )
    lines.append("")

    lines.append("## TM-score Threshold")
    threshold_display = thresholds.copy()
    threshold_display["first_layer_mean_tm_ge_threshold"] = threshold_display[
        "first_layer_mean_tm_ge_threshold"
    ].map(fmt_layer)
    threshold_display["mean_tm_score_at_first_layer"] = threshold_display[
        "mean_tm_score_at_first_layer"
    ].map(fmt)
    threshold_display = threshold_display[
        [
            "short_label",
            "split_category",
            "threshold",
            "crossed_threshold",
            "first_layer_mean_tm_ge_threshold",
            "mean_tm_score_at_first_layer",
        ]
    ]
    threshold_display["threshold"] = threshold_display["threshold"].map(fmt)
    lines.append(
        markdown_table(
            threshold_display,
            list(threshold_display.columns),
            ["Variant", "Split", "Threshold", "Crossed", "First Layer", "Mean TM There"],
        )
    )
    lines.append("")

    lines.append("## Negative Held-out R^2")
    neg_summary = negative[negative["row_type"] == "ever_negative_summary"].copy()
    neg_summary["fraction_test_proteins_ever_negative"] = neg_summary[
        "fraction_test_proteins_ever_negative"
    ].map(fmt)
    neg_summary["first_layer_with_no_negative_r2_through_layer47"] = neg_summary[
        "first_layer_with_no_negative_r2_through_layer47"
    ].map(fmt_layer)
    neg_summary = neg_summary[
        [
            "short_label",
            "n_test_proteins",
            "n_test_proteins_ever_negative",
            "fraction_test_proteins_ever_negative",
            "zero_negative_layers",
            "first_layer_with_no_negative_r2_through_layer47",
        ]
    ]
    lines.append(
        markdown_table(
            neg_summary,
            list(neg_summary.columns),
            ["Variant", "Test n", "Ever Negative n", "Ever Negative Fraction", "Zero-Negative Layers", "Zero Through Layer 47 From"],
        )
    )
    lines.append("")
    lines.append("Per-layer negative R^2 counts are in `publication_negative_r2_summary.csv`.")
    lines.append("")

    lines.append("## Additional Useful Numbers")
    lines.extend(additional_numbers_lines(extras))
    lines.append("")
    return "\n".join(lines)


def additional_numbers_lines(extras: pd.DataFrame) -> list[str]:
    lines: list[str] = []
    best = extras[extras["statistic"].isin(["best_mean_layer", "best_mean_value", "layer47_mean"])].copy()
    best_test = best[best["split_category"] == "test"]
    for short_label in ["Variant A", "Variant B"]:
        for metric in ["r2", "tm_score"]:
            subset = best_test[(best_test["short_label"] == short_label) & (best_test["metric"] == metric)]
            if subset.empty:
                continue
            best_layer = subset[subset["statistic"] == "best_mean_layer"].iloc[0]
            best_value = subset[subset["statistic"] == "best_mean_value"].iloc[0]
            layer47 = subset[subset["statistic"] == "layer47_mean"].iloc[0]
            lines.append(
                f"- {short_label} held-out {best_layer['metric_display']} peaks at layer "
                f"{fmt_layer(best_layer['layer'])} with mean {fmt(best_value['value'])}; "
                f"layer 47 mean is {fmt(layer47['value'])}."
            )

    deltas = extras[
        (extras["short_label"] == "B minus A")
        & (extras["split_category"] == "test")
        & (extras["statistic"].isin(["layer47_paired_delta_mean", "layer47_n_improved", "layer47_fraction_improved"]))
    ]
    for metric in ["r2", "tm_score"]:
        subset = deltas[deltas["metric"] == metric]
        if subset.empty:
            continue
        mean_delta = subset[subset["statistic"] == "layer47_paired_delta_mean"].iloc[0]
        n_improved = subset[subset["statistic"] == "layer47_n_improved"].iloc[0]
        frac_improved = subset[subset["statistic"] == "layer47_fraction_improved"].iloc[0]
        lines.append(
            f"- At layer 47 on held-out proteins, Variant B minus Variant A mean "
            f"{mean_delta['metric_display']} delta is {fmt(mean_delta['value'])}; "
            f"{int(n_improved['value'])}/{int(n_improved['n'])} proteins improved "
            f"({fmt(frac_improved['value'])})."
        )

    corrs = extras[extras["statistic"] == "layer47_pearson_correlation"]
    for _, row in corrs.iterrows():
        lines.append(
            f"- {row['short_label']} layer-47 held-out R^2/TM-score Pearson correlation: "
            f"{fmt(row['value'])} across {int(row['n'])} proteins."
        )
    return lines


def main() -> None:
    metric_frames = []
    for variant in VARIANTS:
        for metric in METRICS:
            metric_frames.append(load_variant_metric(variant, metric))

    raw_metrics = pd.concat(metric_frames, ignore_index=True)
    all_metrics = with_all_split(raw_metrics)

    layer47 = layer47_statistics(all_metrics)
    thresholds = tm_thresholds(all_metrics)
    negative = negative_r2_summary(all_metrics)
    extras = extra_paper_stats(all_metrics)

    layer47 = sort_for_output(layer47)
    thresholds = sort_for_output(thresholds)
    negative = sort_for_output(negative)
    extras = sort_for_output(extras)

    write_csv(layer47, "publication_layer47_stats.csv")
    write_csv(thresholds, "publication_tm_thresholds.csv")
    write_csv(negative, "publication_negative_r2_summary.csv")
    write_csv(extras, "publication_extra_paper_stats.csv")

    report = make_markdown(layer47, thresholds, negative, extras)
    (OUT_DIR / "publication_summary_statistics.md").write_text(report, encoding="utf-8")


if __name__ == "__main__":
    main()

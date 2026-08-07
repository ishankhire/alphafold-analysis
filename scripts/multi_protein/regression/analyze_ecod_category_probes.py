#!/usr/bin/env python3
"""Compare combined base256 probe behaviour across broad ECOD categories."""

from __future__ import annotations

import os
from itertools import combinations_with_replacement
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(ROOT_DIR / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(ROOT_DIR / ".cache"))

import matplotlib
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.spatial.distance import squareform
from scipy.stats import f as f_distribution

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch


DELTA_DIR = ROOT_DIR / "delta_files"
GENERAL_DIR = DELTA_DIR / "general_base256_combined"
PER_PROTEIN_DIR = DELTA_DIR / "per_protein_base256_combined"
OUT_DIR = DELTA_DIR / "ecod_category_analysis"
CATEGORY_PATH = ROOT_DIR / "ecod_protein_categories.csv"

LAYERS = tuple(range(48))
CATEGORIES = ("alpha", "beta", "alpha_beta", "alpha_plus_beta")
CATEGORY_COLORS = {
    "alpha": "#4c78a8",
    "beta": "#f58518",
    "alpha_beta": "#54a24b",
    "alpha_plus_beta": "#e45756",
}
N_FEATURES = 256
N_PERMUTATIONS = 2_000
RANDOM_SEED = 20260807


def require_columns(path: Path, frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def require_layers(name: str, frame: pd.DataFrame) -> None:
    observed = tuple(sorted(int(value) for value in frame["layer"].unique()))
    if observed != LAYERS:
        raise ValueError(f"{name} must contain layers 0-47, found {observed}")


def configure_style() -> None:
    plt.rcParams.update({
        "font.family": "DejaVu Sans", "font.size": 11, "axes.titlesize": 14,
        "axes.labelsize": 12, "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 250, "savefig.bbox": "tight",
    })


def load_categories() -> pd.DataFrame:
    categories = pd.read_csv(CATEGORY_PATH, dtype={"prot": str})
    require_columns(CATEGORY_PATH, categories, {"prot", "category", "length"})
    categories["protein_id"] = categories["prot"].str.zfill(9)
    categories = categories[["protein_id", "category", "length", "primary_arch"]].copy()
    if tuple(categories["category"].value_counts().reindex(CATEGORIES).fillna(0).astype(int)) != (100, 100, 100, 100):
        raise ValueError("ECOD category input must contain 100 proteins in each broad category")
    if categories["protein_id"].duplicated().any():
        raise ValueError("ECOD category input has duplicate protein IDs")
    return categories


def ecod_join(frame: pd.DataFrame, categories: pd.DataFrame, name: str) -> pd.DataFrame:
    joined = frame.merge(categories, on="protein_id", how="inner", validate="many_to_one")
    require_layers(name, joined)
    counts = joined.groupby("category")["protein_id"].nunique().reindex(CATEGORIES).fillna(0).astype(int)
    if tuple(counts) != (100, 99, 99, 100):
        raise ValueError(f"{name} category coverage was {counts.to_dict()}, expected 100/99/99/100")
    return joined


def load_metric_inputs(categories: pd.DataFrame) -> tuple[dict[tuple[str, str], pd.DataFrame], pd.DataFrame]:
    general_per = pd.read_csv(GENERAL_DIR / "per_protein.csv", dtype={"protein_id": str})
    general_geo = pd.read_csv(GENERAL_DIR / "geometry_metrics.csv", dtype={"protein_id": str})
    per_per = pd.read_csv(PER_PROTEIN_DIR / "per_protein.csv", dtype={"protein_id": str})
    per_geo = pd.read_csv(PER_PROTEIN_DIR / "geometry_metrics.csv", dtype={"protein_id": str})
    for path, frame, columns in [
        (GENERAL_DIR / "per_protein.csv", general_per, {"layer", "split", "protein_id", "n_residues", "per_protein_r2"}),
        (GENERAL_DIR / "geometry_metrics.csv", general_geo, {"layer", "split", "protein_id", "n_residues", "tm_score"}),
        (PER_PROTEIN_DIR / "per_protein.csv", per_per, {"layer", "source", "protein_id", "n_residues", "per_protein_r2"}),
        (PER_PROTEIN_DIR / "geometry_metrics.csv", per_geo, {"layer", "source", "protein_id", "n_residues", "tm_score"}),
    ]:
        require_columns(path, frame, columns)
        require_layers(str(path), frame)

    per_per = per_per[per_per["source"].eq("ecod")].copy()
    per_geo = per_geo[per_geo["source"].eq("ecod")].copy()
    inputs = {
        ("shared", "r2"): ecod_join(general_per, categories, "shared R2"),
        ("shared", "tm_score"): ecod_join(general_geo, categories, "shared TM-score"),
        ("per_protein", "r2"): ecod_join(per_per, categories, "per-protein R2"),
        ("per_protein", "tm_score"): ecod_join(per_geo, categories, "per-protein TM-score"),
    }
    metadata = categories[categories["protein_id"].isin(inputs[("per_protein", "r2")]["protein_id"].unique())].copy()
    return inputs, metadata


def metric_summary(inputs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[pd.DataFrame] = []
    for (probe_type, metric), frame in inputs.items():
        value_column = "per_protein_r2" if metric == "r2" else "tm_score"
        grouped = frame.groupby(["layer", "category"], sort=True)[value_column]
        summary = grouped.agg(["count", "mean", "median", "std", "sem"]).reset_index()
        summary = summary.rename(columns={"count": "n_proteins", "std": "standard_deviation"})
        summary["probe_type"] = probe_type
        summary["evaluation_scope"] = "pooled_train_test" if probe_type == "shared" else "ecod_all_pairs"
        summary["metric"] = metric
        rows.append(summary[["probe_type", "evaluation_scope", "metric", "layer", "category", "n_proteins", "mean", "median", "standard_deviation", "sem"]])
    return pd.concat(rows, ignore_index=True).sort_values(["probe_type", "metric", "category", "layer"])


def plot_metric(summary: pd.DataFrame, probe_type: str, metric: str) -> Path:
    subset = summary[(summary["probe_type"] == probe_type) & (summary["metric"] == metric)]
    label = "R$^2$" if metric == "r2" else "TM-score"
    scope = "pooled train + test" if probe_type == "shared" else "ECOD per-protein fits (all pairs)"
    figure, axis = plt.subplots(figsize=(9, 5.4))
    for category in CATEGORIES:
        group = subset[subset["category"] == category].sort_values("layer")
        values = group["mean"].to_numpy(float)
        errors = 1.96 * group["sem"].to_numpy(float)
        axis.plot(group["layer"], values, label=category, color=CATEGORY_COLORS[category], linewidth=2.2)
        axis.fill_between(group["layer"], values - errors, values + errors, color=CATEGORY_COLORS[category], alpha=0.17)
    axis.set(xlabel="AlphaFold layer", ylabel=f"Mean {label}", xlim=(0, 47), title=f"{probe_type.replace('_', '-')} ECOD categories: {label} ({scope})")
    axis.grid(axis="y", color="#dddddd", linewidth=0.8)
    axis.legend(title="ECOD category", ncol=2)
    path = OUT_DIR / f"{probe_type}_ecod_category_{metric}_vs_layer.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def load_coefficient_tensor(categories: pd.DataFrame, protein_ids: list[str]) -> np.ndarray:
    path = PER_PROTEIN_DIR / "coefficients.csv"
    index = {protein_id: position for position, protein_id in enumerate(protein_ids)}
    tensor = np.full((len(LAYERS), len(protein_ids), N_FEATURES), np.nan, dtype=np.float64)
    columns = ["layer", "source", "protein_id", "feature_index", "coefficient_standardized"]
    for chunk in pd.read_csv(path, usecols=columns, dtype={"protein_id": str}, chunksize=250_000):
        chunk = chunk[(chunk["source"] == "ecod") & (chunk["feature_index"] >= 0)]
        chunk = chunk[chunk["protein_id"].isin(index)]
        if chunk.empty:
            continue
        positions = chunk["protein_id"].map(index).to_numpy(int)
        tensor[chunk["layer"].to_numpy(int), positions, chunk["feature_index"].to_numpy(int)] = chunk["coefficient_standardized"].to_numpy(float)
    if not np.isfinite(tensor).all():
        missing = int(np.size(tensor) - np.isfinite(tensor).sum())
        raise ValueError(f"Coefficient tensor is incomplete or non-finite ({missing} values)")
    return tensor


def pair_summary(similarity: np.ndarray, labels: np.ndarray, layer: int) -> tuple[list[dict[str, object]], float, float]:
    rows: list[dict[str, object]] = []
    within_values: list[np.ndarray] = []
    between_values: list[np.ndarray] = []
    for first_index, second_index in combinations_with_replacement(range(len(CATEGORIES)), 2):
        first, second = CATEGORIES[first_index], CATEGORIES[second_index]
        a = np.flatnonzero(labels == first)
        b = np.flatnonzero(labels == second)
        if first == second:
            values = similarity[np.ix_(a, a)][np.triu_indices(len(a), k=1)]
            within_values.append(values)
            relation = "within"
        else:
            values = similarity[np.ix_(a, b)].ravel()
            between_values.append(values)
            relation = "between"
        rows.append({"layer": layer, "category_a": first, "category_b": second, "relation": relation,
                     "n_protein_pairs": len(values), "mean_cosine_similarity": float(values.mean()),
                     "median_cosine_similarity": float(np.median(values)), "standard_deviation": float(values.std(ddof=0))})
    return rows, float(np.concatenate(within_values).mean()), float(np.concatenate(between_values).mean())


def coefficient_analysis(metadata: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, object], np.ndarray, list[str]]:
    metadata = metadata.sort_values("protein_id").reset_index(drop=True)
    protein_ids = metadata["protein_id"].tolist()
    labels = metadata["category"].to_numpy()
    tensor = load_coefficient_tensor(metadata, protein_ids)
    category_rows: list[dict[str, object]] = []
    contrast_rows: list[dict[str, object]] = []
    mean_similarity = np.zeros((len(protein_ids), len(protein_ids)), dtype=np.float64)
    layer_similarities: list[np.ndarray] = []
    for layer in LAYERS:
        vectors = tensor[layer]
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        if np.any(norms == 0):
            raise ValueError(f"Layer {layer} contains a zero coefficient vector")
        normalized = vectors / norms
        similarity = normalized @ normalized.T
        layer_similarities.append(similarity)
        mean_similarity += similarity / len(LAYERS)
        rows, within, between = pair_summary(similarity, labels, layer)
        category_rows.extend(rows)
        contrast_rows.append({"layer": layer, "within_category_mean_cosine": within, "between_category_mean_cosine": between, "within_minus_between": within - between})
    contrasts = pd.DataFrame(contrast_rows)
    selected_layer = int(contrasts.loc[contrasts["within_minus_between"].idxmax(), "layer"])
    details = {"selected_layer": selected_layer, "labels": labels, "mean_similarity": mean_similarity,
               "selected_similarity": layer_similarities[selected_layer]}
    return pd.DataFrame(category_rows), contrasts, details, tensor, protein_ids


def plot_cosine_outputs(category_pairs: pd.DataFrame, contrasts: pd.DataFrame, details: dict[str, object]) -> tuple[Path, Path]:
    figure, axis = plt.subplots(figsize=(9, 5.4))
    for column, color, label in [("within_category_mean_cosine", "#4c78a8", "within category"), ("between_category_mean_cosine", "#999999", "between categories"), ("within_minus_between", "#e45756", "within − between")]:
        axis.plot(contrasts["layer"], contrasts[column], color=color, linewidth=2, label=label)
    axis.set(xlabel="AlphaFold layer", ylabel="Mean cosine similarity", xlim=(0, 47), title="ECOD per-protein coefficient similarity")
    axis.grid(axis="y", color="#dddddd", linewidth=0.8)
    axis.legend()
    contrast_path = OUT_DIR / "coefficient_cosine_within_between_vs_layer.png"
    figure.savefig(contrast_path)
    plt.close(figure)

    layer = int(details["selected_layer"])
    matrix = np.full((len(CATEGORIES), len(CATEGORIES)), np.nan)
    subset = category_pairs[category_pairs["layer"] == layer]
    for _, row in subset.iterrows():
        i, j = CATEGORIES.index(row["category_a"]), CATEGORIES.index(row["category_b"])
        matrix[i, j] = matrix[j, i] = row["mean_cosine_similarity"]
    figure, axis = plt.subplots(figsize=(6.4, 5.5))
    image = axis.imshow(matrix, cmap="coolwarm", vmin=-1, vmax=1)
    axis.set(xticks=range(4), yticks=range(4), xticklabels=CATEGORIES, yticklabels=CATEGORIES,
             title=f"Mean coefficient cosine similarity (layer {layer})")
    plt.setp(axis.get_xticklabels(), rotation=25, ha="right")
    for i in range(4):
        for j in range(4):
            axis.text(j, i, f"{matrix[i, j]:.2f}", ha="center", va="center", fontsize=10)
    figure.colorbar(image, ax=axis, label="Mean cosine similarity")
    heatmap_path = OUT_DIR / f"coefficient_cosine_category_heatmap_layer{layer:02d}.png"
    figure.savefig(heatmap_path)
    plt.close(figure)
    return contrast_path, heatmap_path


def plot_dendrogram(details: dict[str, object], metadata: pd.DataFrame) -> Path:
    layer = int(details["selected_layer"])
    similarity = np.asarray(details["selected_similarity"])
    distance = np.clip(1.0 - similarity, 0.0, 2.0)
    np.fill_diagonal(distance, 0.0)
    hierarchy = linkage(squareform(distance, checks=False), method="average")
    figure, axis = plt.subplots(figsize=(16, 8))
    tree = dendrogram(hierarchy, no_labels=True, color_threshold=0, above_threshold_color="#555555", ax=axis)
    leaves = tree["leaves"]
    labels = np.asarray(details["labels"])
    maximum = float(np.max(hierarchy[:, 2]))
    tile_y = np.full(len(leaves), -0.035 * maximum if maximum else -0.035)
    axis.scatter(np.arange(len(leaves)) * 10 + 5, tile_y, marker="s", s=12,
                 c=[CATEGORY_COLORS[labels[index]] for index in leaves], clip_on=False)
    axis.set_ylim(bottom=float(tile_y[0] * 2), top=None)
    axis.set(title=f"ECOD protein coefficient clustering (layer {layer})", xlabel="Proteins (leaf order; color = ECOD category)", ylabel="Cosine distance")
    axis.legend(handles=[Patch(color=CATEGORY_COLORS[category], label=category) for category in CATEGORIES],
                title="ECOD category", loc="upper left", ncol=2)
    path = OUT_DIR / f"coefficient_dendrogram_layer{layer:02d}.png"
    figure.savefig(path)
    plt.close(figure)
    return path


def permutation_test(details: dict[str, object]) -> pd.DataFrame:
    similarity = np.asarray(details["mean_similarity"])
    labels = np.asarray(details["labels"])
    upper = np.triu_indices(len(labels), k=1)
    values = similarity[upper]
    first, second = labels[upper[0]], labels[upper[1]]
    observed = float(values[first == second].mean() - values[first != second].mean())
    rng = np.random.default_rng(RANDOM_SEED)
    null = np.empty(N_PERMUTATIONS, dtype=float)
    for number in range(N_PERMUTATIONS):
        shuffled = rng.permutation(labels)
        same = shuffled[upper[0]] == shuffled[upper[1]]
        null[number] = values[same].mean() - values[~same].mean()
    p_value = (1 + int(np.count_nonzero(null >= observed))) / (N_PERMUTATIONS + 1)
    return pd.DataFrame([{"statistic": "mean_across_layer_within_minus_between_cosine", "observed_value": observed,
                          "null_mean": float(null.mean()), "null_standard_deviation": float(null.std(ddof=1)),
                          "permutation_p_value_greater": p_value, "n_permutations": N_PERMUTATIONS,
                          "random_seed": RANDOM_SEED}])


def length_adjusted_effects(inputs: dict[tuple[str, str], pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (probe_type, metric), frame in inputs.items():
        value_name = "per_protein_r2" if metric == "r2" else "tm_score"
        layer_means = frame.groupby("layer")[value_name].mean()
        layer = int(layer_means.idxmax())
        data = frame[frame["layer"] == layer].copy()
        y = data[value_name].to_numpy(float)
        length = np.log1p(data["n_residues"].to_numpy(float))
        x_base = np.column_stack([np.ones(len(data)), length])
        dummy = np.column_stack([(data["category"] == category).to_numpy(float) for category in CATEGORIES[1:]])
        x_full = np.column_stack([x_base, dummy])
        beta_base = np.linalg.lstsq(x_base, y, rcond=None)[0]
        beta_full = np.linalg.lstsq(x_full, y, rcond=None)[0]
        rss_base = float(np.sum((y - x_base @ beta_base) ** 2))
        rss_full = float(np.sum((y - x_full @ beta_full) ** 2))
        added_df, residual_df = dummy.shape[1], len(data) - x_full.shape[1]
        f_statistic = ((rss_base - rss_full) / added_df) / (rss_full / residual_df)
        p_value = float(f_distribution.sf(f_statistic, added_df, residual_df))
        rows.append({"probe_type": probe_type, "evaluation_scope": "pooled_train_test" if probe_type == "shared" else "ecod_all_pairs",
                     "metric": metric, "selected_layer": layer, "n_proteins": len(data), "length_transform": "log1p(n_residues)",
                     "category_reference": "alpha", "category_effect_f_statistic": f_statistic, "numerator_df": added_df,
                     "denominator_df": residual_df, "category_effect_p_value": p_value,
                     "partial_r2": (rss_base - rss_full) / rss_base, "beta_effect": beta_full[2],
                     "alpha_beta_effect": beta_full[3], "alpha_plus_beta_effect": beta_full[4]})
    return pd.DataFrame(rows)


def write_report(summary: pd.DataFrame, contrasts: pd.DataFrame, permutation: pd.DataFrame, effects: pd.DataFrame) -> Path:
    best_lines = []
    for (probe, metric), group in summary.groupby(["probe_type", "metric"]):
        winner = group.loc[group["mean"].idxmax()]
        best_lines.append(f"- {probe} {metric}: {winner['category']} at layer {int(winner['layer'])} ({winner['mean']:.4f})")
    selected_layer = int(contrasts.loc[contrasts["within_minus_between"].idxmax(), "layer"])
    report = "# ECOD category analysis\n\n"
    report += "Uses 398 annotated ECOD proteins. The category file contains 400 entries; `000196893` (beta) and `001681437` (alpha_beta) are absent from both probe outputs. Shared-probe results pool saved train and test rows.\n\n"
    report += "## Best category-layer means\n\n" + "\n".join(best_lines) + "\n\n"
    report += f"## Coefficients\n\nThe largest observed within-minus-between cosine gap occurs at layer {selected_layer}. The global across-layer permutation p-value is {permutation.iloc[0]['permutation_p_value_greater']:.4g}.\n\n"
    effect_lines = []
    for _, row in effects.iterrows():
        effect_lines.append(
            f"- {row['probe_type']} {row['metric']} (layer {int(row['selected_layer'])}): "
            f"partial R² {row['partial_r2']:.4f}, category-effect p = {row['category_effect_p_value']:.4g}."
        )
    report += "## Length-adjusted category effects\n\n" + "\n".join(effect_lines) + "\n"
    path = OUT_DIR / "analysis_report.md"
    path.write_text(report)
    return path


def main() -> None:
    configure_style()
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    categories = load_categories()
    inputs, metadata = load_metric_inputs(categories)
    summary = metric_summary(inputs)
    summary.to_csv(OUT_DIR / "category_metric_layer_summary.csv", index=False)
    metadata.to_csv(OUT_DIR / "category_metadata.csv", index=False)
    metadata.groupby("category")["length"].agg(["count", "mean", "median", "std", "min", "max"]).reset_index().to_csv(OUT_DIR / "category_length_summary.csv", index=False)
    for probe_type, metric in inputs:
        plot_metric(summary, probe_type, metric)
    pair_rows, contrasts, details, _, _ = coefficient_analysis(metadata)
    pair_rows.to_csv(OUT_DIR / "coefficient_cosine_category_pairs_by_layer.csv", index=False)
    contrasts.to_csv(OUT_DIR / "coefficient_cosine_within_between_by_layer.csv", index=False)
    plot_cosine_outputs(pair_rows, contrasts, details)
    plot_dendrogram(details, metadata)
    permutation = permutation_test(details)
    permutation.to_csv(OUT_DIR / "coefficient_similarity_permutation_test.csv", index=False)
    effects = length_adjusted_effects(inputs)
    effects.to_csv(OUT_DIR / "length_adjusted_category_effects.csv", index=False)
    report = write_report(summary, contrasts, permutation, effects)
    print(f"Saved ECOD category analysis -> {OUT_DIR}")
    print(f"Report -> {report}")


if __name__ == "__main__":
    main()

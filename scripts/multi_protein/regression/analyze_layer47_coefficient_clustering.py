#!/usr/bin/env python3
"""Cluster per-protein layer-47 base256 regression coefficients.

The primary analysis L2-normalizes each protein's standardized coefficient
vector, so Euclidean k-means emphasizes coefficient direction. An unchanged
standardized-coefficient analysis is retained as a magnitude-sensitive check.
"""

from __future__ import annotations

import os
import hashlib
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[3]
os.environ.setdefault("MPLCONFIGDIR", str(PROJECT_ROOT / ".mplconfig"))
os.environ.setdefault("XDG_CACHE_HOME", str(PROJECT_ROOT / ".cache"))
os.environ.setdefault("LOKY_MAX_CPU_COUNT", "1")

import matplotlib
import numpy as np
import pandas as pd
from scipy.stats import chi2_contingency, kruskal
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score, silhouette_score

matplotlib.use("Agg")
import matplotlib.pyplot as plt


LAYER = 47
N_FEATURES = 256
K_VALUES = tuple(range(1, 11))
RANDOM_SEED = 20260807
FULL_N_INIT = 50
STABILITY_N_INIT = 20
N_STABILITY_REPEATS = 25
STABILITY_SUBSAMPLE_FRACTION = 0.80

INPUT_DIR = PROJECT_ROOT / "delta_files" / "per_protein_base256_combined"
COEFFICIENT_PATH = INPUT_DIR / "model_artifacts" / "coefficient_index.csv"
PER_PROTEIN_PATH = INPUT_DIR / "per_protein.csv"
CATEGORY_PATH = PROJECT_ROOT / "ecod_protein_categories.csv"
OUTPUT_DIR = INPUT_DIR / "layer47_coefficient_clustering"

REPRESENTATION_LABELS = {
    "direction": "Direction (unit-normalized)",
    "magnitude_sensitive": "Magnitude-sensitive",
}
ECOD_CATEGORIES = ("alpha", "beta", "alpha_beta", "alpha_plus_beta")
CLUSTER_COLORS = tuple(plt.get_cmap("tab10").colors)
SOURCE_COLORS = {"cameo": "#4c78a8", "ecod": "#f58518"}
ECOD_CATEGORY_COLORS = {
    "alpha": "#4c78a8",
    "beta": "#f58518",
    "alpha_beta": "#54a24b",
    "alpha_plus_beta": "#e45756",
}


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 12,
            "axes.labelsize": 11,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
        }
    )


def require_columns(path: Path, frame: pd.DataFrame, columns: set[str]) -> None:
    missing = sorted(columns.difference(frame.columns))
    if missing:
        raise ValueError(f"{path} is missing required columns: {missing}")


def load_categories(path: Path) -> pd.DataFrame:
    if not path.exists():
        return pd.DataFrame(columns=["protein_id", "ecod_category"])
    categories = pd.read_csv(path, dtype={"prot": str})
    require_columns(path, categories, {"prot", "category"})
    categories = categories[["prot", "category"]].copy()
    categories["protein_id"] = categories["prot"].str.zfill(9)
    categories = categories.rename(columns={"category": "ecod_category"})
    categories = categories[["protein_id", "ecod_category"]]
    if categories["protein_id"].duplicated().any():
        raise ValueError(f"{path} contains duplicate protein IDs")
    return categories


def load_layer47_coefficients(
    coefficient_path: Path = COEFFICIENT_PATH,
    per_protein_path: Path = PER_PROTEIN_PATH,
    category_path: Path = CATEGORY_PATH,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Load a validated protein-by-feature layer-47 coefficient matrix."""
    metadata = pd.read_csv(per_protein_path, dtype={"protein_id": str})
    require_columns(
        per_protein_path,
        metadata,
        {"layer", "protein_id", "source", "n_residues"},
    )
    if "evaluation_scope" in metadata:
        metadata = metadata.loc[metadata["evaluation_scope"].eq("all_pairs")].copy()
    metadata = metadata.loc[metadata["layer"].eq(LAYER)].copy()
    if metadata.empty:
        raise ValueError(f"{per_protein_path} contains no layer-{LAYER} rows")
    if metadata["protein_id"].duplicated().any():
        raise ValueError(f"{per_protein_path} has duplicate layer-{LAYER} protein rows")

    if "size_bin" not in metadata:
        cluster_dir = PROJECT_ROOT / "scripts" / "cluster_scripts"
        import sys
        if str(cluster_dir) not in sys.path:
            sys.path.insert(0, str(cluster_dir))
        from regression_protocol import load_master_manifest
        size_bins = {
            entry.protein_id: entry.size_bin
            for entry in load_master_manifest()
            if entry.eligible
        }
        metadata["size_bin"] = metadata["protein_id"].map(size_bins)
        if metadata["size_bin"].isna().any():
            raise ValueError("Canonical metadata contains proteins absent from the master manifest")

    header = pd.read_csv(coefficient_path, nrows=0).columns
    if "feature_index" not in header:
        index = pd.read_csv(coefficient_path, dtype={"protein_id": str})
        require_columns(
            coefficient_path,
            index,
            {"layer", "protein_id", "shard_path", "array_row", "fit_status", "file_sha256"},
        )
        index = index.loc[index["layer"].eq(LAYER)].copy()
        if index.empty or index["protein_id"].duplicated().any():
            raise ValueError(f"{coefficient_path} lacks a unique layer-{LAYER} coefficient index")
        shard_paths = index["shard_path"].unique()
        digests = index["file_sha256"].unique()
        if len(shard_paths) != 1 or len(digests) != 1:
            raise ValueError("Layer-47 coefficient index must reference one consistently hashed shard")
        result_root = coefficient_path.parents[1]
        shard_path = result_root / str(shard_paths[0])
        observed_digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
        if observed_digest != str(digests[0]):
            raise ValueError(f"Coefficient shard digest mismatch for {shard_path}")
        with np.load(shard_path) as archive:
            coefficients_array = np.asarray(archive["coefficient_standardized"], dtype=np.float64)
            archive_ids = np.asarray(archive["protein_id"]).astype(str)
            feature_names = np.asarray(archive["feature_names"]).astype(str).tolist()
        rows = index["array_row"].astype(int).to_numpy()
        if np.any(rows < 0) or np.any(rows >= len(archive_ids)):
            raise ValueError("Coefficient index contains out-of-range array rows")
        if not np.array_equal(archive_ids[rows], index["protein_id"].astype(str).to_numpy()):
            raise ValueError("Coefficient index protein IDs do not match the NPZ shard")
        matrix = coefficients_array[rows]
        coefficient_ids = index["protein_id"].astype(str).tolist()
        wide = pd.DataFrame(matrix, index=coefficient_ids).sort_index()
        wide.index.name = "protein_id"
        matrix = wide.to_numpy(dtype=np.float64)
    else:
        matrix, wide, feature_names = _load_legacy_coefficient_csv(coefficient_path)

    metadata_ids = set(metadata["protein_id"])
    coefficient_ids = set(wide.index.astype(str))
    if metadata_ids != coefficient_ids:
        missing_coefficients = sorted(metadata_ids.difference(coefficient_ids))
        missing_metadata = sorted(coefficient_ids.difference(metadata_ids))
        raise ValueError(
            "Layer-47 protein sets differ between inputs: "
            f"missing coefficients={missing_coefficients[:5]}, "
            f"missing metadata={missing_metadata[:5]}"
        )
    if matrix.shape != (len(metadata), N_FEATURES) or len(feature_names) != N_FEATURES:
        raise ValueError(f"Unexpected layer-47 coefficient matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Layer-47 coefficient matrix contains non-finite values")
    metadata = metadata.set_index("protein_id").loc[wide.index].reset_index()
    metadata = metadata.merge(load_categories(category_path), on="protein_id", how="left")
    return matrix, metadata, feature_names


def _load_legacy_coefficient_csv(
    coefficient_path: Path,
) -> tuple[np.ndarray, pd.DataFrame, list[str]]:
    """Retain read compatibility for already-copied historical coefficient CSVs."""
    columns = [
        "layer", "protein_id", "feature_index", "feature_name", "coefficient_standardized",
    ]
    chunks: list[pd.DataFrame] = []
    for chunk in pd.read_csv(
        coefficient_path,
        usecols=columns,
        dtype={"protein_id": str},
        chunksize=500_000,
    ):
        selected = chunk.loc[
            chunk["layer"].eq(LAYER) & chunk["feature_index"].ge(0)
        ].copy()
        if not selected.empty:
            chunks.append(selected)
    if not chunks:
        raise ValueError(f"{coefficient_path} contains no layer-{LAYER} coefficients")
    coefficients = pd.concat(chunks, ignore_index=True)

    if coefficients.duplicated(["protein_id", "feature_index"]).any():
        raise ValueError(
            f"{coefficient_path} has duplicate layer-{LAYER} protein-feature rows"
        )
    observed_features = tuple(sorted(int(value) for value in coefficients["feature_index"].unique()))
    if observed_features != tuple(range(N_FEATURES)):
        raise ValueError(
            f"{coefficient_path} expected feature indices 0-{N_FEATURES - 1}, "
            f"found {observed_features}"
        )
    counts = coefficients.groupby("protein_id")["feature_index"].size()
    invalid_counts = counts[counts.ne(N_FEATURES)]
    if not invalid_counts.empty:
        raise ValueError(
            f"{coefficient_path} has incomplete layer-{LAYER} vectors for "
            f"{invalid_counts.index.tolist()[:5]}"
        )

    feature_names_by_index = (
        coefficients[["feature_index", "feature_name"]]
        .drop_duplicates()
        .sort_values("feature_index")
    )
    if feature_names_by_index["feature_index"].duplicated().any() or len(
        feature_names_by_index
    ) != N_FEATURES:
        raise ValueError("Feature indices do not map one-to-one to feature names")
    feature_names = feature_names_by_index["feature_name"].tolist()

    wide = coefficients.pivot(
        index="protein_id",
        columns="feature_index",
        values="coefficient_standardized",
    ).sort_index()
    matrix = wide.to_numpy(dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != N_FEATURES:
        raise ValueError(f"Unexpected layer-47 coefficient matrix shape: {matrix.shape}")
    if not np.isfinite(matrix).all():
        raise ValueError("Layer-47 coefficient matrix contains non-finite values")

    return matrix, wide, feature_names


def build_representations(matrix: np.ndarray) -> dict[str, np.ndarray]:
    matrix = np.asarray(matrix, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[1] != N_FEATURES:
        raise ValueError(f"Expected an n-by-{N_FEATURES} coefficient matrix")
    norms = np.linalg.norm(matrix, axis=1)
    if np.any(~np.isfinite(norms)) or np.any(norms == 0):
        raise ValueError("Cannot normalize non-finite or zero coefficient vectors")
    return {
        "direction": matrix / norms[:, None],
        "magnitude_sensitive": matrix.copy(),
    }


def fit_kmeans_sweep(
    matrix: np.ndarray,
    representation: str,
    *,
    k_values: tuple[int, ...] = K_VALUES,
    random_seed: int = RANDOM_SEED,
    full_n_init: int = FULL_N_INIT,
    stability_n_init: int = STABILITY_N_INIT,
    n_stability_repeats: int = N_STABILITY_REPEATS,
    stability_fraction: float = STABILITY_SUBSAMPLE_FRACTION,
) -> tuple[pd.DataFrame, dict[int, KMeans], int]:
    """Fit the k sweep and return diagnostics, fitted models, and selected k."""
    matrix = np.asarray(matrix, dtype=np.float64)
    if not k_values or k_values[0] != 1 or any(k < 1 for k in k_values):
        raise ValueError("k_values must begin at 1 and contain positive integers")
    if max(k_values) >= len(matrix):
        raise ValueError("Every k must be smaller than the number of proteins")

    models: dict[int, KMeans] = {}
    diagnostics: list[dict[str, object]] = []
    rng = np.random.default_rng(random_seed + (0 if representation == "direction" else 1))
    n_subsample = max(2, int(np.ceil(stability_fraction * len(matrix))))

    for k in k_values:
        model = KMeans(
            n_clusters=k,
            n_init=full_n_init,
            random_state=random_seed,
            algorithm="lloyd",
        ).fit(matrix)
        models[k] = model
        counts = np.bincount(model.labels_, minlength=k)
        silhouette = np.nan
        stability_mean = np.nan
        stability_std = np.nan
        if k >= 2:
            silhouette = float(silhouette_score(matrix, model.labels_))
            stability_values: list[float] = []
            for repeat in range(n_stability_repeats):
                subset = np.sort(rng.choice(len(matrix), size=n_subsample, replace=False))
                subsample_model = KMeans(
                    n_clusters=k,
                    n_init=stability_n_init,
                    random_state=random_seed + 10_000 * k + repeat,
                    algorithm="lloyd",
                ).fit(matrix[subset])
                stability_values.append(
                    adjusted_rand_score(model.labels_, subsample_model.predict(matrix))
                )
            stability_mean = float(np.mean(stability_values))
            stability_std = float(np.std(stability_values, ddof=1))
        diagnostics.append(
            {
                "representation": representation,
                "layer": LAYER,
                "n_proteins": len(matrix),
                "n_features": matrix.shape[1],
                "k": k,
                "inertia": float(model.inertia_),
                "silhouette": silhouette,
                "stability_ari_mean": stability_mean,
                "stability_ari_std": stability_std,
                "min_cluster_size": int(counts.min()),
                "max_cluster_size": int(counts.max()),
            }
        )

    frame = pd.DataFrame(diagnostics)
    k1_inertia = float(frame.loc[frame["k"].eq(1), "inertia"].iloc[0])
    frame["inertia_fraction_of_k1"] = frame["inertia"] / k1_inertia
    eligible = frame.loc[frame["k"].ge(2)]
    selected_k = int(
        eligible.sort_values(["silhouette", "k"], ascending=[False, True]).iloc[0]["k"]
    )
    frame["selected_by_max_silhouette"] = frame["k"].eq(selected_k)
    return frame, models, selected_k


def relabel_by_cluster_size(
    labels: np.ndarray, centers: np.ndarray
) -> tuple[np.ndarray, np.ndarray, dict[int, int]]:
    """Map the largest cluster to zero, breaking equal-size ties by old ID."""
    labels = np.asarray(labels, dtype=int)
    counts = np.bincount(labels, minlength=len(centers))
    order = sorted(range(len(centers)), key=lambda old: (-int(counts[old]), old))
    mapping = {old: new for new, old in enumerate(order)}
    remapped = np.asarray([mapping[int(old)] for old in labels], dtype=int)
    return remapped, np.asarray(centers)[order], mapping


def categorical_association(
    assignments: pd.DataFrame,
    representation: str,
    variable: str,
    subset_name: str,
) -> dict[str, object]:
    data = assignments.loc[
        assignments["representation"].eq(representation), ["cluster_id", variable]
    ].dropna()
    table = pd.crosstab(data["cluster_id"], data[variable])
    if table.shape[0] < 2 or table.shape[1] < 2:
        statistic = p_value = effect = np.nan
    else:
        statistic, p_value, _, _ = chi2_contingency(table.to_numpy(), correction=False)
        denominator = len(data) * min(table.shape[0] - 1, table.shape[1] - 1)
        effect = float(np.sqrt(statistic / denominator))
    return {
        "representation": representation,
        "variable": variable,
        "evaluation_subset": subset_name,
        "n_proteins": len(data),
        "n_groups": table.shape[1],
        "test": "chi_square",
        "statistic": statistic,
        "p_value": p_value,
        "effect_name": "cramers_v",
        "effect_value": effect,
    }


def length_association(assignments: pd.DataFrame, representation: str) -> dict[str, object]:
    data = assignments.loc[assignments["representation"].eq(representation)].copy()
    values = np.log1p(data["n_residues"].to_numpy(dtype=np.float64))
    labels = data["cluster_id"].to_numpy(dtype=int)
    groups = [values[labels == cluster] for cluster in sorted(np.unique(labels))]
    statistic, p_value = kruskal(*groups)
    grand_mean = float(values.mean())
    total_sum_squares = float(np.sum((values - grand_mean) ** 2))
    between_sum_squares = float(
        sum(len(group) * (float(group.mean()) - grand_mean) ** 2 for group in groups)
    )
    eta_squared = between_sum_squares / total_sum_squares if total_sum_squares else np.nan
    return {
        "representation": representation,
        "variable": "log1p_n_residues",
        "evaluation_subset": "all_proteins",
        "n_proteins": len(data),
        "n_groups": len(groups),
        "test": "kruskal_wallis",
        "statistic": float(statistic),
        "p_value": float(p_value),
        "effect_name": "eta_squared",
        "effect_value": eta_squared,
    }


def build_selected_outputs(
    raw_coefficients: np.ndarray,
    metadata: pd.DataFrame,
    feature_names: list[str],
    representations: dict[str, np.ndarray],
    models: dict[str, dict[int, KMeans]],
    selected_ks: dict[str, int],
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, object]]:
    assignment_frames: list[pd.DataFrame] = []
    centroid_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    plot_details: dict[str, object] = {}
    raw_norms = np.linalg.norm(raw_coefficients, axis=1)

    for representation, matrix in representations.items():
        selected_k = selected_ks[representation]
        model = models[representation][selected_k]
        labels, centers, mapping = relabel_by_cluster_size(
            model.labels_, model.cluster_centers_
        )
        old_for_new = {new: old for old, new in mapping.items()}
        distances = model.transform(matrix)[np.arange(len(matrix)), model.labels_]
        pca = PCA(n_components=2, svd_solver="full").fit(matrix)
        coordinates = pca.transform(matrix)
        center_coordinates = pca.transform(
            np.asarray([model.cluster_centers_[old_for_new[new]] for new in range(selected_k)])
        )

        frame = metadata.copy()
        frame.insert(0, "representation", representation)
        frame.insert(1, "selected_k", selected_k)
        frame["cluster_id"] = labels
        frame["coefficient_l2_norm"] = raw_norms
        frame["distance_to_centroid"] = distances
        frame["pca_1"] = coordinates[:, 0]
        frame["pca_2"] = coordinates[:, 1]
        assignment_frames.append(frame)

        for cluster_id in range(selected_k):
            mask = labels == cluster_id
            mean_raw = np.mean(raw_coefficients[mask], axis=0, dtype=np.float64)
            for feature_index in range(N_FEATURES):
                centroid_rows.append(
                    {
                        "representation": representation,
                        "layer": LAYER,
                        "selected_k": selected_k,
                        "cluster_id": cluster_id,
                        "n_proteins": int(mask.sum()),
                        "feature_index": feature_index,
                        "feature_name": feature_names[feature_index],
                        "centroid_in_clustering_space": centers[cluster_id, feature_index],
                        "mean_standardized_coefficient": mean_raw[feature_index],
                    }
                )

            subset = frame.loc[mask]
            annotated = subset["ecod_category"].notna()
            row: dict[str, object] = {
                "representation": representation,
                "layer": LAYER,
                "selected_k": selected_k,
                "cluster_id": cluster_id,
                "n_proteins": len(subset),
                "protein_fraction": len(subset) / len(frame),
                "mean_coefficient_l2_norm": subset["coefficient_l2_norm"].mean(),
                "std_coefficient_l2_norm": subset["coefficient_l2_norm"].std(ddof=1),
                "mean_n_residues": subset["n_residues"].mean(),
                "median_n_residues": subset["n_residues"].median(),
                "n_cameo": int(subset["source"].eq("cameo").sum()),
                "cameo_fraction": subset["source"].eq("cameo").mean(),
                "n_ecod": int(subset["source"].eq("ecod").sum()),
                "ecod_fraction": subset["source"].eq("ecod").mean(),
                "n_ecod_with_category": int(annotated.sum()),
            }
            for size_bin in sorted(metadata["size_bin"].dropna().unique()):
                size_bin_int = int(size_bin)
                count = int(subset["size_bin"].eq(size_bin).sum())
                row[f"n_size_bin_{size_bin_int}"] = count
                row[f"size_bin_{size_bin_int}_fraction"] = count / len(subset)
            for category in ECOD_CATEGORIES:
                count = int(subset["ecod_category"].eq(category).sum())
                row[f"n_ecod_category_{category}"] = count
                row[f"ecod_category_{category}_fraction"] = (
                    count / int(annotated.sum()) if annotated.any() else np.nan
                )
            summary_rows.append(row)

        plot_details[representation] = {
            "labels": labels,
            "coordinates": coordinates,
            "center_coordinates": center_coordinates,
            "explained_variance_ratio": pca.explained_variance_ratio_,
            "selected_k": selected_k,
        }

    assignments = pd.concat(assignment_frames, ignore_index=True)
    centroids = pd.DataFrame(centroid_rows)
    summaries = pd.DataFrame(summary_rows)
    associations: list[dict[str, object]] = []
    for representation in representations:
        associations.append(length_association(assignments, representation))
        associations.append(
            categorical_association(assignments, representation, "source", "all_proteins")
        )
        associations.append(
            categorical_association(assignments, representation, "size_bin", "all_proteins")
        )
        associations.append(
            categorical_association(
                assignments, representation, "ecod_category", "annotated_ecod_only"
            )
        )
    return assignments, centroids, summaries, pd.DataFrame(associations), plot_details


def plot_diagnostics(diagnostics: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(14, 8.5), sharex="col")
    for row_index, representation in enumerate(REPRESENTATION_LABELS):
        subset = diagnostics.loc[diagnostics["representation"].eq(representation)]
        selected = subset.loc[subset["selected_by_max_silhouette"]].iloc[0]
        axes[row_index, 0].plot(subset["k"], subset["inertia_fraction_of_k1"], "o-", color="#4c78a8")
        axes[row_index, 0].set_ylabel("Inertia / k=1 inertia")
        eligible = subset.loc[subset["k"].ge(2)]
        axes[row_index, 1].plot(eligible["k"], eligible["silhouette"], "o-", color="#f58518")
        axes[row_index, 1].scatter([selected["k"]], [selected["silhouette"]], marker="*", s=150, color="#e45756", zorder=3)
        axes[row_index, 1].set_ylabel("Silhouette")
        axes[row_index, 2].errorbar(eligible["k"], eligible["stability_ari_mean"], yerr=eligible["stability_ari_std"], marker="o", color="#54a24b", capsize=3)
        axes[row_index, 2].set_ylabel("Subsample stability (ARI)")
        for axis in axes[row_index]:
            axis.grid(axis="y", color="#dddddd", linewidth=0.8)
            axis.set_xticks(K_VALUES)
        axes[row_index, 0].text(-0.24, 0.5, REPRESENTATION_LABELS[representation], transform=axes[row_index, 0].transAxes, rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
    for axis, title in zip(axes[0], ("Elbow diagnostic", "Separation", "Perturbation stability")):
        axis.set_title(title)
    for axis in axes[-1]:
        axis.set_xlabel("Number of clusters (k)")
    figure.suptitle("Layer-47 per-protein coefficient clustering diagnostics", fontsize=15)
    figure.savefig(path)
    plt.close(figure)


def plot_pca(plot_details: dict[str, object], path: Path) -> None:
    figure, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for axis, representation in zip(axes, REPRESENTATION_LABELS):
        details = plot_details[representation]
        labels = np.asarray(details["labels"])
        coordinates = np.asarray(details["coordinates"])
        centers = np.asarray(details["center_coordinates"])
        selected_k = int(details["selected_k"])
        for cluster_id in range(selected_k):
            mask = labels == cluster_id
            axis.scatter(coordinates[mask, 0], coordinates[mask, 1], s=20, alpha=0.72, color=CLUSTER_COLORS[cluster_id], label=f"Cluster {cluster_id} (n={mask.sum()})")
        axis.scatter(centers[:, 0], centers[:, 1], marker="X", s=150, color="black", edgecolor="white", linewidth=0.8, label="Centroids")
        variance = np.asarray(details["explained_variance_ratio"])
        axis.set_xlabel(f"PC1 ({100 * variance[0]:.1f}% variance)")
        axis.set_ylabel(f"PC2 ({100 * variance[1]:.1f}% variance)")
        axis.set_title(f"{REPRESENTATION_LABELS[representation]}: selected k={selected_k}")
        axis.grid(color="#eeeeee", linewidth=0.7)
        axis.legend(fontsize=8)
    figure.suptitle("Layer-47 coefficient profiles in two-dimensional PCA views", fontsize=15)
    figure.savefig(path)
    plt.close(figure)


def plot_cluster_profiles(centroids: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 1, figsize=(15, 6.5), constrained_layout=True)
    for axis, representation in zip(axes, REPRESENTATION_LABELS):
        subset = centroids.loc[centroids["representation"].eq(representation)]
        matrix = subset.pivot(index="cluster_id", columns="feature_index", values="mean_standardized_coefficient").sort_index().to_numpy()
        limit = float(np.max(np.abs(matrix)))
        image = axis.imshow(matrix, aspect="auto", interpolation="nearest", cmap="coolwarm", vmin=-limit, vmax=limit)
        axis.axvline(127.5, color="black", linewidth=1.1)
        axis.set_yticks(range(matrix.shape[0]), labels=[f"Cluster {value}" for value in range(matrix.shape[0])])
        axis.set_ylabel(REPRESENTATION_LABELS[representation])
        axis.set_xticks([0, 63, 127, 191, 255])
        figure.colorbar(image, ax=axis, label="Mean standardized coefficient", fraction=0.025, pad=0.015)
    axes[-1].set_xlabel("Feature index (pair_ij channels 0–127 | pair_ji channels 128–255)")
    figure.suptitle("Layer-47 selected-cluster coefficient profiles", fontsize=15)
    figure.savefig(path)
    plt.close(figure)


def plot_metadata(assignments: pd.DataFrame, path: Path) -> None:
    figure, axes = plt.subplots(2, 3, figsize=(15, 8.5))
    for row_index, representation in enumerate(REPRESENTATION_LABELS):
        subset = assignments.loc[assignments["representation"].eq(representation)]
        clusters = sorted(int(value) for value in subset["cluster_id"].unique())
        length_groups = [subset.loc[subset["cluster_id"].eq(cluster), "n_residues"].to_numpy() for cluster in clusters]
        axes[row_index, 0].boxplot(length_groups, tick_labels=[str(value) for value in clusters], showfliers=False)
        axes[row_index, 0].set_ylabel("Residues")
        axes[row_index, 0].set_xlabel("Cluster")

        source_table = pd.crosstab(subset["cluster_id"], subset["source"], normalize="index").reindex(index=clusters, columns=SOURCE_COLORS, fill_value=0)
        bottom = np.zeros(len(clusters))
        for source in SOURCE_COLORS:
            values = source_table[source].to_numpy()
            axes[row_index, 1].bar(clusters, values, bottom=bottom, label=source.upper(), color=SOURCE_COLORS[source])
            bottom += values
        axes[row_index, 1].set_ylim(0, 1)
        axes[row_index, 1].set_ylabel("Protein fraction")
        axes[row_index, 1].set_xlabel("Cluster")
        axes[row_index, 1].set_xticks(clusters)
        axes[row_index, 1].legend(fontsize=8)

        annotated = subset.dropna(subset=["ecod_category"])
        category_table = pd.crosstab(annotated["cluster_id"], annotated["ecod_category"], normalize="index").reindex(index=clusters, columns=ECOD_CATEGORIES, fill_value=0)
        bottom = np.zeros(len(clusters))
        for category in ECOD_CATEGORIES:
            values = category_table[category].to_numpy()
            axes[row_index, 2].bar(clusters, values, bottom=bottom, label=category, color=ECOD_CATEGORY_COLORS[category])
            bottom += values
        axes[row_index, 2].set_ylim(0, 1)
        axes[row_index, 2].set_ylabel("Annotated ECOD fraction")
        axes[row_index, 2].set_xlabel("Cluster")
        axes[row_index, 2].set_xticks(clusters)
        axes[row_index, 2].legend(fontsize=7, ncol=2)

        for axis in axes[row_index]:
            axis.grid(axis="y", color="#eeeeee", linewidth=0.7)
        axes[row_index, 0].text(-0.28, 0.5, REPRESENTATION_LABELS[representation], transform=axes[row_index, 0].transAxes, rotation=90, va="center", ha="center", fontsize=12, fontweight="bold")
    for axis, title in zip(axes[0], ("Protein length", "Dataset source", "Broad ECOD category")):
        axis.set_title(title)
    figure.suptitle("Layer-47 selected-cluster metadata diagnostics", fontsize=15)
    figure.savefig(path)
    plt.close(figure)


def write_report(
    diagnostics: pd.DataFrame,
    summaries: pd.DataFrame,
    associations: pd.DataFrame,
    assignments: pd.DataFrame,
    path: Path,
) -> None:
    lines = [
        "# Layer-47 coefficient clustering",
        "",
        f"Input: `{COEFFICIENT_PATH.relative_to(PROJECT_ROOT)}`",
        "",
        f"Loaded {assignments['protein_id'].nunique()} proteins with 256 finite standardized coefficients at layer 47. The intercept was excluded.",
        "",
        "## Method",
        "",
        "K-means was evaluated for k=1–10 directly in 256 coefficient dimensions. The primary direction representation L2-normalizes each protein vector; the magnitude-sensitive representation leaves the standardized vector unchanged. Candidate k is the maximum-silhouette solution over k=2–10. PCA is used only for display. Stability is adjusted Rand index against 25 fits trained on deterministic 80% protein subsamples.",
        "",
        "## Selected candidate solutions",
        "",
    ]
    for representation in REPRESENTATION_LABELS:
        selected = diagnostics.loc[
            diagnostics["representation"].eq(representation)
            & diagnostics["selected_by_max_silhouette"]
        ].iloc[0]
        cluster_sizes = summaries.loc[
            summaries["representation"].eq(representation), "n_proteins"
        ].astype(int).tolist()
        lines.append(
            f"- **{REPRESENTATION_LABELS[representation]}:** k={int(selected['k'])}; "
            f"silhouette={selected['silhouette']:.4f}; mean subsample ARI={selected['stability_ari_mean']:.4f}; "
            f"cluster sizes={cluster_sizes}."
        )

    lines.extend(["", "## Metadata associations", ""])
    for representation in REPRESENTATION_LABELS:
        subset = associations.loc[associations["representation"].eq(representation)]
        values = {row["variable"]: row for _, row in subset.iterrows()}
        lines.append(
            f"- **{REPRESENTATION_LABELS[representation]}:** log-length eta²="
            f"{values['log1p_n_residues']['effect_value']:.4f}; source Cramér's V="
            f"{values['source']['effect_value']:.4f}; size-bin Cramér's V="
            f"{values['size_bin']['effect_value']:.4f}; ECOD-category Cramér's V="
            f"{values['ecod_category']['effect_value']:.4f}."
        )

    lines.extend(
        [
            "",
            "## Interpretation",
            "",
            "The selected k values are visualization candidates, not proof that proteins form discrete biological classes. The inertia curve, silhouette, stability, cluster-size balance, and metadata associations should be read together. Strong length, size-bin, or CAMEO/ECOD-source associations indicate that a partition may reflect dataset composition or coefficient scale rather than a distinct structural category. ECOD-category tests use only proteins with available broad-category annotations. All association p-values are descriptive because the clusters and k values were learned from the same data.",
            "",
            "## Generated artifacts",
            "",
            "- `layer47_kmeans_k_diagnostics.csv`",
            "- `layer47_kmeans_assignments.csv`",
            "- `layer47_kmeans_centroids.csv`",
            "- `layer47_kmeans_cluster_summary.csv`",
            "- `layer47_kmeans_metadata_associations.csv`",
            "- `layer47_kmeans_diagnostics.png`",
            "- `layer47_kmeans_pca_clusters.png`",
            "- `layer47_kmeans_cluster_profiles.png`",
            "- `layer47_kmeans_metadata_diagnostics.png`",
        ]
    )
    path.write_text("\n".join(lines) + "\n")


def require_nonempty_outputs(paths: list[Path]) -> None:
    for path in paths:
        if not path.exists() or path.stat().st_size == 0:
            raise ValueError(f"Expected nonempty output: {path}")


def main() -> None:
    configure_style()
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    matrix, metadata, feature_names = load_layer47_coefficients()
    representations = build_representations(matrix)

    diagnostic_frames: list[pd.DataFrame] = []
    models: dict[str, dict[int, KMeans]] = {}
    selected_ks: dict[str, int] = {}
    for representation, values in representations.items():
        frame, fitted, selected_k = fit_kmeans_sweep(values, representation)
        diagnostic_frames.append(frame)
        models[representation] = fitted
        selected_ks[representation] = selected_k
        selected = frame.loc[frame["k"].eq(selected_k)].iloc[0]
        print(
            f"{REPRESENTATION_LABELS[representation]}: selected k={selected_k}, "
            f"silhouette={selected['silhouette']:.4f}, "
            f"stability ARI={selected['stability_ari_mean']:.4f}",
            flush=True,
        )
    diagnostics = pd.concat(diagnostic_frames, ignore_index=True)

    assignments, centroids, summaries, associations, plot_details = build_selected_outputs(
        matrix,
        metadata,
        feature_names,
        representations,
        models,
        selected_ks,
    )

    output_paths = {
        "diagnostics_csv": OUTPUT_DIR / "layer47_kmeans_k_diagnostics.csv",
        "assignments_csv": OUTPUT_DIR / "layer47_kmeans_assignments.csv",
        "centroids_csv": OUTPUT_DIR / "layer47_kmeans_centroids.csv",
        "summary_csv": OUTPUT_DIR / "layer47_kmeans_cluster_summary.csv",
        "associations_csv": OUTPUT_DIR / "layer47_kmeans_metadata_associations.csv",
        "diagnostics_png": OUTPUT_DIR / "layer47_kmeans_diagnostics.png",
        "pca_png": OUTPUT_DIR / "layer47_kmeans_pca_clusters.png",
        "profiles_png": OUTPUT_DIR / "layer47_kmeans_cluster_profiles.png",
        "metadata_png": OUTPUT_DIR / "layer47_kmeans_metadata_diagnostics.png",
        "report": OUTPUT_DIR / "layer47_kmeans_report.md",
    }
    diagnostics.to_csv(output_paths["diagnostics_csv"], index=False, float_format="%.10g")
    assignments.to_csv(output_paths["assignments_csv"], index=False, float_format="%.10g")
    centroids.to_csv(output_paths["centroids_csv"], index=False, float_format="%.10g")
    summaries.to_csv(output_paths["summary_csv"], index=False, float_format="%.10g")
    associations.to_csv(output_paths["associations_csv"], index=False, float_format="%.10g")
    plot_diagnostics(diagnostics, output_paths["diagnostics_png"])
    plot_pca(plot_details, output_paths["pca_png"])
    plot_cluster_profiles(centroids, output_paths["profiles_png"])
    plot_metadata(assignments, output_paths["metadata_png"])
    write_report(diagnostics, summaries, associations, assignments, output_paths["report"])
    require_nonempty_outputs(list(output_paths.values()))
    print(f"Saved layer-47 clustering outputs to {OUTPUT_DIR}", flush=True)


if __name__ == "__main__":
    main()

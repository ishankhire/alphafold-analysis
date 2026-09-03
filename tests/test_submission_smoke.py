from __future__ import annotations

import csv
import hashlib
import json
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
MODELS = {
    "shared_ols_base256",
    "per_protein_ols_base256",
    "sequence_gated_ols_base256",
    "shared_xgboost_base256_fixed_s4",
}
LAYERS = set(range(48))
RESULTS_ROOT = ROOT / "results"
SOURCE_DIR = RESULTS_ROOT / "publication_plots/source_data"
PLOT_ROOT = RESULTS_ROOT / "publication_plots"
METADATA_DIR = RESULTS_ROOT / "regression_5fold_v1/metadata"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


class SubmissionSmokeTest(unittest.TestCase):
    def test_frozen_cohort(self) -> None:
        rows = read_rows(METADATA_DIR / "regression_outer_folds_v1.csv")
        keys = {(row["source"], row["protein_id"]) for row in rows}
        self.assertEqual(len(rows), 552)
        self.assertEqual(len(keys), 552)
        self.assertEqual({int(row["outer_fold"]) for row in rows}, set(range(5)))

    def test_frozen_metadata_inventory(self) -> None:
        self.assertEqual(
            {path.name for path in METADATA_DIR.iterdir() if path.is_file()},
            {
                "fold_digests.json",
                "per_protein_pair_folds_v1.csv",
                "regression_outer_folds_v1.csv",
            },
        )
        with (METADATA_DIR / "fold_digests.json").open() as handle:
            digests = json.load(handle)
        self.assertEqual(
            set(digests),
            {
                "cohort_digest",
                "construction_parameters",
                "cv_protocol_version",
                "master_manifest_digest",
                "outer_fold_digest",
                "pair_fold_manifest_digest",
                "per_fold_length_bin_counts",
                "per_fold_protein_counts",
            },
        )

    def test_probe_source_tables(self) -> None:
        expected = {
            "r2_plot_summary.csv": (192, None),
            "tm_score_plot_summary.csv": (192, None),
            "spatial_distance_mae_plot_summary.csv": (
                768,
                {"0-5", "5-10", "10-40", "40+"},
            ),
            "triangle_plot_summary.csv": (192, None),
            "decodability_consistency_points.csv": (192, None),
        }
        for name, (row_count, categories) in expected.items():
            with self.subTest(name=name):
                rows = read_rows(SOURCE_DIR / name)
                self.assertEqual(len(rows), row_count)
                self.assertEqual({row["analysis_name"] for row in rows}, MODELS)
                self.assertEqual({int(row["layer"]) for row in rows}, LAYERS)
                protein_counts = {
                    int(row.get("n_proteins") or row["n_proteins_total"])
                    for row in rows
                }
                self.assertEqual(protein_counts, {552})
                if categories is not None:
                    self.assertEqual(
                        {row["category_label"] for row in rows}, categories
                    )

    def test_ablation_sources(self) -> None:
        ablation_dir = RESULTS_ROOT / "layerwise_ablation_csvs"
        paths = sorted(ablation_dir.glob("*.csv"))
        self.assertEqual(len(paths), 4)
        for path in paths:
            with self.subTest(name=path.name):
                rows = read_rows(path)
                self.assertEqual(len(rows), 554)
                self.assertTrue(all(f"L{layer}" in rows[0] for layer in LAYERS))

    def test_submission_figures(self) -> None:
        expected = {
            PLOT_ROOT / "main_figures/main_figure_01_mean_per_protein_r2.pdf",
            PLOT_ROOT / "main_figures/main_figure_03_tm_score.pdf",
            PLOT_ROOT
            / "main_figures/main_figure_15_spatial_distance_mae_combined.pdf",
            PLOT_ROOT
            / "main_figures/main_figure_all_triangles_decodability_consistency.pdf",
            PLOT_ROOT
            / "main_figures/main_figure_all_triangles_peak_troughs_overlay.pdf",
            PLOT_ROOT
            / "layerwise_ablation_trajectories/combined_layerwise_ablation_tm_score_trajectories.pdf",
        }
        actual = set(PLOT_ROOT.glob("main_figures/*.pdf")) | set(
            PLOT_ROOT.glob("layerwise_ablation_trajectories/*.pdf")
        )
        self.assertEqual(actual, expected)
        self.assertTrue(all(path.stat().st_size > 0 for path in actual))

    def test_figure_manifest(self) -> None:
        rows = read_rows(PLOT_ROOT / "figure_manifest.csv")
        self.assertEqual(len(rows), 5)
        for row in rows:
            with self.subTest(figure_id=row["figure_id"]):
                pdf_path = ROOT / row["pdf_path"]
                self.assertTrue(pdf_path.is_relative_to(PLOT_ROOT))
                self.assertEqual(sha256_file(pdf_path), row["pdf_sha256"])
                source_paths = [
                    ROOT / value for value in row["source_csvs"].split(";")
                ]
                source_hashes = row["source_sha256"].split(";")
                self.assertEqual(len(source_paths), len(source_hashes))
                self.assertTrue(
                    all(path.is_relative_to(SOURCE_DIR) for path in source_paths)
                )
                self.assertEqual(
                    [sha256_file(path) for path in source_paths], source_hashes
                )

    def test_xgboost_selection_provenance(self) -> None:
        source = (
            RESULTS_ROOT
            / "xgboost_hyperparameter_search/shared/winner_cap_sweep/"
            "cross_layer_selection_table.csv"
        )
        rows = read_rows(source)
        self.assertEqual(
            {row["config_id"] for row in rows}, {f"S{i}" for i in range(6)}
        )
        self.assertEqual(
            sha256_file(source),
            "4890bdd629a7ad29ec6ed166de533bd0ca8db1659af17cb2b4ea849b2b527569",
        )


if __name__ == "__main__":
    unittest.main()

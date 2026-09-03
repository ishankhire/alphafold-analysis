from __future__ import annotations

import csv
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
SOURCE_DIR = ROOT / "delta_files/publication_plots/sept_2_plots/source_data"
PLOT_ROOT = ROOT / "delta_files/publication_plots/sept_2_plots"


def read_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


class SubmissionSmokeTest(unittest.TestCase):
    def test_frozen_cohort(self) -> None:
        rows = read_rows(
            ROOT
            / "delta_files/regression_5fold_v1/metadata/regression_outer_folds_v1.csv"
        )
        keys = {(row["source"], row["protein_id"]) for row in rows}
        self.assertEqual(len(rows), 552)
        self.assertEqual(len(keys), 552)
        self.assertEqual({int(row["outer_fold"]) for row in rows}, set(range(5)))

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
        ablation_dir = ROOT / "delta_files/layerwise_ablation_csvs"
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


if __name__ == "__main__":
    unittest.main()

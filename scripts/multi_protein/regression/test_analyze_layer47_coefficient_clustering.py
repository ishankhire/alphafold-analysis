#!/usr/bin/env python3
"""Focused tests for layer-47 coefficient clustering."""

from __future__ import annotations

import hashlib
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import adjusted_rand_score

from scripts.multi_protein.regression.analyze_layer47_coefficient_clustering import (
    N_FEATURES,
    build_selected_outputs,
    build_representations,
    fit_kmeans_sweep,
    load_layer47_coefficients,
)


class Layer47CoefficientClusteringTests(unittest.TestCase):
    def write_inputs(self, root: Path, *, duplicate: bool = False, missing: bool = False) -> tuple[Path, Path, Path]:
        protein_ids = ["cameo_A", "000000001", "000000002", "cameo_B"]
        metadata_rows = []
        coefficient_rows = []
        for protein_number, protein_id in enumerate(protein_ids):
            source = "ecod" if protein_id.isdigit() else "cameo"
            metadata_rows.append({"layer": 47, "protein_id": protein_id, "source": source, "n_residues": 80 + protein_number * 10, "size_bin": protein_number % 2})
            metadata_rows.append({"layer": 46, "protein_id": protein_id, "source": source, "n_residues": 80 + protein_number * 10, "size_bin": protein_number % 2})
            coefficient_rows.append({"layer": 47, "protein_id": protein_id, "feature_index": -1, "feature_name": "intercept", "coefficient_standardized": 10.0})
            for feature_index in range(N_FEATURES):
                if missing and protein_id == protein_ids[-1] and feature_index == N_FEATURES - 1:
                    continue
                coefficient_rows.append({"layer": 47, "protein_id": protein_id, "feature_index": feature_index, "feature_name": f"feature_{feature_index}", "coefficient_standardized": protein_number + feature_index / 1000})
            coefficient_rows.append({"layer": 46, "protein_id": protein_id, "feature_index": 0, "feature_name": "feature_0", "coefficient_standardized": -999.0})
        if duplicate:
            coefficient_rows.append(coefficient_rows[1].copy())

        coefficient_path = root / "coefficients.csv"
        metadata_path = root / "per_protein.csv"
        category_path = root / "categories.csv"
        pd.DataFrame(coefficient_rows).to_csv(coefficient_path, index=False)
        pd.DataFrame(metadata_rows).to_csv(metadata_path, index=False)
        pd.DataFrame(
            [
                {"prot": "000000001", "category": "alpha"},
                {"prot": "000000002", "category": "beta"},
            ]
        ).to_csv(category_path, index=False)
        return coefficient_path, metadata_path, category_path

    def test_loader_filters_intercept_and_other_layers(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            paths = self.write_inputs(Path(directory))
            matrix, metadata, names = load_layer47_coefficients(*paths)
        self.assertEqual(matrix.shape, (4, N_FEATURES))
        self.assertEqual(names[0], "feature_0")
        self.assertNotIn(-999.0, matrix)
        categories = metadata.set_index("protein_id")["ecod_category"].to_dict()
        self.assertEqual(categories["000000001"], "alpha")
        self.assertTrue(pd.isna(categories["cameo_A"]))

    def test_loader_rejects_duplicate_or_missing_features(self) -> None:
        for option in ("duplicate", "missing"):
            with self.subTest(option=option), tempfile.TemporaryDirectory() as directory:
                kwargs = {option: True}
                paths = self.write_inputs(Path(directory), **kwargs)
                with self.assertRaises(ValueError):
                    load_layer47_coefficients(*paths)

    def test_loader_reads_canonical_npz_index(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            protein_ids = ["000000001", "000000002", "cameo_A", "cameo_B"]
            metadata_path = root / "per_protein.csv"
            pd.DataFrame([
                {
                    "layer": 47,
                    "protein_id": protein_id,
                    "source": "ecod" if protein_id.isdigit() else "cameo",
                    "n_residues": 80 + index,
                    "size_bin": index,
                    "evaluation_scope": "all_pairs",
                }
                for index, protein_id in enumerate(protein_ids)
            ]).to_csv(metadata_path, index=False)
            category_path = root / "categories.csv"
            pd.DataFrame(columns=["prot", "category"]).to_csv(category_path, index=False)
            shard_relative = "model_artifacts/coefficient_shards/layer_47.npz"
            shard_path = root / shard_relative
            shard_path.parent.mkdir(parents=True)
            matrix = np.arange(len(protein_ids) * N_FEATURES, dtype=np.float64).reshape(
                len(protein_ids), N_FEATURES,
            )
            np.savez_compressed(
                shard_path,
                coefficient_standardized=matrix,
                protein_id=np.asarray(protein_ids),
                feature_names=np.asarray([f"feature_{index}" for index in range(N_FEATURES)]),
            )
            digest = hashlib.sha256(shard_path.read_bytes()).hexdigest()
            index_path = root / "model_artifacts" / "coefficient_index.csv"
            pd.DataFrame([
                {
                    "layer": 47,
                    "protein_id": protein_id,
                    "shard_path": shard_relative,
                    "array_row": index,
                    "fit_status": "fitted",
                    "file_sha256": digest,
                }
                for index, protein_id in enumerate(protein_ids)
            ]).to_csv(index_path, index=False)
            loaded, metadata, names = load_layer47_coefficients(
                index_path, metadata_path, category_path,
            )
        np.testing.assert_array_equal(loaded, matrix)
        self.assertEqual(metadata["protein_id"].tolist(), protein_ids)
        self.assertEqual(names[-1], f"feature_{N_FEATURES - 1}")

    def test_normalization_rejects_zero_vector(self) -> None:
        matrix = np.ones((3, N_FEATURES), dtype=np.float64)
        matrix[1] = 0
        with self.assertRaises(ValueError):
            build_representations(matrix)

    def test_deterministic_sweep_recovers_two_clusters(self) -> None:
        rng = np.random.default_rng(4)
        first = rng.normal(loc=-4.0, scale=0.1, size=(12, N_FEATURES))
        second = rng.normal(loc=4.0, scale=0.1, size=(12, N_FEATURES))
        matrix = np.vstack([first, second]).astype(np.float64)
        truth = np.repeat([0, 1], 12)
        arguments = dict(
            k_values=(1, 2, 3),
            random_seed=12,
            full_n_init=5,
            stability_n_init=3,
            n_stability_repeats=3,
        )
        first_result = fit_kmeans_sweep(matrix, "magnitude_sensitive", **arguments)
        second_result = fit_kmeans_sweep(matrix, "magnitude_sensitive", **arguments)
        diagnostics, models, selected_k = first_result
        self.assertEqual(selected_k, 2)
        self.assertGreater(adjusted_rand_score(truth, models[2].labels_), 0.99)
        pd.testing.assert_frame_equal(diagnostics, second_result[0])

    def test_selected_output_assembly_accepts_existing_layer_column(self) -> None:
        rng = np.random.default_rng(9)
        matrix = np.vstack(
            [
                rng.normal(-2, 0.1, size=(6, N_FEATURES)),
                rng.normal(2, 0.1, size=(6, N_FEATURES)),
            ]
        ).astype(np.float64)
        metadata = pd.DataFrame(
            {
                "protein_id": [f"protein_{index}" for index in range(len(matrix))],
                "layer": 47,
                "source": ["cameo"] * 6 + ["ecod"] * 6,
                "n_residues": np.arange(len(matrix)) + 50,
                "size_bin": [0] * 6 + [1] * 6,
                "ecod_category": [np.nan] * 6 + ["alpha"] * 3 + ["beta"] * 3,
            }
        )
        representations = build_representations(matrix)
        models = {}
        for name, values in representations.items():
            _, fitted, _ = fit_kmeans_sweep(
                values,
                name,
                k_values=(1, 2),
                full_n_init=3,
                stability_n_init=2,
                n_stability_repeats=2,
            )
            models[name] = fitted
        assignments, centroids, summaries, associations, details = build_selected_outputs(
            matrix,
            metadata,
            [f"feature_{index}" for index in range(N_FEATURES)],
            representations,
            models,
            {name: 2 for name in representations},
        )
        self.assertEqual(len(assignments), 2 * len(matrix))
        self.assertEqual(len(centroids), 2 * 2 * N_FEATURES)
        self.assertEqual(len(summaries), 4)
        self.assertEqual(len(associations), 8)
        self.assertEqual(set(details), set(representations))


if __name__ == "__main__":
    unittest.main()

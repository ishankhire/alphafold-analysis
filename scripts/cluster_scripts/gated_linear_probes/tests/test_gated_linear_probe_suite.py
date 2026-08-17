import csv
import json
import math
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


SUITE_DIR = Path(__file__).parents[1]
CLUSTER_DIR = SUITE_DIR.parent
GENERAL_DIR = CLUSTER_DIR / "generalized_model_ecod"
for path in (SUITE_DIR, CLUSTER_DIR, GENERAL_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import aggregate_gated_linear_probe_suite as aggregate
import gated_linear_probe_common as common
import prepare_gated_linear_probe_suite as preparation
from regression_output_helpers import (
    CSV_SCHEMAS,
    build_model_status_row,
    build_runtime_row,
)
from regression_protocol import ManifestEntry, sequence_separation_bin_codes, strict_lower_triangle_indices


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


def synthetic_record(protein_id, role, coordinates):
    n_residues = len(coordinates)
    pair_i, pair_j = strict_lower_triangle_indices(n_residues)
    distance = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
    return {
        "protein_id": protein_id,
        "source": "cameo",
        "master_role": role,
        "protein_split": "test" if role == "test" else "development",
        "n_residues": n_residues,
        "n_pairs": len(pair_i),
        "size_bin": 0,
        "metadata": {
            "pair_i": pair_i,
            "pair_j": pair_j,
            "y_true": distance.astype(np.float64),
            "coordinates": np.asarray(coordinates, dtype=np.float64),
            "sequence_bin_code": sequence_separation_bin_codes(pair_i, pair_j),
            "aa_pair_code": np.zeros(len(pair_i), dtype=np.int16),
            "distance_bin_code": np.zeros(len(pair_i), dtype=np.int16),
        },
    }


class ProtocolAndTaskMappingTests(unittest.TestCase):
    def test_production_and_smoke_task_mappings(self):
        self.assertEqual(common.configured_layers("production"), tuple(range(48)))
        self.assertEqual(common.configured_layers("smoke"), (0,))
        self.assertEqual(common.layer_for_task(47, "production"), 47)
        self.assertEqual(common.joint_task_for_id(287, "production"), (47, 5))
        self.assertEqual(common.joint_task_for_id(5, "smoke"), (0, 5))
        with self.assertRaises(ValueError):
            common.joint_task_for_id(6, "smoke")

    def test_smoke_subset_preserves_frozen_roles(self):
        entries = []
        for size_bin in range(5):
            for role in ("fit", "validation", "test"):
                entries.append(ManifestEntry(
                    manifest_version="regression_master_split_v1",
                    protein_id=f"{size_bin}_{role}",
                    source="cameo",
                    eligibility_status="eligible",
                    exclusion_reason="",
                    n_residues=3,
                    n_pairs=3,
                    size_bin=size_bin,
                    master_role=role,
                    protein_split="test" if role == "test" else "development",
                    outer_split_seed=42,
                    development_split_seed=43,
                ))
        selected = preparation.smoke_manifest_subset(entries)
        self.assertEqual(len(selected), 15)
        self.assertEqual({entry.master_role for entry in selected}, {"fit", "validation", "test"})
        for size_bin in range(5):
            self.assertEqual(
                {entry.master_role for entry in selected if entry.size_bin == size_bin},
                {"fit", "validation", "test"},
            )

    def test_fit_is_float64_protein_equal_and_replayable(self):
        rng = np.random.default_rng(7)
        x = rng.normal(size=(320, common.FEATURE_DIM)).astype(np.float64)
        ids = np.asarray(["fit_a"] * 80 + ["validation_b"] * 240, dtype=object)
        beta = rng.normal(size=common.FEATURE_DIM)
        y = x @ beta + 1.25
        fitted = common.fit_ols(x, y.astype(np.float64), ids)
        prediction = common.predict_ols(fitted, x)
        self.assertLess(float(np.max(np.abs(prediction - y))), 1e-8)
        raw_prediction = fitted["intercept_raw_scale"] + x @ fitted["coefficient_raw_scale"]
        np.testing.assert_allclose(prediction, raw_prediction, atol=1e-8, rtol=1e-10)
        self.assertEqual(fitted["feature_mean"].dtype, np.float64)


class RecombinationAndSchemaTests(unittest.TestCase):
    def test_fragments_recombine_and_overlap_is_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            first = Path(directory) / "first.npz"
            second = Path(directory) / "second.npz"
            common.atomic_savez_compressed(
                first,
                flat_indices=np.asarray([0, 2]),
                predictions=np.asarray([1.0, 3.0]),
                conditional_predictions=np.asarray([2.0, 2.0]),
            )
            common.atomic_savez_compressed(
                second,
                flat_indices=np.asarray([1, 3]),
                predictions=np.asarray([2.0, np.nan]),
                conditional_predictions=np.asarray([2.0, 2.0]),
            )
            prediction, conditional = common.merge_prediction_fragments(4, [first, second])
            np.testing.assert_allclose(prediction[:3], [1.0, 2.0, 3.0])
            self.assertTrue(math.isnan(prediction[3]))
            np.testing.assert_allclose(conditional, 2.0)
            with self.assertRaisesRegex(ValueError, "overlap"):
                common.merge_prediction_fragments(4, [first, first])

    def test_partial_coverage_retains_geometry_exclusion_and_canonical_headers(self):
        coordinates = np.asarray([
            [0.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0], [0.0, 0.0, 1.0],
        ])
        records = [
            synthetic_record("development", "fit", coordinates),
            synthetic_record("locked_test", "test", coordinates + 0.2),
        ]
        total = sum(record["n_pairs"] for record in records)
        prediction = np.concatenate([record["metadata"]["y_true"] for record in records])
        prediction[-1] = np.nan
        conditional = np.full(total, 1.0, dtype=np.float64)
        status = [build_model_status_row(
            analysis_name=common.ANALYSIS_JOINT,
            model_family=common.MODEL_FAMILY,
            feature_dim=common.FEATURE_DIM,
            layer=0,
            gate_aa_pair="ALA-ALA",
            gate_sequence_bin="1",
            fit_status="fitted",
        )]
        runtime = build_runtime_row(
            analysis_name=common.ANALYSIS_JOINT,
            model_family=common.MODEL_FAMILY,
            layer=0,
            completion_status="complete",
            n_fitted_proteins=1,
            n_evaluated_proteins=2,
            n_fitted_pairs=6,
            n_evaluated_pairs=11,
            elapsed_seconds=1.0,
        )
        manifest = {
            "run_mode": "smoke",
            "configuration_digest": "test",
            "master_split_digest": "split",
            "global_development_mean": 1.0,
            "input_roots": {},
        }
        with tempfile.TemporaryDirectory() as directory:
            shard = common.build_layer_shard(
                directory, common.ANALYSIS_JOINT, 0, records, prediction, conditional,
                status, runtime, manifest,
            )
            for filename in aggregate.SHARD_FILENAMES:
                with (shard / filename).open(newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertEqual(tuple(reader.fieldnames), CSV_SCHEMAS[filename])
            per_rows = read_csv(shard / "per_protein.csv")
            test_row = next(row for row in per_rows if row["protein_id"] == "locked_test")
            self.assertEqual(test_row["prediction_status"], "partial_coverage")
            self.assertLess(float(test_row["pair_coverage"]), 1.0)
            geometry = read_csv(shard / "geometry_metrics.csv")
            test_geometry = next(row for row in geometry if row["protein_id"] == "locked_test")
            self.assertEqual(test_geometry["geometry_status"], "incomplete_pair_coverage")
            self.assertEqual(test_geometry["rmsd"], "NaN")
            pooled = read_csv(shard / "pooled_metrics.csv")
            self.assertEqual({row["protein_split"] for row in pooled}, {"development", "test"})


if __name__ == "__main__":
    unittest.main()

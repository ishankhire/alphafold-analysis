import csv
import json
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
import consolidate_joint_gated_layer as joint_consolidation
import gated_linear_probe_common as common
import prepare_gated_linear_probe_suite as preparation
from regression_protocol import MASTER_MANIFEST_DIGEST, sequence_separation_bin_codes, spatial_distance_bin_codes


def read_csv(path):
    with Path(path).open(newline="") as handle:
        return list(csv.DictReader(handle))


class SyntheticCanonicalSuiteIntegrationTest(unittest.TestCase):
    def test_smoke_workers_recombine_consolidate_and_replay(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root = temp / "gated_linear_probes_smoke"
            destination = root / "consolidated"
            records = []
            rng = np.random.default_rng(2026)
            n_residues = 5
            pair_i, pair_j = np.tril_indices(n_residues, k=-1)
            for protein_index in range(15):
                protein_id = f"synthetic_{protein_index:02d}"
                protein_dir = temp / protein_id
                protein_dir.mkdir()
                pair_block = rng.normal(size=(n_residues, n_residues, 128)).astype(np.float32)
                np.save(protein_dir / f"{protein_id}_pair_block_0.npy", pair_block)
                coordinates = rng.normal(size=(n_residues, 3)).astype(np.float64)
                y_true = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
                role = "fit" if protein_index < 5 else "validation" if protein_index < 10 else "test"
                aa_codes = np.zeros(len(pair_i), dtype=np.int16)
                if role == "test":
                    aa_codes[-1] = 1  # This gate has no development support: explicit partial coverage.
                records.append({
                    "protein_id": protein_id,
                    "source": "cameo",
                    "master_role": role,
                    "protein_split": "test" if role == "test" else "development",
                    "n_residues": n_residues,
                    "n_pairs": len(pair_i),
                    "size_bin": protein_index % 5,
                    "protein_dir": protein_dir,
                    "metadata": {
                        "pair_i": pair_i,
                        "pair_j": pair_j,
                        "y_true": y_true,
                        "coordinates": coordinates,
                        "sequence_bin_code": sequence_separation_bin_codes(pair_i, pair_j),
                        "aa_pair_code": aa_codes,
                        "distance_bin_code": spatial_distance_bin_codes(y_true),
                    },
                })

            with mock.patch.object(common, "MIN_TRAIN_ROWS", 5):
                preparation_manifest = {
                    "run_mode": "smoke",
                    "configuration_digest": common.configuration_digest("smoke"),
                    "master_split_digest": MASTER_MANIFEST_DIGEST,
                    "prepared_protein_digest": "synthetic-smoke-digest",
                    "prepared_proteins": 15,
                    "prepared_role_counts": {"fit": 5, "validation": 5, "test": 5},
                    "input_roots": {},
                    "global_development_mean": float(np.mean([
                        np.mean(record["metadata"]["y_true"])
                        for record in records if record["protein_split"] == "development"
                    ])),
                    "versions": common.version_payload(),
                }
                baseline_rows = preparation.compute_conditional_baselines(records)
                baselines = {
                    (row["analysis_name"], row["gate_aa_pair"], row["gate_sequence_bin"]): {
                        "prediction": row["prediction"],
                        "n_training_proteins": row["n_training_proteins"],
                        "n_training_pairs": row["n_training_pairs"],
                    }
                    for row in baseline_rows
                }
                common.atomic_write_json(
                    root / common.PREPARATION_MANIFEST_FILENAME, preparation_manifest,
                )
                with (
                    mock.patch.dict(os.environ, {common.RUN_MODE_ENV: "smoke"}),
                    mock.patch.object(common, "suite_root", return_value=root),
                    mock.patch.object(common, "load_context", return_value=(preparation_manifest, records)),
                    mock.patch.object(common, "load_conditional_baselines", return_value=baselines),
                    mock.patch.object(common, "log", return_value=None),
                ):
                    common.run_analysis_task(common.ANALYSIS_SEQUENCE, task_id=0)
                    common.run_analysis_task(common.ANALYSIS_AA, task_id=0)
                    for task_id in range(6):
                        common.run_analysis_task(common.ANALYSIS_JOINT, task_id=task_id)
                    # A repeated task must verify and reuse every gate checkpoint.
                    common.run_analysis_task(common.ANALYSIS_SEQUENCE, task_id=0)

                with (
                    mock.patch.dict(os.environ, {common.RUN_MODE_ENV: "smoke"}),
                    mock.patch.object(joint_consolidation, "suite_root", return_value=root),
                    mock.patch.object(
                        joint_consolidation,
                        "load_context",
                        return_value=(preparation_manifest, records),
                    ),
                    mock.patch.object(joint_consolidation, "log", return_value=None),
                    mock.patch.object(aggregate, "log", return_value=None),
                ):
                    joint_consolidation.main(task_id=0)

                with (
                    mock.patch.dict(os.environ, {common.RUN_MODE_ENV: "smoke"}),
                    mock.patch.object(aggregate, "suite_root", return_value=root),
                    mock.patch.object(aggregate, "load_context", return_value=(preparation_manifest, records)),
                    mock.patch.object(aggregate, "log", return_value=None),
                ):
                    aggregate.main()

            for analysis_name in common.ANALYSIS_NAMES:
                analysis_dir = destination / analysis_name
                for filename in (
                    "run_manifest.json", "per_protein.csv", "pooled_metrics.csv",
                    "model_status.csv", "runtime.csv", "geometry_metrics.csv",
                    "geometry_summary.csv", "subgroup_metrics/by_sequence_separation.csv",
                    "subgroup_metrics/by_amino_acid_pair.csv",
                    "subgroup_metrics/by_spatial_distance.csv",
                    "model_artifacts/coefficient_index.csv",
                ):
                    self.assertTrue((analysis_dir / filename).exists(), filename)
                self.assertFalse((analysis_dir / "summary.md").exists())
                run_manifest = json.loads((analysis_dir / "run_manifest.json").read_text())
                self.assertEqual(run_manifest["requested_layers"], [0])
                self.assertEqual(run_manifest["master_split_digest"], MASTER_MANIFEST_DIGEST)
                shard_manifest = json.loads(
                    (analysis_dir / "layer_shards" / "layer_00" / "shard_manifest.json").read_text()
                )
                self.assertEqual(
                    run_manifest["configuration_digest"], shard_manifest["configuration_digest"]
                )
                self.assertEqual(len(read_csv(analysis_dir / "per_protein.csv")), 15)

            sequence_rows = read_csv(
                destination / common.ANALYSIS_SEQUENCE / "per_protein.csv"
            )
            self.assertTrue(all(row["prediction_status"] == "complete" for row in sequence_rows))
            self.assertEqual(
                {row["protein_split"] for row in sequence_rows}, {"development", "test"}
            )
            sequence_status = read_csv(
                destination / common.ANALYSIS_SEQUENCE / "model_status.csv"
            )
            supported_status = next(
                row for row in sequence_status if int(row["n_training_pairs"]) > 0
            )
            self.assertEqual(int(supported_status["n_training_proteins"]), 10)
            joint_rows = read_csv(
                destination / common.ANALYSIS_JOINT / "per_protein.csv"
            )
            test_joint = [row for row in joint_rows if row["protein_split"] == "test"]
            self.assertTrue(all(row["prediction_status"] == "partial_coverage" for row in test_joint))
            joint_geometry = read_csv(
                destination / common.ANALYSIS_JOINT / "geometry_metrics.csv"
            )
            test_geometry = [row for row in joint_geometry if row["protein_split"] == "test"]
            self.assertTrue(all(row["geometry_status"] == "incomplete_pair_coverage" for row in test_geometry))

            index_rows = read_csv(
                destination / common.ANALYSIS_SEQUENCE / "model_artifacts"
                / "coefficient_index.csv"
            )
            fitted_index = next(row for row in index_rows if row["fit_status"].startswith("fitted"))
            shard_path = destination / common.ANALYSIS_SEQUENCE / fitted_index["shard_path"]
            with np.load(shard_path) as data:
                row_index = int(fitted_index["array_row"])
                x = rng.normal(size=common.FEATURE_DIM)
                standardized = (
                    data["intercept_standardized"][row_index]
                    + ((x - data["feature_mean"][row_index]) / data["feature_scale"][row_index])
                    @ data["coefficient_standardized"][row_index]
                )
                raw = (
                    data["intercept_raw_scale"][row_index]
                    + x @ data["coefficient_raw_scale"][row_index]
                )
                self.assertAlmostEqual(float(standardized), float(raw), places=9)


if __name__ == "__main__":
    unittest.main()

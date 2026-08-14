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
import gated_linear_probe_common as common
import prepare_gated_linear_probe_suite as preparation


class SyntheticSuiteIntegrationTest(unittest.TestCase):
    def test_all_four_analyses_checkpoint_and_aggregate(self):
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            root = temp / "gated_linear_probes_smoke"
            destination = root / "consolidated"
            records, assignments = [], []
            rng = np.random.default_rng(2026)
            n_residues = 8
            pair_i, pair_j = np.tril_indices(n_residues, k=-1)
            raw_sequence = np.abs(pair_i - pair_j).astype(np.int32)
            for protein_index in range(6):
                protein_id = f"synthetic_{protein_index}"
                protein_dir = temp / protein_id
                protein_dir.mkdir()
                pb = rng.normal(size=(n_residues, n_residues, 128)).astype(np.float32)
                np.save(protein_dir / f"{protein_id}_pair_block_0.npy", pb)
                y_true = (2.0 + raw_sequence + protein_index * 0.1).astype(np.float64)
                metadata = {
                    "y_true": y_true,
                    "raw_sequence_separation": raw_sequence,
                    "sequence_bin_code": common.sequence_bin_codes(raw_sequence),
                    "aa_pair_code": ((np.arange(len(pair_i)) + protein_index) % 210).astype(np.int16),
                    "distance_bin_code": common.distance_bin_codes(y_true),
                    "chain_valid_mask": np.ones(len(pair_i), dtype=bool),
                    "canonical_validity_mask": np.ones(len(pair_i), dtype=bool),
                }
                split = "train" if protein_index < 5 else "test"
                record = {
                    "protein_id": protein_id, "source": "synthetic", "split": split,
                    "n_residues": n_residues, "size_bin": protein_index % 5,
                    "n_pairs": len(pair_i), "protein_dir": protein_dir, "metadata": metadata,
                }
                records.append(record)
                assignments.append({
                    "split": split, "protein_id": protein_id, "source": "synthetic",
                    "n_residues": n_residues, "size_bin": protein_index % 5,
                    "n_pairs": len(pair_i), "target_variance": float(np.var(y_true)),
                })

            configuration = common.configuration_payload("smoke")
            split_digest = common.sha256_bytes(common.canonical_json_bytes(assignments))
            manifest = {
                **configuration, "configuration_digest": common.configuration_digest("smoke"),
                "split_digest": split_digest, "run_mode": "smoke",
            }
            common.atomic_write_json(root / "run_manifest.json", manifest)
            common.atomic_write_csv(root / "split_assignments.csv", preparation.SPLIT_FIELDNAMES, assignments)
            common.atomic_write_csv(root / "skipped_proteins.csv", preparation.SKIPPED_FIELDNAMES, [])
            common.atomic_write_csv(root / "excluded_pairs_summary.csv", preparation.EXCLUDED_FIELDNAMES, [])
            preparation._write_baselines(root, records)
            preparation._write_true_distance_statistics(root, records)

            with (
                mock.patch.dict(os.environ, {common.RUN_MODE_ENV: "smoke"}),
                mock.patch.object(common, "suite_root", return_value=root),
                mock.patch.object(common, "load_context", return_value=(manifest, records)),
                mock.patch.object(common, "MIN_TRAIN_ROWS", 5),
                mock.patch.object(common, "log", return_value=None),
            ):
                common.run_analysis_task("analysis_1_sequence_gated", task_id=0)
                common.run_analysis_task("analysis_2_aa_pair_gated", task_id=0)
                for task_id in range(6):
                    common.run_analysis_task("analysis_4_aa_sequence_gated", task_id=task_id)
                common.run_analysis_task("analysis_1_sequence_gated", task_id=0)

            with (
                mock.patch.dict(os.environ, {common.RUN_MODE_ENV: "smoke"}),
                mock.patch.object(aggregate, "suite_root", return_value=root),
            ):
                aggregate.main()

            self.assertTrue((destination / "analysis_1_sequence_gated" / "gate_summary.csv").exists())
            self.assertTrue((destination / "analysis_2_aa_pair_gated" / "coefficients.csv").exists())
            self.assertTrue((destination / "analysis_3_true_distance_statistics"
                             / "true_distance_category_stats.csv").exists())
            self.assertTrue((destination / "analysis_4_aa_sequence_gated"
                             / "coefficient_index.csv").exists())
            task_manifest = json.loads((root / "analysis_1_sequence_gated" / "task_shards"
                                        / "task_000" / "task_manifest.json").read_text())
            self.assertTrue(task_manifest["completed"])
            self.assertEqual(task_manifest["completed_gate_keys"], task_manifest["expected_gate_keys"])
            with self.assertRaisesRegex(ValueError, "Missing task manifest"):
                aggregate._load_analysis(root, "analysis_1_sequence_gated", 2)


if __name__ == "__main__":
    unittest.main()

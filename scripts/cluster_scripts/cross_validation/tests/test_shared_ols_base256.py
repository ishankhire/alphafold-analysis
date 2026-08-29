import csv
import json
import shutil
import sys
import tempfile
import unittest
from pathlib import Path
from types import MappingProxyType, SimpleNamespace

import numpy as np


TEST_DIR = Path(__file__).resolve().parent
CV_DIR = TEST_DIR.parent
CLUSTER_DIR = CV_DIR.parent
for directory in (CLUSTER_DIR, CV_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import aggregate_shared_ols_base256 as aggregate
import cross_validation_common as cv
import shared_ols_base256_common as shared
from regression_protocol import FEATURE_DIM, strict_lower_triangle_indices


class SyntheticFoldBundle:
    def __init__(self, records):
        entries = {
            record.protein_id: SimpleNamespace(
                protein_id=record.protein_id,
                source=record.source,
                n_residues=record.n_residues,
                n_pairs=record.targets.size,
                outer_fold=record.outer_fold,
            )
            for record in records
        }
        self.outer_by_protein = MappingProxyType(entries)
        self.eligible_protein_ids = tuple(sorted(entries))
        self.fold_digests = MappingProxyType({
            "cohort_digest": "synthetic-cohort",
            "outer_fold_digest": "synthetic-outer-folds",
            "pair_fold_manifest_digest": "synthetic-pair-folds",
        })

    def protein_ids_for_outer_fold(self, outer_fold):
        return tuple(
            protein_id for protein_id in self.eligible_protein_ids
            if self.outer_by_protein[protein_id].outer_fold == int(outer_fold)
        )

    def training_protein_ids(self, heldout_fold):
        heldout = set(self.protein_ids_for_outer_fold(heldout_fold))
        return tuple(
            protein_id for protein_id in self.eligible_protein_ids
            if protein_id not in heldout
        )


def synthetic_records():
    records = []
    pair_i, pair_j = strict_lower_triangle_indices(13)
    for fold in range(5):
        rng = np.random.default_rng(500 + fold)
        coordinates = rng.normal(size=(13, 3))
        targets = np.linalg.norm(
            coordinates[pair_i] - coordinates[pair_j], axis=1,
        )
        features = rng.normal(scale=0.15, size=(targets.size, FEATURE_DIM))
        features[:, 0] = targets
        features[:, 1] = np.abs(pair_i - pair_j)
        records.append(shared.ProteinLayerRecord(
            protein_id=f"p{fold}", source="cameo", n_residues=13,
            outer_fold=fold, coordinates=coordinates,
            chain_ids=tuple("A" for _ in range(13)), pair_i=pair_i.copy(),
            pair_j=pair_j.copy(), targets=targets,
            synthetic_features=features,
        ))
    return records


class SharedOLSUnitTests(unittest.TestCase):
    def test_fit_replay_and_artifact_round_trip(self):
        records = synthetic_records()
        features = {
            record.protein_id: record.synthetic_features for record in records[:4]
        }
        targets = {record.protein_id: record.targets for record in records[:4]}
        fitted, training_y, training_predictions = shared.fit_shared_ols(features, targets)
        self.assertEqual(fitted.coefficient_standardized.shape, (FEATURE_DIM,))
        self.assertEqual(training_y.shape, training_predictions.shape)
        self.assertEqual(fitted.fit_status, "fitted")
        heldout = shared.predict_shared_ols(fitted, records[4].synthetic_features)
        self.assertEqual(heldout.shape, records[4].targets.shape)
        self.assertTrue(np.all(np.isfinite(heldout)))

        task = cv.map_fold_layer_task(0)
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coefficients.npz"
            shared.write_coefficient_artifact(path, fitted, task)
            loaded = shared.load_coefficient_artifact(
                path, expected_task=task,
                expected_scaler_digest=fitted.scaler_digest,
            )
            np.testing.assert_allclose(
                loaded["coefficient_standardized"], fitted.coefficient_standardized,
            )
            self.assertEqual(loaded["global_baseline_digest"], fitted.global_baseline_digest)

    def test_launcher_has_locked_stage_three_resources_and_interface(self):
        launcher = (CV_DIR / "shared_ols_base256_fold_layer.sbatch").read_text()
        required = (
            "#SBATCH --array=0-239%4", "#SBATCH --mem=120g",
            "#SBATCH --time=00:30:00", "#SBATCH --constraint=\"scratch&work\"",
            "#SBATCH --mail-type=BEGIN,END,FAIL",
            "#SBATCH --mail-user=ikhire3@gatech.edu",
            "SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}",
            'srun python3 -u "$SCRIPT_DIR/shared_ols_base256_fold_layer.py"',
        )
        self.assertTrue(all(value in launcher for value in required))


class SharedOLSIntegrationResumeCorruptionTests(unittest.TestCase):
    def test_fivefold_task_family_aggregates_resumes_and_rejects_corruption(self):
        records = synthetic_records()
        bundle = SyntheticFoldBundle(records)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "shared_ols_base256"
            for fold in range(5):
                task = cv.map_fold_layer_task(cv.fold_layer_task_id(fold, 0))
                shared.execute_fold_layer_task(
                    task=task, records=records,
                    training_protein_ids=bundle.training_protein_ids(fold),
                    heldout_protein_ids=bundle.protein_ids_for_outer_fold(fold),
                    fold_digests=bundle.fold_digests,
                    input_digest="synthetic-input-v1", root=root,
                )

            first = cv.map_fold_layer_task(0)
            shared.execute_fold_layer_task(
                task=first, records=records,
                training_protein_ids=bundle.training_protein_ids(0),
                heldout_protein_ids=bundle.protein_ids_for_outer_fold(0),
                fold_digests=bundle.fold_digests,
                input_digest="synthetic-input-v1", root=root,
                feature_loader=lambda record: self.fail("resume reopened features"),
            )

            with self.assertRaisesRegex(ValueError, "all 48 layers"):
                aggregate.aggregate_shared_ols(
                    root=root, fold_bundle=bundle, layers=(0,),
                )
            aggregate.aggregate_shared_ols(
                root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
            )
            with (root / cv.PER_PROTEIN_FILENAME).open(newline="") as handle:
                per_rows = list(csv.DictReader(handle))
            with (root / cv.REPLAY_INDEX_FILENAME).open(newline="") as handle:
                replay_rows = list(csv.DictReader(handle))
            with (root / cv.MODEL_STABILITY_FILENAME).open(newline="") as handle:
                stability_rows = list(csv.DictReader(handle))
            with (root / cv.STABLE_FIT_METRICS_FILENAME).open(newline="") as handle:
                stable_fit_rows = list(csv.DictReader(handle))
            self.assertEqual(len(per_rows), 5)
            self.assertEqual({row["outer_fold"] for row in per_rows}, set("01234"))
            self.assertEqual(len(replay_rows), 5)
            self.assertEqual(len(stability_rows), FEATURE_DIM)
            self.assertTrue(all(row["n_models"] == "5" for row in stability_rows))
            self.assertEqual(stable_fit_rows[0]["n_proteins_included"], "5")
            manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(manifest["completed_task_ids"], [0, 48, 96, 144, 192])
            self.assertFalse(manifest["raw_pair_predictions_saved"])

            first_shard = shared.task_shard_path(root, first)
            saved_shard = root / "saved_missing_task_0"
            first_shard.rename(saved_shard)
            with self.assertRaisesRegex(ValueError, "Shard inventory mismatch"):
                aggregate.aggregate_shared_ols(
                    root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
                )
            saved_shard.rename(first_shard)

            checkpoint_path = first_shard / shared.TASK_CHECKPOINT_FILENAME
            checkpoint_bytes = checkpoint_path.read_bytes()
            checkpoint = json.loads(checkpoint_bytes)
            checkpoint["status"] = "incomplete"
            checkpoint_path.write_text(json.dumps(checkpoint))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                aggregate.aggregate_shared_ols(
                    root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
                )
            checkpoint_path.write_bytes(checkpoint_bytes)

            duplicate_shard = first_shard.parent / "duplicate_task_0"
            shutil.copytree(first_shard, duplicate_shard)
            with self.assertRaisesRegex(ValueError, "Duplicate physical shard"):
                aggregate.aggregate_shared_ols(
                    root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
                )
            shutil.rmtree(duplicate_shard)

            runtime_path = shared.task_shard_path(root, first) / "runtime.csv"
            runtime_bytes = runtime_path.read_bytes()
            same_size_corruption = bytearray(runtime_bytes)
            same_size_corruption[0] ^= 1
            runtime_path.write_bytes(same_size_corruption)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                aggregate.aggregate_shared_ols(
                    root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
                )
            runtime_path.write_bytes(runtime_bytes)

            runtime_path.write_text(runtime_path.read_text() + "corruption\n")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                shared.execute_fold_layer_task(
                    task=first, records=records,
                    training_protein_ids=bundle.training_protein_ids(0),
                    heldout_protein_ids=bundle.protein_ids_for_outer_fold(0),
                    fold_digests=bundle.fold_digests,
                    input_digest="synthetic-input-v1", root=root,
                )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                aggregate.validate_task_shard(root, first, bundle)


if __name__ == "__main__":
    unittest.main()

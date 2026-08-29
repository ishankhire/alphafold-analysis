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

import aggregate_sequence_gated_ols_base256 as aggregate
import cross_validation_common as cv
import sequence_gated_ols_base256_common as gated
from regression_protocol import (
    FEATURE_DIM,
    SEQUENCE_BIN_LABELS,
    sequence_separation_bin_codes,
    strict_lower_triangle_indices,
)


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
    n_residues = 66
    pair_i, pair_j = strict_lower_triangle_indices(n_residues)
    codes = sequence_separation_bin_codes(pair_i, pair_j)
    for index in range(10):
        fold = index % 5
        rng = np.random.default_rng(900 + index)
        coordinates = rng.normal(size=(n_residues, 3))
        targets = np.linalg.norm(
            coordinates[pair_i] - coordinates[pair_j], axis=1,
        )
        features = rng.normal(scale=0.2, size=(targets.size, FEATURE_DIM))
        features[:, 0] = targets
        records.append(gated.ProteinLayerRecord(
            protein_id=f"p{index:02d}",
            source="cameo" if index % 2 == 0 else "ecod",
            n_residues=n_residues, outer_fold=fold,
            coordinates=coordinates,
            chain_ids=tuple("A" for _ in range(n_residues)),
            pair_i=pair_i.copy(), pair_j=pair_j.copy(), targets=targets,
            synthetic_features=features,
        ))
    return records


def fake_fit(label, value):
    scale = np.ones(FEATURE_DIM, dtype=np.float64)
    zeros = np.zeros(FEATURE_DIM, dtype=np.float64)
    return gated.SequenceGateOLSFit(
        sequence_gate=label,
        coefficient_standardized=zeros.copy(),
        coefficient_raw_scale=zeros.copy(),
        intercept_standardized=float(value),
        intercept_raw_scale=float(value),
        feature_mean=zeros.copy(), feature_scale=scale,
        scaler_digest=f"scaler-{label}",
        conditional_baseline_value=float(value + 10),
        conditional_baseline_digest=f"conditional-{label}",
        global_baseline_value=0.0, global_baseline_digest="global",
        design_rank=FEATURE_DIM, largest_singular_value=1.0,
        smallest_singular_value=1.0, condition_number=1.0,
        fit_status="fitted", support_flags="",
        n_training_proteins=5, n_training_pairs=300,
    )


class SequenceGatedOLSUnitTests(unittest.TestCase):
    def test_gate_isolation_train_only_scaling_and_protein_equal_weighting(self):
        feature_blocks = {}
        target_blocks = {}
        code_blocks = {}
        expected_protein_means = []
        rng = np.random.default_rng(71)
        for protein_index, n_gate_rows in enumerate((52, 53, 54, 55, 80)):
            protein_id = f"train-{protein_index}"
            gate_features = rng.normal(size=(n_gate_rows, FEATURE_DIM)) + protein_index * 3.0
            other_features = np.full((17, FEATURE_DIM), 1e6 + protein_index)
            features = np.vstack((gate_features, other_features))
            targets = features[:, 0] * 0.25 + 2.0
            codes = np.concatenate((
                np.zeros(n_gate_rows, dtype=np.int16),
                np.ones(17, dtype=np.int16),
            ))
            feature_blocks[protein_id] = features
            target_blocks[protein_id] = targets
            code_blocks[protein_id] = codes
            expected_protein_means.append(np.mean(gate_features, axis=0))
        global_baseline = cv.fit_train_only_baseline(target_blocks)
        result = gated.fit_sequence_gate_ols(
            feature_blocks, target_blocks, code_blocks,
            sequence_gate="1",
            global_baseline_value=global_baseline.value,
            global_baseline_digest=global_baseline.baseline_digest,
        )
        expected_mean = np.mean(np.vstack(expected_protein_means), axis=0)
        np.testing.assert_allclose(result.fit.feature_mean, expected_mean, rtol=1e-12, atol=1e-12)
        self.assertEqual(result.fit.n_training_proteins, 5)
        self.assertEqual(result.fit.n_training_pairs, sum((52, 53, 54, 55, 80)))
        self.assertNotAlmostEqual(result.fit.feature_mean[0], np.mean(np.vstack(list(feature_blocks.values())), axis=0)[0])
        self.assertEqual(result.training_targets.shape, result.training_predictions.shape)
        self.assertTrue(np.all(np.isfinite(result.training_predictions)))

    def test_recombination_requires_all_gates_and_exact_once_coverage(self):
        fits = {
            label: fake_fit(label, code)
            for code, label in enumerate(SEQUENCE_BIN_LABELS)
        }
        features = np.zeros((6, FEATURE_DIM), dtype=np.float64)
        codes = np.arange(6, dtype=np.int16)
        predictions, conditional = gated.recombine_sequence_gate_predictions(
            fits, features, codes,
        )
        np.testing.assert_array_equal(predictions, np.arange(6, dtype=np.float64))
        np.testing.assert_array_equal(conditional, np.arange(10, 16, dtype=np.float64))
        incomplete = dict(fits)
        del incomplete["50+"]
        with self.assertRaisesRegex(ValueError, "exactly the six"):
            gated.recombine_sequence_gate_predictions(incomplete, features, codes)
        with self.assertRaisesRegex(ValueError, "exact-once complete"):
            gated.recombine_sequence_gate_predictions(fits, features, np.array([0, 1, 2, 3, 4, -1]))

    def test_undefined_cross_chain_pairs_and_unsupported_gates_fail(self):
        pair_i, pair_j = strict_lower_triangle_indices(3)
        record = gated.ProteinLayerRecord(
            protein_id="cross-chain", source="cameo", n_residues=3,
            outer_fold=0, coordinates=np.zeros((3, 3)),
            chain_ids=("A", "A", "B"), pair_i=pair_i, pair_j=pair_j,
            targets=np.ones(pair_i.size),
            synthetic_features=np.zeros((pair_i.size, FEATURE_DIM)),
        )
        with self.assertRaisesRegex(ValueError, "undefined sequence separation"):
            gated.validated_sequence_gate_codes(record)

        features = {f"p{i}": np.zeros((60, FEATURE_DIM)) for i in range(5)}
        targets = {protein_id: np.ones(60) for protein_id in features}
        codes = {protein_id: np.zeros(60, dtype=np.int16) for protein_id in features}
        with self.assertRaisesRegex(ValueError, "Invalid sequence gate 50\\+"):
            gated.fit_sequence_gate_ols(
                features, targets, codes, sequence_gate="50+",
                global_baseline_value=1.0, global_baseline_digest="global",
            )

    def test_launcher_has_locked_stage_five_resources_and_interface(self):
        launcher = (CV_DIR / "sequence_gated_ols_base256_fold_layer.sbatch").read_text()
        required = (
            "#SBATCH --array=0-239%8", "#SBATCH --mem=120g",
            "#SBATCH --time=03:00:00", "#SBATCH --constraint=\"scratch&work\"",
            "#SBATCH --mail-type=BEGIN,END,FAIL",
            "#SBATCH --mail-user=ikhire3@gatech.edu",
            "SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}",
            'srun python3 -u "$SCRIPT_DIR/sequence_gated_ols_base256_fold_layer.py"',
        )
        self.assertTrue(all(value in launcher for value in required))


class SequenceGatedOLSIntegrationResumeCorruptionTests(unittest.TestCase):
    def test_fivefold_gate_family_aggregates_resumes_and_rejects_corruption(self):
        records = synthetic_records()
        bundle = SyntheticFoldBundle(records)
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "sequence_gated_ols_base256"
            for fold in range(5):
                task = cv.map_fold_layer_task(cv.fold_layer_task_id(fold, 0))
                gated.execute_fold_layer_task(
                    task=task, records=records,
                    training_protein_ids=bundle.training_protein_ids(fold),
                    heldout_protein_ids=bundle.protein_ids_for_outer_fold(fold),
                    fold_digests=bundle.fold_digests,
                    input_digest="synthetic-input-v1", root=root,
                )

            first = cv.map_fold_layer_task(0)
            gated.execute_fold_layer_task(
                task=first, records=records,
                training_protein_ids=bundle.training_protein_ids(0),
                heldout_protein_ids=bundle.protein_ids_for_outer_fold(0),
                fold_digests=bundle.fold_digests,
                input_digest="synthetic-input-v1", root=root,
                feature_loader=lambda record: self.fail("resume reopened features"),
            )

            with self.assertRaisesRegex(ValueError, "all 48 layers"):
                aggregate.aggregate_sequence_gated_ols(
                    root=root, fold_bundle=bundle, layers=(0,),
                )
            aggregate.aggregate_sequence_gated_ols(
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
            with (root / cv.POOLED_METRICS_FILENAME).open(newline="") as handle:
                pooled_rows = list(csv.DictReader(handle))
            self.assertEqual(len(per_rows), 10)
            self.assertEqual({row["outer_fold"] for row in per_rows}, set("01234"))
            self.assertEqual(len(replay_rows), 30)
            self.assertEqual({row["sequence_gate"] for row in replay_rows}, set(SEQUENCE_BIN_LABELS))
            self.assertEqual(len(stability_rows), len(SEQUENCE_BIN_LABELS) * FEATURE_DIM)
            self.assertTrue(all(row["n_models"] == "5" for row in stability_rows))
            self.assertEqual(stable_fit_rows[0]["n_proteins_included"], "10")
            self.assertEqual(
                pooled_rows[0]["conditional_baseline_type"],
                gated.CONDITIONAL_BASELINE_TYPE,
            )
            manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(manifest["completed_task_ids"], [0, 48, 96, 144, 192])
            self.assertEqual(manifest["expected_replay_artifacts"], 30)
            self.assertFalse(manifest["raw_pair_predictions_saved"])

            first_shard = gated.task_shard_path(root, first)
            saved_shard = root / "saved_missing_task_0"
            first_shard.rename(saved_shard)
            with self.assertRaisesRegex(ValueError, "Shard inventory mismatch"):
                aggregate.aggregate_sequence_gated_ols(
                    root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
                )
            saved_shard.rename(first_shard)

            checkpoint_path = first_shard / gated.TASK_CHECKPOINT_FILENAME
            checkpoint_bytes = checkpoint_path.read_bytes()
            checkpoint = json.loads(checkpoint_bytes)
            checkpoint["status"] = "incomplete"
            checkpoint_path.write_text(json.dumps(checkpoint))
            with self.assertRaisesRegex(ValueError, "incomplete"):
                aggregate.aggregate_sequence_gated_ols(
                    root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
                )
            checkpoint_path.write_bytes(checkpoint_bytes)

            duplicate_shard = first_shard.parent / "duplicate_task_0"
            shutil.copytree(first_shard, duplicate_shard)
            with self.assertRaisesRegex(ValueError, "Duplicate physical shard"):
                aggregate.aggregate_sequence_gated_ols(
                    root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
                )
            shutil.rmtree(duplicate_shard)

            runtime_path = gated.task_shard_path(root, first) / "runtime.csv"
            runtime_bytes = runtime_path.read_bytes()
            same_size_corruption = bytearray(runtime_bytes)
            same_size_corruption[0] ^= 1
            runtime_path.write_bytes(same_size_corruption)
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                aggregate.aggregate_sequence_gated_ols(
                    root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
                )
            runtime_path.write_bytes(runtime_bytes)

            runtime_path.write_text(runtime_path.read_text() + "corruption\n")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                gated.execute_fold_layer_task(
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

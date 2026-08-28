import csv
import json
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

import aggregate_per_protein_ols_base256 as aggregate
import cross_validation_common as cv
import per_protein_ols_base256_common as per_protein
from regression_protocol import (
    FEATURE_DIM,
    N_PAIR_FOLDS,
    pair_fold_assignment_digest,
    pair_fold_seed,
    strict_lower_triangle_indices,
    within_protein_pair_folds,
)
from shared_ols_base256_common import ProteinLayerRecord


class SyntheticFoldBundle:
    def __init__(self, records):
        outer_entries = {}
        pair_entries = {}
        for record in records:
            assignments = within_protein_pair_folds(
                record.n_residues, record.protein_id,
            )
            counts = tuple(
                int(value) for value in np.bincount(assignments, minlength=N_PAIR_FOLDS)
            )
            outer_entries[record.protein_id] = SimpleNamespace(
                protein_id=record.protein_id,
                source=record.source,
                n_residues=record.n_residues,
                n_pairs=record.targets.size,
                outer_fold=record.outer_fold,
            )
            pair_entries[record.protein_id] = SimpleNamespace(
                protein_id=record.protein_id,
                source=record.source,
                n_residues=record.n_residues,
                n_pairs=record.targets.size,
                fold_counts=counts,
                pair_fold_seed=pair_fold_seed(record.protein_id),
                assignment_digest=pair_fold_assignment_digest(
                    record.n_residues, record.protein_id,
                ),
            )
        self.outer_by_protein = MappingProxyType(outer_entries)
        self.pair_by_protein = MappingProxyType(pair_entries)
        self.eligible_protein_ids = tuple(sorted(outer_entries))
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


def synthetic_records():
    records = []
    pair_i, pair_j = strict_lower_triangle_indices(8)
    for fold in range(5):
        rng = np.random.default_rng(900 + fold)
        coordinates = rng.normal(size=(8, 3))
        targets = np.linalg.norm(
            coordinates[pair_i] - coordinates[pair_j], axis=1,
        )
        features = rng.normal(scale=0.05, size=(targets.size, FEATURE_DIM))
        features[:, 0] = targets
        features[:, 1] = np.abs(pair_i - pair_j)
        records.append(ProteinLayerRecord(
            protein_id=f"p{fold}", source="cameo", n_residues=8,
            outer_fold=fold, coordinates=coordinates,
            chain_ids=tuple("A" for _ in range(8)), pair_i=pair_i.copy(),
            pair_j=pair_j.copy(), targets=targets,
            synthetic_features=features,
        ))
    return records


class PerProteinOLSUnitTests(unittest.TestCase):
    def test_pair_fold_fit_uses_train_only_scaler_and_exact_once_assembly(self):
        record = synthetic_records()[0]
        assignments = within_protein_pair_folds(record.n_residues, record.protein_id)
        assembler = cv.OOFPairAssembler(
            record.targets, with_conditional_baseline=True,
        )
        fits = []
        for pair_fold in range(N_PAIR_FOLDS):
            fit, training, _, heldout_predictions = per_protein.fit_pair_fold_ols(
                record.protein_id, record.synthetic_features, record.targets,
                assignments, pair_fold,
            )
            heldout = np.flatnonzero(assignments == pair_fold)
            expected_mean = np.mean(
                np.asarray(record.synthetic_features[training], dtype=np.float64), axis=0,
            )
            np.testing.assert_allclose(fit.feature_mean, expected_mean)
            assembler.add_fold(
                pair_fold, heldout, heldout_predictions, fit.baseline_value,
                conditional_baseline_predictions=fit.baseline_value,
            )
            fits.append(fit)
        assembled = assembler.finalize()
        np.testing.assert_array_equal(assembled["pair_folds"], assignments)
        self.assertTrue(np.all(np.isfinite(assembled["predictions"])))
        self.assertEqual(
            np.bincount(assembled["pair_folds"], minlength=N_PAIR_FOLDS).tolist(),
            np.bincount(assignments, minlength=N_PAIR_FOLDS).tolist(),
        )

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "layer_00.npz"
            per_protein.write_coefficient_shard(path, 0, fits)
            loaded = per_protein.load_coefficient_shard(
                path, expected_layer=0,
                expected_responsibilities=[(record.protein_id, fold) for fold in range(5)],
            )
            self.assertEqual(np.asarray(loaded["coefficient_standardized"]).shape, (5, FEATURE_DIM))

    def test_launcher_has_locked_stage_four_resources_and_interface(self):
        launcher = (CV_DIR / "per_protein_ols_base256_layer.sbatch").read_text()
        required = (
            "#SBATCH --array=0-47%12", "#SBATCH --mem=12g",
            "#SBATCH --time=02:00:00", "#SBATCH --constraint=\"scratch\"",
            "#SBATCH --mail-type=BEGIN,END,FAIL",
            "#SBATCH --mail-user=ikhire3@gatech.edu",
            "SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}",
            'srun python3 -u "$SCRIPT_DIR/per_protein_ols_base256_layer.py"',
        )
        self.assertTrue(all(value in launcher for value in required))


class PerProteinOLSIntegrationResumeCorruptionTests(unittest.TestCase):
    def test_layer_family_aggregates_resumes_and_rejects_corruption(self):
        records = synthetic_records()
        bundle = SyntheticFoldBundle(records)
        load_counts = {record.protein_id: 0 for record in records}

        def counted_loader(record):
            load_counts[record.protein_id] += 1
            return np.asarray(record.synthetic_features, dtype=np.float32)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory) / "per_protein_ols_base256"
            per_protein.execute_layer_task(
                layer=0, records=records, fold_bundle=bundle,
                input_digest="synthetic-input-v1", root=root,
                feature_loader=counted_loader,
            )
            self.assertEqual(set(load_counts.values()), {1})

            per_protein.execute_layer_task(
                layer=0, records=records, fold_bundle=bundle,
                input_digest="synthetic-input-v1", root=root,
                feature_loader=lambda record: self.fail("resume reopened features"),
            )
            with self.assertRaisesRegex(ValueError, "all 48 layers"):
                aggregate.aggregate_per_protein_ols(
                    root=root, fold_bundle=bundle, layers=(0,),
                )
            aggregate.aggregate_per_protein_ols(
                root=root, fold_bundle=bundle, layers=(0,), allow_partial=True,
            )
            with (root / cv.PER_PROTEIN_FILENAME).open(newline="") as handle:
                per_rows = list(csv.DictReader(handle))
            with (root / cv.REPLAY_INDEX_FILENAME).open(newline="") as handle:
                replay_rows = list(csv.DictReader(handle))
            with (root / cv.STABLE_FIT_METRICS_FILENAME).open(newline="") as handle:
                stable_rows = list(csv.DictReader(handle))
            self.assertEqual(len(per_rows), 5)
            self.assertTrue(all(row["n_evaluated_pairs"] == "28" for row in per_rows))
            self.assertEqual(len(replay_rows), 25)
            self.assertEqual(len({row["artifact_path"] for row in replay_rows}), 1)
            self.assertEqual(
                len({(row["protein_id"], row["pair_fold"]) for row in replay_rows}), 25,
            )
            self.assertEqual(stable_rows[0]["n_proteins_included"], "0")
            manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(manifest["completed_task_ids"], [0])
            self.assertEqual(manifest["expected_replay_artifacts"], 25)
            self.assertFalse(manifest["raw_pair_predictions_saved"])

            runtime_path = per_protein.task_shard_path(root, 0) / "runtime.csv"
            runtime_path.write_text(runtime_path.read_text() + "corruption\n")
            with self.assertRaisesRegex(ValueError, "mismatch"):
                per_protein.execute_layer_task(
                    layer=0, records=records, fold_bundle=bundle,
                    input_digest="synthetic-input-v1", root=root,
                )
            with self.assertRaisesRegex(ValueError, "mismatch"):
                aggregate.validate_task_shard(root, 0, bundle)


if __name__ == "__main__":
    unittest.main()

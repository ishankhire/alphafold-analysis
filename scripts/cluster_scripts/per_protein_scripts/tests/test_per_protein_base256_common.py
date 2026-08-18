from __future__ import annotations

import csv
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np


ROOT = Path(__file__).resolve().parents[4]
CLUSTER_DIR = ROOT / "scripts" / "cluster_scripts"
PER_DIR = CLUSTER_DIR / "per_protein_scripts"
for path in (CLUSTER_DIR,):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

from scripts.cluster_scripts.per_protein_scripts import per_protein_base256_common as common
from scripts.cluster_scripts.per_protein_scripts import (
    aggregate_per_protein_base256_combined_shards as aggregate,
)
import regression_output_helpers as outputs
import regression_protocol as protocol


def synthetic_record(protein_id: str, protein_split: str) -> dict[str, object]:
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float64,
    )
    pair_i, pair_j = protocol.strict_lower_triangle_indices(4)
    y_true = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
    train_idx, heldout_idx = protocol.within_protein_pair_split(len(y_true), protein_id)
    return {
        "protein_id": protein_id,
        "source": "cameo" if protein_id == "dev" else "ecod",
        "master_role": "fit" if protein_split == "development" else "test",
        "protein_split": protein_split,
        "protein_dir": Path("/unused"),
        "n_residues": 4,
        "n_pairs": len(y_true),
        "size_bin": 0,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "coordinates": coordinates,
        "y_true": y_true,
        "train_idx": train_idx,
        "heldout_idx": heldout_idx,
        "sequence_codes": protocol.sequence_separation_bin_codes(pair_i, pair_j),
        "aa_codes": np.zeros(len(y_true), dtype=np.int16),
        "spatial_codes": protocol.spatial_distance_bin_codes(y_true),
        "split_seed": protocol.within_protein_split_seed(protein_id),
        "split_digest": protocol.within_protein_pair_split_digest(len(y_true), protein_id),
    }


def synthetic_features(record: dict[str, object], layer: int) -> np.ndarray:
    del layer
    n_pairs = int(record["n_pairs"])
    base = np.arange(n_pairs, dtype=np.float64)[:, None]
    channels = np.arange(protocol.FEATURE_DIM, dtype=np.float64)[None, :]
    return base + channels * 0.01


class SplitScalingAndStatusTests(unittest.TestCase):
    def test_training_scaler_excludes_heldout_rows(self):
        features = np.asarray([[0.0, 1.0], [2.0, 3.0], [1000.0, 1000.0]])
        mean, scale = common.training_scaler(features, np.asarray([0, 1]))
        np.testing.assert_allclose(mean, [1.0, 2.0])
        np.testing.assert_allclose(scale, [1.0, 1.0])

    def test_status_flags_and_selected_aggregation_policy(self):
        status, flags = common.fit_status_from_diagnostics(100, 200, 1e13)
        self.assertEqual(status, "fitted_underdetermined")
        self.assertEqual(flags, "underdetermined;rank_deficient;ill_conditioned")
        self.assertEqual(
            common.configuration_payload([0])["unstable_fit_aggregation_policy"],
            "include_with_flags",
        )

    def test_pair_manifest_digest_survives_csv_string_round_trip(self):
        rows = common.build_pair_split_rows(
            [synthetic_record("dev", "development"), synthetic_record("test", "test")]
        )
        string_rows = [{key: str(value) for key, value in row.items()} for row in rows]
        self.assertEqual(
            common.pair_split_manifest_digest(rows),
            common.pair_split_manifest_digest(string_rows),
        )


class FittingAndOutputTests(unittest.TestCase):
    def test_fit_replay_and_three_scope_rows(self):
        record = synthetic_record("dev", "development")
        with mock.patch.object(common, "build_base256_feature_matrix", synthetic_features):
            status, predictions, fitted = common.fit_single_protein_layer(record, 0)
        self.assertTrue(str(status["fit_status"]).startswith("fitted"))
        features = synthetic_features(record, 0)
        raw_replay = features @ fitted["coefficient_raw_scale"] + fitted["intercept_raw_scale"]
        np.testing.assert_allclose(raw_replay, predictions, rtol=1e-11, atol=1e-11)
        rows = common.build_scope_rows(record, 0, predictions)
        self.assertEqual([row["evaluation_scope"] for row in rows], list(common.EVALUATION_SCOPES))
        self.assertTrue(all(tuple(row) == outputs.PER_PROTEIN_COLUMNS for row in rows))
        self.assertEqual(rows[0]["n_expected_pairs"], len(record["train_idx"]))
        self.assertEqual(rows[1]["n_expected_pairs"], len(record["heldout_idx"]))
        self.assertEqual(rows[2]["n_expected_pairs"], record["n_pairs"])

    def test_failed_rows_are_explicit_and_have_zero_coverage(self):
        record = synthetic_record("dev", "development")
        predictions = np.full(record["n_pairs"], np.nan)
        rows = common.build_scope_rows(record, 0, predictions, prediction_status="fit_failed")
        self.assertTrue(all(row["prediction_status"] == "fit_failed" for row in rows))
        self.assertTrue(all(row["n_evaluated_pairs"] == 0 for row in rows))
        self.assertTrue(all(row["pair_coverage"] == 0.0 for row in rows))

    def test_synthetic_layer_and_consolidation(self):
        records = [synthetic_record("dev", "development"), synthetic_record("test", "test")]
        pair_rows = common.build_pair_split_rows(records)
        pair_digest = common.pair_split_manifest_digest(pair_rows)
        configuration = common.configuration_payload([0])
        configuration["pair_split_manifest_digest"] = pair_digest
        configuration["global_development_training_mean"] = (
            common.global_development_training_mean(records)
        )
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            outputs.atomic_write_csv(
                root / "metadata" / "pair_splits.csv", common.PAIR_SPLIT_COLUMNS, pair_rows,
            )
            outputs.atomic_write_csv(
                root / "metadata" / "preparation_failures.csv",
                common.PREPARATION_FAILURE_COLUMNS,
                [],
            )
            with mock.patch.object(common, "build_base256_feature_matrix", synthetic_features):
                failures = common.run_layer(
                    root=root,
                    layer=0,
                    records=records,
                    global_mean=common.global_development_training_mean(records),
                    pair_split_digest=pair_digest,
                    configuration=configuration,
                    checkpoint_every=1,
                )
            self.assertEqual(failures, 0)
            counts = aggregate.aggregate_result_family(
                root,
                expected_layers=[0],
                expected_proteins=2,
                require_canonical_coverage=False,
            )
            self.assertEqual(counts["per_protein.csv"], 6)
            self.assertEqual(counts["pooled_metrics.csv"], 6)
            self.assertEqual(counts["model_status.csv"], 2)
            self.assertEqual(counts["geometry_metrics.csv"], 2)
            self.assertEqual(counts["geometry_summary.csv"], 28)
            self.assertEqual(counts["subgroup_metrics/by_sequence_separation.csv"], 6)
            self.assertEqual(counts["subgroup_metrics/by_amino_acid_pair.csv"], 210)
            self.assertEqual(counts["subgroup_metrics/by_spatial_distance.csv"], 5)
            with (root / "per_protein.csv").open(newline="") as handle:
                per_rows = list(csv.DictReader(handle))
            test_heldout = [
                row for row in per_rows
                if row["protein_split"] == "test" and row["evaluation_scope"] == "heldout_pairs"
            ]
            self.assertEqual(len(test_heldout), 1)
            with np.load(root / "model_artifacts/coefficient_shards/layer_00.npz") as archive:
                self.assertEqual(archive["coefficient_standardized"].shape, (2, 256))
                self.assertEqual(archive["feature_scale"].shape, (2, 256))


if __name__ == "__main__":
    unittest.main()

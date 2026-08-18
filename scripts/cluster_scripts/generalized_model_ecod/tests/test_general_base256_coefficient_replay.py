from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np


REPLAY_DIR = Path(__file__).resolve().parents[1]
CLUSTER_DIR = REPLAY_DIR.parent
for path in (REPLAY_DIR, CLUSTER_DIR):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

import aggregate_general_base256_coefficient_replay as aggregate
import general_base256_coefficient_replay_common as common
import regression_output_helpers as outputs
import regression_protocol as protocol


def manifest_entry(protein_id: str, protein_split: str) -> protocol.ManifestEntry:
    role = "test" if protein_split == "test" else "fit"
    return protocol.ManifestEntry(
        manifest_version=protocol.MASTER_MANIFEST_VERSION,
        protein_id=protein_id,
        source="cameo" if protein_id == "dev" else "ecod",
        eligibility_status="eligible",
        exclusion_reason="",
        n_residues=4,
        n_pairs=6,
        size_bin=0,
        master_role=role,
        protein_split=protein_split,
        outer_split_seed=protocol.OUTER_SPLIT_SEED,
        development_split_seed=protocol.DEVELOPMENT_SPLIT_SEED,
    )


def synthetic_record(entry: protocol.ManifestEntry) -> dict[str, object]:
    coordinates = np.asarray(
        [[0.0, 0.0, 0.0], [1.0, 0.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]],
        dtype=np.float64,
    )
    pair_i, pair_j = protocol.strict_lower_triangle_indices(4)
    y_true = np.linalg.norm(coordinates[pair_i] - coordinates[pair_j], axis=1)
    return {
        "protein_id": entry.protein_id,
        "source": entry.source,
        "master_role": entry.master_role,
        "protein_split": entry.protein_split,
        "n_residues": 4,
        "n_pairs": 6,
        "size_bin": 0,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "coordinates": coordinates,
        "residues": (
            ("A", "ALA", 1), ("A", "ARG", 2),
            ("A", "ASN", 3), ("A", "ASP", 4),
        ),
        "y_true": y_true,
    }


def synthetic_features(record: dict[str, object], layer: int) -> np.ndarray:
    base = np.arange(int(record["n_pairs"]), dtype=np.float64)[:, None]
    channels = np.arange(protocol.FEATURE_DIM, dtype=np.float64)[None, :]
    offset = 0.25 if record["protein_split"] == "test" else 0.0
    return base * 0.01 + channels * 0.0001 + layer + offset


def write_legacy_coefficients(path: Path, *, duplicate: bool = False) -> None:
    rows = []
    for index in (-1, *range(protocol.FEATURE_DIM)):
        name = (
            "intercept" if index == -1
            else f"pair_ij_ch_{index}" if index < 128
            else f"pair_ji_ch_{index - 128}"
        )
        rows.append({
            "variant": common.ANALYSIS_NAME,
            "feature_dim": protocol.FEATURE_DIM,
            "split_id": 0,
            "layer": 0,
            "feature_index": index,
            "feature_name": name,
            "coefficient_standardized": 1.0 if index == -1 else index / 1000.0,
            "coefficient_raw_scale": 2.0 if index == -1 else (index + 1) / 10000.0,
        })
    if duplicate:
        rows[-1]["feature_index"] = 254
        rows[-1]["feature_name"] = "pair_ji_ch_126"
    outputs.atomic_write_csv(path, common.LEGACY_COEFFICIENT_COLUMNS, rows)


def exact_legacy_summary(
    per_rows: list[dict[str, object]], pooled_rows: list[dict[str, object]],
) -> dict[str, str]:
    result: dict[str, str] = {}
    pooled = {row["protein_split"]: row for row in pooled_rows}
    for split, prefix in (("development", "train"), ("test", "test")):
        local = [row for row in per_rows if row["protein_split"] == split]
        result[f"{prefix}_mean_per_protein_r2"] = str(np.mean([float(row["r2"]) for row in local]))
        result[f"{prefix}_pair_weighted_r2"] = str(pooled[split]["pooled_r2"])
        result[f"{prefix}_pair_weighted_mae"] = str(pooled[split]["pooled_mae"])
        result[f"{prefix}_pair_weighted_rmse"] = str(pooled[split]["pooled_rmse"])
    return result


class InputValidationTests(unittest.TestCase):
    def test_coefficient_parser_normalizes_feature_names_and_preserves_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coefficients.csv"
            write_legacy_coefficients(path)
            bundle = common.load_coefficient_bundle(path, expected_layers=[0])
        layer = bundle.by_layer[0]
        self.assertEqual(layer.intercept_raw, 2.0)
        self.assertEqual(layer.coefficient_raw.shape, (256,))
        self.assertEqual(layer.canonical_rows[1]["feature_name"], "pair_ij_ch_000")
        self.assertEqual(layer.canonical_rows[-1]["feature_name"], "pair_ji_ch_127")
        self.assertEqual(layer.canonical_rows[-1]["coefficient_raw_scale"], 0.0256)

    def test_coefficient_parser_rejects_duplicate_or_missing_index(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "coefficients.csv"
            write_legacy_coefficients(path, duplicate=True)
            with self.assertRaisesRegex(ValueError, "Duplicate coefficient index"):
                common.load_coefficient_bundle(path, expected_layers=[0])

    def test_historical_split_must_match_canonical_outer_membership(self):
        entries = (manifest_entry("dev", "development"), manifest_entry("test", "test"))
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "split_assignments.csv"
            rows = [{
                "split_id": 0,
                "split": "train" if entry.protein_split == "development" else "test",
                "protein_id": entry.protein_id,
                "n_residues": entry.n_residues,
                "size_bin": entry.size_bin,
                "n_pairs": entry.n_pairs,
                "target_variance": 1.0,
            } for entry in entries]
            outputs.atomic_write_csv(path, common.LEGACY_SPLIT_COLUMNS, rows)
            bundle = common.load_historical_split_bundle(path, entries=entries)
            self.assertEqual(bundle.split_by_id, {"dev": "development", "test": "test"})
            rows[0]["split"] = "test"
            outputs.atomic_write_csv(path, common.LEGACY_SPLIT_COLUMNS, rows)
            with self.assertRaisesRegex(ValueError, "outer split mismatch"):
                common.load_historical_split_bundle(path, entries=entries)


class ReplayAndOutputTests(unittest.TestCase):
    def setUp(self):
        self.entries = (manifest_entry("dev", "development"), manifest_entry("test", "test"))
        self.records = [synthetic_record(entry) for entry in self.entries]
        raw = np.linspace(-0.02, 0.03, protocol.FEATURE_DIM, dtype=np.float64)
        rows = []
        for index in (-1, *range(protocol.FEATURE_DIM)):
            value = 1.5 if index == -1 else float(raw[index])
            rows.append({
                "analysis_name": common.ANALYSIS_NAME,
                "model_family": common.MODEL_FAMILY,
                "layer": 0,
                "protein_id": "",
                "gate_aa_pair": "",
                "gate_sequence_bin": "",
                "feature_index": index,
                "feature_name": "intercept" if index == -1 else protocol.FEATURE_NAMES[index],
                "coefficient_standardized": value,
                "coefficient_raw_scale": value,
            })
        raw.setflags(write=False)
        self.model = common.LayerCoefficients(0, 1.5, raw, tuple(rows))

    def test_raw_replay_matches_dense_formula(self):
        features = synthetic_features(self.records[0], 0)
        observed = common.predict_raw(features, self.model)
        expected = 1.5 + features @ self.model.coefficient_raw
        np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)

    def test_canonical_tables_and_legacy_tolerance(self):
        global_mean = protocol.protein_equal_mean([
            np.asarray(self.records[0]["y_true"], dtype=np.float64)
        ])
        tables = common.evaluate_records(
            self.records, 0, self.model, global_mean, feature_loader=synthetic_features,
        )
        self.assertEqual(len(tables["per_protein.csv"]), 2)
        self.assertEqual(len(tables["pooled_metrics.csv"]), 2)
        self.assertEqual(len(tables["geometry_metrics.csv"]), 2)
        self.assertEqual(len(tables["geometry_summary.csv"]), 28)
        self.assertTrue(all(tuple(row) == outputs.PER_PROTEIN_COLUMNS for row in tables["per_protein.csv"]))
        legacy = exact_legacy_summary(tables["per_protein.csv"], tables["pooled_metrics.csv"])
        comparisons = common.legacy_comparison_rows(
            0, tables["per_protein.csv"], tables["pooled_metrics.csv"], legacy,
        )
        self.assertEqual(len(comparisons), 8)
        self.assertTrue(all(row["comparison_status"] == "pass" for row in comparisons))
        legacy["test_pair_weighted_r2"] = str(float(legacy["test_pair_weighted_r2"]) + 0.01)
        failed = common.legacy_comparison_rows(
            0, tables["per_protein.csv"], tables["pooled_metrics.csv"], legacy,
        )
        self.assertTrue(any(row["comparison_status"] == "fail" for row in failed))

    def test_synthetic_shard_consolidation_and_checksum_rejection(self):
        global_mean = protocol.protein_equal_mean([
            np.asarray(self.records[0]["y_true"], dtype=np.float64)
        ])
        tables = common.evaluate_records(
            self.records, 0, self.model, global_mean, feature_loader=synthetic_features,
        )
        legacy = exact_legacy_summary(tables["per_protein.csv"], tables["pooled_metrics.csv"])
        comparisons = common.legacy_comparison_rows(
            0, tables["per_protein.csv"], tables["pooled_metrics.csv"], legacy,
        )
        status = outputs.build_model_status_row(
            analysis_name=common.ANALYSIS_NAME,
            model_family=common.MODEL_FAMILY,
            feature_dim=protocol.FEATURE_DIM,
            layer=0,
            fit_status="historical_fit_replayed",
            support_flags=(
                "historical_coefficients_six_decimal;design_diagnostics_unavailable;"
                "historical_p2_p5_p8_deviations"
            ),
            n_training_proteins=1,
            n_training_pairs=6,
        )
        runtime = outputs.build_runtime_row(
            analysis_name=common.ANALYSIS_NAME,
            model_family=common.MODEL_FAMILY,
            layer=0,
            completion_status="complete",
            n_fitted_proteins=1,
            n_evaluated_proteins=2,
            n_fitted_pairs=6,
            n_evaluated_pairs=12,
            elapsed_seconds=1.0,
        )
        tables.update({
            "model_status.csv": [status],
            "runtime.csv": [runtime],
            "model_artifacts/coefficients.csv": list(self.model.canonical_rows),
            "metadata/legacy_metric_comparison.csv": comparisons,
        })
        with tempfile.TemporaryDirectory() as directory:
            temp = Path(directory)
            coefficient_path = temp / "source_coefficients.csv"
            split_path = temp / "source_split.csv"
            coefficient_path.write_text("synthetic\n")
            split_path.write_text("synthetic\n")
            coefficients = common.CoefficientBundle(
                coefficient_path, outputs.sha256_file(coefficient_path), {0: self.model},
            )
            historical = common.HistoricalSplitBundle(
                split_path,
                outputs.sha256_file(split_path),
                "synthetic-split-digest",
                {"dev": "development", "test": "test"},
            )
            root = temp / "canonical"
            common.write_layer_shard(
                root,
                0,
                tables,
                invariants={
                    "output_schema_version": outputs.OUTPUT_SCHEMA_VERSION,
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "master_manifest_digest": protocol.MASTER_MANIFEST_DIGEST,
                    "historical_split_digest": historical.semantic_digest,
                    "coefficient_source_sha256": coefficients.sha256,
                    "historical_split_source_sha256": historical.sha256,
                    "layer_input_digest": "synthetic-input",
                    "code_digest": common.code_digest(),
                },
                replay={
                    "protein_id": "dev",
                    "pair_indices": [0],
                    "feature_sha256": "unused",
                    "predictions": [0.0],
                    "absolute_tolerance": 1e-10,
                },
            )
            # Rewriting a completed shard must replace its manifest cleanly.
            common.write_layer_shard(
                root,
                0,
                tables,
                invariants={
                    "output_schema_version": outputs.OUTPUT_SCHEMA_VERSION,
                    "protocol_version": protocol.PROTOCOL_VERSION,
                    "master_manifest_digest": protocol.MASTER_MANIFEST_DIGEST,
                    "historical_split_digest": historical.semantic_digest,
                    "coefficient_source_sha256": coefficients.sha256,
                    "historical_split_source_sha256": historical.sha256,
                    "layer_input_digest": "synthetic-input",
                    "code_digest": common.code_digest(),
                },
                replay={
                    "protein_id": "dev",
                    "pair_indices": [0],
                    "feature_sha256": "unused",
                    "predictions": [0.0],
                    "absolute_tolerance": 1e-10,
                },
            )
            counts = aggregate.aggregate_result_family(
                root,
                expected_layers=[0],
                entries=self.entries,
                coefficients=coefficients,
                historical_split=historical,
                verify_replay=False,
            )
            self.assertEqual(counts["per_protein.csv"], 2)
            self.assertEqual(counts["model_artifacts/coefficients.csv"], 257)
            manifest = json.loads((root / "run_manifest.json").read_text())
            self.assertEqual(manifest["output_contract_status"], "complete_with_unavailable_historical_fit_diagnostics")
            self.assertEqual(manifest["completed_layers"], [0])
            with self.assertRaisesRegex(ValueError, "duplicates"):
                aggregate.aggregate_result_family(
                    root,
                    expected_layers=[0, 0],
                    entries=self.entries,
                    coefficients=coefficients,
                    historical_split=historical,
                    verify_replay=False,
                )
            coefficient_shard = root / "layer_shards" / "layer_00" / "model_artifacts" / "coefficients.csv"
            with coefficient_shard.open("a") as handle:
                handle.write("corruption\n")
            with self.assertRaisesRegex(ValueError, "checksum mismatch"):
                aggregate.validate_layer_shard(
                    root, 0, self.entries, coefficients, historical,
                )


if __name__ == "__main__":
    unittest.main()

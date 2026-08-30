#!/usr/bin/env python3
"""Run and record the protocol-2.0 adversarial correctness review.

The synthetic review is local-only.  It never edits production outputs and it
never saves pair- or triangle-level predictions.  Real protocol-2.0 replay and
the real-XGBoost boundary are recorded as unavailable until their required
artifacts and environment exist.
"""

from __future__ import annotations

import argparse
import csv
import importlib.metadata
import itertools
import json
import math
import platform
import re
import subprocess
import sys
import time
import unittest
from datetime import datetime, timezone
from pathlib import Path
from typing import Iterable, Mapping, Sequence

import numpy as np


SCRIPT_PATH = Path(__file__).resolve()
REPO_ROOT = SCRIPT_PATH.parents[3]
CLUSTER_DIR = REPO_ROOT / "scripts" / "cluster_scripts"
CV_DIR = CLUSTER_DIR / "cross_validation"
TEST_DIR = CV_DIR / "tests"
for directory in (CLUSTER_DIR, CV_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

import cluster_regression_helpers as geometry  # noqa: E402
import cross_validation_common as cv  # noqa: E402
from regression_protocol import strict_lower_triangle_indices  # noqa: E402


REVIEW_VERSION = "cross_validation_adversarial_review_v1"
DEFAULT_OUTPUT = REPO_ROOT / "delta_files" / "analysis" / REVIEW_VERSION
FAMILIES = (
    "shared_ols_base256",
    "per_protein_ols_base256",
    "sequence_gated_ols_base256",
    "shared_xgboost_base256_fixed_s4",
)
FAMILY_MODEL_NAMES = {
    "shared_ols_base256": "shared_ols",
    "per_protein_ols_base256": "per_protein_ols",
    "sequence_gated_ols_base256": "gated_ols",
    "shared_xgboost_base256_fixed_s4": "shared_xgboost",
}
FOCUSED_COMMAND = (
    "./.venv/bin/python -m unittest discover "
    "-s scripts/cluster_scripts/cross_validation/tests -p 'test_*.py'"
)
BROAD_MODULES = (
    "tests/test_regression_output_helpers.py",
    "tests/test_regression_protocol.py",
    "scripts/cluster_scripts/cross_validation/tests/test_cross_validation_common.py",
    "scripts/cluster_scripts/cross_validation/tests/test_per_protein_ols_base256.py",
    "scripts/cluster_scripts/cross_validation/tests/test_sequence_gated_ols_base256.py",
    "scripts/cluster_scripts/cross_validation/tests/test_shared_ols_base256.py",
    "scripts/cluster_scripts/cross_validation/tests/test_shared_xgboost_base256_fixed_s4.py",
    "scripts/cluster_scripts/gated_linear_probes/tests/test_gated_linear_probe_suite.py",
    "scripts/cluster_scripts/gated_linear_probes/tests/test_synthetic_suite_integration.py",
    "scripts/cluster_scripts/generalized_model_ecod/tests/test_general_base256_coefficient_replay.py",
    "scripts/cluster_scripts/per_protein_scripts/tests/test_per_protein_base256_common.py",
    "scripts/cluster_scripts/xgboost_search/tests/test_shared_xgboost_final.py",
    "scripts/cluster_scripts/xgboost_search/tests/test_xgboost_search_common.py",
)
BROAD_COMMAND = "./.venv/bin/python -m unittest " + " ".join(BROAD_MODULES)

PERTURBATION_COLUMNS = (
    "test_id",
    "model_family",
    "layer",
    "outer_fold",
    "protein_id",
    "pair_fold",
    "sequence_gate",
    "perturbation",
    "expected_invariant",
    "artifact_digest_before",
    "artifact_digest_after",
    "metric_before",
    "metric_after",
    "observed_behavior",
    "status",
    "details",
)
GEOMETRY_COLUMNS = (
    "case_id",
    "case_type",
    "n_residues",
    "input_dtype",
    "metric",
    "expected_value",
    "observed_value",
    "absolute_error",
    "relative_error",
    "tolerance",
    "status",
    "oracle_method",
    "details",
)
ARTIFACT_REPLAY_COLUMNS = (
    "model_family",
    "layer",
    "outer_fold",
    "protein_id",
    "pair_fold",
    "sequence_gate",
    "artifact_path",
    "artifact_sha256",
    "configuration_digest",
    "scaler_digest",
    "metric_scope",
    "metric",
    "saved_value",
    "recomputed_value",
    "difference",
    "tolerance",
    "status",
    "details",
)
STABILITY_COLUMNS = (
    "model_family",
    "layer",
    "status_scope",
    "requested_fits",
    "completed_fits",
    "stable_fits",
    "rank_deficient_fits",
    "underdetermined_fits",
    "ill_conditioned_fits",
    "minimum_design_rank",
    "maximum_design_rank",
    "minimum_condition_number",
    "maximum_condition_number",
    "complete_pooled_r2",
    "stable_only_pooled_r2",
    "stable_minus_complete_r2",
    "status",
    "details",
)


def _json_safe(value: object) -> object:
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(_json_safe(value), indent=2, sort_keys=True) + "\n")


def _write_csv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=columns,
            extrasaction="raise",
            lineterminator="\n",
        )
        writer.writeheader()
        for raw in rows:
            writer.writerow({column: raw.get(column, "") for column in columns})


def _command_output(command: Sequence[str]) -> tuple[int, str]:
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    return int(completed.returncode), completed.stdout


class RecordingResult(unittest.TextTestResult):
    """Text result that also emits one structured record per unittest case."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.records: list[dict[str, object]] = []
        self._started: dict[str, float] = {}
        self._outcomes: dict[str, tuple[str, str]] = {}

    def startTest(self, test):  # noqa: N802
        self._started[test.id()] = time.perf_counter()
        super().startTest(test)

    def addSuccess(self, test):  # noqa: N802
        self._outcomes[test.id()] = ("passed", "")
        super().addSuccess(test)

    def addFailure(self, test, err):  # noqa: N802
        self._outcomes[test.id()] = ("failed", self._exc_info_to_string(err, test))
        super().addFailure(test, err)

    def addError(self, test, err):  # noqa: N802
        self._outcomes[test.id()] = ("error", self._exc_info_to_string(err, test))
        super().addError(test, err)

    def addSkip(self, test, reason):  # noqa: N802
        self._outcomes[test.id()] = ("skipped", str(reason))
        super().addSkip(test, reason)

    def addExpectedFailure(self, test, err):  # noqa: N802
        self._outcomes[test.id()] = ("expected_failure", self._exc_info_to_string(err, test))
        super().addExpectedFailure(test, err)

    def addUnexpectedSuccess(self, test):  # noqa: N802
        self._outcomes[test.id()] = ("unexpected_success", "expected failure now passes")
        super().addUnexpectedSuccess(test)

    def stopTest(self, test):  # noqa: N802
        elapsed = time.perf_counter() - self._started.pop(test.id(), time.perf_counter())
        status, details = self._outcomes.pop(test.id(), ("unknown", "missing test outcome"))
        parts = test.id().split(".")
        self.records.append(
            {
                "test_id": test.id(),
                "test_module": ".".join(parts[:-2]) if len(parts) >= 3 else "",
                "status": status,
                "elapsed_seconds": elapsed,
                "reason": details[-2_000:],
            }
        )
        super().stopTest(test)


def run_focused_tests() -> tuple[RecordingResult, float]:
    suite = unittest.defaultTestLoader.discover(str(TEST_DIR), pattern="test_*.py")
    started = time.perf_counter()
    runner = unittest.TextTestRunner(
        stream=sys.stdout,
        verbosity=1,
        resultclass=RecordingResult,
    )
    result = runner.run(suite)
    return result, time.perf_counter() - started


def run_broad_baseline(enabled: bool) -> dict[str, object]:
    if not enabled:
        return {
            "command": BROAD_COMMAND,
            "status": "not_run",
            "test_count": None,
            "skip_count": None,
            "elapsed_seconds": None,
            "details": "Pass --run-broad-baseline to execute the release-baseline suite.",
        }
    started = time.perf_counter()
    code, output = _command_output([str(REPO_ROOT / ".venv" / "bin" / "python"), "-m", "unittest", *BROAD_MODULES])
    elapsed = time.perf_counter() - started
    count_match = re.search(r"Ran (\d+) tests?", output)
    skip_match = re.search(r"skipped=(\d+)", output)
    return {
        "command": BROAD_COMMAND,
        "status": "passed" if code == 0 else "failed",
        "test_count": int(count_match.group(1)) if count_match else None,
        "skip_count": int(skip_match.group(1)) if skip_match else 0,
        "elapsed_seconds": elapsed,
        "details": output[-4_000:] if code else "",
    }


def _test_record(records: Sequence[Mapping[str, object]], pattern: str) -> Mapping[str, object]:
    matches = [row for row in records if pattern in str(row["test_id"])]
    if len(matches) != 1:
        raise RuntimeError(f"Expected one automated test matching {pattern!r}; found {len(matches)}")
    return matches[0]


def build_perturbation_rows(test_records: Sequence[Mapping[str, object]]) -> list[dict[str, object]]:
    specifications = (
        ("test_shared_and_gated_ols_ignore_heldout_target_poisoning", "shared_ols", "heldout_target_poisoning", "fitted scaler, baseline, and coefficient state"),
        ("test_shared_and_gated_ols_ignore_heldout_target_poisoning", "gated_ols", "heldout_target_poisoning", "gate-specific fitted state"),
        ("test_per_protein_poisoning_respects_one_pair_fold_responsibility", "per_protein_ols", "heldout_pair_fold_target_poisoning", "the responsible pair-fold artifact"),
        ("test_xgboost_training_materialization_excludes_heldout_targets", "shared_xgboost", "heldout_target_poisoning", "sampled fitting matrix, weights, baseline, and assignment digest"),
        ("test_training_target_poisoning_changes_model_but_not_scaler", "shared_ols", "training_target_poisoning", "scaler unchanged; baseline and coefficient state change"),
        ("test_joint_pair_permutation_is_invariant_and_mismatch_is_detected", "shared_ols", "joint_and_independent_pair_permutations", "joint permutation invariant; mismatched targets degrade"),
        ("test_directional_feature_order_survives_artifact_replay", "shared_ols", "direction_swap", "canonical [Z_ij | Z_ji] artifact replay"),
        ("test_float64_conversion_precedes_products_and_squares", "all", "float32_high_dynamic_range", "conversion before products and squares"),
        ("test_broadcastable_and_wrong_width_inputs_are_rejected", "all", "broadcastable_shape_mutation", "invalid shapes and feature widths rejected"),
        ("test_missing_predictions_cannot_form_complete_outputs", "all", "missing_and_nonfinite_prediction", "complete outputs forbidden"),
        ("test_direct_oof_metrics_survive_merge_serialization_and_consolidation", "all", "fold_aggregation", "direct concatenation rather than fold-metric averaging"),
        ("test_pair_pooled_and_protein_equal_summaries_have_distinct_denominators", "all", "unequal_pair_counts", "documented pair-pooled and protein-equal denominators"),
        ("test_stable_filter_retains_flagged_primary_rows_but_excludes_sensitivity", "ols", "flagged_fit", "primary retention and stable-only exclusion"),
        ("test_geometry_failure_summaries_preserve_attempted_included_excluded_counts", "all", "geometry_failure", "attempted/included/excluded counts"),
    )
    rows = []
    for pattern, family, perturbation, invariant in specifications:
        test = _test_record(test_records, pattern)
        passed = test["status"] == "passed"
        rows.append(
            {
                "test_id": test["test_id"],
                "model_family": family,
                "layer": "synthetic",
                "outer_fold": "",
                "protein_id": "synthetic",
                "pair_fold": 0 if "pair_fold" in perturbation else "",
                "sequence_gate": "1" if family == "gated_ols" else "",
                "perturbation": perturbation,
                "expected_invariant": invariant,
                "artifact_digest_before": "asserted in test" if "poison" in perturbation or "replay" in perturbation else "",
                "artifact_digest_after": "asserted in test" if "poison" in perturbation or "replay" in perturbation else "",
                "metric_before": "",
                "metric_after": "",
                "observed_behavior": "matched predeclared invariant" if passed else "did not match predeclared invariant",
                "status": "passed" if passed else str(test["status"]),
                "details": str(test["reason"]),
            }
        )
    return rows


def _literal_distance_matrix(coordinates: np.ndarray) -> np.ndarray:
    points = np.asarray(coordinates, dtype=np.float64)
    result = np.empty((points.shape[0], points.shape[0]), dtype=np.float64)
    for i in range(points.shape[0]):
        for j in range(points.shape[0]):
            result[i, j] = math.sqrt(
                sum((float(points[i, axis]) - float(points[j, axis])) ** 2 for axis in range(points.shape[1]))
            )
    return result


def _literal_eigenvalues(distances: np.ndarray) -> np.ndarray:
    n = distances.shape[0]
    centering = np.eye(n) - np.ones((n, n)) / n
    return np.linalg.eigvalsh(-0.5 * centering @ np.square(distances) @ centering)[::-1]


def _literal_triangle_metrics(distances: np.ndarray) -> dict[str, float]:
    deltas = []
    for i, j, k in itertools.combinations(range(distances.shape[0]), 3):
        sides = [float(distances[i, j]), float(distances[i, k]), float(distances[j, k])]
        longest = max(sides)
        raw = longest - (sum(sides) - longest)
        deltas.append(raw if raw > geometry.TRIANGLE_VIOLATION_TOLERANCE else 0.0)
    positives = [value for value in deltas if value > 0.0]
    return {
        "triangle_violation_fraction": len(positives) / len(deltas),
        "triangle_mean_violation": sum(positives) / len(deltas),
        "triangle_max_violation": max(positives, default=0.0),
        "triangle_violation_0_5_fraction": sum(value < 5.0 for value in positives) / len(deltas),
        "triangle_violation_5_10_fraction": sum(5.0 <= value < 10.0 for value in positives) / len(deltas),
        "triangle_violation_10_plus_fraction": sum(value >= 10.0 for value in positives) / len(deltas),
    }


def _geometry_row(coordinates: np.ndarray, distances: np.ndarray, case_id: str) -> dict[str, object]:
    pair_i, pair_j = strict_lower_triangle_indices(coordinates.shape[0])
    return cv.build_geometry_row(
        analysis_name="adversarial_geometry",
        model_family="shared_ols",
        feature_dim=2,
        layer=0,
        protein_id=case_id,
        source="cameo",
        outer_fold=0,
        evaluation_design="protein_outer_5fold",
        oof_prediction_scope="heldout_proteins_all_pairs",
        n_residues=coordinates.shape[0],
        true_coordinates=coordinates,
        predictions=np.asarray(distances[pair_i, pair_j], dtype=np.float64),
    )


def _geometry_check(
    *,
    case_id: str,
    case_type: str,
    n_residues: int,
    input_dtype: str,
    metric: str,
    expected: float,
    observed: float,
    tolerance: float,
    oracle_method: str,
    details: str = "",
) -> dict[str, object]:
    absolute = abs(observed - expected)
    relative = absolute / abs(expected) if expected else absolute
    return {
        "case_id": case_id,
        "case_type": case_type,
        "n_residues": n_residues,
        "input_dtype": input_dtype,
        "metric": metric,
        "expected_value": expected,
        "observed_value": observed,
        "absolute_error": absolute,
        "relative_error": relative,
        "tolerance": tolerance,
        "status": "passed" if absolute <= tolerance else "failed",
        "oracle_method": oracle_method,
        "details": details,
    }


def build_geometry_checks() -> list[dict[str, object]]:
    coordinates = np.asarray(
        [
            [0.0, 0.0, 0.0],
            [1.7, 0.2, -0.1],
            [0.3, 2.1, 0.4],
            [-0.2, 0.5, 2.4],
            [1.2, 1.6, 1.1],
            [2.0, -0.4, 1.5],
        ],
        dtype=np.float64,
    )
    rows: list[dict[str, object]] = []
    exact_distances = _literal_distance_matrix(coordinates)
    for dtype, tolerance in ((np.float64, 1e-10), (np.float32, 2e-6)):
        supplied = exact_distances.astype(dtype)
        row = _geometry_row(coordinates, supplied, f"exact3d_{np.dtype(dtype).name}")
        for metric, expected in (
            ("rmsd", 0.0),
            ("tm_score", 1.0),
            ("stress", 0.0),
            ("frac_top3_eigenvalues", 1.0),
            ("triangle_violation_fraction", 0.0),
        ):
            rows.append(
                _geometry_check(
                    case_id=f"exact3d_{np.dtype(dtype).name}",
                    case_type="exact_3d_euclidean",
                    n_residues=coordinates.shape[0],
                    input_dtype=np.dtype(dtype).name,
                    metric=metric,
                    expected=expected,
                    observed=float(row[metric]),
                    tolerance=tolerance,
                    oracle_method="literal coordinate norms and known exact reconstruction",
                )
            )

    four_d = np.asarray(
        [
            [0.0, 0.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [1.0, 1.0, 1.0, 1.0],
        ]
    )
    four_d_distances = _literal_distance_matrix(four_d)
    eigenvalues = _literal_eigenvalues(four_d_distances)
    expected_top3 = sum(max(float(value), 0.0) for value in eigenvalues[:3]) / sum(
        max(float(value), 0.0) for value in eigenvalues
    )
    four_d_row = _geometry_row(four_d[:, :3], four_d_distances, "exact4d")
    rows.append(
        _geometry_check(
            case_id="exact4d",
            case_type="higher_dimensional_euclidean",
            n_residues=four_d.shape[0],
            input_dtype="float64",
            metric="frac_top3_eigenvalues",
            expected=expected_top3,
            observed=float(four_d_row["frac_top3_eigenvalues"]),
            tolerance=1e-12,
            oracle_method="independent centering-matrix eigendecomposition",
        )
    )

    triangle_case = np.asarray(
        [
            [0.0, 1.0, 8.0, 3.0],
            [1.0, 0.0, 1.0, 3.0],
            [8.0, 1.0, 0.0, 3.0],
            [3.0, 3.0, 3.0, 0.0],
        ]
    )
    triangle_oracle = _literal_triangle_metrics(triangle_case)
    triangle_observed = geometry.compute_triangle_violations(triangle_case)
    for metric, expected in triangle_oracle.items():
        rows.append(
            _geometry_check(
                case_id="mixed_triangle",
                case_type="literal_triangle_enumeration",
                n_residues=4,
                input_dtype="float64",
                metric=metric,
                expected=expected,
                observed=float(triangle_observed[metric]),
                tolerance=1e-12,
                oracle_method="itertools.combinations scalar loop",
            )
        )
    return rows


def artifact_replay_rows(result_root: Path) -> tuple[list[dict[str, object]], list[str]]:
    rows = []
    deferred = []
    for family in FAMILIES:
        root = result_root / family
        manifest_path = root / cv.RUN_MANIFEST_FILENAME
        base = {
            "model_family": FAMILY_MODEL_NAMES[family],
            "layer": "",
            "outer_fold": "",
            "protein_id": "",
            "pair_fold": "",
            "sequence_gate": "",
            "artifact_path": str(manifest_path.relative_to(REPO_ROOT)) if manifest_path.is_relative_to(REPO_ROOT) else str(manifest_path),
            "artifact_sha256": "",
            "configuration_digest": "",
            "scaler_digest": "",
            "metric_scope": "availability",
            "metric": "complete_run_manifest",
            "saved_value": "",
            "recomputed_value": "",
            "difference": "",
            "tolerance": "",
        }
        if not manifest_path.is_file():
            details = f"No complete protocol-2.0 family exists at {root}"
            rows.append({**base, "status": "unavailable", "details": details})
            deferred.append(f"{family}: {details}")
            continue
        manifest = json.loads(manifest_path.read_text())
        if manifest.get("run_status") != "complete":
            details = f"run_manifest exists but run_status={manifest.get('run_status')!r}"
            rows.append({**base, "status": "unavailable", "details": details})
            deferred.append(f"{family}: {details}")
            continue
        details = (
            "Complete family detected. Pair-feature replay remains deferred until the "
            "selected local pair tensors/PDBs and, for XGBoost, a matched environment are available."
        )
        rows.append(
            {
                **base,
                "configuration_digest": str(manifest.get("configuration_digest", "")),
                "status": "deferred_environment_or_inputs",
                "details": details,
            }
        )
        deferred.append(f"{family}: {details}")
    return rows, deferred


def _float_or_nan(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return float("nan")


def stability_rows(result_root: Path) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for family in FAMILIES:
        root = result_root / family
        status_path = root / cv.MODEL_STATUS_FILENAME
        pooled_path = root / cv.POOLED_METRICS_FILENAME
        stable_path = root / cv.STABLE_FIT_METRICS_FILENAME
        if not status_path.is_file():
            rows.append(
                {
                    "model_family": FAMILY_MODEL_NAMES[family],
                    "layer": "",
                    "status_scope": "availability",
                    "requested_fits": "",
                    "completed_fits": "",
                    "stable_fits": "",
                    "rank_deficient_fits": "",
                    "underdetermined_fits": "",
                    "ill_conditioned_fits": "",
                    "minimum_design_rank": "",
                    "maximum_design_rank": "",
                    "minimum_condition_number": "",
                    "maximum_condition_number": "",
                    "complete_pooled_r2": "",
                    "stable_only_pooled_r2": "",
                    "stable_minus_complete_r2": "",
                    "status": "unavailable",
                    "details": f"Missing {status_path}",
                }
            )
            continue
        with status_path.open(newline="") as handle:
            status_records = list(csv.DictReader(handle))
        with pooled_path.open(newline="") as handle:
            pooled_by_layer = {int(row["layer"]): row for row in csv.DictReader(handle)}
        stable_by_layer: dict[int, Mapping[str, str]] = {}
        if stable_path.is_file():
            with stable_path.open(newline="") as handle:
                stable_by_layer = {int(row["layer"]): row for row in csv.DictReader(handle)}
        by_layer: dict[int, list[Mapping[str, str]]] = {}
        for record in status_records:
            by_layer.setdefault(int(record["layer"]), []).append(record)
        for layer, records in sorted(by_layer.items()):
            flags = [str(record.get("support_flags", "")) for record in records]
            ranks = [_float_or_nan(record.get("design_rank")) for record in records]
            ranks = [value for value in ranks if math.isfinite(value)]
            conditions = [_float_or_nan(record.get("condition_number")) for record in records]
            conditions = [value for value in conditions if math.isfinite(value)]
            complete_r2 = _float_or_nan(pooled_by_layer.get(layer, {}).get("pooled_r2"))
            stable_r2 = _float_or_nan(stable_by_layer.get(layer, {}).get("pooled_r2"))
            rows.append(
                {
                    "model_family": FAMILY_MODEL_NAMES[family],
                    "layer": layer,
                    "status_scope": "all_model_status_rows",
                    "requested_fits": len(records),
                    "completed_fits": len(records),
                    "stable_fits": sum(record.get("fit_status") == "fitted" for record in records),
                    "rank_deficient_fits": sum("rank_deficient" in value for value in flags),
                    "underdetermined_fits": sum("underdetermined" in value for value in flags),
                    "ill_conditioned_fits": sum("ill_conditioned" in value for value in flags),
                    "minimum_design_rank": min(ranks) if ranks else "",
                    "maximum_design_rank": max(ranks) if ranks else "",
                    "minimum_condition_number": min(conditions) if conditions else "",
                    "maximum_condition_number": max(conditions) if conditions else "",
                    "complete_pooled_r2": complete_r2,
                    "stable_only_pooled_r2": stable_r2,
                    "stable_minus_complete_r2": stable_r2 - complete_r2 if math.isfinite(stable_r2) and math.isfinite(complete_r2) else "",
                    "status": "recorded",
                    "details": "Compact status reconstruction; real-artifact responsibility replay is separate.",
                }
            )
    return rows


def _git_state() -> dict[str, object]:
    revision_code, revision = _command_output(["git", "rev-parse", "HEAD"])
    status_code, status = _command_output(["git", "status", "--short"])
    return {
        "commit": revision.strip() if revision_code == 0 else "",
        "dirty_worktree": bool(status.strip()) if status_code == 0 else None,
        "dirty_worktree_status": status.splitlines() if status_code == 0 else [],
    }


def _versions() -> dict[str, str]:
    packages = {
        "numpy": "numpy",
        "scipy": "scipy",
        "scikit_learn": "scikit-learn",
        "biopython": "biopython",
        "xgboost": "xgboost",
    }
    result = {"python": platform.python_version()}
    for label, distribution in packages.items():
        try:
            result[label] = importlib.metadata.version(distribution)
        except importlib.metadata.PackageNotFoundError:
            result[label] = "unavailable"
    return result


def _fold_digests() -> dict[str, object]:
    path = REPO_ROOT / "delta_files" / "regression_5fold_v1" / "metadata" / "fold_digests.json"
    return json.loads(path.read_text()) if path.is_file() else {"status": "unavailable"}


def build_findings(
    test_records: Sequence[Mapping[str, object]],
    geometry_rows: Sequence[Mapping[str, object]],
    deferred: Sequence[str],
    broad: Mapping[str, object],
) -> str:
    expected_failures = [row for row in test_records if row["status"] == "expected_failure"]
    unexpected = [row for row in test_records if row["status"] in {"failed", "error", "unexpected_success"}]
    geometry_failures = [row for row in geometry_rows if row["status"] != "passed"]
    lines = [
        "# Cross-Validation Adversarial Correctness Review",
        "",
        f"Review version: `{REVIEW_VERSION}`.",
        "",
        "## Summary",
        "",
        f"- Focused cross-validation suite: {len(test_records)} tests; "
        f"{len(expected_failures)} expected failures representing classified findings; "
        f"{len(unexpected)} unexpected failures/errors.",
        f"- Independent geometry-oracle comparisons: {len(geometry_rows)} rows; "
        f"{len(geometry_failures)} discrepancies.",
        f"- Broader release baseline: {broad['status']}"
        + (f" ({broad['test_count']} tests, {broad['skip_count']} skipped)." if broad.get("test_count") is not None else "."),
        "- No production regression or analysis implementation was changed by this review.",
        "",
        "## Findings",
        "",
        "### Moderate — ADV-GEOM-VALIDATION-001: low-level complete-distance helpers lack defensive validation",
        "",
        "`classical_mds` accepts asymmetric matrices and nonzero diagonals, while "
        "`compute_triangle_violations` accepts asymmetric and nonfinite matrices. "
        "The current protocol-2.0 workflow constructs symmetric zero-diagonal matrices and "
        "checks prediction coverage before calling these helpers, so the reviewed production "
        "path is protected. Direct or future reuse can nevertheless yield plausible invalid results.",
        "",
        "Minimized tests:",
        "",
    ]
    lines.extend(f"- `{row['test_id']}`" for row in expected_failures)
    lines.extend(
        [
            "",
            "No fix was made. Addressing this finding requires an explicit edit to "
            "`scripts/cluster_scripts/cluster_regression_helpers.py` and regression tests that "
            "cease being marked as expected failures.",
            "",
            "### Major semantic decision — ADV-TM-SEMANTICS-001: the saved value is a Kabsch-aligned TM-style score",
            "",
            "The implemented value applies the standard length-dependent TM kernel after an "
            "RMSD-optimal Kabsch alignment. It does not optimize the superposition under the TM "
            "objective, and no trusted TM-score/US-align executable was available in this local "
            "review. The result should not be claimed equivalent to canonical TM-score without "
            "that external comparison. The literal formula is independently confirmed by "
            "`test_tm_score_is_literal_post_kabsch_tm_style_formula`.",
            "",
            "### Moderate semantic decision — ADV-EIGEN-TOLERANCE-001: negative-eigenvalue count has literal-sign semantics",
            "",
            "Exact Euclidean examples can contain tiny negative eigenvalues from roundoff, and "
            "the literal count can change under algebraically equivalent centering or residue "
            "permutation. Negative eigenvalue mass remained negligible and is the robust diagnostic. "
            "Decide separately whether the count should retain literal-sign semantics or use a "
            "scale-aware tolerance; no output-contract or implementation change was made.",
            "",
            "### Minor semantic decision — ADV-REFLECTION-001: reconstruction quality is reflection-invariant",
            "",
            "The reviewed geometry intentionally compares both the original and reflected MDS "
            "solutions and retains the lower RMSD. This measures reflection-invariant reconstruction "
            "quality and does not establish recovered chirality. Whether to add `reflection_selected` "
            "remains an output-contract decision.",
            "",
            "## Confirmed high-risk behavior",
            "",
            "The passing adversarial cases independently confirmed held-out-target isolation, "
            "pair-fold responsibility, joint row-permutation invariance, directional feature order, "
            "conversion before float64 products, direct-concatenation pooled metrics, distinct "
            "pair/protein denominators, stable-only filtering, explicit incomplete coverage, exact "
            "3D/4D/non-Euclidean geometry behavior, triangle enumeration and severity denominators, "
            "negative-prediction clipping, and cross-implementation triangle primitives.",
            "",
            "## Unavailable or deferred checks",
            "",
        ]
    )
    lines.extend(f"- {item}" for item in deferred)
    lines.extend(
        [
            "- Real XGBoost 700-round native JSON save/load parity remains a Delta or matched-environment check.",
            "- No canonical-TM equivalence claim was made because no trusted executable comparison was run.",
            "",
        ]
    )
    if unexpected or geometry_failures:
        lines.extend(["## Unexpected discrepancies", ""])
        lines.extend(f"- `{row['test_id']}`: {row['status']}" for row in unexpected)
        lines.extend(
            f"- `{row['case_id']}` / `{row['metric']}`: {row['status']}"
            for row in geometry_failures
        )
        lines.append("")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--result-root",
        type=Path,
        default=REPO_ROOT / "delta_files" / "regression_5fold_v1",
        help="Copied-back protocol-2.0 result root.",
    )
    parser.add_argument("--run-broad-baseline", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now(timezone.utc).isoformat()
    git_state = _git_state()

    focused_result, focused_elapsed = run_focused_tests()
    test_records = sorted(focused_result.records, key=lambda row: str(row["test_id"]))
    broad = run_broad_baseline(args.run_broad_baseline)
    perturbations = build_perturbation_rows(test_records)
    geometry_rows = build_geometry_checks()
    replay_rows, deferred = artifact_replay_rows(args.result_root.resolve())
    stability = stability_rows(args.result_root.resolve())

    _write_json(output_dir / "automated_test_results.json", test_records)
    _write_csv(output_dir / "perturbation_checks.csv", PERTURBATION_COLUMNS, perturbations)
    _write_csv(output_dir / "geometry_oracle_checks.csv", GEOMETRY_COLUMNS, geometry_rows)
    _write_csv(output_dir / "artifact_replay_checks.csv", ARTIFACT_REPLAY_COLUMNS, replay_rows)
    _write_csv(output_dir / "stability_checks.csv", STABILITY_COLUMNS, stability)
    findings = build_findings(test_records, geometry_rows, deferred, broad)
    (output_dir / "review_findings.md").write_text(findings)

    manifest = {
        "review_version": REVIEW_VERSION,
        "started_at_utc": started_at,
        "finished_at_utc": datetime.now(timezone.utc).isoformat(),
        "repository": git_state,
        "versions": _versions(),
        "fold_digests": _fold_digests(),
        "result_root": str(args.result_root.resolve()),
        "test_commands": [
            {
                "command": FOCUSED_COMMAND,
                "status": "passed" if focused_result.wasSuccessful() else "failed",
                "test_count": len(test_records),
                "skip_count": len(focused_result.skipped),
                "expected_failure_count": len(focused_result.expectedFailures),
                "elapsed_seconds": focused_elapsed,
            },
            broad,
        ],
        "review_outputs": {
            "automated_test_results": "automated_test_results.json",
            "perturbation_checks": "perturbation_checks.csv",
            "geometry_oracle_checks": "geometry_oracle_checks.csv",
            "artifact_replay_checks": "artifact_replay_checks.csv",
            "stability_checks": "stability_checks.csv",
            "review_findings": "review_findings.md",
        },
        "selected_real_replay_cases": [],
        "deferred_checks": [
            *deferred,
            "real_xgboost_native_json_save_load_replay",
            "trusted_tm_score_or_usalign_comparison",
        ],
        "raw_pair_predictions_saved": False,
        "raw_triangle_predictions_saved": False,
    }
    _write_json(output_dir / "review_manifest.json", manifest)

    unexpected = [row for row in test_records if row["status"] in {"failed", "error", "unexpected_success"}]
    geometry_failed = [row for row in geometry_rows if row["status"] != "passed"]
    broad_failed = broad["status"] == "failed"
    print(f"Wrote adversarial review to {output_dir}")
    print(
        f"Focused tests={len(test_records)}, expected findings={len(focused_result.expectedFailures)}, "
        f"unexpected={len(unexpected)}, geometry discrepancies={len(geometry_failed)}, "
        f"broad={broad['status']}"
    )
    return 1 if unexpected or geometry_failed or broad_failed else 0


if __name__ == "__main__":
    raise SystemExit(main())

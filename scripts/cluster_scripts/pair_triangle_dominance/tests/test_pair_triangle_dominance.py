from __future__ import annotations

import json
import math
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pytest

WORKFLOW_DIR = Path(__file__).resolve().parents[1]
CLUSTER_DIR = WORKFLOW_DIR.parent
for directory in (WORKFLOW_DIR, CLUSTER_DIR):
    if str(directory) not in sys.path:
        sys.path.insert(0, str(directory))

from aggregate_pair_triangle_dominance import aggregate_results
from pair_triangle_dominance_common import (
    CHANNEL_SUMMARY_COLUMNS,
    CONFIGURATION_DIGEST,
    PER_PROTEIN_COLUMNS,
    PROTEIN_METRICS,
    STATUS_CODES,
    VARIANTS,
    channel_measurements,
    load_sample_bundle,
    read_csv,
    reduce_protein_tensor,
    sample_triples,
    transform_variant,
    unrank_triple,
    validate_samples,
)
from pair_triangle_dominance_layer import run_layer
from prepare_pair_triangle_samples import prepare_samples
from regression_protocol import load_master_manifest


@dataclass(frozen=True)
class Entry:
    protein_id: str
    source: str
    eligibility_status: str
    n_residues: int
    manifest_version: str = "synthetic_manifest_v1"


def synthetic_entries() -> tuple[Entry, ...]:
    return (
        Entry("protein_a", "cameo", "eligible", 4),
        Entry("protein_b", "cameo", "eligible", 5),
    )


def tensor_path(input_root: Path, protein_id: str, layer: int) -> Path:
    return input_root / protein_id / f"{protein_id}_pair_block_{layer}.npy"


def write_tensor(input_root: Path, entry: Entry, layer: int, seed: int) -> Path:
    path = tensor_path(input_root, entry.protein_id, layer)
    path.parent.mkdir(parents=True, exist_ok=True)
    tensor = np.random.default_rng(seed).normal(
        size=(entry.n_residues, entry.n_residues, 128)
    ).astype(np.float32)
    np.save(path, tensor)
    return path


def resolver(input_root: Path):
    return lambda protein_id, layer: tensor_path(input_root, protein_id, layer)


def prepare_synthetic(tmp_path: Path) -> tuple[Path, Path, tuple[Entry, ...]]:
    output = tmp_path / "output"
    inputs = tmp_path / "inputs"
    entries = synthetic_entries()
    prepare_samples(entries, output, expected_protein_count=2)
    return output, inputs, entries


def test_canonical_manifest_has_exact_cameo_cohort() -> None:
    cameo = [
        entry
        for entry in load_master_manifest()
        if entry.source == "cameo" and entry.eligible
    ]
    assert len(cameo) == 154
    assert len({entry.protein_id for entry in cameo}) == 154
    assert "7ebt_B" in {entry.protein_id for entry in cameo}
    assert "7re6_A" in {entry.protein_id for entry in cameo}
    assert "7ebt_A" not in {entry.protein_id for entry in cameo}


def test_triple_unranking_is_bijective_and_canonical() -> None:
    for n_residues in range(3, 12):
        triples = [
            unrank_triple(rank, n_residues)
            for rank in range(math.comb(n_residues, 3))
        ]
        assert len(set(triples)) == math.comb(n_residues, 3)
        assert set(triples) == {
            (i, j, k)
            for i in range(n_residues)
            for j in range(i)
            for k in range(j)
        }
        for rank, (i, j, k) in enumerate(triples):
            assert i > j > k
            assert math.comb(i, 3) + math.comb(j, 2) + k == rank


def test_sampling_is_deterministic_unique_and_capped() -> None:
    first, first_ranks, first_seed = sample_triples(
        "protein", 100, sample_cap=73
    )
    second, second_ranks, second_seed = sample_triples(
        "protein", 100, sample_cap=73
    )
    assert first_seed == second_seed
    assert np.array_equal(first, second)
    assert np.array_equal(first_ranks, second_ranks)
    validate_samples(first, first_ranks, 100, expected_count=73)
    assert len({tuple(row) for row in first}) == 73


def test_margin_examples_clipping_bounds_and_zero_scale() -> None:
    sides = np.asarray(
        [
            [[5.0], [3.0], [1.0]],
            [[5.0], [3.0], [2.0]],
            [[4.0], [3.0], [2.0]],
            [[0.0], [0.0], [0.0]],
            [[-5.0], [3.0], [1.0]],
            [[-5.0], [-3.0], [-1.0]],
        ]
    )
    metrics, positive, zero = channel_measurements(sides)
    assert metrics[0, 0, 0] == 1.0
    assert metrics[0, 0, 2] == pytest.approx(1.0 / 9.0)
    assert metrics[1, 0, 0] == 0.0
    assert not positive[1, 0]
    assert metrics[2, 0, 0] == -1.0
    assert metrics[2, 0, 1] == 0.0
    assert np.all(metrics[3] == 0.0)
    assert zero[3, 0]
    assert np.all(metrics[:, :, 1] >= 0.0)
    assert np.all(metrics[:, :, 3] >= 0.0)
    assert np.all(np.abs(metrics[:, :, 2]) <= 1.0 + 1e-14)
    assert np.array_equal(metrics[:, :, 1], np.maximum(metrics[:, :, 0], 0.0))
    assert np.array_equal(metrics[:, :, 3], np.maximum(metrics[:, :, 2], 0.0))


def test_directional_indexing_and_symmetrization_cancellation() -> None:
    lower = np.asarray([[[5.0], [2.0], [1.0]]])
    reverse = np.asarray([[[-5.0], [4.0], [-3.0]]])
    assert np.array_equal(transform_variant(lower, reverse, "directional_raw"), lower)
    assert np.array_equal(
        transform_variant(lower, reverse, "directional_abs"), np.abs(lower)
    )
    abs_then = transform_variant(lower, reverse, "abs_then_sym")
    sym_then = transform_variant(lower, reverse, "sym_then_abs")
    assert abs_then[0, 0, 0] == 5.0
    assert sym_then[0, 0, 0] == 0.0
    assert abs_then[0, 2, 0] == 2.0
    assert sym_then[0, 2, 0] == 1.0


def test_chunk_boundaries_do_not_change_reductions() -> None:
    rng = np.random.default_rng(4)
    tensor = rng.normal(size=(7, 7, 128)).astype(np.float32)
    triples = np.asarray(
        [(i, j, k) for i in range(7) for j in range(i) for k in range(j)],
        dtype=np.int32,
    )
    small = reduce_protein_tensor(tensor, triples, chunk_size=3)
    whole = reduce_protein_tensor(tensor, triples, chunk_size=len(triples))
    for field in (
        "triangle_metrics",
        "triangle_positive_fraction",
        "channel_metric_mean",
        "channel_positive_fraction",
        "channel_zero_scale_fraction",
    ):
        assert np.allclose(getattr(small, field), getattr(whole, field), atol=2e-15)
    assert np.allclose(
        np.mean(small.triangle_metrics, axis=0),
        np.mean(small.channel_metric_mean, axis=1),
        atol=2e-15,
    )
    assert np.allclose(
        np.mean(small.triangle_positive_fraction, axis=0),
        np.mean(small.channel_positive_fraction, axis=1),
        atol=2e-15,
    )


def test_preparation_round_trip_and_corruption_rejection(tmp_path: Path) -> None:
    output, _, _ = prepare_synthetic(tmp_path)
    bundle = load_sample_bundle(output, expected_protein_count=2)
    assert bundle.protein_ids == ("protein_a", "protein_b")
    assert len(bundle.ranks_for(0)) == math.comb(4, 3)
    assert len(bundle.ranks_for(1)) == math.comb(5, 3)
    with np.load(output / "metadata/triangle_samples.npz", allow_pickle=False) as archive:
        assert all(archive[name].dtype != object for name in archive.files)
    sample_path = output / "metadata/triangle_samples.npz"
    sample_path.write_bytes(sample_path.read_bytes() + b"corruption")
    with pytest.raises(ValueError, match="hash mismatch"):
        load_sample_bundle(output, expected_protein_count=2)


def test_layer_complete_rows_channel_shapes_and_restart_skip(tmp_path: Path) -> None:
    output, inputs, entries = prepare_synthetic(tmp_path)
    for index, entry in enumerate(entries):
        write_tensor(inputs, entry, 0, index)
    calls = []

    def counting_resolver(protein_id: str, layer: int) -> Path:
        calls.append((protein_id, layer))
        return tensor_path(inputs, protein_id, layer)

    status = run_layer(
        0,
        output,
        pair_path_resolver=counting_resolver,
        expected_protein_count=2,
        checkpoint_batch_size=1,
    )
    assert status["status"] == "complete"
    assert len(calls) == 2
    rows = read_csv(output / "layer_shards/per_protein_metrics_layer_00.csv", PER_PROTEIN_COLUMNS)
    assert len(rows) == 2 * 4 * 5
    assert all(row["status"] == "complete" for row in rows)
    with np.load(
        output / "layer_shards/per_protein_channel_metrics_layer_00.npz",
        allow_pickle=False,
    ) as archive:
        assert archive["channel_metric_mean"].shape == (2, 4, 128, 4)
        assert archive["channel_positive_fraction"].shape == (2, 4, 128)
        assert archive["channel_zero_scale_fraction"].shape == (2, 4, 128)
        assert np.all(archive["protein_status_code"] == STATUS_CODES["complete"])
        assert str(archive["configuration_digest"].item()) == CONFIGURATION_DIGEST
    calls.clear()
    skipped = run_layer(
        0,
        output,
        pair_path_resolver=counting_resolver,
        expected_protein_count=2,
    )
    assert skipped["skipped"]
    assert calls == []


@pytest.mark.parametrize("failure_kind", ["missing", "nonfinite", "shape"])
def test_layer_failures_have_explicit_rectangular_rows(
    tmp_path: Path, failure_kind: str
) -> None:
    output, inputs, entries = prepare_synthetic(tmp_path)
    write_tensor(inputs, entries[0], 0, 0)
    bad_path = tensor_path(inputs, entries[1].protein_id, 0)
    if failure_kind == "nonfinite":
        bad_path = write_tensor(inputs, entries[1], 0, 1)
        tensor = np.load(bad_path)
        tensor[0, 0, 0] = np.nan
        np.save(bad_path, tensor)
    elif failure_kind == "shape":
        bad_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(bad_path, np.zeros((5, 5, 127), dtype=np.float32))
    status = run_layer(
        0,
        output,
        pair_path_resolver=resolver(inputs),
        expected_protein_count=2,
        fail_on_incomplete=False,
    )
    assert status["status"] == "incomplete"
    rows = read_csv(output / "layer_shards/per_protein_metrics_layer_00.csv", PER_PROTEIN_COLUMNS)
    failed = [row for row in rows if row["protein_id"] == entries[1].protein_id]
    assert len(failed) == len(VARIANTS) * len(PROTEIN_METRICS)
    assert all(row["status"] != "complete" and row["failure_reason"] for row in failed)
    assert all(int(row["n_finite_triangles"]) == 0 for row in failed)
    with pytest.raises(ValueError, match="incomplete"):
        aggregate_results(output, layers=(0,), expected_protein_count=2)


def test_valid_checkpoint_resume_and_corruption_rejection(tmp_path: Path) -> None:
    output, inputs, entries = prepare_synthetic(tmp_path)
    for index, entry in enumerate(entries):
        write_tensor(inputs, entry, 0, index)
    interrupted_calls = []

    def interrupting_resolver(protein_id: str, layer: int) -> Path:
        interrupted_calls.append(protein_id)
        if protein_id == "protein_b":
            raise KeyboardInterrupt
        return tensor_path(inputs, protein_id, layer)

    with pytest.raises(KeyboardInterrupt):
        run_layer(
            0,
            output,
            pair_path_resolver=interrupting_resolver,
            expected_protein_count=2,
            checkpoint_batch_size=1,
        )
    resumed_calls = []

    def resumed_resolver(protein_id: str, layer: int) -> Path:
        resumed_calls.append(protein_id)
        return tensor_path(inputs, protein_id, layer)

    run_layer(
        0,
        output,
        pair_path_resolver=resumed_resolver,
        expected_protein_count=2,
        checkpoint_batch_size=1,
    )
    assert resumed_calls == ["protein_b"]

    second_output = tmp_path / "second_output"
    prepare_samples(entries, second_output, expected_protein_count=2)
    with pytest.raises(KeyboardInterrupt):
        run_layer(
            0,
            second_output,
            pair_path_resolver=interrupting_resolver,
            expected_protein_count=2,
            checkpoint_batch_size=1,
        )
    checkpoint_dir = second_output / ".checkpoints/layer_00"
    manifest = json.loads((checkpoint_dir / "checkpoint.json").read_text())
    metrics = checkpoint_dir / manifest["artifacts"]["metrics"]["filename"]
    metrics.write_text(metrics.read_text() + "corruption")
    with pytest.raises(ValueError, match="Corrupted layer checkpoint artifact"):
        run_layer(
            0,
            second_output,
            pair_path_resolver=resolver(inputs),
            expected_protein_count=2,
            checkpoint_batch_size=1,
        )


def test_one_layer_end_to_end_aggregation_is_protein_equal(tmp_path: Path) -> None:
    output, inputs, entries = prepare_synthetic(tmp_path)
    for index, entry in enumerate(entries):
        write_tensor(inputs, entry, 0, index + 20)
    run_layer(
        0,
        output,
        pair_path_resolver=resolver(inputs),
        expected_protein_count=2,
    )
    manifest = aggregate_results(output, layers=(0,), expected_protein_count=2)
    assert manifest["per_protein_row_count"] == 2 * 4 * 5
    assert manifest["layer_summary_row_count"] == 4 * 5
    assert manifest["channel_summary_row_count"] == 4 * 128 * 6
    per_rows = read_csv(output / "per_protein_metrics.csv", PER_PROTEIN_COLUMNS)
    layer_rows = read_csv(output / "layer_summary.csv")
    channel_rows = read_csv(output / "channel_summary.csv", CHANNEL_SUMMARY_COLUMNS)
    assert len(channel_rows) == 4 * 128 * 6
    selected = [
        row
        for row in per_rows
        if row["representation_variant"] == "directional_raw"
        and row["metric"] == "signed_margin"
    ]
    summary = next(
        row
        for row in layer_rows
        if row["representation_variant"] == "directional_raw"
        and row["metric"] == "signed_margin"
    )
    assert float(summary["protein_equal_mean"]) == pytest.approx(
        np.mean([float(row["mean"]) for row in selected])
    )
    assert int(summary["n_proteins_finite"]) == 2
    assert not any(path.name.startswith("triangle_") for path in output.glob("*.csv"))
    with pytest.raises(FileNotFoundError, match="Missing layer"):
        aggregate_results(output, layers=(0, 1), expected_protein_count=2)


def test_flat_deployment_modules_import(tmp_path: Path) -> None:
    for filename in (
        "pair_triangle_dominance_common.py",
        "prepare_pair_triangle_samples.py",
        "pair_triangle_dominance_layer.py",
        "aggregate_pair_triangle_dominance.py",
    ):
        shutil.copy2(WORKFLOW_DIR / filename, tmp_path / filename)
    (tmp_path / "regression_protocol.py").write_text(
        "def load_master_manifest():\n    return ()\n", encoding="utf-8"
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            "import prepare_pair_triangle_samples, pair_triangle_dominance_layer, "
            "aggregate_pair_triangle_dominance",
        ],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stderr

import csv
import gzip
import importlib.util
import json
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[1] / "diagnostic_salt_bridge_classification_probe.py"
SPEC = importlib.util.spec_from_file_location("salt_probe", MODULE)
probe = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(probe)


def annotation_row(**overrides):
    row = {
        "protein_id": "protein",
        "pair_id": "7",
        "n_residues": "4",
        "i": "0",
        "j": "2",
        "resname_i": "LYS",
        "resname_j": "GLU",
        "sequence_separation": "2",
        "ca_distance": "8.0",
        "dssp_3state_i": "helix",
        "dssp_3state_j": "beta",
        "relative_sasa_i": "0.30",
        "relative_sasa_j": "0.10",
        "salt_bridge": "1",
        "salt_bridge_valid": "1",
        # These fields must never enter matching or ordinary features.
        "charged_atom_distance": "3.2",
        "min_heavy_atom_distance": "2.8",
    }
    row.update(overrides)
    return row


def candidate(pair_id, label, ca=8.0, sasa=0.1, seq=8, acid="GLU", base="LYS"):
    return {
        "protein_id": "protein",
        "pair_id": pair_id,
        "n_residues": 20,
        "i": min(pair_id, pair_id + 1),
        "j": max(pair_id, pair_id + 1),
        "acid_index": 1,
        "base_index": 3,
        "acid_name": acid,
        "base_name": base,
        "sequence_separation": seq,
        "sequence_bin": probe.separation_bin(seq),
        "ca_distance": ca,
        "acid_dssp": "helix",
        "base_dssp": "coil",
        "relative_sasa_acid": sasa,
        "relative_sasa_base": sasa,
        "label": label,
        "match_set_id": "",
        "match_tier": None,
        "match_cost": None,
    }


def test_candidate_is_canonicalized_as_acid_then_base_and_complete_case():
    parsed, reason = probe.parse_candidate_row(annotation_row())
    assert reason is None
    assert parsed["acid_index"] == 2
    assert parsed["base_index"] == 0
    assert parsed["acid_name"] == "GLU"
    assert parsed["base_name"] == "LYS"
    assert parsed["acid_dssp"] == "beta"
    assert parsed["base_dssp"] == "helix"
    assert parsed["relative_sasa_acid"] == 0.10
    assert parsed["relative_sasa_base"] == 0.30

    missing, reason = probe.parse_candidate_row(
        annotation_row(relative_sasa_i="")
    )
    assert missing is None
    assert reason == "missing_or_invalid_ordinary_covariate"
    noncandidate, reason = probe.parse_candidate_row(
        annotation_row(resname_i="ALA", resname_j="GLY")
    )
    assert noncandidate is None
    assert reason == "not_acidic_basic_candidate"


def test_annotation_loader_fails_fast_when_every_dssp_run_failed(tmp_path):
    (tmp_path / "annotation_config.json").write_text(
        json.dumps({"config_hash": "failed_dssp"})
    )
    with (tmp_path / "protein_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["protein_id", "status", "residue_count", "failure_reason"],
        )
        writer.writeheader()
        for index in range(50):
            writer.writerow(
                {
                    "protein_id": f"p{index:02d}",
                    "status": "success",
                    "residue_count": 10,
                    "failure_reason": "mkdssp returned non-zero exit status 1",
                }
            )
    try:
        probe.load_annotation_records(tmp_path)
    except RuntimeError as exc:
        assert "all 50 annotation proteins report DSSP failures" in str(exc)
    else:
        raise AssertionError("expected a complete-case DSSP failure")


def test_ordinary_baseline_is_simple_and_excludes_label_defining_distances():
    parsed, _ = probe.parse_candidate_row(annotation_row())
    matrix = probe.build_ordinary_features([parsed])
    assert matrix.shape == (1, 10)
    assert "charged_atom_distance" not in probe.ORDINARY_FEATURE_NAMES
    assert "min_heavy_atom_distance" not in probe.ORDINARY_FEATURE_NAMES
    assert probe.ORDINARY_FEATURE_NAMES == [
        "ca_distance",
        "log1p_sequence_separation",
        "relative_sasa_acid",
        "relative_sasa_base",
        "acid_is_glu",
        "base_is_lys",
        "acid_dssp_is_helix",
        "acid_dssp_is_beta",
        "base_dssp_is_helix",
        "base_dssp_is_beta",
    ]


def test_representation_features_use_canonical_cross_and_diagonal_order():
    block = np.empty((4, 4, probe.N_CHANNELS), dtype=np.float32)
    for i in range(4):
        for j in range(4):
            block[i, j, :] = 100 * i + 10 * j + np.arange(probe.N_CHANNELS)
    row = candidate(1, 1)
    row["acid_index"], row["base_index"] = 1, 3
    cross, diagonal = probe.extract_representation_features(block, [row])
    np.testing.assert_allclose(cross[0, :128], block[1, 3])
    np.testing.assert_allclose(cross[0, 128:], block[3, 1])
    np.testing.assert_allclose(diagonal[0, :128], block[1, 1])
    np.testing.assert_allclose(diagonal[0, 128:], block[3, 3])
    assert cross.dtype == np.float64
    assert diagonal.dtype == np.float64


def test_matching_is_unique_same_identity_and_prefers_strict_tier():
    positives = [candidate(10, 1, ca=8.0), candidate(20, 1, ca=10.0)]
    negatives = [
        candidate(30, 0, ca=8.1),
        candidate(40, 0, ca=10.1),
        candidate(50, 0, ca=8.0, acid="ASP"),
    ]
    matched, diagnostics, unmatched = probe.match_protein_candidates(
        "protein", positives + negatives
    )
    assert not unmatched
    assert diagnostics["n_matched_positive"] == 2
    assert diagnostics["n_matches_tier_0"] == 2
    negatives_used = [row["pair_id"] for row in matched if row["label"] == 0]
    assert sorted(negatives_used) == [30, 40]
    assert len(set(negatives_used)) == len(negatives_used)
    assert all(row["acid_name"] == "GLU" for row in matched)


def test_fixed_split_is_deterministic_disjoint_and_has_30_10_10_counts():
    records = [
        {
            "protein_id": f"p{index:02d}",
            "n_positive": (index * 7) % 13,
            "n_residues": 50 + index * 3,
        }
        for index in range(50)
    ]
    first = probe.assign_protein_splits(records)
    second = probe.assign_protein_splits(list(reversed(records)))
    assert first == second
    assert len(first) == 50
    assert list(first.values()).count("train") == 30
    assert list(first.values()).count("validation") == 10
    assert list(first.values()).count("test") == 10


def test_equal_protein_weights_and_train_standardizer():
    proteins = np.asarray(["a", "a", "a", "b"], dtype=object)
    weights = probe.equal_protein_weights(proteins)
    assert np.isclose(weights[proteins == "a"].sum(), weights[proteins == "b"].sum())
    matrix = np.asarray([[0.0], [0.0], [0.0], [10.0]])
    mean, scale = probe.fit_weighted_standardizer(matrix, weights)
    assert np.isclose(mean[0], 5.0)
    transformed = probe.transform_matrix(matrix, mean, scale)
    assert transformed.dtype == np.float64


def test_regularized_fit_selects_only_from_validation_data():
    rng = np.random.default_rng(42)
    train_proteins = np.repeat([f"train_{index}" for index in range(6)], 10)
    validation_proteins = np.repeat([f"validation_{index}" for index in range(3)], 10)
    train_y = np.tile([0, 1], 30).astype(np.int8)
    validation_y = np.tile([0, 1], 15).astype(np.int8)
    train_x = rng.normal(size=(len(train_y), 10))
    validation_x = rng.normal(size=(len(validation_y), 10))
    train_x[:, 0] += 1.5 * train_y
    validation_x[:, 0] += 1.5 * validation_y
    train_bundle = {
        "y": train_y,
        "protein_ids": train_proteins,
        "matched_mask": np.ones(len(train_y), dtype=bool),
    }
    validation_bundle = {
        "y": validation_y,
        "protein_ids": validation_proteins,
        "matched_mask": np.ones(len(validation_y), dtype=bool),
    }
    fitted = probe.fit_variant(
        "ordinary",
        train_x,
        validation_x,
        train_bundle,
        validation_bundle,
        probe.Deadline(60),
    )
    assert fitted["selected_C"] in probe.C_GRID
    assert 0.0 <= fitted["threshold"] <= 1.0
    assert sum(row["selected"] for row in fitted["search_rows"]) == 1


def test_prediction_shard_schema_and_scope_rows(tmp_path):
    metadata = []
    for index, label in enumerate((1, 0)):
        row = candidate(index + 1, label)
        row.update(is_matched=True, match_set_id="m1", match_tier=0, match_cost=0.1)
        metadata.append(row)
    bundle = {
        "metadata": metadata,
        "y": np.asarray([1, 0], dtype=np.int8),
        "protein_ids": np.asarray(["protein", "protein"], dtype=object),
        "matched_mask": np.asarray([True, True]),
    }
    path = tmp_path / "predictions.csv.gz"
    probe.write_prediction_shard(
        path,
        "ordinary",
        {"validation": bundle, "test": bundle},
        {
            "validation": np.asarray([0.9, 0.1]),
            "test": np.asarray([0.8, 0.2]),
        },
        0.5,
    )
    with gzip.open(path, "rt", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert list(rows[0]) == probe.PREDICTION_FIELDS
    assert len(rows) == 8  # two pairs x two splits x matched/natural scopes
    assert {row["evaluation_scope"] for row in rows} == {
        "matched",
        "natural_candidate_pool",
    }


def test_synthetic_fifty_protein_end_to_end(tmp_path, monkeypatch):
    annotation_root = tmp_path / "annotation_root"
    pair_root = tmp_path / "pair_blocks"
    (annotation_root / "annotations" / "per_protein").mkdir(parents=True)
    (annotation_root / "annotation_config.json").write_text(
        json.dumps({"config_hash": "synthetic_annotation_hash"})
    )
    pair_fields = [
        "protein_id",
        "pair_id",
        "n_residues",
        "i",
        "j",
        "resname_i",
        "resname_j",
        "sequence_separation",
        "ca_distance",
        "dssp_3state_i",
        "dssp_3state_j",
        "relative_sasa_i",
        "relative_sasa_j",
        "salt_bridge",
        "salt_bridge_valid",
    ]
    manifests = []
    for protein_index in range(50):
        protein_id = f"p{protein_index:02d}"
        manifests.append(
            {"protein_id": protein_id, "status": "success", "residue_count": 8}
        )
        rows = [
            [protein_id, 0, 8, 0, 1, "ASP", "LYS", 1, 8.0, "helix", "coil", 0.1, 0.2, 1, 1],
            [protein_id, 1, 8, 2, 3, "ASP", "LYS", 1, 8.1, "helix", "coil", 0.1, 0.2, 0, 1],
            [protein_id, 2, 8, 4, 5, "GLU", "ARG", 1, 9.0, "beta", "coil", 0.2, 0.3, 1, 1],
            [protein_id, 3, 8, 6, 7, "GLU", "ARG", 1, 9.1, "beta", "coil", 0.2, 0.3, 0, 1],
        ]
        with gzip.open(
            annotation_root / "annotations" / "per_protein" / f"{protein_id}.csv.gz",
            "wt",
            newline="",
        ) as handle:
            writer = csv.writer(handle)
            writer.writerow(pair_fields)
            writer.writerows(rows)
        protein_pair_root = pair_root / protein_id
        protein_pair_root.mkdir(parents=True)
        generator = np.random.default_rng(protein_index)
        block = generator.normal(size=(8, 8, probe.N_CHANNELS)).astype(np.float32)
        block[0, 1] += 1.0
        block[4, 5] += 1.0
        np.save(protein_pair_root / f"{protein_id}_pair_block_47.npy", block)
    with (annotation_root / "protein_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(
            handle, fieldnames=["protein_id", "status", "residue_count"]
        )
        writer.writeheader()
        writer.writerows(manifests)

    monkeypatch.setattr(probe, "C_GRID", (0.01, 1.0))
    monkeypatch.setattr(
        probe, "MIN_MATCHED_POSITIVES", {"train": 1, "validation": 1, "test": 1}
    )
    monkeypatch.setattr(probe, "MIN_CONTRIBUTING_PROTEINS", 1)
    monkeypatch.setenv("CSV_ROOT", str(annotation_root))
    monkeypatch.setenv("PAIR_BLOCKS_ROOT", str(pair_root))
    monkeypatch.setenv("PROBE_DEADLINE_SECONDS", "600")
    probe.main()

    output = annotation_root / "classification" / "salt_bridge" / "layer_47"
    for relative in (
        "classification_config.json",
        "split_assignments.csv",
        "matching_diagnostics.csv",
        "matched_pair_assignments.csv.gz",
        "regularization_search.csv",
        "interaction_type_summary.csv",
        "interaction_type_per_protein.csv",
        "coefficients.csv",
        "predictions.csv.gz",
        "classification_exclusions.csv",
        "summary.txt",
        "visualizations/layer47_model_comparison.png",
    ):
        assert (output / relative).exists()
    with (output / "interaction_type_summary.csv").open(newline="") as handle:
        summary_rows = list(csv.DictReader(handle))
    assert len(summary_rows) == 30
    assert {row["model_variant"] for row in summary_rows} == set(probe.VARIANT_ORDER)

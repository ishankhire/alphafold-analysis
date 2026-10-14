import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest


MODULE = (
    Path(__file__).parents[1]
    / "diagnostic_shared_ols_mds_pdb_export.py"
)
spec = importlib.util.spec_from_file_location("shared_ols_mds_pdb_export", MODULE)
diagnostic = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = diagnostic
spec.loader.exec_module(diagnostic)


def test_fixed_layers_and_montage_proteins():
    assert diagnostic.LAYERS == (0, 10, 20, 30, 40, 47)
    assert diagnostic.PROTEIN_IDS == (
        "7b3a_A",
        "7dkk_A",
        "7dq9_A",
        "7ebt_B",
        "7f6e_B",
        "7kdx_B",
        "7mro_A",
        "7pbk_A",
        "7tbs_A",
        "7tkv_A",
    )


def test_prediction_uses_directional_standardized_artifact():
    rng = np.random.default_rng(4)
    pair_block = rng.normal(size=(5, 5, 128)).astype(np.float32)
    pair_i, pair_j = diagnostic.strict_lower_triangle_indices(5)
    targets = np.ones(pair_i.size, dtype=np.float64)
    record = SimpleNamespace(
        protein_id="synthetic",
        n_residues=5,
        pair_i=pair_i,
        pair_j=pair_j,
        targets=targets,
    )
    mean = rng.normal(size=256)
    scale = rng.uniform(0.5, 2.0, size=256)
    coefficients = rng.normal(size=256)
    artifact = {
        "feature_mean": mean,
        "feature_scale": scale,
        "coefficient_standardized": coefficients,
        "intercept_standardized": 3.25,
    }
    observed = diagnostic.predict_distances(pair_block, record, artifact)
    features = np.concatenate(
        [pair_block[pair_i, pair_j, :], pair_block[pair_j, pair_i, :]], axis=1
    ).astype(np.float64)
    expected = 3.25 + ((features - mean) / scale) @ coefficients
    np.testing.assert_allclose(observed, expected, rtol=0.0, atol=0.0)


def test_exact_euclidean_distances_reconstruct_and_align():
    rng = np.random.default_rng(9)
    true_coordinates = rng.normal(size=(15, 3))
    pair_i, pair_j = diagnostic.strict_lower_triangle_indices(len(true_coordinates))
    predictions = diagnostic.compute_distance_matrix(true_coordinates)[pair_i, pair_j]
    reconstruction, n_negative = diagnostic.reconstruct_mds_coordinates(
        len(true_coordinates), true_coordinates, pair_i, pair_j, predictions
    )
    assert n_negative == 0
    assert reconstruction.rmsd < 1e-6
    assert reconstruction.tm_score > 0.999999
    assert reconstruction.stress < 1e-12
    assert reconstruction.frac_top3_eigenvalues > 0.999999


def test_pdb_writer_breaks_connections_at_chain_boundaries(tmp_path):
    path = tmp_path / "trace.pdb"
    residues = (("A", "GLY", 1), ("A", "ALA", 2), ("B", "SER", 1))
    coordinates = np.arange(9, dtype=np.float64).reshape(3, 3)
    diagnostic.write_ca_trace_pdb(
        path,
        residues=residues,
        coordinates=coordinates,
        b_factors=np.asarray([0.0, 1.25, 2.5]),
        remarks=("synthetic",),
    )
    text = path.read_text()
    assert text.count("ATOM") == 3
    assert "CONECT    1    2" in text
    assert "CONECT    2    3" not in text
    assert text.endswith("END\n")


def test_saved_geometry_parity_accepts_legacy_and_optimized_scores():
    fixed = np.random.default_rng(9).normal(size=(30, 3))
    moving = fixed.copy()
    moving[-5:] += 15
    pair_i, pair_j = diagnostic.strict_lower_triangle_indices(30)
    distances = diagnostic.compute_distance_matrix(moving)[pair_i, pair_j]
    reconstruction, negatives = diagnostic.reconstruct_mds_coordinates(30, fixed, pair_i, pair_j, distances)
    record = SimpleNamespace(protein_id="toy", coordinates=fixed, n_residues=30)
    regression = {name: 0.0 for name in ("r2", "mae", "rmse", "bias_pred_minus_true", "mean_predicted_distance")}
    saved = dict(geometry_status="complete", rmsd=reconstruction.rmsd,
                 tm_score=diagnostic.compute_tm_score(fixed, reconstruction.coordinates, 30),
                 frac_top3_eigenvalues=reconstruction.frac_top3_eigenvalues,
                 negative_eigenvalue_fraction=reconstruction.negative_eigenvalue_fraction,
                 stress=reconstruction.stress, n_negative_eigenvalues=reconstruction.n_negative_eigenvalues,
                 n_negative_predictions=negatives)
    arguments = dict(layer=0, record=record, regression=regression, reconstruction=reconstruction,
                     n_negative_predictions=negatives, saved_per_protein={(0,"toy"): dict(
                         regression, prediction_status="complete")}, saved_geometry={(0,"toy"): saved})
    diagnostic.validate_saved_metrics(**arguments)
    saved["tm_score"] = reconstruction.tm_score
    diagnostic.validate_saved_metrics(**arguments, saved_tm_score_method=diagnostic.TM_SCORE_METHOD)
    saved["tm_score"] = -1
    with pytest.raises(ValueError, match="tm_score"):
        diagnostic.validate_saved_metrics(**arguments)


def test_launcher_has_required_resources_and_interface():
    launcher = (
        Path(__file__).parents[1]
        / "diagnostic_shared_ols_mds_pdb_export.sbatch"
    ).read_text()
    required = (
        "#SBATCH --mem=8g",
        "#SBATCH --time=00:30:00",
        '#SBATCH --constraint="scratch&work"',
        "#SBATCH --mail-type=BEGIN,END,FAIL",
        "#SBATCH --mail-user=ikhire3@gatech.edu",
        "SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}",
        'srun python3 -u "$SCRIPT_DIR/diagnostic_shared_ols_mds_pdb_export.py"',
    )
    assert all(value in launcher for value in required)

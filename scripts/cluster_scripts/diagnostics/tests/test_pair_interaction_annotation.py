import importlib.util
from pathlib import Path

import numpy as np


MODULE = Path(__file__).parents[1] / "diagnostic_pair_interaction_annotation.py"
spec = importlib.util.spec_from_file_location("annotation", MODULE)
annotation = importlib.util.module_from_spec(spec)
spec.loader.exec_module(annotation)


def residue(name, charged=(), side=(), rsasa=None):
    return {"resname": name, "charged": np.asarray(charged, dtype=float).reshape(-1, 3),
            "side": np.asarray(side, dtype=float).reshape(-1, 3), "dssp_key": ("A", (" ", 1, " ")),
            "relative_sasa": rsasa}


def test_salt_boundary_arg_ne_exclusion_and_missing_atoms():
    asp = residue("ASP", charged=[(0, 0, 0)])
    lys = residue("LYS", charged=[(4, 0, 0)])
    labels = annotation.pair_labels(asp, lys, None, {}, {})
    assert labels["salt_bridge"] == 1 and labels["salt_bridge_valid"] == 1
    arg_ne_only = residue("ARG", charged=[])
    assert annotation.pair_labels(asp, arg_ne_only, None, {}, {})["salt_bridge_valid"] == 0
    assert annotation.pair_labels(residue("ALA"), residue("GLY"), None, {}, {})["salt_bridge"] == 0


def test_hydrophobic_and_core_boundaries_and_gly_validity():
    a = residue("LEU", side=[(0, 0, 0)], rsasa=.2)
    b = residue("VAL", side=[(4.5, 0, 0)], rsasa=.2)
    labels = annotation.pair_labels(a, b, None, {}, {})
    assert labels["hydrophobic_contact"] == 1
    assert labels["hydrophobic_core_contact"] is None  # DSSP/rSASA mapping is invalid
    a["dssp_key"] = ("A", (" ", 1, " ")); b["dssp_key"] = ("A", (" ", 2, " "))
    dssp = {a["dssp_key"]: ("L", "-", 1, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0), b["dssp_key"]: ("V", "-", 1, 0, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0)}
    assert annotation.pair_labels(a, b, dssp, {}, {})["hydrophobic_core_contact"] == 1
    assert annotation.pair_labels(residue("GLY"), residue("LEU"), None, {}, {})["hydrophobic_contact"] == 0


def test_three_state_codes():
    assert annotation.three_state("P") == "coil"
    assert annotation.three_state("E") == "beta"
    assert annotation.three_state("H") == "helix"

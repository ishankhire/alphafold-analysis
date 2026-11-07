"""Input and output roots shared by the transfer follow-up diagnostics."""
from __future__ import annotations

import os
from pathlib import Path

from canonical_storage import csv_root


def experiment_root(variable: str, name: str) -> Path:
    value = os.environ.get(variable)
    return Path(value).expanduser().resolve() if value else Path.home() / "diagnostic_experiments" / name


PRODUCER = csv_root() / "regression_5fold_v1"
ABLATION_CSV_DIR = Path(os.environ.get("LAYERWISE_ABLATION_CSV_DIR", csv_root() / "layerwise_ablation_csvs"))
TRANSFER_ROOT = experiment_root("ALL_LAYER_TRANSFER_OUTPUT_ROOT", "cross_layer_probe_transfer_all_layers_cameo_v1")
GAIN_ROOT = experiment_root("GAIN_POSITION_OUTPUT_ROOT", "gain_and_chain_position_cameo_v1")
TRIANGLE_ROOT = experiment_root("TRIANGLE_STRATA_OUTPUT_ROOT", "triangle_strata_cameo_v1")
PROTEIN_STRATA = {
    "shared_ols_base256": experiment_root("NATIVE_OLS_STRATA_ROOT", "native_error_strata_v1") / "data/protein_strata_cameo.npz",
    "shared_xgboost_base256_fixed_s4": experiment_root("NATIVE_XGBOOST_STRATA_ROOT", "native_error_strata_xgboost_v1") / "data/protein_strata_cameo.npz",
}

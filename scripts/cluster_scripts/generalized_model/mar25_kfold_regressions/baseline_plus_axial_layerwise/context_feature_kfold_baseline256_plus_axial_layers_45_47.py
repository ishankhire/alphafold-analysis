#!/usr/bin/env python3
"""
Strict size-stratified 5-fold regression for baseline256_plus_axial on layers 45-47.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
GENERALIZED_MODEL_DIR = SCRIPT_DIR.parent
if str(GENERALIZED_MODEL_DIR) not in sys.path:
    sys.path.insert(0, str(GENERALIZED_MODEL_DIR))

from context_feature_kfold_common import make_variant, run_variant_job  # noqa: E402


TITLE = "Context Feature K-Fold Regression - baseline256_plus_axial (layers 45-47)"


def main():
    run_variant_job(
        make_variant("baseline256_plus_axial", use_axial=True),
        TITLE,
        layer_indices=list(range(45, 48)),
        output_filename_suffix="45-47",
    )


if __name__ == "__main__":
    main()

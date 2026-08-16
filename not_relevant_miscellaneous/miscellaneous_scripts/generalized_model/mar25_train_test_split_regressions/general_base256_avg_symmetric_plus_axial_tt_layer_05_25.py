#!/usr/bin/env python3
"""
Strict size-stratified train/test regression for general_base256_avg_symmetric_plus_axial on layers 5-25.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from general_train_test_common import make_variant, run_variant_job  # noqa: E402


TITLE = "General Train/Test Regression - base256_avg_symmetric_plus_axial (layers 5-25)"


def main():
    run_variant_job(
        make_variant("general_base256_avg_symmetric_plus_axial", pair_mode="avg_symmetric", use_axial=True),
        TITLE,
        layer_indices=list(range(5, 26)),
        output_filename_suffix="5-25",
    )


if __name__ == "__main__":
    main()

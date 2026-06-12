#!/usr/bin/env python3
"""
Layer-47 size-stratified train/test regression for combined CAMEO+ECOD base256.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from general_train_test_common import make_variant, run_variant_job  # noqa: E402


TITLE = "Combined CAMEO+ECOD Train/Test Regression - base256 (layer 47)"


def main():
    run_variant_job(
        make_variant("general_base256_combined", pair_mode="concat_directional"),
        TITLE,
        layer_indices=[47],
        output_filename_suffix="layer47",
    )


if __name__ == "__main__":
    main()

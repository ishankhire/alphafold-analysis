#!/usr/bin/env python3
"""
48-layer size-stratified train/test regression for the general base256+diagonal feature set.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from general_train_test_common import make_variant, run_variant_job  # noqa: E402


TITLE = "General Train/Test Regression - base256_plus_diagonal"


def main():
    run_variant_job(
        make_variant("general_base256_plus_diagonal", pair_mode="concat_directional", use_self_diag=True),
        TITLE,
    )


if __name__ == "__main__":
    main()

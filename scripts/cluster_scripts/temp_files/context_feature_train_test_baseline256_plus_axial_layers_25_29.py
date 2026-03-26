#!/usr/bin/env python3
"""
Temporary single-split regression for baseline256_plus_axial on layers 25-29.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from context_feature_train_test_common import make_variant, run_variant_train_test_job  # noqa: E402


TITLE = "Context Feature Train/Test Regression - baseline256_plus_axial (layers 25-29)"


def main():
    run_variant_train_test_job(
        make_variant("baseline256_plus_axial", use_axial=True),
        TITLE,
        layer_indices=list(range(25, 30)),
        output_filename_suffix="25-29",
    )


if __name__ == "__main__":
    main()

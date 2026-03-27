#!/usr/bin/env python3
"""
Strict size-stratified 5-fold regression for baseline256 on layers 5-9.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from context_feature_kfold_common import make_variant, run_variant_job  # noqa: E402


TITLE = "Context Feature K-Fold Regression - baseline256 (layers 5-9)"


def main():
    run_variant_job(
        make_variant("baseline256"),
        TITLE,
        layer_indices=list(range(5, 10)),
        output_filename_suffix="5-9",
    )


if __name__ == "__main__":
    main()

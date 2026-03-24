#!/usr/bin/env python3
"""
48-layer strict size-stratified 5-fold regression for the baseline256+diagonal feature set.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from context_feature_kfold_common import make_variant, run_variant_job  # noqa: E402


TITLE = "Context Feature K-Fold Regression - baseline256_plus_diagonal"


def main():
    run_variant_job(
        make_variant("baseline256_plus_diagonal", use_self_diag=True),
        TITLE,
    )


if __name__ == "__main__":
    main()

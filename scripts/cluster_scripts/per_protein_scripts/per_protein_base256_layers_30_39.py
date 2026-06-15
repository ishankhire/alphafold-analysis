#!/usr/bin/env python3
"""
Layers 30-39 per-protein base256 regressions for combined CAMEO+ECOD.
"""

import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from per_protein_base256_common import (  # noqa: E402
    make_base256_per_protein_variant,
    run_per_protein_job,
)


TITLE = "Combined CAMEO+ECOD Per-Protein Regression - base256 (layers 30-39)"


def main():
    run_per_protein_job(
        make_base256_per_protein_variant(),
        TITLE,
        layer_indices=range(30, 40),
        output_filename_suffix="layers30_39",
    )


if __name__ == "__main__":
    main()

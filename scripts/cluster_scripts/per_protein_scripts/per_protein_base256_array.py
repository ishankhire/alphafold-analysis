#!/usr/bin/env python3
"""Slurm-array entrypoint for per-layer per-protein base256 regressions."""

import os
import sys
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))

from per_protein_base256_common import (  # noqa: E402
    make_base256_per_protein_variant,
    run_per_protein_job,
)


def requested_layer():
    """Map one Slurm array task ID to one Evoformer layer."""
    task_id = os.environ.get("SLURM_ARRAY_TASK_ID")
    if task_id is None or task_id == "":
        raise ValueError("SLURM_ARRAY_TASK_ID is required for per_protein_base256_array.py")

    layer = int(task_id)
    if layer < 0 or layer > 47:
        raise ValueError(f"SLURM_ARRAY_TASK_ID={layer} is outside layer range 0-47")
    return layer


def main():
    layer = requested_layer()
    title = f"Combined CAMEO+ECOD Per-Protein Regression - base256 (layer {layer})"
    run_per_protein_job(
        make_base256_per_protein_variant(),
        title,
        layer_indices=[layer],
        output_filename_suffix=f"layers{layer:02d}_{layer:02d}",
    )


if __name__ == "__main__":
    main()

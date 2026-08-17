#!/usr/bin/env python3
"""Recombine six joint-gate task fragments into one canonical layer shard."""

from __future__ import annotations

import os

from aggregate_gated_linear_probe_suite import finish_joint_layer_shards, validate_task
from gated_linear_probe_common import (
    ANALYSIS_JOINT,
    layer_for_task,
    load_context,
    log,
    run_mode,
    suite_root,
)


def main(task_id: int | None = None) -> None:
    mode = run_mode()
    layer_task = int(os.environ.get("SLURM_ARRAY_TASK_ID", "0")) if task_id is None else int(task_id)
    layer = layer_for_task(layer_task, mode)
    root = suite_root(mode)
    preparation_manifest, records = load_context(root)
    joint_manifests = [
        validate_task(
            root,
            ANALYSIS_JOINT,
            layer_task * 6 + sequence_code,
            preparation_manifest,
        )
        for sequence_code in range(6)
    ]
    finish_joint_layer_shards(
        root,
        records,
        preparation_manifest,
        joint_manifests,
        layers=(layer,),
    )
    log(f"Completed joint canonical layer shard {layer}")


if __name__ == "__main__":
    main()

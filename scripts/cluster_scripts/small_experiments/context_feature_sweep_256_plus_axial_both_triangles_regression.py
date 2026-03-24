#!/usr/bin/env python3
"""
context_feature_sweep_256_plus_axial_both_triangles_regression.py
=================================================================
Small experiment for a single layer-47 linear probe that uses:
- the 256d center pair representation: pair(i, j) + pair(j, i)
- 4 axial neighbors around the lower-triangle center (i, j)
- 4 axial neighbors around the mirrored upper-triangle center (j, i)

This yields 10 pair representations total:
2 center + 4 lower-axial + 4 upper-axial = 1280 dense features.
"""

import importlib.util
import os
import time
from pathlib import Path

import numpy as np


ROOT_DIR = Path(__file__).resolve().parents[3]
BASE_SCRIPT_PATH = (
    ROOT_DIR
    / "scripts"
    / "cluster_scripts"
    / "diagnostics"
    / "diagnostic_context_feature_sweep_regression.py"
)

spec = importlib.util.spec_from_file_location(
    "diagnostic_context_feature_sweep_base",
    str(BASE_SCRIPT_PATH),
)
base = importlib.util.module_from_spec(spec)
if spec.loader is None:  # pragma: no cover - fixed local import path
    raise ImportError(f"Could not load base diagnostic script: {BASE_SCRIPT_PATH}")
spec.loader.exec_module(base)


TITLE = "Small Experiment - 256d Context Sweep with Lower+Upper Axial Features"
OUTPUT_PREFIX = "context_feature_sweep_256_plus_axial_both_triangles"
VARIANT_NAME = "baseline256_plus_axial_both_triangles"


base.TITLE = TITLE
base.SUMMARY_CSV_NAME = f"{OUTPUT_PREFIX}_variant_summary.csv"
base.LEADERBOARD_CSV_NAME = f"{OUTPUT_PREFIX}_leaderboard.csv"
base.PER_PROTEIN_CSV_NAME = f"{OUTPUT_PREFIX}_per_protein.csv"
base.SPLIT_CSV_NAME = f"{OUTPUT_PREFIX}_split.csv"
base.COEFFICIENT_CSV_NAME = f"{OUTPUT_PREFIX}_coefficients.csv"
base.ALPHA_CV_CSV_NAME = f"{OUTPUT_PREFIX}_alpha_cv.csv"
base.ALPHA_SELECTION_CSV_NAME = f"{OUTPUT_PREFIX}_alpha_selection.csv"
base.SUMMARY_TXT_NAME = f"{OUTPUT_PREFIX}_summary.txt"


VARIANT = base.make_variant(VARIANT_NAME, "256", use_axial=True)
VARIANT["use_upper_axial"] = True


def feature_names_for_variant(variant):
    """Return the ordered dense feature labels for the dual-axial experiment."""
    if variant["base_mode"] != "256":
        raise ValueError(f"Unsupported base_mode: {variant['base_mode']}")

    names = [f"pair_ij_ch_{idx}" for idx in range(base.N_CHANNELS)]
    names.extend([f"pair_ji_ch_{idx}" for idx in range(base.N_CHANNELS)])

    for offset in base.AXIAL_NEIGHBOR_OFFSETS:
        label = base.format_offset_label(offset)
        names.extend([f"lower_context_{label}_ch_{idx}" for idx in range(base.N_CHANNELS)])

    for offset in base.AXIAL_NEIGHBOR_OFFSETS:
        label = base.format_offset_label(offset)
        names.extend([f"upper_context_{label}_ch_{idx}" for idx in range(base.N_CHANNELS)])

    return names


def feature_dim_for_variant(variant):
    """Return the dense feature width for the fixed variant."""
    return len(feature_names_for_variant(variant))


def build_triangle_context_chunk(pb, center_i, center_j, offsets, chunk_dtype, triangle):
    """Build zero-padded axial context features for one requested triangle."""
    n_pairs = center_i.size
    chunk = np.zeros((n_pairs, len(offsets) * base.N_CHANNELS), dtype=chunk_dtype)
    n_residues = int(pb.shape[0])
    write_col = 0

    for di, dj in offsets:
        src_i = center_i + di
        src_j = center_j + dj
        valid = (
            (src_i >= 0)
            & (src_i < n_residues)
            & (src_j >= 0)
            & (src_j < n_residues)
        )
        if triangle == "lower":
            valid &= src_i > src_j
        elif triangle == "upper":
            valid &= src_i < src_j
        else:  # pragma: no cover - fixed caller set
            raise ValueError(f"Unsupported triangle: {triangle}")

        if np.any(valid):
            chunk[valid, write_col:write_col + base.N_CHANNELS] = pb[src_i[valid], src_j[valid], :]
        write_col += base.N_CHANNELS

    return chunk


def build_protein_feature_chunk(record, variant):
    """Load one protein pair block and build the 10-representation feature rows."""
    if variant["base_mode"] != "256":
        raise ValueError(f"Unsupported base_mode: {variant['base_mode']}")

    pb_path = base.get_pair_block_path(record["protein_dir"], record["protein_id"], base.LAYER)
    pb = np.load(str(pb_path), mmap_mode="r")
    pair_i = record["lt_i"]
    pair_j = record["lt_j"]
    chunk_dtype = base.chunk_dtype_for_variant(variant)

    base_ij = np.asarray(pb[pair_i, pair_j, :], dtype=chunk_dtype)
    base_ji = np.asarray(pb[pair_j, pair_i, :], dtype=chunk_dtype)
    lower_axial = build_triangle_context_chunk(
        pb=pb,
        center_i=pair_i,
        center_j=pair_j,
        offsets=base.AXIAL_NEIGHBOR_OFFSETS,
        chunk_dtype=chunk_dtype,
        triangle="lower",
    )
    upper_axial = build_triangle_context_chunk(
        pb=pb,
        center_i=pair_j,
        center_j=pair_i,
        offsets=base.AXIAL_NEIGHBOR_OFFSETS,
        chunk_dtype=chunk_dtype,
        triangle="upper",
    )

    del pb
    return np.concatenate([base_ij, base_ji, lower_axial, upper_axial], axis=1).astype(
        chunk_dtype,
        copy=False,
    )


base.feature_names_for_variant = feature_names_for_variant
base.feature_dim_for_variant = feature_dim_for_variant
base.build_protein_feature_chunk = build_protein_feature_chunk


def main():
    t0 = time.time()
    pair_blocks_root_env = os.environ.get("PAIR_BLOCKS_ROOT")
    pdb_root_env = os.environ.get("PDB_ROOT")
    csv_root_env = os.environ.get("CSV_ROOT")

    if not pair_blocks_root_env or not pdb_root_env or not csv_root_env:
        raise ValueError("PAIR_BLOCKS_ROOT, PDB_ROOT, and CSV_ROOT must all be set.")

    pair_blocks_root = Path(pair_blocks_root_env).resolve()
    pdb_root = Path(pdb_root_env).resolve()
    csv_dir = Path(csv_root_env).resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))
    print(f"Pair blocks root: {pair_blocks_root}")
    print(f"PDB root:         {pdb_root}")
    print(f"Output CSV dir:   {csv_dir}")
    print(f"Variant:          {VARIANT['name']}")
    print(f"Feature dim:      {feature_dim_for_variant(VARIANT)}")

    all_proteins, records, skipped = base.load_protein_records(pair_blocks_root, pdb_root)
    if not records:
        print("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    base.assign_size_bins(records, base.N_SIZE_BINS)
    train_records, test_records = base.split_records(records)
    split_rows = base.build_split_rows(train_records, test_records)

    print(f"Loaded proteins: {len(records)}")
    print(f"Train proteins:  {len(train_records)}")
    print(f"Test proteins:   {len(test_records)}")

    summary_rows = []
    per_protein_rows = []
    coefficient_rows = []
    alpha_cv_rows = []
    alpha_selection_rows = []

    base.flush_outputs(
        csv_dir=csv_dir,
        summary_rows=summary_rows,
        per_protein_rows=per_protein_rows,
        split_rows=split_rows,
        coefficient_rows=coefficient_rows,
        alpha_cv_rows=alpha_cv_rows,
        alpha_selection_rows=alpha_selection_rows,
        all_proteins=all_proteins,
        loaded_records=records,
        skipped=skipped,
        elapsed_seconds=float(time.time() - t0),
    )

    print(f"\n[Run] {VARIANT['name']}", flush=True)
    result = base.evaluate_variant(VARIANT, train_records, test_records)
    summary_rows.append(result["summary_row"])
    per_protein_rows.extend(result["per_protein_rows"])
    coefficient_rows.extend(result["coefficient_rows"])

    base.flush_outputs(
        csv_dir=csv_dir,
        summary_rows=summary_rows,
        per_protein_rows=per_protein_rows,
        split_rows=split_rows,
        coefficient_rows=coefficient_rows,
        alpha_cv_rows=alpha_cv_rows,
        alpha_selection_rows=alpha_selection_rows,
        all_proteins=all_proteins,
        loaded_records=records,
        skipped=skipped,
        elapsed_seconds=float(time.time() - t0),
    )

    print(
        "  "
        f"test mean per-protein R2={result['summary_row']['test_mean_per_protein_r2']:.4f}, "
        f"test pair-weighted R2={result['summary_row']['test_pair_weighted_r2']:.4f}"
    )

    print(f"\nSaved summary CSV -> {csv_dir / base.SUMMARY_CSV_NAME}")
    print(f"Saved leaderboard CSV -> {csv_dir / base.LEADERBOARD_CSV_NAME}")
    print(f"Saved per-protein CSV -> {csv_dir / base.PER_PROTEIN_CSV_NAME}")
    print(f"Saved split CSV -> {csv_dir / base.SPLIT_CSV_NAME}")
    print(f"Saved coefficient CSV -> {csv_dir / base.COEFFICIENT_CSV_NAME}")
    print(f"Saved alpha CV CSV -> {csv_dir / base.ALPHA_CV_CSV_NAME}")
    print(f"Saved alpha selection CSV -> {csv_dir / base.ALPHA_SELECTION_CSV_NAME}")
    print(f"Saved text summary -> {csv_dir / base.SUMMARY_TXT_NAME}")


if __name__ == "__main__":
    main()

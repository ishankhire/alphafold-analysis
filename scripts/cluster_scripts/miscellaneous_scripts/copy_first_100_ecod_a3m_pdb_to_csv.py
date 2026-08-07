#!/usr/bin/env python3
"""
Copy the first 100 TM-score filtered ECOD alignment and layer-47 PDB files to
Delta CSV storage.

This script copies, for each of the first 100 ECOD proteins in the shared
dataset order:
  - the MSA file at
    /work/hdd/bfna/thayes/ecod/alignments/{protein_id}/{protein_id}/bfd_uniclust_hits.a3m
  - the predicted layer-47 PDB file from the ECOD Delta outputs

Outputs go to:
  $HOME/csv_files/ecod_first100_inputs/

Within that directory, each protein gets its own subdirectory containing:
  - {protein_id}.a3m
  - {protein_id}_model_1_ptm_unrelaxed.pdb

The script also writes:
  - protein_file_manifest.csv
  - summary.txt
"""

import csv
import os
import shutil
import sys
from datetime import datetime
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
GENERALIZED_ECOD_DIR = CLUSTER_SCRIPTS_DIR / "generalized_model_ecod"
for _path in (CLUSTER_SCRIPTS_DIR, GENERALIZED_ECOD_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

from general_train_test_common import (  # noqa: E402
    ECOD_PAIR_BLOCKS_ROOT,
    ECOD_PDB_ROOT,
    ECOD_PROTEIN_IDS,
    resolve_pdb_file,
)


MAX_ECOD_PROTEINS = 100
LAST_LAYER = 47
A3M_ROOT = Path("/work/hdd/bfna/thayes/ecod/alignments")
CSV_ROOT = Path(os.environ.get("CSV_ROOT") or (Path.home() / "csv_files")).expanduser().resolve()
OUTPUT_BASENAME = "ecod_first100_inputs"
MANIFEST_CSV_NAME = "protein_file_manifest.csv"
SUMMARY_TXT_NAME = "summary.txt"

MANIFEST_FIELDNAMES = [
    "index",
    "protein_id",
    "a3m_source_path",
    "pdb_source_path",
    "protein_output_dir",
    "a3m_output_path",
    "pdb_output_path",
    "a3m_status",
    "pdb_status",
]


def now_timestamp():
    """Return a readable local timestamp for logs and summaries."""
    return datetime.now().astimezone().isoformat(timespec="seconds")


def log_progress(message):
    """Print one timestamped progress line."""
    print(f"[{now_timestamp()}] {message}", flush=True)


def selected_protein_ids():
    """Return the first 100 ECOD protein IDs in the shared manifest order."""
    return list(ECOD_PROTEIN_IDS[:MAX_ECOD_PROTEINS])


def resolve_a3m_path(protein_id):
    """Return the canonical ECOD alignment path supplied by the user."""
    return A3M_ROOT / protein_id / protein_id / "bfd_uniclust_hits.a3m"


def expected_pdb_path(protein_id):
    """Return the canonical layer-47 ECOD predicted PDB path."""
    return (
        ECOD_PDB_ROOT
        / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_{LAST_LAYER}"
        / "predictions"
        / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    )


def resolve_layer47_pdb_path(protein_id):
    """Resolve the best available ECOD predicted PDB path for one protein."""
    protein_dir = ECOD_PAIR_BLOCKS_ROOT / f"{protein_id}_save_layers_0_47"
    resolved = resolve_pdb_file(protein_id, protein_dir, ECOD_PDB_ROOT)
    if resolved is not None:
        return resolved
    return expected_pdb_path(protein_id)


def copy_if_present(source_path, destination_path):
    """Copy one file if present and return a compact status string."""
    if not source_path.exists():
        return "missing_source"
    destination_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source_path, destination_path)
    return "copied"


def write_manifest(path, rows):
    """Write the per-protein copy manifest."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=MANIFEST_FIELDNAMES)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def write_summary(path, output_root, manifest_rows):
    """Write a compact run summary for cluster-side inspection."""
    a3m_copied = sum(row["a3m_status"] == "copied" for row in manifest_rows)
    pdb_copied = sum(row["pdb_status"] == "copied" for row in manifest_rows)
    complete = sum(
        row["a3m_status"] == "copied" and row["pdb_status"] == "copied"
        for row in manifest_rows
    )
    failures = [
        row
        for row in manifest_rows
        if row["a3m_status"] != "copied" or row["pdb_status"] != "copied"
    ]

    with path.open("w") as handle:
        handle.write("ECOD first-100 input copy summary\n")
        handle.write(f"Timestamp:        {now_timestamp()}\n")
        handle.write(f"Output root:      {output_root}\n")
        handle.write(f"CSV root:         {CSV_ROOT}\n")
        handle.write(f"Proteins listed:  {len(manifest_rows)}\n")
        handle.write(f"Proteins complete:{complete}\n")
        handle.write(f"A3M copied:       {a3m_copied}\n")
        handle.write(f"PDB copied:       {pdb_copied}\n")
        handle.write(f"Proteins failed:  {len(failures)}\n")
        if failures:
            handle.write("\nFailures:\n")
            for row in failures:
                handle.write(
                    f"- {row['protein_id']}: "
                    f"a3m={row['a3m_status']}, pdb={row['pdb_status']}\n"
                )


def main():
    output_root = CSV_ROOT / OUTPUT_BASENAME
    output_root.mkdir(parents=True, exist_ok=True)

    protein_ids = selected_protein_ids()
    manifest_rows = []

    log_progress(f"CSV root:          {CSV_ROOT}")
    log_progress(f"Output root:       {output_root}")
    log_progress(f"Selected proteins: {len(protein_ids)}")

    for index, protein_id in enumerate(protein_ids, start=1):
        a3m_source = resolve_a3m_path(protein_id)
        pdb_source = resolve_layer47_pdb_path(protein_id)
        protein_output_dir = output_root / protein_id
        a3m_output = protein_output_dir / f"{protein_id}.a3m"
        pdb_output = protein_output_dir / pdb_source.name

        log_progress(
            f"[{index:03d}/{len(protein_ids):03d}] Copying {protein_id}"
        )
        a3m_status = copy_if_present(a3m_source, a3m_output)
        pdb_status = copy_if_present(pdb_source, pdb_output)

        manifest_rows.append(
            {
                "index": index,
                "protein_id": protein_id,
                "a3m_source_path": str(a3m_source),
                "pdb_source_path": str(pdb_source),
                "protein_output_dir": str(protein_output_dir),
                "a3m_output_path": str(a3m_output),
                "pdb_output_path": str(pdb_output),
                "a3m_status": a3m_status,
                "pdb_status": pdb_status,
            }
        )

    manifest_path = output_root / MANIFEST_CSV_NAME
    summary_path = output_root / SUMMARY_TXT_NAME
    write_manifest(manifest_path, manifest_rows)
    write_summary(summary_path, output_root, manifest_rows)

    failures = [
        row
        for row in manifest_rows
        if row["a3m_status"] != "copied" or row["pdb_status"] != "copied"
    ]
    log_progress(f"Manifest written to {manifest_path}")
    log_progress(f"Summary written to  {summary_path}")
    log_progress(f"Completed with {len(failures)} protein-level failures")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Read-only size inventory for the Delta storage audit; never deletes inputs.

Run with ~/protein_venv/bin/python from the canonical flat deployment. The
cohort manifest can be the 554-row master manifest or the 552-row outer folds.
Only file metadata and a bounded sample of NPY headers are read, never arrays.
"""

from __future__ import annotations

import argparse
import ast
import csv
import os
import re
import stat
import struct
import time
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOTS = (
    "/work/hdd/bekh/thayes/openfold_reps",
    "/work/hdd/bekh/thayes/openfold_reps_outputs",
    "/work/hdd/bekh/thayes/openfold_reps_outputs_sm",
    "/work/hdd/bfna/thayes/ecod/experiments",
    "/work/hdd/bfna/thayes/ecod/experiments_TRI_END_FIX",
    "/projects/bekh/thayes/alignments",
    "/work/hdd/bfna/thayes/ecod/alignments",
    "/projects/bekh/thayes/final_cameo_outputs",
    "/projects/bekh/thayes/final_cameo_outputs_ae",
    "/work/nvme/bekh/thayes/cameo_reps",
    "/work/nvme/bekh/thayes/cameo_triplets",
    "/work/hdd/bfna/cameo/openfold_outputs_layerwise_ablation",
    "/work/hdd/bekh/openfold_outputs_layerwise_ablation",
    "/work/hdd/bfna/openfold_outputs_layerwise_ablation",
    "/work/hdd/bfna/openfold_outputs_layerwise_ablation_TRI_END_FIX",
    "/work/nvme/bekh/openfold_outputs_layerwise_ablation",
    "/work/hdd/bekh/openfold_outputs_global_ablations",
    "/work/hdd/bfna/openfold_outputs_global_ablations_TRI_END_FIX",
    "/work/hdd/bfna/openfold_outputs_fixed_ablation",
    "/work/hdd/bfna/thayes/FINAL_RESULT_CSVS",
    "/work/hdd/bfna/thayes/FINAL_RESULTS_CSVS_TRI_END_FIX_PATCHED_FROM_OLD",
    "/work/hdd/bfna/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_PATCHED_FROM_OLD",
    "/work/hdd/bfna/thayes/FINAL_RESULT_CSVS_TRI_END_FIX_STANDARDIZED",
    "/work/hdd/bfna/cameo/layerwise_tm_csvs",
    "/work/hdd/bfna/cameo/global_tm_csvs",
    "/work/hdd/bfna/cameo/layerwise_tm_csvs_TRI_END_FIX",
    "/work/hdd/bfna/cameo/global_tm_csvs_TRI_END_FIX",
    "/work/hdd/bfna/thayes/ecod/layerwise_tm_csvs_TRI_END_FIX_STANDARDIZED",
    "/work/hdd/bfna/thayes/ecod/global_tm_csvs_TRI_END_FIX_STANDARDIZED",
    "/work/hdd/bfna/thayes/cameo_data/pairwise_dist_mats",
)
PAIR_RE = re.compile(r"(?P<protein>.+)_pair_block_(?P<layer>\d+)\.npy$")
PROTEIN_RE = re.compile(r"(?<![A-Za-z0-9])(?:\d{9}|[0-9][A-Za-z0-9]{3}_[A-Za-z0-9])(?=$|[^A-Za-z0-9])")


def log(message: str) -> None:
    print(f"[{datetime.now(timezone.utc).isoformat()}] {message}", flush=True)


def load_cohort(path: Path) -> set[str]:
    with path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    proteins = {row["protein_id"] for row in rows}
    if not proteins or len(proteins) != len(rows):
        raise ValueError("Cohort manifest must contain unique, nonempty protein IDs")
    # Keep excluded master-manifest IDs too: older workflows still use 554 IDs.
    return proteins


def file_category(path: Path) -> str:
    name = path.name.lower()
    if PAIR_RE.fullmatch(path.name):
        return "pair_tensor"
    if "msa_block_" in name and name.endswith(".npy"):
        return "msa_tensor"
    if "pair_block_" in name and name.endswith(".npy"):
        return "pair_tensor_variant"
    if name.endswith((".pdb", ".pdb.gz", ".cif", ".cif.gz", ".mmcif")):
        return "structure"
    if name.endswith((".a3m", ".sto", ".hhr", ".aln")):
        return "alignment_or_template_search"
    if name.endswith((".pkl", ".pickle")):
        return "pickle_inspect_contents_before_cleanup"
    if "attn" in str(path).lower() or "attention" in name:
        return "attention_or_triplet"
    if name.endswith((".npy", ".npz", ".h5", ".hdf5", ".pt", ".pth")):
        return "other_array_or_model"
    if name.endswith((".out", ".err", ".log")):
        return "log"
    if name.endswith((".csv", ".json", ".tsv", ".txt")):
        return "table_metadata_or_text"
    return "other"


def cohort_scope(path: Path, proteins: set[str]) -> str:
    ids = set(PROTEIN_RE.findall(str(path)))
    if ids & proteins:
        return "cohort_id_present"
    return "other_id_present" if ids else "no_id_detected"


def npy_header(path: Path) -> dict:
    with path.open("rb") as handle:
        if handle.read(6) != b"\x93NUMPY":
            raise ValueError("Invalid NPY magic")
        version = tuple(handle.read(2))
        if version == (1, 0):
            width, fmt, encoding = 2, "<H", "latin1"
        elif version in ((2, 0), (3, 0)):
            width, fmt = 4, "<I"
            encoding = "utf8" if version == (3, 0) else "latin1"
        else:
            raise ValueError(f"Unsupported NPY version {version}")
        length = struct.unpack(fmt, handle.read(width))[0]
        if length > 1_000_000:
            raise ValueError("NPY header exceeds the inventory safety bound")
        return ast.literal_eval(handle.read(length).decode(encoding))


def nonoverlapping_roots(roots: list[Path]) -> list[Path]:
    result: list[Path] = []
    # Use absolute lexical paths; do not follow directory symlinks.
    for root in sorted(set(roots), key=lambda p: (len(p.parts), str(p))):
        if not any(root == parent or parent in root.parents for parent in result):
            result.append(root)
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", action="append", type=Path,
                        help="Scan only these roots; repeat for multiple roots")
    parser.add_argument("--cohort-manifest", type=Path,
                        default=Path.home() / "openfold_interpretability_canonical/regression_protocol_data/regression_master_split_v1.csv")
    parser.add_argument("--output-dir", type=Path,
                        default=Path.home() / "csv_files/storage_audit_2026-10-07")
    parser.add_argument("--npy-header-samples", type=int, default=3,
                        help="Maximum header samples per root/category/cohort group")
    args = parser.parse_args()
    if args.npy_header_samples < 0:
        parser.error("--npy-header-samples must be nonnegative")
    proteins = load_cohort(args.cohort_manifest.expanduser())
    output = Path(os.path.abspath(args.output_dir.expanduser()))
    roots = nonoverlapping_roots([Path(os.path.abspath(p.expanduser())) for p in args.root] if args.root else
                                [Path(p) for p in DEFAULT_ROOTS] + [Path.home() / "csv_files"])
    if any(root == output or output in root.parents for root in roots):
        parser.error("A scan root cannot be the output directory or inside it")
    output.mkdir(parents=True, exist_ok=False)
    summaries = defaultdict(lambda: [0, 0, 0])
    root_rows = []
    header_counts = defaultdict(int)
    seen_hardlinks = set()
    file_count = error_count = 0
    started = time.monotonic()
    with (output / "files.csv").open("w", newline="") as file_handle, \
         (output / "errors.csv").open("w", newline="") as error_handle, \
         (output / "npy_header_samples.csv").open("w", newline="") as header_handle:
        files = csv.writer(file_handle, lineterminator="\n")
        errors = csv.writer(error_handle, lineterminator="\n")
        headers = csv.writer(header_handle, lineterminator="\n")
        files.writerow(["root", "path", "category", "cohort_scope", "logical_bytes",
                        "allocated_bytes", "unique_allocated_bytes", "mtime_ns", "nlink"])
        errors.writerow(["root", "path", "error"])
        headers.writerow(["root", "path", "category", "cohort_scope", "shape", "dtype", "fortran_order"])

        def record_error(root: Path, path: object, message: object) -> None:
            nonlocal error_count
            errors.writerow([root, path, str(message)])
            error_count += 1

        for root in roots:
            root_started = time.monotonic()
            before_errors, before_files = error_count, file_count
            log(f"scan started root={root}")
            try:
                root_stat = root.lstat()
                if not stat.S_ISDIR(root_stat.st_mode):
                    raise ValueError("Root is not a real directory (symlinks are not followed)")
            except (OSError, ValueError) as exc:
                record_error(root, root, exc)
                root_rows.append([str(root), "unavailable", 0, error_count - before_errors, 0])
                continue
            for directory, directories, names in os.walk(
                root, followlinks=False,
                onerror=lambda exc: record_error(root, exc.filename, exc),
            ):
                directory_path = Path(directory)
                directories[:] = [name for name in directories
                                  if directory_path / name != output]
                for name in names:
                    path = directory_path / name
                    try:
                        metadata = path.lstat()
                        if not stat.S_ISREG(metadata.st_mode):
                            continue
                        category = file_category(path)
                        scope = cohort_scope(path, proteins)
                        key = (str(root), category, scope)
                        allocated = metadata.st_blocks * 512
                        inode = (metadata.st_dev, metadata.st_ino)
                        unique_allocated = allocated
                        if metadata.st_nlink > 1:
                            unique_allocated = 0 if inode in seen_hardlinks else allocated
                            seen_hardlinks.add(inode)
                        files.writerow([root, path, category, scope, metadata.st_size,
                                        allocated, unique_allocated, metadata.st_mtime_ns, metadata.st_nlink])
                        totals = summaries[key]
                        totals[0] += 1
                        totals[1] += metadata.st_size
                        totals[2] += unique_allocated
                        file_count += 1
                        if path.suffix == ".npy" and header_counts[key] < args.npy_header_samples:
                            header_counts[key] += 1
                            header = npy_header(path)
                            headers.writerow([root, path, category, scope, header["shape"],
                                              header["descr"], header["fortran_order"]])
                    except (OSError, ValueError, SyntaxError, KeyError, struct.error) as exc:
                        record_error(root, path, exc)
                    if file_count and file_count % 100_000 == 0:
                        log(f"progress files={file_count} elapsed_seconds={time.monotonic() - started:.1f}")
            elapsed = time.monotonic() - root_started
            root_rows.append([str(root), "partial" if error_count > before_errors else "complete",
                              file_count - before_files, error_count - before_errors, f"{elapsed:.3f}"])
            file_handle.flush()
            error_handle.flush()
            header_handle.flush()
            log(f"scan finished root={root} files={file_count - before_files} elapsed_seconds={elapsed:.1f}")
    with (output / "summary.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["root", "category", "cohort_scope", "file_count", "logical_bytes",
                         "unique_allocated_bytes", "logical_TB", "unique_allocated_TB"])
        for key, values in sorted(summaries.items(), key=lambda item: -item[1][2]):
            writer.writerow([*key, *values, f"{values[1] / 1e12:.9f}", f"{values[2] / 1e12:.9f}"])
    with (output / "roots.csv").open("w", newline="") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["root", "status", "file_count", "error_count", "elapsed_seconds"])
        writer.writerows(root_rows)
    log(f"inventory complete files={file_count} errors={error_count} output={output} elapsed_seconds={time.monotonic() - started:.1f}")
    if error_count:
        log("Inventory is partial; consult roots.csv and errors.csv before using totals.")


if __name__ == "__main__":
    main()

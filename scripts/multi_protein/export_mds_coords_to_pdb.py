#!/usr/bin/env python3
"""
export_mds_coords_to_pdb.py
===========================
Convert reconstructed residue coordinates from a CSV into PyMOL-friendly
CA-trace PDB files.

By default this reads:
    delta_files/mds_reconstruction/csv_files/
        mds_reconstructed_coords_layer47.csv

and writes:
    delta_files/mds_reconstruction/pdb_files/
        mds_reconstructed_coords_layer47/
            <protein_id>_mds_reconstructed_ca.pdb
            <protein_id>_reference_ca.pdb

The reconstructed PDB stores per-residue deviation in the B-factor column so
PyMOL can color the CA trace by reconstruction error.
"""

from __future__ import annotations

import argparse
import csv
from collections import OrderedDict
from dataclasses import dataclass
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT
    / "delta_files"
    / "mds_reconstruction"
    / "csv_files"
    / "mds_reconstructed_coords_layer47.csv"
)
DEFAULT_OUTPUT_PARENT = (
    PROJECT_ROOT
    / "delta_files"
    / "mds_reconstruction"
    / "pdb_files"
)

REQUIRED_COLUMNS = {
    "protein_id",
    "residue_idx",
    "residue_name",
    "residue_number",
    "mds_x",
    "mds_y",
    "mds_z",
    "pdb_x",
    "pdb_y",
    "pdb_z",
    "deviation_angstrom",
}


@dataclass(frozen=True)
class ResidueRow:
    protein_id: str
    residue_idx: int
    residue_name: str
    residue_number: int
    mds_x: float
    mds_y: float
    mds_z: float
    pdb_x: float
    pdb_y: float
    pdb_z: float
    deviation_angstrom: float


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Export reconstructed MDS coordinates to CA-trace PDB files."
    )
    parser.add_argument(
        "--input-csv",
        type=Path,
        default=DEFAULT_INPUT_CSV,
        help="Path to a reconstructed-coordinate CSV.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help=(
            "Directory where PDB files will be written. Defaults to "
            "delta_files/mds_reconstruction/pdb_files/<input_csv_stem>/"
        ),
    )
    return parser.parse_args()


def derive_chain_id(protein_id: str) -> str:
    suffix = protein_id.rsplit("_", 1)[-1].strip()
    return suffix[0] if suffix else "A"


def parse_csv(input_csv: Path) -> "OrderedDict[str, list[ResidueRow]]":
    if not input_csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {input_csv}")

    grouped_rows: "OrderedDict[str, list[ResidueRow]]" = OrderedDict()

    with input_csv.open(newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None:
            raise ValueError(f"CSV is missing a header row: {input_csv}")

        missing = REQUIRED_COLUMNS.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                f"CSV is missing required columns: {', '.join(sorted(missing))}"
            )

        for row in reader:
            parsed = ResidueRow(
                protein_id=row["protein_id"].strip(),
                residue_idx=int(row["residue_idx"]),
                residue_name=row["residue_name"].strip().upper(),
                residue_number=int(row["residue_number"]),
                mds_x=float(row["mds_x"]),
                mds_y=float(row["mds_y"]),
                mds_z=float(row["mds_z"]),
                pdb_x=float(row["pdb_x"]),
                pdb_y=float(row["pdb_y"]),
                pdb_z=float(row["pdb_z"]),
                deviation_angstrom=float(row["deviation_angstrom"]),
            )
            grouped_rows.setdefault(parsed.protein_id, []).append(parsed)

    if not grouped_rows:
        raise ValueError(f"CSV contains no residue rows: {input_csv}")

    return grouped_rows


def atom_name_field(atom_name: str) -> str:
    if len(atom_name) >= 4:
        return atom_name[:4]
    return f" {atom_name:<3s}"


def format_atom_line(
    serial: int,
    residue_name: str,
    chain_id: str,
    residue_number: int,
    x: float,
    y: float,
    z: float,
    occupancy: float,
    bfactor: float,
) -> str:
    atom_field = atom_name_field("CA")
    residue_field = residue_name[:3].upper()
    safe_bfactor = min(max(bfactor, 0.0), 999.99)
    return (
        f"ATOM  {serial:5d} {atom_field}{' ':1s}{residue_field:>3s} {chain_id:1s}"
        f"{residue_number:4d}{' ':1s}   "
        f"{x:8.3f}{y:8.3f}{z:8.3f}"
        f"{occupancy:6.2f}{safe_bfactor:6.2f}          "
        f"{'C':>2s}\n"
    )


def format_ter_line(serial: int, residue_name: str, chain_id: str, residue_number: int) -> str:
    residue_field = residue_name[:3].upper()
    return (
        f"TER   {serial:5d}      {residue_field:>3s} {chain_id:1s}"
        f"{residue_number:4d}\n"
    )


def format_conect_line(serial_a: int, serial_b: int) -> str:
    return f"CONECT{serial_a:5d}{serial_b:5d}\n"


def write_pdb(
    output_path: Path,
    protein_id: str,
    rows: list[ResidueRow],
    coord_prefix: str,
    bfactor_mode: str,
) -> None:
    chain_id = derive_chain_id(protein_id)

    with output_path.open("w") as handle:
        handle.write(f"REMARK CA trace exported from {protein_id}\n")
        handle.write(f"REMARK Coordinates source: {coord_prefix}\n")
        if bfactor_mode == "deviation":
            handle.write("REMARK B-factor stores per-residue deviation_angstrom\n")
        else:
            handle.write("REMARK B-factor set to 0.00 for reference coordinates\n")

        for serial, row in enumerate(rows, start=1):
            x = getattr(row, f"{coord_prefix}_x")
            y = getattr(row, f"{coord_prefix}_y")
            z = getattr(row, f"{coord_prefix}_z")
            bfactor = row.deviation_angstrom if bfactor_mode == "deviation" else 0.0
            handle.write(
                format_atom_line(
                    serial=serial,
                    residue_name=row.residue_name,
                    chain_id=chain_id,
                    residue_number=row.residue_number,
                    x=x,
                    y=y,
                    z=z,
                    occupancy=1.00,
                    bfactor=bfactor,
                )
            )

        last_row = rows[-1]
        handle.write(
            format_ter_line(
                serial=len(rows) + 1,
                residue_name=last_row.residue_name,
                chain_id=chain_id,
                residue_number=last_row.residue_number,
            )
        )

        for serial in range(1, len(rows)):
            handle.write(format_conect_line(serial, serial + 1))

        handle.write("END\n")


def export_grouped_rows(grouped_rows: "OrderedDict[str, list[ResidueRow]]", output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)

    for protein_id, rows in grouped_rows.items():
        reconstructed_path = output_dir / f"{protein_id}_mds_reconstructed_ca.pdb"
        reference_path = output_dir / f"{protein_id}_reference_ca.pdb"

        write_pdb(
            output_path=reconstructed_path,
            protein_id=protein_id,
            rows=rows,
            coord_prefix="mds",
            bfactor_mode="deviation",
        )
        write_pdb(
            output_path=reference_path,
            protein_id=protein_id,
            rows=rows,
            coord_prefix="pdb",
            bfactor_mode="zero",
        )

        print(
            f"{protein_id}: {len(rows)} residues -> "
            f"{reconstructed_path} ; {reference_path}"
        )


def main() -> int:
    args = parse_args()
    input_csv = args.input_csv.resolve()
    output_dir = (
        args.output_dir.resolve()
        if args.output_dir is not None
        else (DEFAULT_OUTPUT_PARENT / input_csv.stem).resolve()
    )

    grouped_rows = parse_csv(input_csv)
    export_grouped_rows(grouped_rows, output_dir)

    print(f"Wrote {2 * len(grouped_rows)} PDB files to {output_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

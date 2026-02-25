from pathlib import Path

import numpy as np

from lib.output_writers import ensure_output_dirs, save_coordinates_csv, save_distance_csv


def test_output_writer_helpers(tmp_path: Path):
    csv_dir, vis_dir = ensure_output_dirs(tmp_path)
    assert csv_dir.exists()
    assert vis_dir.exists()

    residue_data = {
        "residues": [("A", "ALA", 1), ("A", "GLY", 2)],
        "coordinates": np.array([[1.0, 2.0, 3.0], [3.0, 2.0, 1.0]]),
    }
    coords_csv = csv_dir / "coords.csv"
    dist_csv = csv_dir / "dist.csv"
    save_coordinates_csv(coords_csv, residue_data)
    save_distance_csv(dist_csv, np.array([[0.0, 1.2], [1.2, 0.0]]), residue_data)
    assert coords_csv.exists()
    assert dist_csv.exists()

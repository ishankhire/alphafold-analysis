"""
Shared output helpers for local scripts.
"""

import csv


def ensure_output_dirs(output_dir):
    """Create and return (csv_dir, vis_dir) under output_dir."""
    csv_dir = output_dir / "csv_files"
    vis_dir = output_dir / "visualizations"
    csv_dir.mkdir(parents=True, exist_ok=True)
    vis_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, vis_dir


def save_coordinates_csv(path, residue_data):
    """Save residue coordinate table (chain, residue, number, x, y, z)."""
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["chain", "residue", "number", "x", "y", "z"])
        for res, coord in zip(residue_data["residues"], residue_data["coordinates"]):
            writer.writerow(
                [res[0], res[1], res[2], f"{coord[0]:.3f}", f"{coord[1]:.3f}", f"{coord[2]:.3f}"]
            )


def save_distance_csv(path, dist_matrix, residue_data):
    """Save square distance matrix with residue labels as header row."""
    labels = [f"{res[0]}_{res[1]}{res[2]}" for res in residue_data["residues"]]
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(labels)
        for row in dist_matrix:
            writer.writerow([f"{d:.4f}" for d in row])

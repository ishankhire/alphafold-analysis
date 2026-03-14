#!/usr/bin/env python3
"""
diagnostic_regression.py
========================
Diagnostic script to investigate the R-squared drop observed when scaling from
25 to 184 proteins.  Runs layer-47-only linear regression on 3 random samples
of 100 proteins (each <= 500 residues) with an 80/20 train/test split.

No k-fold CV, no MDS, no triangle inequality analysis.  Focus is entirely on
per-protein and pooled R-squared to diagnose heterogeneity and scale effects.

Outputs (inside CSV_ROOT/):
    diagnostic_pooled_metrics.csv       — one row per sample run
    diagnostic_per_protein_metrics.csv  — one row per (sample, protein)
    diagnostic_summary.txt              — human-readable summary
"""

import csv
import math
import os
import sys
import time
import traceback
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

from Bio.PDB import PDBParser


# ===========================================================================
# Configuration
# ===========================================================================

N_CHANNELS = 128
FEATURE_DIM = 2 * N_CHANNELS  # 256
LAYER = 47                     # only layer 47
TEST_SIZE = 0.2                # 80/20 split
MAX_RESIDUES = 500             # exclude proteins with > 500 residues
N_SAMPLES = 3                  # 3 random subsets
SAMPLE_SIZE = 100              # 100 proteins per sample
SAMPLE_SEEDS = [42, 123, 456]  # one seed per sample run


# ===========================================================================
# Directories (set via environment variables exported by sbatch files)
# ===========================================================================

PAIR_BLOCKS_ROOT = Path(os.environ["PAIR_BLOCKS_ROOT"])
PDB_ROOT = Path(os.environ["PDB_ROOT"])
CSV_DIR = Path(os.environ["CSV_ROOT"])


# ===========================================================================
# Helper functions — data loading
# ===========================================================================

def find_pdb_file(protein_id):
    """Locate the predicted PDB file for a protein."""
    pdb_path = (
        PDB_ROOT
        / f"my_outputs_align_{protein_id}_representations_load_from_saved_layer_47"
        / "predictions"
        / f"{protein_id}_model_1_ptm_unrelaxed.pdb"
    )
    return pdb_path if pdb_path.exists() else None


def get_pair_block_path(protein_dir, protein_id, layer):
    """Return the path to a pair block .npy file for a given layer."""
    return protein_dir / f"{protein_id}_pair_block_{layer}.npy"


def discover_proteins(pair_blocks_root):
    """Find all protein subdirectories under the pair blocks root."""
    proteins = []
    for entry in sorted(pair_blocks_root.iterdir()):
        if not entry.is_dir():
            continue
        proteins.append((entry.name, entry))
    return proteins


# ===========================================================================
# Helper functions — coordinate extraction & distance computation
# ===========================================================================

def extract_ca_coordinates(pdb_path):
    """Extract CA atom coordinates from a PDB file across all chains."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", str(pdb_path))

    residues = []
    coordinates = []

    model = structure[0]
    for chain in model:
        for residue in chain:
            het_flag = residue.get_id()[0]
            if het_flag == "W":
                continue
            if "CA" in residue:
                ca = residue["CA"]
                residues.append((
                    chain.get_id(),
                    residue.get_resname(),
                    residue.get_id()[1],
                ))
                coordinates.append(ca.get_coord())

    return {
        "residues": residues,
        "coordinates": np.array(coordinates) if coordinates else np.empty((0, 3)),
    }


def compute_distance_matrix(coords):
    """Vectorized pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


# ===========================================================================
# Helper — build per-protein feature chunk
# ===========================================================================

def build_feature_chunk(protein_record, layer):
    """Load one protein/layer pair block and extract its dense feature chunk."""
    pb_path = get_pair_block_path(
        protein_record["protein_dir"], protein_record["protein_id"], layer
    )
    pb = np.load(str(pb_path))
    chunk = np.concatenate([
        pb[protein_record["pb_i"], protein_record["pb_j"], :],
        pb[protein_record["pb_j"], protein_record["pb_i"], :],
    ], axis=1)
    del pb
    return chunk


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()

    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(f"{'=' * 60}")
    print("DIAGNOSTIC REGRESSION — layer 47, 3 random samples of 100")
    print(f"{'=' * 60}")
    print(f"Pair blocks root:   {PAIR_BLOCKS_ROOT}")
    print(f"PDB root:           {PDB_ROOT}")
    print(f"Output CSV dir:     {csv_dir}")
    print(f"Layer:              {LAYER}")
    print(f"Max residues:       {MAX_RESIDUES}")
    print(f"Sample size:        {SAMPLE_SIZE}")
    print(f"Sample seeds:       {SAMPLE_SEEDS}")
    print(f"Split strategy:     80/20 train/test (protein-level)")

    # ------------------------------------------------------------------
    # Phase 1: Discover and validate all proteins
    # ------------------------------------------------------------------
    all_proteins = discover_proteins(PAIR_BLOCKS_ROOT)
    print(f"\nDiscovered {len(all_proteins)} protein directories total.")

    protein_data = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        try:
            pdb_path = find_pdb_file(protein_id)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                print(f"[SKIP] {protein_id}: no predicted PDB file found")
                continue

            pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
            if not pb0_path.exists():
                skipped.append((protein_id, "pair block 0 not found"))
                print(f"[SKIP] {protein_id}: pair block 0 not found at {pb0_path}")
                continue

            ca_data = extract_ca_coordinates(pdb_path)
            n_ca = len(ca_data["residues"])
            if n_ca == 0:
                skipped.append((protein_id, "no CA atoms found in PDB"))
                print(f"[SKIP] {protein_id}: no CA atoms found in PDB")
                continue

            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = pb0.shape[0]
            del pb0

            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                print(f"[SKIP] {protein_id}: CA count {n_ca} != pair block dim {pb_dim}")
                continue

            dist_matrix = compute_distance_matrix(ca_data["coordinates"])
            struct_i, struct_j = np.triu_indices(n_ca, k=1)
            y_protein = dist_matrix[struct_i, struct_j]

            protein_data.append({
                "protein_id": protein_id,
                "protein_dir": protein_dir,
                "n_residues": n_ca,
                "coordinates": ca_data["coordinates"],
                "pb_i": struct_i,
                "pb_j": struct_j,
                "y": y_protein,
                "mean_distance": float(np.mean(y_protein)),
                "std_distance": float(np.std(y_protein)),
                "max_distance": float(np.max(y_protein)),
                "n_pairs": len(y_protein),
            })
            print(f"[OK]   {protein_id}: {n_ca} residues, {len(y_protein):,} pairs")

        except Exception as exc:
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {exc}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(protein_data)} proteins (skipped {len(skipped)}).")

    # ------------------------------------------------------------------
    # Phase 2: Filter by residue count
    # ------------------------------------------------------------------
    before_filter = len(protein_data)
    excluded_proteins = [d for d in protein_data if d["n_residues"] > MAX_RESIDUES]
    protein_data = [d for d in protein_data if d["n_residues"] <= MAX_RESIDUES]
    after_filter = len(protein_data)
    n_excluded = before_filter - after_filter

    print(f"\n{'=' * 60}")
    print(f"RESIDUE FILTER: excluded {n_excluded} proteins with > {MAX_RESIDUES} residues")
    print(f"  ({before_filter} -> {after_filter} proteins remaining)")
    if excluded_proteins:
        print("  Excluded proteins:")
        for d in sorted(excluded_proteins, key=lambda x: x["n_residues"], reverse=True):
            print(f"    {d['protein_id']}: {d['n_residues']} residues")
    print(f"{'=' * 60}")

    if after_filter < SAMPLE_SIZE:
        print(
            f"\nERROR: Only {after_filter} proteins remain after filtering, "
            f"but need {SAMPLE_SIZE} for sampling. Exiting."
        )
        sys.exit(1)

    # Print residue count distribution of remaining proteins
    residue_counts = [d["n_residues"] for d in protein_data]
    print(f"\nRemaining protein residue distribution:")
    print(f"  Min: {min(residue_counts)}, Max: {max(residue_counts)}, "
          f"Mean: {np.mean(residue_counts):.1f}, Median: {np.median(residue_counts):.1f}")

    # ------------------------------------------------------------------
    # Phase 3: Sample loop
    # ------------------------------------------------------------------
    pooled_rows = []
    per_protein_rows = []

    for sample_idx in range(N_SAMPLES):
        seed = SAMPLE_SEEDS[sample_idx]
        rng = np.random.RandomState(seed)
        sample_indices = rng.choice(len(protein_data), size=SAMPLE_SIZE, replace=False)
        sample_proteins = [protein_data[i] for i in sample_indices]

        print(f"\n{'=' * 60}")
        print(f"SAMPLE {sample_idx} (seed={seed}): {SAMPLE_SIZE} proteins")
        sample_residues = [d["n_residues"] for d in sample_proteins]
        sample_pairs = sum(d["n_pairs"] for d in sample_proteins)
        print(f"  Residues: min={min(sample_residues)}, max={max(sample_residues)}, "
              f"mean={np.mean(sample_residues):.1f}, median={np.median(sample_residues):.1f}")
        print(f"  Total pairs: {sample_pairs:,}")

        # 80/20 train/test split (protein-level)
        train_proteins, test_proteins = train_test_split(
            sample_proteins, test_size=TEST_SIZE, random_state=seed
        )

        y_train = np.concatenate([d["y"] for d in train_proteins])
        y_test = np.concatenate([d["y"] for d in test_proteins])

        # Build offset tracking
        train_offsets = []
        offset = 0
        for d in train_proteins:
            n_pairs = len(d["y"])
            train_offsets.append((d["protein_id"], offset, offset + n_pairs))
            offset += n_pairs

        test_offsets = []
        offset = 0
        for d in test_proteins:
            n_pairs = len(d["y"])
            test_offsets.append((d["protein_id"], offset, offset + n_pairs))
            offset += n_pairs

        print(f"  Train: {len(train_proteins)} proteins, {len(y_train):,} pairs")
        print(f"  Test:  {len(test_proteins)} proteins, {len(y_test):,} pairs")

        # Load layer 47 features
        print(f"  Loading layer {LAYER} pair blocks...")
        t_load = time.time()
        chunk_map = {}
        for d in sample_proteins:
            chunk_map[d["protein_id"]] = build_feature_chunk(d, LAYER)
        print(f"  Loaded in {time.time() - t_load:.1f}s")

        X_train_raw = np.vstack([chunk_map[d["protein_id"]] for d in train_proteins])
        X_test_raw = np.vstack([chunk_map[d["protein_id"]] for d in test_proteins])

        # Fit model
        print(f"  Fitting StandardScaler + LinearRegression...")
        t_fit = time.time()
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train_raw)
        X_test = scaler.transform(X_test_raw)

        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        print(f"  Fit in {time.time() - t_fit:.1f}s")

        # Pooled metrics
        r2_train = r2_score(y_train, y_pred_train)
        r2_test = r2_score(y_test, y_pred_test)
        mae_train = mean_absolute_error(y_train, y_pred_train)
        mae_test = mean_absolute_error(y_test, y_pred_test)
        rmse_train = math.sqrt(np.mean((y_train - y_pred_train) ** 2))
        rmse_test = math.sqrt(np.mean((y_test - y_pred_test) ** 2))

        # Per-protein metrics (train)
        sample_per_protein = []
        for d, (pid, start, end) in zip(train_proteins, train_offsets):
            y_true_p = y_train[start:end]
            y_pred_p = y_pred_train[start:end]
            per_r2 = float(r2_score(y_true_p, y_pred_p)) if end - start > 1 else float("nan")
            per_mae = float(mean_absolute_error(y_true_p, y_pred_p))
            per_rmse = float(math.sqrt(np.mean((y_true_p - y_pred_p) ** 2)))
            residuals = y_true_p - y_pred_p

            # Feature variance for this protein
            feat_chunk = chunk_map[pid]
            mean_feat_var = float(np.mean(np.var(feat_chunk, axis=0)))

            row = {
                "sample_idx": sample_idx,
                "seed": seed,
                "protein_id": pid,
                "split": "train",
                "n_residues": d["n_residues"],
                "n_pairs": d["n_pairs"],
                "per_protein_r2": per_r2,
                "per_protein_mae": per_mae,
                "per_protein_rmse": per_rmse,
                "mean_true_distance": d["mean_distance"],
                "std_true_distance": d["std_distance"],
                "max_true_distance": d["max_distance"],
                "mean_feature_variance": mean_feat_var,
                "mean_residual": float(np.mean(residuals)),
                "std_residual": float(np.std(residuals)),
            }
            sample_per_protein.append(row)
            per_protein_rows.append(row)

        # Per-protein metrics (test)
        for d, (pid, start, end) in zip(test_proteins, test_offsets):
            y_true_p = y_test[start:end]
            y_pred_p = y_pred_test[start:end]
            per_r2 = float(r2_score(y_true_p, y_pred_p)) if end - start > 1 else float("nan")
            per_mae = float(mean_absolute_error(y_true_p, y_pred_p))
            per_rmse = float(math.sqrt(np.mean((y_true_p - y_pred_p) ** 2)))
            residuals = y_true_p - y_pred_p

            feat_chunk = chunk_map[pid]
            mean_feat_var = float(np.mean(np.var(feat_chunk, axis=0)))

            row = {
                "sample_idx": sample_idx,
                "seed": seed,
                "protein_id": pid,
                "split": "test",
                "n_residues": d["n_residues"],
                "n_pairs": d["n_pairs"],
                "per_protein_r2": per_r2,
                "per_protein_mae": per_mae,
                "per_protein_rmse": per_rmse,
                "mean_true_distance": d["mean_distance"],
                "std_true_distance": d["std_distance"],
                "max_true_distance": d["max_distance"],
                "mean_feature_variance": mean_feat_var,
                "mean_residual": float(np.mean(residuals)),
                "std_residual": float(np.std(residuals)),
            }
            sample_per_protein.append(row)
            per_protein_rows.append(row)

        # Summarize per-protein test R2
        test_r2s = [
            r["per_protein_r2"] for r in sample_per_protein
            if r["split"] == "test" and not math.isnan(r["per_protein_r2"])
        ]
        n_negative_r2_test = sum(1 for r in test_r2s if r < 0)
        mean_per_protein_r2_train = float(np.mean([
            r["per_protein_r2"] for r in sample_per_protein
            if r["split"] == "train" and not math.isnan(r["per_protein_r2"])
        ]))
        mean_per_protein_r2_test = float(np.mean(test_r2s)) if test_r2s else float("nan")
        median_per_protein_r2_test = float(np.median(test_r2s)) if test_r2s else float("nan")

        pooled_rows.append({
            "sample_idx": sample_idx,
            "seed": seed,
            "n_train_proteins": len(train_proteins),
            "n_test_proteins": len(test_proteins),
            "n_train_pairs": len(y_train),
            "n_test_pairs": len(y_test),
            "r2_train": r2_train,
            "r2_test": r2_test,
            "mae_train": mae_train,
            "mae_test": mae_test,
            "rmse_train": rmse_train,
            "rmse_test": rmse_test,
            "mean_per_protein_r2_train": mean_per_protein_r2_train,
            "mean_per_protein_r2_test": mean_per_protein_r2_test,
            "median_per_protein_r2_test": median_per_protein_r2_test,
            "n_negative_r2_test": n_negative_r2_test,
        })

        # Console summary for this sample
        print(f"\n  Results:")
        print(f"    Pooled train R2={r2_train:.4f}, MAE={mae_train:.2f}, RMSE={rmse_train:.2f}")
        print(f"    Pooled test  R2={r2_test:.4f}, MAE={mae_test:.2f}, RMSE={rmse_test:.2f}")
        print(f"    Per-protein test R2: mean={mean_per_protein_r2_test:.4f}, "
              f"median={median_per_protein_r2_test:.4f}")
        print(f"    Proteins with negative test R2: {n_negative_r2_test}/{len(test_r2s)}")

        # Clean up memory
        del chunk_map, X_train_raw, X_test_raw, X_train, X_test
        del y_pred_train, y_pred_test

    # ------------------------------------------------------------------
    # Phase 4: Cross-sample stability
    # ------------------------------------------------------------------
    print(f"\n{'=' * 60}")
    print("CROSS-SAMPLE STABILITY")
    print(f"{'=' * 60}")
    pooled_test_r2s = [r["r2_test"] for r in pooled_rows]
    mean_pp_test_r2s = [r["mean_per_protein_r2_test"] for r in pooled_rows]
    print(f"  Pooled test R2 across samples: {[f'{v:.4f}' for v in pooled_test_r2s]}")
    print(f"    mean={np.mean(pooled_test_r2s):.4f}, std={np.std(pooled_test_r2s):.4f}")
    print(f"  Mean per-protein test R2: {[f'{v:.4f}' for v in mean_pp_test_r2s]}")
    print(f"    mean={np.mean(mean_pp_test_r2s):.4f}, std={np.std(mean_pp_test_r2s):.4f}")

    # ------------------------------------------------------------------
    # Phase 5: Save CSV outputs
    # ------------------------------------------------------------------

    # CSV 1: pooled metrics
    pooled_csv = csv_dir / "diagnostic_pooled_metrics.csv"
    pooled_fields = [
        "sample_idx", "seed",
        "n_train_proteins", "n_test_proteins", "n_train_pairs", "n_test_pairs",
        "r2_train", "r2_test", "mae_train", "mae_test", "rmse_train", "rmse_test",
        "mean_per_protein_r2_train", "mean_per_protein_r2_test",
        "median_per_protein_r2_test", "n_negative_r2_test",
    ]
    with open(pooled_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pooled_fields)
        writer.writeheader()
        for row in pooled_rows:
            row_fmt = {
                key: (f"{value:.6f}" if isinstance(value, float) else value)
                for key, value in row.items()
            }
            writer.writerow(row_fmt)
    print(f"\nSaved pooled metrics CSV -> {pooled_csv}")

    # CSV 2: per-protein metrics
    pp_csv = csv_dir / "diagnostic_per_protein_metrics.csv"
    pp_fields = [
        "sample_idx", "seed", "protein_id", "split",
        "n_residues", "n_pairs", "per_protein_r2", "per_protein_mae", "per_protein_rmse",
        "mean_true_distance", "std_true_distance", "max_true_distance",
        "mean_feature_variance", "mean_residual", "std_residual",
    ]
    with open(pp_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=pp_fields)
        writer.writeheader()
        for row in per_protein_rows:
            row_fmt = {
                key: (f"{value:.6f}" if isinstance(value, float) else value)
                for key, value in row.items()
            }
            writer.writerow(row_fmt)
    print(f"Saved per-protein metrics CSV -> {pp_csv}")

    # ------------------------------------------------------------------
    # Phase 6: Outlier consistency analysis
    # ------------------------------------------------------------------
    protein_r2_across_samples = defaultdict(list)
    for row in per_protein_rows:
        if row["split"] == "test":
            protein_r2_across_samples[row["protein_id"]].append(row["per_protein_r2"])

    consistent_failures = []
    for pid, r2_list in sorted(protein_r2_across_samples.items()):
        if len(r2_list) >= 2 and all(r2 < 0 for r2 in r2_list):
            consistent_failures.append((pid, r2_list))

    if consistent_failures:
        print(f"\nConsistent failures (negative R2 in 2+ samples):")
        for pid, r2_list in consistent_failures:
            print(f"  {pid}: R2 = {[f'{v:.4f}' for v in r2_list]}")
    else:
        print(f"\nNo consistent failures found across samples.")

    # ------------------------------------------------------------------
    # Summary text file
    # ------------------------------------------------------------------
    summary_path = csv_dir / "diagnostic_summary.txt"
    with open(summary_path, "w") as f:
        f.write("Diagnostic Regression — Layer 47, 3 Random Samples\n")
        f.write("=" * 55 + "\n\n")
        f.write(f"Layer:              {LAYER}\n")
        f.write(f"Max residues:       {MAX_RESIDUES}\n")
        f.write(f"Sample size:        {SAMPLE_SIZE}\n")
        f.write(f"Sample seeds:       {SAMPLE_SEEDS}\n")
        f.write(f"Split:              80/20 train/test (protein-level)\n")
        f.write(f"Feature dim:        {FEATURE_DIM}\n\n")

        f.write(f"Proteins discovered:  {len(all_proteins)}\n")
        f.write(f"Proteins validated:   {before_filter}\n")
        f.write(f"Excluded (>{MAX_RESIDUES} res): {n_excluded}\n")
        f.write(f"Proteins remaining:   {after_filter}\n\n")

        if excluded_proteins:
            f.write("Excluded proteins:\n")
            for d in sorted(excluded_proteins, key=lambda x: x["n_residues"], reverse=True):
                f.write(f"  {d['protein_id']}: {d['n_residues']} residues\n")
            f.write("\n")

        f.write("Per-sample pooled metrics:\n")
        f.write(f"  {'Sample':<8} {'Seed':<6} {'R2_train':<10} {'R2_test':<10} "
                f"{'MAE_test':<10} {'RMSE_test':<10} {'MeanPP_R2':<10} "
                f"{'MedPP_R2':<10} {'Neg_R2':<8}\n")
        for row in pooled_rows:
            f.write(
                f"  {row['sample_idx']:<8} {row['seed']:<6} "
                f"{row['r2_train']:<10.4f} {row['r2_test']:<10.4f} "
                f"{row['mae_test']:<10.2f} {row['rmse_test']:<10.2f} "
                f"{row['mean_per_protein_r2_test']:<10.4f} "
                f"{row['median_per_protein_r2_test']:<10.4f} "
                f"{row['n_negative_r2_test']:<8}\n"
            )

        f.write(f"\nCross-sample stability:\n")
        f.write(f"  Pooled test R2:        mean={np.mean(pooled_test_r2s):.4f}, "
                f"std={np.std(pooled_test_r2s):.4f}\n")
        f.write(f"  Mean per-protein R2:   mean={np.mean(mean_pp_test_r2s):.4f}, "
                f"std={np.std(mean_pp_test_r2s):.4f}\n")

        # Top 10 best and worst test proteins across all samples
        all_test_rows = [
            r for r in per_protein_rows
            if r["split"] == "test" and not math.isnan(r["per_protein_r2"])
        ]
        all_test_rows_sorted = sorted(all_test_rows, key=lambda x: x["per_protein_r2"])

        f.write(f"\nTop 10 worst per-protein test R2 (across all samples):\n")
        for r in all_test_rows_sorted[:10]:
            f.write(
                f"  {r['protein_id']} (sample {r['sample_idx']}): "
                f"R2={r['per_protein_r2']:.4f}, "
                f"n_res={r['n_residues']}, "
                f"max_dist={r['max_true_distance']:.1f}A, "
                f"mean_feat_var={r['mean_feature_variance']:.4f}\n"
            )

        f.write(f"\nTop 10 best per-protein test R2 (across all samples):\n")
        for r in all_test_rows_sorted[-10:]:
            f.write(
                f"  {r['protein_id']} (sample {r['sample_idx']}): "
                f"R2={r['per_protein_r2']:.4f}, "
                f"n_res={r['n_residues']}, "
                f"max_dist={r['max_true_distance']:.1f}A, "
                f"mean_feat_var={r['mean_feature_variance']:.4f}\n"
            )

        if consistent_failures:
            f.write(f"\nConsistent failures (negative R2 in 2+ samples):\n")
            for pid, r2_list in consistent_failures:
                f.write(f"  {pid}: R2 = {[f'{v:.4f}' for v in r2_list]}\n")
        else:
            f.write(f"\nNo consistent failures across samples.\n")

        if skipped:
            f.write(f"\nSkipped proteins during validation ({len(skipped)}):\n")
            for protein_id, reason in skipped:
                f.write(f"  {protein_id}: {reason}\n")

    print(f"Saved summary -> {summary_path}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 60}")
    print("DIAGNOSTIC RUN COMPLETE")
    print(f"Elapsed time: {elapsed:.1f} seconds ({elapsed / 60.0:.1f} minutes)")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()

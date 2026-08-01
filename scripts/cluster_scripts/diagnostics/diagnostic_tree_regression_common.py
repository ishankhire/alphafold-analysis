#!/usr/bin/env python3
"""
Shared helpers for fast layer-47 tree-based distance diagnostics on Delta.
"""

import csv
import hashlib
import json
import math
import os
import sys
import time
import traceback
from collections import Counter
from pathlib import Path

import numpy as np
from Bio.PDB import PDBParser
from sklearn.model_selection import train_test_split


N_CHANNELS = 128
FEATURE_DIM = N_CHANNELS + 6
LAYER = 47
TEST_SIZE = 0.2
VAL_SIZE_WITHIN_TRAIN = 0.2
RANDOM_STATE = 42
VAL_RANDOM_STATE = 43
MAX_TRAIN_PAIRS_PER_PROTEIN = 5000
FLOAT_EPS = 1e-12

PAIR_BLOCKS_ROOT = Path(os.environ["PAIR_BLOCKS_ROOT"])
PDB_ROOT = Path(os.environ["PDB_ROOT"])
CSV_DIR = Path(os.environ["CSV_ROOT"])

FEATURE_NAMES = [f"pair_ch_{idx}" for idx in range(N_CHANNELS)] + [
    "n_residues",
    "seq_sep",
    "log1p_seq_sep",
    "is_seq_sep_1",
    "is_seq_sep_le_2",
    "is_seq_sep_le_4",
]

SEQ_BUCKET_SPECS = [
    ("1", 1, 1),
    ("2", 2, 2),
    ("3-4", 3, 4),
    ("5-8", 5, 8),
    ("9+", 9, None),
]


def format_value(value):
    """Stable CSV formatting."""
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    if isinstance(value, bool):
        return "true" if value else "false"
    return value


def write_csv(path, fieldnames, rows):
    """Write a CSV with stable formatting."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


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
    """Return the path to a pair block .npy file."""
    return protein_dir / f"{protein_id}_pair_block_{layer}.npy"


def discover_proteins(pair_blocks_root):
    """Find all protein directories under the pair blocks root."""
    proteins = []
    for entry in sorted(pair_blocks_root.iterdir()):
        if entry.is_dir():
            proteins.append((entry.name, entry))
    return proteins


def extract_ca_coordinates(pdb_path):
    """Extract CA atoms from all chains in the PDB."""
    parser = PDBParser(QUIET=True)
    structure = parser.get_structure("predicted", str(pdb_path))

    residues = []
    coordinates = []
    for chain in structure[0]:
        for residue in chain:
            if residue.get_id()[0] == "W":
                continue
            if "CA" not in residue:
                continue
            residues.append((chain.get_id(), residue.get_resname(), residue.get_id()[1]))
            coordinates.append(residue["CA"].get_coord())

    return {
        "residues": residues,
        "coordinates": (
            np.array(coordinates, dtype=np.float32)
            if coordinates
            else np.empty((0, 3), dtype=np.float32)
        ),
    }


def compute_distance_matrix(coords):
    """Vectorized pairwise Euclidean distance matrix."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    return np.linalg.norm(diff, axis=2)


def compute_r2(y_true, y_pred):
    """Compute unweighted R-squared."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    mean_true = float(np.mean(y_true))
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - mean_true) ** 2))
    if ss_tot <= FLOAT_EPS:
        return float("nan")
    return float(1.0 - (ss_res / ss_tot))


def compute_mae(y_true, y_pred):
    """Compute MAE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(np.mean(np.abs(y_true - y_pred)))


def compute_rmse(y_true, y_pred):
    """Compute RMSE."""
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    if y_true.size == 0:
        return float("nan")
    return float(math.sqrt(np.mean((y_true - y_pred) ** 2)))


def deterministic_seed(label, offset=0):
    """Derive a stable integer seed from a label."""
    digest = hashlib.sha256(label.encode("utf-8")).digest()
    return int.from_bytes(digest[:8], byteorder="little", signed=False) + int(offset)


def deterministic_sample_indices(label, n_rows, cap, offset=0):
    """Sample row indices reproducibly."""
    if cap is None or n_rows <= cap:
        return np.arange(n_rows, dtype=np.int64)
    rng = np.random.default_rng(deterministic_seed(label, offset=offset))
    out = rng.choice(n_rows, size=cap, replace=False)
    out.sort()
    return out.astype(np.int64, copy=False)


def load_protein_records():
    """Discover valid proteins and cache lower-triangle targets + scalar metadata."""
    all_proteins = discover_proteins(PAIR_BLOCKS_ROOT)
    protein_data = []
    skipped = []

    for protein_id, protein_dir in all_proteins:
        try:
            pdb_path = find_pdb_file(protein_id)
            if pdb_path is None:
                skipped.append((protein_id, "no predicted PDB file found"))
                continue

            pb0_path = get_pair_block_path(protein_dir, protein_id, 0)
            if not pb0_path.exists():
                skipped.append((protein_id, "pair block 0 not found"))
                continue

            ca_data = extract_ca_coordinates(pdb_path)
            n_ca = len(ca_data["residues"])
            if n_ca < 2:
                skipped.append((protein_id, "fewer than 2 CA atoms"))
                continue

            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = int(pb0.shape[0])
            del pb0
            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                continue

            lt_i, lt_j = np.tril_indices(n_ca, k=-1)
            if lt_i.size == 0:
                skipped.append((protein_id, "no lower-triangle residue pairs"))
                continue

            seq_sep = (lt_i - lt_j).astype(np.float32, copy=False)
            dist_matrix = compute_distance_matrix(ca_data["coordinates"])
            y_abs = np.asarray(dist_matrix[lt_i, lt_j], dtype=np.float32)

            protein_data.append(
                {
                    "protein_id": protein_id,
                    "protein_dir": protein_dir,
                    "n_residues": n_ca,
                    "lt_i": lt_i,
                    "lt_j": lt_j,
                    "seq_sep": seq_sep,
                    "y_abs": y_abs,
                    "n_pairs": int(y_abs.size),
                }
            )
        except Exception as exc:  # pragma: no cover - diagnostic path
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            traceback.print_exc()

    return all_proteins, protein_data, skipped


def split_records(records):
    """Create protein-level test split, then fit/validation split inside train."""
    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )
    train_set = set(int(idx) for idx in train_idx)
    test_set = set(int(idx) for idx in test_idx)
    train_records = [record for idx, record in enumerate(records) if idx in train_set]
    test_records = [record for idx, record in enumerate(records) if idx in test_set]

    fit_indices = np.arange(len(train_records))
    fit_idx, val_idx = train_test_split(
        fit_indices,
        test_size=VAL_SIZE_WITHIN_TRAIN,
        random_state=VAL_RANDOM_STATE,
    )
    fit_set = set(int(idx) for idx in fit_idx)
    val_set = set(int(idx) for idx in val_idx)
    fit_records = [record for idx, record in enumerate(train_records) if idx in fit_set]
    val_records = [record for idx, record in enumerate(train_records) if idx in val_set]
    return fit_records, val_records, train_records, test_records


def build_split_rows(fit_records, val_records, train_records, test_records):
    """Build protein-level split membership rows."""
    rows = []
    train_ids = {record["protein_id"] for record in train_records}
    fit_ids = {record["protein_id"] for record in fit_records}
    val_ids = {record["protein_id"] for record in val_records}

    for record in train_records:
        rows.append(
            {
                "protein_id": record["protein_id"],
                "split_group": "train",
                "split_role": "fit" if record["protein_id"] in fit_ids else "validation",
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
            }
        )
    for record in test_records:
        if record["protein_id"] in train_ids:
            continue
        rows.append(
            {
                "protein_id": record["protein_id"],
                "split_group": "test",
                "split_role": "test",
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
            }
        )
    return rows


def build_matrix(records, sample_cap=None, sample_offset=0):
    """Materialize one split into a dense matrix."""
    row_counts = []
    selections = []
    for record in records:
        local_idx = deterministic_sample_indices(
            label=f"{record['protein_id']}:{record['n_pairs']}",
            n_rows=record["n_pairs"],
            cap=sample_cap,
            offset=sample_offset,
        )
        selections.append(local_idx)
        row_counts.append(int(local_idx.size))

    total_rows = int(sum(row_counts))
    x = np.empty((total_rows, FEATURE_DIM), dtype=np.float32)
    y = np.empty((total_rows,), dtype=np.float64)
    seq_sep_rows = np.empty((total_rows,), dtype=np.int32)
    offsets = []

    start = 0
    for record, local_idx, n_rows in zip(records, selections, row_counts):
        end = start + n_rows
        pb_path = get_pair_block_path(record["protein_dir"], record["protein_id"], LAYER)
        pb = np.load(str(pb_path), mmap_mode="r")

        lt_i = record["lt_i"][local_idx]
        lt_j = record["lt_j"][local_idx]
        seq_sep = record["seq_sep"][local_idx]

        x[start:end, :N_CHANNELS] = pb[lt_i, lt_j, :]
        x[start:end, N_CHANNELS] = float(record["n_residues"])
        x[start:end, N_CHANNELS + 1] = seq_sep
        x[start:end, N_CHANNELS + 2] = np.log1p(seq_sep)
        x[start:end, N_CHANNELS + 3] = (seq_sep == 1).astype(np.float32, copy=False)
        x[start:end, N_CHANNELS + 4] = (seq_sep <= 2).astype(np.float32, copy=False)
        x[start:end, N_CHANNELS + 5] = (seq_sep <= 4).astype(np.float32, copy=False)

        y[start:end] = record["y_abs"][local_idx]
        seq_sep_rows[start:end] = seq_sep.astype(np.int32, copy=False)
        offsets.append((start, end))

        del pb
        start = end

    weights = compute_pair_weights(row_counts)
    return {
        "x": x,
        "y": y,
        "seq_sep": seq_sep_rows,
        "offsets": offsets,
        "row_counts": row_counts,
        "weights": weights,
    }


def compute_pair_weights(row_counts):
    """Each protein contributes equal total loss mass."""
    if not row_counts:
        return np.empty((0,), dtype=np.float64)
    pieces = []
    for n_rows in row_counts:
        pieces.append(np.full(n_rows, 1.0 / max(n_rows, 1), dtype=np.float64))
    weights = np.concatenate(pieces)
    if weights.size > 0:
        weights *= weights.size / np.sum(weights)
    return weights


def summarize_predictions(split_name, records, offsets, y_true, y_pred):
    """Summarize pooled and per-protein metrics for one split."""
    per_rows = []
    per_r2 = []
    negative_r2 = 0

    for record, (start, end) in zip(records, offsets):
        y_true_p = y_true[start:end]
        y_pred_p = y_pred[start:end]
        r2_value = compute_r2(y_true_p, y_pred_p)
        if not math.isnan(r2_value):
            per_r2.append(r2_value)
            if r2_value < 0.0:
                negative_r2 += 1

        per_rows.append(
            {
                "split": split_name,
                "protein_id": record["protein_id"],
                "n_residues": record["n_residues"],
                "n_pairs_used": int(end - start),
                "per_protein_r2": r2_value,
                "per_protein_mae": compute_mae(y_true_p, y_pred_p),
                "per_protein_rmse": compute_rmse(y_true_p, y_pred_p),
                "mean_true_distance": float(np.mean(y_true_p)),
                "mean_predicted_distance": float(np.mean(y_pred_p)),
                "mean_error_pred_minus_true": float(np.mean(y_pred_p - y_true_p)),
            }
        )

    return {
        "mean_per_protein_r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
        "median_per_protein_r2": float(np.median(per_r2)) if per_r2 else float("nan"),
        "negative_r2_count": int(negative_r2),
        "negative_r2_fraction": (
            float(negative_r2 / len(per_r2)) if per_r2 else float("nan")
        ),
        "pair_weighted_r2": compute_r2(y_true, y_pred),
        "pair_weighted_mae": compute_mae(y_true, y_pred),
        "pair_weighted_rmse": compute_rmse(y_true, y_pred),
        "per_rows": per_rows,
    }


def build_seq_sep_diagnostics(split_name, seq_sep_rows, y_true, y_pred):
    """Aggregate diagnostics by sequence-separation bucket."""
    rows = []
    seq_sep_rows = np.asarray(seq_sep_rows, dtype=np.int32)
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)

    for label, min_sep, max_sep in SEQ_BUCKET_SPECS:
        if max_sep is None:
            mask = seq_sep_rows >= min_sep
        else:
            mask = (seq_sep_rows >= min_sep) & (seq_sep_rows <= max_sep)

        count = int(np.sum(mask))
        if count == 0:
            rows.append(
                {
                    "split": split_name,
                    "seq_bucket": label,
                    "n_pairs": 0,
                    "mean_true_distance": float("nan"),
                    "mean_predicted_distance": float("nan"),
                    "mean_error_pred_minus_true": float("nan"),
                    "r2": float("nan"),
                    "fraction_pred_lt_3A": float("nan"),
                }
            )
            continue

        y_true_bucket = y_true[mask]
        y_pred_bucket = y_pred[mask]
        rows.append(
            {
                "split": split_name,
                "seq_bucket": label,
                "n_pairs": count,
                "mean_true_distance": float(np.mean(y_true_bucket)),
                "mean_predicted_distance": float(np.mean(y_pred_bucket)),
                "mean_error_pred_minus_true": float(np.mean(y_pred_bucket - y_true_bucket)),
                "r2": compute_r2(y_true_bucket, y_pred_bucket),
                "fraction_pred_lt_3A": float(np.mean(y_pred_bucket < 3.0)),
            }
        )

    sep1_mask = seq_sep_rows == 1
    rows.append(
        {
            "split": split_name,
            "seq_bucket": "seq_sep_eq_1_exact",
            "n_pairs": int(np.sum(sep1_mask)),
            "mean_true_distance": (
                float(np.mean(y_true[sep1_mask])) if np.any(sep1_mask) else float("nan")
            ),
            "mean_predicted_distance": (
                float(np.mean(y_pred[sep1_mask])) if np.any(sep1_mask) else float("nan")
            ),
            "mean_error_pred_minus_true": (
                float(np.mean(y_pred[sep1_mask] - y_true[sep1_mask]))
                if np.any(sep1_mask)
                else float("nan")
            ),
            "r2": (
                compute_r2(y_true[sep1_mask], y_pred[sep1_mask])
                if np.any(sep1_mask)
                else float("nan")
            ),
            "fraction_pred_lt_3A": (
                float(np.mean(y_pred[sep1_mask] < 3.0)) if np.any(sep1_mask) else float("nan")
            ),
        }
    )
    return rows


def build_feature_importance_rows(feature_importances):
    """Format feature importances into sorted rows."""
    rows = []
    for idx, importance in enumerate(np.asarray(feature_importances, dtype=np.float64)):
        rows.append(
            {
                "feature_index": idx,
                "feature_name": FEATURE_NAMES[idx],
                "importance": float(importance),
            }
        )
    rows.sort(key=lambda row: float(row["importance"]), reverse=True)
    return rows


def parse_xgb_json_dump(booster):
    """Parse XGBoost trees from JSON dumps without pandas."""
    dump = booster.get_dump(dump_format="json", with_stats=True)
    node_rows = []
    root_rows = []
    split_counter = Counter()
    split_gain = Counter()

    def walk(tree_index, node, depth):
        node_id = int(node["nodeid"])
        is_leaf = "leaf" in node
        feature_name = node.get("split", "")
        split_condition = node.get("split_condition", float("nan"))
        gain = float(node.get("gain", float("nan")))
        cover = float(node.get("cover", float("nan")))

        row = {
            "tree_index": tree_index,
            "node_id": node_id,
            "depth": depth,
            "feature_name": feature_name,
            "split_condition": split_condition,
            "yes_child": node.get("yes", ""),
            "no_child": node.get("no", ""),
            "missing_child": node.get("missing", ""),
            "gain": gain,
            "cover": cover,
            "leaf_value": node.get("leaf", ""),
            "is_leaf": is_leaf,
        }
        node_rows.append(row)

        if not is_leaf:
            split_counter[feature_name] += 1
            if not math.isnan(gain):
                split_gain[feature_name] += gain
            if depth == 0:
                root_rows.append(
                    {
                        "tree_index": tree_index,
                        "feature_name": feature_name,
                        "split_condition": split_condition,
                        "gain": gain,
                        "cover": cover,
                    }
                )
            for child in node.get("children", []):
                walk(tree_index, child, depth + 1)

    for tree_index, tree_json in enumerate(dump):
        walk(tree_index, json.loads(tree_json), depth=0)

    top_split_rows = []
    for feature_name, count in split_counter.most_common():
        top_split_rows.append(
            {
                "feature_name": feature_name,
                "split_count": int(count),
                "total_gain": float(split_gain.get(feature_name, 0.0)),
            }
        )
    return dump, node_rows, root_rows, top_split_rows


def map_xgb_importance_names(importance_map):
    """Map xgboost feature keys like f3 to readable names."""
    rows = []
    for key, value in importance_map.items():
        if key.startswith("f"):
            index = int(key[1:])
            feature_name = FEATURE_NAMES[index] if index < len(FEATURE_NAMES) else key
        else:
            index = -1
            feature_name = key
        rows.append(
            {
                "feature_index": index,
                "feature_name": feature_name,
                "importance": float(value),
            }
        )
    rows.sort(key=lambda row: float(row["importance"]), reverse=True)
    return rows


def run_search(configs, fit_bundle, val_bundle, fit_records, val_records, train_model_fn):
    """Evaluate a list of configs on the fit/validation split."""
    search_rows = []
    best = None

    for config_index, config in enumerate(configs):
        t0 = time.time()
        model, aux = train_model_fn(config, fit_bundle, val_bundle)
        val_pred = np.asarray(model.predict(val_bundle["x"]), dtype=np.float64)
        fit_pred = np.asarray(model.predict(fit_bundle["x"]), dtype=np.float64)
        val_metrics = summarize_predictions(
            "validation_search",
            val_records,
            val_bundle["offsets"],
            val_bundle["y"],
            val_pred,
        )
        fit_metrics = summarize_predictions(
            "fit_search",
            fit_records,
            fit_bundle["offsets"],
            fit_bundle["y"],
            fit_pred,
        )

        row = {
            "config_index": config_index,
            "elapsed_seconds": float(time.time() - t0),
            "fit_mean_per_protein_r2": fit_metrics["mean_per_protein_r2"],
            "validation_mean_per_protein_r2": val_metrics["mean_per_protein_r2"],
            "validation_median_per_protein_r2": val_metrics["median_per_protein_r2"],
            "validation_negative_r2_fraction": val_metrics["negative_r2_fraction"],
        }
        for key, value in config.items():
            row[key] = value
        if aux:
            row.update(aux)
        search_rows.append(row)

        candidate_score = float(val_metrics["mean_per_protein_r2"])
        if best is None or candidate_score > best["score"]:
            best = {
                "score": candidate_score,
                "config": dict(config),
                "row": dict(row),
            }

    return best, search_rows


def write_summary_text(
    path,
    title,
    elapsed_seconds,
    all_proteins,
    protein_data,
    skipped,
    fit_records,
    val_records,
    train_records,
    test_records,
    best_row,
    train_metrics,
    test_metrics,
):
    """Write a human-readable summary."""
    with path.open("w") as handle:
        handle.write(f"{title}\n")
        handle.write("=" * len(title) + "\n\n")
        handle.write(f"Pair blocks root: {PAIR_BLOCKS_ROOT}\n")
        handle.write(f"PDB root:         {PDB_ROOT}\n")
        handle.write(f"Output CSV dir:   {CSV_DIR.resolve()}\n")
        handle.write(f"Layer:            {LAYER}\n")
        handle.write("Representation:   lower triangle only, pb[i,j,:] only\n")
        handle.write(
            f"Train cap:        {MAX_TRAIN_PAIRS_PER_PROTEIN:,} pairs per fit/train protein\n"
        )
        handle.write(
            f"Split seeds:      test={RANDOM_STATE}, validation={VAL_RANDOM_STATE}\n\n"
        )

        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(protein_data)}\n")
        handle.write(f"Proteins skipped:    {len(skipped)}\n")
        handle.write(f"Fit proteins:        {len(fit_records)}\n")
        handle.write(f"Validation proteins: {len(val_records)}\n")
        handle.write(f"Train proteins:      {len(train_records)}\n")
        handle.write(f"Test proteins:       {len(test_records)}\n")
        handle.write(f"Elapsed seconds:     {elapsed_seconds:.1f}\n\n")

        handle.write("Best configuration:\n")
        for key in sorted(best_row.keys()):
            handle.write(f"  {key}: {format_value(best_row[key])}\n")
        handle.write("\n")

        handle.write("Final metrics:\n")
        handle.write(
            f"  Train mean per-protein R2: {train_metrics['mean_per_protein_r2']:.4f}\n"
        )
        handle.write(
            f"  Test mean per-protein R2:  {test_metrics['mean_per_protein_r2']:.4f}\n"
        )
        handle.write(
            f"  Test median per-protein R2: {test_metrics['median_per_protein_r2']:.4f}\n"
        )
        handle.write(
            f"  Test negative R2 fraction: {test_metrics['negative_r2_fraction']:.4f}\n"
        )
        handle.write("\n")

        ranked = sorted(
            test_metrics["per_rows"],
            key=lambda row: (
                math.inf
                if math.isnan(float(row["per_protein_r2"]))
                else float(row["per_protein_r2"])
            ),
        )
        handle.write("Worst 10 test proteins:\n")
        for row in ranked[:10]:
            handle.write(
                f"  {row['protein_id']}: R2={float(row['per_protein_r2']):.4f}, "
                f"mean(pred-true)={float(row['mean_error_pred_minus_true']):+.4f}\n"
            )
        handle.write("\nBest 10 test proteins:\n")
        for row in reversed(ranked[-10:]):
            handle.write(
                f"  {row['protein_id']}: R2={float(row['per_protein_r2']):.4f}, "
                f"mean(pred-true)={float(row['mean_error_pred_minus_true']):+.4f}\n"
            )

        if skipped:
            handle.write("\nSkipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")


def fail_missing_dependency(package_name, exc):
    """Emit a consistent missing-dependency error."""
    print(f"ERROR: Required package '{package_name}' is not available.")
    print(f"Import error: {exc}")
    sys.exit(1)

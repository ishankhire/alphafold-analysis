#!/usr/bin/env python3
"""
diagnostic_float_precision_128d_regression.py
=============================================
Minimal layer-47 precision check for 128d lower-triangle linear regression.

Runs the same standardized, protein-equal weighted 128d probe twice:
- once with float32 features/targets
- once with float64 features/targets

Outputs:
    Prints test mean per-protein R^2 to stdout
    Writes diagnostic_float_precision_128d_regression.csv in CSV_ROOT
"""

import gc
import math
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(SCRIPT_DIR))
if str(CLUSTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_SCRIPTS_DIR))

from diagnostic_tree_regression_common import (  # noqa: E402
    CSV_DIR,
    FLOAT_EPS,
    LAYER,
    N_CHANNELS,
    RANDOM_STATE,
    TEST_SIZE,
    load_protein_records,
    write_csv,
)


TITLE = "Diagnostic Float Precision 128d Regression - Layer 47"
N_SIZE_BINS = 5
OUTPUT_CSV_NAME = "diagnostic_float_precision_128d_regression.csv"
OUTPUT_FIELDNAMES = [
    "dtype",
    "n_train_proteins",
    "n_test_proteins",
    "n_train_rows",
    "n_test_rows",
    "test_mean_per_protein_r2",
    "elapsed_seconds",
]


def assign_size_bins(records, n_bins):
    ordered = sorted(records, key=lambda row: (row["n_residues"], row["protein_id"]))
    total = len(ordered)
    for rank, record in enumerate(ordered):
        size_bin = int((rank * n_bins) // max(total, 1))
        if size_bin >= n_bins:
            size_bin = n_bins - 1
        record["size_bin"] = int(size_bin)


def choose_stratify_labels(records):
    if len(records) < 2:
        return None

    labels = np.asarray([record["size_bin"] for record in records], dtype=np.int32)
    unique, counts = np.unique(labels, return_counts=True)
    if unique.size < 2:
        return None
    if np.min(counts) < 2:
        return None

    n_total = len(records)
    n_test = max(1, int(math.ceil(TEST_SIZE * n_total)))
    n_train = n_total - n_test
    if n_test < unique.size or n_train < unique.size:
        return None
    return labels


def split_records(records):
    indices = np.arange(len(records), dtype=np.int64)
    stratify_labels = choose_stratify_labels(records)
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=stratify_labels,
    )
    train_set = set(int(idx) for idx in train_idx)
    test_set = set(int(idx) for idx in test_idx)
    train_records = [record for idx, record in enumerate(records) if idx in train_set]
    test_records = [record for idx, record in enumerate(records) if idx in test_set]
    return train_records, test_records


def prepare_records(records):
    prepared = []
    for record in records:
        target_variance = float(np.var(record["y_abs"], dtype=np.float64))
        if target_variance <= FLOAT_EPS:
            target_variance = FLOAT_EPS
        prepared_record = dict(record)
        prepared_record["target_variance"] = target_variance
        prepared.append(prepared_record)
    return prepared


def build_center_chunk_map(records):
    chunk_map = {}
    for record in records:
        pb_path = record["protein_dir"] / f"{record['protein_id']}_pair_block_{LAYER}.npy"
        pb = np.load(str(pb_path), mmap_mode="r")
        chunk = np.asarray(pb[record["lt_i"], record["lt_j"], :], dtype=np.float32)
        del pb
        chunk_map[record["protein_id"]] = chunk
        record["feature_mean"] = np.asarray(np.mean(chunk, axis=0), dtype=np.float64)
        record["feature_sq_mean"] = np.asarray(np.mean(chunk * chunk, axis=0), dtype=np.float64)
    return chunk_map


def compute_scaler(train_records):
    mean_acc = np.zeros((N_CHANNELS,), dtype=np.float64)
    sq_acc = np.zeros((N_CHANNELS,), dtype=np.float64)
    for record in train_records:
        mean_acc += record["feature_mean"]
        sq_acc += record["feature_sq_mean"]
    mean = mean_acc / max(len(train_records), 1)
    second_moment = sq_acc / max(len(train_records), 1)
    var = np.maximum(second_moment - mean * mean, 0.0)
    scale = np.sqrt(var)
    scale[scale < FLOAT_EPS] = 1.0
    return mean, scale


def materialize_split(records, chunk_map):
    x = np.vstack([chunk_map[record["protein_id"]] for record in records]).astype(
        np.float32,
        copy=False,
    )
    y = np.concatenate([record["y_abs"] for record in records]).astype(np.float64, copy=False)
    return x, y


def compute_train_weights(train_records):
    pieces = [
        np.full(record["n_pairs"], 1.0 / record["n_pairs"], dtype=np.float64)
        for record in train_records
    ]
    weights = np.concatenate(pieces)
    weights *= weights.size / np.sum(weights)
    return weights


def mean_per_protein_r2(records, y_true, y_pred):
    values = []
    start = 0
    for record in records:
        end = start + int(record["n_pairs"])
        y_true_p = np.asarray(y_true[start:end], dtype=np.float64)
        y_pred_p = np.asarray(y_pred[start:end], dtype=np.float64)
        start = end
        if y_true_p.size == 0:
            continue
        mean_y = float(np.mean(y_true_p))
        ss_res = float(np.sum((y_true_p - y_pred_p) ** 2))
        ss_tot = float(np.sum((y_true_p - mean_y) ** 2))
        if ss_tot <= FLOAT_EPS:
            continue
        values.append(float(1.0 - (ss_res / ss_tot)))
    if not values:
        return float("nan")
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def run_one_precision(dtype_name, np_dtype, train_records, test_records, chunk_map, scaler_mean, scaler_scale):
    t0 = time.time()
    x_train_base, y_train_base = materialize_split(train_records, chunk_map)
    x_test_base, y_test_base = materialize_split(test_records, chunk_map)

    x_train = x_train_base.astype(np_dtype, copy=True)
    x_test = x_test_base.astype(np_dtype, copy=True)
    y_train = y_train_base.astype(np_dtype, copy=True)

    mean_cast = scaler_mean.astype(np_dtype, copy=False)
    scale_cast = scaler_scale.astype(np_dtype, copy=False)
    x_train -= mean_cast
    x_train /= scale_cast
    x_test -= mean_cast
    x_test /= scale_cast

    model = LinearRegression(copy_X=False)
    model.fit(
        x_train,
        y_train,
        sample_weight=compute_train_weights(train_records),
    )
    y_pred_test = np.asarray(model.predict(x_test), dtype=np.float64)
    r2_value = mean_per_protein_r2(test_records, y_test_base, y_pred_test)

    row = {
        "dtype": dtype_name,
        "n_train_proteins": int(len(train_records)),
        "n_test_proteins": int(len(test_records)),
        "n_train_rows": int(x_train.shape[0]),
        "n_test_rows": int(x_test.shape[0]),
        "test_mean_per_protein_r2": float(r2_value),
        "elapsed_seconds": float(time.time() - t0),
    }

    del x_train_base, x_test_base, x_train, x_test, y_train
    gc.collect()
    return row


def main():
    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))

    all_proteins, protein_data, skipped = load_protein_records()
    del all_proteins
    if not protein_data:
        print("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    records = prepare_records(protein_data)
    assign_size_bins(records, N_SIZE_BINS)
    train_records, test_records = split_records(records)
    chunk_map = build_center_chunk_map(records)
    scaler_mean, scaler_scale = compute_scaler(train_records)

    rows = []
    for dtype_name, np_dtype in [("float32", np.float32), ("float64", np.float64)]:
        row = run_one_precision(
            dtype_name=dtype_name,
            np_dtype=np_dtype,
            train_records=train_records,
            test_records=test_records,
            chunk_map=chunk_map,
            scaler_mean=scaler_mean,
            scaler_scale=scaler_scale,
        )
        rows.append(row)
        print(f"{dtype_name}: test mean per-protein R^2 = {row['test_mean_per_protein_r2']:.6f}")

    if skipped:
        print(f"Skipped proteins: {len(skipped)}")

    write_csv(csv_dir / OUTPUT_CSV_NAME, OUTPUT_FIELDNAMES, rows)
    print(f"Saved CSV -> {csv_dir / OUTPUT_CSV_NAME}")


if __name__ == "__main__":
    main()

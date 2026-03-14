#!/usr/bin/env python3
"""
Fast layer-47 DecisionTreeRegressor diagnostic with protein-level splits.
"""

import time

import numpy as np
from sklearn.tree import DecisionTreeRegressor, export_text

from diagnostic_tree_regression_common import (
    CSV_DIR,
    FEATURE_NAMES,
    MAX_TRAIN_PAIRS_PER_PROTEIN,
    build_feature_importance_rows,
    build_matrix,
    build_seq_sep_diagnostics,
    build_split_rows,
    fail_missing_dependency,
    format_value,
    load_protein_records,
    run_search,
    split_records,
    summarize_predictions,
    write_csv,
    write_summary_text,
)

try:
    import sklearn  # noqa: F401
except Exception as exc:  # pragma: no cover - environment specific
    fail_missing_dependency("scikit-learn", exc)


MODEL_NAME = "decision_tree"
TITLE = "Diagnostic Decision Tree Regression - Layer 47"
SUMMARY_CSV_NAME = "diagnostic_decision_tree_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_decision_tree_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_decision_tree_split.csv"
SEQ_DIAGNOSTICS_CSV_NAME = "diagnostic_decision_tree_seq_sep_diagnostics.csv"
SEARCH_CSV_NAME = "diagnostic_decision_tree_search.csv"
TREE_NODES_CSV_NAME = "diagnostic_decision_tree_nodes.csv"
FEATURE_IMPORTANCE_CSV_NAME = "diagnostic_decision_tree_feature_importance.csv"
TREE_TEXT_NAME = "diagnostic_decision_tree_tree.txt"
SUMMARY_TXT_NAME = "diagnostic_decision_tree_summary.txt"

CONFIGS = [
    {"max_depth": 4, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 4, "min_samples_leaf": 25, "min_samples_split": 50},
    {"max_depth": 5, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 5, "min_samples_leaf": 25, "min_samples_split": 50},
    {"max_depth": 6, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 6, "min_samples_leaf": 25, "min_samples_split": 50},
    {"max_depth": 7, "min_samples_leaf": 10, "min_samples_split": 20},
    {"max_depth": 7, "min_samples_leaf": 50, "min_samples_split": 100},
    {"max_depth": 8, "min_samples_leaf": 25, "min_samples_split": 50},
    {"max_depth": 8, "min_samples_leaf": 50, "min_samples_split": 100},
    {"max_depth": 10, "min_samples_leaf": 25, "min_samples_split": 50},
    {"max_depth": 10, "min_samples_leaf": 50, "min_samples_split": 100},
]


def train_model(config, fit_bundle, _val_bundle):
    """Fit one decision tree config."""
    model = DecisionTreeRegressor(
        random_state=42,
        max_depth=int(config["max_depth"]),
        min_samples_leaf=int(config["min_samples_leaf"]),
        min_samples_split=int(config["min_samples_split"]),
    )
    model.fit(fit_bundle["x"], fit_bundle["y"], sample_weight=fit_bundle["weights"])
    return model, {"trained_tree_depth": int(model.get_depth()), "trained_n_leaves": int(model.get_n_leaves())}


def build_node_rows(model):
    """Extract node-level metadata from sklearn's tree structure."""
    tree = model.tree_
    left = tree.children_left
    right = tree.children_right
    features = tree.feature
    thresholds = tree.threshold
    impurity = tree.impurity
    n_node_samples = tree.n_node_samples
    weighted = tree.weighted_n_node_samples
    values = tree.value.reshape(tree.node_count, -1)

    depths = np.zeros(tree.node_count, dtype=np.int32)
    stack = [(0, 0)]
    while stack:
        node_id, depth = stack.pop()
        depths[node_id] = depth
        if left[node_id] != right[node_id]:
            stack.append((int(left[node_id]), depth + 1))
            stack.append((int(right[node_id]), depth + 1))

    rows = []
    for node_id in range(tree.node_count):
        feature_index = int(features[node_id])
        is_leaf = left[node_id] == right[node_id]
        rows.append(
            {
                "node_id": node_id,
                "depth": int(depths[node_id]),
                "left_child": int(left[node_id]),
                "right_child": int(right[node_id]),
                "feature_index": feature_index,
                "feature_name": "" if is_leaf or feature_index < 0 else FEATURE_NAMES[feature_index],
                "threshold": float(thresholds[node_id]),
                "impurity": float(impurity[node_id]),
                "n_node_samples": int(n_node_samples[node_id]),
                "weighted_n_node_samples": float(weighted[node_id]),
                "value": float(values[node_id][0]),
                "is_leaf": bool(is_leaf),
            }
        )
    return rows


def main():
    t_start = time.time()
    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)

    print(TITLE)
    print("=" * len(TITLE))
    print(f"Output CSV dir: {csv_dir}")
    print(f"Train cap:      {MAX_TRAIN_PAIRS_PER_PROTEIN:,} pairs per train protein")
    print(f"Configs:        {len(CONFIGS)}")

    all_proteins, protein_data, skipped = load_protein_records()
    if not protein_data:
        print("ERROR: No proteins loaded successfully.")
        raise SystemExit(1)

    fit_records, val_records, train_records, test_records = split_records(protein_data)
    split_rows = build_split_rows(fit_records, val_records, train_records, test_records)
    write_csv(
        csv_dir / SPLIT_CSV_NAME,
        ["protein_id", "split_group", "split_role", "n_residues", "n_pairs"],
        split_rows,
    )

    fit_bundle = build_matrix(fit_records, sample_cap=MAX_TRAIN_PAIRS_PER_PROTEIN, sample_offset=0)
    val_bundle = build_matrix(val_records, sample_cap=None, sample_offset=0)

    best, search_rows = run_search(CONFIGS, fit_bundle, val_bundle, fit_records, val_records, train_model)
    write_csv(csv_dir / SEARCH_CSV_NAME, list(search_rows[0].keys()), search_rows)

    train_bundle = build_matrix(train_records, sample_cap=MAX_TRAIN_PAIRS_PER_PROTEIN, sample_offset=1000)
    test_bundle = build_matrix(test_records, sample_cap=None, sample_offset=0)

    final_model, final_aux = train_model(best["config"], train_bundle, None)
    train_pred = np.asarray(final_model.predict(train_bundle["x"]), dtype=np.float64)
    test_pred = np.asarray(final_model.predict(test_bundle["x"]), dtype=np.float64)

    train_metrics = summarize_predictions(
        "train_sampled_final",
        train_records,
        train_bundle["offsets"],
        train_bundle["y"],
        train_pred,
    )
    test_metrics = summarize_predictions(
        "test_full",
        test_records,
        test_bundle["offsets"],
        test_bundle["y"],
        test_pred,
    )

    summary_row = {
        "model_name": MODEL_NAME,
        "feature_dim": len(FEATURE_NAMES),
        "n_loaded_proteins": len(protein_data),
        "n_fit_proteins": len(fit_records),
        "n_validation_proteins": len(val_records),
        "n_train_proteins": len(train_records),
        "n_test_proteins": len(test_records),
        "n_train_rows_used": int(train_bundle["x"].shape[0]),
        "n_test_rows_used": int(test_bundle["x"].shape[0]),
        "train_cap_per_protein": MAX_TRAIN_PAIRS_PER_PROTEIN,
        "best_validation_mean_per_protein_r2": best["score"],
        "train_mean_per_protein_r2": train_metrics["mean_per_protein_r2"],
        "test_mean_per_protein_r2": test_metrics["mean_per_protein_r2"],
        "test_median_per_protein_r2": test_metrics["median_per_protein_r2"],
        "test_negative_r2_fraction": test_metrics["negative_r2_fraction"],
        "train_pair_weighted_r2": train_metrics["pair_weighted_r2"],
        "test_pair_weighted_r2": test_metrics["pair_weighted_r2"],
        "train_pair_weighted_mae": train_metrics["pair_weighted_mae"],
        "test_pair_weighted_mae": test_metrics["pair_weighted_mae"],
        "train_pair_weighted_rmse": train_metrics["pair_weighted_rmse"],
        "test_pair_weighted_rmse": test_metrics["pair_weighted_rmse"],
        "tree_depth": int(final_model.get_depth()),
        "n_leaves": int(final_model.get_n_leaves()),
        "elapsed_seconds": float(time.time() - t_start),
    }
    summary_row.update(best["config"])
    summary_row.update(final_aux)

    write_csv(csv_dir / SUMMARY_CSV_NAME, list(summary_row.keys()), [summary_row])

    per_rows = list(train_metrics["per_rows"]) + list(test_metrics["per_rows"])
    write_csv(
        csv_dir / PER_PROTEIN_CSV_NAME,
        [
            "split",
            "protein_id",
            "n_residues",
            "n_pairs_used",
            "per_protein_r2",
            "per_protein_mae",
            "per_protein_rmse",
            "mean_true_distance",
            "mean_predicted_distance",
            "mean_error_pred_minus_true",
        ],
        per_rows,
    )

    seq_rows = []
    seq_rows.extend(
        build_seq_sep_diagnostics("train_sampled_final", train_bundle["seq_sep"], train_bundle["y"], train_pred)
    )
    seq_rows.extend(
        build_seq_sep_diagnostics("test_full", test_bundle["seq_sep"], test_bundle["y"], test_pred)
    )
    write_csv(
        csv_dir / SEQ_DIAGNOSTICS_CSV_NAME,
        [
            "split",
            "seq_bucket",
            "n_pairs",
            "mean_true_distance",
            "mean_predicted_distance",
            "mean_error_pred_minus_true",
            "r2",
            "fraction_pred_lt_3A",
        ],
        seq_rows,
    )

    tree_text = export_text(final_model, feature_names=FEATURE_NAMES, decimals=4)
    (csv_dir / TREE_TEXT_NAME).write_text(tree_text)

    node_rows = build_node_rows(final_model)
    write_csv(
        csv_dir / TREE_NODES_CSV_NAME,
        [
            "node_id",
            "depth",
            "left_child",
            "right_child",
            "feature_index",
            "feature_name",
            "threshold",
            "impurity",
            "n_node_samples",
            "weighted_n_node_samples",
            "value",
            "is_leaf",
        ],
        node_rows,
    )

    importance_rows = build_feature_importance_rows(final_model.feature_importances_)
    write_csv(
        csv_dir / FEATURE_IMPORTANCE_CSV_NAME,
        ["feature_index", "feature_name", "importance"],
        importance_rows,
    )

    write_summary_text(
        csv_dir / SUMMARY_TXT_NAME,
        TITLE,
        elapsed_seconds=float(summary_row["elapsed_seconds"]),
        all_proteins=all_proteins,
        protein_data=protein_data,
        skipped=skipped,
        fit_records=fit_records,
        val_records=val_records,
        train_records=train_records,
        test_records=test_records,
        best_row=summary_row,
        train_metrics=train_metrics,
        test_metrics=test_metrics,
    )

    print("Decision tree diagnostic complete.")
    print(f"Best validation mean per-protein R2: {format_value(best['score'])}")
    print(f"Test mean per-protein R2:           {format_value(test_metrics['mean_per_protein_r2'])}")


if __name__ == "__main__":
    main()

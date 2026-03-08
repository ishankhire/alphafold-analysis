#!/usr/bin/env python3
"""
Fast layer-47 XGBoost diagnostic with protein-level splits.
"""

import time

import numpy as np

from diagnostic_tree_regression_common import (
    CSV_DIR,
    FEATURE_NAMES,
    MAX_TRAIN_PAIRS_PER_PROTEIN,
    build_matrix,
    build_seq_sep_diagnostics,
    build_split_rows,
    fail_missing_dependency,
    format_value,
    load_protein_records,
    map_xgb_importance_names,
    parse_xgb_json_dump,
    run_search,
    split_records,
    summarize_predictions,
    write_csv,
    write_summary_text,
)

try:
    from xgboost import XGBRegressor
except Exception as exc:  # pragma: no cover - environment specific
    fail_missing_dependency("xgboost", exc)


MODEL_NAME = "xgboost"
TITLE = "Diagnostic XGBoost Regression - Layer 47"
SUMMARY_CSV_NAME = "diagnostic_xgboost_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_xgboost_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_xgboost_split.csv"
SEQ_DIAGNOSTICS_CSV_NAME = "diagnostic_xgboost_seq_sep_diagnostics.csv"
SEARCH_CSV_NAME = "diagnostic_xgboost_search.csv"
TREE_DUMP_NAME = "diagnostic_xgboost_tree_dump.txt"
TREE_NODES_CSV_NAME = "diagnostic_xgboost_nodes.csv"
ROOT_SPLITS_CSV_NAME = "diagnostic_xgboost_root_splits.csv"
TOP_SPLITS_CSV_NAME = "diagnostic_xgboost_top_split_features.csv"
IMPORTANCE_GAIN_CSV_NAME = "diagnostic_xgboost_feature_importance_gain.csv"
IMPORTANCE_WEIGHT_CSV_NAME = "diagnostic_xgboost_feature_importance_weight.csv"
IMPORTANCE_COVER_CSV_NAME = "diagnostic_xgboost_feature_importance_cover.csv"
SUMMARY_TXT_NAME = "diagnostic_xgboost_summary.txt"

CONFIGS = [
    {
        "max_depth": 4,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 4,
        "learning_rate": 0.10,
        "n_estimators": 500,
        "min_child_weight": 5,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 5,
        "learning_rate": 0.05,
        "n_estimators": 800,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 5,
        "learning_rate": 0.10,
        "n_estimators": 500,
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 6,
        "learning_rate": 0.05,
        "n_estimators": 700,
        "min_child_weight": 10,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 1.0,
    },
    {
        "max_depth": 6,
        "learning_rate": 0.10,
        "n_estimators": 400,
        "min_child_weight": 10,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "reg_lambda": 2.0,
    },
    {
        "max_depth": 7,
        "learning_rate": 0.05,
        "n_estimators": 600,
        "min_child_weight": 20,
        "subsample": 0.8,
        "colsample_bytree": 0.7,
        "reg_lambda": 2.0,
    },
    {
        "max_depth": 7,
        "learning_rate": 0.10,
        "n_estimators": 350,
        "min_child_weight": 20,
        "subsample": 0.9,
        "colsample_bytree": 0.7,
        "reg_lambda": 2.0,
    },
]

EARLY_STOPPING_ROUNDS = 30


def train_model(config, fit_bundle, val_bundle):
    """Fit one XGBoost config with validation early stopping."""
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(config["n_estimators"]),
        min_child_weight=float(config["min_child_weight"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_lambda=float(config["reg_lambda"]),
        early_stopping_rounds=EARLY_STOPPING_ROUNDS,
        eval_metric="rmse",
    )
    model.fit(
        fit_bundle["x"],
        fit_bundle["y"],
        sample_weight=fit_bundle["weights"],
        eval_set=[(val_bundle["x"], val_bundle["y"])],
        verbose=False,
    )
    best_iteration = getattr(model, "best_iteration", None)
    best_score = getattr(model, "best_score", None)
    return model, {
        "best_iteration": -1 if best_iteration is None else int(best_iteration),
        "best_score_rmse": float("nan") if best_score is None else float(best_score),
    }


def train_final_model(config, n_estimators, train_bundle):
    """Refit on the full sampled train split without validation leakage."""
    model = XGBRegressor(
        objective="reg:squarederror",
        tree_method="hist",
        random_state=42,
        n_jobs=4,
        max_depth=int(config["max_depth"]),
        learning_rate=float(config["learning_rate"]),
        n_estimators=int(n_estimators),
        min_child_weight=float(config["min_child_weight"]),
        subsample=float(config["subsample"]),
        colsample_bytree=float(config["colsample_bytree"]),
        reg_lambda=float(config["reg_lambda"]),
        eval_metric="rmse",
    )
    model.fit(
        train_bundle["x"],
        train_bundle["y"],
        sample_weight=train_bundle["weights"],
        verbose=False,
    )
    return model


def write_tree_dump_text(path, dump):
    """Write raw boosted tree dumps."""
    pieces = []
    for tree_index, tree_dump in enumerate(dump):
        pieces.append(f"tree={tree_index}")
        pieces.append(tree_dump)
        pieces.append("")
    path.write_text("\n".join(pieces))


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

    best_n_estimators = int(best["config"]["n_estimators"])
    searched_best_iteration = int(best["row"].get("best_iteration", -1))
    if searched_best_iteration >= 0:
        best_n_estimators = min(best_n_estimators, searched_best_iteration + 1)

    final_model = train_final_model(best["config"], best_n_estimators, train_bundle)
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

    booster = final_model.get_booster()
    dump, node_rows, root_rows, top_split_rows = parse_xgb_json_dump(booster)
    write_tree_dump_text(csv_dir / TREE_DUMP_NAME, dump)
    write_csv(
        csv_dir / TREE_NODES_CSV_NAME,
        [
            "tree_index",
            "node_id",
            "depth",
            "feature_name",
            "split_condition",
            "yes_child",
            "no_child",
            "missing_child",
            "gain",
            "cover",
            "leaf_value",
            "is_leaf",
        ],
        node_rows,
    )
    write_csv(
        csv_dir / ROOT_SPLITS_CSV_NAME,
        ["tree_index", "feature_name", "split_condition", "gain", "cover"],
        root_rows,
    )
    write_csv(
        csv_dir / TOP_SPLITS_CSV_NAME,
        ["feature_name", "split_count", "total_gain"],
        top_split_rows,
    )

    write_csv(
        csv_dir / IMPORTANCE_GAIN_CSV_NAME,
        ["feature_index", "feature_name", "importance"],
        map_xgb_importance_names(booster.get_score(importance_type="gain")),
    )
    write_csv(
        csv_dir / IMPORTANCE_WEIGHT_CSV_NAME,
        ["feature_index", "feature_name", "importance"],
        map_xgb_importance_names(booster.get_score(importance_type="weight")),
    )
    write_csv(
        csv_dir / IMPORTANCE_COVER_CSV_NAME,
        ["feature_index", "feature_name", "importance"],
        map_xgb_importance_names(booster.get_score(importance_type="cover")),
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
        "best_iteration_search": searched_best_iteration,
        "final_n_estimators": best_n_estimators,
        "n_boosted_rounds": len(dump),
        "elapsed_seconds": float(time.time() - t_start),
    }
    summary_row.update(best["config"])
    summary_row["best_score_rmse"] = best["row"].get("best_score_rmse", float("nan"))
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

    print("XGBoost diagnostic complete.")
    print(f"Best validation mean per-protein R2: {format_value(best['score'])}")
    print(f"Test mean per-protein R2:           {format_value(test_metrics['mean_per_protein_r2'])}")


if __name__ == "__main__":
    main()

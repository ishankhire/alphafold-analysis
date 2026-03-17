#!/usr/bin/env python3
"""
diagnostic_lin_tree_context_reg.py
=================================
Layer-47 diagnostic regression script for protein-aware linear-tree context
variants with per-protein geometry analysis.

Requested variants (all with protein-equal training weights):
1. lower_triangle_center_only_linear_tree
2. lower_triangle_plus_axial_context_linear_tree

Model policy for every variant:
- linear-tree regressor
- max_depth=2
- split only on raw n_residues
- n_residues is used only for splitting (excluded from leaf regression)
- protein-aware min_samples_split / min_samples_leaf are enforced via a local
  runtime patch of lineartree split search

Outputs (inside CSV_ROOT/):
    diagnostic_lin_tree_context_reg_variant_summary.csv
    diagnostic_lin_tree_context_reg_per_protein.csv
    diagnostic_lin_tree_context_reg_split.csv
    diagnostic_lin_tree_context_reg_rules.csv
    diagnostic_lin_tree_context_reg_leaf_coefficients_raw.csv
    diagnostic_lin_tree_context_reg_threshold_search.csv
    diagnostic_lin_tree_context_reg_root_split_geometry_metrics.csv
    diagnostic_lin_tree_context_reg_test_scatter_sample.csv
    diagnostic_lin_tree_context_reg_geometry_metrics.csv
    diagnostic_lin_tree_context_reg_geometry_summary.csv
    diagnostic_lin_tree_context_reg_leaf_weight_diagnostics.csv
    diagnostic_lin_tree_context_reg_summary.txt
"""

import csv
import math
import os
import sys
import time
import traceback
from pathlib import Path

import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_array, check_X_y

try:
    from lineartree import LinearTreeRegressor
    from lineartree import _classes as lineartree_classes
except Exception as exc:  # pragma: no cover - handled explicitly in main
    LinearTreeRegressor = None
    lineartree_classes = None
    LINEARTREE_IMPORT_ERROR = exc
else:
    LINEARTREE_IMPORT_ERROR = None


SCRIPT_DIR = Path(__file__).resolve().parent
CLUSTER_SCRIPTS_DIR = SCRIPT_DIR.parent
if str(CLUSTER_SCRIPTS_DIR) not in sys.path:
    sys.path.insert(0, str(CLUSTER_SCRIPTS_DIR))

from cluster_regression_helpers import (  # noqa: E402
    analyze_protein_geometry,
    compute_distance_matrix,
    discover_proteins,
    extract_ca_coordinates,
    find_pdb_file,
    get_pair_block_path,
)


THRESHOLD_SEARCH_FIELDS = [
    "variant",
    "node_path",
    "node_depth",
    "split_feature_index",
    "split_feature_name",
    "threshold",
    "left_raw_rows",
    "right_raw_rows",
    "left_weight_mass",
    "right_weight_mass",
    "left_loss",
    "right_loss",
    "weighted_left_loss",
    "weighted_right_loss",
    "weighted_total_loss",
    "left_r2",
    "right_r2",
    "test_eval_scope",
    "test_eval_n_proteins",
    "test_eval_n_pairs",
    "test_mean_per_protein_r2",
    "selected_root_test_median_per_protein_r2",
    "selected_root_test_negative_r2_count",
    "selected_root_test_negative_r2_fraction",
    "selected_root_tm_score_mean",
    "selected_root_rmsd_mean",
    "selected_root_stress_mean",
    "selected_root_triangle_violation_fraction_mean",
    "selected_root_neg_eigenvalue_fraction_mean",
    "best_so_far",
]

_THRESHOLD_SEARCH_CSV_PATH = None
_THRESHOLD_SEARCH_ROWS = []


# ===========================================================================
# Compatibility helpers
# ===========================================================================

def _apply_lineartree_sklearn_compat_patch():
    """Patch lineartree for sklearn versions where BaseEstimator._validate_data is absent."""
    if LinearTreeRegressor is None:
        return
    if hasattr(LinearTreeRegressor, "_validate_data"):
        return

    def _compat_validate_data(self, X, y=None, **kwargs):
        params = dict(kwargs)
        params.pop("reset", None)
        params.pop("validate_separately", None)

        if y is None:
            params.pop("multi_output", None)
            params.pop("y_numeric", None)
            try:
                return check_array(X, **params)
            except TypeError:
                if "force_all_finite" in params and "ensure_all_finite" not in params:
                    params["ensure_all_finite"] = params.pop("force_all_finite")
                return check_array(X, **params)

        try:
            return check_X_y(X, y, **params)
        except TypeError:
            if "force_all_finite" in params and "ensure_all_finite" not in params:
                params["ensure_all_finite"] = params.pop("force_all_finite")
            return check_X_y(X, y, **params)

    setattr(LinearTreeRegressor, "_validate_data", _compat_validate_data)


def _downsample_sorted_thresholds(thresholds, max_thresholds):
    """Keep up to max_thresholds evenly spaced sorted thresholds."""
    if len(thresholds) <= max_thresholds:
        return thresholds

    keep_indices = np.linspace(0, len(thresholds) - 1, num=max_thresholds)
    keep_indices = np.rint(keep_indices).astype(np.int64)
    keep_indices = np.unique(keep_indices)

    if keep_indices.size < max_thresholds:
        needed = max_thresholds - keep_indices.size
        extras = np.setdiff1d(np.arange(len(thresholds), dtype=np.int64), keep_indices)
        keep_indices = np.sort(np.concatenate([keep_indices, extras[:needed]]))

    return [thresholds[int(idx)] for idx in keep_indices[:max_thresholds]]


def _emit_threshold_search_row(row):
    """Append one threshold diagnostic row and mirror it to live stdout."""
    global _THRESHOLD_SEARCH_CSV_PATH, _THRESHOLD_SEARCH_ROWS

    _THRESHOLD_SEARCH_ROWS.append(dict(row))

    test_mean_pp_r2 = row.get("test_mean_per_protein_r2", float("nan"))
    if isinstance(test_mean_pp_r2, str):
        try:
            test_mean_pp_r2 = float(test_mean_pp_r2)
        except ValueError:
            test_mean_pp_r2 = float("nan")

    print(
        "[THRESH] "
        f"variant={row['variant']} "
        f"path={row['node_path'] or 'ROOT'} "
        f"depth={row['node_depth']} "
        f"feature={row['split_feature_name']} "
        f"threshold={row['threshold']:.6f} "
        f"left_mass={row['left_weight_mass']:.3f} "
        f"right_mass={row['right_weight_mass']:.3f} "
        f"left_r2={row['left_r2']:.4f} "
        f"right_r2={row['right_r2']:.4f} "
        f"test_mean_pp_r2={test_mean_pp_r2:.4f} "
        f"weighted_total_loss={row['weighted_total_loss']:.5f} "
        f"best_so_far={'yes' if row['best_so_far'] else 'no'}",
        flush=True,
    )

    if _THRESHOLD_SEARCH_CSV_PATH is None:
        return

    with _THRESHOLD_SEARCH_CSV_PATH.open("a", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=THRESHOLD_SEARCH_FIELDS)
        writer.writerow({key: format_value(row.get(key, "")) for key in THRESHOLD_SEARCH_FIELDS})


def _emit_split_search_status(message):
    """Print live split-search status lines for tail -f monitoring."""
    print(message, flush=True)


def _sync_threshold_search_output():
    """Rewrite the threshold-search CSV from in-memory rows."""
    if _THRESHOLD_SEARCH_CSV_PATH is None:
        return
    write_csv(_THRESHOLD_SEARCH_CSV_PATH, THRESHOLD_SEARCH_FIELDS, _THRESHOLD_SEARCH_ROWS)


def _build_protein_aware_thresholds(x_column, weights, min_leaf_mass, max_thresholds):
    """Build protein-aware cutpoints from unique split values in the current node."""
    if x_column.size == 0 or weights is None:
        return []

    values = np.asarray(x_column, dtype=np.float64)
    masses = np.asarray(weights, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    values = values[order]
    masses = masses[order]

    unique_values, first_idx = np.unique(values, return_index=True)
    if unique_values.size < 2:
        return []

    mass_per_value = np.add.reduceat(masses, first_idx)
    cumulative_mass = np.cumsum(mass_per_value)
    total_mass = float(cumulative_mass[-1])

    thresholds = []
    for idx in range(unique_values.size - 1):
        left_mass = float(cumulative_mass[idx])
        right_mass = total_mass - left_mass
        if left_mass < min_leaf_mass or right_mass < min_leaf_mass:
            continue

        left_value = float(unique_values[idx])
        right_value = float(unique_values[idx + 1])
        thresholds.append(0.5 * (left_value + right_value))

    return _downsample_sorted_thresholds(thresholds, max_thresholds)


def _predict_threshold_split(x, split_col, threshold, linear_features, model_left, model_right):
    """Predict one split model on a dense matrix."""
    y_pred = np.empty((x.shape[0],), dtype=np.float64)
    if x.shape[0] == 0:
        return y_pred

    mask_right = x[:, split_col] > threshold
    mask_left = ~mask_right

    if np.any(mask_left):
        left_mesh = np.ix_(mask_left, linear_features)
        y_pred[mask_left] = np.asarray(model_left.predict(x[left_mesh]), dtype=np.float64)
    if np.any(mask_right):
        right_mesh = np.ix_(mask_right, linear_features)
        y_pred[mask_right] = np.asarray(model_right.predict(x[right_mesh]), dtype=np.float64)
    return y_pred


def summarize_masked_predictions(y_true, y_pred, split_records, offsets, row_mask):
    """Compute per-protein summaries on a boolean-selected subset of rows."""
    row_mask = np.asarray(row_mask, dtype=bool)
    if row_mask.size == 0 or not np.any(row_mask):
        return {
            "mean_per_protein_r2": float("nan"),
            "median_per_protein_r2": float("nan"),
            "negative_r2_count": 0,
            "negative_r2_fraction": float("nan"),
            "n_selected_proteins": 0,
            "n_selected_pairs": 0,
            "per_rows": [],
        }

    per_rows = []
    per_r2 = []
    negative_r2_count = 0
    n_selected_pairs = int(row_mask.sum())

    for record, (start, end) in zip(split_records, offsets):
        local_mask = row_mask[start:end]
        if not np.any(local_mask):
            continue

        y_true_p = y_true[start:end][local_mask]
        y_pred_p = y_pred[start:end][local_mask]
        per_protein_r2 = compute_r2(y_true_p, y_pred_p)
        if not math.isnan(per_protein_r2):
            per_r2.append(per_protein_r2)
            if per_protein_r2 < 0.0:
                negative_r2_count += 1

        error = y_pred_p - y_true_p
        per_rows.append(
            {
                "protein_id": record["protein_id"],
                "n_residues": record["n_residues"],
                "n_pairs": int(local_mask.sum()),
                "per_protein_r2": per_protein_r2,
                "mean_true_distance": float(np.mean(y_true_p)),
                "std_true_distance": float(np.std(y_true_p)),
                "mean_predicted_distance": float(np.mean(y_pred_p)),
                "std_predicted_distance": float(np.std(y_pred_p)),
                "mean_error_pred_minus_true": float(np.mean(error)),
            }
        )

    return {
        "mean_per_protein_r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
        "median_per_protein_r2": float(np.median(per_r2)) if per_r2 else float("nan"),
        "negative_r2_count": int(negative_r2_count),
        "negative_r2_fraction": (
            float(negative_r2_count / len(per_r2)) if per_r2 else float("nan")
        ),
        "n_selected_proteins": len(per_rows),
        "n_selected_pairs": n_selected_pairs,
        "per_rows": per_rows,
    }


def evaluate_candidate_threshold_on_test(
    test_bundle,
    ancestor_thresholds,
    split_col,
    threshold,
    linear_features,
    model_left,
    model_right,
):
    """Evaluate one candidate threshold on the appropriate test scope."""
    if test_bundle is None:
        return {
            "test_eval_scope": "unavailable",
            "test_eval_n_proteins": 0,
            "test_eval_n_pairs": 0,
            "test_mean_per_protein_r2": float("nan"),
        }

    x_test = test_bundle["x"]
    y_test = test_bundle["y"]
    test_records = test_bundle["records"]
    test_offsets = test_bundle["offsets"]

    if ancestor_thresholds:
        row_mask = lineartree_classes._predict_branch(x_test, ancestor_thresholds)
        eval_scope = "node_routed_subset"
    else:
        row_mask = np.ones((x_test.shape[0],), dtype=bool)
        eval_scope = "full_test"

    y_pred = np.full((x_test.shape[0],), np.nan, dtype=np.float64)
    if np.any(row_mask):
        routed_x = x_test[row_mask]
        y_pred[row_mask] = _predict_threshold_split(
            routed_x,
            split_col=split_col,
            threshold=threshold,
            linear_features=linear_features,
            model_left=model_left,
            model_right=model_right,
        )

    metrics = summarize_masked_predictions(y_test, y_pred, test_records, test_offsets, row_mask)
    return {
        "test_eval_scope": eval_scope,
        "test_eval_n_proteins": metrics["n_selected_proteins"],
        "test_eval_n_pairs": metrics["n_selected_pairs"],
        "test_mean_per_protein_r2": metrics["mean_per_protein_r2"],
    }


def get_root_split_artifacts(model):
    """Return root split metadata and child models if the fitted tree split at root."""
    root_node = getattr(model, "_nodes", {}).get("")
    if root_node is None or not getattr(root_node, "children", None):
        return None

    left_path, right_path = root_node.children
    left_child = getattr(model, "_nodes", {}).get(left_path)
    if left_child is None:
        left_child = getattr(model, "_leaves", {}).get(left_path)
    right_child = getattr(model, "_nodes", {}).get(right_path)
    if right_child is None:
        right_child = getattr(model, "_leaves", {}).get(right_path)
    if left_child is None or right_child is None:
        return None

    threshold_tuples = list(getattr(left_child, "threshold", []))
    if not threshold_tuples:
        return None

    split_col, _, threshold = threshold_tuples[0]
    return {
        "split_col": int(split_col),
        "threshold": float(threshold),
        "left_path": left_path,
        "right_path": right_path,
        "left_model": left_child.model,
        "right_model": right_child.model,
    }


def annotate_selected_root_threshold_row(variant_name, selected_root_summary):
    """Populate the chosen root-threshold row with selected-root test diagnostics."""
    if selected_root_summary is None:
        return

    target_threshold = float(selected_root_summary["threshold"])
    best_row = None
    best_diff = float("inf")
    for row in _THRESHOLD_SEARCH_ROWS:
        if row.get("variant") != variant_name:
            continue
        if row.get("node_path", "") != "":
            continue
        diff = abs(float(row.get("threshold", float("nan"))) - target_threshold)
        if diff < best_diff:
            best_diff = diff
            best_row = row

    if best_row is not None:
        best_row["selected_root_test_median_per_protein_r2"] = selected_root_summary[
            "test_median_per_protein_r2"
        ]
        best_row["selected_root_test_negative_r2_count"] = selected_root_summary[
            "test_negative_r2_count"
        ]
        best_row["selected_root_test_negative_r2_fraction"] = selected_root_summary[
            "test_negative_r2_fraction"
        ]
        best_row["selected_root_tm_score_mean"] = selected_root_summary["tm_score_mean"]
        best_row["selected_root_rmsd_mean"] = selected_root_summary["rmsd_mean"]
        best_row["selected_root_stress_mean"] = selected_root_summary["stress_mean"]
        best_row["selected_root_triangle_violation_fraction_mean"] = selected_root_summary[
            "triangle_violation_fraction_mean"
        ]
        best_row["selected_root_neg_eigenvalue_fraction_mean"] = selected_root_summary[
            "neg_eigenvalue_fraction_mean"
        ]

    _sync_threshold_search_output()


def evaluate_selected_root_split(model, variant_name, test_bundle):
    """Evaluate the chosen root split as an implicit depth-1 model on test."""
    root_artifacts = get_root_split_artifacts(model)
    if root_artifacts is None:
        return None, []

    x_test = test_bundle["x"]
    y_test = test_bundle["y"]
    test_records = test_bundle["records"]
    y_pred = _predict_threshold_split(
        x_test,
        split_col=root_artifacts["split_col"],
        threshold=root_artifacts["threshold"],
        linear_features=model._linear_features,
        model_left=root_artifacts["left_model"],
        model_right=root_artifacts["right_model"],
    )

    test_metrics = summarize_predictions(y_test, y_pred, test_records)
    geometry_rows = collect_geometry_rows(
        variant_name,
        "test_selected_root_split",
        test_records,
        y_test,
        y_pred,
    )
    geometry_summary = summarize_geometry_rows(geometry_rows)
    geom_row = geometry_summary[0] if geometry_summary else {}

    summary_row = {
        "variant": variant_name,
        "threshold": float(root_artifacts["threshold"]),
        "test_mean_per_protein_r2": test_metrics["mean_per_protein_r2"],
        "test_median_per_protein_r2": test_metrics["median_per_protein_r2"],
        "test_negative_r2_count": test_metrics["negative_r2_count"],
        "test_negative_r2_fraction": test_metrics["negative_r2_fraction"],
        "tm_score_mean": geom_row.get("tm_score_mean", float("nan")),
        "rmsd_mean": geom_row.get("rmsd_mean", float("nan")),
        "stress_mean": geom_row.get("stress_mean", float("nan")),
        "triangle_violation_fraction_mean": geom_row.get(
            "triangle_violation_fraction_mean",
            float("nan"),
        ),
        "neg_eigenvalue_fraction_mean": geom_row.get(
            "neg_eigenvalue_fraction_mean",
            float("nan"),
        ),
    }
    return summary_row, geometry_rows


def _apply_lineartree_protein_threshold_patch():
    """Patch lineartree split search to support protein-aware weighted thresholds."""
    if lineartree_classes is None:
        return

    current = getattr(lineartree_classes, "_parallel_binning_fit", None)
    if current is None:
        return
    if getattr(current, "_protein_weight_patch", False):
        return

    def _protein_aware_parallel_binning_fit(
        split_feat,
        _self,
        X,
        y,
        weights,
        support_sample_weight,
        bins,
        loss,
    ):
        """Find the best split while optionally enforcing weighted thresholds."""
        n_sample, _ = X.shape
        feval = lineartree_classes.CRITERIA[_self.criterion]

        split_t = None
        split_col = None
        left_node = (None, None, None, None)
        right_node = (None, None, None, None)
        largs_left = {"classes": None}
        largs_right = {"classes": None}
        variant_name = getattr(_self, "_diagnostic_variant_name", "unknown_variant")
        node_path = getattr(_self, "_current_node_path", "")
        feature_names = getattr(_self, "_diagnostic_feature_names", None)

        use_weighted_thresholds = bool(
            weights is not None and getattr(_self, "_protein_aware_thresholds", False)
        )
        max_thresholds = int(
            getattr(
                _self,
                "_protein_aware_max_thresholds",
                LINEAR_TREE_MAX_PROTEIN_THRESHOLDS,
            )
        )

        if use_weighted_thresholds:
            min_split_mass = float(
                getattr(_self, "_weighted_min_samples_split", _self._min_samples_split)
            )
            node_mass = float(weights.sum())
            if node_mass < min_split_mass:
                _emit_split_search_status(
                    "[NODE] "
                    f"variant={variant_name} path={node_path or 'ROOT'} "
                    f"skipped_split_search node_mass={node_mass:.3f} "
                    f"required_min_split={min_split_mass:.3f}"
                )
                return loss, split_t, split_col, left_node, right_node
        elif n_sample < _self._min_samples_split:
            return loss, split_t, split_col, left_node, right_node

        _emit_split_search_status(
            "[NODE] "
            f"variant={variant_name} path={node_path or 'ROOT'} "
            f"begin_split_search n_rows={n_sample} "
            f"node_mass={(float(weights.sum()) if weights is not None else float(n_sample)):.3f} "
            f"parent_loss={loss:.5f}"
        )

        for col, _bin in zip(split_feat, bins):
            if use_weighted_thresholds:
                min_leaf_mass = float(
                    getattr(_self, "_weighted_min_samples_leaf", _self._min_samples_leaf)
                )
                candidate_thresholds = _build_protein_aware_thresholds(
                    X[:, col],
                    weights,
                    min_leaf_mass=min_leaf_mass,
                    max_thresholds=max_thresholds,
                )
            else:
                candidate_thresholds = _bin

            for q in candidate_thresholds:
                mask = X[:, col] > q
                n_left = int((~mask).sum())
                n_right = int(mask.sum())

                if use_weighted_thresholds:
                    left_mass = float(weights[~mask].sum())
                    right_mass = float(weights[mask].sum())
                    if left_mass < min_leaf_mass or right_mass < min_leaf_mass:
                        continue
                elif n_left < _self._min_samples_leaf or n_right < _self._min_samples_leaf:
                    continue

                left_mesh = np.ix_(~mask, _self._linear_features)
                right_mesh = np.ix_(mask, _self._linear_features)

                model_left = lineartree_classes.deepcopy(_self.base_estimator)
                model_right = lineartree_classes.deepcopy(_self.base_estimator)

                if hasattr(_self, "classes_"):
                    largs_left["classes"] = np.unique(y[~mask])
                    largs_right["classes"] = np.unique(y[mask])
                    if len(largs_left["classes"]) == 1:
                        model_left = lineartree_classes.DummyClassifier(
                            strategy="most_frequent"
                        )
                    if len(largs_right["classes"]) == 1:
                        model_right = lineartree_classes.DummyClassifier(
                            strategy="most_frequent"
                        )

                if weights is None:
                    model_left.fit(X[left_mesh], y[~mask])
                    loss_left = feval(model_left, X[left_mesh], y[~mask], **largs_left)
                    wloss_left = loss_left * (n_left / n_sample)

                    model_right.fit(X[right_mesh], y[mask])
                    loss_right = feval(model_right, X[right_mesh], y[mask], **largs_right)
                    wloss_right = loss_right * (n_right / n_sample)
                else:
                    if support_sample_weight:
                        model_left.fit(X[left_mesh], y[~mask], sample_weight=weights[~mask])
                        model_right.fit(X[right_mesh], y[mask], sample_weight=weights[mask])
                    else:
                        model_left.fit(X[left_mesh], y[~mask])
                        model_right.fit(X[right_mesh], y[mask])

                    loss_left = feval(
                        model_left,
                        X[left_mesh],
                        y[~mask],
                        weights=weights[~mask],
                        **largs_left,
                    )
                    wloss_left = loss_left * (weights[~mask].sum() / weights.sum())

                    loss_right = feval(
                        model_right,
                        X[right_mesh],
                        y[mask],
                        weights=weights[mask],
                        **largs_right,
                    )
                    wloss_right = loss_right * (weights[mask].sum() / weights.sum())

                total_loss = round(wloss_left + wloss_right, 5)
                left_pred = np.asarray(model_left.predict(X[left_mesh]), dtype=np.float64)
                right_pred = np.asarray(model_right.predict(X[right_mesh]), dtype=np.float64)
                current_node = getattr(_self, "_nodes", {}).get(node_path)
                ancestor_thresholds = (
                    list(getattr(current_node, "threshold", [])) if current_node is not None else []
                )
                test_metrics = evaluate_candidate_threshold_on_test(
                    getattr(_self, "_diagnostic_test_bundle", None),
                    ancestor_thresholds=ancestor_thresholds,
                    split_col=int(col),
                    threshold=float(q),
                    linear_features=_self._linear_features,
                    model_left=model_left,
                    model_right=model_right,
                )
                feat_name = (
                    feature_names[int(col)]
                    if feature_names is not None and 0 <= int(col) < len(feature_names)
                    else f"feature_{int(col)}"
                )
                is_best = bool(total_loss < loss)
                _emit_threshold_search_row(
                    {
                        "variant": variant_name,
                        "node_path": node_path,
                        "node_depth": len(node_path),
                        "split_feature_index": int(col),
                        "split_feature_name": feat_name,
                        "threshold": float(q),
                        "left_raw_rows": n_left,
                        "right_raw_rows": n_right,
                        "left_weight_mass": (
                            float(weights[~mask].sum()) if weights is not None else float(n_left)
                        ),
                        "right_weight_mass": (
                            float(weights[mask].sum()) if weights is not None else float(n_right)
                        ),
                        "left_loss": float(loss_left),
                        "right_loss": float(loss_right),
                        "weighted_left_loss": float(wloss_left),
                        "weighted_right_loss": float(wloss_right),
                        "weighted_total_loss": float(total_loss),
                        "left_r2": compute_r2(y[~mask], left_pred),
                        "right_r2": compute_r2(y[mask], right_pred),
                        **test_metrics,
                        "selected_root_test_median_per_protein_r2": float("nan"),
                        "selected_root_test_negative_r2_count": "",
                        "selected_root_test_negative_r2_fraction": float("nan"),
                        "selected_root_tm_score_mean": float("nan"),
                        "selected_root_rmsd_mean": float("nan"),
                        "selected_root_stress_mean": float("nan"),
                        "selected_root_triangle_violation_fraction_mean": float("nan"),
                        "selected_root_neg_eigenvalue_fraction_mean": float("nan"),
                        "best_so_far": is_best,
                    }
                )
                if total_loss < loss:
                    split_t = q
                    split_col = col
                    loss = total_loss
                    left_node = (
                        model_left,
                        loss_left,
                        wloss_left,
                        n_left,
                        largs_left["classes"],
                    )
                    right_node = (
                        model_right,
                        loss_right,
                        wloss_right,
                        n_right,
                        largs_right["classes"],
                    )

        if split_col is None:
            _emit_split_search_status(
                "[NODE] "
                f"variant={variant_name} path={node_path or 'ROOT'} "
                "finished_split_search best_threshold=None"
            )
        else:
            feat_name = (
                feature_names[int(split_col)]
                if feature_names is not None and 0 <= int(split_col) < len(feature_names)
                else f"feature_{int(split_col)}"
            )
            _emit_split_search_status(
                "[NODE] "
                f"variant={variant_name} path={node_path or 'ROOT'} "
                f"finished_split_search best_feature={feat_name} "
                f"best_threshold={float(split_t):.6f} best_weighted_loss={float(loss):.5f}"
            )
        return loss, split_t, split_col, left_node, right_node

    _protein_aware_parallel_binning_fit._protein_weight_patch = True
    lineartree_classes._parallel_binning_fit = _protein_aware_parallel_binning_fit

    original_grow = LinearTreeRegressor._grow
    if not getattr(original_grow, "_diagnostic_node_path_patch", False):

        def _diagnostic_grow(self, X, y, weights=None):
            """Mirror lineartree growth while tracking the active node path."""
            n_sample, self.n_features_in_ = X.shape
            self.feature_importances_ = np.zeros((self.n_features_in_,))

            bins = np.linspace(0, 1, self.max_bins)[1:-1]
            bins = np.quantile(X, bins, axis=0)
            bins = list(bins.T)
            bins = [
                np.unique(X[:, c]) if c in self._categorical_features else np.unique(q)
                for c, q in enumerate(bins)
            ]

            support_sample_weight = lineartree_classes.has_fit_parameter(
                self.base_estimator,
                "sample_weight",
            )

            queue = [""]
            self._nodes = {}
            self._leaves = {}

            largs = {"classes": None}
            model = lineartree_classes.deepcopy(self.base_estimator)
            if weights is None or not support_sample_weight:
                model.fit(X[:, self._linear_features], y)
            else:
                model.fit(X[:, self._linear_features], y, sample_weight=weights)

            if hasattr(self, "classes_"):
                largs["classes"] = self.classes_

            loss = lineartree_classes.CRITERIA[self.criterion](
                model,
                X[:, self._linear_features],
                y,
                weights=weights,
                **largs,
            )
            loss = round(loss, 5)

            self._nodes[""] = lineartree_classes.Node(
                id=0,
                n_samples=n_sample,
                model=model,
                loss=loss,
                classes=largs["classes"],
            )

            start = np.repeat(True, n_sample)
            mask = start.copy()

            i = 1
            while len(queue) > 0:
                active_path = queue[-1]
                self._current_node_path = active_path

                if len(active_path) >= self.max_depth:
                    self._leaves[active_path] = self._nodes[active_path]
                    del self._nodes[active_path]
                    queue.pop()
                    if len(queue) > 0:
                        next_path = queue[-1]
                        loss = self._nodes[next_path].loss
                        mask = lineartree_classes._predict_branch(
                            X,
                            self._nodes[next_path].threshold,
                            start.copy(),
                        )
                    continue

                if weights is None:
                    split_t, split_col, left_node, right_node = self._split(
                        X[mask],
                        y[mask],
                        bins,
                        support_sample_weight,
                        loss=loss,
                    )
                else:
                    split_t, split_col, left_node, right_node = self._split(
                        X[mask],
                        y[mask],
                        bins,
                        support_sample_weight,
                        weights[mask],
                        loss=loss,
                    )

                if split_col is None or len(active_path) >= self.max_depth:
                    self._leaves[active_path] = self._nodes[active_path]
                    del self._nodes[active_path]
                    queue.pop()
                else:
                    model_left, loss_left, wloss_left, n_left, class_left = left_node
                    model_right, loss_right, wloss_right, n_right, class_right = right_node
                    self.feature_importances_[split_col] += loss - wloss_left - wloss_right

                    self._nodes[active_path + "L"] = lineartree_classes.Node(
                        id=i,
                        parent=active_path,
                        model=model_left,
                        loss=loss_left,
                        w_loss=wloss_left,
                        n_samples=n_left,
                        threshold=self._nodes[active_path].threshold[:] + [
                            (split_col, "L", split_t)
                        ],
                    )

                    self._nodes[active_path + "R"] = lineartree_classes.Node(
                        id=i + 1,
                        parent=active_path,
                        model=model_right,
                        loss=loss_right,
                        w_loss=wloss_right,
                        n_samples=n_right,
                        threshold=self._nodes[active_path].threshold[:] + [
                            (split_col, "R", split_t)
                        ],
                    )

                    if hasattr(self, "classes_"):
                        self._nodes[active_path + "L"].classes = class_left
                        self._nodes[active_path + "R"].classes = class_right

                    self._nodes[active_path].children = (active_path + "L", active_path + "R")

                    i += 2
                    queue.pop()
                    queue.extend([active_path + "R", active_path + "L"])

                if len(queue) > 0:
                    next_path = queue[-1]
                    loss = self._nodes[next_path].loss
                    mask = lineartree_classes._predict_branch(
                        X,
                        self._nodes[next_path].threshold,
                        start.copy(),
                    )

            self.node_count = i
            self._current_node_path = ""
            return self

        _diagnostic_grow._diagnostic_node_path_patch = True
        LinearTreeRegressor._grow = _diagnostic_grow


# ===========================================================================
# Configuration
# ===========================================================================

N_CHANNELS = 128
LAYER = 47
TEST_SIZE = 0.2
RANDOM_STATE = 42
FLOAT_EPS = 1e-12
LINEAR_TREE_MAX_DEPTH = 2
LINEAR_TREE_MIN_SAMPLES_SPLIT = 20
LINEAR_TREE_MIN_SAMPLES_LEAF = 10
LINEAR_TREE_MAX_PROTEIN_THRESHOLDS = 10
THRESHOLD_POLICY = f"protein_aware_midpoints_capped_{LINEAR_TREE_MAX_PROTEIN_THRESHOLDS}"
SCATTER_SAMPLE_PER_VARIANT = 100_000

CENTER_OFFSETS = [(0, 0)]
AXIAL_OFFSETS = [(0, 0), (1, 0), (-1, 0), (0, 1), (0, -1)]

SUMMARY_CSV_NAME = "diagnostic_lin_tree_context_reg_variant_summary.csv"
PER_PROTEIN_CSV_NAME = "diagnostic_lin_tree_context_reg_per_protein.csv"
SPLIT_CSV_NAME = "diagnostic_lin_tree_context_reg_split.csv"
TREE_RULES_CSV_NAME = "diagnostic_lin_tree_context_reg_rules.csv"
TREE_COEF_CSV_NAME = "diagnostic_lin_tree_context_reg_leaf_coefficients_raw.csv"
THRESHOLD_SEARCH_CSV_NAME = "diagnostic_lin_tree_context_reg_threshold_search.csv"
ROOT_SPLIT_GEOMETRY_CSV_NAME = "diagnostic_lin_tree_context_reg_root_split_geometry_metrics.csv"
SCATTER_SAMPLE_CSV_NAME = "diagnostic_lin_tree_context_reg_test_scatter_sample.csv"
GEOMETRY_METRICS_CSV_NAME = "diagnostic_lin_tree_context_reg_geometry_metrics.csv"
GEOMETRY_SUMMARY_CSV_NAME = "diagnostic_lin_tree_context_reg_geometry_summary.csv"
LEAF_WEIGHT_DIAGNOSTICS_CSV_NAME = "diagnostic_lin_tree_context_reg_leaf_weight_diagnostics.csv"
SUMMARY_TXT_NAME = "diagnostic_lin_tree_context_reg_summary.txt"


def _feature_dim_for_slots(n_slots, bidirectional):
    """Return feature width including final n_residues split column."""
    slot_dim = 2 * N_CHANNELS if bidirectional else N_CHANNELS
    return (n_slots * slot_dim) + 1


def build_variant_specs():
    """Define requested context variants."""
    return [
        {
            "name": "lower_triangle_center_only_linear_tree",
            "feature_mode": "lower_triangle_center_only",
            "offsets": CENTER_OFFSETS,
            "bidirectional": False,
            "feature_dim": _feature_dim_for_slots(len(CENTER_OFFSETS), False),
            "max_depth": LINEAR_TREE_MAX_DEPTH,
        },
        {
            "name": "lower_triangle_plus_axial_context_linear_tree",
            "feature_mode": "lower_triangle_plus_axial_context",
            "offsets": AXIAL_OFFSETS,
            "bidirectional": False,
            "feature_dim": _feature_dim_for_slots(len(AXIAL_OFFSETS), False),
            "max_depth": LINEAR_TREE_MAX_DEPTH,
        },
    ]


VARIANTS = build_variant_specs()
for variant in VARIANTS:
    variant["split_feature_index"] = variant["feature_dim"] - 1


# ===========================================================================
# Directories
# ===========================================================================

PAIR_BLOCKS_ROOT = Path(os.environ["PAIR_BLOCKS_ROOT"])
PDB_ROOT = Path(os.environ["PDB_ROOT"])
CSV_DIR = Path(os.environ["CSV_ROOT"])


# ===========================================================================
# Split and weighting helpers
# ===========================================================================

def split_global_records(records):
    """Create a single deterministic protein-level train/test split."""
    if len(records) < 2:
        raise ValueError("Need at least two proteins to split into train/test.")

    indices = np.arange(len(records))
    train_idx, test_idx = train_test_split(
        indices,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
    )

    train_set = set(int(i) for i in train_idx)
    test_set = set(int(i) for i in test_idx)
    train_records = [record for idx, record in enumerate(records) if idx in train_set]
    test_records = [record for idx, record in enumerate(records) if idx in test_set]
    return train_records, test_records


def compute_split_pair_weights(split_records, normalize_mean=False):
    """Return per-pair sample weights so each protein contributes equally."""
    pieces = [
        np.full(record["n_pairs"], 1.0 / record["n_pairs"], dtype=np.float64)
        for record in split_records
    ]
    if not pieces:
        return np.empty((0,), dtype=np.float64)

    weights = np.concatenate(pieces)
    if normalize_mean and weights.size > 0:
        weights *= weights.size / np.sum(weights)
    return weights


def build_offsets(split_records):
    """Build row offsets for one split."""
    offsets = []
    start = 0
    for record in split_records:
        end = start + record["n_pairs"]
        offsets.append((start, end))
        start = end
    return offsets


# ===========================================================================
# Feature construction helpers
# ===========================================================================

def format_offset_label(offset):
    """Create a stable string label for one offset."""
    di, dj = offset
    return f"di_{di:+d}_dj_{dj:+d}".replace("+", "p").replace("-", "m")


def get_feature_names(variant):
    """Return ordered feature names for one variant."""
    names = []
    for offset in variant["offsets"]:
        offset_label = format_offset_label(offset)
        if variant["bidirectional"]:
            names.extend([f"{offset_label}_pb_ab_{idx}" for idx in range(N_CHANNELS)])
            names.extend([f"{offset_label}_pb_ba_{idx}" for idx in range(N_CHANNELS)])
        else:
            names.extend([f"{offset_label}_pb_ab_{idx}" for idx in range(N_CHANNELS)])
    names.append("n_residues")
    return names


def build_protein_feature_chunk(record, variant):
    """Load one protein pair block and build dense feature rows for one variant."""
    pb_path = get_pair_block_path(record["protein_dir"], record["protein_id"], LAYER)
    pb = np.load(str(pb_path), mmap_mode="r")

    pair_i = record["pair_i"]
    pair_j = record["pair_j"]
    n_pairs = record["n_pairs"]
    n_residues = float(record["n_residues"])
    feature_dim = int(variant["feature_dim"])
    chunk = np.zeros((n_pairs, feature_dim), dtype=np.float32)

    write_col = 0
    for di, dj in variant["offsets"]:
        src_i = pair_i + di
        src_j = pair_j + dj
        in_bounds = (
            (src_i >= 0)
            & (src_i < record["n_residues"])
            & (src_j >= 0)
            & (src_j < record["n_residues"])
        )

        if variant["bidirectional"]:
            valid = in_bounds
            if np.any(valid):
                chunk[valid, write_col:write_col + N_CHANNELS] = pb[src_i[valid], src_j[valid], :]
                write_col += N_CHANNELS
                chunk[valid, write_col:write_col + N_CHANNELS] = pb[src_j[valid], src_i[valid], :]
                write_col += N_CHANNELS
            else:
                write_col += 2 * N_CHANNELS
        else:
            valid = in_bounds & (src_i > src_j)
            if np.any(valid):
                chunk[valid, write_col:write_col + N_CHANNELS] = pb[src_i[valid], src_j[valid], :]
            write_col += N_CHANNELS

    chunk[:, -1] = n_residues
    del pb
    return chunk


def build_split_matrix(split_records, variant):
    """Build dense matrix and target vector for one split and one variant."""
    n_rows = int(sum(record["n_pairs"] for record in split_records))
    x = np.empty((n_rows, variant["feature_dim"]), dtype=np.float32)
    y = np.empty((n_rows,), dtype=np.float64)

    offset = 0
    for record in split_records:
        chunk = build_protein_feature_chunk(record, variant)
        end = offset + record["n_pairs"]
        x[offset:end, :] = chunk
        y[offset:end] = record["y_abs"]
        offset = end
        del chunk

    return x, y


def fit_and_apply_standardization(x_train, x_test, n_scale_features):
    """Standardize leading feature columns using train-only moments."""
    if n_scale_features <= 0:
        return

    mean = np.mean(x_train[:, :n_scale_features], axis=0, dtype=np.float64)
    scale = np.std(x_train[:, :n_scale_features], axis=0, dtype=np.float64)
    scale[scale < FLOAT_EPS] = 1.0

    x_train[:, :n_scale_features] -= mean
    x_train[:, :n_scale_features] /= scale
    x_test[:, :n_scale_features] -= mean
    x_test[:, :n_scale_features] /= scale


# ===========================================================================
# Metrics helpers
# ===========================================================================

def compute_r2(y_true, y_pred):
    """Compute vanilla R-squared."""
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


def summarize_predictions(y_true, y_pred, split_records):
    """Compute per-protein R2 rows and split-level summaries."""
    offsets = build_offsets(split_records)
    per_rows = []
    per_r2 = []
    negative_r2_count = 0

    for record, (start, end) in zip(split_records, offsets):
        y_true_p = y_true[start:end]
        y_pred_p = y_pred[start:end]
        per_r2_val = compute_r2(y_true_p, y_pred_p)

        if not math.isnan(per_r2_val):
            per_r2.append(per_r2_val)
            if per_r2_val < 0.0:
                negative_r2_count += 1

        error = y_pred_p - y_true_p
        per_rows.append(
            {
                "protein_id": record["protein_id"],
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
                "per_protein_r2": per_r2_val,
                "mean_true_distance": float(np.mean(y_true_p)),
                "std_true_distance": float(np.std(y_true_p)),
                "mean_predicted_distance": float(np.mean(y_pred_p)),
                "std_predicted_distance": float(np.std(y_pred_p)),
                "mean_error_pred_minus_true": float(np.mean(error)),
            }
        )

    return {
        "mean_per_protein_r2": float(np.mean(per_r2)) if per_r2 else float("nan"),
        "median_per_protein_r2": float(np.median(per_r2)) if per_r2 else float("nan"),
        "negative_r2_count": int(negative_r2_count),
        "negative_r2_fraction": (
            float(negative_r2_count / len(per_r2)) if per_r2 else float("nan")
        ),
        "per_rows": per_rows,
    }


def summarize_geometry_rows(geometry_rows):
    """Aggregate geometry metrics per variant and split."""
    metric_keys = [
        "per_protein_r2",
        "tm_score",
        "rmsd",
        "stress",
        "frac_top3_eigenvalues",
        "n_negative_eigenvalues",
        "neg_eigenvalue_fraction",
        "n_negative_predictions",
        "triangle_violation_fraction",
        "triangle_mean_violation",
        "triangle_max_violation",
    ]

    grouped = {}
    for row in geometry_rows:
        key = (row["variant"], row["split"])
        grouped.setdefault(key, []).append(row)

    summary_rows = []
    for (variant_name, split_name), rows in sorted(grouped.items()):
        out = {
            "variant": variant_name,
            "split": split_name,
            "n_proteins": len(rows),
        }
        for metric_key in metric_keys:
            values = [
                float(row[metric_key])
                for row in rows
                if isinstance(row[metric_key], (int, float, np.floating, np.integer))
                and not math.isnan(float(row[metric_key]))
            ]
            out[f"{metric_key}_mean"] = float(np.mean(values)) if values else float("nan")
            out[f"{metric_key}_std"] = float(np.std(values)) if values else float("nan")
        summary_rows.append(out)

    return summary_rows


def make_failed_geometry_row(variant_name, split_name, record, per_protein_r2):
    """Return a placeholder geometry row when reconstruction fails."""
    return {
        "variant": variant_name,
        "split": split_name,
        "protein_id": record["protein_id"],
        "n_residues": record["n_residues"],
        "n_pairs": record["n_pairs"],
        "per_protein_r2": per_protein_r2,
        "rmsd": float("nan"),
        "tm_score": float("nan"),
        "frac_top3_eigenvalues": float("nan"),
        "n_negative_eigenvalues": -1,
        "neg_eigenvalue_fraction": float("nan"),
        "n_negative_predictions": -1,
        "stress": float("nan"),
        "triangle_violation_fraction": float("nan"),
        "triangle_mean_violation": float("nan"),
        "triangle_max_violation": float("nan"),
    }


# ===========================================================================
# Tree export helpers
# ===========================================================================

def format_path_conditions(threshold_tuples, feature_names):
    """Format Node.threshold tuples into an AND-separated path expression."""
    if not threshold_tuples:
        return "ROOT"

    pieces = []
    for col, side, threshold in threshold_tuples:
        col_idx = int(col)
        feat_name = (
            feature_names[col_idx]
            if 0 <= col_idx < len(feature_names)
            else f"feature_{col_idx}"
        )
        th = float(threshold)
        if side == "L":
            pieces.append(f"{feat_name} <= {th:.6f}")
        else:
            pieces.append(f"{feat_name} > {th:.6f}")
    return " AND ".join(pieces)


def extract_tree_rules(model, variant_name, feature_names):
    """Extract one row per split node with explicit split rules."""
    summary = model.summary()

    path_by_id = {}
    node_by_id = {}
    for path, node in getattr(model, "_nodes", {}).items():
        node_id = int(node.id)
        path_by_id[node_id] = path
        node_by_id[node_id] = node
    for path, node in getattr(model, "_leaves", {}).items():
        node_id = int(node.id)
        path_by_id[node_id] = path
        node_by_id[node_id] = node

    rows = []
    for node_id, info in sorted(summary.items(), key=lambda item: int(item[0])):
        if "col" not in info:
            continue

        split_col = int(info["col"])
        threshold = float(info["th"])
        children = info.get("children", ())
        left_child_id = int(children[0]) if len(children) >= 1 else -1
        right_child_id = int(children[1]) if len(children) >= 2 else -1
        feature_name = (
            feature_names[split_col]
            if 0 <= split_col < len(feature_names)
            else f"feature_{split_col}"
        )

        path = path_by_id.get(int(node_id), "")
        node_obj = node_by_id.get(int(node_id))
        path_rule = format_path_conditions(
            [] if node_obj is None else list(getattr(node_obj, "threshold", [])),
            feature_names,
        )

        rows.append(
            {
                "variant": variant_name,
                "node_id": int(node_id),
                "node_path": path,
                "depth": len(path),
                "split_feature_index": split_col,
                "split_feature_name": feature_name,
                "threshold": threshold,
                "left_child_node_id": left_child_id,
                "right_child_node_id": right_child_id,
                "n_samples": int(info.get("samples", -1)),
                "loss": float(info.get("loss", float("nan"))),
                "path_rule": path_rule,
                "split_rule": (
                    f"{feature_name} <= {threshold:.6f} -> {left_child_id}; "
                    f"> {threshold:.6f} -> {right_child_id}"
                ),
            }
        )

    return rows


def extract_leaf_coefficients(model, variant_name, feature_names):
    """Extract node and leaf linear model coefficients in long-form raw scale."""
    rows = []

    combined_nodes = []
    for node_path, node in getattr(model, "_nodes", {}).items():
        combined_nodes.append(("split", node_path, node))
    for leaf_path, leaf in getattr(model, "_leaves", {}).items():
        combined_nodes.append(("leaf", leaf_path, leaf))

    combined_nodes.sort(key=lambda item: int(item[2].id))

    for node_type, node_path, node in combined_nodes:
        node_id = int(node.id)
        node_model = getattr(node, "model", None)
        if node_model is None or not hasattr(node_model, "coef_"):
            continue

        coef = np.asarray(node_model.coef_, dtype=np.float64).reshape(-1)
        intercept = float(node_model.intercept_)
        path_rule = format_path_conditions(
            list(getattr(node, "threshold", [])), feature_names
        )
        n_samples = int(getattr(node, "n_samples", -1))

        for coef_idx, coef_value in enumerate(coef):
            feat_name = (
                feature_names[coef_idx]
                if 0 <= coef_idx < len(feature_names)
                else f"feature_{coef_idx}"
            )
            rows.append(
                {
                    "variant": variant_name,
                    "node_type": node_type,
                    "node_id": node_id,
                    "node_path": node_path,
                    "node_depth": len(node_path),
                    "n_samples": n_samples,
                    "path_rule": path_rule,
                    "intercept_raw": intercept,
                    "coef_index": int(coef_idx),
                    "feature_name": feat_name,
                    "coefficient_raw": float(coef_value),
                }
            )

    return rows


def count_tree_split_nodes(model):
    """Count internal split nodes in fitted tree."""
    summary = model.summary()
    return int(sum(1 for info in summary.values() if "col" in info))


def extract_leaf_weight_diagnostics(
    model,
    variant_name,
    feature_names,
    x_train,
    train_weights,
):
    """Export raw row counts and protein-equivalent mass for each node/leaf."""
    rows = []
    configured_min_split = float(
        getattr(model, "_weighted_min_samples_split", model.min_samples_split)
    )
    configured_min_leaf = float(
        getattr(model, "_weighted_min_samples_leaf", model.min_samples_leaf)
    )

    combined_nodes = []
    for node_path, node in getattr(model, "_nodes", {}).items():
        combined_nodes.append(("split", node_path, node))
    for leaf_path, leaf in getattr(model, "_leaves", {}).items():
        combined_nodes.append(("leaf", leaf_path, leaf))

    combined_nodes.sort(key=lambda item: int(item[2].id))

    for node_type, node_path, node in combined_nodes:
        mask = lineartree_classes._predict_branch(x_train, node.threshold)
        raw_rows = int(mask.sum())
        weight_mass = float(train_weights[mask].sum()) if raw_rows > 0 else 0.0
        rows.append(
            {
                "variant": variant_name,
                "node_type": node_type,
                "node_id": int(node.id),
                "node_path": node_path,
                "depth": len(node_path),
                "raw_row_count": raw_rows,
                "protein_equivalent_weight_mass": weight_mass,
                "path_rule": format_path_conditions(node.threshold, feature_names),
                "configured_weighted_min_samples_split": configured_min_split,
                "configured_weighted_min_samples_leaf": configured_min_leaf,
            }
        )

    return rows


# ===========================================================================
# Scatter sampling helpers
# ===========================================================================

def allocate_protein_equal_sample_counts(test_records, sample_size, rng):
    """Allocate sample counts approximately equally per protein."""
    if not test_records:
        return np.empty((0,), dtype=np.int64)

    capacities = np.array([record["n_pairs"] for record in test_records], dtype=np.int64)
    total_capacity = int(np.sum(capacities))
    target = min(int(sample_size), total_capacity)
    n_proteins = len(test_records)

    if target <= 0:
        return np.zeros((n_proteins,), dtype=np.int64)

    counts = np.full(n_proteins, target // n_proteins, dtype=np.int64)
    remainder = target % n_proteins
    if remainder > 0:
        bump = rng.permutation(n_proteins)[:remainder]
        counts[bump] += 1

    np.minimum(counts, capacities, out=counts)
    assigned = int(np.sum(counts))
    deficit = target - assigned

    while deficit > 0:
        eligible = np.flatnonzero(capacities > counts)
        if eligible.size == 0:
            break
        step = min(deficit, eligible.size)
        chosen = rng.choice(eligible, size=step, replace=False)
        counts[chosen] += 1
        deficit -= step

    return counts


def sample_test_rows_for_variant(variant_name, test_records, y_true, y_pred, sample_size, rng):
    """Sample test prediction rows with protein-equal allocation."""
    offsets = build_offsets(test_records)
    sample_counts = allocate_protein_equal_sample_counts(test_records, sample_size, rng)

    rows = []
    for idx, (record, (start, end)) in enumerate(zip(test_records, offsets)):
        take_n = int(sample_counts[idx])
        if take_n <= 0:
            continue

        n_pairs = int(record["n_pairs"])
        if take_n >= n_pairs:
            chosen_local = np.arange(n_pairs, dtype=np.int64)
        else:
            chosen_local = rng.choice(n_pairs, size=take_n, replace=False)

        seq_sep = record["seq_sep"]
        for local_idx in chosen_local:
            global_idx = start + int(local_idx)
            rows.append(
                {
                    "variant": variant_name,
                    "split": "test",
                    "protein_id": record["protein_id"],
                    "n_residues": int(record["n_residues"]),
                    "seq_sep": float(seq_sep[local_idx]),
                    "actual_distance": float(y_true[global_idx]),
                    "predicted_distance": float(y_pred[global_idx]),
                }
            )

    return rows


# ===========================================================================
# Variant execution
# ===========================================================================

def collect_geometry_rows(variant_name, split_name, split_records, y_true, y_pred):
    """Run per-protein geometry analysis for one split."""
    rows = []
    offsets = build_offsets(split_records)
    for record, (start, end) in zip(split_records, offsets):
        per_protein_r2 = compute_r2(y_true[start:end], y_pred[start:end])
        try:
            metrics = analyze_protein_geometry(
                record["n_residues"],
                record["coordinates"],
                record["pair_i"],
                record["pair_j"],
                y_pred[start:end],
            )
            rows.append(
                {
                    "variant": variant_name,
                    "split": split_name,
                    "protein_id": record["protein_id"],
                    "n_residues": record["n_residues"],
                    "n_pairs": record["n_pairs"],
                    "per_protein_r2": per_protein_r2,
                    **metrics,
                }
            )
        except Exception:
            rows.append(make_failed_geometry_row(variant_name, split_name, record, per_protein_r2))
    return rows


def run_variant(variant, train_records, test_records, variant_index):
    """Fit one variant and return summary + per-protein rows + diagnostics."""
    t0 = time.time()

    feature_names = get_feature_names(variant)
    split_idx = int(variant["split_feature_index"])

    x_train, y_train = build_split_matrix(train_records, variant)
    x_test, y_test = build_split_matrix(test_records, variant)
    test_offsets = build_offsets(test_records)

    fit_and_apply_standardization(x_train, x_test, n_scale_features=split_idx)
    if feature_names[split_idx] != "n_residues":
        raise ValueError(
            f"Expected split feature to be n_residues, got {feature_names[split_idx]}"
        )

    train_weights = compute_split_pair_weights(train_records, normalize_mean=False)

    if LinearTreeRegressor is None:
        raise RuntimeError(
            "lineartree is not available but linear-tree variants were requested. "
            f"Import error: {LINEARTREE_IMPORT_ERROR}"
        )

    model = LinearTreeRegressor(
        base_estimator=LinearRegression(copy_X=False),
        max_depth=int(variant["max_depth"]),
        min_samples_split=LINEAR_TREE_MIN_SAMPLES_SPLIT,
        min_samples_leaf=LINEAR_TREE_MIN_SAMPLES_LEAF,
        split_features=[split_idx],
        linear_features=list(range(split_idx)),
        n_jobs=1,
    )
    model._diagnostic_variant_name = variant["name"]
    model._diagnostic_feature_names = feature_names
    model._diagnostic_test_bundle = {
        "x": x_test,
        "y": y_test,
        "records": test_records,
        "offsets": test_offsets,
    }
    model._protein_aware_thresholds = True
    model._weighted_min_samples_split = float(LINEAR_TREE_MIN_SAMPLES_SPLIT)
    model._weighted_min_samples_leaf = float(LINEAR_TREE_MIN_SAMPLES_LEAF)
    model._protein_aware_max_thresholds = int(LINEAR_TREE_MAX_PROTEIN_THRESHOLDS)
    model.fit(x_train, y_train, sample_weight=train_weights)

    trained_leaf_count = float(len(getattr(model, "_leaves", {})))
    trained_split_count = float(count_tree_split_nodes(model))

    y_pred_train = np.asarray(model.predict(x_train), dtype=np.float64)
    y_pred_test = np.asarray(model.predict(x_test), dtype=np.float64)

    train_metrics = summarize_predictions(y_train, y_pred_train, train_records)
    test_metrics = summarize_predictions(y_test, y_pred_test, test_records)
    train_geometry_rows = collect_geometry_rows(
        variant["name"],
        "train",
        train_records,
        y_train,
        y_pred_train,
    )
    test_geometry_rows = collect_geometry_rows(
        variant["name"],
        "test",
        test_records,
        y_test,
        y_pred_test,
    )

    geometry_rows = train_geometry_rows + test_geometry_rows
    geometry_lookup = {
        row["split"]: row for row in summarize_geometry_rows(geometry_rows)
    }
    train_geom_summary = geometry_lookup.get("train", {})
    test_geom_summary = geometry_lookup.get("test", {})

    rules_rows = extract_tree_rules(model, variant["name"], feature_names)
    coef_rows = extract_leaf_coefficients(model, variant["name"], feature_names)
    leaf_weight_rows = extract_leaf_weight_diagnostics(
        model,
        variant["name"],
        feature_names,
        x_train,
        train_weights,
    )
    selected_root_summary, root_split_geometry_rows = evaluate_selected_root_split(
        model,
        variant["name"],
        model._diagnostic_test_bundle,
    )
    annotate_selected_root_threshold_row(variant["name"], selected_root_summary)

    rng = np.random.default_rng(RANDOM_STATE + variant_index)
    scatter_rows = sample_test_rows_for_variant(
        variant_name=variant["name"],
        test_records=test_records,
        y_true=y_test,
        y_pred=y_pred_test,
        sample_size=SCATTER_SAMPLE_PER_VARIANT,
        rng=rng,
    )

    summary_row = {
        "variant": variant["name"],
        "model_type": "linear_tree",
        "feature_mode": variant["feature_mode"],
        "feature_dim": variant["feature_dim"],
        "uses_protein_equal_weighting": True,
        "weight_normalization": "per_protein_sum_1",
        "threshold_policy": THRESHOLD_POLICY,
        "max_candidate_thresholds_per_node": LINEAR_TREE_MAX_PROTEIN_THRESHOLDS,
        "max_depth": int(variant["max_depth"]),
        "min_samples_split": LINEAR_TREE_MIN_SAMPLES_SPLIT,
        "min_samples_leaf": LINEAR_TREE_MIN_SAMPLES_LEAF,
        "split_feature_index": split_idx,
        "split_feature_name": feature_names[split_idx],
        "trained_split_count": trained_split_count,
        "trained_leaf_count": trained_leaf_count,
        "n_train_proteins": len(train_records),
        "n_test_proteins": len(test_records),
        "n_train_pairs": int(y_train.size),
        "n_test_pairs": int(y_test.size),
        "scatter_test_samples": int(len(scatter_rows)),
        "train_mean_per_protein_r2": train_metrics["mean_per_protein_r2"],
        "test_mean_per_protein_r2": test_metrics["mean_per_protein_r2"],
        "test_median_per_protein_r2": test_metrics["median_per_protein_r2"],
        "test_negative_r2_count": test_metrics["negative_r2_count"],
        "test_negative_r2_fraction": test_metrics["negative_r2_fraction"],
        "train_mean_tm_score": train_geom_summary.get("tm_score_mean", float("nan")),
        "test_mean_tm_score": test_geom_summary.get("tm_score_mean", float("nan")),
        "test_mean_stress": test_geom_summary.get("stress_mean", float("nan")),
        "test_mean_triangle_violation_fraction": test_geom_summary.get(
            "triangle_violation_fraction_mean",
            float("nan"),
        ),
        "selected_root_test_mean_per_protein_r2": (
            float("nan") if selected_root_summary is None
            else selected_root_summary["test_mean_per_protein_r2"]
        ),
        "selected_root_threshold": (
            float("nan") if selected_root_summary is None
            else selected_root_summary["threshold"]
        ),
        "selected_root_test_median_per_protein_r2": (
            float("nan") if selected_root_summary is None
            else selected_root_summary["test_median_per_protein_r2"]
        ),
        "selected_root_test_negative_r2_count": (
            "" if selected_root_summary is None
            else selected_root_summary["test_negative_r2_count"]
        ),
        "selected_root_test_negative_r2_fraction": (
            float("nan") if selected_root_summary is None
            else selected_root_summary["test_negative_r2_fraction"]
        ),
        "selected_root_test_mean_tm_score": (
            float("nan") if selected_root_summary is None
            else selected_root_summary["tm_score_mean"]
        ),
        "selected_root_test_mean_rmsd": (
            float("nan") if selected_root_summary is None
            else selected_root_summary["rmsd_mean"]
        ),
        "selected_root_test_mean_stress": (
            float("nan") if selected_root_summary is None
            else selected_root_summary["stress_mean"]
        ),
        "selected_root_test_mean_triangle_violation_fraction": (
            float("nan") if selected_root_summary is None
            else selected_root_summary["triangle_violation_fraction_mean"]
        ),
        "selected_root_test_mean_neg_eigenvalue_fraction": (
            float("nan") if selected_root_summary is None
            else selected_root_summary["neg_eigenvalue_fraction_mean"]
        ),
        "elapsed_seconds": float(time.time() - t0),
    }

    per_rows = []
    test_rows = []
    for row in train_metrics["per_rows"]:
        out = dict(row)
        out["variant"] = variant["name"]
        out["model_type"] = "linear_tree"
        out["feature_mode"] = variant["feature_mode"]
        out["split"] = "train"
        per_rows.append(out)

    for row in test_metrics["per_rows"]:
        out = dict(row)
        out["variant"] = variant["name"]
        out["model_type"] = "linear_tree"
        out["feature_mode"] = variant["feature_mode"]
        out["split"] = "test"
        per_rows.append(out)
        test_rows.append(out)

    del x_train, x_test, y_train, y_test, y_pred_train, y_pred_test

    return (
        summary_row,
        per_rows,
        test_rows,
        rules_rows,
        coef_rows,
        scatter_rows,
        geometry_rows,
        leaf_weight_rows,
        root_split_geometry_rows,
    )


# ===========================================================================
# CSV helpers
# ===========================================================================

def format_value(value):
    """Stable CSV float formatting."""
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (np.floating, float)):
        value = float(value)
        if math.isnan(value) or math.isinf(value):
            return "nan"
        return f"{value:.6f}"
    if isinstance(value, (np.integer, int)):
        return int(value)
    return value


def write_csv(path, fieldnames, rows):
    """Write CSV with stable field ordering."""
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: format_value(row.get(key, "")) for key in fieldnames})


def configure_threshold_search_output(path):
    """Create the threshold-search CSV and enable append logging."""
    global _THRESHOLD_SEARCH_CSV_PATH, _THRESHOLD_SEARCH_ROWS
    _THRESHOLD_SEARCH_CSV_PATH = path
    _THRESHOLD_SEARCH_ROWS = []
    write_csv(path, THRESHOLD_SEARCH_FIELDS, [])


# ===========================================================================
# Main
# ===========================================================================

def main():
    t_start = time.time()

    if LinearTreeRegressor is None:
        print("ERROR: lineartree package is required for linear-tree variants.")
        print(f"Import error: {LINEARTREE_IMPORT_ERROR}")
        print("Install in environment: pip install --upgrade linear-tree")
        sys.exit(1)

    _apply_lineartree_sklearn_compat_patch()
    _apply_lineartree_protein_threshold_patch()

    csv_dir = CSV_DIR.resolve()
    csv_dir.mkdir(parents=True, exist_ok=True)
    configure_threshold_search_output(csv_dir / THRESHOLD_SEARCH_CSV_NAME)

    print(f"{'=' * 72}")
    print("DIAGNOSTIC CONTEXT LINEAR TREE REGRESSION — layer 47")
    print(f"{'=' * 72}")
    print(f"Pair blocks root:     {PAIR_BLOCKS_ROOT}")
    print(f"PDB root:             {PDB_ROOT}")
    print(f"Output CSV dir:       {csv_dir}")
    print(f"Layer:                {LAYER}")
    print(f"Train/test split:     80/20 protein-level, seed={RANDOM_STATE}")
    print("Training weights:     protein-equal (per protein sums to 1.0)")
    print(
        "Tree policy:          max_depth=2, split only on raw n_residues, "
        f"weighted min_split={LINEAR_TREE_MIN_SAMPLES_SPLIT}, "
        f"weighted min_leaf={LINEAR_TREE_MIN_SAMPLES_LEAF}"
    )
    print(
        "Threshold policy:     "
        f"{THRESHOLD_POLICY} ({LINEAR_TREE_MAX_PROTEIN_THRESHOLDS} candidates per node)"
    )
    print("Split selection:      greedy weighted training loss only")
    print("Leakage policy:       no validation/test set used for split selection")
    print("Threshold diagnostics: training loss selects splits; test metrics are logged only")
    print("Scaling policy:       z-score all non-n_residues features")
    print(
        "Scatter sample:       test-only, protein-equal, up to "
        f"{SCATTER_SAMPLE_PER_VARIANT:,} rows per variant"
    )
    print(f"Variants:             {len(VARIANTS)}")

    all_proteins = discover_proteins(PAIR_BLOCKS_ROOT)
    print(f"\nDiscovered {len(all_proteins)} protein directories.")

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
            if n_ca < 2:
                skipped.append((protein_id, "fewer than 2 CA atoms"))
                print(f"[SKIP] {protein_id}: fewer than 2 CA atoms")
                continue

            pb0 = np.load(str(pb0_path), mmap_mode="r")
            pb_dim = int(pb0.shape[0])
            del pb0

            if n_ca != pb_dim:
                skipped.append((protein_id, f"CA count {n_ca} != pair block dim {pb_dim}"))
                print(f"[SKIP] {protein_id}: CA count {n_ca} != pair block dim {pb_dim}")
                continue

            pair_i, pair_j = np.tril_indices(n_ca, k=-1)
            if pair_i.size == 0:
                skipped.append((protein_id, "no lower-triangle residue pairs"))
                print(f"[SKIP] {protein_id}: no lower-triangle residue pairs")
                continue

            dist_matrix = compute_distance_matrix(ca_data["coordinates"])
            y_abs = np.asarray(dist_matrix[pair_i, pair_j], dtype=np.float32)
            seq_sep = (pair_i - pair_j).astype(np.float32, copy=False)

            protein_data.append(
                {
                    "protein_id": protein_id,
                    "protein_dir": protein_dir,
                    "n_residues": n_ca,
                    "pair_i": pair_i,
                    "pair_j": pair_j,
                    "n_pairs": int(y_abs.size),
                    "y_abs": y_abs,
                    "seq_sep": seq_sep,
                    "coordinates": ca_data["coordinates"],
                }
            )
            print(
                f"[OK]   {protein_id}: {n_ca} residues, {y_abs.size:,} lower-triangle pairs"
            )

        except Exception as exc:  # pragma: no cover - diagnostic path
            skipped.append((protein_id, f"unexpected error -- {exc}"))
            print(f"[SKIP] {protein_id}: unexpected error -- {exc}")
            traceback.print_exc()

    if not protein_data:
        print("\nERROR: No proteins loaded successfully. Exiting.")
        sys.exit(1)

    print(f"\nSuccessfully loaded {len(protein_data)} proteins (skipped {len(skipped)}).")

    train_records, test_records = split_global_records(protein_data)
    print(
        f"Global split: {len(train_records)} train / {len(test_records)} test proteins"
    )

    split_rows = []
    for record in train_records:
        split_rows.append(
            {
                "scope": "global_shared",
                "protein_id": record["protein_id"],
                "split": "train",
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
            }
        )
    for record in test_records:
        split_rows.append(
            {
                "scope": "global_shared",
                "protein_id": record["protein_id"],
                "split": "test",
                "n_residues": record["n_residues"],
                "n_pairs": record["n_pairs"],
            }
        )

    split_fields = ["scope", "protein_id", "split", "n_residues", "n_pairs"]
    write_csv(csv_dir / SPLIT_CSV_NAME, split_fields, split_rows)
    print(f"Saved split CSV -> {csv_dir / SPLIT_CSV_NAME}")

    summary_rows = []
    per_protein_rows = []
    tree_rule_rows = []
    tree_coef_rows = []
    scatter_sample_rows = []
    geometry_rows = []
    leaf_weight_rows = []
    root_split_geometry_rows = []
    ranking_lookup = {}

    for variant_index, variant in enumerate(VARIANTS):
        print(f"\n{'-' * 72}")
        print(
            f"Running variant: {variant['name']} "
            f"(feature_mode={variant['feature_mode']})"
        )
        (
            summary_row,
            variant_per_rows,
            variant_test_rows,
            rules_rows,
            coef_rows,
            scatter_rows,
            variant_geometry_rows,
            variant_leaf_weight_rows,
            variant_root_split_geometry_rows,
        ) = run_variant(
            variant,
            train_records,
            test_records,
            variant_index=variant_index,
        )
        summary_rows.append(summary_row)
        per_protein_rows.extend(variant_per_rows)
        tree_rule_rows.extend(rules_rows)
        tree_coef_rows.extend(coef_rows)
        scatter_sample_rows.extend(scatter_rows)
        geometry_rows.extend(variant_geometry_rows)
        leaf_weight_rows.extend(variant_leaf_weight_rows)
        root_split_geometry_rows.extend(variant_root_split_geometry_rows)

        rankable = [
            row for row in variant_test_rows if not math.isnan(float(row["per_protein_r2"]))
        ]
        ranking_lookup[variant["name"]] = sorted(
            rankable, key=lambda row: float(row["per_protein_r2"])
        )
        print(
            f"  test mean per-protein R2={summary_row['test_mean_per_protein_r2']:.4f}, "
            f"test median={summary_row['test_median_per_protein_r2']:.4f}, "
            f"root_split_test_mean={summary_row['selected_root_test_mean_per_protein_r2']:.4f}, "
            f"TM_test={summary_row['test_mean_tm_score']:.4f}, "
            f"stress_test={summary_row['test_mean_stress']:.4f}, "
            f"tri_viol_test={summary_row['test_mean_triangle_violation_fraction']:.6f}, "
            f"splits={int(summary_row['trained_split_count'])}, "
            f"leaves={int(summary_row['trained_leaf_count'])}, "
            f"elapsed={summary_row['elapsed_seconds']:.1f}s"
        )

    geometry_summary_rows = summarize_geometry_rows(geometry_rows)
    _sync_threshold_search_output()

    summary_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "feature_dim",
        "uses_protein_equal_weighting",
        "weight_normalization",
        "threshold_policy",
        "max_candidate_thresholds_per_node",
        "max_depth",
        "min_samples_split",
        "min_samples_leaf",
        "split_feature_index",
        "split_feature_name",
        "trained_split_count",
        "trained_leaf_count",
        "n_train_proteins",
        "n_test_proteins",
        "n_train_pairs",
        "n_test_pairs",
        "scatter_test_samples",
        "train_mean_per_protein_r2",
        "test_mean_per_protein_r2",
        "test_median_per_protein_r2",
        "test_negative_r2_count",
        "test_negative_r2_fraction",
        "selected_root_test_mean_per_protein_r2",
        "selected_root_threshold",
        "selected_root_test_median_per_protein_r2",
        "selected_root_test_negative_r2_count",
        "selected_root_test_negative_r2_fraction",
        "train_mean_tm_score",
        "test_mean_tm_score",
        "selected_root_test_mean_tm_score",
        "selected_root_test_mean_rmsd",
        "test_mean_stress",
        "selected_root_test_mean_stress",
        "test_mean_triangle_violation_fraction",
        "selected_root_test_mean_triangle_violation_fraction",
        "selected_root_test_mean_neg_eigenvalue_fraction",
        "elapsed_seconds",
    ]
    write_csv(csv_dir / SUMMARY_CSV_NAME, summary_fields, summary_rows)
    print(f"\nSaved summary CSV -> {csv_dir / SUMMARY_CSV_NAME}")

    per_protein_fields = [
        "variant",
        "model_type",
        "feature_mode",
        "split",
        "protein_id",
        "n_residues",
        "n_pairs",
        "per_protein_r2",
        "mean_true_distance",
        "std_true_distance",
        "mean_predicted_distance",
        "std_predicted_distance",
        "mean_error_pred_minus_true",
    ]
    write_csv(csv_dir / PER_PROTEIN_CSV_NAME, per_protein_fields, per_protein_rows)
    print(f"Saved per-protein CSV -> {csv_dir / PER_PROTEIN_CSV_NAME}")

    rules_fields = [
        "variant",
        "node_id",
        "node_path",
        "depth",
        "split_feature_index",
        "split_feature_name",
        "threshold",
        "left_child_node_id",
        "right_child_node_id",
        "n_samples",
        "loss",
        "path_rule",
        "split_rule",
    ]
    write_csv(csv_dir / TREE_RULES_CSV_NAME, rules_fields, tree_rule_rows)
    print(f"Saved tree rules CSV -> {csv_dir / TREE_RULES_CSV_NAME}")

    coef_fields = [
        "variant",
        "node_type",
        "node_id",
        "node_path",
        "node_depth",
        "n_samples",
        "path_rule",
        "intercept_raw",
        "coef_index",
        "feature_name",
        "coefficient_raw",
    ]
    write_csv(csv_dir / TREE_COEF_CSV_NAME, coef_fields, tree_coef_rows)
    print(f"Saved node/leaf coefficients CSV -> {csv_dir / TREE_COEF_CSV_NAME}")

    print(f"Saved threshold search CSV -> {csv_dir / THRESHOLD_SEARCH_CSV_NAME}")

    scatter_fields = [
        "variant",
        "split",
        "protein_id",
        "n_residues",
        "seq_sep",
        "actual_distance",
        "predicted_distance",
    ]
    write_csv(csv_dir / SCATTER_SAMPLE_CSV_NAME, scatter_fields, scatter_sample_rows)
    print(f"Saved scatter sample CSV -> {csv_dir / SCATTER_SAMPLE_CSV_NAME}")

    geometry_fields = [
        "variant",
        "split",
        "protein_id",
        "n_residues",
        "n_pairs",
        "per_protein_r2",
        "rmsd",
        "tm_score",
        "frac_top3_eigenvalues",
        "n_negative_eigenvalues",
        "neg_eigenvalue_fraction",
        "n_negative_predictions",
        "stress",
        "triangle_violation_fraction",
        "triangle_mean_violation",
        "triangle_max_violation",
    ]
    write_csv(csv_dir / GEOMETRY_METRICS_CSV_NAME, geometry_fields, geometry_rows)
    print(f"Saved geometry metrics CSV -> {csv_dir / GEOMETRY_METRICS_CSV_NAME}")

    root_split_geometry_fields = [
        "variant",
        "split",
        "protein_id",
        "n_residues",
        "n_pairs",
        "per_protein_r2",
        "rmsd",
        "tm_score",
        "frac_top3_eigenvalues",
        "n_negative_eigenvalues",
        "neg_eigenvalue_fraction",
        "n_negative_predictions",
        "stress",
        "triangle_violation_fraction",
        "triangle_mean_violation",
        "triangle_max_violation",
    ]
    write_csv(
        csv_dir / ROOT_SPLIT_GEOMETRY_CSV_NAME,
        root_split_geometry_fields,
        root_split_geometry_rows,
    )
    print(f"Saved root split geometry CSV -> {csv_dir / ROOT_SPLIT_GEOMETRY_CSV_NAME}")

    geometry_summary_fields = ["variant", "split", "n_proteins"]
    geometry_metrics = [
        "per_protein_r2",
        "tm_score",
        "rmsd",
        "stress",
        "frac_top3_eigenvalues",
        "n_negative_eigenvalues",
        "neg_eigenvalue_fraction",
        "n_negative_predictions",
        "triangle_violation_fraction",
        "triangle_mean_violation",
        "triangle_max_violation",
    ]
    for metric_key in geometry_metrics:
        geometry_summary_fields.extend([f"{metric_key}_mean", f"{metric_key}_std"])
    write_csv(
        csv_dir / GEOMETRY_SUMMARY_CSV_NAME,
        geometry_summary_fields,
        geometry_summary_rows,
    )
    print(f"Saved geometry summary CSV -> {csv_dir / GEOMETRY_SUMMARY_CSV_NAME}")

    leaf_weight_fields = [
        "variant",
        "node_type",
        "node_id",
        "node_path",
        "depth",
        "raw_row_count",
        "protein_equivalent_weight_mass",
        "path_rule",
        "configured_weighted_min_samples_split",
        "configured_weighted_min_samples_leaf",
    ]
    write_csv(
        csv_dir / LEAF_WEIGHT_DIAGNOSTICS_CSV_NAME,
        leaf_weight_fields,
        leaf_weight_rows,
    )
    print(
        "Saved leaf weight diagnostics CSV -> "
        f"{csv_dir / LEAF_WEIGHT_DIAGNOSTICS_CSV_NAME}"
    )

    summary_path = csv_dir / SUMMARY_TXT_NAME
    with summary_path.open("w") as handle:
        handle.write("Diagnostic Context Linear Tree Regression — Layer 47\n")
        handle.write("=" * 64 + "\n\n")
        handle.write(f"Pair blocks root:   {PAIR_BLOCKS_ROOT}\n")
        handle.write(f"PDB root:           {PDB_ROOT}\n")
        handle.write(f"Output CSV dir:     {csv_dir}\n")
        handle.write(f"Layer:              {LAYER}\n")
        handle.write("Representation:     one lower-triangle row per unordered pair\n")
        handle.write("Training weighting: protein-equal (each protein sums to 1.0)\n")
        handle.write("Model family:       linear-tree only\n")
        handle.write(
            "Tree policy:        max_depth=2, split only on raw n_residues, "
            f"weighted min_split={LINEAR_TREE_MIN_SAMPLES_SPLIT}, "
            f"weighted min_leaf={LINEAR_TREE_MIN_SAMPLES_LEAF}\n"
        )
        handle.write(
            "Threshold policy:   "
            f"{THRESHOLD_POLICY} ({LINEAR_TREE_MAX_PROTEIN_THRESHOLDS} candidates per node)\n"
        )
        handle.write("Split selection:    greedy weighted training loss only\n")
        handle.write("Leakage policy:     no validation/test set used for split selection\n")
        handle.write("Threshold diagnostics: candidate test metrics are logged only for analysis\n")
        handle.write("Scaling policy:     z-score all non-n_residues features\n")
        handle.write(
            "Scatter sampling:   test-only, protein-equal, up to "
            f"{SCATTER_SAMPLE_PER_VARIANT:,} rows per variant\n"
        )
        handle.write(
            f"Train/test split:   protein-level 80/20, seed={RANDOM_STATE}\n\n"
        )
        handle.write(f"Proteins discovered: {len(all_proteins)}\n")
        handle.write(f"Proteins loaded:     {len(protein_data)}\n")
        handle.write(f"Proteins skipped:    {len(skipped)}\n")
        handle.write(f"Train proteins:      {len(train_records)}\n")
        handle.write(f"Test proteins:       {len(test_records)}\n\n")

        handle.write("Variant comparison:\n")
        handle.write(
            "  Variant"
            + " " * 39
            + "FeatDim  Splits  Leaves  TestMeanPP_R2  TestTM     Stress   TriViol\n"
        )
        for row in summary_rows:
            handle.write(
                f"  {row['variant']:<54}"
                f"{int(row['feature_dim']):>7d}"
                f"{int(row['trained_split_count']):>8d}"
                f"{int(row['trained_leaf_count']):>8d}"
                f"{row['test_mean_per_protein_r2']:>15.4f}"
                f"{row['test_mean_tm_score']:>10.4f}"
                f"{row['test_mean_stress']:>10.4f}"
                f"{row['test_mean_triangle_violation_fraction']:>10.6f}\n"
            )

        best_row = max(
            summary_rows,
            key=lambda row: float(row["test_mean_per_protein_r2"]),
        )
        handle.write(
            "\nBest variant by test mean per-protein R2: "
            f"{best_row['variant']} ({best_row['test_mean_per_protein_r2']:.4f})\n"
        )

        for variant_name, ranked_rows in ranking_lookup.items():
            handle.write(f"\nTop 10 worst test proteins for {variant_name}:\n")
            for row in ranked_rows[:10]:
                handle.write(
                    f"  {row['protein_id']}: R2={row['per_protein_r2']:.4f}, "
                    f"n_res={row['n_residues']}, "
                    f"mean(pred-true)={row['mean_error_pred_minus_true']:+.4f}\n"
                )

            handle.write(f"Top 10 best test proteins for {variant_name}:\n")
            for row in reversed(ranked_rows[-10:]):
                handle.write(
                    f"  {row['protein_id']}: R2={row['per_protein_r2']:.4f}, "
                    f"n_res={row['n_residues']}, "
                    f"mean(pred-true)={row['mean_error_pred_minus_true']:+.4f}\n"
                )

            geom_test = next(
                (
                    row for row in geometry_summary_rows
                    if row["variant"] == variant_name and row["split"] == "test"
                ),
                None,
            )
            if geom_test is not None:
                handle.write(
                    "  Test geometry summary: "
                    f"TM={geom_test['tm_score_mean']:.4f}, "
                    f"RMSD={geom_test['rmsd_mean']:.4f}, "
                    f"Stress={geom_test['stress_mean']:.4f}, "
                    f"TriViol={geom_test['triangle_violation_fraction_mean']:.6f}, "
                    f"NegEigFrac={geom_test['neg_eigenvalue_fraction_mean']:.6f}\n"
                )
            handle.write(
                "  Selected root split: "
                f"meanPP_R2={next(row for row in summary_rows if row['variant'] == variant_name)['selected_root_test_mean_per_protein_r2']:.4f}, "
                f"medianPP_R2={next(row for row in summary_rows if row['variant'] == variant_name)['selected_root_test_median_per_protein_r2']:.4f}, "
                f"negFrac={next(row for row in summary_rows if row['variant'] == variant_name)['selected_root_test_negative_r2_fraction']:.4f}, "
                f"TM={next(row for row in summary_rows if row['variant'] == variant_name)['selected_root_test_mean_tm_score']:.4f}, "
                f"Stress={next(row for row in summary_rows if row['variant'] == variant_name)['selected_root_test_mean_stress']:.4f}\n"
            )

        handle.write("\nGeometry summary by variant/split:\n")
        for row in geometry_summary_rows:
            handle.write(
                f"  {row['variant']} ({row['split']}): "
                f"TM={row['tm_score_mean']:.4f}, "
                f"RMSD={row['rmsd_mean']:.4f}, "
                f"Stress={row['stress_mean']:.4f}, "
                f"TriViol={row['triangle_violation_fraction_mean']:.6f}, "
                f"NegEigFrac={row['neg_eigenvalue_fraction_mean']:.6f}\n"
            )

        if skipped:
            handle.write("\nSkipped proteins:\n")
            for protein_id, reason in skipped:
                handle.write(f"  {protein_id}: {reason}\n")

        handle.write(
            "\nNote: diagnostic_lin_tree_context_reg_geometry_metrics.csv includes "
            "both train and test proteins with per-protein predictive and "
            "geometry diagnostics.\n"
        )
        handle.write(
            "Note: diagnostic_lin_tree_context_reg_leaf_weight_diagnostics.csv "
            "reports raw row counts and protein-equivalent training mass for "
            "every node and leaf.\n"
        )
        handle.write(
            "Note: diagnostic_lin_tree_context_reg_threshold_search.csv records "
            "every tested threshold with training losses plus diagnostic test-set "
            "mean per-protein R2 for the relevant scope.\n"
        )
        handle.write(
            "Note: diagnostic_lin_tree_context_reg_root_split_geometry_metrics.csv "
            "stores per-protein test geometry metrics for the selected root split.\n"
        )

    print(f"Saved summary text -> {summary_path}")

    elapsed = time.time() - t_start
    print(f"\n{'=' * 72}")
    print("DIAGNOSTIC CONTEXT LINEAR TREE RUN COMPLETE")
    print(f"Elapsed time: {elapsed:.1f} seconds ({elapsed / 60.0:.1f} minutes)")
    print(f"{'=' * 72}")


if __name__ == "__main__":
    main()

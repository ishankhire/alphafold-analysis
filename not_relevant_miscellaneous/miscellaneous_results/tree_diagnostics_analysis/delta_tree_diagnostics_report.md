# Delta Tree Diagnostics Report

## 1. What each CSV means

These artifacts all come from the same layer-47 lower-triangle probe setup.
Every decision-tree / XGBoost row is one unordered residue pair `(i, j)` with `i > j`.
The feature vector is 134-dimensional:
- 128 pair channels: `pair_ch_0, pair_ch_1, pair_ch_2, pair_ch_3, pair_ch_4, pair_ch_5, ...`
- 1 protein-size scalar: `n_residues`
- 2 raw/continuous sequence-separation scalars: `seq_sep`, `log1p_seq_sep`
- 3 sequence-separation indicator scalars: `is_seq_sep_1`, `is_seq_sep_le_2`, `is_seq_sep_le_4`

Training uses a protein-level split, not a pair-level split:
- 184 proteins loaded
- 147 train proteins / 37 test proteins
- inside the 147-train set, 117 fit proteins / 30 validation proteins for model selection
- training loss is protein-equal because each protein gets total sample weight 1, even though proteins contribute different numbers of residue pairs
- train proteins are capped at 5,000 sampled pairs each; validation and test use all available pairs

That matters for interpretability: importance values describe what the model used under this exact split, weighting rule, and sampling cap. They do not prove a feature is biologically causal or universally necessary.

### `diagnostic_decision_tree_variant_summary.csv`
- One row for the selected final decision-tree model after validation search.
- `best_validation_mean_per_protein_r2`: score used to pick hyperparameters; this is the mean of per-protein `R^2` values on the validation proteins.
- `train_mean_per_protein_r2` / `test_mean_per_protein_r2`: average per-protein `R^2` on sampled train rows / full test rows.
- `train_pair_weighted_r2` / `test_pair_weighted_r2`: pooled `R^2` over all rows in that split, so big proteins matter more than in mean-per-protein `R^2`.
- `test_negative_r2_fraction`: fraction of proteins where predictions are worse than predicting that protein's mean distance.
- `tree_depth` and `n_leaves`: realized complexity of the final fitted tree.

### `diagnostic_decision_tree_search.csv`
- One row per tested depth / minimum leaf / minimum split configuration.
- `fit_mean_per_protein_r2` is in-sample on the fit proteins used to train that config.
- `validation_mean_per_protein_r2` is the selection metric; if this rises while fit performance rises much faster, that is the usual sign of overfitting.

### `diagnostic_decision_tree_feature_importance.csv`
- This is `DecisionTreeRegressor.feature_importances_` from scikit-learn.
- It is the normalized total weighted impurity decrease attributed to each feature across the final tree.
- Values sum to about 1 across all 134 features.
- A large value means the feature caused large variance reductions high in the tree or across many heavy branches.
- A zero means the final tree never split on that feature.
- This metric is model-internal and relative: it says what the tree used, not whether dropping the feature leaves held-out performance unchanged.

### `diagnostic_decision_tree_nodes.csv`
- One row per fitted node in the final decision tree.
- `feature_name` and `threshold` define the split at internal nodes.
- `weighted_n_node_samples` is the total protein-equal sample mass reaching that node.
- `value` is the node's predicted distance.
- This file is the most concrete interpretability artifact because it shows where a feature appears, how deep it appears, and how much weighted data it controls.

### `diagnostic_decision_tree_tree.txt`
- Text dump of the final decision tree from `sklearn.tree.export_text`.
- Good for quick inspection, but hard to read once the tree is deep.

### `diagnostic_decision_tree_seq_sep_diagnostics.csv`
- Buckets pairs by sequence separation (`1`, `2`, `3-4`, `5-8`, `9+`, plus exact `seq_sep_eq_1_exact`).
- `r2` is computed inside each bucket only.
- Negative `R^2` in tiny-distance buckets can coexist with small absolute error because the true-distance variance is tiny there.
- `fraction_pred_lt_3A` helps check whether the model is willing to make very short-distance predictions.

### `diagnostic_xgboost_variant_summary.csv`
- Same role as the decision-tree summary, but for the final XGBoost ensemble.
- `best_iteration_search` is the early-stopped boosting round from the fit/validation search.
- `final_n_estimators` is the number of boosting rounds used when refitting on the full sampled train split.
- `n_boosted_rounds` should match the actual number of dumped trees in the final ensemble.

### `diagnostic_xgboost_search.csv`
- One row per tested XGBoost hyperparameter configuration.
- `best_iteration` and `best_score_rmse` come from validation early stopping.
- `validation_mean_per_protein_r2` remains the model-selection target.

### `diagnostic_xgboost_nodes.csv`
- One row per node in every boosted tree.
- `gain` is the loss reduction at that split node.
- `cover` is the weighted amount of training data reaching that split node.
- Summing `gain` over all split nodes in one tree gives a useful tree-level proxy for how much that tree contributed to training loss reduction.

### `diagnostic_xgboost_root_splits.csv`
- One row per boosted tree root split.
- Helpful for checking which features repeatedly control the first decision in each tree.

### `diagnostic_xgboost_top_split_features.csv`
- Derived by parsing all split nodes in the dumped trees.
- `split_count` is how often a feature is used anywhere in the ensemble.
- `total_gain` is the sum of split gains over all those uses.
- This is different from XGBoost's built-in `gain` importance, which is an average gain per split.

### `diagnostic_xgboost_feature_importance_gain.csv`
- `booster.get_score(importance_type='gain')`.
- This is average gain per split for each feature, not total gain.
- Good default importance metric when asking which features tend to make valuable splits.

### `diagnostic_xgboost_feature_importance_weight.csv`
- `booster.get_score(importance_type='weight')`.
- Pure split frequency: how many times a feature was used.
- High weight with modest gain usually means a feature is useful as a routing helper but not a dominant driver of performance.

### `diagnostic_xgboost_feature_importance_cover.csv`
- `booster.get_score(importance_type='cover')`.
- Average covered sample mass for splits using that feature.
- This says where the feature tends to appear in the tree structure, not how much predictive value it adds.

### `diagnostic_linear_tree_variant_summary.csv`
- Summary of the separate linear-tree experiment.
- Important here because it gives indirect ablation evidence for sequence-separation features in a related but not identical model family.

### `diagnostic_linear_tree_rules.csv`
- Learned size-based split rules for each linear-tree variant.
- For `lower_triangle_only_linear_tree`, the learned size buckets are the exact categories used below for coefficient/importance overlap.

### `diagnostic_linear_tree_leaf_coefficients_raw.csv`
- Long-form raw coefficients for each linear-model leaf.
- In `lower_triangle_only_linear_tree`, features are `pb_ij_0` ... `pb_ij_127`, which map directly to `pair_ch_0` ... `pair_ch_127` in the tree/XGBoost runs.
- Taking absolute coefficient magnitude is a reasonable ranking heuristic here, but it is still not a formal significance test because correlated channels can trade off with one another.

## 2. Headline performance

| Model | Test mean per-protein R2 | Test pair-weighted R2 | Train mean per-protein R2 | Train/test gap |
| --- | --- | --- | --- | --- |
| Decision tree | 0.9204 | 0.9279 | 0.9509 | 0.0305 |
| XGBoost | 0.9612 | 0.9684 | 0.9842 | 0.0230 |

Interpretation:
- These are not random-luck training scores; both models are being judged on 37 held-out proteins.
- XGBoost clearly generalizes better than the single tree on this split.
- The small train/test gap for XGBoost and the zero negative-`R^2` fraction argue against the result being a fluke caused only by a few easy proteins.
- Interpretability is still limited because this is one random protein split, one training-pair cap, and one set of correlated pair channels.

## 3. How important were `n_residues` and sequence-separation features?

| Model | Feature | Primary importance | Primary rank | Split count | Earliest depth | Classification |
| --- | --- | --- | --- | --- | --- | --- |
| Decision tree | n_residues | 0.013518 | 6 | 20 | 4 | used_nontrivial |
| Decision tree | seq_sep | 0.001232 | 19 | 13 | 6 | used_nontrivial |
| Decision tree | log1p_seq_sep | 0.000353 | 29 | 4 | 9 | used_rare_low_effect |
| Decision tree | is_seq_sep_1 | 0.000000 | 116 | 0 | NA | unused |
| Decision tree | is_seq_sep_le_2 | 0.000000 | 117 | 0 | NA | unused |
| Decision tree | is_seq_sep_le_4 | 0.000000 | 118 | 0 | NA | unused |
| XGBoost (gain) | n_residues | 11735.82 | 21 | 1239 | 0 | used_nontrivial |
| XGBoost (gain) | seq_sep | 11018.75 | 22 | 589 | 1 | used_nontrivial |
| XGBoost (gain) | log1p_seq_sep | 15818.00 | 16 | 180 | 1 | used_nontrivial |
| XGBoost (gain) | is_seq_sep_1 | 0.00 | 134 | 0 | NA | unused |
| XGBoost (gain) | is_seq_sep_le_2 | 0.00 | 134 | 0 | NA | unused |
| XGBoost (gain) | is_seq_sep_le_4 | 197.55 | 132 | 2 | 5 | used_rare_low_effect |

Removal verdict for `n_residues`: **uncertain without ablation**.
Removal verdict for sequence-separation features as a group: **probably similar but not identical**.

What this means in practice:
- In the single decision tree, `n_residues` is real but secondary: importance `0.013518`, rank 6. `seq_sep` and `log1p_seq_sep` are much smaller, and the three bin indicators are unused.
- In XGBoost, `n_residues` is used often (`weight=1239`) but with moderate average gain relative to the dominant pair channels. That pattern means it helps routing, but it is not the main source of predictive power.
- `seq_sep` and `log1p_seq_sep` are also used, but their gains are far below the top pair channels. `n_residues` does appear as a root split in 12 boosted trees, but the strongest early trees are still pair-channel driven.
- Indirect ablation evidence from the linear-tree experiment is weakly against large sequence-separation gains: `both_triangles_linear_tree` test mean per-protein `R^2` is 0.8487, `both_triangles_plus_seqdist_linear_tree` is 0.8493, and `both_triangles_plus_seqdist_plus_bins_linear_tree` is 0.8492. That is a tiny delta.
- So the best reading is: these scalars are not useless, but the pair channels dominate. Removing them would probably not destroy performance, yet you should not expect *exactly* the same result without running a true ablation.

Sequence-bucket behavior supports that interpretation:
| Model | Bucket | Test R2 | Mean pred-true |
| --- | --- | --- | --- |
| Decision tree | seq_sep=1 | -0.7172 | -0.0008 |
| XGBoost | seq_sep=1 | -52.8399 | +0.0291 |
| Decision tree | seq_sep>=9 | 0.9209 | +0.1550 |
| XGBoost | seq_sep>=9 | 0.9653 | +0.0431 |

Short-range buckets remain hard even when sequence-separation features are present, which again suggests those features are not a magic driver of the overall score.

## 4. Making sense of the tree

The single decision tree is hard to read because the selected model has depth 10 and 805 leaves. The useful way to read it is not as one giant rule list, but as a hierarchy:
- start with the top few splits, because those move the most weighted sample mass
- then inspect which heavy branches later introduce size or sequence-separation conditions
- finally treat deep leaves as local refinements, not global explanations

The top-level visualization is saved as `decision_tree_top_levels.png` and intentionally truncates at depth 3 so the structure is readable.

Highest-coverage decision-tree leaves:
| Leaf node | Depth | Weighted sample fraction | Prediction | Path |
| --- | --- | --- | --- | --- |
| 124 | 10 | 3.6% | 24.32 | pair_ch_23 <= -30.244 AND pair_ch_64 <= 10.889 AND pair_ch_73 <= 4.940 AND pair_ch_111 > -6.125 AND pair_ch_113 <= 7.808 AND pair_ch_58 > -25.344 AND pair_ch_15 <= 11.463 AND pair_ch_40 <= -6.755 AND pair_ch_58 > 48.643 AND pair_ch_4 > -2.295 |
| 1307 | 10 | 1.5% | 19.42 | pair_ch_23 > -30.244 AND pair_ch_58 > -13.130 AND pair_ch_80 <= 83.153 AND pair_ch_73 <= 16.629 AND pair_ch_15 <= 1.885 AND pair_ch_43 > 3.326 AND pair_ch_40 > -26.730 AND pair_ch_113 > -7.841 AND pair_ch_62 > -3.174 AND pair_ch_23 > -28.459 |
| 139 | 10 | 1.4% | 21.64 | pair_ch_23 <= -30.244 AND pair_ch_64 <= 10.889 AND pair_ch_73 <= 4.940 AND pair_ch_111 > -6.125 AND pair_ch_113 <= 7.808 AND pair_ch_58 > -25.344 AND pair_ch_15 > 11.463 AND pair_ch_113 <= -0.720 AND pair_ch_23 > -49.543 AND pair_ch_58 > 49.553 |
| 146 | 10 | 1.1% | 22.71 | pair_ch_23 <= -30.244 AND pair_ch_64 <= 10.889 AND pair_ch_73 <= 4.940 AND pair_ch_111 > -6.125 AND pair_ch_113 <= 7.808 AND pair_ch_58 > -25.344 AND pair_ch_15 > 11.463 AND pair_ch_113 > -0.720 AND pair_ch_58 > 44.274 AND pair_ch_23 > -47.829 |
| 145 | 10 | 1.0% | 23.77 | pair_ch_23 <= -30.244 AND pair_ch_64 <= 10.889 AND pair_ch_73 <= 4.940 AND pair_ch_111 > -6.125 AND pair_ch_113 <= 7.808 AND pair_ch_58 > -25.344 AND pair_ch_15 > 11.463 AND pair_ch_113 > -0.720 AND pair_ch_58 > 44.274 AND pair_ch_23 <= -47.829 |
| 1275 | 10 | 1.0% | 18.83 | pair_ch_23 > -30.244 AND pair_ch_58 > -13.130 AND pair_ch_80 <= 83.153 AND pair_ch_73 <= 16.629 AND pair_ch_15 <= 1.885 AND pair_ch_43 <= 3.326 AND pair_ch_69 <= -1.548 AND pair_ch_80 > 27.046 AND pair_ch_80 > 41.805 AND pair_ch_58 > 11.985 |
| 1425 | 10 | 1.0% | 15.86 | pair_ch_23 > -30.244 AND pair_ch_58 > -13.130 AND pair_ch_80 > 83.153 AND pair_ch_107 <= 5.287 AND pair_ch_114 <= -3.448 AND pair_ch_30 > -1.481 AND pair_ch_64 <= 1.702 AND pair_ch_1 > -8.879 AND pair_ch_14 <= 8.190 AND pair_ch_22 <= 6.685 |
| 1303 | 10 | 1.0% | 20.56 | pair_ch_23 > -30.244 AND pair_ch_58 > -13.130 AND pair_ch_80 <= 83.153 AND pair_ch_73 <= 16.629 AND pair_ch_15 <= 1.885 AND pair_ch_43 > 3.326 AND pair_ch_40 > -26.730 AND pair_ch_113 > -7.841 AND pair_ch_62 <= -3.174 AND pair_ch_73 <= -1.493 |

Interpretation:
- The root and early heavy branches are dominated by pair channels, not scalar features.
- `n_residues` and `seq_sep` mostly appear deeper, where the model is carving up special cases rather than defining the main global structure.

Best XGBoost trees by total split gain (training-loss reduction inside the final ensemble):
| Tree rank | Tree index | Total gain | Max depth | Root feature | Root gain |
| --- | --- | --- | --- | --- | --- |
| 1 | 0 | 124447110 | 6 | pair_ch_23 | 61475772 |
| 2 | 1 | 112282628 | 6 | pair_ch_64 | 49858472 |
| 3 | 2 | 102026244 | 6 | pair_ch_23 | 50308640 |
| 4 | 3 | 91161863 | 6 | pair_ch_23 | 45416836 |
| 5 | 4 | 81025580 | 6 | pair_ch_73 | 37884104 |
| 6 | 5 | 73660500 | 6 | pair_ch_73 | 34129464 |
| 7 | 6 | 69406462 | 6 | pair_ch_23 | 34167320 |
| 8 | 7 | 63226088 | 6 | pair_ch_23 | 30805438 |

Caveat: these are the trees that reduced training objective the most. XGBoost does not expose a clean per-tree held-out score, so this is the best available notion of a 'best tree' from the dumped ensemble.

Did any depth-4 trees behave fine?

| Model family | Depth-4 validation mean per-protein R2 | Best family validation R2 | Delta |
| --- | --- | --- | --- |
| Decision tree | 0.8650 | 0.9395 | -0.0746 |
| XGBoost | 0.9640 | 0.9705 | -0.0065 |

Interpretation:
- A depth-4 **single decision tree** is noticeably worse than the selected depth-10 tree, so in that family depth 4 does *not* look sufficient.
- A depth-4 **XGBoost ensemble** is much closer to the best XGBoost setup, so yes: shallow boosted trees behaved fine.

## 5. Would more training pairs per protein help?

- The current run cannot answer that exactly because the cap never changes. There is no counterfactual artifact with 10,000 or full-train pairs.
- The cap is active for 133/147 train proteins (90.5%). So this question is real; the cap is binding for most proteins.
- Train proteins have median 27261 lower-triangle pairs before capping, far above 5,000 for many proteins.
- XGBoost already has only a modest train/test gap (0.0230), so there is no obvious sign that it is starved for data in a way that would guarantee a large win from more sampled pairs.
- The single tree also does not look variance-limited in the usual way; its main limitation is representational simplicity relative to XGBoost.
- So the responsible answer is: you can make an informed guess that more pairs might help a bit, especially for large proteins, but a real answer requires rerunning with a cap sweep.

## 6. Linear-tree vs tree/XGBoost feature overlap

Here the comparison is intentionally restricted to `lower_triangle_only_linear_tree`, because only that variant lives in the same 128-channel feature space as the decision-tree/XGBoost runs.
The learned size buckets from that model are:
- `<= 252`
- `253-374`
- `375-507`
- `> 507`

Within each bucket, pair channels are ranked by absolute linear coefficient magnitude. Those rankings are then compared with the decision-tree importance rank and the XGBoost gain / weight / cover ranks over the same 128 channels.

Most significant top-K overlaps after FDR correction:
| Bucket | Comparison | Top-K | Overlap | Spearman rho | FDR-adjusted p |
| --- | --- | --- | --- | --- | --- |
| <= 252 | xgboost_weight | 20 | 12 | 0.276 | 1.869e-05 |
| <= 252 | xgboost_gain | 30 | 17 | 0.379 | 9.407e-05 |
| 375-507 | xgboost_weight | 20 | 11 | 0.334 | 9.407e-05 |
| 253-374 | xgboost_weight | 20 | 10 | 0.377 | 0.0004503 |
| 375-507 | decision_tree_importance | 20 | 10 | 0.359 | 0.0004503 |
| 375-507 | xgboost_gain | 20 | 10 | 0.317 | 0.0004503 |
| <= 252 | xgboost_gain | 20 | 10 | 0.379 | 0.0004503 |
| 253-374 | xgboost_gain | 30 | 15 | 0.377 | 0.0007516 |

How to read this:
- The hypergeometric p-value asks whether two top-K lists overlap more than expected by chance under a random ranking over 128 channels.
- The FDR correction matters because many bucket/metric/K combinations are tested.
- Spearman `rho` uses the full 128-feature ranking and is less brittle than a single top-K cutoff.

Interpretation:
- Significant overlap means the linear-tree leaf and the nonlinear tree family are repeatedly highlighting some of the same channels. That makes the overlap interpretable, not just anecdotal.
- Non-significant overlap does **not** mean one of the models is wrong. Correlated channels, different inductive biases, and bucket-specific fits can easily reshuffle ranks.
- Decision-tree importance and XGBoost gain are the cleaner comparison targets. XGBoost weight and cover are more structural than predictive.

## 7. Bottom line

- The results look real enough to analyze: held-out protein performance is high, especially for XGBoost, and not obviously due to noise.
- The pair channels dominate predictive power.
- `n_residues` matters some, mainly as a routing / context feature, but it does not look like the main story.
- Sequence-separation features matter a little but probably would not radically change the result if removed; exact ablation still needs a rerun.
- The single decision tree is interpretable only in compressed form. The top levels and heavy branches are useful; the full 805-leaf dump is not a human-scale explanation.
- Depth-4 boosted trees look fine. Depth-4 single trees do not.
- The linear-tree overlap analysis can tell you whether the same pair channels keep resurfacing across model classes, which is the strongest evidence here that some channels are genuinely robustly important rather than ranking highly by chance on one fit.

# Test2 Updated Delta Validation and Geometry Analysis

Generated: 2026-03-02 21:45:43

## Inputs
- Updated data: `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/updated_test2_linear_reg_data`
- Old reference data: `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/test2_linear_reg_data`
- Output root: `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/proteins/multi_protein_results/test2_updated_delta_analysis`

## Validation Note
- Regression outputs are unchanged: `test2_kfold_regression_r2.csv`=match, `test2_kfold_regression_r2_summary.csv`=match, `test2_kfold_regression_coefficients.csv`=match
- Old MDS rows: 1200; updated rows: 6000; updated test rows: 1200.
- Old `test2_kfold_mds_metrics.csv` matches the updated `split=test` subset: yes.
- Updated `test2_kfold_mds_summary.csv` differs by design because it includes both train and test rows.

## Data Integrity
- Unique proteins in updated MDS metrics: 25
- Train rows: 4800 (192.0 per protein)
- Test rows: 1200 (48.0 per protein)
- Aggregated rows per layer after per-protein averaging: 25

## R^2 Across Layers
- Train-only and 5-row-mean protein trajectories were plotted separately so the averaged view does not hide split effects.
- Layer 47 train-test gap: mean=0.0058, median=0.0073, std=0.0123.
- Largest absolute layer-47 train-test R^2 gaps: 7bcj_A (-0.0405), 7au7_A (0.0319), 6xqj_A (0.0191).

## TM-score vs R^2
- Pooled train+test correlation is descriptive only because the rows are repeated measurements across folds and layers.
- Pooled correlation: Pearson=0.5773, Spearman=0.7528 (n=6000).
- Within-slice correlation (safer): Pearson mean=0.5163, median=0.5765, positive fraction=0.940; Spearman mean=0.4738.
- Residualized by layer (subtracting per-layer means): Pearson=0.3604, Spearman=0.4198.

## TM-score vs Protein Size
- Pooled correlation: Pearson=0.4137, Spearman=0.4139 (n=6000).
- Within-slice correlation: Pearson mean=0.4744, Spearman mean=0.4629.

## Joint TM-score Models
- Layer 47 model R^2: `R2-only`=0.3746, `size-only`=0.2703, `R2+size`=0.3974, `adjusted`=0.3975.
- All-row model R^2: `R2-only`=0.3332, `size-only`=0.1712, `R2+size`=0.3780, `adjusted`=0.5531.
- The adjusted all-row model includes layer and split controls only as a confounder check, not as scientific predictors.

## Geometry Trends
- Top-3 eigenvalue fraction, triangle violation fraction, and triangle violation severity were plotted as per-protein 5-row means across layers.
- Triangle violation fraction vs TM-score: pooled Pearson=0.4163, within-slice Pearson mean=0.6489.
- Triangle violation fraction vs per-protein R^2: pooled Pearson=0.2327, within-slice Pearson mean=0.5571.

## Test-Only Size vs Held-Out R^2
- Layer 47 test-only correlation: Pearson=0.6469, Spearman=0.7492.
- Across-layer test-only correlation means: Pearson=0.5805, Spearman=0.7430.

## Key Outputs
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/proteins/multi_protein_results/test2_updated_delta_analysis/visualizations/per_protein_train_mean_r2_across_layers.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/proteins/multi_protein_results/test2_updated_delta_analysis/visualizations/per_protein_five_row_mean_r2_across_layers.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/proteins/multi_protein_results/test2_updated_delta_analysis/visualizations/per_protein_train_minus_test_r2_across_layers.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/proteins/multi_protein_results/test2_updated_delta_analysis/visualizations/layer47_tm_vs_r2_by_fold.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/proteins/multi_protein_results/test2_updated_delta_analysis/visualizations/layer47_tm_vs_size_by_fold.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/proteins/multi_protein_results/test2_updated_delta_analysis/visualizations/per_protein_mean_top3_eigenvalue_fraction_across_layers.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/proteins/multi_protein_results/test2_updated_delta_analysis/visualizations/test_only_size_vs_r2_layer47.png`

## Suggested Follow-Up
- If the residualized TM-score vs R^2 correlation stays sizable, test whether specific proteins drive the effect by repeating the analysis leave-one-protein-out.
- If size effects look nonlinear in the scatter plots, add a follow-up sensitivity model with `log(n_residues)`.

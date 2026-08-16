# Diagnostic Precision / Row-Cap / PCA Analysis

## Main Conclusions
- Float precision is clearly a major part of the old failure mode: the all-row protein-equal global fits jump from `128d float32 = -0.0044` to `128d float64 = 0.8748`, and from `256d float32 = 0.0520` to `256d float64 = 0.9051`.
- Row-capping explains why the older small / capped analyses could look much healthier in float32: for both 128d and 256d, the float32 variants are strong at `5k` and `20k` rows per protein but collapse only when all rows are used.
- Float precision is not the whole story. Even after switching to float64, test `R^2` still rises with protein size. For the global all-row float64 models, Pearson correlation between `n_residues` and test per-protein `R^2` is `0.646` for 128d and `0.606` for 256d.
- Manual size bucketing still matters after fixing precision. The `128d` bucketed float64 model improves overall test mean per-protein `R^2` from `0.8748` to `0.8859`.
- The bucketed model mostly helps the smallest proteins: in the `<=255` bucket, test mean per-protein `R^2` goes from `0.8443` to `0.8712`. It reduces the size-dependence correlation from `0.646` to `0.338`.
- Training weights still matter in float64. The `256d` float64 pair-equal model reaches `0.8567`, below the protein-equal float64 result `0.9051`. The pair-equal model especially gives back performance on the smallest proteins.
- The PCA sweep suggests the 128d decoder has substantial low-rank structure, but not an extremely tiny intrinsic dimension. The top `20` PCs already recover a lot, but the curve keeps improving through `50`, `100`, and `128`.

## Size-Bucket Test R^2
| Variant | <=255 | 256-371 | 372-500 | 501+ |
| --- | ---: | ---: | ---: | ---: |
| 128d f32 cap=all | -0.2037 | 0.2409 | 0.2421 | 0.3666 |
| 128d f64 cap=all | 0.8443 | 0.9055 | 0.9158 | 0.9388 |
| 128d f64 bucketed | 0.8712 | 0.8955 | 0.9205 | 0.8730 |
| 256d f32 cap=all | -0.1231 | 0.2504 | 0.2732 | 0.4106 |
| 256d f64 cap=all | 0.8817 | 0.9286 | 0.9369 | 0.9541 |
| 256d f64 pair-equal | 0.8137 | 0.8955 | 0.9188 | 0.9432 |

## Bucketed-Coefficient Similarity
- Raw-scale cosine similarity between the smallest and largest bucket decoders is `0.5507`.
- The most similar pair is `256-371` vs `372-500` at `0.9258`.
- Relative to the global `128d` float64 decoder, cosine similarity is `0.9106` for `<=255` and `0.7774` for `501+`.

Top raw-scale coefficient shifts between `<=255` and `501+`:
- `pair_ch_84`: abs diff `0.3560` (`<=255`=-0.0490, `501+`=-0.4049)
- `pair_ch_6`: abs diff `0.3248` (`<=255`=0.1044, `501+`=0.4292)
- `pair_ch_111`: abs diff `0.3238` (`<=255`=-0.1324, `501+`=0.1914)
- `pair_ch_106`: abs diff `0.3033` (`<=255`=-0.0231, `501+`=-0.3264)
- `pair_ch_13`: abs diff `0.2841` (`<=255`=-0.0827, `501+`=-0.3668)

## Spectrum Interpretation
- 128d protein-equal all rows: float32 cutoff removes `124` of `128` singular directions, leaving `4` directions above the heuristic threshold.
- 256d protein-equal all rows: float32 cutoff removes `253` of `256` singular directions, leaving `3` directions above the heuristic threshold.

## Key Output Files
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/rowcap_precision_test_r2.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/pca_sweep_test_r2.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/test_bucket_r2_key_variants.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/bucketed_128d_raw_cosine_heatmap.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/test_size_vs_r2_key_variants.png`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/key_variant_overview.csv`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/test_bucket_r2_summary.csv`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/test_size_r2_correlation.csv`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/bucketed_128d_coefficient_cosine_similarity.csv`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/bucketed_128d_top_shifted_channels.csv`
- `/Users/ishankhire/Desktop/ai_safety/AlphaFoldResearch/protein/delta_files/diagnostic_precision_rowcap_pca_csv_files/analysis/spectrum_cutoff_interpretation.csv`

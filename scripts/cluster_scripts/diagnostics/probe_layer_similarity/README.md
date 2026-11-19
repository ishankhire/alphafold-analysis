# Probe layer similarity

Diagnostic comparison of the frozen protocol-2.0 distance probes across the 48
Evoformer layers. It fits nothing and reads only stored artifacts. Follow the
[cluster guide](../../../../docs/cluster/README.md) for authorization and
deletion rules.

## Design

- Cohort: the frozen 552 proteins. Layer pairs: all 48 × 48, stored as the
  upper triangle (`layer_a <= layer_b`).
- **Weight cosine.** Shared linear probe and each of the six sequence-gated
  probes, in standardized and raw-scale coefficient space, intercept excluded.
  Cosine is taken within an outer fold and then averaged over the five folds.
  The cosine between folds at the same layer is reported as a noise floor.
  Per-protein linear probes are left out.
- **Prediction similarity.** Pearson and Spearman correlation between the stored
  out-of-fold predicted distances at two layers, for all four probes.
- **Error similarity.** The same correlations on signed error, probe distance
  minus final structure distance. The final structure is OpenFold's layer-47
  predicted PDB, the probes' training target.
- Prediction and error similarity are computed inside each protein and then
  summarized across proteins with equal weight. Nothing is pooled over residue
  pairs, and figures show protein means only.
- `7stz_C` is included, with no separate sensitivity columns.

Completion gates, both required for `analysis_status=complete`:

- every coefficient artifact matches its replay-index checksum, and every final
  structure PDB matches the checksum recorded by the predicted-distance export;
- per-protein MAE recomputed from the stored predictions matches the producer's
  `per_protein.csv` within `1e-3` Å for every probe, layer and protein. This
  confirms the pair order and the distance target.

## Inputs and outputs

- Producers and predictions: `$CSV_ROOT/regression_5fold_v1/`, with `CSV_ROOT`
  defaulting to `/work/hdd/bfna/ikhire`.
- Final structures: the CAMEO and ECOD PDB paths in `datasets.md`, resolved
  through `cluster_regression_helpers.resolve_combined_pdb_path`.
- Transfer matrix for the figures: `$ALL_LAYER_TRANSFER_OUTPUT_ROOT/transfer_matrix.csv`,
  default `~/diagnostic_experiments/cross_layer_probe_transfer_all_layers_v1/`.
  Figures that need it are skipped if it is missing.
- Output root: `$PROBE_LAYER_SIMILARITY_OUTPUT_ROOT`, default
  `~/diagnostic_experiments/probe_layer_similarity_v1/`.

| File | Rows | One row per | Principal columns |
|---|---:|---|---|
| `layer_pair_similarity.csv` | 35,280 | probe, gate, measure, layer pair | `averaged_over`, `layer_offset`, `n`, `mean`, `sd`, `median`, `q25`, `q75` |
| `weight_cosine_by_fold.csv` | 82,320 | probe, gate, coefficient space, fold, layer pair | `cosine` |
| `weight_fold_noise_floor.csv` | 672 | probe, gate, coefficient space, layer | `mean_cross_fold_cosine`, `min_cross_fold_cosine` |
| `effectiveness_by_layer.csv` | 576 | probe, layer, metric (`r2`, `mae`, `tm_score`) | protein `mean`, `sd`, `median`, `q25`, `q75` |
| `parity_checks.csv` | 192 | probe, layer | `max_abs_mae_difference`, `status` |
| `per_protein_similarity.npz` | array | protein, probe, measure, layer, layer | float32 correlations; stays on Delta |
| `run_manifest.json` | one object | run | `analysis_status`, `code_revision`, parity summary |

Measures in `layer_pair_similarity.csv`: `weight_cosine_standardized`,
`weight_cosine_raw_scale` (`averaged_over=outer_fold`), and
`prediction_pearson`, `prediction_spearman`, `error_pearson`, `error_spearman`
(`averaged_over=protein`). `gate` is empty except for the sequence-gated weights.

The figure stage writes nine PNG and PDF files to `figures/`.

### Transfer by stratum

`transfer_by_stratum.py` reads the all-layer transfer run's `layer_shards/` and
replays nothing. For each source layer, target layer and canonical stratum (six
sequence-separation bins, five final-structure-distance bins, and all pairs) it
takes each protein's mean absolute error and averages over proteins. It writes
`transfer_by_stratum.csv` (25,344 rows; one per stratum axis, stratum, source
layer and target layer; `n_proteins`, `mean_protein_mae`, `median_protein_mae`,
`mean_protein_excess_mae`, `median_protein_excess_mae`,
`native_target_mean_protein_mae`) and four heatmap figures, `10_` to `13_`. Its
gate is that the all-pairs mean reproduces `transfer_matrix.csv`. Submit
`transfer_by_stratum.sbatch` after the transfer aggregation; job `24134288`
took 23 seconds and 1.2 GiB.

## Deploy and submit

Copy into `$HOME/openfold_interpretability_canonical/`: every `.py` and
`.sbatch` file in this directory, this README as
`probe_layer_similarity_README.md`, and `canonical_storage.py` and
`cluster_regression_helpers.py` from `../..`. Record `git_commit` in
`deployment_manifest.json`.

Announce each job first. Launchers use `~/protein_venv` and leave `--mail-user`
to the submit command.

```bash
cd "$HOME/openfold_interpretability_canonical"
MAIL=--mail-user="${CLUSTER_NOTICE_EMAIL:?Set CLUSTER_NOTICE_EMAIL}"
SIM=$(sbatch --parsable "$MAIL" probe_layer_similarity.sbatch)
sbatch --parsable "$MAIL" --dependency=afterok:$SIM make_probe_layer_similarity_figures.sbatch
```

`PROBE_LAYER_SIMILARITY_MAX_PROTEINS=<n>` limits the run to the first `n`
proteins for a pilot; such a run is marked `partial_cohort` and figures refuse it
unless `PROBE_LAYER_SIMILARITY_ALLOW_PARTIAL=1`.

Tests, on synthetic data only:

```bash
python -m pytest scripts/cluster_scripts/diagnostics/tests/test_probe_layer_similarity.py -q
```

# Native error strata

Diagnostic of where each native `shared_ols_base256` probe reads away from the
final structure. It bins the existing packed OOF predictions by fine sequence
separation and final structure distance at all 48 layers. It fits nothing and
opens neither probes nor pair tensors. Follow the
[cluster guide](../../../../docs/cluster/README.md) for authorization and deletion
rules.

## Design

- Target: the Cα distance in OpenFold's final predicted structure. Signed error is
  probe distance minus final structure distance.
- Bins: 13 sequence-separation and 17 distance bins, defined in `strata_bins.py`,
  which nest exactly in the canonical subgroup bins.
- Cohort view: `build_strata.py` covers all 552 proteins, as `all_552` and
  `excluding_7stz_C`. It writes its table only after the bins reconcile with
  `validation.csv` and the canonical subgroup metrics at every layer.
- Protein view: `build_protein_strata.py` covers the 154 CAMEO proteins, and
  `prepare_protein_page.py` tests stated claims protein by protein. Its claims
  table does not match the 551-protein cohort view.
- Target check: `verify_training_target.py` recomputes the statistics in
  `training_metrics.csv`, confirming these distances are the probes' training
  targets (240 fit rows, 50 baseline rows, and 552 PDB hashes matched).
- XGBoost view: `native_strata_xgboost.sbatch` runs the protein view for
  `shared_xgboost_base256_fixed_s4` by setting `NATIVE_ERROR_STRATA_PROBE` and
  `NATIVE_ERROR_STRATA_COHORT=cameo`. OLS stays the default. `build_cameo_strata.py`
  sums the protein view into `native_error_strata_cameo.csv`, because the
  all-protein table needs ECOD, and `compare_to_ols.py` compares both probes on the
  same pairs. Per-protein reconciliation uses `publication_subgroup_mae_5fold_v1`,
  because v1's signed shards exist only for OLS. Its figures reuse the OLS colour
  scales. `verify_training_target.py` is OLS-only: XGBoost fits on sampled pairs, and
  its configuration records the same final structure target.

The ECOD final structure PDBs are unavailable as of 2026-11-06. Until they return,
`build_strata.py` and `verify_training_target.py` cannot run, so
`native_error_strata.csv`, `native_strata.npz`, `checks.json`, and
`training_target_check.json` cannot be rebuilt. `plot_strata.py` and the protein
view still run.

## Deploy

Copy these flat into `$HOME/openfold_interpretability_canonical/`:

- from this directory: every `.py` and `.sbatch` file, `page_text_*.json`,
  `protein_page_template.html`, and this README as
  `native_error_strata_README.md`;
- from `../..`: `canonical_storage.py`, `cluster_regression_helpers.py`,
  `regression_output_helpers.py`, and `regression_protocol.py`, with
  `regression_protocol_data/regression_master_split_v1.csv`.

## Inputs and outputs

- Packed predictions: `$CSV_ROOT/regression_5fold_v1/predicted_distances_5fold_v1/`.
  `CSV_ROOT` defaults to `/work/hdd/bfna/ikhire`.
- Canonical tables: `$CSV_ROOT/regression_5fold_v1/shared_ols_base256/` and
  `$CSV_ROOT/regression_5fold_v1/metadata/`, plus the ten-layer run's `layer_shards/`
  and `source_probe_status.csv` under
  `$CSV_ROOT/diagnostic_experiments/cross_layer_probe_transfer_v1/`.
- PDBs: the paths in `datasets.md`. ECOD also needs
  `$CROSS_LAYER_ECOD_EXPERIMENTS_ROOT`, the folder holding `representation_saves/`
  and `openfold_reps_outputs_sm/`.
- Output root: `$NATIVE_ERROR_STRATA_OUTPUT_ROOT`, default
  `~/diagnostic_experiments/native_error_strata_v1/`.
- Slurm logs: `slurm-<job>.{out,err}`, in the deployment directory.

| File under the output root | Rows | One row per | Principal columns |
|---|---:|---|---|
| `data/native_error_strata.csv` | 21,216 | layer, scope, stratum, with empty strata kept | `mean_signed_error`, `sd_signed_error`, `mae`, `rmse`, `n_proteins`, `support_status` |
| `data/native_strata.npz` | array | layer, sequence bin, distance bin | additive error sums, whole and for `7stz_C` |
| `data/checks.json`, `data/training_target_check.json` | one object | run | reconciliation results |
| `data/protein_strata_cameo.npz` | array | protein, layer, stratum | sparse per-protein error sums |
| `data/claims_check_cameo.csv` | 11 | claim | proteins satisfying each claim |
| `data/protein_strata_cameo_check.json` | one object | run | per-protein reconciliation |
| `data/protein_page_payload_cameo.json` | one object | run | packed data for the protein page |
| `figures/<scope>/` | PNG, PDF | overview | mean and SD of signed error at all layers |
| `figures/<scope>/by_layer/` | PNG | layer | mean ± SD grid |
| `data/native_error_strata_cameo.csv` | 21,216 | layer, scope, stratum, CAMEO runs only | same columns as the cohort table |
| `data/strata_vs_ols_cameo.csv` | one per layer and filled stratum | layer, stratum, 153 CAMEO proteins | `mean_signed_error`, `mae`, and their `_ols` counterparts |
| `figures/mae_difference_vs_ols_all_layers.*` | PNG, PDF | overview | MAE minus OLS MAE at all layers |

The launchers finish by building single-file HTML pages under the output root
with `make_page.py` and `make_protein_page.py`. Copies of the CAMEO-run pages, as
published, are in `delta_files/diagnostic_plots/native_error_strata/` and
`delta_files/diagnostic_plots/native_error_strata_xgboost/`.
`../tests/test_protein_page.js` exercises a built page:
`node ../tests/test_protein_page.js <page.html>`.

## Submit

Announce each job first. Launchers use `~/protein_venv` (see the
[cluster guide](../../../../docs/cluster/README.md#python-environment)) and leave
`--mail-user` to the submit command. A resubmission skips any scan whose output
already exists.

```bash
cd "$HOME/openfold_interpretability_canonical"
MAIL=--mail-user="${CLUSTER_NOTICE_EMAIL:?Set CLUSTER_NOTICE_EMAIL}"
sbatch "$MAIL" protein_strata.sbatch                         # CAMEO protein view
sbatch "$MAIL" native_strata_xgboost.sbatch                  # XGBoost CAMEO views; output root native_error_strata_xgboost_v1
```

When the ECOD PDBs return:

```bash
export CROSS_LAYER_ECOD_EXPERIMENTS_ROOT=/path/holding/both/folders
sbatch "$MAIL" native_error_strata.sbatch                    # cohort view, target check, plots
```

# Layerwise geometric decoding in OpenFold

This is the compact companion repository for the NeurIPS workshop submission
whose main text is `draft_4.tex`. It contains the four probe implementations,
the frozen cross-validation manifests, plot-ready source data, and the six PDFs
included by that draft. Raw pair representations, per-pair predictions, model
artifacts, exploratory analyses, logs, and unused figures are intentionally
omitted.

## Experiment

OpenFold v2 was run without recycling. At each of its 48 Evoformer layers, the
two directional 128-dimensional pair representations were concatenated into a
256-dimensional feature for each unordered residue pair. Four distance probes
were evaluated:

- one shared ordinary-least-squares (OLS) model across proteins;
- independent per-protein OLS models;
- OLS models gated by sequence separation; and
- one shared XGBoost model.

Shared probes use five protein-level outer folds. Per-protein probes use five
pair-level folds within each protein. Reported curves are protein-equal means;
confidence bands are paired-protein bootstrap 95% intervals.

The source pool has 554 CAMEO and ECOD proteins. The frozen probe cohort has
552: ECOD proteins `000196893` and `001681437` are excluded because their layer
0 pair-tensor dimensions do not match their residue counts. The manifests in
`delta_files/regression_5fold_v1/metadata/` record the exact cohort and folds.

## Contents

- `scripts/cluster_scripts/cross_validation/`: the four Delta/SLURM probe
  workflows and their aggregators.
- `scripts/multi_protein/`: the two local renderers.
- `delta_files/publication_plots/sept_2_plots/source_data/`: compact source
  tables for the five probe figures.
- `delta_files/layerwise_ablation_csvs/`: Pair Transition and MSA Transition
  tables used by the combined ablation figure.
- `delta_files/publication_plots/sept_2_plots/`: the six final PDFs and a hash
  manifest for the five probe figures.

The training entrypoints expect the original pair tensors and predicted PDBs
at the Delta paths defined in `cluster_regression_helpers.py`; those large
inputs are not redistributed. Submit the four `*_layer.sbatch` or
`*_fold_layer.sbatch` files from `scripts/cluster_scripts/cross_validation/`,
then submit the matching `aggregate_*.sbatch` jobs after their arrays finish.

## Reproduce the release

Use Python 3.13:

```bash
python3.13 -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/python scripts/multi_protein/generate_sept_2_publication_plots.py
./.venv/bin/python scripts/multi_protein/plot_layerwise_ablation_tm_scores.py
./.venv/bin/python -m unittest discover -s tests
```

The first renderer recreates the five probe PDFs and their SHA-256 manifest;
the second recreates the combined layerwise-ablation PDF. Both fail if their
input or output inventory differs from the submission release.

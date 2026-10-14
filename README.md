# AlphaFold Pair Representation Analysis

This repository studies what AlphaFold/OpenFold pair representations encode
across 48 layers, mainly with distance-regression probes.

## Key links

- [Project summary](project-summary.md) explains the scientific findings and limitations.
- [Current sources and status](docs/project_status.md) identifies the latest data,
  result families, plots, and pending work.
- [Regression methodology](docs/methodology/regression_methodology.md) defines
  the shared methodology for all regression variants.
- [Fivefold regression results](delta_files/regression_5fold_v1/) contains the
  current protocol-2.0 outputs.
- [September 2 publication plots](delta_files/publication_plots/sept_2_plots/)
  contains the designated paper figures, source tables, and provenance manifest;
  optimized TM-score rescoring and the corresponding figure refresh are pending.
- [Delta cluster guide](docs/cluster/README.md) covers deployment and job submission.
- [Datasets](datasets.md) defines the input pool and input locations.

## Repository layout

- `scripts/cluster_scripts/` — cluster-ready regression and analysis workflows.
- `scripts/multi_protein/` — local analysis and post-processing scripts.
- `docs/` — current status, methodology, experiment specifications, and cluster guidance.
- `docs/historical/` — completed/superseded plans and dated audits.
- `delta_files/` — current regression outputs and analyses.
- `delta_files/historical_outputs/` — archived output bundles; some older outputs
  remain at established paths for existing analysis scripts.
- `tests/` — automated checks for the analysis code.

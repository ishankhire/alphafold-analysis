# AlphaFold Pair Representation Analysis

This repository studies what AlphaFold/OpenFold pair representations encode
across 48 layers, mainly with distance-regression probes.

## Key links

- [Regression methodology](docs/methodology/regression_methodology.md) defines
  the shared methodology for all regression variants.
- [Paper figure results](delta_files/analysis/paper_figure_results.md) summarizes
  the main results.
- [Main analysis results](delta_files/analysis/) contains the primary result
  tables and reports, and [paper figures](delta_files/analysis/paper_figures/)
  contains the figures.
- [Experiment plans](docs/plans/) contains more detailed plans used to carry
  out some regressions.

## Repository layout

- `scripts/cluster_scripts/` — cluster-ready regression and analysis workflows.
- `scripts/multi_protein/` — local analysis and post-processing scripts.
- `docs/` — methodology, plans, and cluster notes.
- `delta_files/` — current regression outputs and analyses.
- `historical_outputs/` — archived outputs from earlier workflows.
- `tests/` — automated checks for the analysis code.
- [`datasets.md`](datasets.md) — dataset membership and input locations.

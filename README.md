# AlphaFold Pair Representation Analysis

This repository studies what AlphaFold/OpenFold pair representations encode
across 48 layers, mainly with distance-regression probes.

## Key links

- [Regression methodology](docs/methodology/regression_methodology.md) defines
  the shared methodology for all regression variants.
- [Fivefold regression results](delta_files/regression_5fold_v1/) contains the
  current protocol-2.0 outputs.
- [September 1 publication plots](delta_files/publication_plots/sept_1_plots/)
  contains the current paper figures, source tables, and provenance manifest.
- [Cross-validation correctness review](delta_files/analysis/cross_validation_adversarial_review_v1/)
  contains the current protocol-2.0 adversarial audit artifacts.
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

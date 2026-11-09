# null_triangles_cameo_v1 (superseded)

This is the CAMEO-only diagnostic version of the null-triangle experiment: 154 proteins, Nulls A/B/C, K=10, with nulls scored on a sample of up to 1M triangles per protein. Delta array `23656699` and aggregation `23656700` produced it in Nov 2026.

- `tables/`: the aggregated tables plus the local `RESULTS.md` and `sanity_checks.md`. These were previously in `delta_files/diagnostic_experiments/null_triangles_cameo_v1/`.
- `figures/`: these were previously in `delta_files/diagnostic_plots/null_triangles_cameo_v1/`.
- Code: commit `4741ee5`, in `scripts/cluster_scripts/diagnostics/null_triangles/` and `scripts/multi_protein/null_triangles/analyze_null_triangles.py`.
- Delta copy: `/work/hdd/bfna/ikhire/diagnostic_experiments/null_triangles_cameo_v1/`, left in place.

It is superseded by `null_triangles_5fold_v1`, which covers all 552 proteins and scores every triangle. That experiment is in `scripts/cluster_scripts/null_triangles/` and `delta_files/regression_5fold_v1/analysis/null_triangles_5fold_v1/`.

# Biological-Context Probe Figures

These figures summarize the held-out-protein results from the local biological-
context distance-update probe. Regenerate every table and figure from this
directory with:

```bash
MPLCONFIGDIR=../../../.mplconfig XDG_CACHE_HOME=../../../.cache \
  ../../../.venv/bin/python analyze_biological_context_probe.py
```

## Figures

- `biological_context_headline_performance.png`: mean held-out performance for
  every bundle and primary estimator, averaged across five transitions. Error
  bars span the minimum and maximum transition values.
- `biological_context_ablation_gains.png`: transition-level declared bundle
  gains with paired 95% held-out protein-bootstrap intervals.
- `biological_context_performance_by_transition.png`: held-out controls,
  interpretable biology-plus-triangle, and pair-center performance at each
  transition.
- `biological_context_feature_family_importance.png`: held-out HistGB family
  permutation importance, averaged across five transitions and three repeats.
- `biological_context_target_distributions.png`: held-out target quantiles,
  meaningful-change coverage, and shrink/expand balance by transition.

## Supporting tables

- `headline_performance_summary.csv`: one row per target, estimator, and bundle;
  mean/minimum/maximum headline metric across the five transitions.
- `ablation_gain_summary.csv`: one row per target, estimator, and declared
  contrast; cross-transition gain range and count of transition-specific 95%
  intervals excluding zero.
- `transition_performance_summary.csv`: one row per transition, target,
  estimator, and selected bundle.
- `feature_family_importance_summary.csv`: one row per bundle, target,
  estimator, and feature family, aggregated over transitions and repeats.
- `target_distribution_summary.csv`: one row per held-out transition with
  target quantiles, meaningful-change coverage, and direction balance.

All classifier headline metrics exclude pairs with `abs(delta_d) <= 0.5 A`.
Regression headline metrics use every eligible local pair. Predictive effects
are diagnostic associations with decoded distances, not causal mechanisms or
physical residue movements.

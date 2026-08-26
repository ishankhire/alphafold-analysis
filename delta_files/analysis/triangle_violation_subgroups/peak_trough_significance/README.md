# Search-adjusted triangle-metric peaks and troughs

## Scope and result definition

This analysis tests local extrema in the five focused triangle-consistency figure families using paired measurements from the same 111 proteins. It uses only `violation_fraction` and `mean_relative_violation_all`; R2 is not read.

Every layer 5--42 was tested as a possible peak and trough before candidate filtering. A reported supported extremum is the strict maximum/minimum in its +/-5 neighborhood, has the required fitted slope directions, and passes an outcome-specific Benjamini--Hochberg threshold of q <= 0.05.

## Candidate and supported counts

| Role | Metric | Direction | Shape-qualified candidates | BH-supported |
|---|---|---|---|---|
| Secondary | violation_fraction | Peak | 103 | 75 |
| Secondary | violation_fraction | Trough | 62 | 32 |
| Primary | mean_relative_violation_all | Peak | 112 | 88 |
| Primary | mean_relative_violation_all | Trough | 68 | 24 |

## Supported extrema by model

| Model | Metric | Peaks | Troughs |
|---|---|---|---|
| Shared linear | violation_fraction | 21 | 8 |
| Shared linear | mean_relative_violation_all | 27 | 9 |
| Per-protein linear | violation_fraction | 19 | 11 |
| Per-protein linear | mean_relative_violation_all | 21 | 11 |
| XGBoost | violation_fraction | 16 | 4 |
| XGBoost | mean_relative_violation_all | 19 | 2 |
| Sequence gated | violation_fraction | 19 | 9 |
| Sequence gated | mean_relative_violation_all | 21 | 2 |

## All-triangle supported layers

This compact view answers the original unstratified question. `None` means that no layer met all shape and search-adjusted significance requirements.

| Model | Metric | Peak layers | Trough layers |
|---|---|---|---|
| Shared linear | violation_fraction | 34 | None |
| Shared linear | mean_relative_violation_all | 10, 34, 40 | None |
| Per-protein linear | violation_fraction | 10 | 22 |
| Per-protein linear | mean_relative_violation_all | 6, 30 | 22 |
| XGBoost | violation_fraction | None | None |
| XGBoost | mean_relative_violation_all | 7, 30 | None |
| Sequence gated | violation_fraction | 11, 34 | None |
| Sequence gated | mean_relative_violation_all | 10, 34 | None |

## Most recurrent exact layers

Counts below are numbers of the 52 trajectories with a supported extremum at that exact layer. Nearby-event comparisons within two layers are retained in the alignment tables.

| Metric | Direction | Layer | Trajectory count |
|---|---|---|---|
| mean_relative_violation_all | Peak | 34 | 22 |
| mean_relative_violation_all | Peak | 30 | 14 |
| mean_relative_violation_all | Peak | 6 | 12 |
| mean_relative_violation_all | Peak | 10 | 9 |
| mean_relative_violation_all | Peak | 32 | 8 |
| mean_relative_violation_all | Peak | 40 | 7 |
| mean_relative_violation_all | Peak | 7 | 4 |
| mean_relative_violation_all | Peak | 9 | 4 |
| mean_relative_violation_all | Trough | 22 | 14 |
| mean_relative_violation_all | Trough | 35 | 3 |
| mean_relative_violation_all | Trough | 18 | 2 |
| mean_relative_violation_all | Trough | 20 | 2 |
| mean_relative_violation_all | Trough | 12 | 1 |
| mean_relative_violation_all | Trough | 23 | 1 |
| mean_relative_violation_all | Trough | 42 | 1 |
| violation_fraction | Peak | 34 | 24 |
| violation_fraction | Peak | 30 | 10 |
| violation_fraction | Peak | 32 | 7 |
| violation_fraction | Peak | 6 | 5 |
| violation_fraction | Peak | 10 | 5 |
| violation_fraction | Peak | 40 | 5 |
| violation_fraction | Peak | 8 | 4 |
| violation_fraction | Peak | 9 | 4 |
| violation_fraction | Trough | 22 | 13 |
| violation_fraction | Trough | 24 | 4 |
| violation_fraction | Trough | 23 | 3 |
| violation_fraction | Trough | 35 | 3 |
| violation_fraction | Trough | 20 | 2 |
| violation_fraction | Trough | 27 | 2 |
| violation_fraction | Trough | 12 | 1 |
| violation_fraction | Trough | 18 | 1 |

## Largest supported raw effects

The table shows the five largest conservative raw prominences for each metric and direction. The x100 magnitude is in percentage points for violation fraction and normalized-burden points for mean relative violation. Raw magnitudes are most interpretable within a metric; paired Cohen's dz aids comparison across differently scaled categories.

| Metric | Direction | Model | Category | Layer | Prominence x100 | Cohen dz | BH q |
|---|---|---|---|---|---|---|---|
| violation_fraction | Peak | Shared linear | rms_size:Small | 9 | 1.971 | 0.511 | 1.46e-05 |
| violation_fraction | Peak | Sequence gated | rms_size:Small | 9 | 1.780 | 0.618 | 1.48e-07 |
| violation_fraction | Peak | Sequence gated | area:Low | 11 | 1.496 | 0.485 | 4.25e-05 |
| violation_fraction | Peak | Shared linear | relative_slack:Low | 34 | 1.409 | 0.318 | 1.35e-02 |
| violation_fraction | Peak | Shared linear | rms_size:Medium | 34 | 1.404 | 0.580 | 8.70e-07 |
| violation_fraction | Trough | Per-protein linear | area:Low | 22 | 0.765 | 0.900 | 1.18e-13 |
| violation_fraction | Trough | Shared linear | rms_size:Large | 35 | 0.680 | 0.500 | 2.29e-05 |
| violation_fraction | Trough | Per-protein linear | rms_size:Small | 22 | 0.598 | 0.934 | 3.05e-14 |
| violation_fraction | Trough | Per-protein linear | relative_slack:Low | 22 | 0.524 | 0.844 | 2.07e-12 |
| violation_fraction | Trough | Per-protein linear | angle_shape:Clearly Obtuse | 22 | 0.450 | 0.922 | 4.40e-14 |
| mean_relative_violation_all | Peak | Shared linear | area:Low | 9 | 0.600 | 0.409 | 6.07e-04 |
| mean_relative_violation_all | Peak | Shared linear | relative_slack:Low | 10 | 0.589 | 0.463 | 8.80e-05 |
| mean_relative_violation_all | Peak | Sequence gated | area:Low | 9 | 0.506 | 0.581 | 7.40e-07 |
| mean_relative_violation_all | Peak | Shared linear | angle_shape:Clearly Obtuse | 10 | 0.482 | 0.458 | 1.06e-04 |
| mean_relative_violation_all | Peak | Shared linear | rms_size:Small | 9 | 0.482 | 0.391 | 1.14e-03 |
| mean_relative_violation_all | Trough | Per-protein linear | area:Low | 22 | 0.129 | 0.710 | 2.09e-09 |
| mean_relative_violation_all | Trough | Per-protein linear | rms_size:Small | 22 | 0.107 | 0.681 | 8.13e-09 |
| mean_relative_violation_all | Trough | Per-protein linear | relative_slack:Low | 22 | 0.107 | 0.846 | 2.16e-12 |
| mean_relative_violation_all | Trough | Shared linear | rms_size:Large | 35 | 0.078 | 0.474 | 5.59e-05 |
| mean_relative_violation_all | Trough | Shared linear | rms_size:Medium | 22 | 0.075 | 0.401 | 8.07e-04 |

## Interpretation of magnitude

Each row reports separate center-versus-left and center-versus-right paired mean contrasts. `raw_prominence` is the smaller of those two contrasts, so it is a conservative descriptive magnitude. `standardized_prominence` is the smaller paired Cohen's dz. The side-specific 95% intervals are ordinary unadjusted t intervals; BH-adjusted q-values, rather than those intervals, determine search-adjusted support.

Violation-fraction x100 columns are percentage-point effects. Relative-burden x100 columns express the average normalized triangle-violation burden on a 100-point scale; they are not percent changes from baseline.

## Cross-model and cross-category agreement

For the primary metric's peaks, 41 of 48 within-model category pairs and 38 of 78 same-category model pairs shared at least one supported event within two layers. This indicates substantial category alignment but only partial model alignment. Unmatched events are descriptive differences and are not formal tests that peak locations differ.

| Comparison | Metric | Direction | Trajectory pairs | Pairs with exact match | Pairs with +/-2 match | Exact event matches | +/-2 event matches |
|---|---|---|---|---|---|---|---|
| Across categories | mean_relative_violation_all | Peak | 48 | 23 | 41 | 27 | 54 |
| Across categories | mean_relative_violation_all | Trough | 48 | 7 | 7 | 7 | 7 |
| Across categories | violation_fraction | Peak | 48 | 17 | 30 | 19 | 35 |
| Across categories | violation_fraction | Trough | 48 | 8 | 9 | 8 | 9 |
| Across models | mean_relative_violation_all | Peak | 78 | 28 | 38 | 38 | 52 |
| Across models | mean_relative_violation_all | Trough | 78 | 5 | 9 | 5 | 9 |
| Across models | violation_fraction | Peak | 78 | 30 | 44 | 32 | 52 |
| Across models | violation_fraction | Trough | 78 | 5 | 16 | 6 | 17 |

`alignment_comparisons.csv` records exact, within-two-layer, and unmatched supported events. `alignment_summary.csv` has one row per trajectory pair. Exact selected layers are always retained; the tolerance is only a secondary broad-stage label.

## Files

- `all_layer_tests.csv`: all 7,904 predefined layer/direction hypotheses.
- `candidate_extrema.csv`: strict-window candidates with the required slope shape.
- `trajectory_summary.csv`: candidate and supported layers per trajectory.
- `alignment_comparisons.csv`: event-level model/category comparisons.
- `alignment_summary.csv`: pair-level agreement counts.
- `annotated_trajectories/`: hollow candidate markers and filled supported markers.

## Caveats

These are search-adjusted local-extremum findings, not independent replication. BH targets the expected false-discovery proportion across the predefined search; it does not assign a probability that an individual extremum is true. Selection can still make reported effect magnitudes optimistic. Independent proteins would be needed for fully external confirmation.

The t-tests treat proteins, not triangles, as independent units. The analysis does not bootstrap, test ten neighbors individually, or make formal claims that extremum locations differ between models or categories. The per-protein linear probe retains its mixed fitted/held-out edge interpretation from the source workflow.

All component tests used n=111; all BH families contained 3952 hypotheses.

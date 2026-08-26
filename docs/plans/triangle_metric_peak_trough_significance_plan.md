# Triangle-Metric Peak and Trough Significance Analysis

## Summary

Create a local analysis of peaks and troughs in the five focused
triangle-consistency figure families. Use only the two triangle metrics and the
underlying paired measurements for the same 111 proteins; do not use R2.

Mean relative violation burden is the primary outcome and violation fraction
is secondary. The workflow uses the agreed strict-window candidate definition,
side-average paired t-tests, and slope-shape check. It does not run the ten
individual-neighbor tests.

## Statistical method

- Analyze four models and the 13 category curves in `all_triangles`, `rms_size`,
  `area`, `relative_slack`, and `angle_shape`.
- Test layers 5--42 so every location has five layers on each side.
- A peak must exceed every group mean in its +/-5 window and have a positive
  fitted left slope and negative fitted right slope. A trough reverses all
  three conditions.
- For each protein, contrast the center with the mean of layers `k-5..k-1` and
  separately with the mean of layers `k+1..k+5`. Orient both contrasts so a
  positive value supports the proposed extremum.
- Run two one-sample one-sided t-tests over the 111 paired protein contrasts
  and combine them with `max(p_left, p_right)`.
- Compute peak and trough tests for every eligible layer before filtering.
  Apply Benjamini--Hochberg separately to the 3,952 hypotheses for each
  outcome. A supported extremum must pass the strict window, slope shape, and
  outcome-specific `q <= 0.05` threshold.

Report the center and side means, left and right raw contrasts with unadjusted
two-sided 95% t intervals, paired Cohen's dz for each side, raw prominence as
the smaller mean contrast, and standardized prominence as the smaller dz.
Display violation-fraction contrasts in percentage points and relative-burden
contrasts in raw units and multiplied by 100.

Compare supported extrema across models at fixed category and across
categories at fixed model. Preserve exact layers and also flag extrema within
two layers as the same broad event.

## Outputs

Read:

```text
delta_files/triangle_violation_subgroups/per_protein_triangle_metrics.csv
```

Write:

```text
delta_files/analysis/triangle_violation_subgroups/peak_trough_significance/
  README.md
  all_layer_tests.csv
  candidate_extrema.csv
  trajectory_summary.csv
  alignment_comparisons.csv
  alignment_summary.csv
  figure_manifest.csv
  generation_summary.json
  annotated_trajectories/
    figure_01_all_triangles_peak_troughs.{pdf,png}
    figure_02_rms_size_peak_troughs.{pdf,png}
    figure_03_area_peak_troughs.{pdf,png}
    figure_04_relative_slack_peak_troughs.{pdf,png}
    figure_05_angle_shape_peak_troughs.{pdf,png}
```

`all_layer_tests.csv` has one row per model, category, outcome, direction, and
eligible layer. `candidate_extrema.csv` is its shape-qualified subset.
`trajectory_summary.csv` has one row per model-category-outcome trajectory.
Alignment outputs retain event-level matches and pair-level summaries.

## Acceptance criteria

- Validate exactly 277,056 focused input rows, four models, 48 layers, 111
  proteins, 13 category curves, no duplicate keys, and finite outcomes.
- Emit exactly 7,904 all-layer hypothesis rows and two BH families of 3,952.
- Reject plateaus, wrong-way slope shapes, and incomplete edge windows.
- Handle zero-variance and nonfinite contrasts with explicit statuses.
- Verify peak/trough sign reversal, combined p-value construction, and BH
  ordering with synthetic unit tests.
- Produce five annotated figure families in PDF and PNG. Hollow markers denote
  shape-qualified candidates and filled markers denote BH-supported extrema.
- Document search-adjusted interpretation, effect-size caveats, and the lack of
  bootstrap, R2, and individual-neighbor testing.

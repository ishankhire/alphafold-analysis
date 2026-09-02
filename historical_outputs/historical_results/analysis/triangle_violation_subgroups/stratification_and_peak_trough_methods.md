# Triangle stratification and peak/trough methods

## Scope

The triangle subgroups are fixed from each protein's final-PDB C-alpha
geometry. They are not recomputed from decoded distances and do not vary by
model or Evoformer layer. All reported trajectory values are protein-equal
means over the same 111 analysis proteins under the `all_edges` evaluation
scope.

The peak/trough analysis uses the five focused category systems
(`all_triangles`, `rms_size`, `area`, `relative_slack`, and `angle_shape`) and
two outcomes (`violation_fraction` and `mean_relative_violation_all`). The
`angle_shape_exact90` and `rms_by_angle` systems appear in the comprehensive
subgroup diagnostics but are not included in the current peak/trough tests.
R-squared and other decodability measurements are not used in peak/trough
detection.

## Reference-triangle geometry and strata

For each unordered triple of C-alpha residues, sort its three reference side
lengths as

\[
a \le b \le c.
\]

Thus, the largest interior angle is the angle opposite `c`.

### RMS size

Reference RMS size is

\[
r = \sqrt{\frac{a^2+b^2+c^2}{3}}.
\]

This is a triangle-size measure in angstroms, not an error or an RMSE between
predicted and reference values. `small`, `medium`, and `large` are defined by
the within-protein one-third and two-thirds quantiles of `r` over all reference
triangles in that protein.

### Area

For reference C-alpha coordinates `x_i`, `x_j`, and `x_k`, area is

\[
A = \frac{1}{2}\left\|(x_j-x_i)\mathbin{\times}(x_k-x_i)\right\|.
\]

`low`, `medium`, and `high` are the within-protein area tertiles.

### Relative slack

Reference relative slack is

\[
s = \frac{a+b-c}{c}, \qquad 0 \le s \le 1.
\]

Low slack identifies nearly degenerate triangles close to equality in the
triangle inequality. High slack indicates a larger geometric margin. `low`,
`medium`, and `high` are the within-protein slack tertiles.

For RMS size, area, and relative slack, if `q1` and `q2` are the corresponding
within-protein one-third and two-thirds quantiles, the half-open bins are

\[
\begin{aligned}
\text{small/low} &: x < q_1,\\
\text{medium} &: q_1 \le x < q_2,\\
\text{large/high} &: x \ge q_2.
\end{aligned}
\]

Consequently, tied values at a boundary can make the category counts slightly
unequal despite the use of tertile thresholds.

### Largest-angle classes

The largest reference angle is

\[
\theta = \cos^{-1}\left(\frac{a^2+b^2-c^2}{2ab}\right).
\]

The broad angle-shape categories are

- `clearly_acute`: \(\theta < 88^\circ\);
- `near_right`: \(88^\circ \le \theta \le 92^\circ\);
- `clearly_obtuse`: \(\theta > 92^\circ\).

In particular, an acute angle between 88 and 90 degrees belongs to
`near_right`, not `clearly_acute`.

### Exact-90 classes

The separate exact-angle system uses

\[
\Delta = c^2-a^2-b^2
\]

and the scale-aware tolerance

\[
\tau = 10^{-12}\max(c^2, a^2+b^2, 1).
\]

Its categories are

- `acute`: \(\Delta < -\tau\);
- `right_within_tolerance`: \(-\tau \le \Delta \le \tau\);
- `obtuse`: \(\Delta > \tau\).

The `right_within_tolerance` category has no finite support in the current
analysis. `rms_by_angle` is the Cartesian product of the three RMS-size bins
and the three broad 88/92-degree angle classes, giving nine categories.
`all_triangles` contains every unordered residue triple, for a total of
\(\binom{R}{3}\) triangles in a protein with `R` residues.

## Violation outcomes used for peak/trough testing

Decoded pair distances are first clipped below at zero. After sorting the
three decoded sides as \(\hat a \le \hat b \le \hat c\), the absolute violation
amount is

\[
\delta =
\begin{cases}
\hat c-\hat a-\hat b, &
  \hat c-\hat a-\hat b > 10^{-12}\ \text{angstrom},\\
0, & \text{otherwise}.
\end{cases}
\]

Within a reference-defined subgroup,

\[
\texttt{violation_fraction}
= \frac{\#\{\delta>0\}}{N}
\]

and

\[
\texttt{mean_relative_violation_all}
= \frac{1}{N}\sum_{t=1}^{N}\frac{\delta_t}{r_t},
\]

where `r_t` is that triangle's **reference** RMS size. Nonviolating triangles
therefore contribute zero to the latter mean. These two per-protein outcomes
are the values used in the paired layer tests.

## Peak and trough candidate detection

Only center layers 5 through 42 are eligible, leaving five layers on both
sides. For an eligible layer `l`, let `y_p,l` be one protein's outcome and let
`bar_y_l` be the protein-equal mean across the 111 proteins.

A shape-qualified peak must satisfy both conditions:

1. `bar_y_l` is strictly greater than each of the ten individual neighboring
   layer means at `l-5,...,l-1,l+1,...,l+5`.
2. An OLS line fitted to the six group means at `l-5,...,l` has positive
   slope, while an OLS line fitted to `l,...,l+5` has negative slope.

A shape-qualified trough reverses all inequalities and slope directions. The
slopes are only shape requirements; they are not separate significance tests.

Every eligible layer is statistically tested as both a peak and a trough
before this shape filtering is applied. Thus the multiple-testing correction
does not operate only on visually selected candidate layers.

## Paired test statistic and null hypothesis

For each protein, define the center-versus-neighborhood contrasts

\[
D_{L,p}=y_{p,l}-\frac{1}{5}\sum_{j=l-5}^{l-1}y_{p,j},
\qquad
D_{R,p}=y_{p,l}-\frac{1}{5}\sum_{j=l+1}^{l+5}y_{p,j}.
\]

For a peak, separate one-sided one-sample t-tests across proteins test whether
\(E[D_L]>0\) and \(E[D_R]>0\). For a trough, they test whether both means are
negative, implemented by testing `-D_L` and `-D_R` for positive means.

For either signed contrast `D`, the component statistic is

\[
t=\frac{\bar D}{s_D/\sqrt{111}}, \qquad df=110.
\]

The joint test is an intersection-union test: both sides must support the
requested direction. Therefore,

- the peak null is \(E[D_L]\le0\) **or** \(E[D_R]\le0\), versus both being
  positive;
- the trough null is \(E[D_L]\ge0\) **or** \(E[D_R]\ge0\), versus both being
  negative.

The joint p-value is

\[
p_{\mathrm{joint}}=\max(p_L,p_R).
\]

There is no additional combined t-statistic. The ten neighbors are not tested
individually; each component compares the center with one five-layer average.
Proteins, rather than triangles, are the independent units.

## Benjamini-Hochberg correction

Benjamini-Hochberg (BH) controls the expected false-discovery proportion among
the reported discoveries across a predefined family of hypotheses. For `m`
ordered p-values \(p_{(1)}\le\cdots\le p_{(m)}\), the monotone adjusted values
are

\[
q_{(i)}=\min_{j\ge i}\left(1,\frac{m}{j}p_{(j)}\right).
\]

BH is applied separately to each of the two outcome metrics. Within one metric,
the family pools all models, focused categories, eligible layers, and both
peak/trough directions:

\[
\underbrace{4\times13}_{52\ \text{trajectories}}
\times\underbrace{38}_{\text{layers 5--42}}
\times\underbrace{2}_{\text{directions}}
=3{,}952\ \text{hypotheses}.
\]

An event is reported as a supported extremum only if it is shape-qualified and
has BH `q <= 0.05`. A q-value is not the probability that a particular event
is false; it is a search-level false-discovery adjustment.

For descriptive magnitude, `raw_prominence` is the smaller of the two signed
mean contrasts, and `standardized_prominence` is the smaller paired Cohen's
`dz`. Side-specific 95% t intervals are unadjusted; BH q-values determine
search-adjusted support.

## Interpretation limits

These results provide search-adjusted evidence for local extrema in these 111
proteins, not independent replication. Candidate selection can make reported
effect sizes optimistic. The analysis does not bootstrap proteins, test the
ten neighboring layers separately, or formally test whether two models or
categories have different extremum locations.

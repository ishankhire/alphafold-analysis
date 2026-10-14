# Pair-Representation Triangle-Dominance Analysis Plan

## 1. Goal and fixed scope

Measure channelwise triangle-inequality-style margins directly in OpenFold pair
representations, without fitting a probe or using a structural target. For each
pair-representation layer `0..47`, compare four fixed ways of turning the two
directional entries for each unordered residue pair into 128 scalar side values.

The pilot cohort is all 154 TM-score-filtered CAMEO proteins in `datasets.md`.
Use the same frozen protein set, sampled residue triples, transformation
definitions, channel order, and numerical rules at every layer. A missing or
invalid tensor produces an explicit incomplete row; a protein may not disappear
at one layer and reappear silently at another.

This is a descriptive latent-representation analysis. Pair channels are signed,
basis-dependent features rather than physical lengths. Positive margins in the
three nonnegative variants can be described as triangle-inequality-style
violations. The signed directional variant instead measures algebraic dominance
and can have a positive margin merely because one or more side values are
negative. Neither result establishes that the pair representation is a metric
space.

There is no training, target, PDB parsing, model selection, layer selection, or
sequence-separation filtering. Report all 48 layers as trajectories.

Cluster output:

```text
$HOME/csv_files/pair_representation_triangle_dominance/
```

Audited local copy:

```text
delta_files/pair_representation_triangle_dominance/
```

Local analysis output:

```text
delta_files/analysis/pair_representation_triangle_dominance/
```

## 2. Frozen triangle population and sampling

For a protein with `R` residues, the population contains all
`C(R, 3)` unordered triples. Represent each triple in canonical strict-lower
order as

\[
i>j>k.
\]

Sample uniformly without replacement from this population:

\[
n_{\mathrm{sample}}=\min\left(20{,}000, {R\choose3}\right).
\]

Proteins with no triples remain in the cohort with an explicit
`insufficient_residues` status. No triple is included or excluded using sequence
separation, pair-representation values, or any structural property.

Create the sample once before any layer job:

1. assign every canonical triple a deterministic integer rank;
2. derive a protein-specific seed from the protein ID, a fixed base seed, and a
   versioned domain string;
3. sample integer ranks without replacement using a named, versioned NumPy
   generator;
4. unrank them into `i,j,k`, sort by sampled rank for stable storage; and
5. save the actual sampled indices, per-protein offsets, algorithm metadata, and
   SHA-256 digests in a compressed NPZ plus a CSV manifest.

Saving the compact index arrays makes the sample authoritative and avoids
depending on future random-number-library behavior. Every layer and all four
variants load exactly these indices. The expected full-cohort sample is at most
`154 * 20,000 = 3,080,000` triples per layer.

## 3. Pair-side transformation variants

Let `Z[p,q,c]` be channel `c` of the pair representation and let `p>q` for an
unordered residue pair. Define the four variants exactly as follows.

| Variant | Side value for pair `{p,q}` and channel `c` |
|---|---|
| `directional_raw` | `Z[p,q,c]` |
| `directional_abs` | `abs(Z[p,q,c])` |
| `abs_then_sym` | `(abs(Z[p,q,c]) + abs(Z[q,p,c])) / 2` |
| `sym_then_abs` | `abs((Z[p,q,c] + Z[q,p,c]) / 2)` |

The first two use the repository's canonical strict-lower direction. Do not add
the opposite direction as a fifth variant. The distinction between
`abs_then_sym` and `sym_then_abs` is scientifically meaningful: the latter can
cancel directional entries with opposite signs, while the former cannot.

Load each `(R, R, 128)` pair tensor once per protein-layer. On-disk tensors may
remain `float32`, but convert gathered side blocks to `float64` before taking
absolute values, symmetrizing, calculating margins, normalizing, or accumulating
statistics. Reject nonfinite inputs rather than silently replacing them.

## 4. Channelwise margins and normalization

For one sampled triple, variant, and channel, let its three transformed side
values be `a`, `b`, and `d`. Define

\[
M=\max(a,b,d),
\qquad
m=2M-(a+b+d).
\]

Thus `m` is the largest side minus the sum of the other two. For example,
`(5,3,1)` gives `m=1`.

Use a common L1 scale for every variant:

\[
S=|a|+|b|+|d|,
\qquad
r=\begin{cases}
m/S,&S>0,\\
0,&S=0.
\end{cases}
\]

This reduces to division by `a+b+d` for the three nonnegative variants and
remains finite and bounded for `directional_raw`. All-zero channel triples have
zero signed and clipped normalized values and increment a zero-scale diagnostic
count.

Calculate four primary channelwise measurements:

| Metric | Definition |
|---|---|
| `signed_margin` | `m` |
| `positive_violation` | `max(m, 0)` |
| `signed_normalized_margin` | `r` |
| `positive_normalized_violation` | `max(r, 0)` |

Use the exact boundary `m > 0` for a positive violation; do not import the
Angstrom tolerance used by decoded-distance analyses because these values have
no physical unit. Also calculate `is_positive = 1[m > 0]` as a secondary
frequency diagnostic.

The words `raw_scale` and `normalized` distinguish scaling in filenames and
plots. Avoid calling an unnormalized result an "absolute magnitude," because
that phrase would be confused with the `directional_abs` transformation.

## 5. Triangle, protein, and channel reductions

### 5.1 Triangle totals

For each sampled triangle and representation variant, average each of the four
primary measurements over all 128 channels:

\[
T_g=\frac{1}{128}\sum_{c=0}^{127}g_c.
\]

Also calculate the fraction of channels with a positive margin. These transient
triangle totals are used only to form summaries and are never written as rows.

### 5.2 Per-protein summaries

For each protein-layer-variant-metric cell, summarize the sampled triangle
totals with:

- mean;
- population standard deviation (`ddof=0`);
- median;
- p90 and p99; and
- finite triangle count and support status.

The protein-level mean is the authoritative scalar used in layer trajectories.
Retain zero-valued clipped results rather than conditioning on violation.

For every protein-layer-variant-channel, also calculate the mean of each
channelwise primary measurement over sampled triangles, the positive-margin
fraction, and the zero-scale fraction. Save these compact sufficient summaries
in compressed layer NPZ shards, not as triangle rows.

### 5.3 Protein-equal layer summaries

For every layer, variant, and metric, aggregate the authoritative protein means
with equal weight per complete protein. Save the protein-equal mean, population
standard deviation, median, q25, q75, and attempted/finite protein counts.

Do not weight proteins by `C(R,3)` or by sampled-triangle count. Do not make a
sample-pooled trajectory a co-primary result. Proteins with fewer than 20,000
population triples still receive one equal protein weight when their summary is
finite.

For every layer, variant, channel, and channel metric, first calculate the
within-protein sampled-triangle summary and then apply the same protein-equal
aggregation. This preserves channel-level trajectories without letting longer
proteins dominate.

## 6. Exact output contract

```text
pair_representation_triangle_dominance/
  run_manifest.json
  per_protein_metrics.csv
  layer_summary.csv
  channel_summary.csv
  layer_status.csv
  runtime.csv
  metadata/
    cameo_cohort.csv
    sample_manifest.csv
    triangle_samples.npz
    sample_provenance.json
  layer_shards/
    per_protein_metrics_layer_00.csv
    per_protein_channel_metrics_layer_00.npz
    layer_status_00.csv
    runtime_layer_00.csv
    ...
    per_protein_metrics_layer_47.csv
    per_protein_channel_metrics_layer_47.npz
    layer_status_47.csv
    runtime_layer_47.csv
```

No pair-level, triangle-level, or channel-by-triangle measurement file is part
of the output contract.

### 6.1 Sampling metadata

`cameo_cohort.csv` has one row per one of the 154 requested CAMEO proteins and
records `protein_id`, `source`, manifest eligibility, expected residue count,
and cohort status.

`sample_manifest.csv` has one row per protein with principal columns:

```text
sample_version,protein_id,n_residues,n_population_triangles,
n_sampled_triangles,base_seed,derived_seed,rank_order,unranking_algorithm,
rng_algorithm,sample_offset_start,sample_offset_stop,sample_sha256,status
```

`triangle_samples.npz` stores non-object arrays loadable with
`allow_pickle=False`: protein identifiers in fixed-width Unicode, protein
offsets, sorted sampled ranks, and concatenated `tri_i`, `tri_j`, and `tri_k`
indices. `sample_provenance.json` records configuration, library versions,
manifest and NPZ hashes, and the complete-sample semantic digest.

### 6.2 Per-protein metrics

`per_protein_metrics.csv` and each layer CSV shard use one row per
protein-layer-variant-metric. A complete run contains
`154 * 48 * 4 * 5 = 147,840` rows: four primary metrics plus the secondary
`positive_channel_fraction` metric. Principal columns are:

```text
analysis_name,schema_version,layer,protein_id,source,
representation_variant,metric,n_residues,n_population_triangles,
n_sampled_triangles,sample_sha256,mean,std,median,p90,p99,
n_finite_triangles,zero_scale_channel_fraction,status,failure_reason
```

For `positive_channel_fraction`, distribution columns summarize the transient
per-triangle fraction of positive channels. For other rows they summarize the
corresponding 128-channel triangle total. The same zero-scale diagnostic may be
repeated across a protein-layer-variant's metric rows for a rectangular schema.

### 6.3 Channel shards and summaries

Each `per_protein_channel_metrics_layer_XX.npz` stores numeric arrays indexed by
the fixed protein, variant, channel, and metric orders declared in its shard
metadata. The authoritative arrays include:

```text
channel_metric_mean       shape (154, 4, 128, 4)
channel_positive_fraction shape (154, 4, 128)
channel_zero_scale_fraction shape (154, 4, 128)
protein_status_code       shape (154,)
```

`channel_summary.csv` has one row per layer-variant-channel-metric. With the
four primary metrics plus positive and zero-scale fractions, a complete file
contains `48 * 4 * 128 * 6 = 147,456` rows. Principal columns are:

```text
analysis_name,schema_version,layer,representation_variant,channel,metric,
protein_equal_mean,protein_std,protein_median,protein_q25,protein_q75,
n_proteins_attempted,n_proteins_finite,support_status
```

### 6.4 Layer summaries, status, and runtime

`layer_summary.csv` has one row per layer-variant-metric and the same
protein-equal summary fields as `channel_summary.csv`, omitting `channel`.

`layer_status.csv` has one row per layer and records expected/completed/failed
protein counts, expected variant and channel coverage, input and sample digests,
shard hashes, and completion status. `runtime.csv` has one row per layer plus
one preparation and one aggregation row, with start/end timestamps,
`elapsed_seconds`, requested resources, processed protein and sampled-triangle
counts, optional peak RSS, and status.

`run_manifest.json` freezes the analysis and schema versions, all formulas,
variant and metric orders, cohort and sampling hashes, layer range, precision,
sample cap, chunk size, software versions, output hashes, and known scientific
limitations.

## 7. Implementation layout and computation

Add cluster-ready files only under `scripts/cluster_scripts/`:

```text
scripts/cluster_scripts/pair_triangle_dominance/
  pair_triangle_dominance_common.py
  prepare_pair_triangle_samples.py
  prepare_pair_triangle_samples.sbatch
  pair_triangle_dominance_layer.py
  pair_triangle_dominance_layer.sbatch
  aggregate_pair_triangle_dominance.py
  aggregate_pair_triangle_dominance.sbatch
  tests/test_pair_triangle_dominance.py
scripts/multi_protein/
  analyze_pair_triangle_dominance.py
```

The local analyzer is explicitly local-only. It reads consolidated copied
results and writes layer trajectories for all 16 primary variant-metric
combinations, a positive-channel-frequency diagnostic, channel heatmaps, and
ranked channel tables. Plot protein-equal summaries and show the number of
finite proteins; do not silently interpolate incomplete layers.

The layer runner uses one Slurm array task per layer. For each protein:

1. validate the frozen sample record and tensor shape `(R, R, 128)`;
2. load the pair tensor once with plain `np.load(...)`;
3. process sampled triples in fixed 2,000-triangle chunks;
4. gather both directions for the three edges once per chunk;
5. derive the four variants, margins, normalizers, and reductions in `float64`;
6. update bounded per-protein distribution and channel-summary accumulators;
7. discard chunk arrays before moving to the next chunk; and
8. checkpoint the layer after each small protein batch.

Use atomic checkpoint replacement and atomic final shard promotion. A restarted
task verifies configuration, sample, cohort, and prior-row digests before
continuing. It must never mix partial rows from different samples or code
configurations. Consolidation validates all 48 final shards before writing
top-level outputs.

Chunk boundaries are computational only and must not change sample membership,
row order, or numerical definitions. Accumulate sums in `float64`; calculate
per-triangle distribution summaries from the complete 20,000-or-smaller vector
for that protein rather than averaging chunk quantiles.

## 8. Delta resources and execution policy

Submit the full array after authorization, following the
[cluster guide](../cluster/README.md); no separate timing pilot is required.
Freeze the 20,000-triplet cap before production and
do not adapt it by layer, protein length, observed values, or task timing.

Use these initial launcher requests, based on the closest completed workflows
in `docs/plans/suggested_sbatch_memory.md` and
`docs/plans/suggested_sbatch_walltime.md`:

| Stage | Task shape | Initial request |
|---|---|---|
| Sample preparation | all 154 proteins | `1` CPU, `4g`, `00:30:00` |
| Layer analysis | array `0-47%12` | `4` CPUs, `8g`, `01:00:00` per task |
| Aggregation | all 48 shards | `4` CPUs, `8g`, `00:30:00` |

The layer target is about 15 minutes, with up to about 30 minutes acceptable;
the one-hour request supplies failure and filesystem headroom without changing
the fixed sample. Four CPUs support bounded parallel work across the four
transformations; do not request them unless the implementation actually uses
them. If the implementation remains single-threaded, lower the request to one
CPU before submission.

The complete production array, not a calibration job, supplies the first
workflow-specific elapsed and MaxRSS measurements. After it finishes, add its
representative successful-task maximum to both shared resource guides before a
rerun. Do not resize from failed, cached, restart-skipped, or incomplete tasks.

The current Delta [running-jobs documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/running_jobs.html)
supports CPU Slurm arrays, reports a 30-minute default walltime when no explicit
time is provided, and exposes `SLURM_ARRAY_TASK_ID` and `SLURM_SUBMIT_DIR`. The
launcher must request the `cpu` partition and an explicit time. Pair tensors
remain on `/work/hdd`; scripts and compact outputs use the established HOME
layout. Following Delta's [data-management guidance](https://docs.ncsa.illinois.edu/systems/delta/en/latest/user_guide/data_mgmt.html),
avoid repeated small-file I/O against HOME or `/work/hdd`; node-local `/tmp`
may hold transient checkpoints, but every needed result must be copied to the
final output before the task exits.

Every sbatch file must include:

```text
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=ikhire3@gatech.edu
```

Resolve entrypoints with:

```bash
SCRIPT_DIR=${SLURM_SUBMIT_DIR:-$(pwd)}
srun python3 -u "$SCRIPT_DIR/<entrypoint>.py"
```

Deploy all cluster files flat to `$HOME/openfold_interpretability_canonical`,
activate `~/protein_venv`, set `CSV_ROOT=$HOME/csv_files`, and submit from the
deployment directory. Implementation and tests do not themselves authorize
deployment or job submission.

## 9. Validation and acceptance criteria

- Freeze exactly the 154 eligible CAMEO proteins from the canonical manifest
  and require all 48 layers for completion.
- Verify every sampled triple satisfies `i>j>k`, lies in range, is unique within
  its protein, and has the exact expected sample count.
- Verify sample indices and digests are identical across layers and variants.
- Test `(5,3,1)` as signed margin `1` and normalized margin `1/9`; test
  nonviolating, equality, all-zero, mixed-sign, and all-negative examples.
- Test directional indexing and the difference between `abs_then_sym` and
  `sym_then_abs`, including opposite-sign directional values that cancel only
  in the latter.
- Verify `positive_violation = max(signed_margin, 0)` and its normalized analogue
  elementwise; clipped outputs must be nonnegative.
- Require finite normalized values, exact zero behavior for zero scale, and
  expected bounds within floating-point tolerance.
- Verify channel means reproduce each protein's mean triangle total within
  tolerance; averaging over channels and triangles must agree by linearity.
- Verify layer summaries are arithmetic means of complete protein-level means,
  with no triangle-count weighting.
- Require exact row counts, unique primary keys, declared array shapes, matching
  shard/configuration hashes, and explicit failure rows.
- Save no measured pair, triangle, or channel-by-triangle rows.
- Test restart safety, corrupted-sample rejection, missing tensor handling,
  nonfinite tensor rejection, aggregation failure on incomplete layers, and
  deterministic output ordering.
- Run focused tests with the local `./.venv` and apply CSV/JSON/NPZ round-trip
  checks before committing implementation.

## 10. Reporting order and interpretation

Report the protein-equal 48-layer trajectories in this order:

1. positive normalized violation for all four variants;
2. positive raw-scale violation;
3. signed normalized margin;
4. signed raw-scale margin;
5. positive-channel frequency; and
6. channel-resolved heatmaps and the channels driving major extrema.

Interpret agreement between `abs_then_sym` and `sym_then_abs` as evidence that
directional sign cancellation is not driving the result; divergence between
them localizes that sensitivity. Treat the two canonical-direction variants as
residue-order-dependent diagnostics. Discuss trajectories and channel
concentration, but do not assign physical distance units, claim rotation
invariance, or describe the pair representation as an actual learned metric.

## 11. Frozen pre-implementation decisions

No question blocks implementation. The cohort, 48-layer coverage, four
transformations, strict-lower direction, L1 normalization, exact zero rule,
20,000 uniform samples per protein, indiscriminate triple population,
protein-equal aggregation, channel-level summaries, compact-output policy, and
no-calibration decision are fixed. Any change must be made before production,
versioned in the manifest, and applied uniformly to all layers and proteins.

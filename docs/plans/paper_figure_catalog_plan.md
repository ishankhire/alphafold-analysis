# Paper Figure Catalog and Analysis Plan

## 1. Purpose

This plan defines a provisional catalog of 20 separately numbered main-paper
figures and 30 separately numbered supplemental figures. The figures can be
combined into multi-panel layouts later. The immediate goal is to produce a
thorough and interpretable summary of how final-structure information becomes
accessible across the 48 OpenFold Evoformer layers, while clearly separating
figures that can be made from current artifacts from analyses that require new
prediction replay, annotation, or cluster work.

The primary scientific narrative is:

1. final predicted residue-pair distances become increasingly accessible;
2. shared linear, gated linear, nonlinear, and protein-specific readouts expose
   different aspects of that information;
3. pairwise accuracy and global geometric consistency do not necessarily emerge
   together;
4. sequence scale, amino-acid identity, protein architecture, and secondary
   structure can reveal which relationships are refined at different stages;
5. layer-to-layer gains and reversals are scientifically informative and should
   not be hidden by reporting only layer 0 and layer 47.

This is a figure-content plan, not an implementation specification. The output
contracts below are included so later scripts have unambiguous targets.

## 2. Interpretation guardrails

- Each probe predicts C-alpha distances from a frozen pair representation at
  one layer. Probe accuracy measures accessibility to that readout, not that
  OpenFold itself implements the fitted regression.
- The target is OpenFold's final predicted PDB. The reconstruction TM-score is
  therefore the TM-score between a probe-derived reconstruction and OpenFold's
  final prediction. It must not be confused with the experimental-structure
  TM-score used to select the dataset.
- Shared OLS, sequence-gated OLS, AA-pair-gated OLS, and shared XGBoost are
  evaluated on locked held-out proteins. Per-protein OLS is evaluated on
  held-out pairs within a protein used to fit that protein's probe. Its curve is
  a protein-specific accessibility ceiling with a different generalization
  scope, not an equal held-out-protein competitor.
- Complete-matrix geometry metrics require predictions for every pair. The
  per-protein OLS geometry rows therefore use `all_pairs` and include the
  protein's fitting pairs as well as its held-out pairs. Geometry captions must
  disclose this additional in-sample component.
- CAMEO and ECOD remain one analysis dataset except for the explicitly named
  ECOD-category analyses. ECOD category figures exclude CAMEO because
  `ecod_protein_categories.csv` does not annotate CAMEO proteins.
- AA-pair x sequence-separation gating is excluded throughout unless a future
  question explicitly requires it.
- Whole-protein primary summaries use arithmetic means over proteins. Pooled
  pair metrics are retained for subgroup analyses and sensitivity figures.
- An amino-acid identity pair such as CYS-CYS is not proof of a disulfide bond.
  Analyses of actual disulfides, salt bridges, hydrogen bonds, or other
  interactions require structural annotations and belong in the additional
  analysis tier.

## 3. Analysis readiness labels

Every proposed figure receives one of four labels.

| Label | Meaning |
|---|---|
| **Ready** | Existing canonical CSVs contain the plotted quantities; only local tabulation and plotting are needed. |
| **Local derived** | Existing canonical CSVs are sufficient, but a new local aggregation, bootstrap, matching, or statistical summary is needed. |
| **Pending run** | The necessary workflow already exists and is currently running or awaiting copied-back results. |
| **New computation** | New prediction replay, pair-level statistics, structural annotation, or cluster computation is required. |

Figures should not silently fall back to historical result families when a
canonical family is incomplete. A figure manifest must record its source paths,
source hashes, evaluation scope, and readiness status.

## 4. Source result families

The five primary distance-probe families are:

| Display name | Canonical local source | Primary paper scope |
|---|---|---|
| Shared linear | `delta_files/general_base256_combined_canonical/general_base256_combined_canonical/` | `protein_split=test`, `evaluation_scope=all_pairs` |
| Per-protein linear | `delta_files/per_protein_base256_combined_canonical/` | `protein_split=test`, `evaluation_scope=heldout_pairs` for five-model comparisons; all proteins' held-out pairs for protein-specific descriptive analyses |
| XGBoost | `delta_files/shared_xgboost_base256_combined/` | `protein_split=test`, `evaluation_scope=all_pairs` |
| Sequence gated | `delta_files/gated_linear_probes/consolidated/analysis_1_sequence_gated/` | `protein_split=test`, `evaluation_scope=all_pairs` |
| AA gated | `delta_files/gated_linear_probes/consolidated/analysis_2_aa_pair_gated/` | `protein_split=test`, `evaluation_scope=all_pairs` |

The secondary-structure source is the four-model, 48-layer workflow described
in `docs/plans/secondary_structure_model_comparison_plan.md`. The existing
six-layer shared-OLS diagnostic can support preliminary figures, but it is not
the final source once the full comparison is available.

## 5. Output organization and common files

Question-specific derived tables belong in separate folders under
`delta_files/analysis/`:

```text
delta_files/analysis/
  comparing_variants/
  geometry_consistency/
  sequence_gated/
  amino_acid_gated/
  per_protein_refinement/
  secondary_structure/
  paper_figures/
    main/
    supplemental/
    figure_manifest.csv
```

Each final numbered figure is written as both PNG and PDF. Filenames use
`main_figure_XX_<short_name>.{png,pdf}` or
`supp_figure_SXX_<short_name>.{png,pdf}`. Question-specific CSVs remain in their
analysis folders instead of being copied beside every figure.

`figure_manifest.csv` has one row per numbered figure. Principal columns are:

```text
figure_id, figure_set, title, readiness, evaluation_population,
evaluation_scope, metric, source_files, plot_data_file, png_file, pdf_file,
script_file, generated_at, source_digest, notes
```

The common visual conventions are:

- layer is always shown as 0 through 47, with integer layer labels and the same
  horizontal direction;
- model names, colors, and line styles are fixed across every figure;
- mean per-protein R2, TM-score, and top-three eigenvalue fraction use a y-axis
  from 0 to 1; negative-eigenvalue mass is shown on its own nonnegative axis;
- MAE and violation magnitude use angstroms;
- triangle violation fraction is labeled as a fraction or percentage and may
  use a focused range because all values occupy a small part of 0 to 1;
- uncertainty intervals use a paired protein bootstrap when the underlying
  rows are per-protein, resampling proteins rather than residue pairs;
- primary figures show the locked test population unless a different scope is
  named directly in the title or caption;
- per-protein OLS always uses a visibly distinct line style and caption note
  about its within-protein evaluation scope;
- curves are not smoothed in the authoritative plot data. A lightly smoothed
  guide may be overlaid only if the unsmoothed trajectory remains visible.

## 6. Main-paper figure catalog

### Main Figure 1 — Mean per-protein R2 across layers

**Question:** How much final-distance variation is accessible to each readout?

Plot five layer trajectories for the arithmetic mean of finite per-protein R2.
Use the locked 111 test proteins and `all_pairs` for the four shared models;
use `heldout_pairs` for per-protein OLS. Show paired protein-bootstrap 95%
intervals. Fix the y-axis to 0--1.

**Readiness:** Local derived.

**Plot data:**
`delta_files/analysis/comparing_variants/mean_per_protein_trajectories.csv`, one
row per model, layer, metric, protein split, and evaluation scope. Principal
columns: `model`, `layer`, `metric`, `mean`, `median`, `ci_low`, `ci_high`,
`n_proteins`, `protein_split`, `evaluation_scope`, `generalization_note`.

**Figure:** `paper_figures/main/main_figure_01_mean_per_protein_r2.*`.

### Main Figure 2 — Mean per-protein MAE across layers

**Question:** What is the typical absolute distance error for a protein?

Use the same population, model ordering, and uncertainty procedure as Figure 1.
Show MAE in angstroms with a zero origin and a shared y-range suitable for all
five curves.

**Readiness:** Local derived.

**Plot data:** the `metric=mae` rows of
`mean_per_protein_trajectories.csv`.

**Figure:** `paper_figures/main/main_figure_02_mean_per_protein_mae.*`.

### Main Figure 3 — Mean reconstruction TM-score across layers

**Question:** When does the decoded distance matrix support a globally similar
three-dimensional reconstruction?

Plot mean per-protein reconstruction TM-score for all five models. Use the
geometry rows corresponding to complete `all_pairs` predictions on the 111
master-test proteins. Fix the y-axis to 0--1. For per-protein OLS, explicitly
note that the complete matrix contains both its within-protein fitting and
held-out pairs. The caption must also say that the reconstruction reference is
OpenFold's final predicted structure.

**Readiness:** Ready.

**Plot data:**
`delta_files/analysis/comparing_variants/geometry_model_trajectories.csv`, one
row per model, layer, geometry metric, protein split, and evaluation scope.
Principal columns: `model`, `layer`, `geometry_metric`, `mean`, `std`, `median`,
`q25`, `q75`, `n_proteins`, `protein_split`, `evaluation_scope`.

**Figure:** `paper_figures/main/main_figure_03_tm_score.*`.

### Main Figure 4 — MDS reconstruction stress across layers

**Question:** How closely do distances in the three-dimensional classical-MDS
reconstruction reproduce the decoded distances?

Plot mean normalized MDS stress for all five models on a nonnegative y-axis.
Define stress as the RMS difference between decoded pair distances and distances
in the 3D MDS reconstruction, normalized by the RMS decoded distance. Lower is
better and zero denotes exact reconstruction.

**Readiness:** Ready.

**Plot data:** `geometry_model_trajectories.csv`, `geometry_metric=stress`.

**Figure:** `paper_figures/main/main_figure_04_top3_eigenvalue_fraction.*`.

### Main Figure 5 — Triangle violation fraction across layers

**Question:** When do independently decoded pair distances become locally
metric-consistent?

Plot the mean per-protein fraction of exact triangles violating the triangle
inequality. Use a focused fraction or percentage y-axis while retaining zero.
Emphasize nonmonotonicity rather than presuming that every layer improves.

**Readiness:** Ready.

**Plot data:** `geometry_model_trajectories.csv`,
`geometry_metric=triangle_violation_fraction`.

**Figure:** `paper_figures/main/main_figure_05_triangle_violation_fraction.*`.

### Main Figure 6 — Mean triangle violation magnitude across layers

**Question:** When violations occur, how severe are they?

For each protein and layer with at least one violation, calculate the
conditional mean positive magnitude
`triangle_mean_violation / triangle_violation_fraction`, then average that
quantity over proteins and plot it in angstroms for all five models. The
division is valid because the stored `triangle_mean_violation` is total
positive violation divided by all triangles, while the fraction supplies the
violating-triangle proportion. This figure complements Figure 5 because a
model may reduce the number and conditional severity of violations differently.
Do not divide already aggregated summary means; derive the conditional value
from each row of `geometry_metrics.csv` first.

**Readiness:** Local derived.

**Plot data:** `geometry_model_trajectories.csv`,
`geometry_metric=conditional_triangle_violation_magnitude`. Retain the stored
all-triangle `triangle_mean_violation` as a separate row for supplemental use.

**Figure:** `paper_figures/main/main_figure_06_triangle_violation_magnitude.*`.

### Main Figure 7 — Layer-to-layer R2 gain

**Question:** At which layers does distance accessibility improve, stall, or
reverse?

For each model, plot
`mean_per_protein_r2(layer) - mean_per_protein_r2(layer - 1)` for layers 1--47,
with a horizontal zero line. Do not cumulative-smooth the authoritative curve.
This figure is intended to reveal refinement bursts and mid-trunk reversals.

**Readiness:** Local derived.

**Plot data:**
`delta_files/analysis/comparing_variants/layerwise_gains.csv`, one row per
model, transition, metric, and scope. Principal columns: `model`,
`from_layer`, `to_layer`, `metric`, `value_from`, `value_to`, `gain`,
`ci_low`, `ci_high`, `n_paired_proteins`, `evaluation_scope`.

**Figure:** `paper_figures/main/main_figure_07_layerwise_r2_gain.*`.

### Main Figure 8 — Decodability versus geometric consistency

**Question:** Does accurate distance decoding precede geometrically coherent
distance decoding?

Trace each model through a phase plot with mean per-protein R2 on the x-axis
and triangle violation fraction on the y-axis. Use `all_pairs` R2 for this
figure so the accuracy and geometry coordinates refer to the same complete
matrix; for per-protein OLS this is an in-sample/held-out mixture and is labeled
as such. Connect consecutive layers and encode layer by a common sequential
color scale. Label at least layers 0, 10, 20, 30, 40, and 47. A sensitivity
version using TM-score and mean violation magnitude belongs in the supplement.

**Readiness:** Local derived.

**Plot data:**
`delta_files/analysis/geometry_consistency/decodability_consistency.csv`, one
row per model, layer, metric pair, and scope. Principal columns: `model`,
`layer`, `accuracy_metric`, `accuracy_value`, `consistency_metric`,
`consistency_value`, `protein_split`, `evaluation_scope`, `n_proteins`.

**Figure:** `paper_figures/main/main_figure_08_decodability_consistency.*`.

### Main Figure 9 — Sequence-separation MAE trajectories

**Question:** Are local and long-range relationships refined at different
stages, and does the answer depend on the readout?

Show pooled test MAE for all six canonical sequence-separation bins (`1`,
`2-4`, `5-9`, `10-19`, `20-49`, `50+`) across all 48 layers and all five
models. This is intentionally information-dense; the first version may use all
30 trajectories on one graph, while the later paper layout may facet by model
or separation bin without changing the figure number or data.

Use MAE rather than R2 as the primary measure because separation 1 has very
little target variance and consequently unstable R2. Per-protein OLS uses its
held-out-pair subgroup rows; the other models use all test pairs.

**Readiness:** Ready.

**Plot data:**
`delta_files/analysis/sequence_gated/sequence_separation_trajectories.csv`, one
row per model, layer, sequence bin, and scope. Principal columns: `model`,
`layer`, `sequence_bin`, `n_proteins`, `n_pairs`, `pooled_r2`, `pooled_mae`,
`pooled_rmse`, `pooled_bias`, `protein_split`, `evaluation_scope`,
`support_status`.

**Figure:** `paper_figures/main/main_figure_09_sequence_separation_mae.*`.

### Main Figure 10 — Sequence-separation baselines and model error

**Question:** How much of each bin's performance is explained by its average
distance alone?

For each of the six bins, compare model MAE with a leak-free baseline that
predicts the protein-equal development-set mean distance for that bin. Plot raw
MAE in angstroms; do not require a fraction-of-baseline-improvement metric.
Baseline values are constant over layer and must be repeated only for plotting,
not re-estimated per layer. Show all five models, with the baseline as a strong
neutral reference.

The canonical subgroup CSVs do not store bin-specific baseline errors, so this
figure requires a small canonical replay or sufficient-statistic aggregation
on the fixed development/test split. It must not estimate the mean from test
targets.

**Readiness:** New computation.

**Plot data:**
`delta_files/analysis/sequence_gated/sequence_separation_baselines.csv`, one row
per sequence bin, evaluation split, and baseline type. Principal columns:
`sequence_bin`, `baseline_type`, `fit_protein_count`, `evaluation_protein_count`,
`fit_pair_count`, `evaluation_pair_count`, `development_mean_distance`,
`pooled_r2`, `pooled_mae`, `pooled_rmse`, `pooled_bias`, `split_digest`.

The plotting join is saved as
`sequence_separation_model_baseline_comparison.csv`, one row per model, layer,
sequence bin, and comparison type.

**Figure:** `paper_figures/main/main_figure_10_sequence_separation_baselines.*`.

### Main Figure 11 — All 210 amino-acid-pair R2 trajectories

**Question:** Do different amino-acid pairs become linearly decodable at
different stages?

Plot all 210 unordered standard amino-acid-pair pooled R2 trajectories on one
graph using the AA-gated model. Use thin, partially transparent lines and a
0--1 y-axis. Highlight only a small, predeclared set in the legend, such as
CYS-CYS, GLY-GLY, PRO-PRO, opposite-charge pairs, aromatic-aromatic pairs, and
the median trajectory; all 210 curves remain visible.

AA-gated OLS is the primary decoder because this question allows an
identity-specific linear readout and its current subgroup R2 range remains
interpretable on the requested 0--1 scale. Shared OLS, XGBoost, sequence-gated,
and per-protein sensitivity versions appear in the supplement.

**Readiness:** Ready.

**Plot data:**
`delta_files/analysis/amino_acid_gated/amino_acid_pair_trajectories.csv`, one
row per model, layer, unordered AA pair, and scope. Principal columns: `model`,
`layer`, `aa_pair`, `aa_1`, `aa_2`, `n_proteins`, `n_pairs`, `pooled_r2`,
`pooled_mae`, `pooled_rmse`, `pooled_bias`, `support_status`, `protein_split`,
`evaluation_scope`.

**Figure:** `paper_figures/main/main_figure_11_aa_pair_r2_210.*`.

### Main Figure 12 — All 210 amino-acid-pair MAE trajectories

**Question:** Are apparent AA-pair differences in explained variance also
visible in absolute angstrom error?

Use the same 210 AA-gated trajectories and highlighting rules as Figure 11,
but plot pooled MAE in angstroms. The R2 and MAE figures must be interpreted
together because AA pairs differ in target-distance distributions.

**Readiness:** Ready.

**Plot data:** the `pooled_mae` column of
`amino_acid_pair_trajectories.csv`.

**Figure:** `paper_figures/main/main_figure_12_aa_pair_mae_210.*`.

### Main Figure 13 — AA-pair refinement-stage distribution

**Question:** Which AA pairs become accessible early, and which require later
layers?

For each of 210 AA pairs, calculate an R2 onset summary using the unsmoothed
AA-gated trajectory:

- `peak_r2` is the maximum over layers 0--47;
- `L50` is the earliest layer reaching 50% of the improvement from layer 0 to
  `peak_r2`;
- `L90` is the earliest layer reaching 90% of that improvement;
- `peak_layer` is the earliest layer attaining `peak_r2`;
- `late_reversal` is `peak_r2 - r2_layer47`.

Plot all pairs sorted by `L50`, with `L90` and `peak_layer` as secondary marks.
Pairs with negligible total improvement are flagged rather than assigned an
overinterpreted onset. This operationalizes "refinement stage" without forcing
the raw trajectory to be monotone.

**Readiness:** Local derived.

**Plot data:**
`delta_files/analysis/amino_acid_gated/amino_acid_pair_refinement_timing.csv`,
one row per model and AA pair. Principal columns: `model`, `aa_pair`,
`r2_layer0`, `r2_layer47`, `peak_r2`, `peak_layer`, `total_peak_gain`, `L50`,
`L90`, `late_reversal`, `n_pairs_min`, `n_proteins_min`, `timing_status`.

**Figure:** `paper_figures/main/main_figure_13_aa_pair_refinement_timing.*`.

### Main Figure 14 — Where AA gating helps over a shared linear decoder

**Question:** Does amino-acid identity require a specialized readout for only
particular residue pairs or stages?

For each AA pair and layer, plot
`pooled_mae(shared OLS) - pooled_mae(AA-gated OLS)`. Positive values indicate
an AA-gating advantage. Use a 210-pair by 48-layer heatmap, ordering pairs by
the timing result from Figure 13. A secondary contour or annotation may show
the corresponding R2 difference.

This is more diagnostic than comparing only the whole-protein AA-gated and
shared curves, which are currently very similar.

**Readiness:** Local derived.

**Plot data:**
`delta_files/analysis/amino_acid_gated/aa_gating_pairwise_gain.csv`, one row per
AA pair and layer. Principal columns: `layer`, `aa_pair`, `shared_pooled_r2`,
`aa_gated_pooled_r2`, `delta_r2`, `shared_pooled_mae`,
`aa_gated_pooled_mae`, `mae_reduction`, `n_pairs_shared`,
`n_pairs_aa_gated`, `support_status`.

**Figure:** `paper_figures/main/main_figure_14_aa_gating_pairwise_gain.*`.

### Main Figure 15 — Accuracy by true spatial-distance bin

**Question:** Does local contact geometry become accurate before global
dimensions?

Plot pooled MAE across layers for the five canonical true-distance bins
`0-5`, `5-10`, `10-20`, `20-40`, and `40+` angstroms and all five models.
True distance is used only for evaluation stratification. The first version may
place all trajectories on one graph; later faceting is allowed.

**Readiness:** Ready.

**Plot data:**
`delta_files/analysis/comparing_variants/spatial_distance_trajectories.csv`, one
row per model, layer, distance bin, and evaluation scope. Principal columns:
`model`, `layer`, `distance_bin`, `n_proteins`, `n_pairs`, `pooled_r2`,
`pooled_mae`, `pooled_rmse`, `pooled_bias`, `support_status`,
`evaluation_scope`.

**Figure:** `paper_figures/main/main_figure_15_spatial_distance_mae.*`.

### Main Figure 16 — Per-protein refinement by broad ECOD category

**Question:** Are all-alpha, all-beta, alpha/beta, and alpha+beta proteins
refined at different stages?

Use per-protein OLS held-out-pair trajectories and all usable ECOD proteins,
not just master-test proteins, because every protein supplies an independent
within-protein held-out-pair evaluation. Join protein IDs to
`ecod_protein_categories.csv`, zero-padding `prot` to nine digits. Plot the
mean per-protein R2 trajectory for the four balanced broad categories:
`alpha`, `beta`, `alpha_beta`, and `alpha_plus_beta`.

Show raw category means in the main curve and report length-adjusted category
effects alongside them. Adjustment should use continuous log length and should
not use the `tm_score` filtering variable as a biological predictor in the
primary analysis.

**Readiness:** Local derived.

**Plot data:**
`delta_files/analysis/per_protein_refinement/ecod_category_trajectories.csv`,
one row per category, layer, metric, and adjustment type. Principal columns:
`category`, `layer`, `metric`, `adjustment`, `mean`, `ci_low`, `ci_high`,
`n_proteins`, `mean_length`, `evaluation_scope`.

Length-adjusted inferential summaries are saved to
`ecod_category_effects.csv`, one row per layer, metric, and category contrast.
Principal columns: `layer`, `metric`, `reference_category`, `category`,
`effect`, `standard_error`, `ci_low`, `ci_high`, `p_value`,
`multiple_test_method`, `adjustment_formula`, `n_proteins`.

**Figure:** `paper_figures/main/main_figure_16_ecod_category_refinement.*`.

### Main Figure 17 — Per-protein refinement by ECOD architecture

**Question:** Do finer structural architectures show distinct refinement
timing within broad secondary-structure classes?

Use `primary_arch` from `ecod_protein_categories.csv`, which supplies 12
balanced or near-balanced architecture labels. Plot length-adjusted mean
per-protein R2 trajectories. Preserve architecture names rather than collapsing
them after observing results. Because 12 curves are dense, direct labels at
layer 47 or a paired trajectory/timing view are preferred to a large legend.

**Readiness:** Local derived.

**Plot data:**
`delta_files/analysis/per_protein_refinement/ecod_architecture_trajectories.csv`,
one row per architecture, layer, metric, and adjustment type. Principal
columns mirror `ecod_category_trajectories.csv` with `primary_arch` replacing
`category`.

The corresponding model table is `ecod_architecture_effects.csv`, one row per
layer, metric, and architecture contrast, with the same inferential fields as
`ecod_category_effects.csv`.

**Figure:** `paper_figures/main/main_figure_17_ecod_architecture_refinement.*`.

### Main Figure 18 — Protein length and refinement timing

**Question:** Are longer proteins refined later or simply harder at every
layer?

Use all usable proteins in the per-protein OLS family. Derive per-protein
`L50`, `L90`, peak layer, layer-47 score, and total gain for both R2 and MAE.
Plot continuous residue count against `L50`, with a fitted trend and points
colored by data source only as a technical diagnostic. A companion layer-bin
trajectory may be added later as a panel, but the numbered figure remains the
continuous analysis.

Report both the unadjusted association and an ECOD-only sensitivity adjusted
for broad category. Do not interpret CAMEO versus ECOD source as a biological
effect.

**Readiness:** Local derived.

**Plot data:**
`delta_files/analysis/per_protein_refinement/protein_refinement_timing.csv`, one
row per protein, model, evaluation scope, and metric. Principal columns:
`protein_id`, `source`, `n_residues`, `model`, `metric`, `value_layer0`,
`value_layer47`, `best_value`, `peak_layer`, `total_gain`, `L50`, `L90`,
`late_reversal`, `timing_status`, `ecod_category`, `primary_arch`.

Associations are saved in `protein_refinement_predictors.csv`, one row per
outcome, predictor, population, and model. Principal columns: `outcome`,
`predictor`, `population`, `model`, `coefficient`, `standard_error`, `ci_low`,
`ci_high`, `p_value`, `adjustment_formula`, `n_proteins`.

**Figure:** `paper_figures/main/main_figure_18_length_refinement_timing.*`.

### Main Figure 19 — Alpha-helix emergence across readouts

**Question:** At what layer does final-structure alpha-helical geometry become
recognizable, and does it persist?

Use the running four-model secondary-structure comparison. Plot helix-versus-
strand AUROC across all 48 layers for shared OLS, sequence-gated OLS,
per-protein OLS, and XGBoost. Include the true-structure ceiling as a reference.
The caption must state that DSSP labels known helix locations and the decoded
distance pattern is being tested at those locations; this is not an unlabeled
de novo helix classifier.

The final version should also report the fraction of helix segments that cross
the frozen threshold and remain above it. That quantity can become a later
panel without changing the numbered figure.

**Readiness:** Pending run. A six-layer shared-OLS preliminary result is ready.

**Plot data:**
`delta_files/analysis/secondary_structure/helix_emergence_trajectories.csv`, one
row per model, layer, evaluation scope, and helix metric. Principal columns:
`model`, `layer`, `metric`, `value`, `ci_low`, `ci_high`, `n_proteins`,
`n_helix_residues`, `n_strand_residues`, `threshold_source`,
`evaluation_scope`.

**Figure:** `paper_figures/main/main_figure_19_helix_emergence.*`.

### Main Figure 20 — Beta-sheet adjacency and register emergence

**Question:** Does the representation place beta strands near one another
before resolving their residue-level register?

Use the same four-model comparison. Plot true bridge-partner discrimination
against matched random pairs, shift-1 controls, and shift-2 controls across all
48 layers. Preserve parallel and antiparallel results as separate line styles
or later panels. Include the true-distance ceiling so failure to resolve
register can be separated from readout limitations.

**Readiness:** Pending run. A six-layer shared-OLS preliminary result is ready.

**Plot data:**
`delta_files/analysis/secondary_structure/beta_register_trajectories.csv`, one
row per model, layer, sheet orientation, and control. Principal columns:
`model`, `layer`, `sheet_orientation`, `control_type`, `n_bridges`,
`mean_partner_distance`, `mean_control_distance`, `margin`, `auroc`,
`paired_correct_fraction`, `ci_low`, `ci_high`, `evaluation_scope`.

**Figure:** `paper_figures/main/main_figure_20_beta_register_emergence.*`.

## 7. Supplemental figure catalog

The first supplemental catalog contains 30 separately numbered figures. These
favor robustness, complete metric coverage, and alternative readouts rather
than introducing unrelated stories.

| ID | Figure content | Readiness | Primary data |
|---|---|---|---|
| S01 | Pair-pooled whole-protein R2 for the five models | Ready | canonical `pooled_metrics.csv` files |
| S02 | Pair-pooled whole-protein MAE for the five models | Ready | canonical `pooled_metrics.csv` files |
| S03 | Mean per-protein RMSE for the five models | Local derived | canonical `per_protein.csv` files |
| S04 | Mean per-protein signed bias for the five models | Local derived | canonical `per_protein.csv` files |
| S05 | Per-protein R2 distributions at layers 0, 10, 20, 30, 40, 47 | Local derived | canonical `per_protein.csv` files |
| S06 | Per-protein MAE distributions at the six landmark layers | Local derived | canonical `per_protein.csv` files |
| S07 | Per-protein reconstruction TM-score distributions at landmark layers | Ready | canonical `geometry_metrics.csv` files |
| S08 | MDS stress trajectories | Ready | canonical `geometry_summary.csv` files |
| S09 | Negative-eigenvalue fraction trajectories | Ready | canonical `geometry_summary.csv` files |
| S10 | Negative predicted-distance count or fraction across layers | Ready | canonical geometry files |
| S11 | Triangle violation severity fractions: 0--5, 5--10, and 10+ angstroms | Ready | canonical `geometry_summary.csv` files |
| S12 | Maximum triangle violation magnitude across layers | Ready | canonical `geometry_summary.csv` files |
| S13 | Layer-to-layer mean-per-protein MAE improvement | Local derived | `layerwise_gains.csv` |
| S14 | Layer-to-layer reconstruction TM-score gain | Local derived | geometry trajectories |
| S15 | Sequence-separation pooled R2, with explicit low-variance warning | Ready | sequence subgroup CSVs |
| S16 | Sequence-separation pooled RMSE | Ready | sequence subgroup CSVs |
| S17 | Sequence-separation pooled signed bias | Ready | sequence subgroup CSVs |
| S18 | Full six-bin model-versus-development-mean baseline comparison | New computation | Figure 10 baseline table |
| S19 | All 210 AA-pair R2 trajectories for shared OLS | Ready | AA subgroup CSV |
| S20 | All 210 AA-pair MAE trajectories for shared OLS | Ready | AA subgroup CSV |
| S21 | All 210 AA-pair R2 trajectories for XGBoost | Ready | AA subgroup CSV |
| S22 | All 210 AA-pair MAE trajectories for XGBoost | Ready | AA subgroup CSV |
| S23 | All 210 AA-pair R2 trajectories for sequence-gated OLS | Ready | AA subgroup CSV |
| S24 | All 210 AA-pair MAE trajectories for sequence-gated OLS | Ready | AA subgroup CSV |
| S25 | All 210 AA-pair R2 trajectories for per-protein OLS | Ready | AA subgroup CSV; held-out-pair scope |
| S26 | All 210 AA-pair MAE trajectories for per-protein OLS | Ready | AA subgroup CSV; held-out-pair scope |
| S27 | 20-by-20 AA-pair matrix at layers 0, 20, and 47 | Local derived | `amino_acid_pair_trajectories.csv` |
| S28 | True spatial-distance-bin R2 and bias sensitivity | Ready | spatial-distance subgroup CSVs |
| S29 | Per-protein trajectory heatmap sorted by R2 `L50`, with ECOD labels where available | Local derived | `protein_refinement_timing.csv` plus per-protein CSV |
| S30 | Secondary-structure calibration, persistence, and failure diagnostics | Pending run | secondary-structure comparison outputs |

Supplemental files use the same plot-data tables as the main figures when
possible. Any supplemental-only derived table must document row granularity and
principal fields in the implementing script's module documentation and in
`figure_manifest.csv`.

## 8. Statistical and aggregation rules

### 8.1 Whole-protein means and uncertainty

For model-level R2, MAE, RMSE, and bias, calculate the arithmetic mean over
protein rows after filtering to the declared split and evaluation scope. R2
excludes undefined values and reports the finite-protein count. Confidence
intervals resample proteins with replacement using one fixed recorded seed.
For a pairwise comparison between models, use the intersection of protein IDs
and apply the same bootstrap draw to both models.

Geometry summary means can be read from `geometry_summary.csv`; uncertainty
bands or distribution figures should use `geometry_metrics.csv` so proteins,
not summary rows, are resampled.

### 8.2 Sequence-separation metrics

Figure 9 uses the authoritative pooled subgroup metrics because the canonical
contract does not retain per-protein by sequence-bin rows. Separation-1 R2 is
not used as a headline metric because its target variance is extremely small.
No R2 value is clipped to fit a 0--1 axis; the supplemental R2 figure must show
or explicitly mark off-scale negative values.

The Figure 10 baseline is estimated from development proteins only with the
same protein-equal weighting used by the canonical methodology. Its evaluation
uses the locked test pairs. One baseline is calculated for each of the six bins
and reused at every layer. The figure reports raw baseline and model MAE rather
than a derived fraction of baseline error removed.

### 8.3 Amino-acid pairs

Figures 11--14 use the 210 unordered canonical labels in the subgroup output.
They do not average pair-specific R2 values into a single-amino-acid R2 because
R2 is not additive and pairs containing a given amino acid overlap.

The 210-line plots intentionally show raw pooled subgroup trajectories. They
answer whether the existing prediction accuracy differs by AA-pair identity;
they do not by themselves establish a biochemical cause. Differences may be
driven by true-distance, sequence-separation, or structural-context
distributions. A matched AA-pair analysis is retained as a possible future
analysis rather than silently changing the requested raw plot.

For descriptive overlays only, AA pairs may be assigned predeclared,
overlapping chemistry tags:

- hydrophobic-hydrophobic;
- aromatic-containing;
- same-charge and opposite-charge;
- polar-containing;
- Gly-containing;
- Pro-containing;
- Cys-Cys.

These tags do not define bonds. Any statistical comparison between chemistry
tags must account for their overlap and should be clearly labeled exploratory.

### 8.4 Refinement timing

R2 timing uses first passage to a fraction of the layer-0-to-peak improvement.
For error metrics, improvement is defined with the sign reversed so lower error
is better. A trajectory is `negligible_gain` if its total improvement is below
a predeclared practical threshold; the threshold is fixed before comparing
categories and recorded in the output.

Because first passage can be followed by a reversal, every timing table also
stores peak layer, final value, and late reversal. A sensitivity analysis may
require the threshold to remain crossed for three consecutive layers, but it
must not replace the raw trajectory.

### 8.5 ECOD category analyses

`ecod_protein_categories.csv` contains 400 proteins, four broad categories of
100 each, and 12 architecture labels. Canonical input validation may reduce the
usable count; unmatched or excluded proteins are reported rather than silently
dropped.

The category figures use per-protein OLS held-out pairs for their primary
trajectory because that scope is valid for every fitted protein. A locked-test
shared-OLS version may be added as a sensitivity analysis, but its smaller ECOD
test population should not be mixed with the all-ECOD per-protein curve.

Length adjustment uses continuous `log(n_residues)`. Broad category and
architecture effects should be estimated at every layer using a declared
regression or stratified permutation procedure, with multiplicity correction
across tested layers and contrasts. The raw means remain visible because the
adjusted model is a sensitivity analysis rather than a substitute for the
observed trajectories.

### 8.6 Protein trajectory organization without mandatory clustering

Clustering is not required for the initial paper figures. The first
heterogeneity display is a heatmap of all per-protein trajectories sorted by
R2 `L50`, annotated with length and ECOD category where available.

If a later exploratory clustering analysis is pursued, each protein is
represented by its 48-layer trajectory after subtracting its layer-0 value and
scaling by its total usable improvement. This emphasizes trajectory shape and
timing rather than absolute difficulty. A PCA or distance-preserving embedding
may precede hierarchical clustering. Cluster stability under protein bootstrap
and enrichment for known categories must be reported; learned clusters are
descriptive and are not independent evidence for biological classes.

### 8.7 Broader per-protein predictors

The per-protein program should expand in stages rather than requiring every
structural annotation before the first figures are made.

Immediately available predictors are residue count, mean true pair distance,
target-distance variance, layer-0 difficulty, final-layer difficulty, source
as a technical diagnostic, and ECOD broad category/architecture when present.
The running secondary-structure workflow can later add helix fraction, strand
fraction, mixed alpha/beta composition, segment counts, and segment lengths.
New structural processing would be needed for relative contact order, radius
of gyration, long-range-contact fraction, solvent exposure, disulfide count,
and domain count.

The consolidated annotation target is
`delta_files/analysis/per_protein_refinement/protein_attributes.csv`, with one
row per protein. Principal columns are `protein_id`, `source`, `n_residues`,
`mean_true_distance`, `target_variance`, `ecod_category`, `primary_arch`,
`helix_fraction`, `strand_fraction`, `relative_contact_order`,
`radius_of_gyration`, `long_range_contact_fraction`, `disulfide_count`, and a
status/source column for every derived attribute. Missing future attributes
remain explicit rather than excluding the protein.

Associations with `L50`, `L90`, total improvement, final accuracy, and late
reversal are appended to `protein_refinement_predictors.csv`. Continuous
predictors remain continuous in the primary model; bins are used only for
visualization. Secondary-structure, compactness, and contact-order predictors
should be interpreted as related structural descriptors rather than mutually
independent causal explanations.

## 9. Additional analyses that require more than current outputs

These are intentionally outside the initial 20+30 catalog. They should be
considered after the first figure generation pass and added only if they reveal
a distinct paper claim.

### 9.1 Structural case studies and reconstructed PDBs

Select proteins only after aggregate criteria are fixed: representative early,
late, nonmonotonic, and difficult trajectories rather than visually appealing
cherry-picked examples. Replay predictions, reconstruct coordinates, export
PDBs, and generate consistent-view structural renderings plus residue-pair
error maps.

Proposed output root:

```text
delta_files/analysis/structural_case_studies/
```

Principal files would be `case_selection.csv` with one row per selected protein
and selection criterion, `reconstruction_metrics.csv` with one row per protein,
model, and layer, and `pdb_exports/` with model/layer-specific structures.

### 9.2 Interaction-conditioned refinement

Use structurally confirmed salt bridges, disulfides, backbone hydrogen bonds,
beta bridges, aromatic contacts, and hydrophobic-core contacts with matched
negative pairs. Measure when each interaction becomes distinguishable after
matching or adjusting for true distance, sequence separation, amino-acid
identity, and protein. This analysis is described conceptually in
`docs/plans/biological_interpretability_experiments_plan.md` and the detailed
pair-classification plan.

Proposed output root:

```text
delta_files/analysis/interaction_conditioned_refinement/
```

### 9.3 Matched AA-pair trajectories

To distinguish amino-acid effects from different context distributions,
construct matched or reweighted AA-pair evaluations with common true-distance
and sequence-separation distributions. Save one row per AA pair, layer, model,
matching stratum, and metric. This requires pair-level replay or sufficient
statistics not present in the canonical subgroup CSVs.

Proposed output root:

```text
delta_files/analysis/amino_acid_gated/matched_context/
```

### 9.4 Protein-specific coefficient analyses

Coefficient analyses should follow, rather than define, trajectory groups.
Candidate questions include same-layer cosine alignment with the shared
coefficient, resampling stability, association between coefficient alignment
and shared-model failure, and cross-protein coefficient transfer within ECOD
architectures. Raw coefficient clustering alone is not a primary conclusion
because correlated representation features can make OLS coefficients unstable.

Proposed output root:

```text
delta_files/analysis/per_protein_refinement/coefficient_analysis/
```

### 9.5 Triangle-context explanations of layer updates

Test whether third-residue contexts predict the next-layer change in decoded
distance and whether recurring contexts explain the nonmonotonic accuracy and
triangle-consistency trajectories. This belongs under the existing planned
`delta_files/triangle_context_distance_delta/` family rather than being folded
into plotting scripts.

## 10. Recommended implementation order

1. Build a strict read-only source loader and figure manifest for the five
   canonical result families.
2. Generate Figures 1--9 and 11--15 plus the ready supplemental figures from
   existing CSVs.
3. Generate the local-derived ECOD, length, timing, gain, and heatmap analyses.
4. Audit and copy back the running secondary-structure comparison, then replace
   preliminary Figure 19--20 inputs with the four-model 48-layer outputs.
5. Add the development-only sequence-bin baseline computation for Figure 10.
6. Review the first 20 figures for redundancy before combining them into final
   paper panels.
7. Choose additional pair replay, structural case studies, interaction
   annotations, or coefficient analyses only after the aggregate figures show
   which questions remain unresolved.

## 11. Completion criteria for the initial figure set

The initial catalog is complete when:

- every main and supplemental figure has a PNG, PDF, plot-data CSV, and manifest
  row;
- every figure's population and evaluation scope are stated in its title,
  caption, or manifest note;
- the five model families use consistent names and aesthetics;
- whole-protein primary figures use mean-per-protein values and protein-level
  uncertainty;
- subgroup figures use the authoritative pooled values and disclose that scope;
- no separation-1 R2 is clipped or presented as an ordinary 0--1 accuracy
  trajectory;
- all sequence baselines are fit without locked-test targets;
- AA-pair figures contain exactly 210 canonical unordered pairs per completed
  layer and report sparse-support failures;
- ECOD joins report matched and unmatched counts and preserve the per-protein
  leading-zero ID convention;
- secondary-structure figures use the complete four-model output once the
  running workflow finishes; and
- readiness labels in the manifest match the actual artifact state.

# Project Summary: How Protein-Folding Models Build Geometry

## Executive summary

This project studies how a protein-folding AI develops an internal description of three-dimensional structure. The main object of study is OpenFold's pair representation: a feature vector attached to every ordered pair of amino acids and updated through 48 Evoformer layers.

The central method is probing. I freeze OpenFold, read its internal pair features at a chosen layer, and train a separate model to predict the final C-alpha distance between two residues. A successful probe shows that distance information is available in the representation; it does not show that OpenFold itself explicitly performs the same regression.

Several findings now stand out:

- A single linear rule shared across proteins already generalizes well. Its
  mean held-out per-protein $R^2$ rises from 0.494 at layer 0 to a peak of
  0.941 at layer 44.
- A separate linear rule for every protein reaches $R^2=0.988$, but this is an
  in-protein fit rather than a held-out-protein result. It is better interpreted
  as a flexible upper reference than as a fair winner.
- Sequence-separation-gated linear probes reach $R^2=0.957$ at layer 47 and
  dramatically improve short-range distances. Their largest advantage is for
  residues that are close in sequence, not for long-range pairs.
- A shared XGBoost probe reaches $R^2=0.981$ at layer 47 on its held-out split,
  suggesting that late representations contain important nonlinear structure.
  This result is promising but still comes from a three-layer hyperparameter
  search with a different split and feature convention from the main linear
  experiment.
- Decoded distances contain recognizable alpha-helix and beta-sheet signals.
  Sequence gating improves helix-versus-strand discrimination from AUROC 0.870
  to 0.982 at layer 47.
- Both the general and gated probes recognize beta-sheet partners much better
  than random pairs, but neither reliably recovers the exact one-residue
  register of the two strands at the final layer.
- Multidimensional scaling (MDS) turns predicted distances into plausible 3D
  traces, with best mean TM-scores of 0.869 for the held-out general probe and
  0.942 for per-protein probes.
- High distance accuracy is not the same as geometric consistency. Late-layer
  distance matrices still contain triangle-inequality violations and
  non-Euclidean spectral components, and a matched-$R^2$ random-error control
  can produce higher MDS TM-scores than the real probe errors.

Taken together, the results suggest that OpenFold's representation becomes increasingly rich in geometry, but that the encoded information is conditional on pair type, partly nonlinear, and not automatically organized as one exact Euclidean distance matrix.

## The scientific question

Protein-folding systems are often summarized as black boxes that map an amino acid sequence to a final 3D structure. That description hides the most interesting intermediate question: how does a fold become represented inside the network before the final coordinates are produced?

The project asks four related questions:

1. At which layers can a simple model decode residue-pair distances?
2. Is the same decoding rule transferable across proteins, or does each
   protein require its own rule?
3. Are important geometric signals linear, nonlinear, or conditional on
   information such as sequence separation and amino-acid identity?
4. Do accurate individual distances combine into a coherent three-dimensional
   protein, or are additional geometric constraints still missing?

The longer-term goal is to compare representation formation across several protein-folding models, including OpenFold, Boltz, and ESMFold. The current quantitative results are from OpenFold.

## What is being decoded

For a protein with $R$ residues, each OpenFold layer stores a pair tensor with shape $R \times R \times 128$. The two entries for a residue pair, $(i,j)$ and $(j,i)$, need not be identical.

The main linear probes concatenate both directions, producing 256 input features for every unordered residue pair. The target is the Euclidean distance between the two residues' C-alpha atoms in OpenFold's final predicted PDB.

This target choice is important. The probe measures how well intermediate features anticipate the model's own eventual structure. It is not, by itself, a measurement of agreement with an experimental structure.

The main benchmark began with 554 TM-score-filtered proteins from the CAMEO and ECOD collections. Two proteins had incompatible residue counts, leaving 552 usable proteins. The principal shared linear split contains 441 training proteins and 111 held-out test proteins.

## How the metrics should be read

The main distance metrics are:

- **Per-protein $R^2$:** the fraction of distance variance explained within one
  protein, averaged equally across proteins.
- **Pooled $R^2$:** the fraction of variance explained after pooling all residue
  pairs, so proteins with more pairs have more influence.
- **MAE:** mean absolute distance error in angstroms. This is especially useful
  when the target has very low variance.
- **RMSE:** root mean squared error in angstroms, which penalizes large misses
  more strongly than MAE.
- **TM-score after MDS:** similarity between the reconstructed C-alpha trace
  and the final predicted structure after alignment.

Mean per-protein $R^2$ is the headline measure because it gives each protein equal weight. Pooled metrics remain useful for understanding the average residue pair.

An $R^2$ near 1 is strong. An $R^2$ near 0 is no better than predicting the mean target value, and a negative $R^2$ is worse than that baseline. However, $R^2$ becomes unstable when the true values barely vary. Adjacent C-alpha distances, for example, average about 3.808 Å with a standard deviation of only 0.066 Å on the held-out set. MAE is more interpretable for that regime.

## The four main model families

The project currently compares four ways of reading geometry from pair representations.

| Model | Core question | Evaluation design |
|---|---|---|
| Shared linear, or general | Is one linear rule transferable across proteins? | Train proteins and held-out proteins are disjoint |
| Per-protein linear | How much can a protein-specific linear rule fit? | One model per protein, fitted and evaluated on all of that protein's pairs |
| Sequence-separation-gated linear | Does a different linear rule help each sequence range? | Six rules per layer; same 441/111 split as the general probe |
| Shared XGBoost | Is substantial information available through nonlinear feature interactions? | Protein-level fit/validation/test split and validation-selected trees |

The evaluation designs are not identical. In particular, the per-protein model is evaluated in sample. The XGBoost search also uses a different protein split, only 21 of its 111 test proteins overlap the general probe's test set, and it uses 128 symmetrically averaged pair features rather than the general probe's 256 directional features.

The four-way numbers are therefore a structured comparison, not a perfectly controlled tournament.

## Headline distance results

At layer 47, the available mean per-protein $R^2$ values are:

| Model | Layer-47 mean $R^2$ | Scope |
|---|---:|---|
| Shared linear | 0.936 | Held-out proteins |
| Per-protein linear | 0.985 | In-protein fit |
| Sequence-separation-gated linear | 0.957 | Held-out proteins |
| Shared XGBoost | 0.981 | Held-out proteins on a different split |

The best observed mean $R^2$ for each current sweep is:

| Model | Best observed layer | Mean $R^2$ |
|---|---:|---:|
| Shared linear | 44 | 0.941 |
| Per-protein linear | 45 | 0.988 |
| Sequence-separation-gated linear | 47 | 0.957 |
| Shared XGBoost | 47 | 0.981 |

The gated sweep currently covers layers 0, 10, 20, 30, 40, and 47. The XGBoost hyperparameter search currently covers anchor layers 0, 23, and 47. “Best observed” should be read within those sampled layers.

## What changes through depth

The shared and per-protein linear trajectories show a broad transition from weak or protein-specific early geometry to highly decodable late geometry.

| Layer | Shared linear mean $R^2$ | Per-protein linear mean $R^2$ | Gap |
|---:|---:|---:|---:|
| 0 | 0.494 | 0.720 | 0.226 |
| 10 | 0.611 | 0.819 | 0.208 |
| 20 | 0.843 | 0.954 | 0.110 |
| 30 | 0.831 | 0.952 | 0.121 |
| 40 | 0.894 | 0.977 | 0.082 |
| 47 | 0.936 | 0.985 | 0.049 |

The narrowing gap is suggestive. Early features support much better decoding when the rule can adapt to one protein. Late features support nearly the same linear direction across proteins, even though the in-protein model retains an advantage.

Because the per-protein evaluation is in sample, the gap mixes several effects: protein specificity, model flexibility, and optimistic evaluation. It should not be interpreted as a pure measure of memorization.

The shared trajectory is also not monotonic. Mean $R^2$ rises to 0.843 at layer 20, dips to 0.831 at layer 30, then climbs sharply through the 40s. The final layer is slightly worse than layers 44–46. This suggests that later updates may reorganize representations for the structure module rather than continually optimizing linear distance accessibility.

The coefficient directions themselves change strongly. The shared linear coefficient vectors at layers 0 and 47 have cosine similarity -0.035, close to orthogonal. Adjacent layers are usually much more similar, but notable shifts appear near the beginning, around layer 13, and again in the final layers. Geometry is not merely added to a fixed readout direction; the readout itself is reorganized across depth.

## Why sequence separation matters

Residue pairs occupy very different geometric regimes depending on their distance along the amino-acid sequence. Adjacent residues are constrained by the peptide backbone. Pairs separated by a few residues encode local secondary structure. Very distant sequence pairs often describe tertiary contacts or unrelated regions of the fold.

The sequence-gated experiment trains six separate linear decoders per sampled layer:

- separation 1;
- separations 2–4;
- separations 5–9;
- separations 10–19;
- separations 20–49;
- separation 50 or more.

The six predictions are recombined before whole-protein metrics are computed. This preserves a single overall evaluation while letting the decoding rule depend on sequence range.

### Whole-protein improvement

| Layer | General mean $R^2$ | Gated mean $R^2$ | General pooled MAE | Gated pooled MAE |
|---:|---:|---:|---:|---:|
| 0 | 0.494 | 0.553 | 6.432 Å | 6.016 Å |
| 10 | 0.611 | 0.663 | 5.872 Å | 5.484 Å |
| 20 | 0.843 | 0.873 | 3.043 Å | 2.870 Å |
| 30 | 0.831 | 0.871 | 3.089 Å | 2.845 Å |
| 40 | 0.894 | 0.930 | 2.445 Å | 2.193 Å |
| 47 | 0.936 | 0.957 | 1.906 Å | 1.723 Å |

At layer 47, sequence gating improves mean per-protein $R^2$ by about 0.022 and reduces mean per-protein MAE from 1.683 Å to 1.375 Å.

### The close-pair MAE result

The improvement is highly concentrated at short sequence separations.

| Sequence separation | General layer-47 MAE | Gated layer-47 MAE | Reduction |
|---|---:|---:|---:|
| 1 | 3.149 Å | 0.036 Å | 98.8% |
| 2–4 | 2.485 Å | 0.257 Å | 89.7% |
| 5–9 | 1.670 Å | 0.497 Å | 70.2% |
| 10–19 | 1.537 Å | 0.891 Å | 42.0% |
| 20–49 | 1.574 Å | 1.363 Å | 13.4% |
| 50+ | 2.004 Å | 1.993 Å | 0.6% |

This pattern gives a clear interpretation of the whole-protein gain. One global linear rule is pulled between extremely different distance regimes. Gating lets the local models learn backbone- and motif-specific mappings while leaving the long-range rule almost unchanged.

A model that uses raw sequence separation alone reaches only mean per-protein $R^2=0.108$ and pooled $R^2=0.292$. The improvement therefore does not come from simply looking up a typical distance for each sequence range. OpenFold's pair features supply most of the predictive information; gating changes how that information is read.

The fitted sequence-gate coefficient vectors are also meaningfully different. Their mean off-diagonal cosine similarity is 0.267 at layer 0 and 0.299 at layer 47. The model is not learning six nearly identical copies of one probe.

## What XGBoost adds

Linear probes answer whether a target is directly accessible along one global feature direction. XGBoost can combine thresholds and feature interactions, testing whether the representation contains useful nonlinear structure that a single linear map misses.

The shared XGBoost search used a protein-equal training objective, a maximum of 10,000 sampled training pairs per fit protein, and separate fit, validation, and test protein sets. Hyperparameters were selected using validation mean per-protein $R^2$.

| Anchor layer | Validation-selected test mean $R^2$ | Test pooled $R^2$ |
|---:|---:|---:|
| 0 | 0.571 | 0.582 |
| 23 | 0.923 | 0.913 |
| 47 | 0.981 | 0.978 |

The selected layer-47 model used depth-4 trees, learning rate 0.1, and minimum child weight 5. Its best iteration was the final allowed iteration, 699. That means performance had not clearly saturated within the search budget and a longer training schedule may still change the result.

Relative to the shared linear trajectory, the observed mean-$R^2$ differences at the three anchor layers are approximately +0.077, +0.074, and +0.046. Because the split and feature construction differ, those deltas are indicative rather than strict ablations.

A separate tree diagnostic supports the same qualitative conclusion: pair channels dominate nonlinear predictive power, while protein length and sequence separation act mainly as secondary routing variables. Direct feature ablation is still needed before assigning causal importance to any channel.

## Amino-acid-gated probes

Sequence position is not the only possible conditioning variable. Another set of probes asks whether different unordered amino-acid pairs require different linear rules.

Three gated variants are available at six representative layers:

- sequence separation only: 6 gates per layer;
- amino-acid pair only: 210 gates per layer;
- amino-acid pair crossed with sequence separation: up to 1,260 gates per
  layer.

At layer 47, their whole-protein results are:

| Model | Mean per-protein $R^2$ | Pooled MAE | Test-pair coverage |
|---|---:|---:|---:|
| General | 0.9356 | 1.906 Å | 100% |
| Amino-acid-pair gated | 0.9361 | 1.889 Å | 100% |
| Sequence gated | 0.9574 | 1.723 Å | 100% |
| Amino-acid pair × sequence gated | 0.9551 | 1.745 Å | 99.90% |

Amino-acid gating alone adds almost nothing over the general model. The joint model is strong, but it does not improve on sequence gating alone. It also creates sparse cells: 7,050 of 7,560 requested joint fits succeeded, while 510 had insufficient training rows.

Coefficient similarity supports the same interpretation. Across distinct amino-acid-pair gates, mean cosine similarity rises from 0.721 at layer 0 to 0.862 at layer 47. In contrast, distinct sequence gates remain far less aligned. Late OpenFold features appear to support a fairly common amino-acid-independent distance direction, while different sequence ranges still benefit from different readouts.

This does not prove amino-acid identity is irrelevant. Rare residue pairs have limited support, correlated channels can hide alternative solutions, and a nonlinear conditional model could behave differently. It does show that a large bank of amino-acid-specific linear probes is not currently justified by whole-protein accuracy.

## Predicting alpha helices and beta pleated sheets

Distance accuracy is scientifically more useful if it corresponds to recognizable protein substructures. I therefore tested whether decoded distances recover the local signature of alpha helices and the partner pattern of beta pleated sheets.

Secondary-structure labels are assigned by DSSP on OpenFold's final predicted PDBs. All calibration choices are frozen using training proteins, and the reported results use the same 111 held-out proteins as the shared linear probe.

### Alpha-helix method

For each candidate residue, the analysis collects its decoded distances to positions $i+2$, $i+3$, and $i+4$. These three distances form a local geometric signature. Training structures define empirical helix and strand centroids, and a score asks whether a decoded triplet looks closer to the helix pattern or the strand pattern.

This is not a newly trained helix classifier at every layer. It replays frozen distance probes and evaluates the geometry those probes imply.

### Helix-versus-strand results

| Layer | General AUROC | Sequence-gated AUROC |
|---:|---:|---:|
| 0 | 0.553 | 0.924 |
| 10 | 0.727 | 0.981 |
| 20 | 0.866 | 0.980 |
| 30 | 0.852 | 0.982 |
| 40 | 0.855 | 0.984 |
| 47 | 0.870 | 0.982 |

The general model develops useful helix information gradually. Sequence gating makes the local pattern strongly visible even at layer 0 and remains near AUROC 0.98 thereafter. This is consistent with the close-pair MAE result: helix geometry depends heavily on distances at short, specific sequence offsets, precisely where gating helps most.

The held-out analysis includes 6,541 helix residues and 4,726 strand residues. The AUROCs are descriptive point estimates; uncertainty intervals have not yet been added.

### Beta-sheet method

For beta sheets, DSSP provides paired residues across strands. A decoded partner distance is compared with two types of controls:

- random pairs matched for sequence separation, testing coarse sheet-partner
  recognition;
- partners shifted by one or two residues along the opposing strand, testing
  exact register.

There are 3,167 held-out sheet bridges in the combined analysis.

### Beta-sheet results

| Layer | General partner-vs-random AUROC | Gated partner-vs-random AUROC | General partner-vs-shift-1 AUROC | Gated partner-vs-shift-1 AUROC |
|---:|---:|---:|---:|---:|
| 0 | 0.914 | 0.925 | 0.765 | 0.742 |
| 20 | 0.944 | 0.948 | 0.492 | 0.535 |
| 40 | 0.958 | 0.966 | 0.651 | 0.668 |
| 47 | 0.935 | 0.963 | 0.219 | 0.411 |

Both models identify genuine sheet partners far better than matched random pairs, and sequence gating gives a modest improvement. At layer 47, random controls are predicted about 14.0 Å farther away than partners for the general probe and 15.1 Å farther for the gated probe.

Exact register is much weaker and nonmonotonic. At layer 47, the mean distance margin for a one-residue shift is -3.58 Å for the general probe and -1.05 Å for the gated probe. A negative margin means the shifted residue is predicted closer than the DSSP partner on average.

The conservative conclusion is therefore that the probes recover beta-sheet membership and broad cross-strand proximity, but not reliable final-layer strand alignment. “The sheet is here” is more decodable than “this is the exact partner residue.”

## From distances to 3D with MDS

A complete predicted distance matrix can be converted into coordinates using classical multidimensional scaling. MDS double-centers the squared distance matrix, decomposes the resulting Gram matrix, and retains the top three positive eigen-directions as 3D coordinates.

This provides a useful visualization of intermediate structure. It also tests whether decoded distances can support a recognizable global fold.

| Model | Layer 0 mean TM-score | Best mean TM-score | Best layer | Layer 47 mean TM-score |
|---|---:|---:|---:|---:|
| Shared linear | 0.421 | 0.869 | 45 | 0.852 |
| Per-protein linear | 0.497 | 0.942 | 46 | 0.935 |

The same broad phase transition appears in 3D. The shared model reaches mean TM-score 0.706 by layer 20 and 0.805 by layer 40. Per-protein probes reach 0.818 and 0.900 at those layers.

The best distance-$R^2$ layer and best MDS layer are close but not identical. For the general probe, $R^2$ peaks at layer 44 while TM-score peaks at layer 45. At layer 47, the correlation between protein-level $R^2$ and TM-score is only 0.482 for the general probe. Accurate scalar distances help, but they do not fully determine reconstruction quality.

## Geometric consistency: the unresolved issue

MDS always returns a 3D point cloud after discarding incompatible spectral components. Before this step, negative predicted distances are counted and clipped to zero. A good MDS reconstruction therefore does not prove that the decoded matrix was already a valid set of Euclidean distances.

Several diagnostics expose the gap.

### Triangle inequalities

At late high-performing layers, the mean fraction of residue triples violating the triangle inequality is:

| Model and layer | Violation fraction |
|---|---:|
| General, layer 45 | 9.37% |
| General, layer 47 | 9.02% |
| Per-protein, layer 45 | 4.21% |
| Per-protein, layer 47 | 4.52% |

Most violations are below 5 Å, and severe violations above 10 Å are rare, but even the symmetric, nonnegative-clipped matrices are not metrics. Moreover, the general probe's lowest mean violation fraction occurs at layer 1, when its distance $R^2$ is still only about 0.50. Triangle consistency and target accuracy do not improve in lockstep.

### Spectral diagnostics

At layer 45, only 64.6% of the general model's positive MDS eigenvalue mass is captured by the top three dimensions. The corresponding per-protein value is 76.6%. The magnitude of negative eigenvalues is 36.1% of positive eigenvalue mass for the general probe and 22.8% for the per-protein probes.

These numbers improve through depth but remain far from an exact rank-3 Euclidean geometry. The per-protein decoder produces a more 3D-compatible matrix, consistent with its higher TM-score and lower MDS stress.

### Matched-$R^2$ error control

A separate null experiment adds independent Gaussian errors to true distance matrices while matching each protein's observed probe $R^2$. Surprisingly, these matched-accuracy controls yield higher mean MDS TM-scores:

| Layer | Observed general TM-score | Matched-$R^2$ null TM-score |
|---:|---:|---:|
| 0 | 0.421 | 0.774 |
| 20 | 0.706 | 0.901 |
| 40 | 0.805 | 0.927 |
| 47 | 0.852 | 0.954 |

This does not mean the probe contains no geometry. It means that the structure of its errors matters. Real residuals are correlated across residue pairs and distance regimes; the same scalar $R^2$ with independent noise can be easier for MDS to average into a correct fold.

The null is deliberately simple. It does not preserve the real residual distribution, distance-dependent error variance, or pairwise correlations. A stronger test would replay coefficients and permute residuals in ways that preserve more of that structure.

The main geometric lesson is that future probes should optimize or constrain global consistency, not only independent distance error.

## Additional evidence from salt bridges

The pair representation also supports a different task: classifying salt bridges between acidic and basic residues. This smaller experiment uses 30 training, 10 validation, and 10 test proteins, and compares cross-residue pair features with features taken only from the two residue diagonals.

Cross-residue features achieve matched-test AUROC 0.867 at layer 0, peak at 0.965 around layer 31, and remain strong at 0.941 at layer 47. Average precision peaks at 0.966 around layer 42.

Diagonal-only features are much weaker: their best AUROC is 0.679. Adding diagonal features to the cross-residue representation produces no consistent ranking gain.

This supports a broader conclusion than distance regression alone. The pair representation contains information about specific residue-residue interactions, and that information resides primarily in cross-residue features.

Performance on the natural, highly imbalanced candidate pool is less smooth than performance on matched controls. This is a reminder that a clean probe benchmark can hide distribution shift and class-imbalance problems.

## Reliability and methodological controls

Several supporting experiments changed how the main results should be computed and interpreted.

### Numerical precision

Dense least-squares fitting over millions of residue pairs is numerically sensitive. In a layer-47 diagnostic, all-row float32 fits collapsed:

- a 128-feature model moved from $R^2=-0.004$ in float32 to 0.875 in float64;
- a 256-feature model moved from $R^2=0.052$ in float32 to 0.905 in float64.

Smaller row-capped fits appeared healthy in float32, which initially obscured the problem. The current dense linear workflow therefore materializes features and targets in float64 before scaling and fitting.

### Protein weighting

Large proteins contribute quadratically more residue pairs than small proteins. In the same diagnostic, a protein-equal 256-feature fit reached $R^2=0.905$, whereas pair-equal weighting reached 0.857. Weighting is not a neutral implementation choice; it changes which proteins the decoder serves best.

### Protein size

Even after correcting precision, performance remains correlated with protein length. The float64 256-feature diagnostic has a length-versus-$R^2$ Pearson correlation of 0.606. Size-bucketed models help the smallest proteins, implying that one global decoder still absorbs protein-scale effects.

Protein-specific coefficient clustering reinforces this caution. A stable two-cluster solution exists at layer 47, but cluster assignments are strongly associated with protein size and dataset source. The clusters should not be treated as discovered biological classes without stronger controls.

### Coefficient stability

An earlier five-fold analysis found high same-layer coefficient cosine similarity across folds, generally around 0.97–0.98. The exact channel weights are therefore reasonably reproducible within that setup, even though coefficient rankings remain hard to interpret when channels are correlated.

## What the current evidence supports

The combined results support the following claims:

1. OpenFold pair representations contain substantial information about the
   model's final residue-pair distances from the earliest saved layer.
2. Distance information becomes much more transferable and linearly accessible
   through the Evoformer stack, especially in the late layers.
3. Protein-specific fitting helps most early and less late, consistent with an
   increasingly shared geometric code.
4. Sequence separation defines genuinely different decoding regimes. It is
   especially important for adjacent and short-range pairs.
5. Amino-acid identity alone is a much weaker gating variable for linear
   distance probes.
6. Nonlinear tree models recover information that the shared linear model
   misses, especially at the late anchor layer.
7. Local helix geometry and coarse beta-sheet pairing are decodable, while
   exact beta-sheet register remains unresolved.
8. High per-pair accuracy can coexist with globally inconsistent distances.
9. MDS reconstruction quality depends on correlated error structure, not just
   scalar $R^2$.
10. Numerical precision, protein weighting, size distribution, and evaluation
    split can materially alter conclusions.

## What the current evidence does not establish

The experiments do not yet establish that:

- OpenFold explicitly uses the fitted probe directions;
- a high probe score identifies a causal feature or mechanism;
- the decoded geometry is already represented as one coherent 3D coordinate
  system;
- performance against the final predicted PDB equals accuracy against an
  experimental structure;
- the per-protein linear score is directly comparable with held-out shared
  models;
- the current XGBoost result is a final 48-layer trajectory;
- amino-acid identity has no nonlinear or context-dependent role;
- apparent best layers would remain best on a new confirmatory dataset;
- coefficient clusters correspond to biological protein classes;
- the secondary-structure point estimates have fully quantified uncertainty.

## Next experiments

The most informative next steps are:

1. Complete sequence-gated and XGBoost sweeps across all 48 layers with a
   shared protein split and matched feature conventions.
2. Evaluate all four model families on exactly the same held-out proteins and
   report both protein-equal and pair-pooled metrics.
3. Decode complete distance matrices from the gated and XGBoost probes, then
   compare MDS, TM-score, triangle violations, and spectra with the general and
   per-protein models.
4. Add uncertainty intervals and paired tests for layer-to-layer and
   model-to-model differences.
5. Test geometry-aware objectives or projections that enforce nonnegative,
   symmetric, triangle-consistent, low-dimensional distances.
6. Replace the independent matched-$R^2$ null with residual-permutation and
   correlation-preserving controls.
7. Evaluate probe reconstructions against experimental structures as well as
   OpenFold's final prediction.
8. Run direct ablations for sequence separation, protein size, and highly
   ranked pair channels.
9. Investigate why beta-sheet membership is visible while final-layer register
   becomes unreliable.
10. Repeat the analysis in Boltz and ESMFold to distinguish OpenFold-specific
    mechanisms from shared principles of folding models.

## Broader significance

The project is moving from the question “can distance be decoded?” toward a more mechanistic question: “what kind of geometric object is encoded, for which residue pairs, and under what readout?”

The answer so far is not one simple global coordinate system. OpenFold's pair representation supports a strong shared linear geometry, several distinct sequence-dependent local geometries, and an even stronger nonlinear decoder. It carries recognizable helix, sheet, and interaction signals, yet its decoded distance predictions remain imperfectly compatible with 3D Euclidean space.

That combination is scientifically useful. The successes identify where structural information becomes accessible; the failures reveal what ordinary probes discard. Together they provide a path toward understanding not just the final fold, but the internal computational steps by which a folding model constructs it.

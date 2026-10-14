# Project Summary: Geometry in OpenFold Pair Representations

## Question and approach

This project asks when and how OpenFold's Evoformer pair representation makes protein geometry accessible. Each of its 48 saved layers contains a 128-channel feature vector for every ordered residue pair. We freeze OpenFold, fit separate probes to those features, and ask how well they predict C-alpha distances in OpenFold's final predicted structure. Probe accuracy shows that a readout can recover distance information; it does not show that OpenFold uses that readout internally or that the features are causal.

The default dataset combines 154 CAMEO and 400 ECOD proteins selected with TM-score > 0.7. The shared eligible cohort contains 552 proteins after two technical exclusions. Every primary regression uses one unordered observation per residue pair, excludes self-pairs, and draws its target from the final predicted PDB. The target measures agreement with OpenFold's own output, not with an experimental structure.

## Current primary comparison

The canonical protocol is version 2.0: fivefold out-of-fold (OOF) predictions over the same 552 proteins and all 48 layers. The 111 proteins in the earlier test set are retained as fold 0, but the primary results aggregate predictions from all five folds. The four model families are:

| Probe | What it tests | OOF evaluation |
|---|---|---|
| Shared linear | A common linear readout across proteins | Unseen protein IDs |
| Sequence-gated linear | Separate linear readouts for six sequence-separation ranges | Unseen protein IDs |
| Shared XGBoost | Nonlinear thresholds and interactions | Unseen protein IDs |
| Per-protein linear | A readout tailored to one protein | Unseen residue pairs within a seen protein |

The linear models use both pair directions (256 features). The fixed XGBoost model is evaluated on the same five protein folds, but its configuration was selected using earlier, non-nested work on this cohort. Its OOF score is therefore a useful comparison, not an untouched hyperparameter-selection estimate. The per-protein model answers a different generalization question and should not be ranked as though it predicts unseen proteins.

Mean per-protein $R^2$ gives every protein equal weight. At layer 47, the current fivefold OOF results are:

| Probe | Layer-47 mean per-protein $R^2$ | Best layer and mean $R^2$ |
|---|---:|---:|
| Shared linear | 0.934 | 44: 0.940 |
| Sequence-gated linear | 0.954 | 45: 0.958 |
| Shared XGBoost | 0.981 | 46: 0.982 |
| Per-protein linear | 0.984 | 45: 0.986 |

The shared linear probe rises from $R^2=0.496$ at layer 0 to 0.934 at layer 47. Distance information becomes much more accessible to a transferable linear rule through the stack, although the trajectory is not monotonic. Sequence gating helps most for nearby sequence positions: at layer 47, adjacent-residue mean absolute error is 3.04 Å for the shared linear probe and 0.036 Å for the gated probe. This gain is concentrated in local geometry rather than uniformly across all residue pairs. XGBoost's higher late-layer score indicates that a nonlinear readout extracts information missed by one shared linear map.

Earlier 80/20 analyses and amino-acid-pair gating remain useful exploratory comparisons, but they are not part of this four-family protocol-2.0 result. In the earlier gated analysis, amino-acid identity alone added little to whole-protein linear accuracy compared with sequence separation.

## From pair distances to 3D

We reconstruct C-alpha traces by applying classical multidimensional scaling (MDS) to each protein's complete OOF distance matrix. The saved results and September 2 figures evaluate the TM kernel after Kabsch alignment, rather than optimizing the alignment for TM-score. At layer 47, these mean post-Kabsch TM-style scores against OpenFold's final predicted structure are 0.842 for shared linear, 0.862 for sequence-gated linear, 0.925 for XGBoost, and 0.931 for per-protein linear. Each model reaches its best mean saved score around layers 45–46 rather than necessarily at the final layer. Optimized TM-score rescoring and the corresponding publication refresh are pending; these values do not yet use the updated scoring method.

A strong distance score does not guarantee a valid three-dimensional geometry. At layer 47, the mean fraction of residue triples that violate the triangle inequality is 8.7% for shared linear, 6.3% for sequence-gated linear, 4.0% for XGBoost, and 4.9% for per-protein linear. MDS can still return a plausible trace because it projects an imperfect distance matrix into three dimensions.

The current fivefold outputs include geometry diagnostics, sequence- and spatial-distance error summaries, and a replay of triangle-violation subgroups. The September 2 publication bundle reports protein-equal means with paired-protein bootstrap intervals and records figure provenance.

## Interpretation and next steps

These results support a progressively more accessible, partly conditional and nonlinear geometric signal in OpenFold's pair representation. They do not establish that the network uses the fitted probe coefficients, that decoded distances already form one exact Euclidean structure, or that accuracy against OpenFold's prediction equals accuracy against experimental structures. The fivefold split holds out protein IDs, but does not explicitly hold out homologous sequence or fold families.

The most useful next steps are to test generalization across protein families and experimental structures, investigate which correlated distance errors obstruct 3D reconstruction, and compare the same questions in other folding models.

For exact methods and files, see [the canonical regression methodology](docs/methodology/regression_methodology.md), [the output contract](docs/methodology/regression_outputs.md), [the fivefold results](delta_files/regression_5fold_v1/), and [the September 2 figure bundle](delta_files/publication_plots/sept_2_plots/README.md). [Current sources and status](docs/project_status.md) records available artifacts and pending updates.

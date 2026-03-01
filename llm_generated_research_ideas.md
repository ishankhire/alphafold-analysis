# LLM-Generated Research Ideas

Current baseline: pooled multi-protein distance regression using pair features `[pb[i,j], pb[j,i]]`. Prior work in this repo already explored diagonal self-features, amino-acid augmentation, MDS reconstruction, and protein-level generalization. The ideas below are the next interpretability-focused directions to test.

For any new analysis, compare against the same baseline setup and reuse the same evaluation frame where possible: pooled pair-split metrics, protein-level split metrics, early/mid/late layer comparisons, sequence-separation bins, and MDS/geometry diagnostics for any probe that changes predicted distances.

## 1. Neighbor-Context Pair Features

**Idea**  
Augment the feature vector for pair `(i, j)` with nearby pair cells such as `(i-1, j)`, `(i+1, j)`, `(i, j-1)`, `(i, j+1)`. A second variant can add diagonal neighbors like `(i-1, j-1)` and `(i+1, j+1)`.

**What to test**  
Compare the baseline regression against a radius-1 neighbor-augmented regression. Keep the train/test split, scaling, and layer-by-layer structure identical to the current baseline.

**Why it matters**  
The single pair cell may not contain all local folding context. Nearby cells may encode local continuity, secondary-structure regularity, or short-range interaction context that helps the model infer distance.

**How to run it**  
Construct expanded feature vectors by concatenating neighbor pair representations, then fit the same regression probe per layer. If possible, include a dimension-matched control so any gain is not just from adding parameters.

**What to look for**  
Check whether R^2 improves, whether MDS-based geometry improves, and whether gains are concentrated in specific layer ranges.

## 2. Symmetric vs Antisymmetric Decomposition

**Idea**  
Replace the raw ordered features with a symmetric part and an antisymmetric part:
`S = (pb[i,j] + pb[j,i]) / 2` and `A = (pb[i,j] - pb[j,i]) / 2`.

**What to test**  
Train three probes: symmetric-only, antisymmetric-only, and combined `[S, A]`.

**Why it matters**  
Distance is a symmetric target, so most useful signal should live in the symmetric component. This is a cleaner interpretation than only observing that upper and lower triangles are similar.

**How to run it**  
Build the transformed features for each pair and reuse the existing regression pipeline with no other structural changes.

**What to look for**  
If symmetric features retain nearly all predictive power, that supports a strong interpretability claim. If antisymmetric signal matters more in early layers, that may indicate directional information is later compressed away.

## 3. Residue-Environment Summary Features

**Idea**  
Give each residue a compact context summary derived from its row or column in the pair matrix. Examples: row mean, column mean, local diagonal summaries, or simple variance-like aggregates.

**What to test**  
Append residue-level summary features for residue `i` and residue `j` to the base pair feature vector.

**Why it matters**  
The distance between `i` and `j` may depend on each residue's broader interaction profile, not only on the direct pair embedding for that single cell.

**How to run it**  
Compute compact summary vectors per residue for each layer, then concatenate the summaries for both residues with the baseline pair features before fitting the regression.

**What to look for**  
See whether compact summaries help as much as or more than raw neighbor concatenation. If they do, that suggests the relevant missing information is residue-level context rather than just local grid adjacency.

## 4. Sequence-Separation Stratification

**Idea**  
Analyze performance separately for short-, medium-, and long-range pairs using sequence separation `|i-j|`.

**What to test**  
Bin residue pairs by sequence separation and compute regression metrics, coefficient behavior, or feature importance separately by bin.

**Why it matters**  
Local backbone geometry and long-range tertiary contacts are different prediction problems. The same channels may not be responsible for both regimes.

**How to run it**  
Keep the same probe, but evaluate outputs separately within predefined separation bins. This can be layered on top of most other analyses in this document.

**What to look for**  
Look for channel specialization by range and whether certain feature additions help mainly on local or long-range pairs.

## 5. Cross-Layer Update Prediction

**Idea**  
Try to predict how pair representations change from one layer to the next, focusing on the delta `pb_(l+1) - pb_l`.

**What to test**  
For each adjacent layer pair `(l, l+1)`, predict the next-layer update from the current layer. Start with linear regression and only use a small nonlinear model as a comparison if needed.

**Why it matters**  
If layer-to-layer updates are predictable, that suggests the network is applying a structured refinement rule instead of performing an opaque, fully unconstrained transformation at each step.

**How to run it**  
Treat the current-layer pair representation as input and the per-channel delta as target. Compare using only the central pair cell versus adding neighbor context.

**What to look for**  
Check whether some layer ranges are easier to predict, whether context helps substantially, and whether the update appears low-rank or concentrated in a small set of channels.

## 6. Supervised Low-Dimensional Subspaces

**Idea**  
Find a compact subspace that is specifically predictive of distance, rather than a high-variance subspace like PCA.

**What to test**  
Use supervised dimensionality-reduction methods such as PLS or sparse linear models and compare them to the full feature space.

**Why it matters**  
PCA already showed distance information survives in a small subspace, but PCA is unsupervised. A supervised subspace may isolate the directions that are most directly tied to geometry.

**How to run it**  
Fit the supervised projection on training data only, then train a simple regression model in the reduced space and compare performance to the full baseline.

**What to look for**  
Measure how many dimensions are needed to recover most of the signal, and whether those dimensions align with interpretable variables like sequence separation, contact status, or layer depth.

## 7. Sparse Autoencoders for Latent Feature Discovery

**Idea**  
Train sparse autoencoders on selected pair-representation layers to discover latent units that may correspond to reusable structural features.

**What to test**  
Apply the method to a few informative layers first, such as one early, one middle, and one late layer, instead of all 48 immediately.

**Why it matters**  
Sparse autoencoders may expose more disentangled latent features than raw channels, which could make it easier to associate activations with structural motifs or interaction patterns.

**How to run it**  
Train an autoencoder with a sparsity constraint on sampled pair features, then correlate latent activations with simple known variables such as sequence separation, pair type, contact status, or geometry metrics.

**What to look for**  
Look for latent units that consistently align with interpretable behaviors, and for units whose activation tracks geometric quality or specific interaction classes.

## 8. Geometry-First Evaluation of New Probes

**Idea**  
Treat geometric coherence as a primary evaluation target, not just pairwise R^2.

**What to test**  
For each new feature set or probe design, evaluate both pairwise regression metrics and geometry metrics such as TM-score, RMSD, MDS stress, triangle inequality violations, and negative eigenvalue behavior.

**Why it matters**  
The existing MDS work already showed that strong pairwise R^2 does not guarantee a physically coherent 3D reconstruction. Some ideas may help geometry without changing pairwise accuracy much.

**How to run it**  
Reuse the current MDS reconstruction pipeline after each new regression variant, so every feature change is judged by both marginal accuracy and global structural consistency.

**What to look for**  
Pay attention to cases where a feature change gives only a small R^2 gain but clearly improves geometric realizability or reduces metric violations.

## Suggested Order

Start with neighbor-context features, then symmetric vs antisymmetric decomposition, then cross-layer update prediction. Reserve sparse autoencoders for later, once the most informative layers or subspaces are clearer. The general goal is interpretability first, with raw performance gains treated as supporting evidence rather than the only success criterion.

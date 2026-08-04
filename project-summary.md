# Project Summary: Interpreting Protein-Folding Models

## Research focus

This project studies how AI systems for protein folding build internal
representations of three-dimensional structure. The current experiments use
**OpenFold** pair representations, with the analysis designed to extend to
other folding models such as **Boltz** and **ESMFold**. The goal is not simply
to measure final structure accuracy, but to understand the intermediate
computations that turn sequence- and evolutionary-information into a fold.

## Central question

At what stage, and in what form, do a model's internal features encode protein
geometry? In OpenFold, the main object of study is the pair representation: a
feature vector for every residue pair at every one of 48 Evoformer layers. By
probing these representations layer by layer, the project tracks the emergence
of spatial knowledge during inference.

## Main analyses

- **Distance decoding:** predict C-alpha distances between residue pairs from
  internal pair features. Linear probes establish what is directly accessible;
  XGBoost and other tree-based probes test for nonlinear structure that linear
  models miss.
- **Generalization versus capacity:** compare one shared probe across proteins
  with separate per-protein probes. Protein-level train/validation/test splits,
  size stratification, and k-fold checks distinguish transferable signals from
  protein-specific fitting.
- **Layer dynamics:** measure how predictive performance, learned feature
  directions, and geometry change from early to late layers. Feature variants
  test directional pair features, symmetrization, diagonal/self features, and
  local or axial context.
- **Intermediate structures:** decode a full distance matrix from each layer,
  then reconstruct a 3D C-alpha trace with classical multidimensional scaling
  (MDS). These intermediate reconstructions make the evolving fold directly
  inspectable and can be exported for molecular visualization.
- **Geometric validity:** evaluate reconstructed distances and structures using
  MDS stress, spectral diagnostics, TM-score, RMSD, negative-distance rates,
  and exact triangle-inequality violations. This separates accurate local
  distance prediction from distances that form a coherent 3D protein.
- **Interpretation and comparison:** aggregate results across proteins and
  layers, analyze coefficients or feature importance, and relate decodability
  to protein size and structural reconstruction quality.

## Data and evaluation

The primary benchmark is a TM-score-filtered combined CAMEO + ECOD collection
(currently 554 discovered proteins). Each protein contributes saved pair-block
tensors and predicted PDB coordinates. Analyses use C-alpha distance matrices
as structural targets and report per-protein as well as pooled results, with
held-out proteins kept separate wherever the question is generalization.

## Why this matters

Protein-folding models are often treated as black boxes that map a sequence to
a final structure. This work instead asks how a fold becomes represented inside
the network: whether geometry is linearly available or requires nonlinear
decoding, when a globally consistent structure appears, and which internal
features support it. The longer-term aim is a comparative, mechanistic account
of representation formation across modern protein-folding AI models.

## Implementation at a glance

The codebase is Python-based (NumPy, SciPy, scikit-learn, XGBoost, Biopython,
and Matplotlib) and runs larger layer/protein sweeps on the Delta SLURM cluster.
Outputs are structured CSV summaries, regression and geometry diagnostics,
plots, and visualization-ready reconstructed structures.

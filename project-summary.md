# Project Summary: AlphaFold2 Pair Representation Analysis

## What is AlphaFold2?

AlphaFold2 is a deep learning system developed by DeepMind that predicts protein 3D structures from amino acid sequences with near-experimental accuracy. It uses a neural network architecture called the Evoformer, which processes two key representations through 48 layers:

- **Single representation:** per-residue features (one vector per amino acid)
- **Pair representation:** pairwise residue features (one vector per pair of amino acids)

The pair representation is a 3D tensor of shape `(R, R, 128)` at each layer, where `R` is the number of residues and `128` is the number of feature channels. These representations evolve through the 48 Evoformer layers and ultimately guide the structure module that produces the final 3D coordinates.

## What This Project Does

This project performs **mechanistic interpretability** on AlphaFold's pair representations. We extract the pair representation tensors at every Evoformer layer (0 through 47) for multiple proteins, then use linear probing techniques to understand what geometric and structural information these representations encode.

### Core Analysis Pipeline

1. **Data extraction:** Pair block tensors are saved per-layer per-protein as `.npy` files.
2. **Structure parsing:** Predicted PDB structures are parsed to extract C-alpha coordinates and compute pairwise distance matrices (ground-truth distances between residues).
3. **Linear regression probes:** We train linear models to predict pairwise residue distances from the 128-channel pair representation features at each layer. This reveals how much distance information is linearly accessible at each depth of the network.
4. **Layer-wise analysis:** We track regression R-squared values across all 48 layers to see how distance information builds up through the network.
5. **Multi-protein aggregation:** Results are aggregated across multiple proteins to identify general patterns rather than protein-specific artifacts.

### Analysis Variants

- **Per-protein regression:** Layer-wise R-squared curves for individual proteins
- **Multi-protein regression:** Combined analysis across all proteins
- **Logistic regression with diagonal features:** Classification-based probes incorporating self-representation (diagonal) features
- **K-fold cross-validation:** Robust evaluation of probe generalization

## Goals

The overarching goals of this project are:

1. **Interpretability:** Understand what information AlphaFold's pair representations carry at each layer — specifically, how much they encode about the physical distances between residue pairs in the folded protein.
2. **Layer dynamics:** Characterize how structural information emerges and refines across the 48 Evoformer layers. Do early layers capture coarse distance patterns? Do later layers sharpen fine-grained spatial relationships?
3. **Representation understanding:** Determine whether simple linear probes can decode meaningful structural signals, or whether the information is encoded in more complex, nonlinear ways.
4. **Protein folding insight:** Use these findings to better understand the computational mechanism by which AlphaFold transforms sequence-level features into accurate 3D structure predictions.

## Technical Stack

- **Languages:** Python
- **Key libraries:** NumPy, SciPy, Matplotlib, Biopython, scikit-learn
- **Compute:** SLURM clusters (Delta and ICE) for large-scale regression across proteins and layers
- **Data format:** `.npy` pair block tensors, PDB structure files, CSV result tables

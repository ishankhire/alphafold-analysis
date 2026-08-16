# Pair-Interaction Classification Probe Plan

## Goal

Test whether OpenFold pair representations encode specific biochemical and
structural interactions beyond the information available from residue identity,
pairwise distance, sequence separation, secondary structure, and solvent
exposure.

This is a classification analogue of the existing distance probes. Each
training example is one unordered residue pair from one protein at one trunk
layer. The main input is the bidirectional pair representation, and the output
is the probability that the pair participates in a specified interaction.

The initial experiment should treat each interaction as a separate binary
classification task. It should not combine all interaction types into a single
multiclass problem because interaction labels can overlap.

## Initial interaction tasks

Start with three annotations that can be defined relatively reliably from
atomic coordinates:

1. Salt bridge versus matched non-salt-bridge.
2. Backbone hydrogen bond or beta bridge versus matched non-interaction.
3. Hydrophobic-core contact versus matched non-core contact.

Possible later tasks include disulfide bonds, aromatic interactions,
domain-interface contacts, and specific secondary-structure pairings. These
should be added only after the initial labeling and evaluation pipeline has
been validated.

## Inputs and targets

For protein `p`, unordered residue pair `(i, j)`, and layer `l`, the primary
model input is:

```text
X_pair = concatenate(pair_block_l[i, j, :], pair_block_l[j, i, :])
```

This produces the existing 256-feature bidirectional representation. Feature
matrices should be materialized as `float64` before scaling and fitting.

The binary target comes from structural annotation, for example:

```text
y_salt_bridge = 1 if residues i and j form a salt bridge, otherwise 0
```

Labels must be computed from structure and chemistry, not from the pair
representation. Experimental reference structures are preferred for the final
scientific evaluation. The saved unrelaxed OpenFold structures may be used to
develop and debug the pipeline, but results based on them should be described
as prediction of interactions in OpenFold's eventual output rather than
independent experimental biology.

## Structural annotation

Generate annotations automatically rather than manually.

- Salt bridge: one residue is ASP/GLU, the other is LYS/ARG, and the closest
  charged side-chain atoms are within a fixed threshold such as 4.0 A.
  Histidine should initially be excluded because its protonation is ambiguous.
- Backbone hydrogen bond and beta bridge: use DSSP assignments and hydrogen-
  bond/bridge relationships where available.
- Hydrophobic-core contact: both residues are hydrophobic, their side-chain
  heavy atoms are within a fixed contact threshold such as 4.5 A, and both are
  buried according to a fixed relative-SASA cutoff.

Annotations are multilabel: the same residue pair may satisfy more than one
interaction definition. Save missing-atom, residue-mapping, DSSP, and SASA
failures explicitly rather than silently treating them as negative examples.

## Matched controls

Unmatched negative pairs would make the classification task trivial. For each
positive pair, choose negative pairs matched as closely as practical on:

- amino-acid identities or chemical classes;
- C-alpha distance bin;
- sequence-separation bin;
- secondary-structure context;
- relative solvent exposure;
- protein-size bin.

For example, a salt-bridge negative should preferably be another
ASP/GLU--LYS/ARG pair at a similar C-alpha distance whose charged side chains
do not form a salt bridge. Matching should be performed without using pair
representations.

## Model comparison

Train the following models separately for each interaction type and each of
the 48 layers:

1. **Ordinary-information baseline.** Inputs are residue identities,
   C-alpha distance, sequence separation, secondary structure, and solvent
   exposure.
2. **Pair-representation probe.** Input is only the 256 pair-representation
   features. Use regularized logistic regression as the primary probe.
3. **Combined model.** Inputs are the ordinary baseline features plus the 256
   pair-representation features.

The main scientific comparison is the held-out performance of the combined
model versus the ordinary-information baseline. Improvement from adding the
pair representation would show that it contains interaction-specific signal
beyond the explicitly controlled chemistry and geometry.

The pair-only model is still useful for showing when the interaction becomes
linearly accessible, but it is not sufficient by itself to establish that the
signal is more than residue identity or distance.

## Split and evaluation policy

- Treat the TM-score-filtered CAMEO and ECOD proteins as one combined dataset.
- Split entire proteins, never individual residue pairs, so pairs from the
  same protein cannot appear in both training and evaluation.
- Prefer family-, topology-, or fold-aware separation when reliable ECOD/CATH
  mappings are available. At minimum, use a fixed size-stratified
  protein-level train/validation/test split.
- Use validation data for regularization and threshold selection. Do not use
  the test set for model or threshold selection.
- Reuse identical protein assignments, matched examples, and labels across
  layers and model variants.
- Report AUROC and average precision, with average precision treated as
  especially important for rare interactions. Also report balanced accuracy,
  precision, recall, and calibration where useful.
- Report per-protein metrics in addition to pair-pooled metrics so large
  proteins do not dominate the conclusion.

## Intended outputs

Write aggregate outputs under:

```text
delta_files/pair_interaction_classification/
```

Write annotation outputs under:

```text
delta_files/pair_interaction_classification/annotations/
```

The intended files are:

- `annotations/per_protein/<protein_id>.csv`: one row per unordered residue
  pair. Principal fields are `protein_id`, `source`, residue indices and
  names, `sequence_separation`, `ca_distance`,
  `min_heavy_atom_distance`, DSSP assignments, relative SASA values,
  interaction flags, annotation source, and annotation-validity fields.
- `annotations/annotation_counts.csv`: one row per protein and interaction
  type, with positive, valid-negative, matched-negative, excluded, and total
  pair counts.
- `annotations/exclusions.csv`: one row per annotation failure or excluded
  protein/pair, with identifiers and reason.
- `split_assignments.csv`: one row per usable protein with `protein_id`,
  `source`, residue count, size bin, optional family/fold identifiers, and
  `split`.
- `matched_pair_assignments.csv`: one row per selected positive or negative
  pair and interaction task, including match-set identifier, class label,
  matching variables, and split.
- `interaction_type_summary.csv`: one row per interaction type, layer, model
  variant, and evaluation split. Principal fields are pair-pooled and
  mean-per-protein AUROC, average precision, balanced accuracy, precision,
  recall, calibration error, pair/protein counts, and combined-model gain over
  the ordinary baseline.
- `interaction_type_per_protein.csv`: one row per protein, interaction type,
  layer, and model variant, with class counts and per-protein metrics.
- `coefficients.csv`: one row per interaction type, layer, model variant,
  fold, and feature, with standardized and raw-scale coefficients where
  applicable.
- `predictions.csv`: one row per held-out matched residue pair, interaction
  type, layer, and model variant, with label and predicted probability. If the
  complete file is too large, save it as deterministic per-layer or
  per-interaction shards.
- `summary.txt`: dataset counts, annotation source, definitions and cutoffs,
  split digest, matching policy, model settings, failures, and principal
  held-out results.
- `visualizations/interaction_decodability_over_layers.<png|pdf>`: held-out
  AUROC and average-precision curves for the three model variants.
- `visualizations/pair_representation_gain_over_layers.<png|pdf>`: combined
  model minus ordinary-baseline performance by interaction and layer.

Per-protein predicted structures and annotation artifacts should remain beside
the relevant protein only if needed for inspection; aggregate tables and plots
belong in the multi-protein result directory above.

## Interpretation criteria

A positive result requires more than a high pair-only score. The strongest
evidence would be:

1. the combined model improves materially over the ordinary baseline on
   entirely held-out proteins;
2. the improvement is consistent across proteins rather than driven by a few
   large structures;
3. the result survives stricter matching and family/fold-aware splits;
4. different interaction types exhibit reproducible and distinguishable
   emergence trajectories across layers.

This experiment establishes which biochemical interaction concepts are
encoded and when they become accessible. It does not by itself establish that
the model causally uses those concepts or recover three-residue rules. Those
questions belong to subsequent triangle-context and intervention experiments.

# Biological Interpretability Experiment Plan

## Overall intention

The long-term goal is to use intermediate OpenFold pair representations to
identify biologically interpretable rules about how residue relationships are
formed and refined. The aim is not only to show that properties such as
pairwise distance are linearly decodable. The desired result is a description
of which biochemical and structural contexts cause the model to revise a
residue-pair relationship, followed by tests of whether those apparent rules
are causally used by the model.

Intermediate trunk layers should initially be described as a computational
geometry-formation or constraint-refinement trajectory, not as physical time
in real protein folding. Any proposed biological rule should be checked
against known chemistry and, where practical, experimental structures or
measurements rather than only OpenFold's own final prediction.

The three experiments below form a progression from concept discovery, to
mechanistic explanation, to causal validation.

## Experiment 1: Interaction-type decoding beyond distance

**Question:** Does the pair representation encode why two residues are close,
rather than only their distance?

### Brief steps

1. Annotate residue pairs with interaction types such as salt bridges,
   backbone hydrogen bonds or beta bridges, hydrophobic-core contacts,
   aromatic interactions, and disulfide bonds.
2. Construct matched positive and negative examples with similar amino-acid
   identities, C-alpha distance, sequence separation, secondary structure,
   and solvent exposure.
3. At every layer, compare an ordinary chemistry/geometry baseline, a
   pair-representation classifier, and a combined classifier.
4. Measure when each interaction becomes decodable and whether the pair
   representation improves over the matched ordinary-information baseline.
5. Inspect stable representation directions and layer trajectories to produce
   candidate biological concepts for the later experiments.

A particularly interesting result would be that, at similar distance and
residue identity, the pair representation distinguishes a genuine interaction
from an incidental close pair.

### Evaluation and outputs

Split entire proteins, preferably with family-, topology-, or fold-aware
separation. Report pair-pooled and mean-per-protein AUROC and average
precision. Save aggregate results under:

```text
delta_files/pair_interaction_classification/
```

Principal files are `interaction_type_summary.csv` with one row per
interaction type, layer, model variant, and split; and
`interaction_type_per_protein.csv` with one row per protein, interaction type,
layer, and model variant. The detailed version of this experiment is in
`docs/plans/pair_interaction_classification_probe_plan.md`.

## Experiment 2: Three-residue context predicts distance corrections

**Question:** When the decoded distance between residues `i` and `j` changes
between layers, can that change be explained by a third residue `k`?

### Brief steps

1. Establish a comparable decoded-distance coordinate across layers. Compare
   the existing layer-specific probes with a shared or layer-aligned decoder
   so changes are not artifacts of changing probe coordinates.
2. For every residue pair and layer, calculate the next-layer decoded-distance
   change:

   ```text
   delta_d(l, i, j) = predicted_d(l + 1, i, j) - predicted_d(l, i, j)
   ```

3. Represent candidate triangle contexts `(i, k, j)` using decoded distances,
   triangle slack, the `z_ik` and `z_kj` pair representations, residue chemical
   classes, sequence separations, secondary structure, and solvent exposure.
4. Compare predictors based on current distance, the center pair `z_ij`,
   geometric triangle context, triangle pair representations, and triangle
   context plus biochemical identity.
5. Use sparse models, shallow trees, or another auditable method to identify
   recurring rules, such as particular contexts causing an overlong pair to
   contract or an inconsistent contact to separate.
6. Repeat with held-out proteins and stricter family/fold splits. Do not use
   random pair splits within a single protein for the main conclusion.

This experiment should align context construction with the model's triangle
operations rather than treating generic ranked nearest neighbors as
interchangeable features.

### Evaluation and outputs

Save outputs under:

```text
delta_files/triangle_context_distance_delta/
```

Intended files are:

- `distance_delta_summary.csv`: one row per layer, model variant, and split;
  principal fields include held-out R2, MAE, context gain, protein count, and
  pair count.
- `distance_delta_per_protein.csv`: one row per protein, layer, and model
  variant.
- `mediator_rules.csv`: one row per candidate rule, including layer range,
  residue classes, structural context, effect direction, effect size, support,
  and held-out stability.
- `split_assignments.csv`: one row per protein with source, size/fold fields,
  and evaluation split.
- `visualizations/`: layerwise predictive performance and selected-rule plots.

## Experiment 3: Causal mediator interventions

**Question:** Does a suspected third-residue mediator actually cause the model
to revise the relationship between residues `i` and `j`?

### Brief steps

1. Select roughly 10--20 high-confidence, well-supported rules from Experiment
   2 and find matched instances across held-out proteins.
2. Apply sequence interventions at mediator `k`, such as Gly-to-Ala,
   Pro-to-Ala, charge reversal, or hydrophobic-to-polar substitutions.
3. Where model instrumentation permits, directly ablate, patch, or scale the
   `k`-mediated triangle contribution to the update of `z_ij`.
4. Measure the immediate next-layer decoded-distance change, its persistence
   through later layers, the final C-alpha distance, contact formation or loss,
   secondary-structure changes, confidence, clashes, and chain integrity.
5. Compare against matched non-mediator mutations, chemically conservative
   substitutions, random triangle interventions of equal magnitude, and
   interventions outside the proposed effective layer window.
6. Treat a rule as causally supported only when its predicted directional
   effect generalizes across proteins and exceeds the matched controls.

### Evaluation and outputs

Save outputs under:

```text
delta_files/causal_triangle_interventions/
```

Intended files are:

- `causal_interventions.csv`: one row per protein, residue triple,
  intervention, and intervention layer; principal fields include original and
  intervened decoded distances, final distances, contact changes, confidence
  changes, and control identifiers.
- `causal_rule_summary.csv`: one row per candidate rule, with mean causal
  effect, confidence interval, protein count, instance count, and
  control-adjusted effect.
- `matched_controls.csv`: one row per intervention/control match.
- `visualizations/`: rule-level effect plots and a small number of structural
  case studies.

## Possible final paper narrative

1. Pair representations encode specific biochemical interaction concepts in
   addition to pairwise distance.
2. Different interaction and contact classes are refined at different trunk
   stages.
3. Layer-to-layer geometric corrections can be explained by structured
   three-residue context that mirrors the model's triangle operations.
4. Selected mediator rules survive protein/fold-level generalization and are
   confirmed through mutations or internal interventions.

This would support a stronger conclusion than probe accuracy alone: the model
uses identifiable, context-dependent rules to propagate biochemical and
geometric constraints through pair space.

# Project Summary: How Protein-Folding Models Build Geometry

## Research question

This project asks how a protein-folding AI turns sequence and evolutionary
information into three-dimensional structure. I study OpenFold's *pair
representation*: a 128-dimensional feature vector for every pair of amino
acids, updated through 48 Evoformer layers. Rather than examining only the
final predicted structure, I train simple probes at different layers to ask
when distances and recognizable structural motifs become accessible. The main
benchmark contains 552 usable proteins from a CAMEO and ECOD
set; shared-model results below are measured on 111 held-out proteins.

## Four distance-prediction models

$R^2$ measures how much of the variation in residue-pair distances a model
explains. At layer 47, mean per-protein $R^2$ was:

| Model | Mean $R^2$ | Interpretation |
|---|---:|---|
| Shared linear (general) | 0.936 | One transferable rule for all proteins |
| Per-protein linear | 0.985 | One tailored rule per protein |
| Sequence-separation-gated linear | 0.957 | Separate rules for six sequence-distance ranges |
| Shared XGBoost | 0.981 | A nonlinear, tree-based transferable rule |

The per-protein result is an in-protein fit using all residue pairs, not a
held-out-protein test, so it is best viewed as a capacity ceiling. XGBoost's
gain over the shared linear model shows that distance information
is present nonlinearly. I am also training sequence-separation-gated models,
which distinguish adjacent, short-range, and long-range pairs. This matters
most locally: at layer 47, pooled mean absolute error for pairs separated by
1, 2–4, and 5–9 sequence positions fell from **3.149, 2.485, and 1.670 Å** with
the general probe to **0.036, 0.257, and 0.497 Å** with gating.

## Helices and beta sheets

I next asked whether decoded distances recover the signatures of alpha
helices and beta pleated sheets. On held-out proteins at layer 47, the general
probe distinguished helix-like from strand-like local geometry with AUROC
0.870; sequence gating raised this to 0.982. Both models also placed true
beta-sheet partners much closer than matched random pairs (AUROC 0.935 general,
0.963 gated). Exact sheet register remains less convincing: at the final
layer, a partner shifted by one residue was sometimes predicted closer than
the true partner. The probes therefore capture sheet membership better than
precise strand alignment.

## Interpretation and geometric caveats

Amino-acid-pair-gated probes provided little improvement by themselves
($R^2=0.936$ at layer 47), while combining amino-acid identity with sequence
gating reached 0.955. This suggests sequence separation is the more important
specialization variable, although rare amino-acid/gate combinations have
limited training support.

Finally, I reconstruct 3D C-alpha traces from predicted distance matrices
using multidimensional scaling (MDS). Best mean TM-scores were 0.869 for the
held-out general probe and 0.942 for per-protein probes. These reconstructions
are encouraging but not proof that the raw predictions form coherent 3D
geometry: late-layer distance matrices still violate triangle inequalities in
about 9% of triples for the general model and 4% for per-protein models, and an
equal-$R^2$ random-error control can produce higher MDS TM-scores. MDS imposes
Euclidean structure, so correlated errors and geometric consistency—not just
distance $R^2$—remain central open questions.

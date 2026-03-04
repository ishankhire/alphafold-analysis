# Regression Problems

## Crux of the problem

The sharp drop from about `0.93` on the small 25-protein run to about `0.53`
on the 184-protein run is not explained by a simple code bug in
`cluster_multi_protein_regression.py`. The regression pipeline is the same:
`StandardScaler + LinearRegression` on concatenated pairwise residue features.

The main issue is conceptual:

- One global linear model is being fit across all proteins at once.
- The loss and reported pooled `R^2` are computed over all residue pairs
  concatenated together.
- A protein with many residues contributes vastly more pair samples because the
  number of residue pairs scales as `n_residues * (n_residues - 1) / 2`.

This means the model is effectively weighted by protein size. Large proteins
dominate the fit, while small and medium proteins can perform very poorly
without strongly affecting the pooled metric.

## Why the earlier runs looked much better

The earlier 10-protein and 25-protein results were not representative of the
full dataset:

- They used only the first sorted proteins, not a random subset.
- That small subset was much easier and more internally consistent.
- On the 25-protein run, every held-out protein had positive per-protein `R^2`.
- On the 184-protein run, many proteins fail badly even though the pooled score
  is still moderate.

So the small-run `R^2` values were optimistic baselines.

## What the 184-protein results show

The full run indicates underfitting / heterogeneity, not a bad train-test split:

- Best pooled test `R^2` is about `0.5347`.
- Best pooled train `R^2` is about `0.5404`.
- The train-test gap is small, so the model is not mainly overfitting.

But the per-protein picture is much worse:

- Mean per-protein test `R^2` is near `0` at the best layer.
- Many proteins have negative per-protein `R^2`.
- Short proteins are especially bad; large proteins perform much better on
  average.

This means the shared linear mapping is not stable across the full protein set.

## Practical interpretation

The core problem is that the current evaluation asks one linear probe to learn a
single universal mapping from pair representation to distance across a highly
heterogeneous protein set, then scores it with a pair-weighted metric that hides
protein-level failures.

In short:

- The model is too coarse for the dataset.
- The metric overweights large proteins.
- The small-subset baselines were biased upward.

## Most likely fixes

- Report mean and median per-protein `R^2`, not only pooled pair-level `R^2`.
- Reweight training so each protein contributes equally instead of by pair
  count.
- Use randomized or stratified subsets for benchmarking instead of the first
  sorted proteins.
- Add protein-level normalization or extra features (for example sequence
  separation, protein length, or normalized distances).

# Float Precision and Protein-Size Regression Takeaways

Generated: 2026-03-22 22:29:20 EDT

## Scope
- This note captures the main conclusions from the float-precision and protein-size debugging discussion around the global multi-protein linear baselines.
- The main question was whether the drop from the smaller 25-protein analyses to the later 184-protein analyses was mainly a protein-size effect, a float32 numerical effect, or some combination of both.

## What Seems Established
- There are at least two overlapping issues:
  - a float32 numerical problem in some exact global linear baselines
  - a real size / heterogeneity / weighting problem that still shows up in float32-backed scripts
- The strongest evidence for the numerical issue is the dedicated `diagnostic_float_precision_128d_regression.py` check: on the same split and same data, the `float32` run had near-zero / slightly negative held-out mean per-protein `R^2`, while the `float64` run jumped to strong performance.
- The strongest evidence for the size / heterogeneity issue is that all-protein pooled baselines improve a lot when the training setup changes in protein-aware ways:
  - protein-equal weighting helps
  - excluding very large proteins helps
  - separate size-bucket models help dramatically

## What We Can Actually Prove About Dtypes
- Pair-block arrays on disk are float32-backed. A local pair-block check showed `proteins/6tf4_A/pair_blocks/6tf4_A_pair_block_47.npy` has dtype `float32`.
- `scripts/multi_protein/regression/cluster_multi_protein_regression.py` is effectively a float32-backed path in this environment:
  - it uses `StandardScaler` followed by `LinearRegression`
  - local inspection showed this sklearn path preserves float32 input through scaling and into the linear solve
  - important caveat: this script uses a pooled pair-level split, not a protein-level split
- The saved 25-protein and 184-protein k-fold lineages in this repo were also produced by `StandardScaler + LinearRegression` scripts, so they were effectively float32-backed in this environment:
  - `scripts/cluster_scripts/test2_multi_protein_regression.py`
  - `scripts/cluster_scripts/cluster_multi_protein_regression.py`
- `scripts/cluster_scripts/diagnostics/diagnostic_lin_tree_context_reg.py` is definitely float32-backed:
  - it allocates feature matrices as float32
  - the installed `lineartree` package explicitly validates input as float32 before fitting
- `scripts/cluster_scripts/diagnostics/diagnostic_residue_weighted_regression.py` is also definitely float32-backed:
  - scaler statistics are stored as float32
  - train / test matrices are built as float32
  - `LinearRegression` is fit directly on those float32 matrices
- Honest caveat: this proves the current code paths and the saved-output lineages we inspected are float32-backed. It does not prove the exact commit or runtime details for every historical cluster run unless the exact job script and commit are known.

## Why Float64 Can Help Even If The Raw Data Started As Float32
- Widening from float32 to float64 does not recover missing raw digits from disk. In that narrow sense, it really is "the same values stored more exactly afterward."
- It still changes the later arithmetic:
  - centering and scaling happen in float64 instead of float32
  - dot products and least-squares operations happen in float64
  - the fitted coefficients are computed in float64
- The most important solver detail is the least-squares cutoff used by sklearn's dense `LinearRegression` path:
  - it calls `scipy.linalg.lstsq` with `cond = max(X.shape) * eps(dtype)`
  - for the large 128d precision diagnostic, this was about `0.808` in float32 versus about `1.5e-9` in float64
  - SciPy treats singular values smaller than `cond * largest_singular_value` as zero
- That means the float32 and float64 fits are not just "the same solve with slightly different rounding." The float32 solve can drop many more weak directions and act like a much lower-rank regression.

## Why The 25-Protein Setup Likely Looked So Much Better
- The smaller 25-protein benchmark was easier in more than one way.
- First, it had far fewer training rows. The saved 25-protein folds had roughly `0.74M` to `0.99M` training residue pairs, while the 184-protein folds had roughly `6.15M` to `7.14M`.
- In the float32 dense least-squares path, that row-count increase directly makes the SVD cutoff harsher:
  - 25-protein max train fold: float32 cutoff scale about `0.118`
  - 184-protein max train fold: float32 cutoff scale about `0.851`
- Second, larger proteins do not just add "more of the same." They also contribute far more pair rows because residue pairs scale roughly like `n(n-1)/2`.
- So adding larger proteins changes both:
  - the distribution of the training data
  - the weighting of the loss, because very large proteins contribute many more residue-pair examples
- Third, the 25-protein subset appears to have been optimistic / nonrepresentative relative to the full discovered-protein set. This was already hinted at in earlier repo notes, and the later protein-aware diagnostics support that interpretation.

## What Seems Most Likely Right Now
- The big 25-protein to 184-protein drop is probably not one single issue.
- The most likely combined explanation is:
  - the 25-protein benchmark was easier and not fully representative
  - pooled pair-level training heavily upweights large proteins
  - the best linear decoder appears to shift across size regimes or related structural regimes
  - the huge increase in row count makes float32 least-squares much less reliable
- So the answer is probably "both": larger proteins matter, and the much larger total number of training rows also matters numerically.

## Interpreting The Linear-Tree Size Story
- The gradual leaf-coefficient cosine shifts in the linear-tree diagnostics do not look like an obvious pure float artifact.
- Reason: the tree-based and context-based diagnostics remain strong even though those code paths are also float32-backed.
- That suggests the size-conditioned coefficient differences may be capturing something real about how the best decoding rule changes across protein regimes.
- At the same time, that does not by itself prove there is a single clean "protein size axis" in the representation.
- Other things that may covary with size and drive the coefficient shifts:
  - sequence-separation mix
  - typical distance scale
  - fold complexity
  - domain structure
  - prevalence of long-range contacts

## Best Next Analyses
- Run a fixed 184-protein protein-level split and sweep `row_cap_per_protein` in something like `{1k, 5k, 20k, all}` for the same global baseline in both `float32` and `float64`.
  - This is the cleanest single test of "row-count-driven float32 failure" versus "real cross-protein heterogeneity."
- Match total training-row count between the 25-protein and 184-protein settings.
  - If 184 proteins still underperform badly at the same total row count, heterogeneity is a bigger culprit.
  - If performance mostly recovers, row count and numerical issues are playing a major role.
- Build a cross-bucket transfer matrix.
  - Train one linear model per size bucket and evaluate each model on every other size bucket.
  - This tests whether the learned decoder is mainly bucket-specific or transfers well across size regimes.
- Re-run the exact global baseline with a custom least-squares call where the rank cutoff is controlled directly.
  - This isolates "dtype-dependent truncation" from generic float32 rounding effects.
- Compute singular-value or effective-rank diagnostics on capped standardized design matrices.
  - Overlay the float32 and float64 cutoffs on the same spectrum.
  - This would give the most direct explanation for why the float32 solver is discarding useful directions.
- Repeat the linear-tree and bucketed analyses across multiple random protein splits.
  - This tests whether the learned thresholds and leaf-coefficient trends are stable or mostly split-specific noise.

## Recommended Single Next Experiment
- If only one follow-up should be run first, the best choice is:
  - fixed 184-protein protein-level split
  - one global baseline
  - `row_cap_per_protein` sweep such as `{5k, 20k, all}`
  - compare `float32` and `float64`
- That experiment should separate three possibilities quickly:
  - mainly a float32 row-count issue
  - mainly a true cross-protein heterogeneity issue
  - a combination of both

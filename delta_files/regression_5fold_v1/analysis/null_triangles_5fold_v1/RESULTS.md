# Null models for triangle violations: results

552 proteins (154 CAMEO + 398 ECOD), layers 0–47, every triangle, K=3 draws per null, protein-equal means, 95% paired-protein bootstrap (2000 replicates, seed 20260902). Below / above: the 95% band of observed − null lies entirely below / above 0. Second-peak contrast: per-protein mean over L30–34 minus mean over L20–22, paired bootstrap; it survives when the interval lies above 0.

## Shared linear

**Normalized mean violation**

- Observed vs Null A (Gaussian), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null B (permuted), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.005027 [0.004562, 0.005538]; in Null B alone: 0.003435 [0.002829, 0.004088]; in observed − Null B: 0.001593 [0.001321, 0.001865] (survives).

**Violation fraction**

- Observed vs Null A (Gaussian), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null B (permuted), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.01905 [0.01724, 0.0209]; in Null B alone: 0.009606 [0.007654, 0.01156]; in observed − Null B: 0.009441 [0.008811, 0.01011] (survives).

## Per-protein linear

**Normalized mean violation**

- Observed vs Null A (Gaussian), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null B (permuted), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.001779 [0.001648, 0.001915]; in Null B alone: 0.001118 [0.0009515, 0.001294]; in observed − Null B: 0.0006609 [0.0005913, 0.0007365] (survives).

**Violation fraction**

- Observed vs Null A (Gaussian), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null B (permuted), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.008764 [0.008377, 0.009162]; in Null B alone: 0.003346 [0.002898, 0.003795]; in observed − Null B: 0.005417 [0.005133, 0.005701] (survives).

## XGBoost

**Normalized mean violation**

- Observed vs Null A (Gaussian), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null B (permuted), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.0005355 [0.0003961, 0.0006791]; in Null B alone: -0.0006658 [-0.001017, -0.0003119]; in observed − Null B: 0.001201 [0.0008954, 0.001509] (survives).

**Violation fraction**

- Observed vs Null A (Gaussian), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null B (permuted), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.00359 [0.002613, 0.00459]; in Null B alone: -0.00238 [-0.003419, -0.001242]; in observed − Null B: 0.005969 [0.005279, 0.006703] (survives).

## Sequence-gated

**Normalized mean violation**

- Observed vs Null A (Gaussian), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null B (permuted), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.002171 [0.001977, 0.002374]; in Null B alone: 0.0005017 [0.000169, 0.0008181]; in observed − Null B: 0.001669 [0.001443, 0.001909] (survives).

**Violation fraction**

- Observed vs Null A (Gaussian), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null B (permuted), B10: below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.01054 [0.009384, 0.01177]; in Null B alone: 0.002822 [0.001488, 0.004218]; in observed − Null B: 0.007723 [0.007108, 0.008362] (survives).

## Binning sensitivity (Shared linear, Null B)

- Normalized mean violation, B5: observed below Null B in L0–L47; gap at L47 -0.003651 [-0.004641, -0.002988].
- Normalized mean violation, B10: observed below Null B in L0–L47; gap at L47 -0.003432 [-0.004186, -0.002926].
- Normalized mean violation, B20: observed below Null B in L0–L47; gap at L47 -0.003192 [-0.003758, -0.002787].
- Normalized mean violation, B10xS6: observed below Null B in L0–L47; gap at L47 -0.003298 [-0.004042, -0.002811].
- Violation fraction, B5: observed below Null B in L0–L47; gap at L47 -0.01338 [-0.0163, -0.01111].
- Violation fraction, B10: observed below Null B in L0–L47; gap at L47 -0.01348 [-0.01588, -0.0117].
- Violation fraction, B20: observed below Null B in L0–L47; gap at L47 -0.01254 [-0.01455, -0.01103].
- Violation fraction, B10xS6: observed below Null B in L0–L47; gap at L47 -0.01312 [-0.01548, -0.01137].

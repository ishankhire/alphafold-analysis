# Null models for triangle violations: results

154 CAMEO proteins, layers 0–47, protein-equal means, 95% paired-protein bootstrap (2000 replicates, seed 20260902). Gaps compare observed and null on the same triangle sample. Below / above: the 95% band of observed − null lies entirely below / above 0. Second-peak contrast: per-protein mean over L30–34 minus mean over L20–22, paired bootstrap; it survives when the interval lies above 0.

## Shared OLS

**Normalized mean violation**

- Observed vs Null B (permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null C (residue offsets + permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.003654 [0.002656, 0.004986]; in Null B alone: 0.00166 [0.0003949, 0.003305]; in observed − Null B: 0.001996 [0.001387, 0.002588] (survives).

**Violation fraction**

- Observed vs Null B (permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null C (residue offsets + permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.01369 [0.01003, 0.01775]; in Null B alone: 0.003796 [4.982e-05, 0.008291]; in observed − Null B: 0.009892 [0.008554, 0.01124] (survives).

## Per-protein OLS

**Normalized mean violation**

- Observed vs Null B (permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null C (residue offsets + permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.002552 [0.002251, 0.002899]; in Null B alone: 0.001781 [0.00136, 0.002226]; in observed − Null B: 0.0007773 [0.0005884, 0.0009766] (survives).

**Violation fraction**

- Observed vs Null B (permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null C (residue offsets + permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.01072 [0.009916, 0.01155]; in Null B alone: 0.004863 [0.003809, 0.005898]; in observed − Null B: 0.00587 [0.005213, 0.00649] (survives).

## Seq-gated OLS

**Normalized mean violation**

- Observed vs Null B (permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null C (residue offsets + permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.001296 [0.0009425, 0.00166]; in Null B alone: -0.00071 [-0.001389, -2.541e-05]; in observed − Null B: 0.002005 [0.001471, 0.002591] (survives).

**Violation fraction**

- Observed vs Null B (permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null C (residue offsets + permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.005995 [0.003835, 0.008212]; in Null B alone: -0.001769 [-0.00421, 0.0008951]; in observed − Null B: 0.007746 [0.006377, 0.009131] (survives).

## XGBoost

**Normalized mean violation**

- Observed vs Null B (permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null C (residue offsets + permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.0004821 [0.0001768, 0.0008091]; in Null B alone: -0.001194 [-0.002233, -0.0001591]; in observed − Null B: 0.00168 [0.0007875, 0.002656] (survives).

**Violation fraction**

- Observed vs Null B (permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Observed vs Null C (residue offsets + permuted): below in L0–L47; above in none; matches (band includes 0) in none.
- Second peak in observed: 0.003833 [0.001812, 0.005924]; in Null B alone: -0.002418 [-0.004988, 0.0001658]; in observed − Null B: 0.006273 [0.004512, 0.00802] (survives).

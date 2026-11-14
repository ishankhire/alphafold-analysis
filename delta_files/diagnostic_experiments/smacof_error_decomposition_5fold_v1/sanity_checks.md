# SMACOF error decomposition: sanity checks

## Gates (all layers passed before anything was written)

- Identity and stress: 741888 rows checked, 0 failures. Largest |total² − (inconsistent² + consistent² + cross)|: SMACOF 9.09e-13 Å², classical 1.82e-12 Å².
- SMACOF raw stress ≤ classical raw stress on every row; largest ratio 0.987129.
- Control pairing with the triangle run: 741888 vectors matched on negative-value count (and violation count on small proteins); 556416 shuffled-control histogram checks.
- Classical-MDS start vs canonical geometry: worst gaps {'parity_negative': 0, 'parity_rmsd': 1.017891737831178e-05, 'parity_stress': 2.2286160847317937e-08}.

## Stress minimum

- Scale ratio Σd̂·p / Σp² over converged runs: min 1.00000000, max 1.00000000 (1 at a stress minimum). Classical MDS: min 0.7494, max 1.7629.
- Converged runs: 741888 of 741888. Iterations: median 136, 95th percentile 670, max 18765.

## Tolerance

Runs stopped at a relative stress decrease of 1e-10 per iteration. Change in each row's RMS term had the run stopped when a looser tolerance was first met:

| Stop at | Part | Rows changed by > 1e-4 Å | > 1e-2 Å | Largest change (Å) | Largest change in a cohort curve (Å) |
|---|---|---|---|---|---|
| 1e-6 | inconsistent | 36292 | 1178 | 0.504 | 0.00165 |
| 1e-6 | consistent | 619550 | 18547 | 1.894 | 0.00782 |
| 1e-8 | inconsistent | 1225 | 110 | 0.263 | 0.00040 |
| 1e-8 | consistent | 198678 | 682 | 1.698 | 0.00192 |

## Agreement with the canonical accuracy tables

- √total² against the canonical per-protein RMSE, rows with no negative prediction (31935 of 105984): largest gap 7.64e-08 Å (the cache is float32).

## Monte Carlo spread of the controls (3 draws)

| Control | Term | Mean per-protein draw SD (Å²) | Median bootstrap half-width of the cohort mean (Å²) |
|---|---|---|---|
| Control: Gaussian-matched noise | smacof_inconsistent_sq | 0.1324 | 1.991 |
| Control: Gaussian-matched noise | smacof_consistent_sq | 0.1563 | 4.496 |
| Control: shuffled errors | smacof_inconsistent_sq | 0.05331 | 1.889 |
| Control: shuffled errors | smacof_consistent_sq | 0.09671 | 4.462 |

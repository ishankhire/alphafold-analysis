# Sanity checks: per-channel MDS

Code revision: `3102531`.

| Check | Value | Pass |
|---|---|---|
| Proteins and layers | 154 proteins, 48 layers | yes |
| Channel rows | 946176 (expected 946176) | yes |
| Unique (layer, protein, channel) keys | True | yes |
| Constant channels | 0 | yes |
| Stress finite and non-negative on ok rows | min 0.1307, max 0.8004 | yes |
| Kept sign has the lower stress | max excess -2.99e-08 | yes |
| True distances embed exactly (raw) | max stress 1.09e-15 | yes |
| Sequence separation is one-dimensional (raw) | min top-1 fraction 1.000000 | yes |
| Batched MDS parity with helpers.classical_mds | max difference 1.80e-14 | yes |
| Probe rows joined | 7392 of 7392 | yes |
| Summary matches the channel table | max difference 2.22e-16 | yes |

Share of protein-channel pairs whose stress is below their permuted stress: layer 0 0.897, layer 12 0.947, layer 24 0.996, layer 36 0.984, layer 47 0.979.

# Shared linear coefficient comparisons across layers

The primary figure, `shared_ols_layer_cosine_heatmap.png` (also PDF), compares 256 standardized coefficients, excluding intercepts. The raw-scale companion uses coefficients in original feature units. Features retain the fixed order [Z_ij channels 0–127, Z_ji channels 0–127].

For each layer pair, compute cosine similarity within each outer fold, then average the five values equally. This compares models trained on matched protein sets; coefficient vectors are not averaged before calculating cosine similarity. The fixed color scale runs from −1 to +1. Similarity measures coefficient direction in the saved channel coordinates, not vector magnitude or prediction transfer.

Each `*_matrix.csv` has 48 layer rows and 48 similarity columns plus a layer index. `fold_similarities.csv` retains one row per fold/ordered layer pair (11,520 rows), with standardized and raw-scale cosines. `source_artifacts.csv` records all 240 inputs, their verified replay-index checksums, and saved fit statuses; all fits are retained. `generation_summary.json` records aggregation and descriptive statistics.

Regenerate from the repository root with `./.venv/bin/python scripts/multi_protein/plot_shared_linear_layer_cosines.py`. The source NPZ coefficient artifacts must be present locally; they are ignored by Git.

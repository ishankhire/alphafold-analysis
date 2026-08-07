# Layer-47 coefficient clustering

Input: `delta_files/per_protein_base256_combined/coefficients.csv`

Loaded 552 proteins with 256 finite standardized coefficients at layer 47. The intercept was excluded.

## Method

K-means was evaluated for k=1–10 directly in 256 coefficient dimensions. The primary direction representation L2-normalizes each protein vector; the magnitude-sensitive representation leaves the standardized vector unchanged. Candidate k is the maximum-silhouette solution over k=2–10. PCA is used only for display. Stability is adjusted Rand index against 25 fits trained on deterministic 80% protein subsamples.

## Selected candidate solutions

- **Direction (unit-normalized):** k=2; silhouette=0.2091; mean subsample ARI=0.9182; cluster sizes=[337, 215].
- **Magnitude-sensitive:** k=2; silhouette=0.4341; mean subsample ARI=0.9608; cluster sizes=[459, 93].

## Metadata associations

- **Direction (unit-normalized):** log-length eta²=0.3595; source Cramér's V=0.4143; size-bin Cramér's V=0.6125; ECOD-category Cramér's V=0.1842.
- **Magnitude-sensitive:** log-length eta²=0.2472; source Cramér's V=0.4538; size-bin Cramér's V=0.5035; ECOD-category Cramér's V=0.1439.

## Interpretation

The selected k values are visualization candidates, not proof that proteins form discrete biological classes. The inertia curve, silhouette, stability, cluster-size balance, and metadata associations should be read together. Strong length, size-bin, or CAMEO/ECOD-source associations indicate that a partition may reflect dataset composition or coefficient scale rather than a distinct structural category. ECOD-category tests use only proteins with available broad-category annotations. All association p-values are descriptive because the clusters and k values were learned from the same data.

## Generated artifacts

- `layer47_kmeans_k_diagnostics.csv`
- `layer47_kmeans_assignments.csv`
- `layer47_kmeans_centroids.csv`
- `layer47_kmeans_cluster_summary.csv`
- `layer47_kmeans_metadata_associations.csv`
- `layer47_kmeans_diagnostics.png`
- `layer47_kmeans_pca_clusters.png`
- `layer47_kmeans_cluster_profiles.png`
- `layer47_kmeans_metadata_diagnostics.png`

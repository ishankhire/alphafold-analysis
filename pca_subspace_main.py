# -*- coding: utf-8 -*-
"""
analyze_pca_similarity.py

Purpose:
  - Load pair representation blocks: (r, r, C) per layer from .npy files.
  - Spatially reduce to samples x channels (flatten or mean).
  - Compute PCA per layer (basis and explained variance ratio).
  - Compute subspace similarity between layers using principal angles.

I/O assumptions:
  Files live at: {base_dir}/{protein}/{protein}_pair_block_{i}.npy
  Each file has shape (r, r, C) with the same r and C across layers.

Usage:
  python analyze_pca_similarity.py \
      --base_dir /path/to/data \
      --protein  example_protein \
      --layers   8 \
      --k        32 \
      --metric   affinity \
      --spatial_mode flatten \
      --subsample 5000 \
      --seed 0

Only dependencies: numpy (uses numpy.linalg.svd).
"""

import os
import argparse
import numpy as np

# ----------------------------
# Utilities
# ----------------------------
def to64(x: np.ndarray) -> np.ndarray:
    return x.astype(np.float64, copy=False)

def load_layer_paths(base_dir: str, protein: str, layers: int):
    """Return canonical .npy paths for each layer."""
    root = os.path.join(base_dir, protein)
    layer_indices = list(range(layers))  # All layers from 0 to layers-1
    return [os.path.join(root, f"{protein}_pair_block_{i}.npy") for i in layer_indices]

def load_block(path: str, mmap_mode: str = "r") -> np.memmap:
    """Load a single (r, r, C) block with memory mapping."""
    return np.load(path, mmap_mode=mmap_mode)

def spatial_reduce(arr: np.ndarray, mode: str = "flatten",
                   subsample: int | None = None,
                   seed: int | None = 0) -> np.ndarray:
    """
    Convert (r, r, C) -> (N, C).
    - mode='mean': returns (1, C)
    - mode='flatten': returns (r*r, C) or a subsampled subset
    """
    r, _, C = arr.shape
    if mode == "mean":
        return to64(arr.mean(axis=(0, 1), keepdims=True))

    if mode == "flatten":
        X = arr.reshape(r * r, C)
        if subsample is not None and subsample < X.shape[0]:
            rng = np.random.default_rng(seed)
            idx = rng.choice(X.shape[0], size=subsample, replace=False)
            X = X[idx]
        return to64(X)

    raise ValueError(f"Unknown spatial mode: {mode}")

# ----------------------------
# PCA
# ----------------------------
def pca_topk(X: np.ndarray, k: int = 32, center: bool = True):
    """
    PCA via SVD on (N, C) data.
    Returns:
      U: (C, k_eff)  top-k right singular vectors (PC loadings on channels)
      evr: (k,)      explained variance ratio for top-k components
      mu: (C,)       mean used for centering
    """
    X = to64(X)
    if center:
        mu = X.mean(axis=0, keepdims=True)
        Xc = X - mu
    else:
        mu = np.zeros((1, X.shape[1]), dtype=np.float64)
        Xc = X

    # economy SVD
    U_samp, S, Vt = np.linalg.svd(Xc, full_matrices=False)  # Xc = U_samp @ diag(S) @ Vt
    V = Vt.T                                                # PCs in columns
    k_eff = min(k, V.shape[1])

    U = V[:, :k_eff]                                        # (C, k_eff)
    total = S @ S if S.size else 1e-12
    ev_full = (S**2) / max(total, 1e-12)

    evr = np.zeros(k, dtype=np.float64)
    evr[:k_eff] = ev_full[:k_eff]
    return U, evr, mu.squeeze(0)

def project_and_reshape(arr: np.ndarray, U: np.ndarray, mu: np.ndarray) -> np.ndarray:
    """
    Project (r, r, C) onto top k PCs and return as (r, r, k).
    
    Args:
      arr: (r, r, C) original spatial array
      U: (C, k) PC basis from pca_topk
      mu: (C,) mean from pca_topk
    
    Returns:
      (r, r, k) projected data maintaining spatial structure
    """
    r, _, C = arr.shape
    k = U.shape[1]
    
    # Flatten to (r*r, C)
    X = arr.reshape(r * r, C).astype(np.float64)
    
    # Center and project: (r*r, C) @ (C, k) = (r*r, k)
    X_centered = X - mu[np.newaxis, :]
    X_proj = X_centered @ U
    
    # Reshape back to spatial: (r, r, k)
    return X_proj.reshape(r, r, k)

def save_pca_projections(base_dir: str,
                        protein: str,
                        layers: int,
                        k: int = 32,
                        output_dir: str = "pca_projections",
                        seed: int = 0):
    """
    Compute PCA for each layer and save projected (r, r, k) arrays.
    
    Files saved as: {output_dir}/{protein}_pair_block_{i}_pca{k}.npy
    """
    paths = load_layer_paths(base_dir, protein, layers)
    layer_indices = list(range(layers))  # All layers from 0 to layers-1

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    for i, (layer_idx, path) in enumerate(zip(layer_indices, paths)):
        print(f"Processing layer {layer_idx}...")
        
        # Load original (r, r, C)
        arr = load_block(path, mmap_mode="r")
        r, _, C = arr.shape
        
        # Flatten for PCA
        X = arr.reshape(r * r, C).astype(np.float64)
        
        # Compute PCA
        k_eff = min(k, C)
        U, evr, mu = pca_topk(X, k=k_eff, center=True)
        
        # Project back maintaining spatial structure
        arr_proj = project_and_reshape(arr, U, mu)
        
        # Save
        output_path = os.path.join(output_dir, f"{protein}_pair_block_{layer_idx}_pca{k}.npy")
        np.save(output_path, arr_proj.astype(np.float32))
        
        print(f"  Saved {output_path} with shape {arr_proj.shape}")
        print(f"  Explained variance (top 5 PCs): {evr[:5]}")
    
    print(f"\nAll PCA projections saved to {output_dir}/")

# ----------------------------
# Subspace similarity
# ----------------------------
def principal_cosines(U: np.ndarray, V: np.ndarray) -> np.ndarray:
    """
    Principal cosines between two column-orthonormal bases U, V in R^{C x k}.
    """
    k = min(U.shape[1], V.shape[1])
    if k == 0:
        return np.zeros(0, dtype=np.float64)
    M = U[:, :k].T @ V[:, :k]                 # (k, k)
    _, s, _ = np.linalg.svd(M, full_matrices=False)
    return np.clip(s[:k], 0.0, 1.0)

def subspace_similarity(U: np.ndarray, V: np.ndarray,
                        metric: str = "affinity",
                        normalize: bool = True) -> float:
    """
    Supported metrics:
      Similarities:
        - 'affinity'   : mean cos^2(theta_i)  in [0,1]
        - 'maxcorr'    : max cos^2
        - 'mincorr'    : min cos^2
      Distances (optionally normalized to [0,1] similarity):
        - 'chordal'    : sum(1 - cos^2), normalize -> 1 - d^2/k
        - 'projection' : sqrt(1 - min cos^2), normalize -> 1 - d
        - 'procrustes' : 2 * sum(1 - cos),   normalize -> 1 - d^2/(2k)
        - 'martin'     : sum(log((1+c)/(1-c)))^2, normalize -> 1/(1 + d^2)
        - 'grassmann'  : ||theta||_2, normalize by max (all θ=π/2)
    """
    c = principal_cosines(U, V)
    k = len(c)
    eps = 1e-12
    metric = metric.lower()

    # Similarities
    if metric == "affinity":
        return float(np.mean(c**2)) if k else 0.0
    if metric == "maxcorr":
        return float(np.max(c**2)) if k else 0.0
    if metric == "mincorr":
        return float(np.min(c**2)) if k else 0.0

    # Distances -> optionally convert to similarity
    if metric == "chordal":
        d2 = float(np.sum(1 - c**2))
        return (1 - d2 / max(k, eps)) if normalize else d2

    if metric == "projection":
        d = float(np.sqrt(1 - np.min(c**2))) if k else 0.0
        return (1 - d) if normalize else d

    if metric == "procrustes":
        d2 = float(2 * np.sum(1 - c))
        return (1 - d2 / max(2 * k, eps)) if normalize else d2

    if metric == "martin":
        z = np.log((1 + c) / np.maximum(1e-9, 1 - c))
        d2 = float(np.sum(z**2))
        return (1 / (1 + d2)) if normalize else d2

    if metric in ("grassmann", "geodesic", "riemannian"):
        theta = np.arccos(np.clip(c, -1.0, 1.0))
        d = float(np.linalg.norm(theta))
        if not normalize:
            return d
        dmax = 0.5 * np.pi * np.sqrt(k) if k else 1.0
        return 1 - d / max(dmax, eps)

    raise ValueError(f"Unknown metric: {metric}")

# ----------------------------
# Core: PCA + similarity
# ----------------------------
def analyze(base_dir: str,
            protein: str,
            layers: int,
            k: int = 32,
            spatial_mode: str = "flatten",
            subsample: int | None = None,
            seed: int = 0,
            metric: str = "affinity",
            normalize_similarity: bool = True):
    """
    Returns:
      layer_names: [str]                 length L
      sim_mat    : (L, L)                subspace similarity matrix
      evr_mat    : (L, k)                explained variance ratio per layer
      bases      : dict[name] -> (C, k') PC bases per layer
    """
    paths = load_layer_paths(base_dir, protein, layers)
    # Detect C from first layer for sanity
    arr0 = load_block(paths[0], mmap_mode="r")
    r0, r1, C = arr0.shape
    assert r0 == r1, "First layer must be square in spatial dims"

    # Run PCA per layer
    layer_names = [f"layer{idx:02d}" for idx in range(layers)]
    bases = {}
    evr_rows = []

    for i, path in enumerate(paths):
        arr = load_block(path, mmap_mode="r")
        assert arr.shape[2] == C, f"Channel mismatch at {path}"
        X = spatial_reduce(arr, mode=spatial_mode, subsample=subsample, seed=seed)

        if k > C:
            k_eff = C
        else:
            k_eff = k

        U, evr, _ = pca_topk(X, k=k_eff, center=True)
        bases[layer_names[i]] = U

        # pad/truncate evr to length k for a consistent matrix
        evr_row = np.zeros(k, dtype=np.float64)
        evr_row[:U.shape[1]] = evr[:U.shape[1]]
        evr_rows.append(evr_row)

    evr_mat = np.stack(evr_rows, axis=0)  # (L, k)

    # Similarity matrix
    L = len(layer_names)
    sim_mat = np.zeros((L, L), dtype=np.float64)
    for i in range(L):
        for j in range(L):
            sim_mat[i, j] = subspace_similarity(
                bases[layer_names[i]],
                bases[layer_names[j]],
                metric=metric,
                normalize=normalize_similarity,
            )

    return layer_names, sim_mat, evr_mat, bases

# ----------------------------
# CLI
# ----------------------------
def main():
    p = argparse.ArgumentParser(description="PCA + Subspace Similarity (minimal)")
    p.add_argument("--base_dir", required=True)
    p.add_argument("--protein", required=True)
    p.add_argument("--layers", type=int, required=True)
    p.add_argument("--k", type=int, default=32)
    p.add_argument("--spatial_mode", default="flatten", choices=["flatten", "mean"])
    p.add_argument("--subsample", type=int, default=None)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--metric", default="affinity",
                   choices=["affinity", "maxcorr", "mincorr",
                            "chordal", "projection", "procrustes", "martin", "grassmann"])
    p.add_argument("--normalize_similarity", type=lambda x: str(x).lower() in {"1","true","yes","y","on"},
                   default=True)
    p.add_argument("--save_projections", action="store_true",
                   help="Save (r, r, k) PCA projections for each layer")
    p.add_argument("--output_dir", default="pca_projections",
                   help="Directory to save PCA projections")

    args = p.parse_args()

    # Save PCA projections if requested
    if args.save_projections:
        save_pca_projections(
            base_dir=args.base_dir,
            protein=args.protein,
            layers=args.layers,
            k=args.k,
            output_dir=args.output_dir,
            seed=args.seed
        )
        return

    names, sim, evr, _ = analyze(
        base_dir=args.base_dir,
        protein=args.protein,
        layers=args.layers,
        k=args.k,
        spatial_mode=args.spatial_mode,
        subsample=args.subsample,
        seed=args.seed,
        metric=args.metric,
        normalize_similarity=args.normalize_similarity
    )

    # Simple, verifiable stdout
    np.set_printoptions(precision=4, suppress=True, linewidth=140)
    print("\nLayers:")
    print(", ".join(names))
    print(f"\nSubspace similarity matrix [{args.metric}]:")
    print(sim)
    print(f"\nExplained variance ratio (per layer, first {args.k} PCs):")
    print(evr)

if __name__ == "__main__":
    main()

"""Deterministic, compact triangle-dominance primitives."""
from __future__ import annotations
import hashlib, json, math
from pathlib import Path
import numpy as np

VARIANTS = ("directional_raw", "directional_abs", "abs_then_sym", "sym_then_abs")
PRIMARY_METRICS = ("signed_margin", "positive_violation", "signed_normalized_margin", "positive_normalized_violation")
ALL_METRICS = PRIMARY_METRICS + ("positive_channel_fraction", "zero_scale_fraction")
N_CHANNELS, SAMPLE_CAP, CHUNK_SIZE = 128, 20_000, 2_000
SAMPLE_VERSION = "pair_triangle_samples_v1"

def sha256_bytes(data): return hashlib.sha256(data).hexdigest()
def sha256_file(path): return sha256_bytes(Path(path).read_bytes())

def derived_seed(protein_id, base_seed=20260827):
    return int.from_bytes(hashlib.sha256(f"{SAMPLE_VERSION}|{base_seed}|{protein_id}".encode()).digest()[:8], "little")

def unrank_combination(rank, n):
    """Unrank C(n,3) in strict-lower canonical order i>j>k."""
    rank = int(rank); i = n - 1
    while math.comb(i, 2) <= rank: rank -= math.comb(i, 2); i -= 1
    j = i - 1
    while rank >= j:
        rank -= j; j -= 1
    return i, j, rank

def sample_triples(protein_id, n, base_seed=20260827):
    population = math.comb(n, 3) if n >= 3 else 0
    count = min(SAMPLE_CAP, population)
    if not count: return np.empty((0,3), np.int32), np.empty(0, np.int64), derived_seed(protein_id, base_seed)
    rng = np.random.default_rng(derived_seed(protein_id, base_seed))
    ranks = np.sort(rng.choice(population, count, replace=False).astype(np.int64))
    triples = np.array([unrank_combination(r, n) for r in ranks], dtype=np.int32)
    return triples, ranks, derived_seed(protein_id, base_seed)

def validate_samples(triples, n):
    a = np.asarray(triples)
    return (a.ndim == 2 and a.shape[1] == 3 and (len(a)==0 or
        np.all((a[:,0] > a[:,1]) & (a[:,1] > a[:,2])) and np.all((a >= 0) & (a < n))) and
        len({tuple(x) for x in a}) == len(a))

def transform_sides(z, triples):
    i,j,k = np.asarray(triples).T
    # edges are (i,j), (i,k), (j,k), each canonical strict-lower direction.
    lo = np.stack((z[i,j], z[i,k], z[j,k]), axis=1).astype(np.float64, copy=False)
    hi = np.stack((z[j,i], z[k,i], z[k,j]), axis=1).astype(np.float64, copy=False)
    return (lo, np.abs(lo), (np.abs(lo)+np.abs(hi))/2, np.abs((lo+hi)/2))

def channel_measurements(sides):
    a = np.asarray(sides, dtype=np.float64)
    maximum = np.max(a, axis=1); margin = 2*maximum - np.sum(a, axis=1)
    scale = np.sum(np.abs(a), axis=1)
    normalized = np.divide(margin, scale, out=np.zeros_like(margin), where=scale > 0)
    return (margin, np.maximum(margin, 0), normalized, np.maximum(normalized, 0)), scale == 0

def summarize(values):
    x = np.asarray(values, dtype=np.float64); x = x[np.isfinite(x)]
    if not len(x): return (np.nan,)*5 + (0,)
    return (float(np.mean(x)), float(np.std(x)), float(np.median(x)), float(np.quantile(x,.9)), float(np.quantile(x,.99)), len(x))

def atomic_save_npz(path, **arrays):
    path = Path(path); tmp = path.with_name(path.name + ".tmp.npz")
    np.savez_compressed(tmp, **arrays); tmp.replace(path)

__all__ = ["VARIANTS","PRIMARY_METRICS","ALL_METRICS","N_CHANNELS","SAMPLE_CAP","CHUNK_SIZE","SAMPLE_VERSION","derived_seed","unrank_combination","sample_triples","validate_samples","transform_sides","channel_measurements","summarize","atomic_save_npz","sha256_file"]

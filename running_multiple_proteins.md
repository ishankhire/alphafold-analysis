# Running Analysis on Multiple Proteins — Reasoning & Best Practices

This document captures the reasoning behind how to robustly scale the AlphaFold pair representation analysis from a single protein (7B3A) to ~500 proteins on a compute cluster. It is intended as a reference for implementing the changes incrementally.

---

## The Core Problem: Residue Alignment

The current codebase uses `PAIR_OFFSET = 4` to map CIF residue indices to pair block indices. This works for 7B3A because the first 4 residues (`GHMA`) are unmodeled, giving a clean linear shift. **This breaks for any protein where unmodeled residues appear in the middle or end of the sequence, or where there are multiple non-contiguous gaps.**

For 500 proteins, each may have a completely different gap pattern:
- Protein A: first 3 residues missing → offset = 3
- Protein B: residues 50–52 missing (internal loop) → no linear offset works at all
- Protein C: residues 1–5 AND 100–102 missing → multiple non-contiguous gaps

The fix is to build a proper mapping from entity sequence position (= pair block index) to resolved structure residue, using the mmCIF file's `_pdbx_poly_seq_scheme` table.

---

## #1: Fix Residue Alignment — `lib/alignment.py`

### Why `_pdbx_poly_seq_scheme`

This is the authoritative table in every mmCIF file. It has one row per entity sequence position per chain, with:
- `seq_id`: 1-based position in the entity sequence. **Subtract 1 → pair block index (0-based).**
- `auth_mon_id`: residue name if modeled in the deposited structure, `'?'` if unmodeled.
- `asym_id`: chain identifier.

The existing `MMCIF2Dict` from Biopython (already imported in all scripts) exposes this table directly.

### The `AlignmentResult` dataclass

```python
from dataclasses import dataclass
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict

@dataclass
class AlignmentResult:
    protein_id: str
    chain_id: str
    pb_indices: np.ndarray   # shape (n_modeled,) int32
                             # pb_indices[struct_idx] = pair_block_idx
    n_entity: int            # total entity sequence length = pair block dimension (e.g. 280)
    n_modeled: int           # number of resolved residues (e.g. 276)
```

`pb_indices` is the key: it is a 1D integer array of length `n_modeled`. For a pair `(i, j)` where `i` and `j` are indices into the resolved structure (0-based), the corresponding location in the pair block is `(pb_indices[i], pb_indices[j])`.

### `build_alignment()`

```python
def build_alignment(cif_path: str, chain_id: str, protein_id: str) -> AlignmentResult:
    d = MMCIF2Dict(cif_path)
    seq_ids   = d['_pdbx_poly_seq_scheme.seq_id']
    auth_mons = d['_pdbx_poly_seq_scheme.auth_mon_id']
    asym_ids  = d['_pdbx_poly_seq_scheme.asym_id']

    chain_rows = [
        (int(s) - 1, a)          # 0-based pair block index, auth residue name
        for s, a, c in zip(seq_ids, auth_mons, asym_ids)
        if c == chain_id
    ]
    n_entity = max(pb_idx for pb_idx, _ in chain_rows) + 1
    pb_indices = np.array(
        [pb_idx for pb_idx, auth in chain_rows if auth != '?'],
        dtype=np.int32
    )
    return AlignmentResult(protein_id, chain_id, pb_indices, n_entity, len(pb_indices))
```

### Replacing `PAIR_OFFSET` in existing scripts

Every occurrence of:
```python
data[i + PAIR_OFFSET, j + PAIR_OFFSET, :]
```
becomes:
```python
data[alignment.pb_indices[i], alignment.pb_indices[j], :]
```

Masks (near/far) stay in structure space (`n_modeled × n_modeled`) and are unaffected. This pattern is correct for all gap types.

**Verification:** Run `build_alignment` on 7B3A and confirm `pb_indices` is `[4, 5, 6, ..., 279]` — identical to the old `+ 4` offset, so regression R² values should be bit-for-bit unchanged.

### Mandatory sanity checks

Run these before any analysis, per protein. If they fail, **abort the protein entirely** — a bad alignment produces silently wrong results, which is worse than no result.

```python
def validate_alignment(alignment: AlignmentResult, pair_block_path: str, logger):
    r = np.load(pair_block_path, mmap_mode='r').shape[0]
    assert r == alignment.n_entity, \
        f"pair block dim {r} != entity seq len {alignment.n_entity}"
    assert alignment.pb_indices.max() < r, \
        "pb_index out of bounds"
    assert np.all(np.diff(alignment.pb_indices) > 0), \
        "pb_indices not strictly increasing (gap in sequence order?)"
    # Also: count CA atoms from Biopython and confirm == alignment.n_modeled
    logger.info(
        f"Alignment OK: {alignment.n_modeled}/{alignment.n_entity} residues modeled "
        f"({alignment.n_entity - alignment.n_modeled} unmodeled)"
    )
```

---

## #2: Vectorize Inner Pair Loops

All analysis scripts use nested Python `for i: for j:` loops over residue pairs. At 276 residues × 48 layers × 500 proteins, this is the difference between hours and days on the cluster.

Replace with vectorized numpy using `np.triu_indices`:

```python
# Precompute once per protein
struct_i, struct_j = np.triu_indices(n_modeled, k=1)   # k=1 excludes diagonal
pb_i = alignment.pb_indices[struct_i]                   # pair block row indices
pb_j = alignment.pb_indices[struct_j]                   # pair block col indices

# Per layer (e.g. cosine change):
vec_prev = data_prev[pb_i, pb_j, :]   # shape (n_pairs, C)
vec_curr = data_curr[pb_i, pb_j, :]

norms_prev = np.linalg.norm(vec_prev, axis=1)
norms_curr = np.linalg.norm(vec_curr, axis=1)
cos_sim = np.einsum('nc,nc->n', vec_prev, vec_curr) / \
          np.maximum(norms_prev * norms_curr, 1e-10)
cosine_change = 1.0 - cos_sim    # shape (n_pairs,)

# Near/far masks in structure space — same indexing as before
near_mask_flat = near_mask[struct_i, struct_j]   # bool array, shape (n_pairs,)
far_mask_flat  = far_mask[struct_i, struct_j]
```

Scripts to vectorize: `analyze_layer_updates.py`, `analyze_spatial_range_magnitudes.py`, `probe_layer_regression.py`, `linear_regression_distance.py`.

---

## #3: Parameterize All Scripts

No script should have `protein = "7b3a_A"` or a hardcoded `PROTEIN_DIR`. Each script should accept command-line arguments via `argparse`, or better, all analyses should be consolidated into a single orchestrator script.

**Proposed single-protein entry point:**
```bash
python scripts/run_protein.py \
    --protein_dir proteins/1abc \
    --chain A \
    --analyses all \
    --skip_plots          # suppress matplotlib on cluster (saves time & avoids display issues)
```

**Shared `lib/` modules to extract:**
- `lib/alignment.py` — `build_alignment()`, `validate_alignment()`, `AlignmentResult`
- `lib/structure.py` — `load_ca_coords()`, `compute_dist_matrix()` (move from `extract_coordinates.py`)
- `lib/pair_blocks.py` — `load_pair_block()`, vectorized masking helpers

This prevents the common bug where a fix applied to one script isn't applied to others, because all scripts call the same shared function.

---

## #4: Per-Protein Error Isolation

Never let one failed analysis abort the whole protein's run. Wrap each analysis in a try/except:

```python
results = {}
for analysis_name, fn in analyses.items():
    try:
        results[analysis_name] = fn(alignment, protein_dir, logger)
        logger.info(f"{analysis_name}: SUCCESS")
    except Exception as e:
        results[analysis_name] = {"status": "FAILED", "error": str(e)}
        logger.error(f"{analysis_name}: FAILED — {e}", exc_info=True)

# Always write a summary, even if analyses failed
import json
with open(protein_dir / "run_summary.json", "w") as f:
    json.dump(results, f, indent=2)
```

**Exception:** alignment failure should be fatal for that protein. Skip writing any analysis outputs if alignment fails — partial outputs from a bad alignment are dangerous.

---

## #5: Logging

Replace bare `print()` with Python's `logging` module. On the cluster, stdout goes to SLURM log files. Structured logging lets you grep for errors across 500 job outputs (`grep "ERROR" batch/logs/*.err`).

```python
import logging

def setup_logger(log_path) -> logging.Logger:
    logger = logging.getLogger("protein_analysis")
    logger.setLevel(logging.INFO)
    fh = logging.FileHandler(log_path)
    fh.setFormatter(logging.Formatter('%(asctime)s %(levelname)s %(message)s'))
    logger.addHandler(fh)
    sh = logging.StreamHandler()   # also print warnings/errors to stderr
    sh.setLevel(logging.WARNING)
    logger.addHandler(sh)
    return logger
```

---

## Future Work (Items 6–8, not yet implemented)

The following will be needed once items 1–5 are in place and verified against 7B3A:

**6. SLURM array job structure** — `batch/submit_array.sh` with `--array=1-500%50` (the `%50` throttles concurrent jobs to avoid I/O saturation on shared cluster filesystems). One protein per task, driven by `batch/protein_list.txt` (one `<pdb_id> <chain>` per line).

**7. Per-protein output layout** — Each protein gets its own directory (`proteins/<pdb_id>/`) with `alignment.json`, `run_summary.json`, `run.log`, and subdirectories for CSVs and visualizations. Visualizations are skipped on the cluster (`--skip_plots`) and can be regenerated locally on demand.

**8. Post-batch audit script** — After all 500 proteins run, `batch/audit_results.py` aggregates `alignment_summary.csv` (flags unusual gap patterns) and `layer_regression_r2.csv` (flags outliers, e.g., R² at layer 47 < 0.5 could indicate an alignment error). This is the highest-value safeguard for subtle bugs that don't crash but produce wrong results.

# Multi-Protein Regression: Step-by-Step Implementation Plan

Train a single linear regression model across all proteins (pooling all residue-pair embeddings
and their CA-CA distances). Three prerequisites: a shared `lib/` module, a general residue
alignment (not a hardcoded offset), and vectorized pair loops.

---

## Final directory layout

```
protein/
├── lib/
│   ├── __init__.py
│   ├── alignment.py
│   └── structure.py
├── scripts/
│   ├── extract_coordinates.py          ← thin wrapper; imports from lib/
│   ├── single_protein/                 ← UNCHANGED (leave as-is)
│   │   └── regression/ ...
│   ├── tests/
│   │   └── test_alignment.py           ← alignment sanity checks
│   └── multi_protein/
│       ├── __init__.py
│       └── regression/
│           ├── __init__.py
│           └── probe_layer_regression_multi.py
└── proteins/
    ├── 7b3a/ ...
    └── multi_protein_results/          ← created at runtime
        ├── csv_files/
        └── visualizations/
```

---

## Step 1: `lib/__init__.py`

Empty file. Makes `lib` importable as a Python package.

All scripts that import from `lib/` must add the project root to `sys.path` at the top:
```python
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
# Adjust the number of dirname() calls so this points to the project root.
```

---

## Step 2: `lib/alignment.py`

### Imports
```python
from dataclasses import dataclass
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict
```

### `AlignmentResult` dataclass
```python
@dataclass
class AlignmentResult:
    protein_id: str          # e.g. "7b3a"
    chain_id: str            # e.g. "A"
    pb_indices: np.ndarray   # shape (n_modeled,), dtype=int32
                             # pb_indices[i] = pair-block row/col index for structure residue i
    n_entity: int            # total entity sequence length = pair-block dimension (e.g. 280)
    n_modeled: int           # number of resolved residues (e.g. 276)
```

### `build_alignment(cif_path, chain_id, protein_id) -> AlignmentResult`

1. `d = MMCIF2Dict(cif_path)` — reads the whole mmCIF into a dict
2. Extract three parallel lists from the `_pdbx_poly_seq_scheme` table:
   - `seq_ids   = d['_pdbx_poly_seq_scheme.seq_id']`      — strings, 1-based entity positions
   - `auth_mons = d['_pdbx_poly_seq_scheme.auth_mon_id']` — residue name, or `'?'` if unmodeled
   - `asym_ids  = d['_pdbx_poly_seq_scheme.asym_id']`     — internal chain label
3. Filter to rows where `asym_id == chain_id`:
   ```python
   chain_rows = [
       (int(s) - 1, auth)      # 0-based pair-block index, residue name
       for s, auth, asym in zip(seq_ids, auth_mons, asym_ids)
       if asym == chain_id
   ]
   ```
4. `n_entity = max(pb_idx for pb_idx, _ in chain_rows) + 1`
5. `pb_indices = np.array([pb_idx for pb_idx, auth in chain_rows if auth != '?'], dtype=np.int32)`
6. Return `AlignmentResult(protein_id, chain_id, pb_indices, n_entity, len(pb_indices))`

**Expected result for 7B3A chain A:**
`pb_indices = [4, 5, ..., 279]` (length 276), `n_entity=280`, `n_modeled=276`.
This is numerically identical to the old `PAIR_OFFSET = 4` pattern.

### `validate_alignment(alignment, first_pb_path, n_ca_atoms) -> None`

Raises `AssertionError` with a descriptive message if any check fails. The caller must catch
this and skip the protein entirely — silently wrong alignment is worse than no result.

```python
r = np.load(first_pb_path, mmap_mode='r').shape[0]
assert r == alignment.n_entity,   \
    f"{alignment.protein_id}: pair block dim {r} != entity len {alignment.n_entity}"
assert alignment.pb_indices.max() < r,  \
    f"{alignment.protein_id}: pb_index out of bounds"
assert np.all(np.diff(alignment.pb_indices) > 0),  \
    f"{alignment.protein_id}: pb_indices not strictly increasing"
assert n_ca_atoms == alignment.n_modeled, \
    f"{alignment.protein_id}: CIF CA count {n_ca_atoms} != alignment n_modeled {alignment.n_modeled}"
```

---

## Step 3: `lib/structure.py`

### Functions to migrate from `scripts/extract_coordinates.py` (verbatim, with docstrings)
- `extract_residue_coordinates(cif_file, chain_id=None)` → returns `{'residues': [...], 'coordinates': np.ndarray}`
- `detect_chain(protein_dir: Path) -> str | None`

### New: `compute_distance_matrix(coords: np.ndarray) -> np.ndarray`

Vectorized replacement for the Python upper-triangle loop in `extract_coordinates.py`:
```python
def compute_distance_matrix(coords: np.ndarray) -> np.ndarray:
    """Vectorized pairwise Euclidean distance. Output: (n, n), symmetric, 0 diagonal."""
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]  # (n, n, 3)
    return np.linalg.norm(diff, axis=2)                         # (n, n)
```
Result is numerically identical to the old loop-based version.

### New: `load_distance_matrix(csv_path) -> np.ndarray`

Convenience loader for pre-computed `residue_distances.csv` files:
```python
def load_distance_matrix(csv_path) -> np.ndarray:
    return np.genfromtxt(csv_path, delimiter=',', skip_header=1)
```

### Update `scripts/extract_coordinates.py`

Remove the inline definitions of `extract_residue_coordinates` and `compute_distance_matrix`.
Replace with imports from `lib/structure.py`. Keep `process_protein()` and `main()` unchanged.
External behaviour must be identical.

---

## Step 4: `scripts/tests/test_alignment.py`

Standalone script (no pytest). Run with `python scripts/tests/test_alignment.py` from the project root.

Path setup: script is 2 levels deep (`scripts/tests/`), so two `os.path.dirname` calls reach
`scripts/`, then one more reaches the project root — add that to `sys.path`.

### Test 1 — `build_alignment` produces correct `pb_indices` for 7B3A
```python
alignment = build_alignment("proteins/7b3a/7b3a.cif", "A", "7b3a")
assert alignment.n_entity == 280
assert alignment.n_modeled == 276
assert list(alignment.pb_indices) == list(range(4, 280)), \
    f"Expected [4..279], got {alignment.pb_indices[:5]}..."
print("PASS: 7B3A pb_indices == [4..279]")
```

### Test 2 — `validate_alignment` passes on 7B3A
```python
ca_result = extract_residue_coordinates("proteins/7b3a/7b3a.cif", "A")
n_ca = len(ca_result['residues'])
first_pb = "proteins/7b3a/pair_blocks/7b3a_A_pair_block_0.npy"
validate_alignment(alignment, first_pb, n_ca)   # must not raise
print(f"PASS: validate_alignment OK ({n_ca} CA atoms, n_entity={alignment.n_entity})")
```

### Test 3 — pair index spot-check
```python
# Structure residue 0 → pb index 4; structure residue 10 → pb index 14
assert alignment.pb_indices[0] == 4
assert alignment.pb_indices[10] == 14
# Confirm fancy indexing into the pair block works without error
pb = np.load(first_pb)
vec = pb[alignment.pb_indices[0], alignment.pb_indices[10], :]
assert vec.shape == (128,)
print("PASS: pair block fancy indexing with pb_indices works")
```

### Test 4 — distance matrix shape and symmetry
```python
coords = ca_result['coordinates']
dm = compute_distance_matrix(coords)
assert dm.shape == (276, 276)
assert np.allclose(dm, dm.T)
assert np.all(np.diag(dm) == 0)
print("PASS: distance matrix shape and symmetry")
```

### Test 5 — `load_distance_matrix` matches `compute_distance_matrix`
```python
dm_csv = load_distance_matrix("proteins/7b3a/csv_files/residue_distances.csv")
assert np.allclose(dm, dm_csv, atol=1e-3)   # CSV saved at 3 d.p.
print("PASS: load_distance_matrix matches computed result")
```

---

## Step 5: `scripts/multi_protein/regression/probe_layer_regression_multi.py`

### Imports
```python
import sys, os
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import r2_score
```

After `sys.path.insert` (4 `dirname` calls from `__file__` to reach the project root):
```python
from lib.alignment import build_alignment, validate_alignment
from lib.structure import extract_residue_coordinates, compute_distance_matrix, detect_chain
```

### Pre-computation pass

Iterate over all subdirs of `proteins/` except `multi_protein_results/`. For each protein:

```python
proteins_root = Path(ROOT_DIR) / "proteins"
protein_dirs = sorted([p for p in proteins_root.iterdir()
                       if p.is_dir() and p.name != 'multi_protein_results'])
protein_data = []

for protein_dir in protein_dirs:
    protein_id = protein_dir.name
    cif_files = list(protein_dir.glob("*.cif"))
    if not cif_files:
        print(f"[SKIP] {protein_id}: no CIF file"); continue
    cif_path = cif_files[0]

    chain_id = detect_chain(protein_dir)
    if chain_id is None:
        print(f"[SKIP] {protein_id}: could not detect chain"); continue

    try:
        alignment = build_alignment(str(cif_path), chain_id, protein_id)
        first_pb = protein_dir / "pair_blocks" / f"{protein_id}_{chain_id}_pair_block_0.npy"
        if not first_pb.exists():
            print(f"[SKIP] {protein_id}: pair blocks not found"); continue

        ca_result = extract_residue_coordinates(str(cif_path), chain_id)
        n_ca = len(ca_result['residues'])
        validate_alignment(alignment, str(first_pb), n_ca)

        coords = ca_result['coordinates']
        dist_matrix = compute_distance_matrix(coords)

        struct_i, struct_j = np.triu_indices(alignment.n_modeled, k=1)
        pb_i = alignment.pb_indices[struct_i]
        pb_j = alignment.pb_indices[struct_j]
        y_protein = dist_matrix[struct_i, struct_j]

        protein_data.append({
            'protein_id': protein_id,
            'chain_id':   chain_id,
            'pb_dir':     protein_dir / "pair_blocks",
            'pb_i':       pb_i,
            'pb_j':       pb_j,
            'y':          y_protein,
        })
        print(f"[OK] {protein_id}: {alignment.n_modeled} residues, {len(y_protein)} pairs")

    except AssertionError as e:
        print(f"[SKIP] {protein_id}: alignment failed — {e}"); continue
    except Exception as e:
        print(f"[SKIP] {protein_id}: unexpected error — {e}"); continue
```

### Pair block filename pattern

Filenames follow `{protein_dir_name}_pair_block_{layer}.npy` where `protein_dir_name` is the
full directory name including chain suffix (e.g. `6tf4_A`). Construct as:
```python
pb_path = d['pb_dir'] / f"{d['protein_id']}_pair_block_{layer}.npy"
```
where `d['protein_id'] = protein_dir.name` (e.g. `"6tf4_A"`).

Also store per-protein structural indices in `protein_data` during the pre-computation pass:
```python
struct_i, struct_j = np.triu_indices(alignment.n_modeled, k=1)
protein_data.append({
    ...
    'struct_i': struct_i,   # structure residue indices (for saving predictions)
    'struct_j': struct_j,
})
```

After `protein_data` is built, compute cumulative pair offsets so that rows in the pooled
arrays can be mapped back to individual proteins:
```python
protein_offsets = []   # list of (protein_id, start, end) in y_all index space
offset = 0
for d in protein_data:
    n = len(d['y'])
    protein_offsets.append((d['protein_id'], offset, offset + n))
    offset += n
```

### Pooled y and fixed train/test split

```python
if not protein_data:
    raise RuntimeError("No proteins loaded successfully.")

y_all = np.concatenate([d['y'] for d in protein_data])
n_total_pairs = len(y_all)
print(f"\nTotal pairs across {len(protein_data)} proteins: {n_total_pairs}")

all_idx = np.arange(n_total_pairs)
idx_train, idx_test = train_test_split(all_idx, test_size=0.2, random_state=42)
y_train = y_all[idx_train]
y_test  = y_all[idx_test]
```

### Layer loop with feature standardization

Fit `StandardScaler` on training rows only; apply to train and test. This makes coefficients
directly comparable across features (same pattern as `probe_layer_regression_standardized.py`).

Track coefficients for every layer. On the last layer, generate full predictions across all pairs
(train + test) for saving and plotting.

```python
results = []
all_coefs = []          # one dict per layer
last_layer_preds = None # filled on final layer
n_layers = 48
C = 128  # channels per pair direction

for layer in range(n_layers):
    X_chunks = []
    for d in protein_data:
        pb_path = d['pb_dir'] / f"{d['protein_id']}_pair_block_{layer}.npy"
        pb = np.load(pb_path)
        X_chunk = np.concatenate([
            pb[d['pb_i'], d['pb_j'], :],    # upper triangle: pair_block[i, j, :]
            pb[d['pb_j'], d['pb_i'], :],    # lower triangle: pair_block[j, i, :]
        ], axis=1)                          # shape: (n_pairs_this_protein, 256)
        X_chunks.append(X_chunk)
        del pb

    X_all = np.vstack(X_chunks)             # shape: (n_total_pairs, 256)

    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_all[idx_train])
    X_test  = scaler.transform(X_all[idx_test])

    model = LinearRegression()
    model.fit(X_train, y_train)
    r2 = r2_score(y_test, model.predict(X_test))
    results.append(r2)
    print(f"Layer {layer:2d}  R² = {r2:.4f}")

    # Save coefficients for this layer (intercept + 256 feature coefficients)
    coef_row = {'layer': layer, 'intercept': model.intercept_}
    coef_row.update({f'coef_{i}': c for i, c in enumerate(model.coef_)})
    all_coefs.append(coef_row)

    # On the last layer, predict on ALL pairs (train+test) for output CSVs and plots
    if layer == n_layers - 1:
        X_all_std = scaler.transform(X_all)
        y_pred_all = model.predict(X_all_std)
        last_layer_preds = y_pred_all   # shape: (n_total_pairs,)

    del X_all, X_train, X_test
```

### Outputs

**CSV 1:** `proteins/multi_protein_results/csv_files/layer_regression_r2_multi.csv`
Columns: `layer,r2,n_proteins,n_total_pairs` (include `n_proteins` and `n_total_pairs` on every row).

**CSV 2:** `proteins/multi_protein_results/csv_files/layer_regression_coefficients_multi.csv`
Columns: `layer,intercept,coef_0,coef_1,...,coef_255`
One row per layer (48 rows total). Features 0–127 are the upper-triangle channels
(`pair_block[i,j,:]`); features 128–255 are the lower-triangle channels (`pair_block[j,i,:]`).

**CSV 3:** `proteins/multi_protein_results/csv_files/predicted_distances_last_layer_multi.csv`
Columns: `protein_id,struct_i,struct_j,actual_distance_A,predicted_distance_A,split`
One row per residue pair across all included proteins. `split` is `"train"` or `"test"`.
Build a global split-label array then slice per protein using `protein_offsets`:
```python
split_labels = np.empty(n_total_pairs, dtype=object)
split_labels[idx_train] = 'train'
split_labels[idx_test]  = 'test'
```

**Plot 1:** `proteins/multi_protein_results/visualizations/layer_regression_r2_multi.png`
Line plot, x=layer, y=R², title showing number of proteins and total pairs.

**Plot 2:** `proteins/multi_protein_results/visualizations/pred_vs_actual_all_last_layer_multi.png`
Scatter plot: x=actual CA-CA distance (Å), y=predicted CA-CA distance (Å), for ALL proteins
combined, using ALL pairs (train + test) from the last layer. Color-code each protein with a
distinct colour. Add a y=x diagonal reference line. Title: `"Layer 47: Predicted vs Actual
Distance (all proteins)"`. Legend shows protein IDs and their pair counts.

**Plot 3:** `proteins/multi_protein_results/visualizations/pred_vs_actual_per_protein_last_layer_multi.png`
Grid of subplots (one panel per included protein), arranged in rows of 2.
Each panel: scatter of predicted vs actual distances for that protein (all pairs), y=x diagonal,
protein ID + R² in the panel title. Use `plt.tight_layout()`. If an odd number of proteins,
leave the last cell blank.

**Console:** Print a summary at the end listing which proteins were included and which were skipped.

---

## Verification checklist (run in this order)

1. `python scripts/tests/test_alignment.py` — all 5 tests must print PASS
2. `python scripts/extract_coordinates.py` — CSV outputs must be identical to pre-refactor (no behaviour change)
3. `python scripts/multi_protein/regression/probe_layer_regression_multi.py` — with only 7B3A
   present, layer-47 R² should be close to (but not necessarily identical to) the single-protein
   standardized result, since the train/test split is over the same pairs but standardization
   is now fit on the pooled training set

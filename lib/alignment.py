"""
lib/alignment.py
================
General residue alignment between an mmCIF structure file and AlphaFold pair blocks.

The entity sequence stored in the pair blocks may include residues not present in the
deposited structure (e.g. expression tags, disordered loops). This module reads the
_pdbx_poly_seq_scheme table from the CIF to build an explicit mapping from structure
residue index → pair-block index, replacing any hardcoded offsets.
"""

from dataclasses import dataclass
import numpy as np
from Bio.PDB.MMCIF2Dict import MMCIF2Dict


@dataclass
class AlignmentResult:
    """Mapping between a resolved CIF structure and the AlphaFold pair-block indices.

    Attributes
    ----------
    protein_id : str
        PDB identifier, e.g. "7b3a".
    chain_id : str
        Internal asym_id chain label used in the CIF, e.g. "A".
    pb_indices : np.ndarray, shape (n_modeled,), dtype=int32
        pb_indices[i] = pair-block row/column index for the i-th resolved
        structure residue. Length equals n_modeled.
    n_entity : int
        Total entity sequence length = pair-block matrix dimension (e.g. 280).
    n_modeled : int
        Number of resolved residues in the deposited structure (e.g. 276).
    """
    protein_id: str
    chain_id: str
    pb_indices: np.ndarray
    n_entity: int
    n_modeled: int


def build_alignment(cif_path: str, chain_id: str, protein_id: str) -> AlignmentResult:
    """Build an explicit residue alignment from the _pdbx_poly_seq_scheme CIF table.

    Parameters
    ----------
    cif_path : str
        Path to the .cif structure file.
    chain_id : str
        Internal asym_id to filter on (e.g. "A").
    protein_id : str
        Human-readable identifier used in error messages (e.g. "7b3a").

    Returns
    -------
    AlignmentResult
        Contains pb_indices mapping each resolved residue to its pair-block index.

    Notes
    -----
    _pdbx_poly_seq_scheme.seq_id is 1-based. Pair-block indices are 0-based, so
    we subtract 1 from every seq_id. Rows where auth_mon_id == '?' correspond to
    unmodeled (unresolved) residues and are excluded from pb_indices.

    Expected result for 7B3A chain A:
        pb_indices = [4, 5, ..., 279]  (length 276)
        n_entity = 280, n_modeled = 276
    """
    d = MMCIF2Dict(cif_path)

    seq_ids = d['_pdbx_poly_seq_scheme.seq_id']       # 1-based entity positions
    auth_mons = d['_pdbx_poly_seq_scheme.auth_mon_id'] # residue name or '?' if unmodeled
    asym_ids = d['_pdbx_poly_seq_scheme.asym_id']      # internal chain label

    # Filter to rows belonging to the requested chain
    chain_rows = [
        (int(s) - 1, auth)   # 0-based pair-block index, residue name
        for s, auth, asym in zip(seq_ids, auth_mons, asym_ids)
        if asym == chain_id
    ]

    if not chain_rows:
        raise ValueError(f"{protein_id}: no rows found for chain '{chain_id}' in _pdbx_poly_seq_scheme")

    n_entity = max(pb_idx for pb_idx, _ in chain_rows) + 1

    pb_indices = np.array(
        [pb_idx for pb_idx, auth in chain_rows if auth != '?'],
        dtype=np.int32
    )

    return AlignmentResult(
        protein_id=protein_id,
        chain_id=chain_id,
        pb_indices=pb_indices,
        n_entity=n_entity,
        n_modeled=len(pb_indices),
    )


def validate_alignment(alignment: AlignmentResult, first_pb_path: str, n_ca_atoms: int) -> None:
    """Validate that an AlignmentResult is consistent with the pair blocks and CIF CA count.

    Raises AssertionError with a descriptive message if any check fails. The caller
    should catch this and skip the protein entirely — silently wrong alignment is
    worse than no result.

    Parameters
    ----------
    alignment : AlignmentResult
        Result of build_alignment().
    first_pb_path : str
        Path to any single pair-block .npy file (e.g. layer 0). Used to read the
        actual pair-block matrix dimension.
    n_ca_atoms : int
        Number of CA atoms extracted from the CIF (should equal alignment.n_modeled).
    """
    r = np.load(first_pb_path, mmap_mode='r').shape[0]

    assert r == alignment.n_entity, (
        f"{alignment.protein_id}: pair block dim {r} != entity len {alignment.n_entity}"
    )
    assert alignment.pb_indices.max() < r, (
        f"{alignment.protein_id}: pb_index out of bounds"
    )
    assert np.all(np.diff(alignment.pb_indices) > 0), (
        f"{alignment.protein_id}: pb_indices not strictly increasing"
    )
    assert n_ca_atoms == alignment.n_modeled, (
        f"{alignment.protein_id}: CIF CA count {n_ca_atoms} != alignment n_modeled {alignment.n_modeled}"
    )

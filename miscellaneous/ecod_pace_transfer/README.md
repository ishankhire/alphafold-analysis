# ECOD transfer from PACE

Globus CLI batch files for copying the cohort's ECOD data from the PI's copy on
PACE Phoenix back to Delta.

- Source: `/storage/cedar/cedar0/cedarp-gkrishnan46-0/ecod/`
- Destination: `/work/hdd/bfna/ikhire/thayes/ecod/`

Each batch line is `--recursive <source path> <destination path>`, relative to
those two roots.

| File | Contents | Status |
|---|---|---|
| `ecod_ids.txt` | The 400 TM > 0.7 ECOD IDs from `datasets.md` | — |
| `pair_blocks.batch` | `experiments/representation_saves/{ID}_save_layers_0_47/` for each ID; about 450 GB (estimated) | Not yet run |
| `pdbs_layer47.batch` | Layer-47 folders of `experiments/openfold_reps_outputs_sm/` | Not used. On 2026-11-08 the whole `openfold_reps_outputs_sm/` folder (all 48 layers) was copied instead. |

Globus flattened the Nov 8 transfer into `thayes/ecod/`, so it had to be moved
into `experiments/openfold_reps_outputs_sm/` afterwards. After a transfer,
check that the files are at the destination path. Current status is in the
[Delta data inventory](../../docs/cluster/delta_data_inventory.md).

# ECOD transfer from PACE

Globus CLI batch files that copied the cohort's ECOD data from the PI's copy on
PACE Phoenix back to Delta. All the transfers are done; the resulting Delta
paths are in [`datasets.md`](../../datasets.md#ecod).

- Source: `/storage/cedar/cedar0/cedarp-gkrishnan46-0/ecod/`
- Destination: `/work/hdd/bfna/ikhire/thayes/ecod/`

Each batch line is `--recursive <source path> <destination path>`, relative to
those two roots. To copy more, submit a batch file with the Globus CLI (installed in
`~/.globus-cli-venv` on the laptop):

```
globus transfer \
  6df312ab-ad7c-4bbc-9369-450c82f0cb92:/storage/cedar/cedar0/cedarp-gkrishnan46-0/ecod/ \
  2d66a243-4a3f-4578-9d7f-1935fb5fba8f:/work/hdd/bfna/ikhire/thayes/ecod/ \
  --batch <file>.batch --sync-level checksum
```

The first ID is the PACE Phoenix collection and the second is NCSA Delta.

| File | Contents | Status |
|---|---|---|
| `ecod_ids.txt` | The 400 TM > 0.7 ECOD IDs from `datasets.md` | — |
| `pair_blocks.batch` | `experiments/representation_saves/{ID}_save_layers_0_47/` for each ID (48 pair blocks and 48 MSA blocks each) | ✅ Succeeded 2026-11-13: 400 folders, 38,400 files, 476.4 GB, no faults. Globus task `a673c23a-dfc8-11f1-96c2-0affc602d875` |
| `alignments.batch` | `alignments/{ID}/` for each ID (MSAs and template hits, in a nested `{ID}/` folder) | ✅ Succeeded 2026-11-13: 400 folders, 1,600 files, 6.3 GB, no faults. Globus task `a72ad97d-dfc8-11f1-849c-0affc602d875` |
| `pdbs_layer47.batch` | Layer-47 folders of `experiments/openfold_reps_outputs_sm/` | Not used. On 2026-11-08 the whole `openfold_reps_outputs_sm/` folder (all 48 layers) was copied instead. |

Globus flattened the Nov 8 transfer into `thayes/ecod/`, so it had to be moved
into `experiments/openfold_reps_outputs_sm/` afterwards.
The Nov 13 batch transfers kept their paths. After a transfer, check that the
files are at the destination path. What is still only on PACE is in the
[Delta data inventory](../../docs/cluster/delta_data_inventory.md).

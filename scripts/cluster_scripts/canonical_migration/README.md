# Canonical collection migration

Deploy the runtime allowlist using `canonical_deployment.py`, including this
entrypoint, launcher, shared helpers, frozen master split, and the retained
Linux TMscore binary. `deployment_manifest.json` records their hashes and Git
revision. Local authority inputs are the twelve refreshed geometry/manifest
files and the read-only geometry validation report; keep them under
`migration_inputs/{authority.json,refreshed/<family>/<filename>}` in the fresh
flat deployment `/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`.

Submit from that deployment with `sbatch verify_canonical_results.sbatch`.
Logs are `canonical_migration_<jobid>.{out,err}` in the submission directory.
The single CPU, 3 GiB, one-hour job copies the explicit HOME allowlist into
`/work/hdd/bfna/ikhire/.regression_5fold_v1_migration_v1/`, validates copied
files and protocol interfaces, publishes into the existing
`/work/hdd/bfna/ikhire/regression_5fold_v1/`, and audits group access. It refuses
active writers and destination conflicts. A failed or interrupted migration
requires inspection before retrying; the tool preserves all partial state.

Job `23142906` timed out after 1h00m01s, after all 40,341 pre-publication
checks passed and every new component was renamed into the final root. The
collection remains `publishing`; migration status is `verifying_access`.
Do not rerun the original launcher against this state. After inspecting the
terminal job state, manifests, alias, and complete cutover, deploy the recovery
implementation and submit `sbatch resume_canonical_results.sbatch` from the
same flat deployment. Logs are `canonical_resume_<jobid>.{out,err}` there.
The recovery uses one CPU, 6 GiB, and four hours: the original Python step
peaked at 3,129,088 KiB, and filesystem permission work exceeded the original
one-hour limit. Directory/file ACL operations are now batched across directories.

`--resume-published` accepts only an incomplete, fully cut-over migration with
matching roots, source mapping, authority, and passing prior verification.
It rehashes all inventory files and HOME sources and repeats protocol,
geometry, resident-compact-output, and deployment checks before finishing
permissions and access audits. It does not copy or rename result components.
Prior manifests and verification are preserved in a new
`_migration_v1/recovery_<UTC timestamp>_<jobid>/` directory. That directory
also contains `verification.csv`, `access_verification.csv`, and
`final_access_verification.csv`, each with the same per-check columns as the
main audit. Successful checks append to `_migration_v1/verification.csv`;
the migration manifest retains the original job/Git identity and records
recovery job/Git identities separately. Completed collections and conflicting
or partial cutovers are refused. Preserve old deployment files and inventory
in a new backup directory before deploying recovery. Nothing is deleted.

The audit contract is `_migration_v1/inventory.csv` (one row per copied file:
source, destination, authority, bytes, SHA-256), `verification.csv` (one row per
check: expected, observed, passed), `migration_manifest.json` (source mapping,
original/refreshed hashes, extent, provenance), and `collection_manifest.json`
(component locations and completion). Auxiliary authority, resident snapshot,
and access reports remain beside them. No fitting, production aggregation,
scoring finalization, or deletion occurs. Resident physical validation remains
`consolidation_only`, with representative reads across all four probes and
layers 0, 23, 47. Historical hashes and absolute provenance paths are retained.

Only the twelve authoritative refreshed files and runtime code are uploaded.
The coefficient, gain/split importance, and TM-score-comparison derived bundles
remain repository-only. New production runs must use explicit fresh roots;
mutating entrypoints refuse a collection carrying `collection_manifest.json`
before creating locks or logs. See the normative methodology/output contract
and `docs/plans/delta_canonical_results_migration_plan.md` for scope.

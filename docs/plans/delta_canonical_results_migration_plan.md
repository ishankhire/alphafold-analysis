# Delta canonical-results migration and refactor

Status: implementation committed and pushed as `d18b486`; migration job
`23142906` is running. Monitoring stopped at the user's request. Publication,
access-audit completion, resource accounting, and final runbook/status cutover
must be confirmed after the job finishes; do not describe the collection as
complete yet. The approved scope below remains authoritative.

## Goal and operating conditions

Consolidate the latest canonical protocol-2.0 fivefold regression collection
under `/work/hdd/bfna/ikhire/regression_5fold_v1/`, and deploy active cluster code
flat under `/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`.

Follow the current repository `AGENTS.md`,
[cluster guide](../cluster/README.md), and normative
[methodology](../methodology/regression_methodology.md) and
[output contract](../methodology/regression_outputs.md). Consult the live
[Delta documentation](https://docs.ncsa.illinois.edu/systems/delta/en/latest/)
before implementation. Do not change experimental definitions or duplicate
them in workflow-specific documentation.

- **The first implementation action must be `ssh delta hostname`. If SSH fails,
  stop implementation and report the error before editing code, deploying,
  copying data, changing permissions, or submitting jobs.** If connectivity
  fails later, stop further implementation and report the partial state.
- Begin remote execution commands with the approved `ssh delta` prefix. Avoid
  placing options before `delta`, which previously missed the approved command
  rule. This uses the existing approval rule; it does not disable sandbox
  restrictions. File-transfer tools may require their own approvals.
- **Delete nothing from any directory.** Preserve all existing HOME and local
  files, old deployments, Slurm scripts, output logs, checkpoints, temporary
  directories, and staging remnants. Do not use deletion flags or automatic
  staging cleanup. Any later deletion requires separate user permission.
- Use copy, verification, and cutover. Do not move original data or replace
  original HOME results. Only newly copied staging directories may be renamed
  into their final destination during publication.
- Give the verified bfna allocation group read/traverse access; keep writes
  owner-controlled. Git remains the source of truth for code.
- Preserve completed outputs and historical checkpoint identities. Future
  reruns require new explicit output roots, rather than compatibility changes
  to resume completed outputs in place.
- Omit local-only derived-analysis bundles from the migration. Keep their
  tracked tables and plots in the repository/GitHub. The necessary exceptions
  are the twelve authoritative refreshed geometry/manifest files described
  below, plus deployment of active code.

## Inspection findings and authority

Remote observations below were obtained on October 20, 2026. They must be
rechecked after SSH succeeds; they are not a claim about subsequent changes.

- `ssh delta` succeeded and reached `dt-login03.delta.ncsa.illinois.edu`.
- The bfna allocation group is `delta_bfna`, GID `20197`; `ikhire` belongs to it.
- `$HOME/csv_files/regression_5fold_v1/` contains all four completed families,
  retained models, frozen metadata, and physical layer shards.
- The complete triangle and publication-subgroup bundles live separately under
  `$HOME/csv_files/analysis/`.
- HDD already contains finalized distance exports and optimized TM-score/PDB
  outputs. The shared code destination did not yet exist.
- Existing HDD permissions mix group-writable directories with owner-only
  manifests. ACL masks prevent effective group read access to some manifests.
- All four HOME `geometry_metrics.csv` hashes match the original scoring-source
  hashes. Those tables still contain the old TM-scores.
- The local canonical geometry tables contain the optimized TM-scores. Calling
  the pure validation/preparation function in
  `scripts/multi_protein/analyze_tm_score_optimization.py` passed all 240 checks,
  verified the 105,984-row ledger checksum, and proposed no byte changes.
- The optimized root manifest agrees locally and remotely, with SHA-256
  `433b94133f4ce392c236a00563511def03764cffc7870c7ca41621e9891b83eb`.
- The optimized manifest records `validation.mode=consolidation_only`.
  Individual resident prediction/PDB artifacts were not exhaustively reread.

Later implementation preflights failed locally, before reaching Delta:

```text
No user exists for uid 501
```

Implementation stopped as agreed. No migration code, data, deployment,
permissions, or jobs were changed.

The four refreshed local `geometry_metrics.csv` files are explicitly Git-ignored;
GitHub does not contain those revised per-protein tables. Their summaries and
refresh manifests are tracked. Upload all twelve matching files together.

## Migration inventory

Unless explicitly absolute, destinations below are relative to
`/work/hdd/bfna/ikhire/regression_5fold_v1/`.

| Destination | Authoritative source and treatment |
|---|---|
| `metadata/` | Copy the three frozen manifests from HOME after matching their hashes against local metadata. |
| `shared_ols_base256/` | Copy the complete HOME family, including artifacts, shards, checkpoints, subgroup tables, logs, and provenance. |
| `per_protein_ols_base256/` | Same treatment. |
| `sequence_gated_ols_base256/` | Same treatment. |
| `shared_xgboost_base256_fixed_s4/` | Same treatment, including retained boosters and fixed configuration. |
| Each family's `geometry_metrics.csv`, `geometry_summary.csv`, and `run_manifest.json` | Overlay the verified, refreshed local versions after preserving the original HOME versions. The shared canonical tables must contain the **new optimized TM-scores**. |
| `triangle_violation_subgroups_5fold_v1/` | Copy the complete bundle from `$HOME/csv_files/analysis/triangle_violation_subgroups_5fold_v1/`. |
| `analysis/triangle_violation_subgroups_5fold_v1` | Use the relative symlink `../triangle_violation_subgroups_5fold_v1`, exposing the same complete bundle through both paths. This is the recorded default from the unanswered optional layout question. |
| `analysis/publication_subgroup_mae_5fold_v1/` | Copy the complete HOME analysis directory, including final tables, manifest, and layer shards. |
| `predicted_distances_5fold_v1/`, `tm_score_optimized_v1/` | Retain their existing HDD contents and historical manifests. Change access permissions only. |

The frozen metadata filenames are `regression_outer_folds_v1.csv`,
`per_protein_pair_folds_v1.csv`, and `fold_digests.json`.

The twelve local authoritative replacements are under
`delta_files/regression_5fold_v1/<family>/`. Do not run the comparison script's
`main` during read-only inspection: it writes derived reports even without
`--apply-geometry`. Its `prepare_geometry_refresh` function validates without
applying updates. Preserve all non-TM geometry values and original fit provenance.

Use an explicit allowlist. Do not copy the complete legacy root indiscriminately.
Exclude the separate `shared_ols_mds_pdb_exports` experiment, historical
implementations, unrelated diagnostics, raw protein inputs, and the separate
publication-figure tree. Leave existing untracked local MDS exports untouched.

Do **not** upload these local-only derived bundles:

- `analysis/linear_coefficients_analysis/`
- `analysis/xgboost_gain_dimensions/`
- `analysis/xgboost_split_dimensions/`
- `analysis/tm_score_optimization_comparison_v1/`

Their existing tracked files remain in the repository. Retained model
coefficients and core feature-importance artifacts already on Delta remain in
scope; omitting local derived analyses does not omit those model artifacts.

### Existing interfaces to preserve and verify

Preserve the existing protocol-2.0 schemas, frozen 552-protein cohort, layers
0–47, and OOF evaluation scopes. These checks refer to the normative output
contract rather than introducing a new protocol.

| File/interface | Row granularity, coverage, and principal fields |
|---|---|
| `per_protein.csv` | 26,496 protein/layer rows per family; identity, fold, evaluation scope, coverage/status, `r2`, `mae`, `rmse`, and bias. |
| `geometry_metrics.csv` | 26,496 protein/layer rows per family; identity, scope, coverage/status, optimized `tm_score`, RMSD, spectral/stress, negative-prediction, and triangle metrics. |
| `geometry_summary.csv` | Family/layer/geometry-metric summaries; support counts and `mean`, `std`, `median`, `q25`, `q75`. |
| `pooled_metrics.csv`, `fold_metrics.csv` | Respectively 48 family/layer rows and 240 family/layer/fold rows per family; pooled metrics, support, and baseline fields. |
| `model_artifacts/replay_index.csv` | Relative artifact paths, SHA-256, configuration/scaler digests, status, and fit responsibility. Expected rows: shared OLS 240; per-protein OLS 132,480; sequence-gated OLS 1,440; XGBoost 240. |
| `per_protein_subgroup_mae.csv` | 1,165,824 protein/layer/model/category rows; pair counts, summed absolute error, MAE, scope, and support status. |
| Triangle tables | Preserve existing protein/category/metric granularity and distinct protein-equal and triangle-pooled summaries. Verify manifest-declared coverage and counts. |
| Distance, score, and PDB indexes | 105,984 model/protein/layer records each; retain relative artifact paths, checksums, scores where applicable, and statuses. |

## Active-code and access changes

Edit and test source files locally, commit and push, then copy the relevant
runtime files into the **fresh shared** flat deployment. Preserve the existing
HOME deployment and all pre-existing remote Slurm files and logs.

- Centralize storage resolution for active fivefold, triangle,
  publication-subgroup, distance-export, and TM-score workflows. Preserve
  `CSV_ROOT` as the parent of `regression_5fold_v1`; default deployed workflows
  to `/work/hdd/bfna/ikhire`, while repository runs retain local `delta_files`
  defaults. Leave older implementations and unrelated diagnostics alone.
- Route triangle production to the complete producer directory and publication
  subgroup output beneath `regression_5fold_v1/analysis/`. Remove active
  launchers' unconditional HOME overrides.
- Give active local TM-score, coefficient, and importance analyses explicit
  input/output-root overrides. Preserve their current local defaults and
  support provenance paths outside the checkout. This does not require
  uploading or regenerating their local-only derived bundles.
- Protect the published collection from mutating entrypoints before they
  create locks or output files. Reruns require a new explicit root. Keep
  existing identity-mismatch checks intact and ensure launcher logging does
  not modify protected result trees before that check.
- Keep historical checkpoint digests, code hashes, and recorded absolute paths
  unchanged. Readers resolve retained artifacts from the supplied root and
  relative indexes; they must not require current code hashes to match
  historical producer hashes. Do not rerun preparation, aggregation, or scoring
  finalization to make historical manifests match the new deployment.
- Deploy non-test runtime files from `cross_validation/`,
  `triangle_violation_subgroups/`, `publication_figures/`,
  `predicted_distances/`, and `tm_scores/`, plus shared helpers/configuration,
  uniquely named workflow READMEs, and
  `regression_protocol_data/regression_master_split_v1.csv`. Flatten source
  organization as required by the deployment convention. Copy and checksum the
  existing Linux `TMscore` binary; do not deploy a macOS binary.
- Replace the deployment verifier's stale embedded code hashes with
  `deployment_manifest.json`, recording the Git commit, source-to-deployed
  filename mapping, hashes, and binary provenance. Use its Git revision when
  deployed code lacks `.git`.
- Use local `./.venv` for tests and parity checks; retain `~/protein_venv` as
  the Delta runtime convention. Verify SBATCH references and required mail
  settings. Include the `work` filesystem dependency for HDD access; do not
  change regression memory or walltime unrelated to this migration.

Within the results and fresh deployment trees, set group ownership to
`delta_bfna`. Use directory mode `2750`, ordinary file mode `0640`, and
executable mode `0750`; set default ACLs for group read/traverse without write
access. Audit effective ACL masks and ancestor traversal. Do not recursively
change permissions on unrelated trees or on the old HOME deployment.

Normalize permissions explicitly after copying and after future successful
finalization: `umask` alone does not make files created through `mkstemp`
group-readable. Preserve owner write access. Raw input access and the existing
input paths are outside the migration.

## Execution, verification, and cutover

1. **Run the SSH gate, then recheck and record.** Confirm unchanged source
   hashes, destination collisions, quota, ownership, and active writers. Wait
   for relevant writers to finish; do not cancel jobs for convenience. Create
   sibling staging directory
   `/work/hdd/bfna/ikhire/.regression_5fold_v1_migration_v1/`. If a planned new
   destination already exists, verify identical content or stop on a conflict;
   do not overwrite unrelated files.
2. **Copy and preserve provenance.** Copy only allowlisted trees without
   deletion flags. Preserve the twelve original family geometry/manifest files
   under `_migration_v1/original_family_tables/<family>/`, then apply the
   twelve verified local replacements to the staged family trees.
3. **Verify before publication.** Add a cluster verification entrypoint and
   launcher using one CPU, 3 GiB, one hour, `scratch&work`, and required mail
   settings. It streams hashes/tables, writes audit reports, and performs no
   fitting, preparation, production aggregation, or scoring finalization.
   The provisional memory request uses the documented consolidation comparator
   with additional walltime for copied-file hashing. Reconsult the repository
   memory/walltime guides and live Delta documentation before writing this
   launcher. Record actual resource use after execution.
4. **Publish and cut over.** Publish verified new directories into the existing
   results root, create the relative triangle alias, and normalize permissions.
   Retain existing HDD export directories in place. Mark the collection
   complete only after all checks pass. Preserve staging remnants.

Write these audit files beneath the final results root:

- `_migration_v1/inventory.csv`: one row per copied file; source, destination,
  authority, byte count, and SHA-256.
- `_migration_v1/verification.csv`: one row per integrity, coverage,
  deployment, or access check; expected/observed values and pass status.
- `_migration_v1/migration_manifest.json`: source mapping, original/refreshed
  hashes, verification extent, and execution provenance.
- `collection_manifest.json`: collection version, completion status, component
  locations, and audit references. Distinguish shared components from the
  intentionally repository-only derived analyses.

Acceptance checks must establish:

- SHA-256 equality for every copied file and retained replay artifact.
- Frozen cohort/fold agreement, complete layer coverage, expected row counts,
  and unique evaluation identities.
- Exact equality with authoritative local geometry replacements; unchanged
  non-TM values and matching optimized-ledger provenance.
- Unchanged resident export manifests and compact-output hashes, plus
  representative distance/PDB reads across layers 0, 23, and 47 and all four
  probes.
- Correct group read/traverse access and absence of group-write access
  throughout the shared trees, including effective ACL masks.
- Passing local path-resolution, flat-deployment, geometry-refresh, and
  existing focused workflow tests. Retain the four documented geometry-oracle
  expected failures; fixing unrelated scientific code is outside this task.
- Published-root write refusal and changed-identity rejection, with successful
  initialization against a fresh test root.

Preserve `validation.mode=consolidation_only`. Migration checks are not an
exhaustive physical audit of resident prediction arrays and reconstructed PDBs.

Commit and push significant code changes before deployment. After successful
cutover, update active runbooks, project status, and `branch-analysis.md`, then
commit and push the documentation. Explicitly document the new shared code
deployment root and the intentionally omitted local-only derived bundles.
Report exact changed cluster paths and verification job IDs in chat. Do not
delete old copies or staging data.

## Execution handoff — October 23, 2026

- The first implementation action, `ssh delta hostname`, succeeded on
  `dt-login03.delta.ncsa.illinois.edu`. Frozen metadata and optimized manifest
  hashes matched the recorded authority; quota/group membership were checked,
  and no active jobs were present before submission.
- Implementation `d18b486` was pushed before deploying 61 runtime files flat at
  `/work/hdd/bfna/ikhire/openfold_interpretability_canonical/`. Its
  `deployment_manifest.json` records Git/source mapping and binary provenance.
  The retained Linux TMscore SHA-256 is
  `74c009402184dfb35871b0341d5878f11ce5772cbfe15eb86acba842696d9855`.
- The twelve refreshed geometry/manifest files and read-only authority report
  were uploaded under that deployment's `migration_inputs/`. Local geometry
  preparation passed all 240 checks, verified the 105,984-row ledger, and
  proposed no byte changes. Shared deployment access was normalized; HOME
  deployment/results and untracked local MDS exports were left untouched.
- Job `23142906` uses one CPU, 3 GiB, one hour, and `scratch&work`. Logs are
  `canonical_migration_23142906.{out,err}` in the shared deployment. Staging is
  `/work/hdd/bfna/ikhire/.regression_5fold_v1_migration_v1/`.
- Last observed: all 13,312 copied files passed hash/byte checks and HOME source
  rechecks. Frozen metadata and shared-OLS retained-model/schema/coverage
  checks passed; remaining protocol, geometry, resident-export, publication,
  and access checks were still running. No migration failure was reported.
  Monitoring stopped at the user's request, not because completion was verified.
- Local verification: 135 tests passed, four documented geometry-oracle
  expected failures retained. External input/output overrides reproduced the
  existing TM-score, coefficient, gain, and split numeric reports byte for byte
  in separate temporary directories; repository derived bundles were unchanged.

After the job finishes, inspect `sacct -j 23142906`, logs, final
`collection_manifest.json`, and `_migration_v1/verification.csv`. Record actual
elapsed/MaxRSS and confirm every check passed before updating project status
and completing the shared-storage runbook cutover. Preserve partial state if
it failed; the migration entrypoint deliberately refuses an uninspected retry
against existing staging or published destinations. Do not delete or restart
completed production workflows. Resident physical validation remains
`consolidation_only` with representative reads; omitted derived bundles remain
repository-only. Commit/push final documentation before synchronizing changed
runtime READMEs and their deployment inventory, retaining prior deployed copies.

"""Migration boundaries: flat imports, protected writes, provenance and collisions."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile
import csv
from types import SimpleNamespace
import time

import pytest

CLUSTER = Path(__file__).resolve().parents[2]
REPOSITORY = CLUSTER.parents[1]
for path in (CLUSTER, CLUSTER / 'cross_validation', CLUSTER / 'canonical_migration',
             CLUSTER / 'predicted_distances', CLUSTER / 'tm_scores'):
    sys.path.insert(0, str(path))

import canonical_storage as storage
from canonical_deployment import runtime_mapping, build_package
from migrate_canonical_results import copy_file, safe_path
import migrate_canonical_results as migration


def test_storage_defaults_and_overrides(tmp_path, monkeypatch):
    monkeypatch.delenv('CSV_ROOT', raising=False)
    assert storage.csv_root() == REPOSITORY / 'delta_files'
    monkeypatch.setattr(storage, '__file__', str(tmp_path / 'flat/canonical_storage.py'))
    assert storage.csv_root() == Path('/work/hdd/bfna/ikhire')
    monkeypatch.setenv('CSV_ROOT', str(tmp_path / 'fresh'))
    assert storage.result_root() == tmp_path / 'fresh/regression_5fold_v1'
    assert storage.workflow_root('triangle') == tmp_path / 'fresh/regression_5fold_v1/triangle'
    monkeypatch.setenv('TRIANGLE_OUTPUT_ROOT', str(tmp_path / 'triangle_rerun'))
    assert storage.workflow_root('triangle', 'TRIANGLE_OUTPUT_ROOT') == tmp_path / 'triangle_rerun'


def test_protected_mutators_do_not_create_locks_or_outputs(tmp_path, monkeypatch):
    root = tmp_path / 'published'
    root.mkdir()
    (root / 'collection_manifest.json').write_text('{"status":"complete"}')
    alias = tmp_path / 'alias'
    alias.symlink_to(root, target_is_directory=True)
    before = sorted(root.rglob('*'))
    import predicted_distance_export_common as export
    import recompute_tm_scores as tm
    from regression_output_helpers import atomic_write_json
    for target in (root, alias):
        with pytest.raises(PermissionError):
            with export.task_lock(target / 'predictions', 'test'):
                pytest.fail('protected lock was acquired')
        with pytest.raises(PermissionError):
            tm.ensure_output(target / 'scores', None)
        with pytest.raises(PermissionError):
            atomic_write_json(target / 'new/manifest.json', {})
    for module, function in (
        ('shared_ols_base256_common', 'run_fold_layer_task'),
        ('per_protein_ols_base256_common', 'run_layer_task'),
        ('sequence_gated_ols_base256_common', 'run_fold_layer_task'),
        ('shared_xgboost_base256_fixed_s4_common', 'run_fold_layer_task'),
        ('aggregate_shared_ols_base256', 'aggregate_shared_ols'),
        ('aggregate_per_protein_ols_base256', 'aggregate_per_protein_ols'),
        ('aggregate_sequence_gated_ols_base256', 'aggregate_sequence_gated_ols'),
        ('aggregate_shared_xgboost_base256_fixed_s4', 'aggregate_shared_xgboost'),
    ):
        with pytest.raises(PermissionError):
            getattr(importlib.import_module(module), function)(root=root / 'family')
    assert sorted(root.rglob('*')) == before
    fresh = tmp_path / 'fresh'
    with export.task_lock(fresh, 'test'):
        assert (fresh / '.locks/test.lock').is_file()
    atomic_write_json(fresh / 'run_manifest.json', {'run_status': 'prepared'})


def test_copy_preserves_sources_and_rejects_conflicts(tmp_path):
    source, destination = tmp_path / 'source', tmp_path / 'stage/result'
    source.write_bytes(b'original')
    digest = copy_file(source, destination)
    assert copy_file(source, destination) == digest
    destination.write_bytes(b'conflict')
    with pytest.raises(ValueError, match='Conflicting'):
        copy_file(source, destination)
    assert source.read_bytes() == b'original'
    assert destination.read_bytes() == b'conflict'
    with pytest.raises(ValueError, match='Unsafe'):
        safe_path(tmp_path, '../escape')


def test_flat_package_imports_and_launcher_references(tmp_path, monkeypatch):
    binary = tmp_path / 'TMscore'
    binary.write_bytes(b'\x7fELFtest fixture')
    package = tmp_path / 'deployment.tar'
    manifest = build_package(REPOSITORY, binary, package)
    flat = tmp_path / 'flat'
    flat.mkdir()
    with tarfile.open(package) as archive:
        archive.extractall(flat, filter='data')
    assert all('tests/' not in r['source'] for r in manifest['files'])
    assert len({r['deployed'] for r in manifest['files']}) == len(manifest['files'])
    code = '''
import json
from pathlib import Path
import canonical_storage, cross_validation_common
import predicted_distance_export_common as export
import recompute_tm_scores, consolidate_tm_scores
import triangle_violation_subgroup_replay as triangle
import replay_per_protein_subgroup_mae as publication
from verify_cross_validation_deployment import verify
root = Path.cwd()
assert canonical_storage.deployment_revision(root) == json.loads((root/'deployment_manifest.json').read_text())['git_commit']
assert cross_validation_common.fivefold_result_root() == Path('/work/hdd/bfna/ikhire/regression_5fold_v1')
assert triangle.output_root() == cross_validation_common.fivefold_result_root()/'triangle_violation_subgroups_5fold_v1'
assert publication.output_root() == cross_validation_common.fivefold_result_root()/'analysis/publication_subgroup_mae_5fold_v1'
assert export.readme_source().is_file()
'''
    env = {**os.environ, 'PYTHONDONTWRITEBYTECODE': '1'}
    env.pop('CSV_ROOT', None)
    subprocess.run([sys.executable, '-c', code], cwd=flat, env=env, check=True)
    for name, source in runtime_mapping(CLUSTER).items():
        if source.suffix != '.sbatch':
            continue
        content = source.read_text()
        subprocess.run(['bash', '-n', str(source)], check=True)
        assert '--mail-type=BEGIN,END,FAIL' in content
        assert '--mail-user=ikhire3@gatech.edu' in content
        assert '--constraint="scratch&work"' in content
        for entrypoint in content.split('$SCRIPT_DIR/')[1:]:
            filename = entrypoint.split('"')[0]
            if filename.endswith('.py'):
                assert (flat / filename).is_file(), (name, filename)
        for line in content.splitlines():
            if line.startswith(('#SBATCH -o ', '#SBATCH -e ')):
                assert 'regression_5fold_v1/' not in line


@pytest.fixture
def interrupted_publication(tmp_path):
    args = SimpleNamespace(root=tmp_path / 'results', stage=tmp_path / 'stage',
                           deployment=tmp_path / 'deployment', home=tmp_path / 'home')
    authority = {'schema': 'test_authority'}
    audit_root = args.root / '_migration_v1'
    audit_root.mkdir(parents=True)
    args.stage.mkdir()
    args.deployment.mkdir()
    for name in (*migration.FAMILIES, 'metadata', migration.TRIANGLE,
                 migration.PUBLICATION, *migration.RESIDENT):
        (args.root / name).mkdir(parents=True, exist_ok=True)
    (args.root / 'analysis' / migration.TRIANGLE).symlink_to(f'../{migration.TRIANGLE}')
    source = tmp_path / 'original_HOME.csv'
    source.write_text('preserved')
    relative = Path(next(iter(migration.FAMILIES))) / 'data.csv'
    checksum = copy_file(source, args.root / relative)
    with (audit_root / 'inventory.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('source', 'destination', 'authority', 'byte_count', 'sha256'))
        writer.writeheader()
        writer.writerow(dict(source=str(source), destination=str(args.root / relative),
                             authority='original_HOME', byte_count=source.stat().st_size, sha256=checksum))
    manifest = dict(schema='canonical_results_migration_v1', status='verifying_access',
                    result_root=str(args.root), stage_root=str(args.stage),
                    deployment_root=str(args.deployment), authority=authority,
                    source_mapping={k: str(v) for k, v in migration.source_mapping(args.home).items()},
                    verification_job_id='old_job', copied_files=1, git_commit='old_commit',
                    validation={'mode': 'consolidation_only'})
    migration.write_json(audit_root / 'migration_manifest.json', manifest)
    migration.write_json(audit_root / 'authority.json', authority)
    migration.write_json(audit_root / 'resident_snapshot.json', {})
    migration.write_json(args.root / 'collection_manifest.json', dict(
        collection_version='regression_5fold_v1_migration_v1', status='publishing'))
    migration.write_json(args.deployment / 'deployment_manifest.json', {'git_commit': 'new_commit'})
    audit = migration.Audit(audit_root / 'verification.csv')
    audit.check('deployment:last_file', True, True)
    audit.close()
    return args, authority, relative


@pytest.mark.parametrize('change', ['complete', 'identity', 'missing_component', 'failed_audit', 'staged_component'])
def test_resume_rejects_conflicting_or_completed_publications(interrupted_publication, change):
    args, authority, relative = interrupted_publication
    audit_root = args.root / '_migration_v1'
    if change == 'complete':
        path = args.root / 'collection_manifest.json'
        value = migration.read_json(path)
        value['status'] = 'complete'
        migration.write_json(path, value)
    elif change == 'identity':
        authority = {'schema': 'changed_authority'}
    elif change == 'missing_component':
        (args.root / 'metadata').rename(args.root / 'retained_metadata')
    elif change == 'failed_audit':
        with (audit_root / 'verification.csv').open('a') as handle:
            handle.write('failed,True,False,False\n')
    else:
        (args.stage / 'metadata').mkdir()
    before = {p: p.read_bytes() for p in args.root.rglob('*') if p.is_file()}
    with pytest.raises(ValueError):
        migration.inspect_published_resume(args, authority)
    assert all(p.read_bytes() == data for p, data in before.items())


def test_resume_rehashes_published_files_and_preserves_old_evidence(interrupted_publication, monkeypatch):
    args, authority, relative = interrupted_publication
    audit_root = args.root / '_migration_v1'
    previous = (audit_root / 'verification.csv').read_bytes()
    calls = []
    monkeypatch.setenv('SLURM_JOB_ID', 'new_job')
    monkeypatch.setattr(migration, 'verify_tables', lambda root, resident, *_: calls.append(('tables', root, resident)))
    monkeypatch.setattr(migration, 'verify_geometry', lambda root, resident, *_: calls.append(('geometry', root, resident)))
    monkeypatch.setattr(migration, 'verify_deployment', lambda *_: [('runtime', 'hash', 'hash', True)])
    monkeypatch.setattr(migration, 'finish_publication', lambda *a: calls.append(('finish', a[1], a[3])))
    migration.resume_published(args, authority, time.perf_counter())
    assert calls[:2] == [('tables', args.root, args.root), ('geometry', args.root, args.root)]
    manifest, recovery = calls[-1][1:]
    assert manifest['initial_verification_job_id'] == 'old_job'
    assert manifest['verification_job_id'] == 'new_job'
    assert manifest['git_commit'] == 'old_commit'
    assert manifest['recovery_attempts'][0]['git_commit'] == 'new_commit'
    assert (recovery / 'previous_verification.csv').read_bytes() == previous
    assert (args.root / relative).read_text() == 'preserved'
    assert not list(args.stage.iterdir())
    assert (audit_root / 'verification.csv').read_bytes().startswith(previous)


def test_resume_aborts_before_finalization_if_published_copy_changed(interrupted_publication, monkeypatch):
    args, authority, relative = interrupted_publication
    (args.root / relative).write_text('corrupted')
    previous = (args.root / '_migration_v1/verification.csv').read_bytes()
    monkeypatch.setattr(migration, 'finish_publication', lambda *_: pytest.fail('finalized corrupt copy'))
    with pytest.raises(ValueError, match='copy_hash'):
        migration.resume_published(args, authority, time.perf_counter())
    assert (args.root / '_migration_v1/verification.csv').read_bytes() == previous
    assert migration.read_json(args.root / 'collection_manifest.json')['status'] == 'publishing'


def test_permission_normalization_batches_across_directories(tmp_path, monkeypatch):
    import grp
    import stat
    monkeypatch.setattr(storage, 'SHARED_PARENT', tmp_path)
    monkeypatch.setattr(grp, 'getgrnam', lambda _: SimpleNamespace(gr_gid=os.getgid()))
    calls = []
    monkeypatch.setattr(storage.subprocess, 'run', lambda command, **_: calls.append(command))
    chmod_calls = []
    real_chmod = os.chmod
    def record_chmod(path, mode, **kwargs):
        chmod_calls.append((Path(path), mode))
        real_chmod(path, mode, **kwargs)
    monkeypatch.setattr(storage.os, 'chmod', record_chmod)
    root = tmp_path / 'results'
    for i in range(260):
        directory = root / str(i)
        directory.mkdir(parents=True)
        (directory / 'data.csv').write_text('data')
    executable = root / 'tool'
    executable.write_text('tool')
    executable.chmod(0o700)
    (root / 'alias').symlink_to(root / '0/data.csv')
    storage.normalize_shared_tree(root)
    assert len(calls) == 7  # Two directory batches (access/default), two file batches, one executable.
    assert all('alias' not in p for call in calls for p in call[3:])
    assert stat.S_IMODE(executable.stat().st_mode) == 0o750
    # Some local filesystems discard setgid; check both the request and access bits.
    assert all((root / str(i), 0o2750) in chmod_calls for i in range(260))
    assert all(stat.S_IMODE((root / str(i)).stat().st_mode) & 0o777 == 0o750 for i in range(260))
    assert all(stat.S_IMODE((root / str(i) / 'data.csv').stat().st_mode) == 0o640 for i in range(260))

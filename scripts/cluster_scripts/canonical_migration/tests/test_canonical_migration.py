"""Migration boundaries: flat imports, protected writes, provenance and collisions."""
import importlib
import json
import os
from pathlib import Path
import subprocess
import sys
import tarfile

import pytest

CLUSTER = Path(__file__).resolve().parents[2]
REPOSITORY = CLUSTER.parents[1]
for path in (CLUSTER, CLUSTER / 'cross_validation', CLUSTER / 'canonical_migration',
             CLUSTER / 'predicted_distances', CLUSTER / 'tm_scores'):
    sys.path.insert(0, str(path))

import canonical_storage as storage
from canonical_deployment import runtime_mapping, build_package
from migrate_canonical_results import copy_file, safe_path


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

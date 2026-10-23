#!/usr/bin/env python3
"""Copy, audit, and publish the explicitly allowlisted canonical collection.

Never delete, fit, replay production predictions, or rewrite resident manifests.
Every destination collision must be byte-identical before it can be reused.
"""
from __future__ import annotations

import argparse
import csv
from collections import Counter
import grp
import json
import math
import os
import resource
import shutil
import stat
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

SCRIPT_DIR = Path(__file__).resolve().parent
for directory in (SCRIPT_DIR, SCRIPT_DIR.parent, SCRIPT_DIR.parent / 'cross_validation',
                  SCRIPT_DIR.parent / 'predicted_distances', SCRIPT_DIR.parent / 'triangle_violation_subgroups'):
    sys.path.insert(0, str(directory))
from canonical_deployment import sha256
from canonical_storage import SHARED_PARENT, normalize_shared_tree, assert_writable
from verify_cross_validation_deployment import verify as verify_deployment

FAMILIES = {'shared_ols_base256': 240, 'per_protein_ols_base256': 132480,
            'sequence_gated_ols_base256': 1440, 'shared_xgboost_base256_fixed_s4': 240}
TRIANGLE = 'triangle_violation_subgroups_5fold_v1'
PUBLICATION = 'analysis/publication_subgroup_mae_5fold_v1'
RESIDENT = ('predicted_distances_5fold_v1', 'tm_score_optimized_v1')
TABLES = ('geometry_metrics.csv', 'geometry_summary.csv', 'run_manifest.json')
OMITTED = ('linear_coefficients_analysis', 'xgboost_gain_dimensions',
           'xgboost_split_dimensions', 'tm_score_optimization_comparison_v1')


def log(message):
    print(f'[{datetime.now(timezone.utc).isoformat()}] {message}', flush=True)


def read_json(path):
    return json.loads(Path(path).read_text())


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + '\n')


def rows(path):
    with Path(path).open(newline='') as handle:
        yield from csv.DictReader(handle)


def safe_path(root, relative):
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or path == root:
        raise ValueError(f'Unsafe retained path: {relative}')
    return path


class Audit:
    def __init__(self, path):
        self.handle = Path(path).open('w', newline='')
        self.writer = csv.DictWriter(self.handle, fieldnames=('check', 'expected', 'observed', 'passed'))
        self.writer.writeheader()
        self.count = 0

    def check(self, label, expected, observed):
        passed = expected == observed
        self.writer.writerow(dict(check=label, expected=str(expected), observed=str(observed), passed=passed))
        self.handle.flush()
        self.count += 1
        if not passed:
            raise ValueError(f'{label}: expected {expected!r}; observed {observed!r}')

    def close(self):
        self.handle.close()


def copy_file(source, destination):
    """Exclusive new copy, or identical reuse. Preserve all source files."""
    source, destination = Path(source), Path(destination)
    if source.is_symlink():
        raise ValueError(f'Unexpected source symlink: {source}')
    before = sha256(source)
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists():
        if not destination.is_file() or sha256(destination) != before:
            raise ValueError(f'Conflicting migration destination: {destination}')
    else:
        with source.open('rb') as incoming, destination.open('xb') as outgoing:
            shutil.copyfileobj(incoming, outgoing, 1024 * 1024)
        shutil.copystat(source, destination)
    if sha256(source) != before or sha256(destination) != before:
        raise ValueError(f'Copy/source changed: {source}')
    return before


def resident_snapshot(root):
    snapshot = {}
    for name in RESIDENT:
        directory = root / name
        # Compact tables, manifests and metadata; physical arrays/PDBs remain in place.
        paths = list(directory.glob('*.csv')) + list(directory.glob('*.json'))
        paths += [p for p in (directory / 'metadata').glob('*') if p.is_file()]
        manifest = read_json(directory / 'run_manifest.json')
        paths += [safe_path(directory, relative) for relative in manifest.get('output_sha256', {})]
        for path in paths:
            snapshot[str(path.relative_to(root))] = sha256(path)
    return snapshot


def source_mapping(home):
    legacy = home / 'csv_files'
    return {**{name: legacy / 'regression_5fold_v1' / name for name in FAMILIES},
            'metadata': legacy / 'regression_5fold_v1/metadata',
            TRIANGLE: legacy / 'analysis' / TRIANGLE,
            PUBLICATION: legacy / PUBLICATION}


def stage(args, authority):
    if args.stage.exists():
        raise ValueError(f'Staging already exists; inspect before any retry: {args.stage}')
    for name in (*FAMILIES, 'metadata', TRIANGLE, PUBLICATION, '_migration_v1', 'collection_manifest.json'):
        if (args.root / name).exists():
            raise ValueError(f'Publication collision: {args.root / name}')
    sources = source_mapping(args.home)
    for name, checksum in authority['metadata_sha256'].items():
        if sha256(sources['metadata'] / name) != checksum:
            raise ValueError(f'Frozen metadata changed: {name}')
    optimized = args.root / RESIDENT[1]
    if sha256(optimized / 'run_manifest.json') != authority['optimized_manifest_sha256']:
        raise ValueError('Optimized root manifest changed')
    for family in FAMILIES:
        if sha256(sources[family] / 'geometry_metrics.csv') != authority['geometry_provenance'][family]['source_geometry_sha256']:
            raise ValueError(f'Original HOME geometry changed: {family}')
    snapshot = resident_snapshot(args.root)
    audit = args.stage / '_migration_v1'
    audit.mkdir(parents=True)
    copy_file(args.authority, audit / 'authority.json')
    write_json(audit / 'resident_snapshot.json', snapshot)
    with (audit / 'inventory.csv').open('w', newline='') as handle:
        writer = csv.DictWriter(handle, fieldnames=('source', 'destination', 'authority', 'byte_count', 'sha256'))
        writer.writeheader()
        count = 0
        for component, source in sources.items():
            log(f'Copying {source} -> {args.stage / component}')
            for path in sorted(source.rglob('*')):
                if component == 'metadata' and path.name not in authority['metadata_sha256']:
                    continue
                if path.is_symlink():
                    raise ValueError(f'Unexpected symlink: {path}')
                if path.is_dir():
                    (args.stage / component / path.relative_to(source)).mkdir(parents=True, exist_ok=True)
                    continue
                relative = str(Path(component) / path.relative_to(source))
                # Preserve originals separately; the staged family receives authoritative replacements.
                if component in FAMILIES and path.parent == source and path.name in TABLES:
                    relative = f'_migration_v1/original_family_tables/{component}/{path.name}'
                destination = args.stage / relative
                checksum = copy_file(path, destination)
                writer.writerow(dict(source=str(path), destination=str(args.root / relative),
                                     authority='original_HOME', byte_count=path.stat().st_size, sha256=checksum))
                count += 1
        for relative, expected in authority['replacements'].items():
            source = args.replacements / relative
            if sha256(source) != expected:
                raise ValueError(f'Authoritative upload checksum mismatch: {relative}')
            copy_file(source, args.stage / relative)
            writer.writerow(dict(source=f'local:delta_files/regression_5fold_v1/{relative}',
                                 destination=str(args.root / relative), authority='refreshed_local',
                                 byte_count=source.stat().st_size, sha256=expected))
            count += 1
    log(f'Staged {count} files; originals and resident outputs preserved')
    return sources


def numeric_equal(a, b):
    if a == b:
        return True
    try:
        a, b = float(a), float(b)
        return (math.isnan(a) and math.isnan(b)) or math.isclose(a, b, rel_tol=1e-12, abs_tol=1e-12)
    except ValueError:
        return False


def table_check(path, expected_count, keys, audit, *, proteins=None, families=None):
    """Stream rows; retain only compact identity tuples, never a table of dictionaries."""
    identities, layers, pids, models = set(), set(), set(), set()
    count = 0
    coverage = Counter()
    for row in rows(path):
        key = tuple(row[k] for k in keys)
        if key in identities:
            raise ValueError(f'Duplicate evaluation identity: {path}/{key}')
        identities.add(key)
        count += 1
        if 'layer' in row:
            layer = int(row['layer'])
            if layer not in range(48):
                raise ValueError(f'Invalid layer: {path}/{layer}')
            layers.add(layer)
        if proteins is not None:
            pid = row['protein_id']
            if pid not in proteins:
                raise ValueError(f'Unexpected protein: {path}/{pid}')
            pids.add(pid)
            if 'outer_fold' in row and int(row['outer_fold']) != int(proteins[pid]['outer_fold']):
                raise ValueError(f'Fold mismatch: {path}/{pid}')
        if proteins is not None and families is not None:
            coverage[(row['analysis_name'], int(row['layer']), row['protein_id'])] += 1
        if families is not None:
            if row['analysis_name'] not in families:
                raise ValueError(f'Unexpected family: {path}')
            models.add(row['analysis_name'])
        if 'evaluation_scope' in row and row['evaluation_scope'] != 'oof_all_pairs':
            raise ValueError(f'Invalid OOF scope: {path}')
        if 'cv_protocol_version' in row and row['cv_protocol_version'] != '2.0':
            raise ValueError(f'Invalid protocol: {path}')
        if 'pair_coverage' in row and float(row['pair_coverage']) != 1:
            raise ValueError(f'Incomplete pairs: {path}')
        for column in ('status', 'prediction_status', 'geometry_status'):
            if column in row and row[column] != 'complete':
                raise ValueError(f'Incomplete {column}: {path}')
    if proteins is not None and families is not None:
        for family in families:
            per_identity = 1
            if path.name == 'per_protein_subgroup_mae.csv':
                per_identity = 11
            elif path.name == 'per_protein_triangle_metrics.csv':
                per_identity = 28 if family == 'per_protein_ols_base256' else 25
            for layer in range(48):
                for pid in proteins:
                    if coverage[(family, layer, pid)] != per_identity:
                        raise ValueError(f'Incomplete protein/layer/family coverage: {path}/{family}/{layer}/{pid}')
    audit.check(f'rows:{path}', expected_count, count)
    if 'layer' in keys:
        audit.check(f'layers:{path}', set(range(48)), layers)
    if proteins is not None:
        audit.check(f'cohort:{path}', set(proteins), pids)
    if families is not None:
        audit.check(f'families:{path}', set(families), models)


def verify_geometry(stage_root, resident_root, authority, audit):
    score_root = resident_root / RESIDENT[1]
    ledger = {(r['analysis_name'], r['protein_id'], r['layer']): r for r in rows(score_root / 'tm_scores.csv')}
    audit.check('ledger_checksum', authority['ledger_sha256'], sha256(score_root / 'tm_scores.csv'))
    for family in FAMILIES:
        directory = stage_root / family
        original = stage_root / '_migration_v1/original_family_tables' / family
        for filename in TABLES:
            audit.check(f'local_authority:{family}/{filename}', authority['replacements'][f'{family}/{filename}'],
                        sha256(directory / filename))
        old_manifest = read_json(original / 'run_manifest.json')
        refreshed_manifest = read_json(directory / 'run_manifest.json')
        refresh = refreshed_manifest.pop('tm_score_refresh')
        audit.check(f'fit_provenance:{family}', old_manifest, refreshed_manifest)
        audit.check(f'optimized_provenance:{family}', authority['geometry_provenance'][family], refresh)
        old_rows = list(rows(original / 'geometry_metrics.csv'))
        new_rows = list(rows(directory / 'geometry_metrics.csv'))
        audit.check(f'geometry_row_count:{family}', len(old_rows), len(new_rows))
        changed = 0
        for old, new in zip(old_rows, new_rows):
            if any(old[k] != new[k] for k in old if k != 'tm_score'):
                raise ValueError(f'Non-TM geometry changed: {family}')
            score = ledger[(family, new['protein_id'], new['layer'])]
            if not numeric_equal(new['tm_score'], score['tm_score']) or not numeric_equal(old['tm_score'], score['tm_score_saved']):
                raise ValueError(f'Geometry/ledger TM mismatch: {family}')
            changed += old['tm_score'] != new['tm_score']
        audit.check(f'non_TM_geometry:{family}', True, True)
        audit.check(f'optimized_TM_changed:{family}', True, changed > 0)
        old_summary, new_summary = list(rows(original / 'geometry_summary.csv')), list(rows(directory / 'geometry_summary.csv'))
        audit.check(f'summary_row_count:{family}', len(old_summary), len(new_summary))
        for old, new in zip(old_summary, new_summary):
            allowed = ('mean', 'std', 'median', 'q25', 'q75') if old['geometry_metric'] == 'tm_score' else ()
            if any(not numeric_equal(old[k], new[k]) for k in old if k not in allowed):
                raise ValueError(f'Non-TM summary changed: {family}')
        audit.check(f'non_TM_summary:{family}', True, True)


def verify_tables(stage_root, resident_root, authority, audit):
    from cross_validation_common import load_cv_folds, V2_CSV_SCHEMAS

    from triangle_violation_subgroup_common import (FAMILY_SPECS, validate_family_manifest,
                                                    load_and_validate_replay_index)
    bundle = load_cv_folds(metadata_dir=stage_root / 'metadata')
    proteins = {r['protein_id']: r for r in rows(stage_root / 'metadata/regression_outer_folds_v1.csv')}
    audit.check('frozen_cohort_size', 552, len(proteins))
    for name, checksum in authority['metadata_sha256'].items():
        audit.check(f'frozen_metadata:{name}', checksum, sha256(stage_root / 'metadata' / name))
    for spec in FAMILY_SPECS:
        root = stage_root / spec.family_dir
        manifest = validate_family_manifest(root, spec, bundle.fold_digests)
        replay = load_and_validate_replay_index(root, spec, manifest)
        audit.check(f'replay_artifacts:{spec.analysis_name}', FAMILIES[spec.analysis_name], len(replay))
        del replay
        for filename, count, keys in (
            ('per_protein.csv', 26496, ('protein_id', 'layer')),
            ('geometry_metrics.csv', 26496, ('protein_id', 'layer')),
            ('pooled_metrics.csv', 48, ('layer',)),
            ('fold_metrics.csv', 240, ('layer', 'outer_fold')),
            ('geometry_summary.csv', 672, ('layer', 'geometry_metric')),
        ):
            with (root / filename).open(newline='') as handle:
                audit.check(f'schema:{spec.analysis_name}/{filename}', list(V2_CSV_SCHEMAS[filename]), next(csv.reader(handle)))
            table_check(root / filename, count, keys, audit,
                        proteins=proteins if 'protein_id' in keys else None,
                        families=(spec.analysis_name,))
        audit.check(f'family_manifest:{spec.analysis_name}', 'complete', manifest['run_status'])
    triangle = read_json(stage_root / TRIANGLE / 'run_manifest.json')
    for key in ('cohort_digest', 'outer_fold_digest', 'pair_fold_manifest_digest'):
        audit.check(f'triangle:{key}', bundle.fold_digests[key], triangle[key])
    audit.check('triangle_complete', 'complete', triangle['analysis_status'])
    audit.check('triangle_manifest_rows', 2729088, triangle['n_per_protein_triangle_metric_rows'])
    table_check(stage_root / TRIANGLE / 'per_protein_triangle_metrics.csv', 2729088,
                ('layer', 'analysis_name', 'protein_id', 'category_system', 'category_label'),
                audit, proteins=proteins, families=FAMILIES)
    table_check(stage_root / TRIANGLE / 'triangle_metric_summary.csv', triangle['n_triangle_metric_summary_rows'],
                ('layer', 'analysis_name', 'category_system', 'category_label', 'metric'), audit, families=FAMILIES)
    audit.check('triangle_geometry_reproduction', 0, triangle['n_geometry_reproduction_failures'])
    publication = read_json(stage_root / PUBLICATION / 'run_manifest.json')
    audit.check('publication_complete', 'complete', publication['analysis_status'])
    audit.check('publication_manifest_rows', 1165824, publication['n_per_protein_subgroup_rows'])
    table_check(stage_root / PUBLICATION / 'per_protein_subgroup_mae.csv', 1165824,
                ('layer', 'analysis_name', 'protein_id', 'category_scope', 'category_label'),
                audit, proteins=proteins, families=FAMILIES)
    for name, filename in ((RESIDENT[0], 'prediction_index.csv'),
                           (RESIDENT[1], 'tm_scores.csv'), (RESIDENT[1], 'pdb_index.csv')):
        table_check(resident_root / name / filename, 105984, ('layer', 'analysis_name', 'protein_id'),
                    audit, proteins=proteins, families=FAMILIES)
    for name in RESIDENT:
        root = resident_root / name
        manifest = read_json(root / 'run_manifest.json')
        audit.check(f'resident_complete:{name}', 'complete', manifest['run_status'])
        for key in ('output_sha256', 'final_files_sha256', 'metadata_sha256'):
            for relative, checksum in manifest.get(key, {}).items():
                audit.check(f'resident_hash:{name}/{relative}', checksum, sha256(safe_path(root, relative)))
        for layer, checksum in manifest['layer_manifest_sha256'].items():
            if name == RESIDENT[0]:
                path = root / 'layers' / f'layer_{int(layer):02d}' / 'layer_manifest.json'
            else:
                path = root / 'layers' / f'layer_{int(layer):02d}' / 'layer_manifest.json'
            audit.check(f'resident_layer_hash:{path}', checksum, sha256(path))
    audit.check('resident_validation_extent', 'consolidation_only',
                read_json(resident_root / RESIDENT[1] / 'run_manifest.json')['validation']['mode'])
    # Readers validate against retained manifest identities, without comparing current code hashes.
    from predicted_distance_export_common import load_predicted_distances
    from cluster_regression_helpers import extract_ca_coordinates
    pid = sorted(proteins)[0]
    requested = {(family, str(layer), pid) for family in FAMILIES for layer in (0, 23, 47)}
    pdb_rows = {(r['analysis_name'], r['layer'], r['protein_id']): r
                for r in rows(resident_root / RESIDENT[1] / 'pdb_index.csv')
                if (r['analysis_name'], r['layer'], r['protein_id']) in requested}
    for family, layer, pid in sorted(requested):
        values = load_predicted_distances(resident_root / RESIDENT[0], int(layer), pid, family)
        audit.check(f'distance_read:{family}/{layer}/{pid}', int(proteins[pid]['n_pairs']), len(values))
        row = pdb_rows[family, layer, pid]
        path = safe_path(resident_root / RESIDENT[1], row['pdb_path'])
        audit.check(f'pdb_hash:{family}/{layer}/{pid}', row['pdb_sha256'], sha256(path))
        coordinates = extract_ca_coordinates(path)['coordinates']
        audit.check(f'pdb_read:{family}/{layer}/{pid}', int(row['n_residues']), len(coordinates))


def verify_inventory(args, audit):
    count = 0
    for row in rows(args.stage / '_migration_v1/inventory.csv'):
        relative = Path(row['destination']).relative_to(args.root)
        destination = args.stage / relative
        audit.check(f'copy_hash:{relative}', row['sha256'], sha256(destination))
        audit.check(f'copy_bytes:{relative}', int(row['byte_count']), destination.stat().st_size)
        if row['authority'] == 'original_HOME':
            audit.check(f'source_unchanged:{row["source"]}', row['sha256'], sha256(row['source']))
        count += 1
    return count


def access_audit(root, audit):
    """Inspect modes and every effective ACL, including default masks."""
    gid = grp.getgrnam('delta_bfna').gr_gid
    file_count = directory_count = 0
    for directory, _, files in os.walk(root):
        path = Path(directory)
        info = path.stat()
        if info.st_gid != gid or stat.S_IMODE(info.st_mode) != 0o2750:
            raise ValueError(f'Incorrect shared directory access: {path}')
        directory_count += 1
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            info = path.stat()
            if info.st_gid != gid or stat.S_IMODE(info.st_mode) not in (0o640, 0o750):
                raise ValueError(f'Incorrect shared file access: {path}')
            file_count += 1
    process = subprocess.Popen(['getfacl', '-R', '-p', str(root)], stdout=subprocess.PIPE, text=True)
    current, mask, default_mask, entries = '', 'rwx', 'rwx', []

    def check_acl():
        for default, tag, who, permissions in entries:
            effective_mask = default_mask if default else mask
            effective = {c for c in permissions if c != '-'} & {c for c in effective_mask if c != '-'}
            if tag == 'group' and ('w' in effective or 'r' not in effective):
                raise ValueError(f'Incorrect effective group ACL: {current}/{tag}:{who}')
            if tag == 'group' and Path(current).is_dir() and 'x' not in effective:
                raise ValueError(f'Missing effective group traverse: {current}')
    for line in process.stdout:
        line = line.strip()
        if not line:
            check_acl()
            entries, mask, default_mask = [], 'rwx', 'rwx'
        elif line.startswith('# file: '):
            current = line[8:]
        elif not line.startswith('#'):
            fields = line.split()[0].split(':')
            default = fields[0] == 'default'
            if default:
                fields = fields[1:]
            tag, who, permissions = fields
            if tag == 'mask':
                if default:
                    default_mask = permissions
                else:
                    mask = permissions
            entries.append((default, tag, who, permissions))
    check_acl()
    if process.wait() != 0:
        raise ValueError(f'getfacl failed: {root}')
    audit.check(f'access_files:{root}', True, file_count > 0)
    audit.check(f'access_directories:{root}', True, directory_count > 0)
    log(f'Access audit: {root}: {file_count} files, {directory_count} directories')


def publish(args, authority, sources, started, count):
    write_json(args.root / 'collection_manifest.json',
               {'collection_version': 'regression_5fold_v1_migration_v1', 'status': 'publishing'})
    # Only newly copied staging directories are renamed; existing resident outputs stay put.
    for name in (*FAMILIES, 'metadata', TRIANGLE):
        source, destination = args.stage / name, args.root / name
        if destination.exists():
            raise ValueError(f'Cutover collision: {destination}')
        source.rename(destination)
    analysis = args.root / 'analysis'
    analysis.mkdir(exist_ok=True)
    (args.stage / PUBLICATION).rename(args.root / PUBLICATION)
    alias = analysis / TRIANGLE
    if alias.exists() or alias.is_symlink():
        raise ValueError(f'Alias collision: {alias}')
    alias.symlink_to(f'../{TRIANGLE}')
    (args.stage / '_migration_v1').rename(args.root / '_migration_v1')
    audit_root = args.root / '_migration_v1'
    manifest = {'schema': 'canonical_results_migration_v1', 'status': 'verifying_access',
                'source_mapping': {k: str(v) for k, v in sources.items()},
                'result_root': str(args.root), 'stage_root': str(args.stage),
                'deployment_root': str(args.deployment), 'git_commit': read_json(args.deployment / 'deployment_manifest.json')['git_commit'],
                'verification_job_id': os.environ.get('SLURM_JOB_ID'), 'copied_files': count,
                'authority': authority, 'validation': {'mode': 'consolidation_only',
                'resident_physical_artifacts': 'representative_reads_only_layers_0_23_47_all_four_probes'},
                'no_deletions': True, 'original_HOME_preserved': True}
    write_json(audit_root / 'migration_manifest.json', manifest)
    normalize_shared_tree(args.root)
    normalize_shared_tree(args.deployment)
    # Repair only the user-owned ancestor; allocation ancestors are audited, never rewritten.
    os.chmod(args.root.parent, 0o2750)
    subprocess.run(['setfacl', '-m', 'g::r-x,m::r-x,d:g::r-x,d:m::r-x', str(args.root.parent)], check=True)
    audit = Audit(audit_root / 'access_verification.csv')
    try:
        access_audit(args.root, audit)
        access_audit(args.deployment, audit)
        for path in (Path('/work'), Path('/work/hdd'), SHARED_PARENT.parent, SHARED_PARENT):
            audit.check(f'ancestor_traverse:{path}', True, bool(path.stat().st_mode & stat.S_IXGRP))
        audit.check('triangle_relative_alias', f'../{TRIANGLE}', os.readlink(alias))
        audit.check('triangle_alias_target', (args.root / TRIANGLE).resolve(), alias.resolve())
    finally:
        audit.close()
    # Merge the access checks into the normative single verification table.
    with (audit_root / 'verification.csv').open('a', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=('check', 'expected', 'observed', 'passed'))
        writer.writerows(rows(audit_root / 'access_verification.csv'))
    manifest.update(status='verifying_final_access', completed_at=datetime.now(timezone.utc).isoformat(),
                    elapsed_seconds=time.perf_counter() - started,
                    process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    write_json(audit_root / 'migration_manifest.json', manifest)
    collection = {'collection_version': 'regression_5fold_v1_migration_v1', 'status': 'verifying_final_access',
                  'completed_at': manifest['completed_at'],
                  'components': {name: name for name in (*FAMILIES, 'metadata', TRIANGLE, PUBLICATION, *RESIDENT)},
                  'triangle_analysis_alias': f'analysis/{TRIANGLE}',
                  'repository_only_derived_analyses': [f'analysis/{name}' for name in OMITTED],
                  'audit': {name: f'_migration_v1/{name}' for name in ('inventory.csv', 'verification.csv', 'migration_manifest.json')},
                  'validation': manifest['validation'], 'deployment_root': str(args.deployment)}
    write_json(args.root / 'collection_manifest.json', collection)
    # Validate write refusal without attempting a write or creating a lock.
    try:
        assert_writable(args.root / next(iter(FAMILIES)) / '.locks')
    except PermissionError:
        log('Published-root write refusal passed')
    else:
        raise ValueError('Published-root write guard did not refuse')
    final_audit = Audit(audit_root / 'final_access_verification.csv')
    normalize_shared_tree(args.root)
    try:
        access_audit(args.root, final_audit)
        final_audit.check('published_root_write_refusal', True, True)
    finally:
        final_audit.close()
    with (audit_root / 'verification.csv').open('a', newline='') as output:
        writer = csv.DictWriter(output, fieldnames=('check', 'expected', 'observed', 'passed'))
        writer.writerows(rows(audit_root / 'final_access_verification.csv'))
    manifest.update(status='complete', elapsed_seconds=time.perf_counter() - started,
                    process_peak_rss_mb=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024)
    collection['status'] = 'complete'
    write_json(audit_root / 'migration_manifest.json', manifest)
    write_json(args.root / 'collection_manifest.json', collection)
    os.chmod(audit_root / 'final_access_verification.csv', 0o640)
    log(f'COMPLETE: published {count} copied files at {args.root}')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=SHARED_PARENT / 'regression_5fold_v1')
    parser.add_argument('--stage', type=Path, default=SHARED_PARENT / '.regression_5fold_v1_migration_v1')
    parser.add_argument('--deployment', type=Path, default=SHARED_PARENT / 'openfold_interpretability_canonical')
    parser.add_argument('--home', type=Path, default=Path.home())
    parser.add_argument('--authority', type=Path, required=True)
    parser.add_argument('--replacements', type=Path, required=True)
    parser.add_argument('--execute', action='store_true', required=True)
    args = parser.parse_args()
    for name in ('root', 'stage', 'deployment', 'home', 'authority', 'replacements'):
        setattr(args, name, getattr(args, name).expanduser().resolve())
    started = time.perf_counter()
    authority = read_json(args.authority)
    if authority.get('schema') != 'canonical_migration_authority_v1' or not all(r['passed'] for r in authority['geometry_checks']):
        raise ValueError('Invalid local authority/geometry checks')
    # Recheck writers immediately before copying. The migration job itself is the only permitted job.
    jobs = subprocess.check_output(['squeue', '-h', '-u', os.environ['USER'], '-o', '%A'], text=True).split()
    if set(jobs) - {os.environ.get('SLURM_JOB_ID', '')}:
        raise ValueError(f'Active jobs must finish before migration: {jobs}')
    sources = stage(args, authority)
    audit = Audit(args.stage / '_migration_v1/verification.csv')
    try:
        count = verify_inventory(args, audit)
        log('Copied-file hashes verified; checking protocol interfaces and retained model artifacts')
        verify_tables(args.stage, args.root, authority, audit)
        verify_geometry(args.stage, args.root, authority, audit)
        for relative, checksum in read_json(args.stage / '_migration_v1/resident_snapshot.json').items():
            audit.check(f'resident_unchanged:{relative}', checksum, sha256(args.root / relative))
        for path, expected, observed, _ in verify_deployment(args.deployment, args.stage / 'metadata'):
            audit.check(f'deployment:{path}', expected, observed)
        log(f'Pre-publication verification passed: {audit.count} checks')
    finally:
        audit.close()
    publish(args, authority, sources, started, count)


if __name__ == '__main__':
    main()

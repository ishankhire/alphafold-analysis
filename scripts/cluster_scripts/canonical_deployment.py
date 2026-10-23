#!/usr/bin/env python3
"""Build the explicit, flat runtime deployment and its verifiable Git inventory."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import subprocess
import tarfile
from pathlib import Path

WORKFLOWS = ('cross_validation', 'triangle_violation_subgroups', 'publication_figures',
             'predicted_distances', 'tm_scores', 'canonical_migration')
HELPERS = ('canonical_storage.py', 'canonical_deployment.py', 'cluster_regression_helpers.py',
           'regression_output_helpers.py', 'regression_protocol.py')


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def runtime_mapping(cluster):
    mapping = {name: cluster / name for name in HELPERS}
    for workflow in WORKFLOWS:
        for path in sorted((cluster / workflow).iterdir()):
            if not path.is_file() or path.suffix not in ('.py', '.sbatch', '.json', '.md'):
                continue
            if path.name == '__init__.py' or path.name.startswith('test_'):
                continue
            name = f'{workflow}_README.md' if path.name == 'README.md' else path.name
            if path.name == 'README.md':
                name = {'cross_validation': 'fivefold_regression_README.md',
                        'predicted_distances': 'predicted_distance_export_README.md',
                        'publication_figures': 'publication_subgroup_mae_README.md'}.get(workflow, name)
            if name in mapping:
                raise ValueError(f'Flat deployment filename collision: {name}')
            mapping[name] = path
    name = 'regression_protocol_data/regression_master_split_v1.csv'
    mapping[name] = cluster / name
    return mapping


def build_package(repository, binary, output):
    repository, binary = Path(repository).resolve(), Path(binary).resolve()
    if binary.read_bytes()[:4] != b'\x7fELF':
        raise ValueError('Deployment requires the retained Linux ELF TMscore binary')
    cluster = repository / 'scripts/cluster_scripts'
    revision = subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=repository, text=True).strip()
    mapping = runtime_mapping(cluster)
    manifest = {'schema': 'canonical_deployment_v1', 'git_commit': revision, 'files': [],
                'binary_provenance': {'source': '/u/ikhire/openfold_interpretability_canonical/TMscore',
                                      'format': 'Linux ELF x86-64', 'sha256': sha256(binary)},
                'metadata_sha256': {p.name: sha256(p) for p in sorted(
                    (repository / 'delta_files/regression_5fold_v1/metadata').iterdir()) if p.is_file()}}
    mapping['TMscore'] = binary
    with tarfile.open(output, 'x') as archive:
        for name, source in sorted(mapping.items()):
            relative = str(source.relative_to(repository)) if source.is_relative_to(repository) else str(source)
            manifest['files'].append({'source': relative, 'deployed': name, 'sha256': sha256(source)})
            info = archive.gettarinfo(str(source), arcname=name)
            info.mode = 0o750 if name == 'TMscore' else 0o640
            with source.open('rb') as handle:
                archive.addfile(info, handle)
        data = (json.dumps(manifest, indent=2) + '\n').encode()
        info = tarfile.TarInfo('deployment_manifest.json')
        info.size, info.mode = len(data), 0o640
        archive.addfile(info, io.BytesIO(data))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--repository', type=Path, default=Path(__file__).resolve().parents[2])
    parser.add_argument('--linux-binary', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    manifest = build_package(args.repository, args.linux_binary, args.output)
    print(f"Packaged {len(manifest['files'])} runtime files at Git {manifest['git_commit']}")


if __name__ == '__main__':
    main()

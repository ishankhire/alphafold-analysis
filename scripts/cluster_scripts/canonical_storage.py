"""Storage resolution and write protection for the published canonical collection."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from pathlib import Path

SHARED_PARENT = Path('/work/hdd/bfna/ikhire')
RESULT_NAME = 'regression_5fold_v1'


def csv_root(value=None):
    if value is not None:
        return Path(value).expanduser().resolve()
    if os.environ.get('CSV_ROOT'):
        return Path(os.environ['CSV_ROOT']).expanduser().resolve()
    module = Path(__file__).resolve().parent
    if module.name == 'cluster_scripts' and module.parent.name == 'scripts':
        return module.parents[1] / 'delta_files'
    return SHARED_PARENT


def result_root(value=None):
    return csv_root(value) / RESULT_NAME


def workflow_root(name, environment=None, *, analysis=False):
    if environment and os.environ.get(environment):
        return Path(os.environ[environment]).expanduser().resolve()
    root = result_root()
    return root / 'analysis' / name if analysis else root / name


def assert_writable(path):
    """Refuse before locks/logs/files; resolve aliases to protect both layouts."""
    root = Path(path).expanduser().resolve()
    for ancestor in (root, *root.parents):
        if (ancestor / 'collection_manifest.json').exists():
            raise PermissionError(
                f'Published collection is protected: {ancestor}; use a new explicit output root')


def deployment_revision(directory):
    directory = Path(directory).resolve()
    try:
        return subprocess.check_output(['git', 'rev-parse', 'HEAD'], cwd=directory,
                                       stderr=subprocess.DEVNULL, text=True).strip()
    except (OSError, subprocess.CalledProcessError):
        for root in (directory, *directory.parents):
            manifest = root / 'deployment_manifest.json'
            if manifest.is_file():
                return json.loads(manifest.read_text())['git_commit']
        return ''


def normalize_shared_tree(root):
    """Explicitly repair mkstemp modes after successful shared finalization."""
    root = Path(root).resolve()
    if not root.is_relative_to(SHARED_PARENT):
        return
    import grp
    gid = grp.getgrnam('delta_bfna').gr_gid
    for directory, _, files in os.walk(root):
        path = Path(directory)
        os.chown(path, -1, gid)
        os.chmod(path, 0o2750)
        subprocess.run(['setfacl', '--set', 'u::rwx,g::r-x,m::r-x,o::---', str(path)], check=True)
        subprocess.run(['setfacl', '--default', '--set', 'u::rwx,g::r-x,m::r-x,o::---', str(path)], check=True)
        ordinary, executables = [], []
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            executable = bool(path.stat().st_mode & 0o111)
            os.chown(path, -1, gid)
            os.chmod(path, 0o750 if executable else 0o640)
            (executables if executable else ordinary).append(str(path))
        for paths, acl in ((ordinary, 'u::rw-,g::r--,m::r--,o::---'),
                           (executables, 'u::rwx,g::r-x,m::r-x,o::---')):
            for offset in range(0, len(paths), 256):
                subprocess.run(['setfacl', '--set', acl, *paths[offset:offset + 256]], check=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-write', type=Path, required=True)
    assert_writable(parser.parse_args().check_write)

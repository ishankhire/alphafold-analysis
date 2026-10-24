"""Storage resolution and write protection for the published canonical collection."""
from __future__ import annotations

import argparse
import json
import os
import stat
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
    directories, ordinary, executables = [], [], []

    def apply_acl(paths, acl, *, default=False):
        if paths:
            subprocess.run(['setfacl', *(['--default'] if default else []),
                            '--set', acl, *paths], check=True)

    for directory, _, files in os.walk(root):
        path = Path(directory)
        info = path.stat()
        if info.st_gid != gid:
            os.chown(path, -1, gid)
        if info.st_gid != gid or stat.S_IMODE(info.st_mode) != 0o2750:
            os.chmod(path, 0o2750)
        directories.append(str(path))
        if len(directories) == 256:
            apply_acl(directories, 'u::rwx,g::r-x,m::r-x,o::---')
            apply_acl(directories, 'u::rwx,g::r-x,m::r-x,o::---', default=True)
            directories.clear()
        for name in files:
            path = Path(directory) / name
            if path.is_symlink():
                continue
            info = path.stat()
            executable = bool(info.st_mode & 0o111)
            if info.st_gid != gid:
                os.chown(path, -1, gid)
            mode = 0o750 if executable else 0o640
            if stat.S_IMODE(info.st_mode) != mode:
                os.chmod(path, mode)
            paths = executables if executable else ordinary
            paths.append(str(path))
            if len(paths) == 256:
                apply_acl(paths, 'u::rwx,g::r-x,m::r-x,o::---' if executable
                          else 'u::rw-,g::r--,m::r--,o::---')
                paths.clear()
    apply_acl(directories, 'u::rwx,g::r-x,m::r-x,o::---')
    apply_acl(directories, 'u::rwx,g::r-x,m::r-x,o::---', default=True)
    apply_acl(ordinary, 'u::rw-,g::r--,m::r--,o::---')
    apply_acl(executables, 'u::rwx,g::r-x,m::r-x,o::---')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--check-write', type=Path, required=True)
    assert_writable(parser.parse_args().check_write)

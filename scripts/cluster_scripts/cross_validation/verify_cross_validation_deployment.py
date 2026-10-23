#!/usr/bin/env python3
"""Verify the flat deployment against its committed-source inventory."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from canonical_deployment import sha256
from canonical_storage import csv_root


def verify(deployment_root, metadata_root, *, hashes=True):
    root = Path(deployment_root)
    manifest = json.loads((root / 'deployment_manifest.json').read_text())
    if manifest.get('schema') != 'canonical_deployment_v1' or len(manifest['git_commit']) != 40:
        raise ValueError('Invalid deployment provenance')
    checks = []
    for record in manifest['files']:
        relative = Path(record['deployed'])
        if relative.is_absolute() or '..' in relative.parts:
            raise ValueError('Unsafe deployed path')
        path = root / relative
        observed = sha256(path) if path.is_file() and hashes else path.is_file()
        expected = record['sha256'] if hashes else True
        checks.append((str(path), expected, observed, observed == expected))
    for name, checksum in manifest['metadata_sha256'].items():
        path = Path(metadata_root) / name
        observed = sha256(path) if path.is_file() and hashes else path.is_file()
        expected = checksum if hashes else True
        checks.append((str(path), expected, observed, observed == expected))
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--deployment-root', type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument('--csv-root', type=Path, default=csv_root())
    parser.add_argument('--existence-only', action='store_true')
    args = parser.parse_args()
    checks = verify(args.deployment_root, args.csv_root / 'regression_5fold_v1/metadata',
                    hashes=not args.existence_only)
    for path, _, _, passed in checks:
        print(f"{'OK' if passed else 'FAIL'} {path}")
    return int(any(not check[3] for check in checks))


if __name__ == '__main__':
    raise SystemExit(main())

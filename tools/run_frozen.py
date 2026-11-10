"""Run preserved experiment code with an in-memory manifest path shim.

On non-Windows systems only, json.loads/json.load results containing a
top-level files_sha256 mapping receive normalized path keys. No file bytes,
hash values, data text, scores or model logic are changed. Windows is a no-op.
"""
from __future__ import annotations

import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import runpy
import sys


def normalize_manifest_paths(value, *, enabled=None):
    """Normalize only top-level files_sha256 keys; preserve all other values."""
    if enabled is None:
        enabled = os.name != 'nt'
    if not enabled or not isinstance(value, dict):
        return value
    hashes = value.get('files_sha256')
    if not isinstance(hashes, dict):
        return value
    normalized = {}
    for key, digest in hashes.items():
        if not isinstance(key, str):
            raise TypeError('Manifest path keys must be strings')
        path = key.replace('\\', '/')
        if path in normalized and normalized[path] != digest:
            raise ValueError('Manifest paths collide with different hash values')
        normalized[path] = digest
    result = value.copy()
    result['files_sha256'] = normalized
    return result


@contextmanager
def manifest_json_paths(*, enabled=None):
    """Scope the JSON shim to one script execution and restore it afterward."""
    if enabled is None:
        enabled = os.name != 'nt'
    if not enabled:
        yield
        return
    original_loads = json.loads

    def loads(*args, **kwargs):
        return normalize_manifest_paths(original_loads(*args, **kwargs), enabled=True)

    # The standard-library json.load implementation delegates to json.loads,
    # so this also covers json.load without modifying the decoder or strings.
    json.loads = loads
    try:
        yield
    finally:
        json.loads = original_loads


def run_script(script, script_args=(), *, enabled=None):
    """Match normal Python script argv/path behavior; leave cwd unchanged."""
    target = Path(script).resolve(strict=True)
    if not target.is_file():
        raise ValueError('Target must be a Python script file')
    previous_argv = sys.argv[:]
    previous_path = sys.path[:]
    sys.argv = [str(target), *script_args]
    if sys.path:
        sys.path[0] = str(target.parent)
    else:
        sys.path.append(str(target.parent))
    try:
        with manifest_json_paths(enabled=enabled):
            return runpy.run_path(str(target), run_name='__main__')
    finally:
        sys.argv = previous_argv
        sys.path[:] = previous_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('script', type=Path)
    parser.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args()
    run_script(args.script, args.script_args)


if __name__ == '__main__':
    main()

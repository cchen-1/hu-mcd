#!/usr/bin/env python3
"""Create and verify a new research archive without deleting its source.

Remote filesystem processing requires a Slurm compute allocation. Symlinks are
preserved, not followed; externally referenced datasets need their own archive.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import socket
import tarfile
from datetime import datetime, timezone


def digest(path):
    h = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(4 * 1024**2), b''):
            h.update(block)
    return h.hexdigest()


def pack(root, target, includes, excludes):
    root, target = root.resolve(), target.resolve()
    if target.is_relative_to(root):
        raise ValueError('Archive destination must be outside source root')
    if any(str(p).startswith(('/scratch/', '/QRISdata/')) for p in (root, target)):
        if not os.environ.get('SLURM_JOB_ID') or not re.fullmatch(r'bun[0-9]+', socket.gethostname().split('.')[0]):
            raise RuntimeError('Remote processing requires a Slurm compute allocation')
    for name in includes + excludes:
        p = Path(name)
        if p.is_absolute() or '..' in p.parts:
            raise ValueError('Include/exclude paths must stay relative to the root')
    includes = includes or [p.name for p in sorted(root.iterdir())]
    for name in includes:
        if not os.path.lexists(root / name):
            raise FileNotFoundError(root / name)
    paths = []
    def visit(p):
        rel = p.relative_to(root).as_posix()
        if any(rel == e or rel.startswith(e.rstrip('/') + '/') for e in excludes):
            return
        paths.append(p)
        if p.is_dir() and not p.is_symlink():
            for child in sorted(p.iterdir()):
                visit(child)
    for name in includes:
        visit(root / name)
    if len(paths) != len(set(paths)):
        raise ValueError('Overlapping include paths')
    target.parent.mkdir(parents=True, exist_ok=True)
    manifest = target.with_suffix(target.suffix + '.manifest.jsonl')
    receipt = target.with_suffix(target.suffix + '.receipt.json')
    partial = target.with_suffix(target.suffix + '.partial')
    if any(p.exists() for p in (target, manifest, receipt, partial)):
        raise FileExistsError('Refusing to replace an existing archive or attempt')
    expected = {}
    with partial.open('xb') as raw, manifest.open('x') as listing:
        with tarfile.open(fileobj=raw, mode='w:gz', compresslevel=1, dereference=False) as archive:
            for i, p in enumerate(paths):
                name = p.relative_to(root).as_posix()
                before = p.lstat()
                info = archive.gettarinfo(str(p), arcname=name)
                row = {'path': name, 'bytes': info.size, 'type': info.type.decode(), 'link': info.linkname}
                if info.isreg():
                    class Reader:
                        def __init__(self, stream):
                            self.stream, self.hash = stream, hashlib.sha256()
                        def read(self, n=-1):
                            data = self.stream.read(n)
                            self.hash.update(data)
                            return data
                    with p.open('rb') as stream:
                        reader = Reader(stream)
                        archive.addfile(info, reader)
                        row['sha256'] = reader.hash.hexdigest()
                else:
                    archive.addfile(info)
                after = p.lstat()
                if (before.st_size, before.st_mtime_ns, before.st_ino) != (after.st_size, after.st_mtime_ns, after.st_ino):
                    raise RuntimeError('Source changed during archival: ' + name)
                expected[name] = row
                listing.write(json.dumps(row) + '\n')
                if i % 10000 == 0:
                    print(json.dumps({'stage': 'packing', 'entries': i, 'total': len(paths)}), flush=True)
    # Independently reread the finished archive and check every regular-file hash.
    seen = set()
    with tarfile.open(partial, 'r|gz') as archive:
        for member in archive:
            row = expected[member.name]
            if member.name in seen or member.type.decode() != row['type'] or member.linkname != row['link']:
                raise RuntimeError('Archive member identity mismatch')
            seen.add(member.name)
            if member.isreg():
                h = hashlib.sha256()
                with archive.extractfile(member) as stream:
                    for block in iter(lambda: stream.read(4 * 1024**2), b''):
                        h.update(block)
                if h.hexdigest() != row['sha256'] or member.size != row['bytes']:
                    raise RuntimeError('Archive checksum mismatch: ' + member.name)
            if member.islnk() and member.linkname not in seen:
                raise RuntimeError('Hardlink target must precede its member')
    if seen != set(expected):
        raise RuntimeError('Archive inventory mismatch')
    result = {'status': 'VERIFIED', 'created_utc': datetime.now(timezone.utc).isoformat(),
              'source': str(root), 'archive': str(target), 'includes': includes, 'excludes': excludes,
              'entries': len(expected), 'regular_files': sum(r['type'] == '0' for r in expected.values()),
              'symlinks': [r for r in expected.values() if r['type'] == '2'],
              'sha256': digest(partial), 'bytes': partial.stat().st_size,
              'manifest_sha256': digest(manifest), 'source_retained': True,
              'job_id': os.environ.get('SLURM_JOB_ID'), 'host': socket.gethostname()}
    # Hard-link publication is exclusive: an existing destination cannot be replaced.
    os.link(partial, target)
    partial.unlink()
    with receipt.open('x') as stream:
        json.dump(result, stream, indent=2)
        stream.write('\n')
    print(json.dumps(result), flush=True)
    return result


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, required=True)
    p.add_argument('--target', type=Path, required=True)
    p.add_argument('--include', action='append', default=[])
    p.add_argument('--exclude', action='append', default=[])
    a = p.parse_args()
    pack(a.root, a.target, a.include, a.exclude)

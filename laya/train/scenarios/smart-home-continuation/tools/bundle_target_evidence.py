"""Bundle raw experiment evidence deterministically; initialization weights stay separate."""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import tarfile
from pathlib import Path


def bundle(root: Path):
    archive = root / 'evidence.tar.gz'
    if archive.exists():
        raise FileExistsError(f'refusing to replace evidence archive: {archive}')
    files = []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if not path.is_file() or path.is_symlink():
            continue
        if rel.parts[0] == 'initialization' or '__pycache__' in rel.parts:
            continue
        if len(rel.parts) == 1 and (path.suffix == '.md' or path.name in {
            '.gitignore', 'evidence.tar.gz', 'evidence.sha256', 'evidence-files-sha256.json'
        }):
            continue
        files.append(path)
    hashes = {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    manifest = root / 'evidence-files-sha256.json'
    manifest.write_text(json.dumps(hashes, indent=2)+'\n')
    files.append(manifest)
    with archive.open('xb') as raw, gzip.GzipFile(filename='', mode='wb', fileobj=raw, mtime=0) as zipped:
        with tarfile.open(fileobj=zipped, mode='w') as tar:
            for path in sorted(files):
                info = tar.gettarinfo(str(path), arcname=str(path.relative_to(root)))
                info.uid = info.gid = info.mtime = 0
                info.uname = info.gname = ''
                info.mode = 0o644
                with path.open('rb') as stream:
                    tar.addfile(info, stream)
    digest = hashlib.sha256(archive.read_bytes()).hexdigest()
    (root / 'evidence.sha256').write_text(f'{digest}  evidence.tar.gz\n')
    print(json.dumps({'files': len(files), 'bytes': archive.stat().st_size, 'sha256': digest}))


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('root', type=Path)
    bundle(ap.parse_args().root)

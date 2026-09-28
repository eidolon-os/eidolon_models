"""Derived JSONL slice views; never edit the frozen training inputs."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from eidolon_laya_train.records import read_jsonl,write_jsonl
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
root=a.dataset/'slices'
if (root/'manifest.json').exists():raise ValueError('slice manifest already exists')
rows=defaultdict(list)
for split in ('train','val'):
 for r in read_jsonl(a.dataset/f'{split}.jsonl'):rows[(r.meta['slice'],split)].append(r)
manifest={'role':'derived views only; source train/val remain unchanged','files':{}}
for sid in sorted({sid for sid,split in rows}):
 d=root/sid;d.mkdir(exist_ok=True,parents=True)
 for split in ('train','val'):
  f=d/f'{split}.jsonl'
  if f.exists():raise ValueError('refusing overwrite')
  write_jsonl(f,rows[(sid,split)])
  manifest['files'][str(f.relative_to(root))]={'sha256':hashlib.sha256(f.read_bytes()).hexdigest(),'records':len(rows[(sid,split)]),'families':len({r.meta['family'] for r in rows[(sid,split)]})}
(root/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print('slice views',len(manifest['files']))

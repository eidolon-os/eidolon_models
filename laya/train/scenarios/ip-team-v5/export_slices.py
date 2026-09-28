"""Derive per-slice views without changing the frozen train/dev inputs."""
import argparse
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from eidolon_laya_train.records import read_jsonl, write_jsonl
p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);a=p.parse_args()
out=a.dataset/'slices'
if out.exists():raise ValueError('refusing to overwrite slice export')
out.mkdir()
rows=defaultdict(list)
for split in ('train','val'):
 for row in read_jsonl(a.dataset/f'{split}.jsonl'):rows[(row.meta['slice'],split)].append(row)
families=json.loads((a.dataset/'reviewed-families.json').read_text())
manifest={}
for sid in sorted({r['slice'] for r in families}):
 d=out/sid;d.mkdir()
 (d/'families.json').write_text(json.dumps([r for r in families if r['slice']==sid],ensure_ascii=False,indent=2))
 for split in ('train','val'):
  write_jsonl(d/f'{split}.jsonl',rows[(sid,split)])
 for f in sorted(d.glob('*.json*')):
  manifest[str(f.relative_to(out))]=hashlib.sha256(f.read_bytes()).hexdigest()
(out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2))
print({'slices':len({r['slice'] for r in families}),'files':len(manifest)})

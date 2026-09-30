"""Read existing local artifacts only; no inference or external requests."""
import json,hashlib
from collections import Counter
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
report={}
for split in ('train','dev'):
 p=ROOT/f'evals/participation/data/companion-ip-v1/{split}.jsonl'
 rows=[json.loads(x) for x in p.read_text().splitlines() if x.strip()]
 snapshots={r['meta']['snapshot']:r for r in rows}
 counts=Counter()
 for r in snapshots.values():
  s=r['state']; q=r['questions']['move']['criteria'];t=s['trigger']
  counts['snapshots']+=1
  counts['clarify_available']+=any(k=='clarify' or k.startswith('clarify:') for k in q)
  counts['clarify_gold']+=any(k=='clarify' or k.startswith('clarify:') for k in r['labels']['move']['gold'])
  counts['companion_trigger']+=t.get('author','').startswith('M')
  counts['user_trigger']+=t.get('same_as_user_request',False) or t.get('author')=='user'
  counts['companion_task']+=r['meta']['task']=='companion'
 report[split]={'sha256':hashlib.sha256(p.read_bytes()).hexdigest(),'records':len(rows),'unique_snapshot_counts':dict(counts)}
report['limitations']=['Historical text projection explicitly replaces uncertainty/clarification with abstention; not complete product contract.', 'Counts do not establish multi-step rollout correctness or production distribution.', 'Existing DEV reused for selection; not final generalization evidence.']
(HERE/'coverage.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps(report,ensure_ascii=False,indent=2))

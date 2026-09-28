"""Write metadata-only production ledger, without credentials or raw prompts."""
import hashlib
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
RAW=HERE.parents[1]/'private/ip-team-v8-boundaries'
calls=[dict(file=p.name,**json.loads(p.read_text())) for p in sorted((RAW/'calls').glob('*.json'))]
audit=json.loads((HERE/'audit-decisions.json').read_text())
rows=[json.loads(p.read_text()) for p in sorted((RAW/'reviewed').glob('*.json'))]
accepted=[r for r in rows if audit.get(r['id'],{}).get('decision')=='accept']
ledger={'limit':128,'calls_made':len(calls),'call_statuses':dict(Counter(c['status'] for c in calls)),
        'known_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),
        'families_generated':len(rows),'families_accepted':dict(Counter(r['split'] for r in accepted)),
        'pending_audit':sum(r['id'] not in audit for r in rows),'decisions':dict(Counter(a['decision'] for a in audit.values())),
        'audit_sha256':hashlib.sha256((HERE/'audit-decisions.json').read_bytes()).hexdigest(),'calls':calls}
(HERE/'generation-ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in ledger.items() if k!='calls'},ensure_ascii=False))

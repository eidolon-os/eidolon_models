"""Compact read-only review queue and budget ledger; never assigns labels."""
import json
from pathlib import Path
from collections import Counter
HERE=Path(__file__).resolve().parent
root=HERE.parents[1]/'private/ip-team-v6-pairs'
rows=[json.loads(p.read_text()) for p in sorted((root/'reviewed').glob('*.json'))]
for r in rows:
 if r['status']=='agreed_pending_audit':
  print(json.dumps({'id':r['id'],'pair':r['pair'],'review':r['review']},ensure_ascii=False))
 else:print(json.dumps({'id':r['id'],'status':r['status'],'reason':r.get('reason')},ensure_ascii=False))
calls=[dict(file=str(p),**json.loads(p.read_text())) for p in sorted((root/'calls').glob('*.json'))]
ledger={'max_paid_calls':32,'calls_made':len(calls),'pending':sum(c['status']=='pending' for c in calls),'failed':sum(c['status']=='failed' for c in calls),'usage_missing_calls':sum(not c.get('usage') for c in calls),'total_tokens':sum((c.get('usage') or {}).get('total_tokens',0) for c in calls),'statuses':dict(Counter(r['status'] for r in rows)),'calls':calls}
(HERE/'generation-ledger.json').write_text(json.dumps(ledger,ensure_ascii=False,indent=2))
print(json.dumps({k:v for k,v in ledger.items() if k!='calls'},ensure_ascii=False))

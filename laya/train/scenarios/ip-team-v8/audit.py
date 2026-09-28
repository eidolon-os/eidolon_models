"""Read-only display of unaudited synthetic families and production accounting."""
import json
from collections import Counter
from pathlib import Path

HERE=Path(__file__).resolve().parent
RAW=HERE.parents[1]/'private/ip-team-v8-boundaries'
audit=json.loads((HERE/'audit-decisions.json').read_text()) if (HERE/'audit-decisions.json').exists() else {}
rows=[json.loads(p.read_text()) for p in sorted((RAW/'reviewed').glob('*.json'))]
for r in rows:
    if r['id'] in audit:continue
    print('\n'+r['id']+' '+r['status'])
    if 'scenes' not in r:
        print(r.get('reason'));continue
    print('CAST '+json.dumps(r['scenes'][0]['cast'],ensure_ascii=False))
    for d in r['scenes']:print(d['id'],d['user'],d.get('gold'),d.get('why'))
print('accepted_by_split',dict(Counter(r['split'] for r in rows if audit.get(r['id'],{}).get('decision')=='accept')))
print('pending_audit',sum(r['id'] not in audit for r in rows),'calls',len(list((RAW/'calls').glob('*.json'))))

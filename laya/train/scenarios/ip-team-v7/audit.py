"""Render the audit queue without any model predictions."""
import argparse,json,re
from pathlib import Path
from collections import Counter
p=argparse.ArgumentParser();p.add_argument('pattern',nargs='?',default='.*');a=p.parse_args()
HERE=Path(__file__).resolve().parent;root=HERE.parents[1]/'private/ip-team-v7-generalization'
rows=[]
for path in sorted((root/'reviewed').glob('*.json')):
 rows+=json.loads(path.read_text())['results']
for r in rows:
 if not re.search(a.pattern,r['id']):continue
 if r['status'] in ('annotated_pending_audit','agreed_pending_audit'):
  print(json.dumps({'id':r['id'],'scene':r['scene']},ensure_ascii=False))
 else:print(json.dumps({'id':r['id'],'status':r['status'],'reason':r.get('reason')},ensure_ascii=False))
print('totals',dict(Counter(r['status'] for r in rows)))

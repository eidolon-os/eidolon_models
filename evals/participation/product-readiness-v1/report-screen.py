import hashlib,json,math,statistics
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent;RUN=ROOT/'jevk5/runs/product-readiness-v1'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def latency(rs):
 a=sorted(r['elapsed_ms'] for r in rs)
 return {'n':len(a),'p50_ms':statistics.median(a),'p95_ms':a[math.ceil(.95*len(a))-1]}
report={};cases={r['id']:r for r in read(HERE/'cases.jsonl')}
for alias in ('decider','jevk5'):
 rs=read(RUN/(alias+'.jsonl'));cfg=json.loads((RUN/(alias+'.config.json')).read_text())
 assert len(rs)==48 and len({(r['id'],r['reverse']) for r in rs})==48
 assert cfg['cases_sha256']==sha(HERE/'cases.jsonl') and cfg['script_sha256']==sha(HERE/'screen.py')
 assert cfg['design_sha256']==sha(HERE/'SCREEN-DESIGN.md') and cfg['build_sha256']==sha(HERE/'build.py')
 for r in rs:
  c=cases[r['id']];assert r['gold']==c['labels']['move']['gold']
  assert r['correct']==(r['pred'] in r['gold'])
  assert r['pred']==max(r['probabilities'],key=r['probabilities'].get)
 variants={}
 for rev in (False,True):
  x=[r for r in rs if r['reverse']==rev]
  speech=sum(r['forbid_speech'] and r['pred'].split(':')[0] in ('respond','clarify') for r in x)
  groups={}
  for action in ('respond','clarify','wait','finish','abstain'):
   a=[r for r in x if all(g.split(':')[0]==action for g in r['gold'])]
   groups[action]={'n':len(a),'correct':sum(r['correct'] for r in a)}
  passed=sum(r['correct'] for r in x)>=22 and speech==0 and all(groups[k]['correct']==groups[k]['n'] for k in ('clarify','abstain'))
  variants[str(rev)]={'correct':sum(r['correct'] for r in x),'n':24,'wrong_speech':speech,'groups':groups,'passed':passed,
    'tasks':{t:{'n':12,'correct':sum(r['correct'] for r in x if r['task']==t)} for t in ('companion','ip_team')}}
 families={f:[r for r in rs if r['family']==f] for f in {r['family'] for r in rs}}
 report[alias]={'variants':variants,'screen_passed':all(x['passed'] for x in variants.values()),
  'same_answer':sum(next(r for r in rs if r['id']==cid and not r['reverse'])['pred']==next(r for r in rs if r['id']==cid and r['reverse'])['pred'] for cid in cases),
  'both_orders_correct':sum(all(r['correct'] for r in rs if r['id']==cid) for cid in cases),
  'all_family_states_and_orders_correct':sum(all(r['correct'] for r in x) for x in families.values()),
  'warm_latency':latency([r for r in rs if not r['first_call']]),'first_call_ms':rs[0]['elapsed_ms'],
  'max_observed_driver_bytes':max(r['mps_driver_allocated_bytes'] for r in rs),
  'failures':[{'id':r['id'],'reverse':r['reverse'],'gold':r['gold'],'pred':r['pred']} for r in rs if not r['correct']],
  'input_tokens_range':[min(r['tokens'] for r in rs),max(r['tokens'] for r in rs)],'run_config':cfg}
report['limitations']=['Authored 24-snapshot diagnostic, not held-out generalization.', 'Scripted public replies, not live rollout.', 'Measured local tokenize/infer/map, not HTTP/network/cold-load latency.', 'Production hardware and latency target remain unresolved.', 'Clarification task mapped to generic instruction; real reply chain not tested.']
(HERE/'screen-results.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
print(json.dumps({a:{k:v for k,v in report[a].items() if k not in ('run_config','failures')} for a in ('decider','jevk5')},ensure_ascii=False,indent=2))

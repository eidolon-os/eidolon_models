import collections,hashlib,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'jevk5/runs/d3-prompt-ab-20261019'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def group(gold):
 a={x.split(':')[0] for x in gold}
 return next(iter(a)) if len(a)==1 else 'mixed'
def timing(xs):
 a=sorted(x['elapsed_ms'] for x in xs)
 return {'n':len(a),'p50_ms':statistics.median(a),'p95_ms':a[math.ceil(.95*len(a))-1]} if a else None
cfg=json.loads((OUT/'config.json').read_text());data=json.loads((HERE/'cases.json').read_text())
assert all(sha(ROOT/p)==h for p,h in cfg['hashes'].items())
rs=read(OUT/'static.jsonl');loops=read(OUT/'rollouts.jsonl');calls=read(OUT/'calls.jsonl');attempts=read(OUT/'attempts.jsonl')
assert len(rs)==112 and len(loops)==6 and len(calls)==len(attempts)
assert len({(r['policy'],r['id']) for r in rs})==112
bycase={r['id']:r for r in data['cases']};byresult={(r['policy'],r['id']):r for r in rs}
bycall={x['tag']:x for x in calls}
for r in data['cases']:
 b=bycall['baseline|static:'+r['id']]['request'];e=bycall['enhanced|static:'+r['id']]['request']
 assert b['state']==e['state']==json.dumps(r['state'],ensure_ascii=False)
 assert e['questions']['move']['instructions']==b['questions']['move']['instructions']+data['addition']
 assert list(e['questions']['move']['criteria'].items())==list(b['questions']['move']['criteria'].items())
 assert b['questions']==r['questions']
 for p in ('baseline','enhanced'):
  x=byresult[p,r['id']];assert x['gold']==r['gold']
  if 'pred' in x:assert x['correct']==(x['pred'] in x['gold'])
summary={'policies':{},'paired_changes':[],'gates':{},'actual_calls':dict(collections.Counter(x['kind'] for x in calls)),
 'failed_calls':sum(not x['ok'] for x in calls)}
for p in ('baseline','enhanced'):
 per={}
 for s in ('old_diagnostic','new_diagnostic'):
  xs=[r for r in rs if r['policy']==p and r['suite']==s]
  gs={}
  for g in ('respond','clarify','wait','finish','abstain'):
   ys=[r for r in xs if group(r['gold'])==g]
   if ys:gs[g]={'n':len(ys),'correct':sum(r['correct'] for r in ys)}
  fs=collections.defaultdict(list)
  for r in xs:fs[r['family']].append(r)
  per[s]={'n':len(xs),'correct':sum(r['correct'] for r in xs),'groups':gs,
          'all_variants_correct':sum(all(r['correct'] for r in ys) for ys in fs.values()),'contexts':len(fs)}
 staticcalls=[c for c in calls if c['kind']=='d3' and c['tag'].startswith(p+'|static:')]
 tokens=[c.get('response',{}).get('usage',{}).get('input_tokens') for c in staticcalls]
 tokens=[t for t in tokens if isinstance(t,(int,float))]
 per['static_timing']=timing(staticcalls);per['static_input_tokens']={'n':len(tokens),'mean':statistics.mean(tokens) if tokens else None,'total':sum(tokens)}
 per['episodes']=[{'id':r['id'],'passed':r['passed'],'terminals':[t['terminal'] for t in r.get('turns',[])]} for r in loops if r['policy']==p]
 summary['policies'][p]=per
for r in data['cases']:
 b=byresult['baseline',r['id']];e=byresult['enhanced',r['id']]
 if b.get('pred')!=e.get('pred'):
  summary['paired_changes'].append({'id':r['id'],'suite':r['suite'],'gold':r['gold'],'baseline':b.get('pred'),'enhanced':e.get('pred'),
   'fixed':not b['correct'] and e['correct'],'regressed':b['correct'] and not e['correct']})
b=summary['policies']['baseline'];e=summary['policies']['enhanced']
improve=all(e[s]['correct']>b[s]['correct'] for s in ('old_diagnostic','new_diagnostic'))
no_group_loss=all(e['new_diagnostic']['groups'][g]['correct']>=b['new_diagnostic']['groups'][g]['correct'] for g in ('respond','wait','finish','abstain'))
loop_pass=all(x['passed'] for x in e['episodes'])
summary['gates']={'both_suites_improve':improve,'new_regression_groups_no_net_loss':no_group_loss,'enhanced_all_three_episodes':loop_pass,
 'automated_keep_signal':improve and no_group_loss and loop_pass,'manual_content_review_required':True}
(HERE/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(HERE/'VALIDATION.json').write_text(json.dumps({'passed':True,'meaning':'experiment integrity, not product acceptance','paired_static_cases':56,
 'actual_calls':summary['actual_calls'],'failed_calls':summary['failed_calls'],'only_appended_instructions_changed':True,
 'raw_sha256':sha(OUT/'calls.jsonl'),'hashes_verified':True},indent=2))
lines=['# A/B真实回复文本闭环','','两策略独立生成回复；用户后续输入为固定脚本。非生产Agent或设备验收。','']
for r in loops:
 lines+=['## '+r['id']+' · '+r['policy'],'',f"动作/人选/终态：{r['passed']}",'']
 for t in r.get('turns',[]):
  lines+=['用户：'+t['expected']['user'],'']
  for x in t['events']:lines.append('- '+x['decision']+('：'+x['text'] if 'text' in x else ''))
  lines+=['','终态：'+t['terminal'],'']
(HERE/'TRANSCRIPTS.md').write_text('\n'.join(lines))
print(json.dumps(summary,ensure_ascii=False,indent=2))

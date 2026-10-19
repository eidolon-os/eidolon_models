import collections,hashlib,json,math,statistics
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];OUT=ROOT/'jevk5/runs/d3-followup-20261019'
def read(p):return [json.loads(s) for s in p.read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def timing(xs):
 v=sorted(x['elapsed_ms'] for x in xs)
 return {'n':len(v),'p50_ms':statistics.median(v),'p95_ms':v[math.ceil(.95*len(v))-1]} if v else None
cfg=json.loads((OUT/'config.json').read_text());static=read(OUT/'static.jsonl');loops=read(OUT/'rollouts.jsonl');calls=read(OUT/'calls.jsonl');attempts=read(OUT/'attempts.jsonl')
for field,path in [('cases_sha256','cases.json'),('script_sha256','run.py'),('build_sha256','build.py'),('design_sha256','DESIGN.md')]:assert cfg[field]==sha(HERE/path)
assert len(attempts)==len(calls) and len(static)==32 and len(loops)==12
counts=collections.Counter(c['kind'] for c in calls)
assert counts['d3']<=80 and counts['reply']<=24
case_map={r['id']:r for r in json.loads((HERE/'cases.json').read_text())['static']}
for r in static:
 src=case_map[r['id']]
 if 'pred' in r:assert r['correct']==(r['pred'] in src['gold'])
for c in calls:
 if c['kind']=='d3':
  assert set(c['request'])=={'model','state','questions'}
  assert set(json.loads(c['request']['state']))=={'user_request','scene_goal','candidates','allowed_actions','remaining_replies','prior_public_messages','trigger'}
 else:
  assert set(json.loads(c['request']['messages'][1]['content']))=={'speaker_id','character','public_state','task'}
family={}
for fid in {r['family'] for r in case_map.values()}:
 rs=[r for r in static if case_map[r['id']]['family']==fid]
 family[fid]={'correct':sum(r['correct'] for r in rs),'n':len(rs),'predictions':{case_map[r['id']]['variant']:r.get('pred') for r in rs}}
summary={'static_correct':sum(r['correct'] for r in static),'static_total':32,'families':family,
 'all_variants_correct_contexts':sum(x['correct']==x['n'] for x in family.values()),
 'episodes_passed':sum(r['passed'] for r in loops),'episodes_total':12,
 'episode_results':[{'id':r['id'],'reverse':r['reverse'],'passed':r['passed'],'terminals':[t['terminal'] for t in r.get('turns',[])]} for r in loops],
 'calls':dict(counts),'failed_calls':sum(not r['ok'] for r in calls),
 'timing':{k:timing([c for c in calls if c['kind']==k]) for k in ('d3','reply')},
 'all_gates_passed':all(r['correct'] for r in static) and all(r['passed'] for r in loops),
 'reply_finish_reasons':dict(collections.Counter(c['response']['choices'][0]['finish_reason'] for c in calls if c['kind']=='reply' and c['ok']))}
(HERE/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
(HERE/'VALIDATION.json').write_text(json.dumps({'passed':True,'meaning':'audit integrity only, not model acceptance',
 'static_records':32,'episode_records':12,'attempts':len(attempts),'calls':dict(counts),'retry':0,
 'input_output_hashes_verified':True,'public_payload_fields_verified':True,'raw_calls_sha256':sha(OUT/'calls.jsonl'),
 'rollouts_sha256':sha(OUT/'rollouts.jsonl')},indent=2))
lines=['# 真实生成文本闭环逐轮记录','','虚构角色/用户脚本，真实deepseek-v4-flash回复，真实d3决策；非真机或生产Agent链路。','']
for r in loops:
 lines+=['## '+r['id']+' · '+('逆序' if r['reverse'] else '原序'),'',f"自动动作/角色/终态检查：{r['passed']}",'']
 for turn in r.get('turns',[]):
  lines+=['用户：'+turn['expected']['user'],'']
  for e in turn['events']:
   lines+=['- 决策：'+e['decision']+('；实际回复：'+e['text'] if 'text' in e else '')]
  lines+=['', '终态：'+turn['terminal'],'']
(HERE/'TRANSCRIPTS.md').write_text('\n'.join(lines))
print(json.dumps(summary,ensure_ascii=False,indent=2))

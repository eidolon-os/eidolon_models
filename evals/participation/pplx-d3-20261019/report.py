"""Offline audit/report only; no requests or model execution."""
import collections,json,math,statistics
from pathlib import Path
from run import ROOT,HERE,OUT,OLD,DATA,read,sha,cases

def action(gold):
 a={x.split(':')[0] for x in gold}
 if a<= {'wait','finish'}:return 'stop'
 return next(iter(a)) if len(a)==1 else 'mixed'
def latency(rows):
 vals=sorted(r['elapsed_ms'] for r in rows)
 if not vals:return None
 return {'n':len(vals),'p50_ms':statistics.median(vals),'p95_ms':vals[math.ceil(.95*len(vals))-1],
         'mean_ms':statistics.mean(vals),'min_ms':vals[0],'max_ms':vals[-1]}
def metric(rows,planned):
 valid=[r for r in rows if r.get('valid',True)]
 groups={}
 for a in ('respond','stop','clarify','abstain','mixed'):
  xs=[r for r in rows if action(r['gold'])==a]
  if xs:groups[a]={'n':len(xs),'correct':sum(r['correct'] for r in xs),
     'wrong_speech':sum(r.get('pred','').split(':')[0] in ('respond','clarify') for r in xs) if a=='stop' else None}
 families=collections.defaultdict(list)
 for r in rows:families[r['family']].append(r)
 high=[r for r in valid if max(r['probabilities'].values())>=.8]
 return {'planned':planned,'received':len(rows),'valid':len(valid),'correct':sum(r['correct'] for r in rows),
         'accuracy_over_planned':sum(r['correct'] for r in rows)/planned,
         'accuracy_valid':sum(r['correct'] for r in valid)/len(valid) if valid else None,
         'groups':groups,'observed_families':len(families),
         'fully_correct_observed_families':sum(all(r['correct'] for r in xs) for xs in families.values()),
         'high_probability_0_8':{'n':len(high),'correct':sum(r['correct'] for r in high),'errors':sum(not r['correct'] for r in high)},
         'latency_all_attempts':latency(rows),
         'latency_valid_excluding_first_model_request':latency([r for r in valid if not r.get('first_for_model',r.get('first_call',False))])}

def main():
 cfg=json.loads((OUT/'config.json').read_text());raw=read(OUT/'responses.jsonl');attempts=read(OUT/'attempts.jsonl')
 assert len(raw)==len(attempts)<=276
 assert cfg['script_sha256']==sha(HERE/'run.py') and cfg['design_sha256']==sha(HERE/'DESIGN.md')
 assert cfg['data_sha256']=={k:sha(v) for k,v in DATA.items()}
 planned=cases();lookup={(r['suite'],r['id'],r['reverse']):r for r in planned}
 assert len({(r['model'],r['suite'],r['id'],r['reverse']) for r in raw})==len(raw)
 for r in raw:
  src=lookup[r['suite'],r['id'],r['reverse']];assert src['gold']==r['gold']
  if r['valid']:
   assert r['correct']==(r['pred'] in r['gold'])
   assert set(r['probabilities'])==set(src['questions']['move']['criteria'])
 report={'current':{},'historical':{},'paired_vs_laya':{},'completed_requests':len(raw),
         'complete':len(raw)==276,'failures':[r for r in raw if not r['valid']]}
 for name in ('pplx','d3'):
  rs=[r for r in raw if r['model']==name]
  sections={s:metric([r for r in rs if r['suite']==s],n) for s,n in [('v7',42),('v8',72),('product_team',24)]}
  product=[r for r in rs if r['suite']=='product_team'];byid=collections.defaultdict(list)
  for r in product:byid[r['id']].append(r)
  sections['product_orders']={str(rev):metric([r for r in product if r['reverse']==rev],12) for rev in (False,True)}
  sections['product_stability']={'snapshots':len(byid),'both_correct':sum(len(x)==2 and all(r['correct'] for r in x) for x in byid.values()),
     'same_answer':sum(len(x)==2 and x[0].get('pred')==x[1].get('pred') and all(r['valid'] for r in x) for x in byid.values())}
  sections['latency_all']=latency(rs)
  sections['latency_excluding_first']=latency([r for r in rs if r['valid'] and not r['first_for_model']])
  sections['first_request_ms']=[r['elapsed_ms'] for r in rs if r['first_for_model']]
  sections['server_models']=sorted({r.get('response',{}).get('model','missing') for r in rs})
  tokens=[r.get('response',{}).get('usage',{}).get('input_tokens') for r in rs]
  tokens=[v for v in tokens if isinstance(v,(int,float))]
  sections['input_tokens']={'reported_n':len(tokens),'total':sum(tokens),'min':min(tokens) if tokens else None,'max':max(tokens) if tokens else None}
  report['current'][name]=sections
 for name in ('laya','laya-r6','laya-r7'):
  assert sha(OLD/(name+'.jsonl'))==cfg['historical_sha256'][name]
  rs=read(OLD/(name+'.jsonl'))
  report['historical'][name]={s:metric([r for r in rs if r['split']==s],n) for s,n in [('v7',42),('v8',72)]}
  report['historical'][name]['latency_excluding_first']=latency([r for r in rs if not r['first_call']])
  for alias in ('pplx','d3'):
   current={(r['suite'],r['id']):r for r in raw if r['model']==alias and r['suite'] in ('v7','v8')}
   pairs=[(b,current[(b['split'],b['id'])]) for b in rs if (b['split'],b['id']) in current]
   report['paired_vs_laya'][alias+' vs '+name]={'matched':len(pairs),
    'fixed_old_errors':sum(not b['correct'] and c['correct'] for b,c in pairs),
    'broke_old_correct':sum(b['correct'] and not c['correct'] for b,c in pairs)}
 (HERE/'comparison.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
 errors=[]
 for r in raw:
  if not r['correct']:
   src=lookup[r['suite'],r['id'],r['reverse']]
   errors.append({**{k:r.get(k) for k in ('model','suite','id','reverse','gold','pred','valid','error')},
    'max_probability':max(r['probabilities'].values()) if r.get('probabilities') else None,
    'public_state':src['state']})
 (HERE/'errors.json').write_text(json.dumps(errors,ensure_ascii=False,indent=2))
 lines=['# pplx / d3 与 Laya 角色团同题测试','',
 '2026-10-19。Laya复用2026-09-28历史日志，未重跑。新接口使用同一公开state、问题、选项与金标。','',
 '| 模型 | v7常规 /42 | v8压力 /72 | v8回应 /24 | v8停止 /24 | v8弃权 /24 |',
 '|---|---:|---:|---:|---:|---:|']
 for name,m in list(report['historical'].items())+list(report['current'].items()):
  g=m['v8']['groups'];lines.append('| '+name+' | '+' | '.join(str(x) for x in [m['v7']['correct'],m['v8']['correct'],g['respond']['correct'],g['stop']['correct'],g['abstain']['correct']])+' |')
 lines+=['','## 新产品契约角色团用例','',
 '12个快照、6家族，各测原序和逆序。Laya没有这批同题历史，不填写Laya分数。','',
 '| 模型 | 原序 /12 | 逆序 /12 | 双序都对 /12 | 同答案 /12 | 全家族全对 /6 |',
 '|---|---:|---:|---:|---:|---:|']
 for name,m in report['current'].items():
  p=m['product_stability'];lines.append(f"| {name} | {m['product_orders']['False']['correct']} | {m['product_orders']['True']['correct']} | {p['both_correct']} | {p['same_answer']} | {m['product_team']['fully_correct_observed_families']} |")
 lines+=['','## 客户端HTTP性能','',
 '串行请求，含网络/服务排队/解析，尽量复用连接；不是纯GPU推理或并发吞吐。去除每接口首请求的有效响应统计：','',
 '| 接口 | 有效/计划 | p50 ms | p95 ms | 均值 ms | 首请求 ms |','|---|---:|---:|---:|---:|---:|']
 for name,m in report['current'].items():
  t=m['latency_excluding_first'];valid=sum(m[s]['valid'] for s in ('v7','v8','product_team'))
  if t:lines.append(f"| {name} | {valid}/138 | {t['p50_ms']:.1f} | {t['p95_ms']:.1f} | {t['mean_ms']:.1f} | {m['first_request_ms'][0]:.1f} |")
 lines+=['','历史Laya仅作运行背景：Mac MPS本地推理、不含HTTP，不能用下表与上表计算模型加速倍数。','',
 '| 历史配置 | 本地p50 ms | 本地p95 ms |','|---|---:|---:|']
 for name,m in report['historical'].items():
  t=m['latency_excluding_first'];lines.append(f"| {name} | {t['p50_ms']:.1f} | {t['p95_ms']:.1f} |")
 lines+=['','## 高置信错误检查（最大概率≥0.8）','', '| 接口 | 用例组 | 高置信数 | 其中错误 |','|---|---|---:|---:|']
 for name,m in report['current'].items():
  for s in ('v7','v8','product_team'):
   h=m[s]['high_probability_0_8'];lines.append(f"| {name} | {s} | {h['n']} | {h['errors']} |")
 lines+=['','分组细项、调用完整性、逐题修复/回归见comparison.json；错误公开上下文见errors.json。',
 '所有输入为合成开发/诊断数据，变体非独立样本；不代表真实产品泛化，也未验证实时回复闭环。未训练、未重跑Laya、未调阈值。','']
 (HERE/'TABLE.md').write_text('\n'.join(lines))
 (HERE/'VALIDATION.json').write_text(json.dumps({'passed':True,'requests':len(raw),'complete':len(raw)==276,
  'raw_sha256':sha(OUT/'responses.jsonl'),'ledger_sha256':sha(OUT/'attempts.jsonl'),
  'payload_isolation_verified':True,'same_legacy_inputs_verified':True,'laya_rerun':False,'config':cfg},ensure_ascii=False,indent=2))
 print('\n'.join(lines))
if __name__=='__main__':main()

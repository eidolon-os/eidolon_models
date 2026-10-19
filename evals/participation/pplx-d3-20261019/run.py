"""Bounded user-requested HTTP comparison on frozen synthetic public cases."""
import copy,hashlib,http.client,json,math,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3];HERE=Path(__file__).resolve().parent
OUT=ROOT/'jevk5/runs/pplx-d3-20261019'
HOST='decider.yangtzeailab.com'
MODELS={'pplx':'pplx-decider-v1.1-27b','d3':'d3'}
DATA={'v7':ROOT/'laya/train/data/generated/ip-team-v7/generalization1/named/val.jsonl',
      'v8':ROOT/'laya/train/data/generated/ip-team-v8/boundaries1/named/val.jsonl'}
OLD=ROOT/'laya/train/runs/backbone-comparison-v1'
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def cases():
 result=[]
 pre=json.loads((OLD/'preflight-v2.json').read_text())
 for suite,path in DATA.items():
  assert sha(path)==pre['data_sha256'][suite]
  for r in read(path):
   result.append({'suite':suite,'id':r['id'],'family':r['meta']['family'],'reverse':False,
    'gold':r['labels']['move']['gold'],'state':r['state'],'questions':r['questions']})
 for r in read(ROOT/'evals/participation/product-readiness-v1/cases.jsonl'):
  if r['meta']['task']!='ip_team':continue
  for rev in (False,True):
   q=copy.deepcopy(r['questions'])
   if rev:q['move']['criteria']=dict(reversed(list(q['move']['criteria'].items())))
   result.append({'suite':'product_team','id':r['id'],'family':r['meta']['family'],'reverse':rev,
    'gold':r['labels']['move']['gold'],'state':r['state'],'questions':q})
 assert len(result)==138
 for alias in ('laya','laya-r6','laya-r7'):
  historical=read(OLD/(alias+'.jsonl'));assert len(historical)==114
  for r,b in zip(result[:114],historical):
   assert (r['suite'],r['id'],r['gold'])==(b['split'],b['id'],b['gold'])
 return result

def payload(r,alias):
 return {'model':MODELS[alias],'state':json.dumps(r['state'],ensure_ascii=False),
         'questions':{'move':copy.deepcopy(r['questions']['move'])}}

def main():
 rs=cases();OUT.mkdir(parents=True,exist_ok=False)
 cfg={'max_requests':276,'retries':0,'timeout_seconds':30,'concurrency':1,
      'endpoint_host':HOST,'models':MODELS,'script_sha256':sha(__file__),'design_sha256':sha(HERE/'DESIGN.md'),
      'data_sha256':{k:sha(v) for k,v in DATA.items()},
      'product_sha256':sha(ROOT/'evals/participation/product-readiness-v1/cases.jsonl'),
      'api_doc_sha256':sha(Path('/Users/manson/ai/decider/delta_router_eval/docs/decider-api-usage.md')),
      'historical_sha256':{a:sha(OLD/(a+'.jsonl')) for a in ('laya','laya-r6','laya-r7')},
      'timing':'client json encode + HTTP request/read + JSON validation, persistent connections where server permits; includes network',
      'model_revision':'server-reported names only; weights and hardware not independently verified'}
 (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
 connections={a:http.client.HTTPConnection(HOST,timeout=30) for a in MODELS}
 consecutive={a:0 for a in MODELS};disabled=set();done=0
 with (OUT/'attempts.jsonl').open('x') as ledger,(OUT/'responses.jsonl').open('x') as f:
  for i,r in enumerate(rs):
   for alias in (('pplx','d3') if i%2==0 else ('d3','pplx')):
    if alias in disabled:continue
    record={k:r[k] for k in ('suite','id','family','reverse','gold')};record['model']=alias
    record['request_index']=done;record['first_for_model']=i==0
    # Log before network; failed attempts are counted and not retried.
    ledger.write(json.dumps({'model':alias,'suite':r['suite'],'id':r['id'],'reverse':r['reverse'],'timestamp':time.time()},ensure_ascii=False)+'\n');ledger.flush()
    done+=1;t=time.perf_counter()
    try:
     body=json.dumps(payload(r,alias),ensure_ascii=False).encode()
     connections[alias].request('POST',f'/{alias}/v1/systemone',body,{'Content-Type':'application/json'})
     response=connections[alias].getresponse();raw=response.read(2*1024*1024+1)
     record['http_status']=response.status
     record['server_timing']=response.getheader('Server-Timing')
     assert len(raw)<=2*1024*1024,'response too large'
     parsed=json.loads(raw)
     record['response']=parsed
     if response.status!=200:raise ValueError(f'HTTP {response.status}')
     answer=parsed['answers']['move'];pred=answer['choice'];probs=answer['probabilities']
     keys=set(r['questions']['move']['criteria'])
     assert pred in keys and set(probs)==keys,'option mismatch'
     assert all(isinstance(x,(float,int)) and math.isfinite(x) and 0<=x<=1 for x in probs.values())
     assert abs(sum(probs.values())-1)<.01,'invalid probability sum'
     assert probs[pred]>=max(probs.values())-1e-6,'choice not top probability'
     record.update(pred=pred,probabilities=probs,correct=pred in r['gold'],valid=True)
     consecutive[alias]=0
    except Exception as exc:
     record.update(valid=False,correct=False,error_type=type(exc).__name__,error=str(exc))
     connections[alias].close();connections[alias]=http.client.HTTPConnection(HOST,timeout=30)
     consecutive[alias]+=1
     if consecutive[alias]>=3:disabled.add(alias)
    record['elapsed_ms']=1000*(time.perf_counter()-t)
    f.write(json.dumps(record,ensure_ascii=False)+'\n');f.flush()
   if (i+1)%20==0 or i==len(rs)-1:print('cases',i+1,len(rs),'requests',done,'disabled',sorted(disabled),flush=True)
   if len(disabled)==2:break
 for c in connections.values():c.close()
 assert done<=276 and sha(__file__)==cfg['script_sha256']
 (OUT/'completion.json').write_text(json.dumps({'requests':done,'disabled':sorted(disabled),'complete':done==276},indent=2))
 print('finished',done,flush=True)
if __name__=='__main__':main()

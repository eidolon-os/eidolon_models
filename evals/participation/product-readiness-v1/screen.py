import argparse,copy,hashlib,json,math,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
LEGACY=ROOT/'laya/train/runs/backbone-comparison-v1'
sys.path[:0]=[str(ROOT/'jevk5/src'),str(LEGACY/'runtime-deps'),str(LEGACY/'models/decider')]
from build import CLARIFY

def digest(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def read(p):return [json.loads(s) for s in Path(p).read_text().splitlines() if s.strip()]
def public(r,reverse=False):
 q=copy.deepcopy(r['questions'])
 if reverse:q['move']['criteria']=dict(reversed(list(q['move']['criteria'].items())))
 return {'state':copy.deepcopy(r['state']),'questions':q}
def mapped(state,pred):
 if pred=='abstain':return {'status':'abstained','proposal':None}
 action,_,cid=pred.partition(':')
 assert action in state['allowed_actions']
 if action in ('respond','clarify'):
  assert cid in state['candidates']
  proposal={'action':action,'participants':[cid],'instruction':CLARIFY if action=='clarify' else ''}
 else:
  assert action in ('wait','finish') and not cid
  proposal={'action':action,'participants':[],'instruction':''}
 return {'status':'decided','proposal':proposal}

def main(alias):
 import torch
 assert torch.backends.mps.is_available()
 torch.set_num_threads(4);torch.manual_seed(71)
 rows=read(HERE/'cases.jsonl')
 out=ROOT/'jevk5/runs/product-readiness-v1';out.mkdir(exist_ok=True)
 dest=out/(alias+'.jsonl'); assert not dest.exists()
 config={'alias':alias,'cases_sha256':digest(HERE/'cases.jsonl'),'script_sha256':digest(__file__),
  'design_sha256':digest(HERE/'SCREEN-DESIGN.md'),'build_sha256':digest(HERE/'build.py'),
  'forward_limit':48,'torch':torch.__version__,'device':'mps','dtype':'float16'}
 start=time.perf_counter()
 if alias=='jevk5':
  from eidolon_models_jevk5.engine import Engine
  engine=Engine(ROOT/'jevk5/models/jevk5/c4f7fdb3/weights')
  config['weight_sha256']=engine.base_sha256
  def preflight(r):return len(engine.encode(r)[0])
  def predict(r):return engine.predict(r)[0]
 else:
  from decider.infer import Decider
  engine=Decider(str(LEGACY/'models/decider'),device='mps',dtype=torch.float16,use_graphs=False)
  from eidolon_models_jevk5.data import digest as stream_digest
  config['weight_sha256']=stream_digest(LEGACY/'models/decider/model.safetensors')
  def args(r):
   q=r['questions']['move']
   return json.dumps(r['state'],ensure_ascii=False),[{'question':q['instructions'],'options':[k+': '+v for k,v in q['criteria'].items()]}]
  def preflight(r):
   state,qs=args(r)
   _,items=engine._decide_items([(state,qs)],8192)
   ids=items[0]['ids']
   assert len(engine.m.tok.encode('Context:\n'+state))<8192
   assert engine.m.tok.unk_token_id not in ids
   assert len(ids)<4096
   return len(ids)
  def predict(r):
   state,qs=args(r)
   p=engine.decide(state,qs,max_ctx_tokens=8192)[0]['probs_list']
   return dict(zip(r['questions']['move']['criteria'],p))
 torch.mps.synchronize();config['load_and_hash_seconds']=time.perf_counter()-start
 inputs=[(r,rev,public(r,rev)) for r in rows for rev in (False,True)]
 tokens=[preflight(p) for r,rev,p in inputs]
 (out/(alias+'.config.json')).write_text(json.dumps(config,indent=2))
 with dest.open('x') as f:
  for i,((r,rev,p),nt) in enumerate(zip(inputs,tokens)):
   torch.mps.synchronize();t=time.perf_counter()
   probs=predict(p);pred=max(probs,key=probs.get);result=mapped(p['state'],pred)
   torch.mps.synchronize();ms=(time.perf_counter()-t)*1000
   assert abs(sum(probs.values())-1)<.002 and all(math.isfinite(x) and 0<=x<=1 for x in probs.values())
   rec={'id':r['id'],'reverse':rev,'family':r['meta']['family'],'task':r['meta']['task'],'gold':r['labels']['move']['gold'],
        'pred':pred,'correct':pred in r['labels']['move']['gold'],'forbid_speech':r['meta']['forbid_speech'],
        'probabilities':probs,'mapped':result,'tokens':nt,'elapsed_ms':ms,'first_call':i==0,
        'mps_driver_allocated_bytes':torch.mps.driver_allocated_memory()}
   f.write(json.dumps(rec,ensure_ascii=False)+'\n');f.flush()
   if (i+1)%12==0:print(alias,i+1,48,flush=True)
 assert digest(__file__)==config['script_sha256'] and digest(HERE/'cases.jsonl')==config['cases_sha256']
 print(alias,'completed',flush=True)
if __name__=='__main__':
 p=argparse.ArgumentParser();p.add_argument('alias',choices=['decider','jevk5']);main(p.parse_args().alias)

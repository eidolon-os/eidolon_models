import copy,importlib.util,json,sys,hashlib
from pathlib import Path
HERE=Path(__file__).resolve().parent;ROOT=HERE.parents[2];PREV=HERE.parent/'d3-followup-20261019'
sys.path.insert(0,str(PREV))
spec=importlib.util.spec_from_file_location('prior_driver',PREV/'run.py');old=importlib.util.module_from_spec(spec);spec.loader.exec_module(old)
OUT=ROOT/'jevk5/runs/d3-prompt-ab-20261019';old.OUT=OUT
TOTAL={'d3':0,'reply':0};CAP={'d3':136,'reply':12}
def sha(p):return hashlib.sha256(Path(p).read_bytes()).hexdigest()
def questions(q,policy,addition):
 out=copy.deepcopy(q)
 if policy=='enhanced':out['move']['instructions']+=addition
 return out
class Client(old.BudgetClient):
 def __init__(self,key,policy,addition):super().__init__(key);self.policy=policy;self.addition=addition
 def call(self,kind,payload,tag):
  if TOTAL[kind]>=CAP[kind] or self.bad[kind]>=3:raise RuntimeError('global_budget_or_consecutive_errors')
  TOTAL[kind]+=1
  return super().call(kind,payload,self.policy+'|'+tag)
 def decide(self,s,q,tag):return super().decide(s,questions(q,self.policy,self.addition),tag)
def main():
 data=json.loads((HERE/'cases.json').read_text());key=old.dotenv_values(ROOT.parent/'eidolon_agent/config/.env').get('EIDOLON_AGENT_LLM_API_KEY');assert key
 OUT.mkdir(parents=True,exist_ok=False)
 config={'hashes':{str(p.relative_to(ROOT)):sha(p) for p in [HERE/'experiment.py',HERE/'prepare.py',HERE/'cases.json',HERE/'DESIGN.md',PREV/'run.py',PREV/'build.py']},
  'caps':CAP,'policy_delta':'append instructions only','retries':0,'actual_production_agent':False}
 (OUT/'config.json').write_text(json.dumps(config,indent=2))
 clients={p:Client(key,p,data['addition']) for p in ('baseline','enhanced')};static=[];loops=[]
 for i,r in enumerate(data['cases']):
  for p in (('baseline','enhanced') if i%2==0 else ('enhanced','baseline')):
   x={k:r[k] for k in ('id','family','variant','suite','gold')};x['policy']=p
   try:
    pred=clients[p].decide(r['state'],r['questions'],'static:'+r['id'])
    x.update(pred=pred,correct=pred in r['gold'])
   except Exception as e:x.update(correct=False,error_type=type(e).__name__)
   static.append(x)
   with (OUT/'static.jsonl').open('a') as f:f.write(json.dumps(x,ensure_ascii=False)+'\n')
  if (i+1)%14==0:print('static pairs',i+1,56,flush=True)
 # Runtime state/policy code is unchanged; use the same new cast for both arms.
 old.CAST=data['cast']
 for i,case in enumerate(data['episodes']):
  for p in (('baseline','enhanced') if i%2==0 else ('enhanced','baseline')):
   try:r=old.episode(clients[p],case,False,data)
   except Exception as e:r={'id':case['id'],'passed':False,'error_type':type(e).__name__}
   r['policy']=p;loops.append(r)
   with (OUT/'rollouts.jsonl').open('a') as f:f.write(json.dumps(r,ensure_ascii=False)+'\n')
   print('episode',case['id'],p,r['passed'],flush=True)
 for client in clients.values():client.http.close()
 assert all(sha(ROOT/p)==h for p,h in config['hashes'].items())
 assert all(TOTAL[k]<=CAP[k] for k in TOTAL)
 summary={'actual_calls':TOTAL,'static':{p:{s:sum(r['correct'] for r in static if r['policy']==p and r['suite']==s) for s in ('old_diagnostic','new_diagnostic')} for p in clients},
 'episodes':{p:sum(r['passed'] for r in loops if r['policy']==p) for p in clients}}
 (OUT/'summary.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True)
if __name__=='__main__':main()

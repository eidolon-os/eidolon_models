"""Reload and fixed stratified option-order diagnostic; not used to choose epochs."""
import argparse,copy,json,random,time
from pathlib import Path
from eidolon_models_jevk5.data import read,group,digest
from eidolon_models_jevk5.engine import Engine
p=argparse.ArgumentParser();p.add_argument('--model',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
p.add_argument('--data',type=Path,required=True);p.add_argument('--epoch',type=int);p.add_argument('--out',type=Path,required=True);a=p.parse_args()
records=read(a.data/'dev.jsonl');name='baseline' if a.epoch is None else f'epoch-{a.epoch}.dev'
expected={r['id']:r for r in read(a.run/(name+'.jsonl'))}
selected={}
for r in records:
 key=(r['meta']['dataset'],group(r['labels']['move']['gold']))
 if key[1]!='mixed':selected.setdefault(key,r)
engine=Engine(a.model,adapter=None if a.epoch is None else a.run/f'epoch-{a.epoch}')
checks=[]
for i,((dataset,action),r) in enumerate(selected.items()):
 probs,_=engine.predict(r);old=expected[r['id']]
 delta=max(abs(probs[k]-old['probabilities'][k]) for k in probs)
 assert max(probs,key=probs.get)==old['pred'] and delta<1e-5
 modified=copy.deepcopy(r);criteria=list(modified['questions']['move']['criteria'].items())
 random.Random(917+i).shuffle(criteria)
 if criteria==list(r['questions']['move']['criteria'].items()):criteria.reverse()
 modified['questions']['move']['criteria']=dict(criteria)
 altered,_=engine.predict(modified);pred=max(altered,key=altered.get)
 checks.append(dict(id=r['id'],dataset=dataset,group=action,original=old['pred'],permuted=pred,
   gold=old['gold'],original_correct=old['correct'],permuted_correct=pred in old['gold'],
   same_answer=old['pred']==pred,reload_max_probability_delta=delta,permuted_probabilities=altered))
 print(name,i+1,len(selected),flush=True)
result=dict(configuration=name,n=len(checks),reload_parity_passed=True,
 max_reload_probability_delta=max(r['reload_max_probability_delta'] for r in checks),
 same_answers=sum(r['same_answer'] for r in checks),original_correct=sum(r['original_correct'] for r in checks),
 permuted_correct=sum(r['permuted_correct'] for r in checks),both_correct=sum(r['original_correct'] and r['permuted_correct'] for r in checks),
 rows=checks,selection='first DEV record in each dataset/action group, excluding mixed; fixed seed917; no fitting',
 script_sha256=digest(__file__))
# Fixed small TRAIN diagnostic; do not confuse this with a full train evaluation.
probe=[];counts={};snapshots=set()
for r in read(a.data/'train.jsonl'):
 key=(r['meta']['task'],group(r['labels']['move']['gold']))
 snapshot=r['meta']['snapshot']
 if key[1]=='mixed' or snapshot in snapshots or counts.get(key,0)>=4:continue
 snapshots.add(snapshot);counts[key]=counts.get(key,0)+1
 probs,_=engine.predict(r);pred=max(probs,key=probs.get)
 probe.append(dict(id=r['id'],task=key[0],group=key[1],pred=pred,gold=r['labels']['move']['gold'],correct=pred in r['labels']['move']['gold']))
result['train_probe']={'selection':'first four distinct snapshots per task/action group, or all if fewer; diagnostic subset only','rows':probe}
a.out.write_text(json.dumps(result,ensure_ascii=False,indent=2));print(json.dumps({k:v for k,v in result.items() if k not in ('rows','train_probe')}))

"""Read-only leakage/coverage diagnostics; no model-guided data changes."""
import json,re
from collections import Counter
from pathlib import Path
from eidolon_models_jevk5.data import read,digest
root=Path('evals/participation/data/companion-ip-v1')
train,dev=read(root/'train.jsonl'),read(root/'dev.jsonl')
def canonical(r):
 mapping={k:'ROLE'+str(v) for k,v in r['meta']['slot_to_source'].items()}
 def visit(x):
  if isinstance(x,dict):return {visit(k):visit(v) for k,v in x.items()}
  if isinstance(x,list):return [visit(v) for v in x]
  if isinstance(x,str):return re.sub(r'M\d+',lambda m:mapping.get(m[0],m[0]),x)
  return x
 return json.dumps(visit({'state':r['state'],'question':r['questions']['move']}),ensure_ascii=False,sort_keys=True)
def text(r):
 s=r['state']['user_request']
 for c in r['state']['candidates'].values():s=s.replace(c['name'],'人物')
 return re.sub(r'[^\w]','',s)
def grams(s):return {s[i:i+3] for i in range(max(1,len(s)-2))}
train_keys={canonical(r):r['id'] for r in train}
collisions=[{'train':train_keys[canonical(r)],'dev':r['id']} for r in dev if canonical(r) in train_keys]
assert not collisions
train_unique={r['meta']['snapshot']:r for r in train};dev_unique={r['meta']['snapshot']:r for r in dev}
near=[]
for x in dev_unique.values():
 a=grams(text(x));best=None
 for y in train_unique.values():
  b=grams(text(y));score=len(a&b)/max(1,len(a|b))
  if best is None or score>best[0]:best=(score,y['id'])
 if best[0]>=.65:near.append({'dev':x['id'],'nearest_train':best[1],'similarity':best[0]})
result={'train_records':len(train),'dev_records':len(dev),'train_snapshots':len(train_unique),'dev_snapshots':len(dev_unique),
 'cross_split_family_overlap':sorted({r['meta']['family'] for r in train}&{r['meta']['family'] for r in dev}),
 'canonical_public_input_collisions':collisions,'user_3gram_near_duplicates_threshold_0_65':near,
 'dev_origins':dict(Counter(r['meta']['origin'] for r in dev)),
 'meaning':'No canonical exact collision does not prove semantic/template independence. Near-text check is diagnostic only.',
 'input_sha256':{s:digest(root/f'{s}.jsonl') for s in ['train','dev']}}
Path('jevk5/experiments/companion-ip-v1/DATA-AUDIT.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
print(json.dumps(result,ensure_ascii=False,indent=2))

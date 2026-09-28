"""Family-aware diagnostics; confidence is reported, never fitted on test."""
import json
from collections import defaultdict
from pathlib import Path
import numpy as np

def metrics(records,rows):
 byid={r.id:r for r in records};families=defaultdict(list);slices=defaultdict(list)
 for p in rows:
  meta=byid[p['record_id']].meta
  families[meta['family']].append(p);slices[meta['slice']].append(meta['family'])
 famacc={f:sum(p['correct'] for p in ps)/len(ps) for f,ps in families.items()}
 sliceacc={s:float(np.mean([famacc[f] for f in set(fs)])) for s,fs in slices.items()}
 rng=np.random.default_rng(71);boots=[]
 for _ in range(2000):
  scores=[]
  for s,fs in slices.items():
   vals=[famacc[f] for f in sorted(set(fs))];scores.append(float(np.mean(rng.choice(vals,len(vals),replace=True))))
  boots.append(float(np.mean(scores)))
 def subset(pred):return [p for p in rows if pred(p)]
 def basic(ps):return {'n':len(ps),'correct':sum(p['correct'] for p in ps)}
 def canonical(p):
  value=p['pred'];m=byid[p['record_id']].meta
  return f"respond:{m['slot_to_source'][value.split(':')[1]]}" if value.startswith('respond:') else value
 stop=subset(lambda p:set(p['gold'])<={'wait','finish'})
 speak=subset(lambda p:all(g.startswith('respond:') for g in p['gold']))
 unique=subset(lambda p:len(p['gold'])==1 and p['gold'][0].startswith('respond:'))
 ambiguous=subset(lambda p:p['gold']==['abstain'])
 confidence={}
 for t in (.5,.7,.8,.9):
  selected=subset(lambda p:p['p_top']>=t and p['pred']!='abstain')
  confidence[str(t)]={'covered':len(selected),'correct':sum(p['correct'] for p in selected),'errors':sum(not p['correct'] for p in selected),'coverage':len(selected)/len(rows)}
 return {'records':len(rows),'generation_batches':len({r.meta['batch'] for r in records}),'families':len(families),'correct':sum(p['correct'] for p in rows),'family_accuracy':float(np.mean(list(famacc.values()))),
  'slice_macro_accuracy':float(np.mean(list(sliceacc.values()))),'worst_slice_accuracy':min(sliceacc.values()),
  'exploratory_family_bootstrap_95':np.quantile(boots,[.025,.975]).tolist(),
  'interval_caveat':'descriptive family resampling only; same-batch/teacher correlation and distribution shift not captured; not a release confidence bound',
  'generation_batches_by_slice':{sid:len({r.meta['batch'] for r in records if r.meta['slice']==sid}) for sid in slices},
  'slices':{s:{'families':len(set(fs)),'accuracy':sliceacc[s]} for s,fs in slices.items()},
  'must_stop':{**basic(stop),'wrong_speech':sum(p['pred'].startswith('respond:') for p in stop)},
  'must_speak':{**basic(speak),'missed_speech':sum(not p['pred'].startswith('respond:') for p in speak)},
  'unique_member':{**basic(unique),'wrong_member':sum(p['pred'].startswith('respond:') and not p['correct'] for p in unique)},
  'must_abstain':basic(ambiguous),'multi_gold':basic(subset(lambda p:len(p['gold'])>1)),
  'exact_equivariant_families':sum(len({canonical(p) for p in ps})==1 for ps in families.values()),
  'all_variants_correct_families':sum(all(p['correct'] for p in ps) for ps in families.values()),
  'confidence':confidence,
  'candidate_counts':{str(n):basic([p for p in rows if byid[p['record_id']].meta['members']==n]) for n in sorted({r.meta['members'] for r in records})},
  'constant_baselines':{k:sum(k in p['gold'] for p in rows)/len(rows) for k in ('respond:M0','wait','finish','abstain')}}

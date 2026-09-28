"""Freeze split-disjoint GLM scenes and two matched projection variants."""
import argparse
import hashlib
import json
import random
import re
from collections import Counter,defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from eidolon_sdk.biz.participation import Candidate,Context,Constraints,DecisionRequest,Message
from eidolon_models_laya.participation_text import project_request,check_capacity
from eidolon_models_laya.sequence import Tokenizer,build_sequence,to_internal
from eidolon_laya_train.model import record_items
from eidolon_laya_train.records import Record,write_jsonl
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parents[1]
RAW=TRAIN/'private/ip-team-v7-generalization/reviewed'

def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def all_rows():return [r for p in sorted(RAW.glob('*.json')) for r in json.loads(p.read_text())['results']]

def expand(row,variant,names_pool,named):
 d=row['scene'];fid=row['id'];n=len(d['cast']);rng=random.Random(f'v7:{fid}:{variant}')
 original=[c['name'] for c in d['cast']];unique=list(dict.fromkeys(original))
 replacements=unique if variant==0 or not row.get('rename_safe',False) else rng.sample(sorted(names_pool),len(unique))
 renames=dict(zip(unique,replacements))
 pattern=re.compile('|'.join(re.escape(x) for x in sorted(unique,key=len,reverse=True)))
 render=lambda x:pattern.sub(lambda m:renames[m.group()],x)
 ids=[f"member-{hashlib.sha256(f'{fid}:{variant}:{i}'.encode()).hexdigest()[:16]}" for i in range(n)]
 order=list(range(n))
 if variant:order=order[1:]+order[:1];rng.shuffle(order)
 if variant and order==list(range(n)):order=order[1:]+order[:1]
 def msg(mid,by,tx):return Message(message_id=mid,author_kind='user' if by=='user' else 'companion',author_id='owner' if by=='user' else ids[by],text=render(tx))
 user=msg('user-request','user',d['user'])
 before=[msg(f'b{i}',m['by'],m['text']) for i,m in enumerate(d['before'])]
 after=[msg(f'a{i}',m['by'],m['text']) for i,m in enumerate(d['after'])]
 history=before+[user]+after
 req=DecisionRequest(decision_id=f'{fid}-v{variant}',context_ref=f'{fid}-v{variant}',context_version=len(history),membership_revision=1,cancellation_epoch=0,timeout_ms=3000,
  user_request=user,trigger=history[-1],context=Context(recent_messages=tuple(history)),
  candidates=tuple(Candidate(companion_id=ids[i],display_name=renames[original[i]],description=render(d['cast'][i]['persona'])) for i in order),constraints=Constraints(remaining_replies=8))
 state,q,slots=project_request(req,include_names=named)
 byid={v:k for k,v in slots.items()}
 gold=[f'respond:{byid[ids[int(g.split(":")[1])]]}' if g.startswith('respond:') else g for g in d['gold']]
 return Record(id=f'{fid}~v{variant}',scenario='ip-team-v7-text',source='llm:glm-5.3-flash',state=state,questions={'move':q},labels={'move':{'gold':gold}},split=row['split'],tags=[row['slice'],f'family:{fid}'],meta={
  'family':fid,'slice':row['slice'],'batch':row['batch'],'renamed':bool(variant and row.get('rename_safe',False)),'variant':variant,'members':n,'synthetic':True,'projection':'named' if named else 'compact','request':req.model_dump(mode='json'),
  'slot_to_source':{slot:ids.index(cid) for slot,cid in slots.items()},'generator_sha256':row['generator_sha256'],'reviewer_sha256':row.get('reviewer_sha256'),
  'annotation':row.get('annotation_mode','generated_labels_and_blind_review'),'reason':d['why']})

def normalized(d):
 names=sorted({c['name'] for c in d['cast']},key=len,reverse=True)
 s=d['user'];s=re.sub('|'.join(map(re.escape,names)),'NAME',s)
 return re.sub(r'\s|[，。！？、；：,.!?;:]','',s)

def main():
 ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
 if a.out.exists():raise ValueError('immutable output exists')
 audit=json.loads((HERE/'audit-decisions.json').read_text());rows=all_rows()
 if set(audit)!={r['id'] for r in rows}:raise ValueError('all rows require explicit audit')
 accepted=[dict(r,slice=audit[r['id']].get('observed_slice',r.get('slice')),planned_slice=r.get('slice'),rename_safe=audit[r['id']].get('rename_safe',False)) for r in rows if audit[r['id']]['decision']=='accept']
 if any(r['status'] not in ('agreed_pending_audit','annotated_pending_audit') for r in accepted):raise ValueError('unreviewed scene')
 counts=Counter(r['split'] for r in accepted)
 if counts['train']<36 or counts['val']<12 or counts['test']<12:raise ValueError(f'insufficient independent families {counts}')
 pools={split:{c['name'] for r in accepted if r['split']==split for c in r['scene']['cast']} for split in counts}
 # Save near-duplicate diagnostics, never silently drop test rows after scoring.
 similarity=[];seen={}
 for r in accepted:
  public={k:v for k,v in r['scene'].items() if k not in ('id','gold','why')}
  sig=json.dumps(public,ensure_ascii=False,sort_keys=True)
  if sig in seen:raise ValueError(f"exact duplicated public scene: {seen[sig]}, {r['id']}")
  seen[sig]=r['id']
 for i,r in enumerate(accepted):
  u=normalized(r['scene']);ug={u[k:k+3] for k in range(max(1,len(u)-2))}
  for t in accepted[:i]:
   if r['split']==t['split']:continue
   v=normalized(t['scene']);vg={v[k:k+3] for k in range(max(1,len(v)-2))}
   j=len(ug&vg)/max(1,len(ug|vg))
   if j>=.65:similarity.append({'a':r['id'],'b':t['id'],'user_trigram_jaccard':round(j,4)})
 tokpath=TRAIN/'runs/r14/checkpoint/tokenizer';tok=Tokenizer(tokpath);hf=AutoTokenizer.from_pretrained(tokpath,local_files_only=True)
 lineage={'counts':dict(counts),'generation_batches':{split:len({r['batch'] for r in accepted if r['split']==split}) for split in counts},'actions_by_split':{split:dict(Counter(g.split(':')[0] for r in accepted if r['split']==split for g in r['scene']['gold'])) for split in counts},'slice_families':{split:dict(Counter(r['slice'] for r in accepted if r['split']==split)) for split in counts},'audit_sha256':digest(HERE/'audit-decisions.json'),'projection_sha256':digest(HERE.parents[2]/'src/eidolon_models_laya/participation_text.py'),'prepare_sha256':digest(Path(__file__)),'source_sha256':{p.name:digest(p) for p in sorted(RAW.glob('*.json'))},'files':{},'lengths':{},'cross_split_near_user_text':similarity}
 material={}
 for mode in ('compact','named'):
  splits=defaultdict(list);lengths=[]
  for row in accepted:
   for variant in range(2):
    rec=expand(row,variant,pools[row['split']],mode=='named')
    lengths.append(check_capacity(tok,rec.state,rec.questions['move'],2048,256))
    ids,markers,cut=build_sequence(tok,rec.state,to_internal(rec.questions['move']),2048,256)
    it=record_items(rec,hf,{'max_len':2048,'head_max_len':256})[0]
    assert not cut and ids==it['ids'] and markers==it['markers']
    splits[row['split']].append(rec)
  material[mode]=splits
  lineage['lengths'][mode]={'min':min(lengths),'max':max(lengths),'parity_records':len(lengths)}
 for split in counts:
  for x,y in zip(material['compact'][split],material['named'][split],strict=True):
   assert x.state==y.state and x.labels==y.labels and x.id==y.id
 a.out.mkdir(parents=True)
 for mode,splits in material.items():
  for split,recs in splits.items():
   path=a.out/(f'sealed/{mode}/test.jsonl' if split=='test' else f'{mode}/{split}.jsonl')
   write_jsonl(path,recs);lineage['files'][str(path.relative_to(a.out))]={'sha256':digest(path),'records':len(recs)}
   for sid in sorted({r.meta['slice'] for r in recs}):write_jsonl(a.out/'slices'/sid/mode/f'{split}.jsonl',[r for r in recs if r.meta['slice']==sid])
 (a.out/'reviewed-scenes.json').write_text(json.dumps(accepted,ensure_ascii=False,indent=2))
 (a.out/'lineage.json').write_text(json.dumps(lineage,ensure_ascii=False,indent=2))
 print(json.dumps(lineage,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

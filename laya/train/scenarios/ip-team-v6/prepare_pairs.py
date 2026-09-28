"""Freeze GLM-labelled paired families, preserving branch identity permutations."""
import argparse
import hashlib
import importlib.util
import json
import re
from pathlib import Path
from transformers import AutoTokenizer
from eidolon_laya_train.records import read_jsonl,write_jsonl
from eidolon_laya_train.model import record_items
from eidolon_models_laya.participation_text import check_capacity
from eidolon_models_laya.sequence import Tokenizer,build_sequence,to_internal
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parents[1]
s=importlib.util.spec_from_file_location('prepare_v5',HERE.parent/'ip-team-v5/prepare.py')
v5=importlib.util.module_from_spec(s);s.loader.exec_module(v5)


def expand_pair(row,branch,variant):
    d=row['pair']; names={d['name_'+s]:s for s in reversed('abc')}
    # Equal display names map to one surface form; author IDs remain distinct.
    pattern=re.compile('|'.join(re.escape(n) for n in sorted(names,key=len,reverse=True)))
    render=lambda text:pattern.sub(lambda m:'{'+names[m.group()]+'}',text)
    case={**{f'role_{s}':render(d[f'role_{s}']) for s in 'abc'},'user':render(d['user_'+branch]),
          'peer_author':d['peer_author_'+branch],'peer_text':render(d['peer_text_'+branch]),
          'earlier_author':'','earlier_text':'','acceptable':d['acceptable_'+branch], 'reason':d['reason_'+branch]}
    entry={'case':case,'id':row['id'],'split':row['split'],'same_name':row['slice']=='same_name',
           'source':'llm:glm-5.3-flash','slice':row['slice'],'review_reason':row['review']['reason_'+branch],
           'generator_sha256':row['generator_sha256']}
    r=v5.expand(entry,variant)
    r.id=f"{row['id']}-{branch}~v{variant}"
    r.scenario='ip-team-v6-pairs'
    r.meta.update(pair_id=row['id'],branch=branch,reviewer_sha256=row.get('reviewer_sha256'),schema=row.get('schema','duplicated_fields_v1'))
    r.meta['request']['decision_id']+=f'-{branch}'
    return r


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise ValueError('immutable output exists')
    source=TRAIN/'private/ip-team-v6-pairs/reviewed'
    audit=json.loads((HERE/'audit-decisions.json').read_text())
    rows=[json.loads(p.read_text()) for p in sorted(source.glob('*.json'))]
    if set(audit)!={r['id'] for r in rows}:raise ValueError('audit must cover every family')
    tokpath=TRAIN/'runs/r14/checkpoint/tokenizer';tok=Tokenizer(tokpath);hf=AutoTokenizer.from_pretrained(tokpath,local_files_only=True)
    splits={'train':[],'val':[]};accepted=[];lengths=[]
    for r in rows:
        if audit[r['id']]['decision']!='accept':continue
        if r['status']!='agreed_pending_audit':raise ValueError('cannot silently relabel disagreement')
        if set(r['pair']['acceptable_x']) & set(r['pair']['acceptable_y']):raise ValueError('pair does not discriminate choices')
        accepted.append(r)
        for variant in range(4):
            pair=[expand_pair(r,b,variant) for b in 'xy']
            assert pair[0].state['candidates']==pair[1].state['candidates']
            assert pair[0].questions==pair[1].questions
            for rec in pair:
                lengths.append(check_capacity(tok,rec.state,rec.questions['move']))
                ids,markers,cut=build_sequence(tok,rec.state,to_internal(rec.questions['move']),2048,256)
                item=record_items(rec,hf,{'max_len':2048,'head_max_len':256})[0]
                assert not cut and ids==item['ids'] and markers==item['markers']
                splits[r['split']].append(rec)
    # Keep previously audited train only; old val is regression-only, never enters train or selection.
    old=TRAIN/'data/generated/ip-team-v5/pilot1'
    splits['train']+=list(read_jsonl(old/'train.jsonl'))
    if not splits['val']:raise ValueError('no new development families')
    a.out.mkdir(parents=True)
    lineage={'source':str(source),'prior_train_sha256':v5.digest(old/'train.jsonl'),'audit_sha256':v5.digest(HERE/'audit-decisions.json'),
             'reviewed_source_sha256':{p.name:v5.digest(p) for p in sorted(source.glob('*.json'))},
             'prepare_source_sha256':v5.digest(Path(__file__)),
             'split_policy':'story pair and all permutations remain within preassigned train/val; old dev regression only',
             'min_tokens':min(lengths),'max_tokens':max(lengths),'new_families':{r['id']:r['split'] for r in accepted},
             'projection_sha256':v5.digest(HERE.parents[2]/'src/eidolon_models_laya/participation_text.py')}
    for split,recs in splits.items():
        write_jsonl(a.out/f'{split}.jsonl',recs)
        lineage[split]={'records':len(recs),'families':len({r.meta['family'] for r in recs}),'sha256':v5.digest(a.out/f'{split}.jsonl')}
    assert not {r.meta['family'] for r in splits['train']} & {r.meta['family'] for r in splits['val']}
    (a.out/'lineage.json').write_text(json.dumps(lineage,ensure_ascii=False,indent=2))
    (a.out/'reviewed-pairs.json').write_text(json.dumps(accepted,ensure_ascii=False,indent=2))
    for r in accepted:
        p=a.out/'slices'/r['slice'];p.mkdir(parents=True,exist_ok=True)
        (p/(r['id']+'.json')).write_text(json.dumps(r,ensure_ascii=False,indent=2))
    print(json.dumps(lineage,ensure_ascii=False,indent=2))
if __name__=='__main__':main()

"""Freeze complete audited families; refuse under-covered action groups."""
import argparse
import hashlib
import importlib.util
import json
from collections import Counter, defaultdict
from pathlib import Path
from transformers import AutoTokenizer
from eidolon_models_laya.sequence import Tokenizer, build_sequence, to_internal
from eidolon_models_laya.participation_text import check_capacity
from eidolon_laya_train.model import record_items
from eidolon_laya_train.records import write_jsonl

HERE = Path(__file__).resolve().parent
LAYA = HERE.parents[2]
RAW = LAYA/'train/private/ip-team-v8-boundaries'
s = importlib.util.spec_from_file_location('v7_prepare', HERE.parent/'ip-team-v7/prepare.py')
v7 = importlib.util.module_from_spec(s)
s.loader.exec_module(v7)


def group(gold):
    if gold and all(g.startswith('respond:') for g in gold):
        return 'respond'
    if gold and set(gold) <= {'wait','finish'}:
        return 'stop'
    if gold == ['abstain']:
        return 'abstain'
    return 'mixed'


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--out', type=Path, required=True)
    a = ap.parse_args()
    if a.out.exists():
        raise ValueError('immutable dataset exists')
    audit = json.loads((HERE/'audit-decisions.json').read_text())
    rows = [json.loads(p.read_text()) for p in sorted((RAW/'reviewed').glob('*.json'))]
    if set(audit) != {r['id'] for r in rows}:
        raise ValueError('all rows need explicit audit')
    accepted = [r for r in rows if audit[r['id']]['decision']=='accept']
    for r in accepted:
        if r['status'] != 'annotated_pending_audit' or Counter(group(d['gold']) for d in r['scenes']) != Counter({'respond':1,'stop':1,'abstain':1}):
            raise ValueError('complete valid three-group family required')
    counts = Counter(r['split'] for r in accepted)
    if any(counts[sp] < n for sp,n in [('train',20),('val',10),('test',10)]):
        raise ValueError(f'coverage gate failed: {dict(counts)}')
    exact = {};similar=[]
    for r in accepted:
        public=[{k:v for k,v in d.items() if k not in ('gold','why','id')} for d in r['scenes']]
        for d in public:
            sig=json.dumps(d,ensure_ascii=False,sort_keys=True)
            if sig in exact:raise ValueError('exact duplicate public input')
            exact[sig]=r['id']
    for i,r in enumerate(accepted):
        for other in accepted[:i]:
            if r['split']==other['split']:continue
            for x in r['scenes']:
                for y in other['scenes']:
                    u=v7.normalized(x);v=v7.normalized(y)
                    ug={u[j:j+3] for j in range(max(1,len(u)-2))};vg={v[j:j+3] for j in range(max(1,len(v)-2))}
                    score=len(ug&vg)/max(1,len(ug|vg))
                    if score>=.65:similar.append({'a':r['id']+x['id'],'b':other['id']+y['id'],'score':score})
    tokpath=LAYA/'models/laya-multilingual/1c5edc17/torch/tokenizer'
    tok=Tokenizer(tokpath);hf=AutoTokenizer.from_pretrained(tokpath,local_files_only=True)
    output=defaultdict(list);lengths=[]
    for r in accepted:
        for d in r['scenes']:
            g=group(d['gold']);snapshot=r['id']+'-'+d['id']
            row={'id':snapshot,'scene':d,'split':r['split'],'slice':g,'batch':r['batch'],
                 'generator_sha256':r['generator_sha256'],'reviewer_sha256':r['reviewer_sha256'],
                 'rename_safe':False,'annotation_mode':'glm_facts_and_blind_labels_assistant_accept_only'}
            for variant in (0,1):
                rec=v7.expand(row,variant,set(),True)
                rec.scenario='ip-team-v8-boundaries'
                rec.meta.update(family=r['id'],snapshot=snapshot,group=g,world=r['plan']['world'])
                rec.tags=[g,'family:'+r['id']]
                lengths.append(check_capacity(tok,rec.state,rec.questions['move'],2048,256))
                ids,markers,cut=build_sequence(tok,rec.state,to_internal(rec.questions['move']),2048,256)
                train=record_items(rec,hf,{'max_len':2048,'head_max_len':256})[0]
                assert not cut and ids==train['ids'] and markers==train['markers']
                output[r['split']].append(rec)
    lineage={'families':dict(counts),'records':{k:len(v) for k,v in output.items()},
             'action_family_counts':{sp:{g:counts[sp] for g in ('respond','stop','abstain')} for sp in counts},
             'min_tokens':min(lengths),'max_tokens':max(lengths),'parity_records':len(lengths),
             'audit_sha256':v7.digest(HERE/'audit-decisions.json'),'prepare_sha256':v7.digest(Path(__file__)),
             'source_sha256':{p.name:v7.digest(p) for p in sorted((RAW/'reviewed').glob('*.json'))},
             'cross_split_near_user_text':similar,'files':{}}
    names={sp:{c['name'] for r in accepted if r['split']==sp for c in r['scenes'][0]['cast']} for sp in counts}
    lineage['shared_display_names']={f'{a}/{b}':sorted(names[a]&names[b]) for a,b in [('train','val'),('train','test'),('val','test')]}
    lineage['candidate_count_families']={sp:dict(Counter(len(r['scenes'][0]['cast']) for r in accepted if r['split']==sp)) for sp in counts}
    lineage['world_families']={sp:dict(Counter(r['plan']['world'] for r in accepted if r['split']==sp)) for sp in counts}
    a.out.mkdir(parents=True)
    for split,recs in output.items():
        path=a.out/('sealed/named/test.jsonl' if split=='test' else f'named/{split}.jsonl')
        write_jsonl(path,recs)
        lineage['files'][str(path.relative_to(a.out))]={'records':len(recs),'sha256':v7.digest(path)}
    (a.out/'lineage.json').write_text(json.dumps(lineage,ensure_ascii=False,indent=2))
    (a.out/'reviewed-families.json').write_text(json.dumps(accepted,ensure_ascii=False,indent=2))
    print(json.dumps({k:v for k,v in lineage.items() if k!='source_sha256'},ensure_ascii=False,indent=2))


if __name__=='__main__':main()

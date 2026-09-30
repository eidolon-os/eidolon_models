"""Import only old train/DEV, plus locally audited new synthetic families."""
import argparse
import copy
import json
import random
from collections import Counter,defaultdict
from pathlib import Path

from .data import digest,group,payload,read,validate,write

INSTRUCTIONS = ('选择一个合适的下一步。依据用户意图、角色关系和已公开内容选择回应者，不机械轮流。'
    '暂停等用户选wait；已完成选finish；无法可靠选择或需要消歧选abstain。'
    '安静陪伴可以文字回应；只提及名字不等于点名。')


def project(scene,gold,plan,variant):
    count=len(scene['cast']);order=list(range(count))
    if variant:order.reverse()
    slots={source:f'M{pos+variant*3}' for pos,source in enumerate(order)}
    candidates={slots[i]:{'name':scene['cast'][i]['name'],'character':scene['cast'][i]['persona']} for i in order}
    actions={f'respond:{slots[i]}':f"由{scene['cast'][i]['name']}（{slots[i]}）回应" for i in order}
    actions.update(wait='暂停等用户',finish='本轮已完成',abstain='无法可靠选择或需要澄清')
    if variant:
        items=list(actions.items());random.Random(173).shuffle(items);actions=dict(items)
    state={'user_request':scene['user'],'scene_goal':'','candidates':candidates,
        'allowed_actions':['respond','clarify','wait','finish'],'remaining_replies':8,
        'prior_public_messages':[{'author':'user' if m['speaker']==-1 else slots[m['speaker']],'text':m['text']} for m in scene['before']],
        'trigger':{'same_as_user_request':True}}
    mapped=[f"respond:{slots[int(g.split(':')[1])]}" if g.startswith('respond:') else g for g in gold]
    family='new:'+plan['id'];snapshot=family+':'+scene['id']
    return validate(dict(id=snapshot+f'~v{variant}',state=state,
        questions={'move':{'type':'choice','instructions':INSTRUCTIONS,'criteria':actions}},
        labels={'move':{'gold':mapped}},split=plan['split'],meta=dict(family=family,snapshot=snapshot,
            task=plan['task'],dataset=plan['task'],variant=variant,synthetic=True,
            slot_to_source={v:k for k,v in slots.items()},annotation='GLM blind review plus explicit local audit')))


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--repo',type=Path,required=True)
    ap.add_argument('--generated',type=Path,required=True);ap.add_argument('--audit',type=Path,required=True)
    ap.add_argument('--authored',type=Path)
    ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    if a.out.exists():raise ValueError('immutable dataset already exists')
    result=defaultdict(list);sources={}
    for version,folder in [('v7','generalization1'),('v8','boundaries1')]:
        for split,old in [('train','train'),('dev','val')]:
            p=a.repo/f'laya/train/data/generated/ip-team-{version}/{folder}/named/{old}.jsonl'
            sources[str(p.relative_to(a.repo))]=digest(p)
            for original in read(p):
                r=copy.deepcopy(original);m=r['meta'];r['id']=version+':'+r['id']
                m['snapshot']=version+':'+m.get('snapshot',m['family']);m['family']=version+':'+m['family']
                m.update(task='ip_team',dataset=version,origin='legacy_glm');r['split']=split
                result[split].append(validate(r))
    audit=json.loads(a.audit.read_text());seen=set();rejections=[];quarantined=[]
    for p in sorted((a.generated/'reviewed').glob('*.json')):
        d=json.loads(p.read_text());sources[str(p)]=digest(p)
        if d['status']=='quarantined_generation_timeout':
            assert d['scenes']==[] and d['reviews']==[]
            quarantined.append(d['plan']['id']);continue
        reviews={r['id']:r for r in d['reviews']}
        for s in d['scenes']:
            key=d['plan']['id']+':'+s['id'];seen.add(key)
            decision=audit[key]
            if not decision['reason'].strip():raise ValueError('audit rationale required')
            if decision['decision']=='reject':rejections.append(key);continue
            review=reviews[s['id']]
            if decision['decision']=='accept':
                if not review['valid']:raise ValueError('invalid teacher review accepted')
                gold=review['gold']
            elif decision['decision']=='correct':gold=decision['gold']
            else:raise ValueError('audit decision')
            for v in (0,1):
                r=project(s,gold,d['plan'],v);r['meta']['origin']='new_glm_audited'
                result[d['plan']['split']].append(r)
    if set(audit)!=seen or len(seen)+4*len(quarantined)!=48:raise ValueError('all generated scenes need explicit audit; missing families require explicit quarantine')
    if a.authored:
        sources[str(a.authored)]=digest(a.authored)
        for family in json.loads(a.authored.read_text()):
            if family['plan']['split']!='dev':raise ValueError('authored diagnostics DEV only')
            for scene in family['scenes']:
                public={k:scene[k] for k in ('id','user','before')};public['cast']=family['cast']
                for v in (0,1):
                    r=project(public,scene['gold'],family['plan'],v)
                    r['meta'].update(origin='assistant_authored_diagnostic',annotation=scene['why'])
                    result['dev'].append(r)
    families={k:{r['meta']['family'] for r in rows} for k,rows in result.items()}
    if families['train']&families['dev']:raise ValueError('family leakage')
    exact=defaultdict(set)
    for split,rows in result.items():
        for r in rows:
            # Metadata/labels excluded; compare normalized public state plus instructions/options.
            state,q,_=payload(r);exact[json.dumps([json.loads(state),q],sort_keys=True,ensure_ascii=False)].add(split)
    if any(len(s)>1 for s in exact.values()):raise ValueError('exact public input crosses split')
    coverage={sp:dict(Counter((r['meta']['task']+'/'+group(r['labels']['move']['gold'])) for r in rows)) for sp,rows in result.items()}
    for sp in ['train','dev']:
        for task in ['companion','ip_team']:
            for action in ['respond','stop']:
                if coverage[sp].get(task+'/'+action,0)<2:raise ValueError('missing basic task/action coverage')
    a.out.mkdir(parents=True)
    for split,rows in result.items():write(a.out/f'{split}.jsonl',rows)
    manifest=dict(sources=sources,audit_sha256=digest(a.audit),prepare_sha256=digest(__file__),
        files={sp:dict(sha256=digest(a.out/f'{sp}.jsonl'),records=len(rows),families=len(families[sp])) for sp,rows in result.items()},
        coverage=coverage,rejected_scenes=rejections,quarantined_families=quarantined,scope='synthetic train and seen DEV only; no test files imported')
    (a.out/'manifest.json').write_text(json.dumps(manifest,ensure_ascii=False,indent=2));print(json.dumps(manifest['files']),flush=True)


if __name__=='__main__':main()

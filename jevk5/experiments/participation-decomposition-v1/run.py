"""Frozen public-input decomposition diagnostic; no fitting or network access."""
import copy
import hashlib
import json
import math
import random
import statistics
import time
from pathlib import Path

from eidolon_models_jevk5.data import read, digest, group
from eidolon_models_jevk5.engine import Engine
from eidolon_models_jevk5.evaluate import summarize, gate

ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent
DATA = ROOT / 'evals/participation/data/companion-ip-v1/dev.jsonl'
BASE = ROOT / 'jevk5/runs/companion-ip-v1-seed71/baseline.jsonl'
OUT = ROOT / 'jevk5/runs/participation-decomposition-v1'
ACTION = '仅判断现在是否需要文字回应，暂不选择具体角色，也不要因为角色同名或指代不清而拒绝判断时机。按用户最新意图及公开上下文：邀请交流、提问、允许文字的安静陪伴选respond；明确要求暂时不说话、等待后续通知选wait；明确结束本轮选finish；连是否需要回应都无法判断才选unclear。情绪低落不自动意味着禁止回复；别把完成某个活动自动当作结束聊天。'
VERIFY = '仅核验proposed_responder是否符合用户对回应者的要求，暂不判断回应时机。依据公开内容，不得猜测未提供的身份或经历。若用户指定某人或某段经历，必须有足够证据把该指向落实到此候选；同名候选缺少区分依据或指定的人不在候选内选ambiguous。若公开信息明确指向其他候选选other。若用户向大家开放交流、没有限制谁来回应，合理候选可选supported，不要求唯一人选。只提及名字不等于要求那个人回答。'

def proposal(probabilities):
    eligible = {k:v for k,v in probabilities.items() if k.startswith('respond:')}
    return max(eligible, key=eligible.get)


def inputs(row, proposed):
    # Only these public fields are carried into inference. No gold/meta/id/tag.
    state = copy.deepcopy(row['state'])
    common = ' 原任务规则：' + row['questions']['move']['instructions']
    action = {'state': state, 'questions': {'move': {'type': 'choice', 'instructions': ACTION + common,
        'criteria': {'respond':'当前需要文字回应，身份另行判断', 'wait':'暂不出声，等待用户', 'finish':'本轮明确结束', 'unclear':'无法判断是否需要回应'}}}}
    verify_state = copy.deepcopy(state)
    verify_state['proposed_responder'] = proposed.removeprefix('respond:')
    verify = {'state': verify_state, 'questions': {'move': {'type': 'choice', 'instructions': VERIFY + common,
        'criteria': {'supported':'此人符合要求，或属于开放交流的合理候选', 'ambiguous':'所需身份信息缺失或有歧义，不能可靠落实指向', 'other':'公开信息明确要求其他候选'}}}}
    # Balanced deterministic order independent of labels; repeated reversed below.
    for name, item in [('action',action),('verify',verify)]:
        pairs = list(item['questions']['move']['criteria'].items())
        seed = int(hashlib.sha256((name + json.dumps(state,ensure_ascii=False,sort_keys=True)).encode()).hexdigest()[:16],16)
        random.Random(seed).shuffle(pairs)
        item['questions']['move']['criteria'] = dict(pairs)
    return action, verify


def combine(base_pred, proposed, action, verify):
    verified = proposed if verify == 'supported' else 'abstain'
    repair = verified if base_pred.startswith('respond:') else base_pred
    decomposed = verified if action == 'respond' else ('abstain' if action == 'unclear' else action)
    return repair, decomposed


def measure(engine, item):
    start=time.perf_counter()
    probs,tokens=engine.predict(item)
    assert all(math.isfinite(v) and 0 <= v <= 1 for v in probs.values())
    assert abs(sum(probs.values())-1)<.002
    return {'pred':max(probs,key=probs.get),'probabilities':probs,'tokens':tokens,'elapsed_ms':1000*(time.perf_counter()-start)}


def main():
    rows=read(DATA); baseline=read(BASE)
    assert digest(DATA)=='82c4c8be8123f7b3dcc179a9f05ae7a77ecd43c703f5c39389d9373993cf8a50'
    assert len(rows)==len(baseline)==152
    for r,b in zip(rows,baseline):
        assert r['split']=='dev' and r['id']==b['id'] and r['labels']['move']['gold']==b['gold']
    # First record in each measured task/action stratum, frozen before inference.
    probes={}
    for r in rows:
        key=(r['meta']['dataset'],group(r['labels']['move']['gold']))
        if key[1]!='mixed': probes.setdefault(key,r['id'])
    OUT.mkdir(exist_ok=False)
    cfg={'data_sha256':digest(DATA),'baseline_sha256':digest(BASE),'script_sha256':digest(__file__),
         'design_sha256':digest(HERE/'DESIGN.md'),'max_new_forwards':304+2*len(probes),
         'order_probe_ids':list(probes.values()),'model':'JevK5 c4f7fdb3, unadapted',
         'external_calls':0,'fit':False,'sealed_test':False,'release':False}
    (OUT/'config.json').write_text(json.dumps(cfg,ensure_ascii=False,indent=2))
    engine=Engine(ROOT/'jevk5/models/jevk5/c4f7fdb3/weights')
    raw=[]; variants={'repair':[],'decomposed':[]}; order=[]
    with (OUT/'raw.jsonl').open('x') as f:
        for i,(r,b) in enumerate(zip(rows,baseline)):
            p=proposal(b['probabilities']); ai,vi=inputs(r,p)
            a,v=measure(engine,ai),measure(engine,vi)
            preds=combine(b['pred'],p,a['pred'],v['pred'])
            item={'id':r['id'],'proposal':p,'action':a,'verify':v,'repair':preds[0],'decomposed':preds[1]}
            if r['id'] in probes.values():
                for q in (ai,vi):
                    c=q['questions']['move']['criteria']; q['questions']['move']['criteria']=dict(reversed(list(c.items())))
                ar,vr=measure(engine,ai),measure(engine,vi)
                reverse=combine(b['pred'],p,ar['pred'],vr['pred'])
                item['reversed']={'action':ar,'verify':vr,'repair':reverse[0],'decomposed':reverse[1]}
                order.append({'id':r['id'],'action_same':a['pred']==ar['pred'],'verify_same':v['pred']==vr['pred'],
                    'repair_same':preds[0]==reverse[0],'decomposed_same':preds[1]==reverse[1]})
            raw.append(item);f.write(json.dumps(item,ensure_ascii=False)+'\n');f.flush()
            for name,pred in zip(variants,preds):
                x={k:b[k] for k in ('id','dataset','family','origin','gold')}
                # Baseline proposal pass is REQUIRED at runtime. Historical estimate only.
                extra=v['elapsed_ms'] if name=='repair' and b['pred'].startswith('respond:') else 0
                if name=='decomposed':extra=a['elapsed_ms']+(v['elapsed_ms'] if a['pred']=='respond' else 0)
                x.update(pred=pred,correct=pred in b['gold'],elapsed_ms=b['elapsed_ms']+extra)
                variants[name].append(x)
            if (i+1)%10==0 or i==151:print('completed',i+1,152,flush=True)
    report={'baseline':{'metrics':summarize(baseline),'gate':gate(summarize(baseline))},'order_probe':order,
            'latency_note':'Estimated sequential total includes historical baseline proposal pass; not same-run benchmark. Cold load, queue, network excluded.',
            'stage_latency_ms':{s:{'median':statistics.median(x[s]['elapsed_ms'] for x in raw)} for s in ('action','verify')}}
    for name,rs in variants.items():
        metrics=summarize(rs)
        v8=metrics['v8']['groups']
        signal=metrics['v7']['correct']>=36 and v8['abstain']['correct']>=12 and v8['respond']['correct']>=22 and v8['stop']['correct']>=22
        report[name]={'metrics':metrics,'gate':gate(metrics),'direction_signal':signal,
            'fixed_baseline_errors':sum(not b['correct'] and x['correct'] for b,x in zip(baseline,rs)),
            'broke_baseline_correct':sum(b['correct'] and not x['correct'] for b,x in zip(baseline,rs))}
        (OUT/(name+'.jsonl')).write_text(''.join(json.dumps(x,ensure_ascii=False)+'\n' for x in rs))
    stages={}
    for dataset in ('v7','v8','companion','ip_team'):
        items=[(r,b,x) for r,b,x in zip(rows,baseline,raw) if b['dataset']==dataset]
        known=[(b,x) for r,b,x in items if 'abstain' not in b['gold']]
        stages[dataset]={'action_non_abstain':{'n':len(known),'correct':sum(x['action']['pred'] in {('respond' if g.startswith('respond:') else g) for g in b['gold']} for b,x in known)},
            'abstain_rejected':{'n':sum(b['gold']==['abstain'] for r,b,x in items),'correct':sum(x['verify']['pred']!='supported' for r,b,x in items if b['gold']==['abstain'])},
            'valid_proposal_retained':{'n':sum(x['proposal'] in b['gold'] for r,b,x in items),'correct':sum(x['verify']['pred']=='supported' for r,b,x in items if x['proposal'] in b['gold'])}}
    report['stages']=stages
    report['completed_forwards']=304+2*len(order)
    assert digest(__file__)==cfg['script_sha256'] and digest(DATA)==cfg['data_sha256']
    (OUT/'summary.json').write_text(json.dumps(report,ensure_ascii=False,indent=2))
    print(json.dumps({'completed_forwards':report['completed_forwards'],'gates':{k:report[k]['gate'] for k in variants}},ensure_ascii=False),flush=True)

if __name__=='__main__':main()

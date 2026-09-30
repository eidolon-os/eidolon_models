"""Per-task/action metrics; never hide lost coverage behind total accuracy."""
import json
import math
import time
import statistics
from collections import defaultdict
from pathlib import Path

from .data import group


def summarize(rows):
    buckets=defaultdict(list)
    for r in rows:
        buckets[r['dataset']].append(r)
    result={}
    for name,rs in buckets.items():
        families=defaultdict(list)
        for r in rs:families[r['family']].append(r)
        groups={}
        for action in ('respond','stop','abstain','mixed'):
            items=[r for r in rs if group(r['gold'])==action]
            if items:
                groups[action]={'n':len(items),'correct':sum(r['correct'] for r in items),
                    'wrong_speech':sum(r['pred'].startswith('respond:') for r in items if action=='stop'),
                    'missed_response':sum(not r['pred'].startswith('respond:') for r in items if action=='respond')}
        result[name]={'n':len(rs),'correct':sum(r['correct'] for r in rs),'groups':groups,
            'families':len(families),'complete_families_correct':sum(all(r['correct'] for r in fs) for fs in families.values()),
            'latency_ms':{'median':statistics.median(r['elapsed_ms'] for r in rs),
                          'p95':sorted(r['elapsed_ms'] for r in rs)[min(len(rs)-1,math.ceil(.95*len(rs))-1)]},
            'by_origin':{o:{'n':sum(r['origin']==o for r in rs),'correct':sum(r['correct'] for r in rs if r['origin']==o)} for o in sorted({r['origin'] for r in rs})}}
    return result


def gate(metrics):
    failures=[]
    if metrics['v7']['correct']<36:failures.append('v7_below_36_of_42')
    for dataset in ('v8','companion','ip_team'):
        if dataset not in metrics:
            failures.append(dataset+'_missing');continue
        for action in ('respond','stop','abstain'):
            if action not in metrics[dataset]['groups']:failures.append(dataset+'_'+action+'_unmeasured')
        for action,x in metrics[dataset]['groups'].items():
            if x['correct']/x['n']<.7:failures.append(dataset+'_'+action)
            if action=='stop' and x['wrong_speech']/x['n']>.1:failures.append(dataset+'_wrong_speech')
    return {'passed':not failures,'failures':failures}


def evaluate(engine, records, out):
    out=Path(out)
    rows=[]
    with out.open('x') as f:
        for i,r in enumerate(records):
            start=time.perf_counter();probs,tokens=engine.predict(r)
            assert all(math.isfinite(x) and 0<=x<=1 for x in probs.values()) and abs(sum(probs.values())-1)<.002
            pred=max(probs,key=probs.get);gold=r['labels']['move']['gold']
            row=dict(id=r['id'],dataset=r['meta']['dataset'],family=r['meta']['family'],origin=r['meta']['origin'],
                gold=gold,pred=pred,correct=pred in gold,probabilities=probs,input_tokens=tokens,
                elapsed_ms=(time.perf_counter()-start)*1000)
            rows.append(row);f.write(json.dumps(row,ensure_ascii=False)+'\n');f.flush()
            if i%20==0 or i==len(records)-1:print('evaluate',out.name,i+1,len(records),flush=True)
    metrics=summarize(rows)
    summary={'metrics':metrics,'gate':gate(metrics)}
    out.with_suffix('.summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    return summary

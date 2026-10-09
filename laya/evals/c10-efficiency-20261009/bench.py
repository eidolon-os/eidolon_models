"""Isolated fixed-weight scheduling experiment. Never invokes Agent or an actuator."""
import argparse
import itertools
import json
import statistics
import time
import types
import urllib.request
from pathlib import Path

from eidolon_models_laya.config import Settings
from eidolon_models_laya.engine import load_engine
from eidolon_models_laya.sequence import collate


def optimal(backend, lengths, costs):
    buckets = [backend._bucket(n) for n in lengths]
    choices = [[c for c, bs in backend._placement.items() if b in bs] for b in buckets]
    if len(lengths) > 8:
        raise ValueError('Experimental exhaustive scheduler is bounded to eight questions')
    best, score = None, float('inf')
    for assignment in itertools.product(*choices):
        loads = {c: 0.0 for c in backend._placement}
        for core, bucket in zip(assignment, buckets):
            loads[core] += costs[str(core)][str(bucket)]
        if max(loads.values()) < score:
            best, score = assignment, max(loads.values())
    return {c: sorted([(i, buckets[i]) for i, core in enumerate(best) if core == c], key=lambda x: x[1])
            for c in backend._placement}


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', type=Path, required=True)
    p.add_argument('--requests', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    p.add_argument('--placement', default='1:128,256,384,512|2:128,256')
    p.add_argument('--rounds', type=int, default=3)
    p.add_argument('--all', action='store_true')
    p.add_argument('--bucket160', action='store_true')
    args = p.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    def production_info():
        with urllib.request.urlopen('http://127.0.0.1:8771/v1/info', timeout=3) as r:
            return json.load(r)
    before = production_info()
    (args.out/'production-before.json').write_text(json.dumps(before))
    engine, manifest = load_engine(Settings(backend='rknn', model_dir=args.model, max_len=512,
        head_max_len=512, rknn_placement=args.placement, speculative=False))
    assert manifest.revision == 'e8254243'
    be = engine.backend
    cases = [json.loads(line) for line in args.requests.read_text().splitlines()]
    if not args.all:
        cases = [c for c in cases if c['set'] == 'conversation']
    original = be.schedule
    all_buckets = list(be._buckets)
    # Isolated per-core runtime measurements, including input and scoring CPU work in submit.
    costs = {}
    ids, qs, items, cuts = engine._prepare(cases[0]['body']['state'], cases[0]['body']['questions'], False)
    sample = items[0]
    for core, buckets in be._placement.items():
        costs[str(core)] = {}
        for bucket in buckets:
            item = {**sample, 'ids': (sample['ids'] + [engine.tokenizer.pad_token_id] * bucket)[:bucket]}
            batch = collate([item], engine.tokenizer.pad_token_id)
            be.schedule = lambda lengths, c=core, b=bucket: {c: [(0, b)]}
            times = []
            for i in range(7):
                start = time.perf_counter(); be.forward(batch)
                if i >= 2:
                    times.append((time.perf_counter()-start)*1000)
            costs[str(core)][str(bucket)] = statistics.median(times)
    be.schedule = original
    (args.out/'costs.json').write_text(json.dumps(costs, indent=2))
    print('COSTS', costs, flush=True)
    schedules = []
    for case in cases:
        _, _, items, _ = engine._prepare(case['body']['state'], case['body']['questions'], False)
        lengths = [len(item['ids']) for item in items]
        if args.bucket160:
            be._buckets = [b for b in all_buckets if b != 160]
            baseline_schedule = original(lengths)
            be._buckets = all_buckets
            candidate_schedule = original(lengths)
        else:
            baseline_schedule = original(lengths)
            candidate_schedule = optimal(be, lengths, costs)
        schedules.append({'id': case['id'], 'lengths': lengths, 'baseline': baseline_schedule,
                          'measured': candidate_schedule})
    (args.out/'schedules.json').write_text(json.dumps(schedules, indent=2))
    for case in cases[:2]:
        engine.predict(**case['body'])
    reference = {}; rows = []; differences = []
    with (args.out/'responses.jsonl').open('w') as f:
        for repeat in range(args.rounds):
            for index, case in enumerate(cases):
                for mode in (['baseline', 'measured'] if (repeat+index)%2 == 0 else ['measured', 'baseline']):
                    if args.bucket160:
                        be._buckets = [b for b in all_buckets if b != 160] if mode == 'baseline' else all_buckets
                        be.schedule = original
                    else:
                        be.schedule = original if mode == 'baseline' else types.MethodType(
                            lambda self, lengths: optimal(self, lengths, costs), be)
                    pred = engine.predict(**case['body'])
                    row = {'id': case['id'], 'set': case['set'], 'repeat': repeat, 'mode': mode,
                           'ms': pred.total_ms, 'forward_ms': pred.forward_ms,
                           'answers': pred.answers, 'truncated': pred.truncated}
                    # Compare full decoded probabilities, not just argmax, across both schedules/repeats.
                    if case['id'] in reference and pred.answers != reference[case['id']]:
                        differences.append({'id':case['id'],'repeat':repeat,'mode':mode})
                    reference.setdefault(case['id'], pred.answers)
                    rows.append(row);f.write(json.dumps(row, ensure_ascii=False)+'\n');f.flush()
                if (index+1)%10 == 0:
                    print('progress', repeat, index+1, flush=True)
    after = production_info()
    def stats(mode):
        a=sorted(r['ms'] for r in rows if r['mode']==mode)
        return {'n':len(a),'p50':statistics.median(a),'p95':a[int(.95*(len(a)-1))],
                'mean':statistics.mean(a),'max':max(a),'over_1000ms':sum(t>1000 for t in a)}
    summary = {'baseline':stats('baseline'),'measured':stats('measured'),
        'differences':differences,'truncated':sum(bool(r['truncated']) for r in rows),
        'changed_schedules':sum(s['baseline']!=s['measured'] for s in schedules),
        'experiment': 'bucket160' if args.bucket160 else 'scheduling',
        'production_served_before':before['service']['served'],
        'production_served_after':after['service']['served'],'engine':engine.describe()}
    (args.out/'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary), flush=True)

if __name__ == '__main__':
    main()

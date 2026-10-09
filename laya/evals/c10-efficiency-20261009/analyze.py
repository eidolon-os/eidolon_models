"""Compare paired latency, classifier outputs and actual model-first acceptance."""
import argparse
import importlib.util
import asyncio
import json
import statistics
from pathlib import Path

from eidolon_agent.infra.interpretation.adapters.laya import LayaInterpreter
from eidolon_agent.domain.smarthome.command import SmartHomeCommand
from eidolon_agent.domain.smarthome.tests.conftest import FakeDirectory, FakeExecutor, home_registry
from eidolon_sdk.biz.interpretation import InterpretationRequest


class Saved(LayaInterpreter):
    async def predict(self, body, *, timeout_ms):
        return self.payload


async def main(folder):
    rows = [json.loads(line) for line in (folder/'responses.jsonl').read_text().splitlines()]
    pairs = {}
    for row in rows:
        pairs.setdefault((row['id'],row['repeat']), {})[row['mode']] = row
    baseline = json.loads((Path(__file__).parent.parent/'model-first-20261008/c10.json').read_text())
    requests = {f'{group}-{i}': item['model']['request']
                for group in ('conversation','independent','context_safety')
                for i,item in enumerate(baseline[group])}
    model = Saved('http://127.0.0.1:1')
    directory = FakeDirectory(home_registry())
    command = SmartHomeCommand(directory=directory, executor=FakeExecutor(directory), interpreter=model, min_confidence=.8)
    spec = importlib.util.spec_from_file_location('previous_analysis', Path(__file__).parent.parent/'c10-npu-20261009/analyze.py')
    previous = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(previous)
    choices = []; decisions = []; legacy_differences = []; max_diff = 0; question_count = 0; decision_count = 0
    try:
        for (identity, repeat), pair in pairs.items():
            assert set(pair) == {'baseline','measured'}
            left,right = pair['baseline']['answers'],pair['measured']['answers']
            assert set(left)==set(right)
            for q in left:
                question_count += 1
                if left[q]['choice'] != right[q]['choice']:
                    choices.append({'id':identity,'repeat':repeat,'question':q,
                                    'before':left[q]['choice'],'after':right[q]['choice']})
                max_diff = max(max_diff,*(abs(v-right[q]['probabilities'][k]) for k,v in left[q]['probabilities'].items()))
            if identity not in requests and previous.legacy(left) != previous.legacy(right):
                legacy_differences.append({'id':identity,'repeat':repeat,'before':previous.legacy(left),'after':previous.legacy(right)})
            if identity in requests:
                results=[]
                for answers in (left,right):
                    model.payload={'answers':answers,'model':'laya-smart-home','revision':'e8254243',
                                   'features':{'question_state':True}}
                    result = await model.interpret(InterpretationRequest.model_validate(requests[identity]))
                    accepted=command._confident(result) and result.proposal is not None
                    results.append((accepted,result.proposal.model_dump(mode='json') if accepted else None))
                decision_count+=1
                if results[0]!=results[1]:decisions.append({'id':identity,'repeat':repeat,'results':results})
    finally:
        await model.aclose()
    stats={}
    for group in sorted({r['set'] for r in rows}):
        stats[group]={}
        for mode in ('baseline','measured'):
            a=sorted(r['ms'] for r in rows if r['set']==group and r['mode']==mode)
            stats[group][mode]={'n':len(a),'mean':statistics.mean(a),'p50':statistics.median(a),
                'p95':a[int(.95*(len(a)-1))],'max':max(a)}
    out={'pairs':len(pairs),'question_pairs':question_count,'choice_differences':choices,
         'max_probability_difference':max_diff,'model_first_pairs':decision_count,
         'model_first_decision_differences':decisions,
         'legacy_decision_differences':legacy_differences,'by_set':stats}
    (folder/'analysis.json').write_text(json.dumps(out,ensure_ascii=False,indent=2)+'\n')
    print(json.dumps(out,ensure_ascii=False,indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('folder',type=Path)
    asyncio.run(main(p.parse_args().folder))

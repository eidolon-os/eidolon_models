"""Compare saved local HTTP answers and threshold decisions with frozen offline reports."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from target_report import summarize_report


def check(run: Path):
    results = {}
    all_ms = []
    differences = []
    max_dp = 0.0
    scored = 0
    for path in sorted((run / 'service-responses').glob('*.jsonl')):
        name = path.name.removesuffix('.responses.jsonl')
        folder = 'acceptance' if name in ('target-accept', 'target-accept-single', 'controls-accept') else 'eval'
        report = json.loads((run / folder / f'{name}.json').read_text())
        responses = [json.loads(x) for x in path.read_text().splitlines()]
        lookup = {x['id']: x for x in responses}
        if len(lookup) != report['n_records'] or len(lookup) != len(responses):
            raise ValueError(f'incomplete or duplicated HTTP responses: {name}')
        actual = copy.deepcopy(report)
        for row in actual['policy_rows']:
            value = lookup[row['record_id']]['answers'][row['qid']]
            if row['gold']:
                scored += 1
            if row['pred'] != value['choice']:
                differences.append({'set': name, 'id': row['record_id'], 'qid': row['qid']})
            max_dp = max(max_dp, max(abs(p - value['probabilities'][k])
                                     for k, p in row['probabilities'].items()))
            row.update(pred=value['choice'], p_top=value['probabilities'][value['choice']],
                       probabilities=value['probabilities'],
                       correct=bool(row['gold'] and value['choice'] in row['gold']))
        before, after = summarize_report(report), summarize_report(actual)
        same = ((before['continuation'] or {}).get('counts') == (after['continuation'] or {}).get('counts')
                and before['single']['0.8']['outcomes'] == after['single']['0.8']['outcomes'])
        ms = sorted(x['ms'] for x in responses)
        all_ms += ms
        results[name] = {'n_requests': len(ms), 'same_threshold_outcomes': same,
                         'p50_ms': ms[len(ms)//2], 'p95_ms': ms[int(.95*(len(ms)-1))], 'max_ms': max(ms)}
    if not all_ms:
        raise ValueError('no saved responses')
    all_ms.sort()
    first = json.loads((run / 'service-first-direct-response.json').read_text())
    passed = not differences and all(x['same_threshold_outcomes'] for x in results.values())
    return {'status': 'passed' if passed else 'failed', 'sets': results, 'requests': len(all_ms),
            'scored_answers': scored, 'choice_differences': differences, 'max_probability_difference': max_dp,
            'all_threshold_outcomes_identical': all(x['same_threshold_outcomes'] for x in results.values()),
            'p50_ms': all_ms[len(all_ms)//2], 'p95_ms': all_ms[int(.95*(len(all_ms)-1))], 'max_ms': max(all_ms),
            'first_direct_request_server_timing_ms': first['timing_ms'],
            'note': 'Mac only; excludes failed proxy request and first direct request; model load reported separately'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    args = ap.parse_args()
    result = check(args.run)
    (args.run / 'service-summary.json').write_text(json.dumps(result, indent=2)+'\n')
    print(json.dumps(result))
    raise SystemExit(0 if result['status'] == 'passed' else 1)

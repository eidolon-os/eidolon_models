"""Recompute the registered development gates; never opens frozen acceptance data."""
import argparse
import json
from pathlib import Path

from cmetrics import summarize
from target_report import summarize_report


def check(run: Path, baseline: Path, adjudication: Path):
    def read(path):
        return json.loads(path.read_text())

    def policy(name):
        return read(run / 'policy' / f'{name}.json')

    excluded = {r['id'] for r in read(adjudication)['rows'] if r['exclude_from_adjudicated_gate']}
    if excluded != {'smart-home-continuation/b23a0ffbed78'}:
        raise ValueError('C7/C8/C9 registered adjudication excludes exactly b23a0ffbed78')
    semantic = {}
    for name, path in [('baseline', baseline / 'eval-c/c-dev.json'),
                       ('candidate', run / 'eval/c-dev.json')]:
        rows = [r for r in read(path)['policy_rows']
                if r['qid'] in ('pick', 'follow') and r['record_id'] not in excluded]
        if len(rows) != 644:
            raise ValueError('expected the registered 644-row semantic regression set')
        semantic[name] = summarize(rows, .95, .5)
    mined, target, cont, single = [policy(n) for n in
                                  ('mined-regression', 'target-dev', 'c-dev', 'locked-accept')]
    old_cont = summarize_report(read(baseline / 'eval-c/c-dev.json'))['continuation']
    old_single = summarize_report(read(baseline / 'eval-final/locked-accept.json'))['single']['0.8']
    now_single = single['single']['0.8']
    checks = {
        'mined_six_all_redo': mined['continuation']['n'] == 6 and
            mined['continuation']['counts']['正确交还'] == 6 and mined['overall']['acc'] == 1,
        'target_dev_wrong_execute_zero': target['continuation']['counts']['错误执行'] == 0,
        'target_dev_takeover_at_least_50pct': target['continuation']['takeover'] >= .5,
        'c_dev_semantic_wrong_execute_not_increased':
            semantic['candidate']['counts']['错误执行'] <= semantic['baseline']['counts']['错误执行'],
        'c_dev_takeover_drop_at_most_3pp':
            cont['continuation']['takeover'] >= old_cont['takeover'] - .03,
        'c_dev_paired_gate': cont['paired_gate']['passed'],
        'historical_single_paired_gate': single['paired_gate']['passed'],
        'historical_single_wrong_execute_not_increased':
            now_single['wrong_execute'] <= old_single['wrong_execute'],
        'historical_single_coverage_drop_at_most_3pp':
            now_single['correct_control_coverage'] >= old_single['correct_control_coverage'] - .03,
    }
    if (run / 'data/intent-dev.jsonl').exists():
        intent = policy('intent-dev')['single']['0.8']
        checks['intent_dev_wrong_execute_zero'] = intent['wrong_execute'] == 0
        checks['intent_dev_control_coverage_at_least_50pct'] = intent['correct_control_coverage'] >= .5
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'excluded_from_semantic_gate': sorted(excluded), 'c_dev_semantic': semantic,
            'c_dev_registered': cont['continuation'], 'target_dev': target['continuation'],
            'historical_single_wrong_execute': [old_single['wrong_execute'], now_single['wrong_execute']],
            'historical_single_coverage': [old_single['correct_control_coverage'],
                                           now_single['correct_control_coverage']],
            'scope': 'development/history only; not new acceptance, NPU parity or deployment approval'}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    ap.add_argument('--baseline', type=Path, required=True)
    ap.add_argument('--adjudication', type=Path, required=True)
    ap.add_argument('--out', type=Path, required=True)
    args = ap.parse_args()
    result = check(args.run, args.baseline, args.adjudication)
    args.out.write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps({'status': result['status'], 'failed': [k for k, v in result['checks'].items()
                                                          if not v]}, ensure_ascii=False))
    raise SystemExit(0 if result['status'] == 'passed' else 1)

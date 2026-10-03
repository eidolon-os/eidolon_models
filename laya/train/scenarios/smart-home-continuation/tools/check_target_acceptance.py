"""Check the pre-registered one-shot synthetic acceptance, without changing the model."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from cmetrics import summarize
from target_report import summarize_report


def check(run: Path):
    freeze = json.loads((run / 'acceptance-freeze.json').read_text())
    for name, sha in freeze['acceptance_files'].items():
        if hashlib.sha256((run / 'data' / name).read_bytes()).hexdigest() != sha:
            raise ValueError(f'acceptance data changed: {name}')
    weights = hashlib.sha256((run / 'checkpoint/model.safetensors').read_bytes()).hexdigest()
    config = hashlib.sha256((run / 'checkpoint/rl_agent_config.json').read_bytes()).hexdigest()
    frozen_files = freeze['files']
    if weights != next(v for k, v in frozen_files.items() if k.endswith('/model.safetensors')):
        raise ValueError('weights changed after freeze')
    if config != next(v for k, v in frozen_files.items() if k.endswith('/rl_agent_config.json')):
        raise ValueError('calibration changed after freeze')
    reports = {}
    for name in ('target-accept', 'target-accept-single', 'controls-accept'):
        report = json.loads((run / 'acceptance' / f'{name}.json').read_text())
        if report['checkpoint_sha256'] != weights or report['checkpoint_config_sha256'] != config:
            raise ValueError(f'report provenance differs: {name}')
        reports[name] = report
    target = reports['target-accept']['policy_rows']
    same = [r for r in target if 'same-focus' in r['tags']]
    switches = [r for r in target if r['gold'] == ['重新理解']]
    if len(same) != 44 or len(switches) != 88 or len(target) != 132:
        raise ValueError('unexpected frozen target set shape')
    same_summary = summarize(same, .95, .5)
    switch_summary = summarize(switches, .95, .5)
    control = summarize_report(reports['controls-accept'])['continuation']
    if control['n'] != 40 or reports['target-accept-single']['n_records'] != 44:
        raise ValueError('unexpected frozen controls/single set shape')
    checks = {
        'switch_wrong_execute_zero': switch_summary['counts']['错误执行'] == 0,
        'same_focus_takeover_at_least_50pct': same_summary['takeover'] >= .5,
        'controls_wrong_execute_zero': control['counts']['错误执行'] == 0,
        'controls_wrong_cancel_zero': control['counts']['错误取消'] == 0,
    }
    return {'status': 'passed' if all(checks.values()) else 'failed', 'checks': checks,
            'same_focus': same_summary, 'switches': switch_summary, 'controls': control,
            'single': {k: v for k, v in summarize_report(reports['target-accept-single'])['single']['0.8'].items()
                       if k != 'decisions'},
            'limitations': '216 self-authored synthetic records, correlated contrasts; not independent human blind labels',
            'weights_sha256': weights, 'config_sha256': config}


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--run', type=Path, required=True)
    a = ap.parse_args()
    result = check(a.run)
    (a.run / 'acceptance-gate.json').write_text(json.dumps(result, ensure_ascii=False, indent=2)+'\n')
    print(json.dumps(result, ensure_ascii=False, indent=2))
    raise SystemExit(0 if result['status'] == 'passed' else 1)

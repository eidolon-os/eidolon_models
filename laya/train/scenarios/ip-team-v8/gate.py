"""Freeze a test permit only when the preregistered development gate passes."""
import argparse
import hashlib
import json
from pathlib import Path


def passes(s, regression):
    groups=s['action_groups']
    return (set(groups)=={'respond','stop','abstain'}
            and all(g['accuracy']>=.7 for g in groups.values())
            and s['must_stop']['n']>0
            and s['must_stop']['wrong_speech']/s['must_stop']['n']<=.1
            and s['must_abstain']['n']>0
            and s['must_abstain']['correct']/s['must_abstain']['n']>=.7
            and regression['records']==42 and regression['correct']>=27
            and regression['must_stop']['n']==6 and regression['must_stop']['wrong_speech']<=4)


def digest(p):return hashlib.sha256(p.read_bytes()).hexdigest()


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--base',type=Path,required=True);p.add_argument('--regression-summary',type=Path,required=True);a=p.parse_args()
    dest=a.dataset/'gate.json'
    if dest.exists():raise ValueError('gate already frozen')
    s=json.loads((a.run/'val/summary.json').read_text())
    regression=json.loads(a.regression_summary.read_text())
    result={'passed':passes(s,regression),'dev_summary_sha256':digest(a.run/'val/summary.json'),
            'regression_summary_sha256':digest(a.regression_summary),'regression':regression,
            'run':str(a.run.resolve()),'rule':'Every action >=70%; stop wrong speech <=10%; abstain recall >=70%; old dev correct >=27/42 and wrong speech <=4/6',
            'test_inference_permitted':passes(s,regression)}
    dest.write_text(json.dumps(result,indent=2))
    if result['passed']:
        selection={'projection':'named','run':str(a.run.resolve()),'test_not_used_for_selection':True,
                   'dataset_lineage_sha256':digest(a.dataset/'lineage.json'),
                   'sealed_test_sha256':digest(a.dataset/'sealed/named/test.jsonl'),
                   'allowed_test_checkpoints':{str(x.resolve()):digest(x/'model.safetensors') for x in [a.run/'checkpoint',a.base]}}
        (a.dataset/'selection.json').write_text(json.dumps(selection,indent=2))
    print(json.dumps(result,indent=2))


if __name__=='__main__':main()

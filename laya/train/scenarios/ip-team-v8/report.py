"""Collect completed offline runs, including an explicit unopened-test state."""
import argparse
import hashlib
import json
from pathlib import Path

HERE=Path(__file__).resolve().parent
LAYA=HERE.parents[2]


def read(p):return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--base-run',type=Path,required=True);a=p.parse_args()
    result={'dataset':read(a.dataset/'lineage.json'),'production':read(HERE/'generation-ledger.json'),
            'length_shortcut':read(a.dataset/'length-shortcut.json'),'gate':read(a.dataset/'gate.json'),
            'test_status':'permitted' if read(a.dataset/'gate.json')['passed'] else 'not_opened_dev_gate_failed','runs':{}}
    for label,root in [('trained',a.run),('base',a.base_run)]:
        result['runs'][label]={'path':str(root.resolve()),'lineage':read(root/'lineage.json')}
        for split in ('train','val'):
            result['runs'][label][split]=read(root/split/'summary.json')
    result['training']=read(a.run/'checkpoint/train_summary.json')
    if result['gate']['passed']:
        for name in ('selected','base'):
            path=LAYA/f'train/runs/ip-ensemble-v8-{name}-test/test/summary.json'
            if path.exists():result[name+'_test']=read(path)
        if all(k in result for k in ('selected_test','base_test')):result['test_status']='completed_once'
    (HERE/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    paths=sorted(HERE.glob('*.py'))+[LAYA/'tests/test_ip_v8.py',LAYA/'src/eidolon_models_laya/participation_text.py',LAYA/'src/eidolon_laya_train/train.py',HERE.parent/'ip-team-v7/prepare.py',HERE.parent/'ip-team-v7/evaluate.py',HERE.parent/'ip-team-v7/generate.py',HERE.parent/'ip-team-v5/generate_glm.py',HERE.parent/'ip-team-v5/run_experiment.py']
    (HERE/'code-manifest.json').write_text(json.dumps({str(x.relative_to(LAYA)):hashlib.sha256(x.read_bytes()).hexdigest() for x in paths},indent=2))
    for label,r in result['runs'].items():
        for split in ('train','val'):
            s=r[split]
            print(label,split,json.dumps({k:s[k] for k in ['records','correct','action_groups','all_six_correct_families','must_stop','must_speak','must_abstain','unique_member','confidence']},ensure_ascii=False))


if __name__=='__main__':main()

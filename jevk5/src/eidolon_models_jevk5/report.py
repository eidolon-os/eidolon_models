"""Verify a completed bounded run and write a compact, reproducible comparison."""
import argparse
import json
import math
from pathlib import Path

from .data import digest,read


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--run',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True);a=ap.parse_args()
    selection=json.loads((a.run/'selection.json').read_text())
    config=json.loads((a.run/'config.json').read_text())
    for name,expected in config['code_sha256'].items():
        assert digest(Path(__file__).parent/name)==expected, f'run source changed: {name}'
    assert digest(a.data/'train.jsonl')==config['train_sha256']
    assert digest(a.data/'dev.jsonl')==config['dev_sha256']
    dev={r['id']:r for r in read(a.data/'dev.jsonl')};reports={};raw_files={}
    lines=['# 首轮适配开发集对照','','| 模型 | 数据组 | 正确/总数 | 回应正确 | 停止正确 | 弃权正确 | 应停误发言 | median / p95 ms |','|---|---|---:|---:|---:|---:|---:|---:|']
    for path in [a.run/'baseline.jsonl',*sorted(a.run.glob('epoch-*.dev.jsonl'))]:
        rows=read(path);assert len(rows)==len(dev) and {r['id'] for r in rows}==set(dev)
        for r in rows:
            assert r['gold']==dev[r['id']]['labels']['move']['gold']
            p=r['probabilities'];assert set(p)==set(dev[r['id']]['questions']['move']['criteria'])
            assert all(math.isfinite(x) and 0<=x<=1 for x in p.values()) and abs(sum(p.values())-1)<.002
            assert r['pred']==max(p,key=p.get) and r['correct']==(r['pred'] in r['gold'])
        summary=json.loads(path.with_suffix('.summary.json').read_text());reports[path.stem]=summary;raw_files[path.name]=digest(path)
        for dataset,m in summary['metrics'].items():
            def score(group):
                x=m['groups'].get(group);return f"{x['correct']}/{x['n']}" if x else '未测'
            stop=m['groups'].get('stop',{});lat=m['latency_ms']
            lines.append(f"| {path.stem} | {dataset} | {m['correct']}/{m['n']} | {score('respond')} | {score('stop')} | {score('abstain')} | {stop.get('wrong_speech','未测')} | {lat['median']:.0f} / {lat['p95']:.0f} |")
    result=dict(selection=selection,reports=reports,verified=True,raw_sha256=raw_files,
        config_sha256=digest(a.run/'config.json'),dataset_manifest_sha256=digest(a.data/'manifest.json'))
    a.out.mkdir(parents=True,exist_ok=True)
    (a.out/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (a.out/'TABLE.md').write_text('\n'.join(lines)+'\n')
    print('\n'.join(lines));print(json.dumps(selection))


if __name__=='__main__':main()

"""Paired prompt-policy audit, with the gates fixed in DESIGN.md."""
import importlib.util
import json
from collections import defaultdict
from pathlib import Path

HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('direction_run',HERE/'run.py');run=importlib.util.module_from_spec(s);s.loader.exec_module(run)
s=importlib.util.spec_from_file_location('base_report',HERE.parent/'backbone-comparison-v1/report.py');base=importlib.util.module_from_spec(s);s.loader.exec_module(base)


def gate(summary,baseline_v7):
    v7,v8=summary['v7'],summary['v8'];fail=[]
    if v7['correct']<baseline_v7:fail.append('v7_regression')
    for key in ['must_speak','must_stop','must_abstain']:
        x=v8[key]
        if x['correct']/x['n']<.7:fail.append(key)
    if v8['must_stop']['wrong_speech']/v8['must_stop']['n']>.1:fail.append('wrong_speech')
    return {'passed':not fail,'failures':fail}


def main():
    raw={(v,r['id']):r for v,r in run.c.records()};result={'models':{},'scope':'Existing DEV; policies fixed before scoring. No independent-generalization or release claim.'}
    changes=[]
    lines=['# 固定规则提示对照','','| 模型/提示 | v7/42 | v8/72 | 回应/24 | 停止/24 | 弃权/24 | v8六变体全对家族/12 | 方向门槛 |','|---|---:|---:|---:|---:|---:|---:|---|']
    for alias in ['decider','jevk5']:
        configs={};predictions={}
        for variant in ['original','A','B']:
            p=run.c.OUT/f'{alias}.jsonl' if variant=='original' else run.OUT/f'{alias}-{variant}.jsonl'
            if not p.exists():continue
            rows=[json.loads(l) for l in p.read_text().splitlines()]
            assert len({(r['split'],r['id']) for r in rows})==len(rows)
            if len(rows)!=114:continue
            assert set((r['split'],r['id']) for r in rows)==set(raw)
            summaries={v:base.aggregate([r for r in rows if r['split']==v],raw) for v in run.c.DATA}
            configs[variant]={'metrics':summaries,'gate':gate(summaries,30 if alias=='decider' else 36),'sha256':run.c.digest(p)}
            predictions[variant]={(r['split'],r['id']):r for r in rows}
            a,b=summaries['v7'],summaries['v8']
            lines.append(f"| {alias}/{variant} | {a['correct']} | {b['correct']} | {b['must_speak']['correct']} | {b['must_stop']['correct']} | {b['must_abstain']['correct']} | {b['all_variants_correct_families']} | {'通过' if configs[variant]['gate']['passed'] else '未过'} |")
        result['models'][alias]={'configurations':configs}
        for variant in ['A','B']:
            if variant not in predictions:continue
            paired={}
            for split in run.c.DATA:
                keys=[k for k in raw if k[0]==split]
                paired[split]={
                    'fixed':sum(not predictions['original'][k]['correct'] and predictions[variant][k]['correct'] for k in keys),
                    'regressed':sum(predictions['original'][k]['correct'] and not predictions[variant][k]['correct'] for k in keys),
                }
                for k in keys:
                    old,new=predictions['original'][k],predictions[variant][k]
                    if old['pred']!=new['pred']:
                        changes.append(dict(model=alias,variant=variant,split=split,id=k[1],
                            family=raw[k]['meta']['family'],slice=raw[k]['meta']['slice'],
                            gold=new['gold'],original=old['pred'],policy=new['pred'],
                            originally_correct=old['correct'],policy_correct=new['correct']))
            configs[variant]['paired_vs_original']=paired
        if 'A' in predictions and 'B' in predictions:
            wording={}
            for v in run.c.DATA:
                keys=sorted(k for k in raw if k[0]==v)
                fam=defaultdict(list)
                for k in keys:fam[raw[k]['meta']['family']].append(k)
                wording[v]={'n':len(keys),'same_answer':sum(predictions['A'][k]['pred']==predictions['B'][k]['pred'] for k in keys),
                            'both_correct':sum(predictions['A'][k]['correct'] and predictions['B'][k]['correct'] for k in keys),
                            'all_family_variants_both_correct':sum(all(predictions['A'][k]['correct'] and predictions['B'][k]['correct'] for k in ks) for ks in fam.values()),
                            'families':len(fam)}
            result['models'][alias].update(wording_stability=wording,both_policies_pass=all(configs[p]['gate']['passed'] for p in ['A','B']))
    (HERE/'comparison.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
    (HERE/'answer-changes.json').write_text(json.dumps(changes,ensure_ascii=False,indent=2))
    (HERE/'TABLE.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))


if __name__=='__main__':main()

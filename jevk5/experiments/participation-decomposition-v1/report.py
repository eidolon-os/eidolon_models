"""Validate completed frozen diagnostic and render its metrics, without fitting."""
import json
import math
from pathlib import Path
from eidolon_models_jevk5.data import read,digest
from eidolon_models_jevk5.evaluate import summarize,gate

ROOT=Path(__file__).resolve().parents[3]
HERE=Path(__file__).resolve().parent
RUN=ROOT/'jevk5/runs/participation-decomposition-v1'

def main():
    cfg=json.loads((RUN/'config.json').read_text())
    summary=json.loads((RUN/'summary.json').read_text())
    raw=read(RUN/'raw.jsonl')
    data=read(ROOT/'evals/participation/data/companion-ip-v1/dev.jsonl')
    base=read(ROOT/'jevk5/runs/companion-ip-v1-seed71/baseline.jsonl')
    assert digest(HERE/'run.py')==cfg['script_sha256']
    assert digest(HERE/'DESIGN.md')==cfg['design_sha256']
    assert digest(ROOT/'evals/participation/data/companion-ip-v1/dev.jsonl')==cfg['data_sha256']
    assert digest(ROOT/'jevk5/runs/companion-ip-v1-seed71/baseline.jsonl')==cfg['baseline_sha256']
    assert len(raw)==len(data)==152 and len({r['id'] for r in raw})==152
    count=0
    for r,d,b in zip(raw,data,base):
        assert r['id']==d['id']==b['id']
        assert r['proposal'] in b['probabilities'] and r['proposal'].startswith('respond:')
        for x in [r]+([r['reversed']] if 'reversed' in r else []):
            for stage in ('action','verify'):
                p=x[stage]['probabilities']; count+=1
                assert abs(sum(p.values())-1)<.002 and all(math.isfinite(v) and 0<=v<=1 for v in p.values())
                assert max(p,key=p.get)==x[stage]['pred']
    assert count==cfg['max_new_forwards']==summary['completed_forwards']==328
    for name in ('repair','decomposed'):
        rs=read(RUN/(name+'.jsonl')); assert len(rs)==152
        for x,b,r in zip(rs,base,raw):
            assert x['id']==b['id'] and x['gold']==b['gold'] and x['pred']==r[name]
            assert x['correct']==(x['pred'] in x['gold'])
        assert summarize(rs)==summary[name]['metrics']
        assert gate(summary[name]['metrics'])==summary[name]['gate']
    validation={'passed':True,'new_forwards':count,'external_calls':0,'trained':False,'sealed_test_used':False,'deployed':False,
        'immutable_inputs_verified':True,'public_input_poison_check':'passed before execution',
        'raw_sha256':digest(RUN/'raw.jsonl'),'config':cfg}
    (HERE/'VALIDATION.json').write_text(json.dumps(validation,ensure_ascii=False,indent=2))
    (HERE/'comparison.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    lines=['# 决策拆分诊断结果表','','152条冻结DEV，76个快照；原模型，无适配与训练。', '',
        '| 方案 | v7 /42 | v8 /72 | v8回应 /24 | v8停止 /24 | v8弃权 /24 | 陪伴 /22 | 新IP /16 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|']
    for name,label in [('baseline','原始'),('repair','原始+核验'),('decomposed','动作+核验')]:
        m=summary[name]['metrics'];g=m['v8']['groups']
        vals=[m['v7']['correct'],m['v8']['correct'],*[g[k]['correct'] for k in ('respond','stop','abstain')],m['companion']['correct'],m['ip_team']['correct']]
        lines.append('| '+label+' | '+' | '.join(map(str,vals))+' |')
    lines+=['','## 细分与保守拒绝代价','','| 方案 | 数据组 | 回应正确 | 停止正确 | 应停误开口 | 弃权正确 | 全家族正确 |', '|---|---|---:|---:|---:|---:|---:|']
    for name in ('baseline','repair','decomposed'):
        for ds,m in summary[name]['metrics'].items():
            g=m['groups'];v=[]
            for k in ('respond','stop'):v.append(f"{g[k]['correct']}/{g[k]['n']}")
            v.append(f"{g['stop']['wrong_speech']}/{g['stop']['n']}")
            v.append(f"{g['abstain']['correct']}/{g['abstain']['n']}")
            v.append(f"{m['complete_families_correct']}/{m['families']}")
            lines.append('| '+' | '.join([name,ds]+v)+' |')
    lines+=['','## 阶段诊断','','| 数据组 | 非弃权动作正确 | 应弃权时核验拒绝 | 正确人选核验保留 |','|---|---:|---:|---:|']
    for ds,m in summary['stages'].items():
        lines.append('| '+' | '.join([ds]+[f"{v['correct']}/{v['n']}" for v in m.values()])+' |')
    lines+=['','## 修复、回归与换序','']
    for name in ('repair','decomposed'):
        m=summary[name]
        lines.append(f"- {name}：修复{m['fixed_baseline_errors']}条，破坏原正确{m['broke_baseline_correct']}条；预登记方向信号={m['direction_signal']}；候选门槛={m['gate']['passed']}。")
    for stage in ('action','verify','repair','decomposed'):
        lines.append(f"- 换序{stage}：{sum(x[stage+'_same'] for x in summary['order_probe'])}/12一致。小样本诊断，不代表总体稳定率。")
    lines+=['','## 延迟','','总时间含历史提案前向+本轮实测阶段，仅为串行估算，非同期基准；不得把缓存提案当免费。','', '| 方案 | 数据组 | 估算中位ms | 估算p95ms |','|---|---|---:|---:|']
    for name in ('baseline','repair','decomposed'):
        for ds,m in summary[name]['metrics'].items():
            t=m['latency_ms'];lines.append(f"| {name} | {ds} | {t['median']:.0f} | {t['p95']:.0f} |")
    lines+=['','新阶段实测中位ms：'+json.dumps(summary['stage_latency_ms'],ensure_ascii=False),'',
        '同一快照两变体不独立；开发集已多次用于选型。无封存测试，无真实用户泛化结论。身份核验只拒绝、不能改选第二人选。','']
    (HERE/'TABLE.md').write_text('\n'.join(lines))
    print('\n'.join(lines[:11]))

if __name__=='__main__':main()

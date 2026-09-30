"""Two-epoch bounded last-attention-block LoRA; frozen-base and DEV controls."""
import argparse
import json
import random
import time
from collections import Counter,defaultdict
from pathlib import Path

import torch

from .data import digest,read,validate
from .engine import Engine,acceptable_set_loss
from .evaluate import evaluate


def main():
    ap=argparse.ArgumentParser();ap.add_argument('--model',type=Path,required=True)
    ap.add_argument('--data',type=Path,required=True);ap.add_argument('--out',type=Path,required=True)
    ap.add_argument('--seed',type=int,default=71);ap.add_argument('--smoke',action='store_true');a=ap.parse_args()
    if a.out.exists():raise ValueError('immutable run output already exists')
    a.out.mkdir(parents=True)
    torch.manual_seed(a.seed);rng=random.Random(a.seed)
    train=[validate(r) for r in read(a.data/'train.jsonl')]
    dev=[] if a.smoke else [validate(r) for r in read(a.data/'dev.jsonl')]
    if {r['meta']['family'] for r in train}&{r['meta']['family'] for r in dev}:raise ValueError('family leakage')
    cfg=dict(seed=a.seed,epochs=2,rank=8,alpha=16,lr=1e-4,accumulation=4,
        targets='last full-attention q/v/o projections',train_sha256=digest(a.data/'train.jsonl'),
        dev_sha256=None if a.smoke else digest(a.data/'dev.jsonl'),script_sha256=digest(__file__),
        temperature=1.22,loss='negative log probability of acceptable action set')
    cfg['code_sha256']={p.name:digest(p) for p in Path(__file__).parent.glob('*.py')}
    cfg['dataset_manifest_sha256']=digest(a.data/'manifest.json') if (a.data/'manifest.json').exists() else None
    (a.out/'config.json').write_text(json.dumps(cfg,indent=2))
    engine=Engine(a.model)
    ids,names=engine.encode(train[0])
    with torch.no_grad():before=engine.logits(ids,len(names)).clone()
    engine.add_adapter()
    with torch.no_grad():after=engine.logits(ids,len(names)).clone()
    parity=float((before-after).abs().max());assert parity<1e-5
    params=[p for p in engine.runtime.model.parameters() if p.requires_grad]
    opt=torch.optim.AdamW(params,lr=cfg['lr'],weight_decay=.01)
    print('adapter',sum(p.numel() for p in params),'zero_parity',parity,flush=True)
    if a.smoke:
        start=time.perf_counter()
        indices=[names.index(x) for x in train[0]['labels']['move']['gold']]
        loss=acceptable_set_loss(engine.logits(ids,len(names)),indices);loss.backward()
        assert all(p.grad is not None and torch.isfinite(p.grad).all() for p in params)
        grad=float(torch.nn.utils.clip_grad_norm_(params,1.0));opt.step();opt.zero_grad(set_to_none=True)
        with torch.no_grad():updated=engine.logits(ids,len(names));new_loss=float(acceptable_set_loss(updated,indices))
        assert not torch.equal(updated,before)
        engine.save_adapter(a.out/'adapter',dict(smoke_only=True))
        summary=dict(before_loss=float(loss.detach()),after_loss=new_loss,grad_norm=grad,
            seconds=time.perf_counter()-start,zero_adapter_max_delta=parity,
            parameters=sum(p.numel() for p in params),mps_allocated_bytes=torch.mps.driver_allocated_memory())
        (a.out/'smoke.json').write_text(json.dumps(summary,indent=2));print(json.dumps(summary),flush=True);return
    baseline=evaluate(engine,dev,a.out/'baseline.jsonl')
    (a.out/'resources.json').write_text(json.dumps(dict(device='mps',dtype='float16',
        adapter_dtype='float32',trainable_parameters=sum(p.numel() for p in params),
        torch=torch.__version__,mps_allocated_after_baseline=torch.mps.driver_allocated_memory()),indent=2))
    snapshots=defaultdict(list)
    for r in train:snapshots[r['meta']['snapshot']].append(r)
    families=Counter(r['meta']['family'] for rs in snapshots.values() for r in rs[:1])
    task_families=defaultdict(set)
    for r in train:task_families[r['meta']['task']].add(r['meta']['family'])
    reports=[]
    for epoch in range(1,3):
        selected=[rs[(epoch-1)%len(rs)] for rs in snapshots.values()]
        rng.shuffle(selected)
        weights=[1/(len(task_families[r['meta']['task']])*families[r['meta']['family']]) for r in selected]
        mean=sum(weights)/len(weights);weights=[w/mean for w in weights]
        with (a.out/f'epoch-{epoch}.train.jsonl').open('x') as f:
            for start in range(0,len(selected),4):
                batch=selected[start:start+4]
                opt.zero_grad(set_to_none=True)
                for j,r in enumerate(batch):
                    ids,names=engine.encode(r)
                    logits=engine.logits(ids,len(names));gold=[names.index(x) for x in r['labels']['move']['gold']]
                    loss=acceptable_set_loss(logits,gold)
                    if not torch.isfinite(loss):raise RuntimeError('nonfinite training loss')
                    (loss*weights[start+j]/len(batch)).backward()
                    f.write(json.dumps(dict(id=r['id'],loss=float(loss.detach()),weight=weights[start+j]))+'\n')
                if not all(p.grad is not None and torch.isfinite(p.grad).all() for p in params):raise RuntimeError('nonfinite gradients')
                grad=float(torch.nn.utils.clip_grad_norm_(params,1.0));opt.step();f.flush()
                print('train',epoch,min(start+4,len(selected)),len(selected),'grad',grad,flush=True)
        checkpoint=a.out/f'epoch-{epoch}'
        engine.save_adapter(checkpoint,dict(base=str(a.model.resolve()),seed=a.seed,epoch=epoch))
        report=evaluate(engine,dev,a.out/f'epoch-{epoch}.dev.jsonl');reports.append(report)
        if report['metrics']['v7']['correct']<30:
            print('stop: severe normal-dialogue regression',flush=True);break
    candidates=[(i+1,r) for i,r in enumerate(reports) if r['gate']['passed']]
    # No candidate is better than a failed, silently promoted one.
    chosen=max(candidates,key=lambda x:sum(v['correct']/v['n'] for v in x[1]['metrics'].values()))[0] if candidates else None
    (a.out/'selection.json').write_text(json.dumps(dict(selected_epoch=chosen,release_approved=False,
        reason='DEV gate; independent validation and second seed still required' if chosen else 'No epoch passed all DEV gates'),indent=2))
    print('complete',chosen,flush=True)


if __name__=='__main__':main()

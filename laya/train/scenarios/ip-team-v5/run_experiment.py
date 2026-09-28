"""Serial bounded MPS train/eval with family-level and real SDK replay metrics."""
from __future__ import annotations
import argparse
import hashlib
import json
import statistics
import time
from collections import defaultdict
from pathlib import Path
import numpy as np
import torch
from eidolon_sdk.biz.participation import DecisionRequest, validate_proposal
from eidolon_laya_train.train import train
from eidolon_laya_train.model import load_checkpoint
from eidolon_laya_train.records import read_jsonl
from eidolon_laya_train.evaluate import score_records
from eidolon_models_laya.engine import DecisionEngine
from eidolon_models_laya.sequence import Tokenizer, ModelConfig, INPUT_NAMES
from eidolon_models_laya.participation import ParticipationAdapter, LayaMovePredictor, ContextTooLong
from eidolon_models_laya.participation_text import CompactTextPredictor, project_request, check_capacity


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def summarize(rows, records):
    families=defaultdict(list); slices=defaultdict(list)
    for r in rows:
        meta=records[r['record_id']].meta
        families[meta['family']].append(r['correct'])
        slices[meta['slice']].append(r)
    silent=[r for r in rows if all(not g.startswith('respond:') for g in r['gold'])]
    speaking=[r for r in rows if all(g.startswith('respond:') for g in r['gold'])]
    confident=[r for r in rows if r['p_top']>=.8]
    rate=lambda x:sum(x)/len(x) if x else None
    return {'records':len(rows),'families':len(families),'acceptable_accuracy':rate([r['correct'] for r in rows]),
            'family_macro_accuracy':rate([rate(v) for v in families.values()]),
            'all_variants_correct_families':sum(all(v) for v in families.values()),
            'silent_wrong_speech':{'errors':sum(r['pred'].startswith('respond:') for r in silent),'n':len(silent)},
            'speaking_missed':{'errors':sum(not r['pred'].startswith('respond:') for r in speaking),'n':len(speaking)},
            'uncalibrated_threshold_0.8':{'covered':len(confident),'accuracy':rate([r['correct'] for r in confident])},
            'unique_gold':{'n':sum(len(r['gold'])==1 for r in rows),'accuracy':rate([r['correct'] for r in rows if len(r['gold'])==1])},
            'multiple_gold':{'n':sum(len(r['gold'])>1 for r in rows),'accuracy':rate([r['correct'] for r in rows if len(r['gold'])>1])},
            'constant_choice_accuracy':{choice:rate([choice in r['gold'] for r in rows]) for choice in ('respond:M0','wait','finish','abstain')},
            'slices':{k:{'n':len(v),'correct':sum(r['correct'] for r in v)} for k,v in slices.items()}}


class LoadedBackend:
    name='torch-mps-existing-model'
    def __init__(self, loaded): self.loaded=loaded
    def forward(self,batch):
        tensors=[torch.from_numpy(batch[k]).to(self.loaded.device) for k in INPUT_NAMES]
        with torch.no_grad():
            logits, act=self.loaded.model(*tensors)
        torch.mps.synchronize()
        return logits.float().cpu().numpy(),act.float().cpu().numpy()


def evaluate(checkpoint, dataset, out):
    loaded=load_checkpoint(checkpoint,'mps'); loaded.model.eval()
    # All models compared on identical v5 context budget; explicit, never inherited 1024.
    loaded.cfg.update(max_len=2048,head_max_len=256,temperature=[1.,1.,1.],temperature_by_options={})
    out.mkdir(parents=True,exist_ok=False)
    summary={'checkpoint':str(checkpoint),'checkpoint_sha256':file_hash(checkpoint/'model.safetensors'),
             'temperature':1.0,'evaluation_role':'consumed synthetic development; not independent acceptance'}
    val=[]
    for split in ('train','val'):
        records=list(read_jsonl(dataset/f'{split}.jsonl'))
        by_id={r.id:r for r in records}
        rows=score_records(loaded,records,batch_size=4)
        summary[split]=summarize(rows,by_id)
        (out/f'{split}-predictions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
        if split=='val': val=records
    engine=DecisionEngine(LoadedBackend(loaded),Tokenizer(checkpoint/'tokenizer'),ModelConfig.from_dict(loaded.cfg))
    adapter=ParticipationAdapter(CompactTextPredictor(engine),model_version=summary['checkpoint_sha256'],policy_version='v5-offline-uncalibrated',min_confidence=.8)
    replay=[]; times=[]
    for r in val:
        req=DecisionRequest.model_validate(r.meta['request'])
        start=time.perf_counter(); result=adapter.decide(req); times.append((time.perf_counter()-start)*1000)
        validate_proposal(req,result)
        replay.append({'record_id':r.id,'result':result.model_dump(mode='json')})
    summary['sdk_replay']={'requests':len(replay),'valid_results':len(replay),'decided':sum(r['result']['status']=='decided' for r in replay),
                           'median_ms_including_projection':statistics.median(times),'p95_ms':float(np.percentile(times,95))}
    # Deterministic over-capacity check using repeated generated public text, not a quality sample.
    req=DecisionRequest.model_validate(val[0].meta['request'])
    long_user=req.user_request.model_copy(update={'text':req.user_request.text*200})
    long_req=req.model_copy(update={'user_request':long_user,'trigger':long_user,'context':req.context.model_copy(update={'recent_messages':(long_user,)})})
    result=adapter.decide(long_req); validate_proposal(long_req,result)
    summary['sdk_replay']['over_capacity_abstained']=result.status=='abstained'
    (out/'sdk-replay.json').write_text(json.dumps(replay,ensure_ascii=False,indent=2))
    (out/'summary.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2))
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--init',type=Path,required=True);p.add_argument('--eval-only',action='store_true');a=p.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no CPU fallback')
    if a.run.exists():raise ValueError('immutable run already exists')
    a.run.mkdir(parents=True)
    if a.eval_only:
        evaluate(a.init,a.dataset,a.run/'eval');return
    cfg={'init':str(a.init.resolve()),'model_name':a.run.name,'epochs':6,'batch_size':4,'grad_accum':2,
         'lr_encoder':5e-5,'lr_head':2e-4,'unfreeze_layers':8,'brier_weight':.5,'proper_weight':0.,
         'warmup':.04,'weight_decay':.01,'max_len':2048,'head_max_len':256,'seed':71,'device':'mps','checkpoint_metric':'accuracy'}
    (a.run/'config.json').write_text(json.dumps(cfg,indent=2))
    (a.run/'lineage.json').write_text(json.dumps({'dataset':str(a.dataset.resolve()),'dataset_lineage_sha256':hashlib.sha256((a.dataset/'lineage.json').read_bytes()).hexdigest(),
                                               'init_sha256':file_hash(a.init/'model.safetensors')},indent=2))
    train(cfg,a.dataset,a.run/'checkpoint',log=lambda s:print(s,flush=True))
    torch.mps.empty_cache()
    evaluate(a.run/'checkpoint',a.dataset,a.run/'eval')

if __name__=='__main__':main()

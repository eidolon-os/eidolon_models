"""Fixed-recipe distribution vs acceptable-set comparison, serial MPS only."""
import argparse
import importlib.util
import json
from pathlib import Path
import torch
from eidolon_laya_train.train import train
from eidolon_laya_train.records import read_jsonl
HERE=Path(__file__).resolve().parent;TRAIN=HERE.parents[1]
s=importlib.util.spec_from_file_location('v5_runner',HERE.parent/'ip-team-v5/run_experiment.py')
v5=importlib.util.module_from_spec(s);s.loader.exec_module(v5)


def pair_metrics(dataset,eval_dir):
    records={r.id:r for r in read_jsonl(dataset/'val.jsonl')}
    predictions=json.loads((eval_dir/'val-predictions.json').read_text())
    pairs={}
    for p in predictions:
        meta=records[p['record_id']].meta
        if 'pair_id' in meta:pairs.setdefault((meta['pair_id'],meta['variant']),{})[meta['branch']]=p
    if any(set(v)!=set('xy') for v in pairs.values()):raise ValueError('incomplete paired evaluation')
    outcomes=[{'family':family,'variant':variant,'both_correct':v['x']['correct'] and v['y']['correct'],'prediction_changed':v['x']['pred']!=v['y']['pred']} for (family,variant),v in pairs.items()]
    families={f:all(r['both_correct'] for r in outcomes if r['family']==f) for f,_ in pairs}
    result={'paired_variants':len(outcomes),'both_correct':sum(r['both_correct'] for r in outcomes),
            'prediction_changed':sum(r['prediction_changed'] for r in outcomes),'families':len(families),
            'all_branches_all_variants_correct_families':sum(families.values()),'detail':outcomes}
    (eval_dir/'paired-summary.json').write_text(json.dumps(result,indent=2))
    print(json.dumps({k:v for k,v in result.items() if k!='detail'}),flush=True)


def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True)
    p.add_argument('--init',type=Path,required=True);p.add_argument('--loss',choices=['distribution','acceptable_set'],default='distribution');p.add_argument('--eval-only',action='store_true');a=p.parse_args()
    if not torch.backends.mps.is_available():raise RuntimeError('MPS unavailable; no CPU fallback')
    if a.run.exists():raise ValueError('immutable run exists')
    a.run.mkdir(parents=True)
    if a.eval_only:checkpoint=a.init
    else:
        cfg={'init':str(a.init.resolve()),'model_name':a.run.name,'epochs':8,'batch_size':4,'grad_accum':2,'lr_encoder':5e-5,'lr_head':2e-4,
             'unfreeze_layers':8,'brier_weight':.5,'proper_weight':0.,'warmup':.04,'weight_decay':.01,'max_len':2048,'head_max_len':256,
             'seed':71,'device':'mps','checkpoint_metric':'accuracy','loss_mode':a.loss}
        (a.run/'config.json').write_text(json.dumps(cfg,indent=2))
        (a.run/'lineage.json').write_text(json.dumps({'dataset':str(a.dataset.resolve()),'dataset_lineage_sha256':v5.file_hash(a.dataset/'lineage.json'),
                                                     'init_sha256':v5.file_hash(a.init/'model.safetensors'),'train_source_sha256':v5.file_hash(HERE.parents[2]/'src/eidolon_laya_train/train.py')},indent=2))
        checkpoint=a.run/'checkpoint'
        train(cfg,a.dataset,checkpoint,log=lambda x:print(x,flush=True))
        torch.mps.empty_cache()
    v5.evaluate(checkpoint,a.dataset,a.run/'eval')
    pair_metrics(a.dataset,a.run/'eval')
    # Previously consumed v5 data remains diagnostic, never checkpoint selection.
    v5.evaluate(checkpoint,TRAIN/'data/generated/ip-team-v5/pilot1',a.run/'old-regression')
if __name__=='__main__':main()

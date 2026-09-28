"""One bounded MPS experiment; sealed test requires a passing dev gate."""
import argparse
import importlib.util
import json
import statistics
import time
from pathlib import Path
import torch
from eidolon_sdk.biz.participation import DecisionRequest,validate_proposal
from eidolon_laya_train.train import train
from eidolon_laya_train.records import read_jsonl
from eidolon_laya_train.model import load_checkpoint
from eidolon_laya_train.evaluate import score_records
from eidolon_models_laya.engine import DecisionEngine
from eidolon_models_laya.participation import ParticipationAdapter
from eidolon_models_laya.participation_text import CompactTextPredictor
from eidolon_models_laya.sequence import Tokenizer,ModelConfig
HERE=Path(__file__).resolve().parent
s=importlib.util.spec_from_file_location('v8_metrics',HERE/'evaluate.py');m=importlib.util.module_from_spec(s);s.loader.exec_module(m)
s=importlib.util.spec_from_file_location('v5_backend',HERE.parent/'ip-team-v5/run_experiment.py');v5=importlib.util.module_from_spec(s);s.loader.exec_module(v5)

def evaluate(loaded,checkpoint,path,out,named):
 records=list(read_jsonl(path));rows=score_records(loaded,records,batch_size=4);result=m.metrics(records,rows)
 out.mkdir(parents=True,exist_ok=False)
 (out/'predictions.json').write_text(json.dumps(rows,ensure_ascii=False,indent=2))
 engine=DecisionEngine(v5.LoadedBackend(loaded),Tokenizer(checkpoint/'tokenizer'),ModelConfig.from_dict(loaded.cfg))
 adapter=ParticipationAdapter(CompactTextPredictor(engine,include_names=named),model_version=v5.file_hash(checkpoint/'model.safetensors'),policy_version='v8-offline-uncalibrated',min_confidence=.8)
 replay=[];times=[]
 for r in records:
  req=DecisionRequest.model_validate(r.meta['request']);start=time.perf_counter();proposal=adapter.decide(req);times.append((time.perf_counter()-start)*1000);validate_proposal(req,proposal)
  replay.append({'id':r.id,'result':proposal.model_dump(mode='json')})
 result['sdk_replay']={'valid_results':len(replay),'decided':sum(r['result']['status']=='decided' for r in replay),'median_ms':statistics.median(times)}
 (out/'sdk-replay.json').write_text(json.dumps(replay,ensure_ascii=False,indent=2));(out/'summary.json').write_text(json.dumps(result,ensure_ascii=False,indent=2))
 print(json.dumps({'dataset':str(path),'metrics':result},ensure_ascii=False),flush=True)

def main():
 p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--run',type=Path,required=True);p.add_argument('--init',type=Path,required=True);p.add_argument('--projection',choices=['compact','named'],required=True);p.add_argument('--eval-only',action='store_true');p.add_argument('--test',action='store_true');a=p.parse_args()
 if not torch.backends.mps.is_available():raise RuntimeError('MPS required; no CPU fallback')
 if a.run.exists():raise ValueError('immutable run exists')
 if a.test:
  selection=json.loads((a.dataset/'selection.json').read_text())
  if selection['projection']!=a.projection or str(a.init.resolve()) not in selection['allowed_test_checkpoints']:raise ValueError('test only frozen selected model and matching base')
  if v5.file_hash(a.init/'model.safetensors')!=selection['allowed_test_checkpoints'][str(a.init.resolve())]:raise ValueError('selected weights changed')
  if v5.file_hash(a.dataset/'lineage.json')!=selection['dataset_lineage_sha256'] or v5.file_hash(a.dataset/'sealed'/a.projection/'test.jsonl')!=selection['sealed_test_sha256']:raise ValueError('sealed data changed')
 a.run.mkdir(parents=True)
 if a.eval_only or a.test:checkpoint=a.init
 else:
  cfg={'init':str(a.init.resolve()),'model_name':a.run.name,'epochs':4,'batch_size':4,'grad_accum':2,'lr_encoder':3e-5,'lr_head':1e-4,'unfreeze_layers':8,'brier_weight':.5,'proper_weight':0.,'warmup':.04,'weight_decay':.01,'max_len':2048,'head_max_len':256,'seed':71,'device':'mps','checkpoint_metric':'accuracy','loss_mode':'distribution'}
  (a.run/'config.json').write_text(json.dumps(cfg,indent=2));checkpoint=a.run/'checkpoint'
  train(cfg,a.dataset/a.projection,checkpoint,log=lambda x:print(x,flush=True));torch.mps.empty_cache()
 (a.run/'lineage.json').write_text(json.dumps({'dataset_sha256':v5.file_hash(a.dataset/'lineage.json'),'projection':a.projection,'init_sha256':v5.file_hash(a.init/'model.safetensors'),'checkpoint_sha256':v5.file_hash(checkpoint/'model.safetensors'),'train_source_sha256':v5.file_hash(HERE.parents[2]/'src/eidolon_laya_train/train.py')},indent=2))
 loaded=load_checkpoint(checkpoint,'mps');loaded.model.eval();loaded.cfg.update(max_len=2048,head_max_len=256,temperature=[1.,1.,1.],temperature_by_options={})
 if a.test:evaluate(loaded,checkpoint,a.dataset/'sealed'/a.projection/'test.jsonl',a.run/'test',a.projection=='named')
 else:
  for split in ('train','val'):evaluate(loaded,checkpoint,a.dataset/a.projection/f'{split}.jsonl',a.run/split,a.projection=='named')
if __name__=='__main__':main()
